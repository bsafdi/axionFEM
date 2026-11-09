// General sparse LU for the fluids module: see cpp/include/axionfem/fluids/sparse_lu.hpp.
#include "axionfem/fluids/sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>

#include <Eigen/SparseLU>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/omp_shim.hpp"
#include "axionfem/sym_solver.hpp"

#ifdef AXF_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#ifdef AXF_STRUMPACK_MPI
#include <StrumpackSparseSolverMPIDist.hpp>
#endif
#endif

namespace axf::fluids {

namespace {
int rank_of(MPI_Comm c) {
  int r = 0;
  MPI_Comm_rank(c, &r);
  return r;
}
int size_of(MPI_Comm c) {
  int s = 1;
  MPI_Comm_size(c, &s);
  return s;
}
double wtime() { return MPI_Wtime(); }
}  // namespace

// ------------------------------------------------------------------------------------------------ CsrMatrix
void CsrMatrix::mult(const double* x, double* y) const {
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) s += val[k] * x[col[k]];
    y[i] = s;
  }
}

CsrMatrix CsrMatrix::from_eigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A0) {
  AXF_REQUIRE(A0.rows() == A0.cols(), "CsrMatrix: the matrix must be square");
  Eigen::SparseMatrix<double, Eigen::RowMajor> A = A0;
  A.makeCompressed();
  CsrMatrix c;
  c.n = static_cast<int>(A.rows());
  c.rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + c.n + 1);
  c.col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
  c.val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
  return c;
}

CsrMatrix CsrMatrix::from_eigen(const Eigen::SparseMatrix<double, Eigen::ColMajor>& A) {
  // the storage-order conversion transposes the arrays, not the matrix: this is the step SymSolver skips
  return from_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor>(A));
}

Eigen::SparseMatrix<double, Eigen::ColMajor> CsrMatrix::to_eigen() const {
  Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> M(n, n, nnz(), rowptr.data(), col.data(),
                                                                    val.data());
  Eigen::SparseMatrix<double, Eigen::ColMajor> A(M);
  A.makeCompressed();
  return A;
}

// ------------------------------------------------------------------------------------------------ DistCsr
std::vector<int> DistCsr::row_dist() const {
  const int P = size_of(comm);
  std::vector<int> counts(P), dist(P + 1, 0);
  int lr = local_rows;
  MPI_Allgather(&lr, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int p = 0; p < P; ++p) dist[p + 1] = dist[p] + counts[p];
  AXF_REQUIRE(dist[P] == n_global, "DistCsr: local row counts do not add up to n_global");
  return dist;
}

void DistCsr::mult(const double* x_local, double* y_local) const {
  const auto dist = row_dist();
  const int P = size_of(comm);
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> x(n_global);
  MPI_Allgatherv(x_local, local_rows, MPI_DOUBLE, x.data(), counts.data(), dist.data(), MPI_DOUBLE, comm);
  for (int i = 0; i < local_rows; ++i) {
    double s = 0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) s += val[k] * x[col[k]];
    y_local[i] = s;
  }
}

CsrMatrix DistCsr::gather(int root) const {
  const int P = size_of(comm), me = rank_of(comm);
  const auto dist = row_dist();
  int nnz_l = rowptr.empty() ? 0 : rowptr[local_rows];
  std::vector<int> nnzc(P), nnzd(P + 1, 0), rc(P);
  MPI_Gather(&nnz_l, 1, MPI_INT, nnzc.data(), 1, MPI_INT, root, comm);
  for (int p = 0; p < P; ++p) {
    nnzd[p + 1] = nnzd[p] + nnzc[p];
    rc[p] = dist[p + 1] - dist[p];
  }
  CsrMatrix A;
  if (me == root) {
    A.n = static_cast<int>(n_global);
    A.rowptr.assign(A.n + 1, 0);
    A.col.resize(nnzd[P]);
    A.val.resize(nnzd[P]);
  }
  // row lengths, then shift into a global row pointer
  std::vector<int> len(local_rows);
  for (int i = 0; i < local_rows; ++i) len[i] = rowptr[i + 1] - rowptr[i];
  MPI_Gatherv(len.data(), local_rows, MPI_INT, me == root ? A.rowptr.data() + 1 : nullptr, rc.data(),
              dist.data(), MPI_INT, root, comm);
  MPI_Gatherv(col.data(), nnz_l, MPI_INT, A.col.data(), nnzc.data(), nnzd.data(), MPI_INT, root, comm);
  MPI_Gatherv(val.data(), nnz_l, MPI_DOUBLE, A.val.data(), nnzc.data(), nnzd.data(), MPI_DOUBLE, root, comm);
  if (me == root)
    for (int i = 0; i < A.n; ++i) A.rowptr[i + 1] += A.rowptr[i];
  return A;
}

void require_single_rank(const mfem::Mesh& mesh, const char* what) {
  const auto* pm = dynamic_cast<const mfem::ParMesh*>(&mesh);
  if (pm == nullptr) return;
  AXF_REQUIRE(size_of(pm->GetComm()) == 1,
              axf::strcat(what, " needs one MPI rank: it is a one-rank operator and would solve one independent "
                                "problem per mesh partition on ", size_of(pm->GetComm()), " ranks"),
              "mpi", "Run on one rank (mpirun -np 1).");
}

DistCsr DistCsr::from_serial(const CsrMatrix& A, MPI_Comm comm) {
  AXF_REQUIRE(size_of(comm) == 1, "DistCsr::from_serial needs a one-rank communicator");
  DistCsr d;
  d.comm = comm;
  d.n_global = A.n;
  d.local_rows = A.n;
  d.rowptr = A.rowptr;
  d.col = A.col;
  d.val = A.val;
  return d;
}

DistCsr DistCsr::distribute(const CsrMatrix& A, const std::vector<int>& dist, MPI_Comm comm) {
  const int P = size_of(comm), me = rank_of(comm);
  AXF_REQUIRE(static_cast<int>(dist.size()) == P + 1 && dist[0] == 0 && dist[P] == A.n,
              "DistCsr::distribute: dist must have nranks+1 entries from 0 to n");
  DistCsr d;
  d.comm = comm;
  d.n_global = A.n;
  d.row_begin = dist[me];
  d.local_rows = dist[me + 1] - dist[me];
  const int k0 = A.rowptr[dist[me]], k1 = A.rowptr[dist[me + 1]];
  d.rowptr.resize(d.local_rows + 1);
  for (int i = 0; i <= d.local_rows; ++i) d.rowptr[i] = A.rowptr[dist[me] + i] - k0;
  d.col.assign(A.col.begin() + k0, A.col.begin() + k1);
  d.val.assign(A.val.begin() + k0, A.val.begin() + k1);
  return d;
}

void DistCsr::sum_duplicates() {
  std::vector<std::pair<int, double>> row;
  std::vector<int> rp(1, 0), cc;
  std::vector<double> vv;
  cc.reserve(col.size());
  vv.reserve(val.size());
  for (int i = 0; i < local_rows; ++i) {
    row.clear();
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) row.emplace_back(col[k], val[k]);
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const int start = static_cast<int>(cc.size());
    for (const auto& e : row) {
      if (static_cast<int>(cc.size()) > start && cc.back() == e.first) vv.back() += e.second;
      else {
        cc.push_back(e.first);
        vv.push_back(e.second);
      }
    }
    rp.push_back(static_cast<int>(cc.size()));
  }
  rowptr.swap(rp);
  col.swap(cc);
  val.swap(vv);
}

std::string DistCsr::contract_violation() const {
  if (static_cast<int>(rowptr.size()) != local_rows + 1 || (local_rows >= 0 && !rowptr.empty() && rowptr[0] != 0))
    return "row pointer array has the wrong size or does not start at 0";
  if (col.size() != val.size() || static_cast<long>(col.size()) != rowptr[local_rows])
    return "column/value arrays do not match the row pointers";
  std::vector<int> c;
  for (int i = 0; i < local_rows; ++i) {
    if (rowptr[i + 1] < rowptr[i]) return strcat("row pointers decrease at local row ", i);
    c.assign(col.begin() + rowptr[i], col.begin() + rowptr[i + 1]);
    for (int j : c)
      if (j < 0 || j >= n_global) return strcat("column index ", j, " out of range in global row ", row_begin + i);
    std::sort(c.begin(), c.end());
    if (std::adjacent_find(c.begin(), c.end()) != c.end())
      return strcat("duplicate column index in global row ", row_begin + i,
                    " (call DistCsr::sum_duplicates() to sum them)");
  }
  return "";
}

double DistCsr::frobenius_norm() const {
  double s = 0, g = 0;
  for (double v : val) s += v * v;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
  return std::sqrt(g);
}

// ------------------------------------------------------------------------------------------------ backends
const char* to_string(LUBackend b) {
  switch (b) {
    case LUBackend::EigenLU: return "eigen";
    case LUBackend::StrumpackLU: return "strumpack";
    case LUBackend::StrumpackMPI: return "strumpack-mpi";
  }
  return "?";
}

LUBackend lu_backend_from_string(const std::string& s) {
  if (s == "eigen") return LUBackend::EigenLU;
  if (s == "strumpack") return LUBackend::StrumpackLU;
  if (s == "strumpack-mpi") return LUBackend::StrumpackMPI;
  throw Error("unknown sparse LU backend '" + s + "'", "", "Use eigen, strumpack or strumpack-mpi.");
}

bool lu_backend_available(LUBackend b) {
  switch (b) {
    case LUBackend::EigenLU: return true;
    case LUBackend::StrumpackLU:
#ifdef AXF_STRUMPACK
      return true;
#else
      return false;
#endif
    case LUBackend::StrumpackMPI:
#ifdef AXF_STRUMPACK_MPI
      return true;
#else
      return false;
#endif
  }
  return false;
}

struct SparseLU::Impl {
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<int> dist;  // row distribution of the DistCsr
  // serial backends (rank 0 only)
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigen;
#ifdef AXF_STRUMPACK
  // Every call into this solver goes through the STRUMPACK guard of sym_solver.cpp (axf::strumpack_guarded_life
  // around construction and destruction, axf::strumpack_guarded_call around the rest, the ordering also under
  // axf::metis_mutex()): the EM solvers call the same STRUMPACK in the same process, and STRUMPACK keeps process-wide
  // state (per-thread timer lists, thread parameters, the std::new_handler, METIS's signal handlers) that every
  // caller has to coordinate.  In a build against the static MPI STRUMPACK (AXF_STRUMPACK_PREFIX) libaxionfem keeps
  // its STRUMPACK symbols private and this library links a second copy: the guard's locks still order these calls
  // with the EM solvers', but the guard empties the timer lists of libaxionfem's copy, not of this one (whose list
  // for thread number 0 then keeps a few entries per factorization and solve).
  std::unique_ptr<strumpack::StrumpackSparseSolver<double, int>> sp;
#endif
#ifdef AXF_STRUMPACK_MPI
  // Through the collective guard, like the distributed full-wave solver (model3d_fullwave.cpp;
  // axf::strumpack_guarded_collective_life around construction and destruction, axf::strumpack_guarded_collective_call
  // around the rest): one collective caller per rank, outside any OpenMP parallel region (thread number 0, which the
  // timer vector always covers). With the serialized guard every call holds the process-wide STRUMPACK lock across
  // its collectives. With the concurrent guard (8.0.x) the calls just run, so a distributed fluids solve that runs
  // while another thread of the same rank is inside a guarded STRUMPACK call is not coordinated with it. In neither
  // mode does its ordering take axf::metis_mutex() (METIS's signal handlers), and the std::new_handler is left to
  // STRUMPACK.
  struct MpiDelete {
    void operator()(strumpack::StrumpackSparseSolverMPIDist<double, int>* s) const {
      axf::strumpack_guarded_collective_life([&] { delete s; });
    }
  };
  std::unique_ptr<strumpack::StrumpackSparseSolverMPIDist<double, int>, MpiDelete> spmpi;
#endif
};

#ifdef AXF_STRUMPACK
namespace {
template <class S>
void configure_strumpack(S& s, bool matching) {
  auto& o = s.options();
  o.set_verbose(false);
  o.set_reordering_method(strumpack::ReorderingStrategy::METIS);
  o.set_matching(matching ? strumpack::MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING : strumpack::MatchingJob::NONE);
  o.set_compression(strumpack::CompressionType::NONE);
  o.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);  // plain LU solve: the residual we report is the LU's
}
std::string rc_name(strumpack::ReturnCode rc) {
  switch (rc) {
    case strumpack::ReturnCode::SUCCESS: return "SUCCESS";
    case strumpack::ReturnCode::MATRIX_NOT_SET: return "MATRIX_NOT_SET";
    case strumpack::ReturnCode::REORDERING_ERROR: return "REORDERING_ERROR";
    case strumpack::ReturnCode::ZERO_PIVOT: return "ZERO_PIVOT";
    case strumpack::ReturnCode::NO_CONVERGENCE: return "NO_CONVERGENCE";
    default: return "error " + std::to_string(static_cast<int>(rc));
  }
}
}  // namespace
#endif

SparseLU::SparseLU(LUBackend backend) : impl_(new Impl), backend_(backend) {
  AXF_REQUIRE(lu_backend_available(backend),
              std::string("sparse LU backend '") + to_string(backend) + "' is not in this build",
              "", backend == LUBackend::StrumpackMPI
                      ? "Configure axionFEM with -DAXF_STRUMPACK_PREFIX=<MPI STRUMPACK prefix> "
                        "(scripts/build_strumpack_mpi.sh), or use eigen/strumpack."
                      : "Build with STRUMPACK (scripts/build_strumpack.sh), or use eigen.");
  stats_.backend = to_string(backend);
}

SparseLU::~SparseLU() {
#ifdef AXF_STRUMPACK
  if (impl_ && impl_->sp) axf::strumpack_guarded_life([&] { impl_->sp.reset(); });
#endif
}

namespace {
// Collective failure: every rank passes its local message ("" = fine); if any rank failed, all ranks throw the
// message of the lowest failing rank. Prevents the deadlock of a rank-local throw before a collective call.
void collective_check(MPI_Comm comm, const std::string& local_msg, const std::string& hint) {
  const int me = rank_of(comm);
  int mine = local_msg.empty() ? size_of(comm) : me, first = 0;
  MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, comm);
  if (first == size_of(comm)) return;
  std::string msg = local_msg;
  int len = static_cast<int>(msg.size());
  MPI_Bcast(&len, 1, MPI_INT, first, comm);
  msg.resize(len);
  MPI_Bcast(msg.data(), len, MPI_CHAR, first, comm);
  throw Error(msg + strcat(" (reported by rank ", first, ")"), "", hint);
}
const char* kSingularHint =
    "The factorization met a zero pivot or a singular matrix. If this is a saddle point, check that every "
    "nullspace was removed (pressure constant: pin or mean-value multiplier) and that the boundary conditions "
    "determine the solution. Note that a singular saddle point is NOT reliably reported by a zero pivot: use "
    "SparseLU::set_kernel_probe().";
}  // namespace

SparseLU::ThreadGuard::ThreadGuard(int n) : saved_(omp_get_max_threads()), active_(n > 0) {
  if (active_) omp_set_num_threads(n);
}
SparseLU::ThreadGuard::~ThreadGuard() {
  if (active_) omp_set_num_threads(saved_);
}

void SparseLU::factorize(const DistCsr& A) {
  ThreadGuard tg(threads_);
  A_ = &A;
  impl_->comm = A.comm;
  impl_->dist = A.row_dist();
  const int me = rank_of(A.comm);
  stats_ = LUStats{};
  stats_.backend = to_string(backend_);
  stats_.n = A.n_global;
  long nnz_l = A.rowptr.empty() ? 0 : A.rowptr[A.local_rows];
  MPI_Allreduce(&nnz_l, &stats_.nnz, 1, MPI_LONG, MPI_SUM, A.comm);
  stats_.threads = omp_get_max_threads();
  stats_.ranks = size_of(A.comm);

  collective_check(A.comm, A.contract_violation(), "Fix the CSR input (see sparse_lu.hpp, input contract).");
  if (!probe_.empty()) {
    AXF_REQUIRE(static_cast<int>(probe_.size()) == A.local_rows, "SparseLU: kernel probe has the wrong local size");
    std::vector<double> y(A.local_rows);
    A.mult(probe_.data(), y.data());
    double loc[2] = {0, 0}, glob[2];
    for (int i = 0; i < A.local_rows; ++i) {
      loc[0] += y[i] * y[i];
      loc[1] += probe_[i] * probe_[i];
    }
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
    const double nA = A.frobenius_norm();
    stats_.kernel_probe = (nA > 0 && glob[1] > 0) ? std::sqrt(glob[0]) / (nA * std::sqrt(glob[1])) : 0.0;
    AXF_REQUIRE(stats_.kernel_probe > probe_tol_,
                strcat("the matrix maps the kernel probe to ~0 (||A z|| / (||A|| ||z||) = ", stats_.kernel_probe,
                       "): a nullspace was not removed, and an LU would return an arbitrary component along it"),
                "", "Remove the nullspace first (pressure: pin or mean-value multiplier; or a Krylov solver with "
                    "OrthoSolver).");
  }

  if (backend_ == LUBackend::StrumpackMPI) {
#ifdef AXF_STRUMPACK_MPI
    axf::strumpack_guarded_collective_life([&] {
      impl_->spmpi.reset();  // the old solver first, as for the shared-memory backend
      impl_->spmpi.reset(new strumpack::StrumpackSparseSolverMPIDist<double, int>(A.comm, false));
    });
    auto& s = *impl_->spmpi;
    configure_strumpack(s, matching_);
    double t0 = wtime();
    auto rc = strumpack::ReturnCode::SUCCESS;
    axf::strumpack_guarded_collective_call([&] {
      s.set_distributed_csr_matrix(A.local_rows, A.rowptr.data(), A.col.data(), A.val.data(), impl_->dist.data(),
                                   false);
      rc = s.reorder();
    });
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI reordering failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
    double t1 = wtime();
    axf::strumpack_guarded_collective_call([&] { rc = s.factor(); });
    // STRUMPACK-MPI reports ZERO_PIVOT only on the rank that met it: reduce before anything collective
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI factorization failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
    double t2 = wtime();
    stats_.analyze_seconds = t1 - t0;
    stats_.factor_seconds = t2 - t1;
    axf::strumpack_guarded_collective_call([&] {
      stats_.factor_nnz = static_cast<long>(s.factor_nonzeros());
      stats_.factor_mem_bytes = static_cast<double>(s.factor_memory());
    });
#endif
    return;
  }

  // serial backends: gather to rank 0
  CsrMatrix G = A.gather(0);
  std::string why;
  if (me == 0) {
    if (backend_ == LUBackend::EigenLU) {
      double t0 = wtime();
      Eigen::SparseMatrix<double> M = G.to_eigen();
      impl_->eigen.analyzePattern(M);
      double t1 = wtime();
      impl_->eigen.factorize(M);
      double t2 = wtime();
      stats_.analyze_seconds = t1 - t0;
      stats_.factor_seconds = t2 - t1;
      if (impl_->eigen.info() != Eigen::Success) why = "Eigen SparseLU failed: " + impl_->eigen.lastErrorMessage();
      else stats_.factor_nnz = static_cast<long>(impl_->eigen.nnzL() + impl_->eigen.nnzU());
    } else {
#ifdef AXF_STRUMPACK
      axf::strumpack_guarded_life([&] {
        impl_->sp.reset();  // the old solver first: STRUMPACK's constructor and destructor pair up new_handler swaps
        impl_->sp.reset(new strumpack::StrumpackSparseSolver<double, int>(false, true));
      });
      auto& s = *impl_->sp;
      configure_strumpack(s, matching_);
      double t0 = wtime();
      auto rc = strumpack::ReturnCode::SUCCESS;
      axf::strumpack_guarded_call([&] {
        // TRUE CSR: row pointers / column indices / values of A itself (not the CSC arrays of an Eigen matrix)
        s.set_csr_matrix(G.n, G.rowptr.data(), G.col.data(), G.val.data(), false);
        std::lock_guard<std::mutex> metis(axf::metis_mutex());
        rc = s.reorder();
      });
      double t1 = wtime();
      if (rc == strumpack::ReturnCode::SUCCESS) axf::strumpack_guarded_call([&] { rc = s.factor(); });
      double t2 = wtime();
      stats_.analyze_seconds = t1 - t0;
      stats_.factor_seconds = t2 - t1;
      if (rc != strumpack::ReturnCode::SUCCESS) why = "STRUMPACK LU failed (" + rc_name(rc) + ")";
      else {
        axf::strumpack_guarded_call([&] {
          stats_.factor_nnz = static_cast<long>(s.factor_nonzeros());
          stats_.factor_mem_bytes = static_cast<double>(s.factor_memory());
        });
      }
#endif
    }
  }
  collective_check(A.comm, why, kSingularHint);
  double tt[3] = {stats_.analyze_seconds, stats_.factor_seconds, stats_.factor_mem_bytes};
  MPI_Bcast(tt, 3, MPI_DOUBLE, 0, A.comm);
  stats_.analyze_seconds = tt[0];
  stats_.factor_seconds = tt[1];
  stats_.factor_mem_bytes = tt[2];
  long fn = stats_.factor_nnz;
  MPI_Bcast(&fn, 1, MPI_LONG, 0, A.comm);
  stats_.factor_nnz = fn;
}

std::vector<double> SparseLU::raw_solve(const std::vector<double>& b_local) {
  const DistCsr& A = *A_;
  const int me = rank_of(A.comm), P = size_of(A.comm);
  std::vector<double> x(A.local_rows, 0.0);
  if (backend_ == LUBackend::StrumpackMPI) {
#ifdef AXF_STRUMPACK_MPI
    auto rc = strumpack::ReturnCode::SUCCESS;
    axf::strumpack_guarded_collective_call([&] { rc = impl_->spmpi->solve(b_local.data(), x.data(), false); });
    collective_check(A.comm, rc == strumpack::ReturnCode::SUCCESS ? "" : "STRUMPACK-MPI solve failed (" +
                                                                            rc_name(rc) + ")", kSingularHint);
#endif
    return x;
  }
  const auto& dist = impl_->dist;
  std::vector<int> counts(P);
  for (int p = 0; p < P; ++p) counts[p] = dist[p + 1] - dist[p];
  std::vector<double> b, xg;
  if (me == 0) {
    b.resize(A.n_global);
    xg.resize(A.n_global);
  }
  MPI_Gatherv(b_local.data(), A.local_rows, MPI_DOUBLE, b.data(), counts.data(), dist.data(), MPI_DOUBLE, 0, A.comm);
  std::string why;
  if (me == 0) {
    if (backend_ == LUBackend::EigenLU) {
      Eigen::Map<Eigen::VectorXd> bv(b.data(), A.n_global);
      Eigen::VectorXd xv = impl_->eigen.solve(bv);
      if (impl_->eigen.info() != Eigen::Success) why = "Eigen SparseLU solve failed";
      else std::copy(xv.data(), xv.data() + A.n_global, xg.data());
    } else {
#ifdef AXF_STRUMPACK
      auto rc = strumpack::ReturnCode::SUCCESS;
      axf::strumpack_guarded_call([&] { rc = impl_->sp->solve(b.data(), xg.data(), false); });
      if (rc != strumpack::ReturnCode::SUCCESS) why = "STRUMPACK solve failed (" + rc_name(rc) + ")";
#endif
    }
  }
  collective_check(A.comm, why, kSingularHint);  // before the scatter: no rank may be left waiting in it
  MPI_Scatterv(xg.data(), counts.data(), dist.data(), MPI_DOUBLE, x.data(), A.local_rows, MPI_DOUBLE, 0, A.comm);
  return x;
}

std::vector<double> SparseLU::solve(const std::vector<double>& b_local) {
  AXF_REQUIRE(A_ != nullptr, "SparseLU::solve before factorize");
  const DistCsr& A = *A_;
  AXF_REQUIRE(static_cast<int>(b_local.size()) == A.local_rows, "SparseLU::solve: wrong local rhs size");
  double t0 = wtime();
  auto rel_res = [&](const std::vector<double>& x, std::vector<double>& r) {
    A.mult(x.data(), r.data());
    double loc[2] = {0, 0}, glob[2];
    for (int i = 0; i < A.local_rows; ++i) {
      r[i] = b_local[i] - r[i];
      loc[0] += r[i] * r[i];
      loc[1] += b_local[i] * b_local[i];
    }
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
    return glob[1] > 0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
  };
  // The residual the caller GATES on: over the rows of set_residual_rows() only (the Stokes solver
  // excludes its Dirichlet rows, which are identity rows carrying the boundary data), relative to the norm of the
  // rhs over the same rows. Without a mask it is rel_res. Used by the Krylov fallback only: the refinement and the
  // reported stats keep their definitions.
  auto gate_res = [&](const std::vector<double>& x, std::vector<double>& r) {
    if (row_mask_.empty()) return rel_res(x, r);
    A.mult(x.data(), r.data());
    double loc[2] = {0, 0}, glob[2];
    for (int i = 0; i < A.local_rows; ++i) {
      if (!row_mask_[i]) continue;
      const double ri = b_local[i] - r[i];
      loc[0] += ri * ri;
      loc[1] += b_local[i] * b_local[i];
    }
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, A.comm);
    return glob[1] > 0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
  };
  ThreadGuard tg(threads_);
  std::vector<double> x = raw_solve(b_local), r(A.local_rows);
  double res = rel_res(x, r);
  stats_.rel_residual_before = res;
  stats_.refine_steps = 0;
  // iterative refinement with the true matrix; stops on stagnation (a step that gains less than 10 %) once
  // min_refine_ is done, and keeps the best iterate
  std::vector<double> best = x;
  double best_res = res;
  while (std::isfinite(res) && (res > refine_rtol_ || stats_.refine_steps < min_refine_ || to_stagnation_) &&
         stats_.refine_steps < max_refine_ && res > 0) {
    std::vector<double> dx = raw_solve(r);
    for (int i = 0; i < A.local_rows; ++i) x[i] += dx[i];
    const double prev = res;
    res = rel_res(x, r);
    ++stats_.refine_steps;
    if (std::isfinite(res) && res < best_res) {
      best = x;
      best_res = res;
    }
    if (stats_.refine_steps >= min_refine_ && !(res < 0.9 * prev)) break;  // stagnation: < 10 % gained
  }
  // When the plain refinement (a Richardson iteration with the LU) stagnates or diverges above the target -- a
  // poor static-pivoting factorization of a 3D saddle point (STRUMPACK-MPI, MC64; observed on a 3D traction problem at
  // 0.86 M unknowns: plain solve 0.14, refinement diverging) -- restarted GMRES right-preconditioned by the same LU, from the best iterate. Opt-in
  // (set_krylov_fallback; the 3D API turns it on), and it never runs when the refinement reached the target.
  stats_.krylov_iterations = 0;
  stats_.krylov_fallback = false;
  stats_.matching = matching_;
  double best_gate = std::numeric_limits<double>::infinity();
  if (krylov_fallback_ && std::isfinite(best_res) && max_refine_ > 0) best_gate = gate_res(best, r);
  if (krylov_fallback_ && std::isfinite(best_res) && best_gate > refine_rtol_ && max_refine_ > 0) {
    stats_.krylov_fallback = true;
    const int n = A.local_rows, m = 40, maxit = 400;
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
      double l = 0, g = 0;
      for (int i = 0; i < n; ++i) l += a[i] * c[i];
      MPI_Allreduce(&l, &g, 1, MPI_DOUBLE, MPI_SUM, A.comm);
      return g;
    };
    // stopping norm: the gated residual's denominator (rows of the mask; all rows without one)
    double bn2 = 0;
    for (int i = 0; i < n; ++i)
      if (row_mask_.empty() || row_mask_[i]) bn2 += b_local[i] * b_local[i];
    double bn2g = 0;
    MPI_Allreduce(&bn2, &bn2g, 1, MPI_DOUBLE, MPI_SUM, A.comm);
    const double bnorm = std::sqrt(bn2g);
    std::vector<double> xg = best, w(n);
    int it = 0;
    while (it < maxit && bnorm > 0) {
      A.mult(xg.data(), w.data());
      std::vector<double> r0(n);
      for (int i = 0; i < n; ++i) r0[i] = b_local[i] - w[i];
      const double beta = std::sqrt(dot(r0, r0));
      if (!(gate_res(xg, w) > 0.5 * refine_rtol_)) break;  // the gated residual (not the full one) decides
      std::vector<std::vector<double>> V(1, r0), Z;
      for (int i = 0; i < n; ++i) V[0][i] /= beta;
      std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
      std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
      g[0] = beta;
      int k = 0;
      for (; k < m && it < maxit; ++k, ++it) {
        Z.push_back(raw_solve(V[k]));
        A.mult(Z[k].data(), w.data());
        for (int j = 0; j <= k; ++j) {  // modified Gram-Schmidt
          H[j][k] = dot(w, V[j]);
          for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i];
        }
        H[k + 1][k] = std::sqrt(dot(w, w));
        for (int j = 0; j < k; ++j) {  // previous Givens rotations
          const double t = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
          H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
          H[j][k] = t;
        }
        const double den = std::hypot(H[k][k], H[k + 1][k]);
        cs[k] = den > 0 ? H[k][k] / den : 1.0;
        sn[k] = den > 0 ? H[k + 1][k] / den : 0.0;
        const double hk1 = H[k + 1][k];
        H[k][k] = den;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];
        if (!(hk1 > 0)) {
          ++k;
          ++it;
          break;
        }
        V.emplace_back(n);
        for (int i = 0; i < n; ++i) V[k + 1][i] = w[i] / hk1;
        if (std::fabs(g[k + 1]) / bnorm <= 0.5 * refine_rtol_) {
          ++k;
          ++it;
          break;
        }
      }
      std::vector<double> y(k, 0.0);  // back substitution H y = g
      for (int j = k - 1; j >= 0; --j) {
        double t = g[j];
        for (int l = j + 1; l < k; ++l) t -= H[j][l] * y[l];
        y[j] = H[j][j] != 0 ? t / H[j][j] : 0.0;
      }
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) xg[i] += y[j] * Z[j][i];
    }
    stats_.krylov_iterations = it;
    const double rg_gate = gate_res(xg, r);
    const double rg = rel_res(xg, r);
    if (std::isfinite(rg_gate) && rg_gate < best_gate) {
      best = xg;
      best_res = rg;
    }
  }
  x = std::move(best);
  res = best_res;
  stats_.rel_residual = res;
  stats_.solve_seconds = wtime() - t0;
  if (max_refine_ > 0 && throw_on_miss_)  // collective: res is the same on every rank
    AXF_REQUIRE(std::isfinite(res) && res <= refine_rtol_,
                strcat("sparse LU (", to_string(backend_), ") missed its accuracy target: relative residual ", res,
                       " > ", refine_rtol_, " after ", stats_.refine_steps, " refinement steps (plain solve ",
                       stats_.rel_residual_before, ")"),
                "", "Likely causes: a singular operator (an undetected nullspace, incompatible data), or an ill-scaled "
                    "one (large coefficient contrast, high aspect ratio). Check the boundary conditions; for a scaled "
                    "problem loosen the target (SparseLU::set_refinement; in the Stokes API: direct_tol).");
  return x;
}

}  // namespace axf::fluids
