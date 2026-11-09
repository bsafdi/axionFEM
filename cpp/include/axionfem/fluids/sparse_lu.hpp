// General (nonsymmetric, indefinite) sparse direct solvers for the fluids module.
//
// axf::SymSolver is symmetric-only by design: it hands Eigen's CSC arrays to STRUMPACK as if they were CSR and
// uses LDL^T without pivoting. Neither is valid for Oseen/Newton Jacobians, nor safe for the indefinite Stokes
// saddle point. This file is the general path:
//
//   * CsrMatrix / DistCsr: TRUE compressed-row storage (row pointers, column indices and values of A, not of A^T).
//     DistCsr is a block-row distribution: each rank holds rows [row_begin, row_begin + local_rows) with GLOBAL
//     column indices, the layout of a merged mfem::HypreParMatrix and of STRUMPACK's MPIDist interface.
//   * SparseLU: one interface over three backends,
//       - EigenLU         Eigen::SparseLU with COLAMD ordering (serial; the robust baseline),
//       - StrumpackLU     STRUMPACK multifrontal LU, shared memory (OpenMP), METIS ordering, MC64 matching,
//       - StrumpackMPI    STRUMPACK distributed LU (StrumpackSparseSolverMPIDist), only in a build against an MPI
//                         STRUMPACK (AXF_STRUMPACK_MPI).
//     The serial backends gather the matrix to rank 0, factor there, and scatter the solution; so every backend
//     accepts the same DistCsr on any number of ranks (rank invariance of the direct path).
//
// Accuracy contract. MC64 matching keeps the zero (2,2) block of a saddle point from
// producing a zero pivot, but it does NOT make the factorization accurate: STRUMPACK pivots only inside the fronts,
// and on Stokes matrices the plain LU solve can stop at a relative residual of 1e-9..1e-8. Every solve() therefore
// runs residual-checked iterative refinement with the caller's (true) matrix, up to `max_refine` steps, until
// ||A x - b|| <= refine_rtol ||b||, and THROWS (on every rank) when the target is missed.
//
// Input contract: within each row, column indices may come in any order but must be unique (duplicates are refused;
// call sum_duplicates() first), in range [0, n_global), with nondecreasing row pointers. A matrix with a known
// nullspace is not detected by the factorization (a singular saddle point is often "solved" with an arbitrary
// constant): pass the null vector with set_kernel_probe() and factorize() refuses the matrix when A z ~ 0.
//
// Errors are collective: a failure on any rank (e.g. a ZERO_PIVOT that STRUMPACK-MPI reports on one rank only) is
// reduced over the communicator and thrown on all ranks, so no rank is left waiting in a collective call.
//
// An internal API that may change; it is kept free of anything EM- or spec-specific. Index type: int (nnz and
// n_global < 2^31; larger 3D problems would need 64-bit indices).
#pragma once

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Sparse>

namespace mfem {
class Mesh;
}

namespace axf::fluids {

// Sequential CSR. rowptr has n+1 entries; col/val have rowptr[n] entries. Column indices need not be sorted.
struct CsrMatrix {
  int n = 0;
  std::vector<int> rowptr, col;
  std::vector<double> val;
  long nnz() const { return rowptr.empty() ? 0 : rowptr.back(); }
  // y = A x
  void mult(const double* x, double* y) const;
  // From an Eigen matrix of either storage order: the result is always CSR of the SAME matrix.
  static CsrMatrix from_eigen(const Eigen::SparseMatrix<double, Eigen::ColMajor>& A);
  static CsrMatrix from_eigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
  Eigen::SparseMatrix<double, Eigen::ColMajor> to_eigen() const;
};

// Refuse a mesh that is partitioned over more than one MPI rank (a ParMesh on a communicator of size > 1): the serial
// one-rank operators (ScalarTransport, BoussinesqSolver) would otherwise solve one independent problem per partition.
// A plain mfem::Mesh is serial by construction and is accepted.
void require_single_rank(const mfem::Mesh& mesh, const char* what);

// Block-row distributed CSR with global column indices. On one rank it is a CsrMatrix with row_begin = 0.
struct DistCsr {
  MPI_Comm comm = MPI_COMM_WORLD;
  long n_global = 0;
  long row_begin = 0;
  int local_rows = 0;
  std::vector<int> rowptr, col;  // col: GLOBAL indices (int: hypre here is built with 32-bit HYPRE_BigInt)
  std::vector<double> val;
  // row offsets of all ranks (size nranks+1), collective
  std::vector<int> row_dist() const;
  // y_local = (A x)_local with x distributed like the rows; collective (gathers x)
  void mult(const double* x_local, double* y_local) const;
  // gather the whole matrix onto `root` (empty on the other ranks); collective
  CsrMatrix gather(int root = 0) const;
  static DistCsr from_serial(const CsrMatrix& A, MPI_Comm comm = MPI_COMM_SELF);
  // Block-row distribution of a serial matrix held identically on every rank of comm: rank p keeps rows
  // [dist[p], dist[p+1]) (dist of size nranks+1; empty ranks allowed). Collective.
  static DistCsr distribute(const CsrMatrix& A, const std::vector<int>& dist, MPI_Comm comm);
  // Sort the column indices of each row and sum duplicate entries (as Eigen's setFromTriplets does).
  void sum_duplicates();
  // Check the input contract; returns "" or a description of the first violation (local rows). Not collective.
  std::string contract_violation() const;
  // Frobenius norm over all ranks. Collective.
  double frobenius_norm() const;
};

enum class LUBackend { EigenLU, StrumpackLU, StrumpackMPI };
const char* to_string(LUBackend b);
// "eigen" | "strumpack" | "strumpack-mpi"; throws axf::Error with the accepted names otherwise
LUBackend lu_backend_from_string(const std::string& s);
bool lu_backend_available(LUBackend b);

struct LUStats {
  std::string backend;
  long n = 0, nnz = 0;
  long factor_nnz = -1;            // fill (nnz(L)+nnz(U)); -1 if the backend does not report it
  double factor_mem_bytes = -1;    // STRUMPACK's own estimate (factors only); -1 for Eigen
  double analyze_seconds = 0, factor_seconds = 0, solve_seconds = 0;
  double rel_residual_before = -1; // ||A x0 - b|| / ||b|| of the plain LU solve (before refinement)
  double rel_residual = -1;        // after refinement: what solve() guarantees (<= refine_rtol)
  int refine_steps = 0;
  int krylov_iterations = 0;       // GMRES iterations after a stagnated refinement (0 when not needed)
  bool krylov_fallback = false;    // the GMRES fallback ran (the refinement stagnated above the gated target)
  bool matching = true;            // MC64 matching/scaling was used by the factorization behind these stats
  double kernel_probe = -1;        // ||A z|| / (||A||_F ||z||) of the probe vector (set_kernel_probe), -1 if none
  int threads = 1, ranks = 1;
};

class SparseLU {
 public:
  explicit SparseLU(LUBackend backend);
  ~SparseLU();
  SparseLU(const SparseLU&) = delete;
  SparseLU& operator=(const SparseLU&) = delete;

  // Analyze + factor. Collective over A.comm. Throws axf::Error on a singular/failed factorization.
  // A is NOT copied: it must stay alive and unchanged until the last solve() (used for the residual check).
  void factorize(const DistCsr& A);
  // Solve A x = b for the local block rows of b; returns the local block of x. Collective.
  std::vector<double> solve(const std::vector<double>& b_local);
  const LUStats& stats() const { return stats_; }
  LUBackend backend() const { return backend_; }

  // STRUMPACK options (before factorize). MC64 matching is on by default: without it the zero (2,2) block of a
  // saddle point can give ZERO_PIVOT. Accuracy comes from the refinement in solve(), not from the matching. (On 3D
  // Taylor-Hood saddle points the matching DAMAGES the factorization -- a plain residual of 0.14 against 5e-16 without
  // it was observed -- so the 3D Stokes path runs without it and retries with it, see StokesOptions::lu_matching.)
  void set_matching(bool on) { matching_ = on; }
  // Iterative refinement: at most max_steps corrections with the caller's matrix, target ||r|| <= rtol ||b||.
  // max_steps = 0 disables it (the plain LU solve; for measurements only). solve() throws when rtol is missed.
  void set_refinement(int max_steps, double rtol) {
    max_refine_ = max_steps;
    refine_rtol_ = rtol;
  }
  // Throw when the refinement target is missed (default true). false: return the best iterate and report
  // stats().rel_residual; the caller's own acceptance test decides (the Stokes solver does).
  void set_throw_on_miss(bool on) { throw_on_miss_ = on; }
  // Refine until a step gains less than 10 % (or max_steps), even below the target: the best attainable accuracy
  // (the Stokes path uses it; small velocities next to O(1) pressures need it, e.g. the exact no-flow state)
  void set_refine_to_stagnation(bool on) { to_stagnation_ = on; }
  // When the refinement stagnates above its target, restarted GMRES preconditioned by the LU (opt-in)
  void set_krylov_fallback(bool on) { krylov_fallback_ = on; }
  // Rows (local; true = counted) of the residual the Krylov fallback stops on: the caller's acceptance rows. The
  // Stokes solver passes its non-Dirichlet rows. Empty: all rows.
  void set_residual_rows(std::vector<char> mask) { row_mask_ = std::move(mask); }
  // OpenMP threads for factorize() and solve() (0: leave the global setting). 1 is deterministic for STRUMPACK.
  void set_threads(int n) { threads_ = n; }
  // At least this many refinement steps even when the target is met (a "tightened" solve; capped by max_steps).
  void set_min_refinement(int steps) { min_refine_ = steps; }
  // A vector z (local block rows) with A z = 0 when a nullspace was NOT removed (e.g. the constant pressure).
  // factorize() refuses A if ||A z|| <= tol ||A||_F ||z||.
  void set_kernel_probe(std::vector<double> z_local, double tol = 1e-12) {
    probe_ = std::move(z_local);
    probe_tol_ = tol;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  LUBackend backend_;
  bool matching_ = true;
  int max_refine_ = 3;
  int min_refine_ = 0;
  bool throw_on_miss_ = true;
  bool to_stagnation_ = false;
  bool krylov_fallback_ = false;
  std::vector<char> row_mask_;
  int threads_ = 0;
  struct ThreadGuard {
    explicit ThreadGuard(int n);
    ~ThreadGuard();
    int saved_;
    bool active_;
  };
  double refine_rtol_ = 1e-12;
  std::vector<double> probe_;
  double probe_tol_ = 1e-12;
  LUStats stats_;
  std::vector<double> raw_solve(const std::vector<double>& b_local);  // one backend solve, collective
  const DistCsr* A_ = nullptr;  // the caller's matrix, for the residual check
};

}  // namespace axf::fluids
