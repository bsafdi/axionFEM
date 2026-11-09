// Sparse direct solver for the symmetric systems of the axisymmetric engine: Eigen's SimplicialLDLT, or
// STRUMPACK's multifrontal factorization (OpenMP, METIS nested dissection) for large systems when axionFEM is
// built with it (CMake: STRUMPACK found in AXF_DEPS -> AXF_STRUMPACK). The interface mirrors the Eigen solver
// subset the engine uses: analyzePattern / factorize / info / solve.
//
// The matrix must be symmetric with both triangles assembled: the STRUMPACK path hands Eigen's column-major
// (CSC) arrays to set_csr_matrix as if they were CSR, which is the same matrix only for A = A^T, so
// analyzePattern verifies it instead of silently solving with A^T.
//
// Repeated factorizations of the same pattern (Newton iterations, a frequency sweep) reuse the ordering and
// the symbolic factorization computed by analyzePattern; factorize() falls back to a fresh analysis if the
// pattern changes.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

namespace axf {

// Solve with a factorized Eigen SimplicialLDLT for SEVERAL right-hand sides at once.
//
// Eigen's own sparse triangular solve loops over the columns of the right-hand side
// (SparseTriangularView::solveInPlace), so it reads the whole factor once per column: the rank-N
// low-rank update of the axisymmetric engine's exterior map would read it N times, which dominates a
// Newton iteration.  This does the substitutions once for the whole block.  The arithmetic per column
// is the one Eigen does, in the same order, so the result is bit-identical (cpp/tests/test_solvers.cpp
// checks that).  `s` must be a factorized SimplicialLDLT (unit lower factor plus vectorD()).
template <class Ldlt>
Eigen::MatrixXd blocked_ldlt_solve(const Ldlt& s, const Eigen::MatrixXd& B) {
  using LMat = typename std::decay<decltype(s.matrixL().nestedExpression())>::type;
  const LMat& L = s.matrixL().nestedExpression();
  const auto D = s.vectorD();
  const Eigen::Index n = B.rows();
  const bool perm = s.permutationP().size() > 0;
  // transposed (rank x n) so that the values of one unknown across all right-hand sides are
  // contiguous: the substitutions update one such group per nonzero of the factor
  Eigen::MatrixXd X = (perm ? Eigen::MatrixXd(s.permutationP() * B) : B).transpose();
  for (Eigen::Index j = 0; j < n; ++j) {  // L y = P b   (unit diagonal, column-major sweep)
    for (typename LMat::InnerIterator it(L, static_cast<int>(j)); it; ++it)
      if (it.index() > j) X.col(it.index()) -= it.value() * X.col(j);
  }
  // D z = y, by the reciprocal, as Eigen's m_diag.asDiagonal().inverse() does
  for (Eigen::Index i = 0; i < n; ++i) X.col(i) *= 1.0 / D(i);
  for (Eigen::Index j = n - 1; j >= 0; --j) {  // L^T x = z
    for (typename LMat::InnerIterator it(L, static_cast<int>(j)); it; ++it)
      if (it.index() > j) X.col(j) -= it.value() * X.col(it.index());
  }
  Eigen::MatrixXd R = X.transpose();
  return perm ? Eigen::MatrixXd(s.permutationPinv() * R) : R;
}

// Optional backends of this build (CMake finds them under AXF_DEPS).
bool have_strumpack();
bool have_metis();
// STRUMPACK version ("8.0.0"), or "" in a build without it.
const char* strumpack_version();
// Which guard coordinates the calls into STRUMPACK (sym_solver.cpp): "concurrent" (STRUMPACK 8.0.x: calls from
// different OpenMP threads of a team run at the same time), "serialized" (any other version, or a build configured
// with -DAXF_STRUMPACK_SERIALIZE=ON: one STRUMPACK call at a time in the process, the MPI-distributed solvers'
// included, from OpenMP thread 0 only, and frequency-parallel STRUMPACK sweeps run serially), or "" without
// STRUMPACK.
const char* strumpack_guard_mode();
// strumpack_guard_mode() is "serialized": work that would call STRUMPACK from several OpenMP threads at once must run
// on one thread instead.
bool strumpack_calls_serialized();
// The guard around the calls into STRUMPACK (sym_solver.cpp): its state, for tests.
//   timer_entries: entries left in STRUMPACK's process-wide task-timer lists (each call empties its own: 0);
//   timer_lists: how many lists exist (one per OpenMP thread number that may call in);
//   max_same_thread_number: most calls seen at once on one thread number's lock (1 unless the lock is broken);
//   max_concurrent: most calls seen at once in the process (> 1 when OpenMP threads of a team call in; always 1
//   with the serialized guard);
//   max_concurrent_during_life: most calls seen at once while a solver was built or destroyed (1: exclusive);
//   num_threads, task_recursion_cutoff_level: STRUMPACK's thread parameters now (set from the OpenMP thread count
//   of the last guarded call outside a parallel region);
//   internals_known: the concurrent guard, which reads STRUMPACK's internals. The serialized guard does not:
//   timer_entries, timer_lists, num_threads and task_recursion_cutoff_level are then 0 and mean nothing.
// Without STRUMPACK all are 0.
struct StrumpackGuardStats {
  std::size_t timer_entries = 0, timer_lists = 0;
  int max_same_thread_number = 0, max_concurrent = 0, max_concurrent_during_life = 0;
  int num_threads = 0, task_recursion_cutoff_level = 0;
  bool internals_known = false;
};
StrumpackGuardStats strumpack_guard_stats(bool reset_maxima = false);
// strumpack_guard_stats().timer_entries: 0 also when the guard does not read the lists (internals_known false)
std::size_t strumpack_timer_entries();
// testing only: every guarded call holds its locks for `ms` more milliseconds (0: off)
void strumpack_guard_test_hold(int ms);
// One process-wide mutex around the METIS calls (the shared-memory STRUMPACK solvers' ordering, Eigen's
// MetisOrdering, MFEM's Mesh::GeneratePartitioning in the 3D engine, the chemistry and the fluids bindings): METIS's
// GKlib swaps the process's SIGABRT and SIGTERM handlers around each call, so overlapping calls can leave its handler
// installed (sym_solver.cpp). The ordering inside the MPI-distributed STRUMPACK solvers' reorder (METIS on the root
// rank) does not take it.
std::mutex& metis_mutex();
// The same guard for STRUMPACK solvers that live outside sym_solver.cpp (the fluids SparseLU): f runs under
// strumpack_guarded_life when it constructs or destroys a solver, and under strumpack_guarded_call when it calls
// set_csr_matrix, reorder, factor, solve or a factor statistic; a reorder inside f also takes metis_mutex() (after the
// guard's locks).  f must not take the Python GIL or wait for another thread that calls into STRUMPACK: either would
// deadlock.  Without STRUMPACK both just call f.
void strumpack_guarded_call(const std::function<void()>& f);
void strumpack_guarded_life(const std::function<void()>& f);
// The guard for the MPI-distributed STRUMPACK solvers (SparseSolverMPIDist: the distributed full-wave factor in
// model3d_fullwave.cpp, the fluids strumpack-mpi backend), whose calls are collective over the solver's communicator:
// f runs under strumpack_guarded_collective_life when it constructs or destroys a solver, and under
// strumpack_guarded_collective_call when it calls set_distributed_csr_matrix, reorder, factor, solve, inertia or a
// factor statistic (or builds, uses and destroys a solver in one go). Every rank makes the same calls.
//  - Serialized guard: f runs under the process-wide lock of every other STRUMPACK call, held across f's collectives,
//    and the call variant is refused (an axf::Error) from an OpenMP thread other than 0, as strumpack_guarded_call is.
//    While f waits for the other ranks no other STRUMPACK call of its process can start; that cannot deadlock as long
//    as one thread per rank uses the distributed solvers at a time (MPI does not allow two threads of one rank to
//    make collective calls on the same communicator at once anyway). The call variant is refused when another thread
//    of the process is inside a collective call already, instead of waiting behind it; the life variant, which runs
//    in destructors, waits.
//  - Concurrent guard (8.0.x): f just runs. These calls are then not coordinated with STRUMPACK calls from other
//    threads of the same process (a Python thread running a guarded solve while another runs a distributed one would
//    push to the same timer list as it); one STRUMPACK caller per rank is assumed while a distributed solver works.
// In both modes the std::new_handler is left to STRUMPACK: the solver's constructor installs its out-of-memory abort
// and its destructor puts back the handler it found. Without STRUMPACK both just call f.
void strumpack_guarded_collective_call(const std::function<void()>& f);
void strumpack_guarded_collective_life(const std::function<void()>& f);

class SymSolver {
 public:
  using SpMat = Eigen::SparseMatrix<double>;
  // systems with at least this many unknowns use STRUMPACK (when available); $AXF_STRUMPACK_MIN_DOFS
  // overrides it (docs/validation/solvers.md has the measurement behind the default)
  static int& threshold();
  SymSolver();
  ~SymSolver();
  SymSolver(const SymSolver&) = delete;
  SymSolver& operator=(const SymSolver&) = delete;
  explicit SymSolver(const SpMat& A) : SymSolver() { compute(A); }
  void analyzePattern(const SpMat& A);
  void factorize(const SpMat& A);
  void compute(const SpMat& A) {
    analyzePattern(A);
    factorize(A);
  }
  Eigen::ComputationInfo info() const { return info_; }
  Eigen::VectorXd solve(const Eigen::VectorXd& b) const;
  Eigen::MatrixXd solve(const Eigen::MatrixXd& B) const;
  // which backend factorized the last matrix ("eigen-ldlt" or "strumpack")
  const char* backend() const;
  // force a backend: -1 by size (threshold), 0 Eigen, 1 STRUMPACK (when built in); before analyzePattern
  void set_backend(int b) { force_ = b; }
  // Lossy compression of the STRUMPACK fronts, for a factor used as a preconditioner whose error an outer Krylov
  // refinement removes: "none" (default, exact) or "blr" (block low-rank, relative tolerance rel_tol, fronts of
  // separators with at least min_sep_size unknowns; < 0 keeps STRUMPACK's default). Before analyzePattern.
  struct Compression {
    std::string type = "none";
    double rel_tol = 1e-4;
    int min_sep_size = -1;
  };
  void set_compression(const Compression& c) { compression_ = c; }
  const Compression& compression() const { return compression_; }
  // nonzeros of the factor (fill)
  long factor_nonzeros() const;
  struct InertiaResult {
    bool accurate = false;
    long negative = -1, zero = -1, positive = -1;
    std::string status = "FACTORIZATION_UNAVAILABLE";
  };
  // Optional diagnostic on an existing STRUMPACK factor. Pivoted or otherwise
  // unsupported inertia returns no counts; it never changes the factorization.
  InertiaResult inertia() const;
  // factorizations that reused the symbolic analysis, and symbolic analyses performed (one per
  // analyzePattern, plus one per factorize() that saw a changed pattern)
  int reused_factorizations() const { return reused_; }
  int symbolic_analyses() const { return symbolic_; }

 private:
  struct Strumpack;
  Eigen::SimplicialLDLT<SpMat> ldlt_;
  std::unique_ptr<Strumpack> sp_;
  bool use_sp_ = false;
  bool factorized_ = false;
  int force_ = -1;
  Compression compression_;
  int reused_ = 0, symbolic_ = 0;
  Eigen::ComputationInfo info_ = Eigen::Success;
};

// Sparse direct solver for COMPLEX SYMMETRIC (not Hermitian) systems, A = A^T, such as the damped harmonic
// operator K (1 - i eta) - omega^2 M of the torsional module: Eigen's SparseLU (COLAMD), or STRUMPACK's
// multifrontal LU in complex<double> (OpenMP, METIS nested dissection) when built in. As SymSolver, the
// STRUMPACK path passes Eigen's CSC arrays as CSR (the same matrix only for A = A^T, which analyzePattern
// checks), and a frequency sweep reuses the ordering and symbolic factorization of analyzePattern.
class ComplexSymSolver {
 public:
  using CSpMat = Eigen::SparseMatrix<std::complex<double>>;
  using CVec = Eigen::VectorXcd;
  // backend: -1 by size (SymSolver::threshold(), like the real solver), 0 Eigen SparseLU, 1 STRUMPACK
  explicit ComplexSymSolver(int backend = -1);
  ~ComplexSymSolver();
  ComplexSymSolver(const ComplexSymSolver&) = delete;
  ComplexSymSolver& operator=(const ComplexSymSolver&) = delete;
  void analyzePattern(const CSpMat& A);
  void factorize(const CSpMat& A);
  Eigen::ComputationInfo info() const { return info_; }
  CVec solve(const CVec& b) const;
  const char* backend() const { return use_sp_ ? "strumpack" : "eigen-sparselu"; }
  long factor_nonzeros() const;
  int symbolic_analyses() const { return symbolic_; }
  int reused_factorizations() const { return reused_; }

 private:
  struct Strumpack;
  struct EigenLU;
  std::unique_ptr<Strumpack> sp_;
  std::unique_ptr<EigenLU> lu_;
  int force_ = -1;
  bool use_sp_ = false;
  int symbolic_ = 0, reused_ = 0;
  Eigen::ComputationInfo info_ = Eigen::Success;
};

}  // namespace axf
