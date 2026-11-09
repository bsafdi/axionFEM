#include "axionfem/sym_solver.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "axionfem/core.hpp"

#include <Eigen/SparseLU>

#ifdef AXF_STRUMPACK
#include <array>
#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <pthread.h>
#include <thread>

#include <StrumpackConfig.h>
// The concurrent guard reads and writes STRUMPACK 8.0 internals (its task-timer lists and thread parameters) and was
// checked against that version only. Any other version, or a build configured with -DAXF_STRUMPACK_SERIALIZE=ON,
// gets the serialized guard, which uses STRUMPACK's public interface only.
#if !defined(AXF_STRUMPACK_SERIALIZE) && STRUMPACK_VERSION_MAJOR == 8 && STRUMPACK_VERSION_MINOR == 0
#define AXF_STRUMPACK_CONCURRENT 1
#endif
#include <StrumpackSparseSolver.hpp>
#ifdef AXF_STRUMPACK_CONCURRENT
#include <StrumpackParameters.hpp>
#include <misc/TaskTimer.hpp>
#endif

#include "axionfem/omp_shim.hpp"
#endif

namespace axf {

std::mutex& metis_mutex() {
  static std::mutex m;
  return m;
}

namespace {
template <class F>
decltype(auto) with_metis(F&& f) {
  std::lock_guard<std::mutex> lock(metis_mutex());
  return f();
}
}  // namespace

#ifdef AXF_STRUMPACK
namespace {

// Every call into a shared-memory STRUMPACK solver goes through strumpack_call (reorder, factor, solve, inertia,
// factor_nonzeros) or strumpack_life (construction, destruction); the MPI-distributed solvers go through
// strumpack_collective (sym_solver.hpp, strumpack_guarded_collective_call/_life). There are two implementations: the
// concurrent guard for STRUMPACK 8.0.x, described next, and the serialized guard (further down) for every other
// version.
//
// The concurrent guard. STRUMPACK 8.0.0 keeps process-wide state that its callers have to coordinate:
//  - each reorder, factor and solve appends a TaskTimer to TaskTimer::time_log_list.list[omp_get_thread_num()]
//    (the calling thread's number), whether or not STRUMPACK_TASK_TIMERS is defined.  The vector of lists is
//    sized once, to omp_get_max_threads() when the library is loaded (1 in a process started with
//    OMP_NUM_THREADS=1, as batch scripts often do); nothing locks it and nothing empties it.  A call from OpenMP thread k >= that size
//    (the torsion module's parallel_frequencies) wrote past the vector and corrupted the heap; two calls on the
//    same thread number (Python threads, which are all thread 0) race on one std::list;
//  - the solver's constructor installs an "out of memory -> abort()" std::new_handler and its destructor puts
//    back the handler it saw, so lifetimes that do not end in reverse order leave STRUMPACK's handler
//    installed; the constructor also writes params::task_recursion_cutoff_level;
//  - params::num_threads, and with it the depth to which the elimination tree and the dense kernels spawn OpenMP
//    tasks, is fixed when the library is loaded.  With 1 at load and af.set_threads(16) afterwards, the
//    factorization ran on 16 threads with no tree or kernel tasks, only the extend-add taskloop: 1.5-1.9x slower
//    numerical factorization on 0.36-0.09 M unknowns than with 16 threads at load.
// So, with a reader-writer lock over the timer vector and the thread parameters:
//  - every guarded call first makes sure, under the exclusive lock, that the lists cover the calling thread and,
//    outside a parallel region, that the thread parameters match the current OpenMP thread count (STRUMPACK's own
//    formula), then holds the lock shared for the call and empties its thread's list afterwards (the lists are
//    never read: no STRUMPACK_TASK_TIMERS).  A frequency thread's first call is its solver's construction, which
//    grows the lists under the exclusive lock it holds anyway;
//  - strumpack_call (reorder, factor, solve, inertia, factor_nonzeros) also holds a mutex for the calling thread's
//    number: calls from different OpenMP threads of one team run concurrently, each pushing to its own list;
//    calls on the same thread number (Python threads) run one at a time;
//  - strumpack_life (construction, destruction) holds it exclusively, since the constructor stores
//    params::task_recursion_cutoff_level, which running factorizations read, and restores the std::new_handler
//    it found, so STRUMPACK's abort handler never outlives the call whatever order solvers die in.  The lock
//    prefers writers, so a frequency thread building its solver waits for the factorizations already running,
//    not for every later one.
//  - METIS (the shared-memory solvers' ordering, Eigen's MetisOrdering in axi_poloidal.cpp, and MFEM's mesh
//    partitioning on more than one rank: model3d.cpp, chem/current_distribution_3d.cpp, fluids_bindings*.cpp) runs
//    under one more process-wide mutex, metis_mutex(): GKlib wraps every METIS call in gk_sigtrap/gk_siguntrap,
//    which install its own SIGABRT and SIGTERM handlers process-wide and put back the ones they saw.  Two
//    overlapping calls can leave gk_sigthrow installed, and then abort() or SIGTERM (scancel, a SLURM time limit)
//    ends in SIGSEGV.  Its thread-local state (memory, jump buffers) is safe.  Taken last (after the thread number's
//    mutex); the cost is at most p orderings one after another at the start of a torsion team.  The distributed
//    solvers' reorder (METIS on the root rank, inside a collective call) does not take it.
// Inside one call STRUMPACK's own tasks push no timers: dense fronts have none, the BLR front's starts only for
// print_compressed_front_stats(), and the HSS and HODLR fronts (which have their own) cannot be reached through
// Strumpack::apply (none or blr).  The writer-preferring lock makes a construction wait for every factorization
// in flight: harmless at the start of a team, but a solver built in the middle of a sweep would stall the team for
// up to one factorization.
// The MPI-distributed solvers are not coordinated by this guard (strumpack_collective just runs them): their calls
// are collective, and holding the reader-writer lock or a thread number's mutex across them was not validated.
// One STRUMPACK caller per rank is assumed while a distributed solver works (sym_solver.hpp).
//
// The serialized guard (any other STRUMPACK version, or -DAXF_STRUMPACK_SERIALIZE=ON): whether another version keeps
// the same process-wide state, in the same places, has not been checked, so it touches no STRUMPACK internals and
// gives up concurrency instead:
//  - every STRUMPACK call of the process (strumpack_call, strumpack_life and the distributed solvers'
//    strumpack_collective alike) runs alone under one mutex, so two calls never share a timer list and no
//    constructor writes a parameter that a running factorization reads. A distributed solver's call holds it across
//    its collectives, until every rank has taken part; a second thread that tries a collective call meanwhile is
//    refused rather than queued behind it, since the first one may be waiting for a rank on which the second one's
//    peer holds the lock (sym_solver.hpp);
//  - strumpack_call is refused (an axf::Error) from an OpenMP thread other than number 0: 8.0 writes its timers to
//    the list of the calling thread's number in a vector sized at load, and nothing here can see that vector's size.
//    The library's own frequency-parallel STRUMPACK sweeps (torsion's parallel_frequencies) run serially in this
//    mode (strumpack_calls_serialized()); Python threads are all thread number 0;
//  - strumpack_life restores the std::new_handler it found, as the concurrent guard does (standard library only);
//    strumpack_collective leaves it to the distributed solver, as the concurrent guard does;
//  - not done: the timer lists are not emptied (in 8.0, a few entries per call stay until the process ends), and
//    STRUMPACK's thread parameters stay as the library set them at load (in 8.0, threads added later with
//    af.set_threads() factorize without the elimination-tree tasks: slower, not wrong);
//  - METIS runs under metis_mutex() as above (the distributed solvers' reorder holds the one mutex instead, which
//    keeps it apart from every other STRUMPACK ordering but not from MFEM's partitioning or Eigen's METIS ordering).
// The first guarded call prints a one-time warning that says so.
constexpr int kSlots = 64;

#ifdef AXF_STRUMPACK_CONCURRENT
// A reader-writer lock that lets a waiting writer in before new readers (glibc's default prefers readers, which
// would keep a frequency thread that wants to build its solver waiting until the other threads ran out of work).
#ifdef __GLIBC__
class RwLock {
 public:
  RwLock() {
    pthread_rwlockattr_t a;
    pthread_rwlockattr_init(&a);
    pthread_rwlockattr_setkind_np(&a, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    pthread_rwlock_init(&l_, &a);
    pthread_rwlockattr_destroy(&a);
  }
  ~RwLock() { pthread_rwlock_destroy(&l_); }
  RwLock(const RwLock&) = delete;
  RwLock& operator=(const RwLock&) = delete;
  void lock() { pthread_rwlock_wrlock(&l_); }
  void unlock() { pthread_rwlock_unlock(&l_); }
  void lock_shared() { pthread_rwlock_rdlock(&l_); }
  void unlock_shared() { pthread_rwlock_unlock(&l_); }

 private:
  pthread_rwlock_t l_;
};
#else
using RwLock = std::shared_mutex;
#endif

#endif

struct Gate {
#ifdef AXF_STRUMPACK_CONCURRENT
  RwLock rw;
  std::array<std::mutex, kSlots> slot;
#else
  std::mutex all;                       // the serialized guard: one STRUMPACK call at a time in the process
  std::atomic<bool> collective{false};  // a distributed solver's (collective) call is in progress
#endif
  std::array<std::atomic<int>, kSlots> in_slot{};
  std::atomic<int> active{0}, max_active{0}, max_same{0}, max_life{0}, hold_ms{0};
};

Gate& gate() {
  static Gate g;
  return g;
}

void atomic_max(std::atomic<int>& m, int v) {
  int c = m.load();
  while (v > c && !m.compare_exchange_weak(c, v)) {
  }
}

// counters for the tests (strumpack_guard_stats); a test may ask every call to hold its lock for a while
struct Counted {
  int slot;
  explicit Counted(int s) : slot(s) {
    Gate& g = gate();
    atomic_max(g.max_same, ++g.in_slot[slot]);
    atomic_max(g.max_active, ++g.active);
    if (const int ms = g.hold_ms.load(); ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
  ~Counted() {
    --gate().in_slot[slot];
    --gate().active;
  }
  Counted(const Counted&) = delete;
  Counted& operator=(const Counted&) = delete;
};

#ifdef AXF_STRUMPACK_CONCURRENT
int cutoff_for(int nt) { return nt == 1 ? 0 : static_cast<int>(std::log2(nt)) + 3; }

// under the shared or the exclusive lock
bool needs_update(int tid, bool top) {
  if (strumpack::TaskTimer::time_log_list.list.size() < static_cast<std::size_t>(tid + 1)) return true;
  if (!top) return false;
  const int nt = omp_get_max_threads();
  return strumpack::params::num_threads != nt || strumpack::params::task_recursion_cutoff_level != cutoff_for(nt);
}

// under the exclusive lock
void update(int tid, bool top) {
  auto& lists = strumpack::TaskTimer::time_log_list.list;
  const std::size_t need = static_cast<std::size_t>(std::max(tid + 1, omp_get_max_threads()));
  if (lists.size() < need) lists.resize(need);
  if (top) {
    const int nt = omp_get_max_threads();
    strumpack::params::num_threads = nt;
    strumpack::params::task_recursion_cutoff_level = cutoff_for(nt);
  }
}

// The shared lock, with the timer lists and thread parameters brought up to date first (under the exclusive lock).
std::shared_lock<RwLock> enter_shared(int tid, bool top) {
  Gate& g = gate();
  std::shared_lock<RwLock> shared(g.rw);
  while (needs_update(tid, top)) {
    shared.unlock();
    {
      std::unique_lock<RwLock> exclusive(g.rw);
      update(tid, top);
    }
    shared.lock();
  }
  return shared;
}

// reorder, factor, solve, inertia, factor_nonzeros: shared, plus the calling thread number's mutex
template <class F>
decltype(auto) strumpack_call(F&& f) {
  Gate& g = gate();
  const int tid = omp_get_thread_num();
  const auto shared = enter_shared(tid, !omp_in_parallel());
  std::lock_guard<std::mutex> slot(g.slot[tid % kSlots]);
  struct Clear {
    int tid;
    ~Clear() { strumpack::TaskTimer::time_log_list.list[tid].clear(); }
  } clear{tid};
  Counted counted(tid % kSlots);
  return f();
}

// construction and destruction (and a solver built and used in one go): exclusive, the new_handler restored
template <class F>
decltype(auto) strumpack_life(F&& f) {
  Gate& g = gate();
  const int tid = omp_get_thread_num();
  std::unique_lock<RwLock> exclusive(g.rw);
  update(tid, !omp_in_parallel());
  struct Restore {
    int tid;
    std::new_handler handler;
    ~Restore() {
      std::set_new_handler(handler);
      strumpack::TaskTimer::time_log_list.list[tid].clear();
    }
  } restore{tid, std::get_new_handler()};
  Counted counted(tid % kSlots);
  atomic_max(g.max_life, g.active.load());  // 1 while lifetime calls are exclusive (tests)
  return f();
}

// a solver built, used and destroyed in one go (the inertia self-check)
template <class F>
decltype(auto) strumpack_once(F&& f) {
  return strumpack_life(std::forward<F>(f));
}

// the MPI-distributed solvers: not coordinated by the concurrent guard (see above)
template <class F>
decltype(auto) strumpack_collective(F&& f, bool /*may_refuse*/) {
  return f();
}

#else  // the serialized guard

void notice_serialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    log(0, strcat("warning: STRUMPACK ", strumpack_version(),
#ifdef AXF_STRUMPACK_SERIALIZE
                  " with -DAXF_STRUMPACK_SERIALIZE=ON",
#else
                  " has not been validated with axionFEM's concurrent STRUMPACK guard (8.0.x has)",
#endif
                  ": every STRUMPACK call in this process runs alone, and frequency-parallel STRUMPACK sweeps run "
                  "one frequency at a time (slower there; docs/install.md, Sparse direct solvers)"));
  });
}

void require_thread_zero() {
  AXF_REQUIRE(omp_get_thread_num() == 0,
              strcat("STRUMPACK ", strumpack_version(), " was called from OpenMP thread ", omp_get_thread_num(),
                     "; with this STRUMPACK version axionFEM calls it from thread 0 only"),
              "", "Call the solver outside OpenMP parallel regions, or build against STRUMPACK 8.0.x "
                  "(scripts/build_strumpack.sh).");
}

template <class F>
decltype(auto) strumpack_call(F&& f) {
  require_thread_zero();
  notice_serialized();
  std::lock_guard<std::mutex> all(gate().all);
  Counted counted(0);
  return f();
}

template <class F>
decltype(auto) strumpack_life(F&& f) {
  notice_serialized();
  Gate& g = gate();
  std::lock_guard<std::mutex> all(g.all);
  struct Restore {
    std::new_handler handler;
    ~Restore() { std::set_new_handler(handler); }
  } restore{std::get_new_handler()};
  Counted counted(omp_get_thread_num() % kSlots);
  atomic_max(g.max_life, g.active.load());
  return f();
}

// a solver built, used and destroyed in one go: its reorder and factor push timers, so thread 0 only
template <class F>
decltype(auto) strumpack_once(F&& f) {
  require_thread_zero();
  return strumpack_life(std::forward<F>(f));
}

// An MPI-distributed solver's call (may_refuse: set_distributed_csr_matrix, reorder, factor, solve, inertia, factor
// statistics; thread 0 only) or construction or destruction (!may_refuse: never refused, since it runs in
// destructors). Every rank makes the same calls, collectively; the process-wide lock is held across them.
template <class F>
decltype(auto) strumpack_collective(F&& f, bool may_refuse) {
  if (may_refuse) require_thread_zero();
  notice_serialized();
  Gate& g = gate();
  bool idle = false;
  if (may_refuse) {
    if (!g.collective.compare_exchange_strong(idle, true))
      throw Error("two threads of one process called MPI-distributed STRUMPACK solvers at the same time", "",
                  "Use the distributed solvers from one thread per rank at a time.");
  } else {
    while (!g.collective.compare_exchange_weak(idle, true)) {
      idle = false;
      std::this_thread::yield();
    }
  }
  struct Release {
    std::atomic<bool>& busy;
    ~Release() { busy = false; }
  } release{g.collective};
  std::lock_guard<std::mutex> all(g.all);
  Counted counted(omp_get_thread_num() % kSlots);
  if (!may_refuse) atomic_max(g.max_life, g.active.load());
  return f();
}
#endif  // AXF_STRUMPACK_CONCURRENT

}  // namespace

StrumpackGuardStats strumpack_guard_stats(bool reset) {
  Gate& g = gate();
  StrumpackGuardStats st;
#ifdef AXF_STRUMPACK_CONCURRENT
  std::unique_lock<RwLock> exclusive(g.rw);
  for (const auto& l : strumpack::TaskTimer::time_log_list.list) st.timer_entries += l.size();
  st.timer_lists = strumpack::TaskTimer::time_log_list.list.size();
  st.num_threads = strumpack::params::num_threads;
  st.task_recursion_cutoff_level = strumpack::params::task_recursion_cutoff_level;
  st.internals_known = true;
#else
  std::lock_guard<std::mutex> all(g.all);
#endif
  st.max_same_thread_number = g.max_same.load();
  st.max_concurrent = g.max_active.load();
  st.max_concurrent_during_life = g.max_life.load();
  if (reset) {
    g.max_same = 0;
    g.max_active = 0;
    g.max_life = 0;
  }
  return st;
}

void strumpack_guard_test_hold(int ms) { gate().hold_ms = std::max(0, ms); }

// The entry points for STRUMPACK solvers outside this file: the shared-memory backend of cpp/src/fluids/sparse_lu.cpp,
// and the distributed solvers (SparseSolverMPIDist in model3d_fullwave.cpp and in the fluids strumpack-mpi backend).
void strumpack_guarded_call(const std::function<void()>& f) { strumpack_call(f); }
void strumpack_guarded_life(const std::function<void()>& f) { strumpack_life(f); }
void strumpack_guarded_collective_call(const std::function<void()>& f) { strumpack_collective(f, true); }
void strumpack_guarded_collective_life(const std::function<void()>& f) { strumpack_collective(f, false); }

const char* strumpack_guard_mode() {
#ifdef AXF_STRUMPACK_CONCURRENT
  return "concurrent";
#else
  return "serialized";
#endif
}

#else
StrumpackGuardStats strumpack_guard_stats(bool) { return {}; }
void strumpack_guard_test_hold(int) {}
void strumpack_guarded_call(const std::function<void()>& f) { f(); }
void strumpack_guarded_life(const std::function<void()>& f) { f(); }
void strumpack_guarded_collective_call(const std::function<void()>& f) { f(); }
void strumpack_guarded_collective_life(const std::function<void()>& f) { f(); }
const char* strumpack_guard_mode() { return ""; }
#endif

bool strumpack_calls_serialized() { return std::string(strumpack_guard_mode()) == "serialized"; }

std::size_t strumpack_timer_entries() { return strumpack_guard_stats(false).timer_entries; }

bool have_strumpack() {
#ifdef AXF_STRUMPACK
  return true;
#else
  return false;
#endif
}

bool have_metis() {
#ifdef AXF_HAVE_METIS
  return true;
#else
  return false;
#endif
}

const char* strumpack_version() {
#ifdef AXF_STRUMPACK
  static const std::string v = strcat(STRUMPACK_VERSION_MAJOR, ".", STRUMPACK_VERSION_MINOR, ".",
                                      STRUMPACK_VERSION_PATCH);
  return v.c_str();
#else
  return "";
#endif
}

// Unknowns above which STRUMPACK's threaded multifrontal factorization beats Eigen's serial simplicial
// LDLT on the meridional meshes.  Measured on the fiducial toroid with the exterior map kept out of the
// sparse matrix, one cluster node with 32 threads (benchmarks/solver_scaling.py ->
// docs/validation/solvers.md).  Engine solve seconds, Eigen+AMD / Eigen+METIS / STRUMPACK:
//    26 k unknowns  0.31 / 0.43 / 0.49
//    94 k           1.43 / 1.80 / 1.75
//   357 k           7.84 / 8.79 / 7.36
//   856 k         717.8  / 26.6 / 22.6
// STRUMPACK first wins at 3.6e5, so the default sits just below that; a much lower threshold (30000, say) would
// choose the slower backend over most of the useful range.
static constexpr int kDefaultMinDofs = 300000;

int& SymSolver::threshold() {
  static int t = [] {
    const char* v = std::getenv("AXF_STRUMPACK_MIN_DOFS");
    if (!v || !*v) return kDefaultMinDofs;
    // strtod, not atoi: "1e9" (a common way of writing "never use STRUMPACK") parses as 1 with atoi,
    // and anything above INT_MAX wraps to a negative number, so both would mean "always STRUMPACK".
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (end == v || (end && *end) || !std::isfinite(x) || x < 0) {
      log(1, strcat("warning: AXF_STRUMPACK_MIN_DOFS='", v, "' is not a non-negative number; using the "
                    "default ", kDefaultMinDofs));
      return kDefaultMinDofs;
    }
    if (x > static_cast<double>(INT_MAX)) return INT_MAX;  // "never"
    return static_cast<int>(x);
  }();
  return t;
}

namespace {

// SymSolver's contract is a symmetric matrix with both triangles assembled: Eigen's SimplicialLDLT
// reads only the lower triangle, and the STRUMPACK path passes the column-major (CSC) arrays to
// set_csr_matrix, which is the same matrix only for A = A^T.  An asymmetric matrix would therefore be
// solved as neither A nor A^T without a word, so check it.  Two deterministic probe vectors: |A x -
// A^T x| is zero to round-off for a symmetric A and of the size of |A x| otherwise, and the two
// products cost four sparse passes with no random access (a per-entry lookup of the transposed entry
// would be O(nnz) cache misses).
bool symmetric_to_round_off(const Eigen::SparseMatrix<double>& A) {
  const Eigen::Index n = A.rows();
  if (n != A.cols()) return false;
  Eigen::VectorXd x(n);
  unsigned s = 12345u;
  for (int probe = 0; probe < 2; ++probe) {
    for (Eigen::Index i = 0; i < n; ++i) {  // xorshift: reproducible, no <random> state
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      x(i) = static_cast<double>(s) / 2147483648.0 - 1.0;
    }
    const Eigen::VectorXd y = A * x, yt = A.transpose() * x;
    const double d = (y - yt).cwiseAbs().maxCoeff();
    const double scale = y.cwiseAbs().maxCoeff() + yt.cwiseAbs().maxCoeff();
    if (d > 1e-10 * scale) return false;
  }
  return true;
}

}  // namespace

#ifdef AXF_STRUMPACK
struct SymSolver::Strumpack {
  strumpack::StrumpackSparseSolver<double, int> s{false, true};
  std::vector<int> rowptr, col;
  std::vector<double> val;
  int n = 0;
  Strumpack() {
    auto& o = s.options();
    o.set_verbose(false);
    o.set_reordering_method(strumpack::ReorderingStrategy::METIS);
    o.set_matching(strumpack::MatchingJob::NONE);
    o.set_compression(strumpack::CompressionType::NONE);
    o.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
  }
  // CSR of a symmetric matrix = CSC of the same matrix (Eigen's default storage)
  void load(const SymSolver::SpMat& A) {
    n = static_cast<int>(A.rows());
    rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
  }
  bool same_pattern(const SymSolver::SpMat& A) const {
    return n == static_cast<int>(A.rows()) && static_cast<long>(col.size()) == A.nonZeros() &&
           std::equal(rowptr.begin(), rowptr.end(), A.outerIndexPtr()) &&
           std::equal(col.begin(), col.end(), A.innerIndexPtr());
  }
  void load_values(const SymSolver::SpMat& A) { val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros()); }
  void set_matrix() { s.set_csr_matrix(n, rowptr.data(), col.data(), val.data(), true); }
  void update_values() { s.update_matrix_values(n, rowptr.data(), col.data(), val.data(), true); }
  static std::string rc_name(strumpack::ReturnCode rc) {
    switch (rc) {
      case strumpack::ReturnCode::SUCCESS: return "SUCCESS";
      case strumpack::ReturnCode::MATRIX_NOT_SET: return "MATRIX_NOT_SET";
      case strumpack::ReturnCode::REORDERING_ERROR: return "REORDERING_ERROR";
      case strumpack::ReturnCode::ZERO_PIVOT: return "ZERO_PIVOT";
      case strumpack::ReturnCode::NO_CONVERGENCE: return "NO_CONVERGENCE";
      default: return "INACCURATE_INERTIA";
    }
  }
  void apply(const SymSolver::Compression& c) {
    auto& o = s.options();
    if (c.type == "blr") {
      o.set_compression(strumpack::CompressionType::BLR);
      o.set_compression_rel_tol(c.rel_tol);
      if (c.min_sep_size >= 0) o.set_compression_min_sep_size(c.min_sep_size);
    } else {
      AXF_REQUIRE(c.type == "none", "unknown STRUMPACK compression '" + c.type + "'", "", "Use none or blr.");
      o.set_compression(strumpack::CompressionType::NONE);
    }
  }
  // Ordering + symbolic factorization.  Kept out of factorize() so that the Newton iterations (and a
  // frequency sweep) reuse it: set_csr_matrix resets STRUMPACK's `reordered_` flag, so calling it on
  // every factorization reruns METIS every time.
  void reorder() {
    const auto rc = with_metis([&] { return s.reorder(); });
    AXF_REQUIRE(rc == strumpack::ReturnCode::SUCCESS,
                "STRUMPACK could not order the field system (" + rc_name(rc) + ")", "",
                "Set AXF_STRUMPACK_MIN_DOFS=1e18 to use the Eigen LDLT backend instead, or coarsen the mesh.");
  }
};
#else
struct SymSolver::Strumpack {};
#endif

SymSolver::SymSolver() = default;
SymSolver::~SymSolver() {
#ifdef AXF_STRUMPACK
  if (sp_) strumpack_life([&] { sp_.reset(); });
#endif
}

void SymSolver::analyzePattern(const SpMat& A) {
  factorized_ = false;
  AXF_REQUIRE(symmetric_to_round_off(A),
              "internal: SymSolver was given an unsymmetric matrix (it factorizes the symmetric part "
              "only, and the STRUMPACK backend would solve with its transpose)");
#ifdef AXF_STRUMPACK
  use_sp_ = force_ >= 0 ? force_ == 1 : A.rows() >= threshold();
  if (use_sp_) {
    SpMat C = A;
    C.makeCompressed();
    strumpack_life([&] {
      sp_.reset();  // the old solver first: STRUMPACK's constructor and destructor pair up its new_handler swaps
      sp_ = std::make_unique<Strumpack>();
    });
    sp_->apply(compression_);
    sp_->load(C);
    strumpack_call([&] {
      sp_->set_matrix();
      sp_->reorder();
    });
    ++symbolic_;
    return;
  }
#endif
  use_sp_ = false;
  ldlt_.analyzePattern(A);
  ++symbolic_;
}

void SymSolver::factorize(const SpMat& A) {
  factorized_ = false;
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    SpMat C = A;
    C.makeCompressed();
    const bool same = sp_->same_pattern(C);
    const auto rc = strumpack_call([&] {
      if (same) {  // the usual case: reuse the ordering and the symbolic factorization
        sp_->load_values(C);
        sp_->update_values();
      } else {  // the pattern changed (not expected in a Newton loop): analyze again
        sp_->load(C);
        sp_->set_matrix();
        sp_->reorder();
      }
      return sp_->s.factor();
    });
    if (same)
      ++reused_;
    else
      ++symbolic_;
    if (rc != strumpack::ReturnCode::SUCCESS) {
      log(1, "STRUMPACK factorization failed: " + Strumpack::rc_name(rc));
      info_ = Eigen::NumericalIssue;
    } else {
      info_ = Eigen::Success;
      factorized_ = true;
    }
    return;
  }
#endif
  ldlt_.factorize(A);
  info_ = ldlt_.info();
  factorized_ = info_ == Eigen::Success;
}

Eigen::VectorXd SymSolver::solve(const Eigen::VectorXd& b) const {
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    Eigen::VectorXd x(b.size());
    const auto rc = strumpack_call([&] { return sp_->s.solve(b.data(), x.data()); });
    AXF_REQUIRE(rc == strumpack::ReturnCode::SUCCESS,
                "STRUMPACK could not solve the factorized field system (" + Strumpack::rc_name(rc) + ")");
    return x;
  }
#endif
  return ldlt_.solve(b);
}

Eigen::MatrixXd SymSolver::solve(const Eigen::MatrixXd& B) const {
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    Eigen::MatrixXd X(B.rows(), B.cols());
    if (B.cols() > 0) {
      // one blocked solve, not one per column (STRUMPACK's dense kernels work on the whole block)
      const auto rc = strumpack_call([&] {
        return sp_->s.solve(static_cast<int>(B.cols()), B.data(), static_cast<int>(B.rows()), X.data(),
                            static_cast<int>(X.rows()));
      });
      AXF_REQUIRE(rc == strumpack::ReturnCode::SUCCESS,
                  "STRUMPACK could not solve the factorized field system (" + Strumpack::rc_name(rc) + ")");
    }
    return X;
  }
#endif
  return ldlt_.solve(B);
}

long SymSolver::factor_nonzeros() const {
#ifdef AXF_STRUMPACK
  if (use_sp_) return static_cast<long>(strumpack_call([&] { return sp_->s.factor_nonzeros(); }));
#endif
  return static_cast<long>(ldlt_.matrixL().nestedExpression().nonZeros());
}

SymSolver::InertiaResult SymSolver::inertia() const {
  InertiaResult result;
  if (!factorized_ || info_ != Eigen::Success) return result;
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    // Some installed STRUMPACK versions report SUCCESS while counting the
    // absolute LU diagonal as positive: 8.0.0's dense fronts (FrontDense and
    // FrontDenseMPI::matrix_inertia) test the sign of std::abs(pivot), so every
    // nonzero pivot counts as positive; 8.1.0 tests the real part. Test the
    // backend independently on a connected, diagonally dominant matrix with one
    // eigenvalue of each sign (8.0.0 returns 0 negative, 2 positive) and withhold
    // every count if it fails. No field-system settings or factors are changed
    // by this small check.
    static const bool backend_inertia_verified = strumpack_once([]() {
      Strumpack check;
      const int ptr[3] = {0, 2, 4}, col[4] = {0, 1, 0, 1};
      const double val[4] = {-2.0, 0.25, 0.25, 2.0};
      check.s.set_csr_matrix(2, ptr, col, val, true);
      if (with_metis([&] { return check.s.reorder(); }) != strumpack::ReturnCode::SUCCESS ||
          check.s.factor() != strumpack::ReturnCode::SUCCESS) return false;
      int negative = 0, zero = 0, positive = 0;
      const auto code = check.s.inertia(negative, zero, positive);
      return code == strumpack::ReturnCode::SUCCESS && negative == 1 && zero == 0 && positive == 1;
    });
    if (!backend_inertia_verified) {
      result.status = "BACKEND_INERTIA_SELF_CHECK_FAILED";
      return result;
    }
    int negative = -1, zero = -1, positive = -1;
    const auto code = strumpack_call([&] { return sp_->s.inertia(negative, zero, positive); });
    result.status = Strumpack::rc_name(code);
    if (code == strumpack::ReturnCode::SUCCESS) {
      if (negative < 0 || zero < 0 || positive < 0 ||
          static_cast<long>(negative) + zero + positive != sp_->n) {
        result.status = "INCONSISTENT_COUNTS";
        return result;
      }
      result.accurate = true;
      result.negative = negative;
      result.zero = zero;
      result.positive = positive;
    }
    return result;
  }
#endif
  result.status = "UNAVAILABLE_BACKEND";
  return result;
}

const char* SymSolver::backend() const { return use_sp_ ? "strumpack" : "eigen-ldlt"; }

// ------------------------------------------------------------------ complex symmetric systems
namespace {

bool complex_symmetric_to_round_off(const Eigen::SparseMatrix<std::complex<double>>& A) {
  const Eigen::Index n = A.rows();
  if (n != A.cols()) return false;
  Eigen::VectorXcd x(n);
  unsigned s = 54321u;
  for (Eigen::Index i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const double re = static_cast<double>(s) / 2147483648.0 - 1.0;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    x(i) = std::complex<double>(re, static_cast<double>(s) / 2147483648.0 - 1.0);
  }
  // A^T x, not the adjoint: the system is complex symmetric
  const Eigen::VectorXcd y = A * x;
  const Eigen::VectorXcd yt = A.transpose() * x;
  const double d = (y - yt).cwiseAbs().maxCoeff();
  const double scale = y.cwiseAbs().maxCoeff() + yt.cwiseAbs().maxCoeff();
  return d <= 1e-10 * scale;
}

}  // namespace

struct ComplexSymSolver::EigenLU {
  Eigen::SparseLU<CSpMat, Eigen::COLAMDOrdering<int>> lu;
};

#ifdef AXF_STRUMPACK
struct ComplexSymSolver::Strumpack {
  using C = std::complex<double>;
  strumpack::StrumpackSparseSolver<C, int> s{false, true};
  std::vector<int> rowptr, col;
  std::vector<C> val;
  int n = 0;
  Strumpack() {
    auto& o = s.options();
    o.set_verbose(false);
    o.set_reordering_method(strumpack::ReorderingStrategy::METIS);
    o.set_matching(strumpack::MatchingJob::NONE);
    o.set_compression(strumpack::CompressionType::NONE);
    o.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
  }
  void load(const CSpMat& A) {
    n = static_cast<int>(A.rows());
    rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
  }
  bool same_pattern(const CSpMat& A) const {
    return n == static_cast<int>(A.rows()) && static_cast<long>(col.size()) == A.nonZeros() &&
           std::equal(rowptr.begin(), rowptr.end(), A.outerIndexPtr()) &&
           std::equal(col.begin(), col.end(), A.innerIndexPtr());
  }
};
#else
struct ComplexSymSolver::Strumpack {};
#endif

ComplexSymSolver::ComplexSymSolver(int backend) : force_(backend) {}
ComplexSymSolver::~ComplexSymSolver() {
#ifdef AXF_STRUMPACK
  if (sp_) strumpack_life([&] { sp_.reset(); });
#endif
}

void ComplexSymSolver::analyzePattern(const CSpMat& A) {
  AXF_REQUIRE(complex_symmetric_to_round_off(A),
              "internal: ComplexSymSolver was given a matrix that is not complex symmetric (A != A^T)");
  AXF_REQUIRE(A.isCompressed(), "internal: ComplexSymSolver needs a compressed matrix");
#ifdef AXF_STRUMPACK
  use_sp_ = force_ >= 0 ? force_ == 1 : A.rows() >= SymSolver::threshold();
  if (use_sp_) {
    strumpack_life([&] {
      sp_.reset();
      sp_ = std::make_unique<Strumpack>();
    });
    const auto rc = strumpack_call([&] {
      sp_->load(A);
      sp_->s.set_csr_matrix(sp_->n, sp_->rowptr.data(), sp_->col.data(), sp_->val.data(), true);
      return with_metis([&] { return sp_->s.reorder(); });
    });
    AXF_REQUIRE(rc == strumpack::ReturnCode::SUCCESS,
                "STRUMPACK could not order the complex harmonic system", "",
                "Use solver.backend = \"eigen\" in the request, or coarsen the mesh.");
    ++symbolic_;
    return;
  }
#else
  AXF_REQUIRE(force_ != 1, "solver backend \"strumpack\" was requested, but this build has no STRUMPACK", "",
              "Use solver.backend = \"auto\" or \"eigen\", or build with scripts/build_strumpack.sh.");
#endif
  use_sp_ = false;
  lu_ = std::make_unique<EigenLU>();
  lu_->lu.analyzePattern(A);
  ++symbolic_;
}

void ComplexSymSolver::factorize(const CSpMat& A) {
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    const bool same = sp_->same_pattern(A);
    const auto rc = strumpack_call([&] {
      if (same) {
        sp_->val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
        sp_->s.update_matrix_values(sp_->n, sp_->rowptr.data(), sp_->col.data(), sp_->val.data(), true);
      } else {
        sp_->load(A);
        sp_->s.set_csr_matrix(sp_->n, sp_->rowptr.data(), sp_->col.data(), sp_->val.data(), true);
        with_metis([&] { return sp_->s.reorder(); });
      }
      return sp_->s.factor();
    });
    if (same)
      ++reused_;
    else
      ++symbolic_;
    info_ = rc == strumpack::ReturnCode::SUCCESS ? Eigen::Success : Eigen::NumericalIssue;
    return;
  }
#endif
  AXF_REQUIRE(lu_ != nullptr, "internal: ComplexSymSolver::factorize before analyzePattern");
  lu_->lu.factorize(A);
  ++reused_;
  info_ = lu_->lu.info();
}

ComplexSymSolver::CVec ComplexSymSolver::solve(const CVec& b) const {
#ifdef AXF_STRUMPACK
  if (use_sp_) {
    CVec x(b.size());
    const auto rc = strumpack_call([&] { return sp_->s.solve(b.data(), x.data()); });
    AXF_REQUIRE(rc == strumpack::ReturnCode::SUCCESS, "STRUMPACK could not solve the complex harmonic system");
    return x;
  }
#endif
  return lu_->lu.solve(b);
}

long ComplexSymSolver::factor_nonzeros() const {
#ifdef AXF_STRUMPACK
  if (use_sp_) return static_cast<long>(strumpack_call([&] { return sp_->s.factor_nonzeros(); }));
#endif
  if (!lu_) return 0;
  return static_cast<long>(lu_->lu.nnzL() + lu_->lu.nnzU());
}

}  // namespace axf
