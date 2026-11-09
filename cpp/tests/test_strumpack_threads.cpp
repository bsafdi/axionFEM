// STRUMPACK called from several threads of one process.
// CMake runs this test with OMP_NUM_THREADS=1, so STRUMPACK's timer vector has one list at load.
//   1. ComplexSymSolver and SymSolver with STRUMPACK from an OpenMP team of 4: before the guard, the timers of
//      threads 1-3 were written past that vector (heap corruption).  Correct solutions, emptied lists, one list per
//      thread number, and the calls of the team overlapped (each call is held 50 ms; max_concurrent >= 2).
//   2. Two std::threads (both OpenMP thread number 0, as Python threads are) factorize at the same time: their
//      calls must not overlap (max_same_thread_number == 1).  With the per-thread-number lock removed this
//      check fails.
//      The team also checks: constructions and destructions ran alone
//      (max_concurrent_during_life == 1); the std::new_handler and the SIGABRT/SIGTERM dispositions are unchanged
//      afterwards (METIS's GKlib swaps those around every call; 2D-grid matrices so that METIS takes a few ms);
//      STRUMPACK's thread parameters follow a top-level solve (4 threads: num_threads 4, cutoff 5) and are not
//      changed by the team.
//   3. The std::new_handler is the one the caller had after solvers are re-analysed and destroyed out of order.
// With the serialized guard (STRUMPACK other than 8.0.x, or -DAXF_STRUMPACK_SERIALIZE=ON; strumpack_guard_mode()) the
// team's checks are replaced by that guard's contract: thread 0 of a team solves, threads 1-3 are refused with a clear
// error instead of writing STRUMPACK's per-thread state, and no two calls ever overlap (max_concurrent == 1); 2 and 3
// are the same, without the checks of STRUMPACK's internals, which that guard does not read.
//   4. The guard of the MPI-distributed solvers (strumpack_guarded_collective_call/_life), with stand-in calls (no MPI
//      STRUMPACK needed). Serialized guard: a collective call holds the process-wide lock for its whole duration (the
//      stand-in waits 300 ms, as a call waiting for the other ranks would); a second thread's collective call is
//      refused meanwhile instead of queued behind it; a lifetime call and a shared-memory solve from other threads
//      wait for it; a collective call from an OpenMP thread other than 0 is refused. Concurrent guard: both just run.
#include <Eigen/Sparse>
#include <atomic>
#include <chrono>
#include <csignal>
#include <complex>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "axionfem/omp_shim.hpp"
#include "axionfem/sym_solver.hpp"

static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

using axf::ComplexSymSolver;
using axf::SymSolver;
using C = std::complex<double>;

// complex symmetric tridiagonal system, diagonally dominant
static ComplexSymSolver::CSpMat ctri(int n, double shift) {
  std::vector<Eigen::Triplet<C>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, C(2.5 + shift, 0.01));
    if (i) {
      t.emplace_back(i, i - 1, C(-1, 0));
      t.emplace_back(i - 1, i, C(-1, 0));
    }
  }
  ComplexSymSolver::CSpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// complex symmetric 5-point system on a g x g grid (enough for METIS to take a few ms)
static ComplexSymSolver::CSpMat cgrid(int g, double shift) {
  std::vector<Eigen::Triplet<C>> t;
  auto id = [g](int i, int j) { return i * g + j; };
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      t.emplace_back(id(i, j), id(i, j), C(4.5 + shift, 0.01));
      if (i) {
        t.emplace_back(id(i, j), id(i - 1, j), C(-1, 0));
        t.emplace_back(id(i - 1, j), id(i, j), C(-1, 0));
      }
      if (j) {
        t.emplace_back(id(i, j), id(i, j - 1), C(-1, 0));
        t.emplace_back(id(i, j - 1), id(i, j), C(-1, 0));
      }
    }
  ComplexSymSolver::CSpMat A(g * g, g * g);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

static bool default_disposition(int sig) {
  struct sigaction sa;
  sigaction(sig, nullptr, &sa);
  return sa.sa_handler == SIG_DFL;
}

// real symmetric tridiagonal system, positive definite
static SymSolver::SpMat rtri(int n, double shift) {
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 2.5 + shift);
    if (i) {
      t.emplace_back(i, i - 1, -1.0);
      t.emplace_back(i - 1, i, -1.0);
    }
  }
  SymSolver::SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

static void omp_team() {
  const int n = 4000, g = 120, p = 4;
  std::atomic<int> bad{0};
  // a top-level STRUMPACK solve sets the thread parameters from the current thread count (1 at load here)
  omp_set_num_threads(4);
  {
    SymSolver top;
    top.set_backend(1);
    top.compute(rtri(n, 0.0));
  }
  const auto before = axf::strumpack_guard_stats(true);
  std::printf("top level with 4 threads: num_threads %d, cutoff %d\n", before.num_threads,
              before.task_recursion_cutoff_level);
  CHECK(before.num_threads == 4 && before.task_recursion_cutoff_level == 5,
        "thread parameters not updated: num_threads %d, cutoff %d (want 4, 5)", before.num_threads,
        before.task_recursion_cutoff_level);
  CHECK(default_disposition(SIGABRT) && default_disposition(SIGTERM), "signal dispositions changed before the team");
  const std::new_handler handler = std::get_new_handler();
  axf::strumpack_guard_test_hold(50);
#ifdef AXF_HAVE_OPENMP
  omp_set_max_active_levels(2);
#endif
#pragma omp parallel num_threads(p)
  {
    const double shift = 0.1 * omp_get_thread_num();
    ComplexSymSolver cs(1);
    SymSolver rs;
    rs.set_backend(1);
    const auto A = cgrid(g, shift);
    const auto R = rtri(n, shift);
    cs.analyzePattern(A);
    rs.analyzePattern(R);
#pragma omp barrier
    for (int rep = 0; rep < 2; ++rep) {
      cs.factorize(A);
      rs.factorize(R);
      const ComplexSymSolver::CVec b = ComplexSymSolver::CVec::Ones(g * g);
      const Eigen::VectorXd br = Eigen::VectorXd::Ones(n);
      const double ec = (A * cs.solve(b) - b).norm() / b.norm();
      const double er = (R * rs.solve(br) - br).norm() / br.norm();
      if (!(ec < 1e-10) || !(er < 1e-10) || std::string(cs.backend()) != "strumpack" ||
          std::string(rs.backend()) != "strumpack")
        ++bad;
    }
  }
  axf::strumpack_guard_test_hold(0);
  const auto st = axf::strumpack_guard_stats(true);
  std::printf("team of %d: bad %d, timer entries %zu, lists %zu, most at once %d, same thread number %d, "
              "during construction %d\n", p,
              bad.load(), st.timer_entries, st.timer_lists, st.max_concurrent, st.max_same_thread_number,
              st.max_concurrent_during_life);
  CHECK(bad == 0, "team solves wrong or not STRUMPACK (%d)", bad.load());
  CHECK(st.timer_entries == 0, "timer entries left: %zu", st.timer_entries);
  CHECK(st.timer_lists >= static_cast<std::size_t>(p), "timer lists %zu < %d threads", st.timer_lists, p);
  CHECK(st.max_same_thread_number == 1, "calls on one thread number overlapped (%d)", st.max_same_thread_number);
  CHECK(st.max_concurrent >= 2, "the team's calls never overlapped (%d): concurrency lost", st.max_concurrent);
  CHECK(st.max_concurrent_during_life == 1, "a solver was built or destroyed while other calls ran (%d)",
        st.max_concurrent_during_life);
  CHECK(st.num_threads == 4 && st.task_recursion_cutoff_level == 5,
        "the team changed the thread parameters: num_threads %d, cutoff %d", st.num_threads,
        st.task_recursion_cutoff_level);
  CHECK(std::get_new_handler() == handler, "STRUMPACK's new_handler outlived the team's solvers");
  CHECK(default_disposition(SIGABRT), "SIGABRT no longer has its default action (METIS's handler left installed)");
  CHECK(default_disposition(SIGTERM), "SIGTERM no longer has its default action (METIS's handler left installed)");
}

static void python_like_threads() {
  const int n = 4000;
  std::atomic<int> bad{0}, ready{0};
  axf::strumpack_guard_stats(true);
  axf::strumpack_guard_test_hold(50);
  auto work = [&](int k) {
    ComplexSymSolver cs(1);
    const auto A = ctri(n, 0.3 * k);
    cs.analyzePattern(A);
    ++ready;
    while (ready.load() < 2) std::this_thread::yield();
    for (int rep = 0; rep < 3; ++rep) {
      cs.factorize(A);
      const ComplexSymSolver::CVec b = ComplexSymSolver::CVec::Ones(n);
      if (!((A * cs.solve(b) - b).norm() / b.norm() < 1e-10)) ++bad;
    }
  };
  std::thread t0(work, 0), t1(work, 1);
  t0.join();
  t1.join();
  axf::strumpack_guard_test_hold(0);
  const auto st = axf::strumpack_guard_stats(true);
  std::printf("two plain threads: bad %d, timer entries %s, same thread number %d, most at once %d\n", bad.load(),
              st.internals_known ? std::to_string(st.timer_entries).c_str() : "(not read)", st.max_same_thread_number,
              st.max_concurrent);
  CHECK(bad == 0, "plain-thread solves wrong (%d)", bad.load());
  if (st.internals_known) CHECK(st.timer_entries == 0, "timer entries left: %zu", st.timer_entries);
  CHECK(st.max_same_thread_number == 1, "two calls on thread number 0 overlapped (%d)", st.max_same_thread_number);
  if (axf::strumpack_calls_serialized())
    CHECK(st.max_concurrent == 1, "serialized guard: two STRUMPACK calls overlapped (%d)", st.max_concurrent);
}

// The serialized guard in an OpenMP team: thread 0 solves, the other threads are refused (an axf::Error naming the
// OpenMP thread) before STRUMPACK runs, nothing overlaps, and the new_handler and METIS's signal dispositions are
// the caller's afterwards.
static void omp_team_serialized() {
  const int n = 4000, p = 4;
  std::atomic<int> bad{0}, let_through{0}, refused{0}, team{0};
  omp_set_num_threads(p);
  axf::strumpack_guard_stats(true);
  const std::new_handler handler = std::get_new_handler();
  axf::strumpack_guard_test_hold(20);
#pragma omp parallel num_threads(p)
  {
    const int tid = omp_get_thread_num();
    if (tid == 0) team = omp_get_num_threads();
    try {
      SymSolver rs;
      rs.set_backend(1);
      const auto R = rtri(n, 0.1 * tid);
      rs.compute(R);
      const Eigen::VectorXd br = Eigen::VectorXd::Ones(n);
      if (!((R * rs.solve(br) - br).norm() / br.norm() < 1e-10) || std::string(rs.backend()) != "strumpack") ++bad;
      if (tid != 0) ++let_through;
    } catch (const std::exception& e) {
      if (tid != 0 && std::string(e.what()).find("OpenMP thread") != std::string::npos)
        ++refused;
      else
        ++bad;
    }
  }
  axf::strumpack_guard_test_hold(0);
  const auto st = axf::strumpack_guard_stats(true);
  std::printf("serialized team of %d: bad %d, let through %d, refused %d, most at once %d, during construction %d\n",
              team.load(), bad.load(), let_through.load(), refused.load(), st.max_concurrent,
              st.max_concurrent_during_life);
  CHECK(bad == 0, "thread 0's solve wrong, or a refusal without the expected message (%d)", bad.load());
  CHECK(let_through == 0, "%d team threads other than 0 were let into STRUMPACK", let_through.load());
  CHECK(refused == team - 1, "%d of %d team threads other than 0 refused", refused.load(), team.load() - 1);
  CHECK(st.max_concurrent == 1, "two STRUMPACK calls overlapped (%d)", st.max_concurrent);
  CHECK(st.max_concurrent_during_life == 1, "a solver was built or destroyed while other calls ran (%d)",
        st.max_concurrent_during_life);
  CHECK(!st.internals_known, "the serialized guard reported STRUMPACK internals");
  CHECK(std::get_new_handler() == handler, "STRUMPACK's new_handler outlived the team's solvers");
  CHECK(default_disposition(SIGABRT), "SIGABRT no longer has its default action (METIS's handler left installed)");
  CHECK(default_disposition(SIGTERM), "SIGTERM no longer has its default action (METIS's handler left installed)");
}

static void new_handler_restored() {
  const std::new_handler before = std::get_new_handler();
  {
    ComplexSymSolver s(1);
    s.analyzePattern(ctri(500, 0.0));
    s.analyzePattern(ctri(501, 0.0));  // re-analysis replaces a live STRUMPACK solver
    auto a = std::make_unique<ComplexSymSolver>(1), b = std::make_unique<ComplexSymSolver>(1);
    a->analyzePattern(ctri(500, 0.0));
    b->analyzePattern(ctri(500, 0.0));
    a.reset();  // not in reverse order of construction
    b.reset();
  }
  CHECK(std::get_new_handler() == before, "STRUMPACK's new_handler outlived its solvers");
}

static void collective_guard() {
  int ran = 0;
  axf::strumpack_guarded_collective_call([&] { ++ran; });
  axf::strumpack_guarded_collective_life([&] { ++ran; });
  CHECK(ran == 2, "the collective guard did not run its functions (%d of 2)", ran);
  if (!axf::strumpack_calls_serialized()) {
    std::printf("collective guard: runs its functions (concurrent mode)\n");
    return;
  }
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  auto us = [t0] { return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count(); };
  axf::strumpack_guard_stats(true);
  std::atomic<bool> inside{false};
  std::atomic<long long> left_at{-1}, life_at{-1}, solve_at{-1};
  std::thread holder([&] {
    axf::strumpack_guarded_collective_call([&] {
      inside = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      left_at = us();
    });
  });
  while (!inside.load()) std::this_thread::yield();
  std::string refusal;
  try {
    axf::strumpack_guarded_collective_call([] {});
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  std::thread life([&] { axf::strumpack_guarded_collective_life([&] { life_at = us(); }); });
  std::atomic<int> bad{0};
  std::thread solve([&] {
    const int n = 500;
    ComplexSymSolver cs(1);
    const auto A = ctri(n, 0.0);
    cs.analyzePattern(A);
    cs.factorize(A);
    const ComplexSymSolver::CVec b = ComplexSymSolver::CVec::Ones(n);
    if (!((A * cs.solve(b) - b).norm() / b.norm() < 1e-10)) ++bad;
    solve_at = us();
  });
  holder.join();
  life.join();
  solve.join();
  // OpenMP thread 1 of a team is refused
  std::atomic<int> team{0}, omp_ran{0}, omp_refused{0};
  omp_set_num_threads(2);
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) team = omp_get_num_threads();
    if (omp_get_thread_num() == 1) {
      try {
        axf::strumpack_guarded_collective_call([&] { ++omp_ran; });
      } catch (const std::exception& e) {
        if (std::string(e.what()).find("OpenMP thread") != std::string::npos) ++omp_refused;
      }
    }
  }
  omp_set_num_threads(1);
  axf::strumpack_guarded_collective_call([&] { ++ran; });  // free again
  const auto st = axf::strumpack_guard_stats(true);
  std::printf("collective guard (serialized): refused while busy %d, holder left %.0f ms, lifetime call %.0f ms, "
              "solve done %.0f ms, OpenMP thread 1 refused %d of %d, most at once %d\n", !refusal.empty(),
              left_at / 1e3, life_at / 1e3, solve_at / 1e3, omp_refused.load(), team.load() - 1, st.max_concurrent);
  CHECK(refusal.find("two threads of one process") != std::string::npos,
        "a second thread's collective call was not refused while one was in progress ('%s')", refusal.c_str());
  CHECK(left_at >= 0 && life_at >= left_at, "a lifetime call ran while a collective call held the lock");
  CHECK(solve_at >= left_at, "a shared-memory solve finished while a collective call held the lock");
  CHECK(bad == 0, "the shared-memory solve was wrong");
  CHECK(omp_ran == 0 && omp_refused == team - 1, "OpenMP thread 1: ran %d, refused %d", omp_ran.load(),
        omp_refused.load());
  CHECK(ran == 3, "the collective guard stayed busy after its call");
  CHECK(st.max_concurrent == 1, "two STRUMPACK calls overlapped (%d)", st.max_concurrent);
}

int main() {
  if (!axf::have_strumpack()) {
    std::printf("SKIP: no STRUMPACK in this build\n");
    return 0;
  }
  std::printf("STRUMPACK %s, guard mode %s\n", axf::strumpack_version(), axf::strumpack_guard_mode());
  if (axf::strumpack_calls_serialized())
    omp_team_serialized();
  else
    omp_team();
  python_like_threads();
  new_handler_restored();
  collective_guard();
  if (failures) std::printf("%d failure(s)\n", failures);
  else std::printf("all STRUMPACK thread checks passed\n");
  return failures ? 1 : 0;
}
