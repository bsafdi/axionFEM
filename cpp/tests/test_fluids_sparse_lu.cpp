// fluids general sparse LU (cpp/include/axionfem/fluids/sparse_lu.hpp):
//   1. CsrMatrix::from_eigen gives the CSR of the SAME matrix from either Eigen storage order.
//   2. Every available backend solves a deliberately NONSYMMETRIC system (an Oseen-type convection-diffusion
//      block with strong convection, coupled to a divergence-like constraint with a zero block) to round-off,
//      checked against the known solution.
//   3. The CSC-as-CSR mistake SymSolver makes for symmetric matrices is detected: handing the column-major
//      arrays of the same nonsymmetric matrix as CSR gives an O(1) error, so test 2 would fail on that bug.
//   4. The serial backends gather a DistCsr: one-rank DistCsr and the CsrMatrix give the same answer.
//   5. The shared-memory STRUMPACK backend goes through the STRUMPACK guard of sym_solver.cpp: its calls are counted
//      there, its construction runs alone, and it leaves no entries in STRUMPACK's process-wide timer lists.
//   6. On MPI_COMM_WORLD: the nonsymmetric matrix distributed with uneven rows and an empty
//      rank, every backend; duplicate entries refused (and summed by sum_duplicates); a numerically singular matrix
//      fails on EVERY rank (no deadlock: STRUMPACK-MPI reports the zero pivot on one rank only); refinement stats.
#include <mpi.h>

#include <Eigen/Sparse>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/fluids/sparse_lu.hpp"
#include "axionfem/sym_solver.hpp"

using namespace axf::fluids;
static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

using SpMat = Eigen::SparseMatrix<double>;

// [ L  G ; D  0 ]: L = -Lap + Pe (c . grad) on a g x g grid (upwind-free central convection: nonsymmetric),
// G a forward-difference "gradient" (g*g x m), D = G^T scaled differently (so the coupling is nonsymmetric too)
static SpMat oseen_like(int g, double pe, int m) {
  const int n = g * g;
  std::vector<Eigen::Triplet<double>> t;
  auto id = [&](int i, int j) { return i * g + j; };
  const double cx = 1.0, cy = 0.5;
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      const int k = id(i, j);
      t.emplace_back(k, k, 4.0);
      if (i > 0) t.emplace_back(k, id(i - 1, j), -1.0 - 0.5 * pe * cx);
      if (i + 1 < g) t.emplace_back(k, id(i + 1, j), -1.0 + 0.5 * pe * cx);
      if (j > 0) t.emplace_back(k, id(i, j - 1), -1.0 - 0.5 * pe * cy);
      if (j + 1 < g) t.emplace_back(k, id(i, j + 1), -1.0 + 0.5 * pe * cy);
    }
  for (int c = 0; c < m; ++c) {
    const int a = (c * 7) % n, b = (c * 13 + 1) % n;
    t.emplace_back(a, n + c, 1.0);
    t.emplace_back(b, n + c, -1.0);
    t.emplace_back(n + c, a, 0.7);
    t.emplace_back(n + c, b, -1.3);
  }
  SpMat A(n + m, n + m);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

static double rel(const std::vector<double>& a, const Eigen::VectorXd& b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d += (a[i] - b(i)) * (a[i] - b(i));
    s += b(i) * b(i);
  }
  return std::sqrt(d / s);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  const SpMat A = oseen_like(30, 3.0, 25);
  const int n = static_cast<int>(A.rows());
  const double asym = (A - SpMat(A.transpose())).norm() / A.norm();
  CHECK(asym > 0.1, "test matrix is not strongly nonsymmetric (%g)", asym);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  Eigen::VectorXd xs(n);
  for (int i = 0; i < n; ++i) xs(i) = U(rng);
  const Eigen::VectorXd b = A * xs;
  std::vector<double> bv(b.data(), b.data() + n);

  // 1. storage orders
  {
    const CsrMatrix c1 = CsrMatrix::from_eigen(A);
    const CsrMatrix c2 = CsrMatrix::from_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor>(A));
    CHECK(c1.rowptr == c2.rowptr && c1.col == c2.col && c1.val == c2.val, "from_eigen differs by storage order");
    std::vector<double> y(n);
    c1.mult(xs.data(), y.data());
    CHECK(rel(y, b) < 1e-14, "CsrMatrix::mult is not A*x (%g)", rel(y, b));
  }
  const DistCsr D = DistCsr::from_serial(CsrMatrix::from_eigen(A), MPI_COMM_SELF);

  // 2. every available backend
  for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
    if (!lu_backend_available(be)) {
      std::printf("skip %s (not in this build)\n", to_string(be));
      continue;
    }
    axf::strumpack_guard_stats(true);  // reset the guard's maxima
    SparseLU lu(be);
    lu.factorize(D);
    const auto x = lu.solve(bv);
    const double ex = rel(x, xs);
    std::printf("%-14s err %.2e residual %.2e fill %ld\n", to_string(be), ex, lu.stats().rel_residual,
                lu.stats().factor_nnz);
    CHECK(ex < 1e-10, "%s: solution error %g", to_string(be), ex);
    CHECK(lu.stats().rel_residual < 1e-13, "%s: residual %g", to_string(be), lu.stats().rel_residual);
    CHECK(lu.stats().factor_nnz > 0, "%s: no fill reported", to_string(be));
    // 5. the STRUMPACK guard
    if (be == LUBackend::StrumpackLU) {
      const auto st = axf::strumpack_guard_stats(false);
      CHECK(st.max_concurrent >= 1 && st.max_concurrent_during_life == 1,
            "strumpack: the calls did not go through the STRUMPACK guard (max_concurrent %d, during a "
            "construction %d)", st.max_concurrent, st.max_concurrent_during_life);
      if (st.internals_known)   // the concurrent guard (STRUMPACK 8.0.x) empties them; the serialized one cannot see them
        CHECK(st.timer_entries == 0, "strumpack: %zu entries left in STRUMPACK's timer lists", st.timer_entries);
    }
  }

  // 3. the CSC-as-CSR hand-over solves with A^T: it must be caught by the check above
  {
    CsrMatrix wrong;
    wrong.n = n;
    wrong.rowptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);  // column pointers, posing as rows
    wrong.col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    wrong.val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
    const DistCsr Dw = DistCsr::from_serial(wrong, MPI_COMM_SELF);
    for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU}) {
      if (!lu_backend_available(be)) continue;
      SparseLU lu(be);
      lu.factorize(Dw);
      const auto x = lu.solve(bv);
      const double ex = rel(x, xs);
      CHECK(ex > 1e-3, "%s: CSC passed as CSR went undetected (error %g)", to_string(be), ex);
    }
  }

  // 4. structurally singular saddle point (a zero column) is reported, not solved
  {
    SpMat S(3, 3);
    S.insert(0, 0) = 1;
    S.insert(1, 0) = 1;
    S.insert(0, 1) = 2;
    S.insert(1, 1) = 2;  // rank 1 in the top-left, third row/col empty -> singular
    S.makeCompressed();
    const DistCsr Ds = DistCsr::from_serial(CsrMatrix::from_eigen(SpMat(S)), MPI_COMM_SELF);
    bool threw = false;
    try {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(Ds);
    } catch (const axf::Error&) {
      threw = true;
    }
    CHECK(threw, "Eigen SparseLU accepted a singular matrix silently");
  }

  // 5. backend names
  CHECK(lu_backend_from_string("strumpack-mpi") == LUBackend::StrumpackMPI, "backend name parse");
  bool threw = false;
  try {
    lu_backend_from_string("umfpack");
  } catch (const axf::Error&) {
    threw = true;
  }
  CHECK(threw, "unknown backend name accepted");

  // 6. distributed (MPI_COMM_WORLD)
  {
    int P = 1, me = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    const CsrMatrix C = CsrMatrix::from_eigen(A);
    // uneven rows 1:2:3:...; rank 1 empty when P > 1
    std::vector<int> dist(P + 1, 0);
    {
      std::vector<double> w(P);
      double tot = 0;
      for (int p = 0; p < P; ++p) tot += (w[p] = (P > 1 && p == 1) ? 0.0 : p + 1.0);
      double acc = 0;
      for (int p = 0; p < P; ++p) {
        acc += w[p];
        dist[p + 1] = static_cast<int>(std::lround(n * acc / tot));
      }
      dist[P] = n;
    }
    const DistCsr Dd = DistCsr::distribute(C, dist, MPI_COMM_WORLD);
    std::vector<double> bl(bv.begin() + dist[me], bv.begin() + dist[me + 1]);
    for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
      if (!lu_backend_available(be)) continue;
      SparseLU lu(be);
      lu.factorize(Dd);
      const auto x = lu.solve(bl);
      double loc = 0, gl = 0;
      for (int i = 0; i < Dd.local_rows; ++i) loc += std::pow(x[i] - xs(dist[me] + i), 2);
      MPI_Allreduce(&loc, &gl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double ex = std::sqrt(gl) / xs.norm();
      if (me == 0) std::printf("distributed %d ranks %-14s err %.2e residual %.2e (plain %.2e, %d steps)\n", P,
                               to_string(be), ex, lu.stats().rel_residual, lu.stats().rel_residual_before,
                               lu.stats().refine_steps);
      CHECK(ex < 1e-10 && lu.stats().rel_residual <= 1e-12, "distributed %s: error %g", to_string(be), ex);
    }
    // duplicates: refused; summed by sum_duplicates()
    DistCsr dup = Dd;
    {
      std::vector<int> rp(1, 0), cc;
      std::vector<double> vv;
      for (int i = 0; i < dup.local_rows; ++i) {
        for (int k = dup.rowptr[i]; k < dup.rowptr[i + 1]; ++k) {  // split every entry in two halves
          cc.push_back(dup.col[k]);
          vv.push_back(0.5 * dup.val[k]);
          cc.push_back(dup.col[k]);
          vv.push_back(0.5 * dup.val[k]);
        }
        rp.push_back(static_cast<int>(cc.size()));
      }
      dup.rowptr.swap(rp);
      dup.col.swap(cc);
      dup.val.swap(vv);
    }
    bool threw = false;
    try {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(dup);
    } catch (const axf::Error&) {
      threw = true;
    }
    CHECK(threw, "duplicate CSR entries were not refused");
    dup.sum_duplicates();
    {
      SparseLU lu(LUBackend::EigenLU);
      lu.factorize(dup);
      lu.solve(bl);
      CHECK(lu.stats().rel_residual <= 1e-12, "after sum_duplicates: residual %g", lu.stats().rel_residual);
    }
    // numerically singular (row 3 = row 1 + row 2 of a small dense-ish matrix): every rank must throw
    {
      SpMat S0 = oseen_like(6, 2.0, 0);
      Eigen::SparseMatrix<double, Eigen::RowMajor> R(S0);
      R.row(2) = R.row(0) + R.row(1);
      R.makeCompressed();
      R.prune(0.0);
      const CsrMatrix Cs = CsrMatrix::from_eigen(R);
      std::vector<int> ds(P + 1, 0);
      for (int p = 0; p <= P; ++p) ds[p] = static_cast<int>(static_cast<long>(Cs.n) * p / P);
      const DistCsr Ds = DistCsr::distribute(Cs, ds, MPI_COMM_WORLD);
      std::vector<double> bs(Ds.local_rows, 1.0);
      for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
        if (!lu_backend_available(be)) continue;
        int mine = 0;
        try {
          SparseLU lu(be);
          lu.factorize(Ds);
          lu.solve(bs);  // a backend that factors a singular matrix must still miss the residual target
        } catch (const axf::Error&) {
          mine = 1;
        }
        int all = 0;
        MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        CHECK(all == 1, "%s: a singular matrix did not fail on every rank", to_string(be));
      }
    }
  }

  std::printf(failures ? "test_fluids_sparse_lu: %d FAILURES\n" : "test_fluids_sparse_lu: OK\n", failures);
  MPI_Finalize();
  return failures ? 1 : 0;
}
