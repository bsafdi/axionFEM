// Sparse direct solver utilities of the axisymmetric engine.
//   1. blocked_ldlt_solve (several right-hand sides in one pass over the factor) must give what Eigen's own
//      column-by-column solve gives -- it is what the exterior map's low-rank update uses. Both do the same
//      operations in the same order, so the results agree to rounding: bit for bit when the compiler keeps every
//      multiply and add separate, within a few ulp when it fuses some of them into FMA instructions differently
//      in the two loops (GCC with -march=<x86-64 with FMA>, as Cray's compiler wrappers pass; clang's default
//      -ffp-contract=on wherever FMA exists, e.g. Apple silicon).
//   2. the Woodbury form (A + V C V^T)^-1 that the engine applies must agree with a direct
//      factorization of the assembled A + V C V^T, which is the pair of paths the engine switches
//      between (AXF_DTN_DENSE).
//   3. SymSolver: the backend is reported, the threshold parses, and repeated factorizations of the
//      same pattern reuse the symbolic analysis.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

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

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

// 2D 5-point Laplacian on a g x g grid, shifted to be positive definite
static SpMat laplacian(int g, double shift = 0.0) {
  const int n = g * g;
  std::vector<Trip> t;
  auto id = [&](int i, int j) { return i * g + j; };
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      const int k = id(i, j);
      t.emplace_back(k, k, 4.0 + shift);
      if (i > 0) t.emplace_back(k, id(i - 1, j), -1.0);
      if (i + 1 < g) t.emplace_back(k, id(i + 1, j), -1.0);
      if (j > 0) t.emplace_back(k, id(i, j - 1), -1.0);
      if (j + 1 < g) t.emplace_back(k, id(i, j + 1), -1.0);
    }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

int main() {
  const int g = 24;
  SpMat A = laplacian(g, 0.3);
  const int n = static_cast<int>(A.rows());
  const int N = 12;  // rank of the low-rank term (the engine's dtn_modes)

  // ---------------------------------------------------------------- 1. blocked multi-RHS solve
  {
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, N);
    Eigen::SimplicialLDLT<SpMat> ldlt(A);
    CHECK(ldlt.info() == Eigen::Success, "LDLT of the test Laplacian failed");
    const Eigen::MatrixXd X0 = ldlt.solve(B);               // Eigen, column by column
    const Eigen::MatrixXd X1 = axf::blocked_ldlt_solve(ldlt, B);  // one pass over the factor
    // a few ulp of the largest entry: what differently fused multiply-adds can change, far below any
    // algorithmic difference (a wrong order of the triangular solves or a missed scaling is O(1))
    const double tol = 16.0 * std::numeric_limits<double>::epsilon() * X0.cwiseAbs().maxCoeff();
    double worst = 0;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs(X0(i, j) - X1(i, j)));
    CHECK(worst <= tol, "blocked_ldlt_solve differs from Eigen's solve by %.3e (tolerance %.3e, rounding only)",
          worst, tol);
    const double res = (A * X1 - B).norm() / B.norm();
    CHECK(res < 1e-12, "blocked solve residual %.3e", res);
    // a single column, and an empty block, must work too
    CHECK((axf::blocked_ldlt_solve(ldlt, Eigen::MatrixXd(B.col(0))) - X0.col(0)).cwiseAbs().maxCoeff() <= tol,
          "blocked_ldlt_solve is wrong for one column");
    CHECK(axf::blocked_ldlt_solve(ldlt, Eigen::MatrixXd(n, 0)).cols() == 0, "empty block");
  }

  // ---------------------------------------------------------------- 2. Woodbury vs assembled
  {
    // V: N columns supported on the "boundary" (the first 3g unknowns), as the exterior map's U is
    Eigen::MatrixXd Vd = Eigen::MatrixXd::Zero(n, N);
    for (int b = 0; b < 3 * g; ++b)
      for (int k = 0; k < N; ++k) Vd(b, k) = std::sin(0.7 * (b + 1) * (k + 1)) / (1.0 + k);
    Eigen::VectorXd c(N);
    for (int k = 0; k < N; ++k) c(k) = 0.5 + k;  // positive, as the DtN coefficients are
    const Eigen::MatrixXd Dd = Vd * c.asDiagonal() * Vd.transpose();
    const SpMat Afull = A + SpMat(Dd.sparseView());
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    Eigen::SimplicialLDLT<SpMat> direct(Afull);
    CHECK(direct.info() == Eigen::Success, "LDLT of the assembled system failed");
    const Eigen::VectorXd x_direct = direct.solve(b);

    // Woodbury, exactly as Cholesky::factorize/solve in cpp/src/axi_poloidal.cpp does it
    Eigen::SimplicialLDLT<SpMat> sparse_only(A);
    Eigen::MatrixXd M = Vd.transpose() * axf::blocked_ldlt_solve(sparse_only, Vd);
    M.diagonal() += c.cwiseInverse();
    M = 0.5 * (M + M.transpose().eval());
    Eigen::FullPivLU<Eigen::MatrixXd> cap(M);
    CHECK(cap.isInvertible(), "the capacitance matrix is singular");
    Eigen::VectorXd x = sparse_only.solve(b);
    x -= sparse_only.solve(Eigen::VectorXd(Vd * cap.solve(Eigen::VectorXd(Vd.transpose() * x))));

    const double rel = (x - x_direct).norm() / x_direct.norm();
    CHECK(rel < 1e-10, "Woodbury and the assembled factorization differ by %.3e", rel);
    const double res = (Afull * x - b).norm() / b.norm();
    CHECK(res < 1e-11, "Woodbury residual %.3e", res);
  }

  // ---------------------------------------------------------------- 3. SymSolver
  {
    setenv("AXF_STRUMPACK_MIN_DOFS", "1e18", 1);  // read once per process: "never STRUMPACK"
    CHECK(axf::SymSolver::threshold() > 1000000, "AXF_STRUMPACK_MIN_DOFS=1e18 parsed as %d (atoi gives 1)",
          axf::SymSolver::threshold());
    axf::SymSolver s;
    s.analyzePattern(A);
    CHECK(std::string(s.backend()) == "eigen-ldlt", "backend is '%s', expected eigen-ldlt", s.backend());
    s.factorize(A);
    CHECK(s.info() == Eigen::Success, "SymSolver factorization failed");
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x = s.solve(b);
    CHECK((A * x - b).norm() / b.norm() < 1e-12, "SymSolver residual too large");
    CHECK(s.symbolic_analyses() == 1, "symbolic analyses %d, expected 1", s.symbolic_analyses());
    // refactorizing the same pattern must not analyze again
    SpMat A2 = A;
    for (int k = 0; k < A2.nonZeros(); ++k) A2.valuePtr()[k] *= 1.01;
    s.factorize(A2);
    CHECK(s.info() == Eigen::Success, "refactorization failed");
    CHECK(s.symbolic_analyses() == 1, "refactorization re-analyzed the pattern (%d analyses)",
          s.symbolic_analyses());
    if (axf::have_strumpack()) {
      axf::SymSolver sp;
      sp.set_backend(1);
      sp.analyzePattern(A);
      CHECK(!sp.inertia().accurate, "symbolic-only factor cannot supply inertia");
      CHECK(std::string(sp.backend()) == "strumpack", "forced backend is '%s'", sp.backend());
      sp.factorize(A);
      CHECK(sp.info() == Eigen::Success, "STRUMPACK factorization failed");
      const auto pd_inertia = sp.inertia();
      const bool inertia_backend_unavailable = pd_inertia.status == "BACKEND_INERTIA_SELF_CHECK_FAILED";
      if (inertia_backend_unavailable)
        std::printf("(backend mixed-sign inertia self-check failed: all field-system counts withheld)\n");
      CHECK(inertia_backend_unavailable ||
            (pd_inertia.accurate && pd_inertia.negative == 0 && pd_inertia.zero == 0 && pd_inertia.positive == n),
            "positive-definite inertia failed: %s", pd_inertia.status.c_str());
      Eigen::VectorXd xs = sp.solve(b);
      CHECK((A * xs - b).norm() / b.norm() < 1e-11, "STRUMPACK residual too large");
      CHECK((xs - x).norm() / x.norm() < 1e-10, "STRUMPACK and Eigen disagree");
      sp.factorize(A2);  // same pattern: reuse the ordering and symbolic factorization
      CHECK(sp.symbolic_analyses() == 1 && sp.reused_factorizations() == 2,
            "STRUMPACK did not reuse the symbolic analysis (%d analyses, %d reused)",
            sp.symbolic_analyses(), sp.reused_factorizations());
      Eigen::MatrixXd Bm = Eigen::MatrixXd::Random(n, 3);
      CHECK((A2 * sp.solve(Bm) - Bm).norm() / Bm.norm() < 1e-10, "STRUMPACK multi-RHS solve is wrong");
      // A congruence scaling preserves signs. A forced row pivot must not be
      // mistaken for a valid symmetric inertia certificate.
      SpMat mixed(4, 4);
      std::vector<Trip> diagonal = {{0,0,-2.0}, {1,1,3.0}, {2,2,-5.0}, {3,3,7.0}};
      mixed.setFromTriplets(diagonal.begin(), diagonal.end());
      sp.compute(mixed);
      const auto signs = sp.inertia();
      CHECK(inertia_backend_unavailable ?
              (!signs.accurate && signs.negative == -1 && signs.zero == -1 && signs.positive == -1 &&
               signs.status == "BACKEND_INERTIA_SELF_CHECK_FAILED") :
              (signs.accurate && signs.negative == 2 && signs.zero == 0 && signs.positive == 2),
            "mixed-sign diagonal inertia failed: %s", signs.status.c_str());
      SpMat pivoted(2,2);
      std::vector<Trip> off_diagonal = {{0,1,1.0}, {1,0,1.0}};
      pivoted.setFromTriplets(off_diagonal.begin(), off_diagonal.end());
      sp.compute(pivoted);
      CHECK(sp.info() == Eigen::Success, "pivoted indefinite factorization failed");
      const auto invalid = sp.inertia();
      CHECK(!invalid.accurate && invalid.negative == -1 && invalid.zero == -1 && invalid.positive == -1,
            "pivoted inertia must withhold all counts: %s", invalid.status.c_str());
    } else {
      std::printf("(no STRUMPACK in this build: the backend comparison is skipped)\n");
    }
    // an unsymmetric matrix must be refused, not silently solved as its transpose
    SpMat U = A;
    U.coeffRef(1, n - 2) = 7.0;
    U.makeCompressed();
    bool threw = false;
    try {
      axf::SymSolver bad;
      bad.analyzePattern(U);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "SymSolver accepted an unsymmetric matrix");
  }

  std::printf(failures ? "test_solvers: %d FAILURES\n" : "test_solvers: ok\n", failures);
  return failures ? 1 : 0;
}
