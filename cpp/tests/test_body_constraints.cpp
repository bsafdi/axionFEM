#include "../src/body_constraints.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <limits>
#include <stdexcept>

int main() {
  // A small asymmetry in two independently computed lift responses. Both target
  // currents must hold, even when the exact continuum reaction matrix is symmetric.
  Eigen::Matrix2d matrix;
  matrix << 4.e8, 1.e8 + 1.5e5, 1.e8, 3.e8;
  Eigen::Vector2d exact(1.3e-7, 1.1e-6);
  const Eigen::Vector2d rhs = matrix * exact;
  const Eigen::Vector2d old = matrix.ldlt().solve(rhs);
  if ((matrix*old-rhs).lpNorm<Eigen::Infinity>() < .1) return 1;  // a symmetric LDLT solve must fail here
  const auto result = axf::detail::solve_body_constraints(matrix, rhs);
  if ((result.coefficients-exact).lpNorm<Eigen::Infinity>() > 1e-20 ||
      result.residual.lpNorm<Eigen::Infinity>() > 1e-10 || result.backward_error > 1e-14 ||
      result.asymmetry < 1e-5 || result.reciprocal_condition <= 0.) return 2;
  // Simultaneously permuting loop identities must not change their coefficients.
  Eigen::Matrix2d swap;
  swap << 0, 1, 1, 0;
  const auto permuted = axf::detail::solve_body_constraints(swap*matrix*swap, swap*rhs);
  if ((swap*permuted.coefficients-exact).norm() > 1e-20) return 3;
  // Scaling the current units preserves the coefficients and backward stability.
  const auto scaled = axf::detail::solve_body_constraints(1e-9*matrix, 1e-9*rhs);
  if ((scaled.coefficients-exact).norm() > 1e-20) return 4;
  Eigen::Matrix2d singular;
  singular << 1, 1, 1, 1;
  try { axf::detail::solve_body_constraints(singular, rhs); return 5; }
  catch (const std::runtime_error&) {}
  matrix(0,0) = std::numeric_limits<double>::quiet_NaN();
  try { axf::detail::solve_body_constraints(matrix, rhs); return 6; }
  catch (const std::runtime_error&) {}
  std::puts("nonsymmetric superconducting loop constraints: passed");
  return 0;
}
