#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace axf::detail {

// Numerical lift responses need not give a symmetric reaction matrix. In particular,
// a true residual on one lift may contaminate only one off-diagonal entry. Solving
// one triangle with LDLT then leaves a nonzero prescribed current on another loop.
struct BodyConstraintSolution {
  Eigen::VectorXd coefficients;
  Eigen::VectorXd residual;
  double backward_error;
  double asymmetry;
  double reciprocal_condition;
};

inline BodyConstraintSolution solve_body_constraints(const Eigen::MatrixXd& matrix,
                                                      const Eigen::VectorXd& rhs) {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols() || rhs.size() != matrix.rows() ||
      !matrix.allFinite() || !rhs.allFinite())
    throw std::runtime_error("invalid superconducting loop reaction matrix or target");
  const auto factor = matrix.fullPivLu();
  if (!factor.isInvertible())
    throw std::runtime_error("singular superconducting loop reaction matrix");
  BodyConstraintSolution out;
  out.coefficients = factor.solve(rhs);
  out.residual = matrix * out.coefficients - rhs;
  const double norm = matrix.cwiseAbs().rowwise().sum().maxCoeff();
  const double scale = norm * out.coefficients.lpNorm<Eigen::Infinity>() + rhs.lpNorm<Eigen::Infinity>();
  out.backward_error = scale > 0 ? out.residual.lpNorm<Eigen::Infinity>() / scale : 0.;
  out.asymmetry = (matrix - matrix.transpose()).norm() / matrix.norm();
  out.reciprocal_condition = factor.rcond();
  if (!out.coefficients.allFinite() || !std::isfinite(out.backward_error) || !std::isfinite(out.asymmetry) ||
      !std::isfinite(norm) || !std::isfinite(scale) || !std::isfinite(out.reciprocal_condition) ||
      out.backward_error > 1e-12)
    throw std::runtime_error("superconducting loop reaction solve has an excessive backward error");
  return out;
}

}  // namespace axf::detail
