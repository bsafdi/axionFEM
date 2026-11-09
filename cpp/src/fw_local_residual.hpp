// Private: the order-2 full-wave error indicator's local problem on one element (FwErrorIndicator::EvaluateResidual,
// model3d_fw_indicator.cpp; docs/physics/fullwave_error_indicator.md, "On the complement of the local gradients").
// Eigen only, so that it is compiled in every configuration (the 3D engine needs MPI; its unit test,
// cpp/tests/test_fw_indicator_local.cpp, does not).
#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace axf {

// The local error x in the free ND_q dofs of one element: (nu CC - omega^2 eps MM) x = b tested against, and solved
// for, x in range(Q), Q = I - P, P = Gr S^-1 Gr^T MM (the MM-orthogonal complement of the local gradients Gr, free dofs
// x m'). It is solved in the basis Q C, C the unit vectors of COMPLEMENT (positions in the free dofs: those outside a
// tree of Gr, fw_gradient_complement): the curl block is C^T CC C (CC Q = CC), the mass block C^T (MM - MG S^-1 MG^T) C
// with MG = MM Gr, the load C^T Q^T b, and x = Q C y. range(Q) does not depend on C, so neither does x (posed on
// span(C) with the unprojected mass, the result would depend on the tree, which round-off picks, and could be nearly
// resonant). On range(Q) the first local
// resonance lies at k h_max >= 1.0-1.7 on reasonable shapes (h_max the longest edge), at 9.7 or more without conductor
// faces or edges, but from about 0.27 on a flat sliver at a perfect-conductor edge: no regularization is applied.
// fw_local_kh_warning flags elements outside that range, as a heuristic (it does not detect a resonance).
struct FwLocalResidual {
  Eigen::VectorXcd x;          // the local error in the free dofs
  double magnetic = 0.0;       // |nu| x^H CC x
  double electric = 0.0;       // omega^2 |eps| x^H MM x
  std::complex<double> form;   // nu x^T CC x - omega^2 eps x^T MM x
};

// MMf, CCf: the free dofs' mass and curl-curl matrices; Gr: the local gradients; MG = MMf Gr; S: the Cholesky factor of
// Gr^T MMf Gr (unused without gradients); b: the load in the free dofs.
FwLocalResidual fw_local_residual(const Eigen::MatrixXd& MMf, const Eigen::MatrixXd& CCf, const Eigen::MatrixXd& Gr,
                                  const Eigen::MatrixXd& MG, const Eigen::LLT<Eigen::MatrixXd>& S,
                                  const std::vector<int>& complement, const Eigen::VectorXcd& b,
                                  std::complex<double> nu, std::complex<double> eps, double w2);

// the free dofs outside a tree of Gr (the first rank() pivots of the column-pivoted QR of Gr^T), as positions
std::vector<int> fw_gradient_complement(const Eigen::MatrixXd& Gr);

// A tetrahedron's size and shape from its four vertices: the longest edge h_max, the shortest edge and the smallest
// height (3 V over the largest face's area).
struct FwLocalShape {
  double h_max = 0.0;
  double min_edge = 0.0;
  double min_height = 0.0;
};
FwLocalShape fw_local_shape(const double (&vertices)[4][3]);

// The heuristic warning on the local problems' validated range (docs/physics/fullwave_error_indicator.md): bit 0 when
// |k| h_max > 1 (past the first local resonance's lower bound on reasonable shapes, 1.0-1.7; in a conductor, outside
// the validated |k| h <~ 1), bit 1 when the element is sliver-like (smallest height below 0.2 of its shortest edge:
// a flat test sliver, 0.042, against 0.58-0.97 for the reference, Kuhn, 5:1 and needle shapes) and
// |k| h_max > 0.2 (a flat sliver resonated from 0.27). k is the local wave number, |k| = omega sqrt(|eps| / |nu|). It
// is not a resonance detector: it flags where the documented bounds do not cover the element.
constexpr double kFwLocalKhBound = 1.0;
constexpr double kFwLocalSliverRatio = 0.2;
constexpr double kFwLocalSliverKhBound = 0.2;
int fw_local_kh_warning(const FwLocalShape& shape, double k);

}  // namespace axf
