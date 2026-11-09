// 1D porous electrode, two coupled potentials (beta; formulation: docs/guides/chemistry.md).
//
// x in [0, l]: solid potential phi1 (conductivity sigma), electrolyte potential phi2 (conductivity kappa), volumetric
// exchange j = a_v i(eta), eta = phi1 - phi2 - E_eq, i(eta) linearized Butler-Volmer (the only kinetics verified;
// set_exchange refuses bv and Tafel):
//     -(sigma phi1')' + j = 0,   -(kappa phi2')' - j = 0,
// x = 0 current collector: phi1 = 0 (grounded), i2 = -kappa phi2' = 0;  x = l separator: i1 = -sigma phi1' = 0 and
// i2 = I (galvanostatic) or phi2 = -V (potentiostatic).  Weak form on Lagrange P1/P2 for both fields, every term at
// the declared quadrature degree; Newton with the criterion of the 2D solver (a linear exchange converges in one step and
// stagnates on the next); sparse LU.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>

#include "axionfem/chem/current_distribution.hpp"

namespace axf::chem {

class PorousElectrode1D {
 public:
  PorousElectrode1D(double length, int n_elements, int order, int degree);
  ~PorousElectrode1D();
  PorousElectrode1D(const PorousElectrode1D&) = delete;
  PorousElectrode1D& operator=(const PorousElectrode1D&) = delete;

  void set_conductivities(double sigma, double kappa);
  void set_exchange(double a_v, const Reaction& r);
  void set_galvanostatic(double I);    // A/m^2, the cell current through the separator
  void set_potentiostatic(double V);   // phi1(0) - phi2(l) = V

  SolveReport solve(const NewtonOptions& opt);

  int order() const { return order_; }
  int degree() const { return degree_; }
  int num_elements() const;
  int num_dofs() const;               // per field
  Eigen::VectorXd state() const;      // [phi1 dofs, phi2 dofs]
  Eigen::VectorXd node_x() const;     // dof coordinates (per field)
  // raw data: per element quadrature at `degree`: x, w (with the Jacobian), phi1, phi1', phi2, phi2'
  Eigen::MatrixXd quadrature(int degree) const;   // n x 6
  // A z - b over both fields, without the Dirichlet elimination (the reaction rows at the collector/separator)
  Eigen::VectorXd residual_full() const;
  int collector_row() const;    // phi1 at x = 0
  int separator_row() const;    // phi2 at x = l
  double abs_action_sum() const;  // sum over all rows of |A| |z| (the identities' round-off scale)
  double h() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  int order_, degree_;
};

}  // namespace axf::chem
