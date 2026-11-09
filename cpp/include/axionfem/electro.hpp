// Axisymmetric electrostatics: capacitance matrices, parasitic and readout-gap capacitance.
//
// Solves div(eps grad phi) = 0 on the meridian half-plane with the weight r, on the H1 space of the
// magnetoquasistatic engine (axi::Space).  phi is a scalar, so (unlike the poloidal flux function
// psi = r A_phi) there is no condition on the symmetry axis -- the weight r vanishes there by itself --
// and the conductor ties carry no 1/r factor.
//
//  * Conductors (the connected bodies of `superconductor` and `conductor` materials, zero-thickness
//    sheets included) are equipotentials: every degree of freedom of a body is tied to one master
//    value, which is either a prescribed potential or follows from a prescribed net charge (a floating
//    conductor has zero net charge).
//  * The exterior of the outer semicircle is an exact Dirichlet-to-Neumann map for the Laplace
//    potential: phi = sum_{n>=0} a_n (Rb/rho)^{n+1} P_n(cos theta), so the n = 0 multipole carries the
//    net charge to infinity.  Alternatively the outer sphere is grounded (phi = 0 on it).
//  * A body may instead be declared `insulating`: its elements are left out of the domain and its
//    surface carries the natural condition n.D = 0.  This models the readout gap, where the two
//    terminal faces of ONE conductor body are driven against each other (docs/design_format.md).
//
// Energies are 3D values (J); the assembled form is per radian, W = 2 pi * (1/2) phi^T K phi, and the
// charge of body k is the reaction Q_k = 2 pi sum_{i tied to k} (K phi)_i.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"

namespace axf::axi {

// Role of one element in the electrostatic problem.
struct ElectroRole {
  int body = -1;            // >= 0: inside conductor body `body` (equipotential, element not assembled)
  bool insulating = false;  // body left out of the domain with the natural condition (no ties)
  double eps = 0;           // permittivity (F/m) of a dielectric element
};

// Boundary condition of one conductor.
struct Terminal {
  bool prescribed = false;  // potential given; otherwise floating with the prescribed net charge
  double V = 0;             // volts
  double charge = 0;        // coulombs (floating conductors: 0)
};

class ElectroProblem {
 public:
  // `roles` has one entry per element; conductors are numbered 0..nbody-1.
  // `sheet_body_by_attr` maps the boundary attribute of a zero-thickness conductor sheet to its body.
  // `extra_dofs` ties further degrees of freedom to a body (readout terminal faces), as (body, dofs).
  ElectroProblem(const Space& sp, std::vector<ElectroRole> roles, int nbody,
                 std::map<int, int> sheet_body_by_attr = {},
                 std::vector<std::pair<int, std::vector<int>>> extra_dofs = {});

  // Exterior treatment.  grounded: phi = 0 on the outer semicircle; otherwise the exact Laplace DtN
  // map with `modes` multipoles.  eps_out is the permittivity just outside the boundary.
  void set_outer(int outer_attr, double radius, double eps_out, bool grounded, int modes);
  // Conductor names, used in error messages only.
  void set_names(std::vector<std::string> names) { names_ = std::move(names); }

  struct Solution {
    Eigen::VectorXd phi;                        // potential at all DOFs (V)
    std::vector<double> V, Q;                   // per body: potential (V), net charge (C) from reactions
    std::vector<double> Q_gauss;                // per body: charge from the surface flux integral (C)
    double energy = 0;                          // stored energy (J), from the assembled form
    double energy_field = 0;                    // stored energy (J), by quadrature (independent path)
    double charge_exterior = 0;                 // net charge seen by the exterior map (C, open boundary)
    double charge_exterior_flux = 0;            // net charge from -int eps grad(phi).n dA on the sphere
    double residual = 0;                        // relative residual of the linear solve
  };

  // Maxwell capacitance matrix C(i, j) = Q_i for V_j = 1 V and every other conductor at 0 V.
  // One factorization and one solve per conductor; `cols` receives the per-column solutions.
  Eigen::MatrixXd capacitance(std::vector<Solution>* cols = nullptr);
  // One solve with mixed prescribed/floating conditions.
  Solution solve(const std::vector<Terminal>& t);

  // Stored energy (J) of an arbitrary potential field, by quadrature over the elements plus the
  // exterior multipole energy: an independent path to 2 pi * (1/2) phi^T K phi.
  double energy_quadrature(const Eigen::VectorXd& phi) const;
  // Per-body charge (C) from int eps (grad phi).n dA over the conductor surfaces (Gauss's law).
  std::vector<double> charges_gauss(const Eigen::VectorXd& phi) const;
  // Net charge (C) implied by the n = 0 multipole of the exterior map (0 for a grounded sphere).
  // NOTE: this is an algebraic *identity*, not a discretization check -- see exterior_charge_flux.
  // 1^T K phi = eps_out p_0 exactly (the stiffness annihilates the constant and 1^T U = 2 Rb delta_n0
  // on a sphere), and 4 pi eps_out Rb a_0 = 2 pi eps_out p_0, so the sum of the reaction charges equals
  // the monopole up to the solver residual and the geometric error of the boundary arc.
  double exterior_charge(const Eigen::VectorXd& phi) const;
  // Net charge (C) from the *strong* flux integral -int eps grad(phi).n dA over the outer semicircle,
  // evaluated from the interior element gradients with n outward.  The DtN map imposes the exterior
  // Neumann data only weakly, so the difference from exterior_charge() is a genuine discretization
  // error that falls under h/p refinement.  Zero when there is no exterior map.
  double exterior_charge_flux(const Eigen::VectorXd& phi) const;

  int nbody() const { return nbody_; }
  const nlohmann::json& diagnostics() const { return diag_; }
  // Multipole coefficients a_n (V) of the exterior expansion.
  Eigen::VectorXd multipoles(const Eigen::VectorXd& phi) const;

 private:
  void setup();
  void fill(Solution& s, const Eigen::VectorXd& phi) const;
  std::string bname(int b) const;

  const Space& sp_;
  std::vector<ElectroRole> roles_;
  int nbody_ = 0;
  std::vector<std::string> names_;
  std::map<int, int> sheet_body_;
  std::vector<std::pair<int, std::vector<int>>> extra_;
  int outer_attr_ = 0, modes_ = 0;
  double Rb_ = 0, eps_out_ = 0;
  bool grounded_ = false;
  bool ready_ = false;
  nlohmann::json diag_;

  // DOF classification: 0 unused, 1 free, 2 tied to a body, 3 fixed to zero.
  std::vector<int> kind_, index_, tie_;
  int nfree_ = 0, nred_ = 0;
  Eigen::SparseMatrix<double> P_, K_, A_;
  std::vector<int> bdofs_;
  Eigen::MatrixXd U_;    // int N_b P_n(cos theta) r dl / Rb over the outer semicircle
  Eigen::VectorXd coef_; // DtN coefficients (eps_out/Rb) (n+1)(2n+1)/2

  // Prepared surface quadrature for Gauss's law: one point per (face, side).
  struct FluxPt {
    int e = 0;        // dielectric element
    PointEval pe;     // prepared point
    double w = 0;     // 2 pi r |dl/dxi| weight times the quadrature weight
    double nr = 0, nz = 0;  // unit normal pointing from the conductor into the dielectric
    double eps = 0;
    int body = 0;
  };
  std::vector<FluxPt> flux_;
  std::vector<FluxPt> oflux_;  // the same on the outer semicircle, n outward (exterior flux check)
};

}  // namespace axf::axi
