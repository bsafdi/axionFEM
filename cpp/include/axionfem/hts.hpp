// Axisymmetric critical-state (screening-current) model for HTS bulks and homogenized tape stacks.
//
// Formulation (docs/validation/hts.md): time-dependent A-formulation with the single scalar unknown
// A = A_phi.  Axisymmetric azimuthal currents carry no electrostatic potential, so the azimuthal
// electric field is exactly E = -dA/dt, and the designated HTS regions obey the E-J power law
//
//     E = E0 (|J|/Jc(B, T))^n sign(J)   <=>   J = Jc(B, T) (|E|/E0)^(1/n) sign(E),
//
// with Jc from a conductor's critical surface (conductor.hpp) times its fill factor (homogenized
// stack).  Everywhere else the medium is linear (vacuum and magnetic materials at their nominal mu_r;
// the caller warns that a nonlinear B-H law is not used here) and ideal superconductors stay exact
// masters (r A = psi) as in the static problem.  The weak form per radian is
//
//     int nu r B[A].B[v] dr dz + b_DtN(A, v) = int r (scale(t) J_src + J_hts(E)) v dr dz,
//
// discretized by backward Euler and solved by Newton per step.  The Newton unknown is the INCREMENT
// of A over the step, so E = -(increment)/dt carries no cancellation error even where the field is
// frozen; the Jacobian uses dJ/dE with |E| floored at e_floor*E0 (the residual is exact, and the
// derivative of Jc with respect to B is neglected: an inexact Newton with a line search).
//
// A background field enters as a reduced formulation: the total potential is A = A_red + scale(t) A_bg
// and only A_red is discretized (A_bg is a vacuum field, so it drops out of the stiffness term, as in
// the static engine).
//
// Integrals are per radian; reported energies, losses and moments are 3D values (J, A m^2).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"
#include "axionfem/conductor.hpp"

namespace axf::axi {

using json = nlohmann::json;

// A homogenized HTS material: its critical surface, power-law parameters and tape orientation.
struct HTSMaterial {
  std::string name;
  const JcLaw* jc = nullptr;
  double n = 25, E0 = 1e-4, T_op = 4.2;
  double fill = 1.0;                   // engineering Jc = fill * Jc(B, T_op)
  double normal_r = 0, normal_z = 1;   // unit normal of the tape plane in the (r, z) half-plane
  double e_floor = 1e-6;               // Jacobian floor: |E| >= e_floor * E0
  // Engineering critical current density [A/m^2] in the local field.
  double Jc_eng(double Br, double Bz) const;
  double J_of_E(double E, double Jc_e) const;   // regularized power law
  double dJdE(double E, double Jc_e) const;     // its derivative
  double psi_of_E(double E, double Jc_e) const;  // dissipation potential int_0^E J(e) de (convex)
};

struct HTSRole {
  bool master = false;     // ideal superconductor: DOFs tied to a master psi (field excluded)
  int master_id = -1;
  double nu = 0;           // reluctivity (m/H)
  double J = 0;            // source current density per unit drive scale [A/m^2]
  int hts = -1;            // index into the HTS material list (-1: not a critical-state region)
};

class CriticalState {
 public:
  CriticalState(const Space& sp, std::vector<HTSRole> roles, std::vector<HTSMaterial> mats,
                std::vector<Master> masters);
  void set_open_boundary(int outer_attr, double radius, int modes);
  // Background per unit drive scale: psi at every DOF (superconductor ties hold the total flux
  // function) plus A_phi and B at any point.
  void set_background(std::vector<double> psi_at_dof, std::function<double(double, double)> A_bg,
                      std::function<void(double, double, double&, double&)> B_bg);
  void set_newton(double tol, int max_it) { newton_tol_ = tol; max_newton_ = max_it; }
  // Assemble (constant stiffness, DtN, prolongation) and factorize the linear part.
  void setup();

  struct State {
    Eigen::VectorXd y;   // reduced unknowns (free DOFs, then the unknown master flux functions)
    Eigen::VectorXd u;   // A_red at all DOFs
    Eigen::VectorXd du;  // increment of u over the last step (0 for the initial state)
    Eigen::VectorXd dy;  // increment of y over the last step (initial guess for the next)
    double t = 0, scale = 0, dt = 0, dscale = 0;
    double W_J = 0;   // stored energy of the REDUCED field (3D); with a background drive the applied
                      // field's own energy is not included, and the discrete energy identity uses this one
    double loss_J = 0, work_J = 0;   // cumulative dissipation and input work (3D)
  };

  // Initial state at t0.  `penetrated` (field cooled): the linear solution with the HTS regions
  // transparent, i.e. no screening currents.  Otherwise (zero-field cooled) A_red = 0, which
  // requires scale0 = 0.
  State start(double t0, double scale0, bool penetrated);

  struct StepInfo {
    int newton = 0;
    double residual = 0, rel_residual = 0, update = 0;  // update: ||Newton step|| / ||increment||
    const char* converged_on = "none";                  // "residual" | "update" | "none"
    double loss_J = 0, work_J = 0, dW_J = 0;
    double numerical_dissipation_J = 0;  // pi dU^T K dU: the backward-Euler term in the balance
    double balance_rel = 0, be_balance_rel = 0;
    bool converged = true;
  };
  StepInfo advance(State& s, double t_new, double scale_new, int verbose = 0);

  // Total field (reduced + background) and A_phi at a point of element e.
  void field(const State& s, int e, const PointEval& pe, double& Br, double& Bz, double* A = nullptr) const;
  // Azimuthal electric field [V/m] of the last step at a point (0 outside HTS regions is NOT assumed:
  // E = -dA/dt everywhere).
  double E_at(const State& s, int e, const PointEval& pe) const;
  // Engineering current density [A/m^2] and, if asked, the local engineering Jc, in an HTS element.
  double J_at(const State& s, int e, const PointEval& pe, double* Jc_eng = nullptr) const;

  const std::vector<HTSRole>& roles() const { return roles_; }
  const std::vector<HTSMaterial>& materials() const { return mats_; }
  const std::vector<Master>& masters() const { return masters_; }
  double master_psi(const State& s, int m) const;
  int n_free() const { return nfree_; }

 private:
  void assemble_nonlinear(const State& prev, const Eigen::VectorXd& du, bool want_K, Eigen::VectorXd& g,
                          std::vector<Eigen::Triplet<double>>& trip, double dt, double dscale, double* loss,
                          double* work_bg, double* potential) const;
  Eigen::VectorXd u_of(const Eigen::VectorXd& y, double scale) const;
  const Space& sp_;
  std::vector<HTSRole> roles_;
  std::vector<HTSMaterial> mats_;
  std::vector<Master> masters_;
  std::vector<double> psi_bg_;
  std::function<double(double, double)> A_bg_;
  std::function<void(double, double, double&, double&)> B_bg_;
  int outer_attr_ = 0, modes_ = 0;
  double Rb_ = 0;
  // DOF map: kind 0 unused, 1 free, 2 tied to a master, 3 fixed zero (axis)
  std::vector<int> kind_, index_, master_;
  std::vector<double> w_;
  std::vector<int> gidx_;
  int nfree_ = 0, ng_ = 0, nred_ = 0;
  Eigen::SparseMatrix<double> P_, K_, Kr_;
  Eigen::VectorXd u_fixed_, F_, Iext_;  // fixed-master part of u, source vector, master currents
  std::vector<int> hts_elems_;
  // background values at the quadrature points of the HTS elements (per unit scale)
  std::vector<double> q_Abg_, q_Brbg_, q_Bzbg_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> lin_;   // linear part (initial state)
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol_;  // Newton Jacobian (pattern reused)
  double newton_tol_ = 1e-4;
  int max_newton_ = 60;
  bool ready_ = false, chol_ready_ = false;
};

}  // namespace axf::axi
