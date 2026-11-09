// Steady incompressible Navier–Stokes, planar and axisymmetric with swirl, Taylor–Hood (beta; formulation:
// docs/guides/fluids.md).
//
//   rho (u.grad)u - div sigma_form(u, p) = f,   div u = 0,   rho > 0 constant, mu(x) > 0
//
// DYNAMIC units: the viscosity coefficient of StokesOptions is the dynamic viscosity mu (nu := mu in the Stokes
// assembly), the force a force density, p the physical pressure.  The linear part (viscous form, pressure coupling,
// Dirichlet/axis elimination, traction loads, nullspace detection and treatment) is the Stokes assembly,
// reused unchanged; the convective term is cpp/include/axionfem/fluids/convection.hpp (one kernel).
//
// Nonlinear solver (our own driver, nonlinear_driver.hpp):
// a TRUE Stokes start (no convective term in matrix or residual), Picard (Oseen) steps, Newton with the exact
// Jacobian and an optional Armijo line search on the dual-norm merit (lambda = 1, 1/2, ..., 1/64; 1/64 accepted
// without decrease and counted), Re continuation as viscosity scale factors, pseudo-transient continuation
// (J + rho/dt M_u, switched evolution relaxation, release to Newton).  Every residual is the discrete DUAL norm
// ||R||_*^2 = R_u^T M_u^-1 R_u + R_p^T M_p^-1 R_p over the rows of the system actually solved; the gate and the stop
// are the steady relative residual ||R(x)||_* / ||R(x_lift)||_*.  Direct sparse LU only.
//
// Only tested features are offered: Dirichlet, axis, traction and do-nothing boundaries; slip, symmetry, periodic,
// a pressure point, grad-div, MINRES and swirl-free axisymmetry are REFUSED (with a message saying what to use).
#pragma once

#include <mfem.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axionfem/fluids/convection.hpp"
#include "axionfem/fluids/nonlinear_driver.hpp"
#include "axionfem/fluids/mhd.hpp"
#include "axionfem/fluids/sparse_lu.hpp"
#include "axionfem/fluids/stokes.hpp"

namespace axf::fluids {

// TransientNewton / TransientSemiImplicit solve ONE time step of the system the steady driver solves plus the
// time term set by set_time_term (used by TransientNavierStokes, transient_ns.hpp), starting from the state set by
// set_state (Dirichlet DOFs lifted to the current data); no Stokes start.
enum class NonlinearStrategy { Newton, PicardOnly, PseudoTransient, TransientNewton, TransientSemiImplicit };
const char* to_string(NonlinearStrategy s);

struct NonlinearOptions {
  NonlinearStrategy strategy = NonlinearStrategy::Newton;
  int n_picard = 3;                 // Picard steps after the Stokes start (Newton strategy, first stage)
  bool line_search = true;          // Armijo on the dual-norm merit
  double armijo_c1 = 1e-4;
  int line_search_halvings = 6;     // minimum step 2^-6 = 1/64, accepted (and counted) without decrease
  double newton_rtol = 1e-12;       // on the STEADY relative residual (dual norm)
  int newton_maxit = 30;            // per continuation stage / after the PTC release
  int picard_maxit = 200;           // PicardOnly
  double ptc_dt0 = 0.05, ptc_release_dt = 1e12;
  int ptc_maxit = 200;
  std::vector<double> viscosity_scales;  // continuation schedule s_1..s_m (mu_j = s_j mu); empty = one stage (1)
};

struct NavierStokesOptions {
  StokesOptions stokes;             // nu := mu (dynamic); solver must be Direct, grad_div 0, axisymmetric with swirl
  double rho = 1.0;                 // constant density (> 0)
  ConvectiveForm convection = ConvectiveForm::Standard;
  int quad_order_convection = 0;    // 0 = exact minimum: 3p - 1 planar, 3p axisymmetric
  int quad_order_mass = 0;          // 0 = 2p + 2 (the mass matrices of the dual norm and of PTC)
  NonlinearOptions nonlinear;
  // inductionless MHD (mhd.hpp). Requires geometry Planar25D; adds the potential block and the Lorentz force to
  // the residual and Jacobian; rho = 0 is then allowed (a Stokes-like operator with the Lorentz coupling).
  const MhdOptions* mhd = nullptr;
};

// R over ALL velocity true DOFs (no elimination) and the pressure rows, with the SOLVED operators (viscous form and
// rules, convective form, the final stage's viscosity scale): for volume-integral (Babuška–Miller) functionals.
struct FullResidual {
  std::vector<double> r_u, r_p;      // r_u ordered byVDIM like the velocity space
  std::vector<double> node_xy;       // coordinates of the scalar velocity nodes (2 per node)
  std::vector<std::pair<int, std::vector<long>>> attribute_nodes;  // nodes on each boundary attribute (corners twice)
};

struct NavierStokesDiagnostics {
  StokesDiagnostics linear;         // from the Stokes assembly and the last LU
  std::vector<NonlinearStep> history;
  bool converged = false;
  int iterations = 0;               // steps after the Stokes start
  double relative_residual = -1;    // final steady relative residual
  double residual_lift = -1;        // ||R(x_lift)||_* at the final stage's viscosity
  int line_search_min_step_accepts = 0;
  std::string convection, strategy, failure;  // failure: why converged = false ("" if converged)
  std::vector<int> quad_orders;     // viscous, convection, load, boundary, mass
  double seconds = 0;
  std::vector<double> viscosity_scales;
  long lu_fallbacks = 0;            // direct solves repeated on another configuration (cumulative; 0 for steady solves)
  std::string lu_fallback_backend;  // the last fallback used ("strumpack-no-matching" | "eigen" | "")
};

class NavierStokesSolver {
 public:
  NavierStokesSolver(mfem::ParMesh& mesh, const NavierStokesOptions& opt, const StokesBoundary& bc);
  ~NavierStokesSolver();
  NavierStokesSolver(const NavierStokesSolver&) = delete;
  NavierStokesSolver& operator=(const NavierStokesSolver&) = delete;

  const NavierStokesDiagnostics& solve();  // never throws on non-convergence
  const NavierStokesDiagnostics& diagnostics() const { return diag_; }
  const NavierStokesOptions& options() const { return opt_; }
  // fields of a converged solve (throw otherwise); pressure mean-free (r-weighted) for an enclosed flow
  const mfem::ParGridFunction& velocity() const;
  const mfem::ParGridFunction& pressure() const;
  mfem::ParFiniteElementSpace& velocity_space();
  // MHD (options().mhd != nullptr): the potential of a converged solve (mean-free when it is defined up to a constant)
  // and the coupling diagnostics; the potential is refused (throws) for a non-MHD solver.
  const mfem::GridFunction& potential() const;
  const MhdDiagnostics& mhd_diagnostics() const;

  // ---- Time-stepper interface: internal, used by TransientNavierStokes (transient_ns.hpp) and its Python binding;
  // not a stable API. All on true-DOF vectors, MPI-safe.
  // data at the current time: re-evaluate force/traction/Dirichlet coefficients (StokesSolver::refresh_data) and take
  // the new lift, load and continuity right-hand side
  void refresh_data();
  // momentum residual += zero_ess(c M u - hist) with M the (consistent, r-weighted) velocity mass; c = rho alpha_0/dt
  // (NOT multiplied here), hist a full velocity vector; the Jacobian gets + c M. c = 0 clears the term.
  void set_time_term(double c, const mfem::Vector& hist);
  // advecting velocity of the semi-implicit strategy (full true-DOF vector)
  void set_advecting(const mfem::Vector& w);
  // current state (true DOFs; pressure internal, not normalized); set_state replaces Dirichlet DOFs by the lift
  void set_state(const mfem::Vector& u, const mfem::Vector& p, bool lift_boundary = true);
  const mfem::Vector& state_u() const;
  const mfem::Vector& state_p() const;
  const mfem::Vector& lift() const;
  mfem::ParFiniteElementSpace& pressure_space();
  // subtract the r-weighted mean of p when the flow is enclosed (the reported, mean-free pressure)
  void normalize_pressure(mfem::Vector& p) const;
  // fields of the current state regardless of the steady convergence flag (a time step's own gate is diagnostics())
  const mfem::ParGridFunction& velocity_field() const;
  const mfem::ParGridFunction& pressure_field() const;
  int velocity_size() const;
  int pressure_size() const;
  void mass_mult(const mfem::Vector& x, mfem::Vector& y) const;       // y = M_u x (full, r-weighted in axisymmetry)
  void mass_mult_time(const mfem::Vector& x, mfem::Vector& y) const;  // y = the time mass times x (M_u)
  void zero_essential(mfem::Vector& v) const;
  // sqrt(x^T M x) with the (r-weighted in axisymmetry) velocity mass M
  double mass_norm(const mfem::Vector& x) const;
  // discrete L2 (Leray) projection of u_target onto the discretely divergence-free fields with the nodal Dirichlet
  // data; sets the state.
  bool project_divergence_free(const mfem::Vector& u_target);

  // Taylor-test / force API on the free DOFs, one rank only
  std::vector<double> state() const;
  std::vector<double> residual(const std::vector<double>& x_free, double viscosity_scale = 1.0) const;
  std::vector<double> jacobian_action(const std::vector<double>& x_free, const std::vector<double>& d,
                                      double viscosity_scale = 1.0) const;
  DistCsr jacobian(const std::vector<double>& x_free, double viscosity_scale = 1.0) const;
  double dual_norm(const std::vector<double>& R) const;
  std::pair<long, long> layout() const;
  FullResidual full_residual() const;
  // Babuška–Miller reaction with the convective term of the solved form; dynamic units
  Reaction reaction(const std::vector<int>& body, const std::vector<double>& center = {},
                    ViscousForm residual_form = ViscousForm::SymmetricStress) const;

 struct Impl;  // opaque (navier_stokes_internal.hpp); public only so that BoussinesqSolver's solver system can name it

 private:
  friend class BoussinesqSolver;
  bool lin_solve_(const mfem::HypreParMatrix& Juu, const mfem::Vector& Ru, const mfem::Vector& Rp, mfem::Vector& du,
                  mfem::Vector& dp, int& refine);
  std::unique_ptr<Impl> impl_;
  NavierStokesOptions opt_;
  NavierStokesDiagnostics diag_;
};

}  // namespace axf::fluids
