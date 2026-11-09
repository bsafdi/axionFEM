// Steady Boussinesq natural convection, planar and axisymmetric (beta; docs/guides/fluids.md).
//
//   rho0 (u.grad)u - div sigma_form(u, p) = f_u + rho0 beta g (T - T_ref) e_up,   div u = 0,
//   rho0 c_p u.grad T - div(k grad T) = Q
//
// The core is DIMENSIONAL (rho0, c_p, beta g, T_ref, k(x), mu(x)); (Ra, Pr) appear only in the Python helper.  p is the
// pressure relative to the reference hydrostatic state.  Temperature is P2 (Taylor-Hood P2/P1 velocity-pressure; P1
// temperature is refused here).  The momentum part is the steady Navier-Stokes solver (NavierStokesSolver: assembly,
// convection kernel, dual norm, monolithic direct matrix), the energy part is the shared axf::transport::ScalarTransport with
// beta = rho0 c_p u_h (convective form, SUPG with the exact Jacobian optional).  ONE monolithic Newton solve on
// (u, p, T) with the exact coupled Jacobian, started from the Dirichlet lift, through the generic NonlinearDriver
// (Armijo line search optional); the continuation scales the buoyancy beta g (a schedule must end at 1).  Segregated,
// Picard, pseudo-transient and linear starts, conservative forms, tensor conductivity, reactions, P1 T, T-dependent
// properties and a transient solve are refused.  One MPI rank.
#pragma once

#include <mfem.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axionfem/fluids/navier_stokes.hpp"
#include "axionfem/transport/scalar_transport.hpp"

namespace axf::fluids {

struct BoussinesqOptions {
  BoussinesqOptions() { ns.nonlinear.n_picard = 0; }  // no Picard steps in the coupled solve
  NavierStokesOptions ns;                 // rho = rho0; nonlinear.{newton_rtol, newton_maxit, line_search} are used
  double cp = 1.0;
  double beta_g = 0.0;                    // beta * g (dimensional)
  bool t_ref_set = false;                 // T_ref is REQUIRED (no default)
  double t_ref = 0.0;
  int up_sign = 1;                        // +1: buoyancy along +e_up (gravity along -e_up); e_up = e_y (planar) / e_z
  transport::ScalarData k;                // thermal conductivity
  transport::BatchVec grad_k;             // grad k (needed for SUPG with a variable k)
  transport::ScalarData heat_source;      // Q
  int t_order = 2;
  bool supg = false;
  std::vector<double> buoyancy_scales;    // continuation s_1 .. s_m multiplying beta g; empty = {1}; must end at 1
  struct TBc {
    std::vector<int> attrs;
    transport::BcKind kind;
    transport::BcData data;
  };
  std::vector<TBc> t_bcs;
  std::string start = "lift";             // the only start
  bool transient = false;                 // refused
};

struct BoussinesqDiagnostics {
  bool converged = false;
  int iterations = 0;
  double relative_residual = -1, residual_lift = -1, seconds = 0;
  int line_search_min_step_accepts = 0;
  std::vector<NonlinearStep> history;
  std::string failure, start = "lift", strategy = "newton", tau_rule = "none", convection, energy_form = "convective";
  std::vector<double> buoyancy_scales;
  long n_velocity = 0, n_pressure = 0, n_temperature = 0;
  double lu_residual = -1;
  StokesDiagnostics linear;
  std::vector<int> quad_orders;           // viscous, convection, load, boundary, mass, transport
};

class BoussinesqSolver {
 public:
  BoussinesqSolver(mfem::ParMesh& mesh, const BoussinesqOptions& opt, const StokesBoundary& bc);
  ~BoussinesqSolver();
  BoussinesqSolver(const BoussinesqSolver&) = delete;
  BoussinesqSolver& operator=(const BoussinesqSolver&) = delete;

  const BoussinesqDiagnostics& solve();  // never throws on non-convergence
  const BoussinesqDiagnostics& diagnostics() const { return diag_; }
  const BoussinesqOptions& options() const { return opt_; }
  const mfem::ParGridFunction& velocity() const;
  const mfem::ParGridFunction& pressure() const;   // relative to the reference hydrostatic state; mean-free if enclosed
  const mfem::GridFunction& temperature() const;
  transport::ScalarTransport& transport() { return *tr_; }
  mfem::ParFiniteElementSpace& velocity_space();

  // free-DOF API (x_free = (u_free, p, T_free)), one rank
  std::vector<double> state() const;
  std::vector<double> residual(const std::vector<double>& x_free, double buoyancy_scale = 1.0) const;
  std::vector<double> jacobian_action(const std::vector<double>& x_free, const std::vector<double>& d,
                                      double buoyancy_scale = 1.0) const;
  double dual_norm(const std::vector<double>& R) const;
  std::array<long, 3> layout() const;
  // r_T = R(T_h; phi_i) over EVERY temperature dof (unconstrained; the converged solution)
  std::vector<double> full_residual_T() const;
  std::vector<double> boundary_weights(const std::vector<int>& attrs) const { return tr_->boundary_weights(attrs); }

 struct Impl;  // opaque (boussinesq.cpp)

 private:
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<NavierStokesSolver> ns_;
  std::unique_ptr<transport::ScalarTransport> tr_;
  BoussinesqOptions opt_;
  BoussinesqDiagnostics diag_;
};

}  // namespace axf::fluids
