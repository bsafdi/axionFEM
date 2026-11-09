// Transient incompressible Navier-Stokes, planar and axisymmetric with swirl: variable-step BDF1/2, Newton per step or
// semi-implicit, Milne error estimate and step controller, projected initial data (beta; formulation:
// docs/guides/fluids.md).
//
//   rho (du/dt + (u.grad)u) - div sigma_form(u, p) = f,   div u = 0,   u(t_0) = u_0     (DYNAMIC units)
//
// One time step solves the steady Navier-Stokes system plus the BDF mass term through NavierStokesSolver (convection
// kernel, residual, Jacobians and driver reused, not forked). History states are full true-DOF vectors (Dirichlet values
// included); the data hook is called with the three data times before every step.
#pragma once

#include <mfem.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "axionfem/fluids/navier_stokes.hpp"

namespace axf::fluids {

enum class TimeScheme { Newton, SemiImplicit };

struct TransientOptions {
  NavierStokesOptions ns;     // nonlinear.strategy is set by the scheme; the steady strategy fields are ignored
  int max_order = 2;          // BDF order 1 or 2
  TimeScheme scheme = TimeScheme::Newton;
  double safety = 0.9, ratio_cap = 2.0, ratio_floor = 0.2;  // step controller: safety factor, step-ratio cap and floor
};

struct StepResult {
  bool converged = false;
  double relative_residual = -1;
  int iterations = 0;
  int lu_refine = 0;
  double t = 0, dt = 0;
  int order = 0;
  bool has_estimate = false;
  double est_norm = -1;       // the REPORTED norm of the estimate
  double ctrl_norm = -1;      // the norm the controller compares with rtol S_u
  std::string failure;
};

struct RunStats {
  bool all_converged = true;
  double max_relative_residual = 0;
  long newton_iterations = 0;
  long solves = 0;
  long lu_fallbacks = 0;       // direct solves that needed another configuration (visible; see navier_stokes.hpp)
  std::string lu_fallback_backend;
  std::string failure;
};

struct AdaptiveResult {
  std::vector<double> times;  // t_0 ... t_N (accepted)
  std::vector<int> orders;
  int rejected = 0, failed_solves = 0;
  double dt_max_ratio = 1.0, dt_min_ratio = 1.0, dt_max = 0, dt_min = 0;
  bool reached = false;
};

class TransientNavierStokes {
 public:
  // hook(t_force, t_dirichlet, t_traction): the caller updates its time-dependent coefficients (and retabulates)
  using DataHook = std::function<void(double, double, double)>;
  TransientNavierStokes(mfem::ParMesh& mesh, const TransientOptions& opt, const StokesBoundary& bc, DataHook hook);
  ~TransientNavierStokes();
  TransientNavierStokes(const TransientNavierStokes&) = delete;
  TransientNavierStokes& operator=(const TransientNavierStokes&) = delete;

  NavierStokesSolver& core() { return *ns_; }
  const TransientOptions& options() const { return opt_; }

  // nodal P2 interpolation of a vector coefficient (true-DOF vector)
  mfem::Vector interpolate(mfem::VectorCoefficient& f);
  // state at t0: the nodal interpolant of u0, projected (Leray) unless project = false; returns false
  // if the projection solve failed
  bool initialize(double t0, mfem::VectorCoefficient& u0, bool project);
  // replace the history (newest first) by given states (single-step hook); pressure zero
  void set_history(const std::vector<double>& times, const std::vector<mfem::Vector>& states);

  // one trial step of size dt (order <= force_order if > 0); the state is not committed
  StepResult step_trial(double dt, int force_order = 0);
  void commit();
  StepResult advance(double dt);  // step_trial + commit when converged

  AdaptiveResult run_adaptive(double T, double rtol, double S_u, double h0);

  double time() const;
  int levels() const { return static_cast<int>(hist_.size()); }
  const mfem::Vector& state_u() const { return hist_.front().u; }
  const mfem::Vector& state_p() const { return hist_.front().p; }
  const mfem::Vector& trial_u() const { return trial_.u; }
  const mfem::Vector& trial_p() const { return trial_.p; }
  const mfem::Vector& estimate() const { return est_; }
  const RunStats& stats() const { return stats_; }
  void reset_stats() { stats_ = RunStats(); }
  // BDF coefficients alpha_0..alpha_k and the extrapolation (public for the unit tests)
  static std::vector<double> bdf_coefficients(int order, double dt, double dt_prev);

 private:
  struct Level {
    double t = 0;
    mfem::Vector u, p;
  };
  mfem::ParMesh& mesh_;
  TransientOptions opt_;
  DataHook hook_;
  std::unique_ptr<NavierStokesSolver> ns_;
  std::vector<Level> hist_;  // newest first, at most 3
  Level trial_;
  mfem::Vector est_;
  RunStats stats_;
  void refresh(double tf, double tg, double ttr);
  mfem::Vector milne_estimate(int order, double tn, const mfem::Vector& un, bool& ok) const;
  double reported_norm(const mfem::Vector& e) const;
  double controller_norm(const mfem::Vector& e) const;
  StepResult step_impl(double dt, int force_order, bool want_estimate);
};

}  // namespace axf::fluids
