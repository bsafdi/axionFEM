// Generic steady nonlinear driver: the Newton / Picard step of the steady Navier-Stokes solver with the Armijo line
// search on the discrete dual-norm merit and the continuation-stage loop, shared by NavierStokesSolver::solve() and the
// Boussinesq solver (the SAME code).  The driver owns no physics: a NonlinearSystem supplies the residual, its dual
// norm and the linear solve for a step; the state x and the residual R are single vectors in the system's own layout
// (Navier-Stokes: (u, p); Boussinesq: (u, p, T)).  The step uses elementwise vector updates only, in a fixed order, so
// that the Navier-Stokes results do not depend on which system drives them.
#pragma once

#include <mfem.hpp>

#include "axionfem/core.hpp"

#include <string>
#include <vector>

namespace axf::fluids {

struct NonlinearStep {
  std::string kind;                 // stokes | picard | newton | ptc
  double residual = -1;             // steady relative residual AFTER the step
  double step = 1;                  // lambda used
  double dt = 0;                    // PTC pseudo-time step used (0 otherwise)
  int stage = 0;
  double viscosity_scale = 1;       // the stage scale (Navier-Stokes: viscosity; Boussinesq: buoyancy beta g)
  int lu_refine = 0;
  double lu_residual = -1;          // relative residual of the step's linear solve; -1 if not recorded
};

enum class StepMode { Newton, Picard };

class NonlinearSystem {
 public:
  virtual ~NonlinearSystem() = default;
  // R(x) at the stage scale s (full layout; essential rows zero)
  virtual void residual(const mfem::Vector& x, double s, mfem::Vector& R) = 0;
  virtual double dual_norm(const mfem::Vector& R) = 0;
  // solve J(x; s, mode) d = -R; d has zero essential entries; false + failure on a failed solve
  virtual bool solve_linear(const mfem::Vector& x, const mfem::Vector& R, double s, StepMode mode, mfem::Vector& d,
                            int& refine, double& lu_residual, std::string& failure) = 0;
};

struct NewtonControl {
  bool line_search = true;
  double armijo_c1 = 1e-4;
  int line_search_halvings = 6;
  double rtol = 1e-12;
  int maxit = 30;
  int n_picard = 3;
};

class NonlinearDriver {
 public:
  NonlinearDriver(NonlinearSystem& sys, const NewtonControl& ctl, std::vector<NonlinearStep>& history,
                  int& iterations, int& min_step_accepts, std::string& failure)
      : sys_(sys), ctl_(ctl), hist_(history), iterations_(iterations), min_accepts_(min_step_accepts),
        failure_(failure) {}

  // one Newton (or Picard) step at scale s with the stage denominator den: updates x, R and r (relative residual)
  bool step(mfem::Vector& x, mfem::Vector& R, double s, double den, bool picard, int stage, double& r);
  // the continuation stages: for each scale, (n_picard Picard steps in the first), then Newton to ctl.rtol; `lift_norm`
  // gives ||R(x_lift)||_* at a scale; r, den in/out (den and residual_lift are updated per stage)
  template <class LiftNorm>
  bool run_stages(mfem::Vector& x, mfem::Vector& R, const std::vector<double>& scales, LiftNorm&& lift_norm,
                  double& den, double& residual_lift, double& r);
  std::string scale_name = "scale";  // wording of the stage scale in the failure text ("viscosity scale" for Navier-Stokes)
  bool fail(const std::string& why) {
    if (failure_.empty()) failure_ = why;
    return false;
  }

 private:
  NonlinearSystem& sys_;
  NewtonControl ctl_;
  std::vector<NonlinearStep>& hist_;
  int& iterations_;
  int& min_accepts_;
  std::string& failure_;
};

template <class LiftNorm>
bool NonlinearDriver::run_stages(mfem::Vector& x, mfem::Vector& R, const std::vector<double>& scales,
                                 LiftNorm&& lift_norm, double& den, double& residual_lift, double& r) {
  bool ok = true;
  for (size_t j = 0; ok && j < scales.size(); ++j) {
    const double s = scales[j];
    if (j > 0) {
      den = lift_norm(s);
      residual_lift = den;
      sys_.residual(x, s, R);
      r = sys_.dual_norm(R) / den;
    }
    if (j == 0)
      for (int k = 0; ok && k < ctl_.n_picard && r > ctl_.rtol; ++k) ok = step(x, R, s, den, true, 0, r);
    int k = 0;
    while (ok && r > ctl_.rtol && k < ctl_.maxit) {
      ok = step(x, R, s, den, false, static_cast<int>(j), r);
      ++k;
    }
    if (ok && r > ctl_.rtol)
      ok = fail(strcat("Newton budget exhausted in continuation stage ", j + 1, " of ", scales.size(), " (", scale_name,
                       " ", s, ")"));
  }
  return ok;
}

}  // namespace axf::fluids
