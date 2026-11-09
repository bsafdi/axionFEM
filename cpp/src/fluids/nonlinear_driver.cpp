// Generic steady nonlinear driver: see cpp/include/axionfem/fluids/nonlinear_driver.hpp.
#include "axionfem/fluids/nonlinear_driver.hpp"

#include <cmath>

namespace axf::fluids {

bool NonlinearDriver::step(mfem::Vector& x, mfem::Vector& R, double s, double den, bool picard, int stage,
                           double& r) {
  sys_.residual(x, s, R);
  const double m0 = sys_.dual_norm(R);
  StepMode mode = StepMode::Newton;
  if (picard) mode = StepMode::Picard;
  mfem::Vector d, tx, TR;
  int refine = 0;
  double lures = -1;
  std::string why;
  if (!sys_.solve_linear(x, R, s, mode, d, refine, lures, why)) {
    if (!why.empty() && failure_.empty()) failure_ = why;
    return fail(std::string("linear solve failed in a ") + (picard ? "Picard" : "Newton") + " step");
  }
  double lam = 1.0;
  const bool ls = !picard && ctl_.line_search;
  if (ls) {
    bool ok = false;
    for (int h = 0; h <= ctl_.line_search_halvings; ++h) {
      lam = std::ldexp(1.0, -h);
      tx = x;
      tx.Add(lam, d);
      sys_.residual(tx, s, TR);
      const double m = sys_.dual_norm(TR);
      if (std::isfinite(m) && m <= (1.0 - ctl_.armijo_c1 * lam) * m0) {
        ok = true;
        break;
      }
    }
    if (!ok) ++min_accepts_;  // lam = 2^-halvings accepted without sufficient decrease
    x = tx;
    R = TR;
  } else {
    x.Add(1.0, d);
    sys_.residual(x, s, R);
  }
  r = sys_.dual_norm(R) / den;
  NonlinearStep h;
  h.kind = picard ? "picard" : "newton";
  h.residual = r;
  h.step = lam;
  h.dt = 0.0;
  h.stage = stage;
  h.viscosity_scale = s;
  h.lu_refine = refine;
  h.lu_residual = lures;
  hist_.push_back(h);
  ++iterations_;
  if (!std::isfinite(r)) return fail("the residual is not finite (divergence)");
  return true;
}

}  // namespace axf::fluids
