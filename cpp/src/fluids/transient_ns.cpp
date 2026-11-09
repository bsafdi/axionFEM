// Transient Navier-Stokes time stepper: see include/axionfem/fluids/transient_ns.hpp.
#include "axionfem/fluids/transient_ns.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"

namespace axf::fluids {

std::vector<double> TransientNavierStokes::bdf_coefficients(int order, double dt, double dt_prev) {
  if (order == 1) return {1.0, -1.0};
  AXF_REQUIRE(order == 2, "only BDF1 and BDF2 are offered", "order");
  const double w = dt / dt_prev;
  return {(1 + 2 * w) / (1 + w), -(1 + w), w * w / (1 + w)};
}

TransientNavierStokes::TransientNavierStokes(mfem::ParMesh& mesh, const TransientOptions& opt,
                                             const StokesBoundary& bc, DataHook hook)
    : mesh_(mesh), opt_(opt), hook_(std::move(hook)) {
  AXF_REQUIRE(opt.max_order == 1 || opt.max_order == 2, "transient Navier-Stokes offers BDF order 1 or 2 only",
              "TransientOptions.max_order", "Higher orders and generalized-alpha are not offered.");
  AXF_REQUIRE(opt.ns.convection == ConvectiveForm::Standard,
              "transient Navier-Stokes: the skew convective form is not offered", "NavierStokesOptions.convection",
              "Use the standard form.");
  AXF_REQUIRE(opt.ns.stokes.pressure_order < 0 || opt.ns.stokes.pressure_order == opt.ns.stokes.velocity_order - 1,
              "transient Navier-Stokes: Taylor-Hood only", "StokesOptions.pressure_order");
  AXF_REQUIRE(hook_ != nullptr, "a data hook is required", "hook");
  TransientOptions o = opt;
  o.ns.nonlinear.strategy = opt.scheme == TimeScheme::Newton ? NonlinearStrategy::TransientNewton
                                                             : NonlinearStrategy::TransientSemiImplicit;
  opt_ = o;
  ns_.reset(new NavierStokesSolver(mesh, o.ns, bc));
}

TransientNavierStokes::~TransientNavierStokes() = default;

double TransientNavierStokes::time() const {
  AXF_REQUIRE(!hist_.empty(), "the solver is not initialized", "initialize");
  return hist_.front().t;
}

void TransientNavierStokes::refresh(double tf, double tg, double ttr) {
  hook_(tf, tg, ttr);
  ns_->refresh_data();
}

mfem::Vector TransientNavierStokes::interpolate(mfem::VectorCoefficient& f) {
  mfem::ParGridFunction g(&ns_->velocity_space());
  g.ProjectCoefficient(f);
  mfem::Vector t(ns_->velocity_size());
  g.GetTrueDofs(t);
  return t;
}

bool TransientNavierStokes::initialize(double t0, mfem::VectorCoefficient& u0, bool project) {
  refresh(t0, t0, t0);
  mfem::Vector uI = interpolate(u0);
  mfem::Vector p0(ns_->pressure_size());
  p0 = 0.0;
  bool ok = true;
  if (project) {
    ok = ns_->project_divergence_free(uI);
  } else {
    ns_->set_state(uI, p0, true);
  }
  Level L;
  L.t = t0;
  L.u = ns_->state_u();
  L.p = ns_->state_p();
  hist_.clear();
  hist_.push_back(L);
  return ok;
}

void TransientNavierStokes::set_history(const std::vector<double>& times, const std::vector<mfem::Vector>& states) {
  AXF_REQUIRE(!times.empty() && times.size() == states.size() && times.size() <= 3,
              "set_history needs 1 to 3 (time, state) pairs, newest first", "set_history");
  hist_.clear();
  for (size_t i = 0; i < times.size(); ++i) {
    Level L;
    L.t = times[i];
    L.u = states[i];
    L.p.SetSize(ns_->pressure_size());
    L.p = 0.0;
    hist_.push_back(L);
  }
  for (size_t i = 1; i < hist_.size(); ++i)
    AXF_REQUIRE(hist_[i].t < hist_[i - 1].t, "history times must decrease", "set_history");
}

double TransientNavierStokes::reported_norm(const mfem::Vector& e) const { return ns_->mass_norm(e); }

double TransientNavierStokes::controller_norm(const mfem::Vector& e) const { return ns_->mass_norm(e); }

// Milne device: est = kappa (u^n - u_pred), u_pred the extrapolation of degree `order` through the last order+1 levels
// (kappa = C / (C + P) from the error constants of the corrector and the predictor at the current step ratios); zero
// on the Dirichlet/axis DOFs.
mfem::Vector TransientNavierStokes::milne_estimate(int order, double tn, const mfem::Vector& un, bool& ok) const {
  ok = static_cast<int>(hist_.size()) >= order + 1;
  mfem::Vector est(un.Size());
  est = 0.0;
  if (!ok) return est;
  mfem::Vector up(un.Size());
  up = 0.0;
  for (int i = 0; i <= order; ++i) {
    double li = 1.0;
    for (int j = 0; j <= order; ++j)
      if (j != i) li *= (tn - hist_[j].t) / (hist_[i].t - hist_[j].t);
    up.Add(li, hist_[i].u);
  }
  const double s1 = tn - hist_[0].t;
  const double s2 = hist_[0].t - hist_[1].t;
  const double s3 = order == 2 ? hist_[1].t - hist_[2].t : s2;
  const double w = s1 / s2;
  double kappa;
  if (order == 1) {
    kappa = w / (1 + 2 * w);
  } else {
    const double C = s1 * s1 * s1 * (1 + w) * (1 + w) / (6 * w * (1 + 2 * w));
    const double P = s1 * (s1 + s2) * (s1 + s2 + s3) / 6.0;
    kappa = C / (C + P);
  }
  est = un;
  est -= up;
  est *= kappa;
  ns_->zero_essential(est);
  return est;
}

StepResult TransientNavierStokes::step_impl(double dt, int force_order, bool want_estimate) {
  AXF_REQUIRE(!hist_.empty(), "the solver is not initialized", "initialize");
  AXF_REQUIRE(dt > 0 && std::isfinite(dt), "the time step must be positive", "dt");
  const Level& prev = hist_.front();
  const double tn = prev.t + dt;
  int order = opt_.max_order;
  if (force_order > 0) order = std::min(order, force_order);
  std::vector<Level> L = hist_;
  if (order == 2 && L.size() < 2) order = 1;  // the BDF1 starter
  const double dt_prev = order == 2 ? prev.t - L[1].t : dt;
  const double wr = dt / dt_prev;
  std::vector<double> a = bdf_coefficients(order, dt, dt_prev);
  const double rho = ns_->options().rho;
  const double c = rho * a[0] / dt;
  mfem::Vector hv(ns_->velocity_size()), mu(ns_->velocity_size());
  hv = 0.0;
  ns_->set_time_term(c, hv);
  for (int j = 1; j <= order; ++j) {
    ns_->mass_mult_time(L[j - 1].u, mu);
    hv.Add(-a[j] * rho / dt, mu);
  }
  if (opt_.scheme == TimeScheme::SemiImplicit) {
    mfem::Vector w = L[0].u;
    if (order == 2) {
      const double we = wr;
      w *= (1.0 + we);
      w.Add(-we, L[1].u);
    }
    ns_->set_advecting(w);
  }
  refresh(tn, tn, tn);
  // Newton start: the previous state, extrapolated linearly in time when two levels exist (a better initial guess, one
  // Newton iteration fewer; the converged state is the same to the Newton tolerance). The semi-implicit solve is
  // linear: its start only fixes the residual that is solved for.
  mfem::Vector u_start = prev.u;
  if (opt_.scheme == TimeScheme::Newton && hist_.size() >= 2) {
    const double we = dt / (prev.t - hist_[1].t);
    u_start.Add(we, prev.u);
    u_start.Add(-we, hist_[1].u);
  }
  ns_->set_state(u_start, prev.p, true);
  ns_->set_time_term(c, hv);
  const NavierStokesDiagnostics& d = ns_->solve();
  StepResult r;
  r.converged = d.converged;
  r.relative_residual = d.relative_residual;
  r.iterations = d.iterations;
  r.lu_refine = d.linear.lu.refine_steps;
  r.failure = d.failure;
  r.t = tn;
  r.dt = dt;
  r.order = order;
  stats_.solves += 1;
  stats_.lu_fallbacks = d.lu_fallbacks;
  if (!d.lu_fallback_backend.empty()) stats_.lu_fallback_backend = d.lu_fallback_backend;
  stats_.newton_iterations += d.iterations;
  stats_.max_relative_residual = std::max(stats_.max_relative_residual, d.relative_residual);
  if (!d.converged) {
    stats_.all_converged = false;
    if (stats_.failure.empty()) stats_.failure = d.failure;
  }
  trial_.t = tn;
  trial_.u = ns_->state_u();
  trial_.p = ns_->state_p();
  ns_->set_time_term(0.0, hv);  // clear (steady paths unchanged afterwards)
  if (want_estimate) {
    bool ok = false;
    est_ = milne_estimate(order, tn, trial_.u, ok);
    r.has_estimate = ok;
    if (ok) {
      r.est_norm = reported_norm(est_);
      r.ctrl_norm = controller_norm(est_);
    }
  }
  return r;
}

StepResult TransientNavierStokes::step_trial(double dt, int force_order) {
  return step_impl(dt, force_order, true);
}

void TransientNavierStokes::commit() {
  Level L = trial_;
  hist_.insert(hist_.begin(), L);
  if (hist_.size() > 3) hist_.resize(3);
}

StepResult TransientNavierStokes::advance(double dt) {
  StepResult r = step_impl(dt, 0, false);
  if (r.converged) commit();
  return r;
}

AdaptiveResult TransientNavierStokes::run_adaptive(double T, double rtol, double S_u, double h0) {
  AXF_REQUIRE(rtol > 0 && S_u > 0 && h0 > 0, "rtol, S_u and h0 must be positive", "run_adaptive");
  AdaptiveResult R;
  double t = time();
  R.times.push_back(t);
  AXF_REQUIRE(T > t, "T must exceed the current time", "T");
  const double cap = opt_.ratio_cap;
  double dt = std::min(h0, T - t), prev_dt = 0;
  int nacc = 0;
  bool first_ratio = true;
  R.dt_max_ratio = 0;
  R.dt_min_ratio = 1e300;
  while (T - t > 1e-13 * T) {
    double dtt = dt;
    if (t + 1.1 * dtt >= T) dtt = T - t;  // land on T (a leftover below ~10 % merges into this step)
    int ord = (nacc < 2 || opt_.max_order == 1) ? 1 : 2;
    StepResult r;
    double estnorm = -1;
    bool failed = false;
    if (nacc == 0) {
      // first step: step doubling (no history for a Milne estimate): est = 2 (u_dt - u_two half steps)
      r = step_impl(dtt, 1, false);
      if (!r.converged) {
        failed = true;
      } else {
        Level full = trial_;
        std::vector<Level> backup = hist_;
        StepResult h1 = step_impl(dtt / 2, 1, false);
        if (!h1.converged) {
          failed = true;
        } else {
          commit();
          StepResult h2 = step_impl(dtt / 2, 1, false);
          if (!h2.converged) failed = true;
          else {
            est_ = full.u;
            est_ -= trial_.u;
            est_ *= 2.0;
            ns_->zero_essential(est_);
            estnorm = controller_norm(est_);
          }
        }
        hist_ = backup;
        trial_ = full;
      }
    } else {
      r = step_impl(dtt, ord, true);
      if (!r.converged) failed = true;
      else if (!r.has_estimate) {
        stats_.all_converged = false;
        stats_.failure = "no error estimate available (history too short)";
        break;
      } else estnorm = r.ctrl_norm;
    }
    if (failed) {
      ++R.failed_solves;
      dt = dtt / 2;
      if (dt < 1e-14 * T) break;
      continue;
    }
    const double err = estnorm / (rtol * S_u);
    const int p1 = (nacc == 0 ? 1 : ord) + 1;
    double fac = opt_.safety * std::pow(std::max(err, 1e-10), -1.0 / p1);
    fac = std::min(std::max(fac, opt_.ratio_floor), cap);
    const bool accept = err <= 1.0;
    if (accept) {
      commit();
      t = r.t;
      R.times.push_back(t);
      R.orders.push_back(nacc == 0 ? 1 : r.order);
      if (prev_dt > 0) {
        const double rat = dtt / prev_dt;
        R.dt_max_ratio = first_ratio ? rat : std::max(R.dt_max_ratio, rat);
        R.dt_min_ratio = std::min(R.dt_min_ratio, rat);
        first_ratio = false;
      }
      prev_dt = dtt;
      R.dt_max = std::max(R.dt_max, dtt);
      R.dt_min = R.dt_min == 0 ? dtt : std::min(R.dt_min, dtt);
      ++nacc;
      dt = dtt * fac;
    } else {
      ++R.rejected;
      dt = dtt * std::min(fac, 0.9);
    }
  }
  if (first_ratio) R.dt_max_ratio = R.dt_min_ratio = 1.0;
  R.reached = std::fabs(T - t) <= 1e-12 * T;
  return R;
}

}  // namespace axf::fluids
