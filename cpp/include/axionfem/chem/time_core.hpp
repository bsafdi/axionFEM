// The 0D time-integration core `axf::time` (beta; formulation: docs/guides/chemistry.md).  It lives in the chemistry
// sources; it is not used by the fluids module.
//
// Problem:  M y' = f(t, y) + s(t),  M constant (identity, diagonal or full; possibly singular: index-1 DAEs in the
// reaction-invariant form), f with an analytic Jacobian, s an optional additive source (used for manufactured
// solutions).
//
// Methods (all implicit, full Newton with the analytic Jacobian, dense LU; the systems are 0D and small):
//   - BDF-k, k = 1..5, VARIABLE-COEFFICIENT form on the actual past times (Lagrange derivative at t_{n+1}); local
//     error estimate d = E (y_{n+1} - y^p), y^p the degree-k interpolant through t_n .. t_{n-k},
//     E = (1/alpha_0) / (1/alpha_0 + (t_{n+1} - t_{n-k}))  (estimator limit 1 as h -> 0);
//   - one-step ESDIRK methods with an explicit first stage given by a Butcher tableau (A, b, bhat, c): ESDIRK3(2)4L[2]SA
//     (Kennedy & Carpenter 2003, implicit part of ARK3(2)4L[2]SA; p = 3, stage order 2, embedded order 2, stiffly
//     accurate, L-stable) and the theta = 1/2 member (trapezoid; the L-stability negative control) through the SAME
//     driver; estimate est = (embedded, lower order) - (advanced, higher order);
//   - drivers: prescribed grids (fixed step = uniform grid; every accepted step end is a grid time), adaptive runs with
//     stop-time outputs (every output time is an accepted step end), PI step control, events located by Brent on the
//     dense output, and a variable-order BDF1..5 driver (order selection from divided differences).
// Dense output: BDF -- the interpolating polynomial of the step's order through the step's nodes; one-step -- cubic
// Hermite on (y_n, y'_n, y_{n+1}, y'_{n+1}) (value order 3).
#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace axf::time {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct Problem {
  int n = 0;
  std::function<Vec(double, const Vec&)> f;    // right-hand side
  std::function<Mat(double, const Vec&)> jac;  // df/dy
  Mat mass;                                    // empty: identity
  std::function<Vec(double)> source;           // optional additive source (manufactured solutions)
  // opaque owners of foreign callables held by f/jac/source (the Python bindings keep their objects here so the
  // garbage collector can traverse them); the core never looks inside
  std::vector<std::shared_ptr<void>> held;

  Vec rhs(double t, const Vec& y) const;       // f + s
  Mat jacobian(double t, const Vec& y) const;
  Vec mass_times(const Vec& v) const;
  Mat mass_matrix() const;
  bool identity_mass() const { return mass.size() == 0; }
  void validate() const;                       // throws std::invalid_argument with what/how to fix
  std::vector<int> algebraic_rows() const;      // rows whose mass row is all zero (0 = f_i)
};

struct NewtonOptions {
  double rtol = 1e-10;   // relative residual: differential rows |G|/max(|c0 M y|, |f|), algebraic rows |G_i|/(|J||y|)_i; >= 1 update
  int max_iters = 25;
};

struct NewtonResult {
  bool converged = false;
  bool stagnated = false;  // stopped at round-off (update below 4 eps |y|) before reaching rtol
  double relres = 0.0;
  int iters = 0;           // linear solves
  int jac_evals = 0;
};

struct Stats {
  bool converged = true;       // every nonlinear solve that was accepted converged
  double max_relres = 0.0;     // worst final relative residual over accepted solves
  long newton_iters = 0;       // linear solves over accepted solves
  long solves = 0;             // accepted nonlinear solves
  long steps = 0;              // accepted steps
  long rejected = 0;           // error-test rejections
  long newton_failures = 0;    // steps retried because Newton failed
  long jac_evals = 0;
  long stagnated = 0;          // accepted solves stopped at round-off above the Newton tolerance
  void add(const NewtonResult& r);
};

// G(y) = c0 M y - w - f(t, y) - s(t) = 0 by full Newton from the initial guess y.
NewtonResult newton_solve(const Problem& P, double t, double c0, const Vec& w, Vec& y, const NewtonOptions& o);

// y'(t) consistent with M y' = f: ODE / invertible M: M^{-1} f; singular M: the algebraic rows differentiated,
// [M; W^T J] y' = [f; -W^T f_t] (W spans null(M^T); f_t by a central difference), least squares.
Vec consistent_derivative(const Problem& P, double t, const Vec& y);

// ------------------------------------------------------------------------------------------------------------ BDF
// Coefficients alpha_j (j = 0..k) of y'(t_{n+1}) ~ sum_j alpha_j y_{n+1-j} on the nodes {t_{n+1}, t_n, ...,
// t_{n+1-k}} (nodes[0] = t_{n+1}); variable-coefficient (Lagrange) form.
std::vector<double> bdf_coefficients(const std::vector<double>& nodes);

struct BdfStep {
  Vec y;
  Vec est;            // raw local-error estimate of y (NaN if not available)
  Vec y_pred;
  NewtonResult newton;
  bool ok = false;
};

// One BDF-k step of size h from the history th[0] = t_n, th[1] = t_{n-1}, ... (newest FIRST) with values yh.
// Needs >= k history points; the estimate needs k+1 (or k = 1 with yp_n, the derivative at t_n: Hermite predictor).
BdfStep bdf_advance(const Problem& P, const std::vector<double>& th, const std::vector<Vec>& yh, double h, int k,
                    const NewtonOptions& o, const Vec* yp_n = nullptr);

// --------------------------------------------------------------------------------------------------- one-step
struct Tableau {
  std::string name;
  Mat A;
  Vec b, bhat, c;
  int order = 0, embedded_order = 0;
  bool has_embedded = false;
  void validate() const;  // explicit first stage, nonzero diagonal afterwards, stiffly accurate
};
Tableau tableau_esdirk32();     // ESDIRK3(2)4L[2]SA (Kennedy & Carpenter 2003, ARK3(2)4L[2]SA implicit part)
Tableau tableau_theta_half();   // theta = 1/2 (trapezoid) written as a 2-stage ESDIRK: the L-stability control

struct RkStep {
  Vec y;
  Vec est;       // (embedded, lower order) - (advanced, higher order); NaN without an embedded method
  Vec yp_end;    // stage derivative of the last stage = y'(t_{n+1}) (stiffly accurate)
  Vec y_embedded;  // the embedded (lower-order) solution y + h sum bhat_j k_j itself (NaN without one)
  NewtonResult worst;
  int newton_iters = 0, jac_evals = 0, solves = 0, stagnated = 0;
  bool ok = false;
  bool advances_higher = true;
};

// One step from (t, y) with the consistent derivative yp (the first, explicit stage).
RkStep rk_advance(const Problem& P, const Tableau& tb, double t, const Vec& y, const Vec& yp, double h,
                  const NewtonOptions& o);

// --------------------------------------------------------------------------------------------------- results
struct EventHit {
  int index = 0;
  double t = 0.0;
  Vec y;
};

struct Trajectory {
  int kind = 0;                    // 0: BDF (Lagrange dense output), 1: one-step (Hermite dense output)
  std::vector<double> t;           // accepted step ends, t[0] = initial time
  std::vector<Vec> y;
  std::vector<Vec> yp;             // derivatives at step ends (one-step runs)
  std::vector<int> order;          // order used to reach t[i] (0 for the initial point)
  Stats stats;
  std::vector<EventHit> events;
  Vec dense(double tq) const;      // value of the dense output at tq in [t.front(), t.back()]
};

struct EventSpec {
  std::function<double(double, const Vec&)> g;
};

// Prescribed grid (fixed step = uniform grid): BDF-k.  y_start holds the values at times[0..m-1]; with m >= k the run
// starts from them ("exact start"), with m = 1 it self-starts: k = 2 one BDF1 step; k >= 3 the declared starter
// (ESDIRK3(2) with one Richardson extrapolation per step: local error O(h^5)).
Trajectory integrate_grid_bdf(const Problem& P, int k, std::vector<double> times, const std::vector<Vec>& y_start,
                              const NewtonOptions& o);
// Prescribed grid, one-step method (consistent derivative at times[0], then first-same-as-last).
Trajectory integrate_grid_rk(const Problem& P, const Tableau& tb, std::vector<double> times, const Vec& y0,
                             const NewtonOptions& o);

struct AdaptiveOptions {
  std::string method = "bdf2";   // "bdf2" (BDF1 start, then BDF2), "vsvo" (BDF1..max_order), "onestep" (tableau)
  int max_order = 5;             // vsvo
  Tableau tableau;               // onestep
  double rtol = 1e-6;
  Vec atol;                      // per component (size n)
  double h0 = 0.0;               // <= 0: automatic
  double safety = 0.9, facmin = 0.2, facmax_bdf = 2.0, facmax_rk = 5.0;
  bool pi_control = true;        // Gustafsson PI (bdf2, onestep); vsvo uses I control with a hold
  long max_steps = 2000000;
  NewtonOptions newton;
};

// Adaptive run from (t0, y0) through the sorted output times t_out (the last one is the end): every output time is
// an accepted step end (stop-time mode); every accepted step is recorded.  Error norm: weighted RMS with weights
// atol_i + rtol max(|y_n,i|, |y_{n+1},i|) over all components.
Trajectory integrate_adaptive(const Problem& P, const AdaptiveOptions& o, double t0, const Vec& y0,
                              const std::vector<double>& t_out, const std::vector<EventSpec>& events = {});

// Weighted RMS norm used by the controllers.
double wrms(const Vec& e, const Vec& ya, const Vec& yb, double rtol, const Vec& atol);

}  // namespace axf::time
