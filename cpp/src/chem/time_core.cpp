// Time core (see cpp/include/axionfem/chem/time_core.hpp).
#include "axionfem/chem/time_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace axf::time {

namespace {
constexpr double kEps = std::numeric_limits<double>::epsilon();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

bool all_finite(const Vec& v) { return v.allFinite(); }

// Lagrange interpolation of the values ys at the nodes xs, evaluated at x.
Vec lagrange(const std::vector<double>& xs, const std::vector<Vec>& ys, double x) {
  Vec out = Vec::Zero(ys[0].size());
  const size_t m = xs.size();
  for (size_t j = 0; j < m; ++j) {
    double l = 1.0;
    for (size_t i = 0; i < m; ++i)
      if (i != j) l *= (x - xs[i]) / (xs[j] - xs[i]);
    out += l * ys[j];
  }
  return out;
}

// Divided difference y[x_0, ..., x_m] (Newton form), x and y in any order.
Vec divided_difference(const std::vector<double>& xs, const std::vector<Vec>& ys) {
  std::vector<Vec> d(ys.begin(), ys.end());
  const size_t m = xs.size();
  for (size_t lev = 1; lev < m; ++lev)
    for (size_t i = 0; i + lev < m; ++i) d[i] = (d[i + 1] - d[i]) / (xs[i + lev] - xs[i]);
  return d[0];
}

double brent_root(const std::function<double(double)>& g, double a, double b, double ga, double gb) {
  // Brent's method (Brent 1973, zeroin), tolerance at round-off.
  double c = a, gc = ga, d = b - a, e = d;
  for (int it = 0; it < 200; ++it) {
    if ((gb > 0 && gc > 0) || (gb < 0 && gc < 0)) { c = a; gc = ga; d = b - a; e = d; }
    if (std::fabs(gc) < std::fabs(gb)) { a = b; b = c; c = a; ga = gb; gb = gc; gc = ga; }
    const double tol = 2 * kEps * std::fabs(b) + 1e-300;
    const double m = 0.5 * (c - b);
    if (std::fabs(m) <= tol || gb == 0.0) return b;
    if (std::fabs(e) >= tol && std::fabs(ga) > std::fabs(gb)) {
      double p, q, r, s = gb / ga;
      if (a == c) { p = 2 * m * s; q = 1 - s; }
      else { q = ga / gc; r = gb / gc; p = s * (2 * m * q * (q - r) - (b - a) * (r - 1)); q = (q - 1) * (r - 1) * (s - 1); }
      if (p > 0) q = -q; else p = -p;
      if (2 * p < std::min(3 * m * q - std::fabs(tol * q), std::fabs(e * q))) { e = d; d = p / q; }
      else { d = m; e = d; }
    } else { d = m; e = d; }
    a = b; ga = gb;
    b += (std::fabs(d) > tol) ? d : (m > 0 ? tol : -tol);
    gb = g(b);
  }
  return b;
}
}  // namespace

// ------------------------------------------------------------------------------------------------------- Problem
Vec Problem::rhs(double t, const Vec& y) const {
  Vec v = f(t, y);
  if (v.size() != n)
    throw std::invalid_argument("time Problem: f(t, y) returned " + std::to_string(v.size()) + " entries, expected n = " +
                                std::to_string(n) + " (one per unknown)");
  if (source) {
    Vec s = source(t);
    if (s.size() != n)
      throw std::invalid_argument("time Problem: source(t) returned " + std::to_string(s.size()) +
                                  " entries, expected n = " + std::to_string(n));
    v += s;
  }
  return v;
}
Mat Problem::jacobian(double t, const Vec& y) const {
  Mat J = jac(t, y);
  if (J.rows() != n || J.cols() != n)
    throw std::invalid_argument("time Problem: jac(t, y) returned a " + std::to_string(J.rows()) + " x " +
                                std::to_string(J.cols()) + " matrix, expected n x n with n = " + std::to_string(n));
  return J;
}
std::vector<int> Problem::algebraic_rows() const {
  std::vector<int> rows;
  if (identity_mass()) return rows;
  for (int i = 0; i < n; ++i)
    if (mass.row(i).cwiseAbs().maxCoeff() == 0.0) rows.push_back(i);
  return rows;
}
Vec Problem::mass_times(const Vec& v) const { return identity_mass() ? v : Vec(mass * v); }
Mat Problem::mass_matrix() const { return identity_mass() ? Mat(Mat::Identity(n, n)) : mass; }
void Problem::validate() const {
  if (n <= 0) throw std::invalid_argument("time Problem: n must be positive (number of unknowns)");
  if (!f || !jac) throw std::invalid_argument("time Problem: both f(t, y) and jac(t, y) are required");
  if (mass.size() && (mass.rows() != n || mass.cols() != n))
    throw std::invalid_argument("time Problem: mass matrix must be n x n (or empty for the identity)");
}

void Stats::add(const NewtonResult& r) {
  converged = converged && r.converged;
  max_relres = std::max(max_relres, r.relres);
  newton_iters += r.iters;
  jac_evals += r.jac_evals;
  if (r.stagnated) ++stagnated;
  ++solves;
}

double wrms(const Vec& e, const Vec& ya, const Vec& yb, double rtol, const Vec& atol) {
  double s = 0.0;
  const int n = static_cast<int>(e.size());
  for (int i = 0; i < n; ++i) {
    const double w = atol[i] + rtol * std::max(std::fabs(ya[i]), std::fabs(yb[i]));
    const double r = e[i] / w;
    s += r * r;
  }
  return std::sqrt(s / std::max(n, 1));
}

// ------------------------------------------------------------------------------------------------------- Newton
NewtonResult newton_solve(const Problem& P, double t, double c0, const Vec& w, Vec& y, const NewtonOptions& o) {
  // Relative residual (the Newton convergence test): differential rows |G_i| / max(|c0 M y|_inf, |f|_inf);
  // algebraic rows (an all-zero mass row: 0 = g_i(y)) |G_i| / (|J| |y|)_i, the size of the row's own terms, which
  // the global scale cannot measure because f_i -> 0 at the solution.  relres = the largest of the two.
  NewtonResult R;
  const Mat Mm = P.mass_matrix();
  const std::vector<int> alg = P.algebraic_rows();
  std::vector<char> is_alg(P.n, 0);
  for (int i : alg) is_alg[i] = 1;
  Mat Jac;
  Vec r;
  auto eval = [&](const Vec& yy) {
    Vec F = P.rhs(t, yy);
    Jac = P.jacobian(t, yy);
    ++R.jac_evals;
    Vec My = c0 * (Mm * yy);
    r = My - w - F;
    double gd = 0.0, gF = 0.0, ga = 0.0;
    for (int i = 0; i < P.n; ++i) {
      if (is_alg[i]) {
        const double si = std::max((Jac.row(i).cwiseAbs() * yy.cwiseAbs())(0, 0), 1e-300);
        ga = std::max(ga, std::fabs(r[i]) / si);
      } else {
        gd = std::max(gd, std::fabs(r[i]));
        gF = std::max(gF, std::fabs(F[i]));
      }
    }
    const double scale = std::max({My.lpNorm<Eigen::Infinity>(), F.lpNorm<Eigen::Infinity>(), 1e-300});
    (void)gF;
    return std::max(gd / scale, ga);
  };
  R.relres = eval(y);
  const double first = R.relres;
  for (int it = 0;; ++it) {
    if (!std::isfinite(R.relres)) return R;
    // at least one Newton update: a predictor whose residual is small relative to |c0 M y| (large for small steps)
    // is not yet the corrector solution
    if (it > 0 && R.relres <= o.rtol) { R.converged = true; return R; }
    if (it >= o.max_iters) return R;
    if (R.relres > 1e8 * std::max(first, 1e-8)) return R;   // diverging
    Eigen::PartialPivLU<Mat> lu(c0 * Mm - Jac);
    Vec d = lu.solve(-r);
    ++R.iters;
    if (!all_finite(d)) { R.relres = kNaN; return R; }
    y += d;
    R.relres = eval(y);
    if (d.lpNorm<Eigen::Infinity>() <= 4 * kEps * std::max(y.lpNorm<Eigen::Infinity>(), 1e-300)) {
      // the update is at round-off: Newton cannot improve further (stagnation is convergence)
      R.converged = std::isfinite(R.relres);
      R.stagnated = R.relres > o.rtol;
      return R;
    }
  }
}

Vec consistent_derivative(const Problem& P, double t, const Vec& y) {
  Vec F = P.rhs(t, y);
  if (P.identity_mass()) return F;
  Eigen::JacobiSVD<Mat> svd(P.mass, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Vec& sv = svd.singularValues();
  int r = 0;
  for (int i = 0; i < sv.size(); ++i)
    if (sv[i] > 1e-12 * sv[0]) ++r;
  if (r == P.n) return P.mass.partialPivLu().solve(F);
  Mat W = svd.matrixU().rightCols(P.n - r);            // null(M^T)
  const double dt = 1e-6 * std::max(1.0, std::fabs(t));
  Vec Ft = (P.rhs(t + dt, y) - P.rhs(t - dt, y)) / (2 * dt);
  Mat A(P.n + (P.n - r), P.n);
  A << P.mass, W.transpose() * P.jacobian(t, y);
  Vec b(P.n + (P.n - r));
  b << F, -W.transpose() * Ft;
  return A.colPivHouseholderQr().solve(b);
}

// ---------------------------------------------------------------------------------------------------------- BDF
std::vector<double> bdf_coefficients(const std::vector<double>& nodes) {
  const int k = static_cast<int>(nodes.size()) - 1;
  if (k < 1) throw std::invalid_argument("bdf_coefficients: need at least two nodes");
  std::vector<double> alpha(k + 1, 0.0);
  const int kk = k;
  const double x0 = nodes[0];
  for (int m = 1; m <= kk; ++m) alpha[0] += 1.0 / (x0 - nodes[m]);
  for (int j = 1; j <= kk; ++j) {
    double num = 1.0, den = 1.0;
    for (int m = 1; m <= kk; ++m)
      if (m != j) num *= (x0 - nodes[m]);
    for (int m = 0; m <= kk; ++m)
      if (m != j) den *= (nodes[j] - nodes[m]);
    alpha[j] = num / den;
  }
  return alpha;
}

BdfStep bdf_advance(const Problem& P, const std::vector<double>& th, const std::vector<Vec>& yh, double h, int k,
                    const NewtonOptions& o, const Vec* yp_n) {
  BdfStep S;
  if (k < 1 || k > 6) throw std::invalid_argument("bdf_advance: order must be 1..6");
  if (static_cast<int>(th.size()) < k || th.size() != yh.size())
    throw std::invalid_argument("bdf_advance: BDF-k needs k history points (newest first), times and values paired");
  const double t1 = th[0] + h;
  std::vector<double> nodes(k + 1);
  nodes[0] = t1;
  for (int j = 0; j < k; ++j) nodes[j + 1] = th[j];
  const std::vector<double> alpha = bdf_coefficients(nodes);

  // predictor
  const bool full_pred = static_cast<int>(th.size()) >= k + 1;
  double psi = kNaN;
  if (full_pred) {
    std::vector<double> xs(th.begin(), th.begin() + k + 1);
    std::vector<Vec> ys(yh.begin(), yh.begin() + k + 1);
    S.y_pred = lagrange(xs, ys, t1);
    psi = 1.0;
    for (int j = 0; j <= k; ++j) psi *= (t1 - th[j]);
  } else if (k == 1 && yp_n) {
    S.y_pred = yh[0] + h * (*yp_n);
    psi = h * h;
  } else {
    // fewer points than the predictor needs: extrapolate with what there is (no estimate)
    const int m = static_cast<int>(th.size());
    std::vector<double> xs(th.begin(), th.begin() + m);
    std::vector<Vec> ys(yh.begin(), yh.begin() + m);
    S.y_pred = lagrange(xs, ys, t1);
  }

  Vec w;
  const double c0 = alpha[0];
  {
    Vec hist = Vec::Zero(P.n);
    for (int j = 1; j <= k; ++j) hist += alpha[j] * yh[j - 1];
    w = -P.mass_times(hist);
  }
  S.y = S.y_pred;
  S.newton = newton_solve(P, t1, c0, w, S.y, o);
  S.ok = S.newton.converged && all_finite(S.y);
  if (std::isnan(psi)) {
    S.est = Vec::Constant(P.n, kNaN);
    return S;
  }
  // E = (phi'/alpha0) / (phi'/alpha0 + psi), phi' = prod_{i=1..k} (t1 - nodes[i])
  double phip = 1.0;
  for (int i = 1; i <= k; ++i) phip *= (t1 - nodes[i]);
  const double a = phip / alpha[0];
  const double E = a / (a + psi);
  S.est = E * (S.y - S.y_pred);
  return S;
}

// ---------------------------------------------------------------------------------------------------- one-step
void Tableau::validate() const {
  const int s = static_cast<int>(b.size());
  if (A.rows() != s || A.cols() != s || c.size() != s)
    throw std::invalid_argument("Tableau " + name + ": A must be s x s and b, c of length s");
  if (A(0, 0) != 0.0) throw std::invalid_argument("Tableau " + name + ": the first stage must be explicit");
  for (int i = 1; i < s; ++i)
    if (!(A(i, i) > 0)) throw std::invalid_argument("Tableau " + name + ": diagonal entries after stage 1 must be > 0");
  for (int j = 0; j < s; ++j)
    if (std::fabs(A(s - 1, j) - b[j]) > 1e-15)
      throw std::invalid_argument("Tableau " + name + ": the driver needs a stiffly accurate tableau (b = last row of A)");
  if (has_embedded && bhat.size() != s) throw std::invalid_argument("Tableau " + name + ": bhat must have length s");
}

Tableau tableau_esdirk32() {
  Tableau T;
  T.name = "ESDIRK3(2)4L[2]SA";
  const double g = 1767732205903.0 / 4055673282236.0;
  T.A = Mat::Zero(4, 4);
  T.A(1, 0) = g; T.A(1, 1) = g;
  T.A(2, 0) = 2746238789719.0 / 10658868560708.0;
  T.A(2, 1) = -640167445237.0 / 6845629431997.0;
  T.A(2, 2) = g;
  T.A(3, 0) = 1471266399579.0 / 7840856788654.0;
  T.A(3, 1) = -4482444167858.0 / 7529755066697.0;
  T.A(3, 2) = 11266239266428.0 / 11593286722821.0;
  T.A(3, 3) = g;
  T.b = T.A.row(3).transpose();
  T.bhat.resize(4);
  T.bhat << 2756255671327.0 / 12835298489170.0, -10771552573575.0 / 22201958757719.0,
      9247589265047.0 / 10645013368117.0, 2193209047091.0 / 5459859503100.0;
  T.c.resize(4);
  T.c << 0.0, 2 * g, 3.0 / 5.0, 1.0;
  T.order = 3;
  T.embedded_order = 2;
  T.has_embedded = true;
  return T;
}

Tableau tableau_theta_half() {
  Tableau T;
  T.name = "theta=1/2 (trapezoid)";
  T.A = Mat::Zero(2, 2);
  T.A(1, 0) = 0.5; T.A(1, 1) = 0.5;
  T.b = T.A.row(1).transpose();
  T.c.resize(2);
  T.c << 0.0, 1.0;
  T.order = 2;
  T.has_embedded = false;
  return T;
}

RkStep rk_advance(const Problem& P, const Tableau& tb, double t, const Vec& y, const Vec& yp, double h,
                  const NewtonOptions& o) {
  RkStep R;
  const int s = static_cast<int>(tb.b.size());
  std::vector<Vec> K(s);
  K[0] = yp;
  Vec Y = y;
  R.ok = true;
  for (int i = 1; i < s; ++i) {
    Vec base = y;
    for (int j = 0; j < i; ++j) base += h * tb.A(i, j) * K[j];
    const double ga = tb.A(i, i);
    const double c0 = 1.0 / (h * ga);
    Vec w = c0 * P.mass_times(base);
    Y = base + h * ga * K[i - 1];
    NewtonResult nr = newton_solve(P, t + tb.c[i] * h, c0, w, Y, o);
    R.newton_iters += nr.iters;
    R.jac_evals += nr.jac_evals;
    ++R.solves;
    if (nr.stagnated) ++R.stagnated;
    if (!nr.converged || !all_finite(Y)) {
      R.ok = false;
      R.worst = nr;
      return R;
    }
    if (i == 1 || nr.relres > R.worst.relres) R.worst = nr;
    K[i] = (Y - base) * c0;
  }
  R.worst.converged = true;
  R.y = Y;                 // stiffly accurate
  R.yp_end = K[s - 1];
  if (tb.has_embedded) {
    Vec e = Vec::Zero(P.n);
    for (int j = 0; j < s; ++j) e += (tb.bhat[j] - tb.b[j]) * K[j];
    R.est = h * e;
    R.y_embedded = y;
    for (int j = 0; j < s; ++j) R.y_embedded += h * tb.bhat[j] * K[j];
  } else {
    R.est = Vec::Constant(P.n, kNaN);
    R.y_embedded = Vec::Constant(P.n, kNaN);
  }
  return R;
}

// ------------------------------------------------------------------------------------------------- dense output
Vec Trajectory::dense(double tq) const {
  if (t.empty()) throw std::runtime_error("dense output of an empty trajectory");
  if (tq < t.front() - 1e-12 * std::max(1.0, std::fabs(t.front())) ||
      tq > t.back() + 1e-12 * std::max(1.0, std::fabs(t.back())))
    throw std::out_of_range("dense output: t outside the integrated interval");
  if (t.size() == 1) return y[0];
  size_t i = static_cast<size_t>(std::upper_bound(t.begin(), t.end(), tq) - t.begin());
  if (i == 0) i = 1;
  if (i >= t.size()) i = t.size() - 1;
  // segment [t[i-1], t[i]]
  if (kind == 1) {
    const double h = t[i] - t[i - 1];
    const double s = (tq - t[i - 1]) / h;
    const double h00 = (1 + 2 * s) * (1 - s) * (1 - s), h10 = s * (1 - s) * (1 - s);
    const double h01 = s * s * (3 - 2 * s), h11 = s * s * (s - 1);
    return h00 * y[i - 1] + h10 * h * yp[i - 1] + h01 * y[i] + h11 * h * yp[i];
  }
  int k = std::max(1, order[i]);
  k = std::min<int>(k, static_cast<int>(i));
  std::vector<double> xs;
  std::vector<Vec> ys;
  for (int j = 0; j <= k; ++j) {
    xs.push_back(t[i - j]);
    ys.push_back(y[i - j]);
  }
  return lagrange(xs, ys, tq);
}

namespace {
void check_events(Trajectory& tr, const std::vector<EventSpec>& ev, std::vector<double>& gprev) {
  const size_t i = tr.t.size() - 1;
  for (size_t e = 0; e < ev.size(); ++e) {
    const double g1 = ev[e].g(tr.t[i], tr.y[i]);
    const double g0 = gprev[e];
    gprev[e] = g1;
    if (!(g0 != 0.0) || !std::isfinite(g0) || !std::isfinite(g1)) continue;
    if (g1 != 0.0 && (g0 > 0) == (g1 > 0)) continue;
    double te = tr.t[i];
    if (g1 != 0.0) {
      auto g = [&](double s) { return ev[e].g(s, tr.dense(s)); };
      te = brent_root(g, tr.t[i - 1], tr.t[i], g0, g1);
    }
    tr.events.push_back({static_cast<int>(e), te, tr.dense(te)});
  }
}

void check_grid(const std::vector<double>& times) {
  if (times.size() < 2) throw std::invalid_argument("prescribed grid: need at least two times");
  for (size_t i = 1; i < times.size(); ++i)
    if (!(times[i] > times[i - 1])) throw std::invalid_argument("prescribed grid: times must increase strictly");
}
}  // namespace

// ------------------------------------------------------------------------------------------------ grid drivers
Trajectory integrate_grid_rk(const Problem& P, const Tableau& tb, std::vector<double> times, const Vec& y0,
                             const NewtonOptions& o) {
  P.validate();
  tb.validate();
  check_grid(times);
  Trajectory tr;
  tr.kind = 1;
  tr.t.push_back(times[0]);
  tr.y.push_back(y0);
  Vec yp = consistent_derivative(P, times[0], y0);
  tr.yp.push_back(yp);
  tr.order.push_back(0);
  for (size_t i = 1; i < times.size(); ++i) {
    RkStep R = rk_advance(P, tb, tr.t.back(), tr.y.back(), yp, times[i] - tr.t.back(), o);
    if (!R.ok) {
      tr.stats.converged = false;
      tr.stats.max_relres = std::max(tr.stats.max_relres, R.worst.relres);
      throw std::runtime_error("integrate_grid_rk: Newton failed at t = " + std::to_string(tr.t.back()) +
                               " (relative residual " + std::to_string(R.worst.relres) +
                               "); use smaller steps or check the Jacobian");
    }
    tr.stats.converged = tr.stats.converged && true;
    tr.stats.max_relres = std::max(tr.stats.max_relres, R.worst.relres);
    tr.stats.newton_iters += R.newton_iters;
    tr.stats.jac_evals += R.jac_evals;
    tr.stats.solves += R.solves;
    tr.stats.stagnated += R.stagnated;
    ++tr.stats.steps;
    Vec yn = R.y;
    yp = R.yp_end;
    tr.t.push_back(times[i]);
    tr.y.push_back(yn);
    tr.yp.push_back(yp);
    tr.order.push_back(tb.order);
  }
  return tr;
}

namespace {
// One step [ta, tb] of ESDIRK3(2) with one Richardson extrapolation (local error O(h^5)): the BDF k >= 3 starter.
Vec extrapolated_esdirk_step(const Problem& P, double ta, const Vec& ya, double tb_, const NewtonOptions& o, Stats& st) {
  const Tableau T = tableau_esdirk32();
  const double H = tb_ - ta;
  auto run = [&](int m) {
    Vec y = ya;
    double t = ta;
    for (int j = 0; j < m; ++j) {
      Vec yp = consistent_derivative(P, t, y);
      const double tn = (j == m - 1) ? tb_ : ta + H * (j + 1) / m;
      RkStep R = rk_advance(P, T, t, y, yp, tn - t, o);
      if (!R.ok) throw std::runtime_error("BDF starter (extrapolated ESDIRK3(2)): Newton failed");
      st.max_relres = std::max(st.max_relres, R.worst.relres);
      st.newton_iters += R.newton_iters;
      st.jac_evals += R.jac_evals;
      st.solves += R.solves;
      st.stagnated += R.stagnated;
      y = R.y;
      t = tn;
    }
    return y;
  };
  Vec y1 = run(1), y2 = run(2);
  return y2 + (y2 - y1) / 7.0;
}
}  // namespace

Trajectory integrate_grid_bdf(const Problem& P, int k, std::vector<double> times, const std::vector<Vec>& y_start,
                              const NewtonOptions& o) {
  P.validate();
  check_grid(times);
  if (k < 1 || k > 5) throw std::invalid_argument("integrate_grid_bdf: order must be 1..5");
  if (y_start.empty()) throw std::invalid_argument("integrate_grid_bdf: y_start needs the initial value");
  Trajectory tr;
  tr.kind = 0;
  const size_t N = times.size() - 1;
  const size_t m = std::min<size_t>(y_start.size(), N + 1);
  for (size_t i = 0; i < m; ++i) {
    tr.t.push_back(times[i]);
    tr.y.push_back(y_start[i]);
    tr.order.push_back(static_cast<int>(std::min<size_t>(i, k)));
  }
  auto bdf_one = [&](size_t i, int kk) {
    std::vector<double> th;
    std::vector<Vec> yh;
    for (int j = 0; j < kk && j < static_cast<int>(tr.t.size()); ++j) {
      th.push_back(tr.t[tr.t.size() - 1 - j]);
      yh.push_back(tr.y[tr.y.size() - 1 - j]);
    }
    BdfStep S = bdf_advance(P, th, yh, times[i] - tr.t.back(), kk, o);
    if (!S.ok)
      throw std::runtime_error("integrate_grid_bdf: Newton failed at t = " + std::to_string(times[i]) +
                               " (relative residual " + std::to_string(S.newton.relres) + ")");
    tr.stats.add(S.newton);
    ++tr.stats.steps;
    Vec yn = S.y;
    tr.t.push_back(times[i]);
    tr.y.push_back(yn);
    tr.order.push_back(kk);
  };
  size_t i = m;
  if (m < static_cast<size_t>(k)) {
    // self start: values at times[m .. k-1]
    for (; i < static_cast<size_t>(k) && i <= N; ++i) {
      if (k <= 2) {
        bdf_one(i, static_cast<int>(std::min<size_t>(i, k - 1)));   // BDF1 (k = 2)
      } else {
        Vec yn = extrapolated_esdirk_step(P, tr.t.back(), tr.y.back(), times[i], o, tr.stats);
        ++tr.stats.steps;
        tr.t.push_back(times[i]);
        tr.y.push_back(yn);
        tr.order.push_back(static_cast<int>(i));
      }
    }
  }
  for (; i <= N; ++i) bdf_one(i, k);
  return tr;
}

// -------------------------------------------------------------------------------------------- adaptive driver
namespace {
double initial_step(const Problem& P, double t0, const Vec& y0, const Vec& yp0, double rtol, const Vec& atol,
                    int p, double span) {
  // Hairer, Norsett & Wanner I, II.4 (starting step size), with the consistent derivative.
  const Vec z = Vec::Zero(P.n);
  const double d0 = wrms(y0, y0, y0, rtol, atol), d1 = wrms(yp0, y0, y0, rtol, atol);
  double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 * span : 0.01 * d0 / d1;
  h0 = std::min(h0, span);
  Vec y1 = y0 + h0 * yp0;
  Vec yp1;
  try {
    yp1 = consistent_derivative(P, t0 + h0, y1);
  } catch (...) {
    return h0 * 1e-2;
  }
  const double d2 = wrms(yp1 - yp0, y0, y0, rtol, atol) / h0;
  const double mx = std::max(d1, d2);
  const double h1 = mx <= 1e-15 ? std::max(1e-6 * span, h0 * 1e-3) : std::pow(0.01 / mx, 1.0 / (p + 1));
  (void)z;
  return std::min({100 * h0, h1, span});
}
}  // namespace

Trajectory integrate_adaptive(const Problem& P, const AdaptiveOptions& o, double t0, const Vec& y0,
                              const std::vector<double>& t_out, const std::vector<EventSpec>& events) {
  P.validate();
  if (t_out.empty()) throw std::invalid_argument("integrate_adaptive: give at least one output time (the end)");
  for (size_t i = 0; i < t_out.size(); ++i)
    if (!(t_out[i] > (i ? t_out[i - 1] : t0)))
      throw std::invalid_argument("integrate_adaptive: output times must increase strictly and exceed t0");
  if (o.atol.size() != P.n) throw std::invalid_argument("integrate_adaptive: atol must have one entry per unknown");
  const bool rk = o.method == "onestep";
  const bool vsvo = o.method == "vsvo";
  if (!rk && !vsvo && o.method != "bdf2")
    throw std::invalid_argument("integrate_adaptive: method must be 'bdf2', 'vsvo' or 'onestep' (got '" + o.method + "')");
  if (rk) o.tableau.validate();
  if (rk && !o.tableau.has_embedded) throw std::invalid_argument("integrate_adaptive: the tableau has no embedded method");
  const int maxk = vsvo ? std::max(1, std::min(o.max_order, 5)) : 2;
  const double tend = t_out.back();
  const double span = tend - t0;

  Trajectory tr;
  tr.kind = rk ? 1 : 0;
  tr.t.push_back(t0);
  tr.y.push_back(y0);
  tr.order.push_back(0);
  Vec yp = consistent_derivative(P, t0, y0);
  tr.yp.push_back(yp);
  std::vector<double> gprev;
  for (const auto& e : events) gprev.push_back(e.g(t0, y0));

  const int p_ctrl = rk ? o.tableau.embedded_order : 1;
  double h = o.h0 > 0 ? o.h0 : initial_step(P, t0, y0, yp, o.rtol, o.atol, p_ctrl, span);
  int k = 1;
  int nsame = 0;            // steps since the last change of order or step size (vsvo hold)
  int nfail = 0;            // consecutive failures on the current step
  double err_prev = -1.0;
  size_t iout = 0;
  long nsteps = 0;
  while (iout < t_out.size()) {
    if (++nsteps > o.max_steps)
      throw std::runtime_error("integrate_adaptive: more than max_steps steps; loosen the tolerances or check the problem");
    const double t = tr.t.back();
    const Vec y = tr.y.back();   // a COPY: tr.y grows (and may reallocate) when the step is accepted
    const double target = t_out[iout];
    double hs = h;
    bool lands = false;
    const double rem = target - t;
    if (hs >= rem * (1 - 1e-12)) { hs = rem; lands = true; }
    else if (2 * hs > rem) hs = 0.5 * rem;
    if (!(hs > 1e-14 * std::max(1.0, std::fabs(t))))
      throw std::runtime_error("integrate_adaptive: step size underflow at t = " + std::to_string(t));
    const double t1 = lands ? target : t + hs;

    Vec ynew, est;
    NewtonResult nr;
    bool ok;
    Vec yp_end;
    int used_k = k;
    long extra_solves = 0, extra_stag = 0;
    if (rk) {
      RkStep R = rk_advance(P, o.tableau, t, y, yp, t1 - t, o.newton);
      ok = R.ok;
      nr = R.worst;
      nr.iters = R.newton_iters;
      nr.jac_evals = R.jac_evals;
      nr.stagnated = false;
      if (ok) { ynew = R.y; est = R.est; yp_end = R.yp_end; extra_solves = R.solves - 1; extra_stag = R.stagnated; }
    } else {
      std::vector<double> th;
      std::vector<Vec> yh;
      const size_t nh = tr.t.size();
      for (size_t j = 0; j < nh && j < 8; ++j) {
        th.push_back(tr.t[nh - 1 - j]);
        yh.push_back(tr.y[nh - 1 - j]);
      }
      used_k = std::min<int>(k, static_cast<int>(nh));
      // the estimate needs k+1 points (k = 1 at the start: Hermite predictor with y'_0)
      if (static_cast<int>(nh) < used_k + 1 && used_k > 1) used_k = static_cast<int>(nh) - 1;
      BdfStep S = bdf_advance(P, th, yh, t1 - t, used_k, o.newton, nh == 1 ? &yp : nullptr);
      ok = S.ok;
      nr = S.newton;
      if (ok) { ynew = S.y; est = S.est; }
    }
    if (!ok) {
      ++tr.stats.newton_failures;
      ++tr.stats.rejected;
      h = 0.25 * hs;
      nsame = 0;
      ++nfail;
      continue;
    }
    double err = wrms(est, y, ynew, o.rtol, o.atol);
    if (!std::isfinite(err)) err = 1e10;
    const int pe = rk ? o.tableau.embedded_order : used_k;   // estimate is O(h^{pe+1})
    if (err > 1.0) {
      ++tr.stats.rejected;
      ++nfail;
      double fac = std::max(o.facmin, o.safety * std::pow(err, -1.0 / (pe + 1)));
      if (nfail >= 3) {
        fac = std::min(fac, 0.25);
        if (vsvo && k > 1) k -= 1;
      }
      h = hs * std::min(fac, 0.9);
      nsame = 0;
      continue;
    }
    // accept
    nfail = 0;
    tr.stats.add(nr);
    tr.stats.solves += extra_solves;   // Stats::add counts one solve per step
    tr.stats.stagnated += extra_stag;
    ++tr.stats.steps;
    tr.t.push_back(t1);
    tr.y.push_back(ynew);
    tr.order.push_back(rk ? o.tableau.order : used_k);
    if (rk) { yp = yp_end; tr.yp.push_back(yp); }
    check_events(tr, events, gprev);
    if (lands) ++iout;

    const double e = std::max(err, 1e-10);
    double fac;
    if (rk || !vsvo) {
      fac = o.safety * std::pow(e, -1.0 / (pe + 1));
      if (o.pi_control && err_prev > 0)
        fac = o.safety * std::pow(e, -0.7 / (pe + 1)) * std::pow(err_prev, 0.4 / (pe + 1));
      const double fmax = rk ? o.facmax_rk : o.facmax_bdf;
      fac = std::min(fmax, std::max(o.facmin, fac));
      if (!rk) {
        const int nh = static_cast<int>(tr.t.size());
        k = std::min(2, nh - 1);   // BDF1 for the first two steps, then BDF2
        if (k < 1) k = 1;
      }
      err_prev = e;
      h = hs * fac;
    } else {
      // variable order: candidates k-1, k, k+1 from divided differences of the accepted values
      const int nh = static_cast<int>(tr.t.size());
      double best = o.safety * std::pow(e, -1.0 / (used_k + 1));
      int knew = used_k;
      ++nsame;
      if (nsame >= used_k + 1) {
        auto est_order = [&](int j) {   // local error of BDF-j from the (j+1)-th divided difference
          std::vector<double> xs;
          std::vector<Vec> ys;
          for (int m = 0; m <= j + 1; ++m) {
            xs.push_back(tr.t[nh - 1 - m]);
            ys.push_back(tr.y[nh - 1 - m]);
          }
          Vec Y = divided_difference(xs, ys);
          double phip = 1.0, a0 = 0.0;
          for (int m = 1; m <= j; ++m) {
            phip *= (xs[0] - xs[m]);
            a0 += 1.0 / (xs[0] - xs[m]);
          }
          return Vec(Y * (phip / a0));
        };
        if (used_k > 1) {
          const double em = std::max(wrms(est_order(used_k - 1), y, ynew, o.rtol, o.atol), 1e-10);
          const double r = o.safety / 1.3 * std::pow(em, -1.0 / used_k);
          if (r > best) { best = r; knew = used_k - 1; }
        }
        if (used_k < maxk && nh >= used_k + 3) {
          const double ep = std::max(wrms(est_order(used_k + 1), y, ynew, o.rtol, o.atol), 1e-10);
          const double r = o.safety / 1.4 * std::pow(ep, -1.0 / (used_k + 2));
          if (r > best) { best = r; knew = used_k + 1; }
        }
      }
      double fac = std::min(o.facmax_bdf, std::max(o.facmin, best));
      if (fac > 1.0 && (nsame < used_k + 1 || fac < 1.2) && knew == used_k) fac = 1.0;   // hold / deadband
      if (knew != used_k || fac != 1.0) nsame = 0;
      k = knew;
      h = hs * fac;
    }
  }
  return tr;
}

}  // namespace axf::time
