// Set-valued AAA with Hermite data, and the adaptive frequency sweep (docs/physics/adaptive_sweep.md).
#include "axionfem/rational_fit.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf::rational {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// the sweep stops only on a fit that reproduces its own solves to this multiple of its fit tolerance
constexpr double kConsistentFactor = 10.0;
// consecutive solves within tolerance / 100 of their predictions that close a sweep on their own
constexpr int kTightConfirmations = 4;

// Standard normal draws by Box-Muller on mt19937's raw output, which the standard specifies; std::normal_distribution
// is implementation-defined, so with it the noise refits, and a marginal pole's class, would differ between standard
// libraries. std::log, std::sin and std::cos may still differ in the last bit.
class Gauss {
 public:
  explicit Gauss(unsigned seed) : rng_(seed) {}
  double operator()() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = (static_cast<double>(rng_()) + 0.5) / 4294967296.0;
    const double u2 = (static_cast<double>(rng_()) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(u1));
    spare_ = radius * std::sin(2 * units::pi * u2);
    has_spare_ = true;
    return radius * std::cos(2 * units::pi * u2);
  }

 private:
  std::mt19937 rng_;
  bool has_spare_ = false;
  double spare_ = 0;
};

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const double below = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + below);
  }
  return m;
}

// The weighted, linearized least-squares problem on fixed samples (sorted, distinct frequencies).
struct Problem {
  const std::vector<Sample>* samples = nullptr;
  int M = 0;
  int K = 0;
  Eigen::MatrixXd value_weight;        // M x K: 1 / max(|F_ik|, floor_k); 0 for an inactive output
  Eigen::MatrixXd derivative_weight;   // M x K: min(h_i, max(|F_ik|, floor_k) / |F'_ik|) x value weight; 0: no data
  Eigen::VectorXd floor;
  Eigen::VectorXd scale;

  double f(int i) const { return (*samples)[i].f; }
  cplx F(int i, int k) const { return (*samples)[i].value(k); }
  cplx G(int i, int k) const { return (*samples)[i].derivative(k); }
};

// Rows of the linearized conditions for the support set S (sample indices), one column per support point:
//   value at a sample i outside S:       sum_j w_j (F_i - F_j) / (f_i - f_j) = 0                  (F_i D - N = 0)
//   derivative at a sample i outside S:  sum_j w_j [F'_i / (f_i - f_j) - (F_i - F_j) / (f_i - f_j)^2] = 0
//   derivative at a support point i:     F'_i w_i + sum_{j != i} w_j (F_i - F_j) / (f_i - f_j) = 0
// (the last two are d/df of (F_i + F'_i (f - f_i)) D(f) - N(f) at f_i, and r'(f_i) = F'_i at a support point).
Eigen::MatrixXcd loewner(const Problem& P, const std::vector<int>& S) {
  const int m = static_cast<int>(S.size());
  std::vector<char> in_support(P.M, 0);
  for (int j : S) in_support[j] = 1;
  int rows = 0;
  for (int k = 0; k < P.K; ++k)
    for (int i = 0; i < P.M; ++i) {
      if (!in_support[i] && P.value_weight(i, k) > 0) ++rows;
      if (P.derivative_weight(i, k) > 0) ++rows;
    }
  Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(std::max(rows, m), m);   // zero rows pad an underdetermined system
  int r = 0;
  for (int k = 0; k < P.K; ++k)
    for (int i = 0; i < P.M; ++i) {
      const double wv = P.value_weight(i, k);
      const double wd = P.derivative_weight(i, k);
      if (!in_support[i] && wv > 0) {
        for (int a = 0; a < m; ++a) {
          const int j = S[a];
          L(r, a) = wv * (P.F(i, k) - P.F(j, k)) / (P.f(i) - P.f(j));
        }
        ++r;
      }
      if (wd > 0) {
        for (int a = 0; a < m; ++a) {
          const int j = S[a];
          if (j == i) {
            L(r, a) = wd * P.G(i, k);
            continue;
          }
          const double d = P.f(i) - P.f(j);
          const cplx dF = P.F(i, k) - P.F(j, k);
          if (in_support[i]) L(r, a) = wd * dF / d;
          else L(r, a) = wd * (P.G(i, k) / d - dF / (d * d));
        }
        ++r;
      }
    }
  return L;
}

Model make_model(const Problem& P, const std::vector<int>& S, const Eigen::VectorXcd& w) {
  Model model;
  model.weights = w;
  model.values.resize(static_cast<int>(S.size()), P.K);
  for (size_t a = 0; a < S.size(); ++a) {
    model.support.push_back(P.f(S[a]));
    model.values.row(static_cast<int>(a)) = (*P.samples)[S[a]].value.transpose();
  }
  return model;
}

// The weights for the support set S: the right singular vector of the smallest singular value.
Model solve_weights(const Problem& P, const std::vector<int>& S) {
  const Eigen::MatrixXcd L = loewner(P, S);
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(L, Eigen::ComputeFullV);
  const Eigen::VectorXcd w = svd.matrixV().col(static_cast<int>(S.size()) - 1);
  return make_model(P, S, w);
}

// The largest weighted misfit per sample: values (zero at support points) and derivatives.
Eigen::VectorXd sample_errors(const Problem& P, const Model& model) {
  Eigen::VectorXd err = Eigen::VectorXd::Zero(P.M);
  for (int i = 0; i < P.M; ++i) {
    const Eigen::VectorXcd r = model.value(P.f(i));
    bool needs_derivative = false;
    for (int k = 0; k < P.K; ++k) needs_derivative = needs_derivative || P.derivative_weight(i, k) > 0;
    Eigen::VectorXcd dr;
    if (needs_derivative) dr = model.derivative(P.f(i));
    for (int k = 0; k < P.K; ++k) {
      double e = P.value_weight(i, k) * std::abs(r(k) - P.F(i, k));
      if (P.derivative_weight(i, k) > 0) e = std::max(e, P.derivative_weight(i, k) * std::abs(dr(k) - P.G(i, k)));
      if (std::isnan(e)) e = kInf;
      err(i) = std::max(err(i), e);
    }
  }
  return err;
}

// Froissart doublets: poles whose residues are negligible in every output (Nakatsukasa, Sete and Trefethen 2018,
// Sec. 5): the support point nearest to one is removed and the weights are solved again.
// - Candidates: a residue (relative to the output's scale and the band width) below 100 times the larger of the fit
//   tolerance and the achieved misfit, the misfit counted up to the noise cap (a misfit above it means too few
//   samples, not noise), and always below doublet_tolerance. Noisy data fitted below their noise give doublets with
//   residues of a few times the misfit (1e-9 noise: up to 3.4x; 1e-11: up to 11x, on test_rational_fit's case 3).
// - The candidate threshold alone cannot tell a doublet from a weak genuine resonance: in one test cavity the genuine
//   poles have strengths 4e-6 to 0.46, some below the threshold. What
//   tells them apart is the samples: a doublet's removal leaves the fit to the samples within 10 x the (capped)
//   misfit, a genuine pole's does not.
// - So a removal is kept only if the refitted samples stay within the larger of the fit tolerance, 10 x the capped
//   misfit and the misfit itself (an under-resolved fit is not made worse). All candidates are first removed
//   together (overfitted noise leaves many doublets, and removing one leaves its neighbours' support needed); once
//   that is refused, they are removed one at a time, weakest first, and a refused candidate stays.
// - Both allowances stop at the sweep's consistency bound, kConsistentFactor x the fit tolerance (unless the misfit
//   is above it already). The misfit entering the cleanup is usually the greedy's stopping value, just below the fit
//   tolerance, not the data noise; an allowance of 100 x that let a removal push a consistent fit past the bound, and
//   the sweep then took exact derivative rows for inconsistent ones (all 17 Hermite drops in the trials at
//   tolerance 1e-3).
// Returns how many poles were removed.
int cleanup(const Problem& P, std::vector<int>& S, Model& model, const FitOptions& o) {
  if (model.size() < 2) return 0;
  const double width = P.f(P.M - 1) - P.f(0);
  const double misfit = sample_errors(P, model).maxCoeff();
  const double noise = std::min(misfit, o.noise_cap);
  const double resolution = std::max(o.doublet_tolerance, 100.0 * std::max(o.fit_tolerance, noise));
  const double consistent = o.fit_tolerance > 0 ? kConsistentFactor * o.fit_tolerance : kInf;
  const double allowed_single = std::max(misfit, std::min(std::max(o.fit_tolerance, 10.0 * noise), consistent));
  const double allowed_together = std::max(misfit, std::min(std::max(o.fit_tolerance, 100.0 * noise), consistent));
  int removed = 0;
  bool together = true;    // remove every candidate at once, until that is refused
  std::vector<int> kept;   // support points (sample indices) whose removal was refused
  while (S.size() >= 2) {
    const std::vector<cplx> poles = model.poles();
    const Eigen::MatrixXcd res = model.residues(poles);
    // candidates: (significance, position in S of the nearest support point not yet refused)
    std::vector<std::pair<double, int>> candidates;
    for (size_t p = 0; p < poles.size(); ++p) {
      double significance = 0;
      for (int k = 0; k < P.K; ++k)
        if (std::isfinite(P.scale(k))) significance = std::max(significance, std::abs(res(p, k)) / P.scale(k));
      if (!(significance < resolution * width)) continue;
      int a_near = 0;
      for (size_t a = 1; a < S.size(); ++a)
        if (std::abs(P.f(S[a]) - poles[p]) < std::abs(P.f(S[a_near]) - poles[p])) a_near = static_cast<int>(a);
      if (std::find(kept.begin(), kept.end(), S[a_near]) != kept.end()) continue;
      candidates.push_back({significance, a_near});
    }
    if (candidates.empty()) break;
    std::sort(candidates.begin(), candidates.end());
    std::vector<int> cleaned = S;
    int dropped = 0;
    if (together) {
      // for each candidate pole the nearest support point still in the set (as Chebfun's cleanup)
      for (size_t p = 0; p < poles.size() && cleaned.size() > 1; ++p) {
        double significance = 0;
        for (int k = 0; k < P.K; ++k)
          if (std::isfinite(P.scale(k))) significance = std::max(significance, std::abs(res(p, k)) / P.scale(k));
        if (!(significance < resolution * width)) continue;
        size_t a_near = 0;
        for (size_t a = 1; a < cleaned.size(); ++a)
          if (std::abs(P.f(cleaned[a]) - poles[p]) < std::abs(P.f(cleaned[a_near]) - poles[p])) a_near = a;
        cleaned.erase(cleaned.begin() + static_cast<long>(a_near));
        ++dropped;
      }
    } else {
      cleaned.erase(cleaned.begin() + candidates.front().second);
      dropped = 1;
    }
    Model refit = solve_weights(P, cleaned);
    const double refit_misfit = sample_errors(P, refit).maxCoeff();
    if (!(refit_misfit <= (together ? allowed_together : allowed_single))) {
      if (together) together = false;
      else kept.push_back(S[candidates.front().second]);
      continue;
    }
    S = cleaned;
    model = refit;
    removed += dropped;
  }
  return removed;
}

}  // namespace

// ------------------------------------------------------------------ Model

Eigen::VectorXcd Model::value(double f) const {
  const int K = static_cast<int>(values.cols());
  Eigen::VectorXcd N = Eigen::VectorXcd::Zero(K);
  cplx D = 0.0;
  for (int j = 0; j < size(); ++j) {
    if (f == support[j]) return values.row(j).transpose();
    const cplx c = weights(j) / (f - support[j]);
    N += c * values.row(j).transpose();
    D += c;
  }
  return N / D;
}

// With j the nearest support point and delta = f - f_j, r = (w_j F_j + delta n) / (w_j + delta d), where n and d are
// the sums over the other support points. Then
//   r' = [w_j (n - F_j d) + delta w_j (n' - F_j d') + delta^2 (n' d - n d')] / (w_j + delta d)^2,
// which has no cancellation near f_j (the quotient rule on N/D loses about eps f / delta there, 3e-2 at delta = 1e-15
// f) and is the support-point formula at delta = 0.
Eigen::VectorXcd Model::derivative(double f) const {
  const int K = static_cast<int>(values.cols());
  if (size() == 0) return Eigen::VectorXcd::Zero(K);
  int j = 0;
  for (int i = 1; i < size(); ++i)
    if (std::fabs(f - support[i]) < std::fabs(f - support[j])) j = i;
  const double delta = f - support[j];
  const Eigen::VectorXcd Fj = values.row(j).transpose();
  Eigen::VectorXcd n = Eigen::VectorXcd::Zero(K);
  Eigen::VectorXcd dn = Eigen::VectorXcd::Zero(K);
  cplx d = 0.0;
  cplx dd = 0.0;
  for (int i = 0; i < size(); ++i) {
    if (i == j) continue;
    const cplx c = weights(i) / (f - support[i]);
    const cplx dc = -c / (f - support[i]);
    n += c * values.row(i).transpose();
    dn += dc * values.row(i).transpose();
    d += c;
    dd += dc;
  }
  const cplx wj = weights(j);
  const cplx q = wj + delta * d;
  const Eigen::VectorXcd numerator = wj * (n - Fj * d) + delta * wj * (dn - Fj * dd) + delta * delta * (dn * d - n * dd);
  return numerator / (q * q);
}

std::vector<cplx> Model::poles() const {
  const int m = size();
  std::vector<cplx> out;
  if (m < 2) return out;
  // Zeros of D(f) = sum_j w_j / (f - f_j), in the scaled variable x = (f - c) / h (the support's centre and half
  // width; D's zeros do not move under the affine map, and E is then O(1) whatever the frequencies' magnitude): the
  // finite eigenvalues of the pencil (E, B),
  //   E = [0 w^T; 1 diag(x_j)],  B = diag(0, 1, ..., 1),
  // computed as lambda = s + 1/mu from the eigenvalues mu of (E - s B)^-1 B; the infinite eigenvalues are mu = 0.
  // The shift s lies in the upper half plane, away from the poles of a passive response (e^{-i omega t}).
  double lo = support[0];
  double hi = support[0];
  for (double x : support) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  const double c = 0.5 * (lo + hi);
  const double h = hi > lo ? 0.5 * (hi - lo) : std::max(std::fabs(lo), 1.0);
  const double wnorm = weights.cwiseAbs().maxCoeff();
  Eigen::MatrixXcd E = Eigen::MatrixXcd::Zero(m + 1, m + 1);
  Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(m + 1, m + 1);
  for (int j = 0; j < m; ++j) {
    E(0, j + 1) = weights(j) / wnorm;
    E(j + 1, 0) = 1.0;
    E(j + 1, j + 1) = (support[j] - c) / h;
    B(j + 1, j + 1) = 1.0;
  }
  for (double height : {1.0, 2.7, 0.37}) {   // a second or third shift if s is (nearly) a pole itself
    const cplx s(0.0, height);
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(E - s * B);
    if (!lu.isInvertible()) continue;
    const Eigen::MatrixXcd T = lu.solve(B);
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(T, false);
    if (es.info() != Eigen::Success) continue;
    for (int i = 0; i < m + 1; ++i) {
      const cplx mu = es.eigenvalues()(i);
      if (std::abs(mu) > 1e-8) out.push_back(c + h * (s + 1.0 / mu));   // poles within 1e8 half widths
    }
    std::sort(out.begin(), out.end(), [](cplx a, cplx b) { return a.real() < b.real(); });
    return out;
  }
  return out;
}

Eigen::MatrixXcd Model::residues(const std::vector<cplx>& poles) const {
  const int K = static_cast<int>(values.cols());
  Eigen::MatrixXcd res(static_cast<int>(poles.size()), K);
  for (size_t p = 0; p < poles.size(); ++p) {
    Eigen::VectorXcd N = Eigen::VectorXcd::Zero(K);
    cplx dD = 0.0;
    for (int j = 0; j < size(); ++j) {
      const cplx c = weights(j) / (poles[p] - support[j]);
      N += c * values.row(j).transpose();
      dD -= c / (poles[p] - support[j]);
    }
    res.row(static_cast<int>(p)) = (N / dD).transpose();
  }
  return res;
}

// ------------------------------------------------------------------ fit

double relative_error(const Eigen::VectorXcd& a, const Eigen::VectorXcd& ref, const Eigen::VectorXd& floor) {
  double err = 0;
  for (int k = 0; k < ref.size(); ++k) {
    const double e = std::abs(a(k) - ref(k)) / std::max(std::abs(ref(k)), floor(k));
    if (std::isnan(e)) return kInf;
    err = std::max(err, e);
  }
  return err;
}

namespace {

// The weighted problem on samples whose frequencies are already the normalized x (see fit()).
Problem make_problem(const std::vector<Sample>& samples, const FitOptions& o) {
  Problem P;
  P.samples = &samples;
  P.M = static_cast<int>(samples.size());
  P.K = static_cast<int>(samples[0].value.size());
  // floors per group: a fraction of the median over samples of the group's largest magnitude; a group that is
  // zero throughout (or not finite) is inactive. An output with a floor pair (a, b) uses
  // floor_fraction x sqrt(median |F_a| x median |F_b|) instead (an off-diagonal Z_jk against its ports' Z_jj, Z_kk),
  // not below 1e-6 of its group's
  int groups = 0;
  for (int g : o.group) groups = std::max(groups, g + 1);
  std::vector<double> group_size(groups, 0.0);
  for (int g = 0; g < groups; ++g) {
    std::vector<double> largest;
    for (const Sample& s : samples) {
      double x = 0;
      for (int k = 0; k < P.K; ++k)
        if (o.group[k] == g) x = std::max(x, std::abs(s.value(k)));
      largest.push_back(x);
    }
    group_size[g] = median(largest);
  }
  std::vector<double> output_size(P.K, 0.0);   // median over samples of |F_k|
  for (int k = 0; k < P.K; ++k) {
    std::vector<double> mags;
    for (const Sample& s : samples) mags.push_back(std::abs(s.value(k)));
    output_size[k] = median(mags);
  }
  P.floor.resize(P.K);
  P.scale.resize(P.K);
  for (int k = 0; k < P.K; ++k) {
    const double G = group_size[o.group[k]];
    P.floor(k) = std::isfinite(G) && G > 0 ? o.floor_fraction * G : kInf;
    if (std::isfinite(P.floor(k)) && static_cast<int>(o.floor_pair.size()) == P.K && o.floor_pair[k].first >= 0) {
      const double pair = std::sqrt(output_size[o.floor_pair[k].first] * output_size[o.floor_pair[k].second]);
      P.floor(k) = std::max(o.floor_fraction * pair, 1e-6 * G);
    }
    P.scale(k) = std::max(output_size[k], P.floor(k));
  }
  std::vector<double> h(P.M, 0.0);   // distance to the nearest other sample
  for (int i = 0; i < P.M; ++i) {
    double d = kInf;
    if (i > 0) d = std::min(d, samples[i].f - samples[i - 1].f);
    if (i + 1 < P.M) d = std::min(d, samples[i + 1].f - samples[i].f);
    h[i] = std::isfinite(d) ? d : 0.0;
  }
  P.value_weight = Eigen::MatrixXd::Zero(P.M, P.K);
  P.derivative_weight = Eigen::MatrixXd::Zero(P.M, P.K);
  for (int i = 0; i < P.M; ++i)
    for (int k = 0; k < P.K; ++k) {
      if (!std::isfinite(P.floor(k))) continue;
      const double magnitude = std::max(std::abs(P.F(i, k)), P.floor(k));
      P.value_weight(i, k) = 1.0 / magnitude;
      const cplx G = samples[i].derivative.size() == P.K ? P.G(i, k) : cplx(std::nan(""), 0.0);
      if (!o.hermite[k] || !std::isfinite(G.real()) || !std::isfinite(G.imag())) continue;
      // the derivative's row weight: a length over which F changes by itself, at most the sample spacing
      const double local = std::abs(G) > 0 ? magnitude / std::abs(G) : kInf;
      P.derivative_weight(i, k) = std::min(h[i], local) / magnitude;
    }
  return P;
}

// The samples in the normalized frequency x = (f - c) / s of fit() (the derivatives times s).
std::vector<Sample> normalized(const std::vector<Sample>& samples, double& c, double& s) {
  const double lo = samples.front().f;
  const double hi = samples.back().f;
  c = 0.5 * (lo + hi);
  s = hi > lo ? 0.5 * (hi - lo) : std::max(std::fabs(lo), 1.0);
  std::vector<Sample> scaled = samples;
  for (Sample& x : scaled) {
    x.f = (x.f - c) / s;
    if (x.derivative.size() > 0) x.derivative *= s;
  }
  return scaled;
}

// The fit itself, on samples whose frequencies are already the normalized x (see fit()).
Fit fit_normalized(const std::vector<Sample>& samples, const FitOptions& o) {
  const Problem P = make_problem(samples, o);

  // greedy (AAA): add the worst-fitted sample as a support point until every sample is fitted
  Eigen::VectorXcd mean = Eigen::VectorXcd::Zero(P.K);
  for (const Sample& s : samples) mean += s.value;
  mean /= static_cast<double>(P.M);
  Eigen::VectorXd err(P.M);
  for (int i = 0; i < P.M; ++i) {
    err(i) = 0;
    for (int k = 0; k < P.K; ++k) err(i) = std::max(err(i), P.value_weight(i, k) * std::abs(P.F(i, k) - mean(k)));
  }
  std::vector<int> S;
  std::vector<char> in_support(P.M, 0);
  Model model;
  while (static_cast<int>(S.size()) < P.M) {
    int next = -1;
    for (int i = 0; i < P.M; ++i)
      if (!in_support[i] && (next < 0 || err(i) > err(next) || std::isnan(err(i)))) next = i;
    S.push_back(next);
    in_support[next] = 1;
    model = solve_weights(P, S);
    err = sample_errors(P, model);
    if (err.maxCoeff() <= o.fit_tolerance) break;
  }
  Fit out;
  out.floor = P.floor;
  out.scale = P.scale;
  out.doublets_removed = cleanup(P, S, model, o);
  out.model = model;
  out.support_order = S;
  out.sample_error = sample_errors(P, model).maxCoeff();
  // the comparison model: the same samples with the last-chosen support point dropped
  if (S.size() > 1) {
    std::vector<int> fewer(S.begin(), S.end() - 1);
    Model lower = solve_weights(P, fewer);
    out.doublets_removed += cleanup(P, fewer, lower, o);
    out.lower = lower;
  } else if (P.M > 1) {
    // One support point fits every sample (a constant response, such as a matched line's Z): the comparison is
    // AAA's starting approximation, the mean of the samples, as a one-point model. Without it the estimate would be
    // undefined and such a sweep could never converge (e.g. the robin_absorbing case of the gap validation).
    Model lower;
    lower.support = model.support;
    lower.weights = Eigen::VectorXcd::Ones(1);
    lower.values = mean.transpose();
    out.lower = lower;
  }
  return out;
}

}  // namespace

// The fit runs in x = (f - c) / s, c and s the samples' centre and half span, with the derivatives times s. The
// barycentric weights do not change under the affine map, so the model maps back by f_j = c + s x_j alone (each
// support point is set back to its sample's own frequency). In f itself the support points' derivative rows carry an
// extra factor of the spacing, so the rows' relative weights would depend on the unit: in Hz the same trials needed up
// to 2.6 x the solves and stopped falsely up to 9 x the tolerance.
Fit fit(const std::vector<Sample>& samples, const FitOptions& o) {
  AXF_REQUIRE(!samples.empty(), "internal: a rational fit needs at least one sample");
  const int K = static_cast<int>(samples[0].value.size());
  AXF_REQUIRE(static_cast<int>(o.group.size()) == K && static_cast<int>(o.hermite.size()) == K,
              "internal: rational fit options need one group and one Hermite flag per output");
  for (size_t i = 1; i < samples.size(); ++i)
    AXF_REQUIRE(samples[i].f > samples[i - 1].f, "internal: rational fit samples must have increasing frequencies");
  double c = 0;
  double s = 1;
  const std::vector<Sample> scaled = normalized(samples, c, s);
  Fit out = fit_normalized(scaled, o);
  // back to f: each support point is exactly its sample's frequency
  auto restore = [&](Model& model) {
    for (double& f : model.support)
      for (size_t i = 0; i < scaled.size(); ++i)
        if (scaled[i].f == f) {
          f = samples[i].f;
          break;
        }
  };
  restore(out.model);
  restore(out.lower);
  return out;
}

Eigen::VectorXd estimated_error(const Fit& fit, const std::vector<double>& f, const Model* previous) {
  Eigen::VectorXd e(static_cast<int>(f.size()));
  for (size_t i = 0; i < f.size(); ++i) {
    if (fit.lower.size() == 0) {
      e(static_cast<int>(i)) = kInf;
      continue;
    }
    const Eigen::VectorXcd r = fit.model.value(f[i]);
    double x = relative_error(fit.lower.value(f[i]), r, fit.floor);
    if (previous && previous->size() > 0) x = std::max(x, relative_error(previous->value(f[i]), r, fit.floor));
    e(static_cast<int>(i)) = x;
  }
  return e;
}

std::vector<cplx> spurious_comparison_poles(const Fit& fit, double band_width) {
  std::vector<cplx> out;
  if (fit.lower.size() < 2) return out;
  const std::vector<cplx> model_poles = fit.model.size() > 1 ? fit.model.poles() : std::vector<cplx>();
  for (const cplx& q : fit.lower.poles()) {
    if (!(q.imag() > 1e-9 * std::abs(q)) || q.imag() > kSpuriousReach * band_width) continue;
    bool shared = false;
    for (const cplx& m : model_poles) shared = shared || std::abs(m - q) < q.imag();
    if (!shared) out.push_back(q);
  }
  return out;
}

Eigen::VectorXd spurious_terms(const Fit& fit, const std::vector<cplx>& spurious, const std::vector<double>& f) {
  Eigen::VectorXd e = Eigen::VectorXd::Zero(static_cast<int>(f.size()));
  if (spurious.empty() || fit.lower.size() == 0) return e;
  const Eigen::MatrixXcd res = fit.lower.residues(spurious);
  for (size_t i = 0; i < f.size(); ++i) {
    const Eigen::VectorXcd r = fit.model.value(f[i]);
    for (int k = 0; k < r.size(); ++k) {
      double term = 0;
      for (size_t q = 0; q < spurious.size(); ++q) term += std::abs(res(static_cast<int>(q), k) / (f[i] - spurious[q]));
      const double x = term / std::max(std::abs(r(k)), fit.floor(k));
      if (std::isfinite(x)) e(static_cast<int>(i)) = std::max(e(static_cast<int>(i)), x);
    }
  }
  return e;
}

std::vector<double> band_grid(double f_min, double f_max, int n, bool log_spacing) {
  std::vector<double> g(std::max(n, 2));
  const int last = static_cast<int>(g.size()) - 1;
  for (int i = 0; i <= last; ++i) {
    const double t = double(i) / last;
    g[i] = log_spacing ? f_min * std::pow(f_max / f_min, t) : f_min + t * (f_max - f_min);
  }
  g.front() = f_min;
  g.back() = f_max;
  return g;
}

// ------------------------------------------------------------------ pole classes

std::vector<std::vector<cplx>> leave_one_out_matches(const std::vector<cplx>& poles, const std::vector<size_t>& targets,
                                                     const std::vector<Sample>& samples, const FitOptions& options,
                                                     int& refits) {
  // a refit's pole matches a target pole within half the distance to the target's nearest neighbour among `poles`
  std::vector<double> reach(targets.size(), kInf);
  for (size_t b = 0; b < targets.size(); ++b)
    for (size_t q = 0; q < poles.size(); ++q)
      if (q != targets[b]) reach[b] = std::min(reach[b], 0.5 * std::abs(poles[q] - poles[targets[b]]));
  std::vector<std::vector<cplx>> matched(targets.size());
  refits = 0;
  if (samples.size() < 4) return matched;
  for (size_t drop = 0; drop < samples.size(); ++drop) {
    std::vector<Sample> rest;
    for (size_t i = 0; i < samples.size(); ++i)
      if (i != drop) rest.push_back(samples[i]);
    const Fit refit = fit(rest, options);
    ++refits;
    const std::vector<cplx> qs = refit.model.size() > 1 ? refit.model.poles() : std::vector<cplx>();
    for (size_t b = 0; b < targets.size(); ++b) {
      const cplx p = poles[targets[b]];
      double best = kInf;
      cplx at = p;
      for (const cplx& q : qs)
        if (std::abs(q - p) < best) {
          best = std::abs(q - p);
          at = q;
        }
      if (best < reach[b]) matched[b].push_back(at);
    }
  }
  return matched;
}

// The passivity line Im p > 1e-12 |p| alone sits inside the fit's own resolution of Im p, which is 1e-12 to 1e-9
// relative depending on the fit's conditioning: on the four-port gap case (every solve a support point) the strongest
// lossless resonance was non_passive in about 30 % of noise realizations and had a spurious Q ~ 1e11 in another 30 %
// So Im p is judged against its own uncertainty: the
// leave-one-out refits (how much the pole depends on the solves: a jackknife robust to one solve, and the offset of
// their mean from the pole) and the spread over refits under the solves' own noise (how much it depends on their last digits),
// in quadrature. Either alone missed cases on FEM test samples under 1e-10 noise: the leave-one-out spread of
// a cavity mode was 1.3e-5 Hz while the noise moved its Im p by 3e-4 Hz. The comparison model r- is not used: it
// locates poles about 1e6 times worse than the model and would hide genuine Q of 1e9-1e11.
std::vector<PoleInfo> classify_poles(const Fit& fit, const std::vector<Sample>& samples, const FitOptions& options,
                                     const PoleOptions& po) {
  std::vector<PoleInfo> out;
  if (fit.model.size() < 2) return out;
  const std::vector<cplx> poles = fit.model.poles();
  const Eigen::MatrixXcd residues = fit.model.residues(poles);
  const int K = static_cast<int>(residues.cols());
  const double ratio = std::pow(po.f_max / po.f_min, 1.0 / (po.candidates - 1));
  std::vector<size_t> in_band;
  for (size_t p = 0; p < poles.size(); ++p)
    if (poles[p].real() >= po.f_min && poles[p].real() <= po.f_max) in_band.push_back(p);
  if (in_band.empty()) return out;
  // a refit's pole matches a model pole within half the distance to the model pole's nearest neighbour
  std::vector<double> reach(in_band.size(), kInf);
  for (size_t b = 0; b < in_band.size(); ++b)
    for (size_t q = 0; q < poles.size(); ++q)
      if (q != in_band[b]) reach[b] = std::min(reach[b], 0.5 * std::abs(poles[q] - poles[in_band[b]]));
  // leave-one-out refits: for each in-band pole, the matching poles
  int refits = 0;
  const std::vector<std::vector<cplx>> matched = leave_one_out_matches(poles, in_band, samples, options, refits);
  // refits under the solves' noise: deviations from the model's pole. The error of a complex-arithmetic solve has a
  // phase, which carries Im p: a real factor alone missed the scatter of a Q 1e11 line's Im p by 900 x.
  // An output whose every sample lies on one axis (all real or
  // all imaginary: a lossless model's Z and Psi, solved in real arithmetic) keeps it, with a real factor.
  auto on_axis = [&](auto get) {
    bool real = true;
    bool imaginary = true;
    for (const Sample& x : samples) {
      const cplx v = get(x);
      if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) continue;
      real = real && v.imag() == 0.0;
      imaginary = imaginary && v.real() == 0.0;
    }
    return real || imaginary;
  };
  std::vector<char> value_on_axis(K, 0);
  std::vector<char> derivative_on_axis(K, 0);
  for (int k = 0; k < K; ++k) {
    value_on_axis[k] = on_axis([k](const Sample& x) { return x.value(k); }) ? 1 : 0;
    derivative_on_axis[k] =
        on_axis([k](const Sample& x) { return x.derivative.size() > k ? x.derivative(k) : cplx(0.0, 0.0); }) ? 1 : 0;
  }
  std::vector<std::vector<cplx>> perturbed(in_band.size());
  if (po.noise > 0 && po.noise_refits > 0) {
    Gauss gauss(20261029);
    auto perturb = [&](cplx v, bool axis) {
      if (axis) return v * (1.0 + po.noise * gauss());
      const double a = gauss();
      const double b = gauss();
      return v * (1.0 + po.noise * cplx(a, b) / std::sqrt(2.0));
    };
    for (int r = 0; r < po.noise_refits; ++r) {
      std::vector<Sample> noisy = samples;
      for (Sample& x : noisy) {
        for (int k = 0; k < x.value.size(); ++k) x.value(k) = perturb(x.value(k), value_on_axis[k] != 0);
        for (int k = 0; k < x.derivative.size(); ++k)
          x.derivative(k) = perturb(x.derivative(k), derivative_on_axis[k] != 0);
      }
      const Fit refit = axf::rational::fit(noisy, options);
      const std::vector<cplx> qs = refit.model.size() > 1 ? refit.model.poles() : std::vector<cplx>();
      for (size_t b = 0; b < in_band.size(); ++b) {
        const cplx p = poles[in_band[b]];
        double best = kInf;
        cplx at = p;
        for (const cplx& q : qs)
          if (std::abs(q - p) < best) {
            best = std::abs(q - p);
            at = q;
          }
        if (best < reach[b]) perturbed[b].push_back(at - p);
      }
    }
  }
  for (size_t b = 0; b < in_band.size(); ++b) {
    const size_t p = in_band[b];
    PoleInfo info;
    info.p = poles[p];
    for (int k = 0; k < K; ++k)
      if (std::isfinite(fit.scale(k))) info.strength = std::max(info.strength, std::abs(residues(p, k)) / fit.scale(k));
    const double step = po.log_spacing ? info.p.real() * (ratio - 1.0) : (po.f_max - po.f_min) / (po.candidates - 1);
    info.visible = info.strength / std::max(std::fabs(info.p.imag()), step);
    info.refits = refits;
    info.reproduced = static_cast<int>(matched[b].size());
    const int n = info.reproduced;
    info.sigma_re = kInf;
    info.sigma_im = kInf;
    if (n >= 2) {
      // The leave-one-out part: the jackknife over every refit that has the pole. Dropping the refit farthest from
      // the refits' median would stop the one refit without the only solve near a narrow pole from inflating sigma
      // (up to 4000 x the actual scatter of Im p). But that refit is also the only one that shows a
      // bad near-pole solve, and the FEM's errors are largest there (up to 24 x pole_noise): with such errors the
      // trimmed sigma left 8-15 % of resolved Q beyond 3 sigma and flipped lossless poles, against <= 0.7 % for the
      // plain jackknife. The inflation is conservative, so the plain jackknife is used.
      cplx mean = 0.0;
      for (const cplx& q : matched[b]) mean += q;
      mean /= static_cast<double>(n);
      double sre = 0;
      double sim = 0;
      for (const cplx& q : matched[b]) {
        sre += (q.real() - mean.real()) * (q.real() - mean.real());
        sim += (q.imag() - mean.imag()) * (q.imag() - mean.imag());
      }
      const double jack = (n - 1.0) / n;
      // and the model's own offset from where its refits put the pole: the jackknife measures the refits' spread about
      // their mean, so a model that is itself the outlier (every refit agreeing, the full fit off) would look certain.
      // In the trials such a fit put a lossless pole 258 sigma above the axis (test_rational_fit case 17)
      info.sigma_re = std::hypot(std::sqrt(jack * sre), mean.real() - info.p.real());
      info.sigma_im = std::hypot(std::sqrt(jack * sim), mean.imag() - info.p.imag());
    }
    if (std::isfinite(info.sigma_im) && !perturbed[b].empty()) {
      double nre = 0;
      double nim = 0;
      for (const cplx& d : perturbed[b]) {
        nre += d.real() * d.real();
        nim += d.imag() * d.imag();
      }
      const double m = static_cast<double>(perturbed[b].size());
      info.sigma_re = std::hypot(info.sigma_re, std::sqrt(nre / m));
      info.sigma_im = std::hypot(info.sigma_im, std::sqrt(nim / m));
    }
    const double line = std::max(po.real_axis * std::abs(info.p), po.significance * info.sigma_im);
    info.non_passive = info.p.imag() > line;
    info.lossy = -info.p.imag() > line;
    info.Q = info.p.imag() != 0.0 ? info.p.real() / (2 * std::fabs(info.p.imag())) : kInf;
    info.Q_sigma = info.lossy ? info.Q * info.sigma_im / std::fabs(info.p.imag()) : kInf;
    // the true |Im p| can be up to |Im p| + 3 sigma_Im, not only the line: Re p / (2 line) exceeded the true Q by up to
    // 1.21
    const double widest = std::fabs(info.p.imag()) + po.significance * info.sigma_im;
    info.Q_min = info.p.real() / (2 * std::max(po.real_axis * std::abs(info.p), widest));
    const bool resolved = refits >= 3 && 2 * n >= refits &&
                          std::hypot(info.sigma_re, info.sigma_im) <= std::max(std::fabs(info.p.imag()), step);
    if (!resolved) info.cls = "unresolved";
    else if (info.non_passive) info.cls = "non_passive";
    else if (!(info.visible >= po.tolerance)) info.cls = "negligible";
    else if (info.Q < po.min_Q) info.cls = "background";
    else info.cls = "resonance";
    out.push_back(info);
  }
  return out;
}

// ------------------------------------------------------------------ adaptive sweep

// After a converged sweep: Froissart doublets the cleanup kept. The cleanup's own allowance stops at 10 x the fit
// tolerance, the consistency bound, and kept such a doublet in trial 88 of the trials (tolerance 1e-5): its needle,
// too narrow for any candidate, put the converged model 24-44 x the tolerance off.
// Candidates: in-band poles on the real axis (|Im p| below a quarter grid step r), invisible one candidate step away
// (strength / max(|Im p|, step) below the tolerance), with a needle (|res| / |Im p| against the model next to them)
// that reaches the tolerance. For each, the support point nearest to it is dropped and the weights are solved again
// at that fixed degree. The candidate is removed only if that refit:
//  - has lost it (no pole within r of it): a doublet tied to that support point disappears with it, a genuine mode
//    that the other solves determine does not. A majority of full leave-one-out refits would delete
//    well-located genuine weak modes (a full refit stops at a lower degree and drops a marginal pole however well it
//    is located), and a retry on the next-nearest support point would keep the evidence-carrying solve as a support
//    point, which defeats the solves check;
//  - reproduces the solves to kInvisibleFraction x the tolerance, in the fit's measure and against the output's
//    scale;
//  - changes on the candidate grid, outside r of every pole on the real axis, by at most
//    max(kInvisibleFraction x tol, min(kInvisibleGridFraction x tol, tol - the sweep's estimate there)), in the fit's
//    measure: 0.1 x tol, plus up to 0.15 x tol of the estimate's own slack;
//  - keeps every other pole on the real axis, one to one, moved by at most min(kInvisibleFraction x tol x r,
//    tol x 1e-6 |p|): near a lossless pole a shift dp puts the model dp / |f - p| off, so the bound keeps that below
//    the tolerance down to 1e-6 of the pole (0.1 x tol x r alone is 8.7e-9 relative at the default grid and turned
//    correct models wrong at 1e-6-1e-5 of a strong pole);
//  - adds no pole on the real axis whose own needle reaches the tolerance.
// A refused candidate stays. Returns how many were removed.
int remove_invisible_poles(Fit& fit, const std::vector<Sample>& samples, const FitOptions& fo,
                           const AdaptiveOptions& o, const Model* previous) {
  if (fit.model.size() < 2 || fit.support_order.size() != static_cast<size_t>(fit.model.size())) return 0;
  double c = 0;
  double s = 1;
  const std::vector<Sample> scaled = normalized(samples, c, s);
  const Problem P = make_problem(scaled, fo);
  const double ratio = std::pow(o.f_max / o.f_min, 1.0 / (o.candidates - 1));
  auto step_at = [&](double f) { return o.log_spacing ? f * (ratio - 1.0) : (o.f_max - o.f_min) / (o.candidates - 1); };
  const std::vector<double> grid = band_grid(o.f_min, o.f_max, o.candidates, o.log_spacing);
  const double allowed = kInvisibleFraction * o.tolerance;
  // A change of the model, in the larger of the fit's measure (relative to max(|r|, floor)) and the output's scale.
  // The fit's measure alone is diluted next to a strong mode, where |r| is 1e2-1e3 x the scale.
  auto change = [&](const Eigen::VectorXcd& a, const Eigen::VectorXcd& ref) {
    double e = relative_error(a, ref, fit.floor);
    for (int k = 0; k < P.K; ++k)
      if (std::isfinite(P.scale(k)) && P.scale(k) > 0) e = std::max(e, std::abs(a(k) - ref(k)) / P.scale(k));
    return std::isnan(e) ? kInf : e;
  };
  std::vector<int> S = fit.support_order;
  Model current = make_model(P, S, fit.model.weights);   // the weights do not change under the affine map
  const Model original = current;
  auto near_real_in_band = [&](const cplx& q) {   // q in x
    const double f = c + s * q.real();
    return f > o.f_min && f < o.f_max && s * std::fabs(q.imag()) < 0.25 * step_at(f);
  };
  // its needle: the term res / (i Im p) at the pole against the model next to it (at Re p +- r), in the fit's measure
  auto needle_of = [&](const Model& m, const std::vector<cplx>& poles, const Eigen::MatrixXcd& res, size_t p) {
    const double r = 0.25 * step_at(c + s * poles[p].real()) / s;   // in x
    const Eigen::VectorXcd below = m.value(poles[p].real() - r);
    const Eigen::VectorXcd above = m.value(poles[p].real() + r);
    double needle = 0;
    for (int k = 0; k < P.K; ++k) {
      if (!std::isfinite(P.floor(k))) continue;
      const double background = std::max(std::min(std::abs(below(k)), std::abs(above(k))), P.floor(k));
      needle = std::max(needle, std::abs(res(p, k)) / (std::fabs(poles[p].imag()) * background));
    }
    return needle;
  };
  const std::vector<cplx> original_poles = original.poles();
  const Eigen::MatrixXcd original_res = original.residues(original_poles);
  std::vector<size_t> candidates;
  for (size_t p = 0; p < original_poles.size(); ++p) {
    if (!near_real_in_band(original_poles[p])) continue;
    const double f = c + s * original_poles[p].real();
    const double width = s * std::fabs(original_poles[p].imag());
    double strength = 0;   // [Hz]: the residue in f is s x the residue in x
    for (int k = 0; k < P.K; ++k)
      if (std::isfinite(P.scale(k))) strength = std::max(strength, s * std::abs(original_res(p, k)) / P.scale(k));
    if (!(strength / std::max(width, step_at(f)) < o.tolerance)) continue;
    if (!(needle_of(original, original_poles, original_res, p) >= o.tolerance)) continue;
    candidates.push_back(p);
  }
  if (candidates.empty()) return 0;
  // the grid's allowance: 0.1 x tol, plus up to 0.15 x tol of the sweep's own slack (tol - its estimate) there
  const Eigen::VectorXd estimate = estimated_error(fit, grid, previous);
  std::vector<double> allowed_grid(grid.size(), allowed);
  for (size_t i = 0; i < grid.size(); ++i) {
    const double slack = std::isfinite(estimate(static_cast<int>(i))) ? o.tolerance - estimate(static_cast<int>(i)) : 0;
    allowed_grid[i] = std::max(allowed, std::min(kInvisibleGridFraction * o.tolerance, slack));
  }
  // the original model's in-band poles on the real axis [x], each with its r [x] and its allowed shift [x]
  struct NearReal {
    cplx q;
    double r;
    double shift;
  };
  std::vector<NearReal> near_real;
  for (const cplx& q : original_poles)
    if (near_real_in_band(q)) {
      const double r = 0.25 * step_at(c + s * q.real()) / s;
      near_real.push_back({q, r, std::min(allowed * r, o.tolerance * 1e-6 * std::abs(c + s * q) / s)});
    }
  std::vector<cplx> removed_poles;   // [x]
  int removed = 0;
  for (size_t cand : candidates) {
    const cplx target = original_poles[cand];
    const double r_target = 0.25 * step_at(c + s * target.real()) / s;
    // the pole in the current model (earlier removals move it slightly)
    const std::vector<cplx> poles = current.poles();
    int p_now = -1;
    for (size_t p = 0; p < poles.size(); ++p)
      if (std::abs(poles[p] - target) < r_target && (p_now < 0 || std::abs(poles[p] - target) <
                                                                     std::abs(poles[p_now] - target)))
        p_now = static_cast<int>(p);
    if (p_now < 0) continue;   // gone with an earlier removal
    int nearest = 0;
    for (size_t a = 1; a < S.size(); ++a)
      if (std::abs(P.f(S[a]) - poles[p_now]) < std::abs(P.f(S[nearest]) - poles[p_now])) nearest = static_cast<int>(a);
    std::vector<int> fewer = S;
    fewer.erase(fewer.begin() + nearest);
    const Model refit = solve_weights(P, fewer);
    const std::vector<cplx> qs = refit.poles();
    // the refit at fixed degree without that support point must have lost the candidate
    bool ok = true;
    for (const cplx& q : qs) ok = ok && !(std::abs(q - target) < r_target);
    // the solves: in the fit's measure and against the output's scale
    ok = ok && sample_errors(P, refit).maxCoeff() <= allowed;
    for (int i = 0; i < P.M && ok; ++i) ok = change(refit.value(P.f(i)), (*P.samples)[i].value) <= allowed;
    // the candidate grid, outside r of every real-axis pole (within r the sweep itself does not look; those poles
    // are checked one by one below), in the fit's measure. Against the output's scale the grid would refuse any
    // removal next to a strong mode: a shift of the strong pole well inside its allowance changed the model 0.4 x the
    // tolerance in that measure 100 kHz from it (trial 88)
    for (size_t i = 0; i < grid.size() && ok; ++i) {
      const double x = (grid[i] - c) / s;
      bool near_pole = false;
      for (const auto& nr : near_real) near_pole = near_pole || std::fabs(x - nr.q.real()) < nr.r;
      if (near_pole) continue;
      ok = relative_error(refit.value(x), original.value(x), fit.floor) <= allowed_grid[i];
    }
    // the poles on the real axis, matched one to one between the model and the refit (each within its r): every one
    // but the removed ones stays, moved by at most its allowed shift; a pole of the refit without a partner is a new
    // needle, accepted only below the tolerance
    if (ok) {
      const Eigen::MatrixXcd qres = refit.residues(qs);
      std::vector<char> taken(qs.size(), 0);
      for (const auto& nr : near_real) {
        if (!ok) break;
        if (std::abs(nr.q - target) < r_target) continue;
        bool gone = false;
        for (const cplx& z : removed_poles) gone = gone || std::abs(nr.q - z) < nr.r;
        if (gone) continue;
        int best = -1;
        for (size_t j = 0; j < qs.size(); ++j)
          if (!taken[j] && std::abs(qs[j] - nr.q) < nr.r &&
              (best < 0 || std::abs(qs[j] - nr.q) < std::abs(qs[best] - nr.q)))
            best = static_cast<int>(j);
        ok = best >= 0 && std::abs(qs[best] - nr.q) <= nr.shift;
        if (best >= 0) taken[best] = 1;
      }
      for (size_t j = 0; j < qs.size() && ok; ++j)
        if (!taken[j] && near_real_in_band(qs[j])) ok = needle_of(refit, qs, qres, j) < o.tolerance;
    }
    if (!ok) continue;
    S = fewer;
    current = refit;
    removed_poles.push_back(target);
    ++removed;
  }
  if (removed == 0) return 0;
  fit.sample_error = sample_errors(P, current).maxCoeff();
  fit.support_order = S;
  fit.model = current;
  for (size_t a = 0; a < S.size(); ++a) fit.model.support[a] = samples[S[a]].f;
  return removed;
}

AdaptiveResult adaptive_sweep(const AdaptiveOptions& o, const FitOptions& fo,
                              const std::function<std::optional<Sample>(double)>& solve) {
  AXF_REQUIRE(o.f_max > o.f_min && o.f_min > 0, "internal: the adaptive sweep needs 0 < f_min < f_max");
  AdaptiveResult out;
  std::vector<Sample> samples;
  auto attempt = [&](double f) {
    // a frequency already attempted (only possible if an agreed decision overrode this rank's) is not solved again
    const bool repeated = std::find(out.attempted.begin(), out.attempted.end(), f) != out.attempted.end();
    out.attempted.push_back(f);
    if (repeated) {
      out.failed.push_back(f);
      return std::optional<Sample>();
    }
    std::optional<Sample> s = solve(f);
    if (!s) {
      out.failed.push_back(f);
      return s;
    }
    s->f = f;
    auto at = std::lower_bound(samples.begin(), samples.end(), f, [](const Sample& a, double x) { return a.f < x; });
    samples.insert(at, *s);
    return s;
  };
  for (double f : o.initial) attempt(f);
  if (samples.empty()) {   // nothing to fit, and more solves of a setup that failed everywhere would not help
    out.stop_reason = "no usable solve among the initial points";
    out.estimated_error = kInf;
    out.last_prediction_error = kInf;
    return out;
  }
  FitOptions current = fo;   // Hermite rows are dropped if they turn out inconsistent with the values (below)
  out.hermite = std::find(fo.hermite.begin(), fo.hermite.end(), 1) != fo.hermite.end();

  const std::vector<double> grid = band_grid(o.f_min, o.f_max, o.candidates, o.log_spacing);
  const double ratio = std::pow(o.f_max / o.f_min, 1.0 / (o.candidates - 1));
  // a quarter of the grid step at f: candidates closer than this to a solved frequency or to a pole on the real axis
  // are not taken
  auto quarter_step = [&](double f) {
    return o.log_spacing ? 0.25 * f * (ratio - 1.0) : 0.25 * (o.f_max - o.f_min) / (o.candidates - 1);
  };
  double prediction_error = kInf;
  double predicted_at = 0;
  int agreed = 0;                  // consecutive newest solves within the tolerance of their predictions
  int tight = 0;                   // consecutive newest solves within tolerance / 100 of their predictions
  int derivative_disagreements = 0;   // consecutive newest solves whose derivative disagrees (below)
  double derivative_error = 0;     // the newest solve's derivative against the model's prediction
  std::optional<Model> previous;   // the model fitted before the newest solve
  bool solved_since_fit = false;   // a solve was added since the last fit (the initial ones excepted)
  while (true) {
    AdaptiveStep step;
    step.solves = static_cast<int>(out.attempted.size());
    step.samples = static_cast<int>(samples.size());
    step.prediction_error = prediction_error;
    step.predicted_at = predicted_at;
    std::vector<double> candidates;
    std::vector<char> solvable;
    Eigen::VectorXd estimate;
    // the comparison model's spurious poles: above the real axis (a passive response has none), at most kSpuriousReach
    // band widths above it, and with no pole of the model within their width; their terms in r- are r-'s artifact,
    // not r's error
    std::vector<cplx> spurious_comparison;
    Eigen::VectorXd unexcused;   // the estimate where those terms do not account for it (the closure's gate)
    if (!samples.empty()) {
      if (solved_since_fit && out.fit.model.size() > 0) previous = out.fit.model;
      solved_since_fit = false;
      out.fit = fit(samples, current);
      step.support_points = out.fit.model.size();
      step.sample_fit_error = out.fit.sample_error;
      step.doublets_removed = out.fit.doublets_removed;
      // candidates: the grid, and around every pole of the model in the band (r: a quarter grid step there)
      //  - a resolved pole (|Im p| >= r): its peak and half-power points;
      //  - a pole on the real axis (|Im p| < r) that the comparison model confirms (a pole within tolerance x r): the
      //    points at r, and nothing closer, where the relative error of a lossless resonance is not controlled;
      //  - an unconfirmed one (a spurious pole on the real axis, or a resonance only just found): points from r/64
      //    to r, so that the estimate sees its spike.
      // No frequency is solved closer than r to a pole on the real axis (a lossless resonance is singular there) or
      // than a quarter grid step to a frequency already attempted.
      // a pole on the real axis counts as confirmed if the comparison model r- or the model fitted before the newest
      // solve has one within tolerance x r. r- alone locates poles far worse than the model, so a genuine lossless
      // resonance could stay unconfirmed for good, with the estimate peaking at its r/64 points, inside the zone where
      // no solve is allowed
      std::vector<cplx> lower_poles = out.fit.lower.size() > 1 ? out.fit.lower.poles() : std::vector<cplx>();
      spurious_comparison = spurious_comparison_poles(out.fit, o.f_max - o.f_min);
      if (previous && previous->size() > 1)
        for (const cplx& q : previous->poles()) lower_poles.push_back(q);
      candidates = grid;
      std::vector<std::pair<double, double>> keep_out;   // not evaluated (confirmed) or not solved (all)
      std::vector<std::pair<double, double>> no_solve;
      for (const cplx& p : out.fit.model.poles()) {
        if (!(p.real() > o.f_min && p.real() < o.f_max)) continue;
        const double r = quarter_step(p.real());
        std::vector<double> near;
        if (std::fabs(p.imag()) >= r) {
          near = {p.real() - std::fabs(p.imag()), p.real(), p.real() + std::fabs(p.imag())};
        } else {
          bool confirmed = false;
          for (const cplx& q : lower_poles) confirmed = confirmed || std::abs(p - q) <= o.tolerance * r;
          no_solve.push_back({p.real() - r, p.real() + r});
          near = {p.real() - r, p.real() + r};
          if (confirmed) {
            keep_out.push_back({p.real() - r, p.real() + r});
          } else {
            for (double c : {1.0 / 64, 1.0 / 16, 0.25}) {
              near.push_back(p.real() - c * r);
              near.push_back(p.real() + c * r);
            }
          }
        }
        for (double x : near)
          if (x > o.f_min && x < o.f_max) candidates.push_back(x);
      }
      std::sort(candidates.begin(), candidates.end());
      auto inside = [](double c, const std::vector<std::pair<double, double>>& zones) {
        for (const auto& [a, b] : zones)
          if (c > a * (1 + 1e-12) && c < b * (1 - 1e-12)) return true;
        return false;
      };
      std::vector<double> evaluated;
      for (double c : candidates)
        if (!inside(c, keep_out)) evaluated.push_back(c);
      candidates = evaluated;
      // a refused frequency keeps later solves at an eighth of the gap between the solves around it: a solve that
      // fails tends to fail near it too, and each attempt costs a full solve
      for (double f : out.failed) {
        double below = o.f_min;
        double above = o.f_max;
        for (const Sample& s : samples) {
          if (s.f < f) below = std::max(below, s.f);
          if (s.f > f) above = std::min(above, s.f);
        }
        const double r = std::max(quarter_step(f), 0.125 * (above - below));
        no_solve.push_back({f - r, f + r});
      }
      solvable.assign(candidates.size(), 1);
      for (size_t i = 0; i < candidates.size(); ++i) {
        bool ok = !inside(candidates[i], no_solve);
        for (double f : out.attempted) ok = ok && std::fabs(candidates[i] - f) >= quarter_step(f);
        solvable[i] = ok ? 1 : 0;
      }
      estimate = estimated_error(out.fit, candidates);
      // The spurious poles' own terms |res_q / (f - q)| in r- (res_q: r-'s residues), in the fit's measure: a
      // candidate is excused (the gate) only where they reach half its estimate, i.e. where they can account for it.
      // A fixed region instead, Re q +- Im q x max(3, 2 x peak / tolerance) with the peak of the total estimate,
      // capped at the band width rather than clipped to it, is not tied to q's own size and covered the whole band
      // in every committed closure of the trials.
      const Eigen::VectorXd terms = spurious_terms(out.fit, spurious_comparison, candidates);
      // and how far the newest solve moved the model anywhere in the band: the fit before it against the fit now
      if (previous && o.compare_previous) {
        for (size_t i = 0; i < candidates.size(); ++i) {
          const double change = relative_error(previous->value(candidates[i]), out.fit.model.value(candidates[i]),
                                               out.fit.floor);
          step.model_change = std::max(step.model_change, change);
          estimate(static_cast<int>(i)) = std::max(estimate(static_cast<int>(i)), change);
        }
      }
      unexcused = estimate;
      for (int i = 0; i < unexcused.size(); ++i)
        if (terms(i) >= 0.5 * estimate(i)) unexcused(i) = 0;
    }
    // the next frequency: the largest estimate among the solvable candidates; without two solves, the middle of the
    // widest gap
    double next = 0;
    double largest = kInf;
    double largest_at = 0;
    double largest_unexcused = kInf;   // the largest estimate without the comparison model's spurious poles
    double next_unexcused = 0;         // where that is largest among the solvable candidates
    if (samples.size() >= 2 && !candidates.empty()) {
      // The stopping maximum leaves out the candidates within a quarter grid step of a solved frequency: the data pin
      // the model there (to the fit tolerance), and no solve may be placed there, so an estimate peaking there could
      // never fall (a sweep can stall on f_min that way). Candidates near a
      // refused frequency stay in.
      auto pinned = [&](double c) {
        for (const Sample& x : samples)
          if (std::fabs(c - x.f) < quarter_step(x.f)) return true;
        return false;
      };
      int at = -1;
      int solve_at = -1;
      for (int i = 0; i < estimate.size(); ++i) {
        const bool nan = std::isnan(estimate(i));
        if (!pinned(candidates[i]) && (at < 0 || estimate(i) > estimate(at) || nan)) at = i;
        if (solvable[i] && (solve_at < 0 || estimate(i) > estimate(solve_at) || nan)) solve_at = i;
      }
      // a near-tie (a flat estimate, as of a one-point fit) goes to the candidate farthest from every attempted
      // frequency, so that confirming solves probe the widest gaps
      if (solve_at >= 0 && std::isfinite(estimate(solve_at))) {
        const double top = estimate(solve_at);
        double widest = -1;
        for (int i = 0; i < estimate.size(); ++i) {
          if (!solvable[i] || !(estimate(i) >= top * (1 - 1e-6))) continue;
          double gap = kInf;
          for (double f : out.attempted) gap = std::min(gap, std::fabs(candidates[i] - f));
          if (gap > widest) {
            widest = gap;
            solve_at = i;
          }
        }
      }
      if (at >= 0) {
        largest = estimate(at);
        largest_at = candidates[at];
      }
      if (solve_at >= 0) next = candidates[solve_at];
      largest_unexcused = 0;
      int unexcused_at = -1;
      double largest_solvable = 0;
      for (int i = 0; i < unexcused.size(); ++i) {
        const double x = std::isnan(unexcused(i)) ? kInf : unexcused(i);
        if (!pinned(candidates[i])) largest_unexcused = std::max(largest_unexcused, x);
        if (solvable[i] && (unexcused_at < 0 || x > largest_solvable)) {
          unexcused_at = i;
          largest_solvable = x;
        }
      }
      // only where the unexcused estimate itself reaches the tolerance: otherwise the redirect would go to the first
      // solvable candidate
      if (unexcused_at >= 0 && largest_solvable >= o.tolerance) next_unexcused = candidates[unexcused_at];
    } else if (!candidates.empty() || samples.empty()) {
      std::vector<double> tried = out.attempted;
      tried.push_back(o.f_min);
      tried.push_back(o.f_max);
      std::sort(tried.begin(), tried.end());
      double gap = -1;
      for (size_t i = 1; i < tried.size(); ++i)
        if (tried[i] - tried[i - 1] > gap) {
          gap = tried[i] - tried[i - 1];
          next = 0.5 * (tried[i] + tried[i - 1]);
        }
    }
    step.estimated_error = largest;
    step.estimated_error_at = largest_at > 0 ? largest_at : next;
    step.next_frequency = next;
    // The fit must reproduce its own solves to 10 x its fit tolerance before the sweep can stop (the greedy aims at
    // the fit tolerance; the FEM sweeps end at or below it). A larger misfit means data inconsistent at that level
    // (derivatives that do not belong to the values, or solves noisier than the fit tolerance), which Froissart
    // doublets at the support points absorb.
    const bool consistent = out.fit.sample_error <= kConsistentFactor * current.fit_tolerance;
    const bool hermite_in_use = std::find(current.hermite.begin(), current.hermite.end(), 1) != current.hermite.end();
    // The decision: 0 continue, 1 converged, 2 out of solves, 3 nowhere left to solve, 4 refit without derivatives,
    // 5 converged on confirmations alone.
    // - Derivative rows are dropped as soon as the newest solves confirm the model while the fit cannot reproduce
    //   its own solves, or as soon as the derivatives of the newest solves disagree with the model whose values they
    //   confirm, not only once the estimate passes: with inconsistent derivatives the doublets they cause keep the
    //   estimate high, and the sweep runs out of solves.
    // - A sweep whose estimate stays above the tolerance only through the terms of spurious poles of the comparison
    //   model (above the real axis by at most kSpuriousReach band widths, not in the model) closes once
    //   kTightConfirmations solves in a row came within tolerance / 100 of their predictions and the newest solve
    //   moved the model by less than the tolerance (such a pole of r- held a lossy sweep of the trials for 49 solves
    //   while every solve agreed to 1e-8). Without that restriction the closure stopped noise-bound sweeps (fit
    //   tolerance at the noise) with errors up to 5e3 x the tolerance where no solve could go. The protection is
    //   mostly the tight confirmations; the gate makes sure the estimate that remains is r-'s artifact.
    const int needed = std::max(o.confirmations, 1);
    const bool confirmed = agreed >= needed;
    const bool only_comparison = largest >= o.tolerance && largest_unexcused < o.tolerance;
    const bool closable = tight >= kTightConfirmations && consistent && step.model_change < o.tolerance;
    // When the gate alone refuses the closure, the next solve goes where the estimate without r-'s spurious terms is
    // largest: the overall maximum sat at such a term, and solves there confirmed the model 50 times in a row while
    // the peak that held the gate was never solved (trial 166)
    if (closable && largest >= o.tolerance && !only_comparison && next_unexcused > 0) next = next_unexcused;
    double stop = 0;
    if (largest < o.tolerance && confirmed && consistent) stop = 1;
    else if (only_comparison && closable) stop = 5;
    else if (hermite_in_use && ((confirmed && !consistent) || derivative_disagreements >= needed)) stop = 4;
    else if (static_cast<int>(out.attempted.size()) >= o.max_solves) stop = 2;
    else if (next <= 0) stop = 3;
    std::vector<double> decision = {stop, next};
    if (o.agree) o.agree(decision);
    log(1, strcat("adaptive sweep: ", step.solves, " solves (", step.samples, " fitted), ", step.support_points,
                  " support points, fit misfit ", step.sample_fit_error, ", estimated error ", largest, " at ",
                  step.estimated_error_at, " Hz (model change ", step.model_change, "), newest solve against its "
                  "prediction ", prediction_error, " (tolerance ", o.tolerance, "); next ", decision[1], " Hz"));
    out.history.push_back(step);
    out.estimated_error = largest;
    out.last_prediction_error = prediction_error;
    if (decision[0] == 4.0) {
      // every derivative row is dropped from here on, and the confirmations start again
      for (char& h : current.hermite) h = 0;
      out.hermite = false;
      if (derivative_disagreements >= needed)
        out.hermite_note = strcat("dropped after ", out.attempted.size(), " solves: the derivatives of the ",
                                  derivative_disagreements, " newest solves missed the model's by more than 100 x ",
                                  "its value error (newest: ", derivative_error, ")");
      else
        out.hermite_note = strcat("dropped after ", out.attempted.size(), " solves: with derivatives the fit missed ",
                                  "its own solves by ", out.fit.sample_error, " (above ", kConsistentFactor,
                                  " x the fit tolerance) while the newest solves confirmed it");
      previous.reset();
      agreed = 0;
      tight = 0;
      derivative_disagreements = 0;
      continue;
    }
    if (decision[0] != 0.0) {
      out.converged = decision[0] == 1.0 || decision[0] == 5.0;
      if (decision[0] == 1.0) out.stop_reason = "converged";
      else if (decision[0] == 5.0)
        out.stop_reason = strcat("converged: ", kTightConfirmations, " solves in a row within tolerance / 100 of ",
                                 "their predictions; the estimate stayed above the tolerance only through the terms ",
                                 "of spurious poles of the comparison model");
      else if (decision[0] == 2.0) out.stop_reason = "max_solves";
      else out.stop_reason = "no admissible candidate frequency";
      break;
    }
    next = decision[1];
    Eigen::VectorXcd predicted;
    Eigen::VectorXcd predicted_derivative;
    double nearest = kInf;   // the distance to the nearest sample before this one (the derivative rows' length)
    if (!samples.empty()) {
      predicted = out.fit.model.value(next);
      predicted_derivative = out.fit.model.derivative(next);
      for (const Sample& x : samples) nearest = std::min(nearest, std::fabs(x.f - next));
    }
    const std::optional<Sample> s = attempt(next);
    solved_since_fit = true;
    predicted_at = next;
    prediction_error = s && predicted.size() == s->value.size() ? relative_error(predicted, s->value, out.fit.floor)
                                                                 : kInf;
    agreed = prediction_error < o.tolerance ? agreed + 1 : 0;
    tight = prediction_error <= 0.01 * o.tolerance ? tight + 1 : 0;
    // the derivative's prediction error, in the derivative rows' measure: |F' - r'| over the length within which F
    // changes by itself (at most the distance to the nearest sample), relative to max(|F|, floor)
    derivative_error = 0;
    if (s && s->derivative.size() == s->value.size() && predicted_derivative.size() == s->value.size()) {
      for (int k = 0; k < s->value.size(); ++k) {
        if (!current.hermite[k] || !std::isfinite(out.fit.floor(k))) continue;
        const cplx G = s->derivative(k);
        if (!std::isfinite(G.real()) || !std::isfinite(G.imag())) continue;
        const double magnitude = std::max(std::abs(s->value(k)), out.fit.floor(k));
        const double local = std::abs(G) > 0 ? magnitude / std::abs(G) : kInf;
        const double length = std::min(nearest, local);
        derivative_error = std::max(derivative_error, std::abs(G - predicted_derivative(k)) * length / magnitude);
      }
    }
    // a confirmed value whose derivative misses by far more than the value does
    const bool disagrees = prediction_error < o.tolerance &&
                           derivative_error > 100.0 * std::max(prediction_error, current.fit_tolerance);
    derivative_disagreements = disagrees ? derivative_disagreements + 1 : 0;
  }
  out.samples = samples;
  if (previous) out.previous = *previous;
  out.fit_consistent = out.fit.sample_error <= kConsistentFactor * current.fit_tolerance;
  out.options = current;
  out.sample_error_before_removal = out.fit.sample_error;
  if (out.converged) {
    out.invisible_poles_removed =
        remove_invisible_poles(out.fit, samples, current, o, previous && o.compare_previous ? &*previous : nullptr);
    if (out.invisible_poles_removed > 0)
      log(1, strcat("adaptive sweep: ", out.invisible_poles_removed, " invisible pole(s) on the real axis removed ",
                    "after convergence; the model now misses its solves by ", out.fit.sample_error, " (",
                    out.sample_error_before_removal, " before)"));
  }
  return out;
}

}  // namespace axf::rational
