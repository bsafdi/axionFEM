// The rational fit and the adaptive sweep of the 3D full-wave engine (rational_fit.cpp,
// docs/physics/adaptive_sweep.md) on synthetic resonant responses with known poles:
//   1. exact rational data (two lossy in-band poles, out-of-band background): the fit's values and poles;
//   2. Hermite data: the same accuracy from fewer samples than values alone;
//   3. Froissart doublets: an overfitted noisy smooth response has no spurious spike in the band after cleanup;
//   4. the adaptive sweep, lossy: converges, the newest solve agrees with its prediction, the true error is below
//      the tolerance, the poles and Q are found;
//   5. lossless poles on the real axis, with a refused band of frequencies around an initial point (solve returns
//      nothing there): the refused frequencies are recorded and skipped, no solve lands on a pole, and the sweep
//      converges;
//   6. the decision hook sees every decision (the MPI caller makes them rank 0's);
//   7-14: weak genuine poles survive the cleanup; unit invariance; no usable solve;
//      the max_solves budget; inconsistent derivatives are dropped; a changed decision is obeyed; the model's
//      derivative next to a support point; a constant response converges;
//   15-16: lossless poles of a fit with every solve a support point are not
//      non_passive and get no Q, and a Q of 1e9 is resolved; the same on the four-port FEM sweep's solves under noise
//      (cpp/tests/data/adaptive_four_ports_samples.txt); the "no admissible candidate" stop;
//   17: a model whose pole is the outlier among its leave-one-out refits
//      (cpp/tests/data/adaptive_highq_trial11_samples.txt); its Q_min is a 3 sigma bound;
//   18-20: the lossy cavity's FEM solves under phase noise: resolved Q within 3 sigma
//      (cpp/tests/data/adaptive_cavity_lossy_samples.txt); the closure's gate helpers, and the noise-bound rule;
//      trial 88's near-real doublet is removed after convergence (cpp/tests/data/adaptive_trial88_samples.txt);
//   21: one bad solve next to a Q 1e9 pole
//      (cpp/tests/data/adaptive_highq1e9_samples.txt); 22: a weak genuine mode next to a strong one is
//      not removed.
#include "axionfem/rational_fit.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using axf::rational::cplx;
namespace R = axf::rational;

namespace {

// Gaussian noise that is the same on every platform. std::normal_distribution is implementation-defined (libstdc++ and
// libc++ draw different numbers from the same engine), so the noisy-data checks would see different data on different
// standard libraries. This is the algorithm of libstdc++ (the Marsaglia polar method on 53-bit uniforms made of two
// 32-bit engine outputs), so on GCC builds the draws are the ones std::normal_distribution gives.
class Normal {
 public:
  Normal(double mean, double stddev) : mean_(mean), stddev_(stddev) {}
  double operator()(std::mt19937& g) {
    double v;
    if (saved_available_) {
      saved_available_ = false;
      v = saved_;
    } else {
      double x, y, r2;
      do {
        x = 2.0 * uniform(g) - 1.0;
        y = 2.0 * uniform(g) - 1.0;
        r2 = x * x + y * y;
      } while (r2 > 1.0 || r2 == 0.0);
      const double mult = std::sqrt(-2 * std::log(r2) / r2);
      saved_ = x * mult;
      saved_available_ = true;
      v = y * mult;
    }
    return v * stddev_ + mean_;
  }

 private:
  static double uniform(std::mt19937& g) {   // in [0, 1)
    double sum = 0.0;
    sum += static_cast<double>(g());
    sum += static_cast<double>(g()) * 4294967296.0;
    const double u = sum / 18446744073709551616.0;
    return u >= 1.0 ? std::nextafter(1.0, 0.0) : u;
  }
  double mean_, stddev_;
  double saved_ = 0.0;
  bool saved_available_ = false;
};

struct Response {
  std::vector<cplx> poles;   // in f (e^{-i omega t}: Im p <= 0)
  std::vector<double> a;     // Z strengths
  std::vector<double> b;     // Psi strengths
  double L0 = 0.3;

  // outputs: Z (with dZ/df) and Psi (no derivative), as a single-port sweep row
  R::Sample at(double f) const {
    cplx Z = cplx(0, -L0) * f;
    cplx dZ = cplx(0, -L0);
    cplx Psi = 1.0;
    for (size_t n = 0; n < poles.size(); ++n) {
      const cplx p2 = poles[n] * poles[n];
      Z += cplx(0, a[n]) * f / (p2 - f * f);
      dZ += cplx(0, a[n]) * (p2 + f * f) / ((p2 - f * f) * (p2 - f * f));
      Psi += b[n] * p2 / (p2 - f * f);
    }
    R::Sample s;
    s.f = f;
    s.value = Eigen::VectorXcd(2);
    s.value << Z, Psi;
    s.derivative = Eigen::VectorXcd(2);
    s.derivative << dZ, cplx(std::nan(""), 0.0);
    return s;
  }
};

R::FitOptions options(double fit_tolerance) {
  R::FitOptions o;
  o.group = {0, 1};
  o.hermite = {1, 0};
  o.fit_tolerance = fit_tolerance;
  return o;
}

// the largest pointwise error of the model on a fine grid of [lo, hi], away from real poles
double true_error(const R::Model& model, const Response& r, double lo, double hi, const Eigen::VectorXd& floor) {
  double worst = 0;
  for (int i = 0; i <= 4000; ++i) {
    const double f = lo + (hi - lo) * i / 4000.0;
    bool near_pole = false;
    for (const cplx& p : r.poles) near_pole = near_pole || (p.imag() == 0 && std::fabs(f - p.real()) < 1e-6);
    if (near_pole) continue;
    worst = std::max(worst, R::relative_error(model.value(f), r.at(f).value, floor));
  }
  return worst;
}

// the model's pole nearest to p
cplx nearest_pole(const R::Model& model, cplx p) {
  cplx best(1e300, 0);
  for (const cplx& q : model.poles())
    if (std::abs(q - p) < std::abs(best - p)) best = q;
  return best;
}

double quality(cplx p) { return p.real() / (2 * std::fabs(p.imag())); }

// a sweep's solves as benchmarks/adaptive_sweep.py samples writes them: "f nports, Z row-major and Psi as re im
// pairs, then dZ/df row-major as re im pairs" per line; lines starting with '#' are skipped
std::vector<R::Sample> read_samples(const std::string& path, int& nports) {
  std::ifstream in(path);
  std::vector<R::Sample> samples;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    R::Sample s;
    is >> s.f >> nports;
    const int nz = nports * nports;
    s.value.resize(nz + nports);
    s.derivative = Eigen::VectorXcd::Constant(nz + nports, cplx(std::nan(""), 0.0));
    for (int k = 0; k < nz + nports; ++k) {
      double re = 0;
      double im = 0;
      is >> re >> im;
      s.value(k) = cplx(re, im);
    }
    for (int k = 0; k < nz; ++k) {
      double re = 0;
      double im = 0;
      is >> re >> im;
      s.derivative(k) = cplx(re, im);
    }
    samples.push_back(s);
  }
  return samples;
}

// the engine's fit options for an n-port sweep
R::FitOptions engine_options(int nports, double fit_tolerance) {
  const int nz = nports * nports;
  R::FitOptions o;
  for (int k = 0; k < nz + nports; ++k) {
    o.group.push_back(k < nz ? 0 : 1);
    o.hermite.push_back(k < nz ? 1 : 0);
    if (k < nz) o.floor_pair.push_back({(k / nports) * (nports + 1), (k % nports) * (nports + 1)});
    else o.floor_pair.push_back({-1, -1});
  }
  o.fit_tolerance = fit_tolerance;
  return o;
}

int failures = 0;

void check(bool ok, const char* what, double value) {
  std::printf("%-58s %.3e  %s\n", what, value, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

}  // namespace

int main() {
  Response lossy;
  lossy.poles = {cplx(0.55, -0.55 / 400), cplx(1.2301234, -1.2301234 / 2000), cplx(1.5703217, -1.5703217 / 60),
                 cplx(2.6, -2.6 / 100), cplx(3.3, -3.3 / 100)};
  lossy.a = {0.4, 0.2, 0.05, 0.6, 1.0};
  lossy.b = {0.3, 0.1, -0.2, 0.4, 0.2};

  // 1. exact data at 16 linearly spaced samples of [1, 2]
  {
    std::vector<R::Sample> s;
    for (int i = 0; i < 16; ++i) s.push_back(lossy.at(1.0 + i / 15.0));
    const R::Fit fit = R::fit(s, options(1e-12));
    const double err = true_error(fit.model, lossy, 1.0, 2.0, fit.floor);
    check(err < 1e-8, "1. exact data: model error on a fine grid", err);
    double dp = 0;
    double dq = 0;
    for (int n : {1, 2}) {
      const cplx q = nearest_pole(fit.model, lossy.poles[n]);
      dp = std::max(dp, std::abs(q - lossy.poles[n]) / std::abs(lossy.poles[n]));
      dq = std::max(dq, std::fabs(quality(q) / quality(lossy.poles[n]) - 1));
    }
    check(dp < 1e-9, "1. exact data: in-band poles (relative)", dp);
    check(dq < 1e-6, "1. exact data: in-band Q (relative)", dq);
    // the same data in Hz of a 100 MHz band (the poles' pencil is scaled to the support; unscaled it was singular)
    const double hz = 1e8;
    std::vector<R::Sample> sh = s;
    for (R::Sample& x : sh) {
      x.f *= hz;
      x.derivative /= hz;
    }
    const R::Fit fh = R::fit(sh, options(1e-12));
    double dh = 0;
    for (int n : {1, 2})
      dh = std::max(dh, std::abs(nearest_pole(fh.model, lossy.poles[n] * hz) - lossy.poles[n] * hz) /
                            std::abs(lossy.poles[n] * hz));
    check(dh < 1e-9, "1. exact data in Hz (x 1e8): in-band poles (relative)", dh);
  }

  // 2. Hermite data: 7 samples with derivatives against 7 without (Z's derivative dropped)
  {
    std::vector<R::Sample> with;
    std::vector<R::Sample> without;
    for (int i = 0; i < 7; ++i) {
      with.push_back(lossy.at(1.0 + i / 6.0));
      without.push_back(with.back());
      without.back().derivative(0) = cplx(std::nan(""), 0.0);
    }
    const R::Fit fh = R::fit(with, options(1e-12));
    const R::Fit fv = R::fit(without, options(1e-12));
    const double eh = true_error(fh.model, lossy, 1.0, 2.0, fh.floor);
    const double evalue = true_error(fv.model, lossy, 1.0, 2.0, fv.floor);
    std::printf("2. 7 samples: error with Hermite data %.3e, values only %.3e\n", eh, evalue);
    check(eh < 1e-2 && eh < 1e-2 * evalue, "2. Hermite data: error (and 100x below values only)", eh);
  }

  // 3. Froissart doublets: a smooth response with 1e-9 noise, overfitted (fit tolerance 0, every sample a support
  // point); after cleanup the model has no spike between the samples
  {
    Response smooth;
    smooth.poles = {cplx(2.6, -0.026), cplx(0.4, -0.004)};
    smooth.a = {0.5, 0.2};
    smooth.b = {0.3, 0.1};
    std::mt19937 rng(3);
    Normal noise(0.0, 1e-9);
    std::vector<R::Sample> s;
    for (int i = 0; i < 40; ++i) {
      R::Sample x = smooth.at(1.0 + i / 39.0);
      for (int k = 0; k < 2; ++k) x.value(k) *= 1.0 + noise(rng);
      x.derivative(0) *= 1.0 + noise(rng);
      s.push_back(x);
    }
    const R::Fit fit = R::fit(s, options(0.0));
    const double err = true_error(fit.model, smooth, 1.0, 2.0, fit.floor);
    std::printf("3. overfitted noisy data: %d support points, %d doublets removed\n", fit.model.size(),
                fit.doublets_removed);
    check(err < 1e-7 && fit.doublets_removed > 0, "3. overfitted noisy data: error between samples", err);
  }

  // 4. adaptive sweep, lossy poles in [1, 2]
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-4;
    o.max_solves = 40;
    o.initial = {1.0, 1.25, 1.5, 1.75, 2.0};
    o.log_spacing = false;
    int decisions = 0;
    o.agree = [&](std::vector<double>& d) {
      ++decisions;
      (void)d;
    };
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-7), [&](double f) {
      return std::optional<R::Sample>(lossy.at(f));
    });
    const double err = true_error(res.fit.model, lossy, 1.0, 2.0, res.fit.floor);
    std::printf("4. lossy sweep: %zu solves, %s, estimate %.2e, last prediction %.2e\n", res.attempted.size(),
                res.stop_reason.c_str(), res.estimated_error, res.last_prediction_error);
    check(res.converged, "4. lossy sweep converged (1 = yes)", res.converged ? 1.0 : 0.0);
    check(res.last_prediction_error < 1e-4, "4. lossy sweep: newest solve against its prediction",
          res.last_prediction_error);
    check(err < 1e-4, "4. lossy sweep: true error over the band", err);
    double dq = 0;
    for (int n : {1, 2}) dq = std::max(dq, std::fabs(quality(nearest_pole(res.fit.model, lossy.poles[n])) /
                                                      quality(lossy.poles[n]) - 1));
    check(dq < 1e-6, "4. lossy sweep: Q of the in-band poles (relative)", dq);
    check(decisions == static_cast<int>(res.history.size()), "6. one agreed decision per fit", decisions);
    check(res.attempted.size() <= 20, "4. lossy sweep: solves", static_cast<double>(res.attempted.size()));
  }

  // 5. lossless poles on the real axis; frequencies in (1.24, 1.30) are refused (the initial point 1.25 among them)
  {
    Response lossless = lossy;
    for (cplx& p : lossless.poles) p = cplx(p.real(), 0.0);
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-3;
    o.max_solves = 40;
    o.initial = {1.0, 1.25, 1.5, 1.75, 2.0};
    o.log_spacing = true;
    double closest = 1e300;
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double f) {
      for (const cplx& p : lossless.poles) closest = std::min(closest, std::fabs(f - p.real()) / p.real());
      if (f > 1.24 && f < 1.30) return std::optional<R::Sample>();
      return std::optional<R::Sample>(lossless.at(f));
    });
    const double err = true_error(res.fit.model, lossless, 1.0, 2.0, res.fit.floor);
    std::printf("5. lossless sweep: %zu solves, %zu refused, %s, closest solve to a pole %.2e (relative)\n",
                res.attempted.size(), res.failed.size(), res.stop_reason.c_str(), closest);
    check(res.converged, "5. lossless sweep with refusals converged (1 = yes)", res.converged ? 1.0 : 0.0);
    bool recorded = true;
    for (double f : res.failed) recorded = recorded && f > 1.24 && f < 1.30;
    for (const R::Sample& s : res.samples) recorded = recorded && !(s.f > 1.24 && s.f < 1.30);
    recorded = recorded && !res.failed.empty();
    check(recorded, "5. refused frequencies recorded, not fitted (1 = yes)", recorded ? 1.0 : 0.0);
    check(closest > 1e-6, "5. no solve on a lossless pole (closest, relative)", closest);
    double dp = 0;
    for (int n : {1, 2}) dp = std::max(dp, std::abs(nearest_pole(res.fit.model, lossless.poles[n]) -
                                                    lossless.poles[n]) / lossless.poles[n].real());
    check(dp < 1e-8, "5. lossless sweep: pole positions (relative)", dp);
    check(err < 1e-3, "5. lossless sweep: true error over the band", err);
  }

  // 7. weak genuine poles survive the cleanup: three in-band poles of strengths 1e-5 to 1e-4 of the strong one (the
  // weakest of one test cavity were 4e-6 of scale x band), 1e-10 noise, sampled as a sweep would (on a grid and next to
  // the poles); the fit at tolerance 1e-6 must keep all of them
  {
    Response weak;
    weak.poles = {cplx(1.21, -1.21 / 3e4), cplx(1.43, -1.43 / 1e4), cplx(1.61, -1.61 / 5e3), cplx(1.8, -0.018),
                  cplx(2.7, -0.03), cplx(0.5, -0.01)};
    weak.a = {2e-6, 1e-5, 3e-6, 0.3, 1.0, 0.4};
    weak.b = {1e-5, -2e-5, 1e-5, 0.2, 0.3, 0.1};
    std::mt19937 rng(11);
    Normal noise(0.0, 1e-10);
    std::vector<double> f;
    for (int i = 0; i <= 12; ++i) f.push_back(1.0 + i / 12.0);
    for (int n = 0; n < 3; ++n)
      for (double c : {-3.0, -1.0, 1.0, 3.0}) f.push_back(weak.poles[n].real() + c * std::fabs(weak.poles[n].imag()));
    std::sort(f.begin(), f.end());
    std::vector<R::Sample> s;
    for (double x : f) {
      R::Sample y = weak.at(x);
      for (int k = 0; k < 2; ++k) y.value(k) *= 1.0 + noise(rng);
      y.derivative(0) *= 1.0 + noise(rng);
      s.push_back(y);
    }
    const R::Fit fit = R::fit(s, options(1e-6));
    double dp = 0;
    for (int n = 0; n < 3; ++n)
      dp = std::max(dp, std::abs(nearest_pole(fit.model, weak.poles[n]) - weak.poles[n]) / std::fabs(weak.poles[n].imag()));
    std::printf("7. weak poles: %d support points, %d doublets removed\n", fit.model.size(), fit.doublets_removed);
    check(dp < 1e-2, "7. weak genuine poles kept (pole error / half-width)", dp);
  }

  // 8. unit invariance: the same sweep with the band [1, 2] and in Hz (x 7e7) takes the same decisions
  {
    auto run = [&](double unit) {
      R::AdaptiveOptions o;
      o.f_min = unit;
      o.f_max = 2 * unit;
      o.tolerance = 1e-4;
      o.max_solves = 40;
      o.initial = R::band_grid(unit, 2 * unit, 5, true);
      o.log_spacing = true;
      return R::adaptive_sweep(o, options(1e-7), [&](double f) {
        R::Sample s = lossy.at(f / unit);
        s.derivative /= unit;
        return std::optional<R::Sample>(s);
      });
    };
    const R::AdaptiveResult a = run(1.0);
    const R::AdaptiveResult b = run(7e7);
    double worst = a.attempted.size() == b.attempted.size() ? 0.0 : 1.0;
    for (size_t i = 0; i < std::min(a.attempted.size(), b.attempted.size()); ++i)
      worst = std::max(worst, std::fabs(b.attempted[i] / 7e7 - a.attempted[i]) / a.attempted[i]);
    std::printf("8. unit invariance: %zu and %zu solves\n", a.attempted.size(), b.attempted.size());
    check(worst < 1e-9, "8. same frequencies in [1, 2] and in Hz (relative)", worst);
  }

  // 9. no usable solve: the sweep stops after the initial points, unconverged, with nothing fitted
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.max_solves = 20;
    o.initial = {1.0, 1.5, 2.0};
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double) { return std::optional<R::Sample>(); });
    const bool ok = !res.converged && res.attempted.size() == 3 && res.samples.empty() && res.fit.model.size() == 0 &&
                    res.stop_reason == "no usable solve among the initial points";
    check(ok, "9. no usable solve: unconverged, 3 attempts (1 = yes)", ok ? 1.0 : 0.0);
  }

  // 10. the budget: max_solves reached unconverged
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-8;
    o.max_solves = 7;
    o.initial = {1.0, 1.25, 1.5, 1.75, 2.0};
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-10), [&](double f) {
      return std::optional<R::Sample>(lossy.at(f));
    });
    const bool ok = !res.converged && res.attempted.size() == 7 && res.stop_reason == "max_solves";
    check(ok, "10. budget: unconverged after exactly 7 solves (1 = yes)", ok ? 1.0 : 0.0);
  }

  // 11. derivatives that do not belong to the values (a bias of 1e-2 of |Z| per band width, as a missing term of
  // A' gave): the fit misses its own solves, the derivative rows are dropped, and the sweep converges on values only
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-3;
    o.max_solves = 40;
    o.initial = R::band_grid(1.0, 2.0, 5, false);
    o.log_spacing = false;
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double f) {
      R::Sample s = lossy.at(f);
      s.derivative(0) += 1e-2 * std::abs(s.value(0));
      return std::optional<R::Sample>(s);
    });
    const double err = true_error(res.fit.model, lossy, 1.0, 2.0, res.fit.floor);
    std::printf("11. biased derivatives: %zu solves, %s, Hermite %s (%s)\n", res.attempted.size(),
                res.stop_reason.c_str(), res.hermite ? "kept" : "dropped", res.hermite_note.c_str());
    check(res.converged && !res.hermite && res.fit_consistent, "11. biased derivatives dropped, converged (1 = yes)",
          res.converged && !res.hermite ? 1.0 : 0.0);
    check(err < 1e-3, "11. biased derivatives: true error over the band", err);
  }

  // 12. a changed decision is obeyed: the hook moves every next frequency by +1e-5 (as rank 0's broadcast would)
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-3;
    o.max_solves = 12;
    o.initial = {1.0, 1.25, 1.5, 1.75, 2.0};
    o.log_spacing = false;
    std::vector<double> moved;
    o.agree = [&](std::vector<double>& d) {
      if (d[0] == 0.0) {
        d[1] += 1e-5;
        moved.push_back(d[1]);
      }
    };
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double f) {
      return std::optional<R::Sample>(lossy.at(f));
    });
    bool obeyed = moved.size() + 5 == res.attempted.size();
    for (size_t i = 0; i < moved.size() && obeyed; ++i) obeyed = res.attempted[5 + i] == moved[i];
    check(obeyed, "12. the agreed next frequency is the one solved (1 = yes)", obeyed ? 1.0 : 0.0);
  }

  // 13. the model's derivative next to a support point: no cancellation
  {
    std::vector<R::Sample> s;
    for (int i = 0; i < 9; ++i) {
      R::Sample x = lossy.at(1.0 + i / 8.0);
      x.f *= 1e8;
      x.derivative /= 1e8;
      s.push_back(x);
    }
    const R::Fit fit = R::fit(s, options(1e-12));
    const double fj = fit.model.support[2];
    const Eigen::VectorXcd exact = fit.model.derivative(fj);
    double worst = 0;
    for (double rho : {1e-15, 1e-14, 1e-13, 1e-12}) {
      const Eigen::VectorXcd near = fit.model.derivative(fj * (1 + rho));
      worst = std::max(worst, (near - exact).cwiseAbs().maxCoeff() / exact.cwiseAbs().maxCoeff());
    }
    check(worst < 1e-9, "13. derivative within 1e-12 of a support point (relative)", worst);
  }

  // 14. a constant response (a matched line's Z, with 1e-9 noise): one support point fits it, and the sweep converges
  // (the comparison model is the samples' mean, not undefined)
  {
    R::AdaptiveOptions o;
    o.f_min = 1e8;
    o.f_max = 3e8;
    o.tolerance = 1e-3;
    o.max_solves = 30;
    o.initial = R::band_grid(1e8, 3e8, 5, false);
    o.log_spacing = false;
    std::mt19937 rng(3);
    Normal noise(0, 1);
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double f) {
      R::Sample s;
      s.f = f;
      s.value = Eigen::VectorXcd(2);
      // two draws, taken in a fixed order (the order of evaluation of function arguments is unspecified)
      const double n_im = noise(rng);
      const double n_re = noise(rng);
      s.value << cplx(50.0 * (1 + 1e-9 * n_re), 1e-9 * n_im), cplx(0.0, 0.0);
      s.derivative = Eigen::VectorXcd(2);
      s.derivative << cplx(1e-9 * noise(rng) * 50.0 / 1e8, 0.0), cplx(std::nan(""), 0.0);
      return std::optional<R::Sample>(s);
    });
    std::printf("14. constant response: %zu solves, %s, %d support points\n", res.attempted.size(),
                res.stop_reason.c_str(), res.fit.model.size());
    check(res.converged && res.attempted.size() <= 8, "14. constant response converged (solves)",
          static_cast<double>(res.attempted.size()));
  }

  // 15. pole classes of lossless and high-Q poles when every solve is a support point:
  // 7 solves of a lossless response with 1e-10 noise, 100 realizations. The fitted Im p of the lossless poles
  // scatters above and below the 1e-12 |p| line; judged against its leave-one-out uncertainty, no in-band lossless
  // pole may be non_passive or get a Q. A pole of Q 1e9, sampled at its peak's neighbourhood, keeps its Q.
  {
    Response lossless;
    lossless.poles = {cplx(0.55, 0.0), cplx(1.2301234, 0.0), cplx(1.5703217, 0.0), cplx(2.6, 0.0), cplx(3.3, 0.0)};
    lossless.a = {0.4, 0.2, 0.05, 0.6, 1.0};
    lossless.b = {0.3, 0.1, -0.2, 0.4, 0.2};
    R::PoleOptions po;
    po.f_min = 1.0;
    po.f_max = 2.0;
    po.tolerance = 1e-3;
    po.log_spacing = false;
    std::mt19937 rng(11);
    Normal noise(0.0, 1e-10);
    int old_rule = 0;     // Im p above 1e-12 |p| or below -1e-12 |p| (the plain-threshold classes, for comparison)
    int flipped = 0;      // non_passive or given a Q now
    int resonances = 0;   // in-band lossless poles classed resonance
    for (int r = 0; r < 100; ++r) {
      std::vector<R::Sample> s;
      for (double x : {1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0}) {
        R::Sample y = lossless.at(x);
        for (int k = 0; k < 2; ++k) y.value(k) *= 1.0 + noise(rng);
        y.derivative(0) *= 1.0 + noise(rng);
        s.push_back(y);
      }
      const R::FitOptions o = options(1e-6);
      const R::Fit fit = R::fit(s, o);
      for (const R::PoleInfo& info : R::classify_poles(fit, s, o, po)) {
        if (std::fabs(info.p.imag()) > 1e-12 * std::abs(info.p)) ++old_rule;
        if (info.non_passive || info.lossy) ++flipped;
        if (info.cls == "resonance") ++resonances;
      }
    }
    std::printf("15. lossless, 7 solves, 100 realizations: %d poles off the 1e-12 line (old rule), %d flipped now, "
                "%d resonances (of 200)\n", old_rule, flipped, resonances);
    check(flipped == 0, "15. lossless poles: non_passive or given a Q", flipped);
    check(resonances == 200, "15. lossless poles classed resonance (of 200)", resonances);
    Response highq;
    highq.poles = {cplx(1.4, -1.4 / 2e9), cplx(2.6, -0.026), cplx(0.5, -0.005)};
    highq.a = {0.1, 0.6, 0.4};
    highq.b = {0.2, 0.4, 0.3};
    R::AdaptiveOptions ao;
    ao.f_min = 1.0;
    ao.f_max = 2.0;
    ao.tolerance = 1e-5;
    ao.max_solves = 40;
    ao.initial = R::band_grid(1.0, 2.0, 5, false);
    ao.log_spacing = false;
    const R::AdaptiveResult res = R::adaptive_sweep(ao, options(1e-8), [&](double f) {
      return std::optional<R::Sample>(highq.at(f));
    });
    po.tolerance = 1e-5;
    double q_error = 1e300;
    for (const R::PoleInfo& info : R::classify_poles(res.fit, res.samples, res.options, po))
      if (std::fabs(info.p.real() - 1.4) < 1e-6 && info.lossy) q_error = std::fabs(info.Q / 1e9 - 1);
    check(q_error < 1e-2, "15. Q 1e9 resolved (relative error)", q_error);
  }

#ifdef AXF_SOURCE_DIR
  // 15b. an FEM case: the four-port gap sweep's 8 solves (lossless, every solve a support point;
  // benchmarks/adaptive_sweep.py gap four_ports), refitted with the engine's options under 1e-10 noise. Its
  // strongest resonance (140.37 MHz) left the 1e-12 |p| line in about 60 % of realizations; it must stay a lossless
  // resonance.
  {
    std::ifstream in(std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_four_ports_samples.txt");
    std::vector<R::Sample> samples;
    int nports = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream is(line);
      R::Sample s;
      is >> s.f >> nports;
      const int nz = nports * nports;
      s.value.resize(nz + nports);
      s.derivative = Eigen::VectorXcd::Constant(nz + nports, cplx(std::nan(""), 0.0));
      for (int k = 0; k < nz + nports; ++k) {
        double re = 0;
        double im = 0;
        is >> re >> im;
        s.value(k) = cplx(re, im);
      }
      for (int k = 0; k < nz; ++k) {
        double re = 0;
        double im = 0;
        is >> re >> im;
        s.derivative(k) = cplx(re, im);
      }
      samples.push_back(s);
    }
    const int nz = nports * nports;
    R::FitOptions o;   // the engine's
    for (int k = 0; k < nz + nports; ++k) {
      o.group.push_back(k < nz ? 0 : 1);
      o.hermite.push_back(k < nz ? 1 : 0);
      if (k < nz) o.floor_pair.push_back({(k / nports) * (nports + 1), (k % nports) * (nports + 1)});
      else o.floor_pair.push_back({-1, -1});
    }
    o.fit_tolerance = 1e-6;
    R::PoleOptions po;
    po.f_min = 1.15e8;
    po.f_max = 1.5e8;
    po.tolerance = 1e-3;
    po.log_spacing = false;
    std::mt19937 rng(17);
    Normal noise(0.0, 1e-10);
    int old_rule = 0;
    int flipped = 0;
    int resonance = 0;
    int found = 0;
    for (int r = 0; r < 40; ++r) {
      std::vector<R::Sample> s = samples;
      for (auto& x : s) {
        for (int k = 0; k < x.value.size(); ++k) x.value(k) *= 1.0 + noise(rng);
        for (int k = 0; k < nz; ++k) x.derivative(k) *= 1.0 + noise(rng);
      }
      const R::Fit fit = R::fit(s, o);
      for (const R::PoleInfo& info : R::classify_poles(fit, s, o, po)) {
        if (std::fabs(info.p.real() - 1.4037e8) > 1e5) continue;
        ++found;
        if (std::fabs(info.p.imag()) > 1e-12 * std::abs(info.p)) ++old_rule;
        if (info.non_passive || info.lossy) ++flipped;
        if (info.cls == "resonance") ++resonance;
      }
    }
    std::printf("15b. four-port FEM samples, 40 realizations of 1e-10 noise: 140.37 MHz found %d, off the 1e-12 line "
                "%d (old rule), flipped now %d, resonance %d\n", found, old_rule, flipped, resonance);
    check(samples.size() == 8 && found == 40 && flipped == 0 && resonance == 40,
          "15b. four-port resonance stays lossless (flipped)", flipped);
  }

  // 17. the model as the outlier: the 8 solves of the trials'
  // highq trial 11 (one pole of Q 1e11 at 181.614428 MHz, Im p = -9.08e-4 Hz; 1e-10 noise; benchmarks/
  // adaptive_sweep_trials.cpp highq 1e11 1e-10, AXF_TRIAL_DUMP=11; cpp/tests/data/adaptive_highq_trial11_samples.txt).
  // The 6-point fit puts the pole at Im p = +1.85e-3 Hz while every leave-one-out refit has it near -9.1e-4 Hz; the
  // refits' spread about their own mean was 1.05e-5 Hz, and the pole was non_passive at 176 sigma. With the model's
  // offset from the refits in sigma it is not non_passive, and any bound Q_min is below the true Q.
  {
    std::ifstream in(std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_highq_trial11_samples.txt");
    std::vector<R::Sample> samples;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream is(line);
      R::Sample s;
      int nports = 0;
      double zr = 0;
      double zi = 0;
      double pr = 0;
      double pi = 0;
      double dr = 0;
      double di = 0;
      is >> s.f >> nports >> zr >> zi >> pr >> pi >> dr >> di;
      s.value = Eigen::VectorXcd(2);
      s.value << cplx(zr, zi), cplx(pr, pi);
      s.derivative = Eigen::VectorXcd(2);
      s.derivative << cplx(dr, di), cplx(std::nan(""), 0.0);
      samples.push_back(s);
    }
    const R::FitOptions o = options(1e-6);
    R::PoleOptions po;
    po.f_min = 1e8;
    po.f_max = 2e8;
    po.tolerance = 1e-3;
    po.log_spacing = true;
    po.noise = 1e-10;
    const R::Fit fit = R::fit(samples, o);
    const R::PoleInfo* target = nullptr;
    const std::vector<R::PoleInfo> classed = R::classify_poles(fit, samples, o, po);
    for (const R::PoleInfo& info : classed)
      if (std::fabs(info.p.real() - 181614428.0) < 1e3) target = &info;
    const bool found = target != nullptr;
    std::printf("17. highq trial 11: %zu solves, %d support points; pole Im p %+.3e Hz, sigma_Im %.3e Hz, class %s, "
                "Q_min %.3g\n", samples.size(), fit.model.size(), found ? target->p.imag() : 0.0,
                found ? target->sigma_im : 0.0, found ? target->cls.c_str() : "missing", found ? target->Q_min : 0.0);
    const bool ok = found && samples.size() == 8 && !target->non_passive && target->cls == "resonance" &&
                    (target->lossy || target->Q_min <= 1e11);
    check(ok, "17. the outlying model's pole is not non_passive (1 = yes)", ok ? 1.0 : 0.0);
    // Q_min is the 3 sigma bound Re p / (2 (|Im p| + 3 sigma_Im)), not Re p / (2 line)
    const double bound = found ? target->p.real() / (2 * (std::fabs(target->p.imag()) + 3 * target->sigma_im)) : 0;
    check(found && std::fabs(target->Q_min / bound - 1) < 1e-12, "17. Q_min = Re p / (2 (|Im p| + 3 sigma)) (ratio)",
          found ? target->Q_min / bound : 0.0);
  }

  // 18. phase noise: the lossy cavity's 23 FEM
  // solves (benchmarks/adaptive_sweep.py cavity_lossy, 8 lossy poles of Q 6.7e4 to 6.9e6) under relative noise 1e-8
  // with a random phase, 5 realizations, classed with that noise. The plain jackknife is conservative here (it withholds
  // some Q: sigma_Im up to about 700 Hz at 178.09 MHz against an actual scatter of 0.1 Hz); what must hold is coverage:
  // no pole non_passive, and every resolved Q within 3 standard errors of the noise-free fit's.
  {
    int nports = 0;
    const std::vector<R::Sample> samples =
        read_samples(std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_cavity_lossy_samples.txt", nports);
    const R::FitOptions o = engine_options(nports, 1e-6);
    R::PoleOptions po;
    po.f_min = 1e8;
    po.f_max = 2.7e8;
    po.tolerance = 1e-3;
    po.log_spacing = false;
    po.noise = 1e-8;
    std::vector<R::PoleInfo> genuine;   // the noise-free fit's in-band poles
    for (const R::PoleInfo& info : R::classify_poles(R::fit(samples, o), samples, o, po)) genuine.push_back(info);
    std::mt19937 rng(17);
    Normal g(0.0, 1.0);
    int poles = 0;
    int resolved = 0;
    int outside = 0;
    int non_passive = 0;
    for (int r = 0; r < 5; ++r) {
      std::vector<R::Sample> s = samples;
      for (auto& x : s) {
        for (int k = 0; k < x.value.size(); ++k) {
          const double a = g(rng);
          const double b = g(rng);
          x.value(k) *= 1.0 + 1e-8 * cplx(a, b) / std::sqrt(2.0);
        }
      }
      const std::vector<R::PoleInfo> classed = R::classify_poles(R::fit(s, o), s, o, po);
      for (const R::PoleInfo& ref : genuine) {
        const R::PoleInfo* best = nullptr;
        for (const R::PoleInfo& info : classed)
          if (!best || std::fabs(info.p.real() - ref.p.real()) < std::fabs(best->p.real() - ref.p.real())) best = &info;
        if (!best || std::fabs(best->p.real() - ref.p.real()) > 1e3) continue;
        ++poles;
        if (best->non_passive) ++non_passive;
        if (!best->lossy) continue;
        ++resolved;
        if (std::fabs(best->Q - ref.Q) > 3 * best->Q_sigma) ++outside;
      }
    }
    std::printf("18. lossy cavity under phase noise 1e-8, 5 realizations: %zu genuine poles, found %d times, %d with "
                "a resolved Q, %d beyond 3 sigma, %d non_passive\n", genuine.size(), poles, resolved, outside,
                non_passive);
    check(samples.size() == 23 && genuine.size() == 8 && poles == 40 && resolved >= 20 && outside == 0 &&
              non_passive == 0,
          "18. phase noise: resolved Q within 3 sigma (beyond)", outside);
  }

  // 21. one bad near-pole solve: the 10 first sweeps of the trials'
  // highq Q 1e9 row at noise 1e-10 (benchmarks/adaptive_sweep_trials.cpp highq 1e9 1e-10, AXF_TRIAL_DUMP=0..9;
  // cpp/tests/data/adaptive_highq1e9_samples.txt), with the solve nearest the pole multiplied by
  // 1 + 24e-10 e^{i phi} at 8 phases (the FEM's measured errors next to poles reach 24 x pole_noise), classed with the
  // noise 1e-10. A trimmed jackknife (without the refit farthest from the median) drops exactly the refit that shows
  // such a solve: 27 of the 80 Q lay beyond 3 sigma (z up to 9). The plain jackknife: 4, up to 9 sigma; none may be
  // non_passive. This is a regression guard (it pins the known residual at 4), not a coverage test: the residual is
  // documented in Limits. Of the 4, three are one 7-solve sweep at phases 1-3; in the fourth no solve lies within
  // 1e-3 of the pole, outside the measured FEM error structure (x24 only next to poles).
  {
    std::ifstream in(std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_highq1e9_samples.txt");
    std::vector<std::vector<R::Sample>> sweeps;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        sweeps.emplace_back();
        continue;
      }
      std::istringstream is(line);
      R::Sample s;
      int nports = 0;
      double zr = 0;
      double zi = 0;
      double pr = 0;
      double pi = 0;
      double dr = 0;
      double di = 0;
      is >> s.f >> nports >> zr >> zi >> pr >> pi >> dr >> di;
      s.value = Eigen::VectorXcd(2);
      s.value << cplx(zr, zi), cplx(pr, pi);
      s.derivative = Eigen::VectorXcd(2);
      s.derivative << cplx(dr, di), cplx(std::nan(""), 0.0);
      sweeps.back().push_back(s);
    }
    const R::FitOptions o = options(1e-6);
    R::PoleOptions po;
    po.f_min = 1e8;
    po.f_max = 2e8;
    po.tolerance = 1e-3;
    po.log_spacing = true;
    po.noise = 1e-10;
    int total = 0;
    int outside = 0;
    int non_passive = 0;
    for (const auto& samples : sweeps) {
      double f0 = 0;   // the in-band pole: the noise-free fit's resonance
      for (const R::PoleInfo& b : R::classify_poles(R::fit(samples, o), samples, o, po))
        if (b.cls == "resonance") f0 = b.p.real();
      size_t near = 0;
      for (size_t i = 1; i < samples.size(); ++i)
        if (std::fabs(samples[i].f - f0) < std::fabs(samples[near].f - f0)) near = i;
      for (int k = 0; k < 8; ++k) {
        std::vector<R::Sample> s = samples;
        s[near].value *= 1.0 + 24e-10 * std::polar(1.0, 2 * 3.14159265358979 * k / 8);
        for (const R::PoleInfo& info : R::classify_poles(R::fit(s, o), s, o, po)) {
          if (std::fabs(info.p.real() - f0) > 1e3) continue;
          ++total;
          if (info.non_passive) ++non_passive;
          if (info.lossy && std::fabs(info.Q - 1e9) > 3 * info.Q_sigma) ++outside;
        }
      }
    }
    std::printf("21. highq Q 1e9, the nearest solve x (1 + 24 nu e^{i phi}): %d poles, %d Q beyond 3 sigma, %d "
                "non_passive\n", total, outside, non_passive);
    check(sweeps.size() == 10 && total == 80 && outside <= 4 && non_passive == 0,
          "21. regression guard: Q beyond 3 sigma (of 80, at most 4)", outside);
  }

  // 20. trial 88: the final 9 solves of the trials' lossless row at
  // tolerance 1e-5, noise 1e-11, in Hz (benchmarks/adaptive_sweep_trials.cpp lossless 1e-5 1e-11 1e8,
  // AXF_TRIAL_DUMP=88; the response's poles in the file's '#' lines). The fit keeps a near-real doublet at
  // 164.915001 MHz, of strength 2.6e-14, that the capped cleanup cannot remove; its needle put the converged model
  // 24 x the tolerance off. remove_invisible_poles must remove it and leave the model within the tolerance.
  {
    const std::string path = std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_trial88_samples.txt";
    int nports = 0;
    const std::vector<R::Sample> samples = read_samples(path, nports);
    Response truth;   // in x = f / 1e8
    {
      std::ifstream in(path);
      std::string line;
      while (std::getline(in, line)) {
        if (line.rfind("# pole", 0) != 0) continue;
        std::istringstream is(line.substr(6));
        double re = 0;
        double im = 0;
        double a = 0;
        double b = 0;
        is >> re >> im >> a >> b;
        truth.poles.push_back(cplx(re, im));
        truth.a.push_back(a);
        truth.b.push_back(b);
      }
    }
    const double unit = 1e8;
    const R::FitOptions o = options(1e-8);
    R::Fit fit = R::fit(samples, o);
    auto error_over_band = [&](const R::Model& model) {
      double worst = 0;
      for (int i = 0; i <= 20000; ++i) {
        const double x = 1.0 + i / 20000.0;
        bool near_pole = false;
        for (const cplx& q : truth.poles) near_pole = near_pole || std::fabs(x - q.real()) < 1e-6;
        if (near_pole) continue;
        worst = std::max(worst, R::relative_error(model.value(x * unit), truth.at(x).value, fit.floor));
      }
      return worst;
    };
    auto doublet = [](const R::Model& model) {
      int n = 0;
      for (const cplx& q : model.poles()) n += std::fabs(q.real() - 164915001.0) < 10.0 ? 1 : 0;
      return n;
    };
    const double before = error_over_band(fit.model);
    const int had = doublet(fit.model);
    R::AdaptiveOptions ao;
    ao.f_min = unit;
    ao.f_max = 2 * unit;
    ao.tolerance = 1e-5;
    ao.candidates = 2000;
    ao.log_spacing = true;
    const int removed = R::remove_invisible_poles(fit, samples, o, ao, nullptr);
    const double after = error_over_band(fit.model);
    std::printf("20. trial 88: %zu solves, doublet at 164.915001 MHz %d; true error / tolerance %.3g before, %.3g "
                "after %d removal(s); misfit %.3g\n", samples.size(), had, before / ao.tolerance, after / ao.tolerance,
                removed, fit.sample_error);
    check(samples.size() == 9 && had == 1 && before > 10 * ao.tolerance, "20. the doublet is there before (error)",
          before);
    check(removed >= 1 && doublet(fit.model) == 0 && after < ao.tolerance, "20. removed; true error after", after);
    check(fit.sample_error <= R::kInvisibleFraction * ao.tolerance, "20. misfit after the removal", fit.sample_error);
  }

  // 22. a weak genuine mode next to a strong one: the final solves of
  // the trials' close-pair family (benchmarks/adaptive_sweep_trials.cpp lossless 1e-3 1e-11 1e8 with AXF_TRIAL_STREAMS=1,
  // AXF_TRIAL_WEAK="3e-7 0", AXF_TRIAL_WEAK_NEAR=3e-4, AXF_TRIAL_DUMP=155;
  // cpp/tests/data/adaptive_weak_pair_trial155_samples.txt): a lossless mode of strength 3e-7 at 1.8152532, 3e-4 from a
  // strong one. A removal whose checks are relative to |r(f)| deletes it (the strong neighbour dilutes them) and
  // leaves the model 2-4 x the tolerance off next to it. The leave-one-out refits reproduce it, so it
  // must stay.
  {
    const std::string path = std::string(AXF_SOURCE_DIR) + "/cpp/tests/data/adaptive_weak_pair_trial155_samples.txt";
    int nports = 0;
    const std::vector<R::Sample> samples = read_samples(path, nports);
    Response truth;   // in x = f / 1e8
    {
      std::ifstream in(path);
      std::string line;
      while (std::getline(in, line)) {
        if (line.rfind("# pole", 0) != 0) continue;
        std::istringstream is(line.substr(6));
        double re = 0;
        double im = 0;
        double a = 0;
        double b = 0;
        is >> re >> im >> a >> b;
        truth.poles.push_back(cplx(re, im));
        truth.a.push_back(a);
        truth.b.push_back(b);
      }
    }
    const double unit = 1e8;
    const double fw = truth.poles.back().real();   // the weak mode
    R::Fit fit = R::fit(samples, options(1e-6));
    auto has_weak = [&](const R::Model& m) {
      for (const cplx& q : m.poles())
        if (std::fabs(q.real() / unit - fw) < 1e-5 * fw) return true;
      return false;
    };
    auto local_error = [&](const R::Model& m) {   // next to the weak mode, 1e-6 to 1e-4 (relative) from it
      double worst = 0;
      for (double d : {1e-6, 3e-6, 1e-5, 3e-5, 1e-4})
        for (double sgn : {-1.0, 1.0}) {
          const double x = fw * (1 + sgn * d);
          worst = std::max(worst, R::relative_error(m.value(x * unit), truth.at(x).value, fit.floor));
        }
      return worst;
    };
    const bool before = has_weak(fit.model);
    R::AdaptiveOptions ao;
    ao.f_min = unit;
    ao.f_max = 2 * unit;
    ao.tolerance = 1e-3;
    ao.candidates = 2000;
    ao.log_spacing = true;
    const int removed = R::remove_invisible_poles(fit, samples, options(1e-6), ao, nullptr);
    const bool after = has_weak(fit.model);
    const double err = local_error(fit.model);
    std::printf("22. weak mode 3e-4 from a strong one: in the fit %d, after the removal %d (%d removed); error next to "
                "it %.3g x tolerance\n", before ? 1 : 0, after ? 1 : 0, removed, err / ao.tolerance);
    check(samples.size() == 14 && before && after && err < ao.tolerance, "22. the weak mode stays (local error)", err);
  }
#endif

  // 19. the closure's gate. The comparison model is the model plus
  // one term c / (f - q) in each output, q above the band's axis: q is a spurious pole of r-, its own term accounts
  // for the whole estimate (to the fits' accuracy), and the gate's excused set is where that term is at least half
  // the estimate. A q farther above the axis than a quarter of the band is not taken; a pole below the axis neither.
  {
    std::vector<R::Sample> base;
    std::vector<R::Sample> plus;
    std::vector<R::Sample> far;
    const cplx q(1.47, 0.02);
    const cplx q_far(1.47, 0.4);
    const cplx c(2e-4, 1e-4);
    for (int i = 0; i <= 40; ++i) {
      const double f = 1.0 + i / 40.0;
      R::Sample s = lossy.at(f);
      base.push_back(s);
      R::Sample t = s;
      for (int k = 0; k < 2; ++k) t.value(k) += c / (f - q);
      t.derivative(0) -= c / ((f - q) * (f - q));
      plus.push_back(t);
      R::Sample u = s;
      for (int k = 0; k < 2; ++k) u.value(k) += 30.0 * c / (f - q_far);
      u.derivative(0) -= 30.0 * c / ((f - q_far) * (f - q_far));
      far.push_back(u);
    }
    R::Fit fit = R::fit(base, options(1e-11));
    fit.lower = R::fit(plus, options(1e-11)).model;
    const std::vector<cplx> spurious = R::spurious_comparison_poles(fit, 1.0);
    const bool found = spurious.size() == 1 && std::abs(spurious[0] - q) < 1e-6;
    std::vector<double> f;
    for (int i = 0; i <= 400; ++i) f.push_back(1.0 + i / 400.0);
    const Eigen::VectorXd estimate = R::estimated_error(fit, f);
    const Eigen::VectorXd terms = R::spurious_terms(fit, spurious, f);
    double mismatch = 0;   // the term against the estimate it accounts for
    for (int i = 0; i < estimate.size(); ++i) mismatch = std::max(mismatch, std::fabs(terms(i) / estimate(i) - 1));
    std::printf("19. gate: %zu spurious pole(s) of r- (%.6f%+.6fi); estimate up to %.3g, its own term within %.2g\n",
                spurious.size(), spurious.empty() ? 0.0 : spurious[0].real(),
                spurious.empty() ? 0.0 : spurious[0].imag(), estimate.maxCoeff(), mismatch);
    check(found && mismatch < 1e-3, "19. the spurious pole's term is the estimate (mismatch)", mismatch);
    R::Fit farther = fit;
    farther.lower = R::fit(far, options(1e-11)).model;
    const std::vector<cplx> none = R::spurious_comparison_poles(farther, 1.0);
    R::Fit below = fit;
    std::vector<R::Sample> lower_half = plus;
    for (size_t i = 0; i < lower_half.size(); ++i) {
      const double x = base[i].f;
      for (int k = 0; k < 2; ++k) lower_half[i].value(k) = base[i].value(k) + c / (x - std::conj(q));
      lower_half[i].derivative(0) = base[i].derivative(0) - c / ((x - std::conj(q)) * (x - std::conj(q)));
    }
    below.lower = R::fit(lower_half, options(1e-11)).model;
    const std::vector<cplx> passive = R::spurious_comparison_poles(below, 1.0);
    check(none.empty() && passive.empty(), "19. none 0.4 band widths up, none below the axis (count)",
          static_cast<double>(none.size() + passive.size()));
    // the noise-bound rule: the fit tolerance below 10 x the solves' accuracy
    const bool rule = R::noise_bound(1e-7, 1e-7) && R::noise_bound(9e-7, 1e-7) && !R::noise_bound(1e-6, 1e-7);
    check(rule, "19. noise_bound(fit tolerance, noise) (1 = as documented)", rule ? 1.0 : 0.0);
  }

  // 16. no admissible candidate: every frequency but the initial ones is refused, and the refused neighbourhoods
  // close the band before max_solves
  {
    R::AdaptiveOptions o;
    o.f_min = 1.0;
    o.f_max = 2.0;
    o.tolerance = 1e-3;
    o.max_solves = 500;
    o.initial = {1.0, 1.5, 2.0};
    o.log_spacing = false;
    const R::AdaptiveResult res = R::adaptive_sweep(o, options(1e-6), [&](double f) {
      if (f == 1.0 || f == 1.5 || f == 2.0) return std::optional<R::Sample>(lossy.at(f));
      return std::optional<R::Sample>();
    });
    const bool ok = !res.converged && res.stop_reason == "no admissible candidate frequency" &&
                    static_cast<int>(res.attempted.size()) < o.max_solves;
    std::printf("16. no admissible candidate: %zu attempts, %s\n", res.attempted.size(), res.stop_reason.c_str());
    check(ok, "16. no admissible candidate: unconverged before the budget (1 = yes)", ok ? 1.0 : 0.0);
  }

  std::printf("%s\n", failures == 0 ? "all rational-fit checks passed" : "rational-fit checks FAILED");
  return failures == 0 ? 0 : 1;
}
