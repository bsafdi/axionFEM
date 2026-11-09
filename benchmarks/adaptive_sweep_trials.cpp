// Random-trial statistics of the adaptive sweep's driver and pole classes (cpp/src/rational_fit.cpp;
// docs/physics/adaptive_sweep.md, "Trials"). Not part of the build: compile it against
// rational_fit.cpp and core.cpp; it writes benchmarks/records/adaptive_sweep_trials.json.
//
//   adaptive_sweep_trials KIND TOLERANCE NOISE UNIT TRIALS [HERMITE [BIAS]]
//     KIND       lossless | lossy | mixed (0-6 in-band poles, each lossless or lossy with Q = 10^U(1, 7) with equal
//                probability, a draw sequence of its own: seed 99)
//     TOLERANCE  the sweep's tolerance
//     NOISE      relative Gaussian noise on every value and derivative
//     UNIT       the band is [UNIT, 2 UNIT]: 1 (normalized) or 1e8 (Hz, as production runs)
//     TRIALS     random responses
//     HERMITE    1 (default): dZ/df as derivative rows; 0: values only
//     BIAS       b (default 0): b x |Z| per band width added to dZ/df, derivatives inconsistent with the values (a
//                missing term of A')
//   adaptive_sweep_trials highq Q NOISE TRIALS [TOLERANCE]
//     one in-band pole of quality Q (0: lossless) among out-of-band ones, in Hz: how its class and Q come out
//   adaptive_sweep_trials refitnoise FILE F_MIN F_MAX TOLERANCE SPACING REALIZATIONS NOISE [phase]
//     a sweep's solves (FILE: one line per solve, "f nports, Z row-major and Psi as re im pairs, then dZ/df row-major
//     as re im pairs", benchmarks/adaptive_sweep.py samples) refitted with the engine's options under relative noise:
//     how each in-band pole of the noise-free fit is classed, the actual RMS scatter of its fitted Im p over the
//     realizations against the median sigma_Im the classes use
//
// Noise: relative Gaussian, magnitude only (each value times 1 + NOISE g, g real) by default; with the environment
// variable AXF_TRIAL_PHASE=1 (refitnoise: the argument `phase`) with a phase, 1 + NOISE (g1 + i g2) / sqrt(2), as the
// error of a complex-arithmetic solve ("phase noise").
// By default the responses and the noise come from one random stream, as in the records, so a change of the
// driver that changes one sweep's number of solves changes the responses and the noise of every later trial. With
// AXF_TRIAL_STREAMS=1 (trials and highq modes) each trial's noise has its own stream (seed 1000003 (t + 1)): two
// drivers then see the same responses and noise trial by trial, for a before/after comparison.
// Exact replays of a shared-stream trial: AXF_TRIAL_STATE_OUT=t:FILE writes the random state before trial t (run
// with the driver that defines the trial, e.g. the baseline); AXF_TRIAL_STATE_IN=FILE starts there (trial index t)
// and runs TRIALS trials, so another driver replays the same response and noise.
// Weak modes (trials mode): AXF_TRIAL_WEAK="AW QW" (or AW,QW) adds one weak mode of strength AW and quality QW (0: lossless) to
// each response, at 1.1 + 0.8 U in the band (its own stream, seed 4242 + t), or with AXF_TRIAL_WEAK_NEAR=d at the
// relative offset d from the first in-band pole (a close pair; trials without one are skipped).
// Tallied: whether the fit before the removal after
// convergence has the weak mode (within 1e-5), whether the final model does.
// Needles (trials mode): every converged model is also evaluated at Re q (1, 1 +- 1e-9, 1e-8, 1e-7) for each of its
// in-band poles with |Im q| below a quarter grid step that is not within 1e-6 of a true pole; a needle above the
// tolerance counts in needle_wrong (the 20001-point grid cannot see one narrower than 5e-5 of the band).
//
// Each trial: Z = -0.3 i f + sum_n i a_n f / (p_n^2 - f^2) with dZ/df, Psi = 1 + sum_n b_n p_n^2 / (p_n^2 - f^2), in
// the band's normalized frequency x = f / UNIT; 0-3 poles in [1, 2], two above it, one below; a_n = 10^U(-2, 0),
// b_n ~ N(0, 0.3); lossless (real p_n) or Q = 10^U(1, 5); 5 log-spaced initial points; max_solves 60; fit tolerance
// max(1e-3 x tolerance, 1e-10), as the engine. "True" error: the final model against the exact response on 20001
// points, in the fit's measure, outside 1e-6 of a real pole. For each converged trial whose true error exceeds the
// tolerance, wrong_detail gives where: the distance of the worst point to the nearest true pole and that pole's error
// in the model (both relative), and whether the point lies inside |f - p| < |dp| / tolerance of a lossless pole.
// Pole classes (rational::classify_poles, as the engine): of the true in-band poles, how many are found as
// `resonance`, how many lossless ones are classed non_passive or get a resolved Q, how many lossy ones get a Q off by
// more than 10 %; and the `resonance` poles of converged sweeps that match no true pole (spurious). Also: sweeps
// flagged noise-bound (rational::noise_bound: the fit tolerance below kNoiseBoundFactor x the noise) and the wrong
// stops among the others; closures on tight confirmations and the wrong ones among them; invisible near-real poles
// removed after convergence. Prints one JSON object.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/rational_fit.hpp"

using axf::rational::cplx;
namespace R = axf::rational;

#ifdef AXF_TRIAL_BASELINE
// compiled against a rational_fit.cpp from before these were added (the before/after comparison of a fix)
namespace axf::rational {
constexpr double kNoiseBoundFactor = 10.0;
inline bool noise_bound(double fit_tolerance, double noise) { return fit_tolerance < kNoiseBoundFactor * noise; }
}  // namespace axf::rational
#define AXF_TRIAL_REMOVED(res) 0
#else
#define AXF_TRIAL_REMOVED(res) (res).invisible_poles_removed
#endif

namespace {

bool phase_noise() {
  const char* v = std::getenv("AXF_TRIAL_PHASE");
  return v && std::atoi(v) != 0;
}

// relative noise on a complex value: magnitude only, or with a phase (two draws)
cplx perturb(cplx v, double level, bool with_phase, std::mt19937& rng, std::normal_distribution<double>& N) {
  if (!with_phase) return v * (1.0 + level * N(rng));
  const double a = N(rng);
  const double b = N(rng);
  return v * (1.0 + level * cplx(a, b) / std::sqrt(2.0));
}

// the engine's pole options; the data noise is the trial's (the engine takes its solves' relative residual), at
// least 1e-12 as the engine's
R::PoleOptions pole_options(double f_min, double f_max, double tol, bool log_spacing, double noise) {
  R::PoleOptions po;
  po.f_min = f_min;
  po.f_max = f_max;
  po.tolerance = tol;
  po.candidates = 2000;
  po.log_spacing = log_spacing;
  po.noise = std::max(noise, 1e-12);
  return po;
}

// how the true in-band poles come out in a final fit
struct PoleTally {
  int true_in_band = 0;
  int found_resonance = 0;        // matched (within 1e-6 relative in Re p) and classed resonance
  int lossless_non_passive = 0;   // a lossless true pole classed non_passive
  int lossless_given_Q = 0;       // a lossless true pole given a resolved Q
  int lossy_Q_off = 0;            // a lossy true pole whose Q is missing or off by more than 10 %
  int lossy_Q_unresolved = 0;     // a lossy true pole reported lossless within the fit
  int lossy_Q_outside = 0;        // a resolved Q more than 3 standard errors from the true Q
  int lossy_bound_violated = 0;   // an unresolved lossy pole whose bound Q_min exceeds the true Q
  int spurious_resonance = 0;     // resonance poles matching no true pole (converged sweeps)

  void add(const PoleTally& t) {
    true_in_band += t.true_in_band;
    found_resonance += t.found_resonance;
    lossless_non_passive += t.lossless_non_passive;
    lossless_given_Q += t.lossless_given_Q;
    lossy_Q_off += t.lossy_Q_off;
    lossy_Q_unresolved += t.lossy_Q_unresolved;
    lossy_Q_outside += t.lossy_Q_outside;
    lossy_bound_violated += t.lossy_bound_violated;
    spurious_resonance += t.spurious_resonance;
  }
};

PoleTally tally(const std::vector<R::PoleInfo>& classed, const std::vector<cplx>& truth, double unit, bool converged) {
  PoleTally t;
  for (const cplx& q : truth) {
    if (!(q.real() >= 1.0 && q.real() <= 2.0)) continue;
    ++t.true_in_band;
    const R::PoleInfo* best = nullptr;
    for (const auto& c : classed)
      if (!best || std::fabs(c.p.real() / unit - q.real()) < std::fabs(best->p.real() / unit - q.real())) best = &c;
    if (!best || std::fabs(best->p.real() / unit - q.real()) > 1e-6 * q.real()) continue;
    if (best->cls == "resonance") ++t.found_resonance;
    const bool lossless = q.imag() == 0.0;
    if (lossless && best->non_passive) ++t.lossless_non_passive;
    if (lossless && best->lossy) ++t.lossless_given_Q;
    if (!lossless) {
      const double Qtrue = q.real() / (2 * std::fabs(q.imag()));
      if (!best->lossy) ++t.lossy_Q_unresolved;
      if (!best->lossy || std::fabs(best->Q / Qtrue - 1) > 0.1) ++t.lossy_Q_off;
      if (best->lossy && std::fabs(best->Q - Qtrue) > 3 * best->Q_sigma) ++t.lossy_Q_outside;
      if (!best->lossy && !best->non_passive && best->Q_min > Qtrue) ++t.lossy_bound_violated;
    }
  }
  if (converged) {
    for (const auto& c : classed) {
      if (c.cls != "resonance") continue;
      bool matches = false;
      for (const cplx& q : truth) matches = matches || std::fabs(c.p.real() / unit - q.real()) <= 1e-3 * q.real();
      if (!matches) ++t.spurious_resonance;
    }
  }
  return t;
}

void print_tally(const PoleTally& t) {
  std::printf("\"true_in_band_poles\": %d, \"found_as_resonance\": %d, \"lossless_classed_non_passive\": %d, "
              "\"lossless_given_Q\": %d, \"lossy_Q_off_10pc\": %d, \"lossy_Q_unresolved\": %d, "
              "\"lossy_Q_outside_3_sigma\": %d, \"lossy_Q_min_above_true\": %d, \"spurious_resonance\": %d",
              t.true_in_band, t.found_resonance, t.lossless_non_passive, t.lossless_given_Q, t.lossy_Q_off,
              t.lossy_Q_unresolved, t.lossy_Q_outside, t.lossy_bound_violated, t.spurious_resonance);
}

// the trials' response with poles p (in x = f / unit), and its exact sample at x
struct Response {
  std::vector<cplx> p;
  std::vector<double> a;
  std::vector<double> b;
  double unit = 1;
  double bias = 0;   // b x |Z| per band width added to dZ/df

  R::Sample at(double x, bool biased) const {
    cplx Z = cplx(0, -0.3) * x;
    cplx dZ = cplx(0, -0.3);
    cplx Psi = 1.0;
    for (size_t n = 0; n < p.size(); ++n) {
      const cplx p2 = p[n] * p[n];
      Z += cplx(0, a[n]) * x / (p2 - x * x);
      dZ += cplx(0, a[n]) * (p2 + x * x) / ((p2 - x * x) * (p2 - x * x));
      Psi += b[n] * p2 / (p2 - x * x);
    }
    if (biased) dZ += bias * std::abs(Z);   // the band width is 1 in x
    R::Sample s;
    s.f = x * unit;
    s.value = Eigen::VectorXcd(2);
    s.value << Z, Psi;
    s.derivative = Eigen::VectorXcd(2);
    s.derivative << dZ / unit, cplx(std::nan(""), 0.0);   // d/df
    return s;
  }
};

R::FitOptions trial_fit_options(double tol, bool hermite) {
  R::FitOptions fo;
  fo.group = {0, 1};
  fo.hermite = {static_cast<char>(hermite ? 1 : 0), 0};
  fo.fit_tolerance = std::max(1e-3 * tol, 1e-10);
  return fo;
}

R::AdaptiveOptions trial_options(double unit, double tol) {
  R::AdaptiveOptions o;
  o.f_min = unit;
  o.f_max = 2 * unit;
  o.tolerance = tol;
  o.max_solves = 60;
  o.initial = R::band_grid(unit, 2 * unit, 5, true);
  o.log_spacing = true;
  return o;
}

int trials(int argc, char** argv) {
  const std::string kind = argv[1];
  const bool lossless = kind == "lossless";
  const bool mixed = kind == "mixed";
  const bool with_phase = phase_noise();
  const char* streams_env = std::getenv("AXF_TRIAL_STREAMS");
  const bool streams = streams_env && std::atoi(streams_env) != 0;
  const double tol = std::atof(argv[2]);
  const double noise_level = std::atof(argv[3]);
  const double unit = std::atof(argv[4]);
  const int ntrials = std::atoi(argv[5]);
  const bool hermite = argc > 6 ? std::atoi(argv[6]) != 0 : true;
  const double bias = argc > 7 ? std::atof(argv[7]) : 0.0;
  std::mt19937 rng(mixed ? 99 : 7);
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> N(0, 1);
  int t_first = 0;
  if (const char* in = std::getenv("AXF_TRIAL_STATE_IN")) {
    std::ifstream is(in);
    is >> t_first >> rng >> N;
    if (!is) {   // a missing or unreadable state would silently replay trial 0 of the seed
      std::fprintf(stderr, "adaptive_sweep_trials: cannot read the random state AXF_TRIAL_STATE_IN=%s\n", in);
      std::exit(3);
    }
  }
  int state_out_trial = -1;
  std::string state_out_file;
  if (const char* out = std::getenv("AXF_TRIAL_STATE_OUT")) {
    const std::string v(out);
    state_out_trial = std::atoi(v.substr(0, v.find(':')).c_str());
    state_out_file = v.substr(v.find(':') + 1);
  }
  double weak_a = 0;
  double weak_q = 0;
  if (const char* w = std::getenv("AXF_TRIAL_WEAK")) {   // "AW QW" or "AW,QW"
    std::string v(w);
    std::replace(v.begin(), v.end(), ',', ' ');
    std::istringstream is(v);
    is >> weak_a >> weak_q;
  }
  const char* weak_near_env = std::getenv("AXF_TRIAL_WEAK_NEAR");
  const double weak_near = weak_near_env ? std::atof(weak_near_env) : 0.0;
  int skipped = 0;
  int weak_found_before = 0;
  int weak_in_final = 0;
  int weak_removed = 0;
  std::vector<int> weak_removed_trials;
  int needle_wrong = 0;
  int removal_made_wrong = 0;   // grid error above the tolerance after the removal, not before
  int removal_fixed = 0;
  double removal_made_wrong_worst = 0;
  double needle_worst = 0;
  int converged_count = 0;
  int not_converged = 0;
  int noise_bound = 0;
  int wrong_not_noise_bound = 0;
  double worst_wrong_not_noise_bound = 0;
  int closure_wrong = 0;
  double closure_worst = 0;
  int invisible_removed = 0;
  int wrong = 0;
  double worst_wrong = 0;
  double sum_solves = 0;
  int max_solves = 0;
  int hermite_dropped = 0;
  int by_confirmations = 0;
  int wrong_near_pole = 0;
  std::vector<std::string> wrong_detail;
  std::vector<int> unconverged_trials;
  std::vector<double> ratios;
  PoleTally poles;
  for (int t = t_first; t < t_first + ntrials; ++t) {
    if (t == state_out_trial) {
      std::ofstream os(state_out_file);
      os << t << "\n" << rng << "\n" << N << "\n";
      if (!os) {
        std::fprintf(stderr, "adaptive_sweep_trials: cannot write AXF_TRIAL_STATE_OUT=%s\n", state_out_file.c_str());
        std::exit(3);
      }
      state_out_trial = -2;   // written
    }
    const int n_in = static_cast<int>(U(rng) * (mixed ? 7 : 4));
    std::vector<double> fn;
    for (int i = 0; i < n_in; ++i) fn.push_back(1 + U(rng));
    for (int i = 0; i < 2; ++i) fn.push_back(2 + 4 * U(rng));
    fn.push_back(0.2 + 0.8 * U(rng));
    Response r;
    r.unit = unit;
    r.bias = bias;
    for (double f : fn) {
      const bool pole_lossless = mixed ? U(rng) < 0.5 : lossless;
      double Q = 1e300;
      if (!pole_lossless) Q = mixed ? std::pow(10.0, 1 + 6 * U(rng)) : std::pow(10.0, 1 + 4 * U(rng));
      r.p.push_back(cplx(f, pole_lossless ? 0.0 : -0.5 * f / Q));
      r.a.push_back(std::pow(10.0, -2 + 2 * U(rng)));
      r.b.push_back(0.3 * N(rng));
    }
    double fw = 0;   // the weak mode, if any
    if (weak_a > 0) {
      std::mt19937 wrng(4242u + static_cast<unsigned>(t));
      std::uniform_real_distribution<double> wU(0, 1);
      fw = 1.1 + 0.8 * wU(wrng);
      if (weak_near > 0) {
        double strong = 0;
        for (const cplx& q : r.p)
          if (q.real() > 1.0 && q.real() < 2.0) {
            strong = q.real();
            break;
          }
        if (strong == 0) {
          ++skipped;
          continue;
        }
        fw = strong * (1 + weak_near);
      }
      r.p.push_back(weak_q > 0 ? cplx(fw, -0.5 * fw / weak_q) : cplx(fw, 0.0));
      r.a.push_back(weak_a);
      r.b.push_back(weak_a);
    }
    const R::FitOptions fo = trial_fit_options(tol, hermite);
    std::mt19937 noise_rng(1000003u * static_cast<unsigned>(t + 1));
    std::normal_distribution<double> noise_N(0, 1);
    // AXF_TRIAL_LOG=t: the sweep log of trial t on stderr (the trials before it run as usual, so the random draws match)
    const char* log_trial = std::getenv("AXF_TRIAL_LOG");
    const bool logged = log_trial && std::atoi(log_trial) == t;
    axf::log_level() = logged ? 1 : 0;
    const R::AdaptiveResult res = R::adaptive_sweep(trial_options(unit, tol), fo, [&](double f) {
      R::Sample s = r.at(f / unit, true);
      std::mt19937& g = streams ? noise_rng : rng;
      std::normal_distribution<double>& n = streams ? noise_N : N;
      for (int k = 0; k < 2; ++k) s.value(k) = perturb(s.value(k), noise_level, with_phase, g, n);
      s.derivative(0) = perturb(s.derivative(0), noise_level, with_phase, g, n);
      return std::optional<R::Sample>(s);
    });
    double err = 0;
    double x_worst = 1.0;
    for (int i = 0; i <= 20000; ++i) {
      const double x = 1.0 + i / 20000.0;
      bool near = false;
      for (const cplx& q : r.p) near = near || (q.imag() == 0.0 && std::fabs(x - q.real()) < 1e-6);
      if (near) continue;
      const double e = R::relative_error(res.fit.model.value(x * unit), r.at(x, false).value, res.fit.floor);
      if (e > err) {
        err = e;
        x_worst = x;
      }
    }
    if (!res.converged) {
      ++not_converged;
      unconverged_trials.push_back(t);
    } else {
      ++converged_count;
      // needles: the model at its own near-real in-band poles that match no true pole
      const double quarter = 0.25 * (std::pow(2.0, 1.0 / 1999) - 1.0);   // relative, the 2000-point log grid
      double needle = 0;
      if (res.fit.model.size() > 1)
        for (const cplx& q : res.fit.model.poles()) {
          const double x0 = q.real() / unit;
          if (!(x0 > 1.0 && x0 < 2.0) || std::fabs(q.imag()) >= quarter * q.real()) continue;
          bool genuine = false;
          for (const cplx& z : r.p) genuine = genuine || std::fabs(x0 - z.real()) < 1e-6 * z.real();
          if (genuine) continue;
          for (double d : {0.0, 1e-9, -1e-9, 1e-8, -1e-8, 1e-7, -1e-7}) {
            const double x = x0 * (1 + d);
            const double e = R::relative_error(res.fit.model.value(x * unit), r.at(x, false).value, res.fit.floor);
            needle = std::max(needle, std::isfinite(e) ? e : 1e300);
          }
        }
      if (needle > tol) ++needle_wrong;
      // the removal after convergence: the loop's last fit (before it) against the final model, on the same grid
      if (AXF_TRIAL_REMOVED(res) > 0) {
        const R::Fit pre = R::fit(res.samples, res.options);
        double before = 0;
        for (int i = 0; i <= 20000; ++i) {
          const double x = 1.0 + i / 20000.0;
          bool near = false;
          for (const cplx& q : r.p) near = near || (q.imag() == 0.0 && std::fabs(x - q.real()) < 1e-6);
          if (near) continue;
          before = std::max(before, R::relative_error(pre.model.value(x * unit), r.at(x, false).value, pre.floor));
        }
        if (before <= tol && err > tol) {
          ++removal_made_wrong;
          removal_made_wrong_worst = std::max(removal_made_wrong_worst, err / tol);
        }
        if (before > tol && err <= tol) ++removal_fixed;
      }
      needle_worst = std::max(needle_worst, needle / tol);
      if (fw > 0) {
        auto has_weak = [&](const R::Model& m) {
          if (m.size() < 2) return false;
          for (const cplx& q : m.poles())
            if (std::fabs(q.real() - fw * unit) < 1e-5 * fw * unit) return true;
          return false;
        };
        const bool before = has_weak(R::fit(res.samples, res.options).model);
        const bool after = has_weak(res.fit.model);
        weak_found_before += before ? 1 : 0;
        weak_in_final += after ? 1 : 0;
        weak_removed += before && !after ? 1 : 0;
        if (before && !after) weak_removed_trials.push_back(t);
      }
    }
    if (res.converged && res.stop_reason != "converged") ++by_confirmations;
    const bool flagged = R::noise_bound(fo.fit_tolerance, noise_level);
    if (flagged) ++noise_bound;
    invisible_removed += AXF_TRIAL_REMOVED(res);
    if (res.converged && err > tol && !flagged) {
      ++wrong_not_noise_bound;
      worst_wrong_not_noise_bound = std::max(worst_wrong_not_noise_bound, err / tol);
    }
    if (res.converged && res.stop_reason != "converged" && err > tol) {
      ++closure_wrong;
      closure_worst = std::max(closure_worst, err / tol);
    }
    if (res.converged && err > tol) {
      ++wrong;
      worst_wrong = std::max(worst_wrong, err / tol);
      // Where the error sits: the distance to the nearest true pole and that pole's error in the model (both relative
      // to the pole). Near a lossless pole the pointwise error is about |dp| / |f - p|, so it exceeds the tolerance
      // inside |f - p| < |dp| / tolerance whatever the fit (docs/physics/adaptive_sweep.md, Limits).
      size_t n_near = 0;
      for (size_t n = 1; n < r.p.size(); ++n) {
        if (std::fabs(x_worst - r.p[n].real()) < std::fabs(x_worst - r.p[n_near].real())) n_near = n;
      }
      double dp = 1e300;
      for (const cplx& q : res.fit.model.poles()) dp = std::min(dp, std::abs(q / unit - r.p[n_near]));
      const double d = std::fabs(x_worst - r.p[n_near].real()) / r.p[n_near].real();
      const double rel_dp = dp / std::abs(r.p[n_near]);
      const bool inside = r.p[n_near].imag() == 0.0 && d < rel_dp / tol;
      if (inside) ++wrong_near_pole;
      char buf[256];
      std::snprintf(buf, sizeof buf,
                    "{\"trial\": %d, \"error_over_tolerance\": %.3g, \"distance_to_pole\": %.3g, \"pole_error\": %.3g, "
                    "\"inside_pole_error_over_tolerance\": %s}",
                    t, err / tol, d, rel_dp, inside ? "true" : "false");
      wrong_detail.push_back(buf);
    }
    // AXF_TRIAL_DUMP=t: the response of trial t ("# pole re im a b" lines) and its final solves on stderr, as
    // refitnoise reads them (a regression case: cpp/tests/data/adaptive_trial88_samples.txt)
    const char* dump_trial = std::getenv("AXF_TRIAL_DUMP");
    if (dump_trial && std::atoi(dump_trial) == t) {
      for (size_t n = 0; n < r.p.size(); ++n)
        std::fprintf(stderr, "# pole %.17g %.17g %.17g %.17g\n", r.p[n].real(), r.p[n].imag(), r.a[n], r.b[n]);
      for (const R::Sample& x : res.samples)
        std::fprintf(stderr, "%.17g 1 %.17g %.17g %.17g %.17g %.17g %.17g\n", x.f, x.value(0).real(), x.value(0).imag(),
                     x.value(1).real(), x.value(1).imag(), x.derivative(0).real(), x.derivative(0).imag());
    }
    if (hermite && !res.hermite) ++hermite_dropped;
    ratios.push_back(err / tol);
    sum_solves += res.attempted.size();
    max_solves = std::max(max_solves, static_cast<int>(res.attempted.size()));
    const auto classed =
        R::classify_poles(res.fit, res.samples, res.options, pole_options(unit, 2 * unit, tol, true, noise_level));
    poles.add(tally(classed, r.p, unit, res.converged));
    if (logged) {
      std::fprintf(stderr, "trial %d: %s after %zu solves, true error %.3g x tolerance; poles of r and r-:\n", t,
                   res.stop_reason.c_str(), res.attempted.size(), err / tol);
      for (const cplx& q : res.fit.model.poles()) std::fprintf(stderr, "  r  %.9g %+.4g i\n", q.real(), q.imag());
      if (res.fit.lower.size() > 1)
        for (const cplx& q : res.fit.lower.poles()) std::fprintf(stderr, "  r- %.9g %+.4g i\n", q.real(), q.imag());
      for (const cplx& q : r.p) std::fprintf(stderr, "  true %.9g %+.4g i\n", q.real() * unit, q.imag() * unit);
    }
  }
  if (state_out_trial >= 0) {   // the requested trial was not reached
    std::fprintf(stderr, "adaptive_sweep_trials: trial %d of AXF_TRIAL_STATE_OUT not in this run\n", state_out_trial);
    std::exit(3);
  }
  std::sort(ratios.begin(), ratios.end());
  std::printf("{\"poles\": \"%s\", \"tolerance\": %.0e, \"noise\": %.0e, \"phase_noise\": %s, "
              "\"noise_streams\": %s, \"unit_Hz\": %.0e, \"hermite\": %s, "
              "\"derivative_bias\": %.0e, \"trials\": %d, \"mean_solves\": %.2f, \"max_solves\": %d, "
              "\"not_converged\": %d, \"converged_on_confirmations\": %d, \"converged_but_wrong\": %d, "
              "\"worst_wrong_over_tolerance\": %.3g, \"median_error_over_tolerance\": %.3g, \"hermite_dropped\": %d, "
              "\"wrong_inside_pole_neighbourhood\": %d, \"noise_bound\": %d, \"wrong_not_noise_bound\": %d, "
              "\"worst_wrong_not_noise_bound\": %.3g, \"closure_wrong\": %d, \"closure_worst\": %.3g, "
              "\"invisible_poles_removed\": %d, \"needle_wrong\": %d, \"needle_worst_over_tolerance\": %.3g, "
              "\"weak_mode\": [%.1e, %g, %g], \"trials_skipped\": %d, \"weak_found_before_removal\": %d, "
              "\"weak_in_final_model\": %d, \"weak_removed\": %d, \"converged\": %d, "
              "\"removal_made_grid_wrong\": %d, \"removal_made_grid_wrong_worst\": %.3g, "
              "\"removal_fixed_grid\": %d, \"first_trial\": %d, ",
              kind.c_str(), tol, noise_level, with_phase ? "true" : "false", streams ? "true" : "false", unit,
              hermite ? "true" : "false", bias,
              ntrials, sum_solves / std::max(1, ntrials - skipped), max_solves, not_converged, by_confirmations, wrong, worst_wrong,
              ratios[ratios.size() / 2], hermite_dropped, wrong_near_pole, noise_bound, wrong_not_noise_bound,
              worst_wrong_not_noise_bound, closure_wrong, closure_worst, invisible_removed, needle_wrong, needle_worst,
              weak_a, weak_q, weak_near, skipped, weak_found_before, weak_in_final, weak_removed, converged_count,
              removal_made_wrong, removal_made_wrong_worst, removal_fixed, t_first);
  print_tally(poles);
  std::printf(", \"unconverged_trials\": [");
  for (size_t i = 0; i < unconverged_trials.size(); ++i) std::printf("%s%d", i ? ", " : "", unconverged_trials[i]);
  std::printf("], \"weak_removed_trials\": [");
  for (size_t i = 0; i < weak_removed_trials.size(); ++i) std::printf("%s%d", i ? ", " : "", weak_removed_trials[i]);
  std::printf("], \"wrong_detail\": [");
  for (size_t i = 0; i < wrong_detail.size(); ++i) {
    std::printf("%s%s", i ? ", " : "", wrong_detail[i].c_str());
  }
  std::printf("]}\n");
  return 0;
}

// one in-band pole of quality Q (0: lossless), the band [1e8, 2e8] Hz
int highq(int argc, char** argv) {
  const double Qtrue = std::atof(argv[2]);
  const double noise_level = std::atof(argv[3]);
  const int ntrials = std::atoi(argv[4]);
  const double tol = argc > 5 ? std::atof(argv[5]) : 1e-3;
  const double unit = 1e8;
  const bool with_phase = phase_noise();
  const char* streams_env = std::getenv("AXF_TRIAL_STREAMS");
  const bool streams = streams_env && std::atoi(streams_env) != 0;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> N(0, 1);
  int not_converged = 0;
  double sum_solves = 0;
  PoleTally poles;
  std::vector<double> q_error;
  for (int t = 0; t < ntrials; ++t) {
    Response r;
    r.unit = unit;
    const double f0 = 1.1 + 0.8 * U(rng);
    r.p.push_back(Qtrue > 0 ? cplx(f0, -0.5 * f0 / Qtrue) : cplx(f0, 0.0));
    r.a.push_back(std::pow(10.0, -2 + 2 * U(rng)));
    r.b.push_back(0.3 * N(rng));
    for (int i = 0; i < 2; ++i) {
      const double f = 2.2 + 3.8 * U(rng);
      r.p.push_back(cplx(f, -0.5 * f / std::pow(10.0, 1 + 3 * U(rng))));
      r.a.push_back(std::pow(10.0, -2 + 2 * U(rng)));
      r.b.push_back(0.3 * N(rng));
    }
    const double fl = 0.2 + 0.7 * U(rng);
    r.p.push_back(cplx(fl, -0.5 * fl / 30.0));
    r.a.push_back(std::pow(10.0, -2 + 2 * U(rng)));
    r.b.push_back(0.3 * N(rng));
    std::mt19937 noise_rng(1000003u * static_cast<unsigned>(t + 1));
    std::normal_distribution<double> noise_N(0, 1);
    std::mt19937& g = streams ? noise_rng : rng;
    std::normal_distribution<double>& n = streams ? noise_N : N;
    const R::AdaptiveResult res = R::adaptive_sweep(trial_options(unit, tol), trial_fit_options(tol, true),
                                                    [&](double f) {
                                                      R::Sample s = r.at(f / unit, false);
                                                      for (int k = 0; k < 2; ++k)
                                                        s.value(k) = perturb(s.value(k), noise_level, with_phase, g, n);
                                                      s.derivative(0) = perturb(s.derivative(0), noise_level,
                                                                               with_phase, g, n);
                                                      return std::optional<R::Sample>(s);
                                                    });
    if (!res.converged) ++not_converged;
    sum_solves += res.attempted.size();
    // AXF_TRIAL_DUMP=t: the final solves of trial t on stderr, as refitnoise reads them (a regression case)
    const char* dump_trial = std::getenv("AXF_TRIAL_DUMP");
    if (dump_trial && std::atoi(dump_trial) == t) {
      for (const R::Sample& x : res.samples) {
        std::fprintf(stderr, "%.17g 1 %.17g %.17g %.17g %.17g %.17g %.17g\n", x.f, x.value(0).real(), x.value(0).imag(),
                     x.value(1).real(), x.value(1).imag(), x.derivative(0).real(), x.derivative(0).imag());
      }
    }
    const auto classed =
        R::classify_poles(res.fit, res.samples, res.options, pole_options(unit, 2 * unit, tol, true, noise_level));
    poles.add(tally(classed, r.p, unit, res.converged));
    for (const auto& c : classed)
      if (Qtrue > 0 && c.lossy && std::fabs(c.p.real() / unit - f0) < 1e-6 * f0)
        q_error.push_back(std::fabs(c.Q / Qtrue - 1));
  }
  std::sort(q_error.begin(), q_error.end());
  const double q_median = q_error.empty() ? 0 : q_error[q_error.size() / 2];
  const double q_worst = q_error.empty() ? 0 : q_error.back();
  std::printf("{\"mode\": \"highq\", \"Q\": %.3g, \"noise\": %.0e, \"phase_noise\": %s, \"noise_streams\": %s, "
              "\"tolerance\": %.0e, \"trials\": %d, \"mean_solves\": %.2f, \"not_converged\": %d, \"Q_relative_error_median\": %.3g, "
              "\"Q_relative_error_worst\": %.3g, ",
              Qtrue, noise_level, with_phase ? "true" : "false", streams ? "true" : "false", tol, ntrials, sum_solves / ntrials, not_converged,
              q_median, q_worst);
  print_tally(poles);
  std::printf("}\n");
  return 0;
}

// a sweep's solves refitted under noise with the engine's options: how each in-band pole of the noise-free fit is
// classed
int refitnoise(int argc, char** argv) {
  std::ifstream in(argv[2]);
  const double fmin = std::atof(argv[3]);
  const double fmax = std::atof(argv[4]);
  const double tol = std::atof(argv[5]);
  const bool log_spacing = std::string(argv[6]) == "log";
  const int realizations = std::atoi(argv[7]);
  const double noise_level = std::atof(argv[8]);
  const bool with_phase = argc > 9 && std::string(argv[9]) == "phase";
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
  std::sort(samples.begin(), samples.end(), [](const R::Sample& a, const R::Sample& b) { return a.f < b.f; });
  const int nz = nports * nports;
  R::FitOptions fo;   // the engine's
  for (int k = 0; k < nz + nports; ++k) {
    fo.group.push_back(k < nz ? 0 : 1);
    fo.hermite.push_back(k < nz ? 1 : 0);
    const int j = k / nports;
    const int l = k % nports;
    if (k < nz) fo.floor_pair.push_back({j * nports + j, l * nports + l});
    else fo.floor_pair.push_back({-1, -1});
  }
  fo.fit_tolerance = std::max(1e-3 * tol, 1e-10);
  const R::PoleOptions po = pole_options(fmin, fmax, tol, log_spacing, noise_level);
  const auto base = R::classify_poles(R::fit(samples, fo), samples, fo, po);
  const size_t B = base.size();
  std::vector<int> n_np(B, 0);
  std::vector<int> n_res(B, 0);
  std::vector<int> n_lossy(B, 0);
  std::vector<int> n_unres(B, 0);
  std::vector<int> n_other(B, 0);
  std::vector<int> n_missing(B, 0);
  std::vector<std::vector<double>> q_values(B);
  std::vector<double> ratio_max(B, -1e300);   // Im p / sigma_Im over the realizations: the margin to the 3 sigma line
  std::vector<double> ratio_min(B, 1e300);
  std::vector<std::vector<double>> im_values(B);      // the fitted Im p over the realizations
  std::vector<std::vector<double>> sigma_values(B);   // the sigma_Im the classes used
  // per realization: z = (Im p - the noise-free fit's) / sigma_Im, and for a resolved Q (Q - the noise-free Q) /
  // Q_uncertainty; the RMS scatter against the median sigma is distorted by heavy tails
  std::vector<int> z_im_beyond(B, 0);
  std::vector<double> z_im_max(B, 0);
  std::vector<int> z_q_beyond(B, 0);
  std::vector<int> z_q_count(B, 0);
  std::vector<double> z_q_max(B, 0);
  std::mt19937 rng(17);
  std::normal_distribution<double> N(0, 1);
  for (int rz = 0; rz < realizations; ++rz) {
    std::vector<R::Sample> s = samples;
    for (auto& x : s) {
      for (int k = 0; k < x.value.size(); ++k) x.value(k) = perturb(x.value(k), noise_level, with_phase, rng, N);
      for (int k = 0; k < nz; ++k) x.derivative(k) = perturb(x.derivative(k), noise_level, with_phase, rng, N);
    }
    const auto cl = R::classify_poles(R::fit(s, fo), s, fo, po);
    for (size_t b = 0; b < B; ++b) {
      const R::PoleInfo* best = nullptr;
      for (const auto& c : cl)
        if (!best || std::abs(c.p - base[b].p) < std::abs(best->p - base[b].p)) best = &c;
      if (!best || std::fabs(best->p.real() - base[b].p.real()) > 1e-6 * base[b].p.real()) {
        ++n_missing[b];
        continue;
      }
      im_values[b].push_back(best->p.imag());
      sigma_values[b].push_back(best->sigma_im);
      if (best->sigma_im > 0 && std::isfinite(best->sigma_im)) {
        const double z = std::fabs(best->p.imag() - base[b].p.imag()) / best->sigma_im;
        z_im_max[b] = std::max(z_im_max[b], z);
        z_im_beyond[b] += z > 3 ? 1 : 0;
      }
      if (best->lossy && base[b].lossy && best->Q_sigma > 0 && std::isfinite(best->Q_sigma)) {
        const double z = std::fabs(best->Q - base[b].Q) / best->Q_sigma;
        ++z_q_count[b];
        z_q_max[b] = std::max(z_q_max[b], z);
        z_q_beyond[b] += z > 3 ? 1 : 0;
      }
      if (best->sigma_im > 0 && std::isfinite(best->sigma_im)) {
        ratio_max[b] = std::max(ratio_max[b], best->p.imag() / best->sigma_im);
        ratio_min[b] = std::min(ratio_min[b], best->p.imag() / best->sigma_im);
      }
      if (best->cls == "non_passive") ++n_np[b];
      else if (best->cls == "resonance") ++n_res[b];
      else if (best->cls == "unresolved") ++n_unres[b];
      else ++n_other[b];
      if (best->lossy) {
        ++n_lossy[b];
        q_values[b].push_back(best->Q);
      }
    }
  }
  std::printf("{\"mode\": \"refitnoise\", \"file\": \"%s\", \"noise\": %.1e, \"phase_noise\": %s, "
              "\"realizations\": %d, \"samples\": %zu, \"poles\": [",
              argv[2], noise_level, with_phase ? "true" : "false", realizations, samples.size());
  for (size_t b = 0; b < B; ++b) {
    std::sort(q_values[b].begin(), q_values[b].end());
    double q_spread = 0;
    if (!q_values[b].empty()) q_spread = (q_values[b].back() - q_values[b].front()) / q_values[b][q_values[b].size() / 2];
    // the actual RMS scatter of the fitted Im p about the noise-free fit's, against the median sigma_Im
    double scatter = 0;
    for (double v : im_values[b]) scatter += (v - base[b].p.imag()) * (v - base[b].p.imag());
    if (!im_values[b].empty()) scatter = std::sqrt(scatter / im_values[b].size());
    std::sort(sigma_values[b].begin(), sigma_values[b].end());
    const double sigma_median = sigma_values[b].empty() ? 0.0 : sigma_values[b][sigma_values[b].size() / 2];
    std::printf("%s{\"f_MHz\": %.6f, \"im_over_abs\": %.3e, \"noise_free_class\": \"%s\", \"noise_free_Q\": %s, "
                "\"noise_free_sigma_im_Hz\": %.3e, \"non_passive\": %d, \"resonance\": %d, \"unresolved\": %d, "
                "\"other\": %d, \"missing\": %d, \"given_Q\": %d, \"Q_median\": %.6g, \"Q_spread_relative\": %.3g, "
                "\"im_over_sigma_min\": %.3g, \"im_over_sigma_max\": %.3g, \"im_scatter_Hz\": %.3g, "
                "\"sigma_im_median_Hz\": %.3g, \"z_im_beyond_3\": %d, \"z_im_max\": %.3g, \"z_Q_resolved\": %d, "
                "\"z_Q_beyond_3\": %d, \"z_Q_max\": %.3g}",
                b ? ", " : "", base[b].p.real() / 1e6, base[b].p.imag() / std::abs(base[b].p), base[b].cls.c_str(),
                base[b].lossy ? std::to_string(base[b].Q).c_str() : "null", base[b].sigma_im, n_np[b], n_res[b],
                n_unres[b], n_other[b], n_missing[b], n_lossy[b],
                q_values[b].empty() ? 0.0 : q_values[b][q_values[b].size() / 2], q_spread, ratio_min[b], ratio_max[b],
                scatter, sigma_median, z_im_beyond[b], z_im_max[b], z_q_count[b], z_q_beyond[b], z_q_max[b]);
  }
  std::printf("]}\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  axf::log_level() = 0;
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "highq" && argc >= 5) return highq(argc, argv);
  if (mode == "refitnoise" && argc >= 9) return refitnoise(argc, argv);
  if ((mode == "lossless" || mode == "lossy" || mode == "mixed") && argc >= 6) return trials(argc, argv);
  std::fprintf(stderr,
               "usage: adaptive_sweep_trials KIND TOLERANCE NOISE UNIT TRIALS [HERMITE [BIAS]]\n"
               "       adaptive_sweep_trials highq Q NOISE TRIALS [TOLERANCE]\n"
               "       adaptive_sweep_trials refitnoise FILE F_MIN F_MAX TOLERANCE SPACING REALIZATIONS NOISE "
               "[phase]\n");
  return 2;
}
