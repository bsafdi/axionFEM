// Barycentric rational fits of vector-valued frequency responses, and the adaptive frequency sweep built on them
// (docs/physics/adaptive_sweep.md).
//
// The model is AAA's barycentric form (Nakatsukasa, Sete and Trefethen 2018) with one denominator shared by every
// output (set-valued AAA):
//
//   r_k(f) = sum_j w_j F_jk / (f - f_j)  /  sum_j w_j / (f - f_j),      type (m-1, m-1),
//
// which interpolates the values F_jk at the m support frequencies f_j. The weights w minimize a linearized residual
// over all samples (the smallest right singular vector of a stacked Loewner matrix, ||w|| = 1). Outputs with
// derivative data (Hermite) add the linearized conditions r'(f_i) = F'_ik at every sample, so each solve counts
// twice for them. Rows are weighted relative to each value, max(|F_ik|, floor_k), the same pointwise measure that
// the error estimate and the stopping rule use; the floor is a fraction of the output group's typical size.
//
// The adaptive sweep fits, estimates the error on a dense candidate grid (with points around the model's poles) as the
// larger difference between the fit and the fit with one support point fewer, or the fit before the newest solve,
// solves where that is largest, and stops when the estimate is below the tolerance over the band and each of the two
// newest solves agreed with the model fitted before it.
#pragma once

#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace axf::rational {

using cplx = std::complex<double>;

// One solved frequency: K output values and, for outputs with derivative data, dF/df (NaN where there is none).
struct Sample {
  double f = 0;
  Eigen::VectorXcd value;
  Eigen::VectorXcd derivative;
};

// r(f) = sum_j w_j F_j / (f - f_j) / sum_j w_j / (f - f_j); interpolates F_j at the support frequencies f_j.
struct Model {
  std::vector<double> support;
  Eigen::VectorXcd weights;
  Eigen::MatrixXcd values;   // support points x outputs

  int size() const { return static_cast<int>(support.size()); }
  Eigen::VectorXcd value(double f) const;
  Eigen::VectorXcd derivative(double f) const;
  // poles in the complex f plane: the zeros of sum_j w_j / (f - f_j) (at most m - 1)
  std::vector<cplx> poles() const;
  // residues of every output at the given poles: N_k(p) / D'(p) (poles x outputs)
  Eigen::MatrixXcd residues(const std::vector<cplx>& poles) const;
};

struct FitOptions {
  std::vector<int> group;            // per output: its scale group (the floor of the error measure)
  std::vector<char> hermite;         // per output: derivative data are used where finite
  double fit_tolerance = 1e-8;       // the greedy stops once every sample is fitted to this (weighted)
  double floor_fraction = 1e-2;      // floor_k = floor_fraction x median over samples of the group's largest |F|
  double doublet_tolerance = 1e-13;  // Froissart doublet: every residue below this x scale x band width
  double noise_cap = 1e-6;           // a misfit counts as data noise up to this (the solves' true-residual gate)
  // per output (optional): a pair (a, b) of outputs whose median sizes set its floor, floor_fraction x
  // sqrt(median |F_a| median |F_b|) (Z_jk against Z_jj and Z_kk); (-1, -1) or empty: the group's floor
  std::vector<std::pair<int, int>> floor_pair;
};

struct Fit {
  Model model;
  // the same samples with one support point fewer: the error estimate's comparison (with one support point: the
  // samples' mean, AAA's starting approximation; empty only with a single sample)
  Model lower;
  Eigen::VectorXd floor;       // per output, for the error measure
  Eigen::VectorXd scale;       // per output: max(median |F_k|, floor_k), for residues
  double sample_error = 0;     // the largest weighted misfit at the samples (values and derivatives)
  int doublets_removed = 0;    // spurious poles removed by the cleanup (model and lower together)
  std::vector<int> support_order;   // sample indices of the support, in the order the greedy chose them
};

// max_k |a_k - ref_k| / max(|ref_k|, floor_k): the pointwise relative error measure of the fit
double relative_error(const Eigen::VectorXcd& a, const Eigen::VectorXcd& ref, const Eigen::VectorXd& floor);

// Set-valued AAA with Hermite data on samples sorted by frequency with distinct frequencies (at least one).
Fit fit(const std::vector<Sample>& samples, const FitOptions& options);

// The error estimate at the frequencies f: relative_error(lower(f), model(f), floor).
// With `previous` (the model fitted before the newest solve), the larger of that and the change from previous: the
// sweep's own estimate.
Eigen::VectorXd estimated_error(const Fit& fit, const std::vector<double>& f, const Model* previous = nullptr);

// The closure's gate (adaptive_sweep, stop "converged: ... spurious poles of the comparison model"). The comparison
// model's spurious poles: above the real axis by more than 1e-9 |q| (a passive response has none) and by at most
// kSpuriousReach x band_width, with no pole of the model within Im q. Farther away a pole's term is smooth over the
// band and could stand for a genuine smooth error of the model.
constexpr double kSpuriousReach = 0.25;
std::vector<cplx> spurious_comparison_poles(const Fit& fit, double band_width);
// Those poles' own terms in r-, sum_q |res_q / (f - q)| (r-'s residues), in the fit's measure (relative to
// max(|r(f)|, floor), the largest over the outputs). The gate excuses a candidate only where they reach half its
// estimate.
Eigen::VectorXd spurious_terms(const Fit& fit, const std::vector<cplx>& spurious, const std::vector<double>& f);

struct AdaptiveOptions {
  double f_min = 0;
  double f_max = 0;
  double tolerance = 1e-3;
  int max_solves = 30;              // every attempted frequency, the initial ones and failed ones included
  std::vector<double> initial;      // the first frequencies, solved in this order
  bool log_spacing = true;          // of the candidate grid
  int candidates = 2000;            // candidate grid points over the band (pole neighbourhoods are added)
  // the estimate also takes how far the newest solve moved the model (the fit before it against the fit after it)
  bool compare_previous = true;
  // consecutive newest solves that must agree with their predictions: on random resonant responses, 1 left 1-2.5 % of
  // the sweeps up to 13x the tolerance off somewhere in the band, 2 at most 1 % and 1.6x, for about one more solve
  int confirmations = 2;
  // called with {stop flag, next frequency} before every decision: an MPI caller makes every rank take rank 0's
  std::function<void(std::vector<double>&)> agree;
};

struct AdaptiveStep {
  int solves = 0;                   // frequencies attempted before this fit
  int samples = 0;                  // successful solves in the fit
  int support_points = 0;
  double sample_fit_error = 0;
  int doublets_removed = 0;
  double estimated_error = 0;       // the largest estimate over the candidates
  double estimated_error_at = 0;    // where
  double next_frequency = 0;        // the largest estimate where a solve is allowed (next unless the sweep stops)
  double model_change = 0;          // its part from the newest solve: the largest change of the model it caused
  double prediction_error = 0;      // the newest solve against the model fitted before it (+inf: none yet)
  double predicted_at = 0;          // that solve's frequency (0: none)
};

struct AdaptiveResult {
  std::vector<Sample> samples;      // the successful solves, sorted by frequency
  std::vector<double> attempted;    // every frequency in solve order
  std::vector<double> failed;       // the attempted frequencies without a usable solve
  Fit fit;
  bool converged = false;
  std::string stop_reason;
  double estimated_error = 0;
  double last_prediction_error = 0;
  std::vector<AdaptiveStep> history;
  Model previous;                   // the model fitted before the newest solve (empty: none)
  bool hermite = false;             // the final fit used derivative rows
  std::string hermite_note;         // why they were dropped, if they were
  bool fit_consistent = false;      // the final fit reproduces its solves to 10 x the fit tolerance (before removals)
  FitOptions options;               // the options of the final fit (Hermite flags cleared if they were dropped)
  // After a converged sweep, Froissart doublets the cleanup kept are removed (remove_invisible_poles).
  // fit.sample_error is then the misfit after the removal.
  int invisible_poles_removed = 0;
  double sample_error_before_removal = 0;
};

// A removal of an invisible pole after convergence may cost the model this fraction of the tolerance at the solves and
// on the candidate grid, and move another pole on the real axis by this fraction of the tolerance times its quarter
// grid step (and at most tol x 1e-6 |p|)
constexpr double kInvisibleFraction = 0.1;
// ... and on the grid up to this fraction where the sweep's estimate leaves that much slack (tol - estimate). Trial
// 88's removal changes the model by up to 0.116 x the tolerance in the low band, where the estimate is 0.10-0.12 x
constexpr double kInvisibleGridFraction = 0.25;

// A sweep is noise-bound when its fit tolerance is below this many times the solves' relative accuracy: the fit then
// follows the noise, and its stop (converged or not) is unreliable. In the trials under phase noise at the fit
// tolerance 7-10 % of such sweeps stopped wrong.
constexpr double kNoiseBoundFactor = 10.0;
inline bool noise_bound(double fit_tolerance, double noise) { return fit_tolerance < kNoiseBoundFactor * noise; }

// After a converged sweep (adaptive_sweep calls it): removes Froissart doublets, i.e. in-band poles on the real axis
// (|Im p| below a quarter grid step of `options`) that are invisible one candidate step away, whose needle reaches
// the tolerance, and which the refit without their nearest support point (at fixed degree) loses; the checks are in
// rational_fit.cpp. `previous` as for estimated_error (the grid allowance uses the sweep's estimate). Updates
// fit.model, fit.support_order and fit.sample_error; returns how many were removed.
int remove_invisible_poles(Fit& fit, const std::vector<Sample>& samples, const FitOptions& fit_options,
                           const AdaptiveOptions& options, const Model* previous);

// solve(f) returns the sample at f, or nothing when that frequency could not be solved (it is recorded as failed and
// excluded from the fit and from later candidates).
AdaptiveResult adaptive_sweep(const AdaptiveOptions& options, const FitOptions& fit_options,
                              const std::function<std::optional<Sample>(double)>& solve);

// The candidate grid of [f_min, f_max]: n points, log or linear.
std::vector<double> band_grid(double f_min, double f_max, int n, bool log_spacing);

// Classing the model's poles in [f_min, f_max] (docs/physics/adaptive_sweep.md, "Pole classes").
struct PoleOptions {
  double f_min = 0;
  double f_max = 0;
  double tolerance = 1e-3;
  int candidates = 2000;            // the sweep's candidate grid (its step is the resolution of `visible`)
  bool log_spacing = true;
  double real_axis = 1e-12;         // |Im p| / |p| below this is rounding: on the real axis
  double significance = 3.0;        // Im p counts (as loss, or as growth) only beyond this many standard errors
  double min_Q = 2.0;               // broader poles are the band's background
  double noise = 1e-10;             // the solves' relative accuracy: refits under noise of this size (below)
  int noise_refits = 8;
};

struct PoleInfo {
  cplx p;
  double strength = 0;              // the largest residue over the outputs, relative to the output's scale [Hz]
  double visible = 0;               // strength / max(|Im p|, one candidate step): its size at its half-width
  int refits = 0;                   // leave-one-out refits (one per solve)
  int reproduced = 0;               // of them, those with a matching pole
  // standard errors of Re p and Im p [Hz], in quadrature: the jackknife over the leave-one-out refits, the offset of
  // their mean from the model's pole, and the RMS deviation over refits under the solves' noise
  double sigma_re = 0;
  double sigma_im = 0;
  bool lossy = false;               // -Im p > max(real_axis |p|, significance sigma_im): Q is resolved
  bool non_passive = false;         // Im p > max(real_axis |p|, significance sigma_im): grows in time
  double Q = 0;                     // Re p / (2 |Im p|) (infinite on the real axis)
  double Q_sigma = 0;               // its standard error (lossy poles)
  // a 3 sigma lower bound when the loss is not resolved (passive, not lossy): Re p / (2 max(real_axis |p|,
  // |Im p| + significance sigma_Im))
  double Q_min = 0;
  std::string cls;                  // unresolved, non_passive, negligible, background or resonance
};

// The leave-one-out refits of `samples` (one per solve, with `options`, when there are at least 4): for each pole
// poles[targets[b]], the refits' poles that match it (the nearest within half the distance to its nearest neighbour
// among `poles`). `refits` receives the number of refits (classify_poles).
std::vector<std::vector<cplx>> leave_one_out_matches(const std::vector<cplx>& poles, const std::vector<size_t>& targets,
                                                     const std::vector<Sample>& samples, const FitOptions& options,
                                                     int& refits);

// Every in-band pole of fit.model with its uncertainty and class. The uncertainty combines the jackknife over the
// leave-one-out refits of `samples` with `options` (the final fit's), the offset of their mean from the pole, and the
// spread over `noise_refits` refits of the samples perturbed by relative Gaussian noise of size `noise` (a fixed seed;
// with a random phase, except on outputs whose every sample is real or every sample imaginary); each pole is matched
// to the refit's nearest pole within half the distance to its own nearest neighbour. Classes, in order:
//  - unresolved: reproduced by fewer than half the refits, or its position uncertain by more than max(|Im p|, step);
//  - non_passive: Im p significantly above the real axis;
//  - negligible: visible below the tolerance;
//  - background: Q < min_Q (from Im p as fitted);
//  - resonance: otherwise.
std::vector<PoleInfo> classify_poles(const Fit& fit, const std::vector<Sample>& samples, const FitOptions& options,
                                     const PoleOptions& pole_options);

}  // namespace axf::rational
