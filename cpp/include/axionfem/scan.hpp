// Axion signal, readout noise and optimal-scan relations.
//
// Conventions follow Chaudhuri et al. (arXiv:1803.01627, 1904.05806); PSD normalization follows
// Foster, Rodd & Safdi (arXiv:1711.10489).  See docs/physics/scan_and_signal.md
// (equation tags "(S*)" below) and docs/physics/scan_cpp_api.md (API reference).
//
// signal.hpp (included) holds Secs. 1-4 of the spec: axion inputs, SHM line shape,
// Q_axion, circuit/noise PSDs, G-bar and xi_opt, bandwidths, Asimov statistics.
// This header holds Sec. 5-7: scan rate, F_req, time allocation, reach curves, the
// exact discrete (per-tuning) Asimov sum, the finite-band kernel/LP, long-detector factors.
//
// Interface units: SI (Hz, rad/s, s, K, H, Wb), masses in eV, couplings in GeV^-1,
// rho in GeV/cm^3; FEM geometry in natural HL (Psi [T m^3], L_p [m]); see signal.hpp.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "axionfem/signal.hpp"
#include "axionfem/spec.hpp"

namespace axf::scan {

// ================================================================== original API (kept)
// |g_agamma(m_a)| in GeV^-1 for the QCD axion line: (alpha / 2 pi f_a)|E/N - C_QCD|  (S2.1).
double g_agg_GeV(double m_a_eV, const spec::ScanSpec& s);

// Axion mass (eV) <-> frequency (Hz): f = m_a / (2 pi hbar).
double mass_to_freq(double m_a_eV);
double freq_to_mass(double f_Hz);

// Local dark-matter density in natural units (eV^4).
double rho_eV4(double rho_GeV_cm3);

// Required scan figure of merit F_req = Psi^4/L_p^2 for covering
// [f_min, f_max] along the QCD line in the live time
// (using G_bar ~ omega/(6 sqrt3 eta T), valid for n >> 1, n >> eta_A)  (S5.4).
struct Requirement {
  double F_nat_eV = 0;     // eV^-2
  double F_nat = 0;        // T^4 m^10 (natural Heaviside-Lorentz units)
  double F_SI = 0;         // T^4 H^2 m^8
  double g_min_GeV = 0;    // coupling at the lowest mass
};
Requirement F_required(const spec::Benchmark& b, const spec::ScanSpec& s);

// Exact optimal-coupling G_bar(omega0) for occupation n and eta_A  (S3.10).
double G_bar(double n_occ, double eta_A);
// Bose-Einstein occupation at frequency f_Hz (S3.1); same as n_bose(2 pi f, T).
double occupation(double f_Hz, double T_K);

// Convert SI <-> natural F.
double F_SI_to_nat(double F_SI);

// ================================================================== Sec. 5: scan strategy
// A function of the axion mass [eV] (target coupling in GeV^-1, or Q, T_K, eta_A).
using MassFunction = std::function<double(double m_a_eV)>;

// Target coupling along a QCD line (S2.1); default DFSZ-I with C_gamma = 1.77.
MassFunction qcd_line(double E_over_N = kEoverN_DFSZ_I, double C_gamma = kCgammaLatticeBrandt,
                      double ma_fa_GeV2 = kMaFa_GeV2);

// Conventions of the scan formulas, with their defaults (signal.hpp).
struct ScanParams {
  double Q_axion = kQaxionDefault;  // used by the LOCAL formulas (S5.1-S5.8); the exact
                                    // discrete sums integrate the line itself (v0, vobs)
  double Theta = kTheta95;          // Asimov TS of the 95% limit (S4.5)
  double rho_GeV_cm3 = kRhoDM_GeV_cm3;
  double v0_kms = kV0_kms;          // SHM line used by the exact sums (S5.9) and Q_axion_shm
  double vobs_kms = kVobs_kms;
  bool exact_Gbar = true;           // false: G-bar = hbar w0/(6 sqrt3 eta k_B T) (S3.11)
};

// G-bar_opt of the detector at tuning omega0: exact (S3.10) or classical limit (S3.11).
double G_bar_at(const Detector& d, double omega0_rad_s, bool exact = true);

// Local optimal scan rate d omega0/dt [rad/s^2]  (S5.2):
//   (pi^2/8)(Q_a/Theta) Phi_a^4/(hbar^2 L^2) Q G-bar,  Phi_a [Wb], L [H].
double scan_rate_SI(double Phi_a_Wb, double L_H, double Q, double Gbar, double Q_axion = kQaxionDefault,
                    double Theta = kTheta95);
// Same for a detector tuned to omega0 = omega_a at coupling g  (S5.1).  [rad/s^2]
double scan_rate(const Detector& d, double m_a_eV, double g_GeV, const ScanParams& p = {});

// Closed-form requirement  (S5.4), SI form of spec Sec. 5.2:
//   F_req = (12 sqrt3/pi^2) Theta eta k_B T hbar / (Q_a Q kappa_SI(g1)^4 t) [1 - (m1/m2)^4]
// (n >> 1, n >> eta, g proportional to m, constant Q, T, eta).  g_min_GeV = g1.
Requirement F_req_closed_form(double eta_A, double T_K, double Q, double g1_GeV, double t_s, double m1_eV,
                              double m2_eV, const ScanParams& p = {});
// General requirement  (S5.3):  F_req = 8 Theta hbar^2/(pi^2 Q_a t) int_{w1}^{w2} dw / (Q kappa_SI(g_t)^4 G-bar),
// with exact G-bar (unless p.exact_Gbar = false), arbitrary target g_t(m) and Q, T, eta(m).
Requirement F_req_general(double m1_eV, double m2_eV, double t_s, const MassFunction& g_target_GeV,
                          const MassFunction& Q_of_m, const MassFunction& T_K_of_m,
                          const MassFunction& eta_A_of_m, const ScanParams& p = {});
Requirement F_req_general(double m1_eV, double m2_eV, double t_s, double Q, double T_K, double eta_A,
                          const MassFunction& g_target_GeV = qcd_line(), const ScanParams& p = {});

// ------------------------------------------------------------------ time allocation (S5.5-S5.7)
// Dwell density tau(omega0) = time per unit omega0 [s per rad/s].
struct Allocation {
  enum class Kind { Target, LogUniform };
  Kind kind = Kind::LogUniform;
  double m1_eV = 0, m2_eV = 0;  // scan band
  double t_total_s = 0;         // live time the allocation is normalised to
  double t_req_s = 0;           // Target: local-approx. time to reach g_target exactly (S5.6);
                                // LogUniform: equals t_total_s
  // tau(omega0) [s/(rad/s)].  The formula is NOT truncated at the band edges: the
  // schedule is tau(omega0) for omega(m1) <= omega0 <= omega(m2) and zero outside.
  std::function<double(double omega0_rad_s)> tau;
};
// Target-curve allocation (minimum total time to reach g_target everywhere), rescaled to
// t_total:  tau = Theta / [(pi^2/8) Q_a Q G-bar kappa_a(g_t)^4 F] * t_total/t_req  (S5.5-S5.6).
Allocation allocation_target(const Detector& d, double m1_eV, double m2_eV, double t_total_s,
                             const MassFunction& g_target_GeV = qcd_line(), const ScanParams& p = {});
// Log-uniform allocation  tau = t/(omega0 ln(omega2/omega1))  (S5.7; [C] Eq. 245).
Allocation allocation_log_uniform(double m1_eV, double m2_eV, double t_total_s);
double tau_log_uniform(double omega0_rad_s, double t_total_s, double omega1_rad_s, double omega2_rad_s);
// t_req = int dw0 / (d omega0/dt at g_target)  [s]  (S5.6), local approximation.
double time_target_curve_s(const Detector& d, double m1_eV, double m2_eV,
                           const MassFunction& g_target_GeV = qcd_line(), const ScanParams& p = {});

// ------------------------------------------------------------------ reach (S5.8)
// Local reach: g95^4 = 8 Theta hbar^2 / (pi^2 tau Q_a Q G-bar kappa_SI(1 GeV^-1)^4 F_SI)
// (= (S5.8) in SI) at omega0 = omega_a.  [GeV^-1]
double g95_local(const Detector& d, double m_a_eV, double tau_s_per_rad_s, const ScanParams& p = {});

// What a reach/scan call actually covered, echoed in its results so that an output document is
// self-describing (band, number of settings, allocation, live time, benchmark parameters).
// Every field is filled from the inputs of the call that produced the curve; zero/empty means
// "not applicable to this kind of call" (e.g. Q, T, eta_A for a dispersive detector curve).
struct ScanBand {
  double m1_eV = 0, m2_eV = 0;        // mass range of the schedule
  double f_min_Hz = 0, f_max_Hz = 0;  // the same range in frequency
  double live_time_s = 0;             // live time the allocation is normalised to (Sum of dwells)
  double t_req_s = 0;                 // Target allocation: time to reach g_target exactly (S5.6)
  std::string allocation;             // "target" | "log_uniform" | "discrete" | "lp" | "custom"
  int n_settings = 0;                 // tunings/settings in the schedule (0 = local continuum)
  int n_targets = 0;                  // masses the reach was evaluated at
  std::string stat;                   // "weak" | "limit" | "discovery" (discrete/exact sums)
  // benchmark/detector inputs (a dispersive curve reports the values at m1)
  double Psi_SI_T_H_m2 = 0, L_H = 0, Q = 0, T_K = 0, eta_A = 0, F_SI_T4_H2_m8 = 0;
  // scan conventions actually used (ScanParams)
  double Q_axion = 0, Theta = 0, rho_GeV_cm3 = 0, v0_kms = 0, vobs_kms = 0;
  bool exact_Gbar = true;
};

struct ReachCurve {
  std::vector<double> m_eV, f_Hz, g95_GeV;
  std::vector<double> tau_s_per_rad_s;  // local: dwell density used; discrete: empty
  std::vector<double> s_max;            // discrete only: max per-bin s at g95 (weak-signal check)
  ScanBand band;                        // what the call covered (self-describing output)
};
ReachCurve reach_local(const Detector& d, const Allocation& a, const std::vector<double>& m_eV,
                       const ScanParams& p = {});
// n_points log-spaced masses over [a.m1_eV, a.m2_eV] (endpoints included).
ReachCurve reach_local(const Detector& d, const Allocation& a, int n_points, const ScanParams& p = {});
// One call: detector + band + live time -> allocation (built with p) -> local reach.
ReachCurve reach_local(const Detector& d, double m1_eV, double m2_eV, double live_time_s, Allocation::Kind kind,
                       int n_points = 200, const MassFunction& g_target_GeV = qcd_line(),
                       const ScanParams& p = {});

// ------------------------------------------------------------------ discrete exact sum (S5.9-S5.10)
// A list of tunings: centre omega0 [rad/s], dwell t [s], and xi (empty -> xi_opt(n(omega0)), S3.9).
struct Tunings {
  std::vector<double> omega0_rad_s, t_s, xi;
};
// Asimov TS of each tuning for an axion (m, g)  (S4.1, S5.9):
//   Theta_i = (t_i/2pi) int_0^{vmax} omega_a v dv h(s_i(omega_a(1+v^2/2))),
// with the full |Z|^2, exact n(omega), omega^2 backaction, R(omega) (S3.6), exact SHM line,
// Gauss-Legendre with n_v nodes on [0, vobs + 7 v0], omega - omega0 = (omega_a - omega0) + omega_a v^2/2.
// s_max (optional) receives the largest per-bin s over all tunings and nodes.
std::vector<double> theta_per_tuning(const Detector& d, double m_a_eV, double g_GeV, const Tunings& tun,
                                     Stat stat = Stat::Weak, const ScanParams& p = {}, int n_v = 256,
                                     double* s_max = nullptr);
double theta_discrete(const Detector& d, double m_a_eV, double g_GeV, const Tunings& tun,
                      Stat stat = Stat::Weak, const ScanParams& p = {}, int n_v = 256);
// g95 solving Theta(m, g95) = p.Theta  (S5.10): weak -> closed form (Theta ~ g^4); limit/discovery
// -> Brent root in ln g (Theta monotone in g).  s_max_at_g95 (optional) as above at g95.
double g95_discrete(const Detector& d, double m_a_eV, const Tunings& tun, Stat stat = Stat::Limit,
                    const ScanParams& p = {}, int n_v = 256, double* s_max_at_g95 = nullptr);
// Discrete reach over masses; only tunings within +-window_half_widths noise-doubling
// half-widths of omega_a are summed (spec Sec. 10: +-40 -> truncation < 1e-5).
// Logs a warning (axf::log level 1) if any s_max > 0.05 (weak-signal limit invalid, Sec. 5.5).
ReachCurve reach_discrete(const Detector& d, const Tunings& tun, const std::vector<double>& m_eV,
                          Stat stat = Stat::Limit, const ScanParams& p = {}, double window_half_widths = 40.0,
                          int n_v = 256);

// Tunings at omega_a + hw * linspace(-H, H, 2 H S + 1), hw = half the noise-doubling width at
// omega_a, dwell t = a.tau(omega0) hw/S (H = half_widths, S = steps_per_hw).  (spec Sec. 8 check)
Tunings dense_tunings_around(const Detector& d, double m_a_eV, const Allocation& a, double half_widths = 40.0,
                             double steps_per_hw = 6.0);
// Discretise an allocation over its band: steps delta = Dw_2x(omega0)/steps_per_fw, tunings at the
// step mid-points, dwell tau(omega0) delta  (spec Sec. 5.5).
Tunings discretise_allocation(const Detector& d, const Allocation& a, double steps_per_fw = 10.0);
// Number of tunings N = int dw0 steps_per_fw / Dw_2x(w0) over [m1, m2]  (spec Sec. 5.5).
double n_tunings(const Detector& d, double m1_eV, double m2_eV, double steps_per_fw = 10.0);

// ================================================================== Sec. 6: finite band
// Kernel K_j(m) = Theta per unit dwell of setting j (omega0_j, xi_j) at coupling g [1/s]  (S6.1-S6.2).
std::vector<double> kernel_row(const Detector& d, double m_a_eV, double g_GeV,
                               const std::vector<double>& omega0_rad_s, const std::vector<double>& xi,
                               Stat stat = Stat::Weak, const ScanParams& p = {}, int n_v = 96);
// Narrow-line kernel (pi/8) Q_a omega_a^3 Phi_a^4 / N_V(omega_a)^2  [1/s]  (S6.3).
double kernel_narrow_line(const Detector& d, double m_a_eV, double g_GeV, double omega0_rad_s, double xi,
                          const ScanParams& p = {});
// Minimum-total-time schedule  min sum_j t_j  s.t.  sum_j K[k][j] t_j >= Theta, t >= 0  (S6.4).
// Dense simplex on the dual LP (max Theta sum y, K^T y <= 1, y >= 0; feasible at y = 0),
// with K rescaled to O(1) per target; t_j are the dual multipliers.  Afterwards t is scaled
// up (if needed) so that min_k (K t)_k >= Theta exactly.
struct LPSchedule {
  std::vector<double> t_s;
  double t_total_s = 0;
  double min_coverage = 0;  // min_k (K t)_k / Theta  (>= 1)
  int pivots = 0;
};
LPSchedule lp_schedule(const std::vector<std::vector<double>>& K, double Theta = kTheta95);

// ------------------------------------------------ Sec. 6 with the EXACT network noise
// The kernels above get N_V(omega) from the single-pole Detector (S3.6).  A real readout
// network is reduced to that pole only to second order in the detuning (readout_circuit.md
// limitation 5); across a wide band (low-Q normal-metal pickup, strongly dispersive Z_p near
// a self-resonance) the exact N_V(omega) of (C4.5) must be used instead.  The functions below
// take it as data: everything the kernel needs from the readout is N_V(omega) [V^2 s] per
// setting plus the pickup's Psi_SI, since (S6.1) is K = (1/2pi) int domega h(S_Va/N_V) and
// (S3.2) S_Va depends only on Phi_a = kappa_a(g, rho) Psi_SI and the line shape.
//
// A tabulated N_V(omega) for ONE setting (one tuning capacitance and one coupling), as
// produced by `axionfem.circuit.Readout.noise_curve`.  Interpolation is linear in
// (log omega, log N_V): every term of (C4.5) is a power law in omega or in the detuning
// (|Z|^2 ~ R^2 + (2 L dw)^2), so log-log is exact in each limit and smooth through the
// resonance.  Outside the table the end slope is continued (the kernel is negligible far off
// resonance, but no evaluation is refused).
struct NoiseCurve {
  std::vector<double> omega_rad_s;  // strictly ascending, all > 0
  std::vector<double> N_V_V2_s;     // all > 0
  double at(double omega_rad_s) const;
  void validate(const char* where) const;  // throws axf::Error with a specific message
};
using NoiseFunction = std::function<double(double omega_rad_s)>;

// Exact finite-band kernel of each setting from its own N_V(omega)  (S6.1-S6.2), [1/s]:
//   K_j(m) = (1/2pi) int domega h(s_j), s_j = S_{V,a}(omega; Phi_a)/N_{V,j}(omega),
// integrated in v exactly as (S5.9) does.  psi_SI = mu0 Psi [T H m^2] of the PICKUP (the
// network's L_eff never enters: it is already inside N_V).  Pass Psi(omega_a) per target to
// get a frequency-dependent coupling (limitation 7).
std::vector<double> kernel_row_noise(double psi_SI_T_H_m2, double m_a_eV, double g_GeV,
                                     const std::vector<NoiseCurve>& noise, Stat stat = Stat::Weak,
                                     const ScanParams& p = {}, int n_v = 96);
std::vector<double> kernel_row_noise(double psi_SI_T_H_m2, double m_a_eV, double g_GeV,
                                     const std::vector<NoiseFunction>& noise, Stat stat = Stat::Weak,
                                     const ScanParams& p = {}, int n_v = 96);
// Narrow-line kernel (S6.3) from the exact N_V at the axion frequency:
//   K = (pi/8) Q_a omega_a^3 Phi_a^4 / N_V(omega_a)^2   [1/s].
double kernel_narrow_line_noise(double psi_SI_T_H_m2, double m_a_eV, double g_GeV, double N_V_at_omega_a_V2_s,
                                const ScanParams& p = {});

// g95 of a finite-band schedule with the exact noise: solve sum_j t_j K_j(m, g) = p.Theta
// (S6.4 evaluated at one target).  Weak: closed form (K ~ g^4); Limit/Discovery: Brent in ln g.
// s_max_at_g95 (optional) receives the largest per-bin s at g95 (weak-signal check, Sec. 5.5).
double g95_band(double psi_SI_T_H_m2, double m_a_eV, const std::vector<NoiseCurve>& noise,
                const std::vector<double>& t_s, Stat stat = Stat::Limit, const ScanParams& p = {},
                int n_v = 96, double* s_max_at_g95 = nullptr);
// Reach of one exact-noise schedule over a list of targets.  psi_SI may hold one value per
// target (frequency-dependent coupling) or a single value for all of them.  The returned
// band echoes the schedule (allocation "lp", n_settings, live time = sum t_j).
ReachCurve reach_band(const std::vector<double>& psi_SI_T_H_m2, const std::vector<double>& m_eV,
                      const std::vector<NoiseCurve>& noise, const std::vector<double>& t_s,
                      Stat stat = Stat::Limit, const ScanParams& p = {}, int n_v = 96);

// ================================================================== Sec. 7: long detectors
// Psi^2/L_p = g W_mag R^2 (g the dimensionless geometry factor), so W_mag R^2 = sqrt(F_req)/g.
double g_solenoid_long(double x);  // (1-x^2)^2 / (4 ln(1/x))   (long coaxial solenoid)
double g_toroid_long(double y);    // y^2 ln(1/y)  (= g_CORE)   (long toroid)
struct LongOptimum {
  double x = 0, g = 0;
};
LongOptimum solenoid_long_optimum();  // root of 1 - x^2 + 4 x^2 ln x = 0: x = 0.533543, g = 0.2036322
LongOptimum toroid_long_optimum();    // y = e^{-1/2}, g = 1/(2e)

}  // namespace axf::scan
