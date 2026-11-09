// Axion inputs, halo line shape, readout-circuit signal/noise PSDs and Asimov
// statistics: docs/physics/scan_and_signal.md Secs. 1-4 (equation tags "(S*)"
// below refer to that spec).  The scan machinery built on top of this lives in
// scan.hpp (which includes this header).
//
// Interface units (SI unless the name says otherwise):
//   angular frequencies omega [rad/s], frequencies f [Hz], times [s],
//   temperatures [K], axion masses m_a [eV], couplings g [GeV^-1],
//   DM density rho [GeV/cm^3], halo speeds v0, vobs [km/s].
//   FEM-facing geometry in natural Heaviside-Lorentz units:
//   Psi [T m^3] and L_p [m]; SI equivalents Psi_SI = mu0 Psi [T H m^2],
//   L = mu0 L_p [H]  (S1.3).
//   PSDs use the Foster et al. (1711.10489) normalisation
//   <X^2> = (1/pi) int_0^inf domega S_X(omega)  (S3.2-normalisation, Sec. 3.2):
//   voltage PSDs in V^2 s, current PSDs in A^2 s.
#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace axf::scan {

// ------------------------------------------------------------------ defaults
inline constexpr double kQaxionDefault = 6.8e5;    // rounded Q_axion (exact SHM: Q_axion_shm() = 6.7795e5)
inline constexpr double kTheta95 = 2.71;           // [F] Eq. 54 (exact (Phi^-1(0.95))^2 = 2.7055)
inline constexpr double kRhoDM_GeV_cm3 = 0.3;      // local DM density
inline constexpr double kV0_kms = 220.0;           // SHM circular speed
inline constexpr double kVobs_kms = 232.0;         // lab speed in the galactic frame
inline constexpr double kMaFa_GeV2 = 5.70e-3;      // m_a f_a (1511.02867)
inline constexpr double kCgammaLatticeBrandt = 1.77;  // default C_gamma (lattice)
inline constexpr double kCgammaChiPT = 1.92;
inline constexpr double kCgammaLatticeLin = 1.965;
inline constexpr double kEoverN_KSVZ = 0.0;
inline constexpr double kEoverN_DFSZ_I = 8.0 / 3.0;
inline constexpr double kEoverN_DFSZ_II = 2.0 / 3.0;
inline constexpr double kAlphaLEE = 0.75;          // [F] Eq. 62

// ------------------------------------------------------------------ Sec. 1-2: axion inputs
// omega = m_a c^2 / hbar [rad/s] and its inverse [eV].
double omega_from_mass(double m_a_eV);
double mass_from_omega(double omega_rad_s);

// |g_agg| on a QCD-axion line [GeV^-1]  (S2.1):
//   g = alpha/(2 pi f_a) |E/N - C_gamma|,  f_a = (m_a f_a)/m_a.
double g_qcd_GeV(double m_a_eV, double E_over_N = kEoverN_DFSZ_I,
                 double C_gamma = kCgammaLatticeBrandt, double ma_fa_GeV2 = kMaFa_GeV2);

// kappa_a = g sqrt(2 rho_DM), so that the axion flux AMPLITUDE is Phi_a = kappa_a Psi  (S2.2-S2.3).
double kappa_a_eV(double g_GeV, double rho_GeV_cm3 = kRhoDM_GeV_cm3);     // natural HL [eV]
double kappa_a_inv_m(double g_GeV, double rho_GeV_cm3 = kRhoDM_GeV_cm3);  // [1/m]: Phi[Wb] = kappa[1/m] Psi[T m^3]
double kappa_a_SI(double g_GeV, double rho_GeV_cm3 = kRhoDM_GeV_cm3);     // [1/H]: Phi[Wb] = kappa_SI Psi_SI
// Flux amplitude Phi_a [Wb] for a natural-unit Psi [T m^3].
double flux_amplitude_Wb(double g_GeV, double psi_nat_T_m3, double rho_GeV_cm3 = kRhoDM_GeV_cm3);

// ------------------------------------------------------------------ Sec. 2.2-2.3: halo
// SHM lab-frame speed distribution f(v), v in units of c, int_0^inf f dv = 1, no
// escape-speed cut  (S2.4; [F] Eq. 9).  Evaluated as
// f = v/(sqrt(pi) v0 vobs) e^{-(v-vobs)^2/v0^2} (1 - e^{-4 v vobs/v0^2}) (no cancellation).
double f_shm(double v_over_c, double v0_kms = kV0_kms, double vobs_kms = kVobs_kms);

// Normalised line shape in angular frequency  (S2.5):
//   L(omega) = f(v)/(omega_a v), v = sqrt(2 (omega - omega_a)/omega_a), int L domega = 1  [s].
// line_shape_detuned takes domega = omega - omega_a directly (use it near the sqrt edge
// at omega = omega_a to avoid cancellation).  Zero for omega <= omega_a.
double line_shape(double omega_rad_s, double omega_a_rad_s, double v0_kms = kV0_kms,
                  double vobs_kms = kVobs_kms);
double line_shape_detuned(double domega_rad_s, double omega_a_rad_s, double v0_kms = kV0_kms,
                          double vobs_kms = kVobs_kms);

// Q_axion = int_0^inf dv f(v)^2 / v = omega_a int L^2 domega  (S2.6), by
// adaptive Gauss-Kronrod quadrature in v (the integrand is smooth in v; rel. tol 1e-13).
// Q_axion_shm(220, 232) = 6.779483e5 (rounded to 6.8e5 for the default kQaxionDefault).
double Q_axion_shm(double v0_kms = kV0_kms, double vobs_kms = kVobs_kms);

// Order-of-magnitude coherence time 1/(f v0 vobs) [s] ([F] Sec. III B; spec Sec. 2.3).
double coherence_time_s(double f_Hz, double v0_kms = kV0_kms, double vobs_kms = kVobs_kms);

// Gauss-Legendre nodes/weights in v (units of c) on [0, vobs + 7 v0] (spec S5.9 / Sec. 10).
void line_v_grid(int n_v, double v0_kms, double vobs_kms, std::vector<double>& v,
                 std::vector<double>& w);

// ------------------------------------------------------------------ Sec. 3: circuit and noise
// Exact Bose-Einstein occupation n = 1/expm1(hbar omega / k_B T)  (S3.1); 0 for T <= 0.
double n_bose(double omega_rad_s, double T_K);

// On-resonance noise number A = xi^2 + 2(2n+1) xi + eta_A^2  (S3.7).
double A_noise(double xi, double n, double eta_A);
// G-bar(xi) = xi / A^{3/2}  (S3.8; [C] F50, F53).
double G_bar_xi(double xi, double n, double eta_A);
// xi_opt = 2 eta^2 / (2n+1 + sqrt((2n+1)^2 + 8 eta^2))  (S3.9; [C] F54).
// (G_bar(n, eta) = G_bar_xi(xi_opt, n, eta), (S3.10), is declared in scan.hpp.)
double xi_opt(double n, double eta_A);
// High-occupation limits of G-bar_opt  (S3.11):
//   G_bar_highT(n, eta)          = 1 / (3 sqrt3 eta (2n+1))           (n >> 1, n >> eta)
//   G_bar_classical(w0, T, eta)  = hbar w0 / (6 sqrt3 eta k_B T)       (additionally 2n+1 -> 2 k_B T/hbar w0)
double G_bar_highT(double n, double eta_A);
double G_bar_classical(double omega0_rad_s, double T_K, double eta_A);

// Sensitivity bandwidths, full widths in rad/s  (S3.12).  xi <= 0 selects xi_opt(n, eta).
struct Bandwidths {
  double linewidth = 0;         // omega0/Q
  double noise_doubling = 0;    // (omega0/Q) sqrt(A)/xi        ([C] F45-F46)
  double thermal_gt_squid = 0;  // (omega0/Q) sqrt(2xi(2n+1) - eta^2 - xi^2)/xi; 0 if never
  double equivalent = 0;        // (pi/4)(omega0/Q) sqrt(A)/xi = int domega0 theta/theta_max
  double xi = 0;                // the xi used
};
Bandwidths sensitivity_bandwidths(double omega0_rad_s, double Q, double n, double eta_A, double xi = -1.0);

// Amplitude ring-up time 2Q/omega0 [s] (spec Sec. 5.5).
double ringup_time_s(double omega0_rad_s, double Q);

// SQUID coupling realising a given xi  (S3.5), SI, Foster-normalised PSDs:
//   M = sqrt(2 R0 S_Phi / (hbar omega0 xi)),  eta_A = 2 sqrt(S_Phi S_J)/hbar.
// S_Phi [Wb^2 s] and S_J [A^2 s] are half the usual one-sided PSDs in Wb^2/Hz, A^2/Hz.
struct SquidCoupling {
  double M_H = 0;
  double eta_A = 0;
};
SquidCoupling squid_coupling_from_xi(double xi, double omega0_rad_s, double R0_ohm, double S_Phi_Wb2_s,
                                     double S_J_A2_s);

// Lumped-element detector + readout.  FEM-facing natural-HL geometry (spec Sec. 1.2).
struct Detector {
  double psi_nat_T_m3 = 0;  // Psi = (1/I) int A_I . B_lab d^3x  [T m^3]  (S1.1)
  double L_p_m = 0;         // L_p = (1/I^2) int |B_I|^2 d^3x  [m]; L_SI = mu0 L_p [H]  (S1.2)
  double Q = 0;             // intrinsic resonator quality factor omega0 L / R0
  double T_K = 0;           // loss-weighted physical temperature [K]
  double eta_A = 0;         // amplifier noise 2 sqrt(S_Phi S_J)/hbar (=1 at the SQL)  (S3.5)
  double R_exp = 0;         // R(omega) = R0 (omega/omega0)^R_exp within a tuning (S3.6); 0 = constant

  static Detector from_natural(double psi_T_m3, double L_p_m, double Q, double T_K, double eta_A);
  // From SI values Psi_SI [T H m^2] and L [H]  (S1.3).
  static Detector from_SI(double psi_SI_T_H_m2, double L_H, double Q, double T_K, double eta_A);

  double psi_SI() const;  // [T H m^2]
  double L_H() const;     // [H]
  double F_nat() const;   // Psi^4/L_p^2 [T^4 m^10]  (S1.4)
  double F_SI() const;    // mu0^2 F_nat [T^4 H^2 m^8]  (S1.4)
};

// Noise PSDs for one tuning (omega0, xi) at frequency omega  (S3.3-S3.6), SI, Foster norm.:
//   Z2      |Z|^2 = R(omega)^2 + L^2 (omega-omega0)^2 (omega+omega0)^2/omega^2   [Ohm^2]
//   SV_th   R(omega) hbar omega (2 n(omega,T) + 1)                              [V^2 s]
//   SV_ba   omega^2 M^2 S_J = hbar eta^2 R0 omega^2 / (2 xi omega0)            [V^2 s]
//   SI_imp  S_Phi/M^2 = hbar xi omega0 / (2 R0)                                 [A^2 s]
//   N_V     SV_th + SV_ba + Z2 SI_imp  (total noise referred to a series EMF)   [V^2 s]
//   lambda_B N_V / Z2  (total current-noise PSD)                                [A^2 s]
// with R0 = omega0 L / Q.  noise_psd_detuned takes domega = omega - omega0 separately so
// that |Z|^2 is free of cancellation near resonance.
struct NoisePSD {
  double Z2 = 0, SV_th = 0, SV_ba = 0, SI_imp = 0, N_V = 0, lambda_B = 0;
};
NoisePSD noise_psd(double omega_rad_s, double omega0_rad_s, const Detector& d, double xi);
NoisePSD noise_psd_detuned(double omega_rad_s, double domega_rad_s, double omega0_rad_s,
                           const Detector& d, double xi);

// Mean axion EMF PSD  (S3.2):  S_{V,a}(omega) = (pi/2) Phi_a^2 (omega^4/omega_a^2) L(omega)  [V^2 s].
double signal_psd_V(double omega_rad_s, double omega_a_rad_s, double Phi_a_Wb,
                    double v0_kms = kV0_kms, double vobs_kms = kVobs_kms);

// Signal and noise PSDs of one tuning on a frequency grid ("signal PSD vs noise PSD at a
// tuning").  xi <= 0 selects xi_opt(n(omega0)).  s = lambda_a/lambda_B = S_{V,a}/N_V (S3.6).
struct PSDCurve {
  double f0_Hz = 0, xi = 0, m_a_eV = 0, g_GeV = 0, Phi_a_Wb = 0;
  std::vector<double> f_Hz;
  std::vector<double> S_Va_V2_s;       // axion EMF PSD (S3.2)
  std::vector<double> N_V_V2_s;        // total noise referred to a series EMF (S3.6)
  std::vector<double> lambda_a_A2_s;   // axion current PSD S_{V,a}/|Z|^2
  std::vector<double> lambda_B_A2_s;   // noise current PSD (S3.6)
  std::vector<double> SV_th_V2_s, SV_ba_V2_s, SI_imp_A2_s;  // noise components (S3.3-S3.5)
  std::vector<double> s;               // per-bin signal-to-noise ratio
};
PSDCurve psd_at_tuning(const Detector& d, double f0_Hz, double m_a_eV, double g_GeV,
                       const std::vector<double>& f_Hz, double xi = -1.0,
                       double rho_GeV_cm3 = kRhoDM_GeV_cm3, double v0_kms = kV0_kms,
                       double vobs_kms = kVobs_kms);

// ------------------------------------------------------------------ Sec. 4: statistics
// Asimov TS densities per bin  (S4.2-S4.4; [F] Eqs. 41, 44):
//   weak s^2;  limit 2[ln(1+s) - s/(1+s)] (null Asimov data);  discovery 2[s - ln(1+s)].
// limit/discovery use their Taylor series for |s| < 0.05 (no cancellation).
enum class Stat { Weak, Limit, Discovery };
double h_weak(double s);
double h_limit(double s);
double h_discovery(double s);
double h_stat(Stat st, double s);

// Standard normal CDF Phi(x) and quantile Phi^{-1}(p) (Acklam + one Halley step, ~1e-15).
double norm_cdf(double x);
double norm_ppf(double p);
double theta95_exact();  // (Phi^{-1}(0.95))^2 = 2.705543...

// Expected-limit bands |q| = (Phi^{-1}(0.95) +- N)^2  (S4.5; [F] Eq. 56): {upper, lower}.
std::pair<double, double> limit_band_q(double n_sigma);

// Look-elsewhere trials factor N_ma = ln(f_max/f_min)/(alpha v0 vobs)  (S4.6; [F] Eq. 62).
double n_independent_masses(double f_min_Hz, double f_max_Hz, double v0_kms = kV0_kms,
                            double vobs_kms = kVobs_kms, double alpha_lee = kAlphaLEE);

// Local TS threshold for a global significance  (S4.7; [F] Eqs. 58-60):
//   TS = [Phi^{-1}(1 - p/N_ma)]^2,  p = sided * (1 - Phi(sigma)).  sided = 1 (spec default) or 2.
double ts_discovery_threshold(double global_sigma, double N_ma, int sided = 1);

// Weak-signal ratio g_disc/g_95 = (TS_thresh/Theta)^{1/4}  (spec Sec. 4.2).
double g_disc_over_g95_weak(double TS_thresh, double Theta = kTheta95);

// ------------------------------------------------------------------ numerics (shared)
namespace detail {
// Gauss-Legendre nodes/weights on [-1, 1] (Newton on P_n; |error| ~ 1e-15).
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w);
// Adaptive Gauss-Kronrod (7/15) quadrature of a smooth function on [a, b].
// Throws axf::Error if the tolerance is not met within max_intervals subintervals.
double integrate(const std::function<double(double)>& f, double a, double b, double rel_tol = 1e-12,
                 double abs_tol = 0.0, int max_intervals = 4000);
// Brent's root finder on a bracketing interval [a, b] (f(a) f(b) <= 0).
double brent_root(const std::function<double(double)>& f, double a, double b, double xtol = 1e-14,
                  int max_iter = 300);
}  // namespace detail

}  // namespace axf::scan
