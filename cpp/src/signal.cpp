// Axion inputs, halo, readout noise and statistics: docs/physics/scan_and_signal.md
// Secs. 1-4.  Every formula is tagged with its spec equation "(S*)".
#include "axionfem/signal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "axionfem/core.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/units.hpp"

namespace axf::scan {

using namespace axf::units;

namespace {

constexpr double kSqrtPi = 1.77245385090551602729816748334;
const double kSqrt3 = std::sqrt(3.0);

// Speeds km/s -> units of c (spec Sec. 2.2).
inline double kms_to_c(double v_kms) { return v_kms * 1e3 / c; }

// f(v)/v for the SHM (S2.4), stable for all v >= 0:
//   f/v = e^{-(v-vo)^2/v0^2} (1 - e^{-4 v vo/v0^2}) / (sqrt(pi) v0 vo).
// vo -> 0 limit (isotropic Maxwellian): 4 v e^{-v^2/v0^2} / (sqrt(pi) v0^3).
inline double f_over_v(double v, double v0, double vo) {
  if (v <= 0) return 0.0;
  if (vo <= 0) return 4.0 * v * std::exp(-v * v / (v0 * v0)) / (kSqrtPi * v0 * v0 * v0);
  const double dv = v - vo;
  return std::exp(-dv * dv / (v0 * v0)) * (-std::expm1(-4.0 * v * vo / (v0 * v0))) / (kSqrtPi * v0 * vo);
}

}  // namespace

// ============================================================ Sec. 1-2: axion inputs
double omega_from_mass(double m_a_eV) { return m_a_eV * eV / hbar; }
double mass_from_omega(double omega_rad_s) { return omega_rad_s * hbar / eV; }

// (S2.1)
double g_qcd_GeV(double m_a_eV, double E_over_N, double C_gamma, double ma_fa_GeV2) {
  AXF_REQUIRE(m_a_eV > 0, strcat("g_qcd_GeV: axion mass must be positive, got ", m_a_eV, " eV"));
  const double fa_GeV = ma_fa_GeV2 / (m_a_eV * 1e-9);
  return alpha / (2.0 * pi * fa_GeV) * std::abs(E_over_N - C_gamma);
}

// (S2.2) kappa_a = g sqrt(2 rho) in natural HL [eV].
double kappa_a_eV(double g_GeV, double rho_GeV_cm3) {
  return g_GeV * 1e-9 * std::sqrt(2.0 * rho_eV4(rho_GeV_cm3));
}

// (S2.3) kappa_a[1/m] = g_SI sqrt(2 rho_SI hbar c), g_SI = g/(1e9 e) [1/J], rho_SI [J/m^3].
double kappa_a_inv_m(double g_GeV, double rho_GeV_cm3) {
  const double g_SI = g_GeV / GeV;
  const double rho_SI = rho_GeV_cm3 * GeV * 1e6;
  return g_SI * std::sqrt(2.0 * rho_SI * hbar * c);
}

// (S2.3) kappa_SI = kappa[1/m]/mu0 [1/H].
double kappa_a_SI(double g_GeV, double rho_GeV_cm3) { return kappa_a_inv_m(g_GeV, rho_GeV_cm3) / mu0; }

// (S2.3) Phi_a[Wb] = kappa[1/m] Psi[T m^3].
double flux_amplitude_Wb(double g_GeV, double psi_nat_T_m3, double rho_GeV_cm3) {
  return kappa_a_inv_m(g_GeV, rho_GeV_cm3) * psi_nat_T_m3;
}

// ============================================================ Sec. 2.2-2.3: halo
// (S2.4)
double f_shm(double v_over_c, double v0_kms, double vobs_kms) {
  return v_over_c * f_over_v(v_over_c, kms_to_c(v0_kms), kms_to_c(vobs_kms));
}

// (S2.5) L(omega) = f(v)/(omega_a v), v = sqrt(2 domega/omega_a).
double line_shape_detuned(double domega, double omega_a, double v0_kms, double vobs_kms) {
  if (!(domega > 0) || !(omega_a > 0)) return 0.0;
  const double v = std::sqrt(2.0 * domega / omega_a);
  return f_over_v(v, kms_to_c(v0_kms), kms_to_c(vobs_kms)) / omega_a;
}
double line_shape(double omega, double omega_a, double v0_kms, double vobs_kms) {
  return line_shape_detuned(omega - omega_a, omega_a, v0_kms, vobs_kms);
}

// (S2.6) Q_a = int_0^inf dv f^2/v = int_0^inf v (f/v)^2 dv; the integrand is smooth in v
// (no sqrt edge, which only appears in the omega variable).  Truncated at vobs + 12 v0
// (integrand < e^{-144} of its peak).
double Q_axion_shm(double v0_kms, double vobs_kms) {
  AXF_REQUIRE(v0_kms > 0 && vobs_kms >= 0,
              strcat("Q_axion_shm: need v0 > 0 and vobs >= 0 km/s, got v0=", v0_kms, ", vobs=", vobs_kms));
  const double v0 = kms_to_c(v0_kms), vo = kms_to_c(vobs_kms);
  auto fn = [&](double v) {
    const double g = f_over_v(v, v0, vo);
    return v * g * g;
  };
  // Split at the peak region so the adaptive rule starts with resolved panels.
  const double vmax = vo + 12.0 * v0;
  const double vmid = std::min(vo + v0, 0.5 * vmax);
  return detail::integrate(fn, 0.0, vmid, 1e-14) + detail::integrate(fn, vmid, vmax, 1e-14);
}

double coherence_time_s(double f_Hz, double v0_kms, double vobs_kms) {
  return 1.0 / (f_Hz * kms_to_c(v0_kms) * kms_to_c(vobs_kms));
}

void line_v_grid(int n_v, double v0_kms, double vobs_kms, std::vector<double>& v, std::vector<double>& w) {
  AXF_REQUIRE(n_v >= 2, strcat("line_v_grid: need at least 2 nodes, got ", n_v));
  detail::gauss_legendre(n_v, v, w);
  const double vmax = kms_to_c(vobs_kms) + 7.0 * kms_to_c(v0_kms);  // spec S5.9: v_max = vobs + 7 v0
  for (int i = 0; i < n_v; ++i) {
    v[i] = 0.5 * vmax * (v[i] + 1.0);
    w[i] = 0.5 * vmax * w[i];
  }
}

// ============================================================ Sec. 3: circuit and noise
// (S3.1)
double n_bose(double omega_rad_s, double T_K) {
  if (!(T_K > 0)) return 0.0;
  return 1.0 / std::expm1(hbar * omega_rad_s / (kB * T_K));
}

// (S3.7)
double A_noise(double xi, double n, double eta) { return xi * xi + 2.0 * (2.0 * n + 1.0) * xi + eta * eta; }
// (S3.8)
double G_bar_xi(double xi, double n, double eta) { return xi / std::pow(A_noise(xi, n, eta), 1.5); }
// (S3.9)
double xi_opt(double n, double eta) {
  const double b = 2.0 * n + 1.0;
  return 2.0 * eta * eta / (b + std::sqrt(b * b + 8.0 * eta * eta));
}
// (S3.11)
double G_bar_highT(double n, double eta) { return 1.0 / (3.0 * kSqrt3 * eta * (2.0 * n + 1.0)); }
double G_bar_classical(double omega0, double T_K, double eta) {
  return hbar * omega0 / (6.0 * kSqrt3 * eta * kB * T_K);
}

// (S3.12)
Bandwidths sensitivity_bandwidths(double omega0, double Q, double n, double eta, double xi) {
  if (!(xi > 0)) xi = xi_opt(n, eta);
  Bandwidths b;
  b.xi = xi;
  const double A = A_noise(xi, n, eta);
  b.linewidth = omega0 / Q;
  b.noise_doubling = b.linewidth * std::sqrt(A) / xi;
  b.thermal_gt_squid = b.linewidth * std::sqrt(std::max(2.0 * xi * (2.0 * n + 1.0) - eta * eta - xi * xi, 0.0)) / xi;
  b.equivalent = 0.25 * pi * b.linewidth * std::sqrt(A) / xi;
  return b;
}

double ringup_time_s(double omega0, double Q) { return 2.0 * Q / omega0; }

// (S3.5)
SquidCoupling squid_coupling_from_xi(double xi, double omega0, double R0, double S_Phi, double S_J) {
  SquidCoupling s;
  s.M_H = std::sqrt(2.0 * R0 * S_Phi / (hbar * omega0 * xi));
  s.eta_A = 2.0 * std::sqrt(S_Phi * S_J) / hbar;
  return s;
}

// ------------------------------------------------------------ Detector (S1.3-S1.4)
Detector Detector::from_natural(double psi_T_m3, double L_p_m, double Q, double T_K, double eta_A) {
  return Detector{psi_T_m3, L_p_m, Q, T_K, eta_A, 0.0};
}
Detector Detector::from_SI(double psi_SI_T_H_m2, double L_H, double Q, double T_K, double eta_A) {
  return Detector{psi_si_to_nat(psi_SI_T_H_m2), inductance_si_to_nat(L_H), Q, T_K, eta_A, 0.0};
}
double Detector::psi_SI() const { return mu0 * psi_nat_T_m3; }
double Detector::L_H() const { return mu0 * L_p_m; }
double Detector::F_nat() const {
  const double p2 = psi_nat_T_m3 * psi_nat_T_m3;
  return p2 * p2 / (L_p_m * L_p_m);
}
double Detector::F_SI() const { return F_nat_to_si(F_nat()); }

// ------------------------------------------------------------ noise PSDs (S3.3-S3.6)
NoisePSD noise_psd_detuned(double omega, double domega, double omega0, const Detector& d, double xi) {
  AXF_REQUIRE(omega > 0 && omega0 > 0 && xi > 0 && d.Q > 0 && d.L_p_m > 0,
              strcat("noise_psd: need omega, omega0, xi, Q, L_p > 0 (got omega=", omega, ", omega0=", omega0,
                     ", xi=", xi, ", Q=", d.Q, ", L_p=", d.L_p_m, ")"));
  const double L = d.L_H();
  const double R0 = omega0 * L / d.Q;                                             // (S3.1)
  const double R = d.R_exp == 0.0 ? R0 : R0 * std::pow(omega / omega0, d.R_exp);  // (S3.6) R(omega)
  const double X = L * domega * (omega + omega0) / omega;                         // omega L - 1/(omega C)
  NoisePSD p;
  p.Z2 = R * R + X * X;                                                           // (S3.1)
  p.SV_th = R * hbar * omega * (2.0 * n_bose(omega, d.T_K) + 1.0);                // (S3.3)
  p.SV_ba = hbar * d.eta_A * d.eta_A * R0 * omega * omega / (2.0 * xi * omega0);  // (S3.5)
  p.SI_imp = hbar * xi * omega0 / (2.0 * R0);                                     // (S3.5)
  p.N_V = p.SV_th + p.SV_ba + p.Z2 * p.SI_imp;                                    // (S3.6)
  p.lambda_B = (p.SV_th + p.SV_ba) / p.Z2 + p.SI_imp;                             // (S3.6)
  return p;
}
NoisePSD noise_psd(double omega, double omega0, const Detector& d, double xi) {
  return noise_psd_detuned(omega, omega - omega0, omega0, d, xi);
}

// (S3.2)
double signal_psd_V(double omega, double omega_a, double Phi_a_Wb, double v0_kms, double vobs_kms) {
  const double r2 = (omega / omega_a) * (omega / omega_a);
  return 0.5 * pi * Phi_a_Wb * Phi_a_Wb * r2 * r2 * omega_a * omega_a *
         line_shape(omega, omega_a, v0_kms, vobs_kms);
}

PSDCurve psd_at_tuning(const Detector& d, double f0_Hz, double m_a_eV, double g_GeV,
                       const std::vector<double>& f_Hz, double xi, double rho, double v0_kms,
                       double vobs_kms) {
  const double w0 = 2.0 * pi * f0_Hz;
  const double wa = omega_from_mass(m_a_eV);
  const double fa = wa / (2.0 * pi);
  if (!(xi > 0)) xi = xi_opt(n_bose(w0, d.T_K), d.eta_A);
  PSDCurve out;
  out.f0_Hz = f0_Hz;
  out.xi = xi;
  out.m_a_eV = m_a_eV;
  out.g_GeV = g_GeV;
  out.Phi_a_Wb = flux_amplitude_Wb(g_GeV, d.psi_nat_T_m3, rho);
  const size_t N = f_Hz.size();
  out.f_Hz = f_Hz;
  for (auto* v : {&out.S_Va_V2_s, &out.N_V_V2_s, &out.lambda_a_A2_s, &out.lambda_B_A2_s, &out.SV_th_V2_s,
                  &out.SV_ba_V2_s, &out.SI_imp_A2_s, &out.s})
    v->resize(N);
  for (size_t i = 0; i < N; ++i) {
    const double w = 2.0 * pi * f_Hz[i];
    // detunings formed from the frequency differences (no cancellation at the line edge)
    const double r = w / wa;
    const double SVa = 0.5 * pi * out.Phi_a_Wb * out.Phi_a_Wb * r * r * w * w *
                       line_shape_detuned(2.0 * pi * (f_Hz[i] - fa), wa, v0_kms, vobs_kms);  // (S3.2)
    const NoisePSD p = noise_psd_detuned(w, 2.0 * pi * (f_Hz[i] - f0_Hz), w0, d, xi);
    out.S_Va_V2_s[i] = SVa;
    out.N_V_V2_s[i] = p.N_V;
    out.lambda_a_A2_s[i] = SVa / p.Z2;
    out.lambda_B_A2_s[i] = p.lambda_B;
    out.SV_th_V2_s[i] = p.SV_th;
    out.SV_ba_V2_s[i] = p.SV_ba;
    out.SI_imp_A2_s[i] = p.SI_imp;
    out.s[i] = SVa / p.N_V;  // (S3.6)
  }
  return out;
}

// ============================================================ Sec. 4: statistics
// (S4.2)
double h_weak(double s) { return s * s; }

// (S4.3) 2[ln(1+s) - s/(1+s)] = sum_{k>=2} (-1)^k 2(k-1)/k s^k
double h_limit(double s) {
  if (std::abs(s) < 0.05) {
    double sum = 0.0, sk = s * s, sign = 1.0;
    for (int k = 2; k <= 26; ++k) {
      sum += sign * 2.0 * (k - 1) / k * sk;
      sk *= s;
      sign = -sign;
    }
    return sum;
  }
  return 2.0 * (std::log1p(s) - s / (1.0 + s));
}

// (S4.4) 2[s - ln(1+s)] = sum_{k>=2} (-1)^k (2/k) s^k
double h_discovery(double s) {
  if (std::abs(s) < 0.05) {
    double sum = 0.0, sk = s * s, sign = 1.0;
    for (int k = 2; k <= 26; ++k) {
      sum += sign * 2.0 / k * sk;
      sk *= s;
      sign = -sign;
    }
    return sum;
  }
  return 2.0 * (s - std::log1p(s));
}

double h_stat(Stat st, double s) {
  switch (st) {
    case Stat::Weak: return h_weak(s);
    case Stat::Limit: return h_limit(s);
    case Stat::Discovery: return h_discovery(s);
  }
  return h_weak(s);
}

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

namespace {
// Lower-tail quantile for 0 < p <= 0.5: Acklam's rational approximation (|rel err| < 1.2e-9)
// followed by two Halley steps on Phi(x) - p (Phi evaluated through erfc, accurate in the
// lower tail).
double ppf_lower(double p) {
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01,  -1.328068155288572e+01};
  static const double cc[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                              -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
  static const double dd[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                              3.754408661907416e+00};
  double x;
  if (p < 0.02425) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) /
        ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1.0);
  } else {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }
  for (int it = 0; it < 2; ++it) {
    const double e = norm_cdf(x) - p;
    const double u = e * std::sqrt(2.0 * pi) * std::exp(0.5 * x * x);
    x -= u / (1.0 + 0.5 * x * u);
  }
  return x;
}
}  // namespace

double norm_ppf(double p) {
  if (!(p > 0)) return -std::numeric_limits<double>::infinity();
  if (!(p < 1)) return std::numeric_limits<double>::infinity();
  if (p > 0.5) return -ppf_lower(1.0 - p);  // 1 - p is exact for p in [0.5, 1] (Sterbenz)
  return ppf_lower(p);
}

double theta95_exact() {
  const double z = norm_ppf(0.95);
  return z * z;
}

// (S4.5)
std::pair<double, double> limit_band_q(double n_sigma) {
  const double z = norm_ppf(0.95);
  return {(z + n_sigma) * (z + n_sigma), (z - n_sigma) * (z - n_sigma)};
}

// (S4.6)
double n_independent_masses(double f_min_Hz, double f_max_Hz, double v0_kms, double vobs_kms, double alpha_lee) {
  AXF_REQUIRE(f_max_Hz > f_min_Hz && f_min_Hz > 0,
              strcat("n_independent_masses: need 0 < f_min < f_max, got ", f_min_Hz, ", ", f_max_Hz));
  return std::log(f_max_Hz / f_min_Hz) / (alpha_lee * kms_to_c(v0_kms) * kms_to_c(vobs_kms));
}

// (S4.7)  Phi^{-1}(1 - q) = -Phi^{-1}(q) is evaluated in the tail (q = p/N_ma), so the
// threshold keeps full precision even when p/N_ma ~ 1e-14.
double ts_discovery_threshold(double global_sigma, double N_ma, int sided) {
  AXF_REQUIRE(sided == 1 || sided == 2, strcat("ts_discovery_threshold: sided must be 1 or 2, got ", sided));
  AXF_REQUIRE(N_ma >= 1, strcat("ts_discovery_threshold: N_ma must be >= 1, got ", N_ma));
  const double p = sided * 0.5 * std::erfc(global_sigma / std::sqrt(2.0));
  const double q = p / N_ma;
  if (!(q < 0.5)) return 0.0;
  const double z = norm_ppf(q);
  return z * z;
}

double g_disc_over_g95_weak(double TS_thresh, double Theta) { return std::pow(TS_thresh / Theta, 0.25); }

// ============================================================ numerics
namespace detail {

void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  AXF_REQUIRE(n >= 1, "gauss_legendre: need n >= 1");
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  const int m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5)), pp = 1.0;
    for (int it = 0; it < 100; ++it) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      const double z1 = z;
      z = z1 - p1 / pp;
      if (std::abs(z - z1) < 1e-16) break;
    }
    // final derivative at the converged node
    {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
  if (n % 2 == 1) x[m - 1] = 0.0;
}

namespace {
struct Panel {
  double a, b, result, err, resabs;
};

// QUADPACK qk15 (Piessens et al.) with its error heuristic.
Panel qk15(const std::function<double(double)>& f, double a, double b) {
  static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                                0.207784955007898467600689403773245, 0.0};
  static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
  static const double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                               0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
  const double eps = std::numeric_limits<double>::epsilon();
  const double centr = 0.5 * (a + b), hl = 0.5 * (b - a), dhl = std::abs(hl);
  double fv1[7], fv2[7];
  const double fc = f(centr);
  double resg = fc * wg[3], resk = fc * wgk[7], resabs = std::abs(resk);
  for (int j = 0; j < 3; ++j) {
    const int jt = 2 * j + 1;
    const double ab = hl * xgk[jt];
    const double f1 = f(centr - ab), f2 = f(centr + ab);
    fv1[jt] = f1;
    fv2[jt] = f2;
    resg += wg[j] * (f1 + f2);
    resk += wgk[jt] * (f1 + f2);
    resabs += wgk[jt] * (std::abs(f1) + std::abs(f2));
  }
  for (int j = 0; j < 4; ++j) {
    const int jt = 2 * j;
    const double ab = hl * xgk[jt];
    const double f1 = f(centr - ab), f2 = f(centr + ab);
    fv1[jt] = f1;
    fv2[jt] = f2;
    resk += wgk[jt] * (f1 + f2);
    resabs += wgk[jt] * (std::abs(f1) + std::abs(f2));
  }
  const double reskh = 0.5 * resk;
  double resasc = wgk[7] * std::abs(fc - reskh);
  for (int j = 0; j < 7; ++j) resasc += wgk[j] * (std::abs(fv1[j] - reskh) + std::abs(fv2[j] - reskh));
  Panel p{a, b, resk * hl, std::abs((resk - resg) * hl), resabs * dhl};
  resasc *= dhl;
  if (resasc != 0.0 && p.err != 0.0) p.err = resasc * std::min(1.0, std::pow(200.0 * p.err / resasc, 1.5));
  if (p.resabs > std::numeric_limits<double>::min() / (50.0 * eps)) p.err = std::max(50.0 * eps * p.resabs, p.err);
  return p;
}
}  // namespace

double integrate(const std::function<double(double)>& f, double a, double b, double rel_tol, double abs_tol,
                 int max_intervals) {
  if (a == b) return 0.0;
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<Panel> P{qk15(f, a, b)};
  for (;;) {
    double I = 0, E = 0, floor_err = 0;
    size_t worst = 0;
    for (size_t i = 0; i < P.size(); ++i) {
      I += P[i].result;
      E += P[i].err;
      floor_err += 50.0 * eps * P[i].resabs;
      if (P[i].err > P[worst].err) worst = i;
    }
    if (!std::isfinite(I))
      throw Error(strcat("integrate: non-finite integrand on [", a, ", ", b, "]"), "",
                  "check the inputs of the quantity being integrated (temperatures, masses, couplings)");
    if (E <= std::max(abs_tol, rel_tol * std::abs(I)) || E <= 2.0 * floor_err) return I;
    if (static_cast<int>(P.size()) >= max_intervals)
      throw Error(strcat("integrate: tolerance ", rel_tol, " not reached on [", a, ", ", b, "] (estimated error ",
                         E, " for integral ", I, ")"),
                  "", "the integrand may be singular or badly scaled; split the range or loosen the tolerance");
    const Panel w = P[worst];
    const double mid = 0.5 * (w.a + w.b);
    P[worst] = qk15(f, w.a, mid);
    P.push_back(qk15(f, mid, w.b));
  }
}

// Brent (1973) zeroin.
double brent_root(const std::function<double(double)>& f, double a, double b, double xtol, int max_iter) {
  double fa = f(a), fb = f(b);
  if (fa == 0) return a;
  if (fb == 0) return b;
  AXF_REQUIRE(fa * fb < 0, strcat("brent_root: root not bracketed: f(", a, ")=", fa, ", f(", b, ")=", fb));
  double cc = a, fc = fa, d = b - a, e = d;
  const double eps = std::numeric_limits<double>::epsilon();
  for (int it = 0; it < max_iter; ++it) {
    if ((fb > 0) == (fc > 0)) {
      cc = a;
      fc = fa;
      d = e = b - a;
    }
    if (std::abs(fc) < std::abs(fb)) {
      a = b; b = cc; cc = a;
      fa = fb; fb = fc; fc = fa;
    }
    const double tol = 2.0 * eps * std::abs(b) + 0.5 * xtol;
    const double m = 0.5 * (cc - b);
    if (std::abs(m) <= tol || fb == 0) return b;
    if (std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
      double p, q, r;
      const double s = fb / fa;
      if (a == cc) {
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0) q = -q; else p = -p;
      if (2.0 * p < std::min(3.0 * m * q - std::abs(tol * q), std::abs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = m;
        e = m;
      }
    } else {
      d = m;
      e = m;
    }
    a = b;
    fa = fb;
    b += (std::abs(d) > tol) ? d : (m > 0 ? tol : -tol);
    fb = f(b);
  }
  throw Error("brent_root: no convergence", "", "increase max_iter or check the bracket");
}

}  // namespace detail

}  // namespace axf::scan
