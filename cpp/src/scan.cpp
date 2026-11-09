// Scan rate, requirement, time allocation, reach, exact discrete Asimov sums,
// finite-band kernel/LP and long-detector factors: docs/physics/scan_and_signal.md
// Secs. 5-7.  Every formula carries its spec tag "(S*)".
#include "axionfem/scan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf::scan {

using namespace axf::units;

// ================================================================== original API
double mass_to_freq(double m_a_eV) { return m_a_eV * eV / (2.0 * pi * hbar); }
double freq_to_mass(double f_Hz) { return 2.0 * pi * hbar * f_Hz / eV; }

// (S2.1).  |E/N - C| as in the spec: g is a magnitude (KSVZ has E/N - C < 0).
double g_agg_GeV(double m_a_eV, const spec::ScanSpec& s) {
  const double m_GeV = m_a_eV * 1e-9;
  const double fa_GeV = s.ma_fa / m_GeV;
  return alpha / (2.0 * pi * fa_GeV) * std::abs(s.E_over_N - s.C_QCD);
}

double rho_eV4(double rho_GeV_cm3) {
  const double cm_inv_eV = 1e-2 / hbarc_eVm;  // 1 cm in eV^-1
  return rho_GeV_cm3 * 1e9 / (cm_inv_eV * cm_inv_eV * cm_inv_eV);
}

double F_SI_to_nat(double F_SI) { return F_SI / (mu0 * mu0); }

// (S5.4) in natural units (eV).
Requirement F_required(const spec::Benchmark& b, const spec::ScanSpec& s) {
  const double m1 = freq_to_mass(s.f_min), m2 = freq_to_mass(s.f_max);
  const double g1 = g_agg_GeV(m1, s) * 1e-9;  // eV^-1
  const double T_eV = kB * b.T / eV;
  const double rho = rho_eV4(s.rho_DM);
  const double t_eV = s.live_time * year * eV / hbar;  // eV^-1
  const double ratio4 = std::pow(m1 / m2, 4);
  Requirement r;
  r.F_nat_eV = 3.0 * std::sqrt(3.0) / (pi * pi) * s.Theta95 * b.eta_A * T_eV /
               (s.Q_axion * b.Q * rho * rho * std::pow(g1, 4) * t_eV) * (1.0 - ratio4);
  const double T4 = std::pow(tesla_in_eV2(), 4), m10 = std::pow(metre_in_inv_eV(), 10);
  r.F_nat = r.F_nat_eV / (T4 * m10);
  r.F_SI = r.F_nat * mu0 * mu0;
  r.g_min_GeV = g1 * 1e9;
  return r;
}

double occupation(double f_Hz, double T_K) {
  const double x = h * f_Hz / (kB * T_K);
  return 1.0 / std::expm1(x);
}

// (S3.10) G-bar at xi_opt (S3.9).
double G_bar(double n, double eta) {
  const double a = 2.0 * n + 1.0;
  const double xi = 2.0 * eta * eta / (a + std::sqrt(a * a + 8.0 * eta * eta));
  return xi / std::pow(xi * xi + 2.0 * a * xi + eta * eta, 1.5);
}

// ================================================================== helpers
namespace {

constexpr double kSqrt3 = 1.73205080756887729352744634151;

// 1 T^4 m^10 in eV^-2 (S1.4), with the same conversions as F_required.
double F_T4m10_in_inv_eV2() { return std::pow(tesla_in_eV2(), 4) * std::pow(metre_in_inv_eV(), 10); }

Requirement requirement_from_SI(double F_SI, double g_min_GeV) {
  Requirement r;
  r.F_SI = F_SI;
  r.F_nat = F_si_to_nat(F_SI);
  r.F_nat_eV = r.F_nat * F_T4m10_in_inv_eV2();
  r.g_min_GeV = g_min_GeV;
  return r;
}

void check_detector(const Detector& d, const char* where) {
  AXF_REQUIRE(d.psi_nat_T_m3 > 0 && d.L_p_m > 0 && d.Q > 0 && d.T_K >= 0 && d.eta_A > 0,
              strcat(where, ": detector needs Psi > 0, L_p > 0, Q > 0, T >= 0, eta_A > 0 (got Psi=", d.psi_nat_T_m3,
                     " T m^3, L_p=", d.L_p_m, " m, Q=", d.Q, ", T=", d.T_K, " K, eta_A=", d.eta_A, ")"),
              "", "Psi and L_p are natural-HL values (SI / mu0); use Detector::from_SI for SI inputs");
}

void check_band(double m1, double m2, const char* where) {
  AXF_REQUIRE(m1 > 0 && m2 > m1, strcat(where, ": need 0 < m1 < m2, got m1=", m1, " eV, m2=", m2, " eV"));
}

// Per-(tuning, node) signal-to-noise ratios s at a reference coupling plus the quadrature
// weights t_i/(2 pi) omega_a v_k w_k of (S5.9).  s scales as g^2.
struct SGrid {
  size_t nt = 0, nv = 0;
  double g_ref = 0;
  std::vector<double> s, wt;
};

SGrid s_grid(const Detector& d, double m_a_eV, double g_GeV, const Tunings& tun, const ScanParams& p, int n_v) {
  check_detector(d, "theta_per_tuning");
  AXF_REQUIRE(m_a_eV > 0 && g_GeV > 0, strcat("theta_per_tuning: need m_a > 0 and g > 0, got ", m_a_eV, ", ", g_GeV));
  AXF_REQUIRE(tun.omega0_rad_s.size() == tun.t_s.size() && (tun.xi.empty() || tun.xi.size() == tun.t_s.size()),
              strcat("theta_per_tuning: Tunings vectors differ in length (omega0 ", tun.omega0_rad_s.size(), ", t ",
                     tun.t_s.size(), ", xi ", tun.xi.size(), ")"),
              "", "xi may be empty (then xi_opt(n(omega0)) is used per tuning)");
  SGrid G;
  G.nt = tun.t_s.size();
  G.nv = static_cast<size_t>(n_v);
  G.g_ref = g_GeV;
  std::vector<double> v, w;
  line_v_grid(n_v, p.v0_kms, p.vobs_kms, v, w);
  const double wa = omega_from_mass(m_a_eV);
  const double Phi = flux_amplitude_Wb(g_GeV, d.psi_nat_T_m3, p.rho_GeV_cm3);
  std::vector<double> dm(G.nv), om(G.nv), SVa(G.nv), jac(G.nv);
  for (size_t k = 0; k < G.nv; ++k) {
    dm[k] = 0.5 * wa * v[k] * v[k];  // omega - omega_a, exact to roundoff
    om[k] = wa + dm[k];
    const double r = om[k] / wa;
    // (S3.2) with L = f(v)/(omega_a v)  (S2.5)
    const double L = f_shm(v[k], p.v0_kms, p.vobs_kms) / (wa * v[k]);
    SVa[k] = 0.5 * pi * Phi * Phi * r * r * r * r * wa * wa * L;
    jac[k] = wa * v[k] * w[k];  // d omega = omega_a v dv
  }
  G.s.resize(G.nt * G.nv);
  G.wt.resize(G.nt * G.nv);
  for (size_t i = 0; i < G.nt; ++i) {
    const double w0 = tun.omega0_rad_s[i];
    const double xi = tun.xi.empty() ? xi_opt(n_bose(w0, d.T_K), d.eta_A) : tun.xi[i];  // (S3.9)
    const double pref = tun.t_s[i] / (2.0 * pi);
    for (size_t k = 0; k < G.nv; ++k) {
      const NoisePSD np = noise_psd_detuned(om[k], (wa - w0) + dm[k], w0, d, xi);  // (S3.3-S3.6)
      G.s[i * G.nv + k] = SVa[k] / np.N_V;                                         // (S3.6)
      G.wt[i * G.nv + k] = pref * jac[k];
    }
  }
  return G;
}

// Sum (S5.9) at coupling g = g_ref sqrt(r):  sum_ik wt_ik h(r s_ik).
double theta_sum(const SGrid& G, double r, Stat st, std::vector<double>* per_tuning = nullptr,
                 double* s_max = nullptr) {
  double total = 0.0, smax = 0.0;
  if (per_tuning) per_tuning->assign(G.nt, 0.0);
  for (size_t i = 0; i < G.nt; ++i) {
    double acc = 0.0;
    for (size_t k = 0; k < G.nv; ++k) {
      const double s = r * G.s[i * G.nv + k];
      smax = std::max(smax, s);
      acc += G.wt[i * G.nv + k] * h_stat(st, s);
    }
    if (per_tuning) (*per_tuning)[i] = acc;
    total += acc;
  }
  if (s_max) *s_max = smax;
  return total;
}

double half_width_noise_doubling(const Detector& d, double omega) {
  return 0.5 * sensitivity_bandwidths(omega, d.Q, n_bose(omega, d.T_K), d.eta_A).noise_doubling;  // (S3.12)
}

}  // namespace

// ================================================================== Sec. 5.1: scan rate
MassFunction qcd_line(double E_over_N, double C_gamma, double ma_fa_GeV2) {
  return [=](double m) { return g_qcd_GeV(m, E_over_N, C_gamma, ma_fa_GeV2); };
}

double G_bar_at(const Detector& d, double omega0, bool exact) {
  if (exact) return G_bar(n_bose(omega0, d.T_K), d.eta_A);  // (S3.10) with exact n (S3.1)
  return G_bar_classical(omega0, d.T_K, d.eta_A);           // (S3.11)
}

// (S5.2)
double scan_rate_SI(double Phi_a_Wb, double L_H, double Q, double Gbar, double Q_axion, double Theta) {
  const double p2 = Phi_a_Wb * Phi_a_Wb;
  return pi * pi / 8.0 * Q_axion / Theta * p2 * p2 / (hbar * hbar * L_H * L_H) * Q * Gbar;
}

// (S5.1)/(S5.2) at omega0 = omega_a.
double scan_rate(const Detector& d, double m_a_eV, double g_GeV, const ScanParams& p) {
  check_detector(d, "scan_rate");
  const double w = omega_from_mass(m_a_eV);
  const double Phi = flux_amplitude_Wb(g_GeV, d.psi_nat_T_m3, p.rho_GeV_cm3);  // (S2.3)
  return scan_rate_SI(Phi, d.L_H(), d.Q, G_bar_at(d, w, p.exact_Gbar), p.Q_axion, p.Theta);
}

// ================================================================== Sec. 5.2: requirement
// (S5.4), SI form.
Requirement F_req_closed_form(double eta_A, double T_K, double Q, double g1_GeV, double t_s, double m1_eV,
                              double m2_eV, const ScanParams& p) {
  check_band(m1_eV, m2_eV, "F_req_closed_form");
  const double k1 = kappa_a_SI(g1_GeV, p.rho_GeV_cm3);
  const double k2 = k1 * k1;
  const double r = m1_eV / m2_eV, r2 = r * r;
  const double F_SI = 12.0 * kSqrt3 / (pi * pi) * p.Theta * eta_A * kB * T_K * hbar /
                      (p.Q_axion * Q * k2 * k2 * t_s) * (1.0 - r2 * r2);
  return requirement_from_SI(F_SI, g1_GeV);
}

// (S5.3): F_SI = 8 Theta hbar^2/(pi^2 Q_a t) int dw/(Q kappa_SI^4 G-bar), integrated in ln w.
Requirement F_req_general(double m1_eV, double m2_eV, double t_s, const MassFunction& g_t,
                          const MassFunction& Q_of_m, const MassFunction& T_of_m, const MassFunction& eta_of_m,
                          const ScanParams& p) {
  check_band(m1_eV, m2_eV, "F_req_general");
  AXF_REQUIRE(t_s > 0, strcat("F_req_general: live time must be positive, got ", t_s, " s"));
  auto integrand = [&](double lnw) {
    const double w = std::exp(lnw);
    const double m = mass_from_omega(w);
    const double T = T_of_m(m), eta = eta_of_m(m);
    const double G = p.exact_Gbar ? G_bar(n_bose(w, T), eta) : G_bar_classical(w, T, eta);
    const double k = kappa_a_SI(g_t(m), p.rho_GeV_cm3), k2 = k * k;
    return w / (Q_of_m(m) * k2 * k2 * G);
  };
  const double I = detail::integrate(integrand, std::log(omega_from_mass(m1_eV)), std::log(omega_from_mass(m2_eV)),
                                     1e-12);
  const double F_SI = 8.0 * p.Theta * hbar * hbar / (pi * pi * p.Q_axion * t_s) * I;
  return requirement_from_SI(F_SI, g_t(m1_eV));
}

Requirement F_req_general(double m1_eV, double m2_eV, double t_s, double Q, double T_K, double eta_A,
                          const MassFunction& g_t, const ScanParams& p) {
  return F_req_general(
      m1_eV, m2_eV, t_s, g_t, [Q](double) { return Q; }, [T_K](double) { return T_K; },
      [eta_A](double) { return eta_A; }, p);
}

// ================================================================== Sec. 5.3: allocation
// (S5.6) t_req = int dw0 tau(w0) with tau = 1/(d omega0/dt at g_t), in ln w0.
double time_target_curve_s(const Detector& d, double m1_eV, double m2_eV, const MassFunction& g_t,
                           const ScanParams& p) {
  check_band(m1_eV, m2_eV, "time_target_curve_s");
  auto integrand = [&](double lnw) {
    const double w = std::exp(lnw);
    const double m = mass_from_omega(w);
    return w / scan_rate(d, m, g_t(m), p);
  };
  return detail::integrate(integrand, std::log(omega_from_mass(m1_eV)), std::log(omega_from_mass(m2_eV)), 1e-12);
}

// (S5.5-S5.6)
Allocation allocation_target(const Detector& d, double m1_eV, double m2_eV, double t_total_s,
                             const MassFunction& g_t, const ScanParams& p) {
  check_detector(d, "allocation_target");
  AXF_REQUIRE(t_total_s > 0, strcat("allocation_target: live time must be positive, got ", t_total_s, " s"));
  Allocation a;
  a.kind = Allocation::Kind::Target;
  a.m1_eV = m1_eV;
  a.m2_eV = m2_eV;
  a.t_total_s = t_total_s;
  a.t_req_s = time_target_curve_s(d, m1_eV, m2_eV, g_t, p);
  const double scale = t_total_s / a.t_req_s;
  a.tau = [d, g_t, p, scale](double w0) {
    const double m = mass_from_omega(w0);
    return scale / scan_rate(d, m, g_t(m), p);
  };
  return a;
}

// (S5.7)
double tau_log_uniform(double omega0, double t_total_s, double omega1, double omega2) {
  return t_total_s / (omega0 * std::log(omega2 / omega1));
}

Allocation allocation_log_uniform(double m1_eV, double m2_eV, double t_total_s) {
  check_band(m1_eV, m2_eV, "allocation_log_uniform");
  Allocation a;
  a.kind = Allocation::Kind::LogUniform;
  a.m1_eV = m1_eV;
  a.m2_eV = m2_eV;
  a.t_total_s = a.t_req_s = t_total_s;
  const double w1 = omega_from_mass(m1_eV), w2 = omega_from_mass(m2_eV);
  a.tau = [=](double w0) { return tau_log_uniform(w0, t_total_s, w1, w2); };
  return a;
}

// ================================================================== Sec. 5.4: reach
// (S5.8) in SI:  Theta = tau (pi^2/8) Q_a Q G-bar kappa_SI(g)^4 F_SI / hbar^2 with kappa ~ g.
double g95_local(const Detector& d, double m_a_eV, double tau, const ScanParams& p) {
  check_detector(d, "g95_local");
  AXF_REQUIRE(tau > 0, strcat("g95_local: dwell density must be positive, got ", tau, " s/(rad/s)"));
  const double w = omega_from_mass(m_a_eV);
  const double k1 = kappa_a_SI(1.0, p.rho_GeV_cm3), k2 = k1 * k1;  // kappa at g = 1 GeV^-1
  const double G = G_bar_at(d, w, p.exact_Gbar);
  const double g4 = 8.0 * p.Theta * hbar * hbar / (pi * pi * tau * p.Q_axion * d.Q * G * k2 * k2 * d.F_SI());
  return std::pow(g4, 0.25);
}

namespace {
// Fill the self-describing band echo (ScanBand) from a detector and the scan conventions.
void fill_band(ScanBand& b, const Detector& d, double m1_eV, double m2_eV, const ScanParams& p) {
  b.m1_eV = m1_eV;
  b.m2_eV = m2_eV;
  b.f_min_Hz = mass_to_freq(m1_eV);
  b.f_max_Hz = mass_to_freq(m2_eV);
  b.Psi_SI_T_H_m2 = d.psi_SI();
  b.L_H = d.L_H();
  b.Q = d.Q;
  b.T_K = d.T_K;
  b.eta_A = d.eta_A;
  b.F_SI_T4_H2_m8 = d.F_SI();
  b.Q_axion = p.Q_axion;
  b.Theta = p.Theta;
  b.rho_GeV_cm3 = p.rho_GeV_cm3;
  b.v0_kms = p.v0_kms;
  b.vobs_kms = p.vobs_kms;
  b.exact_Gbar = p.exact_Gbar;
}

const char* stat_name(Stat s) {
  return s == Stat::Weak ? "weak" : (s == Stat::Limit ? "limit" : "discovery");
}
}  // namespace

ReachCurve reach_local(const Detector& d, const Allocation& a, const std::vector<double>& m_eV,
                       const ScanParams& p) {
  AXF_REQUIRE(static_cast<bool>(a.tau), "reach_local: the allocation has no dwell density",
              "", "build it with allocation_target or allocation_log_uniform");
  ReachCurve rc;
  rc.m_eV = m_eV;
  for (double m : m_eV) {
    const double w = omega_from_mass(m);
    const double tau = a.tau(w);
    rc.f_Hz.push_back(w / (2.0 * pi));
    rc.tau_s_per_rad_s.push_back(tau);
    rc.g95_GeV.push_back(g95_local(d, m, tau, p));
  }
  fill_band(rc.band, d, a.m1_eV, a.m2_eV, p);
  rc.band.live_time_s = a.t_total_s;
  rc.band.t_req_s = a.t_req_s;
  rc.band.allocation = a.kind == Allocation::Kind::Target ? "target" : "log_uniform";
  rc.band.n_targets = static_cast<int>(m_eV.size());
  rc.band.stat = "weak";  // (S5.8) is the weak-signal Asimov closed form
  return rc;
}

ReachCurve reach_local(const Detector& d, const Allocation& a, int n_points, const ScanParams& p) {
  AXF_REQUIRE(n_points >= 2, strcat("reach_local: need n_points >= 2, got ", n_points));
  std::vector<double> m(n_points);
  const double l1 = std::log(a.m1_eV), l2 = std::log(a.m2_eV);
  for (int i = 0; i < n_points; ++i) m[i] = std::exp(l1 + (l2 - l1) * i / (n_points - 1));
  m.front() = a.m1_eV;
  m.back() = a.m2_eV;
  return reach_local(d, a, m, p);
}

ReachCurve reach_local(const Detector& d, double m1_eV, double m2_eV, double live_time_s, Allocation::Kind kind,
                       int n_points, const MassFunction& g_t, const ScanParams& p) {
  const Allocation a = kind == Allocation::Kind::Target ? allocation_target(d, m1_eV, m2_eV, live_time_s, g_t, p)
                                                        : allocation_log_uniform(m1_eV, m2_eV, live_time_s);
  return reach_local(d, a, n_points, p);
}

// ================================================================== Sec. 5.4: exact discrete sums
// (S5.9)
std::vector<double> theta_per_tuning(const Detector& d, double m_a_eV, double g_GeV, const Tunings& tun, Stat stat,
                                     const ScanParams& p, int n_v, double* s_max) {
  const SGrid G = s_grid(d, m_a_eV, g_GeV, tun, p, n_v);
  std::vector<double> th;
  theta_sum(G, 1.0, stat, &th, s_max);
  return th;
}

double theta_discrete(const Detector& d, double m_a_eV, double g_GeV, const Tunings& tun, Stat stat,
                      const ScanParams& p, int n_v) {
  const SGrid G = s_grid(d, m_a_eV, g_GeV, tun, p, n_v);
  return theta_sum(G, 1.0, stat);
}

// (S5.10)
double g95_discrete(const Detector& d, double m_a_eV, const Tunings& tun, Stat stat, const ScanParams& p, int n_v,
                    double* s_max_at_g95) {
  const double g_ref = 1e-19;  // any value: s scales as g^2
  const SGrid G = s_grid(d, m_a_eV, g_ref, tun, p, n_v);
  const double th_w = theta_sum(G, 1.0, Stat::Weak);
  AXF_REQUIRE(th_w > 0, strcat("g95_discrete: no sensitivity at m_a = ", m_a_eV, " eV (Theta = 0)"), "",
              "add tunings near omega_a or check the dwell times");
  const double g_w = g_ref * std::pow(p.Theta / th_w, 0.25);  // weak: Theta ~ g^4
  double g = g_w;
  if (stat != Stat::Weak) {
    auto f = [&](double lg) {
      const double r = std::exp(2.0 * (lg - std::log(g_ref)));
      return theta_sum(G, r, stat) - p.Theta;
    };
    // h_limit, h_discovery <= s^2, so the root lies at g >= g_w.
    double lo = std::log(g_w) - 2.0, hi = std::log(g_w) + 2.0;
    int it = 0;
    while (f(hi) < 0 && it++ < 30) hi += 2.0;
    AXF_REQUIRE(f(hi) >= 0, "g95_discrete: could not bracket the exact-Asimov root");
    g = std::exp(detail::brent_root(f, lo, hi, 1e-13));
  }
  if (s_max_at_g95) {
    double smax = 0;
    theta_sum(G, (g / g_ref) * (g / g_ref), Stat::Weak, nullptr, &smax);
    *s_max_at_g95 = smax;
  }
  return g;
}

ReachCurve reach_discrete(const Detector& d, const Tunings& tun, const std::vector<double>& m_eV, Stat stat,
                          const ScanParams& p, double window_half_widths, int n_v) {
  ReachCurve rc;
  rc.m_eV = m_eV;
  if (!m_eV.empty()) {
    const auto mm = std::minmax_element(m_eV.begin(), m_eV.end());
    fill_band(rc.band, d, *mm.first, *mm.second, p);
  }
  rc.band.allocation = "discrete";
  rc.band.n_settings = static_cast<int>(tun.omega0_rad_s.size());
  rc.band.n_targets = static_cast<int>(m_eV.size());
  rc.band.stat = stat_name(stat);
  for (double t : tun.t_s) rc.band.live_time_s += t;
  double worst = 0;
  for (double m : m_eV) {
    const double w = omega_from_mass(m);
    const double hw = half_width_noise_doubling(d, w);
    Tunings sel;
    for (size_t i = 0; i < tun.omega0_rad_s.size(); ++i) {
      if (window_half_widths > 0 && std::abs(tun.omega0_rad_s[i] - w) > window_half_widths * hw) continue;
      sel.omega0_rad_s.push_back(tun.omega0_rad_s[i]);
      sel.t_s.push_back(tun.t_s[i]);
      if (!tun.xi.empty()) sel.xi.push_back(tun.xi[i]);
    }
    rc.f_Hz.push_back(w / (2.0 * pi));
    if (sel.t_s.empty()) {
      rc.g95_GeV.push_back(std::numeric_limits<double>::infinity());
      rc.s_max.push_back(0.0);
      continue;
    }
    double smax = 0;
    rc.g95_GeV.push_back(g95_discrete(d, m, sel, stat, p, n_v, &smax));
    rc.s_max.push_back(smax);
    worst = std::max(worst, smax);
  }
  if (worst > 0.05)
    log(1, stat == Stat::Weak
               ? strcat("reach_discrete: max per-bin s at g95 is ", worst,
                        " > 0.05; the weak-signal (s^2) Asimov form is inaccurate there (spec Sec. 5.5) -- use Stat::Limit")
               : strcat("reach_discrete: max per-bin s at g95 is ", worst,
                        " > 0.05 (outside the weak-signal regime; the exact Asimov h was used, but local "
                        "formulas (S5.1-S5.8) underestimate g95 there, spec Sec. 5.5)"));
  return rc;
}

Tunings dense_tunings_around(const Detector& d, double m_a_eV, const Allocation& a, double half_widths,
                             double steps_per_hw) {
  AXF_REQUIRE(static_cast<bool>(a.tau), "dense_tunings_around: the allocation has no dwell density");
  const double w = omega_from_mass(m_a_eV);
  const double hw = half_width_noise_doubling(d, w);
  const int N = static_cast<int>(2.0 * half_widths * steps_per_hw) + 1;
  const double dx = 2.0 * half_widths / (N - 1);
  const double step = hw / steps_per_hw;
  Tunings t;
  for (int i = 0; i < N; ++i) {
    const double x = (i == N - 1) ? half_widths : -half_widths + i * dx;  // numpy.linspace
    const double w0 = w + hw * x;
    t.omega0_rad_s.push_back(w0);
    t.t_s.push_back(a.tau(w0) * step);
  }
  return t;
}

Tunings discretise_allocation(const Detector& d, const Allocation& a, double steps_per_fw) {
  AXF_REQUIRE(static_cast<bool>(a.tau), "discretise_allocation: the allocation has no dwell density");
  AXF_REQUIRE(steps_per_fw > 0, strcat("discretise_allocation: steps_per_fw must be > 0, got ", steps_per_fw));
  const double w1 = omega_from_mass(a.m1_eV), w2 = omega_from_mass(a.m2_eV);
  Tunings t;
  double e = w1;
  while (e < w2) {
    double delta = 2.0 * half_width_noise_doubling(d, e) / steps_per_fw;
    delta = std::min(delta, w2 - e);
    const double c = e + 0.5 * delta;
    t.omega0_rad_s.push_back(c);
    t.t_s.push_back(a.tau(c) * delta);
    e += delta;
  }
  return t;
}

double n_tunings(const Detector& d, double m1_eV, double m2_eV, double steps_per_fw) {
  check_band(m1_eV, m2_eV, "n_tunings");
  auto integrand = [&](double lnw) {
    const double w = std::exp(lnw);
    return steps_per_fw * w / (2.0 * half_width_noise_doubling(d, w));
  };
  return detail::integrate(integrand, std::log(omega_from_mass(m1_eV)), std::log(omega_from_mass(m2_eV)), 1e-12);
}

// ================================================================== Sec. 6: finite band
// (S6.1-S6.2): Theta per unit dwell = theta_per_tuning with t = 1 s.
std::vector<double> kernel_row(const Detector& d, double m_a_eV, double g_GeV, const std::vector<double>& omega0,
                               const std::vector<double>& xi, Stat stat, const ScanParams& p, int n_v) {
  Tunings t{omega0, std::vector<double>(omega0.size(), 1.0), xi};
  return theta_per_tuning(d, m_a_eV, g_GeV, t, stat, p, n_v);
}

// (S6.3)
double kernel_narrow_line(const Detector& d, double m_a_eV, double g_GeV, double omega0, double xi,
                          const ScanParams& p) {
  check_detector(d, "kernel_narrow_line");
  const double wa = omega_from_mass(m_a_eV);
  const double Phi = flux_amplitude_Wb(g_GeV, d.psi_nat_T_m3, p.rho_GeV_cm3);
  const double NV = noise_psd(wa, omega0, d, xi).N_V;
  const double P2 = Phi * Phi;
  return pi / 8.0 * p.Q_axion * wa * wa * wa * P2 * P2 / (NV * NV);
}

// (S6.4) via the dual LP (see header).
LPSchedule lp_schedule(const std::vector<std::vector<double>>& K, double Theta) {
  const size_t nt = K.size();
  AXF_REQUIRE(nt > 0 && !K[0].empty(), "lp_schedule: empty kernel matrix");
  const size_t ns = K[0].size();
  std::vector<double> best(nt, 0.0);
  for (size_t k = 0; k < nt; ++k) {
    AXF_REQUIRE(K[k].size() == ns, strcat("lp_schedule: kernel row ", k, " has ", K[k].size(), " entries, expected ", ns));
    for (double v : K[k]) {
      AXF_REQUIRE(v >= 0 && std::isfinite(v), strcat("lp_schedule: kernel entries must be finite and >= 0 (row ", k, ")"));
      best[k] = std::max(best[k], v);
    }
    AXF_REQUIRE(best[k] > 0, strcat("lp_schedule: target ", k, " has no sensitive setting"), "",
                "add settings near that mass or drop the target");
  }
  // Scaling: a_kj = K_kj / best_k (max 1 per target), t = t_unit t', b_k = Theta/(best_k t_unit) in (0, 1].
  double t_unit = 0;
  for (size_t k = 0; k < nt; ++k) t_unit = std::max(t_unit, Theta / best[k]);
  std::vector<double> b(nt);
  for (size_t k = 0; k < nt; ++k) b[k] = Theta / (best[k] * t_unit);
  // Dual tableau: rows = settings j, columns = y_k (nt) | slack_j (ns) | rhs.
  const size_t nc = nt + ns, W = nc + 1;
  std::vector<double> T(ns * W, 0.0), r(nc, 0.0);
  std::vector<size_t> basis(ns);
  for (size_t j = 0; j < ns; ++j) {
    for (size_t k = 0; k < nt; ++k) T[j * W + k] = K[k][j] / best[k];
    T[j * W + nt + j] = 1.0;
    T[j * W + nc] = 1.0;
    basis[j] = nt + j;
  }
  for (size_t k = 0; k < nt; ++k) r[k] = b[k];
  const double tol_r = 1e-12, tol_p = 1e-12;
  LPSchedule out;
  int degenerate = 0;
  const int max_pivots = 50 * static_cast<int>(nc);
  for (;;) {
    // entering column: Dantzig (largest reduced cost); Bland after a run of degenerate pivots
    size_t e = nc;
    if (degenerate < 50) {
      double rb = tol_r;
      for (size_t c = 0; c < nc; ++c)
        if (r[c] > rb) { rb = r[c]; e = c; }
    } else {
      for (size_t c = 0; c < nc; ++c)
        if (r[c] > tol_r) { e = c; break; }
    }
    if (e == nc) break;  // optimal
    // ratio test (ties -> smallest basic index)
    size_t l = ns;
    double best_ratio = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < ns; ++j) {
      const double a = T[j * W + e];
      if (a > tol_p) {
        const double ratio = std::max(T[j * W + nc], 0.0) / a;  // rhs >= 0 up to round-off
        if (l == ns || ratio < best_ratio - 1e-15 * best_ratio ||
            (std::abs(ratio - best_ratio) <= 1e-15 * best_ratio && basis[j] < basis[l])) {
          best_ratio = ratio;
          l = j;
        }
      }
    }
    AXF_REQUIRE(l < ns, "lp_schedule: dual LP unbounded (primal infeasible)");
    degenerate = (best_ratio <= 0.0) ? degenerate + 1 : 0;
    // pivot
    double* rowl = &T[l * W];
    const double piv = rowl[e];
    for (size_t c = 0; c < W; ++c) rowl[c] /= piv;
    for (size_t j = 0; j < ns; ++j) {
      if (j == l) continue;
      double* rowj = &T[j * W];
      const double f = rowj[e];
      if (f == 0.0) continue;
      for (size_t c = 0; c < W; ++c) rowj[c] -= f * rowl[c];
      rowj[e] = 0.0;
    }
    const double fr = r[e];
    for (size_t c = 0; c < nc; ++c) r[c] -= fr * rowl[c];
    r[e] = 0.0;
    basis[l] = e;
    AXF_REQUIRE(++out.pivots < max_pivots, "lp_schedule: simplex did not terminate (cycling?)");
  }
  // primal t'_j = dual multiplier of setting row j = -(reduced cost of slack j)
  out.t_s.assign(ns, 0.0);
  for (size_t j = 0; j < ns; ++j) out.t_s[j] = std::max(0.0, -r[nt + j]) * t_unit;
  double cov = std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < nt; ++k) {
    double s = 0;
    for (size_t j = 0; j < ns; ++j) s += K[k][j] * out.t_s[j];
    cov = std::min(cov, s / Theta);
  }
  AXF_REQUIRE(cov > 0.5, strcat("lp_schedule: simplex returned an infeasible schedule (min coverage ", cov, ")"));
  if (cov < 1.0)
    for (double& t : out.t_s) t /= cov;  // enforce feasibility after round-off
  out.min_coverage = std::max(cov, 1.0);
  out.t_total_s = 0;
  for (double t : out.t_s) out.t_total_s += t;
  return out;
}

// ================================================== Sec. 6 with the exact network noise (C4.5)
void NoiseCurve::validate(const char* where) const {
  AXF_REQUIRE(omega_rad_s.size() == N_V_V2_s.size(),
              strcat(where, ": NoiseCurve has ", omega_rad_s.size(), " frequencies but ", N_V_V2_s.size(),
                     " noise values"));
  AXF_REQUIRE(omega_rad_s.size() >= 2,
              strcat(where, ": NoiseCurve needs at least 2 samples, got ", omega_rad_s.size()), "",
              "sample N_V(omega) over the setting's resonance (axionfem.circuit.Readout.noise_curve)");
  for (size_t i = 0; i < omega_rad_s.size(); ++i) {
    AXF_REQUIRE(omega_rad_s[i] > 0 && std::isfinite(omega_rad_s[i]),
                strcat(where, ": NoiseCurve omega[", i, "] = ", omega_rad_s[i], " is not a positive finite number"));
    AXF_REQUIRE(N_V_V2_s[i] > 0 && std::isfinite(N_V_V2_s[i]),
                strcat(where, ": NoiseCurve N_V[", i, "] = ", N_V_V2_s[i], " V^2 s is not positive and finite"), "",
                "every dissipative branch must have Re Z > 0, or the amplifier an S_Phi > 0");
    if (i) AXF_REQUIRE(omega_rad_s[i] > omega_rad_s[i - 1],
                       strcat(where, ": NoiseCurve frequencies must be strictly ascending (omega[", i, "] = ",
                              omega_rad_s[i], " <= omega[", i - 1, "] = ", omega_rad_s[i - 1], ")"));
  }
}

// log-log interpolation; the end slope is continued outside the table.
double NoiseCurve::at(double w) const {
  const size_t n = omega_rad_s.size();
  size_t i = 0;
  if (w <= omega_rad_s.front())
    i = 0;
  else if (w >= omega_rad_s.back())
    i = n - 2;
  else
    i = static_cast<size_t>(std::upper_bound(omega_rad_s.begin(), omega_rad_s.end(), w) - omega_rad_s.begin()) - 1;
  const double l0 = std::log(omega_rad_s[i]), l1 = std::log(omega_rad_s[i + 1]);
  const double y0 = std::log(N_V_V2_s[i]), y1 = std::log(N_V_V2_s[i + 1]);
  return std::exp(y0 + (y1 - y0) * (std::log(w) - l0) / (l1 - l0));
}

namespace {
// The (S5.9) quadrature with N_V taken from an arbitrary function of omega: the signal PSD
// (S3.2) and the weights are exactly those of s_grid, only the noise is external.
SGrid s_grid_noise(double psi_SI, double m_a_eV, double g_GeV, const std::vector<NoiseFunction>& NV,
                   const std::vector<double>& t_s, const ScanParams& p, int n_v, const char* where) {
  AXF_REQUIRE(psi_SI > 0 && std::isfinite(psi_SI),
              strcat(where, ": need Psi_SI > 0 [T H m^2], got ", psi_SI), "",
              "Psi_SI = mu0 Psi; use results['coupling']['Psi_SI_H_T_m2'] or Psi(omega) from a full-wave sweep");
  AXF_REQUIRE(m_a_eV > 0 && g_GeV > 0, strcat(where, ": need m_a > 0 and g > 0, got ", m_a_eV, ", ", g_GeV));
  AXF_REQUIRE(!NV.empty(), strcat(where, ": no settings (the noise list is empty)"));
  AXF_REQUIRE(NV.size() == t_s.size(),
              strcat(where, ": ", NV.size(), " noise curves but ", t_s.size(), " dwell times"));
  AXF_REQUIRE(n_v >= 8, strcat(where, ": need n_v >= 8 quadrature nodes, got ", n_v));
  SGrid G;
  G.nt = NV.size();
  G.nv = static_cast<size_t>(n_v);
  G.g_ref = g_GeV;
  std::vector<double> v, w;
  line_v_grid(n_v, p.v0_kms, p.vobs_kms, v, w);
  const double wa = omega_from_mass(m_a_eV);
  const double Phi = kappa_a_SI(g_GeV, p.rho_GeV_cm3) * psi_SI;  // (S2.3)
  std::vector<double> om(G.nv), SVa(G.nv), jac(G.nv);
  for (size_t k = 0; k < G.nv; ++k) {
    om[k] = wa + 0.5 * wa * v[k] * v[k];  // omega - omega_a exact to round-off
    const double r = om[k] / wa;
    const double L = f_shm(v[k], p.v0_kms, p.vobs_kms) / (wa * v[k]);  // (S2.5)
    SVa[k] = 0.5 * pi * Phi * Phi * r * r * r * r * wa * wa * L;       // (S3.2)
    jac[k] = wa * v[k] * w[k];
  }
  G.s.resize(G.nt * G.nv);
  G.wt.resize(G.nt * G.nv);
  for (size_t i = 0; i < G.nt; ++i) {
    AXF_REQUIRE(static_cast<bool>(NV[i]), strcat(where, ": setting ", i, " has no noise function"));
    const double pref = t_s[i] / (2.0 * pi);
    for (size_t k = 0; k < G.nv; ++k) {
      const double nv = NV[i](om[k]);
      AXF_REQUIRE(nv > 0 && std::isfinite(nv),
                  strcat(where, ": setting ", i, " gave N_V = ", nv, " V^2 s at omega = ", om[k], " rad/s"), "",
                  "N_V must be positive and finite everywhere the axion line is sampled");
      G.s[i * G.nv + k] = SVa[k] / nv;  // (S3.6)
      G.wt[i * G.nv + k] = pref * jac[k];
    }
  }
  return G;
}

std::vector<NoiseFunction> as_functions(const std::vector<NoiseCurve>& nc, const char* where) {
  std::vector<NoiseFunction> f;
  f.reserve(nc.size());
  for (const auto& c : nc) {
    c.validate(where);
    f.push_back([&c](double w) { return c.at(w); });
  }
  return f;
}
}  // namespace

// (S6.1-S6.2) with the exact N_V(omega): Theta per unit dwell of each setting.
std::vector<double> kernel_row_noise(double psi_SI, double m_a_eV, double g_GeV,
                                     const std::vector<NoiseFunction>& NV, Stat stat, const ScanParams& p,
                                     int n_v) {
  const std::vector<double> t(NV.size(), 1.0);
  const SGrid G = s_grid_noise(psi_SI, m_a_eV, g_GeV, NV, t, p, n_v, "kernel_row_noise");
  std::vector<double> th;
  theta_sum(G, 1.0, stat, &th);
  return th;
}

std::vector<double> kernel_row_noise(double psi_SI, double m_a_eV, double g_GeV,
                                     const std::vector<NoiseCurve>& noise, Stat stat, const ScanParams& p,
                                     int n_v) {
  return kernel_row_noise(psi_SI, m_a_eV, g_GeV, as_functions(noise, "kernel_row_noise"), stat, p, n_v);
}

// (S6.3) with the exact N_V(omega_a).
double kernel_narrow_line_noise(double psi_SI, double m_a_eV, double g_GeV, double N_V, const ScanParams& p) {
  AXF_REQUIRE(psi_SI > 0 && m_a_eV > 0 && g_GeV > 0,
              strcat("kernel_narrow_line_noise: need Psi_SI, m_a, g > 0, got ", psi_SI, ", ", m_a_eV, ", ", g_GeV));
  AXF_REQUIRE(N_V > 0 && std::isfinite(N_V),
              strcat("kernel_narrow_line_noise: N_V must be positive and finite, got ", N_V, " V^2 s"));
  const double wa = omega_from_mass(m_a_eV);
  const double Phi = kappa_a_SI(g_GeV, p.rho_GeV_cm3) * psi_SI;
  const double P2 = Phi * Phi;
  return pi / 8.0 * p.Q_axion * wa * wa * wa * P2 * P2 / (N_V * N_V);
}

// (S6.4) at one target: g95 solving sum_j t_j K_j(m, g) = Theta.
double g95_band(double psi_SI, double m_a_eV, const std::vector<NoiseCurve>& noise, const std::vector<double>& t_s,
                Stat stat, const ScanParams& p, int n_v, double* s_max_at_g95) {
  const double g_ref = 1e-19;  // any value: s scales as g^2
  const SGrid G = s_grid_noise(psi_SI, m_a_eV, g_ref, as_functions(noise, "g95_band"), t_s, p, n_v, "g95_band");
  const double th_w = theta_sum(G, 1.0, Stat::Weak);
  AXF_REQUIRE(th_w > 0, strcat("g95_band: no sensitivity at m_a = ", m_a_eV, " eV (Theta = 0)"), "",
              "add settings tuned near omega_a or increase the dwell times");
  const double g_w = g_ref * std::pow(p.Theta / th_w, 0.25);  // weak: Theta ~ g^4
  double g = g_w;
  if (stat != Stat::Weak) {
    auto f = [&](double lg) {
      const double r = std::exp(2.0 * (lg - std::log(g_ref)));
      return theta_sum(G, r, stat) - p.Theta;
    };
    double lo = std::log(g_w) - 2.0, hi = std::log(g_w) + 2.0;
    int it = 0;
    while (f(hi) < 0 && it++ < 30) hi += 2.0;
    AXF_REQUIRE(f(hi) >= 0, "g95_band: could not bracket the exact-Asimov root");
    g = std::exp(detail::brent_root(f, lo, hi, 1e-13));
  }
  if (s_max_at_g95) {
    double smax = 0;
    theta_sum(G, (g / g_ref) * (g / g_ref), Stat::Weak, nullptr, &smax);
    *s_max_at_g95 = smax;
  }
  return g;
}

ReachCurve reach_band(const std::vector<double>& psi_SI, const std::vector<double>& m_eV,
                      const std::vector<NoiseCurve>& noise, const std::vector<double>& t_s, Stat stat,
                      const ScanParams& p, int n_v) {
  AXF_REQUIRE(!m_eV.empty(), "reach_band: no target masses");
  AXF_REQUIRE(psi_SI.size() == 1 || psi_SI.size() == m_eV.size(),
              strcat("reach_band: Psi_SI has ", psi_SI.size(), " entries; give one per target mass (",
                     m_eV.size(), ") or a single value"),
              "", "one value per mass is how a frequency-dependent coupling Psi(omega) enters");
  ReachCurve rc;
  rc.m_eV = m_eV;
  double worst = 0;
  for (size_t k = 0; k < m_eV.size(); ++k) {
    const double w = omega_from_mass(m_eV[k]);
    double smax = 0;
    rc.f_Hz.push_back(w / (2.0 * pi));
    rc.g95_GeV.push_back(
        g95_band(psi_SI.size() == 1 ? psi_SI[0] : psi_SI[k], m_eV[k], noise, t_s, stat, p, n_v, &smax));
    rc.s_max.push_back(smax);
    worst = std::max(worst, smax);
  }
  const auto mm = std::minmax_element(m_eV.begin(), m_eV.end());
  rc.band.m1_eV = *mm.first;
  rc.band.m2_eV = *mm.second;
  rc.band.f_min_Hz = mass_to_freq(*mm.first);
  rc.band.f_max_Hz = mass_to_freq(*mm.second);
  rc.band.allocation = "lp";
  rc.band.n_settings = static_cast<int>(noise.size());
  rc.band.n_targets = static_cast<int>(m_eV.size());
  rc.band.stat = stat_name(stat);
  rc.band.Psi_SI_T_H_m2 = psi_SI.front();
  rc.band.Q_axion = p.Q_axion;
  rc.band.Theta = p.Theta;
  rc.band.rho_GeV_cm3 = p.rho_GeV_cm3;
  rc.band.v0_kms = p.v0_kms;
  rc.band.vobs_kms = p.vobs_kms;
  rc.band.exact_Gbar = p.exact_Gbar;
  for (double t : t_s) rc.band.live_time_s += t;
  if (worst > 0.05 && stat == Stat::Weak)
    log(1, strcat("reach_band: max per-bin s at g95 is ", worst,
                  " > 0.05; the weak-signal (s^2) Asimov form is inaccurate there (spec Sec. 5.5) -- use "
                  "Stat::Limit"));
  return rc;
}

// ================================================================== Sec. 7: long detectors
double g_solenoid_long(double x) { return (1.0 - x * x) * (1.0 - x * x) / (4.0 * std::log(1.0 / x)); }
double g_toroid_long(double y) { return y * y * std::log(1.0 / y); }

LongOptimum solenoid_long_optimum() {
  // d g_sol/dx = 0  <=>  1 - x^2 + 4 x^2 ln x = 0  (on 0 < x < 1)
  const double x = detail::brent_root([](double x) { return 1.0 - x * x + 4.0 * x * x * std::log(x); }, 0.3, 0.8,
                                      1e-15);
  return {x, g_solenoid_long(x)};
}

LongOptimum toroid_long_optimum() {
  const double y = std::exp(-0.5);
  return {y, g_toroid_long(y)};
}

}  // namespace axf::scan
