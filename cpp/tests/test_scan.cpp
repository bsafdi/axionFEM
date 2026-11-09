// Unit test of the signal / noise / scan / statistics machinery (cpp/src/signal.cpp,
// cpp/src/scan.cpp) against the independent numpy reference
// tests/reference/scan_reference.py and its values tests/reference/scan_reference_values.json
// (spec: docs/physics/scan_and_signal.md).
//
// Tolerances (relative unless stated; chosen from the numerics of both sides):
//   1e-14..1e-12  closed-form expressions evaluated identically (g, G-bar, xi, bandwidths, rates)
//   1e-8          anything converted through units.hpp's hbarc_eVm (1.973269804e-7, CODATA-rounded;
//                 the reference uses hbar c/e = 1.9732698034e-7 -> 3e-10 in hbarc, ~1e-9 in F[eV^-2])
//   1e-10..1e-9   quantities from adaptive quadrature (Q_axion, F_req general, t_req) or root finds
//   1e-8          exact discrete Asimov sums (same Gauss-Legendre rule; Brent vs brentq)
//   bound         LEE thresholds: the reference evaluates ndtri(1 - p/N), which rounds 1 - p/N to
//                 1 ulp; the induced error 2 delta_q/(q z^2) is used as tolerance (up to 2e-4 at 5 sigma),
//                 and the reference path itself is reproduced to 1e-12.
//   1e-6          LP optimum (reference: HiGHS, primal feasibility 1e-7, then a feasibility rescale)
// Set AXF_TEST_VERBOSE=1 to print every check.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/scan.hpp"
#include "axionfem/units.hpp"

using json = nlohmann::json;
using namespace axf::scan;
namespace U = axf::units;

namespace {

struct Group {
  int n = 0, fail = 0;
  double max_err = 0, tol_min = 1e300, tol_max = 0;
};
std::map<std::string, Group> groups;
std::vector<std::string> order;
bool verbose = false;
int n_fail = 0, n_total = 0;

void record(const std::string& grp, const std::string& name, double val, double ref, double err, double tol,
            const char* kind) {
  if (!groups.count(grp)) order.push_back(grp);
  Group& g = groups[grp];
  const bool ok = std::isfinite(val) && err <= tol;
  ++g.n;
  ++n_total;
  if (std::string(kind) != "bool") {  // boolean checks do not enter the error/tolerance summary
    g.max_err = std::max(g.max_err, std::isfinite(err) ? err : 1e300);
    g.tol_min = std::min(g.tol_min, tol);
    g.tol_max = std::max(g.tol_max, tol);
  }
  if (!ok) {
    ++g.fail;
    ++n_fail;
  }
  if (!ok || verbose)
    std::printf("  [%s] %-14s %-62s value=%.15g ref=%.15g %s_err=%.3e tol=%.1e\n", ok ? "PASS" : "FAIL", grp.c_str(),
                name.c_str(), val, ref, kind, err, tol);
}

void check_rel(const std::string& grp, const std::string& name, double val, double ref, double tol) {
  const double err = ref != 0 ? std::abs(val / ref - 1.0) : std::abs(val);
  record(grp, name, val, ref, err, tol, "rel");
}
void check_abs(const std::string& grp, const std::string& name, double val, double ref, double tol) {
  record(grp, name, val, ref, std::abs(val - ref), tol, "abs");
}
#define CHECK(grp, cond) record(grp, #cond, (cond) ? 1.0 : 0.0, 1.0, (cond) ? 0.0 : 1.0, 0.5, "bool")

double J(const json& j, const std::string& a, const std::string& b) { return j.at(a).at(b).get<double>(); }

std::vector<double> geomspace(double a, double b, int n) {
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) x[i] = std::exp(std::log(a) + (std::log(b) - std::log(a)) * i / (n - 1));
  x.front() = a;
  x.back() = b;
  return x;
}

const double YEAR = U::year;

}  // namespace

int main() {
  verbose = std::getenv("AXF_TEST_VERBOSE") != nullptr;
  const std::string path = std::string(AXF_SOURCE_DIR) + "/tests/reference/scan_reference_values.json";
  std::ifstream in(path);
  if (!in) {
    std::printf("cannot open %s\n", path.c_str());
    return 1;
  }
  const json R = json::parse(in);

  const double m1 = 0.5e-9, m2 = 4.1e-9, t3 = 3 * YEAR;
  const MassFunction gdfsz = qcd_line(kEoverN_DFSZ_I, 1.77);
  ScanParams p68;  // default conventions: Q_a = 6.8e5, Theta = 2.71, rho = 0.3
  const double Qa = Q_axion_shm(220, 232);
  ScanParams pQa = p68;
  pQa.Q_axion = Qa;

  // ------------------------------------------------------------ units (units.hpp)
  {
    const json& u = R.at("units_table");
    const std::string g = "units";
    check_rel(g, "hbar_J_s", U::hbar, J(R, "units_table", "hbar_J_s"), 1e-15);
    check_rel(g, "mu0", U::mu0, J(R, "units_table", "mu0_H_per_m"), 1e-15);
    check_rel(g, "kB", U::kB, J(R, "units_table", "kB_J_per_K"), 1e-15);
    check_rel(g, "alpha", U::alpha, J(R, "units_table", "alpha_em"), 1e-15);
    check_rel(g, "year", U::year, J(R, "units_table", "year_s"), 1e-15);
    check_rel(g, "hbar_eV_s", U::hbar / U::eV, J(R, "units_table", "hbar_eV_s"), 1e-15);
    check_rel(g, "one_K_in_eV", U::kB / U::eV, J(R, "units_table", "one_K_in_eV"), 1e-15);
    check_rel(g, "hbarc_eV_m (units.hpp rounded)", U::hbarc_eVm, J(R, "units_table", "hbarc_eV_m"), 1e-9);
    check_rel(g, "one_m_in_inv_eV", U::metre_in_inv_eV(), J(R, "units_table", "one_m_in_inv_eV"), 1e-9);
    check_rel(g, "one_T_in_eV2", U::tesla_in_eV2(), J(R, "units_table", "one_T_in_eV2"), 1e-9);
    check_rel(g, "one_T4_m10_in_inv_eV2", std::pow(U::tesla_in_eV2(), 4) * std::pow(U::metre_in_inv_eV(), 10),
              u.at("one_T4_m10_in_inv_eV2 (F)").get<double>(), 1e-8);
    check_rel(g, "one_GeV_per_cm3_in_eV4", rho_eV4(1.0), J(R, "units_table", "one_GeV_per_cm3_in_eV4"), 1e-8);
    check_rel(g, "one_T2_m3_in_J = 1/mu0", 1.0 / U::mu0, u.at("one_T2_m3_in_J (energy, =1/mu0)").get<double>(), 1e-14);
    check_rel(g, "1 T2 m3 via eV conversions", std::pow(U::tesla_in_eV2(), 2) * std::pow(U::metre_in_inv_eV(), 3) * U::eV,
              1.0 / U::mu0, 1e-8);
    check_rel(g, "Psi unit T m3 in eV^-1", U::tesla_in_eV2() * std::pow(U::metre_in_inv_eV(), 3),
              u.at("one_T_m3_in_inv_eV (Psi)").get<double>(), 1e-8);
    // flux quantum: pi hbar/e [Wb] x (1 Wb natural) = pi/e_HL
    const double wb_nat = U::tesla_in_eV2() * std::pow(U::metre_in_inv_eV(), 2);
    check_rel(g, "flux quantum pi/e_HL", U::pi * U::hbar / U::e * wb_nat,
              u.at("flux_quantum_natural_pi_over_e").get<double>(), 1e-8);
  }

  // ------------------------------------------------------------ axion inputs (S2.1-S2.3)
  {
    const std::string g = "axion";
    const json& a = R.at("axion");
    check_rel(g, "f1_Hz mass_to_freq", mass_to_freq(m1), a.at("f1_Hz").get<double>(), 1e-14);
    check_rel(g, "f2_Hz mass_to_freq", mass_to_freq(m2), a.at("f2_Hz").get<double>(), 1e-14);
    check_rel(g, "f1_Hz omega_from_mass/2pi", omega_from_mass(m1) / (2 * U::pi), a.at("f1_Hz").get<double>(), 1e-14);
    check_rel(g, "freq_to_mass(f1)", freq_to_mass(a.at("f1_Hz").get<double>()), m1, 1e-14);
    check_rel(g, "mass_from_omega", mass_from_omega(omega_from_mass(m2)), m2, 1e-15);
    check_rel(g, "rho_DM_eV4", rho_eV4(0.3), a.at("rho_DM_eV4").get<double>(), 1e-8);
    const std::vector<std::pair<std::string, double>> models = {
        {"KSVZ", kEoverN_KSVZ}, {"DFSZ-I", kEoverN_DFSZ_I}, {"DFSZ-II", kEoverN_DFSZ_II}};
    const std::vector<std::pair<std::string, double>> cs = {
        {"chiPT_1511.02867", 1.92}, {"lattice_Brandt", 1.77}, {"lattice_Lin", 1.965}};
    for (auto& [mn, en] : models)
      for (auto& [cn, cv] : cs) {
        const std::string key = "g_" + mn + "_" + cn + "_at_0.5neV_GeV^-1";
        check_rel(g, key, g_qcd_GeV(m1, en, cv), a.at(key).get<double>(), 1e-13);
        axf::spec::ScanSpec s;
        s.E_over_N = en;
        s.C_QCD = cv;
        check_rel(g, key + " (g_agg_GeV)", g_agg_GeV(m1, s), a.at(key).get<double>(), 1e-13);
      }
    check_rel(g, "g_DFSZ-I 1 neV", gdfsz(1e-9), a.at("g_DFSZ-I_C1.77_at_1neV_GeV^-1").get<double>(), 1e-13);
    check_rel(g, "g_DFSZ-I 4.1 neV", gdfsz(m2), a.at("g_DFSZ-I_C1.77_at_4.1neV_GeV^-1").get<double>(), 1e-13);
    check_rel(g, "kappa eV", kappa_a_eV(1e-19, 0.3), a.at("kappa_a_g1e-19_rho0.3_eV").get<double>(), 1e-8);
    check_rel(g, "kappa SI H^-1", kappa_a_SI(1e-19, 0.3), a.at("kappa_a_g1e-19_rho0.3_SI_H^-1").get<double>(), 1e-12);
    check_rel(g, "kappa m^-1", kappa_a_inv_m(1e-19, 0.3), a.at("kappa_a_g1e-19_rho0.3_m^-1").get<double>(), 1e-12);
    check_rel(g, "kappa m^-1 via natural route", kappa_a_eV(1e-19, 0.3) * U::metre_in_inv_eV(),
              kappa_a_inv_m(1e-19, 0.3), 1e-8);
    check_rel(g, "flux amplitude Wb = kappa Psi", flux_amplitude_Wb(1e-19, 7.0, 0.3), 7.0 * kappa_a_inv_m(1e-19, 0.3),
              1e-15);
  }

  // ------------------------------------------------------------ Q_axion (S2.6)
  {
    const std::string g = "Q_axion";
    const json& q = R.at("Q_axion");
    check_rel(g, "SHM 220/232", Qa, q.at("Qa_SHM_v0_220_vobs_232").get<double>(), 1e-11);
    check_rel(g, "220/220", Q_axion_shm(220, 220), q.at("Qa_v0_220_vobs_220").get<double>(), 1e-11);
    check_rel(g, "220/247", Q_axion_shm(220, 247), q.at("Qa_v0_220_vobs_247(June max approx)").get<double>(), 1e-11);
    check_rel(g, "sqrt3*220 / 235.2 ([C] B1 literal)", Q_axion_shm(std::sqrt(3.0) * 220, 220 + 29.8 * 0.51),
              q.at("Qa_v0_sqrt3x220_vobs_235.2(1803.01627 B1 literal)").get<double>(), 1e-11);
    check_rel(g, "235/250", Q_axion_shm(235, 250), q.at("Qa_v0_235_vobs_250(SHM++-like)").get<double>(), 1e-11);
    check_rel(g, "vobs giving 6.4e5 (root xtol 1e-6 km/s)",
              Q_axion_shm(220, q.at("vobs_kms_giving_6.4e5_at_v0_220").get<double>()), 6.4e5, 1e-8);
    // isotropic limit vobs -> 0: f = 4 v^2 e^{-v^2/v0^2}/(sqrt(pi) v0^3), Q_a = 2/(pi v0^2)
    const double v0c = 220e3 / U::c;
    check_rel(g, "vobs = 0 (Maxwellian) = 2/(pi v0^2)", Q_axion_shm(220, 0), 2 / (U::pi * v0c * v0c), 1e-12);
    check_rel(g, "vobs = 1e-6 km/s -> Maxwellian limit", Q_axion_shm(220, 1e-6), 2 / (U::pi * v0c * v0c), 1e-10);
    // Q_a = omega_a int L^2 domega (S2.5-S2.6), Gauss-Legendre in v
    std::vector<double> v, w;
    line_v_grid(400, 220, 232, v, w);
    const double wa = 1.0e6;
    double s = 0;
    for (size_t k = 0; k < v.size(); ++k) {
      const double L = line_shape_detuned(0.5 * wa * v[k] * v[k], wa);
      s += wa * L * L * wa * v[k] * w[k];
    }
    check_rel(g, "Q_a via line shape (S2.5)", s, Qa, 1e-9);
    double norm = 0;
    for (size_t k = 0; k < v.size(); ++k) norm += f_shm(v[k]) * w[k];
    check_rel(g, "int f dv = 1 (S2.4)", norm, 1.0, 1e-9);
  }

  // ------------------------------------------------------------ G-bar grid (S3.7-S3.11)
  {
    const std::string g = "Gbar_grid";
    for (const auto& e : R.at("Gbar_grid")) {
      const double n = e.at("n").get<double>(), eta = e.at("eta_A").get<double>();
      const std::string tag = "n=" + std::to_string(n) + ",eta=" + std::to_string(eta);
      const double x = xi_opt(n, eta);
      check_rel(g, "xi_opt " + tag, x, e.at("xi_opt").get<double>(), 1e-13);
      check_rel(g, "G_bar " + tag, G_bar(n, eta), e.at("Gbar_opt").get<double>(), 1e-13);
      check_rel(g, "G_bar_xi(xi_opt) " + tag, G_bar_xi(x, n, eta), e.at("Gbar_opt").get<double>(), 1e-13);
      check_rel(g, "A_opt " + tag, A_noise(x, n, eta), e.at("A_opt").get<double>(), 1e-13);
      check_rel(g, "G_bar_highT " + tag, G_bar_highT(n, eta),
                e.at("Gbar_highT_1/(6sqrt3 eta (n+1/2))").get<double>(), 1e-13);
      check_rel(g, "G_bar_xi(1) " + tag, G_bar_xi(1.0, n, eta), e.at("Gbar_at_xi=1").get<double>(), 1e-13);
      // xi_opt maximises G-bar (self test)
      CHECK(g, G_bar_xi(x * (1 + 1e-4), n, eta) < G_bar_xi(x, n, eta) &&
                   G_bar_xi(x * (1 - 1e-4), n, eta) < G_bar_xi(x, n, eta));
    }
    check_rel(g, "G_bar(0,1) = 4/27", G_bar(0, 1), 4.0 / 27.0, 1e-15);
    check_rel(g, "occupation(f,T) = n_bose(2 pi f,T)", occupation(1e5, 0.01), n_bose(2 * U::pi * 1e5, 0.01), 1e-8);
  }

  // ------------------------------------------------------------ F_req (S5.3-S5.4)
  const std::map<std::string, std::vector<double>> bench = {
      {"optimistic", {2e7, 0.010, 0.1}}, {"conservative_toroid_CORE", {1e6, 0.030, 10.0}},
      {"conservative_solenoid", {1e5, 0.030, 10.0}}};
  std::map<std::string, double> F_closed, F_exact;
  {
    const std::string g = "F_req";
    const double g1 = gdfsz(m1);
    for (auto& [name, b] : bench) {
      const json& f = R.at("F_req").at(name);
      const double Q = b[0], T = b[1], eta = b[2];
      const Requirement c = F_req_closed_form(eta, T, Q, g1, t3, m1, m2, p68);
      F_closed[name] = c.F_nat;
      check_rel(g, name + " closed T4m10", c.F_nat, f.at("F_req_closed_form_T4_m10").get<double>(), 1e-12);
      check_rel(g, name + " closed SI", c.F_SI, f.at("F_req_closed_form_SI_T4_H2_m8").get<double>(), 1e-12);
      check_rel(g, name + " closed SI direct", c.F_SI, f.at("F_req_SI_direct_formula_T4_H2_m8").get<double>(), 1e-12);
      check_rel(g, name + " closed eV^-2", c.F_nat_eV, f.at("F_req_closed_form_eV^-2").get<double>(), 1e-8);
      check_rel(g, name + " sqrt closed", std::sqrt(c.F_nat), f.at("sqrt_F_req_T2_m5").get<double>(), 1e-12);
      check_rel(g, name + " g_min", c.g_min_GeV, g1, 1e-15);
      // original API (natural route, ScanSpec band = [f1, f2])
      axf::spec::Benchmark B;
      B.Q = Q;
      B.T = T;
      B.eta_A = eta;
      axf::spec::ScanSpec S;
      S.f_min = mass_to_freq(m1);
      S.f_max = mass_to_freq(m2);
      const Requirement o = F_required(B, S);
      check_rel(g, name + " F_required (original API) T4m10", o.F_nat, f.at("F_req_closed_form_T4_m10").get<double>(), 1e-8);
      check_rel(g, name + " F_required (original API) SI", o.F_SI, f.at("F_req_closed_form_SI_T4_H2_m8").get<double>(), 1e-8);
      check_rel(g, name + " F_required eV^-2", o.F_nat_eV, f.at("F_req_closed_form_eV^-2").get<double>(), 1e-8);
      // general, exact G-bar
      const Requirement e = F_req_general(m1, m2, t3, Q, T, eta, gdfsz, p68);
      F_exact[name] = e.F_nat;
      check_rel(g, name + " exact Gbar T4m10", e.F_nat, f.at("F_req_exactGbar_T4_m10").get<double>(), 1e-10);
      check_abs(g, name + " exact/closed - 1", e.F_nat / c.F_nat - 1, f.at("exact_over_closed_minus1").get<double>(), 1e-11);
      check_rel(g, name + " exact Gbar, Q_a exact", F_req_general(m1, m2, t3, Q, T, eta, gdfsz, pQa).F_nat,
                f.at("F_req_exactGbar_Qa_exact_6.7795e5_T4_m10").get<double>(), 1e-10);
      ScanParams pth = p68;
      pth.Theta = theta95_exact();
      check_rel(g, name + " exact Gbar, Theta exact", F_req_general(m1, m2, t3, Q, T, eta, gdfsz, pth).F_nat,
                f.at("F_req_exactGbar_Theta95_2.7055_T4_m10").get<double>(), 1e-10);
      check_rel(g, name + " closed omega2 = 1 MHz", F_req_closed_form(eta, T, Q, g1, t3, m1, freq_to_mass(1e6), p68).F_nat,
                f.at("F_req_closed_omega2_1MHz_T4_m10").get<double>(), 1e-12);
      // general with classical G-bar reproduces the closed form (S5.3 -> S5.4)
      ScanParams pcl = p68;
      pcl.exact_Gbar = false;
      check_rel(g, name + " general(classical G) = closed", F_req_general(m1, m2, t3, Q, T, eta, gdfsz, pcl).F_nat,
                c.F_nat, 1e-11);
    }
  }

  // ------------------------------------------------------------ long detectors (Sec. 7)
  {
    const std::string g = "long_detectors";
    const json& L = R.at("long_detectors");
    const LongOptimum so = solenoid_long_optimum(), to = toroid_long_optimum();
    check_rel(g, "x_max (ref: bounded minimiser, ~1e-8)", so.x, L.at("x_max").get<double>(), 1e-7);
    check_rel(g, "g_sol_max", so.g, L.at("g_sol_max").get<double>(), 1e-13);
    check_rel(g, "g_sol(0.534)", g_solenoid_long(0.534), L.at("g_sol_at_0.534").get<double>(), 1e-14);
    check_rel(g, "y_max", to.x, L.at("y_max_toroid").get<double>(), 1e-15);
    check_rel(g, "g_tor_max", to.g, L.at("g_tor_max").get<double>(), 1e-15);
    for (auto& [name, b] : bench) {
      (void)b;
      const double sF = std::sqrt(F_closed[name]);
      const double WR2 = sF / so.g;  // T^2 m^5 (HL); x 1/mu0 J
      check_rel(g, name + " solenoid W R^2 MJ m^2", WR2 / U::mu0 * 1e-6,
                L.at(name + "_solenoid_WmagR2_MJ_m2").get<double>(), 1e-12);
      check_rel(g, name + " solenoid B^2 H R^4", 2 * WR2 / U::pi, L.at(name + "_solenoid_B2HR4_T2_m5").get<double>(), 1e-12);
      check_rel(g, name + " toroid W R^2 MJ m^2", sF / to.g / U::mu0 * 1e-6,
                L.at(name + "_toroid_WmagR2_MJ_m2").get<double>(), 1e-12);
      check_rel(g, name + " toroid Bin^2 H R^4", 4 * std::exp(2.0) * sF / U::pi,
                L.at(name + "_toroid_Bin2HR4_T2_m5").get<double>(), 1e-12);
    }
  }

  // ------------------------------------------------------------ bandwidths (S3.12)
  {
    const std::string g = "bandwidths";
    for (auto& [name, b] : bench) {
      for (auto [lab, m] : {std::pair<std::string, double>{"f1", m1}, {"f2", m2}}) {
        const json& e = R.at("bandwidths").at(name + "_" + lab);
        const double Q = b[0], T = b[1], eta = b[2];
        const double w = omega_from_mass(m);
        const double n = n_bose(w, T);
        const Bandwidths bw = sensitivity_bandwidths(w, Q, n, eta);
        const std::string t = name + "_" + lab + " ";
        check_rel(g, t + "n", n, e.at("n").get<double>(), 1e-13);
        check_rel(g, t + "xi_opt", bw.xi, e.at("xi_opt").get<double>(), 1e-13);
        check_rel(g, t + "linewidth", bw.linewidth / (2 * U::pi), e.at("resonator_linewidth_Hz").get<double>(), 1e-13);
        check_rel(g, t + "noise doubling", bw.noise_doubling / (2 * U::pi), e.at("fw_noise_doubling_Hz").get<double>(), 1e-13);
        check_rel(g, t + "thermal > SQUID", bw.thermal_gt_squid / (2 * U::pi), e.at("fw_thermal_gt_squid_Hz").get<double>(),
                  1e-12);
        check_rel(g, t + "equivalent", bw.equivalent / (2 * U::pi), e.at("equivalent_width_Hz").get<double>(), 1e-13);
        check_rel(g, t + "fw/f0", bw.noise_doubling / w, e.at("fw_noise_doubling_over_f0").get<double>(), 1e-13);
        check_rel(g, t + "ring-up 2Q/w0", ringup_time_s(w, Q), e.at("ringup_amplitude_2Q_over_omega0_s").get<double>(), 1e-14);
        check_rel(g, t + "coherence 1/(f v0 vobs)", coherence_time_s(w / (2 * U::pi)),
                  e.at("coherence_1/(f v0 vobs)_s").get<double>(), 1e-13);
        check_rel(g, t + "coherence Q_a/f", Qa / (w / (2 * U::pi)), e.at("coherence_Qa/f_s").get<double>(), 1e-11);
      }
    }
  }

  // ------------------------------------------------------------ schedule, optimistic at F = F_req(exact)
  {
    const std::string g = "schedule";
    const json& s = R.at("schedule_optimistic");
    const auto& b = bench.at("optimistic");
    const Detector detF = Detector::from_natural(std::pow(F_exact["optimistic"], 0.25), 1.0, b[0], b[1], b[2]);
    check_rel(g, "t_req = 3 yr at F_req(exact)", time_target_curve_s(detF, m1, m2, gdfsz, p68) / YEAR,
              s.at("t_req_years_at_F=F_req(exact G)").get<double>(), 1e-10);
    check_rel(g, "N tunings (step = FW/10)", n_tunings(detF, m1, m2, 10), s.at("N_tunings_step=FW/10").get<double>(), 1e-9);
    for (auto [lab, m] : {std::pair<std::string, double>{"f1", m1}, {"f2", m2}}) {
      const double tau = 1.0 / scan_rate(detF, m, gdfsz(m), p68);
      const double w = omega_from_mass(m);
      const double fw = sensitivity_bandwidths(w, b[0], n_bose(w, b[1]), b[2]).noise_doubling;
      check_rel(g, "tau " + lab, tau, s.at("tau_" + lab + "_s_per_rad_s").get<double>(), 1e-9);
      check_rel(g, "dwell per tuning " + lab, tau * fw / 10, s.at("dwell_per_tuning_" + lab + "_s").get<double>(), 1e-9);
    }
    // discretised allocation: sum of dwells = live time, count ~ n_tunings
    const Allocation a = allocation_target(detF, m1, m2, t3, gdfsz, p68);
    check_rel(g, "allocation t_req = 3 yr", a.t_req_s / YEAR, 3.0, 1e-10);
    const Tunings tun = discretise_allocation(detF, a, 10);
    double tsum = 0;
    for (double t : tun.t_s) tsum += t;
    check_rel(g, "discretised allocation: sum t_i = t (midpoint rule)", tsum, t3, 1e-6);
    check_rel(g, "discretised allocation: count vs N", static_cast<double>(tun.t_s.size()),
              s.at("N_tunings_step=FW/10").get<double>(), 1e-4);
  }

  // ------------------------------------------------------------ toy scan rates (S5.1-S5.2)
  const Detector toyA = Detector::from_natural(10.0, 1.0, 2e7, 0.010, 0.1);
  const Detector toyB = Detector::from_natural(100.0, 1.0, 1e6, 0.030, 10.0);
  const std::vector<std::pair<std::string, Detector>> toys = {{"toyA", toyA}, {"toyB", toyB}};
  {
    const std::string g = "scan_rate";
    const json& s = R.at("scan_rate_toy");
    for (auto& [tn, det] : toys) {
      for (auto [ml, m] : {std::pair<std::string, double>{"0.5", m1}, {"1", 1e-9}, {"4.1", m2}}) {
        const double r = scan_rate(det, m, gdfsz(m), p68);
        const std::string k = tn + "_m" + ml + "neV_gDFSZ_";
        check_rel(g, k + "domega0dt", r, s.at(k + "domega0dt_rad_s2").get<double>(), 1e-12);
        check_rel(g, k + "dnu0dt Hz/yr", r / (2 * U::pi) * YEAR, s.at(k + "dnu0dt_Hz_per_yr").get<double>(), 1e-12);
      }
      const double r = scan_rate(det, 1e-9, 1e-19, p68);
      check_rel(g, tn + " 1 neV g=1e-19", r, s.at(tn + "_m1neV_g1e-19_domega0dt_rad_s2").get<double>(), 1e-12);
      ScanParams pcl = p68;
      pcl.exact_Gbar = false;
      check_abs(g, tn + " classical/exact - 1", scan_rate(det, 1e-9, 1e-19, pcl) / r - 1,
                s.at(tn + "_m1neV_g1e-19_highT_over_exact_minus1").get<double>(), 1e-12);
      // SI form with explicit Phi_a and L (S5.2)
      const double Phi = flux_amplitude_Wb(1e-19, det.psi_nat_T_m3);
      check_rel(g, tn + " scan_rate_SI", scan_rate_SI(Phi, det.L_H(), det.Q, G_bar_at(det, omega_from_mass(1e-9))), r, 1e-15);
      // natural-unit route (S5.1): (pi^2/8)(Q_a/Theta) Phi^4/L^2 Q G [eV^2] -> rad/s^2
      const double Phi_nat = kappa_a_eV(1e-19) * det.psi_nat_T_m3 * U::tesla_in_eV2() * std::pow(U::metre_in_inv_eV(), 3);
      const double L_nat = det.L_p_m * U::metre_in_inv_eV();
      const double r_nat = U::pi * U::pi / 8 * 6.8e5 / 2.71 * std::pow(Phi_nat, 4) / (L_nat * L_nat) * det.Q *
                           G_bar_at(det, omega_from_mass(1e-9));
      const double s_in_inv_eV = U::eV / U::hbar;
      check_rel(g, tn + " natural (S5.1) vs SI (S5.2)", r_nat * s_in_inv_eV * s_in_inv_eV, r, 1e-8);
    }
    // DMRadio-GUT Eq. 3 anchor
    const json& G = R.at("DMRadio_GUT_anchor");
    const double F_gut = 4 * std::pow(0.1, 4) * std::pow(16.0, 4) * std::pow(10.0, 10.0 / 3.0);
    check_rel(g, "GUT F", F_gut, G.at("F_T4_m10").get<double>(), 1e-14);
    const Detector detG = Detector::from_natural(std::pow(F_gut, 0.25), 1.0, 2e7, 0.010, 0.1);
    ScanParams pG;
    pG.Q_axion = 6.4e5;
    pG.Theta = 9.0;
    pG.rho_GeV_cm3 = 0.45;
    pG.exact_Gbar = false;
    const double rG = scan_rate(detG, freq_to_mass(1e5), 1e-19, pG);
    check_rel(g, "GUT 41 kHz/yr anchor", rG / (2 * U::pi) * YEAR / 1e3, G.at("dnu_dt_kHz_per_yr").get<double>(), 1e-12);
    // Detector::from_SI round trip
    const Detector dsi = Detector::from_SI(toyA.psi_SI(), toyA.L_H(), toyA.Q, toyA.T_K, toyA.eta_A);
    check_rel(g, "Detector::from_SI round trip Psi", dsi.psi_nat_T_m3, toyA.psi_nat_T_m3, 1e-15);
    check_rel(g, "Detector F_SI = mu0^2 F_nat", toyA.F_SI(), U::mu0 * U::mu0 * 1e4, 1e-15);
  }

  // ------------------------------------------------------------ local reach (S5.5-S5.8)
  {
    const std::string g = "reach_local";
    const json& r = R.at("reach_local_3yr_0.5-4.1neV");
    const std::vector<std::pair<std::string, double>> ms = {{"0.5", m1}, {"1", 1e-9}, {"2", 2e-9}, {"4.1", m2}};
    std::vector<double> mv;
    for (auto& [l, m] : ms) mv.push_back(m);
    for (auto& [tn, det] : toys) {
      for (std::string alloc : {"target_DFSZ", "log_uniform"}) {
        const Allocation a = alloc == "target_DFSZ" ? allocation_target(det, m1, m2, t3, gdfsz, p68)
                                                    : allocation_log_uniform(m1, m2, t3);
        const ReachCurve rc = reach_local(det, a, mv, p68);
        for (size_t i = 0; i < ms.size(); ++i) {
          const std::string k = tn + "_" + alloc + "_m" + ms[i].first + "neV_g95_";
          check_rel(g, k + "GeV^-1", rc.g95_GeV[i], r.at(k + "GeV^-1").get<double>(), 1e-9);
          check_rel(g, k + "over_gDFSZ", rc.g95_GeV[i] / gdfsz(mv[i]), r.at(k + "over_gDFSZ").get<double>(), 1e-9);
        }
        // one-call overload: endpoints agree
        const ReachCurve rc2 = reach_local(det, m1, m2, t3,
                                           alloc == "target_DFSZ" ? Allocation::Kind::Target : Allocation::Kind::LogUniform,
                                           11, gdfsz, p68);
        check_rel(g, tn + "_" + alloc + " overload endpoint m1", rc2.g95_GeV.front(), rc.g95_GeV.front(), 1e-14);
        check_rel(g, tn + "_" + alloc + " overload endpoint m2", rc2.g95_GeV.back(), rc.g95_GeV.back(), 1e-14);
      }
    }
    // target reach / DFSZ = (F_req/F)^(1/4)  (spec Sec. 5.4)
    const Allocation a = allocation_target(toyA, m1, m2, t3, gdfsz, p68);
    check_rel(g, "toyA target/DFSZ = (F_req/F)^(1/4)", g95_local(toyA, 1e-9, a.tau(omega_from_mass(1e-9)), p68) / gdfsz(1e-9),
              std::pow(F_exact["optimistic"] / toyA.F_nat(), 0.25), 1e-9);
  }

  // ------------------------------------------------------------ discrete exact Asimov sums (S5.9-S5.10)
  {
    const std::string g = "discrete";
    const json& D = R.at("discrete_scan_check");
    const std::vector<std::pair<std::string, double>> ms = {
        {"0.50005", m1 * 1.0001}, {"1", 1e-9}, {"3", 3e-9}, {"4.0996", m2 * 0.9999}};
    for (auto& [tn, det] : toys) {
      const Allocation a = allocation_target(det, m1, m2, t3, gdfsz, p68);
      for (auto& [ml, m] : ms) {
        const json& e = D.at(tn + "_m" + ml + "neV");
        const std::string k = tn + "_m" + ml + " ";
        const Tunings tun = dense_tunings_around(det, m, a, 40.0, 6.0);
        CHECK(g, static_cast<int>(tun.t_s.size()) == e.at("N_tunings").get<int>());
        std::vector<double> ts = tun.t_s;
        std::nth_element(ts.begin(), ts.begin() + ts.size() / 2, ts.end());
        check_rel(g, k + "median dwell", ts[ts.size() / 2], e.at("dwell_per_tuning_s").get<double>(), 1e-10);
        const double w = omega_from_mass(m);
        check_rel(g, k + "ring-up", ringup_time_s(w, det.Q), e.at("ringup_2Q_over_omega0_s").get<double>(), 1e-14);
        const double eqw = sensitivity_bandwidths(w, det.Q, n_bose(w, det.T_K), det.eta_A).equivalent;
        check_rel(g, k + "time within equiv. width", a.tau(w) * eqw, e.at("total_time_within_equiv_width_s").get<double>(),
                  1e-10);
        const double g_loc_true = g95_local(det, m, a.tau(w), pQa);
        const double g_loc_68 = g95_local(det, m, a.tau(w), p68);
        check_rel(g, k + "g95 local (Q_a true)", g_loc_true, e.at("g95_local_Qa_true").get<double>(), 1e-9);
        check_rel(g, k + "g95 local (Q_a 6.8e5)", g_loc_68, e.at("g95_local_Qa_6.8e5").get<double>(), 1e-9);
        double smax = 0;
        const double g_w = g95_discrete(det, m, tun, Stat::Weak, p68, 256, &smax);
        const double g_l = g95_discrete(det, m, tun, Stat::Limit, p68);
        check_rel(g, k + "g95 discrete weak", g_w, e.at("g95_discrete_weak").get<double>(), 1e-8);
        check_rel(g, k + "g95 discrete exact Asimov", g_l, e.at("g95_discrete_exactAsimov").get<double>(), 1e-8);
        check_abs(g, k + "discrete/local - 1", g_w / g_loc_true - 1,
                  e.at("discrete_over_local(Qa_true)_minus1").get<double>(), 1e-8);
        check_abs(g, k + "exact/weak - 1", g_l / g_w - 1, e.at("exact_over_weak_minus1").get<double>(), 1e-8);
        check_rel(g, k + "max s at g95", smax, e.at("max_s_per_bin_at_g95").get<double>(), 1e-7);
        // consistency: Theta(g95) = Theta95, per-tuning sum, reach_discrete
        check_rel(g, k + "Theta_limit(g95_limit) = 2.71", theta_discrete(det, m, g_l, tun, Stat::Limit, p68), 2.71, 1e-10);
        const std::vector<double> th = theta_per_tuning(det, m, g_w, tun, Stat::Weak, p68);
        double sum = 0;
        for (double x : th) sum += x;
        check_rel(g, k + "sum Theta_i(g95_weak) = 2.71", sum, 2.71, 1e-10);
        if (ml == "1") {
          const ReachCurve rd = reach_discrete(det, tun, {m}, Stat::Limit, p68, 41.0);
          check_rel(g, k + "reach_discrete = g95_discrete", rd.g95_GeV[0], g_l, 1e-12);
          const double g_d = g95_discrete(det, m, tun, Stat::Discovery, p68);
          CHECK(g, g_d > g_w && g_d < g_l);  // h_lim <= h_disc <= h_weak
        }
      }
    }
  }

  // ------------------------------------------------------------ statistics (S4.2-S4.7)
  {
    const std::string g = "statistics";
    const json& s = R.at("statistics");
    check_rel(g, "Theta95 exact", theta95_exact(), s.at("Theta95_exact_(Phi^-1(0.95))^2").get<double>(), 1e-14);
    for (int k : {1, 2}) {
      const auto q = limit_band_q(k);
      check_rel(g, "q band +" + std::to_string(k), q.first, s.at("q_band_+" + std::to_string(k) + "sigma").get<double>(), 1e-14);
      check_rel(g, "q band -" + std::to_string(k), q.second, s.at("q_band_-" + std::to_string(k) + "sigma").get<double>(),
                1e-13);
    }
    const double f1 = mass_to_freq(m1), f2 = mass_to_freq(m2);
    const double Nma = n_independent_masses(f1, f2);
    check_rel(g, "N_ma scan", Nma, s.at("N_ma_scan").get<double>(), 1e-13);
    const double Nf = n_independent_masses(100, 1e8);
    check_rel(g, "N_ma Foster example", Nf, s.at("Foster_example_N_ma_100Hz-100MHz").get<double>(), 1e-13);
    // tolerance from the reference's rounding of 1 - p/N (see header comment)
    auto ts_check = [&](const std::string& key, double sigma, double N, int sided) {
      const double TS = ts_discovery_threshold(sigma, N, sided);
      const double q = sided * norm_cdf(-sigma) / N;
      const double tol = 2.0 * 1.2e-16 / (q * TS) + 1e-12;
      check_rel(g, key + " (tol from ref. rounding)", TS, s.at(key).get<double>(), tol);
      // reproduce the reference evaluation path exactly: [Phi^-1(1 - p/N)]^2
      const double z = norm_ppf(1.0 - q);
      check_rel(g, key + " (reference path)", z * z, s.at(key).get<double>(), 1e-12);
    };
    ts_check("TS_thresh_3sigma_global", 3, Nma, 1);
    ts_check("TS_thresh_5sigma_global", 5, Nma, 1);
    ts_check("TS_thresh_5sigma_local(N=1)", 5, 1, 1);
    ts_check("Foster_example_TS3sigma_one_sided", 3, Nf, 1);
    ts_check("Foster_example_TS5sigma_one_sided", 5, Nf, 1);
    ts_check("Foster_example_TS3sigma_two_sided", 3, Nf, 2);
    ts_check("Foster_example_TS5sigma_two_sided", 5, Nf, 2);
    ts_check("TS_thresh_3sigma_global_two_sided", 3, Nma, 2);
    ts_check("TS_thresh_5sigma_global_two_sided", 5, Nma, 2);
    check_rel(g, "TS 3 sigma local = 9", ts_discovery_threshold(3, 1), 9.0, 1e-13);
    check_rel(g, "TS 5 sigma local = 25", ts_discovery_threshold(5, 1), 25.0, 1e-13);
    check_rel(g, "g_disc/g95 5 sigma", g_disc_over_g95_weak(s.at("TS_thresh_5sigma_global").get<double>()),
              s.at("g_disc5sigma_over_g95").get<double>(), 1e-14);
    check_rel(g, "g_disc/g95 3 sigma", g_disc_over_g95_weak(s.at("TS_thresh_3sigma_global").get<double>()),
              s.at("g_disc3sigma_over_g95").get<double>(), 1e-14);
    check_rel(g, "norm_ppf(norm_cdf(-7.3))", norm_ppf(norm_cdf(-7.3)), -7.3, 1e-14);
    check_rel(g, "norm_ppf(0.975)", norm_ppf(0.975), 1.959963984540054, 1e-14);
    // h functions (S4.2-S4.4): series vs closed form across the switch, and known series
    // small s: compare with the closed forms in long double (cancellation ~1e-16/s relative)
    for (long double s1 : {1e-3L, 1e-2L, 4e-2L}) {
      const double hl = static_cast<double>(2 * (std::log1p(s1) - s1 / (1 + s1)));
      const double hd = static_cast<double>(2 * (s1 - std::log1p(s1)));
      check_rel(g, "h_limit small s vs long-double closed form", h_limit(static_cast<double>(s1)), hl, 1e-14);
      check_rel(g, "h_discovery small s vs long-double closed form", h_discovery(static_cast<double>(s1)), hd, 1e-14);
    }
    check_rel(g, "h_limit(1e-9) = s^2 (1 - 4s/3)", h_limit(1e-9), 1e-18 * (1 - 4e-9 / 3), 1e-15);
    for (double x : {0.0499999, 0.0500001}) {
      check_rel(g, "h_limit continuity at 0.05", h_limit(x), 2 * (std::log1p(x) - x / (1 + x)), 1e-13);
      check_rel(g, "h_discovery continuity at 0.05", h_discovery(x), 2 * (x - std::log1p(x)), 1e-13);
    }
    check_rel(g, "h_limit(1)", h_limit(1.0), 2 * (std::log(2.0) - 0.5), 1e-15);
  }

  // ------------------------------------------------------------ PSDs (S3.2-S3.6) self tests
  {
    const std::string g = "psd";
    // (1/pi) int S_{V,a} domega = (omega_a Phi)^2/2 <(omega/omega_a)^4>  (S3.2 check)
    std::vector<double> v, w;
    line_v_grid(400, 220, 232, v, w);
    const double wa = 2 * U::pi * 1e5, Phi = 1e-20;
    double I = 0, m4 = 0;
    for (size_t k = 0; k < v.size(); ++k) {
      const double om = wa * (1 + 0.5 * v[k] * v[k]);
      I += signal_psd_V(om, wa, Phi) * wa * v[k] * w[k];
      m4 += f_shm(v[k]) * std::pow(om / wa, 4) * w[k];
    }
    check_rel(g, "signal PSD normalisation", I / U::pi, 0.5 * wa * wa * Phi * Phi * m4, 1e-8);
    // <(1+v^2/2)^4> - 1 = 2<v^2> + (3/2)<v^4> + O(<v^6>~1e-17), shifted Maxwellian with
    // sigma^2 = v0^2/2: <v^2> = vo^2 + 3 sigma^2, <v^4> = 15 sigma^4 + vo^4 + 10 sigma^2 vo^2.
    const double v0c = 220e3 / U::c, voc = 232e3 / U::c, s2 = 0.5 * v0c * v0c;
    const double mv2 = voc * voc + 3 * s2, mv4 = 15 * s2 * s2 + std::pow(voc, 4) + 10 * s2 * voc * voc;
    check_abs(g, "<(omega/m)^4> - 1 vs analytic moments", m4 - 1, 2 * mv2 + 1.5 * mv4, 5e-15);
    // The reference self-test value carries a ~7e-12 absolute error (it forms omega/m - 1 at m = 1).
    check_abs(g, "<(omega/m)^4> - 1 vs reference (ref. cancellation)", m4 - 1,
              R.at("self_tests").at("signal_psd_omega4_correction").get<double>(), 2e-11);
    // noise components: lambda_B |Z|^2 = N_V; on resonance N_V = (hbar w0 R0/2xi) A with A at n(omega0)  (S3.7)
    const Detector d = toyB;
    const double w0 = omega_from_mass(1e-9);
    const double n0 = n_bose(w0, d.T_K), xi = xi_opt(n0, d.eta_A);
    const NoisePSD np = noise_psd(w0, w0, d, xi);
    check_rel(g, "lambda_B Z2 = N_V", np.lambda_B * np.Z2, np.N_V, 1e-14);
    const double R0 = w0 * d.L_H() / d.Q;
    check_rel(g, "N_V(omega0) = hbar w0 R0 A/(2 xi)  (S3.7)", np.N_V, U::hbar * w0 * R0 * A_noise(xi, n0, d.eta_A) / (2 * xi),
              1e-13);
    // detuned: N_V = (hbar w0 R0/2xi)(A + xi^2 x^2), x = 2Q dw/w0, to O(dw/w0)
    const double dw = 3.0 * w0 / d.Q;
    const NoisePSD nd = noise_psd(w0 + dw, w0, d, xi);
    const double x = 2 * d.Q * dw / w0;
    check_rel(g, "N_V detuned vs local form (S3.7), O(1e-5)", nd.N_V,
              U::hbar * w0 * R0 * (A_noise(xi, n0, d.eta_A) + xi * xi * x * x) / (2 * xi), 1e-4);
    // SQUID coupling (S3.5): S_Phi/M^2 and M^2 S_J reproduce SI_imp and the backaction
    const double S_Phi = 1e-40, S_J = std::pow(d.eta_A * U::hbar / 2, 2) / S_Phi;
    const SquidCoupling sc = squid_coupling_from_xi(xi, w0, R0, S_Phi, S_J);
    check_rel(g, "eta_A from S_Phi S_J", sc.eta_A, d.eta_A, 1e-14);
    check_rel(g, "S_Phi/M^2 = SI_imp", S_Phi / (sc.M_H * sc.M_H), np.SI_imp, 1e-13);
    check_rel(g, "w0^2 M^2 S_J = SV_ba", w0 * w0 * sc.M_H * sc.M_H * S_J, np.SV_ba, 1e-13);
    // psd_at_tuning: consistent with the pointwise functions
    const double fa = mass_to_freq(1e-9);
    std::vector<double> f;
    for (int i = 0; i < 50; ++i) f.push_back(fa * (1 + 2e-6 * i / 49.0));
    const PSDCurve pc = psd_at_tuning(d, fa, 1e-9, 1e-19, f);
    const NoisePSD n7 = noise_psd(2 * U::pi * f[7], 2 * U::pi * fa, d, pc.xi);
    check_rel(g, "psd_at_tuning N_V", pc.N_V_V2_s[7], n7.N_V, 1e-8);
    check_rel(g, "psd_at_tuning S_Va", pc.S_Va_V2_s[7],
              signal_psd_V(2 * U::pi * f[7], omega_from_mass(1e-9), flux_amplitude_Wb(1e-19, d.psi_nat_T_m3)), 1e-8);
    check_rel(g, "psd_at_tuning s = lambda_a/lambda_B", pc.s[7], pc.lambda_a_A2_s[7] / pc.lambda_B_A2_s[7], 1e-13);
    CHECK(g, pc.S_Va_V2_s[0] == 0.0 && pc.S_Va_V2_s[1] > 0.0);  // line edge at omega = omega_a
    // Lorentzian identity used in (S5.1): int dx/(A + xi^2 x^2)^2 = pi/(2 xi A^{3/2}), x = tan(t)
    const double A = 7.3, xl = 0.37;
    const double Il = detail::integrate(
        [&](double t) {
          const double xx = std::tan(t), c2 = std::cos(t) * std::cos(t);
          return 1.0 / ((A + xl * xl * xx * xx) * (A + xl * xl * xx * xx) * c2);
        },
        -U::pi / 2, U::pi / 2, 1e-13);
    check_rel(g, "Lorentzian identity (S5.1)", Il, U::pi / (2 * xl * std::pow(A, 1.5)), 1e-12);
  }

  // ------------------------------------------------------------ finite band (Sec. 6)
  {
    const std::string g = "finite_band";
    const json& F = R.at("finite_band_demo");
    const Detector detC = Detector::from_natural(100.0, 1.0, 1e4, 0.030, 10.0);
    const double fa = 120.9e3, fb = 400e3;
    const double ma = freq_to_mass(fa), mb = freq_to_mass(fb);
    check_rel(g, "t_local_formula (Q_a true)", time_target_curve_s(detC, ma, mb, gdfsz, pQa),
              F.at("t_local_formula_s").get<double>(), 1e-10);
    const double fw_a = sensitivity_bandwidths(1.0, detC.Q, n_bose(omega_from_mass(ma), detC.T_K), detC.eta_A).noise_doubling;
    check_rel(g, "fw/f0 at fa", fw_a, F.at("fw_over_f0_at_fa").get<double>(), 1e-13);
    // settings: 200 log-spaced omega0 x xi_opt*(0.3, 1, 3)
    const std::vector<double> m_t = geomspace(ma, mb, 120), w_set = geomspace(0.5 * ma, 1.6 * mb, 200);
    std::vector<double> w_all, xi_all;
    for (double fac : {0.3, 1.0, 3.0})
      for (double m : w_set) {
        const double w = omega_from_mass(m);
        w_all.push_back(w);
        xi_all.push_back(fac * xi_opt(n_bose(w, detC.T_K), detC.eta_A));
      }
    std::vector<std::vector<double>> K;
    for (double m : m_t) K.push_back(kernel_row(detC, m, gdfsz(m), w_all, xi_all, Stat::Weak, p68, 64));
    // narrow-line kernel (S6.3) vs full kernel
    double dev = 0;
    for (size_t k = 0; k < m_t.size(); k += 20) {
      const double kmax = *std::max_element(K[k].begin(), K[k].end());
      for (size_t j = 0; j < w_all.size(); ++j) {
        if (K[k][j] <= 1e-3 * kmax) continue;
        dev = std::max(dev, std::abs(kernel_narrow_line(detC, m_t[k], gdfsz(m_t[k]), w_all[j], xi_all[j], pQa) / K[k][j] - 1));
      }
    }
    check_rel(g, "narrow-line/full kernel max |dev|", dev, F.at("narrowline_over_fullline_kernel_max_abs_dev").get<double>(),
              1e-6);
    // local (Lorentzian-derived) schedule evaluated with the exact finite-band kernel
    const std::vector<double> m_loc = geomspace(ma, mb, 801), m_chk = geomspace(ma, mb, 481);
    std::vector<double> w_loc, xi_loc, t_loc;
    for (double m : m_loc) {
      const double w = omega_from_mass(m);
      w_loc.push_back(w);
      xi_loc.push_back(xi_opt(n_bose(w, detC.T_K), detC.eta_A));
    }
    const size_t nl = w_loc.size();
    double tsum = 0;
    for (size_t j = 0; j < nl; ++j) {
      const double dw = j == 0 ? w_loc[1] - w_loc[0]
                               : (j == nl - 1 ? w_loc[nl - 1] - w_loc[nl - 2] : 0.5 * (w_loc[j + 1] - w_loc[j - 1]));
      t_loc.push_back(dw / scan_rate(detC, m_loc[j], gdfsz(m_loc[j]), pQa));
      tsum += t_loc.back();
    }
    check_rel(g, "local schedule time", tsum, F.at("local_schedule_time_s(Qa_true)").get<double>(), 1e-10);
    double cmin = 1e300, cimin = 1e300, cimax = 0;
    for (double m : m_chk) {
      const std::vector<double> kr = kernel_row(detC, m, gdfsz(m), w_loc, xi_loc, Stat::Weak, p68, 64);
      double c = 0;
      for (size_t j = 0; j < nl; ++j) c += kr[j] * t_loc[j];
      c /= 2.71;
      cmin = std::min(cmin, c);
      if (m > ma * (1 + fw_a) && m < mb * (1 - fw_a)) {
        cimin = std::min(cimin, c);
        cimax = std::max(cimax, c);
      }
    }
    check_rel(g, "local schedule coverage min (all)", cmin, F.at("local_schedule_coverage_min_all").get<double>(), 1e-9);
    check_rel(g, "local schedule coverage min (interior)", cimin, F.at("local_schedule_coverage_interior_min").get<double>(), 1e-9);
    check_rel(g, "local schedule coverage max (interior)", cimax, F.at("local_schedule_coverage_interior_max").get<double>(), 1e-9);
    // LP schedule (S6.4)
    const LPSchedule lp = lp_schedule(K, 2.71);
    check_rel(g, "LP total time", lp.t_total_s, F.at("t_LP_finite_band_s").get<double>(), 1e-6);
    check_rel(g, "LP / local time", lp.t_total_s / time_target_curve_s(detC, ma, mb, gdfsz, pQa),
              F.at("LP_over_local").get<double>(), 1e-6);
    CHECK(g, lp.min_coverage >= 1.0 - 1e-12);
    int used = 0;
    for (double t : lp.t_s) used += t > 1e-9 * lp.t_total_s;
    double cdense = 1e300;
    for (double m : m_chk) {
      const std::vector<double> kr = kernel_row(detC, m, gdfsz(m), w_all, xi_all, Stat::Weak, p68, 64);
      double c = 0;
      for (size_t j = 0; j < kr.size(); ++j) c += kr[j] * lp.t_s[j];
      cdense = std::min(cdense, c / 2.71);
    }
    // the LP optimum is unique in value, not necessarily in t: loose checks on solution details
    check_rel(g, "LP min coverage on dense grid (loose)", cdense, F.at("min_coverage_dense_grid_(Theta/Theta95)").get<double>(),
              5e-3);
    std::printf("  finite band LP: %d pivots, %d settings used (reference %d), dense-grid coverage %.6f (ref %.6f)\n",
                lp.pivots, used, F.at("n_settings_used").get<int>(), cdense,
                F.at("min_coverage_dense_grid_(Theta/Theta95)").get<double>());
  }

  // ------------------------------------- finite band with an EXTERNAL N_V(omega)  (C4.5)
  {
    const std::string g = "exact_noise";
    const Detector det = Detector::from_natural(100.0, 1.0, 1e4, 0.030, 10.0);
    const double f0 = 120.9e3, w0 = 2 * U::pi * f0;
    const double m = freq_to_mass(f0), gd = gdfsz(m);
    const double xi = xi_opt(n_bose(w0, det.T_K), det.eta_A);

    // (1) NoiseCurve.at is exact on its nodes and for a pure power law between them
    NoiseCurve pw;
    for (double w : geomspace(0.5 * w0, 2.0 * w0, 7)) {
      pw.omega_rad_s.push_back(w);
      pw.N_V_V2_s.push_back(1e-30 * std::pow(w / w0, 3.0));
    }
    double dev = 0;
    for (double w : geomspace(0.5 * w0, 2.0 * w0, 101))
      dev = std::max(dev, std::abs(pw.at(w) / (1e-30 * std::pow(w / w0, 3.0)) - 1.0));
    check_rel(g, "NoiseCurve log-log interpolation of a power law", 1.0 + dev, 1.0, 1e-13);

    // (2) fed the single pole's OWN N_V(omega), (S6.1) with an external N_V must reproduce
    //     kernel_row; the only difference is the interpolation of the table
    NoiseCurve nc;
    for (double w : geomspace(w0 * (1 - 3e-3), w0 * (1 + 3e-3), 4001)) {
      nc.omega_rad_s.push_back(w);
      nc.N_V_V2_s.push_back(noise_psd(w, w0, det, xi).N_V);
    }
    const double K_det = kernel_row(det, m, gd, {w0}, {xi}, Stat::Weak, pQa, 96)[0];
    const double K_ext = kernel_row_noise(det.psi_SI(), m, gd, {nc}, Stat::Weak, pQa, 96)[0];
    check_rel(g, "kernel_row_noise vs kernel_row (same N_V)", K_ext, K_det, 1e-8);
    const double nv_a = nc.at(omega_from_mass(m));
    check_rel(g, "kernel_narrow_line_noise vs kernel_narrow_line",
              kernel_narrow_line_noise(det.psi_SI(), m, gd, nv_a, pQa),
              kernel_narrow_line(det, m, gd, w0, xi, pQa), 1e-13);

    // (3) K ~ Phi_a^4 ~ (g Psi)^4   (S6.2 with S3.2)
    check_rel(g, "K scales as g^4", kernel_row_noise(det.psi_SI(), m, 2 * gd, {nc}, Stat::Weak, pQa, 96)[0],
              16.0 * K_ext, 1e-12);
    check_rel(g, "K scales as Psi^4", kernel_row_noise(3 * det.psi_SI(), m, gd, {nc}, Stat::Weak, pQa, 96)[0],
              81.0 * K_ext, 1e-12);

    // (4) g95_band inverts the schedule exactly in the weak-signal limit, and reach_band
    //     reproduces g95_discrete for the same single setting
    const double t_need = pQa.Theta / K_ext;
    check_rel(g, "g95_band inverts sum t K = Theta",
              g95_band(det.psi_SI(), m, {nc}, {t_need}, Stat::Weak, pQa, 96), gd, 1e-9);
    check_rel(g, "g95_band vs g95_discrete (same noise)",
              g95_band(det.psi_SI(), m, {nc}, {t_need}, Stat::Limit, pQa, 96),
              g95_discrete(det, m, Tunings{{w0}, {t_need}, {xi}}, Stat::Limit, pQa, 96), 1e-7);
    const ReachCurve rb = reach_band({det.psi_SI()}, {m}, {nc}, {t_need}, Stat::Weak, pQa, 96);
    check_rel(g, "reach_band g95", rb.g95_GeV[0], gd, 1e-9);
    CHECK(g, rb.band.n_settings == 1 && rb.band.allocation == "lp" && rb.band.n_targets == 1);
    check_rel(g, "reach_band band live time", rb.band.live_time_s, t_need, 1e-15);

    // (5) the ScanBand echo of a local reach describes the call
    const double m1 = freq_to_mass(120.9e3), m2 = freq_to_mass(991.4e3);
    const ReachCurve rl = reach_local(det, m1, m2, 3 * 365.25 * 86400.0, Allocation::Kind::Target, 11,
                                      gdfsz, pQa);
    CHECK(g, rl.band.allocation == "target" && rl.band.n_targets == 11 && rl.band.stat == "weak");
    check_rel(g, "band m1", rl.band.m1_eV, m1, 1e-15);
    check_rel(g, "band f_max", rl.band.f_max_Hz, 991.4e3, 1e-12);
    check_rel(g, "band live time", rl.band.live_time_s, 3 * 365.25 * 86400.0, 1e-15);
    check_rel(g, "band F_SI", rl.band.F_SI_T4_H2_m8, det.F_SI(), 1e-15);
    check_rel(g, "band Q_axion", rl.band.Q_axion, pQa.Q_axion, 1e-15);
    const ReachCurve rd = reach_discrete(det, Tunings{{w0}, {t_need}, {xi}}, {m}, Stat::Limit, pQa);
    CHECK(g, rd.band.allocation == "discrete" && rd.band.n_settings == 1 && rd.band.stat == "limit");

    // (6) bad input is rejected with a specific message
    bool threw = false;
    try {
      NoiseCurve bad;
      bad.omega_rad_s = {2.0, 1.0};
      bad.N_V_V2_s = {1e-30, 1e-30};
      kernel_row_noise(det.psi_SI(), m, gd, {bad}, Stat::Weak, pQa, 96);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(g, threw);
  }

  // ------------------------------------------------------------ summary
  std::printf("\n%-16s %6s %6s %12s %20s\n", "group", "checks", "fail", "max err", "tol range");
  for (const auto& name : order) {
    const Group& gr = groups[name];
    std::printf("%-16s %6d %6d %12.3e %9.1e..%-9.1e\n", name.c_str(), gr.n, gr.fail, gr.max_err, gr.tol_min, gr.tol_max);
  }
  std::printf("\n%d / %d checks passed\n", n_total - n_fail, n_total);
  return n_fail == 0 ? 0 : 1;
}
