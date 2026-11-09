// Unit test of the surface-resistance models (cpp/src/surface_resistance.cpp; docs/physics/surface_resistance.md).
// Reference values: independent scipy evaluations (adaptive quad of the RS integrals; numerical solution of the RS
// half-space integral equation for general p: benchmarks/surface_resistance_specularity.py), and the numbers of an
// independent blind derivation of the same formulas.
#include <cmath>
#include <cstdio>
#include <string>

#include "axionfem/spec.hpp"
#include "axionfem/surface_resistance.hpp"
#include "axionfem/units.hpp"

namespace {
int n_fail = 0, n_ok = 0;
void check(const std::string& what, double got, double want, double rtol) {
  const double err = std::fabs(got / want - 1);
  const bool ok = err <= rtol;
  (ok ? n_ok : n_fail)++;
  std::printf("%s %-60s got %.10e want %.10e rel %.2e (tol %.1e)\n", ok ? "ok  " : "FAIL", what.c_str(), got, want,
              err, rtol);
}
}  // namespace

int main() {
  using namespace axf::rs;
  constexpr double pi = axf::units::pi, mu0 = axf::units::mu0;
  const double rhol = 6.6e-16;  // copper rho * l [ohm m^2]

  // ---- classical limit (l << delta): RS -> sqrt(pi f mu0 rho), both p
  {
    const double f = 1e9, rho = 1.68e-8, rl = rho * 1e-12;  // l = 1 pm: alpha ~ 1e-10
    check("classical limit, diffuse", anomalous(f, rho, rl, 0.0), skin(f, rho), 1e-6);
    check("classical limit, specular", anomalous(f, rho, rl, 1.0), skin(f, rho), 1e-6);
    check("classical limit, p = 0.4", anomalous(f, rho, rl, 0.4), skin(f, rho), 1e-6);
    check("classical: X = R (diffuse)", anomalous_Z_diffuse(f, rho, rl).imag(), skin(f, rho), 1e-6);
  }
  // ---- extreme anomalous limit: R -> (sqrt3/(16 pi) rho_l (mu0 w)^2)^(1/3) (diffuse), 8/9 of it (specular)
  {
    const double f = 1e9, w = 2 * pi * f;
    const double Rinf = std::cbrt(std::sqrt(3.0) / (16 * pi) * rhol * (mu0 * w) * (mu0 * w));
    check("R_inf closed form (diffuse) = 1.1234 mOhm at 1 GHz", anomalous_extreme(f, rhol, 0.0), Rinf, 1e-14);
    check("R_inf at 1 GHz (independent derivation 1.123 mOhm)", Rinf, 1.123e-3, 5e-4);
    check("R_inf(specular) = 8/9 R_inf(diffuse)", anomalous_extreme(f, rhol, 1.0), 8.0 / 9.0 * Rinf, 1e-14);
    for (double alpha : {1e12, 1e15}) {
      // rho from alpha = 3/4 mu0 w l^2 / rho at fixed rho_l: rho^3 = 3/4 mu0 w rho_l^2 / alpha
      const double rho = std::cbrt(0.75 * mu0 * w * rhol * rhol / alpha);
      check("alpha check", anomalous_alpha(f, rho, rhol), alpha, 1e-12);
      const double fit = 1 + 1.157 * std::pow(alpha, -0.2757);  // Chambers' diffuse fit, error < 1e-3 here
      const auto Zd = anomalous_Z_diffuse(f, rho, rhol);
      check("extreme limit diffuse vs R_inf (1 + 1.157 alpha^-0.2757), alpha " + std::to_string(alpha), Zd.real(),
            Rinf * fit, 2e-4);
      check("extreme limit X/R -> sqrt 3, alpha " + std::to_string(alpha), Zd.imag() / Zd.real(), std::sqrt(3.0),
            alpha > 1e13 ? 1e-3 : 3e-3);
      check("extreme limit specular / diffuse -> 8/9, alpha " + std::to_string(alpha),
            anomalous(f, rho, rhol, 1.0) / Zd.real(), 8.0 / 9.0, alpha > 1e13 ? 5e-4 : 2e-3);
    }
    // Chambers' fit across alpha = 3 .. 1e9 (its stated range), diffuse: agreement within 2e-3
    for (double alpha : {3.0, 10.0, 100.0, 1e3, 1e4, 1e6, 1e9}) {
      const double rho = std::cbrt(0.75 * mu0 * w * rhol * rhol / alpha);
      check("Chambers fit alpha " + std::to_string(alpha), anomalous(f, rho, rhol, 0.0),
            Rinf * (1 + 1.157 * std::pow(alpha, -0.2757)), 2e-3);
    }
  }
  // ---- copper RRR 100 (rho = 1.68e-10 ohm m), diffuse: scipy quad = independent derivation (1.353, 3.717, 5.781 mOhm)
  check("Cu RRR100 1 GHz", anomalous(1e9, 1.68e-10, rhol, 0.0), 1.353456459e-3, 1e-8);
  check("Cu RRR100 5 GHz", anomalous(5e9, 1.68e-10, rhol, 0.0), 3.716853468e-3, 1e-8);
  check("Cu RRR100 10 GHz", anomalous(10e9, 1.68e-10, rhol, 0.0), 5.780878115e-3, 1e-8);
  check("Cu RRR100 TM010 (1.147425278 GHz)", anomalous(1.147425278e9, 1.68e-10, rhol, 0.0), 1.474048838e-3, 1e-8);
  check("Cu ratio 4 K / 300 K at 1.5 GHz (independent derivation 0.1746)",
        anomalous(1.5e9, 1.68e-10, rhol, 0.0) / skin(1.5e9, 1.68e-8), 0.17462148808, 1e-8);
  // relaxation (v_F = 1.57e6 m/s): scipy trapezoid of the complex-l integral, -0.8867 % at 10 GHz, RRR 100
  check("relaxation omega tau = 0.157 (RRR 100, 10 GHz)",
        anomalous(10e9, 1.68e-10, rhol, 0.0, 1.57e6) / anomalous(10e9, 1.68e-10, rhol, 0.0), 1 - 8.867e-3, 2e-5);
  // ---- general specularity: numerical half-space solution at off-grid (alpha, p)
  {
    struct Ref { double alpha, p, R; };  // R / (omega mu0 l)
    for (Ref r : {Ref{544.0, 0.35, 4.07279629e-2}, Ref{3.0e6, 0.75, 1.88289903e-3}, Ref{20.0, 0.5, 1.52535492e-1}}) {
      const double f = 1e9, w = 2 * pi * f;
      const double rho = std::cbrt(0.75 * mu0 * w * rhol * rhol / r.alpha), ell = rhol / rho;
      check("general p (RS integral equation), alpha " + std::to_string(r.alpha) + " p " + std::to_string(r.p),
            anomalous(f, rho, rhol, r.p) / (w * mu0 * ell), r.R, 5e-4);
    }
    check("w(alpha, 0) = 1", specularity_weight(100.0, 0.0), 1.0, 0.0);
  }
  // ---- BCS, Nb (A = 2e-4 ohm K at 1.5 GHz, Delta/kTc = 17.67/9.25): independent derivation 10.93 nOhm (2 K),
  // 532.5 nOhm (4.2 K)
  check("BCS Nb 1.3 GHz 2 K", bcs(1.3e9, 2.0, 2e-4, 1.5e9, 17.67 / 9.25, 9.25), 1.0932322604e-8, 1e-9);
  check("BCS Nb 1.3 GHz 4.2 K", bcs(1.3e9, 4.2, 2e-4, 1.5e9, 17.67 / 9.25, 9.25), 5.325323121e-7, 1e-9);
  // BCS log factor: L(f, T)/L(f_ref, T), L = ln(2.246 kT/hf); independent check: 0.61 (4.2 K), 0.54 (2 K) at 10 GHz
  check("BCS log ratio 10 GHz / 1.5 GHz, 4.2 K", bcs_log_ratio(10e9, 1.5e9, 4.2), 0.610885674739, 1e-11);
  check("BCS log ratio 10 GHz / 1.5 GHz, 2 K", bcs_log_ratio(10e9, 1.5e9, 2.0), 0.541042850697, 1e-11);
  check("BCS log ratio at f_ref", bcs_log_ratio(1.5e9, 1.5e9, 2.0), 1.0, 1e-15);
  check("BCS with log factor", bcs(10e9, 2.0, 2e-4, 1.5e9, 17.67 / 9.25, 9.25, true),
        bcs(10e9, 2.0, 2e-4, 1.5e9, 17.67 / 9.25, 9.25) * 0.541042850697, 1e-11);
  // ---- RS crossover: exact p = 0 and p = 1 integrals against an independent scipy quad (piecewise in ln t, epsrel
  // 1e-13; an independent check's alpha = 1 values 0.63573323764 / 0.59983679081), R / (omega mu0 l)
  {
    struct Q { double alpha, R0, R1; };
    for (Q r : {Q{0.01, 6.1285173000162, 6.1170719971131}, Q{0.1, 1.9488734080957, 1.9228794097869},
                Q{1.0, 0.63573323763879, 0.59983679081250}, Q{3.0, 0.38052672116493, 0.34724897870733},
                Q{10.0, 0.22174017704880, 0.19527786389721}}) {
      const double f = 1e9, w = 2 * pi * f;
      const double rho = std::cbrt(0.75 * mu0 * w * rhol * rhol / r.alpha), ell = rhol / rho;
      check("crossover diffuse (quad), alpha " + std::to_string(r.alpha), anomalous(f, rho, rhol, 0.0) / (w * mu0 * ell),
            r.R0, 1e-10);
      check("crossover specular (quad), alpha " + std::to_string(r.alpha),
            anomalous(f, rho, rhol, 1.0) / (w * mu0 * ell), r.R1, 1e-10);
    }
  }
  // ---- flux flow: Z = sqrt(i mu0 w (i mu0 w lambda^2 + s rho_v)); reference: independent finite-difference solution
  // of dE/dz = -i w mu0 H, dH/dz = -E/rho~ on a half-space (scipy solve_banded, Richardson, ~1e-8; also an
  // independent check: 0.031 ohm Nb3Sn, 0.092 ohm NbTi at 5 GHz, 8 T, free flux flow)
  check("flux flow Nb3Sn 5 GHz 8 T (vortex dominated)", flux_flow(5e9, 1.5e-7, 110e-9, 8, 25, 0, 0, 1),
        3.062841394577e-02, 1e-7);
  check("flux flow NbTi 5 GHz 8 T (vortex dominated)", flux_flow(5e9, 6e-7, 300e-9, 8, 11, 0, 0, 1), 9.243172571572e-02,
        1e-7);
  check("flux flow, pinned, |rho_v| ~ 50 mu0 w lambda^2", flux_flow(3e9, 1.5e-7, 1.1e-7, 1, 10, 1e9, 0, 1),
        1.062779146028e-02, 1e-7);
  check("flux flow with creep eps 0.3", flux_flow(3e9, 1.5e-7, 1.1e-7, 1, 10, 5e9, 0.3, 1), 6.767133876567e-03, 1e-7);
  check("flux flow, |rho_v| ~ 0.08 mu0 w lambda^2", flux_flow(1e9, 1e-9, 40e-9, 0.1, 10, 1e10, 0, 1),
        1.191793960833e-06, 1e-7);
  {  // limits: small vortex resistivity -> Re rho_v / (2 lambda); vortex dominated -> sqrt(mu0 w rho_ff / 2)
    const double f = 2e9, w = 2 * pi * f, lam = 5e-8, rn = 1e-15, B = 1, Bc2 = 10, fp = 4e9;
    const double rff = rn * B / Bc2, rv = rff * f * f / (f * f + fp * fp);  // Re rho_v (GR)
    check("small |rho_v| (1.1e-6 mu0 w lambda^2): Re rho_v / (2 lambda)", flux_flow(f, rn, lam, B, Bc2, fp, 0, 1),
          rv / (2 * lam), 1e-6);
    const double rn2 = 1e-6, lam2 = 1e-10;  // free flux flow, rho_ff = 1e-7 >> mu0 w lambda^2 = 1.6e-16
    check("vortex dominated: sqrt(mu0 w rho_ff / 2)", flux_flow(f, rn2, lam2, B, Bc2, 0, 0, 1),
          std::sqrt(mu0 * w * rn2 * B / Bc2 / 2), 1e-8);
    check("creep eps = 1: free flux flow for any f0", flux_flow(f, rn, lam, B, Bc2, fp, 1.0, 1),
          flux_flow(f, rn, lam, B, Bc2, 0, 0, 1), 1e-13);
    check("orientation s scales rho_v", flux_flow(f, rn, lam, B, Bc2, fp, 0, 0.25),
          vortex_Z(f, lam, 0.25 * vortex_resistivity(f, rn, B, Bc2, fp)).real(), 1e-15);
    const double z = flux_flow(f, rn, lam, B, Bc2, fp, 0, 0.0);
    (z == 0 ? n_ok : n_fail)++;
    std::printf("%s flux flow with orientation factor 0 is exactly 0 (got %g)\n", z == 0 ? "ok  " : "FAIL", z);
  }

  // ---- builtin materials: `copper` is the diffuse quasi-static RS model with rho = 1.7e-10 ohm m and
  // the copper rho*l; `copper_classical` is the classical skin (same rho)
  {
    const auto mats = axf::spec::builtin_materials();
    const auto &cu = mats.at("copper").rf, &cc = mats.at("copper_classical").rf;
    const double rho = 1.7e-10;
    (cu.model == "anomalous" && cc.model == "skin" ? n_ok : n_fail)++;
    std::printf("%s builtin copper model '%s', copper_classical '%s'\n",
                cu.model == "anomalous" && cc.model == "skin" ? "ok  " : "FAIL", cu.model.c_str(), cc.model.c_str());
    for (double f : {1.209e5, 1e6, 1e7, 1e8, 1.147e9, 1e10}) {
      check("builtin copper = anomalous(rho 1.7e-10, rho_l 6.6e-16, p 0) at f " + std::to_string(f), cu.Rs(f),
            anomalous(f, rho, rhol, 0.0, 0.0), 1e-15);
      check("builtin copper_classical = skin(rho 1.7e-10) at f " + std::to_string(f), cc.Rs(f), skin(f, rho), 1e-15);
    }
    // the size of the change (docs/physics/surface_resistance.md Sec. 1.1 table, exact RS integral by scipy quad)
    check("copper / copper_classical at 1 MHz (1/0.976)", cu.Rs(1e6) / cc.Rs(1e6), 1 / 0.976, 1e-3);
    check("copper / copper_classical at 100 MHz (1/0.770)", cu.Rs(1e8) / cc.Rs(1e8), 1 / 0.770, 1e-3);
    check("copper / copper_classical at 1 GHz (1/0.604)", cu.Rs(1e9) / cc.Rs(1e9), 1 / 0.604, 1e-3);
    // low frequency (alpha = 6e-4 at 1 kHz, 6e-6 at 10 Hz): reduces to the classical value
    check("copper -> classical at 1 kHz (alpha 6e-4)", cu.Rs(1e3), cc.Rs(1e3), 1e-3);
    check("copper -> classical at 10 Hz (alpha 6e-6)", cu.Rs(10.0), cc.Rs(10.0), 2e-5);
    const bool warn_cu = !cu.warnings(1e9).empty(), warn_cc = !cc.warnings(1e9).empty();
    (!warn_cu && warn_cc ? n_ok : n_fail)++;
    std::printf("%s skin_anomalous warning at 1 GHz: copper %d (want 0), copper_classical %d (want 1)\n",
                !warn_cu && warn_cc ? "ok  " : "FAIL", int(warn_cu), int(warn_cc));
  }

  std::printf("%d ok, %d failed\n", n_ok, n_fail);
  return n_fail ? 1 : 0;
}
