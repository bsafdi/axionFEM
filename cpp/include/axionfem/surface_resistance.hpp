// Surface-resistance models R_s(f) of RF walls (beta; formulas, regimes and sources:
// docs/physics/surface_resistance.md; the formulas and reference numbers were also checked by an independent blind
// derivation).
//
// Pure functions of the material parameters, shared by every caller of spec::RFLoss::Rs (the 2D driven sweep and
// cavity Q0 in model.cpp, and the 3D cavity Q0 in model3d_fullwave.cpp). SI units.
#pragma once

#include <complex>

namespace axf::rs {

// Classical (local) skin effect: R_s = sqrt(pi f mu0 rho).
double skin(double f_Hz, double rho_ohm_m);

// Reuter-Sondheimer anomalous skin effect, isotropic free-electron metal, quasi-static unless v_F > 0.
//   rho: dc resistivity at the operating temperature; rho_l: the material constant rho * l (mean free path l);
//   p: specularity of the surface (0 diffuse, 1 specular); v_F > 0 includes the relaxation time
//   tau = l / v_F through sigma(q, omega) = sigma0/(1 + i omega tau) F(q l / (1 + i omega tau)); v_F = 0: omega tau -> 0.
// p = 0 and p = 1 are the exact RS integrals (evaluated to ~1e-12 by the trapezoid rule in log q, which converges
// exponentially for these analytic integrands); 0 < p < 1 interpolates R between them with the weight
// w(alpha, p) tabulated from a numerical solution of the RS half-space integral equation
// (benchmarks/surface_resistance_specularity.py; accuracy <~ 3e-4 of R).
std::complex<double> anomalous_Z_diffuse(double f_Hz, double rho_ohm_m, double rho_l_ohm_m2, double v_F = 0);
std::complex<double> anomalous_Z_specular(double f_Hz, double rho_ohm_m, double rho_l_ohm_m2, double v_F = 0);
double anomalous(double f_Hz, double rho_ohm_m, double rho_l_ohm_m2, double p, double v_F = 0);
// The RS parameter alpha = (3/2) (l / delta_classical)^2 = (3/4) mu0 omega l^2 / rho.
double anomalous_alpha(double f_Hz, double rho_ohm_m, double rho_l_ohm_m2);
// Extreme anomalous limit (l -> infinity, omega tau -> 0): R_inf = (sqrt(3)/(16 pi) rho_l (mu0 omega)^2)^(1/3) for
// diffuse scattering, times 8/9 for specular; intermediate p by the alpha -> infinity weight of the table.
double anomalous_extreme(double f_Hz, double rho_l_ohm_m2, double p);
// Specularity weight w(alpha, p) = (R(p) - R(1)) / (R(0) - R(1)) (bilinear in log10 alpha and p; alpha clamped
// to the table range 1e-2 .. 1e12, outside which R(0) - R(1) -> 0 or w has converged).
double specularity_weight(double alpha, double p);

// BCS surface resistance in the low-temperature approximation (T < Tc/2, hbar omega << k T),
//   R_BCS = A_ohm_K / T (f / f_ref)^2 exp(-Delta_over_kTc * Tc / T) [* L(f, T) / L(f_ref, T) if log_factor]
// (Padamsee-style; for Nb A = 2e-4 ohm K at f_ref = 1.5 GHz, Delta/kTc = 17.67 K / 9.25 K), with
// L(f, T) = ln(kBcst * k T / (h f)), kBcst = 2.246 (the slowly varying factor of Mattis-Bardeen in this limit,
// Ciovati arXiv:1501.07398 Eq. 34; L is clamped at 0 where h f > 2.246 k T, outside the formula's validity).
// Residual resistance is added by the caller.
double bcs(double f_Hz, double T_K, double A_ohm_K, double f_ref_Hz, double Delta_over_kTc, double Tc_K,
           bool log_factor = false);
// The factor L(f, T) / L(f_ref, T) by which the plain f^2 law misses the log factor (1 at f = f_ref).
double bcs_log_ratio(double f_Hz, double f_ref_Hz, double T_K);

// Vortex (mixed-state) losses of a type-II superconductor, local two-fluid-free London + vortex model
// (Gittleman-Rosenblum pinning, Coffey-Clem creep in the form of Pompeo & Silva PRB 78, 094503 Eq. 2, Bardeen-Stephen
// flux-flow resistivity; e^{i omega t}):
//   rho_ff = rho_n |B| / B_c2,   rho_v = rho_ff (eps + i f / f0) / (1 + i f / f0)
// (eps = 0: no creep, f0 = depinning frequency, rho_v = rho_ff / (1 - i f0 / f); f0 = 0: free flux flow rho_ff).
std::complex<double> vortex_resistivity(double f_Hz, double rho_n_ohm_m, double B_T, double Bc2_T, double f0_Hz,
                                        double creep_eps = 0);
// Surface impedance of a thick superconductor with London depth lambda and (effective) vortex resistivity rho_v:
//   Z = sqrt(i mu0 omega (i mu0 omega lambda^2 + rho_v))   (Coffey-Clem complex penetration depth, no quasiparticles).
// Re Z -> Re rho_v / (2 lambda) for |rho_v| << mu0 omega lambda^2, and -> Re sqrt(i mu0 omega rho_v) (vortex-dominated,
// sqrt(mu0 omega rho_ff / 2) for free flux flow) for |rho_v| >> mu0 omega lambda^2.
std::complex<double> vortex_Z(double f_Hz, double lambda_m, std::complex<double> rho_v);
// The vortex surface resistance Re Z - Re Z(rho_v = 0) (= Re Z: the vortex-free London Z is reactive) with the
// vortex resistivity scaled by the orientation factor s = sin^2(angle between the RF current and the vortices),
// s in [0, 1] (1: current perpendicular to B, full Lorentz force; 0: current parallel to B). See
// docs/physics/surface_resistance.md Sec. 4. ORDER-OF-MAGNITUDE MODEL.
double flux_flow(double f_Hz, double rho_n_ohm_m, double lambda_m, double B_T, double Bc2_T, double f0_Hz,
                 double creep_eps, double orientation);

}  // namespace axf::rs
