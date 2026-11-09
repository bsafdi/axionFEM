// Surface-resistance models (beta). See cpp/include/axionfem/surface_resistance.hpp and
// docs/physics/surface_resistance.md.
#include "axionfem/surface_resistance.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf::rs {

namespace {

#include "rs_specularity_table.inc"

using cd = std::complex<double>;
constexpr double pi = axf::units::pi, mu0 = axf::units::mu0;

// RS / Pippard nonlocal conductivity kernel F(z) = sigma(q)/sigma0 at z = q l (complex with relaxation).
cd F_kernel(cd z) {
  if (std::abs(z) < 0.1) {  // series 3 sum_k (-1)^(k+1) z^(2k-2) / (4k^2 - 1), error ~ |z|^12
    const cd z2 = z * z;
    return 1.0 + z2 * (-1.0 / 5 + z2 * (3.0 / 35 + z2 * (-1.0 / 21 + z2 * (1.0 / 33 + z2 * (-3.0 / 143)))));
  }
  return 1.5 * ((1.0 + z * z) * std::atan(z) - z) / (z * z * z);
}

struct Scales {
  double w, sigma0, ell, q0, h;
  cd g;  // 1 + i omega tau
};

Scales scales(double f, double rho, double rho_l, double v_F) {
  if (!(f > 0) || !(rho > 0) || !(rho_l > 0))
    throw Error(strcat("anomalous skin effect needs f > 0, rho > 0 and rho_l > 0 (got f = ", f, ", rho = ", rho,
                       ", rho_l = ", rho_l, ")"));
  Scales s;
  s.w = 2 * pi * f;
  s.sigma0 = 1.0 / rho;
  s.ell = rho_l / rho;
  const double wt = v_F > 0 ? s.w * s.ell / v_F : 0.0;
  s.g = cd(1.0, wt);
  const double beta = mu0 * s.w * s.ell * s.ell / rho;
  // decay length of the field / l: beta^(-1/3) (anomalous) or beta^(-1/2) (classical)
  s.q0 = (beta > 1 ? std::cbrt(beta) : std::sqrt(beta)) / s.ell;
  // trapezoid step in x = ln(q/q0): the error ~ exp(-2 pi d / h), d = distance of the nearest singularity
  // from the real x axis (pi/4 for omega tau = 0; atan(1/omega tau) for the relaxation branch points)
  s.h = std::min(0.05, std::atan2(1.0, wt) / 8);
  return s;
}

constexpr double kXmax = 50.0;  // q from q0 e^-50 to q0 e^50: both tails decay at least like e^-|x|

}  // namespace

double skin(double f, double rho) { return std::sqrt(pi * f * mu0 * rho); }

double anomalous_alpha(double f, double rho, double rho_l) {
  const double ell = rho_l / rho;
  return 0.75 * mu0 * 2 * pi * f * ell * ell / rho;
}

cd anomalous_Z_diffuse(double f, double rho, double rho_l, double v_F) {
  // Z = i pi omega mu0 / int_0^inf ln(1 + i omega mu0 sigma(q) / q^2) dq   (Reuter-Sondheimer, p = 0; e^{i omega t})
  const Scales s = scales(f, rho, rho_l, v_F);
  const cd a = cd(0.0, s.w * mu0 * s.sigma0) / s.g;
  cd I = 0;
  const int n = static_cast<int>(std::ceil(kXmax / s.h));
  for (int k = -n; k <= n; ++k) {
    const double q = s.q0 * std::exp(k * s.h);
    I += std::log(1.0 + a * F_kernel(q * s.ell / s.g) / (q * q)) * q;
  }
  I *= s.h;
  return cd(0.0, pi * s.w * mu0) / I;
}

cd anomalous_Z_specular(double f, double rho, double rho_l, double v_F) {
  // Z = (2 i omega mu0 / pi) int_0^inf dq / (q^2 + i omega mu0 sigma(q))   (p = 1)
  const Scales s = scales(f, rho, rho_l, v_F);
  const cd a = cd(0.0, s.w * mu0 * s.sigma0) / s.g;
  cd I = 0;
  const int n = static_cast<int>(std::ceil(kXmax / s.h));
  for (int k = -n; k <= n; ++k) {
    const double q = s.q0 * std::exp(k * s.h);
    I += q / (q * q + a * F_kernel(q * s.ell / s.g));
  }
  I *= s.h;
  return cd(0.0, 2 * s.w * mu0 / pi) * I;
}

double specularity_weight(double alpha, double p) {
  if (!(p >= 0 && p <= 1)) throw Error(strcat("specularity p must be in [0, 1], got ", p));
  if (p == 0) return 1.0;
  if (p == 1) return 0.0;
  const double la = std::clamp(std::log10(alpha), kRsW_LA0, kRsW_LA0 + (kRsW_NA - 1) * kRsW_DLA);
  double ua = (la - kRsW_LA0) / kRsW_DLA;
  int i = std::min(static_cast<int>(ua), kRsW_NA - 2);
  ua -= i;
  double up = p * (kRsW_NP - 1);
  int j = std::min(static_cast<int>(up), kRsW_NP - 2);
  up -= j;
  return (1 - ua) * ((1 - up) * kRsW[i][j] + up * kRsW[i][j + 1]) +
         ua * ((1 - up) * kRsW[i + 1][j] + up * kRsW[i + 1][j + 1]);
}

double anomalous(double f, double rho, double rho_l, double p, double v_F) {
  if (!(p >= 0 && p <= 1)) throw Error(strcat("specularity p must be in [0, 1], got ", p));
  if (p == 0) return anomalous_Z_diffuse(f, rho, rho_l, v_F).real();
  const double R1 = anomalous_Z_specular(f, rho, rho_l, v_F).real();
  if (p == 1) return R1;
  const double R0 = anomalous_Z_diffuse(f, rho, rho_l, v_F).real();
  return R1 + specularity_weight(anomalous_alpha(f, rho, rho_l), p) * (R0 - R1);
}

double anomalous_extreme(double f, double rho_l, double p) {
  const double w = 2 * pi * f;
  const double R0 = std::cbrt(std::sqrt(3.0) / (16 * pi) * rho_l * (mu0 * w) * (mu0 * w));
  return R0 * (8.0 / 9.0 + specularity_weight(1e300, p) / 9.0);
}

double bcs_log_ratio(double f, double f_ref, double T) {
  const double c = 2.246 * axf::units::kB * T / axf::units::h;  // 2.246 k T / h [Hz]
  return std::max(std::log(c / f), 0.0) / std::log(c / f_ref);
}

double bcs(double f, double T, double A, double f_ref, double Delta_over_kTc, double Tc, bool log_factor) {
  if (!(T > 0) || !(T < Tc)) throw Error(strcat("BCS surface resistance needs 0 < T < Tc (T = ", T, " K, Tc = ", Tc, " K)"));
  const double r = A / T * (f / f_ref) * (f / f_ref) * std::exp(-Delta_over_kTc * Tc / T);
  return log_factor ? r * bcs_log_ratio(f, f_ref, T) : r;
}

cd vortex_resistivity(double f, double rho_n, double B, double Bc2, double f0, double eps) {
  const double rho_ff = rho_n * std::fabs(B) / Bc2;
  if (!(f0 > 0)) return rho_ff;
  const cd x(0.0, f / f0);
  return rho_ff * (eps + x) / (1.0 + x);
}

cd vortex_Z(double f, double lambda, cd rho_v) {
  const double wm = 2 * pi * f * mu0;
  return std::sqrt(cd(0.0, wm) * (cd(0.0, wm * lambda * lambda) + rho_v));
}

double flux_flow(double f, double rho_n, double lambda, double B, double Bc2, double f0, double eps, double s) {
  if (!(s > 0) || B == 0) return 0.0;
  return vortex_Z(f, lambda, s * vortex_resistivity(f, rho_n, B, Bc2, f0, eps)).real();
}

}  // namespace axf::rs
