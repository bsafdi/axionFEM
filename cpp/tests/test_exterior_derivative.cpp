// The w-derivative of the 3D full-wave exterior coefficients, used by the exact dZ/dw (model3d_fullwave.cpp):
//   axi::hankel_log_derivative_dx against a Richardson difference of axi::hankel_log_derivative, n <= 40,
//   x = 0.05..52 (the Core runs reach x ~ 4.6 with n ~ 20);
//   its small-x limit d gamma_n/dx = -2x/(2n-1) at x = 1e-3, where the form x + (gamma - n)(gamma + n + 1)/x
//   loses digits to the n - n cancellation;
//   axi::fw_exterior_coefficient_dw against a Richardson difference of axi::fw_exterior_coefficient (TE and TM).
#include "axionfem/fullwave.hpp"
#include "axionfem/units.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <initializer_list>

using axf::axi::cplx;

// centred difference, Richardson-extrapolated: truncation O(h^4)
template <class F>
cplx richardson(F f, double x, double h) {
  const cplx d1 = (f(x + h) - f(x - h)) / (2.0 * h);
  const cplx d2 = (f(x + h / 2) - f(x - h / 2)) / h;
  return (4.0 * d2 - d1) / 3.0;
}

int main() {
  const int N = 40;
  const double R1 = 1.3;
  double worst_dx = 0, worst_small = 0, worst_te = 0, worst_tm = 0;
  for (double x : {0.05, 0.3, 1.0, 4.6, 20.0, 52.0}) {
    const auto dg = axf::axi::hankel_log_derivative_dx(x, N);
    // step: Richardson truncation (h/x)^4 matters near x ~ n (5e-7 at h = 0.01 x, x = 20, n = 22); rounding,
    // ~ n eps / h, matters at small x, where gamma_n - n ~ -x^2/(2n-1) is nearly polynomial and tolerates h > x/100
    const double h = 1e-3 * std::max(x, 1.0);
    const double w = x * axf::units::c / R1;
    for (int n = 0; n <= N; ++n) {
      auto gamma = [&](double xx) { return axf::axi::hankel_log_derivative(xx, N)[n]; };
      const cplx fd = richardson(gamma, x, h);
      worst_dx = std::max(worst_dx, std::abs(dg[n] - fd) / std::abs(dg[n]));
      const cplx g = gamma(x);
      for (bool te : {true, false}) {
        auto coef = [&](double ww) {
          return axf::axi::fw_exterior_coefficient(te, n, ww, R1, gamma(ww * R1 / axf::units::c));
        };
        const cplx exact = axf::axi::fw_exterior_coefficient_dw(te, w, R1, g, dg[n]);
        const double err = std::abs(exact - richardson(coef, w, h * axf::units::c / R1)) / std::abs(exact);
        if (te) worst_te = std::max(worst_te, err);
        else worst_tm = std::max(worst_tm, err);
      }
    }
  }
  const double xs = 1e-3;
  const auto dgs = axf::axi::hankel_log_derivative_dx(xs, N);
  for (int n = 2; n <= N; ++n) {
    const double lead = -2.0 * xs / (2.0 * n - 1.0);
    worst_small = std::max(worst_small, std::abs(dgs[n] - lead) / std::abs(lead));
  }
  std::printf("d gamma/dx vs Richardson %.2e, small-x limit %.2e, dc_TE/dw %.2e, dc_TM/dw %.2e\n", worst_dx,
              worst_small, worst_te, worst_tm);
  if (worst_dx > 1e-7) return 1;
  if (worst_small > 1e-5) return 2;
  if (worst_te > 1e-7) return 3;
  if (worst_tm > 1e-7) return 4;
  std::puts("exterior coefficient derivatives: passed");
  return 0;
}
