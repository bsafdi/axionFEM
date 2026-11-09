// The moments of a grid background outside the domain sphere (axi::background_exterior, shared by the 3D and the
// axisymmetric full-wave sweeps; docs/physics/fullwave_background.md), on a smooth analytic field sampled on a grid:
//   - the same moments, bit for bit, with 1 and 4 OpenMP threads (each sphere fills its own column);
//   - the grid cut at its box (the inscribed sphere inside R1: two arcs per sphere, extra radial breaks), which no
//     validation run reaches: G_n at k -> 0 and at one k against an independent (r, z) quadrature of the box minus
//     the ball, with panel breaks at the grid lines and at the sphere, for a box about the centre and one beside it;
//   - a grid inside the sphere: no nodes, G_n = 0.
#include "axionfem/background.hpp"
#include "axionfem/fullwave.hpp"
#include "axionfem/spec.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using axf::axi::cplx;

namespace {

const double kPi = 3.14159265358979323846;

// a softened dipole, psi = m r^2 / D^{3/2} with D = r^2 + (z - z0)^2 + a^2 (divergence-free by construction):
// Bz = (1/r) dpsi/dr, Br = -(1/r) dpsi/dz
axf::spec::Background grid(double r_max, double z_lo, double z_hi, double h) {
  const double m = 0.02;
  const double a = 0.08;
  const double z0 = 0.05;
  axf::spec::Background b;
  b.type = "grid";
  const int nr = static_cast<int>(std::round(r_max / h)) + 1;
  const int nz = static_cast<int>(std::round((z_hi - z_lo) / h)) + 1;
  for (int i = 0; i < nr; ++i) b.r.push_back(r_max * i / (nr - 1));
  for (int j = 0; j < nz; ++j) b.z.push_back(z_lo + (z_hi - z_lo) * j / (nz - 1));
  for (int j = 0; j < nz; ++j)
    for (int i = 0; i < nr; ++i) {
      const double r = b.r[i];
      const double dz = b.z[j] - z0;
      const double D = r * r + dz * dz + a * a;
      b.Bz_grid.push_back(m * (2.0 * std::pow(D, -1.5) - 3.0 * r * r * std::pow(D, -2.5)));
      b.Br.push_back(3.0 * m * r * dz * std::pow(D, -2.5));
    }
  return b;
}

void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {  // on [0, 1], Newton on P_n
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double t = std::cos(kPi * (i + 0.75) / (n + 0.5));
    double dp = 1.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0;
      double p1 = t;
      for (int k = 2; k <= n; ++k) {
        const double p2 = ((2.0 * k - 1.0) * t * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (t * p1 - p0) / (t * t - 1.0);
      const double dt = p1 / dp;
      t -= dt;
      if (std::fabs(dt) < 1e-16) break;
    }
    x[i] = 0.5 * (1.0 - t);
    w[i] = 1.0 / ((1.0 - t * t) * dp * dp);
  }
}

std::vector<double> breaks(double a, double b, const std::vector<double>& nodes, std::vector<double> extra) {
  std::vector<double> out = {a, b};
  for (double x : nodes)
    if (x > a && x < b) out.push_back(x);
  for (double x : extra)
    if (x > a && x < b) out.push_back(x);
  std::sort(out.begin(), out.end());
  std::vector<double> uniq;
  for (double x : out)
    if (uniq.empty() || x - uniq.back() > 1e-14) uniq.push_back(x);
  return uniq;
}

// G_n over the box minus the ball by an (r, z) quadrature: each z line from the sphere (or the axis) to r_max
void independent_G(const axf::BackgroundField& bg, const axf::spec::Background& b, double R1, double cz, int N,
                   double k, std::vector<double>& Gs, std::vector<cplx>& Gk) {
  Gs.assign(N + 1, 0.0);
  Gk.assign(N + 1, 0.0);
  std::vector<double> gx;
  std::vector<double> gw;
  gauss_legendre(10, gx, gw);
  const double r_max = b.r.back();
  std::vector<double> P;
  std::vector<double> dy;
  std::vector<cplx> h;
  // z breaks: the grid lines, the sphere's poles and where the sphere crosses r = r_max (the r range starts or ends
  // there: a kink in z)
  std::vector<double> zextra = {cz - R1, cz + R1, cz};
  if (R1 > r_max) {
    zextra.push_back(cz - std::sqrt(R1 * R1 - r_max * r_max));
    zextra.push_back(cz + std::sqrt(R1 * R1 - r_max * r_max));
  }
  const auto zb = breaks(b.z.front(), b.z.back(), b.z, zextra);
  for (size_t iz = 0; iz + 1 < zb.size(); ++iz) {
    for (int a = 0; a < 10; ++a) {
      const double z = zb[iz] + (zb[iz + 1] - zb[iz]) * gx[a];
      const double wz = (zb[iz + 1] - zb[iz]) * gw[a];
      const double dz = z - cz;
      const double r_lo = std::sqrt(std::max(0.0, R1 * R1 - dz * dz));
      if (!(r_max > r_lo)) continue;
      const auto rb = breaks(r_lo, r_max, b.r, {});
      for (size_t ir = 0; ir + 1 < rb.size(); ++ir) {
        for (int c = 0; c < 10; ++c) {
          const double r = rb[ir] + (rb[ir + 1] - rb[ir]) * gx[c];
          const double wr = (rb[ir + 1] - rb[ir]) * gw[c];
          if (!(r > 0)) continue;
          const double rho = std::hypot(r, dz);
          const double weight = 2 * kPi * r * wz * wr * bg.psi(r, z) / r;
          axf::axi::dtheta_y0(dz / rho, N, dy);
          axf::axi::hankel_ratio(k, R1, rho, N, h);
          double q = R1 / rho;
          for (int n = 1; n <= N; ++n) {
            q *= R1 / rho;
            Gs[n] += weight * dy[n] * q;
            Gk[n] += weight * dy[n] * h[n];
          }
        }
      }
    }
  }
}

bool same(const axf::axi::BackgroundExterior& x, const axf::axi::BackgroundExterior& y) {
  if (x.N != y.N || x.rho != y.rho || x.tail != y.tail || x.moments != y.moments) return false;
  return x.rho_s == y.rho_s && x.tail_mismatch == y.tail_mismatch && x.tail_samples == y.tail_samples;
}

}  // namespace

int main() {
  int failures = 0;
  const int N = 12;
  const int panels = 128;

  // 1. thread-count independence (a continued grid and a cut one)
  {
    const auto b = grid(1.2, -1.2, 1.2, 0.02);
    const axf::BackgroundField bg(b);
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto one = axf::axi::background_exterior(bg, 1.2, -1.2, 1.2, 0.45, 0.0, 80, panels);
    const auto one_cut = axf::axi::background_exterior(bg, 1.2, -1.2, 1.2, 1.3, 0.0, 80, panels);
    omp_set_num_threads(4);
    const auto four = axf::axi::background_exterior(bg, 1.2, -1.2, 1.2, 0.45, 0.0, 80, panels);
    const auto four_cut = axf::axi::background_exterior(bg, 1.2, -1.2, 1.2, 1.3, 0.0, 80, panels);
    omp_set_num_threads(saved);
    const bool ok = same(one, four) && same(one_cut, four_cut) && !one.rho.empty() && one.rho_s > 0 &&
                    !one_cut.rho.empty() && one_cut.rho_s == 0.0;
    std::printf("threads 1 and 4: moments %s (%zu and %zu radial nodes)\n", ok ? "bit-identical" : "DIFFER",
                one.rho.size(), one_cut.rho.size());
    if (!ok) ++failures;
#else
    std::printf("threads: built without OpenMP, skipped\n");
#endif
  }

  // 2. the grid cut at its box, against an independent quadrature
  struct Case {
    const char* name;
    double r_max, z_lo, z_hi, R1, cz;
  };
  for (const Case& c : {Case{"box about the centre", 0.4, -0.3, 0.3, 0.45, 0.0},
                        Case{"box beside the centre", 0.5, 0.1, 0.6, 0.3, 0.0},
                        Case{"off-centre domain", 0.4, -0.5, 0.2, 0.35, -0.1}}) {
    const auto b = grid(c.r_max, c.z_lo, c.z_hi, 0.02);
    const axf::BackgroundField bg(b);
    const auto ext = axf::axi::background_exterior(bg, c.r_max, c.z_lo, c.z_hi, c.R1, c.cz, N, panels);
    const double k = 3.3;
    std::vector<double> Gs;
    std::vector<cplx> Gk;
    independent_G(bg, b, c.R1, c.cz, N, k, Gs, Gk);
    const auto g0 = ext.G_static();
    const auto g1 = ext.G(k);
    double scale = 0;
    double err0 = 0;
    double err1 = 0;
    for (int n = 1; n <= N; ++n) scale = std::max(scale, std::fabs(Gs[n]));
    for (int n = 1; n <= N; ++n) {
      err0 = std::max(err0, std::fabs(g0[n] - Gs[n]) / scale);
      err1 = std::max(err1, std::abs(g1[n] - Gk[n]) / scale);
    }
    double gmax = 0;
    const double check =
        axf::axi::background_exterior_quadrature_check(bg, c.r_max, c.z_lo, c.z_hi, c.R1, c.cz, panels, ext, &gmax);
    const bool ok = ext.rho_s == 0.0 && !ext.rho.empty() && scale > 0 && err0 < 1e-5 && err1 < 1e-5 && check < 1e-4;
    std::printf("cut at its box, %s: |G - independent| / max|G| = %.2e (k -> 0), %.2e (k = %g); check %.2e%s\n",
                c.name, err0, err1, k, check, ok ? "" : "  FAIL");
    if (!ok) ++failures;
  }

  // 3. a grid inside the domain sphere
  {
    const auto b = grid(0.3, -0.2, 0.2, 0.02);
    const axf::BackgroundField bg(b);
    const auto ext = axf::axi::background_exterior(bg, 0.3, -0.2, 0.2, 0.45, 0.0, N, panels);
    const auto g = ext.G(2.0);
    double gabs = 0;
    for (const auto& v : g) gabs = std::max(gabs, std::abs(v));
    const bool ok = ext.rho.empty() && ext.rho_s == 0.0 && gabs == 0.0;
    std::printf("grid inside the sphere: %zu nodes, max|G| = %g%s\n", ext.rho.size(), gabs, ok ? "" : "  FAIL");
    if (!ok) ++failures;
  }

  std::printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}
