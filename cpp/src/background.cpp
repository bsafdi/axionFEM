#include "axionfem/background.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"

namespace axf {

BackgroundField::BackgroundField(const spec::Background& b) {
  scale_ = b.scale;
  hermite_ = b.interpolation != "bilinear";
  if (b.type == "uniform") {
    type_ = 1;
    Bz0_ = b.Bz * b.scale;
  } else if (b.type == "grid") {
    type_ = 2;
    r_ = b.r;
    z_ = b.z;
    Br_ = b.Br;
    Bz_ = b.Bz_grid;
    for (auto& v : Br_) v *= scale_;
    for (auto& v : Bz_) v *= scale_;
    const size_t nr = r_.size(), nz = z_.size();
    auto at = [nr](size_t i, size_t j) { return j * nr + i; };
    // Hermite data in (u = r^2, z): psi is analytic in u near the axis (psi = B0 u/2 + O(u^2)).
    //   psi_u = Bz/2,  psi_z = -r Br,  psi_uz from finite differences.
    u_.resize(nr);
    for (size_t i = 0; i < nr; ++i) u_[i] = r_[i] * r_[i];
    pr_.assign(nr * nz, 0.0);
    pz_.assign(nr * nz, 0.0);
    prz_.assign(nr * nz, 0.0);
    for (size_t j = 0; j < nz; ++j)
      for (size_t i = 0; i < nr; ++i) {
        pr_[at(i, j)] = 0.5 * Bz_[at(i, j)];
        pz_[at(i, j)] = -r_[i] * Br_[at(i, j)];
      }
    // finite-difference derivative of a nodal array along u (dir 0) or z (dir 1), second order
    auto diff = [&](const std::vector<double>& f, size_t i, size_t j, int dir) {
      const std::vector<double>& x = dir == 0 ? u_ : z_;
      const size_t n = dir == 0 ? nr : nz, k = dir == 0 ? i : j;
      auto F = [&](size_t kk) { return dir == 0 ? f[at(kk, j)] : f[at(i, kk)]; };
      auto d3 = [&](size_t a, size_t b, size_t c, size_t at_k) {  // derivative of the parabola through a, b, c
        const double xa = x[a], xb = x[b], xc = x[c], xk = x[at_k];
        return F(a) * ((xk - xb) + (xk - xc)) / ((xa - xb) * (xa - xc)) +
               F(b) * ((xk - xa) + (xk - xc)) / ((xb - xa) * (xb - xc)) +
               F(c) * ((xk - xa) + (xk - xb)) / ((xc - xa) * (xc - xb));
      };
      if (n < 3) return (F(1) - F(0)) / (x[1] - x[0]);
      if (k == 0) return d3(0, 1, 2, 0);
      if (k == n - 1) return d3(n - 3, n - 2, n - 1, n - 1);
      return d3(k - 1, k, k + 1, k);
    };
    // mixed derivative: average of d(Bz/2)/dz and -d(r Br)/du (equal for a divergence-free field)
    for (size_t j = 0; j < nz; ++j)
      for (size_t i = 0; i < nr; ++i) prz_[at(i, j)] = 0.5 * (diff(pr_, i, j, 1) + diff(pz_, i, j, 0));
    // nodal psi: psi(0, z) = 0 and psi = int (Bz/2) du along each row (cubic Hermite quadrature with
    // end slopes; the legacy scheme uses the trapezoid rule in r)
    psi_.assign(nr * nz, 0.0);
    for (size_t j = 0; j < nz; ++j)
      for (size_t i = 1; i < nr; ++i) {
        double inc;
        if (hermite_) {
          const double h = u_[i] - u_[i - 1], f0 = pr_[at(i - 1, j)], f1 = pr_[at(i, j)];
          inc = 0.5 * h * (f0 + f1) + h * h / 12.0 * (diff(pr_, i - 1, j, 0) - diff(pr_, i, j, 0));
        } else {
          const double a = Bz_[at(i - 1, j)] * r_[i - 1], c = Bz_[at(i, j)] * r_[i];
          inc = 0.5 * (a + c) * (r_[i] - r_[i - 1]);
        }
        psi_[at(i, j)] = psi_[at(i - 1, j)] + inc;
      }
  }
}

bool BackgroundField::inside(double r, double z) const {
  if (type_ != 2) return true;
  return r >= r_.front() && r <= r_.back() && z >= z_.front() && z <= z_.back();
}

void BackgroundField::locate(double r, double z, int& i, int& j, double& tr, double& tz) const {
  const int nr = static_cast<int>(r_.size()), nz = static_cast<int>(z_.size());
  r = std::clamp(r, r_.front(), r_.back());
  z = std::clamp(z, z_.front(), z_.back());
  i = std::clamp(static_cast<int>(std::upper_bound(r_.begin(), r_.end(), r) - r_.begin()) - 1, 0, nr - 2);
  j = std::clamp(static_cast<int>(std::upper_bound(z_.begin(), z_.end(), z) - z_.begin()) - 1, 0, nz - 2);
  tr = (r - r_[i]) / (r_[i + 1] - r_[i]);
  tz = (z - z_[j]) / (z_[j + 1] - z_[j]);
}

void BackgroundField::hermite(double r, double z, double& p, double& p_u, double& p_z) const {
  const int nr = static_cast<int>(r_.size()), nz = static_cast<int>(z_.size());
  const double u = std::clamp(r * r, u_.front(), u_.back());
  z = std::clamp(z, z_.front(), z_.back());
  const int i = std::clamp(static_cast<int>(std::upper_bound(u_.begin(), u_.end(), u) - u_.begin()) - 1, 0, nr - 2);
  const int j = std::clamp(static_cast<int>(std::upper_bound(z_.begin(), z_.end(), z) - z_.begin()) - 1, 0, nz - 2);
  const double hu = u_[i + 1] - u_[i], hz = z_[j + 1] - z_[j];
  const double t = (u - u_[i]) / hu, s = (z - z_[j]) / hz;
  // cubic Hermite basis on [0, 1]: values at node 0/1 (H0, H1), slopes at node 0/1 (G0, G1)
  auto basis = [](double x, double v[4], double d[4]) {
    v[0] = 2 * x * x * x - 3 * x * x + 1;
    v[1] = -2 * x * x * x + 3 * x * x;
    v[2] = x * x * x - 2 * x * x + x;
    v[3] = x * x * x - x * x;
    d[0] = 6 * x * x - 6 * x;
    d[1] = -6 * x * x + 6 * x;
    d[2] = 3 * x * x - 4 * x + 1;
    d[3] = 3 * x * x - 2 * x;
  };
  double vu[4], du[4], vz[4], dz[4];
  basis(t, vu, du);
  basis(s, vz, dz);
  p = p_u = p_z = 0;
  for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 2; ++c) {
      const size_t n = static_cast<size_t>(j + c) * nr + (i + a);
      const double cv = psi_[n], cu = pr_[n] * hu, cz = pz_[n] * hz, cuz = prz_[n] * hu * hz;
      auto term = [&](const double* fu, const double* fz) {
        return cv * fu[a] * fz[c] + cu * fu[2 + a] * fz[c] + cz * fu[a] * fz[2 + c] + cuz * fu[2 + a] * fz[2 + c];
      };
      p += term(vu, vz);
      p_u += term(du, vz) / hu;
      p_z += term(vu, dz) / hz;
    }
}

double BackgroundField::psi(double r, double z) const {
  if (type_ == 1) return 0.5 * Bz0_ * r * r;
  if (type_ != 2 || !inside(r, z)) return 0.0;
  if (hermite_) {
    double p, pu, pz;
    hermite(r, z, p, pu, pz);
    return p;
  }
  int i, j;
  double tr, tz;
  locate(r, z, i, j, tr, tz);
  const size_t nr = r_.size();
  // legacy: bilinear in (r^2, z)
  double r0 = r_[i] * r_[i], r1 = r_[i + 1] * r_[i + 1];
  double ts = (r1 > r0) ? (r * r - r0) / (r1 - r0) : tr;
  ts = std::clamp(ts, 0.0, 1.0);
  auto P = [&](int ii, int jj) { return psi_[jj * nr + ii]; };
  return (1 - tz) * ((1 - ts) * P(i, j) + ts * P(i + 1, j)) + tz * ((1 - ts) * P(i, j + 1) + ts * P(i + 1, j + 1));
}

void BackgroundField::B(double r, double z, double& Br, double& Bz) const {
  Br = Bz = 0;
  if (type_ == 1) {
    Bz = Bz0_;
    return;
  }
  if (type_ != 2 || !inside(r, z)) return;
  if (hermite_) {
    // Bz = (1/r) d psi/dr = 2 d psi/du, Br = -(1/r) d psi/dz (psi_z = O(r^2) near the axis): continuous,
    // divergence-free and consistent with psi
    double p, pu, pz;
    hermite(r, z, p, pu, pz);
    Bz = 2 * pu;
    Br = r > 0 ? -pz / r : 0.0;
    return;
  }
  int i, j;
  double tr, tz;
  locate(r, z, i, j, tr, tz);
  const size_t nr = r_.size();
  auto bil = [&](const std::vector<double>& g) {
    return (1 - tz) * ((1 - tr) * g[j * nr + i] + tr * g[j * nr + i + 1]) +
           tz * ((1 - tr) * g[(j + 1) * nr + i] + tr * g[(j + 1) * nr + i + 1]);
  };
  Br = bil(Br_);
  Bz = bil(Bz_);
}

}  // namespace axf
