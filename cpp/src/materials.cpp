#include "axionfem/materials.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf {

using units::mu0;

double BHLaw::nu(double B) const {
  if (B < 1e-12) return dHdB(0.0);
  return H(B) / B;
}

double BHLaw::B_of_H(double h) const {
  if (h <= 0) return 0.0;
  double lo = 0, hi = std::max(h / dHdB(0.0), 1e-9);
  while (H(hi) < h) {
    lo = hi;
    hi *= 2;
  }
  double B = 0.5 * (lo + hi);
  for (int it = 0; it < 200; ++it) {
    const double f = H(B) - h;
    if (f > 0) hi = B;
    else lo = B;
    if (std::fabs(f) <= 1e-13 * h || hi - lo <= 1e-15 * hi) break;
    double Bn = B - f / dHdB(B);
    B = (Bn > lo && Bn < hi) ? Bn : 0.5 * (lo + hi);
  }
  return B;
}

double BHLaw::dnu_dB2(double B) const {
  // d(H/B)/d(B^2) = (H' - H/B) / (2 B^2); finite as B -> 0 for smooth laws.
  if (B < 1e-9) return 0.0;
  return (dHdB(B) - H(B) / B) / (2.0 * B * B);
}

// ---------------------------------------------------------------- knee
double KneeBH::H(double B) const {
  if (B <= Bs_) return B / (mu0 * mur_);
  return Bs_ / (mu0 * mur_) + (B - Bs_) / mu0;
}
double KneeBH::dHdB(double B) const { return B <= Bs_ ? 1.0 / (mu0 * mur_) : 1.0 / mu0; }
double KneeBH::energy(double B) const {
  if (B <= Bs_) return 0.5 * B * B / (mu0 * mur_);
  double d = B - Bs_;
  return 0.5 * Bs_ * Bs_ / (mu0 * mur_) + Bs_ / (mu0 * mur_) * d + 0.5 * d * d / mu0;
}

// ---------------------------------------------------------------- table
TableBH::TableBH(std::vector<double> B, std::vector<double> H) : B_(std::move(B)), H_(std::move(H)) {
  if (B_.empty() || B_[0] != 0.0 || H_[0] != 0.0) {
    B_.insert(B_.begin(), 0.0);
    H_.insert(H_.begin(), 0.0);
  }
  const size_t n = B_.size();
  for (size_t i = 1; i < n; ++i) {
    AXF_REQUIRE(B_[i] > B_[i - 1] && H_[i] > H_[i - 1], "B-H table must be strictly increasing");
  }
  // Fritsch-Carlson slopes dH/dB.
  std::vector<double> d(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) d[i] = (H_[i + 1] - H_[i]) / (B_[i + 1] - B_[i]);
  m_.assign(n, 0.0);
  m_[0] = d[0];
  m_[n - 1] = d[n - 2];
  for (size_t i = 1; i + 1 < n; ++i) m_[i] = (d[i - 1] * d[i] <= 0) ? 0.0 : 0.5 * (d[i - 1] + d[i]);
  for (size_t i = 0; i + 1 < n; ++i) {
    double a = m_[i] / d[i], b = m_[i + 1] / d[i];
    double s = a * a + b * b;
    if (s > 9.0) {
      double t = 3.0 / std::sqrt(s);
      m_[i] = t * a * d[i];
      m_[i + 1] = t * b * d[i];
    }
  }
  // Cumulative energy at nodes (Simpson is exact for cubics).
  E_.assign(n, 0.0);
  for (size_t i = 0; i + 1 < n; ++i) {
    double bm = 0.5 * (B_[i] + B_[i + 1]);
    E_[i + 1] = E_[i] + (B_[i + 1] - B_[i]) / 6.0 * (H_[i] + 4.0 * this->H(bm) + H_[i + 1]);
  }
}

int TableBH::seg(double B) const {
  auto it = std::upper_bound(B_.begin(), B_.end(), B);
  int i = static_cast<int>(it - B_.begin()) - 1;
  return std::clamp(i, 0, static_cast<int>(B_.size()) - 2);
}

double TableBH::H(double B) const {
  if (B >= B_.back()) return H_.back() + (B - B_.back()) / mu0;
  int i = seg(B);
  double h = B_[i + 1] - B_[i], t = (B - B_[i]) / h;
  double h00 = (1 + 2 * t) * (1 - t) * (1 - t), h10 = t * (1 - t) * (1 - t);
  double h01 = t * t * (3 - 2 * t), h11 = t * t * (t - 1);
  return h00 * H_[i] + h10 * h * m_[i] + h01 * H_[i + 1] + h11 * h * m_[i + 1];
}

double TableBH::dHdB(double B) const {
  if (B >= B_.back()) return 1.0 / mu0;
  int i = seg(B);
  double h = B_[i + 1] - B_[i], t = (B - B_[i]) / h;
  double d00 = 6 * t * t - 6 * t, d10 = 3 * t * t - 4 * t + 1, d01 = -6 * t * t + 6 * t, d11 = 3 * t * t - 2 * t;
  return (d00 * H_[i] + d01 * H_[i + 1]) / h + d10 * m_[i] + d11 * m_[i + 1];
}

double TableBH::energy(double B) const {
  if (B >= B_.back()) {
    double d = B - B_.back();
    return E_.back() + H_.back() * d + 0.5 * d * d / mu0;
  }
  int i = seg(B);
  double bm = 0.5 * (B_[i] + B);
  return E_[i] + (B - B_[i]) / 6.0 * (H_[i] + 4.0 * H(bm) + H(B));
}

std::unique_ptr<BHLaw> make_bh(const spec::BHCurve& c) {
  if (c.model == "table") return std::make_unique<TableBH>(c.B, c.H);
  return std::make_unique<KneeBH>(c.mu_r, c.B_sat);
}

}  // namespace axf
