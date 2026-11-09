// Magnetic constitutive laws.
#pragma once

#include <memory>
#include <vector>

#include "axionfem/spec.hpp"

namespace axf {

// Anhysteretic, monotone B-H law H(B) for soft iron.
class BHLaw {
 public:
  virtual ~BHLaw() = default;
  virtual double H(double B) const = 0;      // A/m, B >= 0
  virtual double dHdB(double B) const = 0;   // A/(m T)
  virtual double energy(double B) const = 0; // J/m^3: int_0^B H db
  // Reluctivity nu = H/B (m/H) and d nu / d(B^2).
  double nu(double B) const;
  double dnu_dB2(double B) const;
  // Initial (B -> 0) reluctivity.
  double nu0() const { return dHdB(0.0); }
  // Inverse law B(H) for H >= 0 (safeguarded Newton on the monotone H(B)).
  double B_of_H(double H) const;
};

// H = B/(mu0 mu_r) below B_sat, then slope 1/mu0 (a simple saturation model).
class KneeBH : public BHLaw {
 public:
  KneeBH(double mu_r, double B_sat) : mur_(mu_r), Bs_(B_sat) {}
  double H(double B) const override;
  double dHdB(double B) const override;
  double energy(double B) const override;

 private:
  double mur_, Bs_;
};

// Monotone piecewise-cubic Hermite interpolation (Fritsch-Carlson) of a
// tabulated curve through (0, 0); linear continuation with slope 1/mu0 above
// the last point.
class TableBH : public BHLaw {
 public:
  TableBH(std::vector<double> B, std::vector<double> H);
  double H(double B) const override;
  double dHdB(double B) const override;
  double energy(double B) const override;

 private:
  std::vector<double> B_, H_, m_, E_;  // nodes, values, slopes, cumulative energy
  int seg(double B) const;
};

std::unique_ptr<BHLaw> make_bh(const spec::BHCurve& c);

}  // namespace axf
