// External static field in the meridional plane: flux function psi(r, z) and B(r, z).
#pragma once

#include <vector>

#include "axionfem/spec.hpp"

namespace axf {

class BackgroundField {
 public:
  BackgroundField() = default;
  explicit BackgroundField(const spec::Background& b);
  bool active() const { return type_ != 0; }
  bool inside(double r, double z) const;
  double psi(double r, double z) const;  // Wb/rad (r A_phi)
  void B(double r, double z, double& Br, double& Bz) const;

 private:
  int type_ = 0;  // 0 none, 1 uniform, 2 grid
  bool hermite_ = true;  // bicubic Hermite psi (C1, consistent B) or the legacy bilinear scheme
  double Bz0_ = 0, scale_ = 1;
  std::vector<double> r_, z_, Br_, Bz_, psi_;
  std::vector<double> u_;              // u = r^2 nodes
  std::vector<double> pr_, pz_, prz_;  // d psi/du = Bz/2, d psi/dz = -r Br, d2 psi/du dz at the nodes
  void locate(double r, double z, int& i, int& j, double& tr, double& tz) const;
  // bicubic Hermite evaluation in (u = r^2, z): psi, d psi/du, d psi/dz
  void hermite(double r, double z, double& p, double& p_u, double& p_z) const;
};

}  // namespace axf
