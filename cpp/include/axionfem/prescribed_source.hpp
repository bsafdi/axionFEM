#pragma once

#include <array>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace axf {

// A prescribed, real static field in SI, evaluated at the actual FE quadrature
// points. Segment currents are exact finite-wire Biot--Savart contributions;
// finite winding packs still require converged source quadrature and arc geometry.
// This class does not solve the static response of superconductors or iron.
class PrescribedMagneticSource {
 public:
  using Point = std::array<double, 3>;
  static PrescribedMagneticSource from_json(const nlohmann::json& source);
  Point field_T(const Point& x_m) const;
  nlohmann::json summary() const;
  bool is_uniform() const { return kind_ == "uniform"; }
  const std::string& kind() const { return kind_; }
  // kind=toroidal_box / axial_box with an azimuthal sector: its [phi0, phi1] in degrees (false: no sector)
  bool box_sector_deg(double out[2]) const {
    if (!box_sector_) return false;
    out[0] = box_phi_deg_[0];
    out[1] = box_phi_deg_[1];
    return true;
  }
  // Largest distance of any source current from `center`, including the rotated copies of an azimuthal average
  // (rotation about the global z axis). Straight segments: the maximum is at an endpoint.
  double max_distance_m(const Point& center) const;
  // Kirchhoff check for a radiation exterior: the largest net current leaving any segment endpoint (endpoints
  // within `tolerance_m` are one node), divided by the largest |segment current|. Zero for closed circuits.
  double max_relative_node_imbalance(double tolerance_m) const;
  // Whether any source filament (or rotated copy of an azimuthal average) meets the closed straight-sided
  // tetrahedron `v` (vertices in metres). False for a uniform field. B0 ~ 1/d near a filament makes Gaussian
  // quadrature of the source overlap non-convergent in such elements; the full-wave model counts them.
  bool meets_tetrahedron(const std::array<Point, 4>& v) const;
  // Current-carrying filaments with every rotated copy of an azimuthal average precomputed, and their bounding
  // boxes, for repeated element tests (build once, then call `proximity` per element).
  struct Filament {
    Point a, b, lo, hi;
  };
  std::vector<Filament> filaments() const;
  struct Proximity {
    bool meets = false;                 // a filament meets the closed straight-sided tetrahedron
    double distance_over_size = 1e300;  // min distance from the tetrahedron centroid to a filament / element size
  };
  // `size` is the largest bounding-box edge of the tetrahedron. Filaments whose bounding box is farther than
  // `cutoff` x size from the element's are skipped, so distance_over_size is exact only below `cutoff`.
  static Proximity proximity(const std::vector<Filament>& filaments, const std::array<Point, 4>& v, double cutoff);

 private:
  struct Segment {
    Point a, b;
    double current_A;
  };
  Point unaveraged_field_T(const Point& x_m) const;
  std::vector<Segment> segments_;
  Point uniform_T_{{0., 0., 0.}};
  // kind=toroidal_box: B = B0 phi_hat (profile uniform) or B0 r0/rho phi_hat (profile inverse_r) for
  // r0 <= rho <= r1, z0 <= z <= z1 about the global z axis; zero elsewhere.
  double box_B0_T_ = 0, box_r_[2] = {0, 0}, box_z_[2] = {0, 0};
  bool box_inverse_r_ = false;
  // optional azimuthal sector [phi0, phi1] in degrees (atan2(y, x), counter-clockwise from +x); default all
  bool box_sector_ = false;
  double box_phi_deg_[2] = {0, 360};
  std::string kind_, file_;
  int average_samples_ = 0;
  nlohmann::json metadata_;
};

}  // namespace axf
