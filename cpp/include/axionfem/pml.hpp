// Perfectly matched layers by complex coordinate stretching (time dependence e^{-i omega t}).  Beta
// (COMSOL import of PML domains; docs/comsol_import.md).
//
// A layer maps each stretched local coordinate q (Cartesian x, y, z; cylindrical radius rho and axial coordinate;
// spherical radius r) inside [q0, q0 + sign d] to
//     q~ = q0 + sign (d xi + (1 + i) F lambda xi^p),   xi = sign (q - q0) / d in [0, 1],
// so ds/dq = s = 1 + (1 + i) (F lambda / d) p xi^(p-1).  Outgoing waves e^{i k q} decay as e^{-k F lambda xi^p}
// (amplitude e^{-2 pi F} at the end of the layer for lambda = 2 pi / k).  This is COMSOL's polynomial stretching
// (scaling factor F, curvature parameter p, typical wavelength lambda: q~ - q0 = (1 - j) F lambda xi^p for
// e^{+j omega t}) plus the identity term d xi, which keeps s >= 1 at the interface for any p; any stretch that is
// analytic in the layer is reflectionless in the continuum, so the solution outside the layers does not depend on it
// up to the discretization of the layer.
//
// In the original coordinates the stretched problem is an anisotropic medium: with the Jacobian J = d x~ / d x,
// diagonal in the local orthonormal frame (e_1, e_2, e_3) with entries (J_1, J_2, J_3),
//     eps' = eps Lambda,  mu' = mu Lambda,   Lambda = det(J) J^{-1} J^{-T} = sum_i (J_1 J_2 J_3 / J_i^2) e_i e_i^T.
// Cartesian: J = (s_x, s_y, s_z).  Cylindrical (rho, phi, axial): (s_rho, rho~/rho, s_axial).  Spherical
// (r, theta, phi): (s_r, r~/r, r~/r).  Axisymmetric 2D designs use the cylindrical form about the symmetry axis
// (components ordered (r, z, phi)); planar 2D designs (x, y, z) use Cartesian or cylindrical about a point.
#pragma once

#include <array>
#include <complex>
#include <string>

#include <nlohmann/json.hpp>

namespace axf::pml {

using cplx = std::complex<double>;

struct Direction {
  bool on = false;
  double q0 = 0, d = 0;  // layer start and thickness (d > 0)
  int sign = 1;          // the layer extends towards +q (1) or -q (-1)
};

struct Layer {
  std::string name;      // design region(s) it applies to, for reports
  std::string type;      // "cartesian" | "cylindrical" | "spherical"
  int space = 3;         // 3: (x, y, z); 2: planar (x, y, z = out of plane); 1: axisymmetric (r, z, phi)
  double center[3] = {0, 0, 0};
  double axis[3] = {0, 0, 1};  // cylindrical (3D): unit axis
  double scaling = 1.0;        // F
  double curvature = 1.0;      // p >= 1
  double wavelength = 0.0;     // lambda [m]; 0: the vacuum wavelength c / f
  // Stretched directions (cartesian: x, y, z; cylindrical: rho, (unused), axial; spherical: r, -, -), one per side:
  // dir[i] towards +q (from the physical domain's upper face), dirn[i] towards -q. A point of a layer element is
  // stretched in coordinate i only where it lies beyond the physical domain on that side, so one layer may be a
  // frame, a box shell or side + cap + corner (COMSOL's usual PML domains).
  Direction dir[3], dirn[3];
  const Direction& pick(int i, double q) const;

  // Local coordinates (cartesian: x, y, z; cylindrical: rho, 0, axial; spherical: r, 0, 0) of a point given in the
  // components of `space` (3D: x, y, z; planar: x, y; axisymmetric: r, z).
  void local(const double* x, double q[3]) const;
  // Lambda at point x for angular frequency omega: Lt (3x3, row-major, components of `space`; for 2D the
  // out-of-plane component is index 2) and its inverse Li.  Points outside every active direction give identity.
  // La / Lia (optional): the SPD magnitudes sum_i |Lambda_i| e_i e_i^T and sum_i |1/Lambda_i| e_i e_i^T (preconditioners).
  // dLt / dLi (optional): d Lt / d omega and d Li / d omega (the exact dZ/dw). With the vacuum wavelength c / f
  // (wavelength 0) every stretch J_i - 1 is proportional to lambda ~ 1/omega, so dJ_i/domega = -(J_i - 1)/omega; with
  // a fixed wavelength the tensors do not depend on omega.
  void tensors(const double* x, double omega, cplx Lt[9], cplx Li[9], double* La = nullptr,
               double* Lia = nullptr, cplx* dLt = nullptr, cplx* dLi = nullptr) const;
  nlohmann::json summary() const;
};

// Extents of local coordinates (per index 0..2) of a set of points.
struct Extents {
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  void add(const double q[3]) {
    for (int i = 0; i < 3; ++i) { lo[i] = std::min(lo[i], q[i]); hi[i] = std::max(hi[i], q[i]); }
  }
};

// Activate the directions of `layer` from the extents of its own elements (`region`) and of the non-PML field
// domain (`physical`), both in the layer's local coordinates; `tol` is an absolute length tolerance. Coordinate i is
// stretched on the + side when the layer reaches beyond the physical domain's upper face (from that face to the
// layer's end) and likewise on the - side (radial coordinates: + side only). Throws (with the reason) when no
// direction is stretched or a spherical layer overlaps the physical domain.
void configure(Layer& layer, const Extents& physical, const Extents& region, double tol);

// Parse one entry of fullwave.pml (type, center, axis, scaling_factor, curvature, wavelength_m) into a Layer
// (directions unset).  `space` as in Layer.
Layer from_json(const nlohmann::json& j, int space, const std::string& where);

}  // namespace axf::pml
