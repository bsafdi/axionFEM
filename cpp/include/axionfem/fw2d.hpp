// 2D full-wave model on an imported mesh: planar ("symmetry": "planar", fields independent of z, out-of-plane
// thickness D) or axisymmetric ("symmetry": "axisymmetric" with mesh.file; m = 0 fields).  Beta
// (COMSOL 2D / 2D-axisymmetric import, docs/comsol_import.md; physics: docs/physics/fw2d.md).
//
// Time dependence e^{-i omega t}.  The 3D field splits into two decoupled families:
//   * "out": E = u e_n out of the plane (planar E_z, axisymmetric E_phi); H1 elements;
//   * "in":  E in the plane (planar (E_x, E_y), axisymmetric (E_r, E_z)); Nedelec elements.
// Weak form per family (w = 2 pi r axisymmetric, D planar; every integral is the 3D one):
//   int (curl E)^T nu' (curl v) w dA - omega^2 int E^T eps' v w dA + sum_Robin beta oint E_t . v_t w dl
//       = i omega int J . v w dA + (port) i omega I int_Gamma t . v dl
// with nu' = (mu Lambda)^{-1}, eps' = eps_c Lambda (Lambda = 1 outside PML layers, axionfem/pml.hpp).
// Curl components: out family planar (d_y u, -d_x u), axisymmetric (-d_z u, d_r u + u / r); in family the scalar
// d_1 E_2 - d_2 E_1 (the out-of-plane component, up to sign).
// Boundary groups (mesh.info_file): pec (E_t = 0, essential), pmc (natural), axis (axisymmetric r = 0: E_phi = 0;
// natural for the in family), port (lumped port, in family: uniform current I across the edge,
// K = I / w t; Z = V / I with V = -int_Gamma t . E dl), impedance (beta = -i omega / Z_s, Z_s = sqrt(mu_m / eps_c,m)
// of the group's material, or a given Zs_ohm), sheet (interior thin layer with E_t continuous:
// beta = -i omega Y_s, Y_s = -(2 i / eta_m) tan(k_m d / 2)), scattering (first-order absorbing:
// beta = -i omega / eta; spherical waves add nothing in the E formulation, cylindrical waves +nu/(2 rho) for E_t
// along the axis and -nu/(2 rho) for E_t azimuthal).
// Eigenmodes (closed, lossless): H1 problems, the out family in E (PEC: u = 0) and the in family through its dual
// H = H_n e_n (alpha = 1 / eps, beta = mu; PMC and the axis: u = 0, PEC natural); impedance walls are PEC there,
// with the perturbative wall Q = omega U / P_wall, P_wall = 1/2 oint Re(Z_s) |H_t|^2.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/model.hpp"
#include "axionfem/spec.hpp"

namespace axf {

class Fw2DModel {
 public:
  explicit Fw2DModel(spec::Design d);
  ~Fw2DModel();
  Fw2DModel(const Fw2DModel&) = delete;
  Fw2DModel& operator=(const Fw2DModel&) = delete;

  void run(const SolveOptions& o);
  bool solved() const { return solved_; }
  const nlohmann::json& results() const { return R_; }
  const spec::Design& design() const { return d_; }

  // Fields at points (x1, x2) = (x, y) planar or (r, z) axisymmetric, per unit port current (sweep) or for the
  // normalized eigenmode: {"E": [[re, im] x 3 components (in-plane 1, in-plane 2, out of plane)] per point,
  // "H": same, "valid": [0/1]}.  Points outside the field domain or in PEC are invalid (zero, never physical).
  // what = "sweep" (index: frequency index) or "mode" (family "TE"/"TM", index within the family).
  nlohmann::json sample(const std::string& what, const std::string& family, int index, const std::vector<double>& x1,
                        const std::vector<double>& x2) const;

  struct Impl;

 private:
  spec::Design d_;
  nlohmann::json R_;
  bool solved_ = false;
  std::unique_ptr<Impl> p_;
};

}  // namespace axf
