// Cross-section eigensolver for z-uniform cavity haloscopes ("symmetry": "xsec"; beta).
//
// A cavity whose cross-section does not depend on z (posts, vanes, partitions and dielectric rods spanning the full
// length, no end gaps) has two families of z-independent modes, each a scalar Helmholtz problem on the (x, y)
// cross-section (time dependence e^{-i omega t}, k0 = omega / c):
//
//   TM_z (E = E_z z^):  -div((1/mu_r) grad E_z) = k0^2 eps_r E_z,  E_z = 0 on every conductor (wall, posts, sheets).
//                       Weak form int (1/mu_r) grad u . grad v = k0^2 int eps_r u v: eps_r enters the mass matrix,
//                       E_z and (1/mu_r) dE_z/dn are continuous across dielectric interfaces (tangential E, H).
//                       With flat end caps at z = 0 and z = L these are exact 3D modes (TM_mn0) for any L.
//   TE_z (H = H_z z^):  -div((1/eps_r) grad H_z) = k0^2 mu_r H_z, natural condition on conductors; H_z is split
//                       (two values) across a conducting sheet with the cavity on both sides. These are the cutoff
//                       modes of the guide (C = 0): end caps would need p >= 1, i.e. z dependence. Kept for mode maps.
//
// Modes with z dependence (TM_mnp / TE_mnp with p >= 1, hybrid modes of dielectric loaded guides) are NOT captured.
//
// Per mode: f, form factor C = |int E_z dA|^2 / (A int eps_r |E_z|^2 dA) for a uniform axial B0 (the length cancels;
// A = cavity area, conductors excluded, dielectrics included; also C_full with the area enclosed by the outer wall),
// wall losses per unit length (P' = (R_s/2) oint |H_t|^2 dl, G = omega int mu |H|^2 dA / oint |H_t|^2 dl), the end-cap
// term for a given length L (TM_z: G_cap = omega L int mu |H|^2 dA / (2 int |H|^2 dA)), dielectric losses, Q0.
// Fields: E, H (real amplitudes, complex fields E_c = i E, H_c = H as in the axisymmetric cavity code), normalized to
// int eps_r |E|^2 dA = 1 (per unit length: E in m^-1).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/model.hpp"
#include "axionfem/spec.hpp"

namespace axf {

class XsecModel {
 public:
  explicit XsecModel(spec::Design d);
  ~XsecModel();
  XsecModel(const XsecModel&) = delete;
  XsecModel& operator=(const XsecModel&) = delete;

  // mesh (axf-mesher, "plane" mode) and solve; uses workdir, mesher, verbose and order of the options
  void run(const SolveOptions& o);
  bool solved() const { return solved_; }
  const nlohmann::json& results() const { return R_; }
  const spec::Design& design() const { return d_; }

  int mode_count(const std::string& family) const;
  // Fields of mode `index` of `family` ("TM"/"TE"; the order of results.fullwave.eigenmodes[family].modes),
  // normalized to int eps_r |E|^2 dA = 1 over the cavity cross-section.
  struct ModeField {
    std::string family;
    int index = 0;
    double f_Hz = 0, norm = 0;
    std::vector<double> x, y;
    std::vector<double> E[3], H[3];  // (x, y, z) components
    std::vector<double> eps_r, mu_r;
    std::vector<int> elem;
    std::vector<int> tris;  // sub-triangles (mode_mesh only), 3 point indices each
  };
  // on a lattice of `subdiv` intervals per edge of every cavity element
  ModeField mode_mesh(const std::string& family, int index, int subdiv = 2) const;
  // at points; NaN outside the cavity
  ModeField mode_sample(const std::string& family, int index, const std::vector<double>& x,
                        const std::vector<double>& y) const;
  // All elements on a lattice for drawing: x, y, tris, region index per point (`regions`: names; cavity elements
  // flagged), and the cavity wall edges (polylines per body).
  nlohmann::json geometry(int subdiv = 2) const;

  struct Impl;

 private:
  spec::Design d_;
  nlohmann::json R_;
  bool solved_ = false;
  std::unique_ptr<Impl> p_;
};

}  // namespace axf
