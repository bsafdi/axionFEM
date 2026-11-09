// Vibrations: axisymmetric m = 0 torsional dynamics on the meridional mesh.
//
// Why a separate module.  The elasticity solver (mech.hpp) is the torsion-free axisymmetric
// problem in u = (u_r, u_z).  The vibration study of a lumped-element axion detector needs the
// *azimuthal* displacement u_phi instead: for a solenoid with a coaxial pickup the coupling weight
// w = (n x B_p) x B_lab/(mu0 I_p) is purely azimuthal, so only u_phi couples the static field into
// the pickup flux at first order.  For axisymmetric (m = 0) motion u_phi decouples completely from
// (u_r, u_z), and the problem is scalar:
//
//   strains      eps_rphi = (1/2)(d_r u - u/r),   eps_zphi = (1/2) d_z u,
//   energy       U = 2 pi int (mu/2)[(d_r u - u/r)^2 + (d_z u)^2] r dr dz,
//   kinetic      T = 2 pi int (rho/2) (du/dt)^2 r dr dz,
//
// so, per radian,
//   K_ij = int mu [(d_r N_i - N_i/r)(d_r N_j - N_j/r) + d_z N_i d_z N_j] r dr dz,
//   M_ij = int rho N_i N_j r dr dz.
//
// The rigid rotation u_phi = c r is an exact zero mode of K (both strains vanish), which is used as
// a validation check.  u_phi = 0 is imposed on the axis, and the quotient u/r is replaced by its
// limit d_r u at quadrature points with r <= Space::rtol, exactly as the hoop term of mech.hpp.
//
// Two analyses are provided.
//  * modes(): the lowest eigenfrequencies of K x = (2 pi f)^2 M x by M-orthogonal Lanczos on
//    (K + sigma M)^{-1} M (the shifted matrix is SPD, so a Cholesky factorization suffices), with
//    full reorthogonalization, a reported residual per mode and a Sylvester inertia count that
//    checks that no eigenvalue below the highest returned one was missed.
//  * harmonic(): the complex response to a prescribed azimuthal body-force field g(r, z)
//    (the P_p interpolant of nodal values) with structural damping.  Time dependence
//    e^{-i omega t}, as the full-wave engine:
//      (K (1 - i eta) - omega^2 M) u = F,     F_i = int g N_i r dr dz,
//    and the *generalised mobility* of the conjugate pair (lambda, q) with q = 2 pi F^T u, i.e.
//    q = int g u_phi dV over the full ring and Y = -i omega q (velocity -i omega u) at unit load
//    scale.  The fluctuation-dissipation theorem then gives the Brownian spectrum of q directly,
//    S_q(omega) = 4 kB T Re[Y]/omega^2 (one-sided); Re[Y] does not depend on the convention.
//    For a thin tube driven by a load uniform through the wall, below the first thickness-shear
//    cutoff f < c_T/(2 t), the band mean of Re[Y] over many modes tends to the damped infinite
//    T(0,1) waveguide value, and as eta -> 0 to (int g dV)^2 kappa(k)^2 Y_inf,
//    Y_inf = 1/(2 b sqrt(S rho_s)), kappa the load's normalised axial transform. At finite eta the
//    damped value differs from the lossless one at first order in eta, with a sign set by the
//    shape of |kappa(k)|^2 (benchmarks/torsion.py, damped_waveguide_re_y).  Surface tractions above that cutoff also drive the T(0, n >= 2) branches, and
//    Re[Y] exceeds the T(0,1) value (by 4-17x in a tested case): the 1D asymptote does not
//    hold there.  The all-branch infinite-tube mobility is axionfem.torsion_waveguide (Python), and the
//    production path above the cutoff is this solver on a resolved wall: add_volume_load /
//    add_face_load (exact quadrature-point loads), harmonic_F (complex symmetric direct solves,
//    frequencies in parallel) and resolution() (elements per shear wavelength);
//    docs/physics/torsion_highfreq.md (part B) and docs/validation/torsion_highfreq.md.
#ifndef AXIONFEM_TORSION_HPP
#define AXIONFEM_TORSION_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/axi.hpp"
// json comes from core.hpp

namespace axf::axi {

struct TorsionRole {
  bool solid = false;  // element belongs to the structure
  double mu = 0;       // shear modulus [Pa]
  double rho = 0;      // mass density [kg/m^3]
};

struct TorsionModes {
  std::vector<double> f_Hz;                   // eigenfrequencies, ascending
  std::vector<double> residual;               // ||K x - lam M x||_2/(||K x||_2 + |lam| ||M x||_2)
  std::vector<double> modal_mass;             // 2 pi x^T M x (= 1 after normalisation)
  std::vector<std::vector<double>> shape;     // optional mass-normalised u_phi per DOF
  json diagnostics;
};

struct TorsionResponse {
  double f_Hz = 0;
  double q_re = 0, q_im = 0;   // generalised coordinate per unit load scale
  double Y_re = 0, Y_im = 0;   // generalised mobility Y = -i omega q (e^{-i omega t})
  int lu_ok = 1;
};

// Value of a distributed load at a point of element e (the point carries r, z and the shape functions).
using TorsionLoadFn = std::function<double(int e, const PointEval& pe)>;

// Integrals of an applied load over the full ring.
struct TorsionLoadMoments {
  double force_N = 0;       // int g dV + int tau dA
  double torque_N_m = 0;    // int g r dV + int tau r dA
  double zmin = 1e300, zmax = -1e300;
  void add(const TorsionLoadMoments& o);
  json to_json() const;
};

struct TorsionHarmonicOptions {
  double eta = 1e-4;
  int backend = -1;               // -1 by size, 0 Eigen SparseLU, 1 STRUMPACK (ComplexSymSolver)
  int parallel_frequencies = 1;   // frequencies factorized concurrently (each with its own solver);
                                  // 0 = auto: probe both backends and the memory, then choose
  int verbose = 0;
};

// Memory this job may still use [bytes]: the smallest cgroup limit (v2 memory.max / v1
// memory.limit_in_bytes over the process's cgroup and every ancestor, "/" included) less that cgroup's usage
// without reclaimable file cache; else SLURM_MEM_PER_NODE or SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE less the RSS;
// bounded by MemAvailable.  `opts` overrides for tests: {"proc": dir, "cgroup": dir, "env": {name: value}}.
// Returns {available_bytes, found, source, rss_bytes, sources}.
json torsion_available_memory(const json& opts);
// The auto mode's decisions from measured inputs (pure; tested directly): stage "probe" (whether to probe Eigen
// SparseLU after STRUMPACK) and stage "choose" (backend and concurrency).
json torsion_auto_choice(const json& in);

class Torsion {
 public:
  Torsion(const Space& sp, std::vector<TorsionRole> roles);

  TorsionModes modes(int nmodes, bool want_shapes = false, int lanczos_extra = 24,
                     int verbose = 0) const;
  // `load` holds one value per DOF of the space (zero outside the structure is enforced).
  std::vector<TorsionResponse> harmonic(const std::vector<double>& load,
                                        const std::vector<double>& f_Hz, double eta,
                                        int verbose = 0) const;
  // Modal participation of a nodal load, for the modal cross-check of harmonic().
  std::vector<double> participation(const TorsionModes& m,
                                    const std::vector<double>& load) const;
  // The same from an assembled per-radian load vector (free DOFs).
  std::vector<double> participation_F(const TorsionModes& m, const std::vector<double>& F) const;
  // Per-radian load vector of a nodal azimuthal body force.
  std::vector<double> load_vector(const std::vector<double>& load) const;
  // Add to a per-radian load vector F (free DOFs, size n_free()):
  //  * a body force g [N/m^3] over the given elements (non-structure elements are skipped);
  //  * an azimuthal surface traction tau [Pa] on mesh faces, each given as (face, structure element
  //    whose edge it is); the traction is integrated with the element's shape functions on the face.
  // Both are integrated at quadrature points (no interpolation of the load), so a load restricted
  // to a region or a boundary has exact edges.  Returns the integrals of what was added.
  TorsionLoadMoments add_volume_load(std::vector<double>& F, const std::vector<int>& elements,
                                     const TorsionLoadFn& g) const;
  TorsionLoadMoments add_face_load(std::vector<double>& F, const std::vector<std::pair<int, int>>& faces,
                                   const TorsionLoadFn& tau) const;
  // Harmonic response to an assembled per-radian load vector: one complex symmetric factorization per
  // frequency, one symbolic analysis per solver, frequencies optionally in parallel.  `timing`
  // receives the backend, factor size and the wall time per frequency.
  std::vector<TorsionResponse> harmonic_F(const std::vector<double>& F, const std::vector<double>& f_Hz,
                                          const TorsionHarmonicOptions& opt, json* timing = nullptr) const;
  // Mesh resolution of the structure at frequency f: elements per shear wavelength c_T/f, with the
  // element size h = longest vertex-to-vertex edge, per region (`region_of` names element e).
  json resolution(double f_Hz, const std::function<std::string(int)>& region_of) const;
  // The applied load as integrated (the P_p interpolant over the structure, full ring):
  // force_N = int g dV, torque_N_m = int g r dV, and the axial extent of its support.
  json load_moments(const std::vector<double>& load) const;

  const std::vector<TorsionRole>& roles() const { return roles_; }
  int n_free() const { return nfree_; }
  double solid_mass_kg() const { return mass_kg_; }
  double solid_volume_m3() const { return vol_m3_; }
  const json& diagnostics() const { return diag_; }

 private:
  const Space& sp_;
  std::vector<TorsionRole> roles_;
  std::vector<int> idx_;   // DOF -> free index (-1 if constrained)
  int nfree_ = 0;
  double mass_kg_ = 0, vol_m3_ = 0;
  json diag_;
  void assemble();
  // assembled per-radian matrices on the free DOFs
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace axf::axi

#endif  // AXIONFEM_TORSION_HPP
