// Model3D: full 3D magnetoquasistatic design, A-formulation with Nedelec
// elements (MFEM) and hypre AMS.
//
// Static problem:  curl nu curl A = J  in the meshed domain (nonlinear iron by
// Newton's method), n x A = 0 on the outer sphere.  Superconducting bodies are
// meshed but all their degrees of freedom are eliminated: inside a body
// A = sum_j alpha_j h_j, where h_j are closed, non-exact cochains spanning
// H^1(body) (cohomology), so that B = curl A = 0 in the body and only the
// fluxoids alpha_j remain.  Each alpha_j is either fixed (frozen flux) or
// determined by a zero net-current condition (Schur complement).
//
// Pickup problem: the readout current I flows through the port region
// (conduction between the two terminal patches where the port touches
// conductors); superconductors and RF conductors are eliminated bodies with
// conserved fluxoids.  L_p = 2 W_I / I^2 and Psi = (1/I) int A_I . B_lab.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/materials.hpp"
#include "axionfem/model.hpp"
#include "axionfem/spec.hpp"

namespace axf {

class Model3D {
 public:
  explicit Model3D(spec::Design d);
  ~Model3D();
  void run(const SolveOptions& o);
  const json& results() const { return results_; }
  const spec::Design& design() const { return d_; }
  // B (static) and pickup field per ampere at points; NaN outside.
  struct Sample {
    double B[3] = {0, 0, 0}, pB[3] = {0, 0, 0};
    int region = -1;
  };
  std::vector<Sample> sample(const std::vector<double>& x, const std::vector<double>& y,
                             const std::vector<double>& z) const;
  json scene(const SceneOptions& o = SceneOptions()) const;
  void write_paraview(const std::string& dir) const;
  // Solved fields on disk: the serial mesh, the gathered A_lab/A_pick and the model state (MFEM mesh/grid-function
  // formats plus state.json). save() is collective; rank 0 writes. load() needs one process and restores results,
  // scenes and sampling without solving again (e.g. after an MPI run of `axf solve --save`).
  void save(const std::string& dir) const;
  static std::unique_ptr<Model3D> load(const std::string& dir);

  struct Impl;

 private:
  // time-harmonic Maxwell (design key `fullwave`): complex E field, port feed, absorbing shell (model3d_fullwave.cpp)
  void fullwave_analysis(const SolveOptions& o);
  // electrostatics (design key `electrostatics`): H1 potential solve, Maxwell capacitance matrix of the
  // conductor bodies, pairwise capacitances, readout-gap capacitance (model3d_electro.cpp)
  void electrostatics_analysis(const SolveOptions& o);
  // mechanics and heat in 3D (design keys `mechanics`, `thermal`): linear elasticity with Lorentz, Maxwell-pressure, prescribed-
  // pressure and cooldown loads; steady conduction with k(T) (model3d_mech.cpp)
  void mechanics_thermal_analysis(const SolveOptions& o);
  void run_once(const SolveOptions& o);  // mesh + solve + post (run() adds adaptive re-meshing)
  // Zienkiewicz-Zhu indicators (squared, relative) per serial element of the static and pickup fields
  std::vector<double> error_indicators() const;
  spec::Design d_;
  std::unique_ptr<Impl> p_;
  json results_;
};

// Dispatch on design.symmetry: AxiModel or Model3D.
// MPI rank of this process (0 without MPI or before MPI is initialized)
int mpi_rank();

json run_any(const json& design_doc, const SolveOptions& o);
// Mesh a design without solving (axisymmetric or 3D): writes o.workdir/mesh.msh (2D) or mesh3d.msh (3D)
// with the mesher input (*_in.json: resolved shapes and the physical tag of every region) and its report
// (*_info.json). Returns {"ok", "symmetry", "mesh", "mesher_input", "mesher_info"}.
json mesh_any(const json& design_doc, const SolveOptions& o);

}  // namespace axf
