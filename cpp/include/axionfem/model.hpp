// AxiModel: one axisymmetric design, meshed and solved, with post-processing.
#pragma once

#include <complex>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"
#include "axionfem/background.hpp"
#include "axionfem/conductor.hpp"
#include "axionfem/fullwave.hpp"
#include "axionfem/hts.hpp"
#include "axionfem/mech.hpp"
#include "axionfem/spec.hpp"

namespace axf {

using json = nlohmann::json;

struct SceneOptions {
  double view_scale = 1.3;   // field window relative to the device extent
  int subdiv = 1;            // 1: mesh vertices, 2: + edge midpoints (field and region triangulations)
  int outline_samples = 6;   // points per curved boundary edge in region outlines
  int contours = 28;         // number of flux-function contour levels
  bool fields = true;
};

struct SolveOptions {
  std::string workdir = "axf_run";
  std::string mesher;     // path to axf-mesher (auto if empty)
  int verbose = 1;
  int order = 0;          // FE order override (0: design.mesh.order)
  double pickup_current = 1.0;  // A
  std::string size_field_file;  // adaptive refinement (set by the adaptive loop)
  // Estimate (and, for linear designs, remove) the numerical self-force of each powered azimuthal
  // winding.  Costs one extra linear solve per such winding (see "winding self-force floor" in model.cpp).
  bool winding_self_force = true;
  // Samples per body in forces.distributions.  Consecutive pieces are merged down to this many, so
  // the net force is unchanged whatever the value; it only sets how finely the load is resolved and
  // how big results.json gets (~160 bytes per sample per body).  0 keeps the net forces and the
  // closure check but writes no sample arrays.
  int force_samples = 200;
  // Sensitivity evaluations only need the figures of merit: skip peak fields, forces (and their
  // extra self-force solves), electrostatics, RF losses and the port capacitance, which together
  // cost more than the solve itself.
  bool minimal_post = false;
  // Generate the mesh (workdir/mesh.msh or workdir/mesh3d.msh, with *_in.json and *_info.json) and stop:
  // no solve.  Used by `axf mesh` and the COMSOL export (docs/comsol_export.md).
  bool mesh_only = false;
};

class AxiModel {
 public:
  explicit AxiModel(spec::Design d);
  ~AxiModel();
  void mesh(const SolveOptions& o);
  void solve(const SolveOptions& o);
  // mesh + solve (with adaptive re-meshing when design.mesh.adapt.cycles > 0)
  void run(const SolveOptions& o);
  // Zienkiewicz-Zhu indicators of the static and pickup fields per element (relative, squared, summing to the
  // estimated relative energy-norm error squared); recovery per region (material interfaces are not smoothed)
  std::vector<double> error_indicators() const;

  // ------------------------------------------------ shape sensitivities (see cpp/src/adjoint.cpp)
  // Use a prepared mesh (e.g. morphed from another design) instead of running the mesher.
  void set_mesh(axi::MeshBundle mb, int order = 0);
  // Evaluate the given poloidal states instead of solving for them: solve() then post-processes a
  // frozen state on this mesh, which is the partial derivative part of the adjoint.  Linear only.
  void freeze_poloidal(std::vector<double> y_lab, std::vector<double> ic_lab, std::vector<double> y_pick,
                       std::vector<double> ic_pick);

  const json& results() const { return results_; }
  const spec::Design& design() const { return d_; }
  const axi::MeshBundle& mesh_bundle() const { return mb_; }
  const axi::Space& space() const { return *sp_; }
  bool solved() const { return solved_; }
  const axi::PoloidalProblem* poloidal_lab() const { return a_lab_.get(); }
  const axi::PoloidalProblem* poloidal_pickup() const { return a_pick_.get(); }
  const axi::ToroidalProblem* toroidal_lab() const { return h_lab_.get(); }
  const axi::ToroidalProblem* toroidal_pickup() const { return h_pick_.get(); }
  const axi::ASolution& poloidal_lab_solution() const { return a_lab_sol_; }
  const axi::ASolution& poloidal_pickup_solution() const { return a_pick_sol_; }
  const axi::HSolution& toroidal_lab_solution() const { return h_lab_sol_; }
  const axi::HSolution& toroidal_pickup_solution() const { return h_pick_sol_; }
  double pickup_current() const { return I_pick_; }

  // Field sampling at physical points; points outside the mesh get NaN.
  struct Sample {
    double Br = 0, Bz = 0, Bphi = 0, psi = 0, A = 0;  // static (lab) field, flux function, A_phi
    double pBr = 0, pBz = 0, pBphi = 0, ppsi = 0;      // pickup-mode field per ampere, its flux function
    int region = -1;                                    // element attribute (-1 outside)
  };
  std::vector<Sample> sample(const std::vector<double>& r, const std::vector<double>& z) const;

  // ---- mechanics/thermal (cpp/src/model_mech.cpp) ----
  // Displacement, stress and temperature at physical points.  Points outside the structure (or
  // outside the thermal domain) report solid = false / thermal = false and zeros; points outside
  // the mesh get NaN.
  struct MechSample {
    double ur = 0, uz = 0;                      // displacement [m]
    double s_rr = 0, s_zz = 0, s_tt = 0, s_rz = 0, von_mises = 0;  // stress [Pa]
    double T_K = 0;                             // temperature [K]
    int region = -1;
    bool solid = false, thermal = false;
  };
  std::vector<MechSample> sample_mech(const std::vector<double>& r, const std::vector<double>& z) const;
  bool has_mechanics() const { return elast_ != nullptr; }

  // ---- vibrations: axisymmetric m = 0 torsional dynamics (cpp/src/model_torsion.cpp) ----
  // One JSON in, one JSON out, so that a study can drive it without new bindings:
  //   {"nmodes": n, "shapes": bool, "lanczos_extra": k,
  //    "load": [one value per DOF], "freqs_Hz": [...], "eta": loss factor,
  //    "regions": {"<name>": {"mu_Pa"|"E_Pa"(+"nu"), "density", "exclude"}},
  //    "dofs": bool, "verbose": level}   (other keys are errors; eta >= 0, nmodes >= 0)
  // The response carries the structure's mass, the eigenfrequencies with their residuals, the
  // inertia completeness check and the modal participation of the load, the applied load's
  // integrals, and the generalised mobility Y = -i omega q (time dependence e^{-i omega t}) of the
  // conjugate pair (load scale, q = int g u_phi dV).  See axionfem/torsion.hpp.
  json torsion(const json& request) const;
  bool has_thermal() const { return cond_ != nullptr; }
  // ---- end mechanics/thermal ----

  // Evaluate fields in element e at reference point ip (serial) or at a prepared point (thread-safe).
  Sample eval(int e, const mfem::IntegrationPoint& ip) const;
  Sample eval_pe(int e, const axi::PointEval& pe) const;

  // Critical-state (HTS screening-current) quantities of the final state at physical points:
  // engineering current density, local engineering Jc, azimuthal electric field and the total field.
  // `hts` is false outside the critical-state regions (where J and Jc are 0).
  struct HTSSample {
    double J = 0, Jc = 0, E = 0, Br = 0, Bz = 0, A = 0;
    bool hts = false;
    int region = -1;
  };
  std::vector<HTSSample> hts_sample(const std::vector<double>& r, const std::vector<double>& z) const;
  bool has_hts() const { return hts_cs_ != nullptr; }

  // Region name of an element attribute.
  std::string region_name(int attr) const;

  // Full-wave RF fields per readout ampere at fullwave.scene_frequency (complex amplitudes, e^{-i omega t}):
  // B and E as (r, phi, z) components. False when no full-wave field is stored.
  bool fw_fields(int e, const axi::PointEval& pe, axi::cplx B[3], axi::cplx E[3]) const;
  double fw_scene_frequency() const { return fw_sol_ ? fw_sol_f_ : 0.0; }

  // ---- cavity eigenmode fields (cpp/src/cavity_modes.cpp; designs with a `cavity` section) ----
  // Fields of eigenmode `index` of `family` ("TE"/"TM"; the order of results.fullwave.eigenmodes[family].modes),
  // normalized to int eps_r |E|^2 dV = 1 over the cavity [E in m^-3/2, H in the same units / ohm], real amplitudes
  // (r, phi, z) of the complex fields (i E, H) (e^{-i omega t}; E and H are 90 degrees apart). The overall sign is
  // fixed so that the largest component of E at the point of largest |E| is positive. `flux` is r H_phi (TM, E lines
  // are its contours) or r E_phi / omega = r A_phi (TE, H lines), in the same normalization.
  struct CavityModeField {
    std::string family;
    int index = 0;
    double f_Hz = 0, norm = 0;  // norm: factor applied to the eigenvector
    std::vector<double> r, z, E[3], H[3], flux, eps_r, mu_r;
    std::vector<int> elem;          // element of each point (-1: outside the cavity, fields NaN)
    std::vector<uint32_t> tris;     // mesh(): sub-triangles (3 point indices each)
  };
  int cavity_mode_count(const std::string& family) const;
  // On the cavity mesh: a lattice of (subdiv + 1)(subdiv + 2)/2 points per element (element-wise, so points on
  // element edges are repeated with each element's value), subdiv^2 sub-triangles per element.
  CavityModeField cavity_mode_mesh(const std::string& family, int index, int subdiv = 2) const;
  // At physical points (r, z); NaN outside the cavity.
  CavityModeField cavity_mode_sample(const std::string& family, int index, const std::vector<double>& r,
                                     const std::vector<double>& z) const;

  // Viewer scene (schema axionfem.scene/1, see docs/scene_format.md).
  json scene(const SceneOptions& o = SceneOptions()) const;
  // VTK XML unstructured grid of the (subdivided) field mesh.
  void write_vtu(const std::string& path, const SceneOptions& o = SceneOptions()) const;

 private:
  bool frozen_ = false;  // poloidal states given from outside (sensitivity evaluations)
  std::vector<double> frozen_y_lab_, frozen_ic_lab_, frozen_y_pick_, frozen_ic_pick_;
  bool iron_toroidal_ = false;  // nonlinear iron magnetized by the static toroidal field
  void post(const SolveOptions& o);
  // Mesh check for a solver-failure message: the worst elements and the fix. For a warning (thrown = false) the
  // worst five also go to mb_.info["worst_elements"] (results mesh.info.worst_elements); a thrown error has none.
  std::string mesh_note(bool thrown = false);
  // Peak fields, forces, RF losses, port capacitance and winding linkages (everything the figures
  // of merit do not need); skipped when SolveOptions::minimal_post is set.
  void post_fields(const SolveOptions& o, double Lp, json& checks, std::map<std::string, double>& peak_region,
                   double& peak_all, std::map<std::string, double>& sc_peak,
                   std::map<std::string, double>& sc_peak_regular);
  void fullwave_analysis(const SolveOptions& o);
  void hts_analysis(const SolveOptions& o);
  std::unique_ptr<axi::CriticalState> hts_cs_;
  std::unique_ptr<axi::CriticalState::State> hts_state_;
  std::vector<std::unique_ptr<JcLaw>> hts_laws_;  // outlive the HTSMaterial pointers
  void electrostatics_analysis(const SolveOptions& o);  // cpp/src/axi_electro.cpp
  std::function<double(int, const axi::PointEval&)> mu_rf_iron_;  // RF (secant) permeability of iron
  std::unique_ptr<axi::AxiWave> fw_te_, fw_tm_;
  std::map<std::string, std::vector<axi::Eigenmode>> fw_modes_;  // eigenvectors per family (cavity analysis)
  std::unique_ptr<axi::AxiWave::Solution> fw_sol_;  // stored RF solution (scenes)
  std::string fw_sol_family_;
  double fw_sol_f_ = 0;
  std::vector<std::complex<double>> fw_eps_;         // complex permittivity per element at the scene frequency
  spec::Design d_;
  axi::MeshBundle mb_;
  std::unique_ptr<axi::Space> sp_;
  std::unique_ptr<axi::PoloidalProblem> a_lab_, a_pick_;
  std::unique_ptr<axi::ToroidalProblem> h_lab_, h_pick_;
  axi::ASolution a_lab_sol_, a_pick_sol_;
  axi::HSolution h_lab_sol_, h_pick_sol_;
  std::vector<std::unique_ptr<BHLaw>> bh_;
  double I_pick_ = 1.0;
  std::vector<double> coil_J_;  // per element: current density of persistent-circuit windings
  BackgroundField bg_;
  // total static poloidal field and A_phi (reduced solution + background) at a point of element e
  void static_poloidal(int e, const axi::PointEval& pe, double& Br, double& Bz, double& A) const;
  // ---- mechanics/thermal (cpp/src/model_mech.cpp) ----
  void mechanics_thermal(const SolveOptions& o);
  std::unique_ptr<axi::Elasticity> elast_;
  std::unique_ptr<axi::Conduction> cond_;
  axi::ElasticSolution elast_sol_;
  axi::ThermalSolution cond_sol_;
  std::vector<std::unique_ptr<axi::KTable>> ktables_;
  // ---- end mechanics/thermal ----
  json results_;
  bool solved_ = false;
};

// Parse, mesh, solve; returns the results document.
json run_design(const json& design_doc, const SolveOptions& o);

}  // namespace axf
