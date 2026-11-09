// Axisymmetric structural mechanics and steady heat conduction on the meridional mesh.
//
// Both problems live on the same H1 space as the magnetoquasistatic engine (axi::Space) and are
// assembled over the *solid* elements only: vacuum, air, ports and gaps are excluded, so the
// structure is whatever the design says is made of matter.  Every integral carries the axisymmetric
// weight r (2 pi r dr dz for a volume); the 2D integrals below are per radian unless a name says
// otherwise.
//
//  * Elasticity (small strain, isotropic, torsion-free): unknowns u = (u_r, u_z) with
//      eps = [eps_rr, eps_zz, eps_tt, gamma_rz] = [du_r/dr, du_z/dz, u_r/r, du_r/dz + du_z/dr],
//      sigma = lambda tr(eps) I + 2 mu eps - (3 lambda + 2 mu) eps_th I,
//    i.e. a uniform stress-free (thermal) strain eps_th enters as a prestress.  On the symmetry
//    axis u_r = 0 is imposed exactly (a Dirichlet condition), and u_r/r is replaced by its limit
//    du_r/dr at quadrature points that fall on r = 0.  Touching solid regions share mesh nodes and
//    are therefore *bonded* (no contact, no sliding, no gaps).
//
//    Loads: a body-force density callback (the Lorentz force J x B in windings), pressure loads on
//    mesh faces of the structure (the Maxwell pressure B^2/2 mu0 on superconductor surfaces, or a
//    prescribed pressure), and the thermal prestress.  Supports fix u_r and/or u_z on whole
//    regions.  Each connected solid component without an axial support keeps one rigid axial
//    translation; the solver pins one of its axial degrees of freedom (exact when the component's
//    net axial load vanishes, which the solution reports) and afterwards shifts the component so
//    that its volume-weighted mean u_z is zero.
//
//    Reactions are the residual K u - F at the constrained degrees of freedom, i.e. the force the
//    supports apply to the structure.  Axial equilibrium (sum of applied loads + sum of reactions
//    = 0) is an exact identity of the discrete system and is reported as a closure check.
//
//  * Conduction: div(k grad T) + q = 0 with k constant or tabulated k(T) (piecewise linear).  A
//    k(T) table is handled by a Picard iteration on k evaluated at the previous temperature.
//    Fixed temperatures are Dirichlet conditions on regions or on zero-thickness curves; every
//    other boundary of the thermal domain is adiabatic (the natural condition).  The heat leaving
//    through the fixed temperatures is -2 pi sum_fixed (K T - F)_i and equals the generated power
//    exactly for the discrete system: the balance is a check on the assembly and the solver.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"

namespace axf::axi {

using json = nlohmann::json;

// Connected components of the elements selected by `in`, joined through *shared degrees of
// freedom*: exactly the connectivity of the bonded finite-element assembly (two bodies that touch
// at a single node are one component).  Returns the component id of each element (-1 if not in).
std::vector<int> solid_components(const Space& sp, const std::function<bool(int)>& in, int& ncomp);

// ---------------------------------------------------------------- elasticity

struct ElasticRole {
  bool solid = false;         // element belongs to the structure
  double lambda = 0, mu = 0;  // Lame constants [Pa]
  double eps_th = 0;          // isotropic stress-free strain (negative for a cooldown contraction)
  bool body_force = false;    // evaluate the body-force callback in this element
  int support = -1;           // support group index (-1: free)
};

struct SupportGroup {
  std::string name;
  bool fix_r = true, fix_z = true;
};

// A pressure load on one mesh face of the structure.  The traction on the structure is t = -p n
// with n the unit normal of `solid_elem` pointing away from it, so p > 0 compresses.
struct PressureFace {
  int face = -1;
  int solid_elem = -1;   // element of the structure that carries the traction
  int other_elem = -1;   // element on the other side (the field side for magnetic pressure)
  int load = -1;         // opaque tag handed back to the pressure callback
};

// p [Pa] at the integration point FT is currently set to.
using PressureFn = std::function<double(const PressureFace&, mfem::FaceElementTransformations&)>;
// body-force density [N/m^3] at a point of element e
using BodyForceFn = std::function<void(int e, const PointEval& pe, double& fr, double& fz)>;

struct ElasticSolution {
  std::vector<double> ur, uz;               // displacement at every DOF [m]
  std::vector<double> reaction_Fz_N;        // per support group: total axial force [N]
  std::vector<double> reaction_Fr_per_rad;  // per support group: radial resultant [N/rad]
  double pin_Fz_N = 0;                      // axial force at the rigid-mode pins (should be ~0)
  double applied_Fz_N = 0;                  // total applied axial load [N]
  double applied_Fr_per_rad = 0;            // applied radial resultant [N/rad]
  double load_scale_N = 0;                  // int (|f_r| + |f_z|) dV + faces: applied-load magnitude
  double axial_load_scale_N = 0;            // int |f_z| dV + faces: AXIAL applied-load magnitude [N]
  double assembly_scale_N = 0;              // 2 pi |F|_1: the scale of the sums in the closure check
  double closure_Fz = 0;                    // |applied + reactions| / applied-load scale
  double closure_Fz_N = 0;                  // |applied + reactions| [N]
  double closure_Fz_round_off = 0;          // |applied + reactions| / assembly_scale_N
  // per connected component without an axial support: its net applied axial load [N]
  std::vector<double> free_component_Fz_N;
  std::vector<int> component_of_elem;       // component id of each element (-1: not solid)
  json diagnostics;
};

class Elasticity {
 public:
  Elasticity(const Space& sp, std::vector<ElasticRole> roles, std::vector<SupportGroup> supports);
  void set_body_force(BodyForceFn f) { body_ = std::move(f); }
  void set_pressure(std::vector<PressureFace> faces, PressureFn p);
  ElasticSolution solve(int verbose = 0);

  // Displacement and stress at a point of element e (zero outside the structure).
  void displacement(const ElasticSolution& s, int e, const PointEval& pe, double& ur, double& uz) const;
  void stress(const ElasticSolution& s, int e, const PointEval& pe, double& srr, double& szz, double& stt,
              double& srz) const;
  static double von_mises(double srr, double szz, double stt, double srz);
  const std::vector<ElasticRole>& roles() const { return roles_; }

 private:
  const Space& sp_;
  std::vector<ElasticRole> roles_;
  std::vector<SupportGroup> sup_;
  BodyForceFn body_;
  std::vector<PressureFace> pfaces_;
  PressureFn pfn_;
};

// ---------------------------------------------------------------- conduction

// Piecewise-linear k(T); clamped to the end values outside the tabulated range.
struct KTable {
  std::vector<double> T, k;
  double at(double t) const;
  bool empty() const { return T.empty(); }
};

struct ThermalRole {
  bool solid = false;          // element belongs to the thermal domain
  double k = 0;                // constant conductivity [W/(m K)]
  const KTable* kt = nullptr;  // k(T) table; overrides k when set
  double q = 0;                // volumetric heat source [W/m^3]
  bool fixed = false;          // prescribed temperature on the whole element
  double T_fix = 0;            // [K]
};

struct ThermalSolution {
  std::vector<double> T;      // [K] at every DOF (0 outside the thermal domain)
  double generated_W = 0;     // 2 pi int q r dr dz
  double outflow_W = 0;       // heat leaving through the prescribed temperatures (net, can cancel)
  double heat_removed_W = 0;  // heat leaving at prescribed temperatures that absorb heat [W]
  double heat_added_W = 0;    // heat entering at prescribed temperatures that supply heat [W]
  double balance_rel = 0;     // |generated - outflow| / |generated|
  double T_min = 0, T_max = 0;
  int picard_iterations = 0;
  double picard_change = 0;
  bool converged = true;
  json diagnostics;
};

class Conduction {
 public:
  // fixed_curve_attr: boundary attribute of a zero-thickness curve -> prescribed temperature [K].
  Conduction(const Space& sp, std::vector<ThermalRole> roles, std::map<int, double> fixed_curve_attr = {});
  ThermalSolution solve(int verbose = 0, double tol = 1e-12, int max_picard = 100);
  double value(const ThermalSolution& s, int e, const PointEval& pe) const;
  // conductive heat flux q = -k grad T [W/m^2]
  void flux(const ThermalSolution& s, int e, const PointEval& pe, double& qr, double& qz) const;
  const std::vector<ThermalRole>& roles() const { return roles_; }

 private:
  const Space& sp_;
  std::vector<ThermalRole> roles_;
  std::map<int, double> fixed_curve_;
};

}  // namespace axf::axi
