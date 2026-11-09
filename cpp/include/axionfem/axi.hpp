// Axisymmetric magnetoquasistatic engine.
//
// Two scalar problems on the meridional half-plane (r >= 0, z):
//
//  * Poloidal fields (B_r, B_z) from azimuthal currents: unknown A = A_phi.
//    a(A, v) = int nu [ r dA/dz dv/dz + (1/r) d(rA)/dr d(rv)/dr ] dr dz
//            = int nu r (B[A] . B[v]) dr dz,   B[A] = (-dA/dz, dA/dr + A/r).
//    psi = r A is the flux function (flux per radian).  Superconducting bodies
//    are "masters": every DOF of the body has r A = psi_k, so the body excludes
//    the field exactly, and psi_k is either fixed (frozen fluxoid) or follows
//    from a prescribed net azimuthal current.  The outer semicircle carries an
//    exact spherical-harmonic Dirichlet-to-Neumann condition.  Nonlinear iron
//    is solved by Newton's method with an energy line search.
//
//  * Toroidal fields B_phi from poloidal currents: G = r H_phi equals the
//    poloidal current linked by the circle (r, z) divided by 2 pi.  In
//    magnetoquasistatics G is constant in every connected current-free
//    region ("levels"), vanishes inside superconductors, and varies only in
//    current-carrying regions (windings, readout ports), where it follows a
//    unit profile from the operator above with nu -> 1.  Levels are fixed by
//    windings (ampere-turns), ports (readout current) and superconductor
//    constraints (net poloidal current or enclosed toroidal flux).
//
// All energies are 3D values (J); 2D integrals are per radian unless noted.
#pragma once

#include <functional>
#include <mutex>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/materials.hpp"
#include "axionfem/spec.hpp"

namespace axf::axi {

using json = nlohmann::json;

inline constexpr int kAirAttr = 1;
inline constexpr int kAxisAttr = 900;
inline constexpr int kOuterAttr = 901;
inline constexpr int kCurveAttr0 = 1000;

// ---------------------------------------------------------------- mesh
struct MeshBundle {
  std::unique_ptr<mfem::Mesh> mesh;
  std::map<int, std::string> region_of_attr;  // element attribute -> region name ("air")
  std::map<std::string, int> attr_of_region;
  std::map<std::string, int> attr_of_curve;   // boundary attribute of each curve
  double domain_radius = 0;
  double domain_center_z = 0;  // z of the centre of the outer semicircle (spherical origin of the DtN)
  double device_radius = 0;    // farthest part of the device from (0, domain_center_z)
  json info;
};

struct MeshOptions {
  std::string workdir = ".";
  std::string mesher;  // path to axf-mesher (empty: search)
  int verbose = 0;
  std::string size_field_file;  // adaptive refinement: target sizes on a previous mesh (axf-mesher format)
};

MeshBundle build_mesh(const spec::Design& d, const MeshOptions& opt);
// Run a helper program (posix_spawn), stdout+stderr to `log`; returns its exit status (-1: signal).
int run_process(const std::vector<std::string>& args, const std::string& log);
std::string find_mesher(const std::string& hint = "");

// ---------------------------------------------------------------- space
// Values of shape functions and their physical derivatives at one point.
struct PointEval {
  double r = 0, z = 0, w = 0;  // physical point, quadrature weight * |J| (0 for non-quadrature points)
  int nd = 0;
  std::vector<double> N, dN;   // N[i], dN[2 i + k] (k = 0: d/dr, 1: d/dz)
  const mfem::IntegrationPoint* ip = nullptr;  // reference point (may be null)
  double shape(int i) const { return N[i]; }
  double dshape(int i, int k) const { return dN[2 * i + k]; }
};

// Reference-element values on an integration rule, shared by all elements.
struct RefTable {
  int nq = 0, nd = 0, ng = 0;
  std::vector<double> wq, N, dN, G, dG;  // field (N, dN) and geometry (G, dG) values, reference coords
  const mfem::IntegrationRule* ir = nullptr;
};

class Space {
 public:
  Space(mfem::Mesh& mesh, int order);
  mfem::Mesh& mesh;
  int order;
  std::unique_ptr<mfem::FiniteElementCollection> fec;
  std::unique_ptr<mfem::FiniteElementSpace> fes;
  std::vector<double> r, z;  // physical DOF coordinates
  double rtol = 0;           // |r| below this is "on the axis" (set from the geometry nodes first)
  int ndof() const { return fes->GetNDofs(); }
  const std::vector<int>& edofs(int e) const { return edofs_[e]; }
  // element neighbours across faces: (neighbour element, face index)
  std::vector<std::vector<std::pair<int, int>>> nbr;
  std::vector<int> face_bdr_attr;  // boundary/curve attribute of each face (0 if none)
  int quad_order() const { return 2 * order + 4; }
  const mfem::IntegrationRule& quad_rule() const;

  // Table for a persistent integration rule (thread-safe after first call).
  const RefTable& table(const mfem::IntegrationRule& ir) const;
  // Fast, thread-safe evaluation at point q of a table.
  void eval(int e, const RefTable& t, int q, PointEval& pe) const;
  // Evaluation at an arbitrary reference point (not thread-safe).
  void eval_at(int e, const mfem::IntegrationPoint& ip, PointEval& pe) const;

 private:
  std::vector<std::vector<int>> edofs_;
  const mfem::FiniteElement* field_fe_ = nullptr;
  const mfem::FiniteElement* geom_fe_ = nullptr;
  int ng_ = 0;
  std::vector<double> gx_;  // element geometry nodes: (x, y) pairs, ng_ per element
  mutable std::map<const mfem::IntegrationRule*, std::unique_ptr<RefTable>> tables_;
  void fill_geometry(int e, const double* G, const double* dG, const double* dNref, int nd, PointEval& pe) const;
};

inline void eval_point(const Space& sp, int e, const mfem::IntegrationPoint& ip, PointEval& pe) {
  sp.eval_at(e, ip, pe);
}

// ---------------------------------------------------------------- poloidal (A) problem
struct ARole {
  bool master = false;     // DOFs tied to a master value psi_m (field excluded)
  int master_id = -1;
  double nu = 0;           // linear reluctivity (m/H) for integrated elements
  const BHLaw* bh = nullptr;                 // nonlinear iron, overrides nu
  std::function<double(int, const PointEval&)> nu_at;  // optional nu(element, point) (linearized RF)
  double J = 0;            // source azimuthal current density (A/m^2)
};

struct Master {
  std::string name;
  bool fixed = false;  // fixed psi (Wb/rad) else prescribed net current
  double psi = 0;      // value when fixed
  double current = 0;  // prescribed net azimuthal current (A) when not fixed
};

// A closed superconducting circuit of stranded azimuthal windings in series.
// Its current I is an extra global unknown (J = turns * I / area in each part)
// fixed by conservation of the total flux linkage
//   lambda = sum_parts (turns/area) int 2 pi psi dr dz = flux_linkage.
struct CircuitCoil {
  std::string name;
  struct Part {
    std::vector<int> elements;
    double turns = 1, area = 0;
  };
  std::vector<Part> parts;
  double flux_linkage = 0;  // Wb-turns
};

struct ASolution {
  mfem::Vector u;                      // A_phi at all DOFs (T m)
  std::vector<double> y;               // reduced unknowns: free DOFs, then the unknown masters' psi
  std::vector<double> psi;             // master flux functions (Wb/rad)
  std::vector<double> current;         // master net azimuthal currents (A)
  std::vector<double> coil_current;    // circuit coil currents (A)
  // Reduced residual of this state (zero at a solution; nonzero when a state is evaluated on a
  // morphed mesh or with perturbed sources, which is how shape sensitivities are assembled).
  std::vector<double> residual_y, residual_circuit;
  double energy = 0;                   // stored magnetic energy (J, 3D; int H.dB for iron)
  double energy_quadratic = 0;         // 0.5 int B.H dV (J), equals energy for linear media
  json diagnostics;
};

// Mesh-height-to-thickness ratio h_perp/d below which a thin layer is flagged at element order p:
// max(4, p^2/4), from the measured onset of indefiniteness (docs/physics/thin_magnetic_layer.md).
int thin_layer_warn_ratio(int order);

// A thin magnetic layer on a curve (docs/physics/thin_magnetic_layer.md): linear mu_r > 1, thickness d.
struct ThinLayer {
  std::string name;
  int attr = 0;          // boundary attribute of the curve
  double mu_r = 1;
  double thickness = 0;  // m
  int curve_index = -1;  // index in the design's `curves` (error locations)
  std::string where() const { return curve_index >= 0 ? "/curves/" + std::to_string(curve_index) : "/curves"; }
};

class PoloidalProblem {
 public:
  PoloidalProblem(const Space& sp, std::vector<ARole> roles, std::vector<Master> masters,
                  std::map<int, int> sheet_master_by_attr = {});
  ~PoloidalProblem();
  // Exact spherical-harmonic DtN on the outer semicircle of radius `radius` about (r, z) = (0, center_z).
  void set_open_boundary(int outer_attr, double radius, int modes, double center_z = 0.0);
  void add_circuit_coil(CircuitCoil c) { coils_.push_back(std::move(c)); }
  // Reduced formulation with a background field: superconducting ties hold the TOTAL flux
  // function, i.e. r A_i = psi_m - psi_bg(i) at tied DOF i.
  void set_tie_offset(std::vector<double> psi_bg_at_dof) { tie_offset_ = std::move(psi_bg_at_dof); }
  // Thin magnetic layers: each DOF on a layer gets one copy per side of the layer (u then has ndof()
  // >= space().ndof() entries), and the layer adds a surface energy.  Call before solve()/evaluate().
  // region_of(e) names element e's region for error messages (optional).
  void set_thin_layers(const std::vector<ThinLayer>& layers,
                       const std::function<std::string(int)>& region_of = nullptr);
  // Background field B_bg(r, z) of a reduced (scattered-field) formulation: linear magnetic regions and
  // thin layers respond to it.  Without this call they do not see the background.
  void set_background_field(std::function<void(double, double, double&, double&)> Bbg) { bg_B_ = std::move(Bbg); }
  ASolution solve(int verbose = 0, double newton_tol = 1e-10, int max_newton = 60);
  // Per-layer diagnostics of a solution (empty array without layers).
  json thin_layer_report(const ASolution& s) const;
  // DOFs of element e in this problem's numbering (the space's, plus the layer copies).
  const std::vector<int>& edofs(int e) const { return ed_.empty() ? sp_.edofs(e) : ed_[e]; }
  int ndof() const { return ed_.empty() ? sp_.ndof() : static_cast<int>(orig_.size()); }

  const Space& space() const { return sp_; }
  const std::vector<ARole>& roles() const { return roles_; }
  const std::vector<Master>& masters() const { return masters_; }
  // B at a point of element e (zero inside master elements).
  void field(const ASolution& s, int e, const PointEval& pe, double& Br, double& Bz, double* A = nullptr) const;

  // ------------------------------------------------ shape sensitivities (linear problems)
  // The reduced unknowns y (free DOFs, then the flux function of each unknown master) and the
  // circuit currents have a mesh-independent meaning, because u = P(X) y + u0(X) rebuilds the
  // constrained DOFs from the node positions X.  A state solved on one mesh can therefore be
  // evaluated, and its residual measured, on a morphed mesh with the same topology, which is what
  // the discrete adjoint needs (see cpp/src/adjoint.cpp).
  // Outputs of a frozen state (energies, master reactions) on the current mesh; no solve.
  ASolution evaluate(const std::vector<double>& y, const std::vector<double>& coil_current = {}) const;
  // Reduced residual g = P^T (K u - b) - I_ext - C I_c and the circuit linkage residuals (empty y: y = 0).
  // Both vanish at the solution; their derivative with respect to a design parameter drives
  // the adjoint.
  void residual(const std::vector<double>& y, const std::vector<double>& coil_current, std::vector<double>& g,
                std::vector<double>& g_circuit) const;
  // Gradient with respect to (y, I_c) of the stored energy W = pi u^T K u [J].
  void energy_gradient(const std::vector<double>& y, std::vector<double>& g, std::vector<double>& g_circuit) const;
  // Value and (optionally) gradient with respect to y of the linear functional
  //   J = 2 pi int w(e, x) A_phi r dr dz,  with A_phi = psi_m / r inside master bodies.
  double linear_functional(const std::vector<double>& y, const std::function<double(int, const PointEval&)>& w,
                           std::vector<double>* grad = nullptr) const;
  // Solve the symmetric tangent system [[K, -C], [-C^T, 0]] [x; xc] = [b; bc] (linear problems only);
  // the factorization is built once and reused for further adjoint right-hand sides.
  void solve_tangent(const std::vector<double>& b, const std::vector<double>& bc, std::vector<double>& x,
                     std::vector<double>& xc) const;

 private:
  struct Reduction;   // constraint reduction u = P y + u0, DtN operator, circuit couplings
  struct Assembled;   // assembled operator, gradient and energies at one state
  struct Tangent;     // cached factorization for adjoint solves
  std::unique_ptr<Reduction> reduce() const;
  void assemble_at(const Reduction& R, const Eigen::VectorXd& u, bool want_K, Assembled& out) const;
  ASolution outputs(const Reduction& R, const Eigen::VectorXd& y, const Eigen::VectorXd& Icur) const;
  void residual_impl(const Reduction& R, const Eigen::VectorXd& y, const Eigen::VectorXd& Icur,
                     std::vector<double>& g, std::vector<double>& gc) const;
  mutable std::unique_ptr<Tangent> tangent_;
  const Space& sp_;
  std::vector<ARole> roles_;
  std::vector<Master> masters_;
  std::map<int, int> sheet_master_;
  std::vector<CircuitCoil> coils_;
  std::vector<double> tie_offset_;
  // thin layers (docs/physics/thin_magnetic_layer.md)
  struct LayerFace {
    int be = -1, layer = -1;
    int e[2] = {-1, -1};           // the elements on the two sides
    std::vector<int> dofs[2];      // the face DOFs of each side, in boundary-element order
  };
  std::vector<ThinLayer> layers_;
  std::vector<std::vector<int>> ed_;  // element DOFs with layer copies (empty: no layers)
  std::vector<int> orig_;             // DOF -> the space's DOF it copies (identity below space().ndof())
  std::vector<LayerFace> lfaces_;
  std::vector<int> layer_end_;        // space DOF -> layer whose unsplit end it is, else -1 (empty: no layers)
  std::function<void(double, double, double&, double&)> bg_B_;
  void layer_pivot_guard(double min_pivot) const;  // throws when the layered system is not positive definite
  int outer_attr_ = 0;
  double Rb_ = 0, center_z_ = 0;
  int modes_ = 0;
};

// ---------------------------------------------------------------- toroidal (G) problem
enum class HKind { Level, Excluded, Profile };

struct HRole {
  HKind kind = HKind::Level;
  double mu = 0;          // permeability (H/m) for level/profile elements
  std::function<double(int, const PointEval&)> mu_at;  // optional mu(element, point)
  int profile = -1;       // profile index
  int body = -1;          // excluded-body index (filled by the solver)
};

struct Profile {
  std::string name;
  enum class Kind { Winding, Port } kind = Profile::Kind::Winding;
  double current = 0;      // ampere-turns (winding) or readout current (port), A
  bool has_inside_point = false;
  double inside_r = 0, inside_z = 0;
};

// Constraint supplier for each excluded (superconducting/perfectly conducting)
// body enclosing a cavity: given the body's region names, return the
// constraint for its poloidal loops.
using BodyConstraintFn = std::function<spec::Constraint(const std::vector<std::string>& regions)>;

struct HSolution {
  std::vector<std::vector<int>> termA, termB;  // per profile: terminal faces (outside / inside)
  std::vector<double> G_elem;      // level G (A) of each Level element (0 otherwise)
  std::vector<double> La, Lb;      // per profile: G on the "outer/outside" and "inner/inside" terminals
  mfem::Vector unit;               // unit profiles H-values at DOFs (G-profile 0 -> 1)
  double energy = 0;               // J
  json diagnostics;
};

class ToroidalProblem {
 public:
  ToroidalProblem(const Space& sp, const MeshBundle& mb, std::vector<HRole> roles, std::vector<Profile> profiles,
                  BodyConstraintFn constraints);
  HSolution solve(int verbose = 0);
  void field(const HSolution& s, int e, const PointEval& pe, double& Hphi, double& mu) const;
  const std::vector<HRole>& roles() const { return roles_; }

 private:
  const Space& sp_;
  const MeshBundle& mb_;
  std::vector<HRole> roles_;
  std::vector<Profile> profiles_;
  BodyConstraintFn constraints_;
};

// Utility: connected components of elements satisfying `in` across faces for
// which `cross(face)` is true.  Returns component id per element (-1 if not in).
std::vector<int> components(const Space& sp, const std::function<bool(int)>& in,
                            const std::function<bool(int)>& cross, int& ncomp);

// Closed boundary loops of an element set: lists of faces; loop 0 is the
// outer loop (largest enclosed area).
std::vector<std::vector<int>> boundary_loops(const Space& sp, const std::vector<char>& in_set);

double element_area(const Space& sp, int e);
void element_center(const Space& sp, int e, double& r, double& z);

}  // namespace axf::axi
