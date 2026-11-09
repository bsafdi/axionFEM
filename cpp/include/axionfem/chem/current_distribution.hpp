// Primary and secondary current distribution (beta; formulation: docs/guides/chemistry.md).
//
// Electrolyte potential phi on a 2D triangle mesh (planar, or the axisymmetric meridian (r, z) with every integral
// weighted by r: "per radian"), Lagrange P1/P2 (P2 optionally isoparametric), conductivity kappa,
//     -div(kappa grad phi) = f,   kappa grad phi . n_out = sum_j i_j(eta_j) + g  on kinetic electrodes,
//     phi = phi_s - E_eq on primary electrodes, kappa grad phi . n_out = q on insulators,
// eta_j = phi_s - phi - E_eq,j, anodic currents positive.  Potentiostatic (phi_s given) or galvanostatic control
// (phi_s unknown with the bordered row int (sum_j i_j + g) dA = I; a galvanostatic PRIMARY electrode is a floating
// conductor: its dofs are tied to one unknown and the tie row is its reaction current = I).  Optional half-space
// even-mode DtN on an outer arc (the discrete map is the one of the disk cases).
//
// Newton: the criterion of the verification cases (iterate to stagnation, step limiter on |d eta|, halving
// line search, exponent clip).  The Jacobian, bordered rows included, is symmetric positive definite for
// monotone kinetics (di/deta > 0) and a grounded problem (a potentiostatic electrode or the DtN exterior; refused
// otherwise), so the linear solver is a sparse LDL^T (Eigen) with one step of iterative
// refinement.  Every result carries diagnostics; raw quadrature data are exposed for the check module.
//
// No axf::Design, Model or JSON here; the Python layer (python/axionfem/chem) builds the inputs.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mfem {
class Mesh;
class FiniteElementCollection;
class FiniteElementSpace;
}  // namespace mfem

namespace axf::chem {

constexpr double kFaraday = 96485.33212;    // C/mol (exact, SI 2019)
constexpr double kGasConstant = 8.314462618;  // J/(mol K) (exact)

struct Reaction {
  enum class Kind { BV, Linear, TafelAnodic, TafelCathodic };
  Kind kind = Kind::BV;
  double i0 = 0.0;      // A/m^2
  double E_eq = 0.0;    // V
  double alpha_a = 0.5, alpha_c = 0.5;
  double A = 0.0;       // Tafel slope V/decade (> 0 anodic, < 0 cathodic)
};
Reaction::Kind reaction_kind_from_string(const std::string& s);  // "bv", "linear", "tafel_a", "tafel_c"

// i(eta) and di/deta of one reaction; exponent arguments are clipped at +-clip (clipped set when active).
double reaction_current(const Reaction& r, double eta, double f, double clip, double* didEta, bool* clipped);

struct MeshInput {
  std::vector<double> vertices;    // nv x 2
  std::vector<int> triangles;      // ne x 3
  std::vector<int> element_attr;   // ne (>= 1)
  std::vector<int> boundary;       // nb x 2
  std::vector<int> boundary_attr;  // nb (>= 1)
  // P2 only: positions of edge nodes that differ from the straight midpoint (isoparametric geometry)
  std::vector<int> edge_node_vertices;  // m x 2 (vertex pair of the edge)
  std::vector<double> edge_node_xy;     // m x 2
};

struct ElectrodeSpec {
  std::string name;
  std::vector<int> attrs;
  bool primary = false;            // Dirichlet phi = phi_s - E_eq (or a floating tie if galvanostatic)
  double E_eq = 0.0;               // primary electrodes only
  std::vector<Reaction> reactions; // kinetic electrodes
  bool galvanostatic = false;
  double phi_s = 0.0;              // potentiostatic value, or the initial value of a galvanostatic electrode
  double current = 0.0;            // galvanostatic: 3D total (axisymmetric) or per unit depth (planar), anodic > 0
  std::vector<double> g;           // optional manufactured source at boundary_points(attrs) (MMS)
};

struct NewtonOptions {
  double stag = 1e-13;     // stop once relres <= stag and an iteration did not halve it (or relres == 0)
  int max_iter = 60;
  double gate = 1e-12;     // converged = final relres <= gate and no clamp active
  double limiter_rt_f = 10.0;  // max |d eta| per step, in units of RT/F (<= 0: no limiter)
  double ls_min = 1.0 / 64.0;  // halving line search down to this step
  double clip = 200.0;     // exponent clip
  double temperature = 0.0;  // K; 0 = the problem's temperature (set_temperature, default 298.15 K)
  int refinement_steps = 1;  // iterative refinement of each linear solve
  // initial guess: "zero" (phi = 0, galvanostatic phi_s as given: the default start), "state" (warm start from
  // set_state(): continuation in the applied potential), "primary" (the primary distribution, eta = 0 on every
  // electrode: robust for large cell voltages)
  std::string initial = "zero";
};

struct SolveReport {
  bool converged = false;
  double relative_residual = 0.0;
  int iterations = 0;
  std::vector<double> residual_history;  // relres of the initial state and after every step
  std::vector<double> step_lengths;
  std::string stop;  // "stagnation", "zero", "max_iter", "linear_solver_failed"
  int limited_steps = 0;
  int line_search_halvings = 0;
  bool clamp_active = false;
  double field_residual = 0.0, constraint_residual = 0.0;
  int unknowns = 0;
  long factor_nonzeros = 0;
  double seconds = 0.0;
  std::string initial;
  // kinetic electrodes whose current is below 1e3 eps x the field residual scale on their dofs: not resolved
  std::vector<std::string> unresolved_currents;
};

// raw quadrature data: w carries the (surface) Jacobian but NOT the axisymmetric r
struct DomainQuadrature {
  Eigen::MatrixXd x;       // n x 2
  Eigen::VectorXd w;
  Eigen::VectorXd phi;
  Eigen::MatrixXd grad;    // n x 2
  Eigen::VectorXi element; // element of each point
};
struct BoundaryQuadrature {
  Eigen::MatrixXd x;
  Eigen::VectorXd w;
  Eigen::VectorXd phi;
  Eigen::MatrixXd grad;    // one-sided (adjacent element) gradient
  Eigen::MatrixXd normal;  // unit outward normal
  Eigen::VectorXi bdr_element;
};

class CurrentDistribution {
 public:
  using SpMat = Eigen::SparseMatrix<double>;
  CurrentDistribution(const MeshInput& mesh, int order, bool axisymmetric, int domain_degree, int boundary_degree);
  ~CurrentDistribution();
  CurrentDistribution(const CurrentDistribution&) = delete;
  CurrentDistribution& operator=(const CurrentDistribution&) = delete;

  int order() const { return order_; }
  bool axisymmetric() const { return axi_; }
  int domain_degree() const { return deg_dom_; }
  int boundary_degree() const { return deg_bdr_; }

  // ---- coefficients (before solve; quadrature points at the assembly degrees)
  Eigen::MatrixXd domain_points() const;
  Eigen::MatrixXd boundary_points(const std::vector<int>& attrs) const;
  void set_conductivity_by_attribute(const std::map<int, double>& kappa);
  void set_conductivity_at_points(const Eigen::VectorXd& kappa);  // aligned with domain_points()
  void set_source_at_points(const Eigen::VectorXd& f);
  void set_insulator_flux(int attr, const Eigen::VectorXd& q);    // aligned with boundary_points({attr})
  int add_electrode(const ElectrodeSpec& e);
  void set_exterior_halfspace(int attr, double Rb, int modes, double kappa);

  // ---- solve; its options (clip, ...) are kept for every later evaluation (residual, jacobian, relres,
  // newton_update, residual_full).  Temperature has one source of truth, the problem's T: set_temperature(T), or a
  // solve whose options name a temperature (which then becomes the problem's T); every solve and evaluation uses it
  SolveReport solve(const NewtonOptions& opt);
  void set_temperature(double T);
  // change an electrode's control value after set-up (continuation): phi_s of a potentiostatic electrode, the current
  // of a galvanostatic one (3D total / per unit depth)
  void set_control(int e, double value);
  double temperature() const { return temperature_; }
  const SolveReport& report() const { return report_; }

  // ---- state z = [unknown field dofs, tie values, galvanostatic phi_s]
  int num_unknowns();
  Eigen::VectorXd state() const { return z_; }
  void set_state(const Eigen::VectorXd& z);
  Eigen::VectorXd initial_state();
  Eigen::VectorXd residual(const Eigen::VectorXd& z);   // z-space residual incl. tie and bordered rows
  SpMat jacobian(const Eigen::VectorXd& z);
  Eigen::VectorXd newton_update(const Eigen::VectorXd& z);  // z + dz, one undamped Newton step
  double relres(const Eigen::VectorXd& z);
  Eigen::VectorXd phi_dofs() const;   // full dof vector of the current state
  std::vector<double> phi_s() const;  // metal potential per electrode (current state)
  std::vector<std::string> electrode_names() const;

  // ---- raw data
  DomainQuadrature domain_quadrature(int degree) const;
  BoundaryQuadrature boundary_quadrature(const std::vector<int>& attrs, int degree) const;
  // weak-form residual per dof (all dofs, per radian), without the kinetic term of electrode `exclude` (-1: none)
  Eigen::VectorXd residual_full(int exclude) const;
  std::vector<int> attr_dofs(const std::vector<int>& attrs) const;
  std::vector<int> electrode_dofs(int e) const;
  std::vector<int> electrode_attrs(int e) const;
  SpMat boundary_mass(const std::vector<int>& attrs) const;  // r-weighted in axisymmetry, all dofs
  Eigen::MatrixXd mesh_vertices() const;
  Eigen::MatrixXd dof_coordinates() const;  // physical position of every (nodal) dof
  Eigen::MatrixXi mesh_elements() const;
  int num_elements() const;
  int num_dofs() const;
  double h_max() const;
  double area() const;
  std::map<std::string, double> diagnostics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  int order_, deg_dom_, deg_bdr_;
  bool axi_;
  Eigen::VectorXd z_;
  SolveReport report_;
  double temperature_ = 298.15;
  NewtonOptions eval_opt_ = [] { NewtonOptions o; o.temperature = 298.15; return o; }();
  void check_state(const Eigen::VectorXd& z, const char* where);
};

}  // namespace axf::chem
