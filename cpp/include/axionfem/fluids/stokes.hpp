// Taylor–Hood Stokes on an mfem::ParMesh (beta; formulation: docs/guides/fluids.md).
//
//   -div(2 nu eps(u)) + grad p = f      (SymmetricStress)      or      -div(nu grad u) + grad p = f   (Laplacian)
//                        div u = 0
//
// Weak form, symmetric saddle point (both forms):  a(u,v) - (p, div v) = (f, v) + <g, v>_traction,  -(q, div u) = 0
//   K = [ A  B^T ]     A = viscous block, B = -(q, div u)
//       [ B   0  ]
// Kinematic units throughout (stress, pressure and force divided by the density).
//
// Geometry::Axisymmetric is the meridional (r, z) = (x, y) formulation with continuity tested by r q, the r-weighted
// viscous form with the hoop terms, and the swirl component u_theta as a third velocity component (components
// ordered u_r, u_z, u_theta).
//
// Boundary conditions are EXPLICIT: every boundary attribute of the mesh must carry
// exactly one condition; unassigned, unknown or repeated attributes are refused. The pressure nullspace is detected
// ALGEBRAICALLY (B^T 1 = 0 after elimination) and must agree with what the conditions imply; a disagreement (e.g.
// a forgotten axis or a mislabelled wall) is refused.
//
// Consistency: when the constant pressure is in the kernel, the continuity right-hand side is projected
// onto 1-perp before ANY solve (the P2-interpolated Dirichlet data are flux-compatible only up to O(h^4)); the
// removed flux defect is reported, and every path (pin, multiplier, Ortho) then solves the same consistent system.
// The convergence gate is the relative residual of that system over the non-Dirichlet rows.
//
// Further features:
//  * both viscous forms with VARIABLE nu: the Laplacian form is the PDE -div(nu grad u) + grad p = f as written
//    (it differs from the stress form inside the domain by -div(nu (grad u)^T));
//  * grad-div stabilization gamma (div u, div v), gamma in viscosity units (NOT scaled by nu);
//  * slip / symmetry: u.n = 0 imposed strongly by a local rotation of the nodal velocity (normal component
//    eliminated); the tangential traction of the chosen form vanishes naturally. Normal at a node: the given
//    normal coefficient, else the outward normal of the (possibly curved) boundary averaged over the attribute's
//    boundary elements at that node. Shared nodes: Dirichlet wins; two
//    non-parallel slip/symmetry normals at a node give u = 0 there;
//  * periodic pairs on a mesh made periodic by make_periodic_mesh (meshes.hpp), with a prescribed pressure drop
//    (body force) or flow rate (one Lagrange multiplier, the pressure drop is reported);
//  * explicit quadrature orders (viscous / load / boundary), so the discrete problem does not depend on MFEM's
//    defaults; a P1/P1 (equal-order) pair only with allow_unstable_pair (negative control).
//
// The library knows nothing about the EM design spec: input is a mesh, coefficients and boundary conditions per
// attribute; output is grid functions plus diagnostics.
#pragma once

#include <mfem.hpp>

#include <memory>
#include <string>
#include <vector>

#include "axionfem/fluids/sparse_lu.hpp"

namespace axf::fluids {

enum class ViscousForm { Laplacian, SymmetricStress };
// Planar25D: planar z-invariant flow with THREE velocity components (u_x, u_y, u_z); u_z has the Laplacian
// operator, no pressure coupling and the body-force component G_z (Laplacian form, no grad-div; slip/periodic and
// the stress form are refused). Used by the inductionless-MHD duct cases.
enum class Geometry { Planar, Axisymmetric, Planar25D };
// How the constant-pressure nullspace of an enclosed flow is removed (a numerical device; the reported pressure
// then has zero mean, r-weighted in axisymmetry). A prescribed pressure VALUE is a boundary condition instead:
// StokesOptions::pressure_point.
//   Auto           none if the nullspace is absent; else MeanMultiplier (STRUMPACK), PinPressure (Eigen), Ortho (Krylov)
//   None           assert there is none (refused if the BCs leave one)
//   PinPressure    direct path: p = 0 at the pressure DOF nearest pin_point (ties: smallest global index); refused
//                  when the flow has no pressure nullspace
//   MeanMultiplier direct path: bordered system with one Lagrange multiplier for int p dx = 0
//   Ortho          Krylov path: OrthoSolver (zero-sum pressure) in the preconditioner
enum class Nullspace { Auto, None, PinPressure, MeanMultiplier, Ortho };
enum class LinearSolverKind { Direct, MinresBlockDiag };

// Boundary-condition kinds (one per boundary attribute). Slip and Symmetry are the same conditions in planar
// Stokes (u.n = 0, zero tangential traction of the chosen form); both names exist for COMSOL parity. Periodic:
// planar only, on a mesh made periodic beforehand.
enum class BCKind { Dirichlet, DoNothing, Traction, Axis, Slip, Symmetry, Periodic };
// What drives a periodic pair: nothing (G = 0), a prescribed pressure drop per period, or a prescribed flow rate.
enum class PeriodicDrive { None, PressureDrop, FlowRate };

const char* to_string(ViscousForm f);
const char* to_string(Nullspace n);
const char* to_string(BCKind k);

struct BoundaryCondition {
  int attribute = 0;
  BCKind kind = BCKind::DoNothing;
  // Dirichlet: the velocity (nullptr = no-slip). Traction: g = sigma_F n in the chosen form F (nullptr = 0).
  // Dimension = number of velocity components.
  mfem::VectorCoefficient* value = nullptr;
  // Slip / Symmetry: the wall normal evaluated at the velocity nodes (any length, any sign; nullptr = the
  // geometric outward normal averaged at the node).
  mfem::VectorCoefficient* normal = nullptr;
  // Periodic: this attribute is identified with `partner` = attribute + translation (the partner needs no
  // condition of its own). Drive: PressureDrop value = p(x) - p(x + translation) (per period), so that
  // p = p_per - (value/|d|) (x . d/|d|); FlowRate value = Q, imposed as the PERIOD-AVERAGED section flux
  // (1/|d|) ∫_Omega u . d/|d| dx = Q. It equals the flux through every section only when the non-periodic boundaries
  // are impermeable.
  int partner = 0;
  std::vector<double> translation;
  PeriodicDrive drive = PeriodicDrive::None;
  double drive_value = 0.0;
};

struct StokesBoundary {
  std::vector<BoundaryCondition> conditions;
  StokesBoundary& dirichlet(int attr, mfem::VectorCoefficient* g = nullptr) {
    conditions.push_back({attr, BCKind::Dirichlet, g});
    return *this;
  }
  StokesBoundary& do_nothing(int attr) {
    conditions.push_back({attr, BCKind::DoNothing, nullptr});
    return *this;
  }
  StokesBoundary& traction(int attr, mfem::VectorCoefficient* g) {
    conditions.push_back({attr, BCKind::Traction, g});
    return *this;
  }
  StokesBoundary& axis(int attr) {
    conditions.push_back({attr, BCKind::Axis, nullptr});
    return *this;
  }
  StokesBoundary& slip(int attr, mfem::VectorCoefficient* normal = nullptr) {
    BoundaryCondition c{attr, BCKind::Slip, nullptr};
    c.normal = normal;
    conditions.push_back(c);
    return *this;
  }
  StokesBoundary& symmetry(int attr, mfem::VectorCoefficient* normal = nullptr) {
    BoundaryCondition c{attr, BCKind::Symmetry, nullptr};
    c.normal = normal;
    conditions.push_back(c);
    return *this;
  }
  StokesBoundary& periodic(int attr, int partner, std::vector<double> translation,
                           PeriodicDrive drive = PeriodicDrive::None, double value = 0.0) {
    BoundaryCondition c{attr, BCKind::Periodic, nullptr};
    c.partner = partner;
    c.translation = std::move(translation);
    c.drive = drive;
    c.drive_value = value;
    conditions.push_back(c);
    return *this;
  }
};

// The COMSOL "pressure point constraint": p_h(x0) = value. x0 must be a pressure node (a mesh vertex for P1).
// The reported pressure is RAW (not mean-free). Refused when the flow has no pressure nullspace.
struct PressurePoint {
  bool active = false;
  std::vector<double> x0;
  double value = 0.0;
};

struct StokesOptions {
  int velocity_order = 2;  // Taylor–Hood: pressure order velocity_order - 1
  // pressure order; -1 = velocity_order - 1 (Taylor–Hood). Equal order (P1/P1) is unstable without stabilization
  // and is accepted only with allow_unstable_pair (the inf-sup negative control).
  int pressure_order = -1;
  bool allow_unstable_pair = false;
  ViscousForm form = ViscousForm::SymmetricStress;
  Geometry geometry = Geometry::Planar;
  bool swirl = true;                        // axisymmetric only
  mfem::Coefficient* nu = nullptr;          // kinematic viscosity (required); constant or variable in both forms
  double grad_div = 0.0;                    // gamma >= 0 of gamma (div u, div v), viscosity units (planar)
  // quadrature orders (0 = default): viscous / divergence / pressure-mass blocks 2p+2, loads 2p+8, boundary 2p+8
  // (p = velocity order). Exposed so that coefficients can be tabulated on exactly these rules.
  int quad_order_viscous = 0, quad_order_load = 0, quad_order_boundary = 0;
  int viscous_quad_order() const { return quad_order_viscous > 0 ? quad_order_viscous : 2 * velocity_order + 2; }
  int load_quad_order() const { return quad_order_load > 0 ? quad_order_load : 2 * velocity_order + 8; }
  int boundary_quad_order() const { return quad_order_boundary > 0 ? quad_order_boundary : 2 * velocity_order + 8; }
  // direct path: at least this many iterative-refinement steps (a "tight" solver, for solver-sensitivity checks)
  int min_refine_steps = 0;
  // direct path: GMRES preconditioned by the LU when the refinement stagnates above its target (3D saddle points
  // where STRUMPACK's static pivoting gives a poor factorization); off by default, on in the 3D API
  bool lu_krylov_fallback = false;
  // direct path: MC64 matching/scaling in STRUMPACK (default true, the 2D behaviour). On 3D Taylor-Hood
  // saddle points the matching permutes the structure and leaves a plain-solve residual of 0.1 .. 18 (measured at N = 32);
  // false (the 3D API's default) factors without it (residual 5e-16, faster). If that factorization fails, or its
  // solve misses the gate, the solve is repeated with the matching (diagnostics lu_matching_retry).
  bool lu_matching = true;
  // direct path: OpenMP threads of the sparse LU (factorization and solves). 1 (default) is deterministic: threaded
  // STRUMPACK changes the last digits and the refinement count from run to run; 0 = all threads.
  int lu_threads = 1;
  // slip/symmetry: a node whose adjacent slip boundary elements' normals spread by more than this angle (degrees)
  // is a geometric corner (u = 0), whatever the attribute labels. A genuine corner that turns by
  // LESS than this is averaged into one u.n = 0 constraint, and up to sin(angle/2) of the tangential velocity there
  // leaks through the walls: diagnostics report the largest averaged kink (slip_max_averaged_kink_deg); lower the
  // angle when that is a real corner.
  double slip_corner_angle_deg = 30.0;
  // velocity nullspace (rigid translations/rotation left free by slip/periodic conditions): refused unless false
  // Criterion of the continuous problem: a rigid motion with zero viscous energy in the
  // chosen form (stress form: all; Laplacian form: translations only) whose rms misfit with the homogeneous
  // boundary conditions on the (curved) boundary is <= velocity_nullspace_tol (geometric, mesh independent).
  bool refuse_velocity_nullspace = true;
  double velocity_nullspace_tol = 1e-3;  // rms boundary misfit below which a rigid motion counts as free
  mfem::VectorCoefficient* force = nullptr; // body force per unit mass
  LinearSolverKind solver = LinearSolverKind::Direct;
  LUBackend lu = LUBackend::EigenLU;
  Nullspace nullspace = Nullspace::Auto;
  std::vector<double> pin_point;            // PinPressure: default the bounding-box centre
  PressurePoint pressure_point;
  // gate of the direct path: relative residual of the consistent system (non-Dirichlet rows) <= direct_tol. The
  // LU refinement aims at min(1e-12, 1e-2 direct_tol) and stops early on stagnation; it never throws for a missed
  // target: the gate decides `converged` (ill-scaled systems, e.g. a viscosity contrast of 100 or more, stagnate
  // above 1e-12; loosen direct_tol for them).
  double direct_tol = 1e-10;
  // MINRES: rtol on MFEM's preconditioned norm, AND the true relative residual must be <= minres_true_factor*rtol
  double krylov_rtol = 1e-10;
  double minres_true_factor = 100.0;
  int krylov_maxit = 5000;
  bool pressure_amg = false;                // pressure block: Jacobi on M_p/nu (false) or BoomerAMG
  int print_level = 0;
  // Units: the core is unit-agnostic, but its gates are relative residuals of the consistent system and MINRES
  // starts from the Dirichlet lift, so the results are invariant under nu -> s nu and under L -> s L, nu -> s nu
  // (tested), so problems may be posed in O(1) units.
};

struct StokesDiagnostics {
  long n_velocity = 0, n_pressure = 0, n_total = 0;
  std::string solver, nullspace, viscous_form;
  bool bc_enclosed = false;        // what the boundary conditions imply (no DoNothing/Traction attribute)
  bool nullspace_detected = false; // algebraic: ||B^T 1|| <= 1e-10 ||B||_F ||1|| after elimination
  double kernel_defect = -1;       // ||B^T 1|| / (||B||_F ||1||)
  long pinned_dof = -1;            // global pressure true-DOF index (PinPressure / pressure point node)
  double multiplier = 0;           // MeanMultiplier: value of the Lagrange multiplier (round-off after projection)
  // Dirichlet-data flux defect, reported on every path when the nullspace is present (0 otherwise):
  double flux_defect = 0;          // |1^T g_p| / ||g_p||  (g_p: continuity right-hand side before the projection)
  double flux_defect_abs = 0;      // |1^T g_p| = |int g_h . n| (discrete)
  double grad_div = 0;
  long n_slip_nodes = 0;           // velocity nodes with a rotated (u.n = 0) frame
  long n_slip_corner_nodes = 0;    // nodes with two non-parallel slip normals (u = 0)
  long n_axis_point_nodes = 0;     // axisymmetric: velocity nodes on r = 0 outside every axis edge (u_r = u_theta = 0)
  double slip_max_averaged_kink_deg = 0;  // largest turning angle averaged into one slip frame (below the corner angle)
  // per periodic pair (in the order of the conditions): the pressure gradient G = drop/|d| (prescribed or the
  // multiplier of the flow-rate constraint) and the flow rate (prescribed, or 0 when not constrained)
  std::vector<double> periodic_pressure_gradient, periodic_flow_rate;
  std::string velocity_nullspace = "none";  // detected rigid motions left free (only when not refused)
  double velocity_nullspace_discrete_residual = -1;
  bool velocity_nullspace_near = false;  // solved, but a rigid motion is nearly free (misfit < 10 tol): ill-conditioned; also set by a round-off test of exact discrete singularity
  double velocity_nullspace_residual = -1;  // rms boundary misfit of the least constrained zero-energy rigid motion
  bool converged = false;          // HARD flag: velocity()/pressure() refuse to return a field from a failed solve
  int iterations = 0;
  double rel_residual = -1;        // ||K x - b|| / ||b|| over the non-Dirichlet rows, b the consistent rhs (the gate)
  double rel_residual_all_rows = -1;
  double res_momentum = -1, res_continuity = -1;  // per-block absolute residual norms (non-Dirichlet rows)
  double minres_preconditioned = -1;              // MINRES: MFEM's final preconditioned norm
  double pressure_point_value = 0;  // p_h(x0) after the solve (pressure point)
  double assemble_seconds = 0, setup_seconds = 0, solve_seconds = 0;
  LUStats lu;
  std::string lu_matching_retry_reason;
  double lu_first_attempt_residual_before = -1;  // the matching-free attempt's plain-solve residual, kept when it is retried
  int lu_first_attempt_krylov_iterations = 0;    // ... and its GMRES iterations
  bool lu_matching_retry = false;  // the matching-free factorization failed or missed the gate; solved with the matching
  int ranks = 1, threads = 1;
};

// Variational (Babuška–Miller) reaction on a set of boundary attributes: with w = sum over the
// body's velocity nodes of phi_k e (or the rigid rotation), F(w) = -[a_S(u_h, w) - (p_h, div w) - (f, w) - <g, w>].
// Kinematic (divide by rho for nothing; multiply by rho for physical units). Sign: the force EXERTED BY THE FLUID ON
// THE BODY. Planar: per unit depth; torque about `center` (z component). Axisymmetric: F_z and the torque about the
// axis, both including the 2 pi; F_r = 0 by symmetry.
struct Reaction {
  double force[3] = {0, 0, 0};
  double torque = 0;
  std::string residual_form;  // "solved" (physical), "stress" or "laplacian" (raw residuals of that operator)
};

class StokesSolver {
 public:
  StokesSolver(mfem::ParMesh& mesh, const StokesOptions& opt, const StokesBoundary& bc);
  ~StokesSolver();
  StokesSolver(const StokesSolver&) = delete;
  StokesSolver& operator=(const StokesSolver&) = delete;

  // Assemble the operator and right-hand side (also called by solve() when needed).
  void assemble();
  // Time stepping: re-evaluate the DATA-dependent parts of the assembled system after the (time-dependent) coefficients of the
  // force, the traction and the Dirichlet values changed: the Dirichlet lift, the load vector, the continuity
  // right-hand side (flux defect projected when the nullspace is present). Matrices are untouched. Needs assemble()
  // first; refused with slip/periodic conditions (not offered with convection).
  void refresh_data();
  // Solve; returns diagnostics. Never throws on non-convergence: check diagnostics().converged.
  const StokesDiagnostics& solve();
  const StokesDiagnostics& diagnostics() const { return diag_; }
  const StokesOptions& options() const { return opt_; }

  // Solution fields. Throw axf::Error unless the last solve converged. Pressure: zero mean (r-weighted, i.e. the 3D
  // mean, in axisymmetry) for an enclosed flow
  // without a pressure point; raw with a pressure point or an open boundary.
  const mfem::ParGridFunction& velocity() const;
  const mfem::ParGridFunction& pressure() const;
  mfem::ParFiniteElementSpace& velocity_space() { return *ufes_; }
  mfem::ParFiniteElementSpace& pressure_space() { return *pfes_; }

  // Babuška–Miller reaction on the body attributes (after a converged solve). center: torque reference point
  // (planar; default origin). residual_form Laplacian is the negative control: on moving or slip walls the
  // L-form residual is the pseudo-traction, not the force.
  // The raw residual of the requested form's operator (+ the solved grad-div); a diagnostic when it is not the solved
  // form: the S residual of an L solution is not Galerkin-consistent (it carries -nu (div u_h, div w)).
  Reaction reaction(const std::vector<int>& body_attributes, const std::vector<double>& center = {},
                    ViscousForm residual_form = ViscousForm::SymmetricStress) const;
  // The PHYSICAL force/torque (Cauchy traction) from the residual of the SOLVED operator, grad-div included; valid on
  // any wall (at rest, moving, slip). Stress form: the solved residual. Laplacian form: the solved residual plus
  // nu [ (grad u_h^T, grad w) - (div u_h, div w) ] (r-weighted with the hoop entries in axisymmetry), a null
  // Lagrangian: it vanishes for every w that is zero on the boundary (the functional stays Galerkin-consistent) and
  // turns the pseudo-traction nu du/dn - p n into the Cauchy traction (on a rotating sphere the raw L residual gives
  // 2/3 of the torque). Needs a CONSTANT viscosity in the Laplacian form (refused otherwise: with variable nu the
  // Laplacian form is not the Stokes momentum equation and its "force" is not defined).
  Reaction physical_reaction(const std::vector<int>& body_attributes, const std::vector<double>& center = {}) const;

  // Power put into the fluid through the velocity prescribed on `attributes` (Babuška–Miller, the solved form and
  // grad-div): R(w) with w = u_h at the velocity nodes of those attributes and 0 elsewhere, R(v) = a(u_h, v) +
  // gamma (div u_h, div v) - (p_h, div v) - (f, v) - <g, v>. For a lid-driven cavity it equals the dissipation
  // a(u_h, u_h) + gamma ||div u_h||^2 to round-off (planar only).
  double boundary_power(const std::vector<int>& attributes) const;

  // Assembled blocks after elimination (for inf-sup estimates and benchmarks); with slip conditions they are in
  // the rotated nodal frame: A (viscous, Dirichlet rows/cols
  // identity), B (Dirichlet columns zero), M_p (pressure mass weighted by 1/nu, r/nu in axisymmetry), and the
  // essential velocity true DOFs. Valid after assemble().
  const mfem::HypreParMatrix& viscous_block() const;
  const mfem::HypreParMatrix& divergence_block() const;
  const mfem::HypreParMatrix& pressure_mass() const;
  const mfem::Array<int>& essential_velocity_dofs() const;
  // New matrix: the unweighted vector H1 Gram (grad:grad + mass) of the velocity space, Dirichlet rows/cols
  // eliminated (identity), for the discrete inf-sup estimate.
  std::unique_ptr<mfem::HypreParMatrix> velocity_h1_gram() const;

  // The eliminated monolithic matrix (true DOFs, rows block-distributed, consistent rhs) for the solver
  // benchmarks; with_nullspace applies the direct nullspace treatment. Without it, a matrix whose pressure
  // nullspace is present is refused unless allow_singular (the caller then owns the consequences).
  DistCsr monolithic(bool with_nullspace, std::vector<double>* rhs_local = nullptr, bool allow_singular = false);

 private:
  // the Navier-Stokes solver reads the assembled system (cpp/src/fluids/stokes_internal.hpp) and solves its
  // linearizations with this class's direct-path layout
  friend class NavierStokesSolver;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  mfem::ParMesh& mesh_;
  StokesOptions opt_;
  StokesBoundary bc_;
  std::unique_ptr<mfem::FiniteElementCollection> ufec_, pfec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> ufes_, pfes_;
  std::unique_ptr<mfem::ParGridFunction> u_, p_;
  StokesDiagnostics diag_;
  void validate_boundary();
  void setup_slip();
  // stokes3d.cpp: 3D slip/symmetry frames (one, two or three normal constraints per node)
  void setup_3d();
  void detect_velocity_nullspace();
  void rotate_operators();
  void solve_direct();
  Reaction reaction_impl(const std::vector<int>& body, const std::vector<double>& center, ViscousForm residual_form,
                         bool physical) const;
  void solve_minres();
  void finish_pressure();
  long nearest_pressure_dof(const std::vector<double>& pt, double* dist) const;
};

}  // namespace axf::fluids
