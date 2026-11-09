// Scalar-transport operator (beta; formulation: docs/guides/fluids.md).
//
//   steady (a = capacity: rho c_p for heat, 1 for species):
//     convective             beta.grad T                   - div(kappa grad T) + s a T + sigma T = f
//     conservative_flux      -(T beta, grad v) form         (natural datum: TOTAL flux)
//     conservative_divergence beta.grad T + (div beta) T    (natural datum: diffusive flux)
//   with beta = scale * (callable b | finite-element velocity u_h): the advective flux velocity.
//
// Planar or axisymmetric ((r, z) = mesh (x, y); every integral carries the weight r).  P1 or P2 temperature.  SUPG with
// the ONE rule "shakib_hK" (Shakib, h_K = longest edge, C = 1, evaluated pointwise at the assembly quadrature
// points); the full strong residual (second derivatives of T_h, grad kappa . grad T, the axisymmetric hoop term
// kappa dT/dr / r, the shift and history source) enters the stabilization, and the Jacobian is EXACT: tau(u) and the
// streamline test function are differentiated.  Flux data use the INWARD convention: g = n . kappa grad T.
//
// The operator is linear-algebra only: residual / Jacobian / outputs on every degree of freedom (Dirichlet rows are
// reported unconstrained, which is what the variational boundary flux needs), plus the scalar steady solve() used by
// `solve_heat` and the Boussinesq solver.  ONE rank only: a partitioned (multi-rank) mesh is refused.  Quadrature degree
// 2p + 4 for every domain and boundary term.
#pragma once

#include <mfem.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "axionfem/fluids/sparse_lu.hpp"

namespace axf::transport {

enum class Geometry { Planar, Axisymmetric };
enum class AdvectiveForm { Convective, ConservativeFlux, ConservativeDivergence };
// Robin: n.kappa grad T = h (T_ext - T) (diffusive flux) with the convective forms; with ConservativeFlux the same data
// give the TOTAL-flux Robin j_out = h (T - T_ext).  Zero: zero natural datum of the form (a
// zero total flux with ConservativeFlux: a wall or symmetry line, NOT an outflow; the Python layer refuses 'outflow' there).
enum class BcKind { Dirichlet, Flux, Robin, TotalFlux, Zero, Axis };
const char* to_string(Geometry g);
const char* to_string(AdvectiveForm f);
const char* to_string(BcKind k);

// Batch data: evaluated once for ALL points of a kind (the points are physical (x, y) pairs, row-major)
using BatchFn = std::function<void(const double* xy, int n, double* out)>;              // n values
using BatchVec = std::function<void(const double* xy, int n, int ncomp, double* out)>;  // n * ncomp values

struct ScalarData {
  double value = 0.0;
  BatchFn fn;  // wins over value when set
};

struct BcData {
  ScalarData g;    // Dirichlet value; inward diffusive flux (Flux); inward total flux (TotalFlux)
  ScalarData h;    // Robin coefficient
  ScalarData ext;  // Robin external value
};

struct Velocity {
  BatchVec fn;                                  // callable b at points (n x 2); beta = scale * b
  BatchFn div_fn;                               // div b at points (ConservativeDivergence/Flux with a callable)
  const mfem::FiniteElementSpace* fes = nullptr;  // finite-element velocity (vdim 2 or 3: meridional part advects)
  double scale = 1.0;
  bool is_field() const { return fes != nullptr; }
  bool is_set() const { return fes != nullptr || static_cast<bool>(fn); }
};

struct TransportOptions {
  Geometry geometry = Geometry::Planar;
  int order = 2;       // 1 or 2
  int quad_order = 0;  // 0 = 2p + 4; lower is refused
  AdvectiveForm form = AdvectiveForm::Convective;
  bool supg = false;
  std::string tau_rule = "shakib_hK";  // the only rule offered
};

struct TauSample {
  int element = 0;
  double x = 0, y = 0, tau = 0;
};

struct SolveDiagnostics {
  bool converged = false;
  double relative_residual = -1;  // ||R(T_h)||_* / ||R(T_lift)||_* (free rows, consistent r-weighted mass)
  double residual_lift = -1;
  int iterations = 0;
  double lu_residual = -1;
  long dofs = 0, free_dofs = 0;
  std::string failure;
};

class ScalarTransport {
 public:
  ScalarTransport(mfem::Mesh& mesh, const TransportOptions& opt);
  ~ScalarTransport();
  ScalarTransport(const ScalarTransport&) = delete;
  ScalarTransport& operator=(const ScalarTransport&) = delete;

  // ---- data (before finalize())
  void set_capacity(double a);
  void set_shift(double s);  // s a T (the shift of an implicit time step, e.g. BDF)
  void set_diffusivity(const ScalarData& kappa, const BatchVec& grad_kappa = {});
  void set_velocity(const Velocity& v);
  void set_source(const ScalarData& f);
  void add_bc(const std::vector<int>& attrs, BcKind kind, const BcData& data = {});
  void finalize();

  // time-step hook: the history source h (a finite-element function in T's space) enters the residual AND the SUPG residual
  void set_history_source(const mfem::Vector& h);
  void clear_history_source();
  // time-step hook: change of the shift between steps (re-evaluates tau); state save/restore is the caller's T vector
  double shift() const { return shift_; }

  // ---- algebra on EVERY dof (state T has the Dirichlet values on the Dirichlet dofs)
  int ndof() const { return fes_->GetNDofs(); }
  const mfem::FiniteElementSpace& space() const { return *fes_; }
  mfem::FiniteElementSpace& space() { return *fes_; }
  const std::vector<char>& essential_mask() const { return ess_; }
  const mfem::Vector& lift() const { return lift_; }  // Dirichlet values on Dirichlet dofs, 0 elsewhere
  // R_i = R(T; phi_i), natural and SUPG terms included, Dirichlet rows NOT replaced. u: the FE velocity coefficients
  // (true dofs of the velocity space) when the velocity is a field.
  void residual_full(const mfem::Vector& T, const mfem::Vector* u, mfem::Vector& R) const;
  // dR/dT (all rows and columns)
  mfem::SparseMatrix jacobian(const mfem::Vector& T, const mfem::Vector* u) const;
  // dR/du (ndof x u.Size()): the advection coupling, tau(u) and the streamline test function differentiated
  mfem::SparseMatrix advection_jacobian_u(const mfem::Vector& T, const mfem::Vector& u) const;
  // consistent r-weighted mass matrix, capacity-FREE (M_ab = int N_a N_b r; NOT a-weighted); used by the dual norm
  const mfem::SparseMatrix& mass() const { return *mass_; }
  // nodal v_Gamma of the CLOSED part: 1 at every dof of the boundary elements with these attributes
  std::vector<double> boundary_weights(const std::vector<int>& attrs) const;
  std::vector<TauSample> tau_samples(const mfem::Vector& T, const mfem::Vector* u) const;
  int quad_order() const { return qdeg_; }
  const TransportOptions& options() const { return opt_; }
  // the steady solve (as in the Navier-Stokes solver) of a LINEAR scalar problem (prescribed velocity or fixed field): Newton in one step,
  // reported in the dual norm; extra steps only if the gate is missed
  SolveDiagnostics solve(mfem::Vector& T, const mfem::Vector* u, fluids::LUBackend lu, double rtol, int maxit = 4);
  double dual_norm(const mfem::Vector& R_full) const;  // over the free rows; consistent mass
  // flux through the boundary from the discrete gradient (a pointwise, non-variational output): int -kappa dT/dn
  double surface_flux(const mfem::Vector& T, const std::vector<int>& attrs) const;
  // mesh size
  int element_count() const { return mesh_.GetNE(); }

 private:
  struct Impl;
  struct Pt;
  void point(int e, int q, const double* Te, const double* ue, const double* he, Pt& p) const;
  mfem::Mesh& mesh_;
  TransportOptions opt_;
  std::unique_ptr<mfem::FiniteElementCollection> fec_;
  std::unique_ptr<mfem::FiniteElementSpace> fes_;
  std::unique_ptr<Impl> impl_;
  std::vector<char> ess_;
  mfem::Vector lift_;
  std::unique_ptr<mfem::SparseMatrix> mass_;
  AdvectiveForm declared_ = AdvectiveForm::Convective;  // the form the user asked for (validation of the data kinds)
  double cap_ = 1.0, shift_ = 0.0;
  int qdeg_ = 8;
  bool finalized_ = false;
  mutable std::unique_ptr<fluids::SparseLU> mass_lu_;
  mutable fluids::DistCsr mass_csr_;
  mutable std::vector<int> free_map_;  // dof -> free index (-1 for Dirichlet)
  mutable int nfree_ = 0;
  fluids::LUBackend lu_backend_ = fluids::LUBackend::EigenLU;

 public:
  void set_lu_backend(fluids::LUBackend b) { lu_backend_ = b; }
  int free_index(int dof) const { return free_map_[dof]; }
  int n_free() const { return nfree_; }
};

}  // namespace axf::transport
