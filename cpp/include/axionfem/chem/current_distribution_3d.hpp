// 3D secondary/primary current distribution on MPI ranks (beta; formulation: docs/guides/chemistry.md).  MPI build
// only.
//
// Electrolyte potential phi on a tetrahedral mesh (Lagrange P1/P2, P2 optionally isoparametric by given edge nodes),
// distributed over MPI_COMM_WORLD by METIS (mfem::ParMesh), with the PDE, conventions and kinetics of the 2D solver
// (current_distribution.hpp):
//     -div(kappa grad phi) = f,   kappa grad phi . n_out = sum_j i_j(eta_j) + g  on kinetic electrodes,
//     phi = phi_s - E_eq on (potentiostatic) primary electrodes, kappa grad phi . n_out = q on insulators,
// eta_j = phi_s - phi - E_eq,j, anodic currents positive.  A kinetic electrode is potentiostatic or galvanostatic
// (phi_s unknown, bordered row int (sum_j i_j + g) dA = I, the electrode current reduced over all ranks).
//
// Newton: the criterion of the 2D solver (iterate to stagnation, |d eta| limiter at the kinetic quadrature points, halving
// line search on the Euclidean norm of the full residual, exponent clip).  The Jacobian K + M' is symmetric positive
// definite for monotone kinetics; the linear solver is CG preconditioned by BoomerAMG (hypre), the bordered row
// eliminated by its Schur complement (two solves with one preconditioner).  Raw quadrature data are per rank (local
// elements / boundary elements); true-dof vectors are the rank's owned true dofs.
#pragma once

#ifndef AXF_NO_MPI

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "axionfem/chem/current_distribution.hpp"

namespace axf::chem {

struct Mesh3DInput {
  std::vector<double> vertices;         // nv x 3 (the whole mesh, on every rank)
  std::vector<int> tets;                // ne x 4
  std::vector<int> element_attr;        // ne
  std::vector<int> boundary;            // nb x 3 (boundary triangles)
  std::vector<int> boundary_attr;       // nb
  std::vector<int> edge_node_vertices;  // m x 2: P2 edges whose node is not the straight midpoint
  std::vector<double> edge_node_xyz;    // m x 3
};

struct DomainQuadrature3D {
  Eigen::MatrixXd x;     // n x 3
  Eigen::VectorXd w;     // weights with |det J|
  Eigen::VectorXd phi;
  Eigen::MatrixXd grad;  // n x 3
};
struct BoundaryQuadrature3D {
  Eigen::MatrixXd x;
  Eigen::VectorXd w;       // with the surface Jacobian
  Eigen::VectorXd phi;
  Eigen::MatrixXd grad;    // adjacent element's gradient
  Eigen::MatrixXd normal;  // unit outward normal (of the isoparametric map at the point)
};

class CurrentDistribution3D {
 public:
  CurrentDistribution3D(const Mesh3DInput& mesh, int order, int domain_degree, int boundary_degree);
  ~CurrentDistribution3D();
  CurrentDistribution3D(const CurrentDistribution3D&) = delete;
  CurrentDistribution3D& operator=(const CurrentDistribution3D&) = delete;

  int order() const { return order_; }
  int domain_degree() const { return deg_dom_; }
  int boundary_degree() const { return deg_bdr_; }
  int rank() const;
  int ranks() const;

  // ---- coefficients (local quadrature points at the assembly degrees)
  Eigen::MatrixXd domain_points() const;
  Eigen::VectorXi domain_point_elements() const;  // local element of each domain point
  Eigen::MatrixXd element_centroids() const;      // local elements (vertex mean)
  Eigen::MatrixXd boundary_points(const std::vector<int>& attrs) const;
  void set_conductivity_constant(double kappa);
  void set_conductivity_at_points(const Eigen::VectorXd& kappa);
  void set_source_at_points(const Eigen::VectorXd& f);
  void set_insulator_flux(int attr, const Eigen::VectorXd& q);
  int add_electrode(const ElectrodeSpec& e);   // kinetic (potentiostatic/galvanostatic) or potentiostatic primary

  // ---- solve (the Newton criterion above; options kept for later evaluations)
  SolveReport solve(const NewtonOptions& opt);
  const SolveReport& report() const { return report_; }
  void set_linear_tolerance(double rel, int max_iter);
  // change an electrode's control after add_electrode (continuation): phi_s of a potentiostatic electrode, the current
  // of a galvanostatic one; then solve(initial = "state") warm-starts from the previous solution.  Large
  // overpotentials (beyond a few volts) need such a continuation (initial = "primary" is not available in 3D): Newton starts by default
  // from phi = 0, so the whole applied potential first sits on the interface, and far above the solution each Newton
  // step on the exponential branch lowers alpha_a f eta by about one (eta by RT/(alpha_a F)), so a few volts need many
  // iterations.  The step limiter (|d eta| <= 10 RT/F per step) is not
  // the cause: it caps the large steps Newton takes when it approaches the solution from below, and it applies to
  // every kinetic electrode, linear ones included
  void set_control(int e, double value);

  // ---- state: owned true dofs of phi + phi_s of the galvanostatic electrodes (same on every rank)
  Eigen::VectorXd state() const;
  void set_state(const Eigen::VectorXd& z);
  std::vector<double> phi_s() const;
  // the state after one more undamped Newton step from the current state (the state itself is unchanged)
  Eigen::VectorXd newton_update() ;
  // full z-space residual (owned rows + bordered rows) and the Jacobian action at z (Taylor tests)
  Eigen::VectorXd residual(const Eigen::VectorXd& z);
  Eigen::VectorXd jacobian_apply(const Eigen::VectorXd& z, const Eigen::VectorXd& v);
  int num_true_dofs() const;
  long global_true_dofs() const;

  // ---- raw data (local)
  DomainQuadrature3D domain_quadrature(int degree, int elem_begin = 0, int elem_end = -1) const;
  BoundaryQuadrature3D boundary_quadrature(const std::vector<int>& attrs, int degree) const;
  // weak-form residual on the owned true dofs, without the kinetic term of electrode `exclude` (-1: none)
  Eigen::VectorXd residual_full(int exclude) const;
  std::vector<int> attr_true_dofs(const std::vector<int>& attrs) const;   // owned true dofs on these attributes

  // ---- mesh information
  int local_elements() const;
  long global_elements() const;
  std::vector<int> owned_cells_per_rank() const;   // gathered on every rank
  double h_max() const;                            // global
  double volume() const;                           // global
  // the serial mesh the ParMesh was built from (rank 0 only; empty elsewhere)
  Eigen::MatrixXd mesh_vertices() const;
  Eigen::MatrixXi mesh_elements() const;
  Eigen::MatrixXd geometry_nodes() const;          // every geometry node (vertices + P2 edge nodes)
  std::map<std::string, double> diagnostics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  int order_, deg_dom_, deg_bdr_;
  SolveReport report_;
};

}  // namespace axf::chem

#endif  // AXF_NO_MPI
