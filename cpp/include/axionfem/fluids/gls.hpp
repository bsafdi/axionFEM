// Stabilized equal-order P1/P1 incompressible flow, planar: residual-based GLS = SUPG + PSPG + LSIC (beta;
// formulation: docs/guides/fluids.md). The discrete problem:
//
//   R_u,a = sum_K int [ rho (u_t + (w.grad)u).N_a e_i + a_visc(mu;u)(N_a e_i) - p d_i N_a - f_i N_a
//                      + tau_supg rho ((w.grad N_a) R_i) + tau_c rho (div u) d_i N_a ]
//   R_p,a = sum_K int [ (div u) N_a + (tau_pspg / rho) grad N_a . R ]
//   R     = rho (u_t + (w.grad)u) + grad p - div sigma_visc(u; mu) - f      (strong residual, element-wise, grad mu kept)
//   tau_supg = [ (2/dt)^2 + CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2),  tau_pspg = [ CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)
//   tau_c = 1 / (tau_pspg tr G),  G from the EQUILATERAL reference triangle, C_I = 36, CG = 1, 6-point degree-4 rule.
//
// ONE element routine (templated on the scalar type) is the residual, the exact Jacobian (forward-mode differentiation:
// tau differentiated with the state) and the Jacobian action; tau_at() calls the same tau routine the assembly calls.
// Nonlinear driver: the conventions of the steady Navier-Stokes solver (a true Stokes start, Newton with the exact
// Jacobian, Armijo line search on the discrete dual-norm merit lambda = 1, 1/2, ..., 1/64, viscosity continuation as
// scale factors, steady relative residual ||R(x)||_* / ||R(x_lift)||_*) on raw arrays, because the Navier-Stokes
// driver is tied to MFEM Taylor-Hood spaces. Direct sparse LU (axf::fluids::SparseLU) only. Planar only: axisymmetry is refused.
//
// Enclosed flows: the constant pressure is removed by a mean-value multiplier row (c^T p = 0, c_a = int N_a), which also
// absorbs the incompatibility of the P1-interpolated Dirichlet flux; the pressure rows of the solved system are
// R_p + lambda c.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "axionfem/fluids/sparse_lu.hpp"

namespace axf::fluids {

enum class GlsMode { Stokes, Oseen, NavierStokes };
enum class GlsForm { Laplacian, Stress };

struct GlsStabilization {
  bool supg = true, pspg = true, lsic = true;  // all off / PSPG off: NEGATIVE CONTROLS only (the unstabilized pair is singular)
};

struct GlsNonlinear {
  double newton_rtol = 1e-12;
  int newton_maxit = 40;      // per continuation stage
  bool line_search = true;
  double armijo_c1 = 1e-4;
  int line_search_halvings = 6;
  std::vector<double> viscosity_scales;  // mu_j = s_j mu, empty = {1}
};

struct GlsOptions {
  GlsMode mode = GlsMode::NavierStokes;
  GlsForm form = GlsForm::Stress;
  double rho = 1.0;
  double dt = 0.0;  // 0 = steady; > 0: one implicit step, BDF1 or BDF2 with the supplied history
  int bdf = 1;
  GlsStabilization stab;
  LUBackend lu = LUBackend::EigenLU;
  int lu_threads = 1;
  GlsNonlinear nl;
};

// Fixed rule: the 6-point degree-4 rule of benchmarks/fluids/cases/f5_gls_check.rule_points() (barycentric-like
// reference points on the triangle (0,0),(1,0),(0,1); weights sum to 1/2 as in the check module's literals) plus the
// centroid (index 6: the data arrays carry it; the residual uses the 6 rule points).
constexpr int GLS_NQ = 6;
constexpr int GLS_NQ_ALL = 7;
const double* gls_rule_points();   // 6 x 2
const double* gls_rule_weights();  // 6 (weights of the check module: tabulated weight / 2 each ... sums to 1/2)

struct GlsStep {
  std::string kind;  // stokes | newton
  double residual = -1, step = 1, viscosity_scale = 1;
  int stage = 0, lu_refine = 0;
  double lu_residual = -1, lu_residual_before = -1;  // relative LU residual of this step's linear solve (after / before refinement)
};

struct GlsDiagnostics {
  bool converged = false;
  int iterations = 0;
  double relative_residual = -1, residual_lift = -1;
  int line_search_min_step_accepts = 0;
  std::string failure, nullspace = "mean_multiplier", lu_backend;
  bool nullspace_detected = false;  // true iff every boundary vertex is Dirichlet (enclosed flow: constant-pressure nullspace)
  double multiplier = 0.0, kernel_defect = -1;
  std::vector<GlsStep> history;
  std::vector<double> viscosity_scales;
  double seconds = 0, lu_residual = -1, lu_residual_before = -1;
  int lu_refine_steps = 0;
  long n_free_u = 0, n_pressure = 0, n_total = 0;
};

// All arrays row-major. Quadrature arrays have GLS_NQ_ALL columns per element: the 6 rule points in the element's own
// vertex order, then the centroid. mu: either mu_q [ne,7] + gmu_q [ne,7,2] (analytic) or mu_nodal [nv] (P1 nodal viscosity,
// element-wise gradient); f_q [ne,7,2]; adv_q [ne,7,2] (Oseen). hist1/hist2: nodal velocity history [nv,2].
struct GlsData {
  std::vector<double> V;   // nv*2
  std::vector<int> T;      // ne*3
  std::vector<double> mu_q, gmu_q, mu_nodal, f_q, adv_q, hist1, hist2;
  std::vector<int> dirichlet_nodes;       // vertex indices
  std::vector<double> dirichlet_values;   // n*2
};

class GlsProblem {
 public:
  GlsProblem(GlsData data, GlsOptions opt);
  ~GlsProblem();
  GlsProblem(const GlsProblem&) = delete;
  GlsProblem& operator=(const GlsProblem&) = delete;

  const GlsDiagnostics& solve();  // never throws on non-convergence
  const GlsDiagnostics& diagnostics() const { return diag_; }
  const GlsOptions& options() const { return opt_; }
  int n_vertices() const;
  int n_elements() const;
  // fields of a solve (u [nv,2] with the Dirichlet data, p [nv] mean-free)
  const std::vector<double>& u() const;
  const std::vector<double>& p() const;
  const std::vector<int>& dirichlet_nodes() const;

  // the full residual (R_u [nv,2], R_p [nv]) at a nodal state: all rows, no elimination, no multiplier
  void residual_full(const std::vector<double>& u, const std::vector<double>& p, std::vector<double>& Ru,
                     std::vector<double>& Rp, double viscosity_scale = 1.0) const;
  // the free-DOF API for Taylor tests: x = [free velocity DOFs (vertex-major, component fastest), all pressures]
  std::pair<long, long> layout() const;
  std::vector<double> state() const;
  std::vector<double> residual(const std::vector<double>& x, double viscosity_scale = 1.0) const;
  std::vector<double> jacobian_action(const std::vector<double>& x, const std::vector<double>& d,
                                      double viscosity_scale = 1.0) const;
  // tau_supg, tau_pspg, tau_lsic [ne, 6] at the rule points for the advecting velocity w (nodal P1 [nv,2])
  void tau_at(const std::vector<double>& w_nodal, std::vector<double>& tau_supg, std::vector<double>& tau_pspg,
              std::vector<double>& tau_lsic) const;
  double dual_norm(const std::vector<double>& Ru_full, const std::vector<double>& Rp_full) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  GlsOptions opt_;
  GlsDiagnostics diag_;
};

}  // namespace axf::fluids
