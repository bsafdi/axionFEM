#include <Eigen/Dense>
#ifdef AXF_HAVE_METIS
#include <iostream>  // Eigen/MetisSupport reports METIS errors on std::cerr

#include <Eigen/MetisSupport>
#endif
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>

#include "axionfem/omp_shim.hpp"

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/sym_solver.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using units::mu0;
using units::pi;

namespace {

// Associated Legendre P_n^1(x) for n = 1..N (unnormalized, no Condon-Shortley sign).
void legendre_p1(double x, int N, std::vector<double>& P) {
  P.assign(N + 1, 0.0);
  double s = std::sqrt(std::max(0.0, 1.0 - x * x));
  if (N >= 1) P[1] = s;
  if (N >= 2) P[2] = 3.0 * x * s;
  for (int n = 2; n < N; ++n) P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - (n + 1.0) * P[n - 1]) / n;
}

struct DofMap {
  // kind: 0 unused, 1 free, 2 tied to master, 3 fixed zero (axis)
  std::vector<int> kind, index, master;
  std::vector<double> w;
  int nfree = 0;
};

Vec to_eigen(const std::vector<double>& v, int n) {
  Vec x = Vec::Zero(n);
  for (int i = 0; i < n && i < static_cast<int>(v.size()); ++i) x(i) = v[i];
  return x;
}

// Sparse Cholesky with a choice of fill-reducing ordering.  Approximate minimum degree (Eigen's
// default) is cheap to compute and fine for small systems; METIS nested dissection costs more in
// the symbolic phase but gives much less fill on the large meridional meshes, so the factorization
// is faster and the factor needs less memory.  AXF_ORDERING = amd | metis | strumpack | auto
// overrides the choice.  METIS is optional (the sequential wheels are built without it).
//
// Measured on the fiducial toroid, exterior map as a low-rank update, one cluster node, 32 threads
// (benchmarks/solver_scaling.py -> docs/validation/solvers.md).  Ordering + factorization seconds and
// factor nonzeros, AMD / METIS / STRUMPACK:
//    26 k unknowns   0.02+0.13 / 0.13+0.14 / 0.14+0.14 s      0.90 / 0.86 / 1.79 M
//    94 k            0.11+0.78 / 0.56+0.74 / 0.56+0.45        4.31 / 3.79 / 7.82
//   357 k            0.52+5.17 / 2.50+4.18 / 2.45+1.96       21.5  / 17.7 / 36.3
//   856 k          690.41+21.6 / 6.61+14.6 / 7.18+8.10       60.8  / 47.6 / 97.5
// Eigen's own AMD is the cheapest ordering while it works, and it fails unpredictably on these
// matrices: 0.52 s at 357 k but 690 s at 856 k and 491-741 s on the 601 k toroid template, where it
// has also been seen to segfault inside Eigen's minimum_degree_ordering (reproduced with and without
// the exterior map in the matrix, so the dense block is not the cause; also on a 4.9 M-DOF CORE
// model).  It is not a size limit as
// such -- the same ordering handles a 1.1 M-unknown CORE matrix in 2.3 s and a 640 k 5-point
// Laplacian in 4.4 s -- so the choice below is deliberately conservative: METIS costs at most about
// 1 s more up to a few 1e5 unknowns and never collapses, so it takes over early.  Eigen's COLAMD is
// not an alternative (SimplicialCholesky wants the inverse permutation, which COLAMDOrdering does not
// produce: unusable even at 26 k unknowns).
constexpr int kMetisFrom = 150000;  // reduced unknowns above which METIS nested dissection is used
// Without METIS (the sequential wheels) there is no safe ordering above this size, so the solver says
// so instead of grinding for ten minutes or crashing.  AXF_ORDERING=amd overrides it.
constexpr int kAmdUnsafeFrom = 400000;

// eigen_only: thin layers need the Eigen LDLT, whose pivots give the definiteness guard (STRUMPACK does not
// expose them), whatever the size or AXF_ORDERING/AXF_STRUMPACK_MIN_DOFS say.
std::string ordering_choice(int nred, bool eigen_only = false) {
  if (eigen_only) {
    // The Eigen orderings as without layers (AXF_ORDERING=amd|metis are honoured, with the same notes and the
    // same refusal of AMD above kAmdUnsafeFrom unless asked for by name); only STRUMPACK is excluded.
    std::string s = std::getenv("AXF_ORDERING") ? std::getenv("AXF_ORDERING") : "";
    bool amd_by_name = s == "amd";  // only AXF_ORDERING=amd takes the risk above kAmdUnsafeFrom
    if (s == "strumpack") {
      log(1, "note: AXF_ORDERING=strumpack ignored: thin magnetic layers use the Eigen LDLT (definiteness guard); "
             "using auto");
      s.clear();
    }
    if (s == "metis" && !have_metis()) {
      log(1, "warning: AXF_ORDERING=metis but this build has no METIS; using the AMD ordering");
      s = "amd";
    }
    if (s == "amd" && amd_by_name) {
      if (nred >= kAmdUnsafeFrom)
        log(0, strcat("warning: AXF_ORDERING=amd with ", nred, " unknowns. Eigen's minimum-degree ordering "
                      "has been measured at 491-741 s on matrices this size and has been seen to segfault "
                      "inside Eigen (docs/validation/solvers.md); continuing because it was asked for."));
      else
        log(1, "note: AXF_ORDERING=amd with thin magnetic layers: Eigen LDLT with the AMD ordering");
      return "amd";
    }
    if (s == "amd") s.clear();  // AXF_ORDERING=metis without METIS: the automatic AMD, with its size limit
    if (s == "metis") return "metis";
    if (!s.empty() && s != "auto")
      log(1, "warning: AXF_ORDERING='" + s + "' is not one of amd, metis, auto; using auto");
    if (have_metis()) return "metis";
    if (nred >= kAmdUnsafeFrom) {
      throw Error(strcat("this build has no fill-reducing ordering that is safe for ", nred,
                         " unknowns with thin magnetic layers (which need the Eigen LDLT for their definiteness "
                         "guard; Eigen's own AMD ordering takes minutes, and has been seen to crash, at this size)"),
                  "/mesh", "Use a build with METIS (docs/install.md), coarsen the mesh, or set AXF_ORDERING=amd "
                           "to run it anyway.");
    }
    return "amd";
  }
  std::string want;
  bool amd_by_name = false;  // AXF_ORDERING=amd: the caller takes the risk above kAmdUnsafeFrom
  if (const char* e = std::getenv("AXF_ORDERING")) {
    const std::string s(e);
    if (s == "strumpack") {
      if (have_strumpack()) return "strumpack";
      log(1, "warning: AXF_ORDERING=strumpack but this build has no STRUMPACK; using auto");
    } else if (s == "metis" && !have_metis()) {
      log(1, "warning: AXF_ORDERING=metis but this build has no METIS; using the AMD ordering");
      want = "amd";
    } else if (s == "amd" || s == "metis") {
      want = s;
      amd_by_name = s == "amd";
    } else if (s != "auto" && !s.empty()) {
      log(1, "warning: AXF_ORDERING='" + s + "' is not one of amd, metis, auto; using auto");
    }
  }
  // STRUMPACK (multifrontal, METIS nested dissection, OpenMP) above AXF_STRUMPACK_MIN_DOFS when built
  // in.  threshold() is read here even without STRUMPACK, so that a misspelled value is reported.
  const int sp_from = SymSolver::threshold();
  if (!have_strumpack() && std::getenv("AXF_STRUMPACK_MIN_DOFS")) {
    log(1, "note: AXF_STRUMPACK_MIN_DOFS is set but this build has no STRUMPACK "
           "(scripts/build_strumpack.sh); the Eigen LDLT backend is used at every size");
  }
  if (want.empty() && have_strumpack() && nred >= sp_from) return "strumpack";
  if (want.empty()) want = (have_metis() && nred >= kMetisFrom) ? "metis" : "amd";
  if (want == "amd" && nred >= kAmdUnsafeFrom) {
    if (amd_by_name) {
      log(0, strcat("warning: AXF_ORDERING=amd with ", nred, " unknowns. Eigen's minimum-degree ordering "
                    "has been measured at 491-741 s on matrices this size and has been seen to segfault "
                    "inside Eigen (docs/validation/solvers.md); continuing because it was asked for."));
    } else {
      throw Error(strcat("this build has no fill-reducing ordering that is safe for ", nred,
                         " unknowns: Eigen's own AMD ordering takes minutes, and has been seen to crash, "
                         "on 2D matrices this size (docs/validation/solvers.md)"),
                  "/mesh",
                  "Use a build with METIS or STRUMPACK (docs/install.md, \"Sparse direct solvers\"), coarsen "
                  "the mesh (mesh.size, mesh.surface_size or a lower mesh.order), or set AXF_ORDERING=amd "
                  "to run it anyway.");
    }
  }
  return want;
}

// Sparse Cholesky of A, optionally plus a symmetric low-rank term V diag(c) V^T that is kept out of the
// sparse factorization (Woodbury).  The exterior map (DtN) of the open boundary is exactly of that form:
// assembling it instead makes the |boundary dofs| outer-boundary DOFs a clique of the sparsity graph,
// which adds 17-25 % to the fill of every backend and makes STRUMPACK warn about the depth of its
// elimination tree.  With
//     (A + V C V^T)^-1 = A^-1 - A^-1 V M^-1 V^T A^-1,   M = C^-1 + V^T A^-1 V,
// a factorization costs N = rank(V) extra back-substitutions (for M) and a solve one extra.  C is
// positive definite (the DtN coefficients are positive) and so is A, hence so is M.
class Cholesky {
 public:
  explicit Cholesky(std::string which) : which_(std::move(which)) {}
  const std::string& ordering() const { return which_; }
  bool metis() const { return which_ == "metis"; }
  bool strumpack() const { return which_ == "strumpack"; }
  // Set before analyzePattern: the system is A + V diag(c) V^T.  c must be positive.
  void set_low_rank(const SpMat& V, const Vec& c) {
    V_ = V;
    c_ = c;
    AXF_REQUIRE(V_.cols() == c_.size(), "internal: low-rank term with mismatched rank");
    AXF_REQUIRE(c_.size() == 0 || c_.minCoeff() > 0, "internal: low-rank term must be positive definite");
  }
  int low_rank() const { return static_cast<int>(V_.cols()); }
  // (A + V C V^T) x, for residual checks
  Vec apply(const SpMat& A, const Vec& x) const {
    Vec y = A * x;
    if (low_rank() > 0) y += V_ * Vec(c_.cwiseProduct(Vec(V_.transpose() * x)));
    return y;
  }
  const char* backend() const { return strumpack() ? sp_.backend() : "eigen-ldlt"; }
  std::string solver_name() const {
    if (strumpack()) return strcat("STRUMPACK ", strumpack_version(), " multifrontal");
    return "Eigen SimplicialLDLT";
  }
  int symbolic_analyses() const { return strumpack() ? sp_.symbolic_analyses() : 1; }
  int reused_factorizations() const { return strumpack() ? sp_.reused_factorizations() : 0; }
  void analyzePattern(const SpMat& A) { analyzePattern_sparse(A); }
  // Called with the smallest pivot after each sparse factorization, before the low-rank update (Eigen
  // backends only); used by the thin-layer definiteness guard.
  void set_pivot_guard(std::function<void(double)> g) { pivot_guard_ = std::move(g); }
  void factorize(const SpMat& A) {
    factorize_sparse(A);
    if (pivot_guard_ && info() == Eigen::Success && !strumpack()) pivot_guard_(min_pivot());
    if (info() != Eigen::Success || low_rank() == 0) return;
    // capacitance matrix M = C^-1 + V^T A^-1 V, in column blocks so that A^-1 V is never stored whole
    const int N = low_rank();
    Eigen::MatrixXd M(N, N);
    // As few blocks as a 32 MB right-hand side allows: a blocked triangular solve streams the factor
    // once for the whole block, so N single solves would read it N times.  The flops do not depend on
    // the block size, only how often the factor is read, so a modest block is enough and keeps the
    // peak memory of this step bounded (the block, its transpose and the result).
    const long budget = (32L << 20) / (8L * std::max<long>(1, A.rows()));
    const int blk = static_cast<int>(std::max(1L, std::min<long>(N, budget)));
    for (int j0 = 0; j0 < N; j0 += blk) {
      const int nj = std::min(blk, N - j0);
      M.middleCols(j0, nj) = V_.transpose() * solve_sparse(Eigen::MatrixXd(V_.middleCols(j0, nj)));
    }
    M.diagonal() += c_.cwiseInverse();
    M = 0.5 * (M + M.transpose().eval());  // symmetric up to round-off; make it exactly so
    // N x N (N <= dtn_modes), so a full-pivot LU costs nothing and stays right even if the tangent
    // matrix A is indefinite, where the capacitance matrix need not be positive definite.
    cap_.compute(M);
    AXF_REQUIRE(cap_.isInvertible(),
                strcat("the exterior-map (DtN) update of the field system is singular (rank ", cap_.rank(),
                       " of ", N, ")"),
                "", "Set AXF_DTN_DENSE=1 to assemble the exterior map into the matrix instead, and please "
                    "report the design.");
  }
  Eigen::ComputationInfo info() const {
    if (strumpack()) return sp_.info();
#ifdef AXF_HAVE_METIS
    if (metis()) return m_.info();
#endif
    return a_.info();
  }
  Vec solve(const Vec& b) const {
    Vec x = solve_sparse(b);
    if (low_rank() > 0) x -= solve_sparse(Vec(V_ * cap_.solve(Vec(V_.transpose() * x))));
    return x;
  }
  Eigen::MatrixXd solve(const Eigen::MatrixXd& B) const {
    Eigen::MatrixXd X = solve_sparse(B);
    if (low_rank() > 0) X -= solve_sparse(Eigen::MatrixXd(V_ * cap_.solve(Eigen::MatrixXd(V_.transpose() * X))));
    return X;
  }
  // smallest pivot of the sparse LDLT factor (NaN for STRUMPACK, which does not expose it)
  double min_pivot() const {
    if (strumpack()) return std::nan("");
#ifdef AXF_HAVE_METIS
    if (metis()) return m_.vectorD().minCoeff();
#endif
    return a_.vectorD().minCoeff();
  }
  // nonzeros in L (fill): the memory the factorization needs
  long nonzeros_L() const {
    if (strumpack()) return sp_.factor_nonzeros();
#ifdef AXF_HAVE_METIS
    if (metis()) return static_cast<long>(m_.matrixL().nestedExpression().nonZeros());
#endif
    return static_cast<long>(a_.matrixL().nestedExpression().nonZeros());
  }

 private:
  void analyzePattern_sparse(const SpMat& A) {
    if (strumpack()) {
      sp_.set_backend(1);
      sp_.analyzePattern(A);
      return;
    }
#ifdef AXF_HAVE_METIS
    if (metis()) {
      {
        // METIS swaps the process's signal handlers: one call at a time (sym_solver.hpp)
        std::lock_guard<std::mutex> lock(metis_mutex());
        m_.analyzePattern(A);
      }
      // Eigen's MetisOrdering prints to std::cerr and returns an EMPTY permutation when METIS
      // fails (out of memory, say); SimplicialCholesky then silently falls back to the natural
      // ordering, which still gives the right answer but can need orders of magnitude more fill.
      AXF_REQUIRE(m_.permutationP().size() == A.rows(),
                  "the METIS ordering failed (it leaves the permutation empty, which would silently "
                  "fall back to the natural ordering and may exhaust memory)", "",
                  "Set AXF_ORDERING=amd, or coarsen the mesh.");
      return;
    }
#endif
    a_.analyzePattern(A);
  }
  void factorize_sparse(const SpMat& A) {
    if (strumpack()) { sp_.factorize(A); return; }
#ifdef AXF_HAVE_METIS
    if (metis()) { m_.factorize(A); return; }
#endif
    a_.factorize(A);
  }
  Vec solve_sparse(const Vec& b) const {
    if (strumpack()) return Vec(sp_.solve(Eigen::VectorXd(b)));
#ifdef AXF_HAVE_METIS
    if (metis()) return Vec(m_.solve(b));
#endif
    return Vec(a_.solve(b));
  }
  // several right-hand sides at once: blocked_ldlt_solve reads the factor once instead of once per
  // column, which is what the low-rank update of the exterior map needs (N = dtn modes columns)
  Eigen::MatrixXd solve_sparse(const Eigen::MatrixXd& b) const {
    if (strumpack()) return sp_.solve(b);
#ifdef AXF_HAVE_METIS
    if (metis()) return blocked_ldlt_solve(m_, b);
#endif
    return blocked_ldlt_solve(a_, b);
  }

  std::string which_;
  SymSolver sp_;
  Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> a_;
#ifdef AXF_HAVE_METIS
  Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::MetisOrdering<int>> m_;
#endif
  SpMat V_;
  Vec c_;
  Eigen::FullPivLU<Eigen::MatrixXd> cap_;
  std::function<void(double)> pivot_guard_;
};

}  // namespace

// Constraint reduction u = P y + u0 plus the exterior (DtN) operator and the circuit couplings.
// Everything here depends on the mesh only through the DOF coordinates and the element geometry,
// so rebuilding it on a morphed mesh gives the perturbed discrete problem with the same unknowns.
struct PoloidalProblem::Reduction {
  DofMap dm;
  int nfree = 0, ng = 0, nred = 0, nm = 0, nc = 0, nd = 0, ne = 0;
  std::vector<int> gidx;        // master -> global unknown index (-1 if fixed)
  SpMat P;
  Vec u0, Iext;
  // The exterior map (DtN) is the rank-dtn_modes term  K += Udtn diag(dtn_coef) Udtn^T, with Udtn
  // nonzero only in the outer-boundary rows.  It is kept in that factored form -- applied to vectors
  // through dtn_apply(), and handed to the solver as a Woodbury update through Vdtn = P^T Udtn --
  // instead of being assembled into K: a dense |bdofs| x |bdofs| block makes the outer boundary a
  // clique of the sparsity graph, which costs 17-25 % of the factor nonzeros on every backend and makes
  // STRUMPACK warn about the depth of its elimination tree ("could lead to segmentation faults", which
  // the low-rank form does not provoke).  AXF_DTN_DENSE=1 restores the assembled
  // block in Kdtn as a reference path; the two agree to round-off (tests/test_solvers.py,
  // cpp/tests/test_solvers.cpp).
  bool dtn_dense = false;
  SpMat Kdtn;   // only with AXF_DTN_DENSE=1
  SpMat Udtn;   // nd x dtn_modes otherwise
  SpMat Vdtn;   // P^T Udtn, the same term in the reduced unknowns
  Vec dtn_coef;
  int dtn_modes = 0;
  Vec dtn_apply(const Vec& u) const {  // K_ext u
    if (dtn_dense) return Vec(Kdtn * u);
    if (Udtn.cols() == 0) return Vec(Vec::Zero(u.size()));
    return Vec(Udtn * Vec(dtn_coef.cwiseProduct(Vec(Udtn.transpose() * u))));
  }
  std::vector<Vec> cfull;       // circuit linkage vectors (full DOF space)
  Eigen::MatrixXd Cr;           // P^T cfull
  Vec lam0;
  // Background drive of a scattered-field formulation (empty without one, or when nothing responds to it):
  // the gradient of the cross terms int (nu - nu0) B_bg.B[u] over magnetic regions and of the jump part of
  // int nu0 B_bg.B[u] at thin layers (docs/physics/thin_magnetic_layer.md, "Background fields").
  Vec fbg;
  // Thin-layer quadrature points: per radian, the layer energy at a point is
  //   w [ Pinv/2 (A0 - A1)^2 + cn0 B0^2 + cn1 B1^2 ],  A_s = N.u_s,  B_s = G.u_s + beta,
  // with G_k = (1/r) d(r N_k)/dl the normal flux density of shape k and cn_s = (d/4)(nu - nu_s) the
  // half strip of side s (docs/physics/thin_magnetic_layer.md).
  struct LayerQ {
    int face = -1;  // index into lfaces_
    double w = 0, Pinv = 0, cn[2] = {0, 0}, beta = 0, r = 0;
    std::vector<double> N, G;
  };
  std::vector<LayerQ> lq;
  Vec full(const Vec& y) const { return Vec(P * y + u0); }
  Vec linkage(const Vec& u) const {
    Vec v(nc);
    for (int c = 0; c < nc; ++c) v(c) = cfull[c].dot(u) - lam0(c);
    return v;
  }
};

struct PoloidalProblem::Assembled {
  SpMat K;
  Vec grad;
  double energy = 0, energy_quad = 0;
  double energy_lin = 0;  // the background's linear work (in `energy`, not stored energy)
};

// Factorized tangent operator, cached for repeated adjoint solves.
struct PoloidalProblem::Tangent {
  std::unique_ptr<Reduction> red;
  SpMat Kr;
  std::unique_ptr<Cholesky> chol;  // the forward solve's solver and ordering
  Eigen::MatrixXd B, S;  // B = Kr^-1 Cr, S = Cr^T B (circuits)
};

PoloidalProblem::PoloidalProblem(const Space& sp, std::vector<ARole> roles, std::vector<Master> masters,
                                 std::map<int, int> sheet_master_by_attr)
    : sp_(sp), roles_(std::move(roles)), masters_(std::move(masters)), sheet_master_(std::move(sheet_master_by_attr)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: role table size mismatch");
}

PoloidalProblem::~PoloidalProblem() = default;

void PoloidalProblem::set_open_boundary(int outer_attr, double radius, int modes, double center_z) {
  outer_attr_ = outer_attr;
  Rb_ = radius;
  modes_ = modes;
  center_z_ = center_z;
}

void PoloidalProblem::field(const ASolution& s, int e, const PointEval& pe, double& Br, double& Bz,
                            double* A) const {
  if (roles_[e].master) {
    Br = Bz = 0;
    if (A) *A = pe.r > 0 ? s.psi[roles_[e].master_id] / pe.r : 0.0;
    return;
  }
  const auto& d = edofs(e);
  double a = 0, dr = 0, dz = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    double u = s.u(d[i]);
    a += u * pe.N[i];
    dr += u * pe.dN[2 * i];
    dz += u * pe.dN[2 * i + 1];
  }
  Br = -dz;
  Bz = dr + (pe.r > 0 ? a / pe.r : dr);  // on the axis A/r -> dA/dr
  if (A) *A = a;
}

// ---------------------------------------------------------------- reduction
std::unique_ptr<PoloidalProblem::Reduction> PoloidalProblem::reduce() const {
  auto out = std::make_unique<Reduction>();
  Reduction& R = *out;
  R.dtn_dense = [] {  // reference path for the exterior map
    const char* e = std::getenv("AXF_DTN_DENSE");
    return e && *e && std::string(e) != "0";
  }();
  const int nd = ndof();
  const int ne = sp_.mesh.GetNE();
  const int nm = static_cast<int>(masters_.size());
  // the space's DOF of each DOF (layer copies share the coordinates of their original)
  auto og = [&](int i) { return orig_.empty() ? i : orig_[i]; };
  R.nd = nd;
  R.ne = ne;
  R.nm = nm;

  DofMap& dm = R.dm;
  dm.kind.assign(nd, 0);
  dm.master.assign(nd, -1);
  dm.w.assign(nd, 0.0);
  dm.index.assign(nd, -1);
  auto tie = [&](int dof, int m, const std::string& what) {
    if (dm.master[dof] >= 0 && dm.master[dof] != m) {
      if (!layer_end_.empty() && layer_end_[og(dof)] >= 0) {
        const ThinLayer& L = layers_[layer_end_[og(dof)]];
        throw Error(strcat("superconducting bodies '", masters_[dm.master[dof]].name, "' and '", masters_[m].name,
                           "' meet at an end of the thin magnetic layer '", L.name, "' at (r, z) = (",
                           sp_.r[og(dof)], ", ", sp_.z[og(dof)], "): the layer separates them only along its faces"),
                    L.where() + "/path",
                    "Extend the layer past the contact (the bodies then meet only across the layer), or separate "
                    "or merge the bodies.");
      }
      throw Error(strcat("superconducting bodies '", masters_[dm.master[dof]].name, "' and '", masters_[m].name,
                         "' touch at (r, z) = (", sp_.r[og(dof)], ", ", sp_.z[og(dof)], ")"),
                  "/regions", "Touching superconductors form one body; separate them or merge them. (" + what + ")");
    }
    dm.master[dof] = m;
  };
  for (int e = 0; e < ne; ++e) {
    if (roles_[e].master) for (int d : edofs(e)) tie(d, roles_[e].master_id, "region");
  }
  mfem::Array<int> bd;
  for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
    auto it = sheet_master_.find(sp_.mesh.GetBdrAttribute(be));
    if (it == sheet_master_.end()) continue;
    sp_.fes->GetBdrElementDofs(be, bd);
    if (orig_.empty()) {
      for (int d : bd) tie(d, it->second, "curve");
      continue;
    }
    // With thin layers, a sheet face ties the copies that the two elements next to it use, as a region
    // ties its own elements' copies: a sheet that ends on one face of a layer (a T) leaves the far side's
    // copy free, so the layer's flux passes; a sheet that crosses the layer has faces on both sides and
    // ties both copies.
    int fe[2] = {-1, -1};
    sp_.mesh.GetFaceElements(sp_.mesh.GetBdrElementFaceIndex(be), &fe[0], &fe[1]);
    for (int e : fe) {
      if (e < 0) continue;
      const auto& d0 = sp_.edofs(e);
      const auto& de = edofs(e);
      for (int d : bd) {
        const long pos = std::find(d0.begin(), d0.end(), d) - d0.begin();
        if (pos < static_cast<long>(d0.size())) tie(de[pos], it->second, "curve");
      }
    }
  }
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e) {
    if (!roles_[e].master) for (int d : edofs(e)) used[d] = 1;
  }
  int nfree = 0;
  for (int i = 0; i < nd; ++i) {
    if (sp_.r[og(i)] == 0.0) {
      dm.kind[i] = 3;
      if (dm.master[i] >= 0 && !(masters_[dm.master[i]].fixed && masters_[dm.master[i]].psi == 0.0)) {
        throw Error("superconducting body '" + masters_[dm.master[i]].name + "' touches the symmetry axis",
                    "/superconductors", "A body on the axis has psi = 0; the solver fixes it automatically only when "
                    "set up by solve_design. Check custom master definitions.");
      }
    } else if (dm.master[i] >= 0) {
      dm.kind[i] = 2;
      dm.w[i] = 1.0 / sp_.r[og(i)];
    } else if (used[i]) {
      dm.kind[i] = 1;
      dm.index[i] = nfree++;
    }
  }
  dm.nfree = nfree;
  R.gidx.assign(nm, -1);
  int ng = 0;
  for (int m = 0; m < nm; ++m) if (!masters_[m].fixed) R.gidx[m] = ng++;
  R.nfree = nfree;
  R.ng = ng;
  R.nred = nfree + ng;
  AXF_REQUIRE(nfree > 0, "no free degrees of freedom: the domain is fully constrained");

  std::vector<Trip> pt;
  R.u0 = Vec::Zero(nd);
  for (int i = 0; i < nd; ++i) {
    if (dm.kind[i] == 1) pt.emplace_back(i, dm.index[i], 1.0);
    else if (dm.kind[i] == 2) {
      int m = dm.master[i];
      if (masters_[m].fixed) R.u0(i) = masters_[m].psi * dm.w[i];
      else pt.emplace_back(i, nfree + R.gidx[m], dm.w[i]);
      if (!tie_offset_.empty()) R.u0(i) -= tie_offset_[og(i)] * dm.w[i];
    }
  }
  R.P.resize(nd, R.nred);
  R.P.setFromTriplets(pt.begin(), pt.end());
  R.Iext = Vec::Zero(R.nred);
  for (int m = 0; m < nm; ++m) if (!masters_[m].fixed) R.Iext(nfree + R.gidx[m]) = masters_[m].current;

  // ---------------------------------------------------------- open boundary (DtN)
  std::vector<Trip> dtn_trip;
  if (outer_attr_ > 0 && modes_ > 0) {
    std::map<int, int> bidx;  // dof -> boundary index
    std::vector<int> bdofs;
    int nbe = 0;
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      if (sp_.mesh.GetBdrAttribute(be) != outer_attr_) continue;
      ++nbe;
      sp_.fes->GetBdrElementDofs(be, bd);
      for (int d : bd) if (!bidx.count(d)) { bidx[d] = static_cast<int>(bdofs.size()); bdofs.push_back(d); }
    }
    const int N = std::min(modes_, std::max(4, nbe));
    R.dtn_modes = N;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(bdofs.size(), N);
    std::vector<double> Pn;
    mfem::Vector shape, x(2);
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      if (sp_.mesh.GetBdrAttribute(be) != outer_attr_) continue;
      sp_.fes->GetBdrElementDofs(be, bd);
      const mfem::FiniteElement* fe = sp_.fes->GetBE(be);
      mfem::ElementTransformation* T = sp_.mesh.GetBdrElementTransformation(be);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * sp_.order + 14);
      shape.SetSize(fe->GetDof());
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        fe->CalcShape(ip, shape);
        double r = x(0), zz = x(1);
        // polar angle about the domain centre (0, center_z_), which is where the multipole
        // expansion of the exterior field is taken
        double theta = std::atan2(r, zz - center_z_);
        double dl = ip.weight * T->Weight();
        legendre_p1(std::cos(theta), N, Pn);
        for (int i = 0; i < bd.Size(); ++i) {
          int bi = bidx[bd[i]];
          double c = shape(i) * r * dl / Rb_;
          for (int n = 1; n <= N; ++n) U(bi, n - 1) += c * Pn[n];
        }
      }
    }
    // K_bb += (1/(mu0 Rb)) sum_n n/N_n U_n U_n^T
    Eigen::VectorXd coef(N);
    for (int n = 1; n <= N; ++n) coef(n - 1) = (1.0 / (mu0 * Rb_)) * n * (2.0 * n + 1.0) / (2.0 * n * (n + 1.0));
    R.dtn_coef = coef;
    if (R.dtn_dense) {  // reference path: the same operator assembled into the matrix
      Eigen::MatrixXd D = U * coef.asDiagonal() * U.transpose();
      for (size_t a = 0; a < bdofs.size(); ++a)
        for (size_t b = 0; b < bdofs.size(); ++b) dtn_trip.emplace_back(bdofs[a], bdofs[b], D(a, b));
    } else {  // nd x N with the boundary rows of U: K_ext = Udtn diag(coef) Udtn^T
      std::vector<Trip> ut;
      ut.reserve(bdofs.size() * static_cast<size_t>(N));
      for (size_t a = 0; a < bdofs.size(); ++a)
        for (int n = 0; n < N; ++n)
          if (U(a, n) != 0.0) ut.emplace_back(bdofs[a], n, U(a, n));
      R.Udtn.resize(nd, N);
      R.Udtn.setFromTriplets(ut.begin(), ut.end());
    }
  }
  R.Kdtn.resize(nd, nd);
  R.Kdtn.setFromTriplets(dtn_trip.begin(), dtn_trip.end());
  // the exterior map in the reduced unknowns: sparse, because the outer-boundary DOFs are free DOFs
  // (or tied to a body), so at most |bdofs| x N nonzeros
  if (R.Udtn.cols() > 0) R.Vdtn = (R.P.transpose() * R.Udtn).pruned();

  // ---------------------------------------------------------- persistent circuits
  // c_full[c](i) = sum_parts (turns/area) int N_i r dr dz;  lambda_c / (2 pi) = c_full[c] . u
  const int nc = static_cast<int>(coils_.size());
  R.nc = nc;
  R.cfull.assign(nc, Vec::Zero(nd));
  R.Cr.resize(R.nred, nc);
  R.lam0.resize(nc);
  const RefTable& tab = sp_.table(sp_.quad_rule());
  for (int c = 0; c < nc; ++c) {
    PointEval pe;
    for (auto& part : coils_[c].parts) {
      AXF_REQUIRE(part.area > 0, "circuit '" + coils_[c].name + "' has a winding of zero area");
      const double k = part.turns / part.area;
      for (int e : part.elements) {
        const auto& d = edofs(e);
        for (int q = 0; q < tab.nq; ++q) {
          sp_.eval(e, tab, q, pe);
          for (size_t i = 0; i < d.size(); ++i) R.cfull[c](d[i]) += k * pe.w * pe.r * pe.N[i];
        }
      }
    }
    R.Cr.col(c) = R.P.transpose() * R.cfull[c];
    R.lam0(c) = coils_[c].flux_linkage / (2 * pi);
  }

  // ---------------------------------------------------------- thin layers
  for (size_t fi = 0; fi < lfaces_.size(); ++fi) {
    const LayerFace& F = lfaces_[fi];
    const ThinLayer& L = layers_[F.layer];
    const double dthk = L.thickness, mu = L.mu_r * mu0, nu = 1.0 / mu;
    // the medium the layer displaces: a half strip of width d/2 on each side (a superconducting side:
    // mu_s = 0 and no normal correction there, since B_n = 0 on it)
    double mus = 0, cn[2] = {0, 0};
    for (int k = 0; k < 2; ++k) {
      const ARole& ro = roles_[F.e[k]];
      if (ro.master) continue;
      mus += 0.5 / ro.nu;
      cn[k] = 0.25 * dthk * (nu - ro.nu);
    }
    // Both sides superconducting: with two different bodies the layer between them carries their flux
    // difference, [A]^2/(2P) with P = mu d (no displaced medium); with one body both copies are tied alike.
    const double P = dthk * (mu - mus);
    if (!(P > 0)) {
      throw Error(strcat("thin layer '", L.name, "' (mu_r = ", L.mu_r, ") lies in a medium at least as permeable ",
                         "as itself (mean mu_r of the two sides ", mus / mu0, ")"),
                  L.where() + "/material", "A thin layer must be more permeable than the material around it; mesh the parts "
                             "as regions instead.");
    }
    const mfem::FiniteElement* fe = sp_.fes->GetBE(F.be);
    mfem::ElementTransformation* T = sp_.mesh.GetBdrElementTransformation(F.be);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * sp_.order + 4);
    const int n = fe->GetDof();
    mfem::Vector shape(n), x(2);
    mfem::DenseMatrix dshape(n, 1);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const double r = x(0);
      if (r <= sp_.rtol) continue;
      fe->CalcShape(ip, shape);
      fe->CalcDShape(ip, dshape);
      const mfem::DenseMatrix& J = T->Jacobian();
      const double jl = T->Weight();  // |dx/dxi|
      const double tr = J(0, 0) / jl, tz = J(1, 0) / jl;
      Reduction::LayerQ Q;
      Q.face = static_cast<int>(fi);
      Q.r = r;
      Q.w = ip.weight * jl * r;
      Q.Pinv = 1.0 / P;
      Q.cn[0] = cn[0];
      Q.cn[1] = cn[1];
      Q.N.resize(n);
      Q.G.resize(n);
      for (int k = 0; k < n; ++k) {
        Q.N[k] = shape(k);
        Q.G[k] = dshape(k, 0) / jl + shape(k) * tr / r;  // (1/r) d(r N_k)/dl
      }
      if (bg_B_) {
        double br, bz;
        bg_B_(r, x(1), br, bz);
        Q.beta = tr * bz - tz * br;  // (1/r) d(r A_bg)/dl
      }
      R.lq.push_back(std::move(Q));
    }
  }

  // ---------------------------------------------------------- background drive
  if (bg_B_) {
    Vec f = Vec::Zero(nd);
    bool any = false;
    PointEval pe;
    const double nu0 = 1.0 / mu0;
    for (int e = 0; e < ne; ++e) {
      const ARole& ro = roles_[e];
      const auto& d = edofs(e);
      const auto& d0 = sp_.edofs(e);
      const bool split = !ed_.empty() && d != d0;
      // Superconducting elements that hold layer copies take part in the jump part (with no volume term):
      // the dropped sum of nu0 B_bg.B[w] over the original numbering is legitimate only if w is tied in
      // them, which depends on which side kept the original number.
      if (ro.master && !split) continue;
      // nonlinear iron with a background is refused by the caller; the RF-linearized nu_at never has one
      const double dnu = (ro.master || ro.bh || ro.nu_at) ? 0.0 : ro.nu - nu0;
      if (dnu == 0.0 && !split) continue;
      any = true;
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double br, bz;
        bg_B_(pe.r, pe.z, br, bz);
        const double wr = pe.w * pe.r;
        for (size_t i = 0; i < d.size(); ++i) {
          const double bb = br * (-pe.dN[2 * i + 1]) + bz * (pe.dN[2 * i] + pe.N[i] / pe.r);
          if (dnu != 0.0) f(d[i]) += wr * dnu * bb;
          if (split && d[i] != d0[i]) {  // the jump part of int nu0 B_bg.B[u]
            f(d[i]) += wr * nu0 * bb;
            f(d0[i]) -= wr * nu0 * bb;
          }
        }
      }
    }
    // a layer's normal term with the total B_n = B_n[u] + beta: linear part 2 cn_s beta B_s
    for (const auto& Q : R.lq) {
      if (Q.beta == 0.0) continue;
      any = true;
      const LayerFace& F = lfaces_[Q.face];
      for (int s2 = 0; s2 < 2; ++s2)
        for (size_t k = 0; k < Q.G.size(); ++k) f(F.dofs[s2][k]) += Q.w * 2.0 * Q.cn[s2] * Q.beta * Q.G[k];
    }
    if (any) R.fbg = f;
  }
  return out;
}

// ---------------------------------------------------------------- assembly
void PoloidalProblem::assemble_at(const Reduction& R, const Vec& u, bool want_K, Assembled& out) const {
  const int nd = R.nd, ne = R.ne;
  out.energy = out.energy_quad = out.energy_lin = 0;
  const RefTable& tab = sp_.table(sp_.quad_rule());
  const int nth = omp_get_max_threads();
  std::vector<std::vector<Trip>> trips(want_K ? nth : 0);
  std::vector<Vec> grads(nth, Vec::Zero(nd));
  std::vector<double> en(nth, 0.0), enq(nth, 0.0);
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    PointEval pe;
    std::vector<double> br, bz;
    Eigen::MatrixXd Ke;
    Eigen::VectorXd ge;
    Vec& gr = grads[tid];
#pragma omp for schedule(dynamic, 256)
    for (int e = 0; e < ne; ++e) {
      const ARole& ro = roles_[e];
      if (ro.master) continue;
      const auto& d = edofs(e);
      const int n = static_cast<int>(d.size());
      Ke.setZero(n, n);
      ge.setZero(n);
      br.resize(n);
      bz.resize(n);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        const double r = pe.r, wr = pe.w * r;
        double A = 0, Br = 0, Bz = 0;
        for (int i = 0; i < n; ++i) {
          br[i] = -pe.dN[2 * i + 1];
          bz[i] = pe.dN[2 * i] + pe.N[i] / r;
          A += u(d[i]) * pe.N[i];
          Br += u(d[i]) * br[i];
          Bz += u(d[i]) * bz[i];
        }
        const double B = std::sqrt(Br * Br + Bz * Bz);
        double nu, dnu = 0, wdens;
        if (ro.bh) {
          nu = ro.bh->nu(B);
          dnu = ro.bh->dnu_dB2(B);
          wdens = ro.bh->energy(B);
        } else {
          nu = ro.nu_at ? ro.nu_at(e, pe) : ro.nu;
          wdens = 0.5 * nu * B * B;
        }
        en[tid] += wr * (wdens - ro.J * A);
        enq[tid] += wr * 0.5 * nu * B * B;
        for (int i = 0; i < n; ++i) {
          const double bi = Br * br[i] + Bz * bz[i];
          ge(i) += wr * (nu * bi - ro.J * pe.N[i]);
          if (!want_K) continue;
          for (int j = 0; j < n; ++j) {
            const double bj = Br * br[j] + Bz * bz[j];
            Ke(i, j) += wr * (nu * (br[i] * br[j] + bz[i] * bz[j]) + 2.0 * dnu * bi * bj);
          }
        }
      }
      for (int i = 0; i < n; ++i) {
        gr(d[i]) += ge(i);
        if (want_K)
          for (int j = 0; j < n; ++j) trips[tid].emplace_back(d[i], d[j], Ke(i, j));
      }
    }
  }
  out.grad = Vec::Zero(nd);
  for (int t = 0; t < nth; ++t) {
    out.grad += grads[t];
    out.energy += en[t];
    out.energy_quad += enq[t];
  }
  // thin layers (docs/physics/thin_magnetic_layer.md)
  if (!R.lq.empty()) {
    std::vector<Trip> lt;
    for (const auto& Q : R.lq) {
      const LayerFace& F = lfaces_[Q.face];
      const int n = static_cast<int>(Q.N.size());
      double A[2] = {0, 0}, B[2] = {0, 0};
      for (int s = 0; s < 2; ++s)
        for (int k = 0; k < n; ++k) {
          const double v = u(F.dofs[s][k]);
          A[s] += Q.N[k] * v;
          B[s] += Q.G[k] * v;
        }
      const double jump = A[0] - A[1];
      // the reaction field's quadratic part (the background's linear part is in R.fbg)
      const double quad = Q.Pinv * 0.5 * jump * jump + Q.cn[0] * B[0] * B[0] + Q.cn[1] * B[1] * B[1];
      out.energy += Q.w * quad;
      out.energy_quad += Q.w * quad;
      for (int k = 0; k < n; ++k) {
        out.grad(F.dofs[0][k]) += Q.w * (Q.Pinv * jump * Q.N[k] + 2 * Q.cn[0] * B[0] * Q.G[k]);
        out.grad(F.dofs[1][k]) += Q.w * (-Q.Pinv * jump * Q.N[k] + 2 * Q.cn[1] * B[1] * Q.G[k]);
      }
      if (!want_K) continue;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          const double nn = Q.w * Q.Pinv * Q.N[i] * Q.N[j], gg = Q.w * Q.G[i] * Q.G[j];
          lt.emplace_back(F.dofs[0][i], F.dofs[0][j], nn + 2 * Q.cn[0] * gg);
          lt.emplace_back(F.dofs[1][i], F.dofs[1][j], nn + 2 * Q.cn[1] * gg);
          lt.emplace_back(F.dofs[0][i], F.dofs[1][j], -nn);
          lt.emplace_back(F.dofs[1][i], F.dofs[0][j], -nn);
        }
    }
    if (want_K) trips.push_back(std::move(lt));
  }
  // background drive (scattered-field form): a constant gradient, linear work in the energy
  if (R.fbg.size() > 0) {
    out.grad += R.fbg;
    const double lin = R.fbg.dot(u);
    out.energy += lin;
    out.energy_lin += lin;
  }
  // exterior (DtN) energy
  Vec Du = R.dtn_apply(u);
  out.grad += Du;
  double edtn = 0.5 * u.dot(Du);
  out.energy += edtn;
  out.energy_quad += edtn;
  if (want_K) {
    size_t tot = 0;
    for (auto& v : trips) tot += v.size();
    std::vector<Trip> all;
    all.reserve(tot);
    for (auto& v : trips) all.insert(all.end(), v.begin(), v.end());
    out.K.resize(nd, nd);
    out.K.setFromTriplets(all.begin(), all.end());
    if (R.dtn_dense) out.K += R.Kdtn;  // otherwise the exterior map stays outside the sparse matrix
  }
}

// ---------------------------------------------------------------- outputs of a state
ASolution PoloidalProblem::outputs(const Reduction& R, const Vec& y, const Vec& Icur) const {
  ASolution s;
  const int nd = R.nd, nm = R.nm, nc = R.nc;
  Vec u = R.full(y);
  Assembled a;
  assemble_at(R, u, false, a);
  s.u.SetSize(nd);
  for (int i = 0; i < nd; ++i) s.u(i) = u(i);
  s.y.assign(y.data(), y.data() + y.size());
  s.psi.assign(nm, 0.0);
  s.current.assign(nm, 0.0);
  for (int m = 0; m < nm; ++m) s.psi[m] = masters_[m].fixed ? masters_[m].psi : y(R.nfree + R.gidx[m]);
  // reaction (net azimuthal current) on each body: sum_i w_i dE/du_i
  for (int i = 0; i < nd; ++i) {
    if (R.dm.kind[i] != 2) continue;
    double gi = a.grad(i);
    for (int c = 0; c < nc; ++c) gi -= Icur(c) * R.cfull[c](i);
    s.current[R.dm.master[i]] += R.dm.w[i] * gi;
  }
  s.coil_current.assign(Icur.data(), Icur.data() + nc);
  {  // reduced residual of this state (free of charge: the gradient is already assembled)
    Vec gv = R.P.transpose() * a.grad - R.Iext;
    if (nc > 0) gv -= R.Cr * Icur;
    s.residual_y.assign(gv.data(), gv.data() + gv.size());
    if (nc > 0) {
      Vec gi = -R.linkage(u);
      s.residual_circuit.assign(gi.data(), gi.data() + gi.size());
    }
  }
  // stored energy alone (the assembled energy includes the source work -int J A)
  double src = 0;
  {
    PointEval pe;
    const RefTable& tab = sp_.table(sp_.quad_rule());
    for (int e = 0; e < R.ne; ++e) {
      if (roles_[e].master || roles_[e].J == 0.0) continue;
      const auto& d = edofs(e);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double A = 0;
        for (size_t i = 0; i < d.size(); ++i) A += u(d[i]) * pe.N[i];
        src += pe.w * pe.r * roles_[e].J * A;
      }
    }
  }
  s.energy = 2.0 * pi * (a.energy + src - a.energy_lin);
  s.energy_quadratic = 2.0 * pi * a.energy_quad;
  json circ = json::array();
  {
    Vec lk = R.linkage(u);
    for (int c = 0; c < nc; ++c) {
      // The achieved linkage is a cancellation: for the usual target of 0 Wb the individual
      // contributions are large and sum to round-off.  Report the scale of the integral that
      // produced it (the sum of the absolute contributions), so that a correctly enforced
      // constraint can be told from a failure to enforce one -- comparing the residual with the
      // target alone would flag every converged zero-flux circuit.
      double absscale = 0;
      for (int i = 0; i < nd; ++i) absscale += std::fabs(R.cfull[c](i) * u(i));
      circ.push_back({{"name", coils_[c].name}, {"current_A", Icur(c)},
                      {"flux_linkage_Wb", 2 * pi * (lk(c) + R.lam0(c))}, {"target_Wb", coils_[c].flux_linkage},
                      {"linkage_scale_Wb", 2 * pi * absscale}});
    }
  }
  s.diagnostics = {{"dofs", nd}, {"free_dofs", R.nfree}, {"globals", R.ng}, {"masters", nm},
                   {"circuits", circ}};
  return s;
}

ASolution PoloidalProblem::solve(int verbose, double newton_tol, int max_newton) {
  Timer timer;
  auto red = reduce();
  const Reduction& R = *red;
  const int nc = R.nc;
  const bool nonlinear = std::any_of(roles_.begin(), roles_.end(), [](const ARole& r) { return !r.master && r.bh; });

  // Reduced Lagrangian L(y, I) = E(P y + u0) - Iext . y - I . (lambda(y)/2pi - lambda0/2pi)
  Vec y = Vec::Zero(R.nred);
  Vec Icur = Vec::Zero(nc);
  json hist = json::array();
  // Eigen LDLT (AMD or METIS ordering), or STRUMPACK for large systems (never with thin layers)
  Cholesky chol(ordering_choice(R.nred, !lfaces_.empty()));
  if (!lfaces_.empty()) chol.set_pivot_guard([this](double mp) { layer_pivot_guard(mp); });
  // the exterior map stays outside the sparse factorization (Woodbury) unless AXF_DTN_DENSE=1
  if (R.Vdtn.cols() > 0) chol.set_low_rank(R.Vdtn, R.dtn_coef);
  double symbolic_s = 0, factor_s = 0, solve_res = 0;
  long fill = 0;
  bool converged = false;
  int it = 0;
  double res0 = -1, res = 0;
  Assembled a;
  for (; it < max_newton; ++it) {
    Vec u = R.full(y);
    assemble_at(R, u, true, a);
    Vec g = R.P.transpose() * a.grad - R.Iext;
    if (nc > 0) g -= R.Cr * Icur;
    Vec gI = nc > 0 ? Vec(-R.linkage(u)) : Vec();
    res = std::sqrt(g.squaredNorm() + (nc > 0 ? gI.squaredNorm() : 0.0));
    if (res0 < 0) res0 = std::max(res, 1e-300);
    SpMat Kr = R.P.transpose() * a.K * R.P;
    if (it == 0) {
      Timer ts;
      chol.analyzePattern(Kr);
      symbolic_s = ts.seconds();
    }
    {
      Timer tf;
      chol.factorize(Kr);
      factor_s += tf.seconds();
    }
    // AXF_TEST_AXI_FACTORIZATION_FAILURE=1 reports a failed factorization (tests of the SolverFailure path)
    const char* forced = std::getenv("AXF_TEST_AXI_FACTORIZATION_FAILURE");
    const bool forced_failure = forced && std::string(forced) == "1";
    if (chol.info() != Eigen::Success || forced_failure) {
      throw SolverFailure("the field system could not be factorized (singular or indefinite)", "",
                  "Typical causes: a floating region with no reference (e.g. iron or vacuum completely enclosed "
                  "without a defined state), or a zero-volume region.");
    }
    if (it == 0) fill = chol.nonzeros_L();
    Vec dy, dI;
    if (nc > 0) {
      // Newton on [g; gI] = 0 with gI = dL/dI = -linkage:
      //   K dy - C dI = -g ;  -C^T dy = -gI   ->  dy = a0 + B dI,  (C^T B) dI = gI - C^T a0
      Vec a0 = chol.solve(Vec(-g));
      Eigen::MatrixXd B = chol.solve(R.Cr);
      Eigen::MatrixXd S = R.Cr.transpose() * B;
      dI = S.ldlt().solve(gI - R.Cr.transpose() * a0);
      dy = a0 + B * dI;
    } else {
      dy = chol.solve(Vec(-g));
    }
    {
      // Check the solve instead of trusting it: the Newton system is
      //   K dy - Cr dI = -g,   -Cr^T dy = -gI,
      // with K = Kr + the exterior-map update, so this also verifies the Woodbury application of that
      // update.  Each block residual is measured against the size of the terms that must cancel, not
      // against one of them: with a persistent circuit the first iterate has g = 0 exactly (all the
      // drive is in the flux-linkage constraint), and the constraint's own target is often exactly
      // zero.  Measured 1e-14 to 6e-10 from 2.6e4 to 1.9e5 unknowns (it grows with the condition
      // number); anything near 1 means the factor is unusable.
      const Vec Kdy = chol.apply(Kr, dy);
      Vec r = Kdy + g;
      double scale = Kdy.norm() + g.norm();
      if (nc > 0) {
        const Vec CdI = R.Cr * dI;
        r -= CdI;
        scale += CdI.norm();
      }
      double rel = r.norm() / std::max(scale, 1e-300);
      if (nc > 0) {
        // the multiplier block, componentwise against |Cr|^T |dy| (the terms that cancel in the dot
        // product), which is main's linkage_scale_Wb in the same spirit
        const Vec Cty = R.Cr.transpose() * dy;
        const Vec aty = R.Cr.cwiseAbs().transpose() * dy.cwiseAbs();
        for (int c = 0; c < nc; ++c) {
          const double den = std::max(std::max(std::fabs(gI(c)), aty(c)), 1e-300);
          rel = std::max(rel, std::fabs(gI(c) - Cty(c)) / den);
        }
      }
      solve_res = std::max(solve_res, rel);
      if (!(rel < 1e-5)) {
        throw SolverFailure(strcat("the factorized field system did not solve (relative residual ", rel, " with the ",
                                   chol.solver_name(), " backend and the ", chol.ordering(), " ordering)"),
                            "", "Try a different solver backend (AXF_ORDERING=amd|metis|strumpack, or "
                                "AXF_STRUMPACK_MIN_DOFS) and please report the design.");
      }
    }
    const Vec Inew = nc > 0 ? Vec(Icur + dI) : Icur;
    auto merit = [&](const Vec& yy, const Vec& uu, double E) {
      double v = E - R.Iext.dot(yy);
      if (nc > 0) v -= Inew.dot(R.linkage(uu));
      return v;
    };
    const double Pi = merit(y, u, a.energy);
    double step = 1.0, Pnew = Pi;
    // line search (Armijo) on the Lagrangian at the updated multipliers; convex in y for monotone B-H laws
    Vec gm = R.P.transpose() * a.grad - R.Iext;
    if (nc > 0) gm -= R.Cr * Inew;
    const double slope = gm.dot(dy);
    if (nonlinear) {
      // Armijo decrease of the Lagrangian. Near convergence the energy change drops below round-off;
      // there a step is accepted only if it reduces the residual norm (a full step can overshoot on
      // piecewise B-H laws while the energy looks unchanged).
      Assembled an;
      for (int ls = 0; ls < 30; ++ls) {
        Vec yn = y + step * dy;
        Vec un = R.full(yn);
        assemble_at(R, un, false, an);
        Pnew = merit(yn, un, an.energy);
        if (Pnew <= Pi + 1e-4 * step * slope) break;
        if (std::fabs(Pnew - Pi) <= 1e-12 * std::fabs(Pi)) {
          Vec gn = R.P.transpose() * an.grad - R.Iext;
          if (nc > 0) gn -= R.Cr * Inew;
          double rn = gn.squaredNorm();
          if (nc > 0) rn += R.linkage(un).squaredNorm();
          if (std::sqrt(rn) < res) break;
        }
        step *= 0.5;
      }
    }
    y += step * dy;
    if (nc > 0) Icur = Inew;
    double dnorm = step * dy.norm() / std::max(y.norm(), 1e-300);
    hist.push_back({{"iter", it}, {"residual", res}, {"step", step}, {"update", dnorm}, {"energy_per_rad", Pi}});
    if (verbose > 1) log(2, strcat("  newton ", it, " res ", res, " step ", step, " du ", dnorm));
    if (!nonlinear) {
      converged = true;
      ++it;
      break;
    }
    // converged: small update from an (almost) full Newton step, and a reduced residual.
    // A tiny update from a collapsed line search is stagnation, not convergence.
    if (dnorm < newton_tol && step >= 0.25 && (res <= 1e-8 * res0 || dnorm < 1e-13)) {
      converged = true;
      ++it;
      break;
    }
  }
  if (!converged) {
    log(1, strcat("warning: Newton did not converge in ", max_newton, " iterations (residual ", res, ")"));
  }

  // ------------------------------------------------------------ outputs
  ASolution s = outputs(R, y, Icur);
  {
    // residual of the returned solution (not of the last Newton iterate)
    double rf = 0;
    for (double v : s.residual_y) rf += v * v;
    for (double v : s.residual_circuit) rf += v * v;
    res = std::sqrt(rf);
    if (nonlinear && !converged && res <= 1e-8 * res0) converged = true;  // converged on the final update
  }
  s.diagnostics["nonlinear"] = nonlinear;
  s.diagnostics["newton_iterations"] = it;
  s.diagnostics["converged"] = converged;
  s.diagnostics["final_residual"] = res;
  s.diagnostics["relative_residual"] = res / std::max(res0, 1e-300);
  s.diagnostics["history"] = hist;
  s.diagnostics["dtn_modes"] = R.dtn_modes;
  s.diagnostics["dtn_assembly"] = R.dtn_modes == 0 ? "none" : (R.dtn_dense ? "dense" : "low-rank");
  s.diagnostics["solver"] = chol.solver_name();
  s.diagnostics["backend"] = chol.backend();
  s.diagnostics["ordering"] = chol.ordering();
  s.diagnostics["symbolic_analyses"] = chol.symbolic_analyses();
  s.diagnostics["reused_factorizations"] = chol.reused_factorizations();
  s.diagnostics["solve_residual"] = solve_res;
  s.diagnostics["reduced_unknowns"] = R.nred;
  s.diagnostics["nonzeros_L"] = fill;
  s.diagnostics["symbolic_seconds"] = symbolic_s;
  s.diagnostics["factorization_seconds"] = factor_s;
  s.diagnostics["seconds"] = timer.seconds();
  return s;
}

// ---------------------------------------------------------------- sensitivity support
void PoloidalProblem::residual_impl(const Reduction& R, const Vec& y, const Vec& Icur, std::vector<double>& g,
                                    std::vector<double>& gc) const {
  Vec u = R.full(y);
  Assembled a;
  assemble_at(R, u, false, a);
  Vec gv = R.P.transpose() * a.grad - R.Iext;
  if (R.nc > 0) gv -= R.Cr * Icur;
  g.assign(gv.data(), gv.data() + gv.size());
  gc.clear();
  if (R.nc > 0) {
    Vec gi = -R.linkage(u);
    gc.assign(gi.data(), gi.data() + gi.size());
  }
}

ASolution PoloidalProblem::evaluate(const std::vector<double>& y, const std::vector<double>& coil_current) const {
  auto red = reduce();
  AXF_REQUIRE(static_cast<int>(y.size()) == red->nred,
              strcat("frozen state has ", y.size(), " unknowns but the problem has ", red->nred,
                     ": the meshes are not equivalent"));
  AXF_REQUIRE(static_cast<int>(coil_current.size()) == red->nc, "frozen state has the wrong number of circuit currents");
  return outputs(*red, to_eigen(y, red->nred), to_eigen(coil_current, red->nc));
}

void PoloidalProblem::residual(const std::vector<double>& y, const std::vector<double>& coil_current,
                               std::vector<double>& g, std::vector<double>& g_circuit) const {
  auto red = reduce();
  AXF_REQUIRE(y.empty() || static_cast<int>(y.size()) == red->nred, "frozen state has the wrong size");
  // empty y: the zero state (the residual is then minus the source)
  residual_impl(*red, y.empty() ? Vec(Vec::Zero(red->nred)) : to_eigen(y, red->nred), to_eigen(coil_current, red->nc),
                g, g_circuit);
}

void PoloidalProblem::energy_gradient(const std::vector<double>& y, std::vector<double>& g,
                                      std::vector<double>& g_circuit) const {
  // W = pi u^T K u (linear media), u = P y + u0  ->  dW/dy = 2 pi P^T K u.  The assembled gradient
  // is K u - b, so the source term is added back.
  auto red = reduce();
  const Reduction& R = *red;
  Vec yv = to_eigen(y, R.nred);
  Vec u = R.full(yv);
  Assembled a;
  assemble_at(R, u, false, a);
  // The source vector b_i = int wr J N_i, which a second assembly without sources would also give,
  // is accumulated here instead.
  Vec b = Vec::Zero(R.nd);
  {
    PointEval pe;
    const RefTable& tab = sp_.table(sp_.quad_rule());
    for (int e = 0; e < R.ne; ++e) {
      if (roles_[e].master || roles_[e].J == 0.0) continue;
      const auto& d = edofs(e);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        for (size_t i = 0; i < d.size(); ++i) b(d[i]) += pe.w * pe.r * roles_[e].J * pe.N[i];
      }
    }
  }
  if (R.fbg.size() > 0) b -= R.fbg;  // the background drive is a source too
  Vec gv = 2.0 * pi * (R.P.transpose() * (a.grad + b));
  g.assign(gv.data(), gv.data() + gv.size());
  g_circuit.assign(R.nc, 0.0);
}

double PoloidalProblem::linear_functional(const std::vector<double>& y,
                                          const std::function<double(int, const PointEval&)>& w,
                                          std::vector<double>* grad) const {
  auto red = reduce();
  const Reduction& R = *red;
  Vec yv = to_eigen(y, R.nred);
  Vec u = R.full(yv);
  Vec c = Vec::Zero(R.nd);
  Vec gm = Vec::Zero(R.nred);
  double val = 0;
  PointEval pe;
  const RefTable& tab = sp_.table(sp_.quad_rule());
  for (int e = 0; e < R.ne; ++e) {
    const ARole& ro = roles_[e];
    const auto& d = edofs(e);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double wq = w(e, pe);
      if (wq == 0.0) continue;
      if (ro.master) {
        if (pe.r <= 0) continue;  // field() reports A = 0 on the axis; stay consistent with it
        // A = psi_m / r inside the body
        const int m = ro.master_id;
        const double coef = pe.w * wq;  // * r * (1/r)
        val += coef * (masters_[m].fixed ? masters_[m].psi : yv(R.nfree + R.gidx[m]));
        if (!masters_[m].fixed) gm(R.nfree + R.gidx[m]) += coef;
      } else {
        const double coef = pe.w * pe.r * wq;
        for (size_t i = 0; i < d.size(); ++i) {
          val += coef * pe.N[i] * u(d[i]);
          c(d[i]) += coef * pe.N[i];
        }
      }
    }
  }
  if (grad) {
    Vec gv = R.P.transpose() * c + gm;
    gv *= 2.0 * pi;
    grad->assign(gv.data(), gv.data() + gv.size());
  }
  return 2.0 * pi * val;
}

void PoloidalProblem::solve_tangent(const std::vector<double>& b, const std::vector<double>& bc,
                                    std::vector<double>& x, std::vector<double>& xc) const {
  const bool nonlinear = std::any_of(roles_.begin(), roles_.end(), [](const ARole& r) { return !r.master && r.bh; });
  AXF_REQUIRE(!nonlinear, "adjoint sensitivities need a linear problem (nonlinear iron is not supported)", "",
              "Use method='fd_remesh' (finite differences with remeshing) for nonlinear designs.");
  if (!tangent_) {
    tangent_ = std::make_unique<Tangent>();
    tangent_->red = reduce();
    const Reduction& R = *tangent_->red;
    Assembled a;
    assemble_at(R, Vec::Zero(R.nd), true, a);  // linear: K does not depend on u
    tangent_->Kr = SpMat(R.P.transpose() * a.K * R.P);
    tangent_->chol = std::make_unique<Cholesky>(ordering_choice(R.nred, !lfaces_.empty()));
    if (!lfaces_.empty()) tangent_->chol->set_pivot_guard([this](double mp) { layer_pivot_guard(mp); });
    // the exterior map is part of the tangent operator, and Kr does not contain it
    if (R.Vdtn.cols() > 0) tangent_->chol->set_low_rank(R.Vdtn, R.dtn_coef);
    tangent_->chol->analyzePattern(tangent_->Kr);
    tangent_->chol->factorize(tangent_->Kr);
    AXF_REQUIRE(tangent_->chol->info() == Eigen::Success, "the adjoint system could not be factorized");
    if (R.nc > 0) {
      tangent_->B = tangent_->chol->solve(R.Cr);
      tangent_->S = R.Cr.transpose() * tangent_->B;
    }
  }
  const Reduction& R = *tangent_->red;
  AXF_REQUIRE(static_cast<int>(b.size()) == R.nred, "tangent right-hand side has the wrong size");
  Vec bv = to_eigen(b, R.nred);
  Vec xv = tangent_->chol->solve(bv);
  Vec xcv = Vec::Zero(R.nc);
  if (R.nc > 0) {
    // [[K, -C], [-C^T, 0]] [x; xc] = [b; bc]
    Vec bcv = to_eigen(bc, R.nc);
    xcv = -tangent_->S.ldlt().solve(bcv + R.Cr.transpose() * xv);
    xv += tangent_->B * xcv;
  }
  x.assign(xv.data(), xv.data() + xv.size());
  xc.assign(xcv.data(), xcv.data() + xcv.size());
}

// ---------------------------------------------------------------- thin magnetic layers
int thin_layer_warn_ratio(int order) { return std::max(4, (order * order + 3) / 4); }

void PoloidalProblem::layer_pivot_guard(double min_pivot) const {
  // The half-strip term is negative; next to elements thinner than a few layer thicknesses it can make the
  // discrete energy indefinite (docs/physics/thin_magnetic_layer.md, "Validity conditions", item 4). The LDLT of
  // the symmetrically permuted matrix has the matrix's inertia, and the exterior map (kept out of it) is positive
  // semidefinite, so a non-positive pivot means the layered energy is not positive definite.
  if (min_pivot > 0) return;
  std::string names, where = "/curves";
  for (const auto& L : layers_) {
    names += (names.empty() ? "'" : ", '") + L.name + "'";
    if (layers_.size() == 1) where = L.where() + "/surface_size";
  }
  throw Error(strcat("the field system with the thin magnetic layer(s) ", names, " is not positive definite ",
                     "(smallest pivot ", min_pivot, "): a layer is too thick for the mesh next to it"),
              where,
              strcat("Coarsen the mesh at the layer: the elements next to it should be at least about ",
                     thin_layer_warn_ratio(sp_.order), " layer thicknesses high at mesh order ", sp_.order,
                     " (the curve's surface_size, and the surface_size or mesh_size of regions next to the layer, "
                     "which can make the mesh there finer; the results report thin_layers[*].min_h_perp_over_d), "
                     "or mesh the shield through its thickness as a region."));
}

void PoloidalProblem::set_thin_layers(const std::vector<ThinLayer>& layers,
                                      const std::function<std::string(int)>& region_of) {
  layers_ = layers;
  ed_.clear();
  orig_.clear();
  lfaces_.clear();
  layer_end_.clear();
  tangent_.reset();
  if (layers_.empty()) return;
  const mfem::Mesh& m = sp_.mesh;
  const int ne = m.GetNE(), nd0 = sp_.ndof();
  std::map<int, int> layer_of_attr;
  for (size_t l = 0; l < layers_.size(); ++l) {
    AXF_REQUIRE(layers_[l].mu_r > 1 && layers_[l].thickness > 0,
                strcat("internal: thin layer '", layers_[l].name, "' needs mu_r > 1 and a positive thickness"));
    layer_of_attr[layers_[l].attr] = static_cast<int>(l);
  }
  auto is_layer_face = [&](int f) { return layer_of_attr.count(sp_.face_bdr_attr[f]) > 0; };
  // outer-boundary DOFs (the exterior map has no copies)
  std::set<int> outer;
  mfem::Array<int> bd;
  for (int be = 0; be < m.GetNBE(); ++be) {
    if (m.GetBdrAttribute(be) != outer_attr_ && m.GetBdrAttribute(be) != kOuterAttr) continue;
    sp_.fes->GetBdrElementDofs(be, bd);
    for (int d : bd) outer.insert(d);
  }
  // the layer faces and their DOFs
  std::set<int> ldofs;
  std::map<int, int> ldof_layer;  // layer DOF -> a layer it belongs to
  std::map<int, std::vector<int>> ldof_face_elems;  // layer DOF -> the elements on both sides of its layer faces
  for (int be = 0; be < m.GetNBE(); ++be) {
    auto it = layer_of_attr.find(m.GetBdrAttribute(be));
    if (it == layer_of_attr.end()) continue;
    const ThinLayer& L = layers_[it->second];
    LayerFace F;
    F.be = be;
    F.layer = it->second;
    m.GetFaceElements(m.GetBdrElementFaceIndex(be), &F.e[0], &F.e[1]);
    if (F.e[0] < 0 || F.e[1] < 0) {
      throw Error(strcat("thin layer '", L.name, "' runs along the boundary of the domain (a face with field on one ",
                         "side only)"),
                  L.where() + "/path", "Keep thin layers inside the domain, off the symmetry axis and away from the outer "
                             "boundary (mesh.domain_radius).");
    }
    for (int k = 0; k < 2; ++k) {
      const ARole& ro = roles_[F.e[k]];
      if (!ro.master && (ro.bh || ro.nu_at)) {
        throw Error(strcat("thin layer '", L.name, "' touches nonlinear (or field-dependent) iron",
                           region_of ? " (region '" + region_of(F.e[k]) + "')" : std::string()),
                    L.where() + "/path",
                    "A thin layer may border vacuum, windings, linear materials and superconductors; mesh the "
                    "layer as a region next to iron.");
      }
    }
    sp_.fes->GetBdrElementDofs(be, bd);
    for (int d : bd) {
      if (outer.count(d)) {
        throw Error(strcat("thin layer '", L.name, "' touches the outer boundary of the domain"), L.where() + "/path",
                    "Increase mesh.domain_radius so that the layer lies inside it.");
      }
      ldofs.insert(d);
      ldof_layer[d] = it->second;
      ldof_face_elems[d].push_back(F.e[0]);
      ldof_face_elems[d].push_back(F.e[1]);
    }
    lfaces_.push_back(std::move(F));
  }
  // split each layer DOF into one copy per fan of elements that the layer faces separate around it (a
  // free end of a layer leaves one fan: no split there)
  std::map<int, std::vector<int>> star;
  for (int e = 0; e < ne; ++e)
    for (int d : sp_.edofs(e))
      if (ldofs.count(d)) star[d].push_back(e);
  layer_end_.assign(nd0, -1);
  std::map<int, std::map<int, int>> remap;  // element -> (space DOF -> copy)
  orig_.resize(nd0);
  for (int i = 0; i < nd0; ++i) orig_[i] = i;
  for (auto& [d, els] : star) {
    std::map<int, int> lab;
    for (int e : els) lab[e] = -1;
    int nf = 0;
    for (int e : els) {
      if (lab[e] >= 0) continue;
      const int id = nf++;
      std::vector<int> stack{e};
      lab[e] = id;
      while (!stack.empty()) {
        const int x = stack.back();
        stack.pop_back();
        for (auto [nb, f] : sp_.nbr[x]) {
          auto it = lab.find(nb);
          if (it == lab.end() || it->second >= 0 || is_layer_face(f)) continue;
          it->second = id;
          stack.push_back(nb);
        }
      }
    }
    if (nf == 1) {
      layer_end_[d] = ldof_layer.at(d);
      // An end of a layer: the two sides join, so [A] = 0 and the layer's flux vanishes here. That is right in
      // a non-magnetic medium, on a superconductor, on the axis, and also where the layer runs along or inside
      // a magnetic region up to its end (its flux then leaves through the faces, which the model resolves; the
      // end has the usual O(d/l) error). A magnetic region met only in front of the end would take the layer's
      // whole flux through the contact, which the model cannot pass: refuse that. Regions within 1 % of
      // mu_r = 1 do not count.
      const double nu_vac = 1.0 / mu0;
      auto magnetic = [&](int e) {
        const ARole& ro = roles_[e];
        if (ro.master) return false;
        return ro.bh != nullptr || static_cast<bool>(ro.nu_at) || std::fabs(nu_vac / ro.nu - 1.0) > 0.01;
      };
      std::set<int> along;  // the regions (attributes) that the end's layer faces run along
      for (int e : ldof_face_elems[d])
        if (magnetic(e)) along.insert(sp_.mesh.GetAttribute(e));
      for (int e : els) {
        if (!magnetic(e) || along.count(sp_.mesh.GetAttribute(e))) continue;
        const ThinLayer& L = layers_[ldof_layer.at(d)];
        throw Error(strcat("thin layer '", L.name, "' ends on the magnetic region '",
                           region_of ? region_of(e) : std::string("?"), "' at (r, z) = (", sp_.r[d], ", ",
                           sp_.z[d], "), which it meets only at its end: the model cannot pass the layer's flux "
                           "into the region through that contact, so the junction would carry none"),
                    L.where() + "/path",
                    "Mesh the junction as a region (the layer's end and the part it meets), make both parts thin "
                    "layers (curves meeting at a junction pass flux between them), or let the layer run along the "
                    "region's surface instead of ending on it.");
      }
    }
    for (int k = 1; k < nf; ++k) {
      const int nw = static_cast<int>(orig_.size());
      orig_.push_back(d);
      for (auto& [e, id] : lab)
        if (id == k) remap[e][d] = nw;
    }
  }
  ed_.resize(ne);
  for (int e = 0; e < ne; ++e) {
    ed_[e] = sp_.edofs(e);
    auto it = remap.find(e);
    if (it == remap.end()) continue;
    for (int& d : ed_[e]) {
      auto jt = it->second.find(d);
      if (jt != it->second.end()) d = jt->second;
    }
  }
  // the DOFs of each face on each side
  for (auto& F : lfaces_) {
    sp_.fes->GetBdrElementDofs(F.be, bd);
    for (int k = 0; k < 2; ++k) {
      const auto& d0 = sp_.edofs(F.e[k]);
      F.dofs[k].resize(bd.Size());
      for (int i = 0; i < bd.Size(); ++i) {
        const auto pos = std::find(d0.begin(), d0.end(), bd[i]) - d0.begin();
        AXF_REQUIRE(pos < static_cast<long>(d0.size()), "internal: a layer face DOF is not a DOF of its element");
        F.dofs[k][i] = ed_[F.e[k]][pos];
      }
    }
  }
}

json PoloidalProblem::thin_layer_report(const ASolution& s) const {
  json out = json::array();
  if (layers_.empty()) return out;
  const mfem::Mesh& m = sp_.mesh;
  const int nd0 = sp_.ndof();
  for (size_t l = 0; l < layers_.size(); ++l) {
    const ThinLayer& L = layers_[l];
    double len = 0, hmin = std::numeric_limits<double>::infinity(), jmax = 0, fmax = 0, Pmin = 0;
    int nfaces = 0;
    std::set<int> split;
    for (const auto& F : lfaces_) {
      if (F.layer != static_cast<int>(l)) continue;
      ++nfaces;
      for (int k = 0; k < 2; ++k)
        for (int d : F.dofs[k])
          if (d >= nd0) split.insert(d);
      mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(m).GetBdrElementTransformation(F.be);
      const mfem::FiniteElement* fe = sp_.fes->GetBE(F.be);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * sp_.order + 4);
      mfem::Vector shape(fe->GetDof()), x(2);
      double flen = 0;
      // the excess permeance as in reduce(): the displaced medium averaged over the field sides
      double mus = 0;
      for (int k = 0; k < 2; ++k)
        if (!roles_[F.e[k]].master) mus += 0.5 / roles_[F.e[k]].nu;
      const double P = L.thickness * (L.mu_r * mu0 - mus);
      if (Pmin == 0 || P < Pmin) Pmin = P;
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        flen += ip.weight * T->Weight();
        fe->CalcShape(ip, shape);
        double jump = 0;
        for (int i = 0; i < shape.Size(); ++i) jump += shape(i) * (s.u(F.dofs[0][i]) - s.u(F.dofs[1][i]));
        // H_t = [A]/P in the layer, B_t = mu H_t; flux carried in excess of the displaced medium 2 pi r [A]
        jmax = std::max(jmax, std::fabs(jump) / P);
        fmax = std::max(fmax, 2 * pi * x(0) * std::fabs(jump));
      }
      len += flen;
      // only the sides that carry a half-strip term (a field side) bear on definiteness
      for (int k = 0; k < 2; ++k)
        if (!roles_[F.e[k]].master) hmin = std::min(hmin, 2.0 * element_area(sp_, F.e[k]) / flen);
    }
    out.push_back({{"name", L.name},
                   {"mu_r", L.mu_r},
                   {"thickness_m", L.thickness},
                   {"length_m", len},
                   {"faces", nfaces},
                   {"split_dofs", split.size()},
                   {"min_h_perp_over_d", std::isfinite(hmin) ? hmin / L.thickness : 0.0},
                   {"P_min_H", Pmin},
                   {"H_layer_max_A_per_m", jmax},
                   {"B_layer_max_T", L.mu_r * mu0 * jmax},
                   {"excess_flux_max_Wb", fmax}});
  }
  return out;
}

}  // namespace axf::axi
