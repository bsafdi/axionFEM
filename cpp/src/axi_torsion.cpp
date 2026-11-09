// Vibrations: axisymmetric m = 0 torsional dynamics.  See axionfem/torsion.hpp.
#include "axionfem/torsion.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <map>
#include <string>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <random>

#include "axionfem/core.hpp"
#include "axionfem/omp_shim.hpp"
#include "axionfem/sym_solver.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using CSpMat = Eigen::SparseMatrix<std::complex<double>>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using CVec = Eigen::VectorXcd;
using units::pi;

struct Torsion::Impl {
  SpMat K, M;                                  // per radian, free DOFs only
  Eigen::SimplicialLDLT<SpMat> shifted;        // (K + sigma M)^{-1}
  double sigma = 0;
  bool factorized = false;
};

Torsion::Torsion(const Space& sp, std::vector<TorsionRole> roles)
    : sp_(sp), roles_(std::move(roles)), impl_(std::make_shared<Impl>()) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(),
              "torsion: one role per element is required");
  assemble();
}

void Torsion::assemble() {
  const int nd = sp_.ndof();
  const mfem::IntegrationRule& ir = sp_.quad_rule();
  const RefTable& tab = sp_.table(ir);
  PointEval pe;

  // A DOF is free when it is touched by the structure and is not on the axis.
  std::vector<char> touched(nd, 0);
  for (int e = 0; e < sp_.mesh.GetNE(); ++e) {
    if (!roles_[e].solid) continue;
    for (int d : sp_.edofs(e)) touched[d] = 1;
  }
  idx_.assign(nd, -1);
  nfree_ = 0;
  int n_axis = 0;
  for (int d = 0; d < nd; ++d) {
    if (!touched[d]) continue;
    if (sp_.r[d] <= sp_.rtol) { ++n_axis; continue; }  // u_phi(0) = 0
    idx_[d] = nfree_++;
  }
  AXF_REQUIRE(nfree_ > 0,
              "torsion: the structure has no free degrees of freedom", "",
              "The structure is the set of elements whose material has a \"mechanical\" block and a "
              "positive density.  Check the design's materials.");

  std::vector<Trip> tk, tm;
  tk.reserve(64 * nfree_);
  tm.reserve(64 * nfree_);
  Eigen::MatrixXd Ke, Me;
  for (int e = 0; e < sp_.mesh.GetNE(); ++e) {
    const TorsionRole& ro = roles_[e];
    if (!ro.solid) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    Ke.setZero(n, n);
    Me.setZero(n, n);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double wr = pe.w * pe.r;
      vol_m3_ += 2 * pi * wr;
      mass_kg_ += 2 * pi * wr * ro.rho;
      for (int i = 0; i < n; ++i) {
        const double dri = pe.dN[2 * i], dzi = pe.dN[2 * i + 1];
        const double hi = pe.r > sp_.rtol ? pe.N[i] / pe.r : dri;
        const double gi = dri - hi;                       // d_r u - u/r
        for (int j = 0; j < n; ++j) {
          const double drj = pe.dN[2 * j], dzj = pe.dN[2 * j + 1];
          const double hj = pe.r > sp_.rtol ? pe.N[j] / pe.r : drj;
          const double gj = drj - hj;
          Ke(i, j) += wr * ro.mu * (gi * gj + dzi * dzj);
          Me(i, j) += wr * ro.rho * pe.N[i] * pe.N[j];
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      const int a = idx_[d[i]];
      if (a < 0) continue;
      for (int j = 0; j < n; ++j) {
        const int b = idx_[d[j]];
        if (b < 0) continue;
        tk.emplace_back(a, b, Ke(i, j));
        tm.emplace_back(a, b, Me(i, j));
      }
    }
  }
  impl_->K.resize(nfree_, nfree_);
  impl_->M.resize(nfree_, nfree_);
  impl_->K.setFromTriplets(tk.begin(), tk.end());
  impl_->M.setFromTriplets(tm.begin(), tm.end());
  diag_ = json{{"free_dofs", nfree_},
               {"axis_dofs_constrained", n_axis},
               {"solid_elements", std::count_if(roles_.begin(), roles_.end(),
                                                [](const TorsionRole& r) { return r.solid; })},
               {"solid_volume_m3", vol_m3_},
               {"solid_mass_kg", mass_kg_},
               {"K_nonzeros", static_cast<long long>(impl_->K.nonZeros())},
               {"M_nonzeros", static_cast<long long>(impl_->M.nonZeros())}};
}

std::vector<double> Torsion::load_vector(const std::vector<double>& load) const {
  AXF_REQUIRE(static_cast<int>(load.size()) == sp_.ndof(),
              "torsion: the load must have one value per degree of freedom of the space");
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  std::vector<double> F(nfree_, 0.0);
  for (int e = 0; e < sp_.mesh.GetNE(); ++e) {
    if (!roles_[e].solid) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      double g = 0.0;
      for (int i = 0; i < n; ++i) g += pe.N[i] * load[d[i]];
      const double wr = pe.w * pe.r;
      for (int i = 0; i < n; ++i) {
        const int a = idx_[d[i]];
        if (a >= 0) F[a] += wr * pe.N[i] * g;
      }
    }
  }
  return F;
}

json Torsion::load_moments(const std::vector<double>& load) const {
  AXF_REQUIRE(static_cast<int>(load.size()) == sp_.ndof(),
              "torsion: the load must have one value per degree of freedom of the space");
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  double force = 0.0;
  double torque = 0.0;
  double zmin = std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();
  for (int e = 0; e < sp_.mesh.GetNE(); ++e) {
    if (!roles_[e].solid) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    bool loaded = false;
    for (int i = 0; i < n; ++i) loaded = loaded || load[d[i]] != 0.0;
    if (!loaded) continue;
    for (int i = 0; i < n; ++i) {
      zmin = std::min(zmin, sp_.z[d[i]]);
      zmax = std::max(zmax, sp_.z[d[i]]);
    }
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      double g = 0.0;
      for (int i = 0; i < n; ++i) g += pe.N[i] * load[d[i]];
      force += 2 * pi * pe.w * pe.r * g;
      torque += 2 * pi * pe.w * pe.r * pe.r * g;
    }
  }
  json out{{"force_N", force}, {"torque_N_m", torque},
           {"note", "integrals of the load as applied: the P_p interpolant of the nodal values over the "
                    "structure's elements (a step between two nodes is smeared over one element)"}};
  if (zmax >= zmin) out["support_z_m"] = {zmin, zmax};
  return out;
}

TorsionModes Torsion::modes(int nmodes, bool want_shapes, int lanczos_extra, int verbose) const {
  AXF_REQUIRE(nmodes > 0, "torsion: nmodes must be positive");
  nmodes = std::min(nmodes, nfree_);
  const SpMat& K = impl_->K;
  const SpMat& M = impl_->M;
  // Shift so that K + sigma M is positive definite even with the rigid-rotation zero mode.  sigma is
  // 1e-6 of the largest diagonal ratio, which grows as 1/h_min^2 and can exceed the lowest non-zero
  // eigenvalue (6x in the 20 m benchmark tube).  That costs no accuracy: lambda = 1/theta - sigma
  // has an absolute error of about eps sigma.
  const double scaleK = K.diagonal().cwiseAbs().maxCoeff();
  const double scaleM = M.diagonal().cwiseAbs().maxCoeff();
  const double sigma = 1e-6 * scaleK / scaleM;
  SpMat A = K;
  A += sigma * M;
  Eigen::SimplicialLDLT<SpMat> chol(A);
  AXF_REQUIRE(chol.info() == Eigen::Success,
              "torsion: the shifted stiffness matrix could not be factorized", "",
              "Typical causes: a region with mu = 0 or density = 0, or a zero-area region.");

  const int m = std::min(nfree_, nmodes + std::max(lanczos_extra, 2 * nmodes));
  Eigen::MatrixXd V(nfree_, m);
  Vec alpha(m), beta(m);
  std::mt19937 rng(12345);
  std::normal_distribution<double> nrm(0.0, 1.0);
  Vec v(nfree_);
  for (int i = 0; i < nfree_; ++i) v[i] = nrm(rng);
  auto Mnorm = [&](const Vec& x) { return std::sqrt(std::max(0.0, x.dot(M * x))); };
  v /= Mnorm(v);
  V.col(0) = v;
  Vec vprev = Vec::Zero(nfree_);
  double bprev = 0.0;
  double theta_scale = 0.0;   // max |alpha|, the scale of theta = 1/(lambda + sigma)
  for (int j = 0; j < m; ++j) {
    Vec w = chol.solve(M * V.col(j));          // B v_j
    alpha[j] = w.dot(M * V.col(j));
    theta_scale = std::max(theta_scale, std::fabs(alpha[j]));
    w -= alpha[j] * V.col(j);
    if (j > 0) w -= bprev * V.col(j - 1);
    // full reorthogonalisation in the M inner product (twice, for stability)
    for (int pass = 0; pass < 2; ++pass) {
      Vec Mw = M * w;
      for (int i = 0; i <= j; ++i) w -= (V.col(i).dot(Mw)) * V.col(i);
    }
    const double bj = Mnorm(w);
    beta[j] = bj;
    if (j + 1 < m) {
      if (bj < 1e-14 * theta_scale) {          // invariant subspace: restart with a random vector
        for (int i = 0; i < nfree_; ++i) w[i] = nrm(rng);
        for (int pass = 0; pass < 2; ++pass) {
          Vec Mw = M * w;
          for (int i = 0; i <= j; ++i) w -= (V.col(i).dot(Mw)) * V.col(i);
        }
        w /= Mnorm(w);
        beta[j] = 0.0;
      } else {
        w /= bj;
      }
      V.col(j + 1) = w;
      bprev = beta[j];
    }
  }
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
  for (int j = 0; j < m; ++j) {
    T(j, j) = alpha[j];
    if (j + 1 < m) {
      T(j, j + 1) = beta[j];
      T(j + 1, j) = beta[j];
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  // theta = 1/(lambda + sigma): the largest theta are the smallest lambda
  std::vector<int> order(m);
  for (int i = 0; i < m; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return es.eigenvalues()[a] > es.eigenvalues()[b]; });

  TorsionModes out;
  out.diagnostics = diag_;
  out.diagnostics["lanczos_steps"] = m;
  out.diagnostics["shift_rad2_s2"] = sigma;
  std::vector<double> lams;   // the Ritz values, unclipped (rigid-mode classification)
  for (int k = 0; k < nmodes; ++k) {
    const double th = es.eigenvalues()[order[k]];
    if (!(th > 0)) continue;
    const double lam = 1.0 / th - sigma;
    Vec x = V * es.eigenvectors().col(order[k]);
    const double mm = x.dot(M * x);
    if (mm <= 0) continue;
    x /= std::sqrt(2 * pi * mm);               // 2 pi x^T M x = 1 (unit modal mass)
    Vec Mx = M * x;
    Vec Kx = K * x;
    Vec res = Kx - lam * Mx;
    // Relative residual in the 2-norm, normalised by the size of the two terms that cancel.
    // (An inf-norm normalised by ||M x|| alone is meaningless: the mass and stiffness entries
    // differ by many orders of magnitude on a fine mesh.)
    const double denom = std::max(Kx.norm() + std::fabs(lam) * Mx.norm(), 1e-300);
    out.f_Hz.push_back(std::sqrt(std::max(lam, 0.0)) / (2 * pi));
    lams.push_back(lam);
    out.residual.push_back(res.norm() / denom);
    out.modal_mass.push_back(2 * pi * x.dot(M * x));
    if (want_shapes) {
      std::vector<double> sh(sp_.ndof(), 0.0);
      for (int d = 0; d < sp_.ndof(); ++d)
        if (idx_[d] >= 0) sh[d] = x[idx_[d]];
      out.shape.push_back(std::move(sh));
    }
  }
  // Unconverged Ritz pairs: relative residual above 1e-6, for the modes that are not rigid (a rigid
  // mode's relative residual is 0/0). A mode is rigid when its Ritz value is at most 1e-6 sigma, an
  // absolute threshold tied to the operator: rigid Ritz values round to |lam| <= ~1e-5 rad^2/s^2, the
  // first elastic eigenvalue is many orders above 1e-6 sigma (a threshold relative to the highest
  // returned mode would fail when that mode is itself rigid).
  const double lam_top = lams.empty() ? 0.0 : std::max(lams.back(), 0.0);
  std::vector<int> unconverged;
  int rigid = 0;
  for (size_t k = 0; k < lams.size(); ++k) {
    if (lams[k] <= 1e-6 * sigma) {
      ++rigid;
      continue;
    }
    if (!(out.residual[k] <= 1e-6)) unconverged.push_back(static_cast<int>(k));
  }
  out.diagnostics["rigid_modes"] = rigid;
  out.diagnostics["unconverged_modes"] = unconverged;
  // Sylvester inertia: the number of negative pivots of an LDL^T factorization of K - lam_bar M is
  // the number of eigenvalues below lam_bar.  With lam_bar just above the highest returned
  // eigenvalue it must equal the number of returned modes; more means a mode (or a copy of a
  // repeated one) was missed.
  if (!out.f_Hz.empty()) {
    const double lam_bar = lam_top * (1.0 + 1e-6) + 1e-6 * sigma;
    SpMat B = K;
    B -= lam_bar * M;
    Eigen::SimplicialLDLT<SpMat> ldl(B);
    json inertia{{"lambda_bar_rad2_s2", lam_bar}, {"returned", static_cast<int>(out.f_Hz.size())}};
    if (ldl.info() == Eigen::Success) {
      const Vec D = ldl.vectorD();
      int below = 0;
      for (int i = 0; i < D.size(); ++i)
        if (D[i] < 0) ++below;
      inertia["eigenvalues_below"] = below;
      inertia["complete"] = below == static_cast<int>(out.f_Hz.size());
    } else {
      inertia["complete"] = nullptr;
      inertia["note"] = "the LDL^T factorization of K - lam_bar M failed; completeness not checked";
    }
    out.diagnostics["inertia"] = inertia;
  }
  if (verbose) {
    for (size_t k = 0; k < out.f_Hz.size(); ++k)
      log(1, strcat("torsion mode ", k, ": f = ", out.f_Hz[k], " Hz, residual ", out.residual[k]));
  }
  return out;
}

std::vector<double> Torsion::participation(const TorsionModes& md,
                                           const std::vector<double>& load) const {
  return participation_F(md, load_vector(load));
}

std::vector<double> Torsion::participation_F(const TorsionModes& md, const std::vector<double>& F) const {
  std::vector<double> c;
  for (const auto& sh : md.shape) {
    double s = 0.0;
    for (int d = 0; d < sp_.ndof(); ++d)
      if (idx_[d] >= 0) s += F[idx_[d]] * sh[d];
    c.push_back(2 * pi * s);
  }
  return c;
}

std::vector<TorsionResponse> Torsion::harmonic(const std::vector<double>& load,
                                               const std::vector<double>& f_Hz, double eta,
                                               int verbose) const {
  TorsionHarmonicOptions opt;
  opt.eta = eta;
  opt.verbose = verbose;
  return harmonic_F(load_vector(load), f_Hz, opt);
}

void TorsionLoadMoments::add(const TorsionLoadMoments& o) {
  force_N += o.force_N;
  torque_N_m += o.torque_N_m;
  zmin = std::min(zmin, o.zmin);
  zmax = std::max(zmax, o.zmax);
}

json TorsionLoadMoments::to_json() const {
  json out{{"force_N", force_N}, {"torque_N_m", torque_N_m}};
  if (zmax >= zmin) out["support_z_m"] = {zmin, zmax};
  return out;
}

TorsionLoadMoments Torsion::add_volume_load(std::vector<double>& F, const std::vector<int>& elements,
                                            const TorsionLoadFn& g) const {
  AXF_REQUIRE(static_cast<int>(F.size()) == nfree_, "internal: torsion load vector has the wrong size");
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  TorsionLoadMoments mom;
  for (int e : elements) {
    if (!roles_[e].solid) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double gv = g(e, pe);
      if (gv == 0.0) continue;
      const double wr = pe.w * pe.r;
      for (int i = 0; i < n; ++i) {
        const int a = idx_[d[i]];
        if (a >= 0) F[a] += wr * pe.N[i] * gv;
      }
      mom.force_N += 2 * pi * wr * gv;
      mom.torque_N_m += 2 * pi * wr * pe.r * gv;
      mom.zmin = std::min(mom.zmin, pe.z);
      mom.zmax = std::max(mom.zmax, pe.z);
    }
  }
  return mom;
}

TorsionLoadMoments Torsion::add_face_load(std::vector<double>& F, const std::vector<std::pair<int, int>>& faces,
                                          const TorsionLoadFn& tau) const {
  AXF_REQUIRE(static_cast<int>(F.size()) == nfree_, "internal: torsion load vector has the wrong size");
  mfem::Mesh& m = sp_.mesh;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp_.order + 6);
  PointEval pe;
  TorsionLoadMoments mom;
  mfem::Vector nor(2), x(2);
  for (const auto& [face, elem] : faces) {
    AXF_REQUIRE(roles_[elem].solid, "internal: a torsion face load names an element outside the structure");
    int e1 = -1, e2 = -1;
    m.GetFaceElements(face, &e1, &e2);
    AXF_REQUIRE(elem == e1 || elem == e2, "internal: a torsion face load face does not touch its element");
    mfem::FaceElementTransformations* FT = m.GetFaceElementTransformations(face);
    const auto& d = sp_.edofs(elem);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      FT->SetAllIntPoints(&ip);
      mfem::CalcOrtho(FT->Jacobian(), nor);
      FT->Transform(ip, x);
      const double wl = ip.weight * nor.Norml2() * x(0);   // r ds
      const mfem::IntegrationPoint& eip = elem == e1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
      sp_.eval_at(elem, eip, pe);
      const double tv = tau(elem, pe);
      if (tv == 0.0) continue;
      for (size_t i = 0; i < d.size(); ++i) {
        const int a = idx_[d[i]];
        if (a >= 0) F[a] += wl * pe.N[i] * tv;
      }
      mom.force_N += 2 * pi * wl * tv;
      mom.torque_N_m += 2 * pi * wl * x(0) * tv;
      mom.zmin = std::min(mom.zmin, x(1));
      mom.zmax = std::max(mom.zmax, x(1));
    }
  }
  return mom;
}

json Torsion::resolution(double f_Hz, const std::function<std::string(int)>& region_of) const {
  AXF_REQUIRE(f_Hz > 0, "internal: torsion resolution needs a positive frequency");
  mfem::Mesh& m = sp_.mesh;
  std::map<std::string, std::vector<double>> per;
  mfem::Array<int> vv;
  for (int e = 0; e < m.GetNE(); ++e) {
    const TorsionRole& ro = roles_[e];
    if (!ro.solid) continue;
    m.GetElementVertices(e, vv);
    double h = 0.0;
    for (int i = 0; i < vv.Size(); ++i)
      for (int j = i + 1; j < vv.Size(); ++j) {
        const double* a = m.GetVertex(vv[i]);
        const double* b = m.GetVertex(vv[j]);
        h = std::max(h, std::hypot(a[0] - b[0], a[1] - b[1]));
      }
    const double lam = std::sqrt(ro.mu / ro.rho) / f_Hz;
    per[region_of(e)].push_back(lam / h);
  }
  std::vector<double> all;
  for (auto& [name, v] : per) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  const double all_p01 = all.empty() ? 0.0 : all[static_cast<size_t>(0.01 * static_cast<double>(all.size() - 1))];
  json regions = json::object();
  double worst = std::numeric_limits<double>::infinity();
  std::string worst_region;
  for (auto& [name, v] : per) {
    std::sort(v.begin(), v.end());
    const double mn = v.front();
    const double p01 = v[static_cast<size_t>(0.01 * static_cast<double>(v.size() - 1))];
    const double med = v[v.size() / 2];
    regions[name] = json{{"min", mn}, {"p01", p01}, {"median", med}, {"elements", v.size()}};
    if (mn < worst) {
      worst = mn;
      worst_region = name;
    }
  }
  return json{{"f_Hz", f_Hz}, {"order", sp_.order}, {"elements_per_wavelength_min", worst},
              {"elements_per_wavelength_p01", all_p01},
              {"worst_region", worst_region}, {"per_region", regions},
              {"definition", "c_T/(f h), h = longest vertex-to-vertex edge of the element"}};
}

namespace {

double read_number_file(const std::string& path) {
  std::ifstream f(path);
  std::string v;
  if (!(f >> v) || v == "max") return 0.0;
  const double x = std::atof(v.c_str());
  return x > 0 && x < 1e18 ? x : 0.0;
}

// A field of a "key value" file (memory.stat), 0 when absent.
double read_stat_field(const std::string& path, const std::string& field) {
  std::ifstream f(path);
  std::string k;
  double v = 0.0;
  while (f >> k >> v)
    if (k == field) return v;
  return 0.0;
}

double read_rss_bytes(const std::string& proc) {
  std::ifstream st(proc + "/self/status");
  std::string line;
  while (std::getline(st, line))
    if (line.rfind("VmRSS:", 0) == 0) return 1024.0 * std::atof(line.c_str() + 6);
  return 0.0;
}

// Restores the OpenMP thread count on scope exit, also when an exception leaves the scope.
struct OmpThreadsGuard {
  int saved;
  explicit OmpThreadsGuard(int n) : saved(omp_get_max_threads()) { omp_set_num_threads(n); }
  ~OmpThreadsGuard() { omp_set_num_threads(saved); }
  OmpThreadsGuard(const OmpThreadsGuard&) = delete;
  OmpThreadsGuard& operator=(const OmpThreadsGuard&) = delete;
};

}  // namespace

json torsion_available_memory(const json& opts) {
  const std::string proc = opts.value("proc", std::string("/proc"));
  const std::string cgroot = opts.value("cgroup", std::string("/sys/fs/cgroup"));
  const bool fake_env = opts.contains("env");
  auto env = [&](const char* name) -> std::string {
    if (fake_env) {
      const json& e = opts.at("env");
      return e.contains(name) ? e.at(name).get<std::string>() : std::string();
    }
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
  };
  const double rss = read_rss_bytes(proc);
  json src = json::object();
  // ---- cgroup: the smallest limit over the process's cgroup and every ancestor, "/" included (in a container
  // namespace the container's own cgroup is "/"), each against that cgroup's own usage less reclaimable file
  // cache (so other processes of the same job count)
  bool cg_found = false;
  double cg_avail = std::numeric_limits<double>::infinity();
  {
    std::ifstream cg(proc + "/self/cgroup");
    std::string line;
    while (std::getline(cg, line)) {
      const size_t c1 = line.find(':'), c2 = line.find(':', c1 + 1);
      if (c1 == std::string::npos || c2 == std::string::npos) continue;
      const std::string ctl = line.substr(c1 + 1, c2 - c1 - 1);
      const bool v2 = ctl.empty();
      const bool v1 = ctl.find("memory") != std::string::npos;
      if (!v2 && !v1) continue;
      std::string path = line.substr(c2 + 1);
      if (path.empty()) path = "/";
      for (;;) {
        const std::string dir = (v2 ? cgroot : cgroot + "/memory") + (path == "/" ? std::string() : path);
        const double limit = read_number_file(dir + (v2 ? "/memory.max" : "/memory.limit_in_bytes"));
        if (limit > 0) {
          double used = read_number_file(dir + (v2 ? "/memory.current" : "/memory.usage_in_bytes"));
          if (used > 0)
            used = std::max(0.0, used - read_stat_field(dir + "/memory.stat",
                                                        v2 ? "inactive_file" : "total_inactive_file"));
          else
            used = rss;
          const double avail = std::max(0.0, limit - used);
          if (avail < cg_avail) {
            cg_avail = avail;
            src["cgroup_limit_bytes"] = limit;
            src["cgroup_used_bytes"] = used;
            src["cgroup_path"] = path;
            src["cgroup_version"] = v2 ? 2 : 1;
          }
          cg_found = true;
        }
        if (path == "/") break;
        const size_t k = path.find_last_of('/');
        path = k == 0 ? std::string("/") : path.substr(0, k);
      }
    }
  }
  // ---- SLURM: used only when no cgroup limit was found (the cgroup is what the kernel enforces).  The CPU
  // count for SLURM_MEM_PER_CPU is SLURM_CPUS_ON_NODE (set for every job), else SLURM_CPUS_PER_TASK, else 1.
  bool sl_found = false;
  double sl_avail = std::numeric_limits<double>::infinity();
  const std::string mpn = env("SLURM_MEM_PER_NODE"), mpc = env("SLURM_MEM_PER_CPU");
  if (!mpn.empty()) {
    const double lim = std::atof(mpn.c_str()) * 1048576.0;
    src["SLURM_MEM_PER_NODE_bytes"] = lim;
    if (lim > 0) {
      sl_found = true;
      sl_avail = std::min(sl_avail, std::max(0.0, lim - rss));
    }
  }
  if (!mpc.empty()) {
    std::string cpus = env("SLURM_CPUS_ON_NODE");
    std::string from = "SLURM_CPUS_ON_NODE";
    if (cpus.empty()) {
      cpus = env("SLURM_CPUS_PER_TASK");
      from = "SLURM_CPUS_PER_TASK";
    }
    const double n = cpus.empty() ? 1.0 : std::max(1.0, std::atof(cpus.c_str()));
    if (cpus.empty()) from = "none (1 CPU)";
    const double lim = std::atof(mpc.c_str()) * 1048576.0 * n;
    src["SLURM_MEM_PER_CPU_x_CPUS_bytes"] = lim;
    src["SLURM_cpus_from"] = from;
    if (lim > 0) {
      sl_found = true;
      sl_avail = std::min(sl_avail, std::max(0.0, lim - rss));
    }
  }
  // ---- the node's MemAvailable bounds everything
  double node = -1.0;
  {
    std::ifstream mi(proc + "/meminfo");
    std::string key, unit;
    double kb = 0.0;
    while (mi >> key >> kb >> unit)
      if (key == "MemAvailable:") {
        node = kb * 1024.0;
        break;
      }
  }
  if (node >= 0) src["MemAvailable_bytes"] = node;
  // "found" is tracked separately from the value, so a source that leaves 0 is kept
  double avail = std::numeric_limits<double>::infinity();
  std::string used_source;
  if (cg_found) {
    avail = cg_avail;
    used_source = "cgroup";
    if (sl_found) src["SLURM_ignored"] = "a cgroup limit was found; it is what the kernel enforces";
  } else if (sl_found) {
    avail = sl_avail;
    used_source = "SLURM";
  }
  if (node >= 0 && node < avail) {
    avail = node;
    used_source = used_source.empty() ? "MemAvailable" : used_source + "+MemAvailable";
  }
  const bool found = std::isfinite(avail);
  return json{{"available_bytes", found ? avail : 0.0}, {"found", found},
              {"source", found ? used_source : std::string("none (no /proc; auto runs serially)")},
              {"rss_bytes", rss}, {"sources", src}};
}

json torsion_auto_choice(const json& in) {
  const std::string stage = in.at("stage").get<std::string>();
  const int threads = in.value("threads", 1);
  const int remaining = in.value("remaining", 0);
  const double avail = in.value("available_bytes", 0.0);
  const bool have_s = in.value("strumpack_available", false);
  if (stage == "probe") {
    // whether to probe Eigen SparseLU after the STRUMPACK probe
    if (!have_s) return json{{"probe_eigen", true}, {"reason", "no STRUMPACK: Eigen is the only backend"}};
    const double fill = in.value("strumpack_fill_per_dof", 0.0);
    const double nfree = in.value("free_dofs", 1.0);
    const double nnz = in.value("matrix_nonzeros", 0.0);
    // Eigen's fill was 1.7x STRUMPACK's on a thin wall and 2.9-3.0x on compact sections: predict
    // one Eigen solve's memory with 3x, the 20 B per nonzero and the 1.5 margin of the measured estimate
    const double predicted = 1.5 * (20.0 * 3.0 * fill * nfree + 16.0 * nnz);
    if (remaining < 8)
      return json{{"probe_eigen", false}, {"reason", "fewer than 8 frequencies left: a probe would not pay off"},
                  {"predicted_eigen_memory_bytes", predicted}};
    if (fill > 90.0)
      return json{{"probe_eigen", false},
                  {"reason", "STRUMPACK fill above 90 per DOF marks a compact section, where Eigen is slower"},
                  {"predicted_eigen_memory_bytes", predicted}};
    if (predicted > 0.8 * avail)
      return json{{"probe_eigen", false}, {"reason", "one Eigen solve would not fit in the available memory"},
                  {"predicted_eigen_memory_bytes", predicted}};
    return json{{"probe_eigen", true}, {"reason", "thin section and memory for an Eigen solve"},
                {"predicted_eigen_memory_bytes", predicted}};
  }
  AXF_REQUIRE(stage == "choose", "internal: torsion_auto_choice stage must be probe or choose");
  const bool probed_e = in.value("eigen_probed", false);
  const double t_e = in.value("eigen_s", 0.0), mem_e = in.value("eigen_memory_bytes", 0.0);
  const double t_s = in.value("strumpack_factor_s", 0.0);
  int p_mem = mem_e > 0 ? static_cast<int>(0.8 * avail / mem_e) : 0;
  const int p_e = probed_e ? std::max(0, std::min({remaining, std::max(1, threads / 2), p_mem})) : 0;
  // concurrent Eigen solves took 0.99-1.09x the probe's time: 1.1 when p > 1, none for one solve
  const double c = p_e > 1 ? 1.1 : 1.0;
  const double thr_e = (probed_e && p_e >= 1 && t_e > 0) ? p_e / (c * t_e) : 0.0;
  const double thr_s = (have_s && t_s > 0) ? 1.0 / t_s : 0.0;
  json out{{"memory_limited_p", p_mem}, {"predicted_throughput_eigen_per_s", thr_e},
           {"predicted_throughput_strumpack_per_s", thr_s}};
  if (have_s && !(thr_e > thr_s)) {
    out["backend"] = "strumpack";
    out["p"] = 1;
  } else if (probed_e) {
    out["backend"] = "eigen";
    out["p"] = std::max(1, p_e);
    if (p_e < 1) out["note"] = "memory for less than one Eigen solve was estimated; running one at a time";
  } else {
    out["backend"] = have_s ? "strumpack" : "eigen";
    out["p"] = 1;
  }
  return out;
}

std::vector<TorsionResponse> Torsion::harmonic_F(const std::vector<double>& Fv, const std::vector<double>& f_Hz,
                                                 const TorsionHarmonicOptions& opt, json* timing) const {
  using clock = std::chrono::steady_clock;
  AXF_REQUIRE(static_cast<int>(Fv.size()) == nfree_, "internal: torsion load vector has the wrong size");
  const auto t_setup = clock::now();
  CVec F(nfree_);
  for (int i = 0; i < nfree_; ++i) F[i] = Fv[i];
  const std::complex<double> one(1.0, 0.0), ii(0.0, 1.0);
  // K and M on one (union) pattern, so that the operator at each frequency is a combination of their
  // value arrays and one analysis per solver serves every frequency.
  SpMat Ku = impl_->K + 0.0 * impl_->M;
  SpMat Mu = impl_->M + 0.0 * impl_->K;
  Ku.makeCompressed();
  Mu.makeCompressed();
  AXF_REQUIRE(Ku.nonZeros() == Mu.nonZeros() &&
                  std::equal(Ku.innerIndexPtr(), Ku.innerIndexPtr() + Ku.nonZeros(), Mu.innerIndexPtr()),
              "internal: torsion K and M do not share a pattern");
  const CSpMat A0 = Ku.cast<std::complex<double>>();
  const long nnz = Ku.nonZeros();
  const int nf = static_cast<int>(f_Hz.size());
  std::vector<TorsionResponse> out(nf);
  std::vector<double> secs(nf, 0.0);
  const int threads_total = omp_get_max_threads();
  const double setup_s = std::chrono::duration<double>(clock::now() - t_setup).count();

  // the operator at frequency i into A (same pattern as A0)
  auto fill_operator = [&](int i, CSpMat& A) {
    const double w = 2 * pi * f_Hz[i];
    const std::complex<double> ck = one - opt.eta * ii;
    std::complex<double>* av = A.valuePtr();
    const double* kv = Ku.valuePtr();
    const double* mv = Mu.valuePtr();
    for (long j = 0; j < nnz; ++j) av[j] = ck * kv[j] - (w * w) * mv[j];
  };
  // factorize and solve frequency i with an analysed solver (fills out[i])
  auto factor_solve = [&](int i, ComplexSymSolver& solver, CSpMat& A) {
    fill_operator(i, A);
    solver.factorize(A);
    TorsionResponse r;
    r.f_Hz = f_Hz[i];
    const double w = 2 * pi * f_Hz[i];
    if (solver.info() != Eigen::Success) {
      r.lu_ok = 0;
    } else {
      CVec u = solver.solve(F);
      const std::complex<double> qq = 2.0 * pi * (F.transpose() * u).value();   // q = 2 pi F^T u, unconjugated
      r.q_re = qq.real();
      r.q_im = qq.imag();
      const std::complex<double> Y = -ii * w * qq;
      r.Y_re = Y.real();
      r.Y_im = Y.imag();
    }
    out[i] = r;
  };
  auto solve_one = [&](int i, ComplexSymSolver& solver, CSpMat& A, bool& analysed) {
    const auto ts = clock::now();
    if (!analysed) {
      fill_operator(i, A);
      solver.analyzePattern(A);
      analysed = true;
    }
    factor_solve(i, solver, A);
    secs[i] = std::chrono::duration<double>(clock::now() - ts).count();
    if (opt.verbose)
      log(1, strcat("torsion harmonic f = ", f_Hz[i], " Hz: Re Y = ", out[i].Y_re, " (", secs[i], " s)"));
  };

  std::set<std::string> backends;
  long factor_nnz = 0;
  int symbolic = 0;
  std::string error;
  auto record = [&](const ComplexSymSolver& s) {
    backends.insert(s.backend());
    factor_nnz = std::max(factor_nnz, s.factor_nonzeros());
    symbolic += s.symbolic_analyses();
  };
  // run the frequencies `idx` with p concurrent solvers of the given backend (-1 by size, 0 Eigen, 1 STRUMPACK);
  // with p = 1 an already analysed solver (the auto probe's) may be handed over.  Concurrent STRUMPACK solvers are
  // safe through the concurrent guard in sym_solver.cpp: each frequency thread pushes STRUMPACK's timers to its own
  // thread number's list, and the orderings (METIS) run one at a time.  The serialized guard (other STRUMPACK
  // versions) refuses calls from threads other than 0, so callers then pass p = 1.
  auto run = [&](const std::vector<int>& idx, int backend, int p, ComplexSymSolver* given = nullptr) {
    const int m = static_cast<int>(idx.size());
    if (m == 0) return;
    p = std::max(1, std::min(p, m));
    if (p == 1 && given) {
      CSpMat A = A0;
      bool analysed = true;
      for (int k = 0; k < m; ++k) solve_one(idx[k], *given, A, analysed);
      return;
    }
    const int inner = std::max(1, threads_total / p);
#ifdef AXF_HAVE_OPENMP
    const int old_levels = omp_get_max_active_levels();
    if (p > 1) omp_set_max_active_levels(2);
#endif
#pragma omp parallel num_threads(p) if (p > 1)
    {
      if (p > 1) omp_set_num_threads(inner);
      ComplexSymSolver solver(backend);
      CSpMat A = A0;
      bool analysed = false, failed = false;
#pragma omp for schedule(dynamic, 1)
      for (int k = 0; k < m; ++k) {
        if (failed) continue;
        try {
          solve_one(idx[k], solver, A, analysed);
        } catch (const std::exception& ex) {
          failed = true;
#pragma omp critical(axf_torsion_err)
          if (error.empty()) error = ex.what();
        }
      }
#pragma omp critical(axf_torsion_stats)
      if (analysed) record(solver);
    }
#ifdef AXF_HAVE_OPENMP
    omp_set_max_active_levels(old_levels);
#endif
    AXF_REQUIRE(error.empty(), "torsion harmonic solve failed: " + error);
  };

  const auto t0 = clock::now();
  json choice;
  int p_used = std::max(1, opt.parallel_frequencies);
  bool serialized = false;
  if (opt.parallel_frequencies >= 1) {
    std::vector<int> all(nf);
    for (int i = 0; i < nf; ++i) all[i] = i;
    // A STRUMPACK version that the concurrent guard was not validated on takes calls from OpenMP thread 0 only
    // (sym_solver.cpp): frequencies that would factorize with STRUMPACK run one after another.
    const bool uses_strumpack = opt.backend == 1 || (opt.backend == -1 && have_strumpack() &&
                                                     A0.rows() >= SymSolver::threshold());
    if (p_used > 1 && uses_strumpack && strumpack_calls_serialized()) {
      p_used = 1;
      serialized = true;
    }
    run(all, opt.backend, p_used);
  } else {
    // ---- auto: the memory limit first; then one STRUMPACK solve (all threads; its
    // factorization timed apart from the analysis, its solver kept for the rest); an Eigen SparseLU probe only
    // where it can pay off and fits; then the faster of p concurrent Eigen solves or serial STRUMPACK.
    const json mem = torsion_available_memory(json::object());
    const double avail = mem["available_bytes"].get<double>();
    const bool have_s = have_strumpack() && opt.backend != 0;
    const bool allow_e = opt.backend != 1;
    int next = 0;
    std::unique_ptr<ComplexSymSolver> sprobe;
    double t_s = 0.0, t_e = 0.0, mem_e = 0.0;
    long fill_s = 0, fill_e = 0;
    std::string probe_reason;
    if (have_s && nf >= 1) {
      sprobe = std::make_unique<ComplexSymSolver>(1);
      CSpMat A = A0;
      fill_operator(next, A);
      sprobe->analyzePattern(A);
      const auto ts = clock::now();
      factor_solve(next, *sprobe, A);
      t_s = std::chrono::duration<double>(clock::now() - ts).count();
      secs[next] = t_s;
      fill_s = sprobe->factor_nonzeros();
      ++next;
    }
    json probe = torsion_auto_choice(json{{"stage", "probe"}, {"strumpack_available", have_s},
                                          {"strumpack_fill_per_dof", have_s ? double(fill_s) / nfree_ : 0.0},
                                          {"free_dofs", nfree_}, {"matrix_nonzeros", double(nnz)},
                                          {"remaining", nf - next}, {"available_bytes", avail}});
    bool probed_e = false;
    if (allow_e && next < nf && probe["probe_eigen"].get<bool>()) {
      const double rss0 = read_rss_bytes("/proc");
      OmpThreadsGuard guard(std::min(2, threads_total));
      ComplexSymSolver sol(0);
      CSpMat A = A0;
      bool an = false;
      solve_one(next, sol, A, an);
      t_e = secs[next];
      fill_e = sol.factor_nonzeros();
      record(sol);
      // resident growth with the solver alive, and a floor from the factor size (16 B values + 4 B index),
      // times 1.5: the factorization's transient workspace is freed before it can be read here (the estimate
      // came out 1.3-1.5x above the measured peak per concurrent solve)
      mem_e = 1.5 * std::max(read_rss_bytes("/proc") - rss0, 20.0 * fill_e + 16.0 * nnz);
      probed_e = true;
      ++next;
    }
    std::vector<int> rest;
    for (int i = next; i < nf; ++i) rest.push_back(i);
    json dec = torsion_auto_choice(json{{"stage", "choose"}, {"threads", threads_total},
                                        {"remaining", static_cast<int>(rest.size())}, {"available_bytes", avail},
                                        {"strumpack_available", have_s}, {"strumpack_factor_s", t_s},
                                        {"eigen_probed", probed_e}, {"eigen_s", t_e}, {"eigen_memory_bytes", mem_e}});
    const std::string chosen = dec["backend"].get<std::string>();
    p_used = dec["p"].get<int>();
    if (chosen == "strumpack" && sprobe) {
      run(rest, 1, 1, sprobe.get());
    } else {
      run(rest, chosen == "strumpack" ? 1 : (allow_e ? 0 : opt.backend), p_used);
    }
    if (sprobe) record(*sprobe);
    choice = json{{"mode", "auto"}, {"threads", threads_total}, {"chosen_backend", chosen}, {"chosen_p", p_used},
                  {"probe_strumpack_factor_s", have_s ? json(t_s) : json(nullptr)},
                  {"probe_strumpack_factor_nnz_per_dof", have_s ? json(double(fill_s) / nfree_) : json(nullptr)},
                  {"eigen_probe", probe}, {"probe_eigen_s", probed_e ? json(t_e) : json(nullptr)},
                  {"probe_eigen_factor_nnz_per_dof", probed_e ? json(double(fill_e) / nfree_) : json(nullptr)},
                  {"eigen_memory_per_solve_bytes", mem_e}, {"memory", mem}, {"decision", dec}};
  }
  if (timing) {
    const double wall = std::chrono::duration<double>(clock::now() - t0).count();
    double sum = 0.0;
    for (double x : secs) sum += x;
    std::string be;
    for (const auto& b : backends) be += (be.empty() ? "" : "+") + b;
    *timing = json{{"backend", be}, {"free_dofs", nfree_}, {"matrix_nonzeros", nnz},
                   {"factor_nonzeros", factor_nnz}, {"symbolic_analyses", symbolic},
                   {"parallel_frequencies", p_used},
                   {"threads_per_frequency", p_used > 1 ? std::max(1, threads_total / p_used) : threads_total},
                   {"setup_s", setup_s}, {"wall_s", wall}, {"seconds_per_frequency", secs},
                   {"mean_seconds_per_frequency", nf ? sum / nf : 0.0},
                   {"effective_seconds_per_frequency", nf ? wall / nf : 0.0}};
    if (!choice.is_null()) (*timing)["parallel_choice"] = choice;
    if (serialized) {
      (*timing)["parallel_frequencies_requested"] = opt.parallel_frequencies;
      (*timing)["strumpack_guard"] = strumpack_guard_mode();
    }
  }
  return out;
}

}  // namespace axf::axi
