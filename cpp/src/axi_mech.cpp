#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#include "axionfem/core.hpp"
#include "axionfem/mech.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using units::pi;

// Connected components through shared degrees of freedom (declared in mech.hpp).
std::vector<int> solid_components(const Space& sp, const std::function<bool(int)>& solid, int& ncomp) {
  const int ne = sp.mesh.GetNE(), nd = sp.ndof();
  std::vector<int> par(ne);
  std::iota(par.begin(), par.end(), 0);
  std::function<int(int)> find = [&](int a) { return par[a] == a ? a : par[a] = find(par[a]); };
  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) par[std::max(a, b)] = std::min(a, b);
  };
  std::vector<int> first(nd, -1);
  for (int e = 0; e < ne; ++e) {
    if (!solid(e)) continue;
    for (int d : sp.edofs(e)) {
      if (first[d] < 0) first[d] = e;
      else unite(first[d], e);
    }
  }
  std::vector<int> comp(ne, -1);
  std::map<int, int> id;
  ncomp = 0;
  for (int e = 0; e < ne; ++e) {
    if (!solid(e)) continue;
    const int rt = find(e);
    auto it = id.find(rt);
    if (it == id.end()) it = id.emplace(rt, ncomp++).first;
    comp[e] = it->second;
  }
  return comp;
}

namespace {

// Selection matrix of the free degrees of freedom: u = P y.
SpMat selection(const std::vector<int>& idx, int n, int nfree) {
  std::vector<Trip> t;
  t.reserve(nfree);
  for (int a = 0; a < n; ++a)
    if (idx[a] >= 0) t.emplace_back(a, idx[a], 1.0);
  SpMat P(n, nfree);
  P.setFromTriplets(t.begin(), t.end());
  return P;
}

}  // namespace

// ---------------------------------------------------------------- k(T)
double KTable::at(double t) const {
  if (T.empty()) return 0.0;
  if (T.size() == 1 || t <= T.front()) return k.front();
  if (t >= T.back()) return k.back();
  const size_t i = static_cast<size_t>(std::upper_bound(T.begin(), T.end(), t) - T.begin());
  const double w = (t - T[i - 1]) / (T[i] - T[i - 1]);
  return k[i - 1] + w * (k[i] - k[i - 1]);
}

// ---------------------------------------------------------------- elasticity
Elasticity::Elasticity(const Space& sp, std::vector<ElasticRole> roles, std::vector<SupportGroup> supports)
    : sp_(sp), roles_(std::move(roles)), sup_(std::move(supports)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: elastic role table size mismatch");
}

void Elasticity::set_pressure(std::vector<PressureFace> faces, PressureFn p) {
  pfaces_ = std::move(faces);
  pfn_ = std::move(p);
}

double Elasticity::von_mises(double srr, double szz, double stt, double srz) {
  const double a = srr - szz, b = szz - stt, c = stt - srr;
  return std::sqrt(0.5 * (a * a + b * b + c * c) + 3.0 * srz * srz);
}

void Elasticity::displacement(const ElasticSolution& s, int e, const PointEval& pe, double& ur, double& uz) const {
  ur = uz = 0;
  if (!roles_[e].solid) return;
  const auto& d = sp_.edofs(e);
  for (size_t i = 0; i < d.size(); ++i) {
    ur += s.ur[d[i]] * pe.N[i];
    uz += s.uz[d[i]] * pe.N[i];
  }
}

void Elasticity::stress(const ElasticSolution& s, int e, const PointEval& pe, double& srr, double& szz, double& stt,
                        double& srz) const {
  srr = szz = stt = srz = 0;
  if (!roles_[e].solid) return;
  const auto& d = sp_.edofs(e);
  double err = 0, ezz = 0, ett = 0, grz = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    const double dr = pe.dN[2 * i], dz = pe.dN[2 * i + 1];
    const double hoop = pe.r > sp_.rtol ? pe.N[i] / pe.r : dr;  // u_r/r -> du_r/dr on the axis
    err += s.ur[d[i]] * dr;
    ezz += s.uz[d[i]] * dz;
    ett += s.ur[d[i]] * hoop;
    grz += s.ur[d[i]] * dz + s.uz[d[i]] * dr;
  }
  const ElasticRole& ro = roles_[e];
  const double tr = err + ezz + ett, s0 = (3.0 * ro.lambda + 2.0 * ro.mu) * ro.eps_th;
  srr = ro.lambda * tr + 2.0 * ro.mu * err - s0;
  szz = ro.lambda * tr + 2.0 * ro.mu * ezz - s0;
  stt = ro.lambda * tr + 2.0 * ro.mu * ett - s0;
  srz = ro.mu * grz;
}

ElasticSolution Elasticity::solve(int verbose) {
  Timer timer;
  mfem::Mesh& m = sp_.mesh;
  const int nd = sp_.ndof(), ne = m.GetNE();
  const int N = 2 * nd;  // 2 a   -> u_r at DOF a ; 2 a + 1 -> u_z at DOF a
  const int ng = static_cast<int>(sup_.size());

  // ------------------------------------------------------------ connectivity and constraints
  int ncomp = 0;
  std::vector<int> comp = solid_components(sp_, [&](int e) { return roles_[e].solid; }, ncomp);
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e)
    if (roles_[e].solid)
      for (int d : sp_.edofs(e)) used[d] = 1;

  std::vector<char> fix_r(nd, 0), fix_z(nd, 0);
  std::vector<int> grp_r(nd, -1), grp_z(nd, -1);  // -1: not fixed by a support, -2: rigid-mode pin
  for (int e = 0; e < ne; ++e) {
    const ElasticRole& ro = roles_[e];
    if (!ro.solid || ro.support < 0) continue;
    const SupportGroup& g = sup_[ro.support];
    for (int d : sp_.edofs(e)) {
      if (g.fix_r && !fix_r[d]) { fix_r[d] = 1; grp_r[d] = ro.support; }
      if (g.fix_z && !fix_z[d]) { fix_z[d] = 1; grp_z[d] = ro.support; }
    }
  }
  int axis_dofs = 0;
  for (int i = 0; i < nd; ++i)
    if (used[i] && sp_.r[i] == 0.0 && !fix_r[i]) { fix_r[i] = 1; ++axis_dofs; }  // u_r = 0 on the axis

  // One axial rigid translation per component without an axial support: pin the lowest DOF index.
  std::vector<char> comp_has_z(ncomp, 0);
  std::vector<int> comp_pin(ncomp, -1);
  for (int e = 0; e < ne; ++e) {
    if (comp[e] < 0) continue;
    for (int d : sp_.edofs(e)) {
      if (fix_z[d]) comp_has_z[comp[e]] = 1;
      if (comp_pin[comp[e]] < 0 || d < comp_pin[comp[e]]) comp_pin[comp[e]] = d;
    }
  }
  int npin = 0;
  for (int c = 0; c < ncomp; ++c) {
    if (comp_has_z[c] || comp_pin[c] < 0) continue;
    fix_z[comp_pin[c]] = 1;
    grp_z[comp_pin[c]] = -2;
    ++npin;
  }

  std::vector<int> idx(N, -1);
  int nfree = 0;
  for (int i = 0; i < nd; ++i) {
    if (!used[i]) continue;
    if (!fix_r[i]) idx[2 * i] = nfree++;
    if (!fix_z[i]) idx[2 * i + 1] = nfree++;
  }
  AXF_REQUIRE(nfree > 0, "elasticity: no free degrees of freedom",
              "Every part of the structure is fixed by a support; remove a support or check mechanics.supports.");

  // ------------------------------------------------------------ assembly
  std::vector<Trip> trip;
  Vec F = Vec::Zero(N);
  ElasticSolution s;
  s.free_component_Fz_N.assign(ncomp, 0.0);
  std::vector<double> comp_Fz(ncomp, 0.0);
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  Eigen::MatrixXd B, Ke;
  Eigen::VectorXd fe;
  for (int e = 0; e < ne; ++e) {
    const ElasticRole& ro = roles_[e];
    if (!ro.solid) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    const double lam = ro.lambda, mu = ro.mu, s0 = (3.0 * lam + 2.0 * mu) * ro.eps_th;
    Eigen::Matrix4d D = Eigen::Matrix4d::Zero();
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) D(a, b) = lam;
    for (int a = 0; a < 3; ++a) D(a, a) += 2.0 * mu;
    D(3, 3) = mu;
    Ke.setZero(2 * n, 2 * n);
    fe.setZero(2 * n);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double wr = pe.w * pe.r;
      B.setZero(4, 2 * n);
      for (int i = 0; i < n; ++i) {
        const double dr = pe.dN[2 * i], dz = pe.dN[2 * i + 1];
        const double hoop = pe.r > sp_.rtol ? pe.N[i] / pe.r : dr;
        B(0, 2 * i) = dr;
        B(2, 2 * i) = hoop;
        B(3, 2 * i) = dz;
        B(1, 2 * i + 1) = dz;
        B(3, 2 * i + 1) = dr;
      }
      Ke += wr * (B.transpose() * D * B);
      if (s0 != 0.0) {
        Eigen::Vector4d sth(s0, s0, s0, 0.0);
        fe += wr * (B.transpose() * sth);
      }
      if (ro.body_force && body_) {
        double fr = 0, fz = 0;
        body_(e, pe, fr, fz);
        for (int i = 0; i < n; ++i) {
          fe(2 * i) += wr * pe.N[i] * fr;
          fe(2 * i + 1) += wr * pe.N[i] * fz;
        }
        s.applied_Fr_per_rad += wr * fr;
        s.applied_Fz_N += 2 * pi * wr * fz;
        s.load_scale_N += 2 * pi * wr * (std::fabs(fr) + std::fabs(fz));
        s.axial_load_scale_N += 2 * pi * wr * std::fabs(fz);
        comp_Fz[comp[e]] += 2 * pi * wr * fz;
      }
    }
    for (int i = 0; i < n; ++i) {
      F(2 * d[i]) += fe(2 * i);
      F(2 * d[i] + 1) += fe(2 * i + 1);
      for (int j = 0; j < n; ++j) {
        trip.emplace_back(2 * d[i], 2 * d[j], Ke(2 * i, 2 * j));
        trip.emplace_back(2 * d[i], 2 * d[j] + 1, Ke(2 * i, 2 * j + 1));
        trip.emplace_back(2 * d[i] + 1, 2 * d[j], Ke(2 * i + 1, 2 * j));
        trip.emplace_back(2 * d[i] + 1, 2 * d[j] + 1, Ke(2 * i + 1, 2 * j + 1));
      }
    }
  }

  // ------------------------------------------------------------ pressure loads (t = -p n)
  if (!pfaces_.empty()) {
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp_.order + 6);
    for (const auto& pf : pfaces_) {
      int e1 = -1, e2 = -1;
      m.GetFaceElements(pf.face, &e1, &e2);
      AXF_REQUIRE(pf.solid_elem == e1 || pf.solid_elem == e2,
                  "internal: pressure face does not touch the element it loads");
      const int which = (pf.solid_elem == e1) ? 1 : 2;
      mfem::FaceElementTransformations* FT = m.GetFaceElementTransformations(pf.face);
      const auto& d = sp_.edofs(pf.solid_elem);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        const double p = pfn_ ? pfn_(pf, *FT) : 0.0;
        mfem::Vector nor(2), x(2);
        mfem::CalcOrtho(FT->Jacobian(), nor);  // element 1 -> element 2, |nor| = dl/dxi
        FT->Transform(ip, x);
        const double len = nor.Norml2();
        double nr = nor(0) / len, nz = nor(1) / len;
        if (which == 2) { nr = -nr; nz = -nz; }  // outward normal of the loaded solid
        const double tr = -p * nr, tz = -p * nz;
        const double wl = ip.weight * len * x(0);  // r dl
        const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
        sp_.eval_at(pf.solid_elem, eip, pe);
        for (size_t i = 0; i < d.size(); ++i) {
          F(2 * d[i]) += wl * pe.N[i] * tr;
          F(2 * d[i] + 1) += wl * pe.N[i] * tz;
        }
        s.applied_Fr_per_rad += wl * tr;
        s.applied_Fz_N += 2 * pi * wl * tz;
        s.load_scale_N += 2 * pi * wl * (std::fabs(tr) + std::fabs(tz));
        s.axial_load_scale_N += 2 * pi * wl * std::fabs(tz);
        comp_Fz[comp[pf.solid_elem]] += 2 * pi * wl * tz;
      }
    }
  }

  // ------------------------------------------------------------ solve
  SpMat K(N, N);
  K.setFromTriplets(trip.begin(), trip.end());
  SpMat P = selection(idx, N, nfree);
  SpMat Kr = P.transpose() * K * P;
  Vec Fr = P.transpose() * F;
  Eigen::SimplicialLDLT<SpMat> chol(Kr);
  AXF_REQUIRE(chol.info() == Eigen::Success, "the elasticity system could not be factorized", "",
              "Typical causes: a region with E = 0 or nu = 0.5, a zero-volume region, or a part of the structure "
              "free to move (add a support in mechanics.supports).");
  Vec y = chol.solve(Fr);
  Vec u = P * y;
  Vec res = K * u - F;
  const double rnorm = (Kr * y - Fr).norm() / std::max(Fr.norm(), 1e-300);

  // ------------------------------------------------------------ outputs
  s.ur.assign(nd, 0.0);
  s.uz.assign(nd, 0.0);
  for (int i = 0; i < nd; ++i) {
    s.ur[i] = u(2 * i);
    s.uz[i] = u(2 * i + 1);
  }
  // Shift each pinned (axially free) component so that its volume-weighted mean u_z vanishes: the
  // pinned solution is only determined up to a rigid axial translation.
  if (npin > 0) {
    std::vector<double> num(ncomp, 0.0), den(ncomp, 0.0);
    for (int e = 0; e < ne; ++e) {
      if (comp[e] < 0 || comp_has_z[comp[e]]) continue;
      const auto& d = sp_.edofs(e);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double uz = 0;
        for (size_t i = 0; i < d.size(); ++i) uz += s.uz[d[i]] * pe.N[i];
        num[comp[e]] += pe.w * pe.r * uz;
        den[comp[e]] += pe.w * pe.r;
      }
    }
    std::vector<double> shift(ncomp, 0.0);
    for (int c = 0; c < ncomp; ++c)
      if (!comp_has_z[c] && den[c] > 0) shift[c] = num[c] / den[c];
    std::vector<char> done(nd, 0);
    for (int e = 0; e < ne; ++e) {
      if (comp[e] < 0 || comp_has_z[comp[e]]) continue;
      for (int d : sp_.edofs(e))
        if (!done[d]) { s.uz[d] -= shift[comp[e]]; done[d] = 1; }
    }
  }

  s.reaction_Fz_N.assign(ng, 0.0);
  s.reaction_Fr_per_rad.assign(ng, 0.0);
  double total_Rz = 0;
  for (int i = 0; i < nd; ++i) {
    if (!used[i]) continue;
    if (fix_r[i] && grp_r[i] >= 0) s.reaction_Fr_per_rad[grp_r[i]] += res(2 * i);
    if (fix_z[i]) {
      total_Rz += 2 * pi * res(2 * i + 1);
      if (grp_z[i] >= 0) s.reaction_Fz_N[grp_z[i]] += 2 * pi * res(2 * i + 1);
      else if (grp_z[i] == -2) s.pin_Fz_N += 2 * pi * res(2 * i + 1);
    }
  }
  // The closure residual is normalised by the size of the terms that are summed (the L1 norm of the
  // nodal load vector), which stays meaningful for a self-balanced load such as a cooldown prestress.
  // The thermal prestress contributes exactly zero to the sum over the axial DOFs (sum_i N_i = 1),
  // so it does not enter applied_Fz_N, but it does set the size of the terms being summed; report
  // that separately (assembly_scale_N) so round-off can be told from a real imbalance.
  for (int a = 0; a < N; ++a) s.assembly_scale_N += 2 * pi * std::fabs(F(a));
  s.closure_Fz_N = std::fabs(s.applied_Fz_N + total_Rz);
  s.closure_Fz = s.closure_Fz_N / std::max({s.load_scale_N, std::fabs(total_Rz), 1e-30});
  s.closure_Fz_round_off = s.closure_Fz_N / std::max(s.assembly_scale_N, 1e-30);
  for (int c = 0; c < ncomp; ++c) s.free_component_Fz_N[c] = comp_has_z[c] ? 0.0 : comp_Fz[c];
  s.component_of_elem = comp;
  s.diagnostics = {{"dofs", 2 * nd},
                   {"free_dofs", nfree},
                   {"solid_elements", static_cast<int>(std::count_if(roles_.begin(), roles_.end(),
                                                                     [](const ElasticRole& r) { return r.solid; }))},
                   {"components", ncomp},
                   {"axis_constrained_dofs", axis_dofs},
                   {"rigid_axial_pins", npin},
                   {"pressure_faces", static_cast<int>(pfaces_.size())},
                   {"relative_residual", rnorm},
                   {"seconds", timer.seconds()}};
  if (verbose > 1) log(2, strcat("  elasticity: ", nfree, " free dofs, residual ", rnorm));
  return s;
}

// ---------------------------------------------------------------- conduction
Conduction::Conduction(const Space& sp, std::vector<ThermalRole> roles, std::map<int, double> fixed_curve_attr)
    : sp_(sp), roles_(std::move(roles)), fixed_curve_(std::move(fixed_curve_attr)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: thermal role table size mismatch");
}

double Conduction::value(const ThermalSolution& s, int e, const PointEval& pe) const {
  if (!roles_[e].solid) return 0.0;
  const auto& d = sp_.edofs(e);
  double t = 0;
  for (size_t i = 0; i < d.size(); ++i) t += s.T[d[i]] * pe.N[i];
  return t;
}

void Conduction::flux(const ThermalSolution& s, int e, const PointEval& pe, double& qr, double& qz) const {
  qr = qz = 0;
  if (!roles_[e].solid) return;
  const auto& d = sp_.edofs(e);
  double dr = 0, dz = 0, t = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    dr += s.T[d[i]] * pe.dN[2 * i];
    dz += s.T[d[i]] * pe.dN[2 * i + 1];
    t += s.T[d[i]] * pe.N[i];
  }
  const double k = roles_[e].kt ? roles_[e].kt->at(t) : roles_[e].k;
  qr = -k * dr;
  qz = -k * dz;
}

ThermalSolution Conduction::solve(int verbose, double tol, int max_picard) {
  Timer timer;
  const int nd = sp_.ndof(), ne = sp_.mesh.GetNE();
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e)
    if (roles_[e].solid)
      for (int d : sp_.edofs(e)) used[d] = 1;

  std::vector<char> fix(nd, 0);
  Vec Tfix = Vec::Zero(nd);
  int nfixed = 0;
  for (int e = 0; e < ne; ++e) {
    if (!roles_[e].solid || !roles_[e].fixed) continue;
    for (int d : sp_.edofs(e)) {
      if (!fix[d]) ++nfixed;
      fix[d] = 1;
      Tfix(d) = roles_[e].T_fix;
    }
  }
  if (!fixed_curve_.empty()) {
    mfem::Array<int> bd;
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      auto it = fixed_curve_.find(sp_.mesh.GetBdrAttribute(be));
      if (it == fixed_curve_.end()) continue;
      sp_.fes->GetBdrElementDofs(be, bd);
      for (int d : bd) {
        if (!used[d]) continue;
        if (!fix[d]) ++nfixed;
        fix[d] = 1;
        Tfix(d) = it->second;
      }
    }
  }
  AXF_REQUIRE(nfixed > 0, "the thermal problem has no prescribed temperature", "/thermal/fixed",
              "Add at least one {\"region\": NAME, \"T_K\": value} (or a curve) so the temperature level is set.");

  std::vector<int> idx(nd, -1);
  int nfree = 0;
  for (int i = 0; i < nd; ++i)
    if (used[i] && !fix[i]) idx[i] = nfree++;
  const bool nonlinear = std::any_of(roles_.begin(), roles_.end(),
                                     [](const ThermalRole& r) { return r.solid && r.kt && !r.kt->empty(); });

  // right-hand side (fixed across the Picard iteration) and the generated power
  const RefTable& tab = sp_.table(sp_.quad_rule());
  Vec F = Vec::Zero(nd);
  ThermalSolution s;
  PointEval pe;
  for (int e = 0; e < ne; ++e) {
    if (!roles_[e].solid || roles_[e].q == 0.0) continue;
    const auto& d = sp_.edofs(e);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double wr = pe.w * pe.r;
      for (size_t i = 0; i < d.size(); ++i) F(d[i]) += wr * pe.N[i] * roles_[e].q;
      s.generated_W += 2 * pi * wr * roles_[e].q;
    }
  }

  double Tm = 0;
  for (int i = 0; i < nd; ++i)
    if (fix[i]) Tm += Tfix(i) / nfixed;
  Vec T = Vec::Zero(nd);
  for (int i = 0; i < nd; ++i) T(i) = used[i] ? (fix[i] ? Tfix(i) : Tm) : 0.0;

  SpMat P = selection(idx, nd, nfree);
  double rnorm = 0;
  int it = 0;
  s.converged = !nonlinear;
  for (; it < (nonlinear ? max_picard : 1); ++it) {
    std::vector<Trip> trip;
    for (int e = 0; e < ne; ++e) {
      if (!roles_[e].solid) continue;
      const auto& d = sp_.edofs(e);
      const int n = static_cast<int>(d.size());
      Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(n, n);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double k = roles_[e].k;
        if (roles_[e].kt && !roles_[e].kt->empty()) {
          double t = 0;
          for (int i = 0; i < n; ++i) t += T(d[i]) * pe.N[i];
          k = roles_[e].kt->at(t);
        }
        const double wr = pe.w * pe.r * k;
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            Ke(i, j) += wr * (pe.dN[2 * i] * pe.dN[2 * j] + pe.dN[2 * i + 1] * pe.dN[2 * j + 1]);
      }
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) trip.emplace_back(d[i], d[j], Ke(i, j));
    }
    SpMat K(nd, nd);
    K.setFromTriplets(trip.begin(), trip.end());
    SpMat Kr = P.transpose() * K * P;
    Vec Td = Vec::Zero(nd);  // the prescribed part of T (zero on the free DOFs)
    for (int i = 0; i < nd; ++i)
      if (fix[i]) Td(i) = Tfix(i);
    Vec rhs = P.transpose() * (F - K * Td);
    Eigen::SimplicialLDLT<SpMat> chol(Kr);
    AXF_REQUIRE(chol.info() == Eigen::Success, "the conduction system could not be factorized", "",
                "Typical causes: a region with k = 0, a zero-volume region, or a part of the thermal domain with no "
                "prescribed temperature anywhere (add one in thermal.fixed).");
    Vec y = chol.solve(rhs);
    Vec Tnew = Td + P * y;
    rnorm = (Kr * y - rhs).norm() / std::max(rhs.norm(), 1e-300);
    double span = 0, change = 0;
    for (int i = 0; i < nd; ++i) {
      if (!used[i]) continue;
      span = std::max(span, std::fabs(Tnew(i)));
      change = std::max(change, std::fabs(Tnew(i) - T(i)));
    }
    s.picard_change = change / std::max(span, 1e-300);
    T = Tnew;
    if (!nonlinear) break;
    if (s.picard_change < tol) {
      s.converged = true;
      ++it;
      break;
    }
  }
  s.picard_iterations = it;
  if (verbose > 1) log(2, strcat("  conduction: ", nfree, " free dofs, picard ", it, " change ", s.picard_change));

  // final residual, heat balance and temperature range
  {
    std::vector<Trip> trip;
    for (int e = 0; e < ne; ++e) {
      if (!roles_[e].solid) continue;
      const auto& d = sp_.edofs(e);
      const int n = static_cast<int>(d.size());
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double k = roles_[e].k;
        if (roles_[e].kt && !roles_[e].kt->empty()) {
          double t = 0;
          for (int i = 0; i < n; ++i) t += T(d[i]) * pe.N[i];
          k = roles_[e].kt->at(t);
        }
        const double wr = pe.w * pe.r * k;
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            trip.emplace_back(d[i], d[j], wr * (pe.dN[2 * i] * pe.dN[2 * j] + pe.dN[2 * i + 1] * pe.dN[2 * j + 1]));
      }
    }
    SpMat K(nd, nd);
    K.setFromTriplets(trip.begin(), trip.end());
    Vec res = K * T - F;
    // Split the reactions by sign: the net sum cancels when heat flows in at one prescribed temperature
    // and out at another (a cold stage between two sinks), which would report outflow_W ~ 0 and a
    // meaningless balance_relative.  heat_removed_W is the load on the sinks that absorb heat.
    double sum = 0, neg = 0, pos = 0;
    for (int i = 0; i < nd; ++i) {
      if (!used[i] || !fix[i]) continue;
      sum += res(i);
      if (res(i) < 0) neg += res(i);
      else pos += res(i);
    }
    s.outflow_W = -2 * pi * sum;
    s.heat_removed_W = -2 * pi * neg;
    s.heat_added_W = 2 * pi * pos;
  }
  s.T.assign(nd, 0.0);
  s.T_min = 1e300;
  s.T_max = -1e300;
  for (int i = 0; i < nd; ++i) {
    s.T[i] = used[i] ? T(i) : 0.0;
    if (!used[i]) continue;
    s.T_min = std::min(s.T_min, T(i));
    s.T_max = std::max(s.T_max, T(i));
  }
  if (s.T_min > s.T_max) { s.T_min = s.T_max = 0; }
  s.balance_rel = std::fabs(s.generated_W - s.outflow_W) /
                  std::max({std::fabs(s.generated_W), s.heat_removed_W, s.heat_added_W, 1e-300});
  s.diagnostics = {{"dofs", nd},
                   {"free_dofs", nfree},
                   {"fixed_dofs", nfixed},
                   {"nonlinear", nonlinear},
                   {"picard_iterations", s.picard_iterations},
                   {"picard_change", s.picard_change},
                   {"converged", s.converged},
                   {"relative_residual", rnorm},
                   {"seconds", timer.seconds()}};
  return s;
}

}  // namespace axf::axi
