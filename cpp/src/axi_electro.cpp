#include "axionfem/electro.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>

#include "axionfem/core.hpp"
#include "axionfem/model.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using units::pi;

namespace {

// Legendre P_n(x), n = 0..N.
void legendre_p(double x, int N, std::vector<double>& P) {
  P.assign(N + 1, 0.0);
  P[0] = 1.0;
  if (N >= 1) P[1] = x;
  for (int n = 1; n < N; ++n) P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0);
}

}  // namespace

ElectroProblem::ElectroProblem(const Space& sp, std::vector<ElectroRole> roles, int nbody,
                               std::map<int, int> sheet_body_by_attr,
                               std::vector<std::pair<int, std::vector<int>>> extra_dofs)
    : sp_(sp), roles_(std::move(roles)), nbody_(nbody), sheet_body_(std::move(sheet_body_by_attr)),
      extra_(std::move(extra_dofs)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: electrostatic role table size mismatch");
  for (auto& r : roles_)
    AXF_REQUIRE(r.body < nbody_, "internal: electrostatic role refers to an unknown conductor");
  for (auto& [attr, body] : sheet_body_)
    AXF_REQUIRE(body >= 0 && body < nbody_, "internal: sheet conductor out of range");
  for (auto& [body, faces] : extra_)
    AXF_REQUIRE(body >= 0 && body < nbody_, "internal: terminal conductor out of range");
}

void ElectroProblem::set_outer(int outer_attr, double radius, double eps_out, bool grounded, int modes) {
  outer_attr_ = outer_attr;
  Rb_ = radius;
  eps_out_ = eps_out;
  grounded_ = grounded;
  modes_ = modes;
  ready_ = false;
}

std::string ElectroProblem::bname(int b) const {
  if (b >= 0 && b < static_cast<int>(names_.size()) && !names_[b].empty()) return "'" + names_[b] + "'";
  return strcat("#", b);
}

void ElectroProblem::setup() {
  if (ready_) return;
  Timer timer;
  const int nd = sp_.ndof(), ne = sp_.mesh.GetNE();
  auto assembled = [&](int e) { return roles_[e].body < 0 && !roles_[e].insulating; };

  // ------------------------------------------------------------ DOF classes
  kind_.assign(nd, 0);
  tie_.assign(nd, -1);
  index_.assign(nd, -1);
  auto tie = [&](int dof, int b, const char* what) {
    if (tie_[dof] >= 0 && tie_[dof] != b) {
      throw Error(strcat("the conductors ", bname(tie_[dof]), " and ", bname(b), " touch at (r, z) = (",
                         sp_.r[dof], ", ", sp_.z[dof], ")"),
                  "/regions",
                  strcat("Touching conductors form one body. Regions and sheets that share a face, an "
                         "edge or a corner are merged automatically, so this means two *readout "
                         "terminals* share a degree of freedom: the port gap is not resolved by the "
                         "mesh. Refine it (mesh.surface_size) or widen the gap. (", what, ")"));
    }
    tie_[dof] = b;
  };
  for (int e = 0; e < ne; ++e)
    if (roles_[e].body >= 0) for (int d : sp_.edofs(e)) tie(d, roles_[e].body, "region");
  // Elements of the outer semicircle.  The axis is a separate boundary attribute (kAxisAttr), but the
  // axis edges are also filtered out geometrically here (every DOF at r = 0): a scalar potential must
  // not be grounded on the symmetry axis, and the filter also protects against a mesh whose axis is
  // tagged "outer".  (The magnetoquasistatic and full-wave engines are insensitive to that, because
  // every boundary integrand there carries a factor r.)
  mfem::Array<int> bd;
  std::vector<char> outer(nd, 0);
  std::vector<int> obe;
  for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
    const int a = sp_.mesh.GetBdrAttribute(be);
    sp_.fes->GetBdrElementDofs(be, bd);
    auto it = sheet_body_.find(a);
    if (it != sheet_body_.end()) for (int d : bd) tie(d, it->second, "sheet");
    if (outer_attr_ > 0 && a == outer_attr_) {
      bool on_axis = true;
      for (int d : bd) if (sp_.r[d] != 0.0) on_axis = false;
      if (on_axis) continue;
      obe.push_back(be);
      for (int d : bd) outer[d] = 1;
    }
  }
  if (outer_attr_ > 0)
    AXF_REQUIRE(!obe.empty(), "no mesh element was found on the outer boundary", "/mesh/domain_radius");
  mfem::Array<int> fd;
  for (auto& [b, faces] : extra_) {
    for (int f : faces) {
      sp_.fes->GetFaceDofs(f, fd);
      for (int d : fd) tie(d, b, "terminal face");
    }
  }
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e)
    if (assembled(e)) for (int d : sp_.edofs(e)) used[d] = 1;
  for (int i = 0; i < nd; ++i) {
    if (tie_[i] >= 0) {
      AXF_REQUIRE(!outer[i],
                  strcat("the conductor ", bname(tie_[i]), " touches the outer boundary at (r, z) = (",
                         sp_.r[i], ", ", sp_.z[i], ")"),
                  "/mesh/domain_radius",
                  "The exterior map assumes vacuum (or a uniform dielectric) outside the domain sphere; "
                  "increase mesh.domain_radius.");
      kind_[i] = 2;
    } else if (grounded_ && outer[i]) {
      kind_[i] = 3;
    } else if (used[i]) {
      kind_[i] = 1;
    }
  }
  // Dielectric islands that touch no conductor, no ground and not the exterior map leave the potential
  // undetermined there (an air pocket inside an insulating body): pin them to zero.
  int pinned = 0;
  {
    int nc = 0;
    std::vector<int> comp = components(sp_, [&](int e) { return assembled(e); }, [](int) { return true; }, nc);
    std::vector<char> anchored(nc, 0);
    for (int e = 0; e < ne; ++e) {
      if (comp[e] < 0) continue;
      for (int d : sp_.edofs(e)) if (kind_[d] == 2 || kind_[d] == 3) anchored[comp[e]] = 1;
    }
    if (outer_attr_ > 0 && !grounded_) {
      for (int be : obe) {
        int e1 = -1, e2 = -1;
        sp_.mesh.GetFaceElements(sp_.mesh.GetBdrElementFaceIndex(be), &e1, &e2);
        if (e1 >= 0 && comp[e1] >= 0) anchored[comp[e1]] = 1;
      }
    }
    std::vector<char> ok(nd, 0);
    for (int e = 0; e < ne; ++e)
      if (comp[e] >= 0 && anchored[comp[e]]) for (int d : sp_.edofs(e)) ok[d] = 1;
    for (int i = 0; i < nd; ++i)
      if (kind_[i] == 1 && !ok[i]) { kind_[i] = 3; ++pinned; }
  }
  nfree_ = 0;
  for (int i = 0; i < nd; ++i) if (kind_[i] == 1) index_[i] = nfree_++;
  nred_ = nfree_ + nbody_;
  AXF_REQUIRE(nfree_ > 0, "the electrostatic problem has no free degrees of freedom");

  // prolongation: phi = P y, y = (free DOFs, body potentials)
  std::vector<Trip> pt;
  for (int i = 0; i < nd; ++i) {
    if (kind_[i] == 1) pt.emplace_back(i, index_[i], 1.0);
    else if (kind_[i] == 2) pt.emplace_back(i, nfree_ + tie_[i], 1.0);
  }
  P_.resize(nd, nred_);
  P_.setFromTriplets(pt.begin(), pt.end());

  // ------------------------------------------------------------ stiffness (per radian)
  std::vector<Trip> tk;
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  Eigen::MatrixXd Ke;
  for (int e = 0; e < ne; ++e) {
    if (!assembled(e)) continue;
    const double eps = roles_[e].eps;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    Ke.setZero(n, n);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double wr = pe.w * pe.r * eps;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          Ke(i, j) += wr * (pe.dN[2 * i] * pe.dN[2 * j] + pe.dN[2 * i + 1] * pe.dN[2 * j + 1]);
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) tk.emplace_back(d[i], d[j], Ke(i, j));
  }

  // ------------------------------------------------------------ exterior map (Laplace DtN)
  // phi = sum_{n>=0} a_n (Rb/rho)^{n+1} P_n(cos theta);  the exterior energy is
  //   W_ext = 2 pi (1/2) sum_n c_n p_n^2,  p_n = sum_b phi_b U(b, n),  U(b, n) = int N_b P_n r dl / Rb,
  //   c_n = (eps_out/Rb) (n+1) (2n+1)/2   [(2n+1)/2 = 1 / int_{-1}^{1} P_n^2 dx].
  bdofs_.clear();
  U_.resize(0, 0);
  coef_.resize(0);
  if (outer_attr_ > 0 && !grounded_ && modes_ > 0) {
    std::map<int, int> bidx;
    const int nbe = static_cast<int>(obe.size());
    for (int be : obe) {
      sp_.fes->GetBdrElementDofs(be, bd);
      for (int d : bd)
        if (!bidx.count(d)) {
          bidx[d] = static_cast<int>(bdofs_.size());
          bdofs_.push_back(d);
        }
    }
    const int N = std::min(modes_, std::max(4, nbe));
    U_ = Eigen::MatrixXd::Zero(bdofs_.size(), N);
    std::vector<double> Pn;
    mfem::Vector shape, x(2);
    for (int be : obe) {
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
        const double r = x(0), dl = ip.weight * T->Weight();
        legendre_p(std::cos(std::atan2(r, x(1))), N - 1, Pn);
        for (int i = 0; i < bd.Size(); ++i) {
          const double c = shape(i) * r * dl / Rb_;
          for (int n = 0; n < N; ++n) U_(bidx[bd[i]], n) += c * Pn[n];
        }
      }
    }
    coef_.resize(N);
    for (int n = 0; n < N; ++n) coef_(n) = eps_out_ / Rb_ * (n + 1.0) * (2.0 * n + 1.0) / 2.0;
    Eigen::MatrixXd D = U_ * coef_.asDiagonal() * U_.transpose();
    for (size_t a = 0; a < bdofs_.size(); ++a)
      for (size_t b = 0; b < bdofs_.size(); ++b) tk.emplace_back(bdofs_[a], bdofs_[b], D(a, b));
  }
  K_.resize(nd, nd);
  K_.setFromTriplets(tk.begin(), tk.end());
  A_ = P_.transpose() * K_ * P_;

  // ------------------------------------------------ surface quadrature on the outer semicircle
  // Used by exterior_charge_flux(): the strong flux -int eps grad(phi).n dA with n outward, which the
  // weakly imposed DtN map only reproduces in the limit, so it is a genuine discretization check.
  oflux_.clear();
  if (outer_attr_ > 0) {
    mfem::Mesh& mm = const_cast<mfem::Mesh&>(sp_.mesh);
    const mfem::IntegrationRule& iro = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp_.order + 6);
    for (int be : obe) {
      const int f = sp_.mesh.GetBdrElementFaceIndex(be);
      int e1 = -1, e2 = -1;
      sp_.mesh.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || !assembled(e1)) continue;
      mfem::FaceElementTransformations* FT = mm.GetFaceElementTransformations(f);
      for (int q = 0; q < iro.GetNPoints(); ++q) {
        FT->SetAllIntPoints(&iro.IntPoint(q));
        FluxPt p;
        p.e = e1;
        p.body = 0;
        p.eps = roles_[e1].eps;
        sp_.eval_at(e1, FT->GetElement1IntPoint(), p.pe);
        p.pe.ip = nullptr;
        mfem::Vector nor(2);
        mfem::CalcOrtho(FT->Jacobian(), nor);  // boundary face: out of element 1, |nor| = dl/dxi
        p.nr = nor(0);
        p.nz = nor(1);
        p.w = 2 * pi * p.pe.r * iro.IntPoint(q).weight;
        oflux_.push_back(std::move(p));
      }
    }
  }

  // ------------------------------------------------------------ surface quadrature for Gauss's law
  flux_.clear();
  {
    const mfem::Mesh& m = sp_.mesh;
    const mfem::IntegrationRule& ir1 = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp_.order + 6);
    auto add = [&](int f, int ev, int which, int body, double eps) {
      mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
      for (int q = 0; q < ir1.GetNPoints(); ++q) {
        FT->SetAllIntPoints(&ir1.IntPoint(q));
        const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
        FluxPt p;
        p.e = ev;
        p.body = body;
        p.eps = eps;
        sp_.eval_at(ev, eip, p.pe);
        p.pe.ip = nullptr;
        mfem::Vector nor(2);
        mfem::CalcOrtho(FT->Jacobian(), nor);  // from element 1 to element 2, |nor| = dl/dxi
        const double sgn = which == 1 ? -1.0 : 1.0;  // conductor -> dielectric
        p.nr = sgn * nor(0);
        p.nz = sgn * nor(1);
        p.w = 2 * pi * p.pe.r * ir1.IntPoint(q).weight;
        flux_.push_back(std::move(p));
      }
    };
    std::map<int, int> face_body;  // readout terminal faces
    for (auto& [b, faces] : extra_) for (int f : faces) face_body[f] = b;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1 = -1, e2 = -1;
      m.GetFaceElements(f, &e1, &e2);
      int body = -1;
      std::vector<std::pair<int, int>> sides;
      auto it = sheet_body_.find(sp_.face_bdr_attr[f]);
      auto jt = face_body.find(f);
      if (it != sheet_body_.end()) {
        body = it->second;
        if (e1 >= 0 && assembled(e1)) sides.push_back({e1, 1});
        if (e2 >= 0 && assembled(e2)) sides.push_back({e2, 2});
      } else if (jt != face_body.end()) {
        body = jt->second;
        if (e1 >= 0 && assembled(e1)) sides.push_back({e1, 1});
        if (e2 >= 0 && assembled(e2)) sides.push_back({e2, 2});
      } else if (e1 >= 0 && e2 >= 0) {
        const bool c1 = roles_[e1].body >= 0, c2 = roles_[e2].body >= 0;
        if (c1 != c2) {
          body = c1 ? roles_[e1].body : roles_[e2].body;
          sides.push_back(c1 ? std::make_pair(e2, 2) : std::make_pair(e1, 1));
        }
      }
      if (body < 0) continue;
      for (auto [ev, which] : sides) add(f, ev, which, body, roles_[ev].eps);
    }
  }

  int nins = 0;
  for (auto& r : roles_) if (r.insulating) ++nins;
  diag_ = {{"dofs", nd},
           {"free_dofs", nfree_},
           {"conductors", nbody_},
           {"pinned_dofs", pinned},
           {"insulating_elements", nins},
           {"boundary", grounded_ ? "grounded" : (outer_attr_ > 0 ? "open" : "none")},
           {"dtn_modes", static_cast<int>(coef_.size())},
           {"setup_seconds", timer.seconds()}};
  ready_ = true;
}

Vec ElectroProblem::multipoles(const Vec& phi) const {
  const int N = static_cast<int>(coef_.size());
  Vec a = Vec::Zero(std::max(N, 0));
  if (N == 0) return a;
  Vec ub(bdofs_.size());
  for (size_t i = 0; i < bdofs_.size(); ++i) ub(i) = phi(bdofs_[i]);
  Vec p = U_.transpose() * ub;
  for (int n = 0; n < N; ++n) a(n) = (2.0 * n + 1.0) / 2.0 * p(n) / Rb_;
  return a;
}

double ElectroProblem::exterior_charge(const Vec& phi) const {
  if (coef_.size() == 0) return 0.0;
  // Q = 4 pi eps_out Rb a_0 (the monopole of the exterior expansion).  See the header: this equals the
  // sum of the reaction charges identically, so it is not a discretization check.
  return 4 * pi * eps_out_ * Rb_ * multipoles(phi)(0);
}

double ElectroProblem::exterior_charge_flux(const Vec& phi) const {
  double Q = 0;
  for (const FluxPt& p : oflux_) {
    const auto& d = sp_.edofs(p.e);
    double gr = 0, gz = 0;
    for (size_t i = 0; i < d.size(); ++i) {
      gr += phi(d[i]) * p.pe.dN[2 * i];
      gz += phi(d[i]) * p.pe.dN[2 * i + 1];
    }
    // Q = int D.n dA = -int eps grad(phi).n dA with n outward (p.nr, p.nz carry the length Jacobian).
    Q -= p.eps * (gr * p.nr + gz * p.nz) * p.w;
  }
  return Q;
}

double ElectroProblem::energy_quadrature(const Vec& phi) const {
  const int ne = sp_.mesh.GetNE();
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  double w2 = 0;
  for (int e = 0; e < ne; ++e) {
    if (roles_[e].body >= 0 || roles_[e].insulating) continue;
    const auto& d = sp_.edofs(e);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      double gr = 0, gz = 0;
      for (size_t i = 0; i < d.size(); ++i) {
        gr += phi(d[i]) * pe.dN[2 * i];
        gz += phi(d[i]) * pe.dN[2 * i + 1];
      }
      w2 += pe.w * pe.r * roles_[e].eps * (gr * gr + gz * gz);
    }
  }
  if (coef_.size() > 0) {  // exterior multipole energy
    Vec ub(bdofs_.size());
    for (size_t i = 0; i < bdofs_.size(); ++i) ub(i) = phi(bdofs_[i]);
    Vec p = U_.transpose() * ub;
    for (int n = 0; n < coef_.size(); ++n) w2 += coef_(n) * p(n) * p(n);
  }
  return pi * w2;  // 2 pi * (1/2) * (per-radian quadratic form)
}

std::vector<double> ElectroProblem::charges_gauss(const Vec& phi) const {
  std::vector<double> Q(nbody_, 0.0);
  for (const FluxPt& p : flux_) {
    const auto& d = sp_.edofs(p.e);
    double gr = 0, gz = 0;
    for (size_t i = 0; i < d.size(); ++i) {
      gr += phi(d[i]) * p.pe.dN[2 * i];
      gz += phi(d[i]) * p.pe.dN[2 * i + 1];
    }
    // Q = -int eps grad(phi).n dA with n pointing from the conductor into the dielectric
    Q[p.body] -= p.eps * (gr * p.nr + gz * p.nz) * p.w;
  }
  return Q;
}

void ElectroProblem::fill(Solution& s, const Vec& y) const {
  s.phi = P_ * y;
  s.V.assign(nbody_, 0.0);
  for (int b = 0; b < nbody_; ++b) s.V[b] = y(nfree_ + b);
  Vec g = K_ * s.phi;
  s.Q.assign(nbody_, 0.0);
  for (int i = 0; i < sp_.ndof(); ++i)
    if (kind_[i] == 2) s.Q[tie_[i]] += 2 * pi * g(i);
  s.energy = pi * s.phi.dot(g);
  s.energy_field = energy_quadrature(s.phi);
  s.Q_gauss = charges_gauss(s.phi);
  s.charge_exterior = exterior_charge(s.phi);
  s.charge_exterior_flux = exterior_charge_flux(s.phi);
}

Eigen::MatrixXd ElectroProblem::capacitance(std::vector<Solution>* cols) {
  setup();
  Timer timer;
  const int n = nfree_, m = nbody_;
  AXF_REQUIRE(m > 0, "the electrostatic problem has no conductor to charge");
  std::vector<Trip> tf;
  Eigen::MatrixXd Afm = Eigen::MatrixXd::Zero(n, m), Amm = Eigen::MatrixXd::Zero(m, m);
  for (int c = 0; c < A_.outerSize(); ++c)
    for (SpMat::InnerIterator it(A_, c); it; ++it) {
      const int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
      if (i < n && j < n) tf.emplace_back(i, j, it.value());
      else if (i < n) Afm(i, j - n) += it.value();
      else if (j >= n) Amm(i - n, j - n) += it.value();
    }
  SpMat Aff(n, n);
  Aff.setFromTriplets(tf.begin(), tf.end());
  Eigen::SimplicialLDLT<SpMat> ch(Aff);
  AXF_REQUIRE(ch.info() == Eigen::Success,
              "the electrostatic system could not be factorized (singular or indefinite)", "",
              "Typical causes: a dielectric region enclosed by insulating walls with no conductor, or a "
              "zero-volume region.");
  // Column j: conductor j at 1 V, every other conductor at 0 V.  The charges come from the reactions
  // Q_i = 2 pi sum_{dof tied to i} (K phi)_dof, so C is not symmetric by construction and the symmetry
  // error measures the accuracy of the linear solves.
  Eigen::MatrixXd C(m, m);
  double res = 0;
  std::vector<Solution> local;
  std::vector<Solution>& out = cols ? *cols : local;
  for (int j = 0; j < m; ++j) {
    Vec x = ch.solve(-Afm.col(j));
    const double scale = std::max(Afm.col(j).norm(), 1e-300);
    const double rj = (Aff * x + Afm.col(j)).norm() / scale;
    res = std::max(res, rj);
    Vec y = Vec::Zero(nred_);
    y.head(n) = x;
    y(n + j) = 1.0;
    Solution s;
    fill(s, y);
    s.residual = rj;
    for (int i = 0; i < m; ++i) C(i, j) = s.Q[i];
    out.push_back(std::move(s));
  }
  (void)Amm;
  diag_["capacitance_residual"] = res;
  diag_["capacitance_seconds"] = timer.seconds();
  return C;
}

ElectroProblem::Solution ElectroProblem::solve(const std::vector<Terminal>& t) {
  setup();
  Timer timer;
  const int n = nfree_, m = nbody_;
  AXF_REQUIRE(static_cast<int>(t.size()) == m, "internal: terminal list size mismatch");
  std::vector<int> ridx(nred_, -1);
  for (int i = 0; i < n; ++i) ridx[i] = i;
  int k = n;
  Vec fixed = Vec::Zero(nred_);
  for (int b = 0; b < m; ++b) {
    if (t[b].prescribed) fixed(n + b) = t[b].V;
    else ridx[n + b] = k++;
  }
  Vec br = Vec::Zero(k);
  for (int b = 0; b < m; ++b)
    if (!t[b].prescribed) br(ridx[n + b]) = t[b].charge / (2 * pi);
  std::vector<Trip> tr;
  for (int c = 0; c < A_.outerSize(); ++c)
    for (SpMat::InnerIterator it(A_, c); it; ++it) {
      const int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
      if (ridx[i] < 0) continue;
      if (ridx[j] >= 0) tr.emplace_back(ridx[i], ridx[j], it.value());
      else br(ridx[i]) -= it.value() * fixed(j);
    }
  SpMat Ar(k, k);
  Ar.setFromTriplets(tr.begin(), tr.end());
  Eigen::SimplicialLDLT<SpMat> ch(Ar);
  AXF_REQUIRE(ch.info() == Eigen::Success,
              "the electrostatic system could not be factorized (singular or indefinite)", "",
              "A floating conductor or a dielectric region with no reference potential: prescribe a "
              "potential, ground the outer boundary, or use the open boundary.");
  Vec yr = ch.solve(br);
  Vec y = fixed;
  for (int i = 0; i < nred_; ++i) if (ridx[i] >= 0) y(i) = yr(ridx[i]);
  Solution s;
  fill(s, y);
  s.residual = (Ar * yr - br).norm() / std::max(br.norm(), 1e-300);
  diag_["solve_seconds"] = timer.seconds();
  return s;
}

}  // namespace axf::axi

// ================================================================ model hook
namespace axf {

using axi::ElectroProblem;
using axi::ElectroRole;
using axi::HKind;
using axi::kOuterAttr;
using axi::MeshBundle;
using axi::PointEval;
using axi::Space;
using axi::Terminal;
using units::mu0;
using units::pi;

namespace {

struct EInfo {
  std::string region;
  const spec::Material* mat = nullptr;
};

bool is_conductor(const spec::Material& m) {
  return m.kind == spec::MaterialKind::Superconductor || m.kind == spec::MaterialKind::Conductor;
}

// Connected conductor bodies: components of superconductor/conductor elements joined with the
// zero-thickness sheets that share vertices with them.  Mirrors find_bodies(include_conductors = true)
// in model.cpp (kept local so that the shared file stays untouched).
struct CBody {
  std::vector<std::string> names;
  std::vector<int> elements;
  std::vector<int> curve_attrs;
};

std::vector<CBody> conductor_bodies(const Space& sp, const MeshBundle& mb, const spec::Design& d,
                                    const std::vector<EInfo>& info, std::vector<int>& body_of_elem,
                                    std::map<int, int>& body_of_curve) {
  const int ne = sp.mesh.GetNE();
  int nc = 0;
  std::vector<int> comp =
      axi::components(sp, [&](int e) { return is_conductor(*info[e].mat); }, [](int) { return true; }, nc);
  std::vector<int> parent(nc);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  };
  // Components that share only a vertex (a corner in the meridian plane) are in contact along a whole
  // circle in 3D, and the H1 space forces their potentials equal at that node anyway: they are one
  // electrical body.  `components` only crosses faces, so unite them here.
  std::map<int, int> vert_comp;
  mfem::Array<int> v;
  for (int e = 0; e < ne; ++e) {
    if (comp[e] < 0) continue;
    sp.mesh.GetElementVertices(e, v);
    for (int x : v) {
      auto it = vert_comp.find(x);
      if (it == vert_comp.end()) vert_comp[x] = comp[e];
      else unite(it->second, comp[e]);
    }
  }
  std::map<int, int> curve_node;
  for (auto& c : d.curves) {
    if (!is_conductor(d.materials.at(c.material))) continue;
    const int attr = mb.attr_of_curve.at(c.name);
    const int node = static_cast<int>(parent.size());
    parent.push_back(node);
    curve_node[attr] = node;
  }
  for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
    auto it = curve_node.find(sp.mesh.GetBdrAttribute(be));
    if (it == curve_node.end()) continue;
    sp.mesh.GetBdrElementVertices(be, v);
    for (int x : v) {
      auto jt = vert_comp.find(x);
      if (jt != vert_comp.end()) unite(it->second, jt->second);
      else vert_comp[x] = it->second;
    }
  }
  std::map<int, int> root_body;
  std::vector<CBody> bodies;
  auto body_of = [&](int node) {
    const int rt = find(node);
    auto it = root_body.find(rt);
    if (it != root_body.end()) return it->second;
    root_body[rt] = static_cast<int>(bodies.size());
    bodies.emplace_back();
    return static_cast<int>(bodies.size()) - 1;
  };
  body_of_elem.assign(ne, -1);
  for (int e = 0; e < ne; ++e) {
    if (comp[e] < 0) continue;
    const int b = body_of(comp[e]);
    body_of_elem[e] = b;
    bodies[b].elements.push_back(e);
  }
  for (auto& [attr, node] : curve_node) {
    const int b = body_of(node);
    body_of_curve[attr] = b;
    bodies[b].curve_attrs.push_back(attr);
  }
  std::vector<std::set<std::string>> names(bodies.size());
  for (int e = 0; e < ne; ++e) if (body_of_elem[e] >= 0) names[body_of_elem[e]].insert(info[e].region);
  for (auto& c : d.curves) {
    auto it = mb.attr_of_curve.find(c.name);
    if (it != mb.attr_of_curve.end() && body_of_curve.count(it->second))
      names[body_of_curve[it->second]].insert(c.name);
  }
  for (size_t b = 0; b < bodies.size(); ++b) bodies[b].names.assign(names[b].begin(), names[b].end());
  return bodies;
}

// Maxwell matrix of the conductors `keep` with all the others floating at zero net charge:
// C_hat = C_pp - C_pf C_ff^{-1} C_fp.
Eigen::MatrixXd reduce_floating(const Eigen::MatrixXd& C, const std::vector<int>& keep) {
  const int m = static_cast<int>(C.rows()), p = static_cast<int>(keep.size());
  std::vector<int> fl;
  for (int i = 0; i < m; ++i)
    if (std::find(keep.begin(), keep.end(), i) == keep.end()) fl.push_back(i);
  Eigen::MatrixXd Cpp(p, p);
  for (int a = 0; a < p; ++a)
    for (int b = 0; b < p; ++b) Cpp(a, b) = C(keep[a], keep[b]);
  if (fl.empty()) return Cpp;
  const int nf = static_cast<int>(fl.size());
  Eigen::MatrixXd Cff(nf, nf), Cfp(nf, p);
  for (int a = 0; a < nf; ++a) {
    for (int b = 0; b < nf; ++b) Cff(a, b) = C(fl[a], fl[b]);
    for (int b = 0; b < p; ++b) Cfp(a, b) = C(fl[a], keep[b]);
  }
  Eigen::LDLT<Eigen::MatrixXd> ld(Cff);
  AXF_REQUIRE(ld.info() == Eigen::Success,
              "the capacitance sub-matrix of the floating conductors could not be factorized", "",
              "C_FF should be positive definite; a conductor with (numerically) zero capacitance to "
              "everything else makes it singular. Check the mesh and the conductor bodies.");
  Eigen::MatrixXd X = ld.solve(Cfp);
  const double scale = std::max(Cfp.cwiseAbs().maxCoeff(), 1e-300);
  AXF_REQUIRE((Cff * X - Cfp).cwiseAbs().maxCoeff() <= 1e-6 * scale,
              "the reduction to floating conductors did not converge (C_FF is ill-conditioned)", "",
              "Two conductors are nearly at the same potential for every excitation; merge them or "
              "refine the mesh.");
  return Cpp - Cfp.transpose() * X;
}

json matrix_json(const Eigen::MatrixXd& C) {
  json out = json::array();
  for (int i = 0; i < C.rows(); ++i) {
    json row = json::array();
    for (int j = 0; j < C.cols(); ++j) row.push_back(C(i, j));
    out.push_back(row);
  }
  return out;
}

}  // namespace

// Electrostatics: the Maxwell capacitance matrix of the conductors, the pairwise (mutual)
// capacitances, and the readout-gap capacitance with the pickup self-resonance.
void AxiModel::electrostatics_analysis(const SolveOptions& o) {
  (void)o;
  const double Lp = results_["coupling"].value("L_p_H", 0.0);
  const bool want_matrix = d_.electrostatics.enabled;
  const bool want_port = h_pick_ != nullptr;
  if (!want_matrix && !want_port) return;
  Timer timer;
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  json& R = results_;
  json& warnings = R["warnings"];
  const double eps0 = 1.0 / (mu0 * units::c * units::c);

  // ------------------------------------------------------------ materials and bodies
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  std::vector<EInfo> info(ne);
  std::vector<double> eps(ne);
  for (int e = 0; e < ne; ++e) {
    info[e].region = region_name(m.GetAttribute(e));
    // "air" is the implicit background outside every named region; the spec parser reserves the name,
    // so this never shadows a user region (it did before: a dielectric shell named "air" was silently
    // treated as vacuum).
    if (info[e].region == "air") info[e].mat = &vacuum;
    else info[e].mat = &d_.materials.at(d_.find_region(info[e].region)->material);
    eps[e] = eps0 * info[e].mat->eps_r;
  }
  std::vector<int> body_of_elem;
  std::map<int, int> body_of_curve;
  auto bodies = conductor_bodies(sp, mb_, d_, info, body_of_elem, body_of_curve);
  const int nb = static_cast<int>(bodies.size());
  auto body_name = [&](int b) { return bodies[b].names.empty() ? std::string("body") : bodies[b].names.front(); };

  // permittivity just outside the domain sphere (the exterior map assumes it is uniform there).
  // Axis edges are filtered out geometrically (all DOFs at r = 0), as in ElectroProblem::setup().
  double eps_out = -1;
  bool eps_out_mixed = false;
  mfem::Array<int> obd;
  for (int be = 0; be < m.GetNBE(); ++be) {
    if (m.GetBdrAttribute(be) != kOuterAttr) continue;
    sp.fes->GetBdrElementDofs(be, obd);
    bool on_axis = true;
    for (int d : obd) if (sp.r[d] != 0.0) on_axis = false;
    if (on_axis) continue;
    int e1 = -1, e2 = -1;
    m.GetFaceElements(m.GetBdrElementFaceIndex(be), &e1, &e2);
    if (e1 < 0) continue;
    if (eps_out < 0) eps_out = eps[e1];
    else if (std::fabs(eps[e1] - eps_out) > 1e-12 * eps_out) eps_out_mixed = true;
  }
  if (eps_out < 0) eps_out = eps0;
  if (eps_out_mixed) {
    eps_out = eps0;
    warnings.push_back("electrostatics: the permittivity is not uniform at the outer boundary; the exterior "
                       "map uses eps0 (move the dielectric away from mesh.domain_radius)");
  }

  // The exterior condition applies to every electrostatic problem of this design, the readout-gap solve
  // included (a hard-coded "open" there would silently ignore `electrostatics.boundary: "grounded"`).
  // It defaults to "open" when the design has no `electrostatics` section.
  const bool grounded = d_.electrostatics.boundary == "grounded";

  // ------------------------------------------------------------ capacitance matrix
  if (want_matrix) {
    AXF_REQUIRE(nb > 0,
                "electrostatics needs at least one conductor: a region or curve of a 'superconductor' or "
                "'conductor' material",
                "/electrostatics", "Add a conductor region, or remove the 'electrostatics' section.");
    std::vector<ElectroRole> roles(ne);
    for (int e = 0; e < ne; ++e) {
      roles[e].body = body_of_elem[e];
      roles[e].eps = eps[e];
    }
    ElectroProblem p(sp, roles, nb, body_of_curve);
    std::vector<std::string> names;
    for (int b = 0; b < nb; ++b) names.push_back(body_name(b));
    p.set_names(names);
    p.set_outer(kOuterAttr, mb_.domain_radius, eps_out, grounded, 80);
    log(1, "solving electrostatics (capacitance matrix)");
    std::vector<ElectroProblem::Solution> cols;
    Eigen::MatrixXd C = p.capacitance(&cols);
    const double cmax = C.cwiseAbs().maxCoeff();

    // checks ---------------------------------------------------------------
    double sym = 0;
    for (int i = 0; i < nb; ++i)
      for (int j = 0; j < i; ++j) sym = std::max(sym, std::fabs(C(i, j) - C(j, i)));
    // symmetrize for the reported matrix and the derived quantities
    Eigen::MatrixXd Cs = 0.5 * (C + C.transpose());
    json dom = json::array();
    double min_slack = 1e300;
    bool signs_ok = true;
    std::vector<double> cg(nb, 0.0);
    for (int i = 0; i < nb; ++i) {
      double off = 0;
      for (int j = 0; j < nb; ++j) {
        if (j == i) continue;
        off += std::fabs(Cs(i, j));
        if (Cs(i, j) > 1e-9 * cmax) signs_ok = false;
        cg[i] += Cs(i, j);
      }
      cg[i] += Cs(i, i);
      const double slack = Cs(i, i) - off;
      min_slack = std::min(min_slack, slack / std::max(Cs(i, i), 1e-300));
      dom.push_back({{"body", body_name(i)}, {"C_ii_F", Cs(i, i)}, {"sum_abs_off_diagonal_F", off},
                     {"slack_F", slack}, {"C_to_ground_F", cg[i]}});
    }
    double e_rel = 0, q_rel = 0, mono_rel = 0, eflux_rel = 0;
    json colj = json::array();
    for (int j = 0; j < nb; ++j) {
      const auto& s = cols[j];
      const double half = 0.5 * Cs(j, j);
      const double de = std::fabs(s.energy_field - half) / std::max(std::fabs(half), 1e-300);
      double dq = 0, qmax = 0;
      for (int i = 0; i < nb; ++i) {
        qmax = std::max(qmax, std::fabs(s.Q[i]));
        dq = std::max(dq, std::fabs(s.Q_gauss[i] - s.Q[i]));
      }
      dq /= std::max(qmax, 1e-300);
      double qtot = 0;
      for (int i = 0; i < nb; ++i) qtot += s.Q[i];
      // the net charge can cancel (a screened conductor), so scale by the largest charge of the column
      const double dm = grounded ? 0.0 : std::fabs(s.charge_exterior - qtot) / std::max(qmax, 1e-300);
      // The strong flux through the outer sphere: a genuine discretization check (see the note below).
      // It applies to both boundary conditions -- with the DtN map the flux must match the net charge
      // that the map carries away, and with a grounded sphere Gauss's law says the flux through it is
      // the charge enclosed -- unlike the monopole identity, which exists only for the open boundary.
      const double df = std::fabs(s.charge_exterior_flux - qtot) / std::max(qmax, 1e-300);
      e_rel = std::max(e_rel, de);
      q_rel = std::max(q_rel, dq);
      mono_rel = std::max(mono_rel, dm);
      eflux_rel = std::max(eflux_rel, df);
      colj.push_back({{"body", body_name(j)}, {"energy_J", s.energy_field}, {"half_C_jj_J", half},
                      {"energy_relative_difference", de}, {"charge_gauss_relative_difference", dq},
                      {"net_charge_C", qtot}, {"exterior_monopole_C", s.charge_exterior},
                      {"exterior_flux_C", s.charge_exterior_flux},
                      {"exterior_flux_relative_difference", df},
                      {"residual", s.residual}});
    }
    // superposition / energy check at a potential vector that is not a unit column
    Eigen::VectorXd Vtest(nb);
    for (int i = 0; i < nb; ++i) Vtest(i) = 1.0 + 0.37 * i;
    Eigen::VectorXd phi_t = Eigen::VectorXd::Zero(sp.ndof());
    for (int j = 0; j < nb; ++j) phi_t += Vtest(j) * cols[j].phi;
    const double W_field = p.energy_quadrature(phi_t);
    const double W_mat = 0.5 * Vtest.dot(Cs * Vtest);
    const double W_rel = std::fabs(W_field - W_mat) / std::max(std::fabs(W_mat), 1e-300);

    // floating conductors: one independent solve against the matrix prediction
    json floatj = json::object();
    if (nb >= 2) {
      std::vector<Terminal> t(nb);
      t[0].prescribed = true;
      t[0].V = 1.0;
      auto sf = p.solve(t);
      // predicted: V_F = -C_FF^{-1} C_F0
      Eigen::MatrixXd Cff(nb - 1, nb - 1);
      Eigen::VectorXd Cf0(nb - 1);
      for (int a = 1; a < nb; ++a) {
        for (int b = 1; b < nb; ++b) Cff(a - 1, b - 1) = Cs(a, b);
        Cf0(a - 1) = Cs(a, 0);
      }
      Eigen::LDLT<Eigen::MatrixXd> ld(Cff);
      AXF_REQUIRE(ld.info() == Eigen::Success,
                  "the capacitance sub-matrix of the floating conductors could not be factorized", "",
                  "C_FF should be positive definite; check the mesh and the conductor bodies.");
      Eigen::VectorXd Vp = -ld.solve(Cf0);
      double dv = 0;
      std::vector<double> got, want;
      for (int a = 1; a < nb; ++a) {
        got.push_back(sf.V[a]);
        want.push_back(Vp(a - 1));
        dv = std::max(dv, std::fabs(sf.V[a] - Vp(a - 1)));
      }
      floatj = {{"driven", body_name(0)}, {"V_driven", 1.0}, {"V_floating", got},
                {"V_floating_from_matrix", want}, {"max_difference_V", dv},
                {"charge_on_driven_C", sf.Q[0]}, {"residual", sf.residual},
                {"note", "body 0 at 1 V, every other conductor floating at zero net charge; the potentials "
                         "must match -C_FF^{-1} C_F0 from the Maxwell matrix"}};
    }

    // pairwise (mutual) capacitances ----------------------------------------
    json pairs = json::array();
    for (int i = 0; i < nb; ++i)
      for (int j = i + 1; j < nb; ++j) {
        Eigen::MatrixXd Ch = reduce_floating(Cs, {i, j});
        pairs.push_back({{"a", body_name(i)},
                         {"b", body_name(j)},
                         {"C_mutual_others_grounded_F", -Cs(i, j)},
                         {"C_mutual_others_floating_F", -Ch(0, 1)},
                         {"C_a_to_ground_others_floating_F", Ch(0, 0) + Ch(0, 1)},
                         {"C_b_to_ground_others_floating_F", Ch(1, 1) + Ch(0, 1)}});
      }

    json out = {{"boundary", grounded ? "grounded" : "open"},
                {"domain_radius_m", mb_.domain_radius},
                {"eps_out_F_per_m", eps_out},
                {"bodies", json::array()},
                {"capacitance_F", matrix_json(Cs)},
                {"capacitance_to_ground_F", cg},
                {"pairs", pairs},
                {"checks", {{"symmetry_absolute_F", sym},
                            {"symmetry_relative", sym / std::max(cmax, 1e-300)},
                            {"diagonal_dominance", dom},
                            {"min_slack_relative", nb > 0 ? min_slack : 0.0},
                            {"off_diagonal_signs_ok", signs_ok},
                            {"columns", colj},
                            {"energy_max_relative_difference", e_rel},
                            {"charge_gauss_max_relative_difference", q_rel},
                            {"exterior_monopole_max_relative_difference", mono_rel},
                            {"exterior_flux_max_relative_difference", eflux_rel},
                            {"energy_superposition", {{"V", std::vector<double>(Vtest.data(), Vtest.data() + nb)},
                                                      {"W_field_J", W_field}, {"W_from_C_J", W_mat},
                                                      {"relative_difference", W_rel}}},
                            {"floating_solve", floatj},
                            {"what_each_check_tests",
                             "ALGEBRAIC IDENTITIES of the discrete system -- they verify the assembly "
                             "and the linear solves, not the discretization: symmetry_* (the matrix is "
                             "built from reactions, not from a symmetrized Schur complement), "
                             "energy_* and energy_superposition (the same quadratic form reassembled by "
                             "quadrature), floating_solve (the exact matrix reduction) and "
                             "exterior_monopole_* (1^T K phi = eps_out p_0 identically, so the sum of "
                             "the reaction charges equals 4 pi eps_out Rb a_0 up to the solver residual "
                             "and the geometric error of the boundary arc). "
                             "DISCRETIZATION CHECKS -- they converge under h/p refinement and are the "
                             "ones to read: charge_gauss_* (reactions against the strong surface "
                             "integral of eps E.n over the conductors) and exterior_flux_* (the net "
                             "charge from the strong flux -int eps grad(phi).n dA over the outer "
                             "sphere, which the weakly imposed DtN map only reproduces in the limit)."}}},
                {"solver", p.diagnostics()},
                {"convention",
                 "Maxwell matrix: C(i, j) = Q_i for V_j = 1 V and every other conductor at 0 V; ground is "
                 "infinity (open) or the outer sphere (grounded). C_to_ground = row sum. The pairwise "
                 "mutual capacitance is -C(i, j) (the lumped capacitor of the equivalent network, i.e. "
                 "with the other conductors grounded); C_mutual_others_floating_F instead holds the other "
                 "conductors at zero net charge (an exact reduction of the matrix)."}};
    for (int i = 0; i < nb; ++i)
      out["bodies"].push_back({{"name", body_name(i)}, {"regions", bodies[i].names}});
    if (!d_.electrostatics.notes.empty()) out["notes"] = d_.electrostatics.notes;
    if (!signs_ok)
      warnings.push_back("electrostatics: a positive off-diagonal capacitance appeared; the Maxwell matrix "
                         "should have C(i, j) <= 0 for i != j (check the mesh and the conductor bodies)");
    if (min_slack < -1e-6)
      warnings.push_back(strcat("electrostatics: the Maxwell matrix is not diagonally dominant (relative "
                                "slack ", min_slack, "); the discretization is too coarse"));
    R["electrostatics"] = out;
  }

  // ------------------------------------------------------------ readout-port gap capacitance
  // The port is a gap between two faces of one conductor body.  Its two terminal faces become the
  // electrodes at V = +-1/2; the capacitance across the readout is the mutual capacitance of the pair.
  if (want_port) {
    std::vector<char> inport(ne, 0);
    for (int e = 0; e < ne; ++e)
      if (h_pick_->roles()[e].kind == HKind::Profile) inport[e] = 1;
    std::vector<int> cfaces;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1 = -1, e2 = -1;
      m.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0 || inport[e1] == inport[e2]) continue;
      const int other = inport[e1] ? e2 : e1;
      if (body_of_elem[other] >= 0) cfaces.push_back(f);
    }
    std::map<int, std::vector<int>> v2f;
    mfem::Array<int> fv;
    for (int f : cfaces) {
      m.GetFaceVertices(f, fv);
      for (int v : fv) v2f[v].push_back(f);
    }
    std::map<int, int> patch;
    int npatch = 0;
    for (int f0 : cfaces) {
      if (patch.count(f0)) continue;
      std::vector<int> st{f0};
      patch[f0] = npatch;
      while (!st.empty()) {
        const int f = st.back();
        st.pop_back();
        m.GetFaceVertices(f, fv);
        for (int v : fv)
          for (int g : v2f[v])
            if (!patch.count(g)) { patch[g] = npatch; st.push_back(g); }
      }
      ++npatch;
    }
    if (npatch != 2) {
      warnings.push_back(strcat("port gap capacitance not computed: the readout port touches conductors on ",
                                npatch, " separate faces (expected 2 terminal faces)"));
    } else {
      std::vector<std::vector<int>> pf(2);
      std::vector<int> pbody(2, -1);
      double area[2] = {0, 0};
      for (auto& [f, pidx] : patch) {
        pf[pidx].push_back(f);
        int e1 = -1, e2 = -1;
        m.GetFaceElements(f, &e1, &e2);
        const int other = inport[e1] ? e2 : e1;
        pbody[pidx] = body_of_elem[other];
        // 2 pi r dl of the terminal face
        mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(m).GetFaceTransformation(f);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
        mfem::Vector x(2);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          T->Transform(ir.IntPoint(q), x);
          area[pidx] += 2 * pi * x(0) * T->Weight() * ir.IntPoint(q).weight;
        }
      }
      const int own = pbody[0];
      json pc = {{"port", d_.pickup->port},
                 {"terminal_area_m2", std::vector<double>{area[0], area[1]}},
                 {"terminals_on_one_body", pbody[0] == pbody[1]}};
      // (a) the port region alone: the two terminal faces at V = +-1/2, no fringing.  What is rigorous
      // is NOT C_gap >= C_reg but the Dirichlet-principle inequality verified in `checks` below:
      // restricting the full-domain minimizer to the port subdomain is admissible for the region
      // problem, so W_reg <= W_full, i.e. C_reg <= C_gap + (C_Ag + C_Bg)/4.
      double C_reg = 0;
      {
        std::vector<ElectroRole> roles(ne);
        for (int e = 0; e < ne; ++e) {
          roles[e].insulating = !inport[e];
          roles[e].eps = eps[e];
        }
        ElectroProblem p(sp, roles, 2, {}, {{0, pf[0]}, {1, pf[1]}});
        p.set_names({"readout terminal A", "readout terminal B"});
        std::vector<ElectroProblem::Solution> cols;
        Eigen::MatrixXd C = p.capacitance(&cols);
        C_reg = -0.5 * (C(0, 1) + C(1, 0));
        pc["C_gap_port_region_F"] = C_reg;
        pc["port_region_solver"] = p.diagnostics();
        pc["checks"]["region_charge_gauss_relative"] =
            std::fabs(cols[0].Q_gauss[0] - cols[0].Q[0]) / std::max(std::fabs(cols[0].Q[0]), 1e-300);
        pc["checks"]["region_symmetry_relative"] =
            std::fabs(C(0, 1) - C(1, 0)) / std::max(C.cwiseAbs().maxCoeff(), 1e-300);
      }
      // (b) the full domain: the body carrying the terminals is cut at the port, i.e. its surface away
      // from the two terminal faces is insulating (n.D = 0) and the two terminal faces are the
      // electrodes; every other conductor floats at zero net charge.
      double C_gap = C_reg;
      if (pbody[0] == pbody[1]) {
        std::vector<ElectroRole> roles(ne);
        std::map<int, int> sheets;
        int next = 2;
        std::vector<int> remap(nb, -1);
        for (int b = 0; b < nb; ++b) if (b != own) remap[b] = next++;
        for (int e = 0; e < ne; ++e) {
          roles[e].eps = eps[e];
          const int b = body_of_elem[e];
          if (b == own) roles[e].insulating = true;
          else if (b >= 0) roles[e].body = remap[b];
        }
        for (auto& [attr, b] : body_of_curve) if (b != own) sheets[attr] = remap[b];
        ElectroProblem p(sp, roles, next, sheets, {{0, pf[0]}, {1, pf[1]}});
        std::vector<std::string> names{"readout terminal A", "readout terminal B"};
        names.resize(next);
        for (int b = 0; b < nb; ++b) if (remap[b] >= 0) names[remap[b]] = body_name(b);
        p.set_names(names);
        p.set_outer(kOuterAttr, mb_.domain_radius, eps_out, grounded, 80);
        log(1, "solving electrostatics (readout gap capacitance)");
        std::vector<ElectroProblem::Solution> cols;
        Eigen::MatrixXd C = p.capacitance(&cols);
        Eigen::MatrixXd Cs = 0.5 * (C + C.transpose());
        Eigen::MatrixXd Ch = reduce_floating(Cs, {0, 1});
        C_gap = -Ch(0, 1);
        const double C_Ag = Ch(0, 0) + Ch(0, 1), C_Bg = Ch(1, 1) + Ch(0, 1);
        pc["C_gap_F"] = C_gap;
        pc["C_terminal_to_ground_F"] = std::vector<double>{C_Ag, C_Bg};
        pc["gap_solver"] = p.diagnostics();
        // The one rigorous relation between the two values.  Drive the terminals at V = +-1/2: the
        // field energy of the full-domain solution is W_full = C_gap/2 + (C_Ag + C_Bg)/8, and its
        // restriction to the port subdomain is admissible for the region-only problem (same mesh, same
        // space, same terminal values, natural conditions elsewhere), whose minimum is W_reg = C_reg/2.
        // Energy is positive, so W_reg <= W_full:  C_reg <= C_gap + (C_Ag + C_Bg)/4.  The inequality is
        // exact for the discrete problems, and it is loose: the extra term is not small.
        const double bound = C_gap + 0.25 * (C_Ag + C_Bg);
        pc["checks"]["dirichlet_bound"] = {
            {"C_gap_port_region_F", C_reg},
            {"C_gap_F", C_gap},
            {"C_terminal_to_ground_F", std::vector<double>{C_Ag, C_Bg}},
            {"C_gap_plus_quarter_terminal_to_ground_F", bound},
            {"bound_over_C_region", C_reg > 0 ? bound / C_reg : 0.0},
            {"holds", bound >= C_reg * (1 - 1e-9)},
            {"C_gap_over_C_region", C_reg > 0 ? C_gap / C_reg : 0.0},
            {"note", "rigorous (Dirichlet principle, exact for the discrete problems): "
                     "C_gap_port_region <= C_gap + (C_A_to_ground + C_B_to_ground)/4. The stronger "
                     "C_gap >= C_gap_port_region is NOT proven; it is observed on every case run so far, "
                     "with C_gap/C_region falling towards 1 as the gap grows relative to the electrode "
                     "spacing (1.02 at the widest gap tested, 5.6 at the narrowest), but the bound above "
                     "is the only guaranteed one."}};
        pc["checks"]["gap_symmetry_relative"] =
            (C - C.transpose()).cwiseAbs().maxCoeff() / std::max(C.cwiseAbs().maxCoeff(), 1e-300);
        // The electrode is a set of degrees of freedom, so the reaction charge also collects the flux
        // that reaches the terminal DOFs from just outside the terminal faces (within one element of
        // the gap mouths); the surface integral over the terminal faces alone therefore differs.
        pc["checks"]["gap_charge_reaction_C"] = cols[0].Q[0];
        pc["checks"]["gap_charge_terminal_faces_C"] = cols[0].Q_gauss[0];
        pc["checks"]["gap_charge_fraction_beyond_terminal_faces"] =
            1.0 - cols[0].Q_gauss[0] / std::max(std::fabs(cols[0].Q[0]), 1e-300);
        pc["checks"]["gap_energy_relative"] =
            std::fabs(cols[0].energy_field - 0.5 * Cs(0, 0)) / std::max(0.5 * std::fabs(Cs(0, 0)), 1e-300);
      } else {
        pc["C_gap_F"] = C_reg;
        pc["note_bodies"] = "the two terminal faces belong to different conductor bodies; only the "
                            "port-region capacitance is computed (use the 'electrostatics' section for the "
                            "full matrix of the two bodies)";
      }
      if (Lp > 0 && C_gap > 0) {
        pc["L_p_H"] = Lp;
        pc["self_resonance_Hz"] = 1.0 / (2 * pi * std::sqrt(Lp * C_gap));
        pc["self_resonance_from_port_region_Hz"] = C_reg > 0 ? 1.0 / (2 * pi * std::sqrt(Lp * C_reg)) : 0.0;
      }
      pc["note"] =
          "C_gap_port_region_F: the electrostatic capacitance of the gap region alone, with natural "
          "(insulating) conditions on the faces open to the cavity -- no fringing. C_gap_F: the same two "
          "terminal faces driven at V = +-1/2 on the whole domain, with the rest of their own conductor "
          "body insulating (n.D = 0) and every other conductor floating -- this adds the fringing field "
          "at the two mouths of the gap. The only rigorous relation between them is the Dirichlet bound "
          "in checks/dirichlet_bound; neither is a proven bound on the other. Both are lumped-element "
          "estimates: the body is electrically continuous around the port, so a capacitance across the "
          "readout exists only in the lumped model. f = 1/(2 pi sqrt(L_p C_gap)) is an ESTIMATE of the "
          "first self-resonance, not a bound in either direction: C_gap is only the gap's share of the "
          "structure's capacitance (which pushes the estimate up), while a distributed line resonates at "
          "(pi/2) / (2 pi sqrt(L C_total)) (which pushes the true pole up). The estimate exceeds the "
          "first pole only while C_gap/C_total < (2/pi)^2 = 0.405; it is 0.09 and 0.16 for the shipped "
          "solenoid and core templates, which is why it came out high there. The pole of L(omega) in a "
          "full-wave sweep (design key 'fullwave') is the reference for distributed structures.";
      // Reference from the full-wave readout impedance, when a sweep is present: the first
      // self-resonance is the pole of L(omega), i.e. where L changes sign.  (The closed-sphere
      // eigenmodes are not used: for a large domain the lowest ones belong to the sphere, not to the
      // readout.)
      if (Lp > 0 && pc.contains("self_resonance_Hz") && R.contains("fullwave") &&
          R["fullwave"].contains("sweep")) {
        const json& sw = R["fullwave"]["sweep"];
        double fa = 0, fb = 0;
        for (size_t i = 1; i < sw.size(); ++i) {
          if (sw[i - 1].value("L_H", 0.0) > 0 && sw[i].value("L_H", 0.0) < 0) {
            fa = sw[i - 1].value("f_Hz", 0.0);
            fb = sw[i].value("f_Hz", 0.0);
            break;
          }
        }
        if (fa > 0) {
          const double f0 = std::sqrt(fa * fb);
          pc["vs_fullwave"] = {
              {"first_pole_bracket_Hz", std::vector<double>{fa, fb}},
              {"first_pole_Hz", f0},
              {"lumped_over_fullwave", pc["self_resonance_Hz"].get<double>() / f0},
              {"C_effective_F", 1.0 / (4 * pi * pi * f0 * f0 * Lp)},
              {"note", "the first self-resonance of the readout is the pole of L(omega) in the full-wave "
                       "sweep (L changes sign); C_effective_F = 1/((2 pi f)^2 L_p) is the capacitance that "
                       "reproduces it, of which C_gap_F is a part"}};
        }
      }
      R["pickup"]["port_capacitance"] = pc;
      const double fsr = pc.value("self_resonance_Hz", 0.0);
      if (fsr > 0 && fsr < d_.scan.f_max)
        warnings.push_back(strcat("the readout-gap capacitance (", C_gap, " F) puts the estimated pickup "
                                  "self-resonance at ", fsr, " Hz, below the top of the scan (",
                                  d_.scan.f_max, " Hz). 1/(2 pi sqrt(L_p C_gap)) is an estimate, not a "
                                  "bound: use a full-wave sweep ('fullwave') for the real first pole"));
    }
  }
  R["timing"]["electrostatics_s"] = timer.seconds();
}

}  // namespace axf
