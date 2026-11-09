// Current-distribution solver; see cpp/include/axionfem/chem/current_distribution.hpp.
#include "axionfem/chem/current_distribution.hpp"

#include <Eigen/SparseCholesky>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "mfem.hpp"

namespace axf::chem {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("axf::chem::CurrentDistribution: " + msg); }

double clipped_exp(double arg, double clip, bool* clipped) {
  if (arg > clip) {
    if (clipped) *clipped = true;
    arg = clip;
  } else if (arg < -clip) {
    if (clipped) *clipped = true;
    arg = -clip;
  }
  return std::exp(arg);
}

// Legendre P_0..P_m at t
std::vector<double> legendre(int m, double t) {
  std::vector<double> P(m + 1);
  P[0] = 1.0;
  if (m >= 1) P[1] = t;
  for (int l = 1; l < m; ++l) P[l + 1] = ((2.0 * l + 1.0) * t * P[l] - l * P[l - 1]) / (l + 1.0);
  return P;
}
}  // namespace

Reaction::Kind reaction_kind_from_string(const std::string& s) {
  if (s == "bv") return Reaction::Kind::BV;
  if (s == "linear") return Reaction::Kind::Linear;
  if (s == "tafel_a") return Reaction::Kind::TafelAnodic;
  if (s == "tafel_c") return Reaction::Kind::TafelCathodic;
  fail("unknown reaction kind '" + s + "': use bv, linear, tafel_a or tafel_c");
}

double reaction_current(const Reaction& r, double eta, double f, double clip, double* didEta, bool* clipped) {
  switch (r.kind) {
    case Reaction::Kind::BV: {
      const double ea = clipped_exp(r.alpha_a * f * eta, clip, clipped);
      const double ec = clipped_exp(-r.alpha_c * f * eta, clip, clipped);
      if (didEta) *didEta = r.i0 * f * (r.alpha_a * ea + r.alpha_c * ec);
      return r.i0 * (ea - ec);
    }
    case Reaction::Kind::Linear: {
      const double k = r.i0 * (r.alpha_a + r.alpha_c) * f;
      if (didEta) *didEta = k;
      return k * eta;
    }
    case Reaction::Kind::TafelAnodic:
    case Reaction::Kind::TafelCathodic: {
      if (r.A == 0.0) fail("Tafel reaction with slope A = 0: give A in V/decade (> 0 anodic, < 0 cathodic)");
      const double s = std::log(10.0) / r.A;
      const double e = clipped_exp(s * eta, clip, clipped);
      const double sign = r.kind == Reaction::Kind::TafelAnodic ? 1.0 : -1.0;
      if (didEta) *didEta = sign * r.i0 * s * e;
      return sign * r.i0 * e;
    }
  }
  return 0.0;
}

// ------------------------------------------------------------------------------------------------ Impl
struct BPoint {
  double x, y, w, nx, ny;
};
struct BElem {
  int be, attr, elem;
  std::vector<int> dofs;         // element dofs of the adjacent element
  std::vector<BPoint> pts;
  std::vector<std::vector<double>> shape;  // per point, per element dof
};

struct Electrode {
  ElectrodeSpec spec;
  std::vector<int> belems;   // indices into Impl::bdr (assembly degree)
  std::vector<double> g;     // per point (concatenated over belems)
  int unknown = -1;          // tie or phi_s unknown (galvanostatic)
  std::vector<int> dofs;
};

struct CurrentDistribution::Impl {
  mfem::Mesh mesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::FiniteElementSpace> fes;
  int ndof = 0;
  bool axi = false;
  int deg_dom = 0, deg_bdr = 0;
  // coefficients
  std::map<int, double> kappa_attr;
  Eigen::VectorXd kappa_pts, f_pts;
  std::map<int, Eigen::VectorXd> flux;  // insulator flux per attr (aligned with points of that attr)
  std::vector<BElem> bdr;               // all boundary elements at the assembly degree
  std::vector<Electrode> el;
  // exterior
  bool dtn = false;
  int dtn_attr = 0, dtn_modes = 0;
  double dtn_Rb = 0, dtn_kappa = 0;
  // assembled
  bool assembled = false;
  std::vector<int> elem_pt0;   // first domain point of each element (assembly degree)
  Eigen::SparseMatrix<double> K, Kabs;
  Eigen::VectorXd Fv, Fb;
  // layout
  bool laid_out = false;
  std::vector<int> kind;      // 0 free, 1 dirichlet, 2 tie
  std::vector<int> uidx;      // unknown of free/tie dofs
  std::vector<double> dval;   // dirichlet value or -E_eq offset of tie dofs
  int nfree = 0, nunk = 0;
  // solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
  bool analyzed = false;
  long factor_nnz = 0;

  double weight_r(double x) const { return axi ? x : 1.0; }

  BElem make_belem(int be, int degree) {
    BElem B;
    B.be = be;
    B.attr = mesh.GetBdrAttribute(be);
    mfem::FaceElementTransformations* FT = mesh.GetBdrFaceTransformations(be);
    if (!FT) fail("boundary element " + std::to_string(be) + " has no adjacent element");
    B.elem = FT->Elem1No;
    mfem::Array<int> dofs;
    fes->GetElementDofs(B.elem, dofs);
    B.dofs.assign(dofs.begin(), dofs.end());
    const mfem::FiniteElement* fe = fes->GetFE(B.elem);
    const int nd = fe->GetDof();
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, degree);
    // centroid of the adjacent element (for the outward check)
    mfem::Array<int> ev;
    mesh.GetElementVertices(B.elem, ev);
    double cx = 0, cy = 0;
    for (int v : ev) {
      cx += mesh.GetVertex(v)[0] / ev.Size();
      cy += mesh.GetVertex(v)[1] / ev.Size();
    }
    mfem::Vector shape(nd), xv(2), nor(2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      FT->SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip = FT->GetElement1IntPoint();
      fe->CalcShape(eip, shape);
      FT->Transform(ip, xv);
      mfem::CalcOrtho(FT->Jacobian(), nor);
      const double len = std::hypot(nor(0), nor(1));
      double nx = nor(0) / len, ny = nor(1) / len;
      if ((xv(0) - cx) * nx + (xv(1) - cy) * ny < 0) {
        nx = -nx;
        ny = -ny;
      }
      B.pts.push_back({xv(0), xv(1), ip.weight * FT->Weight(), nx, ny});
      B.shape.emplace_back(shape.GetData(), shape.GetData() + nd);
    }
    return B;
  }

  std::vector<int> bdr_of(const std::vector<int>& attrs) const {
    std::vector<int> out;
    for (int i = 0; i < (int)bdr.size(); ++i)
      if (std::find(attrs.begin(), attrs.end(), bdr[i].attr) != attrs.end()) out.push_back(i);
    return out;
  }
  int npts_of(const std::vector<int>& bes) const {
    int n = 0;
    for (int i : bes) n += (int)bdr[i].pts.size();
    return n;
  }

  double kappa_at(int elem, int pt_global) const {
    if (kappa_pts.size() > 0 && pt_global >= 0) return kappa_pts(pt_global);
    const int a = mesh.GetAttribute(elem);
    auto it = kappa_attr.find(a);
    if (it == kappa_attr.end())
      fail("no conductivity for element attribute " + std::to_string(a) +
           ": call set_conductivity_by_attribute({attr: kappa}) or set_conductivity_at_points");
    return it->second;
  }

  // conductivity of element e: its attribute value, or its (constant) point values
  double elem_kappa(int e) const {
    if (kappa_pts.size() == 0) return kappa_at(e, -1);
    double lo = 1e300, hi = -1e300;
    for (int g = elem_pt0[e]; g < elem_pt0[e + 1]; ++g) lo = std::min(lo, kappa_pts(g)), hi = std::max(hi, kappa_pts(g));
    if (hi - lo > 1e-12 * hi)
      fail("set_exterior_halfspace: the conductivity varies next to the outer arc (element " + std::to_string(e) +
           "): the half-space map needs a constant far-field conductivity");
    return hi;
  }

  void assemble() {
    if (assembled) return;
    if (kappa_pts.size() == 0 && kappa_attr.empty()) fail("no conductivity set");
    std::vector<Eigen::Triplet<double>> trip;
    Fv = Eigen::VectorXd::Zero(ndof);
    Fb = Eigen::VectorXd::Zero(ndof);
    int gp = 0;
    elem_pt0.assign(mesh.GetNE() + 1, 0);
    for (int e = 0; e < mesh.GetNE(); ++e) {
      elem_pt0[e] = gp;
      const mfem::FiniteElement* fe = fes->GetFE(e);
      mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
      mfem::Array<int> dofs;
      fes->GetElementDofs(e, dofs);
      const int nd = fe->GetDof();
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), deg_dom);
      mfem::Vector shape(nd), xv(2);
      mfem::DenseMatrix dshape(nd, 2);
      Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(nd, nd);
      for (int q = 0; q < ir.GetNPoints(); ++q, ++gp) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        fe->CalcShape(ip, shape);
        fe->CalcPhysDShape(*T, dshape);
        T->Transform(ip, xv);
        const double wr = ip.weight * T->Weight() * weight_r(xv(0));
        const double kap = kappa_at(e, gp);
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b)
            Ke(a, b) += kap * wr * (dshape(a, 0) * dshape(b, 0) + dshape(a, 1) * dshape(b, 1));
        if (f_pts.size() > 0)
          for (int a = 0; a < nd; ++a) Fv(dofs[a]) += f_pts(gp) * shape(a) * wr;
      }
      for (int a = 0; a < nd; ++a)
        for (int b = 0; b < nd; ++b) trip.emplace_back(dofs[a], dofs[b], Ke(a, b));
    }
    elem_pt0[mesh.GetNE()] = gp;
    // insulator flux loads
    for (auto& [attr, q] : flux) {
      auto bes = bdr_of({attr});
      if (q.size() != npts_of(bes)) fail("insulator flux of attribute " + std::to_string(attr) + ": " +
                                         std::to_string(q.size()) + " values for " + std::to_string(npts_of(bes)) +
                                         " boundary points");
      int k = 0;
      for (int i : bes) {
        const BElem& B = bdr[i];
        for (size_t p = 0; p < B.pts.size(); ++p, ++k) {
          const double wr = B.pts[p].w * weight_r(B.pts[p].x);
          for (size_t a = 0; a < B.dofs.size(); ++a) Fb(B.dofs[a]) += q(k) * B.shape[p][a] * wr;
        }
      }
    }
    // half-space even-mode DtN: sum_n kappa Rb (2n+1)(4n+1) c_n(phi) c_n(v)
    if (dtn) {
      std::vector<Eigen::VectorXd> C(dtn_modes, Eigen::VectorXd::Zero(ndof));
      std::set<int> support;
      const auto dbes = bdr_of({dtn_attr});
      if (dbes.empty())
        fail("set_exterior_halfspace: no boundary element has attribute " + std::to_string(dtn_attr) +
             ": give the attribute of the outer arc rho = Rb");
      for (int i : dbes) {
        const BElem& B = bdr[i];
        // the even-mode map assumes the arc rho = Rb in z >= 0 and the conductivity of the adjacent electrolyte
        const double ke = elem_kappa(B.elem);
        if (!(std::abs(ke - dtn_kappa) <= 1e-12 * dtn_kappa))
          fail("set_exterior_halfspace: kappa = " + std::to_string(dtn_kappa) + " but the electrolyte next to the arc (element " +
               std::to_string(B.elem) + ") has kappa = " + std::to_string(ke) +
               ": the far field must have the conductivity of the adjacent electrolyte");
        mfem::Array<int> bv;
        mesh.GetBdrElementVertices(B.be, bv);
        for (int v : bv) {
          const double* X = mesh.GetVertex(v);
          const double rho = std::hypot(X[0], X[1]);
          if (!(std::abs(rho - dtn_Rb) <= 1e-9 * dtn_Rb) || X[1] < -1e-12 * dtn_Rb || X[0] < -1e-12 * dtn_Rb)
            fail("set_exterior_halfspace: boundary attribute " + std::to_string(dtn_attr) + " has a vertex at (r, z) = (" +
                 std::to_string(X[0]) + ", " + std::to_string(X[1]) + "), |x| = " + std::to_string(rho) + " != Rb = " +
                 std::to_string(dtn_Rb) + " (or z < 0): the half-space map needs the arc rho = Rb, z >= 0");
        }
        for (size_t p = 0; p < B.pts.size(); ++p) {
          const BPoint& P = B.pts[p];
          const double rho = std::hypot(P.x, P.y);
          const auto L = legendre(2 * (dtn_modes - 1), P.y / rho);
          for (int n = 0; n < dtn_modes; ++n)
            for (size_t a = 0; a < B.dofs.size(); ++a)
              C[n](B.dofs[a]) += P.w * (P.x / (dtn_Rb * dtn_Rb)) * L[2 * n] * B.shape[p][a];
        }
      }
      for (int d = 0; d < ndof; ++d)
        for (int n = 0; n < dtn_modes; ++n)
          if (C[n](d) != 0.0) {
            support.insert(d);
            break;
          }
      for (int n = 0; n < dtn_modes; ++n) {
        const double lam = dtn_kappa * dtn_Rb * (2.0 * n + 1.0) * (4.0 * n + 1.0);
        for (int a : support)
          for (int b : support) trip.emplace_back(a, b, lam * C[n](a) * C[n](b));
      }
    }
    K.resize(ndof, ndof);
    K.setFromTriplets(trip.begin(), trip.end());
    K.makeCompressed();
    Kabs = K.cwiseAbs();
    assembled = true;
  }

  void layout() {
    if (laid_out) return;
    if (el.empty()) fail("no electrode: add at least one electrode (add_electrode) before solving");
    bool grounded = dtn;
    for (auto& E : el) grounded = grounded || !E.spec.galvanostatic;
    if (!grounded)
      fail("no grounded electrode: every electrode is galvanostatic, so phi is defined only up to a constant. Make "
           "one electrode potentiostatic (the ground, e.g. phi_s = 0 on the counter electrode)");
    kind.assign(ndof, 0);
    uidx.assign(ndof, -1);
    dval.assign(ndof, 0.0);
    // primary electrodes: Dirichlet or tie
    for (auto& E : el) {
      if (!E.spec.primary) continue;
      for (int d : E.dofs) {
        kind[d] = E.spec.galvanostatic ? 2 : 1;
        dval[d] = E.spec.galvanostatic ? -E.spec.E_eq : E.spec.phi_s - E.spec.E_eq;
      }
    }
    nfree = 0;
    for (int d = 0; d < ndof; ++d)
      if (kind[d] == 0) uidx[d] = nfree++;
    nunk = nfree;
    for (auto& E : el) {
      if (!E.spec.galvanostatic) continue;
      E.unknown = nunk++;
      if (E.spec.primary)
        for (int d : E.dofs) uidx[d] = E.unknown;
    }
    laid_out = true;
  }

  const Electrode& electrode(int e, const char* where) const {
    if (e < 0 || e >= (int)el.size())
      fail(std::string(where) + ": electrode index " + std::to_string(e) + " out of range (" + std::to_string(el.size()) +
           " electrodes)");
    return el[e];
  }
  void require_attrs(const std::vector<int>& attrs, const char* where) const {
    for (int a : attrs)
      if (bdr_of({a}).empty())
        fail(std::string(where) + ": no boundary element has attribute " + std::to_string(a));
  }
  // initial guess 'primary': phi = phi_s - E_eq (mean over the electrode's reactions) on every electrode, Laplace with
  // the loads elsewhere; galvanostatic electrodes at their initial phi_s
  Eigen::VectorXd primary_guess(Eigen::VectorXd z) {
    std::vector<int> fix(ndof, 0);
    Eigen::VectorXd val = expand(z);
    for (const Electrode& E : el) {
      double eeq = E.spec.primary ? E.spec.E_eq : 0.0;
      if (!E.spec.primary) {
        for (const auto& r : E.spec.reactions) eeq += r.E_eq / E.spec.reactions.size();
      }
      const double ps = phi_s_of(E, z);
      for (int d : E.dofs) fix[d] = 1, val(d) = ps - eeq;
    }
    std::vector<int> id(ndof, -1);
    int n = 0;
    for (int d = 0; d < ndof; ++d)
      if (!fix[d]) id[d] = n++;
    if (n > 0) {
      std::vector<Eigen::Triplet<double>> t;
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
      for (int c = 0; c < K.outerSize(); ++c)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, c); it; ++it) {
          const int a = (int)it.row(), bb = (int)it.col();
          if (id[a] < 0) continue;
          if (id[bb] >= 0) t.emplace_back(id[a], id[bb], it.value());
          else b(id[a]) -= it.value() * val(bb);
        }
      for (int d = 0; d < ndof; ++d)
        if (id[d] >= 0) b(id[d]) += Fv(d) + Fb(d);
      Eigen::SparseMatrix<double> A(n, n);
      A.setFromTriplets(t.begin(), t.end());
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> s(A);
      if (s.info() != Eigen::Success) fail("initial='primary': the primary problem is singular");
      const Eigen::VectorXd x = s.solve(b);
      for (int d = 0; d < ndof; ++d)
        if (id[d] >= 0) val(d) = x(id[d]);
    }
    for (int d = 0; d < ndof; ++d)
      if (kind[d] == 0) z(uidx[d]) = val(d);
    return z;
  }

  double phi_s_of(const Electrode& E, const Eigen::VectorXd& z) const {
    return E.spec.galvanostatic ? z(E.unknown) : E.spec.phi_s;
  }

  Eigen::VectorXd expand(const Eigen::VectorXd& z) const {
    Eigen::VectorXd phi(ndof);
    for (int d = 0; d < ndof; ++d) {
      if (kind[d] == 0) phi(d) = z(uidx[d]);
      else if (kind[d] == 1) phi(d) = dval[d];
      else phi(d) = z(uidx[d]) + dval[d];
    }
    return phi;
  }
  Eigen::VectorXd expand_delta(const Eigen::VectorXd& dz) const {
    Eigen::VectorXd d(ndof);
    for (int i = 0; i < ndof; ++i) d(i) = kind[i] == 1 ? 0.0 : dz(uidx[i]);
    return d;
  }

  double I_pr(const Electrode& E) const { return axi ? E.spec.current / (2.0 * M_PI) : E.spec.current; }

  struct Eval {
    Eigen::VectorXd Rfull, r, babs;
    std::vector<double> G;            // per electrode: int (sum i + g) dA (per radian)
    std::vector<Eigen::Triplet<double>> M;  // kinetic Jacobian (full dofs)
    std::vector<Eigen::VectorXd> m;   // per electrode: int di v
    std::vector<double> dsum;
    bool clamped = false;
    double field = 0, constraint = 0, relres = 0;
  };

  // kinetic terms of electrode k at state (phi, phi_s)
  void kinetic(int k, const Eigen::VectorXd& phi, double phis, const NewtonOptions& o, Eigen::VectorXd* b,
               Eigen::VectorXd* babs, double* G, std::vector<Eigen::Triplet<double>>* M, Eigen::VectorXd* m,
               double* dsum, bool* clamped) const {
    const Electrode& E = el[k];
    const double f = kFaraday / (kGasConstant * o.temperature);
    int gp = 0;
    for (int i : E.belems) {
      const BElem& B = bdr[i];
      const int nd = (int)B.dofs.size();
      for (size_t p = 0; p < B.pts.size(); ++p, ++gp) {
        const auto& sh = B.shape[p];
        double ph = 0;
        for (int a = 0; a < nd; ++a) ph += sh[a] * phi(B.dofs[a]);
        double cur = 0, dcur = 0, cabs = 0;
        for (const Reaction& r : E.spec.reactions) {
          double di = 0;
          const double ij = reaction_current(r, phis - ph - r.E_eq, f, o.clip, &di, clamped);
          cur += ij;
          dcur += di;
          cabs += std::abs(ij);
        }
        if (!E.g.empty()) {
          cur += E.g[gp];
          cabs += std::abs(E.g[gp]);
        }
        const double wr = B.pts[p].w * weight_r(B.pts[p].x);
        if (G) *G += cur * wr;
        if (dsum) *dsum += dcur * wr;
        for (int a = 0; a < nd; ++a) {
          if (b) (*b)(B.dofs[a]) += cur * sh[a] * wr;
          if (babs) (*babs)(B.dofs[a]) += cabs * sh[a] * wr;
          if (m) (*m)(B.dofs[a]) += dcur * sh[a] * wr;
          if (M)
            for (int c = 0; c < nd; ++c) M->emplace_back(B.dofs[a], B.dofs[c], dcur * sh[a] * sh[c] * wr);
        }
      }
    }
  }

  Eval evaluate(const Eigen::VectorXd& z, const NewtonOptions& o, bool jac, int exclude = -1) {
    Eval ev;
    const Eigen::VectorXd phi = expand(z);
    ev.Rfull = K * phi - Fv - Fb;
    ev.babs = Eigen::VectorXd::Zero(ndof);
    ev.G.assign(el.size(), 0.0);
    ev.m.resize(el.size());
    ev.dsum.assign(el.size(), 0.0);
    for (int k = 0; k < (int)el.size(); ++k) {
      if (el[k].spec.primary) continue;
      Eigen::VectorXd b = Eigen::VectorXd::Zero(ndof);
      ev.m[k] = Eigen::VectorXd::Zero(ndof);
      kinetic(k, phi, phi_s_of(el[k], z), o, &b, &ev.babs, &ev.G[k], jac ? &ev.M : nullptr, &ev.m[k], &ev.dsum[k],
              &ev.clamped);
      if (k != exclude) ev.Rfull -= b;
    }
    ev.r = Eigen::VectorXd::Zero(nunk);
    for (int d = 0; d < ndof; ++d)
      if (kind[d] != 1) ev.r(uidx[d]) += ev.Rfull(d);
    double cons = 0;
    for (int k = 0; k < (int)el.size(); ++k) {
      const Electrode& E = el[k];
      if (!E.spec.galvanostatic) continue;
      const double Ip = I_pr(E);
      if (E.spec.primary) ev.r(E.unknown) -= Ip;   // tie row: reaction current - I
      else ev.r(E.unknown) = ev.G[k] - Ip;           // bordered row
      cons = std::max(cons, std::abs(ev.r(E.unknown)) / std::max(std::abs(Ip), 1e-300));
    }
    // field part of relres over the free dofs
    const Eigen::VectorXd kp = Kabs * phi.cwiseAbs();
    double num = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int d = 0; d < ndof; ++d) {
      if (kind[d] != 0) continue;
      num += ev.Rfull(d) * ev.Rfull(d);
      d1 += kp(d) * kp(d);
      const double l = std::abs(Fv(d)) + std::abs(Fb(d));
      d2 += l * l;
      d3 += ev.babs(d) * ev.babs(d);
    }
    const double den = std::sqrt(d1) + std::sqrt(d2) + std::sqrt(d3);
    ev.field = den > 0 ? std::sqrt(num) / den : std::sqrt(num);
    ev.constraint = cons;
    ev.relres = std::max(ev.field, ev.constraint);
    return ev;
  }

  Eigen::SparseMatrix<double> zjacobian(const Eval& ev) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(K.nonZeros() + ev.M.size());
    for (int c = 0; c < K.outerSize(); ++c)
      for (Eigen::SparseMatrix<double>::InnerIterator it(K, c); it; ++it) {
        const int a = (int)it.row(), b = (int)it.col();
        if (kind[a] != 1 && kind[b] != 1) t.emplace_back(uidx[a], uidx[b], it.value());
      }
    for (const auto& m : ev.M) {
      const int a = m.row(), b = m.col();
      if (kind[a] != 1 && kind[b] != 1) t.emplace_back(uidx[a], uidx[b], m.value());
    }
    for (int k = 0; k < (int)el.size(); ++k) {
      const Electrode& E = el[k];
      if (E.spec.primary || !E.spec.galvanostatic) continue;
      const int s = E.unknown;
      for (int d = 0; d < ndof; ++d) {
        const double v = ev.m[k](d);
        if (v == 0.0 || kind[d] == 1) continue;
        t.emplace_back(uidx[d], s, -v);
        t.emplace_back(s, uidx[d], -v);
      }
      t.emplace_back(s, s, ev.dsum[k]);
    }
    Eigen::SparseMatrix<double> J(nunk, nunk);
    J.setFromTriplets(t.begin(), t.end());
    J.makeCompressed();
    return J;
  }

  bool linsolve(const Eigen::SparseMatrix<double>& J, const Eigen::VectorXd& b, Eigen::VectorXd& x, int refine) {
    if (!analyzed) {
      ldlt.analyzePattern(J);
      analyzed = true;
    }
    ldlt.factorize(J);
    if (ldlt.info() != Eigen::Success) return false;
    factor_nnz = ldlt.matrixL().nestedExpression().nonZeros();
    x = ldlt.solve(b);
    for (int i = 0; i < refine; ++i) x += ldlt.solve(Eigen::VectorXd(b - J * x));
    return x.allFinite();
  }

  double max_deta(const Eigen::VectorXd& dz) const {
    const Eigen::VectorXd dphi = expand_delta(dz);
    double mx = 0;
    for (const Electrode& E : el) {
      if (E.spec.primary) continue;
      const double dps = E.spec.galvanostatic ? dz(E.unknown) : 0.0;
      for (int i : E.belems) {
        const BElem& B = bdr[i];
        for (size_t p = 0; p < B.pts.size(); ++p) {
          double d = 0;
          for (size_t a = 0; a < B.dofs.size(); ++a) d += B.shape[p][a] * dphi(B.dofs[a]);
          mx = std::max(mx, std::abs(dps - d));
        }
      }
    }
    return mx;
  }
};

// ------------------------------------------------------------------------------------------------ construction
CurrentDistribution::CurrentDistribution(const MeshInput& in, int order, bool axisymmetric, int domain_degree,
                                         int boundary_degree)
    : p_(new Impl), order_(order), deg_dom_(domain_degree), deg_bdr_(boundary_degree), axi_(axisymmetric) {
  if (order != 1 && order != 2) fail("element order " + std::to_string(order) + " not supported: use 1 or 2");
  if (domain_degree < 2 * order || boundary_degree < 2 * order)
    fail("quadrature degrees (" + std::to_string(domain_degree) + ", " + std::to_string(boundary_degree) +
         ") below 2 x order = " + std::to_string(2 * order) + ": the mass and stiffness terms would be under-integrated "
         "(the default rule is 2k + 4)");
  const int nv = (int)in.vertices.size() / 2, ne = (int)in.triangles.size() / 3, nb = (int)in.boundary.size() / 2;
  if ((int)in.element_attr.size() != ne) fail("element_attr must have one entry per triangle");
  if ((int)in.boundary_attr.size() != nb) fail("boundary_attr must have one entry per boundary segment");
  if (nv < 3 || ne < 1 || nb < 1) fail("empty mesh");
  Impl& P = *p_;
  P.axi = axisymmetric;
  P.deg_dom = domain_degree;
  P.deg_bdr = boundary_degree;
  P.mesh = mfem::Mesh(2, nv, ne, nb, 2);
  for (int v = 0; v < nv; ++v) P.mesh.AddVertex(in.vertices[2 * v], in.vertices[2 * v + 1]);
  for (int e = 0; e < ne; ++e) {
    if (in.element_attr[e] < 1) fail("element attributes must be >= 1");
    P.mesh.AddTriangle(in.triangles[3 * e], in.triangles[3 * e + 1], in.triangles[3 * e + 2], in.element_attr[e]);
  }
  for (int b = 0; b < nb; ++b) {
    if (in.boundary_attr[b] < 1) fail("boundary attributes must be >= 1");
    P.mesh.AddBdrSegment(in.boundary[2 * b], in.boundary[2 * b + 1], in.boundary_attr[b]);
  }
  P.mesh.FinalizeTriMesh(1, 0, true);
  if (axisymmetric)
    for (int v = 0; v < nv; ++v)
      if (in.vertices[2 * v] < -1e-14) fail("axisymmetric mesh with a vertex at r < 0 (vertex " + std::to_string(v) + ")");
  const int nmid = (int)in.edge_node_vertices.size() / 2;
  if (nmid > 0) {
    if (order != 2) fail("curved edge nodes need order 2 (isoparametric P2)");
    P.mesh.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
    std::map<std::pair<int, int>, int> key;
    for (int i = 0; i < nmid; ++i) {
      int a = in.edge_node_vertices[2 * i], b = in.edge_node_vertices[2 * i + 1];
      key[{std::min(a, b), std::max(a, b)}] = i;
    }
    mfem::GridFunction* nodes = P.mesh.GetNodes();
    const mfem::FiniteElementSpace* nf = nodes->FESpace();
    int used = 0;
    for (int ed = 0; ed < P.mesh.GetNEdges(); ++ed) {
      mfem::Array<int> v;
      P.mesh.GetEdgeVertices(ed, v);
      auto it = key.find({std::min(v[0], v[1]), std::max(v[0], v[1])});
      if (it == key.end()) continue;
      mfem::Array<int> d;
      nf->GetEdgeInteriorDofs(ed, d);
      if (d.Size() != 1) fail("unexpected number of P2 edge nodes");
      for (int c = 0; c < 2; ++c) (*nodes)(nf->DofToVDof(d[0], c)) = in.edge_node_xy[2 * it->second + c];
      ++used;
    }
    if (used != nmid) fail(std::to_string(nmid - used) + " curved edge nodes name a vertex pair that is not an edge");
  }
  P.fec.reset(new mfem::H1_FECollection(order, 2));
  P.fes.reset(new mfem::FiniteElementSpace(&P.mesh, P.fec.get()));
  P.ndof = P.fes->GetVSize();
  for (int be = 0; be < P.mesh.GetNBE(); ++be) P.bdr.push_back(P.make_belem(be, boundary_degree));
}

CurrentDistribution::~CurrentDistribution() = default;

Eigen::MatrixXd CurrentDistribution::domain_points() const {
  const Impl& P = *p_;
  std::vector<double> xs;
  for (int e = 0; e < P.mesh.GetNE(); ++e) {
    mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(P.mesh).GetElementTransformation(e);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(P.mesh.GetElementBaseGeometry(e), deg_dom_);
    mfem::Vector xv(2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      T->Transform(ir.IntPoint(q), xv);
      xs.push_back(xv(0));
      xs.push_back(xv(1));
    }
  }
  Eigen::MatrixXd X(xs.size() / 2, 2);
  for (Eigen::Index i = 0; i < X.rows(); ++i) X(i, 0) = xs[2 * i], X(i, 1) = xs[2 * i + 1];
  return X;
}

Eigen::MatrixXd CurrentDistribution::boundary_points(const std::vector<int>& attrs) const {
  const Impl& P = *p_;
  auto bes = P.bdr_of(attrs);
  Eigen::MatrixXd X(P.npts_of(bes), 2);
  int k = 0;
  for (int i : bes)
    for (const BPoint& q : P.bdr[i].pts) X(k, 0) = q.x, X(k, 1) = q.y, ++k;
  return X;
}

void CurrentDistribution::set_conductivity_by_attribute(const std::map<int, double>& kappa) {
  for (auto& [a, k] : kappa)
    if (!(k > 0)) fail("conductivity of attribute " + std::to_string(a) + " must be > 0");
  p_->kappa_attr = kappa;
  p_->kappa_pts.resize(0);
  p_->assembled = false;
}
void CurrentDistribution::set_conductivity_at_points(const Eigen::VectorXd& kappa) {
  if (kappa.size() != domain_points().rows()) fail("set_conductivity_at_points: size differs from domain_points()");
  if (!(kappa.minCoeff() > 0)) fail("set_conductivity_at_points: conductivity must be > 0 at every point");
  p_->kappa_pts = kappa;
  p_->assembled = false;
}
void CurrentDistribution::set_source_at_points(const Eigen::VectorXd& f) {
  if (f.size() != domain_points().rows()) fail("set_source_at_points: size differs from domain_points()");
  p_->f_pts = f;
  p_->assembled = false;
}
void CurrentDistribution::set_insulator_flux(int attr, const Eigen::VectorXd& q) {
  const auto bes = p_->bdr_of({attr});
  if (bes.empty())
    fail("set_insulator_flux: no boundary element has attribute " + std::to_string(attr) +
         ": give an attribute of the mesh boundary");
  if (q.size() != p_->npts_of(bes))
    fail("set_insulator_flux(" + std::to_string(attr) + "): " + std::to_string(q.size()) + " values, boundary_points([" +
         std::to_string(attr) + "]) has " + std::to_string(p_->npts_of(bes)) + ": evaluate q at those points");
  for (const auto& E : p_->el)
    if (std::find(E.spec.attrs.begin(), E.spec.attrs.end(), attr) != E.spec.attrs.end())
      fail("set_insulator_flux: attribute " + std::to_string(attr) + " belongs to electrode '" + E.spec.name +
           "': an insulator flux goes on an insulating boundary");
  if (!q.allFinite()) fail("set_insulator_flux: non-finite values");
  p_->flux[attr] = q;
  p_->assembled = false;
}

int CurrentDistribution::add_electrode(const ElectrodeSpec& s) {
  Impl& P = *p_;
  if (P.laid_out) fail("add_electrode after the first solve/state call: add every electrode before solving");
  const std::string who = "electrode '" + s.name + "': ";
  if (s.attrs.empty()) fail(who + "no boundary attributes given");
  for (const auto& O : P.el)
    for (int a : s.attrs)
      if (std::find(O.spec.attrs.begin(), O.spec.attrs.end(), a) != O.spec.attrs.end())
        fail("boundary attribute " + std::to_string(a) + " belongs to electrodes '" + O.spec.name + "' and '" + s.name +
             "': give each electrode its own attributes");
  for (int a : s.attrs)
    if (P.flux.count(a)) fail(who + "attribute " + std::to_string(a) + " already carries an insulator flux");
  if (s.primary && !s.reactions.empty())
    fail(who + "a primary electrode has no reactions (phi = phi_s - E_eq); drop the reactions or primary=true");
  for (size_t j = 0; j < s.reactions.size(); ++j) {
    const Reaction& r = s.reactions[j];
    const std::string rw = who + "reaction " + std::to_string(j) + ": ";
    if (!(r.i0 > 0) || !std::isfinite(r.i0)) fail(rw + "i0 must be > 0 (A/m^2), got " + std::to_string(r.i0));
    if (!std::isfinite(r.E_eq)) fail(rw + "E_eq must be finite");
    if (r.kind == Reaction::Kind::BV || r.kind == Reaction::Kind::Linear) {
      if (!(r.alpha_a >= 0 && r.alpha_c >= 0 && r.alpha_a + r.alpha_c > 0))
        fail(rw + "transfer coefficients must be alpha_a >= 0, alpha_c >= 0, alpha_a + alpha_c > 0 (got " +
             std::to_string(r.alpha_a) + ", " + std::to_string(r.alpha_c) + "): the solver needs di/deta > 0");
    } else if (r.kind == Reaction::Kind::TafelAnodic && !(r.A > 0)) {
      fail(rw + "anodic Tafel (tafel_a) needs a slope A > 0 V/decade, got " + std::to_string(r.A));
    } else if (r.kind == Reaction::Kind::TafelCathodic && !(r.A < 0)) {
      fail(rw + "cathodic Tafel (tafel_c) needs a slope A < 0 V/decade, got " + std::to_string(r.A));
    }
  }
  if (!std::isfinite(s.phi_s) || !std::isfinite(s.current)) fail(who + "phi_s and current must be finite");
  Electrode E;
  E.spec = s;
  E.belems = P.bdr_of(s.attrs);
  if (E.belems.empty()) fail("electrode '" + s.name + "': no boundary element carries its attributes");
  if (!s.primary && s.reactions.empty()) fail(who + "a kinetic electrode needs reactions (or primary=true)");
  if (!s.g.empty() && (int)s.g.size() != P.npts_of(E.belems))
    fail("electrode '" + s.name + "': g has " + std::to_string(s.g.size()) + " values, boundary_points has " +
         std::to_string(P.npts_of(E.belems)));
  if (s.galvanostatic && s.current == 0.0)
    fail(who + "galvanostatic with zero current: the relative constraint residual needs I != 0 (use a potentiostatic "
               "electrode, or a mixed-potential set-up, for open circuit)");
  E.g = s.g;
  E.dofs = attr_dofs(s.attrs);
  P.el.push_back(E);
  return (int)P.el.size() - 1;
}

void CurrentDistribution::set_exterior_halfspace(int attr, double Rb, int modes, double kappa) {
  if (!axi_) fail("the half-space DtN is axisymmetric only");
  if (modes < 1 || !(Rb > 0) || !(kappa > 0))
    fail("set_exterior_halfspace: need modes >= 1, Rb > 0, kappa > 0 (got " + std::to_string(modes) + ", " +
         std::to_string(Rb) + ", " + std::to_string(kappa) + ")");
  if (p_->bdr_of({attr}).empty())
    fail("set_exterior_halfspace: no boundary element has attribute " + std::to_string(attr));
  Impl& P = *p_;
  P.dtn = true;
  P.dtn_attr = attr;
  P.dtn_Rb = Rb;
  P.dtn_modes = modes;
  P.dtn_kappa = kappa;
  P.assembled = false;
}

std::vector<int> CurrentDistribution::attr_dofs(const std::vector<int>& attrs) const {
  const Impl& P = *p_;
  P.require_attrs(attrs, "attr_dofs");
  std::set<int> s;
  for (int be = 0; be < P.mesh.GetNBE(); ++be) {
    if (std::find(attrs.begin(), attrs.end(), P.mesh.GetBdrAttribute(be)) == attrs.end()) continue;
    mfem::Array<int> d;
    P.fes->GetBdrElementDofs(be, d);
    for (int x : d) s.insert(x);
  }
  return {s.begin(), s.end()};
}
std::vector<int> CurrentDistribution::electrode_dofs(int e) const { return p_->electrode(e, "electrode_dofs").dofs; }
std::vector<int> CurrentDistribution::electrode_attrs(int e) const {
  return p_->electrode(e, "electrode_attrs").spec.attrs;
}
void CurrentDistribution::set_control(int e, double value) {
  Impl& P = *p_;
  P.electrode(e, "set_control");
  if (!std::isfinite(value)) fail("set_control: value must be finite");
  Electrode& E = P.el[e];
  if (E.spec.galvanostatic) {
    if (value == 0.0) fail("set_control: electrode '" + E.spec.name + "' is galvanostatic: its current must be != 0");
    E.spec.current = value;
  } else {
    E.spec.phi_s = value;
    if (E.spec.primary && P.laid_out)
      for (int d : E.dofs) P.dval[d] = value - E.spec.E_eq;
  }
}
void CurrentDistribution::set_temperature(double T) {
  if (!(T > 0) || !std::isfinite(T)) fail("set_temperature: T must be > 0 K, got " + std::to_string(T));
  temperature_ = T;
  eval_opt_.temperature = T;
}
std::vector<std::string> CurrentDistribution::electrode_names() const {
  std::vector<std::string> n;
  for (auto& E : p_->el) n.push_back(E.spec.name);
  return n;
}

// ------------------------------------------------------------------------------------------------ state
int CurrentDistribution::num_unknowns() {
  p_->layout();
  return p_->nunk;
}
Eigen::VectorXd CurrentDistribution::initial_state() {
  Impl& P = *p_;
  P.layout();
  Eigen::VectorXd z = Eigen::VectorXd::Zero(P.nunk);  // phi = 0
  for (auto& E : P.el)
    if (E.spec.galvanostatic) z(E.unknown) = E.spec.phi_s;   // a tie dof then holds phi_s - E_eq
  return z;
}
void CurrentDistribution::check_state(const Eigen::VectorXd& z, const char* where) {
  p_->assemble();
  p_->layout();
  if (z.size() != p_->nunk)
    fail(std::string(where) + ": z has " + std::to_string(z.size()) + " entries, the problem has " +
         std::to_string(p_->nunk) + " unknowns (num_unknowns(); start from state() or initial_state())");
  if (!z.allFinite()) fail(std::string(where) + ": z has non-finite entries");
}
void CurrentDistribution::set_state(const Eigen::VectorXd& z) {
  check_state(z, "set_state");
  z_ = z;
}
Eigen::VectorXd CurrentDistribution::residual(const Eigen::VectorXd& z) {
  check_state(z, "residual");
  return p_->evaluate(z, eval_opt_, false).r;
}
CurrentDistribution::SpMat CurrentDistribution::jacobian(const Eigen::VectorXd& z) {
  check_state(z, "jacobian");
  return p_->zjacobian(p_->evaluate(z, eval_opt_, true));
}
double CurrentDistribution::relres(const Eigen::VectorXd& z) {
  check_state(z, "relres");
  return p_->evaluate(z, eval_opt_, false).relres;
}
Eigen::VectorXd CurrentDistribution::newton_update(const Eigen::VectorXd& z) {
  check_state(z, "newton_update");
  Impl& P = *p_;
  auto ev = P.evaluate(z, eval_opt_, true);
  Eigen::VectorXd dz;
  if (!P.linsolve(P.zjacobian(ev), -ev.r, dz, 1)) fail("newton_update: the linear solve failed");
  return z + dz;
}
Eigen::VectorXd CurrentDistribution::phi_dofs() const {
  if (z_.size() != p_->nunk || !p_->laid_out) fail("no state: call solve() or set_state() first");
  return p_->expand(z_);
}
std::vector<double> CurrentDistribution::phi_s() const {
  if (!p_->laid_out || z_.size() != p_->nunk) fail("phi_s: no state: call solve() or set_state() first");
  std::vector<double> out;
  for (auto& E : p_->el) out.push_back(E.spec.galvanostatic ? z_(E.unknown) : E.spec.phi_s);
  return out;
}

// ------------------------------------------------------------------------------------------------ Newton
SolveReport CurrentDistribution::solve(const NewtonOptions& oin) {
  const auto t0 = std::chrono::steady_clock::now();
  Impl& P = *p_;
  // one source of truth for T: the problem's temperature (set_temperature, default 298.15 K); a solve that names a
  // temperature (oin.temperature != 0) sets the problem's temperature first
  NewtonOptions o = oin;
  if (o.temperature == 0.0) {
    o.temperature = temperature_;
  } else {
    if (!(o.temperature > 0) || !std::isfinite(o.temperature))
      fail("solve: temperature must be > 0 K (or 0: the problem's temperature), got " + std::to_string(o.temperature));
    temperature_ = o.temperature;
  }
  if (o.max_iter < 0 || !(o.stag >= 0) || !(o.gate > 0) || !(o.ls_min > 0 && o.ls_min <= 1) || !(o.clip > 0))
    fail("solve: invalid Newton options (need max_iter >= 0, stag >= 0, gate > 0, 0 < ls_min <= 1, clip > 0)");
  P.assemble();
  P.layout();
  eval_opt_ = o;   // every later evaluation (residual, jacobian, relres, newton_update, residual_full) uses these
  SolveReport R;
  R.unknowns = P.nunk;
  R.initial = o.initial;
  Eigen::VectorXd z;
  if (o.initial == "zero") {
    z = initial_state();
  } else if (o.initial == "state") {
    if (z_.size() != P.nunk) fail("solve(initial='state'): no state to start from: call set_state(z) first");
    z = z_;
  } else if (o.initial == "primary") {
    z = P.primary_guess(initial_state());
  } else {
    fail("solve: initial must be 'zero' (phi = 0, the default start), 'state' (warm start from set_state) or "
         "'primary' (the primary distribution, eta = 0 on every electrode); got '" + o.initial + "'");
  }
  auto ev = P.evaluate(z, o, false);
  R.residual_history.push_back(ev.relres);
  const double lim = o.limiter_rt_f > 0 ? o.limiter_rt_f * kGasConstant * o.temperature / kFaraday : 0.0;
  R.stop = "max_iter";
  for (int it = 0;; ++it) {
    const auto& h = R.residual_history;
    if (h.back() == 0.0) {
      R.stop = "zero";
      break;
    }
    if (h.size() >= 2 && h.back() <= o.stag && h.back() > 0.5 * h[h.size() - 2]) {
      R.stop = "stagnation";
      break;
    }
    if (it >= o.max_iter) break;
    auto evj = P.evaluate(z, o, true);
    Eigen::VectorXd dz;
    if (!P.linsolve(P.zjacobian(evj), -evj.r, dz, o.refinement_steps)) {
      R.stop = "linear_solver_failed";
      break;
    }
    double t = 1.0;
    const double md = lim > 0 ? P.max_deta(dz) : 0.0;
    Eigen::VectorXd zn;
    double rn;
    if (lim > 0 && md > lim) {
      t = lim / md;
      zn = z + t * dz;
      rn = P.evaluate(zn, o, false).relres;
      ++R.limited_steps;
    } else {
      // merit: the Euclidean norm of the z-space residual (field rows, tie and bordered rows)
      const double r0 = evj.r.norm();
      while (true) {
        zn = z + t * dz;
        const auto evn = P.evaluate(zn, o, false);
        rn = evn.relres;
        if (evn.r.norm() < (1.0 - 1e-4 * t) * r0 || t <= o.ls_min) break;
        t *= 0.5;
        ++R.line_search_halvings;
      }
    }
    z = zn;
    R.step_lengths.push_back(t);
    R.residual_history.push_back(rn);
    R.iterations = it + 1;
  }
  z_ = z;
  auto fin = P.evaluate(z, o, false);
  R.relative_residual = fin.relres;
  R.field_residual = fin.field;
  R.constraint_residual = fin.constraint;
  R.clamp_active = fin.clamped;
  // a kinetic current below the round-off of the field residual is not resolved by any formulation in phi
  {
    const Eigen::VectorXd phi = P.expand(z);
    const Eigen::VectorXd kp = P.Kabs * phi.cwiseAbs();
    for (int k = 0; k < (int)P.el.size(); ++k) {
      const Electrode& E = P.el[k];
      if (E.spec.primary) continue;
      double sc = 0, partial = 0;   // partial: int sum_j |i_j| (+|g|) over the electrode (partition of unity)
      for (int d : E.dofs) sc += kp(d), partial += fin.babs(d);
      if (partial < 1e3 * std::numeric_limits<double>::epsilon() * sc)
        R.unresolved_currents.push_back(E.spec.name);
    }
  }
  R.converged = std::isfinite(fin.relres) && fin.relres <= o.gate && !fin.clamped;
  R.factor_nonzeros = P.factor_nnz;
  R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report_ = R;
  return R;
}

// ------------------------------------------------------------------------------------------------ raw data
DomainQuadrature CurrentDistribution::domain_quadrature(int degree) const {
  const Impl& P = *p_;
  if (degree < 1) fail("domain_quadrature: degree must be >= 1");
  const Eigen::VectorXd phi = phi_dofs();
  std::vector<double> X, W, F, G;
  std::vector<int> E;
  for (int e = 0; e < P.mesh.GetNE(); ++e) {
    const mfem::FiniteElement* fe = P.fes->GetFE(e);
    mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(P.mesh).GetElementTransformation(e);
    mfem::Array<int> dofs;
    P.fes->GetElementDofs(e, dofs);
    const int nd = fe->GetDof();
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), degree);
    mfem::Vector shape(nd), xv(2);
    mfem::DenseMatrix ds(nd, 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*T, ds);
      T->Transform(ip, xv);
      double v = 0, gx = 0, gy = 0;
      for (int a = 0; a < nd; ++a) {
        v += shape(a) * phi(dofs[a]);
        gx += ds(a, 0) * phi(dofs[a]);
        gy += ds(a, 1) * phi(dofs[a]);
      }
      X.push_back(xv(0)), X.push_back(xv(1));
      W.push_back(ip.weight * T->Weight());
      F.push_back(v);
      G.push_back(gx), G.push_back(gy);
      E.push_back(e);
    }
  }
  DomainQuadrature D;
  const int n = (int)W.size();
  D.x.resize(n, 2), D.grad.resize(n, 2), D.w.resize(n), D.phi.resize(n), D.element.resize(n);
  for (int i = 0; i < n; ++i) {
    D.x(i, 0) = X[2 * i], D.x(i, 1) = X[2 * i + 1];
    D.grad(i, 0) = G[2 * i], D.grad(i, 1) = G[2 * i + 1];
    D.w(i) = W[i], D.phi(i) = F[i], D.element(i) = E[i];
  }
  return D;
}

BoundaryQuadrature CurrentDistribution::boundary_quadrature(const std::vector<int>& attrs, int degree) const {
  Impl& P = *p_;
  P.require_attrs(attrs, "boundary_quadrature");
  if (degree < 1) fail("boundary_quadrature: degree must be >= 1");
  const Eigen::VectorXd phi = phi_dofs();
  std::vector<BElem> bes;
  for (int be = 0; be < P.mesh.GetNBE(); ++be)
    if (std::find(attrs.begin(), attrs.end(), P.mesh.GetBdrAttribute(be)) != attrs.end())
      bes.push_back(P.make_belem(be, degree));
  int n = 0;
  for (auto& B : bes) n += (int)B.pts.size();
  BoundaryQuadrature Q;
  Q.x.resize(n, 2), Q.grad.resize(n, 2), Q.normal.resize(n, 2), Q.w.resize(n), Q.phi.resize(n), Q.bdr_element.resize(n);
  int k = 0;
  for (auto& B : bes) {
    mfem::Array<int> dofs;
    P.fes->GetElementDofs(B.elem, dofs);
    const mfem::FiniteElement* fe = P.fes->GetFE(B.elem);
    const int nd = fe->GetDof();
    mfem::FaceElementTransformations* FT = P.mesh.GetBdrFaceTransformations(B.be);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, degree);
    mfem::DenseMatrix ds(nd, 2);
    for (int q = 0; q < ir.GetNPoints(); ++q, ++k) {
      FT->SetAllIntPoints(&ir.IntPoint(q));
      fe->CalcPhysDShape(*FT->Elem1, ds);
      double v = 0, gx = 0, gy = 0;
      for (int a = 0; a < nd; ++a) {
        v += B.shape[q][a] * phi(dofs[a]);
        gx += ds(a, 0) * phi(dofs[a]);
        gy += ds(a, 1) * phi(dofs[a]);
      }
      const BPoint& p = B.pts[q];
      Q.x(k, 0) = p.x, Q.x(k, 1) = p.y, Q.w(k) = p.w, Q.phi(k) = v;
      Q.grad(k, 0) = gx, Q.grad(k, 1) = gy, Q.normal(k, 0) = p.nx, Q.normal(k, 1) = p.ny;
      Q.bdr_element(k) = B.be;
    }
  }
  return Q;
}

Eigen::VectorXd CurrentDistribution::residual_full(int exclude) const {
  Impl& P = *p_;
  if (!P.assembled || !P.laid_out || z_.size() != P.nunk) fail("residual_full: no state: call solve() first");
  if (exclude != -1) P.electrode(exclude, "residual_full");
  return P.evaluate(z_, eval_opt_, false, exclude).Rfull;
}

CurrentDistribution::SpMat CurrentDistribution::boundary_mass(const std::vector<int>& attrs) const {
  const Impl& P = *p_;
  P.require_attrs(attrs, "boundary_mass");
  std::vector<Eigen::Triplet<double>> t;
  for (int i : P.bdr_of(attrs)) {
    const BElem& B = P.bdr[i];
    for (size_t p = 0; p < B.pts.size(); ++p) {
      const double wr = B.pts[p].w * P.weight_r(B.pts[p].x);
      for (size_t a = 0; a < B.dofs.size(); ++a)
        for (size_t c = 0; c < B.dofs.size(); ++c)
          t.emplace_back(B.dofs[a], B.dofs[c], B.shape[p][a] * B.shape[p][c] * wr);
    }
  }
  SpMat M(P.ndof, P.ndof);
  M.setFromTriplets(t.begin(), t.end());
  return M;
}

Eigen::MatrixXd CurrentDistribution::mesh_vertices() const {
  const Impl& P = *p_;
  Eigen::MatrixXd V(P.mesh.GetNV(), 2);
  for (int v = 0; v < P.mesh.GetNV(); ++v) V(v, 0) = P.mesh.GetVertex(v)[0], V(v, 1) = P.mesh.GetVertex(v)[1];
  return V;
}
Eigen::MatrixXd CurrentDistribution::dof_coordinates() const {
  const Impl& P = *p_;
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(P.ndof, 2);
  for (int e = 0; e < P.mesh.GetNE(); ++e) {
    const mfem::FiniteElement* fe = P.fes->GetFE(e);
    mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(P.mesh).GetElementTransformation(e);
    mfem::Array<int> dofs;
    P.fes->GetElementDofs(e, dofs);
    const mfem::IntegrationRule& nodes = fe->GetNodes();
    mfem::Vector xv(2);
    for (int a = 0; a < fe->GetDof(); ++a) {
      T->Transform(nodes.IntPoint(a), xv);
      X(dofs[a], 0) = xv(0), X(dofs[a], 1) = xv(1);
    }
  }
  return X;
}

Eigen::MatrixXi CurrentDistribution::mesh_elements() const {
  const Impl& P = *p_;
  Eigen::MatrixXi E(P.mesh.GetNE(), 3);
  for (int e = 0; e < P.mesh.GetNE(); ++e) {
    mfem::Array<int> v;
    P.mesh.GetElementVertices(e, v);
    for (int c = 0; c < 3; ++c) E(e, c) = v[c];
  }
  return E;
}
int CurrentDistribution::num_elements() const { return p_->mesh.GetNE(); }
int CurrentDistribution::num_dofs() const { return p_->ndof; }
double CurrentDistribution::h_max() const {
  const Impl& P = *p_;
  double h = 0;
  for (int e = 0; e < P.mesh.GetNE(); ++e) {
    mfem::Array<int> v;
    P.mesh.GetElementVertices(e, v);
    for (int a = 0; a < 3; ++a) {
      const double* x = P.mesh.GetVertex(v[a]);
      const double* y = P.mesh.GetVertex(v[(a + 1) % 3]);
      h = std::max(h, std::hypot(x[0] - y[0], x[1] - y[1]));
    }
  }
  return h;
}
double CurrentDistribution::area() const {
  const Impl& P = *p_;
  double s = 0;
  for (int e = 0; e < P.mesh.GetNE(); ++e) s += const_cast<mfem::Mesh&>(P.mesh).GetElementVolume(e);
  return s;
}
std::map<std::string, double> CurrentDistribution::diagnostics() const {
  return {{"dofs_field", (double)p_->ndof},
          {"unknowns", (double)p_->nunk},
          {"n_elements", (double)p_->mesh.GetNE()},
          {"factor_nonzeros", (double)p_->factor_nnz},
          {"domain_degree", (double)deg_dom_},
          {"boundary_degree", (double)deg_bdr_}};
}

}  // namespace axf::chem
