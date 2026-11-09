// Scalar-transport operator (beta): see cpp/include/axionfem/transport/scalar_transport.hpp.
#include "axionfem/transport/scalar_transport.hpp"

#include <algorithm>
#include <cmath>
#include <set>

#include "axionfem/core.hpp"

namespace axf::transport {

const char* to_string(Geometry g) { return g == Geometry::Planar ? "planar" : "axisymmetric"; }
const char* to_string(AdvectiveForm f) {
  switch (f) {
    case AdvectiveForm::Convective: return "convective";
    case AdvectiveForm::ConservativeFlux: return "conservative_flux";
    case AdvectiveForm::ConservativeDivergence: return "conservative_divergence";
  }
  return "?";
}
const char* to_string(BcKind k) {
  switch (k) {
    case BcKind::Dirichlet: return "dirichlet";
    case BcKind::Flux: return "flux";
    case BcKind::Robin: return "robin";
    case BcKind::TotalFlux: return "total_flux";
    case BcKind::Zero: return "zero";
    case BcKind::Axis: return "axis";
  }
  return "?";
}

namespace {

void eval_scalar(const ScalarData& d, const std::vector<double>& xy, std::vector<double>& out) {
  const int n = static_cast<int>(xy.size() / 2);
  out.assign(n, d.value);
  if (d.fn) d.fn(xy.data(), n, out.data());
}

struct BcEntry {
  std::vector<int> attrs;
  BcKind kind;
  BcData data;
};

// one natural-boundary element with its tabulated quadrature data
struct BdrTab {
  int be = 0;
  int entry = 0;
  std::vector<int> dofs;
  std::vector<double> N;  // nq * nb
  std::vector<double> x, y, w;  // w: ip.weight * |J| (no r)
  std::vector<double> g, h, ext;  // per quad point
};

}  // namespace

struct ScalarTransport::Impl {
  // data
  ScalarData kappa_d, src_d;
  BatchVec gradk_fn;
  Velocity vel;
  std::vector<BcEntry> bcs;
  // element tables
  int nE = 0, nd = 0, nq = 0, ndu = 0, vdu = 0;
  std::vector<int> edofs;      // nE * nd
  std::vector<int> uvd;        // nE * 2 * ndu (vdofs of the first two components)
  std::vector<double> Nref;    // nq * nd
  std::vector<double> Nuref;   // nq * ndu
  std::vector<double> X, W, Rr;  // (e, q): x (2), w (no r), r weight
  std::vector<double> dN;      // (e,q): nd * 2
  std::vector<double> lap;     // (e,q): nd
  std::vector<double> dNu;     // (e,q): ndu * 2
  std::vector<double> hK;      // per element: longest edge
  std::vector<double> kap, dk, src, bb, divb;  // tabulated data (kap, src, divb: per point; dk, bb: 2 per point)
  std::vector<double> hist;    // history source (T dofs), empty if none
  std::vector<BdrTab> btabs;
  std::vector<int> dir_attrs;
  std::vector<char> dir_attr_mask;
  std::set<int> axis_attrs;
  bool tabulated = false;
};

// ======================================================================================================== set-up
ScalarTransport::ScalarTransport(mfem::Mesh& mesh, const TransportOptions& opt)
    : mesh_(mesh), opt_(opt), impl_(new Impl) {
  // the operator assembles the local elements with serial matrices: a partitioned mesh would silently solve one
  // independent problem per partition, so more than one rank is refused
  fluids::require_single_rank(mesh, "ScalarTransport");
  AXF_REQUIRE(mesh.Dimension() == 2, "ScalarTransport: only 2D meshes (planar or the (r, z) half plane) are offered",
              "mesh", "3D transport is not offered.");
  AXF_REQUIRE(opt.order == 1 || opt.order == 2, axf::strcat("ScalarTransport: element order ", opt.order,
                                                           " is not offered (1 or 2)"), "order");
  AXF_REQUIRE(opt.tau_rule == "shakib_hK",
              "ScalarTransport: the only SUPG tau rule is 'shakib_hK'; coth, other lengths, other "
              "C and element-constant tau are not offered", "tau_rule", "Use tau_rule='shakib_hK'.");
  const int minq = 2 * opt.order + 4;
  AXF_REQUIRE(opt.quad_order == 0 || opt.quad_order >= minq,
              axf::strcat("ScalarTransport: quadrature degree ", opt.quad_order, " is below 2p + 4 = ", minq),
              "quad_order");
  qdeg_ = opt.quad_order == 0 ? minq : opt.quad_order;
  declared_ = opt.form;
  if (opt.supg)
    AXF_REQUIRE(mesh.GetNodes() == nullptr,
                "ScalarTransport: SUPG on isoparametric (curved) elements is not offered (the mapped Hessian is not "
                "implemented)", "supg", "Use straight-sided meshes, or run Galerkin (supg = false).");
  for (int e = 0; e < mesh.GetNE(); ++e)
    AXF_REQUIRE(mesh.GetElementGeometry(e) == mfem::Geometry::TRIANGLE,
                "ScalarTransport: only triangular meshes are offered", "mesh");
  fec_.reset(new mfem::H1_FECollection(opt.order, 2));
  fes_.reset(new mfem::FiniteElementSpace(&mesh, fec_.get()));
}

ScalarTransport::~ScalarTransport() = default;

void ScalarTransport::set_capacity(double a) {
  AXF_REQUIRE(std::isfinite(a) && a > 0, "capacity must be positive", "capacity");
  cap_ = a;
}
void ScalarTransport::set_shift(double s) { shift_ = s; }
void ScalarTransport::set_diffusivity(const ScalarData& k, const BatchVec& g) {
  impl_->kappa_d = k;
  impl_->gradk_fn = g;
  impl_->tabulated = false;
}
void ScalarTransport::set_velocity(const Velocity& v) {
  AXF_REQUIRE(v.is_set(), "ScalarTransport::set_velocity: neither a callable nor a finite-element field");
  AXF_REQUIRE(!(v.fes && v.fn), "ScalarTransport::set_velocity: give a callable or a field, not both");
  if (v.fes) AXF_REQUIRE(v.fes->GetMesh() == &mesh_, "the velocity field lives on another mesh (no interpolation between meshes)");
  if (v.fes) AXF_REQUIRE(v.fes->GetVDim() >= 2, "the velocity space needs at least two components");
  impl_->vel = v;
  impl_->tabulated = false;
}
void ScalarTransport::set_source(const ScalarData& f) {
  impl_->src_d = f;
  impl_->tabulated = false;
}
void ScalarTransport::add_bc(const std::vector<int>& attrs, BcKind kind, const BcData& data) {
  AXF_REQUIRE(!finalized_, "add_bc after finalize()");
  BcEntry e{attrs, kind, data};
  const int nattr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  for (int a : attrs) {
    AXF_REQUIRE(a >= 1 && a <= nattr, axf::strcat("boundary attribute ", a, " is not a boundary attribute of the mesh"));
    for (const auto& o : impl_->bcs)
      for (int b : o.attrs)
        AXF_REQUIRE(a != b, axf::strcat("boundary attribute ", a, " has two conditions"), "add_bc");
  }
  if (kind == BcKind::TotalFlux)
    AXF_REQUIRE(declared_ == AdvectiveForm::ConservativeFlux,
                "a total-flux datum needs the conservative_flux form (the natural datum of the other forms is the "
                "diffusive flux)", "bc", "Use kind 'flux', or form 'conservative_flux'.");
  if (kind == BcKind::Flux)
    AXF_REQUIRE(declared_ != AdvectiveForm::ConservativeFlux,
                "a diffusive-flux datum on a natural part is refused with conservative_flux (its natural datum is the "
                "total flux)", "bc", "Use kind 'total_flux' (or 'zero'; a robin condition is then a TOTAL-flux Robin "
                "j_out = h (T - T_ext)), or the convective form.");
  if (kind == BcKind::Axis)
    AXF_REQUIRE(opt_.geometry == Geometry::Axisymmetric, "an axis boundary needs geometry = axisymmetric", "bc");
  impl_->bcs.push_back(std::move(e));
}

// ======================================================================================================== finalize
void ScalarTransport::finalize() {
  Impl& I = *impl_;
  AXF_REQUIRE(!finalized_, "finalize() called twice");
  AXF_REQUIRE(I.kappa_d.fn || I.kappa_d.value > 0, "the diffusivity must be set and positive", "kappa");
  // every boundary attribute carries exactly one condition
  const int nattr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  std::vector<char> has(nattr + 1, 0);
  for (const auto& b : I.bcs)
    for (int a : b.attrs) has[a] = 1;
  for (int a = 1; a <= nattr; ++a)
    AXF_REQUIRE(has[a], axf::strcat("boundary attribute ", a, " has no condition (every attribute needs one; use 'zero' for "
                                                            "insulation, symmetry or outflow)"), "bcs");
  if (opt_.supg) {
    AXF_REQUIRE(I.vel.is_set(), "SUPG needs a velocity", "supg");
    if (I.kappa_d.fn)
      AXF_REQUIRE(static_cast<bool>(I.gradk_fn),
                  "SUPG with a variable diffusivity needs grad kappa (the strong residual carries grad kappa . grad T)",
                  "kappa");
  }
  if (opt_.form != AdvectiveForm::Convective && I.vel.fn && !I.vel.fes)
    AXF_REQUIRE(static_cast<bool>(I.vel.div_fn), "the conservative forms with a callable velocity need div b", "velocity");

  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const int nE = mesh_.GetNE();
  I.nE = nE;
  const mfem::FiniteElement* fe0 = fes_->GetFE(0);
  I.nd = fe0->GetDof();
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, qdeg_);
  I.nq = ir.GetNPoints();
  const int nd = I.nd, nq = I.nq;
  const size_t np = static_cast<size_t>(nE) * nq;
  I.edofs.resize(static_cast<size_t>(nE) * nd);
  I.Nref.assign(static_cast<size_t>(nq) * nd, 0.0);
  I.X.assign(np * 2, 0.0);
  I.W.assign(np, 0.0);
  I.Rr.assign(np, 1.0);
  I.dN.assign(np * nd * 2, 0.0);
  I.lap.assign(np * nd, 0.0);
  I.hK.assign(nE, 0.0);
  const mfem::FiniteElement* vfe = nullptr;
  if (I.vel.fes) {
    vfe = I.vel.fes->GetFE(0);
    I.ndu = vfe->GetDof();
    I.vdu = I.vel.fes->GetVDim();
    I.Nuref.assign(static_cast<size_t>(nq) * I.ndu, 0.0);
    I.dNu.assign(np * I.ndu * 2, 0.0);
    I.uvd.resize(static_cast<size_t>(nE) * 2 * I.ndu);
  }
  mfem::Vector sh(nd), shu(I.ndu);
  mfem::DenseMatrix dsh(nd, 2), dshu(std::max(I.ndu, 1), 2), hess(nd, 3);
  mfem::Array<int> dofs, vdofs;
  for (int q = 0; q < nq; ++q) {
    fe0->CalcShape(ir.IntPoint(q), sh);
    for (int a = 0; a < nd; ++a) I.Nref[q * nd + a] = sh(a);
    if (vfe) {
      vfe->CalcShape(ir.IntPoint(q), shu);
      for (int j = 0; j < I.ndu; ++j) I.Nuref[q * I.ndu + j] = shu(j);
    }
  }
  for (int e = 0; e < nE; ++e) {
    fes_->GetElementDofs(e, dofs);
    for (int a = 0; a < nd; ++a) I.edofs[static_cast<size_t>(e) * nd + a] = dofs[a];
    if (vfe) {
      mfem::Array<int> vd;
      I.vel.fes->GetElementDofs(e, vd);
      for (int c = 0; c < 2; ++c)
        for (int j = 0; j < I.ndu; ++j) I.uvd[(static_cast<size_t>(e) * 2 + c) * I.ndu + j] = I.vel.fes->DofToVDof(vd[j], c);
    }
    mfem::ElementTransformation* T = mesh_.GetElementTransformation(e);
    // longest edge (physical)
    mfem::Array<int> vs;
    mesh_.GetElementVertices(e, vs);
    double hmax = 0;
    for (int i = 0; i < 3; ++i) {
      const double* a = mesh_.GetVertex(vs[i]);
      const double* b = mesh_.GetVertex(vs[(i + 1) % 3]);
      hmax = std::max(hmax, std::hypot(a[0] - b[0], a[1] - b[1]));
    }
    I.hK[e] = hmax;
    for (int q = 0; q < nq; ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      const size_t k = static_cast<size_t>(e) * nq + q;
      double x[3] = {0, 0, 0};
      mfem::Vector xv(x, 2);
      T->Transform(ip, xv);
      I.X[2 * k] = x[0];
      I.X[2 * k + 1] = x[1];
      I.W[k] = ip.weight * T->Weight();
      I.Rr[k] = axi ? x[0] : 1.0;
      fe0->CalcPhysDShape(*T, dsh);
      for (int a = 0; a < nd; ++a) {
        I.dN[(k * nd + a) * 2] = dsh(a, 0);
        I.dN[(k * nd + a) * 2 + 1] = dsh(a, 1);
      }
      if (opt_.order == 2) {
        fe0->CalcPhysHessian(*T, hess);
        for (int a = 0; a < nd; ++a) I.lap[k * nd + a] = hess(a, 0) + hess(a, 2);
      }
      if (vfe) {
        vfe->CalcPhysDShape(*T, dshu);
        for (int j = 0; j < I.ndu; ++j) {
          I.dNu[(k * I.ndu + j) * 2] = dshu(j, 0);
          I.dNu[(k * I.ndu + j) * 2 + 1] = dshu(j, 1);
        }
      }
    }
  }

  // ---- tabulate the data at the interior quadrature points
  eval_scalar(I.kappa_d, I.X, I.kap);
  I.dk.assign(np * 2, 0.0);
  if (I.gradk_fn) I.gradk_fn(I.X.data(), static_cast<int>(np), 2, I.dk.data());
  eval_scalar(I.src_d, I.X, I.src);
  for (double k : I.kap)
    AXF_REQUIRE(std::isfinite(k) && k > 0, "the diffusivity is not positive and finite at a quadrature point", "kappa");
  I.bb.assign(np * 2, 0.0);
  I.divb.assign(np, 0.0);
  if (I.vel.fn) {
    I.vel.fn(I.X.data(), static_cast<int>(np), 2, I.bb.data());
    if (I.vel.div_fn) I.vel.div_fn(I.X.data(), static_cast<int>(np), I.divb.data());
  }

  // ---- boundary: essential dofs and lift, natural quadrature tables
  const int nbe = mesh_.GetNBE();
  ess_.assign(fes_->GetNDofs(), 0);
  lift_.SetSize(fes_->GetNDofs());
  lift_ = 0.0;
  const mfem::IntegrationRule& sir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, qdeg_);
  std::map<int, int> attr_entry;
  for (size_t i = 0; i < I.bcs.size(); ++i)
    for (int a : I.bcs[i].attrs) attr_entry[a] = static_cast<int>(i);
  std::vector<std::vector<int>> dir_dofs(I.bcs.size());
  struct DirPt { int dof; double x, y; int entry; };
  std::vector<DirPt> dpts;
  std::set<int> seen_dir;
  for (int be = 0; be < nbe; ++be) {
    const int attr = mesh_.GetBdrAttribute(be);
    const int ent = attr_entry.at(attr);
    const BcEntry& bc = I.bcs[ent];
    mfem::Array<int> bd;
    fes_->GetBdrElementDofs(be, bd);
    const mfem::FiniteElement* bfe = fes_->GetBE(be);
    mfem::ElementTransformation* BT = mesh_.GetBdrElementTransformation(be);
    bool on_axis = true;
    {
      mfem::Array<int> bv;
      mesh_.GetBdrElementVertices(be, bv);
      for (int v : bv) on_axis = on_axis && std::abs(mesh_.GetVertex(v)[0]) < 1e-12;
    }
    if (axi && on_axis && bc.kind != BcKind::Axis && bc.kind != BcKind::Zero) {
      AXF_REQUIRE(false,
                  axf::strcat("a ", to_string(bc.kind), " condition on the symmetry axis r = 0 (boundary attribute ", attr,
                              ") is refused: T is even in r and the axis takes no condition (natural, zero flux)"),
                  "bcs", "Mark the axis attribute with kind 'axis'.");
    }
    if (axi && on_axis && bc.kind == BcKind::Zero)
      AXF_REQUIRE(false, axf::strcat("boundary attribute ", attr, " lies on the axis r = 0: use kind 'axis'"), "bcs");
    if (!axi) AXF_REQUIRE(bc.kind != BcKind::Axis, "an axis boundary needs geometry = axisymmetric", "bcs");
    if (axi && bc.kind == BcKind::Axis) {
      AXF_REQUIRE(on_axis, axf::strcat("boundary attribute ", attr, " is marked 'axis' but is not on r = 0"), "bcs");
    }
    if (bc.kind == BcKind::Dirichlet) {
      const mfem::IntegrationRule& nodes = bfe->GetNodes();
      for (int k = 0; k < bd.Size(); ++k) {
        double x[3] = {0, 0, 0};
        mfem::Vector xv(x, 2);
        BT->Transform(nodes.IntPoint(k), xv);
        ess_[bd[k]] = 1;
        dpts.push_back({bd[k], x[0], x[1], ent});
      }
    }
    if (bc.kind == BcKind::Flux || bc.kind == BcKind::TotalFlux || bc.kind == BcKind::Robin) {
      BdrTab t;
      t.be = be;
      t.entry = ent;
      t.dofs.assign(bd.begin(), bd.end());
      const int nb = bd.Size();
      const int nqb = sir.GetNPoints();
      t.N.assign(static_cast<size_t>(nqb) * nb, 0.0);
      mfem::Vector shb(nb);
      for (int q = 0; q < nqb; ++q) {
        const mfem::IntegrationPoint& ip = sir.IntPoint(q);
        BT->SetIntPoint(&ip);
        bfe->CalcShape(ip, shb);
        for (int a = 0; a < nb; ++a) t.N[q * nb + a] = shb(a);
        double x[3] = {0, 0, 0};
        mfem::Vector xv(x, 2);
        BT->Transform(ip, xv);
        t.x.push_back(x[0]);
        t.y.push_back(x[1]);
        t.w.push_back(ip.weight * BT->Weight());
      }
      I.btabs.push_back(std::move(t));
    }
  }
  // Dirichlet values: the nodal interpolant of the per-attribute data at the Dirichlet nodes
  {
    std::map<int, std::vector<size_t>> by_entry;
    for (size_t i = 0; i < dpts.size(); ++i) by_entry[dpts[i].entry].push_back(i);
    for (auto& kv : by_entry) {
      std::vector<double> xy;
      for (size_t i : kv.second) {
        xy.push_back(dpts[i].x);
        xy.push_back(dpts[i].y);
      }
      std::vector<double> g;
      eval_scalar(I.bcs[kv.first].data.g, xy, g);
      for (size_t j = 0; j < kv.second.size(); ++j) lift_(dpts[kv.second[j]].dof) = g[j];
    }
  }
  // natural data at the boundary quadrature points
  {
    std::map<int, std::vector<size_t>> by_entry;
    for (size_t i = 0; i < I.btabs.size(); ++i) by_entry[I.btabs[i].entry].push_back(i);
    for (auto& kv : by_entry) {
      const BcEntry& bc = I.bcs[kv.first];
      std::vector<double> xy;
      std::vector<std::pair<size_t, size_t>> idx;  // (table, nq)
      for (size_t i : kv.second) {
        const BdrTab& t = I.btabs[i];
        for (size_t q = 0; q < t.x.size(); ++q) {
          xy.push_back(t.x[q]);
          xy.push_back(t.y[q]);
        }
      }
      std::vector<double> g, h, ext;
      if (bc.kind == BcKind::Robin) {
        eval_scalar(bc.data.h, xy, h);
        eval_scalar(bc.data.ext, xy, ext);
      } else {
        eval_scalar(bc.data.g, xy, g);
      }
      size_t pos = 0;
      for (size_t i : kv.second) {
        BdrTab& t = I.btabs[i];
        const size_t n = t.x.size();
        if (bc.kind == BcKind::Robin) {
          t.h.assign(h.begin() + pos, h.begin() + pos + n);
          t.ext.assign(ext.begin() + pos, ext.begin() + pos + n);
        } else {
          t.g.assign(g.begin() + pos, g.begin() + pos + n);
        }
        pos += n;
      }
    }
  }
  // mass matrix M_ab = int N_a N_b r (degree 2p + 2)
  {
    const mfem::IntegrationRule& mir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * opt_.order + 2);
    mass_.reset(new mfem::SparseMatrix(fes_->GetNDofs()));
    mfem::Vector shm(nd);
    mfem::DenseMatrix em(nd);
    mfem::Array<int> md;
    for (int e = 0; e < nE; ++e) {
      mfem::ElementTransformation* T = mesh_.GetElementTransformation(e);
      em = 0.0;
      for (int q = 0; q < mir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = mir.IntPoint(q);
        T->SetIntPoint(&ip);
        fe0->CalcShape(ip, shm);
        double x[3] = {0, 0, 0};
        mfem::Vector xv(x, 2);
        T->Transform(ip, xv);
        const double w = ip.weight * T->Weight() * (axi ? x[0] : 1.0);
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b) em(a, b) += w * shm(a) * shm(b);
      }
      fes_->GetElementDofs(e, md);
      mass_->AddSubMatrix(md, md, em, 0);
    }
    mass_->Finalize(0);
  }
  I.tabulated = true;
  finalized_ = true;
}

// ======================================================================================================== kernels
struct ScalarTransport::Pt {
  double T = 0, g[2] = {0, 0}, lap = 0;
  double beta[2] = {0, 0}, divb = 0;
  double kap = 0, dk[2] = {0, 0}, src = 0;
  double tauS = 0, tau = 0, res = 0, r = 1, h = 0;
};

void ScalarTransport::point(int e, int q, const double* Te, const double* ue, const double* he, Pt& p) const {
  const Impl& I = *impl_;
  const int nd = I.nd, nq = I.nq;
  const size_t k = static_cast<size_t>(e) * nq + q;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  double T = 0, g0 = 0, g1 = 0, lap = 0, hs = 0;
  for (int a = 0; a < nd; ++a) {
    T += I.Nref[q * nd + a] * Te[a];
    g0 += I.dN[(k * nd + a) * 2] * Te[a];
    g1 += I.dN[(k * nd + a) * 2 + 1] * Te[a];
    lap += I.lap[k * nd + a] * Te[a];
    if (he) hs += I.Nref[q * nd + a] * he[a];
  }
  p.T = T;
  p.g[0] = g0;
  p.g[1] = g1;
  p.lap = lap;
  p.r = I.Rr[k];
  p.h = I.hK[e];
  p.kap = I.kap[k];
  p.dk[0] = I.dk[2 * k];
  p.dk[1] = I.dk[2 * k + 1];
  p.src = I.src[k] + hs;
  const double scale = I.vel.scale;
  if (I.vel.fes) {
    double u0 = 0, u1 = 0, dv = 0, hoop = 0;
    const int ndu = I.ndu;
    for (int j = 0; j < ndu; ++j) {
      const double a0 = ue[I.uvd[(static_cast<size_t>(e) * 2 + 0) * ndu + j]];
      const double a1 = ue[I.uvd[(static_cast<size_t>(e) * 2 + 1) * ndu + j]];
      const double Nj = I.Nuref[q * ndu + j];
      u0 += Nj * a0;
      u1 += Nj * a1;
      dv += I.dNu[(k * ndu + j) * 2] * a0 + I.dNu[(k * ndu + j) * 2 + 1] * a1;
      hoop += Nj * a0;
    }
    p.beta[0] = scale * u0;
    p.beta[1] = scale * u1;
    p.divb = scale * (dv + (axi ? hoop / I.Rr[k] : 0.0));
  } else if (I.vel.fn) {
    p.beta[0] = scale * I.bb[2 * k];
    p.beta[1] = scale * I.bb[2 * k + 1];
    p.divb = scale * I.divb[k];
  }
  const bool cons = opt_.form != AdvectiveForm::Convective;
  const double bg = p.beta[0] * g0 + p.beta[1] * g1;
  const double conv = cons ? bg + p.divb * T : bg;
  double diff = p.kap * lap + p.dk[0] * g0 + p.dk[1] * g1;
  if (axi) diff += p.kap * g0 / p.r;
  p.res = shift_ * cap_ * T + conv - diff - p.src;
  if (opt_.supg) {
    const double bn = std::hypot(p.beta[0], p.beta[1]);
    const double h = p.h, pp = opt_.order;
    const double A = std::pow(2 * pp * bn / h, 2) + 9 * std::pow(4 * pp * pp * p.kap / (h * h), 2) +
                     std::pow(shift_ * cap_, 2);
    p.tauS = A > 0 ? 1.0 / std::sqrt(A) : 0.0;
    p.tau = p.tauS;
  }
}

void ScalarTransport::residual_full(const mfem::Vector& T, const mfem::Vector* u, mfem::Vector& R) const {
  AXF_REQUIRE(finalized_, "residual_full before finalize()");
  const Impl& I = *impl_;
  AXF_REQUIRE(!I.vel.fes || (u && u->Size() == I.vel.fes->GetVSize()),
              "the velocity is a finite-element field: pass its coefficients", "u");
  const int nd = I.nd, nq = I.nq;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const bool supg = opt_.supg;
  R.SetSize(fes_->GetNDofs());
  R = 0.0;
  std::vector<double> Te(nd), he(nd), Rl(nd);
  const double* ud = u ? u->GetData() : nullptr;
  const bool cons_flux = opt_.form == AdvectiveForm::ConservativeFlux;
  const bool cons_div = opt_.form == AdvectiveForm::ConservativeDivergence;
  for (int e = 0; e < I.nE; ++e) {
    for (int a = 0; a < nd; ++a) {
      Te[a] = T(I.edofs[static_cast<size_t>(e) * nd + a]);
      if (!I.hist.empty()) he[a] = I.hist[I.edofs[static_cast<size_t>(e) * nd + a]];
    }
    std::fill(Rl.begin(), Rl.end(), 0.0);
    for (int q = 0; q < nq; ++q) {
      Pt p;
      point(e, q, Te.data(), ud, I.hist.empty() ? nullptr : he.data(), p);
      const size_t k = static_cast<size_t>(e) * nq + q;
      const double w = I.W[k] * I.Rr[k];
      const double bgrad = p.beta[0] * p.g[0] + p.beta[1] * p.g[1];
      for (int a = 0; a < nd; ++a) {
        const double Na = I.Nref[q * nd + a];
        const double d0 = I.dN[(k * nd + a) * 2], d1 = I.dN[(k * nd + a) * 2 + 1];
        const double bd = p.beta[0] * d0 + p.beta[1] * d1;
        double v = p.kap * (p.g[0] * d0 + p.g[1] * d1) + shift_ * cap_ * p.T * Na - p.src * Na;
        if (cons_flux) v += -p.T * bd;
        else if (cons_div) v += (bgrad + p.divb * p.T) * Na;
        else v += bgrad * Na;
        if (supg) v += p.tau * bd * p.res;
        Rl[a] += w * v;
      }
    }
    for (int a = 0; a < nd; ++a) R(I.edofs[static_cast<size_t>(e) * nd + a]) += Rl[a];
  }
  // natural boundary terms
  for (const BdrTab& t : I.btabs) {
    const BcEntry& bc = I.bcs[t.entry];
    const int nb = static_cast<int>(t.dofs.size());
    const int nqb = static_cast<int>(t.x.size());
    for (int q = 0; q < nqb; ++q) {
      double Tq = 0;
      for (int a = 0; a < nb; ++a) Tq += t.N[q * nb + a] * T(t.dofs[a]);
      const double rb = axi ? t.x[q] : 1.0;
      if (bc.kind == BcKind::Robin) {
        const double hh = t.h[q];
        for (int a = 0; a < nb; ++a) R(t.dofs[a]) += t.w[q] * rb * hh * (Tq - t.ext[q]) * t.N[q * nb + a];
      } else {
        const double gg = t.g[q];
        for (int a = 0; a < nb; ++a) R(t.dofs[a]) += -t.w[q] * rb * gg * t.N[q * nb + a];
      }
    }
  }
}

mfem::SparseMatrix ScalarTransport::jacobian(const mfem::Vector& T, const mfem::Vector* u) const {
  AXF_REQUIRE(finalized_, "jacobian before finalize()");
  const Impl& I = *impl_;
  const int nd = I.nd, nq = I.nq;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const bool cons_flux = opt_.form == AdvectiveForm::ConservativeFlux;
  const bool cons_div = opt_.form == AdvectiveForm::ConservativeDivergence;
  const bool cons = cons_flux || cons_div;
  mfem::SparseMatrix J(fes_->GetNDofs());
  mfem::DenseMatrix Em(nd);
  mfem::Array<int> dofs(nd);
  std::vector<double> Te(nd), he(nd);
  const double* ud = u ? u->GetData() : nullptr;
  for (int e = 0; e < I.nE; ++e) {
    for (int a = 0; a < nd; ++a) {
      dofs[a] = I.edofs[static_cast<size_t>(e) * nd + a];
      Te[a] = T(dofs[a]);
      if (!I.hist.empty()) he[a] = I.hist[dofs[a]];
    }
    Em = 0.0;
    for (int q = 0; q < nq; ++q) {
      Pt p;
      point(e, q, Te.data(), ud, I.hist.empty() ? nullptr : he.data(), p);
      const size_t k = static_cast<size_t>(e) * nq + q;
      const double w = I.W[k] * I.Rr[k];
      for (int a = 0; a < nd; ++a) {
        const double Na = I.Nref[q * nd + a];
        const double a0 = I.dN[(k * nd + a) * 2], a1 = I.dN[(k * nd + a) * 2 + 1];
        const double bda = p.beta[0] * a0 + p.beta[1] * a1;
        for (int b = 0; b < nd; ++b) {
          const double Nb = I.Nref[q * nd + b];
          const double b0 = I.dN[(k * nd + b) * 2], b1 = I.dN[(k * nd + b) * 2 + 1];
          const double bdb = p.beta[0] * b0 + p.beta[1] * b1;
          double v = p.kap * (a0 * b0 + a1 * b1) + shift_ * cap_ * Na * Nb;
          if (cons_flux) v += -Nb * bda;
          else if (cons_div) v += Na * (bdb + p.divb * Nb);
          else v += Na * bdb;
          if (opt_.supg) {
            double dres = shift_ * cap_ * Nb + (bdb + (cons ? p.divb * Nb : 0.0));
            dres -= p.kap * I.lap[k * nd + b] + p.dk[0] * b0 + p.dk[1] * b1;
            if (axi) dres -= p.kap * b0 / p.r;
            v += p.tau * bda * dres;
          }
          Em(a, b) += w * v;
        }
      }
    }
    J.AddSubMatrix(dofs, dofs, Em, 0);
  }
  for (const BdrTab& t : I.btabs) {
    const BcEntry& bc = I.bcs[t.entry];
    if (bc.kind != BcKind::Robin) continue;
    const int nb = static_cast<int>(t.dofs.size());
    mfem::DenseMatrix Bm(nb);
    Bm = 0.0;
    mfem::Array<int> bd(nb);
    for (int a = 0; a < nb; ++a) bd[a] = t.dofs[a];
    for (int q = 0; q < static_cast<int>(t.x.size()); ++q) {
      const double rb = axi ? t.x[q] : 1.0;
      const double hh = t.h[q];
      for (int a = 0; a < nb; ++a)
        for (int b = 0; b < nb; ++b) Bm(a, b) += t.w[q] * rb * hh * t.N[q * nb + a] * t.N[q * nb + b];
    }
    J.AddSubMatrix(bd, bd, Bm, 0);
  }
  J.Finalize(0);
  return J;
}

mfem::SparseMatrix ScalarTransport::advection_jacobian_u(const mfem::Vector& T, const mfem::Vector& u) const {
  AXF_REQUIRE(finalized_, "advection_jacobian_u before finalize()");
  const Impl& I = *impl_;
  AXF_REQUIRE(I.vel.fes, "advection_jacobian_u needs a finite-element velocity field");
  const int nd = I.nd, nq = I.nq, ndu = I.ndu;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const bool cons_flux = opt_.form == AdvectiveForm::ConservativeFlux;
  const bool cons_div = opt_.form == AdvectiveForm::ConservativeDivergence;
  const double scale = I.vel.scale;
  mfem::SparseMatrix J(fes_->GetNDofs(), I.vel.fes->GetVSize());
  mfem::DenseMatrix Em(nd, 2 * ndu);
  mfem::Array<int> rows(nd), cols(2 * ndu);
  std::vector<double> Te(nd), he(nd);
  for (int e = 0; e < I.nE; ++e) {
    for (int a = 0; a < nd; ++a) {
      rows[a] = I.edofs[static_cast<size_t>(e) * nd + a];
      Te[a] = T(rows[a]);
      if (!I.hist.empty()) he[a] = I.hist[rows[a]];
    }
    for (int c = 0; c < 2; ++c)
      for (int j = 0; j < ndu; ++j) cols[c * ndu + j] = I.uvd[(static_cast<size_t>(e) * 2 + c) * ndu + j];
    Em = 0.0;
    for (int q = 0; q < nq; ++q) {
      Pt p;
      point(e, q, Te.data(), u.GetData(), I.hist.empty() ? nullptr : he.data(), p);
      const size_t k = static_cast<size_t>(e) * nq + q;
      const double w = I.W[k] * I.Rr[k];
      const double h = p.h, pp = opt_.order;
      for (int c = 0; c < 2; ++c) {
        for (int j = 0; j < ndu; ++j) {
          const double Nj = I.Nuref[q * ndu + j];
          const double dNj = I.dNu[(k * ndu + j) * 2 + c];
          const double dbg = scale * Nj * p.g[c];  // delta beta . grad T
          const double ddiv = scale * (dNj + (c == 0 && axi ? Nj / I.Rr[k] : 0.0));
          double dtau = 0.0;
          if (opt_.supg)
            dtau = -std::pow(p.tauS, 3) * std::pow(2 * pp / h, 2) * (p.beta[c] * scale * Nj);
          const double dres = dbg + ((cons_flux || cons_div) ? ddiv * p.T : 0.0);
          for (int a = 0; a < nd; ++a) {
            const double Na = I.Nref[q * nd + a];
            const double a0 = I.dN[(k * nd + a) * 2], a1 = I.dN[(k * nd + a) * 2 + 1];
            const double bda = p.beta[0] * a0 + p.beta[1] * a1;
            double v;
            if (cons_flux) v = -p.T * scale * Nj * I.dN[(k * nd + a) * 2 + c];
            else if (cons_div) v = Na * dres;
            else v = Na * dbg;
            if (opt_.supg) {
              v += p.tau * bda * dres;                                        // tau (beta.grad v) dRes
              v += dtau * bda * p.res;                                        // dtau (beta.grad v) Res
              v += p.tau * (scale * Nj * I.dN[(k * nd + a) * 2 + c]) * p.res;  // tau (dbeta.grad v) Res
            }
            Em(a, c * ndu + j) += w * v;
          }
        }
      }
    }
    J.AddSubMatrix(rows, cols, Em, 0);
  }
  J.Finalize(0);
  return J;
}

// ======================================================================================================== outputs
std::vector<double> ScalarTransport::boundary_weights(const std::vector<int>& attrs) const {
  std::vector<double> w(fes_->GetNDofs(), 0.0);
  for (int be = 0; be < mesh_.GetNBE(); ++be) {
    if (std::find(attrs.begin(), attrs.end(), mesh_.GetBdrAttribute(be)) == attrs.end()) continue;
    mfem::Array<int> bd;
    fes_->GetBdrElementDofs(be, bd);
    for (int k = 0; k < bd.Size(); ++k) w[bd[k]] = 1.0;
  }
  return w;
}

std::vector<TauSample> ScalarTransport::tau_samples(const mfem::Vector& T, const mfem::Vector* u) const {
  const Impl& I = *impl_;
  std::vector<TauSample> out;
  std::vector<double> Te(I.nd), he(I.nd);
  for (int e = 0; e < I.nE; ++e) {
    for (int a = 0; a < I.nd; ++a) {
      Te[a] = T(I.edofs[static_cast<size_t>(e) * I.nd + a]);
      if (!I.hist.empty()) he[a] = I.hist[I.edofs[static_cast<size_t>(e) * I.nd + a]];
    }
    for (int q = 0; q < I.nq; ++q) {
      Pt p;
      point(e, q, Te.data(), u ? u->GetData() : nullptr, I.hist.empty() ? nullptr : he.data(), p);
      const size_t k = static_cast<size_t>(e) * I.nq + q;
      out.push_back({e, I.X[2 * k], I.X[2 * k + 1], p.tau});
    }
  }
  return out;
}

void ScalarTransport::set_history_source(const mfem::Vector& h) {
  AXF_REQUIRE(h.Size() == fes_->GetNDofs(), "history source: wrong size", "h");
  impl_->hist.assign(h.GetData(), h.GetData() + h.Size());
}
void ScalarTransport::clear_history_source() { impl_->hist.clear(); }

double ScalarTransport::surface_flux(const mfem::Vector& T, const std::vector<int>& attrs) const {
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const mfem::IntegrationRule& sir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, qdeg_);
  double total = 0;
  mfem::Array<int> dofs;
  for (int be = 0; be < mesh_.GetNBE(); ++be) {
    if (std::find(attrs.begin(), attrs.end(), mesh_.GetBdrAttribute(be)) == attrs.end()) continue;
    mfem::FaceElementTransformations* ft = mesh_.GetBdrFaceTransformations(be);
    const int el = ft->Elem1No;
    fes_->GetElementDofs(el, dofs);
    const mfem::FiniteElement* fe = fes_->GetFE(el);
    mfem::DenseMatrix dsh(fe->GetDof(), 2);
    for (int q = 0; q < sir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = sir.IntPoint(q);
      ft->SetAllIntPoints(&ip);
      fe->CalcPhysDShape(*ft->Elem1, dsh);
      double g[2] = {0, 0};
      for (int a = 0; a < fe->GetDof(); ++a) {
        g[0] += dsh(a, 0) * T(dofs[a]);
        g[1] += dsh(a, 1) * T(dofs[a]);
      }
      mfem::Vector nor(2);
      mfem::CalcOrtho(ft->Face->Jacobian(), nor);
      nor /= nor.Norml2();
      double x[3];
      mfem::Vector xv(x, 2);
      ft->Elem1->Transform(ft->Elem1->GetIntPoint(), xv);
      const double kap = [&] {
        // diffusivity at the physical point (tabulated tables are per interior point; evaluate the field directly)
        double k0 = impl_->kappa_d.value;
        if (impl_->kappa_d.fn) impl_->kappa_d.fn(x, 1, &k0);
        return k0;
      }();
      total += -ip.weight * ft->Face->Weight() * (axi ? x[0] : 1.0) * kap * (g[0] * nor(0) + g[1] * nor(1));
    }
  }
  return total;
}

// ======================================================================================================== dual norm, solve
namespace {

fluids::CsrMatrix restrict_free(const mfem::SparseMatrix& A, const std::vector<int>& map, int nfree) {
  fluids::CsrMatrix C;
  C.n = nfree;
  C.rowptr.assign(nfree + 1, 0);
  const int* I = A.GetI();
  const int* J = A.GetJ();
  const double* D = A.GetData();
  for (int i = 0; i < A.Height(); ++i) {
    if (map[i] < 0) continue;
    for (int k = I[i]; k < I[i + 1]; ++k)
      if (map[J[k]] >= 0) {
        C.col.push_back(map[J[k]]);
        C.val.push_back(D[k]);
      }
    C.rowptr[map[i] + 1] = static_cast<int>(C.col.size());
  }
  for (int i = 0; i < nfree; ++i)
    if (C.rowptr[i + 1] < C.rowptr[i]) C.rowptr[i + 1] = C.rowptr[i];
  return C;
}

}  // namespace

double ScalarTransport::dual_norm(const mfem::Vector& R) const {
  AXF_REQUIRE(finalized_, "dual_norm before finalize()");
  if (free_map_.empty()) {
    free_map_.assign(fes_->GetNDofs(), -1);
    nfree_ = 0;
    for (int i = 0; i < fes_->GetNDofs(); ++i)
      if (!ess_[i]) free_map_[i] = nfree_++;
  }
  if (!mass_lu_) {
    fluids::CsrMatrix M = restrict_free(*mass_, free_map_, nfree_);
    mass_csr_ = fluids::DistCsr::from_serial(M);
    mass_csr_.sum_duplicates();
    mass_lu_.reset(new fluids::SparseLU(lu_backend_));
    mass_lu_->set_refinement(10, 1e-15);
    mass_lu_->set_throw_on_miss(false);
    mass_lu_->set_refine_to_stagnation(true);
    mass_lu_->factorize(mass_csr_);
  }
  std::vector<double> b(nfree_);
  for (int i = 0; i < fes_->GetNDofs(); ++i)
    if (free_map_[i] >= 0) b[free_map_[i]] = R(i);
  const std::vector<double> y = mass_lu_->solve(b);
  double s = 0;
  for (int i = 0; i < nfree_; ++i) s += b[i] * y[i];
  return std::sqrt(std::max(s, 0.0));
}

SolveDiagnostics ScalarTransport::solve(mfem::Vector& T, const mfem::Vector* u, fluids::LUBackend lu, double rtol,
                                        int maxit) {
  AXF_REQUIRE(finalized_, "solve before finalize()");
  SolveDiagnostics d;
  d.dofs = fes_->GetNDofs();
  lu_backend_ = lu;
  T = lift_;
  mfem::Vector R;
  residual_full(T, u, R);
  d.residual_lift = dual_norm(R);
  d.free_dofs = nfree_;
  if (!(std::isfinite(d.residual_lift))) {
    d.failure = "the residual of the lift is not finite";
    return d;
  }
  if (d.residual_lift == 0.0) {
    d.converged = true;
    d.relative_residual = 0.0;
    return d;
  }
  double rel = 1.0;
  for (int it = 0; it < maxit && rel > rtol; ++it) {
    mfem::SparseMatrix J = jacobian(T, u);
    fluids::CsrMatrix Jf = restrict_free(J, free_map_, nfree_);
    fluids::DistCsr D = fluids::DistCsr::from_serial(Jf);
    D.sum_duplicates();
    std::vector<double> b(nfree_);
    for (int i = 0; i < d.dofs; ++i)
      if (free_map_[i] >= 0) b[free_map_[i]] = -R(i);
    fluids::SparseLU slu(lu);
    slu.set_refinement(10, std::min(1e-13, 1e-2 * rtol));
    slu.set_throw_on_miss(false);
    slu.set_refine_to_stagnation(true);
    slu.set_threads(1);
    try {
      slu.factorize(D);
      const std::vector<double> x = slu.solve(b);
      for (int i = 0; i < d.dofs; ++i)
        if (free_map_[i] >= 0) T(i) += x[free_map_[i]];
    } catch (const std::exception& ex) {
      d.failure = std::string("linear solve failed: ") + ex.what();
      return d;
    }
    d.lu_residual = slu.stats().rel_residual;
    residual_full(T, u, R);
    rel = dual_norm(R) / d.residual_lift;
    ++d.iterations;
  }
  d.relative_residual = rel;
  d.converged = std::isfinite(rel) && rel <= rtol;
  if (!d.converged && d.failure.empty()) d.failure = "not converged";
  return d;
}

}  // namespace axf::transport
