// 3D current distribution on MPI ranks; see cpp/include/axionfem/chem/current_distribution_3d.hpp.
#include "axionfem/chem/current_distribution_3d.hpp"

#ifndef AXF_NO_MPI

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "axionfem/sym_solver.hpp"
#include "mfem.hpp"

namespace axf::chem {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("axf::chem::CurrentDistribution3D: " + msg); }

void ensure_mpi_chem() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) mfem::Mpi::Init();
  mfem::Hypre::Init();
}

double gsum(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
double gmax(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return v;
}
long gsuml(long v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
bool gor(bool b) {
  int v = b ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return v != 0;
}
}  // namespace

struct BPoint3 {
  double x[3], w, n[3];
};
struct BElem3 {
  int be, attr, elem;
  std::vector<int> dofs;  // L-dofs of the adjacent element
  std::vector<BPoint3> pts;
  std::vector<std::vector<double>> shape;
};
struct Electrode3 {
  ElectrodeSpec spec;
  std::vector<int> belems;
  std::vector<double> g;
  int unknown = -1;  // index among the galvanostatic phi_s
};

struct CurrentDistribution3D::Impl {
  int rank = 0, size = 1;
  int order = 1, deg_dom = 0, deg_bdr = 0;
  std::unique_ptr<mfem::ParMesh> pmesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> pfes;
  int nl = 0, nt = 0;
  // rank 0: the serial mesh
  Eigen::MatrixXd V0, G0;
  Eigen::MatrixXi E0;
  // coefficients
  bool kappa_const_set = false;
  double kappa_const = 0;
  Eigen::VectorXd kappa_pts, f_pts;
  std::map<int, Eigen::VectorXd> flux;
  std::vector<int> elem_pt0;
  std::vector<BElem3> bdr;
  std::vector<Electrode3> el;
  int ngalv = 0;
  // assembled
  bool assembled = false;
  std::unique_ptr<mfem::SparseMatrix> Kloc;
  std::unique_ptr<mfem::HypreParMatrix> K, Kabs;
  mfem::Vector Fv, Fb;  // T-vectors
  mfem::Array<int> ess;   // owned essential true dofs
  mfem::Vector ess_val;
  bool laid_out = false;
  // state
  mfem::Vector x;            // T-vector
  std::vector<double> ps;    // galvanostatic phi_s
  NewtonOptions eval_opt;
  double lin_rel = 1e-12;
  int lin_maxit = 2000;
  int lin_iters_max = 0;
  long lin_iters_total = 0;
  double lin_worst = 0;

  long domain_point_count() const {
    long n = 0;
    for (int e = 0; e < pmesh->GetNE(); ++e) n += mfem::IntRules.Get(pmesh->GetElementBaseGeometry(e), deg_dom).GetNPoints();
    return n;
  }
  int nattr_max() const { return pmesh->bdr_attributes.Size() ? pmesh->bdr_attributes.Max() : 0; }

  void prolong(const mfem::Vector& t, mfem::Vector& l) const {
    l.SetSize(nl);
    pfes->GetProlongationMatrix()->Mult(t, l);
  }
  void restrict_add(const mfem::Vector& l, mfem::Vector& t) const {
    mfem::Vector tmp(nt);
    pfes->GetProlongationMatrix()->MultTranspose(l, tmp);
    t += tmp;
  }

  BElem3 make_belem(int be, int degree) {
    BElem3 B;
    B.be = be;
    B.attr = pmesh->GetBdrAttribute(be);
    mfem::FaceElementTransformations* FT = pmesh->GetBdrFaceTransformations(be);
    if (!FT) fail("boundary element " + std::to_string(be) + " has no adjacent element");
    B.elem = FT->Elem1No;
    mfem::Array<int> dofs;
    pfes->GetElementDofs(B.elem, dofs);
    B.dofs.assign(dofs.begin(), dofs.end());
    const mfem::FiniteElement* fe = pfes->GetFE(B.elem);
    const int nd = fe->GetDof();
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(pmesh->GetBdrElementGeometry(be), degree);
    mfem::Vector shape(nd), xv(3), nor(3);
    mfem::IntegrationPoint rc;   // reference centroid of the adjacent element
    rc.Set3(0.25, 0.25, 0.25);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      FT->SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip = FT->GetElement1IntPoint();
      fe->CalcShape(eip, shape);
      FT->Transform(ip, xv);
      mfem::CalcOrtho(FT->Jacobian(), nor);
      const double len = nor.Norml2();
      double n[3] = {nor(0) / len, nor(1) / len, nor(2) / len};
      // outward: the element map's Jacobian at the point applied to (eip - reference centroid), a reference vector
      // pointing out of the element through this face, gives a physical vector on the outside of the face's tangent
      // plane (local test; valid for any non-degenerate curved element)
      {
        const mfem::DenseMatrix& Je = FT->Elem1->Jacobian();
        const double wv[3] = {eip.x - rc.x, eip.y - rc.y, eip.z - rc.z};
        double ov[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d)
          for (int e = 0; e < 3; ++e) ov[d] += Je(d, e) * wv[e];
        if (ov[0] * n[0] + ov[1] * n[1] + ov[2] * n[2] < 0)
          for (double& v : n) v = -v;
      }
      BPoint3 P{{xv(0), xv(1), xv(2)}, ip.weight * len, {n[0], n[1], n[2]}};
      B.pts.push_back(P);
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
  void require_attr_global(int a, const char* where) const {
    long n = (long)bdr_of({a}).size();
    if (gsuml(n) == 0) fail(std::string(where) + ": no boundary element has attribute " + std::to_string(a));
  }

  mfem::Array<int> marker(const std::vector<int>& attrs) const {
    mfem::Array<int> m(std::max(nattr_max(), 1));
    m = 0;
    for (int a : attrs)
      if (a >= 1 && a <= m.Size()) m[a - 1] = 1;
    return m;
  }

  std::unique_ptr<mfem::HypreParMatrix> parallel(mfem::SparseMatrix& loc) const {
    mfem::HypreParMatrix dA(pfes->GetComm(), pfes->GlobalVSize(), pfes->GetDofOffsets(), &loc);
    return std::unique_ptr<mfem::HypreParMatrix>(mfem::RAP(&dA, pfes->Dof_TrueDof_Matrix()));
  }

  void assemble() {
    if (assembled) return;
    if (!kappa_const_set && kappa_pts.size() == 0) fail("no conductivity set");
    Kloc.reset(new mfem::SparseMatrix(nl, nl));
    mfem::Vector Fvl(nl), Fbl(nl);
    Fvl = 0.0, Fbl = 0.0;
    int gp = 0;
    for (int e = 0; e < pmesh->GetNE(); ++e) {
      const mfem::FiniteElement* fe = pfes->GetFE(e);
      mfem::ElementTransformation* T = pmesh->GetElementTransformation(e);
      mfem::Array<int> dofs;
      pfes->GetElementDofs(e, dofs);
      const int nd = fe->GetDof();
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), deg_dom);
      mfem::Vector shape(nd);
      mfem::DenseMatrix ds(nd, 3), Ke(nd, nd);
      Ke = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q, ++gp) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        fe->CalcShape(ip, shape);
        fe->CalcPhysDShape(*T, ds);
        const double w = ip.weight * T->Weight();
        const double kap = kappa_pts.size() ? kappa_pts(gp) : kappa_const;
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b)
            Ke(a, b) += kap * w * (ds(a, 0) * ds(b, 0) + ds(a, 1) * ds(b, 1) + ds(a, 2) * ds(b, 2));
        if (f_pts.size())
          for (int a = 0; a < nd; ++a) Fvl(dofs[a]) += f_pts(gp) * shape(a) * w;
      }
      Kloc->AddSubMatrix(dofs, dofs, Ke);
    }
    Kloc->Finalize();
    for (auto& [attr, q] : flux) {
      auto bes = bdr_of({attr});
      if (q.size() != npts_of(bes)) fail("insulator flux of attribute " + std::to_string(attr) + ": size mismatch");
      int k = 0;
      for (int i : bes) {
        const BElem3& B = bdr[i];
        for (size_t p = 0; p < B.pts.size(); ++p, ++k)
          for (size_t a = 0; a < B.dofs.size(); ++a) Fbl(B.dofs[a]) += q(k) * B.shape[p][a] * B.pts[p].w;
      }
    }
    K = parallel(*Kloc);
    Kabs.reset(mfem::Add(1.0, *K, 0.0, *K));
    {
      hypre_ParCSRMatrix* h = *Kabs;
      for (hypre_CSRMatrix* c : {hypre_ParCSRMatrixDiag(h), hypre_ParCSRMatrixOffd(h)}) {
        double* d = hypre_CSRMatrixData(c);
        const int nnz = hypre_CSRMatrixNumNonzeros(c);
        for (int i = 0; i < nnz; ++i) d[i] = std::abs(d[i]);
      }
    }
    Fv.SetSize(nt), Fb.SetSize(nt);
    Fv = 0.0, Fb = 0.0;
    restrict_add(Fvl, Fv);
    restrict_add(Fbl, Fb);
    assembled = true;
  }

  void layout() {
    if (laid_out) return;
    if (el.empty()) fail("no electrode: add at least one electrode before solving");
    bool grounded = false;
    for (auto& E : el) grounded = grounded || !E.spec.galvanostatic;
    if (!grounded)
      fail("no grounded electrode: every electrode is galvanostatic, so phi is defined only up to a constant. Make "
           "one electrode potentiostatic (the ground)");
    std::vector<int> e_attr;
    std::map<int, double> val;
    for (auto& E : el)
      if (E.spec.primary)
        for (int a : E.spec.attrs) val[a] = E.spec.phi_s - E.spec.E_eq;
    // essential true dofs per primary electrode (a dof on two primary electrodes takes the later one)
    ess.SetSize(0);
    std::map<int, double> tv;
    for (auto& [a, v] : val) {
      mfem::Array<int> l;
      pfes->GetEssentialTrueDofs(marker({a}), l);
      for (int t : l) tv[t] = v;
    }
    ess_val.SetSize((int)tv.size());
    int i = 0;
    for (auto& [t, v] : tv) ess.Append(t), ess_val(i++) = v;
    ngalv = 0;
    for (auto& E : el)
      if (E.spec.galvanostatic) E.unknown = ngalv++;
    laid_out = true;
  }

  double phis_of(const Electrode3& E, const std::vector<double>& p) const {
    return E.spec.galvanostatic ? p[E.unknown] : E.spec.phi_s;
  }

  struct Eval {
    mfem::Vector Rfull, r, babs;   // T-vectors (r: ess rows zeroed)
    std::vector<double> g;         // bordered rows
    std::vector<double> G, dsum;   // per electrode (global)
    std::vector<mfem::Vector> m;   // per electrode: int di v (T-vector)
    std::unique_ptr<mfem::SparseMatrix> M;  // kinetic Jacobian (L-dofs)
    bool clamped = false;
    double field = 0, constraint = 0, relres = 0, merit = 0;
  };

  Eval evaluate(const mfem::Vector& xt, const std::vector<double>& p, const NewtonOptions& o, bool jac,
                int exclude = -1) const {
    Eval ev;
    mfem::Vector phl;
    prolong(xt, phl);
    ev.Rfull.SetSize(nt);
    K->Mult(xt, ev.Rfull);
    ev.Rfull -= Fv;
    ev.Rfull -= Fb;
    mfem::Vector bl(nl), babsl(nl);
    bl = 0.0, babsl = 0.0;
    ev.G.assign(el.size(), 0.0);
    ev.dsum.assign(el.size(), 0.0);
    ev.m.resize(el.size());
    if (jac) ev.M.reset(new mfem::SparseMatrix(nl, nl));
    const double f = kFaraday / (kGasConstant * o.temperature);   // F / (R T)
    bool clamped = false;
    for (int k = 0; k < (int)el.size(); ++k) {
      const Electrode3& E = el[k];
      if (E.spec.primary) continue;
      const double phis = phis_of(E, p);
      mfem::Vector bk(nl), mk(nl);
      bk = 0.0, mk = 0.0;
      int gq = 0;
      double Gl = 0, dl = 0;
      for (int i : E.belems) {
        const BElem3& B = bdr[i];
        const int nd = (int)B.dofs.size();
        mfem::DenseMatrix Me(nd, nd);
        Me = 0.0;
        for (size_t q = 0; q < B.pts.size(); ++q, ++gq) {
          const auto& sh = B.shape[q];
          double ph = 0;
          for (int a = 0; a < nd; ++a) ph += sh[a] * phl(B.dofs[a]);
          double cur = 0, dcur = 0, cabs = 0;
          for (const Reaction& r : E.spec.reactions) {
            double di = 0;
            const double ij = reaction_current(r, phis - ph - r.E_eq, f, o.clip, &di, &clamped);
            cur += ij, dcur += di, cabs += std::abs(ij);
          }
          if (!E.g.empty()) cur += E.g[gq], cabs += std::abs(E.g[gq]);
          const double w = B.pts[q].w;
          Gl += cur * w;
          dl += dcur * w;
          for (int a = 0; a < nd; ++a) {
            bk(B.dofs[a]) += cur * sh[a] * w;
            babsl(B.dofs[a]) += cabs * sh[a] * w;
            mk(B.dofs[a]) += dcur * sh[a] * w;
            if (jac)
              for (int c = 0; c < nd; ++c) Me(a, c) += dcur * sh[a] * sh[c] * w;
          }
        }
        if (jac) {
          mfem::Array<int> d(nd);
          for (int a = 0; a < nd; ++a) d[a] = B.dofs[a];
          ev.M->AddSubMatrix(d, d, Me);
        }
      }
      if (k != exclude) bl += bk;
      // the electrode current of the bordered row, reduced over ALL ranks
      const double Gt = gsum(Gl);
      ev.G[k] = Gt;
      ev.dsum[k] = gsum(dl);
      ev.m[k].SetSize(nt);
      ev.m[k] = 0.0;
      restrict_add(mk, ev.m[k]);
    }
    if (jac) ev.M->Finalize();
    ev.clamped = gor(clamped);
    mfem::Vector bt(nt);
    bt = 0.0;
    restrict_add(bl, bt);
    ev.Rfull -= bt;
    ev.babs.SetSize(nt);
    ev.babs = 0.0;
    restrict_add(babsl, ev.babs);
    ev.r = ev.Rfull;
    for (int t : ess) ev.r(t) = 0.0;
    ev.g.assign(ngalv, 0.0);
    double cons = 0;
    for (int k = 0; k < (int)el.size(); ++k) {
      const Electrode3& E = el[k];
      if (!E.spec.galvanostatic) continue;
      const double I = E.spec.current;
      ev.g[E.unknown] = ev.G[k] - I;
      cons = std::max(cons, std::abs(ev.g[E.unknown]) / std::abs(E.spec.current));
    }
    // relres (run.parameters.newton.relres) over the free true dofs
    mfem::Vector ax(xt), kp(nt);
    for (int i = 0; i < nt; ++i) ax(i) = std::abs(ax(i));
    Kabs->Mult(ax, kp);
    std::vector<char> isess(nt, 0);
    for (int t : ess) isess[t] = 1;
    double s[4] = {0, 0, 0, 0};
    for (int i = 0; i < nt; ++i) {
      if (isess[i]) continue;
      s[0] += ev.Rfull(i) * ev.Rfull(i);
      s[1] += kp(i) * kp(i);
      const double l = std::abs(Fv(i)) + std::abs(Fb(i));
      s[2] += l * l;
      s[3] += ev.babs(i) * ev.babs(i);
    }
    MPI_Allreduce(MPI_IN_PLACE, s, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double den = std::sqrt(s[1]) + std::sqrt(s[2]) + std::sqrt(s[3]);
    ev.field = den > 0 ? std::sqrt(s[0]) / den : std::sqrt(s[0]);
    ev.constraint = cons;
    ev.relres = std::max(ev.field, ev.constraint);
    double g2 = 0;
    for (double v : ev.g) g2 += v * v;
    ev.merit = std::sqrt(s[0] + g2);
    return ev;
  }

  // Jacobian on true dofs (not eliminated); caller owns
  std::unique_ptr<mfem::HypreParMatrix> jacobian(Eval& ev) const {
    mfem::SparseMatrix& M = *ev.M;
    auto Mp = parallel(M);
    return std::unique_ptr<mfem::HypreParMatrix>(mfem::ParAdd(K.get(), Mp.get()));
  }

  bool cg(mfem::HypreParMatrix& A, mfem::Solver& prec, const mfem::Vector& b, mfem::Vector& x) {
    mfem::CGSolver s(MPI_COMM_WORLD);
    s.SetRelTol(lin_rel);
    s.SetAbsTol(0.0);
    s.SetMaxIter(lin_maxit);
    s.SetPrintLevel(mfem::IterativeSolver::PrintLevel().None());
    s.SetPreconditioner(prec);
    s.SetOperator(A);
    s.iterative_mode = false;
    x.SetSize(nt);
    x = 0.0;
    if (gmax(b.Normlinf()) == 0.0) return true;
    s.Mult(b, x);
    lin_iters_max = std::max(lin_iters_max, s.GetNumIterations());
    lin_iters_total += s.GetNumIterations();
    // a CG that stalls at round-off is accepted when its residual is small (Newton iterates the rest)
    mfem::Vector r(nt);
    A.Mult(x, r);
    r -= b;
    const double rel = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, r, r) / mfem::InnerProduct(MPI_COMM_WORLD, b, b));
    lin_worst = std::max(lin_worst, rel);
    return gmax(x.CheckFinite()) == 0 && rel < 1e-6;
  }

  // Newton step (dx, dps) at ev: the bordered system by its Schur complement
  bool newton_step(Eval& ev, mfem::Vector& dx, std::vector<double>& dps) {
    auto J = jacobian(ev);
    // columns of the bordered rows: c_j = -int di v (ess rows zeroed)
    std::vector<mfem::Vector> C;
    for (int k = 0; k < (int)el.size(); ++k)
      if (el[k].spec.galvanostatic) {
        mfem::Vector c(ev.m[k]);
        c *= -1.0;
        for (int t : ess) c(t) = 0.0;
        C.push_back(c);
      }
    std::unique_ptr<mfem::HypreParMatrix> Ae(J->EliminateRowsCols(ess));
    mfem::HypreBoomerAMG amg(*J);
    amg.SetPrintLevel(0);
    mfem::Vector b(ev.r);
    b *= -1.0;
    for (int t : ess) b(t) = 0.0;
    if (!cg(*J, amg, b, dx)) return false;
    dps.assign(ngalv, 0.0);
    if (ngalv == 0) return true;
    std::vector<mfem::Vector> X(ngalv);
    for (int j = 0; j < ngalv; ++j)
      if (!cg(*J, amg, C[j], X[j])) return false;
    Eigen::MatrixXd S(ngalv, ngalv);
    Eigen::VectorXd rhs(ngalv);
    std::vector<double> dsj;
    for (int k = 0; k < (int)el.size(); ++k)
      if (el[k].spec.galvanostatic) dsj.push_back(ev.dsum[k]);
    for (int i = 0; i < ngalv; ++i) {
      rhs(i) = -ev.g[i] - mfem::InnerProduct(MPI_COMM_WORLD, C[i], dx);
      for (int j = 0; j < ngalv; ++j) S(i, j) = (i == j ? dsj[i] : 0.0) - mfem::InnerProduct(MPI_COMM_WORLD, C[i], X[j]);
    }
    const Eigen::VectorXd sol = S.lu().solve(rhs);
    for (int j = 0; j < ngalv; ++j) {
      dps[j] = sol(j);
      dx.Add(-sol(j), X[j]);
    }
    return true;
  }

  double max_deta(const mfem::Vector& dx, const std::vector<double>& dps) const {
    mfem::Vector dl;
    prolong(dx, dl);
    double mx = 0;
    for (const Electrode3& E : el) {
      if (E.spec.primary) continue;
      const double d0 = E.spec.galvanostatic ? dps[E.unknown] : 0.0;
      for (int i : E.belems) {
        const BElem3& B = bdr[i];
        for (size_t q = 0; q < B.pts.size(); ++q) {
          double d = 0;
          for (size_t a = 0; a < B.dofs.size(); ++a) d += B.shape[q][a] * dl(B.dofs[a]);
          mx = std::max(mx, std::abs(d0 - d));
        }
      }
    }
    return gmax(mx);
  }

  void initial() {
    x.SetSize(nt);
    x = 0.0;
    for (int i = 0; i < ess.Size(); ++i) x(ess[i]) = ess_val(i);
    ps.assign(ngalv, 0.0);
    for (auto& E : el)
      if (E.spec.galvanostatic) ps[E.unknown] = E.spec.phi_s;
  }
};

// ------------------------------------------------------------------------------------------------ construction
CurrentDistribution3D::CurrentDistribution3D(const Mesh3DInput& in, int order, int domain_degree, int boundary_degree)
    : p_(new Impl), order_(order), deg_dom_(domain_degree), deg_bdr_(boundary_degree) {
  ensure_mpi_chem();
  Impl& P = *p_;
  MPI_Comm_rank(MPI_COMM_WORLD, &P.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P.size);
  if (order != 1 && order != 2) fail("element order " + std::to_string(order) + " not supported: use 1 or 2");
  if (domain_degree < 2 * order || boundary_degree < 2 * order)
    fail("quadrature degrees below 2 x order: the default rule is 2k + 4");
  P.order = order, P.deg_dom = domain_degree, P.deg_bdr = boundary_degree;
  const int nv = (int)in.vertices.size() / 3, ne = (int)in.tets.size() / 4, nb = (int)in.boundary.size() / 3;
  if ((int)in.element_attr.size() != ne) fail("element_attr must have one entry per tetrahedron");
  if ((int)in.boundary_attr.size() != nb) fail("boundary_attr must have one entry per boundary triangle");
  if (nv < 4 || ne < 1 || nb < 1) fail("empty mesh");
  for (int i = 0; i < 3 * nv; ++i)
    if (!std::isfinite(in.vertices[i])) fail("vertex " + std::to_string(i / 3) + " has a non-finite coordinate");
  for (int i = 0; i < 4 * ne; ++i)
    if (in.tets[i] < 0 || in.tets[i] >= nv)
      fail("tetrahedron " + std::to_string(i / 4) + " names vertex " + std::to_string(in.tets[i]) + ", outside 0.." +
           std::to_string(nv - 1));
  for (int i = 0; i < 3 * nb; ++i)
    if (in.boundary[i] < 0 || in.boundary[i] >= nv)
      fail("boundary triangle " + std::to_string(i / 3) + " names vertex " + std::to_string(in.boundary[i]) +
           ", outside 0.." + std::to_string(nv - 1));
  {
    double hmax = 0;
    for (int e = 0; e < ne; ++e) {
      const double* X[4];
      for (int a = 0; a < 4; ++a) X[a] = &in.vertices[3 * in.tets[4 * e + a]];
      double u[3], v[3], w[3];
      for (int d = 0; d < 3; ++d) u[d] = X[1][d] - X[0][d], v[d] = X[2][d] - X[0][d], w[d] = X[3][d] - X[0][d];
      const double vol = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
                         u[2] * (v[0] * w[1] - v[1] * w[0]);
      double h = 0;
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
          h = std::max(h, std::sqrt((X[a][0] - X[b][0]) * (X[a][0] - X[b][0]) + (X[a][1] - X[b][1]) * (X[a][1] - X[b][1]) +
                                    (X[a][2] - X[b][2]) * (X[a][2] - X[b][2])));
      hmax = std::max(hmax, h);
      if (!(std::abs(vol) > 1e-12 * h * h * h))
        fail("tetrahedron " + std::to_string(e) + " is degenerate (volume " + std::to_string(vol / 6) +
             " for edge length " + std::to_string(h) + "): every tetrahedron needs a positive volume");
    }
  }
  {
    // every rank must have been given the same mesh (each builds its own copy before the partition)
    double sig[2] = {0, 0};
    for (int i = 0; i < 3 * nv; ++i) sig[0] += in.vertices[i] * (1.0 + 1e-3 * (i % 97));
    for (int i = 0; i < 4 * ne; ++i) sig[1] += (double)in.tets[i] * (1.0 + 1e-3 * (i % 89));
    double mn[2] = {sig[0], sig[1]}, mx[2] = {sig[0], sig[1]};
    MPI_Allreduce(MPI_IN_PLACE, mn, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mx, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    long n3[3] = {nv, ne, nb}, m3[3] = {nv, ne, nb};
    MPI_Allreduce(MPI_IN_PLACE, n3, 3, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, m3, 3, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (mn[0] != mx[0] || mn[1] != mx[1] || n3[0] != m3[0] || n3[1] != m3[1] || n3[2] != m3[2])
      fail("the MPI ranks were given different meshes: build the same mesh on every rank");
  }
  mfem::Mesh serial(3, nv, ne, nb, 3);
  for (int v = 0; v < nv; ++v) serial.AddVertex(in.vertices[3 * v], in.vertices[3 * v + 1], in.vertices[3 * v + 2]);
  for (int e = 0; e < ne; ++e) {
    if (in.element_attr[e] < 1) fail("element attributes must be >= 1");
    serial.AddTet(in.tets[4 * e], in.tets[4 * e + 1], in.tets[4 * e + 2], in.tets[4 * e + 3], in.element_attr[e]);
  }
  for (int b = 0; b < nb; ++b) {
    if (in.boundary_attr[b] < 1) fail("boundary attributes must be >= 1");
    serial.AddBdrTriangle(in.boundary[3 * b], in.boundary[3 * b + 1], in.boundary[3 * b + 2], in.boundary_attr[b]);
  }
  serial.FinalizeTetMesh(1, 0, true);
  const int nmid = (int)in.edge_node_vertices.size() / 2;
  if (nmid > 0) {
    if (order != 2) fail("curved edge nodes need order 2 (isoparametric P2)");
    serial.SetCurvature(2, false, 3, mfem::Ordering::byVDIM);
    std::map<std::pair<int, int>, int> key;
    for (int i = 0; i < nmid; ++i) {
      const int a = in.edge_node_vertices[2 * i], b = in.edge_node_vertices[2 * i + 1];
      key[{std::min(a, b), std::max(a, b)}] = i;
    }
    mfem::GridFunction* nodes = serial.GetNodes();
    const mfem::FiniteElementSpace* nf = nodes->FESpace();
    int used = 0;
    for (int ed = 0; ed < serial.GetNEdges(); ++ed) {
      mfem::Array<int> v;
      serial.GetEdgeVertices(ed, v);
      auto it = key.find({std::min(v[0], v[1]), std::max(v[0], v[1])});
      if (it == key.end()) continue;
      mfem::Array<int> d;
      nf->GetEdgeInteriorDofs(ed, d);
      if (d.Size() != 1) fail("unexpected number of P2 edge nodes");
      for (int c = 0; c < 3; ++c) (*nodes)(nf->DofToVDof(d[0], c)) = in.edge_node_xyz[3 * it->second + c];
      ++used;
    }
    if (used != nmid) fail(std::to_string(nmid - used) + " curved edge nodes name a vertex pair that is not an edge");
  }
  if (P.rank == 0) {
    P.V0.resize(serial.GetNV(), 3);
    for (int v = 0; v < serial.GetNV(); ++v)
      for (int d = 0; d < 3; ++d) P.V0(v, d) = serial.GetVertex(v)[d];
    P.E0.resize(serial.GetNE(), 4);
    for (int e = 0; e < serial.GetNE(); ++e) {
      mfem::Array<int> v;
      serial.GetElementVertices(e, v);
      for (int c = 0; c < 4; ++c) P.E0(e, c) = v[c];
    }
    if (serial.GetNodes()) {
      const mfem::GridFunction* g = serial.GetNodes();
      const int nn = g->FESpace()->GetNDofs();
      P.G0.resize(nn, 3);
      for (int i = 0; i < nn; ++i)
        for (int c = 0; c < 3; ++c) P.G0(i, c) = (*g)(g->FESpace()->DofToVDof(i, c));
    } else {
      P.G0 = P.V0;
    }
  }
  int* part = nullptr;
  if (P.size > 1) {  // METIS swaps process-wide signal handlers: under the lock every METIS call takes (sym_solver.hpp)
    std::lock_guard<std::mutex> metis(axf::metis_mutex());
    part = serial.GeneratePartitioning(P.size, 1);
  }
  P.pmesh.reset(new mfem::ParMesh(MPI_COMM_WORLD, serial, part));
  delete[] part;
  P.fec.reset(new mfem::H1_FECollection(order, 3));
  P.pfes.reset(new mfem::ParFiniteElementSpace(P.pmesh.get(), P.fec.get()));
  P.nl = P.pfes->GetVSize();
  P.nt = P.pfes->GetTrueVSize();
  for (int be = 0; be < P.pmesh->GetNBE(); ++be) P.bdr.push_back(P.make_belem(be, boundary_degree));
  P.eval_opt.temperature = 298.15;
}

CurrentDistribution3D::~CurrentDistribution3D() = default;

int CurrentDistribution3D::rank() const { return p_->rank; }
int CurrentDistribution3D::ranks() const { return p_->size; }

Eigen::MatrixXd CurrentDistribution3D::domain_points() const {
  const Impl& P = *p_;
  std::vector<double> xs;
  for (int e = 0; e < P.pmesh->GetNE(); ++e) {
    mfem::ElementTransformation* T = P.pmesh->GetElementTransformation(e);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(P.pmesh->GetElementBaseGeometry(e), deg_dom_);
    mfem::Vector xv(3);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      T->Transform(ir.IntPoint(q), xv);
      xs.insert(xs.end(), {xv(0), xv(1), xv(2)});
    }
  }
  Eigen::MatrixXd X(xs.size() / 3, 3);
  for (Eigen::Index i = 0; i < X.rows(); ++i) X(i, 0) = xs[3 * i], X(i, 1) = xs[3 * i + 1], X(i, 2) = xs[3 * i + 2];
  return X;
}

Eigen::VectorXi CurrentDistribution3D::domain_point_elements() const {
  const Impl& P = *p_;
  std::vector<int> id;
  for (int e = 0; e < P.pmesh->GetNE(); ++e) {
    const int n = mfem::IntRules.Get(P.pmesh->GetElementBaseGeometry(e), deg_dom_).GetNPoints();
    id.insert(id.end(), n, e);
  }
  return Eigen::Map<Eigen::VectorXi>(id.data(), (Eigen::Index)id.size());
}

Eigen::MatrixXd CurrentDistribution3D::element_centroids() const {
  const Impl& P = *p_;
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(P.pmesh->GetNE(), 3);
  for (int e = 0; e < P.pmesh->GetNE(); ++e) {
    mfem::Array<int> v;
    P.pmesh->GetElementVertices(e, v);
    for (int a : v)
      for (int d = 0; d < 3; ++d) C(e, d) += P.pmesh->GetVertex(a)[d] / v.Size();
  }
  return C;
}

Eigen::MatrixXd CurrentDistribution3D::boundary_points(const std::vector<int>& attrs) const {
  const Impl& P = *p_;
  auto bes = P.bdr_of(attrs);
  Eigen::MatrixXd X(P.npts_of(bes), 3);
  int k = 0;
  for (int i : bes)
    for (const BPoint3& q : P.bdr[i].pts) X(k, 0) = q.x[0], X(k, 1) = q.x[1], X(k, 2) = q.x[2], ++k;
  return X;
}

void CurrentDistribution3D::set_conductivity_constant(double kappa) {
  if (!(kappa > 0)) fail("conductivity must be > 0");
  p_->kappa_const = kappa;
  p_->kappa_const_set = true;
  p_->kappa_pts.resize(0);
  p_->assembled = false;
}
void CurrentDistribution3D::set_conductivity_at_points(const Eigen::VectorXd& kappa) {
  if (kappa.size() != p_->domain_point_count()) fail("set_conductivity_at_points: size differs from domain_points()");
  if (kappa.size() && !(kappa.minCoeff() > 0)) fail("set_conductivity_at_points: conductivity must be > 0");
  p_->kappa_pts = kappa;
  p_->kappa_const_set = false;
  p_->assembled = false;
}
void CurrentDistribution3D::set_source_at_points(const Eigen::VectorXd& f) {
  if (f.size() != p_->domain_point_count()) fail("set_source_at_points: size differs from domain_points()");
  p_->f_pts = f;
  p_->assembled = false;
}
void CurrentDistribution3D::set_insulator_flux(int attr, const Eigen::VectorXd& q) {
  Impl& P = *p_;
  P.require_attr_global(attr, "set_insulator_flux");
  if (q.size() != P.npts_of(P.bdr_of({attr}))) fail("set_insulator_flux: size differs from boundary_points([attr])");
  for (const auto& E : P.el)
    if (std::find(E.spec.attrs.begin(), E.spec.attrs.end(), attr) != E.spec.attrs.end())
      fail("set_insulator_flux: attribute " + std::to_string(attr) + " belongs to electrode '" + E.spec.name + "'");
  P.flux[attr] = q;
  P.assembled = false;
}

int CurrentDistribution3D::add_electrode(const ElectrodeSpec& s) {
  Impl& P = *p_;
  if (P.laid_out) fail("add_electrode after the first solve: add every electrode before solving");
  const std::string who = "electrode '" + s.name + "': ";
  if (s.attrs.empty()) fail(who + "no boundary attributes given");
  for (int a : s.attrs) {
    P.require_attr_global(a, "add_electrode");
    if (P.flux.count(a)) fail(who + "attribute " + std::to_string(a) + " already carries an insulator flux");
    for (const auto& O : P.el)
      if (std::find(O.spec.attrs.begin(), O.spec.attrs.end(), a) != O.spec.attrs.end())
        fail("boundary attribute " + std::to_string(a) + " belongs to electrodes '" + O.spec.name + "' and '" + s.name + "'");
  }
  if (s.primary && !s.reactions.empty()) fail(who + "a primary electrode has no reactions");
  if (s.primary && s.galvanostatic)
    fail(who + "a galvanostatic primary electrode (floating conductor) is not available in 3D: use kinetics");
  if (!s.primary && s.reactions.empty()) fail(who + "a kinetic electrode needs reactions (or primary=true)");
  for (const Reaction& r : s.reactions) {
    if (!(r.i0 > 0)) fail(who + "i0 must be > 0");
    if ((r.kind == Reaction::Kind::BV || r.kind == Reaction::Kind::Linear) &&
        !(r.alpha_a >= 0 && r.alpha_c >= 0 && r.alpha_a + r.alpha_c > 0))
      fail(who + "transfer coefficients must be >= 0 with a positive sum");
    if (r.kind == Reaction::Kind::TafelAnodic && !(r.A > 0)) fail(who + "tafel_a needs A > 0");
    if (r.kind == Reaction::Kind::TafelCathodic && !(r.A < 0)) fail(who + "tafel_c needs A < 0");
  }
  if (s.galvanostatic && s.current == 0.0) fail(who + "galvanostatic with zero current");
  Electrode3 E;
  E.spec = s;
  E.belems = P.bdr_of(s.attrs);
  if (!s.g.empty() && (int)s.g.size() != P.npts_of(E.belems))
    fail(who + "g has " + std::to_string(s.g.size()) + " values, boundary_points has " +
         std::to_string(P.npts_of(E.belems)));
  E.g = s.g;
  P.el.push_back(E);
  return (int)P.el.size() - 1;
}

void CurrentDistribution3D::set_control(int e, double value) {
  Impl& P = *p_;
  if (e < 0 || e >= (int)P.el.size())
    fail("set_control: electrode index " + std::to_string(e) + " out of range (" + std::to_string(P.el.size()) +
         " electrodes)");
  if (!std::isfinite(value)) fail("set_control: value must be finite");
  Electrode3& E = P.el[e];
  if (E.spec.galvanostatic) {
    if (value == 0.0) fail("set_control: electrode '" + E.spec.name + "' is galvanostatic: its current must be != 0");
    E.spec.current = value;
  } else {
    E.spec.phi_s = value;
    if (E.spec.primary && P.laid_out) P.laid_out = false;   // the Dirichlet values are rebuilt at the next solve
  }
}

void CurrentDistribution3D::set_linear_tolerance(double rel, int max_iter) {
  if (!(rel > 0) || max_iter < 1) fail("set_linear_tolerance: need rel > 0, max_iter >= 1");
  p_->lin_rel = rel, p_->lin_maxit = max_iter;
}

// ------------------------------------------------------------------------------------------------ Newton
SolveReport CurrentDistribution3D::solve(const NewtonOptions& oin) {
  const auto t0 = std::chrono::steady_clock::now();
  Impl& P = *p_;
  NewtonOptions o = oin;
  if (o.temperature == 0.0) o.temperature = 298.15;
  if (o.max_iter < 0 || !(o.stag >= 0) || !(o.gate > 0) || !(o.ls_min > 0 && o.ls_min <= 1) || !(o.clip > 0))
    fail("solve: invalid Newton options");
  if (o.initial != "zero" && o.initial != "state")
    fail("solve: initial must be 'zero' (phi = 0, the default start) or 'state' (warm start from set_state)");
  P.assemble();
  P.layout();
  P.eval_opt = o;
  P.lin_iters_max = 0, P.lin_iters_total = 0, P.lin_worst = 0;
  if (o.initial == "zero" || P.x.Size() != P.nt) {
    P.initial();
  } else {   // warm start: keep the state, impose the current Dirichlet values (set_control may have changed them)
    for (int i = 0; i < P.ess.Size(); ++i) P.x(P.ess[i]) = P.ess_val(i);
    if ((int)P.ps.size() != P.ngalv) P.initial();
  }
  SolveReport R;
  R.unknowns = (int)std::min<long>(global_true_dofs() + P.ngalv, std::numeric_limits<int>::max());
  R.initial = o.initial;
  mfem::Vector x(P.x);
  std::vector<double> ps = P.ps;
  auto ev0 = P.evaluate(x, ps, o, false);
  R.residual_history.push_back(ev0.relres);
  const double lim = o.limiter_rt_f > 0 ? o.limiter_rt_f / (kFaraday / (kGasConstant * o.temperature)) : 0.0;
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
    auto evj = P.evaluate(x, ps, o, true);
    mfem::Vector dx;
    std::vector<double> dps;
    if (!P.newton_step(evj, dx, dps)) {
      R.stop = "linear_solver_failed";
      break;
    }
    double t = 1.0, rn;
    const double md = lim > 0 ? P.max_deta(dx, dps) : 0.0;
    mfem::Vector xn(P.nt);
    std::vector<double> pn(ps);
    auto apply = [&](double tt) {
      xn = x;
      xn.Add(tt, dx);
      for (int j = 0; j < P.ngalv; ++j) pn[j] = ps[j] + tt * dps[j];
    };
    if (lim > 0 && md > lim) {
      t = lim / md;
      apply(t);
      rn = P.evaluate(xn, pn, o, false).relres;
      ++R.limited_steps;
    } else {
      const double r0 = evj.merit;
      while (true) {
        apply(t);
        const auto evn = P.evaluate(xn, pn, o, false);
        rn = evn.relres;
        if (evn.merit < (1.0 - 1e-4 * t) * r0 || t <= o.ls_min) break;
        t *= 0.5;
        ++R.line_search_halvings;
      }
    }
    x = xn;
    ps = pn;
    R.step_lengths.push_back(t);
    R.residual_history.push_back(rn);
    R.iterations = it + 1;
  }
  P.x = x;
  P.ps = ps;
  auto fin = P.evaluate(x, ps, o, false);
  R.relative_residual = fin.relres;
  R.field_residual = fin.field;
  R.constraint_residual = fin.constraint;
  R.clamp_active = fin.clamped;
  R.converged = std::isfinite(fin.relres) && fin.relres <= o.gate && !fin.clamped;
  R.factor_nonzeros = P.lin_iters_max;   // 3D: the largest CG iteration count (no factorization)
  R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report_ = R;
  return R;
}

Eigen::VectorXd CurrentDistribution3D::state() const {
  const Impl& P = *p_;
  if (P.x.Size() != P.nt) fail("no state: call solve() or set_state() first");
  Eigen::VectorXd z(P.nt + P.ngalv);
  for (int i = 0; i < P.nt; ++i) z(i) = P.x(i);
  for (int j = 0; j < P.ngalv; ++j) z(P.nt + j) = P.ps[j];
  return z;
}
void CurrentDistribution3D::set_state(const Eigen::VectorXd& z) {
  Impl& P = *p_;
  P.assemble();
  P.layout();
  if (z.size() != P.nt + P.ngalv) fail("set_state: wrong size (owned true dofs + galvanostatic phi_s)");
  P.x.SetSize(P.nt);
  for (int i = 0; i < P.nt; ++i) P.x(i) = z(i);
  P.ps.assign(P.ngalv, 0.0);
  for (int j = 0; j < P.ngalv; ++j) P.ps[j] = z(P.nt + j);
}
std::vector<double> CurrentDistribution3D::phi_s() const {
  const Impl& P = *p_;
  if (!P.laid_out || P.x.Size() != P.nt || (int)P.ps.size() != P.ngalv)
    fail("phi_s: no state: call solve() or set_state() first");
  std::vector<double> out;
  for (auto& E : P.el) out.push_back(E.spec.galvanostatic ? P.ps.at(E.unknown) : E.spec.phi_s);
  return out;
}
Eigen::VectorXd CurrentDistribution3D::newton_update() {
  Impl& P = *p_;
  if (P.x.Size() != P.nt) fail("newton_update: no state");
  auto ev = P.evaluate(P.x, P.ps, P.eval_opt, true);
  mfem::Vector dx;
  std::vector<double> dps;
  if (!P.newton_step(ev, dx, dps)) fail("newton_update: the linear solve failed");
  Eigen::VectorXd z = state();
  for (int i = 0; i < P.nt; ++i) z(i) += dx(i);
  for (int j = 0; j < P.ngalv; ++j) z(P.nt + j) += dps[j];
  return z;
}
Eigen::VectorXd CurrentDistribution3D::residual(const Eigen::VectorXd& z) {
  Impl& P = *p_;
  set_state(z);
  auto ev = P.evaluate(P.x, P.ps, P.eval_opt, false);
  Eigen::VectorXd r(P.nt + P.ngalv);
  for (int i = 0; i < P.nt; ++i) r(i) = ev.r(i);
  for (int j = 0; j < P.ngalv; ++j) r(P.nt + j) = ev.g[j];
  return r;
}
Eigen::VectorXd CurrentDistribution3D::jacobian_apply(const Eigen::VectorXd& z, const Eigen::VectorXd& v) {
  Impl& P = *p_;
  set_state(z);
  if (v.size() != z.size()) fail("jacobian_apply: v and z differ in size");
  auto ev = P.evaluate(P.x, P.ps, P.eval_opt, true);
  auto J = P.jacobian(ev);
  mfem::Vector vx(P.nt), y(P.nt);
  for (int i = 0; i < P.nt; ++i) vx(i) = v(i);
  J->Mult(vx, y);
  Eigen::VectorXd out(P.nt + P.ngalv);
  int j = 0;
  std::vector<double> gy(P.ngalv, 0.0);
  for (int k = 0; k < (int)P.el.size(); ++k) {
    if (!P.el[k].spec.galvanostatic) continue;
    const double vp = v(P.nt + P.el[k].unknown);
    y.Add(-vp, ev.m[k]);   // d r / d phi_s = -int di v
    gy[P.el[k].unknown] = -mfem::InnerProduct(MPI_COMM_WORLD, ev.m[k], vx) + ev.dsum[k] * vp;
    ++j;
  }
  for (int t : P.ess) y(t) = 0.0;
  for (int i = 0; i < P.nt; ++i) out(i) = y(i);
  for (int g = 0; g < P.ngalv; ++g) out(P.nt + g) = gy[g];
  return out;
}
int CurrentDistribution3D::num_true_dofs() const { return p_->nt; }
long CurrentDistribution3D::global_true_dofs() const { return (long)p_->pfes->GlobalTrueVSize(); }

// ------------------------------------------------------------------------------------------------ raw data
DomainQuadrature3D CurrentDistribution3D::domain_quadrature(int degree, int e0, int e1) const {
  const Impl& P = *p_;
  if (P.x.Size() != P.nt) fail("domain_quadrature: no state");
  mfem::Vector phl;
  P.prolong(P.x, phl);
  const int ne = P.pmesh->GetNE();
  if (e1 < 0 || e1 > ne) e1 = ne;
  e0 = std::max(0, std::min(e0, e1));
  int n = 0;
  for (int e = e0; e < e1; ++e) n += mfem::IntRules.Get(P.pmesh->GetElementBaseGeometry(e), degree).GetNPoints();
  DomainQuadrature3D D;
  D.x.resize(n, 3), D.grad.resize(n, 3), D.w.resize(n), D.phi.resize(n);
  int k = 0;
  for (int e = e0; e < e1; ++e) {
    const mfem::FiniteElement* fe = P.pfes->GetFE(e);
    mfem::ElementTransformation* T = P.pmesh->GetElementTransformation(e);
    mfem::Array<int> dofs;
    P.pfes->GetElementDofs(e, dofs);
    const int nd = fe->GetDof();
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), degree);
    mfem::Vector shape(nd), xv(3);
    mfem::DenseMatrix ds(nd, 3);
    for (int q = 0; q < ir.GetNPoints(); ++q, ++k) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*T, ds);
      T->Transform(ip, xv);
      double v = 0, g[3] = {0, 0, 0};
      for (int a = 0; a < nd; ++a) {
        const double c = phl(dofs[a]);
        v += shape(a) * c;
        for (int d = 0; d < 3; ++d) g[d] += ds(a, d) * c;
      }
      for (int d = 0; d < 3; ++d) D.x(k, d) = xv(d), D.grad(k, d) = g[d];
      D.w(k) = ip.weight * T->Weight();
      D.phi(k) = v;
    }
  }
  return D;
}

BoundaryQuadrature3D CurrentDistribution3D::boundary_quadrature(const std::vector<int>& attrs, int degree) const {
  Impl& P = *p_;
  if (P.x.Size() != P.nt) fail("boundary_quadrature: no state");
  mfem::Vector phl;
  P.prolong(P.x, phl);
  std::vector<BElem3> bes;
  for (int be = 0; be < P.pmesh->GetNBE(); ++be)
    if (std::find(attrs.begin(), attrs.end(), P.pmesh->GetBdrAttribute(be)) != attrs.end())
      bes.push_back(P.make_belem(be, degree));
  int n = 0;
  for (auto& B : bes) n += (int)B.pts.size();
  BoundaryQuadrature3D Q;
  Q.x.resize(n, 3), Q.grad.resize(n, 3), Q.normal.resize(n, 3), Q.w.resize(n), Q.phi.resize(n);
  int k = 0;
  for (auto& B : bes) {
    const mfem::FiniteElement* fe = P.pfes->GetFE(B.elem);
    const int nd = fe->GetDof();
    mfem::FaceElementTransformations* FT = P.pmesh->GetBdrFaceTransformations(B.be);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(P.pmesh->GetBdrElementGeometry(B.be), degree);
    mfem::DenseMatrix ds(nd, 3);
    for (int q = 0; q < ir.GetNPoints(); ++q, ++k) {
      FT->SetAllIntPoints(&ir.IntPoint(q));
      fe->CalcPhysDShape(*FT->Elem1, ds);
      double v = 0, g[3] = {0, 0, 0};
      for (int a = 0; a < nd; ++a) {
        const double c = phl(B.dofs[a]);
        v += B.shape[q][a] * c;
        for (int d = 0; d < 3; ++d) g[d] += ds(a, d) * c;
      }
      const BPoint3& p = B.pts[q];
      for (int d = 0; d < 3; ++d) Q.x(k, d) = p.x[d], Q.grad(k, d) = g[d], Q.normal(k, d) = p.n[d];
      Q.w(k) = p.w, Q.phi(k) = v;
    }
  }
  return Q;
}

Eigen::VectorXd CurrentDistribution3D::residual_full(int exclude) const {
  const Impl& P = *p_;
  if (P.x.Size() != P.nt || !P.assembled) fail("residual_full: no state: call solve() first");
  auto ev = P.evaluate(P.x, P.ps, P.eval_opt, false, exclude);
  Eigen::VectorXd r(P.nt);
  for (int i = 0; i < P.nt; ++i) r(i) = ev.Rfull(i);
  return r;
}

std::vector<int> CurrentDistribution3D::attr_true_dofs(const std::vector<int>& attrs) const {
  mfem::Array<int> l;
  p_->pfes->GetEssentialTrueDofs(p_->marker(attrs), l);
  return std::vector<int>(l.begin(), l.end());
}

int CurrentDistribution3D::local_elements() const { return p_->pmesh->GetNE(); }
long CurrentDistribution3D::global_elements() const { return (long)p_->pmesh->GetGlobalNE(); }
std::vector<int> CurrentDistribution3D::owned_cells_per_rank() const {
  std::vector<int> out(p_->size);
  int n = p_->pmesh->GetNE();
  MPI_Allgather(&n, 1, MPI_INT, out.data(), 1, MPI_INT, MPI_COMM_WORLD);
  return out;
}
double CurrentDistribution3D::h_max() const {
  const Impl& P = *p_;
  double h = 0;
  for (int e = 0; e < P.pmesh->GetNE(); ++e) {
    mfem::Array<int> v;
    P.pmesh->GetElementVertices(e, v);
    for (int a = 0; a < v.Size(); ++a)
      for (int b = a + 1; b < v.Size(); ++b) {
        const double* x = P.pmesh->GetVertex(v[a]);
        const double* y = P.pmesh->GetVertex(v[b]);
        h = std::max(h, std::sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
                                  (x[2] - y[2]) * (x[2] - y[2])));
      }
  }
  return gmax(h);
}
double CurrentDistribution3D::volume() const {
  double s = 0;
  for (int e = 0; e < p_->pmesh->GetNE(); ++e) s += p_->pmesh->GetElementVolume(e);
  return gsum(s);
}
Eigen::MatrixXd CurrentDistribution3D::mesh_vertices() const { return p_->V0; }
Eigen::MatrixXi CurrentDistribution3D::mesh_elements() const { return p_->E0; }
Eigen::MatrixXd CurrentDistribution3D::geometry_nodes() const { return p_->G0; }
std::map<std::string, double> CurrentDistribution3D::diagnostics() const {
  const Impl& P = *p_;
  return {{"dofs_field", (double)global_true_dofs()},
          {"unknowns", (double)(global_true_dofs() + P.ngalv)},
          {"n_elements", (double)global_elements()},
          {"ranks", (double)P.size},
          {"cg_iterations_max", (double)P.lin_iters_max},
          {"cg_iterations_total", (double)P.lin_iters_total},
          {"cg_worst_relres", P.lin_worst},
          {"cg_rel_tol", P.lin_rel},
          {"domain_degree", (double)deg_dom_},
          {"boundary_degree", (double)deg_bdr_}};
}

}  // namespace axf::chem

#endif  // AXF_NO_MPI
