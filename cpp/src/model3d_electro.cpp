// 3D electrostatics: Maxwell capacitance matrix of the conductor bodies, pairwise capacitances and
// the readout-port gap capacitance.  Same design key (`electrostatics`) and the same result schema as the
// axisymmetric solver (cpp/src/axi_electro.cpp; docs/physics/electrostatics.md#3d, docs/design_format.md,
// docs/validation/electrostatics3d.md); the formulation and the choices specific to 3D follow.
//
// Formulation.  div(eps grad phi) = 0 on the meshed domain, H1 elements of order mesh.order on the same
// ParMesh as the magnetoquasistatic solve.
//
//  * Conductors (the connected bodies of `superconductor` and `conductor` elements) are equipotentials:
//    every degree of freedom of a body is prescribed, one unit-potential solve per body, so the Maxwell
//    matrix needs nbody solves and one BoomerAMG setup.  Charges come from the reactions
//    Q_i = u_i . (K phi), u_i the indicator of body i -- a single global reduction, so the result does not
//    depend on the partition.
//  * The open boundary is the existing infinity shell [R1, R2) that the 3D mesher always builds, mapped
//    onto [R1, infinity) by rho' = f(rho) = R1 (R2 - R1)/(R2 - rho) about the domain centre.  For the
//    scalar potential the exterior becomes the anisotropic permittivity (Jacobian J = diag(f', f/rho, f/rho),
//    eps_eff = eps det J J^-1 J^-T)
//        eps_eff = eps_out [ (f^2 / (f' rho^2)) r r^T + f' (I - r r^T) ],
//    with phi = 0 on the outer sphere of the shell (= phi -> 0 at infinity). The monopole
//    phi ~ 1/f is linear in rho in the continuum. Radius is not a Cartesian polynomial on
//    tetrahedra, so its finite-element representation requires h/p convergence. This replaces
//    the Laplace Dirichlet-to-Neumann series of the axisymmetric solver,
//    whose n = 0 term plays the same role.
//    `boundary: "grounded"` instead leaves the shell out of the domain and puts phi = 0 on the sphere at
//    mesh.domain_radius.
//  * A body may be declared `insulating`: its elements are left out of the domain and its surface carries
//    the natural condition n.D = 0.  This models the readout gap, where the two terminal patches of ONE
//    conductor body are driven against each other (see the axisymmetric note for why the gap capacitance
//    is a lumped-model quantity).
#include <mfem.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "model3d_impl.hpp"

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/spec.hpp"
#include "axionfem/units.hpp"

namespace axf {

namespace {

using units::pi;

double gsum(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
double gdot3(const mfem::Vector& a, const mfem::Vector& b) { return mfem::InnerProduct(MPI_COMM_WORLD, a, b); }
double gnorm3(const mfem::Vector& a) { return std::sqrt(std::max(0.0, gdot3(a, a))); }

// Role of one (serial) element in the electrostatic problem.
struct Role3 {
  int body = -1;            // >= 0: inside conductor body `body` (equipotential, element not assembled)
  bool insulating = false;  // left out of the domain with the natural condition n.D = 0 on its surface
  double eps = 0;           // permittivity [F/m] of a dielectric element
  bool shell = false;       // element of the infinity shell (mapped exterior)
};

enum class Exterior { None, Open, Grounded };

// Permittivity of the computational domain: eps(element) I, the mapped tensor in the infinity shell, and a
// floor of 1e-12 eps0 in conductors, insulating bodies and (when they are not mapped) shell elements.
//
// The floor is a numerical device, not physics: an element left out of the domain altogether would give
// empty matrix rows for the degrees of freedom it alone carries, and hypre's row/column elimination only
// sets the unit diagonal of a row that already has a diagonal entry.  It is harmless because
//  * no free degree of freedom belongs to a conductor element (all of them are tied), so the conductor
//    floor never touches a free row, and its contribution to a reaction vanishes with the element row sums;
//  * a free degree of freedom on the surface of an insulating body gains a coupling of relative size 1e-12
//    to that body's interior, which is prescribed to zero.
// Indexed through loc2ser, so the roles are the serial ones.
class EpsTensor : public mfem::MatrixCoefficient {
 public:
  EpsTensor(const std::vector<Role3>& roles, const std::vector<int>& loc2ser, double R1, double R2,
            const double c[3], double eps_out, bool mapped_shell)
      : mfem::MatrixCoefficient(3), roles_(roles), l2s_(loc2ser), R1_(R1), R2_(R2), eps_out_(eps_out),
        floor_(1e-12 * eps_out), mapped_(mapped_shell) {
    for (int k = 0; k < 3; ++k) c_[k] = c[k];
  }
  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    K.SetSize(3);
    K = 0.0;
    const Role3& r = roles_[l2s_[T.ElementNo]];
    if (r.body >= 0 || r.insulating) {
      for (int i = 0; i < 3; ++i) K(i, i) = floor_;
      return;
    }
    if (r.shell) {
      if (!mapped_) {
        for (int i = 0; i < 3; ++i) K(i, i) = floor_;
        return;
      }
      mfem::Vector x(3);
      T.Transform(ip, x);
      const double d[3] = {x(0) - c_[0], x(1) - c_[1], x(2) - c_[2]};
      const double rho = std::max(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), 1e-300);
      const double gap = std::max(R2_ - rho, 1e-12 * R2_);
      const double f = R1_ * (R2_ - R1_) / gap, fp = f / gap;
      const double ar = f * f / (fp * rho * rho), at = fp;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const double rr = d[i] * d[j] / (rho * rho);
          K(i, j) = eps_out_ * (ar * rr + at * ((i == j ? 1.0 : 0.0) - rr));
        }
      return;
    }
    for (int i = 0; i < 3; ++i) K(i, i) = r.eps;
  }

 private:
  const std::vector<Role3>& roles_;
  const std::vector<int>& l2s_;
  double R1_, R2_, c_[3], eps_out_, floor_;
  bool mapped_;
};

// ----------------------------------------------------------------------------------------------------
// The parallel electrostatic problem.  Roles are indexed by SERIAL elements (so that topology and the
// surface integrals are rank independent); the solve happens on the ParMesh.
class Electro3D {
 public:
  Electro3D(Model3D::Impl& P, std::vector<Role3> roles, int nbody, Exterior ext, double eps_out)
      : P_(P), roles_(std::move(roles)), nbody_(nbody), ext_(ext), eps_out_(eps_out) {
    AXF_REQUIRE(static_cast<int>(roles_.size()) == P_.serial->GetNE(),
                "internal: electrostatic role table size mismatch");
    for (auto& r : roles_) AXF_REQUIRE(r.body < nbody_, "internal: electrostatic role refers to an unknown conductor");
  }

  void set_names(std::vector<std::string> n) { names_ = std::move(n); }
  // Tie the degrees of freedom of a set of serial faces to a body (readout terminal patches).
  void add_terminal(int body, const std::vector<int>& serial_faces) {
    AXF_REQUIRE(body >= 0 && body < nbody_, "internal: terminal conductor out of range");
    for (int f : serial_faces) term_body_[f] = body;
  }

  struct Solution {
    mfem::Vector Phi;                             // true-dof potential [V]
    std::vector<double> V, Q, Q_gauss;            // per body: potential [V], charge [C] (reactions, Gauss)
    double energy = 0, energy_field = 0;          // stored energy [J]: 1/2 phi^T K phi, and by quadrature
    double flux_exterior = 0;                     // charge [C] from the flux through the sphere at R1
    double residual = 0;
    int iterations = 0;
    std::unique_ptr<mfem::GridFunction> serial;   // gathered potential on the serial mesh (every rank)
  };

  void setup();
  // Maxwell capacitance matrix C(i, j) = Q_i for V_j = 1 V, every other conductor at 0 V.
  Eigen::MatrixXd capacitance(std::vector<Solution>* cols);
  Solution solve_prescribed(const std::vector<double>& V);

  double energy_quadrature(const mfem::Vector& Phi) const;
  std::vector<double> charges_gauss(const mfem::GridFunction& phi) const;
  double exterior_flux(const mfem::GridFunction& phi) const;
  std::unique_ptr<mfem::GridFunction> gather(const mfem::Vector& Phi) const;
  mfem::FiniteElementSpace& serial_space() const { return *sh1_; }
  const json& diagnostics() const { return diag_; }
  int nbody() const { return nbody_; }

 private:
  std::string bname(int b) const {
    if (b >= 0 && b < static_cast<int>(names_.size()) && !names_[b].empty()) return "'" + names_[b] + "'";
    return strcat("#", b);
  }

  Model3D::Impl& P_;
  std::vector<Role3> roles_;
  int nbody_ = 0;
  Exterior ext_ = Exterior::Open;
  double eps_out_ = 0;
  std::vector<std::string> names_;
  std::map<int, int> term_body_;  // serial face -> body

  bool ready_ = false;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> h1_;
  std::unique_ptr<mfem::FiniteElementSpace> sh1_;  // serial, for the gathered potential
  std::unique_ptr<EpsTensor> eps_;
  std::unique_ptr<mfem::ParBilinearForm> form_;
  std::unique_ptr<mfem::HypreParMatrix> K_, Kel_, Ke_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::HyprePCG> pcg_;
  mfem::Array<int> ess_;
  std::vector<std::vector<int>> bodyT_;  // per body: owned true dofs
  std::vector<mfem::Vector> U_;          // per body: indicator true-dof vector
  std::vector<int> kind_, tie_;          // per local vdof: 1 free, 2 tied, 3 fixed to zero
  json diag_;
  int total_its_ = 0;
  double max_res_ = 0;
};

void Electro3D::setup() {
  if (ready_) return;
  Timer timer;
  mfem::ParMesh& pm = *P_.pmesh;
  mfem::Mesh& sm = *P_.serial;
  const int ne = pm.GetNE(), nser = sm.GetNE();
  const std::vector<int>& l2s = P_.loc2ser;
  auto assembled = [&](int s) {
    const Role3& r = roles_[s];
    return r.body < 0 && !r.insulating && (!r.shell || ext_ == Exterior::Open);
  };

  fec_ = std::make_unique<mfem::H1_FECollection>(P_.order, 3);
  h1_ = std::make_unique<mfem::ParFiniteElementSpace>(&pm, fec_.get());
  sh1_ = std::make_unique<mfem::FiniteElementSpace>(&sm, fec_.get());
  mfem::ParFiniteElementSpace& h1 = *h1_;
  const int nd = h1.GetVSize();

  // ------------------------------------------------------------------ DOF classes
  // Per-body markers (OR-synchronized across ranks), so that a DOF shared with a body element on another
  // rank is tied here as well.
  std::vector<mfem::Array<int>> bmark(nbody_);
  for (int b = 0; b < nbody_; ++b) {
    bmark[b].SetSize(nd);
    bmark[b] = 0;
  }
  mfem::Array<int> dofs, lf, lo, sf, so;
  for (int e = 0; e < ne; ++e) {
    const int b = roles_[l2s[e]].body;
    if (b < 0) continue;
    h1.GetElementDofs(e, dofs);
    for (int dd : dofs) bmark[b][dd] = 1;
  }
  if (!term_body_.empty()) {  // readout terminal patches: serial faces -> local faces (same element order)
    for (int e = 0; e < ne; ++e) {
      pm.GetElementFaces(e, lf, lo);
      sm.GetElementFaces(l2s[e], sf, so);
      for (int k = 0; k < sf.Size(); ++k) {
        auto it = term_body_.find(sf[k]);
        if (it == term_body_.end()) continue;
        h1.GetFaceDofs(lf[k], dofs);
        for (int dd : dofs) bmark[it->second][dd] = 1;
      }
    }
  }
  for (int b = 0; b < nbody_; ++b) h1.Synchronize(bmark[b]);
  tie_.assign(nd, -1);
  for (int b = 0; b < nbody_; ++b)
    for (int i = 0; i < nd; ++i) {
      if (!bmark[b][i]) continue;
      if (tie_[i] >= 0 && tie_[i] != b)
        throw Error(strcat("the conductors ", bname(tie_[i]), " and ", bname(b), " touch"), "/regions",
                    "Touching conductors form one body; separate them or merge them into one region.");
      tie_[i] = b;
    }
  // ground: the outer sphere of the shell (open) or the sphere at mesh.domain_radius (grounded)
  mfem::Array<int> gmark(nd), amark(nd), smark(nd);
  gmark = 0;
  amark = 0;
  smark = 0;
  for (int e = 0; e < ne; ++e) {
    h1.GetElementDofs(e, dofs);
    if (assembled(l2s[e]))
      for (int dd : dofs) amark[dd] = 1;
    if (roles_[l2s[e]].shell)
      for (int dd : dofs) smark[dd] = 1;
  }
  h1.Synchronize(amark);
  h1.Synchronize(smark);
  if (ext_ == Exterior::Open) {
    for (int be = 0; be < pm.GetNBE(); ++be) {
      if (pm.GetBdrAttribute(be) != axi::kOuterAttr) continue;
      h1.GetBdrElementDofs(be, dofs);
      for (int dd : dofs) gmark[dd] = 1;
    }
    h1.Synchronize(gmark);
  } else if (ext_ == Exterior::Grounded) {
    for (int i = 0; i < nd; ++i) gmark[i] = (smark[i] && amark[i]) ? 1 : 0;
  }
  kind_.assign(nd, 0);
  for (int i = 0; i < nd; ++i) {
    if (tie_[i] >= 0) {
      AXF_REQUIRE(!gmark[i], strcat("the conductor ", bname(tie_[i]), " touches the grounded boundary"),
                  "/mesh/domain_radius", "Increase mesh.domain_radius so that the conductor stays inside.");
      kind_[i] = 2;
    } else if (gmark[i]) {
      kind_[i] = 3;
    } else if (amark[i]) {
      kind_[i] = 1;
    } else {
      kind_[i] = 3;  // a DOF of no assembled element: fixed to zero (it is decoupled)
    }
  }
  // Islands: a dielectric component that touches no conductor, no ground and not the mapped exterior has
  // no reference potential (an air pocket inside an insulating body).  Components are computed on the
  // serial mesh (globally consistent), anchoring is reduced over the ranks, and the stranded DOFs are
  // pinned to zero.
  int pinned = 0;
  {
    const mfem::Table& el2el = sm.ElementToElementTable();
    std::vector<int> comp(nser, -1);
    int nc = 0;
    std::vector<int> st;
    for (int s = 0; s < nser; ++s) {
      if (comp[s] >= 0 || !assembled(s)) continue;
      comp[s] = nc;
      st.push_back(s);
      while (!st.empty()) {
        const int e = st.back();
        st.pop_back();
        const int* row = el2el.GetRow(e);
        for (int k = 0; k < el2el.RowSize(e); ++k) {
          const int n = row[k];
          if (comp[n] < 0 && assembled(n)) {
            comp[n] = nc;
            st.push_back(n);
          }
        }
      }
      ++nc;
    }
    std::vector<int> anchored(std::max(nc, 1), 0);
    for (int e = 0; e < ne; ++e) {
      const int ci = comp[l2s[e]];
      if (ci < 0) continue;
      h1.GetElementDofs(e, dofs);
      for (int dd : dofs)
        if (kind_[dd] == 2 || kind_[dd] == 3) anchored[ci] = 1;
    }
    if (nc > 0) MPI_Allreduce(MPI_IN_PLACE, anchored.data(), nc, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    mfem::Array<int> ok(nd);
    ok = 0;
    for (int e = 0; e < ne; ++e) {
      const int ci = comp[l2s[e]];
      if (ci < 0 || !anchored[ci]) continue;
      h1.GetElementDofs(e, dofs);
      for (int dd : dofs) ok[dd] = 1;
    }
    h1.Synchronize(ok);
    for (int i = 0; i < nd; ++i)
      if (kind_[i] == 1 && !ok[i]) {
        kind_[i] = 3;
        if (h1.GetLocalTDofNumber(i) >= 0) ++pinned;
      }
  }

  // essential true DOFs and the per-body indicators
  std::set<int> ess;
  bodyT_.assign(nbody_, {});
  for (int i = 0; i < nd; ++i) {
    if (kind_[i] == 1) continue;
    const int t = h1.GetLocalTDofNumber(i);
    if (t < 0) continue;
    ess.insert(t);
    if (kind_[i] == 2) bodyT_[tie_[i]].push_back(t);
  }
  ess_.SetSize(0);
  for (int t : ess) ess_.Append(t);
  const int nt = h1.GetTrueVSize();
  U_.assign(nbody_, mfem::Vector());
  for (int b = 0; b < nbody_; ++b) {
    U_[b].SetSize(nt);
    U_[b] = 0.0;
    for (int t : bodyT_[b]) U_[b](t) = 1.0;
  }
  long long nfree = 0;
  for (int i = 0; i < nd; ++i)
    if (kind_[i] == 1 && h1.GetLocalTDofNumber(i) >= 0) ++nfree;
  MPI_Allreduce(MPI_IN_PLACE, &nfree, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  AXF_REQUIRE(nfree > 0, "the electrostatic problem has no free degrees of freedom");
  long long pin_g = pinned;
  MPI_Allreduce(MPI_IN_PLACE, &pin_g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // ------------------------------------------------------------------ assembly
  eps_ = std::make_unique<EpsTensor>(roles_, l2s, P_.sh_R1, P_.sh_R2, P_.sh_c, eps_out_,
                                     ext_ == Exterior::Open);
  form_ = std::make_unique<mfem::ParBilinearForm>(&h1);
  {
    auto* di = new mfem::DiffusionIntegrator(*eps_);
    di->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P_.order + 2));
    form_->AddDomainIntegrator(di);
  }
  form_->Assemble();
  form_->Finalize();
  K_.reset(form_->ParallelAssemble());
  Kel_.reset(form_->ParallelAssemble());
  ensure_diagonal(Kel_, ess_);
  Ke_.reset(Kel_->EliminateRowsCols(ess_));
  amg_ = std::make_unique<mfem::HypreBoomerAMG>(*Kel_);
  amg_->SetPrintLevel(0);
  pcg_ = std::make_unique<mfem::HyprePCG>(*Kel_);
  pcg_->SetTol(1e-13);
  pcg_->SetMaxIter(2000);
  pcg_->SetPrintLevel(0);
  pcg_->SetPreconditioner(*amg_);

  int nins = 0, ncond = 0;
  for (auto& r : roles_) {
    if (r.insulating) ++nins;
    if (r.body >= 0) ++ncond;
  }
  diag_ = {{"dofs", h1.GlobalTrueVSize()},
           {"free_dofs", nfree},
           {"order", P_.order},
           {"conductors", nbody_},
           {"pinned_dofs", pin_g},
           {"insulating_elements", nins},
           {"conductor_elements", ncond},
           {"boundary", ext_ == Exterior::Grounded ? "grounded" : (ext_ == Exterior::Open ? "open" : "none")},
           {"mpi_ranks", P_.nranks},
           {"setup_seconds", timer.seconds()}};
  ready_ = true;
}

std::unique_ptr<mfem::GridFunction> Electro3D::gather(const mfem::Vector& Phi) const {
  mfem::ParGridFunction u(h1_.get());
  u.SetFromTrueDofs(Phi);
  auto g = std::make_unique<mfem::GridFunction>(sh1_.get());
  *g = 0.0;
  mfem::Vector cnt(g->Size());
  cnt = 0.0;
  mfem::Array<int> ld, sd;
  for (size_t e = 0; e < P_.loc2ser.size(); ++e) {
    h1_->GetElementDofs(static_cast<int>(e), ld);
    sh1_->GetElementDofs(P_.loc2ser[e], sd);
    for (int k = 0; k < ld.Size(); ++k) {
      (*g)(sd[k]) += u(ld[k]);
      cnt(sd[k]) += 1.0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, g->GetData(), g->Size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cnt.GetData(), cnt.Size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < g->Size(); ++i)
    if (cnt(i) > 0) (*g)(i) /= cnt(i);
  return g;
}

double Electro3D::energy_quadrature(const mfem::Vector& Phi) const {
  mfem::ParGridFunction u(h1_.get());
  u.SetFromTrueDofs(Phi);
  mfem::ParMesh& pm = *P_.pmesh;
  mfem::DenseMatrix K(3);
  mfem::Vector g(3);
  double W = 0;
  for (int e = 0; e < pm.GetNE(); ++e) {
    mfem::ElementTransformation* T = pm.GetElementTransformation(e);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P_.order + 4);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      eps_->Eval(K, *T, ip);
      u.GetGradient(*T, g);
      W += 0.5 * K.InnerProduct(g, g) * ip.weight * T->Weight();
    }
  }
  return gsum(W);
}

std::vector<double> Electro3D::charges_gauss(const mfem::GridFunction& phi) const {
  std::vector<double> Q(nbody_, 0.0);
  mfem::Mesh& sm = *P_.serial;
  auto assembled = [&](int s) {
    const Role3& r = roles_[s];
    return r.body < 0 && !r.insulating && (!r.shell || ext_ == Exterior::Open);
  };
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P_.order + 4);
  mfem::Vector nor(3), g(3);
  for (int f = 0; f < sm.GetNumFaces(); ++f) {
    int e1 = -1, e2 = -1;
    sm.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0) continue;
    int body = -1;
    std::vector<std::pair<int, int>> sides;  // (element, which side 1/2)
    auto it = term_body_.find(f);
    if (it != term_body_.end()) {
      body = it->second;  // a zero-thickness electrode: both assembled sides contribute
      if (assembled(e1)) sides.push_back({e1, 1});
      if (assembled(e2)) sides.push_back({e2, 2});
    } else {
      const bool c1 = roles_[e1].body >= 0, c2 = roles_[e2].body >= 0;
      if (c1 == c2) continue;
      body = c1 ? roles_[e1].body : roles_[e2].body;
      const int ev = c1 ? e2 : e1;
      if (!assembled(ev)) continue;
      sides.push_back({ev, c1 ? 2 : 1});
    }
    if (body < 0 || sides.empty()) continue;
    mfem::FaceElementTransformations* FT = sm.GetFaceElementTransformations(f);
    for (auto [ev, which] : sides) {
      const double sgn = which == 1 ? -1.0 : 1.0;  // CalcOrtho points from element 1 to element 2
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        mfem::CalcOrtho(FT->Jacobian(), nor);
        const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
        mfem::ElementTransformation* Te = sm.GetElementTransformation(ev);
        Te->SetIntPoint(&eip);
        phi.GetGradient(*Te, g);
        double dn = 0;
        for (int k = 0; k < 3; ++k) dn += g(k) * sgn * nor(k);
        Q[body] -= roles_[ev].eps * dn * ip.weight;  // Q = -int eps grad(phi).n dA, n out of the conductor
      }
    }
  }
  return Q;
}

double Electro3D::exterior_flux(const mfem::GridFunction& phi) const {
  if (ext_ != Exterior::Open) return 0.0;
  mfem::Mesh& sm = *P_.serial;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P_.order + 4);
  mfem::Vector nor(3), g(3);
  double Q = 0;
  for (int f = 0; f < sm.GetNumFaces(); ++f) {
    int e1 = -1, e2 = -1;
    sm.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0) continue;
    const bool s1 = roles_[e1].shell, s2 = roles_[e2].shell;
    if (s1 == s2) continue;
    const int ev = s1 ? e2 : e1, which = s1 ? 2 : 1;
    if (roles_[ev].body >= 0 || roles_[ev].insulating) continue;
    const double sgn = which == 1 ? 1.0 : -1.0;  // n points out of the inner element, into the shell
    mfem::FaceElementTransformations* FT = sm.GetFaceElementTransformations(f);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      FT->SetAllIntPoints(&ip);
      mfem::CalcOrtho(FT->Jacobian(), nor);
      const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
      mfem::ElementTransformation* Te = sm.GetElementTransformation(ev);
      Te->SetIntPoint(&eip);
      phi.GetGradient(*Te, g);
      double dn = 0;
      for (int k = 0; k < 3; ++k) dn += g(k) * sgn * nor(k);
      Q -= roles_[ev].eps * dn * ip.weight;
    }
  }
  return Q;
}

Electro3D::Solution Electro3D::solve_prescribed(const std::vector<double>& V) {
  setup();
  AXF_REQUIRE(static_cast<int>(V.size()) == nbody_, "internal: potential vector size mismatch");
  const int nt = h1_->GetTrueVSize();
  mfem::Vector X(nt), B(nt);
  X = 0.0;
  B = 0.0;
  for (int b = 0; b < nbody_; ++b)
    for (int t : bodyT_[b]) X(t) = V[b];
  Kel_->EliminateBC(*Ke_, ess_, X, B);
  // The eliminated right-hand side carries the prescribed potentials (O(1 V)) on the essential rows and
  // the much smaller Ae X (O(eps0)) on the free ones, so hypre's criterion -- relative to |B| -- says
  // almost nothing about the free block.  Measure the residual on the free rows and refine: the
  // correction solve has zero essential rows, so there its criterion is the honest one.
  auto free_residual = [&](const mfem::Vector& x, mfem::Vector& r) {
    Kel_->Mult(x, r);
    r -= B;
    for (int i : ess_) r(i) = 0.0;
    return gnorm3(r);
  };
  mfem::Vector Bf(B);
  for (int i : ess_) Bf(i) = 0.0;
  const double bn = gnorm3(Bf);
  pcg_->Mult(B, X);
  int its = 0;
  pcg_->GetNumIterations(its);
  total_its_ += its;
  mfem::Vector R(nt), dX(nt), Bc(nt);
  double rn = free_residual(X, R);
  for (int pass = 0; pass < 4 && bn > 0 && rn > 1e-12 * bn; ++pass) {
    Bc = R;
    Bc.Neg();
    dX = 0.0;
    pcg_->Mult(Bc, dX);
    int it2 = 0;
    pcg_->GetNumIterations(it2);
    total_its_ += it2;
    its += it2;
    X += dX;
    rn = free_residual(X, R);
  }
  Solution s;
  s.iterations = its;
  s.residual = bn > 0 ? rn / bn : 0.0;
  max_res_ = std::max(max_res_, s.residual);
  s.Phi = X;
  s.V = V;
  mfem::Vector KPhi(nt);
  K_->Mult(X, KPhi);
  s.Q.assign(nbody_, 0.0);
  for (int b = 0; b < nbody_; ++b) s.Q[b] = gdot3(U_[b], KPhi);
  s.energy = 0.5 * gdot3(X, KPhi);
  s.energy_field = energy_quadrature(X);
  s.serial = gather(X);
  s.Q_gauss = charges_gauss(*s.serial);
  s.flux_exterior = exterior_flux(*s.serial);
  diag_["pcg_iterations_total"] = total_its_;
  diag_["max_relative_residual"] = max_res_;
  return s;
}

Eigen::MatrixXd Electro3D::capacitance(std::vector<Solution>* cols) {
  setup();
  Timer timer;
  AXF_REQUIRE(nbody_ > 0, "the electrostatic problem has no conductor to charge");
  Eigen::MatrixXd C(nbody_, nbody_);
  std::vector<Solution> local;
  std::vector<Solution>& out = cols ? *cols : local;
  for (int j = 0; j < nbody_; ++j) {
    std::vector<double> V(nbody_, 0.0);
    V[j] = 1.0;
    Solution s = solve_prescribed(V);
    for (int i = 0; i < nbody_; ++i) C(i, j) = s.Q[i];
    out.push_back(std::move(s));
  }
  diag_["capacitance_seconds"] = timer.seconds();
  return C;
}

// ----------------------------------------------------------------------------------------------------
// helpers shared with the axisymmetric analysis (kept local: the shared files stay untouched)

bool is_conductor(const spec::Material& m) {
  return m.kind == spec::MaterialKind::Superconductor || m.kind == spec::MaterialKind::Conductor;
}

// Connected components (through faces) of the elements satisfying `in`.
std::vector<int> connected_bodies(const mfem::Mesh& m, const std::function<bool(int)>& in, int& nb) {
  const int ne = m.GetNE();
  std::vector<int> b(ne, -1);
  const mfem::Table& el2el = const_cast<mfem::Mesh&>(m).ElementToElementTable();
  nb = 0;
  std::vector<int> st;
  for (int s = 0; s < ne; ++s) {
    if (b[s] >= 0 || !in(s)) continue;
    b[s] = nb;
    st.push_back(s);
    while (!st.empty()) {
      const int e = st.back();
      st.pop_back();
      const int* row = el2el.GetRow(e);
      for (int k = 0; k < el2el.RowSize(e); ++k) {
        const int n = row[k];
        if (b[n] < 0 && in(n)) {
          b[n] = nb;
          st.push_back(n);
        }
      }
    }
    ++nb;
  }
  return b;
}

// Maxwell matrix of the conductors `keep` with all the others floating at zero net charge:
// C_hat = C_pp - C_pf C_ff^{-1} C_fp.
Eigen::MatrixXd reduce_floating(const Eigen::MatrixXd& C, const std::vector<int>& keep,
                                Eigen::MatrixXd* Vfloat = nullptr) {
  const int m = static_cast<int>(C.rows()), p = static_cast<int>(keep.size());
  std::vector<int> fl;
  for (int i = 0; i < m; ++i)
    if (std::find(keep.begin(), keep.end(), i) == keep.end()) fl.push_back(i);
  Eigen::MatrixXd Cpp(p, p);
  for (int a = 0; a < p; ++a)
    for (int b = 0; b < p; ++b) Cpp(a, b) = C(keep[a], keep[b]);
  if (fl.empty()) {
    if (Vfloat) *Vfloat = Eigen::MatrixXd::Zero(0, p);
    return Cpp;
  }
  const int nf = static_cast<int>(fl.size());
  Eigen::MatrixXd Cff(nf, nf), Cfp(nf, p);
  for (int a = 0; a < nf; ++a) {
    for (int b = 0; b < nf; ++b) Cff(a, b) = C(fl[a], fl[b]);
    for (int b = 0; b < p; ++b) Cfp(a, b) = C(fl[a], keep[b]);
  }
  Eigen::LDLT<Eigen::MatrixXd> ld(Cff);
  AXF_REQUIRE(ld.info() == Eigen::Success && ld.isPositive(),
              "floating capacitance block is singular or not positive definite", "/electrostatics",
              "Check conductor connectivity and refine the mesh.");
  Eigen::MatrixXd S = ld.solve(Cfp);
  AXF_REQUIRE(S.allFinite() && (Cff*S-Cfp).norm() <= 1e-6 * std::max(Cfp.norm(), 1e-300),
              "floating capacitance reduction did not converge", "/electrostatics",
              "Merge electrically connected conductors or refine the mesh.");
  if (Vfloat) *Vfloat = -S;  // V_F = -C_FF^{-1} C_FP V_P
  return Cpp - Cfp.transpose() * S;
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

// ====================================================================================================
// Electrostatics (3D): the Maxwell capacitance matrix of the conductor bodies, the pairwise (mutual)
// capacitances and the readout-port gap capacitance with the pickup self-resonance.
void Model3D::electrostatics_analysis(const SolveOptions& o) {
  (void)o;
  Impl& P = *p_;
  const bool want_matrix = d_.electrostatics.enabled;
  // 3D: the readout-gap analysis (a few extra H1 solves on the whole mesh) runs with an `electrostatics` section;
  // 2D runs it for every port pickup (cheap there)
  const bool want_port = want_matrix && d_.pickup && d_.pickup->kind == spec::Pickup::Kind::Port && P.port_attr > 0;
  if (!want_matrix && !want_port) return;
  AXF_REQUIRE(P.pmesh && P.serial, "internal: electrostatics needs a solved (meshed) 3D model");
  Timer timer;
  json& R = results_;
  if (!R.contains("warnings") || !R["warnings"].is_array()) R["warnings"] = json::array();
  json& warnings = R["warnings"];
  const double Lp = R.contains("coupling") ? R["coupling"].value("L_p_H", 0.0) : 0.0;
  const double eps0 = 1.0 / (units::mu0 * units::c * units::c);
  auto check_residual = [&](const json& dg, const char* what) {
    const double res = dg.value("max_relative_residual", 0.0);
    if (!(res < 1e-8))
      warnings.push_back(strcat("electrostatics (", what, "): a potential solve stopped at a relative residual "
                                "of ", res, " on the free degrees of freedom; the capacitances are not reliable"));
  };

  // --------------------------------------------------------------- materials and conductor bodies
  mfem::Mesh& sm = *P.serial;
  const int nser = sm.GetNE();
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  std::vector<std::string> ser_region(nser);
  std::vector<const spec::Material*> ser_mat(nser);
  std::vector<double> ser_eps(nser);
  std::vector<char> ser_shell(nser, 0);
  for (int s = 0; s < nser; ++s) {
    const int a = sm.GetAttribute(s);
    ser_region[s] = P.region_of_attr.count(a) ? P.region_of_attr.at(a) : "air";
    const spec::Region* g = d_.find_region(ser_region[s]);
    ser_mat[s] = g ? &d_.materials.at(g->material) : &vacuum;
    ser_eps[s] = eps0 * ser_mat[s]->eps_r;
    ser_shell[s] = (P.sh_attr > 0 && a == P.sh_attr) ? 1 : 0;
  }
  int nb = 0;
  std::vector<int> body_of = connected_bodies(sm, [&](int s) { return is_conductor(*ser_mat[s]); }, nb);
  std::vector<std::set<std::string>> bnames(nb);
  for (int s = 0; s < nser; ++s)
    if (body_of[s] >= 0) bnames[body_of[s]].insert(ser_region[s]);
  auto body_name = [&](int b) { return bnames[b].empty() ? std::string("body") : *bnames[b].begin(); };

  const bool have_shell = P.sh_attr > 0 && P.sh_R2 > P.sh_R1;
  const bool grounded = d_.electrostatics.boundary == "grounded";
  Exterior ext = grounded ? Exterior::Grounded : Exterior::Open;
  if (!grounded && !have_shell) {
    warnings.push_back("electrostatics: this mesh has no infinity shell, so the open boundary falls back to "
                       "phi = 0 on the outer sphere (a grounded boundary); capacitances to infinity are "
                       "underestimated");
  }
  // The infinity shell is always vacuum: the mesher refuses a region that reaches into it, so the
  // permittivity just outside the domain sphere is eps0 by construction (unlike the axisymmetric solver,
  // which has to inspect the boundary elements).
  const double eps_out = eps0;

  auto base_roles = [&]() {
    std::vector<Role3> roles(nser);
    for (int s = 0; s < nser; ++s) {
      roles[s].eps = ser_eps[s];
      roles[s].shell = ser_shell[s] != 0;
      if (roles[s].shell) roles[s].eps = eps_out;
    }
    return roles;
  };

  // --------------------------------------------------------------- capacitance matrix
  if (want_matrix) {
    AXF_REQUIRE(nb > 0,
                "electrostatics needs at least one conductor: a region of a 'superconductor' or 'conductor' "
                "material",
                "/electrostatics", "Add a conductor region, or remove the 'electrostatics' section.");
    std::vector<Role3> roles = base_roles();
    for (int s = 0; s < nser; ++s) roles[s].body = body_of[s];
    Electro3D p(P, roles, nb, ext, eps_out);
    std::vector<std::string> names;
    for (int b = 0; b < nb; ++b) names.push_back(body_name(b));
    p.set_names(names);
    log(1, "solving electrostatics (3D capacitance matrix)");
    std::vector<Electro3D::Solution> cols;
    Eigen::MatrixXd C = p.capacitance(&cols);
    const double cmax = C.cwiseAbs().maxCoeff();

    // checks ------------------------------------------------------------------
    double sym = 0;
    for (int i = 0; i < nb; ++i)
      for (int j = 0; j < i; ++j) sym = std::max(sym, std::fabs(C(i, j) - C(j, i)));
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
    double e_rel = 0, q_rel = 0, mono_rel = 0;
    json colj = json::array();
    for (int j = 0; j < nb; ++j) {
      const auto& s = cols[j];
      const double half = 0.5 * Cs(j, j);
      const double de = std::fabs(s.energy_field - half) / std::max(std::fabs(half), 1e-300);
      double dq = 0, qmax = 0, qtot = 0;
      for (int i = 0; i < nb; ++i) {
        qmax = std::max(qmax, std::fabs(s.Q[i]));
        dq = std::max(dq, std::fabs(s.Q_gauss[i] - s.Q[i]));
        qtot += s.Q[i];
      }
      dq /= std::max(qmax, 1e-300);
      // the exterior-flux check needs the sphere at mesh.domain_radius, i.e. the infinity shell
      const double dm = (ext != Exterior::Open || !have_shell)
                            ? 0.0
                            : std::fabs(s.flux_exterior - qtot) / std::max(qmax, 1e-300);
      e_rel = std::max(e_rel, de);
      q_rel = std::max(q_rel, dq);
      mono_rel = std::max(mono_rel, dm);
      colj.push_back({{"body", body_name(j)}, {"energy_J", s.energy_field}, {"half_C_jj_J", half},
                      {"energy_relative_difference", de}, {"charge_gauss_relative_difference", dq},
                      {"net_charge_C", qtot}, {"exterior_monopole_C", s.flux_exterior},
                      {"residual", s.residual}, {"pcg_iterations", s.iterations}});
    }
    // energy / charge at a potential vector that is not a unit column, by superposition AND by an
    // independent solve (3D has no separate floating-conductor solver, so this is the independent path)
    Eigen::VectorXd Vt(nb);
    std::vector<double> Vtv(nb);
    for (int i = 0; i < nb; ++i) {
      Vtv[i] = 1.3 + 0.37 * i;  // not a unit column, also for a single conductor
      Vt(i) = Vtv[i];
    }
    mfem::Vector Phi_t(cols[0].Phi.Size());
    Phi_t = 0.0;
    for (int j = 0; j < nb; ++j) Phi_t.Add(Vt(j), cols[j].Phi);
    const double W_field = p.energy_quadrature(Phi_t);
    const double W_mat = 0.5 * Vt.dot(Cs * Vt);
    const double W_rel = std::fabs(W_field - W_mat) / std::max(std::fabs(W_mat), 1e-300);
    Electro3D::Solution mix = p.solve_prescribed(Vtv);
    Eigen::VectorXd Qm = Cs * Vt;
    double dqm = 0, qms = 0;
    std::vector<double> qgot(nb), qwant(nb);
    for (int i = 0; i < nb; ++i) {
      qgot[i] = mix.Q[i];
      qwant[i] = Qm(i);
      qms = std::max(qms, std::fabs(Qm(i)));
      dqm = std::max(dqm, std::fabs(mix.Q[i] - Qm(i)));
    }
    dqm /= std::max(qms, 1e-300);

    // a floating-conductor state from the exact matrix reduction, with the charges of the floating
    // conductors recomputed by the independent Gauss path (they must vanish)
    json floatj = json::object();
    if (nb >= 2) {
      std::vector<int> keep{0};
      Eigen::MatrixXd Vf;
      reduce_floating(Cs, keep, &Vf);
      std::vector<double> Vfull(nb, 0.0);
      Vfull[0] = 1.0;
      for (int a = 1; a < nb; ++a) Vfull[a] = Vf(a - 1, 0);
      mfem::GridFunction phi_f(&p.serial_space());
      phi_f = 0.0;
      for (int j = 0; j < nb; ++j) phi_f.Add(Vfull[j], *cols[j].serial);
      std::vector<double> Qg = p.charges_gauss(phi_f);
      double qf = 0;
      std::vector<double> Vfl;
      for (int a = 1; a < nb; ++a) Vfl.push_back(Vfull[a]);
      for (int a = 1; a < nb; ++a) qf = std::max(qf, std::fabs(Qg[a]));
      floatj = {{"driven", body_name(0)}, {"V_driven", 1.0}, {"V_floating_from_matrix", Vfl},
                {"charge_on_driven_C", Qg[0]},
                {"max_relative_floating_charge", qf / std::max(std::fabs(Qg[0]), 1e-300)},
                {"note", "body 0 at 1 V and the others at the potentials V_F = -C_FF^-1 C_F0 of the exact "
                         "matrix reduction; the charges are recomputed by the surface-flux (Gauss) path, so "
                         "the floating charges must vanish"}};
    }

    // pairwise (mutual) capacitances ------------------------------------------
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
                {"domain_radius_m", P.mesh_info.value("domain_radius", 0.0)},
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
                            {"energy_superposition", {{"V", Vtv}, {"W_field_J", W_field},
                                                      {"W_from_C_J", W_mat},
                                                      {"relative_difference", W_rel}}},
                            {"mixed_potential_solve",
                             {{"V", Vtv}, {"Q_C", qgot}, {"Q_from_matrix_C", qwant},
                              {"max_relative_difference", dqm}, {"W_field_J", mix.energy_field},
                              {"W_from_C_J", W_mat}, {"residual", mix.residual},
                              {"note", "an independent solve at a non-unit potential vector (not a "
                                       "superposition of the columns)"}}},
                            {"floating_solve", floatj}}},
                {"solver", p.diagnostics()},
                {"exterior",
                 (grounded || !have_shell)
                          ? "phi = 0 on the sphere at mesh.domain_radius"
                          : "infinity shell [R1, R2) mapped onto [R1, infinity) with the scalar tensor "
                            "eps_eff = eps0 [(f^2/(f' rho^2)) r r^T + f' (I - r r^T)], phi = 0 on its outer "
                            "sphere; continuum-exact monopole map with finite-element error requiring h/p convergence"},
                {"convention",
                 "Maxwell matrix: C(i, j) = Q_i for V_j = 1 V and every other conductor at 0 V; ground is "
                 "infinity (open) or the outer sphere (grounded). C_to_ground = row sum. The pairwise "
                 "mutual capacitance is -C(i, j) (the lumped capacitor of the equivalent network, i.e. "
                 "with the other conductors grounded); C_mutual_others_floating_F instead holds the other "
                 "conductors at zero net charge (an exact reduction of the matrix)."}};
    check_residual(p.diagnostics(), "capacitance matrix");
    for (int i = 0; i < nb; ++i)
      out["bodies"].push_back({{"name", body_name(i)},
                               {"regions", std::vector<std::string>(bnames[i].begin(), bnames[i].end())}});
    if (!d_.electrostatics.notes.empty()) out["notes"] = d_.electrostatics.notes;
    if (!signs_ok)
      warnings.push_back("electrostatics: a positive off-diagonal capacitance appeared; the Maxwell matrix "
                         "should have C(i, j) <= 0 for i != j (check the mesh and the conductor bodies)");
    if (min_slack < -1e-6)
      warnings.push_back(strcat("electrostatics: the Maxwell matrix is not diagonally dominant (relative "
                                "slack ", min_slack, "); the discretization is too coarse"));
    R["electrostatics"] = out;
  }

  // --------------------------------------------------------------- readout-port gap capacitance
  // The port region bridges a gap between two patches of ONE conductor body.  Its two terminal patches
  // become the electrodes at V = +-1/2 (the mutual capacitance of the pair is the gap capacitance).
  if (want_port) {
    const int pattr = P.port_attr;
    std::vector<int> tfaces;
    mfem::Array<int> fcs, ori, fed, fo;
    for (int e = 0; e < nser; ++e) {
      if (sm.GetAttribute(e) != pattr) continue;
      sm.GetElementFaces(e, fcs, ori);
      for (int f : fcs) {
        int e1 = -1, e2 = -1;
        sm.GetFaceElements(f, &e1, &e2);
        const int other = e1 == e ? e2 : e1;
        if (other >= 0 && body_of[other] >= 0) tfaces.push_back(f);
      }
    }
    std::map<int, std::vector<int>> edge_faces;
    for (int f : tfaces) {
      sm.GetFaceEdges(f, fed, fo);
      for (int ed : fed) edge_faces[ed].push_back(f);
    }
    std::map<int, int> patch;
    int npatch = 0;
    for (int f0 : tfaces) {
      if (patch.count(f0)) continue;
      std::vector<int> st{f0};
      patch[f0] = npatch;
      while (!st.empty()) {
        const int f = st.back();
        st.pop_back();
        sm.GetFaceEdges(f, fed, fo);
        for (int ed : fed)
          for (int g : edge_faces[ed])
            if (!patch.count(g)) {
              patch[g] = npatch;
              st.push_back(g);
            }
      }
      ++npatch;
    }
    const int pieces = P.port_diagnostics.value("port_pieces", 1);
    if (npatch != 2 && pieces > 1) {
      warnings.push_back(strcat("port gap capacitance not computed: the readout port is a distributed port of ", pieces,
                                " pieces, which the gap-capacitance analysis does not handle yet (its terminals would "
                                "be all V = 1 and all V = 0 patches together)"));
    } else if (npatch != 2) {
      warnings.push_back(strcat("port gap capacitance not computed: the readout port touches conductors on ",
                                npatch, " separate patches (expected 2 terminal patches)"));
    } else {
      std::vector<std::vector<int>> pf(2);
      std::vector<int> pbody(2, -1);
      double area[2] = {0, 0};
      const mfem::IntegrationRule& irf = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order + 2);
      for (auto& [f, pidx] : patch) {
        pf[pidx].push_back(f);
        int e1 = -1, e2 = -1;
        sm.GetFaceElements(f, &e1, &e2);
        const int other = sm.GetAttribute(e1) == pattr ? e2 : e1;
        pbody[pidx] = body_of[other];
        mfem::ElementTransformation* T = sm.GetFaceTransformation(f);
        for (int q = 0; q < irf.GetNPoints(); ++q) {
          T->SetIntPoint(&irf.IntPoint(q));
          area[pidx] += T->Weight() * irf.IntPoint(q).weight;
        }
      }
      const int own = pbody[0];
      json pc = {{"port", d_.pickup->port},
                 {"terminal_area_m2", std::vector<double>{area[0], area[1]}},
                 {"terminals_on_one_body", pbody[0] == pbody[1]}};
      // (a) the port region alone: no fringing, a lower bound on the differential capacitance
      double C_reg = 0;
      {
        std::vector<Role3> roles = base_roles();
        for (int s = 0; s < nser; ++s) roles[s].insulating = sm.GetAttribute(s) != pattr;
        Electro3D p(P, roles, 2, Exterior::None, eps_out);
        p.set_names({"readout terminal A", "readout terminal B"});
        p.add_terminal(0, pf[0]);
        p.add_terminal(1, pf[1]);
        log(1, "solving electrostatics (3D readout gap, port region)");
        std::vector<Electro3D::Solution> cols;
        Eigen::MatrixXd C = p.capacitance(&cols);
        C_reg = -0.5 * (C(0, 1) + C(1, 0));
        pc["C_gap_port_region_F"] = C_reg;
        pc["port_region_solver"] = p.diagnostics();
        check_residual(p.diagnostics(), "readout gap, port region");
        pc["checks"]["region_charge_gauss_relative"] =
            std::fabs(cols[0].Q_gauss[0] - cols[0].Q[0]) / std::max(std::fabs(cols[0].Q[0]), 1e-300);
        pc["checks"]["region_symmetry_relative"] =
            std::fabs(C(0, 1) - C(1, 0)) / std::max(C.cwiseAbs().maxCoeff(), 1e-300);
        pc["checks"]["region_energy_relative"] =
            std::fabs(cols[0].energy_field - 0.5 * C(0, 0)) / std::max(0.5 * std::fabs(C(0, 0)), 1e-300);
      }
      // (b) the whole domain: the body carrying the terminals is cut at the port, i.e. its surface away
      // from the two terminal patches is insulating (n.D = 0) and the two patches are the electrodes;
      // every other conductor floats at zero net charge.
      double C_gap = C_reg;
      if (pbody[0] == pbody[1]) {
        std::vector<Role3> roles = base_roles();
        int next = 2;
        std::vector<int> remap(std::max(nb, 1), -1);
        for (int b = 0; b < nb; ++b)
          if (b != own) remap[b] = next++;
        for (int s = 0; s < nser; ++s) {
          const int b = body_of[s];
          if (b == own) roles[s].insulating = true;
          else if (b >= 0) roles[s].body = remap[b];
        }
        Electro3D p(P, roles, next, ext, eps_out);
        std::vector<std::string> names{"readout terminal A", "readout terminal B"};
        names.resize(next);
        for (int b = 0; b < nb; ++b)
          if (remap[b] >= 0) names[remap[b]] = body_name(b);
        p.set_names(names);
        p.add_terminal(0, pf[0]);
        p.add_terminal(1, pf[1]);
        log(1, "solving electrostatics (3D readout gap, whole domain)");
        std::vector<Electro3D::Solution> cols;
        Eigen::MatrixXd C = p.capacitance(&cols);
        Eigen::MatrixXd Cs = 0.5 * (C + C.transpose());
        Eigen::MatrixXd Ch = reduce_floating(Cs, {0, 1});
        C_gap = -Ch(0, 1);
        pc["C_gap_F"] = C_gap;  // the mutual capacitance, = C_mutual_F
        pc["C_mutual_F"] = C_gap;
        pc["C_reduced_F"] = matrix_json(Ch);
        const double Cdiff = 0.25 * (Ch(0,0) + Ch(1,1) - 2*Ch(0,1));
        pc["C_differential_F"] = Cdiff;
        pc["checks"]["dirichlet_bound"] = {{"holds", Cdiff >= C_reg*(1-1e-9)},
                                              {"C_differential_F", Cdiff}, {"C_port_region_F", C_reg}};
        if (Lp > 0 && Cdiff > 0)
          pc["self_resonance_differential_Hz"] = 1.0/(2*pi*std::sqrt(Lp*Cdiff));
        pc["C_terminal_to_ground_F"] = std::vector<double>{Ch(0, 0) + Ch(0, 1), Ch(1, 1) + Ch(0, 1)};
        pc["gap_solver"] = p.diagnostics();
        check_residual(p.diagnostics(), "readout gap, whole domain");
        pc["checks"]["gap_symmetry_relative"] =
            (C - C.transpose()).cwiseAbs().maxCoeff() / std::max(C.cwiseAbs().maxCoeff(), 1e-300);
        // The electrode is a set of degrees of freedom, so the reaction charge also collects the flux that
        // reaches the terminal DOFs from just outside the terminal patches (within one element of the gap
        // mouths); the surface integral over the patches alone therefore differs.
        pc["checks"]["gap_charge_reaction_C"] = cols[0].Q[0];
        pc["checks"]["gap_charge_terminal_faces_C"] = cols[0].Q_gauss[0];
        pc["checks"]["gap_charge_fraction_beyond_terminal_faces"] =
            1.0 - cols[0].Q_gauss[0] / std::max(std::fabs(cols[0].Q[0]), 1e-300);
        pc["checks"]["gap_energy_relative"] =
            std::fabs(cols[0].energy_field - 0.5 * Cs(0, 0)) / std::max(0.5 * std::fabs(Cs(0, 0)), 1e-300);
      } else {
        pc["C_gap_F"] = C_reg;
        pc["note_bodies"] = "the two terminal patches belong to different conductor bodies; only the "
                            "port-region capacitance is computed (use the 'electrostatics' section for the "
                            "full matrix of the two bodies)";
      }
      if (Lp > 0 && C_gap > 0) {
        pc["L_p_H"] = Lp;
        pc["self_resonance_Hz"] = 1.0 / (2 * pi * std::sqrt(Lp * C_gap));
        pc["self_resonance_from_port_region_Hz"] = C_reg > 0 ? 1.0 / (2 * pi * std::sqrt(Lp * C_reg)) : 0.0;
      }
      pc["note"] =
          "C_gap_F = C_mutual_F is -C_AB after floating-body reduction. C_differential_F is the "
          "capacitance for V_A=+1/2, V_B=-1/2 and equals (C_AA+C_BB-2*C_AB)/4. "
          "C_reduced_F is the complete reduced Maxwell matrix. The Dirichlet bound is "
          "C_differential_F >= C_gap_port_region_F; no such bound is claimed for C_mutual_F. "
          "All reported LC frequencies are lumped estimates, with no guaranteed ordering relative "
          "to a distributed full-wave pole.";
      R["pickup"]["port_capacitance"] = pc;
      const double fsr = pc.value("self_resonance_Hz", 0.0);
      if (fsr > 0 && fsr < d_.scan.f_max)
        warnings.push_back(strcat("the readout-gap capacitance (", C_gap, " F) puts the pickup "
                                  "self-resonance at ", fsr, " Hz, below the top of the scan (",
                                  d_.scan.f_max, " Hz)"));
    }
  }
  R["timing"]["electrostatics_s"] = timer.seconds();
}

}  // namespace axf
