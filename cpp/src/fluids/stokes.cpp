// Taylor–Hood Stokes on ParMesh: see cpp/include/axionfem/fluids/stokes.hpp.
#include "axionfem/fluids/stokes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include <Eigen/Dense>
#include <array>

#include "axionfem/core.hpp"
#include "axionfem/fluids/axi_integrators.hpp"
#include "axionfem/omp_shim.hpp"
#include "stokes_internal.hpp"

namespace axf::fluids {

const char* to_string(ViscousForm f) { return f == ViscousForm::Laplacian ? "laplacian" : "stress"; }
const char* to_string(Nullspace n) {
  switch (n) {
    case Nullspace::Auto: return "auto";
    case Nullspace::None: return "none";
    case Nullspace::PinPressure: return "pin";
    case Nullspace::MeanMultiplier: return "multiplier";
    case Nullspace::Ortho: return "ortho";
  }
  return "?";
}
const char* to_string(BCKind k) {
  switch (k) {
    case BCKind::Dirichlet: return "dirichlet";
    case BCKind::DoNothing: return "do_nothing";
    case BCKind::Traction: return "traction";
    case BCKind::Axis: return "axis";
    case BCKind::Slip: return "slip";
    case BCKind::Symmetry: return "symmetry";
    case BCKind::Periodic: return "periodic";
  }
  return "?";
}

namespace {
constexpr double kNullspaceTol = 1e-10;  // ||B^T 1|| / (||B||_F ||1||) below this: constants are in the kernel

double hypre_frobenius(mfem::HypreParMatrix& M, MPI_Comm comm) {
  mfem::SparseMatrix m;
  M.MergeDiagAndOffd(m);
  double s = 0, g = 0;
  for (int k = 0; k < m.NumNonZeroElems(); ++k) s += m.GetData()[k] * m.GetData()[k];
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
  return std::sqrt(g);
}

// sort the column indices within each row (MergeDiagAndOffd puts the diagonal block first), summing duplicates
void sort_rows(DistCsr& D) { D.sum_duplicates(); }

// -(q, d_x u_x + d_y u_y) coefficient c on a three-component velocity space (Planar25D: u_z does not enter continuity)
class PlanarDivergence3Integrator : public mfem::BilinearFormIntegrator {
 public:
  explicit PlanarDivergence3Integrator(mfem::Coefficient& c) : c_(c) {}
  void AssembleElementMatrix2(const mfem::FiniteElement& tr, const mfem::FiniteElement& te,
                              mfem::ElementTransformation& T, mfem::DenseMatrix& elmat) override {
    const int nu = tr.GetDof(), np = te.GetDof();
    elmat.SetSize(np, 3 * nu);
    elmat = 0.0;
    mfem::Vector Q(np);
    mfem::DenseMatrix dN(nu, 2);
    const mfem::IntegrationRule* ir = IntRule;
    if (!ir) ir = &mfem::IntRules.Get(tr.GetGeomType(), tr.GetOrder() + te.GetOrder() + T.OrderW());
    for (int q = 0; q < ir->GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);
      te.CalcShape(ip, Q);
      tr.CalcPhysDShape(T, dN);
      const double w = ip.weight * T.Weight() * c_.Eval(T, ip);
      for (int i = 0; i < np; ++i)
        for (int c = 0; c < 2; ++c)
          for (int j = 0; j < nu; ++j) elmat(i, c * nu + j) += w * Q(i) * dN(j, c);
    }
  }

 private:
  mfem::Coefficient& c_;
};

// --- explicit quadrature: the inner integrator gets the rule of the given order for each element's geometry
class RuleBFI : public mfem::BilinearFormIntegrator {
 public:
  RuleBFI(mfem::BilinearFormIntegrator* in, int order) : in_(in), order_(order) {}
  void AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             mfem::DenseMatrix& m) override {
    in_->SetIntRule(&mfem::IntRules.Get(el.GetGeomType(), order_));
    in_->AssembleElementMatrix(el, T, m);
  }
  void AssembleElementMatrix2(const mfem::FiniteElement& tr, const mfem::FiniteElement& te,
                              mfem::ElementTransformation& T, mfem::DenseMatrix& m) override {
    in_->SetIntRule(&mfem::IntRules.Get(tr.GetGeomType(), order_));
    in_->AssembleElementMatrix2(tr, te, T, m);
  }

 private:
  std::unique_ptr<mfem::BilinearFormIntegrator> in_;
  int order_;
};

class RuleLFI : public mfem::LinearFormIntegrator {
 public:
  RuleLFI(mfem::LinearFormIntegrator* in, int order) : in_(in), order_(order) {}
  void AssembleRHSElementVect(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                              mfem::Vector& v) override {
    in_->SetIntRule(&mfem::IntRules.Get(el.GetGeomType(), order_));
    in_->AssembleRHSElementVect(el, T, v);
  }
  using mfem::LinearFormIntegrator::AssembleRHSElementVect;

 private:
  std::unique_ptr<mfem::LinearFormIntegrator> in_;
  int order_;
};

bool is_open(BCKind k) { return k == BCKind::DoNothing || k == BCKind::Traction; }
}  // namespace

StokesSolver::StokesSolver(mfem::ParMesh& mesh, const StokesOptions& opt, const StokesBoundary& bc)
    : impl_(new Impl), mesh_(mesh), opt_(opt), bc_(bc) {
  const int dim = mesh.Dimension();
  AXF_REQUIRE(opt.nu != nullptr, "StokesOptions.nu (viscosity coefficient) is required");
  if (auto* cn = dynamic_cast<mfem::ConstantCoefficient*>(opt.nu))
    AXF_REQUIRE(std::isfinite(cn->constant) && cn->constant > 0,
                strcat("the viscosity must be positive and finite (got ", cn->constant, ")"), "StokesOptions.nu");
  const int pord = opt_.pressure_order > 0 ? opt_.pressure_order : opt_.velocity_order - 1;
  AXF_REQUIRE(opt_.velocity_order >= 1 && pord >= 1, "velocity and pressure orders must be >= 1",
              "StokesOptions.velocity_order / pressure_order");
  AXF_REQUIRE(pord < opt_.velocity_order || opt_.allow_unstable_pair,
              strcat("pressure order ", pord, " >= velocity order ", opt_.velocity_order,
                     ": an unstabilized equal-order pair is not inf-sup stable"),
              "StokesOptions.pressure_order",
              "Use Taylor–Hood (pressure order = velocity order - 1); allow_unstable_pair is for negative controls.");
  AXF_REQUIRE(std::isfinite(opt.grad_div) && opt.grad_div >= 0.0, "the grad-div parameter gamma must be finite and >= 0",
              "StokesOptions.grad_div");
  AXF_REQUIRE(std::isfinite(opt.velocity_nullspace_tol) && opt.velocity_nullspace_tol >= 0 &&
                  opt.velocity_nullspace_tol < 1,
              strcat("velocity_nullspace_tol must be finite and in [0, 1) (got ", opt.velocity_nullspace_tol, ")"),
              "StokesOptions.velocity_nullspace_tol");
  AXF_REQUIRE(std::isfinite(opt.direct_tol) && opt.direct_tol > 0 && std::isfinite(opt.krylov_rtol) &&
                  opt.krylov_rtol > 0, "direct_tol and rtol must be finite and positive", "StokesOptions");
  AXF_REQUIRE(std::isfinite(opt.slip_corner_angle_deg) && opt.slip_corner_angle_deg > 0 &&
                  opt.slip_corner_angle_deg < 180, "slip_corner_angle_deg must be in (0, 180)",
              "StokesOptions.slip_corner_angle_deg");
  if (opt.geometry == Geometry::Axisymmetric) {
    AXF_REQUIRE(dim == 2, "the axisymmetric formulation needs a 2D (r, z) mesh");
    impl_->vdim = opt.swirl ? 3 : 2;
  } else if (opt.geometry == Geometry::Planar25D) {
    AXF_REQUIRE(dim == 2, "Planar25D needs a 2D (x, y) mesh", "StokesOptions.geometry");
    AXF_REQUIRE(opt.form == ViscousForm::Laplacian,
                "Planar25D (2.5D MHD) is offered with the Laplacian viscous form only (constant viscosity: the stress "
                "form coincides with it for Dirichlet data, and is not tested here)", "StokesOptions.form",
                "Use form = 'laplacian'.");
    AXF_REQUIRE(opt.grad_div == 0.0, "Planar25D: grad-div is not offered", "StokesOptions.grad_div");
    impl_->vdim = 3;
  } else {
    impl_->vdim = dim;
  }
  ufec_.reset(new mfem::H1_FECollection(opt_.velocity_order, dim));
  pfec_.reset(new mfem::H1_FECollection(pord, dim));
  ufes_.reset(new mfem::ParFiniteElementSpace(&mesh, ufec_.get(), impl_->vdim, mfem::Ordering::byVDIM));
  pfes_.reset(new mfem::ParFiniteElementSpace(&mesh, pfec_.get()));
  u_.reset(new mfem::ParGridFunction(ufes_.get()));
  p_.reset(new mfem::ParGridFunction(pfes_.get()));
  *u_ = 0.0;
  *p_ = 0.0;

  validate_boundary();

  // the conditions imply a pressure nullspace iff no attribute is open (DoNothing/Traction); assemble() checks this
  // against the algebra. Slip, symmetry and periodic boundaries close the domain.
  bool enclosed = true;
  for (const auto& kv : impl_->bc) {
    const bool open = is_open(kv.second.kind);
    enclosed = enclosed && !open;
  }
  diag_.bc_enclosed = enclosed;
  diag_.grad_div = opt.grad_div;
  Nullspace ns = opt.nullspace;
  const bool direct = opt.solver == LinearSolverKind::Direct;
  impl_->direct = direct;
  if (ns == Nullspace::Auto)
    ns = !enclosed ? Nullspace::None
                   : (!direct ? Nullspace::Ortho
                              : (opt.lu == LUBackend::EigenLU ? Nullspace::PinPressure : Nullspace::MeanMultiplier));
  // pinning or removing the mean with an open boundary silently gives a wrong answer
  AXF_REQUIRE(enclosed || ns == Nullspace::None,
              std::string("pressure nullspace treatment '") + to_string(ns) +
                  "' requested, but the flow has an open (do-nothing or traction) boundary, so the pressure is "
                  "already determined",
              "StokesOptions.nullspace",
              "Use Nullspace::Auto or None; pinning p with an open boundary silently changes the solution.");
  AXF_REQUIRE(enclosed || !opt.pressure_point.active,
              "a pressure point was given, but the flow has an open (do-nothing or traction) boundary, which already "
              "determines the pressure", "StokesOptions.pressure_point",
              "Remove the pressure point, or make every boundary a velocity (Dirichlet/axis/slip/periodic) boundary.");
  AXF_REQUIRE(!enclosed || ns != Nullspace::None,
              "the boundary conditions leave a constant-pressure nullspace (no open boundary), but Nullspace::None "
              "was requested",
              "StokesOptions.nullspace", "Use Auto, PinPressure or MeanMultiplier (direct) or Ortho (MINRES).");
  AXF_REQUIRE(!(direct && ns == Nullspace::Ortho), "Nullspace::Ortho is the Krylov treatment",
              "StokesOptions.nullspace", "Use PinPressure or MeanMultiplier with the direct solver.");
  AXF_REQUIRE(!(!direct && (ns == Nullspace::PinPressure || ns == Nullspace::MeanMultiplier)),
              "the MINRES path removes the nullspace with OrthoSolver", "StokesOptions.nullspace",
              "Use Auto or Ortho with LinearSolverKind::MinresBlockDiag.");
  if (opt.pressure_point.active)
    AXF_REQUIRE(static_cast<int>(opt.pressure_point.x0.size()) == mesh.SpaceDimension(),
                "the pressure point x0 must have one coordinate per space dimension", "StokesOptions.pressure_point");
  for (int a : impl_->periodic_order)
    AXF_REQUIRE(direct || impl_->bc[a].drive != PeriodicDrive::FlowRate,
                "a prescribed periodic flow rate needs the direct solver (one bordered multiplier)",
                strcat("StokesBoundary (periodic attribute ", a, ")"),
                "Use LinearSolverKind::Direct, or prescribe the pressure drop.");
  opt_.nullspace = ns;
  diag_.nullspace = to_string(ns);
  diag_.viscous_form = to_string(opt.form);
  diag_.solver = direct ? std::string("direct:") + to_string(opt.lu) : "minres-blockdiag";
}

StokesSolver::~StokesSolver() = default;

void StokesSolver::validate_boundary() {
  Impl& I = *impl_;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  std::vector<int> mesh_attrs(mesh_.bdr_attributes.begin(), mesh_.bdr_attributes.end());
  auto in_mesh = [&](int a) { return std::find(mesh_attrs.begin(), mesh_attrs.end(), a) != mesh_attrs.end(); };
  std::map<int, int> partner_of;  // partner attribute -> its source attribute
  for (const auto& c0 : bc_.conditions) {
    BoundaryCondition c = c0;
    AXF_REQUIRE(in_mesh(c.attribute),
                strcat("boundary condition for attribute ", c.attribute, ", which no boundary element of the mesh "
                       "carries"), "StokesBoundary", "Check the attribute numbers of the mesh.");
    AXF_REQUIRE(!I.bc.count(c.attribute) && !partner_of.count(c.attribute),
                strcat("boundary attribute ", c.attribute, " has more than one condition"
                       " (a periodic partner needs no condition of its own)"),
                "StokesBoundary", "Give each attribute exactly one condition.");
    AXF_REQUIRE(axi || c.kind != BCKind::Axis, "axis boundaries are only meaningful with Geometry::Axisymmetric",
                strcat("StokesBoundary (attribute ", c.attribute, ")"));
    if (c.value)
      AXF_REQUIRE(c.value->GetVDim() == I.vdim,
                  strcat("the value of the ", to_string(c.kind), " condition on attribute ", c.attribute, " has ",
                         c.value->GetVDim(), " components; the formulation has ", I.vdim), "StokesBoundary");
    AXF_REQUIRE(opt_.geometry != Geometry::Planar25D ||
                    (c.kind == BCKind::Dirichlet || c.kind == BCKind::Traction || c.kind == BCKind::DoNothing),
                strcat("Planar25D: boundary condition '", to_string(c.kind), "' on attribute ", c.attribute,
                       " is not offered"), "StokesBoundary", "Use dirichlet, traction or do_nothing.");
    if (c.kind == BCKind::Slip || c.kind == BCKind::Symmetry) {
      AXF_REQUIRE(!axi, strcat("slip/symmetry on attribute ", c.attribute, ": not implemented in axisymmetric geometry"),
                  "StokesBoundary", "Axisymmetric slip/symmetry walls are not implemented; "
                  "the axis r = 0 itself needs kind 'axis'.");
      AXF_REQUIRE(!c.normal || c.normal->GetVDim() == I.vdim,
                  strcat("the normal of the slip condition on attribute ", c.attribute, " has ",
                         c.normal->GetVDim(), " components; ", I.vdim, " expected"), "StokesBoundary");
    }
    if (c.kind == BCKind::Periodic) {
      AXF_REQUIRE(!axi, "periodic conditions are planar only", strcat("StokesBoundary (attribute ", c.attribute, ")"));
      AXF_REQUIRE(c.partner != c.attribute && in_mesh(c.partner),
                  strcat("periodic condition on attribute ", c.attribute, ": partner attribute ", c.partner,
                         " is not another boundary attribute of the mesh"), "StokesBoundary");
      AXF_REQUIRE(!I.bc.count(c.partner) && !partner_of.count(c.partner),
                  strcat("periodic partner attribute ", c.partner, " also has a condition of its own"),
                  "StokesBoundary", "List each periodic pair once, on one of its two attributes.");
      AXF_REQUIRE(static_cast<int>(c.translation.size()) == mesh_.SpaceDimension(),
                  strcat("periodic condition on attribute ", c.attribute, ": the translation needs ",
                         mesh_.SpaceDimension(), " components"), "StokesBoundary");
      double L = 0;
      for (double t : c.translation) L += t * t;
      AXF_REQUIRE(L > 0, "periodic translation of zero length", "StokesBoundary");
      partner_of[c.partner] = c.attribute;
      I.periodic_order.push_back(c.attribute);
    }
    I.bc[c.attribute] = c;
  }
  for (const auto& kv : partner_of)
    if (!I.bc.count(kv.first)) {
      BoundaryCondition pc = I.bc[kv.second];
      pc.attribute = kv.first;
      I.bc[kv.first] = pc;  // the partner carries the same Periodic condition (no DOFs, no terms)
    }
  std::string missing;
  for (int a : mesh_attrs)
    if (!I.bc.count(a)) missing += (missing.empty() ? "" : ", ") + std::to_string(a);
  AXF_REQUIRE(missing.empty(), "no boundary condition for boundary attribute(s) " + missing, "StokesBoundary",
              "Every boundary attribute needs an explicit condition (dirichlet, do_nothing, traction, axis, slip, "
              "symmetry, periodic): an attribute left out would otherwise become a silent traction-free boundary.");

  // periodic attributes must lie on INTERIOR faces: the mesh was made periodic (make_periodic_mesh)
  if (!I.periodic_order.empty()) {
    mesh_.ExchangeFaceNbrData();
    int bad = 0, gbad = 0;
    for (int be = 0; be < mesh_.GetNBE(); ++be) {
      const int a = mesh_.GetBdrAttribute(be);
      if (I.bc[a].kind != BCKind::Periodic) continue;
      int e1 = -1, e2 = -1;
      mesh_.GetFaceElements(mesh_.GetBdrElementFaceIndex(be), &e1, &e2);
      if (e2 == -1 && !mesh_.FaceIsTrueInterior(mesh_.GetBdrElementFaceIndex(be))) ++bad;
    }
    MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, mesh_.GetComm());
    AXF_REQUIRE(gbad == 0,
                strcat(gbad, " boundary elements of periodic attributes lie on the true boundary: the mesh was not "
                       "made periodic"), "StokesBoundary",
                "Build the mesh with make_periodic_mesh(mesh, {{attribute, partner, translation}}) first (the Python "
                "API does this from the periodic conditions).");
  }

  if (!axi) return;
  // axisymmetric geometry: r >= 0, and the attributes that lie on r = 0 are exactly the declared axes
  mfem::Vector pmin, pmax;
  mesh_.GetBoundingBox(pmin, pmax);
  const double tol = 1e-10 * std::max(1.0, pmax.DistanceTo(pmin));
  double rmin = std::numeric_limits<double>::max(), grmin;
  for (int v = 0; v < mesh_.GetNV(); ++v) rmin = std::min(rmin, mesh_.GetVertex(v)[0]);
  MPI_Allreduce(&rmin, &grmin, 1, MPI_DOUBLE, MPI_MIN, mesh_.GetComm());
  AXF_REQUIRE(grmin >= -tol, strcat("the axisymmetric mesh has vertices at r = ", grmin, " < 0"), "mesh",
              "The meridional mesh must lie in r >= 0 (x is r).");
  const int nattr = *std::max_element(mesh_attrs.begin(), mesh_attrs.end());
  std::vector<int> cnt(nattr + 1, 0), off(nattr + 1, 0), gcnt(nattr + 1), goff(nattr + 1);
  mfem::Array<int> vs;
  for (int be = 0; be < mesh_.GetNBE(); ++be) {
    const int a = mesh_.GetBdrAttribute(be);
    ++cnt[a];
    mesh_.GetBdrElementVertices(be, vs);
    for (int v : vs)
      if (std::fabs(mesh_.GetVertex(v)[0]) > tol) ++off[a];
  }
  MPI_Allreduce(cnt.data(), gcnt.data(), nattr + 1, MPI_INT, MPI_SUM, mesh_.GetComm());
  MPI_Allreduce(off.data(), goff.data(), nattr + 1, MPI_INT, MPI_SUM, mesh_.GetComm());
  for (int a : mesh_attrs) {
    const bool on_axis = gcnt[a] > 0 && goff[a] == 0;
    const bool declared = I.bc[a].kind == BCKind::Axis;
    AXF_REQUIRE(on_axis == declared,
                on_axis ? strcat("boundary attribute ", a, " lies on the axis r = 0 but is declared '",
                                 to_string(I.bc[a].kind), "'")
                        : strcat("boundary attribute ", a, " is declared 'axis' but does not lie on r = 0"),
                "StokesBoundary",
                "Declare exactly the boundary on r = 0 as axis (u_r = u_theta = 0 strongly; u_z and p natural).");
  }
}

// Slip/symmetry: nodal frames.
//  * Normal: the CONSISTENT (mass-conserving) normal of Engelman, Sani & Gresho (1982), n_k ∝ sum over the adjacent
//    slip/symmetry boundary elements of ∮ phi_k n ds, so that the tangential DOF keeps B^T 1 = 0 (the constant
//    pressure stays in the kernel) on any mesh, curved or not. A `normal` coefficient on the condition overrides it
//    (the user's choice; on non-symmetric curved meshes it breaks the discrete mass balance of enclosed flows).
//  * Corners are GEOMETRIC, independent of the attribute labels: at each node the unit outward normals of all
//    adjacent slip/symmetry boundary elements (any attribute) are summed; if |sum| / count < cos(angle/2), with
//    angle = StokesOptions::slip_corner_angle_deg (default 30 degrees: above the O(h^2) kinks of isoparametric walls
//    and the 2 pi / N_theta kinks of polygonal ones, below every geometric corner), the node is a corner and both
//    components are eliminated (u = 0); otherwise one frame with u.n = 0.
//  * Dirichlet nodes are never slip nodes (Dirichlet wins). Sums are reduced onto the owning true DOF (parallel safe).
void StokesSolver::setup_slip() {
  if (mesh_.Dimension() == 3 && opt_.geometry == Geometry::Planar) return setup_3d();  // 3D: stokes3d.cpp
  Impl& I = *impl_;
  I.rotated = false;
  I.slip_ess.SetSize(0);
  std::vector<int> slip_attrs;
  for (const auto& kv : I.bc)
    if (kv.second.kind == BCKind::Slip || kv.second.kind == BCKind::Symmetry) slip_attrs.push_back(kv.first);
  diag_.n_slip_nodes = diag_.n_slip_corner_nodes = 0;
  diag_.slip_max_averaged_kink_deg = 0;
  if (slip_attrs.empty()) return;
  AXF_REQUIRE(I.vdim == 2, "slip/symmetry: planar 2D only");
  const int nl = ufes_->GetVSize(), nt = ufes_->GetTrueVSize();
  const mfem::HypreParMatrix* P = ufes_->Dof_TrueDof_Matrix();
  double corner_angle = opt_.slip_corner_angle_deg;
  const double cos_half = corner_angle < 0 ? 2.0 : std::cos(0.5 * corner_angle * M_PI / 180.0);
  // accumulators on local vdofs: consistent normal (2 slots), unit pointwise normals (2 slots), count (slot 0),
  // user normal (2 slots) and how many contributions had one (slot 0)
  mfem::Vector cons(nl), unit(nl), cnt(nl), user(nl), ucnt(nl);
  cons = 0.0;
  unit = 0.0;
  cnt = 0.0;
  user = 0.0;
  ucnt = 0.0;
  mfem::Array<int> dofs;
  mfem::Vector shape;
  for (int attr0 : slip_attrs) {
    const int attr = attr0;
    const BoundaryCondition& c = I.bc[attr0];
    for (int be = 0; be < mesh_.GetNBE(); ++be) {
      if (mesh_.GetBdrAttribute(be) != attr) continue;
      ufes_->GetBdrElementDofs(be, dofs);
      const mfem::FiniteElement* fe = ufes_->GetBE(be);
      mfem::FaceElementTransformations* FT = mesh_.GetBdrFaceTransformations(be);
      shape.SetSize(fe->GetDof());
      // consistent normal: ∫_e phi_k n ds (CalcOrtho: outward from the adjacent element, length = |J|)
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(FT->GetGeometryType(), 2 * fe->GetOrder() + 4);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        fe->CalcShape(ip, shape);
        mfem::Vector n(2);
        mfem::CalcOrtho(FT->Jacobian(), n);
        for (int k = 0; k < dofs.Size(); ++k) {
          cons(ufes_->DofToVDof(dofs[k], 0)) += ip.weight * shape(k) * n(0);
          cons(ufes_->DofToVDof(dofs[k], 1)) += ip.weight * shape(k) * n(1);
        }
      }
      // pointwise unit normals at the nodes (corner detection) and the user's normal
      const mfem::IntegrationRule& nodes = fe->GetNodes();
      for (int k = 0; k < dofs.Size(); ++k) {
        const mfem::IntegrationPoint& ip = nodes.IntPoint(k);
        FT->SetAllIntPoints(&ip);
        mfem::Vector n(2);
        mfem::CalcOrtho(FT->Jacobian(), n);
        const double len = n.Norml2();
        AXF_REQUIRE(len > 0, strcat("slip attribute ", attr, ": degenerate boundary element ", be));
        n /= len;
        const int d0 = ufes_->DofToVDof(dofs[k], 0), d1 = ufes_->DofToVDof(dofs[k], 1);
        unit(d0) += n(0);
        unit(d1) += n(1);
        cnt(d0) += 1.0;
        if (c.normal) {
          mfem::Vector g(2);
          c.normal->Eval(g, *FT->Elem1, FT->GetElement1IntPoint());
          const double gl = g.Norml2();
          AXF_REQUIRE(gl > 0 && std::isfinite(gl), strcat("slip attribute ", attr0, ": the normal coefficient is zero or "
                                                          "not finite at a node"), "StokesBoundary");
          if (g * n < 0) g.Neg();
          user(d0) += g(0) / gl;
          user(d1) += g(1) / gl;
          ucnt(d0) += 1.0;
        }
      }
    }
  }
  mfem::Vector cons_t(nt), unit_t(nt), cnt_t(nt), user_t(nt), ucnt_t(nt);
  P->MultTranspose(cons, cons_t);
  P->MultTranspose(unit, unit_t);
  P->MultTranspose(cnt, cnt_t);
  P->MultTranspose(user, user_t);
  P->MultTranspose(ucnt, ucnt_t);
  std::vector<char> is_dir(nt, 0);
  for (int t : I.dir_tdofs) is_dir[t] = 1;
  I.Rdiag = mfem::SparseMatrix(nt, nt);
  std::vector<char> done(nt, 0), rot_row(nt, 0);
  long nslip = 0, ncorner = 0;
  double max_kink = 0.0;
  for (int dof = 0; dof < ufes_->GetNDofs(); ++dof) {
    const int t0 = ufes_->GetLocalTDofNumber(ufes_->DofToVDof(dof, 0));
    if (t0 < 0 || done[t0]) continue;
    const int t1 = ufes_->GetLocalTDofNumber(ufes_->DofToVDof(dof, 1));
    done[t0] = 1;
    const double m = cnt_t(t0);
    if (m <= 0 || is_dir[t0] || is_dir[t1]) continue;  // not a slip node, or Dirichlet wins
    const double spread = std::hypot(unit_t(t0), unit_t(t1)) / m;
    if (spread >= cos_half && m > 1)  // averaged kink (two normals): turning angle 2 acos(spread)
      max_kink = std::max(max_kink, 2.0 * std::acos(std::min(1.0, spread)) * 180.0 / M_PI);
    if (spread < cos_half) {  // a geometric corner: u = 0
      I.slip_ess.Append(t0);
      I.slip_ess.Append(t1);
      ++ncorner;
      continue;
    }
    mfem::Vector n(2);
    if (ucnt_t(t0) > 0) {
      n(0) = user_t(t0);
      n(1) = user_t(t1);
    } else {
      n(0) = cons_t(t0);
      n(1) = cons_t(t1);
    }
    const double len = n.Norml2();
    AXF_REQUIRE(len > 0, "slip/symmetry: zero nodal normal (the boundary folds back on itself?)", "StokesBoundary",
                "Give the slip condition an explicit normal coefficient.");
    n /= len;
    // u = n w_n + t w_t, t = (-n_y, n_x); w_n lives in slot t0 (eliminated, 0), w_t in slot t1
    I.Rdiag.Set(t0, t0, n(0));
    I.Rdiag.Set(t1, t0, n(1));
    I.Rdiag.Set(t0, t1, -n(1));
    I.Rdiag.Set(t1, t1, n(0));
    done[t1] = 1;
    rot_row[t0] = rot_row[t1] = 1;
    I.slip_ess.Append(t0);
    ++nslip;
  }
  for (int t = 0; t < nt; ++t)
    if (!rot_row[t]) I.Rdiag.Set(t, t, 1.0);
  I.Rdiag.Finalize();
  long loc[2] = {nslip, ncorner}, glob[2];
  MPI_Allreduce(loc, glob, 2, MPI_LONG, MPI_SUM, mesh_.GetComm());
  diag_.n_slip_nodes = glob[0];
  diag_.n_slip_corner_nodes = glob[1];
  MPI_Allreduce(&max_kink, &diag_.slip_max_averaged_kink_deg, 1, MPI_DOUBLE, MPI_MAX, mesh_.GetComm());
  I.rotated = glob[0] > 0;
  if (I.rotated)
    I.R.reset(new mfem::HypreParMatrix(mesh_.GetComm(), ufes_->GlobalTrueVSize(), ufes_->GetTrueDofOffsets(),
                                       &I.Rdiag));
}

void StokesSolver::assemble() {
  Impl& I = *impl_;
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  const double t0 = MPI_Wtime();
  const int nattr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  MPI_Comm comm = mesh_.GetComm();
  const int qv = opt_.viscous_quad_order();
  const int ql = opt_.load_quad_order();
  const int qb = opt_.boundary_quad_order();
  const int qd = opt_.viscous_quad_order();

  if (axi) I.r_coef.reset(new RadiusWeight());  // r (the weight of the axisymmetric measure; axi_integrators.hpp)
  I.markers.clear();
  I.vcoefs.clear();
  I.scoefs.clear();
  I.markers.reserve(2 * I.bc.size() + 2);  // the forms keep pointers to the markers: no reallocation
  auto marker_of = [&](int attr) -> mfem::Array<int>& {
    I.markers.emplace_back(nattr);
    I.markers.back() = 0;
    I.markers.back()[attr - 1] = 1;
    return I.markers.back();
  };

  // essential velocity DOFs: Dirichlet (all components) + axis (u_r, u_theta); boundary values per attribute
  mfem::Array<int> dir_marker(nattr), axis_marker(nattr);
  dir_marker = 0;
  axis_marker = 0;
  *u_ = 0.0;
  for (const auto& kv : I.bc) {
    if (kv.second.kind == BCKind::Dirichlet) dir_marker[kv.first - 1] = 1;
    if (kv.second.kind == BCKind::Axis) axis_marker[kv.first - 1] = 1;
  }
  ufes_->GetEssentialTrueDofs(dir_marker, I.dir_tdofs);
  if (axi) {
    // u_r = u_theta = 0 strongly at EVERY velocity node on the axis (vertices and edge nodes)
    std::vector<int> comps = {0, 2};
    for (int c : comps) {
      if (c >= I.vdim) continue;
      mfem::Array<int> vmark, l;
      ufes_->GetEssentialVDofs(axis_marker, vmark, c);
      for (int vd = 0; vd < vmark.Size(); ++vd)
        if (vmark[vd]) {
          const int t = ufes_->GetLocalTDofNumber(vd);
          if (t >= 0) l.Append(t);
        }
      I.dir_tdofs.Append(l);
    }
  }
  // a velocity node on r = 0 that lies on no axis edge (a domain touching the axis at one vertex, e.g.
  // a cone apex) gets u_r = u_theta = 0 too: the regularity condition holds at every point of the axis, not only on
  // the attributes declared 'axis'. Dirichlet nodes keep their data (Dirichlet wins).
  diag_.n_axis_point_nodes = 0;
  if (axi) {
    mfem::Vector pmin, pmax;
    mesh_.GetBoundingBox(pmin, pmax);
    const double tol = 1e-12 * std::max(1.0, pmax.DistanceTo(pmin));
    mfem::Array<int> on_axis_attr, on_dir, dofs;
    ufes_->GetEssentialVDofs(axis_marker, on_axis_attr, 0);
    ufes_->GetEssentialVDofs(dir_marker, on_dir, 0);
    std::vector<char> seen(ufes_->GetNDofs(), 0);
    long cnt = 0;
    for (int e = 0; e < mesh_.GetNE(); ++e) {
      ufes_->GetElementDofs(e, dofs);
      const mfem::IntegrationRule& nodes = ufes_->GetFE(e)->GetNodes();
      mfem::ElementTransformation* T = mesh_.GetElementTransformation(e);
      for (int k = 0; k < dofs.Size(); ++k) {
        const int d = dofs[k] >= 0 ? dofs[k] : -1 - dofs[k];
        if (seen[d]) continue;
        double x[3];
        mfem::Vector X(x, mesh_.SpaceDimension());
        T->Transform(nodes.IntPoint(k), X);
        if (std::fabs(x[0]) > tol) continue;
        seen[d] = 1;
        const int vd0 = ufes_->DofToVDof(d, 0);
        if (on_axis_attr[vd0] || on_dir[vd0]) continue;  // an axis-edge node (above) or Dirichlet data
        ++cnt;
        for (int c : {0, 2}) {
          if (c >= I.vdim) continue;
          const int t = ufes_->GetLocalTDofNumber(ufes_->DofToVDof(d, c));
          if (t >= 0) I.dir_tdofs.Append(t);
        }
      }
    }
    MPI_Allreduce(&cnt, &diag_.n_axis_point_nodes, 1, MPI_LONG, MPI_SUM, mesh_.GetComm());
  }
  I.dir_tdofs.Sort();
  I.dir_tdofs.Unique();
  // Dirichlet data: P2 Lagrange NODAL interpolation of the given coefficient
  for (const auto& kv : I.bc)
    if (kv.second.kind == BCKind::Dirichlet && kv.second.value) {
      mfem::Array<int> m(nattr);
      m = 0;
      m[kv.first - 1] = 1;
      u_->ProjectBdrCoefficient(*kv.second.value, m);
    }
  setup_slip();

  // forms
  I.a.reset(new mfem::ParBilinearForm(ufes_.get()));
  I.b.reset(new mfem::ParMixedBilinearForm(ufes_.get(), pfes_.get()));
  I.mp.reset(new mfem::ParBilinearForm(pfes_.get()));
  I.f.reset(new mfem::ParLinearForm(ufes_.get()));
  I.pmean.reset(new mfem::ParLinearForm(pfes_.get()));
  I.inv_nu.reset(new mfem::RatioCoefficient(1.0, *opt_.nu));
  if (axi) {
    // axisymmetric quadrature: fixed rules, no point on r = 0 (checked in the integrators);
    // viscous incl. 1/r hoop terms and grad-div on the SAME rule (2p+2, degree 6 for P2), the (div_r v, q)_r
    // coupling on that rule too (degree 3 needed on affine elements), the load on 2p+8
    double gamma = opt_.grad_div;
    I.a->AddDomainIntegrator(
        new AxiViscousIntegrator(*opt_.nu, I.vdim, opt_.form == ViscousForm::SymmetricStress, gamma, qv));
    I.b->AddDomainIntegrator(new AxiDivergenceIntegrator(I.vdim, qd));
    I.mp_coef.reset(new RadiusTimes(*I.inv_nu));
    I.rmean.reset(new RadiusWeight());
    I.pmean->AddDomainIntegrator(new RuleLFI(new mfem::DomainLFIntegrator(*I.rmean), qd));
    if (opt_.force) {
      mfem::VectorCoefficient* fr = I.V(new mfem::ScalarVectorProductCoefficient(*I.r_coef, *opt_.force));
      I.f->AddDomainIntegrator(new RuleLFI(new mfem::VectorDomainLFIntegrator(*fr), ql));
    }
  } else {
    mfem::Coefficient* nu = opt_.nu;
    const bool sym = opt_.form == ViscousForm::SymmetricStress;
    if (opt_.geometry == Geometry::Planar25D) {
      I.a->AddDomainIntegrator(new RuleBFI(new mfem::VectorDiffusionIntegrator(*nu, I.vdim), qv));
    } else if (!sym) {
      I.a->AddDomainIntegrator(new RuleBFI(new mfem::VectorDiffusionIntegrator(*nu), qv));
    } else {
      I.a->AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(I.zero, *nu), qv));  // 2 nu eps:eps
    }
    // grad-div gamma (div u, div v), gamma in viscosity units
    double gamma = opt_.grad_div;
    if (gamma != 0.0) {
      mfem::Coefficient* g = I.S(new mfem::ConstantCoefficient(gamma));
      I.a->AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(*g, I.zero), qv));
    }
    mfem::Coefficient* bcoef = &I.minus_one;
    if (opt_.geometry == Geometry::Planar25D)
      I.b->AddDomainIntegrator(new RuleBFI(new PlanarDivergence3Integrator(*bcoef), qd));
    else
      I.b->AddDomainIntegrator(new RuleBFI(new mfem::VectorDivergenceIntegrator(*bcoef), qd));
    I.mp_coef.reset(new mfem::RatioCoefficient(1.0, *opt_.nu));
    I.pmean->AddDomainIntegrator(new RuleLFI(new mfem::DomainLFIntegrator(I.one), qd));
    if (opt_.force) I.f->AddDomainIntegrator(new RuleLFI(new mfem::VectorDomainLFIntegrator(*opt_.force), ql));
  }
  // traction boundaries: <g, v> (r-weighted in axisymmetry); g is sigma_F n of the chosen form
  for (const auto& kv : I.bc)
    if (kv.second.kind == BCKind::Traction && kv.second.value) {
      mfem::VectorCoefficient* g = kv.second.value;
      if (axi) g = I.V(new mfem::ScalarVectorProductCoefficient(*I.r_coef, *g));
      I.f->AddBoundaryIntegrator(new RuleLFI(new mfem::VectorBoundaryLFIntegrator(*g), qb), marker_of(kv.first));
    }
  I.mp->AddDomainIntegrator(axi ? static_cast<mfem::BilinearFormIntegrator*>(
                                      new RuleBFI(new mfem::MassIntegrator(*I.mp_coef), qd))
                                : new RuleBFI(new mfem::MassIntegrator(*I.mp_coef), qd));
  I.a->Assemble();
  I.a->Finalize();
  I.b->Assemble();
  I.b->Finalize();
  I.mp->Assemble();
  I.mp->Finalize();
  I.f->Assemble();
  I.pmean->Assemble();

  I.A.reset(I.a->ParallelAssemble());
  I.B.reset(I.b->ParallelAssemble());
  I.Mp.reset(I.mp->ParallelAssemble());
  std::unique_ptr<mfem::HypreParVector> F(I.f->ParallelAssemble());
  std::unique_ptr<mfem::HypreParVector> W(I.pmean->ParallelAssemble());
  mfem::Vector Fl(*F);
  I.mean_weights = *W;

  // periodic drives: c = int phi . d_hat; a prescribed drop adds G c to the load (p = p_per - G x.d_hat)
  I.drives.clear();
  diag_.periodic_pressure_gradient.clear();
  diag_.periodic_flow_rate.clear();
  for (int a : I.periodic_order) {
    const BoundaryCondition& c = I.bc[a];
    Impl::Drive d;
    d.attribute = a;
    d.kind = c.drive;
    d.value = c.drive_value;
    double L = 0;
    for (double t : c.translation) L += t * t;
    d.length = std::sqrt(L);
    mfem::Vector dh(I.vdim);
    for (int k = 0; k < I.vdim; ++k) dh(k) = c.translation[k] / d.length;
    mfem::VectorConstantCoefficient dc(dh);
    mfem::ParLinearForm cf(ufes_.get());
    cf.AddDomainIntegrator(new RuleLFI(new mfem::VectorDomainLFIntegrator(dc), ql));
    cf.Assemble();
    std::unique_ptr<mfem::HypreParVector> cv(cf.ParallelAssemble());
    d.c = *cv;
    double G = 0;
    if (d.kind == PeriodicDrive::PressureDrop) {
      G = d.value / d.length;
      Fl.Add(G, d.c);
    }
    diag_.periodic_pressure_gradient.push_back(G);
    diag_.periodic_flow_rate.push_back(d.kind == PeriodicDrive::FlowRate ? d.value : 0.0);
    I.drives.push_back(std::move(d));
  }
  I.Fvec = Fl;
  I.A0.reset(new mfem::HypreParMatrix(*I.A));  // before rotation/elimination: boundary_power() needs it
  I.B0.reset(new mfem::HypreParMatrix(*I.B));  // before elimination: the reaction needs the full operator
  rotate_operators();
  mfem::Vector Fr(Fl.Size());
  if (I.rotated) I.R->MultTranspose(Fl, Fr);
  else Fr = Fl;

  I.ess_tdofs = I.dir_tdofs;
  I.ess_tdofs.Append(I.slip_ess);
  I.ess_tdofs.Sort();
  I.ess_tdofs.Unique();

  I.offsets.SetSize(3);
  I.offsets[0] = 0;
  I.offsets[1] = ufes_->GetTrueVSize();
  I.offsets[2] = I.offsets[1] + pfes_->GetTrueVSize();
  I.rhs.reset(new mfem::BlockVector(I.offsets));
  I.x.reset(new mfem::BlockVector(I.offsets));
  *I.x = 0.0;
  mfem::Vector X(I.x->GetBlock(0).GetData(), I.offsets[1]);
  u_->GetTrueDofs(X);  // boundary values on the Dirichlet DOFs (where the frame is Cartesian); 0 elsewhere
  {
    std::vector<char> keep(X.Size(), 0);
    for (int i : I.dir_tdofs) keep[i] = 1;
    for (int i = 0; i < X.Size(); ++i)
      if (!keep[i]) X(i) = 0.0;
  }
  I.ess_mask.assign(I.offsets[2], 0);
  for (int i : I.ess_tdofs) I.ess_mask[i] = 1;
  detect_velocity_nullspace();
  // flow-rate constraints (bordered): -c.u = -Q |d|  (multiplier G), essential part moved to the right-hand side
  I.borders.clear();
  for (const auto& d : I.drives) {
    if (d.kind != PeriodicDrive::FlowRate) continue;
    Impl::Border b;
    b.u.SetSize(I.offsets[1]);
    if (I.rotated) I.R->MultTranspose(d.c, b.u);
    else b.u = d.c;
    b.u.Neg();
    double lift = 0;
    for (int i : I.ess_tdofs) {
      lift += b.u(i) * X(i);
      b.u(i) = 0.0;
    }
    double glift = 0;
    MPI_Allreduce(&lift, &glift, 1, MPI_DOUBLE, MPI_SUM, comm);
    b.p.SetSize(I.offsets[2] - I.offsets[1]);
    b.p = 0.0;
    b.rhs = -d.value * d.length - glift;
    I.borders.push_back(std::move(b));
  }
  // eliminate the essential velocity DOFs symmetrically from A and from the columns of B
  I.Ae.reset(I.A->EliminateRowsCols(I.ess_tdofs));
  I.Be.reset(I.B->EliminateCols(I.ess_tdofs));
  mfem::Vector& ru = I.rhs->GetBlock(0);
  mfem::Vector& rp = I.rhs->GetBlock(1);
  ru = Fr;
  I.A->EliminateBC(*I.Ae, I.ess_tdofs, X, ru);
  rp = 0.0;
  I.Be->Mult(-1.0, X, 1.0, rp);
  I.Bt.reset(I.B->Transpose());

  // algebraic nullspace check: constants in the kernel iff B^T 1 = 0 after elimination
  {
    mfem::Vector one_p(pfes_->GetTrueVSize()), y(ufes_->GetTrueVSize());
    one_p = 1.0;
    I.B->MultTranspose(one_p, y);
    const double ny = std::sqrt(mfem::InnerProduct(comm, y, y));
    const double nB = hypre_frobenius(*I.B, comm);
    const double n1 = std::sqrt(static_cast<double>(pfes_->GlobalTrueVSize()));
    diag_.kernel_defect = (nB > 0) ? ny / (nB * n1) : 0.0;
    diag_.nullspace_detected = diag_.kernel_defect <= kNullspaceTol;
    bool user_normals = false;
    for (const auto& kv : I.bc) user_normals = user_normals || ((kv.second.kind == BCKind::Slip ||
                                                                 kv.second.kind == BCKind::Symmetry) && kv.second.normal);
    AXF_REQUIRE(!(diag_.bc_enclosed && !diag_.nullspace_detected && user_normals),
                strcat("enclosed flow with a prescribed slip normal: the given nodal normals are not the consistent "
                       "(mass-conserving) normals of this mesh, so the discrete continuity equation loses the "
                       "constant-pressure nullspace (||B^T 1|| / (||B|| ||1||) = ", diag_.kernel_defect, ")"),
                "StokesBoundary (slip normal)",
                "Drop the 'normal' coefficient: the default consistent normal conserves mass on any mesh (they agree "
                "on symmetric meshes); a pointwise exact normal on a non-symmetric curved mesh does not.");
    AXF_REQUIRE(diag_.nullspace_detected == diag_.bc_enclosed,
                diag_.bc_enclosed
                    ? strcat("the boundary conditions declare no open boundary, but the assembled operator has no "
                             "constant-pressure nullspace (||B^T 1|| / (||B|| ||1||) = ", diag_.kernel_defect, ")")
                    : strcat("the boundary conditions declare an open boundary, but the assembled operator has a "
                             "constant-pressure nullspace (||B^T 1|| / (||B|| ||1||) = ", diag_.kernel_defect,
                             "): the open attributes carry no free velocity DOFs"),
                "StokesBoundary", "Check the attribute numbers and kinds against the mesh (a mislabelled wall or "
                                  "axis changes which pressures are determined).");
  }
  // consistency: with the nullspace present, project the continuity rhs onto 1-perp and
  // report the flux defect of the interpolated Dirichlet data
  diag_.flux_defect = diag_.flux_defect_abs = 0;
  if (diag_.nullspace_detected) {
    double loc[2] = {rp.Sum(), rp * rp}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double np = static_cast<double>(pfes_->GlobalTrueVSize());
    diag_.flux_defect_abs = std::fabs(glob[0]);
    diag_.flux_defect = glob[1] > 0 ? std::fabs(glob[0]) / std::sqrt(glob[1]) : 0.0;
    // Euclidean projection onto 1-perp: valid because the constant pressure has unit coefficients in the Lagrange
    // P(k) basis (any order); a modal pressure basis would need the coefficient vector of the constant here
    rp -= glob[0] / np;
  }
  if (opt_.pressure_point.active) {
    double d = 0;
    nearest_pressure_dof(opt_.pressure_point.x0, &d);
    mfem::Vector pmin, pmax;
    mesh_.GetBoundingBox(pmin, pmax);
    AXF_REQUIRE(d <= 1e-9 * pmax.DistanceTo(pmin),
                strcat("the pressure point is not a pressure node (nearest node at distance ", d, ")"),
                "StokesOptions.pressure_point", "Put x0 on a mesh vertex (P1 pressure nodes are the vertices).");
  }

  diag_.n_velocity = ufes_->GlobalTrueVSize();
  diag_.n_pressure = pfes_->GlobalTrueVSize();
  diag_.n_total = diag_.n_velocity + diag_.n_pressure;
  MPI_Comm_size(comm, &diag_.ranks);
  diag_.threads = omp_get_max_threads();
  diag_.assemble_seconds = MPI_Wtime() - t0;
  I.assembled = true;
}

void StokesSolver::refresh_data() {
  Impl& I = *impl_;
  AXF_REQUIRE(I.assembled, "refresh_data() needs an assembled system", "StokesSolver");
  AXF_REQUIRE(!I.rotated && I.drives.empty() && I.borders.empty(),
              "refresh_data(): slip, symmetry and periodic conditions are not offered with time-dependent data",
              "StokesBoundary", "Use Dirichlet, traction, do-nothing or axis conditions.");
  MPI_Comm comm = mesh_.GetComm();
  const int nattr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  *u_ = 0.0;
  for (const auto& kv : I.bc)
    if (kv.second.kind == BCKind::Dirichlet && kv.second.value) {
      mfem::Array<int> m(nattr);
      m = 0;
      m[kv.first - 1] = 1;
      u_->ProjectBdrCoefficient(*kv.second.value, m);  // P2 nodal interpolation, as in assemble()
    }
  I.f->Assemble();
  std::unique_ptr<mfem::HypreParVector> F(I.f->ParallelAssemble());
  I.Fvec = mfem::Vector(*F);
  mfem::Vector X(I.x->GetBlock(0).GetData(), I.offsets[1]);
  u_->GetTrueDofs(X);
  {
    std::vector<char> keep(X.Size(), 0);
    for (int i : I.dir_tdofs) keep[i] = 1;
    for (int i = 0; i < X.Size(); ++i)
      if (!keep[i]) X(i) = 0.0;
  }
  mfem::Vector& rp = I.rhs->GetBlock(1);
  rp = 0.0;
  I.Be->Mult(-1.0, X, 1.0, rp);
  diag_.flux_defect = diag_.flux_defect_abs = 0;
  if (diag_.nullspace_detected) {
    double loc[2] = {rp.Sum(), rp * rp}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double np = static_cast<double>(pfes_->GlobalTrueVSize());
    diag_.flux_defect_abs = std::fabs(glob[0]);
    diag_.flux_defect = glob[1] > 0 ? std::fabs(glob[0]) / std::sqrt(glob[1]) : 0.0;
    rp -= glob[0] / np;
  }
}

// Velocity nullspace. A criterion of the CONTINUOUS
// problem, independent of the mesh: a rigid motion v in span{e_x, e_y, rotation about the bounding-box centre} is
// a nullspace iff
//   (i)  it satisfies the homogeneous boundary conditions geometrically: v = 0 on Dirichlet/axis walls, v.n = 0 on
//        slip/symmetry walls, v periodic across periodic pairs (translations only) and no net flow along a pair
//        with a prescribed flow rate. Measured as the rms misfit
//          m(v)^2 = [sum_Dirichlet ∫|v|^2 + sum_slip ∫(v.n)^2 + flow-rate terms] / ∫_boundary |v|^2
//        on the (curved, isoparametric) boundary, over the 3-dimensional space of rigid motions (generalized
//        eigenproblem); m <= velocity_nullspace_tol (default 1e-3, a GEOMETRIC tolerance: a circle discretized by
//        16 or more UNIFORM P2 arcs with edge nodes at the arc midpoints has m <= 6e-4 for the rotation (+-30 %
//        spacing: 1.1e-3 at 16 arcs); an ellipse squeezed by 1 % has m = 7e-3), and
//   (ii) it has zero viscous energy in the chosen form: every rigid motion in the stress form (eps(v) = 0), only the
//        translations in the Laplacian form (grad v != 0 for a rotation, which is therefore never a nullspace there).
// A nullspace is refused (or only reported with refuse_velocity_nullspace = false). A near-nullspace (m above the
// tolerance, e.g. a slightly squeezed annulus) is solved; m of the least constrained rigid motion is reported as
// velocity_nullspace_residual (a conditioning indicator). Planar only.
void StokesSolver::detect_velocity_nullspace() {
  Impl& I = *impl_;
  diag_.velocity_nullspace = "none";
  diag_.velocity_nullspace_residual = -1;
  diag_.velocity_nullspace_discrete_residual = -1;
  diag_.velocity_nullspace_near = false;
  if (opt_.geometry != Geometry::Planar || I.vdim != 2) return;
  MPI_Comm comm = mesh_.GetComm();
  mfem::Vector pmin, pmax;
  mesh_.GetBoundingBox(pmin, pmax);
  const double cx = 0.5 * (pmin(0) + pmax(0)), cy = 0.5 * (pmin(1) + pmax(1));
  const double Ls = std::max(pmax.DistanceTo(pmin), 1e-300);
  const bool stress = opt_.form == ViscousForm::SymmetricStress;
  const int nc = stress ? 3 : 2;  // (ii): the Laplacian form has no rotation in its kernel
  auto cand = [&](const mfem::Vector& x, int k, double v[2]) {
    if (k == 0) { v[0] = 1; v[1] = 0; }
    else if (k == 1) { v[0] = 0; v[1] = 1; }
    else { v[0] = -(x(1) - cy) / Ls; v[1] = (x(0) - cx) / Ls; }
  };
  bool any_periodic = false;
  for (const auto& kv : I.bc) any_periodic = any_periodic || kv.second.kind == BCKind::Periodic;
  double N[9] = {0}, D[9] = {0}, Dv[9] = {0};
  mfem::Vector x(2), n(2);
  for (int be = 0; be < mesh_.GetNBE(); ++be) {
    const BCKind kind = I.bc[mesh_.GetBdrAttribute(be)].kind;
    if (kind == BCKind::Periodic) continue;  // interior faces of the periodic mesh
    mfem::ElementTransformation* T = mesh_.GetBdrElementTransformation(be);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(T->GetGeometryType(), 12);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const double w = ip.weight * T->Weight();
      mfem::CalcOrtho(T->Jacobian(), n);
      const double nl = n.Norml2();
      if (nl > 0) n /= nl;
      double v[3][2];
      for (int k = 0; k < 3; ++k) cand(x, k, v[k]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const double vv = v[a][0] * v[b][0] + v[a][1] * v[b][1];
          D[3 * a + b] += w * vv;
          if (kind == BCKind::Dirichlet || kind == BCKind::Axis) N[3 * a + b] += w * vv;
          if (kind == BCKind::Slip || kind == BCKind::Symmetry)
            N[3 * a + b] += w * (v[a][0] * n(0) + v[a][1] * n(1)) * (v[b][0] * n(0) + v[b][1] * n(1));
        }
    }
  }
  // volume integrals: the fallback normalization (no boundary at all) and the flow-rate rows
  std::vector<std::array<double, 3>> flow;  // ∫_Omega v_k . d_hat per flow-rate pair
  for (const auto& d : I.drives)
    if (d.kind == PeriodicDrive::FlowRate) flow.push_back({0, 0, 0});
  double vol = 0;
  for (int e = 0; e < mesh_.GetNE(); ++e) {
    mfem::ElementTransformation* T = mesh_.GetElementTransformation(e);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(T->GetGeometryType(), 6);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const double w = ip.weight * T->Weight();
      vol += w;
      double v[3][2];
      for (int k = 0; k < 3; ++k) cand(x, k, v[k]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) Dv[3 * a + b] += w * (v[a][0] * v[b][0] + v[a][1] * v[b][1]);
      size_t j = 0;
      for (const auto& d : I.drives) {
        if (d.kind != PeriodicDrive::FlowRate) continue;
        const BoundaryCondition& c = I.bc[d.attribute];
        const double dx = c.translation[0] / d.length, dy = c.translation[1] / d.length;
        for (int k = 0; k < 3; ++k) flow[j][k] += w * (v[k][0] * dx + v[k][1] * dy);
        ++j;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, N, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, D, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, Dv, 9, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_DOUBLE, MPI_SUM, comm);
  for (auto& f : flow) MPI_Allreduce(MPI_IN_PLACE, f.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
  const bool no_boundary = D[0] + D[4] + D[8] <= 0;
  Eigen::Matrix3d Nm, Dm;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      Dm(a, b) = no_boundary ? Dv[3 * a + b] : D[3 * a + b];
      Nm(a, b) = N[3 * a + b];
    }
  const double scale = Dm.trace() / 3.0;
  // a flow-rate pair forbids net flow along d (an O(1) misfit for that direction)
  for (const auto& f : flow)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) Nm(a, b) += scale * f[a] * f[b] / (vol * vol);
  // a periodic pair forbids the rotation (not periodic under a translation)
  if (any_periodic) Nm(2, 2) += Dm(2, 2);
  Eigen::MatrixXd Ns = Nm.topLeftCorner(nc, nc), Ds = Dm.topLeftCorner(nc, nc);
  for (int i = 0; i < nc; ++i) Ds(i, i) += 1e-300;
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Ns, Ds);
  const auto ev = es.eigenvalues();
  const double tol = opt_.velocity_nullspace_tol;
  diag_.velocity_nullspace_residual = std::sqrt(std::max(ev(0), 0.0));
  const char* rot = "rotation about the bounding-box centre";
  std::string found;
  // exact DISCRETE singularity, independently of the geometric test: a zero-energy rigid motion
  // whose discrete vector (rotated frame, eliminated components set to 0) is annihilated by the viscous block on
  // the free rows and by the flow-rate rows to round-off, ||A z||_free <= 1e3 eps || |A| |z| ||_free (a coarse
  // circle of <= 12 arcs: the consistent normals make the rotation exactly admissible although the polygon-like
  // boundary misfits it geometrically). Only exact singularity reaches this level, never a well-posed problem.
  {
    const int ntd = ufes_->GetTrueVSize();
    mfem::HypreParMatrix Aabs(*I.A);
    {
      hypre_ParCSRMatrix* h = Aabs;
      for (hypre_CSRMatrix* m : {hypre_ParCSRMatrixDiag(h), hypre_ParCSRMatrixOffd(h)}) {
        HYPRE_Real* d = hypre_CSRMatrixData(m);
        const HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(m);
        for (HYPRE_Int k = 0; k < nnz; ++k) d[k] = std::fabs(d[k]);
      }
    }
    std::vector<mfem::Vector> rr(nc);
    std::vector<double> sc(nc, 0.0);
    for (int k = 0; k < nc; ++k) {
      mfem::VectorFunctionCoefficient rc(2, [&, k](const mfem::Vector& xx, mfem::Vector& v) {
        double c[2];
        cand(xx, k, c);
        v.SetSize(2);
        v(0) = c[0];
        v(1) = c[1];
      });
      mfem::ParGridFunction g(ufes_.get());
      g.ProjectCoefficient(rc);
      mfem::Vector c(ntd), z(ntd), Az(ntd), Sz(ntd), za(ntd);
      g.GetTrueDofs(c);
      if (I.rotated) I.R->MultTranspose(c, z);
      else z = c;
      for (int i : I.ess_tdofs) z(i) = 0.0;
      I.A->Mult(z, Az);
      for (int i = 0; i < ntd; ++i) za(i) = std::fabs(z(i));
      Aabs.Mult(za, Sz);
      for (int i : I.ess_tdofs) Az(i) = Sz(i) = 0.0;
      double loc = Sz * Sz, glob = 0;
      MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm);
      sc[k] = std::sqrt(glob);
      // residual vector: the free momentum rows, then one entry per flow-rate row (built at its final size;
      // mfem::Vector::SetSize does not preserve the data when it grows). The flow-rate entries are global scalars:
      // stored on rank 0 only, so that the Allreduce of the Gram counts them once.
      int nflow = 0;
      for (const auto& d : I.drives) nflow += d.kind == PeriodicDrive::FlowRate;
      rr[k].SetSize(ntd + nflow);
      rr[k] = 0.0;
      for (int i = 0; i < ntd; ++i) rr[k](i) = sc[k] > 0 ? Az(i) / sc[k] : 0.0;
      int me = 0;
      MPI_Comm_rank(comm, &me);
      int j = ntd;
      for (const auto& d : I.drives) {
        if (d.kind != PeriodicDrive::FlowRate) continue;
        mfem::Vector cd(ntd), cda(ntd);
        if (I.rotated) I.R->MultTranspose(d.c, cd);
        else cd = d.c;
        for (int i = 0; i < ntd; ++i) cda(i) = std::fabs(cd(i));
        for (int i : I.ess_tdofs) cd(i) = cda(i) = 0.0;  // the eliminated DOFs are not unknowns of the row
        const double num = mfem::InnerProduct(comm, cd, z), den = mfem::InnerProduct(comm, cda, za);
        // relative to the row's own scale |c| |z| (not to the momentum scale): a rigid motion with net flow along
        // the pair violates the constraint by O(1)
        if (me == 0) rr[k](j) = den > 0 ? num / den : 0.0;
        ++j;
      }
    }
    Eigen::MatrixXd Gd(nc, nc);
    for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b) {
        double loc = (sc[a] > 0 && sc[b] > 0) ? rr[a] * rr[b] : (a == b ? 1.0 : 0.0), glob = 0;
        MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm);
        Gd(a, b) = glob;
      }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ed(Gd);
    // the eigenvalue carries round-off of order eps ||G||, so its square root bottoms out near 1e-8: recompute the
    // residual of the least-residual combination directly (accurate to round-off)
    double dres;
    {
      const auto v = ed.eigenvectors().col(0);
      mfem::Vector comb(rr[0].Size());
      comb = 0.0;
      double vn = 0;
      for (int k = 0; k < nc; ++k)
        if (sc[k] > 0) {
          comb.Add(v(k), rr[k]);
          vn += v(k) * v(k);
        }
      double loc = comb * comb, glob = 0;
      MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm);
      dres = vn > 0 ? std::sqrt(glob / vn) : 1.0;
    }
    diag_.velocity_nullspace_discrete_residual = dres;
    if (dres <= 1e3 * std::numeric_limits<double>::epsilon()) {
      const auto v = ed.eigenvectors().col(0);
      const bool isrot = nc == 3 && std::fabs(v(2)) > std::hypot(v(0), v(1));
      found = std::string(isrot ? rot : "translation") + " (discretely exact: the operator is singular, " +
              strcat("residual ", dres, ")");
    }
  }
  for (int i = 0; i < nc && found.empty(); ++i) {  // the geometric test (skipped when exactly singular)
    if (!(std::sqrt(std::max(ev(i), 0.0)) <= tol)) continue;
    const auto v = es.eigenvectors().col(i);
    const double vr = nc == 3 ? std::fabs(v(2)) * std::sqrt(Dm(2, 2)) : 0.0;
    const double vt = std::sqrt(std::max(0.0, v(0) * v(0) * Dm(0, 0) + v(1) * v(1) * Dm(1, 1)));
    std::string what = vr > vt ? std::string(rot)
                               : strcat("translation along (", v(0) / std::hypot(v(0), v(1)), ", ",
                                        v(1) / std::hypot(v(0), v(1)), ")");
    found += (found.empty() ? "" : "; ") + what;
  }
  // a near-nullspace (m below 10 tol) is solved but flagged: the answer may carry a large rigid motion
  diag_.velocity_nullspace_near = found.empty() && diag_.velocity_nullspace_residual >= 0 &&
                                  diag_.velocity_nullspace_residual < 10 * tol;
  if (found.empty()) return;
  diag_.velocity_nullspace = found;
  AXF_REQUIRE(!opt_.refuse_velocity_nullspace,
              strcat("the boundary conditions leave a velocity nullspace: ", found, " (rms boundary misfit ",
                     diag_.velocity_nullspace_residual, " <= velocity_nullspace_tol ", tol,
                     "); the solution would carry an arbitrary rigid motion"),
              "StokesBoundary",
              "Constrain it: a Dirichlet (no-slip or moving) wall, a prescribed periodic flow rate for a free "
              "translation, or for a free rotation (stress form with slip on circular walls about one centre) a "
              "Dirichlet point or the Laplacian form. (StokesOptions.refuse_velocity_nullspace = false only reports it.)");
}

void StokesSolver::rotate_operators() {
  Impl& I = *impl_;
  if (!I.rotated) return;
  I.A.reset(mfem::RAP(I.A.get(), I.R.get()));
  I.B.reset(mfem::ParMult(I.B.get(), I.R.get()));
}

long StokesSolver::nearest_pressure_dof(const std::vector<double>& pt, double* dist) const {
  // deterministic: the smallest global pressure index among the DOFs at the minimal distance (no rank dependence)
  Impl& I = *impl_;
  MPI_Comm comm = mesh_.GetComm();
  const int sdim = mesh_.SpaceDimension();
  mfem::ParGridFunction coord(pfes_.get());
  std::vector<mfem::Vector> xyz(sdim);
  for (int d = 0; d < sdim; ++d) {
    mfem::FunctionCoefficient cx([d](const mfem::Vector& x) { return x(d); });
    coord.ProjectCoefficient(cx);
    coord.GetTrueDofs(xyz[d]);
  }
  const int np_loc = pfes_->GetTrueVSize();
  std::vector<double> dd(np_loc);
  double dmin = std::numeric_limits<double>::max(), gdmin;
  for (int k = 0; k < np_loc; ++k) {
    double s = 0;
    for (int d = 0; d < sdim; ++d) s += std::pow(xyz[d](k) - pt[d], 2);
    dd[k] = std::sqrt(s);
    dmin = std::min(dmin, dd[k]);
  }
  MPI_Allreduce(&dmin, &gdmin, 1, MPI_DOUBLE, MPI_MIN, comm);
  mfem::Vector pmin, pmax;
  mesh_.GetBoundingBox(pmin, pmax);
  const double tie = gdmin + 1e-12 * pmax.DistanceTo(pmin);
  long best = std::numeric_limits<long>::max(), gbest;
  const long off = pfes_->GetMyTDofOffset();
  for (int k = 0; k < np_loc; ++k)
    if (dd[k] <= tie) best = std::min(best, off + k);
  MPI_Allreduce(&best, &gbest, 1, MPI_LONG, MPI_MIN, comm);
  int me = 0;
  MPI_Comm_rank(comm, &me);
  long owner = (gbest >= off && gbest < off + np_loc) ? me : -1, gowner;
  MPI_Allreduce(&owner, &gowner, 1, MPI_LONG, MPI_MAX, comm);
  I.pp_rank = gowner;
  I.pp_local = (me == gowner) ? gbest - off : -1;
  if (dist) *dist = gdmin;
  return gbest;
}

DistCsr StokesSolver::monolithic(bool with_nullspace, std::vector<double>* rhs_local, bool allow_singular) {
  Impl& I = *impl_;
  if (!I.assembled) assemble();
  AXF_REQUIRE(with_nullspace || !diag_.nullspace_detected || allow_singular,
              "monolithic(with_nullspace=false) on a flow with a constant-pressure nullspace: the matrix is singular",
              "", "Pass with_nullspace=true, or allow_singular=true if a singular matrix is what you want.");
  MPI_Comm comm = mesh_.GetComm();
  int me = 0, P = 1;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &P);
  mfem::Array2D<const mfem::HypreParMatrix*> blocks(2, 2);
  blocks(0, 0) = I.A.get();
  blocks(0, 1) = I.Bt.get();
  blocks(1, 0) = I.B.get();
  blocks(1, 1) = nullptr;
  std::unique_ptr<mfem::HypreParMatrix> K(mfem::HypreParMatrixFromBlocks(blocks));
  mfem::SparseMatrix merged;
  K->MergeDiagAndOffd(merged);
  DistCsr D;
  D.comm = comm;
  D.n_global = K->GetGlobalNumRows();
  D.row_begin = K->RowPart()[0];
  D.local_rows = merged.Height();
  AXF_REQUIRE(D.local_rows == I.offsets[2], "monolithic: local row count does not match the block layout");
  D.rowptr.assign(merged.GetI(), merged.GetI() + D.local_rows + 1);
  D.col.assign(merged.GetJ(), merged.GetJ() + merged.NumNonZeroElems());
  D.val.assign(merged.GetData(), merged.GetData() + merged.NumNonZeroElems());
  std::vector<double> rhs(I.offsets[2]);
  for (int i = 0; i < I.offsets[2]; ++i) rhs[i] = (*I.rhs)(i);
  const int nu_loc = I.offsets[1];

  if (with_nullspace && opt_.nullspace == Nullspace::PinPressure) {
    mfem::Vector pmin, pmax;
    mesh_.GetBoundingBox(pmin, pmax);
    std::vector<double> pt = opt_.pin_point;
    if (pt.empty())
      for (int d = 0; d < mesh_.SpaceDimension(); ++d) pt.push_back(0.5 * (pmin(d) + pmax(d)));
    nearest_pressure_dof(pt, nullptr);
    long gpin = (me == I.pp_rank) ? D.row_begin + nu_loc + I.pp_local : 0;
    MPI_Bcast(&gpin, 1, MPI_LONG, static_cast<int>(I.pp_rank), comm);
    diag_.pinned_dof = gpin;
    // zero the pin column everywhere and the pin row on its owner; unit diagonal, zero rhs
    for (int i = 0; i < D.local_rows; ++i) {
      const bool pin_row = (D.row_begin + i == gpin);
      for (int k = D.rowptr[i]; k < D.rowptr[i + 1]; ++k) {
        if (pin_row) D.val[k] = 0.0;
        else if (D.col[k] == gpin) D.val[k] = 0.0;
      }
      if (pin_row) {
        // the (p, p) block is structurally empty: reuse the row's first slot for the unit diagonal
        AXF_REQUIRE(D.rowptr[i + 1] > D.rowptr[i], "pressure pin: the pinned pressure row is empty");
        D.col[D.rowptr[i]] = static_cast<int>(gpin);
        D.val[D.rowptr[i]] = 1.0;
        rhs[i] = 0.0;
      }
    }
  }
  // bordered rows/columns [K V; V^T 0]: the flow-rate constraints, then the mean multiplier; owned by the last rank
  std::vector<const Impl::Border*> borders;
  for (const auto& b : I.borders) borders.push_back(&b);
  Impl::Border mean;
  if (with_nullspace && opt_.nullspace == Nullspace::MeanMultiplier) {
    mean.u.SetSize(nu_loc);
    mean.u = 0.0;
    mean.p = I.mean_weights;
    mean.rhs = 0.0;
    borders.push_back(&mean);
  }
  const int nb = static_cast<int>(borders.size());
  if (nb > 0) {
    const long N = D.n_global;
    std::vector<int> rp(1, 0), cc;
    std::vector<double> vv;
    cc.reserve(D.col.size() + nb * D.local_rows);
    vv.reserve(D.col.size() + nb * D.local_rows);
    for (int i = 0; i < D.local_rows; ++i) {
      for (int k = D.rowptr[i]; k < D.rowptr[i + 1]; ++k) {
        cc.push_back(D.col[k]);
        vv.push_back(D.val[k]);
      }
      for (int j = 0; j < nb; ++j) {
        const double v = i < nu_loc ? borders[j]->u(i) : borders[j]->p(i - nu_loc);
        if (v != 0.0) {
          cc.push_back(static_cast<int>(N + j));
          vv.push_back(v);
        }
      }
      rp.push_back(static_cast<int>(cc.size()));
    }
    std::vector<int> cnt(P), dsp(P + 1, 0);
    const int nloc = D.local_rows;
    MPI_Gather(&nloc, 1, MPI_INT, cnt.data(), 1, MPI_INT, P - 1, comm);
    for (int p = 0; p < P; ++p) dsp[p + 1] = dsp[p] + cnt[p];
    for (int j = 0; j < nb; ++j) {
      std::vector<double> vals(nloc);
      for (int i = 0; i < nloc; ++i) vals[i] = i < nu_loc ? borders[j]->u(i) : borders[j]->p(i - nu_loc);
      std::vector<double> all(me == P - 1 ? dsp[P] : 0);
      MPI_Gatherv(vals.data(), nloc, MPI_DOUBLE, all.data(), cnt.data(), dsp.data(), MPI_DOUBLE, P - 1, comm);
      if (me == P - 1) {
        // global row index of local row i on rank p: row_begin(p) + i; row_begin from the gathered counts
        std::vector<long> rb(P);
        std::vector<long> rbl(P);
        long myrb = D.row_begin;
        MPI_Gather(&myrb, 1, MPI_LONG, rbl.data(), 1, MPI_LONG, P - 1, comm);  // matched below on other ranks
        for (int p = 0; p < P; ++p)
          for (int i = 0; i < cnt[p]; ++i)
            if (all[dsp[p] + i] != 0.0) {
              cc.push_back(static_cast<int>(rbl[p] + i));
              vv.push_back(all[dsp[p] + i]);
            }
        cc.push_back(static_cast<int>(N + j));  // structural diagonal (value 0)
        vv.push_back(0.0);
        rp.push_back(static_cast<int>(cc.size()));
        rhs.push_back(borders[j]->rhs);
      } else {
        long myrb = D.row_begin;
        MPI_Gather(&myrb, 1, MPI_LONG, nullptr, 1, MPI_LONG, P - 1, comm);
      }
    }
    if (me == P - 1) D.local_rows += nb;
    D.rowptr.swap(rp);
    D.col.swap(cc);
    D.val.swap(vv);
    D.n_global = N + nb;
  }
  sort_rows(D);
  if (rhs_local) *rhs_local = std::move(rhs);
  return D;
}

void StokesSolver::solve_direct() {
  Impl& I = *impl_;
  std::vector<double> b;
  const double t0 = MPI_Wtime();
  DistCsr D = monolithic(true, &b);
  // the constant-pressure vector: in the kernel iff the nullspace was not removed -> refused by the LU
  std::vector<double> z(D.local_rows, 0.0);
  for (int i = I.offsets[1]; i < I.offsets[2]; ++i) z[i] = 1.0;
  // the rows the GMRES fallback's stopping test counts: the gate's (non-Dirichlet velocity rows, all pressure and
  // bordered rows)
  std::vector<char> gate_rows(D.local_rows, 1);
  for (int i = 0; i < I.offsets[1]; ++i) gate_rows[i] = I.ess_mask[i] ? 0 : 1;
  // One attempt = factorize + solve with or without MC64 matching. Returns the solution; `st` its LU statistics.
  double tfac = 0;
  auto attempt = [&](bool matching, LUStats& st, std::vector<double>& xo) {
    SparseLU lu(opt_.lu);
    lu.set_matching(matching);
    // up to 10 refinement steps (the target is unchanged): STRUMPACK's static pivoting can leave a plain-solve
    // residual of 1e-3 on traction problems at ~1.5e5 unknowns, which then needs 3+ steps (measured at N = 128)
    lu.set_refinement(10 + opt_.min_refine_steps, std::min(1e-12, 1e-2 * opt_.direct_tol));
    lu.set_min_refinement(opt_.min_refine_steps);
    lu.set_throw_on_miss(false);   // the Stokes gate (rel_residual <= direct_tol) decides `converged`
    lu.set_threads(opt_.lu_threads);
    lu.set_refine_to_stagnation(true);  // best attainable accuracy (cheap: a few extra triangular solves)
    lu.set_krylov_fallback(opt_.lu_krylov_fallback);  // 3D (sparse_lu.hpp)
    if (opt_.lu_krylov_fallback) lu.set_residual_rows(gate_rows);
    lu.set_kernel_probe(z, 1e-14);
    lu.factorize(D);
    tfac = MPI_Wtime();
    xo = lu.solve(b);
    st = lu.stats();
  };
  std::vector<double> x;
  LUStats lst;
  bool matching = opt_.lu_matching;
  if (matching) {
    attempt(true, lst, x);
  } else {
    // matching-free first; if that factorization fails (e.g. a zero pivot on the saddle point's zero block) or
    // its solve misses the gate, again with the matching (the 2D configuration)
    bool failed = false;
    std::string why;
    try {
      attempt(false, lst, x);
      failed = !std::isfinite(lst.rel_residual) || lst.rel_residual > opt_.direct_tol;
      if (failed) why = strcat("matching-free solve missed the gate (residual ", lst.rel_residual, ")");
    } catch (const Error& e) {
      failed = true;
      why = std::string("matching-free factorization failed: ") + e.what();
    }
    if (failed) {
      diag_.lu_matching_retry = true;
      diag_.lu_matching_retry_reason = why.substr(0, 300);
      diag_.lu_first_attempt_residual_before = lst.rel_residual_before;   // diagnostics only: copied before the retry overwrites lst
      diag_.lu_first_attempt_krylov_iterations = lst.krylov_iterations;
      matching = true;
      attempt(true, lst, x);
    }
  }
  const double t1 = tfac;
  diag_.setup_seconds = t1 - t0;
  diag_.solve_seconds = MPI_Wtime() - t1;
  diag_.lu = lst;
  diag_.lu.matching = matching;
  // a solve the GMRES fallback finished is not a plain direct solve; the label says so
  if (diag_.lu.krylov_fallback) diag_.solver = std::string("direct+gmres:") + to_string(opt_.lu);
  std::copy(x.begin(), x.begin() + I.offsets[2], I.x->GetData());
  // bordered unknowns (flow-rate multipliers, then the mean multiplier), owned by the last rank
  const int nb = static_cast<int>(I.borders.size()) + (opt_.nullspace == Nullspace::MeanMultiplier ? 1 : 0);
  I.lambda.assign(nb, 0.0);
  if (nb > 0) {
    int me = 0, P = 1;
    MPI_Comm_rank(mesh_.GetComm(), &me);
    MPI_Comm_size(mesh_.GetComm(), &P);
    if (me == P - 1)
      for (int j = 0; j < nb; ++j) I.lambda[j] = x[I.offsets[2] + j];
    MPI_Bcast(I.lambda.data(), nb, MPI_DOUBLE, P - 1, mesh_.GetComm());
    if (opt_.nullspace == Nullspace::MeanMultiplier) diag_.multiplier = I.lambda.back();
  }
  // the flow-rate multipliers are the pressure gradients G of their pairs
  for (size_t j = 0, k = 0; k < I.drives.size(); ++k)
    if (I.drives[k].kind == PeriodicDrive::FlowRate) diag_.periodic_pressure_gradient[k] = I.lambda[j++];
  diag_.iterations = 1 + diag_.lu.refine_steps;
}

void StokesSolver::solve_minres() {
  Impl& I = *impl_;
  MPI_Comm comm = mesh_.GetComm();
  const double t0 = MPI_Wtime();
  mfem::BlockOperator K(I.offsets);
  K.SetBlock(0, 0, I.A.get());
  K.SetBlock(0, 1, I.Bt.get());
  K.SetBlock(1, 0, I.B.get());
  mfem::HypreBoomerAMG amg(*I.A);
  amg.SetSystemsOptions(impl_->vdim, false);  // byVDIM ordering
  amg.SetPrintLevel(0);
  std::unique_ptr<mfem::Solver> pinner;
  if (opt_.pressure_amg) {
    auto* pa = new mfem::HypreBoomerAMG(*I.Mp);
    pa->SetPrintLevel(0);
    pinner.reset(pa);
  } else {
    pinner.reset(new mfem::HypreDiagScale(*I.Mp));
  }
  std::unique_ptr<mfem::OrthoSolver> ortho;
  mfem::Solver* pprec = pinner.get();
  if (opt_.nullspace == Nullspace::Ortho) {
    ortho.reset(new mfem::OrthoSolver(comm));
    ortho->SetSolver(*pinner);
    ortho->SetOperator(*I.Mp);  // required: sizes the projector and sets the inner solver's operator
    pprec = ortho.get();
  }
  mfem::BlockDiagonalPreconditioner Pc(I.offsets);
  Pc.SetDiagonalBlock(0, &amg);
  Pc.SetDiagonalBlock(1, pprec);
  mfem::MINRESSolver minres(comm);
  minres.SetRelTol(opt_.krylov_rtol);
  minres.SetAbsTol(0.0);
  minres.SetMaxIter(opt_.krylov_maxit);
  minres.SetPrintLevel(opt_.print_level);
  minres.SetPreconditioner(Pc);
  minres.SetOperator(K);
  // start from the Dirichlet lift stored in x by assemble(): with a zero start the
  // unit-diagonal Dirichlet rows dominate the initial residual when the physical rows scale like nu
  minres.iterative_mode = true;
  diag_.setup_seconds = MPI_Wtime() - t0;
  const double t1 = MPI_Wtime();
  minres.Mult(*I.rhs, *I.x);
  diag_.solve_seconds = MPI_Wtime() - t1;
  diag_.iterations = minres.GetNumIterations();
  diag_.minres_preconditioned = minres.GetFinalNorm();
  diag_.converged = minres.GetConverged();
  I.lambda.clear();
}

void StokesSolver::finish_pressure() {
  Impl& I = *impl_;
  mfem::Vector& xp = I.x->GetBlock(1);
  MPI_Comm comm = mesh_.GetComm();
  if (opt_.pressure_point.active) {
    // the pressure-point BC: shift by the constant (the nullspace) so that p_h(x0) = value; reported raw
    double val = 0;
    int me = 0;
    MPI_Comm_rank(comm, &me);
    std::vector<double> x0 = opt_.pressure_point.x0;
    double target = opt_.pressure_point.value;
    diag_.pinned_dof = nearest_pressure_dof(x0, nullptr);  // (the pin may have reused pp_*)
    if (me == I.pp_rank) val = xp(static_cast<int>(I.pp_local));
    MPI_Bcast(&val, 1, MPI_DOUBLE, static_cast<int>(I.pp_rank), comm);
    xp += target - val;
    double v2 = (me == I.pp_rank) ? xp(static_cast<int>(I.pp_local)) : 0.0;
    MPI_Bcast(&v2, 1, MPI_DOUBLE, static_cast<int>(I.pp_rank), comm);
    diag_.pressure_point_value = v2;
    return;
  }
  if (!diag_.nullspace_detected) return;  // open boundary: the pressure is determined, report it raw
  double loc[2] = {I.mean_weights * xp, I.mean_weights.Sum()}, glob[2];
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
  xp -= glob[0] / glob[1];
}

const StokesDiagnostics& StokesSolver::solve() {
  Impl& I = *impl_;
  if (!I.assembled) assemble();
  diag_.converged = false;
  if (opt_.solver == LinearSolverKind::Direct) solve_direct();
  else solve_minres();
  // residual of the consistent system actually solved, over the non-Dirichlet rows (the gate),
  // including the bordered (flow-rate) rows and columns
  mfem::BlockOperator K(I.offsets);
  K.SetBlock(0, 0, I.A.get());
  K.SetBlock(0, 1, I.Bt.get());
  K.SetBlock(1, 0, I.B.get());
  mfem::BlockVector r(I.offsets);
  K.Mult(*I.x, r);
  r -= *I.rhs;
  double cres = 0, crhs = 0;
  for (size_t j = 0; j < I.borders.size() && j < I.lambda.size(); ++j) {
    const auto& bd = I.borders[j];
    for (int i = 0; i < I.offsets[1]; ++i) r(i) += bd.u(i) * I.lambda[j];
    const double c = mfem::InnerProduct(mesh_.GetComm(), bd.u, I.x->GetBlock(0));
    cres += (c - bd.rhs) * (c - bd.rhs);
    crhs += bd.rhs * bd.rhs;
  }
  double loc[5] = {0, 0, 0, 0, 0}, glob[5];  // r_u^2, r_p^2, b_free^2, b_all^2, r_all^2
  for (int i = 0; i < I.offsets[2]; ++i) {
    const double ri = r(i), bi = (*I.rhs)(i);
    loc[3] += bi * bi;
    loc[4] += ri * ri;
    if (I.ess_mask[i]) continue;
    (i < I.offsets[1] ? loc[0] : loc[1]) += ri * ri;
    loc[2] += bi * bi;
  }
  MPI_Allreduce(loc, glob, 5, MPI_DOUBLE, MPI_SUM, mesh_.GetComm());
  glob[1] += cres;  // constraint rows (identical on every rank)
  glob[2] += crhs;
  diag_.res_momentum = std::sqrt(glob[0]);
  diag_.res_continuity = std::sqrt(glob[1]);
  const double den = glob[2] > 0 ? glob[2] : glob[3];
  diag_.rel_residual = den > 0 ? std::sqrt((glob[0] + glob[1]) / den) : std::sqrt(glob[0] + glob[1]);
  diag_.rel_residual_all_rows = glob[3] > 0 ? std::sqrt((glob[4] + cres) / (glob[3] + crhs)) : std::sqrt(glob[4]);
  if (opt_.solver == LinearSolverKind::Direct)
    diag_.converged = std::isfinite(diag_.rel_residual) && diag_.rel_residual <= opt_.direct_tol;
  else  // MFEM's preconditioned-norm flag AND the true residual
    diag_.converged = diag_.converged && std::isfinite(diag_.rel_residual) &&
                      diag_.rel_residual <= opt_.minres_true_factor * opt_.krylov_rtol;
  finish_pressure();
  // back to the Cartesian frame: u = R w
  if (I.rotated) {
    mfem::Vector w(I.x->GetBlock(0));
    I.R->Mult(w, I.x->GetBlock(0));
  }
  // the flow-rate drive enters the load of the residual-based post-processing (reactions, boundary power)
  for (size_t k = 0; k < I.drives.size(); ++k)
    if (I.drives[k].kind == PeriodicDrive::FlowRate) I.Fvec.Add(diag_.periodic_pressure_gradient[k], I.drives[k].c);
  u_->SetFromTrueDofs(I.x->GetBlock(0));
  p_->SetFromTrueDofs(I.x->GetBlock(1));
  return diag_;
}

Reaction StokesSolver::physical_reaction(const std::vector<int>& body, const std::vector<double>& center) const {
  return reaction_impl(body, center, opt_.form, true);
}

Reaction StokesSolver::reaction(const std::vector<int>& body, const std::vector<double>& center,
                                ViscousForm residual_form) const {
  return reaction_impl(body, center, residual_form, false);
}

Reaction StokesSolver::reaction_impl(const std::vector<int>& body, const std::vector<double>& center,
                                     ViscousForm residual_form, bool physical) const {
  Impl& I = *impl_;
  AXF_REQUIRE(I.assembled && diag_.converged, "reaction(): needs a converged solve");
  const bool axi = opt_.geometry == Geometry::Axisymmetric;
  MPI_Comm comm = mesh_.GetComm();
  const int nattr = mesh_.bdr_attributes.Max();
  mfem::Array<int> marker(nattr);
  marker = 0;
  for (int a : body) {
    AXF_REQUIRE(I.bc.count(a), strcat("reaction(): attribute ", a, " is not a boundary attribute of the mesh"));
    marker[a - 1] = 1;
  }
  // the residual operator: the SOLVED operator (A0: form, viscosity, grad-div, explicit quadrature) when
  // residual_form is the solved form, in planar AND axisymmetric geometry (a re-assembly that drops the grad-div term
  // lowers the drag's convergence from order 4 to ~2.4 at gamma = 10); otherwise the
  // requested form assembled on the solver's rule, with the solved grad-div. For a variable viscosity the two forms
  // differ inside the domain: only the solved form's residual is the consistent force; the other is a diagnostic.
  std::unique_ptr<mfem::HypreParMatrix> Aown;
  const mfem::HypreParMatrix* A = nullptr;
  const int qv = opt_.viscous_quad_order();
  if (residual_form == opt_.form) {
    A = I.A0.get();
  } else {
    mfem::ParBilinearForm a(ufes_.get());
    mfem::ConstantCoefficient zero(0.0), gam(opt_.grad_div);
    // the solve's viscous rule (qv) and grad-div at every construction site
    if (axi) a.AddDomainIntegrator(new AxiViscousIntegrator(*opt_.nu, I.vdim, residual_form ==
                                                                              ViscousForm::SymmetricStress,
                                                            opt_.grad_div, qv));
    else if (residual_form == ViscousForm::Laplacian)
      a.AddDomainIntegrator(new RuleBFI(new mfem::VectorDiffusionIntegrator(*opt_.nu), qv));
    else
      a.AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(zero, *opt_.nu), qv));
    if (!axi && opt_.grad_div != 0.0) a.AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(gam, zero), qv));
    a.Assemble();
    a.Finalize();
    Aown.reset(a.ParallelAssemble());
    A = Aown.get();
  }
  // the physical force of a Laplacian-form solution: + nu [(grad u^T, grad w) - (div u, div w)] (null Lagrangian)
  std::unique_ptr<mfem::HypreParMatrix> Cnl;
  if (physical && residual_form == ViscousForm::Laplacian) {
    auto* cn = dynamic_cast<mfem::ConstantCoefficient*>(opt_.nu);
    AXF_REQUIRE(cn != nullptr,
                "physical reaction of a Laplacian-form solution with a VARIABLE viscosity: the Laplacian form "
                "-div(nu grad u) is then not the Stokes momentum equation, and the force on the body is not defined",
                "reaction", "Solve in the stress form (form='stress') for a variable viscosity; residual_form="
                "'laplacian' returns the raw pseudo-traction residual (valid only on no-slip walls at rest).");
    const double nu = cn->constant;
    mfem::ConstantCoefficient pnu(nu), mnu(-nu), zero(0.0);
    mfem::ParBilinearForm c(ufes_.get());
    if (axi) {
      // 2 nu eps:eps - nu (div_r, div_r) - nu grad:grad = nu (grad^T:grad - div div), hoop entries included
      c.AddDomainIntegrator(new AxiViscousIntegrator(pnu, I.vdim, true, -nu, qv));
      c.AddDomainIntegrator(new AxiViscousIntegrator(mnu, I.vdim, false, 0.0, qv));
    } else {
      // Elasticity(lambda = -nu, mu = nu) - nu grad:grad = nu (grad^T:grad - div div)
      c.AddDomainIntegrator(new RuleBFI(new mfem::ElasticityIntegrator(mnu, pnu), qv));
      c.AddDomainIntegrator(new RuleBFI(new mfem::VectorDiffusionIntegrator(mnu), qv));
    }
    c.Assemble();
    c.Finalize();
    Cnl.reset(c.ParallelAssemble());
  }
  mfem::Vector ut, pt;
  u_->GetTrueDofs(ut);
  p_->GetTrueDofs(pt);
  mfem::Vector R(ut.Size()), tmp(ut.Size());
  A->Mult(ut, R);
  if (Cnl) Cnl->AddMult(ut, R);
  I.B0->MultTranspose(pt, tmp);  // B^T p = -(p, div v)
  R += tmp;
  R -= I.Fvec;
  // test functions: e_i (force) and the rigid rotation (torque) restricted to the body's velocity nodes
  auto restricted = [&](mfem::VectorCoefficient& field, int comp_only) {
    mfem::ParGridFunction w(ufes_.get());
    w.ProjectCoefficient(field);
    mfem::Vector wt;
    w.GetTrueDofs(wt);
    mfem::Vector keep(wt.Size());
    keep = 0.0;
    for (int c = 0; c < I.vdim; ++c) {
      if (comp_only >= 0 && c != comp_only) continue;
      mfem::Array<int> l;
      ufes_->GetEssentialTrueDofs(marker, l, c);
      for (int t : l) keep(t) = wt(t);
    }
    return -mfem::InnerProduct(comm, keep, R);
  };
  Reaction out;
  out.residual_form = physical ? "solved" : to_string(residual_form);
  const int vd = I.vdim;
  if (!axi) {
    for (int i = 0; i < mesh_.SpaceDimension(); ++i) {
      mfem::Vector e(vd);
      e = 0.0;
      e(i) = 1.0;
      mfem::VectorConstantCoefficient ec(e);
      out.force[i] = restricted(ec, i);
    }
    const double cx = center.size() > 0 ? center[0] : 0.0, cy = center.size() > 1 ? center[1] : 0.0;
    mfem::VectorFunctionCoefficient rot(vd, [=](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(vd);
      v = 0.0;
      v(0) = -(x(1) - cy);
      v(1) = x(0) - cx;
    });
    out.torque = restricted(rot, -1);
  } else {
    const double twopi = 2.0 * M_PI;
    mfem::Vector e(vd);
    e = 0.0;
    e(1) = 1.0;
    mfem::VectorConstantCoefficient ez(e);
    out.force[2] = twopi * restricted(ez, 1);  // F_z (the z axis is the mesh y)
    if (vd == 3) {
      mfem::VectorFunctionCoefficient rot(vd, [=](const mfem::Vector& x, mfem::Vector& v) {
        v.SetSize(vd);
        v = 0.0;
        v(2) = x(0);  // v_theta = r: the rigid rotation about the axis
      });
      out.torque = twopi * restricted(rot, 2);
    }
  }
  return out;
}

double StokesSolver::boundary_power(const std::vector<int>& attributes) const {
  Impl& I = *impl_;
  AXF_REQUIRE(I.assembled && diag_.converged, "boundary_power(): needs a converged solve");
  AXF_REQUIRE(opt_.geometry == Geometry::Planar, "boundary_power(): planar only");
  MPI_Comm comm = mesh_.GetComm();
  const int nattr = mesh_.bdr_attributes.Max();
  mfem::Array<int> marker(nattr);
  marker = 0;
  for (int a : attributes) {
    AXF_REQUIRE(a >= 1 && a <= nattr && I.bc.count(a),
                strcat("boundary_power(): attribute ", a, " is not a boundary attribute of the mesh"));
    AXF_REQUIRE(I.bc.at(a).kind == BCKind::Dirichlet,
                strcat("boundary_power(): attribute ", a, " is not a Dirichlet (prescribed velocity) boundary"));
    marker[a - 1] = 1;
  }
  mfem::Vector ut, pt;
  u_->GetTrueDofs(ut);
  p_->GetTrueDofs(pt);
  mfem::Vector R(ut.Size()), tmp(ut.Size());
  I.A0->Mult(ut, R);              // the solved viscous form + grad-div, Cartesian, before elimination
  I.B0->MultTranspose(pt, tmp);   // -(p, div v)
  R += tmp;
  R -= I.Fvec;
  mfem::Array<int> l;
  ufes_->GetEssentialTrueDofs(marker, l);
  double s = 0;
  for (int t : l) s += R(t) * ut(t);
  double g = 0;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
  return g;
}

const mfem::HypreParMatrix& StokesSolver::viscous_block() const {
  AXF_REQUIRE(impl_->assembled, "viscous_block(): call assemble() first");
  return *impl_->A;
}
const mfem::HypreParMatrix& StokesSolver::divergence_block() const {
  AXF_REQUIRE(impl_->assembled, "divergence_block(): call assemble() first");
  return *impl_->B;
}
const mfem::HypreParMatrix& StokesSolver::pressure_mass() const {
  AXF_REQUIRE(impl_->assembled, "pressure_mass(): call assemble() first");
  return *impl_->Mp;
}
const mfem::Array<int>& StokesSolver::essential_velocity_dofs() const { return impl_->ess_tdofs; }

std::unique_ptr<mfem::HypreParMatrix> StokesSolver::velocity_h1_gram() const {
  AXF_REQUIRE(impl_->assembled, "velocity_h1_gram(): call assemble() first");
  mfem::ParBilinearForm g(ufes_.get());
  g.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(impl_->vdim));
  g.AddDomainIntegrator(new mfem::VectorMassIntegrator());
  g.Assemble();
  g.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> G(g.ParallelAssemble());
  mfem::Array<int> ess = impl_->ess_tdofs;
  delete G->EliminateRowsCols(ess);
  return G;
}

const mfem::ParGridFunction& StokesSolver::velocity() const {
  AXF_REQUIRE(diag_.converged, "the Stokes solve did not converge; refusing to return its velocity",
              "", strcat("solver ", diag_.solver, ", iterations ", diag_.iterations, ", relative residual ",
                         diag_.rel_residual, " > direct_tol ", opt_.direct_tol, ". Ill-scaled problems (a large "
                         "viscosity contrast, a high aspect ratio) stagnate above the default gate: loosen direct_tol "
                         "knowingly; a singular one (an unconstrained velocity, incompatible data) does not converge "
                         "at all. Inspect diagnostics()."));
  return *u_;
}

const mfem::ParGridFunction& StokesSolver::pressure() const {
  AXF_REQUIRE(diag_.converged, "the Stokes solve did not converge; refusing to return its pressure", "",
              strcat("solver ", diag_.solver, ", iterations ", diag_.iterations, ", relative residual ",
                     diag_.rel_residual));
  return *p_;
}

}  // namespace axf::fluids
