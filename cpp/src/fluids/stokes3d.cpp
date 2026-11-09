// 3D additions to StokesSolver:
//  * 3D slip / symmetry: strong nodal frames with one, two or three normal constraints per velocity node.
// Planar geometry with a 3D mesh only (StokesSolver::setup_slip dispatches here).
#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/fluids/stokes.hpp"
#include "stokes_internal.hpp"

namespace axf::fluids {

namespace {
double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
bool normalize3(double* a) {
  const double l = std::sqrt(dot3(a, a));
  if (!(l > 0) || !std::isfinite(l)) return false;
  for (int i = 0; i < 3; ++i) a[i] /= l;
  return true;
}
void cross3(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
}  // namespace

// 3D slip / symmetry. Per velocity node and per slip/symmetry ATTRIBUTE the
// consistent (mass-conserving) normal sum_e ∮ phi_k n ds of that attribute's boundary elements is accumulated (it is
// the exact face normal on a planar face); a user `normal` coefficient overrides it. Normals of different attributes
// that differ by more than slip_corner_angle_deg are separate constraints: one normal -> u.n = 0 (frame n, t1, t2),
// two -> u.n1 = u.n2 = 0 (u along n1 x n2: the slip-symmetry edge), three or more, or a corner inside one attribute
// (its unit normals spread beyond the angle, as in 2D) -> u = 0. Dirichlet wins at shared nodes. All sums are reduced
// onto the owning true DOF (parallel safe; the three components of a node share an owner).
void StokesSolver::setup_3d() {
  Impl& I = *impl_;
  I.rotated = false;
  I.slip_ess.SetSize(0);
  diag_.n_slip_nodes = diag_.n_slip_corner_nodes = 0;
  diag_.slip_max_averaged_kink_deg = 0;
  MPI_Comm comm = mesh_.GetComm();
  AXF_REQUIRE(I.vdim == 3, "setup_3d: three velocity components expected");

  std::vector<int> slip_attrs;
  for (const auto& kv : I.bc)
    if (kv.second.kind == BCKind::Slip || kv.second.kind == BCKind::Symmetry) slip_attrs.push_back(kv.first);
  if (!slip_attrs.empty()) {
    const int nl = ufes_->GetVSize(), nt = ufes_->GetTrueVSize(), na = static_cast<int>(slip_attrs.size());
    const mfem::HypreParMatrix* P = ufes_->Dof_TrueDof_Matrix();
    const double angle = opt_.slip_corner_angle_deg;
    const double cos_half = std::cos(0.5 * angle * M_PI / 180.0), cos_full = std::cos(angle * M_PI / 180.0);
    // per attribute: consistent normal (3 slots), unit normals (3 slots), count, user normal (3 slots), user count
    // absw: sum of ∮ |phi_k| |n| ds (the scale of the consistent normal: a P2 VERTEX function integrates to 0 on a
    // planar triangle, so there the consistent normal vanishes and every tangential frame conserves mass; the
    // averaged geometric normal is used instead)
    std::vector<mfem::Vector> cons(na), unit(na), cnt(na), user(na), ucnt(na), absw(na);
    for (int a = 0; a < na; ++a)
      for (auto* v : {&cons[a], &unit[a], &cnt[a], &user[a], &ucnt[a], &absw[a]}) {
        v->SetSize(nl);
        *v = 0.0;
      }
    mfem::Array<int> dofs;
    mfem::Vector shape, n(3);
    for (int a = 0; a < na; ++a) {
      const int attr = slip_attrs[a];
      const BoundaryCondition& c = I.bc[attr];
      for (int be = 0; be < mesh_.GetNBE(); ++be) {
        if (mesh_.GetBdrAttribute(be) != attr) continue;
        ufes_->GetBdrElementDofs(be, dofs);
        const mfem::FiniteElement* fe = ufes_->GetBE(be);
        mfem::FaceElementTransformations* FT = mesh_.GetBdrFaceTransformations(be);
        shape.SetSize(fe->GetDof());
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(FT->GetGeometryType(), 2 * fe->GetOrder() + 4);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);
          FT->SetAllIntPoints(&ip);
          fe->CalcShape(ip, shape);
          mfem::CalcOrtho(FT->Jacobian(), n);  // outward from the adjacent element, length = |J|
          for (int k = 0; k < dofs.Size(); ++k) {
            for (int d = 0; d < 3; ++d) cons[a](ufes_->DofToVDof(dofs[k], d)) += ip.weight * shape(k) * n(d);
            absw[a](ufes_->DofToVDof(dofs[k], 0)) += ip.weight * std::fabs(shape(k)) * n.Norml2();
          }
        }
        const mfem::IntegrationRule& nodes = fe->GetNodes();
        for (int k = 0; k < dofs.Size(); ++k) {
          const mfem::IntegrationPoint& ip = nodes.IntPoint(k);
          FT->SetAllIntPoints(&ip);
          mfem::CalcOrtho(FT->Jacobian(), n);
          const double len = n.Norml2();
          AXF_REQUIRE(len > 0, strcat("slip attribute ", attr, ": degenerate boundary element ", be));
          n /= len;
          for (int d = 0; d < 3; ++d) unit[a](ufes_->DofToVDof(dofs[k], d)) += n(d);
          cnt[a](ufes_->DofToVDof(dofs[k], 0)) += 1.0;
          if (c.normal) {
            mfem::Vector g(3);
            c.normal->Eval(g, *FT->Elem1, FT->GetElement1IntPoint());
            const double gl = g.Norml2();
            AXF_REQUIRE(gl > 0 && std::isfinite(gl), strcat("slip attribute ", attr, ": the normal coefficient is zero "
                                                            "or not finite at a node"), "StokesBoundary");
            if (g * n < 0) g.Neg();
            for (int d = 0; d < 3; ++d) user[a](ufes_->DofToVDof(dofs[k], d)) += g(d) / gl;
            ucnt[a](ufes_->DofToVDof(dofs[k], 0)) += 1.0;
          }
        }
      }
    }
    std::vector<mfem::Vector> cons_t(na), unit_t(na), cnt_t(na), user_t(na), ucnt_t(na), absw_t(na);
    for (int a = 0; a < na; ++a) {
      for (auto pr : {std::make_pair(&cons[a], &cons_t[a]), std::make_pair(&unit[a], &unit_t[a]),
                      std::make_pair(&cnt[a], &cnt_t[a]), std::make_pair(&user[a], &user_t[a]),
                      std::make_pair(&ucnt[a], &ucnt_t[a]), std::make_pair(&absw[a], &absw_t[a])}) {
        pr.second->SetSize(nt);
        P->MultTranspose(*pr.first, *pr.second);
      }
    }
    std::vector<char> is_dir(nt, 0);
    for (int t : I.dir_tdofs) is_dir[t] = 1;
    I.Rdiag = mfem::SparseMatrix(nt, nt);
    std::vector<char> rot_row(nt, 0);
    long nslip = 0, ncorner = 0, nedge = 0;
    double max_kink = 0.0;
    for (int dof = 0; dof < ufes_->GetNDofs(); ++dof) {
      int t[3];
      for (int d = 0; d < 3; ++d) t[d] = ufes_->GetLocalTDofNumber(ufes_->DofToVDof(dof, d));
      if (t[0] < 0) continue;  // not owned here
      std::vector<std::array<double, 3>> normals;
      bool corner = false;
      for (int a = 0; a < na; ++a) {
        const double m = cnt_t[a](t[0]);
        if (m <= 0) continue;
        double us[3] = {unit_t[a](t[0]), unit_t[a](t[1]), unit_t[a](t[2])};
        const double spread = std::sqrt(dot3(us, us)) / m;
        if (spread < cos_half) {
          corner = true;
          continue;
        }
        if (m > 1) max_kink = std::max(max_kink, 2.0 * std::acos(std::min(1.0, spread)) * 180.0 / M_PI);
        std::array<double, 3> nn;
        const bool has_user = ucnt_t[a](t[0]) > 0;
        for (int d = 0; d < 3; ++d) nn[d] = has_user ? user_t[a](t[d]) : cons_t[a](t[d]);
        if (!has_user && std::sqrt(dot3(nn.data(), nn.data())) <= 1e-8 * absw_t[a](t[0]))
          for (int d = 0; d < 3; ++d) nn[d] = us[d];  // degenerate consistent normal (P2 vertex): geometric normal
        AXF_REQUIRE(normalize3(nn.data()), "slip/symmetry: zero nodal normal (the boundary folds back on itself?)",
                    "StokesBoundary", "Give the slip condition an explicit normal coefficient.");
        // merge with an earlier attribute's normal when they differ by less than the corner angle
        bool merged = false;
        for (auto& o : normals)
          if (dot3(o.data(), nn.data()) >= cos_full) {
            for (int d = 0; d < 3; ++d) o[d] += nn[d];
            normalize3(o.data());
            merged = true;
            break;
          }
        if (!merged) normals.push_back(nn);
      }
      if (normals.empty() && !corner) continue;  // not a slip node
      const bool dir_node = is_dir[t[0]] || is_dir[t[1]] || is_dir[t[2]];
      if (dir_node) continue;  // Dirichlet wins
      if (corner || normals.size() >= 3) {
        for (int d = 0; d < 3; ++d) I.slip_ess.Append(t[d]);
        ++ncorner;
        continue;
      }
      double e[3][3];
      int nconstr = 1;
      for (int d = 0; d < 3; ++d) e[0][d] = normals[0][d];
      if (normals.size() == 2) {
        const double c01 = dot3(normals[1].data(), e[0]);
        for (int d = 0; d < 3; ++d) e[1][d] = normals[1][d] - c01 * e[0][d];
        AXF_REQUIRE(normalize3(e[1]), "slip/symmetry: two parallel normals at a node (internal error)");
        nconstr = 2;
        ++nedge;
      } else {
        // any unit vector orthogonal to e0: start from the axis least aligned with it
        int ax = 0;
        for (int d = 1; d < 3; ++d)
          if (std::fabs(e[0][d]) < std::fabs(e[0][ax])) ax = d;
        double h[3] = {0, 0, 0};
        h[ax] = 1.0;
        cross3(e[0], h, e[1]);
        normalize3(e[1]);
      }
      cross3(e[0], e[1], e[2]);
      normalize3(e[2]);
      // u = R w, R = [e0 e1 e2] (columns); w_j lives in the slot of component j
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) I.Rdiag.Set(t[i], t[j], e[j][i]);
      for (int d = 0; d < 3; ++d) rot_row[t[d]] = 1;
      for (int j = 0; j < nconstr; ++j) I.slip_ess.Append(t[j]);
      ++nslip;
    }
    for (int i = 0; i < nt; ++i)
      if (!rot_row[i]) I.Rdiag.Set(i, i, 1.0);
    I.Rdiag.Finalize();
    long loc[3] = {nslip, ncorner, nedge}, glob[3];
    MPI_Allreduce(loc, glob, 3, MPI_LONG, MPI_SUM, comm);
    diag_.n_slip_nodes = glob[0];
    diag_.n_slip_corner_nodes = glob[1];
    MPI_Allreduce(&max_kink, &diag_.slip_max_averaged_kink_deg, 1, MPI_DOUBLE, MPI_MAX, comm);
    I.rotated = glob[0] > 0;
    if (I.rotated)
      I.R.reset(new mfem::HypreParMatrix(comm, ufes_->GlobalTrueVSize(), ufes_->GetTrueDofOffsets(), &I.Rdiag));
  }
}

}  // namespace axf::fluids
