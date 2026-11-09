// Inductionless-MHD Lorentz coupling: see cpp/include/axionfem/fluids/mhd.hpp.
#include "axionfem/fluids/mhd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "axionfem/core.hpp"
#include "mhd_internal.hpp"

namespace axf::fluids {

const char* to_string(ElectricKind k) {
  switch (k) {
    case ElectricKind::Insulating: return "insulating";
    case ElectricKind::NormalCurrent: return "normal_current";
    case ElectricKind::Potential: return "potential";
  }
  return "?";
}

namespace {

void cross3(const double* a, const double* b, double* o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}

// New matrix with the flagged rows and columns removed (rz / cz may be empty = none); `unit`: 1 on the diagonal of a
// flagged row (square matrices).
std::unique_ptr<mfem::SparseMatrix> masked(const mfem::SparseMatrix& A, const std::vector<char>& rz,
                                           const std::vector<char>& cz, bool unit) {
  const int m = A.Height(), n = A.Width();
  const int* I = A.GetI();
  const int* J = A.GetJ();
  const double* D = A.GetData();
  int* ni = new int[m + 1];
  std::vector<int> nj;
  std::vector<double> nd;
  ni[0] = 0;
  for (int i = 0; i < m; ++i) {
    const bool rflag = !rz.empty() && rz[i];
    if (rflag && unit) {
      nj.push_back(i);
      nd.push_back(1.0);
    } else if (!rflag) {
      for (int k = I[i]; k < I[i + 1]; ++k) {
        if (!cz.empty() && cz[J[k]]) continue;
        nj.push_back(J[k]);
        nd.push_back(D[k]);
      }
    }
    ni[i + 1] = static_cast<int>(nj.size());
  }
  int* pj = new int[nj.size() ? nj.size() : 1];
  double* pd = new double[nd.size() ? nd.size() : 1];
  std::copy(nj.begin(), nj.end(), pj);
  std::copy(nd.begin(), nd.end(), pd);
  return std::make_unique<mfem::SparseMatrix>(ni, pj, pd, m, n, true, true, false);
}

// J as a linear map of (u, grad phi, 1):  J_c = ju[c][k] u_k + jp[c][d] d_d phi + j0[c]
struct JMap {
  double ju[3][3], jp[3][2], j0[3];
};

JMap make_J(double sigma, const double* B, double E) {
  JMap J{};
  for (int k = 0; k < 3; ++k) {
    double ek[3] = {0, 0, 0}, c[3];
    ek[k] = 1.0;
    cross3(ek, B, c);
    for (int i = 0; i < 3; ++i) J.ju[i][k] = sigma * c[i];
  }
  for (int d = 0; d < 2; ++d) J.jp[d][d] = -sigma;
  J.j0[2] = sigma * E;
  return J;
}

}  // namespace

MhdCoupling::MhdCoupling(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& ufes, const MhdOptions& o,
                         const mfem::Array<int>& u_ess) {
  const double t0 = MPI_Wtime();
  int nranks = 1;
  MPI_Comm_size(mesh.GetComm(), &nranks);
  AXF_REQUIRE(nranks == 1, "MHD: the Lorentz coupling is offered on one MPI rank only", "MPI",
              "Run the MHD solver with one rank (multi-rank MHD is not tested).");
  AXF_REQUIRE(o.sigma != nullptr && o.B != nullptr, "MHD: sigma and B are required", "MhdOptions");
  AXF_REQUIRE(o.B->GetVDim() == 3, "MHD: the imposed field B needs 3 components", "MhdOptions.B");
  AXF_REQUIRE(mesh.Dimension() == 2, "MHD: 2D (x, y) meshes only", "MhdOptions");
  AXF_REQUIRE(ufes.GetVDim() == 3, "MHD needs the three-component (2.5D) velocity space", "StokesOptions.geometry",
              "Use geometry = 'planar25d' (axisymmetric MHD is not offered).");
  AXF_REQUIRE(std::isfinite(o.E_z), "MHD: E_z must be finite", "MhdOptions.E_z");
  const int pphi = o.potential_order;
  AXF_REQUIRE(pphi >= 1 && pphi <= 4, "MHD: potential_order must be in 1..4", "MhdOptions.potential_order");
  const int pu = ufes.FEColl()->GetOrder();
  const int quad = o.quad_order > 0 ? o.quad_order : 2 * std::max(pu, pphi) + 4;
  const int quadb = o.quad_order_boundary > 0 ? o.quad_order_boundary : 2 * pphi + 8;
  diag_.active = true;
  diag_.potential_order = pphi;
  diag_.quad_order = quad;
  diag_.quad_order_boundary = quadb;

  // single rank: true dofs are the vdofs
  for (int i = 0; i < ufes.GetVSize(); ++i)
    AXF_REQUIRE(ufes.GetLocalTDofNumber(i) == i, "MHD: unexpected true-dof numbering", "MPI");

  phifec_ = std::make_unique<mfem::H1_FECollection>(pphi, 2);
  phifes_ = std::make_unique<mfem::FiniteElementSpace>(&mesh, phifec_.get());
  nphi_ = phifes_->GetVSize();
  na_ = nphi_;
  diag_.n_potential = nphi_;
  diag_.n_aux = na_;
  const int nu = ufes.GetVSize();

  // ---- electric conditions: exactly one per boundary attribute
  const int nattr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  std::vector<ElectricKind> kind(nattr + 1, ElectricKind::Insulating);
  std::vector<mfem::Coefficient*> val(nattr + 1, nullptr);
  std::vector<char> given(nattr + 1, 0);
  for (const auto& c : o.electric) {
    AXF_REQUIRE(c.attribute >= 1 && c.attribute <= nattr, strcat("MHD: electric condition on attribute ", c.attribute,
                                                                 ", which the mesh does not carry"), "electric");
    AXF_REQUIRE(!given[c.attribute], strcat("MHD: attribute ", c.attribute, " has more than one electric condition"),
                "electric");
    AXF_REQUIRE(c.kind != ElectricKind::NormalCurrent || c.value != nullptr,
                strcat("MHD: normal-current condition on attribute ", c.attribute, " needs a value j_n"), "electric");
    given[c.attribute] = 1;
    kind[c.attribute] = c.kind;
    val[c.attribute] = c.value;
  }
  for (int a = 1; a <= nattr; ++a) {
    bool in_mesh = false;
    for (int b : mesh.bdr_attributes) in_mesh = in_mesh || b == a;
    AXF_REQUIRE(!in_mesh || given[a], strcat("MHD: boundary attribute ", a, " has no electric condition"), "electric",
                "Give every boundary attribute insulating, normal_current or potential.");
  }

  // ---- Dirichlet potential: nodal interpolation, essential dofs
  mfem::GridFunction g(phifes_.get());
  g = 0.0;
  std::vector<char> az(na_, 0);
  bool any_dir = false;
  for (int a = 1; a <= nattr; ++a) {
    if (kind[a] != ElectricKind::Potential) continue;
    any_dir = true;
    mfem::Array<int> mk(nattr), l;
    mk = 0;
    mk[a - 1] = 1;
    if (val[a]) g.ProjectBdrCoefficient(*val[a], mk);
    phifes_->GetEssentialTrueDofs(mk, l);
    for (int d : l) az[d] = 1;
  }
  lift_.SetSize(na_);
  lift_ = 0.0;
  for (int d = 0; d < nphi_; ++d)
    if (az[d]) {
      lift_(d) = g(d);
      ++diag_.n_potential_dirichlet;
    }
  diag_.phi_nullspace_declared = !any_dir;

  // ---- assembly
  mfem::SparseMatrix Kuu(nu, nu), Kua(nu, na_), Kau(na_, nu), Kaa(na_, na_);
  Fu_.SetSize(nu);
  Fu_ = 0.0;
  Fa_.SetSize(na_);
  Fa_ = 0.0;
  double src_abs = 0.0;  // int |q| + int |j_n| (sets the scale of the compatibility test of a nullspace problem)
  mfem::Array<int> uv, pd, cols;
  mfem::Vector N, M;
  mfem::DenseMatrix dM;
  const double E = o.E_z;
  mfem::Vector bv(3);

  auto eval_B = [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip, double* B) {
    o.B->Eval(bv, T, ip);
    for (int i = 0; i < 3; ++i) B[i] = bv(i);
  };
  auto eval_sigma = [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
    const double s = o.sigma->Eval(T, ip);
    AXF_REQUIRE(std::isfinite(s) && s > 0, strcat("MHD: sigma = ", s, " must be positive and finite"), "sigma");
    return s;
  };

  for (int e = 0; e < mesh.GetNE(); ++e) {
    mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
    const mfem::FiniteElement& fu = *ufes.GetFE(e);
    const mfem::FiniteElement& fp = *phifes_->GetFE(e);
    const int nd = fu.GetDof(), np = fp.GetDof();
    ufes.GetElementVDofs(e, uv);
    phifes_->GetElementDofs(e, pd);
    N.SetSize(nd);
    M.SetSize(np);
    dM.SetSize(np, 2);
    mfem::DenseMatrix Kuu_e(3 * nd, 3 * nd), Kuphi_e(3 * nd, np), Kphiu_e(np, 3 * nd), Kphiphi_e(np, np);
    mfem::Vector Fu_e(3 * nd), Fphi_e(np);
    Kuu_e = 0.0;
    Kuphi_e = 0.0;
    Kphiu_e = 0.0;
    Kphiphi_e = 0.0;
    Fu_e = 0.0;
    Fphi_e = 0.0;
    // ---------------- momentum rows: rule `quad`
    {
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fu.GetGeomType(), quad);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        const double w = ip.weight * T.Weight();
        fu.CalcShape(ip, N);
        fp.CalcShape(ip, M);
        fp.CalcPhysDShape(T, dM);
        const double sig = eval_sigma(T, ip);
        double Bf[3];
        eval_B(T, ip, Bf);
        JMap Jforce = make_J(sig, Bf, E);
        // F = Jforce x Bf = Fu_ku u_k + Fphi_d d_d phi + F0
        double Fuk[3][3], Fphi[3][2], F0[3];
        for (int k = 0; k < 3; ++k) {
          double col[3] = {Jforce.ju[0][k], Jforce.ju[1][k], Jforce.ju[2][k]}, c[3];
          cross3(col, Bf, c);
          for (int i = 0; i < 3; ++i) Fuk[i][k] = c[i];
        }
        for (int d = 0; d < 2; ++d) {
          double col[3] = {Jforce.jp[0][d], Jforce.jp[1][d], Jforce.jp[2][d]}, c[3];
          cross3(col, Bf, c);
          for (int i = 0; i < 3; ++i) Fphi[i][d] = c[i];
        }
        {
          double c[3];
          cross3(Jforce.j0, Bf, c);
          for (int i = 0; i < 3; ++i) F0[i] = c[i];
        }
        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < nd; ++a) {
            Fu_e(c * nd + a) += w * N(a) * F0[c];
            for (int k = 0; k < 3; ++k)
              for (int b = 0; b < nd; ++b) Kuu_e(c * nd + a, k * nd + b) -= w * N(a) * N(b) * Fuk[c][k];
            for (int b = 0; b < np; ++b)
              for (int d = 0; d < 2; ++d) Kuphi_e(c * nd + a, b) -= w * N(a) * dM(b, d) * Fphi[c][d];
          }
      }
    }
    // ---------------- potential row: rule `quad`
    {
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fu.GetGeomType(), quad);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        const double w = ip.weight * T.Weight();
        fu.CalcShape(ip, N);
        fp.CalcShape(ip, M);
        fp.CalcPhysDShape(T, dM);
        const double sig = eval_sigma(T, ip);
        double Br[3];
        eval_B(T, ip, Br);
        const JMap J = make_J(sig, Br, E);
        for (int a = 0; a < np; ++a)
          for (int b = 0; b < np; ++b)
            Kphiphi_e(a, b) += w * sig * (dM(a, 0) * dM(b, 0) + dM(a, 1) * dM(b, 1));
        for (int a = 0; a < np; ++a)
          for (int k = 0; k < 3; ++k)
            for (int b = 0; b < nd; ++b)
              Kphiu_e(a, k * nd + b) -= w * N(b) * (J.ju[0][k] * dM(a, 0) + J.ju[1][k] * dM(a, 1));
        if (o.q) {
          const double qv = o.q->Eval(T, ip);
          for (int a = 0; a < np; ++a) Fphi_e(a) += w * qv * M(a);
          src_abs += std::fabs(w * qv);
        }
      }
    }
    // ---------------- scatter
    Kuu.AddSubMatrix(uv, uv, Kuu_e, 1);
    cols.SetSize(np);
    for (int b = 0; b < np; ++b) cols[b] = pd[b];
    Kua.AddSubMatrix(uv, cols, Kuphi_e, 1);
    Kau.AddSubMatrix(cols, uv, Kphiu_e, 1);
    Kaa.AddSubMatrix(cols, cols, Kphiphi_e, 1);
    for (int i = 0; i < nd * 3; ++i) Fu_(uv[i]) += Fu_e(i);
    for (int a = 0; a < np; ++a) Fa_(pd[a]) += Fphi_e(a);
  }
  // ---- boundary: Fa -= <j_n, psi>
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    const int a = mesh.GetBdrAttribute(be);
    if (kind[a] != ElectricKind::NormalCurrent) continue;
    mfem::ElementTransformation& Tb = *mesh.GetBdrElementTransformation(be);
    const mfem::FiniteElement& fb = *phifes_->GetBE(be);
    phifes_->GetBdrElementDofs(be, pd);
    mfem::Vector Nb(fb.GetDof());
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fb.GetGeomType(), quadb);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tb.SetIntPoint(&ip);
      fb.CalcShape(ip, Nb);
      const double jn = val[a]->Eval(Tb, ip);
      const double w = ip.weight * Tb.Weight();
      for (int i = 0; i < fb.GetDof(); ++i) Fa_(pd[i]) -= w * jn * Nb(i);
      src_abs += std::fabs(w * jn);
    }
  }
  Kuu.Finalize();
  Kua.Finalize();
  Kau.Finalize();
  Kaa.Finalize();

  // ---- potential nullspace: algebraic detection on the Dirichlet-eliminated phi block
  {
    auto KD = masked(Kaa, az, az, true);
    std::vector<double> r(nphi_, 0.0);
    double fro = 0.0;
    long nfree = 0;
    for (int i = 0; i < nphi_; ++i) {
      if (az[i]) continue;
      ++nfree;
      for (int k = KD->GetI()[i]; k < KD->GetI()[i + 1]; ++k) {
        const int j = KD->GetJ()[k];
        if (j >= nphi_) continue;
        r[i] += KD->GetData()[k];
        fro += KD->GetData()[k] * KD->GetData()[k];
      }
    }
    double rn = 0.0;
    for (double v : r) rn += v * v;
    diag_.phi_kernel_defect = (fro > 0 && nfree > 0) ? std::sqrt(rn) / (std::sqrt(fro) * std::sqrt((double)nfree)) : 0.0;
    diag_.phi_nullspace_detected = diag_.phi_kernel_defect <= 1e-10;
    AXF_REQUIRE(diag_.phi_nullspace_detected == diag_.phi_nullspace_declared,
                diag_.phi_nullspace_declared
                    ? strcat("the electric conditions declare no potential boundary, but the assembled potential block "
                             "has no constant nullspace (defect ", diag_.phi_kernel_defect, ")")
                    : strcat("the electric conditions declare a potential boundary, but the potential block has a "
                             "constant nullspace (defect ", diag_.phi_kernel_defect, ")"),
                "electric", "Check the attribute numbers and kinds.");
  }
  std::vector<char> azf = az;
  diag_.phi_pinned_dof = -1;
  if (diag_.phi_nullspace_detected) {
    // project the right-hand side onto 1-perp (compatibility: int q = <j_n, 1>), pin the node nearest the box centre
    double sum = 0.0;
    for (int d = 0; d < nphi_; ++d) sum += Fa_(d);
    diag_.phi_flux_defect_abs = std::fabs(sum);
    // Incompatible data (int q != <j_n, 1>) would be projected away silently and a DIFFERENT problem solved (a
    // uniform nodal sink): refuse beyond a round-off-level tolerance.  Compatible data (the verification cases:
    // defect <= 1.4e-17 relative to the data) pass by many orders of magnitude.
    const double flux_tol = 1e-8 * src_abs + 1e-10;  // absolute floor: round-off-level data (an MMS source q ~ 1e-14)
    AXF_REQUIRE(diag_.phi_flux_defect_abs <= flux_tol,
                strcat("MHD: the potential data are incompatible: with no potential boundary the net current injected "
                       "(int q - <j_n, 1> = ", sum, ") must vanish, but it exceeds 1e-8 (int|q| + int|j_n| = ", src_abs, ") + 1e-10"),
                "MhdOptions.q / electric",
                "Make int q equal the net outward normal current (div J = q with insulating or normal-current walls "
                "only), or add a potential (grounded) boundary attribute.");
    for (int d = 0; d < nphi_; ++d) Fa_(d) -= sum / nphi_;
    mfem::Vector pmin, pmax;
    mesh.GetBoundingBox(pmin, pmax);
    const double cx = 0.5 * (pmin(0) + pmax(0)), cy = 0.5 * (pmin(1) + pmax(1));
    double best = std::numeric_limits<double>::max();
    int bd = 0;
    for (int e = 0; e < mesh.GetNE(); ++e) {
      phifes_->GetElementDofs(e, pd);
      const mfem::IntegrationRule& nodes = phifes_->GetFE(e)->GetNodes();
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      for (int k = 0; k < pd.Size(); ++k) {
        double x[3];
        mfem::Vector X(x, 2);
        T.Transform(nodes.IntPoint(k), X);
        const double d2 = (x[0] - cx) * (x[0] - cx) + (x[1] - cy) * (x[1] - cy);
        if (d2 < best - 1e-14 || (std::fabs(d2 - best) <= 1e-14 && pd[k] < bd)) {
          best = d2;
          bd = pd[k];
        }
      }
    }
    azf[bd] = 1;
    diag_.phi_pinned_dof = bd;
  }
  ess_mask_ = azf;
  // Unit scaling of the potential unknown and its row (a pure change of variables, invisible outside this class): the
  // potential is of the order B u L and its row couples with sigma B, so with phi = s a, s = max|B|, and the potential
  // row divided by rho = max(sigma) max|B|, the blocks Kaa, Kau, Kua are all O(1)..O(sigma B^2) instead of
  // (sigma, sigma B, sigma B) spread over many decades.  This is a CHANGE OF MEASURE of the dual-norm residual (the
  // potential row is down-weighted by 1/row_scale), NOT a lower backward error of the solve (in one test the same state
  // measured 2.8e-10 in the unscaled norm against 2e-10 of the unscaled solve).
  phi_scale_ = 1.0;
  row_scale_ = 1.0;
  {
    double bmax = 0.0, smax = 0.0;
    for (int e = 0; e < mesh.GetNE(); ++e) {
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      const mfem::IntegrationPoint& c = mfem::Geometries.GetCenter(T.GetGeometryType());
      T.SetIntPoint(&c);
      mfem::Vector b(3);
      o.B->Eval(b, T, c);
      bmax = std::max(bmax, b.Norml2());
      smax = std::max(smax, o.sigma->Eval(T, c));
    }
    if (bmax > 1e-8 && smax > 0) {
      phi_scale_ = bmax;
      row_scale_ = smax * bmax;
    }
  }
  if (phi_scale_ != 1.0 || row_scale_ != 1.0) {
    Kua *= phi_scale_;
    Kau *= 1.0 / row_scale_;
    Kaa *= phi_scale_ / row_scale_;
    for (int i = 0; i < na_; ++i) {
      Fa_(i) /= row_scale_;
      lift_(i) /= phi_scale_;
    }
  }
  diag_.phi_scale = phi_scale_;
  diag_.row_scale = row_scale_;
  std::vector<char> uz(nu, 0);
  for (int i : u_ess) uz[i] = 1;
  Kuu_ = std::make_unique<mfem::SparseMatrix>(Kuu);
  Kua_ = std::make_unique<mfem::SparseMatrix>(Kua);
  Kau_ = std::make_unique<mfem::SparseMatrix>(Kau);
  Kaa_ = std::make_unique<mfem::SparseMatrix>(Kaa);
  Kuu_e_ = masked(Kuu, uz, uz, false);
  Kua_e_ = masked(Kua, uz, azf, false);
  Kau_e_ = masked(Kau, azf, uz, false);
  Kaa_e_ = masked(Kaa, azf, azf, true);
  // mass of the dual norm: P(phi) mass, identity on the essential rows
  {
    mfem::SparseMatrix Ma(na_, na_);
    mfem::BilinearForm bf(phifes_.get());
    bf.AddDomainIntegrator(new mfem::MassIntegrator);
    bf.Assemble();
    bf.Finalize();
    const mfem::SparseMatrix& S = bf.SpMat();
    for (int i = 0; i < S.Height(); ++i)
      for (int k = S.GetI()[i]; k < S.GetI()[i + 1]; ++k) Ma.Add(i, S.GetJ()[k], S.GetData()[k]);
    Ma.Finalize();
    Ma_e_ = masked(Ma, azf, azf, true);
  }
  phi_gf_ = std::make_unique<mfem::GridFunction>(phifes_.get());
  *phi_gf_ = 0.0;
  diag_.assemble_seconds = MPI_Wtime() - t0;
}

void MhdCoupling::add_force(const mfem::Vector& u, const mfem::Vector& a, mfem::Vector& Ru) const {
  Kuu_->AddMult(u, Ru, 1.0);
  Kua_->AddMult(a, Ru, 1.0);
  Ru -= Fu_;
}

void MhdCoupling::aux_residual(const mfem::Vector& u, const mfem::Vector& a, mfem::Vector& Ra) const {
  Ra.SetSize(na_);
  Kau_->Mult(u, Ra);
  Kaa_->AddMult(a, Ra, 1.0);
  Ra -= Fa_;
  for (int i = 0; i < na_; ++i)
    if (ess_mask_[i]) Ra(i) = 0.0;
}

void MhdCoupling::set_fields(const mfem::Vector& a) {
  for (int i = 0; i < nphi_; ++i) (*phi_gf_)(i) = a(i) * phi_scale_;
  if (diag_.phi_nullspace_detected) {  // mean-free (the integral of phi)
    mfem::LinearForm one(phifes_.get());
    mfem::ConstantCoefficient c1(1.0);
    one.AddDomainIntegrator(new mfem::DomainLFIntegrator(c1));
    one.Assemble();
    const double mean = (one * *phi_gf_) / one.Sum();
    *phi_gf_ -= mean;
  }
}

}  // namespace axf::fluids
