// Steady Boussinesq natural convection: see cpp/include/axionfem/fluids/boussinesq.hpp.
#include "axionfem/fluids/boussinesq.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"
#include "navier_stokes_internal.hpp"

namespace axf::fluids {

namespace {
void require_one_rank_b(MPI_Comm c, const char* what) {
  AXF_REQUIRE(comm_size(c) == 1, strcat(what, "() needs one MPI rank (the Boussinesq solver runs on one rank)"));
}
}  // namespace

struct BoussinesqSolver::Impl {
  NavierStokesSolver::Impl* I = nullptr;
  BoussinesqSolver* B = nullptr;
  std::unique_ptr<mfem::SparseMatrix> MuT;  // int phi_u^{up} N_T r : (nu_loc x nT)
  double c0 = 0.0;               // rho0 beta g
  double Tref = 0.0;
  int nT = 0;
  const std::vector<char>* essT = nullptr;
  mfem::Vector lift_T;
  std::unique_ptr<mfem::ParGridFunction> ug, pg;
  std::unique_ptr<mfem::GridFunction> Tg;
  mfem::Vector u, p, T;          // the current state
  bool solved = false;
  int nu() const { return I->nu_loc; }
  int np() const { return I->np_loc; }

  void split(const mfem::Vector& x, mfem::Vector& uu, mfem::Vector& pp, mfem::Vector& TT) const {
    uu.SetSize(nu());
    pp.SetSize(np());
    TT.SetSize(nT);
    for (int i = 0; i < nu(); ++i) uu(i) = x(i);
    for (int i = 0; i < np(); ++i) pp(i) = x(nu() + i);
    for (int i = 0; i < nT; ++i) TT(i) = x(nu() + np() + i);
  }
  void join(const mfem::Vector& uu, const mfem::Vector& pp, const mfem::Vector& TT, mfem::Vector& x) const {
    x.SetSize(nu() + np() + nT);
    for (int i = 0; i < nu(); ++i) x(i) = uu(i);
    for (int i = 0; i < np(); ++i) x(nu() + i) = pp(i);
    for (int i = 0; i < nT; ++i) x(nu() + np() + i) = TT(i);
  }

  // buoyancy load vector c0 s M_uT (T - Tref), velocity true dofs
  void buoyancy_load(const mfem::Vector& TT, double s, mfem::Vector& out) const {
    mfem::Vector d(TT);
    d -= Tref;
    out.SetSize(nu());
    out = 0.0;
    MuT->Mult(d, out);
    out *= c0 * s;
  }

  // R_u, R_p, R_T (free rows; essential rows zero)
  void residual(const mfem::Vector& uu, const mfem::Vector& pp, const mfem::Vector& TT, double s, mfem::Vector& Ru,
                mfem::Vector& Rp, mfem::Vector& RT) const {
    I->residual(uu, pp, 1.0, true, Ru, Rp);
    mfem::Vector bl;
    buoyancy_load(TT, s, bl);
    Ru -= bl;
    I->zero_ess(Ru);
    mfem::Vector Rfull;
    B->tr_->residual_full(TT, &uu, Rfull);
    RT.SetSize(nT);
    for (int i = 0; i < nT; ++i) RT(i) = (*essT)[i] ? 0.0 : Rfull(i);
  }
  double dual(const mfem::Vector& Ru, const mfem::Vector& Rp, const mfem::Vector& RT) const {
    const double d = I->dual(Ru, Rp);
    const double t = B->tr_->dual_norm(RT);
    return std::sqrt(d * d + t * t);
  }
};

namespace {

// the monolithic Newton system of the Boussinesq solver for the generic driver
class BousSystem : public NonlinearSystem {
 public:
  BousSystem(BoussinesqSolver::Impl& im, NavierStokesSolver::Impl& I, transport::ScalarTransport& tr,
             const NavierStokesOptions& nso, double rtol)
      : S(im), I(I), tr(tr), nso(nso), rtol(rtol) {}
  BoussinesqSolver::Impl& S;
  NavierStokesSolver::Impl& I;
  transport::ScalarTransport& tr;
  const NavierStokesOptions& nso;
  double rtol;
  double lu_residual_last = -1;

  void residual(const mfem::Vector& x, double s, mfem::Vector& R) override {
    mfem::Vector u, p, T, Ru, Rp, RT;
    S.split(x, u, p, T);
    S.residual(u, p, T, s, Ru, Rp, RT);
    S.join(Ru, Rp, RT, R);
  }
  double dual_norm(const mfem::Vector& R) override {
    mfem::Vector Ru, Rp, RT;
    S.split(R, Ru, Rp, RT);
    return S.dual(Ru, Rp, RT);
  }
  bool solve_linear(const mfem::Vector& x, const mfem::Vector& R, double s, StepMode mode, mfem::Vector& d, int& refine,
                    double& lu_residual, std::string& failure) override {
    mfem::Vector u, p, T, Ru, Rp, RT;
    S.split(x, u, p, T);
    S.split(R, Ru, Rp, RT);
    const int nu = S.nu(), np = S.np(), nT = S.nT;
    // velocity block: viscous + the convective gradient (Newton, or the Oseen matrix in a Picard step)
    auto C = I.conv_grad(u, mode == StepMode::Picard ? ConvectionIntegrator::GradMode::Oseen
                                                     : ConvectionIntegrator::GradMode::Newton);
    auto J = I.velocity_block(1.0, C.get(), 0.0);
    DistCsr D0 = I.mono(*J);
    const int n0 = D0.local_rows;
    // coupling blocks
    mfem::SparseMatrix JTT = tr.jacobian(T, &u);
    std::unique_ptr<mfem::SparseMatrix> JTu(new mfem::SparseMatrix(tr.advection_jacobian_u(T, u)));
    const double cj = S.c0 * s;
    const std::vector<char>& essT = *S.essT;
    DistCsr D;
    D.comm = I.comm;
    D.n_global = n0 + nT;
    D.row_begin = 0;
    D.local_rows = n0 + nT;
    D.rowptr.reserve(n0 + nT + 1);
    D.rowptr.push_back(0);
    const int* MI = S.MuT->GetI();
    const int* MJ = S.MuT->GetJ();
    const double* MD = S.MuT->GetData();
    for (int i = 0; i < n0; ++i) {
      for (int k = D0.rowptr[i]; k < D0.rowptr[i + 1]; ++k) {
        D.col.push_back(D0.col[k]);
        D.val.push_back(D0.val[k]);
      }
      if (i < nu && !I.ess_mask[i])
        for (int k = MI[i]; k < MI[i + 1]; ++k)
          if (!essT[MJ[k]]) {
            D.col.push_back(n0 + MJ[k]);
            D.val.push_back(-cj * MD[k]);
          }
      D.rowptr.push_back(static_cast<int>(D.col.size()));
    }
    const int* TI = JTT.GetI();
    const int* TJ = JTT.GetJ();
    const double* TD = JTT.GetData();
    for (int a = 0; a < nT; ++a) {
      if (essT[a]) {
        D.col.push_back(n0 + a);
        D.val.push_back(1.0);
      } else {
        {
          const int* UI = JTu->GetI();
          const int* UJ = JTu->GetJ();
          const double* UD = JTu->GetData();
          for (int k = UI[a]; k < UI[a + 1]; ++k)
            if (UJ[k] < nu && !I.ess_mask[UJ[k]]) {
              D.col.push_back(UJ[k]);
              D.val.push_back(UD[k]);
            }
        }
        for (int k = TI[a]; k < TI[a + 1]; ++k)
          if (!essT[TJ[k]]) {
            D.col.push_back(n0 + TJ[k]);
            D.val.push_back(TD[k]);
          }
      }
      D.rowptr.push_back(static_cast<int>(D.col.size()));
    }
    D.sum_duplicates();
    std::vector<double> b(D.local_rows, 0.0);
    for (int i = 0; i < nu; ++i) b[i] = -Ru(i);
    for (int i = 0; i < np; ++i) b[nu + i] = -Rp(i);
    for (int i = 0; i < nT; ++i) b[n0 + i] = -RT(i);
    SparseLU lu(nso.stokes.lu);
    lu.set_refinement(10, std::min(1e-12, 1e-2 * rtol));
    lu.set_throw_on_miss(false);
    lu.set_threads(nso.stokes.lu_threads);
    lu.set_refine_to_stagnation(true);
    std::vector<double> z(D.local_rows, 0.0);
    for (int i = nu; i < nu + np; ++i) z[i] = 1.0;
    lu.set_kernel_probe(z, 1e-14);
    try {
      lu.factorize(D);
      const std::vector<double> xs = lu.solve(b);
      d.SetSize(nu + np + nT);
      for (int i = 0; i < nu; ++i) d(i) = I.ess_mask[i] ? 0.0 : xs[i];
      for (int i = 0; i < np; ++i) d(nu + i) = xs[nu + i];
      for (int i = 0; i < nT; ++i) d(nu + np + i) = essT[i] ? 0.0 : xs[n0 + i];
    } catch (const std::exception& e) {
      failure = std::string("linear solve failed: ") + e.what();
      return false;
    }
    refine = lu.stats().refine_steps;
    lu_residual = lu.stats().rel_residual;
    lu_residual_last = lu_residual;
    for (int i = 0; i < d.Size(); ++i)
      if (!std::isfinite(d(i))) {
        failure = "linear solve returned a non-finite step";
        return false;
      }
    return true;
  }
};

}  // namespace

// ================================================================================================ construction
BoussinesqSolver::BoussinesqSolver(mfem::ParMesh& mesh, const BoussinesqOptions& opt, const StokesBoundary& bc)
    : impl_(new Impl), opt_(opt) {
  AXF_REQUIRE(opt.t_ref_set, "Boussinesq: T_ref is required (no default): it fixes the pressure relative to the "
                             "reference hydrostatic state", "T_ref", "Pass T_ref explicitly.");
  AXF_REQUIRE(!opt.transient, "Boussinesq: a transient solve is not offered (steady only; a transient Boussinesq "
                              "solve is not implemented)", "transient",
              "Use the steady solve.");
  AXF_REQUIRE(opt.t_order == 2, "Boussinesq: the temperature is P2 (the coupled solver refuses P1 T)", "t_order",
              "Use t_order = 2; P1 temperature is offered for scalar transport only.");
  AXF_REQUIRE(opt.start == "lift", "Boussinesq: the only start is the Dirichlet lift ('linear' is not offered)", "start");
  AXF_REQUIRE(opt.ns.nonlinear.strategy == NonlinearStrategy::Newton,
              "Boussinesq: segregated, Picard and pseudo-transient strategies are refused; one monolithic Newton solve",
              "strategy", "Use strategy = newton with a buoyancy continuation for high Ra.");
  AXF_REQUIRE(opt.ns.nonlinear.n_picard == 0, "Boussinesq: Picard steps are not offered (n_picard must be 0)",
              "n_picard");
  AXF_REQUIRE(opt.ns.nonlinear.viscosity_scales.size() <= 1 &&
                  (opt.ns.nonlinear.viscosity_scales.empty() || opt.ns.nonlinear.viscosity_scales[0] == 1.0),
              "Boussinesq: the continuation scales beta g (buoyancy_scales), not the viscosity", "viscosity_scales");
  AXF_REQUIRE(opt.ns.stokes.velocity_order == 2, "Boussinesq: Taylor-Hood P2/P1 only", "order");
  AXF_REQUIRE(std::isfinite(opt.cp) && opt.cp > 0, "c_p must be positive", "cp");
  AXF_REQUIRE(std::isfinite(opt.beta_g), "beta g must be finite", "beta_g");
  AXF_REQUIRE(opt.up_sign == 1 || opt.up_sign == -1, "up_sign must be +1 or -1", "gravity");
  for (double s : opt.buoyancy_scales)
    AXF_REQUIRE(std::isfinite(s) && s >= 0, strcat("buoyancy scale ", s, " is not a non-negative number"),
                "buoyancy_scales");
  AXF_REQUIRE(opt.buoyancy_scales.empty() || std::abs(opt.buoyancy_scales.back() - 1.0) <= 1e-12,
              "Boussinesq: a buoyancy schedule must end at 1 (the last stage solves the problem asked for)",
              "buoyancy_scales");
  require_one_rank_b(mesh.GetComm(), "BoussinesqSolver");
  AXF_REQUIRE(opt.ns.mhd == nullptr,
              "Boussinesq: the MHD option is not offered with the coupled temperature solve (the pressure block "
              "would carry the extra potential unknowns)", "mhd");
  NavierStokesOptions nso = opt.ns;
  nso.nonlinear.n_picard = 0;
  ns_.reset(new NavierStokesSolver(mesh, nso, bc));
  NavierStokesSolver::Impl& I = *ns_->impl_;
  Impl& S = *impl_;
  S.I = &I;
  S.B = this;
  const bool axi = nso.stokes.geometry == Geometry::Axisymmetric;

  transport::TransportOptions to;
  to.geometry = axi ? transport::Geometry::Axisymmetric : transport::Geometry::Planar;
  to.order = opt.t_order;
  to.supg = opt.supg;
  to.form = transport::AdvectiveForm::Convective;
  tr_.reset(new transport::ScalarTransport(mesh, to));
  const double a = nso.rho * opt.cp;
  tr_->set_capacity(a);
  tr_->set_diffusivity(opt.k, opt.grad_k);
  tr_->set_source(opt.heat_source);
  transport::Velocity v;
  v.fes = &ns_->velocity_space();
  v.scale = a;
  tr_->set_velocity(v);
  for (const auto& tb : opt.t_bcs) tr_->add_bc(tb.attrs, tb.kind, tb.data);
  tr_->set_lu_backend(nso.stokes.lu);
  tr_->finalize();
  S.nT = tr_->ndof();
  S.essT = &tr_->essential_mask();
  S.lift_T = tr_->lift();

  // buoyancy coupling matrix M_uT = int phi_u^{up} N_T r (rule degree 2p + 4)
  mfem::ParFiniteElementSpace& ufes = ns_->velocity_space();
  int comp = 1;  // (u_x, u_y) planar, (u_r, u_z, u_theta) axisymmetric: the vertical component is index 1
  S.c0 = nso.rho * opt.beta_g * opt.up_sign;
  S.Tref = opt.t_ref;
  {
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, tr_->quad_order());
    S.MuT.reset(new mfem::SparseMatrix(I.nu_loc, S.nT));
    mfem::DenseMatrix em;
    mfem::Array<int> ud, td, rows, cols;
    mfem::Vector shu, sht;
    for (int e = 0; e < mesh.GetNE(); ++e) {
      const mfem::FiniteElement* fu = ufes.GetFE(e);
      const mfem::FiniteElement* ft = tr_->space().GetFE(e);
      ufes.GetElementDofs(e, ud);
      tr_->space().GetElementDofs(e, td);
      em.SetSize(ud.Size(), td.Size());
      em = 0.0;
      mfem::ElementTransformation* Tr = mesh.GetElementTransformation(e);
      shu.SetSize(ud.Size());
      sht.SetSize(td.Size());
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        Tr->SetIntPoint(&ip);
        fu->CalcShape(ip, shu);
        ft->CalcShape(ip, sht);
        double x[3] = {0, 0, 0};
        mfem::Vector xv(x, 2);
        Tr->Transform(ip, xv);
        const double w = ip.weight * Tr->Weight() * (axi ? x[0] : 1.0);
        for (int i = 0; i < ud.Size(); ++i)
          for (int j = 0; j < td.Size(); ++j) em(i, j) += w * shu(i) * sht(j);
      }
      rows.SetSize(ud.Size());
      for (int i = 0; i < ud.Size(); ++i) rows[i] = ufes.DofToVDof(ud[i], comp);
      S.MuT->AddSubMatrix(rows, td, em, 0);
    }
    S.MuT->Finalize(0);
  }
  diag_.tau_rule = opt.supg ? "shakib_hK" : "none";
  diag_.convection = to_string(nso.convection);
  diag_.energy_form = to_string(to.form);
  diag_.quad_orders = {nso.stokes.viscous_quad_order(), I.conv->quad_order(I.order), nso.stokes.load_quad_order(),
                       nso.stokes.boundary_quad_order(), I.qm, tr_->quad_order()};
  diag_.linear = ns_->diag_.linear;
  diag_.n_velocity = I.nu_loc;
  diag_.n_pressure = I.np_loc;
  diag_.n_temperature = S.nT;
}

BoussinesqSolver::~BoussinesqSolver() = default;

mfem::ParFiniteElementSpace& BoussinesqSolver::velocity_space() { return ns_->velocity_space(); }

// ================================================================================================ solve
const BoussinesqDiagnostics& BoussinesqSolver::solve() {
  Impl& S = *impl_;
  NavierStokesSolver::Impl& I = *ns_->impl_;
  const auto& nl = opt_.ns.nonlinear;
  const double t0 = MPI_Wtime();
  diag_.history.clear();
  diag_.converged = false;
  diag_.iterations = 0;
  diag_.line_search_min_step_accepts = 0;
  diag_.failure.clear();
  diag_.linear = ns_->diag_.linear;
  std::vector<double> scales = opt_.buoyancy_scales;
  if (scales.empty()) scales.push_back(1.0);
  diag_.buoyancy_scales = scales;

  BousSystem sys(S, I, *tr_, opt_.ns, nl.newton_rtol);
  NewtonControl ctl;
  ctl.line_search = nl.line_search;
  ctl.armijo_c1 = nl.armijo_c1;
  ctl.line_search_halvings = nl.line_search_halvings;
  ctl.rtol = nl.newton_rtol;
  ctl.maxit = nl.newton_maxit;
  ctl.n_picard = 0;
  NonlinearDriver driver(sys, ctl, diag_.history, diag_.iterations, diag_.line_search_min_step_accepts, diag_.failure);
  driver.scale_name = "buoyancy scale";

  // the Dirichlet lift: (data at the Dirichlet nodes, 0 elsewhere), p = 0
  mfem::Vector pz(I.np_loc);
  pz = 0.0;
  mfem::Vector X, R;
  S.join(I.lift, pz, S.lift_T, X);
  const mfem::Vector X_lift = X;
  auto lift_norm = [&](double s) {
    mfem::Vector Rl;
    sys.residual(X_lift, s, Rl);
    return sys.dual_norm(Rl);
  };
  double r = 1.0, den = lift_norm(scales.front());
  diag_.residual_lift = den;
  bool ok = true;
  if (!std::isfinite(den)) {
    ok = driver.fail("the residual of the lift is not finite");
  } else if (den == 0.0) {
    diag_.relative_residual = 0.0;
    diag_.converged = true;
  } else {
    sys.residual(X, scales.front(), R);
    r = sys.dual_norm(R) / den;
    ok = driver.run_stages(X, R, scales, lift_norm, den, diag_.residual_lift, r);
    diag_.relative_residual = r;
    diag_.converged = ok && std::isfinite(r) && r <= ctl.rtol;
    if (!diag_.converged && diag_.failure.empty()) diag_.failure = "not converged";
  }
  diag_.lu_residual = sys.lu_residual_last;
  diag_.seconds = MPI_Wtime() - t0;
  S.split(X, S.u, S.p, S.T);
  // fields: velocity as solved; pressure mean-free (r-weighted) for an enclosed flow, raw otherwise; T
  S.ug.reset(new mfem::ParGridFunction(&ns_->velocity_space()));
  S.pg.reset(new mfem::ParGridFunction(&I.S->pressure_space()));
  S.ug->SetFromTrueDofs(S.u);
  mfem::Vector pn(S.p);
  if (I.nullspace) {
    double loc[2] = {I.mean_weights * pn, I.mean_weights.Sum()}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, I.comm);
    pn -= glob[0] / glob[1];
  }
  S.pg->SetFromTrueDofs(pn);
  S.Tg.reset(new mfem::GridFunction(&tr_->space()));
  *S.Tg = S.T;
  S.solved = true;
  if (diag_.converged) diag_.failure.clear();
  return diag_;
}

const mfem::ParGridFunction& BoussinesqSolver::velocity() const {
  AXF_REQUIRE(impl_->solved && diag_.converged,
              "the Boussinesq solve did not converge; refusing to return its velocity", "",
              strcat("final steady relative residual ", diag_.relative_residual, " (", diag_.failure, ")"));
  return *impl_->ug;
}
const mfem::ParGridFunction& BoussinesqSolver::pressure() const {
  AXF_REQUIRE(impl_->solved && diag_.converged,
              "the Boussinesq solve did not converge; refusing to return its pressure", "",
              strcat("final steady relative residual ", diag_.relative_residual, " (", diag_.failure, ")"));
  return *impl_->pg;
}
const mfem::GridFunction& BoussinesqSolver::temperature() const {
  AXF_REQUIRE(impl_->solved && diag_.converged,
              "the Boussinesq solve did not converge; refusing to return its temperature", "",
              strcat("final steady relative residual ", diag_.relative_residual, " (", diag_.failure, ")"));
  return *impl_->Tg;
}

// ================================================================================================ free-DOF API
std::array<long, 3> BoussinesqSolver::layout() const {
  const Impl& S = *impl_;
  const NavierStokesSolver::Impl& I = *ns_->impl_;
  long nfu = 0, nfT = 0;
  for (int i = 0; i < I.nu_loc; ++i) nfu += I.ess_mask[i] ? 0 : 1;
  for (int i = 0; i < S.nT; ++i) nfT += (*S.essT)[i] ? 0 : 1;
  return {nfu, I.np_loc - (I.pin_local >= 0 ? 1 : 0), nfT};
}

std::vector<double> BoussinesqSolver::state() const {
  const Impl& S = *impl_;
  const NavierStokesSolver::Impl& I = *ns_->impl_;
  require_one_rank_b(I.comm, "state");
  std::vector<double> x;
  for (int i = 0; i < I.nu_loc; ++i)
    if (!I.ess_mask[i]) x.push_back(S.u(i));
  for (int i = 0; i < I.np_loc; ++i)
    if (i != I.pin_local) x.push_back(S.p(i));
  for (int i = 0; i < S.nT; ++i)
    if (!(*S.essT)[i]) x.push_back(S.T(i));
  return x;
}

namespace {
void scatter3(const std::vector<double>& x, const NavierStokesSolver::Impl& I, const mfem::Vector& liftT,
              const std::vector<char>& essT, int nT, mfem::Vector& u, mfem::Vector& p, mfem::Vector& T,
              bool zero_lift) {
  u.SetSize(I.nu_loc);
  p.SetSize(I.np_loc);
  T.SetSize(nT);
  size_t k = 0;
  auto need = [&] { AXF_REQUIRE(k < x.size(), "state vector too short for the layout", "x_free"); };
  for (int i = 0; i < I.nu_loc; ++i) {
    if (I.ess_mask[i]) {
      u(i) = zero_lift ? 0.0 : I.lift(i);
    } else {
      need();
      u(i) = x[k++];
    }
  }
  for (int i = 0; i < I.np_loc; ++i) {
    if (i == I.pin_local) {
      p(i) = 0.0;
    } else {
      need();
      p(i) = x[k++];
    }
  }
  for (int i = 0; i < nT; ++i) {
    if (essT[i]) {
      T(i) = zero_lift ? 0.0 : liftT(i);
    } else {
      need();
      T(i) = x[k++];
    }
  }
  AXF_REQUIRE(k == x.size(), strcat("state vector has ", x.size(), " entries; the layout has ", k), "x_free");
}
std::vector<double> gather3(const mfem::Vector& Ru, const mfem::Vector& Rp, const mfem::Vector& RT,
                            const NavierStokesSolver::Impl& I, const std::vector<char>& essT) {
  std::vector<double> r;
  for (int i = 0; i < Ru.Size(); ++i)
    if (!I.ess_mask[i]) r.push_back(Ru(i));
  for (int i = 0; i < Rp.Size(); ++i)
    if (i != I.pin_local) r.push_back(Rp(i));
  for (int i = 0; i < RT.Size(); ++i)
    if (!essT[i]) r.push_back(RT(i));
  return r;
}
}  // namespace

std::vector<double> BoussinesqSolver::residual(const std::vector<double>& x, double s) const {
  const Impl& S = *impl_;
  const NavierStokesSolver::Impl& I = *ns_->impl_;
  require_one_rank_b(I.comm, "residual");
  mfem::Vector u, p, T, Ru, Rp, RT;
  scatter3(x, I, S.lift_T, *S.essT, S.nT, u, p, T, false);
  S.residual(u, p, T, s, Ru, Rp, RT);
  return gather3(Ru, Rp, RT, I, *S.essT);
}

std::vector<double> BoussinesqSolver::jacobian_action(const std::vector<double>& x, const std::vector<double>& d,
                                                      double s) const {
  const Impl& S = *impl_;
  NavierStokesSolver::Impl& I = *ns_->impl_;
  require_one_rank_b(I.comm, "jacobian_action");
  mfem::Vector u, p, T, du, dp, dT;
  scatter3(x, I, S.lift_T, *S.essT, S.nT, u, p, T, false);
  scatter3(d, I, S.lift_T, *S.essT, S.nT, du, dp, dT, true);
  auto C = I.conv_grad(u, ConvectionIntegrator::GradMode::Newton);
  mfem::Vector Ju(I.nu_loc), Jp(I.np_loc), t(I.nu_loc);
  I.A0->Mult(du, Ju);
  C->Mult(du, t);
  Ju += t;
  I.Bt->Mult(dp, t);
  Ju += t;
  {
    mfem::Vector tb(I.nu_loc);
    S.MuT->Mult(dT, tb);
    tb *= -S.c0 * s;
    Ju += tb;
  }
  I.zero_ess(Ju);
  I.B->Mult(du, Jp);
  if (I.pin_local >= 0) Jp(static_cast<int>(I.pin_local)) = 0.0;
  // temperature rows: J_Tu du + J_TT dT
  mfem::SparseMatrix JTT = tr_->jacobian(T, &u);
  mfem::Vector JT(S.nT);
  JTT.Mult(dT, JT);
  {
    mfem::SparseMatrix JTu = tr_->advection_jacobian_u(T, u);
    mfem::Vector t2(S.nT);
    JTu.Mult(du, t2);
    JT += t2;
  }
  for (int i = 0; i < S.nT; ++i)
    if ((*S.essT)[i]) JT(i) = 0.0;
  return gather3(Ju, Jp, JT, I, *S.essT);
}

double BoussinesqSolver::dual_norm(const std::vector<double>& R) const {
  const Impl& S = *impl_;
  const NavierStokesSolver::Impl& I = *ns_->impl_;
  require_one_rank_b(I.comm, "dual_norm");
  mfem::Vector Ru, Rp, RT;
  scatter3(R, I, S.lift_T, *S.essT, S.nT, Ru, Rp, RT, true);
  return S.dual(Ru, Rp, RT);
}

std::vector<double> BoussinesqSolver::full_residual_T() const {
  const Impl& S = *impl_;
  AXF_REQUIRE(S.solved && diag_.converged, "full_residual_T(): needs a converged solve");
  mfem::Vector R;
  tr_->residual_full(S.T, &S.u, R);
  return std::vector<double>(R.GetData(), R.GetData() + R.Size());
}

}  // namespace axf::fluids
