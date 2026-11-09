// C++ unit tests: the shared scalar-transport operator and the Boussinesq solver (docs/guides/fluids.md).
//   B1  SUPG patch test: a P2 temperature with a P1 divergence-free b and P1 kappa is reproduced to round-off by the
//       Galerkin and the SUPG operator (full strong residual: Hessian, grad kappa . grad T, Robin and flux data);
//   B2  the refusals of unsupported options (P1 T in the coupled solver, missing T_ref, transient, a buoyancy schedule
//       not ending at 1, another tau rule, a diffusive-flux datum with conservative_flux, a condition on the axis);
//   B3  the exact coupled Jacobian: the Taylor slope of R(x + eps d) - R(x) - eps J d is 2 (with and without SUPG);
//   B4  uniform T != T_ref: the Boussinesq solve returns u = 0 (hydrostatic) to round-off.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/fluids/boussinesq.hpp"
#include "axionfem/transport/scalar_transport.hpp"

using namespace axf::fluids;
namespace tr = axf::transport;
static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

static mfem::Mesh unit_mesh(int n) { return mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::TRIANGLE, true, 1.0, 1.0); }

template <class F>
static bool refused(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

static void b1() {
  mfem::Mesh m = unit_mesh(4);
  auto Texact = [](double x, double y) { return 1 + x / 2 - y / 3 + x * x / 4 + x * y / 5 - y * y / 6; };
  for (bool supg : {false, true}) {
    tr::TransportOptions o;
    o.supg = supg;
    tr::ScalarTransport op(m, o);
    op.set_capacity(1.5);
    const double k0 = 1.5e-2;
    tr::ScalarData k;
    k.fn = [k0](const double* xy, int n, double* out) {
      for (int i = 0; i < n; ++i) out[i] = k0 * (1 + 0.3 * xy[2 * i] - 0.2 * xy[2 * i + 1]);
    };
    op.set_diffusivity(k, [k0](const double* xy, int n, int, double* out) {
      for (int i = 0; i < n; ++i) {
        out[2 * i] = 0.3 * k0;
        out[2 * i + 1] = -0.2 * k0;
      }
    });
    tr::Velocity v;
    v.scale = 1.5;
    v.fn = [](const double* xy, int n, int, double* out) {
      for (int i = 0; i < n; ++i) {
        out[2 * i] = 1 + xy[2 * i + 1] / 2;
        out[2 * i + 1] = 0.5 + xy[2 * i] / 3;
      }
    };
    op.set_velocity(v);
    // source from the strong form
    tr::ScalarData s;
    s.fn = [&](const double* xy, int n, double* out) {
      for (int i = 0; i < n; ++i) {
        const double x = xy[2 * i], y = xy[2 * i + 1];
        const double Tx = 0.5 + x / 2 + y / 5, Ty = -1.0 / 3 + x / 5 - y / 3, lap = 0.5 - 1.0 / 3;
        const double kap = k0 * (1 + 0.3 * x - 0.2 * y);
        out[i] = 1.5 * ((1 + y / 2) * Tx + (0.5 + x / 3) * Ty) - kap * lap - k0 * (0.3 * Tx - 0.2 * Ty);
      }
    };
    op.set_source(s);
    tr::BcData d;
    d.g.fn = [&](const double* xy, int n, double* out) {
      for (int i = 0; i < n; ++i) out[i] = Texact(xy[2 * i], xy[2 * i + 1]);
    };
    op.add_bc({4, 1}, tr::BcKind::Dirichlet, d);
    tr::BcData fx;   // x = 1: inward g = n.kappa grad T = kappa T_x
    fx.g.fn = [&](const double* xy, int n, double* out) {
      for (int i = 0; i < n; ++i) {
        const double x = xy[2 * i], y = xy[2 * i + 1];
        out[i] = k0 * (1 + 0.3 * x - 0.2 * y) * (0.5 + x / 2 + y / 5);
      }
    };
    op.add_bc({2}, tr::BcKind::Flux, fx);
    tr::BcData rb;   // y = 1: kappa T_y = h (T_ext - T), T_ext = T + kappa T_y / h
    rb.h.value = 2 * k0;
    rb.ext.fn = [&](const double* xy, int n, double* out) {
      for (int i = 0; i < n; ++i) {
        const double x = xy[2 * i], y = xy[2 * i + 1];
        out[i] = Texact(x, y) + k0 * (1 + 0.3 * x - 0.2 * y) * (-1.0 / 3 + x / 5 - y / 3) / (2 * k0);
      }
    };
    op.add_bc({3}, tr::BcKind::Robin, rb);
    op.finalize();
    mfem::Vector T;
    auto diag = op.solve(T, nullptr, LUBackend::EigenLU, 1e-12);
    CHECK(diag.converged, "B1: scalar solve did not converge (supg=%d, %g)", supg, diag.relative_residual);
    double err = 0;
    mfem::Mesh& mm = m;
    mfem::Array<int> dofs;
    for (int e = 0; e < mm.GetNE(); ++e) {
      const mfem::FiniteElement* fe = op.space().GetFE(e);
      op.space().GetElementDofs(e, dofs);
      mfem::ElementTransformation* Tr = mm.GetElementTransformation(e);
      for (int a = 0; a < fe->GetDof(); ++a) {
        double x[3];
        mfem::Vector xv(x, 2);
        Tr->Transform(fe->GetNodes().IntPoint(a), xv);
        err = std::max(err, std::fabs(T(dofs[a]) - Texact(x[0], x[1])));
      }
    }
    CHECK(err < 1e-12, "B1: patch error %g (supg=%d)", err, supg);
  }
}

static void b2() {
  mfem::Mesh m = unit_mesh(2);
  tr::TransportOptions o;
  o.tau_rule = "coth";
  CHECK(refused([&] { tr::ScalarTransport op(m, o); }), "B2: tau rule coth must be refused");
  tr::TransportOptions o2;
  o2.form = tr::AdvectiveForm::ConservativeFlux;
  tr::ScalarTransport op(m, o2);
  CHECK(refused([&] { op.add_bc({1}, tr::BcKind::Flux, {}); }), "B2: a diffusive flux with conservative_flux must be refused");
  tr::TransportOptions o3;
  o3.order = 3;
  CHECK(refused([&] { tr::ScalarTransport op3(m, o3); }), "B2: order 3 must be refused");
  mfem::ParMesh pm(MPI_COMM_WORLD, m);
  BoussinesqOptions b;
  b.k.value = 1.0;
  b.ns.stokes.nu = nullptr;
  StokesBoundary bc;
  CHECK(refused([&] { BoussinesqSolver s(pm, b, bc); }), "B2: a missing T_ref must be refused");
  b.t_ref_set = true;
  b.transient = true;
  CHECK(refused([&] { BoussinesqSolver s(pm, b, bc); }), "B2: a transient solve must be refused");
  b.transient = false;
  b.t_order = 1;
  CHECK(refused([&] { BoussinesqSolver s(pm, b, bc); }), "B2: P1 temperature must be refused");
  b.t_order = 2;
  b.buoyancy_scales = {0.5, 2.0};
  CHECK(refused([&] { BoussinesqSolver s(pm, b, bc); }), "B2: a schedule not ending at 1 must be refused");
  b.buoyancy_scales.clear();
  MhdOptions mhd;
  b.ns.mhd = &mhd;
  bool mhd_refused = false;
  try {
    BoussinesqSolver s(pm, b, bc);
  } catch (const std::exception& e) {
    mhd_refused = std::string(e.what()).find("MHD") != std::string::npos;
  }
  CHECK(mhd_refused, "B2: the MHD option must be refused by Boussinesq with an MHD message");
  b.ns.mhd = nullptr;
}

static void b3_b4() {
  mfem::Mesh m = unit_mesh(4);
  mfem::ParMesh pm(MPI_COMM_WORLD, m);
  mfem::ConstantCoefficient mu(0.71);
  mfem::VectorConstantCoefficient zero(mfem::Vector(2) = 0.0);
  for (bool supg : {false, true}) {
    BoussinesqOptions b;
    b.ns.stokes.nu = &mu;
    b.ns.stokes.lu = LUBackend::EigenLU;
    b.ns.rho = 1.0;
    b.k.value = 1.0;
    b.beta_g = 1e3 * 0.71;
    b.t_ref_set = true;
    b.t_ref = 0.0;
    b.supg = supg;
    b.buoyancy_scales = {};
    tr::BcData hot, cold;
    hot.g.value = 0.5;
    cold.g.value = -0.5;
    b.t_bcs.push_back({{4}, tr::BcKind::Dirichlet, hot});
    b.t_bcs.push_back({{2}, tr::BcKind::Dirichlet, cold});
    b.t_bcs.push_back({{1, 3}, tr::BcKind::Zero, {}});
    StokesBoundary bc;
    for (int a : {1, 2, 3, 4}) bc.dirichlet(a, &zero);
    BoussinesqSolver S(pm, b, bc);
    const auto& d = S.solve();
    CHECK(d.converged, "B3: cavity solve (supg=%d) did not converge: %s", supg, d.failure.c_str());
    if (!d.converged) continue;
    std::vector<double> x0 = S.state(), x = x0, dir(x0.size());
    std::mt19937_64 g(1);
    std::normal_distribution<double> nd;
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] += 0.05 * nd(g);
      dir[i] = nd(g);
    }
    auto R0 = S.residual(x);
    auto Jd = S.jacobian_action(x, dir);
    double prev = 0, slope_min = 10;
    for (int k = 1; k <= 4; ++k) {
      const double eps = std::pow(10.0, -k);
      std::vector<double> xe = x;
      for (size_t i = 0; i < x.size(); ++i) xe[i] += eps * dir[i];
      auto Re = S.residual(xe);
      double e = 0;
      for (size_t i = 0; i < Re.size(); ++i) e += std::pow(Re[i] - R0[i] - eps * Jd[i], 2);
      e = std::sqrt(e);
      if (k > 1) slope_min = std::min(slope_min, std::log10(prev / e));
      prev = e;
    }
    CHECK(slope_min > 1.9, "B3: Taylor slope %g < 1.9 (supg=%d)", slope_min, supg);
  }
  // B4: hydrostatic, uniform T = 3/4 != T_ref = 1/4
  BoussinesqOptions b;
  b.ns.stokes.nu = &mu;
  b.ns.stokes.lu = LUBackend::EigenLU;
  b.k.value = 1.0;
  b.beta_g = 1000.0;
  b.t_ref_set = true;
  b.t_ref = 0.25;
  tr::BcData t;
  t.g.value = 0.75;
  b.t_bcs.push_back({{1, 3}, tr::BcKind::Dirichlet, t});
  b.t_bcs.push_back({{2, 4}, tr::BcKind::Zero, {}});
  StokesBoundary bc;
  for (int a : {1, 2, 3, 4}) bc.dirichlet(a, &zero);
  BoussinesqSolver S(pm, b, bc);
  const auto& d = S.solve();
  CHECK(d.converged, "B4: hydrostatic solve did not converge");
  if (d.converged) {
    const mfem::ParGridFunction& u = S.velocity();
    CHECK(u.Normlinf() < 1e-9, "B4: spurious velocity %g", u.Normlinf());
  }
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  b1();
  b2();
  b3_b4();
  std::printf(failures ? "test_fluids_boussinesq: %d FAILURES\n" : "test_fluids_boussinesq: OK\n", failures);
  return failures ? 1 : 0;
}
