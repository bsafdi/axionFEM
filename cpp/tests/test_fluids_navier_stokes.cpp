// Steady Navier–Stokes (cpp/include/axionfem/fluids/{convection,navier_stokes}.hpp): the C++ unit tests (the
// Python ones are tests/test_fluids_navier_stokes_api.py):
//   U2  planar standard residual and Newton Jacobian = MFEM's VectorConvectionNLFIntegrator (Q = rho), element by
//       element (an independent implementation);
//   U3  planar skew Oseen matrix O(w) (rho = 1) = MFEM's SkewSymmetricVectorConvectionNLFIntegrator::AssembleElementGrad
//       at w (the antisymmetrized c_A as an operator) + the element boundary matrix 1/2 int_dK (w.n)(du.v), element by
//       element; and the skew residual = O(u) u;
//   U4  axisymmetric kernel = a 3D Cartesian evaluation of rho ((u.grad)u).v for theta-independent fields (e_r, e_theta
//       frame at phi = 0, d/dy = (1/r) d/dphi), random element data, both forms;
//   U8  the refusals of unsupported options (grad-div, MINRES, a pressure point, slip/periodic, swirl-free axisymmetry) and
//       of the transient strategies with the MHD coupling;
//   U9  (when run on more than one rank, e.g. mpirun -n 4) the solution and the history equal those of a one-rank
//       solve of the same problem to round-off;
//   U10 AxiViscousIntegrator honours the given rule (degree 6 differs from the degree-5 fallback on a hoop-term
//       integral; order 0 is still that fallback), and the Stokes reaction's re-assembly site (stokes.cpp, the other
//       form's residual) runs with the solve's rule (a smoke check: finite torques).
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/fluids/axi_integrators.hpp"
#include "axionfem/fluids/convection.hpp"
#include "axionfem/fluids/navier_stokes.hpp"
#include "axionfem/fluids/stokes.hpp"

using namespace axf::fluids;
static int failures = 0;
static bool rank0 = true;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

static double maxabs(const mfem::DenseMatrix& A) {
  double m = 0;
  for (int i = 0; i < A.Height(); ++i)
    for (int j = 0; j < A.Width(); ++j) m = std::max(m, std::fabs(A(i, j)));
  return m;
}

// a slightly distorted mesh on [0.2, 1.2] x [0, 1] (r > 0 for the axisymmetric checks)
static mfem::Mesh test_mesh() {
  mfem::Mesh m = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  for (int v = 0; v < m.GetNV(); ++v) {
    double* x = m.GetVertex(v);
    const double dx = 0.03 * std::sin(7 * x[0] + 3 * x[1]) * x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    x[0] = 0.2 + x[0] + dx;
    x[1] = x[1] - dx;
  }
  return m;
}

// the element boundary matrix 1/2 int_dK (w.n)(du . v), du, v in the element's vector P2 basis (component blocks)
static void boundary_matrix(mfem::Mesh& mesh, int e, const mfem::FiniteElement& fe, const mfem::Vector& welem,
                            mfem::DenseMatrix& Mb) {
  const int nd = fe.GetDof();
  Mb.SetSize(2 * nd);
  Mb = 0.0;
  mfem::Array<int> edges, ori;
  mesh.GetElementEdges(e, edges, ori);
  mfem::Vector N(nd), nor(2);
  for (int f : edges) {
    mfem::FaceElementTransformations* Tr = mesh.GetFaceElementTransformations(f);
    const bool first = Tr->Elem1No == e;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 8);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      fe.CalcShape(first ? Tr->GetElement1IntPoint() : Tr->GetElement2IntPoint(), N);
      mfem::CalcOrtho(Tr->Jacobian(), nor);  // outward from Elem1, length |J|
      if (!first) nor.Neg();
      double w[2] = {0, 0};
      for (int c = 0; c < 2; ++c)
        for (int a = 0; a < nd; ++a) w[c] += welem(c * nd + a) * N(a);
      const double wn = w[0] * nor(0) + w[1] * nor(1);
      for (int c = 0; c < 2; ++c)
        for (int a = 0; a < nd; ++a)
          for (int b = 0; b < nd; ++b) Mb(c * nd + a, c * nd + b) += 0.5 * ip.weight * wn * N(a) * N(b);
    }
  }
}

static void u2_u3() {
  mfem::Mesh mesh = test_mesh();
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&mesh, &fec, 2);
  std::mt19937 gen(11);
  std::normal_distribution<double> nd01;
  const double rho = 1.37;
  mfem::ConstantCoefficient Q(rho), one(1.0);
  ConvectionIntegrator std_k(rho, 2, false, ConvectiveForm::Standard);
  ConvectionIntegrator skew_k(1.0, 2, false, ConvectiveForm::Skew);
  mfem::VectorConvectionNLFIntegrator mf(Q);
  mfem::SkewSymmetricVectorConvectionNLFIntegrator mfs(one);
  mf.SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 5));
  mfs.SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 5));
  double d_res = 0, d_jac = 0, d_skew = 0, d_skres = 0, scale = 0;
  for (int e = 0; e < mesh.GetNE(); ++e) {
    const mfem::FiniteElement& fe = *fes.GetFE(e);
    mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
    mfem::Vector u(2 * fe.GetDof());
    for (int i = 0; i < u.Size(); ++i) u(i) = nd01(gen);
    mfem::Vector r1, r2;
    mfem::DenseMatrix J1, J2;
    std_k.AssembleElementVector(fe, T, u, r1);
    mf.AssembleElementVector(fe, T, u, r2);
    std_k.set_grad_mode(ConvectionIntegrator::GradMode::Newton);
    std_k.AssembleElementGrad(fe, T, u, J1);
    mf.AssembleElementGrad(fe, T, u, J2);
    r1 -= r2;
    d_res = std::max(d_res, r1.Normlinf() / r2.Normlinf());
    scale = maxabs(J2);
    J1 -= J2;
    d_jac = std::max(d_jac, maxabs(J1) / scale);
    // U3: O_skew(w) = MFEM skew grad at w + 1/2 int_dK (w.n)(du.v)
    mfem::DenseMatrix O, A, Mb;
    skew_k.set_grad_mode(ConvectionIntegrator::GradMode::Oseen);
    skew_k.AssembleElementGrad(fe, T, u, O);
    mfs.AssembleElementGrad(fe, T, u, A);
    boundary_matrix(mesh, e, fe, u, Mb);
    A += Mb;
    const double s = maxabs(O);
    O -= A;
    d_skew = std::max(d_skew, maxabs(O) / s);
    // the skew residual is O(u) u
    mfem::Vector rs, Ou(u.Size());
    skew_k.AssembleElementVector(fe, T, u, rs);
    skew_k.AssembleElementGrad(fe, T, u, O);
    O.Mult(u, Ou);
    Ou -= rs;
    d_skres = std::max(d_skres, Ou.Normlinf() / rs.Normlinf());
  }
  CHECK(d_res < 1e-13, "U2 residual vs MFEM VectorConvectionNLFIntegrator: rel %.3e", d_res);
  CHECK(d_jac < 1e-13, "U2 Newton Jacobian vs MFEM: rel %.3e", d_jac);
  CHECK(d_skew < 1e-12, "U3 skew Oseen vs MFEM skew grad + boundary matrix: rel %.3e", d_skew);
  CHECK(d_skres < 1e-13, "U3 skew residual vs O(u) u: rel %.3e", d_skres);
  if (rank0) std::printf("U2/U3: res %.2e jac %.2e skew %.2e skew-res %.2e\n", d_res, d_jac, d_skew, d_skres);
}

static void u4() {
  mfem::Mesh mesh = test_mesh();
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&mesh, &fec, 3);
  std::mt19937 gen(5);
  std::normal_distribution<double> nd01;
  const double rho = 0.8;
  double dmax = 0;
  for (int skew = 0; skew < 2; ++skew) {
    ConvectionIntegrator k(rho, 3, true, skew ? ConvectiveForm::Skew : ConvectiveForm::Standard);
    for (int e = 0; e < mesh.GetNE(); ++e) {
      const mfem::FiniteElement& fe = *fes.GetFE(e);
      const int nd = fe.GetDof();
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      mfem::Vector u(3 * nd), r1, r2(3 * nd);
      for (int i = 0; i < u.Size(); ++i) u(i) = nd01(gen);
      k.AssembleElementVector(fe, T, u, r1);
      r2 = 0.0;
      mfem::Vector N(nd);
      mfem::DenseMatrix dN(nd, 2);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe.GetGeomType(), 6);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        fe.CalcShape(ip, N);
        fe.CalcPhysDShape(T, dN);
        double x[3];
        mfem::Vector X(x, 2);
        T.Transform(ip, X);
        const double r = x[0];
        // cylindrical components (r, theta, z) from the internal (r, z, theta)
        const int map[3] = {0, 2, 1};
        double c[3] = {0, 0, 0}, dr[3] = {0, 0, 0}, dz[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i)
          for (int a = 0; a < nd; ++a) {
            const double v = u(map[i] * nd + a);
            c[i] += v * N(a);
            dr[i] += v * dN(a, 0);
            dz[i] += v * dN(a, 1);
          }
        // Cartesian at phi = 0: U = (u_r, u_theta, u_z); d_x U = d_r, d_y U = (1/r) d_phi U = (1/r)(-u_t, u_r, 0)
        const double U[3] = {c[0], c[1], c[2]};
        const double dX[3] = {dr[0], dr[1], dr[2]}, dY[3] = {-c[1] / r, c[0] / r, 0.0}, dZ[3] = {dz[0], dz[1], dz[2]};
        double adv[3];
        for (int i = 0; i < 3; ++i) adv[i] = U[0] * dX[i] + U[1] * dY[i] + U[2] * dZ[i];
        const double divU = dX[0] + dY[1] + dZ[2];
        const double W = ip.weight * T.Weight() * r;
        for (int i = 0; i < 3; ++i)
          for (int a = 0; a < nd; ++a)
            r2(map[i] * nd + a) += W * rho * (adv[i] + (skew ? 0.5 * divU * U[i] : 0.0)) * N(a);
      }
      r1 -= r2;
      dmax = std::max(dmax, r1.Normlinf() / r2.Normlinf());
    }
  }
  CHECK(dmax < 1e-13, "U4 axisymmetric kernel vs the 3D Cartesian evaluation: rel %.3e", dmax);
  if (rank0) std::printf("U4: %.2e\n", dmax);
}

// lid-driven cavity on the unit square (attributes: 1 bottom, 2 right, 3 top, 4 left)
struct Cavity {
  mfem::ConstantCoefficient mu{0.01};
  mfem::VectorFunctionCoefficient lid{2, [](const mfem::Vector& x, mfem::Vector& v) {
                                        v.SetSize(2);
                                        v(0) = 16 * x(0) * x(0) * (1 - x(0)) * (1 - x(0));
                                        v(1) = 0;
                                      }};
  StokesBoundary bc() {
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &lid).dirichlet(4);
    return b;
  }
};

static bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const axf::Error&) {
    return true;
  }
  return false;
}

static void u8_refusals() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh pm(MPI_COMM_WORLD, sm);
  Cavity C;
  auto opts = [&]() {
    NavierStokesOptions o;
    o.stokes.nu = &C.mu;
    o.stokes.lu = LUBackend::EigenLU;
    return o;
  };
  {
    auto o = opts();
    o.stokes.grad_div = 1.0;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "grad-div must be refused");
  }
  {
    auto o = opts();
    o.stokes.solver = LinearSolverKind::MinresBlockDiag;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "MINRES must be refused");
  }
  {
    auto o = opts();
    o.stokes.pressure_point.active = true;
    o.stokes.pressure_point.x0 = {0.5, 0.5};
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "a pressure point must be refused");
  }
  {
    auto o = opts();
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3, &C.lid).slip(4);
    CHECK(throws([&] { NavierStokesSolver s(pm, o, b); }), "slip must be refused");
  }
  {
    auto o = opts();
    o.stokes.geometry = Geometry::Axisymmetric;
    o.stokes.swirl = false;
    StokesBoundary b;
    b.dirichlet(1).dirichlet(2).dirichlet(3).axis(4);
    CHECK(throws([&] { NavierStokesSolver s(pm, o, b); }), "swirl-free axisymmetry must be refused");
  }
  {
    auto o = opts();
    o.rho = 0.0;
    CHECK(throws([&] { NavierStokesSolver s(pm, o, C.bc()); }), "rho = 0 must be refused");
  }
  // the transient strategies (time steps, transient_ns.hpp) carry no potential unknowns: refused with the MHD coupling
  // at construction. The message is asserted, not only the throw: the two-component boundary data of this problem
  // would be refused later anyway, by the Planar25D Stokes assembly.
  for (NonlinearStrategy st : {NonlinearStrategy::TransientNewton, NonlinearStrategy::TransientSemiImplicit}) {
    auto o = opts();
    MhdOptions m;
    o.stokes.geometry = Geometry::Planar25D;
    o.stokes.form = ViscousForm::Laplacian;
    o.mhd = &m;
    o.nonlinear.strategy = st;
    std::string what;
    try {
      NavierStokesSolver s(pm, o, C.bc());
    } catch (const axf::Error& e) {
      what = e.message();
    }
    CHECK(what.find("transient time stepping is not offered") != std::string::npos,
          "a transient strategy with MHD must be refused (got '%s')", what.c_str());
  }
}

// U9: the same problem on all ranks of MPI_COMM_WORLD and on one rank (rank 0, MPI_COMM_SELF)
static void u9() {
  int size = 1, me = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  Cavity C;
  C.mu.constant = 1.0 / 300;
  auto run = [&](MPI_Comm comm, std::vector<double>& hist, double& ekin) {
    mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(12, 12, mfem::Element::TRIANGLE, true, 1.0, 1.0);
    mfem::ParMesh pm(comm, sm);
    NavierStokesOptions o;
    o.stokes.nu = &C.mu;
    o.stokes.lu = lu_backend_available(LUBackend::StrumpackMPI) && comm != MPI_COMM_SELF ? LUBackend::StrumpackMPI
                                                                                         : LUBackend::EigenLU;
    o.stokes.nullspace = Nullspace::MeanMultiplier;
    if (o.stokes.lu == LUBackend::EigenLU) o.stokes.nullspace = Nullspace::Auto;
    NavierStokesSolver s(pm, o, C.bc());
    const auto& d = s.solve();
    CHECK(d.converged, "U9 solve did not converge (%s)", d.failure.c_str());
    hist.clear();
    for (const auto& h : d.history) hist.push_back(h.residual);
    mfem::Vector z(2);
    z = 0.0;
    mfem::VectorConstantCoefficient zc(z);
    const double l2 = s.velocity().ComputeL2Error(zc);
    ekin = 0.5 * l2 * l2;
  };
  std::vector<double> hw, hs;
  double ew = 0, es = 0;
  run(MPI_COMM_WORLD, hw, ew);
  if (size == 1) {
    if (rank0) std::printf("U9: one rank (run under mpirun -n 4 for the rank-invariance check); Ekin %.15e\n", ew);
    return;
  }
  if (me == 0) {
    run(MPI_COMM_SELF, hs, es);
    CHECK(std::fabs(ew - es) <= 1e-12 * es, "U9 Ekin %d ranks %.16e vs 1 rank %.16e", size, ew, es);
    CHECK(hw.size() == hs.size(), "U9 history length %zu vs %zu", hw.size(), hs.size());
    for (size_t k = 0; k + 1 < std::min(hw.size(), hs.size()); ++k)  // the last one is at round-off
      CHECK(std::fabs(hw[k] - hs[k]) <= 1e-8 * hs[k] + 1e-14, "U9 history[%zu] %.6e vs %.6e", k, hw[k], hs[k]);
    std::printf("U9: %d ranks vs 1: Ekin rel %.2e, history lengths %zu/%zu\n", size, std::fabs(ew - es) / es,
                hw.size(), hs.size());
  }
}

// U10: AxiViscousIntegrator honours the rule; the Stokes reaction passes the solve's rule (degree 6)
static void u10() {
  mfem::Mesh sm = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  for (int v = 0; v < sm.GetNV(); ++v) sm.GetVertex(v)[0] += 0.05;  // r in [0.05, 1.05]: a strong 1/r variation
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace fes(&sm, &fec, 3);
  mfem::ConstantCoefficient nu(1.0);
  AxiViscousIntegrator i5(nu, 3, false, 0.0, 5), i6(nu, 3, false, 0.0, 6), i0(nu, 3, false, 0.0, 0);
  mfem::DenseMatrix m5, m6, m0;
  i5.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m5);
  i6.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m6);
  i0.AssembleElementMatrix(*fes.GetFE(0), *sm.GetElementTransformation(0), m0);
  mfem::DenseMatrix d(m6);
  d -= m5;
  CHECK(maxabs(d) > 1e-10 * maxabs(m6),
        "U10: degree 5 and 6 rules agree (%.3e): the test integral cannot tell them apart", maxabs(d));
  d = m0;
  d -= m5;
  CHECK(maxabs(d) == 0.0, "U10: the order-0 fallback is no longer the degree-5 rule");
  // the reaction site: the Stokes reaction on a Couette-like axisymmetric problem equals the solved-form residual
  // re-assembled on degree 6 (it differs from degree 5)
  mfem::Mesh am = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 0.5, 0.5);
  for (int v = 0; v < am.GetNV(); ++v) am.GetVertex(v)[0] += 0.5;  // r in [0.5, 1], z in [0, 0.5]
  mfem::ParMesh pm(MPI_COMM_WORLD, am);
  mfem::ConstantCoefficient mu(0.1);
  mfem::VectorFunctionCoefficient wall(3, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(3);
    v = 0.0;
    v(2) = x(0) < 0.5 + 1e-12 ? 2.0 * x(0) : 0.0;
  });
  StokesOptions o;
  o.nu = &mu;
  o.geometry = Geometry::Axisymmetric;
  o.lu = LUBackend::EigenLU;
  o.form = ViscousForm::Laplacian;
  StokesBoundary b;
  b.dirichlet(1).dirichlet(2).dirichlet(3).dirichlet(4, &wall);
  StokesSolver S(pm, o, b);
  S.solve();
  const Reaction rs = S.reaction({4}, {}, ViscousForm::SymmetricStress);  // re-assembled: the fixed site
  o.form = ViscousForm::SymmetricStress;
  StokesSolver S2(pm, o, b);
  S2.solve();
  const Reaction r2 = S2.reaction({4}, {}, ViscousForm::SymmetricStress);  // the solved operator (degree 6)
  CHECK(std::isfinite(rs.torque) && std::isfinite(r2.torque), "U10: reactions not finite");
  if (rank0) std::printf("U10: torque (lap solve, S residual) %.10e, (S solve) %.10e\n", rs.torque, r2.torque);
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  rank0 = mfem::Mpi::WorldRank() == 0;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    u2_u3();
    u4();
    u10();
  }
  u8_refusals();
  u9();
  if (rank0) std::printf(failures ? "test_fluids_navier_stokes: %d FAILURES\n" : "test_fluids_navier_stokes: OK\n",
                         failures);
  return failures ? 1 : 0;
}
