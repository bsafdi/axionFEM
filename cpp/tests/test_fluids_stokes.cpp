// fluids Taylor–Hood Stokes (cpp/include/axionfem/fluids/stokes.hpp, mesh_checks.hpp). Fast smoke tests:
//   1. inf-sup mesh check: MFEM's structured triangle mesh has exactly two corner triangles with all vertices on
//      the boundary; quads are not checked; a one-cell tet cube violates Boffi's condition everywhere.
//   2. MMS (scripts/fluids/gen_mms_stokes2d.py) on two levels, both viscous forms: errors drop at roughly the
//      Taylor–Hood rates (loose bounds: this is a smoke test, not the verification case).
//   3. direct (pin and multiplier) and MINRES agree on the same problem to solver tolerance.
//   4. lints: pinning with an open boundary, None with an enclosed flow -> errors; the Laplacian form with variable nu
//      is accepted (it is the PDE -div(nu grad u) + grad p = f).
//   5. the converged flag is hard: a MINRES capped at 2 iterations refuses to hand out its fields.
//   6. axisymmetric operator with swirl assembles to a symmetric matrix and solves (the formulation checks are in the Python tests).
//   7. Dirichlet data with u.n != 0 pointwise: every path accepts, reports the flux defect and
//      agrees to round-off (the continuity rhs is projected before every solve).
//   8. explicit BCs: missing / unknown / repeated attributes, unimplemented kinds, a mis-declared axis are
//      refused; a matrix with the pressure nullspace left in is refused by every LU backend (kernel probe).
//   9. open-outlet matrices: the LU's refinement reaches 1e-12 on every backend.
//  10. scale invariance to round-off: nu -> 1e-6 nu (u same, p/nu same) and L -> 1e3 L with nu -> 1e3 nu,
//      direct and MINRES.
//  11. Babuska–Miller torque on the rotating inner cylinder of a circular Couette flow: converges to the
//      exact torque; the Laplacian-form residual (negative control) is 37.5 % low.
//  12. pressure point: p_h(x0) = value raw; a non-node x0 and an open boundary are refused.
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdio>

#include <mfem.hpp>

#include "data/independent_mms_generated.hpp"
#include "data/mms_stokes2d_generated.hpp"
#include "axionfem/core.hpp"
#include "axionfem/fluids/mesh_checks.hpp"
#include "axionfem/fluids/meshes.hpp"
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

static double l2_err_u(const mfem::ParGridFunction& u) {
  mfem::VectorFunctionCoefficient ue(2, [](const mfem::Vector& x, mfem::Vector& v) {
    v.SetSize(2);
    mms2d::velocity(x(0), x(1), v.GetData());
  });
  return u.ComputeL2Error(ue);
}

static double l2_diff(const mfem::ParGridFunction& a, const mfem::ParGridFunction& b) {
  mfem::ParGridFunction d(a);
  d -= b;
  mfem::Vector z(a.VectorDim());
  z = 0.0;
  mfem::VectorConstantCoefficient zc(z);
  mfem::ConstantCoefficient zs(0.0);
  return a.VectorDim() > 1 ? d.ComputeL2Error(zc) : d.ComputeL2Error(zs);
}

struct Mms {
  mfem::ConstantCoefficient nu1{1.0};
  mfem::FunctionCoefficient nuv{[](const mfem::Vector& x) { return mms2d::nu_variable(x(0), x(1)); }};
  mfem::VectorFunctionCoefficient flap{2, [](const mfem::Vector& x, mfem::Vector& f) {
                                         f.SetSize(2);
                                         mms2d::force_laplacian(x(0), x(1), f.GetData());
                                       }};
  mfem::VectorFunctionCoefficient fstr{2, [](const mfem::Vector& x, mfem::Vector& f) {
                                         f.SetSize(2);
                                         mms2d::force_stress(x(0), x(1), f.GetData());
                                       }};
  mfem::VectorFunctionCoefficient ubc{2, [](const mfem::Vector& x, mfem::Vector& u) {
                                        u.SetSize(2);
                                        mms2d::velocity(x(0), x(1), u.GetData());
                                      }};
  StokesOptions opts(ViscousForm f, LinearSolverKind s, Nullspace ns = Nullspace::Auto) {
    StokesOptions o;
    o.form = f;
    o.nu = f == ViscousForm::Laplacian ? static_cast<mfem::Coefficient*>(&nu1) : &nuv;
    o.force = f == ViscousForm::Laplacian ? &flap : &fstr;
    o.solver = s;
    o.nullspace = ns;
    o.krylov_rtol = 1e-12;
    return o;
  }
  StokesBoundary bc(const mfem::Mesh& m) {
    StokesBoundary b;
    for (int i = 0; i < m.bdr_attributes.Size(); ++i) b.dirichlet(m.bdr_attributes[i], &ubc);
    return b;
  }
};

// unit square, MFEM's diagonals except at the two corners where MakeCartesian2D puts an all-boundary triangle
static std::unique_ptr<mfem::ParMesh> fixed_corner_square(int n) {
  mfem::Mesh m(2, (n + 1) * (n + 1), 2 * n * n, 4 * n, 2);
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i) m.AddVertex(double(i) / n, double(j) / n);
  auto v = [&](int i, int j) { return i + j * (n + 1); };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool flip = (i == n - 1 && j == 0) || (i == 0 && j == n - 1);
      if (!flip) {
        m.AddTriangle(v(i, j), v(i + 1, j), v(i + 1, j + 1), 1);
        m.AddTriangle(v(i + 1, j + 1), v(i, j + 1), v(i, j), 1);
      } else {
        m.AddTriangle(v(i, j), v(i + 1, j), v(i, j + 1), 1);
        m.AddTriangle(v(i + 1, j), v(i + 1, j + 1), v(i, j + 1), 1);
      }
    }
  for (int i = 0; i < n; ++i) m.AddBdrSegment(v(i, 0), v(i + 1, 0), 1);
  for (int j = 0; j < n; ++j) m.AddBdrSegment(v(n, j), v(n, j + 1), 2);
  for (int i = 0; i < n; ++i) m.AddBdrSegment(v(i + 1, n), v(i, n), 3);
  for (int j = 0; j < n; ++j) m.AddBdrSegment(v(0, j + 1), v(0, j), 4);
  m.FinalizeTriMesh(1, 1, true);
  return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, m);
}

static mfem::ParMesh square(int n, bool quad = false) {
  mfem::Mesh m = mfem::Mesh::MakeCartesian2D(n, n, quad ? mfem::Element::QUADRILATERAL : mfem::Element::TRIANGLE,
                                             true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_WORLD, m);
}

int main(int argc, char** argv) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  rank0 = mfem::Mpi::WorldRank() == 0;

  // 1. inf-sup mesh check
  {
    mfem::Mesh t = mfem::Mesh::MakeCartesian2D(6, 6, mfem::Element::TRIANGLE, true, 1.0, 1.0);
    auto r = check_taylor_hood_mesh(t);
    CHECK(r.checked_elements == 72 && r.all_boundary_vertices.size() == 2,
          "tri mesh: %ld checked, %zu violations (expected 72, 2)", r.checked_elements,
          r.all_boundary_vertices.size());
    mfem::Mesh q = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    CHECK(check_taylor_hood_mesh(q).checked_elements == 0, "quads must not be checked");
    mfem::Mesh c = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
    auto rc = check_taylor_hood_mesh(c);
    CHECK(check_taylor_hood_mesh(q).tensor_all_boundary_vertices.size() == 0, "4x4 quads: no corner quad expected");
    mfem::Mesh q1 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    CHECK(check_taylor_hood_mesh(q1).tensor_all_boundary_vertices.size() == 1, "one-quad mesh must be warned");
    CHECK(rc.checked_elements > 0 && rc.all_boundary_vertices.size() == static_cast<size_t>(rc.checked_elements),
          "one-cell tet cube: every tet must be flagged (%zu of %ld)", rc.all_boundary_vertices.size(),
          rc.checked_elements);
  }

  Mms M;
  // 2. MMS, two levels, both forms (direct, Eigen)
  for (ViscousForm f : {ViscousForm::Laplacian, ViscousForm::SymmetricStress}) {
    double e[2];
    for (int l = 0; l < 2; ++l) {
      mfem::ParMesh mesh = square(4 << l);
      StokesSolver S(mesh, M.opts(f, LinearSolverKind::Direct), M.bc(mesh));
      const auto& d = S.solve();
      CHECK(d.converged, "%s MMS level %d did not converge", to_string(f), l);
      e[l] = l2_err_u(S.velocity());
    }
    const double ord = std::log2(e[0] / e[1]);
    if (rank0) std::printf("MMS %-9s L2(u) %.3e -> %.3e  order %.2f\n", to_string(f), e[0], e[1], ord);
    CHECK(ord > 2.4, "%s: L2(u) order %.2f, expected about 3", to_string(f), ord);
  }

  // 3. direct pin / multiplier / MINRES agree
  {
    mfem::ParMesh mesh = square(8);
    StokesSolver Sp(mesh, M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct, Nullspace::PinPressure),
                    M.bc(mesh));
    StokesSolver Sm(mesh, M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct, Nullspace::MeanMultiplier),
                    M.bc(mesh));
    StokesSolver Si(mesh, M.opts(ViscousForm::SymmetricStress, LinearSolverKind::MinresBlockDiag), M.bc(mesh));
    Sp.solve();
    Sm.solve();
    const auto& di = Si.solve();
    CHECK(Sp.diagnostics().converged && Sm.diagnostics().converged && di.converged, "a path did not converge");
    const double du_pm = l2_diff(Sp.velocity(), Sm.velocity()), dp_pm = l2_diff(Sp.pressure(), Sm.pressure());
    const double du_mi = l2_diff(Sm.velocity(), Si.velocity()), dp_mi = l2_diff(Sm.pressure(), Si.pressure());
    if (rank0) std::printf("pin-vs-multiplier du %.1e dp %.1e; direct-vs-minres du %.1e dp %.1e (%d its); lambda %.1e\n", du_pm,
                dp_pm, du_mi, dp_mi, di.iterations, Sm.diagnostics().multiplier);
    CHECK(du_pm < 1e-10 && dp_pm < 1e-10, "pin and multiplier disagree (%g, %g)", du_pm, dp_pm);
    CHECK(du_mi < 1e-8 && dp_mi < 1e-8, "direct and MINRES disagree (%g, %g)", du_mi, dp_mi);
    CHECK(std::fabs(Sm.diagnostics().multiplier) < 1e-10, "multiplier nonzero for compatible data (%g)",
          Sm.diagnostics().multiplier);
  }

  // 4. lints
  {
    mfem::ParMesh mesh = square(4);
    StokesBoundary open;
    open.dirichlet(1, &M.ubc).dirichlet(2, &M.ubc).dirichlet(3, &M.ubc).do_nothing(4);  // one open side
    auto refused_opts = [&](StokesOptions o, const StokesBoundary& b) {
      try {
        StokesSolver S(mesh, o, b);
      } catch (const axf::Error&) {
        return true;
      }
      return false;
    };
    CHECK(refused_opts(M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct, Nullspace::PinPressure), open),
          "pinning with an open boundary was not refused");
    CHECK(!refused_opts(M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct), open),
          "Auto with an open boundary was refused");
    StokesOptions bad = M.opts(ViscousForm::Laplacian, LinearSolverKind::Direct);
    bad.nu = &M.nuv;
    CHECK(!refused_opts(bad, M.bc(mesh)), "Laplacian form with variable viscosity was refused (it is accepted)");
    CHECK(refused_opts(M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct, Nullspace::None), M.bc(mesh)),
          "Nullspace::None with an enclosed flow was not refused");
  }

  // 5. hard converged flag
  {
    mfem::ParMesh mesh = square(8);
    StokesOptions o = M.opts(ViscousForm::SymmetricStress, LinearSolverKind::MinresBlockDiag);
    o.krylov_maxit = 2;
    StokesSolver S(mesh, o, M.bc(mesh));
    const auto& d = S.solve();
    bool threw = false;
    try {
      S.velocity();
    } catch (const axf::Error&) {
      threw = true;
    }
    CHECK(!d.converged && threw, "a stalled MINRES handed out its velocity");
  }

  // 6. axisymmetric with swirl: symmetric operator, solves
  {
    mfem::Mesh m = mfem::Mesh::MakeCartesian2D(4, 8, mfem::Element::TRIANGLE, true, 1.0, 2.0);
    mfem::ParMesh mesh(MPI_COMM_WORLD, m);
    mfem::ConstantCoefficient nu(1.0);
    mfem::VectorFunctionCoefficient lid(3, [](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(3);
      v = 0.0;
      if (x(1) > 2.0 - 1e-12) v(2) = x(0);
    });
    for (ViscousForm f : {ViscousForm::Laplacian, ViscousForm::SymmetricStress}) {
      StokesOptions o;
      o.geometry = Geometry::Axisymmetric;
      o.form = f;
      o.nu = &nu;
      StokesBoundary b;
      b.dirichlet(1, &lid).dirichlet(2, &lid).dirichlet(3, &lid).axis(4);
      StokesSolver S(mesh, o, b);
      DistCsr K = S.monolithic(false, nullptr, true);
      const CsrMatrix G = K.gather(0);
      double asym = 0, nrm = 0;
      auto E = G.to_eigen();
      asym = (E - Eigen::SparseMatrix<double>(E.transpose())).norm();
      nrm = E.norm();
      CHECK(asym <= 1e-12 * nrm, "axisymmetric %s operator not symmetric (%g)", to_string(f), asym / nrm);
      const auto& d = S.solve();
      CHECK(d.converged, "axisymmetric %s solve failed (residual %g)", to_string(f), d.rel_residual);
    }
  }


  auto refused = [](auto&& fn) {
    try {
      fn();
    } catch (const axf::Error&) {
      return true;
    }
    return false;
  };

  // 7. Dirichlet data with u.n != 0 pointwise (net flux 0, discrete flux defect O(h^4)): all paths accept and agree
  {
    // u = curl psi, psi = exp(y) sin(2x + y); Laplacian form, nu = 1, f = grad p is irrelevant: f = 0
    mfem::VectorFunctionCoefficient g(2, [](const mfem::Vector& x, mfem::Vector& u) {
      u.SetSize(2);
      const double e = std::exp(x(1)), s = std::sin(2 * x(0) + x(1)), c = std::cos(2 * x(0) + x(1));
      u(0) = e * (s + c);
      u(1) = -2.0 * e * c;
    });
    mfem::ConstantCoefficient nu(1.0);
    for (int n : {4, 8}) {
      mfem::ParMesh mesh = square(n);
      StokesBoundary b;
      for (int a = 1; a <= 4; ++a) b.dirichlet(a, &g);
      std::vector<std::unique_ptr<StokesSolver>> S;
      std::vector<std::string> names;
      auto add = [&](LinearSolverKind k, LUBackend lu, Nullspace ns, const char* name) {
        StokesOptions o;
        o.form = ViscousForm::Laplacian;
        o.nu = &nu;
        o.solver = k;
        o.lu = lu;
        o.nullspace = ns;
        o.krylov_rtol = 1e-12;
        S.emplace_back(new StokesSolver(mesh, o, b));
        names.push_back(name);
      };
      add(LinearSolverKind::Direct, LUBackend::EigenLU, Nullspace::PinPressure, "eigen/pin");
      add(LinearSolverKind::Direct, LUBackend::EigenLU, Nullspace::MeanMultiplier, "eigen/multiplier");
      if (lu_backend_available(LUBackend::StrumpackLU))
        add(LinearSolverKind::Direct, LUBackend::StrumpackLU, Nullspace::MeanMultiplier, "strumpack/multiplier");
      add(LinearSolverKind::MinresBlockDiag, LUBackend::EigenLU, Nullspace::Ortho, "minres/ortho");
      for (size_t k = 0; k < S.size(); ++k) {
        const auto& d = S[k]->solve();
        CHECK(d.converged, "flux-defect data n=%d: %s refused (residual %g)", n, names[k].c_str(), d.rel_residual);
        CHECK(d.flux_defect > 0 && d.nullspace_detected, "flux defect not reported (%s)", names[k].c_str());
        if (d.solver.rfind("direct", 0) == 0) CHECK(d.rel_residual < 1e-11, "%s residual %g", names[k].c_str(),
                                                    d.rel_residual);
      }
      for (size_t k = 1; k < S.size(); ++k) {
        if (!S[0]->diagnostics().converged || !S[k]->diagnostics().converged) continue;
        const double du = l2_diff(S[0]->velocity(), S[k]->velocity()), dp = l2_diff(S[0]->pressure(),
                                                                                   S[k]->pressure());
        const double tol = names[k].rfind("minres", 0) == 0 ? 1e-8 : 1e-10;
        CHECK(du < tol && dp < tol, "n=%d: %s vs %s differ (du %g, dp %g)", n, names[0].c_str(), names[k].c_str(),
              du, dp);
      }
      if (rank0) std::printf("flux-defect data n=%d: defect %.2e (abs %.2e), all %zu paths agree\n", n,
                             S[0]->diagnostics().flux_defect, S[0]->diagnostics().flux_defect_abs, S.size());
    }
  }

  // 8. explicit boundary conditions and the nullspace
  {
    mfem::ParMesh mesh = square(4);
    StokesOptions o = M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct);
    StokesBoundary missing;
    missing.dirichlet(1, &M.ubc).dirichlet(2, &M.ubc).dirichlet(3, &M.ubc);
    CHECK(refused([&] { StokesSolver S(mesh, o, missing); }), "an unassigned attribute was not refused");
    StokesBoundary unknown = M.bc(mesh);
    unknown.dirichlet(7);
    CHECK(refused([&] { StokesSolver S(mesh, o, unknown); }), "an attribute absent from the mesh was not refused");
    StokesBoundary twice = M.bc(mesh);
    twice.do_nothing(2);
    CHECK(refused([&] { StokesSolver S(mesh, o, twice); }), "a repeated attribute was not refused");
    StokesBoundary per;  // periodic on a mesh that was not made periodic
    per.dirichlet(1, &M.ubc).dirichlet(3, &M.ubc).periodic(4, 2, {1.0, 0.0});
    CHECK(refused([&] { StokesSolver S(mesh, o, per); }), "a periodic pair on a non-periodic mesh was not refused");
    StokesBoundary slip;
    slip.dirichlet(1, &M.ubc).dirichlet(2, &M.ubc).dirichlet(3, &M.ubc).slip(4);
    CHECK(!refused([&] { StokesSolver S(mesh, o, slip); }), "a slip boundary was refused");
    // axisymmetric: the boundary on r = 0 must be declared axis, and only it
    mfem::Mesh m = mfem::Mesh::MakeCartesian2D(4, 8, mfem::Element::TRIANGLE, true, 1.0, 2.0);
    mfem::ParMesh am(MPI_COMM_WORLD, m);
    mfem::ConstantCoefficient nu(1.0);
    StokesOptions ao;
    ao.geometry = Geometry::Axisymmetric;
    ao.nu = &nu;
    StokesBoundary noaxis;
    noaxis.dirichlet(1).dirichlet(2).dirichlet(3).do_nothing(4);
    CHECK(refused([&] { StokesSolver S(am, ao, noaxis); }), "an undeclared axis was not refused");
    StokesBoundary wrongaxis;
    wrongaxis.dirichlet(1).axis(2).dirichlet(3).axis(4);
    CHECK(refused([&] { StokesSolver S(am, ao, wrongaxis); }), "an axis off r = 0 was not refused");
    // the pressure nullspace left in: monolithic(false) refuses; forced through, every LU refuses via the probe
    StokesSolver S(mesh, o, M.bc(mesh));
    CHECK(refused([&] { S.monolithic(false); }), "monolithic(false) with a nullspace was not refused");
    std::vector<double> b;
    DistCsr K = S.monolithic(false, &b, true);
    std::vector<double> z(K.local_rows, 0.0);
    const int nu_loc = S.velocity_space().GetTrueVSize();
    for (int i = nu_loc; i < K.local_rows; ++i) z[i] = 1.0;
    for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
      if (!lu_backend_available(be)) continue;
      SparseLU lu(be);
      lu.set_kernel_probe(z, 1e-14);
      CHECK(refused([&] { lu.factorize(K); }), "%s factorized a matrix with the pressure nullspace left in",
            to_string(be));
    }
  }

  // 9. an open-outlet matrix: do-nothing on x = 1, corner-fixed mesh, n = 16, nu = 1, the
  //    field of cpp/tests/data/independent_mms_generated.hpp. The plain STRUMPACK solve
  //    stopped at 1e-9..1e-8 there; SparseLU's refinement must reach 1e-12 on every backend.
  {
    auto mesh = fixed_corner_square(16);
    mfem::ConstantCoefficient one(1.0);
    mfem::VectorFunctionCoefficient f(2, [](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(2);
      indep::dnL_f(x(0), x(1), v.GetData());
    });
    mfem::VectorFunctionCoefficient ub(2, [](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(2);
      indep::dnL_u(x(0), x(1), v.GetData());
    });
    double worst_plain = 0;
    for (ViscousForm form : {ViscousForm::Laplacian, ViscousForm::SymmetricStress}) {
      StokesOptions o;
      o.form = form;
      o.nu = &one;
      o.force = &f;
      StokesBoundary b;
      b.dirichlet(1, &ub).do_nothing(2).dirichlet(3, &ub).dirichlet(4, &ub);
      StokesSolver S(*mesh, o, b);
      std::vector<double> rhs;
      DistCsr K = S.monolithic(true, &rhs);
      for (LUBackend be : {LUBackend::EigenLU, LUBackend::StrumpackLU, LUBackend::StrumpackMPI}) {
        if (!lu_backend_available(be)) continue;
        SparseLU plain(be), refined(be);
        plain.set_refinement(0, 1.0);
        plain.factorize(K);
        plain.solve(rhs);
        refined.factorize(K);
        refined.solve(rhs);
        if (be != LUBackend::EigenLU) worst_plain = std::max(worst_plain, plain.stats().rel_residual);
        if (rank0) std::printf("run-10 matrix %-9s %-13s: plain %.1e, refined %.1e (%d step(s))\n", to_string(form),
                               to_string(be), plain.stats().rel_residual, refined.stats().rel_residual,
                               refined.stats().refine_steps);
        CHECK(refined.stats().rel_residual <= 1e-12, "%s %s: refined residual %g", to_string(form), to_string(be),
              refined.stats().rel_residual);
      }
    }
    if (lu_backend_available(LUBackend::StrumpackLU) && rank0)
      std::printf("run-10 matrix: worst plain STRUMPACK residual %.1e (refinement %s)\n", worst_plain,
                  worst_plain > 1e-12 ? "needed" : "not needed on this build");
  }

  // 10. scale invariance (to round-off), direct and MINRES: (a) nu -> s nu, (b) L -> s L with nu -> s nu
  {
    for (LinearSolverKind k : {LinearSolverKind::Direct, LinearSolverKind::MinresBlockDiag}) {
      auto run = [&](double nus, double Ls, mfem::Vector& ut, mfem::Vector& pt) {
        mfem::Mesh m = mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, Ls, Ls);
        mfem::ParMesh mesh(MPI_COMM_WORLD, m);
        mfem::ConstantCoefficient nu(nus);
        // u'(x) = u(x/L); p' = (nus/Ls) p(x/L); f' = (nus/Ls^2) f(x/L)
        mfem::VectorFunctionCoefficient f(2, [=](const mfem::Vector& x, mfem::Vector& v) {
          v.SetSize(2);
          mms2d::force_laplacian(x(0) / Ls, x(1) / Ls, v.GetData());
          v *= nus / (Ls * Ls);
        });
        mfem::VectorFunctionCoefficient g(2, [=](const mfem::Vector& x, mfem::Vector& v) {
          v.SetSize(2);
          mms2d::velocity(x(0) / Ls, x(1) / Ls, v.GetData());
        });
        StokesOptions o;
        o.form = ViscousForm::Laplacian;
        o.nu = &nu;
        o.force = &f;
        o.solver = k;
        o.krylov_rtol = 1e-12;
        StokesBoundary b;
        for (int a = 1; a <= 4; ++a) b.dirichlet(a, &g);
        StokesSolver S(mesh, o, b);
        const auto& d = S.solve();
        CHECK(d.converged, "scale run nu=%g L=%g did not converge (%g)", nus, Ls, d.rel_residual);
        S.velocity().GetTrueDofs(ut);
        S.pressure().GetTrueDofs(pt);
        pt *= Ls / nus;  // dimensionless pressure
        return d.iterations;
      };
      mfem::Vector u0, p0, u1, p1, u2, p2;
      const int i0 = run(1.0, 1.0, u0, p0), i1 = run(1e-6, 1.0, u1, p1), i2 = run(1e3, 1e3, u2, p2);
      auto rel = [](const mfem::Vector& a, const mfem::Vector& b) {
        mfem::Vector d(a);
        d -= b;
        return std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, d, d) / mfem::InnerProduct(MPI_COMM_WORLD, a, a));
      };
      const double tol = k == LinearSolverKind::Direct ? 1e-10 : 1e-8;
      const double a_u = rel(u0, u1), a_p = rel(p0, p1), b_u = rel(u0, u2), b_p = rel(p0, p2);
      if (rank0) std::printf("scale %s: nu*1e-6 du %.1e dp %.1e (its %d->%d); L,nu*1e3 du %.1e dp %.1e (its %d)\n",
                             k == LinearSolverKind::Direct ? "direct" : "minres", a_u, a_p, i0, i1, b_u, b_p, i2);
      CHECK(a_u < tol && a_p < tol && b_u < tol && b_p < tol, "scale invariance broken (%g %g %g %g)", a_u, a_p, b_u,
            b_p);
    }
  }

  // 11. Babuska–Miller torque, circular Couette (R1 = 1 rotating at omega = 1, R2 = 2 at rest, nu = 1):
  //     exact torque on the inner cylinder by the fluid -4 pi nu omega R1^2 R2^2 / (R2^2 - R1^2) = -16 pi / 3
  {
    const double Tex = -16.0 * M_PI / 3.0;
    mfem::ConstantCoefficient nu(1.0);
    mfem::VectorFunctionCoefficient rot(2, [](const mfem::Vector& x, mfem::Vector& v) {
      v.SetSize(2);
      v(0) = -x(1);
      v(1) = x(0);
    });
    double prev = 0;
    for (int l = 0; l < 3; ++l) {
      mfem::Mesh m = annulus_mesh(2 << l, 12 << l, 1.0, 2.0, 2);
      mfem::ParMesh mesh(MPI_COMM_WORLD, m);
      StokesOptions o;
      o.form = ViscousForm::SymmetricStress;
      o.nu = &nu;
      o.lu = lu_backend_available(LUBackend::StrumpackLU) ? LUBackend::StrumpackLU : LUBackend::EigenLU;
      StokesBoundary b;
      b.dirichlet(1, &rot).dirichlet(2);
      StokesSolver S(mesh, o, b);
      const auto& d = S.solve();
      CHECK(d.converged, "Couette level %d did not converge", l);
      const Reaction rS = S.reaction({1}), rL = S.reaction({1}, {}, ViscousForm::Laplacian);
      const double eS = std::fabs(rS.torque - Tex) / std::fabs(Tex), eL = (rL.torque - Tex) / Tex;
      if (rank0) std::printf("Couette level %d: torque S %.10f (rel err %.2e%s) L-residual %.6f (%.1f %% off) "
                             "F = (%.1e, %.1e)\n", l, rS.torque, eS,
                             l ? (", order " + std::to_string(std::log2(prev / eS))).c_str() : "", rL.torque,
                             100 * eL, rS.force[0], rS.force[1]);
      if (l == 2) {
        CHECK(eS < 1e-5, "Couette torque error %g", eS);
        CHECK(std::log2(prev / eS) > 3.0, "Couette torque order %g < 3", std::log2(prev / eS));
        CHECK(std::fabs(eL + 0.375) < 0.02, "the L-form residual control should be 37.5 %% low, got %g", eL);
        CHECK(std::hypot(rS.force[0], rS.force[1]) < 1e-8, "net force on the inner cylinder %g",
              std::hypot(rS.force[0], rS.force[1]));
      }
      prev = eS;
    }
  }

  // 12. pressure point: raw value at a pressure node; a non-node and an open boundary are refused
  {
    mfem::ParMesh mesh = square(8);
    StokesOptions o = M.opts(ViscousForm::SymmetricStress, LinearSolverKind::Direct);
    StokesSolver Sm(mesh, o, M.bc(mesh));
    Sm.solve();
    o.pressure_point.active = true;
    o.pressure_point.x0 = {0.25, 0.5};
    o.pressure_point.value = 3.0;
    StokesSolver Sp(mesh, o, M.bc(mesh));
    const auto& d = Sp.solve();
    CHECK(d.converged && std::fabs(d.pressure_point_value - 3.0) < 1e-12, "pressure point value %g",
          d.pressure_point_value);
    mfem::ParGridFunction diff(Sp.pressure());
    diff -= Sm.pressure();  // a constant: p_point = p_meanfree + c
    double c[2] = {diff.Max(), -diff.Min()}, g[2];
    MPI_Allreduce(c, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    CHECK(std::fabs(g[0] + g[1]) < 1e-9, "pressure point result is not the mean-free one plus a constant (%g)",
          g[0] + g[1]);
    StokesOptions bad = o;
    bad.pressure_point.x0 = {0.26, 0.5};
    CHECK(refused([&] { StokesSolver S(mesh, bad, M.bc(mesh)); S.solve(); }), "a non-node pressure point was "
                                                                               "not refused");
    StokesBoundary open;
    open.dirichlet(1, &M.ubc).dirichlet(2, &M.ubc).dirichlet(3, &M.ubc).do_nothing(4);
    CHECK(refused([&] { StokesSolver S(mesh, o, open); }), "a pressure point with an open boundary was not refused");
  }

  if (rank0) std::printf(failures ? "test_fluids_stokes: %d FAILURES\n" : "test_fluids_stokes: OK\n", failures);
  return failures ? 1 : 0;
}
