// Current-distribution solver (cpp/include/axionfem/chem/current_distribution.hpp), unit tests:
//   1. reaction kinetics: di/deta of every kind against a central difference; Tafel sign conventions; the clip.
//   2. primary cell on a rectangle: phi is linear, P1 exact; reaction currents +-kappa V H / L, balance to round-off.
//   3. secondary (BV) cell, potentiostatic and galvanostatic: 1D closed form (bisection) to 1e-10; the galvanostatic
//      phi_s reproduces the potentiostatic current; Newton converges to stagnation.
//   4. Jacobian (bordered row included) against a central difference of the residual; symmetric.
//   5. axisymmetric r weight: uniform current density on z = 0 of a cylinder gives I = i pi R^2.
//   6. half-space DtN: primary disk current within 2 % of 4 kappa r0 V on a coarse graded-free polar mesh.
//   7. P2 isoparametric: the measure of a quarter arc of radius 1 converges at order 4.
//   8. the temperature of the solve is used by residual/jacobian/relres/newton_update/residual_full.
//   9. invalid input is refused with a message (no crash, no silent acceptance).
//  10. warm start (initial = "state"), continuation (set_control) and initial = "primary": a symmetric BV cell
//      converges at 5, 10 and 20 V.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <functional>
#include <vector>

#include "axionfem/chem/current_distribution.hpp"

using namespace axf::chem;
static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

// structured rectangle [0,Lx]x[0,Ly], nx x ny squares split along (i,j)-(i+1,j+1); attrs 1 x=0, 2 x=Lx, 3 y=0, 4 y=Ly
static MeshInput rect(int nx, int ny, double Lx, double Ly, double x0 = 0.0) {
  MeshInput m;
  auto id = [&](int i, int j) { return i * (ny + 1) + j; };
  for (int i = 0; i <= nx; ++i)
    for (int j = 0; j <= ny; ++j) {
      m.vertices.push_back(x0 + Lx * i / nx);
      m.vertices.push_back(Ly * j / ny);
    }
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) {
      int a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      m.triangles.insert(m.triangles.end(), {a, b, c, a, c, d});
      m.element_attr.insert(m.element_attr.end(), {1, 1});
    }
  for (int j = 0; j < ny; ++j) {
    m.boundary.insert(m.boundary.end(), {id(0, j), id(0, j + 1), id(nx, j), id(nx, j + 1)});
    m.boundary_attr.insert(m.boundary_attr.end(), {1, 2});
  }
  for (int i = 0; i < nx; ++i) {
    m.boundary.insert(m.boundary.end(), {id(i, 0), id(i + 1, 0), id(i, ny), id(i + 1, ny)});
    m.boundary_attr.insert(m.boundary_attr.end(), {3, 4});
  }
  return m;
}

static double sum_at(const Eigen::VectorXd& v, const std::vector<int>& d) {
  double s = 0;
  for (int i : d) s += v(i);
  return s;
}

static double kin_current(CurrentDistribution& cd, int attr, double phis, const std::vector<Reaction>& rx, bool axi) {
  auto q = cd.boundary_quadrature({attr}, cd.boundary_degree());
  const double f = kFaraday / (kGasConstant * 298.15);
  double I = 0;
  for (int k = 0; k < q.w.size(); ++k) {
    double i = 0;
    for (auto& r : rx) i += reaction_current(r, phis - q.phi(k) - r.E_eq, f, 200, nullptr, nullptr);
    I += i * q.w(k) * (axi ? 2 * M_PI * q.x(k, 0) : 1.0);
  }
  return I;
}

int main() {
  const double f = kFaraday / (kGasConstant * 298.15);
  // 1 ---------------------------------------------------------------------------------------------- kinetics
  {
    std::vector<Reaction> rs = {{Reaction::Kind::BV, 3.0, 0.1, 0.3, 0.7, 0},
                                {Reaction::Kind::Linear, 2.0, 0.0, 0.4, 0.6, 0},
                                {Reaction::Kind::TafelAnodic, 1.5, 0.0, 0.5, 0.5, 0.06},
                                {Reaction::Kind::TafelCathodic, 1.5, 0.0, 0.5, 0.5, -0.12}};
    for (auto& r : rs)
      for (double eta : {-0.2, -0.01, 0.0, 0.03, 0.15}) {
        double d = 0;
        reaction_current(r, eta, f, 200, &d, nullptr);
        const double h = 1e-6;
        const double fd = (reaction_current(r, eta + h, f, 200, nullptr, nullptr) -
                           reaction_current(r, eta - h, f, 200, nullptr, nullptr)) / (2 * h);
        CHECK(std::abs(d - fd) <= 1e-6 * std::abs(fd) + 1e-12, "di/deta kind %d eta %g: %g vs FD %g", (int)r.kind, eta, d, fd);
        CHECK(d > 0, "di/deta must be positive (monotone kinetics), kind %d", (int)r.kind);
      }
    CHECK(reaction_current(rs[2], 0.06, f, 200, nullptr, nullptr) > 14.99 &&
          reaction_current(rs[2], 0.06, f, 200, nullptr, nullptr) < 15.01, "anodic Tafel: one decade per A");
    CHECK(reaction_current(rs[3], -0.12, f, 200, nullptr, nullptr) < -14.99, "cathodic Tafel negative, a decade per |A|");
    bool clipped = false;
    reaction_current(rs[0], 100.0, f, 200, nullptr, &clipped);
    CHECK(clipped, "exponent clip not reported");
  }
  // 2 ---------------------------------------------------------------------------------------------- primary cell
  for (int order : {1, 2}) {
    const double L = 0.01, H = 0.005, kap = 2.0, V = 0.3;
    CurrentDistribution cd(rect(6, 3, L, H), order, false, 2 * order + 4, 2 * order + 4);
    cd.set_conductivity_by_attribute({{1, kap}});
    ElectrodeSpec a{"anode", {1}, true, 0.05, {}, false, V, 0.0, {}};
    ElectrodeSpec c{"cathode", {2}, true, -0.02, {}, false, 0.0, 0.0, {}};
    cd.add_electrode(a);
    cd.add_electrode(c);
    auto rep = cd.solve(NewtonOptions{});
    CHECK(rep.converged, "primary cell P%d not converged", order);
    const double Iex = kap * ((V - 0.05) - (0.0 + 0.02)) * H / L;  // phi = phi_s - E_eq: 0.25 and 0.02
    const double Ia = sum_at(cd.residual_full(0), cd.electrode_dofs(0));
    const double Ic = sum_at(cd.residual_full(1), cd.electrode_dofs(1));
    CHECK(std::abs(Ia - Iex) < 1e-12 * Iex, "primary P%d anode current %.15g vs %.15g", order, Ia, Iex);
    CHECK(std::abs(Ia + Ic) < 1e-13 * Iex, "primary P%d balance %g", order, Ia + Ic);
    auto dq = cd.domain_quadrature(6);
    double emax = 0;
    for (int k = 0; k < dq.w.size(); ++k)
      emax = std::max(emax, std::abs(dq.phi(k) - ((V - 0.05) + (0.02 - (V - 0.05)) * dq.x(k, 0) / L)));
    CHECK(emax < 1e-13, "primary P%d field not linear: %g", order, emax);
  }
  // 3 ---------------------------------------------------------------------------------------------- BV cell
  {
    const double L = 0.01, H = 0.005, kap = 1.0, V = 0.6;
    Reaction ra{Reaction::Kind::BV, 20.0, 0.05, 0.3, 0.7, 0}, rc{Reaction::Kind::BV, 5.0, -0.02, 0.6, 0.4, 0};
    // closed form: find i with  V = eta_a(i) - eta_c(-i) + i L / kappa  (bisection on i)
    auto eta_of = [&](const Reaction& r, double i) {  // invert BV by bisection
      double lo = -2, hi = 2;
      for (int it = 0; it < 200; ++it) {
        double m = 0.5 * (lo + hi);
        (reaction_current(r, m, f, 200, nullptr, nullptr) < i ? lo : hi) = m;
      }
      return 0.5 * (lo + hi);
    };
    double lo = 0, hi = 100;
    for (int it = 0; it < 200; ++it) {
      double i = 0.5 * (lo + hi);
      double v = eta_of(ra, i) + ra.E_eq - (eta_of(rc, -i) + rc.E_eq) + i * L / kap;
      (v < V ? lo : hi) = i;
    }
    const double iex = 0.5 * (lo + hi);
    CurrentDistribution cd(rect(8, 2, L, H), 1, false, 6, 6);
    cd.set_conductivity_by_attribute({{1, kap}});
    cd.add_electrode({"anode", {1}, false, 0, {ra}, false, V, 0, {}});
    cd.add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
    auto rep = cd.solve(NewtonOptions{});
    CHECK(rep.converged && rep.stop == "stagnation", "BV cell: converged %d stop %s", rep.converged, rep.stop.c_str());
    const double Ia = kin_current(cd, 1, V, {ra}, false);
    CHECK(std::abs(Ia - iex * H) < 1e-10 * iex * H, "BV cell current %.15g vs closed form %.15g", Ia, iex * H);
    // galvanostatic at that current recovers V
    CurrentDistribution cg(rect(8, 2, L, H), 1, false, 6, 6);
    cg.set_conductivity_by_attribute({{1, kap}});
    cg.add_electrode({"anode", {1}, false, 0, {ra}, true, 0.5, iex * H, {}});
    cg.add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
    auto rg = cg.solve(NewtonOptions{});
    CHECK(rg.converged, "galvanostatic BV cell not converged");
    CHECK(std::abs(cg.phi_s()[0] - V) < 1e-10, "galvanostatic phi_s %.15g vs %.15g", cg.phi_s()[0], V);
    // 4 -------------------------------------------------------------------------------------------- Jacobian
    Eigen::VectorXd z = cg.state();
    for (int i = 0; i < z.size(); ++i) z(i) += 0.01 * std::sin(1.7 * i + 0.3);
    auto J = cg.jacobian(z);
    Eigen::MatrixXd Jd(J);
    CHECK((Jd - Jd.transpose()).cwiseAbs().maxCoeff() <= 1e-12 * Jd.cwiseAbs().maxCoeff(), "Jacobian not symmetric");
    double err = 0, scale = Jd.cwiseAbs().maxCoeff();
    for (int j = 0; j < z.size(); ++j) {
      Eigen::VectorXd zp = z, zm = z;
      const double h = 1e-7;
      zp(j) += h, zm(j) -= h;
      Eigen::VectorXd col = (cg.residual(zp) - cg.residual(zm)) / (2 * h);
      err = std::max(err, (col - Jd.col(j)).cwiseAbs().maxCoeff());
    }
    CHECK(err < 1e-5 * scale, "Jacobian vs central difference: %g (scale %g)", err, scale);
  }
  // 5 ---------------------------------------------------------------------------------------------- axisymmetric
  {
    // cylinder 0<r<R, 0<z<Hc: linear kinetics on z = 0 and z = Hc, insulating r = R; uniform field in z
    const double R = 0.004, Hc = 0.01;
    MeshInput m = rect(4, 6, R, Hc);
    // relabel: rect attrs 1 x=0 (axis), 2 x=R (wall), 3 y=0 (anode), 4 y=Hc (cathode)
    CurrentDistribution cd(m, 2, true, 8, 8);
    cd.set_conductivity_by_attribute({{1, 1.5}});
    Reaction r{Reaction::Kind::Linear, 3.0, 0.0, 0.5, 0.5, 0};
    cd.add_electrode({"anode", {3}, false, 0, {r}, false, 0.2, 0, {}});
    cd.add_electrode({"cathode", {4}, false, 0, {r}, false, 0.0, 0, {}});
    auto rep = cd.solve(NewtonOptions{});
    CHECK(rep.converged, "axisymmetric cylinder not converged");
    const double k = 3.0 * f;  // di/deta
    const double i = 0.2 / (2.0 / k + Hc / 1.5);
    const double I = kin_current(cd, 3, 0.2, {r}, true);
    CHECK(std::abs(I - i * M_PI * R * R) < 1e-10 * i * M_PI * R * R, "axisymmetric current %.15g vs %.15g", I,
          i * M_PI * R * R);
  }
  // 6, 7 ------------------------------------------------------------------------------------------- DtN disk, P2 arc
  {
    // polar quarter disk rho < Rb, theta in [0, pi/2] from +z: n radial x 4n angular (fan at the origin)
    auto disk = [](int nr, int nt, double r0, double Rb, int order) {
      MeshInput m;
      std::vector<double> rho;
      for (int i = 0; i <= nr; ++i) rho.push_back(Rb * i / nr);
      m.vertices = {0.0, 0.0};
      auto id = [&](int i, int j) { return 1 + (i - 1) * (nt + 1) + j; };
      for (int i = 1; i <= nr; ++i)
        for (int j = 0; j <= nt; ++j) {
          double th = 0.5 * M_PI * j / nt;
          m.vertices.push_back(rho[i] * std::sin(th));
          m.vertices.push_back(rho[i] * std::cos(th));
        }
      for (int j = 0; j < nt; ++j) m.triangles.insert(m.triangles.end(), {0, id(1, j), id(1, j + 1)});
      for (int i = 1; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
          int a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
          m.triangles.insert(m.triangles.end(), {a, b, c, a, c, d});
        }
      m.element_attr.assign(m.triangles.size() / 3, 1);
      for (int j = 0; j < nt; ++j) {  // outer arc
        m.boundary.insert(m.boundary.end(), {id(nr, j), id(nr, j + 1)});
        m.boundary_attr.push_back(4);
        if (order == 2) {
          double th = 0.5 * M_PI * (j + 0.5) / nt;
          m.edge_node_vertices.insert(m.edge_node_vertices.end(), {id(nr, j), id(nr, j + 1)});
          m.edge_node_xy.insert(m.edge_node_xy.end(), {Rb * std::sin(th), Rb * std::cos(th)});
        }
      }
      for (int i = 0; i < nr; ++i) {  // axis (j = 0) and plane (j = nt)
        int a0 = i == 0 ? 0 : id(i, 0), a1 = id(i + 1, 0);
        int b0 = i == 0 ? 0 : id(i, nt), b1 = id(i + 1, nt);
        m.boundary.insert(m.boundary.end(), {a0, a1, b0, b1});
        m.boundary_attr.push_back(3);
        m.boundary_attr.push_back(rho[i + 1] <= r0 + 1e-15 ? 1 : 2);
      }
      return m;
    };
    const double r0 = 1e-3, Rb = 5e-3, V = 0.1, kap = 1.0;
    CurrentDistribution cd(disk(40, 32, r0, Rb, 2), 2, true, 8, 8);
    cd.set_conductivity_by_attribute({{1, kap}});
    cd.add_electrode({"disk", {1}, true, 0.0, {}, false, V, 0, {}});
    cd.set_exterior_halfspace(4, Rb, 12, kap);
    auto rep = cd.solve(NewtonOptions{});
    CHECK(rep.converged, "DtN disk not converged");
    const double I = 2 * M_PI * sum_at(cd.residual_full(0), cd.electrode_dofs(0));
    CHECK(std::abs(I / (4 * kap * r0 * V) - 1) < 0.02, "DtN disk current %.6g vs 4 kappa r0 V = %.6g", I, 4 * kap * r0 * V);
    // truncating the exterior at Rb with an insulating arc instead changes the current by > 2 % (the DtN matters)
    CurrentDistribution cn(disk(40, 32, r0, Rb, 2), 2, true, 8, 8);
    cn.set_conductivity_by_attribute({{1, kap}});
    cn.add_electrode({"disk", {1}, true, 0.0, {}, false, V, 0, {}});
    cn.add_electrode({"far", {4}, true, 0.0, {}, false, 0.0, 0, {}});
    cn.solve(NewtonOptions{});
    const double In = 2 * M_PI * sum_at(cn.residual_full(0), cn.electrode_dofs(0));
    CHECK(std::abs(In / I - 1) > 0.02, "a grounded outer arc should differ from the DtN: %g vs %g", In, I);
    // 7: arc length of the curved outer boundary (planar measure, no r) converges at order 4 for P2
    double e[2];
    for (int t = 0; t < 2; ++t) {
      CurrentDistribution ca(disk(4, 4 << t, r0, Rb, 2), 2, false, 8, 8);
      auto q = ca.boundary_points({4});
      (void)q;
      ca.set_conductivity_by_attribute({{1, 1.0}});
      ca.add_electrode({"arc", {4}, true, 0.0, {}, false, 0.0, 0, {}});
      ca.set_state(ca.initial_state());
      auto bq = ca.boundary_quadrature({4}, 8);
      e[t] = std::abs(bq.w.sum() - 0.5 * M_PI * Rb);
    }
    const double p = std::log2(e[0] / e[1]);
    CHECK(p > 3.7, "isoparametric P2 arc length order %g (errors %g, %g)", p, e[0], e[1]);
    // a far-field kappa that differs from the electrolyte, and an arc that is not rho = Rb, are refused
    for (int t = 0; t < 2; ++t) {
      CurrentDistribution cb(disk(8, 8, r0, Rb, 1), 1, true, 6, 6);
      cb.set_conductivity_by_attribute({{1, 2.5}});
      cb.add_electrode({"disk", {1}, true, 0.0, {}, false, V, 0, {}});
      cb.set_exterior_halfspace(4, t == 0 ? Rb : 0.8 * Rb, 12, t == 0 ? 1.0 : 2.5);
      bool ok = false;
      try {
        cb.solve(NewtonOptions{});
      } catch (const std::exception& ex) {
        ok = std::string(ex.what()).find(t == 0 ? "conductivity of the adjacent" : "!= Rb") != std::string::npos;
        if (!ok) std::printf("message: %s\n", ex.what());
      }
      CHECK(ok, "DtN with a wrong %s accepted", t == 0 ? "kappa" : "Rb");
    }
  }
  // 8 --------------------------------------------------------------------------------------------------- temperature
  {
    const double L = 0.01, H = 0.005;
    Reaction ra{Reaction::Kind::BV, 10.0, 0.05, 0.5, 0.5, 0}, rc{Reaction::Kind::BV, 10.0, -0.02, 0.5, 0.5, 0};
    CurrentDistribution cd(rect(8, 2, L, H), 1, false, 6, 6);
    cd.set_conductivity_by_attribute({{1, 1.0}});
    cd.add_electrode({"anode", {1}, false, 0, {ra}, true, 0.5, 0.25, {}});
    cd.add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
    NewtonOptions o;
    o.temperature = 330.0;
    auto rep = cd.solve(o);
    CHECK(rep.converged, "T = 330 K solve not converged");
    const Eigen::VectorXd z = cd.state();
    CHECK(cd.relres(z) < 1e-13, "relres at the T = 330 K solution: %g (evaluated at another temperature?)", cd.relres(z));
    CHECK((cd.newton_update(z) - z).cwiseAbs().maxCoeff() < 1e-12, "newton_update moves the T = 330 K solution");
    CHECK(cd.residual(z).norm() < 1e-12, "residual at the T = 330 K solution: %g", cd.residual(z).norm());
    const Eigen::VectorXd rf = cd.residual_full(-1);
    double rint = 0;
    for (int d = 0; d < rf.size(); ++d) rint = std::max(rint, std::abs(rf(d)));
    CHECK(rint < 1e-12, "residual_full at the T = 330 K solution: %g", rint);
    CHECK(cd.temperature() == 330.0, "temperature() not kept");
    // set_temperature() is honoured by solve() (one source of truth for T)
    CurrentDistribution c2(rect(8, 2, L, H), 1, false, 6, 6);
    c2.set_conductivity_by_attribute({{1, 1.0}});
    c2.add_electrode({"anode", {1}, false, 0, {ra}, true, 0.5, 0.25, {}});
    c2.add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
    c2.set_temperature(330.0);
    auto r2 = c2.solve(NewtonOptions{});
    CHECK(r2.converged && (c2.state() - z).cwiseAbs().maxCoeff() < 1e-12,
          "set_temperature(330) then solve() differs from solve(temperature=330)");
    CHECK(c2.relres(c2.state()) < 1e-13, "relres after set_temperature/solve: %g", c2.relres(c2.state()));
    CurrentDistribution c3(rect(8, 2, L, H), 1, false, 6, 6);
    c3.set_conductivity_by_attribute({{1, 1.0}});
    c3.add_electrode({"anode", {1}, false, 0, {ra}, true, 0.5, 0.25, {}});
    c3.add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
    c3.solve(NewtonOptions{});
    CHECK(std::abs(c3.phi_s()[0] - c2.phi_s()[0]) > 1e-3, "298.15 K and 330 K give the same phi_s: T not used");
    CHECK(c3.temperature() == 298.15, "default temperature %g", c3.temperature());
    bool threw = false;
    try { cd.set_temperature(-1); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "set_temperature(-1) accepted");
  }
  // 9 --------------------------------------------------------------------------------------------------- validation
  {
    auto throws = [](const char* what, auto&& f, const char* needle) {
      try {
        f();
      } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) == std::string::npos)
          std::printf("FAIL: %s: message '%s' lacks '%s'\n", what, e.what(), needle), ++failures;
        return;
      }
      std::printf("FAIL: %s: accepted\n", what);
      ++failures;
    };
    Reaction r{Reaction::Kind::BV, 10.0, 0.0, 0.5, 0.5, 0};
    auto cell = [&]() {
      auto c = std::make_unique<CurrentDistribution>(rect(4, 2, 0.01, 0.005), 1, false, 6, 6);
      c->set_conductivity_by_attribute({{1, 1.0}});
      return c;
    };
    {
      auto c = cell();
      c->add_electrode({"a", {1}, false, 0, {r}, true, 0.5, 0.1, {}});
      c->add_electrode({"c", {2}, false, 0, {r}, false, 0.0, 0, {}});
      throws("phi_s before solve", [&] { c->phi_s(); }, "no state");
      throws("residual with a short z", [&] { c->residual(Eigen::VectorXd::Zero(3)); }, "unknowns");
      throws("relres with a short z", [&] { c->relres(Eigen::VectorXd::Zero(3)); }, "unknowns");
      throws("newton_update with a short z", [&] { c->newton_update(Eigen::VectorXd::Zero(3)); }, "unknowns");
      throws("set_state with a short z", [&] { c->set_state(Eigen::VectorXd::Zero(3)); }, "unknowns");
      throws("boundary_mass of a missing attribute", [&] { c->boundary_mass({42}); }, "attribute 42");
      throws("residual_full before solve", [&] { c->residual_full(-1); }, "no state");
      c->solve(NewtonOptions{});
      throws("residual_full(99)", [&] { c->residual_full(99); }, "out of range");
      throws("initial = 'bogus'", [&] { NewtonOptions o; o.initial = "bogus"; c->solve(o); }, "initial");
    }
    {
      auto c = cell();
      c->add_electrode({"a", {1}, false, 0, {r}, true, 0.5, 0.1, {}});
      c->add_electrode({"c", {2}, false, 0, {r}, true, 0.0, -0.1, {}});
      throws("no grounded electrode", [&] { c->solve(NewtonOptions{}); }, "no grounded electrode");
    }
    {
      auto c = cell();
      c->add_electrode({"a", {1}, false, 0, {r}, false, 0.5, 0, {}});
      throws("attribute in two electrodes", [&] { c->add_electrode({"b", {1}, true, 0.0, {}, false, 0.0, 0, {}}); },
             "belongs to electrodes");
      throws("insulator flux on a missing attribute", [&] { c->set_insulator_flux(7, Eigen::VectorXd::Zero(1)); },
             "attribute 7");
      throws("insulator flux on an electrode", [&] { c->set_insulator_flux(1, c->boundary_points({1}).col(0)); },
             "belongs to electrode");
      Reaction bad = r;
      bad.i0 = -1;
      throws("i0 < 0", [&] { c->add_electrode({"x", {3}, false, 0, {bad}, false, 0, 0, {}}); }, "i0 must be > 0");
      bad = r;
      bad.alpha_a = -0.5;
      throws("alpha < 0", [&] { c->add_electrode({"x", {3}, false, 0, {bad}, false, 0, 0, {}}); }, "transfer coefficients");
      Reaction ta{Reaction::Kind::TafelAnodic, 1.0, 0.0, 0.5, 0.5, -0.06};
      throws("tafel_a with A < 0", [&] { c->add_electrode({"x", {3}, false, 0, {ta}, false, 0, 0, {}}); }, "A > 0");
      Reaction tc{Reaction::Kind::TafelCathodic, 1.0, 0.0, 0.5, 0.5, 0.06};
      throws("tafel_c with A > 0", [&] { c->add_electrode({"x", {3}, false, 0, {tc}, false, 0, 0, {}}); }, "A < 0");
    }
    throws("quadrature degree 0", [&] { CurrentDistribution c(rect(2, 2, 1, 1), 1, false, 0, 0); }, "quadrature degrees");
  }
  // 10 -------------------------------------------------------------------------------------------------- robustness
  {
    // symmetric BV cell, i0 = 10 A/m^2, kappa = 1 S/m, L = 1 cm: large cell voltages
    const double L = 0.01, H = 0.005;
    Reaction ra{Reaction::Kind::BV, 10.0, 0.05, 0.5, 0.5, 0}, rc{Reaction::Kind::BV, 10.0, -0.02, 0.5, 0.5, 0};
    auto make = [&](double V) {
      auto c = std::make_unique<CurrentDistribution>(rect(8, 2, L, H), 1, false, 6, 6);
      c->set_conductivity_by_attribute({{1, 1.0}});
      c->add_electrode({"anode", {1}, false, 0, {ra}, false, V, 0, {}});
      c->add_electrode({"cathode", {2}, false, 0, {rc}, false, 0.0, 0, {}});
      return c;
    };
    NewtonOptions prim;
    prim.initial = "primary";
    for (double V : {5.0, 10.0, 20.0}) {
      auto c = make(V);
      auto rep = c->solve(prim);
      CHECK(rep.converged, "initial='primary' at V = %g: stop %s after %d iterations", V, rep.stop.c_str(), rep.iterations);
      // the same answer by continuation from 2 V with warm starts
      auto w = make(2.0);
      auto r0 = w->solve(NewtonOptions{});
      bool ok = r0.converged;
      for (double v = 3.0; v <= V + 1e-9 && ok; v += 1.0) {
        w->set_control(0, v);
        NewtonOptions st;
        st.initial = "state";
        auto r = w->solve(st);
        ok = r.converged && r.iterations < 60;
      }
      CHECK(ok, "continuation to V = %g did not converge", V);
      if (ok)
        CHECK((w->state() - c->state()).cwiseAbs().maxCoeff() < 1e-10, "continuation and 'primary' differ at V = %g", V);
    }
    // a warm start from the converged state is immediate
    auto c = make(1.0);
    c->solve(NewtonOptions{});
    NewtonOptions st;
    st.initial = "state";
    auto r = c->solve(st);
    CHECK(r.converged && r.iterations <= 2, "warm start from the solution took %d iterations", r.iterations);
  }
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("test_chem_current_distribution: all passed\n");
  return 0;
}
