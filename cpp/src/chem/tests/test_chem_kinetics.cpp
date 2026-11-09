// Chemistry unit tests (kinetics kernel, reactors, axf::time core):
//   1. rates and the analytic Jacobian / dT against central differences (explicit, non-integer orders, both reverse
//      paths, Arrhenius with n and Ea);
//   2. refusals: Wegscheider-inconsistent cycle, orders on a reversible reaction, K_c path with dH in the adiabatic
//      reactor;
//   3. orders of BDF1..4 (exact start) and ESDIRK3(2) on A -> B, final-time error;
//   4. adaptive bdf2 / onestep / vsvo: stop-time outputs are step ends, accuracy, the event time;
//   5. invariant-form equilibrium: consistent initialization and the constraint along an adaptive run.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "axionfem/chem/kinetics.hpp"
#include "axionfem/chem/reactors.hpp"
#include "axionfem/chem/time_core.hpp"

using namespace axf::chem;
using axf::time::Mat;
using axf::time::Vec;
static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

static KineticReaction rx(std::vector<int> r, std::vector<double> nr, std::vector<int> p, std::vector<double> np,
                          double A, double n = 0, double Ea = 0) {
  KineticReaction k;
  k.reac = r; k.nu_r = nr; k.prod = p; k.nu_p = np; k.kf = {A, n, Ea};
  return k;
}

static Mechanism a_to_b() { return Mechanism(2, {rx({0}, {1}, {1}, {1}, 0.7)}); }

int main() {
  // 1
  {
    std::vector<KineticReaction> R;
    R.push_back(rx({0, 1}, {1, 1}, {2}, {1}, 0.6));
    R.back().reversible = true; R.back().has_Kc = true; R.back().Kc = 2.5;
    R.push_back(rx({2}, {2}, {3}, {1}, 0.35, 0.5, 5e3));
    R.push_back(rx({0, 2}, {1, 1}, {3}, {1}, 0.3));
    R.back().order_idx = {0, 2}; R.back().order_val = {0.5, 1.5};
    R.push_back(rx({3}, {1}, {0, 1}, {1, 1}, 0.2));
    R.back().reversible = true; R.back().has_kr = true; R.back().kr = {0.05, 0.5, 2e3};
    Mechanism m(4, R);
    Vec c(4); c << 1.1, 0.8, 0.6, 0.5;
    const double T = 330;
    Vec w, dT; Mat J;
    m.rates(c, T, w, &J, &dT);
    double err = 0, errT = 0;
    for (int j = 0; j < 4; ++j) {
      Vec e = Vec::Zero(4); e[j] = 1e-6;
      Vec wp, wm;
      m.rates(c + e, T, wp, nullptr, nullptr); m.rates(c - e, T, wm, nullptr, nullptr);
      err = std::max(err, ((wp - wm) / 2e-6 - J.col(j)).cwiseAbs().maxCoeff());
    }
    Vec wp, wm;
    m.rates(c, T + 1e-3, wp, nullptr, nullptr); m.rates(c, T - 1e-3, wm, nullptr, nullptr);
    errT = ((wp - wm) / 2e-3 - dT).cwiseAbs().maxCoeff();
    CHECK(err < 1e-8, "Jacobian vs FD %g", err);
    CHECK(errT < 1e-6 * dT.cwiseAbs().maxCoeff(), "dT vs FD %g", errT);
  }
  // 2
  {
    std::vector<KineticReaction> R = {rx({0}, {1}, {1}, {1}, 1.2), rx({1}, {1}, {2}, {1}, 0.7), rx({2}, {1}, {0}, {1}, 0.9)};
    const double K[3] = {2.0, 1.5, 2.0 / 3.0};
    for (int j = 0; j < 3; ++j) { R[j].reversible = true; R[j].has_Kc = true; R[j].Kc = K[j]; }
    bool refused = false;
    try { Mechanism m(3, R); } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("Wegscheider") != std::string::npos;
    }
    CHECK(refused, "inconsistent cycle accepted");
    R[2].Kc = 1.0 / 3.0;
    bool ok = true;
    try { Mechanism m(3, R); } catch (...) { ok = false; }
    CHECK(ok, "consistent cycle refused");
    std::vector<KineticReaction> R2 = {rx({0}, {1}, {1}, {1}, 1.0)};
    R2[0].reversible = true; R2[0].has_Kc = true; R2[0].Kc = 2; R2[0].order_idx = {0}; R2[0].order_val = {0.5};
    refused = false;
    try { Mechanism m(2, R2); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused, "orders on a reversible reaction accepted");
    R2[0].order_idx.clear(); R2[0].order_val.clear(); R2[0].dH = -1e4;
    refused = false;
    try { adiabatic_liquid_problem(Mechanism(2, R2), 4e6); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused, "K_c path with dH accepted in the adiabatic reactor");
  }
  // 3
  {
    auto P = batch_problem(a_to_b(), std::nan(""));
    auto ex = [](double t) { Vec v(2); v << 2 * std::exp(-0.7 * t), 2 - 2 * std::exp(-0.7 * t); return v; };
    axf::time::NewtonOptions o;
    o.rtol = 1e-13;
    for (int k = 1; k <= 5; ++k) {
      double e[2];
      for (int l = 0; l < 2; ++l) {
        const double dt = l ? 0.025 : 0.05;
        std::vector<double> g;
        for (int i = 0; i <= static_cast<int>(std::lround(2.0 / dt)); ++i) g.push_back(i * dt);
        axf::time::Trajectory tr;
        if (k == 5) {
          tr = axf::time::integrate_grid_rk(P, axf::time::tableau_esdirk32(), g, ex(0), o);
        } else {
          std::vector<Vec> ys;
          for (int i = 0; i < k; ++i) ys.push_back(ex(g[i]));
          tr = axf::time::integrate_grid_bdf(P, k, g, ys, o);
        }
        e[l] = std::fabs(tr.y.back()[0] - ex(2.0)[0]);
      }
      const double p = std::log2(e[0] / e[1]), pe = k == 5 ? 3.0 : k;
      CHECK(std::fabs(p - pe) < 0.2, "order %d: observed %g", k, p);
    }
  }
  // 4
  {
    auto P = batch_problem(a_to_b(), std::nan(""));
    for (const char* meth : {"bdf2", "onestep", "vsvo"}) {
      axf::time::AdaptiveOptions o;
      o.method = meth;
      o.rtol = 1e-8;
      o.atol = Vec::Constant(2, 1e-10);
      o.tableau = axf::time::tableau_esdirk32();
      Vec y0(2); y0 << 2, 0;
      std::vector<axf::time::EventSpec> ev = {{[](double, const Vec& y) { return y[0] - 0.5; }}};
      auto tr = axf::time::integrate_adaptive(P, o, 0.0, y0, {0.5, 2.0, 5.0}, ev);
      bool hit = false;
      for (double t : tr.t) hit = hit || t == 2.0;
      CHECK(hit, "%s: output 2.0 not a step end", meth);
      CHECK(std::fabs(tr.y.back()[0] - 2 * std::exp(-3.5)) < 1e-6, "%s: value at 5", meth);
      CHECK(!tr.events.empty() && std::fabs(tr.events[0].t - std::log(4.0) / 0.7) < 1e-4, "%s: event", meth);
    }
  }
  // 5
  {
    EquilibriumReaction e;
    e.reac = {0}; e.nu_r = {1}; e.prod = {1}; e.nu_p = {1}; e.K = 3.0;
    EquilibriumSystem es(Mechanism(3, {rx({1}, {1}, {2}, {1}, 0.6)}), {e}, std::nan(""));
    Vec c0(3); c0 << 1.6, 0.2, 0.0;
    Vec c = es.consistent_init(c0);
    CHECK(std::fabs(c[0] - 0.45) < 1e-15 && std::fabs(c[1] - 1.35) < 1e-15, "projection %g %g", c[0], c[1]);
    axf::time::AdaptiveOptions o;
    o.method = "bdf2"; o.rtol = 1e-8; o.atol = Vec::Constant(3, 1e-10);
    auto tr = axf::time::integrate_adaptive(es.problem(), o, 0.0, c, {1.0, 4.0});
    double g = 0;
    for (const auto& y : tr.y) g = std::max(g, std::fabs(es.constraint(y)[0]));
    CHECK(g < 1e-12, "constraint residual %g", g);
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
