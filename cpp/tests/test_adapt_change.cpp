// Unit test of the adaptive stopping test (cpp/src/adapt.cpp, mesh.adapt.watch) on synthetic history rows:
// the default watch, the 1 % filter, the loss (R_eff) preference, and the never-converged cases (no surfaces, a
// non-finite value, a surface that appeared).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/adapt.hpp"

using json = nlohmann::json;

static int failures = 0;

static void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

static bool near(double a, double b) { return std::fabs(a - b) <= 1e-14 * std::max(1.0, std::fabs(b)); }

static json row(double L, double psi, double W, const json& gamma, const json& loss = nullptr) {
  json r = {{"L_p_H", L}, {"Psi_SI_H_T_m2", psi}, {"W_mag_J", W}};
  if (!gamma.is_null()) r["Gamma"] = gamma;
  if (!loss.is_null()) r["R_eff_ohm"] = loss;
  return r;
}

int main() {
  const std::vector<std::string> dflt = {"Psi", "L_p", "W_mag"};
  const std::vector<std::string> gam = {"Gamma", "L_p"};
  // the default watch: the largest of the three relative changes, Gamma ignored
  {
    const json a = row(1.0, 2.0, 3.0, {{"wall", 1.0}});
    const json b = row(1.01, 2.0, 3.06, {{"wall", 5.0}});
    check(near(axf::adapt_change(a, b, dflt), 0.06 / 3.06), "default watch = max over Psi, L_p, W_mag");
  }
  // Gamma: every surface's Gamma without rf data; a surface below 1 % of the total is ignored
  {
    const json a = row(1.0, 2.0, 3.0, {{"wall", 1.0}, {"lead", 0.001}});
    const json b = row(1.0, 2.0, 3.0, {{"wall", 1.02}, {"lead", 0.002}});
    check(near(axf::adapt_change(a, b, gam), 0.02 / 1.02), "Gamma over surfaces >= 1 % of the total");
  }
  // with rf data the losses R_eff are compared: a copper lead with 0.5 % of Gamma but most of the loss counts
  {
    const json a = row(1.0, 2.0, 3.0, {{"nb", 100.0}, {"cu", 0.4}}, {{"nb", 1e-7}, {"cu", 1e-5}});
    const json b = row(1.0, 2.0, 3.0, {{"nb", 100.0}, {"cu", 0.5}}, {{"nb", 1e-7}, {"cu", 1.25e-5}});
    check(near(axf::adapt_change(a, b, gam), 0.25e-5 / 1.25e-5), "R_eff preferred when rf data exist");
  }
  // never converged: no surfaces, empty, non-finite, a surface that appeared
  {
    const json none = row(1.0, 2.0, 3.0, nullptr);
    check(std::isinf(axf::adapt_change(none, none, gam)), "Gamma watched without surfaces never converges");
    const json empty = row(1.0, 2.0, 3.0, json::object());
    check(std::isinf(axf::adapt_change(empty, empty, gam)), "an empty Gamma never converges");
    const json a = row(1.0, 2.0, 3.0, {{"wall", 1.0}});
    const json nan = row(1.0, 2.0, 3.0, {{"wall", std::nan("")}});
    check(std::isinf(axf::adapt_change(a, nan, gam)), "a non-finite Gamma never converges");
    const json nan_before = row(1.0, 2.0, 3.0, {{"wall", std::nan("")}, {"lead", 0.5}});
    const json fine_now = row(1.0, 2.0, 3.0, {{"wall", 1.0}, {"lead", 0.5}});
    check(std::isinf(axf::adapt_change(nan_before, fine_now, gam)), "a non-finite previous Gamma never converges");
    const json appeared = row(1.0, 2.0, 3.0, {{"wall", 1.0}, {"new", 1.0}});
    check(std::isinf(axf::adapt_change(a, appeared, gam)), "a surface that appeared never converges");
    const json lnan = row(std::nan(""), 2.0, 3.0, nullptr);
    check(std::isinf(axf::adapt_change(row(1.0, 2.0, 3.0, nullptr), lnan, dflt)), "a non-finite L_p never converges");
  }
  // adapt_row: Gamma and R_eff at f_min from pickup.losses
  {
    const json results = {{"coupling", {{"L_p_H", 2.0}, {"Psi_SI_H_T_m2", 1.0}, {"W_mag_J", 3.0}}},
                          {"pickup", {{"losses", {{"surfaces", {{"nb", {{"Gamma", 4.0}}},
                                                               {"cu", {{"Gamma", 0.5},
                                                                       {"R_eff_ohm", {{"f_min", 1e-5}, {"f_max", 3e-5}}}}}}}}}}}};
    const json r = axf::adapt_row(2, 100, 400, results, 0.1);
    check(r["Gamma"]["nb"] == 4.0 && r["Gamma"]["cu"] == 0.5, "row Gamma per surface");
    check(r["R_eff_ohm"].size() == 1 && r["R_eff_ohm"]["cu"] == 1e-5, "row R_eff at f_min for rf surfaces only");
    check(r["L_p_H"] == 2.0 && r["dofs"] == 400 && r["cycle"] == 2, "row coupling and mesh");
  }
  std::printf(failures ? "%d failure(s)\n" : "test_adapt_change: all passed\n", failures);
  return failures ? 1 : 0;
}
