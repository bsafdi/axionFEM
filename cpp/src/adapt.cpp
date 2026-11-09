#include "axionfem/adapt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace axf {

using json = nlohmann::json;

json adapt_row(int cycle, long elements, long dofs, const json& results, double estimated_error) {
  const json c = results.value("coupling", json::object());
  json row = {{"cycle", cycle}, {"elements", elements}, {"dofs", dofs},
              {"Psi_SI_H_T_m2", c.value("Psi_SI_H_T_m2", 0.0)}, {"L_p_H", c.value("L_p_H", 0.0)},
              {"W_mag_J", c.value("W_mag_J", 0.0)}, {"estimated_relative_error", estimated_error}};
  // the pickup's surface-loss factors (pickup.losses), and the losses R_eff at f_min of the surfaces with rf data
  // (with magnetoresistance R_eff uses Gamma_magnetoresistance), so that R_eff and Q can be followed and watched too
  const json pickup = results.value("pickup", json::object());
  if (pickup.contains("losses") && pickup["losses"].contains("surfaces")) {
    json gamma = json::object();
    json loss = json::object();
    for (auto& [surface, v] : pickup["losses"]["surfaces"].items()) {
      gamma[surface] = v.value("Gamma", 0.0);
      if (v.contains("R_eff_ohm")) loss[surface] = v["R_eff_ohm"].value("f_min", 0.0);
    }
    row["Gamma"] = gamma;
    if (!loss.empty()) row["R_eff_ohm"] = loss;
  }
  return row;
}

double adapt_change(const json& previous, const json& row, const std::vector<std::string>& watch) {
  auto rel = [](double before, double now) { return now != 0.0 ? std::fabs(now - before) / std::fabs(now) : 0.0; };
  const double never = std::numeric_limits<double>::infinity();
  double change = 0;
  for (const std::string& w : watch) {
    if (w == "Gamma") {
      // the losses R_eff of the surfaces with rf data when there are any (they set Q), else every surface's Gamma
      const char* key = row.contains("R_eff_ohm") ? "R_eff_ohm" : "Gamma";
      if (!row.contains(key) || !previous.contains(key) || row[key].empty()) return never;  // nothing to converge
      double total = 0;
      for (auto& [surface, g] : row[key].items()) total += g.get<double>();
      if (!std::isfinite(total) || !(total > 0)) return never;
      for (auto& [surface, g] : row[key].items()) {
        const double now = g.get<double>();
        if (now < 0.01 * total) continue;   // a negligible share of the loss cannot hold the loop
        if (!previous[key].contains(surface)) return never;   // a surface that appeared
        const double c = rel(previous[key][surface].get<double>(), now);
        if (!std::isfinite(c)) return never;   // a non-finite previous value (std::max would drop the NaN)
        change = std::max(change, c);
      }
      continue;
    }
    const char* key = w == "Psi" ? "Psi_SI_H_T_m2" : w == "L_p" ? "L_p_H" : "W_mag_J";
    const double c = rel(previous.value(key, 0.0), row.value(key, 0.0));
    if (!std::isfinite(c)) return never;
    change = std::max(change, c);
  }
  return change;
}

}  // namespace axf
