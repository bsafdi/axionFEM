#pragma once
// Adaptive mesh refinement bookkeeping shared by the axisymmetric and the 3D engine (mesh.adapt).
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace axf {

// One history row of an adaptive cycle: mesh size, Psi, L_p, W_mag, the estimated relative error and, when the
// results carry pickup.losses, every surface's loss factor Gamma and, for surfaces with rf data, R_eff at f_min.
nlohmann::json adapt_row(int cycle, long elements, long dofs, const nlohmann::json& results, double estimated_error);

// Largest relative change between two rows over the watched quantities (mesh.adapt.watch: Psi, L_p, W_mag, Gamma).
// "Gamma" compares the losses R_eff of the surfaces with rf data if there are any, else every surface's Gamma, over
// the surfaces holding at least 1 % of the total (a negligible share cannot hold the loop). Infinity (never
// converged) when a watched value is missing or not finite, or a surface appeared.
double adapt_change(const nlohmann::json& previous, const nlohmann::json& row, const std::vector<std::string>& watch);

}  // namespace axf
