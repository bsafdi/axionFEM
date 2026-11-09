// Size-regulating cost model: the two-term form of A. Herve's magnet-cost law (stored energy and cold surface, as
// quoted e.g. in arXiv:1308.6822), with the adjustable coefficients of the design's `cost` section, plus warm iron.
//   C = a [ (W/MJ)^p_E + kappa (A_cold/m^2)^p_A ] + c_Fe M_Fe     [MCHF]
// This is a metric for comparing designs, not a construction estimate. The default coefficients (spec::CostSpec) are
// illustrative; no source is claimed for them.
#pragma once

#include <nlohmann/json.hpp>

#include "axionfem/spec.hpp"

namespace axf::cost {

struct Breakdown {
  double magnet = 0, cryostat = 0, iron = 0, total = 0;  // MCHF
  nlohmann::json to_json() const;
};

Breakdown evaluate(const spec::CostSpec& c, double W_J, double A_cold_m2, double iron_kg);

// For an iron-free design whose shape is fixed, W scales as l^-2 and A_cold as
// l^2 at fixed F; the cost-optimal size factor l* (relative to the evaluated
// design) satisfies l*^3 = (p_E/p_A) W^p_E / (kappa A^p_A) in MJ, m^2 units.
double optimal_scale(const spec::CostSpec& c, double W_J, double A_cold_m2);

}  // namespace axf::cost
