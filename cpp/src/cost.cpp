#include "axionfem/cost.hpp"

#include <cmath>

namespace axf::cost {

nlohmann::json Breakdown::to_json() const {
  return {{"magnet_MCHF", magnet}, {"cryostat_MCHF", cryostat}, {"iron_MCHF", iron}, {"total_MCHF", total}};
}

Breakdown evaluate(const spec::CostSpec& c, double W_J, double A_cold_m2, double iron_kg) {
  Breakdown b;
  b.magnet = c.a * std::pow(W_J / 1e6, c.p_energy);
  b.cryostat = c.a * c.kappa * std::pow(A_cold_m2, c.p_area);
  b.iron = c.iron_cost_per_kg * iron_kg;
  b.total = b.magnet + b.cryostat + b.iron;
  return b;
}

double optimal_scale(const spec::CostSpec& c, double W_J, double A_cold_m2) {
  // C(l) = a [ (W l^-2)^pE + kappa (A l^2)^pA ];  dC/dl = 0
  //   -> 2 pE W^pE l^(-2 pE) = 2 pA kappa A^pA l^(2 pA)
  //   -> l^(2 (pE + pA)) = (pE/pA) W^pE / (kappa A^pA)
  const double W = W_J / 1e6;
  const double rhs = (c.p_energy / c.p_area) * std::pow(W, c.p_energy) / (c.kappa * std::pow(A_cold_m2, c.p_area));
  return std::pow(rhs, 1.0 / (2.0 * (c.p_energy + c.p_area)));
}

}  // namespace axf::cost
