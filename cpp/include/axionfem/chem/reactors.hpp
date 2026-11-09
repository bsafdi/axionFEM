// 0D reactors and equilibrium in the reaction-invariant form (beta; formulation: docs/guides/chemistry.md).
// Each builder returns an axf::time::Problem (right-hand side with its analytic Jacobian, mass matrix for the DAE)
// for the time core (time_core.hpp).  SI units throughout; `source` is an optional additive source (used for
// manufactured solutions), one entry per balance (species, then
// the energy / volume row where the reactor has one) -- for the equilibrium system one entry per SPECIES, projected
// with W like the species balances.
//   batch          c' = omega(c, T) + s
//   cstr           c' = (c_in - c)/tau + omega + s
//   semibatch      c' = (q/V)(c_f - c) + omega + s,  V' = q (+ s_V);              y = [c, V]
//   const_p_gas    N' = V omega(N/V) + s,  V = sum N R T / P  (isothermal);       y = N
//   adiabatic      c' = omega + s,  rho c_p T' = sum_j (-dH_j) q_j + s_T         y = [c, T] (liquid: constant V and
//                  rho c_p); a K_c-path reversible reaction with dH != 0 is refused (K_c is T-independent here)
//   pfr            dF/dV_r = omega(F/v),  v = v0 F_tot / F_tot0 (isothermal, isobaric ideal gas)
//   equilibrium    W c' = W(S_k q_k + s),  0 = g_e(c) = prod c^{nu''} - K prod c^{nu'}  (index 1; W spans the left
//                  null space of the equilibrium stoichiometry S_e); consistent initialization keeps W c and solves
//                  g_e = 0 (Newton).
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "axionfem/chem/kinetics.hpp"
#include "axionfem/chem/time_core.hpp"

namespace axf::chem {

using SourceFn = std::function<Eigen::VectorXd(double)>;

axf::time::Problem batch_problem(const Mechanism& m, double T, SourceFn source = nullptr);
axf::time::Problem cstr_problem(const Mechanism& m, double T, double tau, const Eigen::VectorXd& c_in,
                                SourceFn source = nullptr);
axf::time::Problem semibatch_problem(const Mechanism& m, double T, double q, const Eigen::VectorXd& c_feed,
                                     SourceFn source = nullptr);
// isothermal constant-pressure ideal gas in moles: N' = V omega(N/V) + s, with the volume following the state N
axf::time::Problem const_p_gas_problem(const Mechanism& m, double T, double P, SourceFn source = nullptr);
// the reactor volume of the constant-P gas state N (V = sum N R T / P): what the reactor uses, for converting its
// moles to concentrations
double const_p_gas_volume(double T, double P, const Eigen::VectorXd& N);
axf::time::Problem adiabatic_liquid_problem(const Mechanism& m, double rho_cp, SourceFn source = nullptr);
axf::time::Problem pfr_problem(const Mechanism& m, double T, double v0, double F_tot0);

struct EquilibriumReaction {
  std::vector<int> reac, prod;
  std::vector<double> nu_r, nu_p;
  double K = 0.0;   // SI, (mol/m^3)^{sum nu}
};

class EquilibriumSystem {
 public:
  EquilibriumSystem(const Mechanism& kinetic, std::vector<EquilibriumReaction> eq, double T,
                    SourceFn source = nullptr);
  const axf::time::Problem& problem() const { return P_; }
  const Eigen::MatrixXd& W() const { return W_; }               // invariant rows (n_inv x ns)
  Eigen::VectorXd consistent_init(const Eigen::VectorXd& c0) const;   // projection onto the equilibrium manifold
  Eigen::VectorXd constraint(const Eigen::VectorXd& c) const;   // g_e(c)
  bool dae() const { return dae_; }
  void hold(std::shared_ptr<void> owner) { P_.held.push_back(std::move(owner)); }   // see time::Problem::held

 private:
  Mechanism kin_;
  std::vector<EquilibriumReaction> eq_;
  double T_;
  Eigen::MatrixXd W_;
  axf::time::Problem P_;
  bool dae_ = true;
  void g_and_jac(const Eigen::VectorXd& c, Eigen::VectorXd& g, Eigen::MatrixXd& G) const;
};

}  // namespace axf::chem
