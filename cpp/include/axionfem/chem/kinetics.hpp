// The chemical kinetics kernel (beta; formulation: docs/guides/chemistry.md).
//
// Reaction j:  sum_i nu'_ij X_i -> sum_i nu''_ij X_i,  nu = nu'' - nu'.  Mass-action rate of progress per unit
// volume, in concentrations:  q_j = k_f,j(T) prod_i c_i^{o'_ij} - k_r,j(T) prod_i c_i^{nu''_ij},  o' = nu' unless
// explicit forward orders are given (refused on reversible reactions).  Production omega_i = sum_j nu_ij q_j.
// Non-integer orders: c^o means max(c, 0)^o with derivative o c^{o-1} for c > 0 and 0 otherwise (the Jacobian is NOT
// regularized: it is unbounded as c -> 0+ for o < 1); integer orders are the plain polynomial at any sign.
// k(T) = A T^n exp(-Ea/(R T)); reverse: k_r = k_f / K_c (K_c T-independent, in (conc)^{sum nu_j}) or its own
// Arrhenius triple.  Units: the mechanism states {concentration, time, energy}; the kernel converts every parameter
// to SI (mol/m^3, s, J/mol) once, and rates() works in SI.  Wegscheider: the product of K^m over every null vector m
// of the reversible stoichiometry must be 1 (Kc path: K = K_c; explicit k_r: A_f/A_r with sum m dn = sum m dEa = 0);
// an inconsistent cycle is refused unless allow_nonequilibrium_cycle.
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace axf::chem {

constexpr double kGasConstantExact = 8.31446261815324;   // N_A k_B, J/(mol K), exact (SI 2019)

struct Arrhenius {
  double A = 0.0, n = 0.0, Ea = 0.0;
};

struct KineticReaction {
  std::vector<int> reac;                 // reactant species indices
  std::vector<double> nu_r;              // their stoichiometric coefficients nu'
  std::vector<int> prod;                 // product species indices
  std::vector<double> nu_p;              // nu''
  Arrhenius kf;
  bool reversible = false;
  bool has_Kc = false;
  double Kc = 0.0;
  bool has_kr = false;
  Arrhenius kr;
  std::vector<int> order_idx;            // explicit forward orders (species, order); others keep nu'
  std::vector<double> order_val;
  double dH = 0.0;                       // reaction enthalpy per unit extent (energy units of the mechanism)
};

struct Units {
  double conc = 1.0, time = 1.0, energy = 1.0;   // SI value = user value x factor
  static Units from_strings(const std::string& conc, const std::string& time, const std::string& energy);
};

class Mechanism {
 public:
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;

  Mechanism(int n_species, std::vector<KineticReaction> reactions, const Units& units = Units(),
            bool allow_nonequilibrium_cycle = false);

  int n_species() const { return ns_; }
  int n_reactions() const { return static_cast<int>(rx_.size()); }
  const std::vector<KineticReaction>& reactions_si() const { return rx_; }
  const Mat& stoichiometry() const { return S_; }      // ns x nr, nu'' - nu'
  const Units& units() const { return units_; }
  Vec concentration_to_si(const Vec& c_user) const { return c_user * units_.conc; }

  // q (nr), dq/dc (nr x ns), dq/dT (nr); T may be NaN (isothermal mechanism: every n = Ea = 0), then dq/dT = 0.
  void progress(const Vec& c, double T, Vec& q, Mat* dqdc, Vec* dqdT) const;
  // omega = S q and its derivatives.
  void rates(const Vec& c, double T, Vec& omega, Mat* domega_dc, Vec* domega_dT) const;
  // rate constants at T (and dk/dT); used by the reactor energy balance and the tests
  void rate_constants(int j, double T, double& kf, double& kr, double* dkf, double* dkr) const;

 private:
  int ns_ = 0;
  std::vector<KineticReaction> rx_;   // SI
  std::vector<std::vector<std::pair<int, double>>> fwd_, rev_;   // (species, exponent) of the two mass-action terms
  Mat S_;
  Units units_;
  void check_wegscheider() const;
};

}  // namespace axf::chem
