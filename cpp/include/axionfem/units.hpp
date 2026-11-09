// Physical constants (CODATA 2018 exact/recommended values) and conversions
// between SI and the natural Heaviside-Lorentz units used for detector geometry
// (hbar = c = k_B = 1; fields in tesla, lengths in metres).
#pragma once

#include <cmath>

namespace axf::units {

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double mu0 = 1.25663706212e-6;        // H/m (CODATA 2018)
inline constexpr double c = 299792458.0;                // m/s
inline constexpr double hbar = 1.054571817e-34;         // J s
inline constexpr double h = 6.62607015e-34;             // J s
inline constexpr double kB = 1.380649e-23;              // J/K
inline constexpr double e = 1.602176634e-19;            // C
inline constexpr double eV = 1.602176634e-19;           // J
inline constexpr double GeV = 1.602176634e-10;          // J
inline constexpr double year = 365.25 * 86400.0;        // s (Julian year)
inline constexpr double alpha = 7.2973525693e-3;        // fine-structure constant
inline constexpr double hbarc_eVm = hbar * c / e;        // eV m (exact from the constants above)

// Natural Heaviside-Lorentz conversions.
//  1 m  = 1/hbarc_eVm  eV^-1
//  1 T  = sqrt(hbar^3 c^3 / mu0) ... expressed as eV^2:  1 T = 195.35 eV^2
inline double tesla_in_eV2() {
  // Energy density B^2/(2 mu0) [J/m^3] = B_nat^2/2 [eV^4];  1 J/m^3 = hbarc^3/e eV^4.
  return std::sqrt(hbarc_eVm * hbarc_eVm * hbarc_eVm / (e * mu0));
}
inline double metre_in_inv_eV() { return 1.0 / hbarc_eVm; }

// Inductance: L_SI = mu0 * L_nat (L_nat in metres).
// Coupling integral Psi = (1/I) int A_I . B_lab:  Psi_SI [H T m^2] = mu0 * Psi_nat [T m^3].
// Scan figure of merit F = Psi^4 / L^2:  F_SI [T^4 H^2 m^8] = mu0^2 * F_nat [T^4 m^10].
inline double inductance_si_to_nat(double L_si) { return L_si / mu0; }
inline double psi_si_to_nat(double psi_si) { return psi_si / mu0; }
inline double F_si_to_nat(double F_si) { return F_si / (mu0 * mu0); }
inline double F_nat_to_si(double F_nat) { return F_nat * mu0 * mu0; }

// 1 T^2 m^3 of "int B^2 dV" corresponds to (1/(2 mu0)) J of stored energy.
inline double energy_from_B2V(double B2V) { return B2V / (2.0 * mu0); }

}  // namespace axf::units
