// Critical surfaces Jc(B, T) of technical superconductors and the current-margin arithmetic that
// goes with them (load line, short-sample limit, current-sharing temperature).
//
// Published parametrizations, see docs/validation/hts.md for the references and the reproduced
// reference points:
//  * NbTi: Bottura's practical fit, Jc = (C0/B) b^alpha (1-b)^beta (1-t^1.7)^gamma with
//    b = B/Bc2(T), Bc2(T) = Bc20 (1 - t^1.7), t = T/Tc0.
//  * Nb3Sn: the Durham/ITER "unified" scaling law at fixed axial strain,
//    Jc = (C/B) s(eps) (1 - t^1.52)(1 - t^2) b^p (1 - b)^q with Bc2*(T, eps) = Bc20 s(eps)(1 - t^1.52),
//    Tc*(eps) = Tc0 s(eps)^(1/3) and the deviatoric strain function s(eps).
//  * REBCO: the anisotropic (elliptical) Kim form Jc = Jc0 / [1 + sqrt((k B_par)^2 + B_perp^2)/B0]^b
//    with B_perp normal to the tape, times a reduced-temperature factor.
//  * table: a user table Jc(|B_eff|, T), bilinear in |B_eff| (= the anisotropic effective field when
//    an anisotropy weight k is given) and T.
#pragma once

#include <memory>
#include <string>

#include "axionfem/spec.hpp"

namespace axf {

class JcLaw {
 public:
  virtual ~JcLaw() = default;
  // Critical current density of the superconductor itself [A/m^2]. The field is split into the
  // component normal to the tape plane (B_perp) and the in-plane component (B_par); isotropic laws
  // use only sqrt(B_perp^2 + B_par^2).
  virtual double Jc(double B_perp, double B_par, double T) const = 0;
  virtual const char* law() const = 0;
  // Critical temperature at field B (Jc -> 0); used as the upper bracket of the temperature margin.
  virtual double Tc_at(double B) const = 0;
  virtual bool anisotropic() const { return false; }
  // False for a fit or table that carries no temperature dependence: a temperature margin would then
  // be meaningless and is not reported.
  virtual bool temperature_dependent() const { return true; }
  // Isotropic shorthand: the whole field normal to the tape (the worst case for a tape).
  double Jc_perp(double B, double T) const { return Jc(B, 0.0, T); }
};

std::unique_ptr<JcLaw> make_jc(const spec::Conductor& c);

// Smallest s > 0 with s * J = Jc(s * B_perp, s * B_par, T): the load line (current and field both
// proportional to the magnet current) meets the critical surface there.  Returns 0 if J <= 0 and
// a large value when the conductor never reaches its limit (both are reported, never silently used).
double load_line_scale(const JcLaw& jc, double B_perp, double B_par, double J, double T);

// Current-sharing temperature: Jc(B, T_cs) = J.  Returns T_op when the conductor is already over its
// critical current, and the critical temperature when J <= 0.
double current_sharing_T(const JcLaw& jc, double B_perp, double B_par, double J, double T_op);

}  // namespace axf
