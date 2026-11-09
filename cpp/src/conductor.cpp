#include "axionfem/conductor.hpp"

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"

namespace axf {

namespace {

// Field entering an isotropic law.
inline double babs(double Bp, double Bt) { return std::sqrt(Bp * Bp + Bt * Bt); }

// Bottura's practical fit for NbTi (L. Bottura, IEEE Trans. Appl. Supercond. 10 (2000) 1054):
//   Jc(B, T) = (C0/B) b^alpha (1 - b)^beta (1 - t^1.7)^gamma,
//   t = T/Tc0,  Bc2(T) = Bc20 (1 - t^1.7),  b = B/Bc2(T).
// The fit diverges as B^-(1-alpha) at low field, so B is clamped from below at B_min.
class NbTiBottura : public JcLaw {
 public:
  explicit NbTiBottura(const spec::Conductor& c) : c_(c) {}
  double Jc(double Bp, double Bt, double T) const override {
    const double B = std::max(babs(Bp, Bt), c_.B_min);
    if (!(B > 0) || T >= c_.Tc0) return 0.0;
    const double t17 = std::pow(T / c_.Tc0, 1.7);
    const double Bc2 = c_.Bc20 * (1.0 - t17);
    if (!(Bc2 > 0)) return 0.0;
    const double b = B / Bc2;
    if (b >= 1.0) return 0.0;
    return (c_.C0 / B) * std::pow(b, c_.alpha) * std::pow(1.0 - b, c_.beta) * std::pow(1.0 - t17, c_.gamma);
  }
  double Tc_at(double B) const override {
    // Bc2(Tc) = B  ->  Tc = Tc0 (1 - B/Bc20)^(1/1.7)
    if (B >= c_.Bc20) return 0.0;
    return c_.Tc0 * std::pow(1.0 - B / c_.Bc20, 1.0 / 1.7);
  }
  const char* law() const override { return "nbti_bottura"; }

 private:
  spec::Conductor c_;
};

// Durham/ITER "unified" scaling law for Nb3Sn at fixed axial strain (A. Godeke et al.,
// Supercond. Sci. Technol. 19 (2006) R100; L. Bottura and B. Bordini, IEEE TAS 19 (2009) 1521):
//   Jc = (C/B) s(eps) (1 - t^1.52)(1 - t^2) b^p (1 - b)^q,
//   Bc2*(T, eps) = Bc20 s(eps) (1 - t^1.52),  Tc*(eps) = Tc0 s(eps)^(1/3),  t = T/Tc*,  b = B/Bc2*.
class Nb3SnDurham : public JcLaw {
 public:
  explicit Nb3SnDurham(const spec::Conductor& c) : c_(c), s_(strain_factor(c)) {}
  double Jc(double Bp, double Bt, double T) const override {
    const double B = std::max(babs(Bp, Bt), c_.B_min);
    const double Tcs = c_.Tc0 * std::cbrt(s_);
    if (!(B > 0) || T >= Tcs) return 0.0;
    const double t = T / Tcs;
    const double mdg = 1.0 - std::pow(t, 1.52);
    const double Bc2 = c_.Bc20 * s_ * mdg;
    if (!(Bc2 > 0)) return 0.0;
    const double b = B / Bc2;
    if (b >= 1.0) return 0.0;
    return (c_.C0 / B) * s_ * mdg * (1.0 - t * t) * std::pow(b, c_.p) * std::pow(1.0 - b, c_.q);
  }
  double Tc_at(double B) const override {
    const double Tcs = c_.Tc0 * std::cbrt(s_);
    const double x = 1.0 - B / (c_.Bc20 * s_);  // 1 - t^1.52 at Bc2* = B
    if (x <= 0) return 0.0;
    return Tcs * std::pow(x, 1.0 / 1.52);
  }
  const char* law() const override { return "nb3sn_durham"; }
  // s(eps) with the asymmetric deviatoric-strain model; s = 1 at eps = 0 when Ca2 = 0.
  static double strain_factor(const spec::Conductor& c) {
    const double den = 1.0 - c.Ca1 * c.eps_0a;
    AXF_REQUIRE(den > 0, "Nb3Sn strain function: Ca1 * eps_0a must be < 1", "/conductors");
    const double d = c.Ca1 * c.Ca1 - c.Ca2 * c.Ca2;
    const double esh = d > 0 ? c.Ca2 * c.eps_0a / std::sqrt(d) : 0.0;
    const double e = c.strain;
    const double s = 1.0 + (c.Ca1 * (std::hypot(esh, c.eps_0a) - std::hypot(e - esh, c.eps_0a)) - c.Ca2 * e) / den;
    return std::max(s, 0.0);
  }

 private:
  spec::Conductor c_;
  double s_;
};

// Anisotropic (elliptical) Kim form for REBCO coated conductors, F. Grilli et al.,
// IEEE Trans. Appl. Supercond. 24 (2014) 8000508, Eq. (2):
//   Jc(B_par, B_perp) = Jc0 / [1 + sqrt((k B_par)^2 + B_perp^2)/B0]^b_exp,
// with B_perp normal to the tape.  The reduced-temperature factor ((Tc - T)/(Tc - T_ref))^m_T
// extends the (77 K) field fit to other temperatures and must be refitted for quantitative work.
class RebcoKim : public JcLaw {
 public:
  explicit RebcoKim(const spec::Conductor& c) : c_(c) {}
  double Jc(double Bp, double Bt, double T) const override {
    if (T >= c_.Tc) return 0.0;
    const double Beff = std::max(std::hypot(c_.k_anis * Bt, Bp), c_.B_min);
    const double f = std::pow(1.0 + Beff / c_.B0, c_.b_exp);
    double g = 1.0;
    if (c_.m_T != 0.0) {
      const double den = c_.Tc - c_.T_ref;
      AXF_REQUIRE(den > 0, "REBCO conductor: Tc must exceed T_ref", "/conductors");
      g = std::pow(std::max(c_.Tc - T, 0.0) / den, c_.m_T);
    }
    return c_.Jc0 * g / f;
  }
  double Tc_at(double) const override { return c_.Tc; }
  bool anisotropic() const override { return true; }
  bool temperature_dependent() const override { return c_.m_T != 0.0; }
  const char* law() const override { return "rebco_kim"; }

 private:
  spec::Conductor c_;
};

// User table Jc(|B_eff|, T), bilinear and held constant outside the table in both B and T (so the
// last temperature column is used above it; Tc_at reports that temperature, which bounds the
// temperature margin).  With k_anis != 1 the table is indexed by the anisotropic effective field
// sqrt((k B_par)^2 + B_perp^2), as in the Kim law above.
class TableJc : public JcLaw {
 public:
  explicit TableJc(const spec::Conductor& c) : c_(c) {}
  double Jc(double Bp, double Bt, double T) const override {
    const double B = std::max(c_.k_anis == 1.0 ? babs(Bp, Bt) : std::hypot(c_.k_anis * Bt, Bp), c_.B_min);
    const int nB = static_cast<int>(c_.tab_B.size()), nT = static_cast<int>(c_.tab_T.size());
    const auto frac = [](const std::vector<double>& x, double v, int& i) {
      const int n = static_cast<int>(x.size());
      if (n == 1 || v <= x.front()) { i = 0; return 0.0; }
      if (v >= x.back()) { i = n - 2; return 1.0; }
      i = static_cast<int>(std::upper_bound(x.begin(), x.end(), v) - x.begin()) - 1;
      return (v - x[i]) / (x[i + 1] - x[i]);
    };
    int iB = 0, iT = 0;
    const double tb = frac(c_.tab_B, B, iB), tt = frac(c_.tab_T, T, iT);
    const auto at = [&](int a, int b) { return c_.tab_Jc[static_cast<size_t>(a) * nT + b]; };
    const int jB = nB > 1 ? iB + 1 : iB, jT = nT > 1 ? iT + 1 : iT;
    const double v = (1 - tb) * ((1 - tt) * at(iB, iT) + tt * at(iB, jT)) +
                     tb * ((1 - tt) * at(jB, iT) + tt * at(jB, jT));
    return std::max(v, 0.0);
  }
  // The table says nothing above its last temperature; use it as the critical temperature.
  double Tc_at(double) const override { return c_.tab_T.back(); }
  bool anisotropic() const override { return c_.k_anis != 1.0; }
  bool temperature_dependent() const override { return c_.tab_T.size() > 1; }
  const char* law() const override { return "table"; }

 private:
  spec::Conductor c_;
};

}  // namespace

std::unique_ptr<JcLaw> make_jc(const spec::Conductor& c) {
  if (c.law == "nbti_bottura") return std::make_unique<NbTiBottura>(c);
  if (c.law == "nb3sn_durham") return std::make_unique<Nb3SnDurham>(c);
  if (c.law == "rebco_kim") return std::make_unique<RebcoKim>(c);
  if (c.law == "table") return std::make_unique<TableJc>(c);
  throw Error("unknown conductor law '" + c.law + "'", "/conductors/" + c.name + "/law",
              "Use nbti_bottura, nb3sn_durham, rebco_kim or table.");
}

double load_line_scale(const JcLaw& jc, double Bp, double Bt, double J, double T) {
  if (!(J > 0)) return 0.0;
  // f(s) = s J - Jc(s B, T) is increasing in s: f(0) < 0 and f -> +inf.
  double hi = 1.0;
  for (int i = 0; i < 80 && hi * J < jc.Jc(hi * Bp, hi * Bt, T); ++i) hi *= 1.5;
  double lo = 0.0;
  for (int i = 0; i < 200; ++i) {
    const double m = 0.5 * (lo + hi);
    if (m * J < jc.Jc(m * Bp, m * Bt, T)) lo = m; else hi = m;
  }
  return 0.5 * (lo + hi);
}

double current_sharing_T(const JcLaw& jc, double Bp, double Bt, double J, double T_op) {
  const double Tc = jc.Tc_at(std::sqrt(Bp * Bp + Bt * Bt));
  if (!(J > 0)) return Tc;
  if (jc.Jc(Bp, Bt, T_op) <= J) return T_op;  // already at or over the critical current
  double lo = T_op, hi = std::max(Tc, T_op);
  for (int i = 0; i < 200; ++i) {
    const double m = 0.5 * (lo + hi);
    if (jc.Jc(Bp, Bt, m) > J) lo = m; else hi = m;
  }
  return 0.5 * (lo + hi);
}

}  // namespace axf
