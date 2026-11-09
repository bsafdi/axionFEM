// Kinetics kernel (see cpp/include/axionfem/chem/kinetics.hpp).
#include "axionfem/chem/kinetics.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>


namespace axf::chem {

namespace {
bool is_integer(double e) { return e == std::round(e); }

// c^e with the convention for non-integer orders (max(c, 0)^e) and its derivative
double pw(double c, double e) {
  if (e == 0.0) return 1.0;
  if (is_integer(e)) return std::pow(c, e);
  return c > 0 ? std::pow(c, e) : 0.0;
}
double dpw(double c, double e) {
  if (e == 0.0) return 0.0;
  if (is_integer(e)) return e == 1.0 ? 1.0 : e * std::pow(c, e - 1.0);
  return c > 0 ? e * std::pow(c, e - 1.0) : 0.0;
}

// product of mass-action factors and its gradient (no division: zeros are fine)
double mass_action(const std::vector<std::pair<int, double>>& terms, const Eigen::VectorXd& c,
                   std::vector<std::pair<int, double>>* grad) {
  double p = 1.0;
  for (const auto& [i, e] : terms) p *= pw(c[i], e);
  if (grad) {
    grad->clear();
    for (size_t m = 0; m < terms.size(); ++m) {
      double g = dpw(c[terms[m].first], terms[m].second);
      for (size_t l = 0; l < terms.size(); ++l)
        if (l != m) g *= pw(c[terms[l].first], terms[l].second);
      grad->push_back({terms[m].first, g});
    }
  }
  return p;
}

double unit_factor(const std::string& s, const char* kind) {
  struct U { const char* name; double f; };
  static const U conc[] = {{"mol/m^3", 1.0}, {"mol/m3", 1.0}, {"mol/L", 1e3}, {"mol/l", 1e3}, {"mol/dm^3", 1e3},
                           {"M", 1e3}, {"mol/cm^3", 1e6}, {"mol/cm3", 1e6}, {"kmol/m^3", 1e3}, {"kmol/m3", 1e3}};
  static const U tim[] = {{"s", 1.0}, {"min", 60.0}, {"h", 3600.0}, {"ms", 1e-3}};
  static const U en[] = {{"J/mol", 1.0}, {"kJ/mol", 1e3}, {"cal/mol", 4.184}, {"kcal/mol", 4184.0},
                         {"J/kmol", 1e-3}};
  const U* tab = nullptr;
  size_t n = 0;
  std::string k(kind);
  if (k == "concentration") { tab = conc; n = sizeof(conc) / sizeof(U); }
  else if (k == "time") { tab = tim; n = sizeof(tim) / sizeof(U); }
  else { tab = en; n = sizeof(en) / sizeof(U); }
  for (size_t i = 0; i < n; ++i)
    if (s == tab[i].name) return tab[i].f;
  std::ostringstream m;
  m << "Mechanism units: unknown " << kind << " unit '" << s << "'; use one of";
  for (size_t i = 0; i < n; ++i) m << " '" << tab[i].name << "'";
  throw std::invalid_argument(m.str());
}
}  // namespace

Units Units::from_strings(const std::string& conc, const std::string& time, const std::string& energy) {
  Units u;
  u.conc = unit_factor(conc.empty() ? "mol/m^3" : conc, "concentration");
  u.time = unit_factor(time.empty() ? "s" : time, "time");
  u.energy = unit_factor(energy.empty() ? "J/mol" : energy, "energy");
  return u;
}

Mechanism::Mechanism(int n_species, std::vector<KineticReaction> reactions, const Units& units_in,
                     bool allow_nonequilibrium_cycle)
    : ns_(n_species), rx_(std::move(reactions)), units_(units_in) {
  if (ns_ <= 0) throw std::invalid_argument("Mechanism: the number of species must be positive");
  const double cf = units_.conc, tf = units_.time, ef = units_.energy;
  for (size_t j = 0; j < rx_.size(); ++j) {
    KineticReaction& r = rx_[j];
    auto where = [&]() { return "Mechanism reaction " + std::to_string(j) + ": "; };
    if (r.reac.size() != r.nu_r.size() || r.prod.size() != r.nu_p.size() || r.order_idx.size() != r.order_val.size())
      throw std::invalid_argument(where() + "species and coefficient lists differ in length");
    for (int i : r.reac) if (i < 0 || i >= ns_) throw std::invalid_argument(where() + "reactant index out of range");
    for (int i : r.prod) if (i < 0 || i >= ns_) throw std::invalid_argument(where() + "product index out of range");
    for (int i : r.order_idx) if (i < 0 || i >= ns_) throw std::invalid_argument(where() + "order index out of range");
    if (r.reversible && !r.order_idx.empty())
      throw std::invalid_argument(where() + "explicit forward orders on a REVERSIBLE reaction are refused (the reverse "
                                  "rate would be thermodynamically inconsistent); write it as two irreversible "
                                  "reactions or drop the orders");
    if (r.reversible && (r.has_Kc == r.has_kr))
      throw std::invalid_argument(where() + "a reversible reaction needs exactly one of Kc or kr");
    if (r.reversible && r.has_Kc && !(r.Kc > 0))
      throw std::invalid_argument(where() + "Kc must be positive");
    if (!r.reversible && (r.has_Kc || r.has_kr))
      throw std::invalid_argument(where() + "Kc/kr given on an irreversible reaction; set reversible");
    // forward exponents
    std::vector<std::pair<int, double>> f;
    for (size_t m = 0; m < r.reac.size(); ++m) f.push_back({r.reac[m], r.nu_r[m]});
    for (size_t m = 0; m < r.order_idx.size(); ++m) {
      bool found = false;
      for (auto& [i, e] : f)
        if (i == r.order_idx[m]) { e = r.order_val[m]; found = true; }
      if (!found) f.push_back({r.order_idx[m], r.order_val[m]});
    }
    std::vector<std::pair<int, double>> b;
    for (size_t m = 0; m < r.prod.size(); ++m) b.push_back({r.prod[m], r.nu_p[m]});
    double of = 0.0, orv = 0.0, dnu = 0.0;
    for (auto& [i, e] : f) of += e;
    for (double v : r.nu_p) { orv += v; dnu += v; }
    for (double v : r.nu_r) dnu -= v;
    // to SI
    r.kf.A *= std::pow(cf, 1.0 - of) / tf;
    r.kf.Ea *= ef;
    if (r.has_kr) {
      r.kr.A *= std::pow(cf, 1.0 - orv) / tf;
      r.kr.Ea *= ef;
    }
    if (r.has_Kc) r.Kc *= std::pow(cf, dnu);
    r.dH *= ef;
    fwd_.push_back(f);
    rev_.push_back(b);
  }
  S_ = Mat::Zero(ns_, rx_.size());
  for (size_t j = 0; j < rx_.size(); ++j) {
    for (size_t m = 0; m < rx_[j].reac.size(); ++m) S_(rx_[j].reac[m], j) -= rx_[j].nu_r[m];
    for (size_t m = 0; m < rx_[j].prod.size(); ++m) S_(rx_[j].prod[m], j) += rx_[j].nu_p[m];
  }
  if (!allow_nonequilibrium_cycle) check_wegscheider();
}

void Mechanism::check_wegscheider() const {
  std::vector<int> rev;
  for (size_t j = 0; j < rx_.size(); ++j)
    if (rx_[j].reversible) rev.push_back(static_cast<int>(j));
  if (rev.size() < 2) return;
  Mat Sr(ns_, rev.size());
  for (size_t m = 0; m < rev.size(); ++m) Sr.col(m) = S_.col(rev[m]);
  Eigen::FullPivLU<Mat> lu(Sr);
  lu.setThreshold(1e-10);
  Mat N = lu.kernel();
  if (lu.rank() == static_cast<int>(rev.size())) return;
  for (int c = 0; c < N.cols(); ++c) {
    double slnK = 0, sdn = 0, sdE = 0, scale = 0, scn = 0, scE = 0;
    for (size_t m = 0; m < rev.size(); ++m) {
      const KineticReaction& r = rx_[rev[m]];
      const double mm = N(m, c);
      double lnK, dn = 0, dE = 0;
      if (r.has_Kc) lnK = std::log(r.Kc);
      else { lnK = std::log(r.kf.A / r.kr.A); dn = r.kf.n - r.kr.n; dE = r.kf.Ea - r.kr.Ea; }
      slnK += mm * lnK; sdn += mm * dn; sdE += mm * dE;
      scale += std::fabs(mm * lnK); scn += std::fabs(mm * dn); scE += std::fabs(mm * dE);
    }
    const double tol = 1e-9;
    if (std::fabs(slnK) > tol * (1 + scale) || std::fabs(sdn) > tol * (1 + scn) || std::fabs(sdE) > tol * (1 + scE)) {
      double mx = N.col(c).cwiseAbs().maxCoeff();
      std::ostringstream m;
      m << "Mechanism: reactions [";
      bool first = true;
      for (size_t q = 0; q < rev.size(); ++q)
        if (std::fabs(N(q, c)) > 1e-12 * mx) { m << (first ? "" : ", ") << rev[q]; first = false; }
      m << "] form a cycle with prod K^m = " << std::exp(slnK) << " != 1 (sum m dn = " << sdn << ", sum m dEa = " << sdE
        << "): the equilibrium constants violate detailed balance (Wegscheider).  Make the K's consistent, or set "
           "allow_nonequilibrium_cycle to accept a driven cycle.";
      throw std::invalid_argument(m.str());
    }
  }
}

void Mechanism::rate_constants(int j, double T, double& kf, double& kr, double* dkf, double* dkr) const {
  const KineticReaction& r = rx_[j];
  auto arr = [&](const Arrhenius& a, double& k, double* dk) {
    if (std::isnan(T)) {
      if (a.n != 0.0 || a.Ea != 0.0)
        throw std::invalid_argument("Mechanism reaction " + std::to_string(j) +
                                    ": temperature-dependent rate constant evaluated without a temperature");
      k = a.A;
      if (dk) *dk = 0.0;
      return;
    }
    const double sgn = -1.0;  // exp(-Ea / (R T))
    k = a.A * std::pow(T, a.n) * std::exp(sgn * a.Ea / (kGasConstantExact * T));
    if (dk) *dk = k * (a.n / T - sgn * a.Ea / (kGasConstantExact * T * T));
  };
  arr(r.kf, kf, dkf);
  kr = 0.0;
  if (dkr) *dkr = 0.0;
  if (r.reversible) {
    if (r.has_Kc) {
      kr = kf / r.Kc;
      if (dkr && dkf) *dkr = *dkf / r.Kc;
    } else {
      arr(r.kr, kr, dkr);
    }
  }
}

void Mechanism::progress(const Vec& c, double T, Vec& q, Mat* dqdc, Vec* dqdT) const {
  if (c.size() != ns_) throw std::invalid_argument("Mechanism::progress: c has the wrong length");
  const int nr = n_reactions();
  q.resize(nr);
  if (dqdc) dqdc->setZero(nr, ns_);
  if (dqdT) dqdT->setZero(nr);
  std::vector<std::pair<int, double>> gf, gb;
  for (int j = 0; j < nr; ++j) {
    double kf, kr, dkf, dkr;
    rate_constants(j, T, kf, kr, &dkf, &dkr);
    const double pf = mass_action(fwd_[j], c, dqdc ? &gf : nullptr);
    double pb = 0.0;
    if (rx_[j].reversible) pb = mass_action(rev_[j], c, dqdc ? &gb : nullptr);
    q[j] = kf * pf - kr * pb;
    if (dqdc) {
      for (auto& [i, g] : gf) (*dqdc)(j, i) += kf * g;
      if (rx_[j].reversible)
        for (auto& [i, g] : gb) (*dqdc)(j, i) -= kr * g;
    }
    if (dqdT) (*dqdT)[j] = std::isnan(T) ? 0.0 : dkf * pf - dkr * pb;
  }
}

void Mechanism::rates(const Vec& c, double T, Vec& omega, Mat* domega_dc, Vec* domega_dT) const {
  Vec q;
  Mat dq;
  Vec dqT;
  progress(c, T, q, domega_dc ? &dq : nullptr, domega_dT ? &dqT : nullptr);
  omega = S_ * q;
  if (domega_dc) *domega_dc = S_ * dq;
  if (domega_dT) *domega_dT = S_ * dqT;
}

}  // namespace axf::chem
