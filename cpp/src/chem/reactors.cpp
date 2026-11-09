// Reactors and equilibrium (see cpp/include/axionfem/chem/reactors.hpp).
#include "axionfem/chem/reactors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>


namespace axf::chem {

using axf::time::Problem;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

namespace {
Vec checked_source(const SourceFn& s, double t, int n, const char* who) {
  Vec v = s(t);
  if (v.size() != n)
    throw std::invalid_argument(std::string(who) + ": the source must return " + std::to_string(n) +
                                " entries (one per balance)");
  return v;
}

// rows w (as matrix rows) spanning {w : w^T S = 0}, in reduced row-echelon form of S^T (free variables = 1)
Mat left_null_rref(const Mat& S) {
  Mat A = S.transpose();   // ne x ns
  const int r = static_cast<int>(A.rows()), n = static_cast<int>(A.cols());
  std::vector<int> pivcol;
  int row = 0;
  for (int col = 0; col < n && row < r; ++col) {
    int p = row;
    for (int i = row + 1; i < r; ++i)
      if (std::fabs(A(i, col)) > std::fabs(A(p, col))) p = i;
    if (std::fabs(A(p, col)) < 1e-12) continue;
    A.row(row).swap(A.row(p));
    A.row(row) /= A(row, col);
    for (int i = 0; i < r; ++i)
      if (i != row) A.row(i) -= A(i, col) * A.row(row);
    pivcol.push_back(col);
    ++row;
  }
  std::vector<int> free;
  for (int col = 0; col < n; ++col)
    if (std::find(pivcol.begin(), pivcol.end(), col) == pivcol.end()) free.push_back(col);
  Mat W = Mat::Zero(free.size(), n);
  for (size_t f = 0; f < free.size(); ++f) {
    W(f, free[f]) = 1.0;
    for (size_t pr = 0; pr < pivcol.size(); ++pr) W(f, pivcol[pr]) = -A(pr, free[f]);
  }
  return W;
}
}  // namespace

Problem batch_problem(const Mechanism& m, double T, SourceFn source) {
  Problem P;
  P.n = m.n_species();
  P.f = [m, T](double, const Vec& c) {
    Vec w;
    m.rates(c, T, w, nullptr, nullptr);
    return w;
  };
  P.jac = [m, T](double, const Vec& c) {
    Vec w;
    Mat J;
    m.rates(c, T, w, &J, nullptr);
    return J;
  };
  if (source) {
    const int n = P.n;
    P.source = [source, n](double t) { return checked_source(source, t, n, "batch source"); };
  }
  return P;
}

Problem cstr_problem(const Mechanism& m, double T, double tau, const Vec& c_in, SourceFn source) {
  if (!(tau > 0)) throw std::invalid_argument("cstr: the residence time tau must be positive");
  if (c_in.size() != m.n_species()) throw std::invalid_argument("cstr: c_in needs one entry per species");
  Problem P;
  P.n = m.n_species();
  P.f = [m, T, tau, c_in](double, const Vec& c) {
    Vec w;
    m.rates(c, T, w, nullptr, nullptr);
    return Vec((c_in - c) / tau + w);
  };
  P.jac = [m, T, tau](double, const Vec& c) {
    Vec w;
    Mat J;
    m.rates(c, T, w, &J, nullptr);
    J.diagonal().array() -= 1.0 / tau;
    return J;
  };
  if (source) {
    const int n = P.n;
    P.source = [source, n](double t) { return checked_source(source, t, n, "cstr source"); };
  }
  return P;
}

Problem semibatch_problem(const Mechanism& m, double T, double q, const Vec& c_feed, SourceFn source) {
  const int ns = m.n_species();
  if (c_feed.size() != ns) throw std::invalid_argument("semibatch: c_feed needs one entry per species");
  Problem P;
  P.n = ns + 1;
  P.f = [m, T, q, c_feed, ns](double, const Vec& y) {
    const Vec c = y.head(ns);
    const double V = y[ns];
    Vec w;
    m.rates(c, T, w, nullptr, nullptr);
    Vec out(ns + 1);
    out.head(ns) = (q / V) * (c_feed - c) + w;
    out[ns] = q;
    return out;
  };
  P.jac = [m, T, q, c_feed, ns](double, const Vec& y) {
    const Vec c = y.head(ns);
    const double V = y[ns];
    Vec w;
    Mat Jc;
    m.rates(c, T, w, &Jc, nullptr);
    Mat J = Mat::Zero(ns + 1, ns + 1);
    J.topLeftCorner(ns, ns) = Jc;
    J.topLeftCorner(ns, ns).diagonal().array() -= q / V;
    J.col(ns).head(ns) = -(q / (V * V)) * (c_feed - c);
    return J;
  };
  if (source) {
    const int n = P.n;
    P.source = [source, n](double t) { return checked_source(source, t, n, "semibatch source"); };
  }
  return P;
}

Problem const_p_gas_problem(const Mechanism& m, double T, double Pr, SourceFn source) {
  const int ns = m.n_species();
  if (!(T > 0) || !(Pr > 0)) throw std::invalid_argument("const_p_gas: T and P must be positive");
  const double RT_P = kGasConstantExact * T / Pr;
  Problem P;
  P.n = ns;
  auto vol = [=](const Vec& N) { return N.sum() * RT_P; };
  P.f = [m, T, vol](double, const Vec& N) {
    const double V = vol(N);
    Vec w;
    m.rates(N / V, T, w, nullptr, nullptr);
    return Vec(V * w);
  };
  P.jac = [m, T, vol, RT_P, ns](double, const Vec& N) {
    const double V = vol(N);
    const double dV = RT_P;   // dV/dN_j
    Vec w;
    Mat Jc;
    m.rates(N / V, T, w, &Jc, nullptr);
    Mat dc = Mat::Identity(ns, ns) / V - (N / (V * V)) * Vec::Constant(ns, dV).transpose();
    return Mat(w * Vec::Constant(ns, dV).transpose() + V * Jc * dc);
  };
  if (source) P.source = [source, ns](double t) { return checked_source(source, t, ns, "const_p_gas source"); };
  return P;
}

double const_p_gas_volume(double T, double Pr, const Vec& N) {
  if (!(T > 0) || !(Pr > 0)) throw std::invalid_argument("const_p_gas_volume: T and P must be positive");
  const double RT_P = kGasConstantExact * T / Pr;
  return N.sum() * RT_P;
}

Problem adiabatic_liquid_problem(const Mechanism& m, double rho_cp, SourceFn source) {
  const int ns = m.n_species(), nr = m.n_reactions();
  if (!(rho_cp > 0)) throw std::invalid_argument("adiabatic: rho c_p must be positive");
  Vec mdH(nr);
  for (int j = 0; j < nr; ++j) {
    const KineticReaction& r = m.reactions_si()[j];
    if (r.reversible && r.has_Kc && r.dH != 0.0)
      throw std::invalid_argument("adiabatic reactor: reaction " + std::to_string(j) +
                                  " is reversible through K_c with dH != 0; K_c is temperature-independent here, so "
                                  "the equilibrium would ignore the heat of reaction (van 't Hoff).  Give k_r its own "
                                  "Arrhenius parameters, or use an isothermal reactor");
    mdH[j] = -r.dH / rho_cp;
  }
  Problem P;
  P.n = ns + 1;
  P.f = [m, mdH, ns](double, const Vec& y) {
    Vec q;
    m.progress(y.head(ns), y[ns], q, nullptr, nullptr);
    Vec out(ns + 1);
    out.head(ns) = m.stoichiometry() * q;
    out[ns] = mdH.dot(q);
    return out;
  };
  P.jac = [m, mdH, ns](double, const Vec& y) {
    Vec q, dqT;
    Mat dqc;
    m.progress(y.head(ns), y[ns], q, &dqc, &dqT);
    Mat J(ns + 1, ns + 1);
    J.topLeftCorner(ns, ns) = m.stoichiometry() * dqc;
    J.col(ns).head(ns) = m.stoichiometry() * dqT;
    J.row(ns).head(ns) = mdH.transpose() * dqc;
    J(ns, ns) = mdH.dot(dqT);
    return J;
  };
  if (source) {
    const int n = P.n;
    P.source = [source, n](double t) { return checked_source(source, t, n, "adiabatic source"); };
  }
  return P;
}

Problem pfr_problem(const Mechanism& m, double T, double v0, double F_tot0) {
  const int ns = m.n_species();
  if (!(v0 > 0) || !(F_tot0 > 0)) throw std::invalid_argument("pfr: v0 and F_tot0 must be positive");
  const double dv = v0 / F_tot0;
  Problem P;
  P.n = ns;
  P.f = [m, T, v0, F_tot0](double, const Vec& F) {
    const double v = v0 * F.sum() / F_tot0;
    Vec w;
    m.rates(F / v, T, w, nullptr, nullptr);
    return w;
  };
  P.jac = [m, T, v0, F_tot0, dv, ns](double, const Vec& F) {
    const double v = v0 * F.sum() / F_tot0;
    Vec w;
    Mat Jc;
    m.rates(F / v, T, w, &Jc, nullptr);
    Mat dc = Mat::Identity(ns, ns) / v - (F / (v * v)) * Vec::Constant(ns, dv).transpose();
    return Mat(Jc * dc);
  };
  return P;
}

// ---------------------------------------------------------------------------------------------- equilibrium
EquilibriumSystem::EquilibriumSystem(const Mechanism& kinetic, std::vector<EquilibriumReaction> eq, double T,
                                     SourceFn source)
    : kin_(kinetic), eq_(std::move(eq)), T_(T) {
  const int ns = kin_.n_species(), ne = static_cast<int>(eq_.size());
  if (ne == 0) throw std::invalid_argument("EquilibriumSystem: no equilibrium reaction given");
  Mat Se = Mat::Zero(ns, ne);
  for (int j = 0; j < ne; ++j) {
    if (!(eq_[j].K > 0)) throw std::invalid_argument("EquilibriumSystem: K must be positive");
    for (size_t m = 0; m < eq_[j].reac.size(); ++m) Se(eq_[j].reac[m], j) -= eq_[j].nu_r[m];
    for (size_t m = 0; m < eq_[j].prod.size(); ++m) Se(eq_[j].prod[m], j) += eq_[j].nu_p[m];
  }
  W_ = left_null_rref(Se);
  if (W_.rows() + ne != ns)
    throw std::invalid_argument("EquilibriumSystem: the equilibrium reactions are linearly dependent");
  const Mechanism kin = kin_;
  const std::vector<EquilibriumReaction> eqv = eq_;
  const Mat W = W_;
  const int ni = static_cast<int>(W_.rows());
  P_.n = ns;
  P_.mass = Mat::Zero(ns, ns);
  P_.mass.topRows(ni) = W_;
  auto self = this;
  (void)self;
  auto gfun = [eqv](const Vec& c, Vec& g, Mat& G) {
    const int ne = static_cast<int>(eqv.size());
    g.resize(ne);
    G.setZero(ne, c.size());
    for (int j = 0; j < ne; ++j) {
      // g_j = prod c^{nu''} - K prod c^{nu'} (polynomial form, no division)
      double pb = 1, pf = 1;
      for (size_t m = 0; m < eqv[j].prod.size(); ++m) pb *= std::pow(c[eqv[j].prod[m]], eqv[j].nu_p[m]);
      for (size_t m = 0; m < eqv[j].reac.size(); ++m) pf *= std::pow(c[eqv[j].reac[m]], eqv[j].nu_r[m]);
      g[j] = pb - eqv[j].K * pf;
      for (size_t m = 0; m < eqv[j].prod.size(); ++m) {
        double d = eqv[j].nu_p[m] * std::pow(c[eqv[j].prod[m]], eqv[j].nu_p[m] - 1);
        for (size_t l = 0; l < eqv[j].prod.size(); ++l)
          if (l != m) d *= std::pow(c[eqv[j].prod[l]], eqv[j].nu_p[l]);
        G(j, eqv[j].prod[m]) += d;
      }
      for (size_t m = 0; m < eqv[j].reac.size(); ++m) {
        double d = eqv[j].nu_r[m] * std::pow(c[eqv[j].reac[m]], eqv[j].nu_r[m] - 1);
        for (size_t l = 0; l < eqv[j].reac.size(); ++l)
          if (l != m) d *= std::pow(c[eqv[j].reac[l]], eqv[j].nu_r[l]);
        G(j, eqv[j].reac[m]) -= eqv[j].K * d;
      }
    }
  };
  P_.f = [kin, T, W, gfun, ni, ns](double, const Vec& c) {
    Vec w, g;
    Mat G;
    kin.rates(c, T, w, nullptr, nullptr);
    gfun(c, g, G);
    Vec out(ns);
    out.head(ni) = W * w;
    out.tail(ns - ni) = g;
    return out;
  };
  P_.jac = [kin, T, W, gfun, ni, ns](double, const Vec& c) {
    Vec w, g;
    Mat Jk, G;
    kin.rates(c, T, w, &Jk, nullptr);
    gfun(c, g, G);
    Mat J(ns, ns);
    J.topRows(ni) = W * Jk;
    J.bottomRows(ns - ni) = G;
    return J;
  };
  if (source)
    P_.source = [source, W, ni, ns](double t) {
      Vec s = checked_source(source, t, ns, "equilibrium source");
      Vec out = Vec::Zero(ns);
      out.head(ni) = W * s;
      return out;
    };
}

void EquilibriumSystem::g_and_jac(const Vec& c, Vec& g, Mat& G) const {
  const int ne = static_cast<int>(eq_.size());
  g.resize(ne);
  G.setZero(ne, c.size());
  // same polynomial form as the problem's constraint rows
  Vec f = P_.f(0.0, c);
  Mat J = P_.jac(0.0, c);
  const int ni = static_cast<int>(W_.rows());
  g = f.tail(ne);
  G = J.bottomRows(ne);
  (void)ni;
}

Vec EquilibriumSystem::constraint(const Vec& c) const {
  Vec g;
  Mat G;
  g_and_jac(c, g, G);
  return g;
}

Vec EquilibriumSystem::consistent_init(const Vec& c0) const {
  const int ns = kin_.n_species(), ne = static_cast<int>(eq_.size()), ni = static_cast<int>(W_.rows());
  Mat Wk = W_;
  Vec c = c0;
  const Vec inv0 = Wk * c0;
  const bool nonneg = c0.minCoeff() >= 0.0;   // keep a non-negative start non-negative (damped Newton)
  bool converged = false;
  for (int it = 0; it < 200; ++it) {
    Vec g;
    Mat G;
    g_and_jac(c, g, G);
    Vec F(ns);
    F << Wk * c - inv0, g;
    Mat J(ns, ns);
    J << Wk, G;
    Vec d = J.partialPivLu().solve(-F);
    if (!d.allFinite()) break;
    double a = 1.0;
    if (nonneg)
      for (int i = 0; i < ns; ++i)
        if (d[i] < 0 && c[i] + d[i] < 0) a = std::min(a, 0.9 * c[i] / -d[i]);
    c += a * d;
    if (a == 1.0 &&
        d.lpNorm<Eigen::Infinity>() <= 4 * std::numeric_limits<double>::epsilon() * c.lpNorm<Eigen::Infinity>()) {
      converged = true;
      break;
    }
  }
  // the result must satisfy the equilibrium and keep the invariants, or the call fails
  Vec g;
  Mat G;
  g_and_jac(c, g, G);
  double worst = 0.0;
  for (int j = 0; j < ne; ++j)
    worst = std::max(worst, std::fabs(g[j]) / std::max((G.row(j).cwiseAbs() * c.cwiseAbs())(0, 0), 1e-300));
  const double inv_dev = (Wk * c - inv0).lpNorm<Eigen::Infinity>() /
                         std::max((Wk.cwiseAbs() * c0.cwiseAbs()).lpNorm<Eigen::Infinity>(), 1e-300);
  if (!c.allFinite() || !(worst <= 1e-10) || !(inv_dev <= 1e-12) || !converged)
    throw std::runtime_error(
        "EquilibriumSystem.consistent_init: no consistent state with the invariants W c0 of the given start (relative "
        "constraint residual " + std::to_string(worst) + ", invariant deviation " + std::to_string(inv_dev) +
        "); the start may lie outside the equilibrium manifold's reach (e.g. negative or unreachable invariants)");
  (void)ni;
  return c;
}

}  // namespace axf::chem
