// 1D porous electrode; see cpp/include/axionfem/chem/porous_electrode.hpp.
#include "axionfem/chem/porous_electrode.hpp"

#include <Eigen/LU>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "mfem.hpp"

namespace axf::chem {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("axf::chem::PorousElectrode1D: " + msg); }

// Lagrange shape functions on [0, 1], nodes 0, 1 (, 1/2)
void shape1d(int k, double xi, double* N, double* dN) {
  if (k == 1) {
    N[0] = 1 - xi, N[1] = xi;
    dN[0] = -1, dN[1] = 1;
  } else {
    N[0] = (1 - xi) * (1 - 2 * xi), N[1] = xi * (2 * xi - 1), N[2] = 4 * xi * (1 - xi);
    dN[0] = 4 * xi - 3, dN[1] = 4 * xi - 1, dN[2] = 4 - 8 * xi;
  }
}
}  // namespace

struct PorousElectrode1D::Impl {
  double l = 0, hh = 0;
  int N = 0, k = 1, deg = 0, nn = 0;
  double sigma = 0, kappa = 0, a_v = 0;
  Reaction rx;
  bool have_cond = false, have_ex = false, have_ctl = false, galv = true;
  double I = 0, V = 0;
  Eigen::VectorXd z;
  NewtonOptions opt;

  std::vector<int> conn(int e) const {
    std::vector<int> c{k * e, k * e + k};
    if (k == 2) c.push_back(k * e + 1);
    return c;
  }
  double node_x(int i) const { return l * i / (double)(k * N); }

  // A z - b (unconstrained) and its Jacobian; clamped flags exponent clipping
  void eval(const Eigen::VectorXd& zz, Eigen::VectorXd* R, Eigen::MatrixXd* J, Eigen::VectorXd* babs,
            bool* clamped) const {
    const double sg = sigma, kp = kappa;
    const double f = kFaraday / (kGasConstant * opt.temperature);   // F / (R T)
    const double s_solid = 1.0;
    const double s_el = -1.0;   // the exchange enters the electrolyte with -j
    const int n = 2 * nn;
    if (R) R->setZero(n);
    if (J) J->setZero(n, n);
    if (babs) babs->setZero(n);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, deg);
    const mfem::IntegrationRule& irx =
        mfem::IntRules.Get(mfem::Geometry::SEGMENT, deg);
    double Nq[3], dN[3];
    for (int e = 0; e < N; ++e) {
      const auto c = conn(e);
      const int nd = (int)c.size();
      // conduction
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const auto& ip = ir.IntPoint(q);
        shape1d(k, ip.x, Nq, dN);
        const double w = ip.weight * hh;
        double d1 = 0, d2 = 0;
        for (int a = 0; a < nd; ++a) d1 += dN[a] / hh * zz(c[a]), d2 += dN[a] / hh * zz(nn + c[a]);
        for (int a = 0; a < nd; ++a) {
          if (R) (*R)(c[a]) += sg * d1 * dN[a] / hh * w, (*R)(nn + c[a]) += kp * d2 * dN[a] / hh * w;
          if (J)
            for (int b = 0; b < nd; ++b) {
              (*J)(c[a], c[b]) += sg * dN[a] * dN[b] / (hh * hh) * w;
              (*J)(nn + c[a], nn + c[b]) += kp * dN[a] * dN[b] / (hh * hh) * w;
            }
        }
      }
      // exchange j = a_v i(phi1 - phi2 - E_eq)
      for (int q = 0; q < irx.GetNPoints(); ++q) {
        const auto& ip = irx.IntPoint(q);
        shape1d(k, ip.x, Nq, dN);
        const double w = ip.weight * hh;
        double p1 = 0, p2 = 0;
        for (int a = 0; a < nd; ++a) p1 += Nq[a] * zz(c[a]), p2 += Nq[a] * zz(nn + c[a]);
        double di = 0;
        const double i = reaction_current(rx, p1 - p2 - rx.E_eq, f, opt.clip, &di, clamped);
        const double j = a_v * i, dj = a_v * di;
        for (int a = 0; a < nd; ++a) {
          if (R) (*R)(c[a]) += s_solid * j * Nq[a] * w, (*R)(nn + c[a]) += s_el * j * Nq[a] * w;
          if (babs) (*babs)(c[a]) += std::abs(j) * Nq[a] * w, (*babs)(nn + c[a]) += std::abs(j) * Nq[a] * w;
          if (J)
            for (int b = 0; b < nd; ++b) {
              const double m = dj * Nq[a] * Nq[b] * w;
              (*J)(c[a], c[b]) += s_solid * m;
              (*J)(c[a], nn + c[b]) -= s_solid * m;
              (*J)(nn + c[a], c[b]) += s_el * m;
              (*J)(nn + c[a], nn + c[b]) -= s_el * m;
            }
        }
      }
    }
    // the cell current leaves the electrolyte through the separator: b(phi2(l)) = -I
    if (R && galv) (*R)(nn + (nn - 1)) += I;
  }
  std::vector<int> fixed() const {
    std::vector<int> f{0};
    if (!galv) f.push_back(2 * nn - 1);
    return f;
  }
};

PorousElectrode1D::PorousElectrode1D(double length, int n_elements, int order, int degree)
    : p_(new Impl), order_(order), degree_(degree) {
  if (order != 1 && order != 2) fail("order must be 1 or 2");
  if (!(length > 0) || n_elements < 1) fail("need length > 0 and n_elements >= 1");
  if (degree < 2 * order) fail("quadrature degree below 2 x order (the default rule is 2k + 4)");
  Impl& P = *p_;
  P.l = length, P.N = n_elements, P.k = order, P.deg = degree;
  P.hh = length / n_elements;
  P.nn = order * n_elements + 1;
  P.opt.temperature = 298.15;
}
PorousElectrode1D::~PorousElectrode1D() = default;

void PorousElectrode1D::set_conductivities(double sigma, double kappa) {
  if (!(sigma > 0 && kappa > 0)) fail("conductivities must be > 0");
  p_->sigma = sigma, p_->kappa = kappa, p_->have_cond = true;
}
void PorousElectrode1D::set_exchange(double a_v, const Reaction& r) {
  if (!(a_v > 0)) fail("a_v must be > 0 (m^-1)");
  if (!(r.i0 > 0)) fail("i0 must be > 0");
  // the porous electrode supports linearized Butler-Volmer exchange only (the verified case; nonlinear porous
  // exchange is not implemented): every other kinetics is refused, not run untested
  if (r.kind != Reaction::Kind::Linear)
    fail(std::string("exchange kinetics '") +
         (r.kind == Reaction::Kind::BV ? "bv" : r.kind == Reaction::Kind::TafelAnodic ? "tafel_a" : "tafel_c") +
         "' is not available: the porous electrode supports linearized Butler-Volmer "
         "exchange only; use kind "
         "'linear' (i = i0 (alpha_a + alpha_c) F/(R T) eta). Nonlinear porous exchange is not implemented");
  if (!(r.alpha_a >= 0 && r.alpha_c >= 0 && r.alpha_a + r.alpha_c > 0))
    fail("transfer coefficients must be >= 0 with a positive sum");
  p_->a_v = a_v, p_->rx = r, p_->have_ex = true;
}
void PorousElectrode1D::set_galvanostatic(double I) {
  if (!(I != 0) || !std::isfinite(I)) fail("galvanostatic current must be finite and != 0");
  p_->galv = true, p_->I = I, p_->have_ctl = true;
}
void PorousElectrode1D::set_potentiostatic(double V) {
  if (!std::isfinite(V)) fail("V must be finite");
  p_->galv = false, p_->V = V, p_->have_ctl = true;
}

SolveReport PorousElectrode1D::solve(const NewtonOptions& oin) {
  const auto t0 = std::chrono::steady_clock::now();
  Impl& P = *p_;
  if (!P.have_cond || !P.have_ex || !P.have_ctl)
    fail("set_conductivities, set_exchange and set_galvanostatic/set_potentiostatic before solve");
  P.opt = oin;
  if (P.opt.temperature == 0.0) P.opt.temperature = 298.15;
  const NewtonOptions& o = P.opt;
  const int n = 2 * P.nn;
  const auto fx = P.fixed();
  std::vector<char> isfix(n, 0);
  for (int i : fx) isfix[i] = 1;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
  if (!P.galv) z(2 * P.nn - 1) = -P.V;
  std::vector<int> fr;
  for (int i = 0; i < n; ++i)
    if (!isfix[i]) fr.push_back(i);
  auto relres = [&](const Eigen::VectorXd& zz, double* merit, bool* cl) {
    Eigen::VectorXd R, babs;
    Eigen::MatrixXd J;
    P.eval(zz, &R, &J, &babs, cl);
    // conduction + exchange operator: |J| |z| (the exchange is part of the operator), loads |b|, b_abs = |j|
    const Eigen::VectorXd kz = J.cwiseAbs() * zz.cwiseAbs();
    double num = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int i : fr) {
      num += R(i) * R(i), d1 += kz(i) * kz(i), d3 += babs(i) * babs(i);
      if (P.galv && i == 2 * P.nn - 1) d2 += P.I * P.I;
    }
    if (merit) *merit = std::sqrt(num);
    const double den = std::sqrt(d1) + std::sqrt(d2) + std::sqrt(d3);
    return den > 0 ? std::sqrt(num) / den : std::sqrt(num);
  };
  SolveReport Rp;
  Rp.unknowns = (int)fr.size();
  Rp.initial = "zero";
  bool cl = false;
  Rp.residual_history.push_back(relres(z, nullptr, &cl));
  const double lim = o.limiter_rt_f > 0 ? o.limiter_rt_f / (kFaraday / (kGasConstant * o.temperature)) : 0.0;
  Rp.stop = "max_iter";
  for (int it = 0;; ++it) {
    const auto& h = Rp.residual_history;
    if (h.back() == 0.0) { Rp.stop = "zero"; break; }
    if (h.size() >= 2 && h.back() <= o.stag && h.back() > 0.5 * h[h.size() - 2]) { Rp.stop = "stagnation"; break; }
    if (it >= o.max_iter) break;
    Eigen::VectorXd R;
    Eigen::MatrixXd J;
    bool c2 = false;
    P.eval(z, &R, &J, nullptr, &c2);
    const int m = (int)fr.size();
    Eigen::MatrixXd Jfree(m, m);
    Eigen::VectorXd rf(m);
    for (int a = 0; a < m; ++a) {
      rf(a) = -R(fr[a]);
      for (int b = 0; b < m; ++b) Jfree(a, b) = J(fr[a], fr[b]);
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(Jfree);
    Eigen::VectorXd d = lu.solve(rf);
    d += lu.solve(Eigen::VectorXd(rf - Jfree * d));   // one step of iterative refinement
    if (!d.allFinite()) { Rp.stop = "linear_solver_failed"; break; }
    Eigen::VectorXd dz = Eigen::VectorXd::Zero(n);
    for (int a = 0; a < m; ++a) dz(fr[a]) = d(a);
    // limiter on |d eta| = |d phi1 - d phi2| at the exchange quadrature points (the rule the exchange is assembled
    // with); for P2 the nodal values do not bound a quadratic between the nodes, so the points are evaluated
    double md = 0;
    {
      const mfem::IntegrationRule& irx =
          mfem::IntRules.Get(mfem::Geometry::SEGMENT, P.deg);
      double Nq[3], dN[3];
      for (int e = 0; e < P.N; ++e) {
        const auto c = P.conn(e);
        for (int q = 0; q < irx.GetNPoints(); ++q) {
          shape1d(P.k, irx.IntPoint(q).x, Nq, dN);
          double d = 0;
          for (size_t a = 0; a < c.size(); ++a) d += Nq[a] * (dz(c[a]) - dz(P.nn + c[a]));
          md = std::max(md, std::abs(d));
        }
      }
    }
    double t = 1.0, rn;
    Eigen::VectorXd zn;
    if (lim > 0 && md > lim) {
      t = lim / md;
      zn = z + t * dz;
      bool c3 = false;
      rn = relres(zn, nullptr, &c3);
      ++Rp.limited_steps;
    } else {
      double r0 = 0;
      bool c4 = false;
      relres(z, &r0, &c4);
      while (true) {
        zn = z + t * dz;
        double mn = 0;
        bool c5 = false;
        rn = relres(zn, &mn, &c5);
        if (mn < (1.0 - 1e-4 * t) * r0 || t <= o.ls_min) break;
        t *= 0.5;
        ++Rp.line_search_halvings;
      }
    }
    z = zn;
    Rp.step_lengths.push_back(t);
    Rp.residual_history.push_back(rn);
    Rp.iterations = it + 1;
  }
  P.z = z;
  bool cf = false;
  Rp.relative_residual = relres(z, nullptr, &cf);
  Rp.field_residual = Rp.relative_residual;
  Rp.clamp_active = cf;
  Rp.converged = std::isfinite(Rp.relative_residual) && Rp.relative_residual <= o.gate && !cf;
  Rp.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return Rp;
}

int PorousElectrode1D::num_elements() const { return p_->N; }
int PorousElectrode1D::num_dofs() const { return p_->nn; }
double PorousElectrode1D::h() const { return p_->hh; }
Eigen::VectorXd PorousElectrode1D::state() const {
  if (p_->z.size() == 0) fail("no state: call solve() first");
  return p_->z;
}
Eigen::VectorXd PorousElectrode1D::node_x() const {
  Eigen::VectorXd x(p_->nn);
  for (int i = 0; i < p_->nn; ++i) x(i) = p_->node_x(i);
  return x;
}
Eigen::MatrixXd PorousElectrode1D::quadrature(int degree) const {
  const Impl& P = *p_;
  const Eigen::VectorXd z = state();
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, degree);
  Eigen::MatrixXd Q(P.N * ir.GetNPoints(), 6);
  double Nq[3], dN[3];
  int r = 0;
  for (int e = 0; e < P.N; ++e) {
    const auto c = P.conn(e);
    for (int q = 0; q < ir.GetNPoints(); ++q, ++r) {
      const auto& ip = ir.IntPoint(q);
      shape1d(P.k, ip.x, Nq, dN);
      double p1 = 0, d1 = 0, p2 = 0, d2 = 0;
      for (size_t a = 0; a < c.size(); ++a) {
        p1 += Nq[a] * z(c[a]), d1 += dN[a] / P.hh * z(c[a]);
        p2 += Nq[a] * z(P.nn + c[a]), d2 += dN[a] / P.hh * z(P.nn + c[a]);
      }
      Q.row(r) << P.hh * (e + ip.x), ip.weight * P.hh, p1, d1, p2, d2;
    }
  }
  return Q;
}
Eigen::VectorXd PorousElectrode1D::residual_full() const {
  Eigen::VectorXd R;
  bool c = false;
  p_->eval(state(), &R, nullptr, nullptr, &c);
  return R;
}
int PorousElectrode1D::collector_row() const { return 0; }
int PorousElectrode1D::separator_row() const { return 2 * p_->nn - 1; }
double PorousElectrode1D::abs_action_sum() const {
  Eigen::VectorXd R;
  Eigen::MatrixXd J;
  bool c = false;
  const Eigen::VectorXd z = state();
  p_->eval(z, &R, &J, nullptr, &c);
  return (J.cwiseAbs() * z.cwiseAbs()).sum();
}

}  // namespace axf::chem
