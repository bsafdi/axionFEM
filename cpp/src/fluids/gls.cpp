// Stabilized P1/P1 GLS (SUPG + PSPG + LSIC), planar: see cpp/include/axionfem/fluids/gls.hpp.
#include "axionfem/fluids/gls.hpp"

#include <mpi.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>

#include "axionfem/core.hpp"

namespace axf::fluids {

namespace {

// ----------------------------------------------------------------------------------------------- the fixed rule
// Dunavant-type degree-4 six-point rule exactly as the literals of benchmarks/fluids/cases/f5_gls_check.py (_A1, _W1, _A2, _W2).
constexpr double A1 = 0.445948490915965, W1 = 0.223381589678011;
constexpr double A2 = 0.091576213509771, W2 = 0.109951743655322;
struct Rule {
  double pt[GLS_NQ_ALL][2];
  double wq[GLS_NQ_ALL];  // the check module's wq (sums to ~1/2); the centroid entry is the 1-point rule's weight 1/2
  double N[GLS_NQ_ALL][3];
  Rule() {
    const double a[2] = {A1, A2}, w[2] = {W1, W2};
    int k = 0;
    for (int s = 0; s < 2; ++s) {
      const double b = 1.0 - 2.0 * a[s];
      const double p[3][2] = {{a[s], a[s]}, {a[s], b}, {b, a[s]}};
      for (int t = 0; t < 3; ++t, ++k) {
        pt[k][0] = p[t][0];
        pt[k][1] = p[t][1];
        wq[k] = w[s] / 2;
      }
    }
    pt[6][0] = pt[6][1] = 1.0 / 3.0;
    wq[6] = 0.5;
    for (int q = 0; q < GLS_NQ_ALL; ++q) {
      N[q][0] = 1 - pt[q][0] - pt[q][1];
      N[q][1] = pt[q][0];
      N[q][2] = pt[q][1];
    }
  }
};
const Rule& rule() {
  static const Rule r;
  return r;
}

// ---------------------------------------------------------------------------------------------- forward-mode scalar
template <int N>
struct Dual {
  double v = 0;
  double d[N];
  Dual() {
    for (int k = 0; k < N; ++k) d[k] = 0;
  }
  Dual(double x) : v(x) {  // NOLINT: implicit constants
    for (int k = 0; k < N; ++k) d[k] = 0;
  }
};
inline double val(double x) { return x; }
template <int N>
inline double val(const Dual<N>& x) {
  return x.v;
}
template <int N>
inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v + b.v;
  for (int k = 0; k < N; ++k) r.d[k] = a.d[k] + b.d[k];
  return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v - b.v;
  for (int k = 0; k < N; ++k) r.d[k] = a.d[k] - b.d[k];
  return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a) {
  Dual<N> r;
  r.v = -a.v;
  for (int k = 0; k < N; ++k) r.d[k] = -a.d[k];
  return r;
}
template <int N>
inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v * b.v;
  for (int k = 0; k < N; ++k) r.d[k] = a.d[k] * b.v + a.v * b.d[k];
  return r;
}
template <int N>
inline Dual<N> operator*(const Dual<N>& a, double b) {
  Dual<N> r;
  r.v = a.v * b;
  for (int k = 0; k < N; ++k) r.d[k] = a.d[k] * b;
  return r;
}
template <int N>
inline Dual<N> operator*(double a, const Dual<N>& b) {
  return b * a;
}
template <int N>
inline Dual<N> operator+(const Dual<N>& a, double b) {
  Dual<N> r = a;
  r.v += b;
  return r;
}
template <int N>
inline Dual<N> operator+(double a, const Dual<N>& b) {
  return b + a;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a, double b) {
  Dual<N> r = a;
  r.v -= b;
  return r;
}
template <int N>
inline Dual<N> operator-(double a, const Dual<N>& b) {
  Dual<N> r = -b;
  r.v += a;
  return r;
}
template <int N>
inline Dual<N>& operator+=(Dual<N>& a, const Dual<N>& b) {
  a.v += b.v;
  for (int k = 0; k < N; ++k) a.d[k] += b.d[k];
  return a;
}
// 1 / sqrt(x)
inline double inv_sqrt(double x) { return 1.0 / std::sqrt(x); }
template <int N>
inline Dual<N> inv_sqrt(const Dual<N>& x) {
  Dual<N> r;
  r.v = 1.0 / std::sqrt(x.v);
  const double c = -0.5 * r.v / x.v;
  for (int k = 0; k < N; ++k) r.d[k] = c * x.d[k];
  return r;
}
template <int N>
inline Dual<N> recip(const Dual<N>& x) {
  Dual<N> r;
  r.v = 1.0 / x.v;
  const double c = -r.v * r.v;
  for (int k = 0; k < N; ++k) r.d[k] = c * x.d[k];
  return r;
}
inline double recip(double x) { return 1.0 / x; }

// ---------------------------------------------------------------------------------------------- geometry
struct Geo {
  double area = 0;
  double gradN[3][2];
  double Ji[2][2];  // d xi / d x
};

struct Eval {
  GlsMode mode;
  double scale;
  bool with_time;
  GlsStabilization stab;
};

constexpr double C_I_PINNED = 36.0, CG_PINNED = 1.0;

// tau_supg, tau_pspg, tau_c from the metric. w: advecting velocity at the point, mu: (scaled) viscosity.
template <class S>
void compute_tau(const Geo& g, const S* w, double mu, double rho, double dt, S& tm, S& tp, S& tc) {
  const double Mq[2][2] = {{1.0, 0.5}, {0.5, 1.0}};  // R^T R of the map from the unit RIGHT triangle to the EQUILATERAL one
  double G[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      double s = 0;
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l) s += g.Ji[k][i] * Mq[k][l] * g.Ji[l][j];
      G[i][j] = s;
    }
  const double trG = G[0][0] + G[1][1];
  const double GG = G[0][0] * G[0][0] + G[0][1] * G[0][1] + G[1][0] * G[1][0] + G[1][1] * G[1][1];
  const double nu = mu / rho;
  const double CI = C_I_PINNED;
  S wGw = w[0] * (G[0][0] * w[0] + G[0][1] * w[1]) + w[1] * (G[1][0] * w[0] + G[1][1] * w[1]);
  S steady = CG_PINNED * wGw + CI * nu * nu * GG;
  S tps = inv_sqrt(steady);  // steady tau: PSPG and LSIC (NO dt term)
  S ts = steady;
  if (dt > 0) {
    const double a = 2.0 / dt;
    ts = steady + a * a;
  }
  tm = inv_sqrt(ts);
  tc = recip(tps * trG);
  tp = tps;
}

}  // namespace

// ---------------------------------------------------------------------------------------------- the problem
struct GlsProblem::Impl {
  GlsData D;
  GlsOptions opt;
  int nv = 0, ne = 0;
  std::vector<Geo> geo;
  std::vector<char> isD;
  bool enclosed = false;           // every boundary vertex (a vertex of an edge used by one triangle) is Dirichlet
  int n_open_boundary_vertices = 0;
  std::vector<int> fu;  // free index of u dof (2 v + i), -1 if Dirichlet
  int nfu = 0;
  std::vector<double> lift;  // [nv*2] Dirichlet data, 0 elsewhere
  std::vector<double> cvec;  // int N_a
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> Mu_llt, Mp_llt;
  std::vector<int> free_vertices;  // vertices with free velocity (for the scalar mass block)
  std::vector<int> vfree_index;    // vertex -> index in free_vertices or -1
  std::vector<double> u, p;
  bool solved = false;

  // one element: residual R[9] = (R_u[a*2+i] for a = 0..2, R_p[a] at 6 + a); u[6], p[3] of type S
  template <class S>
  void element(int e, const S* ue, const S* pe, const Eval& ev, S* R) const {
    const Geo& g = geo[e];
    const Rule& rl = rule();
    const GlsOptions& o = opt;
    const double rho = o.rho;
    const int* t = &D.T[3 * e];
    S Gu[2][2] = {{S(0.0), S(0.0)}, {S(0.0), S(0.0)}}, gp[2] = {S(0.0), S(0.0)};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        S s(0.0);
        for (int a = 0; a < 3; ++a) s = s + ue[a * 2 + i] * g.gradN[a][j];
        Gu[i][j] = s;
      }
    for (int j = 0; j < 2; ++j) {
      S s(0.0);
      for (int a = 0; a < 3; ++a) s = s + pe[a] * g.gradN[a][j];
      gp[j] = s;
    }
    const S div = Gu[0][0] + Gu[1][1];
    for (int k = 0; k < 9; ++k) R[k] = S(0.0);
    const bool conv_on = ev.mode != GlsMode::Stokes;
    const bool nodal_mu = !D.mu_nodal.empty();
    double mun[3] = {0, 0, 0}, gmn[2] = {0, 0};
    if (nodal_mu) {
      for (int a = 0; a < 3; ++a) mun[a] = D.mu_nodal[t[a]] * ev.scale;
      for (int j = 0; j < 2; ++j)
        for (int a = 0; a < 3; ++a) gmn[j] += mun[a] * g.gradN[a][j];
    }
    const bool lap = o.form == GlsForm::Laplacian;
    const double dt = ev.with_time ? o.dt : 0.0;
    const bool use_supg = ev.stab.supg;
    const bool use_pspg = ev.stab.pspg;
    const bool use_lsic = ev.stab.lsic;
    for (int q = 0; q < GLS_NQ; ++q) {
      // weights: Galerkin and stabilization on the 6-point rule
      double wG, wS;
      wG = wS = g.area * 2 * rl.wq[q];
      const double* N = rl.N[q];
      S uq[2];
      S pq(0.0);
      for (int i = 0; i < 2; ++i) {
        S s(0.0);
        for (int a = 0; a < 3; ++a) s = s + ue[a * 2 + i] * N[a];
        uq[i] = s;
      }
      for (int a = 0; a < 3; ++a) pq = pq + pe[a] * N[a];
      double mu, gm[2];
      if (nodal_mu) {
        mu = 0;
        for (int a = 0; a < 3; ++a) mu += N[a] * mun[a];
        gm[0] = gmn[0];
        gm[1] = gmn[1];
      } else {
        const size_t qi = static_cast<size_t>(e) * GLS_NQ_ALL + q;
        mu = D.mu_q[qi] * ev.scale;
        gm[0] = D.gmu_q[2 * qi] * ev.scale;
        gm[1] = D.gmu_q[2 * qi + 1] * ev.scale;
      }
      const size_t qi = static_cast<size_t>(e) * GLS_NQ_ALL + q;
      const double f[2] = {D.f_q.empty() ? 0.0 : D.f_q[2 * qi], D.f_q.empty() ? 0.0 : D.f_q[2 * qi + 1]};
      S w[2] = {S(0.0), S(0.0)};
      if (ev.mode == GlsMode::NavierStokes) {
        w[0] = uq[0];
        w[1] = uq[1];
      } else if (ev.mode == GlsMode::Oseen) {
        w[0] = S(D.adv_q[2 * qi]);
        w[1] = S(D.adv_q[2 * qi + 1]);
      }  // Stokes: w = 0
      S conv[2] = {S(0.0), S(0.0)}, ut[2] = {S(0.0), S(0.0)}, sdiv[2] = {S(0.0), S(0.0)}, Rm[2] = {S(0.0), S(0.0)};
      for (int i = 0; i < 2; ++i) conv[i] = conv_on ? Gu[i][0] * w[0] + Gu[i][1] * w[1] : S(0.0);
      for (int i = 0; i < 2; ++i) {
        if (dt <= 0) {
          ut[i] = S(0.0);
        } else if (o.bdf == 1) {
          S u1 = S(0.0);
          for (int a = 0; a < 3; ++a) u1 = u1 + D.hist1[2 * t[a] + i] * N[a];
          ut[i] = (uq[i] - u1) * (1.0 / dt);
        } else {
          S u1 = S(0.0), u2 = S(0.0);
          for (int a = 0; a < 3; ++a) {
            u1 = u1 + D.hist1[2 * t[a] + i] * N[a];
            u2 = u2 + D.hist2[2 * t[a] + i] * N[a];
          }
          ut[i] = (3.0 * uq[i] - 4.0 * u1 + u2) * (1.0 / (2.0 * dt));
        }
      }
      for (int i = 0; i < 2; ++i) {
        S s(0.0);
        for (int j = 0; j < 2; ++j) s = s + (lap ? Gu[i][j] : Gu[i][j] + Gu[j][i]) * gm[j];
        sdiv[i] = s;
      }
      for (int i = 0; i < 2; ++i) {
        S r = rho * ut[i] + rho * conv[i];
        r = r + gp[i];
        r = r - sdiv[i];
        r = r - f[i];
        Rm[i] = r;
      }
      S tm(0.0), tp(0.0), tc(0.0);
      const bool need_tau = use_supg || use_pspg || use_lsic;
      if (need_tau) compute_tau<S>(g, w, mu, rho, dt, tm, tp, tc);
      const double fr_s = rho, fr_p = 1.0 / rho, fr_l = rho;
      S wdotgrad[3];
      for (int a = 0; a < 3; ++a) wdotgrad[a] = w[0] * g.gradN[a][0] + w[1] * g.gradN[a][1];
      for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
          if (wG != 0.0) {
            S visc(0.0);
            for (int j = 0; j < 2; ++j) visc = visc + (lap ? Gu[i][j] : Gu[i][j] + Gu[j][i]) * g.gradN[a][j];
            S gal = (rho * (ut[i] + conv[i]) - f[i]) * N[a] + mu * visc - pq * g.gradN[a][i];
            R[a * 2 + i] = R[a * 2 + i] + gal * wG;
          }
          if (wS != 0.0) {
            S st(0.0);
            if (use_supg) st = st + tm * fr_s * wdotgrad[a] * Rm[i];
            if (use_lsic) st = st + tc * fr_l * div * g.gradN[a][i];
            R[a * 2 + i] = R[a * 2 + i] + st * wS;
          }
        }
        if (wG != 0.0) R[6 + a] = R[6 + a] + div * (N[a] * wG);
        if (wS != 0.0 && use_pspg)
          R[6 + a] = R[6 + a] + tp * fr_p * (g.gradN[a][0] * Rm[0] + g.gradN[a][1] * Rm[1]) * wS;
      }
    }
  }

  // ---- assembly of the full residual (no elimination), scale s, mode
  void residual_all(const std::vector<double>& uu, const std::vector<double>& pp, double s, GlsMode mode,
                    bool with_time, const GlsStabilization& stab, std::vector<double>& Ru,
                    std::vector<double>& Rp) const {
    Ru.assign(2 * nv, 0.0);
    Rp.assign(nv, 0.0);
    Eval ev{mode, s, with_time, stab};
    for (int e = 0; e < ne; ++e) {
      const int* t = &D.T[3 * e];
      double ue[6], pe[3], R[9];
      for (int a = 0; a < 3; ++a) {
        ue[2 * a] = uu[2 * t[a]];
        ue[2 * a + 1] = uu[2 * t[a] + 1];
        pe[a] = pp[t[a]];
      }
      element<double>(e, ue, pe, ev, R);
      for (int a = 0; a < 3; ++a) {
        Ru[2 * t[a]] += R[2 * a];
        Ru[2 * t[a] + 1] += R[2 * a + 1];
        Rp[t[a]] += R[6 + a];
      }
    }
  }

  // ---- directional derivative J d of the full residual (state (uu, pp), direction (du, dp)), same element routine
  void jvp_all(const std::vector<double>& uu, const std::vector<double>& pp, const std::vector<double>& du,
               const std::vector<double>& dp, double s, GlsMode mode, bool with_time, const GlsStabilization& stab,
               std::vector<double>& Ju, std::vector<double>& Jp) const {
    Ju.assign(2 * nv, 0.0);
    Jp.assign(nv, 0.0);
    Eval ev{mode, s, with_time, stab};
    for (int e = 0; e < ne; ++e) {
      const int* t = &D.T[3 * e];
      Dual<1> ue[6], pe[3], R[9];
      for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
          ue[2 * a + i].v = uu[2 * t[a] + i];
          ue[2 * a + i].d[0] = du[2 * t[a] + i];
        }
        pe[a].v = pp[t[a]];
        pe[a].d[0] = dp[t[a]];
      }
      element<Dual<1>>(e, ue, pe, ev, R);
      for (int a = 0; a < 3; ++a) {
        Ju[2 * t[a]] += R[2 * a].d[0];
        Ju[2 * t[a] + 1] += R[2 * a + 1].d[0];
        Jp[t[a]] += R[6 + a].d[0];
      }
    }
  }

  // ---- the Jacobian of the solved system (free u, all p, multiplier): triplets
  Eigen::SparseMatrix<double, Eigen::RowMajor> jacobian(const std::vector<double>& uu, const std::vector<double>& pp,
                                                         double s, GlsMode mode, bool with_time,
                                                         const GlsStabilization& stab) const {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> tr;
    tr.reserve(static_cast<size_t>(ne) * 81 + 2 * nv);
    Eval ev{mode, s, with_time, stab};
    const int n = nfu + nv + 1;
    for (int e = 0; e < ne; ++e) {
      const int* t = &D.T[3 * e];
      Dual<9> ue[6], pe[3], R[9];
      int col[9];
      for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
          const int k = 2 * a + i;
          ue[k].v = uu[2 * t[a] + i];
          ue[k].d[k] = 1.0;
          col[k] = fu[2 * t[a] + i];
        }
        pe[a].v = pp[t[a]];
        pe[a].d[6 + a] = 1.0;
        col[6 + a] = nfu + t[a];
      }
      element<Dual<9>>(e, ue, pe, ev, R);
      for (int r = 0; r < 9; ++r) {
        // row index: R_u rows 2a+i of vertex t[a]; R_p rows 6 + a
        const int row = (r < 6) ? fu[2 * t[r / 2] + (r % 2)] : nfu + t[r - 6];
        if (row < 0) continue;
        for (int k = 0; k < 9; ++k)
          if (col[k] >= 0 && R[r].d[k] != 0.0) tr.emplace_back(row, col[k], R[r].d[k]);
      }
    }
    for (int a = 0; a < nv; ++a) {
      tr.emplace_back(nfu + a, nfu + nv, cvec[a]);
      tr.emplace_back(nfu + nv, nfu + a, cvec[a]);
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    A.setFromTriplets(tr.begin(), tr.end());
    return A;
  }

  // ---- dual norm (consistent P1 mass matrices): R_u over free vertices (both components), R_p over all vertices
  double dual(const std::vector<double>& Ru, const std::vector<double>& Rp_eff) const {
    double s = 0;
    const int nf = static_cast<int>(free_vertices.size());
    if (nf > 0) {
      Eigen::VectorXd b(nf);
      for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < nf; ++k) b(k) = Ru[2 * free_vertices[k] + c];
        Eigen::VectorXd x = Mu_llt.solve(b);
        s += b.dot(x);
      }
    }
    Eigen::VectorXd b(nv);
    for (int a = 0; a < nv; ++a) b(a) = Rp_eff[a];
    Eigen::VectorXd x = Mp_llt.solve(b);
    s += b.dot(x);
    return std::sqrt(std::max(s, 0.0));
  }
};

const double* gls_rule_points() {
  static double p[GLS_NQ * 2];
  const Rule& r = rule();
  for (int q = 0; q < GLS_NQ; ++q) {
    p[2 * q] = r.pt[q][0];
    p[2 * q + 1] = r.pt[q][1];
  }
  return p;
}
const double* gls_rule_weights() {
  static double w[GLS_NQ];
  const Rule& r = rule();
  for (int q = 0; q < GLS_NQ; ++q) w[q] = r.wq[q];
  return w;
}

GlsProblem::GlsProblem(GlsData data, GlsOptions opt) : impl_(new Impl), opt_(std::move(opt)) {
  Impl& I = *impl_;
  I.D = std::move(data);
  I.opt = opt_;
  const GlsData& D = I.D;
  AXF_REQUIRE(D.V.size() % 2 == 0 && D.T.size() % 3 == 0, "GLS mesh arrays have the wrong shape", "mesh",
              "Pass vertices (nv,2) and triangles (ne,3).");
  I.nv = static_cast<int>(D.V.size() / 2);
  I.ne = static_cast<int>(D.T.size() / 3);
  AXF_REQUIRE(opt_.rho > 0, "rho must be positive", "rho", "Constant density > 0.");
  AXF_REQUIRE(opt_.dt >= 0, "dt must be >= 0", "dt", "0 = steady.");
  AXF_REQUIRE(opt_.bdf == 1 || opt_.bdf == 2, "bdf must be 1 or 2", "bdf", "BDF1 or BDF2 (constant step).");
  const size_t nqa = static_cast<size_t>(I.ne) * GLS_NQ_ALL;
  AXF_REQUIRE(!D.mu_nodal.empty() ? D.mu_nodal.size() == static_cast<size_t>(I.nv)
                                  : (D.mu_q.size() == nqa && D.gmu_q.size() == 2 * nqa),
              "viscosity data: give mu_q [ne,7] + gmu_q [ne,7,2] or mu_nodal [nv]", "mu", "See axionfem.fluids.gls.");
  AXF_REQUIRE(D.f_q.empty() || D.f_q.size() == 2 * nqa, "force array must be [ne,7,2]", "force", "");
  AXF_REQUIRE(opt_.mode != GlsMode::Oseen || D.adv_q.size() == 2 * nqa, "Oseen mode needs adv_q [ne,7,2]", "advect",
              "");
  AXF_REQUIRE(opt_.dt == 0 || D.hist1.size() == static_cast<size_t>(2 * I.nv), "a transient step needs hist1 [nv,2]",
              "history", "");
  AXF_REQUIRE(opt_.dt == 0 || opt_.bdf == 1 || D.hist2.size() == static_cast<size_t>(2 * I.nv),
              "BDF2 needs hist2 [nv,2]", "history", "");
  I.geo.resize(I.ne);
  const double refgrad[3][2] = {{-1, -1}, {1, 0}, {0, 1}};
  for (int e = 0; e < I.ne; ++e) {
    const int* t = &D.T[3 * e];
    for (int a = 0; a < 3; ++a)
      AXF_REQUIRE(t[a] >= 0 && t[a] < I.nv, "triangle vertex out of range", "mesh", "");
    const double* P0 = &D.V[2 * t[0]];
    const double* P1 = &D.V[2 * t[1]];
    const double* P2 = &D.V[2 * t[2]];
    const double J00 = P1[0] - P0[0], J10 = P1[1] - P0[1], J01 = P2[0] - P0[0], J11 = P2[1] - P0[1];
    const double det = J00 * J11 - J01 * J10;
    AXF_REQUIRE(det != 0.0, "degenerate triangle", "mesh", "");
    Geo& g = I.geo[e];
    g.area = 0.5 * std::abs(det);
    g.Ji[0][0] = J11 / det;
    g.Ji[0][1] = -J01 / det;
    g.Ji[1][0] = -J10 / det;
    g.Ji[1][1] = J00 / det;
    for (int a = 0; a < 3; ++a)
      for (int j = 0; j < 2; ++j) g.gradN[a][j] = refgrad[a][0] * g.Ji[0][j] + refgrad[a][1] * g.Ji[1][j];
  }
  // Dirichlet bookkeeping
  I.isD.assign(I.nv, 0);
  I.lift.assign(2 * I.nv, 0.0);
  AXF_REQUIRE(D.dirichlet_values.size() == 2 * D.dirichlet_nodes.size(), "dirichlet_values must be [n,2]",
              "dirichlet", "");
  for (size_t k = 0; k < D.dirichlet_nodes.size(); ++k) {
    const int v = D.dirichlet_nodes[k];
    AXF_REQUIRE(v >= 0 && v < I.nv, "Dirichlet vertex out of range", "dirichlet", "");
    I.isD[v] = 1;
    I.lift[2 * v] = D.dirichlet_values[2 * k];
    I.lift[2 * v + 1] = D.dirichlet_values[2 * k + 1];
  }
  // Enclosed flow? The pressure then has exactly the constant nullspace the mean-value multiplier removes. With any
  // non-Dirichlet boundary vertex the pressure is determined (do-nothing outlets etc.), the multiplier would be a wrong
  // extra constraint: solve() refuses, nothing is mis-solved.
  {
    std::map<std::pair<int, int>, int> edge_count;
    for (int e = 0; e < I.ne; ++e)
      for (int a = 0; a < 3; ++a) {
        int v0 = D.T[3 * e + a], v1 = D.T[3 * e + (a + 1) % 3];
        if (v0 > v1) std::swap(v0, v1);
        ++edge_count[{v0, v1}];
      }
    std::vector<char> onb(I.nv, 0);
    for (const auto& kv : edge_count)
      if (kv.second == 1) onb[kv.first.first] = onb[kv.first.second] = 1;
    I.n_open_boundary_vertices = 0;
    for (int v = 0; v < I.nv; ++v)
      if (onb[v] && !I.isD[v]) ++I.n_open_boundary_vertices;
    I.enclosed = I.n_open_boundary_vertices == 0;
    diag_.nullspace_detected = I.enclosed;
  }
  I.fu.assign(2 * I.nv, -1);
  I.vfree_index.assign(I.nv, -1);
  I.nfu = 0;
  for (int v = 0; v < I.nv; ++v)
    if (!I.isD[v]) {
      I.vfree_index[v] = static_cast<int>(I.free_vertices.size());
      I.free_vertices.push_back(v);
      I.fu[2 * v] = I.nfu++;
      I.fu[2 * v + 1] = I.nfu++;
    }
  // c = int N_a, consistent mass matrices
  I.cvec.assign(I.nv, 0.0);
  using Trip = Eigen::Triplet<double>;
  std::vector<Trip> tp, tu;
  for (int e = 0; e < I.ne; ++e) {
    const int* t = &D.T[3 * e];
    for (int a = 0; a < 3; ++a) {
      I.cvec[t[a]] += I.geo[e].area / 3.0;
      for (int b = 0; b < 3; ++b) {
        const double m = I.geo[e].area / 12.0 * (a == b ? 2.0 : 1.0);
        tp.emplace_back(t[a], t[b], m);
        if (I.vfree_index[t[a]] >= 0 && I.vfree_index[t[b]] >= 0)
          tu.emplace_back(I.vfree_index[t[a]], I.vfree_index[t[b]], m);
      }
    }
  }
  Eigen::SparseMatrix<double> Mp(I.nv, I.nv);
  Mp.setFromTriplets(tp.begin(), tp.end());
  I.Mp_llt.compute(Mp);
  const int nf = static_cast<int>(I.free_vertices.size());
  if (nf > 0) {
    Eigen::SparseMatrix<double> Mu(nf, nf);
    Mu.setFromTriplets(tu.begin(), tu.end());
    I.Mu_llt.compute(Mu);
  }
  I.u = I.lift;
  I.p.assign(I.nv, 0.0);
  diag_.n_free_u = I.nfu;
  diag_.n_pressure = I.nv;
  diag_.n_total = I.nfu + I.nv;
}

GlsProblem::~GlsProblem() = default;
int GlsProblem::n_vertices() const { return impl_->nv; }
int GlsProblem::n_elements() const { return impl_->ne; }
const std::vector<int>& GlsProblem::dirichlet_nodes() const { return impl_->D.dirichlet_nodes; }
const std::vector<double>& GlsProblem::u() const {
  AXF_REQUIRE(impl_->solved, "no solution: solve() has not been called", "solve", "");
  return impl_->u;
}
const std::vector<double>& GlsProblem::p() const {
  AXF_REQUIRE(impl_->solved, "no solution: solve() has not been called", "solve", "");
  return impl_->p;
}

void GlsProblem::residual_full(const std::vector<double>& u, const std::vector<double>& p, std::vector<double>& Ru,
                               std::vector<double>& Rp, double s) const {
  const Impl& I = *impl_;
  AXF_REQUIRE(u.size() == static_cast<size_t>(2 * I.nv) && p.size() == static_cast<size_t>(I.nv),
              "state must be u [nv,2], p [nv]", "state", "");
  I.residual_all(u, p, s, opt_.mode, opt_.dt > 0, opt_.stab, Ru, Rp);
}

std::pair<long, long> GlsProblem::layout() const { return {impl_->nfu, impl_->nv}; }

std::vector<double> GlsProblem::state() const {
  const Impl& I = *impl_;
  std::vector<double> x(I.nfu + I.nv);
  for (int k = 0; k < 2 * I.nv; ++k)
    if (I.fu[k] >= 0) x[I.fu[k]] = I.u[k];
  for (int a = 0; a < I.nv; ++a) x[I.nfu + a] = I.p[a];
  return x;
}

std::vector<double> GlsProblem::residual(const std::vector<double>& x, double s) const {
  const Impl& I = *impl_;
  AXF_REQUIRE(x.size() == static_cast<size_t>(I.nfu + I.nv), "x must have layout() entries", "state", "");
  std::vector<double> u = I.lift, p(I.nv);
  for (int k = 0; k < 2 * I.nv; ++k)
    if (I.fu[k] >= 0) u[k] = x[I.fu[k]];
  for (int a = 0; a < I.nv; ++a) p[a] = x[I.nfu + a];
  std::vector<double> Ru, Rp;
  I.residual_all(u, p, s, opt_.mode, opt_.dt > 0, opt_.stab, Ru, Rp);
  std::vector<double> r(I.nfu + I.nv);
  for (int k = 0; k < 2 * I.nv; ++k)
    if (I.fu[k] >= 0) r[I.fu[k]] = Ru[k];
  for (int a = 0; a < I.nv; ++a) r[I.nfu + a] = Rp[a];
  return r;
}

std::vector<double> GlsProblem::jacobian_action(const std::vector<double>& x, const std::vector<double>& d,
                                                double s) const {
  const Impl& I = *impl_;
  AXF_REQUIRE(x.size() == static_cast<size_t>(I.nfu + I.nv) && d.size() == x.size(), "x and d must have layout() entries",
              "state", "");
  std::vector<double> u = I.lift, p(I.nv), du(2 * I.nv, 0.0), dp(I.nv);
  for (int k = 0; k < 2 * I.nv; ++k)
    if (I.fu[k] >= 0) {
      u[k] = x[I.fu[k]];
      du[k] = d[I.fu[k]];
    }
  for (int a = 0; a < I.nv; ++a) {
    p[a] = x[I.nfu + a];
    dp[a] = d[I.nfu + a];
  }
  std::vector<double> Ju, Jp;
  I.jvp_all(u, p, du, dp, s, opt_.mode, opt_.dt > 0, opt_.stab, Ju, Jp);
  std::vector<double> r(I.nfu + I.nv);
  for (int k = 0; k < 2 * I.nv; ++k)
    if (I.fu[k] >= 0) r[I.fu[k]] = Ju[k];
  for (int a = 0; a < I.nv; ++a) r[I.nfu + a] = Jp[a];
  return r;
}

void GlsProblem::tau_at(const std::vector<double>& w_nodal, std::vector<double>& ts, std::vector<double>& tp,
                        std::vector<double>& tc) const {
  const Impl& I = *impl_;
  AXF_REQUIRE(w_nodal.size() == static_cast<size_t>(2 * I.nv), "w must be [nv,2]", "w", "");
  const Rule& rl = rule();
  ts.assign(static_cast<size_t>(I.ne) * GLS_NQ, 0.0);
  tp = ts;
  tc = ts;
  const bool nodal_mu = !I.D.mu_nodal.empty();
  const double dt = opt_.dt;
  for (int e = 0; e < I.ne; ++e) {
    const int* t = &I.D.T[3 * e];
    for (int q = 0; q < GLS_NQ; ++q) {
      double w[2] = {0, 0}, mu;
      for (int a = 0; a < 3; ++a) {
        w[0] += rl.N[q][a] * w_nodal[2 * t[a]];
        w[1] += rl.N[q][a] * w_nodal[2 * t[a] + 1];
      }
      if (nodal_mu) {
        mu = 0;
        for (int a = 0; a < 3; ++a) mu += rl.N[q][a] * I.D.mu_nodal[t[a]];
      } else {
        mu = I.D.mu_q[static_cast<size_t>(e) * GLS_NQ_ALL + q];
      }
      double a, b, c;
      compute_tau<double>(I.geo[e], w, mu, opt_.rho, dt, a, b, c);
      const size_t k = static_cast<size_t>(e) * GLS_NQ + q;
      ts[k] = a;
      tp[k] = b;
      tc[k] = c;
    }
  }
}

double GlsProblem::dual_norm(const std::vector<double>& Ru, const std::vector<double>& Rp) const {
  return impl_->dual(Ru, Rp);
}

// ---------------------------------------------------------------------------------------------- the nonlinear driver
const GlsDiagnostics& GlsProblem::solve() {
  Impl& I = *impl_;
  const GlsNonlinear& nl = opt_.nl;
  AXF_REQUIRE(I.enclosed,
              "open/natural boundaries are not offered by the GLS solver: " + std::to_string(I.n_open_boundary_vertices) +
                  " boundary vertices are not Dirichlet (enclosed flows only; the mean-value multiplier would be a wrong "
                  "constraint on an open-boundary problem)",
              "dirichlet", "Prescribe the velocity on every boundary vertex; do-nothing/symmetry/traction boundaries are not implemented.");
  const double t0 = MPI_Wtime();
  diag_.history.clear();
  diag_.converged = false;
  diag_.iterations = 0;
  diag_.line_search_min_step_accepts = 0;
  diag_.failure.clear();
  diag_.lu_backend = to_string(opt_.lu);
  std::vector<double> scales = nl.viscosity_scales;
  if (scales.empty()) scales.push_back(1.0);
  diag_.viscosity_scales = scales;
  const bool with_time = opt_.dt > 0;
  const int nfu = I.nfu, nv = I.nv;
  const int n = nfu + nv + 1;

  std::vector<double> u = I.lift, p(nv, 0.0);
  double lam_mult = 0.0;
  std::vector<double> Ru, Rp, Tu, Tp;

  auto merit = [&](const std::vector<double>& ru, const std::vector<double>& rp, double lm) {
    std::vector<double> re(nv);
    for (int a = 0; a < nv; ++a) re[a] = rp[a] + lm * I.cvec[a];
    return I.dual(ru, re);
  };
  auto residuals = [&](const std::vector<double>& uu, const std::vector<double>& pp, double s, GlsMode mode,
                       std::vector<double>& ru, std::vector<double>& rp) {
    I.residual_all(uu, pp, s, mode, with_time, opt_.stab, ru, rp);
  };

  double last_lu_res = -1, last_lu_before = -1;  // LU accuracy of the most recent linear solve (per-step record)
  // one linear solve with the Jacobian of mode/scale at the current state; returns du, dp, dlam of -R
  auto lin_solve = [&](double s, GlsMode mode, std::vector<double>& du, std::vector<double>& dp, double& dlam,
                       int& refine) -> bool {
    auto A = I.jacobian(u, p, s, mode, with_time, opt_.stab);
    std::vector<double> b(n, 0.0);
    for (int k = 0; k < 2 * nv; ++k)
      if (I.fu[k] >= 0) b[I.fu[k]] = -Ru[k];
    double cp = 0;
    for (int a = 0; a < nv; ++a) {
      b[nfu + a] = -(Rp[a] + lam_mult * I.cvec[a]);
      cp += I.cvec[a] * p[a];
    }
    b[nfu + nv] = -cp;
    try {
      DistCsr D = DistCsr::from_serial(CsrMatrix::from_eigen(A), MPI_COMM_SELF);
      SparseLU lu(opt_.lu);
      lu.set_refinement(10, std::min(1e-12, 1e-2 * nl.newton_rtol));
      lu.set_throw_on_miss(false);
      lu.set_threads(opt_.lu_threads);
      lu.set_refine_to_stagnation(true);
      lu.factorize(D);
      std::vector<double> x = lu.solve(b);
      const LUStats& st = lu.stats();
      last_lu_res = st.rel_residual;
      last_lu_before = st.rel_residual_before;
      diag_.lu_residual = st.rel_residual;
      diag_.lu_residual_before = st.rel_residual_before;
      diag_.lu_refine_steps = st.refine_steps;
      refine = st.refine_steps;
      du.assign(2 * nv, 0.0);
      dp.assign(nv, 0.0);
      for (int k = 0; k < 2 * nv; ++k)
        if (I.fu[k] >= 0) du[k] = x[I.fu[k]];
      for (int a = 0; a < nv; ++a) dp[a] = x[nfu + a];
      dlam = x[nfu + nv];
      for (double v : x)
        if (!std::isfinite(v)) {
          diag_.failure = "linear solve returned a non-finite solution";
          return false;
        }
    } catch (const std::exception& e) {
      diag_.failure = std::string("linear solve failed: ") + e.what();
      return false;
    }
    return true;
  };

  auto push = [&](const char* kind, double r, double lam, int stage, double s, int refine) {
    GlsStep h;
    h.kind = kind;
    h.residual = r;
    h.step = lam;
    h.stage = stage;
    h.viscosity_scale = s;
    h.lu_refine = refine;
    h.lu_residual = last_lu_res;
    h.lu_residual_before = last_lu_before;
    diag_.history.push_back(h);
  };
  auto fail = [&](const std::string& why) {
    if (diag_.failure.empty()) diag_.failure = why;
    return false;
  };

  auto lift_norm = [&](double s) {
    std::vector<double> a, b, zp(nv, 0.0);
    residuals(I.lift, zp, s, opt_.mode, a, b);
    return merit(a, b, 0.0);
  };

  std::vector<double> du, dp;
  double dlam = 0;
  auto newton_step = [&](double s, double den, int stage, double& r) -> bool {
    residuals(u, p, s, opt_.mode, Ru, Rp);
    const double m0 = merit(Ru, Rp, lam_mult);
    int refine = 0;
    if (!lin_solve(s, opt_.mode, du, dp, dlam, refine)) return fail("linear solve failed in a Newton step");
    double lam = 1.0;
    std::vector<double> tu, tpv;
    double tl = 0;
    if (nl.line_search) {
      bool ok = false;
      for (int h = 0; h <= nl.line_search_halvings; ++h) {
        lam = std::ldexp(1.0, -h);
        tu = u;
        tpv = p;
        for (int k = 0; k < 2 * nv; ++k) tu[k] += lam * du[k];
        for (int a = 0; a < nv; ++a) tpv[a] += lam * dp[a];
        tl = lam_mult + lam * dlam;
        residuals(tu, tpv, s, opt_.mode, Tu, Tp);
        const double m = merit(Tu, Tp, tl);
        if (std::isfinite(m) && m <= (1.0 - nl.armijo_c1 * lam) * m0) {
          ok = true;
          break;
        }
      }
      if (!ok) ++diag_.line_search_min_step_accepts;
      u = tu;
      p = tpv;
      lam_mult = tl;
      Ru = Tu;
      Rp = Tp;
    } else {
      for (int k = 0; k < 2 * nv; ++k) u[k] += du[k];
      for (int a = 0; a < nv; ++a) p[a] += dp[a];
      lam_mult += dlam;
      residuals(u, p, s, opt_.mode, Ru, Rp);
    }
    r = merit(Ru, Rp, lam_mult) / den;
    push("newton", r, lam, stage, s, refine);
    ++diag_.iterations;
    if (!std::isfinite(r)) return fail("the residual is not finite (divergence)");
    return true;
  };

  double r = 0, den = 0;
  bool ok = true;
  const double s0 = scales.front();
  den = lift_norm(s0);
  diag_.residual_lift = den;
  if (!std::isfinite(den)) {
    ok = fail("the residual of the lift is not finite");
  } else if (den == 0.0) {
    diag_.relative_residual = 0.0;
    diag_.converged = true;
  } else {
    // ---- the TRUE Stokes start: the linear Stokes problem (w = 0, no convective term; the time term is kept)
    residuals(u, p, s0, GlsMode::Stokes, Ru, Rp);
    {
      int refine = 0;
      if (!lin_solve(s0, GlsMode::Stokes, du, dp, dlam, refine)) {
        ok = fail("linear solve failed in the Stokes step");
      } else {
        for (int k = 0; k < 2 * nv; ++k) u[k] += du[k];
        for (int a = 0; a < nv; ++a) p[a] += dp[a];
        lam_mult += dlam;
        residuals(u, p, s0, opt_.mode, Ru, Rp);
        r = merit(Ru, Rp, lam_mult) / den;
        push("stokes", r, 1.0, 0, s0, refine);
        if (!std::isfinite(r)) ok = fail("the residual is not finite after the Stokes step");
      }
    }
    if (ok) {
      for (size_t j = 0; ok && j < scales.size(); ++j) {
        const double s = scales[j];
        if (j > 0) {
          den = lift_norm(s);
          diag_.residual_lift = den;
          residuals(u, p, s, opt_.mode, Ru, Rp);
          r = merit(Ru, Rp, lam_mult) / den;
        }
        int k = 0;
        while (ok && r > nl.newton_rtol && k < nl.newton_maxit) {
          ok = newton_step(s, den, static_cast<int>(j), r);
          ++k;
        }
        if (ok && r > nl.newton_rtol) {
          std::ostringstream os;
          os << "Newton budget exhausted in continuation stage " << j + 1 << " of " << scales.size()
             << " (viscosity scale " << s << ")";
          ok = fail(os.str());
        }
      }
    }
    diag_.relative_residual = r;
    diag_.converged = ok && std::isfinite(r) && r <= nl.newton_rtol;
    if (!diag_.converged && diag_.failure.empty()) diag_.failure = "not converged";
  }
  // fields: velocity as solved; pressure mean-free (c-weighted)
  double cp = 0, cs = 0;
  for (int a = 0; a < nv; ++a) {
    cp += I.cvec[a] * p[a];
    cs += I.cvec[a];
  }
  for (int a = 0; a < nv; ++a) p[a] -= cp / cs;
  I.u = u;
  I.p = p;
  I.solved = true;
  diag_.multiplier = lam_mult;
  diag_.seconds = MPI_Wtime() - t0;
  if (diag_.converged) diag_.failure.clear();
  (void)n;
  return diag_;
}

}  // namespace axf::fluids
