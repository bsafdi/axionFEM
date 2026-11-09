#include "axionfem/fullwave.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

#include "axionfem/background.hpp"
#include "axionfem/core.hpp"
#include "axionfem/signal.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using SpMatC = Eigen::SparseMatrix<cplx>;
using Trip = Eigen::Triplet<double>;
using TripC = Eigen::Triplet<cplx>;
using Vec = Eigen::VectorXd;
using VecC = Eigen::VectorXcd;
using units::pi;

namespace {

// Associated Legendre P_n^1(x), n = 1..N (unnormalized, no Condon-Shortley sign), as in the MQS DtN.
void legendre_p1(double x, int N, std::vector<double>& P) {
  P.assign(N + 1, 0.0);
  double s = std::sqrt(std::max(0.0, 1.0 - x * x));
  if (N >= 1) P[1] = s;
  if (N >= 2) P[2] = 3.0 * x * s;
  for (int n = 2; n < N; ++n) P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - (n + 1.0) * P[n - 1]) / n;
}

double inv_norm(int n) { return (2.0 * n + 1.0) / (2.0 * n * (n + 1.0)); }  // 1 / int_{-1}^{1} (P_n^1)^2

}  // namespace

// Eigenpairs of K x = lambda M x by shift-invert Lanczos in the M inner product with full
// reorthogonalization. `inv` maps b to the constrained solution of (K - sigma M) y = b, so every Lanczos vector
// satisfies the constraints. one_sided: sigma lies below the spectrum and the nev smallest lambda > sigma are
// returned (largest positive theta = 1/(lambda - sigma)); otherwise the nev lambda nearest sigma on either side
// (largest |theta|).
// (Declared in fullwave.hpp: also used by the cross-section cavity solver, xsec.cpp.)
std::vector<Eigenmode> lanczos(const SpMat& M, const std::function<Vec(const Vec&)>& inv, double sigma, int nev,
                               json& diag, bool one_sided) {
  const int n = static_cast<int>(M.rows());
  nev = std::min(nev, n);
  int m = std::min(n, std::max(2 * nev + 20, 40));
  std::vector<Eigenmode> out;
  auto mnorm = [&](const Vec& v) { return std::sqrt(std::max(0.0, v.dot(M * v))); };
  for (int attempt = 0; attempt < 5; ++attempt) {
    std::mt19937 rng(20261005u + attempt);
    std::normal_distribution<double> nd;
    Vec q(n);
    for (int i = 0; i < n; ++i) q(i) = nd(rng);
    q = inv(M * q);
    q /= mnorm(q);
    std::vector<Vec> Q{q};
    std::vector<double> alpha, beta;
    bool invariant = false;
    int restarts = 0;
    auto morth = [&](Vec& v) {
      for (int pass = 0; pass < 2; ++pass) {
        Vec Mv = M * v;
        for (auto& qq : Q) v -= qq.dot(Mv) * qq;
      }
    };
    for (int j = 0; j < m; ++j) {
      Vec w = inv(M * Q[j]);
      const double a = Q[j].dot(M * w);
      alpha.push_back(a);
      w -= a * Q[j];
      if (j > 0) w -= beta.back() * Q[j - 1];
      for (int pass = 0; pass < 2; ++pass) {
        Vec Mw = M * w;
        for (auto& qq : Q) w -= qq.dot(Mw) * qq;
      }
      const double b = mnorm(w);
      beta.push_back(b);
      if (j + 1 == m) break;
      if (b <= 1e-13 * std::fabs(a)) {
        // Numerically invariant subspace. With a shift at (or within round-off of) an eigenvalue, theta of that
        // mode exceeds the others by ~1/eps and the first Lanczos vector has lost every other component, so the
        // run would stop with one Ritz pair. Restart with a new block deflated against span(Q): a random vector
        // M-orthogonalized first (so inv() adds no component along the dominant mode beyond round-off), mapped
        // through inv() (constraints), and orthogonalized again; beta = 0 couples the blocks.
        if (static_cast<int>(Q.size()) >= n || restarts >= nev + 8) { invariant = true; break; }
        Vec r(n);
        for (int i = 0; i < n; ++i) r(i) = nd(rng);
        morth(r);
        r = inv(M * r);
        morth(r);
        const double rn = mnorm(r);
        if (!(rn > 0)) { invariant = true; break; }
        beta.back() = 0.0;
        Q.push_back(r / rn);
        ++restarts;
        continue;
      }
      Q.push_back(w / b);
    }
    const int k = static_cast<int>(alpha.size());
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
      T(i, i) = alpha[i];
      if (i + 1 < k) T(i, i + 1) = T(i + 1, i) = beta[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    std::vector<int> order;  // largest positive theta <-> smallest lambda above sigma
    if (one_sided) {
      for (int i = k - 1; i >= 0; --i) if (es.eigenvalues()(i) > 0) order.push_back(i);
    } else {  // largest |theta| <-> lambda nearest sigma (both sides)
      for (int i = 0; i < k; ++i) order.push_back(i);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return std::fabs(es.eigenvalues()(a)) > std::fabs(es.eigenvalues()(b));
      });
    }
    out.clear();
    bool ok = true;
    for (int c = 0; c < static_cast<int>(order.size()) && c < nev; ++c) {
      const int i = order[c];
      const double th = es.eigenvalues()(i);
      const double res = std::fabs(beta.back() * es.eigenvectors()(k - 1, i)) / std::fabs(th);
      if (res > 1e-8 && !invariant) ok = false;
      Eigenmode md;
      const double lam = sigma + 1.0 / th;
      md.omega = std::sqrt(std::max(0.0, lam));
      md.f_Hz = md.omega / (2 * pi);
      md.residual = res;
      Vec x = Vec::Zero(n);
      for (int jj = 0; jj < k; ++jj) x += es.eigenvectors()(jj, i) * Q[jj];
      md.x.assign(x.data(), x.data() + n);
      out.push_back(std::move(md));
    }
    diag = {{"lanczos_steps", k}, {"attempts", attempt + 1}, {"converged", ok}, {"lanczos_restarts", restarts}};
    if (ok || m >= n) break;
    m = std::min(n, 2 * m);
  }
  std::sort(out.begin(), out.end(), [](const Eigenmode& a, const Eigenmode& b) { return a.omega < b.omega; });
  return out;
}

// ---------------------------------------------------------------- complex symmetric LDL^T
// Up-looking sparse LDL^T without pivoting (the algorithm of T. Davis' LDL) for complex symmetric,
// non-Hermitian matrices: no conjugation anywhere. Fill-reducing AMD ordering; the symbolic analysis is
// reused while the pattern is unchanged (a frequency sweep).
bool ComplexSymLDLT::analyze(const SpMatC& A) {
  n_ = static_cast<int>(A.rows());
  SpMat pat = A.real();
  for (int k = 0; k < pat.outerSize(); ++k)
    for (SpMat::InnerIterator it(pat, k); it; ++it) it.valueRef() = 1.0;
  Eigen::AMDOrdering<int> amd;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pinv;
  amd(pat, pinv);
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P = pinv.inverse();
  newidx_.assign(P.indices().data(), P.indices().data() + n_);
  // upper triangle (row <= col) of the permuted matrix, compressed by column
  std::vector<std::vector<std::pair<int, long>>> cols(n_);
  long nnz = 0;
  for (int k = 0; k < A.outerSize(); ++k)
    for (SpMatC::InnerIterator it(A, k); it; ++it) {
      const int i = newidx_[it.row()], j = newidx_[it.col()];
      if (i <= j) cols[j].push_back({i, nnz});
      ++nnz;
    }
  Cp_.assign(n_ + 1, 0);
  Ci_.clear();
  src_.clear();
  for (int j = 0; j < n_; ++j) {
    std::sort(cols[j].begin(), cols[j].end());
    Cp_[j + 1] = Cp_[j] + static_cast<int>(cols[j].size());
    for (auto& [i, q] : cols[j]) { Ci_.push_back(i); src_.push_back(q); }
  }
  nnzA_ = nnz;
  // elimination tree and column counts
  parent_.assign(n_, -1);
  std::vector<int> flag(n_), lnz(n_, 0);
  for (int k = 0; k < n_; ++k) {
    flag[k] = k;
    for (int p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      for (int i = Ci_[p]; i < k && flag[i] != k; i = parent_[i]) {
        if (parent_[i] == -1) parent_[i] = k;
        ++lnz[i];
        flag[i] = k;
      }
    }
  }
  Lp_.assign(n_ + 1, 0);
  for (int k = 0; k < n_; ++k) Lp_[k + 1] = Lp_[k] + lnz[k];
  Li_.assign(Lp_[n_], 0);
  Lx_.assign(Lp_[n_], 0.0);
  D_.assign(n_, 0.0);
  return true;
}

bool ComplexSymLDLT::factorize(const SpMatC& A) {
  if (static_cast<long>(A.nonZeros()) != nnzA_ || A.rows() != n_) analyze(A);
  std::vector<cplx> vals(nnzA_);
  {
    long q = 0;
    for (int k = 0; k < A.outerSize(); ++k)
      for (SpMatC::InnerIterator it(A, k); it; ++it) vals[q++] = it.value();
  }
  std::vector<cplx> Y(n_, 0.0);
  std::vector<int> pattern(n_), flag(n_), lnz(n_, 0);
  double dmax = 0;
  for (int k = 0; k < n_; ++k) {
    int top = n_;
    flag[k] = k;
    for (int p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      int i = Ci_[p];
      Y[i] += vals[src_[p]];
      int len = 0;
      for (; i < k && flag[i] != k; i = parent_[i]) {
        pattern[len++] = i;
        flag[i] = k;
      }
      while (len > 0) pattern[--top] = pattern[--len];
    }
    cplx d = Y[k];
    Y[k] = 0.0;
    for (; top < n_; ++top) {
      const int i = pattern[top];
      const cplx yi = Y[i];
      Y[i] = 0.0;
      const int p2 = Lp_[i] + lnz[i];
      for (int p = Lp_[i]; p < p2; ++p) Y[Li_[p]] -= Lx_[p] * yi;
      const cplx lki = yi / D_[i];
      d -= lki * yi;
      Li_[p2] = k;
      Lx_[p2] = lki;
      ++lnz[i];
    }
    D_[k] = d;
    dmax = std::max(dmax, std::abs(d));
    if (!(std::abs(d) > 1e-14 * dmax)) return false;  // zero pivot: the caller falls back to pivoting LU
  }
  return true;
}

VecC ComplexSymLDLT::solve(const VecC& b) const {
  VecC y(n_);
  for (int i = 0; i < n_; ++i) y(newidx_[i]) = b(i);
  for (int j = 0; j < n_; ++j)
    for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) y(Li_[p]) -= Lx_[p] * y(j);
  for (int j = 0; j < n_; ++j) y(j) /= D_[j];
  for (int j = n_ - 1; j >= 0; --j)
    for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) y(j) -= Lx_[p] * y(Li_[p]);
  VecC x(n_);
  for (int i = 0; i < n_; ++i) x(i) = y(newidx_[i]);
  return x;
}

std::vector<cplx> hankel_log_derivative(double x, int N) {
  std::vector<cplx> g(N + 1, 0.0);
  if (!(x > 0)) {
    for (int n = 0; n <= N; ++n) g[n] = static_cast<double>(n);
    return g;
  }
  // r_n = h_n / h_{n-1}: r_1 = 1/x - i; r_{n+1} = (2n+1)/x - 1/r_n (stable upward for h^(1)).
  // gamma_n = -1 - x h_n'/h_n = n - x h_{n-1}/h_n = n - x / r_n;  gamma_0 = -1 - x h_0'/h_0 = -i x.
  g[0] = cplx(0.0, -x);
  cplx r(1.0 / x, -1.0);
  for (int n = 1; n <= N; ++n) {
    g[n] = static_cast<double>(n) - x / r;
    r = (2.0 * n + 1.0) / x - 1.0 / r;
  }
  return g;
}

std::vector<cplx> hankel_log_derivative_dx(double x, int N) {
  std::vector<cplx> d(N + 1, 0.0);
  if (!(x > 0)) return d;
  // d gamma_n/dx = x + (gamma_n - n)(gamma_n + n + 1)/x with gamma_n - n = -x / r_n (same recurrence as above):
  // d gamma_n/dx = x - (gamma_n + n + 1) / r_n, free of the n - n cancellation at small x; d gamma_0/dx = -i.
  d[0] = cplx(0.0, -1.0);
  cplx r(1.0 / x, -1.0);
  for (int n = 1; n <= N; ++n) {
    const cplx g = static_cast<double>(n) - x / r;
    d[n] = x - (g + static_cast<double>(n) + 1.0) / r;
    r = (2.0 * n + 1.0) / x - 1.0 / r;
  }
  return d;
}

cplx fw_exterior_coefficient(bool te, double st, double w, double R1, cplx gamma) {
  const double eps0 = 1.0 / (units::mu0 * units::c * units::c);
  return te ? (gamma - st) / (units::mu0 * R1) : -w * w * eps0 * R1 / gamma;
}

cplx fw_exterior_coefficient_dw(bool te, double w, double R1, cplx gamma, cplx dgamma) {
  const double eps0 = 1.0 / (units::mu0 * units::c * units::c);
  const double x = w * R1 / units::c;
  return te ? dgamma / (units::mu0 * units::c) : -eps0 * R1 * w * (2.0 * gamma - x * dgamma) / (gamma * gamma);
}

void assoc_legendre_p1(double x, int N, std::vector<double>& P) { legendre_p1(x, N, P); }

std::vector<cplx> exterior_radial_integrals(double k, double Rb, int N) {
  // Exact: int z^{1-n} h_n(z) dz = -z^{1-n} h_{n-1}(z)  =>  J_n = (Rb^2/k) h_{n-1}(k Rb)/h_n(k Rb),
  // with the same stable upward ratio recurrence r_n = h_n/h_{n-1} as the DtN map; -> Rb^3/(2n-1) as k -> 0.
  std::vector<cplx> J(N + 1, 0.0);
  const double z0 = k * Rb;
  if (!(z0 > 0)) {
    for (int n = 1; n <= N; ++n) J[n] = Rb * Rb * Rb / (2.0 * n - 1.0);
    return J;
  }
  const cplx I1(0.0, 1.0);
  cplx r = 1.0 / z0 - I1;  // r_1 = h_1(z0)/h_0(z0)
  for (int n = 1; n <= N; ++n) {
    J[n] = Rb * Rb / (k * r);
    r = (2.0 * n + 1.0) / z0 - 1.0 / r;
  }
  return J;
}

void hankel_ratio(double k, double R1, double rho, int N, std::vector<cplx>& out) {
  out.assign(N + 1, 0.0);
  const double x = k * rho;
  const double x1 = k * R1;
  cplx ratio = (R1 / rho) * std::exp(cplx(0.0, x - x1));
  out[0] = ratio;
  cplx r(1.0 / x, -1.0);
  cplx r1(1.0 / x1, -1.0);
  for (int n = 1; n <= N; ++n) {
    ratio *= r / r1;
    out[n] = ratio;
    r = (2.0 * n + 1.0) / x - 1.0 / r;
    r1 = (2.0 * n + 1.0) / x1 - 1.0 / r1;
  }
}

std::vector<cplx> BackgroundExterior::G(double k) const {
  std::vector<cplx> g(N + 1, 0.0);
  std::vector<cplx> h;
  for (size_t q = 0; q < rho.size(); ++q) {
    hankel_ratio(k, R1, rho[q], N, h);
    for (int n = 1; n <= N; ++n) g[n] += moments[n][q] * h[n];
  }
  if (rho_s > 0) {
    const auto J = exterior_radial_integrals(k, rho_s, N);
    hankel_ratio(k, R1, rho_s, N, h);
    for (int n = 1; n <= N; ++n) g[n] += tail[n] * h[n] * J[n];
  }
  return g;
}

std::vector<double> BackgroundExterior::G_static() const {
  std::vector<double> g(N + 1, 0.0);
  for (size_t q = 0; q < rho.size(); ++q) {
    const double s = R1 / rho[q];
    double p = s;
    for (int n = 1; n <= N; ++n) {
      p *= s;
      g[n] += moments[n][q] * p;
    }
  }
  if (rho_s > 0) {
    const double s = R1 / rho_s;
    double p = s;
    for (int n = 1; n <= N; ++n) {
      p *= s;
      g[n] += tail[n] * p * rho_s * rho_s * rho_s / (2.0 * n - 1.0);
    }
  }
  return g;
}

void dtheta_y0(double u, int N, std::vector<double>& d) {
  assoc_legendre_p1(u, N, d);
  for (int n = 0; n <= N; ++n) d[n] *= -std::sqrt((2.0 * n + 1.0) / (4 * pi));
}

namespace {

// the factors of dtheta_y0: dY_n0/dtheta = s_n P_n^1(cos theta), s_n = -sqrt((2n + 1) / (4 pi)), n = 0..N
std::vector<double> dtheta_y0_factors(int N) {
  std::vector<double> s(N + 1);
  for (int n = 0; n <= N; ++n) s[n] = -std::sqrt((2.0 * n + 1.0) / (4 * pi));
  return s;
}

// scratch of arc_moments: the points' cos(theta) and weight w A_phi, and three orders of P_n^1 at them
struct ArcScratch {
  std::vector<double> x, wa, p0, p1, p2;
};

// adds int_{th0}^{th1} A_phi(rho, theta) dY_n0/dtheta sin(theta) dtheta, n = 1..N, to acc (composite rule gl, about
// panels / pi panels per radian, and at least max(2, panels / 64) on any arc, so that doubling the panels refines a
// short arc too: the quadrature check would not see its error otherwise; 2 at the default 128); s from
// dtheta_y0_factors. Each point's P_n^1 comes from the recurrence of
// assoc_legendre_p1, and each order's sum runs over the points in order, as with dtheta_y0 point by point; the
// recurrence runs over all points of the arc at once (per point it is a chain of divisions, 80 long at the 2D
// exterior map's default order).
void arc_moments(const BackgroundField& bg, const mfem::IntegrationRule& gl, const std::vector<double>& s, double rho,
                 double cz, double th0, double th1, int N, int panels, ArcScratch& t, std::vector<double>& acc) {
  const int nt = std::max({2, panels / 64, static_cast<int>(std::ceil(panels * (th1 - th0) / pi))});
  t.x.clear();
  t.wa.clear();
  for (int p = 0; p < nt; ++p) {
    const double a = th0 + (th1 - th0) * p / nt;
    const double b = th0 + (th1 - th0) * (p + 1) / nt;
    for (int i = 0; i < gl.GetNPoints(); ++i) {
      const double th = a + (b - a) * gl.IntPoint(i).x;
      const double w = (b - a) * gl.IntPoint(i).weight * std::sin(th);
      const double r = rho * std::sin(th);
      const double z = cz + rho * std::cos(th);
      if (!(r > 0)) continue;
      const double A = bg.psi(r, z) / r;
      if (A == 0.0) continue;
      t.x.push_back(std::cos(th));
      t.wa.push_back(w * A);
    }
  }
  const size_t m = t.x.size();
  if (m == 0 || N < 1) return;
  // P_1 = sin(theta), P_2 = 3 x P_1, P_{n+1} = ((2n + 1) x P_n - (n + 1) P_{n-1}) / n; p0, p1, p2 = P_{n-1}, P_n, P_{n+1}
  t.p1.resize(m);
  for (size_t j = 0; j < m; ++j) t.p1[j] = std::sqrt(std::max(0.0, 1.0 - t.x[j] * t.x[j]));
  for (size_t j = 0; j < m; ++j) acc[1] += t.wa[j] * (t.p1[j] * s[1]);
  if (N < 2) return;
  t.p0.swap(t.p1);
  t.p1.resize(m);
  for (size_t j = 0; j < m; ++j) t.p1[j] = 3.0 * t.x[j] * t.p0[j];
  for (size_t j = 0; j < m; ++j) acc[2] += t.wa[j] * (t.p1[j] * s[2]);
  t.p2.resize(m);
  for (int n = 2; n < N; ++n) {
    for (size_t j = 0; j < m; ++j) t.p2[j] = ((2.0 * n + 1.0) * t.x[j] * t.p1[j] - (n + 1.0) * t.p0[j]) / n;
    for (size_t j = 0; j < m; ++j) acc[n + 1] += t.wa[j] * (t.p2[j] * s[n + 1]);
    t.p0.swap(t.p1);
    t.p1.swap(t.p2);
  }
}

}  // namespace

BackgroundExterior background_exterior(const BackgroundField& bg, double r_max, double z_lo, double z_hi, double R1,
                                       double cz, int N, int panels) {
  BackgroundExterior out;
  out.N = N;
  out.R1 = R1;
  out.moments.assign(N + 1, std::vector<double>());
  const double rho_max = std::max(std::hypot(r_max, z_hi - cz), std::hypot(r_max, z_lo - cz));
  const double rho_s = std::min({r_max, z_hi - cz, cz - z_lo});
  if (!(rho_max > R1) || N < 1) return out;
  const bool continued = rho_s > R1;
  const double outer = continued ? rho_s : rho_max;
  std::vector<double> breaks = {R1, outer};
  if (!continued)
    for (double b : {r_max, std::fabs(z_hi - cz), std::fabs(z_lo - cz), std::hypot(r_max, z_hi - cz),
                     std::hypot(r_max, z_lo - cz)})
      if (b > R1 && b < rho_max) breaks.push_back(b);
  std::sort(breaks.begin(), breaks.end());
  const mfem::IntegrationRule& gl = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 15);  // 8 points on [0, 1]
  std::vector<double> rw;
  for (size_t s = 0; s + 1 < breaks.size(); ++s) {
    const double a = breaks[s];
    const double b = breaks[s + 1];
    if (!(b > a)) continue;
    // at least max(4, panels / 32) panels on every radial piece, so that the check's doubled run refines a short piece
    // too (4 at the default 128)
    const int np = std::max({4, panels / 32, static_cast<int>(std::ceil(panels * (b - a) / (outer - R1)))});
    for (int p = 0; p < np; ++p) {
      // quadratic grading towards R1 on the first piece, uniform panels elsewhere
      const double t0 = static_cast<double>(p) / np;
      const double t1 = static_cast<double>(p + 1) / np;
      const double s0 = s == 0 ? t0 * t0 : t0;
      const double s1 = s == 0 ? t1 * t1 : t1;
      const double p0 = a + (b - a) * s0;
      const double p1 = a + (b - a) * s1;
      for (int i = 0; i < gl.GetNPoints(); ++i) {
        out.rho.push_back(p0 + (p1 - p0) * gl.IntPoint(i).x);
        rw.push_back((p1 - p0) * gl.IntPoint(i).weight);
      }
    }
  }
  for (int n = 0; n <= N; ++n) out.moments[n].assign(out.rho.size(), 0.0);
  const std::vector<double> sy = dtheta_y0_factors(std::max(N, 40));
  const long nq = static_cast<long>(out.rho.size());
  // the spheres are independent: each fills its own column of moments (the same arithmetic on any thread count)
#pragma omp parallel
  {
    ArcScratch scratch;
    std::vector<double> acc(N + 1);
#pragma omp for schedule(dynamic, 4)
    for (long q = 0; q < nq; ++q) {
      const double rho = out.rho[q];
      // arcs of the sphere rho inside the box, in u = cos(theta) (the whole sphere inside rho_s)
      const double u_lo = std::max(-1.0, (z_lo - cz) / rho);
      const double u_hi = std::min(1.0, (z_hi - cz) / rho);
      std::vector<std::pair<double, double>> arcs;
      if (r_max >= rho) {
        arcs.push_back({u_lo, u_hi});
      } else {
        const double us = std::sqrt(1.0 - (r_max / rho) * (r_max / rho));
        arcs.push_back({u_lo, std::min(u_hi, -us)});
        arcs.push_back({std::max(u_lo, us), u_hi});
      }
      std::fill(acc.begin(), acc.end(), 0.0);
      for (const auto& [ua, ub] : arcs) {
        if (!(ub > ua)) continue;
        const double th0 = std::acos(std::clamp(ub, -1.0, 1.0));  // theta increases as u decreases
        const double th1 = std::acos(std::clamp(ua, -1.0, 1.0));
        arc_moments(bg, gl, sy, rho, cz, th0, th1, N, panels, scratch, acc);
      }
      for (int n = 1; n <= N; ++n) out.moments[n][q] = 2 * pi * rho * rho * rw[q] * acc[n];
    }
  }
  if (!continued) return out;
  out.rho_s = rho_s;
  out.tail.assign(N + 1, 0.0);
  ArcScratch scratch;
  arc_moments(bg, gl, sy, rho_s, cz, 0.0, pi, N, panels, scratch, out.tail);
  for (int n = 1; n <= N; ++n) out.tail[n] *= 2 * pi;
  // the continuation against the grid where the box reaches beyond rho_s: A_phi(rho_s, theta) = sum alpha_n
  // P_n^1(cos theta) to order 40, from int A_phi dY_n0/dtheta sin(theta) dtheta = -c_n alpha_n 2n(n+1)/(2n+1)
  const int Nt = 40;
  std::vector<double> m(Nt + 1, 0.0);
  arc_moments(bg, gl, sy, rho_s, cz, 0.0, pi, Nt, 4 * panels, scratch, m);
  std::vector<double> alpha(Nt + 1, 0.0);
  for (int n = 1; n <= Nt; ++n) {
    const double c = std::sqrt((2.0 * n + 1.0) / (4 * pi));
    alpha[n] = -m[n] * (2.0 * n + 1.0) / (c * 2.0 * n * (n + 1.0));
  }
  double d2 = 0;
  double a2 = 0;
  std::vector<double> P;
  for (int i = 1; i <= 20; ++i) {
    const double rho = rho_s + (rho_max - rho_s) * i / 20.0;
    for (int j = 0; j < 64; ++j) {
      const double th = pi * (j + 0.5) / 64.0;
      const double r = rho * std::sin(th);
      const double z = cz + rho * std::cos(th);
      if (r > r_max || z < z_lo || z > z_hi) continue;
      const double grid = bg.psi(r, z) / r;
      assoc_legendre_p1(std::cos(th), Nt, P);
      double cont = 0;
      double s = rho_s / rho;
      for (int n = 1; n <= Nt; ++n) {
        s *= rho_s / rho;
        cont += alpha[n] * s * P[n];
      }
      d2 += (grid - cont) * (grid - cont);
      a2 += grid * grid;
      ++out.tail_samples;
    }
  }
  out.tail_mismatch = a2 > 0 ? std::sqrt(d2 / a2) : 0.0;
  return out;
}

int background_exterior_panels() {
  int panels = 128;
  if (const char* v = std::getenv("AXF_FW_BG_PANELS")) {  // experiments
    char* end = nullptr;
    const long value = std::strtol(v, &end, 10);
    AXF_REQUIRE(end != v && *end == '\0' && value >= 4 && value <= 4096,
                strcat("AXF_FW_BG_PANELS must be an integer from 4 to 4096 (got '", v, "')"), "/background");
    panels = static_cast<int>(value);
  }
  return panels;
}

double background_exterior_quadrature_check(const BackgroundField& bg, double r_max, double z_lo, double z_hi,
                                            double R1, double cz, int panels, const BackgroundExterior& coarse,
                                            double* scale) {
  // to order 12 only: at N = 80 the check at full order cost four fifths of the moments.
  // For a grid continued beyond its inscribed sphere the largest
  // moments are the low orders' (n = 1); for a grid cut at its box near R1 they are not (max over n <= 12 was 0.84 of
  // the full maximum in the tested cases), so there the check covers orders <= 12 only. Each order's
  // moments do not depend on N.
  const int n_check = std::min(coarse.N, 12);
  const BackgroundExterior fine = background_exterior(bg, r_max, z_lo, z_hi, R1, cz, n_check, 2 * panels);
  const auto g0 = coarse.G_static();
  const auto g1 = fine.G_static();
  double change = 0;
  double gmax = 0;
  for (int n = 1; n <= coarse.N && n <= fine.N; ++n) {
    change = std::max(change, std::fabs(g0[n] - g1[n]));
    gmax = std::max(gmax, std::fabs(g1[n]));
  }
  if (scale) *scale = gmax;
  return gmax > 0 ? change / gmax : 0.0;
}

AxiWave::AxiWave(const Space& sp, std::vector<WaveRole> roles, std::vector<Master> masters,
                 std::map<int, int> sheet_master_by_attr)
    : sp_(sp), roles_(std::move(roles)), masters_(std::move(masters)), sheet_master_(std::move(sheet_master_by_attr)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: role table size mismatch");
  for (auto& m : masters_)
    AXF_REQUIRE(!m.fixed || m.psi == 0.0, "internal: full-wave masters with a fixed nonzero flux");
}

void AxiWave::set_outer(int outer_attr, double radius, double alpha0, bool radiation, bool closed_dirichlet,
                        int modes, double center_z) {
  outer_attr_ = outer_attr;
  Rb_ = radius;
  center_z_ = center_z;
  alpha0_ = alpha0;
  radiation_ = radiation;
  closed_dirichlet_ = closed_dirichlet;
  modes_ = modes;
  setup_closed_ = -1;
}

int AxiWave::ndof() const { return sp_.ndof() + static_cast<int>(split_src_.size()); }

const std::vector<int>& AxiWave::edofs(int e) const {
  if (!edofs_split_.empty() && !edofs_split_[e].empty()) return edofs_split_[e];
  return sp_.edofs(e);
}

double AxiWave::dof_r(int i) const { return i < sp_.ndof() ? sp_.r[i] : sp_.r[split_src_[i - sp_.ndof()]]; }
double AxiWave::dof_z(int i) const { return i < sp_.ndof() ? sp_.z[i] : sp_.z[split_src_[i - sp_.ndof()]]; }

void AxiWave::split_dofs(std::map<int, std::map<int, int>> remap, std::vector<int> src) {
  AXF_REQUIRE(cfull_.empty(), "internal: split_dofs after add_constraint");
  split_src_ = std::move(src);
  edofs_split_.assign(sp_.mesh.GetNE(), {});
  for (auto& [e, mp] : remap) {
    std::vector<int> d = sp_.edofs(e);
    for (auto& x : d) {
      auto it = mp.find(x);
      if (it != mp.end()) {
        AXF_REQUIRE(it->second >= sp_.ndof() && it->second < ndof(), "internal: split DOF index out of range");
        x = it->second;
      }
    }
    edofs_split_[e] = std::move(d);
  }
  setup_closed_ = -1;
}

void AxiWave::add_constraint(const std::string& name, Eigen::VectorXd c_full) {
  AXF_REQUIRE(c_full.size() == ndof(), "internal: constraint size mismatch");
  cname_.push_back(name);
  cfull_.push_back(std::move(c_full));
}

void AxiWave::elem_raw(int e, Eigen::MatrixXd& Ke, Eigen::MatrixXd& Me) const {
  const WaveRole& ro = roles_[e];
  const RefTable& tab = sp_.table(sp_.quad_rule());
  PointEval pe;
  const int n = static_cast<int>(sp_.edofs(e).size());
  Ke.setZero(n, n);
  Me.setZero(n, n);
  std::vector<double> br(n), bz(n);
  for (int q = 0; q < tab.nq; ++q) {
    sp_.eval(e, tab, q, pe);
    const double wr = pe.w * pe.r;
    const double al = ro.alpha_at ? ro.alpha_at(e, pe) : ro.alpha;
    const double be = ro.beta_at ? ro.beta_at(e, pe) : ro.beta;
    for (int i = 0; i < n; ++i) {
      br[i] = -pe.dN[2 * i + 1];
      bz[i] = pe.dN[2 * i] + pe.N[i] / pe.r;
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        Ke(i, j) += wr * al * (br[i] * br[j] + bz[i] * bz[j]);
        Me(i, j) += wr * be * pe.N[i] * pe.N[j];
      }
  }
}

// Lossless parts of a lossy group for the eigenproblem: the stiffness coefficient alpha (1/eps for TM, 1/mu for
// TE) becomes 1/Re(1/alpha) (1/eps', 1/mu'), the mass coefficient beta becomes Re(beta) (eps', mu'). Evaluated
// at a high frequency, where conduction terms ~ 1/omega vanish.
namespace {
constexpr double kLosslessRefOmega = 1e12;
double lossless_alpha(const WaveGroup& g) { return 1.0 / (1.0 / g.alpha(kLosslessRefOmega)).real(); }
double lossless_beta(const WaveGroup& g) { return g.beta(kLosslessRefOmega).real(); }
}  // namespace

void AxiWave::element_matrices(int e, Eigen::MatrixXd& Ke, Eigen::MatrixXd& Me) const {
  elem_raw(e, Ke, Me);
  const int g = roles_[e].group;
  if (g >= 0) {
    Ke *= lossless_alpha(groups_[g]);
    Me *= lossless_beta(groups_[g]);
  }
}

void AxiWave::setup(bool closed) {
  if (setup_closed_ == (closed ? 1 : 0)) return;
  setup_closed_ = closed ? 1 : 0;
  const int nd = ndof(), ne = sp_.mesh.GetNE(), nm = static_cast<int>(masters_.size());
  kind_.assign(nd, 0);
  master_.assign(nd, -1);
  w_.assign(nd, 0.0);
  index_.assign(nd, -1);
  auto tie = [&](int dof, int m) {
    if (master_[dof] >= 0 && master_[dof] != m)
      throw Error(strcat("perfect conductors '", masters_[master_[dof]].name, "' and '", masters_[m].name,
                         "' touch at (r, z) = (", dof_r(dof), ", ", dof_z(dof), ")"),
                  "/regions", "Touching conductors form one body; separate or merge them.");
    master_[dof] = m;
  };
  auto skipped = [&](int e) { return roles_[e].master || roles_[e].excluded; };
  for (int e = 0; e < ne; ++e)
    if (roles_[e].master) for (int d : edofs(e)) tie(d, roles_[e].master_id);
  mfem::Array<int> bd;
  std::vector<char> outer(nd, 0);
  for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
    const int a = sp_.mesh.GetBdrAttribute(be);
    sp_.fes->GetBdrElementDofs(be, bd);
    auto it = sheet_master_.find(a);
    if (it != sheet_master_.end()) for (int d : bd) tie(d, it->second);
    if (a == outer_attr_) for (int d : bd) outer[d] = 1;
  }
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e)
    if (!skipped(e)) for (int d : edofs(e)) used[d] = 1;
  nfree_ = 0;
  for (int i = 0; i < nd; ++i) {
    if (dof_r(i) == 0.0) {
      kind_[i] = 3;
      AXF_REQUIRE(master_[i] < 0 || masters_[master_[i]].fixed,
                  "conductor '" + (master_[i] >= 0 ? masters_[master_[i]].name : std::string()) +
                      "' touches the symmetry axis but is not fixed",
                  "/superconductors");
    } else if (master_[i] >= 0) {
      kind_[i] = 2;
      w_[i] = 1.0 / dof_r(i);
    } else if (closed && closed_dirichlet_ && outer[i]) {
      kind_[i] = 3;
    } else if (used[i]) {
      kind_[i] = 1;
      index_[i] = nfree_++;
    }
  }
  AXF_REQUIRE(nfree_ > 0, "no free degrees of freedom in the full-wave problem");
  gidx_.assign(nm, -1);
  int ng = 0;
  for (int m = 0; m < nm; ++m) if (!masters_[m].fixed) gidx_[m] = ng++;
  nred_ = nfree_ + ng;
  std::vector<Trip> pt;
  for (int i = 0; i < nd; ++i) {
    if (kind_[i] == 1) pt.emplace_back(i, index_[i], 1.0);
    else if (kind_[i] == 2 && !masters_[master_[i]].fixed) pt.emplace_back(i, nfree_ + gidx_[master_[i]], w_[i]);
  }
  P_.resize(nd, nred_);
  P_.setFromTriplets(pt.begin(), pt.end());

  const int ngr = static_cast<int>(groups_.size());
  std::vector<Trip> tk, tm;
  std::vector<std::vector<Trip>> tkg(ngr), tmg(ngr);
  Eigen::MatrixXd Ke, Me;
  for (int e = 0; e < ne; ++e) {
    if (skipped(e)) continue;
    const WaveRole& ro = roles_[e];
    AXF_REQUIRE(ro.group < ngr, "internal: lossy group index out of range");
    const auto& d = edofs(e);
    const int n = static_cast<int>(d.size());
    elem_raw(e, Ke, Me);
    auto& TK = ro.group >= 0 ? tkg[ro.group] : tk;
    auto& TM = ro.group >= 0 ? tmg[ro.group] : tm;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        TK.emplace_back(d[i], d[j], Ke(i, j));
        TM.emplace_back(d[i], d[j], Me(i, j));
      }
  }
  K_.resize(nd, nd);
  K_.setFromTriplets(tk.begin(), tk.end());
  M_.resize(nd, nd);
  M_.setFromTriplets(tm.begin(), tm.end());
  Kr_ = P_.transpose() * K_ * P_;
  Mr_ = P_.transpose() * M_ * P_;
  Kg_.assign(ngr, SpMat(nd, nd));
  Mg_.assign(ngr, SpMat(nd, nd));
  Kgr_.resize(ngr);
  Mgr_.resize(ngr);
  for (int g = 0; g < ngr; ++g) {
    Kg_[g].setFromTriplets(tkg[g].begin(), tkg[g].end());
    Mg_[g].setFromTriplets(tmg[g].begin(), tmg[g].end());
    Kgr_[g] = P_.transpose() * Kg_[g] * P_;
    Mgr_[g] = P_.transpose() * Mg_[g] * P_;
  }

  // multipole projections of the outer-boundary trace: U(b, n-1) = int N_b P_n^1(cos theta) r dl / Rb
  bdofs_.clear();
  U_.resize(0, 0);
  if (!closed && outer_attr_ > 0 && modes_ > 0) {
    std::map<int, int> bidx;
    int nbe = 0;
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      if (sp_.mesh.GetBdrAttribute(be) != outer_attr_) continue;
      ++nbe;
      sp_.fes->GetBdrElementDofs(be, bd);
      for (int d : bd)
        if (!bidx.count(d)) {
          AXF_REQUIRE(kind_[d] == 1 || kind_[d] == 3, "a conductor touches the outer boundary");
          bidx[d] = static_cast<int>(bdofs_.size());
          bdofs_.push_back(d);
        }
    }
    const int N = std::min(modes_, std::max(4, nbe));
    U_ = Eigen::MatrixXd::Zero(bdofs_.size(), N);
    std::vector<double> Pn;
    mfem::Vector shape, x(2);
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      if (sp_.mesh.GetBdrAttribute(be) != outer_attr_) continue;
      sp_.fes->GetBdrElementDofs(be, bd);
      const mfem::FiniteElement* fe = sp_.fes->GetBE(be);
      mfem::ElementTransformation* T = sp_.mesh.GetBdrElementTransformation(be);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * sp_.order + 14);
      shape.SetSize(fe->GetDof());
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        fe->CalcShape(ip, shape);
        const double r = x(0), dl = ip.weight * T->Weight();
        // polar angle about the domain centre (0, center_z_): the exterior multipole origin
        legendre_p1(std::cos(std::atan2(r, x(1) - center_z_)), N, Pn);
        for (int i = 0; i < bd.Size(); ++i) {
          const double c = shape(i) * r * dl / Rb_;
          for (int n = 1; n <= N; ++n) U_(bidx[bd[i]], n - 1) += c * Pn[n];
        }
      }
    }
  }
}

AxiWave::Solution AxiWave::solve(double omega, const std::vector<cplx>& rhs) {
  Timer timer;
  setup(!radiation_);
  const int nd = ndof(), nm = static_cast<int>(masters_.size()), nc = static_cast<int>(cfull_.size());
  const double k0 = omega / units::c;
  std::vector<TripC> t;
  t.reserve(Kr_.nonZeros() + Mr_.nonZeros());
  for (int c = 0; c < Kr_.outerSize(); ++c)
    for (SpMat::InnerIterator it(Kr_, c); it; ++it) t.emplace_back(it.row(), it.col(), cplx(it.value(), 0));
  for (int c = 0; c < Mr_.outerSize(); ++c)
    for (SpMat::InnerIterator it(Mr_, c); it; ++it) t.emplace_back(it.row(), it.col(), cplx(-omega * omega * it.value(), 0));
  const int ngr = static_cast<int>(groups_.size());
  std::vector<cplx> ag(ngr), bg(ngr);
  for (int gi = 0; gi < ngr; ++gi) {
    ag[gi] = groups_[gi].alpha(omega);
    bg[gi] = groups_[gi].beta(omega);
    for (int c = 0; c < Kgr_[gi].outerSize(); ++c)
      for (SpMat::InnerIterator it(Kgr_[gi], c); it; ++it) t.emplace_back(it.row(), it.col(), ag[gi] * it.value());
    for (int c = 0; c < Mgr_[gi].outerSize(); ++c)
      for (SpMat::InnerIterator it(Mgr_[gi], c); it; ++it)
        t.emplace_back(it.row(), it.col(), -omega * omega * bg[gi] * it.value());
  }
  Eigen::VectorXcd coef;
  Eigen::MatrixXcd D;
  if (U_.size() > 0) {
    const int N = static_cast<int>(U_.cols());
    auto g = hankel_log_derivative(k0 * Rb_, N);
    coef.resize(N);
    for (int n = 1; n <= N; ++n) coef(n - 1) = alpha0_ * g[n] * inv_norm(n) / Rb_;
    D = U_.cast<cplx>() * coef.asDiagonal() * U_.transpose().cast<cplx>();
    for (size_t a = 0; a < bdofs_.size(); ++a) {
      if (kind_[bdofs_[a]] != 1) continue;
      for (size_t b = 0; b < bdofs_.size(); ++b)
        if (kind_[bdofs_[b]] == 1) t.emplace_back(index_[bdofs_[a]], index_[bdofs_[b]], D(a, b));
    }
  }
  SpMatC A(nred_, nred_);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  // complex symmetric LDL^T (fast); pivoting sparse LU if it meets a zero pivot or loses accuracy
  const bool ldl_ok = ldl_.factorize(A);
  Eigen::SparseLU<SpMatC, Eigen::COLAMDOrdering<int>> lu;
  bool use_lu = !ldl_ok;
  auto lu_init = [&]() {
    lu.analyzePattern(A);
    lu.factorize(A);
    if (lu.info() != Eigen::Success) {
      throw SolverFailure(strcat("the full-wave system is singular at f = ", omega / (2 * pi),
                                 " Hz (an undamped resonance of a closed structure?)"));
    }
  };
  if (use_lu) lu_init();
  auto Asolve = [&](const VecC& rhs) -> VecC {
    if (use_lu) return lu.solve(rhs);
    VecC x = ldl_.solve(rhs);
    for (int it = 0; it < 2; ++it) {  // iterative refinement
      VecC r = rhs - A * x;
      if (r.norm() <= 1e-13 * rhs.norm()) break;
      x += ldl_.solve(r);
    }
    return x;
  };
  VecC b = VecC::Zero(nred_);
  VecC rf = VecC::Zero(nd);
  if (!rhs.empty()) {
    AXF_REQUIRE(static_cast<int>(rhs.size()) == nd, "internal: rhs size mismatch");
    for (int i = 0; i < nd; ++i) rf(i) = rhs[i];
    b += P_.transpose().cast<cplx>() * rf;
  }
  for (int m = 0; m < nm; ++m) if (!masters_[m].fixed) b(nfree_ + gidx_[m]) += masters_[m].current;
  VecC y = Asolve(b);
  if (!use_lu && (A * y - b).norm() > 1e-9 * std::max(b.norm(), 1e-300)) {
    use_lu = true;  // unstable without pivoting: redo with LU
    lu_init();
    y = lu.solve(b);
  }
  VecC mu = VecC::Zero(nc);
  if (nc > 0) {
    // A y + C mu = b, C^T y = 0
    Eigen::MatrixXcd Cr(nred_, nc);
    for (int c = 0; c < nc; ++c) Cr.col(c) = (P_.transpose() * cfull_[c]).cast<cplx>();
    Eigen::MatrixXcd Y(nred_, nc);
    for (int c = 0; c < nc; ++c) Y.col(c) = Asolve(Cr.col(c));
    Eigen::MatrixXcd S = Cr.transpose() * Y;
    mu = S.fullPivLu().solve(Cr.transpose() * y);
    y -= Y * mu;
  }
  const double rel_res = nc > 0 ? 0.0 : (A * y - b).norm() / std::max(b.norm(), 1e-300);
  VecC u = P_.cast<cplx>() * y;
  Solution s;
  s.omega = omega;
  s.u.assign(u.data(), u.data() + nd);
  s.psi.assign(nm, 0.0);
  s.current.assign(nm, 0.0);
  s.multiplier.assign(mu.data(), mu.data() + nc);
  for (int m = 0; m < nm; ++m) if (!masters_[m].fixed) s.psi[m] = y(nfree_ + gidx_[m]);
  // reactions on the bodies: sum over tied DOFs of w_i (A u + C mu - f)_i
  VecC g = K_.cast<cplx>() * u - omega * omega * (M_.cast<cplx>() * u) - rf;
  s.group_K.assign(ngr, 0.0);
  s.group_M.assign(ngr, 0.0);
  for (int gi = 0; gi < ngr; ++gi) {
    VecC Ku = Kg_[gi].cast<cplx>() * u, Mu = Mg_[gi].cast<cplx>() * u;
    g += ag[gi] * Ku - omega * omega * bg[gi] * Mu;
    s.group_K[gi] = u.dot(Ku).real();  // u^H K u (dot conjugates its first argument)
    s.group_M[gi] = u.dot(Mu).real();
  }
  VecC ub;
  if (U_.size() > 0) {
    ub.resize(bdofs_.size());
    for (size_t a = 0; a < bdofs_.size(); ++a) ub(a) = u(bdofs_[a]);
    VecC gb = D * ub;
    for (size_t a = 0; a < bdofs_.size(); ++a) g(bdofs_[a]) += gb(a);
    VecC pn = U_.transpose().cast<cplx>() * ub;
    for (int n = 0; n < pn.size(); ++n) s.dtn_im += coef(n).imag() * std::norm(pn(n));
  }
  for (int c = 0; c < nc; ++c) g += mu(c) * cfull_[c].cast<cplx>();
  for (int i = 0; i < nd; ++i) if (kind_[i] == 2) s.current[master_[i]] += w_[i] * g(i);
  s.diagnostics = {{"dofs", nd}, {"unknowns", nred_}, {"constraints", nc}, {"factorization", use_lu ? "LU" : "LDLT"},
                   {"boundary", radiation_ ? "radiation" : "closed"}, {"dtn_modes", U_.cols()},
                   {"kRb", k0 * Rb_}, {"seconds", timer.seconds()}};
  if (nc == 0) s.diagnostics["relative_residual"] = rel_res;
  return s;
}

std::vector<double> AxiWave::reaction(const std::vector<double>& u, double lambda) {
  setup(true);
  const long nd = ndof();
  AXF_REQUIRE(static_cast<long>(u.size()) == nd, "internal: reaction vector size mismatch");
  Eigen::Map<const Vec> x(u.data(), nd);
  Vec g = K_ * x - lambda * (M_ * x);
  for (size_t gi = 0; gi < groups_.size(); ++gi)
    g += lossless_alpha(groups_[gi]) * (Kg_[gi] * x) - lambda * lossless_beta(groups_[gi]) * (Mg_[gi] * x);
  return std::vector<double>(g.data(), g.data() + nd);
}

cplx AxiWave::value(const Solution& s, int e, const PointEval& pe) const {
  if (roles_[e].excluded) return 0.0;
  if (roles_[e].master) return pe.r > 0 ? s.psi[roles_[e].master_id] / pe.r : cplx(0.0);
  const auto& d = edofs(e);
  cplx a = 0;
  for (size_t i = 0; i < d.size(); ++i) a += s.u[d[i]] * pe.N[i];
  return a;
}

double AxiWave::value(const std::vector<double>& u, int e, const PointEval& pe) const {
  if (roles_[e].excluded || roles_[e].master) return 0.0;
  const auto& d = edofs(e);
  double a = 0;
  for (size_t i = 0; i < d.size(); ++i) a += u[d[i]] * pe.N[i];
  return a;
}

std::vector<Eigenmode> AxiWave::eigenmodes(int nev, json& diag, double target_omega) {
  Timer timer;
  setup(true);
  // lossy groups enter with their lossless parts (eps', mu' at high frequency; see lossless_alpha)
  for (size_t gi = 0; gi < groups_.size(); ++gi) {
    Kr_ += lossless_alpha(groups_[gi]) * Kgr_[gi];
    Mr_ += lossless_beta(groups_[gi]) * Mgr_[gi];
  }
  const int nc = static_cast<int>(cfull_.size());
  // Default: a shift below the spectrum keeps K - sigma M positive definite (the constraints remove near-static
  // modes). Targeted: sigma = target_omega^2 inside the spectrum; K - sigma M is indefinite, so the LDL^T
  // without pivoting is checked on a probe solve and replaced by a pivoting sparse LU when it is inaccurate.
  const bool targeted = target_omega > 0;
  const double sigma = targeted ? target_omega * target_omega : -std::pow(0.1 * units::c / Rb_, 2);
  SpMat As = Kr_ - sigma * Mr_;
  As.makeCompressed();
  Eigen::SimplicialLDLT<SpMat> ch;
  Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> lu;
  bool use_lu = false;
  ch.compute(As);
  if (ch.info() != Eigen::Success) {
    AXF_REQUIRE(targeted, "the closed eigenproblem could not be factorized");
    use_lu = true;
  } else if (targeted) {
    std::mt19937 rng(20261021u);
    std::normal_distribution<double> nd;
    Vec b(nred_);
    for (int i = 0; i < nred_; ++i) b(i) = nd(rng);
    const Vec y = ch.solve(b);
    const double rel = (As * y - b).norm() / b.norm();
    if (!(rel < 1e-10)) use_lu = true;
  }
  if (use_lu) {
    lu.analyzePattern(As);
    lu.factorize(As);
    AXF_REQUIRE(lu.info() == Eigen::Success,
                strcat("K - sigma M is singular at the target ", target_omega / (2 * pi),
                       " Hz (the target coincides with an eigenfrequency to round-off)"),
                "/fullwave/target_f_Hz", "Move target_f_Hz slightly.");
  }
  auto fsolve = [&](const auto& b) -> Eigen::MatrixXd {
    if (use_lu) return lu.solve(b);
    return ch.solve(b);
  };
  Eigen::MatrixXd Cr(nred_, nc), Y, S;
  for (int c = 0; c < nc; ++c) Cr.col(c) = P_.transpose() * cfull_[c];
  Eigen::FullPivLU<Eigen::MatrixXd> slu;
  if (nc > 0) {
    Y = fsolve(Cr);
    S = Cr.transpose() * Y;
    slu.compute(S);
  }
  auto inv = [&](const Vec& b) {
    Vec y = fsolve(b);
    if (nc > 0) y -= Y * slu.solve(Cr.transpose() * y);
    return y;
  };
  json ld;
  auto modes = lanczos(Mr_, inv, sigma, nev, ld, !targeted);
  // true residual of the constrained pencil: (K - lambda M) x must lie in span(C)
  Eigen::FullPivLU<Eigen::MatrixXd> clu;
  if (nc > 0) clu.compute(Cr.transpose() * Cr);
  for (auto& m : modes) {
    Eigen::Map<const Vec> x(m.x.data(), static_cast<long>(m.x.size()));
    const Vec Kx = Kr_ * x;
    Vec r = Kx - m.omega * m.omega * (Mr_ * x);
    if (nc > 0) r -= Cr * clu.solve(Cr.transpose() * r);
    m.residual_true = r.norm() / std::max(Kx.norm(), 1e-300);
  }
  for (auto& m : modes) {  // reduced -> all DOFs (the reduction changes with the boundary setup)
    Vec u = P_ * Eigen::Map<const Vec>(m.x.data(), static_cast<long>(m.x.size()));
    m.x.assign(u.data(), u.data() + u.size());
  }
  diag = ld;
  diag["shift_rad2_s2"] = sigma;
  diag["constraints"] = nc;
  diag["factorization"] = use_lu ? "LU" : "LDLT";
  if (targeted) diag["target_f_Hz"] = target_omega / (2 * pi);
  diag["seconds"] = timer.seconds();
  if (!groups_.empty()) diag["note_lossy"] = "lossy materials enter with their lossless permittivity/permeability";
  setup_closed_ = -1;  // the reduced matrices were modified: rebuild for the next solve
  return modes;
}

}  // namespace axf::axi
