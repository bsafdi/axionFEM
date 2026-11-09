#include "axionfem/hts.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "axionfem/core.hpp"
#include "axionfem/fullwave.hpp"  // assoc_legendre_p1 (the DtN angular basis)
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using units::mu0;
using units::pi;

// ---------------------------------------------------------------- constitutive law
double HTSMaterial::Jc_eng(double Br, double Bz) const {
  const double bn = Br * normal_r + Bz * normal_z;
  const double b2 = Br * Br + Bz * Bz;
  const double bpar = std::sqrt(std::max(b2 - bn * bn, 0.0));
  return fill * jc->Jc(std::fabs(bn), bpar, T_op);
}

// Power law with a minimum resistivity: J = Jc (|E|/E0)^(1/n) sign(E) for |E| >= e_floor E0 and the
// linear continuation J = sigma_max E below it, sigma_max = (Jc/E0) e_floor^(1/n - 1).  The pure
// inverse power law has an infinite slope at E = 0, which in the A-formulation turns the (unavoidable)
// round-off of E inside a flux-free region into a current of order Jc; the floor removes that noise
// and keeps the Jacobian exact.  See docs/validation/hts.md.
double HTSMaterial::J_of_E(double E, double Jc_e) const {
  if (E == 0.0 || Jc_e <= 0.0) return 0.0;
  const double x = std::fabs(E) / E0;
  if (x >= e_floor) return (E > 0 ? Jc_e : -Jc_e) * std::pow(x, 1.0 / n);
  return Jc_e * std::pow(e_floor, 1.0 / n - 1.0) * E / E0;
}

double HTSMaterial::dJdE(double E, double Jc_e) const {
  if (Jc_e <= 0.0) return 0.0;
  const double x = std::fabs(E) / E0;
  if (x >= e_floor) return Jc_e / (n * E0) * std::pow(x, 1.0 / n - 1.0);
  return Jc_e / E0 * std::pow(e_floor, 1.0 / n - 1.0);
}

// int_0^E J(e) de: convex in E because J is monotone, which makes the backward-Euler step the
// minimizer of a convex functional (used as the line-search merit function).
double HTSMaterial::psi_of_E(double E, double Jc_e) const {
  if (Jc_e <= 0.0) return 0.0;
  const double x = std::fabs(E) / E0, p = 1.0 + 1.0 / n;
  if (x < e_floor) return 0.5 * Jc_e * E0 * std::pow(e_floor, 1.0 / n - 1.0) * x * x;
  // continuity at x = e_floor fixes the additive constant
  return Jc_e * E0 * (std::pow(x, p) / p + (0.5 - 1.0 / p) * std::pow(e_floor, p));
}

// ---------------------------------------------------------------- setup
CriticalState::CriticalState(const Space& sp, std::vector<HTSRole> roles, std::vector<HTSMaterial> mats,
                             std::vector<Master> masters)
    : sp_(sp), roles_(std::move(roles)), mats_(std::move(mats)), masters_(std::move(masters)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: HTS role table size mismatch");
}

void CriticalState::set_open_boundary(int outer_attr, double radius, int modes) {
  outer_attr_ = outer_attr;
  Rb_ = radius;
  modes_ = modes;
}

void CriticalState::set_background(std::vector<double> psi_at_dof, std::function<double(double, double)> A_bg,
                                   std::function<void(double, double, double&, double&)> B_bg) {
  psi_bg_ = std::move(psi_at_dof);
  A_bg_ = std::move(A_bg);
  B_bg_ = std::move(B_bg);
}

Vec CriticalState::u_of(const Vec& y, double scale) const {
  Vec u = P_ * y + u_fixed_;
  if (!psi_bg_.empty()) {
    for (int i = 0; i < static_cast<int>(kind_.size()); ++i)
      if (kind_[i] == 2) u(i) -= scale * psi_bg_[i] * w_[i];  // ties hold the TOTAL flux function
  }
  return u;
}

void CriticalState::setup() {
  const int nd = sp_.ndof(), ne = sp_.mesh.GetNE();
  const int nm = static_cast<int>(masters_.size());
  kind_.assign(nd, 0);
  master_.assign(nd, -1);
  w_.assign(nd, 0.0);
  index_.assign(nd, -1);
  for (int e = 0; e < ne; ++e)
    if (roles_[e].master)
      for (int d : sp_.edofs(e)) {
        AXF_REQUIRE(master_[d] < 0 || master_[d] == roles_[e].master_id,
                    "superconducting bodies touch in the critical-state problem", "/superconductors");
        master_[d] = roles_[e].master_id;
      }
  std::vector<char> used(nd, 0);
  for (int e = 0; e < ne; ++e)
    if (!roles_[e].master)
      for (int d : sp_.edofs(e)) used[d] = 1;
  nfree_ = 0;
  for (int i = 0; i < nd; ++i) {
    if (sp_.r[i] == 0.0) {
      kind_[i] = 3;
      AXF_REQUIRE(master_[i] < 0 || (masters_[master_[i]].fixed && masters_[master_[i]].psi == 0.0),
                  "superconducting body '" + (master_[i] < 0 ? std::string() : masters_[master_[i]].name) +
                      "' touches the symmetry axis and cannot hold a nonzero flux",
                  "/superconductors");
    } else if (master_[i] >= 0) { kind_[i] = 2; w_[i] = 1.0 / sp_.r[i]; }
    else if (used[i]) { kind_[i] = 1; index_[i] = nfree_++; }
  }
  gidx_.assign(nm, -1);
  ng_ = 0;
  for (int m = 0; m < nm; ++m)
    if (!masters_[m].fixed) gidx_[m] = ng_++;
  nred_ = nfree_ + ng_;
  AXF_REQUIRE(nfree_ > 0, "the critical-state problem has no free degrees of freedom");

  std::vector<Trip> pt;
  u_fixed_ = Vec::Zero(nd);
  for (int i = 0; i < nd; ++i) {
    if (kind_[i] == 1) pt.emplace_back(i, index_[i], 1.0);
    else if (kind_[i] == 2) {
      const int m = master_[i];
      if (masters_[m].fixed) u_fixed_(i) = masters_[m].psi * w_[i];
      else pt.emplace_back(i, nfree_ + gidx_[m], w_[i]);
    }
  }
  P_.resize(nd, nred_);
  P_.setFromTriplets(pt.begin(), pt.end());
  Iext_ = Vec::Zero(nred_);
  for (int m = 0; m < nm; ++m)
    if (!masters_[m].fixed) Iext_(nfree_ + gidx_[m]) = masters_[m].current;

  // ---- stiffness (constant: all media outside the HTS regions are linear) and source vector
  const RefTable& tab = sp_.table(sp_.quad_rule());
  std::vector<Trip> kt;
  F_ = Vec::Zero(nd);
  PointEval pe;
  for (int e = 0; e < ne; ++e) {
    if (roles_[e].master) continue;
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(n, n);
    for (int q = 0; q < tab.nq; ++q) {
      sp_.eval(e, tab, q, pe);
      const double r = pe.r, wr = pe.w * r, nu = roles_[e].nu;
      std::vector<double> br(n), bz(n);
      for (int i = 0; i < n; ++i) {
        br[i] = -pe.dN[2 * i + 1];
        bz[i] = pe.dN[2 * i] + pe.N[i] / r;
        F_(d[i]) += wr * roles_[e].J * pe.N[i];
      }
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Ke(i, j) += wr * nu * (br[i] * br[j] + bz[i] * bz[j]);
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) kt.emplace_back(d[i], d[j], Ke(i, j));
  }
  // ---- exact open boundary (same spherical-harmonic DtN map as the static poloidal problem)
  if (outer_attr_ > 0 && modes_ > 0) {
    std::map<int, int> bidx;
    std::vector<int> bdofs;
    mfem::Array<int> bd;
    int nbe = 0;
    for (int be = 0; be < sp_.mesh.GetNBE(); ++be) {
      if (sp_.mesh.GetBdrAttribute(be) != outer_attr_) continue;
      ++nbe;
      sp_.fes->GetBdrElementDofs(be, bd);
      for (int d : bd)
        if (!bidx.count(d)) { bidx[d] = static_cast<int>(bdofs.size()); bdofs.push_back(d); }
    }
    const int N = std::min(modes_, std::max(4, nbe));
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(bdofs.size(), N);
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
        const double theta = std::atan2(x(0), x(1)), dl = ip.weight * T->Weight();
        assoc_legendre_p1(std::cos(theta), N, Pn);
        for (int i = 0; i < bd.Size(); ++i) {
          const int bi = bidx[bd[i]];
          const double c = shape(i) * x(0) * dl / Rb_;
          for (int n = 1; n <= N; ++n) U(bi, n - 1) += c * Pn[n];
        }
      }
    }
    Vec coef(N);
    for (int n = 1; n <= N; ++n) coef(n - 1) = (1.0 / (mu0 * Rb_)) * n * (2.0 * n + 1.0) / (2.0 * n * (n + 1.0));
    Eigen::MatrixXd D = U * coef.asDiagonal() * U.transpose();
    for (size_t a = 0; a < bdofs.size(); ++a)
      for (size_t b = 0; b < bdofs.size(); ++b) kt.emplace_back(bdofs[a], bdofs[b], D(a, b));
  }
  K_.resize(nd, nd);
  K_.setFromTriplets(kt.begin(), kt.end());
  Kr_ = (P_.transpose() * K_ * P_).pruned();
  lin_.compute(Kr_);
  AXF_REQUIRE(lin_.info() == Eigen::Success,
              "the critical-state field system could not be factorized (singular or indefinite)", "",
              "A floating region without a reference state, or a zero-volume region.");

  // ---- HTS elements and the background values at their quadrature points
  hts_elems_.clear();
  for (int e = 0; e < ne; ++e)
    if (!roles_[e].master && roles_[e].hts >= 0) hts_elems_.push_back(e);
  q_Abg_.assign(hts_elems_.size() * tab.nq, 0.0);
  q_Brbg_.assign(hts_elems_.size() * tab.nq, 0.0);
  q_Bzbg_.assign(hts_elems_.size() * tab.nq, 0.0);
  if (A_bg_ || B_bg_) {
    size_t qi = 0;
    for (int e : hts_elems_)
      for (int q = 0; q < tab.nq; ++q, ++qi) {
        sp_.eval(e, tab, q, pe);
        if (A_bg_) q_Abg_[qi] = A_bg_(pe.r, pe.z);
        if (B_bg_) B_bg_(pe.r, pe.z, q_Brbg_[qi], q_Bzbg_[qi]);
      }
  }
  ready_ = true;
}

// ---------------------------------------------------------------- nonlinear (power-law) term
void CriticalState::assemble_nonlinear(const State& prev, const Vec& du, bool want_K, Vec& gh,
                                       std::vector<Trip>& trip, double dt, double dscale, double* loss,
                                       double* work_bg, double* potential) const {
  gh = Vec::Zero(sp_.ndof());
  trip.clear();
  if (loss) *loss = 0;
  if (work_bg) *work_bg = 0;
  if (potential) *potential = 0;
  const RefTable& tab = sp_.table(sp_.quad_rule());
  const double scale = prev.scale + dscale;
  PointEval pe;
  size_t qi = 0;
  for (int e : hts_elems_) {
    const HTSMaterial& M = mats_[roles_[e].hts];
    const auto& d = sp_.edofs(e);
    const int n = static_cast<int>(d.size());
    for (int q = 0; q < tab.nq; ++q, ++qi) {
      sp_.eval(e, tab, q, pe);
      const double r = pe.r, wr = pe.w * r;
      double dA = 0, Br = 0, Bz = 0;
      for (int i = 0; i < n; ++i) {
        const double ui = prev.u(d[i]) + du(d[i]);
        dA += du(d[i]) * pe.N[i];
        Br -= ui * pe.dN[2 * i + 1];
        Bz += ui * (pe.dN[2 * i] + pe.N[i] / r);
      }
      Br += scale * q_Brbg_[qi];
      Bz += scale * q_Bzbg_[qi];
      const double dA_tot = dA + dscale * q_Abg_[qi];
      const double E = -dA_tot / dt;
      const double Jc_e = M.Jc_eng(Br, Bz);
      const double J = M.J_of_E(E, Jc_e);
      for (int i = 0; i < n; ++i) gh(d[i]) -= wr * J * pe.N[i];
      if (loss) *loss -= wr * J * dA_tot;                         // = int E J dt  (per radian)
      if (work_bg) *work_bg -= wr * J * dscale * q_Abg_[qi];      // work of the background source
      if (potential) *potential += wr * dt * M.psi_of_E(E, Jc_e);
      if (!want_K) continue;
      const double sig = M.dJdE(E, Jc_e) / dt;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) trip.emplace_back(d[i], d[j], wr * sig * pe.N[i] * pe.N[j]);
    }
  }
}

// ---------------------------------------------------------------- initial state
CriticalState::State CriticalState::start(double t0, double scale0, bool penetrated) {
  AXF_REQUIRE(ready_, "internal: call CriticalState::setup() before start()");
  AXF_REQUIRE(penetrated || scale0 == 0.0,
              "a zero-field-cooled critical-state run must start at drive scale 0", "/hts",
              "Start the waveform at 0 and ramp up, or use \"initial\": \"field_cooled\".");
  State s;
  s.t = t0;
  s.scale = scale0;
  // Linear solution with the HTS regions transparent: no screening currents (field-cooled); at
  // scale 0 this is also the zero-field-cooled state.
  Vec u0 = u_of(Vec::Zero(nred_), scale0);
  Vec rhs = scale0 * Vec(P_.transpose() * F_) + Iext_ - Vec(P_.transpose() * (K_ * u0));
  s.y = lin_.solve(rhs);
  s.u = u_of(s.y, scale0);
  s.du = Vec::Zero(sp_.ndof());
  s.dy = Vec::Zero(nred_);
  s.W_J = pi * s.u.dot(K_ * s.u);
  return s;
}

// ---------------------------------------------------------------- one backward-Euler step
CriticalState::StepInfo CriticalState::advance(State& s, double t_new, double scale_new, int verbose) {
  AXF_REQUIRE(ready_, "internal: call CriticalState::setup() before advance()");
  const double dt = t_new - s.t, dscale = scale_new - s.scale;
  AXF_REQUIRE(dt > 0, "critical-state time steps must advance");
  const int nd = sp_.ndof();
  const Vec Kuprev = K_ * s.u;
  Vec du0 = Vec::Zero(nd);
  if (!psi_bg_.empty())
    for (int i = 0; i < nd; ++i)
      if (kind_[i] == 2) du0(i) = -dscale * psi_bg_[i] * w_[i];

  // Newton on the increment: E = -(du . N + dscale A_bg)/dt has no cancellation error.
  // The step is the minimizer of the convex functional
  //   Phi(du) = 1/2 u^T K u - scale F^T u - Iext^T y + dt int r Psi(E(du)),  Psi(E) = int_0^E J,
  // whose gradient is the residual; the line search uses Phi as its merit function (Armijo), which
  // makes the damped Newton iteration globally convergent for the monotone power law.
  Vec gh, g, du, grad;
  std::vector<Trip> trip;
  double loss = 0, work_bg = 0, pot = 0, phi = 0;
  StepInfo info;
  auto residual = [&](const Vec& yy, bool want_K) {
    du = P_ * yy + du0;
    assemble_nonlinear(s, du, want_K, gh, trip, dt, dscale, &loss, &work_bg, &pot);
    grad = Kuprev + K_ * du - scale_new * F_ + gh;
    g = P_.transpose() * grad - Iext_;
    const Vec u = s.u + du;
    phi = 0.5 * u.dot(K_ * u) - scale_new * F_.dot(u) - Iext_.dot(s.y + yy) + pot;
    return g.norm();
  };
  // Reference: the residual of "nothing changes", i.e. the driving force of this step.
  const double res_ref = std::max(residual(Vec::Zero(nred_), false), 1e-300);
  // Initial guess: extrapolate the previous increment when the drive keeps its regime.
  Vec dy = Vec::Zero(nred_);
  if (s.dy.size() == nred_ && s.dt > 0) {
    if (s.dscale != 0.0 && dscale != 0.0) dy = s.dy * (dscale / s.dscale);
    else if (s.dscale == 0.0 && dscale == 0.0) dy = s.dy * (dt / s.dt);
  }
  double res = residual(dy, true);
  if (res > res_ref) { dy.setZero(); res = residual(dy, true); }  // the extrapolation was worse
  int it = 0;
  // Convergence is measured on the Newton update, not on the residual: the Jacobian of the power law
  // spans many orders of magnitude (it is ~ sigma_max in flux-free regions), so a solution that is
  // accurate to round-off can still leave a residual of order 1e-2 of the step's driving force.  The
  // residual ratio and the exact backward-Euler energy identity are reported next to it.
  // The residual test is only an "essentially exact" early exit: res_ref is the residual of a frozen
  // field, which is enormous when the region screens, so newton_tol applied to it would accept an
  // extrapolated guess that is nowhere near the solution (the backward-Euler energy identity then
  // shows a percent-level error).  The update test is the one newton_tol controls.
  const double res_exact = 1e-10;
  double update = 0.0, last_step = 1.0;
  info.converged = res <= res_exact * res_ref;
  while (!info.converged && it < max_newton_) {
    SpMat S(nd, nd);
    S.setFromTriplets(trip.begin(), trip.end());
    const SpMat Jr = Kr_ + SpMat(P_.transpose() * S * P_);
    if (!chol_ready_) { chol_.analyzePattern(Jr); chol_ready_ = true; }
    chol_.factorize(Jr);
    if (chol_.info() != Eigen::Success) {
      chol_.compute(Jr);
      AXF_REQUIRE(chol_.info() == Eigen::Success, "the critical-state Jacobian could not be factorized");
    }
    const Vec dir = chol_.solve(-g);
    // error estimate of the current iterate: the Newton step relative to the total increment it predicts
    update = dir.norm() / std::max((dy + dir).norm(), 1e-300);
    const double phi0 = phi, slope = g.dot(dir);  // slope < 0 for a descent direction
    double step = std::min(1.0, 4.0 * last_step), rn = res;
    bool ok = false;
    for (int ls = 0; ls < 40; ++ls) {
      rn = residual(Vec(dy + step * dir), false);
      if (phi <= phi0 + 1e-4 * step * slope) { ok = true; break; }
      // in the round-off regime of Phi, accept a step that reduces the residual instead
      if (std::fabs(phi - phi0) <= 1e-13 * std::fabs(phi0) && rn < res) { ok = true; break; }
      step *= 0.5;
    }
    ++it;
    if (!ok) break;  // stalled
    last_step = step;
    dy += step * dir;
    res = residual(dy, true);
    if (verbose > 1)
      log(2, strcat("    hts newton ", it, " res/ref ", res / res_ref, " update ", update, " step ", step));
    // A tiny update after a collapsed line search is stagnation, not convergence (as in the static
    // Newton solver), so the update test requires a near-full step.
    info.converged = res <= res_exact * res_ref || (update <= newton_tol_ && step >= 0.25);
  }
  info.newton = it;
  info.converged_on = !info.converged ? "none" : (res <= res_exact * res_ref ? "residual" : "update");
  info.residual = res;
  info.rel_residual = res / res_ref;
  info.update = update;
  // quantities of the accepted increment (the last residual call already used dy)
  res = residual(dy, false);
  const Vec u_new = s.u + du;
  const double W_new = pi * u_new.dot(K_ * u_new);
  info.dW_J = W_new - s.W_J;
  info.loss_J = 2 * pi * loss;
  // The tie offsets of ideal superconductors move with the background field; the work the external
  // source does on them is du0^T grad (zero without a background or without tied bodies).  Including
  // it keeps both the reported input work and the backward-Euler identity exact.
  info.work_J = 2 * pi * (scale_new * F_.dot(du) + Iext_.dot(dy) + work_bg + du0.dot(grad));
  info.numerical_dissipation_J = pi * du.dot(K_ * du);
  const double sc = std::max({std::fabs(info.work_J), std::fabs(info.loss_J), std::fabs(info.dW_J), 1e-300});
  info.balance_rel = (info.work_J - info.loss_J - info.dW_J) / sc;
  info.be_balance_rel = (info.work_J - info.loss_J - info.dW_J - info.numerical_dissipation_J) / sc;

  s.y += dy;
  s.u = u_new;
  s.du = du;
  s.dy = dy;
  s.t = t_new;
  s.scale = scale_new;
  s.dt = dt;
  s.dscale = dscale;
  s.W_J = W_new;
  s.loss_J += info.loss_J;
  s.work_J += info.work_J;
  return info;
}

// ---------------------------------------------------------------- field evaluation
double CriticalState::master_psi(const State& s, int m) const {
  return masters_[m].fixed ? masters_[m].psi : s.y(nfree_ + gidx_[m]);
}

void CriticalState::field(const State& s, int e, const PointEval& pe, double& Br, double& Bz, double* A) const {
  if (roles_[e].master) {  // ideal superconductor: the tie holds the total flux function
    Br = Bz = 0;
    if (A) *A = pe.r > 0 ? master_psi(s, roles_[e].master_id) / pe.r : 0.0;
    return;
  }
  const auto& d = sp_.edofs(e);
  double a = 0, dr = 0, dz = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    const double u = s.u(d[i]);
    a += u * pe.N[i];
    dr += u * pe.dN[2 * i];
    dz += u * pe.dN[2 * i + 1];
  }
  Br = -dz;
  Bz = dr + (pe.r > 0 ? a / pe.r : dr);
  if (A) *A = a;
  if (s.scale != 0.0 && B_bg_) {
    double br = 0, bz = 0;
    B_bg_(pe.r, pe.z, br, bz);
    Br += s.scale * br;
    Bz += s.scale * bz;
  }
  if (A && s.scale != 0.0 && A_bg_) *A += s.scale * A_bg_(pe.r, pe.z);
}

double CriticalState::E_at(const State& s, int e, const PointEval& pe) const {
  if (s.dt <= 0 || roles_[e].master) return 0.0;
  const auto& d = sp_.edofs(e);
  double dA = 0;
  for (size_t i = 0; i < d.size(); ++i) dA += s.du(d[i]) * pe.N[i];
  if (s.dscale != 0.0 && A_bg_) dA += s.dscale * A_bg_(pe.r, pe.z);
  return -dA / s.dt;
}

double CriticalState::J_at(const State& s, int e, const PointEval& pe, double* Jc_eng) const {
  if (roles_[e].hts < 0 || roles_[e].master) {
    if (Jc_eng) *Jc_eng = 0;
    return 0.0;
  }
  const HTSMaterial& M = mats_[roles_[e].hts];
  double Br, Bz;
  field(s, e, pe, Br, Bz);
  const double Jc_e = M.Jc_eng(Br, Bz);
  if (Jc_eng) *Jc_eng = Jc_e;
  return M.J_of_E(E_at(s, e, pe), Jc_e);
}

}  // namespace axf::axi
