// The convection kernel: see cpp/include/axionfem/fluids/convection.hpp.
#include "axionfem/fluids/convection.hpp"

#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/fluids/axi_integrators.hpp"

namespace axf::fluids {

const char* to_string(ConvectiveForm f) { return f == ConvectiveForm::Skew ? "skew" : "standard"; }

namespace {

// a velocity field at one point: component values and their derivatives (d_x, d_y) = (d_r, d_z)
struct PF {
  double v[3] = {0, 0, 0};
  double g[3][2] = {{0, 0}, {0, 0}, {0, 0}};
};

// the coefficients of one assembly
struct Kernel {
  double rho = 1, kappa = 0;      // kappa: coefficient of (div w)(u.v); 1/2 for Temam's form
  bool axi = false;
  bool z3 = false;                // planar 2.5D: a third, out-of-plane component u_z (z-invariant)

  // t (tested with v) and S (tested with grad v) of c(w; u, v) at one point, with the measure weight rw (r in
  // axisymmetry, 1 planar) and the exact radius r (hoop factors).
  void density(const PF& w, const PF& u, double rw, double r, double t[3], double S[3][2]) const {
    for (int i = 0; i < 3; ++i) t[i] = S[i][0] = S[i][1] = 0.0;
    double adv[3] = {0, 0, 0}, divw = 0;
    if (!axi) {
      const int nc = z3 ? 3 : 2;  // components; the derivatives are always (d_x, d_y)
      for (int i = 0; i < nc; ++i)
        for (int j = 0; j < 2; ++j) adv[i] += u.g[i][j] * w.v[j];
      divw = w.g[0][0] + w.g[1][1];
      for (int i = 0; i < nc; ++i) {
        t[i] = rho * adv[i];
        t[i] += rho * kappa * divw * u.v[i];
      }
      return;
    }
    // axisymmetric, internal components (r, z, theta) = (0, 1, 2); derivative columns (d_r, d_z)
    const double hr = rw / r;  // weight times the hoop factor 1/r
    adv[0] = rw * (w.v[0] * u.g[0][0] + w.v[1] * u.g[0][1]) - hr * w.v[2] * u.v[2];
    adv[1] = rw * (w.v[0] * u.g[1][0] + w.v[1] * u.g[1][1]);
    adv[2] = rw * (w.v[0] * u.g[2][0] + w.v[1] * u.g[2][1]) + hr * w.v[2] * u.v[0];
    divw = rw * (w.g[0][0] + w.g[1][1]) + hr * w.v[0];
    for (int i = 0; i < 3; ++i) t[i] = rho * (adv[i] + kappa * divw * u.v[i]);
  }
};

Kernel make_kernel(double rho, bool axi, bool z3, ConvectiveForm form, bool other_form) {
  Kernel k;
  k.axi = axi;
  k.z3 = z3;
  k.rho = rho;
  bool skew = form == ConvectiveForm::Skew;
  if (other_form) skew = !skew;
  k.kappa = skew ? 0.5 : 0.0;
  return k;
}

double radius_at_point(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
  double x[3];
  mfem::Vector X(x, T.GetSpaceDim());
  T.Transform(ip, X);
  AXF_REQUIRE(x[0] > 0.0,
              axf::strcat("axisymmetric convection: quadrature point at r = ", x[0], " (element ", T.ElementNo,
                          "): the hoop terms need every point strictly inside r > 0"),
              "axisymmetric assembly");
  return x[0];
}

}  // namespace

ConvectionIntegrator::ConvectionIntegrator(double rho, int vdim, bool axisymmetric, ConvectiveForm form, int order)
    : rho_(rho), vdim_(vdim), axi_(axisymmetric), form_(form), order_(order) {
  AXF_REQUIRE(std::isfinite(rho) && rho >= 0, axf::strcat("the density must be non-negative and finite (got ", rho, ")"),
              "rho");  // rho = 0 (a Stokes-like operator with exactly zero convection) is offered with MHD only (NavierStokesSolver checks)
  AXF_REQUIRE(axi_ ? vdim_ == 3 : (vdim_ == 2 || vdim_ == 3),
              "ConvectionIntegrator: planar needs 2 velocity components (3 with the out-of-plane u_z), axisymmetric 3 "
              "(with swirl)");
}

int ConvectionIntegrator::quad_order(int p) const {
  if (order_ > 0) return order_;
  return axi_ ? 3 * p : 3 * p - 1;  // exact on affine elements: P2 . grad P2 . P2 (x r)
}

void ConvectionIntegrator::AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                                                 const mfem::Vector& elfun, mfem::Vector& elvect) {
  const int nd = el.GetDof(), vd = vdim_;
  const Kernel K = make_kernel(rho_, axi_, vdim_ == 3 && !axi_, form_, false);
  elvect.SetSize(vd * nd);
  elvect = 0.0;
  mfem::Vector N(nd);
  mfem::DenseMatrix dN(nd, 2);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), quad_order(el.GetOrder()));
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
    T.SetIntPoint(&ip);
    el.CalcShape(ip, N);
    el.CalcPhysDShape(T, dN);
    PF u;
    for (int c = 0; c < vd; ++c)
      for (int a = 0; a < nd; ++a) {
        const double x = elfun(c * nd + a);
        u.v[c] += x * N(a);
        u.g[c][0] += x * dN(a, 0);
        u.g[c][1] += x * dN(a, 1);
      }
    double r = 1, rw = 1;
    if (axi_) {
      r = radius_at_point(T, ip);
      rw = axi_weight_r(T, ip);
    }
    double t[3], S[3][2];
    K.density(u, u, rw, r, t, S);
    const double W = ip.weight * T.Weight();
    for (int c = 0; c < vd; ++c)
      for (int a = 0; a < nd; ++a) elvect(c * nd + a) += W * (t[c] * N(a) + S[c][0] * dN(a, 0) + S[c][1] * dN(a, 1));
  }
}

void ConvectionIntegrator::AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                                               const mfem::Vector& elfun, mfem::DenseMatrix& elmat) {
  const int nd = el.GetDof(), vd = vdim_;
  const Kernel K = make_kernel(rho_, axi_, vdim_ == 3 && !axi_, form_, false);
  // which halves enter (Newton: O + Q; Picard: O)
  const double cQ = mode_ == GradMode::Newton ? 1.0 : 0.0;
  elmat.SetSize(vd * nd);
  elmat = 0.0;
  mfem::Vector N(nd);
  mfem::DenseMatrix dN(nd, 2);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), quad_order(el.GetOrder()));
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
    T.SetIntPoint(&ip);
    el.CalcShape(ip, N);
    el.CalcPhysDShape(T, dN);
    PF u;
    for (int c = 0; c < vd; ++c)
      for (int a = 0; a < nd; ++a) {
        const double x = elfun(c * nd + a);
        u.v[c] += x * N(a);
        u.g[c][0] += x * dN(a, 0);
        u.g[c][1] += x * dN(a, 1);
      }
    double r = 1, rw = 1;
    if (axi_) {
      r = radius_at_point(T, ip);
      rw = axi_weight_r(T, ip);
    }
    const double W = ip.weight * T.Weight();
    for (int k = 0; k < vd; ++k)
      for (int b = 0; b < nd; ++b) {
        PF phi;
        phi.v[k] = N(b);
        phi.g[k][0] = dN(b, 0);
        phi.g[k][1] = dN(b, 1);
        double tO[3], SO[3][2], tQ[3], SQ[3][2];
        K.density(u, phi, rw, r, tO, SO);   // O(u) phi = c(u; phi, .)
        K.density(phi, u, rw, r, tQ, SQ);   // Q(u) phi = c(phi; u, .)
        const int col = k * nd + b;
        for (int c = 0; c < vd; ++c) {
          const double t = tO[c] + cQ * tQ[c];
          const double s0 = SO[c][0] + cQ * SQ[c][0], s1 = SO[c][1] + cQ * SQ[c][1];
          for (int a = 0; a < nd; ++a) elmat(c * nd + a, col) += W * (t * N(a) + s0 * dN(a, 0) + s1 * dN(a, 1));
        }
      }
  }
}

}  // namespace axf::fluids
