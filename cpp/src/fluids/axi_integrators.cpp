#include "axionfem/fluids/axi_integrators.hpp"

#include <cmath>

#include "axionfem/core.hpp"

namespace axf::fluids {

namespace {
// the radius at a quadrature point; a point on (or left of) the axis is refused: the hoop integrands divide by r
double radius_at(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
  double x[3];
  mfem::Vector X(x, T.GetSpaceDim());
  T.Transform(ip, X);
  AXF_REQUIRE(x[0] > 0.0,
              axf::strcat("axisymmetric quadrature point at r = ", x[0], " (element ", T.ElementNo,
                          "): the 1/r hoop terms need every point strictly inside r > 0"),
              "axisymmetric assembly", "Use a quadrature rule without points on the element boundary.");
  return x[0];
}

const mfem::IntegrationRule& rule(const mfem::FiniteElement& el, mfem::ElementTransformation& T, int order,
                                  int fallback) {
  return mfem::IntRules.Get(el.GetGeomType(), order > 0 ? order : fallback + T.OrderW());
}
}  // namespace

double axi_weight_r(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
  double x[3];
  mfem::Vector X(x, T.GetSpaceDim());
  T.Transform(ip, X);
  return x[0];
}

double RadiusTimes::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
  double x[3];
  mfem::Vector X(x, T.GetSpaceDim());
  T.Transform(ip, X);
  const double c = c_.Eval(T, ip);
  return divide_ ? c / x[0] : c * x[0];
}

void AxiViscousIntegrator::AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                                                 mfem::DenseMatrix& elmat) {
  const int nd = el.GetDof();
  MFEM_VERIFY(vdim_ == 2 || vdim_ == 3, "AxiViscousIntegrator: vdim must be 2 or 3");
  const bool stress = stress_;
  const int nc = vdim_ * nd;
  const int ns = stress ? (vdim_ == 3 ? 6 : 4) : (vdim_ == 3 ? 8 : 5);
  elmat.SetSize(nc);
  elmat = 0.0;
  mfem::Vector shape(nd), D(nc);
  mfem::DenseMatrix dshape(nd, 2), S(ns, nc), StS(nc);
  const mfem::IntegrationRule& ir = rule(el, T, order_, 2 * el.GetOrder() + 1);  // +1: 1/r is not polynomial
  const double s2 = std::sqrt(2.0);
  const int ur = 0, uz = nd, ut = 2 * nd;  // component offsets in the element vdofs
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    const auto& ip = ir.IntPoint(q);
    T.SetIntPoint(&ip);
    el.CalcShape(ip, shape);
    el.CalcPhysDShape(T, dshape);
    const double r = radius_at(T, ip);
    const double w0 = ip.weight * T.Weight() * r;
    const double nu = nu_.Eval(T, ip);
    S = 0.0;
    D = 0.0;
    for (int j = 0; j < nd; ++j) {
      const double N = shape(j), Nr = dshape(j, 0), Nz = dshape(j, 1);
      D(ur + j) = Nr + N / r;
      D(uz + j) = Nz;
      if (!stress) {
        S(0, ur + j) = Nr;
        S(1, ur + j) = Nz;
        S(2, uz + j) = Nr;
        S(3, uz + j) = Nz;
        S(4, ur + j) = N / r;
        if (vdim_ == 3) {
          S(5, ut + j) = Nr;
          S(6, ut + j) = Nz;
          S(7, ut + j) = N / r;
        }
      } else {
        S(0, ur + j) = s2 * Nr;                 // sqrt2 e_rr
        S(1, uz + j) = s2 * Nz;                 // sqrt2 e_zz
        S(2, ur + j) = s2 * N / r;              // sqrt2 e_tt
        S(3, ur + j) = Nz;                      // 2 e_rz = d_z u_r + d_r u_z
        S(3, uz + j) = Nr;
        if (vdim_ == 3) {
          S(4, ut + j) = Nr - N / r;  // 2 e_rt
          S(5, ut + j) = Nz;          // 2 e_zt
        }
      }
    }
    mfem::MultAtB(S, S, StS);
    elmat.Add(w0 * nu, StS);
    const double g = gamma_;
    if (g != 0.0) AddMult_a_VVt(w0 * g, D, elmat);
  }
}

void AxiDivergenceIntegrator::AssembleElementMatrix2(const mfem::FiniteElement& trial,
                                                     const mfem::FiniteElement& test, mfem::ElementTransformation& T,
                                                     mfem::DenseMatrix& elmat) {
  const int nu = trial.GetDof(), np = test.GetDof();
  elmat.SetSize(np, vdim_ * nu);
  elmat = 0.0;
  mfem::Vector shu(nu), shp(np);
  mfem::DenseMatrix dsh(nu, 2);
  const mfem::IntegrationRule& ir =
      order_ > 0 ? mfem::IntRules.Get(trial.GetGeomType(), order_)
                 : mfem::IntRules.Get(trial.GetGeomType(), trial.GetOrder() + test.GetOrder() + T.OrderW() + 1);
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    const auto& ip = ir.IntPoint(q);
    T.SetIntPoint(&ip);
    trial.CalcShape(ip, shu);
    trial.CalcPhysDShape(T, dsh);
    test.CalcShape(ip, shp);
    const double r = radius_at(T, ip);
    // continuity tested with r q: r (d_r u_r + u_r / r + d_z u_z)
    const double w = ip.weight * T.Weight();
    for (int i = 0; i < np; ++i)
      for (int j = 0; j < nu; ++j) {
        elmat(i, j) -= w * shp(i) * (r * dsh(j, 0) + shu(j));  // u_r block
        elmat(i, nu + j) -= w * shp(i) * r * dsh(j, 1);        // u_z block
      }
  }
}

}  // namespace axf::fluids
