// Axisymmetric (r, z) Stokes integrators with swirl, on a 2D mesh with x = r, y = z (formulation:
// docs/guides/fluids.md). Velocity components are ordered (u_r, u_z) or (u_r, u_z, u_theta).
//
//   AxiViscousIntegrator      a(u,v) = int nu s(u).s(v) r dr dz + gamma int div_r u div_r v r dr dz with
//     Laplacian form:  s = [d_r u_r, d_z u_r, d_r u_z, d_z u_z, u_r/r, d_r u_t, d_z u_t, u_t/r]
//     stress form:     s = sqrt(2) [e_rr, e_zz, e_tt, sqrt2 e_rz, sqrt2 e_rt, sqrt2 e_zt]
//                      e_rr = d_r u_r, e_zz = d_z u_z, e_tt = u_r/r, e_rz = (d_z u_r + d_r u_z)/2,
//                      e_rt = (d_r u_t - u_t/r)/2, e_zt = d_z u_t / 2
//     grad-div:        div_r u = d_r u_r + u_r/r + d_z u_z, gamma in viscosity units (not scaled by nu), assembled
//                      with the SAME rule as the viscous (1/r) terms
//   AxiDivergenceIntegrator   b(u,q) = -int q (r d_r u_r + u_r + r d_z u_z) dr dz    (continuity tested by r q)
//
// The hoop terms (u_r/r, u_t/r) are singular at r = 0: u_r = u_t = 0 must be imposed strongly on the axis and the
// axis rows eliminated. Quadrature: an explicit rule order (IntRules.Get(geometry, order)); every
// point must have r > 0 (checked: a point on the axis is an error, never a silent division by zero).
#pragma once

#include <mfem.hpp>

namespace axf::fluids {

class AxiViscousIntegrator : public mfem::BilinearFormIntegrator {
 public:
  // order <= 0: 2 p + OrderW + 1 (the default)
  AxiViscousIntegrator(mfem::Coefficient& nu, int vdim, bool stress_form, double grad_div = 0.0, int order = 0)
      : nu_(nu), vdim_(vdim), stress_(stress_form), gamma_(grad_div), order_(order) {}
  void AssembleElementMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             mfem::DenseMatrix& elmat) override;

 private:
  mfem::Coefficient& nu_;
  int vdim_;
  bool stress_;
  double gamma_;
  int order_;
};

class AxiDivergenceIntegrator : public mfem::BilinearFormIntegrator {
 public:
  explicit AxiDivergenceIntegrator(int vdim, int order = 0) : vdim_(vdim), order_(order) {}
  // trial: vector H1 (vdim components, element vdofs in component blocks); test: scalar H1
  void AssembleElementMatrix2(const mfem::FiniteElement& trial, const mfem::FiniteElement& test,
                              mfem::ElementTransformation& T, mfem::DenseMatrix& elmat) override;

 private:
  int vdim_;
  int order_;
};

// The weight r of the axisymmetric measure at a point of T (element or boundary element): the radius at the point.
double axi_weight_r(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

// value = the weight r of the axisymmetric measure (axi_weight_r)
class RadiusWeight : public mfem::Coefficient {
 public:
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override { return axi_weight_r(T, ip); }
};

// r-weighted coefficient wrapper: value = r * c(x) (r = x-coordinate; the measure weight axi_weight_r)
class RadiusTimes : public mfem::Coefficient {
 public:
  explicit RadiusTimes(mfem::Coefficient& c, bool divide = false) : c_(c), divide_(divide) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Coefficient& c_;
  bool divide_;
};

}  // namespace axf::fluids
