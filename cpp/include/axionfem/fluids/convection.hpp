// The convective term of steady incompressible Navier–Stokes: ONE element kernel for both geometries (planar;
// axisymmetric with swirl) and both convective forms, giving the residual, the Oseen (Picard) matrix and the Newton
// Jacobian.
//
//   standard  c(w; u, v) = rho ((w.grad) u, v)
//   skew      c(w; u, v) = rho ((w.grad) u, v) + (rho/2) ((div w) u, v)       (Temam; NO boundary term)
//
// Axisymmetric (r, z) = mesh (x, y), components in the internal order (u_r, u_z, u_theta), weight r:
//   r [(w.grad)u]_r = r (w_r d_r u_r + w_z d_z u_r) - w_t u_t     (centrifugal)
//   r [(w.grad)u]_t = r (w_r d_r u_t + w_z d_z u_t) + w_t u_r     (Coriolis)
//   r [(w.grad)u]_z = r (w_r d_r u_z + w_z d_z u_z),   r div w = r d_r w_r + w_r + r d_z w_z.
// Because v carries no derivative, c(w; u, v) = int t(w, u) . v with a pointwise "force density" t, trilinear in
// (w, u): the residual tests t(u, u), the Oseen matrix O(w) = t(w, phi) and the reaction part Q(u) = t(phi, u), so
// one routine (t) serves all three and the Newton Jacobian is O(u) + Q(u).
//
// Quadrature: the trilinear form is integrated exactly on affine elements by default (3p - 1 planar, 3p
// axisymmetric).
#pragma once

#include <mfem.hpp>

namespace axf::fluids {

enum class ConvectiveForm { Standard, Skew };
const char* to_string(ConvectiveForm f);

class ConvectionIntegrator : public mfem::NonlinearFormIntegrator {
 public:
  enum class GradMode {
    Newton,       // O(u) + Q(u): the exact Jacobian of the residual
    Oseen         // O(u): the Picard (Oseen) operator with the advecting velocity u
  };
  // vdim: 2 (planar) or 3 (axisymmetric with swirl); order <= 0: 3p - 1 planar, 3p axisymmetric
  ConvectionIntegrator(double rho, int vdim, bool axisymmetric, ConvectiveForm form, int order = 0);

  void set_grad_mode(GradMode m) { mode_ = m; }
  GradMode grad_mode() const { return mode_; }
  // the rule degree used for an element of polynomial order p
  int quad_order(int p) const;

  void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             const mfem::Vector& elfun, mfem::Vector& elvect) override;
  void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                           const mfem::Vector& elfun, mfem::DenseMatrix& elmat) override;

 private:
  double rho_;
  int vdim_;
  bool axi_;
  ConvectiveForm form_;
  int order_;
  GradMode mode_ = GradMode::Newton;
};

}  // namespace axf::fluids
