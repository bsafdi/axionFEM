// Internal (not installed): the Lorentz coupling of the inductionless-MHD solver, read by NavierStokesSolver.
#pragma once

#include <mfem.hpp>

#include <memory>
#include <vector>

#include "axionfem/fluids/mhd.hpp"

namespace axf::fluids {

// Extra unknowns a = (phi [, jp]) and the constant (linear) blocks coupling them to the velocity u:
//   R_u = ... + Kuu u + Kua a - Fu            (the momentum rows' Lorentz part: -(J x B, v))
//   R_a = Kau u + Kaa a - Fa                  (the conservative potential row)
// All matrices are indexed by velocity vdofs (== true dofs on one rank) and by aux dofs.  *_e: essential rows/columns
// removed (aux essential = Dirichlet potential + the pinned nullspace node, identity on the aux diagonal).
class MhdCoupling {
 public:
  MhdCoupling(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& ufes, const MhdOptions& o,
              const mfem::Array<int>& u_ess);
  int na() const { return na_; }
  const mfem::SparseMatrix& Kuu_e() const { return *Kuu_e_; }
  const mfem::SparseMatrix& Kua_e() const { return *Kua_e_; }
  const mfem::SparseMatrix& Kau_e() const { return *Kau_e_; }
  const mfem::SparseMatrix& Kaa_e() const { return *Kaa_e_; }
  const mfem::SparseMatrix& mass_e() const { return *Ma_e_; }
  const mfem::Vector& lift() const { return lift_; }
  const std::vector<char>& ess_mask() const { return ess_mask_; }
  // Ru += Kuu u + Kua a - Fu  (full operators; the caller zeroes the essential velocity rows afterwards)
  void add_force(const mfem::Vector& u, const mfem::Vector& a, mfem::Vector& Ru) const;
  // Ra = Kau u + Kaa a - Fa with the essential aux rows zeroed
  void aux_residual(const mfem::Vector& u, const mfem::Vector& a, mfem::Vector& Ra) const;
  void set_fields(const mfem::Vector& a);
  const mfem::GridFunction& potential() const { return *phi_gf_; }
  mfem::FiniteElementSpace& potential_space() { return *phifes_; }
  const MhdDiagnostics& diagnostics() const { return diag_; }

 private:
  int na_ = 0, nphi_ = 0;
  double phi_scale_ = 1.0, row_scale_ = 1.0;  // phi = phi_scale a; the potential row is divided by row_scale
  std::unique_ptr<mfem::FiniteElementCollection> phifec_;
  std::unique_ptr<mfem::FiniteElementSpace> phifes_;
  std::unique_ptr<mfem::SparseMatrix> Kuu_, Kua_, Kau_, Kaa_;
  std::unique_ptr<mfem::SparseMatrix> Kuu_e_, Kua_e_, Kau_e_, Kaa_e_, Ma_e_;
  mfem::Vector Fu_, Fa_, lift_;
  std::vector<char> ess_mask_;
  std::unique_ptr<mfem::GridFunction> phi_gf_;
  MhdDiagnostics diag_;
};

}  // namespace axf::fluids
