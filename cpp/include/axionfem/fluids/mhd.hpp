// Inductionless MHD for the steady Navier-Stokes solver (beta; formulation: docs/guides/fluids.md).
//
// 2.5D planar flow: u = (u_x, u_y, u_z)(x, y), pressure p, electric potential phi, a PRESCRIBED magnetic field B(x, y)
// (three components) and conductivity sigma(x, y); no induced field.
//
//   J  = sigma (-grad phi + E_z e_z + u x B)          (grad = (d_x, d_y, 0); the weak, conservative current)
//   rho (u.grad)u - mu Laplace u + grad p = G_z e_z + J x B + f,   div u = 0,   (J, grad psi) = <j_n, psi> - (q, psi)
//
// The Lorentz coupling is LINEAR in (u, phi): it adds constant blocks to the Navier-Stokes Newton system (the Picard and
// the Newton linearizations coincide on it), and phi is an extra block of unknowns.  The convection and the nonlinear
// driver are those of the steady Navier-Stokes solver (navier_stokes.hpp): this header only describes the data of the
// coupling.
//
// Electric boundary conditions, one per boundary attribute: Insulating (J.n = 0, natural), NormalCurrent (J.n = j_n,
// outward n, natural), Potential (phi = phi_D, strong, P2 nodal interpolation).  With no Potential attribute the
// potential is defined up to a constant (the potential nullspace): it is detected algebraically, removed by pinning one
// node, and the reported potential has zero mean.  Single MPI rank only (refused otherwise).
#pragma once

#include <mfem.hpp>

#include <string>
#include <vector>

namespace axf::fluids {

enum class ElectricKind { Insulating, NormalCurrent, Potential };
const char* to_string(ElectricKind k);

struct ElectricCondition {
  int attribute = 0;
  ElectricKind kind = ElectricKind::Insulating;
  mfem::Coefficient* value = nullptr;  // NormalCurrent: j_n (outward normal); Potential: phi_D; nullptr = 0
};

struct MhdOptions {
  mfem::Coefficient* sigma = nullptr;      // conductivity sigma(x, y) > 0 (required)
  mfem::VectorCoefficient* B = nullptr;    // imposed field (B_x, B_y, B_z)(x, y), 3 components (required)
  double E_z = 0.0;                        // axial electric field E_z (the total potential is phi - E_z z)
  mfem::Coefficient* q = nullptr;          // volumetric current source (div J = q; zero physically, MMS only)
  std::vector<ElectricCondition> electric; // exactly one per boundary attribute
  int potential_order = 2;                 // Lagrange order of phi (the verification cases use P2)
  int quad_order = 0;                      // volume rule of the coupling; 0 = 2 max(p_u, p_phi) + 4
  int quad_order_boundary = 0;             // boundary rule of j_n; 0 = 2 p_phi + 8
};

struct MhdDiagnostics {
  bool active = false;
  long n_potential = 0;                    // potential DOFs
  int n_aux = 0;                           // extra unknowns of the coupling (the potential phi)
  int potential_order = 2;
  bool phi_nullspace_detected = false;     // algebraic: the constants are in the kernel of the eliminated phi block
  bool phi_nullspace_declared = false;     // what the conditions imply (no Potential attribute)
  double phi_kernel_defect = -1;           // ||K_ff 1|| / (||K||_F sqrt(n)) of the eliminated phi block
  double phi_flux_defect_abs = 0;          // |sum of the phi right-hand side| removed by the projection (nullspace only)
  long phi_pinned_dof = -1;                // the pinned potential DOF (nullspace only)
  long n_potential_dirichlet = 0;          // strongly constrained potential DOFs (Dirichlet attributes)
  int quad_order = 0, quad_order_boundary = 0;
  std::string current_form = "galerkin";   // the potential row is the Galerkin row of div J = q with J_h of the force
  double assemble_seconds = 0;
  double phi_scale = 1, row_scale = 1;     // phi = phi_scale a; the potential row is divided by row_scale (1, 1 if not applied)
};

}  // namespace axf::fluids
