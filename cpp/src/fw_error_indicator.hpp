// Private: an error indicator of the 3D full-wave port fields (recovery at order 1; element residual problems and a
// global gradient projection at order 2 and above), evaluated per frequency and port by Model3D::fullwave_analysis
// (model3d_fw_indicator.cpp; docs/physics/fullwave_error_indicator.md).
#pragma once

#include <array>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "fw_local_residual.hpp"
#include "model3d_impl.hpp"

namespace axf {

// The error of a port field E in the Maxwell energy norm
//
//   ||E||^2 = int nu |curl E|^2 + omega^2 |eps_c| |E|^2 dV     over the field elements (outside perfect conductors,
//                                                              the infinity shell and PML layers),
//
// estimated per element as eta_e^2, a magnetic plus an electric part.
//
// Order 1: recovery (Zienkiewicz-Zhu). eta_e^2 = int_e nu |curl E - R(curl E)|^2 + omega^2 |eps_c| |E - R(E)|^2 dV,
// where R averages the element values at the nodes of the continuous H1 space of the field's order p, separately
// within each material (fields jump at material interfaces) and on each side of the interior boundary faces of a
// boundary group (surface ports, sheets, interior PEC faces), which carry surface currents and charges.
//
// Order 2 and above: the recovery's space contains the discrete curl (degree p - 1), so it sees only its jumps (on a
// TEM line with an exact field, 0.33 of the magnetic error at order 2). Instead:
// - magnetic: per element, the local problem for the error in ND_{p+1}(K), on the complement of the local gradients,
//   with the averaged traction n x {nu curl E_h} on interior faces (Residual, model3d_fw_indicator.cpp);
// - electric: the error's projection onto the gradients of the global H1_{p+1} space, (eps grad psi, grad xi) =
//   -(eps E_h, grad xi), with the complex eps_c: it needs no flux, because the exact field's term vanishes
//   (div eps_c E = 0, n . eps_c E continuous). Where all field materials share one loss phase, |eps| stands for eps
//   (a real SPD problem); where the phases differ the complex-symmetric problem is solved. The leading local error of
//   a Nedelec field is a gradient of degree p + 1 (P_p^3 = N_p + grad P_{p+1}): on a smooth field (a TEM line) this
//   part reproduced the exact electric error to 0.997-1.000; next to a crack, where the error is singular, it
//   captured about half.
// At order 2 and above the recovery gives the surface term, and its magnetic part is reported as a diagnostic.
//
// For a port's own impedance the error e = E - E_h obeys Z_h - Z = a(e, e) / (i omega) per ampere (Galerkin
// orthogonality of the complex-symmetric form a). The norm holds a's volume terms over the field elements only, so
// with eta for ||e|| the Z and L estimates leave out the error of the radiating shell, the exterior expansion, PML
// layers and Robin (impedance, sheet, scattering) groups; with PML or Robin groups they are withheld. At order 2 and
// above a(e, e) itself is estimated element by element, with sign. The estimate is not a bound
// (docs/physics/fullwave_error_indicator.md, Validation). On conductor faces the recovery of curl E (n x H, which sets
// surface losses) is reported separately, as the MQS indicator's surface term does for Gamma.
class FwErrorIndicator {
 public:
  // the parts of the operator whose error the volume norm leaves out
  struct Terminations {
    bool radiating_shell = false;
    bool pml = false;
    std::vector<std::string> robin_groups;   // impedance, sheet and scattering boundary groups, "name (type)"
  };
  // in_pml: per local element, nonzero inside a PML layer (empty: no PML). Collective.
  FwErrorIndicator(const Model3D::Impl& P, const spec::Design& d, const std::vector<char>& in_pml,
                   const Terminations& terminations);
  ~FwErrorIndicator();
  // One port field per ampere (true-DOF vectors), with its curl representative (nullptr: E itself), at angular
  // frequency w; eps and nu per local element; Z the port's own impedance. accepted: the solve passed its residual
  // gate (the same on every rank); if not, the estimate is withheld. Collective.
  json Evaluate(const mfem::Vector& Er, const mfem::Vector& Ei, const mfem::Vector* Cr, const mfem::Vector* Ci,
                double w, const std::vector<std::complex<double>>& eps, const std::vector<std::complex<double>>& nu,
                std::complex<double> Z, bool accepted);
  // The last evaluation's eta_e^2 / ||E||^2 per element, as an MFEM L2 (order 0) grid function of this rank's part
  // of the mesh. Each rank writes its own file; a failure on any rank is raised on every rank. Collective.
  void Save(const std::string& path) const;
  // Method, scope and reliability, for the run record.
  json Describe() const;

 private:
  struct Face {
    int face = -1;
    int side = 0;      // the field side: 0 is the face's first element, 1 its second
    int element = -1;  // the field-side (local) element
  };
  // the cuts of the recovery: interior boundary faces of a boundary group with one material on both sides (serial
  // mesh, the same on every rank), and per local element the slot of each of its H1 nodes that lies on a cut
  void FindCuts(const Model3D::Impl& P, const std::vector<int>& serial_group);
  // the element's dof values of E and of the curl source, with the dof transformation applied
  void Load(int e, const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im);
  // the same for face neighbor nbr (an element of another rank across a shared face; face neighbor data exchanged)
  void LoadNbr(int nbr, const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im);
  // E (re, im) and curl E (re, im) of the loaded element at T's integration point
  void Values(mfem::ElementTransformation& T, double out[12]);
  // nodal averages of E and curl E over a group's elements, summed over the ranks that share a node (the H1 space)
  // and, on cuts, per side (slots). Collective.
  void Recover(const std::vector<int>& elements, const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im);
  // the recovered nodal values of element e (12 x its H1 dofs), from the H1 averages or, on cuts, the side's slot
  void Nodal(int e, const mfem::Array<int>& dofs);

  mfem::ParMesh& pm_;
  mfem::ParFiniteElementSpace& fes_;
  int order_ = 2;
  int ne_ = 0;
  Terminations terminations_;
  std::vector<std::string> groups_;    // recovery groups: material names, the same list on every rank
  std::vector<std::string> regions_;   // region names of the breakdown, the same list on every rank
  std::vector<int> group_of_;          // per local element; -1: not a field element
  std::vector<int> region_of_;         // per local element
  std::vector<std::vector<int>> members_;   // local field elements of each group
  std::vector<std::vector<Face>> faces_;    // local conductor faces of each group (by the field side's group)
  std::vector<long> group_size_;            // global number of field elements of each group
  long field_elements_ = 0;
  long conductor_faces_ = 0;
  long excluded_conductor_ = 0, excluded_shell_ = 0, excluded_pml_ = 0;   // global element counts left out
  long cut_faces_ = 0;
  std::vector<std::string> cut_groups_;     // the boundary groups that cut the recovery
  int slots_ = 0;                           // (node on a cut, side) pairs
  std::vector<std::vector<int>> cut_slot_;  // per local element: per H1 node its slot, -1 off the cuts (empty: none)
  std::vector<double> slot_sums_;           // 13 per slot
  double setup_seconds_ = 0.0;   // the construction (lists, conductor faces, cuts, the recovery space), slowest rank
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> space_;   // H1 of order p, 13 components (Ordering::byNODES)
  mfem::Vector sums_;
  std::unique_ptr<mfem::ParGridFunction> e_re_, e_im_, c_re_, c_im_;
  // order 2 and above: the element residual problems of the magnetic part (their reference tables and the faces'
  // kinds, from the serial mesh) and the global gradient projection of the electric part
  struct Residual;
  std::unique_ptr<Residual> res_;
  void SetupResidual(const Model3D::Impl& P, const std::vector<int>& serial_group, const std::vector<char>& closed);
  // per local field element: the local error's magnetic and electric energies and its form a_K(e_K, e_K), and the
  // norm's two parts for E_h. Collective.
  json EvaluateResidual(double w, const std::vector<std::complex<double>>& eps,
                        const std::vector<std::complex<double>>& nu, const mfem::ParGridFunction& c_re,
                        const mfem::ParGridFunction& c_im, std::vector<double>& magnetic, std::vector<double>& electric,
                        std::vector<std::complex<double>>& form, std::vector<double>& norm_m,
                        std::vector<double>& norm_e);
  // H1_{p+1} on the mesh, psi = 0 off the field elements, on conductor faces and edges and on the faces with the
  // element's own traction; the operator (|eps| grad, grad) is assembled when |eps| changes (lossy media)
  const mfem::FiniteElementCollection* nd_fec_ = nullptr;   // the field's ND_p
  std::unique_ptr<mfem::H1_FECollection> gfec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> gspace_;
  mfem::Array<int> gess_;
  std::vector<double> geps_;   // |eps| per local element of the assembled operator (0 off the field)
  std::unique_ptr<mfem::HypreParMatrix> gA_;
  std::unique_ptr<mfem::HypreBoomerAMG> gamg_;
  // where the field materials' loss phases differ: the operator's real and imaginary parts, (Re eps grad, grad) and
  // (Im eps grad, grad), solved together by GMRES with PRESB (gA_ and gamg_ then hold the sum (Re eps + Im eps))
  std::vector<double> geps_re_, geps_im_;
  std::unique_ptr<mfem::HypreParMatrix> gAr_, gAi_;
  const mfem::IntegrationRule* grule_ = nullptr;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> gvshape_, gdshape_;   // per point of grule_: ND_p shapes,
                                                                           // H1_{p+1} reference gradients
  void SetupGradient();
  // per local field element: omega^2 |eps| |grad psi|^2 and -omega^2 eps grad psi . grad psi. Collective.
  json EvaluateGradient(double w, const std::vector<std::complex<double>>& eps, std::vector<double>& energy,
                        std::vector<std::complex<double>>& form);
  // the loaded element (Load, LoadNbr) and work arrays
  int loaded_ = -1;
  const mfem::FiniteElement* loaded_fe_ = nullptr;
  mfem::Array<int> vdofs_;
  mfem::Vector le_re_, le_im_, lc_re_, lc_im_, out3_;
  mfem::DenseMatrix vshape_, cshape_;
  std::vector<double> nodal_;   // 12 x the element's H1 dofs (Nodal)
  std::vector<double> local_;   // the last evaluation's eta_e^2 / ||E||^2
};

// The gradient projection's relative tolerance: 1e-4, or $AXF_FW_INDICATOR_TOL (a number in (0, 1e-4], for tests that
// compare two solvers at a tight tolerance: tests/test_gpu.py). Throws axf::Error on any other value.
double fw_indicator_gradient_tolerance();

// The order-2 local problem (fw_local_residual, fw_gradient_complement) is in fw_local_residual.hpp: MPI-free, so that
// its unit test builds in every configuration.

}  // namespace axf
