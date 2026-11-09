// Private: the state of a 3D model shared by model3d.cpp (MQS) and model3d_fullwave.cpp (full wave).
#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/background.hpp"
#include "axionfem/core.hpp"
#include "axionfem/materials.hpp"
#include "axionfem/mesh_quality.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/spec.hpp"

namespace axf {

// One quadrature rule for every integral of a port feed: the conduction stiffness, the current normalization and
// the full-wave right-hand side. The feed is charge-free at free vertices only if the right-hand side's gradient rows
// (int J . grad phi_j) use the rule the conduction solve made zero (int grad V . grad phi_j). On straight elements
// any exact rule agrees; on curved elements the integrands are not polynomial, and different rules left about 3e-5
// relative charge there.
inline const mfem::IntegrationRule& port_feed_rule(int order) {
  return mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order + 2);
}

// An integer solver option from the environment ($name), or def when it is unset; anything but an integer in int's
// range is an error (a silent 0 or default would select another hypre smoother).
inline int env_int_checked(const char* name, int def) {
  const char* s = std::getenv(name);
  if (!s || !*s) return def;
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(s, &end, 10);
  AXF_REQUIRE(end != s && *end == '\0' && errno != ERANGE && v >= INT_MIN && v <= INT_MAX,
              strcat(name, "='", s, "' is not an integer"), "",
              "Give hypre's option number (docs/validation/gpu.md lists the ones used), or unset it.");
  return static_cast<int>(v);
}

// The integer solver options of the 3D AMS solves (MQS: AXF_AMS_*; full wave: AXF_FW_AMS_*), checked once when a 3D run
// starts, so that a bad value stops it before meshing rather than at the solve that reads it (minutes in on a large
// model). The solves read them again with the same function.
inline void check_integer_solver_env() {
  for (const char* name : {"AXF_AMS_CYCLE", "AXF_AMS_RLX", "AXF_AMS_AMG_COARSEN", "AXF_AMS_AMG_RELAX", "AXF_AMS_SWEEPS",
                           "AXF_AMS_AGG", "AXF_FW_AMS_RLX", "AXF_FW_AMS_AMG_COARSEN", "AXF_FW_AMS_AMG_RELAX",
                           "AXF_FW_AMS_AGG"})
    (void)env_int_checked(name, 0);
}

// Wait for hypre's device work before the host reads or writes hypre's (unified) memory directly. MFEM's HostRead and
// HostReadWrite on a HypreParMatrix only set flags for this memory: they do not synchronize, and hypre's kernels and
// its device-to-device copies (cudaMemcpy) run asynchronously to the host. A no-op unless hypre runs on a GPU.
inline void sync_hypre_device() {
#if defined(HYPRE_USING_GPU)
  if (mfem::HypreUsingGPU()) hypre_SyncDevice();
#endif
}

// A unit diagonal on the rows of eliminated DOFs, added to an assembled form before its parallel assembly (before
// Finalize or FormLinearSystem). The H1 problems posed on part of the mesh (a readout port, a poloidal winding) have a
// coefficient that vanishes elsewhere and eliminate every DOF there, so those rows hold only zeros. The host assembly
// keeps the zeros in the sparsity pattern and the elimination puts 1 on their diagonals; hypre's GPU triple product
// (mfem::RAP when hypre runs on a GPU; MFEM notes it in a FIXME there) drops them, the rows end up empty, and
// BoomerAMG's setup fails (hypre error 13; docs/validation/gpu.md). Exact on either device: the elimination
// overwrites the row and column of every such DOF (diagonal 1, right-hand side its prescribed value).
inline void keep_eliminated_rows(mfem::ParBilinearForm& form, const mfem::Array<int>& ess_tdofs) {
  mfem::ParFiniteElementSpace& fes = *form.ParFESpace();
  mfem::Vector tmark(fes.GetTrueVSize());
  tmark = 0.0;
  for (int t : ess_tdofs) tmark(t) = 1.0;
  mfem::ParGridFunction vmark(&fes);
  vmark.SetFromTrueDofs(tmark);  // shared DOFs take their owner's mark
  mfem::SparseMatrix& m = form.SpMat();
  for (int i = 0; i < fes.GetVSize(); ++i)
    if (vmark(i) > 0.5) m.Add(i, i, 1.0);
}

// The same for an assembled square matrix about to be eliminated on `rows` (local true DOFs): a unit diagonal on each
// of those rows that has no diagonal entry. A product such as G^T M G is all zeros on the nodes where M vanishes; on
// a GPU, hypre drops those rows (keep_eliminated_rows), and an elimination with DIAG_ONE cannot set a diagonal the
// pattern lacks. Exact (the rows are eliminated next) and, when every row has its diagonal, as on the host, a no-op
// that leaves the matrix untouched. Collective.
inline void ensure_diagonal(std::unique_ptr<mfem::HypreParMatrix>& A, const mfem::Array<int>& rows) {
  A->HostRead();
  sync_hypre_device();
  hypre_CSRMatrix* d = hypre_ParCSRMatrixDiag(static_cast<hypre_ParCSRMatrix*>(*A));
  const HYPRE_Int* I = hypre_CSRMatrixI(d);
  const HYPRE_Int* J = hypre_CSRMatrixJ(d);
  std::vector<int> missing;
  for (int r : rows) {
    bool found = false;
    if (I)
      for (HYPRE_Int k = I[r]; k < I[r + 1] && !found; ++k) found = J[k] == r;
    if (!found) missing.push_back(r);
  }
  int count = static_cast<int>(missing.size());
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (count == 0) {
    A->HypreRead();  // back in hypre's memory (GetDiag, for one, verifies it)
    return;
  }
  log(2, strcat("  ", count, " eliminated rows had no diagonal entry (hypre on a GPU): unit diagonal added"));
  mfem::SparseMatrix diag(A->Height());
  for (int r : missing) diag.Set(r, r, 1.0);
  diag.Finalize();
  mfem::HypreParMatrix D(A->GetComm(), A->GetGlobalNumRows(), A->RowPart(), &diag);
  A.reset(mfem::Add(1.0, *A, 1.0, D));
  // the diagonal first in each row, as MFEM's four-argument HypreParMatrix::EliminateBC reads it
  A->HostReadWrite();
  sync_hypre_device();
  hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(static_cast<hypre_ParCSRMatrix*>(*A)));
  A->HypreRead();
}

// The diagonal of a square HypreParMatrix as stored, found by column on the host: 0 for a row with no diagonal entry.
// Use it instead of HypreParMatrix::GetDiag, which returns each row's first stored entry as its diagonal and writes a
// device copy when hypre runs on a GPU. A row that hypre on a GPU left empty then reads the first entry of the next row,
// a positive diagonal, and a cavity's off-wall DOFs would be taken for wall DOFs.
// The same values as GetDiag wherever the diagonal is stored first, as on the host. For a square matrix with the same
// row and column partition only (the diag block's local column r is then row r's diagonal), as every RAP product with
// one P on both sides.
inline mfem::Vector stored_diagonal(mfem::HypreParMatrix& A) {
  AXF_REQUIRE(A.GetGlobalNumRows() == A.GetGlobalNumCols() && A.Height() == A.Width() &&
                  A.RowPart()[0] == A.ColPart()[0],
              "internal: stored_diagonal needs a square matrix with matching row and column partitions", "", "");
  A.HostRead();
  sync_hypre_device();
  hypre_CSRMatrix* d = hypre_ParCSRMatrixDiag(static_cast<hypre_ParCSRMatrix*>(A));
  const HYPRE_Int* I = hypre_CSRMatrixI(d);
  const HYPRE_Int* J = hypre_CSRMatrixJ(d);
  const HYPRE_Real* D = hypre_CSRMatrixData(d);
  const int n = A.Height();
  mfem::Vector diag(n);
  diag = 0.0;
  if (I) {
    for (int r = 0; r < n; ++r) {
      for (HYPRE_Int k = I[r]; k < I[r + 1]; ++k) {
        if (J[k] == r) {
          diag(r) = D[k];
          break;
        }
      }
    }
  }
  A.HypreRead();
  return diag;
}

struct Model3D::Impl {
  std::unique_ptr<mfem::Mesh> serial;
  std::unique_ptr<mfem::ParMesh> pmesh;
  std::map<int, std::string> region_of_attr;
  std::map<std::string, int> attr_of_region;
  int order = 2;
  std::unique_ptr<mfem::ND_FECollection> fec, fec1;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes, fes1;
  std::unique_ptr<mfem::ParGridFunction> A_lab, A_pick;
  bool have_lab = false, have_pick = false;
  double I_pick = 1.0;
  std::vector<std::string> elem_region;
  std::vector<const spec::Material*> elem_mat;
  std::vector<int> body_static, body_rf;  // element -> body index (-1 none)
  std::vector<std::unique_ptr<BHLaw>> bh;
  std::vector<const BHLaw*> elem_bh;
  json mesh_info;
  int sh_attr = -1;
  double sh_R1 = 0, sh_R2 = 0, sh_c[3] = {0, 0, 0};
  // MPI: every rank holds the serial mesh; the ParMesh is partitioned with loc2ser mapping local -> serial
  // elements (ParMesh keeps the serial order within a rank). Topology (bodies, cuts) is computed on the serial
  // mesh and projected; scenes/sampling use a gathered copy of the fields on the serial mesh (rank 0).
  int rank = 0, nranks = 1;
  std::vector<int> loc2ser;
  std::vector<int> ser_attr;                          // serial element attributes
  std::unique_ptr<mfem::FiniteElementSpace> fes1s;    // serial ND1 (topology lifts)
  std::unique_ptr<mfem::ParMesh> view_mesh;           // serial mesh on COMM_SELF (rank 0 when nranks > 1)
  std::unique_ptr<mfem::ParFiniteElementSpace> view_fes;
  std::unique_ptr<mfem::ParGridFunction> view_A_lab, view_A_pick;
  std::vector<int> view_body_static;                  // static-body index of each view element
  mfem::ParMesh& vmesh() const { return view_mesh ? *view_mesh : *pmesh; }
  const mfem::ParGridFunction* vA_lab() const { return view_mesh ? view_A_lab.get() : A_lab.get(); }
  const mfem::ParGridFunction* vA_pick() const { return view_mesh ? view_A_pick.get() : A_pick.get(); }
  const std::vector<int>& vbody_static() const { return view_mesh ? view_body_static : body_static; }
  // readout port (pickup type "port"): conduction potential V (unit voltage, sigma = 1) in the port region;
  // J = port_scale grad V carries the readout current I_pick (used by the full-wave analysis)
  int port_attr = -1;
  double port_scale = 0;
  std::unique_ptr<mfem::H1_FECollection> port_h1c;
  std::unique_ptr<mfem::ParFiniteElementSpace> port_h1;
  std::unique_ptr<mfem::ParGridFunction> port_V;
  // surface (lumped) readout port (pickup.port_boundary, imported meshes): the true-DOF source vector
  // int_S K . N dS of the uniform surface current K carrying I_pick across the port faces (replaces port_V)
  std::unique_ptr<mfem::Vector> port_surface_J;
  struct ExtraPort {
    std::string name, region;
    int attr = -1;
    double scale = 0;
    std::unique_ptr<mfem::H1_FECollection> h1c;
    std::unique_ptr<mfem::ParFiniteElementSpace> h1;
    std::unique_ptr<mfem::ParGridFunction> V;
    json diagnostics;
  };
  std::vector<ExtraPort> extra_ports;
  json port_diagnostics;
  // gathered copy of a partitioned field on the serial mesh (identical on every rank; model3d.cpp)
  std::unique_ptr<mfem::ParGridFunction> gather(const mfem::ParGridFunction& u);
  // full-wave RF field (complex E per readout ampere) at fw_scene_f, for scenes (view copies when nranks > 1)
  double fw_scene_f = 0;
  std::unique_ptr<mfem::ParGridFunction> E_re, E_im, view_E_re, view_E_im;
  const mfem::ParGridFunction* vE_re() const { return view_mesh ? view_E_re.get() : E_re.get(); }
  const mfem::ParGridFunction* vE_im() const { return view_mesh ? view_E_im.get() : E_im.get(); }
  // static source current density J(x) of the windings, kept after the static solve for the 3D mechanics
  // (model3d_mech.cpp); the grid functions and spaces that its loop/poloidal currents read live alongside
  std::vector<std::unique_ptr<mfem::H1_FECollection>> J_keep_fec;
  std::vector<std::unique_ptr<mfem::ParFiniteElementSpace>> J_keep_fes;
  std::vector<std::unique_ptr<mfem::ParGridFunction>> J_keep;
  std::unique_ptr<mfem::VectorCoefficient> J_static;
  // external (background) static field, axisymmetric about the z axis; A_lab holds the reaction field
  std::unique_ptr<BackgroundField> bg;
  int shell_attr_or(int e_attr) const { return sh_attr > 0 && e_attr == sh_attr; }
  // background B (and A) at a physical point; zero without a background
  void bg_at(const mfem::Vector& x, double B[3], double A[3] = nullptr) const {
    for (int k = 0; k < 3; ++k) {
      B[k] = 0;
      if (A) A[k] = 0;
    }
    if (!bg) return;
    const double r = std::hypot(x(0), x(1));
    double br = 0, bz = 0;
    bg->B(r, x(2), br, bz);
    B[0] = r > 0 ? br * x(0) / r : 0.0;
    B[1] = r > 0 ? br * x(1) / r : 0.0;
    B[2] = bz;
    if (A && r > 0) {
      const double a = bg->psi(r, x(2)) / r;  // A_phi
      A[0] = -a * x(1) / r;
      A[1] = a * x(0) / r;
    }
  }
  // 3D eigenmodes kept after the full-wave analysis (true-DOF vectors of fes, M-normalized: int eps_r |E|^2 = 1
  // in units of the k^2-scaled pencil; fw_mode_k2 = k^2 = (omega/c)^2 per mode)
  std::vector<mfem::Vector> fw_modes;
  std::vector<double> fw_mode_k2;
  // mesh check of the last solver failure (worst_elements_note), cached per mesh state
  const mfem::ParMesh* wq_mesh = nullptr;
  long wq_seq = -1, wq_nodes_seq = -1;
  mesh_quality::Summary wq_summary;
};

// Collective (every rank, at the same failure): the worst elements of the partitioned mesh, reduced over the ranks
// (inverted first, then by Jacobian aspect ratio), as the text a solver-failure message appends (" Mesh check: ...
// Fix ..."). Computed only when called, i.e. on a failure path, and once per mesh state; the worst five also go to
// mesh_info["worst_elements"]. `listed_in` names the file the message points to for them ("the results"; for a
// thrown error the retained file, or "" when none is written). model3d.cpp.
std::string worst_elements_note(Model3D::Impl& P, const std::string& listed_in = "the results");

}  // namespace axf
