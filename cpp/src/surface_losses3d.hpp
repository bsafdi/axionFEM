// Private: RF surface losses of the 3D engines (docs/physics/fullwave_surface_losses.md). The face selection, the
// quadrature rule, the field side's permeability and the magnetoresistance weight are shared by the MQS pickup losses
// (pickup.losses, model3d.cpp) and the full-wave surface losses (FwSurfaceLosses, model3d_surface_losses.cpp), so
// that both integrate Gamma = oint |H|^2 dA / I^2 over the same faces in the same way.
#pragma once

#include <algorithm>
#include <complex>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/spec.hpp"
#include "axionfem/units.hpp"
#include "model3d_impl.hpp"

namespace axf {

// A face of the surface-loss integrals: a conductor or superconductor element on exactly one side and an element on
// the other side (the field side, where H is evaluated). Boundary faces carry no loss here.
struct SurfaceLossFace {
  int face = -1;
  int conductor = -1;   // the element on the conductor side
  int field = -1;       // the element on the field side
};

// The surface-loss faces of a mesh, in face order. conductor(e): e is a conductor or superconductor element;
// excluded(e): e may not be a field side (the infinity shell).
template <class Conductor, class Excluded>
std::vector<SurfaceLossFace> surface_loss_faces(const mfem::Mesh& m, Conductor conductor, Excluded excluded) {
  std::vector<SurfaceLossFace> out;
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0 || conductor(e1) == conductor(e2)) continue;
    const int ec = conductor(e1) ? e1 : e2;
    const int ev = ec == e1 ? e2 : e1;
    if (excluded(ev)) continue;
    out.push_back({f, ec, ev});
  }
  return out;
}

// The triangle rule of the surface-loss integrals for Nedelec order p: degree 2p + 2 (|curl E|^2 has degree 2p - 2
// on a straight face; the margin covers curved faces).
inline const mfem::IntegrationRule& surface_loss_rule(int order) {
  return mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * order + 2);
}

// The permeability on a face's field side: mu0 mu_r for linear and iron materials, mu0 otherwise. This is the
// full-wave solve's permeability; the MQS pickup losses replace it next to nonlinear iron by the secant permeability
// of the static field.
inline double surface_loss_mu(const spec::Material& m) {
  const bool linear = m.kind == spec::MaterialKind::Linear || m.kind == spec::MaterialKind::Iron;
  return linear ? units::mu0 * m.mu_r : units::mu0;
}

// dG weighted by the magnetoresistance of a surface's R_s at the static field |B| [T]: dG R_s(f, B) / R_s(f) =
// dG Rs_factor(B), which does not depend on f and never evaluates R_s, as the 2D engine does (a measured R_s table
// need not reach any particular frequency).
inline double surface_loss_mr_weighted(const spec::Material& m, double B, double dG) {
  return dG * m.rf.Rs_factor(B);
}

// What a solve of true relative residual rho resolves. The residual can move Re Z by up to about rho |X| (an estimate:
// the actual error in the test runs was at most
// 0.2 rho |X|, and 2-5e-5 rho |X| in a tight-tolerance rerun).
// - Re Z, and with it R_rad_ohm and Q_rad, is flagged as unresolved where max(Re Z, 0) < kReZResolution rho |X|.
// - Q_total = omega L / (R_loss + max(Re Z, 0)) is withheld only where rho |X| > kQTotalResolution (R_loss + max(Re Z,
//   0)): where the conductor loss dominates, Q_total is resolved even when Re Z is not.
inline constexpr double kReZResolution = 100.0;
inline constexpr double kQTotalResolution = 0.01;

// The full-wave surface losses of the readout ports (docs/physics/fullwave_surface_losses.md). Perturbative: the
// port fields of the perfectly conducting structure, with each surface's R_s(f) applied afterwards (valid while
// R_loss << |X|). Per frequency and port k, on every conductor or superconductor surface b,
//   Gamma_b = oint_b |H_k|^2 dA / |I|^2,   H_k = curl E_k / (i omega mu) on the field side,
// R_eff,b = R_s,b(f) Gamma_b (magnetoresistance-weighted), R_loss = sum_b R_eff,b, Q_loss = omega L / R_loss and the
// tuned Q_loss_tuned = omega L_eff / R_loss with L_eff = L + (omega / 2) dL/domega from the exact dZ/domega. The faces
// are those of pickup.losses (surface_loss_faces on the serial mesh); every rank evaluates the faces whose field side
// it holds, and the sums are reduced over the ranks, so every rank holds the same results.
class FwSurfaceLosses {
 public:
  // epml: per local element its PML layer (-1: none; empty: no PML layers); static_field: B_lab = curl A_lab (and an
  // imposed background) is the static field of the magnetoresistance weight (false: no static solve, rf_only).
  // Collective.
  FwSurfaceLosses(const Model3D::Impl& P, const spec::Design& d, const std::vector<int>& epml, bool static_field);
  // a conductor or superconductor surface face exists (on any rank)
  bool active() const { return faces_ > 0; }
  // something to report: loss faces, or PEC boundary groups (conductors without volume elements; warned about when
  // they are all the conductors there are, or when their materials have rf data)
  bool reported() const { return faces_ > 0 || !pec_groups_.empty(); }
  // The losses at frequency f of the port fields per ampere re[k] + i im[k] (true DOFs; their curl representatives
  // where the solve has them), with the port impedance matrix Z; the exact dZ/domega is read from the row
  // (dZ_domega_H; without it the tuned Q is omitted). Adds the keys to primary_row (port 0) and to each
  // primary_row["port_solutions"][k]. Collective.
  void Evaluate(double f, const std::vector<const mfem::Vector*>& re, const std::vector<const mfem::Vector*>& im,
                const Eigen::MatrixXcd& Z, nlohmann::json& primary_row);
  // The summary block (fullwave.surface_losses) over the sweep's rows; appends warnings.
  nlohmann::json Summary(const nlohmann::json& sweep, bool adaptive, nlohmann::json& warnings) const;

 private:
  struct Point {
    int element = -1;            // local field-side element
    mfem::IntegrationPoint ip;   // in that element's reference coordinates
    double dA = 0;               // area weight
    double inv_mu2 = 0;          // 1 / mu^2 of the field side
    double n[3] = {0, 0, 0};     // unit normal
    double mr = 1;               // magnetoresistance weight R_s(B) / R_s
    int slot = 0;                // surface * regions + field-side region
  };
  const Model3D::Impl& P_;
  mfem::ParMesh& pm_;
  mfem::ParFiniteElementSpace& fes_;
  int order_ = 2;
  std::vector<std::string> surfaces_;                  // conductor surfaces (regions), the same list on every rank
  std::vector<const spec::Material*> material_;        // per surface
  std::vector<std::string> regions_;                   // field-side regions, the same list on every rank
  std::vector<std::vector<char>> pair_;                // [surface][region]: a face of that pair exists
  std::vector<Point> points_;                          // this rank's quadrature points
  std::vector<std::string> pec_groups_;                // PEC boundary groups (not symmetry planes): no loss here
  std::vector<std::string> holes_with_rf_;             // of those, the conductors (mesh.remove_pec) with rf data
  long faces_ = 0, points_total_ = 0, pml_faces_ = 0;  // global counts
  bool any_rf_ = false, any_mr_ = false, static_field_ = true;
  double setup_seconds_ = 0, evaluation_seconds_ = 0;
  int evaluations_ = 0;
};

}  // namespace axf
