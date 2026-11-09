// Axisymmetric full-wave (time-harmonic Maxwell) problems beyond magnetoquasistatics.
//
// Time dependence e^{-i omega t}. Axisymmetric (m = 0) fields split into two families, each a scalar
// problem for an azimuthal unknown u on the H1 space of the MQS engine:
//
//   int alpha r B[u].B[v] - omega^2 int beta r u v + b_DtN(u, v) = f(v)      (per radian),
//   B[u] = (-du/dz, du/dr + u/r) = curl(u phi^).
//
//  * TE0, azimuthal E (loops, toroid sheaths): u = A_phi with E_phi = i omega A_phi, alpha = 1/mu,
//    beta = eps. Perfect conductors are masters (r A = psi_k on the body) exactly as in the MQS pickup
//    problem: closed loops keep psi_k = 0, slit bodies and the pickup carry a prescribed net azimuthal
//    current. A closed outer sphere means A = 0 there.
//  * TM0, azimuthal H (coax, CORE): u = H_phi, alpha = 1/eps, beta = mu. Perfect conductors are
//    excluded with the natural condition n x E = 0 on their surfaces (and on a closed outer sphere).
//    The field is split H = H_MQS + H' with the MQS pickup field H_MQS (exact level fields G/r, which
//    polynomials cannot represent); H' carries the displacement-current correction and vanishes as
//    omega -> 0. Each cavity enclosed by conductors conserves its toroidal flux (zero RF flux),
//    imposed as a linear constraint; this also removes the discrete near-static modes of such cavities.
//
// The exterior is an exact outgoing-wave Dirichlet-to-Neumann map on the outer semicircle (spherical
// Hankel functions h_n^(1) replacing the Laplace coefficients of the MQS map) or a closed perfectly
// conducting sphere. Eigenmodes (closed sphere): shift-invert Lanczos with the constraints.
#pragma once

#include <complex>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <mfem.hpp>
#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"

namespace axf {
class BackgroundField;
}

namespace axf::axi {

using cplx = std::complex<double>;

// gamma_n(x) = -d ln(rho h_n(k rho)) / d ln rho at x = k rho, n = 0..N (outgoing h_n^(1));
// gamma_n(0) = n (the Laplace limit).
std::vector<cplx> hankel_log_derivative(double x, int N);
// d gamma_n/dx, n = 0..N. From the spherical Bessel equation d gamma_n/dx = x + (gamma_n - n)(gamma_n + n + 1)/x;
// evaluated with gamma_n - n = -x / r_n (r_n = h_n / h_{n-1}, the recurrence ratio) as x - (gamma_n + n + 1) / r_n,
// which keeps its digits at small x and large n; d gamma_0/dx = -i. Zero for x <= 0 (gamma_n = n there).
std::vector<cplx> hankel_log_derivative_dx(double x, int N);
// Exterior (Dirichlet-to-Neumann) coefficient of one spherical mode of the 3D full-wave shell at angular frequency
// w on the sphere R1, with gamma = gamma_n(w R1 / c) and st the static TE response of the shell (n, or the capped
// value): TE (gamma - st) / (mu0 R1), TM -w^2 eps0 R1 / gamma.
cplx fw_exterior_coefficient(bool te, double st, double w, double R1, cplx gamma);
// Its w-derivative, dgamma = d gamma_n/dx at x = w R1 / c: TE dgamma / (mu0 c), TM -eps0 R1 w (2 gamma - x dgamma) / gamma^2.
cplx fw_exterior_coefficient_dw(bool te, double w, double R1, cplx gamma, cplx dgamma);
// Associated Legendre P_n^1(x), n = 0..N (P[0] = 0; unnormalized, no Condon-Shortley sign).
void assoc_legendre_p1(double x, int N, std::vector<double>& P);
// Exterior radial integrals J_n = int_{Rb}^inf rho^2 (Rb/rho)^{n+1} h_n(k rho)/h_n(k Rb) d rho, n = 0..N (J[0] = 0),
// with outgoing h_n = h_n^(1): the overlap of a static multipole (rho^-(n+1)) with an outgoing wave. Evaluated on
// the rotated contour rho = Rb + i s (the integrand decays as exp(-k s)); k -> 0 gives Rb^3 / (2n - 1).
std::vector<cplx> exterior_radial_integrals(double k, double Rb, int N);
// h_n(k rho) / h_n(k R1), n = 0..N (outgoing h_n^(1)), from the ratios r_n = h_n / h_{n-1} of the stable upward
// recurrence (as hankel_log_derivative): h_0(x) = -i e^{ix} / x gives (R1 / rho) e^{ik(rho - R1)}, and each order
// multiplies by r_n(k rho) / r_n(k R1). It tends to (R1 / rho)^(n+1) as k -> 0 without forming h_n, which overflows
// there.
void hankel_ratio(double k, double R1, double rho, int N, std::vector<cplx>& out);
// dY_n0/dtheta = -sqrt((2n + 1) / (4 pi)) P_n^1(cos theta), n = 0..N
void dtheta_y0(double u, int N, std::vector<double>& d);

// Exterior part of an imposed background's reciprocity term (full-wave sweeps with a radiating boundary, in 3D and in
// the axisymmetric TM sweep; docs/physics/fullwave_background.md). Outside the domain sphere R1 the readout field is
// outgoing, and A_bg = (psi / r) phi_hat is azimuthal and axisymmetric about the z axis through the domain centre, so
// only the TM (azimuthal H) multipoles of order m = 0 couple to it. Both engines need
//   G_n(k) = int_{rho > R1} A_phi dY_n0/dtheta h_n(k rho) / h_n(k R1) dV,
// - 3D: for the Debye potential a h_n(k rho) Y_n0, E_t(R1) = -(a / R1) h_n(k R1) gamma_n grad_Omega Y_n0 and
//   curl E = -k^2 a h_n(k rho) dY_n0/dtheta phi_hat; with t_n = (E, W_TM,n0), the trace projection of the exterior map,
//   int_{rho > R1} A_bg . curl E dV = sum_n k^2 t_n G_n(k) / (gamma_n(k R1) sqrt(n (n + 1)));
// - axisymmetric: H_phi = sum b_n h_n(k rho) / h_n(k R1) P_n^1(cos theta) outside R1, so
//   int_{rho > R1} A_bg mu0 H_phi dV = -mu0 sum_n b_n G_n(k) / sqrt((2n + 1) / (4 pi)).
// A grid background is zero outside its box [0, r_max] x [z_lo, z_hi], but the field it stands for is not: a magnet's
// A_phi falls off only as 1/rho^2, so cutting G_1 at the box costs O(R1 / rho_max), and at low frequency its imaginary
// part comes from rho ~ 1/k, far outside any grid. So G_n has two parts:
// - inside the sphere rho_s inscribed in the box (about the domain centre) the grid itself: composite Gauss-Legendre in
//   theta on each sphere rho, and in the radius, graded towards R1 (order n falls off as (R1 / rho)^(n+1));
// - outside rho_s its exterior multipoles, A_phi = sum alpha_n (rho_s / rho)^(n+1) P_n^1(cos theta), valid when the
//   magnet's sources lie inside rho_s. Order n's angular moment then scales as (rho_s / rho)^(n+1), and the radial
//   integral is exterior_radial_integrals (closed form): tail_n h_n(k rho_s) / h_n(k R1) J_n(k, rho_s).
// tail_mismatch compares that continuation with the grid where the box reaches beyond rho_s (relative rms of A_phi).
// If rho_s <= R1 the grid is used as given over the box minus the ball (on each sphere at most two arcs: r <= r_max,
// z_lo <= z <= z_hi) and nothing is continued. moments[n][q] holds the frequency-free part
// 2 pi rho_q^2 w_q int A_phi dY_n0/dtheta sin(theta) dtheta; tail[n] the angular moment 2 pi int ... at rho_s.
struct BackgroundExterior {
  int N = 0;
  double R1 = 0;
  double rho_s = 0;  // continuation sphere (0: none)
  double tail_mismatch = 0;
  int tail_samples = 0;
  std::vector<double> rho;
  std::vector<std::vector<double>> moments;  // [n][q], n = 0..N (row 0 unused)
  std::vector<double> tail;                  // [n]

  // G_n(k) without the sliver correction, n = 0..N
  std::vector<cplx> G(double k) const;
  // the k -> 0 limit of G_n: h_n(k rho) / h_n(k R1) -> (R1 / rho)^(n+1), J_n -> rho_s^3 / (2n - 1)
  std::vector<double> G_static() const;
};

// The moments of a grid background bg (box [0, r_max] x [z_lo, z_hi]) outside the sphere R1 centred on the axis at
// z = cz, orders n = 1..N, with `panels` radial panels (at least max(4, panels / 32) per radial piece) and about
// panels / pi angular panels per radian (at least max(2, panels / 64) per arc), of 8 Gauss-Legendre points each. Each
// order's moments do not depend on N. No nodes (rho empty) when the box lies inside the sphere.
BackgroundExterior background_exterior(const BackgroundField& bg, double r_max, double z_lo, double z_hi, double R1,
                                       double cz, int N, int panels);
// The radial panels of background_exterior: 128, or AXF_FW_BG_PANELS (an integer from 4 to 4096; experiments).
int background_exterior_panels();
// The quadrature check of the moments `coarse` (computed with `panels`): the k -> 0 moments again with twice the
// panels, to order min(N, 12), and max_n |G_static(coarse) - G_static(fine)| / max_n |G_static(fine)| over those
// orders; scale, if given, receives the denominator.
double background_exterior_quadrature_check(const BackgroundField& bg, double r_max, double z_lo, double z_hi,
                                            double R1, double cz, int panels, const BackgroundExterior& coarse,
                                            double* scale = nullptr);

// Sparse LDL^T without pivoting for complex symmetric (non-Hermitian) matrices; AMD ordering.
class ComplexSymLDLT {
 public:
  bool analyze(const Eigen::SparseMatrix<cplx>& A);
  bool factorize(const Eigen::SparseMatrix<cplx>& A);  // false on a (near) zero pivot
  Eigen::VectorXcd solve(const Eigen::VectorXcd& b) const;

 private:
  int n_ = -1;
  long nnzA_ = -1;
  std::vector<int> newidx_, Cp_, Ci_, parent_, Lp_, Li_;
  std::vector<long> src_;
  std::vector<cplx> Lx_, D_;
};

struct Eigenmode {
  double omega = 0;     // rad/s
  double f_Hz = 0;
  double residual = 0;  // relative Lanczos residual estimate
  double residual_true = 0;  // ||K x - lambda M x|| / ||K x|| (reduced system, constraint directions removed)
  std::vector<double> x;  // eigenvector at all DOFs
};

// Shift-invert Lanczos for K x = lambda M x (M-inner product, full reorthogonalization; lambda = omega^2, so
// Eigenmode::omega = sqrt(lambda)). `inv` maps b to the (constrained) solution of (K - sigma M) y = b. one_sided:
// sigma lies below the spectrum, the nev smallest lambda > sigma are returned; otherwise the nev lambda nearest
// sigma on either side. Sorted by omega. diag: lanczos_steps, attempts, converged, lanczos_restarts.
std::vector<Eigenmode> lanczos(const Eigen::SparseMatrix<double>& M,
                               const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inv, double sigma, int nev,
                               json& diag, bool one_sided = true);

struct WaveRole {
  bool master = false;    // tied to master master_id (element not assembled)
  int master_id = -1;
  bool excluded = false;  // perfect conductor without ties (element not assembled, natural boundary)
  double alpha = 0;       // stiffness coefficient
  std::function<double(int, const PointEval&)> alpha_at;  // optional point values
  double beta = 0;        // mass coefficient
  std::function<double(int, const PointEval&)> beta_at;
  // Dispersive (lossy) elements: group >= 0 assembles them separately (with the weights above); at each
  // frequency their stiffness and mass are multiplied by the group's complex factors alpha_g(omega), beta_g(omega).
  int group = -1;
};

// Complex, frequency-dependent factors of a group of lossy elements (1 for a lossless medium).
struct WaveGroup {
  std::string name;
  std::function<cplx(double omega)> alpha, beta;
};

class AxiWave {
 public:
  AxiWave(const Space& sp, std::vector<WaveRole> roles, std::vector<Master> masters,
          std::map<int, int> sheet_master_by_attr = {});
  // alpha0: exterior stiffness coefficient (1/mu0 for TE, 1/eps0 for TM). radiation: outgoing DtN with
  // `modes` multipoles; otherwise a closed sphere: u = 0 on it if closed_dirichlet, else natural.
  // The sphere of radius `radius` is centred on the axis at z = center_z.
  void set_outer(int outer_attr, double radius, double alpha0, bool radiation, bool closed_dirichlet, int modes,
                 double center_z = 0.0);
  // homogeneous linear constraint c . u = 0 (c over all DOFs): persistent circuits, trapped flux
  void add_constraint(const std::string& name, Eigen::VectorXd c_full);
  // lossy element groups (WaveRole::group indexes this list)
  void set_groups(std::vector<WaveGroup> g) { groups_ = std::move(g); setup_closed_ = -1; }

  struct Solution {
    double omega = 0;
    std::vector<cplx> u;                 // all DOFs
    std::vector<cplx> psi, current;      // per master: r u on the body, net reaction
    std::vector<cplx> multiplier;        // per constraint
    double dtn_im = 0;                   // Im b(u, conj u): radiated power -pi omega dtn_im (TE), -(pi/omega) dtn_im (TM)
    std::vector<double> group_K, group_M;  // per lossy group: u^H K_g u and u^H M_g u (per radian, real weights)
    json diagnostics;
  };
  // rhs: optional source over all DOFs (added to the prescribed master currents)
  Solution solve(double omega, const std::vector<cplx>& rhs = {});
  // Eigenmodes with the closed sphere and all prescribed currents zero: the lowest ones (target_omega = 0), or
  // the nev nearest target_omega (shift-invert about target_omega^2, modes on both sides of the shift).
  std::vector<Eigenmode> eigenmodes(int nev, json& diag, double target_omega = 0.0);
  // K u - lambda M u at all DOFs, with lossy groups at their lossless parts as in eigenmodes(). At DOFs tied to a
  // conductor (TE masters) this is the weak reaction of an eigenmode: int K_phi N_i r dl over the conductor surface,
  // K_phi = (n x H)_phi the surface current density (consistent-flux recovery of H_t on the walls).
  std::vector<double> reaction(const std::vector<double>& u, double lambda);
  // Element stiffness and mass matrices of element e (local DOF order of edofs(e)), lossy groups at their
  // lossless parts as in eigenmodes(): (Ke - lambda Me) u_e summed over a set of elements is the weak reaction of
  // that set (used for one-sided wall currents on conducting sheets with the cavity on both sides).
  void element_matrices(int e, Eigen::MatrixXd& Ke, Eigen::MatrixXd& Me) const;
  // Split DOFs: element e uses DOF `nw` in place of the Space DOF `old` (remap[e][old] = nw, nw >= Space ndof,
  // new DOFs numbered consecutively; src[nw - Space ndof] = old gives their position). Lets u be discontinuous
  // across a perfectly conducting sheet with the field region on both sides (TM: natural condition on each
  // face). Call before any solve.
  void split_dofs(std::map<int, std::map<int, int>> remap, std::vector<int> src);
  // DOFs of element e and their number (Space DOFs plus split copies)
  const std::vector<int>& edofs(int e) const;
  int ndof() const;
  // u at a point of element e (psi/r inside master elements, 0 in excluded ones)
  cplx value(const Solution& s, int e, const PointEval& pe) const;
  double value(const std::vector<double>& u, int e, const PointEval& pe) const;  // eigenvectors
  const std::vector<WaveRole>& roles() const { return roles_; }

 private:
  void setup(bool closed);
  void elem_raw(int e, Eigen::MatrixXd& Ke, Eigen::MatrixXd& Me) const;  // unit coefficients of lossy groups
  const Space& sp_;
  std::vector<std::vector<int>> edofs_split_;  // per element (empty: Space DOFs)
  std::vector<int> split_src_;                // Space DOF of each split copy
  double dof_r(int i) const;
  double dof_z(int i) const;
  std::vector<WaveRole> roles_;
  std::vector<Master> masters_;
  std::map<int, int> sheet_master_;
  std::vector<std::string> cname_;
  std::vector<Eigen::VectorXd> cfull_;
  int outer_attr_ = 0;
  double Rb_ = 0, alpha0_ = 0, center_z_ = 0;
  bool radiation_ = true, closed_dirichlet_ = true;
  int modes_ = 80;
  int setup_closed_ = -1;
  int nfree_ = 0, nred_ = 0;
  std::vector<int> kind_, index_, master_, gidx_;
  std::vector<double> w_;
  Eigen::SparseMatrix<double> P_, K_, M_, Kr_, Mr_;
  std::vector<int> bdofs_;
  std::vector<WaveGroup> groups_;
  std::vector<Eigen::SparseMatrix<double>> Kg_, Mg_, Kgr_, Mgr_;  // per group: unit matrices (full, reduced)
  Eigen::MatrixXd U_;   // int N_b P_n^1(cos theta) r dl / Rb
  ComplexSymLDLT ldl_;  // symbolic analysis reused across frequencies
};

}  // namespace axf::axi
