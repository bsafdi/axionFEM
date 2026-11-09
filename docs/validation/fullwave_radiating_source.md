# Validation: prescribed full-wave source with a radiating exterior

!!! warning "Beta"
    The prescribed full-wave source is beta. This page describes the automated check and its open points; it has no
    record in the repository.

Feature: `fullwave.source` (Biot-Savart segments) with
`fullwave.boundary = "radiation"`; the reciprocity coupling
`Psi_j = (1/(i omega I_j)) int_all E_j.B0 dV` is the FE-domain volume overlap plus the exterior part
`oint_{R1} Phi0 E_j.n dA` (B0 = -grad Phi0 outside R1; [physics/prescribed_source.md](../physics/prescribed_source.md)).

The check is the slow test `tests/test_fullwave3d_radiating_source.py` (set `AXF_SLOW=1`; it needs the 3D engine and
runs on 4 MPI ranks when `AXF_MPI` is set). The sections below give the model, what the test asserts and why, and
what is still open.

## Test model

Port 0: slit copper loop (magnetic dipole, TE exterior field; its exterior part is a small fraction of `Psi` and
falls as R1⁻³). Port 1: gap of a short electric dipole, two copper arms of 0.24 m (TM exterior field: the exterior part
is larger than `Psi` itself at R1 = 0.55 m, so `Psi_1` is a cancellation between the volume and exterior parts and is
the sign- and scale-sensitive check). Source: closed horizontal square loop (half side 0.12 m) around the dipole gap,
running along the centre line of a closed copper frame. ND2 with curved (order-2) geometry, `mesh.size` 0.08 m,
`surface_size` 0.02 m, source quadrature order 8; f = 20 MHz (ka ≈ 0.02 on the device) and 150 MHz. The test solves
three models of the same device: radiating exterior at R1 = 0.55 m (both frequencies), radiating exterior at
R1 = 0.9 m and a closed PEC sphere of radius 1.2 m (20 MHz).

## 1. Filaments and source quadrature

A filament that crosses source-domain elements, where B0 ~ 1/d, defeats Gaussian quadrature of `int N_i.B0 dV`. For a
TM port the electrostatic part of `int E.B0` vanishes identically, so `Psi_1` is a cancellation and does not converge
with the quadrature order when the filament runs through vacuum elements. The solver therefore runs a
filament-proximity guard (exact segment/tetrahedron test, rotated copies of azimuthal averages included), reports
`fullwave.source.filament_proximity` and warns when an element is met. In the test model the filament runs inside the
copper frame; PEC elements are excluded from the overlap (E = 0 there, as for a real winding), so the singularity is
never sampled and the quadrature converges. The test asserts that no source-domain element is met in any of the three
models.

## 2. Discrete R1 interface

The exterior identity is evaluated on the meshed interface (face normal, Phi0 at the actual rho, radial projection of
the multipoles). On curved order-2 meshes the faceted interface lies very close to R1
(`source.exterior.R1_faceted_min_radius_m` reports its innermost radius), so this test does not exercise the
difference between the two; it matters on straight-sided meshes. The radial projection is exact order by order, with
cross-order leakage O((n − n')(1 − rho/R1)). Containment of the source is checked against 0.95 × the innermost
interface point.

## 3. R1-independence and the PEC reference

The coupling must not depend on where the exterior interface is placed. For the TE port, the radiating models at
R1 = 0.55 and 0.9 m and the PEC sphere agree within the test gate of 0.5 % of `Psi`. For the TM port the gate is
relative to the exterior part: `|Psi(0.55) − Psi(0.9)| < 0.05 |ext(0.55) − ext(0.9)|`, where a missing exterior term
would give 100 % and a sign flip 200 %. In development runs on finer meshes (not recorded in the repository) this
measure decreased steadily under refinement, which identifies the residual R1 dependence of the TM port as
discretization error, dominated by the dipole's own charge distribution (copper arms, edge singularities), which
converges slowly.

A closed PEC sphere is not a usable TM reference at these radii. The dipole port carries a net current element, whose
vector potential ~ 1/R is tangential on the sphere; the PEC image field that cancels it is a uniform field
~ omega mu0 I L/(4 pi R), whose overlap with the source (`int B0 dV = (2/3) mu0 m`) falls only as 1/R, not as the R⁻³
of the radiating `Psi_exterior`. The PEC coupling of the TM port therefore keeps changing with R, and the test uses
the PEC sphere only as a 10 % sanity bound for that port.

## 4. Power balance, dipole formula, Kirchhoff closure, multipoles

- Power balance: the power leaving through R1, integrated from the FE field (Poynting), equals the port balance
  `Re Z |I|²/2` (lossless device) within 1 % for both ports at both frequencies. The Poynting flux uses curl E and so
  converges one order more slowly than the port impedance.
- Magnetic-dipole formula: at 20 MHz (ka ≈ 0.02) the radiation resistance of port 0 matches the static
  magnetic-dipole formula within 1.5 %; the finite loop and the neighbouring copper arms account for the small offset.
  At ka of order 0.15 the static formula no longer applies and the Poynting check is the reference.
- Kirchhoff closure: `max_relative_node_current_imbalance` = 0 for the test loop; open circuits are rejected at parse
  time with a radiation boundary (`tests/test_multiport_fem_spec.py`), and the C++ unit test covers clustering and the
  imbalance measure.
- Source multipoles: the Parseval `unrepresented_fraction` of the exterior expansion is below 1e-3.

## 5. Slow-test gates

| check | gate |
|---|---|
| filament proximity | no source-domain element met (all three models) |
| TE port: R1 = 0.55, 0.9 and PEC 1.2 | pairwise within 0.5 % of `Psi` |
| TM port: exterior part | `|ext(0.55)| > 2 |Psi(0.55)|` |
| TM port: R1-independence | `|Psi(0.55) − Psi(0.9)| < 0.05 |ext(0.55) − ext(0.9)|` |
| TM port against PEC 1.2 | within 10 % |
| Poynting flux against `Re Z` | within 1 %, both ports, both frequencies |
| magnetic-dipole formula, port 0 | within 1.5 % |
| Parseval `unrepresented_fraction` | < 1e-3 |
| relative residual of every port solve | < 1e-6 |

## Open

- Absolute TM limit: the radiating models are R1-consistent up to discretization error, but no independent reference
  constrains an R1-independent error of a few % in a TM-like coupling (PEC does not converge in R; above). A
  single-solution internal-sphere identity check (vol(R' < r < R1) + ext(R1) = ext(R')) or an analytic case would
  close this. TE-like couplings are confirmed against PEC.
- Exact sphere quadrature for the prescribed-source beta_nm; memory of stored Y_nm at large n_max; a rank-local
  `bad_source` throw in the volume assembly.
- Analytic unit test of the surface identity (needs the `collect`/`oint` code factored out of
  `fullwave_analysis`): not done.
- TM-like couplings with a large exterior cancellation converge slowly; quote them only with a refinement study. The
  toroid/solenoid readouts of arXiv:2211.00008 are TE-like (small exterior part).
