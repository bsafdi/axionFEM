# 3D full wave with an imposed background: validation

!!! warning "Beta"
    The imposed background field of the 3D full-wave solve is beta
    ([physics/fullwave_background.md](../physics/fullwave_background.md)).

Formulation: [physics/fullwave_background.md](../physics/fullwave_background.md). Driver
`benchmarks/fullwave3d_background.py`, record `benchmarks/records/fullwave3d_background.json` (8 MPI ranks unless
stated). Tests: `tests/test_fullwave3d_background.py` (AXF_SLOW) and `tests/test_background_3d.py` (default suite:
a reaction field next to the background, a closed domain with `source_check`, a uniform background with the radiating
boundary, the MQS form, and no-background invariance).

**Model.** A small copy of the `solenoid` template (the DMRadio-m³ topology): a solenoid of inner radius 0.2 m,
thickness 0.04 m and height 0.3 m (10⁶ ampere-turns) around a shorted copper coax (walls at 0.1 and 0.18 m, 2 cm
thick) read out across its bottom gap, revolved to 3D with the annular (360°) port. ND2, mesh size 0.06 m, surface
0.02 m, 997,354 unknowns; domain radius R1 = 0.45 m with the radiating exterior. Frequencies kR = 0.02, 0.4 and 0.8
(R = 0.24 m, the device radius; the coax's quarter-wave resonance is at kR ≈ 1.26). The background is the coil's
analytic field (Derby–Olbert sheets integrated over the winding; against the on-axis formula 4e-16, loop
superposition 1e-14, ∇·B 1e-9 of μ0J) on an (r, z) grid out to r = 2 m, |z| = 2 m (2 mm near the coil, 1 cm
beyond). In the background models the coil region is kept as a μr = 1 material, so every 3D model has the same mesh.

## 1. The coil as a winding and as a background, on one mesh

The winding model is the validated path: B_lab = curl A_lab with ∫E·B_lab dV and the multipole exterior identity
∮Φ E_r dA. The background model has no winding: A_lab is zero (copper is transparent to the static field) and the
whole coupling is the new term, ∫curl E·A_bg dV inside R1 plus its exterior term. The axisymmetric engine (its uniform
annular feed is this port) solves the winding model with its own exterior term (∫A_lab μ0H′ outside its domain from
static multipoles), the same integral as the background's exterior term.

| kR | Ψ winding (3D) | Ψ background (3D) | Ψ winding (axisymmetric) | background vs axisymmetric | winding vs axisymmetric |
|---:|---|---|---|---:|---:|
| 0.02 | 3.03510e-9 + 1.26e-14 i | 3.03902e-9 + 1.27e-14 i | 3.03901e-9 + 1.26e-14 i | 2.4e-6 | 1.3e-3 |
| 0.4 | 3.71932e-9 + 1.257e-10 i | 3.72408e-9 + 1.258e-10 i | 3.71887e-9 + 1.253e-10 i | 1.4e-3 | 1.8e-4 |
| 0.8 | 6.96475e-9 + 1.982e-9 i | 6.97399e-9 + 1.984e-9 i | 6.91998e-9 + 1.956e-9 i | 8.5e-3 | 7.2e-3 |

- Z of the two 3D models is identical (relative difference 0): the background enters only Ψ.
- The two 3D paths differ by a constant 1.3e-3 at all three frequencies. At kR = 0.02 the axisymmetric engine sides
  with the background path (2.4e-6). The winding path's exterior identity carries 7.6 % of Ψ at kR = 0.02 (the
  electrostatic part of E crossing R1, cancelled by ∮Φ E_r dA), so its O(h) trace error in E_r shows at the 1e-3
  level; the reciprocal form has no such cancellation (its exterior part is 1.1e-4 of Ψ there).
- At kR = 0.8 both 3D models are 0.7–0.9 % from the axisymmetric engine; L(ω) of the 3D mesh differs from it by
  0.77 % there, so this is the 3D mesh, not the coupling.

**The exterior term.** The background's exterior part against the axisymmetric engine's exterior term for the
winding: 3.367e-13 + 1.265e-14 i vs 3.361e-13 + 1.262e-14 i at kR = 0.02 (0.2 %), 1.133e-10 + 1.113e-10 i vs
1.129e-10 + 1.108e-10 i at 0.4 (0.4 %), −1.705e-10 + 1.107e-9 i vs −1.654e-10 + 1.094e-9 i at 0.8 (1.2 %).

**The volume part** against the axisymmetric engine's background volume coupling in the same reciprocal form:
2.3e-6, 1.3e-3 and 8.6e-3 at the three frequencies. The axisymmetric engine has the background's exterior term too,
from the same moments; its total Ψ is compared with these 3D runs in
[axi_background_exterior.md](axi_background_exterior.md).

**Why the grid is continued.** Integrating the grid only as far as its box (2.83 m) and setting the background to
zero beyond is not enough: A_bg falls off as 1/ρ² (the coil's dipole), so the n = 1 overlap converges only as
R1/ρ_max, and at low frequency its imaginary part comes from ρ ~ 1/k (12 m). The grid is therefore used up to the
sphere inscribed in its box (2 m) and continued beyond by its exterior multipoles, with closed-form radial
integrals.

## 2. The MQS limit

| model | Ψ(kR = 0.02) / reciprocal MQS Ψ | primary form / reciprocal form (diagnostic) |
|---|---:|---:|
| coil as background | 1.000505 | 0.98342 |
| long magnet (section 4) | 1.000807 | 0.95096 |
| uniform background, closed sphere | 1.000687 | 0.88316 |
| axisymmetric engine, coil as winding (its MQS Ψ is the reciprocal form) | 1.000502 | — |

- The reciprocal MQS form ∫(A_lab + A_bg)·B_pick of the 3D MQS solve agrees with the axisymmetric engine's MQS Ψ:
  3.0374835e-9 vs 3.0374848e-9 (coil), 1.3232395e-9 vs 1.3232388e-9 (uniform).
- Ψ(ω)/Ψ_MQS − 1 at kR = 0.02 is the (kR)² dispersion of the shorted coax, the same in 3D and in the axisymmetric
  engine (5.05e-4 vs 5.02e-4).
- The primary form ∫A_pick·B_lab is not a number of the problem when a background is present: the background is cut
  at R1, so the form depends on A_pick's gauge, which the per-element gauge mass, i.e. the mesh, sets. Its ratio to
  the reciprocal value therefore changes with the mesh and with R1 and is only a diagnostic; the Coulomb gauge would
  give 2/3 for a uniform field. The column above is that diagnostic on these meshes.
- **The 3D MQS Ψ with a background is the reciprocal form** (`coupling`, F, g, benchmarks, reach,
  `Psi_over_MQS`); the primary value is `checks.Psi_primary_form`, with a warning when it differs by more than 1e-3.
  The runs of this record were made with the primary form as the reported value: their `coupling.Psi_SI_H_T_m2` (the
  record's `Psi_MQS`, and its `Psi_FW_over_MQS`) is the primary form; every ratio quoted here uses the reciprocal value
  (`checks.Psi_A_lab_dot_B_pick`), which the code reports. The full-wave Ψ(ω) does not depend on that choice.

## 3. Independence of the domain radius and of the grid's extent

- **Grid extent:** the same model with the grid cut at r = 1 m, |z| = 1 m (continued from 1 m instead of 2 m):
  Ψ changes by 2e-12, 2e-9 and 2e-8 at the three frequencies (continuation mismatch 3.1e-6 there, 2.9e-6 for the
  2 m grid). Quadrature check (doubled panels, k → 0 moments): 8e-9.
- **Domain radius:** the background model at R1 = 0.675 m instead of 0.45 m (1,119,294 unknowns): Ψ changes by
  1.2e-4 at kR = 0.4 and 4.6e-4 at 0.8; without the background's exterior term (4.3 % and 15 % of Ψ at R1 = 0.45 m)
  by 2.0e-2 and 0.11. L changes by 7e-6 and 6e-5 between the two meshes. Section 4 does the same with a magnet that
  crosses R1.

## 4. A magnet longer than the domain (as DMRadio-Core's)

The same pickup inside a 1.8 m solenoid (same radii and current density, so it crosses R1 at both ends), its field on a
grid out to r = 2 m, |z| = 3 m (continued from 2 m; continuation mismatch 6.7e-6). The exterior term is 12 % of |Ψ| at
kR = 0.4 and 39 % at 0.8.

| kR | Ψ at R1 = 0.45 m | Ψ at R1 = 0.675 m | relative change | without the background's exterior term |
|---:|---|---|---:|---:|
| 0.02 | 5.387404e-9 | 5.387407e-9 | 6.0e-7 | 1.0e-4 |
| 0.4 | 7.16517e-9 + 6.829e-10 i | 7.16653e-9 + 6.837e-10 i | 2.2e-4 | 5.2e-2 |
| 0.8 | 1.25385e-8 + 7.909e-9 i | 1.25429e-8 + 7.918e-9 i | 7.1e-4 | 2.5e-1 |

The two radii are different meshes (997,354 and 1,119,294 unknowns; L changes by 6e-5 at kR = 0.8). The 3D MQS Ψ
in its primary form changes by 2.8 % between them (5.119e-9 to 5.265e-9: it is gauge-dependent, section 2); its
reciprocal form, the reported value, by 6e-8.

**A grid too small for the continuation.** The long magnet on a grid of r ≤ 0.8 m (inscribed sphere 0.8 m; the
magnet reaches 0.93 m): continuation mismatch 0.23 and the warning. (Ψ at kR = 0.8 moved by 0.6 % here, because the
mismatch lies far out; the diagnostic is what matters.)

## 5. Uniform background

1 T along z, with the closed sphere (`boundary: "pec"`, a cavity of radius R1: complete for that model, not the
open-space answer): Ψ(kR = 0.02) = 1.3241484e-9 against the axisymmetric engine's 1.3241425e-9 for the same closed
model (4.5e-6). With the radiating boundary the exterior term is omitted with its warning
(`exterior_term_background: "omitted (uniform background: ...)"`); Ψ = 1.3240000e-9.

## 6. MPI

The background model at kR = 0.8 on two ranks against eight: Ψ agrees to 8.6e-12, Z to 2.4e-12
(true residuals 1.1e-9 and 1.4e-9). One rank ran out of 96 GB of memory.

## 7. DMRadio-Core with its full magnet

`benchmarks/core_full_magnet.py`, record `benchmarks/records/core_full_magnet.json` (16 ranks). The 3D Core studies
([published_benchmarks.md](published_benchmarks.md) sections 7.1-7.2) shortened the 2.95 m magnet to 0.6 m because a full-wave
domain around it was out of reach. Here the reduced 120 cm design's 3D
meshes are kept (the coil regions as a μr = 1 material: 2,569,994 unknowns for the 30° port, 2,572,804 for the 360°
control, R1 ≈ 1.08 m) and the full split solenoid (bore 0.37 m, 0.6 m outer diameter, 2.95 m, 1.1894e7 ampere-turns)
is the background: its analytic field on a grid out to r = 3 m, |z| = 4 m (continued from 3 m; continuation mismatch
6.7e-5; it crosses R1 at both ends). The axisymmetric engine solves the same full design with the magnet as windings.

| f [MHz] | Ψ 360° (3D) / axisymmetric | L 360° (3D) / axisymmetric | exterior term / Ψ |
|---:|---:|---:|---:|
| 70.0 | 1.0028 | 1.0030 | 2.3 % |
| 88.2 | 1.0054 | 1.0056 | 3.4 % |
| 111.1 | 1.0122 | 1.0124 | 4.9 % |
| 140.0 | 1.0541 | 1.0542 | 6.5 % |

- **What this establishes:** the Ψ ratios are those the reduced design gave with explicit windings (1.0029, 1.0055,
  1.0123, 1.0539, section 7.1 of the benchmark report), and they track L's: the 3D–axisymmetric difference is the 3D
  mesh near the self-resonance, common to Ψ and L. The background coupling adds nothing visible to it, so the reduced
  design's port ratios carry over to the full magnet (30° port against the uniform feed: ×4.51 in scan time, ×4.55
  in section 7.2). L and L_eff of each port equal the reduced runs' (same mesh): the background enters only Ψ.
- MQS: the 3D reciprocal Ψ is 3.15012e-8 (360°) and 3.15120e-8 (30°) against the axisymmetric 3.15073e-8; the
  primary form was 1.3e-3 lower on these meshes (a gauge-dependent diagnostic, section 2).
- **Scan times of this model,** not of DMRadio-Core: DFSZ over 70–140 MHz with the published Q, T and η_A(f), by the
  integrand of `core_readout_ports.scan_time_yr` (a 4-point trapezoid in ln f): 5.93 yr for the uniform feed and
  26.7 yr for a 30° port. Against the published 0.75 yr that is ×7.9 for the CORE topology model with the uniform feed
  (the axisymmetric engine gives ×7.3, benchmark report 6.4) times ×4.5 for a 30° port on that model. DMRadio-Core's
  30° sector sits in the slit of a pillbox closed by copper plates across the bore, a different structure; a localized
  port's penalty depends on it through the slit's m ≠ 0 resonances.
- **Quadrature and the axisymmetric comparison:** the axisymmetric full design on a 61-point sweep,
  its L_eff by finite differences there, gives 5.922 yr with the same four points and trapezoid, which the 3D uniform
  feed matches to 1.7e-3 (F_eff ratios 0.999, 0.999, 1.000, 0.968), and 5.579 yr with all 61 points: the four-point
  value is 6.2 % high. The 5.51 yr of section 6.4 uses 7 points and a finite-difference L_eff, a different method.
- A reproduction of the published number would need DMRadio-Core's own pickup (the plates, the 30° sector of its
  slit, the unpublished slit and plate dimensions), the magnet's segmentation and iron, the conducting bore and
  cryostat at RF (a background is transparent to RF), and a comparison with the published per-frequency rates (Zenodo)
  rather than the integrated time, since |V| and L_eff are not published (benchmark report 7.3).

## Limits

- One mesh per model; no h/p study of the background coupling itself. The comparisons above bound its
  discretization error by that of the readout field (L) on the same mesh.
- The background's sources (here the coil) are not part of the RF model.
- The axisymmetric references use the uniform feed, as these 360° ports do. A localized port is covered by them on an
  axisymmetric structure: A_bg couples only to the m = 0 part of the readout field, which a port of current I drives
  as the uniform feed of I does (Core, section 7: the 30° port and the 360° control differ in Ψ by 2.2e-4 to 1.3e-3,
  the mesh). On a non-axisymmetric structure, such as DMRadio-Core's own pickup, that argument does not hold.
- The axisymmetric engine includes a grid's exterior term (the same moments; 4.3–39 % of Ψ here at kR = 0.4–0.8);
  its Ψ(ω) agrees with these 3D runs to the 3D mesh's level ([axi_background_exterior.md](axi_background_exterior.md)).
  A uniform background's exterior term is omitted in both engines.
