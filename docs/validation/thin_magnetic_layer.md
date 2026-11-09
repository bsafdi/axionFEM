# Thin magnetic layers: validation

Formulation: [physics/thin_magnetic_layer.md](../physics/thin_magnetic_layer.md). Driver
`benchmarks/thin_magnetic_layer.py` (one node, 8 threads, about 6 min). Record
`benchmarks/records/thin_magnetic_layer.json`. Tests: `tests/test_thin_magnetic_layer.py` (37 tests, about 65 s).

**Scope:** the axisymmetric static poloidal and azimuthal pickup problems, with linear layers. Section 6 covers the
junctions of layers with superconductors and magnetic regions; the configurations the model cannot represent are
refused with an error.

All cases use a curve on the layer's mid-surface, μ_r = 1000 unless stated, and a uniform 1 T background along z.
Section 4 checks the background drive of linear regions. S is the shielding factor B0/B_in at the
centre.

## 1. Spherical shell: h-convergence to the model's closed form

Mid-radius R = 1 m. Mesh order p, surface size h at the layer. S_model is the closed form of the thin-layer model, the
2 × 2 system in the physics note. The FE result must converge to it. S_exact is the exact thick shell.

d/R = 1e-3 (S_model = 1.664336331, S_exact = 1.664337052):

| p | h | unknowns | layer faces | min h_⊥/d | S/S_model − 1 | S/S_exact − 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.05 | 8,309 | 63 | 28.6 | 1.49e-6 | 1.06e-6 |
| 2 | 0.025 | 11,670 | 126 | 14.7 | 6.98e-7 | 2.66e-7 |
| 2 | 0.0125 | 35,973 | 252 | 6.47 | 9.86e-8 | −3.34e-7 |
| 2 | 0.00625 | 123,596 | 482 | 3.65 | 9.91e-9 | −4.23e-7 |
| 3 | 0.05 | 18,476 | 63 | 28.6 | 3.77e-9 | −4.29e-7 |
| 3 | 0.025 | 25,937 | 126 | 14.7 | 5.56e-10 | −4.32e-7 |
| 3 | 0.0125 | 80,315 | 252 | 6.47 | 2.5e-11 | −4.33e-7 |
| 3 | 0.00625 | 276,893 | 482 | 3.65 | 7e-13 | −4.33e-7 |

- The discretization error falls asymptotically as about h³ at order 2 and h⁴⁻⁵ at order 3, until round-off
  (about 1e-12).
  - Observed rates: 1.1, 2.8 and 3.3 at order 2; 2.8, 4.5 and 5.2 at order 3.
  - The first order-2 step is pre-asymptotic: it has only 1.4 times the unknowns.
- The error against the exact shell converges to the model error, −4.33e-7 (section 2).

d/R = 1e-2 (S_model = 7.553872, S_exact = 7.554584):

| p | h | min h_⊥/d | S/S_model − 1 | S/S_exact − 1 |
|---:|---:|---:|---:|---:|
| 2 | 0.05 | 2.86 | 3.29e-6 | −9.10e-5 |
| 2 | 0.025 | 1.47 | 1.54e-6 | −9.27e-5 |
| 2 | 0.0125, 0.00625 | — | refused (non-positive pivot) | |
| 3 | 0.05 | 2.86 | 8.2e-9 | −9.42e-5 |
| 3 | 0.025 and finer | — | refused | |

The refusals are the indefiniteness guard (physics note, validity condition 4). The half-strip term is negative, and
with elements thinner than about 0.7 d (order 2) or 1.5 d (order 3) the discrete energy loses definiteness. Those
solves stop with an error rather than returning a number. The onset was measured up to order 8; the warning
threshold max(4, ⌈p²/4⌉) covers it. Note that the finest order-2 and order-3 spheres here (h_⊥/d = 3.65) are below
that threshold and carry the warning. They are accurate, which shows the warning is conservative at low order.

## 2. Spherical shell: model error against d/R

S_model against S_exact (both closed forms), for all μ_r and d/R:

| μ_r | d/R | S_exact | S_model/S_exact − 1 | (S_model − 1)/(S_exact − 1) − 1 | first-order formula, rel. error |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.1 | 6.6490 | −9.44e-3 | −1.11e-2 | +1.33e-1 |
| 100 | 0.03 | 2.8747 | −6.43e-4 | −9.86e-4 | +2.97e-2 |
| 100 | 0.01 | 1.6437 | −4.27e-5 | −1.09e-4 | +5.90e-3 |
| 100 | 0.001 | 1.06524 | −6.67e-8 | −1.09e-6 | +9.19e-5 |
| 1000 | 0.1 | 58.522 | −1.09e-2 | −1.11e-2 | +1.54e-1 |
| 1000 | 0.03 | 20.0895 | −9.33e-4 | −9.82e-4 | +4.33e-2 |
| 1000 | 0.01 | 7.55458 | −9.43e-5 | −1.09e-4 | +1.31e-2 |
| 1000 | 0.003 | 2.98705 | −6.49e-6 | −9.76e-6 | +3.00e-3 |
| 1000 | 0.001 | 1.664337 | −4.33e-7 | −1.08e-6 | +5.99e-4 |
| 10⁴ | 0.1 | 577.26 | −1.10e-2 | −1.11e-2 | +1.56e-1 |
| 10⁴ | 0.01 | 66.664 | −1.07e-4 | −1.09e-4 | +1.48e-2 |
| 10⁴ | 0.001 | 7.65535 | −9.42e-7 | −1.08e-6 | +1.30e-3 |

- The thin-layer model's error in S − 1 is −(13/12 + 1/(2μ_r))(d/R)² + O((d/R)³) (series expansion of the exact shell):
  - −1.083 (d/R)² for μ_r ≫ 1, and −1.088 (d/R)² at μ_r = 100;
  - fitted over d/R from 10⁻³ to 0.1: −1.084 to −1.106 (the table's −1.08e-6 to −1.11e-2).
  - So the model agrees with the exact shell through second order in d/R, at every μ_r, from μ_r d/R = 0.1 to
    1000.
- The first-order formula S = 1 + (2/3)(d/R)(μ_r − 1)²/μ_r is good only to O(d/R). The model is not that formula:
  it is exact for the layer it represents, and it carries the second-order terms of the shielding regime.

The FE thin layer (order 3, μ_r = 1000) agrees with the model to better than 1e-6 and with the exact shell as follows:

| d/R | h | min h_⊥/d | S_FE/S_exact − 1 |
|---:|---:|---:|---:|
| 0.01 | 0.1 | 5.2 | −9.42e-5 |
| 0.003 | 0.05 | 9.5 | −6.49e-6 |
| 0.001 | 0.05 | 28.6 | −4.29e-7 |

**Meshed through the thickness.** The same shell as a `linear` region (an annulus contour, order 2, element size
d/2 and d/4 in the shell), which uses the volume machinery:

| d/R | h = d/2 | h = d/4 |
|---:|---:|---:|
| 0.1 | +2.3e-6 | +1.4e-6 |
| 0.03 | +1.4e-6 | +1.3e-6 |
| 0.01 | +1.2e-6 | +1.2e-6 |

The meshed shell reproduces the exact formula to about 1e-6, so it is an independent check of the background drive
(section 4). At d/R = 0.01 the thin layer costs 18,476 unknowns at order 3 for −9.4e-5. The meshed shell costs
40,130 to 84,649 unknowns at order 2 (h = d/2 and d/4).

## 3. Long tube and closed can (axial field)

The engine is axisymmetric, so the classic long cylinder in a *transverse* field (S = 1 + μ_r d/2R) is a planar
problem it cannot pose. In an axial field an open tube does not shield; it concentrates flux into its wall. A closed
can shields. Both are compared with the wall meshed through its thickness. Both have R = 1 m, d = 5 mm and
μ_r = 1000. The open tube has L = 20 m, the can L = 4 m with end caps that reach the axis. B_z is in T at the
sampled points.

| case | model | h | (10⁻⁶, 0) | (0.5, 0) | (0.9, 0) | (1.5, 0) | (0.5, 0.45 L) |
|---|---|---:|---:|---:|---:|---:|---:|
| open tube | thin | 0.1 | 0.914084 | 0.914353 | 0.914908 | 0.921546 | 0.655683 |
| | thin | 0.05 | 0.914193 | 0.914441 | 0.914999 | 0.922899 | 0.655071 |
| | thin | 0.025 | 0.914239 | 0.914488 | 0.915044 | 0.922883 | 0.654288 |
| | meshed | d/2 | 0.914306 | 0.914554 | 0.915110 | 0.923286 | 0.654322 |
| | meshed | d/4 | 0.914307 | 0.914555 | 0.915111 | 0.922502 | 0.653927 |
| can | thin | 0.1 | 0.457027 | 0.461127 | 0.470935 | 0.648769 | 0.425126 |
| | thin | 0.05 | 0.457243 | 0.461348 | 0.471162 | 0.648996 | 0.424843 |
| | thin | 0.025 | 0.457305 | 0.461435 | 0.471232 | 0.650320 | 0.425045 |
| | meshed | d/2 | 0.457406 | 0.461544 | 0.471343 | 0.650644 | 0.424617 |
| | meshed | d/4 | 0.457335 | 0.461534 | 0.471345 | 0.650002 | 0.424583 |

- **Inside**, where the shield acts, the thin layer at h = 0.025 and the meshed wall at d/4 agree:
  - tube: to 7.4e-5;
  - can: to 6.6e-5 at the centre, and 2.1e-4 and 2.4e-4 at r = 0.5 and 0.9 m.

  The meshed wall is converged inside to 1e-6 (tube) and 4e-6 to 1.6e-4 (can), so this is not its own error. The
  thin layer still changes by about 5e-5 per halving of h, and its interior element size is 4h (0.1-0.4 m). Pointwise
  B is discontinuous between such elements at the 1e-4 level. The differences are at the thin layer's resolution
  here, with the model error (O((d/R)²) ≈ 3e-5) below it.
- **Outside and near the ends**, both models change at the 1e-3 level with the mesh. The sampled points lie in the
  coarser field mesh (element size 0.1 m) away from the wall, so these values are limited by that mesh, not by the
  layer.
- **Infinite-tube relation.** At mid-length of the open tube the layer's excess flux, 2πR[A], should equal
  2πR(μ_r − 1) d B_in. The engine reports it as `excess_flux_max_Wb`; B_in is sampled at r = 0.98 m. The two agree
  to 4.3e-5, 3.7e-5 and 3.7e-5 for h = 0.1, 0.05 and 0.025. This tests the tangential term directly.
- The peak flux density in the layer, `B_layer_max_T`, is 915 T in the open tube and 475 T in the can. These are
  linear-model numbers in a 1 T field, a reminder to compare with saturation (about 0.7 T for mu-metal) and scale
  the drive.
- **Note:** the centre samples are taken at r = 10⁻⁶, not exactly at (0, 0): point location can fail on the axis
  vertex (a sampling limitation unrelated to the layer).

## 4. The background drive of linear regions

A solid μ_r = 10 sphere (R = 1 m) in a uniform 1 T background should have B_in = 3μ_r/(μ_r + 2) = 2.5 T inside.

| | B_z at the centre |
|---|---:|
| exact | 2.5 |
| axionFEM | 2.499983 |

This checks that a linear region responds to a `background` field; designs without a background are unaffected
(section 7).

## 5. A shielded pickup: L_p

A superconducting pickup loop (an nb torus of minor radius 0.02 m at r = 0.5 m) inside a closed can (R = 1 m,
L = 2 m, d = 5 mm, μ_r = 1000).

| shield | h | L_p (H) |
|---|---:|---:|
| none | 0.05 | 2.067854e-6 |
| thin | 0.1 | 2.100592e-6 |
| thin | 0.05 | 2.100602e-6 |
| thin | 0.025 | 2.100608e-6 |
| meshed | d/2 | 2.100613e-6 |
| meshed | d/4 | 2.100614e-6 |

- The can raises L_p by 1.6 %. The thin layer reproduces the meshed wall to 3e-6.
- L_p from the energy and from the flux agree to 1e-6 (tests).
- Ψ is not compared. A pickup that couples to the axion needs a toroidal static field (azimuthal pickup) or a port
  (poloidal pickup), and both use the toroidal-field problem, which does not support layers yet. With only azimuthal
  windings, Ψ of an azimuthal pickup vanishes by symmetry.

**Shape sensitivities.** Take dL_p/dR_can and dW/dR_can for the same can with h = 0.025 and the coil resolved:

| method | dL_p/dR_can | dW/dR_can |
|---|---:|---:|
| adjoint | −6.644426e-8 | −3.153278e-9 |
| fd_morph | −6.644506e-8 | −3.153356e-9 |
| fd_remesh, step 1e-2 | −6.645171e-8 | −3.143872e-9 |
| fd_remesh, step 3e-2 | −6.693692e-8 | −3.178396e-9 |

The adjoint agrees with fd_morph to 1.2e-5 and with fd_remesh within the remeshing noise. In a test at h = 0.1 with
a coarsely meshed coil, fd_remesh of W scattered by 100 %. That noise came from the coil's discretization, as the
resolved run shows, not from the layer.

## 6. Junctions: layers meeting superconductors and magnetic regions

These configurations are the hardest for the model: a junction handled wrongly is off by O(1), so each one is
compared with an exact solution or with the wall meshed through its thickness.

**A layer on a superconducting ball** (R = 1 m, d = 0.01 m, μ_r = 1000, order 3, 1 T). Exact solution: a
superconductor of radius R − d/2 coated by the shell out to R + d/2 (`coated_sc_axis_Bz`). B_z on the axis at
z = 1.5 m, exact 1.5191221:

| h | regions | B_z | (B − 1)/(B_exact − 1) − 1 |
|---:|---|---:|---:|
| 0.1 | [sc, jacket] | 1.5192507 | +2.5e-4 |
| 0.1 | [jacket, sc] | 1.5192507 | +2.5e-4 |
| 0.05 | [sc, jacket] | 1.5192580 | +2.6e-4 |
| 0.05 | [jacket, sc] | 1.5192580 | +2.6e-4 |

- The two region orders agree to every printed digit (an error in the junction would flip the sign of B − 1 for
  one of them).
- The residual 2.5e-4 does not fall with h, so it is the model error for this configuration.

**A layer on a linear core** (μ_c = 5 and 50) against the exact three-region sphere (`core_shell_Bin`), order 3:

| μ_c | d/R | B_in/B_exact − 1 (h = 0.1) | (h = 0.05) |
|---:|---:|---:|---:|
| 5 | 0.01 | +8.76e-5 | +8.76e-5 |
| 5 | 0.003 | +5.83e-6 | +5.87e-6 |
| 50 | 0.01 | +6.53e-5 | +6.54e-5 |
| 50 | 0.003 | +4.96e-6 | +4.97e-6 |

The error scales as (d/R)², like the model error of section 2.

**A superconducting sheet meeting a can wall.** A closed can (R = 1 m, L = 2 m, d = 5 mm, μ_r = 1000) in 1 T has an
annular superconducting sheet at z = 0.3 m from r = 0.4 m. The sheet ends on the wall (touch), crosses it out to
r = 1.2 m (cross), or ends 2 cm short (gap). Thin wall: h = 0.05. Meshed wall: h = d/2.

| sheet | wall | B_z(0, 0) | B_z(0.5, 0) | B_z(0.7, 0.5) | B_r(0.5, 0) |
|---|---|---:|---:|---:|---:|
| touch | thin | 0.32653 | 0.23855 | 0.16487 | −0.01656 |
| touch | meshed | 0.32646 | 0.23974 | 0.16553 | −0.01806 |
| gap 2 cm | thin | 0.32626 | 0.23896 | 0.16462 | −0.01654 |
| gap 2 cm | meshed | 0.32628 | 0.24082 | 0.16652 | −0.01775 |
| cross | thin | 0.98299 | 0.50883 | 0.21392 | −0.43731 |
| cross | meshed | 0.97803 | 0.51306 | 0.21846 | −0.44060 |

- **Touch.** It agrees with the meshed wall to 7e-5 at the centre. (A sheet that tied the far side of the wall would
  cut its flux and give about +98 %.)
- **Cross.** The sheet cuts the wall in both models. The 0.5 %-2 % differences come from the crossing sheet's
  singular free edges at r = 0.4 and 1.2 m, which converge slowly in both models. With finer meshes (h = 0.025 thin,
  size 0.025 meshed) the centre values are 0.98065 and 0.97852. The test therefore uses 1.2e-2 for this case and
  3e-3 for the touching one.

**Superconductors on both sides of a layer.** A closed can (as above) has two annular
superconducting sheets at z = 0.3 m: one inside, from 0.4 m to the wall, and one outside, from the wall to 1.6 m.
There is also an nb ball (R = 0.5 m) with a 5 mm layer on its surface and a superconducting flange from the layer
out to 1.2 m at z = 0.

| case | model | B_z at the points | bodies |
|---|---|---|---|
| two-sheet can, (0, 0), (0.5, 0), (0.7, 0.5), (0.7, 0.1) | thin, h = 0.05 | 0.33591, 0.24479, 0.17049, 0.17711 | inner, outer |
| | meshed, h = d/2 | 0.33571, 0.24594, 0.17111, 0.17783 | inner, outer |
| flanged ball, (0, 0.8), (0.3, 0.6), (0.7, 0.2), (0.55, 0.05) | thin, h = 0.025 | 1.45223, 1.55321, 0.48888, 0.21924 | sc, flange |
| | meshed, h = d/2 | 1.45309, 1.55641, 0.49098, 0.21871 | sc, flange |

- Body detection keeps the two sheets, and the ball and its flange, as separate bodies; merging them would cut the
  layer's flux (1.266 T at the can's centre, and 0.780 T at the ball's (0, 0.8)).
- Two nb rings separated by a layer that extends past them are two bodies too. They match a meshed wall to 5e-3
  (test). A layer that ends exactly where the rings meet is refused, with the hint to extend it.

**A coating on part of a linear core:** a 1 cm layer on the upper hemisphere of a μ_c = 50 or
5 ball, ending at the equator while running along the core. It is accepted. Points (0, 0), (0, 0.6), (0.5, 0.3),
(0.5, −0.3), (0, 1.5), (1.3, 0.2), (0.8, −0.9):

| μ_c | model | B_z |
|---:|---|---|
| 50 | thin, h = 0.1 | 2.56834, 2.22320, 2.31255, 2.81358, 1.57238, 0.60472, 1.36891 |
| 50 | meshed, h = d/4 | 2.56531, 2.22177, 2.30932, 2.81179, 1.57216, 0.60461, 1.36886 |
| 5 | thin, h = 0.1 | 1.62402, 0.93101, 1.17075, 2.05778, 1.47476, 0.68050, 1.24641 |
| 5 | meshed, h = d/4 | 1.61855, 0.92730, 1.16371, 2.05529, 1.47503, 0.67907, 1.24672 |

- Thin against meshed: 1.4e-3 (μ_c = 50) and 6.0e-3 (μ_c = 5) at worst. That is the O(d/ℓ) accuracy of a free end,
  as for the open tube in air.
- The layer's own effect is 30 % and 130 %.

**A thin wall ending on the faces of meshed magnetic caps** (a hybrid can, with the wall ending on the caps' inner
faces) is refused:

> thin layer 'wall' ends on the magnetic region 'cap0' at (r, z) = (1, -1.9975), which it meets only at its end:
> the model cannot pass the layer's flux into the region through that contact, so the junction would carry none
> [at /curves/0/path]

- The hint lists meshing the junction, making both parts layers, or running the layer along the region.
- Caps of μ_r = 1.005 are accepted (test): regions within 1 % of μ_r = 1 do not count.
- A superconducting sheet drawn on the wall's own path is refused, naming both curves.

**Further junctions.**
- **The end refusal is per region.** The hybrid can with a μ_r = 1.05 collar along the wall's last 5 cm is refused,
  naming the cap. So are collars of μ_r = 1.011, 1.05 and 1.5, inside and outside, and iron caps with a collar.
  Accepted, they would be off by +21 to +24 % against the meshed wall.
  - Still accepted: the partial coatings, the side coating and the embedded strip, and caps of μ_r = 500 that the
    wall runs 5 cm into. The latter is 0.3 % from the meshed wall at the centre.
  - A skirt of the caps' material drawn as a *separate* region is refused; drawn as part of the cap region, it is
    accepted.
  - Caps of μ_r = 1.05 without a collar are refused (test), which pins the 1 % threshold from above.
- **A region crossing a layer cuts it.** The can (as above) has an nb plate region r ∈ [0.4, 1.2],
  z ∈ [0.295, 0.305] across its wall.

  | plate | B_z(0, 0) | bodies |
  |---|---:|---|
  | thin wall, plate region | 0.99311 | plate (one) |
  | thin wall, plate as a sheet | 0.98299 | plate |
  | meshed wall, plate cutting it | 0.99053 | plate (one) |

  The plate is one body (split into two bodies it would give 0.32805 T).
- **The crossed pickup ring.** An nb ring (r 0.45-0.55) crossed by a 1 mm layer along r = 0.5:

  | layer | L_p (energy) | L_p (flux) |
  |---|---:|---:|
  | none | 1.606502e-6 | 1.606502e-6 |
  | μ_r = 1.5 | 1.607585e-6 | 1.607585e-6 |
  | μ_r = 1000 | 1.636296e-6 | 1.636296e-6 |

  A pickup conductor that forms several bodies is refused, with or without layers (the energy and flux values would
  differ by 2×). That includes a meshed wall painted over the ring.
- **Sheets meeting at a layer's end** are refused, as documented. This covers two sheets, and a region and a sheet,
  at the open tube's rim. One sheet across the rim is accepted: 0.45924 against 0.46036 meshed at the centre.

**The guard on every backend.** With `AXF_STRUMPACK_MIN_DOFS=0` the d/R = 0.1 sphere at h = 0.05 is still refused,
because designs with layers keep the Eigen LDLT. The refusal is located at `/curves/0/surface_size`, and its hint
gives the order-aware ratio (test `test_layer_too_thick_for_the_mesh_is_refused`).

**The reaction energy with a background.** For the solid μ_r = 10 ball in 1 T, the reported W_mag equals the closed
form (1/2)∫ν|B − B_bg|² dV to 2e-3, including the exterior (`solid_sphere_reaction_energy`). That excludes the
background's linear work, which would otherwise be reported as −W.

## 7. Results without layers are unchanged

`benchmarks/thin_layer_bitid.py` solves designs without layers with two builds, single-threaded, and compares the
result JSON with timings and file paths removed. It is a tool for comparing a build against a reference build; its
output is not stored as a record.
- **Single-threaded is required.** Multithreaded assembly uses an OpenMP dynamic schedule and is not
  bit-reproducible even on one build.
- **The designs:**
  - the toroid and solenoid templates (coarse);
  - CORE with nonlinear iron caps;
  - a superconducting sheet with a persistent circuit, a linear region and an azimuthal pickup;
  - a superconductor in a uniform background;
  - superconducting loops in a background with an azimuthal pickup;
  - adjoint sensitivities of the solenoid and of that design.
- **Expected result:** identical result JSON for all entries. The one part of the thin-layer feature that can
  change a design without layers is section 4, which needs a background and a linear region.

## Limitations

- **Not implemented:**
  - the toroidal-field (G) problem with layers. A shielded toroid needs it, and so does every Ψ with a layer: all
    accepted layer designs have Ψ = 0;
  - a conducting (eddy-current) layer for the RF pickup;
  - nonlinear (saturating) layers;
  - a junction condition from a layer into a meshed magnetic region (refused for now);
  - the critical-state analysis with a background and linear regions (refused for now).
- **3D:** not implemented; see the assessment at the end of the
  [physics page](../physics/thin_magnetic_layer.md#the-3d-engine-what-a-thin-layer-would-take-assessment-not-implemented).
