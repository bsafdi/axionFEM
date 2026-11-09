# The axisymmetric engine's exterior term for a grid background: validation

Formulation: [physics/fullwave_background.md](../physics/fullwave_background.md) ("The exterior term", "The
axisymmetric engine"). Driver `benchmarks/axi_background_exterior.py`, record
`benchmarks/records/axi_background_exterior.json`. "Without the term" columns come from a build that omits the
background's exterior term, "with the term" from the current code; the record has the sha256 of both builds. Runs on
one OpenMP thread, so that results compare bit for bit. Tests: `tests/test_axi_background_exterior.py` (five
default-suite tests, three under AXF_SLOW) and `cpp/tests/test_background_exterior.cpp`.

**Model.** The 3D validation's ([fullwave3d_background.md](fullwave3d_background.md)): a shorted copper coax (walls at
0.1 and 0.18 m) read out across its bottom gap inside a solenoid (inner radius 0.2 m, 0.04 m thick, 0.3 m high,
10⁶ ampere-turns), on the axisymmetric engine's own mesh (order 2, 10,677 unknowns at R1 = 0.45 m). The coil's
analytic field is the grid background (2 mm near the coil, 1 cm beyond, out to r = 2 m and |z| = 2 m; the coil region
kept as a μr = 1 material, so the mesh is the winding model's); the **long magnet** is a 1.8 m solenoid of the same radii
and current density (grid to |z| = 3 m), which crosses R1 at both ends. kR = 0.02, 0.4, 0.8 with R = 0.24 m. The
exterior term uses the grid up to its inscribed sphere (2 m) and its exterior multipoles beyond (continuation mismatch
2.9e-6 for the coil, 6.7e-6 for the long magnet, the same numbers as in 3D: the same code), to order 80.

## 1. 2D against 3D, without and with the term

|Ψ_2D − Ψ_3D| / |Ψ_3D| against the 3D runs of the committed record (`benchmarks/records/fullwave3d_background.json`,
about 1M unknowns):

| model | kR | without the term | with the term | L_2D/L_3D − 1 | exterior term / Ψ | exterior 2D vs 3D |
|---|---:|---:|---:|---:|---:|---:|
| coil grid, R1 = 0.45 m | 0.02 | 1.1e-4 | 2.6e-6 | 2.4e-5 | 1.1e-4 | 2.1e-3 |
| | 0.4 | 4.4e-2 | 1.4e-3 | −1.2e-3 | 4.3 % | 4.0e-3 |
| | 0.8 | 1.6e-1 | 8.4e-3 | −7.6e-3 | 15 % | 1.2e-2 |
| coil grid, R1 = 0.675 m | 0.4 | 2.9e-2 | 1.5e-3 | −1.2e-3 | 2.8 % | 5.1e-3 |
| | 0.8 | 9.9e-2 | 8.8e-3 | −7.7e-3 | 10 % | 1.3e-2 |
| long magnet, R1 = 0.45 m | 0.02 | 3.5e-4 | 4.5e-6 | 2.4e-5 | 3.5e-4 | 2.4e-3 |
| | 0.4 | 1.2e-1 | 1.7e-3 | −1.2e-3 | 12 % | 4.2e-3 |
| | 0.8 | 3.9e-1 | 9.2e-3 | −7.6e-3 | 39 % | 1.2e-2 |
| long magnet, R1 = 0.675 m | 0.02 | 2.5e-4 | 5.1e-6 | 2.5e-5 | 2.5e-4 | 3.4e-3 |
| | 0.4 | 8.9e-2 | 1.9e-3 | −1.2e-3 | 8.7 % | 5.2e-3 |
| | 0.8 | 2.9e-1 | 9.8e-3 | −7.7e-3 | 29 % | 1.3e-2 |

- **With the term the remaining difference is the 3D mesh's.** It equals what the 3D validation found for the other
  terms: the 3D grid model against the 2D winding model 2.4e-6 / 1.4e-3 / 8.5e-3, its volume part against the 2D
  volume integral 2.3e-6 / 1.3e-3 / 8.6e-3. It tracks the 3D model's own L error (−1.2e-3 and −7.6e-3), and 2D sides
  with the independent 2D references (section 2), not with 3D.
- The exterior term itself agrees with 3D to 0.2 / 0.4 / 1.2 %, as the 3D term did with the 2D winding's exterior term.
- The 2 m and the 1 m coil grids (continued from 2 m and from 1 m) give 2D Ψ within 2.4e-12 / 1.9e-9 / 1.9e-8 of each
  other (3D: 2e-12 / 2e-9 / 2e-8).
- (with − without − exterior term)/|Ψ| ≤ 8e-17 for every row: the term is purely additive, and the volume part is
  unchanged.
- The grid whose inscribed sphere (0.8 m) cuts the long magnet (to 0.93 m) is flagged as in 3D (continuation mismatch
  0.23, the warning).

## 2. Against the 2D winding: one mesh, and a domain that encloses the long magnet

The winding model's exterior term is the static multipoles of A_lab outside R1 (exact, since the coil lies inside),
so it is an independent evaluation of the same integral.

| | kR = 0.02 | 0.4 | 0.8 |
|---|---:|---:|---:|
| coil: grid against winding, same mesh: Ψ without the term | 1.1e-4 | 4.3e-2 | 1.5e-1 |
| coil: Ψ with the term | 6.3e-8 | 6.4e-7 | 3.8e-6 |
| coil: exterior term, grid against static multipoles | 2.0e-5 | 2.0e-5 | 2.0e-5 |
| long magnet (grid, R1 = 0.45 m) against the magnet as the winding (R1 = 1.2 m, 76,784 unknowns): Ψ without the term | 3.5e-4 | 1.2e-1 | 3.9e-1 |
| long magnet: Ψ with the term | 1.8e-8 | 8.3e-6 | 2.3e-5 |
| long magnet: \|ΔL\|/L between the two meshes | 1e-8 | 5.0e-6 | 3.1e-5 |
| coil and superconducting disk (r ≤ 0.15 m above the coax): grid against winding, Ψ without the term | 1.1e-4 | 4.3e-2 | 1.6e-1 |
| coil and disk: Ψ with the term | 1.6e-7 | 7.3e-7 | 4.1e-6 |
| coil and disk: reaction exterior + grid exterior against the winding's exterior | 2.1e-5 | 2.2e-5 | 2.2e-5 |

- The 2e-5 of the exterior term is the grid's own accuracy: ψ is integrated from the axis through the winding, where
  B has kinks, so its error scales as h² with the spacing there (2 mm here; the default-suite test's coarse grid gives
  5e-4 at 1 cm and 1.2e-4 at 5 mm). The term is 4.3–15 % of Ψ, so this moves Ψ by ≤ 4e-6.
- The disk case is a reaction field next to the background: the disk's term (static multipoles of A_lab, now without
  the background; 0.34 % and 1.3 % of Ψ at kR = 0.4 and 0.8) and the grid's term (4.7 % and 17 %) add up to the winding
  model's single term.

## 3. Independence of the domain radius

|Ψ(R1) − Ψ(1.2 m)| / |Ψ| with the exterior term, and without it (Ψ − the background's term), at kR = 0.4 / 0.8;
the winding model (its own exterior term) is the control for the mesh's level:

| R1 [m] | coil grid | coil grid, without | long magnet | long magnet, without | coil winding (control) | L |
|---:|---:|---:|---:|---:|---:|---:|
| 0.36 | 2.4e-6 / 6.1e-6 | 5.2e-2 / 2.4e-1 | 5.7e-6 / 1.3e-5 | 1.3e-1 / 5.2e-1 | 2.1e-6 / 6.0e-6 | 1.4e-6 / 8.9e-6 |
| 0.45 | 2.3e-6 / 1.1e-5 | 4.0e-2 / 2.0e-1 | 5.1e-6 / 1.8e-5 | 1.1e-1 / 4.6e-1 | 2.5e-6 / 1.1e-5 | −2.9e-7 / −1.5e-6 |
| 0.54 | 1.8e-6 / 1.4e-5 | 3.1e-2 / 1.6e-1 | 1.4e-6 / 1.3e-5 | 9.1e-2 / 4.0e-1 | 1.9e-6 / 1.4e-5 | 2.7e-6 / 1.8e-5 |
| 0.675 | 9.5e-6 / 4.0e-5 | 2.2e-2 / 1.2e-1 | 1.7e-5 / 6.0e-5 | 6.6e-2 / 3.2e-1 | 9.5e-6 / 4.0e-5 | 1.6e-6 / 1.1e-5 |
| 0.9 | 8.6e-6 / 2.9e-5 | 1.1e-2 / 6.4e-2 | 2.0e-5 / 5.5e-5 | 3.2e-2 / 1.7e-1 | 8.6e-6 / 2.9e-5 | 3.2e-6 / 1.9e-5 |

At kR = 0.02 the spread is ≤ 6e-8 (grids) against 1e-5 to 3e-4 without the term (the control: ≤ 3.4e-7). With the
term, the grid designs are as independent of R1 as the winding model, whose exterior term is validated separately
(`tests/test_fullwave.py`): what is left is the mesh (L moves by the same amount). Without it the
dependence is 1–52 %. The long magnet's field crosses R1 for every R1 < 0.93 m; its exterior term is 45 % of Ψ at
R1 = 0.36 m and kR = 0.8.

## 4. The MQS limit

The exterior term over Ψ_MQS, the long magnet at R1 = 0.45 m (the coil in brackets):

| kR | term / Ψ_MQS | Re term / (Ψ_MQS (kR)²) | Im term / (Ψ_MQS (kR)³) | local order Re, Im |
|---:|---:|---:|---:|---|
| 0.0025 | 5.5e-6 (1.7e-6) | 0.87347 (0.27667) | 1.75819 (0.51931) | |
| 0.005 | 2.2e-5 | 0.87345 | 1.75821 | 1.99997, 3.00001 |
| 0.01 | 8.7e-5 | 0.87337 | 1.75826 | 1.99988, 3.00004 |
| 0.02 | 3.5e-4 (1.1e-4) | 0.87307 (0.27659) | 1.75846 (0.51944) | 1.99951, 3.00017 |
| 0.04 | 1.4e-3 | 0.87188 | 1.75929 | 1.99802, 3.00068 |
| 0.08 | 5.6e-3 | 0.86705 | 1.76258 | 1.99199, 3.00270 |
| 0.16 | 2.3e-2 | 0.84721 | 1.77570 | 1.96661, 3.01070 |
| 0.32 | 9.8e-2 (3.1e-2) | 0.75890 (0.25081) | 1.82699 (0.55180) | 1.84119, 3.04109 |

The term vanishes at the expected rate: its real part as (kR)² (H′ is O(ω²)) and its imaginary part as (kR)³ (the
moments' radiating part is O(k)), with the local orders → 2 and 3 (to 3e-5 at kR = 0.005). For the coil the
coefficients equal the winding model's (0.27668 and 0.51933 at kR = 0.0025) to 2e-5, the grid's accuracy.
Ψ(ω)/Ψ_MQS − 1 is the (kR)² dispersion of the coax (2.0e-4 at kR = 0.01).

## 5. Convergence in the multipole order

`fullwave.dtn_modes` (the order of the exterior map and of the term), long magnet at R1 = 0.45 m, |Ψ_N − Ψ_160|/|Ψ|:

| N | 4 | 6 | 8 | 12 | 16 | 24 | 40 | 80 (default) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kR = 0.4 | 1.6e-7 | 3.3e-6 | 2.7e-8 | 2.6e-10 | 1.0e-11 | 8.8e-13 | 3.2e-13 | 1.7e-14 |
| kR = 0.8 | 2.3e-5 | 1.2e-5 | 9.3e-8 | 1.1e-9 | 3.0e-11 | 2.7e-12 | 1.5e-12 | 2.7e-13 |

The exterior term alone converges the same way (7e-5 at N = 4, 4e-8 at 8, 1e-12 at 24). This readout's field needs
few orders (the pickup lies well inside R1); the default 80 costs nothing in accuracy.

`exterior_last_order_fraction` is the last two orders' share of the term. A magnet
symmetric about the domain centre has only odd orders, so the last order alone read 1e-19 to 1e-26 here at every even
N, converged or not. The last two orders read 2.2e-2 / 3.0e-2 at N = 4 (kR = 0.4 / 0.8), 2.6e-5 / 2.7e-5 at 8,
2.5e-7 / 2.3e-7 at 12, 8.8e-10 / 8.0e-10 at 16, 2.2e-12 / 2.9e-12 at 24 and 8.5e-14 / 8.9e-14 at 80 (round-off),
against the term's own changes of 7.8e-6 / 6.9e-5 at N = 4 and 3e-8 / 5e-8 at 8: no blind spot, and here larger
than the term's change for N = 4 to 24. It is a convergence indicator of the series, not a bound on the error of
truncating at N: `dtn_modes` also sets the exterior map's order, which changes the kept orders' b_n, and the fraction
does not see that. On the coil grid at N = 5 and 6 it reads 5.4e-7 to 1.2e-6 while the term moves by 2.6e-6 to 3.6e-6
against N = 160. A truncation is checked by a run at a larger `dtn_modes`.

## 6. The moments' quadrature

`AXF_FW_BG_PANELS` (radial panels; angular panels per radian about panels/π), long magnet, against 256 panels:

| panels | 16 | 32 | 64 | 128 (default) |
|---|---:|---:|---:|---:|
| exterior term, kR = 0.02 | 2.5e-7 | 2.9e-8 | 2.7e-9 | 2.1e-9 |
| exterior term, kR = 0.8 | 1.1e-6 | 7.6e-8 | 4.7e-9 | 1.9e-9 |
| built-in check (k → 0 moments, doubled panels, orders ≤ 12) | 6.4e-7 | 7.3e-8 | 4.9e-9 | 2.1e-9 |

The built-in check matches the error at the default; the slivers between the order-2 boundary and the sphere are
1e-9 of the moments (`sliver_correction_over_moments`). The check runs to order 12, not to N (at full order it reads
2.6e-5 / 5.8e-7 / 3.6e-8 / 2.1e-9 here; the full order costs four fifths of the moments at N = 80). For a grid
continued beyond its inscribed sphere, as here, the largest moments are the low orders'; for a grid cut at its box near
R1 they are not (in one such case n ≤ 12 held 0.84 of the maximum), so for such grids the check covers orders ≤ 12
only. Every arc gets at least panels/64 angular panels and every radial piece at least panels/32 (2 and 4 at the
default 128), so the doubled run refines short arcs and short radial pieces too.

**The cut-at-box path** (the inscribed sphere inside R1: two arcs per sphere, extra radial breaks), which no run above
reaches, is tested in `cpp/tests/test_background_exterior.cpp` on a smooth analytic field: G_n at k → 0 and k = 3.3 m⁻¹
against an independent (r, z) quadrature of the box minus the ball agree to 4.0e-11, 1.3e-9 and 9.0e-7 of max|G| for a
box about the centre, one beside it and an off-centre domain; the built-in check reads 4.1e-11, 3.0e-10 and 5.8e-7.
Which side limits each number (against a reference quadrature self-converged to 1e-14): the first is
the library's (about 4e-11); the second is the test's quadrature, which has no substitution at the sphere's pole inside
the box (the library is good to 2.9e-10 there, as its check says); the third is the library's, which the check reads at
0.64 of it. A quadrature without a break where the sphere crosses r = r_max reads 1.6e-5 to 5.6e-7 for the first box:
its own error. The same test finds the moments bit-identical with 1 and 4 OpenMP threads.

Where the cut-at-box moments are less accurate (every case below is cut at its box, which is warned about): Where the domain centre's equator lies inside the box, the sphere becomes tangent to
r = r_max at ρ = r_max, the arc opens like √(ρ − r_max), and Gauss-Legendre on uniform panels converges as h^1.5 there:
at the default the moments were good to 2e-9 – 9e-7 of max|G| in nine aggressive geometries, and the check read 0.35 to
0.65 of the error. With a fixed minimum of 4 radial panels, the coil grid cut to |z| ≤ 0.3 m would keep that minimum at
both panel counts and the check would read 3.8e-9 against an error of 2.7e-8 (1.0e-7 at k); the radial minimum of
panels/32 (above) removes that blind spot. All of this is far below the check's warning threshold (1e-4).

## 7. What the term does not change (part I)

Each design was run with and without the term (one OpenMP thread) and every leaf of results.json compared, timings
and file paths aside:

| design | numbers compared | numbers differing | strings differing | keys added / removed |
|---|---:|---:|---:|---|
| coil winding, TM sweep, radiating | 3566 | 0 | 0 | none |
| coil winding and superconducting disk | 3926 | 0 | 0 | none |
| toroid, TE sweep | 5397 | 0 | 0 | none |
| toroid, TE sweep, uniform background | 5423 | 0 | 0 | none |
| uniform background (1 T), radiating (A_lab = 0) | 1763 | 0 | 2: warning, label | 18 added (`Psi_exterior` = 0) |
| uniform background and disk | 2142 | 15, by design: Ψ, F (below) | 2 | 18 added |
| uniform background, closed sphere | 1757 | 0 | 0 | 12 added |
| coil grid, closed sphere | 1757 | 0 | 0 | 14 added (with the grid diagnostics) |
| eigenmodes only, coil winding | 3493 | 0 | 0 | `exterior_term` removed |
| eigenmodes only, coil grid | 1696 | 0 | 0 | `exterior_term` and one warning removed |

The uniform background's own exterior term is omitted, with a warning and `Psi_background.exterior` null (no term
computed). A_lab's term is included: the uniform design with the disk changes by exactly that term,
(Ψ_with − Ψ_without − `Psi_exterior`)/|Ψ| = 4.2e-17,
2.8e-17 and 8.7e-17 at kR = 0.02 / 0.4 / 0.8, the term being 1.8e-5, 0.66 % and 1.8 % of Ψ; L is unchanged. With the
copper coax alone A_lab is zero and so is the term: Ψ is bit-identical. Eigenmode-only runs have no Ψ(ω), so no sweep
labels.

## 8. The 3D sweep (part F)

The 3D sweep computes its moments with the same code (`axi::background_exterior`, parallel over the spheres, the
recurrence over an arc's points at once), so the two engines' exterior terms agree to the mesh (section 1).

## 9. A grid that does not reach past the domain sphere (part C)

The coil's grid cut to r ≤ 0.4 m, |z| ≤ 0.4 m (corners at ρ = 0.57 m):

| kR | Ψ, cut grid, R1 = 0.6 m | Ψ, full grid | \|Δ\|/\|Ψ_full\| | full grid vs the coil as winding |
|---:|---|---|---:|---:|
| 0.02 | 3.038669e-9 + 3.6e-18 i | 3.039010e-9 + 1.26e-14 i | 1.1e-4 | 1.7e-7 |
| 0.4 | 3.602657e-9 + 1.40e-11 i | 3.718854e-9 + 1.25e-10 i | 4.3e-2 | 6.1e-7 |
| 0.8 | 7.055722e-9 + 8.36e-10 i | 6.919972e-9 + 1.96e-9 i | 1.6e-1 | 3.9e-6 |

- At R1 = 0.6 m the box lies inside the sphere: the label is "none (the grid ends inside the domain sphere, where its
  field is cut off: grid_edge_A_over_max)", 0.56 of the domain (by volume) lies outside the grid, A_bg on the grid's
  edge is 0.34 of its maximum, and both warnings fire (coverage; the grid ends inside the sphere).
- At R1 = 0.45 m (inscribed sphere 0.4 m inside R1, box beyond it) the label is "included (grid only: cut at its
  box)", with the coverage warning (0.12 of the domain outside) and the inscribed-sphere warning.
- The MQS Ψ is the same with the cut and the full grid: the TM readout's MQS field lives inside the device, so only the
  sweep can see the cut.

## 10. Uniform backgrounds (part U)

A uniform 1 T background with a superconducting disk above the coax, with superconducting coax walls, with the coil as
the design's winding (R1 = 0.45 and 0.675 m, kR = 0.02 / 0.4 / 0.8), and a CORE-like cavity (`core` template,
Rc = 0.2 m, H = 0.4 m, its coils as μr = 1 regions; R1 = 0.8 and 1.2 m, kR = 0.02 / 0.2 / 0.4 with R = 0.502 m).
A_lab's exterior term, the design's windings and superconductor reactions, is included, as in 3D; the background's
own term (divergent) is not.

| design | A_lab's term / Ψ | Ψ(1.5 R1) vs Ψ(R1), with it | without it |
|---|---|---|---|
| disk | 1.8e-5 / 0.66 % / 1.8 % | 5.5e-4 / 28 % / 116 % | 5.4e-4 / 27 % / 115 % |
| superconducting coax walls | 9.3e-5 / 3.2 % / 7.6 % | 8.9e-4 / 44 % / 159 % | 8.6e-4 / 42 % / 152 % |
| coil as the winding | 7.7e-5 / 2.9 % / 9.7 % | 1.4e-4 / 7.5 % / 38 % | 1.7e-4 / 9.0 % / 45 % |
| CORE-like cavity | 3.7e-6 / 0.037 % / 0.15 % | 1.5e-5 / 0.16 % / 0.78 % | 1.4e-5 / 0.15 % / 0.71 % |

- **The term is A_lab's and nothing else.** The same designs with the uniform field given as a 1 T grid (ψ = r²/2
  exactly) have the same domain integral (Ψ minus the grid's background term equals the uniform run's Ψ to ≤ 7e-14)
  and the same `Psi_exterior` (to ≤ 3e-14); against a run without it, Ψ changes by exactly the term
  ((Ψ_with − Ψ_without − `Psi_exterior`)/|Ψ| ≤ 1.1e-16 in every row).
- **What remains.** The uniform field's own exterior term diverges and is the R1 dependence: between R1 and 1.5 R1 the
  validation's designs move by 0.7–44 % at kR = 0.4 (the CORE-like cavity least, the superconducting walls most) and
  by up to 159 % at kR = 0.8, with or without A_lab's term. The warning says so and suggests varying the
  domain radius or giving the field as a grid; above kR ≈ 0.3 such a Ψ(ω) is not a result for the coax designs.

## 11. A uniform background in 3D (part V)

The disk design in a uniform 1 T in 3D, on part F's coarse mesh (762,290 unknowns, 4 MPI ranks), whose
`Psi_exterior` carries the same A_lab field in its own split (the primary form): |Ψ_2D − Ψ_3D|/|Ψ_3D|.

| kR | 2D without A_lab's term | 2D with it | L_3D/L_2D − 1 |
|---:|---:|---:|---:|
| 0.02 | 1.6e-3 | 1.6e-3 | −8e-5 |
| 0.4 | 5.8e-3 | 1.2e-3 | 1.8e-3 |
| 0.8 | 2.1e-2 | 1.5e-2 | 1.2e-2 |

At kR = 0.4 the term takes the engines from three times the mesh level to it. At 0.02 the residual is the coarse
mesh's reaction field (the MQS Ψ differs by 1.7e-3: 1.14626e-9 in 3D, 1.14820e-9 in 2D); at 0.8 its L error (1.2 %)
dominates. An independent run gave the same numbers.

## Limits

- One 2D mesh per radius; its L converges to ~1e-5 here. The comparison with 3D is limited by the 3D meshes (L off by
  up to 0.77 % at kR = 0.8).
- The term is as good as the grid's continuation: a grid whose inscribed sphere does not enclose the magnet is
  warned about (continuation mismatch), and its term is then approximate.
- With a uniform background both engines omit the background's own exterior term, so Ψ(ω) depends on the domain
  radius above the MQS limit (section 10); both include the static lab field's.
  `Psi_exterior` itself is a different part of Ψ in the two engines (reciprocal form in 2D, primary form in 3D):
  compare Ψ and `Psi_background` only.
- Cost: the moments at order 80 take about 1 s of one core per 2D solve with a grid (full-wave part 1.24 s against
  0.20 s for the winding model here with the check at full order; the check to order 12 costs a fraction of the
  moments); they run in parallel over the spheres.
