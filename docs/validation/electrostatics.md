# Electrostatics validation (axisymmetric)

Numbers from `benchmarks/electrostatics_validation.py`; records `benchmarks/records/electrostatics_*.json`;
tests `tests/test_electrostatics.py`. Formulation and decisions:
[physics/electrostatics.md](../physics/electrostatics.md). The exterior-monopole check is an algebraic identity and
is labelled as one; `exterior_flux_*` is the genuine exterior check. The port-region capacitance is not a proven
lower bound (the rigorous Dirichlet inequality is the relation that holds), and the lumped self-resonance is not an
upper bound.

All cases are axisymmetric, FE order 3 unless stated, open exterior boundary
(exact Laplace Dirichlet-to-Neumann map on the sphere of radius
`mesh.domain_radius`) with ε₀ outside. The capacitance matrix is
C_ij = Q_i for V_j = 1 V, with the charges taken from the reactions
Q = 2π Σ (Kφ) on the tied degrees of freedom.

## Closed-form references

| case | quantity | computed [F] | exact [F] | rel. error |
|---|---|---|---|---|
| isolated sphere, a = 0.5 m | C = 4πε₀a | 5.563250275e-11 | 5.563250277e-11 | −3.4e-10 |
| concentric spheres a = 0.3, shell 0.5–0.6 m | C_inner = 4πε₀/(1/a−1/b) | 8.34487539e-11 | 8.34487542e-11 | −2.5e-09 |
| | C_inner,outer | −8.34487539e-11 | −8.34487542e-11 | −2.5e-09 |
| | C_outer = C_inner + 4πε₀c | 1.50207757e-10 | 1.50207757e-10 | −1.6e-09 |
| sphere a = 0.3 m + dielectric shell to b = 0.5 m, ε_r = 4 | C = 4πε₀/[(1/a−1/b)/ε_r + 1/b] | 4.768499942e-11 | 4.768500238e-11 | −6.2e-08 |
| zero-thickness conductor sheet (`curves` arc), a = 0.4 m | C = 4πε₀a | 4.450600218e-11 | 4.450600222e-11 | −8.7e-10 |
| sphere a = 0.3 m inside a grounded sphere R_b = 1 m | C = 4πε₀/(1/a−1/R_b) | 4.768500234e-11 | 4.768500238e-11 | −8.6e-10 |
| two spheres a = 0.1 m, centres 0.5 m apart | C_11 (bispherical series) | 1.16112174e-11 | 1.16112177e-11 | −2.4e-08 |
| | C_12 (bispherical series) | −2.32645872e-12 | −2.32645883e-12 | −4.7e-08 |

The two equal spheres come out with C_11/C_22 − 1 = 6.7e-10, an independent
check of the body detection and of the matrix assembly.

The two-sphere reference is the exact bispherical series (cosh η = D/2a):
C_11 = 4πε₀a sinh η Σ_{n≥0} 1/sinh((2n+1)η),
C_12 = −4πε₀a sinh η Σ_{n≥1} 1/sinh(2nη). The leading image term alone
(−4πε₀a²/D = −2.2253e-12 F) is 4 % off, so this case tests the off-diagonal
coefficient, not just the geometry.

Ring (torus of major radius R = 0.5 m, minor radius b), against the leading
thin-ring formula C = 4π²ε₀R/ln(8R/b) — the deviation is the expected O(b/R)
correction and shrinks with b/R:

| b/R | C [F] | thin-ring [F] | ratio |
|---|---|---|---|
| 0.040 | 3.3033e-11 | 3.2987e-11 | 1.0014 |
| 0.020 | 2.9183e-11 | 2.9171e-11 | 1.0004 |
| 0.010 | 2.6149e-11 | 2.6146e-11 | 1.0001 |

## Consistency checks (every case)

The results group the checks by what they actually test (the string
`electrostatics.checks.what_each_check_tests` says the same thing in the JSON).

**Algebraic identities of the discrete system.** They verify the assembly and
the linear solves. They do **not** probe the discretization and must not be read
as if they did.

| identity | sphere | concentric | shell | grounded | two spheres |
|---|---|---|---|---|---|
| matrix symmetry (relative) | 0 | 2.6e-16 | 0 | 0 | 4.2e-16 |
| ½VᵀCV vs ∫½ε\|E\|² by quadrature | 3.1e-14 | 1.6e-14 | 4.4e-14 | 1.6e-14 | 8.1e-15 |
| same at a mixed potential vector | 3.1e-14 | 7.9e-14 | 4.4e-14 | 1.6e-14 | 3.0e-14 |
| net charge vs the exterior monopole | 8.1e-10 | 3.6e-10 | 8.0e-10 | 0 | 2.6e-10 |
| floating-conductor solve vs the matrix | n/a | 9.9e-14 V | n/a | n/a | 2.1e-14 V |

The exterior-monopole row is an identity, not a check: on the outer sphere
1ᵀU(:, n) = ∫P_n(cos θ) r dl/R_b = 2R_b δ_{n0} and the stiffness annihilates a
constant, so 2π·1ᵀKφ = 2πε_out p_0 = 4πε_out R_b a_0 = Q_exterior exactly. What
is left is the solver residual on the free rows plus the geometric error of the
boundary arc. The h/p table below shows it saturating at **1.0618e-9 for all six
p ≥ 2 rows** while the capacitance error falls by a factor 715.

**Discretization checks.** These compare the solution with an independently
evaluated *strong* form and converge under h/p refinement; they are the ones to
read.

| discretization check | sphere | concentric | shell | grounded | two spheres |
|---|---|---|---|---|---|
| reaction charge vs ∮εE·dA (Gauss) | 7.5e-07 | 3.3e-06 | 3.2e-06 | 3.2e-06 | 4.1e-05 |
| net charge vs −∮ε∂φ/∂n dA on the outer sphere | 1.4e-06 | 5.7e-07 | 1.3e-06 | 3.3e-06 | 5.6e-07 |

The exterior-flux row integrates the strong normal flux over the outer
semicircle from the interior element gradients. With the DtN map it must match
the net charge that the map carries away, and the map imposes that Neumann data
only weakly, so the difference is a true discretization error. With a **grounded**
sphere there is no exterior map and the monopole identity does not exist at all
(it is exactly 0 in the table above), but Gauss's law still says the flux through
the sphere is the charge enclosed, so the row is a genuine check there too — and
it lands at 3.3e-6, right beside the Gauss check's 3.2e-6.

Also reported: diagonal dominance, min relative slack — 1.00 (sphere), −3.2e-14
(concentric), 1.00 (shell), 1.00 (grounded), 0.80 (two spheres). The zero slack
of the concentric case is the expected equality for a fully screened conductor
(C_ii = Σ_{j≠i}|C_ij| when conductor i sees no ground).

## h/p convergence

Concentric spheres (a = 0.3 m, shell 0.5–0.6 m), inner-conductor capacitance,
`mesh.size = 4 h`, `mesh.surface_size = h`:

| order | h [m] | DOFs | C_inner [F] | rel. error | Gauss check | exterior flux | monopole *identity* |
|---|---|---|---|---|---|---|---|
| 1 | 0.08 | 1858 | 8.3834e-11 | +4.61e-03 | 1.8e-01 | 1.42e-02 | 5.94e-05 |
| 1 | 0.04 | 2133 | 8.3705e-11 | +3.07e-03 | 1.3e-01 | 1.48e-02 | 5.94e-05 |
| 1 | 0.02 | 3259 | 8.3568e-11 | +1.42e-03 | 6.4e-02 | 1.48e-02 | 5.95e-05 |
| 2 | 0.08 | 7267 | 8.3451e-11 | +3.06e-05 | 9.2e-03 | 1.34e-04 | **1.0619e-09** |
| 2 | 0.04 | 8356 | 8.3450e-11 | +1.18e-05 | 5.1e-03 | 1.53e-04 | **1.0619e-09** |
| 2 | 0.02 | 12831 | 8.3449e-11 | +1.68e-06 | 1.3e-03 | 1.54e-04 | **1.0619e-09** |
| 3 | 0.08 | 16228 | 8.3448e-11 | −3.31e-06 | 4.4e-04 | 1.19e-06 | **1.0618e-09** |
| 3 | 0.04 | 18670 | 8.3449e-11 | −6.83e-07 | 1.9e-04 | 1.48e-06 | **1.0618e-09** |
| 3 | 0.02 | 28717 | 8.3449e-11 | −4.28e-08 | 2.3e-05 | 1.50e-06 | **1.0618e-09** |

p-refinement is far more effective than h-refinement, as expected for a smooth
problem with curved (order-2) geometry. Order 2 already gives 10⁻⁵; order 1 is
limited by the linear representation of the potential near the curved surfaces.

The last two columns show the difference between an identity and a check. The monopole column is the
**same number, 1.0618e-9, in all six p ≥ 2 rows** while the capacitance error
falls from 3.1e-5 to 4.3e-8 (a factor 715): it is an identity that has hit the
floor set by the solver tolerance and the boundary geometry, so it cannot
diagnose the discretization. The exterior-flux column does: 1.48e-2 → 1.54e-4 →
1.50e-6 across p at the finest h, tracking the Gauss check. (Within one p the
flux value is flat in h: it is limited by the order of the potential's trace on
the boundary, so it is a p-convergence probe, as the Gauss check is at order 1.)
The record also carries `identity_does_not_track_the_error` and
`exterior_flux_converges`, which the check's pass criterion now includes.

## Readout gap and self-resonance

For a poloidal (port) pickup the readout gap is solved as a two-terminal
problem (`pickup.port_capacitance`): `C_gap_port_region_F` uses the port region
alone (insulating end faces, no fringing), `C_gap_F` drives the two
terminal faces at V = ±½ on the whole domain with the rest of their own
conductor body insulating and every other conductor floating (fringing at the
two mouths included). The reference is the full-wave readout impedance: the
first self-resonance is the pole of L(ω) (the frequency where L changes sign),
bracketed by a 12–16 point sweep.

### What is rigorous between the two gap values

`C_gap_F ≥ C_gap_port_region_F` is **not** proven (it was claimed to be, on the
grounds that restricting a variational problem to a subdomain lowers the energy;
that argument does not apply, because the full-domain solution also stores energy
in the terminals' capacitance to ground). Driving the terminals at V = ±½, the
full-domain field energy is ½C_gap + ⅛(C_Ag + C_Bg) and the region-only minimum
is ½C_region; the restriction of the full-domain solution to the port subdomain
is admissible for the region-only problem, so

    C_gap_port_region ≤ C_gap + ¼(C_A,ground + C_B,ground),

exactly, for the discrete problems. This is emitted as
`pickup.port_capacitance.checks.dirichlet_bound` and is the check the validation
now asserts. It is loose, as a shorted coax (a = 0.05, b = 0.15 m, t = 0.02 m,
L = 0.3 m, order 2) shows:

| gap g [m] | C_region [F] | coax 2πε₀g/ln(b/a) | C_gap [F] | C_gap/C_region | bound/C_region |
|---|---|---|---|---|---|
| 0.005 | 2.5319e-13 | 2.5319e-13 | 1.4101e-12 | 5.57 | 10.94 |
| 0.020 | 1.0128e-12 | 1.0128e-12 | 2.3868e-12 | 2.36 | 3.79 |
| 0.050 | 2.5319e-12 | 2.5319e-12 | 3.9804e-12 | 1.57 | 2.16 |
| 0.100 | 5.0639e-12 | 5.0639e-12 | 6.5219e-12 | 1.29 | 1.58 |
| 0.200 | 1.0128e-11 | 1.0128e-11 | 1.1578e-11 | 1.14 | 1.29 |
| 0.240 (a = 0.14) | 1.9352e-10 | 1.9352e-10 | 1.9770e-10 | 1.02 | 1.03 |

C_region reproduces the coaxial-capacitor formula to 4e-7 or better in every row,
so it is the gap's own capacitance to the accuracy of the solve. C_gap/C_region
falls towards 1 as the gap grows relative to the electrode spacing, and the
guaranteed bound is only useful in the same limit: at the 5 mm gap its
right-hand side is twice C_gap. These rows come from solves on the design's default mesh and are not stored in a
record. C_region is mesh independent to the digits shown; C_gap is not: its fringing part moves by a few per cent with
the surface mesh (the electrode edges, as for the core template below), so read C_gap to two digits.

| template | L_p [H] | C_gap [F] | gap region only [F] | lumped f = 1/(2π√(L_pC_gap)) | full-wave pole | ratio |
|---|---|---|---|---|---|---|
| solenoid (shorted coax, H = 2 m) | 2.4548e-07 | 9.7911e-12 | 9.0121e-13 | 102.7 MHz | 27.2–33.7 MHz | ×3.39 |
| core (toroidal cavity, neck gap) | 3.3556e-07 | 5.1093e-11 | 2.3922e-11 | 38.4 MHz | 13.7–17.6 MHz | ×2.47 |

The full-wave poles come from sweeps of the same designs (not stored in a record). The capacitance that reproduces
the full-wave pole with the MQS L_p is
1.13e-10 F (solenoid) and 3.12e-10 F (core), i.e. the gap capacitance is 9 %
and 16 % of it. **The lumped estimate is an estimate, not a bound in either
direction.** Two effects compete: C_gap is only the gap's share of the
structure's capacitance, which pushes the estimate up, while a uniform line
resonates at (π/2)/(2π√(L C_total)), which pushes the true pole up. The estimate
exceeds the first pole only while C_gap/C_total < (2/π)² = 0.405; here
C_gap/C_eff is 0.09 and 0.16, well below that, which is why both templates come
out high. A compact loop whose gap capacitor dominates (C_gap/C_total → 1) would
show the opposite sign, the lumped value falling a factor 2/π below the pole.
The gap capacitance is the right quantity for a gap that is small compared with
the structure (a tuning capacitor across a compact loop); for a long coax or a
large cavity use the full-wave sweep.

Mesh dependence of the gap value (core template, `mesh.surface_size` scaled):

| surface size | DOFs | C_gap [F] | gap region only [F] | f_SR | charge beyond the terminal faces |
|---|---|---|---|---|---|
| ×1.0 | 136538 | 5.1093e-11 | 2.3922e-11 | 38.44 MHz | 28 % |
| ×0.7 | 189082 | 5.0262e-11 | 2.3922e-11 | 38.75 MHz | 26 % |
| ×0.5 | 255406 | 4.9698e-11 | 2.3922e-11 | 38.97 MHz | 20 % |

The region-only value is mesh independent to 2e-14; the fringing value drifts
by 2.8 % over a 2× refinement because the electrode edges are
Dirichlet–Neumann junctions (the electrode is defined by its degrees of
freedom, and the share of the charge collected beyond the terminal faces
proper shrinks as h → 0). Quote `C_gap_F` to two digits.

## Input handling

| input | behaviour |
|---|---|
| a region named `air` (or `infinity_shell`) | rejected: "'air' is a reserved region name" (otherwise it would silently be vacuum: a dielectric shell named `air` around a sphere would give the bare-sphere value, −30 %; under any other name the same shell gives 4.76850e-11 F, −6.2e-08 from exact) |
| `electrostatics.boundary: "grounded"` with a port pickup | honoured by the readout-gap solve: C_gap = 3.98e-12 F open vs 3.93e-12 F grounded on the a = 0.05, b = 0.15, g = 0.05 m coax (−1.4 %, same mesh), with the terminal-to-ground capacitances larger |
| two conductor regions meeting at a corner | one body (a shared node is a whole circle of contact, and the H1 space forces one potential there): C = 4.62458e-11 F, both region names listed under the single body |
| the floating reduction's factorization | checked (`info()` plus a residual test on the solve) |

## Not covered

- 3D electrostatics: see [electrostatics3d.md](electrostatics3d.md).
- Floating conductors with a prescribed nonzero charge are implemented in the
  solver but not exercised by a closed-form case.
- Dielectrics touching the outer boundary: the exterior map assumes a uniform
  permittivity there and falls back to ε₀ with a warning.
