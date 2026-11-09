# 3D electrostatics validation

Numbers from `benchmarks/electrostatics3d_validation.py` (an MPI build); records
`benchmarks/records/electrostatics3d_*.json`; tests `tests/test_electrostatics3d.py`. Formulation, decisions and
limitations: [physics/electrostatics.md](../physics/electrostatics.md#3d); the axisymmetric counterpart is
[electrostatics.md](electrostatics.md).

All cases are `"symmetry": "3d"` designs, H1 order 2 unless stated, open
exterior boundary: the infinity shell [R_b, 1.25 R_b) that the 3D mesher always
builds, mapped onto [R_b, ∞) for the scalar potential by

    ε_eff = ε₀ [ (f²/(f′ρ²)) rrᵀ + f′ (I − rrᵀ) ],   f(ρ) = R₁(R₂−R₁)/(R₂−ρ),

with R₁ = R_b, R₂ = 1.25 R_b and φ = 0 on the outer sphere of the shell. The
capacitance matrix is C_ij = Q_i for V_j = 1 V, the charges from the reactions
Q_i = u_i·(Kφ) with u_i the indicator of conductor i.

## Closed-form references

| case | quantity | computed [F] | exact [F] | rel. error | H1 DOFs |
|---|---|---|---|---|---|
| isolated sphere, a = 0.5 m | C = 4πε₀a | 5.56376453e-11 | 5.56325028e-11 | +9.2e-05 | 114 641 |
| concentric spheres a = 0.3, shell 0.5–0.6 m | C_inner = 4πε₀/(1/a−1/b) | 8.34534657e-11 | 8.34487542e-11 | +5.7e-05 | 236 081 |
| | C_inner,outer | −8.34534657e-11 | −8.34487542e-11 | +5.7e-05 | |
| | C_outer = C_inner + 4πε₀c | 1.50215264e-10 | 1.50207757e-10 | +5.0e-05 | |
| sphere a = 0.3 m + dielectric shell to b = 0.5 m, ε_r = 4 | C = 4πε₀/[(1/a−1/b)/ε_r + 1/b] | 4.76872475e-11 | 4.76850024e-11 | +4.7e-05 | 189 461 |
| sphere a = 0.3 m inside a grounded sphere R_b = 1 m | C = 4πε₀/(1/a−1/R_b) | 4.76902502e-11 | 4.76850024e-11 | +1.1e-04 | 56 337 |
| two spheres a = 0.1 m on the z axis, centres 0.5 m apart | C_11 (bispherical series) | 1.16141750e-11 | 1.16112177e-11 | +2.6e-04 | 157 527 |
| | C_12 (bispherical series) | −2.32771982e-12 | −2.32645883e-12 | +5.4e-04 | |

The two equal spheres come out with C_11/C_22 − 1 = −7.3e-06, an independent
check of the body detection and the assembly. The reference is the exact
bispherical series (cosh η = D/2a):
C_11 = 4πε₀a sinh η Σ_{n≥0} 1/sinh((2n+1)η),
C_12 = −4πε₀a sinh η Σ_{n≥1} 1/sinh(2nη) — the same series as in
`docs/validation/electrostatics.md`, reproduced to all printed digits
(1.16112177e-11, −2.32645883e-12). The leading image term alone
(−4πε₀a²/D = −2.2253e-12 F) is 4 % off, so C_12 tests the off-diagonal
coefficient, not just the geometry.

## The exterior map and the monopole

A capacitance to infinity *is* the n = 0 multipole of the exterior field, which
the axisymmetric solver takes from the n = 0 term of its Dirichlet-to-Neumann
series. In 3D it comes from the shell: a net charge gives φ = Q/(4πε₀f), and
1/f = (R₂ − ρ)/(R₁(R₂ − R₁)) is linear in the radial coordinate. This is a
continuum-exact map, but ρ is not affine on general Cartesian tetrahedra: neither
the monopole nor higher multipoles are represented exactly at finite mesh order.
Convergence requires h/p refinement and checks at different domain radii.

The same sphere (a = 0.5 m, `surface_size`
0.05 m, `size` 0.15 m) at three domain radii, next to a grounded boundary at the
same radii:

| R_b [m] | C (open) [F] | rel. error vs 4πε₀a | C (grounded) [F] | exact 4πε₀/(1/a−1/R_b) [F] | rel. error | H1 DOFs |
|---|---|---|---|---|---|---|
| 1.0 | 5.56434172e-11 | +1.96e-04 | 1.11270349e-10 | 1.11265006e-10 | +4.8e-05 | 64 052 |
| 1.5 | 5.56379633e-11 | +9.82e-05 | 8.34533197e-11 | 8.34487542e-11 | +5.5e-05 | 93 091 |
| 2.5 | 5.56350384e-11 | +4.56e-05 | 6.95439006e-11 | 6.95406285e-11 | +4.7e-05 | 263 221 |

The open values spread by **1.5e-04** of 4πε₀a over a 2.5× range of domain
radius; the grounded values spread by **0.75**, i.e. 5000× more. The open
boundary is therefore doing its job, monopole included, and the residual spread
is ordinary discretization error (it shrinks with R_b because the shell and the
far field get relatively better resolved).

## Consistency checks

| check | sphere | concentric | dielectric | grounded | two spheres | revolved can+ring |
|---|---|---|---|---|---|---|
| matrix symmetry (relative) | 0 | 8.6e-16 | 0 | 0 | 4.9e-16 | 6.1e-16 |
| ½VᵀCV vs ∫½ε\|E\|² by quadrature | 4.9e-06 | 8.9e-07 | 1.5e-06 | 2.3e-14 | 5.8e-06 | 8.0e-07 |
| same at a mixed potential vector (superposition) | 4.9e-06 | 1.9e-06 | — | — | — | 1.4e-06 |
| independent solve at a mixed potential vector | 8.4e-15 | 1.9e-14 | — | — | — | 1.8e-14 |
| reaction charge vs ∮εE·dA (Gauss) | 4.8e-03 | 1.1e-02 | 1.2e-02 | 1.2e-02 | 1.7e-02 | 9.9e-02 |
| net charge vs the flux at `mesh.domain_radius` | 7.6e-03 | 1.8e-03 | 4.0e-03 | n/a | 3.3e-03 | 1.6e-03 |
| floating-state charges by the Gauss path | n/a | 2.1e-03 | n/a | n/a | 1.3e-04 | 5.2e-17 |
| relative residual on the free DOFs | 5.8e-14 | 5.7e-14 | 2.6e-14 | 2.9e-15 | 7.8e-14 | 5.5e-14 |

The symmetry, energy, superposition and mixed-potential-solve checks are
**algebraic** — they verify the solves and the assembly, and they are at
round-off because each unit-potential column is driven to a free-DOF residual of
~1e-14 by the iterative refinement described in the implementation note.
`charge_gauss_*` and the exterior-flux check are **discretization** checks: they
compare the reaction charge with a surface integral of the normal derivative
evaluated on one element face, which converges more slowly than the capacitance
itself (table below), so they are percent-level on 3D meshes. Read them as mesh
diagnostics, not as implementation checks.

## h/p convergence

Concentric spheres (a = 0.3 m, shell 0.5–0.6 m), inner-conductor capacitance,
`mesh.surface_size = h`, `mesh.size = 2.5 h`, `domain_radius` 1.6 m:

| order | h [m] | H1 DOFs | C_inner [F] | rel. error | Gauss check | flux check | energy check |
|---|---|---|---|---|---|---|---|
| 1 | 0.09 | 6 547 | 8.56821632e-11 | +2.68e-02 | 3.2e-01 | 1.0e-01 | 2.1e-02 |
| 1 | 0.06 | 14 286 | 8.50808864e-11 | +1.96e-02 | 2.5e-01 | 7.0e-02 | 2.1e-02 |
| 1 | 0.04 | 40 229 | 8.42456701e-11 | +9.55e-03 | 1.7e-01 | 4.3e-02 | 1.6e-02 |
| 2 | 0.09 | 49 541 | 8.35043997e-11 | +6.67e-04 | 4.0e-02 | 6.6e-03 | 5.4e-06 |
| 2 | 0.06 | 110 270 | 8.34680846e-11 | +2.32e-04 | 2.5e-02 | 3.0e-03 | 2.0e-06 |
| 2 | 0.04 | 315 172 | 8.34531946e-11 | +5.32e-05 | 1.2e-02 | 1.2e-03 | 4.9e-07 |

Order 2 is 40–180× more accurate than order 1 at a comparable mesh, as expected
for a smooth problem with second-order curved geometry; at order 1 the dominant
error is the linear representation of the potential next to the curved surfaces.
The capacitance converges close to h² at order 2 (ratio 2.9 and 4.4 for h ratios
1.5); the two surface-flux checks converge like h^1.3–h^1.6, which is why they
stay percent-level.

## Non-axisymmetric cases

**Unit cube.** A body with edges and corners, and the published reference
C = 0.66067813 · 4πε₀L (the independent determinations in the literature — Read
1997, Mansfield/Douglas/Garboczi 2001, Hwang & Mascagni 2004 — agree to about
1e-6, far below the discretization error here). Side L = 0.4 m,
`mesh.size = 4 h`, `domain_radius` 1.2 m:

| h = surface_size [m] | H1 DOFs | C [F] | C/(4πε₀L) | rel. error | Gauss check |
|---|---|---|---|---|---|
| 0.040 | 49 616 | 2.95459731e-11 | 0.6638649 | +4.82e-03 | 1.0e-01 |
| 0.028 | 103 434 | 2.94868860e-11 | 0.6625373 | +2.81e-03 | 7.9e-02 |
| 0.020 | 237 769 | 2.94594602e-11 | 0.6619211 | +1.88e-03 | 6.5e-02 |

The sequence falls monotonically towards the published value with an observed
rate of about h^2.2; the remaining +0.19 % at h = 0.02 is the edge/corner
singularity of the cube, where the surface charge density diverges and the
discrete solution is one-sided. This is the expected behaviour of a nodal
finite-element capacitance for a non-smooth conductor — quote two digits for
sharp bodies, or refine the edges.

**Two spheres on no coordinate axis.** The bispherical series depends only on
the radius a and the centre distance D, so the same exact numbers apply to a pair
rotated onto (1,1,1)/√3 with the centre of the pair displaced to
(0.13, −0.09, 0.07) m — a configuration with no symmetry axis through the domain
centre, and therefore a test that nothing in the 3D solver (or in the shell map,
which is centred on the origin) assumes axisymmetry:

| configuration | C_11 [F] | rel. error | C_12 [F] | rel. error | C_11/C_22 − 1 |
|---|---|---|---|---|---|
| on the z axis, centred | 1.16141750e-11 | +2.55e-04 | −2.32771982e-12 | +5.42e-04 | −7.3e-06 |
| along (1,1,1)/√3, off the origin | 1.16141523e-11 | +2.53e-04 | −2.32767206e-12 | +5.21e-04 | −2.2e-05 |

The two configurations agree with each other to 2e-06 in C_11 and 2e-05 in C_12,
i.e. far better than either agrees with the exact answer at this mesh.

## 3D against the axisymmetric engine

A revolved axisymmetric design (`axionfem.convert.to_3d`): a superconducting
pickup ring (r = 0.25–0.30 m, |z| ≤ 0.03 m) inside a closed superconducting can
(0–0.5 m × ±0.5 m, 0.05 m wall) on an ε_r = 3 dielectric support. 2D: order 3,
`surface_size` 0.012, 36 841 DOFs. 3D: order 2, `surface_size` 0.04,
271 762 DOFs.

| entry | axisymmetric [F] | 3D [F] | relative | of C(can, can) |
|---|---|---|---|---|
| C(can, can) | 1.25937305e-10 | 1.26364350e-10 | +3.39e-03 | +3.39e-03 |
| C(can, ring) | −5.96486573e-11 | −6.00213829e-11 | +6.25e-03 | −2.96e-03 |
| C(ring, ring) | 5.96486573e-11 | 6.00213829e-11 | +6.25e-03 | +2.96e-03 |

The largest difference is 3.4e-03 of the biggest matrix entry, and all 3D entries
are high by the same ~0.3–0.6 %, consistent with the h² error of the coarser 3D
mesh (the same one-sided bias as the sphere and the concentric case). The ring is
fully screened in both engines: its capacitance to ground is 5.2e-26 F in 3D
(C_ii = Σ_{j≠i}|C_ij| to 8.6e-16 relative), which is the expected equality for a
conductor that sees no ground.

## Rank independence

Concentric spheres (a = 0.3, shell 0.5–0.6 m, `surface_size` 0.04, 2 conductors),
solved on 1, 4 and 8 MPI ranks (`Design.solve(..., mpi_ranks=n)`, which runs
`axf solve --save` under srun and reloads the saved solution):

| ranks | largest absolute difference from 1 rank [F] | relative to the largest entry |
|---|---|---|
| 4 | 1.73e-24 | 1.2e-14 |
| 8 | 6.72e-25 | 4.5e-15 |

C(1 rank) = [[8.345346569226429e-11, −8.345346569226423e-11],
[−8.345346569226423e-11, 1.5021526386976652e-10]]. The agreement is at the level
of the floating-point summation order, as intended: the charges are a single
global reduction Q_i = u_i·(Kφ) and the topology is computed on the serial mesh
that every rank holds.

## Readout-port gap capacitance

A superconducting ring (major radius 0.3 m, cross-section 0.05 × 0.05 m) split by
a 0.02 m gap filled with a vacuum `port` region, inside a 10 kA·turn coil; the
readout current crosses the gap.

| quantity | value |
|---|---|
| terminal patch area (each) | 2.5000e-03 m² (exactly w·t = 0.0025 m²) |
| `C_gap_port_region_F` (gap region alone, no fringing) | 1.10934621e-12 F |
| parallel-plate estimate ε₀A/g | 1.10677348e-12 F |
| ratio | **1.0023** |
| `C_gap_F` (whole domain, fringing at the two mouths) | 1.76772292e-12 F (1.59× the region value) |
| `C_terminal_to_ground_F` | see the record |
| gap symmetry / energy checks | 6.8e-17 / 2.2e-07 |
| charge collected beyond the terminal patches | 53 % |
| `L_p` (magnetoquasistatic) | 8.86e-07 H |
| `self_resonance_Hz` = 1/(2π√(L_p C_gap)) | 1.27e+08 Hz |
| `self_resonance_from_port_region_Hz` (region value; an estimate, not a bound) | 1.60e+08 Hz |

The port-region capacitance reproduces the parallel-plate value of the gap to
**0.23 %**, which is the quantitative check of the two-electrode problem (the gap
is 0.02 m across a 0.05 × 0.05 m face, so fringing inside the region is small).
Adding the whole domain raises it by 59 % through the fringing field at the two
mouths of the gap, and the fraction of the electrode charge that arrives outside
the terminal patches proper is 53 % — much larger than the 20–28 % of the
axisymmetric templates, because this gap is short and wide. Quote `C_gap_F` to
two digits; it is a lumped-element estimate and a lower bound on the capacitance
that the readout sees (see `docs/validation/electrostatics.md` for the comparison
with the full-wave pole of L(ω), which is axisymmetric only).

**Caveat on this case:** the *magnetoquasistatic* pickup solve of this design hit
the AMS-PCG iteration cap (relative residual 3e-06, a warning in the record),
which is the known AMS robustness problem with thin slits ([ams_robustness.md](ams_robustness.md)).
So `L_p` and the self-resonance above are indicative only. The electrostatic
solves converged to 1e-14 and the capacitances are unaffected.

## Not covered

- A floating conductor with a prescribed **nonzero** charge (the axisymmetric
  solver supports it; 3D recovers a floating state from the exact matrix
  reduction only, checked against the surface-flux charges).
- `pickup.port_capacitance.vs_fullwave` (the comparison with the pole of L(ω)):
  axisymmetric only, because the 3D full-wave solver has no converged sweep yet.
- Zero-thickness conductor sheets: 3D designs have no `curves` (model sheets as
  thin regions), so the axisymmetric sheet-conductor case has no counterpart.
- Dielectrics touching the sphere at `mesh.domain_radius`: ε_out = ε₀ is assumed
  without a check (the mesher refuses a region that reaches *into* the shell, but
  one may touch R_b).
- The electrostatic potential and the surface charge density are not exported to
  scenes or VTU.
