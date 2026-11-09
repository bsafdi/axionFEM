# Robustness of the AMS-preconditioned 3D solves

With a gauge mass scaled with the whole mesh, the static and pickup 3D solves stall or break down next to thin
features, slits and superconducting bodies, and at ND3: the toroid at ND2 with a 1.5° slit hits the 800-iteration cap
in all three solves, the ND3 toroid pickup solve stops at a relative residual of 0.38, CORE at ND2 runs for hours,
and typical counts are 160–370 per ND2 solve.

The cause is the gauge regularization of the curl-curl operator, not AMS. With it scaled per element (the default),
every one of those cases converges to a relative residual of 1e-10 or below: the toroid with a 1.5° slit at ND2 in
200 instead of 2400 capped iterations (9× less solver time), the same design at ND3 in 570 instead of 2400
(3× less), and CORE with iron at ND2, which breaks down after a few iterations with the global gauge mass, in ~150 per
solve. The per-element scaling also avoids a failure on the full-wave diagnostic designs (a solenoid with a coax
pickup inside a large infinity shell), where the global gauge mass gives a static field of pure noise.

This page records the cause, the remedy, and the measurements of both formulations.

- Benchmark: `benchmarks/ams_robustness.py` (cases and solver variants).
- Record: `benchmarks/records/ams_robustness.json`.
- Variant `m2` is the global-gauge formulation with hypre's AMS defaults (iteration counts 42 + 43 + 89 on the ND1
  toroid case), selectable in the same build, so both columns come from one build.

## The cause: the gauge mass was scaled with the whole mesh, not with the element

`curl(ν curl ·)` is singular on gradients, so the assembled operator carries a mass term `ε |u|²`. With a global
gauge mass, `ε = 10⁻⁸ ν₀/L²` with `L` the size of the whole mesh *including the infinity shell*. In an element
of size `h_e` the curl-curl entries scale like `ν_e h_e` and the mass entries like `ε h_e³`, so the gauge
term is a relative perturbation

    κ_eff(e) = ε h_e² / ν_e = 10⁻⁸ (h_e/L)²  (global gauge)

of the local stiffness. With a device of 1 m, a shell of `L ≈ 10 m` and slit elements of `h_e ~ 10⁻³ m`,
κ_eff ~ 10⁻²² — far below double-precision round-off. The gauge term is then *lost* when the matrix is
assembled: AMS's gradient-space (β-Poisson) matrix `Gᵗ A G = ε Gᵗ M G` is numerically singular, its AMG
solve returns garbage, and PCG either stalls or breaks down (`(Br, r) < 0`, `(Ad, d) < 0`).

The `(h_e/L)²` is directly visible in the `solenoid_coax_nd2_R*` cases below, which are the same design
meshed inside a larger and larger infinity shell: with the global ε the static solve stalls at a
relative residual of 0.25 at `domain_radius` = 3 m and at 16 at 4.5 m (a 73 % error in Ψ and a factor 5 in W,
i.e. a static field that is essentially noise), purely because `L` grew. With the per-element ε both
converge to 1e-10.

The remedy is to scale the gauge mass per element,

    ε_e = κ ν_e / h_e²   with h_e the smallest element width,

so that κ_eff = κ *everywhere*, independent of the mesh, the device size and the shell radius. In the
infinity shell `ν_e` is the element mean of the trace/3 of the mapped tensor (which vanishes tangentially
at the outer radius); in iron it is `ν₀`, the largest value `ν(|B|)` can take as it saturates.

### Calibrating κ

κ is bounded on both sides, and both bounds are measured.

**The upper bound is accuracy.** The error of every reported quantity is linear in κ, as first-order
perturbation theory requires (toroid, 1.5° slit, h = 0.45, ND1, against the exactly gauge-free solve
`singular`, i.e. ε = 0 with AMS's `SetSingularProblem`):

| κ (variant) | PCG iterations | max rel. residual | ΔΨ | ΔL_p | ΔW |
|---|---|---|---|---|---|
| 10⁻¹¹ (`gauge1e-11`) | 104 | 4e-11 | -6.4e-07 | -1.3e-06 | -4.8e-09 |
| 10⁻¹² (`new` at order ≤ 2) | 138 | 6e-11 | -6.4e-08 | -1.3e-07 | -4.7e-10 |
| 10⁻¹³ (`gauge1e-13`) | 142 | 6e-11 | -6.4e-09 | -1.3e-08 | -4.5e-11 |
| 10⁻¹⁴ (`gauge1e-14`) | 102 | 7e-11 | -6.7e-10 | -1.3e-09 | -2.2e-12 |
| 10⁻¹⁵ (`gauge1e-15`) | 8 | 2e+01 — breakdown | — | — | — |
| 0 (`gauge0`) | 24 | 1.0 — breakdown | — | — | — |
| 0 + singular mode (`singular`) | 209 | 2e-11 | reference | | |
| 10⁻⁸ ν₀/L² (`gauge_global`, global gauge) | 104 | 4e-11 | -2.8e-09 | -5.1e-09 | -3.7e-10 |

**The lower bound is a round-off floor, and it rises with everything that makes the element matrices worse
conditioned** — the aspect ratio of the elements and the polynomial order — because the gauge term has to
survive the cancellation `Gᵗ K G = 0` in AMS's β-Poisson matrix. Iterations of the first static solve
(`cap` = stopped at the 800-iteration cap, `breakdown` = PCG quit within a few iterations, both reported as
"results are NOT reliable"):

| case | `m2` (global gauge) | 10⁻¹⁴ | 10⁻¹³ | 10⁻¹² | 10⁻¹¹ |
|---|---|---|---|---|---|
| toroid ND1, 1.5° slit | 42 | 31 | 32 | **32** | 32 |
| toroid ND2, 1.5° slit | cap (4.7e-4) | cap (1.9e+02) | 56 | **57** | 57 |
| CORE ND2, iron, coarse surface mesh | breakdown (5 its) | — | 777 ᵃ | **139** | 138 |
| toroid ND3, 3° slit | cap (0.19) | — | cap (1.1) | breakdown (24 its) | **140** |

(bold = the default for that order; ᵃ from an earlier run of the same design and mesh, not in the current
record. The `m2` column is the global gauge mass *and* hypre's AMS defaults.)

The default is therefore **κ = 10⁻¹² at order ≤ 2 and κ = 10⁻¹¹ at order ≥ 3**: three decades above the
floor in the validated (ND1/ND2) regime, where it costs 6e-8 of Ψ — four orders below the 3D-vs-2D
discretization differences that regime is quoted at — and the larger value at order 3, which is the only
one that converges there (10⁻¹² breaks down, 10⁻¹³ stalls). `AXF_GAUGE` overrides it, and
`tests/test_3d.py` compares the default against the gauge-free solve and fails if the perturbation grows
past 2e-8.

## Other changes

- **AMS/AMG options** (`HYPRE_AMSSetAlphaAMGOptions`/`BetaAMGOptions`): strength threshold θ = 0.5 (hypre's
  default is 0.25, θ = 0.5 is the usual 3D value) and no aggressive coarsening (MFEM's default is one
  level). On the ND2 slit case the two together save a quarter of the iterations (263 → 199 at the same
  gauge mass); on the ND1 case they interact non-monotonically with the pickup solve (see below), which is
  why the choice follows the larger cases.
- **The operator and the AMS setup are reused** between solves that share them: the lift solves of
  `solve_with_bodies`, the extra lift solves of each Newton step, and the pickup solves. The same holds for
  the right-hand-side-independent part of the divergence cleaning (the H1 operator `Gᵗ M G`, its BoomerAMG
  setup, the body-gradient modes and their Gram matrix). At the same iteration count this takes the solver
  time of the ND1 case below from 13.0 s to 9.7 s (`m2` vs `m2_plus_reuse`), i.e. a quarter of the cost of a
  three-solve phase; the saving grows with the number of solves (five for CORE with iron, 1 + n per Newton
  step).
- **The preconditioner is applied on the free DOFs only** (`FreeDofSolver`). Eliminated essential DOFs are
  decoupled rows whose residual is zero throughout the iteration; a correction placed there by AMS only
  adds a badly scaled component to the Krylov space. The measured effect is neutral on the cases below,
  but it keeps the body DOFs exactly at their lift values.
- **PCG comes from MFEM** (`mfem::CGSolver`) instead of `HyprePCG`, because hypre's PCG only accepts a
  `HypreSolver` as preconditioner. The iteration counts are identical where both apply (36 vs 36 on a
  superconducting ring, 174 vs 174 on the ND1 toroid case).

## Both formulations, case by case

Generated by `python benchmarks/ams_robustness.py collect runs/ams`; `m2` (the record's name) is the global-gauge
formulation, `new` the default (local) one, and the Ψ/L_p/W columns are relative to `new` (a case whose solve stalled is marked, and
its results are meaningless). "solver" is the time of the assembly, the AMS setup, the cleaning and the
iterations of all solves; "total" includes meshing and post-processing.

<!-- TABLE -->

### toroid_slit1.5_nd1_h0.45 (71,494 DOFs, 1 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 174 (42 + 43 + 89) | 2.5e-11 | 13 | 39 | 0.2 | +6.1e-08 | +1.2e-07 | +1.0e-10 |
| `new` | 138 (32 + 32 + 74) | 5.6e-11 | 10 | 36 | 0.2 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `m2_plus_reuse` | 174 (42 + 43 + 89) | 2.5e-11 | 10 | 36 | 0.2 | +6.1e-08 | +1.2e-07 | +1.0e-10 |
| `gauge_global` | 104 (32 + 32 + 40) | 4.0e-11 | 8 | 34 | 0.2 | +6.1e-08 | +1.2e-07 | +1.1e-10 |
| `gauge1e-11` | 104 (32 + 32 + 40) | 4.1e-11 | 8 | 34 | 0.2 | -5.8e-07 | -1.1e-06 | -4.3e-09 |
| `gauge1e-12` | 138 (32 + 32 + 74) | 5.6e-11 | 10 | 36 | 0.2 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `gauge1e-13` | 142 (32 + 32 + 78) | 5.8e-11 | 10 | 36 | 0.2 | +5.8e-08 | +1.1e-07 | +4.3e-10 |
| `gauge1e-14` | 102 (31 + 32 + 39) | 6.8e-11 | 8 | 34 | 0.2 | +6.3e-08 | +1.3e-07 | +4.7e-10 |
| `gauge1e-15` | 8 (2 + 4 + 2) **stall** | 2.2e+01 | 2 | 29 | 0.2 | -3.0e-01 | -2.9e-01 | -4.0e-02 |
| `gauge0` | 24 (2 + 21 + 1) **stall** | 1.0e+00 | 3 | 29 | 0.2 | -1.0e+00 | -1.0e+00 | -1.0e+00 |
| `amg_m2` | 181 (43 + 44 + 94) | 2.5e-11 | 10 | 36 | 0.2 | +1.4e-11 | +7.3e-13 | -6.2e-13 |
| `theta0.25` | 109 (34 + 34 + 41) | 4.4e-11 | 8 | 34 | 0.2 | +6.7e-12 | +2.8e-13 | -1.6e-12 |
| `agg1` | 114 (35 + 35 + 44) | 2.7e-11 | 7 | 33 | 0.2 | -4.2e-12 | +3.7e-13 | -2.7e-12 |
| `sweeps2` | 118 (28 + 27 + 63) | 6.0e-11 | 10 | 36 | 0.2 | +7.4e-12 | +4.0e-13 | -2.2e-12 |
| `sweeps3` | 109 (26 + 25 + 58) | 3.9e-11 | 10 | 36 | 0.2 | +2.3e-12 | -2.8e-13 | -2.3e-12 |
| `chebyshev` | 168 (37 + 39 + 92) | 5.0e-11 | 12 | 38 | 0.2 | -1.8e-12 | +3.6e-13 | -6.3e-12 |
| `maxit3000` | 138 (32 + 32 + 74) | 5.6e-11 | 10 | 36 | 0.2 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `singular` | 209 (64 + 66 + 79) | 2.1e-11 | 11 | 37 | 0.2 | +6.4e-08 | +1.3e-07 | +4.7e-10 |
| `fgmres` | 141 (32 + 29 + 80) | 5.7e-09 | 10 | 36 | 0.3 | +2.4e-10 | +8.8e-10 | -2.1e-10 |
| `no_proj` | 138 (32 + 32 + 74) | 5.6e-11 | 10 | 36 | 0.2 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

### toroid_slit3_nd2_h0.45 (377,452 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 248 (72 + 78 + 98) | 4.9e-11 | 24 | 59 | 0.7 | +4.7e-09 | +9.4e-09 | +7.2e-11 |
| `new` | 193 (55 + 59 + 79) | 7.9e-11 | 18 | 51 | 0.7 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

### toroid_slit1.5_nd2_h0.45 (382,132 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 2400 (800 + 800 + 800) **stall** | 4.7e-04 | 216 | 273 | 0.7 | -1.6e-04 | +1.9e-05 | +7.5e-05 |
| `new` | 199 (57 + 61 + 81) | 8.5e-11 | 24 | 80 | 0.7 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `m2_plus_reuse` | 2400 (800 + 800 + 800) **stall** | 4.7e-04 | 207 | 263 | 0.7 | -1.6e-04 | +1.9e-05 | +7.5e-05 |
| `gauge_global` | 2400 (800 + 800 + 800) **stall** | 7.5e-02 | 244 | 300 | 0.7 | +6.3e-05 | -7.5e-06 | +9.4e-06 |
| `gauge1e-11` | 199 (57 + 61 + 81) | 1.0e-10 | 24 | 81 | 0.7 | -3.5e-07 | -7.0e-07 | -3.4e-09 |
| `gauge1e-13` | 200 (56 + 61 + 83) | 9.1e-11 | 24 | 81 | 0.7 | +3.5e-08 | +7.0e-08 | +3.4e-10 |
| `gauge1e-14` | 2400 (800 + 800 + 800) **stall** | 1.9e+02 | 246 | 302 | 0.7 | +7.0e-01 | +2.2e+01 | +4.8e-03 |
| `amg_m2` | 263 (77 + 78 + 108) | 7.7e-11 | 26 | 83 | 0.7 | +2.7e-11 | -3.3e-12 | -1.0e-12 |
| `theta0.25` | 211 (59 + 66 + 86) | 7.9e-11 | 24 | 81 | 0.7 | +1.7e-11 | -3.1e-12 | +1.4e-12 |
| `agg1` | 222 (62 + 66 + 94) | 8.6e-11 | 23 | 79 | 0.7 | -9.5e-12 | -1.4e-12 | +2.7e-12 |
| `sweeps2` | 186 (53 + 58 + 75) | 8.9e-11 | 26 | 83 | 0.7 | +1.9e-11 | -1.8e-12 | -6.8e-13 |
| `singular` | 1063 (132 + 131 + 800) **stall** | 1.4e-04 | 87 | 143 | 0.7 | +4.0e-08 | +7.2e-08 | +3.7e-10 |
| `fgmres` | 204 (58 + 55 + 91) | 5.7e-11 | 25 | 82 | 0.7 | +3.0e-10 | +2.8e-12 | -1.2e-10 |
| `no_proj` | 199 (57 + 61 + 81) | 8.5e-11 | 24 | 81 | 0.7 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

### toroid_slit3_nd3_h0.45 (1,095,702 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 2400 (800 + 800 + 800) **stall** | 6.8e+02 | 902 | 977 | 1.6 | +5.6e-01 | +4.7e+02 | +1.6e+02 |
| `new` | 570 (140 + 167 + 263) | 1.9e-10 | 260 | 328 | 1.5 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `gauge1e-12` | 69 (24 + 23 + 22) **stall** | 9.4e-03 | 45 | 111 | 1.5 | -6.4e-05 | -1.1e-05 | -5.9e-07 |
| `gauge1e-13` | 2400 (800 + 800 + 800) **stall** | 1.1e+00 | 1055 | 1123 | 1.5 | +3.8e-04 | +3.0e-03 | -1.4e-05 |
| `maxit3000` | 570 (140 + 167 + 263) | 1.9e-10 | 259 | 326 | 1.5 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

### toroid_ideal_nd3_h0.45 (1,074,561 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 2067 (74 + 800 + 393 + 0 + 800) **stall** | 1.6e-10 | 917 | 1004 | 1.5 | +1.7e-08 | +2.8e-08 | +3.3e-09 |
| `new` | 358 (35 + 109 + 105 + 0 + 109) | 1.1e-09 | 191 | 270 | 1.4 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

### core_iron_nd2_h0.5_coarse_surface (1,475,106 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 51 (5 + 13 + 23 + 3 + 7) **stall** | 1.3e+01 | 62 | 343 | 3.0 | -1.0e+00 | -1.0e+00 | -8.8e-01 |
| `new` | 4771 (139 + 155 + ... + 171, 31 solves) | 5.0e-08 | 1873 | 2273 | 3.0 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `gauge1e-11` | 4766 (138 + 154 + ... + 171, 31 solves) | 5.2e-09 | 2087 | 2522 | 3.0 | -2.0e-06 | -3.4e-06 | -5.3e-07 |

### solenoid_coax_nd2_R3 (967,144 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 901 (800 + 101) **stall** | 2.5e-01 | 205 | 431 | 2.3 | +3.1e-03 | +3.0e-10 | +6.7e-03 |
| `new` | 225 (81 + 144) | 1.1e-10 | 71 | 291 | 2.3 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `gauge1e-13` | 162 (79 + 83) | 9.3e-11 | 56 | 277 | 2.3 | +6.5e-10 | +8.6e-10 | +3.1e-10 |

### solenoid_coax_nd2_R4.5 (1,303,798 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 1600 (800 + 800) **stall** | 1.6e+01 | 452 | 702 | 3.1 | +7.3e-01 | +6.6e-10 | +4.2e+00 |
| `new` | 347 (116 + 231) | 1.4e-10 | 138 | 383 | 3.1 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `gauge1e-13` | 568 (127 + 441) | 9.9e-11 | 209 | 449 | 3.1 | +6.4e-10 | +8.6e-10 | +3.2e-10 |

### sc_sphere_h0.12 (193,678 DOFs, 4 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 79 (79) | 6.8e-12 | 8 | 22 | 0.5 | - | - | +1.2e-15 |
| `new` | 67 (67) | 6.6e-12 | 8 | 21 | 0.5 | - | - | +0.0e+00 |
| `gauge_global` | 67 (67) | 6.6e-12 | 7 | 21 | 0.5 | - | - | +8.8e-16 |
| `amg_m2` | 79 (79) | 6.9e-12 | 7 | 21 | 0.5 | - | - | -2.9e-16 |

### sc_sphere_h0.08 (315,584 DOFs, 8 MPI ranks)

| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] | peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |
|---|---|---|---|---|---|---|---|---|
| `m2` | 81 (81) | 7.6e-12 | 8 | 40 | 0.6 | - | - | +3.5e-15 |
| `new` | 70 (70) | 6.3e-12 | 8 | 31 | 0.6 | - | - | +0.0e+00 |
<!-- /TABLE -->

The pickup solve is consistently the hardest of the three (81 of the 199 iterations of the ND2 slit case),
and its count is sensitive at the factor-of-two level to changes of the operator far below the
discretization error (39 at κ = 10⁻¹⁴ against 78 at 10⁻¹³ on the ND1 case, with the same residual reached):
that solve runs close to the stagnation regime of PCG, so its iteration count should not be read as a smooth
function of the settings.

## What did not work

- **`mfem::LORSolver<mfem::HypreAMS>` (low-order-refined AMS) is not usable on our meshes.** MFEM 4.10's
  `LORBase::ConstructLocalDofPermutation` casts every element to `TensorBasisElement` (quadrilaterals and
  hexahedra) with an `MFEM_ASSERT`, which is compiled out in a release build; on a tetrahedral ND space it
  dereferences a null pointer and the solve segfaults. The meshes produced by `axf-mesher` are
  tetrahedral, so neither `LORSolver` nor `ParLORDiscretization` can be used, with or without the
  (GaussLobatto, IntegratedGLL) basis pair, until MFEM supports simplices there. (The basis pair itself
  has no effect on tetrahedra: `ND_TetrahedronElement` carries a fixed basis.)
- **A two-level p-multigrid V-cycle** (AMS on the nested ND1 space of the same mesh as the coarse solver,
  l1-SSOR or l1-Jacobi on the ND_p operator as the fine-level smoother, `mfem::Multigrid`) — the natural
  substitute for LOR at high order — is far worse than AMS alone: on a superconducting ring at ND2, 800
  iterations at a relative residual of 6e-2 (l1-SSOR) or 1.7e-1 (l1-Jacobi) against 67 iterations at 2e-11.
  A point smoother does not smooth the gradient near-nullspace of H(curl), which is exactly what AMS's own
  Hiptmair smoothing (a correction in the potential space) is for; a usable p-multigrid would need that
  smoother on the fine level, i.e. essentially AMS itself. It is therefore not offered.
- **Preconditioning with a larger gauge mass than the operator** (AMS built from `curl ν curl + ε_pc M`
  with κ_pc ≫ κ, which would be well conditioned by construction) is much worse: 36 iterations with
  κ_pc = κ against 304 with κ_pc = 10⁻⁶ on a small superconducting ring at ND1 (a case outside the
  benchmark, measured with the global gauge). The gradient components the preconditioner then
  mis-scales by κ_pc/κ are excited by round-off and converge at that rate.
- **Rescaling the diagonal of the eliminated (essential) rows** to the mean free diagonal — the unit
  diagonal left by `EliminateRowsCols` is 4–6 orders below the curl-curl diagonal — makes PCG *slower*
  (36 → 78 iterations on the same ring case): those decoupled rows contribute negligibly to the residual as
  long as their diagonal is small, and rescaling them makes the preconditioner's pollution of them visible.
- **AMS's singular mode** (`SetSingularProblem`, with ε = 0, which is the only consistent way to use it)
  works and is exactly gauge-free, but it is slower where it works (209 against 138 iterations on the ND1
  toroid case) and it stalls where it matters: on the ND2 slit case its first two solves converge in 132
  and 131 iterations and the pickup solve runs into the cap at 1.4e-4. It is kept as `AXF_AMS_SINGULAR=1`
  with `AXF_GAUGE=0` and is the reference the gauge perturbation is measured against.
- **FGMRES** instead of PCG is as good as PCG but not better (141 against 138 iterations on the ND1 case,
  204 against 199 on the ND2 slit case) and costs 50 stored vectors; `AXF_SOLVER_FGMRES=1` keeps it
  available. Two or three AMS smoother sweeps buy a few per cent of the iterations (186 against 199 at ND2
  with two sweeps) and lose it again in time per iteration; Chebyshev smoothing is worse (168 against 138 at
  ND1).
- **AMS cycle types outside the 10–14 family crash.** MFEM builds AMS's interpolation matrices for the
  cycle type it was constructed with (13), and the types 1–8 need a single Π block instead of Π_x, Π_y, Π_z:
  `HYPRE_AMSSetCycleType(1)` segfaulted on all 8 ranks of the ND2 slit case (it survived on one rank, 125
  iterations). The knob refuses anything outside 10–14. Within the family, 11 and 14 change the counts
  by ±20 % without a consistent winner, as do two or three smoother sweeps.

## Environment knobs

All of them are solver options: within the solver tolerance the results must not depend on them
(`SolverOpts` in `cpp/src/model3d.cpp`, recorded in `results.json` under `solver.options`).

| variable | default | meaning |
|---|---|---|
| `AXF_GAUGE` | 1e-12, or 1e-11 at order ≥ 3 | κ of the gauge mass `ε_e = κ ν_e/h_e²` |
| `AXF_GAUGE_GLOBAL` | 0 | 1: one `ε = κ ν₀/L²` for the whole mesh (the `m2` formulation, with κ = 1e-8) |
| `AXF_GAUGE_HVOL` | 0 | 1: `h_e = |element|^(1/3)` instead of the smallest element width |
| `AXF_AMS_THETA` | 0.5 | AMG strength threshold inside AMS |
| `AXF_AMS_AGG` | 0 | aggressive-coarsening levels inside AMS |
| `AXF_AMS_CYCLE` | 13 | AMS cycle type (10–14 only; see below) |
| `AXF_AMS_RLX` / `AXF_AMS_SWEEPS` | 2 / 1 | AMS smoother type and sweeps |
| `AXF_AMS_SINGULAR` | 0 | 1: AMS singular mode (use with `AXF_GAUGE=0`) |
| `AXF_AMS_ESS_PROJ` | 1 | 0: let the preconditioner act on the essential DOFs too |
| `AXF_SOLVER_REUSE` | 1 | 0: reassemble the operator, AMS and the cleaning for every solve |
| `AXF_SOLVER_FGMRES` | 0 | 1: FGMRES instead of PCG |
| `AXF_SOLVER_TOL` / `AXF_SOLVER_MAXIT` | 1e-11 / 800 | Krylov tolerance and iteration cap |
| `AXF_SOLVER_PRINT` | 0 | Krylov/AMS print level |

Note for SLURM: where a site sets `SLURM_EXPORT_ENV=NONE`, `srun` does **not** pass these variables to the ranks
unless the script exports `SLURM_EXPORT_ENV=ALL` or uses `srun --export=ALL`.
`benchmarks/ams_robustness.py collect` checks `solver.options` against the variant it asked for and prints
a warning if a variant did not reach the solver.

## Remaining limitations

- **ND3 is solvable, but not validated.** At κ = 10⁻¹¹ the ND3 toroid case converges
  (140 + 167 + 263 iterations, 331 s, residuals ≤ 1.9e-10, against three capped solves and 972 s with `m2`),
  but there is no h/p convergence study of Ψ, L_p and W at order 3 and the gauge bias there is 6e-7 instead
  of 6e-8, so `mesh.order >= 3` keeps its warning and order 2 stays the production choice. A p-convergence
  study should set `AXF_GAUGE` explicitly, so that every order carries the same bias.
- **κ is calibrated, not derived.** It is a round-off floor, so a mesh with worse elements, a much larger
  device-to-shell ratio or a higher order can still fall below it. The failure is loud (the true-residual
  check fires and the warning names the knob), but the solver does not adapt by itself. Estimating the floor
  from the assembled matrices (comparing |Gᵗ K G| with ε |Gᵗ M G|) would remove the calibration; it was not
  attempted.
- **A larger infinity shell does not cost robustness, but it costs iterations.** The
  `solenoid_coax_nd2_R*` cases need 225 and 347 iterations at `domain_radius` 3 m and 4.5 m: the shell's
  mapped tensor becomes more anisotropic as it stretches further, which AMS feels. The global-gauge failure on
  these two designs is pure gauge round-off, not the shell itself.
- **A mesh that does not resolve its thin features stays expensive.** The CORE case's surface mesh is 17×
  coarser than the 10 mm walls and the 10 mm neck gap it has to resolve. That is what makes its round-off
  floor so high (κ = 10⁻¹³ still needs 777 iterations for the first solve where 10⁻¹² needs 139), and
  `surface_size` is the real fix there.
- **One CORE solve stops short of the requested tolerance.** The Krylov tolerance is a reduction of the
  *preconditioned* residual; on the CORE case one of the 31 solves stopped at a true relative residual of
  5.0e-8 with 1e-11 requested (the preconditioned norm underestimates the true one there). It is far inside
  the 1e-6 reliability check and the per-solve residuals are in `results.json`, but a case that needs more
  should lower `AXF_SOLVER_TOL`.
- **CORE at ND2 was not re-measured on one rank.** The global-gauge solver breaks down on it at 1 rank exactly as
  at 8 (2–9 iterations per solve at residuals 0.04–2.9), so the breakdown is not an MPI artefact; the 1-rank
  run of the per-element-gauge solver (31 solves of ~150 iterations on 1.5 M DOFs) is not in the record because of
  its cost. `core_iron_nd2_h0.5_coarse_surface_np1` in the benchmark runs it.
- **The superconducting-sphere case is small.** The design here has 0.3 M DOFs and converges with both
  formulations (81 → 71 iterations); a 6.5 M-DOF sphere at h = 0.08, where a stall was seen, is not in the
  repository.
- **Memory is unchanged** by the new settings (0.7 GB/rank at 0.38 M DOFs, 3.0 GB/rank at 1.48 M DOFs, the
  same for `m2` and `new`): switching off aggressive coarsening costs no measurable memory at these sizes.
- **The 3D full-wave solver's preconditioner is untouched.** `model3d_fullwave.cpp` builds AMS on the SPD
  magnitude operator with hypre's defaults (θ = 0.25, one level of aggressive coarsening) and its own PML
  mass term; the same θ and coarsening settings may help there, but they have not been tested on it.
