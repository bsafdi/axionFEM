# Chemistry verification (beta)

The chemistry module (`axionfem.chem`) is **beta**. This page lists every verification case of the module and what its
last run measured. The formulation is described in the [chemistry guide](../guides/chemistry.md); the case format
and the runner in `benchmarks/fluids/README.md` and `benchmarks/chem/README.md`.

## What the cases check

- **Time integrators**: orders of convergence of BDF1-5 and ESDIRK3(2)4L[2]SA on manufactured problems
  (fixed and variable steps, Prothero-Robinson, dense output and event location), L-stability, the local error
  estimate, the variable-order driver's response to the tolerance, and stiff problems (a Robertson-type and a Van der
  Pol problem with chosen constants, ODE and DAE forms) against tight-tolerance reference solutions.
- **Kinetics kernel**: rates of progress, unit conversion and the analytic Jacobian (Taylor test).
- **Reactors and equilibrium**: batch, CSTR, semibatch, constant-pressure gas, adiabatic and plug-flow reactors
  and DAE equilibrium against closed forms (linear networks through the matrix exponential), mass and element
  balances, and a reaction network against Cantera.
- **Current distribution**: orders of convergence on manufactured potentials (P1/P2, planar,
  axisymmetric, 3D; Butler-Volmer, linear and Tafel kinetics; potentiostatic and galvanostatic), closed forms
  (1D cells, concentric electrodes, the disk electrode, a galvanic couple, the linear porous electrode), current
  balances, and comparisons with DOLFINx 0.11.0, with a series solution of the disk electrode and with the
  axisymmetric electrostatics solver of axionFEM on the same meshes.

## What is not verified

- **No published benchmark value is compared**: the stiff test problems use chosen constants, not the literature
  ones, and the `literature` verdict is `NOT_TESTED` for every case.
- No comparison with experiments; no mass transport, migration, double layer or transient electrochemistry exists
  to verify (see the limitations in the guide).
- The two 3D current-distribution cases (`chem_mms_c3_3d`, `chem_analytic_c3_concentric_3d`, four MPI ranks; their
  case definitions allow 6 and 2 hours in full) were run in **quick mode only** in the last run: their criteria are
  met on the first two levels, but the orders and Richardson extrapolations on the finer levels are not confirmed by
  it.
- The comparison with axionFEM's own electrostatics solver checks the current-distribution assembly against an
  independent code path of the same library, not against another code.

## Last run

Run on 2026-11-08 (axionFEM 0.1.0) in an MPI build (GCC 14, Release; MFEM 4.10.0 with METIS; no STRUMPACK, so the
direct solves use the Eigen fallback) on one 128-core CPU node. Every short case was run in full (`run --cost short`),
the long cases except the two 3D (multi-rank) ones in full, and every case in quick mode (`run --all --quick`); the
table shows the full result where there is one.

Columns: *evidence* is the kind of check (`mms`: manufactured solution, `analytic`: closed form, `conservation`:
balances and identities, `cross_code`: another code or an independent reference computation); *run* is `full` or
`quick` (the first two levels only: orders cannot be judged, the row is a smoke test); *result* is `all criteria met`
(in quick mode `all criteria met (on two levels)` or `smoke only`) or the failed acceptance items; *measured* gives
the observed orders minus the expected ones, the largest deviation of a comparison as a fraction of its tolerance (1 =
at the tolerance), and the number of identity and bound checks.

**Summary.** 37 of the 39 cases were run in full and all 37 meet all their criteria: 237 observed orders of
convergence within each case's tolerance, closed forms and the Cantera, DOLFINx, series and electrostatics references
within their tolerances, balances at round-off.

### Time integrators

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_stiff_robertson_dae_modified` | cross_code, conservation | short | full | all criteria met | 90 comparisons (ref_robertson_modified), largest deviation 0.06 of the allowance; 3 checks |
| `chem_stiff_robertson_modified` | cross_code, conservation | short | full | all criteria met | 90 comparisons (ref_robertson_modified), largest deviation 0.07 of the allowance; 4 checks |
| `chem_stiff_vdp_modified` | cross_code | short | full | all criteria met | 32 comparisons (ref_vdp_modified), largest deviation 0.06 of the allowance; 2 checks |
| `chem_time_dense_events` | mms | short | full | all criteria met | 6 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.03 to 5.8e-03; 3 checks |
| `chem_time_estimator` | mms | short | full | all criteria met | 13 checks |
| `chem_time_lstability` | analytic | short | full | all criteria met | 6 checks |
| `chem_time_mms_fixed` | mms | short | full | all criteria met | 16 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.03 to 0.01 |
| `chem_time_prothero_robinson` | mms | short | full | all criteria met | 6 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.03 to -3.8e-04 |
| `chem_time_varstep` | mms | short | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -0.02 to 6.1e-03; 16 checks |

### Kinetics kernel

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_jacobian_taylor` | mms | short | full | all criteria met | 6 checks |
| `chem_kernel_rates` | analytic | short | full | all criteria met | 21 comparisons (kernel_c1), largest deviation 5.0e-04 of the allowance; 5 checks |
| `chem_kernel_units` | analytic | short | full | all criteria met | 48 comparisons (kernel_c1), largest deviation 5.0e-05 of the allowance |

### Reactors and equilibrium

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_analytic_adiabatic` | analytic, conservation | short | full | all criteria met | 10 comparisons (quadrature_c1), largest deviation 0.07 of the allowance; 2 checks |
| `chem_analytic_batch_kinetics` | analytic, conservation | short | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected -0.04 to 0.03; 106 comparisons (closed_form_c1, expm_c1), largest deviation 0.06 of the allowance; 8 checks |
| `chem_analytic_dae_equilibrium` | analytic | short | full | all criteria met | 3 orders (expected 1.0, 2.0, 3.0), observed minus expected -8.9e-03 to -1.3e-03; 20 comparisons (closed_form_c1), largest deviation 0.06 of the allowance; 2 checks |
| `chem_analytic_reactors` | analytic | short | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -0.03 to 6.1e-03; 48 comparisons (closed_form_c1), largest deviation 0.06 of the allowance; 1 check |
| `chem_conservation_network` | conservation | short | full | all criteria met | 6 checks |
| `chem_cross_cantera_network` | cross_code | short | full | all criteria met | 60 comparisons (cantera_xc), largest deviation 0.07 of the allowance; 1 check |
| `chem_mms_reactors` | mms | short | full | all criteria met | 10 orders (expected 2.0, 3.0), observed minus expected -0.03 to 0.01 |

### Variable-order BDF

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_vo_bdf345_fixed` | mms | short | full | all criteria met | 20 orders (expected 3.0, 4.0, 5.0), observed minus expected -0.06 to 0.04 |
| `chem_vo_own_bdf345` | mms, analytic | short | full | all criteria met | 11 orders (expected 3.0, 4.0, 5.0), observed minus expected -0.05 to 0.02; 27 checks |
| `chem_vo_rtol_sweep` | mms | short | full | all criteria met | 3 checks |
| `chem_vo_stiff_robertson_dae_modified` | cross_code, conservation | short | full | all criteria met | 45 comparisons (ref_robertson_modified), largest deviation 0.05 of the allowance; 3 checks |
| `chem_vo_stiff_robertson_modified` | cross_code, conservation | short | full | all criteria met | 45 comparisons (ref_robertson_modified), largest deviation 0.06 of the allowance; 4 checks |
| `chem_vo_stiff_vdp_modified` | cross_code | short | full | all criteria met | 16 comparisons (ref_vdp_modified), largest deviation 0.11 of the allowance; 2 checks |

### Current distribution, 1D, planar and axisymmetric

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_analytic_c3_cell_1d` | analytic, conservation | short | full | all criteria met | 17 comparisons (b1_closed_form), largest deviation 3.0e-05 of the allowance; 12 checks |
| `chem_analytic_c3_concentric` | analytic, conservation | long | full | all criteria met | 27 Richardson/GCI (orders 0.84 to 3.91); 18 comparisons (b2_closed_form), largest deviation 0.35 of the allowance |
| `chem_analytic_c3_disk_primary` | analytic, conservation, cross_code | long | full | all criteria met | 4 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.06 to 0.02; 2 Richardson/GCI (orders 1.90 to 2.90); 22 comparisons (b4_closed_form, disk_dolfinx), largest deviation 0.35 of the allowance; 14 checks |
| `chem_analytic_c3_galvanic_1d` | analytic, conservation | short | full | all criteria met | 5 comparisons (galvanic_closed_form), largest deviation 4.7e-06 of the allowance; 3 checks |
| `chem_crosscode_c3_disk_secondary` | cross_code, conservation | long | full | all criteria met | 5 Richardson/GCI (orders 3.54 to 3.67); 30 comparisons (disk_dolfinx, disk_series), largest deviation 0.02 of the allowance; 3 checks |
| `chem_crosscode_c3_electroproblem_primary` | cross_code | long | full | all criteria met | 140 comparisons (electroproblem), largest deviation 3.1e-03 of the allowance; 3 checks |
| `chem_mms_c3_annulus_2d` | mms, conservation | long | full | all criteria met | 12 orders (expected 1.0, 2.0, 3.0, 4.0), observed minus expected -0.03 to 1.5e-03; 4 checks |
| `chem_mms_c3_axi` | mms, conservation | long | full | all criteria met | 20 orders (expected 1.0, 2.0, 3.0, 4.0), observed minus expected -0.08 to 0.08; 14 checks |
| `chem_mms_c3_bv_2d` | mms, conservation | long | full | all criteria met | 50 orders (expected 1.0, 2.0, 3.0, 4.0), observed minus expected -0.01 to 0.06; 34 checks |
| `chem_mms_c3_galvanostatic_2d` | mms, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0, 4.0), observed minus expected -8.5e-03 to 0.04; 9 checks |
| `chem_mms_c3_kappa_jump_2d` | mms, conservation | long | full | all criteria met | 8 orders (expected 1.0, 2.0, 3.0), observed minus expected -5.6e-03 to -1.5e-04; 8 checks |

### Current distribution, 3D (MPI) and porous electrode

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `chem_analytic_c3_concentric_3d` | analytic, cross_code, conservation | long | quick | all criteria met (on two levels) | 4 orders (expected 2.0, 4.0), observed minus expected -2.1e-03 to -8.1e-05; 8 Richardson/GCI (orders 1.99 to 3.91); 40 comparisons (b2_closed_form, conc3d_dolfinx), largest deviation 0.46 of the allowance; 10 checks |
| `chem_analytic_c3_porous_1d` | analytic, conservation | short | full | all criteria met | 27 orders (expected 1.0, 2.0, 3.0, 4.0), observed minus expected -0.01 to -1.4e-07; 10 comparisons (porous_closed_form), largest deviation 0.31 of the allowance; 32 checks |
| `chem_mms_c3_3d` | mms, conservation | long | quick | all criteria met (on two levels) | 20 orders (expected 1.0, 2.0, 3.0, 4.0), observed minus expected -0.06 to -5.3e-03; 29 checks |

39 cases: 37 run in full (37 meet all their criteria), 2 run in quick mode only, 0 not run. The full runs contain 237 observed orders of convergence, 34 Richardson/GCI estimates, 873 comparisons with reference values and 253 checks.

## Reference sources

| source | what it is |
|---|---|
| `closed_form_c1`, `expm_c1`, `quadrature_c1`, `kernel_c1` | closed forms of the reactor problems, matrix exponentials of linear networks, a quadrature of the adiabatic closed form, rate expressions evaluated by hand |
| `b1_closed_form`, `b2_closed_form`, `b4_closed_form`, `galvanic_closed_form`, `porous_closed_form` | closed forms of the 1D cell, the concentric electrodes, the disk electrode (primary), the galvanic couple and the linear porous electrode |
| `ref_robertson_modified`, `ref_vdp_modified` | reference trajectories of the two stiff problems from a high-precision Radau IIA-5 integration (mpmath) checked against SciPy's Radau and LSODA at tight tolerances |
| `cantera_xc` | Cantera 3.2.0 (ReactorNet with CVODES) on the same reaction network |
| `disk_dolfinx`, `conc3d_dolfinx` | DOLFINx 0.11.0 on the same discrete problems |
| `disk_series` | Legendre series solution of the disk electrode with linear kinetics |
| `electroproblem` | axionFEM's axisymmetric electrostatics solver on the same meshes (primary distribution) |

The scripts that produced the cross-code and high-precision references are not part of the repository; their
numbers are in the case definitions. Regenerate the tables with `python benchmarks/chem/run.py report DIR [DIR ...]`
from the `--out` directories of `run` (a full result is preferred to a quick one).
