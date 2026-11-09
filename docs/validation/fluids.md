# Fluids verification (beta)

The fluids module (`axionfem.fluids` with `axionfem.transport`) is **beta**. This page lists every verification case
of the module and what its last run measured. The formulation is described in the [fluids guide](../guides/fluids.md);
the case format and the runner in `benchmarks/fluids/README.md`.

## What the cases check

- **Manufactured solutions** (`mms`): observed orders of convergence of the velocity, pressure, temperature and
  potential errors (L2, H1) against the orders of the elements, for Taylor-Hood Stokes (planar, axisymmetric with
  swirl, 3D), steady and transient Navier-Stokes, SUPG scalar transport, Boussinesq coupling, stabilized P1/P1 (GLS)
  and inductionless MHD, with every boundary-condition kind (Dirichlet, traction, do-nothing, slip and symmetry,
  periodic, axis, pressure point).
- **Closed-form solutions** (`analytic`): Couette and Poiseuille flows (reproduced to round-off where the solution is
  in the discrete space), circular Couette, Hagen-Poiseuille, Stokes drag on a sphere, Kovasznay and Taylor-Green
  flows, Burgers vortex, Womersley flow, Graetz problems, the Hartmann, Shercliff and Hunt ducts, rest states.
- **Discrete identities** (`conservation`): energy balances of the convective forms and of the transient scheme, heat
  balance of the Boussinesq solution, mass conservation, solver and quadrature sensitivity.
- **Other codes** (`cross_code`): NGSolve 6.2.2608 and DOLFINx 0.11.0 solving the same discrete problem (same meshes,
  elements, quadrature and boundary data): Stokes cavities (planar, axisymmetric with swirl, 3D), the Navier-Stokes
  cavity at Re = 1000, a transient cavity, natural convection at Ra = 1e5, stabilized P1/P1 cavities and MHD ducts.
  Agreement shows that the same discrete problem is solved; it says nothing about the adequacy of the mesh or model.

## What is not verified

- **No published benchmark value is compared.** Lid-driven cavity, flow past a cylinder, backward-facing step and
  natural-convection reference values were never extracted into cases, so the `literature` verdict is `NOT_TESTED`
  for every case.
- No comparison with experiments.
- The 3D Stokes cases (several MPI ranks each) and the four Womersley cases, whose case definitions allow 1 to 4 hours
  per case in full (`timeout_s`), and `mms_stokes_2d_periodic` were run in **quick mode only** in the last run: their
  orders of convergence and their fine-mesh comparisons are not confirmed by it (the quick rows show what the first
  two levels give). The periodic case did not finish in full because its prescribed-flow-rate variant is slow without
  STRUMPACK: the direct solve with the flow-rate constraint takes 1.5 s at N = 32 and 47 s at N = 64, about 30 times
  longer per refinement.
- Features the module does not offer (see the limitations in the guide) are not covered: open-boundary
  stabilization, turbulence models, 3D Navier-Stokes, transient Boussinesq or MHD, multi-rank solves other than 3D
  Stokes.

## Last run

Run on 2026-11-08 (axionFEM 0.1.0) in an MPI build (GCC 14, Release; MFEM 4.10.0 with METIS; no STRUMPACK, so the
direct solves use the Eigen fallback) on one 128-core CPU node. Every short case was run in full (`run --cost short`),
the long cases except the 3D (multi-rank) and Womersley ones in full, and every case in quick mode (`run --all
--quick`); the table shows the full result where there is one.

Columns: *evidence* is the kind of check (above); *run* is `full` (every refinement level) or `quick` (the first two
levels only: orders and Richardson extrapolation cannot be judged, so such a row is a smoke test and its numbers are
indicative); *result* is `all criteria met` (in quick mode `all criteria met (on two levels)` or `smoke only`) or the
failed acceptance items; *measured* gives the observed orders minus the expected ones over the case's error series,
the largest deviation of a comparison as a fraction of its tolerance (1 = at the tolerance), and the number of
identity and bound checks.

**Summary.** 67 of the 81 cases were run in full and 66 of them meet all their criteria; the observed orders agree
with the expected ones within each case's tolerance (661 observed orders in the full runs), and the cross-code
comparisons agree within their tolerances. The exception is `analytic_hartmann_2d`: every order, comparison and check
passes, but on its finest level the hydrodynamic run (Ha = 0) stops at a relative residual of 1.02e-12, just above the
Newton tolerance 1e-12, so the run is reported as not converged and the `solver` item fails. This is a tolerance at
the round-off floor, not a wrong solution; it is left open.

### Taylor-Hood Stokes, planar

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_annulus_slip_2d` | mms, analytic | long | full | all criteria met | 8 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.01 to 0.05; 1 comparison (closed_form), largest deviation 3.4e-05 of the allowance; 2 checks |
| `analytic_couette_2d` | analytic | short | full | all criteria met | 14 comparisons (closed_form), largest deviation 2.2e-04 of the allowance; 2 checks |
| `analytic_noflow_stokes_2d` | analytic | short | full | all criteria met | 6 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.03 to 0.03; 7 checks |
| `analytic_poiseuille_channel_2d` | analytic, conservation | short | full | all criteria met | 15 comparisons (closed_form), largest deviation 5.7e-05 of the allowance; 5 checks |
| `control_p1p1_stokes_2d` | mms | short | full | all criteria met | 7 checks |
| `mms_stokes_2d` | mms, conservation | long | full | all criteria met | 22 orders (expected 2.0, 3.0), observed minus expected -3.8e-03 to 0.06; 7 checks |
| `mms_stokes_2d_do_nothing` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -1.5e-03 to 0.02; 3 checks |
| `mms_stokes_2d_periodic` | mms | long | quick | smoke only | 12 orders (expected 2.0, 3.0), observed minus expected -0.02 to 0.23; 4 checks |
| `mms_stokes_2d_pressure_point` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -2.2e-04 to 0.06; 6 checks |
| `mms_stokes_2d_slip` | mms | long | full | all criteria met | 6 orders (expected 2.0, 3.0), observed minus expected -2.1e-03 to 0.06; 4 checks |
| `mms_stokes_2d_traction` | mms | long | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected -2.4e-03 to 0.02; 3 checks |
| `selfconv_stokes_cavity_reg_2d` | conservation | long | full | all criteria met | 4 Richardson/GCI (orders 3.94 to 3.95); 5 checks |
| `xcode_stokes_cavity_reg_2d` | cross_code | long | full | all criteria met | 8 comparisons (ngsolve), largest deviation 6.1e-04 of the allowance; 2 checks |

### Taylor-Hood Stokes, axisymmetric with swirl

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_circular_couette_axi` | analytic, conservation | long | full | all criteria met | 11 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.01 to -3.0e-03; 4 comparisons (closed_form), largest deviation 3.9e-05 of the allowance; 3 checks |
| `analytic_hagen_poiseuille_pipe` | analytic, conservation | short | full | all criteria met | 12 comparisons (closed_form), largest deviation 1.3e-05 of the allowance; 5 checks |
| `analytic_stokes_sphere_drag` | analytic, conservation | long | full | all criteria met | 21 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.05 to 3.8e-03; 4 comparisons (closed_form), largest deviation 0.15 of the allowance; 7 checks |
| `mms_stokes_axi` | mms | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected -1.7e-03 to 0.02; 8 checks |
| `mms_stokes_axi_axis` | mms, conservation | long | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -1.7e-03 to 0.02; 14 checks |
| `mms_stokes_axi_do_nothing` | mms | long | full | all criteria met | 8 orders (expected 2.0, 3.0), observed minus expected -3.7e-03 to 0.01; 4 checks |
| `mms_stokes_axi_traction` | mms | long | full | all criteria met | 18 orders (expected 2.0, 3.0), observed minus expected -5.3e-03 to 9.6e-03; 4 checks |
| `xcode_stokes_axi_cavity_swirl` | cross_code | long | full | all criteria met | 8 comparisons (ngsolve), largest deviation 2.7e-04 of the allowance; 2 checks |

### Taylor-Hood Stokes, 3D (MPI)

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_duct_poiseuille_3d` | analytic, conservation | long | quick | smoke only | 1 Richardson/GCI; 1 comparison (series), largest deviation 53.60 of the allowance; 17 checks |
| `analytic_stokes_sphere_3d` | analytic | long | quick | smoke only | 28 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.17 to 0.33; 7 Richardson/GCI; 6 comparisons (closed_form), largest deviation 86.17 of the allowance; 9 checks |
| `control_taylor_hood_3d` | conservation | long | quick | smoke only | 7 checks |
| `mms_stokes_3d` | mms, conservation | long | quick | smoke only | 24 orders (expected 2.0, 3.0), observed minus expected -0.08 to 0.06; 13 checks |
| `mms_stokes_3d_do_nothing` | mms | long | quick | smoke only | 6 orders (expected 2.0, 3.0), observed minus expected -2.4e-03 to 0.05; 9 checks |
| `mms_stokes_3d_slip` | mms, analytic | long | quick | smoke only | 6 orders (expected 2.0, 3.0), observed minus expected -0.02 to 0.04; 11 checks |
| `mms_stokes_3d_traction` | mms | long | quick | smoke only | 16 orders (expected 2.0, 3.0), observed minus expected -0.04 to 0.06; 9 checks |
| `rank_invariance_stokes_3d` | conservation | long | quick | all criteria met (on two levels) | 8 checks |
| `xcode_stokes_cavity_reg_3d` | cross_code | long | quick | all criteria met (on two levels) | 8 comparisons (dolfinx), largest deviation 1.1e-05 of the allowance; 4 checks |

### Steady Navier-Stokes

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_burgers_vortex_axi` | analytic | long | full | all criteria met | 15 orders (expected 2.0, 3.0), observed minus expected 5.7e-04 to 0.10; 4 checks |
| `analytic_circular_couette_ns_axi` | analytic | long | full | all criteria met | 6 orders (expected 2.0), observed minus expected 1.5e-03 to 1.9e-03; 3 checks |
| `analytic_kovasznay_re40` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -1.1e-03 to 0.03; 5 checks |
| `cons_ns_energy_identity` | conservation | long | full | all criteria met | 29 checks |
| `mms_ns_axi` | mms, conservation | long | full | all criteria met | 37 orders (expected 2.0, 3.0), observed minus expected 1.4e-03 to 0.12; 5 checks |
| `mms_ns_newton_jacobian` | mms | short | full | all criteria met | 24 checks |
| `mms_ns_steady_2d` | mms, conservation | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected 1.5e-03 to 0.13; 7 checks |
| `mms_ns_steady_2d_traction` | mms, conservation | long | full | all criteria met | 16 orders (expected 2.0, 3.0), observed minus expected 1.4e-03 to 0.13; 3 checks |
| `selfconv_ns_cavity_continuation` | conservation | long | full | all criteria met | 8 checks |
| `xcode_cavity_re1000` | cross_code | long | full | all criteria met | 24 comparisons (ngsolve), largest deviation 1.4e-04 of the allowance; 1 check |

### Transient Navier-Stokes

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_taylor_green_2d` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected 0.01 to 0.11; 2 checks |
| `analytic_womersley_2d_space` | analytic | long | quick | smoke only | 4 orders (expected 2.0, 3.0), observed minus expected -0.03 to -0.02; 1 check |
| `analytic_womersley_2d_time` | analytic | long | quick | smoke only | 12 orders (expected 1.0, 2.0), observed minus expected -0.37 to 0.08; 3 checks |
| `analytic_womersley_axi_space` | analytic | long | quick | smoke only | 6 orders (expected 2.0, 3.0), observed minus expected -0.04 to -0.02; 1 check |
| `analytic_womersley_axi_time` | analytic | long | quick | smoke only | 12 orders (expected 1.0, 2.0), observed minus expected -0.27 to 0.03; 3 checks |
| `cons_ns_transient_energy` | conservation | long | full | all criteria met | 48 checks |
| `mms_ns_transient_error_control` | mms | long | full | all criteria met | 16 checks |
| `mms_ns_transient_initial` | mms | long | full | all criteria met | 4 orders (expected 2.0, 3.0), observed minus expected -1.0e-03 to -9.8e-04; 3 checks |
| `mms_ns_transient_temporal` | mms | long | full | all criteria met | 96 orders (expected 1.0, 2.0), observed minus expected -0.08 to 9.6e-03; 2 checks |
| `xcode_ns_transient_cavity` | cross_code | long | full | all criteria met | 36 comparisons (ngsolve), largest deviation 2.9e-06 of the allowance; 1 check |

### Scalar transport and Boussinesq

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_advdiff_layer_2d` | analytic | long | full | all criteria met | 7 checks |
| `analytic_conduction_annulus_axi` | analytic | short | full | all criteria met | 4 orders (expected 2.0, 3.0, 4.0), observed minus expected -5.5e-03 to -2.9e-03; 1 check |
| `analytic_conduction_limit_cavity` | analytic, conservation | long | full | all criteria met | 7 checks |
| `analytic_graetz_pipe` | analytic | long | full | all criteria met | 15 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.05 to 2.5e-03; 3 checks |
| `analytic_graetz_planar` | analytic | long | full | all criteria met | 12 orders (expected 2.0, 3.0, 4.0), observed minus expected -0.06 to 0.02; 1 check |
| `analytic_hydrostatic_noflow` | analytic, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -1.9e-05 to 0.57; 17 checks |
| `cons_heat_balance` | conservation | long | full | all criteria met | 2 checks |
| `mms_advdiff_2d` | mms, conservation | long | full | all criteria met | 18 orders (expected 1.0, 2.0, 3.0), observed minus expected -0.03 to 0.04; 9 checks |
| `mms_advdiff_axi` | mms, conservation | long | full | all criteria met | 12 orders (expected 1.0, 2.0, 3.0), observed minus expected -3.4e-03 to 0.01; 10 checks |
| `mms_boussinesq_2d` | mms | long | full | all criteria met | 34 orders (expected 2.0, 3.0), observed minus expected -4.9e-03 to 0.12; 4 checks |
| `mms_boussinesq_axi` | mms | long | full | all criteria met | 24 orders (expected 2.0, 3.0), observed minus expected -3.5e-04 to 0.05; 4 checks |
| `mms_boussinesq_newton_jacobian` | mms | short | full | all criteria met | 16 checks |
| `xcode_natconv_ra1e5` | cross_code | long | full | all criteria met | 28 comparisons (ngsolve), largest deviation 5.1e-06 of the allowance |

### Stabilized P1/P1 (GLS)

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_gls_kovasznay_re40` | analytic, conservation | long | full | all criteria met | 8 orders (expected 1.0, 1.8, 1.9), observed minus expected -0.03 to 0.03; 2 checks |
| `control_gls_no_pspg` | mms | long | full | all criteria met | 14 checks |
| `gls_residual_jacobian` | mms | short | full | all criteria met | 12 checks |
| `gls_tau_pins` | mms | short | full | all criteria met | 21 checks |
| `mms_gls_ns_2d` | mms, conservation | long | full | all criteria met | 32 orders (expected 1.0, 1.5, 1.9, 2.0), observed minus expected -0.10 to 0.08; 2 checks |
| `mms_gls_stokes_2d` | mms, conservation | long | full | all criteria met | 20 orders (expected 1.0, 1.5, 2.0), observed minus expected -0.02 to 0.06; 2 checks |
| `mms_gls_transient_small_dt` | mms | long | full | all criteria met | 8 checks |
| `patch_gls_consistency` | mms | short | full | all criteria met | 6 checks |
| `xcode_gls_cavity` | cross_code | long | full | all criteria met | 48 comparisons (ngsolve), largest deviation 8.0e-08 of the allowance; 1 check |
| `xcode_gls_vs_taylor_hood` | cross_code | long | full | all criteria met | 30 comparisons (ngsolve_th), largest deviation 0.68 of the allowance; 10 checks |

### Inductionless MHD

| case | evidence | cost | run | result | measured |
|---|---|---|---|---|---|
| `analytic_hartmann_2d` | analytic, conservation | long | full | FAILED: solver | 22 orders (expected 2.0, 3.0), observed minus expected -0.05 to -1.7e-03; 11 comparisons (closed_form), largest deviation 0.07 of the allowance; 11 checks |
| `analytic_hartmann_suction_2d` | analytic | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -0.02 to -4.7e-03; 4 comparisons (closed_form), largest deviation 0.02 of the allowance; 8 checks |
| `analytic_hunt_duct_2d` | analytic, conservation | long | full | all criteria met | 15 comparisons (closed_form), largest deviation 0.07 of the allowance; 10 checks |
| `analytic_mhd_annulus_variable_b` | analytic | long | full | all criteria met | 21 orders (expected 2.0, 3.0), observed minus expected -0.03 to 8.8e-03; 10 comparisons (closed_form), largest deviation 0.04 of the allowance; 9 checks |
| `analytic_shercliff_duct_2d` | analytic, conservation | long | full | all criteria met | 11 comparisons (closed_form), largest deviation 0.03 of the allowance; 12 checks |
| `mms_mhd_2d` | mms, conservation | long | full | all criteria met | 21 orders (expected 2.0, 3.0), observed minus expected -4.8e-03 to 0.13; 5 checks |
| `mms_mhd_2d_insulating` | mms, conservation | long | full | all criteria met | 12 orders (expected 2.0, 3.0), observed minus expected -8.6e-03 to 5.4e-04; 6 checks |
| `xcode_mhd_duct_2d` | cross_code | long | full | all criteria met | 14 comparisons (ngsolve), largest deviation 0.01 of the allowance |

81 cases: 67 run in full (66 meet all their criteria), 14 run in quick mode only, 0 not run. The full runs contain 661 observed orders of convergence, 4 Richardson/GCI estimates, 297 comparisons with reference values and 481 checks.

## Reference sources

| source | what it is |
|---|---|
| `closed_form` | exact solution evaluated in double precision |
| `series` | separated-variable series of the rectangular duct in high-precision decimal arithmetic (`tools/fluids_mms/duct_series_reference.py`) |
| `ngsolve` | NGSolve 6.2.2608 on the same discrete problem (scripts and outputs in `tools/fluids_mms/xcode/`) |
| `ngsolve_th` | NGSolve 6.2.2608, Taylor-Hood P2/P1 on the same meshes (the GLS cases compare against it with a tolerance, not to round-off; script `tools/fluids_mms/xcode/ngsolve_th_cavity_reference.py`) |
| `dolfinx` | DOLFINx 0.11.0 with PETSc 3.25.6 on the same discrete problem |

Regenerate the tables with `python benchmarks/fluids/run.py report DIR [DIR ...] --track fluids` from the `--out`
directories of `run` (a full result is preferred to a quick one).
