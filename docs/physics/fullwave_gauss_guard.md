# Gauss-law guard on the ungauged full-wave paths (default)

Status: **default on the ungauged paths (AMS-GMRES and plain STRUMPACK), beta.**
- Validation: `benchmarks/gauss_law_guard.py`, record `benchmarks/records/gauss_law_guard.json` (Validation, below).
- Code: `FwGaussGuard` and its call sites in `cpp/src/model3d_fullwave.cpp`.

Without it, and without validation loads, the ungauged paths accept 10 Hz to 1 kHz on the two-port frame
(`benchmarks/designs/stage1_twoport.json`) with no warning, while their fields are wrong by up to a factor 17 (energy
norm; ×14 in the l2 norm of the DOF vector; run 6 of [fullwave_tree_cotree.md](fullwave_tree_cotree.md)).

## Enabling it

```bash
export AXF_FW_GAUSS_GUARD=correct    # default on the ungauged paths: estimate, warn above 1e-3, correct
export AXF_FW_GAUSS_GUARD=estimate   # estimate and warn only; the fields are left as solved
export AXF_FW_GAUSS_GUARD=off        # neither (the default, and the only value, with AXF_FW_GAUGE=tree-cotree)
```

- Any other value is an error (empty means unset).
- With `AXF_FW_GAUGE=tree-cotree`, anything but `off` is an error: the gauged path solves the Gauss law exactly (with
  its own charge projection). Scripts that switch between the paths in one process must therefore reset the
  variable, as `benchmarks/gauss_law_guard.py` does.

## Why

At low frequency the summed operator A = K − ω²M fixes the gradient (capacitive) part of E only through ω²M. The
forward error of that part grows like (k h)^-2 times the residual, so a solve that passes the 1e-6 true-residual gate
can still be wrong there by many times the field.

Which outputs that reaches depends on the port:
- **A gap port on one conductor** (the two-port frame) has no charged Gauss rows. Its Z = −JᵀE/I² does not see the
  gradient error (run 6: ≤ 5.4e-11 of the gauged path), but the energies, probes, exports and scenes do.
- **A port between distinct conductors** (a capacitive port) changes Z at first order: Z changes by −(G_cᵀJ)·δφ/I².
  On a capacitor test model (plain STRUMPACK, Im Z of that port) the error is 4.7e-4 at 1 kHz, 1.0e-5 at 10 kHz and
  3.8e-8 at 100 kHz uncorrected, and ≤ 6.4e-15 corrected. Those runs are refused at the residual gate anyway; Z
  comes from the rejected rows.

Because K G = 0 exactly, the gradient part of the field obeys a scalar equation without K, the discrete Gauss law:

    G_c^T (b + ω² M E) = 0      (one row per free H1 node and one per floating conductor)

Its residual can be evaluated stably from M alone.

## Method

- **The potential space.** G_c φ, with φ in H1 of the field's order (the de Rham partner of the Nédélec space):
  - zero on the grounded PEC set;
  - constant on each other connected PEC set (one potential per floating conductor).

  The PEC closure is the set of H1 nodes of the elements and boundary faces that the essential DOFs close (the same
  elements and boundary attributes as `essential()`). Its connected sets come from a local union-find per closed
  element and face; the smallest global node number of each set is then exchanged over shared nodes until no label
  changes (fully distributed). The grounded set is the one with the most nodes, normally the outer wall. Setup
  checks that G_c φ vanishes on every essential ND DOF (`essential_edge_leak_relative`, ≤ 1e-12), else the guard is
  not applied.
- **The correction.** E = x + G_c δφ with −ω² S δφ = g, where:
  - S = G_c^T M G_c;
  - g = q + ω² G_c^T M x;
  - q = G_c^T b, with every row that is rounding of its own terms (|q| ≤ 1e-12 (|G_c|^T |b|)) set to zero.

  This projection is nearly the gauged path's default `AXF_FW_TC_CHARGE_PROJECTION=2`: a floating set's scale also
  counts free chord edges with both ends on the set, so its threshold is slightly looser. Without the projection, a
  port feed's rounding charge would put an f^-2 field into the corrected solution. Real charge (terminal conductors,
  truncated sources) is kept.
- **What the correction removes.** Split the error x − E* M-orthogonally into e_⊥ + G_c ψ (G_c^T M e_⊥ = 0). Then
  g = ω² S ψ and δφ = −ψ: the correction removes the whole gradient part of the error and leaves e_⊥, whose size is
  set by the curl block's conditioning, not by (k h)^-2. This holds, and the correction never increases the error's
  M-norm, under three conditions:
  - the S-solve is exact (it is checked: below);
  - the target is the projected Gauss law (intended; the gauged path's semantics);
  - no load column has charged rows. With charged loads, T = −ω²S + WCWᵀ makes the correction oblique,
    δφ = −ψ − T⁻¹WCUᵀe_⊥.
- **The estimate.** `field_fraction` is the M-norm of G_c δφ relative to that of the corrected field x + G_c δφ,
  computed from x + G_c δφ itself. It is the solve's gradient error relative to the field.
  `field_fraction_of_solved`, relative to x, saturates at 1 when the error dominates. Above 1e-3 the run carries a
  warning: removed (default), or not removed (`estimate`).
- **Solves.**
  - S on the free nodes (the closure eliminated): BoomerAMG-PCG at relative tolerance 1e-13.
  - The floating potentials: their Schur complement, with one PCG per floating conductor at setup and a dense
    n_c × n_c LDLT. At most 64 floating conductors; the Y_k and s_k vectors cost 2·n_c H1 vectors per rank.
  - A singular floating block (a field region with no grounded PEC set) leaves the guard off.
  - Without loss, S does not depend on frequency, so the setup is once per sweep, and each correction costs one PCG
    per real and imaginary part. With loss, see below.
  - HyprePCG returns at its iteration cap without an error. So every solve's achieved residual is computed, as a
    backward error in the Jacobi-weighted norm (diag(S)^(-1/2)): ‖S y − f‖ / (‖|S||y|‖ + ‖f‖). Above 1e-10, the setup
    leaves the guard off, and a correction is not applied and warns.
  - That measure matters under a large coefficient jump. In a resistive block (σ = 100 S/m: σ/(ωε0) = 1.8e10 at
    100 Hz) the potential is nearly constant on the block, S y cancels there, and the residual against f alone stalls
    near ε_mach·σ/(ωε0) (3e-7 to 1.8e-6 observed). Against the rows' own terms the same solves are backward stable,
    ≤ 1e-14 (part G). Backward stability is not forward accuracy near a good conductor: see "The floor near
    conductors", below.
  - Up to 4 iterative-refinement passes re-solve for any remaining backward error above 1e-12
    (`pcg_refinements_max`). A refinement changes the field; none ran in parts A-I.
  - A non-finite estimate is treated the same way.
  - When the first pass removed more than the field itself (`field_fraction` > 1), a second pass follows
    (`second_pass`). The PCG tolerance is relative to g, which grows with the error.
- **Lossy materials.** With dielectric loss (tan δ) or conduction (σ), A = K − ω²(M + i M_i) with
  M_i the mass of Im ε_c = ε0 εr tan δ + σ/ω, assembled per frequency from the same coefficient as the operator.
  - The Gauss law and the correction then use M_c = M + i M_i, and S_c = G_c^T M_c G_c is complex symmetric.
  - S_c δφ = g is solved by right-preconditioned, flexible, restarted GMRES (restart 30, at most 600 iterations) on
    its real 2×2 block form. The preconditioner is the real solver above (AMG-PCG and the floating Schur block),
    rebuilt per frequency on the SPD S_p = G_c^T (M + M_i) G_c and applied to each part. A correction costs two PCG
    solves per GMRES iteration.
  - The numerical range of S_p⁻¹ S_c (the quotients vᴴS_c v / vᴴS_p v, for every v) lies on the straight segment
    (1 + i t)/(1 + t), t ≥ 0, from 1 to i, so |λ| ≥ 1/√2 for any loss ratio. The iteration count is bounded but not
    constant: 4-7 GMRES iterations when the loss ratio is ≪ 1 or ≫ 1 everywhere (tan δ = 1e-2; σ ≤ 1e6 S/m at
    100 Hz), 20-30 at intermediate ratios (the lossy capacitor gap, part I) or extreme contrast (copper, part H, where
    BoomerAMG also needs more inner iterations).
  - GMRES runs in the Jacobi-weighted inner product. After each restart it takes the same backward error, with a
    bound on |S_c||y| (|G_c|ᵀ(|M| + |M_i|)|G_c|(|y_r| + |y_i|), which overestimates the rows' terms by a small factor,
    so the reported backward error is somewhat optimistic). It stops at 1e-12, or when the residual has failed to halve
    over two restart cycles (over one, once the backward error is below 1e-11). That backward error decides
    acceptance like the PCG's (`pcg_relative_residual_max`, `gmres_iterations`). The inner PCG solves (a flexible
    GMRES does not need them exact) are recorded apart (`gmres_inner_pcg_residual_max`) and do not gate.
  - Validation loads use the complex Woodbury form.
  - The energy norms of the estimate use M (ε′); the curl representative projects with M + M_i.
  - With loss the split of the error is oblique (G_c^T M_c e_⊥ = 0). The correction still removes the whole gradient
    part, but in the M + M_i norm the projector is bounded by √2, not 1 (continuity 1, coercivity 1/√2 from the
    numerical range above).
  - Without loss, the two real S-solves are those of the lossless path; the acceptance decision uses the backward
    error.
- **The floor near conductors.** Where Im ε_c ≫ Re ε_c (a metal with σ/(ωε0) = κ), the rows
  of S_c carry O(σ/ω) terms that cancel exactly for a potential constant on the region. That island mode is held only
  by the O(ε0) coupling through the region's boundary, so rounding sets its error at up to about ε_mach·κ times the
  field carried there, and the backward error cannot see it. The skin effect and charge relaxation shrink the realized
  error: on the part-G frame with a copper block at 100 Hz (κ = 1.0e16) the loaded-field closure is 3.4e-6 corrected
  (4.5e-3 uncorrected), with the excess in 0.2 % of the DOFs next to the block; at 10 kHz it is 5.7e-10.
  - What it reaches: field values near the conductor (exports, probes, scenes) and the loaded-field closure. Not Z of
    gap ports, U_e, U_m, R_bulk or Ψ: an island constant has no gradient inside the block, and its first-order effect
    on U_e is the island's ε-flux, which vanishes for a source-free block. At copper those agree between 3 and 4 ranks
    to ≤ 1.8e-9.
  - What is reported: per frequency, `loss_contrast` = max over the free elements of Im ε_c / Re ε_c. After every
    applied lossy correction, an estimate-only pass on the corrected field gives `floor_estimate`, a fresh rounding
    realization of what is left; above 1e-8 the run warns that field values near conductive regions carry an error of
    about that size.
- **Validation loads** (`validation_shunt_R_ohm`, `validation_loads`) add U diag(c) U^T to A. Their Gauss rows
  W = G_c^T U, projected like q, enter the residual and, by the Woodbury identity, the correction. A gap port's load
  column projects to zero; a charged port's does not (Validation, part E).
- **Where.**
  - Every port solve and both validation solves (`direct_source`, `direct_loaded`), after the true-residual gate's
    residual is taken and before any output uses the field. The gates are unchanged.
  - The residual kept for the reciprocity identity is updated as r + A G_c δφ = r − ω² M_c G_c δφ (+ the loads):
    exact algebra, with no K applied to a gradient.
- **Magnetic energy and probe curls.** With the guard on, these come from the curl representative E⊥ = E − G_c S⁻¹
  G_c^T M E (`energy_magnetic_scope`), as the gauged path uses its cotree representative. K E⊥ = K E exactly, without
  K's rounding on a large gradient part. Without it, the ungauged U_m of a capacitive port sits at that floor:
  0.3-0.44 J/A² against 7.1e-9.

## Scope (enforced, recorded and warned)

The guard applies only where the assembled operator equals K − ω²(M + i M_i) on the free DOFs: closed
(`fullwave.boundary = "pec"`), no Robin (impedance, sheet, scattering) boundary, no PML. Lossy materials are covered
(M_i ≠ 0). Each frequency checks this in three ways:
- the real block on a random vector (`operator_consistency_relative` < 1e-11). This sees K but not the mass block at
  low frequency, where ω²|M v|/|K v| ~ (k h)²;
- the mass coefficients directly: every element the essential DOFs do not close must have the permittivity energy_M
  was built with, and, in a lossless model, no imaginary part (`mass_coefficient_mismatch` = 0);
- with loss, the imaginary block against −ω² M_i on the same vector (`loss_mass_consistency_relative` < 1e-11). That
  block is a pure mass, so this check holds at any frequency.

Elsewhere the run records why (`fullwave.gauss_law_guard.not_applied`, or the frequency row's
`gauss_law_guard.not_applied`). At quasi-static frequencies (k R < 1e-2) it also warns once after the sweep, naming
the reason and the outputs at risk. The cases:
- radiation, Robin or PML: no closed K and M. The exterior's DtN, the Robin boundary mass and the stretched PML mass
  all enter the Gauss law; not implemented;
- more than 64 floating conductors, a singular floating block, a PEC closure inconsistent with the essential DOFs, or
  a setup PCG that missed its tolerance (with loss, also at a frequency's preconditioner rebuild).

Topology: the potential space spans the curl kernel when the PEC closure carries all of it. That holds for the
generated domains (a ball, with any number of floating conductors, genus-1 ones included: part D). On an imported
mesh whose handles have PMC (natural) boundary parts, harmonic fields can exist that the guard neither estimates nor
corrects. Such runs carry a warning; the gauged path refuses PMC. Surface (lumped) ports are guarded: their real
side-edge charge is kept, and the corrected field includes its 1/ω field. That is right for the discrete problem but
untested.

What it does not fix: the true-residual gate itself, and the curl part's error, which is not estimated.

## Results JSON

- `fullwave.gauss_law_guard` contains:
  - `mode`;
  - `setup`: H1 order and size, label passes, PEC sets, grounded and floating set node counts,
    `essential_edge_leak_relative`, `schur_eigenvalue_ratio`, setup PCG iterations and achieved residual, seconds;
  - or `not_applied`;
  - and `harmonic_fields_note` with PMC groups.
- Per frequency, `gauss_law_guard`: `mode`, `applied`, `operator_consistency_relative`, `mass_coefficient_mismatch`,
  `lossy`, or `not_applied`. With loss also `loss_mass_consistency_relative`, `loss_contrast`, and
  `loss_preconditioner` (the rebuilt S_p: `schur_eigenvalue_ratio`, `setup_pcg_iterations_max`,
  `setup_pcg_relative_residual_max`, `setup_pcg_refinements_max`).
- Per solve (`port_solutions[i].solver.gauss_law_guard`, `direct_source_solver.gauss_law_guard`,
  `direct_loaded_solver.gauss_law_guard`):
  - `applied` (or `not_applied`);
  - `field_fraction`, `field_fraction_of_solved`;
  - `gauss_residual_before` and `gauss_residual_after`: ‖g‖ against the field's own Gauss terms ‖ω²|G_c|^T|M x|‖,
    before and after;
  - `gauss_residual_reduction` = ‖g_after‖/‖g_before‖, what the S-solve achieved. Under a large loss contrast the
    normalization is dominated by the conductor's O(σ/ω) rows, so these three lose meaning there (part G at 100 Hz:
    `gauss_residual_before` 1.3e-8 for a 3.5 % gradient error; the reduction bottoms out at 1e-8 for σ = 100 S/m and
    1e-4 for copper). For dielectric loss they stay informative (part F);
  - `pcg_iterations_max`, `pcg_refinements_max`;
  - `pcg_relative_residual_max`: the largest backward error of the solve's PCG and GMRES solves (the acceptance
    quantity);
  - `lossy`; with loss also `gmres_iterations` (summed over the complex solves), `gmres_inner_pcg_residual_max`, and,
    on an applied correction, `floor_estimate`;
  - `rounding_charge_rows_zeroed` and `charged_rows`, summed over the real and imaginary parts of b and every load
    column;
  - `floating_conductors`, `loads`, `seconds`;
  - `second_pass`, when one ran.

## Validation

`benchmarks/gauss_law_guard.py`, record `benchmarks/records/gauss_law_guard.json`, all parts from one build on
4 MPI ranks.

Parts A-C use the two-port frame (`benchmarks/designs/stage1_twoport.json`, lowest order, 109k DOFs). The guard finds
two PEC sets there: the grounded wall (16,855 nodes) and the floating frame.

The reference is the gauged path, exact there to the f² law with its charge projection
([fullwave_tree_cotree.md](fullwave_tree_cotree.md), run 1). "estimate" is the guard reporting only (the fields as
solved); "correct" is the default.

**A. Fields** (the run-6 design: no validation loads, no source check). Field error = relative l2 norm of the
exported DOF vector against the gauged path (same mesh and partition); estimate = `field_fraction`, ports 0 / 1.

| path | f | estimate | error, estimate mode | error, corrected |
|---|---|---|---|---|
| AMS | 10 Hz | 12.7 / 16.6 | 10.8 / 14.0 | 1.0e-10 / 5.9e-11 |
| AMS | 100 Hz | 3.8e-2 / 2.5e-2 | 3.5e-2 / 2.4e-2 | 1.4e-11 / 8.9e-12 |
| AMS | 1 kHz | 4.1e-4 / 2.7e-4 | 3.9e-4 / 2.2e-4 | 6.9e-12 / 4.1e-12 |
| AMS | 100 kHz | 5.4e-8 / 3.9e-8 | 6.2e-8 / 2.8e-8 | 7.1e-12 / 4.1e-12 |
| plain STRUMPACK | 100 Hz | 5.7e-2 / 3.7e-2 | 6.8e-2 / 3.8e-2 | 2.7e-12 / 1.8e-12 |
| plain STRUMPACK | 1 kHz | 5.2e-4 / 3.5e-4 | 5.8e-4 / 2.4e-4 | 9.3e-13 / 5.4e-14 |
| plain STRUMPACK | 100 kHz | 3.8e-8 / 2.8e-8 | 3.7e-8 / 3.2e-8 | 4.5e-14 / 2.6e-14 |

- **Estimate accuracy.** The estimate is an energy norm. Against the energy-norm error of the uncorrected field,
  √(U_e/U_e,gauged − 1), it agrees to 4-5 digits from 10 Hz to 1 kHz on both paths. Against the l2 error of the DOF
  vector above, it agrees within a factor 0.84-1.49.
- **Corrected outputs.** U_e agrees with the gauged path to ≤ 9e-11. Z agrees to ≤ 5.5e-11, the same with or without
  the correction (a gap port).
- **Plain STRUMPACK at 10 Hz** still fails to factorize (as in run 6), so there is no field to correct.
- **AMS at 10 Hz depends on the rank count.** On 8 ranks the guard ran, then the port's true-residual gate (1.5e-5)
  refused it; on 4 ranks the residual is 1.1e-8 (as in run 6).
- **Below 10 Hz** (2, 3 and 5 Hz) AMS misses the residual gate (5-7e-5). The guard still removes a
  gradient error 40-74 times the field, and leaves the field within 3.6-7.5e-7 of the gauged path, set by the curl
  part.

**B. Validation gates** (the run-1 design: validation shunts and the source check). The loaded-field closure compares
the directly loaded solve with its reconstruction from the port fields (gate 1e-5):

| path | f | estimate mode | corrected |
|---|---|---|---|
| AMS | 100 Hz | 1.6e-2, refused | 1.3e-11, accepted |
| AMS | 1 kHz | 1.2e-4, refused | 6.6e-12, accepted |
| plain STRUMPACK | 100 Hz | 9.2e-3, refused | 2.2e-12, accepted |
| plain STRUMPACK | 1 kHz | 3.4e-7, accepted | 8.3e-13, accepted |

- The corrected closure is comparable to the loaded voltages' own closure (7.9e-12, 4.5e-12, 2.2e-12, 8.3e-13).
- Without projecting the load columns, both corrected paths stop at 1.5e-8 (100 Hz) and 1.5e-9 (1 kHz).
- Source reciprocity is not changed by the guard (AMS 7.7e-9 and 5.1e-9 either way).

**C. Order 2** (the run-6 design on an order-2 mesh, AMS; the gauged path is lowest order only).

| f | estimate (ports 0 / 1) | U_e/f² vs 100 kHz, uncorrected | corrected |
|---|---|---|---|
| 10 Hz | refused at the residual gate (true residual 1.0) | | |
| 100 Hz | 3.9 / 5.4 | +15 / +29 | −3.868e-6 / −3.867e-6 |
| 1 kHz | 8.6e-3 / 9.6e-3 | +7.1e-5 / +8.9e-5 | −3.867e-6 / −3.867e-6 |
| 10 kHz | 1.2e-4 / 9.7e-5 | −3.82e-6 / −3.82e-6 | −3.829e-6 / −3.829e-6 |
| 100 kHz | 8.8e-7 / 6.8e-7 | 0 | 0 |

- Corrected, U_e/f² is the same from 100 Hz to 10 kHz to 4e-8. Its −3.87e-6 against 100 kHz is the physical (kR)²
  dispersion, which the nearly exact 10 kHz uncorrected run shows too.
- At order 2 the ungauged gradient errors are larger than at order 1 (×4-5 at 100 Hz, 1 % at 1 kHz, 9e-7 at 100 kHz).
  So the default changes order-2 results by about 1e-6 at 100 kHz, against about 5e-8 at order 1.

**D. Several floating conductors** (the tree-cotree run-4 topology design: the frame, two floating spheres, a
floating genus-1 torus, a 2 mm plate and a rod on the wall, with validation shunts and the source check). The guard
finds six PEC sets: the grounded wall and five floating ones, with a 5×5 Schur block (eigenvalue ratio 0.18).
Field error against the gauged path (l2, ports 0 / 1), and the loaded-field closure:

| path | f | estimate | field error, corrected | loaded-field closure |
|---|---|---|---|---|
| AMS, estimate mode | 100 Hz | 8.1e-2 / 7.1e-2 | | 0.137, refused |
| AMS, estimate mode | 1 kHz | 1.1e-2 / 1.1e-2 | | 1.9e-4, refused |
| AMS | 100 Hz | 8.1e-2 / 7.1e-2 | 4.6e-12 / 4.9e-12 | 5.4e-12, accepted |
| AMS | 1 kHz | 1.1e-2 / 1.1e-2 | 1.5e-11 / 1.4e-11 | 1.6e-11, accepted |
| plain STRUMPACK | 100 Hz | 1.08 / 0.90 (second pass) | 1.8e-13 / 7.7e-13 | 6.2e-14, accepted |
| plain STRUMPACK | 1 kHz | 8.5e-4 / 6.0e-4 | 2.1e-13 / 6.8e-14 | 1.4e-12, accepted |

- Corrected Z and U_m agree with the gauged path to ≤ 1.5e-12 and ≤ 3e-12.
- On 3 ranks the guard finds the same six sets and Schur spectrum, and the same corrected outputs (to 3e-12).

**E. Charged ports and loads** (the run-2 capacitor, straight, s = 0.01, with shunts on all three ports). Port 2
bridges a floating sphere and its floating shell. Its feed keeps two charged conductor rows, and so does its load
column: the Woodbury term runs with W ≠ 0 (`charged_rows` 2 on the loaded solve). Plain STRUMPACK against the
gauged path:

| f | Z₂₂, uncorrected | Z₂₂, corrected | port-2 field, uncorrected | port-2 field, corrected | loaded-field closure |
|---|---|---|---|---|---|
| 1 MHz | 8.2e-10 | 6.1e-15 | 2.9e-8 | 7.0e-15 | 2.0e-10 → 2.0e-14 |
| 10 MHz | 3.1e-12 | 8.6e-15 | 2.9e-10 | 9.5e-15 | 3.4e-12 → 1.7e-14 |

- Ports 0 and 1 (gap ports on the frame) and the near-null couplings to port 2 are unchanged by the correction.
- AMS is refused at 1 MHz at its own residual gate (port 2), as for this model below 100 kHz. At
  10 MHz it is accepted, and its corrected fields agree with the gauged path to ≤ 1.3e-11.
- At these frequencies the error being removed is already 1e-8 to 1e-12 of the field, so `gauss_residual_reduction`
  is only 1e-5 to 1e-7: the "after" residual sits at rounding. The reduction is informative when the error is large
  (100 Hz: 3e-14 to 5e-14). Below 100 kHz, where the capacitor's gradient error is large, the ungauged paths refuse
  this model at their residual gates or factorization.

**F. A lossy dielectric** (the run-1 design with a slab beside the frame, εr = 2, tan δ = 1e-2, validation shunts
and the source check; AMS). There is no gauged reference: the
tree-cotree path needs a real operator. So the checks are the quasi-static laws against the corrected 100 kHz run,
and the loaded-field closure.

| f | U_e/f² vs 100 kHz | R_bulk/f³ vs 100 kHz | loaded-field closure |
|---|---|---|---|
| 100 Hz, estimate mode | +1.5e-3 / +6.6e-4 | −3.4e-2 | 2.3e-2, refused |
| 100 Hz, corrected | −5.4e-6 / −5.4e-6 | −6.9e-6 | 7.6e-12, accepted |
| 1 kHz, corrected | −5.4e-6 / −5.4e-6 | −6.9e-6 | 9.3e-12, accepted |

- The −5.4e-6 and −6.9e-6 are the physical (kR)² dispersion against the 100 kHz values.
- Uncorrected, the dielectric loss itself (R_bulk, so Q_bulk) is 3.4 % off at 100 Hz: the gradient error reaches
  the loss through ε″|E|².
- 5-6 GMRES iterations per complex solve (the source solve takes 6). The imaginary-block check against −ω²M_i
  agrees to 2e-16.
- 10 Hz is refused on AMS at its own residual gate, corrected or not.

**G. A resistive block** (σ = 100 S/m, so σ/(ωε0) = 1.8e10 at 100 Hz; validation shunts):
- 100 Hz: loaded-field closure 5.5e-3 uncorrected (refused), 5.9e-11 corrected (accepted).
- 10 kHz: 5.4e-7 uncorrected and 6.8e-12 corrected.
- 4-6 GMRES iterations, backward errors ≤ 7e-17 (the outer GMRES; the inner PCG solves ≤ 2.2e-14), no refinements.
- `floor_estimate` 1.7e-12 to 9.7e-11 at 100 Hz, 2.6e-13 to 9.7e-13 at 10 kHz: no warning.
- A residual measured relative to f would read 3e-7 to 1.8e-6 here (about ε_mach·σ/(ωε0)), and the guard would decline
  to apply; this case is why acceptance uses the backward error.

**H. A copper block** (the part-G design with σ = 5.8e7 S/m: σ/(ωε0) = 1.0e16 at 100 Hz):

| f | loaded-field closure, estimate mode | corrected | `floor_estimate` | GMRES (inner PCG) iterations |
|---|---|---|---|---|
| 100 Hz | refused | 3.4e-6, accepted, warned | 5.3e-7 to 4.4e-6 | 22-26 (37-89) |
| 10 kHz | 7.6e-7 | 5.7e-10 | 9.5e-11 to 7.5e-10 | 4-6 (14) |

- At 100 Hz the floor indicator matches the closure's excess (3.4e-6), and each of the four solves warns that field
  values near conductive regions carry that error. Backward errors ≤ 3e-16 throughout: the measure that decides
  acceptance cannot see it (Method, "The floor near conductors").
- At 10 kHz the skin effect clears the block's interior and the floor falls to 1e-9, below the warning.
- The 3.4e-6 is the 4-rank value (3.8e-6 on 3 ranks, with Z, U_e, U_m, R_bulk and Ψ unchanged to ≤ 1.8e-9 between
  them).

**I. A lossy capacitor gap** (the part-E capacitor, its gap σ = 1e-3 S/m: loss ratio 18 at 1 MHz and 1.8 at 10 MHz;
AMS):

| f | loaded-field closure, estimate mode | corrected | GMRES iterations per complex solve |
|---|---|---|---|
| 1 MHz | 8.7e-8 | 3.3e-11 | 22-28 |
| 10 MHz | 1.2e-9 | 1.0e-10 | 30 |

- The loaded solve keeps two charged rows (`charged_rows` 2), so the complex Woodbury term runs with W ≠ 0; the
  reported `gmres_iterations` (47 and 60) sum its two nonzero complex solves.
- The loaded state is driven from the frame's port and barely excites the screened capacitor (about 1e-12 V on its
  load against 2.5 V on the frame's ports), so this is weak evidence for the Woodbury term itself.
- Backward errors ≤ 6e-13, `floor_estimate` ≤ 4e-15.

No refinement pass ran in parts F-I.

Tests: `tests/test_fullwave3d_gauss_guard.py`:
- the 100 Hz estimate against the U_e excess;
- the corrected U_e, U_m and Z against the gauged path, and the PCG residual and Gauss-residual reduction;
- the validation shunts;
- a lossy dielectric: the quasi-static laws of U_e and R_bulk at 100 Hz, corrected and not;
- a conducting block (part G at 100 Hz): refused uncorrected, the corrected closure, `loss_contrast` and
  `floor_estimate`;
- an open (radiating) model's record and warning;
- the option checks.

## Limitations

- Not covered by the validation: a case whose loaded state strongly excites a charged load (part I's loaded voltage
  on the capacitor is about 1e-12 V), and a failing lossy preconditioner or GMRES.
- Treating strongly conducting regions like floating conductors, which would remove the floor near conductors, is
  not implemented; the floor is estimated and warned about instead.
- The guard does not apply to open (radiating) models, Robin boundaries or PML (Scope, above).
