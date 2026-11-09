# Tree-cotree gauged low-frequency full-wave solve (opt-in)

Status: **opt-in, beta.** Cross-checked against the default paths on the two-port frame
(`benchmarks/designs/stage1_twoport.json`; [validation/tree_cotree.md](../validation/tree_cotree.md)), and against a
capacitor's closed form (`benchmarks/tree_cotree_capacitor.py`). With its default charge projection it meets the
low-frequency energy test (U_e = f²c(1+βf²), below) down to 10 Hz; without it, it does not (Validation, below).
Code: `FwTreeCotreeDirect` and its call sites in `cpp/src/model3d_fullwave.cpp`.

## Enabling it

```bash
export AXF_FW_SOLVER=strumpack        # required: the gauge is only implemented on the direct path
export AXF_FW_GAUGE=tree-cotree       # default: none (unchanged behaviour)
export AXF_FW_REFINE_MAX=8            # optional: maximum refinement steps (default 8)
export AXF_FW_TC_CHARGE_PROJECTION=2  # default 2: zero the Gauss rows of P^T b and of the load columns whose
                                      # charge is rounding of their own terms, at free vertices and conductors;
                                      # 1: free vertices only; 0: none (the 3.5 % field error at 10 Hz, below)
export AXF_FW_TC_REFINE=normwise      # experimental: componentwise refines on the componentwise backward error
                                      # (no effect on any result measured)
export AXF_FW_TC_RESIDUAL=extended    # default: the refinement residual in long double, from K, M and G_c;
                                      # double: a double-precision residual (run 4's 1e-9 in Psi)
```

Any other value of these variables is an error (empty means unset). `AXF_FW_GAUGE=tree-cotree` without
`AXF_FW_SOLVER=strumpack` is an error, and so is `AXF_FW_TC_CHARGE_PROJECTION` other than 0 or `AXF_FW_TC_REFINE`
other than normwise without the gauge; `solver_path` names both. The STRUMPACK path itself needs a STRUMPACK-enabled
build and `fullwave.boundary = "pec"`.

## Why

At low frequency (k h << 1) the 3D full-wave field is dominated by its gradient (capacitive) part. The
summed real operator A = K - w^2 M keeps only about log10((k h)^2 / eps) digits of the mass term, and A E
cancels in floating point because K G = 0 holds only in exact arithmetic. Solvers working with the summed matrix
can still reach small residuals (on the two-port frame at 100 Hz, about 1e-12; there is no residual floor), but
their forward error grows with the conditioning, of order (k h_min)^-2 times the residual, which residual gates do
not detect. The gauged path never forms K - w^2 M on gradients.

What the gauge leaves: the right-hand side's own discrete charge. A port feed is charge-free only to
rounding, about 5e-16 of its terms at free vertices and in the conductor rows of uncharged conductors. The Gauss
rows -w^2 G_c^T M E = G_c^T b turn a charge q into a potential of order q / w^2, so the spurious field grows like
f^-2 against the port's own and its energy like f^-4: 3.5 % of the field at 10 Hz on the two-port frame, with every
residual at rounding. The default projection (`AXF_FW_TC_CHARGE_PROJECTION=2`) sets each row whose charge is below
1e-12 of its own terms Σ|G_c||b| to zero, its exact value for a charge-free source, which removes it. Real charge is
O(1) of its terms and is kept: a terminal current at a conductor, and the end faces of a source truncated inside the
domain (a `toroidal_box` sector, an open `axial_box`, open `segments`). Zeroing those rows would instead solve a
different, tree-dependent source.

## Formulation

- **Change of basis on the free (non-PEC) edges**, exact, square and nonsingular:
  E = S_c e + G_c phi.
  - S_c injects cotree-edge coefficients e.
  - G_c is the lowest-order discrete gradient (the MFEM H1 -> ND interpolator, required to have +-1 entries,
    one oriented edge per row) on the edge graph in which every set of nodes connected through PEC (essential)
    edges is contracted to one node. **Each PEC conductor therefore carries one potential.**
  - One contracted node per connected component is grounded: the one with the most nodes (often the outer wall,
    but not always: in the capacitor's two finest meshes it is the shell).
  - The spanning tree is a BFS spanning forest of the contracted free-edge graph; tree edges map one to one to
    the potentials phi (checked).
- **K G = 0 is used exactly.** K and M are assembled separately (the closed-PEC energy operators), and the
  transformed operator is
  P^T A P = [[K_cc - w^2 M_cc, -w^2 (M G_c)_c], [sym, -w^2 G_c^T M G_c]],
  none of whose entries applies K to a gradient. The potential block is negative definite. The cotree block
  K_cc - w^2 M_cc is positive definite at low frequency on every mesh checked; this is **not proven in general**.
- **Residuals in split form.** A E = K E_c - w^2 (M E_c + (M G_c) phi) with E_c = S_c e, so K never acts on the
  gradient part.
- **Factorization.** The y-space matrix is symmetrically diagonally equilibrated and factorized once per
  frequency by the existing sparse LU wrapper with the STRUMPACK backend. The rows are gathered to rank 0: it is a
  root-owned factorization with 32-bit indices, like the plain STRUMPACK path, not a distributed one.
- **Low-rank loads.** Reactive/resistive validation loads U diag(c) U^T are applied exactly through the
  Woodbury identity in the transformed space. One factorization is reused for all port, direct-source and
  loaded right-hand sides.
- **Iterative refinement on the transformed residual.** It works on r_y = P^T b - (A_y + V_y C V_y^T) y. With the
  double residual (`AXF_FW_TC_RESIDUAL=double`) it stops at 1e-16 (scaled norm), when the residual stops halving, or
  after `AXF_FW_REFINE_MAX` steps, and the best iterate is kept; the default (extended) residual stops on the
  correction instead (next bullet). With `AXF_FW_TC_REFINE=componentwise` (double residual only)
  the measure is instead the largest componentwise backward
  error |r_i| / (|A_y| |y| + |V_y| |C| |V_y|ᵀ |y| + |b_y|)_i, stopped at 8 ε or when it stops halving. It changed no
  result measured. On the two-port frame the normwise refinement already reaches rounding componentwise; on the
  capacitor meshes neither reaches it in the curl rows (normwise up to 2.4e-11, componentwise 3-5e-14).
  The potential (Gauss-law) rows of r_y contain no K. Port voltages and reciprocity are y^T r_y, so they need
  exactly this residual small, which is why the refinement works on r_y and not on the original-space split
  residual.
- **The residual in extended precision** (default, `AXF_FW_TC_RESIDUAL=extended`).
  - **The problem.** A_y y cancels heavily: |A_y||y| exceeds |A_y y| about 5000-fold on the run-4 topology model,
    mostly in the Gauss rows, whose terms sum to the (zero) charge. A residual evaluated in double therefore carries
    rounding of about ε |A_y||y|. After one refinement step that is as large as the true residual, so the next
    correction is mostly noise. The iteration stops there, backward stable (componentwise at rounding) but with a
    forward error several times the plain path's: run 4's 1e-9 in Ψ.
  - **The remedy.** The residual is accumulated in long double (64-bit mantissa on x86, 2048× finer) from the
    operators themselves, not from the rounded A_y:
    - E = S_c e + G_c φ;
    - the cotree rows (K S_c e − ω² M E);
    - the Gauss rows −ω² G_cᵀ M E (Ky's Gauss block is exactly zero);
    - the Woodbury term;
    - all against the projected b_y.

    Each correction then comes from the true residual of the discrete problem defined by the stored K, M and the
    exact G_c. That target differs from the factorized A_y by the rounding of the stored GᵀMG and MG: from the
    same first iterate, the second correction is 1.1e-12 against the composed operators and 1.1e-13 against the
    stored matrix, so the two residuals differ in the solution by about 1e-12 in y.
  - **The stopping rule.** The loop stops when the correction falls below 4 ε of y, or when it stops halving.
    - The correction is measured in the equilibrated unknowns, ||D⁻¹δy|| / ||D⁻¹y||; it is the forward-error
      estimate (`refinement_history_relative_correction`, `refinement_stop`).
    - Typically 1 → 1e-12 → 5e-16 in three steps. The first correction is always 1, so a contraction slower than ½
      stops at step 3.
    - A correction that grew, which would mean a divergence, or that is not finite, is undone, and the iterate
      before it is kept.
    - The last correction is 0.2-1.8 ε on run 4 and 2.1-3.6 ε at 871k; larger models may take a fourth step.
  - **The residual-based diagnostics are backward errors, not stopping measures, in this mode.** They are
    evaluated on the true residual, but cannot fall below ε ||A_y||y|| / ||b_y|| even for the correctly rounded y.
    The split residual (the gate) and the residual returned to the reciprocity identity are still evaluated in
    double.
  - **Precision.** long double has a 64-bit mantissa on x86-64 Linux. Where it is plain double (macOS arm64, MSVC),
    the run warns that the extended residual is no more precise than double; the JSON records
    `refinement_residual_mantissa_bits`.
  - **Cost.** One more back-solve, and four long-double sparse products per part (G_c, K, M, G_cᵀ) per step. At
    871k free unknowns (run 5) a solve takes 3.6-4.4 s instead of 2.9-3.1 s, next to a 22 s factorization (the sweep
    116 s instead of 110 s), with about 110 MB of transient memory on rank 0.
  - **Options.** `AXF_FW_TC_RESIDUAL` is checked before assembly, refused without the gauge and named in
    `solver_path`. `AXF_FW_TC_REFINE=componentwise` requires `AXF_FW_TC_RESIDUAL=double`.
- **Acceptance gate.** The original-space true residual, evaluated in the split form, must be below 1e-6 (the
  same threshold as the default path). Above it, multiport, source-check and validation-load designs are refused;
  other designs get a "results are NOT reliable" warning.
- **Derived quantities.** The magnetic energy and the probe B use the cotree curl representative E_c
  (K E = K E_c exactly). Taking the curl of the recombined E would reintroduce the cancellation.

## Restrictions (enforced)

- **Lowest-order Nedelec (order 1) only.** Order 2 is refused.
- **Every boundary group must be PEC.** Only then is the curl kernel, H^1(Omega, dOmega) ~ H_2(Omega), spanned
  exactly by the contracted conductor potentials. With natural (PMC) boundary parts, for example PMC symmetry
  cuts, relative cohomology H^1(Omega, Gamma_PEC) can leave harmonic fields in the cotree block, which the code
  does not handle. Such designs are refused with a message pointing to the default path.
- The material operator must be real and lossless: the STRUMPACK path rejects any nonzero imaginary entry.
- The right-hand side must vanish on PEC edges, and must not be zero.

## Per-frequency checks and diagnostics (results JSON)

- `operator_consistency_relative`: the assembled operator applied to a random free vector, compared with
  K - w^2 M. The run aborts if it is >= 1e-11.
- `gradient_block_check`: on a random pure gradient of the reduced system (phi = 0 on every vertex of an
  essential (PEC) edge), the relative error of the legacy summed matrix
  (`legacy_summed_matrix_relative_error_on_gradients`) and of the separately assembled K
  (`separate_K_on_gradients_relative_to_w2_M`). Both are 0 in exact arithmetic; what they show is the rounding of K
  on a gradient against w^2 M on it, which grows exactly like f^-2 (6.2 at 10 Hz on the two-port frame): the error
  the summed-matrix paths make on gradients.
- Per solve: `split_true_relative_residual`, the refinement history (normwise and componentwise),
  `gauss_block_relative_residual` and `curl_block_relative_residual` (each against its own rows' size; for a
  charge-free source the Gauss rows cancel, so that ratio grows with frequency and is not an accuracy measure),
  `componentwise_backward_error` of the curl, free-vertex Gauss and conductor Gauss rows, the legacy summed-matrix
  residual of the recombined field as a diagnostic only, and `curl_representative_norm_fraction` = ||E_c|| / ||E||.
- Factor info: the fractions of positive cotree and negative potential diagonals, and the STRUMPACK inertia
  with `excess_negative_modes`. **STRUMPACK 8.0.0 reports wrong inertia counts (every nonzero pivot counts as
  positive; fixed in 8.1.0), so do not rely on it with 8.0.x.**
- The right-hand side's discrete charge q_p = (G_cᵀ b)_p, against its own terms s_p = Σ|G_c||b|: a
  row is "uncharged" when |q_p| ≤ 1e-12 s_p (rounding of a charge-free source), "charged" otherwise (a terminal
  current, a truncated source's end faces). `rhs_free_vertex_charge_relative` (over all free vertices) and
  `rhs_uncharged_conductor_charge_relative` are the largest |q_p| over those rows divided by the largest s_p over the
  same rows (a ratio of maxima), `..._solved` the same after the projection; `charged_free_vertex_rows` and
  `charged_conductor_rows` count the charged rows. A port feed has no charged free row, and about 4e-16 of free-vertex
  charge on the two-port frame; at low frequency that rounding still enters the field like 1/ω (below).
- Estimators: the field a Gauss-row source would add, against the solution, in the
  energy norm, √(δEᴴMδE / EᴴME) (a gradient adds its own energy), one more solve with the factor each:
  `rhs_charge_field_fraction` for the right-hand side's rounding-level charge (its uncharged rows, before any
  projection: what projection 2 removes), `rhs_charge_field_fraction_solved` for what of it is left in the right-hand
  side solved with (the output's estimated low-frequency error: 0 with projection 2; with projection 1, its
  conductor rows), and `gauss_residual_field_fraction` for the Gauss rows of the final residual. A run warns when
  `rhs_charge_field_fraction_solved` exceeds 1e-3.
- `charge_projection` (0, 1 or 2), `refinement` (normwise or componentwise) and `refinement_residual` (extended or
  double): the options in force. `refinement_history_relative_correction`: ||δy|| / ||y|| per step.
- `tree_cotree` topology: the edge, cotree and potential counts, the contracted PEC nodes and the grounded roots.
- With `field_export_directory`, the curl representative is exported next to E for each port
  (`port_<n>_curl_representative`).

## Validation

Six validation runs, each with its benchmark script and record.

**Which residual.** The records of runs 1, 2, 3, 5 and 6 and the Gauss-law guard's gauged reference rows were made
with the double refinement residual. Run 4's record uses the extended residual. On run 4 and run 5 the extended residual changes Z by ≤ 5e-13; Ψ, whose overlap cancels 1e4-fold,
moves at the 1e-10 level (run 4).

- **Run 1, the energies on the two-port frame** (`benchmarks/tree_cotree_validation.py`, 10 Hz to 12 MHz, seven
  configurations; [validation/tree_cotree.md](../validation/tree_cotree.md)).
  - **Agreement where every path passes** (≥ 10 kHz): Z and L agree to about 1e-11, U_e and U_m to 6e-11. Ψ agrees
    to 9e-10 against AMS (it is linear in AMS's residual) and to 7e-12 between the direct paths.
  - **The ungauged paths, per frequency** (their refused rows are kept):
    - AMS refuses 10 Hz (the loaded validation solve does not converge), 100 Hz and 1 kHz (loaded-field closure);
      its refused 10 Hz row has Z within 5.5e-11 of the gauged path but U_e/f² 160 and 280 times too large;
    - plain STRUMPACK fails to factorize at 10 Hz and refuses 100 Hz (loaded-field closure), and it accepts 1 kHz,
      with a field error of 5e-4 by the energy test below.
    So the refusals come from the validation loads and a failed factorization, and plain STRUMPACK does not refuse
    everything it gets wrong.
  - **The energy test** (U_e = f²c(1+βf²) through 10 and 100 kHz; every value up to 1 kHz within 1e-8):
    - no projection: fails at 10 Hz (field error 3.5 % / 2.2 %) and 100 Hz (3.5e-4 / 2.0e-4);
    - projection 1 (free vertices): fails at 10 Hz (2.6e-3 / 7.4e-3);
    - **projection 2 (the default): passes at every frequency down to 10 Hz** (d ≤ 4e-13, the energies' noise
      floor; U_e22/U_e11 to 4e-13). It also makes the loaded-field closure exact (4e-15 at 10 Hz, against 4e-9
      unprojected).
  - **Cause:** the right-hand side's rounding-level charge, nothing else. The solver's estimate of the field it
    adds (`rhs_charge_field_fraction`) equals the measured error to two digits at 10 Hz, 100 Hz and 1 kHz; the
    final residual's Gauss rows add 1e-15 (`gauss_residual_field_fraction`); the componentwise backward error is
    at rounding after the usual two refinement steps, and refining componentwise changes nothing.
- **Run 2, a capacitor against its closed form** (`benchmarks/tree_cotree_capacitor.py`, record
  `benchmarks/records/tree_cotree_capacitor.json`).
  - **Setup:** a sphere (a = 5 cm) in a closed copper shell (b = 10 cm) bridged by a thin vacuum port, added to the
    two-port frame: a port between two distinct conductors. The gap is its own region, refined uniformly at five levels (size 10 to 2.5 mm; measured
    h = (V/N)^(1/3) from 6.0 to 1.6 mm); straight-sided, and curved (ND1 on mesh order 2).
  - **The low-frequency C is the static capacitance:** C = 1/(ω Im Z) is the same from 10 Hz to 1 kHz to 1.3e-11 on
    every level; above, it rises by the physical ω²L_port C (1.3e-7 at 100 kHz, 1.3e-5 at 1 MHz).
  - **Accuracy:** against 4πε₀ab/(b−a), the three finest levels are +0.40 %, +0.19 % and +0.10 % (straight) and
    +0.60 %, +0.29 % and +0.15 % (curved; above the straight values, as a P1 energy bound on the exact domain).
    Both converge at the observed order 2.09, and extrapolate to the closed form within about 1e-4 (straight:
    −1.3e-6 with the formal order 2, +5.6e-5 with the observed one, GCI 0.13 %; curved: −9.6e-6 and +8.4e-5, GCI
    0.19 %). The static P1 capacitance of the same straight meshes equals C(10 Hz) to 6e-14 (verification).
  - **Curved port feeds are charge-free** after the shared quadrature rule (`port_feed_rule`): 3e-15 to 2.7e-14 of
    the terms, against 2-3e-5 with separate rules.
  - **The frame's ports on these meshes** show the run-1 mechanism, larger: field errors of 2-6 % at 10 Hz
    without projection, 0.5-2.7 % with projection 1, and with projection 2 only the physical −5.1 to −5.5e-10 change
    of U_e/f² between 10 Hz and 1 kHz, at every level and in both geometries. C is unchanged by the projection (to
    2.9e-14 on straight meshes): the capacitor port's charged conductor rows are kept (2 on the three coarse levels;
    1 on the two finest, where the shell has the most vertices and is the grounded root).
  - **Against plain STRUMPACK** (coarsest and finest straight levels): plain STRUMPACK refuses 100 kHz on its own
    residual gate (capacitive port 1.1e-6 / 1.1e-5; its Z is 3.8e-8 / 1.9e-7 off). At 1 MHz and 12 MHz it is
    accepted and agrees with the gauged path to 8.2e-10 / 1.4e-9 and 4.3e-12 / 3.0e-13. The difference follows plain
    STRUMPACK's capacitive-port residual (1.1e-8 / 1.1e-7 at 1 MHz), which grows on finer meshes; the gauged path's
    is 3e-13 to 6e-12. Agreement to ≤ 1e-9 at ≥ 100 kHz holds on the coarse level at 1 and 12 MHz and on the fine
    level at 12 MHz; on the fine level at 1 MHz plain STRUMPACK's own error makes it 1.4e-9.
  - **These runs' curved meshes were not reproducible run to run:** gmsh's high-order ("HighOrder") patch
    optimizer, which every curved level needed, moved nodes by up to 1 % of the element size between runs of the
    same design (mesh hashes in the record), which moved C by up to 2.4e-5. gmsh orders that optimizer, and its
    elastic pass where it smooths curved surfaces, by the memory addresses of the mesh objects. axf-mesher runs
    both in a process that has not meshed, with address randomization off, and each design tested gives one mesh in
    every repeat (`tests/test_mesh3d_reproducible.py`; scope in [design_format.md](../design_format.md)).
  - **The frame's mesh is not fixed across levels:** its vertices are, but its tetrahedra and the air mesh are not
    (U_m varies by 0.7 %); harmless for C, which the shell screens, and for the 10 Hz test, which compares within a
    level.
- **Run 3, the MQS limit of L** (`benchmarks/tree_cotree_mqs_limit.py`, record
  `benchmarks/records/tree_cotree_mqs_limit.json`).
  - **Setup:** the two-port frame made of `superconductor` (cut by its slot, so simply connected) inside a
    superconducting shell from 0.70 to 0.75 m; one design per mesh level, solved once. The full wave (gauged, 10 Hz
    and 100 Hz) treats both as perfect conductors; the MQS solve each run makes on the same mesh excludes the field
    from both. So L = −Im Z₁₁/ω and the MQS L_p of the same port are compared on the same mesh, five levels (133k to
    899k DOFs).
  - **Result:** they agree to 6e-10 to 2.1e-9 at every level, the same at 10 Hz and 100 Hz. The small offset is static and grows with refinement; plausibly the MQS solve's gauge regularization,
    not checked.
  - **L itself** converges: 4.750, 4.894, 5.021, 5.103 and 5.162e-7 H; the three finest extrapolate to 5.207e-7 H
    (observed order 3.1 in h = DOFs^(-1/3), GCI 1.1 %).
- **Run 4, a topology set** (`benchmarks/tree_cotree_topology.py`, record `benchmarks/records/tree_cotree_topology.json`).
  - **Setup:** the two-port frame plus two floating copper spheres, a rod merged into the wall, a floating torus
    (genus 1) and a 2 mm plate (the 3D mesher has no zero-thickness conductor, so the plate stands in for the sheet).
    The contraction finds six conductor nodes; the rod joins the grounded wall.
  - **Against plain STRUMPACK** (100 kHz, 1 MHz, 10 MHz): Z, U_e and U_m agree to ≤ 7e-13. Ψ agrees to 1.1e-11 at
    10 MHz, but only to 1.4e-9 at 100 kHz and 4e-10 at 1 MHz with the double residual: **1e-10 is missed for Ψ**.
  - **Why:** each feature alone gives ≤ 8.8e-11 (the plate the largest). Without the projection the full set is as
    far from plain STRUMPACK (1.3e-9, 3.2e-10), while switching the projection moves the gauged Ψ by 5e-10. Yet
    every gauged solve is componentwise backward stable (≤ 8e-16). So the difference is the combined model's
    conditioning, worse on the gauged path: its source reciprocity is 4e-10 against plain STRUMPACK's 7.5e-11 (on
    the two-port frame alone both are about 1e-12). Above 100 kHz plain STRUMPACK is the more accurate path for Ψ
    on this model; the gauge pays off below 10 kHz, where the plain path refuses or is wrong.
  - **Most of the gap is the refinement's double-precision residual.** The gauged solves stop at the double residual
    floor (above, "The residual in extended precision"). With the extended residual:

    | | 100 kHz | 1 MHz | 10 MHz |
    |---|---|---|---|
    | gauged source reciprocity, double → extended (4 ranks) | 4.0e-10 → 1.4e-12 | 5.4e-10 → 2.2e-12 | 1.0e-11 → 5.2e-14 |
    | Ψ, gauged against plain, double → extended (4 ranks) | 1.4e-9 → 1.1e-10 | 4.0e-10 → 1.3e-10 | 1.1e-11 → 3.5e-12 |
    | gauged Ψ moved by the factorization's rounding alone, double → extended | 7.4e-10 → 3.0e-13 | 3.7e-10 → 3.7e-13 | |

    - **What the change achieves.** It removes the double path's noise-driven error. Gauged Ψ is reproducible to
      3-4e-13 when only the factorization's rounding changes (it moved 4-7e-10), and the reciprocity is 30-300×
      smaller in every realization.
    - **What it does not show.** Over MPI partitions (3, 4, 5 ranks), Ψ is determined to about 6e-11 by either path:
      the gauged path's spread is 5.5-6.8e-11 and the plain path's 5.4-6.2e-11. The path means are 8.5-9.5e-11
      apart, so "Ψ ≤ 1e-10 against plain STRUMPACK" holds at 3 and 5 ranks and fails at 4. The gauged reciprocity
      also depends on the partition: 1.4e-12 to 1.3e-11.
    - **The gap cannot be attributed with these data.** The source reciprocity does not see symmetric differences
      between the two paths' discrete problems. The plain path's reciprocity mostly measures its direct-source
      solve, not the port solves that carry Ψ. With the plate alone, both reciprocities are ≤ 3.8e-12, yet Ψ
      differs by 5-6e-11.
    - **It is at Ψ's double-precision floor.** Ψ's overlap cancellation is 11,720 (100 kHz), 8,762 (1 MHz) and 333
      (10 MHz). Divided by it, the gap between the paths is 82-129 u at the field level at all three frequencies, and
      the within-path spreads 41-63 u. So at 100 kHz-1 MHz, 1e-10 in Ψ asks the two paths to agree on the field to
      about 1e-14.
    - Switching the charge projection moves Ψ by 2-3e-12 (5e-10 with the double residual). Each feature alone gives
      a gauged reciprocity ≤ 2e-12 (the plate: 5.8e-11 with the double residual). Z agrees with the plain path to ≤ 1.2e-13 in every realization,
      and the U_e/f² test holds to 2e-13 at 10 Hz.
  - **The U_e/f² test** holds down to 10 Hz (|d| ≤ 9e-13, U_e22/U_e11 within 1e-12).
- **Run 6, do the ungauged paths fail closed?** (`benchmarks/tree_cotree_fail_closed.py`, record
  `benchmarks/records/tree_cotree_fail_closed.json`). The two-port frame without validation loads
  and source check, at 10 Hz, 100 Hz and 1 kHz. **No:**
  - AMS accepts all three, with no warning, and its field is wrong by a factor 13-17 at 10 Hz, 3.8 % / 2.5 % at
    100 Hz and 4e-4 at 1 kHz (√ of U_e against the gauged path);
  - plain STRUMPACK fails to factorize at 10 Hz and accepts 100 Hz (5.7 % / 3.7 %) and 1 kHz (5e-4), with no
    warning;
  - Z stays right (≤ 5.4e-11 of the gauged path) and Ψ nearly (≤ 3.9e-8): the field outputs (energies, probes,
    exports, scenes) are what is wrong. The ungauged paths therefore estimate and remove this error by default
    ([fullwave_gauss_guard.md](fullwave_gauss_guard.md); the run-6 record was made with the guard off).
- **Run 5, scale** (`benchmarks/tree_cotree_scale.py`, record `benchmarks/records/tree_cotree_scale.json`, 8 ranks).
  - **Setup:** the Benabou et al. 2022 toroid (the `toroid` template) in 3D with a real 2° slit as the readout port,
    lowest order, closed by the PEC sphere. The whole toroid, not a quarter: the scale is what is tested, and PEC
    symmetry planes would only restrict the excitation. Swept at 1, 10 and 100 kHz on the gauged path and on AMS
    (with the Gauss-law guard), on the same mesh: 1,387,041 edges, **871,255 free unknowns**.
  - **Gauged path at 871k:**
    - topology setup 3.6 s;
    - per frequency, 26 s of setup, 22 s of it the rank-0 factorization;
    - 2.9-3.1 s for its one port's solve (Z is 1×1);
    - peak memory on rank 0 12.2, 13.2 and 14.1 GB after the three frequencies (29.5 GB over the 8 ranks at the
      last): it grows about 1 GB per frequency, as on the smaller meshes (4.1-4.7 GB, 6.2-6.6 GB). Whether it levels
      off (fragmentation, or a factor kept per frequency) is to be checked before long sweeps at this size;
    - two refinement steps (three with the extended residual); split residual 9e-11;
      Gauss and curl blocks 4e-16 and 7e-17 of their row terms;
    - `curl_representative_norm_fraction` 2.51.
  - **32-bit headroom** (STRUMPACK runs with 32-bit indices; that bounds the sparse matrix, which the gather checks,
    and each dense front's indexing, not the factor's total, which is spread over fronts. The factor's 2³¹ is a
    yardstick, not a known limit):
    - the gathered rows use 1.1 % of 2³¹ − 1 and the gauged matrix 2.1e7 nonzeros (1 %);
    - the factor's 7.9e8 nonzeros are 0.37 of it;
    - over 329k, 478k and 871k free unknowns, factor size
      grows as N^1.29 and factorization time as N^1.32 (least squares). So the factor passes 2³¹ nonzeros near
      1.9e6 free unknowns, and the gathered matrix near 8e7. The time exponent rises pairwise, 1.19 from 329k to
      478k and 1.39 from 478k to 871k, as the top separators begin to dominate (N² asymptotically): do not
      extrapolate times with 1.32.
  - **Against AMS at 100 kHz** (corrected by the guard): Z and Ψ agree to 2.1e-12, U_e and U_m to 4.1e-12 and
    1.8e-12. At 1 kHz they agree to ≤ 8.3e-13. The guard removed AMS gradient errors of 7.4e-3 (1 kHz), 6.7e-5
    (10 kHz) and 7.4e-7 (100 kHz) of the field.
  - **The STRUMPACK inertia self-check** fails with STRUMPACK 8.0.0 at every size, as in run 1. The inertia is not
    reported.
  - **`gauss_block_relative_residual`'s scale** includes its rows' own terms |A_y||y|. Run 1's and run 4's records
    divide the Gauss rows' residual by those of A y and b only; under the default charge projection b's Gauss rows
    are zero and A y's are rounding, so that ratio reads 1 by construction there.
- **Sources with real charge** (`tests/test_fullwave3d_tree_cotree_options.py`): a `toroidal_box` source over a 120°
  sector keeps its 81 charged end-face rows; its direct-source solve is identical with projections 0 and 2, Z agrees
  to 7e-14, and Ψ to 1.7e-9 at 1 kHz and 1e-13 at 100 kHz (projection 0's port fields carry their rounding charge's
  field, which a charged source sees). Port feeds, loads, B_lab and uniform sources have only rounding-level rows.

## Open

- **Ψ against plain STRUMPACK on run 4's model, at Ψ's floor.** With the extended residual the refinement's noise is
  removed. What remains is a 6e-11 partition spread in either path and a 9e-11 gap between their means. That is about
  100 u at the field level times Ψ's overlap cancellation of 1e4 (Validation, run 4). A 1e-10 criterion therefore
  sits at the floor of double-precision data for this functional; a Ψ criterion for this model should scale with the
  cancellation. To attribute the gap:
  - evaluate the composed long-double residual on the plain solution mapped to y;
  - assemble the plain matrix from the same stored K and M;
  - compare with a third solve (AMS with the guard, tight tolerance).

  Accumulating the split residual, which the reciprocity identity uses, in long double too would turn its per-solve
  terms into first-order error estimates of Ψ.
- **Sources with quadrature-level charge** (Biot-Savart fields of `segments`): their discrete charge is above
  rounding but not physical; the projection keeps it and the estimators do not see it.
- **The quadrature change on curved meshes** (order 2: one 24-point rule for all three feed integrals rather than rules of 4, 11 and 11 points)
  moves port quantities by about the size of the mismatch it removes: on the order-2 slit loop of
  `test_small_loop_3d_matches_axisymmetric`, the MQS L_p by 1.1e-7 and the full-wave L by 5e-8 (same mesh). Not
  measured on a detector-scale mesh, nor for the poloidal-winding harmonic projection, which
  mixes default rules in the same way (`model3d.cpp`).
- Positive definiteness of the cotree block at low frequency is checked only empirically.
- PMC and other natural boundaries (relative cohomology), higher order, and a distributed factorization are not
  implemented.
