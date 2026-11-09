# Error indicator of the 3D full-wave port fields

Status: **on by default for 3D full-wave sweeps, beta.**
- Code: `FwErrorIndicator` (`cpp/src/fw_error_indicator.hpp`, `cpp/src/model3d_fw_indicator.cpp`) and its call in
  `Model3D::fullwave_analysis` (`cpp/src/model3d_fullwave.cpp`): a recovery estimate at order 1; element residual
  problems and a global gradient projection at order 2 and above.
- Validation: `benchmarks/fullwave_indicator.py`, record `benchmarks/records/fullwave_indicator.json` (Validation,
  below). Refine and compare for small models: `axionfem.convergence.refine_and_compare`.
- Tests: `tests/test_fullwave3d_error_indicator.py`. The TEM-line tests run in the default suite (the half-filled
  conducting line among them); the two-port frame tests and the partition test at order 2 need `AXF_SLOW`.

It is the full-wave counterpart of the MQS indicator behind `mesh.adapt` (`Model3D::error_indicators`),
which estimates the static and pickup fields' errors but not the full-wave port fields'.

**What it is for.** It estimates where a port field is under-resolved and how much, and the error this implies for the
port's own Z and L. It is not a bound in either direction.
- **Order 1 (recovery).** On a TEM line with an exact Z, the L estimate was 1.1-1.4 times the true error where the
  magnetic part dominates, and 3-9 times where the magnetic and electric parts partly cancel.
  - The signed estimate was 0.9-1.05 of the true error where the magnetic part dominates, and 0.27-1.0 where the parts
    partly cancel. Next to a crack and a 270° edge it was 1.1-1.3, and 1.4-3.1 where the parts partly cancel there.
    On the two-port frame it was 0.85-1.18.
  - The validation set has only small models; the order-1 band is not established for large models with many
    conductors.
- **Order 2 (element residual problems and a global gradient projection).** The signed L estimate was **0.39-0.94**
  of the error of L where it was given: a TEM line with an exact Z, a crack and a 270° edge, the two-port frame, a
  revolved coax, and a small CORE-type slotted box (the `mini` cases).
  - **Next to cracks and re-entrant edges it is lowest, and falls under refinement:** 0.39 and 0.48 on 1.3M DOFs,
    tending to about 0.37 and 0.46; on the two-port frame's fine mesh 0.51. There the error can be up to about 2.7
    times the estimate. Realistic designs are full of such edges: slit lips, gap ports, frame corners.
  - **On smooth fields** it was 0.82-0.94 (the TEM line).
  - **Above 1?** Against 2D twins, whose difference also holds the curved geometry's offset, one value came out at
    1.13-1.54 (the CORE-like box's fine mesh at 20 MHz), more likely the offset than the estimate. A magnetic part
    that is itself too high (a fit on that box suggests 2-3 times) would make a magnetic-dominant estimate as far
    over.
  - **The electric part** reproduced a smooth TEM line's exact electric error to 0.997-1.000, but about half of it at
    a crack.
  - **Withheld where the two parts are within a factor 8** (energies): they then partly cancel in L. With the parts'
    measured effectivities a factor 8 keeps the signed estimate within 0.34-1.32 of the error.
  - **Lossy media** are handled with the complex ε. Where the loss phases differ, the electric
    part is no longer a lower bound: it can exceed the error's electric energy by up to 1/cos²(Δφ/2), at most 2
    (`gradient.electric_bound_factor`), and the window above holds only without such contrast.
  - **A conductor meshed as a field region** needs its skin depth resolved, or an impedance boundary instead.
  - **In every direction.** The magnetic part's range (0.37-1) and the conductor statement assume |k|h ≲ 1 in every
    direction. With cells that resolve δ normal to a conductor's surface but are 4 times longer along it (h_t of 1-4 δ,
    |k|h_t of 1.3-5.4; boundary-layer meshes are like that), the magnetic part was 1.04-1.23 of the true magnetic error
    on a conductor-filled TEM line (the tree complement: 0.75-1.12); the totals stayed at 1.00-1.01 because the electric
    part dominated there, and the signed estimate was withheld.
  - **The recovery it replaces** was 0.30 to about 14 here: 0.30-0.38 on the smooth TEM line, 6-14 where the two parts
    cancel, and 1.2-1.4 next to the crack and edge.
- **Order 3 and above** run the same code, but nothing has been validated there.

## What it estimates

For every sweep frequency and every port, the error of the stored port field E (per ampere) in the Maxwell energy norm

    ||E||^2 = ∫ ν |curl E|^2 + ω^2 |ε_c| |E|^2 dV,

over the field elements: everything outside perfect conductors, the infinity shell and PML layers. ν = 1/μ, and
ε_c = ε (1 + i tan δ) + iσ/ω is the complex permittivity. The two terms are 4ω² times the time-averaged magnetic and
electric energy densities, so the norm weighs the field the way its stored energy does. The estimate per element,
η_e², has a magnetic and an electric part.

### Order 1: recovery (Zienkiewicz-Zhu)

Per element,

    η_e^2 = ∫_e ν |curl E − R(curl E)|^2 + ω^2 |ε_c| |E − R(E)|^2 dV,

where R averages the element values of E and of curl E at the nodes of the continuous H1 space of the field's order
p.
- **Within one material.** Elements of different materials do not share nodal values, since the fields jump at
  material interfaces. Regions of the same material share them: E is continuous across a region boundary inside one
  material.
- **Cuts.** The interior faces of a boundary group (a surface port, a transition sheet, an interior PEC face, as COMSOL
  imports produce) carry a surface current or charge, so tangential H or normal E jumps across them.
  - At a node on such a face, R averages each side separately: the elements around the node that are joined through
    faces other than cut faces.
  - Around a cut's free edge the two sides join, and the node keeps one average.
  - Averaged across, the estimate on a TEM line with the port on an interior plane was 241-2039 times the true error
    at order 1, falling only like h. As cuts it is 1.1-1.4 (Validation).
  - `cut_faces` and `cut_groups` report them.
- **Across ranks.** Nodes shared by MPI ranks are summed over the ranks; on cuts, one sum per side over all ranks. The
  cuts are found on the serial mesh, which every rank holds. So the result does not depend on the partition (1, 3 and
  4 ranks agree to 3e-11).
- **What the difference measures.** Tangential E and normal curl E are continuous between elements, so E − R(E)
  comes from the jumps of normal E (tied to the Gauss-law residual) and curl E − R(curl E) from those of tangential
  H (Ampère's law).

### Why the recovery fails at order 2

At order p the discrete curl has degree p − 1, and R averages it into continuous H1_p, a space that contains it. Where
curl E_h is continuous, R returns it unchanged, so the recovery sees only its jumps, not the error.
- **At order 1** the curl is piecewise constant, and the nodal averages reconstruct its gradient, which is why order 1
  works.
- **At order 2** on the TEM line, against the analytic field, the recovery saw 0.33-0.36 of the exact magnetic error
  and 0.48-0.73 of the electric one (`tem_exact`, Validation).
- **The overestimates on CORE-like boxes** were not a smooth-field effect but cancellation: there the magnetic and
  electric errors nearly cancel in L (Validation, the closed box).
- **A recovery one order higher** averages the same element values, and is blind in the same way.

### Order 2 and above: element residual problems and a gradient projection

**The magnetic part: element residual problems.** Per field element K the error e = E − E_h obeys

    a_K(e, v) = iω (J, v)_K − a_K(E_h, v) − ∮_∂K (n × ν curl E)·v      for all v,

with a_K(u, v) = ∫_K ν curl u·curl v − ω² ε_c u·v. It is solved for e_K in ND_{p+1}(K), the local Nédélec space one
order up.
- **The traction.** The unknown n × ν curl E is replaced by the average of the two sides' discrete tractions on an
  interior face (Bank-Weiser style).
  - It is replaced by the element's own on a port or sheet face, a Robin group, or the face to the radiating shell or a
    PML layer (no residual there).
  - It is zero on a natural outer face (PMC, an unmarked outer face).
  - An interior face of a `pmc` group is an ordinary interior face (averaged traction). The solver imposes nothing
    there: a natural condition between two elements is no condition, and the field passes through. Zero traction
    there would count the whole traction as error: 0.43 against 2.8e-4 on a TEM line with such a plane (Validation,
    the interior PMC plane).
  - On conductor faces and edges, e_K's tangential trace is zero.
- **On the complement of the local gradients.** The problem is posed on range(Q), the MM-orthogonal complement of the
  local gradients: Q = I − G_r S⁻¹ G_rᵀ MM with S = G_rᵀ MM G_r (the canonical complement).
  - It is solved in the basis Q C, where C picks the free dofs outside a set of rows of the gradient matrix that form an
    invertible block (a "tree", from a column-pivoted QR). The curl block is unchanged (CC Q = CC), the mass block is
    Cᵀ(MM − MM G_r S⁻¹ G_rᵀ MM)C, and the solution is x = Q C y. range(Q) does not depend on C, so neither does the
    result (`fw_local_residual`, `cpp/src/fw_local_residual.cpp`; `cpp/tests/test_fw_indicator_local.cpp`: for three
    conductor patterns a manufactured local error is returned to 1e-12, and two trees, or G perturbed by 1e-15, agree to
    1e-13). An independent basis of ker(G_rᵀ MM) over all 113 patterns gives the same result to 1e-12 on well-shaped
    elements, up to 2e-9 on a flat sliver.
  - On range(Q) the curl-curl form is positive definite with local eigenvalues (against MM) of order 1/h², whatever C
    is. The first local resonance lies at k·h_max ≥ 1.0-1.7 (h_max the element's longest edge) on reasonable shapes (the
    reference, skewed, Kuhn, 5:1 and needle tetrahedra; 1.01 on the high-contrast line's flattest cell), and at 9.7 or
    more without conductor faces or edges. So no regularization is needed while k·h_max stays below about 1, with one
    exception: a flat sliver next to a perfect-conductor edge resonates from k·h_max ≈ 0.27. `residual_kh_warning`
    (given only where it is nonzero) counts, as a heuristic and not a resonance detector, the elements past |k|·h_max =
    1, and the sliver-like ones (smallest height below 0.2 of the shortest edge: the sliver above has 0.042, reasonable
    shapes 0.58-0.97) past 0.2; |k| = ω√(|ε|/|ν|) is the local wave number. A near-resonance inside these bounds is not
    detected: a reciprocal-condition test would fire only at an exact resonance, after a 1e11-fold inflation.
  - **Cost.** Forming the projected mass block's complement part costs about nc²·m′ flops per element (nc ≈ 26,
    m′ ≈ 19 at order 2), and on affine elements MM G_r comes from per-pattern products (9·nf·m′ flops). Measured on
    the crack line at r2 against posing it on span(C) (below): the residual problems +14 % (+15 µs per element), the
    indicator about +3 %, a frequency within noise.
  - **Why range(Q) and not span(C).** Posed on span(C) itself (the "tree complement"), with the unprojected mass, the
    result depends on which rows form the tree, which is decided by round-off ties (110 of 113 conductor patterns,
    every interior element among them). The result then depends on the OpenBLAS kernel family through MFEM's
    LAPACK-built bases, and span(C)'s smallest local eigenvalue is up to 300 times below range(Q)'s. At kh of 0.2-0.6
    the local problems approach resonance: on the benchmark's coarse probe lines at 1.2 GHz an AVX2 host gave a
    magnetic part 3 times larger and a signed estimate reported at 10-25 times the true error where an AVX-512 host
    withheld it. With range(Q) the three kernel families tried (SkylakeX, Zen, Sandybridge) give the same magnetic part
    to 4e-11 and the same withholding.
  - **A penalized formulation gives the same answer:** the system (ν CC − ω²ε MM + β MM P) x = Qᵀb on all free dofs
    reduces, tested against range(P) and range(Q), to the range(Q) Galerkin problem. The runs in the record's
    `development` section used it (below) and give the same exact-error ratios on the TEM line to six digits.
  - The test functions are projected (MM-orthogonally) off the gradients, where the averaged traction's normal flux
    would be inconsistent. At low frequency that inconsistency, divided by ω², would dominate.
  - The port current J is locally a gradient (J = scale ∇V), so it drops out.
- **What is reported.** The magnetic part is ν‖curl e_K‖²; the local problem also contributes ω²|ε|‖e_K‖² to the
  electric part: negligible in a dielectric ((kh)² of the magnetic), but not in a conductor meshed as a field region,
  where |kh|² = 2(h/δ)² (δ the skin depth); there it is a large share of the element's electric part.
- **Accuracy.** The averaged traction is the estimate's whole input: with own-side tractions only it is 0.05 of the
  error, with the neighbour's 3.3 times.
  - So, like every local estimate, its effectivity carries a mesh- and field-dependent constant: 0.6-0.9 of the exact
    magnetic error on the TEM line.
  - The traction is not equilibrated, so it is an estimate without a bound in either direction.
  - Next to sharp edges and corners it is lower, and falls under refinement: there the averaged traction misses the
    singular field (Validation, the probes).
  - A larger local space (ND_{p+2}) lowered it, to 0.64-0.85.
- **Straight elements** take their local matrices from reference moments. The test for a straight element compares
  the Jacobian at the four vertices, which proves it for geometry of order 2 at most. With higher-order geometry (an
  imported mesh) every element is integrated.
- **The recovery's magnetic part,** which the order-2 estimate replaces, is reported beside it as a diagnostic
  (`recovery_magnetic_relative_error`). It is 0.33 of the exact magnetic error on a smooth field, but above the
  residual problems' next to sharp edges (2-3 times on the probes). Taking the larger of the two would help there and
  not on the coax (Summary). An element-wise combination is open.

**The electric part: the error's projection onto the gradients of H1_{p+1}.** Over the whole field region,

    (ε_c ∇ψ, ∇ξ) = −(ε_c E_h, ∇ξ)      for all ξ of H1_{p+1},

and the electric part is ω²|ε_c| ‖∇ψ‖² per element.
- **No flux is needed.** The exact field's term, (ε_c E, ∇ξ) = ∮ ξ n·ε_c E, vanishes: div ε_c E = 0 in every field
  material (with the discrete load, see below), n·ε_c E is continuous between elements, and it is zero or ξ is zero on
  the boundary.
- **So ∇ψ is the error's own projection onto ∇H1_{p+1}.** With one loss phase (every lossless model), and given a
  converged solve, its energy is a lower bound of the error's electric energy.
- **With phase contrast the projection is oblique** in the |ε_c|-norm.
  With Δφ the spread of arg ε_c over the field materials, Re(e^{−iφ_m}ε_c) ≥ cos(Δφ/2)|ε_c| about the midpoint phase
  φ_m, so ‖∇ψ‖² ≤ ‖e‖²/cos²(Δφ/2): the electric part can exceed the error's whole electric energy by that factor.
  - It is 1.003 for tan δ = 0.1 next to vacuum, 1.52 for the conducting line of the Validation (Δφ = 72°), and up
    to 2 for a good conductor next to a dielectric.
  - `gradient.loss_phase_spread_deg` and `gradient.electric_bound_factor` report Δφ and 1/cos²(Δφ/2).
  - Whether the electric part does exceed the error there is not measured.
- **It captures the leading local error of a smooth field.** The polynomials of degree p split as
  P_p³ = N_p ⊕ ∇P̃_{p+1}, so the leading local error of a Nédélec field of order p is a gradient of degree p + 1. Next
  to edges and slits the error is not smooth, and the projection onto ∇H1_{p+1} sees only part of it: about half at a
  crack (Validation, the probes).
- **Boundaries.** ψ = 0 off the field elements, on conductor faces and edges, and on the faces with the element's own
  traction (ports, sheets, Robin groups, shell and PML interfaces), where n·ε_c E is not zero or not known. ψ is
  natural on PMC and unmarked outer faces, and free on interior `pmc` faces (ordinary interior faces).
- **Floating conductors.** ψ = 0 on every conductor, floating ones included, so the error mode that is constant on a
  floating conductor (its charge) is left out. That can underestimate the capacitive error between floating
  conductors (the Gauss-law guard keeps one constant per floating set for this reason). Not measured.
- **Volume ports, an untested hypothesis.** For a volume port whose MQS potential V_h ∈ H1_p is not exact (radial or
  curved current paths), the exact field with the discrete load gives (ε_c E, ∇ξ) = −(i/ω)(J_h, ∇ξ) ≠ 0 for ξ in the
  H1_{p+1} surplus. The projection would then include the load's own discretization charge, growing like 1/ω² in
  energy. The coax (an axial port, linear V) shows no such growth; a radial-current port would test it.

**Lossy media: the complex ε.** Where every field material has the same loss phase arg ε_c (every lossless model), ε_c
is that phase times |ε_c|, and the real problem with |ε_c| gives the same ψ.
- **The solve then.** One AMG-preconditioned CG solve per port field and real/imaginary part.
- **Where the phases differ** (a dielectric with tan δ next to vacuum, a conductor next to a dielectric) the problem is
  complex-symmetric. The real |ε_c| weight then left the exact field's term nonzero at phase-contrast interfaces (n·|ε|E
  jumps there), and the electric part was a mesh-independent artifact. It is solved
  as the real block system [[A_re, −A_im], [A_im, A_re]] by GMRES, preconditioned by PRESB (Axelsson).
  - PRESB inverts [[A_re, −A_im], [A_im, A_re + 2A_im]] with two AMG cycles on A_re + A_im. With exact inner solves the
    preconditioned eigenvalues lie in [½, 1], whatever the mesh and the phases.
  - It is well posed: arg ε_c lies in [0, π/2], so e^{−iπ/4}ε_c has a positive real part, and the form is coercive.
  - **The stopping test is on the energy.** MFEM's GMRES is left-preconditioned: it stops on the Euclidean norm of
    P⁻¹r. Over the raw H1_{p+1} coefficients that weighs every coefficient alike. Next to a good conductor meshed as a
    field region (σ/ωε₀ = 1e10) the conductor's coefficients are small but carry a weight 1e10 times larger in the
    energy, and they went unresolved: the electric part differed by 6.1 % between 1, 3 and 4 ranks, and was 18-26 %
    above the converged value.
  - **The remedy.** The system and PRESB are scaled symmetrically by d = diag(A_re + A_im)^{−1/2}. The residual GMRES then
    measures is d⁻¹P⁻¹r: the error's estimate weighted with the operator's diagonal, whose Euclidean norm follows the
    projection's |ε_c|-energy. The same copper line then agrees across 1, 3 and 4 ranks to 7e-7, in 81 iterations.
  - **Iterations.** On the half-filled conducting line (1.2k-485k DOFs, 8 ranks) 26, 29, 39 and 49 (25-48 without
    the scaling), and 10-16 with tan δ next to vacuum. On the line with a conductor meshed
    as a field region (σ/ωε₀ = 1e6) 86-92 on 11k-643k DOFs, about flat under refinement; with copper (1e10) 81.
    Block-diagonal AMG on |ε_c| had needed 50-114 on the conducting line.
  - **The restart length is 100.** The Krylov basis then holds up to 100 vectors of twice the H1_{p+1} true dofs; on
    the 643k-DOF high-contrast line it did not raise the peak memory (the field solve's).
  - **Cost at high contrast.** There the projection dominates the indicator: 13 % of a frequency on that line, 82 % of
    it the projection (Validation). On the lossless validation models the PCG path costs a few percent (below).
- **The operator** is assembled once per run, or per frequency where σ > 0 (ε_c then changes with ω).
- `gradient.solver` names the path and the residual's norm.

**Conductors meshed as field regions.** A finite-σ conductor can be meshed as a field region (the complex path then
treats it). Its skin depth must be resolved, or the conductor replaced by an impedance boundary.
- **With an unresolved skin** (copper at 100 MHz, skin depth 6.6 µm against 0.6 mm elements) the field solve often
  fails its residual gate, and the estimate is then withheld. Where it passed, the signed Z estimate's imaginary part
  was 0.03-0.2 Ω against a change of Im Z of 2e-7 Ω between meshes, 17 times the copper's whole internal reactance.
  99.9 % of η² sat in the dielectric next to the copper, whose averaged traction carries the unresolved skin layer.
- **With the skin just resolved** (5.8e5 S/m, 66 µm) the predicted change of Im Z was still 28 times the observed.
- **With the skin resolved** (5.8e3 S/m, 0.66 mm) everything solves, the electric part converges, and the signed
  estimate is withheld on most meshes (the parts within a factor 8).
- **Resolve it in every direction.** Cells thin normal to the surface but long along it (h_t of 1-4 δ) gave a
  magnetic part up to 1.23 of the error (Summary at the top).
- **The canonical complement matters in conductors** (the high-contrast line, Validation): in a conductor
  |kh|² = 2(h/δ)² is large, and the tree basis's extra mass (its gradient parts) over-damps the local error. Against
  exact errors on conductor-filled TEM lines with the high-contrast line's cell shapes the range(Q) parts are the
  closer ones; the tree-basis estimate fails η²/ω ≥ |Z_h − Z| (which |a(e, e)| ≤ ‖e‖² requires) on the coarse cubic
  meshes.

**The solve's tolerance and convergence.** Both paths stop at a relative residual of 1e-4 (at most 500 iterations),
in different norms: hypre's PCG on ⟨Cr, r⟩ (C the AMG cycle), an energy-like norm; GMRES on the diagonally scaled
‖d⁻¹P⁻¹r‖₂ (above).
- **Why 1e-4 suffices.** The CG iterate is the energy projection onto its Krylov space, so the projection's energy is
  quadratic in the iterate's error. Against 1e-6, the estimates moved by at most 6e-9 (relative) with PCG, also on a
  near-cancellation and on an electric-dominant singular field (4.8e-9, 8.6e-9).
  GMRES is not a Galerkin method, so there the change is linear in the residual: 4e-5 on the low-contrast conducting
  line without the scaling; with it, 1 and 4 ranks agree to 4e-6 there and to 7e-7 on copper.
- **A projection that did not converge withholds the port's estimate** (`withheld`, with `gradient`'s residual and
  iterations), as a failed field solve does. A CG iterate underestimates the projection's energy. The decision is
  global: the norms are reduced over the ranks, so every rank takes the same branch through the collectives that
  follow. `AXF_FW_INDICATOR_MAX_ITER` caps the iterations, for tests;
  `AXF_FW_INDICATOR_TOL` (a number in (0, 1e-4]: it can only tighten the default 1e-4) sets the tolerance, for tests
  that compare two solvers sharply (`tests/test_gpu.py`); `gradient.tolerance` records it.
- **Skipped in the deep MQS limit.** Where the electric term is below 1e-6 of the norm (`electric_norm_share`) the
  projection is not computed (`gradient.skipped`).
  - η keeps the residual problems' small ω²|ε|‖e_K‖². At the validation's largest ratio of the parts' relative errors
    (44) the skipped part is at most 0.2 % of η², and below 1 % up to a ratio of 100.
  - `electric_relative_error` is then null, with `electric_relative_error_withheld`: the residual problems' part is
    not the electric error. On the crack at 100 kHz it read 0.06-0.2 % where the electric field's error is 5.5-8 %.

**The accuracy of the electric part.**
- **On a smooth field** it reproduced the TEM line's exact electric error to 0.997-1.000 at orders 1 and 2.
- **At a crack** it was about half of the electric error (Validation, the probes).
- **Local problems were worse.** Local Neumann problems with the averaged normal flux were 9-16 times high on the
  full local space and 1.7-5.7 times on its hierarchical surplus: the flux error is of the error's own order. The
  global projection needs no flux.

**a(e, e) with sign.** Per element a_K(e_K, e_K) + (−ω² ε_c ∇ψ·∇ψ), not conjugated, estimates a(e, e) (next section),
for lossy and radiating models too.

**The curl from the curl representative.** With a Gauss-law guard or the tree-cotree gauge, curl E comes from the curl
representative, as the magnetic energy does: at low frequency E is mostly gradient, and its own curl would carry that
part's rounding. Without the guard (`AXF_FW_GAUSS_GUARD=off`) the recovery's electric part includes the solver's
gradient error (on the two-port frame at 10 Hz, order 1, `electric_relative_error` 0.92 against 0.64 with the guard);
its share of the norm is then negligible (4e-12).

**A failed solve.** A port whose solve did not pass its residual gate (`solver.accepted` false) gets no estimate
(`withheld`, with the reason): its field is not an approximation the indicator can assess. A norm that is zero or not
finite gives null, not 0, and such estimates are left out of the run's maximum (`estimates_left_out_of_max`).

## Z and L

For a port's own impedance, Galerkin orthogonality of the complex-symmetric form a gives Z_h − Z = a(e, e)/(iω) per
ampere, with e = E − E_h. This assumes an exact algebraic solve, the meshed geometry and the same discrete load. The
solver's algebraic error is not part of e, but it does enter the estimate, which is a residual of E_h (Other checks,
the partition). Taking η for ||e||:
- `Z_error_estimate_ohm` = η²/ω;
- `L_relative_error_estimate` = η²/(ω² L), for an inductive port.

**Volume terms only.** a also contains the radiating shell's mapped coefficients, the exterior expansion, PML tensors
and the Robin (impedance, sheet, scattering) boundary terms. The norm holds none of them, so the estimates leave out
the discretization error there.
- **With PML layers or Robin groups** the Z and L estimates are withheld (`Z_L_withheld`, naming the terminations).
  On a PML-terminated TEM line the Z estimate was 2e-5 to 6e-5 of the true error.
- **With the radiating exterior** they are given with `Z_L_scope` saying what is left out. The termination's own
  error must be checked separately, for example by moving the domain radius.
- `fullwave.error_indicator.terminations_left_out` lists what the norm leaves out.

**With sign.** The magnetic and electric errors move L in opposite directions: L_h − L = Re a(e, e)/ω², and for a
lossless closed model (E = iẽ) a(e, e) = −(‖curl ẽ‖²_ν − ω²‖ẽ‖²_ε).
- In the deep MQS limit the magnetic part dominates, and the FEM L lies below the exact one.
- Towards a self-resonance the electric part grows. Where it dominates, the FEM L lies above the exact one; where the
  two are comparable they partly cancel, and `L_relative_error_estimate`, which counts both with the same sign, lies
  above the error of L. It is not a bound either: at the crack at 1.2 GHz it fell to 1.03 of the error under
  refinement, and where the magnetic part dominates next to edges it is as low as the signed estimate (the probes).
- `L_signed_relative_error_estimate` estimates L_h/L − 1:
  - at order 1, −(η_m² − η_e²)/(ω² L), which assumes E = iẽ;
  - at order 2 and above, Re Σ a_K/(ω² L), from the per-element forms (no lossless assumption), with
    `Z_signed_error_estimate_ohm` = Σ a_K/(iω) {re, im}, the estimate of Z_h − Z.

  Between two meshes the change of L should then be the change of this estimate, which needs no reference solution
  (`refine_and_compare` reports both as `changes`).
- **At order 2 and above both are withheld** where the two parts' energies η_m² and η_e² are within a factor 8 of each
  other (`L_signed_withheld`, and `Z_signed_error_estimate_ohm` null).
  - **The window it guarantees.** Let the parts' effectivities be θ_m and θ_e (estimate over true), and r ≥ 8 the
    estimated ratio. Then the signed estimate over the error of L is θ_m(r − 1)/(r − θ_m/θ_e) where the magnetic part
    dominates, and θ_e(r − 1)/(r − θ_e/θ_m) where the electric part does. With the measured ranges (magnetic 0.37-1,
    electric 0.45-1, the probes included) that is **0.34-1.32**, in models without loss-phase contrast.
  - **With phase contrast** the electric part's upper limit is not 1 but `gradient.electric_bound_factor`,
    1/cos²(Δφ/2), up to 2 (the electric part, above). With θ_e = 2 and θ_m = 0.37 the electric-dominant end at r = 8
    is 2·7/(8 − 2/0.37) = 5.4. Whether such θ_e occurs is not measured: on the high-contrast lines the magnetic part
    dominated, or the estimate was withheld.
  - **A factor 4** would allow 0.31-2.3, and a ratio below that is not resolved at all: on the
    CORE-like box at 3 MHz (uniform feed) the estimated parts were 3.6 apart and the true ones about 1.25, and the
    signed estimate would have been 7.5 times the change of L. On the TEM line with cubic cells at 600 MHz it had the
    wrong sign.
  - **What 8 costs.** Of the validation's cases these are withheld at a factor 8 (but not at a factor 4): the TEM line at 200 MHz (5:1 cells, ratios
    4.3-4.8; the signed estimate would be 0.78-0.96 there at a factor 4), the TEM line with cubic cells at
    100 MHz (ratio 5.4; 0.69) and the coax's fine mesh at 4 MHz (ratio 6.0). The change of L between meshes is then
    not predicted for the coax at 4 MHz.
  - **Outside the window.** A part whose effectivity lies outside the measured ranges is not covered. The CORE-like
    box's fit puts its magnetic part about 2 times over at 3 MHz; that case is withheld either way.
  - At order 1 it is kept. Where the parts partly cancel the recovery's was 0.27-1.0 of the TEM line's error, but
    1.4-3.1 next to the crack and the edge at 1.2 GHz (the probes).

## Surface term

On faces between a field element and a meshed conductor, the recovery error of curl E (B, so n × H) is weighted by
1/μ_r² of the field side and reported as `surface_relative_error`.
- **It is the recovery's at every order.** At order 2 and above the recovery runs for this term and for the magnetic
  diagnostic.
- **It is unvalidated at order 2 and above.** There the recovery averages the degree-(p − 1) curl into a space that
  contains it, so it sees only the jumps of curl E, as for the volume (above): on a smooth field it would see about a
  third of the error. Next to edges the recovery's magnetic part was above the residual problems' (the probes), so the
  direction of its error is not known either. `reliability` says so.
- **It is not part of the total.**
- **It shows how well the surface current is resolved,** which sets surface losses. The 3D
  full-wave path computes those losses on the same faces (`surface_Gamma`,
  [fullwave_surface_losses.md](fullwave_surface_losses.md)). The MQS indicator's surface term for Γ does the same for the
  pickup field.
- **It covers meshed conductors only.** PEC boundary groups (imported meshes) are not included: there
  `conductor_faces` is 0 and the key is absent.

## What it does not see

- The geometry's own approximation (curved order-2 elements on a curved wall). On the small CORE-like box below it
  dominates the 3D-to-2D difference.
- The terminations' discretization error (above).
- The solver's algebraic error as such. It is not part of e, but the estimate is a residual of E_h, so it moves the
  estimate in proportion to the field's own relative residual. At order 2 that made 1 and 4 ranks differ by 3e-7
  (Other checks).
- Errors of quantities other than the port fields (Ψ's source field, for example).
- At order 2 and above, part of the error next to sharp edges and corners (down to about 0.4 of it on the probes, and
  falling under refinement), and in the elements next to ports and sheets, where the gradient projection is zero and
  the traction is the element's own.
- The charge mode of floating conductors (ψ = 0 there).
- An unresolved skin layer of a conductor meshed as a field region (above): the estimate is then far too high.
- The recovery (order 1, the surface term and the magnetic diagnostic) still treats an interior `pmc` face as a cut,
  and so loses the jump there. Mild; unchanged, to keep order 1 bit-identical.
- Order 3 and above: the code runs (ND_{p+1} local problems, the H1_{p+1} projection) but nothing has been validated
  there, as for the order-3 field solve itself.

## Enabling it, and its cost

```bash
export AXF_FW_ERROR_INDICATOR=on    # default: every port field of every sweep frequency
export AXF_FW_ERROR_INDICATOR=off   # skip it
```

Any other value is an error (empty means unset). On and off give bit-identical results in every other output (a
default-suite test checks this on the TEM line at order 2).

**Cost.** Measured share of a frequency's time, record `cost` (`benchmarks/fullwave_indicator.py cost`; the two-port
frame, 2 ports, 4 ranks; measured with a variant of the canonical complement whose residual problems cost about twice
as much as those of the present code, so these shares are upper bounds):

| | AMS + guard | plain STRUMPACK | tree-cotree gauged |
|---|---|---|---|
| order 1 (109k DOFs, recovery) | 2.6-2.8 % | 2.4-2.8 % | 8.2-10.6 % |
| order 2 (572k DOFs, residual + gradient + recovery) | 2.9 % (4.4 % at the first frequency) | 5.3 % (7.5 %) | (lowest order only) |

- **The order-1 share varies with the load on the node:** other runs measured 2.6-3.1 % and 7.0-12.2 % (gauged).
- **At order 2 the recovery alone cost 0.5 % and 1.3-1.6 %, so the estimator costs 3-6 times more.** Per port on this
  frame (21k elements per rank):
  - the residual problems take 0.55 s, 26 µs per element (the measured variant; the present code costs about half its
    overhead over the tree complement: the residual problems' section);
  - the gradient projection 0.23 s (11 PCG iterations at the tolerance of 1e-4; 0.26 s and 17 at 1e-6), plus 1.06 s
    once per run for its operator and AMG setup;
  - the recovery, for the surface term and the magnetic diagnostic, 0.24 s.
- **On the crack probe** (1.28M DOFs, 8 ranks; tree complement): 7.9 % of the first frequency and 6.0 % of
  the second (an independent on-off measurement gave 8.2 % and 6.3 %, the difference of the process wall time
  matching the reported times to 5 %). Of the second frequency's 5.7 s: the residual problems 1.8 s, the projection
  1.8 s (9 iterations; 4.0 s with its setup at the first), the recovery 2.1 s. The recovery runs there only for the
  magnetic diagnostic (the fin is a boundary group, not a meshed conductor).
- **Memory.** The H1_{p+1} operator and its AMG are held through the whole sweep, during every later field solve. On
  the crack probe the largest rank's peak RSS rose from 1.01 to 1.13 GB and the sum over the 8 ranks from 7.9 to
  8.9 GB: +11-12 %.
- **On the validation runs at order 2 (32 ranks; with the tree complement in parentheses):**
  - the two-port frame 2.2-2.4 % on 4.5M DOFs, 5.1-6.2 % on 572k (2.1-2.3 % on 4.5M);
  - the mini box (radiating) 1.1-2.1 % (1.0-1.9 %);
  - the closed mini box 5.0-8.5 % (4.5-7.8 %) and the coax 6.4-11.4 % (5.7-10.8 %). Their solves are fast (AMS
    converges in few iterations in the closed sphere), and the indicator's time grows like the element count.
- **The savings taken.** The projection's tolerance 1e-4 instead of 1e-6, 35-45 % fewer
  iterations; and no projection where the electric share of the norm is below 1e-6 (deep MQS).
- **What makes it affordable.**
  - The local matrices of straight elements, including the interior of curved meshes, come from reference moments.
  - The face loops allocate nothing.
  - The gradient operator is assembled once per run (per frequency only where σ > 0).
- `AXF_FW_ERROR_INDICATOR=off` skips it, for example on small gauged or fast-solving runs.
- `Recover` reduces the full 13-component H1 vector once per material group and port: fine for a few materials.

## Results JSON

- `fullwave.error_indicator`:
  - `method`, `norm`, `scope`, `reliability` (the effectivity ranges above, in short, per order);
  - `recovery_order` (the field's order p) and, at order 2 and above, `local_order` (p + 1, the residual problems'
    and the gradient projection's order);
  - `field_elements`, `field_elements_by_material`, `excluded_elements` (`perfect_conductor`, `shell`, `pml`);
  - `conductor_faces`, `cut_faces`, `cut_groups`, `terminations_left_out`;
  - `setup_seconds`: the construction, once per run (slowest rank);
  - after the sweep: `max_estimated_relative_error` and `max_at` (`f_Hz`, `port_index`), both null when every estimate
    was withheld; `estimates_left_out_of_max`; `evaluation_seconds`, the per-port `seconds` summed over the sweep;
  - with `fullwave.field_export_directory`: `files`, the per-element indicator files.
- Per frequency and port (`sweep[i].port_solutions[j].error_indicator`; the frequency row's own `error_indicator` is
  port 0's):
  - `estimated_relative_error`: η/||E||;
  - `magnetic_relative_error` and `electric_relative_error`: each part against its own norm;
  - `electric_norm_share`: the electric term's share of ||E||²;
  - `Z_error_estimate_ohm`, `Z_relative_error_estimate` and, for an inductive port, `L_relative_error_estimate` and
    `L_signed_relative_error_estimate` (null with `L_signed_withheld` when withheld); at order 2 and above also
    `Z_signed_error_estimate_ohm` {re, im} (null when withheld); or `Z_L_withheld`; with the radiating exterior also
    `Z_L_scope`;
  - `by_region`: per region, `share` of η² and `norm_share` of ||E||²;
  - `largest_element`: `center_m`, `region` and `share` of the element with the largest η_e²;
  - `surface_relative_error` (with conductor faces; unvalidated at order 2 and above);
  - `curl_from` (`curl representative` or `E`) and `seconds`;
  - at order 2 and above:
    - `recovery_magnetic_relative_error`, the recovery's magnetic part (a diagnostic, not in the total);
    - `residual_seconds` (the residual problems, slowest rank);
    - `gradient`: the projection's `seconds` (slowest rank), `iterations`, `relative_residual`, `tolerance`,
      `converged` and `solver`; with phase contrast also `loss_phase_spread_deg` and `electric_bound_factor`; or only
      `skipped`, with the reason, below an electric share of 1e-6 (then `electric_relative_error` is null, with
      `electric_relative_error_withheld`);
  - or only `withheld` and `seconds`, for a solve that failed its residual gate; with `gradient` too, for a projection
    that did not converge.
- With `fullwave.field_export_directory`: `f_{i}_port_{j}_error_indicator_rank_{r}.gf`, η_e²/||E||² per element as an
  MFEM L2 (order 0) grid function on `mesh_rank_{r}.mesh`. It shows where the field is under-resolved. A write failure
  on any rank is raised on every rank.

## Validation

`benchmarks/fullwave_indicator.py`, record `benchmarks/records/fullwave_indicator.json`; the record's `provenance`
lists the build and run of every row.
- **Main runs:** every study with the canonical complement, including the 2D twins.
- **Record section `development`** (runs of estimator variants, kept in the record and marked `carried_over`; the
  present code does not regenerate them): every estimator variant side by side, including the
  recovery at order 2 and the TEM line's exact error split. Its residual problems use the
  penalized range(Q) formulation (above), so it is an independent check of the canonical complement: the exact-error
  ratios equal the main runs' to six digits.
- **Against the tree complement** (posing the local problems on span(C), above): order 1 is identical; at order 2 the
  magnetic part differs most where kh is largest (the coarse probes at 1.2 GHz, 6-7 %, their L estimates 7-9 %; the TEM
  line's coarsest meshes, 1-3 % at 100-200 MHz and 16 % on the 5:1 line at 300 MHz, the sheet and PML lines 5.5-5.6 %
  there) and in a conductor meshed as a field region (up to 38 %), elsewhere below 1 %; no signed estimate changes its
  withholding. Against exact errors the 5:1 line's coarsest mesh at 300 MHz has a magnetic part 0.93 of the exact one
  with the canonical complement and 1.32 with the tree complement.

**Kinds of check.**
- **An exact reference.** The parallel-plate TEM line of `benchmarks/fw3d_robin_validation.py` has an exact Z. It
  measures the effectivity directly.
  - The record's `development.tem_exact` also splits the exact error into its magnetic and electric parts, against
    the analytic TEM field.
- **Uniform refinement** (`refine_and_compare`, `mesh.refinements`) keeps the element geometry, so L's change between
  meshes is the field's discretization error alone.
  - The change of L between two meshes against the change of the signed estimate needs no reference at all
    ("predicted/observed"). It equals (θ₀ − θ₁/ρ)/(1 − 1/ρ) for per-mesh effectivities θ₀, θ₁ and an error ratio ρ, so
    it mostly reports the coarse mesh's effectivity.
  - With three meshes or more the reference is the Richardson limit, and the effectivity is given per mesh. Where
    the rate is known (the probes: 1 at a crack, 4/3 at a 270° edge, also the two-port frame) the limit uses it.
- **The axisymmetric twin** (2D full wave on the exact geometry) adds the 3D meshes' geometric error, which the
  indicator does not see.

### The TEM line

PEC plates, PMC sides, surface port, shorted end unless noted; one process (`tem`). True error: L_h/L − 1 against the
exact Z. Each column lists the meshes, coarse to fine (uniform refinement).

| case | order | f | electric share | L estimate / true | signed / true |
|---|---|---|---|---|---|
| 5:1 cells | 1 | 100 MHz | 0.12 | 1.38, 1.44, 1.44, 1.44 | 0.91, 0.96, 0.98, 0.99 |
| 5:1 cells | 1 | 200 MHz | 0.38 | 8.5, 8.9, 8.8, 8.7 | 0.27, 0.51, 0.72, 0.85 |
| cubic cells | 1 | 100 MHz | 0.12 | 1.27, 1.25, 1.23 | 1.05, 1.04, 1.02 |
| cubic cells | 1 | 600 MHz | 0.44 | 3.29, 3.24, 3.16 | 0.97, 0.99, 0.99 |
| port on an interior plane (a cut) | 1 | 100 MHz | 0.03 | 1.08, 1.11, 1.11, 1.10 | 1.00, 1.02, 1.02, 1.01 |
| port on an interior plane (a cut) | 1 | 200 MHz | 0.11 | 1.38, 1.43, 1.42, 1.41 | 0.99, 1.01, 1.01, 1.01 |
| 5:1 cells | 2 | 100 MHz | 0.12 | 0.98, 0.92, 0.91 (recovery: 0.38) | 0.89, 0.83, 0.82 (recovery: 0.31) |
| 5:1 cells | 2 | 200 MHz | 0.38 | 1.36, 1.28, 1.25 (0.58) | withheld: parts 4.8, 4.4, 4.3 apart (0.96, 0.82, 0.78 at a factor 4 with the tree complement) |
| cubic cells | 2 | 100 MHz | 0.12 | 1.00 (0.49) | withheld: parts 5.4 apart (0.69 at a factor 4; recovery 0.34) |
| cubic cells | 2 | 600 MHz | 0.44 | 6.1, 5.3, 5.0 (3.0-2.7) | withheld |
| port on an interior plane (a cut) | 2 | 100 MHz | 0.03 | 0.95, 0.87, 0.86 (0.32-0.34) | 0.94, 0.86, 0.84 (0.31-0.33) |
| port on an interior plane (a cut) | 2 | 200 MHz | 0.11 | 1.02, 0.93, 0.90 (0.36-0.37) | 0.94, 0.85, 0.82 (0.30-0.31) |

- **Order 1 is the recovery.**
- **The order-2 values in parentheses** are the recovery's at order 2 (same meshes).
- **The true errors converge at the formal rate:** about ×4 per level at order 1, ×15-16 at order 2. The cubic-cell
  order-2 rows at 100 MHz reach the solver floor beyond the first mesh (true errors 2e-11 and below) and are left
  out.
- **The unsigned estimate counts both parts with the same sign.** Where the electric part is a third of the norm or
  more, it lies above the error of L (1.3-1.4 at 200 MHz), and where the parts nearly cancel (cubic cells, 600 MHz:
  exact parts 0.76 of each other) it is 5-6 times the error.
  - There the signed estimate is withheld at order 2: the parts are within a factor 8 (a factor 4 would withhold
    only the 600 MHz row).
  - At order 1 the recovery's signed estimate stays good there (0.97-0.99).
- **The resistive sheet and the PML line** (order 2, Robin groups and PML) run with the residual estimator. Z and L are
  withheld, and η is given. On the refined PML mesh the solves fail their residual gate (GMRES, AMS, one process), so
  that run's estimates are withheld.
- **The gradient projection** took 5-11 PCG iterations (relative residual 3e-5 to 9e-5).
- **The canonical complement against the tree complement** differs at order 2 on the coarsest meshes, where kh is
  largest: the signed estimate of the cut at 200 MHz is 0.94 against 0.99, the 5:1 line's unsigned one at 200 MHz
  1.36 against 1.43 (the magnetic part about 3 % lower); one level finer the differences are below 1 %, two levels
  finer below 0.2 %. The largest is a row the table leaves out: the 5:1 line's coarsest mesh at 300 MHz, magnetic part
  −16 % (0.002660 against 0.003167), Z estimate over the true error 2.35 against 3.02. Against the analytic field its
  magnetic part is 0.93 of the exact one (energies), against 1.32 with the tree complement (above the window's magnetic
  0.37-1). The sheet and PML lines' coarsest meshes at 300 MHz differ by −5.6 % and −5.5 %. Order 1 is the same.
- **The recovery's magnetic diagnostic** was 0.25-0.40 of the residual problems' magnetic part (energies) on the
  5:1 cells and the cut, 0.48-0.62 on the cubic cells: on this smooth field it is the low one.

**The exact error split** (record `development.tem_exact`). The magnetic and electric error energies against the
analytic TEM field, each estimate over the exact value (the 5:1 and cubic lines, two meshes each, both frequencies):

| | order 1 | order 2 |
|---|---|---|
| recovery, magnetic | 1.10-1.13 | **0.33-0.36** |
| recovery, electric | 0.94-1.06 | 0.48-0.73 |
| residual problems, magnetic | 0.49-0.72 | 0.60-0.91 |
| gradient part, local, full H1_{p+1} | 2.1-11.6 | 9.5-16.3 |
| gradient part, local, hierarchical surplus | 3.2-17.4 | 1.7-5.7 |
| **gradient part, global projection** | **0.999-1.000** | **0.997-1.000** |

- **At order 2 the recovery sees a third of the magnetic error:** its space contains the discrete curl (above).
- **The global gradient projection is exact** to 3e-3 at both orders: on this smooth line the electric error is the
  error's gradient part. Next to a crack it is not (the probes, below).
- **The residual problems' magnetic part** is 0.84-0.91 on the 5:1 cells and 0.60-0.73 on the cubic cells at order 2.
  The `development` runs solved the canonical complement's problem (the penalized range(Q) system), so these ratios
  are the present code's too (equal to six digits). The tree complement puts the magnetic part up to 6.7 % higher in
  energy, and 1.32 of the exact one on the 5:1 line's coarsest mesh at 300 MHz.
- **At order 1 the recovery is the better estimate** (1.10-1.13 against 0.49-0.72), which is why order 1 keeps it.

### A crack and a 270° edge (the probes)

The shorted parallel-plate line (l = 4 cm, h = 2 cm, W = 4 cm; PEC plates, PMC sides, surface port at x = 0) with a
sharp feature on its far half, y ≤ h/2, x ≥ 1 cm (`probe`):
- **crack:** a zero-thickness PEC fin on z = W/2 (fields like r^{−1/2} at its edge);
- **ridge:** a 2 cm wide perfect-conductor block (270° re-entrant edges, r^{−1/3});
- **crackopen:** the crack with an open (PMC) end, a capacitive stub whose error is electric (1200 times the magnetic).

Four uniformly refined meshes: order 2 at 2.9k, 21k, 163k and 1.28M DOFs, order 1 on the same meshes (0.6k-239k
DOFs). The reference is the order-2 series' Richardson limit at the theoretical rate (1 at the crack, 4/3 at the
edge); the observed rates are 1.01-1.03 and 1.37-1.41. True error: L_h/L − 1 against it; it halves per level at
the crack (−2.1e-2 to −2.5e-3 at 100 MHz).

| case | f | order 2, signed / true | order 2, unsigned / true | order 1 (recovery), signed / true |
|---|---|---|---|---|
| crack | 100 MHz | **0.48, 0.43, 0.40, 0.39** | 0.48-0.39 | 1.13, 1.19, 1.21, 1.22 |
| ridge | 100 MHz | **0.62, 0.54, 0.50, 0.48** | 0.62-0.49 | 1.11, 1.32, 1.31, 1.31 |
| crack | 1.2 GHz (kl = 1) | withheld | 1.44, 1.18, 1.07, 1.03 (tree complement: 1.58, 1.19) | 1.55, 1.46, 1.43, 1.42 |
| ridge | 1.2 GHz | withheld | 4.8, 3.6, 3.1, 3.0 (tree complement: 5.2) | 3.07, 2.42, 2.09, 1.97 |
| crackopen, Im Z (electric) | 100 MHz | **0.50, 0.50, 0.50, 0.49** | | |

- **Next to a crack and a re-entrant edge the estimate converges faster than the error.** On the finest meshes the
  error is 2.6 (crack) and 2.1 (edge) times the estimate. The decrements of the effectivity shrink geometrically, which
  points to limits near 0.37 and 0.46: up to about 2.7 and 2.2 times.
- **Both parts are low there.** The magnetic part is the residual problems', whose averaged traction misses the
  singular field. The electric part at the open crack is half of the electric error on every mesh: the projection
  onto ∇H1_{p+1} sees only part of a singular gradient field.
- **The recovery's magnetic part** (`recovery_magnetic_relative_error`) is 2.5-3.3 times the residual problems' at
  the crack and 2.1-2.9 times at the edge (energies, 100 MHz), the ratio growing under refinement.
- **At 1.2 GHz** the parts are 1.2-2.3 apart, and the signed estimate is withheld. The unsigned estimate is not an
  upper bound either: at the crack it falls to 1.03 of the error of L and keeps falling.
- **Order 1** is 1.1-1.3 where the magnetic part dominates, and 1.4-3.1 at 1.2 GHz, where the parts partly cancel.

### Lossy media: the half-filled line

The 5:1 shorted TEM line with its lower half (y < h/2) filled (`layer`): ε_r = 4
lossless, with tan δ = 0.01 and 0.1, and a conductor (ε_r = 1, σ = 0.0167 S/m, ε″/ε′ = 3 at 100 MHz). Order 2,
100 MHz, four uniformly refined meshes (1.2k, 8.5k, 63k, 485k DOFs).

| fill | `electric_relative_error` | signed Z estimate, finest mesh | predicted/observed change of Z | with \|ε\| instead of ε_c (earlier formulation) |
|---|---|---|---|---|
| ε_r = 4, lossless | 1.3e-3, 8.7e-4, 3.7e-4, 1.2e-4 | −5.9e-7i Ω | 0.76, 0.93 | the same |
| ε_r = 4, tan δ = 0.01 | the lossless values × 1.0001 | 1.1e-8 − 5.9e-7i Ω | 0.76, 0.93 | 4.1e-3, 4.0e-3, 3.9e-3 (r0-r2) |
| ε_r = 4, tan δ = 0.1 | the lossless values × 1.006 | 1.1e-7 − 5.9e-7i Ω | 0.76, 0.93 | 3.9e-2 on every mesh |
| conductor | 1.5e-3, 9.8e-4, 4.1e-4, 1.3e-4 | 6.8e-7 + 4.8e-7i Ω | 0.77 + 0.007i, 0.92 − 0.002i | 0.48-0.49 on every mesh; −5.4 + 10.7i Ω |

- **The electric part converges with the field again,** and with a lossy dielectric it equals the lossless line's to
  1e-4 (tan δ = 0.01) and 6e-3 (tan δ = 0.1). With |ε| it had been a mesh-independent artifact growing like tan δ.
- **The signed Z estimate predicts the change of Z** between consecutive meshes, its real part (the loss) included: on
  the conducting line 0.77 and 0.92 of the complex change, with a phase error below 0.01. With |ε| it had put
  Z_h − Z at −5.4 + 10.7i Ω, against changes of 5e-5 Ω.
- **The solve.** GMRES with PRESB took 26, 29, 39 and 49 iterations on the conducting line (tolerance 1e-4, the
  scaled residual), 10-16 with tan δ. The real path takes 10-14 PCG iterations on the lossless line.
- **This line is not yet asymptotic.** L changes by 3.9e-7, 2.2e-7 and 3.7e-8 per level (1.8 then 5.9 times smaller),
  so a per-mesh effectivity depends on the assumed rate: at the finest mesh 0.51 at the fitted rate 2.6, 1.6 at the
  smooth rate 4. The predicted/observed changes do not need one.
- **The test** `test_a_lossy_fill_uses_the_complex_projection` runs the conducting line on three meshes.

### A conductor meshed as a field region (the high-contrast line)

A parallel-plate line, l = 0.1 m, h = 2 cm, W = 4 cm, with layers in y: 0-5 mm a conductor meshed as a field region,
5-10 mm ε_r = 4 with tan δ = 0.02, vacuum above; open (PMC) end; order 2, 100 MHz (`contrast`, `contrast_partition`).
- **σ = 5.8e3 S/m** (σ/ωε₀ = 1e6, skin depth 0.66 mm, Δφ = 90°, `electric_bound_factor` 2.0), on 1.6k-643k DOFs:
  - the first mesh fails the field solve's residual gate (withheld);
  - the electric part converges: 1.06e-2, 8.6e-3, 4.4e-3 (tree complement: 8.6e-3, 7.2e-3, 3.9e-3; the local
    problems' ω²|ε|‖e_K‖² weighs heavily in a conductor, and the magnetic part differs by 4-38 %, 0.115, 0.065,
    0.032);
  - GMRES took 86, 92 and 92 iterations;
  - the indicator took 11-14 % of a frequency. On the 643k-DOF mesh, indicator on against off (8 ranks): 13.3 % of
    the frequency, 82 % of it the projection; the peak memory did not change (2.20 GB per rank,
    the field solve's peak).
- **Copper** (σ/ωε₀ = 1e10, skin depth 6.6 µm, unresolved), the third mesh (85k DOFs) on 1, 3 and 4 ranks: the electric
  part is 2.608216e-4, 2.608216e-4 and 2.608215e-4 (81 iterations; 1.902318e-4 with the tree complement). Without
  the diagonal scaling it is 2.397e-4, 2.392e-4 and 2.250e-4 (17-28 iterations). The signed Z
  estimate's imaginary part, 0.068 Ω, is the unresolved skin's overestimate (above).

### An interior PMC plane

The 5:1 shorted TEM line with and without its interior faces on y = h/2 as a `pmc` group, order 2, 100 MHz, three
meshes (`pmcplane`). The solver ignores the group, so Z is bit-identical with and without it.
- **The estimate is bit-identical too:** η = 2.78e-4, 6.84e-5 and 1.70e-5 on both lines, and the same signed L
  estimates (−9.3e-8, −5.6e-9, −3.5e-10).
- **With zero traction** on the plane it would be 0.43, 0.31 and 0.22, with a signed L estimate of −25 % to −6 %;
  with the element's own traction 3-11 % low.
- **Only the recovery's diagnostic differs** (1.73e-4 against 1.76e-4 on the first mesh): the recovery still cuts at
  the plane.
- The test `test_an_interior_pmc_plane_changes_nothing` checks the first mesh.

### The two-port frame

A sharp-edged copper frame with two gap ports in a closed sphere (`stage1`); its fields are singular at the edges.
Order 1 on three meshes (109k, 835k, 6.5M DOFs), order 2 on two (572k, 4.46M).
- **Order 1 (recovery).** The change of L between meshes against the signed estimates' change is 0.84 and 0.93
  (100 kHz) and 0.87 and 0.95 (10 MHz). Against the limit at the frame's asymptotic order 4/3 (L near convex 90°
  edges; `stage1_order_4_3`), per mesh: signed 1.00, 1.01, 1.03 (100 kHz) and 1.02, 1.03, 1.06 (10 MHz); unsigned
  1.17, 1.16, 1.17 at 10 MHz.
- **Order 2.** The change of L against the signed estimates' change is 0.61 and 0.60 (100 kHz, 10 MHz). That ratio
  weights the coarse mesh. Per mesh, against the order-4/3 limit:
  - signed **0.58, 0.51** at both frequencies;
  - unsigned 0.58, 0.51 (100 kHz) and 0.69, 0.60 (10 MHz).
  - The fine mesh's 0.51 is the one to read: like the probes, the estimate falls under refinement next to the edges.
  - The recovery was closer here: 0.92 and 0.95 on the changes, 0.94 and 1.08 against the limit (coarse mesh).
  - Next to the frame's edges the averaged traction misses the singular field, and the residual problems underestimate
    it.

### A revolved coax (a new curved case)

The shorted coax of `benchmarks/fullwave3d_background.py` (the `solenoid` template's DMRadio-m³ topology without its
coil) revolved to 3D, read out across its annular bottom gap, in a closed sphere of 0.45 m (`coax`). It has curved
cylindrical and toroidal walls and sharp rectangular wall edges. Order 2; 665k and 5.2M DOFs.
- **kR = 0.02 (4 MHz, magnetic):** the change of L is +2.12e-4.
  - The coarse mesh's signed estimate is −1.20e-4 (unsigned 1.28e-4). The 2D twin (converged to 1e-7) puts the
    coarse mesh 2.03e-4 low: **0.59** of it.
  - On the fine mesh the signed estimate is withheld: the parts are 6.0 apart. The magnetic part fell 17 times in
    energy, the electric part (2.7e-4 of the norm) only 3 times. With a factor 4 it had predicted 0.54 of the change.
  - The fine mesh lies 9.5e-6 above the 2D L, and its unsigned estimate is 8.3e-6: the curved geometry's own error is
    about 1e-5, so this mesh's effectivity is not measurable.
  - The recovery predicted 0.43 of the change (record `development.studies.coax`).
- **kR = 0.4 (80 MHz, electric part 12 times the magnetic):** the change of L is −1.44e-3, predicted −1.01e-3, 0.70
  (the recovery: 0.89). Per mesh against the 2D twin: **0.66, 0.59** (3D − 2D = 2.34e-3 and 9.0e-4; the 2D twin's two
  finest meshes differ by 5e-5 at this frequency).
- **Two meshes only**. The coax's rectangular wall edges make L converge at a singular
  rate asymptotically (about h^{4/3}), yet the 4 MHz estimate dropped 15 times between the meshes, an h⁴ rate. Two
  meshes and the 2D twin, with its geometric offset, fit both histories; under the singular one the fine mesh's
  estimate would be a small fraction of its error. A third mesh (41M DOFs) was not run, so the coax's values are
  coarse-mesh values.

### A small CORE-like box

r 0.3-0.9 m, H 0.8 m, wall and slit 0.1 m, rounded lips; 857k and 6.74M DOFs, one uniform refinement (`mini`). Order
2. Radiating exterior, and for c360 also a closed sphere (`--closed`), whose change of L holds no termination error.
- **n1 (one 30° port).**
  - **3 MHz:** the change of L is +3.36e-3, predicted +2.86e-3, **0.85** (recovery: 0.80).
  - **20 MHz:** the parts are within a factor 2 and the signed estimate is withheld. L changes by only +5.1e-4 against
    an unsigned estimate of 5.9e-3, and the recovery's signed estimate had predicted 1.54.
- **c360 (uniform feed), 20 MHz** (22 % of the norm electric, the electric error 12 times the magnetic): the change of
  L is −4.73e-3, predicted −4.05e-3, **0.86** (recovery: 0.61). Closed sphere: −4.77e-3,
  predicted −4.08e-3, **0.86**.
  - **Per mesh, against the 2D twin**. The 3D L lies 4.76e-3 and 9.3e-6 above 2D,
    and the estimates are 4.64e-3 and 5.95e-4. The 2D difference holds the curved geometry's offset, about −3.8e-4
    at 3 MHz. With that offset the effectivities are 0.90 and **1.54**; with the offset scaled with L (−5.2e-4),
    0.87 and 1.13. The fine mesh is at most 1 only if the offset is −6e-4 or larger.
  - **Inconclusive.** In this lossless model the electric part's projection is a lower bound of the error's
    electric energy (given a converged solve), so θ_e ≤ 1. With the estimated ratio r = 20 the signed estimate is
    θ_e(r − 1)/(r − θ_e/θ_m) of the error: at most 1.10 for θ_m ≥ 0.37. The scaled offset's 1.13 needs θ_m ≈ 0.31,
    the 3 MHz offset's 1.54 θ_m ≈ 0.13. More likely the geometric offset at 20 MHz is larger than at 3 MHz.
- **c360, 3 MHz: cancellation.** L changes by only +2.7e-5, the same with the closed sphere, so the termination is
  not the cause.
  - **The fit.** Fitting both frequencies gives a magnetic error of about 1.5e-4 of ω²L and an electric one of about
    1.2e-4 (the magnetic error the same at both frequencies, the electric one, relative to ω²L, growing like ω²).
    Assuming h⁴ convergence, the coarse mesh's signed errors are −2.9e-5 and +5.1e-3.
  - **The estimates.** The residual estimator's parts are 3.0e-4 and 0.8e-4 (3.6 apart), so its signed estimate would be
    7.5 times the change of L. It is withheld (at a factor 4 too; on the fine mesh the parts are 2.2 apart). The
    unsigned estimate, 3.8e-4, is 1.4 times the fitted energy-norm error.
  - **The fit and the window.** By this fit the magnetic part is about 2.0 times over and the electric part 0.67:
    outside the part effectivities the factor-8 window assumes. The fit assumes h⁴
    convergence and a magnetic error independent of frequency, so it is indicative.
  - **The recovery** estimated 2.3e-4 and 0.6e-4 (3.9 apart) and predicted 6.1 times the change, an effect of the
    same cancellation.
- **Against the 2D twin** (exact geometry) the 3D L stays 4e-4 below 2D at 3 MHz on both meshes: the curved
  geometry's approximation, which the indicator does not see.

### Summary at order 2

The signed L estimate against the error of L, per mesh where a reference exists (coarse to fine), else the change of
L under refinement against the change the estimates predict ("pred/obs", which mostly reports the coarse mesh). The
recovery's values at order 2 come from the record's `development` section (the same meshes); for the TEM line and the
probes the recovery's signed estimate at order 2 is not recorded ("—").

| case | f | residual estimator | recovery |
|---|---|---|---|
| TEM line, 5:1 cells and the cut, exact | 100-200 MHz | 0.82-0.94 per mesh | — |
| TEM line, 5:1 cells, exact | 200 MHz | withheld (parts 4.3-4.8 apart) | — |
| TEM line, cubic cells, exact | 100 MHz | withheld (parts 5.4 apart) | — |
| TEM line, cubic cells, exact | 600 MHz | withheld (parts 0.76 apart) | — |
| crack, Richardson limit at rate 1 | 100 MHz | 0.48, 0.43, 0.40, 0.39 | — |
| 270° edge, at rate 4/3 | 100 MHz | 0.62, 0.54, 0.50, 0.48 | — |
| open crack, Im Z (electric) | 100 MHz | 0.50, 0.50, 0.50, 0.49 | |
| two-port frame, at rate 4/3 | 0.1, 10 MHz | 0.58, 0.51 at both (pred/obs 0.61, 0.60) | 0.94, 1.08 (coarse mesh) |
| revolved coax, against the 2D twin | 4 MHz | 0.59, withheld (parts 6.0 apart) | 0.43 (pred/obs) |
| revolved coax, against the 2D twin | 80 MHz | 0.66, 0.59 (pred/obs 0.70) | 0.89 (pred/obs) |
| mini box, c360 (radiating and closed) | 3 MHz | withheld (parts 3.6 and 2.2 apart) | 6.05 (pred/obs) |
| mini box, c360, pred/obs | 20 MHz | 0.86 (closed: 0.86) | 0.61 |
| mini box, c360, against the 2D twin less the 3 MHz offset | 20 MHz | 0.87-0.90, 1.13-1.54 (inconclusive) | |
| mini box, n1, pred/obs | 3 MHz | 0.85 | 0.80 |
| mini box, n1 | 20 MHz | withheld (parts 1.0-1.1 apart) | 1.54 (pred/obs) |

- **Where it is given and measured, the signed estimate was 0.39-0.94 of the error of L.** It is not a bound in
  either direction: that it was never above the error is a property of the validation set, not of the estimator. Against 2D twins, whose difference also holds the geometry's
  offset, the mini box's fine mesh at 20 MHz came out at 1.13-1.54; that is more likely the offset (the box's
  section above).
  - **Next to cracks and re-entrant edges** it is lowest and falls under refinement: the probes, the two-port frame's
    fine mesh (0.51), the coax (0.59 on its fine mesh at 80 MHz). There the error can be up to about 2.7 times the
    estimate.
  - **On smooth fields** it was 0.82-0.94 (the TEM line, exact), and 0.85-0.86 pred/obs on the mini box.
  - **It can lie above the error** where the minor part is underestimated more than the dominant one, for example a
    smooth magnetic error with an electric error at a slit. The factor 8 bounds that at 1.32 for the measured part
    effectivities.
- **The recovery's recorded values spanned 0.43 to 6 here.** It is closer next to sharp edges (the two-port frame),
  where its magnetic part is above the residual problems' (energies: 2.1-3.3 times on the probes, 1.6-1.8 on the
  two-port frame; 0.8-1.0 on the coax and the mini box, 0.25-0.6 on the TEM line).
  - Taking the larger of the two magnetic parts would help there but not on the coax (0.43).
  - Both are reported (`recovery_magnetic_relative_error`); an element-wise combination is open.
- **Withholding.** The signed estimate is withheld where the parts are within a factor 8. There the unsigned estimate
  was 1.0-14 times the error of L in the validation, and it is not a bound: at the crack at 1.2 GHz it fell to 1.03 and was still falling.
- **The unsigned L estimate** equals the signed one in magnitude where one part dominates (the two-port frame, the probes at 100 MHz, the mini n1
  box at 3 MHz). Where the parts partly cancel it is larger (TEM line at 200 MHz: 1.25-1.36).

**Other checks.**
- **Partition.**
  - Order 1: 1, 3 and 4 ranks give the same estimates to 3e-11 (a slow test checks 1 against 4 ranks).
  - Order 2: 1 and 4 ranks agree to 5e-9 on the shorted line and 3e-7 with the interior port;
    `test_order_2_estimate_does_not_depend_on_the_partition` checks 1e-5.
  - **The 3e-7 is the field solve's algebraic error, not the projection's tolerance**.
    It sits in the magnetic part, which has no iterative solve. With `AXF_FW_RTOL=1e-13` (true residual 5e-12
    instead of 4e-9) the difference fell from 1.5e-7 to 3e-10, and the electric part's from 6e-9 to 6e-12.
  - **Why the field matters.** The estimate is a residual of E_h, so an algebraic perturbation of E_h enters it
    directly, and η² is small (5e-9 of the norm there). On a well-resolved model, check that the solve's true residual
    is well below η².
  - **Why the projection's tolerance barely matters.** Its energy is quadratic in the CG error (CG's iterate is the
    energy projection onto the Krylov space): 1e-4 instead of 1e-6 moved the estimates by at most 6e-9 (PCG) and
    4e-5 (GMRES, the conducting line, without the scaling).
  - **The GMRES path** (scaled): 1 and 4 ranks agree to 4e-6 in the electric part and 3e-5 in the signed
    L estimate on the conducting line (`test_order_2_estimate_does_not_depend_on_the_partition` checks 1e-4), and 1,
    3 and 4 ranks to 7e-7 on the copper line. Without the scaling the copper line's electric part spreads by 6.1 %.
- **Purely additive.** Indicator on and off give bit-identical results in every other output
  (`test_indicator_on_and_off_give_identical_results`, the TEM line at order 2; also compared on 587 result leaves of
  the two-port frame at order 1 and 306 of the crack probe at order 2, which differ only in the timings).
- **Solver paths.** The tree-cotree gauged path, plain STRUMPACK and AMS with the guard agree to 1.7e-11 at order 1
  (`test_the_gauged_path_gives_the_same_estimate` to 1e-8). The gauged path is lowest order only.
- **Large models.** The validation set has only small models. The order-1 band of the TEM line and the two-port
  frame (0.85-1.4) is not established for large models with many conductors and a PMC plane.
