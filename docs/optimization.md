# Shape sensitivities and gradient-based optimization

Every axisymmetric design carries named `parameters` (numbers or expressions; see
[design_format.md](design_format.md)). axionFEM differentiates the figures of merit with
respect to those parameters and uses the gradients to optimize a design.

```python
import axionfem as af

d   = af.template("solenoid")
res = d.solve()
s   = res.sensitivities(params=["R", "H", "x"], objectives=["Psi", "L_p", "F", "W_at_req"])
s["gradient"]["F"]["x"]      # dF/dx in SI units
s["relative"]["F"]["x"]      # d log F / d log x: the % change in F per % change in x

opt = af.optimize(d, objective="g_factor", sense="max", params={"x": (0.3, 0.8)})
af.Design(opt["design"]).solve()
```

`Design.gradient(...)` solves and returns the same dictionary in one call;
`Design.optimize(...)` is a shorthand for `af.optimize`. The MCP server exposes
`sensitivities`, `list_objectives`, `optimize_design` and `optimize_global`. A global search by
differential evolution is `af.optimize(..., method="global")` (see [below](#global-optimization-differential-evolution)).

## What is differentiated

| quantity | meaning |
|---|---|
| `Psi` | signal coupling Ψ = (1/I) ∫A_I·B_lab dV [H T m²] |
| `L_p` | pickup inductance [H] |
| `W_mag` | stored magnetic energy of the design as drawn [J] |
| `F` | scan figure of merit Ψ⁴/L_p² [T⁴ H² m⁸] |
| `g_factor` | g = Ψ²/(L_p W R²) in natural units |
| `A_cold`, `iron_mass`, `device_radius` | geometry of the cold envelope, iron mass, outer radius |
| `F_over_F_req`, `amplitude_scale`, `W_at_req` | relative to a noise benchmark: F/F_req, the field scale s that reaches F_req, and s²W |
| `cost`, `cost_optimal` | the cost metric (a scaling metric for comparing designs, not a construction estimate) at the requirement and at the cost-optimal size |
| `sc_surface_peak`, `B_peak` | peak superconductor surface field (away from sharp corners) and peak \|B\| [T] |
| `sc_surface_at_req`, `B_peak_at_req` | the same peaks after scaling the amplitude to F_req |

`af.objectives()` returns this list. Quantities that refer to a noise benchmark accept one by
name: `"W_at_req:optimistic"`, `"F_over_F_req:conservative_solenoid"`; without a name the design's first
benchmark is used.

`Psi`, `L_p`, `W_mag`, `A_cold`, `iron_mass`, `device_radius` and the two peak fields are
*primitive*: they are read from the solved model. The rest are closed-form functions of them (the
same formulas as the solver's post-processing, checked against it at every call:
`diagnostics.checks.closed_form_vs_post_processing`), differentiated by the chain rule. The
`primitives` entry of the result carries their values and derivatives too.

## Method

The discrete problem is `R(y, p) = 0` with `y` the reduced unknowns of the poloidal (ψ = rA_φ)
problem — the free degrees of freedom followed by the flux function of every unknown
superconducting body — and `p` a design parameter. For a figure of merit `J(y, p)`,

    dJ/dp = ∂J/∂p + λ·∂R/∂p ,      S λ = −∂J/∂y ,

with `S` the symmetric tangent operator (stiffness with the fluxoid ties, the exterior DtN map and
the persistent-circuit constraints). One factorization per poloidal problem serves every objective,
with the solver and the fill-reducing ordering the forward solve chose (Eigen LDLT with AMD or
METIS, or STRUMPACK above `AXF_STRUMPACK_MIN_DOFS`); `∂J/∂y` is analytic (`Psi` is a linear
functional of A_φ, `L_p` and `W_mag` are quadratic).

The partial derivatives at frozen `y` come from central differences of **assembly and
post-processing only** on a *morphed* mesh: for `p ± δ` the geometry is rebuilt from the design,
and the nodes of the nominal mesh are carried onto it, keeping the topology and the DOF numbering
(`cpp/src/morph.cpp`):

* every shape and path decomposes into boundary primitives (segments and circular arcs) in an order
  that does not depend on the numbers, so primitive *k* of the nominal geometry corresponds to
  primitive *k* of the perturbed one, and a node at parameter *t* of one sits at parameter *t* of the
  other (a boolean contributes its children's primitives, which contain the boundary of the result);
* a node that lies on several primitives — a corner, or a crossing created by overlapping
  regions — is placed by least squares over their normal displacements, i.e. on the intersection of
  the perturbed primitives;
* nodes on the outer semicircle scale with the domain radius (which the mesher derives from the
  device extent, so it moves too) about the centre of that semicircle, (0, `mesh.domain_center_z`);
  axis nodes keep r = 0;
* the interior follows a component-wise harmonic (Laplace) extension. A uniform dilation of the
  design is therefore reproduced exactly, because r and z are harmonic.

A sensitivity evaluation post-processes only what the figures of merit need: the peak fields, the
forces (with their self-force solves), the mechanics, the electrostatics and the full-wave sweep are
skipped (`SolveOptions::minimal_post`), so one perturbed evaluation costs about one solve.

The toroidal (G = rH_φ) problems are *re-solved* at every perturbation instead of being
differentiated: their finite-element part lives inside the winding or port region and the rest is a
handful of cavity levels, so they cost almost nothing, and their solution then enters `∂J/∂p`
exactly. The engine therefore spends, per parameter, two assemblies plus two cheap toroidal solves,
and no large factorization at all.

One gradient over *n* parameters costs about `2n` assemblies regardless of how many objectives are
asked for. Measured on the templates (7-8 parameters, 5 objectives, default meshes): 8-15 s for the
adjoint, 20-33 s for the same finite differences on the morphed mesh, 45-77 s with remeshing — 2.2-2.5x
and 5-6x. In 2D the saving is modest because assembly and setup, not the factorization, dominate; what
the adjoint really buys is the absence of remeshing noise (which limits remeshing finite differences
to ~1e-3 relative) and a formulation that carries over to 3D, where the factorization dominates.

### Other methods

`method="fd_morph"` takes central differences of complete solves on the morphed mesh: the same
discrete derivative as the adjoint, used to validate it. `method="fd_remesh"` meshes every perturbed
design from scratch; it is slower and noisy (a new mesh changes the figures of merit by ~1e-5
relative, which the finite difference amplifies by 1/δ), but it is the only method that works for

* **nonlinear iron** (the adjoint refuses it; the tangent of the Newton problem is available but the
  iron's secant permeability in the RF problem is not differentiated yet), and
* parameters that **change the mesh topology**, e.g. switching a region on through
  `"enabled": "iron_t"` or changing the number of polygon points — morphing reports a clear error.

## Conventions and pitfalls

* Derivatives are of the **discrete** figures of merit at a fixed mesh topology. Parameters that only
  control the element sizes (`mesh.size`, `mesh.surface_size`, `region.mesh_size`, `mesh.order`)
  therefore have no derivative, and a parameter that *also* sets them (every template's `mesh.size`
  is an expression in its lengths) gets a warning saying so: the geometry part of the derivative is
  complete, the change in discretization is not included. `mesh.domain_radius` is the exception —
  the morph moves the outer boundary with it. Converge the mesh first (`solve(order=3)`, finer
  `mesh.size`): a gradient is only as trustworthy as the quantity it differentiates.
* Differentiating a parameter that other parameters are defined from (e.g. `R` with
  `tw = "0.01*R"`) gives the **total** derivative, with every dependent expression following.
  Differentiating a parameter that is itself an expression replaces that expression by its value for
  the two perturbed designs, which is the partial derivative at fixed parents.
* A parameter whose value is zero gets an absolute step and a warning: the derivative is then taken
  across zero, which is wrong for a one-sided quantity such as a fillet radius or a thickness.
* `cost_optimal` is scale invariant, so its derivative with respect to a single size parameter is a
  small difference of large contributions (for the toroid dC*/dR and dC*/dH cancel to 1e-5 of each
  other). The descent direction is fine (the validation's gradient-based run lands on the derivative-free
  optimum), but those single numbers carry ~1e-2 relative accuracy rather than 1e-6.
* **Zero derivatives are usually physics.** A copper coax is transparent to the static field, so the
  solenoid's `W_mag` does not depend on `x`; the pickup inductance depends neither on the
  ampere-turns nor on the winding geometry (a stranded coil is transparent to the RF field); the
  quantities at the requirement are invariant under a rescaling of the sources (the amplitude is scaled to
  reach F_req first). The remaining value is round-off, and the validation tables mark those rows.
* The peak-field quantities are maxima over the mesh: piecewise smooth, with a derivative that jumps
  when the location of the maximum moves. They are computed by finite differences on the morphed
  mesh (which also forces the full post-processing of every perturbed evaluation, so they are
  slower), and they are noisier than the smooth quantities. Sharp superconductor corners make the
  peak itself mesh-dependent — round them (`fillet`) before optimizing against a cap.
* Full-wave (`fullwave`) results, 3D designs and the RF loss budget are not differentiated.
* The step `step` is relative (1e-4 by default); `steps={"R": 1e-3}` overrides single parameters in
  absolute units. Smaller is not better: the assembled residual and the frozen functionals are
  differenced, so a step that moves a feature by a micrometre leaves only the last few digits. For
  parameters that set a millimetre-scale feature (a wall thickness, a gap) `step=1e-3` is the better
  default; the validation page shows the adjoint is stable there while the finite differences are not.

## Optimization

```python
out = af.optimize(design, objective="F", sense="max",
                  params={"x": (0.3, 0.8), "H": (1.0, 5.0)},       # bounds
                  constraints=[{"quantity": "W_mag", "type": "<=", "value": 1e8}],
                  max_iter=30)
```

The search runs in units of the starting values, so all variables are O(1); scipy's L-BFGS-B is used
without constraints and SLSQP with them. Every iterate is one mesh, one solve and one gradient, and
the history (parameters, all requested quantities, field-cap status) is returned together with the
optimized design document and a note if the last iterate is not the best one seen. A shape that
fails to mesh or solve is reported as infeasible instead of stopping the search.

Field caps (`cost.sc_wall_cap`, `cost.B_peak_cap`) are reported for every iterate under `caps`; to
enforce one, add a constraint on `sc_surface_at_req` or `B_peak_at_req` and read the caveat above.

Useful formulations:

* maximum coupling at a fixed magnet: `objective="F"`, `sense="max"`, with a constraint
  `W_mag <= W0`;
* least field energy that reaches the requirement: `objective="W_at_req"` over shape parameters at a fixed
  size (a larger magnet of the same shape always needs less energy, so bound or fix the size);
* a peak-field limit at the operating point: a constraint on `B_peak_at_req` or `sc_surface_at_req`.

The cost metric (`cost`, `cost_optimal`) can be an objective too; it is a scaling metric for comparing designs,
not a construction estimate.

The derivative-free `optimize` MCP tool (bounded Powell) remains available for designs the adjoint
does not cover.

## Global optimization (differential evolution)

```python
out = af.optimize(design, method="global", objective="g_factor", sense="max",
                  params={"H": (0.5, 8.0, "log"), "y": (0.2, 0.9)},       # bounds; "log" = log-scaled search
                  constraints=[("sc_surface_at_req", "<=", 0.095)],
                  seed=1, popsize=10, maxiter=40, workers=16, polish=True)
```

`af.optimize(..., method="global")` (also `af.global_optimize`, `Design.optimize`, and the MCP tool
`optimize_global`) searches the whole box with scipy's `differential_evolution`; every other `method`
still selects the gradient-based optimizer above. Each evaluation meshes and solves the design;
requirement quantities (`W_at_req`, `sc_surface_at_req`, `cost`, `cost_optimal`, ...) are read at
F = F_req exactly as for the local optimizer. For linear designs that is the solver's exact amplitude
rescaling (one solve). When the rescaling is not exact — **nonlinear iron**, or fixed sources such as a
background field — the parameter `amplitude` (default `"NI"`) is solved for F = F_req of the objective's
benchmark (log-secant on F, to `amplitude_tol` = 1e-6, typically 3 solves), so `W_at_req` is the energy at the
requirement with saturated iron, not a linear estimate.

* **Infeasible points instead of crashes.** A shape that fails to mesh or solve, a nonlinear solve whose
  Newton iteration did not converge, or a point rejected by `precheck(params)` (cheap geometric rules:
  return `None`/`True` to accept, `False` or a reason string to reject) is a failed point. With a process
  pool (`workers` > 1) so are an evaluation over `timeout` seconds (from submission for one still running,
  its own run time for one that finished) and a worker that dies; the points that
  were running with a dead worker are re-run one at a time, so the failure is charged only to the point
  that caused it. In serial mode (`workers=1`) `timeout` is ignored (with a warning) and a hard crash of
  the solver ends the run. Constraints use the feasibility rule: any feasible point beats any infeasible
  one, infeasible points are ranked by their total relative violation, failed points rank last. No penalty
  weight to tune.
* **Configuration errors stop the run.** An unknown benchmark, `cost_optimal` on an iron design, a
  nonlinear design without an amplitude parameter, `NI` optimized while it is the amplitude, requirement
  quantities on different benchmarks of a nonlinear design, or a callable that the workers cannot load
  raise `ValueError` at the first evaluation that shows them, instead of failing every point.
* **Convergence.** scipy's own test is disabled (it is meaningless with the ranks of infeasible points);
  the search stops when the whole population is feasible and `std(f) <= atol + tol |mean(f)|`, after
  `maxiter` generations, or when `callback(generation, best)` returns True. `message` says which.
  `max_time` is a soft limit, checked after each generation: the search stops, the Powell polish stops,
  and the adjoint polish (which cannot be interrupted) runs only if `7 (polish_max_iter + 1)` evaluations
  of the DE's mean duration (several solves per iteration) fit in the time left.
* **Deterministic for a seed.** The population, mutation and selection come from `seed`; every
  evaluation (in the calling process, in pool workers, or through a `workers` map) and the polish run on
  `threads` OpenMP threads (default 1; the caller's thread count is restored afterwards), so that the
  solver's parallel sums do not depend on the thread count. Serial runs, pool runs and runs from a
  multi-threaded caller give the same history, polish included (tested).
* **Parallel.** `workers=N` evaluates each generation on a pool of N processes (`-1`: the CPUs this
  process may use, divided by `threads`); a callable `workers(fn, tasks)` (any map) sends the evaluations
  elsewhere, e.g. to a SLURM array. Scripts that use a pool need an `if __name__ == "__main__":` guard,
  and callable objectives must be importable by the workers (module level of an importable module, not
  a notebook or `python -c`).
* **Polish.** The best DE point is refined with the adjoint-gradient optimizer when the design is linear
  without fixed sources (no amplitude root was needed), the objective and constraints are named
  quantities, no `precheck` is given and `order` is 0; otherwise with bounded Powell (`polish="adjoint"`,
  `"powell"` or `False` to choose; `"adjoint"` falls back to Powell if the adjoint optimizer fails or the
  point needed the amplitude root, and is refused together with a `precheck`). scipy's own
  finite-difference polish is never used. The polished point is kept only if it is feasible and better.
* **Output.** The structure of the local optimizer (`parameters`, `values`, `caps`, `constraints`,
  `design`, `history` with every evaluation, its `stage`, `generation`, `energy` and `feasible`), plus `de`
  (generations, evaluations, population, convergence, scipy's message), `polish`, `de_best`, `found_by`
  and `feasible`. `evaluations` counts the history entries (DE and Powell points, adjoint iterates,
  precheck rejections), `solves` the FE solves (with the amplitude-root and adjoint-polish solves).
  `values` are keyed by quantity name; a callable by its `__name__`, with `#2`, `#3` ... appended when
  that key is taken (two lambdas, or a callable named like a quantity). The population has
  `max(5, popsize * len(params))` members. The default `workdir` is a new directory per run.

**Known limitations.**
* Where the amplitude root runs (a requirement quantity or any callable on a nonlinear design or one with
  fixed sources), every quantity, also `B_peak`, `W_mag` and `sc_surface_peak`, and every callable is
  read at the root amplitude; on a linear design without fixed sources, non-requirement quantities and
  callables see the design's own `NI`. A constraint such as `("B_peak", "<=", 6)` therefore means
  different things in the two cases: use the `*_at_req` quantities for limits at the operating point.
  A callable on a nonlinear design needs the amplitude parameter and costs the ~3 solves of the root.
* Nothing is written before the run returns (no checkpoint); `max_time` is not checked inside a
  generation or during the initial population.
* A `timeout` restarts the whole pool: the other points that were running are re-run from scratch.
* A feasible objective must satisfy |f| < 1e99 (larger values are failed points; rescale).

**Global or local?** The local optimizer costs a few solves plus one adjoint gradient per iteration
and converges to the nearest optimum from the start values; use it to refine a design, for many
parameters, or when the landscape is known to be smooth and unimodal. DE needs no start and no
gradient and copes with several minima, with parameters that switch regions on and off, with
nonlinear iron and with failed shapes, but costs about `popsize * len(params) * (generations + 1)`
evaluations (times ~3 solves with nonlinear iron): 30-300 evaluations for 1-2 parameters, thousands
for 6-8. In practice: DE on a coarse mesh over
wide bounds, then the local optimizer (or the built-in polish) on the production mesh.

### Validation: known optima from no initial guess

`benchmarks/global_opt_validation.py` (record `benchmarks/records/global_opt_validation.json`, which stores the
commit that produced it) searches the three
cases of the adjoint validation over their whole bounds, with the same designs and meshes, DE population
8-16, seed 20261029, 16 workers:

| case | DE alone (generations) | DE + polish | adjoint optimum | FE solves (DE + polish) | wall time |
|---|---|---|---|---|---|
| toroid, H = R: max g over y in [0.25, 0.9] | y = 0.5693 (4) | y = 0.56748 | 0.5675 | 38 + 3 | 30 s |
| long solenoid, H/R = 8: max g over x in [0.3, 0.75] | x = 0.5211 (3) | x = 0.53138 | 0.5314 (long limit 0.534) | 30 + 4 | 2.2 min |
| current-sheet solenoid: min of the cost metric at the cost-optimal size over H, x (reference readout benchmark) | H −4.1 %, x −0.4 % from the adjoint optimum (4) | H +0.02 %, x −0.002 % from it | (not quoted) | 80 + 5 | 3.9 min |

For the cost case the table gives the distances from the adjoint optimum, not the optimum itself: the cost metric is a
scaling metric, used here only as a smooth objective with an interior optimum, evaluated for a reference readout
benchmark with round values (Q = 10⁶, T = 0.1 K, η_A = 1; the optimum's shape does not depend on the benchmark). DE alone stops on tol = 0.01 after 3-4
generations, with the parameters 0.3 %, 1.9 % and 4.1 % (H/R; x 0.4 %) from the optimum and the objective within
0.02 %. The adjoint polish closes the rest in 1-4 iterations.
The long solenoid's polish is the expensive step here, with 82 s against 52 s of DE. `tests/test_global_opt.py`
repeats the first and third case on coarse meshes.

## Validation

[validation/adjoint.md](validation/adjoint.md): source-scaling and dilation identities that are
exact in closed form, the adjoint against finite differences on the same mesh and against
remeshing finite differences at several step sizes, and three worked optimizations against known
optima (the long-limit hole fraction of a toroid, the coaxial-solenoid x = 0.534, and the optimum of the cost
metric of a finite current-sheet solenoid, a smooth objective with an interior optimum, where the adjoint run
lands on a derivative-free search's optimum to ~3 digits using 4 solves instead of 33). Regression tests are in `tests/test_adjoint.py`.
