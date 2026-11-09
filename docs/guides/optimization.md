# Reach, sensitivities and optimization

In lumped-element experiments, a detector's dimensions set its signal coupling, inductance and stored energy, and
through them the reach of a scan and the field it needs. This guide computes the reach of a design, its shape
sensitivities, and runs local and global searches over its parameters.

## Reach

`Results.reach` evaluates the expected 95 % reach g₉₅(m_a) of the optimal resonant scan for a noise benchmark. By
default the field is scaled so that F = F_req, the amplitude that reaches the QCD target; `amplitude="as_designed"`
keeps the ampere-turns of the design:

```python
import axionfem as af

d = af.template("toroid", R=2.0, H=2.0)
d.set(y=0.6)
res = d.solve("runs/toroid", verbose=0)
r = res.reach("optimistic")
print(r["m_eV"][0], r["g95_GeV"][0])                  # arrays over the scan band
print(r["band"]["f_min_Hz"], r["band"]["live_time_yr"])
```

`axionfem.scan.reach(Psi, L, benchmark)` does the same from Ψ and L directly, and `axionfem.scan.reach_fullwave` from
a full-wave sweep ([Full wave](fullwave.md#reach-from-a-full-wave-sweep)). The benchmark's required field and stored
energy at the requirement are in `res.benchmark(name)`
([signal, noise and optimal scan](../physics/scan_and_signal.md)).

## Shape sensitivities

A shape sensitivity is the change of a figure of merit per change of a design parameter. For an objective $J$,
parameter $p$, field unknowns $y$ and residual $R$, the discrete adjoint gives

$$
\frac{dJ}{dp} = \frac{\partial J}{\partial p}
  + \lambda\cdot\frac{\partial R}{\partial p},
\qquad S\lambda = -\frac{\partial J}{\partial y}.
$$

Here $S$ is the symmetric tangent operator and $\lambda$ is the adjoint field. The mesh moves with the geometry while
keeping its connectivity. The adjoint supports axisymmetric magnetoquasistatic designs with linear materials; 3D and
full-wave results and RF loss budgets are not differentiated.

For a solenoid, `x` is the coax inner radius divided by the winding inner radius. With SciPy installed, we compute
derivatives of the scan figure of merit to quantify its dependence on `x`:

```python
design = af.template("solenoid")
result = design.solve("runs/solenoid", verbose=0)
s = result.sensitivities(params=["x"], objectives=["F"])
print(s["gradient"]["F"]["x"], s["relative"]["F"]["x"])
```

The first number is the derivative in SI units; the second is the fractional change in the objective per fractional
change in `x`. These are derivatives on the current mesh: establish mesh convergence before optimizing. For
sensitivities with nonlinear iron or changes of mesh connectivity, pass `method="fd_remesh"` to
`result.sensitivities`. `af.objectives()` lists the quantities that can be differentiated and optimized.

## Local optimization

We run a bounded search for a local minimum of the stored energy the solenoid needs to reach the scan requirement:

```python
opt = af.optimize(design, objective="W_at_req:optimistic",
                  params={"x": (0.3, 0.8)}, max_iter=30)
print(opt["converged"], opt["message"])
print(opt["parameters"], opt["value"])
```

The value is in joules, after scaling the field to the selected scan requirement. Other physical objectives are the
coupling (`"F"` or `"g_factor"` with `sense="max"`) and the peak fields at the requirement; `af.objectives()` lists
them, and the [optimization reference](../optimization.md#optimization) gives useful formulations (the cost metric
is also available, as a scaling metric for comparing designs). `converged` and `message` describe how the search ended;
`design` contains the returned design document and `history` records the iterations. Field caps are only reported
unless added as explicit `constraints` on `sc_surface_at_req` or `B_peak_at_req`; round sharp superconductor corners
before optimizing a peak-field constraint.

## Global search

`af.global_optimize` searches a box of parameters by differential evolution, with constraints enforced by the
feasibility rule and an optional gradient-based polish of the best point. Parameters that span decades can be
searched in log space (`log=[...]` or a `(low, high, "log")` bound), and `workers` runs candidates in parallel. Each
candidate is a full solve, so budget `popsize` and `maxiter` accordingly; the [API reference](../api.md) lists every
argument, and [sensitivities and optimization](../optimization.md) describes the methods and their validation.

## Next steps

- [Walkthrough notebook](../../examples/reach_optimization.ipynb): reach, required field, size under a peak-field
  limit, sensitivities and a small optimization of a physical figure of merit.
- [Validation of the shape sensitivities](../validation/adjoint.md).
