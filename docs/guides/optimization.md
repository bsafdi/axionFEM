# Shape optimization

In lumped-element experiments, changing a detector's dimensions changes its
signal coupling, inductance and stored energy. In this guide we compute shape
sensitivities and run a bounded search. A shape sensitivity measures the change in one of these
quantities per change in a design parameter. axionFEM provides optimizers that use those
derivatives to search for a local maximum of sensitivity or a local minimum
of stored energy within chosen bounds.

For an objective $J$, parameter $p$, field unknowns $y$ and residual $R$, the
discrete adjoint gives

$$
\frac{dJ}{dp} = \frac{\partial J}{\partial p}
  + \lambda\cdot\frac{\partial R}{\partial p},
\qquad S\lambda = -\frac{\partial J}{\partial y}.
$$

Here $S$ is the symmetric tangent operator and $\lambda$ is the adjoint field.
The mesh moves with the geometry while keeping its connectivity. The adjoint
supports axisymmetric magnetoquasistatic designs with linear materials; 3D,
full-wave results and RF loss budgets are not differentiated.

For a solenoid, `x` is the coax inner radius divided by the winding inner
radius. With SciPy installed, we compute derivatives of the scan figure of merit to
quantify its dependence on `x`:

```python
import axionfem as af

design = af.template("solenoid")
result = design.solve()
s = result.sensitivities(params=["x"], objectives=["F"])
print(s["gradient"]["F"]["x"], s["relative"]["F"]["x"])
```

The first number is the derivative in SI units; the second is the fractional
change in the objective per fractional change in `x`. These are derivatives
on the current mesh: mesh convergence should be established before optimizing.

We now run a bounded search for a local minimum of the stored energy the solenoid needs to
reach the scan requirement:

```python
opt = af.optimize(design, objective="W_at_req:optimistic",
                  params={"x": (0.3, 0.8)}, max_iter=30)
print(opt["converged"], opt["message"])
print(opt["parameters"], opt["value"])
```

The value is in joules, after scaling the field to the selected
[scan requirement](../../examples/lumped_element_demo.ipynb) (the `optimistic` benchmark is an illustrative
preset, and the default scan target an illustrative choice). `af.objectives()` lists the other
objectives, among them the coupling (`"F"` with `sense="max"`) and the peak fields at the requirement. `converged` and
`message` describe the search termination; `design` contains the returned design document and
`history` records the iterations. Field caps are only reported unless added
as explicit `constraints` on `sc_surface_at_req` or `B_peak_at_req`; sharp
superconductor corners should be rounded before optimizing a peak-field constraint.

For sensitivities involving nonlinear iron or changes to mesh connectivity,
we pass `method="fd_remesh"` to `result.sensitivities`. The [API reference](../api.md)
lists arguments and return values.
