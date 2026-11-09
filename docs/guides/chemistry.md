# Chemistry (beta)

Electrochemical cells and reactors are described by reaction kinetics and by the current that flows through the
electrolyte between electrodes. In this guide we use `axionfem.chem`, which has two parts: 0D stiff chemical kinetics
(a kinetics kernel, reactors, equilibrium and implicit time integrators, in `axionfem.chem.kinetics`) and the steady
current distribution in an electrolyte (primary and secondary, 2D planar or axisymmetric, 3D on MPI ranks, and a 1D
porous electrode). The module is beta: its interface may change, and its verification covers manufactured and
closed-form solutions, stiff test problems and comparisons with Cantera and DOLFINx, but no published benchmark value
or experiment.

## What is available

| feature | entry point | build |
|---|---|---|
| kinetics kernel: mass-action rates and their analytic Jacobians | `kinetics.mechanism`, `kinetics.rates` | every build |
| reactors: batch, CSTR, semibatch, constant-pressure gas, adiabatic liquid, plug flow; equilibrium (index-1 DAE) | `kinetics.batch`, `cstr`, `semibatch`, `const_p_gas`, `adiabatic_liquid`, `pfr`, `equilibrium` | every build |
| implicit integrators: BDF1..5, ESDIRK3(2), variable-order BDF, error control, dense output, events | `kinetics.adaptive`, `grid_bdf`, `grid_onestep`, `bdf_step`, `onestep_step` | every build |
| current distribution, 2D planar or axisymmetric, P1/P2 | `chem.CurrentDistribution` | every build, one process |
| current distribution, 3D tetrahedra, P1/P2 | `chem.CurrentDistribution3D` | MPI build, distributed |
| 1D porous electrode (linearized Butler-Volmer only) | `chem.PorousElectrode` | every build |
| mesh families (rectangle, annulus sector, graded disk; 3D box and sectors) | `chem.meshes`, `chem.meshes3d` | every build |

Problems are set up in Python; there is no design-file, viewer or MCP interface. `help(...)` on each entry point
documents the arguments.

## Kinetics and a parallel-plate cell

We integrate a first-order reaction in a batch reactor with the variable-order BDF method and compare it with its
closed form, then solve the secondary current distribution of a parallel-plate cell with linearized Butler-Volmer
kinetics, whose current density has a closed form (ohmic resistance plus two kinetic resistances):

```python
import numpy as np
from axionfem.chem import CurrentDistribution, Reaction, meshes, FARADAY, GAS_CONSTANT
from axionfem.chem import kinetics as K

# 1. Stiff kinetics: A -> B with k = 0.7 1/s in a batch reactor, variable-order BDF with error control
mech = K.mechanism({"species": ["A", "B"],
                    "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": 0.7}]})
tr = K.adaptive(K.batch(mech), "vsvo", 0.0, [2.0, 0.0], t_out=[1.0, 5.0], rtol=1e-8, atol=1e-10)
t, y = np.asarray(tr.t), np.asarray(tr.y)
print("steps:", tr.stats["steps"], " max |c_A - 2 exp(-0.7 t)|:", np.max(np.abs(y[:, 0] - 2 * np.exp(-0.7 * t))),
      " max |c_A + c_B - 2|:", np.max(np.abs(y.sum(axis=1) - 2)))

# 2. Secondary current distribution: a parallel-plate cell (gap L, height Hc) with linearized Butler-Volmer kinetics
L, Hc, kappa, i0, V = 0.01, 0.005, 1.0, 10.0, 0.05     # m, m, S/m, A/m^2, V
f = FARADAY / (GAS_CONSTANT * 298.15)
lin = Reaction("linear", i0)                            # i = i0 (alpha_a + alpha_c) f eta, alpha_a = alpha_c = 0.5
cd = CurrentDistribution(meshes.rect(16, 4, x1=L, y1=Hc), order=2)   # attributes: 1 x = 0, 2 x = L, 3-4 walls
cd.set_conductivity({1: kappa})
cd.add_electrode("anode", [1], reactions=[lin], phi_s=V)
cd.add_electrode("cathode", [2], reactions=[lin], phi_s=0.0)
rep = cd.solve()                                        # walls 3 and 4 are insulating by default
q = cd.boundary_quadrature([1], cd.boundary_degree)     # anode quadrature points: weights w, potential phi
i_num = np.sum(np.asarray(q["w"]) * i0 * f * (V - np.asarray(q["phi"]))) / Hc
i_exact = V / (L / kappa + 2 / (i0 * f))                # ohmic drop plus two kinetic resistances
print("Newton:", rep["converged"], rep["iterations"], " i =", i_num, "A/m^2  exact", i_exact,
      " rel. error", abs(i_num / i_exact - 1))
```

This prints (the last digits can differ between builds), after the one-time beta warning on standard error:

```text
steps: 118  max |c_A - 2 exp(-0.7 t)|: 4.491919924909382e-08  max |c_A + c_B - 2|: 5.0182080713057076e-14
Newton: True 2  i = 3.3028336846435886 A/m^2  exact 3.302833684643331  rel. error 7.793765632868599e-14
```

The potential is linear across the cell, so the P2 solution reproduces the closed-form current to round-off; the
kinetics error is set by the tolerances, and the sum of the concentrations is conserved to round-off.

## Conventions

- SI units throughout; anodic currents are positive; the overpotential is `eta = phi_s - phi - E_eq`. Axisymmetric
  meshes live in the meridian `(r, z)` and every integral carries `r` ("per radian"), while galvanostatic currents
  are 3D totals.
- `add_electrode(name, attributes, reactions=[...], phi_s=...)` adds a kinetic electrode (potentiostatic, or
  galvanostatic with `galvanostatic=True, current=I`), `primary=True, E_eq=...` a primary one; boundaries without an
  electrode are insulating.
- `solve()` raises `chem.ConvergenceError` when Newton does not converge; large cell voltages need continuation
  (`initial="primary"` in 2D, or `solve_continuation`).

`CurrentDistribution3D` needs the MPI build; the kinetics, the 2D current distribution and the porous electrode run in
every build. Species transport, migration, double-layer charging, transient electrochemistry, impedance, battery
models, temperature-dependent equilibrium constants and coupling to the fluids module are not implemented, and the
porous electrode takes linearized Butler-Volmer kinetics only.

The [walkthrough](../../examples/chemistry_demo.ipynb) solves primary, secondary, 3D and porous-electrode problems
against closed forms. The [formulation notes](../physics/chemistry.md) give the details of the Python interface, the
equations, the time integrators and the solvers, and the [validation page](../validation/chemistry.md) lists the
verification cases.
