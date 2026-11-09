# Mechanics and thermal

A detector magnet pushes on its own winding and on every superconductor in its field, and a cryogenic structure
contracts as it cools and carries heat from warmer stages to the coldest one. axionFEM computes these on the same
mesh as the magnetic solve: the magnetic forces, the elastic displacement and stress of the structure under those
forces and under cooldown, and the steady temperature and heat loads with temperature-dependent conductivity.

## Forces

Every magnetostatic solve reports the Lorentz force on each winding, the Maxwell pressure on superconductors and
iron, the distributed load of each body (`res.forces["distributions"]`) and a closure check that the forces on all
bodies sum to zero (Newton's third law). For a powered azimuthal winding the numerical self-force, which is zero in
theory, is reported as the error bar on its net force.

## Stress and cooldown

The **structure** is every region whose material has a `mechanical` block (Young's modulus, Poisson's ratio and the
total thermal contraction α from room temperature); the **thermal domain** every region with a `thermal` block
(conductivity, a number or a table k(T)). Touching regions are bonded. A `mechanics` section applies the magnetic
loads; `"cooldown": true` adds the contraction, `supports` fix regions and `pressures` load surfaces.

This example adds a G10 pad, a G10 ring post and a steel base plate under the quick-start toroid, then solves the
stresses under the magnetic load and the cooldown, and the heat that flows down the post into the 4.2 K stage:

```python
import axionfem as af

d = af.template("toroid", R=2.0, H=2.0)
d.set(y=0.6)
d.doc["materials"] = {
    "g10": {"kind": "linear", "mechanical": {"E_Pa": 28e9, "nu": 0.21, "alpha": 2.4e-3},
            "thermal": {"k_W_mK": [[4, 0.075], [10, 0.12], [20, 0.18], [50, 0.29], [77, 0.35]]}},
    "steel": {"kind": "linear", "mechanical": {"E_Pa": 193e9, "nu": 0.29, "alpha": 2.97e-3},
              "thermal": {"k_W_mK": [[4, 0.3], [77, 8.0]]}},
}
d.doc["regions"] += [
    {"name": "pad", "material": "g10",                           # a G10 ring between the winding and the sheath
     "shape": {"rect": {"r": [1.5, 1.7], "z": ["-H/2-tw-g", "-H/2-tw"]}}},
    {"name": "post", "material": "g10", "mesh_size": 0.01,       # a G10 ring post under the sheath
     "shape": {"rect": {"r": [1.58, 1.60], "z": ["-H/2-tw-g-ts-0.3", "-H/2-tw-g-ts"]}}},
    {"name": "base", "material": "steel",                        # a steel base plate under the post
     "shape": {"rect": {"r": [1.4, 1.8], "z": ["-H/2-tw-g-ts-0.32", "-H/2-tw-g-ts-0.3"]}}},
]
d.doc["mechanics"] = {"cooldown": True}           # magnetic loads plus the thermal contraction of a cooldown
res = d.solve("runs/toroid_support", verbose=0)
for name, v in res.mechanics["regions"].items():
    print(f"{name:8s} max |u| = {v['u_max_m'] * 1e3:7.3f} mm, max von Mises = {v['von_mises_max_Pa'] / 1e6:7.2f} MPa")

h = af.Design(d.doc)                              # the same structure, thermal analysis only
del h.doc["mechanics"]
h.doc["thermal"] = {"fixed": [{"region": "sheath", "T_K": 4.2}, {"region": "base", "T_K": 50.0}]}
rt = h.solve("runs/toroid_support_thermal", verbose=0)
print(f"heat load on the 4.2 K stage: {rt.thermal['checks']['heat_removed_W']:.4f} W")
```

- **Results.** `res.mechanics` holds the maximum displacement and von Mises stress per region, the applied loads and
  checks; `res.sample_mech(r, z)` samples displacement, stress components (`sigma_hoop`, `sigma_zz`, ...), von Mises
  stress and temperature at points.
- **Material data.** The built-in materials (`nb`, `copper`, `iron`, `winding`) carry documented defaults; real
  windings are strongly orthotropic and cryogenic data vary with purity and treatment, so give measured values
  before quoting engineering numbers ([design format](../design_format.md)).
- **Conduction** with k(T) is solved by Picard iteration and checked against the exact Kirchhoff solution of a
  uniform post ([validation](../validation/mechanics_thermal.md)). Every conducting part needs a path to a fixed
  temperature; an isolated part is an error that says which one.
- **3D.** The same sections work in 3D designs with an MPI build ([3D validation](../validation/mech3d.md)).

## Torsional dynamics (beta)

`res.torsion(...)` is an axisymmetric (m = 0) torsional modal and harmonic analysis of the structure: modes,
response to azimuthal loads and the mechanical mobility, with a structural loss factor.
`axionfem.torsion_waveguide` gives the exact torsional branches of an infinite tube for the high-frequency limit. Both
are beta ([validation](../validation/torsion.md)).

## Next steps

- [Walkthrough notebook](../../examples/mechanics_thermal.ipynb): forces on the winding against thin-winding
  estimates, support stresses, cooldown and the heat load with k(T).
- [Lumped-element detectors](lumped_element.md) for the magnetic design itself.
