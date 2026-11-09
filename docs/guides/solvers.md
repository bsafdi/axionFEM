# Designs and solvers

Lumped-element detectors and cavity haloscopes search for axion dark matter.
In this guide we describe axionFEM designs and solvers.

A design is a JSON document describing geometry, materials, magnet windings,
readout and mesh settings in SI units. Numeric fields accept expressions over
named `parameters`; later regions replace earlier ones where they overlap,
and unassigned space is air.

## Design file

We write, load and solve a complete JSON design, `toroid.json`, for a toroidal
magnet with a superconducting pickup sheath to obtain coupling and inductance.

```python
import axionfem as af

af.template("toroid", R=2.0, H=2.0).save("toroid.json")
design = af.load("toroid.json")
print(design.validate())
result = design.solve("runs/toroid")
print(result.Psi, result.L_p)
```

Validation reports `ok: True` and evaluated parameters, or identifies an invalid
field. The solve returns coupling in $\mathrm{H\,T\,m^2}$ and inductance in
henries; we check mesh refinement and residuals before using them.

| Top-level keys | Purpose |
|---|---|
| `schema`, `name`, `symmetry` | Format, name and geometry dimension |
| `parameters` | Named numbers and expressions |
| `regions`, `curves`, `materials` | Volumes, sheets and material properties |
| `windings`, `circuits`, `background` | Static-field sources |
| `superconductors`, `pickup` | Constraints and readout |
| `mesh` | Element sizes and order |
| `fullwave`, `cavity` | Driven frequency sweeps or cavity modes |
| `scan`, `benchmarks`, `cost` | Sensitivity assumptions and cost metric |

## Field models

axionFEM's magnetoquasistatic (MQS) solver neglects displacement current and is
appropriate when the detector is small compared with the signal wavelength.
Its full-wave solver retains displacement current to resolve radiation,
resonances and frequency-dependent coupling.
With time dependence $e^{-i\omega t}$,

$$
\Psi(\omega)=\frac{1}{i\omega I}\int \mathbf E_I\cdot\mathbf B_{\rm lab}\,dV,
\qquad L(\omega)=-\frac{\operatorname{Im}Z(\omega)}{\omega}.
$$

Here $\omega$ is angular frequency, $\mathbf E_I$ is the field driven by pickup
current $I$, $\mathbf B_{\rm lab}$ is the static field and $Z$ is terminal
impedance. Both quantities approach their MQS values at low frequency; negative
inductance indicates a capacitive response.

We add `"fullwave": {"frequencies": [1e5, 1e6, 1e7], "boundary": "radiation"}`
for a sweep in hertz; `pec` instead closes the domain with a perfect conductor.
Each `result.fullwave["sweep"]` row reports `f_Hz`, `Psi_abs` and `L_H`.

Both models default to `"symmetry": "axisymmetric"`, for rotationally symmetric
geometry and excitation. We set `"symmetry": "3d"` for slits or localized feeds;
this requires an [MPI-enabled installation](install.md). Axisymmetric feeds are
distributed uniformly around the axis. Eigenmode calculations are described in
[Cavity haloscopes](../../examples/cavity_demo.ipynb).

## Tree-cotree solve

The tree-cotree solve is beta.
At low frequency, separating gradient fields from magnetic-flux-carrying fields
avoids cancellation in the 3D full-wave matrix. We set `AXF_FW_SOLVER=strumpack`
and `AXF_FW_GAUGE=tree-cotree`; this requires STRUMPACK, first-order Nédélec
elements, real lossless materials and PEC on every boundary. We assess
`split_true_relative_residual`; the factorization occupies one MPI rank.

## Multiport circuits and components

`axionfem.multiport.MultiportCircuit` connects the detector impedance matrix to
external loads, preserving the relative phases of axion-induced voltages.
`axionfem.multiport_components.LosslessTransmissionLine` adds propagation delay
and stored energy for distributed connections. We include only components absent
from the FEM model, and supply reciprocal, passive impedances.
We orient circuit branches along each gap's current profile; circuit current is
opposite the current entering the FEM device.

## 3D ports and magnetic sources

Each gap in `fullwave.ports` touches two conducting terminal patches; `toward`
sets its current direction, while the primary pickup needs a conducting return
through one body. Each sweep row contains `Z_matrix_re_ohm`, `Z_matrix_im_ohm`
and coherent `Psi_vector`: the pickup comes first, matrix rows are measured
ports and columns are driven ports. `fullwave.source` supplies an independent
static field for the axion overlap (beta, below).

## Prescribed static source

The prescribed static source is beta.
For example, we add `"boundary": "pec", "source": {"kind": "uniform", "B_T": [0, 1, 0]}`
inside `fullwave` for a $1\,\mathrm T$ field along the global $y$ axis.
This replaces only the RF overlap field, leaving MQS energies and forces
unchanged; it does not compute screening. With radiation boundaries, we use
closed current segments inside the domain and check source quadrature as well
as RF mesh convergence.
