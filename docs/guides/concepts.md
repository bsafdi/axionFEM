# Concepts

This page introduces the objects every axionFEM workflow uses: the design document, the solve, the results and
their units. The [quick start](../../examples/quickstart.ipynb) runs the same steps in a notebook.

## Designs

A design is one JSON document (schema `axionfem.design/1`) that describes geometry, materials, magnet windings,
superconductors, the readout pickup, mesh settings and any further analyses. In Python it is a `Design` object; on
the command line it is a file. The rules that matter most:

- **SI units throughout**: metres, amperes, tesla, kelvin, webers, hertz.
- **Parameters and expressions.** Any numeric field may be a number or an expression over named `parameters`
  (`"0.6*R"`, `"sqrt(H*R)"`), with the usual precedence (`^` binds tighter than unary minus, so `-2^2` is −4). The
  parameters are also the variables of the shape sensitivities and of the optimizers.
- **Regions are painted in order**: where shapes overlap, later regions win, and space that no region covers is air.
- **Axisymmetric by default.** Shapes live in the meridional half-plane (r ≥ 0, z) and are revolved about the z
  axis. `"symmetry": "3d"` selects the 3D engine (an MPI build; see [3D designs](3d.md)), and `"symmetry": "xsec"`
  the cross-section of a z-uniform cavity.

The [design format](../design_format.md) is the complete reference; `af.schema()` or `axf schema` print its JSON
Schema.

### Templates

Three parametric templates cover the common lumped-element topologies. `af.templates()` lists them with every
parameter, its default and its meaning.

| template | topology | pickup |
|---|---|---|
| `toroid` | toroidal magnet (ABRACADABRA, DMRadio-GUT) | superconducting sheath with a meridional slit (azimuthal pickup) |
| `solenoid` | solenoid with an immersed shorted coaxial pickup (DMRadio-m³) | readout port across the coax gap (poloidal pickup) |
| `core` | split solenoid inside a superconducting toroidal cavity (DMRadio-Core) | neck gap of the cavity's inner wall |

A template returns a `Design`; `set` changes parameters, `save` writes the JSON file and `af.load` reads it back:

```python
import axionfem as af

d = af.template("toroid", R=2.0, H=2.0)
d.set(y=0.6)
print(d.resolved_params())        # every parameter evaluated, including the derived ones
d.save("toroid.json")
d = af.load("toroid.json")
print(d.validate()["ok"])         # True, or an error that names the JSON path and how to fix it
```

`validate` (and `axf validate design.json`) checks structure and values without meshing. An error message says what
is wrong, where (the JSON path) and how to fix it.

## Solving

`Design.solve()` meshes the design with `axf-mesher` (gmsh, run as a separate program) and solves it with the C++
core. It writes its files to a run directory, `runs/<name>` by default:

```python
res = d.solve("runs/toroid", verbose=0)
print(res.summary())
```

`order` overrides the element order (`mesh.order`, default 2) and `threads` the number of OpenMP threads; a 3D
design also takes `mpi_ranks`. The command-line equivalent is `axf solve design.json -o runs/toroid`. Which solver
runs for which design, and the numerical controls, are described in [Solvers and numerical controls](solvers.md).

## Results

`solve` returns a `Results` object (`Results3D` for a 3D design). Its main attributes are the figures of merit of a
lumped-element detector:

| attribute | meaning | unit |
|---|---|---|
| `Psi` | signal coupling Ψ = (1/I)∫ A_I · B_lab dV; the axion flux is Φ_a = κ_a Ψ | H T m² |
| `L_p` | pickup inductance | H |
| `F` | scan figure of merit F = Ψ⁴/L_p² | H² T⁴ m⁸ |
| `W` | stored magnetic energy of the static field | J |
| `B_max` | peak static field | T |

`res.data` holds the complete `results.json`: the coupling, the static and pickup solves with their solver
diagnostics, peak fields per region and on superconductor surfaces, forces, the scan benchmarks and, when the design
asks for them, full-wave sweeps, cavity modes, electrostatics, mechanics and thermal results. Every result carries
`warnings` (non-convergence, singular corners, approximations) and `checks` (independent consistency checks such as
the force closure and the circuit linkage residuals): read the warnings before quoting a number.

```python
print(res.Psi, res.L_p)
print(res.data["warnings"])                 # [] when nothing needs attention
b = res.benchmark("optimistic")             # one scan benchmark: F/F_req, required field, cost metric
print(b["F_over_F_req"], b["B_max_at_req_T"])
print(res.sample([0.5, 1.5, 2.5], [0.0, 0.0, 0.0])["Bphi"])   # B_phi at three points on the midplane [T]
```

`res.show()` displays the fields in a notebook, `res.export_html()` writes a self-contained viewer page and
`res.png()` a PNG snapshot ([Visualization](visualization.md)).

## Units and conventions

- **SI inside, natural units alongside.** Inputs and stored results are SI. The summary and the viewer also report
  Ψ and L_p in natural Heaviside-Lorentz units (Ψ in T m³, L_p in m), the units of much of the axion literature.
- **Time dependence** e^{−iωt} in every frequency-domain quantity. Touchstone files, which assume e^{+jωt}, are
  written with the conjugate ([Touchstone](../physics/touchstone.md)).
- **Scan benchmarks.** `optimistic`, `conservative` and `conservative_solenoid` fix the resonator quality factor,
  temperature and amplifier noise behind F_req ([signal, noise and optimal scan](../physics/scan_and_signal.md)).
  They are illustrative presets; give a design its own (`{"name", "Q", "T", "eta_A"}`) for a real readout.
- **Cost metric.** The benchmark's `cost` is a scaling metric for comparing designs (energy and cold-area terms), not
  a construction estimate.

## Convergence

A number is only as good as its discretization. The quickest check is to solve again at a higher element order, or
with a finer mesh, and compare:

```python
r2 = d.solve("runs/toroid_p2", order=2, verbose=0)
r3 = d.solve("runs/toroid_p3", order=3, verbose=0)
print(abs(r3.Psi / r2.Psi - 1), abs(r3.L_p / r2.L_p - 1))   # relative changes
```

`axionfem.convergence.h_study` scales every mesh size by a list of factors and extrapolates (Richardson, with an
error estimate), and `"mesh": {"adapt": {...}}` refines the mesh adaptively until the watched quantities settle
([mesh size control](../design_format.md#mesh-size-control)). Sharp corners of superconductors carry field
singularities: the peak surface field there grows under refinement, so results also report a regular peak that
excludes the corners.
