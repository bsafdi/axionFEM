# C++ core

The C++ core computes electromagnetic fields, cavity modes, mechanics and
detector couplings from a design. Python handles the user interface and exposes
the results through the [design and results APIs](api.md#designs-and-solves).
Use the public Python API for normal work; this page is for people who extend
the library or call it from C++.

The library, `libaxionfem`, uses C++17 and the `axf` namespace. It builds on
MFEM, with hypre and METIS for the MPI (3D) engine and Eigen for the sparse
direct solves of the axisymmetric engine; STRUMPACK is an optional backend.
`axf-mesher` (`cpp/apps/axf_mesher.cpp`) runs gmsh in a separate process. The
command line `axf` (`cpp/apps/axf.cpp`, see [Command line](cli.md)) and the
private Python extension `axionfem._core` (nanobind,
[`cpp/python/bindings.cpp`](../cpp/python/bindings.cpp)) are thin layers over the
library.

## Checking a build

```python
import axionfem as af

print(af.build_info())
```

`build_info()` returns a dictionary with, among others:

| key | meaning |
|---|---|
| `version`, `mfem` | axionFEM and MFEM versions |
| `mpi`, `engine_3d` | whether the build has MPI and therefore the 3D engine (the sequential build does not) |
| `openmp`, `threads` | OpenMP support and the current thread count |
| `device`, `gpu_backend` | where hypre runs (`AXF_DEVICE`) and the GPU backend compiled in, if any |
| `strumpack`, `strumpack_version`, `strumpack_mpi`, `strumpack_min_dofs`, `metis` | optional sparse-solver backends |
| `fluids` | whether the fluids module (beta, MPI builds only) is compiled in |
| `git_sha`, `git_dirty`, `source_commit`, `source_dirty`, `library` | build provenance: the commit the compiled sources came from, and whether they differed from it |

See [installation](guides/install.md) for build instructions.

## Public headers

The headers in [`cpp/include/axionfem/`](../cpp/include/axionfem) document their
interfaces. The main entry points:

| header | contents |
|---|---|
| `spec.hpp` | the typed design specification: JSON in, validated C++ structures out |
| `templates.hpp`, `expr.hpp` | parametric design templates; the expression evaluator for design parameters |
| `model.hpp`, `axi.hpp` | `AxiModel`: the axisymmetric magnetoquasistatic engine (ψ = rA_φ and G = rH_φ problems) and its post-processing |
| `model3d.hpp`, `topology.hpp` | `Model3D`: the 3D H(curl) engine (MPI); cohomology bases for superconducting bodies |
| `fullwave.hpp`, `fw2d.hpp`, `pml.hpp` | axisymmetric full wave beyond MQS; 2D full wave on an imported mesh ([formulation](physics/fw2d.md)); perfectly matched layers |
| `prescribed_source.hpp`, `background.hpp` | a prescribed static source field; an imposed axisymmetric background field |
| `rational_fit.hpp` | rational fits and the adaptive frequency sweep ([physics](physics/adaptive_sweep.md)) |
| `xsec.hpp`, `surface_resistance.hpp` | cavity cross-section eigensolver; surface-resistance models ([physics](physics/surface_resistance.md)) |
| `electro.hpp` | axisymmetric electrostatics and capacitance matrices ([formulation](physics/electrostatics.md)) |
| `mech.hpp`, `torsion.hpp` | axisymmetric elasticity and heat conduction; torsional dynamics ([formulation](physics/mechanics_thermal.md)) |
| `hts.hpp`, `conductor.hpp`, `materials.hpp` | HTS critical state; conductor critical surfaces; magnetic constitutive laws |
| `adjoint.hpp`, `morph.hpp`, `adapt.hpp` | shape sensitivities; mesh morphing; adaptive refinement |
| `signal.hpp`, `scan.hpp`, `units.hpp`, `cost.hpp` | axion signal, readout noise and optimal scan ([C++ API](physics/scan_cpp_api.md)); constants and unit conversions; the cost metric |
| `render.hpp` | headless PNG snapshots of a scene ([renderer](render.md)) |
| `sym_solver.hpp`, `device.hpp`, `omp_shim.hpp` | sparse direct solver wrapper; CPU/GPU device selection; OpenMP shim |
| `fluids/`, `transport/`, `chem/` | fluids, scalar transport and chemistry (beta) |

The C++ unit tests in `cpp/tests/` (run with `ctest --test-dir build`) show
typical use of each module.
