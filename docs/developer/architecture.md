# Architecture

axionFEM is a C++17 library with a Python package on top. The solves run in C++ on MFEM; Python builds designs,
drives the solves and post-processes the results; a three.js viewer and a C++ rasterizer display them.

## Repository map

| path | contents |
|---|---|
| `cpp/src`, `cpp/include/axionfem` | the library `libaxionfem` (namespace `axf`) |
| `cpp/apps` | the command line `axf` and the mesher `axf-mesher` |
| `cpp/python` | the nanobind bindings of the Python module `axionfem._core` |
| `cpp/src/fluids`, `cpp/src/chem` | the fluids (beta, MPI builds only) and chemistry (beta) solvers, with their own bindings and C++ tests |
| `cpp/tests` | C++ unit tests (run by `ctest`) |
| `python/axionfem` | the Python package: designs, results, scans, circuits, cavities, SRF, COMSOL, viewer, MCP server |
| `viewer` | the three.js viewer (`src/viewer.js`, `template.html`, vendored three.js) copied into the Python package |
| `benchmarks` | validation cases, their reference data and the lean records that tests and docs read |
| `tests` | the pytest suite |
| `examples` | walkthrough notebooks and example designs |
| `docs` | this website; `docs/images` holds the figures and the scripts that make them |
| `scripts` | dependency builds (`fetch_deps.sh`, `build_deps*.sh`, `build_strumpack*.sh`), `env.sh`, the docs build, example SLURM jobs |
| `docker` | the clean Linux images on which the install guide is tested |
| `third_party` | checksums of the pinned dependency sources; the dependencies are built into `third_party/install` |

## The C++ library

`libaxionfem` contains, roughly in the order a solve uses them:

- **Design specification** (`spec`): parsing and validation of the design JSON, expression evaluation over
  parameters, and error messages that name the JSON path.
- **Meshing driver**: writes the geometry for `axf-mesher`, which links gmsh and runs as a separate process, and reads
  the mesh back. gmsh is GPL; keeping it in its own program keeps the library BSD-licensed.
- **Axisymmetric engine**: the scalar formulations for poloidal (ψ = rA_φ) and toroidal (G = rH_φ) fields with an
  exact Dirichlet-to-Neumann open boundary, superconductor constraints by elimination, nonlinear iron, the full-wave
  m = 0 families, eigenmodes, electrostatics, elasticity, conduction and the HTS critical state.
- **3D engine** (`model3d*.cpp`, MPI builds only): H(curl) MQS, full wave and eigenmodes on `mfem::ParMesh` with
  hypre's AMS, the infinity-shell open boundary, electrostatics, mechanics and conduction in 3D. A sequential build
  compiles a stub (`model3d_nompi.cpp`) with the same interface, whose entry points raise a clear error.
- **Post-processing**: energies, inductances, couplings, forces and their closure, surface fields, losses.
- **Signal and scan** (`scan`): noise models, the optimal scan, statistics and reach; cost models.
- **Sensitivities**: mesh velocity (`morph`) and the discrete adjoint (`adjoint`).
- **Scene export and renderer**: the `axionfem.scene/1` format for the viewer and a deterministic software
  rasterizer for PNG snapshots.
- **Sparse solvers**: Eigen LDLᵀ with AMD or METIS orderings, optional STRUMPACK, and the solver diagnostics every
  result carries.

## The Python package

`axionfem._core` is the compiled module. The package around it provides:

- `design` and `results`: `template`, `load`, `Design.solve` (including multi-rank 3D solves through an MPI launcher)
  and `Results` with sampling, scenes, plots and exports;
- physics layers in Python: `scan`, `circuit`, `squid`, `multiport`, `multipole`, `touchstone`, `cavity`,
  `cavity_xsec`, `cavity_track`, `cavity_coupling`, `haloscope`, `srf`, `heterodyne`, `sensitivity`, `global_opt`,
  `convergence`;
- interoperability: `comsol` (export) and `comsol_import` with its readers;
- display: `viewer` (HTML export and live server), `widget` (Jupyter), `render`, `plots`, `draw`;
- `mcp_server`: the MCP server, which runs long tool calls in a worker process;
- beta modules: `fluids` (bindings `_core._fluids`, MPI builds only), `transport`, `chem`.

## Builds

The CMake option `AXF_MPI` follows the MFEM it finds: an MPI MFEM gives the 3D engine and the fluids module, a serial
MFEM the stub. The wheels are built against a serial MFEM (`scripts/build_deps_wheel.sh`), so a wheel has no 3D engine
and no fluids. [Build options and packaging](../install.md) lists every option and what each wheel contains.

## Design principles

- **SI inside**, natural units reported alongside with explicit labels.
- **Exact formulations over penalties**: constraints by elimination, exact open boundaries, curved high-order
  elements.
- **Every result carries diagnostics** (mesh, unknowns, solver residuals, Newton history) and at least one
  independent consistency check.
- **Errors say what, where and how to fix.**
- **Only permissively licensed libraries are linked**; GPL tools run as external programs.

[Design notes](design_notes.md) give the reasoning behind the main choices.
