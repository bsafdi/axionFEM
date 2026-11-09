# AGENTS.md

Instructions for AI coding agents, and for the people who work with them, on axionFEM. Two audiences:

- **Part A**: agents that change axionFEM (code, tests, documentation);
- **Part B**: agents that use axionFEM to design or evaluate axion dark-matter detectors.

axionFEM is developed with Claude Code. `CLAUDE.md` points Claude Code to this file; other agents read it directly.
Human contributors follow the same rules.

## Part A: developing axionFEM

### Repository map

| path | contents |
|---|---|
| `cpp/include/axionfem/`, `cpp/src/` | the C++17 library `libaxionfem` (namespace `axf`) |
| `cpp/src/fluids/`, `cpp/src/numerics/transport/` | fluids library and scalar-transport operator (beta; MPI builds only) |
| `cpp/src/chem/` | chemistry module (beta) and its C++ unit tests |
| `cpp/apps/` | the `axf` command line and `axf-mesher`, the separate program that links gmsh |
| `cpp/python/`, `cpp/src/*/python/` | nanobind bindings of the Python module `axionfem._core` |
| `cpp/tests/` | C++ unit tests (ctest) |
| `python/axionfem/` | Python package: builder API, templates, results, readout and scan tools, viewer server, MCP server |
| `viewer/` | three.js viewer, copied into the package; `viewer/vendor/` is the unmodified three.js release |
| `benchmarks/` | validation drivers and the lean reference records that tests and docs read |
| `tests/` | pytest suites; `tests/premerge.txt` is the quick pre-merge subset |
| `examples/` | executed notebooks (`quickstart`, `lumped_element`, `fullwave`, `readout_squid`, `cavity_haloscope`, `srf_heterodyne`, `mechanics_thermal`, `reach_optimization`, `fluids_beta`, `chemistry_beta`) and example designs |
| `docs/` | documentation site (MkDocs); `docs/developer/` holds the developer guide |
| `scripts/` | dependency builds, `env.sh`, docs tooling, example SLURM scripts |
| `tools/` | optional manufactured-solution and cross-code reference generators (NGSolve, FEniCSx) for the fluids checks |
| `third_party/`, `docker/` | dependency checksums; container recipes of the tested source builds |

### Architecture

`libaxionfem` holds units and constants, design parsing and validation (`spec`, with the expression evaluator), the
meshing driver (it runs `axf-mesher` as an external program and exchanges files with it), the axisymmetric engine
(`axi_*`: magnetostatics in ψ = rA_φ and G = rH_φ, full wave, electrostatics, HTS critical state, elasticity, heat
conduction, torsion), the 3D H(curl) engine (`model3d*`; MPI builds only, a serial build compiles
`model3d_nompi.cpp`, whose entry points raise an error), cavity eigenmodes, materials and surface resistance,
post-processing (energies, inductances, couplings, forces, surface fields), signal and scan statistics, cost
models, shape sensitivities (`morph` mesh velocity, `adjoint` discrete adjoint), the rational fit of the adaptive
sweep, scene export for the viewer and a software rasterizer for PNG snapshots. `axionfem._core` exposes it to
Python, with the submodules `_core.chem` (every build) and `_core._fluids` (MPI builds). The Python package adds the
builder API (`Design`, templates), `Results`, readout circuits and multiports, scan and reach tools, the cavity,
SRF, SQUID and COMSOL modules, the Jupyter widget, the live viewer server and the MCP server. Readout networks,
haloscope sensitivity, SQUID chains and the COMSOL bridges are pure Python; field solves are C++. The rationale
behind these choices is in `docs/developer/design_notes.md`.

### Build

Prerequisites per platform are in `docs/guides/install.md`; every CMake option is in `docs/install.md`.
`scripts/env.sh` puts the dependency prefix on the search paths and loads nothing; for site setup (`module load`
lines) point `AXF_ENV_HOOK` at a file of your own.

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install "scikit-build-core>=0.9" "nanobind>=2.0" numpy
bash scripts/fetch_deps.sh          # pinned sources, checked against third_party/SHA256SUMS.txt
bash scripts/build_deps.sh          # METIS, hypre, MFEM (MPI) into third_party/install
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                 # also writes python/axionfem/_core*.so for PYTHONPATH=python
pip install --no-build-isolation -e ".[notebook,plots,draw,mcp,test]"   # no `mesh`: this build meshes with the SDK
python -c "import axionfem as af; print(af.build_info())"   # mpi, engine_3d, fluids, openmp, ...
```

The sequential build (the wheel configuration: no MPI, no 3D engine, no fluids) is
`bash scripts/build_deps_wheel.sh` followed by `AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[all,test]"`.
Compile in a job allocation on a cluster, not on a login node.

### Tests

```bash
ctest --test-dir build --output-on-failure      # C++ unit tests
python -m pytest -q tests                       # default suite: no slow tests, no multi-rank tests
AXF_SLOW=1 python -m pytest -q tests            # adds the slow tests (3D solves, long sweeps)
AXF_MPI=1 python -m pytest -q tests             # adds the tests that start several MPI ranks
```

- Multi-rank tests start their ranks with `srun` inside a SLURM allocation, else `mpirun` or `mpiexec`
  (`tests/_mpi_launch.py`). `AXF_MPI_LAUNCHER` overrides the choice, for example `srun --overlap` inside an `srun`
  step or `srun --mpi=pmix` for an MPI that needs it.
- In batch scripts that use `srun`, `export SLURM_EXPORT_ENV=ALL`: some sites default to `NONE`, and the ranks then
  start without `OMP_NUM_THREADS`, `AXF_MESHER` or `LD_LIBRARY_PATH`.
- The full suite (`AXF_SLOW=1 AXF_MPI=1`) takes hours and needs 16 MPI tasks, because one test runs on 16 ranks.
  `scripts/slurm/tests.slurm` is an example batch script; add your site's account, partition and QoS on the
  `sbatch` command line.
- `.github/workflows/ci.yml` lists what runs on each pull request (sequential Linux and macOS builds, an MPI build
  with the 3D, fluids and chemistry tests) and what runs weekly.

#### Quick subset and full suite

- **Quick subset.** `tests/premerge.txt` names pytest node IDs, one per line, chosen so that every engine and solver
  path has a representative and the whole list runs in about half an hour on one node.
  `python scripts/premerge_tests.py run` sets `AXF_SLOW=1` and `AXF_MPI=1`, runs the list and fails on any failure
  or on a listed test that was skipped; `scripts/slurm/tests_premerge.slurm` is an example batch script.
- **Before a pull request**, run the default suite, and the quick subset when the change touches a solver or the
  physics. If it touches a path that only the full suite checks (the quantitative 3D cross-checks, the background
  studies, the distributed full-wave ports, the radiating source), run those test files too. CI then runs its
  pull-request jobs; the full suite remains the complete check, run before a release.
- **Keeping the list current.** `python scripts/premerge_tests.py check` (seconds; it needs an importable
  `axionfem`) fails when a listed test no longer exists, or when a file `tests/test_*.py` is neither listed nor
  excluded; `run` also fails when a listed test is skipped. A new test file needs a fast representative in the list
  or an `exclude` line that says why; a renamed or removed test is renamed or removed in the list too; a new engine
  or solver path needs at least one representative. Swap tests rather than grow the list.

#### Adding tests

- A test file that builds or solves 3D designs skips without the 3D engine:
  `pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"], reason=...)` (or a per-test guard).
  `premerge_tests.py check` and `tests/test_packaging.py` fail on a 3D test file without it.
- A test of a beta module skips when the module is not built: `pytest.importorskip("axionfem.fluids", exc_type=ImportError)`.
- Slow tests check `AXF_SLOW`, multi-rank tests `AXF_MPI`.
- Compare with an independent reference (closed form, manufactured solution, independent code, published result)
  and justify the tolerance. Do not loosen a tolerance to make a test pass without understanding the change.

### Conventions

- SI internally. Natural (Heaviside–Lorentz, T and m) values are reported next to the SI ones with explicit
  conversion, and result keys carry their units (`L_p_H`, `Psi_SI_H_T_m2`, `Psi_natural_T_m3`).
- Time-harmonic quantities use e^{-iωt} (Z = R − iωL). Exports for engineering tools (Touchstone) are converted to
  e^{+jωt} at the export boundary only.
- The signal coupling is Ψ = (1/I)∫A_I·B_lab d³x and Φ_a = κ_a Ψ; the scan requirement is 𝓕 = Ψ⁴/L_p² ≥ 𝓕_req
  (`docs/physics/scan_and_signal.md`).
- Every result carries diagnostics (mesh size and order, DOFs, solver residuals, Newton history) and at least one
  independent consistency check (for example L_p from flux and from energy, Newton's third law over all bodies).
  Non-convergence, refused approximations and sharp-corner peaks appear in `warnings`, never silently.
- Prefer exact formulations over penalties: constraints by elimination, Dirichlet-to-Neumann boundaries,
  high-order curved elements. Refuse invalid input instead of projecting it silently.
- Error messages say what is wrong, where in the design (its JSON path) and how to fix it (a `Hint:`). Agents read
  them.

### Scientific honesty

- Scientific correctness over favourable results. Distinguish validated results, approximations and open issues.
- Report convergence (order 2 against order 3, or a finer mesh) before quoting a number.
- Never hand-edit generated numbers. Reference records in `benchmarks/` are regenerated by their scripts; a record
  that carries a source fingerprint is regenerated, not patched.
- Label approximations and their validity range in code, results and docs.
- A capability without validation evidence is labelled **beta** in its docstring, its warnings and its docs.

### Documentation

- A change to behaviour or API updates, in the same pull request: the docstrings (the API reference is generated
  from them), the user-guide page under `docs/guides/`, the physics or validation page if the numerics changed,
  `llms.txt`, and the "Unreleased" section of `CHANGELOG.md`.
- `pip install -r docs/requirements.txt && mkdocs build --strict` must pass.
- Notebooks in `examples/` are executed by the release build (`scripts/docs/execute_notebooks.py`); their outputs
  contain no absolute paths, user or host names, or timings. New figures come from scripts in the repository.

### Code comments

Present tense; explain what the code does and why (the physics, the numerics, a non-obvious invariant); cite the
published source of a formula or benchmark. No history ("was", "used to"), no review, task or milestone IDs, no
dates, no people, no branch names.

### Dependencies

- Only permissively licensed libraries are linked: MFEM, hypre, METIS, Eigen (unmodified headers), nlohmann/json,
  nanobind, the optional STRUMPACK, and the vendored `stb_image_write.h` (`cpp/third_party/`).
- gmsh is GPL. It is used only by `axf-mesher`, a separate program; `libaxionfem`, `_core` and `axf` never link it,
  and the wheels do not redistribute it (the `mesh` extra installs gmsh's own wheel).
- three.js (MIT) is vendored unmodified in `viewer/vendor/`; its checksums are in `viewer/vendor/README.md` and
  checked by `tests/test_viewer.py`.
- A new dependency needs a compatible licence, a pinned version, a checksum in `third_party/SHA256SUMS.txt` and a
  line in the install docs.

## Part B: using axionFEM to design detectors

### Start

- **MCP**: when the `axionfem` server is available (`.mcp.json` in this repository, `docs/guides/mcp.md`), call
  `axionfem_guide` first; it explains the workflow, units and conventions.
- **Python**: `import axionfem as af` in the environment where axionFEM is installed. `af.build_info()` says whether
  this build has the 3D engine (`engine_3d`) and the fluids module (`fluids`).
- `llms.txt` is the shortest map of the API; `docs/design_format.md` lists everything a design can contain. In Claude
  Code, the `axionfem-design` skill (`.claude/skills/axionfem-design/`) summarizes this part.

### Workflow

1. Start from a template (`af.templates()`: toroid, solenoid, core) or write a design document. Geometry is
   axisymmetric in (r, z) unless `"symmetry": "3d"`.
2. `d.validate()`; an error names the JSON path and gives a hint: fix exactly that.
3. `res = d.solve(workdir=...)`; read `res.summary()`, then `res.data["warnings"]` and `res.data["checks"]`.
4. Look at the result: `res.png(view="section")` (open the PNG and inspect it), `res.export_html()`.
5. Check convergence: solve again at order 3 (MCP: `solve_design(name, order=3)`) or on a finer mesh, and quote
   only digits that do not change.
6. Interpret: 𝓕/𝓕_req per benchmark (the built-in benchmarks are illustrative presets), the amplitude scale needed,
   peak fields at the requirement, conductor margins, the reach curve. The cost metric compares designs; it is not a construction estimate.
7. Explore: `d.set(...)` and re-solve, sweeps, shape sensitivities (adjoint, axisymmetric designs with linear
   materials) and optimization (gradient-based, global differential evolution, derivative-free Powell).

### Capability map

| task | Python | MCP tools | status |
|---|---|---|---|
| lumped-element detectors, axisymmetric magnetoquasistatics | `af.template`, `Design.solve` | `create_design`, `solve_design`, `render_design` | |
| 3D designs | `"symmetry": "3d"`, `axionfem.convert.to_3d` | `make_3d` | MPI build |
| full wave beyond MQS, adaptive sweep, Touchstone | design `fullwave` section, `Results.write_touchstone` | `set_fullwave`, `export_touchstone` | 3D: MPI build; 3D current ports validated; prescribed sources, imposed background, PML, error indicator, Gauss-law guard, adaptive sweep, tree-cotree solve, 3D eigenmodes, closed RF-only models, energy/field-export/reactive-load diagnostics and surface-resistance models: beta |
| HTS critical state, conductor margins | design `hts`, `conductors` | `set_hts`, `list_conductors` | |
| electrostatics, mechanics and heat conduction | design `electrostatics`, `mechanics`, `thermal` | `put_design` | 3D: MPI build |
| readout circuits, scans, reach | `axionfem.circuit`, `axionfem.scan`, `axionfem.multiport` | `requirement`; `solve_design` reports each benchmark | |
| multi-band and multipole scans | `axionfem.multipole` | none | beta |
| parameter sweeps, sensitivities, optimization | `Design.set`, `Results.sensitivities`, `af.optimize` | `sweep`, `sensitivities`, `optimize_design`, `optimize_global`, `optimize` | |
| cavity haloscopes | `axionfem.cavity`, `cavity_xsec`, `cavity_track`, `haloscope`, `haloscope_presets` | `create_cavity`, `cavity_scan`, `cavity_presets`, `cavity_preset_limit` | beta |
| SRF heterodyne cavities | `axionfem.srf`, `axionfem.heterodyne` | `srf_design`, `srf_mode_pair` | beta |
| dc SQUID readout chains | `axionfem.squid` | `squid_chain`, `squid_reach` | beta |
| torsional dynamics | `Results.torsion`, `axionfem.torsion_waveguide` | none | beta |
| COMSOL export and import | `axionfem.comsol`, `af.import_comsol` | `export_comsol`, `inspect_comsol`, `import_comsol` | import: beta |
| cavity coupling through a coaxial probe | `axionfem.cavity_coupling` | none | beta |
| incompressible flow, scalar transport, Boussinesq | `axionfem.fluids`, `axionfem.transport` | none | beta, MPI build |
| electrochemistry and kinetics | `axionfem.chem` | none | beta; 3D needs an MPI build |

### Honesty rules

- Report mesh convergence before quoting numbers.
- Say when a result depends on a sharp corner (field singularities; use fillets and the "regular" peak) or on an
  idealization (perfect Meissner response, an assigned Q, the uniform-feed limit of the axisymmetric full-wave
  model).
- Results of beta modules are labelled beta when you report them.
- Never claim a design is feasible beyond what was computed.

### Where conventions live

`docs/physics/scan_and_signal.md` (signal, noise, optimal scan, 𝓕_req), `docs/design_format.md` (the design
document and its results), `docs/guides/` (one page per workflow), `docs/validation/` (what each capability was
checked against).
