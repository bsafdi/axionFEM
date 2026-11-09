# Build options and packaging

This page is the reference behind the [installation guide](guides/install.md): what the wheels contain and how they
are built, every CMake option, the dependency scripts, the optional sparse direct solvers and GPU builds, and
clusters. To install axionFEM, follow the guide; [Building from source](developer/building.md) describes a
developer's build.

| | what you get | what it needs |
|---|---|---|
| [Wheel](#wheels-for-laptops) (`pip install "axionfem[mesh]"`, once on PyPI) | axisymmetric and full-wave engines, scan reach and cost metric, viewer, widget, CLI, MCP server | nothing but Python 3.10+ |
| [Source build](#source-build) | the same, plus the **3D engine** and the fluids module (beta) | a C++17 compiler, MPI, BLAS/LAPACK, CMake |
| [Clusters](#clusters-and-batch-systems) | the same source build, compiled and run in batch jobs | the site's compiler, MPI and CMake modules |

The one thing a wheel cannot give you is the 3D engine (and with it the fluids module). It needs MPI, and a
wheel on PyPI must not require a system MPI, so the wheels are *sequential*. A
3D design in a wheel raises an error that says exactly this and points here.

---

## Wheels for laptops

Version 0.1.0 is not yet on PyPI; until the wheels are uploaded, the sequential source build of the
[installation guide](guides/install.md#sequential-source-build) gives the same package. Once published:

```bash
pip install "axionfem[mesh]"        # mesh: the gmsh wheel, needed to mesh anything
pip install "axionfem[mesh,notebook]"   # also the Jupyter widget through anywidget
pip install "axionfem[draw]"        # circuit drawings: Readout/Circuit/Chain .draw() (schemdraw)
```

Platforms built by CI (release tags and on demand): **manylinux_2_28 x86_64**,
**macOS x86_64** (11.0+ via `macos-15-intel`) and **macOS arm64** (11.0+ via `macos-14`), for CPython 3.10–3.13.
musllinux, PyPy and Linux aarch64 are not built; use a source build. Windows is not supported.

Check what you got:

```python
import axionfem as af
af.build_info()
# {'version': '0.1.0', 'mfem': '4.10', 'device': 'cpu', 'gpu_backend': '',
#  'mpi': False, 'engine_3d': False, 'openmp': True, 'threads': 8}
```

`engine_3d` is the one difference from a source build.

### Why the `[mesh]` extra

Meshing runs in `axf-mesher`, a separate program that links gmsh. gmsh is GPL,
so axionFEM keeps it out of the library and in its own process.
For the same reason **our wheels do not contain gmsh**: they
ship `axf-mesher`, and the GPL library it loads comes from gmsh's own PyPI
wheel, which you install yourself. Without it, solving stops with

```
cannot find the axf-mesher executable ... Installed from a wheel: `pip install gmsh` ...
```

`python/axionfem/_paths.py` finds `libgmsh` in the interpreter's data prefix
(`<venv>/lib/libgmsh.so.4.15`) and puts that directory on the loader path of the
mesher child process. `$AXF_GMSH_ROOT` overrides the search and `$AXF_MESHER`
overrides the executable.

The mesher links libgmsh by its versioned name (`libgmsh.so.4.15` on Linux,
`libgmsh.4.15.dylib` on macOS), so it loads only the gmsh minor version it was
built against. The `[mesh]` extra therefore pins it (`gmsh>=4.15,<4.16`), as does
`build-system.requires`. **A gmsh minor bump needs a rebuild of the wheels**, with
the pins in `pyproject.toml` and `.github/workflows/ci.yml`, the `auditwheel
repair --exclude libgmsh.so.4.NN` name in `[tool.cibuildwheel.linux]` and the SDK
version `GMSH_VER` in `scripts/fetch_deps.sh` and `scripts/build_deps.sh` moved
together; delocate's `--exclude libgmsh` (macOS) has no version in it.
`tests/test_packaging.py::test_gmsh_minor_version_is_pinned_consistently` checks
that they agree.

### Licensing of each wheel

| file in the wheel | licence | note |
|---|---|---|
| `axionfem/*.py`, `axionfem/_core*.so`, `axionfem/libaxionfem.so`, `axionfem/bin/axf` | BSD-3-Clause | our code |
| `axionfem/_libs/libmfem.*` | BSD-3-Clause | MFEM, built serial |
| `axionfem/_viewer/src/viewer.js`, `template.html` | BSD-3-Clause | our code |
| `axionfem/_viewer/vendor/three*` | MIT | three.js r185, with its licence file |
| (header-only, compiled in) Eigen | MPL-2.0 | unmodified |
| (header-only, compiled in) nlohmann/json, nanobind | MIT, BSD-3 | |
| (compiled into `libaxionfem`) stb_image_write | MIT or public domain | `cpp/third_party/`, unmodified |
| `axionfem/bin/axf-mesher` | BSD-3-Clause source; **links GPL gmsh at run time** | see below |
| `axionfem/_libs/libmetis.*` | Apache-2.0 | nested-dissection ordering of the axisymmetric engine's large factorizations |
| **not present**: `libgmsh`, hypre, STRUMPACK, MPI | — | hypre is only needed by the 3D engine; a custom build that bundles hypre or STRUMPACK adds their licences (below) |

The licence texts of every bundled or compiled-in component are in `third_party/licenses/` (its README maps each to
its files), and the wheel carries them in its `.dist-info/licenses/` directory (`wheel.license-files` in
`pyproject.toml`). `third_party/licenses/` also holds hypre's licence for builds against an MPI stack; a build that
bundles STRUMPACK must add STRUMPACK's own `LICENSE`.

`axf-mesher` is the one sensitive file. Its source (`cpp/apps/axf_mesher.cpp`)
is BSD-3-Clause, but the running program is a combined work with GPL-2+ gmsh, so
anyone who redistributes the *combination* must satisfy the GPL. That is easy
here — BSD-3 is GPL-compatible and we publish the source — and the library,
the Python module and the `axf` CLI are untouched by it, because they never link
gmsh and only exchange files with the mesher. We do not distribute gmsh itself:
the user installs it from the gmsh project's wheel under the GPL.

A stricter reading would put `axf-mesher` in a second, GPL-labelled wheel so that the `axionfem` wheel contains
nothing that links GPL code at all. The CMake build and the packaging already keep the mesher a separate target and a
separate process, so that split would be a packaging change only.

### Building a wheel yourself

```bash
pipx run cibuildwheel --platform linux         # uses [tool.cibuildwheel] in pyproject.toml
# or, for the interpreter you are using:
bash scripts/build_deps_wheel.sh                              # serial MFEM, ~5 min
AXIONFEM_DEPS=$PWD/third_party/install-seq pip wheel . -w dist/
```

`scripts/build_deps_wheel.sh` builds the sequential third-party stack:
MFEM 4.10 shared with `MFEM_USE_MPI=NO` (no hypre, no LAPACK, no zlib), METIS
for the axisymmetric engine's nested-dissection ordering, plus the Eigen and
nlohmann/json headers. **Only the 3D engine's sources (the `model3d*` files),
the fluids module (built only with MPI) and the chemistry 3D current
distribution (compiled out without MPI) use MPI or hypre**, so the sequential
build needs no hypre at all, not even in its `mpistubs` mode.
`tests/test_packaging.py::test_only_model3d_uses_mpi_or_hypre` keeps it that way.

`AXF_BUNDLE_LIBS=ON` (the default under scikit-build-core) copies
`libmfem.*` into `axionfem/_libs/` and sets the rpath of `_core`,
`libaxionfem`, `axf` and `axf-mesher` to `$ORIGIN/_libs` (`@loader_path/_libs`
on macOS), so the wheel carries its own MFEM. Only real files are copied, not
the unversioned development symlink: `install(FILES)` dereferences symlinks, so
`libmfem.so` beside `libmfem.so.4.10.0` would be a second full copy, and the
loader wants the SONAME anyway. `auditwheel repair` on Linux and
`delocate-wheel` on macOS then add the remaining system libraries and tag the
wheel. The resulting Linux wheel is about 15 MiB, most of it MFEM.

---

## Source build

The prerequisites for each platform (Debian/Ubuntu, Fedora/RHEL, macOS, clusters) and the exact command sequence,
with what was tested where (clean Ubuntu 24.04 and Rocky Linux 9 containers from `docker/`; macOS not yet), are in
[the installation guide](guides/install.md). In short:

```bash
git clone https://github.com/bsafdi/axionFEM && cd axionFEM
python3 -m venv .venv && source .venv/bin/activate   # Python >= 3.10 (macOS, RHEL 9: python3.12)
pip install "scikit-build-core>=0.9" "nanobind>=2.0" numpy
bash scripts/fetch_deps.sh          # pinned sources (network), verified against third_party/SHA256SUMS.txt
bash scripts/build_deps.sh          # METIS, hypre, MFEM (MPI), the gmsh SDK into third_party/install
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
pip install --no-build-isolation -e ".[test]"
pytest -q tests
```

Dependencies, all pinned in `scripts/fetch_deps.sh`: MFEM 4.10, hypre 2.33,
METIS 5.1, Eigen 3.4, nlohmann/json 3.12, the gmsh 4.15 SDK, nanobind, three.js
r185 (vendored in `viewer/vendor`).

### The dependency scripts

| script | builds | into |
|---|---|---|
| `scripts/fetch_deps.sh` | downloads the pinned archives and verifies them against `third_party/SHA256SUMS.txt`; `--update-checksums` records new ones after a version change | `third_party/src` (`AXF_DEPS_SRC`) |
| `scripts/build_deps.sh` | METIS, hypre and MFEM with MPI, METIS and LAPACK; the Eigen and json headers; the gmsh SDK | `third_party/install` (`AXIONFEM_DEPS`, `AXF_DEPS_PREFIX`) |
| `scripts/build_deps_wheel.sh` | serial MFEM and METIS, the headers (the wheel configuration) | `third_party/install-seq` (`AXF_WHEEL_DEPS`) |
| `scripts/build_strumpack.sh`, `scripts/build_strumpack_mpi.sh`, `scripts/build_deps_gpu.sh` | optional, below | |

They take the compilers from `CC`, `CXX`, `FC` (defaults `cc`, `c++`, `gfortran`) and the MPI wrappers from
`MPICC`, `MPICXX`, `MPIFC` (defaults `mpicc`, `mpicxx`, `mpif90`; on Cray systems `cc`, `CC`, `ftn`), the number
of parallel jobs from `NPROC` (default: all usable cores), and the BLAS/LAPACK vendor for CMake's search from
`AXF_BLA_VENDOR` (`OpenBLAS`, `FlexiBLAS`, `Generic`, `Apple`, `Intel10_64lp`, ...). MFEM finds LAPACK through
CMake (OpenBLAS, reference LAPACK, MKL, or Accelerate on macOS); STRUMPACK and ScaLAPACK take explicit libraries
from `AXF_BLAS_LIBS`/`AXF_LAPACK_LIBS`, else OpenBLAS through `pkg-config` or Homebrew, else CMake's search.
The installed libraries carry an RPATH (Linux) or absolute install names (macOS) for the prefix, so nothing has to
be put on `LD_LIBRARY_PATH`; `source scripts/env.sh` sets the search paths anyway and runs `$AXF_ENV_HOOK` first.

The gmsh SDK (`AXF_GMSH_SDK`): on Linux x86_64 the headless "nox" build, which needs no X11 or OpenGL libraries;
on macOS the x86_64 or arm64 SDK. Other platforms (Linux aarch64, for example) have no SDK; there the mesher is
built against gmsh's PyPI wheel, which CMake finds in the Python environment it uses
(`pip install "gmsh>=4.15,<4.16"`), and which needs the X11/OpenGL libraries listed in the guide.

### The build matrix

| CMake option | values | default | meaning |
|---|---|---|---|
| `AXF_MPI` | `AUTO`, `ON`, `OFF` | `AUTO` | the 3D engine. `AUTO` follows `MFEM_USE_MPI`, so a serial MFEM selects the stub by itself. `ON` against a serial MFEM is a configure error with a hint. |
| `AXF_OPENMP` | `AUTO`, `ON`, `OFF` | `AUTO` | OpenMP in the assembly loops. `OFF` compiles through `cpp/include/axionfem/omp_shim.hpp` and is single-threaded; results are identical. |
| `AXF_GPU` | `OFF`, `CUDA`, `HIP` | `OFF` | see [GPU builds](#gpu-builds). |
| `AXF_USE_STRUMPACK` | `ON`/`OFF` | `ON` | use STRUMPACK for the large axisymmetric systems when it is installed under `AXF_DEPS` (silently off when it is not). See [Sparse direct solvers](#sparse-direct-solvers). |
| `AXF_STRUMPACK_PREFIX` | path | empty | the prefix of an MPI STRUMPACK (`scripts/build_strumpack_mpi.sh`), used instead of the serial one in `AXF_DEPS`: the 3D direct full-wave path then factorizes across the ranks. Needs `AXF_MPI` and a Fortran compiler at configure time. See [Sparse direct solvers](#sparse-direct-solvers). |
| `AXF_BUNDLE_LIBS` | `ON`/`OFF` | `ON` under scikit-build-core | copy the third-party shared libraries (MFEM, hypre, METIS, STRUMPACK) into `axionfem/_libs` with `$ORIGIN` rpaths. |
| `AXF_BUILD_MESHER` | `ON`/`OFF` | `ON` | build `axf-mesher`. Without it nothing can be meshed. |
| `AXF_GMSH_ROOT` | path | — | where to find `include/gmsh.h` and `lib/libgmsh.*`. Searched automatically: `$AXF_GMSH_ROOT`, `$AXF_DEPS/gmsh-sdk`, the interpreter's data prefix (where `pip install gmsh` puts them). |
| `AXF_BUILD_PYTHON`, `AXF_BUILD_TESTS` | `ON`/`OFF` | `ON` | the nanobind module, the C++ unit tests. |
| `AXF_INPLACE_PYTHON_DIR` | path | `python/axionfem` | where a non-wheel build writes `_core`; change it so that several build trees can coexist. |
| `AXF_DEPS` | path | `$AXIONFEM_DEPS` or `third_party/install` | the third-party prefix. |

The two configurations CI builds (`.github/workflows/ci.yml`: every pull request and push to `main`, with the 3D
engine's tests, the fluids and chemistry tests and a two-rank solve in the MPI job; the full suite weekly and on
demand; the macOS jobs follow `docs/guides/install.md` as written and have not run yet):

| | MFEM | hypre, METIS | `AXF_MPI` | 3D engine | used for |
|---|---|---|---|---|---|
| sequential (Linux, macOS) | `MFEM_USE_MPI=NO` | not needed | `OFF` | stub, clear error | the wheels, laptops |
| MPI (Linux; macOS weekly) | `MFEM_USE_MPI=YES` | yes | `ON` | full | clusters, 3D work |

### 3D and MPI

The 3D engine's sources (`cpp/src/model3d*.cpp` and their private headers),
the fluids module and the chemistry 3D current distribution are the only files
that use MPI or hypre (`tests/test_packaging.py` lists them): the 3D H(curl) engine is built on `mfem::ParMesh`,
`mfem::ParFiniteElementSpace` and hypre's AMS/BoomerAMG. None of those types
exist in a serial MFEM, so a sequential build compiles
`cpp/src/model3d_nompi.cpp` instead. It defines the same surface as
`axionfem/model3d.hpp` and every entry point raises

```
the 3D engine is not available in this build of axionFEM (needed for ...)
  [at /symmetry]
  Hint: This build has no MPI: it was configured with AXF_MPI=OFF against an MFEM
  built with MFEM_USE_MPI=NO (the sequential configuration, also that of the
  planned wheels, which cannot require a system MPI). The axisymmetric and
  full-wave engines are unaffected. For 3D ("symmetry": "3d",
  axionfem.convert.to_3d), use the MPI source build: see docs/guides/install.md,
  section "MPI source build".
```

Porting the 3D engine to serial MFEM (`Mesh`, `BilinearForm`, an algebraic
multigrid that is not hypre's) is possible but is a numerics change, not a
packaging one, so it is deliberately not attempted here.

`tests/test_packaging.py` guards both halves: that no other file starts using
MPI or hypre, and that the stub still defines everything the header declares.

---

## Sparse direct solvers

The axisymmetric engine (2D) solves its symmetric systems with a sparse direct
factorization. Two optional libraries change *how*, never *what*: all backends
agree to round-off, which `tests/test_solvers.py` checks on every build.

| | needs | used for | without it |
|---|---|---|---|
| Eigen `SimplicialLDLT`, AMD ordering | nothing (Eigen is always there) | everything small | — |
| the same with METIS nested dissection | `libmetis` under `AXF_DEPS` | systems above 1.5e5 unknowns, where Eigen's AMD becomes unreliable | nothing above 4e5 unknowns: the solve is refused |
| STRUMPACK (multifrontal, OpenMP, METIS) | `scripts/build_strumpack.sh` | systems above `AXF_STRUMPACK_MIN_DOFS` | Eigen's LDLT at every size |

```bash
source scripts/env.sh
bash scripts/build_strumpack.sh      # STRUMPACK 8.0.0 (BSD-3) into $AXIONFEM_DEPS, serial + OpenMP
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # finds it, prints "STRUMPACK 8.0.0"
cmake --build build                  # build/axf, which multi-rank solves launch, and the in-place module
pip install --no-build-isolation -e ".[test]"             # the Python environment's module, now with STRUMPACK
python -c "import axionfem; print(axionfem.build_info())"  # 'strumpack': True, 'metis': True
```

**Other STRUMPACK versions.** STRUMPACK keeps process-wide state (per-thread timer lists, thread parameters, a
`std::new_handler`) that axionFEM coordinates when several threads of one process use it. The guard that lets
OpenMP threads factorize concurrently (`build_info()["strumpack_guard"] == "concurrent"`) relies on STRUMPACK 8.0
internals and is used with 8.0.x only. With any other version CMake prints a warning and the build uses the
serialized guard (`"serialized"`): one STRUMPACK call at a time in the process, from OpenMP thread 0 only, and
frequency-parallel STRUMPACK sweeps (the torsion module's `parallel_frequencies`) run one frequency at a time. The
calls of the distributed solvers below are included: each holds the lock until every rank has taken part, so one
thread per rank may use them at a time. The first STRUMPACK call prints a one-time warning that says so.
`-DAXF_STRUMPACK_SERIALIZE=ON` selects the serialized guard with 8.0.x as well. Tested: 8.0.0 in both modes and
8.1.0 (serialized) with the shared-memory STRUMPACK of `build_strumpack.sh`; the distributed solvers with the MPI
STRUMPACK 8.0.0 of `build_strumpack_mpi.sh` in both modes.

**Distributed STRUMPACK (MPI, optional).** The 3D full-wave direct path
(`AXF_FW_SOLVER=strumpack`) gathers the matrix to rank 0 with the serial build above. Built against an MPI STRUMPACK
instead, it factorizes across the ranks (`docs/validation/fullwave_stage2a_boundaries_solver.md`). The script
needs the MPI Fortran wrapper (`mpif90`, or `MPIFC`), which the STRUMPACK CMake package also needs when axionFEM is
configured, and links MPI's Fortran bindings (OpenMPI's `libmpi_mpifh`, MPICH's `libmpifort`, Cray MPICH's
`libmpifort_<compiler>`, found from the wrapper's link line; `AXF_MPI_FORTRAN_LIBS=/path/to/lib...` names the library
when the wrapper does not show it, and the script warns in that case). Configure a fresh build directory; an existing
one is switched
as well, since CMake drops a cached `STRUMPACK_DIR` from the other prefix:

```bash
source scripts/env.sh
bash scripts/build_strumpack_mpi.sh   # ScaLAPACK 2.2.0 + STRUMPACK 8.0.0 (MPI), static, into third_party/install_mpi
cmake -S . -B build-mpi -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXF_STRUMPACK_PREFIX=$PWD/third_party/install_mpi
cmake --build build-mpi
pip install --no-build-isolation -e ".[test]" -C cmake.define.AXF_STRUMPACK_PREFIX=$PWD/third_party/install_mpi
export AXF_EXE=$PWD/build-mpi/axf      # Design.solve(mpi_ranks=N) launches this executable (default: build/axf)
python -c "import axionfem; print(axionfem.build_info()['strumpack_mpi'])"   # True
```

- **What the script builds:** ScaLAPACK from source against the same BLAS/LAPACK as the rest of the process (a
  system ScaLAPACK linked against another BLAS, e.g. MKL next to OpenBLAS, must not meet it in one process), and
  STRUMPACK from the same pinned tarball as `build_strumpack.sh`.
- **Static libraries:** they keep a second `libstrumpack.so` from being shadowed by the serial one in
  `AXIONFEM_DEPS`.
- **The 2D engine:** that library also serves the 2D engine's sequential solver.
- **Threads:** the distributed solvers make collective MPI calls, so each rank uses them from one thread. With the
  concurrent guard (8.0.x) they are not coordinated with STRUMPACK calls from other threads of the same process
  (for example a 2D solve in another Python thread while a distributed 3D solve runs); with the serialized guard they
  are.
- **Tested:** Ubuntu 24.04 with OpenMPI and with MPICH (build, link and C++ unit tests); with the commands above,
  `AXF_MPI=1 pytest tests/test_fullwave3d_strumpack_mpi.py` passes on Ubuntu 24.04 with OpenMPI (two and three ranks). With a Cray programming
  environment (`MPICC=cc MPICXX=CC MPIFC=ftn`, Cray MPICH, no `AXF_BLAS_LIBS` so that the wrappers link LibSci
  themselves) the script builds and the C++ unit tests pass, but 3D solves with the distributed factor on two and
  three ranks crashed (segmentation fault) in `tests/test_fullwave3d_strumpack_mpi.py`; the cause is not known (the
  wrappers also link LibSci's own ScaLAPACK next to the one built here). Use the serial STRUMPACK there. Naming a
  LibSci library in `AXF_BLAS_LIBS` on Cray gave wrong STRUMPACK solves (two LibSci variants in one process).
- **Run-time control:** `AXF_FW_STRUMPACK_DIST` (`auto` | `1` | `0`).

**Memory of the 3D full-wave direct factorizations** (glibc; `docs/validation/fullwave_memory.md`). Every direct
factorization of a full-wave analysis (plain, distributed and tree-cotree STRUMPACK, and the shift-invert eigenmode
factor of cavity and SRF designs) hands glibc's freed memory back to the system:

| variable | default | meaning |
|---|---|---|
| `AXF_FW_MALLOC_TRIM` | `factor` | `malloc_trim` after every direct factorization and whenever a factor is released; `frequency` also at every frequency start, recording the resident set before and after (a diagnostic); `off` never |
| `AXF_FW_MALLOC_THRESHOLDS` | `fixed` | glibc's trim threshold 1 MiB and mmap threshold 32 MiB (`mallopt`) from the analysis's first direct factorization to its end; then trim 64 MiB and mmap 32 MiB, glibc's own steady state after a large free, for the rest of the process (`mallopt` cannot restore glibc's dynamic adjustment). `glibc` leaves the thresholds alone |
| `AXF_FW_FREQUENCY_PEAK` | `0` | `1`: each row's peak resident set is that frequency's own (a diagnostic) |

- **In process.** `Design.solve()` of a 3D design without `mpi_ranks` runs in the Python process, so after a direct
  full-wave analysis the Python session keeps trim 64 MiB / mmap 32 MiB rather than glibc's dynamic thresholds; numpy
  and scipy run as fast as under glibc's steady state (the 1 MiB trim threshold, which slowed 2-24 MiB numpy
  temporaries 2.5-9.4×, is only in force during the analysis). `AXF_FW_MALLOC_THRESHOLDS=glibc` before the first
  direct analysis of a process keeps glibc's dynamic thresholds throughout.
- **The environment wins.** Where the job sets glibc's thresholds itself (`MALLOC_TRIM_THRESHOLD_`,
  `MALLOC_MMAP_THRESHOLD_` or `glibc.malloc` tunables in `GLIBC_TUNABLES`), axionFEM leaves them alone.
- `fullwave.memory` records what was in force for each analysis (`malloc_thresholds`,
  `malloc_thresholds_applied_ranks`, `malloc_thresholds_before`).

The sequential build (the configuration of the planned PyPI wheels) has neither (a wheel must not depend on a
system BLAS/MPI), so
they use Eigen's LDLT with the AMD ordering, and are limited to about 4e5
unknowns per 2D solve (see below). Run-time controls:

| variable | default | meaning |
|---|---|---|
| `AXF_STRUMPACK_MIN_DOFS` | `300000` | reduced unknowns above which STRUMPACK is used. Accepts `3e5`; `1e18` means "never". |
| `AXF_ORDERING` | `auto` | `amd`, `metis`, `strumpack`, or `auto` (by size). A backend that this build does not have warns and falls back. |

Eigen's own AMD ordering fails unpredictably on large meridional matrices (minutes of ordering, and a
segfault inside Eigen itself on one node; see `docs/validation/solvers.md`), so `auto` uses METIS from
1.5e5 unknowns, and above 4e5 unknowns a build with neither METIS nor STRUMPACK refuses the solve with
an error that says how to fix it (coarsen the mesh, or get one of the two) instead of grinding or
crashing. `AXF_ORDERING=amd` runs it anyway, with a warning.

Every solve reports which backend ran and what it cost, in
`results.json` under the solve's `solver` block (`backend`, `ordering`,
`nonzeros_L`, `symbolic_seconds`, `factorization_seconds`, `solve_residual`),
so a run can be checked after the fact. `docs/validation/solvers.md` has the
measurement behind the default threshold; `benchmarks/solver_scaling.py`
reproduces it.

The exterior map (the DtN boundary condition that makes the finite domain exact,
`docs/design_format.md`) is a rank-`dtn_modes` term. It is applied as a
low-rank (Woodbury) update of the sparse factorization, not assembled into the
matrix: as a dense block it makes the outer-boundary unknowns a clique of the
sparsity graph, which costs 17-25 % of the factor nonzeros and makes STRUMPACK
warn about the depth of its elimination tree. `AXF_DTN_DENSE=1` restores the
assembled block for comparison (`dtn_assembly` in the diagnostics).

---

## GPU builds

`-DAXF_GPU=CUDA` (or `HIP`) builds against an MFEM that has that backend. It is
a *capability*, not a mode: everything runs on the CPU unless `$AXF_DEVICE`
asks for a GPU, so a GPU build still imports on a machine with no GPU.

```bash
# 1. a CUDA third-party stack (hypre + METIS + MFEM with CUDA; STRUMPACK on the host). Needs nvcc, MPI, CMake and
#    the build tools of the CPU stack; no GPU. AXF_CUDA_ARCH lists the GPU architectures to compile for.
AXF_CUDA_ARCH="80;90" bash scripts/build_deps_gpu.sh

# 2. axionFEM against it
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAXF_DEPS=$PWD/third_party/install-cuda \
      -DAXF_GMSH_ROOT=$PWD/third_party/install/gmsh-sdk \
      -DAXF_GPU=CUDA "-DCMAKE_CUDA_ARCHITECTURES=80;90"
cmake --build build-cuda

# 3. run where a GPU is visible
AXF_DEVICE=cuda python -c "import axionfem; print(axionfem.build_info())"
```

| | |
|---|---|
| `AXF_GPU=CUDA` | requires `MFEM_USE_CUDA=YES`; enables the CUDA language and defines `AXF_GPU_CUDA`. axionFEM itself has no device code, so `CMAKE_CUDA_ARCHITECTURES` (default `70;80;86`) only satisfies CMake; the architectures that matter are the stack's (`AXF_CUDA_ARCH`, default `80;90`: A100, H100/H200; `70` V100, `75` T4/RTX 20xx, `86` A40/RTX 30xx, `89` L4/L40/RTX 40xx). |
| `AXF_GPU=HIP` | requires `MFEM_USE_HIP=YES` and `find_package(hip)`; defines `AXF_GPU_HIP`. Not built or tested. |
| `$AXF_DEVICE` | `cpu` (default), `cuda` (the first visible GPU), or `cuda:N` (GPU N; only N = 0 has run, every validation job having one visible GPU); `hip`, `hip:N` in a HIP build. Anything else is refused. Read once per process by `cpp/src/device.cpp`, from the `axf` CLI's `main` and the nanobind module's init. |
| one MPI rank | a GPU run with two or more MPI ranks is refused: it crashed in the validation (a `cudaFree` error in MFEM's memory manager, an invalid rank in a halo exchange; `docs/validation/gpu.md`). |
| what moves | **hypre alone**: AMS, AME, BoomerAMG and hypre's PCG in the 3D engine (MQS, full wave, cavity eigenmodes, the H1 solves, mechanics, conduction, electrostatics). MFEM's own device stays the CPU: the assembly, the MQS CG and full-wave GMRES loops, STRUMPACK's factorizations and all post-processing run on the host, and MFEM copies vectors into hypre's memory at each call. `results.json` says where hypre ran (`solver.hypre_device`). It is not the same algorithm as on the CPU (the H1 BoomerAMG keeps MFEM's GPU options, the mechanics run the systems AMG), so results agree at the solver tolerance. |
| no GPU backend | `AXF_DEVICE=cuda` in a CPU-only build fails at import with a message naming `-DAXF_GPU=CUDA` and this file. |
| no GPU visible | `AXF_DEVICE=cuda` on a node without a GPU fails at the first 3D solve, saying so. |

**What a GPU buys today: correct answers, not speed.** The only solver that can
use one is the 3D engine's hypre, and that needs MPI, so `AXF_GPU` with
`AXF_MPI=OFF` prints a warning and has no GPU path. The axisymmetric and
full-wave engines are CPU sparse direct factorisations (Eigen `SimplicialLDLT`
and a complex `LDL^T`); moving them to a GPU would mean a different solver, not
a build flag.

**Verified (`docs/validation/gpu.md`).**
- **Agreement** (one MPI rank, against the CPU build, at the solver tolerance):
  - on an H100 and an A40: the 3D MQS, the full-wave AMS-GMRES path (Gauss-law
    guard on), the STRUMPACK and tree-cotree full-wave paths, and both cavity
    eigensolvers (AME and shift-invert);
  - in a second check: seven cavity designs (AME and shift-invert) and
    `tests/test_gpu.py` on an A100, `tests/test_gpu.py` on an RTX 8000;
  - by output: L_p, Ψ and W within 1e-12; Z, dZ/dω and L within 3.1e-11; the
    full-wave Ψ vector within 1.2e-8 (A40) and 2.9e-9 (H100); the STRUMPACK,
    tree-cotree and shift-invert paths within 2.5e-12; the AME cavity to its
    eigenvector's accuracy (f 1.6e-11, C 4.8e-10, G and Q 5.4e-7, peak wall
    fields 2.8e-5); every other output, the AME cavity apart, within 1.3e-8,
    about as much as the CPU on 14 ranks differs from 1 rank (1.4e-8);
  - the GPU model matters: on one 3D full-wave case the same executable needs
    1179 GMRES iterations on the A40 and 886 on the H100 (1049 on one CPU
    core).
- **Reproducibility:** the CUDA build run on the CPU is bit-identical to the CPU build; repeated GPU runs are
  bit-identical.
- **Speed:** on 0.4-1.1M-DOF models an H100 is 1.9-3.8 times faster than one CPU
  core and an A40 1.55-2.3 times, but 3.1-5.2 times slower than the 14-16 CPU cores
  per GPU of the test machines. Only hypre runs on the GPU, and a GPU run is one MPI
  rank: two or more ranks are refused, because they crashed.
- **Stack:** it needs hypre's unified memory (`HYPRE_ENABLE_UNIFIED_MEMORY=ON`, as
  the script builds it), and STRUMPACK (host only); a CUDA stack without STRUMPACK refuses the shift-invert
  cavities and the STRUMPACK full-wave paths (rerun the script: it adds what is missing).

---

## Clusters and batch systems

The source build is the same on a cluster; only the environment differs. [3D designs, MPI and
clusters](guides/3d.md) describes runs on several ranks from the user's side.

- Load a C++17 compiler, MPI, CMake (3.20 or newer) and Python 3.10+ with the site's modules, and keep the same
  ones loaded when running (the libraries record the paths of the MPI they were built with). To have
  `source scripts/env.sh` load them, put the `module load` lines in a file and set `AXF_ENV_HOOK` to it.
- The dependency scripts use the MPI compiler wrappers `mpicc` and `mpicxx`; name others with `MPICC` and
  `MPICXX` (on Cray systems, `MPICC=cc MPICXX=CC`). CMake finds MPI from the wrappers on `PATH`, or the wrapper
  can be the compiler (`-DCMAKE_CXX_COMPILER=CC`).
- Compile in a job allocation: `NPROC` caps the dependency builds, `cmake --build build -j N` the main build.
- Run 3D solves with the site's launcher: `srun -n N axf solve design.json -o run` or `mpirun -n N ...`.
  `Design.solve(mpi_ranks=N)` uses `$AXF_MPI_LAUNCHER` if set, else `srun` inside a SLURM job (when `srun` is on
  `PATH`), else `mpirun`, else `mpiexec`. Choose ranks × `OMP_NUM_THREADS` no larger than the cores of the allocation.
  The multi-rank tests (`AXF_MPI=1`) start their ranks the same way. From inside an `srun` step (an interactive
  `srun --pty bash`, say) set `AXF_MPI_LAUNCHER="srun --overlap"`, or the inner `srun` waits for the resources the
  outer step holds; an MPI that `srun` starts only with a plugin option needs it there too (Open MPI through PMIx:
  `AXF_MPI_LAUNCHER="srun --mpi=pmix"`), or `AXF_MPI_LAUNCHER=mpirun`.
- Some sites start `srun` steps with `SLURM_EXPORT_ENV=NONE`, which drops `AXF_*` and `OMP_*` from the ranks;
  `export SLURM_EXPORT_ENV=ALL` in the batch script. `Design.solve` sets it for the ranks it starts.
- On a single node, OpenMPI may print transport warnings (UCX, InfiniBand) or start slowly;
  `export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` (`self,sm` in OpenMPI 5) selects shared memory. Python sessions
  set these two by default for the MPI they initialize, unless they are already set.

A minimal SLURM script for a 3D solve (fill in the account, partition and sizes for your site):

```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source scripts/env.sh                      # with AXF_ENV_HOOK naming your module loads, if any
srun -n "$SLURM_NTASKS" ./build/axf solve design.json -o run
```

---

## Jupyter

```bash
AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[mesh,notebook]"   # sequential build, in a clone
pip install --no-build-isolation -e ".[notebook]"     # MPI build, in a clone (it meshes with the gmsh SDK)
pip install jupyterlab
jupyter lab examples/quickstart.ipynb
```

```python
res = af.template("toroid", R=2.0, H=2.0).solve()
w = res.show()          # the three.js viewer in the cell
w.quantity = "pickup_Bmag"
w.scale = "log"
```

`anywidget` is optional. Without it `show()` falls back to an `<iframe srcdoc>`
of the same self-contained page `export_html()` writes, which needs no extra
packages; the viewer's own control panel still switches the quantity. See
the [viewer reference](viewer.md) for the widget's API and its limitations.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `cannot find the axf-mesher executable` | `pip install gmsh` (wheel), or build the `axf-mesher` target; `$AXF_MESHER` overrides the search. |
| `libgmsh.so.4.15: cannot open shared object file` | the gmsh wheel is missing or in another environment. `python -c "from axionfem import _paths; print(_paths.prepare_environment())"` shows what was found. |
| `the 3D engine is not available in this build` | expected in a wheel; do a source build against an MPI MFEM. |
| `AXF_MPI=ON but this MFEM ... MFEM_USE_MPI=NO` | `AXF_DEPS` points at the sequential stack. Use `third_party/install`, or configure with `-DAXF_MPI=OFF`. |
| `MFEM was not found`, `MPI was not found`, `.../include/eigen3/Eigen/Core is missing` | the configure step names the missing piece and the script that installs it. |
| `does not match its recorded sha256` | a download was damaged or changed upstream: delete it from `third_party/src` and rerun `scripts/fetch_deps.sh`. |
| a crash in a STRUMPACK solve after `OpenBLAS warning: precompiled NUM_THREADS exceeded` | OpenBLAS's own threads inside axionFEM's OpenMP threads (distribution OpenBLAS builds allow 64 threads): `export OPENBLAS_NUM_THREADS=1`, which `scripts/env.sh` sets unless it is set already. |
| `mpirun -n 2` runs two one-rank solves (Ubuntu 24.04 with MPICH) | that MPICH package is built for PMIx but ships the hydra launcher; use OpenMPI on Ubuntu 24.04. |
| `AXF_DEVICE='cuda' but this build ... has no GPU backend` | configure with `-DAXF_GPU=CUDA` against a CUDA MFEM, or unset `AXF_DEVICE`. |
| `AXF_DEVICE='omp' is not supported by this build` | a GPU build takes `cpu`, `cuda` or `cuda:N` only (other MFEM device strings would put MFEM's vectors where axionFEM's host code cannot see them). |
| `nvcc ... features.h: No such file or directory` | the machine has no C library development headers (some GPU nodes do not): build where they are installed. |
| `GLIBCXX_...  not found` after a cluster build | the build's `libstdc++` is newer than the system's. Keep the compiler module of the build loaded when running (an `AXF_ENV_HOOK` for `scripts/env.sh` can do it). CI wheels avoid this through the manylinux toolchain. |
| the widget shows nothing in JupyterLab | an untrusted notebook: JupyterLab strips `<iframe>` from HTML output. Re-run the cell, or trust the notebook. |
| `import axionfem` works but `_core` is the wrong build | several build trees wrote to `python/axionfem/_core*.so`: give each one its own `-DAXF_INPLACE_PYTHON_DIR`. After an editable install (`pip install -e`), Python loads the module that pip compiled, not the in-place one, even with `PYTHONPATH`: re-run the `pip install --no-build-isolation -e ...` line after C++ changes, or `pip uninstall axionfem` to use the in-place module. |
