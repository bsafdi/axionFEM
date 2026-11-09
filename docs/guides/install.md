# Installation

On this page we install axionFEM: a C++17 core that performs the finite-element solves, and a Python package for building designs and inspecting results.

There are two ways to install it, both from a clone of the repository:

| | what you get | what it needs |
|---|---|---|
| [Sequential source build](#sequential-source-build) | axisymmetric and full-wave engines, scan reach and cost metric, viewer, widget, CLI, MCP server | a C++17 compiler, CMake |
| [MPI source build](#mpi-source-build) | the same, plus the **3D engine** and the fluids module (beta) | a C++17 compiler, MPI, BLAS/LAPACK, CMake |

Version 0.1.0 is not yet on PyPI: install it from a clone of the repository with one of the two source builds. The [wheel](#wheel) section describes the packaged form for when it is published.

The 3D engine needs MPI-enabled MFEM and hypre. A wheel on PyPI cannot require a system MPI, so the wheels and the sequential build are serial; a 3D design there raises an error that points to the MPI source build. The fluids module (beta) and the 3D current distributions of the chemistry module (beta) need the MPI build as well.

## Wheel

Wheels of the sequential build are not yet on PyPI. Once published, they install with

```bash
pip install "axionfem[mesh,notebook]"
```

The release workflow builds them for Linux x86_64 (manylinux_2_28) and macOS (x86_64 and arm64), CPython 3.10 to 3.13; until then, and where no wheel is available, the [sequential source build](#sequential-source-build) gives the same package. The extras below apply to both. In a clone they are added at install time: after the sequential build
`AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[mesh,notebook]"`, after the MPI build
`pip install --no-build-isolation -e ".[notebook]"` (the MPI build meshes with the gmsh SDK, so it needs no `mesh`
extra). A plain dependency such as `mcp` or `matplotlib` can also be installed on its own with `pip install` into the
same environment.

| extra | adds |
|---|---|
| `mesh` | gmsh, needed to mesh anything |
| `notebook` | the Jupyter widget (anywidget) |
| `plots` | matplotlib, for `axionfem.plots` |
| `draw` | circuit drawings (schemdraw) |
| `mcp` | the MCP server for AI assistants |
| `test` | pytest and the test dependencies |
| `all` | `mesh`, `notebook`, `plots`, `draw` and `mcp` together |

The `mesh` extra installs gmsh's own PyPI wheel, which the mesher loads at run time. On Linux, that gmsh wheel needs a few X11/OpenGL libraries even without a display; on Debian/Ubuntu they are `libglu1-mesa libgl1 libxcursor1 libxft2 libxinerama1` (Fedora/RHEL: `mesa-libGLU libXcursor libXft libXinerama`).

## Source builds

These instructions are tested in clean x86_64 containers built from `docker/`, whose images install exactly the packages listed below, by running the build commands as written: Ubuntu 24.04 with OpenMPI (MPI build) and without MPI (sequential build), and Rocky Linux 9 with OpenMPI and with MPICH, each with the distribution's compilers and OpenBLAS. The MPI build was also run with a Cray programming environment (GCC and Cray MPICH, `MPICC=cc MPICXX=CC`). macOS (Apple clang and Homebrew) has not been tested yet; CI jobs for it are configured (the sequential build on every pull request, the MPI build weekly) but have not run. Other Linux distributions with the same tools should work; Windows is not supported.

### Prerequisites

**Debian/Ubuntu**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config git curl ca-certificates \
    python3-dev python3-venv libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev
```

Use OpenMPI on Ubuntu 24.04: its MPICH package (`libmpich-dev mpich`) builds axionFEM, but its `mpirun` starts every rank as a separate one-rank job, so 3D solves on several ranks do not run in parallel. MPICH from other sources works (it is tested with RHEL 9's `mpich-devel`).

**RHEL 9 and its rebuilds (Rocky, Alma)**

```bash
sudo dnf install epel-release dnf-plugins-core && sudo dnf config-manager --set-enabled crb
sudo dnf install gcc gcc-c++ make cmake ninja-build pkgconf-pkg-config git tar gzip \
    python3.12 python3.12-devel openmpi-devel openblas-devel
export PATH=/usr/lib64/openmpi/bin:$PATH    # or: module load mpi/openmpi-x86_64
```

For MPICH, install `mpich-devel` instead of `openmpi-devel` and put `/usr/lib64/mpich/bin` on `PATH` (`module load mpi/mpich-x86_64`). The system Python of RHEL 9 is 3.9; the build uses Python 3.12 from the distribution. On Fedora the same packages come without the first line (not tested).

**macOS (Homebrew)**

```bash
xcode-select --install
brew install cmake ninja open-mpi libomp python@3.12
```

Homebrew's `python@3.12` is installed as `python3.12`; the `python3` on `PATH` is usually Apple's Python 3.9, which is too old, so the build steps below create the virtual environment with `python3.12`. BLAS/LAPACK come from Apple's Accelerate framework, and the gmsh SDK for the Mac's architecture is fetched. CMake finds Homebrew's `libomp` by itself, for the CMake build and for `pip install` alike; without it, the build is single-threaded. The 3D engine on macOS has not been validated against the Linux results.

**Clusters**

Load a C++17 compiler, MPI, CMake (3.20 or newer) and Python 3.10+ with your site's module system. `source scripts/env.sh` puts the dependency prefix on the search paths; to run site setup at the same time, put the `module load` lines in a file and set `AXF_ENV_HOOK` to it. The dependency scripts use the MPI compiler wrappers `mpicc` and `mpicxx` unless `MPICC` and `MPICXX` name others (on Cray systems, `MPICC=cc MPICXX=CC`). Compile in a job allocation rather than on a login node, and keep the same compiler and MPI loaded when running.

### MPI source build

```bash
git clone https://github.com/bsafdi/axionFEM && cd axionFEM
python3 -m venv .venv      # Python 3.10 or newer; on macOS (Homebrew) and RHEL 9: python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "scikit-build-core>=0.9" "nanobind>=2.0" numpy
bash scripts/fetch_deps.sh
bash scripts/build_deps.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
pip install --no-build-isolation -e ".[test]"
```

`scripts/fetch_deps.sh` downloads the pinned sources of MFEM 4.10, hypre 2.33, METIS 5.1, Eigen 3.4, nlohmann/json 3.12 and the gmsh 4.15 SDK, and checks each archive against `third_party/SHA256SUMS.txt`. `scripts/build_deps.sh` builds METIS, hypre and MFEM (with MPI) into `third_party/install`; it uses all cores unless `NPROC` limits them, and the variables at the top of the script select other compilers, a BLAS vendor (`AXF_BLA_VENDOR`) or another prefix. CMake builds `libaxionfem`, the Python bindings, the `axf` command and the separate gmsh mesher; it enables the 3D engine because the MFEM build has MPI, and with it the fluids module (beta). `ctest` runs the C++ unit tests, and the last command installs the Python package, compiling it a second time for the Python environment. On Linux x86_64 the gmsh SDK is the headless build, so no X11 or OpenGL libraries are needed.

For the example notebooks and the MCP server, install the extras without `mesh` in the last step:
`pip install --no-build-isolation -e ".[notebook,plots,draw,mcp,test]"`. With a threaded OpenBLAS (Ubuntu's default
`libopenblas0-pthread`), set `export OPENBLAS_NUM_THREADS=1` before running: OpenBLAS's own threads otherwise compete
with axionFEM's OpenMP threads (one 3D test took 1.8 times as long without it). `source scripts/env.sh` sets it.

### Sequential source build

```bash
git clone https://github.com/bsafdi/axionFEM && cd axionFEM
python3 -m venv .venv      # Python 3.10 or newer; on macOS (Homebrew) and RHEL 9: python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
bash scripts/build_deps_wheel.sh
AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[all,test]"
```

The `all` extra adds what the example notebooks and the MCP server use (gmsh, the widget, matplotlib, schemdraw,
the MCP package) and `test` adds pytest and jsonschema (NumPy and SciPy are dependencies of the package itself);
`".[mesh]"` alone is the minimal install. This is the
configuration of the wheels: a serial MFEM and METIS in `third_party/install-seq`, no MPI, and the mesher linked against gmsh's PyPI wheel, which needs the X11/OpenGL libraries listed under [Wheel](#wheel) on Linux. The prerequisites are those above without MPI and BLAS/LAPACK.

### Optional libraries

| script | what it adds | needs |
|---|---|---|
| `scripts/build_strumpack.sh` | STRUMPACK (serial + OpenMP) for large axisymmetric systems; reconfigure CMake afterwards | a Fortran compiler (`gfortran`), OpenMP, BLAS/LAPACK |
| `scripts/build_strumpack_mpi.sh` | distributed STRUMPACK and ScaLAPACK for the 3D direct full-wave path (`-DAXF_STRUMPACK_PREFIX=$PWD/third_party/install_mpi`) | MPI Fortran wrapper (`mpif90`) |
| `scripts/build_deps_gpu.sh` | a CUDA stack for `-DAXF_GPU=CUDA`; `AXF_CUDA_ARCH` lists the GPU architectures (default `80;90`) | the CUDA toolkit |

[Build options and packaging](../install.md) describes every CMake option and these builds in detail.

## Checking the installation

We inspect the build information to check the available engines.

```python
import axionfem as af

print(af.build_info())
```

The dictionary reports versions, device, threads and available engines. For the MPI source build, `mpi` and `engine_3d` should both be `True`; a false `engine_3d` means a 3D design needs the MPI build. The test suite runs with `pytest -q tests` (the 3D test files skip on the sequential build); `AXF_MPI=1` adds the tests that use several MPI ranks and `AXF_SLOW=1` the slow 3D tests (several hours in total). The next step is the [quick start](../../examples/quickstart.ipynb); JupyterLab is installed separately if needed with `pip install jupyterlab`. [Running the tests](../developer/testing.md) describes the test suite.

## Command line

Installation puts `axf` on `PATH`. These commands cover a typical design-to-results workflow:

| Command | Result |
|---|---|
| `axf template toroid > design.json` | Writes a starting design. |
| `axf validate design.json` | Checks the design's structure and values. |
| `axf mesh design.json -o run` | Generates a mesh in `run/`. |
| `axf solve design.json -o run --scene` | Writes `run/results.json` and `run/scene.json`. |
| `axf render run/scene.json -o field.png` | Renders a field image. |
| `axf schema` | Prints the design JSON Schema. |

Running `axf` without arguments displays usage. `mpirun -n N axf solve design.json -o run` solves a 3D design on `N` MPI ranks, and `Design.solve(mpi_ranks=N)` does the same from Python. The corresponding workflows are described in [Concepts](concepts.md), [Visualization](visualization.md) and [COMSOL export and import](comsol.md); [3D designs, MPI and clusters](3d.md) covers runs on several ranks and batch jobs.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `MFEM was not found` when configuring | build the dependencies first (`scripts/build_deps.sh`), or point `-DAXF_DEPS` at their prefix. |
| `MPI was not found` | install MPI with its compiler wrappers, or pass `-DMPI_CXX_COMPILER=/path/to/mpicxx`. |
| `does not match its recorded sha256` | a download was damaged or changed upstream: delete the file in `third_party/src` and run `fetch_deps.sh` again. |
| `nanobind not found` (warning) and no Python module | install `nanobind` into the Python that CMake found, or pass `-DPython_EXECUTABLE`. |
| `cannot find the axf-mesher executable` | wheel: `pip install gmsh`; source build: check that CMake reported `axf-mesher: gmsh ...`. |
| `libGLU.so.1: cannot open shared object file` | the gmsh PyPI wheel needs the X11/OpenGL libraries listed under [Wheel](#wheel). |
| `the 3D engine is not available in this build` | expected in a wheel or sequential build; use the MPI source build. |
| a crash in a STRUMPACK solve after `OpenBLAS warning: precompiled NUM_THREADS exceeded` | OpenBLAS's own threads inside axionFEM's OpenMP threads on a machine with many cores: `export OPENBLAS_NUM_THREADS=1` (`source scripts/env.sh` sets it). |
| OpenMPI warnings about UCX, InfiniBand or `openib` on a single machine | `export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` (OpenMPI 5: `self,sm`) selects the shared-memory transport. |
| `GLIBCXX_... not found` when running | the build's `libstdc++` is newer than the system's: load the same compiler module when running. |
| a C++ change or a new CMake option does not show in Python (`build_info()` unchanged) | an editable install (`pip install -e`) loads the module it compiled itself, not the one `cmake --build build` writes into `python/axionfem/`, even with `PYTHONPATH`: run the `pip install --no-build-isolation -e ...` line again (with the same `-C cmake.define` options), or `pip uninstall axionfem` and use the in-place module. |
