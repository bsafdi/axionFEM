# Installation

In axion dark-matter experiments, finite-element calculations describe detector fields. On this page we install axionFEM, which uses a C++17 core for these calculations and Python for building designs and inspecting results. The 3D engine requires MPI-enabled MFEM and hypre; sequential builds support axisymmetric calculations. Version 0.1.0 is not yet on PyPI, so we build it from a clone of the repository.

## Build and installation

The build requires Python 3.10+ with development headers, a C++17 compiler, CMake 3.20+, Ninja and curl, and for the MPI build also MPI and BLAS/LAPACK. We build and install axionFEM in a virtual environment so that the build tools and package use the same Python interpreter. These commands are tested in clean Ubuntu 24.04 and Rocky Linux 9 containers (`docker/`) and with a Cray programming environment, and the sequential build is tested on macOS (arm64) in CI; the MPI build has not been tested on macOS yet, and Windows is not supported.

### Prerequisites

**Debian/Ubuntu**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config git curl ca-certificates \
    python3-dev python3-venv libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev
```

On Ubuntu 24.04 we use OpenMPI: its MPICH package builds axionFEM, but its `mpirun` starts every rank as a separate one-rank job.

**RHEL 9 and its rebuilds (Rocky, Alma)**

```bash
sudo dnf install epel-release dnf-plugins-core && sudo dnf config-manager --set-enabled crb
sudo dnf install gcc gcc-c++ make cmake ninja-build pkgconf-pkg-config git tar gzip \
    python3.12 python3.12-devel openmpi-devel openblas-devel
export PATH=/usr/lib64/openmpi/bin:$PATH    # or: module load mpi/openmpi-x86_64
```

The system Python of RHEL 9 is 3.9, so the virtual environment below is created with `python3.12`.

**macOS (Homebrew)**

```bash
xcode-select --install
brew install cmake ninja open-mpi libomp python@3.12
```

Homebrew's Python is `python3.12`; BLAS/LAPACK come from Apple's Accelerate framework.

On a shared cluster, we load a C++17 compiler, MPI, CMake and Python with the site's modules and run compilation in a compute allocation. The compiler and MPI environment should remain available when using the installed package. On Cray systems the dependency scripts take `MPICC=cc MPICXX=CC`.

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

The dependency scripts download and build MFEM, hypre, METIS and the other pinned libraries. CMake builds `libaxionfem`, the Python bindings, `axf` and the separate gmsh mesher. It detects MPI support from MFEM and then also builds the fluids module (beta). `ctest` checks the C++ components; the final command installs the Python package. For the example notebooks and the MCP server we install the extras instead: `pip install --no-build-isolation -e ".[notebook,plots,draw,mcp,test]"`. With a threaded OpenBLAS (Ubuntu's default), we set `export OPENBLAS_NUM_THREADS=1` before running; `source scripts/env.sh` sets it.

### Sequential source build

```bash
git clone https://github.com/bsafdi/axionFEM && cd axionFEM
python3 -m venv .venv      # Python 3.10 or newer; on macOS (Homebrew) and RHEL 9: python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
bash scripts/build_deps_wheel.sh
AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[all,test]"
```

This build needs no MPI or BLAS/LAPACK and is the configuration of the planned wheels. It has everything except the 3D engine, the fluids module and the 3D current distributions of the chemistry module. Its mesher uses gmsh's PyPI wheel, which on Linux needs a few X11/OpenGL libraries even without a display (Debian/Ubuntu: `libglu1-mesa libgl1 libxcursor1 libxft2 libxinerama1`).

[Build options and packaging](../install.md) describes the wheels, every CMake option, the optional STRUMPACK and GPU builds, clusters and troubleshooting.

## Python installation check

We inspect the build information to check the available engines.

```python
import axionfem as af

print(af.build_info())
```

The dictionary reports versions, device, threads and available engines. For the MPI source build, `mpi` and `engine_3d` should both be `True`. A false `engine_3d` means an MPI-enabled build is needed for 3D designs. The next step is the [quick start](../../examples/notebook_quickstart.ipynb); JupyterLab is installed separately if needed with `pip install jupyterlab`.

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

Running `axf` without arguments displays usage. `mpirun -n N axf solve design.json -o run` solves a 3D design on `N` MPI ranks. The corresponding workflows are described in [Designs and solvers](solvers.md), [Visualization](visualization.md) and [COMSOL export and import](comsol.md).
