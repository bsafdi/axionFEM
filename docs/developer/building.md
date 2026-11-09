# Building from source

The [installation guide](../guides/install.md) has the prerequisites for each platform and the exact, tested
command sequences for the MPI and the sequential source builds; [Build options and packaging](../install.md) lists
every CMake option and dependency script. This page adds what a developer needs on top: an in-place build that the
Python package uses directly, rebuilding after a change, and several build trees side by side.

## An in-place development build

The MPI source build of the guide configures `build/` and installs the package in editable mode. For day-to-day work
the editable install is optional: a CMake build writes the compiled module `_core` into `python/axionfem/`
(`AXF_INPLACE_PYTHON_DIR`), and the package finds `build/axf` and `build/axf-mesher` next to it, so a checkout runs
with `PYTHONPATH`:

```bash
source scripts/env.sh                       # puts third_party/install on the search paths
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                         # libaxionfem, _core, axf, axf-mesher, the C++ tests
ctest --test-dir build --output-on-failure
PYTHONPATH=$PWD/python python -c "import axionfem as af; print(af.build_info())"
```

After a change to C++ code, `cmake --build build` rebuilds what changed and refreshes `_core`; a change to Python
code needs no rebuild. `pip install --no-build-isolation -e ".[test]"` compiles the project a second time, for the
Python environment, and is only needed when the package must be importable without `PYTHONPATH`.

The two do not mix: once the package is installed in editable mode, `import axionfem` loads the `_core` that pip
compiled into the environment, even with `PYTHONPATH=$PWD/python`, so `cmake --build build` no longer changes what
Python runs. Either re-run the `pip install --no-build-isolation -e ...` line after each C++ change (with the same
`-C cmake.define` options as the first time), or `pip uninstall axionfem` and work with the in-place module. Multi-rank
solves (`Design.solve(mpi_ranks=N)`) always launch `build/axf` (or `$AXF_EXE`), so after a reinstall with other
options the in-process and the multi-rank engines can differ; rebuild both after a change of options.

## Which build is loaded

`af.build_info()` reports the version, the features (`mpi`, `engine_3d`, `openmp`, `metis`, `strumpack`, `fluids`),
the git commit of the sources the module was built from (`git_sha`, `git_dirty`) and where the loaded `libaxionfem`
lives. Check it whenever more than one build exists. The executables can be named explicitly:

| variable | selects |
|---|---|
| `AXF_EXE` | the `axf` executable used by `Design.solve` for multi-rank runs |
| `AXF_BIN` | the `axf` executable behind the `axf` console script (otherwise the same search as `AXF_EXE`) |
| `AXF_MESHER` | the `axf-mesher` executable |
| `AXF_THREADS` | the OpenMP thread count of the core |

## Several build trees

Each build tree writes its `_core` into `AXF_INPLACE_PYTHON_DIR`, by default `python/axionfem`; CMake writes only the
compiled module there. To keep several trees without overwriting each other's module, give each its own directory
holding a copy of the package sources, and point `PYTHONPATH` at its parent:

```bash
mkdir -p runs/pkg-debug && cp -r python/axionfem runs/pkg-debug/
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DAXF_INPLACE_PYTHON_DIR=$PWD/runs/pkg-debug/axionfem
cmake --build build-debug
PYTHONPATH=$PWD/runs/pkg-debug AXF_EXE=$PWD/build-debug/axf AXF_MESHER=$PWD/build-debug/axf-mesher python ...
```

The usual variants are a sequential tree against the wheel stack (`-DAXF_DEPS=$PWD/third_party/install-seq`, built by
`scripts/build_deps_wheel.sh`), which reproduces the wheel configuration and CI's sequential job, and trees with the
optional STRUMPACK or GPU stacks ([build options](../install.md#sparse-direct-solvers)).

## Compilers and clusters

The dependency scripts take the compilers from `CC`, `CXX`, `FC` and the MPI wrappers from `MPICC`, `MPICXX`,
`MPIFC`; on Cray systems `MPICC=cc MPICXX=CC MPIFC=ftn`. On a cluster, compile in a job allocation, keep the same
compiler and MPI loaded when running, and put site setup (`module load` lines) in a file named by `AXF_ENV_HOOK`, which
`scripts/env.sh` runs first ([clusters](../install.md#clusters-and-batch-systems)).
