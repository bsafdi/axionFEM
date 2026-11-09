# Running the tests

axionFEM has two test suites: C++ unit tests run by CTest, and the pytest suite in `tests/`. Most physics tests
compare a solve with a closed-form solution, a manufactured solution, an independent solver or a published result;
the [validation pages](../validation/index.md) describe what each comparison checks.

## C++ unit tests

```bash
ctest --test-dir build --output-on-failure
```

They cover the expression parser, mesh quality and topology checks, the open-boundary maps, the sparse solvers and
STRUMPACK threading, the scan and surface-resistance models, the renderer, the rational fit of the adaptive sweep, and
the C++ parts of the fluids (MPI builds) and chemistry modules.

## The pytest suite

```bash
python -m pytest -q tests                        # the default suite
python -m pytest -q tests/test_cavity.py -k tm010   # one file, tests matching a name
```

The package must be importable: installed, or a source checkout with `PYTHONPATH=python` after a CMake build
([Building from source](building.md)). Tests that need something the build or the machine does not have skip with a
reason: the 3D engine (wheels and sequential builds), the fluids module (MPI builds), an optional package such as
`schemdraw`, or an external data file.

Environment variables select the heavier tests:

| variable | adds |
|---|---|
| `AXF_MPI=1` | tests that start several MPI ranks (up to 16, so the allocation needs 16 slots) |
| `AXF_SLOW=1` | the slow 3D tests: quantitative 3D cross-checks, convergence studies, large full-wave cases |
| `AXIONFEM_BROWSER_TESTS=1` | a headless rendering smoke test of the viewer (needs Firefox) |

`tests/test_gpu.py` runs only with a CUDA build and a visible GPU. The full suite (`AXF_SLOW=1 AXF_MPI=1`) takes
many hours: in a measured run (an MPI build with the serial STRUMPACK, 8 threads per test, 16 tests running side
by side on one 128-core node, which inflates the times somewhat), the per-test times added up to 11 hours, of which
`tests/test_fullwave3d_distributed_port.py` alone took 2 h 47 min and seven other files 20-95 minutes each. Split the
test files over several jobs when the time limit is shorter.

Multi-rank tests start their ranks the way `Design.solve` does (`tests/_mpi_launch.py`): `$AXF_MPI_LAUNCHER` if set,
else `srun` inside a SLURM job, else `mpirun`, else `mpiexec`. From inside an `srun` step set
`AXF_MPI_LAUNCHER="srun --overlap"`.

## The quick subset

`tests/premerge.txt` lists a curated subset of the suite, one pytest node ID per line, that represents every test file
and every engine and solver path: a quick check before merging a branch, in about 30 minutes on one node with 16
tasks of 8 threads (with `AXF_SLOW=1 AXF_MPI=1`). `scripts/premerge_tests.py` keeps the list honest:

```bash
python scripts/premerge_tests.py check    # seconds: every listed test exists, every test file is listed or excluded
python scripts/premerge_tests.py run      # check, then run the listed tests (sets AXF_SLOW=1 and AXF_MPI=1)
```

`run` fails if any test fails, and also if a test named on its own line is skipped: a skipped test covers nothing.
When you add a test file, give it a fast representative in the list or an `exclude` line that says why; when you
rename or remove a test, update the list.

## On a cluster

`scripts/slurm/tests_premerge.slurm` (the quick subset) and `scripts/slurm/tests.slurm` (the whole suite) are example
SLURM jobs. Add your site's account, partition and QoS on the command line:

```bash
sbatch [-A ACCOUNT] [--partition=P] [--qos=Q] scripts/slurm/tests_premerge.slurm
sbatch [-A ACCOUNT] [--partition=P] [--qos=Q] --export=ALL,AXF_SLOW=1 scripts/slurm/tests.slurm
```

Both run `ctest` first and take `AXF_BUILD_DIR`, `AXF_PYTHONPATH`, `AXF_PYTHON` and `AXF_ENV_HOOK` to select the build,
the package and the site setup; the header of each script lists the details. Both also set `SLURM_EXPORT_ENV=ALL`:
some sites start `srun` steps with `NONE`, which drops the `AXF_*` and `OMP_*` variables from the ranks, so do the same
in your own batch scripts that call `srun`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and push to `main`:

| job | what it builds and runs |
|---|---|
| `linux-sequential` | the wheel configuration (serial MFEM, no MPI): `ctest` and the pytest suite without the 3D engine |
| `macos-sequential` | the install guide's macOS prerequisites and sequential build as written, then `pytest` and `ctest` |
| `linux-mpi` | OpenMPI and OpenBLAS with the MPI dependency stack: `ctest`, the 3D tests of the quick subset, the fluids and chemistry tests and a 3D solve on two ranks |
| `packaging` | the file-based packaging tests and an sdist check |

Weekly and on demand, `linux-mpi` runs the whole suite instead, and a macOS MPI build follows the guide. Changes that
touch only documentation run nothing there; the documentation workflow builds the site with `mkdocs build --strict`
([Documentation and figures](docs.md)).

## Writing tests

- **Compare with something independent**: a closed form, a manufactured solution, a second formulation, a published
  number. Say in the test where the reference comes from.
- **3D tests skip without the 3D engine.** A file that builds 3D designs carries
  `pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"], reason=...)` or a per-test guard;
  `premerge_tests.py check` and `tests/test_packaging.py` fail on a 3D file without it. Tests of beta modules skip when
  their module is absent.
- **Slow cases** go behind `AXF_SLOW`, multi-rank cases behind `AXF_MPI`.
- **Reference data** that a test reads is generated by a script in `benchmarks/`, never edited by hand; regenerate the
  record when the computation changes.
