# Fluids and chemistry verification cases (beta)

The fluids (`axionfem.fluids`, `axionfem.transport`) and chemistry (`axionfem.chem`) modules are **beta**. This
directory holds their verification cases: manufactured solutions (MMS), closed-form solutions, conservation
identities and comparisons with other codes, each with explicit acceptance criteria. The chemistry cases live in
`benchmarks/chem/` and use the same runner.

## Layout

| path | content |
|---|---|
| `registry/cases/<id>.json` (fluids), `../chem/registry/cases/<id>.json` (chemistry) | one case definition per file |
| `cases/<module>.py`, `../chem/cases/<module>.py` | the case runners (set-up, solve through the public Python API, raw numbers) and their check modules (norms, identities, orders; numpy only) |
| `acceptance.py` | `evaluate(case, result)`: judges a runner's raw result against the case's acceptance items |
| `run.py` | command-line runner: list, show, run cases and check their criteria |
| `../../tools/fluids_mms/` | the generators of the manufactured fields (`out/*.json`) and of the reference values (series, cross-code); see its README |
| `../chem/casedata/` | the chemistry case data (manufactured fields, mechanisms) |

A case definition has these fields:

- `id`, `capability` (the group the case belongs to, see below), `title`, `physics`;
- `levels`: which kinds of evidence the case provides (`mms`, `analytic`, `conservation`, `cross_code`,
  `literature`);
- `run.runner`: `module` and `function` of the runner (a function `fn(params, workdir, quick)` returning
  `{"runs", "series", "quantities", "notes"}`), its `check_module`, a `cost` class (`short`: seconds to a few minutes
  on one core; `long`: minutes to hours, some with MPI) and a `timeout_s`;
- `run.parameters`: everything the runner reads: meshes and refinement levels, physical data, solver settings, and the
  data files it loads, each as `{"path", "sha256"}` (the runner refuses a file whose hash differs);
- `reference.quantities`: reference values with their uncertainty and source (`closed_form`, a published source, or
  another code);
- `acceptance`: the criteria, one item per checked number: `convergence` (observed order of an error series),
  `functional_convergence` (Richardson extrapolation and grid-convergence index), `comparisons` (a quantity against a
  reference value with a tolerance), `checks` (identities and bounds), `solver_sensitivity` and
  `quadrature_sensitivity`, `tolerance_behaviour` (adaptive integrators) and `solver` (every solve converged).

The verdict columns mean only what they say: `self_consistency` (MMS orders, identities), `exact_solution`
(agreement with a closed form), `cross_code` (agreement with another code solving the same discrete problem) and
`literature` (published values). No current case tests a literature value.

Capability groups (the ids are also the prefixes of the runner modules):

| group | content |
|---|---|
| F1a, F1b, F1c | Taylor-Hood Stokes: planar, axisymmetric with swirl, 3D (MPI) |
| F2a | steady Navier-Stokes, planar and axisymmetric |
| F3a | transient Navier-Stokes, BDF1/BDF2 |
| F4a | scalar transport (SUPG) and Boussinesq natural convection |
| F5 | stabilized P1/P1 (GLS) |
| F13a | inductionless MHD (2.5D planar) |
| C1a-C1d | chemistry: time integrators, kinetics kernel, reactors and equilibrium, variable-order BDF |
| C3a, C3b | chemistry: current distribution (1D, planar, axisymmetric), 3D (MPI) and porous electrode |

## Running

```bash
python benchmarks/fluids/run.py list                       # every case, its group, cost and runner
python benchmarks/fluids/run.py show mms_stokes_2d         # the acceptance items of one case
python benchmarks/fluids/run.py run analytic_couette_2d    # run one case and check its criteria
python benchmarks/fluids/run.py run --cost short --jobs 8 --out results/   # every short case
python benchmarks/fluids/run.py run --all --quick --jobs 16                # smoke test of every case
python benchmarks/chem/run.py run --all --cost short       # the short chemistry cases
python benchmarks/fluids/run.py report results/ --track fluids             # Markdown tables of saved results
```

The fluids module exists only in MPI builds; the runner uses the installed `axionfem` or, if none is importable, the
in-place module under `python/` of the checkout. Cases on several ranks (the 3D Stokes and 3D current-distribution
cases) start MPI through the launcher of `axionfem` (`$AXF_MPI_LAUNCHER`, else `srun` inside a SLURM job, else
`mpirun`/`mpiexec`); inside an interactive `srun` step set `AXF_MPI_LAUNCHER="srun --overlap"`.

`run` prints one line per case (`PASS`, `FAIL`, `ERROR`, `TIMEOUT`) and, with `--out`, writes the raw result and the
item-by-item evaluation of each case as JSON plus a `summary.json`. A full run exits with status 1 unless every case
meets all its criteria.

`report` turns saved results (one or more `--out` directories; a full result wins over a quick one) into one Markdown
table per group, headed by the group's title: the observed orders against the expected ones, the largest comparison
deviation as a fraction of its allowance, the number of checks, and whether the case was run in full, in quick mode
only, or not at all. A closing line gives the totals: cases run in full and in quick mode, cases meeting all their
criteria, and the numbers of observed orders, Richardson estimates, comparisons and checks in the full runs.

`--quick` runs only the first two refinement levels of every series. It is a smoke test of the whole pipeline (set-up,
solve, check module, acceptance) in a fraction of the time; order and Richardson items need more levels and are
usually reported as failed (status `SMOKE`), so a quick run fails only on errors and timeouts.

## What this verification does and does not cover

The cases check the discretizations (orders of convergence on manufactured and closed-form solutions, discrete
identities, agreement with NGSolve, DOLFINx/FEniCSx or Cantera on the same discrete problem) and the solvers'
convergence. They do not validate physical models against experiments, and no published benchmark value is compared
yet. The measured results are summarized (as `report` tables) in `docs/validation/fluids.md` and
`docs/validation/chemistry.md`; the formulation is described in `docs/guides/fluids.md` and `docs/guides/chemistry.md`.
