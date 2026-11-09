# Chemistry verification cases (beta)

The cases of the beta chemistry module `axionfem.chem`: stiff time integrators (BDF, ESDIRK, variable-order BDF),
the mass-action kinetics kernel, reactor models and DAE equilibrium (groups C1a-C1d), and the steady current
distribution in 1D, planar, axisymmetric and 3D geometry plus the 1D porous electrode (groups C3a, C3b).

| path | content |
|---|---|
| `registry/cases/<id>.json` | the case definitions (format: `benchmarks/fluids/README.md`) |
| `cases/` | the runners (`c1_kinetics.py`, `c3_current_distribution.py`, `c3_electroproblem.py`, `c3b_current_distribution.py` with its MPI worker) and check modules (`c1_kinetics_check.py`, `c3_checks.py`, `c3b_checks.py`) |
| `casedata/` | manufactured fields, mechanisms and kernel reference data, each named in the case parameters with its sha256 |
| `refdata/c3/disk_series.json` | the series solution of the primary current distribution on a disk electrode (used by the tutorial notebook) |

Run them with the shared runner:

```bash
python benchmarks/chem/run.py list
python benchmarks/chem/run.py run --all --cost short --jobs 8
python benchmarks/chem/run.py run chem_mms_c3_axi --out results/
```

The 3D cases (`chem_mms_c3_3d`, `chem_analytic_c3_concentric_3d`) need an MPI build and run on 4 ranks. The
reference values of the cross-code cases come from out-of-band computations with SciPy, SUNDIALS, Cantera, DOLFINx
and axionFEM's own axisymmetric electrostatics; those scripts are not part of this repository, only their numbers
(in the case definitions).
