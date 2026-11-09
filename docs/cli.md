# Command line: `axf`

`axf` turns a detector design into a mesh, solved fields and result files.
[Installing axionFEM](guides/install.md) adds it to your `PATH`. The console
script runs the compiled command line shipped with the package
(`axionfem/bin/axf`, or `build/axf` in a source checkout); set `AXF_BIN` to
select a different binary. `export-comsol`, `import-comsol` and
`export-touchstone` are handled in Python before the binary is looked for.

| Command | What it does |
|---|---|
| `axf solve design.json [-o DIR] [--order P] [-v N] [--mesher M] [--scene] [--vtu] [--save] [--no-self-force] [--force-samples N] [--status-file F]` | Solve and write `DIR/results.json`; optionally export a viewer scene (`--scene`), ParaView files (`--vtu`) or saved 3D fields (`--save`). The summary printed on stdout (success or error) is also written by rank 0 to `DIR/axf_status.json` (or `F`). |
| `axf sensitivities design.json [--param P]... [--objective Q]... [--method adjoint\|fd_morph\|fd_remesh] [--step S] [--benchmark NAME] [-o DIR] [--order P] [--mesher M] [-v N]` | Compute axisymmetric [shape sensitivities](optimization.md). |
| `axf validate design.json` | Check the design. |
| `axf mesh design.json [-o DIR] [--mesher M] [-v N]` | Write `DIR/mesh.msh` or `DIR/mesh3d.msh`. |
| `axf resolve design.json` | Print the geometry with expressions evaluated. |
| `axf schema` | Print the [design format](design_format.md) JSON Schema. |
| `axf requirement [--benchmark NAME]` | Print the scan requirement 𝓕_req for the built-in noise benchmarks (`optimistic`, `conservative`, `conservative_solenoid`; all three by default). |
| `axf render scene.json -o out.png [--view section\|3d] [--quantity Q] [--log] [--theme dark]` | Write a [PNG snapshot](render.md). |
| `axf templates` | List parametric templates. |
| `axf template NAME [--set key=value ...]` | Print a design from a template. |
| `axf export-comsol SOURCE [-o OUT] [--format mphtxt\|java\|step] ...` | [Export a design to COMSOL](comsol_export.md). |
| `axf import-comsol model.mph [-o design.json] ...` | [Import a COMSOL model](comsol_import.md). |
| `axf export-touchstone results.json [-o OUT.sNp] [--kind S\|Z\|Y] [--z0 50] [--dense] [--surface-losses]` | Write full-wave port parameters as [Touchstone v1](physics/touchstone.md) (time convention e^{+jωt}). Exit status 0: written; 3: written but not passive; 2: error (message on stderr). |

```bash
axf template toroid --set R=2.0 --set H=2.0 > toroid.json
axf validate toroid.json
axf solve toroid.json -o runs/toroid --scene
axf render runs/toroid/scene.json -o toroid.png --view 3d --quantity Bmag
```

This writes the detector's fields and coupling results to
`runs/toroid/results.json`, a viewer scene to `runs/toroid/scene.json`, and a
magnetic-field snapshot to `toroid.png`.

`axf solve` picks the solver from the design: axisymmetric designs (the
default), z-uniform cavity cross-sections (`"symmetry": "xsec"`), 2D full-wave
models on an imported mesh (`mesh.file`), and 3D designs (`"symmetry": "3d"`).
Run 3D designs with an MPI-enabled build under an MPI launcher
(`mpirun -n N axf solve ...` or your scheduler's equivalent). Axisymmetric,
cross-section and 2D imported-mesh designs run on one process; under a
multi-rank launch only rank 0 works (an axisymmetric result then carries a
warning that the other ranks stayed idle).

Run `axf` without arguments for usage, or use `axf export-comsol --help`,
`axf import-comsol --help` and `axf export-touchstone --help` for their options.
Solver and validation errors are JSON with an `error` message and, when
available, `where` and `hint` fields. Under MPI launchers that split or
interleave long stdout lines, read the summary from the status file rather than
from stdout; `Design.solve(mpi_ranks=N)` does that, with a status file of its
own per launch (`--status-file`).
