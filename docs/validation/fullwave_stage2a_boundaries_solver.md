# Closed RF-only models: symmetry surfaces, PEC holes, lumped gaps and direct factorization

!!! warning "Beta"
    The RF-only closed-PEC path (`fullwave.rf_only`, `mesh.symmetry`, `mesh.remove_pec`) is beta.

The closed-PEC RF-only path supports named electric and magnetic symmetry
surfaces, conductor interiors removed from the mesh, and lumped gaps ending
on a PEC surface or electric image plane. The mesher clips the complete CAD,
preserves connected PEC body identities in boundary attributes, and exports
named boundary groups. The H(curl) electric wall is essential; the magnetic
wall is the natural weak-form boundary. This feature does not make arbitrary
asymmetric hardware a valid quarter model.

`mesh.symmetry` contains objects such as
`{"name":"slit","axis":"y","side":1,"type":"pec"}` and
`{"name":"midplane","axis":"z","side":1,"type":"pmc"}`.
`mesh.remove_pec=true` removes PEC volumes while retaining their
boundary identities. `mesh.domain_solid` selects the actual closed domain.
An existing mesh can be supplied using `mesh.file` and `mesh.info_file`.
`fullwave.rf_only=true` bypasses the DC primary-return requirement; a
capacitor-only geometry still needs a genuine gap with exactly two distinct
terminal patches. No static inductance or flux is fabricated in this mode.

For mirrored module pairs with a bisected differential readout (a quarter model),
physical pair-total currents give `Z_phys=Z_quarter`,
`Psi_phys=2 Psi_quarter`, and `G_phys=G_quarter`. A physical unit-current
field is half the exported quarter unit-current field. Each mirror pair
uses `C_pair=2 C_module`. Full overlap/energy at fixed quarter fields is four
times the quarter integral. These are coordinate maps for the declared
ports, not a universal symmetry multiplication rule. Apply FDT to the
physical impedance; independent bath noise is not a coherent source image.

## Solver selection

`AXF_FW_SOLVER=ams` remains the default. `AXF_FW_SOLVER=strumpack` selects
one real sparse factorization per frequency, reused for both complex parts
and all port, direct-source and loaded-validation right-hand sides. It
requires a closed, lossless real Maxwell block and rejects nonzero imaginary
material operators or exterior low-rank updates. Reactive validation loads
are supported through GMRES refinement against the original loaded operator.

With the serial STRUMPACK, or with `AXF_FW_STRUMPACK_DIST=0`, the real Hypre rows are gathered in MPI true-DOF
order to rank zero, scaled symmetrically, and factorized using the existing STRUMPACK backend. This is a root-owned
factorization, **not a distributed-memory factorization**; factor fill and rank-zero memory constrain its useful
size. The distributed alternative is described below. Original
distributed-operator true residuals and componentwise observable gates remain
mandatory. The direct mode skips the unused SPD/AMS hierarchy.
Factor metadata exports owner,
fill, time and one-factorization reuse. MPI failures are broadcast before
subsequent collectives so a rank-zero factor/solve error does not strand the
other ranks.

**Distributed factorization.** A build against an MPI STRUMPACK (`scripts/build_strumpack_mpi.sh`,
CMake `AXF_STRUMPACK_PREFIX`; `build_info()["strumpack_mpi"]`) factorizes the same equilibrated matrix across the
ranks with STRUMPACK's `SparseSolverMPIDist`.
- **Input:** each rank hands over its own rows of the true-DOF distribution, so the matrix values and right-hand
  sides are not gathered. The sparsity graph still goes to the root, for the METIS ordering.
- **Checks:** before the factorization the code verifies that rows and columns are partitioned in rank order and
  that the matrix is symmetric to round-off (two probes, as `SymSolver` does on the rank-zero path), since STRUMPACK
  is told the pattern is symmetric. The global nonzero count must fit in 32 bits: this STRUMPACK and METIS index
  with `int`, which caps the matrix near 5e7 order-2 DOFs.
- **Per-rank floors:** distribution balances memory; it does not reduce the total. Every rank keeps several vectors
  of the global length (the column scales during setup, STRUMPACK's permutations and equilibration), about 0.1 GB
  per rank at 3.9 M DOFs. The root also holds the gathered graph and runs METIS, and at 64 ranks it was already the
  largest rank. ParMETIS (not built) would remove that floor.
- **Out of memory:** STRUMPACK installs a new-handler that prints "STRUMPACK: out of memory!" and aborts, for as long
  as the factor exists. While a frequency is solved, any failed allocation on any rank, also outside STRUMPACK,
  therefore reports that message.
- **Solves:** both complex parts are one 2-column distributed solve.
- **Selection:** `AXF_FW_STRUMPACK_DIST`: `auto` (the default) distributes on more than one rank, `1` forces it
  (also on one rank), and `0` keeps the rank-zero factor.
- **Metadata:** `direct_factorization` reports `factor_ranks` and the ordering, in place of `factor_owner_rank`.
- **Inertia:** the self-check runs on the distributed solver too, and STRUMPACK 8.0.0 fails it there as well.
- **Agreement:** `tests/test_fullwave3d_strumpack_mpi.py` checks the distributed factor against the rank-zero one
  on the two-port frame (`benchmarks/designs/stage1_twoport.json`), at 1, 2 and 3 ranks.
  - **Same preconditioner:** the same GMRES iteration count for every port, direct-source and resistively loaded
    solve, and fill within 1 %.
  - **Inertia:** the same record.
  - **Answers:** Z and Ψ within 1e-8; measured 1e-13 at 2 ranks.
  - **Where it runs:** it needs an MPI-STRUMPACK build; a build without it skips the test.
- **Cost:** measured on the arXiv:2211.00008 toroid (3.87 M order-2 DOFs, 75.4148 MHz, 0.04 % below pole 2), in
  `benchmarks/strumpack_compression_check.py`, record rows `dist0_*` (rank zero) and `dist1_*` (distributed).
  - **Setup:** the same build, one node per run (two for the 64-rank row), one run each.
  - **Fill:** 4.66e9 to 4.72e9 factor nonzeros. METIS orders a differently distributed graph, so the fill differs
    by under 1 %.
  - **Port solves:** both complex parts of every port, 2 GMRES iterations each.
  - **Peak memory, largest rank:** from the batch scheduler's accounting, in GiB.
  - **Summed per-rank peaks:** from getrusage, in GB, an upper bound on the simultaneous total.

  | Ranks × threads | Path | Factorization | Port solves | Peak memory, largest rank | Summed per-rank peaks |
  |---|---|---|---|---|---|
  | 32 × 1 | rank zero | 481 s | 25.9 s | 46.3 GiB | 86.8 GB |
  | 32 × 1 | distributed | 131 s | 2.8 s | 4.84 GiB | 98.5 GB |
  | 8 × 4 | rank zero | 211 s | 19.4 s | 48.7 GiB | 69.4 GB |
  | 8 × 4 | distributed | 103 s | 3.2 s | 10.4 GiB | 73.0 GB |
  | 4 × 8 | rank zero | 146 s | 11.6 s | 50.0 GiB | 66.1 GB |
  | 4 × 8 | distributed | 114 s | 3.8 s | 17.4 GiB | 64.1 GB |
  | 64 × 1, 2 nodes | distributed | 100 s | 1.4 s | 3.08 GiB | 132.9 GB |

  - **Answers:** every run is accepted with a true residual of 1.2e-7, and Z agrees with the original record to
    ≤ 6.4e-9.
  - **Where the gain is:** the distributed factor is faster in every layout. Against the best threaded rank-zero
    layout (4 × 8) the factorization is 1.3× faster; against one thread per rank, 3.7×.
  - **Memory:** distribution balances memory over the ranks; the total stays about the same. A factor too large
    for one rank can be spread over nodes, up to the 32-bit limit above. Nothing larger than one node has been
    run.
  - **Default:** `auto` (distributed on several ranks) is therefore the default.

Factor metadata also carries an optional inertia diagnostic. Before any
counts are exposed, an independent connected mixed-sign matrix checks the
backend. The installed STRUMPACK 8.0.0 failed this check: its dense-front
implementation takes the absolute diagonal before testing its sign. A
successful library return code therefore cannot establish valid counts.
STRUMPACK 8.1.0 tests the sign of the pivot itself (dense and distributed
dense fronts), and the shared-memory self-check passes with it.
This build records `BACKEND_INERTIA_SELF_CHECK_FAILED`, `accurate=false`,
and null negative/zero/positive counts. Pivoted or otherwise unsupported
inertia also withholds counts. This diagnostic changes neither the field
factor nor its solver settings and supplies no pole-free certificate.

`AXF_FW_GAUGE=tree-cotree` (opt-in, STRUMPACK path only) changes
the basis of the free edge unknowns to E = S_c e + G_c phi, with one nodal
potential per PEC-connected conductor, so that K never acts on the gradient
part. The summed matrix K - w^2 M reaches small residuals at low frequency,
but its forward error grows like (k h_min)^-2 times the residual, which a
residual gate does not detect; the gauged path does not form it on gradients.
It requires lowest-order Nédélec elements and every boundary group to be PEC,
refines on the transformed residual (accumulated in extended precision by
default), zeroes the right-hand side's rounding-level charge by default, and
keeps the original-space split-form residual as the acceptance gate.
Formulation, options and the validation runs 1-6 (a capacitor's closed form,
the MQS limit of L, a topology set, scale, fail-closed checks):
[physics/fullwave_tree_cotree.md](../physics/fullwave_tree_cotree.md), section
Validation; the two-port frame cross-check against the AMS and plain
STRUMPACK paths: [tree_cotree.md](tree_cotree.md).

`AXF_FW_RTOL`, `AXF_FW_MAX_ITER` and `AXF_FW_KDIM` control the iterative
refinement. A preconditioned stopping criterion alone never accepts a
solution: the true relative residual must be below `1e-6`, and the weak
source/load observables use componentwise tolerances. Rejected fields and
diagnostics are retained explicitly as rejected; they must not supply scan
or tuning results.

## Tests and coverage

- `tests/test_fullwave3d_solver_paths.py`: the AMS and the plain (rank-zero) STRUMPACK paths against the exact TEM
  line.
- `tests/test_fullwave3d_strumpack_mpi.py`: the distributed factorization against the rank-zero one, on one and three
  ranks, and the refusal of an unknown mode.
- `tests/test_multiport_fem_spec.py`: the parser for `rf_only`, validation loads and field probes.
- The reactive-load closure and the energy identity: [energy validation](fullwave_stage2a_energy.md).

The full-versus-quarter agreement of symmetry-surface models and the removed-PEC path have no automated test or
record yet.
None of these checks qualifies the spatial resolution of a particular design, hardware losses, or detector accuracy.
