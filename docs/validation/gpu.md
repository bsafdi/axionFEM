# The GPU (CUDA) build: numerical validation and timing

Benchmark `benchmarks/gpu_validation.py`, record `benchmarks/records/gpu_validation.json` (every run with its
executable, the executable's and libaxionfem's sha256, its configuration and its host), test `tests/test_gpu.py`.
Building the CUDA stack: [install.md](../install.md), "GPU builds".

## Summary

- **What runs on the GPU:** hypre. That is AMS in the 3D MQS and full-wave H(curl) solves, AME in the cavity
  eigensolver, and BoomerAMG with hypre's PCG in the H1 solves (port feed, divergence cleaning, Gauss-law guard,
  windings) and in the 3D mechanics, conduction and electrostatics. MFEM, the assembly, the Krylov loops of the MQS
  (CG) and full-wave (GMRES) solves, STRUMPACK's factorizations and all post-processing stay on the CPU.
- **Agreement** (same design and mesh, one rank, the CUDA build on the GPU against the CPU build). The GPU model
  matters: the same executable needs different iteration counts on an A40 and an H100, and lands at a different
  distance from the CPU. Largest relative difference over the cases of the record:

  | | A40 | H100 |
  |---|---|---|
  | 3D MQS: L_p, Ψ, W | 4.4e-13 | 9.5e-13 |
  | 3D MQS: field samples | 8.0e-12 | 8.0e-12 |
  | full wave (AMS-GMRES, guard on): Z, dZ/dω, L | 3.1e-11 | 2.6e-11 |
  | full wave: energies | 6.1e-11 | 5.2e-11 |
  | full wave: Ψ vector | 1.2e-8 | 2.9e-9 |
  | full wave: field probes E / B | 7.1e-11 / 1.0e-8 | 3.1e-11 / 1.4e-9 |
  | full wave, STRUMPACK and tree-cotree paths: Z, Ψ vector, probes | 2.5e-12 | 2.3e-12 |
  | cavities, shift-invert: f, C, G, Q, peak wall fields | 2.2e-13 | 1.5e-14 |
  | cavity, AME: f / C / G and Q / peak wall fields | 1.6e-11 / 4.8e-10 / 5.4e-7 / 2.8e-5 | 1.6e-11 / 4.8e-10 / 5.4e-7 / 2.8e-5 |
  | every other output (census, key scale; the AME cavity apart) | 1.2e-8 | 1.3e-8 |

  For scale, the CPU build on 14 or 16 ranks differs from itself on one rank by up to 1.4e-8 (census, key scale).
  The largest differences are the two-port frame at ND2 (`stage1_o2`): the A40 needs 1179 GMRES iterations, the H100
  886 and one CPU core 1049, and the A40's Ψ vector and B probes are 4-7 times farther from the CPU than the H100's.
- **The order-2 full-wave error indicator** on an A40, on its complex-eps (PRESB) path (`stage1_o2_lossy`), agrees
  with the CPU to the projection's tolerance: the electric parts within 9.4e-6, the magnetic parts within 1e-11.
  `tests/test_gpu.py` checks both projection paths.
- **The CPU path is unaffected:** the CUDA build with `AXF_DEVICE=cpu` gives results bit-identical to the CPU build on
  the same node, the STRUMPACK paths included.
- **Reproducible:** a second GPU run in a new process is bit-identical to the first on the same host.
- **Speed** (engine seconds): against one CPU core the H100 is 1.9-3.8 times faster on the order-2 models (436k-1.14M
  DOFs) and the A40 1.55-2.3 times; on the small ones neither is faster. Against the CPU cores a GPU displaces (14
  per H100, 16 per A40, as MPI ranks) it is 3.1-3.5 (H100) and 3.3-4.3 (A40) times slower: only hypre is on the GPU,
  and one rank does the meshing, assembly, Krylov orthogonalization and post-processing that 14-16 ranks share.
- **Not supported:** two or more MPI ranks with hypre on the GPU (refused; see below).

## How the GPU path is set up

`$AXF_DEVICE` is `cpu` (the default), `cuda` (the first visible GPU) or `cuda:N`; only `cuda` and `cuda:0` have been
run (every validation job had one visible GPU). A GPU request moves **hypre alone** (`configure_hypre_gpu`, called
from `ensure_mpi` in `cpp/src/model3d.cpp`). It sets `HYPRE_SetMemoryLocation(DEVICE)`,
`HYPRE_SetExecutionPolicy(DEVICE)`, `HYPRE_SetUseGpuRand(0)` and `HYPRE_DeviceInitialize()` on the chosen GPU, with
`mfem::Hypre::configure_runtime_policy_from_mfem = false`; MFEM documents this configuration as "hypre uses the GPU
even if MFEM is not". MFEM's own device stays the CPU (`cpp/src/device.cpp`): at each hypre call MFEM copies the host
vectors into hypre's (unified) memory and back. `results.json` records where hypre ran, `solver.hypre_device` (for
example `"cuda:0 (NVIDIA A40)"`), also in rf-only (cavity) results. Repeated setups use hypre's host random
generator (`HYPRE_SetUseGpuRand(0)`), so repeated solves in one process are bit-identical.

**It is not the same algorithm on both devices**, so the GPU agrees with the CPU at the solver tolerance, not at
round-off, and the iteration counts differ:

- the MQS and full-wave AMS run the CPU's options on either device; hypre executes them with its GPU kernels
  (Gauss-Seidel as sparse triangular solves, its own triple products), which round differently;
- the H1 BoomerAMG solves keep MFEM's GPU defaults on a GPU (PMIS coarsening, l1-Jacobi relaxation);
- the 3D mechanics use hypre's unknown-based systems AMG on a GPU, because hypre has no GPU path for the
  rigid-body-mode interpolation of `SetElasticityOptions` (fix 5): 2-7 times the CPU's PCG iterations (below).

Other MFEM device strings (`omp`, `debug`, `ceed-cuda`, ...) are refused with a message: axionFEM's host code is not
written for MFEM device memory (fix 1).

**hypre option numbers.** The AMS options are hypre's integer codes. axionFEM's defaults on either device: smoother 2
(`AXF_AMS_RLX`, `AXF_FW_AMS_RLX`), AMG coarsening 10 = HMIS (`AXF_AMS_AMG_COARSEN`, `AXF_FW_AMS_AMG_COARSEN`), AMG
relaxation 8 = l1 hybrid symmetric Gauss-Seidel, the coarsest level included (`AXF_AMS_AMG_RELAX`,
`AXF_FW_AMS_AMG_RELAX`), aggressive-coarsening levels 0 for the MQS and 1 for the full wave (`AXF_AMS_AGG`,
`AXF_FW_AMS_AGG`). MFEM's own GPU choices, measured below, are smoother 1, coarsening 8 = PMIS and relaxation 18 =
l1-Jacobi. `AXF_AMS_CYCLE` and `AXF_AMS_SWEEPS` set the AMS cycle type (default 13) and smoother sweeps (default 1).
All of these must be integers in int's range: a 3D run checks them when it starts, before meshing, and stops with a
message naming the variable; an empty value means unset. The options in force are recorded in `solver.options` (MQS)
and `fullwave.solver_settings.ams_options`.

With MFEM's GPU options the full-wave AMS needs 2-3 times the GMRES iterations, whose orthogonalization runs on the
host: the two-port frame at ND2 took 1347 s instead of 408 s on an H100, and its true residuals were larger
(1.2-5.1e-8 against 2.2-3.0e-9). That is why the full-wave AMS runs the host options on either device
(`fw_ams_overrides`, `cpp/src/model3d_fullwave.cpp`).

**Several MPI ranks with hypre on the GPU are refused.** Two ranks sharing an A40, and 14 ranks sharing an H100,
crashed or hung in the first MQS solve: `cudaFree` failing with error 717 ("operation not supported on global/shared
address space") in MFEM's `CuMemFree`, `MPI_ERR_RANK: invalid rank` in an `MPI_Irecv` of a halo exchange, or a hang.
With one rank there are no off-diagonal (halo) blocks, which is where MFEM's host memory meets hypre's device memory in
the multi-rank paths. The cause is not pinned down, and one GPU per rank was not tried. `configure_hypre_gpu` refuses
two or more ranks with a message saying so.

**Build notes:**

- hypre 2.33.0 cannot build several CUDA architectures through CMake: it joins `CMAKE_CUDA_ARCHITECTURES` with commas
  and puts that on its target, which nvcc rejects (`'80' is not in 'keyword=value' format`).
  `scripts/build_deps_gpu.sh` patches that one line, and stops if the patch did not apply. The default architectures
  are `75;80;86;90`.
- STRUMPACK (host only, as in the CPU stack) is part of the CUDA stack.
- nvcc needs the system's glibc headers; on GPU nodes without them, build on a CPU node and run on the GPU node.

## GPU-specific handling in the code (fixes 1-7)

hypre and MFEM behave differently when hypre runs on a GPU. Each difference below is handled in axionFEM; the library
behaviours behind them are documented, not patched. Code comments refer to these items by number.

1. **MFEM stays on the host.** With `mfem::Device("cuda")` MFEM keeps the valid copy of every vector it creates (grid
   functions, linear forms, hypre vectors) on the GPU, while axionFEM reads and writes those vectors element by element
   on the host (`v(i)`, `GetData()`). On an A40 that made the divergence cleaning a no-op, left the readout port's
   conduction solve without current, and changed the Newton steps of a saturated iron core. MFEM therefore always runs
   on the host and only hypre goes to the GPU. axionFEM assembles with MFEM's full (legacy) assembly, which has no GPU
   path anyway.
2. **Eliminated rows keep a diagonal.** On a GPU, `mfem::RAP` (`hypre_ParCSRMatMat` + `hypre_ParCSRTMatMatKT`) drops
   rows whose entries are all zero from the sparsity pattern (MFEM flags this in a FIXME in `linalg/hypre.cpp`). The
   H1 problems posed on part of the mesh (the port conduction solve, a poloidal winding's current) have a zero
   coefficient outside it and eliminate every DOF there. On the host those rows keep their (zero) diagonal, which the
   elimination sets to 1; on the GPU they end up empty and BoomerAMG's setup fails (hypre error 13). Reproduced with
   MFEM and hypre alone (a 16³ cube of tetrahedra, the coefficient zero outside x < 1/4): 3468 of 4913 rows empty on
   the GPU, none on the host. `keep_eliminated_rows` (`cpp/src/model3d_impl.hpp`) adds a unit diagonal to the rows of
   the eliminated DOFs before the parallel assembly. It is exact: the elimination overwrites those rows and columns.
3. **An empty imaginary block.** A lossless full-wave model's imaginary block is empty on the GPU for the same reason,
   and MFEM's elimination in `ParSesquilinearForm::FormLinearSystem` would dereference its null data array.
   `keep_eliminated_rows` is applied to both parts of the sesquilinear form, and `ensure_diagonal` precedes every
   other elimination with a unit diagonal of an assembled 3D matrix (cleaning, AMS operator, guard, cavity operator
   and wall mass, electrostatics, heat): it adds a unit diagonal to eliminated rows that have none, and is a no-op when
   every row has one.
4. **Diagonals are found by column.** `HypreParMatrix::GetDiag` takes each row's *first stored entry* as the diagonal
   (MFEM 4.10 `linalg/hypre.cpp`, checked only in debug builds) and, when hypre runs on a GPU, writes the device copy
   of its output. For a row that hypre left empty it returns the next row's diagonal. `stored_diagonal`
   (`cpp/src/model3d_impl.hpp`) finds each row's diagonal by its column on the host, returns 0 for a row without one,
   and replaces every `GetDiag` of the 3D engine (the guard's Jacobi weights, the cavity's wall DOFs, the heat balance,
   the boundary nodes of the wall E_n flux recovery). Where the diagonal is stored first, as on the host, it gives the
   same values as `GetDiag`.
5. **Mechanics AMG.** `HypreBoomerAMG::SetElasticityOptions` (rigid-body-mode interpolation) has no GPU path in
   hypre, and MFEM aborts. On a GPU the 3D mechanics use the systems AMG alone (`SetSystemsOptions`), and
   `mechanics.solver.amg` says so: `"systems (hypre on a GPU: no rigid-body-mode interpolation)"`.
6. **Cavity wall losses.** The wall-current solve takes the wall DOFs to be the rows of the wall mass matrix with a
   positive diagonal. Read with `GetDiag` on a GPU, almost every DOF was taken for a wall DOF and the losses were
   silently absent from `results.json`. The wall DOFs come from `stored_diagonal` (item 4); the wall-current solve
   records its PCG iterations and true relative residual (`losses.wall_current_solve`); and a solve with a residual
   above 1e-6, or with no finite G, is a warning on every device ("cavity: the wall-current solve failed for mode(s)
   ..."). On the three cavities of the record the residual is 5e-14 to 9e-14 (43-58 iterations) on both devices.
7. **The order-2 error indicator's projection.** At order 2 and above the full-wave error indicator solves an
   H1_{p+1} gradient projection whose weight vanishes off the field elements (conductors, the infinity shell, PML
   layers), with every DOF there eliminated. `cpp/src/model3d_fw_indicator.cpp` applies `ensure_diagonal` between
   `ParallelAssemble` and `EliminateRowsCols` on every operator of the projection (`|eps|`, and on the PRESB path the
   real and imaginary parts and their sum), and the PRESB path's Jacobi scaling reads `stored_diagonal` of the sum.
   On the CPU both are no-ops. Without them, on an A40, the `|eps|` path's PCG reported convergence after 0 iterations
   and the indicator's electric part came out tens of times too small without a warning, and the PRESB path stopped in
   BoomerAMG's setup (hypre error 12). `tests/test_gpu.py` catches both.

**Other host reads of hypre's matrices:**

- the two `EliminateBC` overloads read the diagonal differently: the two-argument form (guard, cavity operator, wall
  mass) finds it by its column, so it needs a stored diagonal; the four-argument form (MQS, cleaning, electrostatics,
  heat, mechanics) sets `B(r)` from the row's first stored entry, so it needs the diagonal stored first. hypre's device
  triple product moves every existing diagonal first, the host one reserves it first, and `ensure_diagonal` reorders
  after it adds one;
- the full-wave AMS preconditioner's `pc.FormSystemMatrix(ess, PCop)` needs no helper: its mass term keeps every row
  non-empty;
- `MergeDiagAndOffd` (the STRUMPACK paths) reads on the host after its own `HostRead`;
- host reads and writes of hypre's arrays (`stored_diagonal`, `ensure_diagonal`, the tree-cotree path's copy of the
  discrete gradient) first wait for the device (`sync_hypre_device`, a `cudaDeviceSynchronize` when hypre runs on a
  GPU): MFEM's `HostRead` only sets flags for hypre's unified memory, and hypre's kernels run asynchronously;
- `stored_diagonal` refuses a matrix that is not square with matching row and column partitions;
- `HypreDiagScale` (the wall-current PCG) divides by each row's first entry and is applied after `ensure_diagonal`.

## Agreement

Same design, same mesh (byte-identical `mesh3d.msh` from one mesher; `report` checks it), one rank, against the CPU
build on one rank. The small cases' reference is the CPU build on one core; the order-2 cases' reference is the CPU
build on one core of an A40 node (`cpu_a40`).

Relative differences are max|a − b| / max|b| per quantity, with complex matrices, vectors and sample sets taken as a
whole. "All outputs" is a census of every numeric leaf of `results.json` outside the timings, each difference taken
relative to the largest magnitude under the same key. The last column is the CPU build on 14 ranks against 1 rank,
for scale: a different partition changes the answers about as much. "Two-port frame" is the record's `stage1` case
(`benchmarks/designs/stage1_twoport.json`).

| case (DOFs) | GPU | L_p, Ψ, W | MQS samples | Z, dZ/dω, L | FW Ψ vector | FW energies | FW probes E / B | cavity f / C / G, Q / peak fields | all outputs | CPU 14 ranks, all outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| solenoid (81k) | A40 | 4.3e-14 | 2.3e-12 |  |  |  |  |  | 2.7e-11 | 1.4e-10 |
| solenoid (81k) | H100 | 4.3e-14 | 2.3e-12 |  |  |  |  |  | 2.7e-11 | 1.4e-10 |
| two-port frame (109k) | A40 | 3.0e-14 | 8.0e-12 | 3.1e-12 | 1.8e-09 | 5.7e-12 | 3.8e-12 / 6.2e-10 |  | 2.5e-09 | 1.6e-09 |
| two-port frame (109k) | H100 | 3.0e-14 | 8.0e-12 | 3.1e-12 | 1.8e-09 | 5.7e-12 | 3.8e-12 / 6.2e-10 |  | 2.5e-09 | 1.6e-09 |
| iron, Newton (26k) | A40 | 1.3e-15 | 1.6e-15 |  |  |  |  |  | 1.1e-12 |  |
| iron, Newton (26k) | H100 | 1.3e-15 | 1.6e-15 |  |  |  |  |  | 1.1e-12 |  |
| pillbox, AME (26k) | A40 |  |  |  |  |  |  | 1.6e-11 / 4.8e-10 / 5.4e-07 / 2.8e-05 | 9.6e-05 |  |
| pillbox, AME (26k) | H100 |  |  |  |  |  |  | 1.6e-11 / 4.8e-10 / 5.4e-07 / 2.8e-05 | 9.6e-05 |  |
| pillbox, shift-invert (26k) | A40 |  |  |  |  |  |  | 1.0e-15 / 1.8e-15 / 2.1e-15 / 3.9e-14 | 2.1e-13 |  |
| pillbox, shift-invert (26k) | H100 |  |  |  |  |  |  | 2.1e-16 / 4.8e-16 / 4.2e-16 / 1.5e-14 | 1.9e-13 |  |
| pillbox with rod, shift-invert (42k) | A40 |  |  |  |  |  |  | 9.2e-14 / 4.9e-14 / 2.2e-13 / 6.2e-14 | 2.2e-12 |  |
| pillbox with rod, shift-invert (42k) | H100 |  |  |  |  |  |  | 0 / 1.4e-15 / 7.6e-15 / 2.7e-15 | 1.5e-13 |  |
| two-port frame, STRUMPACK, 1 MHz (109k) | A40 | 3.0e-14 | 8.0e-12 | 7.3e-14 | 2.5e-12 | 1.5e-13 | 7.2e-14 / 1.1e-13 |  | 1.0e-11 |  |
| two-port frame, STRUMPACK, 1 MHz (109k) | H100 | 3.0e-14 | 8.0e-12 | 8.6e-14 | 2.3e-12 | 1.7e-13 | 8.4e-14 / 1.2e-13 |  | 1.0e-11 |  |
| two-port frame, tree-cotree, 0.1 MHz (109k) | A40 | 3.0e-14 | 8.0e-12 | 4.7e-16 | 1.6e-13 | 5.9e-14 | 4.9e-16 / 2.3e-14 |  | 1.0e-11 |  |
| two-port frame, tree-cotree, 0.1 MHz (109k) | H100 | 3.0e-14 | 8.0e-12 | 4.7e-16 | 1.7e-13 | 7.2e-14 | 6.2e-16 / 2.3e-14 |  | 1.0e-11 |  |
| solenoid ND2 (436k) | A40 | 4.3e-13 | 4.0e-12 |  |  |  |  |  | 3.3e-09 | 3.0e-09 |
| solenoid ND2 (436k) | H100 | 2.2e-13 | 3.3e-12 |  |  |  |  |  | 3.4e-09 | 3.0e-09 |
| two-port frame ND2 (572k) | A40 | 3.5e-13 | 4.4e-12 | 3.1e-11 | 1.2e-08 | 6.1e-11 | 7.1e-11 / 1.0e-08 |  | 1.0e-08 | 2.0e-09 |
| two-port frame ND2 (572k) | H100 | 9.5e-13 | 4.0e-12 | 2.6e-11 | 2.9e-09 | 5.2e-11 | 3.1e-11 / 1.4e-09 |  | 3.5e-09 | 2.0e-09 |
| solenoid ND2 finer (1.14M) | A40 | 4.4e-13 | 4.6e-12 |  |  |  |  |  | 1.2e-08 | 1.4e-08 |
| solenoid ND2 finer (1.14M) | H100 | 4.5e-13 | 4.8e-12 |  |  |  |  |  | 1.3e-08 | 1.4e-08 |

- **Counted apart in the census:** solver diagnostics, which differ between preconditioners by construction
  (iteration counts, residuals and backward errors, preconditioned norms, and the consistency checks that are zero or
  one in exact arithmetic and carry the solver's noise; `DIAGNOSTIC` in `benchmarks/gpu_validation.py`), and entries
  at round-off that are zero in exact arithmetic, such as a lossless model's resistance (compared relative to |Z|).
- **The AME cavity** agrees to the eigensolver's tolerance, not further: AME stops at a mode residual of about 1e-5 on
  either device, so f (second order in the eigenvector's error) agrees to 1.6e-11, C to 4.8e-10, G and Q to 5.4e-7,
  and the pointwise peak wall fields (first order) to 2.8e-5. The shift-invert cavities converge to round-off and
  agree to 1e-14.
- **Hardware:** the A40 and the H100 give bit-identical results on the three small MQS and full-wave cases. On the
  cavities and the STRUMPACK paths they differ at 1e-13, where host-side dense or direct algebra enters and the hosts'
  OpenBLAS kernel families differ (Open issues). At ND2 they differ: other iteration counts, and for the two-port
  frame a Ψ vector and B probes 4-7 times farther from the CPU on the A40.

**Iterations** (per solve; the MQS has the static and the pickup solve; the frame's static solve is trivial):

| case | CPU 1 core: MQS PCG | GMRES per solve (total) | guard PCG | A40: MQS PCG | GMRES (total) | guard | H100: MQS PCG | GMRES (total) | guard | CPU 14 ranks: MQS PCG | GMRES (total) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solenoid (81k) | 35/36 |  |  | 35/36 |  |  | 35/36 |  |  | 37/38 |  |
| two-port frame (109k) | 79 | 36-50 (633) | 10-11 | 78 | 33-44 (564) | 18-20 | 78 | 33-44 (564) | 18-20 | 80 | 46-62 (800) |
| iron, Newton (26k) | 35/34/33/33/31 |  |  | 35/34/33/33/31 |  |  | 35/34/33/33/31 |  |  |  |  |
| two-port frame, STRUMPACK, 1 MHz (109k) | 79 | 2-4 (10) | 10-11 | 78 | 2-4 (10) | 19-20 | 78 | 2-4 (10) | 18-20 |  |  |
| solenoid ND2 (436k) | 67/119 |  |  | 65/67 |  |  | 65/119 |  |  | 71/131 |  |
| two-port frame ND2 (572k) | 157 | 81-106 (1049) | 21-22 | 155 | 78-150 (1179) | 34-35 | 155 | 69-88 (886) | 34-35 | 170 | 93-123 (1207) |
| solenoid ND2 finer (1.14M) | 109/201 |  |  | 109/115 |  |  | 109/115 |  |  | 110/113 |  |

- The MQS AMS-PCG counts are those of the CPU on the small cases. At ND2 they depend on the device and on the CPU
  partition.
- The full-wave GMRES counts depend on the GPU model at ND2. The CPU spreads as much across partitions (1049, 1207 and
  1234 GMRES iterations on 1, 14 and 16 ranks).
- The guard's PCG keeps MFEM's GPU BoomerAMG options: 18-20 against 10-11 iterations on the frame, all converged.

**3D mechanics, conduction and electrostatics on a GPU** (BoomerAMG with hypre's PCG; the 3D tests on an H100 and on
the CPU, the same meshes):

| test (free DOFs) | PCG iterations, H100 / CPU | AMG on the H100 / CPU | GPU against CPU |
|---|---:|---|---|
| thin ring in an external field (20,427) | 4284 / 1535 | systems / elasticity | u 6.1e-11, σ 3.3e-13 |
| free cooldown (2103) | 226 / 32 | systems / elasticity | u 1.0e-12 (σ is round-off) |
| partial support (1369) | 98 / 33 | systems / elasticity | u 7.2e-15, σ 6.6e-15 |
| hydrostatic sector (3069) | 153 / 36 | systems / systems (rotated basis) | u 1.5e-13, σ 3.3e-13 |
| Lamé sector (4705) | 142 / 71 | systems / systems (rotated basis) | u 1.6e-15, σ 1.5e-14 |
| conduction slab, quadratic T; k(T) | 63 / 47; 35 / 26 | scalar | T 5.6e-15; 7.3e-15 |
| isolated sphere, 3D electrostatics (20,836) | 604 / 218 (each of its two designs) | scalar | C 1.5e-15 |
| dielectric region (33,246) | 612 / 220 | scalar | C 9.5e-16 |

- The converged results agree to 1e-12 to 6e-11 in the displacements and to round-off elsewhere.
- The systems AMG without rigid-body modes is weaker: 3-7 times the PCG iterations where the CPU runs the elasticity
  AMG, and it degrades on slender, bending-dominated parts (the thin ring needs 4284 iterations at 20k unknowns).
- MFEM's GPU BoomerAMG defaults (PMIS, l1-Jacobi) cost 2-4 times on top of that where both devices run the systems
  AMG (the sectors), and 1.3 times (conduction) to 2.8 times (electrostatics) in the scalar solves.
- The PCG stops at 20,000 iterations with a warning (not an error) above a 1e-8 residual, so a large thin structure on
  a GPU can end at the cap (Open issues).

**The order-2 indicator's complex-eps (PRESB) path on an A40** (`stage1_o2_lossy`: the frame at ND2 with a weakly
conducting block, 0.0167 S/m, in the loop's opening; 575,806 DOFs; 1 and 12 MHz). The block's loss phase differs from
the vacuum's, so the projection takes GMRES with PRESB, with BoomerAMG of A_re + A_im on the GPU.

- The solves agree as on the lossless frame: Z, dZ/dω and L within 3.6e-11, the energies 1.4e-11, the field probes
  E / B 3.7e-11 / 9.1e-10, the Ψ vector 3.5e-8, L_p 3.6e-13.
- The indicator: the electric parts within 5.9e-6 (1 MHz) and 9.4e-6 (12 MHz), the totals 5.1e-9 and 1.0e-6, the
  magnetic parts 7.9e-12 and 1.0e-11. The electric part differs in its sixth digit because the projection stops at a
  relative residual of 1e-4 on both devices and GMRES on the complex-symmetric system is not a Galerkin method, so the
  parts move to first order in the iterate's error. At 1e-10 (`AXF_FW_INDICATOR_TOL`) the two devices agree to
  1.7e-10 on either path (Tests).
- The census's largest differences are in the imaginary parts of the energy matrices, which are round-off here
  (1e-25 to 9e-16 against real parts of 1.3e-7).

## Tests on the GPU

- **ctest of the CUDA build:** 11/11, on the CPU and with `AXF_DEVICE=cuda`, on an H100 and an A40. The C++ unit
  tests do not reach hypre's device path; they show that the build runs there.
- **`tests/test_gpu.py`:** the CUDA build on the GPU against itself on the CPU, asserting where hypre ran each time:
  the 3D MQS solenoid (L_p, Ψ, W to 1e-8); the two-port frame at 0.1 and 12 MHz with the guard on (Z, dZ/dω, Ψ vector
  to 1e-7); the AME pillbox (f, C, G, Q0, Q_wall, the wall's G, the peak wall field, with tolerances set from the AME
  eigenvector's accuracy; no wall-current warning); the free cooldown (the mechanics' AMG label on each device,
  displacements to 1e-7).
- **The order-2 indicator's test** (`test_order2_indicator_on_the_gpu_matches_the_cpu`, two parametrizations): a
  shorted parallel-plate line with a perfect-conductor ridge (outside the field, so rows of every projection operator
  are empty there), 21,424 DOFs at ND2, 0.3 and 1 GHz. `complex_eps_presb` fills the line's upper half with
  0.0167 S/m (GMRES with PRESB); `abs_eps_pcg` is the lossless line (hypre's PCG). Both devices solve the projection
  to 1e-10 (`AXF_FW_INDICATOR_TOL`; default 1e-4) with `OPENBLAS_CORETYPE=Haswell`. It asserts that the projection
  took its path, iterated (iterations > 0, residual > 0) and converged on both devices, that the CPU run reproduces the
  stored CPU numbers to 1e-12, and that the GPU's total, magnetic and electric parts agree with the CPU's to 1e-7. A
  build without fix 7 fails both parametrizations: the PRESB path stops in BoomerAMG's setup (hypre error 12), and the
  PCG path reports 0 iterations.
- The test has passed on A40, H100, A100 and RTX 8000 GPUs. It skips with `CUDA_VISIBLE_DEVICES=` (it asks the CUDA
  driver), without a GPU, with a CPU build, and with a CUDA package whose `$AXF_EXE` is a CPU build; with no executable
  found it says so.
- **The 3D test files with `AXF_DEVICE=cuda`** (H100: `test_3d.py`, `test_fullwave3d_gauss_guard.py`,
  `test_background_3d.py`, `test_fullwave3d_error_indicator.py`, `test_mechanics3d.py`, `test_cavity3d.py`,
  `test_electrostatics3d.py`, `test_mpi.py`): 45 passed, 45 skipped, 0 failed. Among them
  `test_indicator_on_and_off_give_identical_results`, which needs two runs in one process to be bit-identical.
  `test_mpi.py` skips without `AXF_MPI` (a GPU run is one rank), and the cavity solves are `AXF_SLOW`; the record's
  cavity cases and `test_gpu.py` cover them.

## Timing

One GPU against the CPU cores it displaces, on the same nodes: H100 nodes with 112 Sapphire Rapids cores (Xeon
Platinum 8480+) for 8 GPUs, so 14 cores per GPU; A40 nodes with 64 EPYC 7713 cores for 4 GPUs, so 16 cores per GPU.
The CPU runs use those cores as MPI ranks; a GPU run is one rank with the GPU. One CPU core is shown for scale: an
EPYC 7713 core of an A40 node for the order-2 cases, an EPYC 9534 core for the small ones.

Seconds of the engine's own clocks (meshing, MQS, full wave; `compute_seconds` in the record), with the iterations
next to them:

| case (DOFs) | CPU 1 core | CPU 14 cores (H100 node) | CPU 16 cores (A40 node) | H100 + 1 core | A40 + 1 core |
|---|---|---|---|---|---|
| solenoid, MQS (81k) | 12.1 (71) | 4.3 (75) | 5.6 (76) | 16.5 (71) | 27 (71) |
| two-port frame, MQS + 4 frequencies (109k) | 75 (633) | 15.7 (800) | 14.4 (786) | 82 (564) | 118 (564) |
| iron, Newton (26k) | 7.3 (166) |  |  | 14.4 (166) | 20 (166) |
| solenoid ND2, MQS (436k) | 212 (186) | 32 (202) | 37 (199) | 112 (184) | 136 (132) |
| two-port frame ND2, MQS + 3 frequencies (572k) | 2013 (1049) | 168 (1207) | 235 (1234) | 528 (886) | 1223 (1179) |
| solenoid ND2 finer, MQS (1.14M) | 839 (310) | 89 (223) | 112 (224) | 280 (224) | 364 (224) |

**Where the time goes, the MQS AMS-PCG solves alone:**

| case | CPU 1 core | CPU 14 cores | CPU 16 cores | H100 | A40 | A40, MFEM's GPU AMS options |
|---|---|---|---|---|---|---|
| solenoid ND2, MQS (436k) | 118.1 s, 186 its, 635 ms/it | 9.2 s, 202 its, 46 ms/it | 12.4 s, 199 its, 62 ms/it | 30.0 s, 184 its, 163 ms/it | 32.6 s, 132 its, 247 ms/it | 8.8 s, 255 its, 35 ms/it |
| two-port frame ND2, MQS + 3 frequencies (572k) | 141.4 s, 157 its, 900 ms/it | 10.2 s, 170 its, 60 ms/it | 13.6 s, 168 its, 81 ms/it | 25.0 s, 155 its, 162 ms/it | 50.0 s, 155 its, 323 ms/it |  |
| solenoid ND2 finer, MQS (1.14M) | 577.6 s, 310 its, 1863 ms/it | 25.7 s, 223 its, 115 ms/it | 39.7 s, 224 its, 177 ms/it | 61.4 s, 224 its, 274 ms/it | 92.6 s, 224 its, 413 ms/it | 39.6 s, 552 its, 72 ms/it |

- **Against its node share of cores** the H100 is 3.1-3.5 times slower (14 cores) and the A40 3.3-4.3 times (16
  cores; the 1223 s of the frame at ND2 is from a node that ran every part of the run 5-33 % slower than another A40
  node with the same iterations, where it took 1017 s). Only hypre runs on the GPU; everything else runs on one core,
  where the CPU run spreads it over 14-16 ranks: the assembly and elimination (26-78 s against 2-6 s), the meshing
  (12-40 s, serial in every configuration), the full-wave Krylov loop and the post-processing.
- **The AMS-PCG solves alone,** per iteration: the H100 is 3.9-6.8 times faster than one core, the A40 2.6-4.5 times,
  and both are 2.3-4.0 times slower than the node share. The solve times alone mislead where the iteration counts
  differ.
- **With MFEM's GPU options** (`AXF_AMS_RLX=1 AXF_AMS_AMG_COARSEN=8 AXF_AMS_AMG_RELAX=18`) the A40 matches 16 cores at
  about twice the iterations. They are not the default: the AMS robustness study
  ([ams_robustness.md](ams_robustness.md)) was done with Gauss-Seidel.
- **The GPU is mostly idle:** its mean utilization over a run is 18-23 % on the order-2 models and 46-48 % on the
  small frame (`nvidia-smi` twice a second); its peak memory is 0.4-8.1 GiB.
- **Full wave** (the frame at ND2, three frequencies): H100 390 s and 886 GMRES iterations (0.44 s per iteration,
  counting each frequency's assembly and AMS setup and the port and validation solves); A40 1024 s and 1179
  iterations (0.87 s; 837 s and 0.71 s on the quieter node); CPU 124 s and 1207 iterations on 14 cores (0.10 s), 1730 s
  and 1049 iterations on one core (1.65 s). GMRES orthogonalizes its Krylov basis on the host, and MFEM copies every
  vector into hypre's memory and back at each AMS application.

## Reproduce

```bash
# the CUDA stack (hypre, METIS, MFEM with CUDA; STRUMPACK) and axionFEM against it, with its own Python module
# directory (docs/install.md, "GPU builds")
scripts/build_deps_gpu.sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXF_DEPS=$PWD/third_party/install-cuda \
      -DAXF_GMSH_ROOT=$PWD/third_party/install/gmsh-sdk -DAXF_GPU=CUDA \
      -DAXF_INPLACE_PYTHON_DIR=$PWD/build-cuda-py/axionfem
cmake --build build-cuda -j16
cp -rn python/axionfem/. build-cuda-py/axionfem/    # the package sources next to the CUDA _core module

# one configuration per build and device (see the docstring of benchmarks/gpu_validation.py), then the report
python benchmarks/gpu_validation.py run --name cpu --exe build/axf --deps third_party/install
python benchmarks/gpu_validation.py run --name cuda_device_a40 --exe build-cuda/axf \
       --deps third_party/install-cuda --device cuda
python benchmarks/gpu_validation.py report
```

Every solve and sample pins `OPENBLAS_CORETYPE=Haswell` (`AXF_GPUVAL_OPENBLAS_CORETYPE` overrides) and records it in
`run.json`; `report` flags a run whose pin differs from its reference's (`openblas_coretype_mismatch`). A pinned
rerun therefore needs its reference rerun with the same pin.

## Open issues

- **More than one MPI rank with hypre on the GPU** crashes or hangs, and is refused. The likely place is MFEM's
  multi-rank `HypreParMatrix` paths (halo blocks, communication packages) when MFEM's memory is on the host and
  hypre's on the device; not root-caused. One GPU per rank was not tried.
- **Speed.** To beat the CPU cores it displaces, more than hypre has to move: the GMRES and CG vectors resident on the
  GPU (MFEM's device mode, which needs axionFEM's host loops made device-aware, the reason for fix 1), device (partial)
  assembly, and several ranks per GPU job.
- **MFEM's GPU AMS options for the MQS** make the solve 3.7 times faster on the GPU (436k DOFs). They need the AMS
  robustness sweep (`benchmarks/ams_robustness.py`: slits, ND3, iron) repeated before they can be the GPU default.
- **The BoomerAMG options on a GPU** (the guard, the H1 solves, the mechanics, conduction and electrostatics) are
  MFEM's GPU defaults (PMIS, l1-Jacobi): converged, but up to 2-4 times the CPU's iterations, and the mechanics'
  systems AMG up to 7 times. HMIS with hybrid Gauss-Seidel there, as for AMS, and a size-dependent warning for the
  mechanics are not implemented.
- **Not run on a GPU:** the `AXF_SLOW` tests; `cuda:N` with N > 0; HIP; V100s (compute capability 7.0, not in the
  default architectures). The order-2 error indicator has run on an A40 only.
- **Host BLAS kernels.** OpenBLAS (`DYNAMIC_ARCH`) picks its kernels at run time (SkylakeX on AVX-512 hosts, Zen on
  AVX2 hosts), and MFEM (`MFEM_USE_LAPACK`) builds the tetrahedral ND and H1 bases through LAPACK, so the bases differ
  in their last bits between kernel families. That is why results can differ at round-off between hosts, and why the
  GPU validation pins `OPENBLAS_CORETYPE=Haswell` (Zen's bits, which run on every node). The pin is not set in
  `scripts/env.sh`, because it costs BLAS speed on AVX-512 hosts. The order-2 error indicator poses its residual
  problems on the canonical complement ([fullwave_error_indicator.md](../physics/fullwave_error_indicator.md)),
  which keeps its magnetic part independent of the kernel family; on the PRESB path the electric part still differs by
  about 6e-6 between kernel families (the CPU BoomerAMG's coarsening sees matrices that differ in their last bits),
  within the 1e-4 tolerance.
- **Library behaviours documented, not patched:** hypre 2.33.0's multi-architecture CMake (patched in the build
  script); `mfem::RAP` dropping all-zero rows on a GPU (MFEM's FIXME); `HypreParMatrix::GetDiag` taking the first
  stored entry and writing the device copy; `SetElasticityOptions` refused by hypre on a GPU.
