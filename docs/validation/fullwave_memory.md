# Memory across long 3D full-wave sweeps

Driver `benchmarks/fullwave_memory.py`, record `benchmarks/records/fullwave_memory.json` (every run's environment,
build, layout, per-frequency memory series and metrics, and the bit-identity comparisons). Test:
`tests/test_fullwave3d_memory.py` (AXF_SLOW). The environment variables are listed in
[install.md](../install.md).

## Summary

**Question.** On the direct solver paths, rank 0's peak memory grew by about 1 GB per frequency over a sweep (the
tree-cotree scale run, `benchmarks/records/tree_cotree_scale.json`). Adaptive sweeps solve many more frequencies per
job. Is it a leak, data kept per frequency by design, allocator fragmentation, or a library?

**Answer.** Not a leak, and nothing the sweep keeps grows with the frequency count: glibc's own account of the heap the
program holds is the same at the start of every frequency, on every solver path, to 10 MB in 20 frequencies (the sweep
rows, about 0.2 MB per frequency summed over 8 ranks), and valgrind massif agrees. The growth is glibc's malloc keeping a
freed factor's memory resident:

1. **in its free bins**, which `malloc_trim` returns;
2. **in the top chunks of the OpenMP threads' arenas**, which `malloc_trim` does not return (glibc 2.28 shrinks only
   the main arena's top there). A thread arena's top is returned only by `free()`, when it exceeds the trim threshold,
   which glibc raises up to 64 MiB as large blocks are freed. STRUMPACK's OpenMP tasks allocate the fronts in their
   threads' arenas, so the more threads per rank, the more stays resident;
3. **fragmentation across those arenas**: which thread allocates which front changes from one factorization to the
   next, so the next factor cannot reuse all of the last one's freed memory.

## Allocator settings

Both are defaults and both can be switched off:

- `AXF_FW_MALLOC_TRIM=factor` (`factor`, `frequency` or `off`; anything else is refused): `malloc_trim(0)` after every
  direct factorization (its work memory) and whenever a direct factor is released. `frequency` also trims at the start
  of every frequency and records the resident set before and after (`frequency_start_after_malloc_trim`).
- `AXF_FW_MALLOC_THRESHOLDS=fixed`: from the first direct factorization of a full-wave analysis (a sweep or
  shift-invert eigenmodes) to its end, glibc's trim threshold is 1 MiB and its mmap threshold 32 MiB (`mallopt`, which
  also stops their dynamic adjustment), so that `free()` shrinks every arena's top. The thresholds are set on the ranks
  that factorize (rank 0 alone on the gauged path), from inside the factor objects (`FwMallocScope` in
  `cpp/src/model3d_fullwave.cpp`).
- When the analysis ends, also by an exception, the process gets trim 64 MiB and mmap 32 MiB, glibc's own dynamic
  maxima and its steady state after a large free, and the analysis's freed memory is trimmed. `mallopt` has no getter
  and cannot switch the dynamic adjustment back on, so this is the closest to a restore there is. A Python session that
  solves in process is therefore not left with a 1 MiB trim threshold: with it, numpy temporaries of 2-24 MiB ran
  2.5-9.4× slower and a scipy matvec 1.8× slower; after a scoped analysis they run as under glibc's defaults.
- A job that sets glibc's thresholds in its environment (`MALLOC_TRIM_THRESHOLD_`, `MALLOC_MMAP_THRESHOLD_`,
  `glibc.malloc` tunables) keeps them.

## The memory record

Every frequency row of a 3D full-wave sweep carries `memory` (`fw_memory_record` in `cpp/src/model3d_fullwave.cpp`),
taken at the start of the frequency (what the setup and the earlier frequencies left) and at its end;
`fullwave.memory` holds the allocator settings and, after the loop, `after_sweep`. A frequency refused by a validation
gate keeps its record in `fullwave_rejected.json`. Each figure is given as the maximum over the ranks, their sum and
rank 0's (the owner of a gathered factor):

- `rss`, `peak_rss`: VmRSS and VmHWM (Linux `/proc/self/status`); `rss_anon`, `rss_shmem`: RssAnon and RssShmem;
- `heap_held`, `heap_free`: glibc's own account (`malloc_info` totals, 64-bit): the bytes the program holds and the
  free bytes glibc keeps in its arenas (bins and tops, resident or not). A leak or growing retained data shows in
  `heap_held`; allocator retention in `rss` − `heap_held`, which also holds what is resident outside malloc (program
  text, MPI's shared memory, stacks, BLAS buffers: roughly constant after the first frequency);
- `after_factorization_rank0`: rank 0's VmRSS, heap held and heap free right after the frequency's direct
  factorization and its trim. What is resident there and not held is the factorization's work memory left over.

`fullwave.memory` also records, for every full-wave analysis (eigen-only included), what was in force:
`malloc_trim` and `malloc_thresholds_mode` (the settings), `malloc_thresholds` (set by this analysis, or "not set by
this analysis" with what was), `malloc_thresholds_applied_ranks` and `malloc_thresholds_before`.

All memory figures sit under the one `memory` key, so that a bit-for-bit comparison of rows drops them with one
filter; they differ from run to run. The `peak_rss_*_rank_bytes` keys report VmHWM, not getrusage's `ru_maxrss`
(which also carries the peak of the image that exec'd the process).

`AXF_FW_FREQUENCY_PEAK=1` resets the kernel's peak counter at the start of every frequency (`/proc/self/clear_refs`),
so that each row's `peak_rss` is that frequency's own peak and not the running maximum; a failed reset is recorded in
`peak_scope`.

## Measurements

**Model.** The fiducial toroid of Benabou et al. (2022) with a 2-degree slit port, lowest order, closed PEC sphere, on
two meshes: `mid` (444,214 tetrahedra, 527,515 unknowns) and `run5` (1,178,849 tetrahedra, 1,387,041 unknowns); 20
frequencies log-spaced over 1-100 kHz (60 for the longer sweeps). Paths (`PATHS` in the driver): `ams` (AMS-GMRES with
the Gauss-law guard), `strumpack` (the factor of K − ω²M gathered to rank 0; 4.0 GB on `mid`), `strumpack_dist` (the
same factor distributed over the ranks) and `tree_cotree` (the gauged path, factor on rank 0; 1.5 GB on `mid`).

**Without the settings** (glibc's behaviour, 8 ranks × 8 threads):

- AMS is flat to a few MB per rank: it has every per-frequency feature of the direct paths (the Gauss-law guard, the
  error indicator, the exact dZ/dω, the sweep rows) and no factor.
- The heap held is flat on every path: no leak in axionFEM, STRUMPACK (sequential or distributed), MFEM or hypre.
- On the `run5` mesh, rank 0's per-frequency peak rose by up to 3.03 GB (25 %) on the gauged path and 6.88 GB (33 %) on
  plain STRUMPACK, which kept 8.9-17.8 GB of free heap resident between frequencies while the program held 1.14 GB.
- valgrind massif (`scripts/axf_massif.sh`) on a small frame shows the live heap at each factorization's peak flat
  (783-827 MB) with and without the trim; 47 % of it is STRUMPACK's frontal matrices and 21 % its contribution-block
  work vectors.
- Transparent huge pages are not the cause: switching them off for the process (`scripts/axf_nothp.py`) changes the
  growth by amounts of either sign comparable to the run-to-run scatter (up to 0.35 GB).

**With both settings**, at 8 to 64 OpenMP threads per rank on the mid mesh:

- the memory a released factor leaves resident between frequencies is 0-0.07 factors on the plain path and 0.15-0.19
  on the gauged path (one frequency's freed solve vectors, returned by the next factorization's trim), against 0.4-1.5
  with glibc's behaviour. The trim alone works at 8 threads but leaves 0.25 and 0.42 factors resident at 32 and 64;
- the per-frequency peak of the rank-0 factors still rises by 0.25-0.52 GB (6-11 % of the first frequency's) over 20
  frequencies, against +0.55 to +2.2 GB; at run 5's size plain STRUMPACK +1.17 GB (5-6 %), against +4.4 to +6.9 GB;
- over 60 frequencies the distributed path levels off 2 % above the first frequency; the plain path keeps rising by
  about 5-10 MB per frequency at 40-60 solves, 6-9 % above the first in the last third, and is not shown to saturate;
- the factorization itself takes 5-12 % longer (it can no longer reuse resident pages and faults new ones in); a whole
  frequency of this design, with the error indicator and dZ/dω, 1-4 %. One arena (`MALLOC_ARENA_MAX=1`) instead costs
  3-9 %.

**Results unchanged.** Every leaf of the results JSON except timings, memory figures and paths is bit-identical with
and without the settings: ams, plain, distributed and tree-cotree paths on the mid mesh (3726-4890 leaves each),
plain and tree-cotree at run 5's size, and a small frame with its validation features plus the shift-invert
eigensolver on its own; 0 differences in every comparison.

## Test

`tests/test_fullwave3d_memory.py` runs the plain and the gauged path at 8 and 32 threads per rank, each sweep in a
fresh process (MPI ranks, or the `axf` executable in a subprocess). It gates the resident memory left by released
factors at < 0.3 factor and the work memory left right after each factorization, against a negative control with
glibc's behaviour that must exceed 0.4 factor (otherwise the test skips). In the record's calibration the default
reached 0.015-0.190 on the released-factor gate and glibc's behaviour 0.435-3.64. The pre-merge subset
(`tests/premerge.txt`) runs the plain path at 8 threads and the in-process record test.

## Limitations

- **Remaining peak growth** is fragmentation inside each factorization (6-11 % over the first 20 frequencies; on the
  plain path still rising at 60). The adaptive sweep's many-solve behaviour is not measured: at 5-10 MB per solve on
  the mid mesh, a few hundred solves would add 1.5-5 GB on rank 0 if it does not saturate. Every row carries the
  memory record, so a long sweep shows it. `MALLOC_ARENA_MAX=1` in the job's environment lowers the level by about
  0.4 GB on the plain path for 1-5 % more time; it is not set by default.
- **The gauged path** keeps its factor until the next frequency (6.3 GB on rank 0 at run 5's size).
- **After an analysis** the process keeps trim 64 MiB and mmap 32 MiB, not glibc's dynamic adjustment.
- **Address space** is not returned: free chunks stay in the arenas and only their pages are released. That counts
  against an address-space limit (`ulimit -v`), not against the resident set or a batch-system memory limit.
- **Not implemented:** reusing STRUMPACK's ordering and symbolic factorization across the frequencies of a sweep
  (K − ω²M has the same pattern at every frequency). Multi-node runs were not measured.
