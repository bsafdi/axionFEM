#!/bin/bash
# AXF_EXE wrapper: run AXF_REAL_EXE under valgrind massif (heap and its allocation sites), one output per rank in
# AXF_HEAP_DIR.
rank="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"
mkdir -p "${AXF_HEAP_DIR}"
exec valgrind --tool=massif --massif-out-file="${AXF_HEAP_DIR}/massif.rank${rank}.out" --max-snapshots=400 \
  --detailed-freq=4 --threshold=0.5 "${AXF_REAL_EXE}" "$@"
