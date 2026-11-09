#!/bin/bash
# AXF_EXE wrapper: run AXF_REAL_EXE under gperftools' heap profiler (tcmalloc as the allocator); one profile series per
# rank in AXF_HEAP_DIR. Each dump's header line gives the in-use heap at that moment. AXF_TCMALLOC names the tcmalloc
# library (default: the usual location on RHEL-family systems).
rank="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"
mkdir -p "${AXF_HEAP_DIR}"
export LD_PRELOAD="${AXF_TCMALLOC:-/usr/lib64/libtcmalloc.so.4}"
export HEAPPROFILE="${AXF_HEAP_DIR}/axf.rank${rank}"
export HEAP_PROFILE_ALLOCATION_INTERVAL="${HEAP_PROFILE_ALLOCATION_INTERVAL:-2147483648}"
export HEAP_PROFILE_INUSE_INTERVAL="${HEAP_PROFILE_INUSE_INTERVAL:-268435456}"
exec "${AXF_REAL_EXE}" "$@"
