#!/usr/bin/env python3
"""AXF_EXE wrapper: run AXF_REAL_EXE with transparent huge pages disabled for the process (Linux
prctl(PR_SET_THP_DISABLE), kept across exec). A diagnostic of memory growth (benchmarks/fullwave_memory.py)."""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    sys.exit(f"prctl(PR_SET_THP_DISABLE) failed: errno {ctypes.get_errno()}")
exe = os.environ["AXF_REAL_EXE"]
os.execv(exe, [exe] + sys.argv[1:])
