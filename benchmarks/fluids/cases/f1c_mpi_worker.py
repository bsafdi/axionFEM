"""F1c MPI worker (one process per rank, started by f1c_stokes through cases/fluids_mpi.launch): reads the task
file named on the command line, runs f1c_stokes.worker(task) on every rank, and writes this rank's result to
<task>.rank<R>.json.  All the work (solve, Evaluator, the check module's partial sums) is in f1c_stokes; this file
only restores the case's import path inside the MPI child."""
from __future__ import annotations

import json
import sys
import traceback

VERSION = 1


def main(path: str) -> int:
    task = json.loads(open(path).read())
    for p in reversed(task.get("sys_path") or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    import f1c_stokes as R                    # noqa: E402  (the case module, from the restored path)
    from axionfem import fluids                # noqa: E402
    rank = int(fluids._core._fluids.world_rank())
    try:
        out = R.worker(task)
        out["ok"] = True
    except Exception as e:                     # noqa: BLE001  (reported to the parent, which raises)
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    with open(f"{path}.rank{rank}.json", "w") as f:
        json.dump(out, f, default=R.plain)
    if not out["ok"]:
        import os
        os._exit(3)                            # the other ranks may wait in a collective: end the whole job
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
