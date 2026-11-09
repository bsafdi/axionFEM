"""Sparse-direct performance of the axisymmetric poloidal solver at large problem sizes.

Assembly uses per-thread triplets and the factorization is the serial
Eigen SimplicialLDLT.  This measures approximate-minimum-degree (Eigen's default) against METIS
nested dissection (Eigen/MetisSupport) on the CORE template, which is the largest 2D case in the
benchmark set.

Each (case, ordering) runs in its own subprocess, because Eigen's AMD ordering *segfaults* on the
largest case: a crash must be recorded as a result row, not take the whole benchmark down.

Run as a batch job. The conclusion it supports (Eigen's AMD ordering is unreliable on the largest meridional
systems, METIS nested dissection is not) is stated in docs/install.md, "Sparse direct solvers".

usage: ordering_benchmark.py OUT.json [order,refinements ...]
       ordering_benchmark.py --one ORDER REFINEMENTS ORDERING WORKDIR     (one run, internal)
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

CASES = [(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (4, 1), (3, 2)]


def peak_rss_GB() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def run_one(order: int, refinements: int, ordering: str, workdir: Path) -> dict:
    """One solve in this process; prints the row as JSON on stdout."""
    import axionfem as af

    os.environ["AXF_ORDERING"] = ordering
    d = af.template("core")
    d.doc["mesh"]["refinements"] = refinements
    t0 = time.time()
    # self_force off: this measures the two solves every design does, not the extra diagnostic one
    res = d.solve(workdir=workdir, order=order, verbose=0, threads=1, self_force=False)
    wall = time.time() - t0
    out = {"order": order, "refinements": refinements, "ordering": ordering,
           "dofs": res.data["mesh"]["dofs"], "elements": res.data["mesh"]["elements"],
           "wall_seconds": wall, "peak_rss_GB": peak_rss_GB(),
           "Psi": res.Psi, "L_p": res.L_p, "W_J": res.data["coupling"]["W_mag_J"]}
    s = res.data["static"]["poloidal"].get("solver", {})
    if s:
        out["lab"] = {k: s.get(k) for k in ("ordering", "reduced_unknowns", "nonzeros_L",
                                            "symbolic_seconds", "factorization_seconds", "seconds")}
    return out


MARK = "AXF-ROW "


def main() -> None:
    if sys.argv[1] == "--one":
        order, refinements, ordering, workdir = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
        print(MARK + json.dumps(run_one(order, refinements, ordering, workdir)), flush=True)
        return

    out_path = Path(sys.argv[1])
    cases = [tuple(int(x) for x in a.split(",")) for a in sys.argv[2:]] or CASES
    rows: list[dict] = []
    wd = out_path.parent / "ordering_runs"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for order, ref in cases:
        for ordering in ("amd", "metis"):
            tag = f"core_p{order}_r{ref}_{ordering}"
            cmd = [sys.executable, "-u", __file__, "--one", str(order), str(ref), ordering, str(wd / tag)]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            row: dict = {"order": order, "refinements": ref, "ordering": ordering}
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith(MARK)), None)
            if line:
                row = json.loads(line[len(MARK):])
            else:
                # a crash (Eigen's AMD segfaults on the largest case) or any other failure
                row["error"] = f"exit {p.returncode}" + (f" (signal {-p.returncode})" if p.returncode < 0 else "")
                row["wall_seconds"] = time.time() - t0
                row["stderr_tail"] = p.stderr[-1500:]
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "stderr_tail"}), flush=True)
            out_path.write_text(json.dumps({"case": "core", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
