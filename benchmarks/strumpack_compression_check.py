"""Block low-rank (BLR) compression of the full-wave STRUMPACK factor: memory, time, iterations, answer.

    python benchmarks/strumpack_compression_check.py run NAME MODE [TOL] [--ranks N]
    python benchmarks/strumpack_compression_check.py report

With the serial STRUMPACK (or AXF_FW_STRUMPACK_DIST=0) the direct full-wave path factorizes the gathered real
Maxwell matrix on rank 0: at 3.87 M order-2 DOFs (the 2211 toroid on mesh_epoxy_C) the factor has 4.68e9 nonzeros and the run peaks at
50.6 GB on rank 0, 590 s per factorization. The factor is only a preconditioner (GMRES refines against the exact
distributed operator and the acceptance gates are unchanged), so a lossy compressed factor should change the
iteration count, not the accepted answer. Case: benchmarks/beyond_mqs_2211/epoxy_same_mesh.py's design (eps' = 1 on
mesh_epoxy_C, lossless) at 75.4148 MHz, 0.04 % below pole 2 (the case accepted at true residual 1.2e-7). Runs use
AXF_EXE (default build/axf). Writes benchmarks/records/strumpack_compression_check.json. The mesh is the one
epoxy_same_mesh.py uses (its docstring says how to write it).
The dist0_* / dist1_* runs compare the rank-0 factor with the distributed one (AXF_FW_STRUMPACK_DIST=0/1)
from one build against an MPI STRUMPACK, each on one kind of node, at 32 x 1, 4 x 8 and 8 x 4 (ranks x threads) and on
2 nodes (dist1_2x32). Each run writes layout.json (ranks, threads per rank, AXF_FW_STRUMPACK_DIST); report() reads it
back.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE / "beyond_mqs_2211"))

F_HZ = 75.4148e6
RUNS = ROOT / "runs" / "strumpack_compression_check"
OUT = ROOT / "benchmarks" / "records" / "strumpack_compression_check.json"


def run(name: str, mode: str, tol: str | None, ranks: int) -> None:
    import axionfem as af
    import epoxy_same_mesh as E
    os.environ["AXF_FW_SOLVER"] = "strumpack"
    os.environ["AXF_FW_STRUMPACK_COMPRESSION"] = mode
    if tol:
        os.environ["AXF_FW_STRUMPACK_COMPRESSION_TOL"] = tol
    wd = RUNS / name
    t0 = time.time()
    try:
        af.Design(E.design([F_HZ])).solve(wd, verbose=1, mpi_ranks=ranks)
    except Exception as exc:  # noqa: BLE001  (results.json is written by the C++ solver)
        print("solve raised:", type(exc).__name__, exc, flush=True)
    (wd / "wall_seconds.txt").write_text(f"{time.time() - t0:.1f}\n")
    layout = {"mpi_ranks": ranks, "threads_per_rank": int(os.environ.get("OMP_NUM_THREADS", "1")),
              "AXF_FW_STRUMPACK_DIST": os.environ.get("AXF_FW_STRUMPACK_DIST", "auto")}
    (wd / "layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)


# runs before layout.json was written: the layout from the name
OLD_LAYOUTS = {"none_8x4": "8 ranks x 4 threads", "none_4x8": "4 ranks x 8 threads"}


def _layout(wd: Path) -> dict | str:
    f = wd / "layout.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"description": OLD_LAYOUTS.get(wd.name, "32 ranks x 1 thread")}


def report() -> None:
    rows = []
    for wd in sorted(RUNS.iterdir()):
        res = wd / "results.json"
        rss = wd / "maxrss.txt"
        wall = wd / "wall_seconds.txt"
        if not res.exists():
            rej = wd / "fullwave_rejected.json"
            row = {"name": wd.name, "refused": rej.exists()}
            if rej.exists():
                fw = json.loads(rej.read_text()).get("fullwave", {})
                row["gmres_iterations_total"] = fw.get("gmres_iterations_total")
                row["compression"] = (json.loads(rej.read_text()).get("primary_row") or {}).get(
                    "direct_factorization", {}).get("compression")
            log = wd / "mpi_solve.log"
            if log.exists():
                row["solver_line"] = next((ln.strip() for ln in log.read_text().splitlines() if "true residual" in ln), None)
            row["maxrss"] = rss.read_text().strip().splitlines()[-1] if rss.exists() else None
            row["wall_seconds"] = float(wall.read_text()) if wall.exists() else None
            rows.append(row)
            continue
        r = json.loads(res.read_text())
        s = r["fullwave"]["sweep"][0]
        d = s.get("direct_factorization") or {}
        sol = s.get("solver") or {}
        rows.append({"name": wd.name, "backend": d.get("backend"), "factor_ranks": d.get("factor_ranks", 1),
                     "compression": d.get("compression"), "factor_nonzeros": d.get("factor_nonzeros"),
                     "factor_seconds": d.get("seconds"), "gmres_iterations": sol.get("gmres_iterations"),
                     "true_relative_residual": sol.get("true_relative_residual", sol.get("relative_residual")),
                     "accepted": s.get("validation_accepted", sol.get("accepted")),
                     "Z": [s["Z_re_ohm"], s["Z_im_ohm"]], "Psi": [s["Psi_re"], s["Psi_im"]],
                     "maxrss": rss.read_text().strip().splitlines()[-1] if rss.exists() else None,
                     "wall_seconds": float(wall.read_text()) if wall.exists() else None,
                     "layout": _layout(wd)})
    ref = next((x for x in rows if x["name"] == "none" and "Z" in x), None)
    for x in rows:
        if ref and "Z" in x:
            zr = complex(*ref["Z"])
            pr = complex(*ref["Psi"])
            x["Z_rel_to_none"] = abs(complex(*x["Z"]) - zr) / abs(zr)
            x["Psi_rel_to_none"] = abs(complex(*x["Psi"]) - pr) / abs(pr)
        print(json.dumps({k: v for k, v in x.items() if k not in ("Z", "Psi")}, default=str))
    OUT.write_text(json.dumps({"check": "strumpack_compression_check", "f_Hz": F_HZ, "rows": rows}, indent=1) + "\n")
    print("wrote", OUT)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ranks = 32
    if "--ranks" in sys.argv:
        ranks = int(sys.argv[sys.argv.index("--ranks") + 1])
        args = [a for a in args if a != str(ranks)]
    if args[0] == "run":
        run(args[1], args[2], args[3] if len(args) > 3 else None, ranks)
    else:
        report()


if __name__ == "__main__":
    main()
