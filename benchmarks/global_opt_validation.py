"""Global (DE) optimization against the known optima of docs/validation/adjoint.md, from no initial guess.

The three cases of benchmarks/adjoint_validation.py (same designs, meshes and bounds), searched by
differential evolution over the whole box instead of descending from a start value:
  1. toroid with H = R: maximise g over y        (adjoint optimum 0.5675; long limit exp(-1/2) = 0.6065)
  2. long solenoid H/R = 8: maximise g over x     (adjoint optimum 0.5314; long limit 0.534)
  3. finite current-sheet solenoid: minimise the cost metric at the cost-optimal size over H and x (the adjoint
     optimum, and the derivative-free one of benchmarks/m1_validation.py; the cost metric is a scaling metric, used
     here only as a smooth objective with an interior optimum)
The adjoint optima are read from benchmarks/records/adjoint_validation.json.
Record: benchmarks/records/global_opt_validation.json.  Run as a batch job, e.g. with 16 cores:

    python benchmarks/global_opt_validation.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("AXF_PYTHONPATH", str(ROOT / "python")))
import axionfem as af  # noqa: E402

# the readout benchmark of the cost case: the same as m1_validation.REF_BENCH and adjoint_validation.REF_BENCH
REF_BENCH = {"name": "reference", "Q": 1e6, "T": 0.1, "eta_A": 1.0}

def adjoint_optima() -> dict:
    """The optima of the three gradient-based runs of benchmarks/adjoint_validation.py, from its record."""
    rec = json.loads((ROOT / "benchmarks" / "records" / "adjoint_validation.json").read_text())
    o1, o2, o3 = rec["optimization"]
    return {"toroid_y": o1["optimum"]["y"], "solenoid_x": o2["optimum"]["x"],
            "sol_cost": {"H": o3["optimum"]["H"], "x": o3["optimum"]["x"], "cost_optimal": o3["value"]}}


def summary(o: dict) -> dict:
    return {"optimum": o["parameters"], "value": o["value"], "feasible": o["feasible"], "solves": o["solves"],
            "evaluations": o["evaluations"], "converged": o["converged"], "message": o["message"],
            "de": {k: o["de"][k] for k in ("nit", "nfev", "converged", "scipy_message", "evaluations", "population",
                                           "seconds", "seed", "popsize")},
            "de_best": o["de_best"], "polish": o["polish"], "found_by": o["found_by"], "seconds": o["seconds"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--work", default=str(ROOT / "runs" / "global_opt_validation"))
    a = ap.parse_args()
    wd = Path(a.work)
    kw = dict(method="global", seed=a.seed, workers=a.workers, verbose=1)
    ADJOINT = adjoint_optima()
    runs = []

    d = af.template("toroid")
    t0 = time.time()
    o = af.optimize(d, objective="g_factor", sense="max", params={"y": (0.25, 0.9)}, popsize=8, maxiter=20,
                    workdir=wd / "toroid", **kw)
    runs.append({"case": "toroid with H = R: maximise g over y", "reference": {"adjoint_optimum_y": ADJOINT["toroid_y"],
                 "y_long_limit": math.exp(-0.5)}, **summary(o), "wall_s": time.time() - t0})

    d2 = af.template("solenoid", H=8.0, R=1.0)
    d2.doc["mesh"] = {"order": 2, "size": "max(R, H)/30", "surface_size": "0.004*R"}
    t0 = time.time()
    o2 = af.optimize(d2, objective="g_factor", sense="max", params={"x": (0.3, 0.75)}, popsize=8, maxiter=20,
                     workdir=wd / "solenoid_x", **kw)
    runs.append({"case": "long solenoid (H/R = 8): maximise g over x",
                 "reference": {"adjoint_optimum_x": ADJOINT["solenoid_x"], "x_long_limit": 0.534},
                 **summary(o2), "wall_s": time.time() - t0})

    d3 = af.template("solenoid", R=1.0, tw="0.002*R", g="0.001*R", tc="0.001*R", tp="0.002*R",
                     mesh={"order": 3, "size": "max(R,H)/30", "surface_size": "0.002*R"}, benchmarks=[REF_BENCH])
    t0 = time.time()
    o3 = af.optimize(d3, objective="cost_optimal", params={"H": (0.6, 4.0), "x": (0.35, 0.75)}, popsize=8,
                     maxiter=25, benchmark=REF_BENCH["name"], workdir=wd / "solenoid_cost", **kw)
    res = af.Design(o3["design"]).solve(workdir=wd / "solenoid_cost" / "best", verbose=0)
    c = res.benchmark(REF_BENCH["name"])["cost_optimal_size"]
    got = {"R_m": c["device_radius_m"], "H_m": o3["parameters"]["H"] * c["length_scale"], "x": o3["parameters"]["x"],
           "W_MJ": c["W_mag_J"] / 1e6, "cost_MCHF": c["cost"]["total_MCHF"]}
    runs.append({"case": "finite current-sheet solenoid: cost metric at the cost-optimal size over H and x",
                 "reference": {"adjoint": ADJOINT["sol_cost"]},
                 **summary(o3), "computed": got, "wall_s": time.time() - t0})

    import subprocess

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "python"],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "", False
    rec = {"check": "global_opt_validation", "date": time.strftime("%Y-%m-%d %H:%M"), "axionfem_version": af.version(),
           "commit": commit + ("+dirty" if dirty else ""), "module": Path(sys.modules["axionfem.global_opt"].__file__).name, "workers": a.workers, "runs": runs}
    out = ROOT / "benchmarks" / "records" / "global_opt_validation.json"
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    for r in runs:
        print(r["case"], r["optimum"], r["value"], "solves", r["solves"], "found by", r["found_by"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
