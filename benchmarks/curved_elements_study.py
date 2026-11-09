"""Curved (order-2) elements on features smaller than the element size: inverted-element counts per mesher variant.

    python benchmarks/curved_elements_study.py                       # every case x every variant
    python benchmarks/curved_elements_study.py --variants none,elastic,full,auto     # the record's variants

Meshes only (`axf mesh`, which reports the solve's curved-element check as `curved_elements_invalid`, before any
straightening). Cases: `shell`, the curved-element probe benchmarks/designs/curved_probe_shell.json (a niobium
toroidal box shell, 5 mm wall, 0.5 mm inner fillets at 20 mm volume / 8 mm surface size, below the 2 mm size floor),
and `core30`, the reduced DMRadio-Core 30-degree port design (1.5 mm lip fillets, 2 mm port, at 0.16 m / 0.025 m;
without the optimizer the solve straightened about 800 of its curved elements). Variants (axf-mesher
environment): AXF_MESHER_HO = elastic | full (elastic, then gmsh's Jacobian-targeted high-order optimizer) | auto
(the default: the optimizer only where the elastic pass leaves inverted elements) | none, crossed with
AXF_MESHER_CURVATURE_FLOOR (the size floor may not clip curvature sizing on the smallest curve radius). `full` and
`auto` add up to two optimizer passes with 2x and 4x the patch layers where inverted elements remain, each kept only
if it lowers the count (mesh info `high_order_passes`), and the curved steps run in a process that has not meshed,
so a case gives the same mesh in every run. Before the volume elements were loaded in a shuffled order, core30 needed
all three passes (2418 inverted after gmsh's pass and after the 12-layer pass, 0 after the 24-layer one; 2507 s
against about 1000 s for one pass). With the shuffled load (seed 1) gmsh's pass cleared core30 and the full-size
shell; it is a fixed draw that behaves like in-process ones, not a tuned one (over seeds, core30 5 of 5, the full-size
shell 5 of 7; the others needed the 12-layer pass), so the extra passes remain needed.
Writes benchmarks/records/curved_elements_study.json.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

VARIANTS = {"elastic": ("elastic", "0"), "full": ("full", "0"), "none": ("none", "0"),
            "elastic+floor": ("elastic", "1"), "full+floor": ("full", "1"), "auto": ("auto", "0")}
CORE = ROOT / "benchmarks" / "published" / "dmradio_core_reduced_120cm_3d_port30.json"


def case_doc(case: str) -> dict:
    if case == "core30":
        return json.loads(CORE.read_text())
    if case == "shell":
        return json.loads((ROOT / "benchmarks" / "designs" / "curved_probe_shell.json").read_text())
    raise ValueError(f"unknown case {case}")


def run(case: str, variant: str, root: Path, timeout: float, build: Path) -> dict:
    ho, floor = VARIANTS[variant]
    wd = root / f"{case}__{variant}"
    wd.mkdir(parents=True, exist_ok=True)
    design = wd / "design.json"
    design.write_text(json.dumps(case_doc(case), indent=1))
    env = dict(os.environ, AXF_MESHER_HO=ho, AXF_MESHER_CURVATURE_FLOOR=floor,
               AXF_MESHER=str(build / "axf-mesher"))
    t0 = time.time()
    # own process group, so that a timeout also kills the axf-mesher child
    p = subprocess.Popen([str(build / "axf"), "mesh", str(design), "-o", str(wd), "-v", "0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return {"case": case, "variant": variant, "AXF_MESHER_HO": ho, "AXF_MESHER_CURVATURE_FLOOR": floor,
                "wall_s": time.time() - t0, "error": f"timeout: no mesh after {timeout:.0f} s"}
    row = {"case": case, "variant": variant, "AXF_MESHER_HO": ho, "AXF_MESHER_CURVATURE_FLOOR": floor,
           "wall_s": time.time() - t0, "returncode": p.returncode}
    if p.returncode != 0:
        row["error"] = (stderr or stdout)[-2000:]
        return row
    out = json.loads(stdout)
    info = json.loads((wd / "mesh3d_info.json").read_text())
    row.update({"curved_elements_invalid": out.get("curved_elements_invalid"), "elements_3d": info.get("elements_3d"),
                "mesh_seconds": info.get("seconds"), "mesh_size_min": info.get("mesh_size_min"),
                "min_curve_radius": info.get("min_curve_radius"), "high_order_optimize": info.get("high_order_optimize"),
                "high_order_counts": info.get("high_order_counts")})
    for f in wd.glob("mesh3d.msh"):
        f.unlink()                                   # meshes are large; the counts are the record
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="shell,core30")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=2400, help="seconds per mesh")
    ap.add_argument("--build", default="build", help="build directory with axf and axf-mesher")
    a = ap.parse_args()
    cases = a.cases.split(",")
    variants = a.variants.split(",")
    build = (ROOT / a.build).resolve()
    tag = "" if a.build == "build" else f"_{Path(a.build).name}"
    root = ROOT / "runs" / f"curved_elements_study{tag}"
    tasks = [(c, v) for c in cases for v in variants]
    path = ROOT / "benchmarks" / "records" / f"curved_elements_study{tag}.json"
    rows = []
    lock = threading.Lock()

    def one(cv):
        r = run(cv[0], cv[1], root, a.timeout, build)
        with lock:                                   # the record is rewritten after every mesh
            rows.append(r)
            rec = {"check": "curved_elements_study", "date": time.strftime("%Y-%m-%d %H:%M"),
                   "timeout_s": a.timeout, "rows": rows}
            path.write_text(json.dumps(rec, indent=1) + "\n")
        return r

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, tasks))
    rows.sort(key=lambda r: (r["case"], r["variant"]))
    for r in rows:
        print(f"{r['case']:9s} {r['variant']:14s} invalid {r.get('curved_elements_invalid')!s:>6} "
              f"tets {r.get('elements_3d')!s:>8} floor {r.get('mesh_size_min')!s:>10.10} "
              f"r_min {r.get('min_curve_radius')!s:>10.10} mesh {r.get('mesh_seconds') or 0:7.1f} s"
              + (f"  ERROR {r['error'][-300:]}" if "error" in r else ""), flush=True)
    rec = {"check": "curved_elements_study", "date": time.strftime("%Y-%m-%d %H:%M"),
           "timeout_s": a.timeout, "rows": rows}
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
