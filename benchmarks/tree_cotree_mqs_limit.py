"""Tree-cotree validation, run 3: the frame's low-frequency L against MQS with superconductors.

    python benchmarks/tree_cotree_mqs_limit.py run [--ranks N]
    python benchmarks/tree_cotree_mqs_limit.py report

Run 3: the two-port frame's inductance at 100 Hz from the
gauged full-wave path against an MQS solve with a superconducting frame inside a superconducting sphere of 0.7 m, at
three mesh levels; their difference should converge to <= 0.1 %.

One design per level, solved once: the two-port frame (benchmarks/designs/stage1_twoport.json; it is cut by its slot,
so simply connected: no fluxoid condition) made of `superconductor`, inside a superconducting shell from 0.70 to 0.75
m, the domain radius 0.8 m, the full wave at 10 Hz and 100 Hz on AXF_FW_GAUGE=tree-cotree (default charge projection
2). The full wave treats both superconductors as perfect conductors, so its field lives inside the 0.7 m wall; the
MQS solve that every run makes on the same mesh (coupling.L_p_H) excludes the field from both. So each level compares
L = -Im Z11 / w (the primary port, the second open) with the MQS L_p of the same port on the same mesh, and the
levels show how far each is from convergence. Mesh levels: the two-port frame's size and surface size (0.32 m, 0.16 m) scaled by
1/2, 1/(2 sqrt 2) and 1/4 (a first try at 1, 1/sqrt 2 and 1/2 barely refined the frame: L moved 5 % between levels at
109k DOFs each); h = DOFs^(-1/3) for the extrapolation. Writes benchmarks/records/tree_cotree_mqs_limit.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

RUNS = Path(os.environ.get("AXF_TC_MQS_RUNS", ROOT / "runs" / "tree_cotree_mqs_limit")).resolve()
OUT = ROOT / "benchmarks" / "records" / "tree_cotree_mqs_limit.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
SCALES = [0.5, 0.5 / math.sqrt(2), 0.25, 0.25 / math.sqrt(2), 0.125]
FREQUENCIES = [1e1, 1e2]
WALL = (0.70, 0.75)
DOMAIN = 0.8
ENV = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}


def design(scale: float) -> dict:
    doc = json.loads(STAGE1.read_text())
    doc["name"] = f"tree_cotree_mqs_limit_{scale:.3f}"
    for region in doc["regions"]:
        if region["name"] == "loop":
            region["material"] = "superconductor"
    sphere = [{"sphere": {"center": [0, 0, 0], "radius": r}} for r in WALL]
    doc["regions"].append({"name": "wall", "material": "superconductor", "solid": {"difference": [sphere[1], sphere[0]]}})
    doc["mesh"]["domain_radius"] = DOMAIN
    doc["mesh"]["size"] = doc["mesh"]["size"] * scale
    doc["mesh"]["surface_size"] = doc["mesh"]["surface_size"] * scale
    doc["fullwave"]["frequencies"] = FREQUENCIES
    return doc


def solve(name: str, doc: dict, env: dict, ranks: int) -> None:
    import axionfem as af
    for k, v in env.items():
        os.environ[k] = v
    t0 = time.time()
    try:
        af.Design(doc).solve(RUNS / name, verbose=0, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (the report records what is missing)
        print(f"{name} failed: {type(exc).__name__}: {exc}", flush=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name}
    (RUNS / name / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def run(ranks: int) -> None:
    for i, scale in enumerate(SCALES):
        if (RUNS / f"l{i}" / "results.json").exists():   # done (a rerun after adding levels)
            print(f"l{i}: kept", flush=True)
            continue
        solve(f"l{i}", design(scale), ENV, ranks)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


def provenance() -> dict:
    import axionfem as af
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": file_record(exe), "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")),
            "python_core": file_record(af._core.__file__), "driver": file_record(__file__),
            "date": time.strftime("%Y-%m-%d")}


def report() -> None:
    from axionfem.convergence import richardson
    rec = {"check": "tree_cotree_mqs_limit", "provenance": provenance(), "scales": SCALES,
           "frequencies_Hz": FREQUENCIES, "wall_m": WALL, "domain_radius_m": DOMAIN, "levels": [], "extrapolation": {}}
    pts = {"fw_100Hz": [], "mqs": [], "fw_over_mqs": []}
    for i, scale in enumerate(SCALES):
        res = RUNS / f"l{i}" / "results.json"
        level = {"scale": scale}
        if not res.exists():
            level["missing"] = True
            rec["levels"].append(level)
            continue
        d = json.loads(res.read_text())
        sweep = {round(r["f_Hz"]): r for r in d["fullwave"]["sweep"]}
        level.update({"dofs": d["mesh"]["dofs"], "elements": d["mesh"].get("elements"),
                      "mesh_sha256": sha256(RUNS / f"l{i}" / "mesh3d.msh"),
                      "L_fw_H": {f"{f:g}": sweep[round(f)]["L_H"] for f in FREQUENCIES if round(f) in sweep},
                      "L_mqs_H": d["coupling"]["L_p_H"], "solver_path": d["fullwave"].get("solver_path"),
                      "pickup_bodies": d["pickup"].get("bodies"), "warnings": d.get("warnings", [])})
        if "100" in level["L_fw_H"]:
            level["fw_over_mqs_minus_1"] = {f: v / level["L_mqs_H"] - 1 for f, v in level["L_fw_H"].items()}
            h = level["dofs"] ** (-1 / 3)
            pts["fw_100Hz"].append((h, level["L_fw_H"]["100"]))
            pts["mqs"].append((h, level["L_mqs_H"]))
            pts["fw_over_mqs"].append((h, level["L_fw_H"]["100"] / level["L_mqs_H"]))
            print(f"level {i} (scale {scale:.3f}, {level['dofs']} DOFs): L_fw(100 Hz) {level['L_fw_H']['100']:.8e}, "
                  f"L_mqs {level['L_mqs_H']:.8e}, fw/mqs - 1 = {level['fw_over_mqs_minus_1']['100']:+.3e} "
                  f"(10 Hz {level['fw_over_mqs_minus_1'].get('10', float('nan')):+.3e})", flush=True)
        rec["levels"].append(level)
    for key, values in pts.items():
        if len(values) >= 3:   # the three finest levels
            finest = sorted(values)[:3]
            ex = richardson([h for h, _ in finest], [v for _, v in finest])
            rec["extrapolation"][key] = ex
            print(key, "extrapolated:", ex.get("extrapolated"), "order", ex.get("order"), ex.get("behaviour"), flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    # the committed record only from the default run directory (a trial with AXF_TC_MQS_RUNS writes its own copy)
    if RUNS == (ROOT / "runs" / "tree_cotree_mqs_limit").resolve():
        OUT.write_text(json.dumps(rec, indent=1) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_TC_MQS_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        run(int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4")))
    report()


if __name__ == "__main__":
    main()
