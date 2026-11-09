"""Port-feed continuity at low frequency: does each port's electric energy scale as f^2 down to 100 Hz?

    python benchmarks/feed_continuity_check.py BUILD_DIR [NRANKS]

The port feed J = -grad V is the conduction solution in the port region; its discrete divergence at the free port
nodes is that solve's residual, and at low frequency the field amplifies the resulting charge by about (k h)^-2.
Without iterative refinement of the conduction solve to the rounding floor (model3d.cpp), a residual of 1.2e-14 made
the two-port frame's port-2 electric energy 12.3 % high at 100 Hz in every solver path, with every gate passing. This
runs benchmarks/designs/stage1_twoport.json at 100 Hz, 1 kHz, 10 kHz and 100 kHz with the gauged (tree-cotree) and the
plain STRUMPACK path, using BUILD_DIR/axf, and prints U_e,jj / f^2 per port: constant down to 100 Hz when the feed is
divergence free to rounding. Writes benchmarks/records/feed_continuity_check_<BUILD_DIR name>.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
FREQUENCIES = [1e2, 1e3, 1e4, 1e5]
CONFIGS = {"tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"},
           "strumpack": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none"}}
# the gauged-path values of benchmarks/records/tree_cotree_validation.json before the feed-continuity fix
BEFORE = {"U_e11_over_f2": 2.901516e-23, "U_e22_over_f2_1kHz_up": 2.90709e-23, "U_e22_over_f2_100Hz": 3.265455e-23}


def _mpi_launcher(nranks: int) -> list:
    """The launcher prefix for nranks > 1, by the rule of axionfem.Design.solve: $AXF_MPI_LAUNCHER, else srun inside
    a SLURM job when srun is on PATH, else mpirun, else mpiexec."""
    if nranks <= 1:
        return []
    import shlex
    import shutil
    cmd = os.environ.get("AXF_MPI_LAUNCHER")
    if not cmd:
        if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
            cmd = "srun"
        else:
            cmd = "mpirun" if shutil.which("mpirun") or not shutil.which("mpiexec") else "mpiexec"
    return shlex.split(cmd) + ["-n", str(nranks)]


def run(build: Path, name: str, env_extra: dict, nranks: int, root: Path) -> dict:
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    doc = json.loads(DESIGN.read_text())
    doc["fullwave"]["frequencies"] = FREQUENCIES
    design = wd / "design.json"
    design.write_text(json.dumps(doc, indent=1))
    env = dict(os.environ, AXF_MESHER=str(build / "axf-mesher"), **env_extra)
    launcher = _mpi_launcher(nranks)
    t0 = time.time()
    p = subprocess.run(launcher + [str(build / "axf"), "solve", str(design), "-o", str(wd), "-v", "1"],
                       capture_output=True, text=True, env=env)
    (wd / "solve.log").write_text(p.stdout + "\n" + p.stderr)
    out = {"config": name, "env": env_extra, "returncode": p.returncode, "seconds": time.time() - t0}
    res = wd / "results.json"
    if not res.exists():
        out["error"] = (p.stderr or p.stdout)[-1500:]
        return out
    r = json.loads(res.read_text())
    fw = r["fullwave"]
    out["ports"] = [{k: q.get(k) for k in ("region", "conduction_relative_residual", "conduction_refinement_steps")}
                    for q in fw.get("ports", [])]
    rows = []
    for s in fw.get("sweep", []):
        ue = s["energy_electric_J_per_A2"]["re"]
        f = s["f_Hz"]
        rows.append({"f_Hz": f, "U_e11_over_f2": ue[0][0] / f ** 2, "U_e22_over_f2": ue[1][1] / f ** 2,
                     "ratio_22_11": ue[1][1] / ue[0][0], "accepted": s.get("validation_accepted"),
                     "gradient_block_check": (s.get("direct_factorization") or {}).get("gradient_block_check")})
    out["sweep"] = rows
    if "rejected" in fw:
        out["rejected"] = fw["rejected"]
    return out


def main() -> None:
    build = Path(sys.argv[1]).resolve()
    nranks = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    root = ROOT / "runs" / "feed_continuity_check" / build.name
    rec = {"check": "feed_continuity_check", "date": time.strftime("%Y-%m-%d %H:%M"), "build": build.name,
           "before": BEFORE, "configs": {}}
    for name, env in CONFIGS.items():
        c = run(build, name, env, nranks, root)
        rec["configs"][name] = c
        print(f"== {name}: rc {c['returncode']}, {c['seconds']:.0f} s, ports {c.get('ports')}", flush=True)
        for row in c.get("sweep", []):
            print(f"   {row['f_Hz']:9.0f} Hz  U_e11/f^2 {row['U_e11_over_f2']:.6e}  U_e22/f^2 {row['U_e22_over_f2']:.6e}"
                  f"  ratio {row['ratio_22_11']:.7f}  accepted {row['accepted']}", flush=True)
            g = row.get("gradient_block_check") or {}
            if g:
                print(f"             gradient block: legacy {g.get('legacy_summed_matrix_relative_error_on_gradients')}"
                      f"  separate K / w^2 M {g.get('separate_K_on_gradients_relative_to_w2_M')}", flush=True)
        if "error" in c:
            print("   error:", c["error"][-600:], flush=True)
    path = ROOT / "benchmarks" / "records" / f"feed_continuity_check_{build.name}.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
