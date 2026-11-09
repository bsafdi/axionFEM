"""Epoxy-induced resonance shift of the 2211.00008 toroid on ONE mesh: the epoxy design with eps' = 1.

    python benchmarks/beyond_mqs_2211/epoxy_same_mesh.py run POLE [--ranks N]   # one pole, two frequencies
    python benchmarks/beyond_mqs_2211/epoxy_same_mesh.py report

axionFEM's epoxy-induced shift of modes 2/3/4 (-0.61/-0.34/-0.39 %) is about half of COMSOL's (-1.49/-1.19/-1.00 %).
Both of axionFEM's numbers came from two meshes (lossless replica on mesh C, epoxy on mesh_epoxy_C), as COMSOL's came
from two model files. This solves `designs/toroid_small_shield_epoxy.json` with the epoxy replaced by eps' = 1,
tan delta = 0 (so the model is the lossless replica) on the identical mesh_epoxy_C file of the epoxy runs
(the mesh of the epoxy design, generated beforehand; see MESH), lossless, STRUMPACK. Each pole p is bracketed by solves at f_p (1 -+ 2e-3)
(at 4e-4 the lossless matrix is too close to singular: the direct solve stopped at true residual 1.2e-6 and was
refused) and estimated with compare_toroid_comsol's bracket formula, w_p = (w_a sqrt(X'_a) + w_b sqrt(X'_b)) /
(sqrt(X'_a) + sqrt(X'_b)), independent of the residue and the smooth background; the single-point estimates
w + X/X' (X' = dX/dw from the energy identity) are kept as a check. The same-mesh shift is then (epoxy pole on mesh_epoxy_C) / (eps' = 1 pole on mesh_epoxy_C) - 1.
Writes benchmarks/beyond_mqs_2211/axionfem_results/epoxy_same_mesh.json.
The mesh is written once, from the epoxy design's own mesh settings (mesh C), with

    axf mesh benchmarks/beyond_mqs_2211/designs/toroid_small_shield_epoxy.json -o runs/epoxy_same_mesh/mesh_epoxy_C

(or set AXF_BMQS_MESH to the mesh3d.msh of an epoxy run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))

# the mesh of the epoxy design (mesh_epoxy_C, `axf mesh` above); AXF_BMQS_MESH overrides the path
MESH = Path(os.environ.get("AXF_BMQS_MESH", ROOT / "runs" / "epoxy_same_mesh" / "mesh_epoxy_C" / "mesh3d.msh"))
# lossless replica on mesh C, and the epoxy model on mesh_epoxy_C (both computed beforehand)
LOSSLESS_C = [26.751e6, 75.445e6, 116.454e6, 149.957e6]
EPOXY_EC = [26.4223e6, 74.986e6, 116.061e6, 149.371e6]
# COMSOL: modes 2-4 from the recovered COMSOL tables; mode 1 from its poles 26.724 (lossless) and 26.4116 MHz (epoxy)
COMSOL_SHIFT = [26.4116 / 26.724 - 1, -1.49e-2, -1.19e-2, -1.00e-2]
REL = 2e-3
RUNS = ROOT / "runs" / "epoxy_same_mesh"
OUT = HERE / "axionfem_results" / "epoxy_same_mesh.json"


def design(freqs: list[float]) -> dict:
    doc = json.loads((HERE / "designs" / "toroid_small_shield_epoxy.json").read_text())
    doc["name"] = "toroid_small_shield_epoxy_eps1"
    for m in doc["materials"].values():
        if m.get("tan_delta", 0) > 0:
            m["eps_r"] = 1.0
            m["tan_delta"] = 0.0
    doc["fullwave"]["frequencies"] = freqs
    doc["mesh"]["file"] = str(MESH)
    doc["mesh"]["info_file"] = str(MESH.with_name("mesh3d_info.json"))
    return doc


def run(pole: int, ranks: int) -> None:
    import axionfem as af
    f0 = LOSSLESS_C[pole]
    freqs = [f0 * (1 - REL), f0 * (1 + REL)]
    wd = RUNS / f"pole{pole + 1}"
    t0 = time.time()
    try:
        af.Design(design(freqs)).solve(wd, verbose=1, mpi_ranks=ranks)
    except Exception as exc:  # noqa: BLE001  (the C++ solver writes results.json before any Python reload)
        print("solve raised:", type(exc).__name__, exc, flush=True)
    print(f"pole {pole + 1}: {time.time() - t0:.0f} s, results in {wd}", flush=True)


def estimates(wd: Path) -> list[dict]:
    r = json.loads((wd / "results.json").read_text())
    out = []
    for s in r["fullwave"]["sweep"]:
        dz = (s.get("d_ImZ_domega_from_energy_H") or {}).get("re")
        X = -s["Z_im_ohm"]                       # COMSOL sign (inductive > 0), as compare_toroid_comsol.py
        dX = -float(dz[0][0]) if dz else None
        w = 2 * math.pi * s["f_Hz"]
        solver = s.get("solver") or {}
        out.append({"f_Hz": s["f_Hz"], "X_ohm": X, "dX_dw_H": dX,
                    "f_pole_Hz": (w + X / dX) / (2 * math.pi) if dX else None,
                    "accepted": s.get("validation_accepted", solver.get("accepted")),
                    "true_relative_residual": solver.get("true_relative_residual")})
    return out


def report() -> None:
    rows = []
    for p in range(4):
        wd = RUNS / f"pole{p + 1}"
        if not (wd / "results.json").exists():
            rows.append({"pole": p + 1, "missing": True})
            continue
        e = sorted(estimates(wd), key=lambda x: x["f_Hz"])
        fps = [x["f_pole_Hz"] for x in e if x["f_pole_Hz"]]
        a, b = e[0], e[-1]
        if len(e) == 2 and a["X_ohm"] > 0 > b["X_ohm"] and a["dX_dw_H"] and b["dX_dw_H"]:
            sa, sb = math.sqrt(a["dX_dw_H"]), math.sqrt(b["dX_dw_H"])
            fp = (a["f_Hz"] * sa + b["f_Hz"] * sb) / (sa + sb)
        else:
            fp = sum(fps) / len(fps)
        row = {"pole": p + 1, "points": e, "f_pole_eps1_Hz": fp, "spread_rel": (max(fps) - min(fps)) / fp,
               "lossless_meshC_Hz": LOSSLESS_C[p], "epoxy_meshEC_Hz": EPOXY_EC[p],
               "mesh_effect_rel": fp / LOSSLESS_C[p] - 1,
               "shift_same_mesh_rel": EPOXY_EC[p] / fp - 1,
               "shift_two_meshes_rel": EPOXY_EC[p] / LOSSLESS_C[p] - 1,
               "comsol_shift_rel": COMSOL_SHIFT[p]}
        rows.append(row)
        print(f"pole {p + 1}: eps'=1 on mesh_epoxy_C {fp / 1e6:.4f} MHz (two estimates differ by {row['spread_rel']:.1e}); "
              f"mesh C {LOSSLESS_C[p] / 1e6:.4f} ({row['mesh_effect_rel']:+.2e}); shift same mesh "
              f"{row['shift_same_mesh_rel']:+.3%} vs two meshes {row['shift_two_meshes_rel']:+.3%}"
              + (f", COMSOL {COMSOL_SHIFT[p]:+.2%}" if COMSOL_SHIFT[p] else ""), flush=True)
    OUT.write_text(json.dumps({"check": "epoxy_same_mesh", "date": time.strftime("%Y-%m-%d"), "mesh": str(MESH),
                               "rel_offset": REL, "rows": rows}, indent=1) + "\n")
    print("wrote", OUT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("run", "report"))
    ap.add_argument("pole", nargs="?", type=int, help="1..4 (run)")
    ap.add_argument("--ranks", type=int, default=32)
    a = ap.parse_args()
    if a.what == "run":
        run(a.pole - 1, a.ranks)
    else:
        report()


if __name__ == "__main__":
    main()
