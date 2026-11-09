"""Core 3D: separate mesh and dX/dw-stencil effects on F_eff.

Recomputes the fine (core3d) run's F_eff ratios on the fast run's 4 frequencies (the same finite-difference
stencil for dX/dw) and compares them with the fast (core3d_fast) run. Reads runs/published/core3d_*/results.json.
"""
import json
import math
from pathlib import Path

import numpy as np

RUNS = Path("runs/published")
COMMON = [70.0, 88.19, 111.12, 140.0]


def table(name, keep=None):
    r = json.loads((RUNS / name / "results.json").read_text())
    sweep = [s for s in r["fullwave"]["sweep"] if keep is None or round(s["f_Hz"] / 1e6, 2) in keep]
    f = np.array([s["f_Hz"] for s in sweep])
    w = 2 * math.pi * f
    X = w * np.array([s["L_H"] for s in sweep])
    dX = np.gradient(X, w)
    Le = (w * dX + X) / (2 * w)
    psi = np.array([s["Psi_abs"] for s in sweep])
    return {round(fi / 1e6, 2): (p ** 4 / l ** 2, l, xi) for fi, p, l, xi in zip(f, psi, Le, X)}


for port in ("30", "360"):
    fast = {f: table(f"core3d_port{port}_fast")[f][0] / table("core3d_axi_fast")[f][0] for f in COMMON}
    fine4 = {f: table(f"core3d_port{port}", COMMON)[f][0] / table("core3d_axi", COMMON)[f][0] for f in COMMON}
    fine7 = {f: table(f"core3d_port{port}")[f][0] / table("core3d_axi")[f][0] for f in COMMON}
    print(f"port {port}: f   fast(4pt)  fine(4pt, same stencil)  fine(7pt)   fine4/fast-1")
    for f in COMMON:
        print(f"   {f:7.2f}  {fast[f]:.4f}     {fine4[f]:.4f}                 {fine7[f]:.4f}     {fine4[f] / fast[f] - 1:+.2e}")
