"""Shared helpers for the 3D rod-sweep reproductions. beta.

Run artefacts go to $AXF_TRACK_WORK (default runs/cavity_track/<benchmark>/ in the repository); every sweep step is
cached by the hash of its design document (axionfem.cavity_track.sweep), so a re-run only redoes the analysis.
3D solves need MPI ranks: $AXF_TRACK_RANKS (default 16) through Design.solve(mpi_ranks=...).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6
ROOT = Path(__file__).resolve().parents[3]
REFS = ROOT / "benchmarks" / "cavity" / "refs"
RECS = ROOT / "benchmarks" / "records"
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AXF_TRACK_WORK", ROOT / "runs" / "cavity_track"))
RANKS = int(os.environ.get("AXF_TRACK_RANKS", "16"))


def ref(name: str) -> dict:
    return json.loads((REFS / f"{name}.json").read_text())


def skin(sigma: float, T_K: float = 293.0) -> dict:
    return {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.0 / sigma, "T_K": T_K}}


def cyl(x, y, z0, z1, r) -> dict:
    return {"cylinder": {"base": [float(x), float(y), float(z0)], "axis": [0.0, 0.0, float(z1 - z0)], "radius": float(r)}}


def write_record(name: str, rec: dict) -> Path:
    RECS.mkdir(parents=True, exist_ok=True)
    p = RECS / f"{name}.json"
    p.write_text(json.dumps(rec, indent=None, separators=(",", ":"),
                            default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return p


def interp(points, x):
    """Linear interpolation in a digitized [[x, y], ...] series (sorted by x; None outside)."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1 and x1 > x0:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def slope_fit(xs, ys):
    """Least-squares slope and intercept (and the rms residual)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - s * mx
    rms = math.sqrt(sum((y - (s * x + b)) ** 2 for x, y in zip(xs, ys)) / n)
    return s, b, rms
