"""Mode tracking through a rod-angle sweep of the HAYSTAC one-rod cross-section (no end gaps). beta.

The z-uniform limit of the HAYSTAC cavity (refs haystac_rod_2006.json: R = 2 in, rod r = 1 in, pivot 0.475 in from
the axis and from the rod axis) solved with the cross-section backend (TM_z and TE_z families) for theta = 0..180
deg, tracked with axionfem.cavity.sweep. The barrel plus rod is mirror-symmetric about the line through the axis and
the rod centre, so TM_z modes are even or odd about it: modes of opposite parity cross, modes of equal parity
repel. TE_z modes (C = 0) cross every TM_z mode. The TM010-like branch (largest C at theta = 0) is checked against
the method-of-particular-solutions reference cavity_xsec.eccentric_rod_tm_mode at a few angles.

Output: docs/images/cavity_track_xsec_rod.png, benchmarks/records/cavity_track_xsec_rod.json.
Run on a login node: $PY xsec_rod_tracking.py  (~10 min, cached in $AXF_TRACK_WORK/xsec_rod_tracking).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common3d as cm  # noqa: E402

IN = 0.0254
R, A, P = 2 * IN, 1 * IN, 0.475 * IN


def build(theta):
    from axionfem import cavity_xsec as cx

    t = math.radians(theta)
    rod = {"radius": A, "center": [P - P * math.cos(t), P * math.sin(t)]}
    return cx.cylinder_rods_design(R, [rod], modes=12, families=("TM", "TE"), target_f_Hz=5.0e9, h=R / 20,
                                   wall_rf=cx.skin_rf(5.8e7, 293.0), length=10 * IN)


def main():
    import numpy as np

    from axionfem import cavity as cv
    from axionfem import cavity_xsec as cx

    t0 = time.time()
    T = cv.sweep(build, [float(t) for t in range(0, 181, 6)], workdir=cm.WORK / "xsec_rod_tracking", refine=3,
                 min_step=0.75, probe_n=48, track_kw={"degenerate_rtol": 1e-5})
    print(f"{len(T.steps)} steps, {time.time() - t0:.0f} s")
    print(T.summary())
    b = T.branch("max_C")
    D = T.diabatic()
    db = next(x for x in D if any(p["step"] == 0 and p["mode"] == b.points[0]["mode"] for p in x.points))
    checks = []
    for th in (10.0, 30.0, 90.0, 150.0):
        p = min(b.points, key=lambda q: abs(q["value"] - th))
        d = 2 * P * math.sin(math.radians(p["value"]) / 2)
        k0 = 2 * math.pi * p["f_Hz"] / cm.C_LIGHT
        ref = cx.eccentric_rod_tm_mode(R, A, d, k_bracket=(0.97 * k0, 1.03 * k0))
        checks.append({"theta_deg": p["value"], "f_Hz": p["f_Hz"], "f_mps_Hz": ref["f_Hz"],
                       "rel": p["f_Hz"] / ref["f_Hz"] - 1, "C": p["C"], "C_mps": ref.get("C")})
        print(checks[-1])
    rec = {"description": "HAYSTAC one-rod cross-section (no end gaps), rod-angle sweep with mode tracking "
                          "(beta)", "summary": T.summary(), "crossings": T.crossings, "n_steps": len(T.steps),
           "tm010_branch": b.id,
           "tm010_adiabatic": T.branch_table([b]), "tm010_diabatic": T.branch_table([db]), "mps_checks": checks}
    cm.write_record("cavity_track_xsec_rod", rec)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    T.plot(ax=ax, xlabel="rod angle θ [deg]", highlight=(b.id,), vmax=0.8)
    x = [p["x"] for p in db.points]
    ax.plot(x, [p["f_Hz"] * 1e-9 for p in db.points], "--", color="C1", lw=1.2, label="TM010-like, diabatic")
    ax.plot([], [], "-", color="C3", lw=2.2, label="TM010-like, adiabatic")
    ax.plot([], [], "o", mfc="none", mec="C1", label="avoided crossing")
    ax.plot([], [], "x", color="C0", label="crossing")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("HAYSTAC one-rod cross-section: tracked modes (colour: C)", fontsize=9)
    fig.colorbar(ax.collections[0], ax=ax, label="C")
    fig.tight_layout()
    out = cm.ROOT / "docs" / "images" / "cavity_track_xsec_rod.png"
    fig.savefig(out, dpi=150)
    print(out)
    _ = np


if __name__ == "__main__":
    main()
