"""Off-centre tuning rod in the ADMX cavity cross-section: TM010-like mode, figure and field export
(cross-section solver, "symmetry": "xsec"; beta, docs/validation/cavity_xsec.md).

Geometry (benchmarks/cavity/refs/admx_two_rod.json): R = 210 mm, length 1010 mm, ONE copper rod of
50.8 mm diameter on an arm a = 57 mm about a pivot at (r_p, 0) = (122 mm, 0) (r_p, a INFERRED there), rod angle
theta = 120 deg from the radially outward direction: rod centre (r_p + a cos theta, a sin theta). Walls, rod and end caps
use the builtin `copper` (cold copper, anomalous skin effect, an upper bound on Q). Independent check: the method of
particular solutions (cavity_xsec.eccentric_rod_tm_mode) at the same rod offset.

Writes the record benchmarks/records/cavity_xsec_offcentre.json, and into the work directory a figure
(cavity_xsec_offcentre.{png,pdf}) and the field export cavity_xsec_offcentre_field.npz:
  x, y (1D grid axes, m), Ez (ny, nx; normalized int eps_r E_z^2 dA = 1 over the cross-section, m^-1; NaN in metal and
  outside), absEz, cavity_outline (n, 2), rod_outline (n, 2), wall_R_m, rod_center_m, rod_radius_m, f_Hz, C, C_full, Q0,
  Q0_2d, length_m, meta (JSON string).
Usage: python benchmarks/cavity/xsec_offcentre_field.py [workdir]
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from axionfem import cavity_xsec as cx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/xsec_offcentre")
R, ROD, RP, ARM, TH, LEN = 0.210, 0.0254, 0.122, 0.057, 120.0, 1.010


def main():
    c = (RP + ARM * math.cos(math.radians(TH)), ARM * math.sin(math.radians(TH)))
    regions = [{"name": "cavity", "shape": {"disk": {"center": [0.0, 0.0], "radius": R}}, "material": "vacuum"},
               {"name": "rod", "shape": {"disk": {"pivot": [RP, 0.0], "arm": ARM, "angle_deg": TH, "radius": ROD}},
                "material": "copper"}]
    d = cx.xsec_design(regions, name="admx_one_rod", wall_material="copper", length=LEN, modes=4, h=R / 70,
                       surface_size=R / 250)
    res = d.solve(WD, verbose=0)
    m = max((r for r in res.cavity_modes() if r["family"] == "TM"), key=lambda r: r["C"])
    ref = cx.eccentric_rod_tm_mode(R, ROD, math.hypot(*c))
    print({k: m[k] for k in ("f_Hz", "C", "C_full", "Q0", "Q0_2d")}, "MPS:", ref["f_Hz"], ref["C"], flush=True)
    n = 501
    P = res.cavity_mode_probe(m, x=(-R, R, n), y=(-R, R, n))
    t = np.linspace(0, 2 * np.pi, 721)
    img = WD
    meta = {"design": d.doc, "row": {k: m[k] for k in ("family", "index", "f_Hz", "C", "C_full", "Q0", "Q0_2d", "A_m2",
                                                       "A_full_m2", "V_m3", "G_ohm", "G_ohm_2d")},
            "normalization": "int eps_r E_z^2 dA = 1 over the cross-section (E_z in m^-1); sign: int E_z dA > 0",
            "mps_reference": {"f_Hz": ref["f_Hz"], "C": ref["C"]},
            "status": "beta; pivot radius and arm INFERRED (admx_two_rod.json); copper = builtin cold copper",
            "api": "axionfem.cavity_xsec: res = xsec_design(...).solve(); res.cavity_mode_probe(row, x=(-R, R, n), "
                   "y=(-R, R, n))['E_z']"}
    np.savez_compressed(img / "cavity_xsec_offcentre_field.npz", x=P["X"][0], y=P["Y"][:, 0], Ez=P["E_z"],
                        absEz=np.abs(P["E_z"]), cavity_outline=np.c_[R * np.cos(t), R * np.sin(t)],
                        rod_outline=np.c_[c[0] + ROD * np.cos(t), c[1] + ROD * np.sin(t)], wall_R_m=R,
                        rod_center_m=np.array(c), rod_radius_m=ROD, f_Hz=m["f_Hz"], C=m["C"], C_full=m["C_full"],
                        Q0=m["Q0"], Q0_2d=m["Q0_2d"], length_m=LEN, meta=np.array(json.dumps(meta, default=float)))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    res.plot_mode(m, ax=ax, quantity="absEz", subdiv=3, colorbar=False,
                  title=f"TM$_{{010}}$-like |E$_z$|:  f = {m['f_Hz'] / 1e6:.1f} MHz,  C = {m['C']:.3f}")
    ax.set_xlim(-R * 1.04, R * 1.04)
    ax.set_ylim(-R * 1.04, R * 1.04)
    ax.axis("off")
    ax.title.set_fontsize(12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(img / f"cavity_xsec_offcentre.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    rec = ROOT / "benchmarks" / "records" / "cavity_xsec_offcentre.json"
    rec.write_text(json.dumps(meta | {"design": d.name}, indent=1, default=float))


if __name__ == "__main__":
    main()
