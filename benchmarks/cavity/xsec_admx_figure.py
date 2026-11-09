"""ADMX-style two-rod cavity: |E_z| of the TM010-like mode on the cross-section
(cross-section solver, "symmetry": "xsec"; beta, docs/validation/cavity_xsec.md).

Geometry (benchmarks/cavity/refs/admx_two_rod.json): R = 210 mm, rods 50.8 mm diameter on arms
about pivots on one diameter; pivot radius r_p = 122 mm and arm a = 57 mm are INFERRED there (not published).
Rod centres: left (-r_p - a cos th_L, -a sin th_L), right (r_p + a cos th_R, s a sin th_R); s = +1 symmetric
(C2 symmetry), s = -1 anti-symmetric (mirror). Symmetric panel: both rods at 115.5 deg (1910.08638 Fig. 2); the
anti-symmetric angle is solved for so that the TM010-like frequency equals the symmetric one (the published caption:
"the frequency of the TM010 mode is the same in both configurations"; measured from the figure: ~129.6 deg).
Also checks the four published/2D-model frequencies T1-T4 of the reference file.

Writes docs/images/cavity_xsec_admx.png and
benchmarks/records/cavity_xsec_admx.json.  Usage: python benchmarks/cavity/xsec_admx_figure.py [workdir]
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from axionfem import cavity_xsec as cx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/xsec_admx")
R, ROD, RP, ARM = 0.210, 0.0254, 0.122, 0.057
H = R / 70


def rods(thL, thR, s):
    tl, tr = math.radians(thL), math.radians(thR)
    return [{"radius": ROD, "center": [-RP - ARM * math.cos(tl), -ARM * math.sin(tl)], "name": "rod_left"},
            {"radius": ROD, "center": [RP + ARM * math.cos(tr), s * ARM * math.sin(tr)], "name": "rod_right"}]


def solve(thL, thR, s, tag, modes=4):
    d = cx.cylinder_rods_design(R, rods(thL, thR, s), name=f"admx_{tag}", modes=modes, h=H, surface_size=H / 3)
    res = d.solve(WD / tag, verbose=0)
    tm = [r for r in res.cavity_modes() if r["family"] == "TM"]
    return res, max(tm, key=lambda r: r["C"])  # TM010-like: the largest C


def main():
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"geometry": {"R_m": R, "rod_radius_m": ROD, "pivot_radius_m": RP, "arm_m": ARM,
                        "status": "r_p, a INFERRED (admx_two_rod.json)"}, "checks": []}
    # published / independent 2D-model frequencies (admx_two_rod.json analytic_crosschecks)
    for lab, tl, trr, s, pub, fd in (("T1 both at wall", 0, 0, 1, 580.0, 583.0), ("T2 both at centre", 180, 180, 1, 890.0, 888.55),
                                     ("T3 (180, 270)", 180, 270, 1, 773.0, 773.86), ("T4 (0, 120)", 0, 120, 1, 656.4, 653.77)):
        _, m = solve(tl, trr, s, lab.split()[0])
        rec["checks"].append({"label": lab, "f_MHz": m["f_Hz"] / 1e6, "published_MHz": pub, "fd_model_MHz": fd,
                              "rel_vs_fd": m["f_Hz"] / 1e6 / fd - 1, "C": m["C"], "C_full": m["C_full"]})
        print(rec["checks"][-1], flush=True)
    rs, ms = solve(115.5, 115.5, 1, "sym")
    f0 = ms["f_Hz"]
    th = brentq(lambda t: solve(t, t, -1, f"anti_{t:.4f}")[1]["f_Hz"] - f0, 122.0, 137.0, xtol=1e-3)
    ra, ma = solve(th, th, -1, "anti")
    rec["symmetric"] = {"theta_deg": 115.5, "f_Hz": ms["f_Hz"], "C": ms["C"], "C_full": ms["C_full"]}
    rec["anti_symmetric"] = {"theta_deg": th, "f_Hz": ma["f_Hz"], "C": ma["C"], "C_full": ma["C_full"],
                             "published_figure_theta_deg": [129.9, 129.4]}
    print(rec["symmetric"], rec["anti_symmetric"], flush=True)

    img = ROOT / "docs" / "images"
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, res, m, lab in ((axs[0], rs, ms, f"symmetric, θ = 115.5°"), (axs[1], ra, ma, f"anti-symmetric, θ = {th:.1f}°")):
        res.plot_mode(m, ax=ax, quantity="absEz", subdiv=3, colorbar=False,
                      title=f"{lab}\nf = {m['f_Hz'] / 1e6:.1f} MHz, C = {m['C']:.3f} (C_full = {m['C_full']:.3f})")
        ax.set_xlim(-R * 1.03, R * 1.03); ax.set_ylim(-R * 1.03, R * 1.03); ax.axis("off")
    fig.suptitle("ADMX-style two-rod cavity, TM010-like |E_z| (2D cross-section; pivot geometry inferred; beta)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(img / "cavity_xsec_admx.png", dpi=170, facecolor="white")
    plt.close(fig)

    out = ROOT / "benchmarks" / "records" / "cavity_xsec_admx.json"
    out.write_text(json.dumps(rec, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
