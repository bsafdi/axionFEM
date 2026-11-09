"""Infer the unpublished ADMX Run 1A/1B rod pivot radius r_p and arm length a.

Usage:  python3 -I admx_two_rod_infer.py [grid|refine [r_p_mm a_mm]|all|positions]
        ('all' ~10-40 min on one shared core; 'positions' needs admx_two_rod_digitized.json)

Method (INFERRED geometry, see admx_two_rod.json geometry.parameters.pivot_radius / arm_length):
the published simulated TM010-like frequencies at stated rod angles are compared with the ideal
2D z-uniform model of admx_two_rod_2d.py (R = 210 mm, rod radius 25.4 mm, no end gaps, no
armatures).  Targets (all from 3D simulations or text "roughly"):
  T1  both rods at the wall (0, 0)          ~580 MHz   2312.16668 p.5, 2010.00169 p.4 ("580 MHz to 890 MHz")
  T2  both rods at the centre (180, 180)    ~890 MHz   same sentence
  T3  rods at (180, 270)                     773 MHz   2010.00169 Fig. 5 top right (CST)
  T4  rods at (0, 120)                       656.4 MHz 2312.16668 Fig. 8a (COMSOL)
sigma: T1/T2 +/-10 MHz ("roughly"); T3/T4 +/-5 MHz (0.6-0.8%: 2D-vs-3D end gaps/armatures,
angle-label precision).  chi^2 on a (r_p, a) grid at h = 2 mm, then the best point is refined at
h = 1 and 0.5 mm (Richardson) and the model curves used in the overlays are written out.
Writes digitize/admx_two_rod_infer.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from admx_two_rod_2d import tm_modes, rod_centres, form_factor_2d  # noqa: E402

R, RROD = 0.210, 0.0254
TARGETS = [  # (label, theta_L, theta_R, sense_R, f_MHz, sigma_MHz, source)
    ("T1 both at wall", 0, 0, +1, 580.0, 10.0, "2312.16668 p.5 / 2010.00169 p.4 'roughly 580 MHz to 890 MHz'"),
    ("T2 both at centre", 180, 180, +1, 890.0, 10.0, "2312.16668 p.5 / 2010.00169 p.4"),
    ("T3 (180, 270)", 180, 270, +1, 773.0, 5.0, "2010.00169 Fig. 5 top right (CST)"),
    ("T4 (0, 120)", 0, 120, +1, 656.4, 5.0, "2312.16668 Fig. 8a (COMSOL)"),
]
OUT = os.path.join(HERE, "admx_two_rod_infer.json")


def valid(c):
    l, r = c
    return (np.hypot(*l) + RROD < R - 2e-4 and np.hypot(*r) + RROD < R - 2e-4
            and np.hypot(*(l - r)) > 2 * RROD + 2e-4)


def f0(rp, a, tl, tr, s, h):
    c = rod_centres(rp, a, tl, tr, s)
    if not valid(c):
        return None, None
    f, (X, Y, E), info = tm_modes(R, RROD, c, h, nev=3, f_guess=0.5e9)
    Cf, Cv = form_factor_2d(X, Y, E[0], h, R, RROD)
    return f[0] / 1e6, (Cf, Cv, [v / 1e6 for v in f])


def grid():
    rps = np.arange(112.0, 132.01, 2.0)
    As = np.arange(52.0, 66.01, 1.5)
    rows = []
    for rp in rps:
        for a in As:
            if rp + a + RROD * 1e3 > R * 1e3 - 0.2:
                continue
            fs = []
            for (_, tl, tr, s, ft, sg, _) in TARGETS:
                v, _ = f0(rp / 1e3, a / 1e3, tl, tr, s, 2e-3)
                fs.append(v)
            if any(v is None for v in fs):
                continue
            chi2 = sum(((v - t[4]) / t[5]) ** 2 for v, t in zip(fs, TARGETS))
            rows.append([rp, a, *fs, chi2])
            print(rp, a, ["%.1f" % v for v in fs], "chi2=%.2f" % chi2, flush=True)
    return rows


def fit_quadratic(rows):
    A = np.array(rows)
    rp, a, chi2 = A[:, 0], A[:, 1], A[:, -1]
    M = np.c_[np.ones_like(rp), rp, a, rp ** 2, rp * a, a ** 2]
    w = np.exp(-0.5 * (chi2 - chi2.min()) / 4.0)  # emphasise the minimum region
    c, *_ = np.linalg.lstsq(M * w[:, None], chi2 * w, rcond=None)
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    g = np.array([c[1], c[2]])
    x = np.linalg.solve(H, -g)
    cov = 2 * np.linalg.inv(H)  # delta chi2 = 1
    return x, cov, float(chi2.min())


def position_checks():
    """2D f at the rod positions MEASURED in the published field plots (no pivot model needed)."""
    dig = json.load(open(os.path.join(HERE, "admx_two_rod_digitized.json")))
    cases = []
    dd = dig["dd_fig5"]
    cen = dd["outline_pt"][:2]; mm = dd["mm_per_pt"]
    rods = [[(r[0] - cen[0]) * mm / 1e3, -(r[1] - cen[1]) * mm / 1e3] for r in dd["rods_pt"]]
    cases.append(("2010.00169 Fig. 5 (labels 180/270 deg)", 773.0, rods, "CST"))
    a8 = dig["run1a_fig8a"]
    cen = a8["outline_px"][:2]; mm = a8["mm_per_px"]
    rods = [[(r[0] - cen[0]) * mm / 1e3, -(r[1] - cen[1]) * mm / 1e3] for r in a8["rods_px"]]
    cases.append(("2312.16668 Fig. 8a (labels 120/0 deg)", 656.4, rods, "CST-style render; text: COMSOL"))
    for src in ("run1b_fig2", "run1b_analysis_fig16"):
        for pn in ("symmetric", "anti_symmetric"):
            for sc in ("outer_rim_edge_is_wall", "inner_rim_edge_is_wall"):
                g = dig[src]["panels"][pn][sc]["rods"]
                rods = [[q["rod_xy_mm"][0] / 1e3, q["rod_xy_mm"][1] / 1e3] for q in g]
                cases.append((f"{src} {pn} ({sc})", None, rods, "CST (frequency not published; sym = anti by caption)"))
    out = []
    for lab, ft, rods, code in cases:
        fh = {}
        for h in (1e-3, 0.5e-3):
            f, (X, Y, E), _ = tm_modes(R, RROD, [np.array(r) for r in rods], h, nev=3, f_guess=0.5e9)
            Cf, Cv = form_factor_2d(X, Y, E[0], h, R, RROD)
            fh[h] = (f[0] / 1e6, Cf, Cv)
            print(lab, h, fh[h], flush=True)
        f1, f2 = fh[1e-3][0], fh[0.5e-3][0]
        out.append({"label": lab, "rods_xy_mm": [[round(1e3 * v, 2) for v in r] for r in rods],
                    "published_f_MHz": ft, "published_by": code,
                    "f2D_MHz_h1_h05": [f1, f2], "f2D_MHz_richardson": f2 + (f2 - f1) / 3,
                    "C2D_full_disk": fh[0.5e-3][1], "C2D_vacuum_area": fh[0.5e-3][2]})
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if mode in ("positions",):
        res["position_checks"] = position_checks()
        json.dump(res, open(OUT, "w"), indent=1)
        return
    if mode in ("grid", "all"):
        res["grid_h_mm"] = 2.0
        res["grid_columns"] = ["r_p_mm", "a_mm"] + [t[0] + " f_MHz" for t in TARGETS] + ["chi2"]
        res["grid"] = grid()
        json.dump(res, open(OUT, "w"), indent=1)
    if mode in ("refine", "all"):
        x, cov, cmin = fit_quadratic(res["grid"])
        rp, a = float(x[0]), float(x[1])
        res["fit"] = {"r_p_mm": rp, "a_mm": a, "sigma_r_p_mm": float(np.sqrt(cov[0, 0])),
                      "sigma_a_mm": float(np.sqrt(cov[1, 1])),
                      "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                      "chi2_min_grid": cmin, "n_targets": len(TARGETS),
                      "method": "quadratic fit of chi2(r_p, a) on the h = 2 mm grid; 1-sigma from delta chi2 = 1"}
        # round to the reported values and refine with h-convergence
        rp_r, a_r = round(rp), round(a)
        basis = "rounded chi^2 fit"
        if len(sys.argv) >= 4:   # explicit reported values, e.g. combining the fit with the figure measurement
            rp_r, a_r = float(sys.argv[2]), float(sys.argv[3])
            basis = ("combined: chi^2 fit (r_p %.1f +/- %.1f, a %.1f +/- %.1f) and the Fig. 2/16 measurement "
                     "(outer-rim scale r_p 121.9, a 58.4 mm)" % (rp, res["fit"]["sigma_r_p_mm"], a, res["fit"]["sigma_a_mm"]))
        res["reported"] = {"r_p_mm": rp_r, "a_mm": a_r, "basis": basis}
        tab = []
        cfgs = [(t[0], t[1], t[2], t[3], t[4]) for t in TARGETS] + [
            ("Run 1B Fig 2 symmetric (116, 116)", 116.1, 116.2, +1, None),
            ("Run 1B Fig 2 anti-symmetric (130, 130 mirrored)", 130.3, 130.0, -1, None),
            ("2010.00169 Fig 3 both rods 116 deg (symmetric assumed)", 116, 116, +1, None)]
        for (lab, tl, tr, s, ft) in cfgs:
            out = {"label": lab, "theta_L": tl, "theta_R": tr, "sense_R": s, "target_MHz": ft}
            fh = {}
            for h in (2e-3, 1e-3, 0.5e-3):
                v, extra = f0(rp_r / 1e3, a_r / 1e3, tl, tr, s, h)
                fh[h] = (v, extra)
                print(lab, h, v, flush=True)
            f1, f2 = fh[1e-3][0], fh[0.5e-3][0]
            out["f_MHz_h2_h1_h05"] = [fh[2e-3][0], f1, f2]
            out["f_MHz_richardson"] = f2 + (f2 - f1) / 3.0
            out["next_modes_MHz_h05"] = fh[0.5e-3][1][2]
            out["C2D_full_disk_h05"] = fh[0.5e-3][1][0]
            out["C2D_vacuum_area_h05"] = fh[0.5e-3][1][1]
            tab.append(out)
        res["refined_table"] = tab
        # model curve f(theta2) with rod 1 at the wall, for the Fig. 8c overlay
        curve = []
        for t in range(60, 181, 5):
            v, _ = f0(rp_r / 1e3, a_r / 1e3, 0, t, +1, 1e-3)
            curve.append([t, v])
        res["model_curve_rod1_wall_h1mm"] = curve
        # symmetric and anti-symmetric tuning curves (both rods same angle)
        for key, s in (("model_curve_symmetric_h1mm", +1), ("model_curve_antisymmetric_h1mm", -1)):
            cc = []
            for t in range(0, 181, 10):
                v, extra = f0(rp_r / 1e3, a_r / 1e3, t, t, s, 1e-3)
                cc.append([t, v, None if extra is None else extra[0], None if extra is None else extra[1]])
            res[key] = cc
        res["curve_columns"] = "theta_deg, f_MHz (sym/anti: + C2D_full_disk, C2D_vacuum_area)"
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
