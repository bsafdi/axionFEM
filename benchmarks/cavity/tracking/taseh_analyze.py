"""Analysis of the TASEH single-rod 3D sweep (taseh_rod_3d.py): record + figure. beta.

Compares the tracked TM010-like branch with 2205.01477 Fig. 5 (HFSS, digitized in refs/taseh_rod_2205.json):
f(theta) (as predicted with theta = 0 = rod farthest from the axis, then with a fitted theta origin; refs README
tolerance: a few MHz after fitting the origin), Q0(theta) (sigma = 5.8e7 assumed), C(theta) (~2 %). The odd-in-z
family (stage pmc) is added to the mode map for the Fig. 5(a) comparison of the other modes.
Output: benchmarks/records/cavity_track_taseh_rod_3d.json, docs/images/cavity_track_taseh_rod_3d.png.
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common3d as cm  # noqa: E402
import taseh_rod_3d as ts  # noqa: E402


def load(tag):
    from axionfem import cavity_track as ct

    steps = [ct.ModeStep.load(Path(p).parent) for p in glob.glob(str(cm.WORK / ts.BENCH / tag / "step_*" / "track_step.npz"))]
    return ct.track(steps, degenerate_rtol=1e-5) if steps else None


def _tm010(T):
    b = T.branch("max_C")
    D = T.diabatic()
    db = next(x for x in D if any(p["step"] == b.points[0]["step"] and p["mode"] == b.points[0]["mode"] for p in x.points))
    return b, db


def _cmp(points, ref, key, scale=1.0, shift=0.0):
    """(theta, axf, ref) at the reference angles inside the branch range (axf interpolated at theta + shift)."""
    br = [(p["x"], p[key]) for p in points if p.get(key) is not None]
    out = []
    for t, y in ref:
        v = cm.interp(br, t + shift)
        if v is not None:
            out.append((t, v * scale, y))
    return out


def analyze():
    import numpy as np
    from scipy.optimize import minimize_scalar

    T = load("sweep")
    vols = [m.get("V_m3") for s in T.steps for m in s.modes[:1] if m.get("V_m3")]
    geometry_check = {"V_min_m3": min(vols), "V_max_m3": max(vols), "rel_spread": (max(vols) - min(vols)) / max(vols)}
    if geometry_check["rel_spread"] > 1e-5:  # every step has the same cavity volume (catches a solid not removed)
        raise RuntimeError(f"cavity volume differs between steps: {geometry_check}")
    refc = cm.ref("taseh_rod_2205")["reference_values"]["curves"]
    f_ref = [(t, f * 1e9) for t, f in refc[0]["series"][0]["points"]]
    other = refc[1]["series"][0]["points"]
    Q_ref = [(t, q * 1e4) for t, q in refc[2]["series"][0]["points"]]
    C_ref = refc[3]["series"][0]["points"]
    b, db = _tm010(T)
    res = {}
    # f: predicted (no fit), then with a fitted theta origin
    cf = _cmp(db.points, f_ref, "f_Hz")
    d = np.array([r - a for _, a, r in cf])
    res["f_predicted"] = {"n": len(cf), "mean_diff_MHz": d.mean() / 1e6, "rms_diff_MHz": float(np.sqrt((d ** 2).mean())) / 1e6,
                          "max_abs_diff_MHz": float(np.abs(d).max()) / 1e6,
                          "f0_ref_GHz": f_ref[0][1] / 1e9, "f0_axf_GHz": cf[0][1] / 1e9 if cf and cf[0][0] == 0 else None,
                          "f180_ref_GHz": f_ref[-1][1] / 1e9, "f180_axf_GHz": cf[-1][1] / 1e9 if cf and cf[-1][0] == 180 else None,
                          "tuning_range_ref_MHz": (f_ref[-1][1] - f_ref[0][1]) / 1e6,
                          "tuning_range_axf_MHz": (db.points[-1]["f_Hz"] - db.points[0]["f_Hz"]) / 1e6}

    def cost(sh):
        c = _cmp(db.points, f_ref, "f_Hz", shift=sh)
        dd = np.array([r - a for _, a, r in c])
        return float(((dd - dd.mean()) ** 2).mean()) if len(c) > 20 else 1e30

    opt = minimize_scalar(cost, bounds=(-15, 15), method="bounded")
    cfs = _cmp(db.points, f_ref, "f_Hz", shift=opt.x)
    ds = np.array([r - a for _, a, r in cfs])
    res["f_fit_theta_origin"] = {"theta_shift_deg": float(opt.x), "mean_offset_MHz": ds.mean() / 1e6,
                                 "rms_after_offset_MHz": float(np.sqrt(((ds - ds.mean()) ** 2).mean())) / 1e6,
                                 "max_abs_after_offset_MHz": float(np.abs(ds - ds.mean()).max()) / 1e6,
                                 "note": "fit of the theta origin (shift) with a free constant offset: shape of f(theta)"}
    slope_ref = cm.slope_fit(*zip(*[(t, f) for t, f in f_ref if 50 <= t <= 100]))[0] / 1e6
    slope_axf = cm.slope_fit(*zip(*[(p["x"], p["f_Hz"]) for p in db.points if 50 <= p["x"] <= 100]))[0] / 1e6
    res["slope_50_100_MHz_per_deg"] = {"ref": slope_ref, "axf": slope_axf, "text": "~2 MHz/degree (p. 5)"}
    for key, ref, nm in (("Q0", Q_ref, "Q0"), ("C", C_ref, "C")):
        c = _cmp(db.points, ref, key)
        rel = np.array([(r - a) / a for _, a, r in c])
        res[nm] = {"n": len(c), "mean_rel_ref_minus_axf": rel.mean(), "rms_rel": float(np.sqrt((rel ** 2).mean())),
                   "max_abs_rel": float(np.abs(rel).max()),
                   "axf_mean_0_180": float(np.mean([a for _, a, _ in c])), "ref_mean_0_180": float(np.mean([r for _, _, r in c])),
                   "trend_axf_180_over_0": c[-1][1] / c[0][1], "trend_ref_180_over_0": c[-1][2] / c[0][2]}
    pm = load("pmc")
    variants = {}
    for tag in ("variantB", "fillet", "noaxles"):
        Tv = load(tag)
        if Tv is None:
            continue
        rows = []
        for k, s in enumerate(Tv.steps):
            m = max(s.modes, key=lambda q: q.get("C") or 0)
            base = cm.interp([(p["x"], p["f_Hz"]) for p in db.points], float(s.value))
            baseC = cm.interp([(p["x"], p["C"]) for p in db.points], float(s.value))
            baseQ = cm.interp([(p["x"], p["Q0"]) for p in db.points], float(s.value))
            rows.append({"theta": s.value, "f_Hz": m["f_Hz"], "C": m["C"], "Q0": m["Q0"],
                         "df_MHz_vs_baseline": (m["f_Hz"] - base) / 1e6 if base else None,
                         "dC_rel_vs_baseline": m["C"] / baseC - 1 if baseC else None,
                         "dQ_rel_vs_baseline": m["Q0"] / baseQ - 1 if baseQ else None})
        variants[tag] = rows
    rec = {"benchmark": "TASEH single copper rod 3D (2205.01477, HFSS Fig. 5)", "status": "beta",
           "model": {"mesh": ts.PROD, "sigma_S_per_m": ts.SIGMA, "teflon_eps_r": ts.AXLE_EPS, "axle_radius_m": ts.AXLE_R,
                     "rod_ends": "baseline A (arm centrelines at +-57 mm)", "fillet": False, "symmetry": "z PEC plane"},
           "n_steps": len(T.steps), "geometry_check": geometry_check, "summary": T.summary(), "comparison": res, "variants": variants,
           "tm010_diabatic": T.branch_table([db]), "tm010_adiabatic": T.branch_table([b]),
           "pmc_modes": [{"theta": s.value, "f_Hz": [m["f_Hz"] for m in s.modes]} for s in pm.steps] if pm else None}
    print(cm.write_record("cavity_track_taseh_rod_3d", rec))
    print(T.summary())
    for k, v in res.items():
        print(k, v)
    for k, v in variants.items():
        print(k, v)
    _figure(T, pm, db, f_ref, other, Q_ref, C_ref, res)
    return rec


def _figure(T, pm, db, f_ref, other, Q_ref, C_ref, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axs[0]
    ax.plot(*zip(*other), ".", color="0.65", ms=3, label="Fig. 5(a) other modes (HFSS)")
    ax.plot([t for t, _ in f_ref], [f / 1e9 for _, f in f_ref], "D", mfc="none", mec="k", ms=4, label="Fig. 5(a) TM010 (HFSS)")
    T.plot(ax=ax, xlabel="", branches=[], vmax=0.7)
    if pm:
        for k, s in enumerate(pm.steps):
            ax.plot([s.value] * len(s.modes), [m["f_Hz"] / 1e9 for m in s.modes], "_", color="C0", ms=7,
                    label="axionFEM, odd in z (PMC plane)" if k == 0 else None)
    ax.plot([p["x"] for p in db.points], [p["f_Hz"] / 1e9 for p in db.points], "-", color="C3", lw=1.2,
            label="axionFEM TM010-like (tracked)")
    ax.set_ylim(4.5, 5.5)
    ax.set_ylabel("f [GHz]")
    ax.legend(fontsize=6.5, loc="upper left")
    sh = res["f_fit_theta_origin"]["theta_shift_deg"]
    ax.set_title(f"TASEH single rod, 3D (dots: axionFEM even-in-z modes, colour C); fitted origin shift {sh:+.1f}°",
                 fontsize=8)
    axs[1].plot([t for t, _ in Q_ref], [q / 1e4 for _, q in Q_ref], "D", mfc="none", mec="k", ms=4, label="Fig. 5(b) HFSS")
    axs[1].plot([p["x"] for p in db.points], [p["Q0"] / 1e4 for p in db.points], "-o", color="C3", ms=2.5,
                label="axionFEM (σ = 5.8e7 S/m, 293 K)")
    axs[1].set_ylabel("Q0 [1e4]")
    axs[1].legend(fontsize=7)
    axs[2].plot(*zip(*C_ref), "D", mfc="none", mec="k", ms=4, label="Fig. 5(c) HFSS")
    axs[2].plot([p["x"] for p in db.points], [p["C"] for p in db.points], "-o", color="C3", ms=2.5,
                label="axionFEM (V excludes the rod)")
    axs[2].set_ylabel("C")
    axs[2].set_xlabel("rod angle θ [deg]")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    out = cm.ROOT / "docs" / "images" / "cavity_track_taseh_rod_3d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    analyze()
