"""Analysis of the HAYSTAC one-rod 3D sweep (haystac_rod_3d.py): record + figure. beta.

Loads the cached sweep steps (no solver), tracks them, and compares the TM010-like branch with 2006.01248:
- Fig. 3 lower line: df/dtheta over the isolated stretch 5-14 deg (linear fits to both); the offset at equal theta
  and the theta shift that removes it are reported but f(theta = 0) is NOT compared (refs README: not decidable);
- Fig. 6c / 6b (red triangles, 1 x 2 in rod): C and Q at equal frequency and at the CST angle read off the Fig. 3
  lower line (equal theta), with the spread over +-1 deg;
- the odd-in-z family (stage pmc) vs the Fig. 3 upper parallel line.
Output: benchmarks/records/cavity_track_haystac_rod_3d.json, docs/images/cavity_track_haystac_rod_3d.png.
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common3d as cm  # noqa: E402
import haystac_rod_3d as hs  # noqa: E402

V_FULL = math.pi * hs.R ** 2 * hs.H


def load(tag):
    from axionfem import cavity_track as ct

    steps = [ct.ModeStep.load(Path(p).parent) for p in glob.glob(str(cm.WORK / hs.BENCH / tag / "step_*" / "track_step.npz"))]
    return ct.track(steps, degenerate_rtol=1e-5) if steps else None


def tm010_branches(T):
    """Adiabatic branch through the largest-C mode at the first step, and its diabatic (resolved) continuation."""
    b = T.branch("max_C")
    db = T.branch("max_C", diabatic=True)
    maxc = [max(((k, i) for i in range(len(s.modes))), key=lambda ki: s.modes[ki[1]].get("C") or 0) for k, s in enumerate(T.steps)]
    return b, db, maxc


def at_f(points, f, key):
    """Value of `key` on a branch (list of point dicts) interpolated at frequency f (branch monotonic in f)."""
    pts = sorted(((p["f_Hz"], p[key]) for p in points if p.get(key) is not None))
    for (f0, y0), (f1, y1) in zip(pts, pts[1:]):
        if f0 <= f <= f1 and f1 > f0:
            return y0 + (y1 - y0) * (f - f0) / (f1 - f0)
    return None


def analyze(tag: str = "sweep"):
    import numpy as np

    T = load(tag)
    vols = [m.get("V_m3") for s in T.steps for m in s.modes[:1] if m.get("V_m3")]
    geometry_check = {"V_min_m3": min(vols), "V_max_m3": max(vols), "rel_spread": (max(vols) - min(vols)) / max(vols)}
    if geometry_check["rel_spread"] > 1e-5:  # every step has the same cavity volume (catches a solid not removed)
        raise RuntimeError(f"cavity volume differs between steps: {geometry_check}")
    ref = cm.ref("haystac_rod_2006")["reference_values"]["curves"]
    lower = ref[0]["series"][0]["points"]
    upper = ref[0]["series"][1]["points"]
    cloud = ref[1]["series"][0]["points"]
    fig6 = {k: next(c for c in ref if c["source"].endswith(k))["series"][0]["points"] for k in ("6b", "6c")}
    b, db, maxc = tm010_branches(T)
    rows = T.branch_table([db])
    V_vac = None
    for s in T.steps:
        for m in s.modes:
            if m.get("V_m3"):
                V_vac = 2 * m["V_m3"]  # z >= 0 half under mesh.symmetry
                break
        if V_vac:
            break
    # ---- df/dtheta over 5-14 deg
    lo_ref = [(t, f * 1e9) for t, f in lower if 5.0 <= t <= 14.0]
    lo_axf = [(p["x"], p["f_Hz"]) for p in db.points if 5.0 <= p["x"] <= 14.0]
    s_ref, b_ref, r_ref = cm.slope_fit(*zip(*lo_ref))
    s_axf, b_axf, r_axf = cm.slope_fit(*zip(*lo_axf))
    br = [(p["x"], p["f_Hz"]) for p in db.points]
    off = [cm.interp(br, t) - f for t, f in lo_ref if cm.interp(br, t) is not None]
    shift_deg = float(np.mean(off)) / (-s_axf)
    slope = {"window_deg": [5, 14], "ref_MHz_per_deg": s_ref / 1e6, "ref_rms_resid_MHz": r_ref / 1e6,
             "axf_MHz_per_deg": s_axf / 1e6, "axf_rms_resid_MHz": r_axf / 1e6,
             "rel_diff": (s_ref - s_axf) / s_axf, "n_ref": len(lo_ref), "n_axf": len(lo_axf),
             "mean_offset_axf_minus_ref_MHz": float(np.mean(off)) / 1e6,
             "theta_shift_equivalent_deg": shift_deg,
             "note": "f(theta = 0) not compared (refs README: not decidable from Fig. 3)"}
    # TM010-like = the largest-C mode of each step (equal to the tracked diabatic branch where both exist)
    mc = sorted(({"x": float(T.steps[k].value),
                  **{kk: T.steps[k].modes[i].get(kk) for kk in ("f_Hz", "C", "Q0")}} for k, i in maxc),
                key=lambda q: q["x"])
    # offset models on the lower line over 0-14.5 deg (the full isolated stretch) and 5-14 deg: a constant frequency
    # offset, a theta-origin shift (the refs noted that the Fig. 3 lines look straight down to theta ~ 0.5 deg), and
    # both. On 5-14 deg the branch is nearly linear, so the two cannot be told apart; the 0-5 deg points decide.
    def resid(th0, df, ref_pts, br_pts):
        d = [cm.interp(br_pts, t + th0) - df - f for t, f in ref_pts if cm.interp(br_pts, t + th0) is not None]
        return np.array(d)

    fits = {}
    for wname, (w0, w1) in (("0-14.5", (0.0, 14.5)), ("5-14", (5.0, 14.0))):
        pts = [(t, f * 1e9) for t, f in lower if w0 <= t <= w1]
        r0 = resid(0.0, 0.0, pts, br)
        dfc = float(np.mean(r0))
        rc = r0 - dfc
        th0s = np.linspace(-3, 3, 601)
        rms_t = [float(np.sqrt(np.mean(resid(t0, 0.0, pts, br) ** 2))) for t0 in th0s]
        t_best = float(th0s[int(np.argmin(rms_t))])
        best_j = min(((float(np.sqrt(np.mean((resid(t0, 0.0, pts, br) - np.mean(resid(t0, 0.0, pts, br))) ** 2))), t0,
                       float(np.mean(resid(t0, 0.0, pts, br)))) for t0 in th0s))
        fits[wname] = {"n": len(pts), "const_offset_MHz": dfc / 1e6, "const_rms_MHz": float(np.sqrt(np.mean(rc ** 2))) / 1e6,
                       "theta0_only_deg": t_best, "theta0_only_rms_MHz": min(rms_t) / 1e6,
                       "joint": {"theta0_deg": best_j[1], "offset_MHz": best_j[2] / 1e6, "rms_MHz": best_j[0] / 1e6}}
    slope["offset_fits"] = fits
    dfc = fits["0-14.5"]["const_offset_MHz"]
    # the odd-in-z family (PMC plane, stage pmc) vs the Fig. 3 upper parallel line. The odd mode is identified without
    # the reference: among the odd-family modes with E_azimuthal_fraction < 0.01 (meridional E, m = 0-like), the one
    # nearest above our own even TM010-like mode at that angle
    Tp = load("pmc")
    even = {round(float(q["x"]), 6): q["f_Hz"] for q in mc}
    if Tp is not None and Tp.steps:
        rows_up = []
        for st in Tp.steps:
            t = float(st.value)
            fu, fe = cm.interp(upper, t), even.get(round(t, 6))
            if fu is None or fe is None:
                continue
            cand = [m for m in st.modes if (m.get("E_azimuthal_fraction") is not None and m["E_azimuthal_fraction"] < 0.01
                                             and m["f_Hz"] > fe)]
            if not cand:
                continue
            m = min(cand, key=lambda q: q["f_Hz"])
            rows_up.append({"theta": t, "f_odd_GHz": m["f_Hz"] / 1e9, "f_even_GHz": fe / 1e9, "f_upper_GHz": fu,
                            "odd_minus_upper_MHz": (m["f_Hz"] - fu * 1e9) / 1e6,
                            "odd_minus_even_MHz": (m["f_Hz"] - fe) / 1e6, "Q0_odd": m.get("Q0")})
        slope["odd_family_vs_upper_line"] = rows_up
    # slope uncertainty of the reference: digitization +-0.004 GHz per point over a 9-deg baseline
    slope["ref_slope_uncertainty_MHz_per_deg"] = 2 * 4.0 / 9.0 / math.sqrt(max(len(lo_ref) / 2, 1))
    # ---- C, Q at the Fig. 6 points (red triangles); TM010-like = the largest-C mode of each step
    lower_f2t = sorted((f * 1e9, t) for t, f in lower)

    def theta_on_lower(f):
        return cm.interp(lower_f2t, f)

    fig_pts = []
    for (fc, C6), (fq, Q6) in zip(sorted(fig6["6c"]), sorted(fig6["6b"])):
        f = fc * 1e9
        th_ref = theta_on_lower(f)
        in_f = min(q["f_Hz"] for q in mc) <= f <= max(q["f_Hz"] for q in mc)
        if not in_f and (th_ref is None or not (mc[0]["x"] <= th_ref <= mc[-1]["x"])):
            continue
        row = {"f_GHz": fc, "C_ref": C6, "Q_ref": Q6,
               "equal_f": {"theta_axf_deg": at_f(mc, f, "x"), "C_vac": at_f(mc, f, "C"), "Q0": at_f(mc, f, "Q0")},
               "theta_ref_deg": th_ref}
        if th_ref is not None:  # at the CST angle, and at the CST angle + the fitted origin offset theta0
            for key, th in (("equal_theta", th_ref),):
                fv = cm.interp([(q["x"], q["f_Hz"]) for q in mc], th)
                if fv is None:
                    continue
                row[key] = {"theta_deg": th, "C_vac": cm.interp([(q["x"], q["C"]) for q in mc], th),
                            "Q0": cm.interp([(q["x"], q["Q0"]) for q in mc], th), "f_GHz": fv / 1e9}
        for kk in ("equal_f", "equal_theta"):
            if kk in row and row[kk].get("C_vac") is not None and V_vac:
                row[kk]["C_full"] = row[kk]["C_vac"] * V_vac / V_FULL
        fig_pts.append(row)
    # C and Q of the TM010-like mode at the Fig. 6 points vs the max-C series over the neighbouring 1 deg
    # (sensitivity of these crossing-cluster points)
    for row in fig_pts:
        th = row["equal_f"]["theta_axf_deg"] if row["equal_f"]["theta_axf_deg"] is not None else \
            row.get("equal_theta", {}).get("theta_deg")
        if th is None:
            continue
        near = [q for q in mc if abs(q["x"] - th) <= 1.0]
        row["equal_f"]["C_range_pm1deg"] = [min(q["C"] for q in near), max(q["C"] for q in near)]
        row["equal_f"]["Q0_range_pm1deg"] = [min(q["Q0"] for q in near), max(q["Q0"] for q in near)]
    # ---- crossings 0-32 deg
    events = [{k: e.get(k) for k in ("kind", "branches", "x0", "f_mid_Hz", "gap_min_Hz", "exchange", "stay",
                                     "overlaps", "unresolved", "resolution")}
              for e in T.crossings if db.id in e["branches"] or b.id in e["branches"]]
    rec = {"benchmark": "HAYSTAC one-rod 3D with alumina axles (2006.01248)", "status": "beta",
           "model": {"mesh": hs.PROD, "axle_eps_r": hs.AXLE_EPS, "sigma_S_per_m": hs.SIGMA, "symmetry": "z PEC plane",
                     "V_vac_m3": V_vac, "V_full_m3": V_FULL},
           "n_steps": len(T.steps), "geometry_check": geometry_check, "summary": T.summary(), "tm010_diabatic": rows,
           "tm010_adiabatic": T.branch_table([b]),
           "max_C_per_step": [{"theta": T.steps[k].value, **{kk: T.steps[k].modes[i].get(kk) for kk in ("f_Hz", "C", "Q0")}}
                              for k, i in maxc],
           "slope_5_14": slope, "fig6_points": fig_pts, "events_on_tm010": events}
    p = cm.write_record("cavity_track_haystac_rod_3d", rec)
    print(p)
    print(T.summary())
    print(slope)
    for q in fig_pts:
        print(q)
    _figure(T, b, db, mc, lower, upper, cloud, fig6, fig_pts)
    return rec


def _figure(T, b, db, mc, lower, upper, cloud, fig6, fig_pts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    cx = [(t, f) for t, f in cloud if t <= 34]
    ax.plot(*zip(*cx), ".", color="0.8", ms=2, label="Fig. 3 mode map (digitized)", zorder=1)
    ax.plot(*zip(*[(t, f) for t, f in lower if t <= 34]), "s", mfc="none", mec="k", ms=4,
            label="Fig. 3 TM010-like (lower line)", zorder=2)
    ax.plot(*zip(*upper), "^", mfc="none", mec="0.4", ms=4, label="Fig. 3 upper parallel line", zorder=2)
    T.plot(ax=ax, xlabel="rod angle θ [deg]", branches=[], vmax=0.8)
    ax.plot([q["x"] for q in mc], [q["f_Hz"] / 1e9 for q in mc], "-", color="C3", lw=1.6,
            label="axionFEM TM010-like (largest C)", zorder=5)
    ax.plot([p["x"] for p in db.points], [p["f_Hz"] / 1e9 for p in db.points], ":",
            color="C1", lw=1.4, label="axionFEM tracked branch (diabatic)", zorder=6)
    ax.set_xlim(-0.5, 33)
    ax.set_ylim(4.8, 6.05)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("HAYSTAC one-rod, 3D with alumina axles: mode map (dots: axionFEM, colour C)", fontsize=8)
    ax2.plot([q["f_Hz"] / 1e9 for q in mc], [q["C"] for q in mc], "-o", color="C3", ms=3,
             label="axionFEM C_vac, largest-C mode")
    ax2.plot(*zip(*fig6["6c"]), "^", color="k", ms=7, mfc="none", label="Fig. 6c (1 x 2 in, CST)")
    ax2.set_xlabel("f [GHz]")
    ax2.set_ylabel("C (V excludes the rod)")
    ax2.set_xlim(4.8, 6.0)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out = cm.ROOT / "docs" / "images" / "cavity_track_haystac_rod_3d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    analyze(*(sys.argv[1:2] or []))
