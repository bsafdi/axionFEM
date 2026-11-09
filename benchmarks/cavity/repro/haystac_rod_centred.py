"""Reproduction: HAYSTAC one-rod cavity, rod centred (theta = 0), 2006.01248 Figs. 3 and 6 (beta).

    python benchmarks/cavity/repro/haystac_rod_centred.py

Geometry (refs JSON): barrel R = 2 in, H = 10 in; copper rod r = 1 in, h_rod = 9.98 in, 0.010 in (0.254 mm) gaps at
both ends; at theta = 0 the rod is concentric. Axisymmetric approximation: the alumina axles (0.25 in OD at
r = 0.475 in, crossing both gaps) are omitted; refs README: +2-3 MHz systematic, target f(0) = 5.835 GHz (lower
Fig. 3 line, TM010-like), quoted 5.83-5.85 GHz with +-0.015 GHz. Fig. 6c: C_vac = 0.751 at 5.827 GHz.
C is reported with V = vacuum volume (rod excluded; 2006.01248 convention) and V = full barrel.
Walls: copper sigma = 5.8e7 S/m (CST value not stated; Q at theta = 0 is not a target).
Runs:
  - mesh study at the nominal gap: bulk h = 3, 2, 1.5 mm with the gap zone at gap/2, gap/4, gap/8 (order 2) and
    order 3 at the middle mesh;
  - gap sweep G = 0 (rod joined to both endcaps: closed-form coax 5.8666 GHz, C_vac 0.8062), 0.05, 0.1, 0.254,
    0.5, 1.0 mm (each end), and one asymmetric split (0.127 / 0.381 mm);
  - axle estimate: the axle footprint in each gap smeared into an annulus 8.89 < r < 15.24 mm with the
    area-weighted permittivity (gap field normal to the plates: eps_eff = 1 + 0.066 (9.8 - 1) = 1.58);
  - empty barrel TM010 (closed form 2.25871 GHz, C 0.69166, Q(5.8e7) 3.044e4; paper 3.07e4).
Reference analysis: f(theta) is even in theta (mirror symmetry of the pivot
geometry), so df/dtheta = 0 at the concentric position; the digitized Fig. 3 lines are straight down to their first
point. Quadratic fits f0 - a (theta + theta0)^2 over theta < 4, 6, 8, 11 deg give the origin offset theta0 and f0
needed; the angle of the Fig. 6c point (5.827 GHz) is read off the lower line. Verdict: f(theta = 0) is not decidable
from Fig. 3; one verdict per model (no axles / smeared axles) for f and C.
Writes benchmarks/records/cavity_repro_haystac_rod_centred.json and .png next to this script.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "haystac_rod_centred"
IN = 0.0254
R, H, A = 2 * IN, 10 * IN, 1 * IN
G0 = 0.010 * IN
SIGMA = 5.8e7
F_LOWER, F_UPPER, F_TOL = 5.835e9, 5.849e9, 0.015e9
C_FIG6C, F_FIG6C = 0.7513, 5.827e9
V_FULL = math.pi * R ** 2 * H
AXLE_EPS = 1 + (math.pi * (0.125 * IN) ** 2 / (math.pi * ((0.475 + 0.125) ** 2 - (0.475 - 0.125) ** 2) * IN ** 2)) * (9.8 - 1)


def coax_exact():
    from scipy.optimize import brentq
    from scipy.special import j0, j1, y0, y1

    b, a = R, A
    fn = lambda k: j0(k * b) * y0(k * a) - y0(k * b) * j0(k * a)  # noqa: E731
    k = brentq(fn, 1.5 / (b - a), 4.0 / (b - a))
    # C_vac for TM010 coax: E_z = Z0(kr) = J0(kr) Y0(ka) - Y0(kr) J0(ka)
    import numpy as np
    from scipy.integrate import quad
    Z = lambda r: j0(k * r) * y0(k * a) - y0(k * r) * j0(k * a)  # noqa: E731
    num = quad(lambda r: Z(r) * r, a, b, limit=200)[0] ** 2 * (2 * np.pi) ** 2 * H ** 2
    den = math.pi * (b ** 2 - a ** 2) * H * quad(lambda r: Z(r) ** 2 * r, a, b, limit=200)[0] * 2 * np.pi * H
    return {"f_Hz": cm.C_LIGHT * k / (2 * math.pi), "C_vac": num / den, "C_full": num / den * (b ** 2 - a ** 2) / b ** 2}


def fig3_analysis():
    """Fig. 3 lines near theta = 0: slopes, quadratic-origin fits, theta of the Fig. 6c point (refs JSON)."""
    import numpy as np
    from scipy.optimize import curve_fit

    ser = cm.ref("haystac_rod_2006")["reference_values"]["curves"][0]["series"]
    res = {"symmetry": "f(theta) is even in theta (mirror symmetry of the pivot geometry): df/dtheta = 0 at theta = 0",
           "lines": {}}
    for name, sr in (("lower", ser[0]), ("upper", ser[1])):
        P = np.array(sr["points"], float)
        fits = []
        for tmax in (4.0, 6.0, 8.0, 11.0):
            m = P[:, 0] < tmax
            (f0, a, t0), _ = curve_fit(lambda t, f0, a, t0: f0 - a * (t + t0) ** 2, P[m, 0], P[m, 1], p0=(5.86, 0.002, 2.0))
            fits.append({"theta_max_deg": tmax, "n_points": int(m.sum()), "f0_GHz": f0, "a_GHz_per_deg2": a,
                         "theta0_deg": t0})
        m = P[:, 0] <= 2.9
        res["lines"][name] = {"first_point": P[0].tolist(),
                              "slope_first_points_MHz_per_deg": 1e3 * float(np.polyfit(P[m, 0], P[m, 1], 1)[0]),
                              "quadratic_origin_fits": fits}
    P = np.array(ser[0]["points"], float)
    i = int(np.argmax(P[:, 1] < F_FIG6C / 1e9))
    (t1, f1), (t2, f2) = P[i - 1], P[i]
    res["theta_of_fig6c_point_deg"] = t1 + (f1 - F_FIG6C / 1e9) / (f1 - f2) * (t2 - t1)
    res["verdict"] = ("not decidable from Fig. 3: the lines show no flattening at theta -> 0, so f(0) = 5.835 is an "
                      "extrapolation with an uncertainty larger than 15 MHz; compare df/dtheta and C at theta ~ 5-20 deg "
                      "with the 3D model instead")
    return res


def build(tag, gtop, gbot, h, hg, order=2, axle=False):
    outline = [[0, -H / 2], [R, -H / 2], [R, H / 2], [0, H / 2]]
    z0, z1 = -H / 2 + gbot, H / 2 - gtop
    cond = [{"name": "rod", "shape": {"rect": {"r": [0, A], "z": [z0, z1]}}, "mesh_size": h}]
    # refinement (rod painted last, so the zones only refine vacuum): the gap disk at hg, a box of a few gaps around
    # the rod edge at hg, and a 3 mm box around the end region at max(g, hg) (graded towards the bulk h)
    ref = []
    for zs, g in ((H / 2, gtop), (-H / 2, gbot)):
        s = 1 if zs > 0 else -1
        e = "t" if s > 0 else "b"
        ref.append({"name": f"endzone{e}", "shape": {"rect": {"r": [0, A + max(3e-3, 6 * g)],
                                                             "z": sorted([zs, zs - s * max(g + 3e-3, 7 * g)])}},
                    "mesh_size": max(g, 2 * hg, 0.1e-3)})
        ref.append({"name": f"cornerzone{e}", "shape": {"rect": {"r": [A - 4 * g, A + 4 * g], "z": sorted([zs, zs - s * 5 * g])}},
                    "mesh_size": hg})
        ref.append({"name": f"gapzone{e}", "shape": {"rect": {"r": [0, A + 4 * g], "z": sorted([zs, zs - s * g])}},
                    "mesh_size": hg})
    diel = []
    if axle:
        for zs, g in ((H / 2, gtop), (-H / 2, gbot)):
            s = 1 if zs > 0 else -1
            diel.append({"name": f"axle{'t' if s > 0 else 'b'}", "eps_r": AXLE_EPS, "mesh_size": hg,
                         "shape": {"rect": {"r": [(0.475 - 0.125) * IN, (0.475 + 0.125) * IN], "z": sorted([zs, zs - s * g])}}})
    if gtop <= 0 and gbot <= 0:
        cond[0]["shape"]["rect"]["z"] = [-H / 2 - 1e-3, H / 2 + 1e-3]
        ref = [{"name": f"cornerzone{s}", "shape": {"rect": {"r": [A, A + 3e-3], "z": sorted([s * H / 2, s * (H / 2 - 3e-3)])}},
                "mesh_size": hg} for s in (1, -1)]
    return cm.design(tag, outline, h=h, order=order, modes=8, families=("TM",), wall=cm.skin(SIGMA, 300.0),
                     conductors=cond, refine=ref, dielectrics=diel, target_f_Hz=5.85e9,
                     interior_point=[(A + R) / 2, 0.0], t=5e-3, surface_size=h)


def tm010_like(data):
    ms = [m for m in cm.modes(data) if 5.0e9 < m["f_Hz"] < 6.5e9]
    return max(ms, key=lambda m: m["form_factor"]["C"])


def row(data):
    m = tm010_like(data)
    r = cm.mode_row(data, m, V_full=V_FULL)
    r["C_vac"] = r.pop("C")
    r["dofs"] = cm.dofs(data)
    r["others"] = sorted([{"f_Hz": x["f_Hz"], "C_vac": x["form_factor"]["C"]} for x in cm.modes(data)
                          if 5.5e9 < x["f_Hz"] < 6.2e9], key=lambda x: x["f_Hz"])
    return r


MESH = [(2, 3e-3, 2), (2, 2e-3, 4), (2, 1.5e-3, 8), (3, 2e-3, 4)]
PROD = (2, 1.5e-3, 8)
GAPS = [0.0, 0.05e-3, 0.1e-3, G0, 0.5e-3, 1.0e-3]


def main():
    jobs, keys = [], []
    for o, h, dv in MESH:
        tag = f"nominal_o{o}_h{h * 1e3:.1f}_g{dv}"
        jobs.append((tag, build(tag, G0, G0, h, G0 / dv, o)))
    o, h, dv = PROD
    for g in GAPS:
        if g == G0:
            continue
        hg = (g / dv) if g > 0 else G0 / dv
        tag = f"gap{g * 1e3:.3f}"
        jobs.append((tag, build(tag, g, g, h, hg, o)))
    jobs.append(("asym", build("asym", 0.127e-3, 0.381e-3, h, 0.127e-3 / dv, o)))
    jobs.append(("axle", build("axle", G0, G0, h, G0 / dv, o, axle=True)))
    jobs.append(("axle_o3", build("axle_o3", G0, G0, 2e-3, G0 / 4, 3, axle=True)))
    out = {tag: (data, dt) for tag, data, dt in cm.solve_many(jobs, NAME)}
    # empty barrel
    eb = []
    for oo, div in ((2, 8), (2, 16), (3, 8)):
        tag = f"empty_o{oo}_h{div}"
        d = cm.design(tag, [[0, -H / 2], [R, -H / 2], [R, H / 2], [0, H / 2]], h=R / div, order=oo, modes=2,
                      families=("TM",), wall=cm.skin(SIGMA, 300.0))
        (_, data, dt), = cm.solve_many([(tag, d)], NAME)
        m = min(cm.modes(data), key=lambda x: abs(x["f_Hz"] - 2.25871e9))
        eb.append({"order": oo, "h_over_R": 1 / div, "f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": m["losses"]["Q0"]})
    ex = coax_exact()
    study = []
    for o_, h_, dv_ in MESH:
        tag = f"nominal_o{o_}_h{h_ * 1e3:.1f}_g{dv_}"
        r = row(out[tag][0])
        r.update({"order": o_, "h_m": h_, "h_gap_m": G0 / dv_, "seconds": out[tag][1]})
        study.append(r)
    conv = study[-1]
    fin2 = [s for s in study if s["order"] == 2][-1]
    sweep = []
    for g in GAPS:
        tag = f"nominal_o{PROD[0]}_h{PROD[1] * 1e3:.1f}_g{PROD[2]}" if g == G0 else f"gap{g * 1e3:.3f}"
        r = row(out[tag][0])
        r["gap_m"] = g
        sweep.append(r)
    rec = {"benchmark": "HAYSTAC one-rod cavity, rod centred (theta = 0), 2006.01248 (axisymmetric approximation)",
           "assumptions": {"sigma_S_m": SIGMA, "axles": "omitted (main runs); smeared-annulus estimate in 'axle'",
                           "V_full_m3": V_FULL, "C_conventions": "C_vac: V = cavity volume without rod (2006.01248); "
                                                                 "C_full: V = pi R^2 H"},
           "coax_no_gap_exact": ex, "mesh_study": study, "converged": conv,
           "mesh_error_order2_finest_vs_order3": {"f_rel": fin2["f_Hz"] / conv["f_Hz"] - 1, "C_rel": fin2["C_vac"] / conv["C_vac"] - 1,
                                                  "Q_rel": fin2["Q0"] / conv["Q0"] - 1},
           "gap_sweep": sweep, "asymmetric_gap": row(out["asym"][0]), "axle_estimate": row(out["axle"][0]),
           "axle_estimate_o3": row(out["axle_o3"][0]), "axle_eps_eff": AXLE_EPS, "empty_barrel": eb,
           "empty_barrel_exact": {"f_Hz": cm.C_LIGHT * cm.CHI01 / (2 * math.pi * R), "C": 4 / cm.CHI01 ** 2,
                                  "Q0": cm.Z0 * cm.CHI01 * H / (2 * (R + H)) / cm.Rs_skin(cm.C_LIGHT * cm.CHI01 / (2 * math.pi * R), SIGMA),
                                  "Q_paper_predicted": 3.07e4, "Q_measured": 2.74e4}}
    fig3 = fig3_analysis()
    rec["fig3_reference_analysis"] = fig3
    ax3 = rec["axle_estimate_o3"]
    rec["comparison_per_model"] = {
        "no_axles": {"f_Hz": conv["f_Hz"], "df_vs_5.835_MHz_published_minus_FEM": (F_LOWER - conv["f_Hz"]) / 1e6,
                     "C_vac": conv["C_vac"], "C_vac_rel_fig6c_published_minus_FEM": C_FIG6C / conv["C_vac"] - 1},
        "smeared_axles": {"f_Hz": ax3["f_Hz"], "df_vs_5.835_MHz_published_minus_FEM": (F_LOWER - ax3["f_Hz"]) / 1e6,
                          "C_vac": ax3["C_vac"], "C_vac_rel_fig6c_published_minus_FEM": C_FIG6C / ax3["C_vac"] - 1,
                          "note": "first order only: exact for E normal to the plates in the gap; ignores the axle "
                                  "inside the endcap holes and inside the rod, and the steel spindle"},
        "fig6c_point_theta_deg": fig3["theta_of_fig6c_point_deg"],
        "verdict_f": fig3["verdict"],
        "verdict_C": (f"Fig. 6c point is at theta ~ {fig3['theta_of_fig6c_point_deg']:.1f} deg, not 0 (dC/dtheta = 0 at 0 by "
                      f"symmetry, so the angle effect is second order); C_vac (published - FEM)/FEM: no axles "
                      f"{100 * (C_FIG6C / conv['C_vac'] - 1):+.1f} %, smeared axles {100 * (C_FIG6C / ax3['C_vac'] - 1):+.1f} %: "
                      f"the two 2D models bracket the published value; 3D with the axles decides")}
    rec["comparison"] = {"f_lower_line_Hz": F_LOWER, "f_upper_line_Hz": F_UPPER, "tol_Hz": F_TOL,
                         "f_axionFEM_Hz": conv["f_Hz"], "df_vs_lower_MHz": (conv["f_Hz"] - F_LOWER) / 1e6,
                         "within_tol": abs(conv["f_Hz"] - F_LOWER) <= F_TOL,
                         "within_tol_note": "the 5.835 +- 0.015 target is itself not well defined (fig3_reference_analysis)",
                         "axle_shift_estimate_MHz": (rec["axle_estimate_o3"]["f_Hz"] - conv["f_Hz"]) / 1e6,
                         "C_fig6c_vac": C_FIG6C, "C_vac": conv["C_vac"], "C_full": conv["C_full"],
                         "C_vac_rel_fig6c": C_FIG6C / conv["C_vac"] - 1,
                         "Q0_sigma5.8e7": conv["Q0"], "Q_fig6b_theta0_not_a_target": 4637}
    cm.write_record(NAME, rec)
    plot(rec)
    import json
    print(json.dumps({k: rec[k] for k in ("coax_no_gap_exact", "comparison", "mesh_error_order2_finest_vs_order3")}, indent=0))
    for s in study:
        print(s["order"], s["h_m"], s["h_gap_m"], s["dofs"], s["f_Hz"] / 1e9, s["C_vac"], s["Q0"])
    for s in sweep:
        print("gap", s["gap_m"], s["f_Hz"] / 1e9, s["C_vac"], s["C_full"], s["Q0"], [(round(o["f_Hz"] / 1e9, 4), round(o["C_vac"], 4)) for o in s["others"]])
    print("asym", rec["asymmetric_gap"]["f_Hz"], rec["asymmetric_gap"]["C_vac"])
    print("axle", rec["axle_estimate"]["f_Hz"], rec["axle_estimate"]["C_vac"], rec["axle_estimate_o3"]["f_Hz"])
    print("empty", eb, rec["empty_barrel_exact"])


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    sw = rec["gap_sweep"]
    g = [s["gap_m"] * 1e3 for s in sw]
    ax[0].plot(g, [s["f_Hz"] / 1e9 for s in sw], "o-", label="axionFEM TM010-like (no axles)")
    for s in sw:
        for o in s["others"]:
            if o["C_vac"] < 0.1:
                ax[0].plot(s["gap_m"] * 1e3, o["f_Hz"] / 1e9, ".", color="0.6")
    ax[0].axhspan((F_LOWER - F_TOL) / 1e9, (F_LOWER + F_TOL) / 1e9, color="C2", alpha=0.15, label="target 5.835 +- 0.015")
    ax[0].axhline(F_LOWER / 1e9, color="C2", ls="--", lw=0.8)
    ax[0].axhline(F_UPPER / 1e9, color="C3", ls=":", lw=0.8, label="Fig. 3 upper line 5.849")
    ax[0].axhline(rec["coax_no_gap_exact"]["f_Hz"] / 1e9, color="k", ls=":", lw=0.8, label="coax, no gap (exact)")
    ax[0].axvline(0.254, color="0.5", lw=0.6)
    ax[0].plot([0.254], [rec["axle_estimate_o3"]["f_Hz"] / 1e9], "r*", ms=10, label="with smeared-axle estimate")
    ax[0].set_xlabel("rod-endcap gap at each end [mm]")
    ax[0].set_ylabel("f [GHz]")
    ax[0].set_ylim(5.78, 5.96)
    ax[0].set_title("theta = 0: f vs gap (grey: other m = 0 TM modes, C < 0.1)", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].plot(g, [s["C_vac"] for s in sw], "o-", label="C_vac (V without rod)")
    ax[1].plot(g, [s["C_full"] for s in sw], "s-", label="C_full (V = pi R^2 H)")
    ax[1].axhline(C_FIG6C, color="C0", ls="--", lw=0.8, label="Fig. 6c C_vac 0.751 (5.827 GHz)")
    ax[1].axhline(C_FIG6C * (V_FULL - math.pi * A ** 2 * 9.98 * IN) / V_FULL, color="C1", ls="--", lw=0.8,
                  label="same in full-V convention")
    ax[1].plot([0.254], [rec["axle_estimate_o3"]["C_vac"]], "r*", ms=10, label="C_vac with smeared-axle estimate")
    ax[1].axvline(0.254, color="0.5", lw=0.6)
    ax[1].set_xlabel("gap [mm]")
    ax[1].set_ylabel("C")
    ax[1].set_title("form factor vs gap")
    ax[1].legend(fontsize=7)
    st = rec["mesh_study"]
    c = rec["converged"]
    s2 = [s for s in st if s["order"] == 2]
    ax[2].loglog([s["h_gap_m"] * 1e3 for s in s2], [abs(s["f_Hz"] / c["f_Hz"] - 1) for s in s2], "o-", label="f")
    ax[2].loglog([s["h_gap_m"] * 1e3 for s in s2], [abs(s["C_vac"] / c["C_vac"] - 1) for s in s2], "s-", label="C_vac")
    ax[2].loglog([s["h_gap_m"] * 1e3 for s in s2], [abs(s["Q0"] / c["Q0"] - 1) for s in s2], "^-", label="Q0")
    ax[2].set_xlabel("gap-zone element size [mm] (bulk 3, 2, 1.5 mm; order 2)")
    ax[2].set_ylabel("relative change vs order 3")
    ax[2].set_title("mesh study, nominal gap 0.254 mm")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"HAYSTAC one-rod, theta = 0, axisymmetric: f = {c['f_Hz'] / 1e9:.4f} GHz (Fig. 3 lower 5.835), "
                 f"C_vac = {c['C_vac']:.3f}, C_full = {c['C_full']:.3f} (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
