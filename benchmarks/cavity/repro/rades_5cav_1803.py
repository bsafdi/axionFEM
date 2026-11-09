"""3D: RADES 5-subcavity WR-90 filter, 1803.01243 Tables 1-2 (= 2107.03137 Table II). beta.

    python benchmarks/cavity/repro/rades_5cav_1803.py [--stage test|main|plot]

Geometry (2 K design column of Table 1, p. 8 of the arXiv PDF): five WR-90 sub-cavities a x b = 22.86 x 10.16 mm,
lengths 26.68 / 25.00 / 25.00 / 25.00 / 26.68 mm, joined by four inductive irises (walls t = 2.00 mm, full-height
centred apertures W = 8.00 mm). CST eigenmodes (Table 2, p. 9): f, G^2 (= C, Eq. A.5 with V = sum of the cavity
volumes, Eq. A.21), Q_u at sigma = 2e9 S/m (text p. 9). B0 along y (the narrow side b, parallel to the TE101 E).
Not given: corner radius ("corners are rounded to facilitate machining", p. 7) -> swept 0-1.5 mm on the four
y-parallel edges of every sub-cavity pocket; iris centring (assumed); probes (not in the eigenmode model).

Model: closed PEC domain = union of the five pockets and four apertures; 1/8 model with mirror planes x = 0 (PMC),
y = 0 (PEC) and z = 0 (PMC: z-even modes 1, 3, 5; PEC: z-odd modes 2, 4); the full model (no mirror planes) at one
mesh as a cross-check of the symmetry handling. Walls: classical skin effect, sigma = 2e9 S/m (rf `skin`).
Refinement: vacuum boxes (intersected with the domain) around each iris with element size h/3.
Writes benchmarks/records/cavity_repro_rades_5cav_1803.json and benchmarks/cavity/repro/rades_5cav_1803.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "rades_5cav_1803"
MM = 1e-3
A, B = 22.86 * MM, 10.16 * MM
LS = [26.68 * MM, 25.0 * MM, 25.0 * MM, 25.0 * MM, 26.68 * MM]
W, T = 8.0 * MM, 2.0 * MM
SIGMA = 2e9
FT = 8.55e9


def layout():
    """z intervals of the 5 pockets and the 4 irises (centred on z = 0)."""
    tot = sum(LS) + 4 * T
    z = -tot / 2
    cav, iris = [], []
    for i, L in enumerate(LS):
        cav.append((z, z + L))
        z += L
        if i < 4:
            iris.append((z, z + T))
            z += T
    return cav, iris, tot


def domain(r_corner: float = 0.0, W_ap: float = W):
    cav, iris, _ = layout()
    parts = [cm.rounded_box_xz(-A / 2, A / 2, -B / 2, B / 2, z0, z1, r_corner) for z0, z1 in cav]
    parts += [{"box": {"min": [-W_ap / 2, -B / 2, z0], "max": [W_ap / 2, B / 2, z1]}} for z0, z1 in iris]
    return {"union": parts}


def volumes(r_corner: float = 0.0, W_ap: float = W):
    cav, iris, tot = layout()
    corner = (4 - math.pi) * r_corner ** 2 * B  # four rounded edges per pocket
    v_cav = sum((z1 - z0) * A * B - corner for z0, z1 in cav)
    v_ap = sum((z1 - z0) * W_ap * B for z0, z1 in iris)
    return v_cav, v_ap


def build(tag, *, h, order=2, zsym="pmc", r_corner=0.0, full=False, modes=6, refine_factor=3.0, W_ap=W):
    cav, iris, tot = layout()
    dom = domain(r_corner, W_ap)
    m = 2.5 * MM
    hr = h / refine_factor
    refine = []
    for i, (z0, z1) in enumerate(iris):
        box = {"box": {"min": [-W_ap / 2 - m, -B / 2, z0 - m], "max": [W_ap / 2 + m, B / 2, z1 + m]}}
        refine.append({"name": f"iris{i}_ref", "solid": {"intersection": [box, dom]}, "mesh_size": hr})
    zc = [0.5 * (z0 + z1) for z0, z1 in cav]
    if full:
        probes = [[0.3 * MM, 0.3 * MM, z] for z in zc]
        sym = None
        ip = (3.1 * MM, 2.3 * MM, 1.7 * MM)
    else:
        probes = [[0.3 * MM, 0.3 * MM, max(z, 0.5 * MM)] for z in zc[2:]]
        sym = {"x": "pmc", "y": "pec", "z": zsym}
        ip = (3.1 * MM, 2.3 * MM, 1.7 * MM)
    return cm.design3d(tag, dom, h=h, order=order, modes=modes, target_f_Hz=FT, wall=cm.rf_skin(SIGMA, 2.0),
                       B_T=(0.0, 1.0, 0.0), symmetry=sym, refine=refine, interior_point=ip, probes=probes)


def pattern(mode, full: bool, zsym: str) -> str:
    """Signs of E_y at the pocket centres (normalised so that pocket 2 is '+'); 1/8 models mirror z >= 0."""
    pr = mode.get("probes", {})
    E = pr.get("E") if isinstance(pr, dict) else None
    if E is None:
        return "?"
    ey = [e[1] for e in E]
    if not full:
        par = 1.0 if zsym == "pmc" else -1.0
        ey = [par * ey[2], par * ey[1], ey[0], ey[1], ey[2]]
        if zsym == "pec":
            ey[2] = 0.0
    big = max(abs(v) for v in ey) or 1.0
    s = ["0" if abs(v) < 0.05 * big else ("+" if v > 0 else "-") for v in ey]
    if s[1] == "-":
        s = [{"+": "-", "-": "+"}.get(c, c) for c in s]
    return " ".join(s)


def rows(data, full, zsym, r_corner=0.0, W_ap=W):
    v_cav, v_ap = volumes(r_corner, W_ap)
    out = []
    for m in cm.modes3d(data):
        ff, lo = m.get("form_factor", {}), m.get("losses", {})
        C = ff.get("C")
        out.append({"f_Hz": m["f_Hz"], "C_full_volume": C,
                    "C_pockets": None if C is None else C * (v_cav + v_ap) / v_cav,
                    "Q0": lo.get("Q0"), "G_ohm": lo.get("G_ohm"), "residual_true": m.get("residual_true"),
                    "pattern": pattern(m, full, zsym),
                    "Q0_pointwise": (lo["Q0"] * lo["walls"]["outer"]["G_ohm_pointwise"] / lo["walls"]["outer"]["G_ohm"])
                    if lo.get("Q0") and isinstance(lo.get("walls"), dict) and "outer" in lo["walls"] else None,
                    "disagreement": {k: v.get("G_flux_pointwise_disagreement") for k, v in lo.get("walls", {}).items()}
                    if isinstance(lo.get("walls"), dict) else None})
    return out


def jobs_main():
    jobs = []
    # mesh study (order 2), both parities
    for h in (3.0, 2.0, 1.4, 1.0):
        for zs in ("pmc", "pec"):
            jobs.append((f"o2_h{h}_{zs}", build(f"o2_h{h}_{zs}", h=h * MM, zsym=zs), dict(h=h, order=2, zsym=zs)))
    for zs in ("pmc", "pec"):
        jobs.append((f"o3_h1.4_{zs}", build(f"o3_h1.4_{zs}", h=1.4 * MM, order=3, zsym=zs),
                     dict(h=1.4, order=3, zsym=zs)))
    # full model (no mirror planes) at the production mesh
    jobs.append(("full_o2_h2.0", build("full_o2_h2.0", h=2.0 * MM, full=True, modes=10), dict(h=2.0, order=2, full=True)))
    # corner radius sweep at the production mesh (order 2, h = 1.4 mm)
    for r in (0.5, 1.0, 1.5):
        for zs in ("pmc", "pec"):
            jobs.append((f"corner{r}_{zs}", build(f"corner{r}_{zs}", h=1.4 * MM, zsym=zs, r_corner=r * MM),
                         dict(h=1.4, order=2, zsym=zs, r_corner=r)))
    # wall-loss convergence: stronger refinement at the iris edges (re-entrant 270-degree edges), order 3
    for h, rf in ((1.4, 6.0), (1.0, 4.0)):  # (1.0, 6.0): ~5.5e6 DOFs per parity, out of memory on one 128 GB node
        for zs in ("pmc", "pec"):
            tg = f"o3_h{h}_rf{rf:g}_{zs}"
            jobs.append((tg, build(tg, h=h * MM, order=3, zsym=zs, refine_factor=rf),
                         dict(h=h, order=3, zsym=zs, refine_factor=rf)))
    # 298 K aperture W (8.08 / 8.14 mm) is a room-temperature question: not part of the 2 K benchmark
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    a = ap.parse_args()
    if a.stage == "xsec":
        stage_xsec()
        return
    if a.stage == "record":
        record(json.loads((cm.WORK / NAME / "summary.json").read_text()))
        return
    if a.stage == "test":
        d = build("test", h=3.0 * MM, zsym="pmc")
        out = cm.solve_many3d([("test", d)], NAME)
        for r in rows(out[0][1], False, "pmc"):
            print(r)
        return
    jobs = jobs_main()
    out = cm.solve_many3d([(t, d) for t, d, _ in jobs], NAME)
    res = []
    for (tag, data, dt), (_, _, meta) in zip(out, jobs):
        full = meta.get("full", False)
        zs = meta.get("zsym", "pmc")
        res.append({"tag": tag, **meta, "dofs": cm.dofs(data) if "mesh" in data else None, "seconds": dt,
                    "converged": data["fullwave"]["eigenmodes"].get("converged"),
                    "modes": rows(data, full, zs, meta.get("r_corner", 0.0) * MM)})
    (cm.WORK / NAME / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    record(res)


def five(r_pmc, r_pec):
    """The five filter modes (8.3-8.8 GHz) of one mesh from its z-even and z-odd runs, in frequency order."""
    ms = [m for m in r_pmc["modes"] + r_pec["modes"] if 8.3e9 < m["f_Hz"] < 8.8e9]
    if len(ms) != 5:
        raise RuntimeError(f"expected 5 filter modes in 8.3-8.8 GHz, found {len(ms)}")
    return sorted(ms, key=lambda m: m["f_Hz"])


# ---- exact 2D reference --------------------------------
# The filter is uniform along y (full-height irises), so its TE10-family modes E_y(x, z) are the TM_z modes of the
# cross-section solver on the (x, z) plane with end caps at y = +-b/2 (cavity.length = b): f, C and Q (including the
# top and bottom walls) are exact 2D quantities, and the 16 re-entrant iris edges can be refined to 25 um.

def xsec_build(hb, hi, order, m=1.5 * MM):
    from axionfem.cavity_xsec import xsec_design, skin_rf

    def rect(x0, x1, y0, y1):
        return {"rect": {"x": [x0, x1], "y": [y0, y1]}}
    cav, iris, _ = layout()
    parts = [rect(-A / 2, A / 2, z0, z1) for z0, z1 in cav] + [rect(-W / 2, W / 2, z0, z1) for z0, z1 in iris]
    regions = [{"name": "cavity", "shape": {"union": parts}, "material": "vacuum", "mesh_size": hb,
                "surface_size": hb / 2}]
    for i, (z0, z1) in enumerate(iris):
        sh = {"union": [rect(-W / 2 - m, W / 2 + m, z0 - m, z0), rect(-W / 2, W / 2, z0, z1),
                        rect(-W / 2 - m, W / 2 + m, z1, z1 + m)]}
        regions.append({"name": f"iris{i}", "shape": sh, "material": "vacuum", "mesh_size": hi, "surface_size": hi})
    mats = {"wall_cu": {"kind": "conductor", "rf": skin_rf(SIGMA, 2.0)}}
    return xsec_design(regions, name="rades18_xsec", materials=mats, wall_material="wall_cu", length=B, modes=8,
                       families=("TM",), target_f_Hz=FT, interior_point=(3.1 * MM, 1.7 * MM), h=hb,
                       surface_size=hb / 2, order=order)


def stage_xsec():
    """Iris-edge refinement series of the exact 2D model; G_flux and G_pointwise (which converge from opposite sides
    at the re-entrant edges) are extrapolated separately in the edge size; Q0 from their mean, bracket from both."""
    runs = []
    for hi, order in ((0.2, 2), (0.1, 2), (0.05, 2), (0.025, 2), (0.1, 3), (0.05, 3)):
        tag = f"xsec_hb1_hi{hi}_o{order}"
        out = cm.solve_many([(tag, xsec_build(1.0 * MM, hi * MM, order))], NAME)
        data = out[0][1]
        ms = []
        for m in sorted(data["fullwave"]["eigenmodes"]["TM"]["modes"], key=lambda m: m["f_Hz"]):
            if not 8.3e9 < m["f_Hz"] < 8.8e9:
                continue
            lo = m["losses"]
            w = lo["walls"]["wall"]
            ms.append({"f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": lo["Q0"], "G_wall": w["G_ohm"],
                       "G_wall_pointwise": w["G_ohm_pointwise"], "Rs_ohm": w["Rs_ohm"],
                       "Q_end_caps": lo["end_caps"]["Q"]})
        if len(ms) != 5:
            raise RuntimeError(f"{tag}: expected 5 modes, found {len(ms)}")
        runs.append({"tag": tag, "hi_mm": hi, "order": order, "dofs": cm.dofs(data), "modes": ms})
    import numpy as np
    o2 = sorted([r for r in runs if r["order"] == 2], key=lambda r: -r["hi_mm"])
    conv = []
    for k in range(5):
        # joint fit of the two estimators to one limit: G_flux = G_inf + a h^p, G_pointwise = G_inf + b h^p (they
        # converge from opposite sides at the re-entrant edges); p = 1/3 (the H ~ r^(-1/3) edge singularity) is the
        # central value, p = 2/3 a variant; the spread over p and the fit residual give the uncertainty.
        fits = {}
        for p in (1.0 / 3.0, 2.0 / 3.0):
            A_, y = [], []
            for r in o2:
                hp = r["hi_mm"] ** p
                A_.append([1.0, hp, 0.0])
                y.append(r["modes"][k]["G_wall"])
                A_.append([1.0, 0.0, hp])
                y.append(r["modes"][k]["G_wall_pointwise"])
            sol, res, *_ = np.linalg.lstsq(np.array(A_), np.array(y), rcond=None)
            rms = float(np.sqrt(np.mean((np.array(A_) @ sol - np.array(y)) ** 2)))
            fits[f"p={p:.3f}"] = {"G_inf": float(sol[0]), "a": float(sol[1]), "b": float(sol[2]), "rms_ohm": rms}
        fin = o2[-1]["modes"][k]
        Rs, Qe = fin["Rs_ohm"], fin["Q_end_caps"]
        Qs = {kk: 1 / (Rs / v["G_inf"] + 1 / Qe) for kk, v in fits.items()}
        o3 = [r for r in runs if r["order"] == 3][-1]["modes"][k]
        conv.append({"f_Hz": o3["f_Hz"], "C": o3["C"], "G_fits": fits, "Q0": Qs["p=0.333"],
                     "Q0_bracket": sorted(Qs.values()), "Q0_finest_flux": fin["Q0"],
                     "Q0_finest_pointwise": 1 / (Rs / fin["G_wall_pointwise"] + 1 / Qe)})
    out = {"method": "exact 2D (xsec) model of the y-uniform filter, iris-edge refinement 0.2-0.025 mm (order 2) "
                     "and order 3; f, C from order 3 at 0.05 mm; Q0 from a joint h^p fit of the flux and pointwise G to one "
                     "limit (p = 1/3 central, 2/3 variant)",
           "runs": runs, "converged": conv}
    (cm.WORK / NAME / "xsec_summary.json").write_text(json.dumps(out, indent=1, default=float))
    for c in conv:
        print(c["f_Hz"], c["C"], c["Q0"], c["Q0_bracket"])
    return out


def record(res):
    rj = cm.ref("rades_5cav_1803")
    tab = rj["reference_values"]["tables"][0]["rows"]
    meas = rj["measured_values"]["tables"][0]["rows"]
    by = {r["tag"]: r for r in res}

    def pair(prefix):
        return five(by[prefix + "_pmc"], by[prefix + "_pec"])
    meshes = [("o2_h3.0", 2, 3.0, 3), ("o2_h2.0", 2, 2.0, 3), ("o2_h1.4", 2, 1.4, 3), ("o2_h1.0", 2, 1.0, 3),
              ("o3_h1.4", 3, 1.4, 3), ("o3_h1.4_rf6", 3, 1.4, 6), ("o3_h1.0_rf4", 3, 1.0, 4)]
    study = []
    for tag, order, h, rf in meshes:
        if tag + "_pmc" not in by:
            raise RuntimeError(f"mesh-study run {tag} missing from the summary")
        ms = pair(tag)
        study.append({"tag": tag, "order": order, "h_mm": h, "iris_refine_factor": rf,
                      "dofs_per_parity": by[tag + "_pmc"]["dofs"],
                      "f_GHz": [m["f_Hz"] / 1e9 for m in ms], "C": [m["C_full_volume"] for m in ms],
                      "Q0": [m["Q0"] for m in ms], "Q0_pointwise": [m["Q0_pointwise"] for m in ms],
                      "pattern": [m["pattern"] for m in ms],
                      "G_flux_pointwise_disagreement": [m["disagreement"]["outer"] for m in ms]})
    best = study[-1]
    prev = study[-2]
    full = [m for m in by["full_o2_h2.0"]["modes"] if 8.3e9 < m["f_Hz"] < 8.8e9]
    o2h2 = pair("o2_h2.0")
    corners = {}
    for r in (0.5, 1.0, 1.5):
        ms = pair(f"corner{r}")
        base = pair("o2_h1.4")
        corners[f"{r}mm"] = {"df_MHz": [(a["f_Hz"] - b["f_Hz"]) / 1e6 for a, b in zip(ms, base)],
                             "dC_rel": [(a["C_full_volume"] / b["C_full_volume"] - 1) if b["C_full_volume"] else None
                                        for a, b in zip(ms, base)],
                             "dQ_rel": [a["Q0"] / b["Q0"] - 1 for a, b in zip(ms, base)]}
    v_cav, v_ap = volumes()
    comp = []
    for i, row in enumerate(tab):
        f_t, C_t, Q_t = float(row[2]) * 1e9, float(row[3]), float(row[4])
        f, C, Q = best["f_GHz"][i] * 1e9, best["C"][i], best["Q0"][i]
        comp.append({"mode": row[0], "pattern_paper": row[1], "pattern_axionFEM": best["pattern"][i],
                     "f_paper_GHz": f_t / 1e9, "f_axionFEM_GHz": f / 1e9, "df_MHz": (f_t - f) / 1e6,
                     "f_rel": f_t / f - 1, "f_mesh_change_MHz": (best["f_GHz"][i] - prev["f_GHz"][i]) * 1e3,
                     "C_paper": C_t, "C_axionFEM_full_volume": C, "C_axionFEM_pocket_volume": C * (v_cav + v_ap) / v_cav,
                     "Q_paper": Q_t, "Q_axionFEM": Q, "Q_rel": Q_t / Q - 1,
                     "Q_axionFEM_pointwise": best["Q0_pointwise"][i], "Q_rel_vs_pointwise": Q_t / best["Q0_pointwise"][i] - 1,
                     "Q_mesh_change_rel": best["Q0"][i] / prev["Q0"][i] - 1,
                     "f_measured_2K_GHz": float(meas[i][2])})
    spread = {"paper_GHz": [float(r[2]) - float(tab[0][2]) for r in tab],
              "axionFEM_GHz": [f - best["f_GHz"][0] for f in best["f_GHz"]]}
    rec = {"benchmark": "RADES 5-subcavity WR-90 filter, 1803.01243 Table 2 (= 2107.03137 Table II), 3D",
           "model": __doc__.split("Model: ")[1].strip(),
           "assumptions": {"sigma_S_m": SIGMA, "rf_model": "skin", "B0": "uniform, y", "corners": "sharp (swept)",
                           "irises": "centred", "probes": "none"},
           "volumes_m3": {"pockets": v_cav, "apertures": v_ap},
           "mesh_study": study, "converged": best["tag"], "comparison": comp, "mode_spread_vs_mode1": spread,
           "full_model_vs_eighth_model_h2": {"f_GHz": [m["f_Hz"] / 1e9 for m in sorted(full, key=lambda m: m["f_Hz"])],
                                             "eighth_f_GHz": [m["f_Hz"] / 1e9 for m in o2h2],
                                             "C_full_model": [m["C_full_volume"] for m in sorted(full, key=lambda m: m["f_Hz"])],
                                             "C_eighth": [m["C_full_volume"] for m in o2h2],
                                             "Q_full_model": [m["Q0"] for m in sorted(full, key=lambda m: m["f_Hz"])],
                                             "Q_eighth": [m["Q0"] for m in o2h2]},
           "corner_radius_sweep_vs_sharp_o2_h1.4": corners}
    xs = cm.WORK / NAME / "xsec_summary.json"
    if xs.exists():
        rec["xsec_reference"] = json.loads(xs.read_text())
        ref = rec["xsec_reference"]["converged"]
        for c, x in zip(comp, ref):
            c.update({"f_xsec_GHz": x["f_Hz"] / 1e9, "df_vs_xsec_MHz": (c["f_paper_GHz"] * 1e9 - x["f_Hz"]) / 1e6,
                      "C_xsec": x["C"], "Q_xsec": x["Q0"], "Q_xsec_bracket": x["Q0_bracket"],
                      "Q_rel_vs_xsec": c["Q_paper"] / x["Q0"] - 1,
                      "f_3D_minus_xsec_MHz": (c["f_axionFEM_GHz"] * 1e9 - x["f_Hz"]) / 1e6,
                      "Q_3D_over_xsec": c["Q_axionFEM"] / x["Q0"] - 1})
    cm.write_record(NAME, rec)
    for c in comp:
        print(c)
    plot(rec)


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comp = rec["comparison"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    i = [c["mode"] for c in comp]
    ax[0].plot(i, [c["df_MHz"] for c in comp], "o--", mfc="none", label="CST (Table 2) - axionFEM 3D")
    if "df_vs_xsec_MHz" in comp[0]:
        ax[0].plot(i, [c["df_vs_xsec_MHz"] for c in comp], "o-", label="CST (Table 2) - axionFEM exact 2D")
    ax[0].plot(i, [(c["f_measured_2K_GHz"] - c.get("f_xsec_GHz", c["f_axionFEM_GHz"])) * 1e3 for c in comp], "s--",
               label="measured 2 K (Table 3) - axionFEM (sanity)")
    ax[0].axhspan(-0.5, 0.5, color="0.85", label="printed rounding")
    ax[0].set_xlabel("mode")
    ax[0].set_ylabel("f difference [MHz]")
    ax[0].legend(fontsize=7)
    ax[0].set_title("frequencies (axionFEM converged)")
    ax[1].semilogy(i, [max(c["C_paper"], 1e-13) for c in comp], "ks", label="CST G^2 (Table 2)")
    ax[1].semilogy(i, [max(c["C_axionFEM_full_volume"], 1e-13) for c in comp], "ro", mfc="none",
                   label="axionFEM C (V incl. apertures)")
    ax[1].set_xlabel("mode")
    ax[1].set_ylabel("C")
    ax[1].set_ylim(1e-13, 2)
    ax[1].legend(fontsize=7)
    ax[1].set_title("form factor (modes 2, 4: 0 by symmetry)")
    for s in rec["mesh_study"]:
        ax[2].plot(i, s["Q0"], "o-", ms=3, lw=0.8, label=f"{s['tag']}")
    if "Q_xsec" in comp[0]:
        ax[2].plot(i, [c["Q_xsec"] for c in comp], "r^-", lw=2, label="exact 2D, edge-extrapolated (h^(1/3))")
    ax[2].plot(i, [c["Q_paper"] for c in comp], "ks-", lw=2, label="CST (Table 2)")
    ax[2].set_xlabel("mode")
    ax[2].set_ylabel("Q0 (sigma = 2e9 S/m)")
    ax[2].legend(fontsize=6)
    ax[2].set_title("unloaded Q: mesh study")
    fig.suptitle("RADES 5-cavity filter 1803.01243 (3D, 1/8 model); beta", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=110)


if __name__ == "__main__":
    main()
