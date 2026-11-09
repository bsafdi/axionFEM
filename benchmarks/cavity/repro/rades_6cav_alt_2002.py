"""3D: RADES 6-subcavity alternating filter, 2002.07639 (beta).

    python benchmarks/cavity/repro/rades_6cav_alt_2002.py --stage k|filter|slots|record|all

(defaults reproduce the record: --stage all = coupling runs + U_mid_up filter + slot refinement + record)

Reference (arXiv PDF): Table 1 p. 17 (2 K design: a 22.860, b 10.160, l_ext 29.400, l_int 27.000, w 8.900,
t_ind 2.000, h 1.000, t_cap 3.500, g 1.100 mm), Table 2 p. 18 (CST eigenmodes: f, G^2, Q0 at 2 K with
sigma = 2.008e9 S/m and at 300 K with the thermally expanded dimensions and 5.8e7 S/m), Figs 4-6 pp. 14-15 (iris
layouts; coupling k vs w (inductive), k vs h (plain capacitive, Fig 5 right) and k vs h (quasi-capacitive with
g = 1.1 mm "protrusions", Fig 6 right), all from CST eigen-solves of two identical cavities joined by the iris with a
PMC/PEC mid-plane, Eq. 3.2 |k| = |fe^2 - fo^2| / (fe^2 + fo^2)); digitized curves in
benchmarks/cavity/refs/rades_6cav_alt_2002.json.

The capacitive iris is not fully dimensioned (refs gap): the slot's height position and the protrusion outline are not
given (Figs 4, 6 left and 7 show a slot of height h with metal steps around it and, as built, a U-shaped opening).
We do not tune to Table 2. Instead the iris geometry is chosen between six discrete readings by the paper's own iris
characterization (the Figs 5-6 k(h) curves), and only then is the filter solved:
  plain_floor  : slot y in [0, h] (from the floor), no side slots                       -> Fig 5 right
  plain_mid    : slot centred in y, no side slots                                        -> Fig 5 right
  U_floor      : floor slot + side slots x in [0, g], [a - g, a], full height            -> Fig 6 right
  U_floor_half : floor slot + side slots of width g/2                                    -> Fig 6 right
  U_mid_full   : centred slot + side slots of width g, full height                       -> Fig 6 right
  U_mid_up     : centred slot + side slots of width g from the slot up to the ceiling    -> Fig 6 right (chosen)
The inductive iris (centred full-height aperture w) is checked against Fig 5 left.
Coupling runs: two cavities of length l_int joined by the iris (the paper does not give the resonator length used for
Eq. 3.2; l_int is our assumption), mirror plane at the iris centre (PMC: f_e, PEC: f_o), mirror plane x = 0 (PMC);
wall losses are irrelevant for f (skin, 2 K sigma).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "rades_6cav_alt_2002"
MM = 1e-3
A, B = 22.86 * MM, 10.16 * MM
L_EXT, L_INT = 29.4 * MM, 27.0 * MM
W_IND, T_IND = 8.9 * MM, 2.0 * MM
H_CAP, T_CAP, G_CAP = 1.0 * MM, 3.5 * MM, 1.1 * MM
SIG2K, SIG300 = 2.008e9, 5.8e7



def iris_openings(kind: str, z0: float, z1: float, *, a=A, b=B, w=W_IND, h=H_CAP, g=G_CAP, variant="U_floor"):
    """Boxes (in x, y centred on 0) of the opening of one iris occupying z0 < z < z1."""
    def box(x0, x1, y0, y1):
        return {"box": {"min": [x0, y0, z0], "max": [x1, y1, z1]}}
    if kind == "ind":
        return [box(-w / 2, w / 2, -b / 2, b / 2)]
    yb = -b / 2
    if variant == "plain_floor":
        return [box(-a / 2, a / 2, yb, yb + h)]
    if variant == "plain_mid":
        return [box(-a / 2, a / 2, -h / 2, h / 2)]
    gg = g / 2 if variant == "U_floor_half" else g
    if variant in ("U_floor", "U_floor_half"):
        return [box(-a / 2, a / 2, yb, yb + h), box(-a / 2, -a / 2 + gg, yb, b / 2), box(a / 2 - gg, a / 2, yb, b / 2)]
    if variant == "U_mid_up":  # centred slot + side slots of width g from the slot up to the ceiling
        return [box(-a / 2, a / 2, -h / 2, h / 2), box(-a / 2, -a / 2 + g, -h / 2, b / 2),
                box(a / 2 - g, a / 2, -h / 2, b / 2)]
    if variant == "U_mid_full":  # centred slot + side slots of width g over the full height
        return [box(-a / 2, a / 2, -h / 2, h / 2), box(-a / 2, -a / 2 + g, yb, b / 2), box(a / 2 - g, a / 2, yb, b / 2)]
    raise ValueError(variant)


def refine_box(z0, z1, a=A, b=B):
    m = 2.0 * MM
    return {"box": {"min": [-a / 2, -b / 2, z0 - m], "max": [a / 2, b / 2, z1 + m]}}


def k_design(tag, kind, *, variant="U_mid_up", w=W_IND, h=H_CAP, zsym="pmc", hmesh=1.5 * MM, order=2, slot_factor=3.0):
    t = T_IND if kind == "ind" else T_CAP
    z0, z1 = -t / 2, t / 2
    parts = [{"box": {"min": [-A / 2, -B / 2, z1], "max": [A / 2, B / 2, z1 + L_INT]}},
             {"box": {"min": [-A / 2, -B / 2, z0 - L_INT], "max": [A / 2, B / 2, z0]}}]
    parts += iris_openings(kind, z0, z1, w=w, h=h, variant=variant)
    dom = {"union": parts}
    href = min(hmesh / 3, (h / slot_factor if kind == "cap" else hmesh / 3))
    refine = [{"name": "iris_ref", "solid": {"intersection": [refine_box(z0, z1), dom]},
               "mesh_size": href}]
    return cm.design3d(tag, dom, h=hmesh, order=order, modes=4, target_f_Hz=8.4e9, wall=cm.rf_skin(SIG2K, 2.0), B_T=(0, 1, 0),
                       symmetry={"x": "pmc", "z": zsym}, refine=refine,
                       interior_point=(2.1 * MM, 0.3 * MM, z1 + 0.37 * L_INT))


def layout(t_cap=T_CAP, t_ind=T_IND, l_ext=L_EXT, l_int=L_INT):
    seq = [("cav", l_ext), ("cap", t_cap), ("cav", l_int), ("ind", t_ind), ("cav", l_int), ("cap", t_cap),
           ("cav", l_int), ("ind", t_ind), ("cav", l_int), ("cap", t_cap), ("cav", l_ext)]
    tot = sum(x for _, x in seq)
    z = -tot / 2
    out = []
    for kind, L in seq:
        out.append((kind, z, z + L))
        z += L
    return out, tot


def filter_design(tag, *, variant="U_mid_up", dims=None, sigma=SIG2K, T_K=2.0, zsym="pmc", hmesh=1.5 * MM, order=2,
                  slot_factor=3.0,
                  full=False, modes=8):
    d = dict(a=A, b=B, l_ext=L_EXT, l_int=L_INT, w=W_IND, t_ind=T_IND, h=H_CAP, t_cap=T_CAP, g=G_CAP)
    d.update(dims or {})
    a, b = d["a"], d["b"]
    lay, tot = layout(d["t_cap"], d["t_ind"], d["l_ext"], d["l_int"])
    parts, refine, probes = [], [], []
    for kind, z0, z1 in lay:
        if kind == "cav":
            parts.append({"box": {"min": [-a / 2, -b / 2, z0], "max": [a / 2, b / 2, z1]}})
            probes.append([0.3 * MM, 0.0, 0.5 * (z0 + z1)])
        else:
            parts += iris_openings(kind, z0, z1, a=a, b=b, w=d["w"], h=d["h"], g=d["g"], variant=variant)
    dom = {"union": parts}
    for i, (kind, z0, z1) in enumerate(lay):
        if kind != "cav":
            href = d["h"] / slot_factor if kind == "cap" else hmesh / 3
            refine.append({"name": f"iris{i}_ref", "solid": {"intersection": [
                refine_box(z0, z1, a=a, b=b), dom]}, "mesh_size": href})
    sym = None if full else {"x": "pmc", "z": zsym}
    if not full:
        probes = [p if p[2] > 0 else None for p in probes]
        probes = [p for p in probes if p]
    ip = (2.1 * MM, 0.37 * MM, 0.5 * (lay[6][1] + lay[6][2]))
    return cm.design3d(tag, dom, h=hmesh, order=order, modes=modes, target_f_Hz=8.5e9, wall=cm.rf_skin(sigma, T_K),
                       B_T=(0, 1, 0), symmetry=sym, refine=refine, interior_point=ip, probes=probes), d, lay


def k_from(fe, fo):
    return (fe ** 2 - fo ** 2) / (fe ** 2 + fo ** 2)


def lowest(data, f_lo=7.0e9):
    ms = sorted(cm.modes3d(data), key=lambda m: m["f_Hz"])
    ms = [m for m in ms if m["f_Hz"] > f_lo]
    return ms[0]["f_Hz"]


def stage_k(variants=("plain_floor", "plain_mid", "U_floor", "U_floor_half", "U_mid_up", "U_mid_full")):
    jobs = []
    for w in (5.0, 6.0, 7.0, 8.0, 8.9, 10.0, 11.0, 12.0, 13.0):
        for zs in ("pmc", "pec"):
            jobs.append((f"k_ind_w{w}_{zs}", k_design(f"k_ind_w{w}_{zs}", "ind", w=w * MM, zsym=zs), ("ind", "ind", w, zs)))
    for var in variants:
        for h in (0.5, 0.75, 0.93, 1.0, 1.25, 1.5, 2.0):
            for zs in ("pmc", "pec"):
                tg = f"k_cap_{var}_h{h}_{zs}"
                jobs.append((tg, k_design(tg, "cap", variant=var, h=h * MM, zsym=zs), ("cap", var, h, zs)))
    # slot-refinement check of the chosen readings (slot element size h/6)
    for var in ("U_mid_up", "plain_mid"):
        for zs in ("pmc", "pec"):
            tg = f"k_cap_{var}_h1.0_{zs}_s6"
            jobs.append((tg, k_design(tg, "cap", variant=var, h=1.0 * MM, zsym=zs, slot_factor=6.0),
                         ("cap_slot6", var, 1.0, zs)))
    # mesh check of the coupling at the design point
    for hm in (1.0,):
        for var, h in (("U_floor", 1.0), ("plain_floor", 1.0)):
            for zs in ("pmc", "pec"):
                tg = f"k_cap_{var}_h{h}_{zs}_m{hm}"
                jobs.append((tg, k_design(tg, "cap", variant=var, h=h * MM, zsym=zs, hmesh=hm * MM),
                             ("cap_mesh", var, h, zs)))
        for zs in ("pmc", "pec"):
            tg = f"k_ind_w8.9_{zs}_m{hm}"
            jobs.append((tg, k_design(tg, "ind", w=8.9 * MM, zsym=zs, hmesh=hm * MM), ("ind_mesh", "ind", 8.9, zs)))
    out = cm.solve_many3d([(t, d) for t, d, _ in jobs], NAME)
    f = {}
    for (tag, data, dt), (_, meta) in zip(out, [(j[0], j[2]) for j in jobs]):
        kind, var, x, zs = meta
        f[(kind, var, x, zs)] = (lowest(data), cm.dofs(data))
    res = {}
    for (kind, var, x, zs), (fv, n) in f.items():
        if zs != "pmc":
            continue
        fo = f[(kind, var, x, "pec")][0]
        res.setdefault(f"{kind}:{var}", []).append({"x_mm": x, "f_e": fv, "f_o": fo, "k": k_from(fv, fo), "dofs": n})
    for v in res.values():
        v.sort(key=lambda r: r["x_mm"])
    (cm.WORK / NAME / "k_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def filter_rows(data, lay, full, zsym, a, b):
    v_cav = sum((z1 - z0) * a * b for k, z0, z1 in lay if k == "cav")
    V = data["fullwave"]["cavity"]["volume_m3"] * (1 if full else 4)
    out = []
    for m in sorted(cm.modes3d(data), key=lambda m: m["f_Hz"]):
        ff, lo = m.get("form_factor", {}), m.get("losses", {})
        C = ff.get("C")
        pr = m.get("probes", {})
        E = pr.get("E") if isinstance(pr, dict) else None
        out.append({"f_Hz": m["f_Hz"], "C_full_volume": C, "C_pockets": None if C is None else C * V / v_cav,
                    "V_full_m3": V, "V_pockets_m3": v_cav, "Q0": lo.get("Q0"), "G_ohm": lo.get("G_ohm"),
                    "Q0_pointwise": (lo["Q0"] * lo["walls"]["outer"]["G_ohm_pointwise"] / lo["walls"]["outer"]["G_ohm"])
                    if lo.get("Q0") and lo.get("walls", {}).get("outer", {}).get("G_ohm_pointwise") else None,
                    "G_flux_pointwise_disagreement": lo.get("walls", {}).get("outer", {}).get(
                        "G_flux_pointwise_disagreement"),
                    "residual_true": m.get("residual_true"), "Ey_probes": [e[1] for e in E] if E else None})
    return out


def stage_filter(variants=("U_mid_up",)):
    jobs = []
    for var in variants:
        for hm in (2.0, 1.5, 1.0):
            for zs in ("pmc", "pec"):
                tg = f"filt_{var}_m{hm}_{zs}"
                d, dims, lay = filter_design(tg, variant=var, zsym=zs, hmesh=hm * MM)
                jobs.append((tg, d, dict(variant=var, h=hm, zsym=zs, order=2, lay=lay, dims=dims)))
        for zs in ("pmc", "pec"):
            tg = f"filt_{var}_m1.5_o3_{zs}"
            d, dims, lay = filter_design(tg, variant=var, zsym=zs, hmesh=1.5 * MM, order=3)
            jobs.append((tg, d, dict(variant=var, h=1.5, zsym=zs, order=3, lay=lay, dims=dims)))
        # 300 K Q0: thermally expanded dimensions (Table 1 col. 2), sigma = 5.8e7
        exp = dict(a=22.929 * MM, b=10.191 * MM, l_ext=29.489 * MM, l_int=27.081 * MM, w=8.987 * MM,
                   t_ind=2.006 * MM, h=1.063 * MM, t_cap=3.511 * MM, g=1.104 * MM)
        for zs in ("pmc", "pec"):
            tg = f"filt_{var}_300K_m1.5_{zs}"
            d, dims, lay = filter_design(tg, variant=var, zsym=zs, hmesh=1.5 * MM, dims=exp, sigma=SIG300, T_K=300.0)
            jobs.append((tg, d, dict(variant=var, h=1.5, zsym=zs, order=2, lay=lay, dims=dims, T=300)))
            # 300 K on the same mesh as the 2 K reference (order 3, h = 1.5 mm)
            tg = f"filt_{var}_300K_m1.5_o3_{zs}"
            d, dims, lay = filter_design(tg, variant=var, zsym=zs, hmesh=1.5 * MM, order=3, dims=exp, sigma=SIG300,
                                         T_K=300.0)
            jobs.append((tg, d, dict(variant=var, h=1.5, zsym=zs, order=3, lay=lay, dims=dims, T=300)))
        # refine the capacitive slots (element size h_slot/6 and /9 instead of /3)
        for sf in (6.0,):  # h_slot/9: ~1e7 DOFs per parity, beyond one node in 3 h
            for zs in ("pmc", "pec"):
                tg = f"filt_{var}_m1.5_s{sf:g}_{zs}"
                d, dims, lay = filter_design(tg, variant=var, zsym=zs, hmesh=1.5 * MM, slot_factor=sf)
                jobs.append((tg, d, dict(variant=var, h=1.5, zsym=zs, order=2, lay=lay, dims=dims, slot_factor=sf)))
    out = cm.solve_many3d([(t, d) for t, d, _ in jobs], NAME)
    res = []
    for (tag, data, dt), (_, _, meta) in zip(out, jobs):
        res.append({"tag": tag, **{k: v for k, v in meta.items() if k not in ("lay", "dims")},
                    "dims_mm": {k: v / MM for k, v in meta["dims"].items()}, "dofs": cm.dofs(data), "seconds": dt,
                    "converged": data["fullwave"]["eigenmodes"].get("converged"),
                    "modes": filter_rows(data, meta["lay"], False, meta["zsym"], meta["dims"]["a"], meta["dims"]["b"])})
    (cm.WORK / NAME / "filter_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:20000])


# cached runs of another build (AXF_REPRO_OLD_WORK): the rejected U_floor readings come from them
OLD_BUILD_WORK = Path(os.environ["AXF_REPRO_OLD_WORK"]) / NAME if os.environ.get("AXF_REPRO_OLD_WORK") else None


def k_all(base=None):
    """Couplings from every cached coupling run (k_<kind>_<variant>_<x>_<parity>[_m<h>][_s<f>]) under `base`."""
    import re
    out = {}
    base = base or cm.WORK / NAME
    for pe in sorted(base.glob("k_*_pmc*")):
        po = Path(str(pe).replace("_pmc", "_pec"))
        if not (pe / "results.json").exists() or not (po / "results.json").exists():
            continue
        m = re.match(r"k_(ind|cap)_(?:(\w+?)_)?(?:w|h)([\d.]+)_pmc(?:_m([\d.]+))?(?:_s(\d+))?$", pe.name)
        if not m:
            continue
        kind, var, x, hm, sf = m.group(1), m.group(2) or "ind", float(m.group(3)), m.group(4), m.group(5)
        # the cached run must belong to the current geometry definition (design key of the rebuilt document)
        stale = False
        for pdir, zs in ((pe, "pmc"), (po, "pec")):
            kw = dict(zsym=zs, hmesh=float(hm) * MM if hm else 1.5 * MM, slot_factor=float(sf) if sf else 3.0)
            if kind == "ind":
                doc = k_design(pdir.name, "ind", w=x * MM, **kw).doc
            else:
                doc = k_design(pdir.name, "cap", variant=var, h=x * MM, **kw).doc
            if (pdir / "design_key.txt").read_text().strip() != cm._key(doc):
                stale = True
        if stale:
            print("k_all: skipping stale cached run", pe.name)
            continue
        de, do = json.loads((pe / "results.json").read_text()), json.loads((po / "results.json").read_text())
        fe, fo = lowest(de), lowest(do)
        key = f"{kind}:{var}" + (f":mesh{hm}" if hm else "") + (f":slot{sf}" if sf else "")
        out.setdefault(key, []).append({"x_mm": x, "f_e": fe, "f_o": fo, "k": k_from(fe, fo), "dofs": cm.dofs(de)})
    for v in out.values():
        v.sort(key=lambda r: r["x_mm"])
    return out


def six(res, prefix):
    rows = [r for r in res if r["tag"] in (prefix + "_pmc", prefix + "_pec")]
    ms = [m for r in rows for m in r["modes"] if 8.2e9 < m["f_Hz"] < 8.8e9]
    return sorted(ms, key=lambda m: m["f_Hz"]), rows


def record():
    import numpy as np
    rj = cm.ref(NAME)
    curves = {c["source"]: np.array(c["series"][0]["points"], dtype=float) for c in rj["reference_values"]["curves"]
              if c["source"].startswith(("Fig. 5", "Fig. 6"))}
    ks = k_all()
    # readings rejected by the k(h) curves (U_floor, U_floor_half) were solved on the previous build only; every
    # reading rerun on the merged build reproduced to <= 3e-13, so their old-build values are carried over (flagged)
    if OLD_BUILD_WORK is not None and OLD_BUILD_WORK.exists():
        for key, rows in k_all(OLD_BUILD_WORK).items():
            if key not in ks:
                ks[key] = [dict(r, build="another build (carried over)") for r in rows]
    kcmp = {}
    for key, rows in ks.items():
        ref = curves["Fig. 5 (left)"] if key.startswith("ind") else None
        if key.startswith("cap:plain"):
            ref = curves["Fig. 5 (right)"]
        elif key.startswith("cap:U"):
            ref = curves["Fig. 6 (right)"]
        pts = [{"x_mm": r["x_mm"], "k": r["k"], "k_fig": float(np.interp(r["x_mm"], ref[:, 0], ref[:, 1]))}
               for r in rows if ref is not None and ref[:, 0].min() <= r["x_mm"] <= ref[:, 0].max()]
        for p in pts:
            p["rel"] = p["k"] / p["k_fig"] - 1
        kcmp[key] = {"runs": rows, "vs_figure": pts,
                     "rms_rel": float(np.sqrt(np.mean([p["rel"] ** 2 for p in pts]))) if pts else None}
    res = json.loads((cm.WORK / NAME / "filter_summary.json").read_text())
    tab = rj["reference_values"]["tables"][0]["rows"]
    filt = {}
    for var in sorted({r["variant"] for r in res}):
        study = []
        for pre in (f"filt_{var}_m2.0", f"filt_{var}_m1.5", f"filt_{var}_m1.0", f"filt_{var}_m1.5_s6",
                    f"filt_{var}_m1.5_o3"):
            ms, rows = six(res, pre)
            if len(ms) != 6:
                raise RuntimeError(f"{pre}: expected 6 filter modes in 8.2-8.8 GHz, found {len(ms)}")
            study.append({"tag": pre, "dofs_per_parity": rows[0]["dofs"], "f_GHz": [m["f_Hz"] / 1e9 for m in ms],
                          "C": [m["C_full_volume"] for m in ms], "C_pockets": [m["C_pockets"] for m in ms],
                          "Q0": [m["Q0"] for m in ms], "Q0_pointwise": [m.get("Q0_pointwise") for m in ms],
                          "G_flux_pointwise_disagreement": [m.get("G_flux_pointwise_disagreement") for m in ms]})
        best = [s for s in study if s["tag"].endswith("_o3")][0]
        fine = [s for s in study if s["tag"] != best["tag"] and not s["tag"].endswith(("_m2.0", "_m1.5"))]
        hot, _ = six(res, f"filt_{var}_300K_m1.5_o3")
        hot2, _ = six(res, f"filt_{var}_300K_m1.5")
        comp = []
        for i, row in enumerate(tab):
            comp.append({"mode": row[0], "f_paper_GHz": float(row[1]), "f_axionFEM_GHz": best["f_GHz"][i],
                         "df_MHz": (float(row[1]) - best["f_GHz"][i]) * 1e3, "C_paper": float(row[2]),
                         "C_axionFEM": best["C"][i], "C_axionFEM_pockets": best["C_pockets"][i],
                         "Q2K_paper": float(row[3]), "Q2K_axionFEM": best["Q0"][i],
                         "Q2K_rel": float(row[3]) / best["Q0"][i] - 1,
                         "Q2K_axionFEM_pointwise": best["Q0_pointwise"][i],
                         "Q2K_rel_vs_pointwise": float(row[3]) / best["Q0_pointwise"][i] - 1,
                         "G_flux_pointwise_disagreement": best["G_flux_pointwise_disagreement"][i],
                         "Q300K_paper": float(row[4]), "Q300K_axionFEM": hot[i]["Q0"],
                         "Q300K_axionFEM_pointwise": hot[i].get("Q0_pointwise"),
                         "Q300K_rel": float(row[4]) / hot[i]["Q0"] - 1,
                         "Q300K_rel_vs_pointwise": float(row[4]) / hot[i]["Q0_pointwise"] - 1,
                         "Q300K_axionFEM_o2_h1.5": hot2[i]["Q0"],
                         "f_mesh_spread_MHz": [min((s["f_GHz"][i] - best["f_GHz"][i]) * 1e3 for s in fine),
                                               max((s["f_GHz"][i] - best["f_GHz"][i]) * 1e3 for s in fine)],
                         "C_mesh_range": [min(s["C"][i] for s in study), max(s["C"][i] for s in study)]})
        filt[var] = {"mesh_study": study, "converged": best["tag"], "comparison": comp}
    rec = {"benchmark": "RADES 6-subcavity alternating filter 2002.07639 (Tables 1-2, Figs 5-6), 3D",
           "iris_selection": "quasi-capacitive iris geometry chosen among discrete readings of Figs 4/6/7 by the "
                             "paper's own k(h) curves (Fig. 5 right: plain iris; Fig. 6 right: as-built iris); "
                             "never by Table 2",
           "couplings": kcmp, "filter": filt}
    cm.write_record(NAME, rec)
    plot(rec, curves)
    for k, v in kcmp.items():
        print(k, v["rms_rel"])
    for var, f in filt.items():
        for c in f["comparison"]:
            print(var, c)


def plot(rec, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    c = curves["Fig. 5 (left)"]
    ax[0].plot(c[:, 0], c[:, 1], "k-", lw=1, label="Fig. 5 left (CST)")
    r = rec["couplings"].get("ind:ind")
    ax[0].plot([x["x_mm"] for x in r["runs"]], [x["k"] for x in r["runs"]], "ro", mfc="none", label="axionFEM")
    ax[0].set_xlabel("w [mm]")
    ax[0].set_ylabel("k2 (inductive)")
    ax[0].legend(fontsize=7)
    for key, src, col in (("cap:plain_mid", "Fig. 5 (right)", "C0"), ("cap:plain_floor", "Fig. 5 (right)", "C0"),
                          ("cap:U_mid_up", "Fig. 6 (right)", "C3"), ("cap:U_mid_full", "Fig. 6 (right)", "C3"),
                          ("cap:U_floor", "Fig. 6 (right)", "C3")):
        if key not in rec["couplings"]:
            continue
        rr = rec["couplings"][key]["runs"]
        mk = "o" if "mid" in key and "full" not in key else ("s" if "floor" in key else "^")
        ax[1].plot([x["x_mm"] for x in rr], [x["k"] for x in rr], mk, color=col, mfc="none", label=key[4:])
    for src, col in (("Fig. 5 (right)", "C0"), ("Fig. 6 (right)", "C3")):
        c = curves[src]
        ax[1].plot(c[:, 0], c[:, 1], "-", color=col, lw=1, label=src + " (CST)")
    ax[1].set_xlabel("h [mm]")
    ax[1].set_ylabel("k1 (capacitive)")
    ax[1].legend(fontsize=6)
    var = "U_mid_up" if "U_mid_up" in rec["filter"] else sorted(rec["filter"])[0]
    comp = rec["filter"][var]["comparison"]
    i = [c["mode"] for c in comp]
    ax[2].plot(i, [c["df_MHz"] for c in comp], "o-")
    ax[2].axhspan(-0.5, 0.5, color="0.85", label="printed rounding")
    ax[2].set_xlabel("mode")
    ax[2].set_ylabel("f (Table 2) - f (axionFEM) [MHz]")
    ax[2].legend(fontsize=7)
    ax[2].set_title(f"filter, iris {var}")
    ax[3].semilogy(i, [c["C_paper"] for c in comp], "ks", label="G^2 Table 2")
    ax[3].semilogy(i, [max(c["C_axionFEM"], 1e-13) for c in comp], "ro", mfc="none", label="axionFEM C")
    ax[3].set_xlabel("mode")
    ax[3].legend(fontsize=7)
    ax[3].set_title("form factor")
    fig.suptitle("RADES 6-cavity alternating filter 2002.07639 (3D); beta", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="k")
    ap.add_argument("--variants", default="U_mid_up")
    a = ap.parse_args()
    if a.stage in ("k", "all"):
        stage_k()
    if a.stage == "record":
        record()
    if a.stage == "k_more":
        stage_k(tuple(a.variants.split(",")))
    if a.stage in ("filter", "all"):
        stage_filter(tuple(a.variants.split(",")))
    if a.stage == "all":
        record()


if __name__ == "__main__":
    main()
