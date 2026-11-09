"""Reproduction: QUAX hollow-sapphire (two-shell) cavity, 2004.02754 Fig. 4 / Table 1 (beta).

    python benchmarks/cavity/repro/quax_hollow_sapphire_2004.py

Geometry (Fig. 4 table, Fig. 10 drawing, refs JSON): copper cylinder R = 29 mm, h = 180 mm; sapphire shells
(eps 11.2, tan delta 2e-6) with inner radii 10.7 / 19.8 mm and thicknesses 1.9 / 1.6 mm over the full height;
conical recess in each end plate: base diameter 20.7 mm (Fig. 10), height 6 mm, ending in the antenna bore
(diameter 4 mm, ~20 mm long, closed at its end here: far below cutoff). Walls: R_s = 5.5 mOhm (constant).
Reference: f = 10.915 GHz, Q = 1.91e6 (pseudo-TM030); Table 1 one-at-a-time sensitivities.
Cone ambiguity variants: flat ends, sharp cone (no bore), base diameter 21.4 mm (Fig. 4 sketch), cone without
bore, and the as-built tube dimensions. Mesh study at the baseline: order 2 at h = 1.0, 0.7, 0.5 mm and order 3
at h = 0.7 mm (cone zone at h/2).
Writes benchmarks/records/cavity_repro_quax_hollow_sapphire_2004.json and .png next to this script.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "quax_hollow_sapphire_2004"
MM = 1e-3
RS = 5.5e-3
TAN = 2e-6
F_REF, Q_REF = 10.915e9, 1.91e6
BASE = {"R": 29.0, "H": 180.0, "R1": 10.7, "t1": 1.9, "R2": 19.8, "t2": 1.6, "eps": 11.2,
        "cone": True, "rb": 10.35, "hc": 6.0, "rt": 2.0, "bore_r": 2.0, "bore_l": 20.0}
MESH = [(2, 1.0), (2, 0.7), (2, 0.5), (3, 0.7)]
PROD = (2, 0.5)  # mesh used for the variants and Table 1 rows (checked against the study at the baseline)


def outline(p):
    R, z1 = p["R"] * MM, p["H"] / 2 * MM
    if not p["cone"]:
        return [[0, -z1], [R, -z1], [R, z1], [0, z1]]
    rb, hc, rt, br, bl = (p[k] * MM for k in ("rb", "hc", "rt", "bore_r", "bore_l"))
    top = [[R, z1], [rb, z1], [rt, z1 + hc]]
    if bl > 0:
        if br != rt:
            top.append([br, z1 + hc])
        top += [[br, z1 + hc + bl], [0, z1 + hc + bl]]
    else:
        if rt > 0:
            top.append([0, z1 + hc])
    top = [q for q in top]
    bot = [[r, -z] for r, z in reversed(top)]
    pts = bot + top
    if pts[0][0] != 0:
        pts = [[0, pts[0][1]]] + pts
    # remove duplicate consecutive points
    out = []
    for q in pts:
        if not out or abs(out[-1][0] - q[0]) + abs(out[-1][1] - q[1]) > 1e-12:
            out.append(q)
    if out[-1][0] != 0:
        out.append([0, out[-1][1]])
    return out


def build(tag, p, order, h):
    z1 = p["H"] / 2 * MM
    sh = lambda r0, t: {"rect": {"r": [r0 * MM, (r0 + t) * MM], "z": [-z1, z1]}}  # noqa: E731
    diel = [{"name": "shell1", "shape": sh(p["R1"], p["t1"]), "eps_r": p["eps"], "tan_delta": TAN},
            {"name": "shell2", "shape": sh(p["R2"], p["t2"]), "eps_r": p["eps"], "tan_delta": TAN}]
    ref = []
    if p["cone"]:
        ol = outline(p)
        zt = max(q[1] for q in ol)
        for s in (1, -1):
            zz = sorted([s * (z1 - 3 * MM), s * zt])
            ref.append({"name": f"cone_zone{'t' if s > 0 else 'b'}",
                        "shape": {"polygon": {"points": cm.clip_z(ol, zz[0], zz[1])}}, "mesh_size": h * MM / 2})
    return cm.design(tag, outline(p), h=h * MM, order=order, modes=8, families=("TM",), wall=cm.const_Rs(RS),
                     dielectrics=diel, refine=ref, target_f_Hz=F_REF, interior_point=[5 * MM, 0.0], t=3 * MM,
                     surface_size=h * MM)


# decimals of the Table 1 frequencies as printed in 2004.02754 (the refs JSON stores floats, so 10.90 reads as 10.9)
PRINTED_DEC = {"R_cyl_1": 2, "R_cyl_2": 3, "t_cyl_1": 2, "t_cyl_2": 2, "eps_r=9.39": 2, "eps_r=11.5": 2}


def half_unit_MHz(dec: int) -> float:
    """Half a unit in the last printed digit of a frequency in GHz with `dec` decimals, in MHz."""
    return 0.5 * 10 ** (-dec) * 1e3


def tm030(data, f0=F_REF):
    ms = [m for m in cm.modes(data) if abs(m["f_Hz"] / f0 - 1) < 0.05]
    return max(ms, key=lambda m: m["form_factor"]["C"])


def summarize(data, p):
    m = tm030(data)
    r = cm.mode_row(data, m, V_full=math.pi * (p["R"] * MM) ** 2 * p["H"] * MM, diel_regions=("shell1", "shell2"))
    r["dofs"] = cm.dofs(data)
    lo = m["losses"]
    r["Q_wall_only"] = lo["Q_wall"]
    r["G_ohm"] = lo["G_ohm"]
    r["Q_at_tan"] = {t: 1 / (1 / lo["Q_wall"] + r["p_e"] * t) for t in (1e-5, 2e-6, 1e-7)}
    near = sorted(cm.modes(data), key=lambda x: x["f_Hz"])
    r["neighbours"] = [{"f_Hz": x["f_Hz"], "C": x["form_factor"]["C"]} for x in near]
    return r


VARIANTS = {
    "flat": {"cone": False},
    "cone_sharp_nobore": {"rt": 0.0, "bore_l": 0.0},
    "cone_flat_top_nobore": {"bore_l": 0.0},
    "cone_base21.4": {"rb": 10.7},
    "cone_h5": {"hc": 5.0},
    "cone_h7": {"hc": 7.0},
    "as_built_tubes": {"R1": 10.68, "t1": 1.905, "R2": 19.855, "t2": 1.545},
}
TABLE1 = {"R_cyl_1": {"R1": 10.9}, "R_cyl_2": {"R2": 20.0}, "t_cyl_1": {"t1": 2.1}, "t_cyl_2": {"t2": 1.8},
          "eps_r=9.39": {"eps": 9.39}, "eps_r=11.5": {"eps": 11.5}}


def main():
    rj = cm.ref(NAME)
    t1 = rj["reference_values"]["tables"][0]["rows"]
    jobs = []
    for o, h in MESH:
        jobs.append((f"base_o{o}_h{h}", build(f"base_o{o}_h{h}", BASE, o, h)))
    for k, v in {**VARIANTS, **TABLE1}.items():
        p = {**BASE, **v}
        jobs.append((f"{k}_o{PROD[0]}_h{PROD[1]}", build(k, p, *PROD)))
    out = {tag: (data, dt) for tag, data, dt in cm.solve_many(jobs, NAME)}
    study = []
    for o, h in MESH:
        data, dt = out[f"base_o{o}_h{h}"]
        s = summarize(data, BASE)
        s.update({"order": o, "h_mm": h, "seconds": dt})
        study.append(s)
    conv = [s for s in study if s["order"] == 3][0]
    prod = [s for s in study if (s["order"], s["h_mm"]) == PROD][0]
    rec = {"benchmark": "QUAX hollow sapphire 2004.02754 (Fig. 4 geometry, Table 1)",
           "assumptions": {"Rs_ohm": RS, "tan_delta": TAN, "eps_r": 11.2, "baseline_cone": "base diameter 20.7 mm, "
                           "height 6 mm, frustum to the 4 mm bore, bore 20 mm long closed",
                           "C": "eps-weighted, V = cavity volume incl. cone recesses (no metal inside, so V_vac = V); "
                                "C_full uses V = pi R^2 h_cav (cylinder only)"},
           "mesh_study": study, "converged": conv,
           "mesh_error_prod_vs_converged": {"f_rel": prod["f_Hz"] / conv["f_Hz"] - 1, "Q_rel": prod["Q0"] / conv["Q0"] - 1,
                                            "C_rel": prod["C"] / conv["C"] - 1}}
    rec["reference"] = {"f_Hz": F_REF, "Q": Q_REF, "f_rel": F_REF / conv["f_Hz"] - 1, "Q_rel": Q_REF / conv["Q0"] - 1,
                        "f_tol_rel": 0.003, "derived_wall_only_Q_from_Tab1": 5.4e6, "derived_p_e_from_Tab1": 0.167}
    rec["variants"] = {}
    for k, v in VARIANTS.items():
        s = summarize(out[f"{k}_o{PROD[0]}_h{PROD[1]}"][0], {**BASE, **v})
        rec["variants"][k] = {"params": v, **s, "df_vs_base_Hz": s["f_Hz"] - prod["f_Hz"], "f_rel_ref": F_REF / s["f_Hz"] - 1,
                              "Q_rel_ref": Q_REF / s["Q0"] - 1}
    rows = []
    rows.append({"row": "loss tangent 1e-5", "f_paper": 10.915e9, "Q_paper": 0.54e6, "f_FEM": prod["f_Hz"],
                 "df_FEM_MHz": 0.0, "df_paper_MHz": 0.0, "Q_FEM": prod["Q_at_tan"][1e-5]})
    rows.append({"row": "loss tangent 1e-7", "f_paper": 10.916e9, "Q_paper": 4.98e6, "f_FEM": prod["f_Hz"],
                 "df_FEM_MHz": 0.0, "df_paper_MHz": 1.0, "Q_FEM": prod["Q_at_tan"][1e-7]})
    pmap = {"R_cyl_1": "R_cyl_1", "R_cyl_2": "R_cyl_2", "t_cyl_1": "t_cyl_1", "t_cyl_2": "t_cyl_2"}
    for k in TABLE1:
        s = summarize(out[f"{k}_o{PROD[0]}_h{PROD[1]}"][0], {**BASE, **TABLE1[k]})
        if k.startswith("eps"):
            tr = [r for r in t1 if r[0] == "eps_r" and abs(float(r[2]) - TABLE1[k]["eps"]) < 1e-6][0]
        else:
            tr = [r for r in t1 if r[0] == pmap[k]][0]
        fp, Qp = float(tr[3]) * 1e9, float(tr[4]) * 1e6
        rows.append({"row": k, "f_paper": fp, "f_paper_printed": str(tr[3]), "df_paper_rounding_MHz": half_unit_MHz(PRINTED_DEC[k]) + 0.5, "Q_paper": Qp, "f_FEM": s["f_Hz"], "Q_FEM": s["Q0"],
                     "df_FEM_MHz": (s["f_Hz"] - prod["f_Hz"]) / 1e6, "df_paper_MHz": (fp - F_REF) / 1e6,
                     "dQ_rel_FEM": s["Q0"] / prod["Q0"] - 1, "dQ_rel_paper": Qp / Q_REF - 1, "C": s["C"], "p_e": s["p_e"],
                     "neighbours": s["neighbours"]})
    for r in rows:
        r["ddf_MHz"] = r["df_FEM_MHz"] - r["df_paper_MHz"]
        # tolerance = rounding of the printed row f plus that of the printed reference 10.915 GHz (+-0.5 MHz); the
        # loss-tangent rows print 10.915 / 10.916 (+-0.5 MHz each)
        r.setdefault("df_paper_rounding_MHz", 1.0)
        r["within_rounding"] = abs(r["ddf_MHz"]) <= r["df_paper_rounding_MHz"]
    rec["table1"] = rows
    cm.write_record(NAME, rec)
    plot(rec)
    print("converged", {k: conv[k] for k in ("f_Hz", "C", "C_full", "Q0", "Q_wall", "p_e", "G_ohm")})
    print("ref", rec["reference"])
    for k, v in rec["variants"].items():
        print(k, round(v["f_Hz"] / 1e9, 5), round(v["Q0"]), round(v["C"], 4), round(v["p_e"], 4))
    for r in rows:
        print(r["row"], round(r["df_FEM_MHz"], 2), r["df_paper_MHz"], r.get("Q_FEM"), r["Q_paper"])


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    st = [s for s in rec["mesh_study"] if s["order"] == 2]
    c = rec["converged"]
    for key, lab in (("f_Hz", "f"), ("Q0", "Q0"), ("C", "C030")):
        ax[0].loglog([s["h_mm"] for s in st], [max(abs(s[key] / c[key] - 1), 1e-12) for s in st], "o-",
                     label=f"{lab} (vs order 3, h = 0.7 mm)")
    ax[0].set_xlabel("h [mm] (order 2)")
    ax[0].set_ylabel("relative change")
    ax[0].set_title("mesh study, baseline cone")
    ax[0].legend(fontsize=7)
    rows = rec["table1"]
    x = range(len(rows))
    ax[1].bar([i - 0.2 for i in x], [r["df_paper_MHz"] for r in rows], 0.4, label="paper Table 1 (HFSS)")
    ax[1].bar([i + 0.2 for i in x], [r["df_FEM_MHz"] for r in rows], 0.4, label="axionFEM")
    ax[1].set_xticks(list(x), [r["row"] for r in rows], rotation=60, fontsize=7)
    ax[1].set_ylabel("f - f_reference [MHz]")
    ax[1].set_title("Table 1 frequency sensitivities")
    ax[1].legend(fontsize=7)
    ax[2].bar([i - 0.2 for i in x], [r["Q_paper"] / 1e6 for r in rows], 0.4, label="paper")
    ax[2].bar([i + 0.2 for i in x], [r["Q_FEM"] / 1e6 for r in rows], 0.4, label="axionFEM")
    ax[2].set_xticks(list(x), [r["row"] for r in rows], rotation=60, fontsize=7)
    ax[2].set_ylabel("Q0 [1e6]")
    ax[2].set_title(f"Q0 (reference: paper 1.91e6, axionFEM {c['Q0'] / 1e6:.2f}e6)")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"QUAX 2004.02754 hollow sapphire: f = {c['f_Hz'] / 1e9:.4f} GHz (paper 10.915), "
                 f"C030 = {c['C']:.4f}, p_e = {c['p_e']:.3f} (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
