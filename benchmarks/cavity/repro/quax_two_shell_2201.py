"""Reproduction: QUAX two-shell sapphire cavity, 2201.04223 Tables I-II (beta).

    python benchmarks/cavity/repro/quax_two_shell_2201.py

Geometry (room temperature, refs JSON): copper cylinder D = 57.1 mm, L = 420 mm; sapphire tubes ID 21.75 / t 2.35
and ID 41.8 / t 1.94 mm, full length. Not given (swept): the end-cap cones (height 0 = flat .. 10 mm, base radius
10.35 / 10.8 mm, frustum ending in a 4 mm x 20 mm closed bore as in the sister design 2004.02754) and the tube
end gap (0 .. 1.5 mm at each end). Thermal contraction 293 -> 4.2 K: set A (refs JSON): Cu -3.26e-3, sapphire
-8e-4; set B (2410.07774 Table I): Cu -4.36e-3, sapphire -8.1e-4. R_s = 4.83 mOhm (4.2 K) for Q_Cu.
Hard targets (refs README): Delta f(eps) (Table II: 11.44 -> 11.2 = +18.9 MHz), Delta f(T) from contraction
(+8.8 MHz = Tab II eps 11.44 minus Tab I 293 K), p_e = 0.170, C trend vs eps. Soft: absolute f (~1 %), C (~15 %);
Q_Cu only as a cone-calibration quantity. Also: R1 (inner-tube inner radius, +-0.075 mm tolerance) fitted to the
absolute f, then C compared.
Mesh study at the baseline (cone 6 mm, base 10.35 mm, gap 0, 4.2 K, eps 11.2): order 2 at h = 1.0, 0.7, 0.5 mm and
order 3 at 0.7 mm.
Writes benchmarks/records/cavity_repro_quax_two_shell_2201.json and .png next to this script.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "quax_two_shell_2201"
MM = 1e-3
RS4 = 4.83e-3
F4 = 10.4745e9
RT = {"R": 28.55, "L": 420.0, "R1": 10.875, "t1": 2.35, "R2": 20.90, "t2": 1.94}
CONTR = {"A": (3.26e-3, 8e-4), "B": (4.36e-3, 8.1e-4)}
BASE = {"T": "4.2", "contr": "A", "eps": 11.2, "hc": 6.0, "rb": 10.35, "bore_r": 2.0, "bore_l": 20.0, "gap": 0.0,
        "dR1": 0.0, "dR1in": 0.0, "dR1out": 0.0, "dR2in": 0.0, "dRcav": 0.0}
# tolerance sweep (RT values): tube-1 ID +-0.15 mm (radius +-0.075, OD fixed), OD +-0.05 mm (radius +-0.025);
# D_cav 57.1 (Fig. 1a) vs 57 (text); tube-2 inner radius (tolerance not stated; 0.1 mm probe)
TOL = {"dR1in-0.075": {"dR1in": -0.075}, "dR1in+0.075": {"dR1in": 0.075}, "dR1out-0.025": {"dR1out": -0.025},
       "dR1out+0.025": {"dR1out": 0.025}, "dRcav-0.05": {"dRcav": -0.05}, "dR2in-0.1": {"dR2in": -0.1},
       "dR2in+0.1": {"dR2in": 0.1}}
MESH = [(2, 1.0), (2, 0.7), (2, 0.5), (3, 0.7)]
PROD = (2, 0.5)


def geom(p):
    cu, sa = CONTR[p["contr"]] if p["T"] == "4.2" else (0.0, 0.0)
    g = {"R": (RT["R"] + p["dRcav"]) * (1 - cu), "L": RT["L"] * (1 - cu)}
    rt = dict(RT)
    rt["R1"] += p["dR1"] + p["dR1in"]
    rt["t1"] += -p["dR1in"] + p["dR1out"]
    rt["R2"] += p["dR2in"]
    rt["t2"] -= p["dR2in"]
    for k in ("R1", "t1", "R2", "t2"):
        g[k] = rt[k] * (1 - sa)
    # cone and bore scale with copper; gap is the gap at the simulated temperature
    for k in ("hc", "rb", "bore_r", "bore_l"):
        g[k] = p[k] * (1 - cu)
    return g


def outline(g):
    R, z1 = g["R"] * MM, g["L"] / 2 * MM
    if g["hc"] <= 0:
        return [[0, -z1], [R, -z1], [R, z1], [0, z1]]
    rb, hc, br, bl = (g[k] * MM for k in ("rb", "hc", "bore_r", "bore_l"))
    top = [[R, z1], [rb, z1], [br, z1 + hc], [br, z1 + hc + bl], [0, z1 + hc + bl]]
    bot = [[r, -z] for r, z in reversed(top)]
    return bot + top


def build(tag, p, order, h):
    g = geom(p)
    z1 = g["L"] / 2 * MM
    zg = z1 - p["gap"] * MM
    sh = lambda r0, t: {"rect": {"r": [r0 * MM, (r0 + t) * MM], "z": [-zg, zg]}}  # noqa: E731
    diel = [{"name": "tube1", "shape": sh(g["R1"], g["t1"]), "eps_r": p["eps"], "tan_delta": 1e-9},
            {"name": "tube2", "shape": sh(g["R2"], g["t2"]), "eps_r": p["eps"], "tan_delta": 1e-9}]
    he = h / 2 if p["gap"] <= 0 else min(h / 2, p["gap"] / 2)
    ref = []
    ol = outline(g)
    zt = max(q[1] for q in ol)
    for s in (1, -1):
        zz = sorted([s * (z1 - 4 * MM), s * zt])
        ref.append({"name": f"end_zone{'t' if s > 0 else 'b'}",
                    "shape": {"polygon": {"points": cm.clip_z(ol, zz[0], zz[1])}}, "mesh_size": he * MM})
    # the dielectric regions are painted after the refine zones: give their ends the same size via mesh_size=h and
    # rely on the vacuum zone size at the tube ends (the tubes are z-uniform elsewhere)
    return cm.design(tag, outline(g), h=h * MM, order=order, modes=10, families=("TM",), wall=cm.const_Rs(RS4),
                     dielectrics=diel, refine=ref, target_f_Hz=F4, interior_point=[5 * MM, 0.0], t=3 * MM,
                     surface_size=h * MM)


def summarize(data, p):
    ms = [m for m in cm.modes(data) if abs(m["f_Hz"] / F4 - 1) < 0.05]
    m = max(ms, key=lambda x: x["form_factor"]["C"])
    g = geom(p)
    r = cm.mode_row(data, m, V_full=math.pi * (g["R"] * MM) ** 2 * g["L"] * MM, diel_regions=("tube1", "tube2"))
    r["dofs"] = cm.dofs(data)
    r["Q_Cu"] = m["losses"]["Q_wall"]
    r["neighbours"] = sorted([(x["f_Hz"], x["form_factor"]["C"]) for x in cm.modes(data)])
    return r


def cases():
    cs = {}
    for o, h in MESH:
        cs[f"base_o{o}_h{h}"] = (BASE, o, h)
    P = PROD

    def add(tag, **kw):
        cs[f"{tag}_o{P[0]}_h{P[1]}"] = ({**BASE, **kw}, *P)
    for e in (11.44, 11.3, 11.1):
        add(f"eps{e}", eps=e)
    add("T293_eps11.44", T="293", eps=11.44)
    add("contrB_eps11.2", contr="B")
    add("contrB_eps11.44", contr="B", eps=11.44)
    for hc in (0.0, 4.0, 8.0, 10.0):
        add(f"cone_h{hc}", hc=hc)
    for hc in (6.0, 8.0):
        add(f"cone_h{hc}_rb10.8", hc=hc, rb=10.8)
    for gp in (0.5, 1.0, 1.5):
        add(f"gap{gp}", gap=gp)
        add(f"gap{gp}_eps11.44", gap=gp, eps=11.44)
        add(f"gap{gp}_T293", gap=gp, T="293", eps=11.44)
    for hc in (0.0, 10.0):
        add(f"cone_h{hc}_eps11.44", hc=hc, eps=11.44)
        add(f"cone_h{hc}_T293", hc=hc, T="293", eps=11.44)
    for d in (-0.1, -0.2):
        add(f"dR1{d}", dR1=d)
    for k, v in TOL.items():
        add(k, **v)
    return cs


def main():
    rj = cm.ref(NAME)
    cs = cases()
    jobs = [(tag, build(tag, p, o, h)) for tag, (p, o, h) in cs.items()]
    out = {tag: (data, dt) for tag, data, dt in cm.solve_many(jobs, NAME)}
    S = {tag: {**summarize(out[tag][0], cs[tag][0]), "params": cs[tag][0], "order": cs[tag][1], "h_mm": cs[tag][2],
               "seconds": out[tag][1]} for tag in cs}
    P = f"_o{PROD[0]}_h{PROD[1]}"
    base = S["base" + P]
    conv = S["base_o3_h0.7"]
    tab2 = {float(r[0]): {"f": r[1] * 1e9, "C": r[2], "Q_Cu": float(r[3])} for r in rj["reference_values"]["tables"][1]["rows"]}
    rec = {"benchmark": "QUAX two-shell sapphire 2201.04223 (Tables I-II)",
           "assumptions": {"Rs_ohm": RS4, "contraction": CONTR, "baseline": BASE,
                           "C": "eps-weighted, V = cavity volume incl. cone recesses = V_vac (no metal inside); C_full: "
                                "V = pi R^2 L (cylinder only)"},
           "mesh_study": [S[f"base_o{o}_h{h}"] for o, h in MESH],
           "mesh_error_prod_vs_converged": {k: base[k] / conv[k] - 1 for k in ("f_Hz", "C", "p_e", "Q_Cu")},
           "runs": S}

    def df(a, b):
        return (S[a + P]["f_Hz"] - S[b + P]["f_Hz"]) / 1e6

    hard = {}
    # Delta f(eps): 11.44 -> 11.2 at 4.2 K geometry, for every cone/gap configuration available
    hard["df_eps_11.44_to_11.2_MHz"] = {"paper": (tab2[11.2]["f"] - tab2[11.44]["f"]) / 1e6,
                                        "axionFEM_baseline": df("base", "eps11.44"),
                                        "flat": df("cone_h0.0", "cone_h0.0_eps11.44"),
                                        "cone10": df("cone_h10.0", "cone_h10.0_eps11.44"),
                                        **{f"gap{g}": df(f"gap{g}", f"gap{g}_eps11.44") for g in (0.5, 1.0, 1.5)},
                                        "contrB": df("contrB_eps11.2", "contrB_eps11.44")}
    de = hard["df_eps_11.44_to_11.2_MHz"]
    de["axionFEM_rescaled_by_f"] = de["axionFEM_baseline"] * F4 / conv["f_Hz"]
    de["rel_rescaled_published_minus_FEM"] = de["paper"] / de["axionFEM_rescaled_by_f"] - 1
    de["printed_rounding_MHz"] = 0.1
    de["tolerance_note"] = ("no tolerance was set for this target in refs before the run (README: 'hard target', "
                            "19 MHz); Table II prints f to 0.1 MHz, so the residual 0.13 MHz after f rescaling is "
                            "just outside the printed rounding; any % tolerance quoted for it is post hoc")
    hard["df_eps_slope"] = {"paper_f_GHz": [tab2[e]["f"] / 1e9 for e in (11.44, 11.3, 11.2, 11.1)],
                            "axionFEM_f_GHz": [S[t + P]["f_Hz"] / 1e9 for t in ("eps11.44", "eps11.3", "base", "eps11.1")],
                            "paper_C": [tab2[e]["C"] for e in (11.44, 11.3, 11.2, 11.1)],
                            "axionFEM_C": [S[t + P]["C"] for t in ("eps11.44", "eps11.3", "base", "eps11.1")]}
    pf, af = hard["df_eps_slope"]["paper_f_GHz"], hard["df_eps_slope"]["axionFEM_f_GHz"]
    hard["df_eps_slope"]["paper_minus_first_MHz"] = [1e3 * (x - pf[0]) for x in pf]
    hard["df_eps_slope"]["axionFEM_minus_first_MHz"] = [1e3 * (x - af[0]) for x in af]
    pc, ac = hard["df_eps_slope"]["paper_C"], hard["df_eps_slope"]["axionFEM_C"]
    hard["df_eps_slope"]["paper_C_rel_to_eps11.2"] = [x / pc[2] - 1 for x in pc]
    hard["df_eps_slope"]["axionFEM_C_rel_to_eps11.2"] = [x / ac[2] - 1 for x in ac]
    # Delta f(T) from contraction at eps 11.44: Tab II (4.2 K geometry) minus Tab I (293 K)
    hard["df_thermal_MHz"] = {"paper": (tab2[11.44]["f"] - 10.4468e9) / 1e6,
                              "axionFEM_contrA": df("eps11.44", "T293_eps11.44"),
                              "axionFEM_contrB": df("contrB_eps11.44", "T293_eps11.44"),
                              "flat_contrA": df("cone_h0.0_eps11.44", "cone_h0.0_T293"),
                              "cone10_contrA": df("cone_h10.0_eps11.44", "cone_h10.0_T293"),
                              **{f"gap{g}_contrA": df(f"gap{g}_eps11.44", f"gap{g}_T293") for g in (0.5, 1.0, 1.5)}}
    hard["df_total_293_to_4K_MHz"] = {"paper": (10.4745e9 - 10.4468e9) / 1e6, "axionFEM_contrA": df("base", "T293_eps11.44")}
    hard["p_e"] = {"paper": 0.170, "axionFEM_converged": conv["p_e"],
                   "range_over_all_runs": [min(s["p_e"] for s in S.values()), max(s["p_e"] for s in S.values())]}
    rec["hard_targets"] = hard
    soft = {"f_4K_eps11.2": {"paper": F4, "axionFEM_converged": conv["f_Hz"], "rel": F4 / conv["f_Hz"] - 1, "tol": 0.01},
            "C_4K_eps11.2": {"paper": 0.0321, "axionFEM_converged": conv["C"], "axionFEM_C_full": conv["C_full"],
                             "rel": 0.0321 / conv["C"] - 1, "tol": 0.15},
            "Q_Cu_4K": {"paper": 1.10e7, "axionFEM_converged": conv["Q_Cu"], "note": "cone calibration only"}}
    # R1 fit (linear interpolation in dR1 at the production mesh)
    xs = [0.0, -0.1, -0.2]
    ys = [S[("base" if x == 0 else f"dR1{x}") + P]["f_Hz"] for x in xs]
    slope = (ys[2] - ys[0]) / (xs[2] - xs[0])
    dR1 = (F4 - ys[0]) / slope
    Cs = [S[("base" if x == 0 else f"dR1{x}") + P]["C"] for x in xs]
    Cfit = Cs[0] + (Cs[2] - Cs[0]) / (xs[2] - xs[0]) * dR1
    soft["R1_fit"] = {"df_dR1_GHz_per_mm": slope / 1e9, "dR1_mm": dR1, "R1_fit_mm_RT_equiv": RT["R1"] + dR1,
                      "C_at_fit": Cfit, "C_rel_paper": 0.0321 / Cfit - 1,
                      "note": "fit at the baseline cone and zero gap; tube ID tolerance +-0.15 mm (R1 +-0.075 mm)"}
    soft["cone_sweep"] = [{"tag": t, "hc_mm": S[t + P]["params"]["hc"], "rb_mm": S[t + P]["params"]["rb"],
                           "f_GHz": S[t + P]["f_Hz"] / 1e9, "C": S[t + P]["C"], "Q_Cu": S[t + P]["Q_Cu"],
                           "G_ohm": S[t + P]["G_ohm"], "V_L": S[t + P]["V_m3"] * 1e3}
                          for t in ("cone_h0.0", "cone_h4.0", "base", "cone_h8.0", "cone_h10.0", "cone_h6.0_rb10.8",
                                    "cone_h8.0_rb10.8")]
    soft["gap_sweep"] = [{"gap_mm": g, "f_GHz": S[(f"gap{g}" if g else "base") + P]["f_Hz"] / 1e9,
                          "C": S[(f"gap{g}" if g else "base") + P]["C"], "p_e": S[(f"gap{g}" if g else "base") + P]["p_e"],
                          "Q_Cu": S[(f"gap{g}" if g else "base") + P]["Q_Cu"]} for g in (0, 0.5, 1.0, 1.5)]
    tol = {k: {"df_MHz": df(k, "base"), "dC_rel": S[k + P]["C"] / base["C"] - 1} for k in TOL}
    tol["max_df_within_stated_tolerances_MHz"] = (max(tol["dR1in-0.075"]["df_MHz"], tol["dR1in+0.075"]["df_MHz"]) +
                                                  max(tol["dR1out-0.025"]["df_MHz"], tol["dR1out+0.025"]["df_MHz"]) +
                                                  max(tol["dRcav-0.05"]["df_MHz"], 0.0))
    tol["needed_df_MHz"] = (F4 - conv["f_Hz"]) / 1e6
    soft["tolerance_sweep"] = tol
    rec["soft_targets"] = soft
    rec["V_paper_L"] = 1.0776
    v293 = S["T293_eps11.44" + P]["V_m3"] * 1e3
    rec["V_293K_L"] = {"D57.1_with_cones": v293,
                       "D57.0_with_cones": v293 - math.pi * ((RT["R"] * MM) ** 2 - (28.5 * MM) ** 2) * RT["L"] * MM * 1e3,
                       "D57.0_cylinder_only": math.pi * (28.5 * MM) ** 2 * RT["L"] * MM * 1e3,
                       "note": "the volume match with the 2004 cones needs D = 57.1 mm (Fig. 1a); with the text's 57 mm "
                               "the cones do not close the gap to 1.0776 L"}
    cm.write_record(NAME, rec)
    plot(rec)
    print(json_short(hard))
    print(json_short(soft))


def json_short(x):
    import json
    return json.dumps(x, indent=0, default=lambda v: round(v, 6))[:4000]


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, s = rec["hard_targets"], rec["soft_targets"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    e = [11.44, 11.3, 11.2, 11.1]
    ax[0, 0].plot(e, h["df_eps_slope"]["paper_minus_first_MHz"], "o-", label="paper Table II (HFSS)")
    ax[0, 0].plot(e, h["df_eps_slope"]["axionFEM_minus_first_MHz"], "s--", label="axionFEM (baseline cone, gap 0)")
    ax[0, 0].set_xlabel("eps_r")
    ax[0, 0].set_ylabel("f - f(eps = 11.44) [MHz]")
    ax[0, 0].set_title("Delta f(eps), 4.2 K geometry")
    ax[0, 0].legend(fontsize=7)
    a2 = ax[0, 0].twinx()
    a2.plot(e, [100 * x for x in h["df_eps_slope"]["paper_C_rel_to_eps11.2"]], "o:", color="C2")
    a2.plot(e, [100 * x for x in h["df_eps_slope"]["axionFEM_C_rel_to_eps11.2"]], "s:", color="C3")
    a2.set_ylabel("C/C(11.2) - 1 [%] (dotted)")
    labs = [k for k in h["df_thermal_MHz"] if k != "paper"]
    ax[0, 1].bar(range(len(labs)), [h["df_thermal_MHz"][k] for k in labs], color="C1")
    ax[0, 1].axhline(h["df_thermal_MHz"]["paper"], color="k", ls="--", label=f"paper {h['df_thermal_MHz']['paper']:.1f} MHz")
    ax[0, 1].set_xticks(range(len(labs)), labs, rotation=45, fontsize=7)
    ax[0, 1].set_ylabel("f(4.2 K geom) - f(293 K geom) [MHz], eps 11.44")
    ax[0, 1].set_title("Delta f(T) from thermal contraction")
    ax[0, 1].legend(fontsize=7)
    cs = s["cone_sweep"]
    ax[1, 0].plot([c["hc_mm"] for c in cs if c["rb_mm"] == 10.35], [c["f_GHz"] for c in cs if c["rb_mm"] == 10.35], "o-",
                  label="f, rb = 10.35")
    ax[1, 0].plot([c["hc_mm"] for c in cs if c["rb_mm"] == 10.8], [c["f_GHz"] for c in cs if c["rb_mm"] == 10.8], "s",
                  label="f, rb = 10.8")
    ax[1, 0].axhline(s["f_4K_eps11.2"]["paper"] / 1e9, color="k", ls="--", label="paper 10.4745")
    ax[1, 0].set_xlabel("cone height [mm]")
    ax[1, 0].set_ylabel("f [GHz]")
    ax[1, 0].legend(fontsize=7)
    a3 = ax[1, 0].twinx()
    a3.semilogy([c["hc_mm"] for c in cs if c["rb_mm"] == 10.35], [c["Q_Cu"] for c in cs if c["rb_mm"] == 10.35], "^:",
                color="C3")
    a3.axhline(1.10e7, color="C3", ls=":", lw=0.8)
    a3.set_ylabel("Q_Cu (dotted; paper 1.10e7)")
    ax[1, 0].set_title("cone sweep (4.2 K, eps 11.2, gap 0)")
    gs = s["gap_sweep"]
    ax[1, 1].plot([g["gap_mm"] for g in gs], [g["C"] for g in gs], "o-", label="C030 (axionFEM)")
    ax[1, 1].axhline(0.0321, color="k", ls="--", label="paper C030 = 0.0321")
    ax[1, 1].set_xlabel("tube end gap at each end [mm]")
    ax[1, 1].set_ylabel("C030")
    ax[1, 1].legend(fontsize=7, loc="upper right")
    a4 = ax[1, 1].twinx()
    a4.plot([g["gap_mm"] for g in gs], [g["f_GHz"] for g in gs], "s:", color="C1")
    a4.set_ylabel("f [GHz] (dotted)")
    ax[1, 1].set_title("tube end-gap sweep")
    fig.suptitle(f"QUAX 2201.04223 two-shell: p_e = {h['p_e']['axionFEM_converged']:.4f} (paper 0.170), "
                 f"f = {s['f_4K_eps11.2']['axionFEM_converged'] / 1e9:.4f} GHz (paper 10.4745) (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
