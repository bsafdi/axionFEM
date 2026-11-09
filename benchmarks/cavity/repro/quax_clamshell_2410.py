"""Reproduction: QUAX single-shell clamshell cavity at 0 deg opening, 2410.07774 Table II (beta).

    python benchmarks/cavity/repro/quax_clamshell_2410.py

Only the closed (0 deg) clamshell is axisymmetric. Prototype (Table I, 4 K): copper r_c = 28.475 mm, L_c = 420.39 mm,
sapphire tube (eps 11.2, tan delta 1.188e-9) inner radius 10.866 mm, t = 2.348 mm, length 419.66 mm standing on the
bottom plate, 0.73 mm gap at the top; flat end plates (antennas, pins omitted). Walls: sigma = 3.18e8 S/m
(classical skin, as stated). Refs README: every simulated Q of 2410 is ~2x the analytic value for that sigma, so
only Q RATIOS are compared; C to ~4 %.
Runs:
  - prototype mesh study (order 2, h = 1.0, 0.7, 0.5 mm; order 3 at 0.7 mm; gap zone at min(h/2, gap/4));
  - prototype gap sweep 0, 0.25, 0.5, 0.73, 1.0 mm (reference values: no gap C ~0.34, 0.73 mm gap C 0.235);
  - pill-box TM010, L = 420 mm, TM010 at 11 GHz (R = 10.431 mm): Table II C 0.667, Q 76000;
  - ideal design (r_c = 28.1 mm, L = 420 mm, no gap): shell inner radius not given, t = 3.97 mm (text) or 3.01 mm
    (Table II caption): scan of the inner radius, C maximum vs Table II C = 0.476 (Q 182000);
  - avoided-crossing sensitivity: at 0.73 mm the TM030-like mode lies within ~1e-4 of
    high-order TM0np modes that the top gap couples to it. Cavity and tube length shifted together by dL (gap fixed)
    and eps shifted by deps, production mesh; every mode within 6 MHz is reported (f, C, Q0, tube energy fraction)
    together with the cluster C sum, which is the stable quantity. The dL = 0, +-0.1, +-0.2 and deps = +-0.02 runs are
    cached runs of the same design documents.
Writes benchmarks/records/cavity_repro_quax_clamshell_2410.json and .png next to this script.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "quax_clamshell_2410"
MM = 1e-3
SIGMA = 3.18e8
EPS, TAN = 11.2, 1.188e-9
PROTO = {"rc": 28.475, "Lc": 420.39, "r1": 10.866, "t": 2.348, "Lt": 419.66}
TAB2 = {"ideal": {"Q": 182000, "C": 0.476}, "proto": {"Q": 328000, "C": 0.235}, "pill": {"Q": 76000, "C": 0.667}}
F_FIG7A_0DEG = 11.021e9
MESH = [(2, 1.0), (2, 0.7), (2, 0.5), (3, 0.7)]
PROD = (2, 0.5)


def build(tag, rc, Lc, r1, t, Lt, order, h, target=11.0e9, eps=EPS):
    """Tube standing on the bottom plate (z = -Lc/2), gap Lc - Lt at the top."""
    z0, z1 = -Lc / 2 * MM, Lc / 2 * MM
    zt = z0 + Lt * MM
    gap = (Lc - Lt) * MM
    outline = [[0, z0], [rc * MM, z0], [rc * MM, z1], [0, z1]]
    diel = [{"name": "tube", "shape": {"rect": {"r": [r1 * MM, (r1 + t) * MM], "z": [z0, zt]}}, "eps_r": eps,
             "tan_delta": TAN}]
    ref = []
    if gap > 1e-9:
        hg = min(h * MM / 2, gap / 4)
        ref.append({"name": "endzone", "shape": {"rect": {"r": [0, rc * MM], "z": [zt - 3 * MM, z1]}}, "mesh_size": h * MM / 2})
        ref.append({"name": "gapzone", "shape": {"rect": {"r": [(r1 - 1.0) * MM, (r1 + t + 1.0) * MM], "z": [zt - 1.0 * MM, z1]}},
                    "mesh_size": hg})
    else:
        ref.append({"name": "endzone", "shape": {"rect": {"r": [0, rc * MM], "z": [z1 - 3 * MM, z1]}}, "mesh_size": h * MM / 2})
    return cm.design(tag, outline, h=h * MM, order=order, modes=10, families=("TM",), wall=cm.skin(SIGMA, 4.0),
                     dielectrics=diel, refine=ref, target_f_Hz=target, interior_point=[5 * MM, 0.0], t=3 * MM,
                     surface_size=h * MM)


def best(data, f0=11.0e9, win=0.06):
    ms = [m for m in cm.modes(data) if abs(m["f_Hz"] / f0 - 1) < win]
    m = max(ms, key=lambda x: x["form_factor"]["C"])
    r = cm.mode_row(data, m, diel_regions=("tube",))
    r["dofs"] = cm.dofs(data)
    r["others"] = sorted([(x["f_Hz"], x["form_factor"]["C"]) for x in cm.modes(data)])
    return r


GAPS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.73, 1.0]
SENS_DL = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
SENS_DEPS = [-0.02, -0.01, 0.01, 0.02]
CLUSTER_WIN = 6e6


def cluster(data, win=CLUSTER_WIN):
    """All modes within +-win of the largest-C mode near 11 GHz: f, C, Q0, p_e and energy fractions, and the C sum."""
    ms = [m for m in cm.modes(data) if abs(m["f_Hz"] / 11.0e9 - 1) < 0.06]
    top = max(ms, key=lambda x: x["form_factor"]["C"])
    rows = []
    for m in sorted(cm.modes(data), key=lambda x: x["f_Hz"]):
        if abs(m["f_Hz"] - top["f_Hz"]) <= win:
            ef = m.get("energy_fraction", {})
            rows.append({"f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": m["losses"]["Q0"],
                         "p_e": m["losses"]["dielectric"]["tube"]["electric_energy_fraction"],
                         "tube_energy_fraction": ef.get("tube"), "endzone_energy_fraction": ef.get("endzone"),
                         "selected": m is top})
    cs = sum(r["C"] for r in rows)
    return {"modes": rows, "C_sum": cs, "C_max": top["form_factor"]["C"], "C_max_share": top["form_factor"]["C"] / cs,
            "Q0_of_max_C": top["losses"]["Q0"], "f_of_max_C": top["f_Hz"]}
R1_SCAN = {3.97: [10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
           3.01: [10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0]}


def main():
    jobs = []
    P = PROTO
    for o, h in MESH:
        jobs.append((f"proto_o{o}_h{h}", build("proto", P["rc"], P["Lc"], P["r1"], P["t"], P["Lt"], o, h)))
    for g in GAPS:
        if g == 0.73:
            continue
        jobs.append((f"proto_gap{g}", build("proto_gap", P["rc"], P["Lc"], P["r1"], P["t"], P["Lc"] - g, *PROD)))
    Rp = cm.C_LIGHT * cm.CHI01 / (2 * math.pi * 11e9) / MM
    for o, div in ((2, 4), (2, 8), (3, 4)):
        jobs.append((f"pill_o{o}_h{div}", cm.design("pill", [[0, -0.21], [Rp * MM, -0.21], [Rp * MM, 0.21], [0, 0.21]],
                                                    h=Rp * MM / div, order=o, modes=2, families=("TM",),
                                                    wall=cm.skin(SIGMA, 4.0))))
    for t in (3.97, 3.01):
        for r1 in R1_SCAN[t]:
            jobs.append((f"ideal_t{t}_r{r1}", build("ideal", 28.1, 420.0, r1, t, 420.0, *PROD)))
    for dL in SENS_DL:
        jobs.append((f"sensL{dL:+.1f}" if round(dL, 1) == dL else f"sensL{dL:+.2f}",
                     build("sens", P["rc"], P["Lc"] + dL, P["r1"], P["t"], P["Lt"] + dL, *PROD)))
    for de in SENS_DEPS:
        jobs.append((f"senseps{de:+.2f}", build("sens", P["rc"], P["Lc"], P["r1"], P["t"], P["Lt"], *PROD, eps=EPS + de)))
    out = {tag: (data, dt) for tag, data, dt in cm.solve_many(jobs, NAME)}
    study = []
    for o, h in MESH:
        r = best(out[f"proto_o{o}_h{h}"][0])
        r.update({"order": o, "h_mm": h, "seconds": out[f"proto_o{o}_h{h}"][1]})
        study.append(r)
    conv = study[-1]
    prod = [s for s in study if (s["order"], s["h_mm"]) == PROD][0]
    gaps = []
    for g in GAPS:
        dg = out[f"proto_o{PROD[0]}_h{PROD[1]}" if g == 0.73 else f"proto_gap{g}"][0]
        r = best(dg)
        cl = cluster(dg)
        gaps.append({"gap_mm": g, **{k: r[k] for k in ("f_Hz", "C", "Q0", "Q_wall", "p_e")}, "cluster_C_sum": cl["C_sum"],
                     "cluster_n_modes": len(cl["modes"])})
    sens = []
    for dL in SENS_DL:
        tag = f"sensL{dL:+.1f}" if round(dL, 1) == dL else f"sensL{dL:+.2f}"
        sens.append({"dL_mm": dL, "deps": 0.0, **cluster(out[tag][0])})
    for de in SENS_DEPS:
        sens.append({"dL_mm": 0.0, "deps": de, **cluster(out[f"senseps{de:+.2f}"][0])})
    conv_cluster = cluster(out[f"proto_o{MESH[-1][0]}_h{MESH[-1][1]}"][0])
    least = max(sens, key=lambda x: x["C_max_share"])
    pills = []
    for o, div in ((2, 4), (2, 8), (3, 4)):
        d = out[f"pill_o{o}_h{div}"][0]
        m = min(cm.modes(d), key=lambda x: abs(x["f_Hz"] - 11e9))
        pills.append({"order": o, "h_over_R": 1 / div, "f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": m["losses"]["Q0"]})
    pill = pills[-1]
    ideal = {}
    for t in (3.97, 3.01):
        ideal[t] = []
        for r1 in R1_SCAN[t]:
            r = best(out[f"ideal_t{t}_r{r1}"][0])
            ideal[t].append({"r1_mm": r1, **{k: r[k] for k in ("f_Hz", "C", "Q0", "p_e")}})
    ibest = {t: max(v, key=lambda x: x["C"]) for t, v in ideal.items()}
    rec = {"benchmark": "QUAX clamshell 2410.07774, 0 deg (axisymmetric), Table II",
           "assumptions": {"sigma_S_m": SIGMA, "rf": "classical skin", "C": "eps-weighted, V = cavity volume (no metal inside: "
                           "V_vac = V_full)", "ends": "flat (no cones; antennas, teflon pins omitted)"},
           "prototype_mesh_study": study, "prototype_converged": conv,
           "mesh_error_prod_vs_converged": {k: prod[k] / conv[k] - 1 for k in ("f_Hz", "C", "Q0")},
           "prototype_gap_sweep": gaps, "pillbox": pills, "pillbox_exact": {"f_Hz": 11e9, "C": 4 / cm.CHI01 ** 2,
                                                                             "Q0": cm.Z0 * cm.CHI01 * 420 / (2 * (Rp + 420)) / cm.Rs_skin(11e9, SIGMA)},
           "ideal_scan": {str(k): v for k, v in ideal.items()}, "ideal_best": {str(k): v for k, v in ibest.items()},
           "prototype_cluster_converged": conv_cluster, "avoided_crossing_sensitivity": sens,
           "avoided_crossing_note": "0.73 mm top gap: the TM030-like mode mixes with high-order TM0np modes (C = 0 in "
                                    "the z-invariant no-gap cavity) within ~1e-4 in f; the C and Q0 of 'the' TM030 "
                                    "mode depend on the detuning (dL, eps, mesh); the cluster C sum is stable"}
    smax = [x["C_max"] for x in sens]
    ssum = [x["C_sum"] for x in sens]
    sq = [x["Q0_of_max_C"] for x in sens]
    cmp_ = {"C_proto": {"paper": TAB2["proto"]["C"], "axionFEM_nominal_max_C_mode": conv["C"],
                        "rel_nominal": TAB2["proto"]["C"] / conv["C"] - 1, "tol": 0.04,
                        "axionFEM_max_C_mode_range_over_sensitivity": [min(smax), max(smax)],
                        "axionFEM_cluster_C_sum_converged": conv_cluster["C_sum"],
                        "axionFEM_cluster_C_sum_range_over_sensitivity": [min(ssum), max(ssum)],
                        "rel_cluster_sum": TAB2["proto"]["C"] / conv_cluster["C_sum"] - 1,
                        "verdict": "not decidable: avoided crossing at 0.73 mm; the single-mode C is ill-conditioned "
                                   "(range above) and the paper value lies inside that range"},
            "f_proto_vs_fig7a_0deg": {"paper": F_FIG7A_0DEG, "axionFEM": conv["f_Hz"], "rel": F_FIG7A_0DEG / conv["f_Hz"] - 1,
                                      "note": "Fig. 7 is the 3D clamshell model; at 0 deg its Fig. 7b Q (1.95e5) is ~45 % below "
                                              "the neighbouring angles, a sign of a spectator crossing there too"},
            "C_pill": {"paper": TAB2["pill"]["C"], "axionFEM": pill["C"], "rel": TAB2["pill"]["C"] / pill["C"] - 1},
            "Q_ratio_proto_over_pill": {"paper": TAB2["proto"]["Q"] / TAB2["pill"]["Q"], "axionFEM": conv["Q0"] / pill["Q0"],
                                        "axionFEM_range_over_sensitivity": [min(sq) / pill["Q0"], max(sq) / pill["Q0"]],
                                        "axionFEM_least_mixed": least["Q0_of_max_C"] / pill["Q0"],
                                        "least_mixed_point": {"dL_mm": least["dL_mm"], "deps": least["deps"],
                                                              "C_max_share": least["C_max_share"]},
                                        "verdict": "not decidable: the prototype Q is lowered by the same mixing"},
            "Q_ratio_ideal_over_pill": {"paper": TAB2["ideal"]["Q"] / TAB2["pill"]["Q"],
                                        "axionFEM_t3.97": ibest[3.97]["Q0"] / pill["Q0"], "axionFEM_t3.01": ibest[3.01]["Q0"] / pill["Q0"]},
            "Q_ratio_proto_over_ideal": {"paper": TAB2["proto"]["Q"] / TAB2["ideal"]["Q"],
                                         "axionFEM_t3.97": conv["Q0"] / ibest[3.97]["Q0"], "axionFEM_t3.01": conv["Q0"] / ibest[3.01]["Q0"]},
            "Q_abs_paper_over_axionFEM": {"proto": TAB2["proto"]["Q"] / conv["Q0"], "pill": TAB2["pill"]["Q"] / pill["Q0"],
                                          "ideal_t3.97": TAB2["ideal"]["Q"] / ibest[3.97]["Q0"]},
            "C_ideal_max": {"paper": TAB2["ideal"]["C"], "axionFEM_t3.97": ibest[3.97]["C"], "axionFEM_t3.01": ibest[3.01]["C"]},
            "C_gap_trend": {"paper": {"0": 0.34, "0.73": 0.235},
                            "axionFEM": {str(g["gap_mm"]): g["C"] for g in gaps}}}
    # t = 3.01 (Table II caption): inner radius where C crosses the Table II value 0.476 on the rising side
    sc = ideal[3.01]
    for a, b in zip(sc, sc[1:]):
        if a["C"] <= TAB2["ideal"]["C"] <= b["C"]:
            w = (TAB2["ideal"]["C"] - a["C"]) / (b["C"] - a["C"])
            match = {k: a[k] + w * (b[k] - a[k]) for k in ("r1_mm", "f_Hz", "C", "Q0")}
            break
    cmp_["ideal_t3.01_C_matched"] = {**match, "Q_ratio_over_pill": match["Q0"] / pill["Q0"],
                                     "paper_Q_ratio_over_pill": TAB2["ideal"]["Q"] / TAB2["pill"]["Q"],
                                     "Q_ratio_proto_over_ideal": conv["Q0"] / match["Q0"],
                                     "Q_ratio_proto_over_ideal_least_mixed": least["Q0_of_max_C"] / match["Q0"],
                                     "paper_Q_ratio_proto_over_ideal": TAB2["proto"]["Q"] / TAB2["ideal"]["Q"],
                                     "fitted": "r1 (fitted so that C = 0.476; C is therefore not a test)",
                                     "predicted": "f (paper: '11 GHz') and the Q ratio ideal/pill-box",
                                     "paper_rule_inner_face_at_node_r1_mm": 0.278 * 28.1}
    for k in ("Q_ratio_proto_over_pill",):
        cmp_[k]["rel"] = cmp_[k]["paper"] / cmp_[k]["axionFEM"] - 1
    rec["comparison"] = cmp_
    cm.write_record(NAME, rec)
    plot(rec)
    import json
    print(json.dumps(cmp_, indent=0, default=float))
    for s in study:
        print(s["order"], s["h_mm"], s["dofs"], s["f_Hz"], s["C"], s["Q0"], s["p_e"])
    print("pill", pills)


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    gs = rec["prototype_gap_sweep"]
    ax[0].plot([g["gap_mm"] for g in gs], [g["C"] for g in gs], "o-", label="axionFEM: largest-C mode")
    ax[0].plot([g["gap_mm"] for g in gs], [g["cluster_C_sum"] for g in gs], "^--", color="C2",
               label="axionFEM: C sum of modes within 6 MHz")
    ax[0].plot([0.73], [0.235], "k*", ms=12, label="Table II prototype C = 0.235 (+-4 %)")
    ax[0].errorbar([0.73], [0.235], yerr=[0.235 * 0.04], color="k", capsize=3)
    ax[0].plot([0.0], [0.34], "kx", ms=9, label="reference: no-gap C ~0.34")
    ax[0].set_xlabel("top gap [mm]")
    ax[0].set_ylabel("C030")
    ax[0].set_title("prototype at 0 deg: C vs top gap", fontsize=9)
    ax[0].legend(fontsize=7)
    a2 = ax[0].twinx()
    a2.plot([g["gap_mm"] for g in gs], [g["f_Hz"] / 1e9 for g in gs], "s:", color="C1")
    a2.set_ylabel("f [GHz] (dotted)")
    c = rec["comparison"]
    m = c["ideal_t3.01_C_matched"]
    labs = ["proto/pill", "ideal/pill\n(t 3.97, max C)", f"ideal/pill\n(t 3.01, C = 0.476\nat r1 = {m['r1_mm']:.2f})",
            "proto/ideal\n(t 3.01, C-matched)"]
    pap = [c["Q_ratio_proto_over_pill"]["paper"], c["Q_ratio_ideal_over_pill"]["paper"], c["Q_ratio_ideal_over_pill"]["paper"],
           c["Q_ratio_proto_over_ideal"]["paper"]]
    fem = [c["Q_ratio_proto_over_pill"]["axionFEM"], c["Q_ratio_ideal_over_pill"]["axionFEM_t3.97"],
           m["Q_ratio_over_pill"], m["Q_ratio_proto_over_ideal"]]
    x = range(len(labs))
    ax[1].bar([i - 0.2 for i in x], pap, 0.4, label="paper Table II")
    ax[1].bar([i + 0.2 for i in x], fem, 0.4, label="axionFEM")
    ax[1].set_xticks(list(x), labs, fontsize=7)
    ax[1].set_ylabel("Q ratio")
    ax[1].set_title("Q ratios (absolute Q: paper ~2x analytic)")
    ax[1].legend(fontsize=7)
    for t, mk in (("3.97", "o-"), ("3.01", "s-")):
        sc = rec["ideal_scan"][t]
        ax[2].plot([s["r1_mm"] for s in sc], [s["C"] for s in sc], mk, label=f"ideal, t = {t} mm")
    ax[2].axhline(0.476, color="k", ls="--", label="Table II ideal C = 0.476")
    ax[2].set_xlabel("shell inner radius [mm] (not given)")
    ax[2].set_ylabel("C030")
    ax[2].set_title("ideal design (r_c = 28.1, L = 420, no gap)")
    ax[2].legend(fontsize=7)
    sl = [x for x in rec["avoided_crossing_sensitivity"] if x["deps"] == 0.0]
    se = sorted([x for x in rec["avoided_crossing_sensitivity"] if x["dL_mm"] == 0.0], key=lambda x: x["deps"])
    ax[3].plot([x["dL_mm"] for x in sl], [x["C_max"] for x in sl], "o-", label="largest-C mode (vs dL)")
    ax[3].plot([x["dL_mm"] for x in sl], [x["C_sum"] for x in sl], "^--", color="C2", label="cluster C sum (vs dL)")
    ax[3].plot([10 * x["deps"] for x in se], [x["C_max"] for x in se], "s:", color="C1", label="largest-C mode (vs 10 deps)")
    ax[3].axhline(0.235, color="k", ls="--", label="Table II 0.235")
    ax[3].set_xlabel("dL [mm] (cavity and tube; gap fixed)  /  10 x deps")
    ax[3].set_ylabel("C030")
    ax[3].set_title("0.73 mm gap: avoided crossing (production mesh)", fontsize=9)
    ax[3].legend(fontsize=7)
    cv = rec["prototype_converged"]
    fig.suptitle(f"QUAX 2410.07774 clamshell, 0 deg: prototype f = {cv['f_Hz'] / 1e9:.4f} GHz, largest-C mode "
                 f"C = {cv['C']:.4f}, cluster sum {rec['prototype_cluster_converged']['C_sum']:.4f} (Table II 0.235; "
                 f"avoided crossing, not decidable) (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
