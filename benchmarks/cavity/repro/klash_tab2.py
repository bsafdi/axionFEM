"""Reproduction: KLASH empty copper cylinder, 1707.06010 Table 2 (beta).

    python benchmarks/cavity/repro/klash_tab2.py

R = 2 m, L = 4 m and 3.5 m, copper at 300 K. sigma is not stated; 5.8e7 S/m (IACS) is assumed (refs JSON),
classical skin effect (rf model `skin`). Axisymmetric: TM010 f and Q0 (and C010). TE111 (m = 1) is for 3D.
Mesh study: order 2 at h = R/4, R/8, R/16 and order 3 at R/8.
Writes benchmarks/records/cavity_repro_klash_tab2.json and benchmarks/cavity/repro/klash_tab2.png.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "klash_tab2"
SIGMA = 5.8e7
R = 2.0


def main():
    rj = cm.ref("analytic_klash_tab2")
    tab = rj["reference_values"]["tables"][0]["rows"]
    tm = [r for r in tab if r[0] == "TM010"][0]
    table = {4.0: (float(tm[1]) * 1e6, float(tm[2].replace(",", ""))), 3.5: (float(tm[3]) * 1e6, float(tm[4].replace(",", "")))}
    jobs, meta = [], []
    for L in (4.0, 3.5):
        outline = [[0, -L / 2], [R, -L / 2], [R, L / 2], [0, L / 2]]
        for order, div in ((2, 4), (2, 8), (2, 16), (3, 8)):
            tag = f"L{L}_o{order}_h{div}"
            jobs.append((tag, cm.design(tag, outline, h=R / div, order=order, modes=3, families=("TM",),
                                        wall=cm.skin(SIGMA, 300.0))))
            meta.append((L, order, div))
    out = cm.solve_many(jobs, NAME)
    rec = {"benchmark": "KLASH 1707.06010 Table 2, TM010 of the empty copper cylinder (R = 2 m)",
           "assumptions": {"sigma_S_m": SIGMA, "rf_model": "skin (classical)", "T_K": 300}, "cases": []}
    for L in (4.0, 3.5):
        f0 = cm.C_LIGHT * cm.CHI01 / (2 * math.pi * R)
        G0 = cm.Z0 * cm.CHI01 * L / (2 * (R + L))
        Q0 = G0 / cm.Rs_skin(f0, SIGMA)
        runs = []
        for (tag, data, dt), (LL, order, div) in zip(out, meta):
            if LL != L:
                continue
            m = min(cm.modes(data), key=lambda x: abs(x["f_Hz"] - f0))
            lo = m["losses"]
            runs.append({"order": order, "h_over_R": 1 / div, "dofs": cm.dofs(data), "seconds": dt, "f_Hz": m["f_Hz"],
                         "rel_f_exact": m["f_Hz"] / f0 - 1, "G_ohm": lo["G_ohm"], "rel_G_exact": lo["G_ohm"] / G0 - 1,
                         "Q0": lo["Q0"], "C": m["form_factor"]["C"]})
        best = [r for r in runs if r["order"] == 3][0]
        ft, Qt = table[L]
        # printed f has 4 significant figures (57.37 MHz: +-0.005 MHz); Q has 3 (+-500)
        conv = {"f_Hz": best["f_Hz"], "f_exact_Hz": f0, "f_table_Hz": ft, "f_table_minus_FEM_rel": ft / best["f_Hz"] - 1,
                "f_table_tol_rel": 0.005e6 / ft, "Q0": best["Q0"], "Q0_exact": Q0, "Q_table": Qt,
                "Q_table_minus_FEM_rel": Qt / best["Q0"] - 1, "Q_table_tol_rel": 500 / Qt,
                "C010": best["C"], "C010_exact": 4 / cm.CHI01 ** 2,
                "mesh_spread_f_rel": max(abs(r["f_Hz"] / best["f_Hz"] - 1) for r in runs),
                "mesh_spread_Q_rel": max(abs(r["Q0"] / best["Q0"] - 1) for r in runs)}
        conv["f_within_table_precision"] = abs(conv["f_table_minus_FEM_rel"]) <= conv["f_table_tol_rel"]
        conv["Q_within_table_precision"] = abs(conv["Q_table_minus_FEM_rel"]) <= conv["Q_table_tol_rel"]
        rec["cases"].append({"L_m": L, "runs": runs, "converged": conv})
        print(L, {k: v for k, v in conv.items()})
    c4, c35 = rec["cases"][0]["converged"], rec["cases"][1]["converged"]
    rec["Q_ratio_L4_over_L3.5"] = {"table": c4["Q_table"] / c35["Q_table"], "axionFEM": c4["Q0"] / c35["Q0"],
                                   "table_rounding_tol": 500 / c4["Q_table"] + 500 / c35["Q_table"],
                                   "note": "sigma-independent; the absolute Q only checks the sigma = 5.8e7 that refs "
                                           "inferred from this same table (circular)"}
    print("Q ratio", rec["Q_ratio_L4_over_L3.5"])
    rec["not_done"] = "TE111 row (m = 1): needs the 3D solver"
    cm.write_record(NAME, rec)
    plot(rec)


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for c, col in zip(rec["cases"], ("C0", "C1")):
        rr = [r for r in c["runs"] if r["order"] == 2]
        r3 = [r for r in c["runs"] if r["order"] == 3][0]
        ax[0].loglog([r["h_over_R"] for r in rr], [abs(r["rel_f_exact"]) for r in rr], "o-", color=col,
                     label=f"L = {c['L_m']} m, f")
        ax[0].loglog([r["h_over_R"] for r in rr], [abs(r["rel_G_exact"]) for r in rr], "s--", color=col,
                     label=f"L = {c['L_m']} m, G = Q0 Rs")
        ax[0].loglog([r3["h_over_R"]] * 2, [abs(r3["rel_f_exact"]), abs(r3["rel_G_exact"])], "*", color=col, ms=10)
        v = c["converged"]
        ax[1].errorbar([0 + (c["L_m"] == 3.5) * 0.2], [100 * v["f_table_minus_FEM_rel"]], yerr=[100 * v["f_table_tol_rel"]],
                       fmt="o", color=col, capsize=3, label=f"L = {c['L_m']} m")
        ax[1].errorbar([1 + (c["L_m"] == 3.5) * 0.2], [100 * v["Q_table_minus_FEM_rel"]], yerr=[100 * v["Q_table_tol_rel"]],
                       fmt="s", color=col, capsize=3)
    ax[0].set_xlabel("h / R")
    ax[0].set_ylabel("relative error vs closed form")
    ax[0].set_title("convergence (order 2; star: order 3)")
    ax[0].legend(fontsize=7)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks([0.1, 1.1], ["f TM010", "Q0 TM010 (sigma = 5.8e7)"])
    ax[1].set_ylabel("(table - axionFEM)/axionFEM [%]")
    ax[1].set_title("Table 2 vs axionFEM (bars: printed precision)")
    ax[1].legend(fontsize=7)
    fig.suptitle("KLASH 1707.06010 Tab 2, empty Cu cylinder R = 2 m (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
