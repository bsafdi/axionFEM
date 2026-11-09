"""Reproduction: ADMX empty-cylinder frequencies, Braine thesis 2408.03444 Table 6.1 (beta).

    python benchmarks/cavity/repro/admx_tab6_1.py

Axisymmetric m = 0 modes only: TM010, TM011, TM020 (+ C010, C020). TE111 (m = 1) is left for the 3D solver.
References: the closed form (exact) and the table (rounded to 2-4 significant figures; tolerance = half a unit of
the last printed digit of f combined with the rounding of R and H, refs README: +-0.24 % .. 0.31 %).
Mesh study: order 2 at h = R/4, R/8, R/16 and order 3 at R/8.
Writes benchmarks/records/cavity_repro_admx_tab6_1.json and benchmarks/cavity/repro/admx_tab6_1.png.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "admx_tab6_1"
MODES = ["TM010", "TM011", "TM020"]


def exact(R, H):
    f = lambda x, p: cm.C_LIGHT / (2 * math.pi) * math.hypot(x / R, p * math.pi / H)  # noqa: E731
    return {"TM010": f(cm.CHI01, 0), "TM011": f(cm.CHI01, 1), "TM020": f(cm.CHI02, 0)}


def n_tm_below(R, H, fmax):
    n = 0
    for x in (cm.CHI01, cm.CHI02, cm.CHI03):
        p = 0
        while cm.C_LIGHT / (2 * math.pi) * math.hypot(x / R, p * math.pi / H) <= fmax:
            n += 1
            p += 1
    return n


def pick(ms, f):
    return min(ms, key=lambda m: abs(m["f_Hz"] - f))


def tol_table(val_str, R_cm_str, H_cm_str=None, kz2_over_k2=0.0):
    """Relative rounding tolerance of a printed f: half a unit in the last digit of f plus the propagated half units
    of R and H: df/f = (kr/k)^2 dR/R + (kz/k)^2 dH/H (kz = 0 for p = 0, so only R enters there)."""
    def half_ulp(s):
        s = s.strip()
        dec = len(s.split(".")[1]) if "." in s else 0
        return 0.5 * 10 ** (-dec) / float(s)
    tol = half_ulp(val_str) + (1 - kz2_over_k2) * half_ulp(R_cm_str)
    if H_cm_str is not None:
        tol += kz2_over_k2 * half_ulp(H_cm_str)
    return tol


def main():
    rj = cm.ref("analytic_admx_thesis_tab6_1")
    rows = rj["reference_values"]["tables"][0]["rows"]
    cols = rj["reference_values"]["tables"][0]["columns"]
    jobs, meta = [], []
    for row in rows:
        cav, R, H = row[0], float(row[1]) / 100, float(row[2]) / 100
        ex = exact(R, H)
        nm = n_tm_below(R, H, 1.001 * ex["TM020"]) + 2
        outline = [[0, -H / 2], [R, -H / 2], [R, H / 2], [0, H / 2]]
        for order, div in ((2, 4), (2, 8), (2, 16), (3, 8)):
            h = R / div
            tag = f"{cav.replace(' ', '_')}_o{order}_h{div}"
            d = cm.design(tag, outline, h=h, order=order, modes=nm, families=("TM",))
            jobs.append((tag, d))
            meta.append((cav, R, H, order, div, h))
    out = cm.solve_many(jobs, NAME)
    rec = {"benchmark": "ADMX empty cylinders, 2408.03444 Table 6.1 (m = 0 modes)", "cavities": []}
    by = {}
    for (tag, data, dt), (cav, R, H, order, div, h) in zip(out, meta):
        ex = exact(R, H)
        ms = cm.modes(data)
        run = {"order": order, "h_m": h, "h_over_R": 1 / div, "dofs": cm.dofs(data), "seconds": dt}
        for k in MODES:
            m = pick(ms, ex[k])
            run[k] = {"f_Hz": m["f_Hz"], "rel_exact": m["f_Hz"] / ex[k] - 1, "C": m["form_factor"]["C"],
                      "residual_true": m["residual_true"]}
        by.setdefault(cav, {"R_m": R, "H_m": H, "exact_Hz": ex, "runs": []})["runs"].append(run)
    for row in rows:
        cav = row[0]
        c = by[cav]
        best = [r for r in c["runs"] if r["order"] == 3][0]
        c["converged"] = {}
        for k in MODES:
            s = row[cols.index(f"f_{k}")]
            fe = best[k]["f_Hz"]
            tb = float(s) * 1e9
            kz2 = (math.pi / c["H_m"]) ** 2 / (2 * math.pi * c["exact_Hz"][k] / cm.C_LIGHT) ** 2 if k == "TM011" else 0.0
            tol = tol_table(s, row[1], row[2], kz2)
            c["converged"][k] = {"f_axionFEM_Hz": fe, "f_exact_Hz": c["exact_Hz"][k], "rel_vs_exact": fe / c["exact_Hz"][k] - 1,
                                 "f_table_Hz": tb, "table_minus_FEM_rel": tb / fe - 1, "table_rounding_tol_rel": tol,
                                 "within_table_precision": abs(tb / fe - 1) <= tol,
                                 "mesh_spread_rel": max(abs(r[k]["f_Hz"] / fe - 1) for r in c["runs"])}
        c["converged"]["C010"] = {"C": best["TM010"]["C"], "exact": 4 / cm.CHI01 ** 2,
                                  "rel": best["TM010"]["C"] / (4 / cm.CHI01 ** 2) - 1}
        c["converged"]["C020"] = {"C": best["TM020"]["C"], "exact": 4 / cm.CHI02 ** 2,
                                  "rel": best["TM020"]["C"] / (4 / cm.CHI02 ** 2) - 1}
        c["converged"]["C011"] = {"C": best["TM011"]["C"], "exact": 0.0}
        rec["cavities"].append({"cavity": cav, **c})
    rec["not_done"] = "TE111 column (m = 1): needs the 3D solver; not attempted here"
    cm.write_record(NAME, rec)
    plot(rec)
    for c in rec["cavities"]:
        print(c["cavity"], {k: (round(v["f_axionFEM_Hz"] / 1e9, 6), f"{v['table_minus_FEM_rel']:+.2e}",
                                v["within_table_precision"]) for k, v in c["converged"].items() if k.startswith("TM")})


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    mk = {"TM010": "o", "TM011": "s", "TM020": "^"}
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, c in enumerate(rec["cavities"]):
        for k in MODES:
            rr = [r for r in c["runs"] if r["order"] == 2]
            ax[0].loglog([r["h_over_R"] for r in rr], [max(abs(r[k]["rel_exact"]), 1e-16) for r in rr],
                         marker=mk[k], color=cols[i], lw=1, label=f"{c['cavity']} {k}" if k == "TM010" else None)
            r3 = [r for r in c["runs"] if r["order"] == 3][0]
            ax[0].loglog([r3["h_over_R"]], [max(abs(r3[k]["rel_exact"]), 1e-16)], marker=mk[k], color=cols[i],
                         mfc="none", ms=9)
    ax[0].set_xlabel("h / R")
    ax[0].set_ylabel("|f_FEM / f_exact - 1|")
    ax[0].set_title("convergence (order 2 lines; order 3 open markers)\no TM010, s TM011, ^ TM020")
    ax[0].legend(fontsize=7)
    x = 0
    ticks, labels = [], []
    for i, c in enumerate(rec["cavities"]):
        for k in MODES:
            v = c["converged"][k]
            ax[1].errorbar([x], [100 * v["table_minus_FEM_rel"]], yerr=[100 * v["table_rounding_tol_rel"]],
                           fmt=mk[k], color=cols[i], capsize=3)
            ticks.append(x)
            labels.append(f"{c['cavity'].replace('ADMX ', '')}\n{k}")
            x += 1
        x += 0.5
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(ticks, labels, fontsize=6, rotation=90)
    ax[1].set_ylabel("(f_table - f_axionFEM) / f_axionFEM  [%]")
    ax[1].set_title("Table 6.1 vs axionFEM (bars: rounding of f and R)")
    fig.suptitle("ADMX thesis Tab 6.1, empty cylinders, m = 0 (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
