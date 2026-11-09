"""Reproduction: BI-RME probed copper cylinder, 2107.03137 Table I (beta).

    python benchmarks/cavity/repro/birme_tab1.py

D = 90 mm, L = 1 m, sigma = 6e7 S/m (classical skin), axial uniform B0. Table I: f, Q, C of TM010..TM015.
Refs README: only TM010 is a trustworthy target (f offset +0.35 % uniformly, R = 44.835 mm fits; TM01p (p >= 1) Q
and C are probe/noise affected). Runs:
  A. empty cylinder R = 45 mm (stated) and R = 44.835 mm (as-simulated fit), mesh study order 2 h = R/4, R/8,
     R/16 + order 3 R/8; TM010..TM015 f, Q, C.
  B. ambiguity sweep (the probe insertion depth is not given): axisymmetric model of the centred top-cap probe,
     inner pin radius 0.635 mm, PTFE-filled (eps 2.08) coax 0.635/2.11 mm, 10 mm long stub shorted at its end,
     pin protruding d = 0, 2, 5, 10, 15 mm into the cavity, at R = 44.835 mm; TM010 f, Q, C vs d.
Writes benchmarks/records/cavity_repro_birme_tab1.json and benchmarks/cavity/repro/birme_tab1.png.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "birme_tab1"
SIGMA = 6e7
L = 1.0
A_IN, B_OUT, EPS_PTFE, LSTUB = 0.635e-3, 2.11e-3, 2.08, 10e-3
DEPTHS = [0.0, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 10e-3, 15e-3]


def closed(R, p):
    f = cm.C_LIGHT / (2 * math.pi) * math.hypot(cm.CHI01 / R, p * math.pi / L)
    k = 2 * math.pi * f / cm.C_LIGHT
    G = cm.Z0 * cm.CHI01 * L / (2 * (R + L)) if p == 0 else cm.Z0 * k * R * L / (2 * (L + 2 * R))
    return f, G / cm.Rs_skin(f, SIGMA), (4 / cm.CHI01 ** 2 if p == 0 else 0.0)


def probe_design(tag, R, d, h, hp, order=2):
    z1 = L / 2
    outline = [[0, -z1], [R, -z1], [R, z1], [B_OUT, z1], [B_OUT, z1 + LSTUB], [A_IN, z1 + LSTUB], [A_IN, z1 - d],
               [0, z1 - d]]
    diel = [{"name": "ptfe", "shape": {"rect": {"r": [A_IN, B_OUT], "z": [z1, z1 + LSTUB]}}, "eps_r": EPS_PTFE,
             "mesh_size": hp}]
    ref = [{"name": "probe_zone", "shape": {"rect": {"r": [A_IN, 6e-3], "z": [z1 - d - 4e-3, z1]}}, "mesh_size": hp}]
    return cm.design(tag, outline, h=h, order=order, modes=6, families=("TM",), wall=cm.skin(SIGMA, 300.0),
                     dielectrics=diel, refine=ref, interior_point=[R / 2, 0.0], surface_size=h)


def main():
    rj = cm.ref("analytic_birme_tab1")
    tab = rj["reference_values"]["tables"][0]["rows"]
    jobs, meta = [], []
    for R in (0.045, 0.044835):
        outline = [[0, -L / 2], [R, -L / 2], [R, L / 2], [0, L / 2]]
        for order, div in ((2, 4), (2, 8), (2, 16), (3, 8)):
            tag = f"empty_R{R * 1e3:.3f}_o{order}_h{div}"
            jobs.append((tag, cm.design(tag, outline, h=R / div, order=order, modes=6, families=("TM",),
                                        wall=cm.skin(SIGMA, 300.0))))
            meta.append(("empty", R, order, div, None))
    R = 0.044835
    for d in DEPTHS:
        for hp, div in ((0.3e-3, 8), (0.15e-3, 16)):
            tag = f"probe_d{d * 1e3:.0f}mm_hp{hp * 1e3:.2f}"
            jobs.append((tag, probe_design(tag, R, d, R / div, hp)))
            meta.append(("probe", R, 2, div, (d, hp)))
    out = cm.solve_many(jobs, NAME)
    rec = {"benchmark": "BI-RME 2107.03137 Table I (TM01p of a probed Cu cylinder, D = 90 mm, L = 1 m)",
           "assumptions": {"sigma_S_m": SIGMA, "rf_model": "skin", "C": "eps-weighted, V = cavity volume"},
           "empty": [], "probe_sweep": []}
    for R in (0.045, 0.044835):
        runs = []
        for (tag, data, dt), (kind, RR, order, div, extra) in zip(out, meta):
            if kind != "empty" or RR != R:
                continue
            ms = sorted(cm.modes(data), key=lambda m: m["f_Hz"])
            run = {"order": order, "h_over_R": 1 / div, "dofs": cm.dofs(data), "seconds": dt, "modes": []}
            for p in range(6):
                fe, Qe, Ce = closed(R, p)
                m = min(ms, key=lambda x: abs(x["f_Hz"] - fe))
                run["modes"].append({"p": p, "f_Hz": m["f_Hz"], "rel_f_exact": m["f_Hz"] / fe - 1, "Q0": m["losses"]["Q0"],
                                     "rel_Q_exact": m["losses"]["Q0"] / Qe - 1, "C": m["form_factor"]["C"], "C_exact": Ce})
            runs.append(run)
        best = [r for r in runs if r["order"] == 3][0]
        comp = []
        for p, row in enumerate(tab):
            m = best["modes"][p]
            ft, Qt, Ct = float(row[2]) * 1e9, float(row[3]), float(row[4])
            comp.append({"mode": row[1], "f_axionFEM_Hz": m["f_Hz"], "f_table_Hz": ft, "f_rel": ft / m["f_Hz"] - 1,
                         "Q_axionFEM": m["Q0"], "Q_table": Qt, "Q_rel": Qt / m["Q0"] - 1, "C_axionFEM": m["C"],
                         "C_table": Ct, "C_rel": (Ct / m["C"] - 1) if m["C"] > 1e-12 else None,
                         "mesh_spread_f_rel": max(abs(r["modes"][p]["f_Hz"] / m["f_Hz"] - 1) for r in runs)})
        rec["empty"].append({"R_m": R, "runs": runs, "comparison": comp})
        print("R", R, [(c["mode"], f"{c['f_rel']:+.4%}", f"{c['Q_rel']:+.3%}", c["C_axionFEM"], c["C_table"]) for c in comp])
    sweep = {}
    for (tag, data, dt), (kind, RR, order, div, extra) in zip(out, meta):
        if kind != "probe":
            continue
        d, hp = extra
        fe = closed(RR, 0)[0]
        m = min(cm.modes(data), key=lambda x: abs(x["f_Hz"] - fe) if (x.get("form_factor", {}).get("C") or 0) > 0.3 else 1e30)
        sweep.setdefault(d, []).append({"hp_m": hp, "dofs": cm.dofs(data), "f_Hz": m["f_Hz"], "Q0": m["losses"]["Q0"],
                                        "C": m["form_factor"]["C"], "V_m3": data["fullwave"]["cavity"]["volume_m3"]})
    t0 = tab[0]
    for d in DEPTHS:
        rr = sorted(sweep[d], key=lambda x: -x["hp_m"])
        fin = rr[-1]
        rec["probe_sweep"].append({"depth_m": d, "runs": rr, "f_Hz": fin["f_Hz"], "Q0": fin["Q0"], "C": fin["C"],
                                   "f_table_rel": float(t0[2]) * 1e9 / fin["f_Hz"] - 1,
                                   "Q_table_rel": float(t0[3]) / fin["Q0"] - 1, "C_table_rel": float(t0[4]) / fin["C"] - 1,
                                   "mesh_change_f_rel": rr[-1]["f_Hz"] / rr[0]["f_Hz"] - 1,
                                   "mesh_change_C_rel": rr[-1]["C"] / rr[0]["C"] - 1})
        print("probe d", d, rec["probe_sweep"][-1]["f_Hz"], rec["probe_sweep"][-1]["Q0"], rec["probe_sweep"][-1]["C"])
    rec["radius_fit"] = radius_fit(tab)
    rec["probe_sweep_note"] = ("the shorted 10 mm PTFE stub is an artificial closed termination; the C and Q drop at "
                               "d = 5-6 mm is the onset of the stub's own resonance (strongly nonlinear in d), so the "
                               "model cannot say whether a real (matched) probe explains the 1 % C and Q deficits")
    print("radius fit", rec["radius_fit"])
    cm.write_record(NAME, rec)
    plot(rec)


def radius_fit(tab):
    """One-parameter least-squares fits of the six table TM01p frequencies to the closed form: R free (L = 1 m), or
    L free (R = 45 mm). R is FITTED here, so f agreement at the fitted R is not a test; the test is the residual."""
    from scipy.optimize import least_squares

    ft = [float(r[2]) * 1e9 for r in tab]

    def fR(R, Lx, p):
        return cm.C_LIGHT / (2 * math.pi) * math.hypot(cm.CHI01 / R, p * math.pi / Lx)
    out = {}
    for name, x0, fn in (("R_free", 0.045, lambda x, p: fR(x, L, p)), ("L_free", L, lambda x, p: fR(0.045, x, p))):
        sol = least_squares(lambda v: [fn(v[0], p) / ft[p] - 1 for p in range(len(ft))], [x0], x_scale=[x0])
        res = [fn(sol.x[0], p) / ft[p] - 1 for p in range(len(ft))]
        out[name] = {"fitted_m": float(sol.x[0]), "residuals_rel": res, "max_abs_residual_rel": max(abs(r) for r in res)}
    return out


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for e, mk in zip(rec["empty"], ("o", "s")):
        c = e["comparison"]
        p = list(range(len(c)))
        ax[0].plot(p, [100 * x["f_rel"] for x in c], mk + "-", label=f"f, R = {e['R_m'] * 1e3:.3f} mm")
        ax[0].plot(p, [100 * x["Q_rel"] for x in c], mk + "--", label=f"Q0, R = {e['R_m'] * 1e3:.3f} mm")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("p (TM01p)")
    ax[0].set_ylabel("(table - axionFEM)/axionFEM [%]")
    ax[0].set_title("empty cylinder vs Table I")
    ax[0].legend(fontsize=7)
    e = rec["empty"][0]
    for r in e["runs"]:
        if r["order"] == 2:
            ax[1].loglog([r["h_over_R"]] * 3, [abs(r["modes"][0]["rel_f_exact"]), abs(r["modes"][0]["rel_Q_exact"]),
                                               abs(r["modes"][0]["C"] / (4 / cm.CHI01 ** 2) - 1)], "ko", mfc="none")
    rr = [r for r in e["runs"] if r["order"] == 2]
    for i, (lab, key) in enumerate((("f", "rel_f_exact"), ("Q0", "rel_Q_exact"))):
        ax[1].loglog([r["h_over_R"] for r in rr], [abs(r["modes"][0][key]) for r in rr], "-", label=f"TM010 {lab} vs closed form")
    ax[1].set_xlabel("h / R")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("convergence, R = 45 mm, order 2")
    ax[1].legend(fontsize=7)
    ps = [x for x in rec["probe_sweep"] if x["depth_m"] <= 8.5e-3]  # d >= 10 mm: hybridized with the stub resonance
    dd = [x["depth_m"] * 1e3 for x in ps]
    ax[2].plot(dd, [100 * x["f_table_rel"] for x in ps], "o-", label="f: table vs FEM")
    ax[2].plot(dd, [100 * x["Q_table_rel"] for x in ps], "s-", label="Q0: table vs FEM")
    ax[2].plot(dd, [100 * x["C_table_rel"] for x in ps], "^-", label="C: table vs FEM")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("probe pin depth d [mm] (R = 44.835 mm)")
    ax[2].set_ylabel("(table - axionFEM)/axionFEM [%], TM010")
    ax[2].set_title("probe-depth sweep (closed 10 mm PTFE stub; d >= 10 mm\nhybridizes with the stub resonance, not shown)",
                    fontsize=9)
    ax[2].set_ylim(-2, 4)
    ax[2].legend(fontsize=7)
    fig.suptitle("BI-RME 2107.03137 Tab I, D = 90 mm, L = 1 m, sigma = 6e7 (beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=130)


if __name__ == "__main__":
    main()
