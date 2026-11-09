"""BI-RME 2107.03137 Table I with the probe in the BI-RME convention (beta).

    python benchmarks/cavity/repro/birme_probe_port.py

Background (docs/validation/cavity_reproductions.md, BI-RME): at R = 44.835 mm the empty cylinder gives TM010
C = 0.69166 and Q0 = 33 410, against Table I (p. 18 of the arXiv PDF) C = 0.68505 and Q = 33 069 (both ~1 % low).
A probe model that closes the coax with a 10 mm shorted PTFE stub does not work: the stub's own resonance spoils the
sweep.

What BI-RME computes (Sec. 2, pp. 3-5, Eqs. 1-2): the field is expanded in the resonant modes E_m, H_m of the cavity
whose access-port surfaces are covered by PEC; the ports enter only through equivalent magnetic currents on those
surfaces. The Table I "modes" (f_m, Q_m with P_cm = (R_s/2) int_{S_V} |H_m|^2 dS over the cavity walls, Sec. 2.2,
and C_m, Eq. 23) are therefore those of the cavity with the coax port SHORTED AT THE PORT PLANE, not of the cavity
loaded by a matched line. The coax (inner radius 0.635 mm, outer 2.11 mm, eps_r 2.08, Z_c = 50 ohm) enters at the
centre of the top cap (p. 17). Neither the port-plane position nor the pin depth d is given; the probe was "designed
under critical coupling operation regime for the TM010 mode".

Model (axisymmetric, m = 0 TM): copper cylinder R = 44.835 mm (the radius fitted to the six Table I f),
L = 1 m, sigma = 6e7 S/m (skin); a copper pin (same lossy wall) r = 0.635 mm protruding d into the cavity from the top cap; the coax
annulus 0.635-2.11 mm closed by PEC at the port plane, s = 0 (at the wall plane) or s = 1 mm up a PTFE-filled
line (the latter to sample the TEM current cleanly). No matched termination: in the BI-RME convention none is
needed for f_m, Q_m, C_m.

Critical coupling (the only statement about d): in the shorted-port basis the port sees a series resonance at each
f_m; terminating the line in Z_c instead of the short extracts P_ext = |I|^2 Z_c / 2, with I the pin current at the
port plane (to first order in the coupling), so Q_ext = omega W / P_ext = 2 omega W / (|I|^2 Z_c), W the stored
energy. With the mode normalization of Results.cavity_mode_sample (int eps_r |E|^2 = 1, H physical,
W = eps0/2 for the peak-amplitude fields), Q_ext = omega eps0 / (|I|^2 Z_c), I = 2 pi r H_phi in the line. Critical
coupling: Q_ext(d) = Q0(d) defines d_c, at which C_m and Q_m are compared with Table I.
Runs: d = 0 ... 20 mm, two pin-zone meshes; all TM01p (p = 0-5) modes with f, C, Q and the C sum rule.
Writes benchmarks/records/cavity_repro_birme_probe_port.json and benchmarks/cavity/repro/birme_probe_port.png.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "birme_probe_port"
SIGMA = 6e7
L = 1.0
R = 0.044835
A_IN, B_OUT, EPS_PTFE = 0.635e-3, 2.11e-3, 2.08
ZC = cm.Z0 / (2 * math.pi * math.sqrt(EPS_PTFE)) * math.log(B_OUT / A_IN)
DEPTHS = [0.0, 2e-3, 4e-3, 6e-3, 8e-3, 10e-3, 12e-3, 14e-3, 16e-3, 20e-3]
EPS0 = 1 / (cm.MU0 * cm.C_LIGHT ** 2)


def design(tag, d, s, hp, div, order=2):
    z1 = L / 2
    if s > 0:
        outline = [[0, -z1], [R, -z1], [R, z1], [B_OUT, z1], [B_OUT, z1 + s], [A_IN, z1 + s], [A_IN, z1 - d],
                   [0, z1 - d]]
        diel = [{"name": "ptfe", "shape": {"rect": {"r": [A_IN, B_OUT], "z": [z1, z1 + s]}}, "eps_r": EPS_PTFE,
                 "mesh_size": hp}]
    else:
        outline = [[0, -z1], [R, -z1], [R, z1], [A_IN, z1], [A_IN, z1 - d], [0, z1 - d]] if d > 0 else \
            [[0, -z1], [R, -z1], [R, z1], [0, z1]]
        diel = []
    ref = [{"name": "probe_zone", "shape": {"rect": {"r": [0.0 if d == 0 else A_IN, 6e-3],
                                                    "z": [z1 - d - 4e-3, z1]}}, "mesh_size": hp}]
    if d > 0:  # also refine below the pin tip (r < A_IN), which the zone above leaves out
        ref.append({"name": "tip_zone", "shape": {"rect": {"r": [0.0, A_IN], "z": [z1 - d - 4e-3, z1 - d]}},
                    "mesh_size": hp})
    return cm.design(tag, outline, h=R / div, order=order, modes=8, families=("TM",), wall=cm.skin(SIGMA, 300.0),
                     dielectrics=diel, refine=ref, interior_point=[R / 2, 0.0], surface_size=R / div,
                     target_f_Hz=2.6e9)


def closed(p):
    f = cm.C_LIGHT / (2 * math.pi) * math.hypot(cm.CHI01 / R, p * math.pi / L)
    k = 2 * math.pi * f / cm.C_LIGHT
    G = cm.Z0 * cm.CHI01 * L / (2 * (R + L)) if p == 0 else cm.Z0 * k * R * L / (2 * (L + 2 * R))
    return f, G / cm.Rs_skin(f, SIGMA)


def solve_with_current(tag, d, s, hp, div, order=2):
    """Solve (cached JSON) and, when the run is redone here, sample the pin current of each mode."""
    from axionfem.design import Design
    wd = cm.WORK / NAME / tag
    cur = wd / "pin_current.json"
    doc = design(tag, d, s, hp, div, order).doc
    key = cm._key(doc)
    if cur.exists() and (wd / "design_key.txt").exists() and (wd / "design_key.txt").read_text().strip() == key:
        return json.loads((wd / "results.json").read_text()), json.loads(cur.read_text())
    wd.mkdir(parents=True, exist_ok=True)
    res = Design(doc).solve(wd, verbose=0)
    data = res.data
    (wd / "results.json").write_text(json.dumps(data))
    (wd / "design_key.txt").write_text(key)
    rows = res.cavity_modes()
    import numpy as np
    out = []
    for k, row in enumerate(rows):
        if s > 0:
            rr = np.linspace(A_IN + 0.2 * (B_OUT - A_IN), B_OUT - 0.2 * (B_OUT - A_IN), 5)
            zz = np.full_like(rr, L / 2 + 0.5 * s)
        else:  # just inside the cavity next to the pin base (or the cap centre for d = 0)
            rr = np.linspace(A_IN + 0.2 * (B_OUT - A_IN), B_OUT - 0.2 * (B_OUT - A_IN), 5)
            zz = np.full_like(rr, L / 2 - 0.05e-3)
        smp = res.cavity_mode_sample(row, rr, zz)
        rH = np.asarray(smp["flux"], dtype=float)
        out.append({"f_Hz": row["f_Hz"], "I_A": float(2 * math.pi * np.mean(rH)),
                    "I_spread_rel": float(np.ptp(rH) / max(abs(np.mean(rH)), 1e-300))})
    cur.write_text(json.dumps(out))
    return data, out


def one(args):
    return solve_with_current(*args)


def main():
    import concurrent.futures as cf
    cases = []
    for s in (0.0, 1e-3):
        for d in DEPTHS:
            for hp, div in ((0.3e-3, 8), (0.15e-3, 16)):
                cases.append((f"s{s * 1e3:.0f}_d{d * 1e3:.0f}_hp{hp * 1e3:.2f}", d, s, hp, div))
    nw = int(os.environ.get("AXF_REPRO_WORKERS", "1"))
    with cf.ProcessPoolExecutor(max_workers=max(1, nw)) as ex:
        outs = list(ex.map(one, cases))
    rj = cm.ref("analytic_birme_tab1")
    tab = rj["reference_values"]["tables"][0]["rows"]
    rec = {"benchmark": "BI-RME 2107.03137 Table I with a centred top-cap pin in the BI-RME (shorted-port) convention",
           "assumptions": {"R_m": R, "L_m": L, "sigma_S_m": SIGMA, "pin_radius_m": A_IN, "coax_outer_m": B_OUT,
                           "eps_ptfe": EPS_PTFE, "Z_c_ohm": ZC,
                           "Q_ext": "2 omega W / (|I|^2 Z_c), I = pin current at the port plane (shorted-port mode)"},
           "table_I": tab, "empty_closed_form": [dict(zip(("f_Hz", "Q0"), closed(p))) for p in range(6)], "runs": []}
    for (tag, d, s, hp, div), (data, cur) in zip(cases, outs):
        ms = sorted(cm.modes(data), key=lambda m: m["f_Hz"])
        modes = []
        for m, c in zip(ms, sorted(cur, key=lambda x: x["f_Hz"])):
            w = 2 * math.pi * m["f_Hz"]
            Qext = w * EPS0 / (c["I_A"] ** 2 * ZC) if c["I_A"] else None
            modes.append({"f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": m["losses"]["Q0"],
                          "I_A": c["I_A"], "I_spread_rel": c["I_spread_rel"], "Q_ext": Qext})
        # TM01p identification: the 6 modes below 2.7 GHz in frequency order (the pin adds no mode below 2.7 GHz
        # for d <= 20 mm: its quarter-wave resonance is ~3.7 GHz at d = 20 mm)
        tm = [m for m in modes if m["f_Hz"] < 2.70e9][:6]
        rec["runs"].append({"tag": tag, "d_m": d, "s_m": s, "hp_m": hp, "dofs": cm.dofs(data), "modes": modes,
                            "TM01p": tm, "C_sum_TM01p": sum(m["C"] for m in tm)})
    # critical coupling depth (fine mesh, both port planes). C changes like d^5-6 near d_c,
    # so the values at d_c are NOT interpolated: the grid only brackets d_c (log-linear in Q_ext/Q0), then d_c is
    # refined by secant steps on direct solves, and the mixing is reported from the direct solve at d_c, with direct
    # solves at d_c -+ 0.25 mm (the effect of a ~20 % error in the first-order Q_ext) as the uncertainty.
    def beta_at(d, s):
        tag = f"s{s * 1e3:.0f}_d{d * 1e6:.0f}um_hp0.15"
        data, cur = solve_with_current(tag, d, s, 0.15e-3, 16)
        ms = sorted(cm.modes(data), key=lambda m: m["f_Hz"])
        cs = sorted(cur, key=lambda x: x["f_Hz"])
        tm = [{"f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "Q0": m["losses"]["Q0"],
               "Q_ext": 2 * math.pi * m["f_Hz"] * EPS0 / (c["I_A"] ** 2 * ZC)}
              for m, c in zip(ms, cs) if m["f_Hz"] < 2.70e9][:6]
        return tm[0]["Q0"] / tm[0]["Q_ext"], tm
    crit = {}
    for s in (0.0, 1e-3):
        rr = sorted([r for r in rec["runs"] if r["s_m"] == s and r["hp_m"] == 0.15e-3 and r["d_m"] > 0],
                    key=lambda r: r["d_m"])
        pts = [(r["d_m"], math.log(r["TM01p"][0]["Q_ext"] / r["TM01p"][0]["Q0"])) for r in rr if r["TM01p"][0]["Q_ext"]]
        dc = None
        for (d0, y0), (d1, y1) in zip(pts, pts[1:]):
            if y0 > 0 >= y1:
                dc = d0 + (d1 - d0) * y0 / (y0 - y1)
        if dc is None:
            crit[f"s_{s * 1e3:.0f}mm"] = {"d_c_m": None}
            continue
        hist = []
        d_prev, lb_prev = None, None
        for _ in range(4):
            b, tm = beta_at(round(dc, 6), s)
            hist.append({"d_m": round(dc, 6), "beta": b})
            lb = math.log(b)
            if abs(lb) < 2e-3:
                break
            if d_prev is None:
                d_new = dc * math.exp(-lb / 5.0)  # beta ~ d^5-6 near d_c (grid)
            else:
                d_new = dc - lb * (dc - d_prev) / (lb - lb_prev)
            d_prev, lb_prev, dc = dc, lb, d_new
        dc = hist[-1]["d_m"]
        _, at = beta_at(dc, s)
        b_lo, lo = beta_at(round(dc - 0.25e-3, 6), s)
        b_hi, hi = beta_at(round(dc + 0.25e-3, 6), s)
        crit[f"s_{s * 1e3:.0f}mm"] = {
            "d_c_m": dc, "secant_history": hist, "TM01p_at_d_c": at,
            "TM01p_at_d_c_minus_0.25mm": lo, "TM01p_at_d_c_plus_0.25mm": hi, "beta_at_pm_0.25mm": [b_lo, b_hi],
            "C_sum_at_d_c": sum(m["C"] for m in at),
            "C010_table_minus_axionFEM_rel": float(tab[0][4]) / at[0]["C"] - 1,
            "Cp_table_over_axionFEM_minus_1": [float(tab[p][4]) / at[p]["C"] - 1 for p in range(6)],
            "Q_table_over_axionFEM_minus_1": [float(tab[p][3]) / at[p]["Q0"] - 1 for p in range(6)],
            "fraction_of_empty_C010_deficit_explained": (0.6916602761 - at[0]["C"]) / (0.6916602761 - float(tab[0][4]))}
        # depth at which each C_p equals Table I, log-quadratic through the three direct solves
        import numpy as np
        ds = [dc - 0.25e-3, dc, dc + 0.25e-3]
        cross = []
        for p in range(6):
            y = [math.log(m[p]["C"]) for m in (lo, at, hi)]
            cf = np.polyfit(np.array(ds) - dc, y, 2)
            roots = [r.real + dc for r in np.roots(cf - np.array([0, 0, math.log(float(tab[p][4]))]))
                     if abs(r.imag) < 1e-12 and abs(r.real) < 0.6e-3]
            cross.append(min(roots, key=lambda r: abs(r - dc)) if roots else None)
        crit[f"s_{s * 1e3:.0f}mm"]["d_where_C_p_equals_table_m"] = cross
        dfit = float(np.mean([c for c in cross[1:] if c is not None]))
        bfit, fitm = beta_at(round(dfit, 6), s)
        crit[f"s_{s * 1e3:.0f}mm"].update({
            "d_fit_to_p_ge_1_m": round(dfit, 6), "d_fit_spread_m": float(np.ptp([c for c in cross[1:] if c])),
            "beta_at_d_fit": bfit, "TM01p_at_d_fit": fitm,
            "C_sum_at_d_fit": sum(m["C"] for m in fitm), "C_sum_table": sum(float(tab[p][4]) for p in range(6)),
            "Cp_table_over_axionFEM_minus_1_at_d_fit": [float(tab[p][4]) / fitm[p]["C"] - 1 for p in range(6)],
            "Q_table_over_axionFEM_minus_1_at_d_fit": [float(tab[p][3]) / fitm[p]["Q0"] - 1 for p in range(6)]})
    rec["critical_coupling"] = crit
    p = cm.write_record(NAME, rec)
    print(p)
    for r in rec["runs"]:
        if r["hp_m"] == 0.15e-3:
            t0 = r["TM01p"][0]
            print(r["tag"], f"f {t0['f_Hz']:.6e} C {t0['C']:.5f} Q {t0['Q0']:.0f} Qext {t0['Q_ext']:.3e} "
                  f"C011 {r['TM01p'][1]['C']:.2e} Csum {r['C_sum_TM01p']:.5f}")
    print(json.dumps(crit, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = json.loads((cm.RECS / f"cavity_repro_{NAME}.json").read_text())
    tab = rec["table_I"]
    runs = [r for r in rec["runs"] if r["s_m"] == 1e-3 and r["hp_m"] == 0.15e-3]
    runs.sort(key=lambda r: r["d_m"])
    d = [r["d_m"] * 1e3 for r in runs]
    dc = rec["critical_coupling"]["s_1mm"]["d_c_m"] * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for p, col in zip(range(3), ("C0", "C1", "C2")):
        ax[0].semilogy(d, [max(r["TM01p"][p]["C"], 1e-9) for r in runs], "o-", color=col, label=f"TM01{p} (axionFEM)")
        ax[0].axhline(float(tab[p][4]), color=col, ls=":", label=f"Table I TM01{p}")
    ax[0].axvline(dc, color="k", ls="--", lw=0.8, label=f"critical coupling d_c = {dc:.2f} mm")
    ax[0].set_xlabel("pin depth d [mm]")
    ax[0].set_ylabel("C")
    ax[0].set_ylim(1e-6, 1.2)
    ax[0].legend(fontsize=7)
    ax[0].set_title("form factors vs pin depth (port shorted at 1 mm)")
    ax[1].semilogy(d[1:], [r["TM01p"][0]["Q_ext"] for r in runs[1:]], "o-", label="Q_ext (TM010, matched 50 ohm)")
    ax[1].semilogy(d, [r["TM01p"][0]["Q0"] for r in runs], "s-", label="Q0 (TM010)")
    ax[1].axhline(float(tab[0][3]), color="C1", ls=":", label="Table I Q (TM010)")
    ax[1].axvline(dc, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("pin depth d [mm]")
    ax[1].legend(fontsize=7)
    ax[1].set_title("critical coupling: Q_ext = Q0")
    row = rec["critical_coupling"]["s_1mm"]["TM01p_at_d_c"]
    ps = range(6)
    ax[2].semilogy(ps, [float(t[4]) for t in tab], "ks", label="Table I (BI-RME)")
    ax[2].semilogy(ps, [r["C"] for r in row], "ro", mfc="none", label=f"axionFEM at d_c")
    ax[2].semilogy(ps, [max(m["C"], 1e-9) for m in [r for r in rec["runs"] if r["tag"] == "s1_d0_hp0.15"][0]["TM01p"]],
                   "b^", mfc="none", label="axionFEM, no pin")
    ax[2].set_xlabel("p (TM01p)")
    ax[2].set_ylabel("C")
    ax[2].set_ylim(1e-6, 1.5)
    ax[2].legend(fontsize=7)
    ax[2].set_title("C of the six TM01p modes at critical coupling")
    fig.suptitle("BI-RME 2107.03137 Table I: centred top-cap pin, port shorted (BI-RME basis); beta", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm.HERE / f"{NAME}.png", dpi=110)


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
