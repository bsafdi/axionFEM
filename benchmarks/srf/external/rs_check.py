"""External check 3 (beta): axionFEM's Nb surface-resistance models against measured niobium data at 1.3 GHz.

(a) BCS + residual (`bcs`, default Nb parameters of docs/physics/surface_resistance.md: A = 2e-4 Ohm K at
    1.5 GHz, Delta/kTc = 17.67/9.25, Tc = 9.25 K) against R_s(Tc/T) of a 9-cell TESLA cavity, Aune et al.
    PRST-AB 3, 092001 (2000) Fig. 1 (digitized from the PDF vector markers: data/aune2000_fig1_rs.csv).
    The data's abscissa is Tc/T with Aune's Tc (9.2 K, p. 4: "critical temperature of only 9.2 K"), so T = 9.2/x.
    R_res is the one stated fit parameter (least squares in ln R_s over all points). Diagnostics: the model with
    R_res fitted, then the same with the BCS prefactor A also fitted, then A and Delta/kTc fitted (which parameter
    has to move to agree). Also the paper's quoted round numbers (800 nOhm at 4.2 K, 15 nOhm at 2 K, p. 4).
(b) Flux flow (`flux_flow`, free flux flow, current perpendicular to the vortices, s = 1) against the measured
    trapped-flux sensitivity dR_0/dB of 1.3 GHz Nb cavities, Gonnella, Kaufman & Liepe, arXiv:1509.04127,
    Table I (mean free path l, sensitivity in nOhm/mG). Material inputs per row: rho_n = rho*l / l with
    rho*l = 0.37e-15 Ohm m^2 (Gonnella Sec. II, citing their [15]); lambda = lambda_L sqrt(1 + xi0/l) and
    lambda_L = 36 nm, xi0 = 40 nm (Ciovati, arXiv:1501.07398, p. 7 and p. 12); B_c2 = phi0/(2 pi xi^2) (Gonnella
    Eq. 3), with two choices of the coherence length xi(l), since B_c2 needs the Ginzburg-Landau xi:
      "pippard": 1/xi = 1/xi0 + 1/l (Ciovati's nonlocal-kernel range; not the GL xi: it gives
                 B_c2 = 25 T at l = 4 nm);
      "gl":      xi = 0.74 xi0 / sqrt(1 + 0.75 xi0/l) (the standard GL interpolation, e.g. Tinkham; constants from the
                 textbook, not from a downloaded source). B = 1 mG (the sensitivity is linear in B here). Also Gonnella's
    Eq. 4 estimate, R_0 = B/(2 B_c2) R_n with R_n = 1.5 mOhm, B_c2 = 240 mT (their numbers, "~0.3 nOhm/mG").

    python benchmarks/srf/external/rs_check.py      -> benchmarks/srf/results/external_rs.json (+ figure)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from axionfem.cavity import surface_resistance  # noqa: E402

HERE = Path(__file__).parent
OUT = ROOT / "benchmarks/srf/results/external_rs.json"
F = 1.3e9
TC_AUNE = 9.2
NB = {"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 17.67 / 9.25, "Tc_K": 9.25}
PHI0 = 2.067833848e-15
# Gonnella et al. 1509.04127 Table I: (cavity/preparation, l [nm], dl, S [nOhm/mG], dS)
GONNELLA = [("LT1-3 990C over-doping", 4, 1, 3.2, 0.5), ("LT1-2 900C over-doping", 6, 1, 4.7, 0.6),
            ("LT1-2 N-doping + 6um VEP", 19, 6, 3.7, 0.9), ("LT1-3 N-doping + 12um VEP", 34, 10, 3.1, 0.5),
            ("LT1-1 N-doping + 18um VEP", 39, 12, 2.5, 0.6), ("LT1-4 N-doping + 24um VEP", 47, 14, 2.2, 0.2),
            ("LT1-5 N-doping + 30um VEP", 60, 18, 1.87, 0.08), ("LT1-5 N-doping + 40um VEP", 213, 64, 1.06, 0.02),
            ("NR1-3 100um VEP", 800, 100, 0.6, 0.1), ("NR1-3 100um VEP + 120C bake", 120, 365, 0.88, 0.07)]


def load_aune():
    rows, segs = [], []
    for ln in (HERE / "data/aune2000_fig1_rs.csv").read_text().splitlines():
        if ln.startswith("# R_BCS line segment"):
            a, b = ln.split(":", 1)[1].split("[")[0].split("->")
            pa, pb = (tuple(float(v) for v in s.strip(" ()").split(",")) for s in (a, b))
            if abs(pb[0] - pa[0]) > 0.05:
                segs.append((pa, pb))
        elif ln and ln[0].isdigit():
            rows.append(tuple(float(v) for v in ln.split(",")))
    return np.array(rows), segs


def bcs(T, R_res=0.0, A=None, D=None):
    rf = dict(NB, R_res_ohm=R_res, T_K=float(T))
    if A is not None:
        rf["A_ohm_K"] = A
    if D is not None:
        rf["Delta_over_kTc"] = D
    return surface_resistance(rf, F)


def part_a():
    from scipy.optimize import least_squares
    data, segs = load_aune()
    x, R = data[:, 0], data[:, 1] * 1e-9
    T = TC_AUNE / x

    def model(p, mode):
        Rres = math.exp(p[0])
        A = math.exp(p[1]) if mode >= 1 else None
        D = p[2] if mode >= 2 else None
        return np.array([bcs(t, Rres, A, D) for t in T])

    fits = {}
    for mode, name, p0 in ((0, "R_res", [math.log(3e-9)]), (1, "R_res+A", [math.log(3e-9), math.log(2e-4)]),
                           (2, "R_res+A+Delta", [math.log(3e-9), math.log(2e-4), NB["Delta_over_kTc"]])):
        sol = least_squares(lambda p: np.log(model(p, mode)) - np.log(R), p0)
        m = model(sol.x, mode)
        res = np.log(R / m)
        bcs_region = x < 5.0
        fits[name] = {"R_res_nOhm": math.exp(sol.x[0]) * 1e9,
                      "A_ohm_K": math.exp(sol.x[1]) if mode >= 1 else NB["A_ohm_K"],
                      "Delta_over_kTc": sol.x[2] if mode >= 2 else NB["Delta_over_kTc"],
                      "rms_ln_residual": float(np.sqrt(np.mean(res ** 2))),
                      "median_data_over_model_Tc/T<5": float(np.exp(np.median(res[bcs_region]))),
                      "data_over_model": [float(v) for v in np.exp(res)]}
    # the paper's own drawn R_BCS line: slope d ln R / d(Tc/T) and value at 4.2 K / 2 K (Tc = 9.2 K)
    xs = np.array([p for s in segs for p in (s[0][0], s[1][0])])
    ys = np.log(np.array([p for s in segs for p in (s[0][1], s[1][1])]))
    slope, icpt = np.polyfit(xs, ys, 1)   # straight-line slope (diagnostic only: the drawn line is curved)
    # the drawn line has the BCS form R ~ (1/T) exp(-b Tc/T): ln R = c + ln x - b x (x = Tc/T)
    (bb, cc), *_ = np.linalg.lstsq(np.column_stack([-xs, np.ones_like(xs)]), ys - np.log(xs), rcond=None)
    line = lambda xx: math.exp(cc + math.log(xx) - bb * xx)  # noqa: E731  nOhm
    line_rms = float(np.sqrt(np.mean((ys - np.log([line(v) for v in xs])) ** 2)))
    # model's own local slope at the same abscissa (pure BCS term, R_res = 0)
    xm = float(np.mean(xs))
    h = 1e-3
    mslope = (math.log(bcs(TC_AUNE / (xm + h))) - math.log(bcs(TC_AUNE / (xm - h)))) / (2 * h)
    quoted = {"4.2K": {"paper_nOhm": 800, "model_nOhm": bcs(4.2) * 1e9, "paper_line_nOhm": line(TC_AUNE / 4.2)},
              "2.0K": {"paper_nOhm": 15, "model_nOhm": bcs(2.0) * 1e9, "paper_line_nOhm": line(TC_AUNE / 2.0)}}
    return {"data": {"Tc_over_T": x.tolist(), "Rs_nOhm": (R * 1e9).tolist(), "T_K": T.tolist()},
            "fits": fits, "paper_line": {"slope_dlnR_dx": slope, "x_range": [float(xs.min()), float(xs.max())],
                                         "model_slope_same_x": mslope, "at_x": xm,
                                         "bcs_form_fit_Delta_over_kTc": float(bb), "bcs_form_fit_rms_ln": line_rms},
            "quoted": quoted}


def part_b():
    rows = []
    w = 2 * math.pi * F
    for (name, l, dl, S, dS), xmodel in [(g, xm) for xm in ("gl", "pippard") for g in GONNELLA]:
        lm = l * 1e-9
        rho = 0.37e-15 / lm
        lam = 36e-9 * math.sqrt(1 + 40.0 / l)
        xi = 1 / (1 / 40e-9 + 1 / lm) if xmodel == "pippard" else 0.74 * 40e-9 / math.sqrt(1 + 0.75 * 40.0 / l)
        bc2 = PHI0 / (2 * math.pi * xi * xi)
        B = 1e-7  # 1 mG
        rf = dict(NB, R_res_ohm=0.0, T_K=2.0, flux_flow={"Bc2_T": bc2, "rho_n_ohm_m": rho, "lambda_m": lam,
                                                         "f_depin_Hz": 0.0, "B_T": B, "orientation": 1.0})
        _, ff = surface_resistance(rf, F, flux_flow_part=True)
        # pinned-vortex reading: the depinning frequency f0 at which the model equals the measurement,
        # Re rho_v = rho_ff f^2/(f^2 + f0^2) in the low-rho limit -> f0 = f sqrt(model/meas - 1)
        ratio = ff * 1e9 / S
        f0 = F * math.sqrt(ratio - 1) if ratio > 1 else None
        rows.append({"xi_model": xmodel, "cavity": name, "l_nm": l, "dl_nm": dl, "S_meas_nOhm_per_mG": S, "dS": dS,
                     "rho_n_ohm_m": rho, "lambda_nm": lam * 1e9, "xi_nm": xi * 1e9, "Bc2_T": bc2,
                     "low_rho_limit_check": (rho * B / bc2) / (2 * lam) * 1e9,
                     "S_model_free_flux_flow_nOhm_per_mG": ff * 1e9, "model_over_meas": ratio,
                     "f_depin_to_match_Hz": f0})
    gon = (1e-7 / (2 * 0.240)) * 1.5e-3 * 1e9
    return {"rows": rows, "gonnella_eq4_estimate_nOhm_per_mG": gon}


def main():
    a, b = part_a(), part_b()
    out = {"provenance": {"aune": "Aune et al. PRST-AB 3 092001 (2000), arXiv:physics/0003011v1 Fig. 1 p.5, text p.4",
                          "gonnella": "Gonnella, Kaufman, Liepe arXiv:1509.04127v2 Table I, Sec. II",
                          "ciovati": "Ciovati arXiv:1501.07398 (lambda_L = 36 nm p. 7; xi0 ~ 40 nm p. 12)"},
           "bcs_vs_aune": a, "flux_flow_vs_gonnella": b}
    tmp = OUT.with_suffix(".json.tmp")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    tmp.replace(OUT)   # atomic
    for k, v in a["fits"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else None) for kk, vv in v.items() if kk != "data_over_model"})
    print("paper line slope", a["paper_line"]["slope_dlnR_dx"], "model", a["paper_line"]["model_slope_same_x"])
    print("quoted", a["quoted"])
    for r in b["rows"]:
        print(f"{r['xi_model']:8s}{r['cavity']:32s} l={r['l_nm']:4d} S_meas={r['S_meas_nOhm_per_mG']:5.2f} model={r['S_model_free_flux_flow_nOhm_per_mG']:6.2f} "
              f"ratio={r['model_over_meas']:5.2f} f0={r['f_depin_to_match_Hz']}")
    print("Gonnella Eq.4", b["gonnella_eq4_estimate_nOhm_per_mG"])
    try:
        plot(a, b)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def plot(a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x, R = np.array(a["data"]["Tc_over_T"]), np.array(a["data"]["Rs_nOhm"])
    ax[0].semilogy(x, R, "o", mfc="none", color="k", label="Aune 2000 Fig. 1 (digitized)")
    xx = np.linspace(2.5, 6.8, 200)
    for name, st in (("R_res", "-"), ("R_res+A", "--")):
        p = a["fits"][name]
        ax[0].semilogy(xx, [bcs(TC_AUNE / v, p["R_res_nOhm"] * 1e-9, p["A_ohm_K"]) * 1e9 for v in xx], st,
                       label=f"axionFEM bcs, fit {name}: R_res={p['R_res_nOhm']:.2f} nΩ" +
                       (f", A={p['A_ohm_K']:.2e}" if name != "R_res" else ""))
    ax[0].set_xlabel("$T_c/T$ ($T_c$ = 9.2 K)")
    ax[0].set_ylabel("$R_s$ [nΩ] at 1.3 GHz")
    ax[0].legend(fontsize=7)
    gl = [r for r in b["rows"] if r["xi_model"] == "gl"]
    pp = [r for r in b["rows"] if r["xi_model"] == "pippard"]
    l = np.array([r["l_nm"] for r in gl])
    ax[1].errorbar(l, [r["S_meas_nOhm_per_mG"] for r in gl], yerr=[r["dS"] for r in gl], fmt="o",
                   color="k", label="Gonnella 2015 Tab. I (measured)")
    ax[1].plot(l, [r["S_model_free_flux_flow_nOhm_per_mG"] for r in gl], "s", label="axionFEM flux_flow (free, s=1), GL ξ(ℓ)")
    ax[1].plot(l, [r["S_model_free_flux_flow_nOhm_per_mG"] for r in pp], "s", mfc="none", label="same, Pippard ξ(ℓ)")
    ax[1].axhline(b["gonnella_eq4_estimate_nOhm_per_mG"], ls=":", label="Gonnella Eq. 4 estimate")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("mean free path [nm]")
    ax[1].set_ylabel("dR/dB [nΩ/mG]")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    d = ROOT / "docs/srf/figures"
    fig.savefig(d / "external_rs.png", dpi=130)
    fig.savefig(d / "external_rs.pdf")


if __name__ == "__main__":
    main()
