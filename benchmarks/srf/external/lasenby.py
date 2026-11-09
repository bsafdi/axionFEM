"""External check 2 (beta): the closed-cylinder heterodyne pairs against Lasenby, arXiv:1912.11056 (PRD 102,
015008 (2020)), and Berlin et al. arXiv:1912.11048.

Quoted (verbatim locations, text of the downloaded PDF):
- Lasenby Sec. II B (p. 5-6): "drive TE012, and pick up in TM013, which are degenerate at d/R ~ 2.35. This mode pair
  has C01 = 0.19"; "the TE011/TM020 mode pair ... degenerate at d/R ~ 0.79"; "TE011 ... degenerate with TM111 (at
  all height-to-radius ratios)"; Sec. II B 1 (p. 6): "the wall electric fields for the cylindrical TE012 mode are
  everywhere zero"; Sec. III C (p. 14): "TE012 mode is exactly degenerate with the TM112 mode".
  C01 (Lasenby Eq. 5) = (int b.E1)^2 / (int E1^2 int b^2) = eta_10^2.
- Lasenby Table I (labelled IV in the text, p. 15): '60L cylinder' V 60 L, f 1.1 GHz, U_max 690 J, C01 U_max 132 J,
  P_max 27 W, Q0 1.8e11, Q1 9.5e10, B_rms 0.12 T; '900L cylinder' 920 L, 440 MHz, 11 kJ, 2 kJ, 170 W, same Q0, Q1,
  B_rms; with max wall |B| = 0.2 T, R_s = 5 nOhm, Q in the dissipation sense (P = w U/Q), B_rms averaged over time
  and space; R = 20 cm and 50 cm (p. 6, "properties for R = 20 cm and R = 50 cm").
- Lasenby Eq. 24 (p. 9): int dA sin(2 phi) sin(2 pi z/d) sin(3 pi z/d) n.dx = 0.09 C A_w x for a vibration
  dx = sin 2phi sin(2 pi z/d) sin(3 pi z/d) x(t) (C = 1).
- Berlin et al. Sec. IV (p. 9): "overlap factors of eta_10 ~ 0.5 ... for the transitions TM0,n+1,0 <-> TE0n1".

Ours: closed forms (axionfem.srf.cylinder_pair) and an FE-only degeneracy: d/R found by Brent's method on 2D
axisymmetric solves (no closed form used), then eta_10, B_pk/B0, G from the 2D fields at that d/R.

    python benchmarks/srf/external/lasenby.py     -> benchmarks/srf/results/external_lasenby.json
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from axionfem import srf  # noqa: E402
from axionfem.cavity import C_LIGHT, MU0, pillbox_design  # noqa: E402

OUT = ROOT / "benchmarks/srf/results/external_lasenby.json"
TABLE = {"60L": {"R_m": 0.20, "V_L": 60, "f_Hz": 1.1e9, "U_J": 690, "C01U_J": 132, "P_W": 27, "Q0": 1.8e11, "Q1": 9.5e10,
                 "Brms_T": 0.12},
         "900L": {"R_m": 0.50, "V_L": 920, "f_Hz": 440e6, "U_J": 11e3, "C01U_J": 2e3, "P_W": 170, "Q0": 1.8e11,
                  "Q1": 9.5e10, "Brms_T": 0.12}}
BWALL, RS = 0.2, 5e-9


def table_rows(cp):
    """Lasenby Table IV from the closed forms: B0 (spatial rms of the amplitude) = B_wall/(B_pk/B0); time-and-space
    rms = B0/sqrt2; U = B0^2 V/(2 mu0); Q = G/R_s; P = w U/Q0."""
    out = {}
    for k, ref in TABLE.items():
        R = ref["R_m"]
        L = cp["L_over_R"] * R
        V = math.pi * R * R * L
        f = C_LIGHT * cp["pump"]["kR"] / R / (2 * math.pi)
        B0 = BWALL / cp["Bpk_over_B0"]
        U = B0 ** 2 * V / (2 * MU0)
        Q0, Q1 = cp["G_pump_ohm"] / RS, cp["G_signal_ohm"] / RS
        ours = {"V_L": V * 1e3, "f_Hz": f, "U_J": U, "C01U_J": cp["eta_10"] ** 2 * U, "P_W": 2 * math.pi * f * U / Q0,
                "Q0": Q0, "Q1": Q1, "Brms_T": B0 / math.sqrt(2)}
        out[k] = {"lasenby": ref, "ours": ours, "ratio": {q: ours[q] / ref[q] for q in ours}}
    return out


def fe_degeneracy(pair, a=0.1, n=40, order=3):
    """d/R from 2D solves only (Brent on f_signal - f_pump), then figures of merit at that d/R."""
    from scipy.optimize import brentq
    pump, signal = srf.CYLINDER_PAIRS[pair]
    pn, sn = f"{pump[0]}0{pump[1]}{pump[2]}", f"{signal[0]}0{signal[1]}{signal[2]}"
    hist = []
    ftarget = C_LIGHT * srf.cylinder_pair_geometry(pump, signal)["kR"] / a / (2 * math.pi)  # shift-invert centre only
    work = tempfile.mkdtemp(prefix="las_")

    def design(Lr):
        L = Lr * a
        d = pillbox_design(a, L, h=a / n, order=order, modes=10, target_f_Hz=ftarget, families=("TE", "TM"))
        wall = [(0.0, -L / 2), (a, -L / 2), (a, L / 2), (0.0, L / 2)]
        d.srf_contour = {"points": wall, "wall": wall, "V_m3": math.pi * a * a * L,
                         "A_m2": 2 * math.pi * a * L + 2 * math.pi * a * a, "length": L, "z_end": L / 2,
                         "equator_radius": a, "params": {}}
        return d

    def solve(Lr):
        d = design(Lr)
        r = d.solve(f"{work}/L{Lr:.9f}", verbose=0)
        lab = srf.classify_modes(r, d.srf_contour)
        return d, r, lab

    def df(Lr):
        _, _, lab = solve(Lr)
        fp, fs = srf.find_mode(None, pn, labels=lab)["f_Hz"], srf.find_mode(None, sn, labels=lab)["f_Hz"]
        hist.append((Lr, fp, fs))
        return fs - fp

    g = srf.cylinder_pair_geometry(pump, signal)["L_over_R"]
    br = (0.9 * g, 1.1 * g)    # bracket only; the root is located from the FE frequencies
    Lr = brentq(df, *br, xtol=1e-7)
    d, r, lab = solve(Lr)
    fm = srf.figures_of_merit(r, pn, sn, contour=d.srf_contour)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return {"pair": pair, "d_over_R_fe": Lr, "h_over_R": 1 / n, "order": order, "eta_10": fm["eta_10"],
            "C01": fm["eta_10"] ** 2, "Bpk_over_B0": fm["Bpk_over_B0"], "Epk_over_cB0": fm["Epk_over_cB0"],
            "G_pump_ohm": fm["G_pump_ohm"], "G_signal_ohm": fm["G_signal_ohm"], "history": hist}


def eq24_coefficient(Lr):
    """int over all walls of sin^2(2phi) sin^2(2 pi z/d) sin^2(3 pi z/d) dA / A_w (only the barrel has n.dx != 0)."""
    from scipy.integrate import quad
    zint = quad(lambda u: math.sin(2 * math.pi * u) ** 2 * math.sin(3 * math.pi * u) ** 2, 0, 1)[0]  # per unit d
    barrel = 2 * math.pi * 1.0 * Lr * 0.5 * zint
    Aw = 2 * math.pi * Lr + 2 * math.pi
    return barrel / Aw


def main():
    out = {"closed_form": {}, "fe": {}}
    for pair in ("TE012/TM013", "TM020/TE011", "TM030/TE021"):
        cp = srf.cylinder_pair(pair)
        out["closed_form"][pair] = {k: cp[k] for k in ("L_over_R", "xi", "eta_10", "Bpk_over_B0", "Epk_over_cB0",
                                                      "G_pump_ohm", "G_signal_ohm")}
        out["closed_form"][pair]["C01"] = cp["eta_10"] ** 2
        if pair == "TE012/TM013":
            out["table_IV"] = table_rows(cp)
            out["eq24_coefficient"] = {"lasenby": 0.09, "ours": eq24_coefficient(cp["L_over_R"])}
        x = fe_degeneracy(pair)
        out["fe"][pair] = x
        print(pair, "closed", round(cp["L_over_R"], 6), round(cp["eta_10"], 6), "FE", round(x["d_over_R_fe"], 6),
              round(x["eta_10"], 6), flush=True)
    out["quotes"] = {"TE012/TM013": {"d_over_R": 2.35, "C01": 0.19}, "TM020/TE011": {"d_over_R": 0.79},
                     "Berlin_TM0n+1,0/TE0n1_eta": 0.5}
    tmp = OUT.with_suffix(".json.tmp")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    tmp.replace(OUT)   # atomic
    for k, v in out["table_IV"].items():
        print(k, {q: round(r, 3) for q, r in v["ratio"].items()})
    print("eq24", out["eq24_coefficient"])


if __name__ == "__main__":
    main()
