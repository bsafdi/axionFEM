#!/usr/bin/env python
"""Imported COMSOL models (axf import-comsol) vs the original COMSOL tables and the hand-built axionFEM
reproductions of arXiv:2211.00008 (benchmarks/beyond_mqs_2211).  beta.

    python compare_import.py --model toroid|solenoid|epoxy --comsol DIR --import RUN [RUN ...]
        [--label NAME] [--baseline RUN ...] [--extra LABEL=RUN[,RUN] ...] [--tag NAME]

COMSOL tables (e^{+jwt}): toroid/epoxy `source_impedance.npy`, `source_voltage.npy` ([f, Re, Im]; f in GHz for the
lossless toroid, MHz for the epoxy file); solenoid `Source_Impedance_*.txt`, `Port_Voltage_*.txt` (f in MHz).
`Vport` is loaded by Zref = 50 ohm: V_oc = Vport (Zport + 50)/50.  axionFEM (e^{-iwt}): X = -Z_im_ohm,
|V_oc| = w |Psi| (B0_T = 1 <-> Je = 1 A/m^2).  Near a pole the COMSOL curve is interpolated on its 1 kHz grid
(never extrapolated).  Writes axionfem_results/<tag>.json and a figure build/<tag>.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZREF = 50.0


def load_comsol(model: str, d: Path) -> dict:
    if model == "solenoid":
        def load(stem):
            a = np.concatenate([np.loadtxt(d / f"{stem}_{b}.txt", comments="%") for b in ("5_50", "50_150", "150_160")])
            a = a[np.argsort(a[:, 0], kind="stable")]
            _, i = np.unique(np.round(a[:, 0] * 1e3), return_index=True)
            a = a[i]
            return a[:, 0] * 1e6, a[:, 1] + 1j * a[:, 2]
        fz, z = load("Source_Impedance")
        fv, vl = load("Port_Voltage")
    else:
        unit = 1e9 if model == "toroid" else 1e6
        zf = np.load(d / "source_impedance.npy")
        vf = np.load(d / "source_voltage.npy")
        fz, z = zf[0] * unit, zf[1] + 1j * zf[2]
        fv, vl = vf[0] * unit, vf[1] + 1j * vf[2]
    m = (fv >= fz[0]) & (fv <= fz[-1])
    fv, vl = fv[m], vl[m]
    zv = np.interp(fv, fz, z.real) + 1j * np.interp(fv, fz, z.imag)
    return {"f": fz, "Z": z, "fV": fv, "Voc": vl * (zv + ZREF) / ZREF}


def interp(f0, f, y):
    out = np.interp(f0, f, y.real) + 1j * np.interp(f0, f, y.imag)
    return np.where((f0 < f[0]) | (f0 > f[-1]), np.nan + 0j, out)


def load_runs(dirs) -> list:
    rows = []
    for d in dirs:
        p = Path(d) / "results.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        info = {"elements": (r.get("mesh") or {}).get("elements"), "dofs": (r.get("mesh") or {}).get("dofs")}
        for s in (r.get("fullwave") or {}).get("sweep", []):
            w = 2 * math.pi * s["f_Hz"]
            rows.append({"f": s["f_Hz"], "X": -s["Z_im_ohm"], "R": s["Z_re_ohm"],
                         "absVoc": w * math.hypot(s.get("Psi_re", 0.0), s.get("Psi_im", 0.0)), **info})
    rows = sorted({round(r["f"]): r for r in rows}.values(), key=lambda r: r["f"])
    return rows


def poles(f, X):
    """Sign changes + -> - of the reactance between consecutive points (linear in 1/X)."""
    out = []
    for k in range(len(f) - 1):
        if X[k] > 0 > X[k + 1]:
            y0, y1 = 1 / X[k], 1 / X[k + 1]
            out.append(f[k] + (f[k + 1] - f[k]) * y0 / (y0 - y1))
    return out


def comsol_poles(c, fmax):
    f, x = c["f"], c["Z"].imag
    s = np.sign(x)
    idx = np.nonzero((s[:-1] > 0) & (s[1:] < 0) & (np.minimum(abs(x[:-1]), abs(x[1:])) > 10))[0]
    out = []
    for k in idx:
        if f[k] > fmax:
            break
        y0, y1 = 1 / x[k], 1 / x[k + 1]
        out.append(float(f[k] + (f[k + 1] - f[k]) * y0 / (y0 - y1)))
    return out


def table(rows, c):
    out = []
    for r in rows:
        zc = interp(np.array([r["f"]]), c["f"], c["Z"])[0]
        vc = interp(np.array([r["f"]]), c["fV"], c["Voc"])[0]
        w = 2 * math.pi * r["f"]
        out.append({"f_Hz": r["f"], "X_axf_ohm": r["X"], "X_comsol_ohm": float(zc.imag),
                    "L_axf_nH": r["X"] / w * 1e9, "L_comsol_nH": float(zc.imag) / w * 1e9,
                    "X_ratio": r["X"] / zc.imag if np.isfinite(zc.imag) and zc.imag else None,
                    "absVoc_axf_V": r["absVoc"], "absVoc_comsol_V": float(abs(vc)),
                    "Voc_ratio": r["absVoc"] / abs(vc) if np.isfinite(abs(vc)) and abs(vc) else None,
                    "R_axf_ohm": r["R"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["toroid", "solenoid", "epoxy"])
    ap.add_argument("--comsol", type=Path, required=True)
    ap.add_argument("--import", dest="imp", nargs="+", required=True, help="run dirs of the imported design")
    ap.add_argument("--label", default="import (mesh route)")
    ap.add_argument("--baseline", nargs="*", default=[], help="hand-built reproduction run dirs")
    ap.add_argument("--extra", nargs="*", default=[], metavar="LABEL=RUN[,RUN]")
    ap.add_argument("--fmax", type=float, default=150e6)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    c = load_comsol(a.model, a.comsol)
    sets = {a.label: load_runs(a.imp)}
    if a.baseline:
        sets["hand-built reproduction"] = load_runs(a.baseline)
    for spec in a.extra:
        lab, dirs = spec.split("=", 1)
        sets[lab] = load_runs(dirs.split(","))
    rec = {"model": a.model, "comsol_dir": Path(a.comsol).name, "beta": True,
           "comsol_poles_Hz": comsol_poles(c, a.fmax), "sets": {}}
    for lab, rows in sets.items():
        t = table(rows, c)
        p = poles([r["f"] for r in rows], [r["X"] for r in rows])
        rec["sets"][lab] = {"rows": t, "poles_between_points_Hz": p,
                            "elements": rows[0]["elements"] if rows else None, "dofs": rows[0]["dofs"] if rows else None}
    # imported vs hand-built at common frequencies
    if a.baseline:
        bi = {round(r["f"]): r for r in sets["hand-built reproduction"]}
        cmp = []
        for r in sets[a.label]:
            b = bi.get(round(r["f"]))
            if b:
                cmp.append({"f_Hz": r["f"], "X_rel": (r["X"] - b["X"]) / abs(b["X"]),
                            "Voc_rel": (r["absVoc"] - b["absVoc"]) / b["absVoc"] if b["absVoc"] else None})
        rec["import_vs_handbuilt"] = cmp
    out = HERE / "axionfem_results" / f"{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float))
    # print a table
    for lab, s in rec["sets"].items():
        print(f"== {lab} ({s['elements']} tets, {s['dofs']} DOFs); poles between points: "
              + ", ".join(f"{x / 1e6:.3f}" for x in s["poles_between_points_Hz"]))
        print("   f[MHz]   L_axf[nH]  L_com[nH]  X_ratio   |Voc|_axf  |Voc|_com  V_ratio")
        for r in s["rows"]:
            print(f"  {r['f_Hz'] / 1e6:7.2f}  {r['L_axf_nH']:10.3f} {r['L_comsol_nH']:10.3f}  "
                  f"{(r['X_ratio'] or float('nan')):8.5f}  {r['absVoc_axf_V']:9.5g}  {r['absVoc_comsol_V']:9.5g}  "
                  f"{(r['Voc_ratio'] or float('nan')):8.5f}")
    print("COMSOL poles [MHz]:", ", ".join(f"{x / 1e6:.3f}" for x in rec["comsol_poles_Hz"]))
    if a.baseline:
        print("import vs hand-built:", ", ".join(f"{x['f_Hz'] / 1e6:.1f}: X {x['X_rel']:+.2e} V {x['Voc_rel']:+.2e}"
                                                 for x in rec["import_vs_handbuilt"]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
        m = c["f"] <= a.fmax
        ax[0].plot(c["f"][m] / 1e6, c["Z"].imag[m], color="0.6", lw=0.8, label="COMSOL 2022 (table)")
        mv = c["fV"] <= a.fmax
        ax[1].semilogy(c["fV"][mv] / 1e6, abs(c["Voc"][mv]), color="0.6", lw=0.8, label="COMSOL 2022 (table)")
        for (lab, rows), mk in zip(sets.items(), ("o", "s", "^", "v", "D")):
            f = np.array([r["f"] for r in rows]) / 1e6
            ax[0].plot(f, [r["X"] for r in rows], mk, ms=5, mfc="none", label=lab)
            ax[1].semilogy(f, [r["absVoc"] for r in rows], mk, ms=5, mfc="none", label=lab)
        ax[0].set_ylim(-1500, 1500)
        ax[0].set_ylabel("Im Zport [ohm] (COMSOL sign)")
        ax[1].set_ylabel("|V_oc| [V] (Je = 1 A/m^2)")
        ax[1].set_xlabel("f [MHz]")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"{a.model}: axf import-comsol vs COMSOL (beta)")
        (HERE / "build").mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(HERE / "build" / f"{a.tag}.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("no figure:", e)


if __name__ == "__main__":
    main()
