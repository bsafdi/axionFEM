#!/usr/bin/env python
"""Modelling study, shielded lossless toroid (arXiv:2211.00008): one change at a time vs the mesh-C baseline and COMSOL.

Each variant is solved at the same few frequencies (5, 20, 40, 60, 96 MHz plus brackets 26.3/27.2 and 74.5/76.3 MHz
around the first two poles and 46.5/48.5 MHz around the first zero).  For each variant: L(5 MHz), rel. Im Z and
|V_oc| vs COMSOL at every frequency, and the bracketed pole/zero estimates (compare_toroid_comsol.axf_poles_zeros,
energy-identity slopes).  The baseline rows are taken from the production sweep at the same frequencies.

    python compare_toroid_study.py --comsol DIR --baseline RUN[,RUN...] --variant LABEL=RUN [...] [--tag NAME]
Writes axionfem_results/<tag>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from compare_toroid_comsol import (axf_poles_zeros, comsol_poles_zeros, interp_c, load_comsol, load_run,  # noqa: E402
                                   match_points, sweep_table)


def rows_of(spec):
    rows, info = [], []
    for d in spec.split(","):
        r, _, i = load_run(Path(d))
        rows += r
        info.append(dict(run=d, **i))
    return sorted(rows, key=lambda s: s["f"]), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comsol", type=Path, required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--variant", nargs="+", default=[], metavar="LABEL=RUN[,RUN]")
    ap.add_argument("--tag", default="toroid_lossless_modelling_study")
    a = ap.parse_args()
    c = load_comsol(a.comsol)
    poles, zeros = comsol_poles_zeros(c)
    out = {}
    specs = [("baseline (mesh C, 0.2 mm)", a.baseline)] + [tuple(v.split("=", 1)) for v in a.variant]
    fset = None
    for lab, spec in specs:
        rows, info = rows_of(spec)
        rows = [r for r in rows if r["accepted"]]
        if fset is None and lab.startswith("baseline"):
            pass
        out[lab] = dict(runs=info, rows=rows)
    fv = sorted({round(r["f"]) for lab, s in out.items() if not lab.startswith("baseline") for r in s["rows"]})
    res = {}
    for lab, s in out.items():
        rows = [r for r in s["rows"] if round(r["f"]) in fv] if fv else s["rows"]
        r0 = rows[0]
        v0 = interp_c(np.array([r0["f"]]), c["fV"], c["Voc"])[0]
        sign = float(np.sign(r0["Psi"].real * (v0 / 1j).real)) or 1.0
        tab = sweep_table(rows, c, sign)
        sp, sz = axf_poles_zeros(rows)
        res[lab] = dict(runs=s["runs"], n=len(rows),
                        L5_nH=next((t["L_axf_H"] * 1e9 for t in tab if abs(t["f_Hz"] - 5e6) < 1), None),
                        points=[dict(f_MHz=t["f_Hz"] / 1e6, rel_ImZ=t["rel_ImZ"], rel_absVoc=t["rel_absVoc"],
                                     ImZ_ohm=t["ImZ_axf_ohm"], absVoc_V=t["absVoc_axf_V"]) for t in tab],
                        poles=match_points(sp, poles),
                        zeros=match_points([z for z in sz if z["bracket_Hz"][1] - z["bracket_Hz"][0] <= 3e6], zeros))
    rec = dict(what="modelling study (one change at a time), shielded lossless toroid vs COMSOL 2022",
               comsol_L5_nH=float(np.interp(5e6, c["f"], c["Z"].imag) / (2 * np.pi * 5e6) * 1e9), variants=res)
    jp = HERE / "axionfem_results" / f"{a.tag}.json"
    jp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(f"COMSOL L(5 MHz) {rec['comsol_L5_nH']:.2f} nH")
    fl = sorted({p["f_MHz"] for v in res.values() for p in v["points"]})
    print("variant".ljust(30) + "L5[nH] " + " ".join(f"{f:>7.1f}" for f in fl))
    for lab, v in res.items():
        d = {p["f_MHz"]: p for p in v["points"]}
        print(lab[:30].ljust(30) + f"{v['L5_nH'] or float('nan'):7.2f} "
              + " ".join(f"{100 * d[f]['rel_ImZ']:+7.3f}" if f in d else "      -" for f in fl) + "  (ImZ %)")
        print(" " * 37 + " ".join(f"{100 * d[f]['rel_absVoc']:+7.3f}" if f in d else "      -" for f in fl) + "  (|V| %)")
        print(" " * 37 + "poles " + ", ".join(f"{p['f_Hz'] / 1e6:.3f} ({100 * p['rel']:+.3f}%)" for p in v["poles"])
              + "; zeros " + ", ".join(f"{p['f_Hz'] / 1e6:.3f} ({100 * p['rel']:+.3f}%)" for p in v["zeros"]))
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()
