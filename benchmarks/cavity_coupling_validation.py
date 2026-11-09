"""Validation of the receiver coupling coefficient beta of a coaxial antenna port: driven fit vs eigenmode expansion
(cavity_coupling module: beta).

Geometry (axisymmetric, docs/cavity/coupling.md): the pillbox + on-axis semi-rigid coax monopole of Balleyguier,
Part. Accel. 57 (1997) Figs. 7-8 (TM010 at 1146 MHz, cavity 200 mm diameter x 150 mm, coax 1.65/5.35 mm with
eps_r = 2.0, antenna length 0-55 mm); its computed Q_ext column is tabulated in Dhavale & Mittal, APAC 2007
WEPMA133 Table 3 (together with a Kroll-Yu column).  Quotes: docs/cavity/coupling.md Sec. 5.

Cases
  depth     h = 0 ... 55 mm: Q_ext, beta, Q_L from (a) the driven fit and (b) the open/short eigenmode expansion,
            the reference columns, the slender-probe law (independent derivation) and the flush TEM-aperture
            estimate; mesh refinement at a few depths.
  sigma     h = 10 mm, wall conductivity x 1e-2, 1, 1e2 (Q0 x 0.1, 1, 10): Q_ext must not depend on Q0.
  coax_len  h = 10 mm, line length 30 / 50 mm: Q_ext of a matched lossless line does not depend on it.
  mismatch  h = 10 mm, receiver Z0 = 25 ohm (not the line impedance): driven vs expansion with a mismatched load.
Each row also carries the loaded port power at the loaded resonance vs the single-mode haloscope form
P = beta/(1+beta) Q_L O^2/(2 omega) and the driven-vs-expansion Psi(omega) near the resonance.

    python benchmarks/cavity_coupling_validation.py [--out benchmarks/records/cavity_coupling.json] [--quick]
    (parallel: --list; --tag T --out part_T.json per tag; --merge part_*.json --out ...)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Dhavale & Mittal, APAC 2007 WEPMA133, Table 3: penetration depth [mm] -> Qext (Balleyguier method [9], Kroll-Yu)
REF_TABLE3 = {0: (6068050, 3553030), 2.5: (407113, 269016), 5: (81493, 74439), 7.5: (23841, 28111),
              10: (10536, 12086.1), 12.5: (5361, 4449.1), 15: (2790, 2660.16), 17.5: (1682, 1709.18),
              20: (1064, 1061.46), 25: (455, 482.99), 30: (207, 232.46), 35: (106, 118.52), 40: (56.5, 61.87),
              45: (33, 34.15), 50: (19.9, 20.36), 55: (13.1, 13.87)}


def balleyguier_geometry(h_m: float, **kw):
    from axionfem.cavity_coupling import ProbePillbox

    return ProbePillbox(R=0.1, L=0.15, a=0.825e-3, b=2.675e-3, coax_eps_r=2.0, h=h_m, coax_len=0.03,
                        sigma_S_m=5.8e7, **kw)


def slim(out: dict) -> dict:
    o = dict(out)
    if "Psi_check" in out:
        o["Psi_max_rel_diff_vs_peak"] = out["Psi_check"]["max_rel_diff_vs_peak"]
        o["Psi_max_rel_diff_loaded_window"] = out["Psi_check"]["max_rel_diff_loaded_window"]
    return o


def references(geo, Q0: float) -> dict:
    """Independently derived closed forms: slender probe Eqs. (2d)/(2e), flush TEM aperture Eq. (3c)."""
    from axionfem.cavity_coupling import probe_coupling_estimates

    return probe_coupling_estimates(geo, Q0)


def jobs(quick: bool = False) -> list[dict]:
    """Every case as {kind, tag, geo, Z0, extra}; the headline runs use refine = 2 (refine 1 and 3 for convergence)."""
    J = []

    def add(kind, geo, tag, Z0=None, **extra):
        J.append({"kind": kind, "tag": tag, "geo": geo, "Z0": Z0, "extra": extra})

    R2 = 1.0 if quick else 2.0
    for hmm in ([0, 5, 10, 25, 55] if quick else sorted(REF_TABLE3)):
        add("depth", balleyguier_geometry(hmm * 1e-3, refine=R2), f"h{hmm}", h_mm=hmm, refine=R2,
            ref_Q_ext_balleyguier=REF_TABLE3[hmm][0], ref_Q_ext_kroll_yu=REF_TABLE3[hmm][1])
    if not quick:
        for hmm in (0, 10, 25, 55):
            for rf in (1.0, 3.0):
                add("convergence", balleyguier_geometry(hmm * 1e-3, refine=rf), f"h{hmm}_r{rf:g}", h_mm=hmm, refine=rf,
                    ref_Q_ext_balleyguier=REF_TABLE3[hmm][0], ref_Q_ext_kroll_yu=REF_TABLE3[hmm][1])
    g10 = balleyguier_geometry(10e-3, refine=R2)
    for fac in (1e-2, 1e2):
        add("sigma", replace(g10, sigma_S_m=g10.sigma_S_m * fac), f"sigma_x{fac:g}", sigma_factor=fac)
    add("coax_len", replace(g10, coax_len=0.05), "coax50mm", coax_len_m=0.05)
    # 90 mm: the stub's half-wave resonance (~1.09 GHz with the probe length) sits near the mode -> the guard must fire
    add("coax_len", replace(g10, coax_len=0.09), "coax90mm", coax_len_m=0.09)
    add("mismatch", g10, "Z0_25ohm", Z0=25.0)
    return J


def run_job(j: dict, wd: Path) -> dict:
    from axionfem.cavity_coupling import coupling_study

    t0 = time.time()
    o = coupling_study(j["geo"], wd / j["tag"], Z0=j["Z0"], warn=False)
    row = {"case": j["kind"], "tag": j["tag"], **slim(o), **j["extra"], "seconds": time.time() - t0}
    if o["expansion"].get("Q0"):
        row["references"] = references(j["geo"], o["expansion"]["Q0"])
    ex, dr = o["expansion"], o.get("driven", {})
    if o.get("warnings"):
        print(f"  warnings: {o['warnings']}", flush=True)
    print(f"{j['kind']:11s} {j['tag']:14s} Qext exp {ex['Q_ext']:.6g} drv {dr.get('Q_ext', float('nan')):.6g} "
          f"beta {dr.get('beta', float('nan')):.4g} dQext {o.get('comparison', {}).get('Q_ext', float('nan')):+.2e} "
          f"1/QL {1 / ex.get('Q_L', float('inf')):.2e} P {o.get('loaded_power', {}).get('ratio', float('nan')):.6f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return row


def run(out: Path | None, quick: bool = False, tag: str | None = None) -> dict:
    rec = {"schema": "axionfem.cavity_coupling_validation/1", "beta": True, "cases": []}
    with tempfile.TemporaryDirectory(prefix="cav_coupling_") as td:
        for j in jobs(quick):
            if tag is None or j["tag"] == tag:
                rec["cases"].append(run_job(j, Path(td)))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def merge(parts: list[Path], out: Path) -> dict:
    """Merge per-tag records (parallel batch runs) in the order of :func:`jobs`."""
    from axionfem.cavity_coupling import ProbePillbox, coupling_flags, line_resonances

    rows = {}
    for p in parts:
        for r in json.loads(Path(p).read_text())["cases"]:
            # re-evaluate the consistency flags with the current coupling_flags (no re-solve needed)
            geo = ProbePillbox(**r["geometry"])
            r["expansion"]["line_resonances_Hz"] = line_resonances(geo, 2 * r["expansion"]["f_open_Hz"])
            r["warnings"], r["notes"] = coupling_flags(r)
            rows[r["tag"]] = r
    order = [j["tag"] for j in jobs()]
    rec = {"schema": "axionfem.cavity_coupling_validation/1", "beta": True,
           "cases": [rows[t] for t in order if t in rows]}
    out.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def plot(rec: dict, png: Path) -> None:
    """Q_ext vs insertion depth (driven, expansion, Balleyguier / Kroll-Yu columns, slender law) and the
    driven-vs-expansion difference vs 1/Q_L (the O(1/Q) criterion) at the mesh levels run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    c1, c2, c3, c4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    dep = [r for r in rec["cases"] if r["case"] == "depth" and "driven" in r]
    h = np.array([r["h_mm"] for r in dep])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    a = ax[0]
    a.plot(h, [r["driven"]["Q_ext"] for r in dep], "-o", color=c1, lw=2, ms=6, label="axionFEM driven fit (a)")
    a.plot(h, [r["expansion"]["Q_ext"] for r in dep], "s", mfc="none", color=c2, ms=8, mew=1.5,
           label="axionFEM eigenmode expansion (b)")
    a.plot(h, [r["ref_Q_ext_balleyguier"] for r in dep], "^", color=c3, ms=7, label="Balleyguier 1997 (computed)")
    a.plot(h, [r["ref_Q_ext_kroll_yu"] for r in dep], "x", color=c4, ms=7, mew=1.5, label="Kroll-Yu (Dhavale 2007)")
    hs = [r for r in dep if r["h_mm"] > 0]
    a.plot([r["h_mm"] for r in hs], [r["references"]["Q_ext_slender_2d"] for r in hs], "--", color=muted, lw=1.2,
           label="slender-probe law (blind derivation)")
    a.set_yscale("log")
    a.set_xlabel("antenna insertion depth h [mm]", color=ink)
    a.set_ylabel("Q_ext", color=ink)
    a.set_title("Pillbox TM010 + on-axis coax probe", color=ink, fontsize=11)
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25, lw=0.5)
    b = ax[1]
    for case, col, mk, lab in (("convergence", muted, "d", None), ("depth", c1, "o", "refine 2")):
        rows = [r for r in rec["cases"] if r["case"] == case and "comparison" in r]
        for rf in sorted({r.get("refine", 1.0) for r in rows}):
            rr = [r for r in rows if r.get("refine", 1.0) == rf]
            x = [1 / r["expansion"]["Q_L"] for r in rr]
            y = [abs(r["comparison"]["Q_ext"]) for r in rr]
            b.plot(x, y, mk, color=col if case == "depth" else (c2 if rf == 1 else c4), ms=6,
                   label=lab or f"refine {rf:g}")
    xx = np.logspace(-6, 0, 10)
    b.plot(xx, xx, ":", color=muted, lw=1, label="|diff| = 1/Q_L")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("1/Q_L", color=ink)
    b.set_ylabel("|Q_ext(driven)/Q_ext(expansion) - 1|", color=ink)
    b.set_title("Driven vs expansion", color=ink, fontsize=11)
    b.legend(fontsize=8, frameon=False)
    b.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "records" / "cavity_coupling.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", help="run one case (see --list)")
    ap.add_argument("--list", action="store_true", help="print the case tags")
    ap.add_argument("--merge", nargs="+", type=Path, help="merge per-tag records into --out")
    ap.add_argument("--plot", type=Path, help="plot the record --out into this PNG")
    a = ap.parse_args()
    if a.list:
        print("\n".join(j["tag"] for j in jobs(a.quick)))
    elif a.plot:
        plot(json.loads(a.out.read_text()), a.plot)
    elif a.merge:
        merge(a.merge, a.out)
    else:
        run(a.out, a.quick, a.tag)
    sys.exit(0)
