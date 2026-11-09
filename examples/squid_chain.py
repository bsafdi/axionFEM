#!/usr/bin/env python
"""eta(f) and eps_uc(f) of the [DMR-SQUID] (arXiv:2504.20398) chain presets, and their use in a scan.

Physics and conventions: docs/physics/squid_chain.md.

Run (from the repository root, with the package built):

    PYTHONPATH=python .venv/bin/python examples/squid_chain.py [--out docs/physics/figures]

Writes squid_chain_eta.pdf and squid_chain_eps_uc.pdf (black/gray/maroon = low/medium/high-BW
second stage; solid = RT preamps, XXF-1 below 50 MHz and the 300 MHz preamp above; dashed =
CITLF2 cryogenic amplifier, 5-300 MHz; dotted = C1 alone) and prints a table.  These are the
analogues of the paper's Figs. 3-4 (log frequency axis), computed, not digitized.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from axionfem import scan, squid

COLORS = {"low": "k", "medium": "0.5", "high": "maroon"}


def curves(f_lo=1e5, f_hi=3e8, n=400):
    """eta and eps_uc/hbar on log grids, split at the RT preamp switch (50 MHz)."""
    out = {}
    for bw in squid.SECOND_STAGES:
        for amp, (a, b) in (("RT", (f_lo, f_hi)), ("cryo", (squid.CITLF2.f_min_Hz, f_hi))):
            ch = squid.chain(f"C1+{bw}+{amp}")
            parts = [np.geomspace(a, min(b, 50e6) * (1 - 1e-9), n // 2), np.geomspace(50e6, b, n // 4)] \
                if amp == "RT" else [np.geomspace(a, b, n // 2)]
            f = np.concatenate([np.append(p, np.nan) for p in parts])[:-1]
            ok = np.isfinite(f)
            eta, eps = np.full(f.shape, np.nan), np.full(f.shape, np.nan)
            eta[ok], eps[ok] = ch.eta(f[ok]), ch.eps_uc_over_hbar(f[ok])
            out[(bw, amp)] = (f, eta, eps)
    return out


def plot(out, path_eta, path_eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c1 = squid.chain("C1")
    for k, (path, ylab, ref) in enumerate(((path_eta, r"$\eta$", c1.eta(1e6)),
                                           (path_eps, r"$\epsilon_{\rm uc}/\hbar$", c1.eps_uc_over_hbar(1e6)))):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for (bw, amp), (f, eta, eps) in out.items():
            ax.plot(f / 1e6, eta if k == 0 else eps, color=COLORS[bw], ls="-" if amp == "RT" else "--",
                    label=f"{bw} BW {amp}")
        ax.axhline(ref, color="k", ls=":", lw=1, label="C1 alone")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 300)
        ax.set_xlabel("frequency [MHz]")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path)
        fig.savefig(Path(path).with_suffix(".png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/physics/figures", help="directory for the two figures (PDF + PNG)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    print("preset              f [MHz]    eta   eps_uc/hbar  kappa_g^2 (Eq. 48, Q = 1e6, T = 30 mK)")
    for p in squid.presets():
        ch = squid.chain(p)
        for f in ((0.1209, 1.0, 10.0, 100.0) if "cryo" not in p else (10.0, 100.0)):
            print(f"  {p:<18} {f:7.4g} {ch.eta(f * 1e6):7.3f} {ch.eps_uc_over_hbar(f * 1e6):9.3f}"
                  f"   {ch.kappa_g2_scan(f * 1e6, 1e6, 0.03):.4g}")
    v = squid.C1.validity()
    print(f"C1: beta_L = {v['beta_L']:.3f}, Gamma = {v['Gamma']:.2e}, eta_TC = {squid.C1.eta_TC:.4f}, "
          f"on resonance {squid.C1.on_resonance_over_SQL:.3f} x SQL")

    # in a scan: eta(f) of the chain at every tuning (DMRadio-50L-like detector, illustrative)
    PSI, LP = 2.0e-4, 1.0e-6
    for bm in (squid.benchmark("C1+low+RT", Q=1e6, T_K=0.03), {"Q": 1e6, "T": 0.03, "eta_A": 10.0}):
        r = scan.reach(PSI, LP, bm, n=5)
        print(f"reach with eta = {r['band']['eta_A']}: g95(120.9 kHz) = {r['g95_GeV'][0]:.4g} GeV^-1")

    if not a.no_plot:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        plot(curves(), d / "squid_chain_eta.pdf", d / "squid_chain_eps_uc.pdf")
        print(f"wrote {d / 'squid_chain_eta.{pdf,png}'} and {d / 'squid_chain_eps_uc.{pdf,png}'}")


if __name__ == "__main__":
    main()
