#!/usr/bin/env python
"""Planning a scan with the exact readout noise, and a beyond-MQS reach curve.

Physics: docs/physics/scan_and_signal.md Sec. 6 and docs/physics/readout_circuit.md Sec. 5.3;
numbers: docs/validation/finite_band.md.

Run (from the repository root, with the package built):

    PYTHONPATH=python AXF_MESHER=build/axf-mesher .venv/bin/python examples/finite_band.py

Pass ``--no-solve`` to skip the FEM part (section 4) and run in a couple of seconds.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from axionfem import circuit as ck, scan

MU0 = ck.MU0
PSI = MU0 * 100.0      # Psi = 100 T m^3 in SI (T H m^2)
LP = MU0 * 1.0         # L_p = 1 m in SI (H)


def section(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-solve", action="store_true", help="skip the FEM section")
    args = ap.parse_args()

    # ---------------------------------------------------------------------- 1
    section(1, "every reach call echoes the band it scanned")
    r = scan.reach(PSI, LP, "conservative", live_time_yr=3.0, f_min=120.9e3, f_max=991.4e3)
    b = r["band"]
    print(f"  {b['f_min_Hz'] / 1e3:.1f}-{b['f_max_Hz'] / 1e3:.1f} kHz "
          f"({b['m1_eV'] / 1e-9:.3g}-{b['m2_eV'] / 1e-9:.3g} neV), {b['n_targets']} masses, "
          f"{b['allocation']} allocation")
    print(f"  {b['live_time_yr']:.3g} yr live time (the target curve needs "
          f"{b['t_req_s'] / scan.YEAR:.4g} yr for g_target exactly)")
    print(f"  benchmark {b['benchmark']!r}: Q = {b['Q']:.3g}, T = {b['T_K']:.3g} K, "
          f"eta_A = {b['eta_A']:.3g}; F = {b['F_SI_T4_H2_m8']:.4g} T^4 H^2 m^8")
    print(f"  conventions: Q_axion = {b['Q_axion']:.3g}, Theta = {b['Theta']}, "
          f"rho = {b['rho_GeV_cm3']} GeV/cm^3, v0/vobs = {b['v0_kms']}/{b['vobs_kms']} km/s, "
          f"{b['model']} line with C_gamma = {b['C_gamma']}")

    # ---------------------------------------------------------------------- 2
    section(2, "the exact network noise in the finite-band scan (scan_and_signal.md Sec. 6)")
    rd = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=120.9e3, Q=1e4, T_K=0.030, eta_A=10.0,
                           L_in_H=0.1 * LP, tan_delta=1e-6)
    out = scan.finite_band(rd, 120.9e3, 400e3, n_settings=24, n_targets=36, convergence=True)
    s, ls, c = out["schedule"], out["local_schedule"], out["comparison"]
    print(f"  {out['band']['n_settings']} settings, {out['band']['n_targets']} targets, "
          f"{out['band']['noise_grid']['points']} N_V samples per setting")
    print(f"  sensitivity band Delta f_2x/f_0 = {ls['noise_doubling_over_f0'][0]:.4g} at the "
          f"bottom, {ls['noise_doubling_over_f0'][-1]:.4g} at the top")
    print(f"  LP (S6.4) total time {s['t_total_s'] / scan.YEAR:.5g} yr, min coverage "
          f"{s['min_coverage']:.6f}, {int((s['t_s'] > 0).sum())} settings used")
    sp = ls["spacing_over_noise_doubling_width"]
    print(f"  the local (Lorentzian) schedule needs {ls['t_total_s'] / scan.YEAR:.5g} yr and "
          f"covers {ls['coverage_min']:.3f}-{ls['coverage_max']:.3f} x Theta "
          f"({ls['coverage_interior_min']:.3f}-{ls['coverage_interior_max']:.3f} inside the band)")
    print(f"  setting spacing / sensitivity band: {sp[0]:.3f} at the bottom, {sp[-1]:.3f} at the "
          "top -- above ~1/2 the settings under-sample the band and the local schedule leaves")
    print("     holes between them; the LP still covers every target, by reallocating the time")
    print(f"  single-pole reduction: on resonance {c['kernel_on_resonance_max_rel_dev']:.3g}, "
          f"worst kernel {100 * c['kernel_max_rel_dev']:+.2f} %, its plan covers "
          f"{c['min_coverage_of_reduced_schedule']:.5f}, its reach claim is "
          f"{100 * c['g95_same_schedule_max_rel_dev']:+.3f} %")
    print(f"  numerical floor (N_V grid halved): "
          f"{out['convergence']['noise_grid_halved_max_rel_dev']:.3g}")
    print(f"  narrow-line kernel (S6.3) vs the full integral: "
          f"{out['narrow_line']['max_rel_dev']:.3g}")

    # ---------------------------------------------------------------------- 3
    section(3, "where the single pole fails: a normal-metal pickup, R ~ sqrt(f)")
    w1 = 2 * math.pi * 120.9e3
    R0 = w1 * LP / 300.0
    Zp = lambda w: complex(R0 * math.sqrt(w / w1), -w * LP)      # noqa: E731
    for R_exp in (0.0, 0.5):
        rr = ck.series_readout(Z_p=Zp, L_p_H=LP, Psi_SI=PSI, f_Hz=120.9e3, T_K=0.030, eta_A=10.0)
        o = scan.finite_band(rr, 120.9e3, 400e3, n_settings=16, n_targets=30,
                             reduced_R_exp=R_exp)
        c = o["comparison"]
        print(f"  pole with R_exp = {R_exp:g}: worst kernel {100 * c['kernel_max_rel_dev']:+.2f} %, "
              f"time x{c['t_total_ratio_reduced_over_exact']:.4f}, reach claim "
              f"{100 * c['g95_same_schedule_max_rel_dev']:+.3f} %")
    print("  -> the constant-R pole is wrong by tens of per cent across a band six times the")
    print("     centre frequency; Detector::R_exp = 1/2 recovers the exact answer.")

    # ---------------------------------------------------------------------- 4
    section(4, "a beyond-MQS reach curve from a full-wave sweep")
    if args.no_solve:
        print("  (skipped: --no-solve)")
        return
    import axionfem as af

    d = af.template("toroid")
    res0 = d.solve(verbose=0)
    R = res0.coupling["device_radius_m"]
    c_light = 2.99792458e8
    d.doc["fullwave"] = {"frequencies": [k * c_light / (2 * math.pi * R)
                                         for k in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)]}
    res = d.solve(verbose=0)
    tab = ck.fullwave_tables(res.data)
    print(f"  device radius {R:.4g} m; sweep {tab['f_Hz'][0] / 1e6:.3g}-"
          f"{tab['f_Hz'][-1] / 1e6:.3g} MHz (kR {tab['k_R'][0]:.3g}-{tab['k_R'][-1]:.3g})")
    print("   kR    |Psi|/Psi_MQS  L/L_MQS   R_rad [ohm]")
    for i in range(tab["f_Hz"].size):
        print(f"  {tab['k_R'][i]:5.3f}   {tab['Psi_SI'][i] / res0.Psi:9.5f}   "
              f"{tab['L_H'][i] / res0.L_p:8.5f}   {tab['R_rad_ohm'][i]:.4g}")
    try:
        rf = scan.reach_fullwave(res.data, "conservative", live_time_yr=3.0, Q_floor=1e6,
                                 allocation="log_uniform", n=9)
    except ValueError as exc:
        print(f"  local reach refused: {exc}")
        print("  Refine the sweep for unresolved slopes; use a tuned finite-band circuit for broad response.")
        return
    print("\n  log-uniform allocation, full wave vs MQS:")
    print("   f [MHz]   F/F_MQS   Q(fw)      g95(fw)/g95(MQS)")
    for i in range(rf["m_eV"].size):
        print(f"  {rf['f_Hz'][i] / 1e6:8.4g}  {rf['ratio_F_fullwave_over_mqs'][i]:8.4f}  "
              f"{rf['Q'][i]:9.4g}   {rf['ratio_g95_fullwave_over_mqs'][i]:.4f}")
    print(f"\n  band echo: {rf['band']['f_min_Hz'] / 1e6:.4g}-{rf['band']['f_max_Hz'] / 1e6:.4g} MHz, "
          f"kR {rf['band']['k_R'][0]:.3g}-{rf['band']['k_R'][1]:.3g}, "
          f"intrinsic MQS Q {rf['band']['Q_other']:.3g}, {rf['band']['allocation_quadrature']}")


if __name__ == "__main__":
    main()
