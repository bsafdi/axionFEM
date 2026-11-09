#!/usr/bin/env python
"""Readout-circuit networks: from a bare series RLC to a solved toroid's full readout.

Physics and conventions: docs/physics/readout_circuit.md.

Run (from the repository root, with the package built):

    PYTHONPATH=python AXF_MESHER=build/axf-mesher .venv/bin/python examples/readout_circuit.py

Pass ``--no-solve`` to skip the FEM parts (sections 5 and 6) and run in a fraction of a
second; pass ``--plot FILE.png`` to write the noise-budget figure of section 4.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from axionfem import circuit as ck, scan

MU0 = ck.MU0
SEP = "=" * 78


def head(n, title):
    print(f"\n{SEP}\n{n}. {title}\n{SEP}")


# ---------------------------------------------------------------------------- 1
def bare_rlc():
    head(1, "A circuit by hand: series RLC, and the conventions")
    L, C, R = 1e-6, 1e-9, 0.5
    c = ck.Circuit()
    c.source("V", "gnd", "a", emf=1.0)          # 1 V drive; emf drives current p -> m
    c.resistor("R", "a", "b", R, T_K=0.020)     # T_K switches on its Johnson noise
    c.inductor("L", "b", "d", L)
    c.capacitor("C", "d", "gnd", C)

    w0 = 1.0 / math.sqrt(L * C)
    print(f"  f0 = {w0 / 2 / math.pi:.6g} Hz,  Q = w0 L/R = {w0 * L / R:.6g}")
    print("   w/w0     |Z| [ohm]     arg Z [rad]    |I| [A]")
    for x in (0.8, 0.95, 1.0, 1.05, 1.3):
        Z = c.driving_point_impedance(x * w0, "V")
        I = c.solve(x * w0).I("L")
        print(f"  {x:5.2f}   {abs(Z):12.6g}  {np.angle(Z):+12.6f}  {abs(I):12.6g}")
    print(f"  Z_L = -i w L and Z_C = +i/(w C) (the package's e^-iwt convention), so the loop is")
    print(f"  capacitive (Im Z > 0) below resonance:  Im Z(0.8 w0) = "
          f"{c.driving_point_impedance(0.8 * w0, 'V').imag:+.6g} ohm")
    print(f"  MNA matrix asymmetry (0 for a reciprocal network): {c.check_reciprocity(w0):.2g}")


# ---------------------------------------------------------------------------- 2
def flux_transfer():
    head(2, "Flux transfer: superconducting loops, and the three different 'optima'")
    Lp, Lsq = 1e-6, 2e-9
    print("  (a) pickup L_p in series with the SQUID input coil L_in, both superconducting.")
    print("      The frequency-domain solve reproduces flux conservation exactly:")
    for Lin in (0.3e-6, 1e-6, 3e-6):
        M = math.sqrt(Lin * Lsq)                 # k = 1
        Phi = 1e-12
        for f in (1e2, 1e6):
            w = 2 * math.pi * f
            c = ck.Circuit()
            c.source("pickup", "gnd", "a", emf=complex(0, -w) * Phi,
                     Z=lambda ww: complex(0, -ww * Lp))
            c.inductor("L_in", "a", "gnd", Lin)
            I = c.solve(w).I("L_in")
            assert abs(I / (Phi / (Lp + Lin)) - 1) < 1e-10
        print(f"      L_in = {Lin:8.3g} H:  Phi_SQ/Phi = M/(L_p+L_in) = "
              f"{M / (Lp + Lin):.6g}   (I = Phi/(L_p+L_in) at every frequency)")
    best = max(np.geomspace(1e-9, 1e-3, 4001), key=lambda x: math.sqrt(x * Lsq) / (Lp + x))
    print(f"      -> maximal at L_in = {best:.4g} H = L_p: the textbook inductance match,")
    print(f"         Phi_SQ/Phi = (k/2) sqrt(L_sq/L_p) = {0.5 * math.sqrt(Lsq / Lp):.6g}.")
    print("         [C] App. F 3 is explicit that this is the WRONG optimum for a scan.")

    print("\n  (b) maximum power transfer: the conjugate match Z_L = Z_s*")
    Zs = complex(2.0, -3.0)
    cm = ck.Circuit()
    cm.source("V", "gnd", "a", emf=1.0, Z=Zs)
    cm.impedance("ZL", "a", "gnd", Zs.conjugate())
    P = 0.5 * abs(cm.solve(2 * math.pi * 1e5).I("ZL")) ** 2 * Zs.real
    print(f"      Z_s = {Zs}:  P = {P:.6g} W = |V|^2/(8 Re Z_s) = {1 / (8 * Zs.real):.6g} W")

    print("\n  (c) the scan match xi = xi_opt, and the single-frequency noise match xi = eta_A")
    n, eta = 50.0, 1.0
    xo = ck.xi_opt(n, eta)
    print(f"      n = {n:g}, eta_A = {eta:g}:  xi_opt = {xo:.6g}, xi_noise_match = {eta:g}")
    print(f"      sensitivity bandwidth sqrt(A)/xi:  "
          f"{math.sqrt(ck.A_noise(xo, n, eta)) / xo:.1f} linewidths at xi_opt vs "
          f"{math.sqrt(ck.A_noise(eta, n, eta)) / eta:.1f} at the noise match")
    print("      ([BF] Fig. 3 quotes ~173 and ~14 for exactly this case.)")


# ---------------------------------------------------------------------------- 3
def dmradio(plot=None):
    head(3, "A DMRadio-style readout: pickup + tuning C + input coil + leads")
    Psi_SI, L_p = MU0 * 100.0, MU0 * 1.0        # toy B of the scan spec
    f0 = 1.209e5
    r = ck.series_readout(
        L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=f0,
        Q=1e6, T_K=0.030,                        # pickup loss and temperature
        eta_A=10.0,                              # SQUID, 10x above the SQL (DMRadio-m3 target)
        L_in_H=0.1 * L_p, R_in_ohm=1e-8,         # input coil and its loss
        lead_length_m=2.0, L_per_m=1e-7, R_per_m=1e-9, T_lead_K=4.0,   # warm leads
        tan_delta=1e-6,                          # capacitor dielectric loss
        label="toy B + input coil + 4 K leads")
    print(r.summary(f0))
    a = r.analyze(f0)
    print("\n  The noise budget says where to spend effort: the 4 K leads contribute")
    tot = a["noise"]["S_Phi_thermal_Wb2_s"]
    for k, v in sorted(a["thermal_by_branch_Wb2_s"].items(), key=lambda kv: -kv[1]):
        print(f"      {k:10s} {100 * v / tot:6.2f} % of the thermal noise")
    print(f"  and the whole thermal channel is {100 * a['N_V_thermal_V2_s'] / a['N_V_V2_s']:.1f} %"
          f" of N_V (backaction {100 * a['N_V_backaction_V2_s'] / a['N_V_V2_s']:.1f} %,"
          f" imprecision {100 * a['N_V_imprecision_V2_s'] / a['N_V_V2_s']:.2g} %).")
    return r, plot


# ---------------------------------------------------------------------------- 4
def scan_handoff():
    head(4, "Mapping onto the optimal-scan inputs (Q, T, eta_A), and the L_p/L_tot penalty")
    Psi_SI, L_p = MU0 * 10.0, MU0 * 1.0          # toy A
    f0, Q, T, eta = 1.209e5, 2e7, 0.010, 0.1
    w0 = 2 * math.pi * f0
    R = w0 * L_p / Q

    print("  A trivial circuit (pickup + C + ideal amplifier) must reproduce the package's")
    print("  existing scan PSD exactly -- it is the series-loop special case of the network:")
    xi = ck.xi_opt(ck.n_bose(w0, T), eta)
    r0 = ck.series_readout(L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=f0, R_loss_ohm=R, T_K=T,
                           eta_A=eta, xi=xi)
    from axionfem import _core
    m_a = scan.freq_to_mass(f0)
    g = float(scan.g_qcd([m_a])[0])
    f = np.linspace(f0 - 500, f0 + 500, 21)
    ref = _core.psd_at_tuning(Psi_SI, L_p, Q, T, eta, f0, m_a, g, list(f), 0.3)
    mine = np.array([r0.noise_psd(2 * math.pi * x)["N_V_V2_s"] for x in f])
    print(f"      max |N_V(circuit)/N_V(_core.psd_at_tuning) - 1| = "
          f"{np.abs(mine / np.asarray(ref['N_V_V2_s']) - 1).max():.2e}")

    print("\n  Adding an input coil raises L_eff, and the scan rate carries (L_p/L_eff)^2")
    print("  ([C] Eq. F37).  Passing L_eff instead of L_p to scan.reach() does it:")
    print("   L_in/L_p   L_eff/L_p   participation   Q_loaded     F/F(L_in=0)   g95/g95(L_in=0)")
    base = None
    for ratio in (0.0, 0.1, 0.5, 1.0, 3.0):
        rr = ck.series_readout(L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=f0, R_loss_ohm=R, T_K=T,
                               eta_A=eta, L_in_H=ratio * L_p)
        e = rr.equivalent(w0)
        reach = scan.reach(Psi_SI, e["L_eff_H"], {"Q": e["Q_loaded"], "T": e["T_eff_K"],
                                                 "eta_A": e["eta_A_eff"]}, n=4)
        g95 = float(np.asarray(reach["g95_GeV"])[0])
        base = base or g95
        print(f"   {ratio:7.1f}   {e['L_eff_H'] / L_p:9.4f}   {e['participation']:13.4f}   "
              f"{e['Q_loaded']:10.4g}   {(L_p / e['L_eff_H']) ** 2:11.4f}   {g95 / base:14.4f}")
    print("  (the reach degrades as (L_eff/L_p)^(1/2) at fixed R, exactly as expected)")

    print("\n  Several loss channels at different temperatures give T_eff/Q = sum_k T_k/Q_k")
    print("  (scan_and_signal.md Sec. 3.4) -- here the network derives the rule rather than assuming it:")
    R1, R2, T1, T2 = R, R / 3, 0.010, 4.0
    rr = ck.series_readout(L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=f0, R_loss_ohm=R1, T_K=T1,
                           R_in_ohm=R2, T_in_K=T2, eta_A=eta)
    e = rr.equivalent(w0)
    Q1, Q2 = w0 * e["L_eff_H"] / R1, w0 * e["L_eff_H"] / R2
    print(f"      Q1 = {Q1:.4g} at {T1} K, Q2 = {Q2:.4g} at {T2} K")
    print(f"      -> Q_loaded = {e['Q_loaded']:.6g} (= 1/(1/Q1+1/Q2) = {1/(1/Q1+1/Q2):.6g})")
    print(f"      -> T_eff = {e['T_eff_K']:.6g} K, T_eff/Q = {e['T_over_Q_K']:.6g} K"
          f"  (T1/Q1 + T2/Q2 = {T1/Q1 + T2/Q2:.6g})")

    print("\n  The kernel K (Theta per second of dwell) times the equivalent bandwidth over")
    print("  Theta95 reproduces the local scan rate of scan_and_signal.md dw0/dt:")
    for m_eV, ref_rate in ((0.5e-9, 1.921385604e-3), (1e-9, 0.06148433407),
                           (4.1e-9, 71.23328539)):
        fm = scan.mass_to_freq(m_eV)
        wm = 2 * math.pi * fm
        gm = float(scan.g_qcd([m_eV])[0])
        rm = ck.series_readout(L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=fm, Q=Q, T_K=T, eta_A=eta,
                               xi=ck.xi_opt(ck.n_bose(wm, T), eta))
        em = rm.equivalent(wm)
        rate = rm.kernel(wm, gm)["K_per_s"] * 2 * math.pi * em["bandwidth_Hz"]["equivalent"] / 2.71
        print(f"      m_a = {m_eV * 1e9:4.1f} neV (DFSZ): {rate:.9g} rad/s^2  "
              f"(scan_and_signal.md Sec. 8: {ref_rate:.9g}, rel {abs(rate / ref_rate - 1):.1e})")


# ---------------------------------------------------------------------------- 5
def transformers():
    head(5, "Transformer stages, and what they do and do not change")
    Psi_SI, L_p = MU0 * 100.0, MU0 * 1.0
    f0, w0 = 1.209e5, 2 * math.pi * 1.209e5
    common = dict(L_p_H=L_p, Psi_SI=Psi_SI, f_Hz=f0, Q=1e6, T_K=0.030, eta_A=10.0)
    rows = [
        ("plain series", ck.series_readout(L_in_H=1e-8, **common)),
        ("ideal transformer n=0.1", ck.series_readout(transformer_ratio=0.1, L_in_H=1e-8,
                                                     **common)),
        ("real transformer k=0.90", ck.series_readout(L_1_H=0.5 * L_p, L_2_H=1e-8, k_T=0.90,
                                                     L_in_H=1e-8, **common)),
        ("real transformer k=1.00", ck.series_readout(L_1_H=0.5 * L_p, L_2_H=1e-8, k_T=1.0,
                                                     L_in_H=1e-8, **common)),
        ("parasitic C (cap. tap)", ck.series_readout(C_par_F=0.3 / (w0 ** 2 * L_p), **common)),
    ]
    print(f"  {'topology':26s} {'L_eff/L_p':>10s} {'Q_loaded':>11s} {'xi/xi_opt':>10s} "
          f"{'port_coup':>10s} {'eta_A_eff':>10s} {'N_V resid':>10s}")
    for name, r in rows:
        e = r.equivalent(w0)
        print(f"  {name:26s} {e['L_eff_H'] / L_p:10.4f} {e['Q_loaded']:11.4g} "
              f"{e['xi'] / e['xi_opt']:10.4f} {e['port_coupling']:10.6f} "
              f"{e['eta_A_eff']:10.4f} {e['N_V_reduction_resid']:10.1e}")
    print("\n  Every one of these is a lossless reciprocal coupling network, and all of them")
    print("  leave eta_A untouched (port_coupling = 1): they are pure impedance")
    print("  transformations, moving L_eff and the coupling xi but never the amplifier's")
    print("  noise figure.  Only dissipation in the coupling path degrades eta_A_eff:")
    r_lossy = ck.series_readout(C_par_F=0.3 / (w0 ** 2 * L_p), tan_delta_par=1e-2, **common)
    e = r_lossy.equivalent(w0)
    print(f"      parasitic C with tan(delta) = 1e-2: port_coupling = {e['port_coupling']:.8f}, "
          f"eta_A_eff = {e['eta_A_eff']:.6g}")
    print("\n  Nyquist's theorem holds on all of them (a correctness check of the noise code):")
    for name, r in rows:
        resid = max(r.circuit.nyquist_check(2 * math.pi * x * f0, "pickup")
                    for x in (0.7, 1.0, 1.4))
        print(f"      {name:26s} max |sum_b |H_b/H_p|^2 Re Z_b / Re Z_eff - 1| = {resid:.1e}")


# ---------------------------------------------------------------------------- 6
def from_a_solved_design(workdir):
    head(6, "From a solved design: Results.readout()")
    import axionfem as af

    d = af.template("toroid", R=1.0, H=1.0)
    doc = __import__("json").loads(d.to_json())
    doc["scan"] = {"f_min": 1.209e5, "f_max": 9.9138e5}
    doc.setdefault("materials", {})["nb_rf"] = {
        "kind": "superconductor",
        "rf": {"model": "power", "Rs_ohm": 1e-9, "f_ref": 1e6, "exponent": 2.0, "T_K": 0.030}}
    for reg in doc["regions"]:
        if reg.get("material") == "nb":
            reg["material"] = "nb_rf"
    res = af.Design(doc).solve(workdir, verbose=0)
    print(f"  solved toroid: Psi = {res.Psi:.6g} H T m^2, L_p = {res.L_p:.6g} H")
    lo = res.data["pickup"]["losses"]
    print(f"  RF surface losses: T/Q = {lo['T_over_Q_total_K']['f_min']:.4g} K at f_min, "
          f"{lo['T_over_Q_total_K']['f_max']:.4g} K at f_max")

    out = res.readout(f_Hz=1.209e5, benchmark="conservative", L_in_H=0.1 * res.L_p)
    print()
    print(out["readout"].summary(out["f0_Hz"]))
    print("\n  The scan inputs this readout hands to the optimal scan:")
    for k, v in out["scan_inputs"].items():
        if k != "note":
            print(f"      {k:18s} {v:.6g}")
    print("\n  Sweeping the tuning across the band (R_loss ~ f^2 from the surface losses):")
    print(f"   {'f0 [Hz]':>10s} {'Q_loaded':>12s} {'T_eff [K]':>10s} {'xi':>10s} "
          f"{'N_V [V^2 s]':>12s} {'dwell for Theta95 [s]':>22s}")
    for f0 in np.geomspace(1.209e5, 9.9138e5, 5):
        o = res.readout(f_Hz=float(f0), benchmark="conservative", L_in_H=0.1 * res.L_p)
        print(f"   {o['f0_Hz']:10.4g} {o['Q_loaded']:12.5g} {o['T_eff_K']:10.4g} "
              f"{o['xi']:10.4g} {o['noise']['N_V_V2_s']:12.4g} "
              f"{o['kernel']['dwell_for_Theta_s']:22.4g}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-solve", action="store_true", help="skip the FEM sections")
    ap.add_argument("--workdir", default="out/readout_circuit")
    ap.add_argument("--plot", default=None, help="write the noise-budget figure here")
    args = ap.parse_args()

    bare_rlc()
    flux_transfer()
    r, _ = dmradio()
    scan_handoff()
    transformers()
    if not args.no_solve:
        from_a_solved_design(args.workdir)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f0 = 1.209e5
        bw = r.equivalent(2 * math.pi * f0)["bandwidth_Hz"]["noise_doubling"]
        f = np.linspace(f0 - 3 * bw, f0 + 3 * bw, 401)
        sw = r.sweep(f)
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        for key, lab in (("N_V_thermal_V2_s", "thermal"),
                         ("N_V_backaction_V2_s", "backaction"),
                         ("N_V_imprecision_V2_s", "imprecision"),
                         ("N_V_V2_s", "total")):
            ax.plot((f - f0), sw[key], label=lab, lw=2.2 if lab == "total" else 1.4)
        ax.set(yscale="log", xlabel=f"f - {f0 / 1e3:g} kHz  [Hz]",
               ylabel=r"$N_V$  [V$^2$ s]  (EMF-referred)",
               title="Readout noise budget, referred to a series EMF in the pickup")
        ax.legend(frameon=False)
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
