"""Readout-circuit network: closed forms, Nyquist noise, matching, and consistency with the
package's existing scan PSD (docs/physics/readout_circuit.md).

Every test here compares against a closed form, an identity, or an independently computed
reference.  No FEM solve is involved, so the whole file runs in well under a second; the
``Results.readout`` integration is exercised through a light stub and end-to-end in
``examples/readout_circuit.py``.

References: [C] Chaudhuri et al. arXiv:1803.01627, [BF] arXiv:1904.05806,
spec = docs/physics/scan_and_signal.md (tags "(S*)"), circuit spec tags "(C*)".
"""
import math

import numpy as np
import pytest

from axionfem import _core, circuit as ck, scan

HBAR, KB, MU0, PHI0 = ck.HBAR, ck.KB, ck.MU0, ck.PHI0
F0 = 1.209e5
W0 = 2 * math.pi * F0

# toy A and toy B of the scan spec (Sec. 8): Psi [T m^3], L_p [m], Q, T [K], eta_A
TOY_A = (10.0, 1.0, 2e7, 0.010, 0.1)
TOY_B = (100.0, 1.0, 1e6, 0.030, 10.0)


def _toy(toy):
    Psi, Lp, Q, T, eta = toy
    return MU0 * Psi, MU0 * Lp, Q, T, eta


# ======================================================================= conventions
def test_kappa_a_matches_spec_check_values():
    """kappa_a = g sqrt(2 rho) in the package's conventions (spec S2.3 and Sec. 8)."""
    assert ck.kappa_a_SI(1e-19, 0.3) == pytest.approx(8.65882567e-19, rel=1e-8)
    assert ck.kappa_a_inv_m(1e-19, 0.3) == pytest.approx(1.08810013e-24, rel=1e-8)
    # and it agrees with the C++ core's flux amplitude
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_A)
    m_a = scan.freq_to_mass(F0)
    ref = _core.psd_at_tuning(Psi_SI, L_SI, Q, T, eta, F0, m_a, 1e-19, [F0], 0.3)
    assert ck.kappa_a_SI(1e-19, 0.3) * Psi_SI == pytest.approx(ref["Phi_a_Wb"], rel=1e-12)


def test_element_impedances_follow_the_package_time_convention():
    """Z_L = -i omega L, Z_C = +i/(omega C)  (C1.1), as the full-wave engine reports."""
    c = ck.Circuit()
    c.inductor("L", "a", "gnd", 2e-6)
    c.capacitor("C", "b", "gnd", 3e-9)
    c.resistor("R", "d", "gnd", 5.0)
    w = 2 * math.pi * 1e5
    assert c._get("L").Z_at(w) == pytest.approx(complex(0, -w * 2e-6))
    assert c._get("C").Z_at(w) == pytest.approx(complex(0, 1 / (w * 3e-9)))
    assert c._get("R").Z_at(w) == pytest.approx(5.0 + 0j)


# ======================================================================= series RLC
@pytest.mark.parametrize("w_over_w0", [1.0, 0.9, 1.3, 0.5, 2.0])
def test_series_rlc_impedance_current_and_phase(w_over_w0):
    """|Z|, arg Z and the current of a series RLC against the closed form (C1.1)."""
    L, C, R = 1e-6, 1e-9, 0.5
    c = ck.Circuit()
    c.source("V", "gnd", "a", emf=1.0)
    c.resistor("R", "a", "b", R, T_K=0.02)
    c.inductor("L", "b", "d", L)
    c.capacitor("C", "d", "gnd", C)
    w0 = 1.0 / math.sqrt(L * C)
    w = w_over_w0 * w0
    Z_an = complex(R, -(w * L - 1.0 / (w * C)))
    assert c.driving_point_impedance(w, "V") == pytest.approx(Z_an, rel=1e-12, abs=1e-12)
    s = c.solve(w)
    assert s.I("L") == pytest.approx(1.0 / Z_an, rel=1e-12, abs=1e-18)
    # the same current flows in every series branch
    for nm in ("V", "R", "L", "C"):
        assert s.I(nm) == pytest.approx(1.0 / Z_an, rel=1e-12, abs=1e-18)
    assert c.check_reciprocity(w) == 0.0


def test_series_rlc_resonance_and_loaded_Q():
    """omega_0 = 1/sqrt(LC) and Q = omega_0 L/R from the reactance slope and the -3 dB width."""
    L, C, R = 1e-6, 1e-9, 0.5
    r = ck.series_readout(L_p_H=L, C_F=C, R_loss_ohm=R, f_Hz=1.0 / (2 * math.pi * math.sqrt(L * C)))
    w0 = r.resonance()
    assert w0 == pytest.approx(1.0 / math.sqrt(L * C), rel=1e-12)
    eq = r.equivalent(w0)
    assert eq["L_eff_H"] == pytest.approx(L, rel=1e-8)
    assert eq["R_eff_ohm"] == pytest.approx(R, rel=1e-12)
    assert eq["Q_loaded"] == pytest.approx(w0 * L / R, rel=1e-8)
    assert eq["react_resid"] < 1e-12
    # -3 dB full width of |H|^2 must agree with omega_0/Q to O(1/Q^2)
    Q = w0 * L / R
    ws = np.linspace(w0 * (1 - 5 / Q), w0 * (1 + 5 / Q), 20001)
    h2 = np.array([abs(r.H_signal(float(w))) ** 2 for w in ws])
    keep = ws[h2 >= h2.max() / 2]
    assert w0 / (keep[-1] - keep[0]) == pytest.approx(Q, rel=3 / Q)


def test_tune_hits_the_requested_frequency():
    """tune() solves Im Z_eff(2 pi f) = 0 exactly, for a plain loop and with leads."""
    L = 1.26e-6
    for kw in ({}, {"L_in_H": 3e-7, "L_lead_H": 1e-7}, {"L_in_H": 3e-7, "tan_delta": 1e-4}):
        for f in (5e4, 1.209e5, 9.9e5):
            r = ck.series_readout(L_p_H=L, Q=1e6, f_Hz=f, T_K=0.03, **kw)
            assert r.resonance(f) == pytest.approx(2 * math.pi * f, rel=1e-10)
            C = r.circuit._get("C").value
            L_tot = L + kw.get("L_in_H", 0.0) + kw.get("L_lead_H", 0.0)
            assert C == pytest.approx(1.0 / ((2 * math.pi * f) ** 2 * L_tot), rel=1e-6)


# ======================================================================= flux transfer
def test_flux_conservation_in_a_superconducting_loop():
    """A lossless pickup + input coil reproduces I = Phi/(L_p + L_in) at every frequency,
    and the flux transfer M/(L_p+L_in) peaks at the textbook match L_in = L_p  (C3.2)."""
    Lp, Lin, Lsq = 1e-6, 1e-6, 2e-9
    M = math.sqrt(Lin * Lsq)
    Phi = 1e-12
    for f in (1e2, 1e5, 1e7):
        w = 2 * math.pi * f
        c = ck.Circuit()
        c.source("pickup", "gnd", "a", emf=complex(0, -w) * Phi,
                 Z=lambda ww: complex(0, -ww * Lp))
        c.inductor("L_in", "a", "gnd", Lin)
        I = c.solve(w).I("L_in")
        assert I == pytest.approx(Phi / (Lp + Lin), rel=1e-12, abs=1e-24)
        assert abs(M * I) / Phi == pytest.approx(M / (Lp + Lin), rel=1e-12)
    # the match
    Li = np.geomspace(1e-9, 1e-3, 6001)
    best = Li[np.argmax(np.sqrt(Li * Lsq) / (Lp + Li))]
    assert best == pytest.approx(Lp, rel=2e-3)
    assert math.sqrt(Lp * Lsq) / (2 * Lp) == pytest.approx(0.5 * math.sqrt(Lsq / Lp), rel=1e-12)


@pytest.mark.parametrize("L1,L2,Lload,k", [(1e-6, 1e-6, 1e-7, 1.0), (1e-6, 4e-6, 1e-6, 1.0),
                                           (1e-6, 1e-6, 0.0, 0.5), (2e-6, 5e-7, 3e-7, 0.8)])
def test_flux_transformer_against_fluxoid_conservation(L1, L2, Lload, k):
    """Two coupled superconducting loops: I_1, I_2 from fluxoid conservation  (C3.3)."""
    c, M = ck.flux_transformer_circuit(L1, L2, k, Lload)
    w, Phi = 2 * math.pi * 1e5, 1e-12
    s = c.solve(w, {"drive": complex(0, -w) * Phi}, replace=True)
    D = L1 * (L2 * (1 - k * k) + Lload)
    assert s.I("L_1") == pytest.approx(Phi * (L2 + Lload) / D, rel=1e-10, abs=1e-24)
    assert s.I("L_load") == pytest.approx(-Phi * M / D, rel=1e-10, abs=1e-24)
    assert c.check_reciprocity(w) == pytest.approx(0.0, abs=1e-14)


def test_perfect_coupling_with_a_shorted_secondary_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        ck.flux_transformer_circuit(1e-6, 1e-6, k=1.0, L_load_H=0.0)
    with pytest.raises(ValueError, match=r"\|k\| must be <= 1"):
        ck.flux_transformer_circuit(1e-6, 1e-6, k=1.5, L_load_H=1e-7)


def test_ideal_transformer_is_lossless_and_reflects_impedance():
    """V_2 = n V_1, I_1 = -n I_2, V_1 I_1 + V_2 I_2 = 0, Z_in = Z_L/n^2  (Sec. 1.2)."""
    n, RL, Rs = 10.0, 100.0, 1.0
    c = ck.Circuit()
    c.source("V", "gnd", "a", emf=1.0, Z=Rs)
    c.transformer("T", "a", "gnd", "s", "gnd", n)
    c.resistor("RL", "s", "gnd", RL, T_K=0.0)
    w = 2 * math.pi
    s = c.solve(w)
    V1, V2, I2 = s.V("a"), s.V("s"), s.I("T")
    I1 = -n * I2
    assert V2 / V1 == pytest.approx(n, rel=1e-12)
    assert V1 * I1 + V2 * I2 == pytest.approx(0.0, abs=1e-25)
    assert V1 / (n * I2) == pytest.approx(RL / n ** 2, rel=1e-12)      # looking into the primary
    assert c.driving_point_impedance(w, "V") == pytest.approx(Rs + RL / n ** 2, rel=1e-12)
    assert c.check_reciprocity(w) == 0.0


# ======================================================================= Nyquist noise
def test_nyquist_series_resistors_at_different_temperatures():
    """Two resistors in series: the referred noise is sum_k R_k hbar w (2 n_k + 1), i.e. an
    equivalent noise temperature T_eq = (R_1 T_1 + R_2 T_2)/(R_1 + R_2) classically  (C4.1)."""
    R1, R2 = 1.0, 3.0
    w = 2 * math.pi * 1e5
    for T1, T2 in [(0.02, 0.02), (0.02, 4.0), (4.0, 0.02)]:
        c = ck.Circuit()
        c.source("V", "gnd", "a")
        c.resistor("R1", "a", "b", R1, T_K=T1)
        c.resistor("R2", "b", "gnd", R2, T_K=T2)
        th = c.thermal_noise(w, lambda s: s.I("V"))
        H = c.transfer(w, "V", lambda s: s.I("V"))
        N = th["total"] / abs(H) ** 2
        expect = HBAR * w * (R1 * (2 * ck.n_bose(w, T1) + 1) + R2 * (2 * ck.n_bose(w, T2) + 1))
        assert N == pytest.approx(expect, rel=1e-13)
        T_eq = (R1 * T1 + R2 * T2) / (R1 + R2)
        assert N == pytest.approx(2 * KB * T_eq * (R1 + R2), rel=1e-5)   # classical limit
        assert c.nyquist_check(w, "V") == pytest.approx(0.0, abs=1e-13)


def test_nyquist_parallel_resistors_equivalent_noise_temperature():
    """Two resistors in parallel at different T: S_V across them is 2 k T_eq R_par with
    R_par = R_1 R_2/(R_1+R_2) and T_eq = (T_1 R_2 + T_2 R_1)/(R_1+R_2)."""
    R1, R2, T1, T2 = 1.0, 3.0, 0.02, 4.0
    w = 2 * math.pi * 1e5
    c = ck.Circuit()
    c.resistor("R1", "a", "gnd", R1, T_K=T1)
    c.resistor("R2", "a", "gnd", R2, T_K=T2)
    S = c.thermal_noise(w, lambda s: s.V("a"))["total"]
    R_par = R1 * R2 / (R1 + R2)
    T_eq = (T1 * R2 + T2 * R1) / (R1 + R2)
    assert S == pytest.approx(R_par * HBAR * w * (2 * ck.n_bose(w, T_eq) + 1), rel=1e-12)
    assert S == pytest.approx(2 * KB * T_eq * R_par, rel=1e-5)


def _messy_network():
    """A deliberately non-trivial passive network: lossy source branch, lossy capacitor,
    mutual inductance, and a loaded secondary."""
    c = ck.Circuit()
    c.source("P", "gnd", "a", Z=lambda w: complex(0.3, -w * 2e-6), T_K=0.05)
    c.resistor("Rs", "a", "b", 0.7, T_K=0.05)
    c.capacitor("C", "b", "d", 2e-9, T_K=0.05, tan_delta=1e-3)
    c.inductor("L1", "d", "gnd", 3e-6)
    c.inductor("L2", "gnd", "e", 5e-6)
    c.mutual("L1", "L2", k=0.6)
    c.resistor("R2", "e", "f", 0.2, T_K=0.05)
    c.inductor("Lload", "f", "gnd", 1e-6)
    return c


@pytest.mark.parametrize("f", [3e4, 1.209e5, 7e5, 5e6])
def test_nyquist_identity_on_a_network_with_mutual_inductance(f):
    """Nyquist's theorem (C4.3): sum_b |H_b/H_P|^2 Re Z_b = Re Z_eff at every frequency.
    This is temperature-free, so it tests the whole noise-propagation machinery."""
    c = _messy_network()
    w = 2 * math.pi * f
    assert c.nyquist_check(w, "P") == pytest.approx(0.0, abs=1e-9)
    assert c.check_reciprocity(w) == pytest.approx(0.0, abs=1e-14)


def test_lossy_mutual_coupling_is_rejected():
    c = ck.Circuit()
    c.inductor("L1", "a", "gnd", 1e-6)
    c.resistor("R", "b", "gnd", 1.0, T_K=1.0)
    with pytest.raises(ValueError, match="not inductive"):
        c.mutual("L1", "R", M=1e-7)


# ======================================================================= matching
def test_maximum_power_transfer_at_the_conjugate_match():
    """P is maximal at Z_L = Z_s*, with P_max = |V|^2/(8 Re Z_s)."""
    Zs = complex(2.0, -3.0)
    w = 2 * math.pi * 1e5
    c = ck.Circuit()
    c.source("V", "gnd", "a", emf=1.0, Z=Zs)
    c.impedance("ZL", "a", "gnd", Zs.conjugate())
    P = 0.5 * abs(c.solve(w).I("ZL")) ** 2 * Zs.conjugate().real
    assert P == pytest.approx(1.0 / (8 * Zs.real), rel=1e-12)
    assert c.driving_point_impedance(w, "V") == pytest.approx(2 * Zs.real, rel=1e-12, abs=1e-12)
    # no other load does better
    a, b = np.meshgrid(np.linspace(0.2, 6, 60), np.linspace(-9, 9, 73))
    ZL = a + 1j * b
    assert (0.5 * np.abs(1.0 / (Zs + ZL)) ** 2 * a).max() <= P * (1 + 1e-12)


def test_xi_opt_and_G_bar_reproduce_the_spec_check_values():
    """xi_opt (S3.9) and G-bar (S3.8) against docs/physics/scan_and_signal.md Sec. 8."""
    ref = {(0, 1): (0.148148148, 0.5), (1, 1): (0.0611185247, 0.280776406),
           (10, 1): (9.15394060e-3, 0.0474050251), (100, 10): (9.56283279e-5, 0.495073653),
           (1e3, 0.1): (9.61769563e-4, 4.99750122e-6), (1e3, 10): (9.61757554e-6, 0.0499725165),
           (1e4, 20): (4.81100929e-7, 0.0199989601)}
    for (n, eta), (G_ref, xi_ref) in ref.items():
        xi = ck.xi_opt(n, eta)
        assert xi == pytest.approx(xi_ref, rel=1e-8)
        assert ck.G_bar(xi, n, eta) == pytest.approx(G_ref, rel=1e-8)


def test_scan_match_differs_from_the_noise_match_and_reproduces_BF_bandwidths():
    """At xi_opt the readout is backaction dominated and the sensitivity bandwidth is
    ~sqrt(3)(2n+1)/eta linewidths, vs ~2 sqrt(n) at the noise match xi = eta ([BF] Fig. 3:
    "~14" and "~173" at n = 50; note [BF]'s printed 2n/sqrt(3) is a typo for 2 sqrt(3) n)."""
    n, eta = 50.0, 1.0
    bw_noise_match = math.sqrt(ck.A_noise(eta, n, eta)) / eta
    xo = ck.xi_opt(n, eta)
    bw_opt = math.sqrt(ck.A_noise(xo, n, eta)) / xo
    assert bw_noise_match == pytest.approx(14.3, rel=0.02)
    assert bw_noise_match == pytest.approx(2 * math.sqrt(n), rel=0.02)
    assert bw_opt == pytest.approx(175.0, rel=0.02)
    assert bw_opt == pytest.approx(math.sqrt(3) * (2 * n + 1) / eta, rel=0.01)
    # the noise match maximises the on-resonance SNR xi/A, the scan match maximises G-bar
    xs = np.geomspace(1e-4, 1e2, 20001)
    assert xs[np.argmax(xs / np.array([ck.A_noise(x, n, eta) for x in xs]))] == \
        pytest.approx(eta, rel=2e-3)
    assert xs[np.argmax([ck.G_bar(x, n, eta) for x in xs])] == pytest.approx(xo, rel=2e-3)


def test_match_xi_rescales_M():
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta, xi=0.5)
    assert r.equivalent(W0)["xi"] == pytest.approx(0.5, rel=1e-8)
    M_before = r.squid.M_H
    r.match_xi(W0)
    eq = r.equivalent(W0)
    assert eq["xi"] == pytest.approx(eq["xi_opt"], rel=1e-8)
    assert r.squid.M_H == pytest.approx(M_before * math.sqrt(0.5 / eq["xi_opt"]), rel=1e-6)


# ============================================= consistency with the package's scan PSD
@pytest.mark.parametrize("toy", [TOY_A, TOY_B])
def test_trivial_circuit_reproduces_core_psd_at_tuning(toy):
    """The trivial circuit (pickup + tuning C + ideal amplifier, no input coil) must reproduce
    the C++ noise PSD term by term: spec (S3.6) is the series-loop special case of (C4.5)."""
    Psi_SI, L_SI, Q, T, eta = _toy(toy)
    xi = ck.xi_opt(ck.n_bose(W0, T), eta)      # the C++ default when xi <= 0
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta, xi=xi)
    eq = r.equivalent(W0)
    assert eq["L_eff_H"] == pytest.approx(L_SI, rel=1e-8)
    assert eq["Q_loaded"] == pytest.approx(Q, rel=1e-8)
    assert eq["T_eff_K"] == pytest.approx(T, rel=1e-10)
    assert eq["xi"] == pytest.approx(xi, rel=1e-10)
    assert eq["participation"] == pytest.approx(1.0, rel=1e-10)
    assert eq["R_noise_over_R_eff"] == pytest.approx(1.0, rel=1e-12)
    assert eq["backaction_transfer_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert eq["N_V_reduction_resid"] < 1e-12

    m_a = scan.freq_to_mass(F0)
    g = float(scan.g_qcd([m_a])[0])
    bw = eq["bandwidth_Hz"]["noise_doubling"]
    f = np.unique(np.concatenate([np.linspace(F0 - 30 * bw, F0 + 30 * bw, 41),
                                  scan.mass_to_freq(m_a) * (1 + np.linspace(0, 8e-6, 9))]))
    f = f[f > 0]
    ref = _core.psd_at_tuning(Psi_SI, L_SI, Q, T, eta, F0, m_a, g, list(f), 0.3)
    assert ref["xi"] == pytest.approx(xi, rel=1e-12)
    mine = [r.noise_psd(2 * math.pi * float(fi)) for fi in f]
    for key, mk in (("N_V_V2_s", "N_V_V2_s"), ("SV_th_V2_s", "N_V_thermal_V2_s"),
                    ("SV_ba_V2_s", "N_V_backaction_V2_s")):
        got = np.array([d[mk] for d in mine])
        assert np.allclose(got, np.asarray(ref[key]), rtol=1e-11), key
    # imprecision: |Z|^2 S_Phi/M^2 against the core's flat current noise S_Phi/M^2
    Z2 = np.array([abs(r.Z_eff(2 * math.pi * float(fi))) ** 2 for fi in f])
    imp = np.array([d["N_V_imprecision_V2_s"] for d in mine])
    assert np.allclose(imp, Z2 * np.asarray(ref["SI_imp_A2_s"]), rtol=1e-11)


@pytest.mark.parametrize("toy", [TOY_A, TOY_B])
def test_reach_from_the_reduction_equals_the_direct_scan_call(toy):
    """The trivial circuit's reduced inputs reproduce scan.reach() exactly."""
    Psi_SI, L_SI, Q, T, eta = _toy(toy)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta)
    a = r.analyze(F0, reach=True, n=12, f_max=9.9138e5)
    direct = scan.reach(Psi_SI, L_SI, {"Q": Q, "T": T, "eta_A": eta}, n=12, f_max=9.9138e5)
    assert np.allclose(a["reach"]["g95_GeV"], direct["g95_GeV"], rtol=1e-9)
    si = a["scan_inputs"]
    assert (si["L_H"], si["Q"], si["T_K"]) == pytest.approx((L_SI, Q, T), rel=1e-8)


@pytest.mark.parametrize("m_eV,g,rate_ref,toy", [
    (1e-9, 1e-19, 5.518206525e-3, TOY_A),
    (0.5e-9, None, 1.921385604e-3, TOY_A),
    (1e-9, None, 0.06148433407, TOY_A),
    (4.1e-9, None, 71.23328539, TOY_A),
    (1e-9, 1e-19, 9.196994606e-3, TOY_B)])
def test_kernel_times_bandwidth_reproduces_the_spec_scan_rate(m_eV, g, rate_ref, toy):
    """The narrow-line kernel (C5.3) times the equivalent bandwidth over Theta95 is the local
    optimal scan rate of spec (S5.2); check against the reference values of spec Sec. 8."""
    Psi_SI, L_SI, Q, T, eta = _toy(toy)
    g = g if g is not None else float(scan.g_qcd([m_eV])[0])
    f0 = scan.mass_to_freq(m_eV)
    w0 = 2 * math.pi * f0
    xi = ck.xi_opt(ck.n_bose(w0, T), eta)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=f0, Q=Q, T_K=T, eta_A=eta, xi=xi)
    eq = r.equivalent(w0)
    K = r.kernel(w0, g)["K_per_s"]
    rate = K * 2 * math.pi * eq["bandwidth_Hz"]["equivalent"] / 2.71
    assert rate == pytest.approx(rate_ref, rel=1e-8)
    # and the closed form S5.2 directly
    Phi_a = ck.kappa_a_SI(g) * Psi_SI
    closed = (math.pi ** 2 / 8) * (6.8e5 / 2.71) * Phi_a ** 4 / (HBAR ** 2 * L_SI ** 2) \
        * Q * eq["G_bar"]
    assert closed == pytest.approx(rate_ref, rel=1e-8)


# ======================================================================= input coil, leads
@pytest.mark.parametrize("ratio", [0.0, 0.1, 1.0, 3.0])
def test_input_coil_gives_the_Lp_over_Ltot_penalty(ratio):
    """L_eff = L_p + L_in, so F = Psi^4/L_eff^2 drops by (L_p/L_tot)^2 -- [C] Eqs. F26/F37."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_A)
    R = W0 * L_SI / Q
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, R_loss_ohm=R, T_K=T, eta_A=eta,
                          L_in_H=ratio * L_SI)
    eq = r.equivalent(W0)
    assert eq["L_eff_H"] == pytest.approx(L_SI * (1 + ratio), rel=1e-8)
    assert eq["participation"] == pytest.approx(1.0 / (1 + ratio), rel=1e-8)
    assert eq["Q_loaded"] == pytest.approx(W0 * L_SI * (1 + ratio) / R, rel=1e-8)
    # reach degrades as (L_eff/L_p)^{1/2}
    base = scan.reach(Psi_SI, L_SI, {"Q": eq["Q_loaded"], "T": T, "eta_A": eta}, n=4)
    got = scan.reach(Psi_SI, eq["L_eff_H"], {"Q": eq["Q_loaded"], "T": T, "eta_A": eta}, n=4)
    assert np.allclose(np.asarray(got["g95_GeV"]) / np.asarray(base["g95_GeV"]),
                       (1 + ratio) ** 0.5, rtol=1e-8)


def test_several_loss_channels_give_T_eff_over_Q_equal_sum_T_k_over_Q_k():
    """The multi-channel rule T_eff/Q = sum_k T_k/Q_k (spec Sec. 3.4) is
    *derived* by the network rather than assumed."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_A)
    R1, R2, T1, T2 = W0 * L_SI / Q, W0 * L_SI / (3 * Q), 0.010, 4.0
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, R_loss_ohm=R1, T_K=T1,
                          R_in_ohm=R2, T_in_K=T2, eta_A=eta)
    eq = r.equivalent(W0)
    Q1, Q2 = W0 * eq["L_eff_H"] / R1, W0 * eq["L_eff_H"] / R2
    assert eq["Q_loaded"] == pytest.approx(1 / (1 / Q1 + 1 / Q2), rel=1e-8)
    assert eq["T_over_Q_K"] == pytest.approx(T1 / Q1 + T2 / Q2, rel=1e-4)
    assert eq["T_eff_K"] == pytest.approx((R1 * T1 + R2 * T2) / (R1 + R2), rel=1e-4)
    assert set(eq["thermal_by_branch_Wb2_s"]) == {"pickup", "R_in"}
    assert eq["R_noise_over_R_eff"] == pytest.approx(1.0, rel=1e-12)


def test_leads_per_unit_length():
    r = ck.series_readout(L_p_H=1e-6, f_Hz=F0, Q=1e6, T_K=0.03,
                          lead_length_m=2.5, L_per_m=4e-7, R_per_m=2e-9, T_lead_K=4.0)
    assert r.circuit._get("L_lead").value == pytest.approx(2.5 * 4e-7)
    assert r.circuit._get("R_lead").value == pytest.approx(2.5 * 2e-9)
    eq = r.equivalent(W0)
    assert eq["L_eff_H"] == pytest.approx(1e-6 + 1e-6, rel=1e-8)
    assert "R_lead" in eq["thermal_by_branch_Wb2_s"]
    assert eq["T_eff_K"] > 0.03          # the 4 K leads raise the referred temperature


# ======================================================================= transformers
def test_ideal_transformer_stage_reflects_the_input_coil_but_not_eta_A():
    """An ideal transformer of ratio n puts L_in/n^2 in the resonant loop.  The backaction EMF
    is referred with |H_in/H_s|^2 = 1/n^2, but |M_eff|^2 = n^2 M^2 compensates exactly, so
    xi and eta_A_eff are untouched: an ideal transformer is a pure impedance scaling (Sec. 5)."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    n, Lin = 0.1, 1e-8
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                          transformer_ratio=n, L_in_H=Lin)
    eq = r.equivalent(W0)
    assert eq["L_eff_H"] == pytest.approx(L_SI + Lin / n ** 2, rel=1e-6)
    assert eq["backaction_transfer_ratio"] == pytest.approx(1.0 / n ** 2, rel=1e-6)
    assert abs(eq["M_eff_H"]) == pytest.approx(r.squid.M_H / n, rel=1e-6)
    assert eq["eta_A_eff"] == pytest.approx(eta, rel=1e-8)          # unchanged
    assert eq["port_coupling"] == pytest.approx(1.0, rel=1e-8)      # perfectly coupled ports
    assert eq["N_V_reduction_resid"] < 1e-12
    assert r.circuit.nyquist_check(W0, "pickup") == pytest.approx(0.0, abs=1e-12)
    assert eq["scan_inputs"]["eta_A"] == pytest.approx(eq["eta_A_eff"], rel=1e-12)


@pytest.mark.parametrize("k", [0.1, 0.5, 0.9, 1.0])
def test_real_flux_transformer_reflects_the_secondary_without_touching_eta_A(k):
    """Two coupled inductors: the loop sees L_1 - M^2/(L_2 + L_in), and eta_A is untouched at
    every k -- a lossless reciprocal coupling network is a pure impedance transformation."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    L1, L2, Lin = 0.5 * L_SI, 1e-8, 1e-8
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                          L_1_H=L1, L_2_H=L2, k_T=k, L_in_H=Lin)
    eq = r.equivalent(W0)
    M = k * math.sqrt(L1 * L2)
    assert eq["L_eff_H"] == pytest.approx(L_SI + L1 - M ** 2 / (L2 + Lin), rel=1e-6)
    assert eq["port_coupling"] == pytest.approx(1.0, rel=1e-6)
    assert eq["eta_A_eff"] == pytest.approx(eta, rel=1e-6)
    assert eq["N_V_reduction_resid"] < 1e-12
    assert r.circuit.nyquist_check(W0, "pickup") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kw", [
    {"L_in_H": 1e-8},                                                   # plain series
    {"transformer_ratio": 0.1, "L_in_H": 1e-8},                         # ideal transformer
    {"L_1_H": 6e-7, "L_2_H": 1e-8, "k_T": 0.5, "L_in_H": 1e-8},         # flux transformer
    {"C_par_F": 0.3},    # capacitive tap (a fraction of 1/(omega^2 L_p); fixed up below)
])
def test_port_coupling_identity_and_lossless_invariance_of_eta_A(kw):
    """(C5.1a)-(C5.1b): port_coupling = |T_p,in|^2/(|T_pp||T_in,in|), and when all the loss is
    in the pickup branch that equals Re(T_in,in)/|T_in,in| -- which is 1 for every lossless
    coupling network, so eta_A_eff = eta_A."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    C_res = 1.0 / (W0 ** 2 * L_SI)
    kw = dict(kw)
    if "C_par_F" in kw:
        kw["C_par_F"] *= C_res            # express the tap as a fraction of the resonant C
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta, **kw)
    eq = r.equivalent(W0)
    c = r.circuit
    T_pp = c.transfer(W0, "pickup", lambda s: s.I("pickup"))
    T_ii = c.transfer(W0, "L_in", lambda s: s.I("L_in"))
    T_pi = c.transfer(W0, "pickup", lambda s: s.I("L_in"))
    assert eq["port_coupling"] == pytest.approx(abs(T_pi) ** 2 / (abs(T_pp) * abs(T_ii)),
                                               rel=1e-9)                       # (C5.1a)
    assert eq["port_coupling"] == pytest.approx(T_ii.real / abs(T_ii), rel=1e-8)   # (C5.1b)
    assert eq["port_coupling"] == pytest.approx(1.0, rel=1e-6)
    assert eq["eta_A_eff"] == pytest.approx(eta, rel=1e-6)
    # with dissipation in the coupling path it does depart from 1
    r2 = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                           C_par_F=0.3 * C_res, tan_delta_par=1e-2)
    assert abs(r2.equivalent(W0)["port_coupling"] - 1.0) > 1e-6


def test_parasitic_capacitance_is_a_lossless_capacitive_tap():
    """A parasitic C across the pickup and the tuning C form a capacitive divider.  With the
    loop retuned, C_tune = 1/(omega^2 L_p) - C_par and the tap ratio is
    t = C_tune/(C_par + C_tune) = 1 - omega^2 L_p C_par: L_eff, Q, T_eff and eta_A_eff are all
    untouched (the tap is lossless and reciprocal, so port_coupling = 1 exactly), while
    M_eff -> t M and hence xi -> xi/t^2.  A parasitic capacitance is therefore a *coupling*
    knob, not a loss.  At C_par = 1/(omega^2 L_p) -- the pickup self-resonance -- the loop
    cannot be tuned at all."""
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    C_res = 1.0 / (W0 ** 2 * L_SI)
    frac = 0.3
    t = 1.0 - frac
    base = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                          C_par_F=frac * C_res, M_H=base.squid.M_H, S_Phi=base.squid.S_Phi)
    eq, eq0 = r.equivalent(W0), base.equivalent(W0)
    assert r.circuit._get("C").value == pytest.approx(t * C_res, rel=1e-6)
    assert eq["L_eff_H"] == pytest.approx(L_SI, rel=1e-6)
    assert eq["Q_loaded"] == pytest.approx(eq0["Q_loaded"], rel=1e-6)
    assert eq["T_eff_K"] == pytest.approx(eq0["T_eff_K"], rel=1e-6)
    # the signal current divides: a fraction t reaches the input coil
    assert abs(eq["H_signal_Wb_per_V"] / eq0["H_signal_Wb_per_V"]) == pytest.approx(t, rel=1e-5)
    assert abs(eq["M_eff_H"]) == pytest.approx(t * abs(eq0["M_eff_H"]), rel=1e-5)
    assert eq["xi"] == pytest.approx(eq0["xi"] / t ** 2, rel=1e-5)
    # lossless reciprocal coupling: eta_A is untouched
    assert eq["port_coupling"] == pytest.approx(1.0, rel=1e-6)
    assert eq["eta_A_eff"] == pytest.approx(eta, rel=1e-6)
    # the reduction and the Nyquist identity stay exact on this non-series network
    assert eq["react_resid"] < 1e-8
    assert eq["N_V_reduction_resid"] < 1e-10
    assert eq["backaction_transfer_ratio"] == pytest.approx(t ** 2, rel=1e-5)
    assert r.circuit.nyquist_check(W0, "pickup") == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="resonate|bracket|capacitance"):
        ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                          C_par_F=1.2 * C_res)


# ======================================================================= pickup impedance
def test_impedance_table_is_exact_at_the_nodes_and_mqs_consistent():
    f = np.geomspace(1e5, 2e7, 7)
    L = 1.26e-6 * (1 + (f / 4e7) ** 2)
    R = 1e-7 * (f / 1e6) ** 4
    Z = ck.impedance_table(f, R_ohm=R, L_H=L)
    for i, fi in enumerate(f):
        w = 2 * math.pi * fi
        assert Z(w) == pytest.approx(complex(R[i], -w * L[i]), rel=1e-14, abs=1e-20)
    # between nodes L is interpolated, so -Im Z/omega stays within the bracketing L values
    for fi in (1.5e5, 3e6):
        w = 2 * math.pi * fi
        assert L.min() <= -Z(w).imag / w <= L.max()
    with pytest.raises(ValueError, match="outside the tabulated range"):
        ck.impedance_table(f, R_ohm=R, L_H=L, extrapolate="error")(2 * math.pi * 1e3)
    assert ck.impedance_table(f, R_ohm=R, L_H=L, extrapolate="const")(2 * math.pi * 1e3).real \
        == pytest.approx(R[0])


def test_pickup_impedance_from_tabulated_full_wave_sweep():
    sweep = [{"f_Hz": 1e5, "Z_re_ohm": 1e-9, "L_H": 1.0e-6, "R_loss_ohm": 2e-9},
             {"f_Hz": 1e6, "Z_re_ohm": 1e-5, "L_H": 1.1e-6, "R_loss_ohm": 2e-7},
             {"f_Hz": 1e7, "Z_re_ohm": 1e-1, "L_H": 2.0e-6}]
    Z = ck.pickup_impedance({"fullwave": {"sweep": sweep}}, model="fullwave")
    w = 2 * math.pi * 1e6
    assert Z(w) == pytest.approx(complex(1e-5 + 2e-7, -w * 1.1e-6), rel=1e-12)
    with pytest.raises(ValueError, match="no 'fullwave'"):
        ck.pickup_impedance({}, model="fullwave")


def test_pickup_impedance_from_surface_losses_recovers_the_power_law():
    """The solver stores R_eff and Q at the scan band edges but not the band; the band is
    recovered from Q = 2 pi f L_p/R_eff and R(f) is a power law through the two points."""
    L_p, f_lo, f_hi, Rs0, p = 7.43e-7, 1.209e5, 9.9138e5, 3.294372283497383e-11, 2.0
    R_hi = Rs0 * (f_hi / f_lo) ** p
    results = {"coupling": {"L_p_H": L_p},
               "pickup": {"losses": {"surfaces": {"sheath": {
                   "R_eff_ohm": {"f_min": Rs0, "f_max": R_hi},
                   "Q": {"f_min": 2 * math.pi * f_lo * L_p / Rs0,
                         "f_max": 2 * math.pi * f_hi * L_p / R_hi}, "T_K": 0.03}}}}}
    Z = ck.pickup_impedance(results)
    for f in (f_lo, 3.5e5, f_hi):
        w = 2 * math.pi * f
        assert Z(w).real == pytest.approx(Rs0 * (f / f_lo) ** p, rel=1e-10)
        assert -Z(w).imag / w == pytest.approx(L_p, rel=1e-14)


def test_pickup_impedance_from_Q():
    Z = ck.pickup_impedance(L_p_H=1e-6, Q=1e6)
    w = 2 * math.pi * F0
    assert Z(w).real == pytest.approx(w * 1e-6 / 1e6, rel=1e-14)
    Zc = ck.pickup_impedance(L_p_H=1e-6, Q=1e6, f_ref_Hz=F0)
    assert Zc(2 * w).real == pytest.approx(w * 1e-6 / 1e6, rel=1e-14)   # constant R


# ======================================================================= SQUID, signal
def test_squid_from_flux_noise_and_xi_round_trip():
    sq = ck.Squid.from_flux_noise(1e-6, eta_A=10.0, M_H=1e-9)       # 1 uPhi0/sqrt(Hz)
    assert math.sqrt(2 * sq.S_Phi) / PHI0 == pytest.approx(1e-6, rel=1e-12)
    assert 2 * math.sqrt(sq.S_Phi * sq.S_J) / HBAR == pytest.approx(10.0, rel=1e-12)
    sq2 = ck.Squid.from_xi(0.3, W0, 1e-7, 1e-9, eta_A=2.0)
    assert sq2.xi_at(W0, 1e-7) == pytest.approx(0.3, rel=1e-12)
    assert ck.Squid.M_for_xi(0.3, W0, 1e-7, sq2.S_Phi) == pytest.approx(1e-9, rel=1e-12)


def test_signal_flux_scales_with_g_and_Psi():
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta)
    s1 = r.signal(W0, 1e-19)
    s2 = r.signal(W0, 2e-19)
    assert abs(s2["Phi_SQUID_Wb"]) == pytest.approx(2 * abs(s1["Phi_SQUID_Wb"]), rel=1e-12)
    assert s1["Phi_SQUID_Phi0"] == pytest.approx(abs(s1["Phi_SQUID_Wb"]) / PHI0, rel=1e-14)
    assert abs(s1["V_emf_V"]) == pytest.approx(W0 * s1["Phi_a_Wb"], rel=1e-12)
    # on resonance the EMF drives I = V/R and the flux is M I
    eq = r.equivalent(W0)
    assert abs(s1["I_pickup_A"]) == pytest.approx(abs(s1["V_emf_V"]) / eq["R_eff_ohm"], rel=1e-8)
    assert abs(s1["Phi_SQUID_Wb"]) == pytest.approx(r.squid.M_H * abs(s1["I_pickup_A"]), rel=1e-8)


def test_sweep_and_summary_run():
    Psi_SI, L_SI, Q, T, eta = _toy(TOY_B)
    r = ck.series_readout(L_p_H=L_SI, Psi_SI=Psi_SI, f_Hz=F0, Q=Q, T_K=T, eta_A=eta,
                          L_in_H=0.2 * L_SI)
    sw = r.sweep(np.linspace(F0 * 0.999, F0 * 1.001, 11))
    assert sw["N_V_V2_s"].min() > 0
    assert np.allclose(sw["N_V_V2_s"], sw["N_V_thermal_V2_s"] + sw["N_V_backaction_V2_s"]
                       + sw["N_V_imprecision_V2_s"], rtol=1e-12)
    # |H_signal| peaks at the resonance
    assert abs(sw["f_Hz"][np.argmax(sw["H_signal_Wb_per_V"])] / F0 - 1) < 2e-3
    txt = r.summary(F0)
    for key in ("L_eff", "Q_loaded", "T_eff", "xi", "G_bar", "scan inputs", "diagnostics"):
        assert key in txt


# ======================================================================= Results.readout
class _StubResults:
    """A Results-shaped stub: Results.readout only needs data, Psi, L_p, design.name and
    benchmark().  Using the real method keeps the test on the real code path without an
    FEM solve (the end-to-end path is exercised in examples/readout_circuit.py)."""

    def __init__(self, data, name="stub"):
        self.data = data
        self.design = type("D", (), {"name": name})()

    Psi = property(lambda self: self.data["coupling"]["Psi_SI_H_T_m2"])
    L_p = property(lambda self: self.data["coupling"]["L_p_H"])

    def benchmark(self, name):
        return {"amplitude_scale": 126.8}


def _stub(with_losses=True):
    from axionfem.results import Results
    L_p = 7.43e-7
    data = {"coupling": {"Psi_SI_H_T_m2": 7.7e-8, "L_p_H": L_p},
            "pickup": {"port_capacitance": {"C_gap_F": 5e-11}}}
    if with_losses:
        R_lo, f_lo, f_hi = 3.3e-11, 1.209e5, 9.9138e5
        R_hi = R_lo * (f_hi / f_lo) ** 2
        data["pickup"]["losses"] = {
            "T_over_Q_total_K": {"f_min": 1.75e-12, "f_max": 1.44e-11},
            "surfaces": {"sheath": {"R_eff_ohm": {"f_min": R_lo, "f_max": R_hi},
                                    "Q": {"f_min": 2 * math.pi * f_lo * L_p / R_lo,
                                          "f_max": 2 * math.pi * f_hi * L_p / R_hi},
                                    "T_K": 0.03}}}
    s = _StubResults(data)
    s.readout = Results.readout.__get__(s)   # the real method, bound to the stub
    return s


def test_results_readout_uses_the_design_losses():
    out = _stub().readout(f_Hz=F0, benchmark="conservative", L_in_H=1e-7)
    assert out["design"] == "stub" and out["model"] == "mqs"
    assert out["f0_Hz"] == pytest.approx(F0, rel=1e-9)
    assert out["R_eff_ohm"] == pytest.approx(3.3e-11, rel=1e-3)
    assert out["L_eff_H"] == pytest.approx(7.43e-7 + 1e-7, rel=1e-6)
    assert out["T_eff_K"] == pytest.approx(0.030, rel=1e-6)
    assert out["eta_A"] == pytest.approx(10.0)
    assert out["scan_inputs"]["Psi_SI_H_T_m2"] == pytest.approx(7.7e-8)
    assert out["R_noise_over_R_eff"] == pytest.approx(1.0, rel=1e-10)
    assert isinstance(out["readout"], ck.Readout)


def test_results_readout_without_loss_data_needs_a_Q():
    with pytest.raises(ValueError, match="no RF loss data"):
        _stub(with_losses=False).readout(f_Hz=F0)
    out = _stub(with_losses=False).readout(f_Hz=F0, pickup_Q=1e6, benchmark="conservative")
    assert out["Q_loaded"] == pytest.approx(1e6, rel=1e-6)
    out2 = _stub(with_losses=False).readout(f_Hz=F0, benchmark="conservative")   # benchmark Q
    assert out2["Q_loaded"] == pytest.approx(1e6, rel=1e-6)


def test_results_readout_amplitude_required_and_parasitic():
    base = _stub().readout(f_Hz=F0)
    req = _stub().readout(f_Hz=F0, benchmark="conservative", amplitude="required")
    assert req["signal"]["Phi_a_Wb"] == pytest.approx(126.8 * base["signal"]["Phi_a_Wb"], rel=1e-9)
    # the parasitic gap capacitance adds to the tuning capacitance, so C_tune comes out smaller
    par = _stub().readout(f_Hz=F0, benchmark="conservative", parasitic=True)
    C_gap = 5e-11
    assert par["C_tuned_F"] == pytest.approx(base["C_tuned_F"] - C_gap, rel=1e-6)
    assert par["L_eff_H"] == pytest.approx(base["L_eff_H"], rel=1e-6)   # a lossless tap
    with pytest.raises(ValueError, match="amplitude must be"):
        _stub().readout(f_Hz=F0, amplitude="nonsense")


# ======================================================================= error paths
def test_error_messages_are_specific():
    with pytest.raises(ValueError, match="give L_p_H"):
        ck.series_readout()
    with pytest.raises(ValueError, match="must be > 0"):
        ck.series_readout(L_p_H=-1.0, Q=1e6)
    with pytest.raises(ValueError, match="not both"):
        ck.series_readout(L_p_H=1e-6, Q=1e6, R_loss_ohm=1.0)
    with pytest.raises(ValueError, match="no dissipation"):
        ck.series_readout(L_p_H=1e-6, f_Hz=F0)
    with pytest.raises(ValueError, match="duplicate element name"):
        c = ck.Circuit()
        c.resistor("R", "a", "gnd", 1.0)
        c.inductor("R", "a", "gnd", 1e-6)
    with pytest.raises(ValueError, match="both terminals on node"):
        ck.Circuit().resistor("R", "a", "a", 1.0)
    with pytest.raises(ValueError, match="no element named"):
        ck.Circuit().mutual("L1", "L2", M=1.0)
    with pytest.raises(ValueError, match="singular"):
        c = ck.Circuit()
        c.source("V", "gnd", "a", emf=1.0)
        c.short("S", "a", "gnd")
        c.solve(W0)
    with pytest.raises(ValueError, match="lossless"):
        c = ck.Circuit()
        c.source("pickup", "gnd", "a", Z=lambda w: complex(0, -w * 1e-6))
        c.capacitor("C", "a", "gnd", 1e-9)
        c.inductor("L_in", "a", "b", 1e-9)       # dangling, keeps the matrix regular
        c.resistor("x", "b", "gnd", 0.0)
        ck.Readout(circuit=c, squid=ck.Squid(M_H=1e-9), pickup="pickup").equivalent(W0)


def test_capacitive_pickup_above_self_resonance_is_reported():
    """Above the pickup self-resonance a series tuning capacitor cannot resonate the loop."""
    Z = ck.impedance_table([1e5, 1e9], R_ohm=[1e-9, 1e-3], L_H=[1e-6, -1e-6])
    with pytest.raises(ValueError, match="capacitive"):
        ck.series_readout(Z_p=Z, f_Hz=9e8, T_K=0.03)


@pytest.mark.parametrize("kw", [dict(L_p_H=7.43e-7, L_in_H=2e-9), dict(L_p_H=1.4e-6, L_in_H=2e-9),
                                dict(L_p_H=1.4e-6, L_in_H=2e-9, C_par_F=1e-12),
                                dict(L_p_H=1.4e-6, L_in_H=2e-9, L_lead_H=5e-8, R_lead_ohm=1e-4, T_lead_K=4.0)])
def test_resonance_default_guess_includes_the_pickup_inductance(kw):
    """_resonance_guess must include the pickup inductance: summing only the L branches leaves L_p out with an
    input coil (L_in << L_p), guesses >= 19 MHz for a 1 MHz loop, and resonance() then raises."""
    r = ck.series_readout(**kw, f_Hz=1e6, Q=1e6, T_K=0.03)
    assert r._resonance_guess() == pytest.approx(1e6, rel=0.05)
    assert r.resonance() / (2 * math.pi) == pytest.approx(1e6, rel=1e-9)


def test_resonance_guess_does_not_count_an_explicit_pickup_inductor_twice():
    """A hand-built loop whose pickup is an ideal source and whose L_p is an inductor branch: the
    branch is counted once (L_p_H is for the participation ratio only)."""
    L, C = 1.4e-6, 1.0 / ((2 * math.pi * 1e6) ** 2 * 1.4e-6)
    c = ck.Circuit()
    c.source("pickup", "gnd", "a", emf=1.0)
    c.inductor("L_p", "a", "b", L)
    c.resistor("R", "b", "d", 1e-5, T_K=0.03)
    c.capacitor("C", "d", "gnd", C)
    r = ck.Readout(circuit=c, squid=ck.Squid(M_H=1e-9), L_p_H=L, tuner="C")
    assert r._resonance_guess() == pytest.approx(1e6, rel=1e-12)
    assert r.resonance() / (2 * math.pi) == pytest.approx(1e6, rel=1e-9)
