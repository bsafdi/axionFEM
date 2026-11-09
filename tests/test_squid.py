"""SQUID amplifier chain (axionfem.squid; docs/physics/squid_chain.md).

Tests use analytic anchors only: the closed forms of [DMR-SQUID] (arXiv:2504.20398) Eqs. 9-48,
its Tables I-II, and stored numbers of an independent implementation of the same equations.  Figs. 3-4 were not
digitized, so their curves are tested only for shape (ordering, monotonicity), never for values.
"""
from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from axionfem import circuit as ck, scan, squid
from axionfem.circuit import HBAR, KB, PHI0
from axionfem.multiport import MultiportCircuit, MultiportReadout

C1 = squid.C1
A_C1 = KB * C1.T_j_K * C1.L_sq_H / (HBAR * C1.R_j_ohm)    # k T L/(hbar R), the TC unit


# ============================================================== first stage, Eqs. 1-29
def test_c1_tc_anchors():
    """Sec. III B: eta_TC ~ 3.7 (Eq. 29), ~9x SQL on resonance (Eq. 24); Eq. 13; Eq. 1 regime."""
    assert C1.eta_TC == pytest.approx(4 * math.sqrt(2) * A_C1, rel=1e-12, abs=0)
    assert abs(C1.eta_TC - 3.70) < 0.01
    assert C1.on_resonance_over_SQL == pytest.approx(4 * math.sqrt(11) * A_C1, rel=1e-12, abs=0)
    assert 8.5 < C1.on_resonance_over_SQL < 9.0
    assert C1.eps_uc_J_s == pytest.approx(8 * KB * C1.T_j_K * C1.L_sq_H / C1.R_j_ohm, rel=1e-12, abs=0)
    assert C1.eps_uc_J_s / HBAR == pytest.approx(math.sqrt(2) * C1.eta_TC, rel=1e-12, abs=0)
    v = C1.validity()
    assert v["warnings"] == [] and not v["beta_C_checked"]
    # TC beta_L = 2 L I0/Phi0 (Eq. 1 prints half of it; App. A's 0.8 R/L needs the factor 2)
    assert v["beta_L"] == pytest.approx(2 * 200e-12 * 6.3e-6 / PHI0, rel=1e-12, abs=0) and 1.2 < v["beta_L"] < 1.23
    assert v["beta_L_eq1_printed"] == pytest.approx(v["beta_L"] / 2, rel=1e-15, abs=0)
    assert v["Gamma"] == pytest.approx(2 * math.pi * KB * 0.15 / (6.3e-6 * PHI0)) and v["Gamma"] < 1.1e-3
    p = C1.one_sided()
    assert math.sqrt(p["S1_Phi_Wb2_Hz"] / p["S1_J_A2_Hz"]) == pytest.approx(4 / math.sqrt(11) * C1.L_sq_H, rel=1e-12, abs=0)
    assert p["S1_PhiJ_WbA_Hz"] / math.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"]) == pytest.approx(12 / math.sqrt(176))


def test_tc_regime_warnings():
    with pytest.warns(UserWarning, match="Gamma"):
        squid.FirstStage(200e-12, 6.3e-6, 6.0, 8.0)                     # Gamma ~ 0.05+
    with pytest.warns(UserWarning, match="beta_L"):
        squid.FirstStage(2e-9, 6.3e-6, 6.0, 0.15)
    with pytest.warns(UserWarning, match="beta_C"):
        squid.FirstStage(200e-12, 6.3e-6, 6.0, 0.15, C_j_F=1e-11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fs = squid.FirstStage(200e-12, 6.3e-6, 6.0, 0.15, C_j_F=1.45e-12)   # beta_C ~ 1.0
    assert fs.beta_C == pytest.approx(2 * math.pi * 6.3e-6 * 36 * 1.45e-12 / PHI0, rel=1e-12, abs=0)
    with pytest.warns(UserWarning, match="T2"):
        squid.SecondStage(48, 3, T2_K=0.1)
    # a slow device: f_J = 0.38 I0 R/Phi0 ~ 18 MHz, so 10 MHz is not << f_J
    slow = squid.FirstStage(10e-9, 1e-7, 1.0, 1e-3)
    with pytest.warns(UserWarning, match="f_J"):
        squid.Chain(slow).eta(1e7)
    with pytest.raises(ValueError, match="positive"):
        squid.FirstStage(200e-12, 6.3e-6, -6.0, 0.15)


def test_at_eta_rescales_temperature_only():
    b = C1.at_eta(10.0)
    assert b.eta_TC == pytest.approx(10.0, rel=1e-12, abs=0)
    assert b.T_j_K == pytest.approx(0.405078, rel=1e-5, abs=0)          # T_eff of the independent implementation
    assert b.eps_uc_J_s / HBAR == pytest.approx(10 * math.sqrt(2), rel=1e-12, abs=0)


@pytest.mark.parametrize("kappa,L_in", [(0.5, 100e-9), (0.9, 2e-6)])
def test_input_referred_eqs_9_to_12_and_on_resonance_match(kappa, L_in):
    ch = squid.chain("C1")
    f, Q = 2e6, 1e6
    w = 2 * math.pi * f
    M = kappa * math.sqrt(C1.L_sq_H * L_in)                               # Eq. 8
    kT, L, R = KB * C1.T_j_K, C1.L_sq_H, C1.R_j_ohm
    r = ch.input_referred(f, M)
    assert r["S_II_A2_Hz"] == pytest.approx(16 * kT * L / (kappa ** 2 * R * L_in), rel=1e-12, abs=0)       # Eq. 9
    assert r["S_VV_V2_Hz"] == pytest.approx(11 * kT * kappa ** 2 * L_in * L * w ** 2 / R, rel=1e-12, abs=0)  # Eq. 11
    assert r["S_IV_W_Hz"].real == 0 and abs(r["S_IV_W_Hz"].imag) == pytest.approx(12 * kT * L * w / R, rel=1e-12, abs=0)  # Eq. 12
    on = ch.on_resonance(f, Q=Q, M_H=M)
    assert on["R_match_ohm"] == pytest.approx(math.sqrt(11) / 4 * kappa ** 2 * w * L_in, rel=1e-12, abs=0)   # Eq. 20
    assert on["kappa_g2_match"] == pytest.approx(4 / (math.sqrt(11) * Q), rel=1e-12, abs=0)                 # Eq. 22
    assert KB * on["T_min_K"] == pytest.approx(2 * math.sqrt(11) * kT * L * w / R, rel=1e-12, abs=0)        # Eq. 23
    # Eq. 17 at the matched real source reproduces Eq. 19
    assert ch.noise_temperature(f, on["R_match_ohm"], M) == pytest.approx(on["T_min_K"], rel=1e-12, abs=0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_complex_match_eqs_25_26(sign):
    ch = squid.chain("C1", sign=sign)
    f, M = 5e6, 3e-9
    w = 2 * math.pi * f
    Tm = ch.T_min_K(f)
    assert KB * Tm == pytest.approx(2 * math.sqrt(2) * KB * C1.T_j_K * C1.L_sq_H * w / C1.R_j_ohm, rel=1e-12, abs=0)  # Eq. 26
    Z = ch.optimal_source_impedance(f, M)
    assert ch.noise_temperature(f, Z, M) == pytest.approx(Tm, rel=1e-12, abs=0)
    assert np.sign(Z.imag) == -sign
    for dz in (0.05, -0.05, 0.05j, -0.05j):
        assert ch.noise_temperature(f, Z + dz * abs(Z), M) > Tm
    # Clarke's detuned-LCR optimum 2.8 k T L w/R (Eq. 27) is within 1% of 2 sqrt2
    assert abs(2.8 / (2 * math.sqrt(2)) - 1) < 0.011


# ============================================================== second stage and amplifiers, Eqs. 30-40
@pytest.mark.parametrize("bw,L2_nH,L2_table_nH,tau_ns,P_nW", [("low", 20.0, 20, 3.3, 6.00),
                                                                ("medium", 10.16, 10, 1.7, 2.67),
                                                                ("high", 4.16, 4, 0.7, 0.67)])
def test_table_II(bw, L2_nH, L2_table_nH, tau_ns, P_nW):
    ch = squid.chain(f"C1+{bw}+RT")
    st = ch.second
    assert st.L2_H * 1e9 == pytest.approx(L2_nH, rel=1e-12, abs=0)              # N_par(N_ser+2) L_SQ2 + L_int
    assert abs(st.L2_H * 1e9 - L2_table_nH) < 0.2                        # Table II rounding (text's "~3 nH" is not)
    assert ch.tau_s * 1e9 == pytest.approx(tau_ns, abs=0.05)             # tau = L2/R_dyn1, R_dyn1 = 6 ohm
    assert st.P_W * 1e9 == pytest.approx(P_nW, abs=0.006)                # Eq. 30
    assert abs(st.R_dyn_ohm - 50) < 2.5                                  # Eq. 32: designed for 50 ohm


def test_table_I_internal_consistency():
    """Eqs. 38-39 at 20x1 (no lead term) reproduce Table I's referred preamp noise, and its
    quadrature subtraction the tabulated SQUID noise, in both columns."""
    for col in (squid.TABLE_I_PLUS, squid.TABLE_I_MINUS):
        r = squid.table_I_check(col)
        assert r["preamp_referred"] == pytest.approx(r["preamp_referred_table"], rel=3e-3, abs=0)
        assert r["squid_by_quadrature"] == pytest.approx(r["squid_table"], rel=7e-3, abs=0)
        assert r["chain_rule_dV_dI_ohm"] == pytest.approx(r["dV_dI_table_ohm"], rel=1e-3, abs=0)   # |B1| = 1


def test_chain_referred_noise_by_hand():
    """Eqs. 35-39 written in the paper's own (20/N) form, low-BW 48x3 at 1 K, XXF-1, 1 MHz."""
    ch = squid.chain("C1+low+RT")
    f = 1e6
    ns, npar, T2 = 48, 3, 1.0
    phi_sq = (0.297e-6 * PHI0) ** 2 * 20 / (ns * npar) * T2 / 4.0                           # Eq. 36
    v = (0.33e-9 ** 2 + 4 * KB * 200 * 1.0) / 6.80e11 ** 2 * (20 / ns) ** 2                  # Eq. 38
    i = (2.6e-12 * 95.5e-12 / npar) ** 2                                                    # Eq. 39
    tau = (npar * (ns + 2) * 120e-12 + 2e-9) / 6.0
    ref = (200e-12 / 105e-12) ** 2 * (phi_sq + v + i) * (1 + (2 * math.pi * f * tau) ** 2)  # Eq. 35 x M^2
    p = ch.one_sided(f)
    assert float(p["S1_Phi_ref_Wb2_Hz"]) == pytest.approx(ref, rel=1e-12, abs=0)
    assert float(p["S1_Phi_ref_parts_Wb2_Hz"]["second_stage_squid"]) == pytest.approx(
        (200e-12 / 105e-12) ** 2 * phi_sq * (1 + (2 * math.pi * f * tau) ** 2), rel=1e-12, abs=0)
    # roll-off: the referred part grows exactly as 1 + omega^2 tau^2
    f2 = 40e6
    p2 = ch.one_sided(f2)
    assert float(p2["S1_Phi_ref_Wb2_Hz"] / p["S1_Phi_ref_Wb2_Hz"]) == pytest.approx(
        (1 + (2 * math.pi * f2 * tau) ** 2) / (1 + (2 * math.pi * f * tau) ** 2), rel=1e-12, abs=0)


def test_rf_amplifier_noise_is_positive_eq_C1():
    """Eq. 40's leading minus sign is the triple-product sign of (dPhi/dI)_V; with Table I's
    magnitudes the referred noise is 4 k T_n R_dyn/(dV/dPhi)^2 > 0 (Eq. C1)."""
    for bw, (ns, npar) in squid.SECOND_STAGES.items():
        st = squid.SecondStage(ns, npar)
        s = squid.CITLF2.S1_Phi2(st)["rf_amp"]
        assert s > 0
        assert s == pytest.approx(4 * KB * 2.0 * 95.5e-12 / 6.80e11 * 20 / (ns * npar), rel=1e-12, abs=0)          # Eq. C1
        assert s == pytest.approx(4 * KB * 2.0 * st.R_dyn_ohm / st.dV_dPhi_Hz ** 2, rel=1e-3, abs=0)   # matched source
    ch = squid.chain("C1+medium+cryo")
    assert float(ch.one_sided(2e7)["S1_Phi_ref_parts_Wb2_Hz"]["rf_amp"]) > 0


# ============================================================== full system, Eqs. 41-48
@pytest.mark.parametrize("preset", ["C1", "C1+low+RT", "C1+high+cryo"])
def test_eta_eps_eqs_42_43_are_coupling_independent(preset):
    ch = squid.chain(preset)
    f = 20e6
    w = 2 * math.pi * f
    for M in (1e-10, 3e-9):
        r = ch.input_referred(f, M)
        eta = math.sqrt(r["S_VV_V2_Hz"] * r["S_II_sys_A2_Hz"] - r["S_IV_W_Hz"].imag ** 2) / (HBAR * w)   # Eq. 42
        assert eta == pytest.approx(ch.eta(f), rel=1e-12, abs=0)
        assert M ** 2 * r["S_II_sys_A2_Hz"] / (2 * C1.L_sq_H) == pytest.approx(ch.eps_uc(f), rel=1e-12, abs=0)  # Eq. 43
        assert r["S_II_sys_A2_Hz"] == pytest.approx(r["S_II_A2_Hz"] + r["S_II_ref_A2_Hz"], rel=1e-12, abs=0)    # Eq. 41
    assert ch.eta(f) >= C1.eta_TC * (1 - 1e-12)


@pytest.mark.parametrize("preset,f,Q,T", [("C1", 120.9e3, 1e6, 0.03), ("C1+low+RT", 1e6, 1e6, 0.03),
                                          ("C1+medium+cryo", 3e7, 2e5, 0.01)])
def test_eq_48_scan_coupling(preset, f, Q, T):
    ch = squid.chain(preset)
    n = ck.n_bose(2 * math.pi * f, T)                                                       # Eq. 47
    eta, eps = ch.eta(f), ch.eps_uc_over_hbar(f)
    s = 2 * n + 1
    k2 = ch.kappa_g2_scan(f, Q, T)
    assert k2 == pytest.approx(eps * (s + math.sqrt(s * s + 8 * eta ** 2)) / (Q * eta ** 2), rel=1e-12, abs=0)   # Eq. 48
    assert ch.xi_opt(f, T) == pytest.approx(ck.xi_opt(n, eta), rel=1e-12, abs=0)                  # Eq. 46 = S3.9
    # Eq. 46 itself: S_IIsys at the Eq. 48 mutual equals (hbar w/R) xi_opt
    L_tot = 1e-6
    M = ch.M_scan_H(f, Q, L_tot, T)
    R = 2 * math.pi * f * L_tot / Q
    assert ch.input_referred(f, M)["S_II_sys_A2_Hz"] == pytest.approx(HBAR * 2 * math.pi * f / R * ch.xi_opt(f, T), rel=1e-12, abs=0)


def test_chain_reproduces_the_independent_implementation():
    """Stored outputs of an independent implementation of the [DMR-SQUID] chain (C1 alone, C1 at eta_TC = 10, and
    the low-bandwidth chain with its room-temperature amplifier)."""
    c1 = squid.chain("C1")
    assert c1.eta(1e6) == pytest.approx(3.702986, rel=1e-6, abs=0)
    assert c1.eps_uc_over_hbar(1e6) == pytest.approx(5.236814, rel=1e-6, abs=0)
    assert c1.kappa_g2_scan(120.9e3, 1e6, 0.03) == pytest.approx(7.89851e-3, rel=1e-5, abs=0)
    assert c1.kappa_g2_scan(1e6, 1e6, 0.03) == pytest.approx(9.54947e-4, rel=1e-5, abs=0)
    b = squid.chain("C1", first=C1.at_eta(10.0))
    assert b.kappa_g2_scan(120.9e3, 1e6, 0.03) == pytest.approx(2.92481e-3, rel=1e-5, abs=0)
    assert b.kappa_g2_scan(1e6, 1e6, 0.03) == pytest.approx(3.53654e-4, rel=1e-5, abs=0)
    low = squid.chain("C1+low+RT")
    assert low.eta(1e6) == pytest.approx(9.707491, rel=1e-6, abs=0)
    assert low.eps_uc_over_hbar(1e6) == pytest.approx(10.828227, rel=1e-6, abs=0)


# ============================================================== shape of Figs. 3-4 (no values)
def test_figure_3_4_shapes():
    f_lo = np.geomspace(1e6, 4.9e7, 25)
    f_hi = np.linspace(5e7, 3e8, 26)
    f_all = np.concatenate([f_lo, f_hi])
    rt = {bw: squid.chain(f"C1+{bw}+RT") for bw in squid.SECOND_STAGES}
    cr = {bw: squid.chain(f"C1+{bw}+cryo") for bw in squid.SECOND_STAGES}
    for ch in list(rt.values()):
        for g in (ch.eta, ch.eps_uc_over_hbar):
            assert np.all(np.diff(g(f_lo)) > 0) and np.all(np.diff(g(f_hi)) > 0)   # rises with f within a preamp
            assert np.all(g(f_all) > 0)
    # low BW is best at low frequency and worst at the top of the band (Sec. V D: different
    # second stages are optimal in different ranges)
    assert rt["low"].eta(1e6) < rt["medium"].eta(1e6) < rt["high"].eta(1e6)
    assert rt["low"].eta(2.9e8) > max(rt["medium"].eta(2.9e8), rt["high"].eta(2.9e8))
    # the cryogenic amplifier beats the RT preamps for the same array (Figs. 3-4 dashed vs solid)
    f_c = np.linspace(1e7, 3e8, 30)
    for bw in squid.SECOND_STAGES:
        assert np.all(cr[bw].eta(f_c) < rt[bw].eta(f_c))
        assert np.all(cr[bw].eps_uc_over_hbar(f_c) < rt[bw].eps_uc_over_hbar(f_c))
    # every chain is above its first stage
    assert np.all(rt["low"].eta(f_all) > C1.eta_TC)
    assert np.all(rt["low"].eps_uc_over_hbar(f_all) > C1.eps_uc_J_s / HBAR)


def test_band_guards_and_arguments():
    with pytest.raises(ValueError, match="300"):
        squid.chain("C1+low+RT").eta(3.5e8)
    with pytest.raises(ValueError, match="CITLF2"):
        squid.chain("C1+low+cryo").eta(1e6)                    # specified for 5-500 MHz only
    with pytest.raises(ValueError, match="unknown preset"):
        squid.chain("C1+tiny+RT")
    with pytest.raises(ValueError, match="sign"):
        squid.Chain(C1, sign=0.5)
    with pytest.raises(ValueError, match="both"):
        squid.Chain(C1, second=squid.SecondStage(16, 1))
    assert squid.chain("C1").eta([1e5, 1e8]) == pytest.approx([C1.eta_TC] * 2, rel=1e-12, abs=0)
    # Eq. 48 is unconstrained: kappa_g^2 > 1 is reported but no SQUID is built for it
    assert squid.chain("C1").kappa_g2_scan(1e5, 1e3, 0.03) > 1
    with pytest.raises(ValueError, match="not attainable"):
        squid.chain("C1").squid_for_resonator(1e5, 1e3, 1e-6, 0.03)
    # the cryogenic amplifier's noise temperature assumes a matched (~50 ohm) source
    with pytest.raises(ValueError, match="matched"):
        squid.Chain(C1, squid.SecondStage(160, 1), squid.CITLF2).eta(1e7)
    assert set(squid.presets()) == {"C1"} | {f"C1+{b}+{a}" for b in ("low", "medium", "high") for a in ("RT", "cryo")}
    # the RT preset switches from XXF-1 to the 300 MHz preamp at 50 MHz (larger current noise)
    rt = squid.chain("C1+low+RT")
    assert rt.eta(50e6) > rt.eta(49.999e6)


# ============================================================== correlated Squid in the closures
def _series_loop(ch, f0=1e6, L=1.4e-6, L_in=2e-9, Q=1e6, T=0.03):
    """Series RLC + input coil; the SQUID at the Eq. 48 coupling of this loop."""
    w0 = 2 * math.pi * f0
    Lt = L + L_in
    R = w0 * Lt / Q
    sq = ch.squid_for_resonator(f0, Q, Lt, T)
    c = ck.Circuit()
    c.source("pickup", "gnd", "a", emf=1.0)
    c.inductor("L", "a", "b", L)
    c.capacitor("C", "b", "d", 1 / (w0 ** 2 * Lt))
    c.resistor("R", "d", "e", R, T_K=T)
    c.inductor("L_in", "e", "gnd", L_in)
    ro = ck.Readout(circuit=c, squid=sq, pickup="pickup", L_p_H=L)
    m = MultiportCircuit()
    m.multiport("field", [("pick", "gnd", "a")], lambda w: np.array([[-1j * w * L]]), T_K=T)
    m.capacitor("C", "a", "b", 1 / (w0 ** 2 * Lt))
    m.resistor("R", "b", "d", R, T_K=T)
    m.inductor("L_in", "d", "gnd", L_in)
    mr = MultiportReadout(m, sq, {"pick": 1.0})
    return ro, mr, dict(w0=w0, R=R, Q=Q, n=ck.n_bose(w0, T), sq=sq, Lt=Lt)


def test_uncorrelated_squid_noise_unchanged():
    sq = ck.Squid(M_H=1e-9, eta_A=3.0, S_Phi=1e-30, S_J=1e-30)
    assert sq.S_PhiJ == 0.0 and sq.correlation_term(1e6, 1 + 2j) == 0.0
    ch = squid.chain("C1")
    ro, mr, p = _series_loop(ch)
    ro.squid = mr.squid = sq
    for w in (p["w0"], 1.001 * p["w0"]):
        for d in (ro.noise_psd(w), mr.noise_psd(w)):
            assert d["S_Phi_correlation_Wb2_s"] == 0.0
            assert d["S_Phi_total_Wb2_s"] == d["S_Phi_thermal_Wb2_s"] + d["S_Phi_backaction_Wb2_s"] + d["S_Phi_imprecision_Wb2_s"]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_correlated_noise_closed_form(sign):
    """Series loop: the referred cross term is exactly 2 omega S_PhiJ Im Z (docs Sec. 4), so
    N_V = (hbar w0 R/2 xi)[xi^2 (x - x0)^2 + A_corr] near the pole with x0 = s c/xi and
    eta_corr^2 = eta0^2 - c^2 ([C] F44-F57)."""
    ch = squid.chain("C1+low+RT", sign=sign)
    ro, mr, p = _series_loop(ch)
    sq, w0, R = p["sq"], p["w0"], p["R"]
    xi = sq.xi_at(w0, R)
    c = 2 * sq.S_PhiJ / HBAR
    eta0 = 2 * math.sqrt(sq.S_Phi * sq.S_J) / HBAR
    assert eta0 ** 2 - c ** 2 == pytest.approx(sq.eta_A ** 2, rel=1e-12, abs=0)
    assert np.sign(c) == sign
    A = xi ** 2 + 2 * (2 * p["n"] + 1) * xi + sq.eta_A ** 2
    for x in (-3000.0, -400.0, 0.0, 250.0, 2000.0):
        w = w0 * (1 + x / (2 * p["Q"]))
        d, dm = ro.noise_psd(w), mr.noise_psd(w)
        assert dm["S_Phi_total_Wb2_s"] == pytest.approx(d["S_Phi_total_Wb2_s"], rel=1e-10, abs=0)
        assert d["N_V_correlation_V2_s"] == pytest.approx(2 * w * sq.S_PhiJ * ro.Z_eff(w).imag, rel=1e-9, abs=1e-60)
        model = HBAR * w0 * R / (2 * xi) * (xi ** 2 * (x - c / xi) ** 2 + A)
        assert d["N_V_V2_s"] == pytest.approx(model, rel=3e-3, abs=0)       # O(x/Q) and n(omega) corrections


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_correlated_information_integral_is_gbar_of_eta_corr(sign):
    """int domega N_V^-2 over the exact circuit = (w0/2Q)(2 xi/hbar w0 R)^2 pi/(2 xi A^(3/2)) with
    A built from the CORRELATED eta: the scan sees eta(f), not the uncorrelated product (which
    would be 2.345x larger for TC), and the sign s only moves the band."""
    ch = squid.chain("C1+low+RT", sign=sign)
    ro, _, p = _series_loop(ch)
    sq, w0, R, Q = p["sq"], p["w0"], p["R"], p["Q"]
    xi = sq.xi_at(w0, R)
    x0 = 2 * sq.S_PhiJ / HBAR / xi
    hw = math.sqrt(xi ** 2 + 2 * (2 * p["n"] + 1) * xi + sq.eta_A ** 2) / xi
    u = np.concatenate([-np.geomspace(60 * hw, 1e-3 * hw, 600), [0.0], np.geomspace(1e-3 * hw, 60 * hw, 600)])
    x = x0 + u
    w = w0 * (1 + x / (2 * Q))
    N = np.array([ro.noise_psd(float(wi))["N_V_V2_s"] for wi in w])
    num = np.trapezoid(N ** -2, w) if hasattr(np, "trapezoid") else np.trapz(N ** -2, w)

    def analytic(eta):
        A = xi ** 2 + 2 * (2 * p["n"] + 1) * xi + eta ** 2
        return (w0 / (2 * Q)) * (2 * xi / (HBAR * w0 * R)) ** 2 * math.pi / (2 * xi * A ** 1.5)
    assert num / analytic(sq.eta_A) == pytest.approx(1.0, abs=2e-3)
    eta_unc = 2 * math.sqrt(sq.S_Phi * sq.S_J) / HBAR
    assert abs(num / analytic(eta_unc) - 1) > 0.2


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_readout_equivalent_with_correlation(sign):
    ch = squid.chain("C1+low+RT", sign=sign)
    ro, _, p = _series_loop(ch)
    eq = ro.equivalent(p["w0"])
    assert eq["eta_A_eff"] == pytest.approx(ch.eta(1e6), rel=1e-6, abs=0)
    assert eq["eta_A_eff_uncorrelated"] == pytest.approx(ch.eta_uncorrelated(1e6), rel=1e-6, abs=0)
    assert eq["port_coupling"] == pytest.approx(1.0, abs=1e-9)
    assert eq["band_shift_x0"] == pytest.approx(sign * 2 * abs(p["sq"].S_PhiJ) / HBAR / eq["xi"], rel=1e-5, abs=0)
    assert eq["xi"] == pytest.approx(ch.xi_opt(1e6, 0.03), rel=1e-6, abs=0)     # the Eq. 48 mutual realises xi_opt
    assert eq["N_V_reduction_resid"] < 1e-9
    assert eq["scan_inputs"]["eta_A"] == pytest.approx(ch.eta(1e6), rel=1e-6, abs=0)
    assert eq["correlated_reduction_resid"] < 1e-3


@pytest.mark.parametrize("M,sign", [(1e-9, 1.0), (4e-9, -1.0)])
def test_readout_equivalent_rejects_correlated_non_series_network(M, sign):
    """A real flux transformer (reactive background between pickup and coil) adds quadratic
    correlation/backaction terms; the completed-square pole must not be returned (it would give eta^2 < 0 for the
    first case and an 8.7 % error at x = 10 for the second)."""
    def build():
        return ck.series_readout(L_p_H=1e-6, f_Hz=1e6, Q=1e6, T_K=.03, L_1_H=1e-6, L_2_H=1e-6, L_in_H=1e-7,
                                 k_T=math.sqrt(1.1e-6), M_H=1e-9)
    ro = build()
    ro.squid = squid.chain("C1", sign=sign).squid(1e6, M)
    with pytest.raises(ValueError, match="does not describe this network"):
        ro.equivalent(2 * math.pi * 1e6)
    # the exact noise is still available and positive, and the uncorrelated reduction is unchanged
    assert ro.noise_psd(2 * math.pi * 1e6)["N_V_V2_s"] > 0
    ro.squid.S_PhiJ = 0.0
    assert "correlated_reduction_resid" not in ro.equivalent(2 * math.pi * 1e6)


def _with_explicit_squid_loop(circuit, M, J=1.0, L_sq=2e-10):
    """Add (in place) the SQUID loop as a network branch ``L_sq`` (p -> m current ``J`` pinned by an
    ideal current source) coupled to ``L_in`` by the circuit's own mutual stamp ``Z = -i omega M``."""
    circuit.inductor("L_sq", "sq", "gnd", L_sq)
    circuit.current_source("J", "sq", "gnd", J)      # drives J into "sq", i.e. p -> m through L_sq
    circuit.mutual("L_in", "L_sq", M=M)
    return circuit


@pytest.mark.parametrize("M", [1e-9, -3e-9])
def test_backaction_emf_sign_is_the_mutual_stamp(M):
    """docs/physics/readout_circuit.md Sec. 4.2 vs squid_chain.md Sec. 4: the backaction EMF.

    Put the SQUID loop into the network as an inductor carrying a pinned current J, coupled to the
    input coil by the circuit's own mutual inductance (the stamp that also gives Phi_SQ = M I_in).
    The flux it produces at the SQUID output, M I_in(J), must equal H_in times the EMF the code
    assumes, +i omega M J, in every closure that uses Squid.correlation_term (Readout and
    MultiportReadout), for a series loop and for a real flux transformer.  -i omega M J (the old
    wording of readout_circuit.md and multiport.md) fails by a factor -1."""
    ch = squid.chain("C1")
    ro, mr, p = _series_loop(ch)
    ft = ck.series_readout(L_p_H=1e-6, f_Hz=1e6, Q=1e6, T_K=.03, L_1_H=1e-6, L_2_H=1e-6, L_in_H=1e-7,
                           k_T=0.7, M_H=1e-9)
    J = 0.37 - 0.11j
    for r in (ro, mr, ft):
        r.squid.M_H = M
        for w in (p["w0"], 1.0003 * p["w0"], 0.4 * p["w0"]):
            H_in = r.circuit.transfer(w, r.squid.input_coil, r.flux)
            c = _with_explicit_squid_loop(copy.deepcopy(r.circuit), M, J)
            phi_ba = M * c.solve(w, {"J": J}, replace=True).I("L_in")   # every other source off
            assert phi_ba == pytest.approx(H_in * (1j * w * M * J), rel=1e-9, abs=0)
            assert abs(phi_ba + H_in * (1j * w * M * J)) > 1.9 * abs(phi_ba)
            # and the correlated cross term the code adds is 2 Re[conj(phi_ba/J) S_PhiJ]
            r.squid.S_Phi, r.squid.S_J, r.squid.S_PhiJ = 1e-30, 1e-30, 0.6e-30
            scale = 2 * abs(w * M * H_in) * 0.6e-30          # |cross term| <= scale; tolerance relative to it
            assert r.squid.correlation_term(w, H_in) == pytest.approx(
                2 * (np.conj(phi_ba / J) * 0.6e-30).real, rel=0, abs=1e-9 * scale)


def test_correlation_must_be_a_covariance():
    sq = ck.Squid(M_H=1e-9, S_Phi=1e-30, S_J=1e-30, S_PhiJ=2e-30)
    with pytest.raises(ValueError, match="S_PhiJ"):
        sq.correlation_term(1e6, 1.0)


def test_chain_to_circuit_noise_is_foster_normalised():
    ch = squid.chain("C1+medium+RT", sign=-1.0)
    sq = ch.squid(2e6, 4e-9, input_coil="coil")
    p = ch.one_sided(2e6)
    assert sq.S_Phi == pytest.approx(float(p["S1_Phi_Wb2_Hz"]) / 2, rel=1e-15, abs=0)
    assert sq.S_J == pytest.approx(float(p["S1_J_A2_Hz"]) / 2, rel=1e-15, abs=0)
    assert sq.S_PhiJ == pytest.approx(-12 * KB * C1.T_j_K * C1.L_sq_H / C1.R_j_ohm / 2, rel=1e-12, abs=0)
    assert 2 * math.sqrt(sq.S_Phi * sq.S_J - sq.S_PhiJ ** 2) / HBAR == pytest.approx(sq.eta_A, rel=1e-12, abs=0)
    assert sq.eta_A == pytest.approx(ch.eta(2e6), rel=1e-12, abs=0) and sq.input_coil == "coil"


# ============================================================== scan wiring
def test_reach_accepts_a_chain():
    PSI, LP = 2.0e-4, 1.0e-6
    kw = dict(live_time_yr=3.0, f_min=1.209e5, f_max=9.9138e5, n=13)
    # frequency-independent chain (C1 alone): identical to the float-eta C++ path
    c1 = squid.chain("C1")
    a = scan.reach(PSI, LP, {"Q": 1e6, "T": 0.03, "eta_A": C1.eta_TC}, **kw)
    b = scan.reach(PSI, LP, squid.benchmark(c1, Q=1e6, T_K=0.03), **kw)
    assert np.allclose(b["g95_GeV"], a["g95_GeV"], rtol=1e-10, atol=0)
    assert np.allclose(b["tau_s_per_rad_s"], a["tau_s_per_rad_s"], rtol=1e-10, atol=0)
    assert b["band"]["eta_A_source"] == c1.label
    # a real chain: eta(f) enters at every tuning
    low = squid.chain("C1+low+RT")
    r = scan.reach(PSI, LP, squid.benchmark("C1+low+RT", Q=1e6, T_K=0.03), **kw)
    assert r["eta_A"] == pytest.approx(low.eta(r["f_Hz"]), rel=1e-12, abs=0)
    assert r["band"]["eta_A"] == pytest.approx([low.eta(r["f_Hz"][0]), low.eta(r["f_Hz"][-1])], rel=1e-12, abs=0)
    d = scan.reach_dispersive([kw["f_min"], kw["f_max"]], PSI, LP, Q=1e6, T_K=0.03, eta_A=lambda f: low.eta(f),
                              **{k: v for k, v in kw.items() if k not in ("f_min", "f_max")})
    assert np.allclose(r["g95_GeV"], d["g95_GeV"], rtol=1e-12, atol=0)
    # scan.psd evaluates a callable eta at the tuning frequency
    pa = scan.psd(PSI, LP, 5e5, squid.benchmark(low, Q=1e6, T_K=0.03), n=11)
    pb = scan.psd(PSI, LP, 5e5, {"Q": 1e6, "T": 0.03, "eta_A": low.eta(5e5)}, n=11)
    assert np.allclose(np.asarray(pa["N_V_V2_s"]), np.asarray(pb["N_V_V2_s"]), rtol=1e-14, atol=0)


def test_reach_with_a_dispersive_chain():
    """Over 1-40 MHz eta of C1+low+RT rises 9.7 -> 12.3.  Log-uniform dwell: g95 ~ G-bar^(-1/4) and
    G-bar_opt ~ 1/eta (n >> eta, S3.11), so g95(chain)/g95(C1) = (eta(f)/eta_TC)^(1/4) pointwise.
    Target dwell: each search reaches g_target (t_req/t_live)^(1/4), so the ratio is flat."""
    PSI, LP = 2.0e-4, 1.0e-6
    low, c1 = squid.chain("C1+low+RT"), squid.chain("C1")
    kw = dict(live_time_yr=3.0, f_min=1e6, f_max=4e7, n=9)
    a = scan.reach(PSI, LP, squid.benchmark(low, 1e6, 0.5), allocation="log_uniform", **kw)
    b = scan.reach(PSI, LP, squid.benchmark(c1, 1e6, 0.5), allocation="log_uniform", **kw)
    eta = low.eta(a["f_Hz"])
    assert eta[-1] / eta[0] > 1.2
    n = np.array([ck.n_bose(2 * math.pi * f, 0.5) for f in a["f_Hz"]])
    gb = lambda e: np.array([ck.G_bar(ck.xi_opt(ni, ei), ni, ei) for ni, ei in zip(n, e)])
    expect = (gb(np.full(n.shape, C1.eta_TC)) / gb(eta)) ** 0.25          # exact G-bar_opt ratio
    assert np.allclose(a["g95_GeV"] / b["g95_GeV"], expect, rtol=1e-6, atol=0)
    assert a["band"]["t_req_s"] == pytest.approx(3.0 * scan.YEAR, rel=1e-15, abs=0)
    at = scan.reach(PSI, LP, squid.benchmark(low, 1e6, 0.5), allocation="target", **kw)
    bt = scan.reach(PSI, LP, squid.benchmark(c1, 1e6, 0.5), allocation="target", **kw)
    rt = at["g95_GeV"] / bt["g95_GeV"]
    assert np.allclose(rt, rt[0], rtol=1e-6, atol=0)
    assert rt[0] == pytest.approx((at["band"]["t_req_s"] / bt["band"]["t_req_s"]) ** 0.25, rel=1e-6, abs=0)
