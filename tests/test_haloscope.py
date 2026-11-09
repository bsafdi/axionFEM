"""Cavity-haloscope sensitivity layer (axionfem.haloscope; docs/physics/haloscope_sensitivity.md).

Anchors: independent (blind) derivations of the signal power, scan rate and squeezing, hand calculations
(including an independent closed-form reconstruction of every preset limit, doc Sec. 8.1), the
closed forms of Kim et al. 2020, Malnou et al. 2019, Al Kenany et al. 2017 Eq. 5 and HAYSTAC
2301.09721 Eq. A4, published signal-power normalisations, and published exclusion limits.
Factor-2 errors in rates, efficiencies or Q_a must fail. The haloscope module is beta.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate, optimize

from axionfem import haloscope as hs
from axionfem import scan
from axionfem.haloscope_presets import PRESETS, REPRODUCIBLE

H, KB, HBAR = hs.H_PLANCK, hs.KB, hs.HBAR


def approx(expected, **kw):
    """pytest.approx with NO absolute floor: its default abs = 1e-12 makes every comparison of
    powers (~1e-22 W) or couplings (~1e-14 GeV^-1) pass whatever the value."""
    kw.setdefault("abs", 0.0)
    return pytest.approx(expected, **kw)
GHz = 1e9


def ref_cavity(**kw):
    """The independent derivation's reference cavity: B = 8 T, V = 100 L, C = 0.5, Q_L = 5e4 at beta = 2."""
    d = dict(B_T=8.0, V_m3=0.1, C=0.5, Q0=1.5e5, beta=2.0)
    d.update(kw)
    return hs.Cavity(**d)


# ============================================================ signal power (H3)
def test_power_independent_reference():
    """Independent (blind) derivation: 1.61723e-22 W x beta/(1+beta); 8.08613e-23 W at beta = 1."""
    P0 = hs.power_coefficient(1e-15, 1 * GHz, 8.0, 0.1, 0.5, rho_GeV_cm3=0.45) * 5e4
    assert P0 == approx(1.61723e-22, rel=4e-6)          # the reference has 6 digits
    P1 = hs.signal_power(1e-15, 1 * GHz, 8.0, 0.1, 0.5, Q_L=5e4, beta=1.0)
    assert P1 == approx(8.08613e-23, rel=4e-6)
    # Q0 + beta and Q_L + beta give the same answer
    assert hs.signal_power(1e-15, GHz, 8.0, 0.1, 0.5, Q0=1e5, beta=1.0) == approx(P1, rel=1e-14)


def test_power_natural_units_by_hand():
    """P = eta g^2 (rho/m) B^2 V C Q evaluated in natural HL units (eV) and converted to W."""
    hbarc = 1.973269804e-7                       # eV m
    T_eV2 = 195.352771                           # 1 T in eV^2 (scan_and_signal.md S1.1)
    g, f, B, V, C, Q = 3e-15, 4.5 * GHz, 9.0, 1.545e-3, 0.5, 1e4
    m = H * f / hs.EV_J                          # eV
    rho = 0.45 * 7.683505562e-6                  # eV^4
    P_eV2 = (g * 1e-9) ** 2 * rho / m * (B * T_eV2) ** 2 * (V / hbarc ** 3) * C * Q
    P_W = P_eV2 / (HBAR / hs.EV_J) * hs.EV_J     # eV^2 -> eV/s -> W
    assert hs.signal_power(g, f, B, V, C, Q_L=Q, beta=1e300) == approx(P_W, rel=1e-8)


@pytest.mark.parametrize("label, P_pub, kw, rtol", [
    # ADMX: P normalisations without a coupling factor (Eq. 1 of each PRL); Lambda unstated (~4-6% off)
    ("ADMX 1A 1804.05750 Eq.1", 1.9e-22, dict(f=650e6, model="KSVZ", B=6.8, V=0.136, C=0.4, QL=5e4, beta=None), 0.07),
    ("ADMX 1B 1910.08638 Eq.1", 2.2e-23, dict(f=740e6, model="DFSZ", B=7.6, V=0.136, C=0.4, QL=3e4, beta=None), 0.07),
    ("ADMX 1C 2110.06096 Eq.1", 7.7e-23, dict(f=1e9, model="DFSZ", B=7.5, V=0.136, C=0.4, QL=8e4, beta=None), 0.07),
    # CAPP-12TB Eq. 1 at beta = 2 with Lambda^2 = m_a f_a = 5.70e-3 GeV^2
    ("CAPP-12TB 2210.10961 Eq.1", 22.51e-24, dict(f=1.1e9, model="DFSZ", B=10.31, V=0.03685, C=0.6, QL=3.5e4,
                                                   beta=2.0, Lam=75.498), 0.01),
    ("QUAX 1903.06547 Sec.III", 1.85e-25, dict(f=9.067e9, model="KSVZ", B=2.0, V=0.036e-3, C=0.589, QL=2.01e5,
                                                beta=1.0, Lam=78.0), 0.02),
    ("TASEH 2205.05574 p.2", 1.4e-24, dict(f=float(hs.mass_to_freq_Hz(19.6e-6)), model="KSVZ", B=8.0, V=0.234e-3,
                                            C=0.605, QL=6.07e4 / 3, beta=2.0, Lam=78.0), 0.05),
    ("HAYSTAC 1610.02580 p.2 '~5e-24'", 5e-24, dict(f=5.8e9, model="KSVZ", B=9.0, V=1.545e-3, C=0.5, QL=1e4,
                                                     beta=2.0), 0.20),
])
def test_published_power_normalisations(label, P_pub, kw, rtol):
    g = hs.g_gamma_to_gagg(hs.freq_to_mass_eV(kw["f"]), hs.G_GAMMA[kw["model"]], kw.get("Lam", 77.6))
    beta = 1e300 if kw["beta"] is None else kw["beta"]
    P = hs.signal_power(g, kw["f"], kw["B"], kw["V"], kw["C"], Q_L=kw["QL"], beta=beta)
    assert P == approx(P_pub, rel=rtol), label


def test_kim_general_power_and_detuning():
    """Kim 2020 Eq. 7 (= Eq. 18): Q_eff = Q_L Q_a/(Q_L + Q_a), Q_a = f/FWHM; Lorentzian detuning."""
    a = dict(g_GeV=1e-15, f_Hz=GHz, B_T=8.0, V_m3=0.1, C=0.5, Q_L=1e6, beta=2.0)
    P_narrow = hs.signal_power(**a)
    assert hs.signal_power(**a, Q_a_fwhm=1e6) == approx(P_narrow / 2, rel=1e-14)
    assert hs.signal_power(**a, detuning_Hz=GHz / (2 * 1e6)) == approx(P_narrow / 2, rel=1e-12)


def test_signal_power_shm_limits():
    """Q_L << Q_a: the whole SHM line is inside the cavity, Q_eff -> Q_L.  Q_L >> Q_a, cavity on the
    line peak: Q_eff -> Q_L (pi f/2Q_L) s(f_c), the broad-line limit of the blind derivation."""
    f_a, a = 5 * GHz, dict(B_T=8.0, V_m3=1e-3, C=0.5, beta=2.0)
    P_lo = hs.signal_power_shm(1e-14, f_a, f_a, Q_L=1e3, **a)
    assert P_lo == approx(hs.signal_power(1e-14, f_a, Q_L=1e3, **a), rel=1e-4)
    QL = 1e9
    f_c = f_a * (1 + 0.5 * (300e3 / hs.C_LIGHT) ** 2)
    P_hi = hs.signal_power_shm(1e-14, f_c, f_a, Q_L=QL, **a)
    s = float(hs.shm_line_shape(f_c, f_a))
    Q_eff = math.pi * f_c / 2 * s
    assert P_hi == approx(hs.signal_power(1e-14, f_a, Q_L=QL, **a) * Q_eff / QL, rel=2e-2)


# ============================================================ couplings and line shape (H1, H2)
def test_couplings():
    m = 1e-5
    gK = hs.g_axion(m, "KSVZ")
    assert gK == approx(0.97 * hs.ALPHA_EM * 1e-14 / (math.pi * 0.0776 ** 2), rel=1e-12)
    assert hs.g_axion(m, "DFSZ") / gK == approx(0.36 / 0.97, rel=1e-12)
    # package QCD convention (C_gamma = 1.77, m_a f_a = 5.70e-3): KSVZ a few % below
    assert 0.90 < hs.g_axion(m, "KSVZ", "qcd") / gK < 1.0
    assert hs.mass_to_freq_Hz(hs.freq_to_mass_eV(5.7e9)) == approx(5.7e9, rel=1e-14)
    assert float(hs.mass_to_freq_Hz(1e-6)) == approx(scan.mass_to_freq(1e-6), rel=1e-9)
    with pytest.raises(ValueError, match="convention"):
        hs.g_axion(m, "KSVZ", "nope")


def test_shm_line_matches_core():
    """Python SHM = C++ core: Q_a (S2.6) and normalisation; Lorentzian Q_a convention (H2.3)."""
    assert hs.Q_a_shm() == approx(scan.Q_axion_shm(), rel=1e-7)
    assert hs.Q_a_shm(220.0, 220.0) == approx(scan.Q_axion_shm(220.0, 220.0), rel=1e-7)
    fa = 1 * GHz
    f = fa * (1 + np.linspace(0, 2e-5, 200001))
    s = hs.shm_line_shape(f, fa)
    assert np.trapezoid(s, f) == approx(1.0, rel=1e-6)
    assert fa * np.trapezoid(s ** 2, f) == approx(hs.Q_a_shm(), rel=1e-4)
    # Lorentzian of f/FWHM = Q: 1/int s^2 = pi FWHM
    Q = 1e6
    fw = fa / Q
    x = np.linspace(-2e4, 2e4, 2000001) * fw
    s_l = (fw / (2 * math.pi)) / (x ** 2 + fw ** 2 / 4)
    assert fa * np.trapezoid(s_l ** 2, x) == approx(hs.Q_a_lorentzian_fwhm(Q), rel=1e-3)


# ============================================================ noise (H4)
def test_noise_quanta():
    f = 5 * GHz
    x = H * f / (KB * 0.1)
    assert float(hs.n_thermal(f, 0.1)) == approx(1 / math.expm1(x), rel=1e-12)
    assert float(hs.T_sys_quantum(f, 0.0, 0.5)) == approx(H * f / KB, rel=1e-12)  # SQL N_sys = 1
    assert float(hs.sql(0.0).N_sys(f)) == approx(1.0, rel=1e-14)
    assert float(hs.hemt(0.0, 2.0).T_sys(f)) == approx(2.0 + H * f / (2 * KB), rel=1e-12)
    # a bare phase-sensitive amplifier is no better than the SQL (half the signal in one quadrature)
    assert float(hs.squeezed(0.0, 0.0).N_sys(f)) == approx(1.0, rel=1e-14)
    rx = hs.Receiver(T_sys_K=0.5)
    assert float(rx.T_sys(f, 1e5)) == approx(0.5, rel=1e-14)
    with pytest.raises(ValueError, match="phase-sensitive"):
        hs.Receiver(T_cav_K=0.05, N_A=0.0, squeeze_dB=3.0)
    with pytest.raises(ValueError, match="T_sys_K"):
        hs.Receiver(T_cav_K=0.05)


def test_no_isolator_matches_kim():
    """T_in = 0 (no isolator): on resonance N = (n_c + 1/2) 4 beta/(1+beta)^2 + |chi|^2/2 + N_A
    (Kim 2020 Sec. 3 thermal term 4 beta/(1+beta)^2); with an isolator N is flat and beta-free."""
    f, T, NA = 1 * GHz, 0.5, 0.5
    nc = float(hs.n_thermal(f, T))
    for beta in (0.5, 1.0, 2.0, 5.0):
        rx = hs.Receiver(T_cav_K=T, T_in_K=0.0, N_A=NA)
        expect = (nc + 0.5) * 4 * beta / (1 + beta) ** 2 + 0.5 * ((beta - 1) / (beta + 1)) ** 2 + NA
        assert float(rx.N_sys(f, 0.0, Q0=1e5, beta=beta)) == approx(expect, rel=1e-12)
        iso = hs.Receiver(T_cav_K=T, N_A=NA)
        assert float(iso.N_sys(f, 3e4, Q0=1e5, beta=beta)) == approx(nc + 0.5 + NA, rel=1e-12)
    # far off resonance the cavity noise disappears without isolator
    assert float(rx.N_sys(f, 1e9, Q0=1e5, beta=2.0)) == approx(0.5 + NA, rel=1e-6)


def test_haystac_2017_hot_cavity_noise():
    """1610.02580 p.3: k T_S ~ 3 h nu on resonance, 2.2 h nu at 650 kHz detuning (preset T_cav INFERRED)."""
    cav, rx, _ = hs.preset_models("HAYSTAC_2017")
    f = 5.75e9
    assert float(rx.N_sys(f, 0.0, cav.Q0, cav.beta)) == approx(3.0, abs=0.1)
    assert float(rx.N_sys(f, 650e3, cav.Q0, cav.beta)) == approx(2.2, abs=0.1)


# ============================================================ scan rate, beta_opt (H5)
def test_scan_rate_independent_reference():
    """Independent (blind) derivation: 38.3153 Hz/s (Q_a,rad = 1e6) and 12.1961 Hz/s (Q_a,F = 1e6)."""
    rx = hs.Receiver(T_sys_K=1.0)
    r = hs.scan_rate(GHz, ref_cavity(), rx, g_GeV=1e-15, snr=5.0, Q_a=1e6)
    assert r == approx(38.3153, rel=2e-6)
    r_F = hs.scan_rate(GHz, ref_cavity(), rx, g_GeV=1e-15, snr=5.0, Q_a=hs.Q_a_lorentzian_fwhm(1e6))
    assert r_F == approx(12.1961, rel=5e-6)   # the reference has 6 digits
    # pi/27 form at fixed Q0
    P = hs.power_coefficient(1e-15, GHz, 8.0, 0.1, 0.5)
    assert r == approx(math.pi / 27 * P ** 2 * 1e6 * 1.5e5 / (5 * KB * 1.0) ** 2, rel=1e-12)


def test_scan_rate_scaling():
    """df/dt ~ g^4 B^4 V^2 C^2 Q_L / T^2 at fixed beta; ~ 1/SNR^2; ~ Q_a."""
    rx = hs.Receiver(T_sys_K=1.0)
    kw = dict(g_GeV=1e-15, Q_a=1e6)
    r0 = hs.scan_rate(GHz, ref_cavity(), rx, **kw)
    assert hs.scan_rate(GHz, ref_cavity(B_T=16.0), rx, **kw) / r0 == approx(16.0, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(V_m3=0.2), rx, **kw) / r0 == approx(4.0, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(C=1.0), rx, **kw) / r0 == approx(4.0, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(Q0=3e5), rx, **kw) / r0 == approx(2.0, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(), hs.Receiver(T_sys_K=2.0), **kw) / r0 == approx(0.25, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(), rx, g_GeV=2e-15, Q_a=1e6) / r0 == approx(16.0, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(), rx, **kw, snr=10.0) / r0 == approx(0.25, rel=1e-12)
    assert hs.scan_rate(GHz, ref_cavity(), rx, g_GeV=1e-15, Q_a=2e6) / r0 == approx(2.0, rel=1e-12)
    # beta^2/(1+beta)^3 at fixed Q0
    rb = hs.scan_rate(GHz, ref_cavity(beta=1.0), rx, **kw)
    assert rb / r0 == approx((1 / 8) / (4 / 27), rel=1e-12)
    # HAYSTAC 4/5 option
    rh = hs.scan_rate(GHz, ref_cavity(), rx, **kw, prefactor="haystac")
    assert rh / r0 == approx(0.8 / (math.pi / 4), rel=1e-12)


def test_detuning_integral_flat_equals_quadrature():
    """The Gauss rule in theta reproduces pi f/(4 Q_L N^2) when the noise model is not flagged flat."""
    f, Q0, beta = 5 * GHz, 5e4, 2.0
    rx = hs.Receiver(T_cav_K=0.1, T_in_K=0.1 + 1e-12, N_A=0.5)    # flat in value, not in flag
    assert not rx.flat
    N = float(hs.n_thermal(f, 0.1)) + 1.0
    assert hs._detuning_integral(f, Q0, beta, rx) == approx(math.pi * f / (4 * Q0 / 3) / N ** 2, rel=1e-9)


def test_beta_opt():
    cav = ref_cavity()
    assert hs.beta_opt(GHz, cav, hs.Receiver(T_sys_K=1.0), g_GeV=1e-15) == approx(2.0, rel=1e-5)
    assert hs.beta_opt(5 * GHz, cav, hs.sql(0.05), g_GeV=1e-15) == approx(2.0, rel=1e-5)
    # Kim Eq. 20: closed form = numerical maximum of Kim Eq. 19; -> 2 for Q0 << Q_a
    for Q0, Qa in ((1e4, 1e6), (1e6, 1e6), (1e7, 1e6)):
        c = hs.Cavity(B_T=8.0, V_m3=0.1, C=0.5, Q0=Q0)
        res = optimize.minimize_scalar(
            lambda lb: -hs.scan_rate_kim2020(GHz, replace(c, beta=math.exp(lb)), T_sys_K=1.0, g_GeV=1e-15,
                                             Q_a_kim=Qa),
            bounds=(-3, 6), method="bounded", options={"xatol": 1e-9})
        assert math.exp(res.x) == approx(hs.beta_opt_kim2020(Q0, Qa), rel=1e-4)
    assert hs.beta_opt_kim2020(1.0, 1e6) == approx(2.0, rel=1e-5)


def test_kim_vs_optimal():
    """Q_L << Q_a: Kim Eq. 19 = optimal rate x Q_a,Kim/((pi/4) Q_a,rad); Q_L >> Q_a: Kim saturates
    (Q_l Q_a^2/(Q_l + Q_a) -> Q_a^2) while the optimally weighted rate keeps growing ~ Q_L (H5.6)."""
    rx = hs.Receiver(T_sys_K=1.0)
    c = ref_cavity()
    r_opt = hs.scan_rate(GHz, c, rx, g_GeV=1e-15, Q_a=1e6)
    r_kim = hs.scan_rate_kim2020(GHz, c, T_sys_K=1.0, g_GeV=1e-15, Q_a_kim=1e6)
    QL = 5e4
    assert r_kim / r_opt == approx(1e6 / (QL + 1e6) / (math.pi / 4), rel=1e-12)
    big = ref_cavity(Q0=3e8)
    assert hs.scan_rate(GHz, big, rx, g_GeV=1e-15, Q_a=1e6) / r_opt == approx(2000.0, rel=1e-12)
    assert hs.scan_rate_kim2020(GHz, big, T_sys_K=1.0, g_GeV=1e-15, Q_a_kim=1e6) / r_kim < 25.0
    # Kim's no-isolator form: noise k(T_eff 4 beta/(1+beta)^2 + T_add); at beta = 1 equals T_sys = T_eff + T_add
    c1 = replace(c, beta=1.0)
    assert hs.scan_rate_kim2020(GHz, c1, T_eff_K=0.3, T_add_K=0.7, g_GeV=1e-15) == approx(
        hs.scan_rate_kim2020(GHz, c1, T_sys_K=1.0, g_GeV=1e-15), rel=1e-14)
    with pytest.raises(ValueError, match="T_eff_K together with T_add_K"):
        hs.scan_rate_kim2020(GHz, c1, T_sys_K=1.0, T_eff_K=0.3, g_GeV=1e-15)


def test_scan_equals_sum_of_dwells_shm():
    """Separability (H5.4): summing matched-filter dwell SNR^2 over dense tunings equals the scan
    formula with Delta f_eff = f/Q_a,SHM, also when Q_L ~ Q_a (here Q_L = 1e6)."""
    f_a = 5 * GHz
    cav = hs.Cavity(B_T=8.0, V_m3=1e-3, C=0.5, Q0=3e6, beta=2.0)   # Q_L = 1e6
    rx = hs.Receiver(T_sys_K=0.3)
    t, g = 100.0, 1e-14
    QL = 1e6
    step = f_a / QL / 40
    centre = f_a * (1 + 0.5 * (330e3 / hs.C_LIGHT) ** 2)
    f_c = centre + step * np.arange(-1200, 1201)
    snr2 = 0.0
    for fc in f_c:
        gl = hs.g_limit_dwell(f_a, float(fc), cav, rx, t_s=t, snr=1.0)
        snr2 += (g / gl) ** 4
    r1 = hs.scan_rate(f_a, cav, rx, g_GeV=g, snr=1.0)
    assert snr2 == approx(r1 / (step / t), rel=5e-3)


def test_dwell_dicke_by_hand_quax2019():
    """1903.06547 Sec. III: g = g_KSVZ sqrt(1.64 sigma_P/P_KSVZ), sigma_P = k T sqrt(b/t) (Dicke)."""
    cav, rx, a = hs.preset_models("QUAX_2019")
    f = 9.067e9
    g = hs.g_limit_dwell(f, f, cav, rx, t_s=a["t_s"], snr=1.64, Q_a=a["Q_a"])
    gK = hs.g_gamma_to_gagg(hs.freq_to_mass_eV(f), -0.97)
    PK = hs.signal_power(gK, f, 2.0, 36.43e-6, 0.589, Q0=4.02e5, beta=1.0)
    sigma = KB * 15.3 * math.sqrt(7812.5 / 1200.0)
    assert g == approx(gK * math.sqrt(1.64 * sigma / PK), rel=1e-10)
    # narrow line (Q_L = 1e4 << Q_a): SHM matched filter = Dicke with Delta f_eff = f/Q_a,SHM.  At the
    # QUAX Q_L = 2e5 (0.3 Q_a) the line is NOT narrow: the cavity response over the line costs 2.6% in g.
    narrow = replace(cav, Q0=2e4)
    g_shm = hs.g_limit_dwell(f, f, narrow, rx, t_s=1200.0, snr=1.64)
    g_dk = hs.g_limit_dwell(f, f, narrow, rx, t_s=1200.0, snr=1.64, Q_a=hs.Q_a_shm())
    assert g_shm == approx(g_dk, rel=2e-4)
    assert hs.g_limit_dwell(f, f, cav, rx, t_s=1200.0, snr=1.64) / hs.g_limit_dwell(
        f, f, cav, rx, t_s=1200.0, snr=1.64, Q_a=hs.Q_a_shm()) == approx(1.0262, rel=1e-3)
    assert hs.dicke_snr(1e-22, 1.0, 100.0, 1e4) == approx(1e-22 / KB * 0.1, rel=1e-12)


def test_snr_required():
    assert hs.snr_required(3.455, 0.95) == approx(5.1, abs=0.01)     # HAYSTAC 2017
    assert hs.snr_required(3.718, 0.90) == approx(5.0, abs=0.001)    # CAPP
    assert hs.snr_required(snr_target=8.33) == 8.33
    with pytest.raises(ValueError):
        hs.snr_required()


# ============================================================ squeezing (H4.3)
@pytest.mark.parametrize("dB", [1.0, 3.0, 6.0, 10.0, 20.0])
def test_squeezing_matches_malnou_eq2(dB):
    """Ideal squeezing (eff = 1, T = 0, N_A = 0): enhancement and optimal coupling equal
    Malnou et al. 2019 Eq. 2, R ~ sqrt(G) kappa_m^2/[kappa_l kappa_m + (kappa_l - kappa_m)^2/(4G)]^(3/2)."""
    G = 10 ** (dB / 10)
    cav = hs.Cavity(B_T=8.0, V_m3=1.5e-3, C=0.5, Q0=5e4, beta=2.0)
    rx = hs.squeezed(0.0, dB)
    b = hs.beta_opt(5 * GHz, cav, rx, g_GeV=1e-14)
    E = hs.scan_rate(5 * GHz, replace(cav, beta=b), rx, g_GeV=1e-14) / hs.scan_rate(5 * GHz, cav, hs.sql(0.0), g_GeV=1e-14)

    def R(be, g):
        return math.sqrt(g) * be ** 2 / (be + (be - 1) ** 2 / (4 * g)) ** 1.5

    b_mal = optimize.minimize_scalar(lambda x: -R(x, G), bounds=(0.1, 1e3), method="bounded",
                                     options={"xatol": 1e-10}).x
    assert b == approx(b_mal, rel=1e-4)
    assert E == approx(R(b_mal, G) / R(2.0, 1.0), rel=1e-6)


def test_squeezing_E_same_and_direct():
    """Independent (blind) squeezing derivation: 9 dB at the squeezer,
    eta_1 = eta_2 = 0.8, T = 0, N_A = 0: beta_opt = 4.03520, E_same = 1.82395 (vs the unsqueezed
    receiver with the same eta_2 at beta = 2) and E_direct = eta_2^2 E_same = 1.16733 (vs a lossless SQL)."""
    f = 5 * GHz
    cav = hs.Cavity(B_T=8.0, V_m3=1.5e-3, C=0.5, Q0=5e4, beta=2.0)
    rx = hs.squeezed(0.0, 9.0, squeeze_eff=0.8, eta_out=0.8)
    assert rx.output_S == approx(0.440571, rel=2e-6)
    b = hs.beta_opt(f, cav, rx, g_GeV=1e-14)
    assert b == approx(4.03520, rel=1e-5)
    r_sq = hs.scan_rate(f, replace(cav, beta=b), rx, g_GeV=1e-14)
    r_same = hs.scan_rate(f, cav, replace(rx, squeeze_dB=0.0), g_GeV=1e-14)
    r_sql = hs.scan_rate(f, cav, hs.sql(0.0), g_GeV=1e-14)
    assert r_sq / r_same == approx(1.82395, rel=1e-5)
    assert r_sq / r_sql == approx(1.16733, rel=1e-5)
    # the unsqueezed same-eta_2 receiver is exactly eta_2^2 = 0.64 of the lossless SQL (signal x eta_2, noise flat)
    assert r_same / r_sql == approx(0.64, rel=1e-12)


def test_squeezer_gain_from_measured_output():
    """Convention (H4.3): squeeze_dB is the squeezer gain; an output-measured S is converted with
    squeezer_gain_dB, which removes N_A.  4.0 dB at eta = 0.63 with negligible N_A -> G_s = 22.4142
    (13.5052 dB, independent derivation); with HAYSTAC's N_a = 0.08 the measured 4.0 dB is below the loss floor
    (converting it anyway would count the loss twice), so it must raise."""
    S = 10 ** -0.4
    assert hs.squeezer_gain_dB(S, 0.63) == approx(13.5052, abs=1e-4)
    with pytest.raises(ValueError, match="loss floor"):
        hs.squeezer_gain_dB(S, 0.63, N_A=0.08)
    # round trip through the Receiver, with N_A and a split eta
    dB = hs.squeezer_gain_dB(0.55, 0.63, N_A=0.08)
    rx = hs.squeezed(0.0, dB, squeeze_eff=0.63 / 0.794, eta_out=0.794, N_A=0.08)
    assert rx.measured_S(4 * GHz) == approx(0.55, rel=1e-12)
    assert rx.output_S == approx(1 - 0.63 + 0.63 / 10 ** (dB / 10), rel=1e-12)


@pytest.mark.parametrize("name, S_dB, eta, E_pub", [("IIa", 4.0, 0.63, 2.0), ("IIb", 3.7, 0.60, 1.9)])
def test_squeezing_haystac_expected_enhancement(name, S_dB, eta, E_pub):
    """2301.09721 Sec. II C: 'scan rate expected to increase by 1.9 (IIb) and 2.0 (IIa) relative to ... the
    same configuration without squeezing', for the delivered 3.7/4.0 dB at eta = 0.60/0.63.  That is the
    ideal (N_A -> 0) number: reproduced within 3% at T = 0 and N_A = 0, eta split as Table II
    (alpha = sqrt(eta)), and equal to the blind E_same(beta_opt, s) closed form to 1e-5."""
    f = 4.3 * GHz
    a = math.sqrt(eta)
    s = 10 ** (-S_dB / 10)
    rx = hs.squeezed(0.0, hs.squeezer_gain_dB(s, eta), squeeze_eff=eta / a, eta_out=a)
    cav = hs.Cavity(B_T=8.0, V_m3=1.545e-3, C=0.43, Q0=4.5e4, beta=2.0)
    b = hs.beta_opt(f, cav, rx)
    E = hs.scan_rate(f, replace(cav, beta=b), rx) / hs.scan_rate(f, cav, replace(rx, squeeze_dB=0.0))
    assert E == approx(E_pub, rel=0.03), (name, E)
    bo = (2 / s - 1 + math.sqrt((2 / s - 1) ** 2 + 8)) / 2
    E_same = 27 / 4 * bo ** 2 / (math.sqrt(s) * (s * (bo - 1) ** 2 + 4 * bo) ** 1.5)
    assert b == approx(bo, rel=1e-5)
    assert E == approx(E_same, rel=1e-5)


def test_squeezing_haystac_preset_enhancement():
    """With the full Table II receiver (N_a = 0.08, S_C, alpha, G_S) the enhancement at the respective
    optimal beta is 1.68 (IIa) / 1.65 (IIb), below the ideal 2.0/1.9 because N_a is not negligible
    against the squeezed vacuum; the run beta (6.9/6.5) vs the same beta
    unsqueezed gives 2.09/2.02.  Regression snapshot of the diagnosis, not a validation."""
    for name, E_opt, E_run in (("HAYSTAC_2021", 1.677, 2.090), ("HAYSTAC_2023", 1.652, 2.023)):
        cav, rx, _ = hs.preset_models(name)
        f = PRESETS[name]["published"]["f_Hz"]
        un = replace(rx, squeeze_dB=0.0)
        b, bu = hs.beta_opt(f, cav, rx), hs.beta_opt(f, cav, un)
        E = hs.scan_rate(f, replace(cav, beta=b), rx) / hs.scan_rate(f, replace(cav, beta=bu), un)
        assert E == approx(E_opt, rel=2e-3), (name, E)
        assert hs.scan_rate(f, cav, rx) / hs.scan_rate(f, cav, un) == approx(E_run, rel=2e-3)
        assert 4.0 < b < 5.0 and 2.3 < bu < 2.7          # hot cavity pushes the unsqueezed optimum above 2


def test_haystac_eq_A4_noise_model():
    """Receiver.N_sys = 2 x the IF noise of 2301.09721 Eq. A4 (S(T) = (1/2)(1/2 + n), Eq. A3), with
    alpha = eta_out, rho = squeeze_eff, G_S = 1/G_s, S_c from n_cav, and lambda irrelevant (load at T_mc)."""
    cav, rx, _ = hs.preset_models("HAYSTAC_2021")
    f = 4.14e9
    al, rho, GS, Sc, Na = 0.794, 0.63 / 0.794, 0.11, 0.41, 0.08
    Smc = 0.5 * (0.5 + 1 / math.expm1(H * f / (KB * 0.061)))
    Q0, beta = cav.Q0, cav.beta
    for d in (0.0, 2e5, 7e5, 3e6):
        x = 2 * Q0 * d / f
        chi2 = ((beta - 1) ** 2 + x ** 2) / ((beta + 1) ** 2 + x ** 2)
        S_out = Na + 2 * (al * (1 - rho) * chi2 * Smc + rho * al * GS * chi2 * Smc) \
            + 2 * (al * (1 - chi2) * Sc + (1 - al) * Smc)
        assert float(rx.N_sys(f, d, Q0, beta)) == approx(2 * S_out, rel=1e-12), d


def test_eta_out_signal_and_noise():
    """eta_out attenuates the signal and mixes in loss-bath noise; with all baths at one temperature the
    noise is unchanged, so the rate scales as eta_out^2 exactly."""
    f = 5 * GHz
    base = hs.Receiver(T_cav_K=0.05, N_A=0.3)
    lossy = replace(base, eta_out=0.5)
    cav = ref_cavity()
    assert float(lossy.N_sys(f)) == approx(float(base.N_sys(f)), rel=1e-14)
    assert hs.scan_rate(f, cav, lossy, g_GeV=1e-15) / hs.scan_rate(f, cav, base, g_GeV=1e-15) == \
        approx(0.25, rel=1e-12)
    # hot cavity behind the loss: the cavity bump is attenuated by eta_out, the loss adds n_in + 1/2
    hot = hs.Receiver(T_cav_K=0.2, T_in_K=0.05, N_A=0.3, eta_out=0.7)
    nc, ni = float(hs.n_thermal(f, 0.2)), float(hs.n_thermal(f, 0.05))
    on = 0.7 * ((1 / 3) ** 2 * (ni + .5) + (1 - (1 / 3) ** 2) * (nc + .5)) + 0.3 * (ni + .5) + 0.3
    assert float(hot.N_sys(f, 0.0, 1.5e5, 2.0)) == approx(on, rel=1e-12)
    with pytest.raises(ValueError, match="eta_out"):
        hs.Receiver(T_sys_K=1.0, eta_out=0.8)


def test_efficiency_squared_placement():
    """eta_snr multiplies the SNR, Cavity.eta and eta_out the signal power: each enters the rate squared
    and g at fixed rate as eta^-1/2; in a single dwell g goes as eta_snr^-1/2 as well."""
    f, cav = GHz, ref_cavity()
    rx = hs.Receiver(T_sys_K=1.0)
    r0 = hs.scan_rate(f, cav, rx, g_GeV=1e-15)
    assert hs.scan_rate(f, cav, replace(rx, eta_snr=0.5), g_GeV=1e-15) / r0 == approx(0.25, rel=1e-12)
    assert hs.scan_rate(f, replace(cav, eta=0.5), rx, g_GeV=1e-15) / r0 == approx(0.25, rel=1e-12)
    g0 = hs.g_limit_scan(f, cav, rx, rate_Hz_s=10.0)
    assert hs.g_limit_scan(f, cav, replace(rx, eta_snr=0.5), rate_Hz_s=10.0) / g0 == approx(
        math.sqrt(2), rel=1e-12)
    gd = hs.g_limit_dwell(f, f, cav, rx, t_s=100.0)
    assert hs.g_limit_dwell(f, f, cav, replace(rx, eta_snr=0.5), t_s=100.0) / gd == approx(
        math.sqrt(2), rel=1e-12)
    assert hs.g_limit_dwell(f, f, replace(cav, eta=0.5), rx, t_s=100.0) / gd == approx(
        math.sqrt(2), rel=1e-12)


def test_default_Q_a_is_shm_radiometer():
    """Q_a=None means the SHM radiometer Q_a = 6.7795e5 (not the FWHM-style ~1e6 or half of it)."""
    cav, rx = ref_cavity(), hs.Receiver(T_sys_K=1.0)
    assert hs.scan_rate(GHz, cav, rx, g_GeV=1e-15) / hs.scan_rate(GHz, cav, rx, g_GeV=1e-15, Q_a=1e6) == \
        approx(6.7795e5 / 1e6, rel=1e-5)


# ============================================================ limits, projections, presets (H6)
def test_limit_scan_inverts_scan_rate():
    cav, rx = ref_cavity(), hs.Receiver(T_sys_K=1.0)
    g = hs.g_limit_scan(GHz, cav, rx, tau_step_s=100.0, df_step_Hz=1e3, snr=5.0)
    assert hs.scan_rate(GHz, cav, rx, g_GeV=g, snr=5.0) == approx(10.0, rel=1e-10)
    ga = hs.g_limit_scan(np.array([GHz, 2 * GHz]), cav, rx, rate_Hz_s=10.0)
    assert ga.shape == (2,) and ga[0] == approx(g, rel=1e-12)


def test_time_to_cover():
    cav, rx = ref_cavity(), hs.Receiver(T_sys_K=1.0)
    out = hs.time_to_cover(1 * GHz, 1.2 * GHz, cav, rx, g="DFSZ", n=400)
    ref = integrate.quad(lambda f: 1 / hs.scan_rate(f, cav, rx, model="DFSZ"), 1 * GHz, 1.2 * GHz,
                         epsrel=1e-10)[0]
    assert out["t_s"] == approx(ref, rel=1e-5)
    # g_factor 2 -> 16x faster; duty cycle 0.5 -> 2x slower
    out2 = hs.time_to_cover(1 * GHz, 1.2 * GHz, cav, rx, g="DFSZ", g_factor=2.0, duty_cycle=0.5, n=400)
    assert out2["t_s"] == approx(out["t_s"] * 2 / 16, rel=1e-10)


def test_callable_cavity_parameters():
    """FEM per-mode outputs enter as callables of frequency."""
    cav = hs.Cavity(B_T=8.0, V_m3=0.1, C=lambda f: 0.5 * (GHz / f), Q0=lambda f: 1.5e5, beta=2.0)
    rx = hs.Receiver(T_sys_K=1.0)
    assert hs.scan_rate(2 * GHz, cav, rx, g_GeV=1e-15, Q_a=1e6) == approx(
        hs.scan_rate(2 * GHz, ref_cavity(C=0.25), rx, g_GeV=1e-15, Q_a=1e6), rel=1e-14)
    c2 = hs.Cavity.from_QL(8.0, 0.1, 0.5, lambda f: 5e4, 2.0)
    assert c2.at(GHz)["Q0"] == approx(1.5e5)


def test_presets_documented():
    for name, p in PRESETS.items():
        assert p["refs"] and p["published"]["source"] and p["published"]["statement"], name
        keys = set(p["cavity"]) | set(p["receiver"]) | {k for k in p["analysis"] if k in ("rate_Hz_s", "t_s",
                                                                                         "snr_target")}
        src = " ".join(p["sources"])
        missing = [k for k in keys if k not in src and k not in ("Q_L", "eta", "phase_sensitive", "T_cav_K")
                   and not (k == "Q_L" and "Q0" in p["sources"])]
        assert not missing, (name, missing)
        lo, hi = p["f_range_Hz"]
        assert lo <= p["published"]["f_Hz"] <= hi, name


# ---------------------------------------------------------------- independent hand reconstruction
# Every reproducible preset recomputed from the papers' numbers RE-TYPED here (not read from PRESETS,
# so a changed preset value fails too), with CODATA constants and closed forms only:
#   P_pk/g^2 = eta_sig beta/(1+beta) Q_L K,  K = hbar c^3 rho B^2 V C/(mu0 2 pi f) / (GeV in J)^2
#   noise N(d) = N0 + D A(d), A = 4 beta/(1+beta)^2 L(d):
#       int L^2/N^2 dd = pi (f/Q_L)/(4 N0^2) [1 + 4 beta D/(N0 (1+beta)^2)]^(-3/2)   (independent squeezing derivation)
#   rate = (P_pk/hf)^2 I eta_snr^2 Q_a/(SNR^2 f)  ->  g^4 = rate SNR^2 f (hf)^2/((P_pk/g^2)^2 I eta_snr^2 Q_a)
#   Q_a = 6.7795e5 (SHM 220/232 km/s, package Q_axion).
# Phase-sensitive (HAYSTAC II, Eq. A4 in quanta with vacuum 1/2 per quadrature, x2):
#   N0 = 2 [N_a + alpha S_cav v + (1 - alpha) v],  D = 2 alpha [(n_c + 1/2) - S_cav v],
#   v = n(61 mK) + 1/2, S_cav = rho G_S + 1 - rho, n_c + 1/2 = 2 S_C, signal x alpha.
# Dwell, single bin (QUAX 2019): g^2 = SNR k T sqrt(b/t)/(P_pk/g^2).
# Dwell, SHM matched filter (QUAX 2021): SNR^2 = t int [P_pk s(f) L/(h f N)]^2 df by scipy.quad.
_h, _k, _c, _mu0, _GeV = 6.62607015e-34, 1.380649e-23, 299792458.0, 1.25663706212e-6, 1.602176634e-10
_hbar = _h / (2 * math.pi)
_RHO = 0.45 * _GeV * 1e6
_QA = 6.7795e5


def _nb(f, T):
    return 0.0 if T <= 0 else 1.0 / math.expm1(_h * f / (_k * T))


def _K(f, B, V_L, C):
    return _hbar * _c ** 3 * _RHO * B ** 2 * V_L * 1e-3 * C / (_mu0 * 2 * math.pi * f) / _GeV ** 2


def _hand_scan(f, B, V_L, C, Q_L, beta, rate, snr, eta_snr, N0, D=0.0, sig=1.0):
    I = math.pi * (f / Q_L) / (4 * N0 ** 2) * (1 + 4 * beta * D / (N0 * (1 + beta) ** 2)) ** -1.5
    A = sig * beta / (1 + beta) * Q_L * _K(f, B, V_L, C)
    return (rate * snr ** 2 * f * (_h * f) ** 2 / (A ** 2 * I * eta_snr ** 2 * _QA)) ** 0.25


def _haystac2(f, Q_L, beta, al, al_rho, GS, SC, Na, rate):
    rho = al_rho / al
    v = _nb(f, 0.061) + 0.5
    S_cav = rho * GS + 1 - rho
    N0 = 2 * (Na + al * S_cav * v + (1 - al) * v)
    D = 2 * al * (2 * SC - S_cav * v)
    return _hand_scan(f, 8.0, 1.545, 0.43, Q_L, beta, rate, 5.1, 0.91, N0, D, sig=al)


def _quax2021():
    f, T, Q_L, beta, t, snr = 10.40176e9, 0.99 * 6.41 / 5.38, 3.6e4, 76 / 36 - 1, 4203.0, 1.2816
    v0, vo = 220e3 / _c, 232e3 / _c
    F = lambda v: v / (math.sqrt(math.pi) * v0 * vo) * (math.exp(-(v - vo) ** 2 / v0 ** 2)  # noqa: E731
                                                         - math.exp(-(v + vo) ** 2 / v0 ** 2))
    L = lambda v: 1 / (1 + (2 * Q_L * f * v * v / 2 / f) ** 2)                              # noqa: E731
    J = integrate.quad(lambda v: F(v) ** 2 / (f * v) * L(v) ** 2, 1e-12, vo + 8 * v0, limit=400,
                       epsrel=1e-10)[0]                    # int s^2 L^2 df, f = f_a (1 + v^2/2)
    A = beta / (1 + beta) * Q_L * _K(f, 8.1, 0.08056, 0.69) / (_k * T)
    return (snr ** 2 / (t * A ** 2 * J)) ** 0.25


HAND_G = {
    "ADMX_1A": lambda: _hand_scan(660e6, 6.8, 136.0, 0.4, 5e4, 1.0, 35e6 / (78958 * 100.0), 1.2816, 1.0,
                                  _k * 0.5 / (_h * 660e6)),
    "HAYSTAC_2017": lambda: _hand_scan(5.75e9, 9.0, 1.545, 0.5, 1e4, 2.0, 108.8e6 / (6766 * 900.0), 5.1, 0.76,
                                       _nb(5.75e9, 0.127) + 0.5 + 1.35, _nb(5.75e9, 0.47) - _nb(5.75e9, 0.127)),
    "HAYSTAC_2021": lambda: _haystac2(4.14e9, 6180.0, 6.9, 0.794, 0.63, 0.11, 0.41, 0.08, 73e6 / (828 * 3600.0)),
    "HAYSTAC_2023": lambda: _haystac2(4.49e9, 5915.0, 6.5, 0.775, 0.60, 0.08, 0.40, 0.08, 64e6 / (801 * 3600.0)),
    "CAPP_12TB": lambda: _hand_scan(1.10e9, 10.31, 36.85, 0.6, 1.1e5 / 3, 2.0, 20.0, 5.0, 0.88,
                                    _k * 0.215 / (_h * 1.10e9)),
    "TASEH_2022": lambda: _hand_scan(4.75e9, 8.0, 0.234, 0.605, 6.07e4 / 3, 2.0, 105e3 / 2220.0, 5.0, 1.0,
                                     _k * 2.2 / (_h * 4.75e9)),
    "QUAX_2019": lambda: math.sqrt(1.64 * _k * 15.3 * math.sqrt(7812.5 / 1200.0)
                                   / (0.5 * 2.01e5 * _K(9.067e9, 2.0, 0.03643, 0.589))),
    "QUAX_2021": _quax2021,
    "ORGAN_Q_2025": lambda: _hand_scan(6.25e9, 6.26, 0.708, 0.42, 1700.0, 1.45, 200e6 / (489 * 2700.0), 8.33,
                                       0.72, _k * 0.928 / (_h * 6.25e9)),
}


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_preset_limit_equals_hand_computation(name):
    """Pins each reconstructed limit to the independent hand value above (1e-4): a factor 2 in any rate,
    efficiency, Q_a, noise or signal term moves g by >= 2^(1/4) - 1 = 19%."""
    assert set(HAND_G) == set(REPRODUCIBLE)
    assert hs.preset_limit(name)["g_GeV"] == approx(HAND_G[name](), rel=1e-4), name


def test_ak17_eq5_absolute_anchor():
    """AK17 (1611.07123) Eq. 5 written out literally -- independent of our P_coef and line-shape code --
    equals prefactor="haystac" with Q_a = 1e6 to 1e-7 (core h/e); with HAYSTAC 2017's own numbers (N_sys 2.2-3 quanta,
    eta = 0.76 on the SNR, 2.3 x KSVZ at Lambda = 78 MeV) it brackets the published 108.8 MHz / 1692 h."""
    hbar, alpha, eV = _hbar, 7.2973525693e-3, 1.602176634e-19

    def ak17(QL, Qa, Sig, gg, Lam_eV, beta, B, V, C, N):
        Lam = Lam_eV * eV
        return 0.8 * QL * Qa / Sig ** 2 * (gg ** 2 * alpha ** 2 / math.pi ** 2 * hbar ** 3 * _c ** 3 * _RHO
                                           / Lam ** 4) ** 2 * (beta / (1 + beta) * B ** 2 * V * C / (hbar * _mu0 * N)) ** 2
    f = 5.75e9
    gK = hs.g_gamma_to_gagg(hs.freq_to_mass_eV(f), 0.97 * 2.3, 78.0)
    cav = hs.Cavity(B_T=9.0, V_m3=1.545e-3, C=0.5, Q0=3e4, beta=2.0)
    rx = hs.Receiver(T_sys_K=3.0 * _h * f / _k)
    r = hs.scan_rate(f, cav, rx, g_GeV=float(gK), snr=5.1, Q_a=1e6, prefactor="haystac")
    assert r == approx(ak17(1e4, 1e6, 5.1, 0.97 * 2.3, 78e6, 2.0, 9.0, 1.545e-3, 0.5, 3.0), rel=1e-7)
    r_pub = 108.8e6 / (1692 * 3600.0)
    lo = ak17(1e4, 1e6, 5.1 / 0.76, 0.97 * 2.3, 78e6, 2.0, 9.0, 1.545e-3, 0.5, 3.0)
    hi = ak17(1e4, 1e6, 5.1 / 0.76, 0.97 * 2.3, 78e6, 2.0, 9.0, 1.545e-3, 0.5, 2.2)
    assert lo < r_pub < hi


# Reproduction windows (doc Sec. 8), applied after modelling every correction the papers document
# (HAYSTAC alpha and S_C, QUAX 2021 sigma_max, ORGAN filter retention): +-20% in g for threshold and exact
# likelihood limits; [0.60, 1.25] for "likelihood_approx" (ADMX), whose real-data likelihood the zero-excess
# median only approximates (measured excesses move the effective SNR target from 1.28 to ~1.3-3.3).
WINDOW = {"threshold": (0.80, 1.25), "likelihood": (0.80, 1.25), "likelihood_approx": (0.60, 1.25)}


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_preset_reproduction(name):
    r = hs.preset_limit(name)
    lo, hi = WINDOW[PRESETS[name]["analysis"]["limit_type"]]
    assert lo <= r["ratio"] <= hi, (name, r["ratio"])


def test_preset_reproduction_ensemble():
    """No global bias above 10% over the presets whose limit we model like-for-like (not ADMX's)."""
    names = [n for n in REPRODUCIBLE if PRESETS[n]["analysis"]["limit_type"] != "likelihood_approx"]
    ratios = np.array([hs.preset_limit(n)["ratio"] for n in names])
    assert len(names) == 8
    assert abs(np.mean(np.log(ratios))) < math.log(1.10)


# REGRESSION SNAPSHOT: reconstructed/published |g|.  These pins
# carry no physics -- they only flag unintended changes; validation is the hand reconstruction above.
SNAPSHOT_RATIO = {
    "ADMX_1A": 0.749, "HAYSTAC_2017": 1.150, "HAYSTAC_2021": 1.063, "HAYSTAC_2023": 1.009,
    "CAPP_12TB": 0.997, "TASEH_2022": 1.079, "QUAX_2019": 0.931, "QUAX_2021": 1.007, "ORGAN_Q_2025": 0.833,
}


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_preset_regression_snapshot(name):
    assert set(SNAPSHOT_RATIO) == set(REPRODUCIBLE)
    assert hs.preset_limit(name)["ratio"] == approx(SNAPSHOT_RATIO[name], rel=0.01), name


def test_admx_1d_implied_rate():
    """Run 1D publishes no step size/live time: the published limit implies ~8 days of optimal live
    time over 210 MHz, far less than the ~1 yr calendar (doc Sec. 8)."""
    assert "ADMX_1D" not in REPRODUCIBLE
    out = hs.implied_rate("ADMX_1D")
    assert 2 < out["live_time_days"] < 30
    with pytest.raises(ValueError, match="rate"):
        hs.preset_limit("ADMX_1D")
    with pytest.raises(KeyError, match="unknown"):
        hs.preset("nope")


def test_etaout_independent_reference():
    """Independent (blind) derivation of HAYSTAC 2301.09721 Eq. A4: N(Delta) = N0 + D 4beta/(1+beta)^2 L, the
    closed-form beta_opt, on/off rate ratios, the HAYSTAC IIa limit and the 4.0 dB infeasibility, for the
    HAYSTAC_2021 preset.  The reference quotes 6-8 digits; our values agree to <= 3e-9 (n(T) constants differ at
    2e-9)."""
    f = 4.14e9
    cav, rx, _ = hs.preset_models("HAYSTAC_2021")
    off = replace(rx, squeeze_dB=0.0)
    Q0, b = cav.Q0, 6.9
    ab = 4 * b / (1 + b) ** 2

    def n0_d(r):
        N0 = float(r.N_sys(f, 1e12, Q0, b))
        return N0, (float(r.N_sys(f, 0.0, Q0, b)) - N0) / ab

    N0, D = n0_d(rx)
    N0o, Do = n0_d(off)
    assert (N0, D) == (approx(0.63447711, rel=1e-7), approx(1.05017842, rel=1e-7))
    assert (N0o, Do) == (approx(1.24007536, rel=1e-7), approx(0.44458016, rel=1e-7))
    # Lorentzian form in between
    d = f / (2 * 6180.0) * 0.7
    assert float(rx.N_sys(f, d, Q0, b)) == approx(N0 + D * ab / 1.49, rel=1e-12)
    # closed-form beta_opt: beta^2 - (1 + 2D/N0) beta - 2 = 0
    closed = lambda N0, D: 0.5 * (1 + 2 * D / N0 + math.sqrt((1 + 2 * D / N0) ** 2 + 8))  # noqa: E731
    kw = dict(g_GeV=1e-14, snr=5.1, Q_a=6.7795e5)
    bon, boff = hs.beta_opt(f, cav, rx, **kw), hs.beta_opt(f, cav, off, **kw)
    assert bon == approx(closed(N0, D), rel=1e-6) and bon == approx(4.732944, rel=1e-6)
    assert boff == approx(closed(N0o, Do), rel=1e-6) and boff == approx(2.512911, rel=1e-6)
    r = lambda R, bb: hs.scan_rate(f, replace(cav, beta=bb), R, **kw)  # noqa: E731
    assert r(rx, bon) / r(off, boff) == approx(1.677064, rel=1e-6)
    assert r(rx, 6.9) / r(off, 6.9) == approx(2.089866, rel=1e-6)
    # item 4c
    P = hs.signal_power(1e-14, f, 8.0, 1.545e-3, 0.43, Q_L=6180.0, beta=6.9)
    assert P == approx(5.60322e-24, rel=1e-6)
    assert hs._detuning_integral(f, Q0, 6.9, rx) == approx(5.73394787e5, rel=1e-8)
    g = hs.g_limit_scan(f, cav, rx, rate_Hz_s=73e6 / (828 * 3600.0), snr=5.1, Q_a=6.7795e5)
    assert g == approx(1.32846e-14, rel=1e-5)
    assert P * (g / 1e-14) ** 2 == approx(9.88858e-24, rel=1e-5)
    assert g / (1.95 * hs.g_axion(hs.freq_to_mass_eV(f), "KSVZ")) == approx(1.06342, rel=1e-5)
    # item 3: 3.40182 dB is the most an infinitely strong squeezer shows at T -> 0
    rx0 = replace(rx, T_cav_K=0.0, squeeze_dB=300.0)
    assert -10 * math.log10(rx0.measured_S(f)) == approx(3.40182, rel=1e-6)
    s = 10 ** -0.4
    Na_max = (s - 0.37) / (2 * (1 - s))
    assert Na_max == approx(0.02334898, rel=2e-7)   # the reference has 7 digits
    with pytest.raises(ValueError):
        hs.squeezer_gain_dB(s, 0.63, N_A=0.08)
    with pytest.raises(ValueError):
        hs.squeezer_gain_dB(s, 0.63, N_A=Na_max * 1.0001)
    assert hs.squeezer_gain_dB(s, 0.63, N_A=Na_max * 0.99) > 30.0      # attainable only as G_S -> 0
