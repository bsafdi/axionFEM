"""The exact circuit noise N_V(omega) in the finite-band scan, and the beyond-MQS reach.

Physics: docs/physics/scan_and_signal.md Sec. 6 (tags "S6.*"), docs/physics/readout_circuit.md
(tags "C*", limitations 5 and 7).  Validation note with numbers: docs/validation/finite_band.md.

Everything here is a closed form, an identity, or a comparison against an independently
computed reference (``tests/reference/scan_reference_values.json``); no FEM solve is involved.
The handful of tests that need many network solves are gated behind ``AXF_SLOW``.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from axionfem import _core, circuit as ck, scan

MU0 = ck.MU0
SLOW = pytest.mark.skipif(not os.environ.get("AXF_SLOW"),
                          reason="set AXF_SLOW=1 for the many-settings finite-band checks")
REF = json.loads((Path(__file__).parent / "reference" / "scan_reference_values.json").read_text())

# toy B of the scan spec (Sec. 8): Psi = 100 T m^3, L_p = 1 m
PSI = MU0 * 100.0
LP = MU0 * 1.0
BAND = (120.9e3, 400e3)


def _pole_noise_curve(psi, L, Q, T, eta, f0, n=2001, span=3e-3):
    """N_V(omega) of the single-pole detector itself, as a (omega, N_V) table."""
    w0 = 2 * math.pi * f0
    w = np.geomspace(w0 * (1 - span), w0 * (1 + span), n)
    m = scan.freq_to_mass(f0)
    g = float(scan.g_qcd([m])[0])
    d = _core.psd_at_tuning(psi, L, Q, T, eta, f0, m, g, list(w / (2 * math.pi)), 0.3)
    return list(w), [float(x) for x in d["N_V_V2_s"]]


# ============================================================ the C++ exact-noise kernels
def test_kernel_row_noise_reproduces_kernel_row_for_the_same_noise():
    """Fed the single pole's own N_V(omega), (S6.1) with an external N_V must reproduce the
    detector-based kernel: the only difference is the log-log interpolation of the table."""
    Q, T, eta, f0 = 1e4, 0.030, 10.0, BAND[0]
    w0 = 2 * math.pi * f0
    m = scan.freq_to_mass(f0)
    g = float(scan.g_qcd([m])[0])
    xi = _core.xi_opt(_core.bandwidths(w0, Q, T, eta)["n"], eta)
    K_det = _core.kernel_row(PSI, LP, Q, T, eta, m, g, [w0], [xi], "weak", 96)
    w, NV = _pole_noise_curve(PSI, LP, Q, T, eta, f0)
    K_ext = _core.kernel_row_noise(PSI, m, g, [w], [NV], "weak", 96)
    assert K_ext[0] == pytest.approx(K_det[0], rel=1e-8)
    # and the narrow-line form (S6.3) with the exact N_V(omega_a)
    nv_a = float(np.exp(np.interp(math.log(2 * math.pi * scan.mass_to_freq(m)),
                                  np.log(w), np.log(NV))))
    K_nl = _core.kernel_narrow_line_noise([PSI], [m], [g], [nv_a],
                                          scan.Q_axion_shm())[0]
    ref = _core.kernel_narrow_line(PSI, LP, Q, T, eta, m, g, w0, xi, 0.0, scan.Q_axion_shm())
    assert K_nl == pytest.approx(ref, rel=1e-12)
    # the narrow line is good to ~1e-4 here (N_V is flat over the axion line)
    assert abs(K_nl / K_ext[0] - 1.0) < 3e-3


def test_kernel_scales_as_g_to_the_fourth_and_Psi_to_the_fourth():
    """K ~ Phi_a^4 ~ (g Psi)^4  (S6.2 with S3.2)."""
    Q, T, eta, f0 = 1e5, 0.030, 10.0, 2e5
    m = scan.freq_to_mass(f0)
    w, NV = _pole_noise_curve(PSI, LP, Q, T, eta, f0)
    K0 = _core.kernel_row_noise(PSI, m, 1e-19, [w], [NV], "weak", 64)[0]
    assert _core.kernel_row_noise(PSI, m, 2e-19, [w], [NV], "weak", 64)[0] == \
        pytest.approx(16 * K0, rel=1e-12)
    assert _core.kernel_row_noise(3 * PSI, m, 1e-19, [w], [NV], "weak", 64)[0] == \
        pytest.approx(81 * K0, rel=1e-12)


def test_reach_band_inverts_the_schedule_exactly():
    """reach_band solves sum_j t_j K_j(m, g) = Theta; in the weak-signal limit K ~ g^4, so
    g95 = g_ref (Theta/sum t K)^{1/4} must come back exactly."""
    Q, T, eta, f0 = 1e5, 0.030, 10.0, 2e5
    m = scan.freq_to_mass(f0)
    g = float(scan.g_qcd([m])[0])
    w, NV = _pole_noise_curve(PSI, LP, Q, T, eta, f0)
    K = _core.kernel_row_noise(PSI, m, g, [w], [NV], "weak", 96)[0]
    t = 2.71 / K                               # exactly enough dwell for Theta95 at g
    rc = _core.reach_band([PSI], [m], [w], [NV], [t], "weak", 96)
    assert rc["g95_GeV"][0] == pytest.approx(g, rel=1e-9)
    assert rc["band"]["live_time_s"] == pytest.approx(t, rel=1e-15)
    assert rc["band"]["n_settings"] == 1 and rc["band"]["allocation"] == "lp"


def test_noise_curve_interpolation_is_exact_at_the_nodes_and_log_log_between():
    """NoiseCurve.at: exact on the grid, and exact for a pure power law between the nodes
    (the kernel is what exercises it, so check it through a one-setting kernel)."""
    m = scan.freq_to_mass(2e5)
    wa = 2 * math.pi * scan.mass_to_freq(m)
    w = list(np.geomspace(wa * 0.5, wa * 2.0, 7))
    NV = [1e-30 * (x / wa) ** 3 for x in w]          # a pure power law
    K_coarse = _core.kernel_row_noise(PSI, m, 1e-19, [w], [NV], "weak", 64)[0]
    w2 = list(np.geomspace(wa * 0.5, wa * 2.0, 201))
    NV2 = [1e-30 * (x / wa) ** 3 for x in w2]
    K_fine = _core.kernel_row_noise(PSI, m, 1e-19, [w2], [NV2], "weak", 64)[0]
    assert K_coarse == pytest.approx(K_fine, rel=1e-12)


def test_bad_noise_curves_are_rejected_with_a_specific_message():
    m = scan.freq_to_mass(2e5)
    with pytest.raises(Exception, match="strictly ascending"):
        _core.kernel_row_noise(PSI, m, 1e-19, [[3.0, 2.0, 1.0]], [[1e-30] * 3], "weak", 64)
    with pytest.raises(Exception, match="not positive and finite"):
        _core.kernel_row_noise(PSI, m, 1e-19, [[1.0, 2.0]], [[1e-30, 0.0]], "weak", 64)
    with pytest.raises(Exception, match="at least 2 samples"):
        _core.kernel_row_noise(PSI, m, 1e-19, [[1.0]], [[1e-30]], "weak", 64)
    with pytest.raises(Exception, match="one of each per setting"):
        _core.kernel_row_noise(PSI, m, 1e-19, [[1.0, 2.0]], [[1e-30, 1e-30]] * 2, "weak", 64)
    with pytest.raises(Exception, match="Psi_SI > 0"):
        _core.kernel_row_noise(0.0, m, 1e-19, [[1.0, 2.0]], [[1e-30, 1e-30]], "weak", 64)
    with pytest.raises(Exception, match="one per target mass"):
        _core.reach_band([PSI, PSI], [m], [[1.0, 2.0]], [[1e-30, 1e-30]], [1.0], "weak", 64)


def test_lp_schedule_is_exposed_and_feasible():
    """The dual simplex of (S6.4) through the bindings: every target covered at >= Theta."""
    K = [[1.0, 0.1, 0.0], [0.2, 1.0, 0.3], [0.0, 0.1, 1.0]]
    s = _core.lp_schedule(K, 2.71)
    assert s["min_coverage"] >= 1.0 - 1e-12
    cov = np.array(K) @ np.asarray(s["t_s"])
    assert cov.min() >= 2.71 * (1 - 1e-12)
    assert s["t_total_s"] == pytest.approx(sum(s["t_s"]), rel=1e-15)


# ================================================================== the band echo
def test_reach_echoes_the_scan_band():
    """Every reach call must describe what it scanned (mass/frequency range, settings,
    allocation, live time, benchmark and conventions)."""
    r = scan.reach(PSI, LP, "conservative", live_time_yr=3.0, f_min=1.209e5, f_max=9.9138e5,
                   n=17, allocation="target")
    b = r["band"]
    assert b["f_min_Hz"] == pytest.approx(1.209e5, rel=1e-12)
    assert b["f_max_Hz"] == pytest.approx(9.9138e5, rel=1e-12)
    assert b["m1_eV"] == pytest.approx(scan.freq_to_mass(1.209e5), rel=1e-12)
    assert b["m2_eV"] == pytest.approx(scan.freq_to_mass(9.9138e5), rel=1e-12)
    assert b["n_targets"] == 17 and b["allocation"] == "target"
    assert b["live_time_yr"] == pytest.approx(3.0, rel=1e-9)
    assert b["live_time_s"] == pytest.approx(3.0 * scan.YEAR, rel=1e-12)
    assert (b["Q"], b["T_K"], b["eta_A"]) == pytest.approx((1e6, 0.030, 10.0), rel=1e-12)
    assert b["Psi_SI_H_T_m2"] == pytest.approx(PSI, rel=1e-12)
    assert b["L_H"] == pytest.approx(LP, rel=1e-12)
    assert b["F_SI_T4_H2_m8"] == pytest.approx(PSI ** 4 / LP ** 2, rel=1e-12)
    assert (b["Q_axion"], b["Theta"], b["rho_GeV_cm3"]) == pytest.approx((6.8e5, 2.71, 0.3))
    assert (b["v0_kms"], b["vobs_kms"]) == pytest.approx((220.0, 232.0))
    assert b["benchmark"] == "conservative" and b["model"] == "DFSZ"
    assert b["C_gamma"] == pytest.approx(1.77)
    # the t_req of the target allocation is echoed too (S5.6)
    assert b["t_req_s"] > 0
    assert b["statistic"] == "weak" and b["kernel"] == "local"
    # the numbers themselves are unchanged by the echo
    assert np.all(np.isfinite(r["g95_GeV"])) and r["g95_GeV"].size == 17


def test_log_uniform_band_echo_reports_t_req_equal_to_the_live_time():
    r = scan.reach(PSI, LP, "optimistic", live_time_yr=1.0, n=5, allocation="log_uniform")
    assert r["band"]["allocation"] == "log_uniform"
    assert r["band"]["t_req_s"] == pytest.approx(scan.YEAR, rel=1e-12)


# ======================================================= frequency-dependent coupling (3)
def test_coupling_table_is_exact_at_the_nodes_and_log_log_between():
    f = np.array([1e5, 1e6, 1e7])
    P = np.array([1.0, 2.0, 8.0]) * 1e-5
    psi = ck.coupling_table(f, P)
    for fi, Pi in zip(f, P):
        assert psi(2 * math.pi * fi) == pytest.approx(Pi, rel=1e-14)
    # log-log between 1e6 and 1e7: a power law with index log2(4) = 2
    assert psi(2 * math.pi * math.sqrt(1e6 * 1e7)) == pytest.approx(2e-5 * 2.0, rel=1e-12)
    # const extrapolation by default
    assert psi(2 * math.pi * 1e3) == pytest.approx(1e-5, rel=1e-14)
    assert psi(2 * math.pi * 1e9) == pytest.approx(8e-5, rel=1e-14)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        ck.coupling_table(f, P, extrapolate="error")(2 * math.pi * 1e3)


def test_readout_takes_a_frequency_dependent_Psi():
    """Psi_SI may be a callable; the EMF, the flux and the kernel must all use Psi(omega_0)."""
    psi_of = ck.coupling_table([1e5, 1e6], [PSI, 4 * PSI])
    r = ck.series_readout(L_p_H=LP, Psi_SI=psi_of, f_Hz=1e5, Q=1e6, T_K=0.03, eta_A=10.0)
    r_const = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=1e5, Q=1e6, T_K=0.03, eta_A=10.0)
    assert r.Psi_at(2 * math.pi * 1e5) == pytest.approx(PSI, rel=1e-14)
    assert r.Psi_at(2 * math.pi * 1e6) == pytest.approx(4 * PSI, rel=1e-14)
    w0 = 2 * math.pi * 1e5
    assert r.signal(w0)["Phi_a_Wb"] == pytest.approx(r_const.signal(w0)["Phi_a_Wb"], rel=1e-14)
    assert r.equivalent(w0)["scan_inputs"]["Psi_SI_H_T_m2"] == pytest.approx(PSI, rel=1e-14)
    # K ~ Psi^4: a doubling of Psi is a factor 16 at the same noise
    f_mid = math.sqrt(1e5 * 1e6)
    r.tune(f_mid)
    w = 2 * math.pi * f_mid
    r_const.tune(f_mid)
    k_var = r.kernel(w)["K_per_s"]
    k_con = r_const.kernel(w)["K_per_s"]
    assert k_var / k_con == pytest.approx((r.Psi_at(w) / PSI) ** 4, rel=1e-10)


def test_fullwave_tables_and_pickup_coupling_read_the_sweep():
    """Psi(omega) and Z_p(omega) both come out of results["fullwave"]["sweep"]."""
    sweep = [{"f_Hz": 1e6, "k_R": 0.02, "Psi_re": 1e-5, "Psi_im": 0.0, "Z_re_ohm": 1e-6,
              "Z_im_ohm": -2 * math.pi * 1e6 * 1e-6, "L_H": 1e-6},
             {"f_Hz": 1e7, "k_R": 0.2, "Psi_re": 1.0e-5, "Psi_im": 1.0e-5, "Z_re_ohm": 1e-2,
              "Z_im_ohm": -2 * math.pi * 1e7 * 1.1e-6, "L_H": 1.1e-6, "R_loss_ohm": 1e-3}]
    res = {"fullwave": {"sweep": sweep}, "coupling": {"Psi_SI_H_T_m2": 1e-5, "L_p_H": 1e-6}}
    t = ck.fullwave_tables(res)
    assert t["Psi_SI"][0] == pytest.approx(1e-5, rel=1e-14)
    assert t["Psi_SI"][1] == pytest.approx(math.sqrt(2) * 1e-5, rel=1e-14)
    assert t["R_ohm"][1] == pytest.approx(1e-2 + 1e-3, rel=1e-14)
    psi_of = ck.pickup_coupling(res)
    assert psi_of(2 * math.pi * 1e6) == pytest.approx(1e-5, rel=1e-14)
    Zp = ck.pickup_impedance(res, model="fullwave")
    assert Zp(2 * math.pi * 1e6).imag == pytest.approx(-2 * math.pi * 1e6 * 1e-6, rel=1e-12)
    with pytest.raises(ValueError, match="no 'fullwave'/'sweep'"):
        ck.fullwave_tables({"coupling": {}})


# ======================================================= the dispersive / beyond-MQS reach
@pytest.mark.parametrize("allocation", ["target", "log_uniform"])
def test_reach_dispersive_reproduces_reach_for_constant_parameters(allocation):
    """With Psi, L, Q, T, eta_A constant, reach_dispersive must reproduce the C++ reach_local
    (S5.5-S5.8) to the quadrature error of its own dwell normalisation."""
    kw = dict(live_time_yr=3.0, f_min=1.209e5, f_max=9.9138e5, allocation=allocation, n=13)
    a = scan.reach(PSI, LP, {"Q": 1e6, "T": 0.030, "eta_A": 10.0}, **kw)
    f = np.geomspace(1e5, 1.1e6, 9)
    b = scan.reach_dispersive(f, PSI, LP, Q=1e6, T_K=0.030, eta_A=10.0, n_quad=2001, **kw)
    assert np.allclose(b["g95_GeV"], a["g95_GeV"], rtol=1e-10)
    assert np.allclose(b["tau_s_per_rad_s"], a["tau_s_per_rad_s"], rtol=1e-10)
    assert b["band"]["allocation"] == allocation
    assert b["band"]["live_time_yr"] == pytest.approx(3.0)
    assert b["band"]["n_targets"] == 13
    assert "Simpson" in b["band"]["allocation_quadrature"]


def test_reach_dispersive_tracks_F_when_Psi_and_L_vary():
    """g95 ~ F^{-1/4} = (Psi^4/L^2)^{-1/4} at fixed tau, Q, G-bar  (S5.8)."""
    f = np.geomspace(1e5, 1e6, 11)
    psi = PSI * (f / f[0]) ** 0.25        # F rises as f
    out = scan.reach_dispersive(f, psi, LP, Q=1e6, T_K=0.030, eta_A=10.0, n=11,
                                allocation="log_uniform")
    flat = scan.reach_dispersive(f, PSI, LP, Q=1e6, T_K=0.030, eta_A=10.0, n=11,
                                 allocation="log_uniform")
    ratio = out["g95_GeV"] / flat["g95_GeV"]
    expect = (psi / PSI) ** -1.0          # g95 ~ Psi^-1 at fixed L, tau, Q
    assert np.allclose(ratio, expect, rtol=1e-9)
    assert np.allclose(out["F_SI_T4_H2_m8"], psi ** 4 / LP ** 2, rtol=1e-12)


def test_reach_dispersive_marks_the_self_resonant_part_of_the_band():
    """L(omega) <= 0 is above the first self-resonance: no series tuning exists there."""
    f = np.geomspace(1e6, 1e8, 9)
    L = LP * (1.0 - (f / 3e7) ** 2)       # changes sign at 30 MHz
    out = scan.reach_dispersive(f, PSI, L, L_eff_H=LP, R_ohm=np.full(f.size, 1e-3), T_K=0.030, eta_A=10.0,
                                n=25)
    assert np.any(~np.isfinite(out["g95_GeV"]))
    assert np.all(np.isfinite(out["g95_GeV"][out["f_Hz"] < 2.5e7]))
    assert len(out["band"]["self_resonant_f_Hz"]) == int((~np.isfinite(out["g95_GeV"])).sum())
    with pytest.raises(ValueError, match="above its self-resonance"):
        scan.reach_dispersive(f, PSI, L, L_eff_H=LP, R_ohm=np.full(f.size, 1e-3), f_min=4e7, f_max=8e7,
                              T_K=0.030, eta_A=10.0)


def test_reach_dispersive_argument_errors_are_specific():
    f = np.array([1e5, 1e6])
    with pytest.raises(ValueError, match="exactly one of Q= and R_ohm="):
        scan.reach_dispersive(f, PSI, LP, Q=1e6, R_ohm=1e-3)
    with pytest.raises(ValueError, match="exactly one of Q= and R_ohm="):
        scan.reach_dispersive(f, PSI, LP)
    with pytest.raises(ValueError, match="allocation must be"):
        scan.reach_dispersive(f, PSI, LP, Q=1e6, allocation="uniform")
    with pytest.raises(ValueError, match="has 3 values but f_Hz has 2"):
        scan.reach_dispersive(f, np.array([1.0, 2.0, 3.0]) * PSI, LP, Q=1e6)


def test_reach_fullwave_on_a_synthetic_sweep_matches_the_tables():
    """The whole beyond-MQS path on a hand-made sweep: Psi(omega) and L(omega) rising as
    (kR)^2, a radiation resistance rising as omega^4, and the MQS comparison."""
    f = np.geomspace(1e6, 1e7, 25)
    kR = f / 2.4e7
    psi = 1e-5 * (1 + 0.21 * kR ** 2)
    L = 1e-6 * (1 + 0.19 * kR ** 2)
    Rr = 1.35 * (f / 2.34e7) ** 4
    res = {"name": "synthetic", "coupling": {"Psi_SI_H_T_m2": 1e-5, "L_p_H": 1e-6},
           "fullwave": {"sweep": [
               {"f_Hz": float(f[i]), "k_R": float(kR[i]), "Psi_re": float(psi[i]), "Psi_im": 0.0,
                "Z_re_ohm": float(Rr[i]), "Z_im_ohm": float(-2 * math.pi * f[i] * L[i]),
                "L_H": float(L[i]), "R_loss_ohm": 1e-3} for i in range(f.size)]}}
    out = scan.reach_fullwave(res, {"Q": 1e5, "T": 0.030, "eta_A": 20.0}, Q_floor=1e5, n=21,
                              live_time_yr=5.0, allocation="log_uniform")
    assert np.all(np.isfinite(out["g95_GeV"]))
    # F = Psi^4/L^2 is read straight off the interpolated full-wave columns
    assert out["ratio_F_fullwave_over_mqs"][0] == pytest.approx(
        (out["Psi_SI"][0] / 1e-5) ** 4 / (out["L_H"][0] / 1e-6) ** 2, rel=1e-12)
    # Q = omega L/(R_rad + R_loss), capped; radiation eats the top of the band
    assert np.all(out["Q"] <= 1e5 * (1 + 1e-12))
    assert out["Q"][-1] < .5 * out["Q"][0]
    assert out["ratio_g95_fullwave_over_mqs"][-1] > out["ratio_g95_fullwave_over_mqs"][0]
    b = out["band"]
    assert b["benchmark"] == "custom" and b["Q_cap"] == 1e5
    assert b["k_R"][0] == pytest.approx(kR[0], rel=1e-12)
    assert b["live_time_yr"] == pytest.approx(5.0)


# ========================================================= the finite-band planner, fast
def test_finite_band_on_a_plain_series_loop_matches_the_single_pole():
    """A plain series RLC *is* a single pole, so the exact N_V kernel and the reduced one must
    agree: on resonance to the N_V round-off, and the LP plans to the interpolation error."""
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e4, T_K=0.030, eta_A=10.0)
    out = scan.finite_band(r, 1.209e5, 2.0e5, n_settings=6, n_targets=6, n_v=48,
                           points_per_decade=24.0, band_points=50)
    c = out["comparison"]
    assert c["kernel_on_resonance_max_rel_dev"] < 1e-5
    assert c["t_total_ratio_reduced_over_exact"] == pytest.approx(1.0, abs=2e-3)
    assert c["min_coverage_of_reduced_schedule"] > 0.99
    assert out["schedule"]["min_coverage"] >= 1.0 - 1e-12
    assert out["narrow_line"]["max_rel_dev"] < 1e-3
    # the LP plan really covers every target at Theta with the exact kernel
    cov = out["K_per_s"] @ out["schedule"]["t_s"] / 2.71
    assert cov.min() >= 1.0 - 1e-9
    # and the reach it implies is the target coupling there
    assert np.allclose(out["reach"]["g95_GeV"] / out["targets"]["g_target_GeV"], 1.0, rtol=2e-6)


def test_finite_band_band_echo_is_self_describing():
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e4, T_K=0.030, eta_A=10.0)
    out = scan.finite_band(r, 1.209e5, 2.0e5, n_settings=4, n_targets=4, n_v=32,
                           points_per_decade=16.0, band_points=30, xi_factors=(0.5, 1.0),
                           live_time_yr=3.0, reduced=False, narrow_line=False)
    b = out["band"]
    assert b["n_settings"] == 8 and b["n_targets"] == 4
    assert b["allocation"].startswith("lp") and b["kernel"].startswith("exact")
    assert b["xi_factors"] == [0.5, 1.0]
    assert b["f_min_Hz"] == pytest.approx(scan.mass_to_freq(b["m1_eV"]), rel=1e-12)
    assert b["live_time_s"] == pytest.approx(out["schedule"]["t_total_s"], rel=1e-12)
    assert b["requested_live_time_yr"] == 3.0
    assert b["noise_grid"]["points"] == len(out["settings"]["f0_Hz"]) * 0 + \
        b["noise_grid"]["points"]          # present and positive
    assert b["noise_grid"]["points"] > 0
    assert b["Psi_frequency_dependent"] is False
    assert out["coverage_at_live_time"].size == 4


def test_finite_band_rejects_a_readout_without_a_tuner():
    c = ck.Circuit()
    c.source("pickup", "gnd", "a", emf=0.0, Z=lambda w: complex(1.0, -w * LP))
    c.short("L_in", "a", "gnd")
    r = ck.Readout(circuit=c, squid=ck.Squid(M_H=1e-9, eta_A=1.0, S_Phi=1e-30, S_J=1e-30,
                                            input_coil="L_in"), Psi_SI=PSI, L_p_H=LP)
    with pytest.raises(ValueError, match="no tuning capacitor"):
        scan.finite_band(r, 1e5, 2e5)
    with pytest.raises(TypeError, match="must return a circuit.Readout"):
        scan.finite_band(lambda f: 1.0, 1e5, 2e5)


def test_tune_is_exact_over_a_decade_of_frequencies():
    """The bisection in Readout.tune must not accept a bracket whose endpoints are recomputed by
    dividing back out: when the closed form is already exact to round-off, the residual's *sign* at
    such an endpoint is round-off and the search can return the bracket end (a 15 % error in C,
    leaving Im Z_eff != 0).  For a plain series loop C = 1/(omega^2 L)."""
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=1.209e5, Q=1e4, T_K=0.030, eta_A=10.0)
    worst = 0.0
    for f0 in np.geomspace(1e5, 1e6, 200):
        C = r.tune(float(f0))
        w = 2 * math.pi * float(f0)
        worst = max(worst, abs(C * w * w * LP - 1.0))
        assert abs(r.Z_eff(w).imag) < 1e-10 * abs(r.Z_eff(w))
    assert worst < 1e-14


def test_noise_curve_resolves_the_resonance_and_is_positive():
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e5, T_K=0.030, eta_A=10.0)
    w0 = 2 * math.pi * BAND[0]
    nc = r.noise_curve(w0, half_widths=20.0, decades=2.0, points_per_decade=20.0)
    w, NV = nc["omega_rad_s"], nc["N_V_V2_s"]
    assert np.all(np.diff(w) > 0) and np.all(NV > 0)
    assert w0 in w                                      # the resonance itself is a node
    assert nc["N_V_at_omega0_V2_s"] == pytest.approx(NV[np.argmin(np.abs(w - w0))], rel=1e-14)
    assert NV.min() == pytest.approx(nc["N_V_at_omega0_V2_s"], rel=1e-6)   # minimum on resonance
    # the three components add up
    assert np.allclose(NV, nc["N_V_thermal_V2_s"] + nc["N_V_backaction_V2_s"]
                       + nc["N_V_imprecision_V2_s"], rtol=1e-12)
    # extra_omega is merged, not substituted
    nc2 = r.noise_curve(w0, half_widths=20.0, decades=2.0, points_per_decade=20.0,
                        extra_omega=[0.3 * w0, 3.0 * w0])
    assert nc2["omega_rad_s"].size == w.size + 2
    with pytest.raises(ValueError, match="extra_omega must be positive"):
        r.noise_curve(w0, extra_omega=[-1.0])


def test_a_low_Q_pickup_breaks_the_single_pole_and_R_exp_repairs_it():
    """The headline of limitation 5: with R ~ sqrt(f) and a sensitivity band wider than the
    centre frequency, the constant-R pole misestimates the kernel by tens of per cent; told the
    exponent (Detector::R_exp = 1/2) it recovers the exact answer."""
    w1 = 2 * math.pi * BAND[0]
    R0 = w1 * LP / 300.0

    def Zp(w):
        return complex(R0 * math.sqrt(w / w1), -w * LP)

    dev = {}
    for R_exp in (0.0, 0.5):
        r = ck.series_readout(Z_p=Zp, L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], T_K=0.030, eta_A=10.0)
        out = scan.finite_band(r, 1.209e5, 2.4e5, n_settings=5, n_targets=10, n_v=64,
                               points_per_decade=24.0, band_points=60, reduced_R_exp=R_exp)
        dev[R_exp] = out["comparison"]
        assert out["local_schedule"]["noise_doubling_over_f0"][0] > 1.0   # a very wide band
    assert dev[0.0]["kernel_max_rel_dev"] > 0.1                  # the pole is badly wrong
    assert dev[0.0]["g95_same_schedule_max_rel_dev"] > 0.02
    assert dev[0.5]["kernel_max_rel_dev"] < 0.01                 # ... and R_exp fixes it
    assert dev[0.5]["g95_same_schedule_max_rel_dev"] < 1e-3


# ========================================================= the finite-band planner, slow
@SLOW
def test_finite_band_reproduces_the_spec_section_6_demonstration():
    """The reference demonstration of spec Sec. 6, recomputed from the exact circuit noise.

    The reference detector is Q = 10^4 at every tuning with R constant *within* a tuning, i.e.
    a pickup resistance R = omega_0 L_p/Q rebuilt per setting -- which is what the readout
    factory form of finite_band() is for.  Settings and couplings are the reference's.
    """
    R = REF["finite_band_demo"]

    def make(f0):
        return ck.series_readout(Z_p=ck.pickup_impedance(L_p_H=LP, Q=1e4, f_ref_Hz=f0),
                                 L_p_H=LP, Psi_SI=PSI, f_Hz=f0, T_K=0.030, eta_A=10.0)

    out = scan.finite_band(make, *BAND, n_settings=200, n_targets=120,
                           xi_factors=(0.3, 1.0, 3.0), n_v=64, reduced=False, narrow_line=True)
    ls = out["local_schedule"]
    assert ls["noise_doubling_over_f0"][0] == pytest.approx(R["fw_over_f0_at_fa"], rel=1e-6)
    assert ls["t_total_s"] == pytest.approx(R["t_local_formula_s"], rel=1e-3)
    assert out["schedule"]["t_total_s"] == pytest.approx(R["t_LP_finite_band_s"], rel=3e-3)
    assert ls["lp_over_local_time"] == pytest.approx(R["LP_over_local"], rel=3e-3)
    assert ls["coverage_interior_min"] == pytest.approx(
        R["local_schedule_coverage_interior_min"], rel=5e-3)
    assert ls["coverage_interior_max"] == pytest.approx(
        R["local_schedule_coverage_interior_max"], rel=5e-3)
    assert ls["coverage_min"] == pytest.approx(R["local_schedule_coverage_min_all"], rel=0.05)
    assert out["narrow_line"]["max_rel_dev"] < 2e-4


@SLOW
def test_high_Q_network_finite_band_reproduces_the_reduced_detector():
    """A realistic high-Q network (input coil, leads, lossy capacitor): the finite-band result
    with the exact N_V must reproduce the reduced single-pole scan rate and reach."""
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=2e7, T_K=0.010, eta_A=0.1,
                          L_in_H=0.1 * LP, R_in_ohm=1e-9, lead_length_m=2.0, L_per_m=1e-7,
                          R_per_m=1e-9, T_lead_K=4.0, tan_delta=1e-6)
    out = scan.finite_band(r, *BAND, n_settings=32, n_targets=48, n_v=96, convergence=True)
    c = out["comparison"]
    # L_eff = L_p + L_in + L_lead (0.1 L_p and 2 m x 100 nH/m)
    assert out["settings"]["participation"][0] == pytest.approx(
        LP / (LP + 0.1 * LP + 2.0e-7), rel=1e-6)
    assert c["kernel_on_resonance_max_rel_dev"] < 1e-5
    assert c["t_total_ratio_reduced_over_exact"] == pytest.approx(1.0, abs=5e-3)
    assert c["min_coverage_of_reduced_schedule"] > 0.99
    assert c["g95_same_schedule_max_rel_dev"] < 0.01
    assert out["convergence"]["noise_grid_halved_max_rel_dev"] < 0.02
    assert out["convergence"]["n_v_halved_max_rel_dev"] < 1e-6


@SLOW
def test_parasitic_capacitance_stays_a_lossless_tap_in_the_finite_band():
    """readout_circuit.md Sec. 5: a parasitic C across the pickup is a lossless capacitive tap,
    so the pole stays exact.  The apparent kernel discrepancy must therefore fall as the
    band-wide N_V grid resolves the pole at the self-resonance."""
    C_par = 1.0 / ((2 * math.pi * 5e5) ** 2 * LP)
    devs = []
    for bp in (200, 800):
        r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e6, T_K=0.030, eta_A=10.0,
                              C_par_F=C_par)
        out = scan.finite_band(r, *BAND, n_settings=32, n_targets=48, n_v=96, band_points=bp)
        devs.append(out["comparison"])
        assert out["comparison"]["kernel_on_resonance_max_rel_dev"] < 1e-4
    assert devs[1]["kernel_max_rel_dev"] < 0.25 * devs[0]["kernel_max_rel_dev"]
    assert devs[1]["g95_same_schedule_max_rel_dev"] < 0.25 * devs[0]["g95_same_schedule_max_rel_dev"]


# =========================== coupling efficiency, noise referral and circuit guards ==========
def _weak_transformer(Q_p=100.0, k=0.02):
    """A flux transformer with a weak coupling: L_1 = L_2 = L_p, L_in = 0.1 L_p."""
    return ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=Q_p, T_K=0.030, eta_A=10.0,
                             L_1_H=LP, L_2_H=LP, k_T=k, L_in_H=0.1 * LP)


@pytest.mark.parametrize("Q_p,k", [(1e2, 0.01), (1e2, 0.02), (1e2, 0.1), (1e2, 0.5),
                                   (1e3, 0.02), (1e3, 0.1), (1e4, 0.01), (1e4, 0.02),
                                   (1e6, 0.01), (2e7, 0.01)])
def test_port_coupling_is_Q1_over_sqrt_1_plus_Q1_squared(Q_p, k):
    """`port_coupling` is not 1 for a lossless coupling network.

    With all the dissipation in the pickup, (C5.1b) gives port_coupling = cos(arg T_in,in), and
    that equals Q1/sqrt(1 + Q1^2) exactly, with Q1 = omega_0 M^2/((L_2 + L_in) R) the quality
    factor of the loop seen through the coupling with the pickup port open (C5.1c).  A test
    only at Q_p = 1e6-2e7, where Q1 >> 1, would not see the difference."""
    r = _weak_transformer(Q_p, k)
    w0 = 2 * math.pi * BAND[0]
    eq = r.equivalent(w0)
    M = k * LP                                   # k sqrt(L_1 L_2) with L_1 = L_2 = L_p
    R = w0 * LP / Q_p
    Q1 = w0 * M * M / ((LP + 0.1 * LP) * R)
    assert eq["port_coupling"] == pytest.approx(Q1 / math.sqrt(1 + Q1 ** 2), rel=1e-6)
    assert eq["coupling_Q1"] == pytest.approx(Q1, rel=1e-5)
    assert eq["eta_A_eff"] / eq["eta_A"] == pytest.approx(math.sqrt(1 + Q1 ** -2), rel=1e-5)


def test_eta_A_is_degraded_by_a_lossless_weak_coupling():
    """The counter-example: a *lossless* reciprocal coupling network with
    Q1 << 1 degrades eta_A by a large factor, and the invariance claim only holds for Q1 >> 1."""
    low = _weak_transformer(1e2, 0.02).equivalent(2 * math.pi * BAND[0])
    high = _weak_transformer(2e7, 0.1).equivalent(2 * math.pi * BAND[0])
    assert low["coupling_Q1"] < 0.1 and low["eta_A_eff"] / low["eta_A"] > 20
    assert high["coupling_Q1"] > 1e4 and high["eta_A_eff"] == pytest.approx(high["eta_A"], rel=1e-6)


def test_analyze_reach_uses_eta_A_eff_not_eta_A():
    """`analyze(reach=True)` must use eta_A_eff, not eta_A, which overstates the reach wherever
    port_coupling < 1."""
    r = _weak_transformer(1e2, 0.02)
    w0 = 2 * math.pi * BAND[0]
    a = r.analyze(BAND[0], reach=True, n=5, f_max=9.9138e5)
    eq = a
    assert eq["eta_A_eff"] / eq["eta_A"] > 20
    assert a["scan_inputs"]["eta_A"] == pytest.approx(eq["eta_A_eff"], rel=1e-12)
    assert a["reach"]["band"]["eta_A"] == pytest.approx(eq["eta_A_eff"], rel=1e-12)
    ref = scan.reach(r.Psi_at(w0), eq["L_eff_H"],
                     {"Q": eq["Q_loaded"], "T": eq["T_eff_K"], "eta_A": eq["eta_A_eff"]},
                     n=5, f_max=9.9138e5)
    assert np.allclose(a["reach"]["g95_GeV"], ref["g95_GeV"], rtol=1e-12)
    old = scan.reach(r.Psi_at(w0), eq["L_eff_H"],
                     {"Q": eq["Q_loaded"], "T": eq["T_eff_K"], "eta_A": eq["eta_A"]},
                     n=5, f_max=9.9138e5)
    assert old["g95_GeV"][0] < 0.6 * a["reach"]["g95_GeV"][0]     # with the bare eta_A the reach would be ~2.3x too good


@pytest.mark.parametrize("kw", [
    dict(Q=1e4, C_par_F=1.0 / ((2 * math.pi * 5e5) ** 2 * MU0), tan_delta_par=1e-2,
         L_in_H=0.3 * MU0, R_in_ohm=1e-4, lead_length_m=2.0, L_per_m=1e-7, R_per_m=1e-4,
         T_lead_K=4.0, tan_delta=1e-3),
    dict(Q=1e3, L_1_H=MU0, L_2_H=MU0, k_T=0.2, L_in_H=0.1 * MU0, R_in_ohm=1e-4, R_2_ohm=1e-3,
         tan_delta=1e-3),
])
def test_nyquist_identity_needs_the_current_referral(kw):
    """`R_noise_over_R_eff` must not refer the branch noise with the *flux* transfer,
    which is not the Nyquist identity (C4.3) once the branches carry different currents.  The
    current referral is exact; the flux one is not."""
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], T_K=0.030, eta_A=10.0, **kw)
    w0 = 2 * math.pi * BAND[0]
    assert r.equivalent(w0)["R_noise_over_R_eff"] == pytest.approx(1.0, rel=1e-12)
    worst_flux = 0.0
    for frac in (1.0, 0.1, 0.01):
        w = w0 * frac
        assert r.circuit.nyquist_check(w, "pickup") < 1e-12          # the identity holds
        th = r.circuit.thermal_noise(w, r.flux)
        Hs2 = abs(r.H_signal(w)) ** 2
        flux = sum(R * th["transfer2"][nm] / Hs2
                   for nm, R, _ in r.circuit.noisy_branches(w)) / r.Z_eff(w).real
        worst_flux = max(worst_flux, abs(math.log(flux)))
    assert worst_flux > 0.1          # ... and the flux referral is off by tens of per cent or more


def test_reciprocity_diagnostics_say_what_they_are():
    """`check_reciprocity` is a tautology (the MNA stamp is symmetric by
    construction).  The transfer-matrix check goes through the solve, including the ideal
    transformer, and is the one `equivalent` reports."""
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e4, T_K=0.030, eta_A=10.0,
                          transformer_ratio=3.0, L_in_H=0.1 * LP, R_in_ohm=1e-9)
    w0 = 2 * math.pi * BAND[0]
    assert r.circuit.check_reciprocity(w0) == 0.0          # exactly zero: tautological
    assert r.circuit.transfer_reciprocity(w0) < 1e-12      # genuine, and satisfied
    eq = r.equivalent(w0)
    assert eq["reciprocity_resid"] < 1e-12
    assert eq["matrix_symmetry_resid"] == 0.0


def test_a_branch_without_a_temperature_keeps_its_vacuum_noise():
    """`T_K = None` must not drop a dissipative branch's noise, or N_V could fall below the
    quantum limit.  None means T = 0 (2n + 1 = 1); `noise=False` is the explicit opt-out."""
    c = ck.Circuit()
    w = 2 * math.pi * 1e6
    c.source("pickup", "gnd", "a", emf=0.0, Z=lambda ww: complex(0.0, -ww * LP))
    c.resistor("R_none", "a", "b", 1.0)                   # no temperature given
    c.capacitor("C", "b", "d", 1.0 / (w * w * LP), T_K=0.0)
    c.short("L_in", "d", "gnd")
    assert [nm for nm, _, _ in c.noisy_branches(w)] == ["R_none"]
    _, R, T = c.noisy_branches(w)[0]
    assert (R, T) == (1.0, 0.0)
    th = c.thermal_noise(w, lambda s: s.I("pickup"))
    # vacuum noise of a 1 ohm resistor: S_V = hbar omega R, referred through |1/Z|^2 = 1/R^2
    assert th["total"] == pytest.approx(ck.HBAR * w * 1.0 / 1.0 ** 2, rel=1e-12)
    c2 = ck.Circuit()
    c2.source("pickup", "gnd", "a", emf=0.0, Z=lambda ww: complex(0.0, -ww * LP))
    c2.resistor("R_quiet", "a", "b", 1.0, T_K=None)
    c2._get("R_quiet").noise = False
    c2.capacitor("C", "b", "d", 1.0 / (w * w * LP), T_K=0.0)
    c2.short("L_in", "d", "gnd")
    assert c2.noisy_branches(w) == []


def test_series_readout_no_loss_guard_counts_a_lossy_parasitic_capacitor():
    """The no-loss guard counts tan_delta_par (and never adds a dimensionless tan_delta to ohms),
    so a loop whose only loss is the parasitic capacitor is accepted."""
    C_par = 1.0 / ((2 * math.pi * 5e5) ** 2 * LP)
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], T_K=0.030, eta_A=10.0,
                          C_par_F=C_par, tan_delta_par=1e-2)
    assert r.equivalent(2 * math.pi * BAND[0])["R_eff_ohm"] > 0
    with pytest.raises(ValueError, match="no dissipation"):
        ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], T_K=0.030, eta_A=10.0)


def test_resonance_guess_uses_the_tuner_not_a_series_sum_of_every_capacitor():
    """`_resonance_guess` must not combine C_par in series with the tuner, which it is
    not.  The guess must be within the bisection's reach even with a big parasitic C."""
    C_par = 1.0 / ((2 * math.pi * 3e5) ** 2 * LP)
    r = ck.series_readout(L_p_H=LP, Psi_SI=PSI, f_Hz=BAND[0], Q=1e4, T_K=0.030, eta_A=10.0,
                          C_par_F=C_par)
    guess = r._resonance_guess()
    assert guess == pytest.approx(BAND[0], rel=0.3)
    assert r.resonance() / (2 * math.pi) == pytest.approx(BAND[0], rel=1e-6)


def test_impedance_table_extrapolates_R_as_a_power_law():
    """R is extrapolated in log-log, not linearly in log f with a clip at zero: a surface or
    radiation resistance is a power law, which log-log extrapolation continues."""
    f = np.array([1e6, 1e7])
    R = np.array([1e-6, 1e-2])               # R ~ f^2 (a superconductor)
    Z = ck.impedance_table(f, R_ohm=R, L_H=np.array([1e-6, 1e-6]))
    assert Z(2 * math.pi * 1e8).real == pytest.approx(1e2, rel=1e-10)      # the power law
    assert Z(2 * math.pi * 1e5).real == pytest.approx(1e-10, rel=1e-10)
    assert Z(2 * math.pi * math.sqrt(1e6 * 1e7)).real == pytest.approx(1e-4, rel=1e-10)
    # a table with a zero entry falls back to the clipped linear form
    Z0 = ck.impedance_table(f, R_ohm=np.array([0.0, 1e-2]), L_H=np.array([1e-6, 1e-6]))
    assert Z0(2 * math.pi * 1e5).real == 0.0


def test_reach_fullwave_trims_the_sweep_at_the_first_self_resonance():
    """Rows above the first self-resonance (L <= 0) would poison the log-log interpolation of Q
    at the band edge, so they are trimmed and the self-resonance is reported."""
    f = np.geomspace(1e6, 1e8, 11)
    L = np.where(f < 3e7, 1e-6, -1e-6)  # isolate trimming, independently of slope estimation
    res = {"name": "sr", "coupling": {"Psi_SI_H_T_m2": 1e-5, "L_p_H": 1e-6},
           "fullwave": {"sweep": [
               {"f_Hz": float(f[i]), "k_R": float(f[i] / 2.4e7), "Psi_re": 1e-5, "Psi_im": 0.0,
                "Z_re_ohm": 1e-6, "Z_im_ohm": float(-2 * math.pi * f[i] * L[i]),
                "L_H": float(L[i]), "R_loss_ohm": 1e-6} for i in range(f.size)]}}
    out = scan.reach_fullwave(res, {"Q": 1e5, "T": 0.030, "eta_A": 10.0}, Q_floor=1e5, n=11)
    b = out["band"]
    n_used, n_all = b["sweep_rows_used"]
    assert n_all == 11 and 2 <= n_used < 11
    assert b["f_self_resonance_Hz"] == pytest.approx(float(f[n_used]), rel=1e-12)
    assert b["f_max_Hz"] <= b["f_self_resonance_Hz"]
    assert np.all(np.isfinite(out["g95_GeV"])) and np.all(out["L_H"] > 0)
    assert np.all(out["Q"] > 0)
    # the whole sweep above the self-resonance has no scan at all
    bad = {"name": "sr2", "coupling": {"Psi_SI_H_T_m2": 1e-5, "L_p_H": 1e-6},
           "fullwave": {"sweep": [dict(r, L_H=-abs(r["L_H"])) for r in res["fullwave"]["sweep"]]}}
    with pytest.raises(ValueError, match="fewer than two frequencies"):
        scan.reach_fullwave(bad, {"Q": 1e5, "T": 0.030, "eta_A": 10.0}, Q_floor=1e5)
