"""Multi-band ([R]) and 2211.00008 multipole scans: axionfem.multipole.

Spec: docs/physics/multipole_scan.md (tags M* and sources [R], [P], [C]).  References:
- a synthetic lossless Foster-form source table (two poles between 5 and 200 MHz) for the band, slope, unit and
  allocation rules of [R]'s conventions (M0b-M7);
- the C++ scan core (S3.10, S5.2) and scan.reach_fullwave (single-band reduction);
- closed forms (lossless/MQS limits of M11, the continuum sum M13 on an MQS table).
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from axionfem import _core, multipole as mp, scan

ROOT = Path(__file__).resolve().parents[1]


def _foster_table_as_stored():
    """A lossless one-port X(f) = w L0 + sum_k w A_k / (w_k^2 - w^2) (Foster's first form: dX/dw > 0, poles at 90
    and 180 MHz) with a small series resistance, 5-200 MHz at 10 kHz, in COMSOL's exp(+jwt) convention and stored as
    -Z (as some COMSOL tables are); V is a smooth open-circuit EMF per unit source."""
    f = np.round(np.arange(5e6, 200e6 + 1, 1e4))
    w = 2 * np.pi * f
    X = w * 60e-9
    for fk, A in ((90.005e6, 3.4e9), (180.005e6, 2.0e9)):   # poles between grid points
        X = X + w * A / ((2 * np.pi * fk) ** 2 - w ** 2)
    Z = 1e-3 + 1j * X
    V = 1j * 0.9 * f / 5e6
    return f, -Z, V


@pytest.fixture(scope="module")
def nb_table():
    f, Zs, V = _foster_table_as_stored()
    return mp.source_table(f, -Zs, V, convention="exp(+jwt)")


def test_wrong_sign_convention_is_refused():
    f, z, _ = _foster_table_as_stored()
    with pytest.raises(ValueError, match="Re Z < 0"):          # as stored: -Z
        mp.source_table(f, z, np.ones(f.size), convention="exp(+jwt)")
    with pytest.raises(ValueError, match="Re Z < 0"):          # conjugate is not the fix either
        mp.source_table(f, z, np.ones(f.size), convention="exp(-iwt)")
    with pytest.raises(ValueError, match="not inductive"):
        mp.source_table(f, -np.conj(z), np.ones(f.size), convention="exp(+jwt)")


def test_notebook_constants_are_exact_unit_conversions():
    assert mp.VOLT_TO_EV == pytest.approx(3.30227, rel=2e-6)
    assert mp.HENRY_TO_INV_EV == pytest.approx(4.03278e12, rel=2e-6)
    # 10.6049 A/m^2 per (GeV^-1 T) is g B sqrt(2 rho) at rho = 0.45 GeV/cm^3
    assert mp.J_eff_A_per_m2(1.0, 1.0, 0.45) == pytest.approx(10.6049, rel=1e-5)


def test_bands_and_greedy_allocation_on_a_synthetic_table(nb_table):
    """M2: every kept band lies inside a run of tunable points (C or L_t strictly inside its range) at least 5 MHz
    wide, shrunk by 2.5 MHz at both ends; capacitor tuning below the first pole and inductor tuning above it.  M6: the
    greedy allocation spends at most the budget, keeps chunks only inside the bands, and merge='notebook' and
    merge='adjacent' cover the same frequencies up to the rounding split of the former (about one chunk)."""
    f, X = nb_table.f_Hz, nb_table.X_ohm
    bands = mp.valid_intervals(f, X)
    el = mp.tuning_elements(f, X)
    ok = ((el["C_F"] > 30e-12) & (el["C_F"] < 300e-12)) | ((el["L_H"] > 10e-9) & (el["L_H"] < 500e-9))
    kinds = [b["kind"] for b in bands]
    assert kinds[0] == "capacitive" and "inductive" in kinds
    for b in bands:
        lo, hi = b["f_lo_Hz"], b["f_hi_Hz"]
        i0, i1 = np.searchsorted(f, lo - 2.5e6 - 1), np.searchsorted(f, hi + 2.5e6 + 1) - 1
        assert hi > lo and ok[i0:i1 + 1].all()                     # the unshrunk run is tunable ...
        assert (i0 == 0 or not ok[i0 - 1]) and (i1 == f.size - 1 or not ok[i1 + 1])   # ... and maximal
        assert (b["kind"] == "capacitive") == (X[np.searchsorted(f, lo)] > 0)
    assert bands[0]["f_hi_Hz"] < 90e6 < bands[1]["f_lo_Hz"]
    res = mp.multiband_scan(nb_table, Q=1e5, T_K=0.020, eta_A=20.0, B_T=5.0, Leff_rule="notebook",
                            notebook_units=True)
    budget = mp.YEAR_JULIAN_S
    covered = {}
    for merge in ("notebook", "adjacent"):
        g = mp.greedy_allocation(f, res["rate_Hz_per_s"], res["bands"], budget, merge=merge)
        assert 0 < g["time_used_s"] <= budget * (1 + 1e-12) and 0 < g["n_kept"] <= g["n_chunks"]
        iv = np.asarray(g["intervals_MHz"]) * 1e6
        assert all(any(b["f_lo_Hz"] - 1e4 <= a <= c <= b["f_hi_Hz"] + 1e4 for b in res["bands"]) for a, c in iv)
        covered[merge] = float(np.sum(iv[:, 1] - iv[:, 0]))
    assert covered["notebook"] == pytest.approx(covered["adjacent"], abs=2e5)


def test_tuned_slope_differs_from_notebook_only_for_inductor_tuning(nb_table):
    f = nb_table.f_Hz
    a, b = nb_table.L_eff(f, "notebook"), nb_table.L_eff(f, "tuned")
    cap = nb_table.X_ohm > 0
    assert np.allclose(a[cap], b[cap], rtol=1e-12)
    assert np.allclose(b[~cap] - a[~cap], np.abs(nb_table.X_ohm[~cap]) / (2 * np.pi * f[~cap]), rtol=1e-9)


def test_gbar_and_rate_equal_the_cpp_scan_core():
    f = np.array([1e6, 2e7, 1.5e8])
    for T, eta in ((0.02, 20.0), (0.01, 1.0)):
        cpp = [_core.G_bar_opt(1.0 / math.expm1(mp.H_PLANCK * x / (mp.KB * T)), eta) for x in f]
        assert np.allclose(mp.gbar_opt(f, T, eta), cpp, rtol=1e-12)
    # S5.2 with Phi_a = kappa Psi, V = omega Phi_a; same Q_a, Theta, rho on both sides
    psi, L, Q, T, eta = 3e-6, 7.7e-7, 1e6, 0.02, 20.0
    m = np.array([scan.freq_to_mass(x) for x in f])
    g = scan.g_qcd(m)
    V = mp.source_amplitude(2 * np.pi * f * psi, "kappa_SI", g, rho_GeV_cm3=0.3)
    mine = mp.scan_rate(f, V, L, Q, T, eta, Q_a=6.8e5, Theta=2.71)
    cpp = np.asarray(_core.scan_rate_curve([psi] * 3, [L] * 3, [Q] * 3, [T] * 3, [eta] * 3, list(m), list(g),
                                           6.8e5, 2.71, 0.3, True)) / (2 * np.pi)
    assert np.allclose(mine, cpp, rtol=1e-8)


def test_notebook_rate_is_low_by_two_pi_squared(nb_table):
    """Consistency check of the notebook_units switch; the evidence for the (2 pi)^2 is the C++ comparison."""
    p = np.array([45.0, 120.0])
    a = mp.notebook_dnu_dt(p, nb_table, Q=1e5, rounded_constants=False)
    b = mp.notebook_dnu_dt(p, nb_table, Q=1e5, rounded_constants=False, notebook_units=False)
    assert np.allclose(b / a, (2 * np.pi) ** 2, rtol=1e-12)


def _single_band_results(n=2001):
    f = np.geomspace(1e6, 1e7, n)
    L, psi = 1e-6, 1e-5
    R = 2e-4 * (f / 1e6) ** 1.5
    return {"name": "oneband", "coupling": {"Psi_SI_H_T_m2": psi, "L_p_H": L},
            "fullwave": {"sweep": [{"f_Hz": float(x), "Psi_re": psi, "Psi_im": 0.0, "Z_re_ohm": float(r),
                                    "Z_im_ohm": float(-2 * math.pi * x * L), "L_H": L} for x, r in zip(f, R)]}}


def test_single_band_reduces_to_reach_fullwave():
    """One band, target allocation: g = g_t (t_req/T)^(1/4) from the multi-band chain equals
    scan.reach_fullwave (S5.1-S5.8 with Q = omega L/(R + omega L_p/Q_other))."""
    res = _single_band_results()
    T, eta, Qo, yr = 0.03, 20.0, 1e6, 3.0
    fg = np.array([r["f_Hz"] for r in res["fullwave"]["sweep"]])
    f1, f2 = float(fg[100]), float(fg[1800])           # band edges on table points
    ref = scan.reach_fullwave(res, {"Q": Qo, "T": T, "eta_A": eta}, n=15, live_time_yr=yr,
                              compare_mqs=False, f_min=f1, f_max=f2)
    t = mp.table_from_results(res).restrict(f1, f2)
    f = t.f_Hz
    w = 2 * np.pi * f
    Le = t.L_eff(f)
    Qv = w * Le / (t.R_ohm + w * 1e-6 / Qo)
    gt = scan.g_qcd(np.array([scan.freq_to_mass(x) for x in f]))
    V = mp.source_amplitude(t.V, t.source_unit, gt, rho_GeV_cm3=0.3)
    rate = mp.scan_rate(f, V, Le, Qv, T, eta, Q_a=6.8e5, Theta=2.71)
    g, t_req = mp.target_allocation_reach(f, rate, gt, np.ones(f.size, bool), yr * mp.YEAR_JULIAN_S)
    assert t_req == pytest.approx(ref["band"]["t_req_s"], rel=1e-4)
    gi = np.exp(np.interp(np.log(ref["f_Hz"]), np.log(f), np.log(g)))
    assert np.allclose(gi[1:-1], ref["g95_GeV"][1:-1], rtol=1e-4)


def _mqs_table(f=np.arange(5e6, 60e6, 1e3), L=7.7e-7, V5=2.0, R=0.0):
    return mp.SourceTable(f, 2 * np.pi * f * L, np.full(f.size, R), V5 * f / 5e6)


def test_2211_single_pole_limits():
    t = _mqs_table()
    for eta in (1.0, 20.0):
        s = mp.single_pole_ratio(t, eta_A=eta, f_eval=np.array([1e7, 3e7]))
        assert np.allclose(s["g_over_gMQS"], 1.0, rtol=1e-9) and np.allclose(s["Q_B"], 1e6, rtol=1e-9)
    # lossless but dispersive: (V_MQS/V)(Q_B/1e6)^(1/4), independent of eta
    f = np.arange(5e6, 60e6, 1e3)
    tt = mp.SourceTable(f, 2 * np.pi * f * 7.7e-7 * (1 + (f / 4e7) ** 2), 0 * f, 2.0 * f / 5e6 * (1 - (f / 9e7) ** 2))
    a = mp.single_pole_ratio(tt, eta_A=1.0, f_eval=np.array([3e7]))
    b = mp.single_pole_ratio(tt, eta_A=20.0, f_eval=np.array([3e7]))
    assert a["g_over_gMQS"][0] == pytest.approx(b["g_over_gMQS"][0], rel=1e-12)
    assert a["g_over_gMQS"][0] == pytest.approx(1 / a["V_over_VMQS"][0] * (a["Q_B"][0] / 1e6) ** 0.25, rel=1e-12)
    # lossy: the paper's noise model depends on eta, the FDT one does not
    lt = _mqs_table(R=1e-2)
    p1, p20 = (mp.single_pole_ratio(lt, eta_A=e, f_eval=np.array([2e7]))["g_over_gMQS"][0] for e in (1.0, 20.0))
    d1, d20 = (mp.single_pole_ratio(lt, eta_A=e, noise="fdt", f_eval=np.array([2e7]))["g_over_gMQS"][0]
               for e in (1.0, 20.0))
    assert p1 < p20 and d1 == pytest.approx(d20, rel=1e-12) and p20 < d20


@pytest.mark.parametrize("eta", [1.0, 20.0])
def test_multipole_continuum_sum_on_an_mqs_table(eta):
    """On an MQS table every target is resonated once; the continuum sum of the per-tuning
    (paper) dwell gives Theta/Theta95 = pi/(4 sqrt(a)), a = eta hbar w/(2(2kT + eta hbar w))  (M13)."""
    t = _mqs_table()
    sch = mp.mqs_schedule(t, eta_A=eta, f_start_Hz=50e6, f_end_Hz=10e6)
    sch["g_target"] = mp.g_dfsz_notebook
    f = np.array([2e7, 3e7])
    m = mp.multipole_theta(t, sch, f, check_convergence=True)
    assert np.all(np.abs(m["convergence_rel_change"]) < 1e-4)
    w = 2 * np.pi * f
    EA = eta * mp.HBAR * w
    a = EA / (2 * (2 * mp.KB * 0.02 + EA))
    assert np.allclose(m["Theta_over_Theta95"], np.pi / (4 * np.sqrt(a)), rtol=2e-3)
    assert list(m["n_roots"]) == [1, 1]


def test_scan_wrappers_accept_axionfem_results():
    res = _single_band_results(201)
    out = scan.multipole_scan(res, Q_mqs=1e6, T_K=0.03, eta_A=20.0, B_T=None, C_range=(1e-12, 1e-6),
                              L_range=(1e-12, 1e-6), min_width_Hz=1e6, edge_margin_Hz=0.0)
    assert len(out["bands"]) == 1 and out["bands"][0]["kind"] == "capacitive"
    assert scan.reach_multiband is scan.multipole_scan
    r = scan.reach_2211(res, f_start_Hz=9e6, f_end_Hz=2e6, B_T=None)
    sp = r["single_pole"]
    w = 2 * np.pi * r["f_Hz"]
    R = 2e-4 * (r["f_Hz"] / 1e6) ** 1.5
    inner = (r["f_Hz"] > 2e6) & (r["f_Hz"] < 9e6)
    assert np.allclose(sp["Q_B"][inner], (w * 1e-6 / (R + w * 1e-6 / 1e6))[inner], rtol=1e-9)
    assert np.all(np.isfinite(r["g_single_GeV"][inner]))


def test_schedule_spends_no_time_on_inductive_tunings():
    f = np.arange(5e6, 60e6, 1e3)
    X = 2 * np.pi * f * 7.7e-7 * np.where((f > 30e6) & (f < 40e6), -1.0, 1.0)
    t = mp.SourceTable(f, X, 0 * f, 2.0 * f / 5e6)
    s = mp.mqs_schedule(t, f_start_Hz=50e6, f_end_Hz=10e6)
    gap = (s["f_grid_Hz"] > 31e6) & (s["f_grid_Hz"] < 39e6)
    assert np.all(s["t_grid_s"][gap] == 0) and np.all(s["t_grid_s"][~gap & (s["f_grid_Hz"] < 29e6)] > 0)


# ---------------------------------------------------------------- M14: the 2022 computation of 2211.00008
def _jf_reference():
    import importlib.util
    import sys
    p = ROOT / "benchmarks/multipole_scan/reference_jf2022/jf2022.py"
    spec = importlib.util.spec_from_file_location("jf2022_reference", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jf2022_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def _jf_compare(t, F_MHz, Zc, Voc, eta, f_MHz):
    """Port (mp.jf2022_*) vs the transcription of [C] (jf2022.py), fed the same strategy."""
    tr = _jf_reference()
    s = mp.jf2022_schedule(t, eta_A=eta)
    it = tr.Interp(F_MHz, Zc, Voc, scale=tr.coupling_scale(mp.JF_B_T, mp.JF_RHO))
    iq, _, _ = tr.q_bandwidth(it)
    m = tr.mqs_projection(it, eta, mp.JF_T_K)
    h = tr.hf_strategy(it, m, iq, eta, mp.JF_T_K)
    assert s["f_start_Hz"] / 1e6 == pytest.approx(m["start_freq"], rel=1e-5)
    s_same = dict(s, tunings_Hz=h["strategy"][0] * 1e6, dwell_s=h["strategy"][1])
    g_port = mp.jf2022_theta(t, s_same, np.asarray(f_MHz) * 1e6)["g_GeV"]
    ts = tr.tuning_ts_sum(it, np.asarray(f_MHz), h["strategy"][0], h["strategy"][1], eta, mp.JF_T_K)
    g_ref = (tr.TARGET_TS / ts) ** 0.25 * tr.g_x(tr.DFSZ, tr.freq_to_ev(np.asarray(f_MHz)))
    return s, g_port, g_ref


def test_jf2022_port_matches_notebook_transcription_synthetic():
    f = np.arange(5e6, 470e6 + 1, 50e3)
    w = 2 * np.pi * f
    d = 1 - (f / 60.0123e6) ** 2                    # one pole, one zero: capacitive gap 60-90 MHz
    X = w * 1e-6 * (1 - (f / 90.0071e6) ** 2) / d
    R = w * 1e-6 / 2e4 + 1e-6
    V = (f / 5e6) / np.abs(d)
    t = mp.SourceTable(f, X, R, V)
    s, gp, gr = _jf_compare(t, f / 1e6, R + 1j * X, V, 20.0, [7, 25, 50, 75, 120, 200, 400])
    assert 40e6 < s["f_start_Hz"] < 45e6
    assert np.all(s["dwell_s"] > 0) and s["dwell_s"].sum() <= mp.JF_LIVE_S * 1.01
    # [C]'s natural-unit constants agree with SI to ~1e-6
    np.testing.assert_allclose(gp, gr, rtol=2e-5)
    # amplifier-noise frequency weights matter off resonance (the decisive M13 -> M14 change); above the
    # tunings, where the off-resonant |Z|^2 term dominates, [C]'s (w/w_k)^2 weight makes it weaker here
    g13 = mp.jf2022_theta(t, s, np.array([200e6, 400e6]), amp_model="m13")["g_GeV"]
    g14 = mp.jf2022_theta(t, s, np.array([200e6, 400e6]))["g_GeV"]
    assert np.all(g14 > 2.0 * g13)


# the unpacked arXiv:2211.00008 COMSOL tables are not distributed: AXF_COMSOL_2211_DIR names their directory
EPOXY = (Path(os.environ["AXF_COMSOL_2211_DIR"]) if os.environ.get("AXF_COMSOL_2211_DIR")
         else Path("/nonexistent-axf-comsol-files")) / "Toroid_With_Epoxy"


@pytest.mark.skipif(not (EPOXY / "source_impedance.npy").exists(), reason="2022 COMSOL archive not available")
def test_jf2022_port_on_the_2211_epoxy_table():
    tr = _jf_reference()
    F, Zc, Voc = tr.load_tables(EPOXY)
    Vr = np.load(EPOXY / "source_voltage.npy")
    t = mp.source_table(F * 1e6, Zc, Vr[1] + 1j * Vr[2], convention="exp(+jwt)", port_load_ohm=50.0)
    s, gp, gr = _jf_compare(t, F, Zc, Voc, 20.0, [6.5, 22, 30, 78, 130, 220, 400])
    # B = 10 T, rho = 0.45 put the 1-yr endpoint on the paper's Fig. S3 single-pole edge (18.7386 MHz)
    assert s["f_start_Hz"] / 1e6 == pytest.approx(18.7386, abs=2e-3)
    np.testing.assert_allclose(gp, gr, rtol=2e-5)


def _synthetic_jf_table():
    f = np.arange(5e6, 470e6 + 1, 50e3)
    w = 2 * np.pi * f
    d = 1 - (f / 60.0123e6) ** 2
    X = w * 1e-6 * (1 - (f / 90.0071e6) ** 2) / d
    R = w * 1e-6 / 2e4 + 1e-6
    V = (f / 5e6) / np.abs(d)
    return f, X, R, V


def test_jf2022_fast_kernel_equals_verbatim():
    """The vectorised kernel equals the transcribed TuningTS (both amplifier weightings, both tuning
    branches)."""
    tr = _jf_reference()
    f, X, R, V = _synthetic_jf_table()
    it = tr.Interp(f / 1e6, R + 1j * X, V, scale=tr.coupling_scale(mp.JF_B_T, mp.JF_RHO))
    rng = np.random.default_rng(1)
    fs = np.sort(rng.uniform(6, 460, 40))
    rf = np.sort(rng.uniform(19, 150, 800))          # includes the 60-90 MHz capacitive-source (inductor) branch
    wt = rng.uniform(1, 1e4, rf.size)
    assert np.any(X[np.searchsorted(f, rf * 1e6)] < 0)
    for amp in ("jf2022", "axf"):
        np.testing.assert_allclose(tr.tuning_ts_sum_fast(it, fs, rf, wt, 20, 0.02, amp=amp),
                                   tr.tuning_ts_sum(it, fs, rf, wt, 20, 0.02, amp=amp), rtol=1e-12)


def test_jf2022_port_schedule_grid_and_curve():
    """The port's own strategy (step grid, dwell factor, mask) and sensitivity grid equal the transcription's,
    started from the same MQS allocation; capacitive-source tunings are dropped."""
    from scipy import ndimage
    tr = _jf_reference()
    f, X, R, V = _synthetic_jf_table()
    t = mp.SourceTable(f, X, R, V)
    s = mp.jf2022_schedule(t, eta_A=20.0)
    it = tr.Interp(f / 1e6, R + 1j * X, V, scale=tr.coupling_scale(mp.JF_B_T, mp.JF_RHO))
    iq, _, _ = tr.q_bandwidth(it)
    mqs = {"start_freq": s["f_start_Hz"] / 1e6, "freqs": s["mqs_f_Hz"] / 1e6, "collection_time": s["mqs_time_s"]}
    h = tr.hf_strategy(it, mqs, iq, 20.0, mp.JF_T_K)
    np.testing.assert_allclose(s["all_tunings_Hz"] / 1e6, h["hf_freqs"], rtol=1e-10)   # Hz vs MHz round-off
    np.testing.assert_allclose(s["tunings_Hz"] / 1e6, h["strategy"][0], rtol=1e-10)
    np.testing.assert_allclose(s["dwell_s"], h["strategy"][1], rtol=1e-9)
    assert s["all_tunings_Hz"].size > s["tunings_Hz"].size            # the 60-90 MHz gap was masked
    assert np.all(t.at(s["tunings_Hz"])["X_ohm"] >= 0)
    sg = mp.jf2022_sensitivity_grid(t, s)
    np.testing.assert_allclose(sg / 1e6, tr.sensitivity_freqs(iq), rtol=1e-10)
    g = np.linspace(1.0, 2.0, sg.size) ** 3
    x, y = mp.jf2022_plot_curve(sg, g)
    np.testing.assert_array_equal(x, sg[::100])
    np.testing.assert_allclose(y, ndimage.gaussian_filter(g, 101)[::100], rtol=1e-14)


@pytest.mark.skipif(not (EPOXY / "source_impedance.npy").exists(), reason="2022 COMSOL archive not available")
@pytest.mark.parametrize("eta,f_MHz,g_paper", [(1.0, 8.3213, 3.34667e-17), (1.0, 14.107, 2.18920e-17),
                                               (20.0, 18.1397, 5.00762e-17)])
def test_jf2022_port_multipole_vs_paper_below_endpoint(eta, f_MHz, g_paper):
    """External anchor for the per-tuning normalisation: below the 1-yr endpoint only off-resonant tails
    contribute, g is smooth (comb phase and smoothing irrelevant), and the port's own schedule must land on the
    digitized Fig. S3 "Multipole" curve."""
    Vr = np.load(EPOXY / "source_voltage.npy")
    Zr = np.load(EPOXY / "source_impedance.npy")
    t = mp.source_table(Zr[0] * 1e6, Zr[1] + 1j * Zr[2], Vr[1] + 1j * Vr[2], convention="exp(+jwt)",
                        port_load_ohm=50.0)
    s = mp.jf2022_schedule(t, eta_A=eta)
    g = mp.jf2022_theta(t, s, np.array([f_MHz * 1e6]))["g_GeV"][0]
    assert g == pytest.approx(g_paper, rel=0.05)     # a factor-4 error in Theta would be 41 % in g
