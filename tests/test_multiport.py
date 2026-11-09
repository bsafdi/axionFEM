"""Small analytic circuit references; these are not FEM geometry validation."""
import math

import numpy as np
import pytest

from axionfem import circuit as ck
from axionfem.multiport import LossBath, MultiportCircuit, MultiportReadout


def _twoport(z, loads, *, T=0.03, baths=None):
    c = MultiportCircuit()
    c.multiport("field", [("p0", "gnd", "a"), ("p1", "gnd", "b")], z, T_K=T, baths=baths)
    for j, node in enumerate(("a", "b")):
        c.impedance(f"load{j}", node, "gnd", loads[j], T_K=T)
    return c


@pytest.mark.parametrize("frequency", [1e3, 1.209e5, 1e6])
def test_twoport_closure_matches_two_loop_kirchhoff_and_power(frequency):
    w = 2 * np.pi * frequency
    inductance = np.array([[2e-6, 0.4e-6], [0.4e-6, 1e-6]])
    resistance = np.array([[0.4, -0.12], [-0.12, 0.3]])
    z = resistance - 1j * w * inductance
    loads = np.array([0.1 + 1j / (w * 2e-8), 0.25 - 1j * w * 0.2e-6])
    source = np.array([1 + 2j, -0.7 + 0.3j])
    c = _twoport(z, loads)
    sol = c.solve(w, dict(zip(("p0", "p1"), source)))
    current = np.array([sol.I("p0"), sol.I("p1")])
    expected = np.linalg.solve(z + np.diag(loads), source)
    np.testing.assert_allclose(current, expected, rtol=1e-12, atol=1e-15)
    delivered = 0.5 * np.real(np.vdot(current, source))
    dissipated = 0.5 * np.real(np.vdot(current, (resistance + np.diag(loads.real)) @ current))
    assert delivered == pytest.approx(dissipated, rel=2e-12)
    assert c.transfer_reciprocity(w) < 1e-12
    assert c.nyquist_check(w, "p0") < 2e-12


def test_correlated_fdt_matches_independent_resistor_mixture():
    w, T = 2 * np.pi * 1e5, 0.03
    # Two independent physical resistors mapped through an ideal real network:
    # R = U diag(r) U.T. Its noise can be computed in either basis.
    u = np.array([[1.0, -0.4], [0.3, 1.2]])
    r = np.array([0.2, 0.7])
    resistance = u @ np.diag(r) @ u.T
    z = resistance - 1j * w * np.array([[2e-6, 0.7e-6], [0.7e-6, 1e-6]])
    c = _twoport(z, [1j * 0.3, -1j * 0.2], T=T)
    h = np.linalg.inv(z + np.diag([1j * 0.3, -1j * 0.2]))[1]
    factor = ck.HBAR * w * (2 * ck.n_bose(w, T) + 1)
    expected = factor * np.sum(r * abs(h @ u) ** 2)
    measured = c.thermal_noise(w, lambda s: s.I("p1"))["total"]
    assert measured == pytest.approx(expected, rel=2e-13)
    diagonal_only = factor * np.sum(np.diag(resistance) * abs(h) ** 2)
    assert abs(diagonal_only / expected - 1) > 0.01


def test_separate_temperature_matrices_not_an_averaged_bath():
    w = 2 * np.pi * 2e5
    cold = 0.4 * np.outer([1, 0.3], [1, 0.3])
    warm = 0.01 * np.outer([0.2, 1], [0.2, 1])
    z = cold + warm - 1j * np.array([[2, 0.4], [0.4, 1]])
    c = _twoport(z, [1j, 2j], baths=[LossBath("cold", cold, 0.03), LossBath("warm", warm, 4.0)])
    h = np.linalg.inv(z + np.diag([1j, 2j]))[0]
    covariance = ck.HBAR * w * ((2 * ck.n_bose(w, .03) + 1) * cold
                              + (2 * ck.n_bose(w, 4) + 1) * warm)
    noise = c.thermal_noise(w, lambda s: s.I("p0"))
    assert noise["total"] == pytest.approx(float(np.real(h @ covariance @ h.conj())), rel=1e-12)
    assert noise["by_branch"]["field/warm"] > 0
    assert c.nyquist_check(w, "p1") < 1e-12


@pytest.mark.parametrize("ratio", [0.7, 0.999, 1, 1.001, 1.8])
def test_single_port_reduction_matches_existing_squid_readout(ratio):
    f0, lp, lin, q, psi = 1.209e5, 1.1e-6, 0.2e-6, 1e6, 1e-4
    old = ck.series_readout(lp, Psi_SI=psi, f_Hz=f0, Q=q, T_K=.03,
                            L_in_H=lin, M_H=1e-9, eta_A=10)
    c = MultiportCircuit()
    w0 = 2 * np.pi * f0
    r = w0 * lp / q
    c.multiport("field", [("pickup", "gnd", "a")], lambda w: [[r - 1j*w*lp]], T_K=.03)
    c.capacitor("C", "a", "b", old.circuit._get("C").value)
    c.inductor("L_in", "b", "gnd", lin)
    new = MultiportReadout(c, old.squid, lambda w: {"pickup": -1j * w * psi})
    w = w0 * ratio
    assert new.signal(w) == pytest.approx(old.signal(w)["Phi_SQUID_Wb"], rel=1e-12)
    assert new.reference_noise(w, psi) == pytest.approx(old.noise_psd(w)["N_V_V2_s"], rel=1e-12)
    assert new.noise_psd(w)["S_Phi_total_Wb2_s"] == pytest.approx(
        old.noise_psd(w)["S_Phi_total_Wb2_s"], rel=1e-12)


def test_coherent_sources_cancel_and_port_orientation_is_covariant():
    w = 1e6
    z = np.array([[1-2j, .2-.5j], [.2-.5j, 2-3j]])
    loads = np.array([.3+2j, .2+1j])
    c = _twoport(z, loads)
    transfer = np.linalg.inv(z + np.diag(loads))[1]
    source = np.array([1j, -1j*transfer[0]/transfer[1]])
    sq = ck.Squid(M_H=1e-9, input_coil="load1", S_Phi=1e-30, S_J=1e-30)
    read = MultiportReadout(c, sq, dict(zip(("p0", "p1"), source)))
    assert abs(read.flux_per_kappa(w)) < 1e-24
    read.emf_per_kappa = dict(zip(("p0", "p1"), abs(source)))
    assert abs(read.flux_per_kappa(w)) > 1e-12
    # Reverse only the first FEM port's reference direction (and its terminals).
    d = np.diag([-1, 1])
    flipped = MultiportCircuit()
    flipped.multiport("field", [("p0", "a", "gnd"), ("p1", "gnd", "b")], d @ z @ d)
    flipped.impedance("load0", "a", "gnd", loads[0])
    flipped.impedance("load1", "b", "gnd", loads[1])
    src = np.array([1+2j, .5-.3j])
    a = c.solve(w, dict(zip(("p0", "p1"), src)))
    b = flipped.solve(w, dict(zip(("p0", "p1"), d @ src)))
    assert a.I("load1") == pytest.approx(b.I("load1"), rel=1e-13)


@pytest.mark.parametrize("z,match", [
    ([[1, 2], [0, 1]], "reciprocal"),
    ([[1, 2], [2, 1]], "negative dissipated"),
    ([[1, np.nan], [np.nan, 1]], "finite"),
    ([[1]], "2 by 2"),
])
def test_unphysical_or_wrong_matrix_is_rejected(z, match):
    with pytest.raises(ValueError, match=match):
        _twoport(z, [1, 1]).solve(1e6)


def test_incomplete_loss_budget_is_rejected():
    with pytest.raises(ValueError, match="sum"):
        _twoport(np.eye(2), [1, 1], baths=[LossBath("cold", .5*np.eye(2), .03)]).solve(1e6)


def test_unconnected_port_remains_open_in_mna():
    c = MultiportCircuit()
    z = np.array([[1-2j, .3-.4j], [.3-.4j, 2-3j]])
    c.multiport("field", [("p0", "gnd", "a"), ("p1", "gnd", "b")], z)
    c.resistor("load", "a", "gnd", .2)
    solution = c.solve(1e6, {"p0": 1, "p1": .5j})
    assert abs(solution.I("p1")) < 1e-15
    assert solution.I("p0") == pytest.approx(1 / (z[0, 0]+.2), rel=1e-13)


def test_lossless_multiport_still_has_squid_noise():
    c = _twoport(-1j * np.array([[2, .4], [.4, 1]]), [1j, 2j])
    squid = ck.Squid(M_H=1e-9, input_coil="load1", S_Phi=1e-30, S_J=1e-28)
    readout = MultiportReadout(c, squid, {"p0": 1j})
    noise = readout.noise_psd(1e6)
    assert noise["S_Phi_thermal_Wb2_s"] == 0
    assert noise["S_Phi_backaction_Wb2_s"] > 0
    assert noise["S_Phi_total_Wb2_s"] >= squid.S_Phi


def test_negative_loss_near_null_mode_is_not_treated_as_roundoff():
    # A relative 1e-12 negative eigenvalue must not be tolerated against a unit
    # loss elsewhere: it gives a macroscopic negative PSD at resonance.
    c = _twoport(np.diag([-1e-12-1j, 1-1j]), [1j, 2j])
    with pytest.raises(ValueError, match="negative dissipated"):
        c.thermal_noise(1e6, lambda s: s.I("p0"))


def test_rank_one_loss_dark_combination_has_nonnegative_noise():
    r = np.outer([.3, .7], [.3, .7])
    c = _twoport(r - 1j*np.eye(2), [2j, 2j])
    noise = c.thermal_noise(1e6, lambda s: .7*s.I("p0")-.3*s.I("p1"))
    assert noise["total"] >= 0
    assert noise["total"] < 1e-50


def test_exact_signal_zero_is_rejected_by_finite_noise_referral():
    c = _twoport(np.eye(2), [1, 1])
    sq = ck.Squid(M_H=1e-9, input_coil="load1", S_Phi=1e-30)
    read = MultiportReadout(c, sq, {"p0": 1j})
    assert read.flux_per_kappa(1e6) == 0
    assert read.inverse_reference_noise(1e6, 1e-4) == 0
    with pytest.raises(ValueError, match="signal zero"):
        read.reference_noise(1e6, 1e-4)


def test_small_dissipation_cannot_be_omitted_from_bath_budget():
    c = MultiportCircuit()
    c.multiport("field", [("p", "gnd", "a")], [[1e-14-1j]], baths=[])
    c.impedance("load", "a", "gnd", 1j)
    with pytest.raises(ValueError, match="bath resistances must sum"):
        c.thermal_noise(1e6, lambda s: s.I("p"))


def test_small_loss_eigenchannel_cannot_be_hidden_by_large_channel():
    c = _twoport(np.diag([1e-12-1j, 1-1j]), [1j, 2j],
                 baths=[LossBath("incomplete", np.diag([0, 1]), .03)])
    with pytest.raises(ValueError, match="bath resistances must sum"):
        c.thermal_noise(1e6, lambda s: s.I("p0"))
