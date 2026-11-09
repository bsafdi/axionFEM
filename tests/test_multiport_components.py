"""Analytic identities of the multiport circuit components (symmetry reduction, network algebra); these are not
detector FEM gates."""
import numpy as np
import pytest

from axionfem.circuit import HBAR
from axionfem.multiport import MultiportCircuit, MultiportReadout
from axionfem.multiport_components import (
    QuarterSymmetry, LosslessTransmissionLine, calibrate_modal_loss,
    fixed_squid, series_capacitor_impedance, series_capacitor_energy,
    fit_series_capacitor, eliminate_component, omitted_overlap_bound,
)


def test_mixed_module_readout_full_quarter_signal_energy_and_squid_noise():
    symmetry = QuarterSymmetry(1)
    w = 2*np.pi*2e5
    resistance = np.array([[0.03, .004, .001], [.004, .03, .001], [.001, .001, .02]])
    inductance = 1e-6*np.array([[2, .6, .3], [.6, 2, .3], [.3, .3, .8]])
    z = resistance-1j*w*inductance
    psi = np.array([2+3j, 2+3j, .2-.1j])*1e-4
    energy = inductance/4
    ze, pe, ge = symmetry.physical_full_to_reduced(z, psi, energy)
    zq, pq, gq = ze, pe/2, ge
    recovered = symmetry.quarter_to_physical(zq, pq, gq)
    for got, expected in zip(recovered, (ze, pe, ge)):
        np.testing.assert_allclose(got, expected, rtol=1e-15)

    full = MultiportCircuit()
    full.multiport("field", [("m+", "gnd", "a"), ("m-", "gnd", "b"),
                             ("r", "gnd", "c")], z, T_K=.03)
    full.capacitor("c+", "a", "gnd", 8e-8)
    full.capacitor("c-", "b", "gnd", 8e-8)
    full.inductor("L_in", "c", "gnd", 1e-7)
    reduced = MultiportCircuit()
    reduced.multiport("field", [("m", "gnd", "a"), ("r", "gnd", "c")], ze, T_K=.03)
    reduced.capacitor("pair", "a", "gnd", 16e-8)
    reduced.inductor("L_in", "c", "gnd", 1e-7)
    sources = [dict(zip(("m+", "m-", "r"), -1j*w*psi)),
               dict(zip(("m", "r"), -1j*w*pe))]
    rf = MultiportReadout(full, fixed_squid(), sources[0])
    rq = MultiportReadout(reduced, fixed_squid(), sources[1])
    assert rq.flux_per_kappa(w) == pytest.approx(rf.flux_per_kappa(w), rel=1e-13)
    for key in ("S_Phi_thermal_Wb2_s", "S_Phi_backaction_Wb2_s",
                "S_Phi_imprecision_Wb2_s", "S_Phi_total_Wb2_s"):
        assert rq.noise_psd(w)[key] == pytest.approx(rf.noise_psd(w)[key], rel=2e-13, abs=1e-70)
    sf, se = full.solve(w, sources[0]), reduced.solve(w, sources[1])
    i_full = np.array([sf.I(x) for x in ("m+", "m-", "r")])
    i_eff = np.array([se.I(x) for x in ("m", "r")])
    np.testing.assert_allclose(i_full, symmetry.current_lift@i_eff, rtol=2e-13)
    assert np.vdot(i_full, energy@i_full) == pytest.approx(np.vdot(i_eff, ge@i_eff), rel=2e-13)
    assert np.vdot(i_full, energy@i_full) == pytest.approx(4*np.vdot(i_eff/2, gq@(i_eff/2)), rel=2e-13)
    # Thermal image errors would multiply the first value by four; reject that.
    assert abs(rq.noise_psd(w)["S_Phi_thermal_Wb2_s"] / rf.noise_psd(w)["S_Phi_thermal_Wb2_s"] - 4) > 2


@pytest.mark.parametrize("frequency", [1.209e5, 1e6, 3e6, 31e6])
def test_line_open_short_matched_and_generic_stamp(frequency):
    line = LosslessTransmissionLine(.30, 83.18e-9, 12.04e-12)
    w = 2*np.pi*frequency
    z, z0, a = line.impedance(w), line.impedance_ohm, w*line.delay_s
    assert line.loaded_impedance(w, np.inf) == pytest.approx(1j*z0/np.tan(a), rel=1e-14)
    assert line.loaded_impedance(w, 0) == pytest.approx(-1j*z0*np.tan(a), rel=1e-14)
    assert line.loaded_impedance(w, z0) == pytest.approx(z0, rel=1e-14)
    np.testing.assert_array_equal(z, z.T)
    for load in [0, z0, 0.03-1j*w*100e-9]:
        circuit = MultiportCircuit()
        circuit.source("drive", "gnd", "a", 1)
        circuit.multiport("line", [("near", "a", "gnd"), ("far", "b", "gnd")], line.impedance)
        circuit.impedance("load", "b", "gnd", load, noise=False)
        solved = circuit.solve(w)
        assert solved.V("a")/solved.I("near") == pytest.approx(line.loaded_impedance(w, load), rel=2e-9, abs=1e-11)


def test_line_energy_equals_distributed_integral_and_foster_slope():
    line = LosslessTransmissionLine(.779497, 83.18e-9*.779497/.3, 12.04e-12*.779497/.3)
    w = 2*np.pi*1e6
    currents = np.array([1+.2j, -.997-.21j])
    x, weights = np.polynomial.legendre.leggauss(32)
    x, weights = (x+1)*line.length_m/2, weights*line.length_m/2
    voltage, current = line.fields(w, currents, x)
    integrated = np.dot(weights, (line.inductance_H*abs(current)**2+
                                 line.capacitance_F*abs(voltage)**2)/line.length_m)/4
    stored = np.vdot(currents, line.energy_matrix(w)@currents).real
    assert stored == pytest.approx(integrated, rel=1e-11)
    dw = w*1e-4
    slope = (line.impedance(w+dw).imag-line.impedance(w-dw).imag)/(2*dw)
    np.testing.assert_allclose(-slope/4, line.energy_matrix(w), rtol=2e-8)
    ends = line.fields(w, currents, [0, line.length_m])[1]
    np.testing.assert_allclose(ends, [currents[0], -currents[1]], rtol=1e-12)


def test_fixed_mode_q_includes_line_capacitor_and_input_energy_and_linewidth():
    line = LosslessTransmissionLine(.779497, 83.18e-9*.779497/.3, 12.04e-12*.779497/.3)
    w0, lp, lin, Q = 2*np.pi*1.209e5, 1.1e-6, 100e-9, 1e6
    zline = line.impedance(w0)
    i_far = -zline[1, 0]/(zline[1, 1]-1j*w0*lin)
    capacitance = -1/(w0*((-1j*w0*lp)+line.loaded_impedance(w0, -1j*w0*lin)).imag)
    line_energy = np.vdot([1, i_far], line.energy_matrix(w0)@np.array([1, i_far])).real
    extra = series_capacitor_energy(w0, 1, capacitance) + lin*abs(i_far)**2/4 + line_energy
    calibration = calibrate_modal_loss(w0, [1], [[lp/4]], extra_energy_J=extra, Q=Q)
    resistance = calibration.resistance_ohm[0, 0].real
    assert w0*calibration.total_energy_J/calibration.dissipated_power_W == pytest.approx(Q)
    assert resistance > w0*lp/Q*1.25

    def reactance(w):
        return (-1j*w*lp+1j/(w*capacitance)+line.loaded_impedance(w, -1j*w*lin)).imag

    assert abs(reactance(w0)) < 1e-12
    # Half-power frequencies of a fixed capacitor and fixed calibrated R.
    def bisect(target, lo, hi):
        for _ in range(55):
            mid = (lo+hi)/2
            if reactance(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    left = bisect(resistance, w0*(1-2/Q), w0)
    right = bisect(-resistance, w0, w0*(1+2/Q))
    assert w0/(right-left) == pytest.approx(Q, rel=3e-7)


def test_fixed_squid_has_positive_psds_and_declared_eta():
    squid = fixed_squid()
    assert min(squid.S_Phi, squid.S_J) > 0
    assert 2*np.sqrt(squid.S_Phi*squid.S_J)/HBAR == pytest.approx(10)
    for bad in (0, -1, np.nan):
        with pytest.raises(ValueError):
            fixed_squid(flux_noise_one_sided_Phi0=bad)


def test_fem_energy_quadratic_tolerates_recorded_assembly_roundoff_not_wrong_energy():
    g = np.array([[2., .3], [.3+1e-14, 1.]])*1e-7
    result = calibrate_modal_loss(1e6, [1, 1j], g)
    assert 0 < result.energy_hermiticity_relative_error < 1e-12
    assert result.total_energy_J == pytest.approx(3e-7)
    g[1, 0] += 1e-9
    with pytest.raises(ValueError, match="Hermiticity"):
        calibrate_modal_loss(1e6, [1, 1j], g)


@pytest.mark.parametrize("capacitance", [760.76e-12, 91.29e-9, 182.58e-9])
def test_component_c_esl_extraction_and_direct_loaded_operator(capacitance):
    w = 2*np.pi*np.array([1e3, 1.209e5, 1e6, 2.5e6])
    esl = 20e-9
    z = np.array([series_capacitor_impedance(x, capacitance, esl) for x in w])
    fit = fit_series_capacitor(w, z)
    assert fit["C_F"] == pytest.approx(capacitance, rel=1e-13)
    assert fit["ESL_H"] == pytest.approx(esl, rel=1e-9)
    assert fit["relative_residual"] < 2e-14
    # Algebraic Maxwell closure for reactive admittance, independently direct.
    a = np.array([[3., .4], [.4, 2.]]) - w[2]**2*np.diag([1e-15, 2e-15])
    b = 1e-3*np.array([[1.], [.3]])
    y = 1/z[2]
    source = np.array([1+2j, -.2+.1j])
    direct = np.linalg.solve(a-1j*w[2]*y*b@b.T, source)
    empty = np.linalg.solve(a, source)
    response = np.linalg.solve(a, 1j*w[2]*b)
    port_z = -(b.T@response)[0, 0]
    port_v = -(b.T@empty)[0]
    current = -y*port_v/(1+y*port_z)
    hybrid = empty + response[:, 0]*current
    np.testing.assert_allclose(hybrid, direct, rtol=3e-12, atol=1e-12)


def test_component_schur_elimination_retains_coherent_source():
    a = np.array([[6, 1, .7], [1, 5, -.4], [.7, -.4, 2]], dtype=complex)
    q = np.array([1+2j, -.4j, 3-.2j])
    reduced_a, reduced_q = eliminate_component(a[:2, :2], a[:2, 2:], a[2:, 2:], q[:2], q[2:])
    np.testing.assert_allclose(np.linalg.solve(reduced_a, reduced_q), np.linalg.solve(a, q)[:2], rtol=1e-14)
    assert np.linalg.norm(np.linalg.solve(reduced_a, q[:2])-np.linalg.solve(a, q)[:2]) > .1
    assert omitted_overlap_bound(2, 3, 4) == pytest.approx(np.sqrt(6))


@pytest.mark.parametrize("emf_component", [1+2j, -.3+.4j, 0j])
def test_component_source_changes_sign_when_stamped_as_passive_load(emf_component):
    # Two physical FEM one-ports share a voltage; incoming currents add to0.
    # v=Zg*Ig+Vag=Zc*Ic+Vac gives Ic=(Vag-Vac)/(Zg+Zc).
    wg, wc, emf_global = 1-2j, .3+3j, 1+2j
    natural = MultiportCircuit()
    natural.multiport("global", [("g", "gnd", "node")], [[wg]])
    natural.multiport("component", [("c", "gnd", "node")], [[wc]])
    a = natural.solve(1e6, {"g": emf_global, "c": emf_component})
    passive = MultiportCircuit()
    passive.multiport("global", [("g", "gnd", "node")], [[wg]])
    passive.impedance("c", "node", "gnd", wc)
    b = passive.solve(1e6, {"g": emf_global, "c": -emf_component})
    expected = (emf_global-emf_component)/(wg+wc)
    assert b.I("c") == pytest.approx(expected, rel=1e-13, abs=1e-15)
    assert b.I("c") == pytest.approx(-a.I("c"), rel=1e-13, abs=1e-15)
    assert b.V("node") == pytest.approx(a.V("node"), rel=1e-13)


def test_output_snr_integrates_through_exact_signal_zero_without_noise_referral():
    circuit = MultiportCircuit()
    circuit.multiport("field", [("p", "gnd", "a")], [[1]])
    circuit.inductor("L_in", "a", "gnd", 1e-7)
    center = 1e6
    readout = MultiportReadout(circuit, fixed_squid(), lambda w: {"p": 1j*(w-center)})
    w = np.array([center-1, center, center+1])
    weights = np.array([.5, 1, .5])
    line = np.ones(3)/2
    result = readout.output_snr_rate(w, weights, kappa_a=1e-20,
                                   axion_omega=center, line_shape_per_rad_s=line)
    assert result["zero_signal_samples"] == 1
    assert result["theta_per_s"] > 0
    assert result["line_normalization"] == pytest.approx(1)
    manual = 0.0
    for wi, weight in zip(w, weights):
        signal = np.pi/2*1e-40*(wi/center)**2*abs(readout.flux_per_kappa(wi))**2/2
        manual += weight*(signal/readout.noise_psd(wi)["S_Phi_total_Wb2_s"])**2/(2*np.pi)
    assert result["theta_per_s"] == pytest.approx(manual, rel=1e-14, abs=0)
    readout.emf_per_kappa = {"p": 0j}
    assert readout.output_snr_rate(w, weights, kappa_a=1e-20,
                                  axion_omega=center, line_shape_per_rad_s=line)["theta_per_s"] == 0
