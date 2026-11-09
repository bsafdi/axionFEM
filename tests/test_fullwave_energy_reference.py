"""Independent small-matrix checks of the nondispersive Maxwell energy identity.

These validate algebra and signs, not FEM meshing or iterative solves.
"""
import numpy as np
import pytest


def _system():
    rng = np.random.default_rng(20261008)
    h = rng.normal(size=(7, 7))
    k = h.T @ h + 9 * np.eye(7)
    h = rng.normal(size=(7, 7))
    m = h.T @ h / 20 + np.eye(7)
    b = rng.normal(size=(7, 3))
    # A weak readout makes vector-norm-only comparisons insufficient.
    b[:, 2] *= 1e-4
    return k, m, b


def _response(w, k, m, b):
    e = np.linalg.solve(k - w**2 * m, 1j * w * b)
    z = -b.T @ e
    g = (e.conj().T @ m @ e + e.conj().T @ k @ e / w**2) / 4
    return e, z, g


def test_fixed_geometry_reactance_slope_is_total_energy_cross_matrix():
    k, m, b = _system()
    w, h = 0.8, 1e-5
    e, z, g = _response(w, k, m, b)
    derivative = (_response(w + h, k, m, b)[1].imag
                  - _response(w - h, k, m, b)[1].imag) / (2 * h)
    np.testing.assert_allclose(derivative, -4 * g.real, rtol=2e-9, atol=1e-17)
    np.testing.assert_allclose(g, g.conj().T, rtol=1e-13, atol=1e-20)
    assert np.linalg.eigvalsh(g).min() > 0
    currents = np.array([1 + 2j, -0.7 + 0.3j, 3 - 4j])
    field = e @ currents
    integrated = np.real(field.conj() @ (m + k / w**2) @ field) / 4
    assert np.real(currents.conj() @ g @ currents) == pytest.approx(integrated, rel=1e-13)


@pytest.mark.parametrize("kind", ["capacitor", "inductor", "capacitor_esl"])
def test_reactive_operator_closure_and_loaded_field_include_weak_readout(kind):
    k, m, b = _system()
    w = 0.8
    e, z, _ = _response(w, k, m, b)
    c = np.array([0.2, 0.3, 0.15])
    l = np.array([0.4, 0.6, 0.3])
    if kind == "capacitor":
        y = -1j * w * c
        update = -w**2 * b @ np.diag(c) @ b.T
    elif kind == "inductor":
        y = 1j / (w * l)
        update = b @ np.diag(1 / l) @ b.T
    else:
        y = 1 / (0.1 + 1j / (w * c) - 1j * w * l)
        update = -1j * w * b @ np.diag(y) @ b.T
    u = np.array([1, 0, 0])
    direct = np.linalg.solve(k - w**2 * m + update, 1j * w * b @ u)
    voltage = np.linalg.solve(np.eye(3) + z @ np.diag(y), z @ u)
    np.testing.assert_allclose(-b.T @ direct, voltage, rtol=2e-13, atol=1e-20)
    np.testing.assert_allclose(direct, e @ (u - y * voltage), rtol=2e-13, atol=1e-16)


def test_weak_readout_error_is_visible_componentwise():
    predicted = np.array([1.0, 0.5, 1e-8])
    direct = predicted.copy()
    direct[2] *= 1.01
    assert np.linalg.norm(direct-predicted) / np.linalg.norm(predicted) < 1e-9
    allowance = 1e-5*np.abs(predicted) + 1e-12*np.max(np.abs(predicted))
    assert np.max(np.abs(direct-predicted) / allowance) > 1


def test_inexact_source_reciprocity_defect_matches_transpose_residuals():
    """Diagnose inexact fields without accepting their inaccurate voltages."""
    k, m, b = _system()
    w = 0.8
    a = k - w**2*m + 0.03j*m  # symmetric complex, not Hermitian
    q = np.arange(1.0, 8.0)
    rng = np.random.default_rng(812)
    e = np.linalg.solve(a, 1j*w*b)
    s = np.linalg.solve(a, 1j*w*q)
    e += 1e-3*(rng.normal(size=e.shape) + 1j*rng.normal(size=e.shape))
    s += 2e-3*(rng.normal(size=s.shape) + 1j*rng.normal(size=s.shape))
    rp, rs = a@e - 1j*w*b, a@s - 1j*w*q
    direct, reciprocal = -b.T@s, -q.T@e
    predicted_defect = (s.T@rp - e.T@rs)/(1j*w)
    np.testing.assert_allclose(direct-reciprocal, predicted_defect, rtol=2e-12, atol=3e-14)
    # Conjugation would silently invalidate Lorentz reciprocity for this case.
    wrong = (s.conj().T@rp - e.conj().T@rs)/(1j*w)
    assert np.linalg.norm(wrong - predicted_defect) > 1e-3
    allowance = 1e-5*np.abs(reciprocal) + 1e-12*np.max(np.abs(reciprocal))
    assert np.max(np.abs(direct-reciprocal)/allowance) > 1
