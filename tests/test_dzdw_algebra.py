"""The exact dZ/dw identity of model3d_fullwave.cpp on a small dense model.

A(w) = K - w^2 eps_d (1 + i tan delta) M_d - w^2 eps0 M_v - i w sigma M_s + U diag(c(w)) U^T, complex symmetric,
with essential DOFs eliminated symmetrically and the exterior coefficients c_TE = (gamma_n - n)/(mu0 R1),
c_TM = -w^2 eps0 R1 / gamma_n as the operator builds them. The coded formula
    dZ_jk/dw = Z_kj / w - (i / w) e_j^T A'(w) e_k,  e = x / I,  A x_k = i w J_k,  Z_jk = -J_j^T x_k / I^2
must match a Richardson difference of Z(w), with the exterior derivative from the recurrence-ratio form of
d gamma_n/dx (axi::hankel_log_derivative_dx).
"""
import numpy as np

MU0 = 1.25663706212e-6
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 * C0)
R1 = 0.7
MODES = [(1, True), (1, False), (2, True), (2, False), (3, True)]


def gamma_and_dx(x: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.zeros(N + 1, complex)
    d = np.zeros(N + 1, complex)
    g[0] = -1j * x
    d[0] = -1j
    r = 1 / x - 1j
    for n in range(1, N + 1):
        g[n] = n - x / r
        d[n] = x - (g[n] + n + 1) / r
        r = (2 * n + 1) / x - 1 / r
    return g, d


def coefficients(w: float) -> tuple[np.ndarray, np.ndarray]:
    x = w * R1 / C0
    g, dg = gamma_and_dx(x, max(n for n, _ in MODES))
    c = np.array([(g[n] - n) / (MU0 * R1) if te else -w * w * EPS0 * R1 / g[n] for n, te in MODES])
    dc = np.array([dg[n] / (MU0 * C0) if te else -EPS0 * R1 * w * (2 * g[n] - x * dg[n]) / g[n] ** 2
                   for n, te in MODES])
    return c, dc


def test_exact_dzdw_identity() -> None:
    rng = np.random.default_rng(7)
    n = 40
    ess = np.arange(5)

    def spd() -> np.ndarray:
        h = rng.normal(size=(n, n))
        return h.T @ h + np.eye(n)

    K, Md, Mv, Ms = spd() * 1e6, spd(), spd(), spd()
    eps_d, tan, sig = 3 * EPS0, 0.05, 2e-3
    U = rng.normal(size=(n, len(MODES))) * 3.0
    U[ess] = 0.0
    J = rng.normal(size=(n, 3))
    J[ess] = 0.0
    current = 2.0

    def operator(w: float) -> np.ndarray:
        c, _ = coefficients(w)
        A = K - w * w * eps_d * (1 + 1j * tan) * Md - w * w * EPS0 * Mv - 1j * w * sig * Ms + U @ np.diag(c) @ U.T
        A[ess, :] = 0.0
        A[:, ess] = 0.0
        A[ess, ess] = 1.0
        return A

    def impedance(w: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.linalg.solve(operator(w), 1j * w * J * current)
        return -(J * current).T @ x / current ** 2, x / current

    w = 2 * np.pi * 1.5e8
    Z, e = impedance(w)
    _, dc = coefficients(w)
    dA = -2 * w * eps_d * (1 + 1j * tan) * Md - 2 * w * EPS0 * Mv - 1j * sig * Ms + U @ np.diag(dc) @ U.T
    exact = Z.T / w - 1j / w * (e.T @ dA @ e)
    h = 1e-3 * w
    d1 = (impedance(w + h)[0] - impedance(w - h)[0]) / (2 * h)
    d2 = (impedance(w + h / 2)[0] - impedance(w - h / 2)[0]) / h
    fd = (4 * d2 - d1) / 3
    assert np.max(np.abs(exact - fd)) / np.max(np.abs(exact)) < 1e-8
    assert np.max(np.abs(Z - Z.T)) / np.max(np.abs(Z)) < 1e-12
