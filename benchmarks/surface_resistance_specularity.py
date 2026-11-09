"""Reuter-Sondheimer anomalous skin effect for a general specularity p (0 <= p <= 1): numerical solution of the
half-space integral equation, used (i) to check the closed-form p = 0 and p = 1 integrals implemented in
cpp/src/surface_resistance.cpp and (ii) to generate the interpolation weight table w(alpha, p) used there for
0 < p < 1.  Docs: docs/physics/surface_resistance.md (Sec. 2.4).

Equation (Reuter & Sondheimer 1948; quasi-static, isotropic Fermi surface, omega tau << 1), lengths in units of the
mean free path l (s = z / l), beta = mu0 omega l^2 / rho = 4 alpha / 3:

    E''(s) = i beta int_0^inf [k(s - s') + p k(s + s')] E(s') ds',    k(x) = 3/4 (E_1(|x|) - E_3(|x|)),

(int k = 1: the local limit is E'' = i beta E), and Z = -i omega mu0 l E(0) / E'(0).  The image term p k(s + s')
is the specular part of the boundary condition.  Discretization: piecewise-linear E on a geometrically graded mesh,
exact hat-function moments of k through its second antiderivative, three-point E'' at the nodes, E'(0) from
E'(0) = -int_0^inf E'' ds.  Accuracy at p = 0 and 1 against the closed forms: ~1e-5 (printed).

Output: w(alpha, p) = (R(p) - R(1)) / (R(0) - R(1)) on a grid in log10(alpha) and p, written as a C++ include.

Run (~15 min on one core): python benchmarks/surface_resistance_specularity.py [out.inc]
"""
import sys
import time

import numpy as np
from scipy.special import expn

MU0 = 1.25663706212e-6


def _F(t):
    """RS / Pippard conductivity kernel F(t) = sigma(q)/sigma0 at t = q l: 3[(1+t^2) atan t - t] / (2 t^3)."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    s = t < 1e-2
    ts = t[s]
    out[s] = 1 - ts**2 / 5 + 3 * ts**4 / 35 - ts**6 / 21
    tl = t[~s]
    out[~s] = 1.5 * ((1 + tl**2) * np.arctan(tl) - tl) / tl**3
    return out


def z_exact(beta, p):
    """Closed-form RS impedance / (omega mu0 l) for p = 0 (diffuse, log integral) or p = 1 (specular)."""
    t0 = beta ** (1 / 3) if beta > 1 else np.sqrt(beta)
    h = 0.01
    t = t0 * np.exp(np.arange(-60, 60 + h / 2, h))
    if p == 0:
        return 1j * np.pi / (np.sum(np.log(1 + 1j * beta * _F(t) / t**2) * t) * h)
    return 2j / np.pi * np.sum(t / (t * t + 1j * beta * _F(t))) * h


def _En(n, x):
    x = np.abs(x)
    out = np.empty_like(x)
    z = x == 0
    out[z] = 1.0 / (n - 1)
    out[~z] = expn(n, x[~z])
    return out


def _K1(x):  # int_0^x k, odd
    a = np.abs(x)
    return np.sign(x) * 0.75 * ((1 - _En(2, a)) - (1 / 3 - _En(4, a)))


def _K2(x):  # int_0^x K1, even
    a = np.abs(x)
    return 0.75 * (2 * a / 3 - 0.25 + _En(3, a) - _En(5, a))


def z_numeric(beta, p, N=600, span=40.0, tail=10.0):
    """General-p RS impedance / (omega mu0 l) from the half-space integral equation."""
    d = beta ** (-1 / 3) if beta > 1 else beta ** -0.5  # field decay length / l
    S = span * d + (tail if beta > 1 else 0.0)
    h0 = d / 400
    lo, hi = 1.0 + 1e-12, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if h0 * (r**N - 1) / (r - 1) > S:
            hi = r
        else:
            lo = r
    s = np.concatenate([[0.0], np.cumsum(h0 * r ** np.arange(N))])
    n = len(s)
    W = np.zeros((n, n))
    for j in range(n):
        if j > 0:
            h = s[j] - s[j - 1]
            W[:, j] += (_K2(s - s[j - 1]) - _K2(s - s[j])) / h + p * (_K2(s + s[j - 1]) - _K2(s + s[j])) / h
        else:
            W[:, j] += _K1(s - s[0]) - p * _K1(s + s[0])
        if j < n - 1:
            h = s[j + 1] - s[j]
            W[:, j] += (_K2(s - s[j + 1]) - _K2(s - s[j])) / h + p * (_K2(s + s[j + 1]) - _K2(s + s[j])) / h
    A = np.zeros((n, n), complex)
    b = np.zeros(n, complex)
    A[0, 0] = 1
    b[0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        hm, hp = s[i] - s[i - 1], s[i + 1] - s[i]
        A[i, i - 1] += 2 / (hm * (hm + hp))
        A[i, i] += -2 / (hm * hp)
        A[i, i + 1] += 2 / (hp * (hm + hp))
        A[i, :] -= 1j * beta * W[i, :]
    E = np.linalg.solve(A, b)
    Ep0 = -1j * beta * np.trapezoid(W @ E, s)
    return -1j * E[0] / Ep0


LOG10_ALPHA = np.arange(-2.0, 12.0 + 1e-9, 0.5)
P_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 10)


def main(out=None):
    t_start = time.time()
    rows = []
    for la in LOG10_ALPHA:
        beta = 4 * 10**la / 3
        R0, R1 = z_exact(beta, 0).real, z_exact(beta, 1).real
        n0, n1 = z_numeric(beta, 0.0).real, z_numeric(beta, 1.0).real
        row = []
        for p in P_GRID:
            if p == 0:
                row.append(1.0)
            elif p == 1:
                row.append(0.0)
            else:
                # numeric endpoints in the denominator: discretization errors largely cancel in w
                row.append((z_numeric(beta, p).real - n1) / (n0 - n1))
        rows.append(row)
        print(f"log10 alpha = {la:5.1f}: R0 exact {R0:.6e} numeric rel {n0 / R0 - 1:+.1e}; R1 exact {R1:.6e} "
              f"numeric rel {n1 / R1 - 1:+.1e}; (R0-R1)/R0 = {(R0 - R1) / R0:.4f}; w(p=0.5) = {row[5]:.5f}",
              flush=True)
    print(f"elapsed {time.time() - t_start:.0f} s")
    lines = ["// Generated by benchmarks/surface_resistance_specularity.py: RS anomalous skin effect, weight",
             "// w(alpha, p) = (R(p) - R(1)) / (R(0) - R(1)) from the numerical half-space solution (quasi-static).",
             f"// log10(alpha) from {LOG10_ALPHA[0]} to {LOG10_ALPHA[-1]} step 0.5; p from 0 to 1 step 0.1.",
             f"static constexpr int kRsW_NA = {len(LOG10_ALPHA)}, kRsW_NP = {len(P_GRID)};",
             f"static constexpr double kRsW_LA0 = {LOG10_ALPHA[0]}, kRsW_DLA = 0.5;",
             "static constexpr double kRsW[kRsW_NA][kRsW_NP] = {"]
    for row in rows:
        lines.append("    {" + ", ".join(f"{v:.6f}" for v in row) + "},")
    lines.append("};")
    txt = "\n".join(lines) + "\n"
    if out:
        open(out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
