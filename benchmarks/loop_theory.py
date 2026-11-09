"""Input impedance of a gap-fed thin circular loop (Storer 1956; Wu 1962, J. Math. Phys. 3, 1301).

Reference for the 3D full-wave solver, whose readout current enters a superconducting ring through a slit: unlike
the axisymmetric (uniform-feed) solver it sees the non-uniform current modes n >= 1 and the loop's self-resonances.

Time dependence e^{-i omega t} on output (Z = R - i X with X = omega L > 0 for an inductive loop). Loop radius b,
wire radius a (a << b), wavenumber k:

  Y = sum_n Y_n s_n^2,   Y_n = -j / (pi zeta0 a_n)   (e^{+j omega t} in Wu's paper; converted below)
  a_n = (kb/2) (K_{n+1} + K_{n-1}) - n^2 K_n / (kb),   K_{-n} = K_n
  K_0 = (1/pi) ln(8b/a) - (1/2) int_0^{2kb} [Omega_0(x) + j J_0(x)] dx
  K_n = (1/pi) [K_0(n a/b) I_0(n a/b) + C_n] - (1/2) int_0^{2kb} [Omega_{2n}(x) + j J_{2n}(x)] dx
  C_n = ln(4n) + gamma - 2 sum_{m<n} 1/(2m+1)
  Omega_nu(x) = (1/pi) int_0^pi sin(x sin t - nu t) dt   (Lommel-Weber)

s_n = sinc(n Delta/2) spreads the drive over a gap of angular width Delta (the delta-gap sum diverges
logarithmically: the gap capacitance); the series is truncated at n_max ~ 2 b/a beyond which the thin-wire
kernel is no longer valid.
Checks: kb -> 0 gives L = mu0 b (ln(8b/a) - 2) and R = (pi zeta0 / 6) (kb)^4 = 20 pi^2 (kb)^4.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

ZETA0 = 376.730313668
MU0 = 4e-7 * math.pi
C0 = 299792458.0


def _theta_rule(npts=4000):
    t, w = np.polynomial.legendre.leggauss(npts)
    return 0.5 * math.pi * (t + 1), 0.5 * math.pi * w


def _int_omega(nu, X, rule):
    # int_0^X Omega_nu(x) dx = (1/pi) int_0^pi [cos(nu t) - cos(X sin t - nu t)] / sin t dt
    t, w = rule
    s = np.sin(t)
    return float(np.sum(w * (np.cos(nu * t) - np.cos(X * s - nu * t)) / s) / math.pi)


def _int_bessel(nu, X):
    # int_0^X J_nu(x) dx by Gauss-Legendre (smooth integrand)
    x, w = np.polynomial.legendre.leggauss(200)
    x = 0.5 * X * (x + 1)
    return float(0.5 * X * np.sum(w * special.jv(nu, x)))


def _K(n, kb, b_over_a, rule):
    X = 2 * kb
    if n == 0:
        base = math.log(8 * b_over_a) / math.pi
    else:
        e = n / b_over_a
        Cn = math.log(4 * n) + np.euler_gamma - 2 * sum(1.0 / (2 * m + 1) for m in range(n))
        base = (special.k0(e) * special.i0(e) + Cn) / math.pi
    return base - 0.5 * (_int_omega(2 * n, X, rule) + 1j * _int_bessel(2 * n, X))


def loop_impedance(kb: float, b_over_a: float, gap_rad: float = 0.0, n_max: int | None = None) -> complex:
    """Input impedance (ohm, e^{-i omega t}: Z = R - i omega L) of a gap-fed thin loop."""
    rule = _theta_rule()
    if n_max is None:
        n_max = max(4, int(2 * b_over_a))
    K = [_K(n, kb, b_over_a, rule) for n in range(n_max + 2)]
    Y = 0j
    for n in range(n_max + 1):
        an = 0.5 * kb * (K[n + 1] + K[abs(n - 1)]) - n * n * K[n] / kb
        s = 1.0 if n == 0 or gap_rad == 0 else math.sin(n * gap_rad / 2) / (n * gap_rad / 2)
        Y += (1 if n == 0 else 2) * (-1j / (math.pi * ZETA0 * an)) * s * s
    Z = 1.0 / Y  # e^{+j omega t}: Z = R + j omega L
    return Z.conjugate()


def loop_L_R(f_Hz: float, b: float, a: float, gap_rad: float = 0.0, n_max: int | None = None):
    k = 2 * math.pi * f_Hz / C0
    Z = loop_impedance(k * b, b / a, gap_rad, n_max)
    return -Z.imag / (2 * math.pi * f_Hz), Z.real


if __name__ == "__main__":
    b, a = 0.3, 0.59 * 0.02  # 2 cm square section: equivalent radius 0.59 s
    for kb in (0.002, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        f = kb * C0 / (2 * math.pi * b)
        L, R = loop_L_R(f, b, a, gap_rad=math.radians(6.0))
        L0 = MU0 * b * (math.log(8 * b / a) - 2)
        print(f"kb={kb:<6} L={L:.6e} L/L_lowf={L / L0:.5f} R={R:.5e} R_dipole={math.pi * ZETA0 / 6 * kb ** 4:.5e}")
