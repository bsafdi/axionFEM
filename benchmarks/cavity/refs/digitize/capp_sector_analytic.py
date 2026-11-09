"""Closed-form TM_{nu,1} (z-invariant, TM010-like) modes of an ideal PEC sector cell, for the CAPP multi-cell
benchmark (arXiv:1710.06969 Tables 2-3, arXiv:2205.01319 Table 2).

N identical cells of opening angle theta = 2 pi / N, zero-thickness partitions meeting on the axis.
Ez = J_nu(k r) sin(nu phi), 0 <= phi <= theta, nu = pi / theta = N/2, k = chi_{nu,1} / Rc.
Q (walls only, no end caps)  = (2/delta) k^2 int|Ez|^2 dA / oint |dEz/dn|^2 dl
Q (with lossy end caps, height L): for a z-invariant TM mode the two caps add 2 k^2 int|Ez|^2 dA to the loss integral,
   i.e. 1/Q = 1/Q_wall + delta/L  (equivalently 2205.01319 Eq. 5: Q = (Rc/delta)/(1/Qbar + Rc/L)).
C = |int Ez dA|^2 / (A int |Ez|^2 dA)   (uniform B0 || z, eps = 1); same for one cell and for all N in phase.
Usage: python3 -I capp_sector_analytic.py      (prints a table; writes nothing)
"""
import numpy as np
from scipy import special as sp, integrate, optimize

MU0 = 4e-7 * np.pi
C0 = 299792458.0


def chi(nu):
    # first positive zero of J_nu
    f = lambda x: sp.jv(nu, x)
    x0 = nu + 1.8557571 * nu ** (1 / 3) if nu > 0 else 2.0
    xs = np.linspace(max(0.5, nu), nu + 6 + 2 * nu ** (1 / 3), 4000)
    v = f(xs); i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0][0]
    return optimize.brentq(f, xs[i], xs[i + 1], xtol=1e-14)


def sector(N, Rc, sigma, L=None):
    if N == 0:
        nu, theta = 0.0, 2 * np.pi
    else:
        theta = 2 * np.pi / N; nu = np.pi / theta
    x = chi(nu); k = x / Rc; f = k * C0 / (2 * np.pi)
    delta = np.sqrt(2 / (2 * np.pi * f * MU0 * sigma))
    I2 = integrate.quad(lambda r: sp.jv(nu, k * r) ** 2 * r, 0, Rc, epsabs=0, epsrel=1e-13, limit=200)[0]
    I1 = integrate.quad(lambda r: sp.jv(nu, k * r) * r, 0, Rc, epsabs=0, epsrel=1e-13, limit=200)[0]
    ang2 = theta / 2 if N else 2 * np.pi          # int sin^2(nu phi) dphi  (N=0: Ez=J0, int 1 dphi = 2pi)
    ang1 = 2 / nu if N else 2 * np.pi             # int sin(nu phi) dphi
    E2 = ang2 * I2
    arc = k ** 2 * sp.jvp(nu, x) ** 2 * Rc * ang2
    sides = 0.0
    if N:
        sides = 2 * integrate.quad(lambda r: (nu / r) ** 2 * sp.jv(nu, k * r) ** 2 if r > 0 else 0.0, 0, Rc,
                                   epsabs=0, epsrel=1e-13, limit=200)[0]
    Q_wall = (2 / delta) * k ** 2 * E2 / (arc + sides)
    A = theta * Rc ** 2 / 2 if N else np.pi * Rc ** 2
    Cff = (ang1 * I1) ** 2 / (A * E2)
    out = dict(N=N, theta=theta, nu=nu, chi=x, f_GHz=f / 1e9, delta_um=delta * 1e6, Q_no_endcaps=Q_wall,
               Prad_over_Pazi=sides / arc, C=Cff)
    if L is not None:
        # end caps: |H_t|^2 = |grad Ez|^2/(omega mu)^2 on each cap; walls: |dEz/dn|^2/(omega mu)^2 per unit length
        caps = 2 * k ** 2 * E2                  # 2 caps * int |grad Ez|^2 dA  (= k^2 int |Ez|^2 dA)
        Q_full = (2 / delta) * k ** 2 * E2 * L / ((arc + sides) * L + caps)
        out['Q_with_endcaps'] = Q_full
    return out


if __name__ == '__main__':
    print('1710.06969 Table 2/3 check: Rc = 25 mm, sigma = 5.96e7 S/m, no end caps')
    base = sector(0, 0.025, 5.96e7)
    for N in (0, 2, 4, 6, 8):
        o = sector(N, 0.025, 5.96e7)
        print('N=%d nu=%.1f chi=%.6f f=%.4f GHz Q=%.1f Q/Q0=%.4f w/w0=%.4f Prad/Pazi=%.4f C=%.5f' % (
            N, o['nu'], o['chi'], o['f_GHz'], o['Q_no_endcaps'], o['Q_no_endcaps'] / base['Q_no_endcaps'],
            o['f_GHz'] / base['f_GHz'], o['Prad_over_Pazi'], o['C']))
