"""2D z-uniform TM (E_z) eigen-solver for the ADMX two-rod cavity cross-section.

Independent of axionFEM: Shortley-Weller finite differences (second order on curved Dirichlet
boundaries) for  -lap(E_z) = k^2 E_z  in  {|x| < R} minus two disks of radius r_rod,  E_z = 0 on all
metal.  f = c k / (2 pi).  This is the ideal z-uniform limit (rods touching both end caps, no end
gaps, no alumina armatures); it is used ONLY to infer the unpublished pivot geometry (pivot radius
r_p, arm length a) from published CST/COMSOL frequencies, see admx_two_rod_infer.py.

Rod-angle convention (2010.00169 Sec. II.C): theta = 0 -> arm points radially outward (rod at the
wall, r = r_p + a); theta = 180 -> rod toward the centre (r = r_p - a).  Pivots sit on one diameter
at (-r_p, 0) and (+r_p, 0).  For TM frequencies the sense of rotation does not matter (mirror
symmetry), so theta and -theta are equivalent; configurations are labelled (theta_left, theta_right).

Library use:  from admx_two_rod_2d import tm_modes, rod_centres
CLI check:    python3 -I admx_two_rod_2d.py   (empty disk + coax convergence vs closed form)
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import jn_zeros, jv, yv
from scipy.optimize import brentq

C0 = 299792458.0


def rod_centres(r_p, a, th_left_deg, th_right_deg, sense_right=+1):
    """Rod centres (m) for pivots at (-r_p,0) (left) and (+r_p,0) (right).

    Left arm at theta=0 points to -x; right arm at theta=0 points to +x.  Positive theta rotates the
    left arm counter-clockwise; the right arm rotates counter-clockwise for sense_right=+1
    ('symmetric' = C2 point symmetry) or clockwise for -1 ('anti-symmetric' = mirror symmetry about
    the y axis).  A global mirror leaves f unchanged, but the RELATIVE sense matters (Run 1B
    symmetric vs anti-symmetric), so pass sense_right explicitly.
    """
    tl = np.radians(th_left_deg)
    tr = np.radians(th_right_deg)
    left = np.array([-r_p, 0.0]) + a * np.array([-np.cos(tl), -np.sin(tl)])
    right = np.array([r_p, 0.0]) + a * np.array([np.cos(tr), sense_right * np.sin(tr)])
    return left, right


def _dist_along(x0, y0, dx, dy, circles, h):
    """Fraction s in (0,1] of the step h along (dx,dy) from (x0,y0) to the first metal boundary."""
    best = 1.0
    for (cx, cy, rr, inside_is_vac) in circles:
        # |p0 + t d - c|^2 = rr^2, t in (0, h]
        px, py = x0 - cx, y0 - cy
        b = px * dx + py * dy
        cc = px * px + py * py - rr * rr
        disc = b * b - cc
        if disc < 0:
            continue
        sq = np.sqrt(disc)
        for t in (-b - sq, -b + sq):
            if 1e-12 < t <= h * (1 + 1e-12):
                best = min(best, t / h)
    return best


def tm_modes(R, r_rod, centres, h, nev=4, f_guess=None):
    """Lowest TM (E_z, Dirichlet) modes. Returns (f_Hz sorted, list of (X, Y, Ez grid), info)."""
    n = int(np.ceil(R / h)) + 2
    xs = np.arange(-n, n + 1) * h
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    vac = X ** 2 + Y ** 2 < R ** 2
    for (cx, cy) in centres:
        vac &= (X - cx) ** 2 + (Y - cy) ** 2 > r_rod ** 2
    idx = -np.ones(X.shape, int)
    iy, ix = np.nonzero(vac)
    idx[iy, ix] = np.arange(iy.size)
    circles = [(0.0, 0.0, R, True)] + [(cx, cy, r_rod, False) for (cx, cy) in centres]
    rows, cols, vals = [], [], []
    N = iy.size
    for k in range(N):
        j, i = iy[k], ix[k]
        x0, y0 = X[j, i], Y[j, i]
        diag = 0.0
        for (axis, sgn_pairs) in ((0, ((1, 0), (-1, 0))), (1, ((0, 1), (0, -1)))):
            s = []
            nb = []
            for (di, dj) in sgn_pairs:
                jj, ii = j + dj, i + di
                if vac[jj, ii]:
                    # check the segment does not cross metal (thin features)
                    frac = _dist_along(x0, y0, di, dj, circles, h)
                    if frac < 1.0 - 1e-9:
                        s.append(frac); nb.append(None)
                    else:
                        s.append(1.0); nb.append(idx[jj, ii])
                else:
                    s.append(_dist_along(x0, y0, di, dj, circles, h)); nb.append(None)
            hp, hm = s[0] * h, s[1] * h
            # -u'' ~ -2/(hp+hm) [ (u+ - u)/hp - (u - u-)/hm ]
            cp = -2.0 / (hp * (hp + hm))
            cm = -2.0 / (hm * (hp + hm))
            diag += -(cp + cm)
            if nb[0] is not None:
                rows.append(k); cols.append(nb[0]); vals.append(cp)
            if nb[1] is not None:
                rows.append(k); cols.append(nb[1]); vals.append(cm)
        rows.append(k); cols.append(k); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    if f_guess is None:
        f_guess = 0.9 * C0 * 2.404825557695773 / (2 * np.pi * R)
    sigma = (2 * np.pi * f_guess / C0) ** 2
    w, V = sla.eigs(A, k=nev, sigma=sigma, which="LM")
    w = np.real(w)
    o = np.argsort(w)
    f = C0 * np.sqrt(np.abs(w[o])) / (2 * np.pi)
    fields = []
    for m in o:
        E = np.zeros(X.shape)
        v = np.real(V[:, m])
        v = v / v[np.argmax(np.abs(v))]
        E[iy, ix] = v
        fields.append(E)
    info = {"h_m": h, "dofs": int(N), "shift_Hz": f_guess}
    return f, (X, Y, fields), info


def form_factor_2d(X, Y, E, h, R, r_rod, n_rods=2):
    """2D z-uniform, uniform-B C: (int E)^2 / (A int E^2); returns (C_full_disk, C_vacuum_area)."""
    s1 = E.sum() * h * h
    s2 = (E ** 2).sum() * h * h
    A_full = np.pi * R ** 2
    A_vac = A_full - n_rods * np.pi * r_rod ** 2
    return s1 ** 2 / (A_full * s2), s1 ** 2 / (A_vac * s2)


def coax_f(R, r):
    g = lambda k: jv(0, k * r) * yv(0, k * R) - jv(0, k * R) * yv(0, k * r)
    ks = np.linspace(1.0, 4 * np.pi / (R - r), 4000)
    gv = g(ks)
    i = np.nonzero(np.sign(gv[1:]) != np.sign(gv[:-1]))[0][0]
    return C0 * brentq(g, ks[i], ks[i + 1]) / (2 * np.pi)


if __name__ == "__main__":
    R, r = 0.21, 0.0254
    f_ex = C0 * jn_zeros(0, 1)[0] / (2 * np.pi * R)
    for h in (2e-3, 1e-3, 0.5e-3):
        f, _, info = tm_modes(R, r, [], h, nev=2, f_guess=0.5e9)
        print(f"empty  h={h*1e3:.2f} mm  f0={f[0]/1e6:.4f} MHz  exact={f_ex/1e6:.4f}  rel={f[0]/f_ex-1:+.2e}  dofs={info['dofs']}")
    fc = coax_f(R, r)
    for h in (2e-3, 1e-3):
        f, _, info = tm_modes(R, r, [(0.0, 0.0)], h, nev=2, f_guess=0.6e9)
        print(f"coax   h={h*1e3:.2f} mm  f0={f[0]/1e6:.4f} MHz  exact={fc/1e6:.4f}  rel={f[0]/fc-1:+.2e}")
    sys.stdout.flush()
