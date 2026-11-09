"""F13a manufactured solutions (inductionless MHD, 2.5D planar) and their self-checks (SymPy).

PDE, all fields independent of z, u = (u_x, u_y, u_z):
    J   = sigma (-grad phi + E_z e_z + u x B)                    grad = (d/dx, d/dy, 0)
    rho (u.grad) u - mu Lap u + grad p = G_z e_z + J x B + f     (constant mu: both viscous forms coincide here)
    d_x u_x + d_y u_y = 0
    d_x J_x + d_y J_y = q                                         (q: manufactured current source; 0 in mhd_insulating)
Forcing route 1: symbolic (this file).  Route 2: fd_residual() rebuilds every residual with 4th-order central
differences from (u, p, phi, sigma, B) ONLY and must reproduce (f, q) to < 1e-7 relative; it has teeth (a flipped
u x B or a missing sigma gives > 1e-2, checked).

    python3 -B tools/fluids_mms/f13a_fields.py        # writes tools/fluids_mms/out/mhd_*.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
R = sp.Rational


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def grad2(s):
    return [sp.diff(s, x), sp.diff(s, y)]


def spell(e):
    e = sp.sympify(e)
    return {"sympy": sp.sstr(e), "c": sp.ccode(e),
            "numpy": sp.pycode(e, fully_qualified_modules=True).replace("math.", "np.")}


class MHDField:
    def __init__(self, name, desc, u, p, phi, sigma, B, mu, rho_values, Gz, Ez, edges, extra=None):
        self.name, self.desc = name, desc
        self.u = [sp.sympify(c) for c in u]
        self.p, self.phi, self.sigma = sp.sympify(p), sp.sympify(phi), sp.sympify(sigma)
        self.B = [sp.sympify(c) for c in B]
        self.mu, self.Gz, self.Ez = sp.sympify(mu), sp.sympify(Gz), sp.sympify(Ez)
        self.rho_values = rho_values
        self.edges = edges          # {"x0": "dirichlet_phi" | "jn" | "insulating", ...}
        self.extra = extra or {}
        uxB = cross(self.u, self.B)
        gphi = grad2(self.phi)
        self.J = [self.sigma * (-gphi[0] + uxB[0]), self.sigma * (-gphi[1] + uxB[1]), self.sigma * (self.Ez + uxB[2])]
        self.q = sp.diff(self.J[0], x) + sp.diff(self.J[1], y)
        self.F = cross(self.J, self.B)                       # Lorentz force J x B
        self.f = {}
        for rho in rho_values:
            fr = []
            for i in range(3):
                conv = rho * (self.u[0] * sp.diff(self.u[i], x) + self.u[1] * sp.diff(self.u[i], y))
                visc = -self.mu * (sp.diff(self.u[i], x, 2) + sp.diff(self.u[i], y, 2))
                gp = [sp.diff(self.p, x), sp.diff(self.p, y), 0][i]
                fr.append(conv + visc + gp - (self.Gz if i == 2 else 0) - self.F[i])
            self.f[str(rho)] = fr

    def emit(self):
        n = {"x0": (-1, 0), "x1": (1, 0), "y0": (0, -1), "y1": (0, 1)}
        bd = {}
        for e, kind in self.edges.items():
            nx_, ny_ = n[e]
            jn = self.J[0] * nx_ + self.J[1] * ny_
            bd[e] = {"phi_bc": kind, "normal": [nx_, ny_], "jn": spell(jn), "phi": spell(self.phi)}
        out = {"name": self.name, "description": self.desc, "domain": "unit square [0,1]^2",
               "conventions": ("J = sigma(-grad phi + E_z e_z + u x B); f = rho (u.grad)u - mu Lap u + grad p - G_z e_z "
                               "- J x B; q = div J; j_n = J.n with the OUTWARD normal; grad_u[i][j] = d u_i/d x_j"),
               "u": [spell(c) for c in self.u],
               "grad_u": [[spell(sp.diff(c, v)) for v in (x, y)] for c in self.u],
               "p": spell(self.p), "phi": spell(self.phi), "grad_phi": [spell(c) for c in grad2(self.phi)],
               "sigma": spell(self.sigma), "B": [spell(c) for c in self.B], "J": [spell(c) for c in self.J],
               "lorentz": [spell(c) for c in self.F], "q": spell(self.q),
               "f": {rho: [spell(c) for c in fr] for rho, fr in self.f.items()},
               "boundaries": bd,
               "constants": {"mu": float(self.mu), "G_z": float(self.Gz), "E_z": float(self.Ez),
                             "rho_values": [float(r) for r in self.rho_values]},
               "extra": {k: spell(v) for k, v in self.extra.items()}}
        return out


# ---------------------------------------------------------------------------------------------- fields
def field_mixed():
    """T1 mhd_mixed: variable sigma, variable harmonic B with all three components, non-polynomial, inhomogeneous
    velocity on every edge with (u x B).n != 0; phi Dirichlet on x = 0, normal current density j_n on x = 1 and y = 0,
    homogeneous insulation J.n = 0 on y = 1 (built in); current source q != 0."""
    psi = R(1, 4) * (sp.exp(x / 2) * sp.sin(R(3, 2) * y + R(2, 5)) + R(3, 10) * x ** 2 * y)
    u = [sp.diff(psi, y), -sp.diff(psi, x), R(3, 5) * sp.cos(R(13, 10) * x - R(2, 5)) * sp.exp(y / 2) + R(1, 5) * x * y]
    p = sp.cos(R(19, 10) * x + R(1, 5)) * sp.sinh(R(4, 5) * y + R(1, 10)) + R(2, 5) * x * y
    sigma = R(6, 5) + R(2, 5) * sp.sin(R(3, 2) * x - y + R(3, 10))
    chi = R(3, 5) * sp.exp(x / 2) * sp.sin(y / 2)                     # harmonic: B_perp = B0 + grad chi
    B = [R(9, 10) + sp.diff(chi, x), R(3, 5) + sp.diff(chi, y), R(7, 10)]
    mu, Gz, Ez = R(3, 5), R(13, 10), R(9, 20)
    phis = R(1, 2) * sp.sin(R(11, 10) * x + R(7, 10) * y + R(1, 5)) + R(3, 10) * x ** 2 - R(1, 5) * y * sp.exp(x / 3)
    uxB = cross(u, B)
    s1 = uxB[1].subs(y, 1)                                            # J_y(x,1) = 0  <=>  phi_y(x,1) = (u x B)_y(x,1)
    phi = phis + (s1 - sp.diff(phis, y).subs(y, 1)) * y ** 2 / 2
    return MHDField("mhd_mixed", field_mixed.__doc__, u, p, phi, sigma, B, mu, [0, 1], Gz, Ez,
                    {"x0": "dirichlet_phi", "x1": "jn", "y0": "jn", "y1": "insulating"}, extra={"chi": chi})


def field_insulating():
    """T2 mhd_insulating: uniform sigma and B = (0, B_y, B_z); SOURCE-FREE (q = 0) and J.n = 0 on all four edges
    (all walls insulating, phi up to a constant), by construction from a current stream function H vanishing on the
    boundary: J_perp = (H_y, -H_x); u_z is chosen so that (u x B)_perp - J_perp/sigma is a gradient
    (sigma B_y du_z/dy = -Lap H), and phi is its potential.  (u x B).n != 0 on the boundary."""
    sig, By, Bz = R(8, 5), R(6, 5), R(4, 5)
    mu, Gz, Ez = R(3, 5), R(13, 10), R(9, 20)
    psi = R(1, 4) * sp.exp(x / 2 - y / 3) * (1 + x * y + y ** 2 / 2)
    H = R(4, 5) * x * (1 - x) * y * (1 - y) * sp.exp((x + y) / 3)
    lapH = sp.diff(H, x, 2) + sp.diff(H, y, 2)
    uz = R(1, 2) + R(3, 10) * sp.exp(x / 2) - sp.integrate(lapH.subs(y, t), (t, 0, y)) / (sig * By)
    u = [sp.diff(psi, y), -sp.diff(psi, x), sp.simplify(uz)]
    B = [0, By, Bz]
    uxB = cross(u, B)
    gx = uxB[0] - sp.diff(H, y) / sig
    gy = uxB[1] + sp.diff(H, x) / sig
    phi = sp.integrate(gx.subs({x: t, y: 0}), (t, 0, x)) + sp.integrate(gy.subs(y, t), (t, 0, y))
    phi = sp.simplify(phi)
    p = sp.cos(R(13, 10) * x - R(7, 10) * y + R(1, 5)) + R(1, 2) * sp.sin(x + 2 * y)
    fld = MHDField("mhd_insulating", field_insulating.__doc__, u, p, phi, sig, B, mu, [0, 1], Gz, Ez,
                   {"x0": "insulating", "x1": "insulating", "y0": "insulating", "y1": "insulating"}, extra={"H": H})
    fld.H = H
    return fld


# ---------------------------------------------------------------------------------------------- self-checks
def _lam(e):
    return sp.lambdify((x, y), e, modules="numpy")


def fd_residual(fld, rho, pts, h=4e-3, flip_uxB=False, drop_sigma_force=False):
    """Route 2: residuals of momentum (3) and current conservation from u, p, phi, sigma, B by 4th-order central
    differences (never from the symbolic f or q).  Returns max relative deviation from (f, q)."""
    U = [_lam(c) for c in fld.u]
    P, PHI, SIG = _lam(fld.p), _lam(fld.phi), _lam(fld.sigma)
    Bf = [_lam(c) for c in fld.B]
    mu, Gz, Ez = float(fld.mu), float(fld.Gz), float(fld.Ez)

    def d(fn, X, Y, ax):
        if ax == 0:
            return (-fn(X + 2 * h, Y) + 8 * fn(X + h, Y) - 8 * fn(X - h, Y) + fn(X - 2 * h, Y)) / (12 * h)
        return (-fn(X, Y + 2 * h) + 8 * fn(X, Y + h) - 8 * fn(X, Y - h) + fn(X, Y - 2 * h)) / (12 * h)

    def Jfn(X, Y):
        uu = [np.broadcast_to(np.asarray(f(X, Y), float), np.shape(X)) for f in U]
        bb = [np.broadcast_to(np.asarray(f(X, Y), float), np.shape(X)) for f in Bf]
        uxB = [uu[1] * bb[2] - uu[2] * bb[1], uu[2] * bb[0] - uu[0] * bb[2], uu[0] * bb[1] - uu[1] * bb[0]]
        s = -1.0 if flip_uxB else 1.0
        sg = SIG(X, Y)
        return [sg * (-d(PHI, X, Y, 0) + s * uxB[0]), sg * (-d(PHI, X, Y, 1) + s * uxB[1]), sg * (Ez + s * uxB[2])]

    X, Y = pts[:, 0], pts[:, 1]
    J = Jfn(X, Y)
    bb = [np.broadcast_to(np.asarray(f(X, Y), float), X.shape) for f in Bf]
    if drop_sigma_force:
        sg = SIG(X, Y)
        J = [j / sg for j in J]
    F = [J[1] * bb[2] - J[2] * bb[1], J[2] * bb[0] - J[0] * bb[2], J[0] * bb[1] - J[1] * bb[0]]
    worst = 0.0
    for i in range(3):
        ui = U[i]
        lap = (d(lambda a, b: d(ui, a, b, 0), X, Y, 0) + d(lambda a, b: d(ui, a, b, 1), X, Y, 1))
        conv = rho * (U[0](X, Y) * d(ui, X, Y, 0) + U[1](X, Y) * d(ui, X, Y, 1))
        gp = [d(P, X, Y, 0), d(P, X, Y, 1), 0.0][i]
        res = conv - mu * lap + gp - (Gz if i == 2 else 0.0) - F[i]
        fsym = np.broadcast_to(np.asarray(_lam(fld.f[str(rho)][i])(X, Y), float), X.shape)
        worst = max(worst, float(np.max(np.abs(res - fsym)) / max(np.max(np.abs(fsym)), 1.0)))
    divJ = d(lambda a, b: Jfn(a, b)[0], X, Y, 0) + d(lambda a, b: Jfn(a, b)[1], X, Y, 1)
    qs = np.broadcast_to(np.asarray(_lam(fld.q)(X, Y), float), X.shape)
    qscale = max(float(np.max(np.abs(qs))), float(np.max(np.abs(np.r_[J[0], J[1]]))))
    worst_q = float(np.max(np.abs(divJ - qs)) / qscale)
    return worst, worst_q


def gauss_square(n=40):
    g, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (g + 1), 0.5 * w
    X, Y = np.meshgrid(s, s, indexing="ij")
    return X.ravel(), Y.ravel(), np.outer(ws, ws).ravel()


def selfcheck(fld):
    rep = {}
    ok = True

    def need(cond, msg):
        nonlocal ok
        rep[msg] = bool(cond)
        ok &= bool(cond)

    need(sp.simplify(sp.diff(fld.u[0], x) + sp.diff(fld.u[1], y)) == 0, "div u == 0 (symbolic)")
    rng = np.random.default_rng(13)
    pts = 0.1 + 0.8 * rng.random((12, 2))
    for rho in fld.rho_values:
        wf, wq = fd_residual(fld, rho, pts)
        rep[f"fd_residual_rho{rho}_momentum"] = wf
        rep[f"fd_residual_rho{rho}_current"] = wq
        need(wf < 1e-7 and wq < 1e-7, f"FD residual (rho={rho}) reproduces f and q to 1e-7")
    a, b = fd_residual(fld, fld.rho_values[0], pts, flip_uxB=True)
    rep["fd_teeth_flipped_uxB"] = max(a, b)
    need(max(a, b) > 1e-2, "FD check has teeth: flipped u x B > 1e-2")
    a, _ = fd_residual(fld, fld.rho_values[0], pts, drop_sigma_force=True)
    rep["fd_teeth_force_without_sigma"] = a
    need(a > 1e-2, "FD check has teeth: Lorentz force without sigma > 1e-2")
    # boundary conditions
    for e, kind in fld.edges.items():
        nx_, ny_ = {"x0": (-1, 0), "x1": (1, 0), "y0": (0, -1), "y1": (0, 1)}[e]
        jn = fld.J[0] * nx_ + fld.J[1] * ny_
        sub = {x: 0} if e == "x0" else {x: 1} if e == "x1" else {y: 0} if e == "y0" else {y: 1}
        s = np.linspace(0.0, 1.0, 41)
        var = y if e in ("x0", "x1") else x
        jv = _lam(jn.subs(sub)) if True else None
        vals = np.array([float(sp.N(jn.subs(sub).subs(var, sp.Float(v, 30)), 25)) for v in s[::4]])
        uxB = cross(fld.u, fld.B)
        uxbn = uxB[0] * nx_ + uxB[1] * ny_
        uxbn_v = np.array([float(sp.N(uxbn.subs(sub).subs(var, sp.Float(v, 30)), 25)) for v in s[::4]])
        rep[f"edge_{e}_max|(u x B).n|"] = float(np.max(np.abs(uxbn_v)))
        need(np.max(np.abs(uxbn_v)) > 1e-2, f"edge {e}: (u x B).n != 0 (a non-conservative boundary treatment shows)")
        if kind == "insulating":
            need(np.max(np.abs(vals)) < 1e-20, f"edge {e}: J.n == 0 (insulating, 25-digit check)")
        else:
            need(np.max(np.abs(vals)) > 1e-2, f"edge {e}: J.n != 0 (insulation on the wrong edge shows)")
        phv = np.array([float(sp.N(fld.phi.subs(sub).subs(var, sp.Float(v, 30)), 25)) for v in s[::4]])
        rep[f"edge_{e}_max|phi - mean|"] = float(np.max(np.abs(phv - phv.mean())))
    # integral constants and magnitudes
    X, Y, W = gauss_square(40)
    vals = {}
    for nm, e in (("ux", fld.u[0]), ("uy", fld.u[1]), ("uz", fld.u[2]), ("p", fld.p), ("phi", fld.phi),
                  ("Jx", fld.J[0]), ("Jy", fld.J[1]), ("Jz", fld.J[2]), ("q", fld.q)):
        vals[nm] = np.broadcast_to(np.asarray(_lam(e)(X, Y), float), X.shape)
    vis = [np.broadcast_to(np.asarray(_lam(-fld.mu * (sp.diff(c, x, 2) + sp.diff(c, y, 2)))(X, Y), float), X.shape)
           for c in fld.u]
    lor = [np.broadcast_to(np.asarray(_lam(c)(X, Y), float), X.shape) for c in fld.F]
    umag = np.sqrt(vals["ux"] ** 2 + vals["uy"] ** 2 + vals["uz"] ** 2)
    Jmag = np.sqrt(vals["Jx"] ** 2 + vals["Jy"] ** 2 + vals["Jz"] ** 2)
    pm = float(np.sum(W * vals["p"]))
    phim = float(np.sum(W * vals["phi"]))
    c = {"p_mean": pm, "phi_mean": phim,
         "u_max_sampled": float(umag.max()), "p_minus_mean_max_abs_sampled": float(np.max(np.abs(vals["p"] - pm))),
         "phi_max_abs_sampled": float(np.max(np.abs(vals["phi"]))),
         "phi_minus_mean_max_abs_sampled": float(np.max(np.abs(vals["phi"] - phim))),
         "J_max_sampled": float(Jmag.max()), "q_max_abs_sampled": float(np.max(np.abs(vals["q"]))),
         "lorentz_L2": float(np.sqrt(np.sum(W * sum(l ** 2 for l in lor)))),
         "viscous_L2": float(np.sqrt(np.sum(W * sum(v ** 2 for v in vis)))),
         "net_Jx": float(np.sum(W * vals["Jx"])), "net_Jy": float(np.sum(W * vals["Jy"])),
         "joule": float(np.sum(W * Jmag ** 2 / np.broadcast_to(np.asarray(_lam(fld.sigma)(X, Y), float), X.shape)))}
    rep["constants"] = c
    need(0.2 < c["lorentz_L2"] / c["viscous_L2"] < 5, "Lorentz force comparable to the viscous term (0.2..5)")
    if fld.name == "mhd_insulating":
        need(sp.simplify(fld.q) == 0, "q == 0 symbolically (source-free)")
        Jp = [sp.diff(fld.H, y), -sp.diff(fld.H, x)]
        need(all(sp.simplify(fld.J[i] - Jp[i]) == 0 for i in range(2)), "J_perp == curl H symbolically")
        need(abs(c["net_Jx"]) < 1e-12 and abs(c["net_Jy"]) < 1e-12, "int J_perp = 0 (exact field)")
    sg = np.broadcast_to(np.asarray(_lam(fld.sigma)(X, Y), float), X.shape)
    c["sigma_min"], c["sigma_max"] = float(sg.min()), float(sg.max())
    need(sg.min() > 0, "sigma > 0")
    return ok, rep, c


def main():
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1:]
    summary = {}
    for mk in (field_mixed, field_insulating):
        fld = mk()
        if only and fld.name not in only:
            continue
        ok, rep, c = selfcheck(fld)
        data = fld.emit()
        data["constants"].update(c)
        data["selfcheck"] = {k: v for k, v in rep.items() if k != "constants"}
        data["selfcheck_ok"] = ok
        txt = json.dumps(data, indent=1, sort_keys=True) + "\n"
        (OUT / f"{fld.name}.json").write_text(txt)
        summary[fld.name] = {"ok": ok, "sha256": hashlib.sha256(txt.encode()).hexdigest(), "checks": rep}
        print(fld.name, "OK" if ok else "FAILED", hashlib.sha256(txt.encode()).hexdigest())
        for k, v in rep.items():
            if k != "constants":
                print("   ", k, v)
    (OUT / "f13a_selfcheck_report.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    sys.exit(0 if all(s["ok"] for s in summary.values()) else 1)


if __name__ == "__main__":
    main()
