"""F13a: closed-form, fully developed, NON-MMS solutions of the 2.5D inductionless MHD
system with a NON-UNIFORM, div- and curl-free imposed field, in the annulus r1 < r < r2.

Independent derivation (the radial
ODEs are DERIVED from the Cartesian 2.5D equations and solved with sympy.dsolve, then checked back):

  1. radial field     B = (C/r) e_r = (C x/r^2, C y/r^2, 0)      (harmonic potential C ln r)
     ansatz u = u_t(r) e_theta + w(r) e_z, phi = phi(r), p = p(r)
  2. azimuthal field  B = (C/r) e_theta = (-C y/r^2, C x/r^2, 0)  (harmonic potential C theta; line current on the axis)
     ansatz u = w(r) e_z, phi = phi(r), p = p(r)

Steps (each printed and recorded in the report):
  a. div B = 0, curl B = 0 symbolically;
  b. the Cartesian residuals of the ansatz, rewritten in polar coordinates, are theta-independent ODEs (checked at
     several theta), and are those ODEs;
  c. dsolve gives the general solutions; the wall conditions (no-slip, insulating J.n = 0 / grounded phi = 0 for the
     radial field; phi(r1) = 0, phi(r2) = V for the azimuthal field) fix the constants;
  d. the solved fields are substituted back into the CARTESIAN section-1 equations at rational points with every
     parameter symbolic-rational (40 digits): all residuals vanish;
  e. second route, never through the ODEs: 4th-order central differences of the solved Cartesian fields (numpy
     evaluation at the fixed parameters) rebuild every residual: <= 1e-7 relative;
  f. the fixed parameter sets are evaluated in mpmath (40 digits): constants, Q = int w dA, L_theta = int r u_t dA,
     the wall current I = int_{r=r2} J.n ds, and point values, written to f13a_reference/mhd_annulus_reference.json; the fixed
     numpy closed forms in benchmarks/fluids/cases/f13a_annulus_check.py must agree with them (checked here).

    python3 -B tools/fluids_mms/f13a_annulus_fields.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
OUT = HERE / "f13a_reference" / "mhd_annulus_reference.json"   # not tools/fluids_mms/out: that is track scope

x, y = sp.symbols("x y", real=True)
r, th = sp.symbols("r theta", positive=True)
C, sig, mu, G, Ez, V = sp.symbols("C sigma mu G E_z V", positive=True)
r1, r2 = sp.symbols("r1 r2", positive=True)
R = sp.sqrt(x ** 2 + y ** 2)
report = {"steps": {}}


def cart_residuals(u, phi, p, B, rho=0):
    """The Cartesian components of the 2.5D equations (f = 0, q = 0)."""
    uxB = sp.Matrix([u[1] * B[2] - u[2] * B[1], u[2] * B[0] - u[0] * B[2], u[0] * B[1] - u[1] * B[0]])
    J = sig * sp.Matrix([-sp.diff(phi, x) + uxB[0], -sp.diff(phi, y) + uxB[1], Ez + uxB[2]])
    JxB = sp.Matrix([J[1] * B[2] - J[2] * B[1], J[2] * B[0] - J[0] * B[2], J[0] * B[1] - J[1] * B[0]])
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)  # noqa: E731
    conv = [rho * (u[0] * sp.diff(u[i], x) + u[1] * sp.diff(u[i], y)) for i in range(3)]
    gp = [sp.diff(p, x), sp.diff(p, y), 0]
    # residual = rho(u.grad)u - mu lap u + grad p - G e_z - J x B   (= 0)
    mom = [conv[i] - mu * lap(u[i]) + gp[i] - (G if i == 2 else 0) - JxB[i] for i in range(3)]
    cont = sp.diff(u[0], x) + sp.diff(u[1], y)
    charge = sp.diff(J[0], x) + sp.diff(J[1], y)
    return {"mom_x": mom[0], "mom_y": mom[1], "mom_z": mom[2], "continuity": cont, "charge": charge}, J


def to_polar(e):
    return sp.simplify(e.subs({x: r * sp.cos(th), y: r * sp.sin(th)}, simultaneous=True))


def check_field(Bc):
    divB = sp.simplify(sp.diff(Bc[0], x) + sp.diff(Bc[1], y))
    curlB = [sp.simplify(sp.diff(Bc[2], y)), sp.simplify(-sp.diff(Bc[2], x)),
             sp.simplify(sp.diff(Bc[1], x) - sp.diff(Bc[0], y))]
    assert divB == 0 and all(c == 0 for c in curlB), (divB, curlB)
    return {"div_B": str(divB), "curl_B": [str(c) for c in curlB]}


# ============================================================================================ 1. radial field
def derive_radial():
    ut, w, ph, pp = (sp.Function(n) for n in ("u_t", "w", "phi", "p"))
    Bc = sp.Matrix([C * x / R ** 2, C * y / R ** 2, 0])
    rep = {"field": check_field(Bc)}
    u = sp.Matrix([-ut(R) * y / R, ut(R) * x / R, w(R)])
    res, J = cart_residuals(u, ph(R), pp(R), Bc)
    # polar: radial / azimuthal momentum components
    pol = {k: to_polar(v) for k, v in res.items()}
    m_r = sp.simplify(to_polar(res["mom_x"] * x / R + res["mom_y"] * y / R))
    m_t = sp.simplify(to_polar(-res["mom_x"] * y / R + res["mom_y"] * x / R))
    # theta independence (step b)
    for name, e in (("mom_r", m_r), ("mom_t", m_t), ("mom_z", pol["mom_z"]), ("charge", pol["charge"]),
                    ("continuity", pol["continuity"])):
        assert sp.simplify(sp.diff(e, th)) == 0, name
    rep["odes"] = {"mom_r": str(m_r), "mom_t": str(m_t), "mom_z": str(pol["mom_z"]), "charge": str(pol["charge"]),
                   "continuity": str(pol["continuity"])}
    # charge: d/dr(r J_r) = 0 with J_r = -sigma phi'  -> with an insulating wall J_r = 0 everywhere; with phi = 0 on
    # both walls: phi = a ln r + b = 0.  Either way phi' = 0.
    Jr = sp.simplify(to_polar(J[0] * x / R + J[1] * y / R))
    rep["J_r"] = str(Jr)
    phisol = sp.dsolve(sp.Eq(pol["charge"], 0), ph(r)).rhs
    rep["phi_general"] = str(phisol)
    # z and theta momentum with phi' = 0
    # Euler equations: written with sigma = m^2 mu / C^2 and multiplied by r^2/mu (dsolve's Euler hints)
    m = sp.Symbol("m", positive=True)
    to_euler = lambda e: sp.expand(sp.simplify(e.subs(sp.Derivative(ph(r), r), 0).doit()  # noqa: E731
                                              .subs(sig, m ** 2 * mu / C ** 2) * r ** 2 / mu))
    ez = sp.Eq(to_euler(pol["mom_z"]), 0)
    wsol = sp.dsolve(ez, w(r), hint="nth_linear_euler_eq_nonhomogeneous_variation_of_parameters").rhs
    eth = sp.Eq(to_euler(m_t), 0)
    utsol = sp.dsolve(eth, ut(r), hint="nth_linear_euler_eq_nonhomogeneous_variation_of_parameters").rhs
    rep["euler_odes"] = {"z": str(ez.lhs), "theta": str(eth.lhs)}
    rep["w_general"], rep["u_t_general"] = str(wsol), str(utsol)
    # radial momentum: p' (rho = 0) from m_r
    rep["mom_r_with_solution"] = str(sp.simplify(m_r.subs(sp.Derivative(ph(r), r), 0).doit()))
    return Bc, rep


# ============================================================================================ 2. azimuthal field
def derive_azimuthal():
    w, ph, pp = (sp.Function(n) for n in ("w", "phi", "p"))
    Bc = sp.Matrix([-C * y / R ** 2, C * x / R ** 2, 0])
    rep = {"field": check_field(Bc)}
    u = sp.Matrix([0, 0, w(R)])
    res, J = cart_residuals(u, ph(R), pp(R), Bc)
    pol = {k: to_polar(v) for k, v in res.items()}
    m_r = sp.simplify(to_polar(res["mom_x"] * x / R + res["mom_y"] * y / R))
    m_t = sp.simplify(to_polar(-res["mom_x"] * y / R + res["mom_y"] * x / R))
    for name, e in (("mom_r", m_r), ("mom_t", m_t), ("mom_z", pol["mom_z"]), ("charge", pol["charge"])):
        assert sp.simplify(sp.diff(e, th)) == 0, name
    rep["odes"] = {"mom_r": str(m_r), "mom_t": str(m_t), "mom_z": str(pol["mom_z"]), "charge": str(pol["charge"])}
    Jr = sp.simplify(to_polar(J[0] * x / R + J[1] * y / R))
    rep["J_r"] = str(Jr)
    # charge: (r J_r)' = 0  ->  J_r = K/r  ->  phi' = -K/(sigma r) - C w / r
    K = sp.Symbol("K", real=True)
    dphi = sp.solve(sp.Eq(Jr, K / r), sp.Derivative(ph(r), r))[0]
    rep["phi_prime"] = str(dphi)
    ez = sp.simplify(pol["mom_z"].subs(sp.Derivative(ph(r), r), dphi).doit())
    wsol = sp.dsolve(sp.Eq(ez, 0), w(r)).rhs
    rep["w_general"] = str(wsol)
    rep["mom_r_with_solution"] = str(sp.simplify(m_r))
    return Bc, rep, K, wsol, dphi


# ============================================================================================ explicit closed forms
def radial_closed(Cv, sigv, muv, Gv, Ezv, r1v, r2v):
    """Explicit symbolic solution (exact rationals for the parameters; m, n may be irrational)."""
    m = Cv * sp.sqrt(sigv / muv)
    n = sp.sqrt(1 + m ** 2)
    c1, c2, d1, d2 = sp.symbols("c1 c2 d1 d2")
    A = Gv / (muv * (m ** 2 - 4))
    wr = A * r ** 2 + c1 * r ** m + c2 * r ** (-m)
    ur = Ezv / Cv * r + d1 * r ** n + d2 * r ** (-n)
    sc = sp.solve([wr.subs(r, r1v), wr.subs(r, r2v)], [c1, c2], dict=True)[0]
    sd = sp.solve([ur.subs(r, r1v), ur.subs(r, r2v)], [d1, d2], dict=True)[0]
    return wr.subs(sc), ur.subs(sd), m, n


def substitute_back(name, u, phi, p, Bc, params, pts):
    res, J = cart_residuals(u, phi, p, Bc)
    worst = {}
    for k, e in res.items():
        e = e.subs(params)
        vals = [abs(sp.N(e.subs({x: a, y: b}), 40)) for a, b in pts]
        worst[k] = float(max(vals))
    print(f"[{name}] Cartesian residuals at {len(pts)} rational points (40 digits):",
          {k: f"{v:.1e}" for k, v in worst.items()})
    assert max(worst.values()) < 1e-25, worst
    return worst, J


def fd_route(name, fu, fphi, fp, fB, consts, pts, h=1e-3):
    """Second route: 4th-order central differences of the solved fields (numpy), no ODE, no sympy."""
    muv, sigv, Gv, Ezv = consts

    def d(f, X, Y, ax, k=1):
        e = np.array([h, 0.0]) if ax == 0 else np.array([0.0, h])
        if k == 1:
            return (-f(X + 2 * e[0], Y + 2 * e[1]) + 8 * f(X + e[0], Y + e[1]) - 8 * f(X - e[0], Y - e[1])
                    + f(X - 2 * e[0], Y - 2 * e[1])) / (12 * h)
        return (-f(X + 2 * e[0], Y + 2 * e[1]) + 16 * f(X + e[0], Y + e[1]) - 30 * f(X, Y)
                + 16 * f(X - e[0], Y - e[1]) - f(X - 2 * e[0], Y - 2 * e[1])) / (12 * h * h)

    worst = 0.0
    for X, Y in pts:
        X, Y = float(X), float(Y)
        uv = fu(X, Y)
        Bv = fB(X, Y)
        gph = np.array([d(fphi, X, Y, 0), d(fphi, X, Y, 1)])
        uxB = np.cross(uv, Bv)
        J = sigv * np.array([-gph[0] + uxB[0], -gph[1] + uxB[1], Ezv + uxB[2]])
        F = np.cross(J, Bv)
        mom = []
        for i in range(3):
            fi = lambda A, B_, i=i: fu(A, B_)[i]  # noqa: E731
            lap = d(fi, X, Y, 0, 2) + d(fi, X, Y, 1, 2)
            gp = [d(fp, X, Y, 0), d(fp, X, Y, 1), 0.0][i]
            mom.append(-muv * lap + gp - (Gv if i == 2 else 0.0) - F[i])
        Jx = lambda A, B_: sigv * (-(d(fphi, A, B_, 0)) + np.cross(fu(A, B_), fB(A, B_))[0])  # noqa: E731
        Jy = lambda A, B_: sigv * (-(d(fphi, A, B_, 1)) + np.cross(fu(A, B_), fB(A, B_))[1])  # noqa: E731
        charge = d(Jx, X, Y, 0) + d(Jy, X, Y, 1)
        scale = max(muv * 1.0, abs(Gv), float(np.max(np.abs(F))), 1e-300)
        worst = max(worst, max(abs(v) for v in mom) / scale, abs(charge) / max(float(np.max(np.abs(J))), 1e-300))
    print(f"[{name}] second route (FD, no ODE): max relative residual {worst:.1e}")
    assert worst < 1e-7, worst
    return worst


def main():
    pts = [(sp.Rational(7, 5), sp.Rational(3, 7)), (sp.Rational(-11, 10), sp.Rational(9, 13)),
           (sp.Rational(1, 3), sp.Rational(-17, 10)), (sp.Rational(-6, 5), sp.Rational(-6, 5))]
    # ---------------------------------------------------------------------------------------- derivations
    Bc_r, rep_r = derive_radial()
    print("[radial] ODEs:", rep_r["odes"])
    print("[radial] w general:", rep_r["w_general"], " u_t general:", rep_r["u_t_general"])
    Bc_a, rep_a, K, wsol_a, dphi_a = derive_azimuthal()
    print("[azimuthal] ODEs:", rep_a["odes"])
    print("[azimuthal] w general:", rep_a["w_general"], " phi':", rep_a["phi_prime"])
    report["steps"]["radial_derivation"] = rep_r
    report["steps"]["azimuthal_derivation"] = rep_a

    import f13a_annulus_check as AC   # the fixed numpy closed forms under test
    P = AC.PARAMS
    out = {"params": {}, "radial": {}, "azimuthal": {}}
    r1v, r2v = sp.Integer(1), sp.Integer(2)
    muv, sigv, Gv = sp.Rational(1, 2), sp.Integer(2), sp.Integer(1)
    assert (float(muv), float(sigv), float(Gv), 1.0, 2.0) == (P["mu"], P["sigma"], P["G"], P["r1"], P["r2"])
    # ---------------------------------------------------------------------------------------- radial variants
    for key, var in AC.VARIANTS.items():
        if var["field"] != "radial":
            continue
        Cv = sp.nsimplify(var["C"])
        Ezv = sp.nsimplify(var["E_z"])
        wr, ur, m, n = radial_closed(Cv, sigv, muv, Gv, Ezv, r1v, r2v)
        Rr = sp.sqrt(x ** 2 + y ** 2)
        u = sp.Matrix([-ur.subs(r, Rr) * y / Rr, ur.subs(r, Rr) * x / Rr, wr.subs(r, Rr)])
        params = {C: Cv, sig: sigv, mu: muv, G: Gv, Ez: Ezv}
        worst, J = substitute_back(key, u, sp.Integer(0), sp.Integer(0), Bc_r.subs(params), params,
                                   [(a, b) for a, b in pts])
        # wall conditions
        Jn = sp.simplify((J[0] * x + J[1] * y) / Rr)
        assert Jn == 0, Jn
        mp.mp.dps = 40
        wf = sp.lambdify(r, wr, "mpmath")
        uf = sp.lambdify(r, ur, "mpmath")
        Q = 2 * mp.pi * mp.quad(lambda s: wf(s) * s, [1, 1.25, 1.5, 1.75, 2])
        Lt = 2 * mp.pi * mp.quad(lambda s: uf(s) * s ** 2, [1, 1.25, 1.5, 1.75, 2])
        Qs = sp.N(2 * sp.pi * sp.integrate(sp.expand(wr * r), (r, 1, 2)), 30)
        out["radial"][key] = {"m": str(sp.N(m, 30)), "n": str(sp.N(n, 30)), "Q": str(Q), "L_theta": str(Lt),
                              "Q_symbolic_integral": str(Qs),
                              "w_mid": str(wf(mp.mpf("1.5"))), "u_t_mid": str(uf(mp.mpf("1.5"))),
                              "residuals_40digit": worst, "J_dot_n_on_circles": str(Jn)}
        # fixed numpy closed form vs mpmath
        ex = AC.exact(key)
        rs = np.linspace(1.0, 2.0, 41)
        dw = max(abs(float(wf(mp.mpf(float(s)))) - float(ex.w_r(np.array([s]))[0])) for s in rs)
        du = max(abs(float(uf(mp.mpf(float(s)))) - float(ex.ut_r(np.array([s]))[0])) for s in rs)
        dQ = abs(float(Q) - ex.Q) / abs(float(Q))
        dL = abs(float(Lt) - ex.L_theta) / abs(float(Lt))
        sw = max(abs(float(wf(mp.mpf(float(s))))) for s in rs)
        su = max(abs(float(uf(mp.mpf(float(s))))) for s in rs)
        print(f"[{key}] m = {float(m):.6g}: numpy closed form vs 40-digit: w {dw / sw:.1e}, u_t {du / su:.1e}, "
              f"Q {dQ:.1e}, L {dL:.1e}  (Q = {float(Q):.15g}, L_theta = {float(Lt):.15g})")
        assert dw / sw < 1e-13 and du / su < 1e-13 and dQ < 1e-13 and dL < 1e-13
        out["radial"][key]["numpy_vs_mpmath"] = {"w": dw / sw, "u_t": du / su, "Q": dQ, "L_theta": dL}
        # second route
        def fu(X, Y, ex=ex):
            return ex.u(np.array(X), np.array(Y))
        fd = fd_route(key, fu, lambda X, Y: 0.0 * X, lambda X, Y: 0.0 * X,
                      lambda X, Y, ex=ex: ex.B(np.array(X), np.array(Y)),
                      (P["mu"], P["sigma"], P["G"], var["E_z"]), [(1.3, 0.4), (-0.9, 0.8), (0.2, -1.7)])
        out["radial"][key]["fd_route"] = fd
    # ---------------------------------------------------------------------------------------- azimuthal variant
    for key, var in AC.VARIANTS.items():
        if var["field"] != "azimuthal":
            continue
        Cv, Ezv = (sp.nsimplify(var[k]) for k in ("C", "E_z"))
        consts = sorted(wsol_a.free_symbols - {r, C, sig, mu, G, K}, key=str)
        wgen = wsol_a.subs({C: Cv, sig: sigv, mu: muv, G: Gv})
        if var["potential"] == "insulating":
            # insulating walls: J_r = K/r = 0 there, so K = 0; inner wall slides axially, w(r1) = W, w(r2) = 0
            Wv = sp.nsimplify(var["W"])
            s = sp.solve([wgen.subs(r, r1v) - Wv, wgen.subs(r, r2v)], consts, dict=True)[0]
            wK = sp.expand(wgen.subs(s))
            dphK = sp.expand(-K / (sigv * r) - Cv * wK / r)
            phiK = sp.integrate(dphK, (r, r1v, r))
            Ksol = sp.Integer(0)
        else:
            Vv = sp.nsimplify(var["V"])
            s = sp.solve([wgen.subs(r, r1v), wgen.subs(r, r2v)], consts, dict=True)[0]
            wK = sp.expand(wgen.subs(s))
            dphK = sp.expand(-K / (sigv * r) - Cv * wK / r)
            phiK = sp.integrate(dphK, (r, r1v, r))
            Ksol = sp.solve(sp.Eq(phiK.subs(r, r2v), Vv), K)[0]
        wa = sp.simplify(wK.subs(K, Ksol))
        pha = sp.simplify(phiK.subs(K, Ksol))
        pa = -sigv * Ezv * Cv * sp.log(r)
        Rr = sp.sqrt(x ** 2 + y ** 2)
        u = sp.Matrix([0, 0, wa.subs(r, Rr)])
        params = {C: Cv, sig: sigv, mu: muv, G: Gv, Ez: Ezv}
        worst, J = substitute_back(key, u, pha.subs(r, Rr), pa.subs(r, Rr), Bc_a.subs(params), params, pts)
        assert sp.simplify(pha.subs(r, 1)) == 0
        if var["potential"] == "insulating":
            assert sp.simplify(wa.subs(r, 1) - Wv) == 0 and sp.simplify(wa.subs(r, 2)) == 0
            for rw in (1, 2):    # J.n = 0 on both insulating walls, while (u x B).n != 0 on the sliding wall
                assert sp.simplify(((J[0] * x + J[1] * y) / Rr).subs(params).subs({x: rw, y: 0})) == 0
            uxBn = sp.simplify((-wa * Cv / r).subs(r, 1))
            assert uxBn != 0
            print(f"[{key}] insulating walls: J.n = 0 on both; (u x B).n = {uxBn} on the sliding inner wall")
        else:
            assert sp.simplify(pha.subs(r, 2) - Vv) == 0
        Jn2 = sp.simplify(((J[0] * x + J[1] * y) / Rr).subs(params).subs({x: 2, y: 0}))
        mp.mp.dps = 40
        wf = sp.lambdify(r, wa, "mpmath")
        phf = sp.lambdify(r, pha, "mpmath")
        Q = 2 * mp.pi * mp.quad(lambda s_: wf(s_) * s_, [1, 1.5, 2])
        I2 = mp.mpf(str(sp.N(2 * sp.pi * 2 * Jn2, 40)))
        out["azimuthal"][key] = {"K": str(sp.N(Ksol, 30)), "Q": str(Q), "I_wall": str(I2),
                                 "w_mid": str(wf(mp.mpf("1.5"))), "phi_mid": str(phf(mp.mpf("1.5"))),
                                 "residuals_40digit": worst, "w_closed": str(wa), "phi_closed": str(pha)}
        ex = AC.exact(key)
        rs = np.linspace(1.0, 2.0, 41)
        dw = max(abs(float(wf(mp.mpf(float(s_)))) - float(ex.w_r(np.array([s_]))[0])) for s_ in rs)
        dp = max(abs(float(phf(mp.mpf(float(s_)))) - float(ex.phi_r(np.array([s_]))[0])) for s_ in rs)
        sw = max(abs(float(wf(mp.mpf(float(s_))))) for s_ in rs)
        sp_ = max(abs(float(phf(mp.mpf(float(s_))))) for s_ in rs)
        dQ = abs(float(Q) - ex.Q) / abs(float(Q))
        dI = abs(float(I2) - ex.I_wall) / max(abs(float(I2)), 1e-300) if float(I2) != 0 else abs(ex.I_wall)
        dK = (abs(float(sp.N(Ksol, 30)) - ex.K) / abs(float(sp.N(Ksol, 30)))) if Ksol != 0 else abs(ex.K)
        print(f"[{key}] K = {float(sp.N(Ksol)):.15g}: numpy vs 40-digit: w {dw / sw:.1e}, phi {dp / sp_:.1e}, "
              f"Q {dQ:.1e}, I {dI:.1e}, K {dK:.1e}  (Q = {float(Q):.15g}, I_wall = {float(I2):.15g})")
        assert max(dw / sw, dp / sp_, dQ, dI, dK) < 1e-13
        out["azimuthal"][key]["numpy_vs_mpmath"] = {"w": dw / sw, "phi": dp / sp_, "Q": dQ, "I_wall": dI, "K": dK}
        fd = fd_route(key, lambda X, Y, ex=ex: ex.u(np.array(X), np.array(Y)),
                      lambda X, Y, ex=ex: ex.phi(np.array(X), np.array(Y)),
                      lambda X, Y, ex=ex: ex.p(np.array(X), np.array(Y)),
                      lambda X, Y, ex=ex: ex.B(np.array(X), np.array(Y)),
                      (P["mu"], P["sigma"], P["G"], var["E_z"]), [(1.3, 0.4), (-0.9, 0.8), (0.2, -1.7)])
        out["azimuthal"][key]["fd_route"] = fd
    out["params"] = {"mu": P["mu"], "sigma": P["sigma"], "G": P["G"], "r1": P["r1"], "r2": P["r2"],
                     "variants": AC.VARIANTS}
    out["derivation"] = report["steps"]
    out["check_module_sha256"] = hashlib.sha256((ROOT / "benchmarks/fluids/cases/f13a_annulus_check.py").read_bytes()).hexdigest()
    out["sympy"], out["mpmath"], out["numpy"] = sp.__version__, mp.__version__, np.__version__
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("wrote", OUT.relative_to(ROOT), hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
