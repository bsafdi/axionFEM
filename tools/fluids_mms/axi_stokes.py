"""Axisymmetric Stokes with swirl: operators, code generation and self-checks for the F1b verification fields.

Written from the formulation (docs/guides/fluids.md), NOT from solver code.  Meridional half plane (r, z), r >= 0; unknowns (u_r, u_z, u_theta) and p; no theta
dependence.  The 3D velocity is U = u_r e_r + u_z e_z + u_theta e_theta.

Velocity gradient (physical components, rows = component, columns = derivative direction, order (r, z, theta)):

    grad u = [[d_r u_r,  d_z u_r,  -u_t / r],
              [d_r u_z,  d_z u_z,   0      ],
              [d_r u_t,  d_z u_t,   u_r / r]]

    sigma_sym = 2 mu eps(u) - p I,  eps = (grad u + grad u^T) / 2          symmetric-stress form ("sym")
    sigma_lap =   mu grad u  - p I                                           Laplacian (gradient) form ("lap")
    -div sigma_form - gamma grad(div u) = f,   div u = d_r u_r + u_r / r + d_z u_z = 0

Weak form (weight r):
    a_lap(u, v) = int mu [grad u_r . grad v_r + u_r v_r / r^2 + grad u_z . grad v_z + grad u_t . grad v_t
                          + u_t v_t / r^2] r dr dz
    a_sym(u, v) = int 2 mu [e_rr e'_rr + e_zz e'_zz + e_tt e'_tt + 2 e_rz e'_rz + 2 e_rt e'_rt + 2 e_zt e'_zt] r dr dz
        e_rr = d_r u_r, e_zz = d_z u_z, e_tt = u_r / r, e_rz = (d_z u_r + d_r u_z)/2, e_rt = (d_r u_t - u_t/r)/2,
        e_zt = d_z u_t / 2
    b(v, q) = - int q (d_r v_r + v_r / r + d_z v_z) r dr dz       (continuity tested with r q: symmetric system)
    a_form(u, v) + b(v, p) + gamma (div u, div v)_r = (f, v)_r + <g, v>_{r, Gamma_N},   b(u, q) = 0
Natural condition of each form: sigma_form(u, p) n = g on Gamma_N (n outward, (sigma n)_i = sigma_ij n_j).
Axis r = 0: u_r = u_t = 0 (strong), u_z and p natural.

Three independent derivations of the forcing (self-check 2):
  route A  cylindrical tensor calculus with the axisymmetric divergence of a (non-symmetric) tensor;
  route B  the 3D Cartesian field U(X, Y, Z), its Cartesian stress and divergence, projected on (e_r, e_z, e_theta)
           at Y = 0, X = r > 0 (no cylindrical formula used);
  route C  the WEAK form above against parity-respecting test functions vanishing on the non-axis boundary
           (validates the r-weights and hoop terms of the weak form itself).
Emitted expressions use the variables r and z (three spellings: sympy, c, numpy).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

r, z = sp.symbols("r z", real=True)
s_ = sp.Symbol("s", real=True)         # s = r^2 (parity: functions of s are even in r)
FORMS = ("lap", "sym")
COMP = ("r", "z", "t")                 # component order everywhere: (u_r, u_z, u_theta)


# ------------------------------------------------------------------------------------------- operators (route A)
def grad_cyl(u):
    """3x3 physical-component velocity gradient, order (r, z, theta)."""
    ur, uz, ut = u
    return sp.Matrix([[sp.diff(ur, r), sp.diff(ur, z), -ut / r],
                      [sp.diff(uz, r), sp.diff(uz, z), 0],
                      [sp.diff(ut, r), sp.diff(ut, z), ur / r]])


def div_cyl(u):
    return sp.diff(u[0], r) + u[0] / r + sp.diff(u[1], z)


def stress(u, p, mu, form):
    Gm = grad_cyl(u)
    if form == "sym":
        return mu * (Gm + Gm.T) - p * sp.eye(3)
    if form == "lap":
        return mu * Gm - p * sp.eye(3)
    raise ValueError(form)


def div_tensor_cyl(T):
    """(div T)_i = d_j T_ij for an axisymmetric (possibly non-symmetric) tensor, physical components (r, z, t)."""
    rr, rz, rt = T[0, 0], T[0, 1], T[0, 2]
    zr, zz = T[1, 0], T[1, 1]
    tr, tz, tt = T[2, 0], T[2, 1], T[2, 2]
    return sp.Matrix([sp.diff(rr, r) + sp.diff(rz, z) + (rr - tt) / r,
                      sp.diff(zr, r) + sp.diff(zz, z) + zr / r,
                      sp.diff(tr, r) + sp.diff(tz, z) + (tr + rt) / r])


def forcing_route_a(u, p, mu, form, gamma=0):
    dv = div_cyl(u)
    return -div_tensor_cyl(stress(u, p, mu, form)) - gamma * sp.Matrix([sp.diff(dv, r), sp.diff(dv, z), 0])


# ------------------------------------------------------------------------------------------- route B: 3D Cartesian
Xc, Yc, Zc = sp.symbols("X Y Z", real=True)
Xp = sp.Symbol("Xp", positive=True)


def forcing_route_b(u, p, mu, form, gamma=0):
    """Cartesian 3D: U = u_r (X/R, Y/R, 0) + u_t (-Y/R, X/R, 0) + u_z e_Z, R = sqrt(X^2 + Y^2); sigma and its
    divergence in Cartesian components; then Y = 0, X = r > 0, where e_r = e_X, e_theta = e_Y.  Returns (f_r, f_z,
    f_t) as expressions in (r, z)."""
    R = sp.sqrt(Xc ** 2 + Yc ** 2)
    sub = {r: R, z: Zc}
    ur, uz, ut = [sp.sympify(c).subs(sub, simultaneous=True) for c in u]
    P = sp.sympify(p).subs(sub, simultaneous=True)
    M = sp.sympify(mu).subs(sub, simultaneous=True)
    U = [ur * Xc / R - ut * Yc / R, ur * Yc / R + ut * Xc / R, uz]
    X3 = (Xc, Yc, Zc)
    G = sp.Matrix(3, 3, lambda i, j: sp.diff(U[i], X3[j]))
    if form == "sym":
        S = M * (G + G.T) - P * sp.eye(3)
    else:
        S = M * G - P * sp.eye(3)
    dv = sum(G[i, i] for i in range(3))
    f = [-sum(sp.diff(S[i, j], X3[j]) for j in range(3)) - gamma * sp.diff(dv, X3[i]) for i in range(3)]
    at = {Yc: 0}
    fr, ft, fz = [sp.sympify(c).subs(at).subs(Xc, Xp) for c in f]   # (X, Y, Z) components at Y = 0
    back = {Xp: r, Zc: z}
    return sp.Matrix([fr.subs(back, simultaneous=True), fz.subs(back, simultaneous=True),
                      ft.subs(back, simultaneous=True)])


# ------------------------------------------------------------------------------------------- weak form (route C)
def weak_form_density(u, p, mu, form, v, gamma=0):
    """Integrand (WITHOUT the weight r) of a_form(u, v) + b(v, p) + gamma (div u, div v) as written in the module
    docstring; v a 3-vector of expressions."""
    ur, uz, ut = u
    vr, vz, vt = v
    d = lambda f, w: sp.diff(f, w)  # noqa: E731
    if form == "lap":
        a = mu * (d(ur, r) * d(vr, r) + d(ur, z) * d(vr, z) + ur * vr / r ** 2 + d(uz, r) * d(vz, r)
                  + d(uz, z) * d(vz, z) + d(ut, r) * d(vt, r) + d(ut, z) * d(vt, z) + ut * vt / r ** 2)
    else:
        e = lambda w: (d(w[0], r), d(w[1], z), w[0] / r, (d(w[0], z) + d(w[1], r)) / 2,  # noqa: E731
                       (d(w[2], r) - w[2] / r) / 2, d(w[2], z) / 2)
        eu, ev = e(u), e(v)
        a = 2 * mu * (eu[0] * ev[0] + eu[1] * ev[1] + eu[2] * ev[2] + 2 * eu[3] * ev[3] + 2 * eu[4] * ev[4]
                      + 2 * eu[5] * ev[5])
    divv = d(vr, r) + vr / r + d(vz, z)
    divu = d(ur, r) + ur / r + d(uz, z)
    return a - p * divv + gamma * divu * divv


# ------------------------------------------------------------------------------------------- fields
@dataclass
class Boundary:
    name: str                 # "axis", "r1", "z0", "z1", "sphere", "outer", ...
    bc: str                   # dirichlet | traction | do_nothing | axis
    normal: tuple             # outward unit normal (n_r, n_z) as sympy expressions
    param: dict               # {"t": symbol, "r": expr, "z": expr, "range": [a, b]}
    notes: str = ""


@dataclass
class Field:
    name: str
    description: str
    domain: dict
    u: sp.Matrix              # (u_r, u_z, u_t)
    p: sp.Expr
    mu: sp.Expr
    forms: tuple
    boundaries: list
    pressure: str             # mean_zero | determined
    integrate: Callable       # integrate(fn(R, Z)) over the meridional domain, plain dr dz (no weight)
    sample: Callable          # () -> (R, Z) interior sample points
    extra: dict = field(default_factory=dict)
    parity: bool = True       # u_r, u_t odd and u_z, p, mu even in r (False only for the negative-control field)
    exact_solution: bool = False   # closed-form solutions: f must be identically zero
    checks: dict = field(default_factory=dict)


def gauss_rect(r0, r1, z0, z1, n=48):
    g, w = np.polynomial.legendre.leggauss(n)
    rs = 0.5 * (r1 - r0) * (g + 1) + r0
    zs = 0.5 * (z1 - z0) * (g + 1) + z0
    W = np.outer(w, w) * 0.25 * (r1 - r0) * (z1 - z0)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    return lambda fn: float(np.sum(W * fn(RR, ZZ)))


def gauss_half_annulus(a, b, nr=64, nt=128):
    """{a <= rho <= b, 0 <= theta <= pi} in the meridional plane, r = rho sin(theta), z = rho cos(theta)."""
    g, w = np.polynomial.legendre.leggauss(nr)
    rho = 0.5 * (b - a) * (g + 1) + a
    wr = 0.5 * (b - a) * w
    gt, wt = np.polynomial.legendre.leggauss(nt)
    th = 0.5 * np.pi * (gt + 1)
    wth = 0.5 * np.pi * wt
    RH, TH = np.meshgrid(rho, th, indexing="ij")
    W = np.outer(wr, wth) * RH                       # dr dz = rho drho dtheta
    RR, ZZ = RH * np.sin(TH), RH * np.cos(TH)
    return lambda fn: float(np.sum(W * fn(RR, ZZ)))


def sample_rect(r0, r1, z0, z1, n=41, margin=1e-3):
    rs = np.linspace(r0 + margin, r1 - margin, n)
    zs = np.linspace(z0 + margin, z1 - margin, n)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    return lambda: (RR.ravel(), ZZ.ravel())


def sample_half_annulus(a, b, n=41, margin=1e-3):
    rho = np.linspace(a + margin, b - margin, n)
    th = np.linspace(margin, np.pi - margin, 2 * n)
    RH, TH = np.meshgrid(rho, th, indexing="ij")
    return lambda: ((RH * np.sin(TH)).ravel(), (RH * np.cos(TH)).ravel())


# ------------------------------------------------------------------------------------------- code generation
def _np_lambda(expr):
    f = sp.lambdify((r, z), expr, modules="numpy")
    return lambda R_, Z_: np.broadcast_to(np.asarray(f(R_, Z_), dtype=float), np.shape(R_)).copy()


def spell(expr):
    expr = sp.sympify(expr)
    return {"sympy": str(expr), "c": sp.ccode(expr),
            "numpy": sp.pycode(expr, fully_qualified_modules=True).replace("math.", "np.")}


def emit(fld: Field) -> dict:
    u, p, mu = fld.u, fld.p, fld.mu
    Gm = grad_cyl(u)
    out = {"name": fld.name, "description": fld.description, "domain": fld.domain, "pressure": fld.pressure,
           "conventions": {
               "coordinates": "meridional (r, z), r >= 0; variables r, z in every spelling",
               "components": "(u_r, u_z, u_theta); grad_u[i][j] = d u_i / d (r, z)_j (meridional derivatives only)",
               "sigma": "3x3, rows/columns (r, z, theta); sigma_sym = 2 mu eps - p I, sigma_lap = mu grad u - p I with"
                        " grad u including the hoop entries (r,theta) = -u_t/r and (theta,theta) = u_r/r",
               "f": "-div sigma_form - gamma grad div u (= -div sigma_form for these divergence-free fields)",
               "traction": "g = sigma_form n, (sigma n)_i = sigma_ij n_j, n the outward unit normal (n_r, n_z, 0)",
               "weights": "volume integrals int (.) r dr dz, boundary integrals int (.) r ds; physical 3D values "
                          "carry a further factor 2 pi"},
           "u": [spell(c) for c in u], "p": spell(p), "mu": spell(mu),
           "grad_u": [[spell(sp.diff(u[i], v)) for v in (r, z)] for i in range(3)],
           "forms": {}, "boundaries": [], "extra": {}}
    for form in fld.forms:
        S = stress(u, p, mu, form)
        f = sp.zeros(3, 1) if fld.exact_solution else forcing_route_a(u, p, mu, form)
        out["forms"][form] = {"f": [spell(sp.simplify(c) if fld.exact_solution else c) for c in f],
                              "sigma": [[spell(S[i, j]) for j in range(3)] for i in range(3)]}
    for b in fld.boundaries:
        bj = {"name": b.name, "bc": b.bc, "normal": [spell(c) for c in b.normal], "notes": b.notes,
              "param": {k: (spell(v) if isinstance(v, sp.Basic) else v) for k, v in b.param.items() if k != "t"}}
        if b.bc in ("traction", "do_nothing"):
            n = sp.Matrix([b.normal[0], b.normal[1], 0])
            bj["traction"] = {form: [spell(c) for c in stress(u, p, mu, form) * n] for form in fld.forms}
        out["boundaries"].append(bj)
    nu_ = [_np_lambda(c) for c in u]
    np_ = _np_lambda(p)
    ngu = [[_np_lambda(sp.diff(u[i], v)) for v in (r, z)] for i in range(3)]
    I = fld.integrate
    area_r = I(lambda R_, Z_: R_)
    area = I(lambda R_, Z_: np.ones_like(R_))
    pmean_r = I(lambda R_, Z_: np_(R_, Z_) * R_) / area_r
    pmean = I(np_) / area
    const = {
        "area": area, "area_r": area_r, "p_mean": pmean, "p_mean_r": pmean_r,
        "u_L2r": math.sqrt(I(lambda R_, Z_: sum(nu_[i](R_, Z_) ** 2 for i in range(3)) * R_)),
        "u_L2": math.sqrt(I(lambda R_, Z_: sum(nu_[i](R_, Z_) ** 2 for i in range(3)))),
        "ut_L2r": math.sqrt(I(lambda R_, Z_: nu_[2](R_, Z_) ** 2 * R_)),
        "grad_u_L2r": math.sqrt(I(lambda R_, Z_: (sum(ngu[i][j](R_, Z_) ** 2 for i in range(3) for j in range(2))
                                                  + (nu_[0](R_, Z_) ** 2 + nu_[2](R_, Z_) ** 2) / R_ ** 2) * R_)),
        "p_minus_mean_L2r": math.sqrt(I(lambda R_, Z_: (np_(R_, Z_) - pmean_r) ** 2 * R_)),
    }
    xs, ys = fld.sample()
    U = [nu_[i](xs, ys) for i in range(3)]
    const["u_max_sampled"] = float(np.max(np.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)))
    const["ut_max_sampled"] = float(np.max(np.abs(U[2])))
    const["p_max_abs_sampled"] = float(np.max(np.abs(np_(xs, ys))))
    const["p_minus_mean_r_max_abs_sampled"] = float(np.max(np.abs(np_(xs, ys) - pmean_r)))
    const["p_minus_mean_max_abs_sampled"] = float(np.max(np.abs(np_(xs, ys) - pmean)))
    const["mu_min_sampled"] = float(np.min(_np_lambda(mu)(xs, ys)))
    const["mu_max_sampled"] = float(np.max(_np_lambda(mu)(xs, ys)))
    out["constants"] = const
    out["extra"] = {k: (spell(v) if isinstance(v, sp.Basic) else v) for k, v in fld.extra.items()}
    return out


# ------------------------------------------------------------------------------------------- self-checks
class CheckFailed(AssertionError):
    pass


def _mp_eval(e, syms, pt, dps=60):
    import mpmath
    mpmath.mp.dps = dps
    return complex(sp.N(e.subs(dict(zip(syms, pt))), dps))


def _is_zero(expr, pts, tol=1e-40, symbolic=False):
    expr = sp.sympify(expr)
    if expr == 0:
        return True
    if symbolic:
        try:
            if sp.simplify(expr) == 0:
                return True
        except Exception:  # pragma: no cover
            pass
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    for pt in pts:
        vals = [pt[s.name] for s in syms]
        if abs(_mp_eval(expr, syms, vals)) > tol:
            return False
    return True


def _rand_pts(fld, k=6, seed=1):
    rng = random.Random(seed)
    xs, ys = fld.sample()
    idx = [rng.randrange(len(xs)) for _ in range(k)]
    return [{"r": sp.Rational(str(round(float(xs[i]), 6))), "z": sp.Rational(str(round(float(ys[i]), 6)))}
            for i in idx]


def _c_eval(code, R_, Z_):
    py = (code.replace("pow(", "_pow(").replace("exp(", "_m.exp(").replace("sin(", "_m.sin(")
          .replace("cos(", "_m.cos(").replace("sqrt(", "_m.sqrt(").replace("log(", "_m.log(")
          .replace("tanh(", "_m.tanh(").replace("sinh(", "_m.sinh(").replace("cosh(", "_m.cosh(")
          .replace("atan2(", "_m.atan2(").replace("M_PI", "_m.pi"))
    py = py.replace("_m._m.", "_m.").replace("_m.a_m.", "_m.a")
    return eval(py, {"_m": math, "_pow": math.pow}, {"r": R_, "z": Z_})


def fd_residual(fld, form, pts, h=1e-3):
    """3D Cartesian finite-difference residual of the PDE, built from u, p and mu only (NOT from f): at each point
    (r, 0, z) (r > 0), sigma_ij by 4th-order central differences of the Cartesian velocity U(X, Y, Z), then
    d_j sigma_ij by 4th-order differences of sigma.  Returns max |-div sigma - f_emitted| / max |f| over pts, with f
    the emitted forcing of `form`."""
    un = [_np_lambda(c) for c in fld.u]
    pn = _np_lambda(fld.p)
    mn = _np_lambda(fld.mu)

    def U(P):
        X_, Y_, Z_ = P
        R_ = np.sqrt(X_ ** 2 + Y_ ** 2)
        ur, uz, ut = un[0](R_, Z_), un[1](R_, Z_), un[2](R_, Z_)
        return np.array([ur * X_ / R_ - ut * Y_ / R_, ur * Y_ / R_ + ut * X_ / R_, uz])

    def d4(fn, P, j, hh):
        e = np.zeros(3)
        e[j] = hh
        return (-fn(P + 2 * e) + 8 * fn(P + e) - 8 * fn(P - e) + fn(P - 2 * e)) / (12 * hh)

    def sigma(P):
        Gm = np.stack([d4(U, P, j, h) for j in range(3)], axis=-1)       # G[i, j] = dU_i/dX_j
        R_ = np.hypot(P[0], P[1])
        m, pp = mn(R_, P[2]), pn(R_, P[2])
        return (m * (Gm + Gm.T) if form == "sym" else m * Gm) - pp * np.eye(3)

    em_f = [_np_lambda(c) for c in forcing_route_a(fld.u, fld.p, fld.mu, form)] if not fld.exact_solution else None
    worst, scale = 0.0, 0.0
    for (rr, zz) in pts:
        P = np.array([rr, 0.0, zz])
        divs = np.zeros(3)
        for j in range(3):
            divs += d4(lambda Q: sigma(Q)[:, j], P, j, h)
        f_fd = -divs                                                      # Cartesian (X, Y, Z) at Y = 0
        f_cyl = np.array([f_fd[0], f_fd[2], f_fd[1]])                     # (r, z, theta)
        fe = np.zeros(3) if em_f is None else np.array([em_f[i](rr, zz) for i in range(3)], float)
        worst = max(worst, float(np.max(np.abs(f_cyl - fe))))
        scale = max(scale, float(np.max(np.abs(fe))), float(np.max(np.abs(f_cyl))))
    return worst / max(scale, 1e-300), scale


def parity_ok(expr, odd):
    e = sp.sympify(expr)
    d = sp.simplify(e.subs(r, -r) + e) if odd else sp.simplify(e.subs(r, -r) - e)
    return d == 0


def weak_residual(fld, form, nq=40, n_test=3, seed=7):
    """Route C: int [weak density(u, p; v) - f . v] r dr dz for parity-respecting test functions v that vanish on
    every non-axis boundary of a rectangle domain [r0, r1] x [z0, z1] (here r0 = 0).  Returns the max relative
    residual over the test functions and a 'teeth' value (the other form's f, which must give an O(1) residual for
    variable mu)."""
    dom = fld.domain
    r0, r1 = dom["r"]
    z0, z1 = dom["z"]
    rng = random.Random(seed)
    bump = (r1 ** 2 - r ** 2) * (z - z0) * (z1 - z)
    tests = []
    for _ in range(n_test):
        a, b, c = [sp.Rational(rng.randint(1, 9), 7) for _ in range(3)]
        tests.append(sp.Matrix([r * bump * sp.cos(a * r ** 2 + b * z), bump * sp.sin(c * r ** 2 + a * z + 1),
                                r * bump * sp.cos(b * r ** 2 - c * z + sp.Rational(1, 3))]))
    g, w = np.polynomial.legendre.leggauss(nq)
    rs = 0.5 * (r1 - r0) * (g + 1) + r0
    zs = 0.5 * (z1 - z0) * (g + 1) + z0
    W = np.outer(w, w) * 0.25 * (r1 - r0) * (z1 - z0)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    out = {}
    for which in (form, [f_ for f_ in FORMS if f_ != form][0]):
        f = forcing_route_a(fld.u, fld.p, fld.mu, which)
        worst = 0.0
        for v in tests:
            dens = weak_form_density(fld.u, fld.p, fld.mu, form, v) - sum(f[i] * v[i] for i in range(3))
            fn = _np_lambda(dens * r)
            ref = _np_lambda(sum(abs(f[i] * v[i]) for i in range(3)) * r)
            val = float(np.sum(W * fn(RR, ZZ)))
            sc = float(np.sum(W * ref(RR, ZZ)))
            worst = max(worst, abs(val) / sc)
        out[which] = worst
    return out[form], out[[f_ for f_ in FORMS if f_ != form][0]]


def selfcheck(fld: Field, emitted: dict, log=print, heavy=True) -> dict:
    rep = {}

    def need(cond, msg):
        if not cond:
            raise CheckFailed(f"[{fld.name}] {msg}")
        log(f"  ok  {msg}")

    u, p, mu = fld.u, fld.p, fld.mu
    pts = _rand_pts(fld)
    # 1. divergence free
    need(_is_zero(div_cyl(u), pts, symbolic=True), "div u = d_r u_r + u_r/r + d_z u_z = 0")
    # 2. parity (the axis regularity of a smooth 3D field)
    if fld.parity:
        need(parity_ok(u[0], True) and parity_ok(u[2], True), "u_r, u_theta odd in r")
        need(parity_ok(u[1], False) and parity_ok(p, False) and parity_ok(mu, False), "u_z, p, mu even in r")
        rep["parity"] = "u_r, u_t odd; u_z, p, mu even (symbolic)"
    else:
        rep["parity"] = "VIOLATED ON PURPOSE (negative-control field)"
    # 3. forcing: route A vs route B (3D Cartesian), per form, gamma in {0, 1}
    for form in fld.forms:
        for gamma in (0, 1):
            fa = forcing_route_a(u, p, mu, form, gamma)
            fb = forcing_route_b(u, p, mu, form, gamma)
            ok = all(_is_zero(fa[i] - fb[i], pts, tol=1e-35) for i in range(3))
            need(ok, f"forcing route A (cylindrical) == route B (3D Cartesian) [{form}, gamma={gamma}] (60 digits)")
            if fld.exact_solution and gamma == 0:
                need(all(_is_zero(fa[i], pts, tol=1e-35) for i in range(3)), f"exact solution: f == 0 [{form}]")
    rep["forcing_routes"] = "A == B at 6 random points, 60 digits, both forms, gamma 0 and 1"
    # 4. emitted strings round trip and C spelling
    for form in fld.forms:
        fa = sp.zeros(3, 1) if fld.exact_solution else forcing_route_a(u, p, mu, form)
        fe = [sp.sympify(s["sympy"], locals={"r": r, "z": z}) for s in emitted["forms"][form]["f"]]
        need(all(_is_zero(fa[i] - fe[i], pts, tol=1e-35) for i in range(3)), f"emitted f parses back [{form}]")
        worst = 0.0
        for pt in pts[:3]:
            rv, zv = float(pt["r"]), float(pt["z"])
            for i in range(3):
                cv = _c_eval(emitted["forms"][form]["f"][i]["c"], rv, zv)
                nv = float(_np_lambda(fe[i])(np.array(rv), np.array(zv)))
                worst = max(worst, abs(cv - nv) / max(1.0, abs(nv)))
        need(worst < 1e-12, f"C spelling == numpy spelling [{form}] ({worst:.1e})")
    # 5. finite-difference residual per form (3D Cartesian, from u, p, mu only), with teeth
    xs, ys = fld.sample()
    sel = np.linspace(0, len(xs) - 1, 25).astype(int)
    fd_pts = [(float(xs[i]), float(ys[i])) for i in sel if xs[i] > 0.05]
    rep["fd_residual"] = {}
    for form in fld.forms:
        res, sc = fd_residual(fld, form, fd_pts)
        rep["fd_residual"][form] = res
        need(res < 1e-6 or (fld.exact_solution and res * sc < 1e-6),
             f"3D finite-difference residual [{form}] {res:.1e} (scale {sc:.2e})")
    # teeth: the other form's forcing must not satisfy this form's PDE when mu varies
    if len(fld.forms) == 2 and sp.sympify(mu).free_symbols and not fld.exact_solution:
        fl = forcing_route_a(u, p, mu, "lap")
        fs = forcing_route_a(u, p, mu, "sym")
        dmax = max(abs(_np_lambda(fl[i] - fs[i])(xs, ys)).max() for i in range(3))
        need(dmax > 1e-2, f"variable mu: f_sym != f_lap in the interior (max |diff| {dmax:.3f})")
        rep["max_f_sym_minus_f_lap"] = float(dmax)
        dth = abs(_np_lambda(fl[2] - fs[2])(xs, ys)).max()
        need(dth > 1e-2, f"variable mu: the swirl forcing differs between forms (max |diff| {dth:.3f})")
        rep["max_ft_sym_minus_ft_lap"] = float(dth)
    # 6. weak form (route C) against the strong forcing, rectangle domains with the axis only
    if heavy and fld.domain.get("kind") == "rect" and fld.domain["r"][0] == 0 and not fld.exact_solution:
        rep["weak_form_residual"] = {}
        for form in fld.forms:
            own, other = weak_residual(fld, form)
            rep["weak_form_residual"][form] = {"own_f": own, "other_form_f": other}
            need(own < 1e-10, f"weak form (r-weighted, hoop terms) vs strong f [{form}]: {own:.1e}")
            if sp.sympify(mu).free_symbols:
                need(other > 1e-3, f"weak form [{form}] with the other form's f is NOT satisfied ({other:.1e})")
    # 7. boundary identities
    for b in fld.boundaries:
        on = {r: b.param["r"], z: b.param["z"]}
        t = b.param["t"]
        lo, hi = b.param["range"]
        tpts = [{t.name: sp.Rational(k, 7) * (hi - lo) + lo} for k in (1, 3, 5)] if lo != hi else []
        n = sp.Matrix([b.normal[0], b.normal[1], 0])
        if b.bc == "axis" and fld.parity:
            need(all(_is_zero(sp.limit(u[i] if i != 1 else 0, r, 0), [{}]) for i in (0, 2)),
                 f"{b.name}: u_r = u_theta = 0 on the axis")
        if b.bc == "do_nothing":
            for form in fld.forms:
                tr = stress(u, p, mu, form) * n
                mine = fld.extra.get("do_nothing_form", form)
                zero = all(_is_zero(sp.sympify(tr[i]).subs(on, simultaneous=True), tpts, tol=1e-30) for i in range(3))
                if mine == "both":
                    need(zero, f"{b.name}: sigma_{form} n = 0 (do-nothing holds in both forms)")
                elif form == mine:
                    need(zero, f"{b.name}: sigma_{form} n = 0 (this field's do-nothing)")
                else:
                    need(not zero, f"{b.name}: sigma_{form} n != 0 (the other form's condition is violated)")
        if b.bc in ("traction", "do_nothing") and len(fld.forms) == 2 and fld.extra.get("do_nothing_form") != "both" \
                and not fld.checks.get("forms_may_coincide"):
            tl = stress(u, p, mu, "lap") * n
            ts = stress(u, p, mu, "sym") * n
            need(not all(_is_zero(sp.sympify(tl[i] - ts[i]).subs(on, simultaneous=True), tpts, tol=1e-30)
                         for i in range(3)), f"{b.name}: traction data differ between the forms")
            if b.normal[0] != 0:
                need(not _is_zero(sp.sympify(tl[2] - ts[2]).subs(on, simultaneous=True), tpts, tol=1e-30),
                     f"{b.name}: the swirl (theta) traction differs between the forms")
    # 8. flux bookkeeping
    if fld.pressure == "mean_zero" and fld.domain.get("kind") == "rect":
        r0, r1 = fld.domain["r"]
        z0, z1 = fld.domain["z"]
        g, w = np.polynomial.legendre.leggauss(60)
        uzn, urn = _np_lambda(u[1]), _np_lambda(u[0])
        rs = 0.5 * (r1 - r0) * (g + 1) + r0
        zs = 0.5 * (z1 - z0) * (g + 1) + z0
        tot = 0.5 * (r1 - r0) * float(np.sum(w * rs * (uzn(rs, np.full_like(rs, z1)) - uzn(rs, np.full_like(rs, z0)))))
        tot += 0.5 * (z1 - z0) * float(np.sum(w * r1 * urn(np.full_like(zs, r1), zs)))
        if r0 > 0:
            tot -= 0.5 * (z1 - z0) * float(np.sum(w * r0 * urn(np.full_like(zs, r0), zs)))
        need(abs(tot) < 1e-12, f"zero net r-weighted boundary flux int u.n r ds = {tot:.1e}")
        rep["net_flux_r"] = tot
    if fld.checks.get("flux_free"):
        for b in fld.boundaries:
            if b.bc == "axis":
                continue
            on = {r: b.param["r"], z: b.param["z"]}
            t = b.param["t"]
            lo, hi = b.param["range"]
            tpts = [{t.name: sp.Rational(k, 7) * (hi - lo) + lo} for k in (1, 3, 5)]
            un = sum(sp.sympify(u[i]) * b.normal[i] for i in range(2))
            need(_is_zero(sp.simplify(un.subs(on, simultaneous=True)), tpts, tol=1e-30),
                 f"{b.name}: u.n = 0 (flux-free data)")
    need(float(emitted["constants"]["mu_min_sampled"]) > 0, "mu > 0")
    return rep


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=str)
        fh.write("\n")
