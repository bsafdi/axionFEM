"""Planar Stokes operators, code generation and self-checks for the F1a verification fields.

The PDE is written here from its statement, NOT from solver code:

    -div sigma(u, p) - gamma grad(div u) = f,      div u = 0          in Omega
    sigma_S(u, p) = 2 mu eps(u) - p I,   eps(u) = (G + G^T)/2           (symmetric-stress form, "sym")
    sigma_L(u, p) =   mu G      - p I                                   (Laplacian / gradient form, "lap")
    G_ij = d u_i / d x_j   (row i = velocity component, column j = derivative direction)

Natural (traction) condition of each form on Gamma_N, n the OUTWARD unit normal:   sigma_form(u, p) n = g.
grad-div (gamma >= 0, units of viscosity, NOT scaled by mu) does not change f for a divergence-free exact u.

Everything is exact sympy arithmetic (Rational constants).  Emitted expressions come in three spellings:
  "sympy"  exact str() form (round-trips through sympy.sympify),
  "c"      sympy.ccode (C99 <math.h>),
  "numpy"  a numpy expression in x, y (np. prefix).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

x, y = sp.symbols("x y", real=True)
X = (x, y)
FORMS = ("lap", "sym")


# ----------------------------------------------------------------------------------------------- operators
def grad_vec(u):
    """G_ij = d u_i / d x_j."""
    return sp.Matrix(2, 2, lambda i, j: sp.diff(u[i], X[j]))


def div_vec(u):
    return sp.diff(u[0], x) + sp.diff(u[1], y)


def div_tensor(T):
    """(div T)_i = sum_j d T_ij / d x_j."""
    return sp.Matrix([sp.diff(T[i, 0], x) + sp.diff(T[i, 1], y) for i in range(2)])


def stress(u, p, mu, form):
    G = grad_vec(u)
    if form == "sym":
        return mu * (G + G.T) - p * sp.eye(2)
    if form == "lap":
        return mu * G - p * sp.eye(2)
    raise ValueError(form)


def forcing_route_a(u, p, mu, form, gamma=0):
    """Route A: f = -div sigma - gamma grad div u (tensor calculus)."""
    dv = div_vec(u)
    return -div_tensor(stress(u, p, mu, form)) - gamma * sp.Matrix([sp.diff(dv, x), sp.diff(dv, y)])


def forcing_route_b(u, p, mu, form, gamma=0):
    """Route B, written out independently in index notation:
       lap: f_i = -mu lap u_i - (d_j mu)(d_j u_i)                                  + d_i p
       sym: f_i = -mu lap u_i - (d_j mu)(d_j u_i + d_i u_j) - mu d_i (div u)       + d_i p
       plus -gamma d_i(div u) for grad-div."""
    out = []
    dv = div_vec(u)
    for i in range(2):
        lap = sum(sp.diff(u[i], X[j], 2) for j in range(2))
        fi = -mu * lap + sp.diff(p, X[i]) - gamma * sp.diff(dv, X[i])
        for j in range(2):
            fi -= sp.diff(mu, X[j]) * sp.diff(u[i], X[j])
            if form == "sym":
                fi -= sp.diff(mu, X[j]) * sp.diff(u[j], X[i])
        if form == "sym":
            fi -= mu * sp.diff(dv, X[i])
        out.append(fi)
    return sp.Matrix(out)


def velocity_from_streamfunction(psi):
    return sp.Matrix([sp.diff(psi, y), -sp.diff(psi, x)])


# ----------------------------------------------------------------------------------------------- fields
@dataclass
class Boundary:
    name: str                 # e.g. "x1" (the edge x = 1) or "t0" (local edge t = 0)
    bc: str                   # dirichlet | traction | do_nothing | slip | symmetry | periodic_src | periodic_dst
    normal: tuple             # outward unit normal (sympy exact)
    param: dict               # parametrization: {"t": symbol, "x": expr, "y": expr, "range": [a, b]}
    notes: str = ""


@dataclass
class Field:
    name: str
    description: str
    domain: dict              # JSON-able description
    u: sp.Matrix
    p: sp.Expr
    mu: sp.Expr
    forms: tuple
    boundaries: list
    pressure: str             # mean_zero | determined | point | periodic_jump | periodic_flow_rate
    integrate: Callable       # integrate(lambda xs, ys: values) -> float, Gauss rule over the domain
    sample: Callable          # sample() -> (xs, ys) points inside the domain (for max and FD checks)
    extra: dict = field(default_factory=dict)
    gamma_checked: tuple = (0, 1)


def gauss_rect(x0, x1, y0, y1, n=48):
    g, w = np.polynomial.legendre.leggauss(n)
    xs = 0.5 * (x1 - x0) * (g + 1) + x0
    ys = 0.5 * (y1 - y0) * (g + 1) + y0
    W = np.outer(w, w) * 0.25 * (x1 - x0) * (y1 - y0)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return lambda fn: float(np.sum(W * fn(XX, YY)))


def gauss_rotated_rect(theta, Ls=1.0, Lt=1.0, n=48):
    g, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (g + 1)
    S, T = np.meshgrid(Ls * s, Lt * s, indexing="ij")
    W = np.outer(w, w) * 0.25 * Ls * Lt
    c, si = math.cos(theta), math.sin(theta)
    XX, YY = c * S - si * T, si * S + c * T
    return lambda fn: float(np.sum(W * fn(XX, YY)))


def gauss_annulus(r1, r2, nr=48, nt=256):
    g, w = np.polynomial.legendre.leggauss(nr)
    r = 0.5 * (r2 - r1) * (g + 1) + r1
    wr = 0.5 * (r2 - r1) * w
    th = np.arange(nt) * 2 * np.pi / nt
    R, TH = np.meshgrid(r, th, indexing="ij")
    W = np.outer(wr, np.full(nt, 2 * np.pi / nt)) * R
    return lambda fn: float(np.sum(W * fn(R * np.cos(TH), R * np.sin(TH))))


def sample_rect(x0, x1, y0, y1, n=41, margin=1e-3, ny=None, ymargin=None):
    xs = np.linspace(x0 + margin, x1 - margin, n)
    ym = margin if ymargin is None else ymargin
    ys = np.linspace(y0 + ym, y1 - ym, ny or n)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return lambda: (XX.ravel(), YY.ravel())


def sample_rotated_rect(theta, Ls=1.0, Lt=1.0, n=41, margin=1e-3):
    S, T = np.meshgrid(np.linspace(margin, Ls - margin, n), np.linspace(margin, Lt - margin, n), indexing="ij")
    c, si = math.cos(theta), math.sin(theta)
    return lambda: ((c * S - si * T).ravel(), (si * S + c * T).ravel())


def sample_annulus(r1, r2, n=41, margin=1e-3):
    r = np.linspace(r1 + margin, r2 - margin, n)
    th = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
    R, TH = np.meshgrid(r, th, indexing="ij")
    return lambda: ((R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel())


# ----------------------------------------------------------------------------------------------- code generation
def _np_lambda(expr):
    f = sp.lambdify((x, y), expr, modules="numpy")
    return lambda X_, Y_: np.broadcast_to(np.asarray(f(X_, Y_), dtype=float), np.shape(X_)).copy()


def spell(expr):
    expr = sp.sympify(expr)
    return {"sympy": str(expr), "c": sp.ccode(expr),
            "numpy": sp.pycode(expr, fully_qualified_modules=True).replace("math.", "np.")}


def emit(fld: Field) -> dict:
    """All expressions and constants of a field, per form."""
    u, p, mu = fld.u, fld.p, fld.mu
    G = grad_vec(u)
    out = {"name": fld.name, "description": fld.description, "domain": fld.domain, "pressure": fld.pressure,
           "conventions": {"G_ij": "d u_i / d x_j",
                           "sigma_sym": "2 mu eps(u) - p I", "sigma_lap": "mu G - p I",
                           "f": "-div sigma_form - gamma grad div u (= -div sigma_form for these fields)",
                           "traction": "g = sigma_form n, n the outward unit normal",
                           "grad_div": "gamma (div u_h, div v); gamma in viscosity units, not scaled by mu"},
           "u": [spell(c) for c in u], "p": spell(p), "mu": spell(mu),
           "grad_u": [[spell(G[i, j]) for j in range(2)] for i in range(2)],
           "forms": {}, "boundaries": [], "extra": {}}
    for form in fld.forms:
        S = stress(u, p, mu, form)
        f = forcing_route_a(u, p, mu, form)
        out["forms"][form] = {"f": [spell(c) for c in f],
                              "sigma": [[spell(S[i, j]) for j in range(2)] for i in range(2)]}
    for b in fld.boundaries:
        bj = {"name": b.name, "bc": b.bc, "normal": [spell(c) for c in b.normal], "notes": b.notes,
              "param": {k: (spell(v) if isinstance(v, sp.Basic) else v) for k, v in b.param.items() if k != "t"}}
        if b.bc in ("traction", "do_nothing"):
            n = sp.Matrix(b.normal)
            bj["traction"] = {form: [spell(c) for c in stress(u, p, mu, form) * n]
                              for form in fld.forms}
        out["boundaries"].append(bj)
    # constants (numerical, Gauss rule; double precision)
    nu_ = [_np_lambda(c) for c in u]
    np_ = _np_lambda(p)
    ngu = [[_np_lambda(G[i, j]) for j in range(2)] for i in range(2)]
    area = fld.integrate(lambda X_, Y_: np.ones_like(X_))
    pmean = fld.integrate(np_) / area
    const = {
        "area": area,
        "p_mean": pmean,
        "u_L2": math.sqrt(fld.integrate(lambda X_, Y_: nu_[0](X_, Y_) ** 2 + nu_[1](X_, Y_) ** 2)),
        "grad_u_L2": math.sqrt(fld.integrate(lambda X_, Y_: sum(ngu[i][j](X_, Y_) ** 2
                                                               for i in range(2) for j in range(2)))),
        "p_minus_mean_L2": math.sqrt(fld.integrate(lambda X_, Y_: (np_(X_, Y_) - pmean) ** 2)),
    }
    xs, ys = fld.sample()
    const["u_max_sampled"] = float(np.max(np.hypot(nu_[0](xs, ys), nu_[1](xs, ys))))
    const["p_max_abs_sampled"] = float(np.max(np.abs(np_(xs, ys))))
    const["p_minus_mean_max_abs_sampled"] = float(np.max(np.abs(np_(xs, ys) - pmean)))
    const["mu_min_sampled"] = float(np.min(_np_lambda(mu)(xs, ys)))
    const["mu_max_sampled"] = float(np.max(_np_lambda(mu)(xs, ys)))
    out["constants"] = const
    out["extra"] = {k: (spell(v) if isinstance(v, sp.Basic) else v) for k, v in fld.extra.items()}
    return out


# ----------------------------------------------------------------------------------------------- self-checks
class CheckFailed(AssertionError):
    pass


def _has_root(e):
    return any(isinstance(a_, sp.Pow) and a_.exp.is_Rational and not a_.exp.is_Integer for a_ in e.atoms(sp.Pow))


def _mp_zero(e, syms, pts, tol):
    """60-digit evaluation (mpmath) at the given points; True if |e| < tol at all of them."""
    import mpmath
    mpmath.mp.dps = 60
    f = sp.lambdify(syms, e, modules="mpmath")
    for pt in pts:
        v = f(*[mpmath.mpf(str(sp.N(c, 70))) for c in pt])
        if abs(complex(v)) > tol:
            return False
    return True


def _method(e):
    return "60-digit evaluation at 12 random points" if _has_root(sp.sympify(e)) else "symbolically, sympy.simplify"


def _is_zero(expr, pts, tol=1e-40, symbolic=False):
    """Zero test.  symbolic=True: sympy.simplify must return exactly 0 (the round-trip residual and the two-route
    forcing comparison).  For fields containing sqrt(x^2 + y^2) (annulus) simplify does not finish in reasonable time;
    those, and every auxiliary check, are decided by 60-digit evaluation at random points (a non-zero analytic
    expression is not < 1e-40 at all of them)."""
    expr = sp.sympify(expr)
    if expr == 0:
        return True
    if symbolic and not _has_root(expr):
        e = sp.expand(expr)
        return e == 0 or sp.simplify(e) == 0
    if symbolic:
        pts = list(pts) + [(a_ * sp.Rational(21, 20), b_ * sp.Rational(19, 20)) for a_, b_ in pts]
    return _mp_zero(expr, (x, y), pts, tol)


def _rand_pts(fld, k=6, seed=1):
    xs, ys = fld.sample()
    rng = random.Random(seed)
    idx = [rng.randrange(len(xs)) for _ in range(k)]
    return [(sp.Rational(str(round(float(xs[i]), 6))), sp.Rational(str(round(float(ys[i]), 6)))) for i in idx]


def _c_eval(code, X_, Y_):
    ns = {"x": X_, "y": Y_, "exp": np.exp, "sin": np.sin, "cos": np.cos, "sinh": np.sinh, "cosh": np.cosh,
          "tan": np.tan, "log": np.log, "sqrt": np.sqrt, "pow": np.power, "M_PI": math.pi, "atan2": np.arctan2,
          "tanh": np.tanh, "fabs": np.abs}
    return np.asarray(eval(code, {"__builtins__": {}}, ns), dtype=float)


def _d4(fn, X_, Y_, axis, h):
    """Fourth-order central difference of fn along axis 0 (x) or 1 (y)."""
    dx, dy = (h, 0.0) if axis == 0 else (0.0, h)
    return (-fn(X_ + 2 * dx, Y_ + 2 * dy) + 8 * fn(X_ + dx, Y_ + dy) - 8 * fn(X_ - dx, Y_ - dy)
            + fn(X_ - 2 * dx, Y_ - 2 * dy)) / (12 * h)


def fd_residual(u_fns, p_fn, mu_fn, f_fns, form, X_, Y_, h=1e-3):
    """Finite-difference residual of the strong form, from u, p, mu ONLY (independent of the symbolic forcing):
    sigma by fourth-order central differences of u, then div sigma by fourth-order central differences of sigma.
    Returns max|res| / scale, scale = max(|f|, |sigma|, 1)."""
    def sig_comp(i, j):
        def s_ij(xx, yy):
            m = mu_fn(xx, yy)
            gij = _d4(u_fns[i], xx, yy, j, h)
            if form == "sym":
                val = m * (gij + _d4(u_fns[j], xx, yy, i, h))
            else:
                val = m * gij
            if i == j:
                val = val - p_fn(xx, yy)
            return val
        return s_ij

    res, smax = [], 0.0
    for i in range(2):
        divs = _d4(sig_comp(i, 0), X_, Y_, 0, h) + _d4(sig_comp(i, 1), X_, Y_, 1, h)
        res.append(-divs - f_fns[i](X_, Y_))
        smax = max(smax, float(np.max(np.abs(sig_comp(i, 0)(X_, Y_)))), float(np.max(np.abs(sig_comp(i, 1)(X_, Y_)))))
    scale = max(np.max(np.abs(f_fns[0](X_, Y_))), np.max(np.abs(f_fns[1](X_, Y_))), smax, 1.0)
    return float(np.max(np.abs(np.array(res))) / scale)


def selfcheck(fld: Field, emitted: dict, log=print) -> dict:
    """Raises CheckFailed on any failure; returns a summary."""
    rep = {}
    u, p, mu = fld.u, fld.p, fld.mu
    pts = _rand_pts(fld)

    def need(cond, msg):
        if not cond:
            raise CheckFailed(f"[{fld.name}] {msg}")
        log(f"  ok  {msg}")

    # 1. divergence-free, exactly
    need(_is_zero(div_vec(u), pts), "div u == 0 symbolically")
    # 2. forcing by two independent routes, per form and with grad-div gamma in {0, 1}
    for form in fld.forms:
        for gam in fld.gamma_checked:
            d = forcing_route_a(u, p, mu, form, gam) - forcing_route_b(u, p, mu, form, gam)
            need(all(_is_zero(c, pts, symbolic=True) for c in d),
                 f"{form}: route A (tensor) == route B (index), gamma={gam} ({_method(d[0])})")
        f_full = forcing_route_a(u, p, mu, form, 1)
        need(all(_is_zero(c, pts) for c in (f_full - forcing_route_a(u, p, mu, form, 0))),
             f"{form}: grad-div does not change f (div u = 0)")
    # 3. round trip of the EMITTED strings: residual of the PDE with the parsed forcing is zero symbolically
    for form in fld.forms:
        fpar = sp.Matrix([sp.sympify(s["sympy"], locals={"x": x, "y": y}) for s in emitted["forms"][form]["f"]])
        res = forcing_route_b(u, p, mu, form) - fpar      # = -div sigma(u, p) - f_emitted, via the index route
        need(all(_is_zero(c, pts, symbolic=True) for c in res),
             f"{form}: -div sigma(u_exact, p_exact) - f_emitted == 0 ({_method(res[0])})")
        # C spelling evaluates to the same numbers
        xs, ys = fld.sample()
        for i in range(2):
            fc = _c_eval(emitted["forms"][form]["f"][i]["c"], xs, ys)
            fn = _np_lambda(fpar[i])(xs, ys)
            need(np.max(np.abs(fc - fn)) <= 1e-12 * max(1.0, np.max(np.abs(fn))),
                 f"{form}: C spelling of f_{i} == sympy value (max |diff| {np.max(np.abs(fc - fn)):.1e})")
    # 4. finite-difference residual per form, from u, p, mu only
    u_fns = [_np_lambda(c) for c in u]
    p_fn, mu_fn = _np_lambda(p), _np_lambda(mu)
    xs, ys = fld.sample()
    sel = slice(None, None, max(1, len(xs) // 200))
    for form in (() if fld.extra.get("skip_fd") else fld.forms):
        f_fns = [_np_lambda(sp.sympify(s["sympy"], locals={"x": x, "y": y})) for s in emitted["forms"][form]["f"]]
        r = fd_residual(u_fns, p_fn, mu_fn, f_fns, form, xs[sel], ys[sel])
        rep[f"fd_residual_{form}"] = r
        need(r < 1e-7, f"{form}: finite-difference residual (4th order, h=1e-3) {r:.2e} < 1e-7")
        if fld.extra.get("forms_must_differ_interior"):
            other = "lap" if form == "sym" else "sym"
            f_oth = [_np_lambda(c) for c in forcing_route_a(u, p, mu, other)]
            r2 = fd_residual(u_fns, p_fn, mu_fn, f_oth, form, xs[sel], ys[sel])
            rep[f"fd_residual_{form}_with_{other}_forcing"] = r2
            need(r2 > 1e-2, f"{form}: the FD check has teeth: with the {other}-form forcing it gives {r2:.2e} > 1e-2")
    # 5. the two forms are distinguishable where the design says they must be
    if len(fld.forms) == 2 and fld.extra.get("forms_must_differ_interior"):
        d = forcing_route_a(u, p, mu, "sym") - forcing_route_a(u, p, mu, "lap")
        fn = [_np_lambda(c)(xs, ys) for c in d]
        m = float(np.max(np.hypot(*fn)))
        rep["f_sym_minus_f_lap_max"] = m
        need(m > 1e-2, f"f_sym - f_lap = -div(mu G^T) is not ~0 in the interior (max {m:.3g})")
    # 6. viscosity positive
    need(emitted["constants"]["mu_min_sampled"] > 0.05, f"mu > 0 (min sampled {emitted['constants']['mu_min_sampled']:.3g})")
    # 7. boundary conditions
    t = sp.Symbol("t", real=True)
    net_flux = 0.0
    for b in fld.boundaries:
        bt = b.param["t"]
        sub = {x: b.param["x"], y: b.param["y"]}
        n = sp.Matrix(b.normal)
        un = (u.T * n)[0].subs(sub)
        tvec = sp.Matrix([-n[1], n[0]])
        a, bb = b.param["range"]
        if b.bc != "periodic_src" and b.bc != "periodic_dst":
            # 200-point Gauss-Legendre in the edge parameter (double precision; the check threshold is 1e-12)
            integrand = sp.lambdify(bt, un * sp.sqrt(sp.diff(b.param["x"], bt) ** 2 + sp.diff(b.param["y"], bt) ** 2),
                                    modules="numpy")
            gp, gw = np.polynomial.legendre.leggauss(200)
            lo, hi = float(a), float(bb)
            tt = 0.5 * (hi - lo) * (gp + 1) + lo
            net_flux += float(0.5 * (hi - lo) * np.sum(gw * np.broadcast_to(integrand(tt), tt.shape)))
        bpts = [{bt: sp.Rational(k, 7) * (bb - a) + a} for k in range(1, 7)]

        def zero_on_edge(e):
            e = e.subs(sub)
            if not _has_root(e):
                e = sp.simplify(e)
                if e == 0:
                    return True
            return _mp_zero(e, (bt,), [(q[bt],) for q in bpts], 1e-35)

        if b.bc in ("slip", "symmetry"):
            need(zero_on_edge((u.T * n)[0]), f"{b.name} ({b.bc}): u.n == 0")
            for form in fld.forms:
                tt = (tvec.T * stress(u, p, mu, form) * n)[0]
                need(zero_on_edge(tt), f"{b.name} ({b.bc}): tangential traction t.sigma_{form} n == 0")
            need(not zero_on_edge((tvec.T * u)[0]), f"{b.name} ({b.bc}): tangential velocity is NOT zero (slip != no-slip)")
        if b.bc == "do_nothing":
            own = fld.extra.get("do_nothing_form", "both")
            for form in (FORMS if own == "both" else (own,)):
                tr = stress(u, p, mu, form) * n
                need(all(zero_on_edge(c) for c in tr), f"{b.name}: sigma_{form} n == 0 (do-nothing of the {form} form)")
            if own != "both":
                other = "lap" if own == "sym" else "sym"
                tr_oth = stress(u, p, mu, other) * n
                need(not all(zero_on_edge(c) for c in tr_oth),
                     f"{b.name}: sigma_{other} n != 0 (the field discriminates the two do-nothing conditions)")
        # flux-free data (every edge): u.n == 0
        if fld.extra.get("flux_free") and b.bc not in ("periodic_src", "periodic_dst"):
            need(zero_on_edge((u.T * n)[0]), f"{b.name}: u.n == 0 (flux-free Dirichlet data)")
        # cross-BC: an edge must VIOLATE the homogeneous condition of every other natural/slip BC kind used
        # in this field, so that a BC applied on the wrong attribute cannot reproduce the exact solution
        if not fld.extra.get("cross_bc_exempt") and b.bc not in ("periodic_src", "periodic_dst"):
            kinds = {bb_.bc for bb_ in fld.boundaries} & {"slip", "symmetry", "do_nothing"}
            same = {"slip", "symmetry"} if b.bc in ("slip", "symmetry") else {b.bc}   # identical conditions in planar Stokes
            for kind in sorted(kinds - same):
                if kind in ("slip", "symmetry"):
                    sat = [zero_on_edge((u.T * n)[0]) and zero_on_edge((tvec.T * stress(u, p, mu, form) * n)[0])
                           for form in fld.forms]
                else:
                    own = fld.extra.get("do_nothing_form", "both")
                    sat = [all(zero_on_edge(c) for c in stress(u, p, mu, form) * n)
                           for form in (FORMS if own == "both" else (own,))]
                need(not any(sat), f"{b.name} ({b.bc}) violates the {kind} condition (wrong-attribute mapping visible)")
        if b.bc in ("slip", "symmetry") and fld.extra.get("slip_other_form_nonzero"):
            other = "lap" if fld.forms == ("sym",) else "sym"
            tt = (tvec.T * stress(u, p, mu, other) * n)[0]
            need(not zero_on_edge(tt), f"{b.name}: tangential traction of the OTHER form ({other}) != 0 (discriminating)")
        if b.bc == "traction" and fld.extra.get("traction_forms_must_differ"):
            d = (stress(u, p, mu, "sym") - stress(u, p, mu, "lap")) * n
            need(not all(zero_on_edge(c) for c in d), f"{b.name}: traction data differ between forms")
    rep["net_boundary_flux"] = net_flux
    if fld.pressure in ("mean_zero", "point", "periodic_jump", "periodic_flow_rate"):
        need(abs(net_flux) < 1e-12, f"net flux through the non-periodic boundary {net_flux:.1e} == 0 (nullspace compatibility)")
    # 8. periodicity
    per = fld.extra.get("periodic_x")
    if per:
        L = per
        for c in list(u) + [mu, fld.extra["p_periodic"]]:
            need(_is_zero(sp.expand_trig(c.subs(x, x + L) - c), pts), f"periodic in x with period {L}: {str(c)[:40]}...")
        need(_is_zero(p - (fld.extra["p_periodic"] - fld.extra["G"] * x), pts), "p = p_periodic - G x")
    return rep


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=str)
        fh.write("\n")
