"""3D Stokes operators, code generation and self-checks for the F1c verification fields.

The PDE is the planar one in three dimensions; it is written here from its statement, NOT from solver code:

    -div sigma(u, p) - gamma grad(div u) = f,      div u = 0          in Omega (subset of R^3)
    sigma_sym(u, p) = 2 mu eps(u) - p I,   eps(u) = (G + G^T)/2         (symmetric-stress form, "sym")
    sigma_lap(u, p) =   mu G      - p I                                 (Laplacian / gradient form, "lap")
    G_ij = d u_i / d x_j    (row i = velocity component, column j = derivative direction)

Natural (traction) condition of each form, n the OUTWARD unit normal: sigma_form(u, p) n = g.
grad-div gamma (viscosity units, NOT scaled by mu) does not change f for a divergence-free exact u.

Every field may be given in a LOCAL frame x_loc = R^T (x - c) (exact rational rotation R); the emitted expressions
are always in the local coordinates (x, y, z) of that frame, and the check module composes them with the frame
(u_phys(x) = R u_loc(R^T (x - c)), grad u_phys = R G_loc R^T, sigma likewise, f_phys = R f_loc).

Emitted expressions come in three spellings: "sympy" (round-trips through sympify), "c" (sympy.ccode), "numpy"
(np.-prefixed, evaluated by the check module with only numpy in scope).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

x, y, z = sp.symbols("x y z", real=True)
X = (x, y, z)
FORMS = ("lap", "sym")


# ----------------------------------------------------------------------------------------------- operators
def grad_vec(u):
    return sp.Matrix(3, 3, lambda i, j: sp.diff(u[i], X[j]))


def div_vec(u):
    return sum(sp.diff(u[i], X[i]) for i in range(3))


def div_tensor(T):
    return sp.Matrix([sum(sp.diff(T[i, j], X[j]) for j in range(3)) for i in range(3)])


def curl(A):
    return sp.Matrix([sp.diff(A[2], y) - sp.diff(A[1], z),
                      sp.diff(A[0], z) - sp.diff(A[2], x),
                      sp.diff(A[1], x) - sp.diff(A[0], y)])


def stress(u, p, mu, form):
    G = grad_vec(u)
    if form == "sym":
        return mu * (G + G.T) - p * sp.eye(3)
    if form == "lap":
        return mu * G - p * sp.eye(3)
    raise ValueError(form)


def forcing_route_a(u, p, mu, form, gamma=0):
    """Route A: f = -div sigma - gamma grad div u (tensor calculus)."""
    dv = div_vec(u)
    return -div_tensor(stress(u, p, mu, form)) - gamma * sp.Matrix([sp.diff(dv, X[i]) for i in range(3)])


def forcing_route_b(u, p, mu, form, gamma=0):
    """Route B, written out independently in index notation:
       lap: f_i = -mu lap u_i - (d_j mu)(d_j u_i)                            + d_i p
       sym: f_i = -mu lap u_i - (d_j mu)(d_j u_i + d_i u_j) - mu d_i(div u)  + d_i p
       plus -gamma d_i(div u)."""
    out = []
    dv = div_vec(u)
    for i in range(3):
        fi = -mu * sum(sp.diff(u[i], X[j], 2) for j in range(3)) + sp.diff(p, X[i]) - gamma * sp.diff(dv, X[i])
        for j in range(3):
            fi -= sp.diff(mu, X[j]) * sp.diff(u[i], X[j])
            if form == "sym":
                fi -= sp.diff(mu, X[j]) * sp.diff(u[j], X[i])
        if form == "sym":
            fi -= mu * sp.diff(dv, X[i])
        out.append(fi)
    return sp.Matrix(out)


def cayley(a):
    """Exact rational rotation (I - K)^-1 (I + K), K the skew matrix of a = (a1, a2, a3)."""
    a1, a2, a3 = [sp.Rational(v) for v in a]
    K = sp.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    R = (sp.eye(3) - K).inv() * (sp.eye(3) + K)
    R = sp.simplify(R)
    assert sp.simplify(R * R.T - sp.eye(3)) == sp.zeros(3, 3) and sp.simplify(R.det() - 1) == 0
    return R


# ----------------------------------------------------------------------------------------------- fields
@dataclass
class Face:
    name: str          # "x0" = the face x = 0 (local coordinates), ...
    bc: str            # dirichlet | traction | do_nothing | slip | symmetry
    normal: tuple      # outward unit normal in LOCAL coordinates (exact)
    fixed: tuple       # (index, value): coordinate fixed on the face, e.g. (0, 1) for x = 1
    notes: str = ""


@dataclass
class Field:
    name: str
    description: str
    domain: dict
    u: sp.Matrix
    p: sp.Expr
    mu: sp.Expr
    forms: tuple
    faces: list
    pressure: str                   # mean_zero | determined
    frame: dict | None = None       # {"R": 3x3 exact, "c": (0,0,0)} or None (identity)
    box: tuple = ((0, 1), (0, 1), (0, 1))   # local box for sampling and integration, or None for shells
    shell: tuple | None = None      # (r1, r2) spherical shell
    extra: dict = field(default_factory=dict)
    gamma_checked: tuple = (0, 1)
    no_forcing: bool = False         # data-only fields (lid cavity): f = 0 by construction is NOT claimed


def _printer():
    return NumPyPrinter({"fully_qualified_modules": False, "inline": True, "allow_unknown_functions": False})


def np_str(e):
    s = NumPyPrinter(settings={"fully_qualified_modules": True}).doprint(sp.sympify(e))
    return s.replace("numpy.", "np.")


def spell(e):
    e = sp.sympify(e)
    return {"sympy": str(e), "c": sp.ccode(e), "numpy": np_str(e)}


def np_lambda(e):
    return sp.lambdify(X, e, "numpy")


def sample_points(fld: Field, n=400, seed=7):
    rng = np.random.default_rng(seed)
    if fld.shell:
        r1, r2 = fld.shell
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        r = r1 + (r2 - r1) * (0.02 + 0.96 * rng.random(n))
        P = v * r[:, None]
    else:
        lo = np.array([b[0] for b in fld.box], float)
        hi = np.array([b[1] for b in fld.box], float)
        P = lo + (hi - lo) * (0.02 + 0.96 * rng.random((n, 3)))
    return P


def gauss_box(box, n=24):
    g, w = np.polynomial.legendre.leggauss(n)
    axes, ws = [], []
    for a, b in box:
        axes.append(0.5 * (b - a) * (g + 1) + a)
        ws.append(0.5 * (b - a) * w)
    XX = np.meshgrid(*axes, indexing="ij")
    W = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return XX, W


def gauss_shell(r1, r2, nr=32, nt=48, nphi=96):
    g, w = np.polynomial.legendre.leggauss(nr)
    r = 0.5 * (r2 - r1) * (g + 1) + r1
    wr = 0.5 * (r2 - r1) * w
    gt, wt = np.polynomial.legendre.leggauss(nt)           # cos(theta) in [-1, 1]
    ph = np.arange(nphi) * 2 * np.pi / nphi
    R, CT, PH = np.meshgrid(r, gt, ph, indexing="ij")
    ST = np.sqrt(1 - CT ** 2)
    XX = [R * ST * np.cos(PH), R * ST * np.sin(PH), R * CT]
    W = (wr[:, None, None] * R[:, :1, :1] ** 0 * (r ** 2)[:, None, None]) * wt[None, :, None] * (2 * np.pi / nphi)
    return XX, np.broadcast_to(W, R.shape)


def integrate(fld: Field, fn):
    if fld.shell:
        XX, W = gauss_shell(*fld.shell)
    else:
        XX, W = gauss_box(fld.box)
    return float(np.sum(W * fn(*XX)))


def eval_mp(e, pt, dps=50):
    return sp.N(e.subs({x: pt[0], y: pt[1], z: pt[2]}), dps)


def face_points(fld: Field, face: Face, n=7, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pt = [sp.Rational(rng.randint(5, 95), 100) for _ in range(3)]
        pt[face.fixed[0]] = sp.Rational(face.fixed[1])
        out.append(pt)
    return out


def emit(fld: Field) -> dict:
    G = grad_vec(fld.u)
    d = {"name": fld.name, "description": fld.description, "domain": fld.domain, "pressure": fld.pressure,
         "conventions": {"G_ij": "d u_i / d x_j", "sigma_sym": "2 mu eps(u) - p I", "sigma_lap": "mu G - p I",
                         "f": "-div sigma_form - gamma grad div u (gamma-independent: div u = 0)",
                         "traction": "g = sigma_form n, n the outward unit normal",
                         "frame": "expressions in LOCAL coordinates x_loc = R^T (x - c); u_phys = R u_loc, "
                                  "grad_phys = R G_loc R^T, sigma_phys = R sigma_loc R^T, f_phys = R f_loc"},
         "frame": None if fld.frame is None else {"R": [[str(v) for v in fld.frame["R"].row(i)] for i in range(3)],
                                                  "R_float": [[float(v) for v in fld.frame["R"].row(i)]
                                                              for i in range(3)],
                                                  "c": [float(v) for v in fld.frame.get("c", (0, 0, 0))]},
         "u": [spell(c) for c in fld.u], "p": spell(fld.p), "mu": spell(fld.mu),
         "grad_u": [[spell(G[i, j]) for j in range(3)] for i in range(3)],
         "forms": {}, "faces": [{"name": f.name, "bc": f.bc, "normal_local": [str(v) for v in f.normal],
                                 "fixed": [f.fixed[0], str(f.fixed[1])], "notes": f.notes} for f in fld.faces],
         "extra": {k: (spell(v) if isinstance(v, sp.Basic) else v) for k, v in fld.extra.items()}}
    for form in fld.forms:
        S = stress(fld.u, fld.p, fld.mu, form)
        f = sp.zeros(3, 1) if fld.no_forcing else forcing_route_a(fld.u, fld.p, fld.mu, form)
        d["forms"][form] = {"f": [spell(c) for c in f],
                            "sigma": [[spell(S[i, j]) for j in range(3)] for i in range(3)]}
    return d


# ----------------------------------------------------------------------------------------------- self-checks
def fd_residual(fld: Field, form: str, f_exprs, h=1e-3, npts=60):
    """4th-order central differences of sigma built from u, p, mu ONLY (not from f): max |div sigma + f| / scale."""
    uf = [np_lambda(c) for c in fld.u]
    pf, muf = np_lambda(fld.p), np_lambda(fld.mu)
    ff = [np_lambda(c) for c in f_exprs]

    def grad_u(P):
        G = np.zeros(P.shape[:-1] + (3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            for i in range(3):
                fp1, fm1 = uf[i](*(P + e).T), uf[i](*(P - e).T)
                fp2, fm2 = uf[i](*(P + 2 * e).T), uf[i](*(P - 2 * e).T)
                G[..., i, j] = (8 * (fp1 - fm1) - (fp2 - fm2)) / (12 * h)
        return G

    def sigma(P):
        G = grad_u(P)
        mu = np.broadcast_to(muf(*P.T), P.shape[:-1])
        pp = np.broadcast_to(pf(*P.T), P.shape[:-1])
        if form == "sym":
            S = mu[..., None, None] * (G + np.swapaxes(G, -1, -2))
        else:
            S = mu[..., None, None] * G
        return S - pp[..., None, None] * np.eye(3)

    P = sample_points(fld, npts, seed=11)
    divS = np.zeros((len(P), 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        d = (8 * (sigma(P + e) - sigma(P - e)) - (sigma(P + 2 * e) - sigma(P - 2 * e))) / (12 * h)
        divS += d[:, :, j]
    fv = np.stack([np.broadcast_to(c(*P.T), (len(P),)) for c in ff], -1)
    scale = max(np.max(np.abs(fv)), 1.0)
    return float(np.max(np.abs(divS + fv)) / scale)


def selfcheck(fld: Field, emitted: dict, dps=40, npts=10) -> dict:
    rep = {}
    ok = True
    dv = div_vec(fld.u)
    dvs = sp.simplify(dv)
    rep["div_u_symbolic_zero"] = dvs == 0
    if not rep["div_u_symbolic_zero"]:      # decide numerically at 50 digits when simplify does not finish
        rng = random.Random(5)
        vals = []
        for _ in range(npts):
            P = sample_points(fld, 1, seed=rng.randint(0, 10 ** 6))[0]
            vals.append(abs(eval_mp(dv, [sp.Float(c, 30) for c in P])))
        rep["div_u_max_50digit"] = float(max(vals))
        rep["div_u_zero"] = rep["div_u_max_50digit"] < 1e-25
    else:
        rep["div_u_zero"] = True
    ok &= rep["div_u_zero"]
    if not fld.no_forcing:
        rng = random.Random(17)
        pts = [[sp.Float(c, 30) for c in sample_points(fld, 1, seed=rng.randint(0, 10 ** 6))[0]] for _ in range(npts)]
        for form in fld.forms:
            for gamma in fld.gamma_checked:
                a = forcing_route_a(fld.u, fld.p, fld.mu, form, gamma)
                b = forcing_route_b(fld.u, fld.p, fld.mu, form, gamma)
                diff = max(abs(eval_mp(a[i] - b[i], P)) for P in pts for i in range(3))
                rep[f"routes_agree_{form}_gamma{gamma}"] = float(diff)
                ok &= float(diff) < 1e-25
            # the EMITTED sympy string parsed back gives zero PDE residual (route B against the parsed f)
            fp = [sp.sympify(s["sympy"], locals={"x": x, "y": y, "z": z}) for s in emitted["forms"][form]["f"]]
            b = forcing_route_b(fld.u, fld.p, fld.mu, form, 0)
            rt = max(abs(eval_mp(b[i] - fp[i], P)) for P in pts for i in range(3))
            rep[f"emitted_roundtrip_{form}"] = float(rt)
            ok &= float(rt) < 1e-25
            # numpy spelling evaluates to the same numbers
            P = sample_points(fld, 50, seed=19)
            env = {"np": np, "numpy": np, "x": P[:, 0], "y": P[:, 1], "z": P[:, 2]}
            fe = [np.broadcast_to(eval(s["numpy"], {"__builtins__": {}}, env), (50,)) for s in emitted["forms"][form]["f"]]
            fl = [np.broadcast_to(np_lambda(c)(*P.T), (50,)) for c in fp]
            nd = max(float(np.max(np.abs(a_ - b_))) for a_, b_ in zip(fe, fl))
            rep[f"numpy_spelling_{form}"] = nd
            ok &= nd < 1e-11 * max(1.0, max(float(np.max(np.abs(v))) for v in fl))
            # FD residual per form from u, p, mu only; and teeth: the OTHER form's forcing must fail
            r = fd_residual(fld, form, fp)
            rep[f"fd_residual_{form}"] = r
            ok &= r < 1e-7
        if set(fld.forms) == {"lap", "sym"} and fld.mu.free_symbols and grad_vec(fld.u) != sp.zeros(3, 3):
            fl_ = [sp.sympify(s["sympy"], locals={"x": x, "y": y, "z": z}) for s in emitted["forms"]["lap"]["f"]]
            fs_ = [sp.sympify(s["sympy"], locals={"x": x, "y": y, "z": z}) for s in emitted["forms"]["sym"]["f"]]
            r1 = fd_residual(fld, "lap", fs_)
            r2 = fd_residual(fld, "sym", fl_)
            rep["fd_residual_lap_with_sym_forcing"] = r1
            rep["fd_residual_sym_with_lap_forcing"] = r2
            ok &= r1 > 1e-2 and r2 > 1e-2
            P = sample_points(fld, 400, seed=23)
            dmax = max(float(np.max(np.abs(np.broadcast_to(np_lambda(fs_[i] - fl_[i])(*P.T), (400,)))))
                       for i in range(3))
            rep["f_sym_minus_f_lap_max"] = dmax
            ok &= dmax > 1e-2
    rep["status"] = "PASS" if ok else "FAIL"
    return rep


def face_condition_values(fld: Field, face: Face, kind: str, form: str, n=7) -> float:
    """max over sample points of the face of the residual of the HOMOGENEOUS condition `kind` in `form`:
       slip / symmetry: |u.n| + |(I - n n^T) sigma n|;  do_nothing: |sigma n|."""
    nvec = sp.Matrix(face.normal)
    S = stress(fld.u, fld.p, fld.mu, form)
    worst = 0.0
    for P in face_points(fld, face, n):
        sub = {x: P[0], y: P[1], z: P[2]}
        t = (S * nvec).subs(sub)
        if kind in ("slip", "symmetry"):
            un = (fld.u.T * nvec)[0].subs(sub)
            tt = t - (t.T * nvec)[0] * nvec
            val = abs(sp.N(un, 30)) + max(abs(sp.N(c, 30)) for c in tt)
        elif kind == "do_nothing":
            val = max(abs(sp.N(c, 30)) for c in t)
        else:
            raise ValueError(kind)
        worst = max(worst, float(val))
    return worst
