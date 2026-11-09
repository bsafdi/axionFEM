"""F2a (steady incompressible Navier-Stokes) verification fields: operators, fields, code generation, self-checks.

Reference generator.  The PDE is written from the continuum equations, NOT from any solver code:

  rho (u . grad) u - div sigma_form(u, p) = f,     div u = 0
  sigma_sym = 2 mu eps(u) - p I,   sigma_lap = mu G - p I,   G_ij = d u_i / d x_j (row = component)

Planar: Cartesian (x, y).  Axisymmetric with swirl: cylindrical (r, theta, z), fields independent of theta, unknowns
(u_r, u_theta, u_z, p); the gradient in the orthonormal cylindrical frame is

      G = [[d_r u_r,  -u_t / r,  d_z u_r],
           [d_r u_t,   u_r / r,  d_z u_t],
           [d_r u_z,   0,        d_z u_z]]          (rows r, theta, z; columns d/dr, (1/r) d/dtheta, d/dz)

so that (u . grad) u = G u contains the centrifugal term -u_t^2 / r (r row) and the Coriolis term u_r u_t / r
(theta row), and (div T)_i for a (not necessarily symmetric) tensor is

      (div T)_r = d_r T_rr + d_z T_rz + (T_rr - T_tt) / r
      (div T)_t = d_r T_tr + d_z T_tz + (T_tr + T_rt) / r
      (div T)_z = d_r T_zr + d_z T_zz + T_zr / r.

Every axisymmetric forcing is ALSO derived by a second, independent route: the field is embedded in 3D Cartesian
coordinates (U = u_r e_r + u_t e_t + u_z e_z, r = sqrt(x^2 + y^2)), the forcing is computed with Cartesian tensor
calculus and projected on e_r, e_t, e_z; the two routes must agree to 1e-40 at random points (60 digits).  The
finite-difference residual check is done in 3D Cartesian coordinates too, from u, p, mu, rho only.

Emitted spellings: "sympy" (exact str), "c" (C99), "numpy" (np.* in the coordinate names of the field).
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
r, z = sp.symbols("r z", real=True)
R_ = sp.Rational
FORMS = ("lap", "sym")


# ============================================================================================ planar operators
def grad2(u):
    return sp.Matrix(2, 2, lambda i, j: sp.diff(u[i], (x, y)[j]))


def div2(u):
    return sp.diff(u[0], x) + sp.diff(u[1], y)


def divT2(T):
    return sp.Matrix([sp.diff(T[i, 0], x) + sp.diff(T[i, 1], y) for i in range(2)])


def stress2(u, p, mu, form):
    G = grad2(u)
    return (mu * (G + G.T) if form == "sym" else mu * G) - p * sp.eye(2)


def conv2(u, rho):
    return rho * grad2(u) * u


def forcing2_a(u, p, mu, rho, form, convection=True):
    """Route A (tensor calculus): f = rho G u - div sigma_form."""
    f = -divT2(stress2(u, p, mu, form))
    return f + conv2(u, rho) if convection else f


def forcing2_b(u, p, mu, rho, form):
    """Route B (index notation, written out independently):
       f_i = rho u_j d_j u_i - mu lap u_i - (d_j mu)(d_j u_i [+ d_i u_j]) [- mu d_i div u] + d_i p."""
    X = (x, y)
    dv = div2(u)
    out = []
    for i in range(2):
        fi = sum(rho * u[j] * sp.diff(u[i], X[j]) for j in range(2))
        fi += -mu * sum(sp.diff(u[i], X[j], 2) for j in range(2)) + sp.diff(p, X[i])
        for j in range(2):
            fi -= sp.diff(mu, X[j]) * sp.diff(u[i], X[j])
            if form == "sym":
                fi -= sp.diff(mu, X[j]) * sp.diff(u[j], X[i])
        if form == "sym":
            fi -= mu * sp.diff(dv, X[i])
        out.append(fi)
    return sp.Matrix(out)


# ============================================================================================ axisymmetric operators
def gradA(u):
    ur, ut, uz = u
    return sp.Matrix([[sp.diff(ur, r), -ut / r, sp.diff(ur, z)],
                      [sp.diff(ut, r), ur / r, sp.diff(ut, z)],
                      [sp.diff(uz, r), 0, sp.diff(uz, z)]])


def divA(u):
    return sp.diff(r * u[0], r) / r + sp.diff(u[2], z)


def divTA(T):
    return sp.Matrix([sp.diff(T[0, 0], r) + sp.diff(T[0, 2], z) + (T[0, 0] - T[1, 1]) / r,
                      sp.diff(T[1, 0], r) + sp.diff(T[1, 2], z) + (T[1, 0] + T[0, 1]) / r,
                      sp.diff(T[2, 0], r) + sp.diff(T[2, 2], z) + T[2, 0] / r])


def stressA(u, p, mu, form):
    G = gradA(u)
    return (mu * (G + G.T) if form == "sym" else mu * G) - p * sp.eye(3)


def convA(u, rho, centrifugal=True, coriolis=True):
    c = rho * gradA(u) * sp.Matrix(u)
    if not centrifugal:
        c[0] += rho * u[1] ** 2 / r
    if not coriolis:
        c[1] -= rho * u[0] * u[1] / r
    return c


def forcingA_a(u, p, mu, rho, form, convection=True, centrifugal=True, coriolis=True):
    f = -divTA(stressA(u, p, mu, form))
    return f + convA(u, rho, centrifugal, coriolis) if convection else f


def forcingA_cartesian(u, p, mu, rho, form):
    """Route B for the axisymmetric fields: embed in 3D Cartesian coordinates, Cartesian tensor calculus, project on
    (e_r, e_t, e_z).  Returns sympy expressions in (x, y, z3) and the projection helpers."""
    X3, Y3, Z3 = sp.symbols("X3 Y3 Z3", real=True)
    rr = sp.sqrt(X3 ** 2 + Y3 ** 2)
    sub = {r: rr, z: Z3}
    ur, ut, uz = (c.subs(sub) for c in u)
    er = sp.Matrix([X3 / rr, Y3 / rr, 0])
    et = sp.Matrix([-Y3 / rr, X3 / rr, 0])
    ez = sp.Matrix([0, 0, 1])
    U = ur * er + ut * et + uz * ez
    P, MU = p.subs(sub), mu.subs(sub)
    C3 = (X3, Y3, Z3)
    G = sp.Matrix(3, 3, lambda i, j: sp.diff(U[i], C3[j]))
    S = (MU * (G + G.T) if form == "sym" else MU * G) - P * sp.eye(3)
    f = -sp.Matrix([sum(sp.diff(S[i, j], C3[j]) for j in range(3)) for i in range(3)]) + rho * G * U
    return (X3, Y3, Z3), f, (er, et, ez)


# ============================================================================================ fields
@dataclass
class Field:
    name: str
    description: str
    geometry: str               # planar | axisymmetric
    domain: dict
    u: list                     # planar [u_x, u_y]; axisymmetric [u_r, u_theta, u_z]
    p: sp.Expr
    mu: sp.Expr
    rho: sp.Expr
    forms: tuple = FORMS
    pressure: str = "mean_zero"  # mean_zero | determined
    boundaries: list = field(default_factory=list)   # [{"name", "bc", "normal"(planar), "edge": (coord, value)}]
    extra: dict = field(default_factory=dict)
    exact_ns: bool = False       # f == 0 for every form (Kovasznay, Burgers)
    forcing_override: list | None = None   # a given body force with NO exact solution (energy-identity fields)

    @property
    def coords(self):
        return (x, y) if self.geometry == "planar" else (r, z)


def _box(fld):
    d = fld.domain
    return d["c0"], d["c1"], d["d0"], d["d1"]


def ns_sq_dirichlet():
    """Planar, unit square, all-Dirichlet, variable mu, rho != 1: the convective term is O(1) and not a gradient."""
    psi = R_(1, 4) * (sp.exp(R_(3, 5) * x) * sp.sin(R_(9, 5) * y + R_(1, 2))
                      + R_(3, 10) * y ** 2 * sp.cos(R_(21, 10) * x - R_(2, 5)))
    u = [sp.diff(psi, y), -sp.diff(psi, x)]
    p = R_(1, 4) * sp.cos(R_(19, 10) * x + R_(1, 5)) * sp.sinh(R_(4, 5) * y + R_(1, 10)) + R_(1, 10) * x * y
    mu = R_(1, 60) * (R_(3, 2) + R_(1, 2) * sp.sin(2 * x - y + R_(3, 10)) + x * y / 4)
    return Field("ns_sq_dirichlet",
                 "planar NS MMS on [0,1]^2, Dirichlet on all four edges, variable mu ~ 0.02-0.04, rho = 5/4 "
                 "(Re ~ rho U L / mu ~ 40): stream function, so div u = 0 exactly; no x<->y or reflection symmetry; "
                 "u . grad omega != 0 (non-Beltrami: the convective term is NOT a gradient)",
                 "planar", {"kind": "rectangle", "c0": 0, "c1": 1, "d0": 0, "d1": 1}, u, p, mu, R_(5, 4),
                 boundaries=[{"name": n, "bc": "dirichlet"} for n in ("x0", "x1", "y0", "y1")],
                 extra={"forms_must_differ_interior": True})


def ns_sq_traction():
    f = ns_sq_dirichlet()
    f.name = "ns_sq_traction"
    f.description = ("same u, p, mu, rho as ns_sq_dirichlet; traction g = sigma_form n (n = +e_x) on x = 1, where u.n "
                     "changes sign (inflow and outflow parts: the convective boundary term of a fully antisymmetric "
                     "convective form is O(1) there); Dirichlet on x = 0, y = 0, y = 1; pressure determined")
    f.pressure = "determined"
    f.boundaries = [{"name": "x0", "bc": "dirichlet"}, {"name": "y0", "bc": "dirichlet"},
                    {"name": "y1", "bc": "dirichlet"}, {"name": "x1", "bc": "traction", "normal": (1, 0)}]
    f.extra = {"forms_must_differ_interior": True, "traction_un_changes_sign": "x1"}
    return f


def ns_sq_energy():
    """No exact solution: a LINEAR (degree-1) body force on [0,1]^2 with homogeneous Dirichlet data (u = 0 on the
    boundary), constant mu.  Used for the discrete energy identities of the convective forms (M1 checks): every load integral (f, v_h) is exact for any rule of degree >= 3."""
    F0 = R_(3, 1)
    fvec = [F0 * (-(y - R_(2, 5))), F0 * R_(6, 5) * (x - R_(3, 5))]
    return Field("ns_sq_energy", "planar, [0,1]^2, u = 0 on the boundary, mu = 1/100, rho = 1, linear body force "
                 "f = 3 (-(y - 2/5), (6/5)(x - 3/5)) (curl f != 0, so u_h != 0); no exact solution",
                 "planar", {"kind": "rectangle", "c0": 0, "c1": 1, "d0": 0, "d1": 1},
                 [sp.Integer(0), sp.Integer(0)], sp.Integer(0), R_(1, 100), sp.Integer(1),
                 boundaries=[{"name": n, "bc": "dirichlet"} for n in ("x0", "x1", "y0", "y1")],
                 forcing_override=fvec)


def kovasznay(Re=40):
    lam = sp.Rational(Re, 2) - sp.sqrt(sp.Rational(Re * Re, 4) + 4 * sp.pi ** 2)
    u = [1 - sp.exp(lam * x) * sp.cos(2 * sp.pi * y), lam / (2 * sp.pi) * sp.exp(lam * x) * sp.sin(2 * sp.pi * y)]
    p = R_(1, 2) * (1 - sp.exp(2 * lam * x))
    return Field(f"ns_kovasznay_re{Re}",
                 f"Kovasznay flow, Re = {Re} (rho = 1, mu = 1/Re), lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2); exact steady "
                 "NS solution with f = 0 (verified symbolically below, not taken from a source); Dirichlet data from "
                 "the exact solution on all four edges of [-1/2, 1] x [-1/2, 3/2]",
                 "planar", {"kind": "rectangle", "c0": R_(-1, 2), "c1": 1, "d0": R_(-1, 2), "d1": R_(3, 2)},
                 u, p, sp.Rational(1, Re), sp.Integer(1),
                 boundaries=[{"name": n, "bc": "dirichlet"} for n in ("x0", "x1", "y0", "y1")],
                 exact_ns=True, extra={"lambda": lam})


# ---------------------------------------------------------------------------------------------- axisymmetric
def ns_axi_dirichlet():
    """Axisymmetric with swirl on (r, z) in [0,1]^2.  Parity-respecting: u_r, u_t odd in r; u_z, p, mu even.
    Meridional flow from a Stokes stream function Psi = r^2 Phi(r^2, z): u_r = -(1/r) d_z Psi, u_z = (1/r) d_r Psi
    (div u = 0 exactly; regular at the axis).  Swirl u_t = r Theta(r^2, z) depends on z, so the centrifugal force
    rho u_t^2 / r is NOT a gradient, and u_r != 0, so the Coriolis term rho u_r u_t / r is non-zero."""
    s = r ** 2
    Phi = R_(1, 4) * (sp.exp(R_(3, 5) * z) * sp.cos(R_(7, 10) * s + R_(1, 3)) * (1 + s / 5)
                      + R_(2, 5) * s * z ** 2)
    Psi = s * Phi
    ur = sp.expand(-sp.diff(Psi, z) / r)
    uz = sp.expand(sp.diff(Psi, r) / r)
    ut = r * R_(1, 2) * sp.exp(-s / 2) * sp.sin(R_(3, 2) * z + R_(1, 4)) * (1 + z / 3)
    p = R_(1, 4) * sp.cos(R_(6, 5) * s + R_(2, 3) * z + R_(1, 5)) + R_(1, 8) * s * z
    mu = R_(1, 60) * (R_(3, 2) + R_(9, 10) * sp.sin(2 * s + z + R_(3, 10)) + s * z / 2)
    return Field("ns_axi_dirichlet",
                 "axisymmetric NS MMS with swirl on (r, z) in [0,1]^2: Dirichlet on r = 1, z = 0, z = 1; axis r = 0 "
                 "(u_r = u_t = 0 strongly; u_z, p natural); variable mu (even in r), rho = 5/4; parity-respecting",
                 "axisymmetric", {"kind": "rectangle", "c0": 0, "c1": 1, "d0": 0, "d1": 1},
                 [ur, ut, uz], p, mu, R_(5, 4),
                 boundaries=[{"name": "axis", "bc": "axis"}, {"name": "r1", "bc": "dirichlet"},
                             {"name": "z0", "bc": "dirichlet"}, {"name": "z1", "bc": "dirichlet"}],
                 extra={"forms_must_differ_interior": True})


def ns_axi_burgers():
    """Burgers vortex: exact steady axisymmetric NS solution with swirl (f = 0, verified symbolically below):
    u_r = -alpha r / 2, u_z = alpha z, u_t = Gamma / (2 pi r) (1 - exp(-alpha r^2 / (4 nu))).  Its meridional part
    is linear (its convective term is a gradient), but the swirl balance u_r d_r u_t + u_r u_t / r = nu (...) carries
    the Coriolis term, and the radial pressure gradient carries the centrifugal term rho u_t^2 / r."""
    # nu = 1/20 (core radius 2 sqrt(nu / alpha) = 0.45): with nu = 1/40 (core 0.32) the convergence is pre-asymptotic
    # on N = 8..128 (lap uL2 pairwise 3.62 -> 3.08, fit 3.15; unweighted 3.25)
    alpha, nu, Gam = sp.Integer(1), R_(1, 20), sp.pi
    a = alpha / (4 * nu)
    ur, uz = -alpha * r / 2, alpha * z
    ut = Gam / (2 * sp.pi * r) * (1 - sp.exp(-a * r ** 2))
    # p_theta(r) = rho int_0^r u_t^2 / s ds, closed form (checked: d_r p_theta = u_t^2 / r symbolically)
    t = a * r ** 2
    # p_theta = Gamma^2 a / (8 pi^2) I(a r^2),  I(T) = int_0^T (1 - e^-t)^2 / t^2 dt
    #         = -(1 - e^-T)^2 / T + 2 (Ei(-T) - Ei(-2T)) + 2 ln 2   (integration by parts; p_theta(0) = 0)
    ptheta = (Gam ** 2 * a / (8 * sp.pi ** 2)) * (-(1 - sp.exp(-t)) ** 2 / t + 2 * (sp.Ei(-t) - sp.Ei(-2 * t))
                                                  + 2 * sp.log(2))
    p = -alpha ** 2 * (r ** 2 / 4 + z ** 2) / 2 + ptheta
    return Field("ns_axi_burgers",
                 "Burgers vortex, alpha = 1, nu = mu = 1/20, rho = 1, Gamma = pi, on (r, z) in [0,1]^2; Dirichlet data "
                 "from the exact solution on r = 1, z = 0, z = 1; axis conditions on r = 0",
                 "axisymmetric", {"kind": "rectangle", "c0": 0, "c1": 1, "d0": 0, "d1": 1},
                 [ur, ut, uz], p, nu, sp.Integer(1), exact_ns=True,
                 boundaries=[{"name": "axis", "bc": "axis"}, {"name": "r1", "bc": "dirichlet"},
                             {"name": "z0", "bc": "dirichlet"}, {"name": "z1", "bc": "dirichlet"}],
                 extra={"ptheta": ptheta})


def ns_axi_couette():
    """Circular Couette flow between r1 = 1/2 (rotating, Omega_1 = 2, so u_theta(r1) = 1) and r2 = 1 (at rest), cut at
    z = 0 and z = 1/2 with the exact data there: u = (0, A r + B / r, 0), A = -Omega_1 eta^2 / (1 - eta^2),
    B = Omega_1 r1^2 / (1 - eta^2), eta = 1/2; exact for every Re (f = 0, verified below).  rho = 5/4, mu = rho / 40
    (Re = Omega_1 r1 (r2 - r1) / nu = 10).  The convective term is the centrifugal force alone, balanced by the
    pressure p = rho int u_theta^2 / r dr: the non-MMS cell of the centrifugal row with a velocity-independent check
    in the PRESSURE, and (rho != 1) the non-MMS cell of the density row."""
    r1, r2, Om = R_(1, 2), sp.Integer(1), sp.Integer(2)
    eta = r1 / r2
    A = -Om * eta ** 2 / (1 - eta ** 2)
    B = Om * r1 ** 2 / (1 - eta ** 2)
    rho = R_(5, 4)
    ut = A * r + B / r
    p = rho * (A ** 2 * r ** 2 / 2 + 2 * A * B * sp.log(r) - B ** 2 / (2 * r ** 2))
    return Field("ns_axi_couette",
                 "circular Couette flow, eta = 1/2, Omega_1 = 2, rho = 5/4, mu = 1/32 on (r, z) in [1/2, 1] x [0, 1/2]; "
                 "Dirichlet data from the exact solution on all four edges (no axis)",
                 "axisymmetric", {"kind": "rectangle", "c0": R_(1, 2), "c1": 1, "d0": 0, "d1": R_(1, 2)},
                 [sp.Integer(0), ut, sp.Integer(0)], p, rho / 40, rho, exact_ns=True,
                 boundaries=[{"name": n, "bc": "dirichlet"} for n in ("r0", "r1", "z0", "z1")],
                 extra={"off_axis": True})


def ns_axi_energy():
    """No exact solution: body force f = (0, 3 r, 3 r^2) in (r, z) in [0,1]^2 with u = 0 on r = 1, z = 0, z = 1 and the
    axis conditions.  f_theta drives the swirl; f_z = 3 r^2 is not a gradient (d_r f_z != 0), so it drives a
    meridional circulation (u_r != 0) INDEPENDENTLY of the centrifugal force: dropping the centrifugal or the
    Coriolis term then breaks the discrete energy identity by int rho u_t^2 u_r dr dz (with f = (0, 3r, 0) alone the
    meridional flow is driven only by the centrifugal force, and dropping it gives u_r = 0, so the identity could not
    detect the omission).  Every r-weighted load integral is exact for a rule of degree >= 5."""
    return Field("ns_axi_energy", "axisymmetric, [0,1]^2, homogeneous Dirichlet + axis, mu = 1/100, rho = 1, "
                 "f = (0, 3 r, 3 r^2)",
                 "axisymmetric", {"kind": "rectangle", "c0": 0, "c1": 1, "d0": 0, "d1": 1},
                 [sp.Integer(0)] * 3, sp.Integer(0), R_(1, 100), sp.Integer(1),
                 boundaries=[{"name": "axis", "bc": "axis"}, {"name": "r1", "bc": "dirichlet"},
                             {"name": "z0", "bc": "dirichlet"}, {"name": "z1", "bc": "dirichlet"}],
                 forcing_override=[sp.Integer(0), 3 * r, 3 * r ** 2])


ALL = [ns_sq_dirichlet, ns_sq_traction, ns_sq_energy, lambda: kovasznay(40), ns_axi_dirichlet, ns_axi_burgers,
       ns_axi_energy, ns_axi_couette]


# ============================================================================================ code generation
from sympy.printing.pycode import PythonCodePrinter  # noqa: E402


class _Py(PythonCodePrinter):
    def _print_Ei(self, e):
        return f"sc.expi({self._print(e.args[0])})"


def spell(expr, coords):
    """numpy spelling: np.* functions; the exponential integral Ei as sc.expi (sc = scipy.special)."""
    expr = sp.sympify(expr)
    py = _Py({"fully_qualified_modules": True}).doprint(expr).replace("math.", "np.").replace("numpy.", "np.")
    try:
        cc = sp.ccode(expr)
    except Exception:          # Ei has no C99 spelling
        cc = None
    return {"sympy": str(expr), "c": cc, "numpy": py}


def _lam(expr, coords):
    f = sp.lambdify(coords, expr, modules=[{"Ei": _expi}, "numpy"])
    return lambda A, B: np.broadcast_to(np.asarray(f(A, B), dtype=float), np.shape(A)).copy()


def _expi(v):
    import scipy.special as sc
    return sc.expi(v)


def forcing(fld, form, **kw):
    if fld.forcing_override is not None:
        return sp.Matrix(fld.forcing_override)
    if fld.geometry == "planar":
        return forcing2_a(sp.Matrix(fld.u), fld.p, fld.mu, fld.rho, form, **kw)
    return forcingA_a(fld.u, fld.p, fld.mu, fld.rho, form, **kw)


def comp_grad(fld):
    """Component derivatives d_j u_i in the field's coordinates (planar: Cartesian G; axisymmetric: (d_r, d_z) of
    (u_r, u_t, u_z), NOT the cylindrical gradient tensor)."""
    c = fld.coords
    return [[sp.diff(ui, cj) for cj in c] for ui in fld.u]


def gauss_box(fld, n=48):
    a0, a1, b0, b1 = (float(v) for v in _box(fld))
    g, w = np.polynomial.legendre.leggauss(n)
    A = 0.5 * (a1 - a0) * (g + 1) + a0
    B = 0.5 * (b1 - b0) * (g + 1) + b0
    W = np.outer(w, w) * 0.25 * (a1 - a0) * (b1 - b0)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    return AA.ravel(), BB.ravel(), W.ravel()


def sample_box(fld, n=41, margin=1e-3):
    a0, a1, b0, b1 = (float(v) for v in _box(fld))
    A = np.linspace(a0 + margin, a1 - margin, n)
    B = np.linspace(b0 + margin, b1 - margin, n)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    return AA.ravel(), BB.ravel()


def emit(fld: Field) -> dict:
    c = fld.coords
    cn = [str(v) for v in c]
    out = {"name": fld.name, "description": fld.description, "geometry": fld.geometry, "coords": cn,
           "domain": {k: (str(v) if isinstance(v, sp.Basic) else v) for k, v in fld.domain.items()},
           "pressure": fld.pressure, "rho": {"sympy": str(fld.rho), "value": float(fld.rho)},
           "conventions": {
               "planar": "u = (u_x, u_y); grad_u[i][j] = d u_i / d x_j; sigma_sym = 2 mu eps - p I, sigma_lap = mu G - p I; "
                         "f = rho (u.grad)u - div sigma_form; traction g = sigma_form n (outward n)",
               "axisymmetric": "coords (r, z); u = (u_r, u_theta, u_z); grad_u[i][j] = d u_i / d (r, z)_j (component "
                               "derivatives, NOT the cylindrical gradient tensor); f in the (e_r, e_theta, e_z) frame, "
                               "f = rho G u - div sigma_form with G and div as in tools/fluids_mms/f2a_fields.py"},
           "u": [spell(ci, c) for ci in fld.u], "p": spell(fld.p, c), "mu": spell(fld.mu, c),
           "grad_u": [[spell(e, c) for e in row] for row in comp_grad(fld)],
           "forms": {}, "boundaries": fld.boundaries and [dict(b) for b in fld.boundaries],
           "exact_solution": fld.forcing_override is None, "exact_ns": fld.exact_ns}
    for form in fld.forms:
        f = forcing(fld, form)
        ent = {"f": [spell(sp.simplify(e) if fld.exact_ns else e, c) for e in f]}
        if fld.forcing_override is None:
            ent["f_without_convection"] = [spell(e, c) for e in forcing(fld, form, convection=False)]
            if fld.geometry == "axisymmetric":
                ent["f_without_centrifugal"] = [spell(e, c) for e in forcing(fld, form, centrifugal=False)]
                ent["f_without_coriolis"] = [spell(e, c) for e in forcing(fld, form, coriolis=False)]
        if fld.geometry == "planar" and fld.forcing_override is None:
            S = stress2(sp.Matrix(fld.u), fld.p, fld.mu, form)
            ent["sigma"] = [[spell(S[i, j], c) for j in range(2)] for i in range(2)]
        out["forms"][form] = ent
    for b in out["boundaries"]:
        if b.get("bc") == "traction":
            n = sp.Matrix(b["normal"])
            b["normal"] = [str(v) for v in b["normal"]]
            b["traction"] = {form: [spell(e, c) for e in stress2(sp.Matrix(fld.u), fld.p, fld.mu, form) * n]
                             for form in fld.forms}
    if fld.extra:
        out["extra"] = {k: (spell(v, c) if isinstance(v, sp.Basic) else v) for k, v in fld.extra.items()}
    out["constants"] = constants(fld)
    return out


def constants(fld):
    c = fld.coords
    A, B, W = gauss_box(fld)
    wr = A if fld.geometry == "axisymmetric" else np.ones_like(A)
    us = [_lam(e, c)(A, B) for e in fld.u]
    pf = _lam(fld.p, c)(A, B)
    vol = float(np.sum(W * wr))
    pmean = float(np.sum(W * wr * pf) / vol)
    out = {"measure": float(np.sum(W)), "volume_weighted": vol, "p_mean_weighted": pmean,
           "p_mean_unweighted": float(np.sum(W * pf) / np.sum(W)),
           "u_L2_weighted": math.sqrt(float(np.sum(W * wr * sum(v ** 2 for v in us))))}
    sa, sb = sample_box(fld)
    su = [_lam(e, c)(sa, sb) for e in fld.u]
    sp_ = _lam(fld.p, c)(sa, sb)
    out["u_max_sampled"] = float(np.max(np.sqrt(sum(v ** 2 for v in su))))
    out["p_minus_mean_max_abs_sampled"] = float(np.max(np.abs(sp_ - pmean)))
    out["p_max_abs_sampled"] = float(np.max(np.abs(sp_)))
    mu = _lam(fld.mu, c)(sa, sb)
    out["mu_min_sampled"], out["mu_max_sampled"] = float(np.min(mu)), float(np.max(mu))
    out["rho"] = float(fld.rho)
    out["Re_estimate"] = float(fld.rho) * out["u_max_sampled"] * 1.0 / float(np.mean(mu))
    if fld.forcing_override is None:
        f = forcing(fld, FORMS[0])
        fc = forcing(fld, FORMS[0], convection=False)
        fv = np.array([_lam(e, c)(sa, sb) for e in f])
        conv = fv - np.array([_lam(e, c)(sa, sb) for e in fc])
        out["f_max_sampled"] = float(np.max(np.linalg.norm(fv, axis=0)))
        out["convective_term_max_sampled"] = float(np.max(np.linalg.norm(conv, axis=0)))
    return out


# ============================================================================================ self-checks
class CheckFailed(AssertionError):
    pass


def _mp_zero(e, syms, pts, tol=1e-35, dps=60):
    import mpmath
    mpmath.mp.dps = dps
    f = sp.lambdify(syms, e, modules="mpmath")
    worst = 0.0
    for pt in pts:
        v = abs(complex(f(*[mpmath.mpf(str(sp.N(q, dps + 10))) for q in pt])))
        worst = max(worst, float(v))
    return worst < tol, worst


def _rand_pts(fld, k=8, seed=1):
    a0, a1, b0, b1 = (float(v) for v in _box(fld))
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        a = a0 + (a1 - a0) * (0.05 + 0.9 * rng.random())
        b = b0 + (b1 - b0) * (0.05 + 0.9 * rng.random())
        out.append((sp.Rational(str(round(a, 6))), sp.Rational(str(round(b, 6)))))
    return out


def _d4(fn, P, axis, h):
    e = np.zeros(P.shape[-1])
    e[axis] = h
    return (-fn(P + 2 * e) + 8 * fn(P + e) - 8 * fn(P - e) + fn(P - 2 * e)) / (12 * h)


def fd_residual(fld, fvec_sym, form, h=1e-3):
    """Strong-form residual by 4th-order central differences, from u, p, mu, rho ONLY (independent of the symbolic
    forcing).  Planar: in (x, y).  Axisymmetric: the field is embedded in 3D Cartesian coordinates at sample points
    (r cos phi, r sin phi, z) with random phi, the residual is computed with Cartesian differences and projected on
    (e_r, e_t, e_z).  Returns max|res| / max(|f|, |sigma|, |rho u.grad u|, 1)."""
    c = fld.coords
    uf = [_lam(e, c) for e in fld.u]
    pf, muf = _lam(fld.p, c), _lam(fld.mu, c)
    ff = [_lam(e, c) for e in fvec_sym]
    rho = float(fld.rho)
    sa, sb = sample_box(fld, n=15, margin=0.05)
    if fld.geometry == "planar":
        P = np.stack([sa, sb], -1)

        def U(Q):
            return np.stack([uf[0](Q[..., 0], Q[..., 1]), uf[1](Q[..., 0], Q[..., 1])], -1)
        d = 2

        def scal(fn, Q):
            return fn(Q[..., 0], Q[..., 1])
        proj = None
    else:
        phi = np.random.default_rng(7).uniform(0, 2 * np.pi, sa.shape)
        P = np.stack([sa * np.cos(phi), sa * np.sin(phi), sb], -1)

        def cyl(Q):
            return np.hypot(Q[..., 0], Q[..., 1]), Q[..., 2]

        def U(Q):
            rr, zz = cyl(Q)
            c_, s_ = Q[..., 0] / rr, Q[..., 1] / rr
            a, b, w = uf[0](rr, zz), uf[1](rr, zz), uf[2](rr, zz)
            return np.stack([a * c_ - b * s_, a * s_ + b * c_, w], -1)
        d = 3

        def scal(fn, Q):
            return fn(*cyl(Q))
        proj = (np.stack([np.cos(phi), np.sin(phi), 0 * phi], -1), np.stack([-np.sin(phi), np.cos(phi), 0 * phi], -1),
                np.stack([0 * phi, 0 * phi, 1 + 0 * phi], -1))

    def Gij(Q, i, j):
        return _d4(lambda Q_: U(Q_)[..., i], Q, j, h)

    def S(i, j):
        def s(Q):
            m = scal(muf, Q)
            v = m * (Gij(Q, i, j) + (Gij(Q, j, i) if form == "sym" else 0))
            return v - (scal(pf, Q) if i == j else 0)
        return s
    divS = np.stack([sum(_d4(S(i, j), P, j, h) for j in range(d)) for i in range(d)], -1)
    UU = U(P)
    conv = rho * np.stack([sum(Gij(P, i, j) * UU[..., j] for j in range(d)) for i in range(d)], -1)
    res = conv - divS
    if proj is None:
        fcart = np.stack([scal(fn, P) for fn in ff], -1)
        rv = res - fcart
    else:
        rr, zz = np.hypot(P[..., 0], P[..., 1]), P[..., 2]
        fc = [fn(rr, zz) for fn in ff]
        rv = np.stack([np.sum(res * e, -1) for e in proj], -1) - np.stack(fc, -1)
    scale = max(1.0, float(np.max(np.abs(divS))), float(np.max(np.abs(conv))))
    return float(np.max(np.abs(rv)) / scale)


def selfcheck(fld: Field, em: dict, log=print) -> dict:
    rep = {}
    c = fld.coords
    pts = _rand_pts(fld)

    def need(cond, msg):
        if not cond:
            raise CheckFailed(f"[{fld.name}] {msg}")
        log(f"  ok  {msg}")

    def zero(e, tol=1e-35):
        e = sp.sympify(e)
        if e == 0:
            return True, 0.0
        return _mp_zero(e, c, pts, tol)

    if fld.forcing_override is not None:
        # no exact solution: only properties of the given force
        fv = sp.Matrix(fld.forcing_override)
        if fld.geometry == "planar":
            curl = sp.diff(fv[1], x) - sp.diff(fv[0], y)
            need(curl != 0, f"curl f = {curl} != 0 (u_h != 0)")
            need(all(sp.Poly(e, x, y).total_degree() <= 1 for e in fv if e != 0), "f is linear: load integrals exact for degree >= 3")
        else:
            need(fv[1] != 0, "swirl forcing f_theta != 0")
            need(all(sp.Poly(e, r, z).total_degree() <= 2 for e in fv if e != 0), "f has degree <= 2: r-weighted load integrals exact for degree >= 5")
            need(sp.diff(fv[2], r) != 0 or sp.diff(fv[0], z) != 0, "the meridional part of f is not a gradient (drives u_r != 0 without the centrifugal force)")
            need(sp.simplify(fv[2].subs(r, -r) - fv[2]) == 0, "f_z even in r (parity)")
            need(sp.simplify(fv[0].subs(r, -r) + fv[0]) == 0 and sp.simplify(fv[1].subs(r, -r) + fv[1]) == 0,
                 "f_r, f_theta odd in r (parity)")
        return rep

    # 1. divergence free
    dv = div2(sp.Matrix(fld.u)) if fld.geometry == "planar" else divA(fld.u)
    ok = sp.simplify(sp.expand(dv)) == 0 or zero(dv)[0]
    need(ok, "div u == 0 (symbolic, 60-digit fallback)")
    # 2. two independent routes for the forcing, per form
    for form in fld.forms:
        fa = forcing(fld, form)
        if fld.geometry == "planar":
            d = fa - forcing2_b(sp.Matrix(fld.u), fld.p, fld.mu, fld.rho, form)
            good = all(sp.simplify(sp.expand(e)) == 0 for e in d)
            need(good, f"{form}: route A (tensor) == route B (index notation), symbolically")
        else:
            (X3, Y3, Z3), fcart, (er, et, ez) = forcingA_cartesian(fld.u, fld.p, fld.mu, fld.rho, form)
            worst = 0.0
            import mpmath
            mpmath.mp.dps = 60
            lf = [sp.lambdify((X3, Y3, Z3), e, modules=[{"Ei": mpmath.ei}, "mpmath"]) for e in fcart]
            la = [sp.lambdify((r, z), e, modules=[{"Ei": mpmath.ei}, "mpmath"]) for e in fa]
            rng = random.Random(3)
            for (rr, zz) in pts:
                ph = mpmath.mpf(rng.random()) * 2 * mpmath.pi
                R0, Z0 = mpmath.mpf(str(rr)), mpmath.mpf(str(zz))
                Xp, Yp = R0 * mpmath.cos(ph), R0 * mpmath.sin(ph)
                F = [g(Xp, Yp, Z0) for g in lf]
                proj = [F[0] * mpmath.cos(ph) + F[1] * mpmath.sin(ph), -F[0] * mpmath.sin(ph) + F[1] * mpmath.cos(ph), F[2]]
                for k in range(3):
                    worst = max(worst, float(abs(proj[k] - la[k](R0, Z0))))
            rep[f"route_ab_{form}_max_abs"] = worst
            need(worst < 1e-40, f"{form}: cylindrical route == 3D Cartesian route at 8 random points, 60 digits "
                                f"(max |diff| {worst:.1e})")
    # 3. round trip of the emitted strings (sympy and C spellings)
    sa, sb = sample_box(fld, n=21, margin=0.02)
    for form in fld.forms:
        fpar = [sp.sympify(s["sympy"], locals={str(v): v for v in c}) for s in em["forms"][form]["f"]]
        res = [a - b for a, b in zip(forcing(fld, form), fpar)]
        ok = all(zero(e, 1e-40)[0] for e in res)
        need(ok, f"{form}: parsed emitted f == derived f (60 digits, 8 points)")
        for i, s in enumerate(em["forms"][form]["f"]):
            vn = _lam(fpar[i], c)(sa, sb)
            ns = {"np": np, "sc": __import__("scipy.special", fromlist=["expi"]), str(c[0]): sa, str(c[1]): sb}
            vp = np.asarray(eval(s["numpy"], {"__builtins__": {}}, ns), dtype=float)
            need(np.max(np.abs(vn - vp)) <= 1e-11 * max(1.0, np.max(np.abs(vn))),
                 f"{form}: numpy spelling of f_{i} evaluates to the sympy value")
    # 4. exact NS solutions: f == 0 symbolically
    if fld.exact_ns:
        for form in fld.forms:
            f = forcing(fld, form)
            okz = all(sp.simplify(e) == 0 or zero(e, 1e-40)[0] for e in f)
            need(okz, f"{form}: exact NS solution: rho (u.grad)u - div sigma == 0 (symbolic, 60-digit fallback)")
    # 5. FD residual per form, from u, p, mu, rho only; with teeth
    for form in fld.forms:
        rr_ = fd_residual(fld, forcing(fld, form), form)
        rep[f"fd_residual_{form}"] = rr_
        need(rr_ < 1e-7, f"{form}: 4th-order FD residual from u, p, mu, rho only: {rr_:.2e} < 1e-7")
        if not fld.exact_ns:
            r2 = fd_residual(fld, forcing(fld, form, convection=False), form)
            rep[f"fd_residual_{form}_forcing_without_convection"] = r2
            need(r2 > 1e-2, f"{form}: FD check has teeth: forcing without convection gives {r2:.2e} > 1e-2")
        if fld.extra.get("forms_must_differ_interior"):
            other = "lap" if form == "sym" else "sym"
            r3 = fd_residual(fld, forcing(fld, other), form)
            rep[f"fd_residual_{form}_with_{other}_forcing"] = r3
            need(r3 > 1e-2, f"{form}: FD check has teeth: {other}-form forcing gives {r3:.2e} > 1e-2")
        if fld.geometry == "axisymmetric" and not fld.exact_ns:
            for kw in ("centrifugal", "coriolis"):
                r4 = fd_residual(fld, forcing(fld, form, **{kw: False}), form)
                rep[f"fd_residual_{form}_without_{kw}"] = r4
                need(r4 > 1e-3, f"{form}: FD check sees the {kw} term: forcing without it gives {r4:.2e} > 1e-3")
    # 6. non-Beltrami / convection not absorbable by the pressure
    if fld.geometry == "planar":
        U = sp.Matrix(fld.u)
        cv = conv2(U, fld.rho)
        curl_c = sp.diff(cv[1], x) - sp.diff(cv[0], y)
        cc = np.abs(_lam(curl_c, c)(sa, sb))
        rep["curl_convective_max"] = float(cc.max())
        need(cc.max() > 0.05, f"curl(rho (u.grad) u) != 0 (max {cc.max():.3g}): the convective term is not a gradient "
                              "(non-Beltrami)")
    elif fld.extra.get("off_axis"):
        cent = np.abs(_lam(fld.rho * fld.u[1] ** 2 / r, c)(sa, sb))
        rep["centrifugal_max"] = float(cent.max())
        need(cent.max() > 0.1, f"centrifugal force rho u_t^2 / r != 0 (max {cent.max():.3g}): balanced by d_r p")
        fnc = forcingA_a(fld.u, fld.p, fld.mu, fld.rho, "lap", centrifugal=False)
        okc = not all(sp.simplify(e) == 0 for e in fnc)
        need(okc, "without the centrifugal term the exact solution is NOT a solution (it moves the pressure)")
    else:
        cv = convA(fld.u, fld.rho)
        # meridional curl of the (r, z) part and the theta component: neither is absorbable by p(r, z)
        curl_m = sp.diff(cv[0], z) - sp.diff(cv[2], r)
        cm = np.abs(_lam(curl_m, c)(sa, sb))
        ct = np.abs(_lam(cv[1], c)(sa, sb))
        cent = rho_ut2 = fld.rho * fld.u[1] ** 2 / r
        dz_cent = np.abs(_lam(sp.diff(cent, z), c)(sa, sb)) if sp.diff(cent, z) != 0 else np.zeros_like(sa)
        cor = np.abs(_lam(fld.rho * fld.u[0] * fld.u[1] / r, c)(sa, sb))
        rep.update({"meridional_curl_convective_max": float(cm.max()), "theta_convective_max": float(ct.max()),
                    "dz_centrifugal_max": float(dz_cent.max()), "coriolis_max": float(cor.max())})
        if not fld.exact_ns:
            need(cm.max() > 0.05, f"meridional curl of the convective term != 0 (max {cm.max():.3g})")
            need(dz_cent.max() > 0.01, f"d_z(rho u_t^2 / r) != 0 (max {dz_cent.max():.3g}): centrifugal force is not a gradient")
        need(cor.max() > 0.01, f"Coriolis term rho u_r u_t / r != 0 (max {cor.max():.3g})")
        need(ct.max() > 0.01, f"theta component of the convective term != 0 (max {ct.max():.3g})")
        del rho_ut2
        # 7. parity and axis regularity
        ur, ut, uz = fld.u
        for name, e, sgn in (("u_r", ur, -1), ("u_theta", ut, -1), ("u_z", uz, 1), ("p", fld.p, 1), ("mu", fld.mu, 1)):
            okp = sp.simplify(e.subs(r, -r) - sgn * e) == 0
            need(okp, f"parity: {name}(-r) == {'-' if sgn < 0 else '+'}{name}(r)")
        for name, e in (("u_r", ur), ("u_theta", ut)):
            lim = sp.limit(e, r, 0)
            need(sp.simplify(lim) == 0, f"axis: {name}(0, z) == 0")
        for name, e in (("d_r u_z", sp.diff(uz, r)), ("d_r mu", sp.diff(fld.mu, r))):
            need(sp.simplify(sp.limit(e, r, 0)) == 0, f"axis: {name}(0, z) == 0")
    # 8. traction edge with u.n of both signs
    if fld.extra.get("traction_un_changes_sign"):
        b = fld.extra["traction_un_changes_sign"]
        ys = np.linspace(0, 1, 401)
        un = _lam(fld.u[0], c)(np.ones_like(ys), ys)
        rep["traction_un_min_max"] = [float(un.min()), float(un.max())]
        need(un.min() < -0.05 and un.max() > 0.05, f"{b}: u.n takes both signs ({un.min():.3g}, {un.max():.3g})")
    return rep


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.write("\n")
