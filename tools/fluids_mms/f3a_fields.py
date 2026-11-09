"""F3a (transient incompressible Navier-Stokes) verification fields: operators, fields, code generation, self-checks.

Reference generator.  The PDE is written from the continuum equations, NOT from any solver code:

  rho (du/dt + (u . grad) u) - div sigma_form(u, p) = f,     div u = 0
  sigma_sym = 2 mu eps(u) - p I,   sigma_lap = mu G - p I,   G_ij = d u_i / d x_j (row = component)

Planar: Cartesian (x, y).  Axisymmetric with swirl: cylindrical (r, theta, z), fields independent of theta, unknowns
(u_r, u_theta, u_z, p); in the orthonormal frame

      G = [[d_r u_r,  -u_t / r,  d_z u_r],
           [d_r u_t,   u_r / r,  d_z u_t],
           [d_r u_z,   0,        d_z u_z]]          (u . grad) u = G u

      (div T)_r = d_r T_rr + d_z T_rz + (T_rr - T_tt) / r
      (div T)_t = d_r T_tr + d_z T_tz + (T_tr + T_rt) / r
      (div T)_z = d_r T_zr + d_z T_zz + T_zr / r.

The operators are those of tools/fluids_mms/f2a_fields.py, re-typed here with the time derivative added.  Every axisymmetric forcing is ALSO derived through a 3D
Cartesian embedding (route B) and compared at random points with 50 digits.

Emitted spellings: "sympy" (exact str) and "numpy" (np.* in the coordinate names of the field and t; the Womersley
pipe profile uses jv = scipy.special.jv, the only non-numpy name).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
r, z = sp.symbols("r z", real=True)
X3, Y3, Z3 = sp.symbols("X Y Z", real=True)
R_ = sp.Rational
FORMS = ("lap", "sym")
I = sp.I


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


def forcing2_a(u, p, mu, rho, form, convection=True, time=True):
    """Route A (tensor calculus): f = rho du/dt + rho G u - div sigma_form."""
    f = -divT2(stress2(u, p, mu, form))
    if convection:
        f += rho * grad2(u) * sp.Matrix(u)
    if time:
        f += rho * sp.Matrix([sp.diff(c, t) for c in u])
    return f


def forcing2_b(u, p, mu, rho, form):
    """Route B (index notation, written out independently; constant mu):
       f_i = rho d_t u_i + rho u_j d_j u_i - mu lap u_i [- mu d_i div u] + d_i p."""
    X = (x, y)
    dv = div2(u)
    out = []
    for i in range(2):
        fi = rho * sp.diff(u[i], t) + sum(rho * u[j] * sp.diff(u[i], X[j]) for j in range(2))
        fi += -mu * sum(sp.diff(u[i], X[j], 2) for j in range(2)) + sp.diff(p, X[i])
        fi -= sum(sp.diff(mu, X[j]) * sp.diff(u[i], X[j]) for j in range(2))
        if form == "sym":
            fi -= sum(sp.diff(mu, X[j]) * sp.diff(u[j], X[i]) for j in range(2))
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


def forcingA_a(u, p, mu, rho, form, convection=True, time=True, centrifugal=True, coriolis=True):
    f = -divTA(stressA(u, p, mu, form))
    if convection:
        c = rho * gradA(u) * sp.Matrix(u)
        if not centrifugal:
            c[0] += rho * u[1] ** 2 / r
        if not coriolis:
            c[1] -= rho * u[0] * u[1] / r
        f += c
    if time:
        f += rho * sp.Matrix([sp.diff(c_, t) for c_ in u])
    return f


def forcingA_cartesian(u, p, mu, rho, form):
    """Route B: embed in 3D Cartesian coordinates, compute rho(dU/dt + (U.grad)U) - div sigma there, project on the
    cylindrical frame.  Returns a function (r, z, t, phi) -> (f_r, f_t, f_z) evaluated with mpmath."""
    rr = sp.sqrt(X3 ** 2 + Y3 ** 2)
    sub = {r: rr, z: Z3}
    ur, ut, uz = (sp.sympify(c).subs(sub) for c in u)
    cph, sph = X3 / rr, Y3 / rr
    U = sp.Matrix([ur * cph - ut * sph, ur * sph + ut * cph, uz])
    P = sp.sympify(p).subs(sub)
    M = sp.sympify(mu).subs(sub)
    XX = (X3, Y3, Z3)
    G = sp.Matrix(3, 3, lambda i, j: sp.diff(U[i], XX[j]))
    S = (M * (G + G.T) if form == "sym" else M * G) - P * sp.eye(3)
    f = rho * sp.Matrix([sp.diff(U[i], t) for i in range(3)]) + rho * G * U \
        - sp.Matrix([sum(sp.diff(S[i, j], XX[j]) for j in range(3)) for i in range(3)])
    fl = sp.lambdify((X3, Y3, Z3, t), list(f), modules="mpmath")

    def ev(rv, zv, tv, phi):
        X, Y = rv * mp.cos(phi), rv * mp.sin(phi)
        fx, fy, fz = fl(X, Y, zv, tv)
        return (fx * mp.cos(phi) + fy * mp.sin(phi), -fx * mp.sin(phi) + fy * mp.cos(phi), fz)
    return ev


# ============================================================================================ field container
@dataclass
class Field:
    name: str
    geometry: str                      # planar | axisymmetric
    u: list                            # sympy components (planar: ux, uy; axi: ur, ut, uz)
    p: object
    mu: object
    rho: object
    box: tuple                         # (x0, x1, y0, y1) or (r0, r1, z0, z1)
    pressure: str                      # mean_zero | determined
    boundaries: list                   # [{name, kind: dirichlet|traction|axis, where, normal}]
    T: object                          # end time of the fixed runs
    description: str
    exact_ns: bool = False             # f == 0 (exact NS solution)
    space_exact: bool = False          # u in P2, p in P1 for every t (temporal MMS)
    homogeneous: bool = False          # energy fields: u = 0 on the boundary, f given, no exact solution
    data_only: bool = False            # cross-code cavity: u = Dirichlet data only (no exact solution), f = 0
    f_given: list | None = None        # energy fields: the force
    numpy_override: dict = field(default_factory=dict)   # name -> numpy spelling (Womersley: complex closed forms)
    extra: dict = field(default_factory=dict)

    @property
    def coords(self):
        return ("x", "y") if self.geometry == "planar" else ("r", "z")

    @property
    def syms(self):
        return (x, y) if self.geometry == "planar" else (r, z)


def forcing(fld: Field, form, **kw):
    if fld.f_given is not None:
        return sp.Matrix(fld.f_given)
    if fld.geometry == "planar":
        return forcing2_a(fld.u, fld.p, fld.mu, fld.rho, form, **kw)
    return forcingA_a(fld.u, fld.p, fld.mu, fld.rho, form, **kw)


def stress(fld, form):
    if fld.geometry == "planar":
        return stress2(fld.u, fld.p, fld.mu, form)
    return stressA(fld.u, fld.p, fld.mu, form)


def grad(fld):
    """Component derivatives d u_i / d (x, y) or d u_i / d (r, z) (NOT the cylindrical gradient: the check module
    adds the hoop entries itself)."""
    c = fld.syms
    return [[sp.diff(ui, c[j]) for j in range(2)] for ui in fld.u]


# ============================================================================================ the fields
def _time_fns():
    """Distinct smooth, non-polynomial time functions (BDF2 is exact for quadratics in t; these are not)."""
    a = 1 + R_(1, 2) * sp.sin(2 * t + R_(1, 3))
    b = R_(3, 4) * sp.exp(-t / 2) * sp.cos(3 * t / 2 - R_(1, 5))
    c = R_(1, 3) + R_(1, 4) * sp.cos(5 * t / 2 + R_(1, 7))
    return a, b, c


def tr_sq_poly():
    """Planar SPACE-EXACT temporal MMS: u in P2, p in P1 at every t (stream function cubic in space), non-Beltrami,
    rho = 5/4, mu = 1/20, Dirichlet on all edges of [0,1]^2, pressure mean-free."""
    a, b, c = _time_fns()
    psi1 = x ** 2 * y - R_(2, 3) * x * y ** 2 + R_(1, 5) * y ** 3 + R_(1, 4) * x ** 3 - R_(1, 2) * x * y
    psi2 = R_(1, 3) * x ** 3 - x ** 2 * y + R_(3, 5) * y ** 2 + R_(1, 2) * x * y ** 2 - R_(2, 7) * y ** 3
    psi = a * psi1 + b * psi2 + c * (y ** 2 - x ** 2 / 3)
    u = [sp.diff(psi, y), -sp.diff(psi, x)]
    p = sp.sin(t - R_(1, 4)) * (x - R_(1, 2)) + R_(1, 2) * sp.cos(2 * t) * (y - R_(1, 2))
    return Field("f3a_tr_sq_poly", "planar", u, p, R_(1, 20), R_(5, 4), (0, 1, 0, 1), "mean_zero",
                 _box_bnds("planar", ("dirichlet",) * 4), 1,
                 "planar space-exact temporal MMS (u in P2, p in P1), rho = 5/4, mu = 1/20, non-Beltrami",
                 space_exact=True)


def tr_sq_adapt():
    """Planar space-exact field with a fast phase around t = 1/2 (tanh front) for the error-control case."""
    a = 1 + R_(1, 2) * sp.tanh(10 * (t - R_(1, 2)))
    b = R_(3, 4) * sp.cos(3 * t / 2 - R_(1, 5))
    psi1 = x ** 2 * y - R_(2, 3) * x * y ** 2 + R_(1, 5) * y ** 3 + R_(1, 4) * x ** 3 - R_(1, 2) * x * y
    psi2 = R_(1, 3) * x ** 3 - x ** 2 * y + R_(3, 5) * y ** 2 + R_(1, 2) * x * y ** 2 - R_(2, 7) * y ** 3
    psi = a * psi1 + b * psi2
    u = [sp.diff(psi, y), -sp.diff(psi, x)]
    p = R_(1, 2) * sp.tanh(10 * (t - R_(1, 2))) * (x - R_(1, 2)) + R_(1, 2) * sp.cos(2 * t) * (y - R_(1, 2))
    return Field("f3a_tr_sq_adapt", "planar", u, p, R_(1, 20), R_(5, 4), (0, 1, 0, 1), "mean_zero",
                 _box_bnds("planar", ("dirichlet",) * 4), 1,
                 "planar space-exact field with a tanh front at t = 1/2 (width 1/10): error control", space_exact=True)


def tr_axi_poly():
    """Axisymmetric with swirl, SPACE-EXACT: u_r = r g(z, t), u_theta = r (e + h z), u_z = -2 a z - b z^2 + c r^2 + d
    (all P2 in (r, z), parity-respecting, regular at the axis), p = k(t) (z - 1/2) (P1, even in r, r-weighted mean 0);
    Dirichlet on r = 1, z = 0, z = 1, axis conditions on r = 0."""
    al = R_(1, 2) * sp.sin(2 * t + R_(1, 3))
    be = R_(3, 4) * sp.exp(-t / 2) * sp.cos(3 * t / 2 - R_(1, 5))
    ga = R_(1, 3) + R_(1, 4) * sp.cos(5 * t / 2 + R_(1, 7))
    de = R_(1, 4) * sp.sin(t + R_(2, 5))
    ep = 1 + R_(1, 3) * sp.cos(2 * t - R_(1, 2))
    ze = R_(1, 2) * sp.sin(3 * t / 2 + R_(1, 6))
    ur = r * (al + be * z)
    uz = -2 * al * z - be * z ** 2 + ga * r ** 2 + de
    ut = r * (ep + ze * z)
    p = (R_(1, 2) * sp.cos(t + R_(1, 4))) * (z - R_(1, 2))
    return Field("f3a_tr_axi_poly", "axisymmetric", [ur, ut, uz], p, R_(1, 20), R_(5, 4), (0, 1, 0, 1), "mean_zero",
                 _box_bnds("axisymmetric", ("axis", "dirichlet", "dirichlet", "dirichlet")), 1,
                 "axisymmetric space-exact temporal MMS with swirl (u in P2, p in P1), rho = 5/4, mu = 1/20",
                 space_exact=True)


def tr_sq_proj():
    """Initial-data projection MMS: psi = sin^2(pi x) sin^2(pi y) exp(x y / 2) / pi (NON-separable: the nodal
    interpolant of u = curl psi is not discretely divergence-free on the fixed meshes, unlike the Taylor-Green or
    Womersley fields), u = a(t) curl psi vanishes on the boundary of [0,1]^2 with its flux; p non-polynomial;
    rho = 5/4, mu = 1/20."""
    a = 1 + R_(1, 2) * sp.sin(t + R_(1, 4))
    psi = sp.sin(sp.pi * x) ** 2 * sp.sin(sp.pi * y) ** 2 * sp.exp(x * y / 2) / sp.pi
    u = [a * sp.diff(psi, y), -a * sp.diff(psi, x)]
    p = sp.cos(sp.pi * x) * sp.sin(sp.pi * y / 2 + R_(1, 5)) * (1 + R_(1, 3) * sp.cos(2 * t))
    return Field("f3a_tr_sq_proj", "planar", u, p, R_(1, 20), R_(5, 4), (0, 1, 0, 1), "mean_zero",
                 _box_bnds("planar", ("dirichlet",) * 4), R_(1, 100),
                 "initial-projection MMS: non-separable stream function, u = 0 on the boundary, rho = 5/4, mu = 1/20")


def tr_sq_energy():
    """Energy identity, planar: u = 0 on the boundary of [0,1]^2, u(0) = 0, time-dependent polynomial force with
    curl != 0; no exact solution."""
    s = (1 - sp.cos(sp.pi * t)) / 2 + R_(1, 4) * sp.sin(3 * t)
    f = [-3 * s * (y - R_(2, 5)), 3 * s * R_(6, 5) * (x - R_(3, 5))]
    return Field("f3a_tr_sq_energy", "planar", [sp.Integer(0)] * 2, sp.Integer(0), R_(1, 100), R_(5, 4), (0, 1, 0, 1),
                 "mean_zero", _box_bnds("planar", ("dirichlet",) * 4), 1,
                 "energy identity, planar: homogeneous Dirichlet, u(0) = 0, f polynomial in space", homogeneous=True,
                 f_given=f)


def tr_axi_energy():
    s1 = (1 - sp.cos(sp.pi * t)) / 2 + R_(1, 4) * sp.sin(3 * t)
    s2 = R_(1, 2) + R_(1, 2) * sp.sin(2 * t - R_(1, 3))
    f = [sp.Integer(0), 3 * r * s1, 3 * r ** 2 * s2]
    return Field("f3a_tr_axi_energy", "axisymmetric", [sp.Integer(0)] * 3, sp.Integer(0), R_(1, 100), R_(5, 4),
                 (0, 1, 0, 1), "mean_zero", _box_bnds("axisymmetric", ("axis", "dirichlet", "dirichlet", "dirichlet")),
                 1, "energy identity, axisymmetric with swirl: homogeneous Dirichlet, u(0) = 0, f = (0, 3r s1, 3r^2 s2)",
                 homogeneous=True, f_given=f)


def tr_taylor_green():
    """Decaying Taylor-Green vortex on [0,1]^2, TRANSLATED by (a, b) = (3/10, 3/20) so that the boundary data have
    u.n of both signs and the mesh symmetry does not make the nodal interpolant discretely divergence-free (the
    untranslated vortex's interpolant is, to round-off, on the fixed meshes: the initial-projection check would be
    blind).  Exact for every Re, f = 0 for both viscous forms; (u.grad)u is a pure gradient balanced by the pressure,
    which the pressure items therefore test.  Kinetic energy rho/4 F^2 (full periods)."""
    rho, mu = sp.Integer(1), R_(1, 20)
    nu = mu / rho
    F = sp.exp(-2 * sp.pi ** 2 * nu * t)
    X_, Y_ = x + R_(3, 10), y + R_(3, 20)
    u = [sp.sin(sp.pi * X_) * sp.cos(sp.pi * Y_) * F, -sp.cos(sp.pi * X_) * sp.sin(sp.pi * Y_) * F]
    p = rho / 4 * (sp.cos(2 * sp.pi * X_) + sp.cos(2 * sp.pi * Y_)) * F ** 2
    return Field("f3a_tr_taylor_green", "planar", u, p, mu, rho, (0, 1, 0, 1), "mean_zero",
                 _box_bnds("planar", ("dirichlet",) * 4), R_(1, 50),
                 "decaying Taylor-Green vortex translated by (3/10, 3/20), nu = 1/20 (Re = U L / nu = 20), exact "
                 "Dirichlet data", exact_ns=True)


# Womersley: K = omega (pressure-gradient amplitude per unit density), alpha = 4, omega = 2 pi, rho = 5/4
W_ALPHA, W_RHO = 4, R_(5, 4)


def _wom_consts():
    om = 2 * sp.pi
    nu = om / W_ALPHA ** 2
    return om, nu, W_RHO * nu, om


def tr_womersley_2d():
    """Planar Womersley flow in the channel |y| <= 1, x in [0, 1]: -dp/dx = rho K cos(omega t),
    u = Re[(K / (i omega)) (1 - cosh(lam y) / cosh(lam)) e^(i omega t)], lam = sqrt(i omega / nu) = (1 + i) alpha / sqrt 2.
    Walls no-slip; x = 0 and x = 1 traction sigma_form n = g (exact)."""
    om, nu, mu, K = _wom_consts()
    lam = (1 + I) * W_ALPHA / sp.sqrt(2)
    Uhat = (K / (I * om)) * (1 - sp.cosh(lam * y) / sp.cosh(lam))
    U = sp.re(Uhat * sp.exp(I * om * t))
    p = W_RHO * K * sp.cos(om * t) * (R_(1, 2) - x)
    lamc = complex(sp.N(lam, 20))
    Kc = float(sp.N(K / om, 20))
    npU = (f"np.real(({-Kc!r}j) * (1 - np.cosh(({lamc!r}) * y) / np.cosh({lamc!r})) * np.exp(1j * {float(sp.N(om, 20))!r} * t))")
    npUy = (f"np.real(({-Kc!r}j) * (-({lamc!r}) * np.sinh(({lamc!r}) * y) / np.cosh({lamc!r})) "
            f"* np.exp(1j * {float(sp.N(om, 20))!r} * t))")
    return Field("f3a_tr_womersley_2d", "planar", [U, sp.Integer(0)], p, mu, W_RHO, (0, 1, -1, 1), "determined",
                 [{"name": "x0", "kind": "traction", "where": "x = 0", "normal": [-1, 0]},
                  {"name": "x1", "kind": "traction", "where": "x = 1", "normal": [1, 0]},
                  {"name": "y0", "kind": "dirichlet", "where": "y = -1", "normal": [0, -1]},
                  {"name": "y1", "kind": "dirichlet", "where": "y = 1", "normal": [0, 1]}], 1,
                 "planar Womersley flow, alpha = 4, omega = 2 pi (period 1), rho = 5/4, K = omega",
                 exact_ns=True, numpy_override={"u0": npU, "du0_dy": npUy},
                 extra={"alpha": W_ALPHA, "omega": "2 pi", "nu": "omega / alpha^2", "K": "omega",
                        "Uhat_sympy": str(Uhat)})


def tr_womersley_axi():
    """Axisymmetric Womersley flow in the pipe r <= 1, z in [0, 1]: -dp/dz = rho K cos(omega t),
    u_z = Re[(K / (i omega)) (1 - J0(k r) / J0(k)) e^(i omega t)], k = i^(3/2) alpha; u_r = u_theta = 0.
    Wall r = 1 no-slip, axis conditions, z = 0 and z = 1 traction (exact)."""
    om, nu, mu, K = _wom_consts()
    k = W_ALPHA * sp.exp(3 * sp.pi * I / 4)
    Uhat = (K / (I * om)) * (1 - sp.besselj(0, k * r) / sp.besselj(0, k))
    Uz = sp.re(Uhat * sp.exp(I * om * t))
    p = W_RHO * K * sp.cos(om * t) * (R_(1, 2) - z)
    kc = complex(sp.N(k, 20))
    J0k = complex(mp.besselj(0, mp.mpc(kc.real, kc.imag)))
    Kc = float(sp.N(K / om, 20))
    npU = (f"np.real(({-Kc!r}j) * (1 - jv(0, ({kc!r}) * r) / ({J0k!r})) * np.exp(1j * {float(sp.N(om, 20))!r} * t))")
    npUr = (f"np.real(({-Kc!r}j) * (({kc!r}) * jv(1, ({kc!r}) * r) / ({J0k!r})) * np.exp(1j * {float(sp.N(om, 20))!r} * t))")
    return Field("f3a_tr_womersley_axi", "axisymmetric", [sp.Integer(0), sp.Integer(0), Uz], p, mu, W_RHO,
                 (0, 1, 0, 1), "determined",
                 [{"name": "axis", "kind": "axis", "where": "r = 0", "normal": [-1, 0]},
                  {"name": "r1", "kind": "dirichlet", "where": "r = 1", "normal": [1, 0]},
                  {"name": "z0", "kind": "traction", "where": "z = 0", "normal": [0, -1]},
                  {"name": "z1", "kind": "traction", "where": "z = 1", "normal": [0, 1]}], 1,
                 "axisymmetric Womersley pipe flow, alpha = 4, omega = 2 pi, rho = 5/4, K = omega",
                 exact_ns=True, numpy_override={"uz": npU, "duz_dr": npUr},
                 extra={"alpha": W_ALPHA, "omega": "2 pi", "nu": "omega / alpha^2", "K": "omega", "J0_k": [J0k.real, J0k.imag],
                        "Uhat_sympy": str(Uhat)})


def _box_bnds(geometry, kinds):
    if geometry == "planar":
        names = [("x0", "x = 0", [-1, 0]), ("x1", "x = 1", [1, 0]), ("y0", "y = 0", [0, -1]), ("y1", "y = 1", [0, 1])]
    else:
        names = [("axis", "r = 0", [-1, 0]), ("r1", "r = 1", [1, 0]), ("z0", "z = 0", [0, -1]), ("z1", "z = 1", [0, 1])]
    return [{"name": n, "kind": k, "where": w, "normal": nv} for (n, w, nv), k in zip(names, kinds)]


def tr_cavity_x1():
    """Cross-code start-up cavity: u = 0 at t = 0; regularized lid u = 16 x^2 (1 - x)^2 s(t) on y = 1 (zero at the lid
    corners, so no corner rule), s = smoothstep t^2 (3 - 2t) for t < 1, then 1; walls at rest; rho = 1, mu = 1/400.
    The emitted u is (16 x^2 (1 - x)^2 y s(t), 0): its BOUNDARY values are the data (0 on x = 0, 1 and y = 0); the
    interior values mean nothing (no exact solution)."""
    s_ = sp.Piecewise((t ** 2 * (3 - 2 * t), t < 1), (1, True))
    u = [16 * x ** 2 * (1 - x) ** 2 * y * s_, sp.Integer(0)]
    return Field("f3a_tr_cavity_x1", "planar", u, sp.Integer(0), R_(1, 400), sp.Integer(1), (0, 1, 0, 1), "mean_zero",
                 _box_bnds("planar", ("dirichlet",) * 4), 2,
                 "cross-code start-up cavity, Re = 400 (lid max 1, L = 1), regularized lid, smoothstep ramp",
                 data_only=True)


ALL = [tr_cavity_x1, tr_sq_proj, tr_sq_poly, tr_sq_adapt, tr_axi_poly, tr_sq_energy, tr_axi_energy, tr_taylor_green, tr_womersley_2d,
       tr_womersley_axi]


# ============================================================================================ emission
def spell(expr, coords):
    s = sp.printing.numpy.NumPyPrinter({"fully_qualified_modules": True}).doprint(sp.sympify(expr))
    s = s.replace("numpy.", "np.")
    return s


def emit(fld: Field) -> dict:
    c = fld.coords
    cs = list(c) + ["t"]
    d = {"name": fld.name, "geometry": fld.geometry, "coords": cs, "description": fld.description,
         "box": [float(v) for v in fld.box], "T": float(fld.T), "rho": float(fld.rho), "mu": float(fld.mu),
         "constants": {"rho": str(fld.rho), "mu": str(fld.mu), "T": str(fld.T)},
         "pressure": fld.pressure, "boundaries": fld.boundaries, "exact_ns": fld.exact_ns,
         "space_exact": fld.space_exact, "homogeneous": fld.homogeneous, "data_only": fld.data_only,
         "extra": fld.extra}
    ov = fld.numpy_override

    def comp(e, key=None):
        return {"sympy": str(e), "numpy": ov[key] if key and key in ov else spell(e, cs)}
    if fld.name == "f3a_tr_womersley_2d":
        d["u"] = [comp(fld.u[0], "u0"), comp(fld.u[1])]
        d["grad_u"] = [[comp(sp.Integer(0)), comp(sp.diff(fld.u[0], y), "du0_dy")], [comp(sp.Integer(0))] * 2]
    elif fld.name == "f3a_tr_womersley_axi":
        d["u"] = [comp(sp.Integer(0)), comp(sp.Integer(0)), comp(fld.u[2], "uz")]
        d["grad_u"] = [[comp(sp.Integer(0))] * 2, [comp(sp.Integer(0))] * 2,
                       [comp(sp.diff(fld.u[2], r), "duz_dr"), comp(sp.Integer(0))]]
    else:
        d["u"] = [comp(e) for e in fld.u]
        d["grad_u"] = [[comp(e) for e in row] for row in grad(fld)]
    d["p"] = comp(fld.p)
    d["forms"] = {}
    for form in FORMS:
        if fld.exact_ns or fld.data_only:
            f = sp.zeros(len(fld.u), 1)
        else:
            f = forcing(fld, form)
        ent = {"f": [comp(sp.expand(e) if fld.space_exact or fld.homogeneous else e) for e in f]}
        if not fld.homogeneous and not fld.exact_ns and not fld.data_only:
            # forcing with the convection or the time derivative removed: the self-check (step 7) confirms that the
            # finite-difference residual detects either omission
            ent["f_without_convection"] = [comp(e) for e in forcing(fld, form, convection=False)]
            ent["f_without_time"] = [comp(e) for e in forcing(fld, form, time=False)]
        if any(b["kind"] == "traction" for b in fld.boundaries):
            mu = fld.mu
            zero = comp(sp.Integer(0))
            if fld.name == "f3a_tr_womersley_2d":
                # u = (U(y, t), 0): G = [[0, U_y], [0, 0]]; lap sigma = mu G - p I, sym sigma = mu (G + G^T) - p I
                tau = {"sympy": f"{mu}*Derivative(u_x, y)", "numpy": f"{float(mu)!r} * ({ov['du0_dy']})"}
                mp_ = comp(-fld.p)
                ent["sigma"] = [[mp_, tau], [tau if form == "sym" else zero, mp_]]
            elif fld.name == "f3a_tr_womersley_axi":
                # u = (0, 0, U(r, t)): G_zr = U_r only; lap sigma_zr = mu U_r, sigma_rz = 0; sym both
                tau = {"sympy": f"{mu}*Derivative(u_z, r)", "numpy": f"{float(mu)!r} * ({ov['duz_dr']})"}
                mp_ = comp(-fld.p)
                ent["sigma"] = [[mp_, zero, tau if form == "sym" else zero], [zero, mp_, zero], [tau, zero, mp_]]
            else:
                ent["sigma"] = [[comp(e) for e in row] for row in stress(fld, form).tolist()]
        d["forms"][form] = ent
    return d


# ============================================================================================ self-checks
def _rand_pts(fld, k=8, seed=7):
    rnd = random.Random(seed)
    x0, x1, y0, y1 = (float(v) for v in fld.box)
    out = []
    for _ in range(k):
        a = x0 + (x1 - x0) * rnd.uniform(0.05, 0.95)
        b = y0 + (y1 - y0) * rnd.uniform(0.05, 0.95)
        tt = rnd.uniform(0.0, float(fld.T) if fld.T else 1.0)
        out.append((mp.mpf(a), mp.mpf(b), mp.mpf(tt)))
    return out


def _ev(e, fld, P, dps=50):
    with mp.workdps(dps):
        f = sp.lambdify((*fld.syms, t), e, modules="mpmath")
        v = f(*P)
        return mp.re(v) if isinstance(v, mp.mpc) else v


def selfcheck(fld: Field, em: dict, log=print) -> dict:
    import numpy as np
    from scipy.special import jv
    rep = {"field": fld.name, "checks": {}}
    ok = True

    def note(key, val, good):
        nonlocal ok
        rep["checks"][key] = {"value": val if isinstance(val, (str, bool, int)) else float(val), "pass": bool(good)}
        ok = ok and bool(good)
        log(f"  {fld.name}: {key} = {val} {'ok' if good else 'FAIL'}")

    mp.mp.dps = 50
    pts = _rand_pts(fld)
    if fld.data_only:
        import numpy as np
        loc = {"x": np.array([0.0, 1.0, 0.3, 0.5, 0.0]), "y": np.array([0.4, 0.7, 0.0, 1.0, 1.0]),
               "t": np.array([0.5, 0.5, 0.5, 2.0, 0.7])}
        v = np.asarray(eval(em["u"][0]["numpy"], {"__builtins__": {}, "np": np}, loc), float)
        want = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        note("boundary_data_values", float(np.max(np.abs(v - want))), float(np.max(np.abs(v - want))) < 1e-14)
        rep["status"] = "pass" if ok else "fail"
        return rep
    # 1. divergence free
    dv = div2(fld.u) if fld.geometry == "planar" else divA(fld.u)
    if fld.name.startswith("f3a_tr_womersley"):
        note("div_u_numeric", max(abs(_ev(dv, fld, P)) for P in pts), max(abs(_ev(dv, fld, P)) for P in pts) < 1e-40)
    else:
        note("div_u_symbolic_zero", str(sp.simplify(dv)), sp.simplify(dv) == 0)
    # 2. forcing: two routes
    for form in FORMS:
        if fld.exact_ns and fld.name.startswith("f3a_tr_womersley"):
            if form == "sym":
                continue
            om, nu, mu, K = _wom_consts()
            s_ = fld.syms[1] if fld.geometry == "planar" else fld.syms[0]
            Uh = sp.sympify(fld.extra["Uhat_sympy"], locals={"y": y, "r": r})
            lap = sp.diff(Uh, s_, 2) + (sp.diff(Uh, s_) / s_ if fld.geometry == "axisymmetric" else 0)
            ode = I * om * Uh - K - nu * lap
            m = max(abs(complex(sp.N(ode.subs(s_, P[1] if fld.geometry == "planar" else P[0]), 40))) for P in pts)
            note("womersley_amplitude_ode_residual", m, m < 1e-30)
            wall = [sp.N(Uh.subs(s_, v), 40) for v in ((-1, 1) if fld.geometry == "planar" else (1,))]
            note("womersley_wall_value", max(abs(complex(w_)) for w_ in wall), max(abs(complex(w_)) for w_ in wall) < 1e-30)
            continue
        if fld.exact_ns:
            fa = forcing2_a(fld.u, fld.p, fld.mu, fld.rho, form) if fld.geometry == "planar" else \
                forcingA_a(fld.u, fld.p, fld.mu, fld.rho, form)
            m = max(abs(_ev(c_, fld, P)) for P in pts for c_ in fa)
            note(f"exact_ns_residual_{form}", m, m < 1e-35)
            continue
        if fld.homogeneous:
            continue
        fa = forcing(fld, form)
        if fld.geometry == "planar":
            fb = forcing2_b(fld.u, fld.p, fld.mu, fld.rho, form)
            dif = [sp.simplify(sp.expand(fa[i] - fb[i])) for i in range(2)]
            note(f"forcing_routes_agree_{form}", str(dif), all(d_ == 0 for d_ in dif))
        else:
            fb = forcingA_cartesian(fld.u, fld.p, fld.mu, fld.rho, form)
            rnd = random.Random(11)
            m = mp.mpf(0)
            for P in pts:
                phi = mp.mpf(rnd.uniform(0, 6.28))
                va = [_ev(c_, fld, P) for c_ in fa]
                vb = fb(P[0], P[1], P[2], phi)
                m = max(m, max(abs(a_ - b_) for a_, b_ in zip(va, vb)))
            note(f"forcing_3d_embedding_{form}", m, m < 1e-35)
    # 3. emitted numpy spellings evaluate to the sympy values
    ns = {"np": np, "jv": jv}
    rng = np.random.default_rng(3)
    x0, x1, y0, y1 = (float(v) for v in fld.box)
    A = x0 + (x1 - x0) * rng.uniform(0.02, 0.98, 25)
    B = y0 + (y1 - y0) * rng.uniform(0.02, 0.98, 25)
    TT = rng.uniform(0, float(fld.T), 25)
    loc = dict(zip(em["coords"], (A, B, TT)))
    worst = 0.0
    pairs = [(em["u"][i], fld.u[i]) for i in range(len(fld.u))] + [(em["p"], fld.p)]
    for form in FORMS:
        if not fld.exact_ns:
            pairs += list(zip(em["forms"][form]["f"], forcing(fld, form)))
    for spec, e in pairs:
        v = np.broadcast_to(np.asarray(eval(spec["numpy"], {"__builtins__": {}, **ns}, loc), float), A.shape)
        f = sp.lambdify((*fld.syms, t), e, modules=["mpmath"])
        ref = np.array([float(mp.re(f(a_, b_, c_))) for a_, b_, c_ in zip(A, B, TT)])
        worst = max(worst, float(np.max(np.abs(v - ref))))
    note("numpy_spelling_vs_sympy_max", worst, worst < 1e-11)
    # 4. space-exactness: u_i polynomial of total degree <= 2 in space, p of degree <= 1 (axi: u_r / r, u_t / r polynomial)
    if fld.space_exact:
        degs = [sp.Poly(sp.expand(c_), *fld.syms).total_degree() for c_ in fld.u]
        dp = sp.Poly(sp.expand(fld.p), *fld.syms).total_degree()
        note("space_degrees_u_le2_p_le1", f"{degs} / {dp}", max(degs) <= 2 and dp <= 1)
        if fld.geometry == "axisymmetric":
            reg = all(sp.simplify(sp.expand(c_ / r)).is_polynomial(r, z) for c_ in fld.u[:2])
            par = (sp.simplify(fld.u[0].subs(r, -r) + fld.u[0]) == 0 and sp.simplify(fld.u[1].subs(r, -r) + fld.u[1]) == 0
                   and sp.simplify(fld.u[2].subs(r, -r) - fld.u[2]) == 0 and sp.simplify(fld.p.subs(r, -r) - fld.p) == 0)
            note("axi_parity_and_regularity", f"{reg} / {par}", reg and par)
            mean = sp.integrate(sp.integrate(fld.p * r, (r, 0, 1)), (z, 0, 1))
            note("p_r_weighted_mean_zero", str(sp.simplify(mean)), sp.simplify(mean) == 0)
        else:
            mean = sp.integrate(sp.integrate(fld.p, (x, 0, 1)), (y, 0, 1))
            note("p_mean_zero", str(sp.simplify(mean)), sp.simplify(mean) == 0)
        # forcing polynomial in space: exact quadrature possible (degree reported)
        for form in FORMS:
            fd = [sp.Poly(sp.expand(c_), *fld.syms).total_degree() for c_ in forcing(fld, form)]
            note(f"forcing_space_degree_{form}", str(fd), max(fd) <= 3)
        # 5. teeth: the forcing without convection / without du/dt differs (FD-free: symbolic, at random points)
        for kw, lab in (({"convection": False}, "no_convection"), ({"time": False}, "no_time")):
            fa = forcing(fld, "lap")
            fb = forcing(fld, "lap", **kw)
            m = max(abs(_ev(fa[i] - fb[i], fld, P)) for P in pts for i in range(len(fld.u)))
            note(f"teeth_{lab}_max", m, m > 1e-2)
        # 6. non-Beltrami (planar: curl of the convective term != 0; axi: meridional curl, centrifugal not a gradient,
        #    Coriolis != 0)
        if fld.geometry == "planar":
            cv = grad2(fld.u) * sp.Matrix(fld.u)
            curl = sp.diff(cv[1], x) - sp.diff(cv[0], y)
            m = max(abs(_ev(curl, fld, P)) for P in pts)
            note("curl_convective_max", m, m > 1e-2)
        else:
            cv = gradA(fld.u) * sp.Matrix(fld.u)
            curl = sp.diff(cv[0], z) - sp.diff(cv[2], r)
            m = max(abs(_ev(curl, fld, P)) for P in pts)
            note("meridional_curl_convective_max", m, m > 1e-2)
            cen = sp.diff(fld.u[1] ** 2 / r, z)
            m = max(abs(_ev(cen, fld, P)) for P in pts)
            note("dz_centrifugal_max", m, m > 1e-2)
            cor = fld.u[0] * fld.u[1] / r
            m = max(abs(_ev(cor, fld, P)) for P in pts)
            note("coriolis_max", m, m > 1e-2)
        # enclosed flow: zero net boundary flux exactly
        if fld.geometry == "planar":
            flux = (sp.integrate(fld.u[0].subs(x, 1) - fld.u[0].subs(x, 0), (y, 0, 1))
                    + sp.integrate(fld.u[1].subs(y, 1) - fld.u[1].subs(y, 0), (x, 0, 1)))
        else:
            flux = (sp.integrate((fld.u[0] * r).subs(r, 1), (z, 0, 1))
                    + sp.integrate((fld.u[2] * r).subs(z, 1) - (fld.u[2] * r).subs(z, 0), (r, 0, 1)))
        note("net_boundary_flux", str(sp.simplify(flux)), sp.simplify(flux) == 0)
    if fld.homogeneous:
        f = sp.Matrix(fld.f_given)
        fd = [sp.Poly(sp.expand(c_), *fld.syms).total_degree() for c_ in f]
        note("force_space_degree", str(fd), max(fd) <= 2)
        if fld.geometry == "planar":
            cf = sp.diff(f[1], x) - sp.diff(f[0], y)
        else:
            cf = sp.diff(f[0], z) - sp.diff(f[2], r)
        note("curl_f_nonzero", str(sp.simplify(cf)), sp.simplify(cf) != 0)
    # 7. finite-difference residual from the EMITTED spellings only, per viscous form; the forcing variants without
    #    convection or time derivative must fail it (the check is sensitive to both terms)
    if not fld.homogeneous:
        for form in FORMS:
            m = fd_residual(fld, em, form)
            note(f"fd_residual_{form}", m, m < 1e-6)
        if not fld.exact_ns:
            for fk in ("f_without_convection", "f_without_time"):
                m = fd_residual(fld, em, "lap", fk)
                note(f"fd_residual_teeth_{fk}", m, m > 1e-2)
    rep["status"] = "pass" if ok else "fail"
    return rep


def fd_residual(fld, em, form="lap", fkey="f"):
    """rho(dU/dt + (U.grad)U) - mu (lap U [+ grad div U for sym]) + grad P - F by 4th-order central differences of the
    EMITTED numpy spellings only (constant mu), in Cartesian coordinates: planar directly, axisymmetric through the
    3D embedding U = u_r e_r + u_t e_t + u_z e_z at (r cos phi, r sin phi, z), F likewise from the emitted forcing.
    Returns the max residual and, with the forcing variant fkey, how far a wrong forcing is from zero."""
    import numpy as np
    from scipy.special import jv
    ns = {"np": np, "jv": jv}
    cs = em["coords"]

    def comp(spec):
        code = compile(spec["numpy"], "<f>", "eval")
        return lambda a, b, c: float(np.real(eval(code, {"__builtins__": {}, **ns}, dict(zip(cs, (a, b, c))))))
    u = [comp(s_) for s_ in em["u"]]
    P0 = comp(em["p"])
    fsp = em["forms"][form].get(fkey) if not fld.exact_ns else None
    fc = [comp(s_) for s_ in fsp] if fsp else None
    rho, mu = float(fld.rho), float(fld.mu)
    axi = fld.geometry == "axisymmetric"
    dim = 3 if axi else 2

    def Ucart(X, tt):
        if not axi:
            return np.array([u[0](X[0], X[1], tt), u[1](X[0], X[1], tt)])
        rr = float(np.hypot(X[0], X[1]))
        c_, s_ = X[0] / rr, X[1] / rr
        ur, ut, uz = (f(rr, X[2], tt) for f in u)
        return np.array([ur * c_ - ut * s_, ur * s_ + ut * c_, uz])

    def Pc(X, tt):
        return P0(X[0], X[1], tt) if not axi else P0(float(np.hypot(X[0], X[1])), X[2], tt)

    def Fc(X, tt):
        if fc is None:
            return np.zeros(dim)
        if not axi:
            return np.array([fc[0](X[0], X[1], tt), fc[1](X[0], X[1], tt)])
        rr = float(np.hypot(X[0], X[1]))
        c_, s_ = X[0] / rr, X[1] / rr
        fr, ft, fz = (f(rr, X[2], tt) for f in fc)
        return np.array([fr * c_ - ft * s_, fr * s_ + ft * c_, fz])
    h = 2e-3
    W1 = {2: 8 / 12, 1: -1 / 12}

    def d1(fun, X, tt, ax):
        out = 0.0
        for k, w in ((1, 8 / 12), (2, -1 / 12)):
            if ax == dim:
                out = out + w * (fun(X, tt + k * h) - fun(X, tt - k * h))
            else:
                e = np.zeros(dim)
                e[ax] = k * h
                out = out + w * (fun(X + e, tt) - fun(X - e, tt))
        return out / h

    def d2(fun, X, tt, a, b):
        return d1(lambda Y, s: d1(fun, Y, s, b), X, tt, a)
    rng = np.random.default_rng(5)
    x0, x1, y0, y1 = (float(v) for v in fld.box)
    worst = 0.0
    for _ in range(6):
        a = x0 + (x1 - x0) * rng.uniform(0.15, 0.85)
        b = y0 + (y1 - y0) * rng.uniform(0.15, 0.85)
        tt = rng.uniform(0.2, 0.8) * float(fld.T)
        if axi:
            ph = rng.uniform(0, 2 * np.pi)
            X = np.array([a * np.cos(ph), a * np.sin(ph), b])
        else:
            X = np.array([a, b])
        U = Ucart(X, tt)
        G = np.array([d1(Ucart, X, tt, j) for j in range(dim)]).T          # G[i, j] = d U_i / d X_j
        lap = sum(d2(Ucart, X, tt, j, j) for j in range(dim))
        res = rho * (d1(Ucart, X, tt, dim) + G @ U) - mu * lap + np.array([d1(Pc, X, tt, j) for j in range(dim)]) \
            - Fc(X, tt)
        if form == "sym":
            divU = lambda Y, s: sum(d1(Ucart, Y, s, j)[j] for j in range(dim))  # noqa: E731
            res = res - mu * np.array([d1(divU, X, tt, j) for j in range(dim)])
        worst = max(worst, float(np.max(np.abs(res))))
    return worst


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
