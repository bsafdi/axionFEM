"""F4a (scalar transport with SUPG; Boussinesq; axisymmetric) verification fields: operators, fields, Graetz modes,
code generation and self-checks.

Reference generator.  The PDEs are written from the continuum equations, NOT from any solver code.

Scalar transport (heat: a = rho c_p, kappa = k, s = Q; species (chemistry C2): a = 1, kappa = D, sigma = first-order
rate), steady, CONVECTIVE form (required) and CONSERVATIVE form (offered => tested):

    convective:    a b.grad T - div(kappa grad T) + sigma T = s
    conservative:  div(a b T - kappa grad T)    + sigma T = s        (identical when div b = 0)

    Neumann:  q_out = -kappa grad T . n          (outward DIFFUSIVE flux; convective form's natural condition)
              j_out = (a b T - kappa grad T) . n (outward TOTAL flux; conservative form's natural condition)
    Robin:    -kappa grad T . n = h (T - T_ext)

Boussinesq (constant rho0, c_p, beta; mu(x), k(x) variable; gravity vector g_vec = -g e_up, e_up = e_y planar,
e_z axisymmetric):

    rho0 (u.grad)u - div sigma_form(u, p) = f_u + rho0 beta g (T - T_ref) e_up,     div u = 0
    rho0 c_p u.grad T - div(k grad T) = Q

Axisymmetric: cylindrical (r, theta, z), fields independent of theta; a scalar T(r, z) is EVEN in r; the advecting
velocity's meridional part (b_r odd, b_z even) is the only part that advects T;
    b.grad T = b_r d_r T + b_z d_z T,     div(kappa grad T) = (1/r) d_r (r kappa d_r T) + d_z (kappa d_z T).
Every axisymmetric forcing is also derived by a second route (3D Cartesian embedding; agreement to 1e-40 at random
points, 60 digits); every forcing passes a 4th-order finite-difference residual check from the primitive fields
only (3D Cartesian for axisymmetry), with teeth.

Emitted spellings: "sympy" (exact str), "c" (C99), "numpy" (np.* in the coordinate names of the field).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field as dfield

import mpmath as mp
import numpy as np
import sympy as sp

x, y = sp.symbols("x y", real=True)
r, z = sp.symbols("r z", real=True)
R_ = sp.Rational
FORMS = ("lap", "sym")
ADV_FORMS = ("convective", "conservative")


class CheckFailed(RuntimeError):
    pass


# ============================================================================================ scalar operators
def coords_of(geometry):
    return (x, y) if geometry == "planar" else (r, z)


def grad_s(T, geometry):
    a, b = coords_of(geometry)
    return sp.Matrix([sp.diff(T, a), sp.diff(T, b)])


def div_v(w, geometry):
    """Divergence of a meridional/planar vector field (w_0, w_1)."""
    if geometry == "planar":
        return sp.diff(w[0], x) + sp.diff(w[1], y)
    return sp.diff(r * w[0], r) / r + sp.diff(w[1], z)


def scalar_forcing(T, b, kappa, a, sigma, geometry, adv_form="convective", advection=True, diffusion=True):
    """Route A (vector calculus): s = a b.grad T - div(kappa grad T) + sigma T (convective) or
    div(a b T - kappa grad T) + sigma T (conservative)."""
    g = grad_s(T, geometry)
    s = sigma * T
    if advection:
        if adv_form == "convective":
            s += a * (b[0] * g[0] + b[1] * g[1])
        else:
            s += div_v(sp.Matrix([a * b[0] * T, a * b[1] * T]), geometry)
    if diffusion:
        s -= div_v(kappa * g, geometry)
    return s


def scalar_forcing_b(T, b, kappa, a, sigma, geometry):
    """Route B (written out by the product rule, independently):
       planar: s = a (b_x T_x + b_y T_y) - kappa (T_xx + T_yy) - (kappa_x T_x + kappa_y T_y) + sigma T
       axi:    s = a (b_r T_r + b_z T_z) - kappa (T_rr + T_r / r + T_zz) - (kappa_r T_r + kappa_z T_z) + sigma T"""
    c0, c1 = coords_of(geometry)
    T0, T1 = sp.diff(T, c0), sp.diff(T, c1)
    lap = sp.diff(T, c0, 2) + sp.diff(T, c1, 2) + (T0 / r if geometry == "axisymmetric" else 0)
    return (a * (b[0] * T0 + b[1] * T1) - kappa * lap - (sp.diff(kappa, c0) * T0 + sp.diff(kappa, c1) * T1)
            + sigma * T)


# ============================================================================================ NS operators (F2a forms)
def grad2(u):
    return sp.Matrix(2, 2, lambda i, j: sp.diff(u[i], (x, y)[j]))


def divT2(S):
    return sp.Matrix([sp.diff(S[i, 0], x) + sp.diff(S[i, 1], y) for i in range(2)])


def gradA(u):
    ur, ut, uz = u
    return sp.Matrix([[sp.diff(ur, r), -ut / r, sp.diff(ur, z)],
                      [sp.diff(ut, r), ur / r, sp.diff(ut, z)],
                      [sp.diff(uz, r), 0, sp.diff(uz, z)]])


def divTA(S):
    return sp.Matrix([sp.diff(S[0, 0], r) + sp.diff(S[0, 2], z) + (S[0, 0] - S[1, 1]) / r,
                      sp.diff(S[1, 0], r) + sp.diff(S[1, 2], z) + (S[1, 0] + S[0, 1]) / r,
                      sp.diff(S[2, 0], r) + sp.diff(S[2, 2], z) + S[2, 0] / r])


def stress(u, p, mu, form, geometry):
    G = grad2(u) if geometry == "planar" else gradA(u)
    n = G.shape[0]
    return (mu * (G + G.T) if form == "sym" else mu * G) - p * sp.eye(n)


def momentum_forcing(u, p, mu, rho0, form, geometry, T=None, beta_g=0, T_ref=0, convection=True, buoyancy=True,
                     buoyancy_sign=1):
    """f_u = rho0 G u - div sigma_form - rho0 beta g (T - T_ref) e_up (so that the PDE reads
    rho0 (u.grad)u - div sigma = f_u + rho0 beta g (T - T_ref) e_up)."""
    if geometry == "planar":
        G = grad2(u)
        f = -divT2(stress(u, p, mu, form, geometry))
        up = 1
    else:
        G = gradA(u)
        f = -divTA(stress(u, p, mu, form, geometry))
        up = 2
    if convection:
        f += rho0 * G * sp.Matrix(u)
    if T is not None and buoyancy:
        f[up] -= buoyancy_sign * rho0 * beta_g * (T - T_ref)
    return f


def energy_forcing(T, u, k, a, geometry, advection=True):
    """Q = a u.grad T - div(k grad T) with the meridional/planar velocity components."""
    b = (u[0], u[1]) if geometry == "planar" else (u[0], u[2])
    return scalar_forcing(T, b, k, a, 0, geometry, "convective", advection=advection)


# ============================================================================================ fields
@dataclass
class ScalarField:
    name: str
    geometry: str
    T: sp.Expr
    b: tuple
    kappa: sp.Expr
    a: sp.Expr
    box: tuple                                # (c0min, c0max, c1min, c1max)
    boundaries: list                          # [{"name", "type": dirichlet|neumann|robin|symmetry|axis, "h"?}]
    description: str
    sigma_shared: sp.Expr = R_(0)             # reaction coefficient of the 'reaction' forcing variant (C2)
    params: dict = dfield(default_factory=dict)
    notes: str = ""
    kind: str = "scalar_mms"

    @property
    def coords(self):
        return coords_of(self.geometry)


@dataclass
class BousField:
    name: str
    geometry: str
    u: list
    p: sp.Expr
    mu: sp.Expr
    T: sp.Expr
    k: sp.Expr
    rho0: sp.Expr
    cp: sp.Expr
    beta_g: sp.Expr
    T_ref: sp.Expr
    box: tuple
    u_boundaries: list
    T_boundaries: list
    description: str
    exact: bool = True
    kind: str = "boussinesq_mms"
    notes: str = ""

    @property
    def coords(self):
        return coords_of(self.geometry)


# -------------------------------------------------------------------------------------------- scalar MMS
def _bc_sq_scalar(h_over_kappa):
    return [{"name": "x0", "type": "dirichlet"}, {"name": "y0", "type": "dirichlet"},
            {"name": "x1", "type": "neumann"}, {"name": "y1", "type": "robin", "h_over_kappa0": h_over_kappa}]


def heat_sq(kappa0, tag):
    """Planar scalar MMS on [0,1]^2.  b = curl psi_b, psi_b = y - x/2 + (1/5) sin(2x + 1/3) cos(3y/2 - 1/5):
    b_x in [0.7, 1.3], b_y in [0.1, 0.9]: inflow through x = 0 and y = 0, outflow through x = 1 and y = 1 (checked).
    kappa = kappa0 (1 + (2/5) sin(x + 2y + 1/3)) in [0.6, 1.4] kappa0; a = 3/2."""
    psi = y - x / 2 + R_(1, 5) * sp.sin(2 * x + R_(1, 3)) * sp.cos(R_(3, 2) * y - R_(1, 5))
    b = (sp.diff(psi, y), -sp.diff(psi, x))
    T = sp.cos(2 * x - y + R_(1, 5)) * (1 + x * y / 3) + R_(1, 3) * sp.exp((x + 2 * y) / 3)
    kappa = kappa0 * (1 + R_(2, 5) * sp.sin(x + 2 * y + R_(1, 3)))
    return ScalarField(
        name=f"heat_sq_{tag}", geometry="planar", T=T, b=b, kappa=kappa, a=R_(3, 2), box=(0, 1, 0, 1),
        boundaries=_bc_sq_scalar(2), sigma_shared=R_(1, 2), params={"kappa0": str(kappa0)},
        description=("planar scalar MMS: Dirichlet x=0, y=0 (inflow), diffusive-flux Neumann x=1 (outflow), Robin y=1 "
                     "(outflow, h = 2 kappa0); variable kappa; prescribed divergence-free b"))


def heat_axi(kappa0, tag):
    """Axisymmetric scalar MMS on (r, z) in [0,1]^2.  b from the Stokes stream function Psi = s Phi(s, z), s = r^2,
    Phi = 1/2 + (1/5)(1 - s/2) cos(3z/4 + 1/5): b_r = -Psi_z / r = -r Phi_z >= 0 (odd), b_z = Psi_r / r =
    2 Phi + 2 s Phi_s >= 1 (even); div b = 0 exactly.  Inflow z = 0; outflow z = 1 and r = 1 (checked).
    T even in r; kappa = kappa0 (1 + (2/5) sin(s + z + 1/3)) (even); a = 3/2."""
    s = r ** 2
    S = sp.Symbol("S", real=True)
    Phi = R_(1, 2) + R_(1, 5) * (1 - S / 2) * sp.cos(R_(3, 4) * z + R_(1, 5))
    Psi = S * Phi
    br = sp.expand(-(sp.diff(Psi, z)).subs(S, s) / r)
    bz = (2 * sp.diff(Psi, S)).subs(S, s)          # (1/r) d_r Psi = 2 dPsi/ds
    T = sp.cos(R_(3, 4) * s + z / 2 + R_(1, 5)) * (1 + s * z / 4) + R_(1, 4) * sp.exp(z / 2 - s / 3)
    kappa = kappa0 * (1 + R_(2, 5) * sp.sin(s + z + R_(1, 3)))
    return ScalarField(
        name=f"heat_axi_{tag}", geometry="axisymmetric", T=T, b=(sp.simplify(br), sp.simplify(bz)), kappa=kappa,
        a=R_(3, 2), box=(0, 1, 0, 1),
        boundaries=[{"name": "axis", "type": "axis"}, {"name": "z0", "type": "dirichlet"},
                    {"name": "r1", "type": "neumann"}, {"name": "z1", "type": "robin", "h_over_kappa0": 2}],
        sigma_shared=R_(1, 2), params={"kappa0": str(kappa0)},
        description=("axisymmetric scalar MMS: axis natural (symmetry), Dirichlet z=0 (inflow), diffusive-flux Neumann "
                     "r=1 (outflow), Robin z=1 (outflow, h = 2 kappa0; on a face where r varies, so the r weight of "
                     "the Robin term matters); variable kappa; r-weighted measure"))


def heat_sq_patch(kappa0, tag):
    """Patch test (SUPG consistency): T quadratic, b linear and divergence-free, kappa linear: the residual of the
    exact T vanishes POINTWISE, so a consistent method (Galerkin or residual-based SUPG, any tau) reproduces T
    exactly when its load and boundary integrals are exact (degree <= 4 planar)."""
    T = 1 + x / 2 - y / 3 + x ** 2 / 4 + x * y / 5 - y ** 2 / 6
    b = (1 + y / 2, R_(1, 2) + x / 3)
    kappa = kappa0 * (1 + x / 4 + y / 8)
    return ScalarField(
        name=f"heat_sq_patch_{tag}", geometry="planar", T=T, b=b, kappa=kappa, a=R_(3, 2), box=(0, 1, 0, 1),
        boundaries=_bc_sq_scalar(2), params={"kappa0": str(kappa0)}, kind="scalar_patch",
        description="planar SUPG patch test: T in P2, b in P1 (div b = 0), kappa in P1; BCs as heat_sq")


def heat_axi_patch(kappa0, tag):
    """Axisymmetric patch test: T = 1 + z/2 + r^2/3 - z^2/5 (in P2, d_r T = 0 on the axis), b = (r/4, 1 - z/2)
    (div b = (1/r) d_r(r^2/4) - 1/2 = 0), kappa = kappa0 (1 + z/4); the cylindrical residual (with kappa T_r / r)
    of T vanishes pointwise.  Degree <= 5 integrands (the weight r)."""
    T = 1 + z / 2 + r ** 2 / 3 - z ** 2 / 5
    b = (r / 4, 1 - z / 2)
    kappa = kappa0 * (1 + z / 4)
    return ScalarField(
        name=f"heat_axi_patch_{tag}", geometry="axisymmetric", T=T, b=b, kappa=kappa, a=R_(3, 2), box=(0, 1, 0, 1),
        boundaries=[{"name": "axis", "type": "axis"}, {"name": "z0", "type": "dirichlet"},
                    {"name": "z1", "type": "neumann"}, {"name": "r1", "type": "robin", "h_over_kappa0": 2}],
        params={"kappa0": str(kappa0)}, kind="scalar_patch",
        description="axisymmetric SUPG patch test: T in P2, b in P1 (axisymmetric div b = 0), kappa in P1")


def cond_annulus():
    """Variable-conductivity conduction in a hollow cylinder (the non-MMS cell of variable kappa): (r, z) in [1/2, 1] x
    [0, 1], kappa = 1 + r^2, b = 0, a = 1, EXACT solution of -div(kappa grad T) = 0: r kappa T' = 1 gives
    T = ln r - ln(1 + r^2)/2 + z/4 (kappa independent of z, T linear in z).  Dirichlet r = 1/2 and r = 1 (the exact
    values), exact diffusive flux on z = 0, 1 (-kappa/4 and +kappa/4 outward).  The variational outward flux through
    r = 1/2 is exactly int_0^1 (kappa T') r dz = 1 per radian."""
    T = sp.log(r) - sp.log(1 + r ** 2) / 2 + z / 4
    return ScalarField(
        name="cond_annulus", geometry="axisymmetric", T=T, b=(sp.Integer(0), sp.Integer(0)), kappa=1 + r ** 2,
        a=sp.Integer(1), box=(R_(1, 2), 1, 0, 1),
        boundaries=[{"name": "r0", "type": "dirichlet"}, {"name": "r1", "type": "dirichlet"},
                    {"name": "z0", "type": "neumann"}, {"name": "z1", "type": "neumann"}],
        params={}, kind="conduction",
        description="exact radial conduction with kappa = 1 + r^2 in a hollow cylinder (no advection, no source)")


def heat_layer(eps):
    """Outflow boundary layer (exact): b = (1, 0), a = 1, kappa = eps, T = S(x, y) - S(1, y) E(x),
    S = sin(3x/2)(1 + y/2), E = (exp((x - 1)/eps) - exp(-1/eps)) / (1 - exp(-1/eps)) (-eps E'' + E' = 0, E(0) = 0,
    E(1) = 1): s = -eps S_xx + S_x.  T = 0 on x = 0 and x = 1 (Dirichlet), exact diffusive flux on y = 0, 1.  0 <= T
    and sup T = S(1, 1) (approached at x -> 1-).  The layer of width ~eps at x = 1 is never resolved."""
    e = sp.Rational(eps) if not isinstance(eps, sp.Basic) else eps
    S = sp.sin(R_(3, 2) * x) * (1 + y / 2)
    E = (sp.exp((x - 1) / e) - sp.exp(-1 / e)) / (1 - sp.exp(-1 / e))
    T = S - S.subs(x, 1) * E
    return ScalarField(
        name="heat_layer_2d", geometry="planar", T=T, b=(sp.Integer(1), sp.Integer(0)), kappa=e, a=sp.Integer(1),
        box=(0, 1, 0, 1),
        boundaries=[{"name": "x0", "type": "dirichlet"}, {"name": "x1", "type": "dirichlet"},
                    {"name": "y0", "type": "neumann"}, {"name": "y1", "type": "neumann"}],
        params={"eps": str(e)}, kind="scalar_layer",
        description=("exact outflow-layer solution, b = (1, 0), kappa = eps; the layer at x = 1 is unresolved on every "
                     "fixed mesh; T >= 0, sup T = sin(3/2) * 3/2"))


# -------------------------------------------------------------------------------------------- Boussinesq MMS
def bous_sq():
    """Planar Boussinesq MMS on [0,1]^2: u = curl psi (the F2a ns_sq_dirichlet stream function, non-Beltrami), p and mu
    as there; T = (1/2) cos(3x/2 - 2y/3 + 1/5)(1 + xy/4) + (1/5) exp((x - y)/2); k = (1/50)(1 + (2/5) sin(x + 2y + 1/3));
    rho0 = 5/4, c_p = 6/5 (a = 3/2), beta g = 2, T_ref = 1/3, gravity -e_y.  Velocity Dirichlet on all edges; T:
    Dirichlet x=0, y=1; Robin x=1 (h = 1/10); diffusive-flux Neumann y=0."""
    psi = R_(1, 4) * (sp.exp(R_(3, 5) * x) * sp.sin(R_(9, 5) * y + R_(1, 2)) + R_(3, 10) * y ** 2 * sp.cos(R_(21, 10) * x - R_(2, 5)))
    u = [sp.diff(psi, y), -sp.diff(psi, x)]
    p = R_(1, 4) * sp.cos(R_(19, 10) * x + R_(1, 5)) * sp.sinh(R_(4, 5) * y + R_(1, 10)) + x * y / 10
    mu = R_(1, 60) * (R_(3, 2) + R_(1, 2) * sp.sin(2 * x - y + R_(3, 10)) + x * y / 4)
    T = R_(1, 2) * sp.cos(R_(3, 2) * x - R_(2, 3) * y + R_(1, 5)) * (1 + x * y / 4) + R_(1, 5) * sp.exp((x - y) / 2)
    k = R_(1, 50) * (1 + R_(2, 5) * sp.sin(x + 2 * y + R_(1, 3)))
    return BousField(
        name="bous_sq", geometry="planar", u=u, p=p, mu=mu, T=T, k=k, rho0=R_(5, 4), cp=R_(6, 5), beta_g=R_(2),
        T_ref=R_(1, 3), box=(0, 1, 0, 1),
        u_boundaries=[{"name": n, "type": "dirichlet"} for n in ("x0", "x1", "y0", "y1")],
        T_boundaries=[{"name": "x0", "type": "dirichlet"}, {"name": "y1", "type": "dirichlet"},
                      {"name": "x1", "type": "robin", "h": "1/10"}, {"name": "y0", "type": "neumann"}],
        description="planar Boussinesq MMS (coupled u, p, T; buoyancy O(1); variable mu and k)")


def bous_axi():
    """Axisymmetric Boussinesq MMS with swirl on (r, z) in [0,1]^2: u, p, mu of the F2a ns_axi_dirichlet field
    (Psi = s Phi, swirl u_theta = r/2 e^{-s/2} sin(3z/2 + 1/4)(1 + z/3)); T = (1/2) cos(3s/4 - 2z/3 + 1/5)(1 + sz/4)
    + (1/5) exp((z - s)/2) (even); k = (1/50)(1 + (2/5) sin(s + z + 1/3)); rho0 = 5/4, c_p = 6/5, beta g = 2,
    T_ref = 1/3, gravity -e_z.  Velocity Dirichlet on r=1, z=0, z=1 (+ the axis conditions); T: axis natural,
    Dirichlet z=0, Robin z=1 (h = 1/10; r varies along it), diffusive-flux Neumann r=1."""
    s = r ** 2
    S = sp.Symbol("S", real=True)
    Phi = R_(1, 4) * (sp.exp(R_(3, 5) * z) * sp.cos(R_(7, 10) * S + R_(1, 3)) * (1 + S / 5) + R_(2, 5) * S * z ** 2)
    Psi = S * Phi
    ur = sp.expand(-(sp.diff(Psi, z)).subs(S, s) / r)
    uz = (2 * sp.diff(Psi, S)).subs(S, s)
    ut = r * R_(1, 2) * sp.exp(-s / 2) * sp.sin(R_(3, 2) * z + R_(1, 4)) * (1 + z / 3)
    p = R_(1, 4) * sp.cos(R_(6, 5) * s + R_(2, 3) * z + R_(1, 5)) + s * z / 8
    mu = R_(1, 60) * (R_(3, 2) + R_(9, 10) * sp.sin(2 * s + z + R_(3, 10)) + s * z / 2)
    T = R_(1, 2) * sp.cos(R_(3, 4) * s - R_(2, 3) * z + R_(1, 5)) * (1 + s * z / 4) + R_(1, 5) * sp.exp((z - s) / 2)
    k = R_(1, 50) * (1 + R_(2, 5) * sp.sin(s + z + R_(1, 3)))
    return BousField(
        name="bous_axi", geometry="axisymmetric", u=[sp.simplify(ur), ut, sp.simplify(uz)], p=p, mu=mu, T=T, k=k,
        rho0=R_(5, 4), cp=R_(6, 5), beta_g=R_(2), T_ref=R_(1, 3), box=(0, 1, 0, 1),
        u_boundaries=[{"name": "axis", "type": "axis"}] + [{"name": n, "type": "dirichlet"} for n in ("r1", "z0", "z1")],
        T_boundaries=[{"name": "axis", "type": "axis"}, {"name": "z0", "type": "dirichlet"},
                      {"name": "z1", "type": "robin", "h": "1/10"}, {"name": "r1", "type": "neumann"}],
        description="axisymmetric Boussinesq MMS with swirl (coupled u_r, u_theta, u_z, p, T; gravity along -z)")


def hydro(geometry, T_bottom, T_top, tag, beta_g=R_(1000), mu=R_(1), k=R_(1), rho0=R_(1), cp=R_(1), T_ref=R_(1, 4)):
    """Hydrostatic no-flow: enclosed unit box, no slip, T = T_bottom on the bottom, T_top on the top, adiabatic sides
    (axisymmetric: axis natural, r = 1 adiabatic).  EXACT: u = 0, T linear in the vertical coordinate (in P2, so the
    conduction part is exact), p = rho0 beta g int (T - T_ref) d(vertical) (quadratic unless T_top = T_bottom).
    The buoyancy is a gradient: a pressure-robust method gives u_h = 0; Taylor-Hood gives a spurious u_h that
    converges with the pressure's best approximation."""
    v = y if geometry == "planar" else z
    T = T_bottom + (T_top - T_bottom) * v
    p = rho0 * beta_g * sp.integrate(T - T_ref, (v, 0, v))
    zero = sp.Integer(0)
    u = [zero, zero] if geometry == "planar" else [zero, zero, zero]
    bot, top, side = ("y0", "y1", ["x0", "x1"]) if geometry == "planar" else ("z0", "z1", ["r1"])
    ub = ([{"name": n, "type": "dirichlet"} for n in ("x0", "x1", "y0", "y1")] if geometry == "planar" else
          [{"name": "axis", "type": "axis"}] + [{"name": n, "type": "dirichlet"} for n in ("r1", "z0", "z1")])
    tb = [{"name": bot, "type": "dirichlet"}, {"name": top, "type": "dirichlet"}] + \
         [{"name": n, "type": "neumann_zero"} for n in side] + \
         ([{"name": "axis", "type": "axis"}] if geometry == "axisymmetric" else [])
    return BousField(
        name=f"hydro_{'sq' if geometry == 'planar' else 'axi'}_{tag}", geometry=geometry, u=u, p=p, mu=mu, T=T, k=k,
        rho0=rho0, cp=cp, beta_g=beta_g, T_ref=T_ref, box=(0, 1, 0, 1), u_boundaries=ub, T_boundaries=tb,
        kind="hydrostatic",
        description=(f"hydrostatic no-flow ({geometry}): T_bottom = {T_bottom}, T_top = {T_top}, T_ref = {T_ref}, "
                     f"beta g = {beta_g}, mu = {mu}; exact u = 0, p "
                     f"{'linear' if T_top == T_bottom else 'quadratic'} in the vertical coordinate"))


# ============================================================================================ Graetz modes (mpmath)
def _series_coeffs(lam, P, geometry, kmax=400):
    """Power series in w = y^2 (plates, u = (3/2)(1 - y^2)) or w = r^2 (pipe, u = 2(1 - r^2)) of the mode equation
    with axial conduction:  plates  phi'' + (lam^2 + P lam u(y)) phi = 0,  phi'(0) = 0
                            pipe    R'' + R'/r + (lam^2 + P lam u(r)) R = 0,  R'(0) = 0,
    normalized to value 1 on the axis/symmetry line.  With q = alpha - B w (alpha = lam^2 + c P lam, B = c P lam,
    c = 3/2 plates, 2 pipe):  plates c_{k+1} = (B c_{k-1} - alpha c_k) / ((2k+2)(2k+1));
                              pipe   c_{k+1} = (B c_{k-1} - alpha c_k) / (2k+2)^2."""
    c = mp.mpf(3) / 2 if geometry == "planar" else mp.mpf(2)
    alpha = lam ** 2 + c * P * lam
    B = c * P * lam
    cs = [mp.mpf(1)]
    prev = mp.mpf(0)
    for kk in range(kmax):
        den = (2 * kk + 2) * (2 * kk + 1) if geometry == "planar" else (2 * kk + 2) ** 2
        nxt = (B * prev - alpha * cs[-1]) / den
        prev = cs[-1]
        cs.append(nxt)
        if kk > 10 and abs(nxt) < mp.mpf(10) ** (-(mp.mp.dps + 10)) and abs(prev) < mp.mpf(10) ** (-(mp.mp.dps + 10)):
            break
    return cs


def mode_value(lam, P, geometry, w):
    cs = _series_coeffs(lam, P, geometry)
    return mp.fsum(ck * w ** k for k, ck in enumerate(cs))


def mode_deriv(lam, P, geometry, t):
    """d/dt of the mode at coordinate t (y or r)."""
    cs = _series_coeffs(lam, P, geometry)
    return mp.fsum(ck * 2 * k * t ** (2 * k - 1) for k, ck in enumerate(cs) if k > 0)


def graetz_eigen(P, geometry, nmodes, dps=50, bi=None):
    """The nmodes smallest POSITIVE lam with mode(1) = 0 (wall at t = 1, T_wall = 0) or, with a Biot number bi, the
    Robin wall condition mode'(1) + bi mode(1) = 0, by a sign scan and mpmath.findroot, at dps digits."""
    with mp.workdps(dps):
        P = mp.mpf(P)
        if bi is None:
            f = lambda lam: mode_value(lam, P, geometry, mp.mpf(1))  # noqa: E731
        else:
            B = mp.mpf(bi)
            f = lambda lam: mode_deriv(lam, P, geometry, mp.mpf(1)) + B * mode_value(lam, P, geometry, mp.mpf(1))  # noqa: E731
        roots, a, fa = [], mp.mpf("1e-6"), None
        fa = f(a)
        step = mp.mpf("0.01")
        while len(roots) < nmodes:
            bb = a + step
            fb = f(bb)
            if fa * fb < 0:
                roots.append(mp.findroot(f, (a, bb), solver="anderson"))
            a, fa = bb, fb
            if a > 200:
                raise CheckFailed("Graetz scan found too few roots")
        return [mp.mpf(v) for v in roots]


def chebyshev_fit(fn, deg, dps=50):
    """Chebyshev coefficients (first kind, on t in [0, 1] mapped to [-1, 1]) of fn at deg+1 Chebyshev-Gauss points,
    computed at dps digits; returned as floats (repr) plus the max coefficient of the tail as an accuracy proxy."""
    with mp.workdps(dps):
        n = deg + 1
        nodes = [mp.cos(mp.pi * (j + mp.mpf(1) / 2) / n) for j in range(n)]
        vals = [fn((tt + 1) / 2) for tt in nodes]
        coef = []
        for k in range(n):
            s = mp.fsum(vals[j] * mp.cos(mp.pi * k * (j + mp.mpf(1) / 2) / n) for j in range(n))
            coef.append(s * (1 if k == 0 else 2) / n)
        return [float(cv) for cv in coef], float(max(abs(cv) for cv in coef[-6:]))


@dataclass
class GraetzField:
    name: str
    geometry: str
    P: sp.Expr                 # Peclet number a U_m H / kappa (H = half width or radius)
    a: sp.Expr
    kappa: sp.Expr
    Um: sp.Expr
    amps: list
    nmodes: int
    box: tuple
    boundaries: list
    description: str
    kind: str = "graetz_modes"
    bi: object = None


def graetz_plates():
    """Half channel y in [0, 1] (symmetry y = 0, wall y = 1 with T = 0), x in [0, 2]; u = (3/2)(1 - y^2) (U_m = 1),
    a = 2, kappa = 1/8 (Pe = a U_m H / kappa = 16; a != 1 so that a dropped rho c_p is visible).  T = phi_1(y) e^{-lam_1 x} - (1/2) phi_2(y) e^{-lam_2 x}, an exact
    solution of the FULL equation (axial conduction included).  Inlet x = 0 Dirichlet, outlet x = 2 exact diffusive
    flux."""
    return GraetzField(
        name="graetz_plates", geometry="planar", P=R_(16), a=R_(2), kappa=R_(1, 8), Um=R_(1), amps=[R_(1), R_(-1, 2)],
        nmodes=2, box=(0, 2, 0, 1),
        boundaries=[{"name": "x0", "type": "dirichlet"}, {"name": "y1", "type": "dirichlet"},
                    {"name": "y0", "type": "symmetry"}, {"name": "x1", "type": "neumann"}],
        description="planar Graetz modes with axial conduction (half channel, Pe = 16)")


def graetz_pipe_robin():
    """As graetz_pipe, but a ROBIN wall at r = 1: -kappa dT/dr = h T (T_ext = 0), Bi = h R / kappa = 2: the modes satisfy
    R'(1) + 2 R(1) = 0 (the non-MMS Robin cell; the r = 1 wall has r ds = ds, its r weighting is covered by the MMS
    Robin face z = 1)."""
    return GraetzField(
        name="graetz_pipe_robin", geometry="axisymmetric", P=R_(16), a=R_(2), kappa=R_(1, 8), Um=R_(1),
        amps=[R_(1), R_(-1, 2)], nmodes=2, box=(0, 1, 0, 2),
        boundaries=[{"name": "z0", "type": "dirichlet"}, {"name": "r1", "type": "robin", "bi": "2"},
                    {"name": "axis", "type": "axis"}, {"name": "z1", "type": "neumann"}],
        description="axisymmetric Graetz modes with axial conduction and a Robin (convective) wall, Bi = 2, Pe = 16",
        bi=R_(2))


def graetz_pipe():
    """Pipe r in [0, 1] (axis, wall r = 1 with T = 0), z in [0, 2]; u_z = 2(1 - r^2) (U_m = 1), a = 1, kappa = 1/16
    (Pe = a U_m R / kappa = 16).  T = R_1(r) e^{-lam_1 z} - (1/2) R_2(r) e^{-lam_2 z} (exact, axial conduction
    included).  Inlet z = 0 Dirichlet, outlet z = 2 exact diffusive flux, axis natural."""
    return GraetzField(
        name="graetz_pipe", geometry="axisymmetric", P=R_(16), a=R_(2), kappa=R_(1, 8), Um=R_(1), amps=[R_(1), R_(-1, 2)],
        nmodes=2, box=(0, 1, 0, 2),
        boundaries=[{"name": "z0", "type": "dirichlet"}, {"name": "r1", "type": "dirichlet"},
                    {"name": "axis", "type": "axis"}, {"name": "z1", "type": "neumann"}],
        description="axisymmetric Graetz modes with axial conduction (pipe, Pe = 16)")


def flux_plates():
    """Fully developed uniform-wall-flux solution, half channel (exact with axial conduction): u = (3/2)(1 - y^2),
    a = 2, kappa = 1/8, T = A x + g(y), kappa g'' = a A u(y), g(0) = 0 (A = 1/2); wall y = 1: inward flux
    q_w = a A U_m H (energy balance); Nu = 4 H q_w / (kappa (T_w - T_b)) with D_h = 4H (closed form, generator)."""
    A = R_(1, 2)
    a, kappa = R_(2), R_(1, 8)
    u = R_(3, 2) * (1 - y ** 2)
    g = sp.integrate(sp.integrate(a * A * u / kappa, (y, 0, y)), (y, 0, y))
    T = A * x + g
    return ScalarField(
        name="flux_plates", geometry="planar", T=T, b=(u, sp.Integer(0)), kappa=kappa, a=a, box=(0, 2, 0, 1),
        boundaries=[{"name": "x0", "type": "dirichlet"}, {"name": "y0", "type": "symmetry"},
                    {"name": "y1", "type": "neumann"}, {"name": "x1", "type": "neumann"}],
        params={"A": str(A)}, kind="uniform_flux",
        description="planar uniform-wall-flux fully developed solution (exact; T quartic in y)")


def flux_pipe():
    """Fully developed uniform-wall-flux solution in a pipe: u_z = 2(1 - r^2), a = 2, kappa = 1/8, T = A z + g(r),
    (kappa/r)(r g')' = a A u(r), g(0) = 0, A = 1/2; Nu = 2 R q_w / (kappa (T_w - T_b)) (closed form, generator)."""
    A = R_(1, 2)
    a, kappa = R_(2), R_(1, 8)
    u = 2 * (1 - r ** 2)
    rg1 = sp.integrate(r * a * A * u / kappa, (r, 0, r))           # r g'
    g = sp.integrate(sp.simplify(rg1 / r), (r, 0, r))
    T = A * z + g
    return ScalarField(
        name="flux_pipe", geometry="axisymmetric", T=T, b=(sp.Integer(0), u), kappa=kappa, a=a, box=(0, 1, 0, 2),
        boundaries=[{"name": "z0", "type": "dirichlet"}, {"name": "axis", "type": "axis"},
                    {"name": "r1", "type": "neumann"}, {"name": "z1", "type": "neumann"}],
        params={"A": str(A)}, kind="uniform_flux",
        description="axisymmetric uniform-wall-flux fully developed solution (exact; T quartic in r)")


def nusselt_uniform_flux(fld: ScalarField):
    """Closed-form Nu of the uniform-flux fields: Nu = D q_w / (kappa (T_w - T_b)) with T_b the flow-weighted mean
    over the cross-section, T_w the wall value, q_w = -kappa dT/dn_out at the wall (inward heat flux positive here);
    D = 4H (plates, D_h) or 2R (pipe)."""
    if fld.geometry == "planar":
        u = fld.b[0]
        Tw = fld.T.subs(y, 1)
        Tb = sp.integrate(u * fld.T, (y, 0, 1)) / sp.integrate(u, (y, 0, 1))
        qw = fld.kappa * sp.diff(fld.T, y).subs(y, 1)
        D = 4
    else:
        u = fld.b[1]
        Tw = fld.T.subs(r, 1)
        Tb = sp.integrate(u * fld.T * r, (r, 0, 1)) / sp.integrate(u * r, (r, 0, 1))
        qw = fld.kappa * sp.diff(fld.T, r).subs(r, 1)
        D = 2
    return sp.nsimplify(sp.simplify(D * qw / (fld.kappa * (Tw - Tb))))


# ============================================================================================ spelling
class _Py(sp.printing.numpy.NumPyPrinter):
    pass


def spell(expr, coords):
    expr = sp.sympify(expr)
    py = _Py({"fully_qualified_modules": True}).doprint(expr).replace("numpy.", "np.")
    try:
        cc = sp.ccode(expr)
    except Exception:  # noqa: BLE001
        cc = None
    return {"sympy": str(expr), "c": cc, "numpy": py}


def lam(expr, coords):
    f = sp.lambdify(coords, expr, modules="numpy")

    def g(A, B):
        return np.broadcast_to(np.asarray(f(A, B), dtype=float), np.shape(A)).copy()
    return g


def outward_normal(name):
    return {"x0": (-1, 0), "x1": (1, 0), "y0": (0, -1), "y1": (0, 1),
            "r1": (1, 0), "r0": (-1, 0), "z0": (0, -1), "z1": (0, 1), "axis": (-1, 0)}[name]


def edge_coord(name, box):
    """(which coordinate is fixed, its value) for a box edge."""
    c0a, c0b, c1a, c1b = box
    return {"x0": (0, c0a), "x1": (0, c0b), "y0": (1, c1a), "y1": (1, c1b),
            "axis": (0, 0), "r0": (0, c0a), "r1": (0, c0b), "z0": (1, c1a), "z1": (1, c1b)}[name]


# ============================================================================================ emit
def emit_scalar(fld: ScalarField) -> dict:
    c = fld.coords
    g = grad_s(fld.T, fld.geometry)
    out = {"name": fld.name, "kind": fld.kind, "geometry": fld.geometry, "coords": [str(v) for v in c],
           "description": fld.description, "box": [str(v) for v in fld.box], "params": fld.params,
           "a": {"value": float(fld.a), "sympy": str(fld.a)},
           "T": spell(fld.T, c), "grad_T": [spell(gi, c) for gi in g],
           "b": [spell(bi, c) for bi in fld.b], "kappa": spell(fld.kappa, c),
           "grad_kappa": [spell(sp.diff(fld.kappa, v), c) for v in c],
           "s": {f: spell(scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, fld.geometry, f), c) for f in ADV_FORMS},
           "s_no_advection": spell(scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, fld.geometry, advection=False), c),
           "boundaries": []}
    if fld.sigma_shared != 0:
        out["reaction_variant"] = {
            "sigma": {"value": float(fld.sigma_shared), "sympy": str(fld.sigma_shared)},
            "s": {f: spell(scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, fld.sigma_shared, fld.geometry, f), c)
                  for f in ADV_FORMS},
            "note": ("NOT used by F4a (sigma = 0 there): the first-order reaction variant offered to chemistry C2 for the "
                     "shared transport operator")}
    for bd in fld.boundaries:
        e = dict(bd)
        if bd["type"] in ("neumann", "robin", "symmetry"):
            n = outward_normal(bd["name"])
            q = -fld.kappa * (g[0] * n[0] + g[1] * n[1])
            bn = fld.b[0] * n[0] + fld.b[1] * n[1]
            e["q_out"] = spell(q, c)
            e["j_out"] = spell(q + fld.a * bn * fld.T, c)
            e["normal"] = list(n)
            if bd["type"] == "robin":
                h = sp.sympify(bd["h_over_kappa0"]) * sp.sympify(fld.params["kappa0"])
                e["h"] = {"value": float(h), "sympy": str(h)}
                e["T_ext"] = spell(fld.T - q / h, c)
                e["T_ext_conservative"] = spell(fld.T - (q + fld.a * bn * fld.T) / h, c)
                e["note"] = ("convective form: -kappa grad T.n = h (T - T_ext); conservative form (total flux): "
                             "(a b T - kappa grad T).n = h (T - T_ext_conservative)")
        out["boundaries"].append(e)
    return out


def emit_bous(fld: BousField) -> dict:
    c = fld.coords
    geo = fld.geometry
    a = fld.rho0 * fld.cp
    nc = len(fld.u)
    out = {"name": fld.name, "kind": fld.kind, "geometry": geo, "coords": [str(v) for v in c],
           "description": fld.description, "box": [str(v) for v in fld.box], "exact_solution": fld.exact,
           "rho0": {"value": float(fld.rho0), "sympy": str(fld.rho0)}, "cp": {"value": float(fld.cp), "sympy": str(fld.cp)},
           "beta_g": {"value": float(fld.beta_g), "sympy": str(fld.beta_g)},
           "T_ref": {"value": float(fld.T_ref), "sympy": str(fld.T_ref)},
           "gravity": "-e_y" if geo == "planar" else "-e_z",
           "u": [spell(ui, c) for ui in fld.u],
           "grad_u": [[spell(sp.diff(ui, v), c) for v in c] for ui in fld.u],
           "p": spell(fld.p, c), "mu": spell(fld.mu, c), "grad_mu": [spell(sp.diff(fld.mu, v), c) for v in c],
           "T": spell(fld.T, c), "grad_T": [spell(sp.diff(fld.T, v), c) for v in c],
           "k": spell(fld.k, c), "grad_k": [spell(sp.diff(fld.k, v), c) for v in c],
           "forms": {}, "Q": spell(energy_forcing(fld.T, fld.u, fld.k, a, geo), c),
           "Q_no_advection": spell(energy_forcing(fld.T, fld.u, fld.k, a, geo, advection=False), c),
           "u_boundaries": fld.u_boundaries, "T_boundaries": []}
    for form in FORMS:
        f = momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, form, geo, fld.T, fld.beta_g, fld.T_ref)
        fnb = momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, form, geo, fld.T, fld.beta_g, fld.T_ref, buoyancy=False)
        out["forms"][form] = {"f": [spell(fi, c) for fi in f],
                              "f_no_buoyancy": [spell(fi, c) for fi in fnb]}
    g = grad_s(fld.T, geo)
    for bd in fld.T_boundaries:
        e = dict(bd)
        if bd["type"] in ("neumann", "robin"):
            n = outward_normal(bd["name"])
            q = -fld.k * (g[0] * n[0] + g[1] * n[1])
            e["q_out"] = spell(q, c)
            e["normal"] = list(n)
            if bd["type"] == "robin":
                h = sp.sympify(bd["h"])
                e["h"] = {"value": float(h), "sympy": str(h)}
                e["T_ext"] = spell(fld.T - q / h, c)
        out["T_boundaries"].append(e)
    if nc == 3:
        out["swirl"] = True
    return out


def emit_graetz(fld: GraetzField, dps=50, deg=48) -> dict:
    lams = graetz_eigen(fld.P, fld.geometry, fld.nmodes, dps=dps, bi=fld.bi)
    modes = []
    with mp.workdps(dps):
        for n, lm in enumerate(lams):
            Pm = mp.mpf(str(fld.P))
            f = lambda t, lm=lm: mode_value(lm, Pm, fld.geometry, t ** 2)  # noqa: E731
            coef, tail = chebyshev_fit(f, deg, dps)
            df1 = mode_deriv(lm, Pm, fld.geometry, mp.mpf(1))
            modes.append({"n": n + 1, "lambda": mp.nstr(lm, 40), "amplitude": str(fld.amps[n]),
                          "cheb": [repr(v) for v in coef], "cheb_tail_max": tail,
                          "dphi_wall": mp.nstr(df1, 40), "phi_wall": mp.nstr(mode_value(lm, Pm, fld.geometry, mp.mpf(1)), 40),
                          "phi_axis": "1"})
    first = "x" if fld.geometry == "planar" else "z"
    return {"name": fld.name, "kind": fld.kind, "geometry": fld.geometry,
            "coords": ["x", "y"] if fld.geometry == "planar" else ["r", "z"],
            "description": fld.description, "box": [str(v) for v in fld.box],
            "P": str(fld.P), "a": {"value": float(fld.a), "sympy": str(fld.a)},
            "kappa": {"value": float(fld.kappa), "sympy": str(fld.kappa)},
            "Um": str(fld.Um), "velocity_profile": ("(3/2) Um (1 - y^2) along x" if fld.geometry == "planar"
                                                    else "2 Um (1 - r^2) along z"),
            "T_definition": (f"T = sum_n A_n phi_n(t) exp(-lambda_n {first}), t = {'y' if first == 'x' else 'r'}; "
                             "phi_n(t) = chebval(2 t - 1, cheb_n) (numpy.polynomial.chebyshev), phi_n(0) = 1, "
                             + ("phi_n(1) = 0" if fld.bi is None else f"phi_n'(1) + Bi phi_n(1) = 0 with Bi = {fld.bi}")
                             + "; lambda_n to 40 digits (mpmath, power series in t^2 + findroot)"),
            "modes": modes, "boundaries": _graetz_bcs(fld)}


def _graetz_bcs(fld):
    out = []
    for bd in fld.boundaries:
        e = dict(bd)
        if bd["type"] == "robin":
            h = sp.sympify(bd["bi"]) * fld.kappa
            e["h"] = {"value": float(h), "sympy": str(h)}
            e["T_ext"] = {"sympy": "0", "c": "0", "numpy": "0*" + ("r" if fld.geometry == "axisymmetric" else "x")}
            e["note"] = "-kappa dT/dn = h T on the wall (T_ext = 0), h = Bi kappa / R"
        out.append(e)
    return out


# ============================================================================================ self-checks
def _mp_zero_at(expr, syms, pts, tol, dps=60):
    with mp.workdps(dps):
        f = sp.lambdify(syms, expr, modules="mpmath")
        worst = mp.mpf(0)
        for P in pts:
            v = f(*[mp.mpf(str(p)) for p in P])
            worst = max(worst, abs(v))
        return float(worst)


def rand_pts(box, k=8, seed=1, margin=0.05):
    rng = random.Random(seed)
    c0a, c0b, c1a, c1b = (float(v) for v in box)
    return [(c0a + margin + (c0b - c0a - 2 * margin) * rng.random(), c1a + margin + (c1b - c1a - 2 * margin) * rng.random())
            for _ in range(k)]


def cartesian_scalar_forcing(T, b, kappa, a, sigma):
    """Route B for the axisymmetric scalar: embed in 3D Cartesian coordinates (X, Y, Z), T(r, z) -> T(sqrt(X^2+Y^2),
    Z), B = b_r e_r + b_z e_z; s = a B.grad T - div(kappa grad T) + sigma T with Cartesian calculus."""
    X, Y, Z = sp.symbols("X3 Y3 Z3", real=True)
    rr = sp.sqrt(X ** 2 + Y ** 2)
    sub = {r: rr, z: Z}
    T3, k3 = T.subs(sub), kappa.subs(sub)
    br, bz = (sp.sympify(c).subs(sub) for c in b)
    B = [br * X / rr, br * Y / rr, bz]
    V = (X, Y, Z)
    g = [sp.diff(T3, v) for v in V]
    s = a * sum(B[i] * g[i] for i in range(3)) - sum(sp.diff(k3 * g[i], V[i]) for i in range(3)) + sigma * T3
    return s, (X, Y, Z)


def fd4(fn, P, axis, h):
    e = np.zeros(len(P))
    e[axis] = h
    return (-fn(P + 2 * e) + 8 * fn(P + e) - 8 * fn(P - e) + fn(P - 2 * e)) / (12 * h)


def fd_scalar_residual(fld: ScalarField, s_expr, h=1e-3):
    """4th-order FD residual of the scalar PDE from T, b, kappa, a only, divided by max|s|: planar in (x, y);
    axisymmetric in 3D Cartesian coordinates at (r cos phi, r sin phi, z), random phi."""
    pts = rand_pts(fld.box, 12, seed=7)
    rng = np.random.default_rng(3)
    s_f = lam(s_expr, fld.coords)
    worst, smax = 0.0, 0.0
    if fld.geometry == "planar":
        Tf, kf = lam(fld.T, fld.coords), lam(fld.kappa, fld.coords)
        bf = [lam(sp.sympify(bi), fld.coords) for bi in fld.b]
        T_ = lambda Q: float(Tf(np.array(Q[0]), np.array(Q[1])))  # noqa: E731
        k_ = lambda Q: float(kf(np.array(Q[0]), np.array(Q[1])))  # noqa: E731
        for P in pts:
            P = np.array(P, float)
            gT = [fd4(T_, P, i, h) for i in range(2)]
            flux = lambda Q, i: k_(Q) * fd4(T_, Q, i, h)  # noqa: E731
            divf = sum(fd4(lambda Q, i=i: flux(Q, i), P, i, h) for i in range(2))
            res = float(fld.a) * sum(float(bf[i](np.array(P[0]), np.array(P[1]))) * gT[i] for i in range(2)) - divf
            sv = float(s_f(np.array(P[0]), np.array(P[1])))
            worst = max(worst, abs(res - sv))
            smax = max(smax, abs(sv), abs(divf), abs(res + divf), abs(k_(P)) * float(np.linalg.norm(gT)))
    else:
        Tf, kf = lam(fld.T, fld.coords), lam(fld.kappa, fld.coords)
        bf = [lam(sp.sympify(bi), fld.coords) for bi in fld.b]

        def cyl(Q):
            return np.array(math.hypot(Q[0], Q[1])), np.array(Q[2])
        T_ = lambda Q: float(Tf(*cyl(Q)))  # noqa: E731
        k_ = lambda Q: float(kf(*cyl(Q)))  # noqa: E731
        for P2 in pts:
            ph = rng.uniform(0, 2 * math.pi)
            P = np.array([P2[0] * math.cos(ph), P2[0] * math.sin(ph), P2[1]])
            rr, zz = cyl(P)
            br, bz = float(bf[0](rr, zz)), float(bf[1](rr, zz))
            B = np.array([br * P[0] / float(rr), br * P[1] / float(rr), bz])
            gT = [fd4(T_, P, i, h) for i in range(3)]
            divf = sum(fd4(lambda Q, i=i: k_(Q) * fd4(T_, Q, i, h), P, i, h) for i in range(3))
            res = float(fld.a) * float(np.dot(B, gT)) - divf
            sv = float(s_f(rr, zz))
            worst = max(worst, abs(res - sv))
            smax = max(smax, abs(sv), abs(divf), abs(res + divf), abs(k_(P)) * float(np.linalg.norm(gT)))
    return worst / max(smax, 1e-300)


def selfcheck_scalar(fld: ScalarField, em: dict, log=print) -> dict:
    rep = {}

    def need(cond, msg):
        if not cond:
            raise CheckFailed(msg)
        log(f"  ok  {msg}")

    c = fld.coords
    geo = fld.geometry
    # 1. div b = 0
    need(sp.simplify(div_v(sp.Matrix([sp.sympify(v) for v in fld.b]), geo)) == 0, "div b = 0 (symbolic)")
    # 2. two routes
    for f in (["convective", "conservative"]):
        sA = scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo, f)
        sB = scalar_forcing_b(fld.T, fld.b, fld.kappa, fld.a, 0, geo)
        d = _mp_zero_at(sA - sB, c, rand_pts(fld.box), 1e-40)
        need(d < 1e-40, f"forcing ({f}) by route A (vector calculus) = route B (product rule): {d:.1e}")
        rep[f"route_AB_{f}"] = d
    if geo == "axisymmetric":
        sC, V3 = cartesian_scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0)
        sA = scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo)
        rng = random.Random(5)
        worst = 0.0
        with mp.workdps(60):
            fC = sp.lambdify(V3, sC, modules="mpmath")
            fA = sp.lambdify(c, sA, modules="mpmath")
            for P in rand_pts(fld.box):
                ph = mp.mpf(rng.random()) * 2 * mp.pi
                rr, zz = mp.mpf(str(P[0])), mp.mpf(str(P[1]))
                worst = max(worst, float(abs(fC(rr * mp.cos(ph), rr * mp.sin(ph), zz) - fA(rr, zz))))
        need(worst < 1e-40, f"axisymmetric forcing = 3D Cartesian embedding: {worst:.1e}")
        rep["cartesian_route"] = worst
        # parity and axis regularity (fields whose domain contains the axis)
        for nm, e, par in ([] if fld.kind == "conduction" else (("T", fld.T, 1), ("kappa", fld.kappa, 1), ("b_r", sp.sympify(fld.b[0]), -1),
                           ("b_z", sp.sympify(fld.b[1]), 1))):
            need(sp.simplify(e.subs(r, -r) - par * e) == 0, f"parity of {nm} ({'even' if par == 1 else 'odd'} in r)")
        if fld.kind != "conduction":
            need(sp.simplify(sp.diff(fld.T, r).subs(r, 0)) == 0, "d_r T = 0 on the axis")
    # 3. spellings parse back and evaluate
    for key in ("convective", "conservative"):
        e2 = sp.sympify(em["s"][key]["sympy"], locals={str(v): v for v in c})
        d = _mp_zero_at(e2 - scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo, key), c, rand_pts(fld.box), 1e-40)
        need(d < 1e-40, f"emitted sympy forcing ({key}) parses back: {d:.1e}")
        P = np.array(rand_pts(fld.box, 6, 11))
        ns = {"np": np, str(c[0]): P[:, 0], str(c[1]): P[:, 1]}
        vp = np.asarray(eval(em["s"][key]["numpy"], {"__builtins__": {}}, ns), dtype=float)
        vs = lam(e2, c)(P[:, 0], P[:, 1])
        need(np.max(np.abs(vp - vs)) <= 1e-11 * max(1.0, np.max(np.abs(vs))), f"numpy spelling of s ({key}) evaluates")
    # 4. FD residual with teeth
    s_ok = scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo)
    k0 = float(sp.sympify(fld.params.get("kappa0", fld.params.get("eps", "1"))))
    fd = fd_scalar_residual(fld, s_ok)
    tol = 1e-8
    need(fd < tol, f"FD residual (4th order, from T, b, kappa, a only) rel {fd:.1e}")
    rep["fd_residual_rel"] = fd
    if fld.kind == "conduction":
        badk = fd_scalar_residual(fld, scalar_forcing(fld.T, fld.b, sp.Integer(1), fld.a, 0, geo))
        need(badk > 1e3 * max(fd, 1e-13), f"FD teeth: kappa = 1 instead of kappa(r): rel {badk:.2e}")
        rep["fd_teeth_kappa_const"] = badk
        need(sp.simplify(scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo)) == 0, "exact: s = 0 (no source)")
    elif fld.kind not in ("scalar_patch",) and k0 > 1e-3:
        bad = fd_scalar_residual(fld, scalar_forcing(fld.T, fld.b, fld.kappa, fld.a, 0, geo, advection=False))
        need(bad > 1e3 * max(fd, 1e-13), f"FD teeth: forcing without advection rel {bad:.2e}")
        rep["fd_teeth_no_advection"] = bad
        badk = fd_scalar_residual(fld, scalar_forcing(fld.T, fld.b, fld.kappa * 2, fld.a, 0, geo))
        need(badk > 1e3 * max(fd, 1e-13), f"FD teeth: kappa doubled rel {badk:.2e}")
        rep["fd_teeth_kappa"] = badk
    # 5. inflow / outflow signs on the boundaries
    for bd in fld.boundaries:
        if bd["type"] in ("axis", "symmetry"):
            continue
        ax, val = edge_coord(bd["name"], fld.box)
        n = outward_normal(bd["name"])
        t = np.linspace(float(fld.box[2 - 2 * ax]) + 1e-9, float(fld.box[3 - 2 * ax]) - 1e-9, 201)
        Pc = [np.full_like(t, float(val)), t] if ax == 0 else [t, np.full_like(t, float(val))]
        bn = sum(lam(sp.sympify(fld.b[i]), c)(Pc[0], Pc[1]) * n[i] for i in range(2))
        rep[f"bn_{bd['name']}"] = [float(bn.min()), float(bn.max())]
        if bd["type"] in ("neumann", "robin") and fld.kind in ("scalar_mms", "scalar_patch"):
            need(bn.min() >= 0, f"{bd['name']} ({bd['type']}) is an outflow boundary: b.n in [{bn.min():.3f}, {bn.max():.3f}]")
    # 6. Robin data consistency (FD normal derivative)
    for bd in em["boundaries"]:
        if bd["type"] == "robin":
            ax, val = edge_coord(bd["name"], fld.box)
            tt = np.linspace(0.1, 0.9, 5)
            hval = bd["h"]["value"]
            Text = lam(sp.sympify(bd["T_ext"]["sympy"], locals={str(v): v for v in c}), c)
            Tf, kf = lam(fld.T, c), lam(fld.kappa, c)
            n = outward_normal(bd["name"])
            worst = 0.0
            for t0 in tt:
                P = np.array([float(val), t0]) if ax == 0 else np.array([t0, float(val)])
                g = [fd4(lambda Q: float(Tf(np.array(Q[0]), np.array(Q[1]))), P, i, 1e-4) for i in range(2)]
                q = -float(kf(np.array(P[0]), np.array(P[1]))) * (g[0] * n[0] + g[1] * n[1])
                rhs = hval * (float(Tf(np.array(P[0]), np.array(P[1]))) - float(Text(np.array(P[0]), np.array(P[1]))))
                worst = max(worst, abs(q - rhs) / max(abs(q), 1e-300))
            need(worst < 1e-7, f"Robin data on {bd['name']}: -kappa dT/dn = h (T - T_ext) (FD) rel {worst:.1e}")
    return rep


def fd_bous_residual(fld: BousField, f_exprs, Q_expr, form, h=1e-3):
    """4th-order FD residual of momentum (rel to max|f|) and energy (rel to max|Q|) from u, p, mu, T, k and the
    constants only; axisymmetric in 3D Cartesian coordinates (random azimuth)."""
    c = fld.coords
    geo = fld.geometry
    uf = [lam(ui, c) for ui in fld.u]
    pf, muf, Tf, kf = lam(fld.p, c), lam(fld.mu, c), lam(fld.T, c), lam(fld.k, c)
    ff = [lam(fi, c) for fi in f_exprs]
    Qf = lam(Q_expr, c)
    rho0, a, bg, Tr = float(fld.rho0), float(fld.rho0 * fld.cp), float(fld.beta_g), float(fld.T_ref)
    pts = rand_pts(fld.box, 10, seed=13, margin=0.08)
    rng = np.random.default_rng(17)
    wm, mm, we, me = 0.0, 0.0, 0.0, 0.0
    if geo == "planar":
        def U(Q):
            return np.array([float(g(np.array(Q[0]), np.array(Q[1]))) for g in uf])

        def sc(fn):
            return lambda Q: float(fn(np.array(Q[0]), np.array(Q[1])))
        dim, up = 2, 1
        fvec = lambda Q: np.array([float(g(np.array(Q[0]), np.array(Q[1]))) for g in ff])  # noqa: E731
        Qs = sc(Qf)
        Pl = [np.array(P, float) for P in pts]
    else:
        def cyl(Q):
            return np.array(math.hypot(Q[0], Q[1])), np.array(Q[2])

        def U(Q):
            rr, zz = cyl(Q)
            ur, ut, uz = (float(g(rr, zz)) for g in uf)
            c_, s_ = Q[0] / float(rr), Q[1] / float(rr)
            return np.array([ur * c_ - ut * s_, ur * s_ + ut * c_, uz])

        def sc(fn):
            return lambda Q: float(fn(*cyl(Q)))
        dim, up = 3, 2

        def fvec(Q):
            rr, zz = cyl(Q)
            fr, ft, fz = (float(g(rr, zz)) for g in ff)
            c_, s_ = Q[0] / float(rr), Q[1] / float(rr)
            return np.array([fr * c_ - ft * s_, fr * s_ + ft * c_, fz])
        Qs = sc(Qf)
        Pl = []
        for P2 in pts:
            ph = rng.uniform(0, 2 * math.pi)
            Pl.append(np.array([P2[0] * math.cos(ph), P2[0] * math.sin(ph), P2[1]]))
    p_, mu_, T_, k_ = sc(pf), sc(muf), sc(Tf), sc(kf)

    def G(Q, i, j):
        return fd4(lambda S: U(S)[i], Q, j, h)

    def sigma(Q, i, j):
        gij = G(Q, i, j)
        v = mu_(Q) * (gij + G(Q, j, i)) if form == "sym" else mu_(Q) * gij
        return v - (p_(Q) if i == j else 0.0)
    for P in Pl:
        u0 = U(P)
        res = np.zeros(dim)
        for i in range(dim):
            conv = rho0 * sum(u0[j] * G(P, i, j) for j in range(dim))
            divs = sum(fd4(lambda S, i=i, j=j: sigma(S, i, j), P, j, h) for j in range(dim))
            res[i] = conv - divs
        res[up] -= rho0 * bg * (T_(P) - Tr)
        fv = fvec(P)
        wm = max(wm, float(np.max(np.abs(res - fv))))
        mm = max(mm, float(np.max(np.abs(fv))), abs(rho0 * bg * (T_(P) - Tr)), abs(fd4(p_, P, up, h)))
        gT = [fd4(T_, P, j, h) for j in range(dim)]
        divq = sum(fd4(lambda S, j=j: k_(S) * fd4(T_, S, j, h), P, j, h) for j in range(dim))
        er = a * float(np.dot(u0, gT)) - divq
        qv = Qs(P)
        we = max(we, abs(er - qv))
        me = max(me, abs(qv), abs(divq), abs(er + divq), k_(P) * float(np.linalg.norm(gT)))
    return wm / max(mm, 1e-300), we / max(me, 1e-300)


def selfcheck_bous(fld: BousField, em: dict, log=print) -> dict:
    rep = {}

    def need(cond, msg):
        if not cond:
            raise CheckFailed(msg)
        log(f"  ok  {msg}")
    c = fld.coords
    geo = fld.geometry
    a = fld.rho0 * fld.cp
    if geo == "planar":
        dv = sp.diff(fld.u[0], x) + sp.diff(fld.u[1], y)
    else:
        dv = sp.diff(r * fld.u[0], r) / r + sp.diff(fld.u[2], z)
    need(sp.simplify(dv) == 0, "div u = 0 (symbolic)")
    for form in FORMS:
        f = momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, form, geo, fld.T, fld.beta_g, fld.T_ref)
        Q = energy_forcing(fld.T, fld.u, fld.k, a, geo)
        fm, fe = fd_bous_residual(fld, f, Q, form)
        tol = 1e-8
        need(fm < tol and fe < tol, f"{form}: FD residual momentum rel {fm:.1e}, energy rel {fe:.1e}")
        rep[f"fd_{form}"] = [fm, fe]
        if fld.kind == "boussinesq_mms":
            fnb = momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, form, geo, fld.T, fld.beta_g, fld.T_ref, buoyancy=False)
            bm, _ = fd_bous_residual(fld, fnb, Q, form)
            need(bm > 1e3 * max(fm, 1e-13), f"{form}: FD teeth, forcing without buoyancy rel {bm:.2e}")
            fflip = momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, form, geo, fld.T, fld.beta_g, fld.T_ref,
                                     buoyancy_sign=-1)
            sm, _ = fd_bous_residual(fld, fflip, Q, form)
            need(sm > 1e3 * max(fm, 1e-13), f"{form}: FD teeth, buoyancy sign flipped rel {sm:.2e}")
            Qna = energy_forcing(fld.T, fld.u, fld.k, a, geo, advection=False)
            _, en = fd_bous_residual(fld, f, Qna, form)
            need(en > 1e3 * max(fe, 1e-13), f"{form}: FD teeth, energy forcing without advection rel {en:.2e}")
            rep[f"teeth_{form}"] = [bm, sm, en]
        # emitted spelling parses back
        for i, fi in enumerate(f):
            e2 = sp.sympify(em["forms"][form]["f"][i]["sympy"], locals={str(v): v for v in c})
            d = _mp_zero_at(e2 - fi, c, rand_pts(fld.box, 4), 1e-40)
            need(d < 1e-35, f"{form}: emitted f_{i} parses back ({d:.1e})")
    if geo == "axisymmetric":
        for nm, e, par in (("u_r", fld.u[0], -1), ("u_theta", fld.u[1], -1), ("u_z", fld.u[2], 1), ("p", fld.p, 1),
                           ("mu", fld.mu, 1), ("T", fld.T, 1), ("k", fld.k, 1)):
            need(sp.simplify(sp.sympify(e).subs(r, -r) - par * sp.sympify(e)) == 0, f"parity of {nm}")
    if fld.kind == "hydrostatic":
        need(all(sp.sympify(ui) == 0 for ui in fld.u), "exact u = 0")
        bf = fld.rho0 * fld.beta_g * (fld.T - fld.T_ref)
        v = y if geo == "planar" else z
        need(sp.simplify(sp.diff(fld.p, v) - bf) == 0, "dp/d(vertical) = rho0 beta g (T - T_ref): hydrostatic balance")
        need(all(sp.simplify(e) == 0 for e in momentum_forcing(fld.u, fld.p, fld.mu, fld.rho0, "lap", geo, fld.T,
                                                                fld.beta_g, fld.T_ref)), "f_u = 0 (no momentum forcing)")
        need(sp.simplify(energy_forcing(fld.T, fld.u, fld.k, a, geo)) == 0, "Q = 0 (no heat source)")
    return rep


def selfcheck_graetz(fld: GraetzField, em: dict, log=print) -> dict:
    """Each mode: the ODE residual at random points (series, 50 digits) is zero to 1e-35; phi(1) = 0 to 1e-35;
    the Chebyshev representation matches the series to 1e-14 at 200 points (value and derivative); the FULL PDE
    residual of the assembled T (4th-order FD, from the Chebyshev data only, as the check module evaluates it) is
    below 1e-8 relative; a teeth check (lambda perturbed by 1e-3) gives a residual > 1e-5."""
    rep = {}

    def need(cond, msg):
        if not cond:
            raise CheckFailed(msg)
        log(f"  ok  {msg}")
    P = mp.mpf(str(fld.P))
    geo = fld.geometry
    cfac = mp.mpf(3) / 2 if geo == "planar" else mp.mpf(2)
    for md in em["modes"]:
        with mp.workdps(50):
            lm = mp.mpf(md["lambda"])
            if fld.bi is None:
                need(abs(mode_value(lm, P, geo, mp.mpf(1))) < mp.mpf("1e-35"), f"mode {md['n']}: phi(1) = 0")
            else:
                wv = mode_deriv(lm, P, geo, mp.mpf(1)) + mp.mpf(str(fld.bi)) * mode_value(lm, P, geo, mp.mpf(1))
                need(abs(wv) < mp.mpf("1e-35"), f"mode {md['n']}: phi'(1) + Bi phi(1) = 0")
            worst = mp.mpf(0)
            for t in (mp.mpf("0.13"), mp.mpf("0.5"), mp.mpf("0.87")):
                f = lambda tt: mode_value(lm, P, geo, tt ** 2)  # noqa: E731
                d2 = mp.diff(f, t, 2)
                d1 = mp.diff(f, t, 1)
                u = cfac * (1 - t ** 2)
                res = d2 + (d1 / t if geo == "axisymmetric" else 0) + (lm ** 2 + P * lm * u) * f(t)
                worst = max(worst, abs(res))
            need(worst < mp.mpf("1e-30"), f"mode {md['n']}: ODE residual {mp.nstr(worst, 3)} (lambda = {md['lambda'][:14]})")
            cheb = np.array([float(v) for v in md["cheb"]])
            tt = np.linspace(0, 1, 201)
            ser = np.array([float(mode_value(lm, P, geo, mp.mpf(float(t)) ** 2)) for t in tt])
            che = np.polynomial.chebyshev.chebval(2 * tt - 1, cheb)
            dser = np.array([float(mode_deriv(lm, P, geo, mp.mpf(float(t)))) for t in tt])
            dche = 2 * np.polynomial.chebyshev.chebval(2 * tt - 1, np.polynomial.chebyshev.chebder(cheb))
            ev, ed = float(np.max(np.abs(ser - che))), float(np.max(np.abs(dser - dche)))
            need(ev < 1e-14 and ed < 1e-12, f"mode {md['n']}: Chebyshev data = series (value {ev:.1e}, derivative {ed:.1e})")
            rep[f"mode{md['n']}"] = {"cheb_value_err": ev, "cheb_deriv_err": ed, "ode_residual": float(worst)}
    # independent route: double-precision shooting with scipy (DOP853), no series, at the emitted lambda
    from scipy.integrate import solve_ivp
    for md in em["modes"]:
        lmf = float(md["lambda"])
        Pf = float(fld.P)
        cf_ = 1.5 if geo == "planar" else 2.0

        def rhs(t, Y):
            q = lmf ** 2 + Pf * lmf * cf_ * (1 - t ** 2)
            return [Y[1], -q * Y[0] - (Y[1] / t if geo == "axisymmetric" else 0.0)]
        t0 = 1e-6
        q0 = lmf ** 2 + Pf * lmf * cf_
        y0 = [1 - q0 * t0 ** 2 / (2 if geo == "planar" else 4), -q0 * t0 / (1 if geo == "planar" else 2)]
        sol = solve_ivp(rhs, (t0, 1.0), y0, method="DOP853", rtol=1e-13, atol=1e-15)
        end = abs(sol.y[0, -1]) if fld.bi is None else abs(sol.y[1, -1] + float(fld.bi) * sol.y[0, -1])
        need(end < 1e-9, f"mode {md['n']}: independent scipy shooting gives the wall condition {end:.1e} at the emitted lambda")
        rep[f"mode{md['n']}"]["scipy_shooting_phi1"] = end
    # full PDE residual from the emitted data (as the check module will evaluate)
    lams = [float(m["lambda"]) for m in em["modes"]]
    amps = [float(sp.sympify(m["amplitude"])) for m in em["modes"]]
    chs = [np.array([float(v) for v in m["cheb"]]) for m in em["modes"]]
    kap, a = float(fld.kappa), float(fld.a)

    def Tfun(t, s, lams=lams):
        return sum(A * np.polynomial.chebyshev.chebval(2 * t - 1, ch) * math.exp(-lm * s) for A, ch, lm in zip(amps, chs, lams))

    def resid(lams_):
        worst, scale = 0.0, 0.0
        h = 1e-3
        for (t0, s0) in [(0.21, 0.4), (0.55, 1.1), (0.83, 1.7), (0.4, 0.2)]:
            if geo == "planar":
                Q = np.array([s0, t0])
                Tq = lambda Q: Tfun(Q[1], Q[0], lams_)  # noqa: E731
                u = 1.5 * (1 - t0 ** 2)
                adv = a * u * fd4(Tq, Q, 0, h)
                lap = sum(fd4(lambda S, i=i: fd4(Tq, S, i, h), Q, i, h) for i in range(2))
            else:
                ph = 0.7
                Q = np.array([t0 * math.cos(ph), t0 * math.sin(ph), s0])
                Tq = lambda Q: Tfun(math.hypot(Q[0], Q[1]), Q[2], lams_)  # noqa: E731
                u = 2.0 * (1 - t0 ** 2)
                adv = a * u * fd4(Tq, Q, 2, h)
                lap = sum(fd4(lambda S, i=i: fd4(Tq, S, i, h), Q, i, h) for i in range(3))
            res = adv - kap * lap
            worst = max(worst, abs(res))
            scale = max(scale, abs(adv))
        return worst / scale
    rr = resid(lams)
    need(rr < 1e-8, f"full PDE residual (FD, from the emitted data) rel {rr:.1e}")
    bad = resid([lams[0] * (1 + 1e-3)] + lams[1:])
    need(bad > 1e-5, f"teeth: lambda_1 * (1 + 1e-3) gives rel {bad:.1e}")
    rep["pde_residual_rel"] = rr
    rep["pde_teeth"] = bad
    return rep


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")


ALL_SCALAR = [lambda: heat_sq(R_(3, 2), "pe1"), lambda: heat_sq(R_(3, 2) * R_(1, 10 ** 6), "pe1e6"),
              lambda: heat_axi(R_(3, 2), "pe1"), lambda: heat_axi(R_(3, 2) * R_(1, 10 ** 6), "pe1e6"),
              lambda: heat_sq_patch(R_(3, 2) * R_(1, 10 ** 6), "pe1e6"), lambda: heat_axi_patch(R_(3, 2) * R_(1, 10 ** 6), "pe1e6"),
              lambda: heat_sq_patch(R_(3, 2) * R_(1, 100), "pe1e2"), lambda: heat_axi_patch(R_(3, 2) * R_(1, 100), "pe1e2"),
              lambda: heat_layer(R_(1, 10 ** 6)), flux_plates, flux_pipe, cond_annulus]
ALL_BOUS = [bous_sq, bous_axi,
            lambda: hydro("planar", R_(0), R_(1), "strat"), lambda: hydro("planar", R_(3, 4), R_(3, 4), "uniform"),
            lambda: hydro("axisymmetric", R_(0), R_(1), "strat"), lambda: hydro("axisymmetric", R_(3, 4), R_(3, 4), "uniform")]
ALL_GRAETZ = [graetz_plates, graetz_pipe, graetz_pipe_robin]
