"""The F1a verification fields.  One function per field; each returns a
stokes2d.Field.  All constants are exact rationals so that the sympy self-check is exact."""
from __future__ import annotations

import sympy as sp

from stokes2d import (Boundary, Field, x, y, velocity_from_streamfunction, gauss_rect, gauss_rotated_rect,
                      gauss_annulus, sample_rect, sample_rotated_rect, sample_annulus)

R = sp.Rational
t = sp.Symbol("t", real=True)
PI = sp.pi


# ------------------------------------------------------------------ shared ingredients of the unit-square MMS family
def psi_sq():
    """Non-polynomial, no symmetry in x <-> y or under reflection; |u| ~ 1-3."""
    return sp.exp(R(3, 5) * x) * sp.sin(R(9, 5) * y + R(1, 2)) + R(3, 10) * y**2 * sp.cos(R(21, 10) * x - R(2, 5))


def p_sq():
    return sp.cos(R(19, 10) * x + R(1, 5)) * sp.sinh(R(4, 5) * y + R(1, 10)) + R(2, 5) * x * y


def mu_sq():
    """Variable viscosity, 1.178 <= mu <= 2.232 on the unit square, not symmetric under x <-> y."""
    return R(3, 2) + sp.sin(2 * x - y + R(3, 10)) / 2 + x * y / 4


def edges_unit_square(kinds):
    """kinds: {"x0": bc, "x1": bc, "y0": bc, "y1": bc}; outward normals."""
    spec = {"x0": ((-1, 0), 0, t), "x1": ((1, 0), 1, t), "y0": ((0, -1), t, 0), "y1": ((0, 1), t, 1)}
    out = []
    for name in ("x0", "x1", "y0", "y1"):
        nrm, xe, ye = spec[name]
        out.append(Boundary(name, kinds[name], tuple(sp.Integer(c) for c in nrm),
                            {"t": t, "x": sp.sympify(xe), "y": sp.sympify(ye), "range": [0, 1]}))
    return out


SQ_DOMAIN = {"kind": "rect", "x": [0, 1], "y": [0, 1]}


def _sq(name, desc, kinds, pressure, u=None, p=None, mu=None, forms=("lap", "sym"), extra=None):
    return Field(name, desc, SQ_DOMAIN, u if u is not None else velocity_from_streamfunction(psi_sq()),
                 p if p is not None else p_sq(), mu if mu is not None else mu_sq(), forms,
                 edges_unit_square(kinds), pressure, gauss_rect(0, 1, 0, 1), sample_rect(0, 1, 0, 1),
                 extra or {})


# ------------------------------------------------------------------ fields
def sq_dirichlet():
    return _sq("sq_dirichlet", "Unit square, Dirichlet on all four edges, variable mu; pressure up to a constant "
               "(compare mean-free). Also used with grad-div gamma = 1 and for the P1/P1 negative control.",
               dict(x0="dirichlet", x1="dirichlet", y0="dirichlet", y1="dirichlet"), "mean_zero",
               extra={"forms_must_differ_interior": True})


def sq_dirichlet_constmu():
    """Constant mu (= 37/100, deliberately != 1) for the discrete identity  S(gamma=0) == L(gamma=mu)."""
    return _sq("sq_dirichlet_constmu", "Unit square, Dirichlet everywhere, CONSTANT mu = 0.37: interior forcing of the "
               "two forms is identical; used for the discrete identity sym(gamma=0) == lap(gamma=mu).",
               dict(x0="dirichlet", x1="dirichlet", y0="dirichlet", y1="dirichlet"), "mean_zero",
               mu=R(37, 100))


def sq_traction():
    return _sq("sq_traction", "Unit square, Dirichlet on x=0 and y=0, inhomogeneous traction sigma_form n = g_form "
               "on x=1 and y=1 (two different normals); variable mu; pressure determined (no nullspace).",
               dict(x0="dirichlet", x1="traction", y0="dirichlet", y1="traction"), "determined",
               extra={"forms_must_differ_interior": True, "traction_forms_must_differ": True})


def sq_traction_constmu():
    return _sq("sq_traction_constmu", "As sq_traction with constant mu = 1: the forms differ ONLY through the traction "
               "data (isolates the boundary part of the transpose term).",
               dict(x0="dirichlet", x1="traction", y0="dirichlet", y1="traction"), "determined", mu=sp.Integer(1),
               extra={"traction_forms_must_differ": True})


def psi_point():
    """Stream function vanishing on the whole boundary of the unit square: u.n = d psi / dt = 0 on every
    edge, so the P2 interpolant of the Dirichlet data is exactly flux-free and pin-vs-mean comparisons measure the
    pressure-constant treatment, not the O(h^4) Simpson flux defect of generic data.  Tangential velocity != 0."""
    # scaled by 1/25 against the pressure (as T4/T5): at x 1 the velocity-coupled O(h^3) part of the
    # pressure error dominates at N = 32-128 (calibrated fit 2.90 / 2.93)
    return (R(4, 25) * x * (1 - x) * y * (1 - y) * sp.exp(R(3, 5) * x) * sp.sin(R(9, 5) * y + R(1, 2))
            * (1 + x * y / 2))


def sq_point():
    x0 = (R(1, 2), R(1, 4))
    f = _sq("sq_point", "Unit square, Dirichlet everywhere with FLUX-FREE data (psi = 0 on the boundary), pressure "
            "fixed by a point constraint p_h(x0) = p(x0) at x0 = (1/2, 1/4) (an interior vertex of every level).",
            dict(x0="dirichlet", x1="dirichlet", y0="dirichlet", y1="dirichlet"), "point",
            u=velocity_from_streamfunction(psi_point()),
            extra={"forms_must_differ_interior": True, "flux_free": True})
    f.extra["x_point"] = [str(c) for c in x0]
    f.extra["p_at_point"] = sp.N(p_sq().subs({x: x0[0], y: x0[1]}), 17)
    return f


def _do_nothing(form):
    """Do-nothing (sigma_form n = 0) on x = 1 for ONE form; the other form's traction there is non-zero."""
    psi0, p0, mu = psi_sq(), p_sq(), mu_sq()
    if form == "lap":
        # tangential: d_x u_y = -psi_xx = 0 at x=1;   normal: mu d_x u_x - p = 0 at x=1
        a = sp.diff(psi0, x, 2).subs(x, 1)
        psi = psi0 - (x - 1) ** 2 / 2 * a
        u = velocity_from_streamfunction(psi)
        target = (mu * sp.diff(u[0], x)).subs(x, 1)
    else:
        # tangential: mu (d_y u_x + d_x u_y) = mu (psi_yy - psi_xx) = 0 at x=1;  normal: 2 mu d_x u_x - p = 0
        b = (sp.diff(psi0, y, 2) - sp.diff(psi0, x, 2)).subs(x, 1)
        psi = psi0 + (x - 1) ** 2 / 2 * b
        u = velocity_from_streamfunction(psi)
        target = (2 * mu * sp.diff(u[0], x)).subs(x, 1)
    p = p0 + (target - p0.subs(x, 1))
    return _sq(f"sq_do_nothing_{form}", f"Unit square, do-nothing (sigma_{form} n = 0) on x=1, Dirichlet elsewhere; the "
               f"field satisfies the {form}-form natural condition and violates the other one. Pressure determined.",
               dict(x0="dirichlet", x1="do_nothing", y0="dirichlet", y1="dirichlet"), "determined",
               u=u, p=p, mu=mu, forms=(form,), extra={"do_nothing_form": form})


def sq_do_nothing_lap():
    return _do_nothing("lap")


def sq_do_nothing_sym():
    return _do_nothing("sym")


THETA = PI / 6


def rot_slip():
    """Unit square rotated by 30 degrees: local (s, t) in [0,1]^2, x = s cos - t sin, y = s sin + t cos.
    t = 0: slip (u.n = 0, zero tangential traction); s = 1: symmetry (same conditions); s = 0, t = 1: Dirichlet."""
    c, s_ = sp.cos(THETA), sp.sin(THETA)
    S = x * c + y * s_
    T = -x * s_ + y * c
    sl, tl = sp.symbols("s_l t_l", real=True)
    # (1/5)(1-s)^2: keeps g_s(1,t) = 0 and g_t(s,0) = 0 but makes g_s(0,t) = -2/5, so the Dirichlet edge
    # s = 0 VIOLATES the slip conditions (a slip tag on the wrong attribute is then visible).
    g = (2 + sp.cos(PI * sl) / 2 + R(2, 5) * sp.cos(PI * tl) + R(3, 10) * sp.cos(PI * sl) * sp.cos(R(3, 2) * PI * tl)
         + R(1, 5) * (1 - sl) ** 2)
    # 1/25: with psi x 1 the velocity-coupled O(h^3) part of the pressure error dominates the O(h^2)
    # best-approximation part up to N ~ 340; scaled, the P1 pressure is asymptotic at N = 32-128 (calibrated).
    psi_loc = R(1, 25) * sp.sin(PI * tl) * sp.sin(PI * sl) * g
    # stream function is a scalar invariant under rotation: psi(x, y) = psi_loc(S(x,y), T(x,y))
    psi = psi_loc.subs({sl: S, tl: T})
    u = velocity_from_streamfunction(psi)
    p = sp.cos(R(13, 10) * x - R(7, 10) * y + R(1, 5)) + sp.sin(x + 2 * y) / 2
    mu = mu_sq()

    def edge(name, bc, nrm, sx, ty, rng=(0, 1)):
        xe = sx * c - ty * s_
        ye = sx * s_ + ty * c
        return Boundary(name, bc, nrm, {"t": t, "x": xe, "y": ye, "range": list(rng)})

    n_t0 = (s_, -c)          # outward normal of t = 0 is -e_t, e_t = (-sin, cos)
    n_t1 = (-s_, c)
    n_s0 = (-c, -s_)         # -e_s, e_s = (cos, sin)
    n_s1 = (c, s_)
    bnd = [edge("t0", "slip", n_t0, t, 0), edge("s1", "symmetry", n_s1, 1, t),
           edge("s0", "dirichlet", n_s0, 0, t), edge("t1", "dirichlet", n_t1, t, 1)]
    th = float(THETA)
    return Field("rot_slip", "Unit square rotated by 30 deg; slip on local t=0, symmetry on local s=1, Dirichlet on "
                 "s=0 and t=1; variable mu; the slip/symmetry edges are not axis-aligned (u.n = 0 must use the true "
                 "normal). Pressure up to a constant.",
                 {"kind": "rotated_rect", "theta": "pi/6", "s": [0, 1], "t": [0, 1],
                  "map": "x = s cos(theta) - t sin(theta), y = s sin(theta) + t cos(theta)"},
                 u, p, mu, ("lap", "sym"), bnd, "mean_zero", gauss_rotated_rect(th), sample_rotated_rect(th),
                 {"forms_must_differ_interior": True})


def _periodic(name, pressure, desc):
    Lp = 1
    # scaled by 1/25 against the pressure (calibrated: pressure asymptotic at N = 32-128)
    psi_per = R(3, 250) * sp.sin(2 * PI * x + R(2, 5)) * sp.sin(PI * y) * sp.exp(y / 2)
    psi_mean = (y ** 2 * (R(3, 2) - y) + sp.sin(2 * y) / 10) / 25
    u = velocity_from_streamfunction(psi_per + psi_mean)
    G = sp.Integer(2)
    p_per = sp.cos(2 * PI * x + R(3, 10)) * sp.sinh(R(7, 10) * y) + R(1, 5) * sp.cos(2 * PI * x) * y
    p = p_per - G * x
    mu = R(3, 2) + R(2, 5) * sp.sin(2 * PI * x + R(7, 10)) * sp.cos(y) + y / 5
    bnd = edges_unit_square(dict(x0="periodic_src", x1="periodic_dst", y0="dirichlet", y1="dirichlet"))
    Q = sp.simplify(sp.integrate(u[0].subs(x, 0), (y, 0, 1)))
    return Field(name, desc, {"kind": "rect", "x": [0, 1], "y": [0, 1], "periodic": "x, period 1"},
                 u, p, mu, ("lap", "sym"), bnd, pressure, gauss_rect(0, 1, 0, 1), sample_rect(0, 1, 0, 1),
                 {"periodic_x": Lp, "p_periodic": p_per, "G": G, "pressure_drop_per_period": G * Lp,
                  "flow_rate_Q": Q, "forms_must_differ_interior": True})


def per_dp():
    return _periodic("per_dp", "periodic_jump",
                     "Periodic in x (period 1), Dirichlet (moving, u_y = 0) walls y=0,1, p = p_per - G x with the "
                     "pressure drop G = 2 per period PRESCRIBED; variable periodic mu; p_per up to a constant.")


def per_q():
    return _periodic("per_q", "periodic_flow_rate",
                     "Same field as per_dp but the flow rate Q = int_0^1 u_x(0,y) dy is prescribed and the pressure "
                     "drop G is an unknown (Lagrange multiplier); G_h must converge to 2.")


def noflow():
    # phi vanishes to third order on the boundary, so the spurious velocity on the nested piecewise-uniform meshes
    # converges cleanly at 4 / 3 (L2 / H1; interior superconvergence with no O(h^2) boundary strip); with a phi that
    # does not vanish on the boundary (e^x sin(2y + 1/3) + x^2 y) the orders mix 3.5 / 2.5 with pre-asymptotic drift
    # (fit 3.36 lap)
    phi = (16 * x * (1 - x) * y * (1 - y)) ** 3 * sp.exp(x) * sp.sin(2 * y + R(1, 3))
    return Field("noflow", "No-flow (pressure-robustness) test: u = 0, p = phi = (16 x(1-x) y(1-y))^3 e^x sin(2y+1/3), "
                 "f = grad phi, mu = nu constant "
                 "(nu = 1 here; the case scales f and the expected velocity bound by 1/nu). Dirichlet u = 0.",
                 SQ_DOMAIN, sp.Matrix([0, 0]), phi, sp.Integer(1), ("lap", "sym"),
                 edges_unit_square(dict(x0="dirichlet", x1="dirichlet", y0="dirichlet", y1="dirichlet")),
                 "mean_zero", gauss_rect(0, 1, 0, 1), sample_rect(0, 1, 0, 1), {"phi": phi})


# ------------------------------------------------------------------ closed-form (non-MMS) solutions
def poiseuille(variant):
    """Plane Poiseuille in [0,2] x [0,1], mu = 1, G = 2: u = (G y (1-y) / (2 mu), 0), p = G (L - x)."""
    L, H, mu, G = 2, 1, sp.Integer(1), sp.Integer(2)
    u = sp.Matrix([G * y * (H - y) / (2 * mu), 0])
    p = G * (L - x)
    kinds = {"inflow_dirichlet": dict(x0="dirichlet", x1="traction", y0="dirichlet", y1="dirichlet"),
             "pressure_driven": dict(x0="traction", x1="traction", y0="dirichlet", y1="dirichlet")}[variant]
    spec = {"x0": ((-1, 0), 0, t, [0, 1]), "x1": ((1, 0), L, t, [0, 1]), "y0": ((0, -1), t, 0, [0, L]),
            "y1": ((0, 1), t, H, [0, L])}
    bnd = [Boundary(k, kinds[k], tuple(sp.Integer(c) for c in spec[k][0]),
                    {"t": t, "x": sp.sympify(spec[k][1]), "y": sp.sympify(spec[k][2]), "range": spec[k][3]})
           for k in ("x0", "x1", "y0", "y1")]
    return Field(f"poiseuille_{variant}", "Plane Poiseuille, exact in P2/P1. Outlet x=2 traction data per form: lap "
                 "(0,0) = do-nothing; sym (0, mu U'(y)) != 0 (the sym do-nothing condition is NOT satisfied by "
                 "Poiseuille: Heywood-Rannacher-Turek).",
                 {"kind": "rect", "x": [0, 2], "y": [0, 1]}, u, p, mu, ("lap", "sym"), bnd, "determined",
                 gauss_rect(0, 2, 0, 1), sample_rect(0, 2, 0, 1), {"traction_forms_must_differ": True})


def poiseuille_half_symmetry():
    L, mu, G = 2, sp.Integer(1), sp.Integer(2)
    u = sp.Matrix([G * y * (1 - y) / (2 * mu), 0])
    p = G * (L - x)
    bnd = [Boundary("x0", "dirichlet", (-1, 0), {"t": t, "x": sp.Integer(0), "y": t, "range": [0, R(1, 2)]}),
           Boundary("x1", "do_nothing", (1, 0), {"t": t, "x": sp.Integer(2), "y": t, "range": [0, R(1, 2)]}),
           Boundary("y0", "dirichlet", (0, -1), {"t": t, "x": t, "y": sp.Integer(0), "range": [0, 2]}),
           Boundary("y1", "symmetry", (0, 1), {"t": t, "x": t, "y": R(1, 2), "range": [0, 2]})]
    return Field("poiseuille_half_symmetry", "Lower half [0,2] x [0,1/2] of the Poiseuille channel with symmetry on "
                 "y = 1/2; outlet do-nothing (lap form only: the sym outlet needs the traction (0, mu U'))",
                 {"kind": "rect", "x": [0, 2], "y": [0, "1/2"]}, u, p, mu, ("lap",), bnd, "determined",
                 gauss_rect(0, 2, 0, 0.5), sample_rect(0, 2, 0, 0.5), {"do_nothing_form": "lap"})


def plug_rotated():
    """Uniform flow along a channel rotated by 30 deg, slip walls; exact in P1."""
    c, s_ = sp.cos(THETA), sp.sin(THETA)
    U0 = sp.Integer(1)
    u = sp.Matrix([U0 * c, U0 * s_])
    p = sp.Integer(0)

    def edge(name, bc, nrm, sx, ty, rng):
        return Boundary(name, bc, nrm, {"t": t, "x": sx * c - ty * s_, "y": sx * s_ + ty * c, "range": rng})
    bnd = [edge("t0", "slip", (s_, -c), t, 0, [0, 2]), edge("t1", "slip", (-s_, c), t, 1, [0, 2]),
           edge("s0", "dirichlet", (-c, -s_), 0, t, [0, 1]), edge("s2", "do_nothing", (c, s_), 2, t, [0, 1])]
    th = float(THETA)
    return Field("plug_rotated", "Plug flow u = U0 e_s in the channel 0<s<2, 0<t<1 rotated by 30 deg; slip on both "
                 "walls, Dirichlet inflow, do-nothing outflow (both forms: grad u = 0, p = 0).",
                 {"kind": "rotated_rect", "theta": "pi/6", "s": [0, 2], "t": [0, 1]}, u, p, sp.Integer(1),
                 ("lap", "sym"), bnd, "determined", gauss_rotated_rect(th, 2, 1), sample_rotated_rect(th, 2, 1),
                 {"do_nothing_form": "both", "cross_bc_exempt": "plug flow has sigma = 0 everywhere, so every "
                  "edge satisfies do-nothing; a do-nothing inlet would leave the problem singular (caught by the "
                  "solver gate), not silently exact"})


def couette(variant):
    """variant 'periodic': walls y=0 (rest), y=1 (U=1), periodic in x.
    variant 'outflow': Dirichlet inflow x=0, outlet x=1 traction (lap: do-nothing; sym: (0, mu U/H))."""
    mu = sp.Integer(1)
    u = sp.Matrix([y, 0])
    p = sp.Integer(0)
    if variant == "periodic":
        kinds = dict(x0="periodic_src", x1="periodic_dst", y0="dirichlet", y1="dirichlet")
        pressure, extra = "mean_zero", {"periodic_x": 1, "p_periodic": p, "G": sp.Integer(0)}
    else:
        kinds = dict(x0="dirichlet", x1="traction", y0="dirichlet", y1="dirichlet")
        pressure, extra = "determined", {"traction_forms_must_differ": True}
    return Field(f"couette_{variant}", "Plane Couette u = (y, 0), p = 0, mu = 1 on the unit square.", SQ_DOMAIN,
                 u, p, mu, ("lap", "sym"), edges_unit_square(kinds), pressure, gauss_rect(0, 1, 0, 1),
                 sample_rect(0, 1, 0, 1), extra)


def couette_two_layer():
    """Shear-driven two-layer Couette: bottom wall at rest, tangential traction tau on y=1, periodic in x;
    mu = 1 for y < 1/2, mu = 3 for y > 1/2 (interface on a mesh line); exact, piecewise linear."""
    tau, mu1, mu2 = sp.Integer(1), sp.Integer(1), sp.Integer(3)
    mu = sp.Piecewise((mu1, y < R(1, 2)), (mu2, True))
    ux = sp.Piecewise((tau * y / mu1, y < R(1, 2)), (tau / (2 * mu1) + tau * (y - R(1, 2)) / mu2, True))
    u = sp.Matrix([ux, 0])
    p = sp.Integer(0)
    return Field("couette_two_layer", "Traction-driven two-layer Couette: u_x(1) = 1/2 + 1/6 = 2/3; tests the "
                 "tangential-traction sign and a discontinuous mu evaluated per quadrature point.", SQ_DOMAIN,
                 u, p, mu, ("lap", "sym"),
                 edges_unit_square(dict(x0="periodic_src", x1="periodic_dst", y0="dirichlet", y1="traction")),
                 "determined", gauss_rect(0, 1, 0, 1), sample_rect(0, 1, 0, 1, n=40, margin=0.0123),
                 {"periodic_x": 1, "p_periodic": p, "G": sp.Integer(0), "u_top": R(2, 3)})


def annulus(form):
    """Annulus 1 < r < 2, inner cylinder rotating (Omega = 1, Dirichlet), outer wall free slip.
    sym: rigid rotation u = (-y, x) (zero stress);  lap: u_theta = A r + B / r with A = 1/5, B = 4/5."""
    r2 = x ** 2 + y ** 2
    if form == "sym":
        u = sp.Matrix([-y, x])
    else:
        A, B = R(1, 5), R(4, 5)
        u = sp.Matrix([-y, x]) * (A + B / r2)
    p = sp.Integer(0)
    bnd = [Boundary("r1", "dirichlet", (-x, -y), {"t": t, "x": sp.cos(t), "y": sp.sin(t), "range": [0, 2 * PI]}),
           Boundary("r2", "slip", (x / 2, y / 2), {"t": t, "x": 2 * sp.cos(t), "y": 2 * sp.sin(t),
                                                    "range": [0, 2 * PI]})]
    return Field(f"annulus_{form}", "Curved free-slip wall: annulus 1<r<2, inner cylinder rotating at Omega=1, outer "
                 "free slip. The two forms have DIFFERENT exact solutions (sym: rigid rotation; lap: A r + B/r). "
                 "Needs isoparametric P2 geometry (Babuska-paradox control).",
                 {"kind": "annulus", "r": [1, 2]}, u, p, sp.Integer(1), (form,), bnd, "mean_zero",
                 gauss_annulus(1, 2), sample_annulus(1, 2), {"slip_other_form_nonzero": True,
                 "cross_bc_exempt": "closed-form patch: rigid rotation satisfies slip on r = 1 too; the curved MMS "
                                    "annulus_mms_* carries the attribute test"})


def annulus_mms(form):
    """Curved free-slip MMS (fills the curved-slip MMS row).  Annulus 1 < r < 2, Dirichlet on r = 1,
    free slip on r = 2 for the form's own tangential condition.  psi = (r-2) q + (r-2)^2/2 c(theta), with c built
    from q at the projected point (2x/r, 2y/r):
        lap: psi_rr = 0 at r = 2              ->  c = -2 q_r
        sym: -psi_rr + psi_r / r = 0 at r = 2 ->  c = q/2 - 2 q_r
    so that u_r = psi_theta / r = 0 and the tangential traction of THIS form vanishes on r = 2, the other form's does
    not, and the field is non-axisymmetric and non-polynomial (not in the discrete space)."""
    r = sp.sqrt(x ** 2 + y ** 2)
    a, b = sp.symbols("a b", real=True)
    q = sp.exp(a / 3) * sp.sin(b + R(2, 5)) + a * b / 4 + R(1, 2)
    qr = (a * sp.diff(q, a) + b * sp.diff(q, b)) / 2          # radial derivative at radius 2: (a q_a + b q_b)/|(a,b)|
    proj = {a: 2 * x / r, b: 2 * y / r}
    c = (-2 * qr if form == "lap" else q / 2 - 2 * qr).subs(proj)
    # 1/25 against the pressure (as T4-T6): x 1 leaves the pressure pre-asymptotic at N_r <= 32 (calibrated pairwise
    # 3.48, 3.45, 2.99); x 1/25 gives 2.31 -> 2.06 (lap) and 2.42 -> 2.08 (sym) at N_r = 4..16
    psi = ((r - 2) * q.subs({a: x, b: y}) + (r - 2) ** 2 / 2 * c) / 25
    u = velocity_from_streamfunction(psi)
    p = sp.cos(x - y / 2 + R(1, 5)) + sp.sin(y) / 2
    mu = R(3, 2) + sp.sin(x - y / 2 + R(3, 10)) / 2 + x / 10
    bnd = [Boundary("r1", "dirichlet", (-x, -y), {"t": t, "x": sp.cos(t), "y": sp.sin(t), "range": [0, 2 * PI]}),
           Boundary("r2", "slip", (x / 2, y / 2), {"t": t, "x": 2 * sp.cos(t), "y": 2 * sp.sin(t),
                                                    "range": [0, 2 * PI]})]
    return Field(f"annulus_mms_{form}", f"Curved free-slip MMS for the {form} form: annulus 1<r<2, Dirichlet r=1, "
                 f"slip r=2 (u.n = 0 and t.sigma_{form} n = 0; the other form's tangential traction is non-zero). "
                 "Variable mu. Needs isoparametric P2 geometry.",
                 {"kind": "annulus", "r": [1, 2]}, u, p, mu, (form,), bnd, "mean_zero",
                 gauss_annulus(1, 2), sample_annulus(1, 2), {"slip_other_form_nonzero": True})


ALL = [sq_dirichlet, sq_dirichlet_constmu, sq_traction, sq_traction_constmu, sq_point, sq_do_nothing_lap,
       sq_do_nothing_sym, rot_slip, per_dp, per_q, noflow,
       lambda: poiseuille("inflow_dirichlet"), lambda: poiseuille("pressure_driven"), poiseuille_half_symmetry,
       plug_rotated, lambda: couette("periodic"), lambda: couette("outflow"), couette_two_layer,
       lambda: annulus("sym"), lambda: annulus("lap"), lambda: annulus_mms("lap"), lambda: annulus_mms("sym")]
