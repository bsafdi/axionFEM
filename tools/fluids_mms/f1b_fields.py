"""F1b verification fields (axisymmetric Stokes with swirl).

Every MMS field is built from parity-respecting generators:
    Stokes stream function psi = r^2 G(s, z), s = r^2:  u_r = -r d_z G,  u_z = 2 G + 2 s d_s G   (div u = 0)
    u_theta = r C(s, z),  p = P(s, z),  mu = M(s, z)
so u_r, u_theta are odd and u_z, p, mu even in r (the restriction of a smooth 3D field).  All constants are exact
rationals; no field has a symmetry that a defect could hide behind (no r <-> z, no z -> -z symmetry).
"""
from __future__ import annotations

import sympy as sp

from axi_stokes import (Boundary, Field, r, z, s_, gauss_rect, gauss_half_annulus, sample_rect,
                        sample_half_annulus, forcing_route_a, stress)

R_ = sp.Rational
t = sp.Symbol("t", real=True)
s = r ** 2


def _ds(F):
    return sp.diff(F, s_)


def from_generators(G, C):
    """G, C as expressions in (s_, z).  Returns (u_r, u_z, u_t) in (r, z)."""
    sub = {s_: r ** 2}
    ur = -r * sp.diff(G, z).subs(sub)
    uz = (2 * G + 2 * s_ * _ds(G)).subs(sub)
    ut = r * C.subs(sub)
    return sp.Matrix([ur, uz, ut])


def even(F):
    return F.subs(s_, r ** 2)


# --------------------------------------------------------------------------------- the square domain [0,1]^2
SQ = {"kind": "rect", "r": [0, 1], "z": [0, 1], "axis": "r = 0"}
INT_SQ = gauss_rect(0, 1, 0, 1)
SAMPLE_SQ = sample_rect(0, 1, 0, 1)


def sq_boundaries(kinds):
    """kinds: dict edge -> bc for r1, z0, z1 (the axis is always 'axis')."""
    return [Boundary("axis", "axis", (R_(-1), R_(0)), {"t": t, "r": R_(0), "z": t, "range": [0, 1]}),
            Boundary("r1", kinds["r1"], (R_(1), R_(0)), {"t": t, "r": R_(1), "z": t, "range": [0, 1]}),
            Boundary("z0", kinds["z0"], (R_(0), R_(-1)), {"t": t, "r": t, "z": R_(0), "range": [0, 1]}),
            Boundary("z1", kinds["z1"], (R_(0), R_(1)), {"t": t, "r": t, "z": R_(1), "range": [0, 1]})]


# base generators (variable-mu MMS); the 1/5 and 1/2 scales were set by the calibration
G0 = R_(1, 5) * (sp.exp(z / 2) * sp.sin(R_(3, 2) * s_ + R_(2, 3) * z + R_(1, 4)) + s_ * z ** 2 / 3
                 + sp.cos(z + R_(1, 5)) / 2)
C0 = R_(1, 2) * (sp.cos(s_ + z / 2 + R_(1, 3)) + s_ * z / 2 + R_(1, 4))
P_DN = sp.cos(3 * s_ - 2 * z + R_(1, 2))     # raises |p|_H2 so that the O(h^2) pressure term dominates at N = 32-128
P0 = (sp.cos(R_(7, 5) * s_ + R_(1, 5)) * sp.sinh(R_(4, 5) * z + R_(1, 10)) + R_(2, 5) * s_ * z
      + P_DN)                               # P_DN: calibration (distorted-family sym pressure fit 2.16 without it)
M0 = R_(3, 2) + R_(1, 2) * sp.sin(s_ - z / 2 + R_(3, 10)) + s_ * z / 4
MU_CONST = R_(37, 100)


def axi_dirichlet():
    return Field("axi_dirichlet",
                 "Unit square (r, z) in [0,1]^2, axis r = 0; Dirichlet on r = 1, z = 0, z = 1 (data with u.n != 0: "
                 "the r-weighted flux of the P2 interpolant has an O(h^4) defect); swirl; variable mu; "
                 "parity-respecting; pressure nullspace (mean-free comparison).",
                 SQ, from_generators(G0, C0), even(P0), even(M0), ("lap", "sym"),
                 sq_boundaries({"r1": "dirichlet", "z0": "dirichlet", "z1": "dirichlet"}), "mean_zero",
                 INT_SQ, SAMPLE_SQ)


def axi_dirichlet_constmu():
    f = axi_dirichlet()
    f.name, f.mu = "axi_dirichlet_constmu", MU_CONST
    f.description = "As axi_dirichlet with mu = 37/100 (identity sym(gamma=0) = lap(gamma=mu))."
    return f


def axi_traction():
    return Field("axi_traction",
                 "Unit square, axis r = 0; Dirichlet on z = 0; inhomogeneous traction sigma_form n = g_form on "
                 "r = 1 (n = e_r; includes the swirl component sigma_theta,r, which differs between the forms) and "
                 "z = 1 (n = e_z); variable mu; pressure determined.",
                 SQ, from_generators(G0, C0), even(P0), even(M0), ("lap", "sym"),
                 sq_boundaries({"r1": "traction", "z0": "dirichlet", "z1": "traction"}), "determined",
                 INT_SQ, SAMPLE_SQ)


def axi_traction_constmu():
    f = axi_traction()
    f.name, f.mu = "axi_traction_constmu", R_(1)
    f.description = ("As axi_traction with mu = 1: the forms then differ only through the traction data (the "
                     "boundary part of the transpose term, hoop entries included).")
    return f


def axi_do_nothing(form):
    """Do-nothing (homogeneous natural condition of `form`) on z = 1; Dirichlet on r = 1 and z = 0."""
    # velocity scale per form (calibration): sym needs psi/25 for an asymptotic pressure at
    # N = 32-128 (fit 2.85 at scale 1); lap needs scale 1 (at 1/25 its velocity superconverges: 3.35 / 2.39)
    kap = 1 if form == "lap" else R_(1, 25)
    Gb, Cb = kap * G0, kap * C0
    Gzz1 = sp.diff(Gb, z, 2).subs(z, 1)
    if form == "lap":
        G = Gb - R_(1, 2) * (z - 1) ** 2 * Gzz1
    else:
        H = (8 * _ds(Gb) + 4 * s_ * sp.diff(Gb, s_, 2) - sp.diff(Gb, z, 2)).subs(z, 1)
        G = Gb + R_(1, 2) * (z - 1) ** 2 * H
    C = Cb - (z - 1) * sp.diff(Cb, z).subs(z, 1)
    u = from_generators(G, C)
    mu = even(M0)
    P = even(P0)
    k = 1 if form == "lap" else 2
    p = P + (k * mu * sp.diff(u[1], z) - P).subs(z, 1)
    fld = Field(f"axi_do_nothing_{form}",
                f"Unit square, axis r = 0; Dirichlet on r = 1 and z = 0; the {form} form's do-nothing condition "
                f"sigma_{form} n = 0 on z = 1 (and the other form's violated); swirl; variable mu; pressure "
                "determined.", SQ, u, p, mu, ("lap", "sym"),
                sq_boundaries({"r1": "dirichlet", "z0": "dirichlet", "z1": "do_nothing"}), "determined", INT_SQ,
                SAMPLE_SQ, extra={"do_nothing_form": form})
    return fld


def axi_fluxfree():
    G = (1 - s_) * z * (1 - z) * R_(4, 25) * (sp.exp(z / 2) * sp.sin(R_(3, 2) * s_ + R_(2, 3) * z + R_(1, 4))
                                             + R_(1, 2))
    C = C0 / 5
    fld = Field("axi_fluxfree",
                "Unit square, axis r = 0; Dirichlet on r = 1, z = 0, z = 1 with u.n = 0 there (psi = r^2 G, G = 0 on "
                "every non-axis edge: flux-free data, so pressure-constant treatments and the discrete r-weighted "
                "continuity can be compared at round-off); nonzero tangential velocity and swirl on the walls; "
                "variable mu; pressure nullspace.",
                SQ, from_generators(G, C), even(P0), even(M0), ("lap", "sym"),
                sq_boundaries({"r1": "dirichlet", "z0": "dirichlet", "z1": "dirichlet"}), "mean_zero",
                INT_SQ, SAMPLE_SQ)
    fld.checks["flux_free"] = True
    return fld


def axi_parity_swirl_even():
    """NEGATIVE CONTROL (not a case): u_theta = r C + cos(z + 1/3)/5, an EVEN-in-r swirl that does not vanish
    on the axis.  Not the restriction of a 3D field (u_theta e_theta is discontinuous at the axis); the strong axis
    condition u_theta = 0 contradicts it, so the discrete solution cannot converge to it near the axis."""
    u0 = from_generators(G0, C0)
    u = sp.Matrix([u0[0], u0[1], u0[2] + sp.cos(z + R_(1, 3)) / 5])
    return Field("axi_parity_swirl_even",
                 "NEGATIVE CONTROL: axi_dirichlet with u_theta += cos(z + 1/3)/5 (even in r, nonzero on the axis). "
                 "Negative control, not a case.", SQ, u, even(P0), even(M0), ("lap", "sym"),
                 sq_boundaries({"r1": "dirichlet", "z0": "dirichlet", "z1": "dirichlet"}), "mean_zero",
                 INT_SQ, SAMPLE_SQ, parity=False)


def axi_parity_violating():
    """NEGATIVE CONTROL (not a case): psi gets r^3 k(z), so u_z has a term 3 r k(z) (odd in r:
    a cone singularity of the 3D field on the axis) and u_r a term -r^2 k'(z) (even).  Used to show what the parity
    rule protects."""
    k = R_(1, 5) * sp.sin(z + R_(1, 3))
    u0 = from_generators(G0, C0)
    psi_extra = r ** 3 * k
    u = sp.Matrix([u0[0] - sp.diff(psi_extra, z) / r, u0[1] + sp.diff(psi_extra, r) / r, u0[2]])
    fld = Field("axi_parity_violating",
                "NEGATIVE CONTROL: axi_dirichlet plus psi += r^3 sin(z + 1/3)/5 (u_z gets an odd-in-r term: not "
                "the restriction of a smooth 3D field). Negative control, not a case.",
                SQ, sp.Matrix([sp.expand(c) for c in u]), even(P0), even(M0), ("lap", "sym"),
                sq_boundaries({"r1": "dirichlet", "z0": "dirichlet", "z1": "dirichlet"}), "mean_zero",
                INT_SQ, SAMPLE_SQ, parity=False)
    return fld


# --------------------------------------------------------------------------------- closed forms
def hp_pipe():
    """Hagen-Poiseuille: pipe radius 1, length 2, mu = 1/2, u_z = 1 - r^2, p = 4 mu (2 - z)."""
    mu = R_(1, 2)
    u = sp.Matrix([0, 1 - r ** 2, 0])
    p = 4 * mu * (2 - z)
    dom = {"kind": "rect", "r": [0, 1], "z": [0, 2], "axis": "r = 0"}
    b = [Boundary("axis", "axis", (R_(-1), R_(0)), {"t": t, "r": R_(0), "z": t, "range": [0, 2]}),
         Boundary("wall", "dirichlet", (R_(1), R_(0)), {"t": t, "r": R_(1), "z": t, "range": [0, 2]}),
         Boundary("inlet", "traction", (R_(0), R_(-1)), {"t": t, "r": t, "z": R_(0), "range": [0, 1]}),
         Boundary("outlet", "do_nothing", (R_(0), R_(1)), {"t": t, "r": t, "z": R_(2), "range": [0, 1]})]
    fld = Field("hp_pipe", "Hagen-Poiseuille pipe flow (closed form, in P2 x P1): r in [0,1], z in [0,2], mu = 1/2, "
                "u_z = 1 - r^2, p = 4 mu (2 - z); the lap do-nothing condition holds on z = 2, the sym one does not "
                "(sigma_rz = -2 mu r).", dom, u, p, mu, ("lap", "sym"), b, "determined",
                gauss_rect(0, 1, 0, 2), sample_rect(0, 1, 0, 2), extra={"do_nothing_form": "lap",
                "Q_per_radian": R_(1, 4), "Q_def": "int_0^1 u_z r dr (the 3D flow rate is 2 pi Q)"},
                exact_solution=True)
    # force (z) exerted by the fluid on the wall r = 1, per radian: normal out of the wall into the fluid is -e_r,
    # so F_z = int_0^2 (sigma n)_z r dz = -int_0^2 sigma_zr(1) dz  (balances the pressure drop times the area)
    srz = stress(u, p, mu, "sym")[1, 0].subs(r, 1)
    fld.extra["wall_force_z_per_radian"] = -sp.integrate(srz * 1, (z, 0, 2))
    fld.extra["pressure_drop_times_area_per_radian"] = (p.subs(z, 0) - p.subs(z, 2)) * R_(1, 2)
    return fld


ANN = {"kind": "rect", "r": [1, 2], "z": [0, 1], "axis": "none (annulus R1 = 1 < r < R2 = 2)"}
MU_COU = R_(7, 10)


def _ann_boundaries(outer="dirichlet"):
    return [Boundary("r1", "dirichlet", (R_(-1), R_(0)), {"t": t, "r": R_(1), "z": t, "range": [0, 1]}),
            Boundary("r2", outer, (R_(1), R_(0)), {"t": t, "r": R_(2), "z": t, "range": [0, 1]}),
            Boundary("z0", "do_nothing", (R_(0), R_(-1)), {"t": t, "r": t, "z": R_(0), "range": [1, 2]}),
            Boundary("z1", "do_nothing", (R_(0), R_(1)), {"t": t, "r": t, "z": R_(1), "range": [1, 2]})]


def _torque_inner(u, p, mu):
    """Torque (z) exerted by the fluid on the inner cylinder r = 1 (normal out of the body, +e_r), per radian:
    int_0^1 r (sigma n)_theta dz at r = 1, sym (Cauchy) stress."""
    S = stress(u, p, mu, "sym")
    return sp.simplify(sp.integrate((r * S[2, 0]).subs(r, 1), (z, 0, 1)))


def couette_rotation():
    u = sp.Matrix([0, 0, r])
    fld = Field("couette_rotation", "Solid rotation u_theta = r in the annulus 1 < r < 2, 0 < z < 1 (in P2: patch "
                "test, both forms); Dirichlet on both cylinders, do-nothing on z = 0, 1 (holds in both forms); "
                "zero Cauchy stress, so the S-form torque on either cylinder is 0.", ANN, u, R_(0), MU_COU,
                ("lap", "sym"), _ann_boundaries(), "determined", gauss_rect(1, 2, 0, 1), sample_rect(1, 2, 0, 1),
                extra={"do_nothing_form": "both"}, exact_solution=True)
    fld.extra["torque_inner_per_radian"] = _torque_inner(u, 0, MU_COU)
    return fld


def couette_circular():
    A, B = R_(-1, 3), R_(4, 3)
    u = sp.Matrix([0, 0, A * r + B / r])
    fld = Field("couette_circular", "Circular Couette: inner cylinder r = 1 at u_theta = 1, outer r = 2 at rest, "
                "u_theta = -r/3 + 4/(3r) (not in P2), p = 0; do-nothing on z = 0, 1 (holds in both forms); mu = 7/10.",
                ANN, u, R_(0), MU_COU, ("lap", "sym"), _ann_boundaries(), "determined", gauss_rect(1, 2, 0, 1),
                sample_rect(1, 2, 0, 1), extra={"do_nothing_form": "both"}, exact_solution=True)
    fld.extra["torque_inner_per_radian"] = _torque_inner(u, 0, MU_COU)
    return fld


def couette_traction():
    tau = R_(1, 2)
    B = -2 * tau / MU_COU
    A = -B
    u = sp.Matrix([0, 0, A * r + B / r])
    fld = Field("couette_traction", "Shear-driven circular Couette: u_theta = 0 on r = 1; on r = 2 the physical "
                "shear stress sigma_theta,r = 1/2 (sym traction (0, 0, 1/2); the lap form receives its own "
                "pseudo-traction mu d_r u_theta); do-nothing on z = 0, 1; p = 0; mu = 7/10.", ANN, u, R_(0), MU_COU,
                ("lap", "sym"), _ann_boundaries("traction"), "determined", gauss_rect(1, 2, 0, 1),
                sample_rect(1, 2, 0, 1), extra={"do_nothing_form": "both"}, exact_solution=True)
    fld.extra["torque_inner_per_radian"] = _torque_inner(u, 0, MU_COU)
    fld.extra["applied_torque_per_radian"] = 2 * 2 * tau        # r * tau * (R2 dz-length = 2 * 1)
    return fld


# --------------------------------------------------------------------------------- sphere (exact Stokes solutions)
A_SPH, ROUT, MU_SPH = R_(1), R_(4), R_(37, 100)
rho = sp.sqrt(r ** 2 + z ** 2)


def _sphere_boundaries():
    th = sp.Symbol("t", real=True)
    return [Boundary("axis_top", "axis", (R_(-1), R_(0)), {"t": th, "r": R_(0), "z": th, "range": [1, 4]}),
            Boundary("axis_bottom", "axis", (R_(-1), R_(0)), {"t": th, "r": R_(0), "z": th, "range": [-4, -1]}),
            Boundary("sphere", "dirichlet", (-sp.sin(th), -sp.cos(th)),
                     {"t": th, "r": sp.sin(th), "z": sp.cos(th), "range": [0, sp.pi]}),
            Boundary("outer", "dirichlet", (sp.sin(th), sp.cos(th)),
                     {"t": th, "r": 4 * sp.sin(th), "z": 4 * sp.cos(th), "range": [0, sp.pi]})]


def _derive_sphere_translation():
    """Derive (not recall) the Stokes flow past a fixed sphere of radius a in a uniform stream U e_z:
    ansatz psi = (U/2) r^2 (1 + A a/rho + B a^3/rho^3), p = C mu U a z / rho^3; the constants follow from the
    momentum equation at sample points and u = 0 on rho = a; the full residual is then checked symbolically."""
    A, B, C = sp.symbols("A B C")
    U, a, mu = R_(1), A_SPH, MU_SPH
    psi = U / 2 * r ** 2 * (1 + A * a / rho + B * a ** 3 / rho ** 3)
    u = sp.Matrix([-sp.diff(psi, z) / r, sp.diff(psi, r) / r, 0])
    p = C * mu * U * a * z / rho ** 3
    f = forcing_route_a(u, p, mu, "lap")
    eqs = []
    for (rv, zv) in ((R_(1, 2), R_(3, 2)), (R_(2), R_(1, 3)), (R_(3, 2), R_(-1))):
        eqs += [sp.simplify(f[i].subs({r: rv, z: zv})) for i in (0, 1)]
    for th in (R_(1, 3), R_(1)):
        eqs += [sp.simplify(c.subs({r: sp.sin(th), z: sp.cos(th)})) for c in u[:2]]
    sol = sp.solve(eqs, [A, B, C], dict=True)
    assert len(sol) == 1, sol
    sol = sol[0]
    return sp.Matrix([sp.simplify(c.subs(sol)) for c in u]), sp.simplify(p.subs(sol)), sol


def sphere_translation():
    u, p, sol = _derive_sphere_translation()
    fld = Field("sphere_translation", "Stokes flow past a fixed sphere of radius 1 in the uniform stream e_z "
                "(derived from the stream-function ansatz, constants solved, residual checked); mu = 37/100; meridional "
                "domain 1 < rho < 4 (rho^2 = r^2 + z^2) with the EXACT solution as Dirichlet data on rho = 4 (pure "
                "discretization test); axis on both segments; pressure nullspace.",
                {"kind": "half_annulus", "rho": [1, 4], "axis": "r = 0"}, u, p, MU_SPH, ("lap", "sym"),
                _sphere_boundaries(), "mean_zero", gauss_half_annulus(1, 4), sample_half_annulus(1, 4),
                extra={"ansatz_constants": {str(k): str(v) for k, v in sol.items()}}, exact_solution=True)
    th = sp.Symbol("th", real=True)
    S = stress(u, p, MU_SPH, "sym")
    n = sp.Matrix([sp.sin(th), sp.cos(th), 0])           # out of the body, into the fluid
    on = {r: sp.sin(th), z: sp.cos(th)}
    tz = sp.simplify((S * n)[1].subs(on, simultaneous=True))
    Fz = sp.simplify(sp.integrate(tz * sp.sin(th), (th, 0, sp.pi)))  # per radian: int t_z r ds, r = sin, ds = dth
    fld.extra["drag_z_per_radian"] = Fz
    fld.extra["drag_z_3d"] = sp.simplify(2 * sp.pi * Fz)
    return fld


def sphere_rotation():
    u = sp.Matrix([0, 0, A_SPH ** 3 * r / rho ** 3])
    fld = Field("sphere_rotation", "Sphere of radius 1 rotating about the axis with unit angular velocity in fluid "
                "at rest: u_theta = r / rho^3 (checked to solve Stokes with p = 0); exact data on rho = 4; swirl on a "
                "curved boundary; mu = 37/100.", {"kind": "half_annulus", "rho": [1, 4], "axis": "r = 0"}, u, R_(0),
                MU_SPH, ("lap", "sym"), _sphere_boundaries(), "mean_zero", gauss_half_annulus(1, 4),
                sample_half_annulus(1, 4), exact_solution=True)
    th = sp.Symbol("th", real=True)
    S = stress(u, 0, MU_SPH, "sym")
    n = sp.Matrix([sp.sin(th), sp.cos(th), 0])
    on = {r: sp.sin(th), z: sp.cos(th)}
    tth = sp.simplify((S * n)[2].subs(on, simultaneous=True))
    T = sp.simplify(sp.integrate(sp.sin(th) * tth * sp.sin(th), (th, 0, sp.pi)))  # int r * t_theta * r ds
    fld.extra["torque_z_per_radian"] = T
    fld.extra["torque_z_3d"] = sp.simplify(2 * sp.pi * T)
    return fld


ALL = [axi_dirichlet, axi_dirichlet_constmu, axi_traction, axi_traction_constmu,
       lambda: axi_do_nothing("lap"), lambda: axi_do_nothing("sym"), axi_fluxfree, axi_parity_violating, axi_parity_swirl_even,
       hp_pipe, couette_rotation, couette_circular, couette_traction, sphere_translation, sphere_rotation]
