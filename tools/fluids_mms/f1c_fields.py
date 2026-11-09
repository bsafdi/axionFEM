"""F1c verification fields.  All constants are exact rationals.

Design rules:
- non-polynomial, no symmetry between x, y, z (a swapped block or a mis-ordered component is O(1));
- velocity SMALL against the pressure (A / 10): the velocity-coupled O(h^3) part of the pressure error stays below
  the O(h^2) best-approximation part already at N = 8 (pre-asymptotic pressure);
- LOW frequencies (all wave numbers <= 1.3): the 3D levels stop at N = 32 (direct solves of ~0.9 M unknowns), so the
  asymptotic regime must be reached at N = 8 (the chemistry C3b 3D MMS failed pre-asymptotically);
- every face violates the homogeneous condition of every OTHER natural/slip kind its field uses.
"""
from __future__ import annotations

import sympy as sp

from stokes3d import Face, Field, cayley, curl, x, y, z

R_ = sp.Rational
DOM_CUBE = {"kind": "unit cube [0,1]^3 (local coordinates)"}


def faces_cube(bcs: dict):
    out = []
    for i, c in enumerate("xyz"):
        for v, sgn in ((0, -1), (1, 1)):
            n = [0, 0, 0]
            n[i] = sgn
            out.append(Face(f"{c}{v}", bcs.get(f"{c}{v}", "dirichlet"), tuple(n), (i, v)))
    return out


# ------------------------------------------------------------------------------------------------ cube fields
# SPLIT design: on piecewise-uniform nested 3D meshes the velocity error
# induced by the pressure superconverges (order 4/3, the F1a A1 mechanism) and the pressure error induced by the
# velocity is of order 3; with N <= 32 no single scaling keeps both contaminations below ~20 % at N = 8.  Each MMS
# case therefore uses a VELOCITY field (p = 0 or the pressure the boundary condition forces) for the velocity items and
# a PRESSURE field (u = 0) for the pressure items.  Both solve the full Stokes system with variable mu.
A_CUBE = sp.Matrix([sp.sin(R_(4, 5) * y + R_(1, 2) * z + R_(3, 10)) * sp.exp(R_(2, 5) * x),
                    sp.cos(R_(7, 10) * x - R_(3, 5) * z + R_(1, 5)) * (1 + y * z / 3),
                    sp.sin(R_(9, 10) * x + R_(1, 2) * y - R_(1, 10)) * sp.exp(-R_(3, 10) * z)])
U_CUBE = curl(A_CUBE)
P_CUBE = sp.cos(R_(13, 10) * x - R_(7, 10) * y + R_(9, 10) * z + R_(1, 5)) \
    + R_(2, 5) * sp.sinh(R_(3, 5) * y + R_(2, 5) * z - R_(1, 10)) + x * z / 3
MU_CUBE = R_(3, 2) + R_(1, 2) * sp.sin(R_(11, 10) * x - R_(3, 5) * y + R_(4, 5) * z + R_(3, 10)) + x * z / 5
ZERO3 = sp.zeros(3, 1)
TRACTION3 = {"x1": "traction", "y1": "traction", "z1": "traction"}


def cube_u():
    return Field("f1c_cube_u", "T1/T2 velocity field: u = curl A, p = 0, variable mu, both forms (Dirichlet on all "
                 "faces in T1; traction g = sigma_form n on x1, y1, z1 in T2)", DOM_CUBE, U_CUBE, sp.Integer(0),
                 MU_CUBE, ("lap", "sym"), faces_cube({}), "mean_zero", extra={"forms_must_differ_interior": True})


def cube_p():
    return Field("f1c_cube_p", "T1/T2 pressure field: u = 0, p = P (non-polynomial), variable mu", DOM_CUBE, ZERO3,
                 P_CUBE, MU_CUBE, ("lap", "sym"), faces_cube({}), "mean_zero")


def cube_u_constmu(mu, tag, with_p=False):
    return Field(f"f1c_cube_u_mu{tag}", f"constant-mu field, mu = {mu}: " + (
                 "u = curl A AND p = P (T1 identity: the pressure comparison needs an O(1) exact pressure)" if with_p
                 else "velocity field, p = 0 (T2: the forms then differ only through the traction data)"),
                 DOM_CUBE, U_CUBE, P_CUBE if with_p else sp.Integer(0), sp.nsimplify(mu), ("lap", "sym"),
                 faces_cube({}), "mean_zero")


def _dn_u(form):
    """sigma_form(u, p) e_x = 0 on x = 1: u = u0 + curl C, C = (x-1)^2/2 (0, c2(y,z), c3(y,z)); p the function of
    (y, z) that the normal component forces: p(y, z) = k mu d_x u_x (1, y, z), k = 1 (lap) / 2 (sym)."""
    u0 = U_CUBE
    at1 = {x: 1}
    if form == "lap":
        c3 = sp.diff(u0[1], x).subs(at1)
        c2 = -sp.diff(u0[2], x).subs(at1)
    else:
        c3 = (sp.diff(u0[1], x) + sp.diff(u0[0], y)).subs(at1)
        c2 = -(sp.diff(u0[2], x) + sp.diff(u0[0], z)).subs(at1)
    u = u0 + curl(sp.Matrix([0, c2, c3]) * (x - 1) ** 2 / 2)
    k = 1 if form == "lap" else 2
    p = (k * MU_CUBE * sp.diff(u[0], x)).subs(at1)
    return Field(f"f1c_dn_u_{form}", f"T3 velocity field: sigma_{form} n = 0 on x = 1 (the other form's condition is "
                 "violated); Dirichlet elsewhere", DOM_CUBE, u, p, MU_CUBE, ("lap", "sym"),
                 faces_cube({"x1": "do_nothing"}), "determined", extra={"do_nothing_form": form})


def dn_p():
    return Field("f1c_dn_p", "T3 pressure field: u = 0, p = (1 - x) P, so sigma n = -p n = 0 on x = 1 in BOTH forms",
                 DOM_CUBE, ZERO3, (1 - x) * P_CUBE, MU_CUBE, ("lap", "sym"), faces_cube({"x1": "do_nothing"}),
                 "determined", extra={"do_nothing_form": "both"})


# ------------------------------------------------------------------------------------------------ rotated slip cube
R_SLIP = cayley((R_(1, 4), R_(1, 3), R_(1, 5)))
_S, _Z = (x - 1) ** 2, z ** 2


def rot_slip_u():
    """Slip on local z = 0, symmetry on local x = 1 (meeting along the local edge {x = 1, z = 0}), Dirichlet on the
    other four faces.  Parities: u_x, u_y even in z, u_z odd in z (slip on z = 0); u_x odd in s = x - 1, u_y, u_z
    even in s (symmetry on x = 1).  u = curl A with A_x odd z/even s, A_y odd z/odd s, A_z even z/odd s; p = 0."""
    A = sp.Matrix([z * sp.cos(R_(4, 5) * y + R_(3, 5) * _S - R_(1, 2) * _Z + R_(1, 5)),
                   z * (x - 1) * sp.exp(R_(1, 2) * y - R_(3, 10) * _S + R_(2, 5) * _Z),
                   (x - 1) * sp.sin(R_(9, 10) * y + R_(2, 5) * _S + R_(7, 10) * _Z + R_(1, 10))])
    return Field("f1c_rot_slip_u", "T4 velocity field: rotated cube (exact rational Cayley rotation), slip on local "
                 "z0, symmetry on local x1, Dirichlet elsewhere; local frame; p = 0",
                 {"kind": "rotated unit cube: x = R x_loc, x_loc in [0,1]^3"}, curl(A), sp.Integer(0), _MU_SLIP,
                 ("lap", "sym"), faces_cube({"z0": "slip", "x1": "symmetry"}), "mean_zero",
                 frame={"R": R_SLIP, "c": (0, 0, 0)})


_MU_SLIP = R_(3, 2) + R_(1, 2) * sp.sin(R_(11, 10) * _S - R_(3, 5) * y + R_(4, 5) * _Z + R_(3, 10)) + _S * _Z / 5


def rot_slip_p():
    p = sp.cos(R_(13, 10) * y - R_(7, 10) * _S + R_(9, 10) * _Z + R_(1, 5)) \
        + R_(2, 5) * sp.sinh(R_(3, 5) * y + R_(2, 5) * _Z - R_(1, 10)) + _S * y / 3
    return Field("f1c_rot_slip_p", "T4 pressure field: u = 0, p even in z and in x - 1 (slip/symmetry hold "
                 "trivially; the wrong-attribute test is carried by the velocity field)",
                 {"kind": "rotated unit cube"}, ZERO3, p, _MU_SLIP, ("lap", "sym"),
                 faces_cube({"z0": "slip", "x1": "symmetry"}), "mean_zero", frame={"R": R_SLIP, "c": (0, 0, 0)},
                 extra={"cross_bc_exempt": "u = 0: every face satisfies slip; f1c_rot_slip_u carries the test"})


def rot_slip_patch():
    """T4 non-MMS cell: uniform flow along the local edge {x = 1, z = 0}, u_loc = e_y, p = 0: tangent to
    both the slip and the symmetry face, stress-free, in P2; Dirichlet u = e_y on the other faces."""
    return Field("f1c_rot_slip_patch", "T4 patch: uniform flow along the slip-symmetry edge (in P2: exact to round-off)",
                 {"kind": "rotated unit cube"}, sp.Matrix([0, 1, 0]), sp.Integer(0), _MU_SLIP, ("lap", "sym"),
                 faces_cube({"z0": "slip", "x1": "symmetry"}), "mean_zero", frame={"R": R_SLIP, "c": (0, 0, 0)},
                 extra={"cross_bc_exempt": "a uniform tangential flow also satisfies slip on x0 (u.n = 0, zero "
                                           "traction); the MMS velocity field carries the attribute test"})


# ------------------------------------------------------------------------------------------------ cavity lid (X1)
def cavity_lid():
    g = 16 * x ** 2 * (1 - x) ** 2 * 16 * y ** 2 * (1 - y) ** 2
    return Field("f1c_cavity_lid", "X1: regularized lid u = (256 x^2(1-x)^2 y^2(1-y)^2, 0, 0) on z = 1, u = 0 on the "
                 "other faces, mu = 1, f = 0 (DATA ONLY: u below is the lid trace extended by z^0, not a solution)",
                 DOM_CUBE, sp.Matrix([g, 0, 0]), sp.Integer(0), sp.Integer(1), ("lap", "sym"),
                 faces_cube({}), "mean_zero", no_forcing=True)


# ------------------------------------------------------------------------------------------------ sphere (S1)
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
XV = sp.Matrix([x, y, z])
EZ = sp.Matrix([0, 0, 1])
SPH_A, SPH_R, SPH_U, SPH_MU = sp.Integer(1), sp.Integer(4), sp.Integer(1), sp.Rational(37, 100)   # = F1b S1
# Stokes basis fields (velocity, pressure / mu), each checked by the generator (div u = 0, mu lap u = grad p):
BASIS = {
    "uniform": (EZ, sp.Integer(0)),
    "stokeslet": (EZ / r + z * XV / r ** 3, 2 * z / r ** 3),
    "dipole": (EZ / r ** 3 - 3 * z * XV / r ** 5, sp.Integer(0)),
    "quadratic": (r ** 2 * EZ - z * XV / 2, 5 * z),
}


def _sphere(outer: str):
    """Sphere of radius a at rest (u = 0 on r = a); on r = R either the exact unbounded field (uniform flow U e_z at
    infinity) or the uniform velocity U e_z (concentric spheres: the container moves).  Coefficients solved
    symbolically from the four boundary equations (alpha(r) e_z + beta(r) z x on each sphere)."""
    cs = sp.symbols("c0:4")
    names = list(BASIS)
    if outer == "unbounded":
        use = ["uniform", "stokeslet", "dipole"]
    else:
        use = names
    u = sp.zeros(3, 1)
    pp = sp.Integer(0)
    for c, nm in zip(cs, names):
        if nm in use:
            u += c * BASIS[nm][0]
            pp += c * BASIS[nm][1]
    # alpha, beta on the axis point (0, 0, rho) and an off-axis point give the two radial functions
    eqs = []
    rho = sp.Symbol("rho", positive=True)

    def ab(expr_u, rr):
        # on the sphere |x| = rr at (rr, 0, 0): u = alpha e_z (z = 0); at (0, 0, rr): u_z = alpha + beta rr^2
        a_ = expr_u[2].subs({x: rr, y: 0, z: 0})
        b_ = (expr_u[2].subs({x: 0, y: 0, z: rr}) - a_) / rr ** 2
        return sp.simplify(a_), sp.simplify(b_)
    a1, b1 = ab(u, SPH_A)
    eqs += [a1, b1]
    if outer == "unbounded":
        eqs.append(cs[0] - SPH_U)
    else:
        a2, b2 = ab(u, SPH_R)
        eqs += [a2 - SPH_U, b2]
    sol = sp.solve(eqs, [c for c, nm in zip(cs, names) if nm in use], dict=True)[0]
    u = sp.simplify(u.subs(sol))
    pp = sp.simplify(pp.subs(sol)) * SPH_MU
    # force exerted BY the fluid ON the sphere: only the Stokeslet carries it, F = -8 pi mu c_stokeslet e_z (sign
    # checked numerically by the generator against a quadrature of sigma n over r = a)
    c_st = sol[cs[1]]
    drag = sp.nsimplify(sp.simplify(8 * sp.pi * SPH_MU * c_st)) * -1
    name = f"f1c_sphere_{outer}"
    desc = ("S1: Stokes flow past a sphere a = 1 at rest, mu = 37/100 (as F1b), U = 1 along z; shell 1 < r < 4; " +
            ("exact unbounded field as Dirichlet data on r = 4; drag 6 pi mu a U = 111 pi/50"
             if outer == "unbounded" else "concentric container r = 4 moving with U e_z (no-slip); wall-corrected drag"))
    return Field(name, desc, {"kind": "spherical shell 1 < r < 4"}, u, pp, SPH_MU, ("lap", "sym"),
                 [Face("r1", "dirichlet", (0, 0, 0), (0, 1)), Face("r4", "dirichlet", (0, 0, 0), (0, 4))],
                 "mean_zero", box=None, shell=(1.0, 4.0),
                 extra={"drag_z_exact": sp.simplify(drag), "coefficients": {str(k): str(v) for k, v in sol.items()}})


def sphere_unbounded():
    return _sphere("unbounded")


def sphere_concentric():
    return _sphere("concentric")


ALL = [cube_u, cube_p, lambda: cube_u_constmu(sp.Rational(37, 100), "037", True), lambda: cube_u_constmu(1, "1"),
       lambda: _dn_u("lap"), lambda: _dn_u("sym"), dn_p, rot_slip_u, rot_slip_p, rot_slip_patch, cavity_lid,
       sphere_unbounded, sphere_concentric]
