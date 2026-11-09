"""Independent MMS generator for the fluids Stokes tests (cpp/tests/test_fluids_stokes.cpp), written apart from
scripts/fluids/gen_mms_stokes2d.py so that the two cross-check each other.

1. AXI: axisymmetric Stokes with swirl, (r,z) in [0,1]x[0,1], unknowns (u_r, u_z, u_t, p), parity-respecting:
   psi = r^2 (1-r^2) z (1-z) G(r^2,z)   (Stokes stream function; vanishes on r=1, z=0, z=1)
   u_r = -(1/r) d_z psi  (odd), u_z = (1/r) d_r psi (even), u_t = r H(r^2, z) (odd), p = P(r^2, z) (even)
   => u.n = 0 pointwise on r=1, z=0, z=1 (flux-compatible data), tangential velocity nonzero.
   Forcing for (L) constant nu=1 and (S) variable nu(r^2,z), from the cylindrical formulas; CROSS-CHECKED
   numerically against an independent Cartesian 3D route (u(x,y,z) built from the cylindrical field).
2. PLANAR do-nothing fields on [0,1]^2 with the natural boundary at x=1, one per form (each satisfies its form's
   homogeneous condition sigma_form n = 0 on x=1 and violates the other form's).
3. PLANAR enclosed field with u.n != 0 pointwise but zero net flux (P2 interpolant has an O(h^4) flux defect).
Writes independent_mms_generated.hpp (run it from this directory).
"""
import sympy as sp, random, math

r, z, x, y = sp.symbols("r z x y", real=True)
s = r**2
G = sp.exp(z / 2) * sp.cos(s + z / 3) + sp.Rational(1, 3) * s * z
psi = s * (1 - s) * z * (1 - z) * G
ur = -sp.diff(psi, z) / r
uz = sp.diff(psi, r) / r
ut = r * (1 + z) * sp.cos(s / 2 + z) * sp.Rational(1, 2)
p = sp.cos(sp.pi * s) * sp.sin(z + sp.Rational(1, 5)) + s * z
nuv = 1 + sp.Rational(1, 2) * sp.sin(s + z)
ur, uz = sp.expand(sp.cancel(ur)), sp.expand(sp.cancel(uz))
# divergence (cylindrical): d_r u_r + u_r/r + d_z u_z = 0
assert sp.simplify(sp.diff(ur, r) + ur / r + sp.diff(uz, z)) == 0

def lap(f):
    return sp.diff(f, r, 2) + sp.diff(f, r) / r + sp.diff(f, z, 2)

def forcing(nu, form):
    if form == "L":
        fr = -nu * (lap(ur) - ur / r**2) + sp.diff(p, r)
        fz = -nu * lap(uz) + sp.diff(p, z)
        ft = -nu * (lap(ut) - ut / r**2)
        return fr, fz, ft
    err, ezz, ett = sp.diff(ur, r), sp.diff(uz, z), ur / r
    erz = (sp.diff(ur, z) + sp.diff(uz, r)) / 2
    ert = (sp.diff(ut, r) - ut / r) / 2
    ezt = sp.diff(ut, z) / 2
    T = lambda e: 2 * nu * e
    fr = -(sp.diff(r * T(err), r) / r + sp.diff(T(erz), z) - T(ett) / r) + sp.diff(p, r)
    fz = -(sp.diff(r * T(erz), r) / r + sp.diff(T(ezz), z)) + sp.diff(p, z)
    ft = -(sp.diff(r**2 * T(ert), r) / r**2 + sp.diff(T(ezt), z))
    return fr, fz, ft

fL = forcing(sp.Integer(1), "L")
fS = forcing(nuv, "S")

# ---- independent Cartesian 3D cross-check of both forcings at random points
X, Y, Z = sp.symbols("X Y Z", real=True)
R = sp.sqrt(X**2 + Y**2)
sub = {r: R, z: Z}
cth, sth = X / R, Y / R
U = [ (ur * cth - ut * sth).subs(sub), (ur * sth + ut * cth).subs(sub), uz.subs(sub)]
P3 = p.subs(sub)
V = [X, Y, Z]
def cart_force(nu, form):
    nu3 = nu.subs(sub) if hasattr(nu, "subs") else nu
    Gm = [[sp.diff(U[i], V[j]) for j in range(3)] for i in range(3)]
    if form == "L":
        tau = [[nu3 * Gm[i][j] for j in range(3)] for i in range(3)]
    else:
        tau = [[nu3 * (Gm[i][j] + Gm[j][i]) for j in range(3)] for i in range(3)]
    return [-sum(sp.diff(tau[i][j], V[j]) for j in range(3)) + sp.diff(P3, V[i]) for i in range(3)]
rng = random.Random(3)
for form, fc, nu in (("L", fL, sp.Integer(1)), ("S", fS, nuv)):
    F3 = cart_force(nu, form)
    for _ in range(4):
        rr = sp.Rational(rng.randint(15, 95), 100); th = sp.Rational(rng.randint(0, 628), 100); zz = sp.Rational(rng.randint(5, 95), 100)
        c_, s_ = sp.cos(th), sp.sin(th)
        pt = {X: rr * c_, Y: rr * s_, Z: zz}
        fx, fy, fzc = [sp.N(f.subs(pt), 40) for f in F3]
        frc = fx * sp.N(c_, 40) + fy * sp.N(s_, 40)
        ftc = -fx * sp.N(s_, 40) + fy * sp.N(c_, 40)
        cyl = [sp.N(f.subs({r: rr, z: zz}), 40) for f in fc]
        d = max(abs(frc - cyl[0]), abs(fzc - cyl[1]), abs(ftc - cyl[2]))
        sc = max(abs(c) for c in cyl)
        assert d < 1e-20 * max(1, sc), (form, d)
    print("axi forcing", form, "cross-checked against the Cartesian route at 4 points")
# parity: f_r, f_t odd in r; f_z even (check numerically via r -> -r of the analytic expressions)
for form, fc in (("L", fL), ("S", fS)):
    for (f, parity) in zip(fc, (-1, 1, -1)):
        a = sp.N(f.subs({r: sp.Rational(3, 10), z: sp.Rational(2, 5)}), 25)
        b = sp.N(f.subs({r: -sp.Rational(3, 10), z: sp.Rational(2, 5)}), 25)
        assert abs(a - parity * b) < 1e-18 * max(1, abs(a)), (form, a, b)
print("parity ok")

# ---- planar do-nothing fields (natural boundary x = 1); nu = 1 (L), nu_p variable (S)
nu_p = 1 + sp.Rational(1, 2) * sp.sin(2 * x + y)
psi0 = sp.exp(sp.Rational(3, 5) * x) * sp.sin(sp.Rational(9, 5) * y + sp.Rational(1, 2)) + sp.Rational(3, 10) * y**2 * sp.cos(sp.Rational(21, 10) * x - sp.Rational(2, 5))
p0 = sp.cos(sp.Rational(19, 10) * x + sp.Rational(1, 5)) * sp.sinh(sp.Rational(4, 5) * y + sp.Rational(1, 10)) + sp.Rational(2, 5) * x * y
def vel(ps):
    return sp.Matrix([sp.diff(ps, y), -sp.diff(ps, x)])
def planar_force(u, pp, nu, form):
    Xs = [x, y]
    Gm = sp.Matrix(2, 2, lambda i, j: sp.diff(u[i], Xs[j]))
    tau = nu * Gm if form == "L" else nu * (Gm + Gm.T)
    return [-sum(sp.diff(tau[i, j], Xs[j]) for j in range(2)) + sp.diff(pp, Xs[i]) for i in range(2)], tau
# L do-nothing: d_x u_y = 0 and nu d_x u_x - p = 0 on x = 1
psiL = psi0 - sp.Rational(1, 2) * (x - 1)**2 * sp.diff(psi0, x, 2).subs(x, 1)
uL = vel(psiL)
pL = p0 + (sp.diff(uL[0], x) - p0).subs(x, 1)
fLp, tauL = planar_force(uL, pL, sp.Integer(1), "L")
# S do-nothing: shear nu_p(d_y u_x + d_x u_y) = 0 and 2 nu_p d_x u_x - p = 0 on x = 1
psiS = psi0 + sp.Rational(1, 2) * (x - 1)**2 * (sp.diff(psi0, y, 2) - sp.diff(psi0, x, 2)).subs(x, 1)
uS = vel(psiS)
pS = p0 + (2 * nu_p * sp.diff(uS[0], x) - p0).subs(x, 1)
fSp, tauS = planar_force(uS, pS, nu_p, "S")
for name, u_, p_, nu_, own, other in (("L", uL, pL, sp.Integer(1), "L", "S"), ("S", uS, pS, nu_p, "S", "L")):
    for form, expect_zero in ((own, True), (other, False)):
        Xs = [x, y]
        Gm = sp.Matrix(2, 2, lambda i, j: sp.diff(u_[i], Xs[j]))
        tau = nu_ * Gm if form == "L" else nu_ * (Gm + Gm.T)
        trac = [(tau[i, 0] - (p_ if i == 0 else 0)).subs(x, 1) for i in range(2)]
        vals = [max(abs(sp.N(t.subs(y, yy))) for yy in (0.1, 0.37, 0.8)) for t in trac]
        if expect_zero:
            assert max(vals) < 1e-12, (name, form, vals)
        else:
            assert max(vals) > 1e-2, (name, form, vals)
print("do-nothing fields: own form traction-free on x=1, other form not")
# ---- enclosed planar field with u.n != 0 pointwise but zero net flux (u = curl psi0)
uF = vel(psi0)
fFp, _ = planar_force(uF, p0, sp.Integer(1), "L")

def cc(e):
    return sp.ccode(e)
out = ["// GENERATED by gen_independent_mms.py (independent MMS generator). Do not edit.", "#pragma once", "#include <cmath>",
       "namespace indep {"]
def emit_vec(name, comps, args):
    out.append(f"inline void {name}({', '.join('double ' + a for a in args)}, double* v) {{")
    for i, c in enumerate(comps):
        out.append(f"  v[{i}] = {cc(c)};")
    out.append("}")
def emit_s(name, e, args):
    out.append(f"inline double {name}({', '.join('double ' + a for a in args)}) {{ return {cc(e)}; }}")
# axi: exact u (ur,uz,ut), its gradient [d_r, d_z] per component, p, nu, forcings
emit_vec("axi_u", [ur, uz, ut], ["r", "z"])
emit_vec("axi_grad", [sp.diff(c, v) for c in (ur, uz, ut) for v in (r, z)], ["r", "z"])
emit_s("axi_p", p, ["r", "z"])
emit_s("axi_nu", nuv, ["r", "z"])
emit_vec("axi_fL", list(fL), ["r", "z"])
emit_vec("axi_fS", list(fS), ["r", "z"])
for tag, u_, p_, f_ in (("dnL", uL, pL, fLp), ("dnS", uS, pS, fSp), ("flux", uF, p0, fFp)):
    emit_vec(tag + "_u", list(u_), ["x", "y"])
    emit_vec(tag + "_grad", [sp.diff(u_[i], v) for i in range(2) for v in (x, y)], ["x", "y"])
    emit_s(tag + "_p", p_, ["x", "y"])
    emit_vec(tag + "_f", list(f_), ["x", "y"])
emit_s("dn_nu", nu_p, ["x", "y"])
out += ["}  // namespace indep", ""]
open("independent_mms_generated.hpp", "w").write("\n".join(out))
print("wrote independent_mms_generated.hpp", sum(len(l) for l in out), "chars")
