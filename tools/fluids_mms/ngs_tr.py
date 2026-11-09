"""Independent NGSolve reference solver for transient Navier-Stokes.

NGSolve 6.2.2608, Taylor-Hood P2/P1, written from the continuum equations (never from axionFEM code).  Planar and axisymmetric with swirl; both viscous forms; standard convection.
Time: BDF1 / variable-step BDF2 (BDF1 starter), Newton per step (NGSolve symbolic linearization, dual-norm residual
to 1e-12) or semi-implicit (linearly implicit, extrapolated advecting velocity); initial data by the discrete L2
(Leray) projection onto discretely divergence-free fields with the Dirichlet data; adaptive BDF2 with a Milne-type
estimator (quadratic-extrapolation predictor) and an error-per-step controller; single steps from a given history
(effectivity).  Dirichlet data by P2 Lagrange NODAL interpolation (vertex values; edge bubbles set so the midpoint
value is exact), as F2a's ngs_ns.py.  Errors are computed by the fixed check module f3a_transient_check through
an Evaluator adapter.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
import ngsolve as ngs  # noqa: E402
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element1D, FaceDescriptor, Pnt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f3a_transient_check as C  # noqa: E402

ngs.SetNumThreads(1)
_FN = {'sin': ngs.sin, 'cos': ngs.cos, 'exp': ngs.exp, 'sinh': ngs.sinh, 'cosh': ngs.cosh, 'sqrt': ngs.sqrt,
       'log': ngs.log, 'pi': math.pi, 'tanh': lambda a: (ngs.exp(a) - ngs.exp(-a)) / (ngs.exp(a) + ngs.exp(-a))}


# ============================================================================================== coefficient functions
def cf_from_sympy(s, coords, tpar):
    syms = sp.symbols(" ".join(coords) + " t", real=True)
    e = sp.sympify(s, locals={str(v): v for v in syms}) if isinstance(s, str) else s
    if not e.free_symbols:
        return ngs.CF(float(e))
    f = sp.lambdify(syms, e, modules=[_FN, 'math'])
    return f(ngs.x, ngs.y, tpar)


def womersley_cfs(fld, tpar):
    """Real coefficient functions of the Womersley profile U and dU/ds (s = y planar, r axisymmetric), from the complex
    amplitude: planar via sympy real expansion; pipe via the power series of J0 (25 terms, |k r / 2| <= 2)."""
    alpha, om = 4.0, 2 * math.pi
    K = om
    if fld.geometry == "planar":
        y_, t_ = sp.symbols("y t", real=True)
        lam = (1 + sp.I) * 4 / sp.sqrt(2)
        Uh = (2 * sp.pi) / (sp.I * 2 * sp.pi) * (1 - sp.cosh(lam * y_) / sp.cosh(lam))   # K / (i omega), K = omega
        e = sp.expand_complex(Uh * sp.exp(sp.I * 2 * sp.pi * t_))
        U = sp.re(e)
        dU = sp.re(sp.expand_complex(sp.diff(Uh, y_) * sp.exp(sp.I * 2 * sp.pi * t_)))
        f = lambda ex: sp.lambdify((y_, t_), sp.N(ex, 20), modules=[_FN, 'math'])(ngs.y, tpar)  # noqa: E731
        return f(U), f(dU)
    k = alpha * complex(math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4))
    m = np.arange(26)
    coef = np.array([(-1) ** int(j) * (k / 2) ** (2 * int(j)) / math.factorial(int(j)) ** 2 for j in m])  # J0(k r) in r^2j
    J0k = coef.sum()
    a = (K / (1j * om)) * (-coef / J0k)
    a[0] += K / (1j * om)
    re_c, im_c = a.real, a.imag

    def poly(c, deriv=False):
        expr = ngs.CF(0.0)
        for j in range(len(c) - 1, -1, -1):          # Horner in r^2
            expr = expr * ngs.x * ngs.x + float(c[j]) * (2 * j if deriv else 1)
        return expr
    Ur, Ui = poly(re_c), poly(im_c)
    # derivative: sum 2 j c_j r^(2j - 1) = (1/r) sum 2 j c_j r^(2j); write as r * sum 2 j c_j r^(2j - 2)
    dre = np.array([2 * j * re_c[j] for j in range(1, len(re_c))])
    dim = np.array([2 * j * im_c[j] for j in range(1, len(im_c))])

    def poly2(c):
        expr = ngs.CF(0.0)
        for j in range(len(c) - 1, -1, -1):
            expr = expr * ngs.x * ngs.x + float(c[j])
        return expr * ngs.x
    dUr, dUi = poly2(dre), poly2(dim)
    c, s = ngs.cos(om * tpar), ngs.sin(om * tpar)
    return Ur * c - Ui * s, dUr * c - dUi * s


# ============================================================================================== meshes
def ng_mesh(V, T, namer):
    V = np.asarray(V, float)
    T = np.asarray(T).copy()
    P = V[T]
    d = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    T[d < 0] = T[d < 0][:, [0, 2, 1]]
    m = NGMesh(dim=2)
    pids = [m.Add(MeshPoint(Pnt(float(a), float(b), 0.0))) for a, b in V]
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    for t in T:
        m.Add(Element2D(1, [pids[int(k)] for k in t]))
    cnt, orient = {}, {}
    for t in T:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(t[i]), int(t[j])
            key = (min(a, b), max(a, b))
            cnt[key] = cnt.get(key, 0) + 1
            orient[key] = (a, b)
    names = {}
    for key, c in cnt.items():
        if c == 1:
            a, b = orient[key]
            nm = namer(V[a], V[b])
            idx = names.setdefault(nm, len(names) + 1)
            m.Add(Element1D([pids[a], pids[b]], index=idx))
    for nm, idx in names.items():
        m.SetBCName(idx - 1, nm)
    return ngs.Mesh(m)


def box_namer(box, axi=False):
    x0, x1, y0, y1 = box

    def nm(Pa, Pb, tol=1e-12):
        if abs(Pa[0] - x0) < tol and abs(Pb[0] - x0) < tol:
            return "axis" if axi else "x0"
        if abs(Pa[0] - x1) < tol and abs(Pb[0] - x1) < tol:
            return "r1" if axi else "x1"
        if abs(Pa[1] - y0) < tol and abs(Pb[1] - y0) < tol:
            return "z0" if axi else "y0"
        if abs(Pa[1] - y1) < tol and abs(Pb[1] - y1) < tol:
            return "z1" if axi else "y1"
        raise ValueError((Pa, Pb))
    return nm


class NGEval:
    """The check module's Evaluator protocol over NGSolve GridFunctions (affine elements)."""

    def __init__(self, mesh, comps, p):
        self.mesh, self.comps, self.p = mesh, comps, p
        self.n_elements = mesh.ne
        self._cache = {}

    def _pts(self, ref):
        key = ref.tobytes()
        if key not in self._cache:
            ir = ngs.IntegrationRule(points=[tuple(map(float, q)) for q in ref], weights=[1.0] * len(ref))
            self._cache[key] = self.mesh.MapToAllElements(ir, ngs.VOL)
        return self._cache[key]

    def geometry(self, ref):
        pts = self._pts(ref)
        X = ngs.CF((ngs.x, ngs.y))(pts).reshape(self.n_elements, len(ref), 2)
        J = ngs.specialcf.JacobianMatrix(2)(pts)
        det = np.abs(J[:, 0] * J[:, 3] - J[:, 1] * J[:, 2]).reshape(self.n_elements, len(ref))
        return X, det

    def velocity(self, ref):
        pts = self._pts(ref)
        u = np.stack([c(pts)[:, 0] for c in self.comps], -1).reshape(self.n_elements, len(ref), len(self.comps))
        D = np.stack([ngs.grad(c)(pts) for c in self.comps], -2).reshape(self.n_elements, len(ref), len(self.comps), 2)
        return u, D

    def pressure(self, ref):
        if self.p is None:
            return np.zeros((self.n_elements, len(ref)))
        return self.p(self._pts(ref))[:, 0].reshape(self.n_elements, len(ref))


# ============================================================================================== estimator constants
def _lte_const(order, ts):
    """LTE constant of the variable-step BDF (order) at t_n = ts[-1] for u = t^(k+1)/(k+1)! (u^(k+1) = 1): the local
    error delta = u^n - u(t_n) of one step from exact history, to leading order: alpha_0 delta / h = -tau."""
    tn = ts[-1]
    h = [ts[-1] - ts[-2]] + ([ts[-2] - ts[-3]] if order == 2 else [])
    al = C.bdf_coefficients(order, h)
    k1 = order + 1
    u = lambda s: (s - tn) ** k1 / math.factorial(k1)  # noqa: E731
    du = lambda s: (s - tn) ** order / math.factorial(order)  # noqa: E731
    tau = sum(a * u(ts[-1 - j]) for j, a in enumerate(al)) / h[0] - du(tn)
    return -h[0] * tau / al[0]


def _pred_coeffs(order, ts):
    """Lagrange extrapolation through the previous order+1 points evaluated at t_n (Milne predictor of matching order)."""
    pts = [ts[-2 - j] for j in range(order + 1)]
    tn = ts[-1]
    out = []
    for i, ti in enumerate(pts):
        c = 1.0
        for j, tj in enumerate(pts):
            if j != i:
                c *= (tn - tj) / (ti - tj)
        out.append(c)
    return out


def est_factor(order, ts):
    """est = F (u^n - u_pred): u^n - u(t_n) = L h.. u^(k+1), u(t_n) - u_pred = P u^(k+1) -> F = L / (L + P)."""
    L = _lte_const(order, ts)
    tn = ts[-1]
    k1 = order + 1
    u = lambda s: (s - tn) ** k1 / math.factorial(k1)  # noqa: E731
    P = u(tn) - sum(c * u(ts[-2 - j]) for j, c in enumerate(_pred_coeffs(order, ts)))
    return L / (L + P)


# ============================================================================================== problem
class TRProblem:
    """One transient problem on one mesh.  fld: f3a_transient_check.TField; spec: the generated JSON (for sympy)."""

    def __init__(self, mesh, fld, form="lap", bonus=4):
        self.mesh, self.fld, self.form = mesh, fld, form
        self.axi = fld.axi
        self.rho, self.mu = fld.rho, fld.mu
        self.nc = 3 if self.axi else 2
        self.tp = ngs.Parameter(0.0)
        self.bonus = bonus
        bnd = fld.data["boundaries"]
        self.dirichlet = [b["name"] for b in bnd if b["kind"] == "dirichlet"]
        self.traction = [b for b in bnd if b["kind"] == "traction"]
        dset = "|".join(self.dirichlet)
        if self.axi:
            spaces = [ngs.H1(mesh, order=2, dirichlet=dset + "|axis"), ngs.H1(mesh, order=2, dirichlet=dset + "|axis"),
                      ngs.H1(mesh, order=2, dirichlet=dset)]
        else:
            spaces = [ngs.H1(mesh, order=2, dirichlet=dset) for _ in range(self.nc)]
        self.U = spaces
        self.mean_zero = fld.data["pressure"] == "mean_zero"
        sl = spaces + [ngs.H1(mesh, order=1)] + ([ngs.NumberSpace(mesh)] if self.mean_zero else [])
        self.X = ngs.FESpace(sl)
        self.gf = ngs.GridFunction(self.X)
        self.free = self.X.FreeDofs()
        self.rw = ngs.x if self.axi else ngs.CF(1.0)
        self.rw_mass = self.rw
        self._data_cf()
        # history grid functions (velocity components) for up to 3 previous levels, and the advecting velocity W
        self.H = [[ngs.GridFunction(s) for s in spaces] for _ in range(3)]
        self.W = [ngs.GridFunction(s) for s in spaces]
        self.Pold = ngs.GridFunction(ngs.H1(mesh, order=1))
        self.alpha = [ngs.Parameter(0.0) for _ in range(3)]       # alpha_0..2 / dt
        self._build()

    # ------------------------------------------------------------------------------------ data
    def _data_cf(self):
        fld, tp = self.fld, self.tp
        cs = fld.coords[:2]
        d = fld.data
        fkey = "f"
        self.f_cf = ngs.CF(tuple(cf_from_sympy(s["sympy"], cs, tp) for s in d["forms"][self.form][fkey]))
        self.tr_cf = []
        if fld.name.startswith("f3a_tr_womersley"):
            self.tq = tp
            U, dU = womersley_cfs(fld, self.tq)
            p = cf_from_sympy(d["p"]["sympy"], cs, self.tq)
            self.u0_cf = None
            self.wom = (U, dU)
            for b in self.traction:
                n = b["normal"]
                if fld.geometry == "planar":      # x = const: (sigma n)_x = -p n_x, (sigma n)_y = sigma_yx n_x
                    tau = self.mu * dU if self.form == "sym" else ngs.CF(0.0)
                    self.tr_cf.append((b["name"], ngs.CF((-p * n[0], tau * n[0]))))
                else:                             # z = const: (sigma n)_r = sigma_rz n_z, (sigma n)_z = -p n_z
                    tau = self.mu * dU if self.form == "sym" else ngs.CF(0.0)
                    self.tr_cf.append((b["name"], ngs.CF((tau * n[1], 0.0, -p * n[1]))))
        self.u_cf = None
        if not fld.data.get("homogeneous") and not fld.data.get("data_only"):
            if fld.name.startswith("f3a_tr_womersley"):
                U = self.wom[0]
                self.u_cf = ngs.CF((U, 0.0)) if fld.geometry == "planar" else ngs.CF((0.0, 0.0, U))
            else:
                self.u_cf = ngs.CF(tuple(cf_from_sympy(s["sympy"], cs, tp) for s in d["u"]))

    def set_time(self, t):
        self.tp.Set(t)

    def set_dirichlet(self, t, gf=None):
        """P2 nodal interpolation of the exact data (homogeneous fields: 0) on the Dirichlet edges (axis: u_r =
        u_theta = 0 beats the data)."""
        gf = gf or self.gf
        mesh = self.mesh
        edges, verts = set(), set()
        names = set(self.dirichlet) | ({"axis"} if self.axi else set())
        for el in mesh.Elements(ngs.BND):
            if el.mat in names:
                for e in el.edges:
                    edges.add(e.nr)
                for v in el.vertices:
                    verts.add(v.nr)
        homog = self.fld.data.get("homogeneous")

        def g(A, B):
            if homog:
                return np.zeros((len(A), self.nc))
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.array(self.fld.u(A, B, t), float)
            if self.axi:
                v[np.abs(A) < 1e-14, 0:2] = 0.0
            return v
        vl = sorted(verts)
        V = np.array([mesh.vertices[v].point for v in vl])
        gv = g(V[:, 0], V[:, 1])
        comps = gf.components[:self.nc]
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            fd = fes.FreeDofs()
            for k, v in enumerate(vl):
                for dnr in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v)):
                    if not fd[dnr]:
                        gfc.vec[dnr] = gv[k, ci]
        el = sorted(edges)
        mids = []
        for e in el:
            ed = mesh[ngs.NodeId(ngs.EDGE, e)]
            pa, pb = (np.array(mesh[v].point) for v in ed.vertices)
            mids.append(0.5 * (pa + pb))
        mids = np.array(mids)
        gm = g(mids[:, 0], mids[:, 1])
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            fd = fes.FreeDofs()
            probe = ngs.GridFunction(fes)
            phi = None
            for k, e in enumerate(el):
                dofs = [dd for dd in fes.GetDofNrs(ngs.NodeId(ngs.EDGE, e)) if dd >= 0 and not fd[dd]]
                if not dofs:
                    continue
                xm, ym = mids[k]
                if phi is None:
                    probe.vec[:] = 0
                    probe.vec[dofs[0]] = 1.0
                    phi = probe(mesh(xm, ym))
                gfc.vec[dofs[0]] = 0.0
                cur = gfc(mesh(xm, ym))
                gfc.vec[dofs[0]] = (gm[k, ci] - cur) / phi

    # ------------------------------------------------------------------------------------ forms
    def _kin(self, comps):
        r = ngs.x
        if self.axi:
            ur, ut, uz = comps
            G = ngs.CF((ngs.grad(ur)[0], -ut / r, ngs.grad(ur)[1],
                        ngs.grad(ut)[0], ur / r, ngs.grad(ut)[1],
                        ngs.grad(uz)[0], 0, ngs.grad(uz)[1]), dims=(3, 3))
            U = ngs.CF(tuple(comps))
            div = ngs.grad(ur)[0] + ur / r + ngs.grad(uz)[1]
        else:
            ux, uy = comps
            G = ngs.CF((ngs.grad(ux)[0], ngs.grad(ux)[1], ngs.grad(uy)[0], ngs.grad(uy)[1]), dims=(2, 2))
            U = ngs.CF((ux, uy))
            div = ngs.grad(ux)[0] + ngs.grad(uy)[1]
        return U, G, div

    def _visc(self, Gu, Gv):
        if self.form == "lap":
            return self.mu * ngs.InnerProduct(Gu, Gv)
        return 2 * self.mu * ngs.InnerProduct(0.5 * (Gu + Gu.trans), 0.5 * (Gv + Gv.trans))

    def _build(self):
        X, rw = self.X, self.rw
        tr, te = X.TnT()
        nc = self.nc
        u, v = list(tr[:nc]), list(te[:nc])
        p, q = tr[nc], te[nc]
        U, G, divu = self._kin(u)
        Vt, Gt, divv = self._kin(v)
        rho_m = self.rho
        hist = [ngs.CF(tuple(h)) for h in self.H]
        dudt = self.alpha[0] * U + self.alpha[1] * hist[0] + self.alpha[2] * hist[1]
        pres = -p * divv - q * divu
        stokes = self._visc(G, Gt) + pres
        mass = rho_m * ngs.InnerProduct(dudt, Vt)
        cs_ = self.rho
        conv_full = cs_ * ngs.InnerProduct(G * U, Vt)
        Wc = ngs.CF(tuple(self.W))
        conv_semi = cs_ * ngs.InnerProduct(G * Wc, Vt)
        extra = 0
        if self.mean_zero:
            lam, mt = tr[nc + 1], te[nc + 1]
            extra = (lam * q + mt * p)
        dx = ngs.dx(bonus_intorder=self.bonus)
        dxm = dx
        self.a_newton = ngs.BilinearForm(X)
        self.a_newton += (stokes + conv_full + extra) * rw * dx + mass * self.rw_mass * dxm
        self.a_semi = ngs.BilinearForm(X)
        self.a_semi += (stokes + conv_semi + extra) * rw * dx + mass * self.rw_mass * dxm
        f = ngs.LinearForm(X)
        f += ngs.InnerProduct(self.f_cf, Vt) * rw * ngs.dx(bonus_intorder=6)
        for (bn, g) in self.tr_cf:
            f += ngs.InnerProduct(g, Vt) * rw * ngs.ds(bn, bonus_intorder=6)
        self.flin = f
        # dual norm of the residual (F2a): consistent r-weighted mass matrices, Dirichlet rows/cols removed
        md = ngs.BilinearForm(X)
        md += (ngs.InnerProduct(U, Vt) + p * q) * rw * ngs.dx(bonus_intorder=2)
        md.Assemble()
        self.dfree = ngs.BitArray(self.free)
        if self.mean_zero:
            for d in range(X.Range(nc + 1).start, X.Range(nc + 1).stop):
                self.dfree.Clear(d)
        self.Mdinv = md.mat.Inverse(self.dfree, inverse="umfpack")
        self._dv = md.mat.CreateColVector()
        self._do = md.mat.CreateColVector()
        # velocity mass (estimator norm) and the projection system
        mu_ = ngs.BilinearForm(X)
        mu_ += ngs.InnerProduct(U, Vt) * rw * ngs.dx(bonus_intorder=2)
        mu_.Assemble()
        self.Mu = mu_.mat
        proj = ngs.BilinearForm(X)
        proj += (ngs.InnerProduct(U, Vt) - p * divv - q * divu + extra) * rw * ngs.dx(bonus_intorder=2)
        proj.Assemble()
        self.Aproj = proj

    # ------------------------------------------------------------------------------------ algebra
    def dual(self, res):
        a = np.array(res.FV().NumPy(), float)
        a[~np.array(self.dfree)] = 0.0
        self._dv.FV().NumPy()[:] = a
        self._do.data = self.Mdinv * self._dv
        return float(np.sqrt(max(np.dot(a, np.array(self._do.FV().NumPy())), 0.0)))

    def residual(self, form, vec=None):
        vec = vec if vec is not None else self.gf.vec
        res = vec.CreateVector()
        form.Apply(vec, res)
        res.data -= self.flin.vec
        return res

    def lifted_residual(self, form):
        v = self.gf.vec.CreateVector()
        v.data = self.gf.vec
        a = np.array(v.FV().NumPy())
        a[np.array(self.free)] = 0.0
        v.FV().NumPy()[:] = a
        return max(self.dual(self.residual(form, v)), 1e-300)

    def vel_vec(self):
        return [np.array(c.vec.FV().NumPy()) for c in self.gf.components[:self.nc]]

    def set_vel(self, vecs, target=None):
        comps = target if target is not None else self.gf.components[:self.nc]
        for c, a in zip(comps, vecs):
            c.vec.FV().NumPy()[:] = a

    def p_vec(self):
        return np.array(self.gf.components[self.nc].vec.FV().NumPy())

    def free_only(self, vecs):
        """Zero the Dirichlet DOFs of a velocity-like vector (the local error there is zero: the data are exact)."""
        out = []
        for a, fes in zip(vecs, self.U):
            fd = np.array(fes.FreeDofs())
            b = np.array(a, float)
            b[~fd] = 0.0
            out.append(b)
        return out

    def mass_norm(self, vecs):
        g = ngs.GridFunction(self.X)
        for c, a in zip(g.components[:self.nc], vecs):
            c.vec.FV().NumPy()[:] = a
        tmp = g.vec.CreateVector()
        tmp.data = self.Mu * g.vec
        return math.sqrt(max(ngs.InnerProduct(g.vec, tmp), 0.0))

    def snapshot(self):
        """An evaluator of the current state (copies)."""
        comps = []
        for c, fes in zip(self.gf.components[:self.nc], self.U):
            g = ngs.GridFunction(fes)
            g.vec.data = c.vec
            comps.append(g)
        pg = ngs.GridFunction(self.gf.components[self.nc].space)
        pg.vec.data = self.gf.components[self.nc].vec
        return NGEval(self.mesh, comps, pg)

    def eval_of(self, vecs):
        comps = []
        for a, fes in zip(vecs, self.U):
            g = ngs.GridFunction(fes)
            g.vec.FV().NumPy()[:] = a
            comps.append(g)
        return NGEval(self.mesh, comps, None)

    # ------------------------------------------------------------------------------------ initial data
    def initial(self, t0=0.0, kind="leray"):
        """Discrete L2 (Leray) projection of u(t0) onto discretely divergence-free P2 fields with the nodal Dirichlet
        data; alternatives: the Stokes (Ritz) projection ('stokes') and, as a control, the plain nodal interpolant
        ('interp', not discretely divergence-free)."""
        self.set_time(t0)
        self.gf.vec[:] = 0
        if self.u_cf is None:
            self.set_dirichlet(t0)
            return
        if kind == "interp":
            self._nodal_everywhere(t0)          # P2 nodal interpolant everywhere: NOT discretely divergence-free
            return
        if kind == "stokes":                    # legitimate alternative: the Stokes (Ritz) projection, lap form
            return self._stokes_projection(t0)
        self.set_dirichlet(t0)
        rhs = ngs.LinearForm(self.X)
        Vt = ngs.CF(tuple(self.X.TestFunction()[:self.nc]))
        rhs += ngs.InnerProduct(self.u_cf, Vt) * self.rw * ngs.dx(bonus_intorder=8)
        rhs.Assemble()
        r = self.gf.vec.CreateVector()
        r.data = rhs.vec - self.Aproj.mat * self.gf.vec
        self.gf.vec.data += self.Aproj.mat.Inverse(self.free, inverse="umfpack") * r

    def _stokes_projection(self, t0):
        """(grad u_h, grad v) - (p, div v) - (q, div u_h) = (grad u0, grad v), nodal Dirichlet data (planar)."""
        assert not self.axi
        X, nc = self.X, self.nc
        self.set_dirichlet(t0)
        tr, te = X.TnT()
        u, v = list(tr[:nc]), list(te[:nc])
        p, q = tr[nc], te[nc]
        U, G, divu = self._kin(u)
        Vt, Gt, divv = self._kin(v)
        extra = (tr[nc + 1] * q + te[nc + 1] * p) if self.mean_zero else 0
        A = ngs.BilinearForm(X)
        A += (ngs.InnerProduct(G, Gt) - p * divv - q * divu + extra) * ngs.dx(bonus_intorder=2)
        A.Assemble()
        x, y = ngs.x, ngs.y
        G0 = ngs.CF((self.u_cf[0].Diff(x), self.u_cf[0].Diff(y), self.u_cf[1].Diff(x), self.u_cf[1].Diff(y)),
                    dims=(2, 2))
        rhs = ngs.LinearForm(X)
        rhs += ngs.InnerProduct(G0, Gt) * ngs.dx(bonus_intorder=8)
        rhs.Assemble()
        r = self.gf.vec.CreateVector()
        r.data = rhs.vec - A.mat * self.gf.vec
        self.gf.vec.data += A.mat.Inverse(self.free, inverse="umfpack") * r

    def dirichlet_defect(self, t):
        """max |u_h - g(t)| / max(max |g|, 1) over the P2 Dirichlet NODES (vertices and edge midpoints) of the current
        state (the quantity of f1a_stokes_check.dirichlet_nodal_defect), planar."""
        mesh = self.mesh
        pts = []
        for el in mesh.Elements(ngs.BND):
            if el.mat in set(self.dirichlet):
                for v in el.vertices:
                    pts.append(tuple(mesh[v].point))
                pa, pb = (np.array(mesh[v].point) for v in el.vertices)
                pts.append(tuple(0.5 * (pa + pb)))
        P = np.array(sorted(set(pts)))
        gv = np.array(self.fld.u(P[:, 0], P[:, 1], t), float)
        vals = np.array([[c(mesh(x, y)) for c in self.gf.components[:self.nc]] for x, y in P])
        return float(np.max(np.abs(vals - gv)) / max(float(np.max(np.abs(gv))), 1.0))

    def _nodal_everywhere(self, t):
        mesh = self.mesh
        V = np.array([v.point for v in mesh.vertices])
        uv = np.array(self.fld.u(V[:, 0], V[:, 1], t), float)
        comps = self.gf.components[:self.nc]
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            gfc.vec[:] = 0
            for k in range(len(V)):
                for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, k)):
                    gfc.vec[d] = uv[k, ci]
        mids, enrs = [], []
        for e in mesh.edges:
            a, b = e.vertices
            mids.append(0.5 * (np.array(mesh[a].point) + np.array(mesh[b].point)))
            enrs.append(e.nr)
        mids = np.array(mids)
        um = np.array(self.fld.u(mids[:, 0], mids[:, 1], t), float)
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            probe = ngs.GridFunction(fes)
            phi = None
            for k, e in enumerate(enrs):
                dofs = [d for d in fes.GetDofNrs(ngs.NodeId(ngs.EDGE, e)) if d >= 0]
                xm, ym = mids[k]
                if phi is None:
                    probe.vec[:] = 0
                    probe.vec[dofs[0]] = 1.0
                    phi = probe(mesh(xm, ym))
                gfc.vec[dofs[0]] = 0.0
                cur = gfc(mesh(xm, ym))
                gfc.vec[dofs[0]] = (um[k, ci] - cur) / phi
        if self.axi:
            self.set_dirichlet(t)

    # ------------------------------------------------------------------------------------ one step
    def step(self, hist_vecs, ts, order, scheme, newton_rtol=1e-12, maxit=25, guess=None, pold=None):
        """Advance to t_n = ts[-1] from the velocity history hist_vecs = [u^{n-1}, u^{n-2}, ...] at ts[-2], ts[-3], ...
        with BDF(order) on the actual step sizes.  Returns diagnostics; the state is in self.gf."""
        dts = [ts[-1] - ts[-2]] + ([ts[-2] - ts[-3]] if order == 2 else [])
        al = C.bdf_coefficients(order, dts)
        for j in range(3):
            self.alpha[j].Set(al[j] / dts[0] if j < len(al) else 0.0)
        for j, hv in enumerate(hist_vecs[:2]):
            self.set_vel(hv, self.H[j])
        if pold is not None:
            self.Pold.vec.FV().NumPy()[:] = pold
        self.set_time(ts[-1])
        self.flin.Assemble()
        # initial guess: extrapolation (Newton) / history
        if guess is not None:
            self.set_vel(guess)
        else:
            self.set_vel(hist_vecs[0])
        self.set_dirichlet(ts[-1])
        if scheme == "semi":
            if order == 1 or len(hist_vecs) < 2:
                be = [1.0]
            else:
                be = C.extrapolation_coefficients(2, [dts[0], ts[-2] - ts[-3]])
            wv = [sum(b * hist_vecs[j][i] for j, b in enumerate(be)) for i in range(self.nc)]
            self.set_vel(wv, self.W)
            self.a_semi.AssembleLinearization(self.gf.vec)     # affine in the unknowns: exact
            r0 = self.lifted_residual(self.a_semi)
            res = self.residual(self.a_semi)
            self.gf.vec.data -= self.a_semi.mat.Inverse(self.free, inverse="umfpack") * res
            rr = self.dual(self.residual(self.a_semi)) / r0
            return {"converged": bool(rr <= 1e-10), "relative_residual": rr, "iterations": 1}
        r0 = self.lifted_residual(self.a_newton)
        hist = []
        its = 0
        for it in range(maxit):
            res = self.residual(self.a_newton)
            rn = self.dual(res) / r0
            hist.append(rn)
            if rn <= newton_rtol:
                break
            self.a_newton.AssembleLinearization(self.gf.vec)
            self.gf.vec.data -= self.a_newton.mat.Inverse(self.free, inverse="umfpack") * res
            its += 1
        rn = self.dual(self.residual(self.a_newton)) / r0
        return {"converged": bool(rn <= 10 * newton_rtol), "relative_residual": rn, "iterations": its,
                "history": hist}


# ============================================================================================== drivers
def run_fixed(prob, ts, order, scheme, init="leray", store="all", newton_rtol=1e-12, starter="bdf1",
              exact_start=False, callback=None):
    """Integrate on the given grid ts; BDF2 starts with one BDF1 step (starter 'bdf1') or with u^1 exact
    (exact_start).  Returns (evals, ts, diag)."""
    t0 = time.time()
    prob.initial(ts[0], init)
    vecs = [prob.vel_vec()]
    pvecs = [prob.p_vec()]
    evals = [prob.snapshot()] if store == "all" else []
    if callback is not None:
        callback(0, ts[0], prob)
    worst, conv, its = 0.0, True, 0
    for n in range(1, len(ts)):
        k = order
        if order == 2 and n == 1:
            k = 1
            if exact_start:
                prob.gf.vec[:] = 0
                prob._nodal_everywhere(ts[1])
                d = {"converged": True, "relative_residual": 0.0, "iterations": 0}
                vecs.insert(0, prob.vel_vec())
                pvecs.insert(0, prob.p_vec())
                if store == "all":
                    evals.append(prob.snapshot())
                if callback is not None:
                    callback(n, ts[n], prob)
                continue
        hv = vecs[:2]
        tt = list(ts[max(0, n - 2):n + 1])
        guess = None
        if scheme == "newton" and len(vecs) >= 2:
            guess = [2 * a - b for a, b in zip(vecs[0], vecs[1])]
        d = prob.step(hv, tt, k, scheme, newton_rtol=newton_rtol, guess=guess, pold=pvecs[0])
        worst = max(worst, d["relative_residual"])
        conv = conv and d["converged"]
        its += d["iterations"]
        vecs.insert(0, prob.vel_vec())
        pvecs.insert(0, prob.p_vec())
        vecs = vecs[:3]
        pvecs = pvecs[:3]
        if store == "all":
            evals.append(prob.snapshot())
        if callback is not None:
            callback(n, ts[n], prob)
    if store != "all":
        evals = [prob.snapshot()]
    return evals, {"converged": conv, "relative_residual": worst, "iterations": its, "seconds": time.time() - t0,
                   "dt_stats": C.step_stats(ts)}


def run_adaptive(prob, T, rtol, S, h0, order_max=2, ratio_max=2.0, safety=0.9, store=False, newton_rtol=1e-12,
                 all_dofs=False):
    """Adaptive BDF1/2: steps 1 and 2 by BDF1 with step doubling (estimate = two-halves minus one-step), then BDF2
    with the quadratic-extrapolation Milne estimate est = F (u^n - u_pred).  Error per step in the (r-weighted) L2
    mass norm: accept iff ||est|| <= rtol S.  h_new = h min(ratio_max, max(0.2, safety err^(-1/(k+1)))).
    Variants: safety / ratio_max, all_dofs (the Milne estimate NOT restricted to the free DOFs)."""
    t0 = time.time()
    prob.initial(0.0)
    vecs = [prob.vel_vec()]
    pv = prob.p_vec()
    ts = [0.0]
    h = h0
    rej, worst, conv, nsteps = 0, 0.0, True, 0
    tol = rtol * S
    free_only = (lambda v: v) if all_dofs else prob.free_only
    rmax = ratio_max
    while ts[-1] < T - 1e-14:
        h = min(h, T - ts[-1])
        if len(ts) < 3:
            tn = ts[-1] + h
            d1 = prob.step([vecs[0]], [ts[-1], tn], 1, "newton", newton_rtol)
            full = prob.vel_vec()
            tm = ts[-1] + h / 2
            prob.step([vecs[0]], [ts[-1], tm], 1, "newton", newton_rtol)
            half = prob.vel_vec()
            d2 = prob.step([half], [tm, tn], 1, "newton", newton_rtol)
            est = free_only([(a - b) for a, b in zip(prob.vel_vec(), full)])
            k = 1
            worst = max(worst, d1["relative_residual"], d2["relative_residual"])
        else:
            tt = ts[-3:] + [ts[-1] + h]
            d2 = prob.step(vecs[:2], tt, 2, "newton", newton_rtol, guess=[2 * a - b for a, b in zip(vecs[0], vecs[1])])
            worst = max(worst, d2["relative_residual"])
            tpred = ts[-3:] + [tt[-1]]
            pc = _pred_coeffs(2, tpred)
            pred = [sum(c * vecs[j][i] for j, c in enumerate(pc)) for i in range(prob.nc)]
            F = est_factor(2, tpred)
            est = free_only([F * (a - b) for a, b in zip(prob.vel_vec(), pred)])
            k = 2
        err = prob.mass_norm(est) / tol
        conv = conv and d2["converged"]
        if err <= 1.0:
            if k == 1:
                vecs.insert(0, half)
                ts.append(ts[-1] + h / 2)
            vecs.insert(0, prob.vel_vec())
            pv = prob.p_vec()
            ts.append(ts[-1] + h if k == 2 else ts[-1] + h / 2)
            vecs = vecs[:3]
            nsteps += 1
            if k == 1:
                h = h / 2          # the accepted steps were h/2: ratios refer to the last step taken
            fac = min(rmax, max(0.2, safety * max(err, 1e-10) ** (-1.0 / (k + 1))))
        else:
            rej += 1
            fac = max(0.2, safety * err ** (-1.0 / (k + 1)))
        h = h * fac
        if nsteps > 20000:
            raise RuntimeError("too many steps")
    return prob.snapshot(), {"converged": conv, "relative_residual": worst, "rejected": rej, "accepted": nsteps,
                             "ts": ts, "dt_stats": C.step_stats(ts), "seconds": time.time() - t0}


def single_step_effectivity(prob, ts, order, all_dofs=False):
    """One BDF(order) step from the EXACT history (nodal interpolants of the space-exact field at ts[:-1]); returns
    the estimate (as for run_adaptive), the advanced velocity evaluators and the norm of the estimate."""
    hv = []
    for tj in reversed(ts[:-1]):
        prob.gf.vec[:] = 0
        prob._nodal_everywhere(tj)
        hv.append(prob.vel_vec())
    # hv = [u(t_{n-1}), u(t_{n-2}), ...]
    prob.step(hv[:2], list(ts[-(order + 1):]), order, "newton", 1e-13)
    un = prob.vel_vec()
    pc = _pred_coeffs(order, ts)
    pred = [sum(c * hv[j][i] for j, c in enumerate(pc)) for i in range(prob.nc)]
    F = est_factor(order, ts)
    est = (lambda v: v)([F * (a - b) for a, b in zip(un, pred)]) if all_dofs else \
        prob.free_only([F * (a - b) for a, b in zip(un, pred)])
    # the scalar the solver would REPORT as the norm of its estimate: the (r-weighted) L2 mass norm
    nrm = prob.mass_norm(est)
    return prob.eval_of(est), prob.eval_of(un), nrm


def mesh_for(fld, fam, level):
    V, T, h = C.family_mesh(fam, level)
    return ng_mesh(V, T, box_namer(fld.data["box"], fld.axi)), V, T, h
