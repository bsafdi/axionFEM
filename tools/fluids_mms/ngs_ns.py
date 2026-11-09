"""Independent NGSolve reference solver for steady Navier-Stokes.

NGSolve 6.2.2608, Taylor-Hood P2/P1, written from the continuum equations (never
from axionFEM code).  Planar and axisymmetric with swirl; both viscous forms; standard and skew (Temam) convection;
Picard (Oseen) start, then Newton with NGSolve's symbolic linearization (or a hand-written Jacobian, used by the
pseudo-transient continuation), Armijo line search on the residual norm, Re continuation and pseudo-transient
continuation (SER).
Dirichlet data by P2 LAGRANGE NODAL INTERPOLATION (vertex values, edge bubbles set so the midpoint value is exact).  Error norms are computed by the fixed check module f2a_ns_check through an
Evaluator adapter (so the check module is also exercised on real discrete solutions).
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
import f2a_ns_check as C  # noqa: E402

_FN = {'sin': ngs.sin, 'cos': ngs.cos, 'exp': ngs.exp, 'sinh': ngs.sinh, 'cosh': ngs.cosh, 'sqrt': ngs.sqrt,
       'log': ngs.log, 'pi': math.pi, 'atan2': ngs.atan2}


def cf_from_sympy(s: str, coords):
    syms = sp.symbols(" ".join(coords), real=True)
    e = sp.sympify(s, locals={str(v): v for v in syms})
    if not e.free_symbols:
        return ngs.CF(float(e))
    f = sp.lambdify(syms, e, modules=[_FN, 'math'])
    return f(ngs.x, ngs.y)


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


def box_namer(x0, x1, y0, y1, axi=False):
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


# ============================================================================================== evaluator for the check module
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
        return self.p(self._pts(ref))[:, 0].reshape(self.n_elements, len(ref))


# ============================================================================================== problem
class NSProblem:
    """geometry 'planar' | 'axisymmetric'; data: an f2a_ns_check.Field (or None for the cavity); dirichlet: names of
    the Dirichlet boundaries; g: callable(A, B) -> [n, ncomp] Dirichlet data (default: the field's exact u)."""

    def __init__(self, mesh, geometry, mu_cf, rho, f_cf, dirichlet, g, form="lap", conv="standard", mean_zero=True,
                 traction=None, mu_np=None, bonus=4):
        self.mesh, self.geometry, self.form, self.conv = mesh, geometry, form, conv
        self.axi = geometry == "axisymmetric"
        self.rho, self.mu = rho, mu_cf
        self.mu_np = mu_np
        self.bonus = bonus
        nc = 3 if self.axi else 2
        self.nc = nc
        dset = "|".join(dirichlet)
        if self.axi:
            ax = dset + "|axis"
            spaces = [ngs.H1(mesh, order=2, dirichlet=ax), ngs.H1(mesh, order=2, dirichlet=dset + "|axis"),
                      ngs.H1(mesh, order=2, dirichlet=dset)]
        else:
            spaces = [ngs.H1(mesh, order=2, dirichlet=dset) for _ in range(nc)]
        self.U = spaces
        Q = ngs.H1(mesh, order=1)
        sl = spaces + [Q] + ([ngs.NumberSpace(mesh)] if mean_zero else [])
        self.X = ngs.FESpace(sl)
        self.mean_zero = mean_zero
        self.gf = ngs.GridFunction(self.X)
        self.r = ngs.x
        self.rw = ngs.x if self.axi else ngs.CF(1.0)
        self.f_cf = f_cf
        self.dirichlet = dirichlet
        self.g = g
        self.traction = traction or []
        self.nodal_defect = self._set_dirichlet()
        self.free = self.X.FreeDofs()
        self._build()

    # ------------------------------------------------------------------------------------ Dirichlet data
    def _set_dirichlet(self):
        mesh = self.mesh
        edges, verts = set(), set()
        for el in mesh.Elements(ngs.BND):
            if el.mat in self.dirichlet:
                for e in el.edges:
                    edges.add(e.nr)
                for v in el.vertices:
                    verts.add(v.nr)
        V = np.array([mesh.vertices[v].point for v in sorted(verts)]) if verts else np.zeros((0, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            gv = self.g(V[:, 0], V[:, 1]) if len(V) else None
        if self.axi and len(V):
            # shared-node rule: at an axis node the axis condition u_r = u_t = 0 holds (the data agree in the limit)
            on_axis = np.abs(V[:, 0]) < 1e-14
            gv = np.array(gv, float)
            gv[on_axis, 0:2] = 0.0
            if not np.all(np.isfinite(gv)):
                raise RuntimeError("non-finite Dirichlet data")
        comps = self.gf.components[:self.nc]
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            for k, v in enumerate(sorted(verts)):
                for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v)):
                    gfc.vec[d] = gv[k, ci]
        # edge bubbles: midpoint value = data (probe the bubble's midpoint value once per space)
        mids, edofs = [], []
        for e in sorted(edges):
            ed = mesh[ngs.NodeId(ngs.EDGE, e)]
            pa, pb = (np.array(mesh[v].point) for v in ed.vertices)
            mids.append(0.5 * (pa + pb))
            edofs.append(e)
        mids = np.array(mids) if mids else np.zeros((0, 2))
        gm = self.g(mids[:, 0], mids[:, 1]) if len(mids) else None
        worst = 0.0
        for ci, (gfc, fes) in enumerate(zip(comps, self.U)):
            probe = ngs.GridFunction(fes)
            phi = None
            for k, e in enumerate(edofs):
                dofs = [d for d in fes.GetDofNrs(ngs.NodeId(ngs.EDGE, e)) if d >= 0]
                if not dofs:
                    continue
                xm, ym = mids[k]
                if phi is None:
                    probe.vec[:] = 0
                    probe.vec[dofs[0]] = 1.0
                    phi = probe(mesh(xm, ym))
                cur = gfc(mesh(xm, ym))
                gfc.vec[dofs[0]] += (gm[k, ci] - cur) / phi
            for k in range(len(mids)):
                worst = max(worst, abs(gfc(mesh(*mids[k])) - gm[k, ci]))
        return worst

    # ------------------------------------------------------------------------------------ forms
    def _kin(self, comps):
        r = self.r
        if self.axi:
            ur, ut, uz = comps
            G = ngs.CF((ngs.grad(ur)[0], -ut / r, ngs.grad(ur)[1],
                        ngs.grad(ut)[0], ur / r, ngs.grad(ut)[1],
                        ngs.grad(uz)[0], 0, ngs.grad(uz)[1]), dims=(3, 3))
            Gv = G
            U = ngs.CF(tuple(comps))
            div = ngs.grad(ur)[0] + ur / r + ngs.grad(uz)[1]
        else:
            ux, uy = comps
            G = ngs.CF((ngs.grad(ux)[0], ngs.grad(ux)[1], ngs.grad(uy)[0], ngs.grad(uy)[1]), dims=(2, 2))
            Gv = G
            U = ngs.CF((ux, uy))
            div = ngs.grad(ux)[0] + ngs.grad(uy)[1]
        return U, G, Gv, div

    def _visc(self, Gu, Gv):
        if self.form == "lap":
            return self.mu * ngs.InnerProduct(Gu, Gv)
        return 2 * self.mu * ngs.InnerProduct(0.5 * (Gu + Gu.trans), 0.5 * (Gv + Gv.trans))

    def _conv(self, W, GW, divW, U, G, V):
        """c(W; U, V) with (W.grad)U = G_U W in the orthonormal frame (the hoop terms are part of G)."""
        e = ngs.InnerProduct(G * W, V)
        if self.conv == "skew":
            e = e + 0.5 * divW * ngs.InnerProduct(U, V)
        return self.rho * e

    def _build(self):
        X, mesh, rw = self.X, self.mesh, self.rw
        tr, te = X.TnT()
        nc = self.nc
        u, v = list(tr[:nc]), list(te[:nc])
        p, q = tr[nc], te[nc]
        U, G, Gv, divu = self._kin(u)
        Vt, Gt, Gvt, divv = self._kin(v)
        bulk = self._visc(Gv, Gvt) - p * divv - q * divu
        if self.mean_zero:
            lam, mt = tr[nc + 1], te[nc + 1]
            bulk = bulk + (lam * q + mt * p)
        self.tr, self.te = tr, te
        a = ngs.BilinearForm(X)
        a += (bulk + self._conv(U, G, divu, U, G, Vt)) * rw * ngs.dx(bonus_intorder=self.bonus)
        self.a = a
        f = ngs.LinearForm(X)
        if self.f_cf is not None:
            f += ngs.InnerProduct(self.f_cf, Vt) * rw * ngs.dx(bonus_intorder=6)
        for (bn, g) in self.traction:
            f += ngs.InnerProduct(g, Vt) * rw * ngs.ds(bn, bonus_intorder=6)
        f.Assemble()
        self.f = f
        # Stokes and Oseen (Picard) forms with a fixed advecting velocity W
        self.W = [ngs.GridFunction(s_) for s_ in self.U]
        Wc = ngs.CF(tuple(self.W))
        _, GW, _, divW = self._kin(self.W)
        ao = ngs.BilinearForm(X)
        ao += (bulk + self._conv(Wc, GW, divW, U, G, Vt)) * rw * ngs.dx(bonus_intorder=self.bonus)
        self.ao = ao
        # velocity mass matrix (pseudo-transient continuation)
        m = ngs.BilinearForm(X)
        m += ngs.InnerProduct(U, Vt) * rw * ngs.dx
        m.Assemble()
        self.M = m
        # discrete DUAL norm of the residual functional: ||R||_* = (R_u M_u^-1 R_u + R_p M_p^-1 R_p)^1/2
        # over the free velocity DOFs and all pressure DOFs (the mean multiplier excluded), M the (r-weighted) L2 mass
        # matrices: the same number in every basis of the same discrete spaces
        md = ngs.BilinearForm(X)
        md += (ngs.InnerProduct(U, Vt) + p * q) * rw * ngs.dx(bonus_intorder=2)
        md.Assemble()
        self.dfree = ngs.BitArray(self.X.FreeDofs())
        if self.mean_zero:
            for d in range(self.X.Range(self.nc + 1).start, self.X.Range(self.nc + 1).stop):
                self.dfree.Clear(d)
        self.Mdinv = md.mat.Inverse(self.dfree, inverse="umfpack")
        self._dvec = md.mat.CreateColVector()
        self._dout = md.mat.CreateColVector()
        self.norm_kind = "dual"
        # hand-written Newton Jacobian (used by the pseudo-transient continuation); equals the symbolic one
        aj = ngs.BilinearForm(X)
        rho = self.rho
        cj = ngs.InnerProduct(G * Wc, Vt)                         # (W.grad) du
        cj = cj + ngs.InnerProduct(GW * U, Vt)                    # (du.grad) W
        if self.conv == "skew":
            cj = cj + 0.5 * divW * ngs.InnerProduct(U, Vt)
            cj = cj + 0.5 * divu * ngs.InnerProduct(Wc, Vt)
        self.dtinv = ngs.Parameter(0.0)                           # pseudo-transient term (1/dt) (du, v)
        aj += (bulk + rho * cj + self.dtinv * ngs.InnerProduct(U, Vt)) * rw * ngs.dx(bonus_intorder=self.bonus)
        self.aj = aj

    # ------------------------------------------------------------------------------------ algebra
    def residual(self, vec=None):
        vec = vec if vec is not None else self.gf.vec
        res = vec.CreateVector()
        self.a.Apply(vec, res)
        res.data -= self.f.vec
        arr = np.array(res.FV().NumPy())
        arr[~np.array(self.free)] = 0.0
        return arr, res

    def _norm(self, arr):
        if self.norm_kind == "l2":
            return float(np.linalg.norm(arr))
        a = np.array(arr, float)
        a[~np.array(self.dfree)] = 0.0
        self._dvec.FV().NumPy()[:] = a
        self._dout.data = self.Mdinv * self._dvec
        return float(np.sqrt(max(np.dot(a, np.array(self._dout.FV().NumPy())), 0.0)))

    def _setW(self):
        for k in range(self.nc):
            self.W[k].vec.data = self.gf.components[k].vec

    def picard(self, n, stokes_first=True, history=None):
        """n Oseen steps (the first with W = 0 is the Stokes solve if stokes_first)."""
        for it in range(n):
            if it == 0 and stokes_first:
                for w in self.W:
                    w.vec[:] = 0
            else:
                self._setW()
            self.ao.Assemble()
            rhs = self.gf.vec.CreateVector()
            rhs.data = self.f.vec - self.ao.mat * self.gf.vec
            self.gf.vec.data += self.ao.mat.Inverse(self.free, inverse="umfpack") * rhs
            if history is not None:
                history.append({"kind": "picard", "residual": self._norm(self.residual()[0])})

    def jacobian(self, symbolic=True, dt=None):
        if dt is None and symbolic:
            self.a.AssembleLinearization(self.gf.vec)
            return self.a.mat
        self.dtinv.Set(0.0 if dt is None else 1.0 / dt)
        self._setW()
        self.aj.Assemble()
        return self.aj.mat

    def newton(self, rtol=1e-11, maxit=30, line_search=True, r0=None, history=None, symbolic=True, ptc_dt=None,
               ptc_dt_max=1e12):
        """Newton (or pseudo-transient continuation with SER if ptc_dt) on the STEADY residual; residual norms are
        relative to r0 (default: the residual of the lifted zero state).  PTC: (J + M/dt) du =
        R, dt_{k+1} = dt_k r_k / r_{k+1}; once dt > ptc_dt_max the iteration is plain Newton."""
        hist = history if history is not None else []
        r0 = r0 or self.r0()
        arr, res = self.residual()
        rn = self._norm(arr) / r0
        hist.append({"kind": "start", "residual": rn, "step": None})
        dt = ptc_dt
        for it in range(maxit):
            if rn <= rtol:
                break
            J = self.jacobian(symbolic, dt)
            du = self.gf.vec.CreateVector()
            du.data = J.Inverse(self.free, inverse="umfpack") * res
            lam = 1.0
            base = self.gf.vec.CreateVector()
            base.data = self.gf.vec
            while True:
                self.gf.vec.data = base - lam * du
                arr, res = self.residual()
                rnew = self._norm(arr) / r0
                if not line_search or dt is not None or rnew <= (1 - 1e-4 * lam) * rn or lam <= 1 / 64:
                    break           # minimum step 1/64 (accepted even without sufficient decrease)
                lam *= 0.5
            hist.append({"kind": "ptc" if dt is not None else "newton", "residual": rnew, "step": lam, "dt": dt})
            if dt is not None:
                dt = dt * rn / max(rnew, 1e-300)
                if dt > ptc_dt_max:
                    dt = None
            rn = rnew
            if not math.isfinite(rn) or rn > 1e8:
                break
        return hist

    def r0(self):
        """Residual norm of the lifted zero state: Dirichlet data on the boundary nodes, zero elsewhere, p = 0."""
        v = self.gf.vec.CreateVector()
        v.data = self.gf.vec
        fr = np.array(self.free)
        a = np.array(v.FV().NumPy())
        a[fr] = 0.0
        v.FV().NumPy()[:] = a
        arr, _ = self.residual(v)
        return max(self._norm(arr), 1e-300)

    def evaluator(self):
        comps = list(self.gf.components[:self.nc])
        return NGEval(self.mesh, comps, self.gf.components[self.nc])

    def ndof(self):
        return int(sum(self.free))


def solve_field(fld, mesh, form, conv, dirichlet, traction_edges=(), n_picard=3, newton_from="picard",
                rtol=1e-11, line_search=True, symbolic=True):
    """Set up and solve an MMS / exact-solution problem; return (problem, history, seconds)."""
    t0 = time.time()
    cc = fld.coords
    mu_cf = cf_from_sympy(fld.data["mu"]["sympy"], cc)
    fdat = fld.data["forms"][form]
    key = "f"
    f_cf = ngs.CF(tuple(cf_from_sympy(s["sympy"], cc) for s in fdat[key])) if not fld.data.get("exact_ns") else None
    tr = []
    for e in traction_edges:
        b = next(b for b in fld.data["boundaries"] if b["name"] == e)
        tr.append((e, ngs.CF(tuple(cf_from_sympy(s["sympy"], cc) for s in b["traction"][form]))))
    prob = NSProblem(mesh, fld.geometry, mu_cf, fld.rho, f_cf, dirichlet, fld.u, form=form, conv=conv,
                     mean_zero=(fld.data["pressure"] == "mean_zero"), traction=tr)
    hist = []
    r0 = prob.r0()
    if newton_from == "picard":
        prob.picard(n_picard, history=hist)
    elif newton_from == "stokes":
        prob.picard(1, history=hist)
    for h in hist:
        h["residual"] /= r0
    prob.newton(rtol=rtol, r0=r0, history=hist, line_search=line_search, symbolic=symbolic)
    return prob, hist, time.time() - t0
