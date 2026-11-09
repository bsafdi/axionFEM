"""Independent NGSolve reference solver: equal-order P1/P1 GLS (SUPG + PSPG + LSIC) incompressible
flow, planar (written from the formulation below, never from axionFEM code).

NGSolve.  Unknowns (u_x, u_y, p) continuous P1 on triangles.  The discrete problem is
exactly the one stated below (and implemented a second, numpy way in f5_gls_check.residual):

   B(v, q; u, p) = (rho (u_t + (w.grad) u), v) + a_form(mu; u, v) - (p, div v) + (q, div u) - (f, v)
                 + sum_K [ (tau_m rho (w.grad) v, R) + (tau_m / rho grad q, R) + (tau_c rho div u, div v) ]_K,
   R = rho (u_t + (w.grad) u) + grad p - div sigma_visc(u; mu) - f   (second derivatives of a P1 u vanish, grad mu KEPT),
   tau_m = [ (2/dt)^2 + CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)  (SUPG only),
   tau_p = [ CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)  (PSPG and LSIC: NO dt term),  tau_c = 1 / (tau_p tr G),
   G = (dz/dx)^T (dz/dx), z the coordinates of the EQUILATERAL reference triangle of edge 1 (permutation invariant),   C_I = 36, CG = 1,
   every element integral by the degree-4 6-point rule f5_gls_check.RULE_DEG (fixed).

Newton uses NGSolve's symbolic linearization of the full stabilized residual (tau differentiated with the state); a
fixed-tau Jacobian (tau evaluated at a copy of the state) is available as a quasi-Newton variant.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import ngsolve as ngs  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.path.insert(0, str(HERE))
import f5_gls_check as C  # noqa: E402
from ngs_ns import ng_mesh, box_namer, NGEval, cf_from_sympy  # noqa: E402,F401


def rule():
    pts, w = C.rule_points()
    return ngs.IntegrationRule(points=[tuple(map(float, p)) for p in pts], weights=[float(x) for x in w])


class GLS:
    def __init__(self, mesh, rho, mu_cf, gradmu_cf, f_cf, dirichlet, g, form="sym", convection=True, mean_zero=True,
                 stab=("supg", "pspg", "lsic"), CI=36.0, CG=1.0, tau="metric", tauc="g", dt=None, bdf=1,
                 advect=None, nonlinear_s=None, tau_state=None):
        """mu_cf, gradmu_cf: CoefficientFunctions (gradmu a 2-vector CF); f_cf a 2-vector CF (may contain Parameters);
        g(x, y) -> [n, 2] Dirichlet data (P1 nodal interpolation); advect: None (the state u) or a 2-vector CF (Oseen);
        dt: None (steady) or the step (then the BDF history GridFunctions uh1, uh2 must be set by the caller)."""
        self.mesh, self.rho, self.form = mesh, float(rho), form
        self.stab = set(stab)
        self.CI, self.CG, self.tau_kind, self.tauc_kind = CI, CG, tau, tauc
        self.dt, self.bdf = dt, bdf
        self.convection = convection
        dset = "|".join(dirichlet)
        self.U = [ngs.H1(mesh, order=1, dirichlet=dset) for _ in range(2)]
        Q = ngs.H1(mesh, order=1)
        self.mean_zero = mean_zero
        self.X = ngs.FESpace(self.U + [Q] + ([ngs.NumberSpace(mesh)] if mean_zero else []))
        self.gf = ngs.GridFunction(self.X)
        self.dirichlet, self.g = dirichlet, g
        self._set_dirichlet()
        self.free = self.X.FreeDofs()
        self.uh1 = [ngs.GridFunction(s) for s in self.U]      # BDF history u^{n-1}, u^{n-2}
        self.uh2 = [ngs.GridFunction(s) for s in self.U]
        self.mu, self.gmu, self.f = mu_cf, gradmu_cf, f_cf
        self.advect = advect
        self.tau_state = tau_state
        self.ir = rule()
        self._build()

    # ---------------------------------------------------------------- data
    def _set_dirichlet(self):
        mesh = self.mesh
        verts = set()
        for el in mesh.Elements(ngs.BND):
            if el.mat in self.dirichlet:
                for v in el.vertices:
                    verts.add(v.nr)
        verts = sorted(verts)
        V = np.array([mesh.vertices[v].point for v in verts])
        gv = self.g(V[:, 0], V[:, 1])
        for ci in range(2):
            for k, v in enumerate(verts):
                for d in self.U[ci].GetDofNrs(ngs.NodeId(ngs.VERTEX, v)):
                    self.gf.components[ci].vec[d] = gv[k, ci]

    # ---------------------------------------------------------------- the form
    def _fields(self, U, P, V, Q):
        """Symbols: U trial velocity CF, V test, P trial pressure, Q test pressure -> integrand."""
        ux, uy = U
        vx, vy = V
        G = ngs.CF((ngs.grad(ux)[0], ngs.grad(ux)[1], ngs.grad(uy)[0], ngs.grad(uy)[1]), dims=(2, 2))
        Gv = ngs.CF((ngs.grad(vx)[0], ngs.grad(vx)[1], ngs.grad(vy)[0], ngs.grad(vy)[1]), dims=(2, 2))
        Uc, Vc = ngs.CF((ux, uy)), ngs.CF((vx, vy))
        divu = ngs.grad(ux)[0] + ngs.grad(uy)[1]
        divv = ngs.grad(vx)[0] + ngs.grad(vy)[1]
        rho, mu = self.rho, self.mu
        W = (Uc if self.convection else ngs.CF((0.0, 0.0))) if self.advect is None else self.advect   # Stokes: w = 0 (no SUPG, tau diffusive)
        if self.advect is not None:
            ax = self.advect
            Gw = None
        conv = (G * W) if self.convection else ngs.CF((0.0, 0.0))
        # time derivative (BDF)
        if self.dt is None:
            ut = ngs.CF((0.0, 0.0))
        else:
            if self.bdf == 1:
                a0 = 1.0 / self.dt
                ut = a0 * Uc - a0 * ngs.CF((self.uh1[0], self.uh1[1]))
            else:
                ut = (1.5 * Uc - 2.0 * ngs.CF((self.uh1[0], self.uh1[1])) + 0.5 * ngs.CF((self.uh2[0], self.uh2[1]))) / self.dt
        if self.form == "lap":
            visc = mu * ngs.InnerProduct(G, Gv)
            sdiv = G * self.gmu                      # sum_j d_j mu d_j u_i
        else:
            visc = mu * ngs.InnerProduct(G + G.trans, Gv)
            sdiv = (G + G.trans) * self.gmu
        gal = rho * ngs.InnerProduct(ut + conv, Vc) + visc - P * divv + Q * divu
        # strong residual
        Rm = rho * (ut + conv) + ngs.CF((ngs.grad(P)[0], ngs.grad(P)[1])) - sdiv - self.f
        # tau
        J = ngs.specialcf.JacobianMatrix(2)
        Ji = ngs.Inv(J)
        Mq = ngs.CF((1.0, 0.5, 0.5, 1.0), dims=(2, 2))      # R^T R, R: unit right triangle -> equilateral (edge 1)
        Gm = Ji.trans * Mq * Ji
        GG = ngs.InnerProduct(Gm, Gm)
        gg = Gm[0, 0] + Gm[1, 1]                            # tr G (permutation invariant, unlike g.g)
        Wt = W if self.tau_state is None else self.tau_state
        nu = mu / rho
        CI = self.CI
        wGw = ngs.InnerProduct(Wt, Gm * Wt)
        if self.tau_kind == "ugn":
            # Tezduyar-Osawa 2000 UGN: h_UGN = 2|u| / sum_a |u . grad N_a| (the three P1 gradients), tau = (tau1^-2 + tau2^-2 + tau3^-2)^-1/2
            # tau1 = h/(2|u|) -> 1/tau1^2 = (sum_a |u.grad N_a|)^2 ; tau3 = h^2/(4 nu), h = h_UGN
            gN = [Ji.trans * ngs.CF(v) for v in ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))]
            sa = sum(ngs.sqrt(ngs.InnerProduct(W, gk) ** 2 + 1e-30) for gk in gN)
            inv2 = sa * sa
            wn = ngs.sqrt(ngs.InnerProduct(W, W) + 1e-30)
            hU = 2 * wn / sa
            inv3 = (4 * nu / (hU * hU)) ** 2
            t2 = 0.0 if self.dt is None else (2.0 / self.dt) ** 2
            tm = 1.0 / ngs.sqrt(inv2 + inv3 + t2)
            tc = tm * 0 + (wn * hU / 2) * ngs.IfPos(wn * hU / (2 * nu) - 3, 1.0, wn * hU / (6 * nu))   # tau_LSIC
            tcval = tc
        else:
            terms = self.CG * wGw + CI * nu * nu * GG
            tm_s = 1.0 / ngs.sqrt(terms)                    # steady tau (no dt term): PSPG and LSIC
            if self.dt is not None:
                terms = terms + (2.0 / self.dt) ** 2
            tm = 1.0 / ngs.sqrt(terms)                      # SUPG tau (with the (2/dt)^2 term)
            tc = 1.0 / (tm_s * gg)
            self._tm_s = tm_s
        if self.tau_kind != "ugn":
            tc = 1.0 / (self._tm_s * gg)    # LSIC from the STEADY tau_m
            tp = self._tm_s
        else:
            tc = tcval
            tp = tm
        self.cf = {"tau_m": tm, "tau_c": tc}
        stab = ngs.CF(0.0)
        if "supg" in self.stab:
            sw = Gv * W
            stab = stab + tm * rho * ngs.InnerProduct(sw, Rm)
        if "pspg" in self.stab:
            fp = 1.0 / rho
            stab = stab + tp * fp * ngs.InnerProduct(ngs.CF((ngs.grad(Q)[0], ngs.grad(Q)[1])), Rm)
        if "lsic" in self.stab:
            stab = stab + tc * rho * divu * divv
        return gal, stab

    def _build(self):
        X, mesh = self.X, self.mesh
        tr, te = X.TnT()
        self.tr, self.te = tr, te
        gal, stab = self._fields(list(tr[:2]), tr[2], list(te[:2]), te[2])
        bulk = gal
        # body force & mean constraint (the Galerkin f term; the stabilization carries f inside R)
        bulk = bulk - ngs.InnerProduct(self.f, ngs.CF((te[0], te[1])))
        if self.mean_zero:
            bulk = bulk + (tr[3] * te[2] + te[3] * tr[2])
        a = ngs.BilinearForm(X)
        sir = self.ir
        a += bulk * ngs.dx(intrules={ngs.TRIG: self.ir}) + stab * ngs.dx(intrules={ngs.TRIG: sir})
        self.a = a
        # fixed-tau Jacobian (quasi-Newton variant): tau evaluated at a copy of the state
        self.tauW = [ngs.GridFunction(sp_) for sp_ in self.U]
        self.afz = None
        if self.advect is None and self.convection:
            keep = self.tau_state
            self.tau_state = ngs.CF((self.tauW[0], self.tauW[1]))
            g2, st2 = self._fields(list(tr[:2]), tr[2], list(te[:2]), te[2])
            self.tau_state = keep
            b2 = g2 - ngs.InnerProduct(self.f, ngs.CF((te[0], te[1])))
            if self.mean_zero:
                b2 = b2 + (tr[3] * te[2] + te[3] * tr[2])
            self.afz = ngs.BilinearForm(X)
            self.afz += b2 * ngs.dx(intrules={ngs.TRIG: self.ir}) + st2 * ngs.dx(intrules={ngs.TRIG: sir})

    def mass_dual(self):
        """Dual-norm machinery (as for the steady Navier-Stokes cases, on P1/P1: consistent mass matrices, free velocity
        rows)."""
        X = self.X
        tr, te = X.TnT()
        md = ngs.BilinearForm(X)
        md += (tr[0] * te[0] + tr[1] * te[1] + tr[2] * te[2]) * ngs.dx
        md.Assemble()
        dfree = ngs.BitArray(self.X.FreeDofs())
        if self.mean_zero:
            r = X.Range(3)
            for d in range(r.start, r.stop):
                dfree.Clear(d)
        self.dfree = dfree
        self.Mdinv = md.mat.Inverse(dfree, inverse="umfpack")
        self._dv, self._do = md.mat.CreateColVector(), md.mat.CreateColVector()

    def residual(self, vec=None):
        vec = vec if vec is not None else self.gf.vec
        res = vec.CreateVector()
        self.a.Apply(vec, res)
        arr = np.array(res.FV().NumPy())
        arr[~np.array(self.free)] = 0.0
        return arr, res

    def norm(self, arr):
        if not hasattr(self, "Mdinv"):
            self.mass_dual()
        a = np.array(arr, float)
        a[~np.array(self.dfree)] = 0.0
        self._dv.FV().NumPy()[:] = a
        self._do.data = self.Mdinv * self._dv
        return float(np.sqrt(max(np.dot(a, np.array(self._do.FV().NumPy())), 0.0)))

    def r0(self):
        v = self.gf.vec.CreateVector()
        v.data = self.gf.vec
        a = np.array(v.FV().NumPy())
        a[np.array(self.free)] = 0.0
        v.FV().NumPy()[:] = a
        arr, _ = self.residual(v)
        return max(self.norm(arr), 1e-300)

    def newton(self, rtol=1e-11, maxit=40, line_search=True, r0=None, history=None, jac_mode="exact"):
        """Newton on the (steady or one-step) residual; relative dual-norm residuals; Armijo line search."""
        hist = history if history is not None else []
        r0 = r0 or self.r0()
        arr, res = self.residual()
        rn = self.norm(arr) / r0
        hist.append({"kind": "start", "residual": rn, "step": None})
        for it in range(maxit):
            if rn <= rtol:
                break
            if jac_mode == "frozen_tau":
                for k in range(2):
                    self.tauW[k].vec.data = self.gf.components[k].vec
                self.afz.AssembleLinearization(self.gf.vec)
                J = self.afz.mat
            else:
                self.a.AssembleLinearization(self.gf.vec)
                J = self.a.mat
            du = self.gf.vec.CreateVector()
            du.data = J.Inverse(self.free, inverse="umfpack") * res
            lam, base = 1.0, self.gf.vec.CreateVector()
            base.data = self.gf.vec
            while True:
                self.gf.vec.data = base - lam * du
                arr, res = self.residual()
                rnew = self.norm(arr) / r0
                if not line_search or rnew <= (1 - 1e-4 * lam) * rn or lam <= 1 / 64:
                    break
                lam *= 0.5
            hist.append({"kind": "newton", "residual": rnew, "step": lam})
            rn = rnew
            if not math.isfinite(rn) or rn > 1e8:
                break
        return hist

    def evaluator(self):
        return GEval(self.mesh, list(self.gf.components[:2]), self.gf.components[2], self)

    def uvec(self):
        return [np.array(self.gf.components[k].vec.FV().NumPy()) for k in range(3)]


class GEval(NGEval):
    def __init__(self, mesh, comps, p, prob):
        super().__init__(mesh, comps, p)
        self.prob = prob

    def pressure_gradient(self, ref):
        pts = self._pts(ref)
        return ngs.grad(self.p)(pts).reshape(self.n_elements, len(ref), 2)


def solve_mms(fld, mesh, form, mu_scale=1.0, stab=("supg", "pspg", "lsic"), convection=True, nu_nodal=False,
              steps=None, rtol=1e-11, tau="metric", CI=36.0, CG=1.0, tauc="g", oseen=False, jac_mode="exact"):
    """Steady MMS: homotopy in the viscosity scale (geometric steps from 'steps' to 1) with the exact solution as the
    Dirichlet data.  Returns (problem, history, seconds)."""
    t0 = time.time()
    cc = fld.coords
    d = fld.data
    mu_cf = cf_from_sympy(d["mu"]["sympy"], cc)
    gmu = ngs.CF(tuple(cf_from_sympy(s, cc) for s in d["f5"]["grad_mu"]))
    if nu_nodal:
        H = ngs.H1(mesh, order=1)
        gfm = ngs.GridFunction(H)
        # exact nodal interpolation (GridFunction.Set projects: it is NOT the interpolant)
        Vp = np.array([mesh.vertices[i].point for i in range(len(mesh.vertices))])
        gfm.vec.FV().NumPy()[:] = fld.mu(Vp[:, 0], Vp[:, 1])
        mu_cf = gfm
        gmu = ngs.grad(gfm)
    parts = d["f5"]["parts"][form]
    fc = ngs.CF(tuple(cf_from_sympy(s, cc) for s in parts["conv"]))
    fp = ngs.CF(tuple(cf_from_sympy(s, cc) for s in parts["gradp"]))
    fv = ngs.CF(tuple(cf_from_sympy(s, cc) for s in parts["visc"]))
    s = ngs.Parameter(1.0)
    f_cf = (fc if convection else 0 * fc) + fp + s * fv
    if oseen:
        w = ngs.CF(tuple(cf_from_sympy(sx["sympy"], cc) for sx in d["u"]))
    else:
        w = None
    mu_s = s * mu_cf
    gmu_s = s * gmu
    prob = GLS(mesh, fld.rho, mu_s, gmu_s, f_cf, ["x0", "x1", "y0", "y1"], fld.u, form=form, convection=convection,
               mean_zero=True, stab=stab, tau=tau, CI=CI, CG=CG, tauc=tauc, advect=w)
    hist = []
    r0 = prob.r0()
    sched = [1.0] if steps is None else list(steps)
    for sv in sched:
        s.Set(sv)
        prob.newton(rtol=rtol if sv == sched[-1] else 1e-8, r0=r0, history=hist)
    return prob, hist, time.time() - t0
