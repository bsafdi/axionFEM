"""Independent NGSolve reference solver for 2.5D inductionless MHD.

NGSolve 6.2.2608, written from the continuum equations (never from axionFEM code):
    u = (u_x, u_y, u_z) in P2^3, p in P1, phi in P2 (Lagrange), Galerkin, no stabilization, sparse LU (UMFPACK);
    J = sigma(-grad phi + E_z e_z + u x B);  momentum  mu(grad u, grad v) - (p, div v) + rho((u.grad)u, v)
    - (J x B, v) = (f + G_z e_z, v) + <g, v>_traction;  continuity -(q_p, div u);  potential
    -(J, grad psi) - (q, psi) + <j_n, psi>_Gamma_jn = 0 (insulating walls: natural);  Newton for rho > 0.
Dirichlet data by P2 LAGRANGE NODAL interpolation (vertex values + edge-bubble correction at the midpoints);
nullspaces (pressure, potential) by mean-value multipliers.  The meshes come from the fixed builders of
benchmarks/fluids/cases/f13a_mhd_check.py, and the Evaluator below lets that check module evaluate norms and
integrals on the NGSolve solution.  Used by xcode/ngsolve_mhd_duct_reference.py.
"""
from __future__ import annotations

import os

import numpy as np
import ngsolve as ngs
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element1D, FaceDescriptor, Pnt

ngs.SetNumThreads(int(os.environ.get("OMP_NUM_THREADS", "1")))


# ---------------------------------------------------------------------------------------------- mesh
def ng_mesh(V, T, label):
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
    names = []
    for key, c in cnt.items():
        if c == 1:
            a, b = orient[key]
            nm = label(V[a], V[b])
            if nm not in names:
                names.append(nm)
            m.Add(Element1D([pids[a], pids[b]], index=names.index(nm) + 1))
    for i, nm in enumerate(names):
        m.SetBCName(i, nm)
    return ngs.Mesh(m)


def rect_label(x0, x1, y0, y1):
    def lab(a, b):
        for nm, (k, v) in (("x0", (0, x0)), ("x1", (0, x1)), ("y0", (1, y0)), ("y1", (1, y1))):
            if abs(a[k] - v) < 1e-12 and abs(b[k] - v) < 1e-12:
                return nm
        raise RuntimeError("edge not on a side")
    return lab


# ---------------------------------------------------------------------------------------------- solver
class Problem:
    def __init__(self, **kw):
        self.mu = kw.get("mu", 1.0)
        self.rho = kw.get("rho", 0.0)
        self.sigma = kw.get("sigma", ngs.CF(1.0))
        self.B = kw.get("B", [ngs.CF(0.0)] * 3)
        self.Ez = kw.get("Ez", 0.0)
        self.Gz = kw.get("Gz", 0.0)
        self.f = kw.get("f")                       # 3 CFs or None
        self.q = kw.get("q")                       # CF or None
        self.u_dir = kw.get("u_dir", {})           # bc -> 3 CFs
        self.trac = kw.get("trac", {})             # bc -> 3 CFs (sigma_lap n)
        self.phi_dir = kw.get("phi_dir", {})       # bc -> CF
        self.jn = kw.get("jn", {})                 # bc -> CF


def nodal_set(gfc, fes, mesh, data_by_bc):
    """P2 Lagrange nodal interpolation of boundary data: vertex values by Set(dual=True), then each boundary edge's
    bubble coefficient corrected so that the midpoint value equals the data; returns the max nodal defect."""
    defects = []
    if not data_by_bc:
        return 0.0
    # ONE Set over the union of the Dirichlet boundaries (a Set per region resets the other regions' dofs);
    # shared corner nodes: the per-attribute data agree there (fixed shared-node rule), checked below.
    gfc.Set(mesh.BoundaryCF(dict(data_by_bc), default=0), definedon=mesh.Boundaries("|".join(data_by_bc)), dual=True)
    if fes.globalorder < 2:
        return 0.0
    probe = ngs.GridFunction(fes)
    phi_mid = None
    for bc, g in data_by_bc.items():
        for el in mesh.Elements(ngs.BND):
            if el.mat != bc:
                continue
            for e in el.edges:
                vts = mesh[e].vertices
                pa, pb = (np.array(mesh[v].point) for v in vts)
                xm, ym = 0.5 * (pa + pb)
                vdofs = set(sum((list(fes.GetDofNrs(v)) for v in vts), []))
                edofs = [d for d in fes.GetDofNrs(e) if d >= 0 and d not in vdofs]
                if phi_mid is None:
                    probe.vec[:] = 0
                    probe.vec[edofs[0]] = 1.0
                    phi_mid = probe(mesh(xm, ym, 0.0))
                cur = gfc(mesh(xm, ym, 0.0))
                gfc.vec[edofs[0]] += (g(mesh(xm, ym, 0.0)) - cur) / phi_mid
                for pt in (pa, pb, (xm, ym)):
                    defects.append(abs(gfc(mesh(pt[0], pt[1], 0.0)) - g(mesh(pt[0], pt[1], 0.0))))
    return float(max(defects)) if defects else 0.0


def solve(mesh, pb: Problem, phi_order=2):
    udir = "|".join(pb.u_dir.keys())
    pdir = "|".join(pb.phi_dir.keys())
    Vc = ngs.H1(mesh, order=2, dirichlet=udir)
    Q = ngs.H1(mesh, order=1)
    Ph = ngs.H1(mesh, order=phi_order, dirichlet=pdir)
    p_null = not pb.trac
    phi_null = not pb.phi_dir
    spaces = [Vc, Vc, Vc, Q, Ph]
    if p_null:
        spaces.append(ngs.NumberSpace(mesh))
    if phi_null:
        spaces.append(ngs.NumberSpace(mesh))
    X = ngs.FESpace(spaces)
    TR, TE = X.TrialFunction(), X.TestFunction()
    ux, uy, uz, p, ph = TR[:5]
    vx, vy, vz, qq, ps = TE[:5]
    k = 5
    if p_null:
        lp, mp_ = TR[k], TE[k]
        k += 1
    if phi_null:
        lf, mf = TR[k], TE[k]
        k += 1
    sig = pb.sigma
    B = list(pb.B)
    u = [ux, uy, uz]
    v = [vx, vy, vz]
    Ez = pb.Ez
    uxB = [u[1] * B[2] - u[2] * B[1], u[2] * B[0] - u[0] * B[2], u[0] * B[1] - u[1] * B[0]]
    gph = ngs.grad(ph)
    J = [sig * (-gph[0] + uxB[0]), sig * (-gph[1] + uxB[1]), sig * (Ez + uxB[2])]
    F = [J[1] * B[2] - J[2] * B[1], J[2] * B[0] - J[0] * B[2], J[0] * B[1] - J[1] * B[0]]     # J x B
    a = ngs.BilinearForm(X)
    form = 0
    for i in range(3):
        form += pb.mu * ngs.InnerProduct(ngs.grad(u[i]), ngs.grad(v[i]))
        if pb.rho:
            form += pb.rho * (ux * ngs.grad(u[i])[0] + uy * ngs.grad(u[i])[1]) * v[i]
        form += -F[i] * v[i]
        rhs_i = (pb.f[i] if pb.f is not None else 0) + (pb.Gz if i == 2 else 0)
        form += -rhs_i * v[i]
    divu = ngs.grad(ux)[0] + ngs.grad(uy)[1]
    divv = ngs.grad(vx)[0] + ngs.grad(vy)[1]
    form += -p * divv - qq * divu
    if p_null:
        form += lp * qq + mp_ * p
    gps = ngs.grad(ps)
    phi_row = -(J[0] * gps[0] + J[1] * gps[1])
    if pb.q is not None:
        phi_row += -pb.q * ps
    if phi_null:
        form += lf * ps + mf * ph
    a += form * ngs.dx
    a += phi_row * ngs.dx
    for bc, g in pb.trac.items():
        a += -sum(g[i] * v[i] for i in range(3)) * ngs.ds(definedon=mesh.Boundaries(bc))
    for bc, g in pb.jn.items():
        a += g * ps * ngs.ds(definedon=mesh.Boundaries(bc))
    gf = ngs.GridFunction(X)
    nd = []
    for i in range(3):
        nd.append(nodal_set(gf.components[i], Vc, mesh, {bc: g[i] for bc, g in pb.u_dir.items()}))
    if pb.phi_dir:
        nd.append(nodal_set(gf.components[4], Ph, mesh, pb.phi_dir))
    free = X.FreeDofs()
    fm = np.array(free)
    res = gf.vec.CreateVector()
    hist = []
    r0 = None
    for it in range(30):
        a.Apply(gf.vec, res)
        rn = float(np.linalg.norm(np.array(res.FV().NumPy())[fm]))
        r0 = rn if r0 is None else r0
        hist.append(rn)
        if rn <= 1e-13 * max(r0, 1e-300) or (it > 0 and rn < 1e-14):
            break
        a.AssembleLinearization(gf.vec)
        inv = a.mat.Inverse(free, inverse="umfpack")
        gf.vec.data -= inv * res
    rel = hist[-1] / max(hist[0], 1e-300)
    return gf, {"newton_residuals": hist, "relative_residual": rel, "converged": bool(rel < 1e-10),
                "dirichlet_nodal_defect": float(max(nd)) if nd else 0.0, "ndof": int(sum(free)),
                "p_null": p_null, "phi_null": phi_null}


class Evaluator:
    """f13a_mhd_check Evaluator protocol over the NGSolve solution (physical point location)."""

    def __init__(self, mesh, V, T, gf):
        self.mesh, self.V, self.T = mesh, np.asarray(V, float), np.asarray(T)
        self.gf = gf
        self.n_elements = len(T)
        P = self.V[self.T]
        self.a0, self.e1, self.e2 = P[:, 0], P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]

    def geometry(self, ref):
        ref = np.asarray(ref, float)
        x = self.a0[:, None, :] + ref[None, :, 0:1] * self.e1[:, None, :] + ref[None, :, 1:2] * self.e2[:, None, :]
        det = self.e1[:, 0] * self.e2[:, 1] - self.e1[:, 1] * self.e2[:, 0]
        return x, np.repeat(np.abs(det)[:, None], len(ref), axis=1)

    def _pts(self, ref):
        x, _ = self.geometry(ref)
        # nudge points on element boundaries 1e-12 towards the element centroid so the location is unambiguous
        cen = self.a0[:, None, :] + (self.e1[:, None, :] + self.e2[:, None, :]) / 3
        xx = x + 1e-11 * (cen - x)
        return self.mesh(xx[..., 0].ravel(), xx[..., 1].ravel()), x.shape[:2]

    def _ev(self, c, ref, dim):
        mp, sh = self._pts(ref)
        v = np.asarray(c(mp))
        return v.reshape(sh + ((dim,) if dim > 1 else ()))

    def velocity(self, ref):
        cs = self.gf.components
        u = np.stack([self._ev(cs[i], ref, 1) for i in range(3)], axis=-1)
        g = np.stack([self._ev(ngs.grad(cs[i]), ref, 2) for i in range(3)], axis=-2)
        return u, g

    def pressure(self, ref):
        return self._ev(self.gf.components[3], ref, 1)

    def potential(self, ref):
        c = self.gf.components[4]
        return self._ev(c, ref, 1), self._ev(ngs.grad(c), ref, 2)
