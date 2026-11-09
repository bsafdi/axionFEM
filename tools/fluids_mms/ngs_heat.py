"""Independent NGSolve reference solver for scalar transport and Boussinesq.

NGSolve 6.2.2608, written from the continuum equations, never
from axionFEM code:
  * steady scalar transport, planar and axisymmetric, P1 or P2, convective or conservative advective form, Dirichlet
    (P1/P2 LAGRANGE NODAL interpolation), diffusive-flux Neumann (or total flux, conservative form), Robin, symmetry /
    axis natural; residual-based SUPG with the FULL element residual (second derivatives, grad kappa, the axisymmetric
    kappa T_r / r) and three tau rules (coth, Shakib, and either scaled by a factor);
  * Boussinesq: Taylor-Hood P2/P1 + P2 temperature, planar and axisymmetric with swirl, lap / sym viscous forms,
    standard convection, gravity -e_up, optional SUPG on the energy equation with tau(u_h), Newton (NGSolve symbolic
    linearization) from the Stokes + conduction start.
Errors are computed by the fixed check modules (scalar_transport_check, f4a_heat_check) through an Evaluator
adapter, so the check modules are also exercised on real discrete solutions.
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
import scalar_transport_check as S  # noqa: E402
import f4a_heat_check as H  # noqa: E402

_FN = {'sin': ngs.sin, 'cos': ngs.cos, 'exp': ngs.exp, 'sinh': ngs.sinh, 'cosh': ngs.cosh, 'sqrt': ngs.sqrt,
       'log': ngs.log, 'pi': math.pi, 'atan2': ngs.atan2}


def cf(s: str, coords):
    syms = sp.symbols(" ".join(coords), real=True)
    e = sp.sympify(s, locals={str(v): v for v in syms})
    if not e.free_symbols:
        return ngs.CF(float(e))
    return sp.lambdify(syms, e, modules=[_FN, 'math'])(ngs.x, ngs.y)


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
    x0, x1, y0, y1 = (float(v) for v in box)

    def nm(Pa, Pb, tol=1e-12):
        if abs(Pa[0] - x0) < tol and abs(Pb[0] - x0) < tol:
            return ("axis" if x0 == 0.0 else "r0") if axi else "x0"
        if abs(Pa[0] - x1) < tol and abs(Pb[0] - x1) < tol:
            return "r1" if axi else "x1"
        if abs(Pa[1] - y0) < tol and abs(Pb[1] - y0) < tol:
            return "z0" if axi else "y0"
        if abs(Pa[1] - y1) < tol and abs(Pb[1] - y1) < tol:
            return "z1" if axi else "y1"
        raise ValueError((Pa, Pb))
    return nm


def make_mesh(V, T, box, axi):
    return ng_mesh(V, T, box_namer(box, axi))


# ============================================================================================== evaluator
class NGEval:
    """The check modules' Evaluator protocol over NGSolve GridFunctions (affine elements)."""

    def __init__(self, mesh, T=None, comps=None, p=None):
        self.mesh, self.T, self.comps, self.p = mesh, T, comps, p
        self.n_elements = mesh.ne
        self._cache = {}

    def _pts(self, ref):
        key = np.asarray(ref).tobytes()
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

    def scalar(self, ref):
        pts = self._pts(ref)
        return (self.T(pts)[:, 0].reshape(self.n_elements, len(ref)),
                ngs.grad(self.T)(pts).reshape(self.n_elements, len(ref), 2))

    def velocity(self, ref):
        pts = self._pts(ref)
        u = np.stack([c(pts)[:, 0] for c in self.comps], -1).reshape(self.n_elements, len(ref), len(self.comps))
        D = np.stack([ngs.grad(c)(pts) for c in self.comps], -2).reshape(self.n_elements, len(ref), len(self.comps), 2)
        return u, D

    def pressure(self, ref):
        return self.p(self._pts(ref))[:, 0].reshape(self.n_elements, len(ref))


# ============================================================================================== nodal data
def set_nodal(gf, fes, mesh, g, names=None, order=2, verts_only_value=None):
    """P1/P2 Lagrange NODAL interpolation of g(A, B) on the boundary parts `names` (all boundary if None): vertex
    dofs = g(vertex); P2 edge bubbles chosen so the edge-midpoint value equals g(midpoint).  Returns the max nodal
    defect (re-evaluated) and the node coordinates / values."""
    edges, verts = set(), set()
    for el in mesh.Elements(ngs.BND):
        if names is None or el.mat in names:
            for e in el.edges:
                edges.add(e.nr)
            for v in el.vertices:
                verts.add(v.nr)
    vs = sorted(verts)
    V = np.array([mesh.vertices[v].point for v in vs]) if vs else np.zeros((0, 2))
    gv = g(V[:, 0], V[:, 1]) if len(V) else np.zeros(0)
    for k, v in enumerate(vs):
        for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v)):
            gf.vec[d] = gv[k]
    pts, vals = list(V), list(gv)
    if order >= 2 and edges:
        es = sorted(edges)
        mids = []
        ends = []
        for e in es:
            ed = mesh[ngs.NodeId(ngs.EDGE, e)]
            a, b = (mesh[v].point for v in ed.vertices)
            mids.append(0.5 * (np.array(a) + np.array(b)))
            ends.append([v.nr for v in ed.vertices])
        mids = np.array(mids)
        gm = g(mids[:, 0], mids[:, 1])
        probe = ngs.GridFunction(fes)
        for k, e in enumerate(es):
            dofs = fes.GetDofNrs(ngs.NodeId(ngs.EDGE, e))
            if not dofs:
                continue
            probe.vec[:] = 0
            probe.vec[dofs[0]] = 1.0
            bub = probe(mesh(*mids[k]))
            vlin = 0.5 * (gf(mesh(*mesh.vertices[ends[k][0]].point)) + gf(mesh(*mesh.vertices[ends[k][1]].point)))
            gf.vec[dofs[0]] = (gm[k] - vlin) / bub
            for d in dofs[1:]:
                gf.vec[d] = 0.0
        pts += list(mids)
        vals += list(gm)
    P = np.array(pts) if pts else np.zeros((0, 2))
    got = np.array([gf(mesh(*p)) for p in P]) if len(P) else np.zeros(0)
    defect = float(np.max(np.abs(got - np.array(vals)))) if len(P) else 0.0
    return defect, P, np.array(vals)


def nodal_indicator_weights(fes, mesh, names):
    """Coefficient vector (in NGSolve's hierarchical basis) of v_Gamma = the P2 (P1) LAGRANGE NODAL function equal to 1
    at every node (vertex and edge midpoint) of the CLOSED boundary part `names` and 0 at every other node
 (the test function of the variational flux; in a nodal basis it is the indicator vector of the part's DOFs).  At a corner shared with another Dirichlet part, v_Gamma vanishes at the midpoints of the other
    part's edges (unlike the bare vertex hat), so the flux is basis-free."""
    gverts, gedges = set(), set()
    for el in mesh.Elements(ngs.BND):
        if el.mat in names:
            for v in el.vertices:
                gverts.add(v.nr)
            for e in el.edges:
                gedges.add(e.nr)
    w = np.zeros(fes.ndof)
    for v in gverts:
        for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v)):
            w[d] = 1.0
    probe = ngs.GridFunction(fes)
    for e in mesh.edges:
        vs = [v.nr for v in e.vertices]
        a, b = (1.0 if v in gverts else 0.0 for v in vs)
        m = 1.0 if e.nr in gedges else 0.0
        target = m - 0.5 * (a + b)
        if target == 0.0:
            continue
        dofs = fes.GetDofNrs(ngs.NodeId(ngs.EDGE, e.nr))
        if not dofs:
            continue
        pa, pb = (np.array(mesh[ngs.NodeId(ngs.VERTEX, v)].point) for v in vs)
        mid = 0.5 * (pa + pb)
        probe.vec[:] = 0
        probe.vec[dofs[0]] = 1.0
        w[dofs[0]] = target / probe(mesh(*mid))
    return w


def boundary_vertex_weights(fes, mesh, names):
    """Coefficient vector (hierarchical NGSolve basis) of a discrete function equal to 1 on the closure of the
    boundary parts `names` and 0 at every other node: the sum of the vertex hats (edge bubbles 0)."""
    w = np.zeros(fes.ndof)
    for el in mesh.Elements(ngs.BND):
        if el.mat in names:
            for v in el.vertices:
                for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v.nr)):
                    w[d] = 1.0
    return w


# ============================================================================================== scalar transport
_LONG = {}


def longest_edge_gf(mesh):
    """Element-wise longest edge (a function of the element's geometry only), as an L2(order 0) GridFunction."""
    key = id(mesh)
    # the cache holds the mesh itself: an id() can be reused by a later mesh once the earlier one is freed (a stale
    # length field from another mesh gives erratic shakib_long results)
    if key not in _LONG or _LONG[key][0] is not mesh:
        L0 = ngs.L2(mesh, order=0)
        g = ngs.GridFunction(L0)
        for el in mesh.Elements(ngs.VOL):
            P = [np.array(mesh[v].point) for v in el.vertices]
            g.vec[L0.GetDofNrs(el)[0]] = max(np.linalg.norm(P[i] - P[j]) for i, j in ((0, 1), (1, 2), (2, 0)))
        _LONG[key] = (mesh, g)
    return _LONG[key][1]


def tau_cf(bvec, a, kappa, p, rule="coth", scale=1.0, mesh=None):
    """SUPG parameter (the fixed family): rule 'coth': tau = h / (2 p a|b|) (coth Pe_K - 1/Pe_K),
    Pe_K = a|b| h / (2 p kappa); rule 'shakib': tau = [(2 p a|b|/h)^2 + 9 (4 p^2 kappa / h^2)^2]^(-1/2); h = NGSolve's
    mesh_size = sqrt(2|K|) on triangles, or (rules '*_long') the element's longest edge; times `scale`."""
    h = ngs.specialcf.mesh_size
    if rule.endswith("_long"):
        h = longest_edge_gf(mesh)
        rule = rule[:-5]
    nb = ngs.sqrt(bvec[0] * bvec[0] + bvec[1] * bvec[1] + 1e-300)
    if rule == "coth":
        Pe = a * nb * h / (2 * p * kappa)
        xi = ngs.IfPos(Pe - 1e-3, (ngs.exp(2 * ngs.IfPos(Pe - 30, 30, Pe)) + 1) / (ngs.exp(2 * ngs.IfPos(Pe - 30, 30, Pe)) - 1) - 1 / Pe, Pe / 3)
        tau = h / (2 * p * a * nb) * xi
    elif rule == "shakib":
        tau = 1 / ngs.sqrt((2 * p * a * nb / h) ** 2 + 9 * (4 * p * p * kappa / h ** 2) ** 2)
    else:
        raise ValueError(rule)
    return scale * tau


class ScalarProblem:
    """Steady scalar transport on an NGSolve mesh from a generated field (scalar_transport_check.ScalarField data),
    planar or axisymmetric.  bcs: {name: type} with type in dirichlet | neumann | robin | symmetry | axis."""

    def __init__(self, mesh, data, order=2, supg=False, tau_rule="coth", tau_scale=1.0, adv_form="convective",
                 bcs=None, kappa_override=None):
        self.mesh, self.data, self.order = mesh, data, order
        self.geometry = data["geometry"]
        self.axi = self.geometry == "axisymmetric"
        cc = data["coords"]
        self.kind = data["kind"]
        bcs = bcs or {bd["name"]: bd["type"] for bd in data["boundaries"]}
        self.bcs = bcs
        dnames = [n for n, t in bcs.items() if t == "dirichlet"]
        self.fes = ngs.H1(mesh, order=order, dirichlet="|".join(dnames) if dnames else "")
        T, v = self.fes.TnT()
        gfT = ngs.GridFunction(self.fes)
        self.gf = gfT
        if self.kind == "graetz_modes":
            fld = S.ScalarField(data)
            a = float(data["a"]["value"])
            kap = ngs.CF(float(data["kappa"]["value"]))
            gk = (ngs.CF(0.0), ngs.CF(0.0))
            if self.axi:
                b = (ngs.CF(0.0), 2.0 * (1 - ngs.x * ngs.x))
            else:
                b = (1.5 * (1 - ngs.y * ngs.y), ngs.CF(0.0))
            s = ngs.CF(0.0)
            self.exact_np = fld.T
            gradT_np = fld.grad_T
            self._graetz = fld
        else:
            fld = S.ScalarField(data)
            a = float(data["a"]["value"])
            kap = cf(data["kappa"]["sympy"], cc)
            gk = tuple(cf(e["sympy"], cc) for e in data["grad_kappa"])
            b = tuple(cf(e["sympy"], cc) for e in data["b"])
            s = cf(data["s"][adv_form]["sympy"], cc)
            self.exact_np = fld.T
        self.fld = fld
        if kappa_override is not None:
            kap = kappa_override
        rw = ngs.x if self.axi else ngs.CF(1.0)
        self.rw = rw
        gT = ngs.grad(T)
        gv = ngs.grad(v)
        bdT = b[0] * gT[0] + b[1] * gT[1]
        bdv = b[0] * gv[0] + b[1] * gv[1]
        form = ngs.BilinearForm(self.fes, symmetric=False, nonassemble=False)
        if adv_form == "convective":
            form += a * bdT * v * rw * ngs.dx(bonus_intorder=4)
        else:
            form += -a * T * bdv * rw * ngs.dx(bonus_intorder=4)
        form += kap * (gT * gv) * rw * ngs.dx(bonus_intorder=4)
        lin = ngs.LinearForm(self.fes)
        lin += s * v * rw * ngs.dx(bonus_intorder=6)
        # natural boundaries
        self.natural = {}
        for bd in data["boundaries"]:
            nm, tp = bd["name"], bcs.get(bd["name"], bd["type"])
            if tp == "neumann":
                if self.kind == "graetz_modes":
                    q = self._graetz_flux_cf(bd["name"])
                else:
                    key = "q_out" if adv_form == "convective" else "j_out"
                    q = cf(bd[key]["sympy"], cc)
                lin += -q * v * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
            elif tp == "neumann_conservative_data_on_convective":
                q = cf(bd["j_out"]["sympy"], cc)
                lin += -q * v * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
            elif tp == "robin":
                hval = float(bd["h"]["value"])
                Text = cf(bd["T_ext" if adv_form == "convective" else "T_ext_conservative"]["sympy"], cc)
                form += hval * T * v * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
                lin += hval * Text * v * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
            # symmetry / axis: natural, zero flux
        if adv_form == "conservative":
            # outflow part of the total flux on natural boundaries is the data j_out (already in lin); nothing else
            pass
        self.supg = supg
        if supg:
            hes = T.Operator("hesse")
            lap = hes[0] + hes[3]
            if self.axi:
                lap = lap + gT[0] / ngs.IfPos(ngs.x - 1e-300, ngs.x, 1e-300)
            divflux = kap * lap + gk[0] * gT[0] + gk[1] * gT[1]
            resid_T = a * bdT - divflux
            tau = tau_cf(b, a, kap, order, tau_rule, tau_scale, mesh)
            form += tau * a * bdv * resid_T * rw * ngs.dx(bonus_intorder=4)
            lin += tau * a * bdv * s * rw * ngs.dx(bonus_intorder=6)
        self.form, self.lin = form, lin
        # Dirichlet data
        if self.kind == "graetz_modes":
            gfun = fld.T
        else:
            gfun = fld.T
        dset = [n for n, t in bcs.items() if t == "dirichlet"]
        self.nodal_defect, self.dnodes, self.dvals = set_nodal(gfT, self.fes, mesh, gfun, dset, order)
        self.dnames = dset

    def _graetz_flux_cf(self, name):
        """Exact outward diffusive flux of the Graetz modes on the outlet, as a high-order interpolant (order 8, 1D
        Chebyshev data evaluated through a GridFunction of the boundary)."""
        fld = self.fld
        kap = float(self.data["kappa"]["value"])
        Vb = ngs.H1(self.mesh, order=8, definedon=self.mesh.Boundaries(name))
        g = ngs.GridFunction(Vb)

        class _F:
            pass
        # set via point evaluation on a fine 1D Gauss set by least squares: use Set with a CoefficientFunction
        # built from the modes in closed form (exp and a polynomial in the transverse coordinate)
        t = ngs.x if self.axi else ngs.y
        sco = ngs.y if self.axi else ngs.x
        expr = ngs.CF(0.0)
        for lm, amp, ch in fld.modes:
            # Chebyshev series as a polynomial CF via Clenshaw in the variable 2t - 1
            tt = 2 * t - 1
            b1, b2 = ngs.CF(0.0), ngs.CF(0.0)
            for c in ch[::-1][:-1]:
                b1, b2 = 2 * tt * b1 - b2 + float(c), b1
            phi = tt * b1 - b2 + float(ch[0])
            expr = expr + amp * phi * lm * ngs.exp(-lm * sco)
        # outward (+s direction) diffusive flux -kappa dT/ds = kappa sum lam A phi e^{-lam s}; Compile() evaluates the
        # Clenshaw DAG once per point (without it the shared sub-trees are re-evaluated exponentially often)
        return (kap * expr).Compile()

    def solve(self):
        t0 = time.time()
        self.form.Assemble()
        self.lin.Assemble()
        r = self.lin.vec.CreateVector()
        r.data = self.lin.vec - self.form.mat * self.gf.vec
        inv = self.form.mat.Inverse(self.fes.FreeDofs(), inverse="umfpack")
        self.gf.vec.data += inv * r
        res = self.lin.vec.CreateVector()
        res.data = self.form.mat * self.gf.vec - self.lin.vec
        # dual norm over the free rows, relative to the residual of the Dirichlet lift
        T, v = self.fes.TnT()
        mf = ngs.BilinearForm(self.fes)
        mf += T * v * self.rw * ngs.dx(bonus_intorder=6)
        mf.Assemble()
        Minv = mf.mat.Inverse(self.fes.FreeDofs(), inverse="umfpack")
        free = np.array(list(self.fes.FreeDofs()), bool)

        def dual(vec):
            r = vec.CreateVector()
            r.data = vec
            r.FV().NumPy()[~free] = 0.0
            y = r.CreateVector()
            y.data = Minv * r
            return math.sqrt(max(float(np.dot(np.array(r.FV()), np.array(y.FV()))), 0.0))
        lift = self.gf.vec.CreateVector()
        lift.data = self.gf.vec
        lift.FV().NumPy()[free] = 0.0
        r0 = self.lin.vec.CreateVector()
        r0.data = self.form.mat * lift - self.lin.vec
        self.relative_residual = dual(res) / max(dual(r0), 1e-300)
        self.seconds = time.time() - t0
        return self

    def full_residual(self):
        res = self.lin.vec.CreateVector()
        res.data = self.form.mat * self.gf.vec - self.lin.vec
        return np.array(res.FV())

    def outward_flux(self, names):
        R = self.full_residual()
        return S.flux_from_residual(R, weights=nodal_indicator_weights(self.fes, self.mesh, names))

    def evaluator(self):
        return NGEval(self.mesh, T=self.gf)

    def ndof(self):
        return self.fes.ndof


# ============================================================================================== Boussinesq
class BousProblem:
    """Steady Boussinesq on an NGSolve mesh.  data: a generated Boussinesq field (dict) or
    None with explicit coefficients (cavity: params dict).  Unknowns: velocity components (P2), p (P1), T (P2),
    optional mean multiplier."""

    def __init__(self, mesh, geometry, form, rho0, cp, beta_g, T_ref, mu, k, f_u, Q, u_dir, g_u, T_bcs, g_T,
                 mean_zero=True, supg=False, tau_rule="coth", tau_scale=1.0, robin=None, neumann=None, grad_k=None,
                 gamma_graddiv=0.0):
        self.mesh, self.geometry, self.form_name = mesh, geometry, form
        self.axi = geometry == "axisymmetric"
        nc = 3 if self.axi else 2
        self.nc = nc
        dset = "|".join(u_dir)
        if self.axi:
            Us = [ngs.H1(mesh, order=2, dirichlet=dset + "|axis"), ngs.H1(mesh, order=2, dirichlet=dset + "|axis"),
                  ngs.H1(mesh, order=2, dirichlet=dset)]
        else:
            Us = [ngs.H1(mesh, order=2, dirichlet=dset) for _ in range(nc)]
        Tdir = [n for n, t in T_bcs.items() if t == "dirichlet"]
        VT = ngs.H1(mesh, order=2, dirichlet="|".join(Tdir))
        Q1 = ngs.H1(mesh, order=1)
        spaces = Us + [Q1, VT] + ([ngs.NumberSpace(mesh)] if mean_zero else [])
        X = ngs.FESpace(spaces)
        self.X, self.Us, self.VT, self.mean_zero = X, Us, VT, mean_zero
        self.gf = ngs.GridFunction(X)
        tr = X.TrialFunction()
        te = X.TestFunction()
        u, v = list(tr[:nc]), list(te[:nc])
        p, q = tr[nc], te[nc]
        T, w = tr[nc + 1], te[nc + 1]
        r = ngs.x
        rw = r if self.axi else ngs.CF(1.0)
        self.rw = rw

        def G(ul):
            if self.axi:
                ur, ut, uz = ul
                return ngs.CF((ngs.grad(ur)[0], -ut / r, ngs.grad(ur)[1],
                               ngs.grad(ut)[0], ur / r, ngs.grad(ut)[1],
                               ngs.grad(uz)[0], 0, ngs.grad(uz)[1]), dims=(3, 3))
            return ngs.CF((ngs.grad(ul[0]), ngs.grad(ul[1])), dims=(2, 2))

        def div(ul):
            if self.axi:
                return ngs.grad(ul[0])[0] + ul[0] / r + ngs.grad(ul[2])[1]
            return ngs.grad(ul[0])[0] + ngs.grad(ul[1])[1]
        Gu, Gv = G(u), G(v)
        U = ngs.CF(tuple(u))
        V = ngs.CF(tuple(v))
        if form == "sym":
            visc = mu * ngs.InnerProduct(Gu + Gu.trans, Gv)
        else:
            visc = mu * ngs.InnerProduct(Gu, Gv)
        conv = rho0 * ngs.InnerProduct(Gu * U, V)
        up = 2 if self.axi else 1
        a = rho0 * cp
        mer = (u[0], u[2]) if self.axi else (u[0], u[1])
        gT = ngs.grad(T)
        gw = ngs.grad(w)
        udT = mer[0] * gT[0] + mer[1] * gT[1]
        bf = ngs.BilinearForm(X)
        integrand = (visc + conv - p * div(v) - q * div(u)
                     - rho0 * beta_g * (T - T_ref) * v[up]
                     + a * udT * w + k * (gT * gw)) * rw
        if gamma_graddiv:
            integrand = integrand + gamma_graddiv * div(u) * div(v) * rw
        if mean_zero:
            lam_, mu_ = tr[nc + 2], te[nc + 2]
            integrand = integrand + (lam_ * q + mu_ * p) * rw
        bf += integrand * ngs.dx(bonus_intorder=6)
        lf = ngs.LinearForm(X)
        if f_u is not None:
            lf += ngs.InnerProduct(f_u, V) * rw * ngs.dx(bonus_intorder=6)
        if Q is not None:
            lf += Q * w * rw * ngs.dx(bonus_intorder=6)
        for nm, (hval, Text) in (robin or {}).items():
            bf += hval * T * w * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
            lf += hval * Text * w * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
        for nm, qout in (neumann or {}).items():
            lf += -qout * w * rw * ngs.ds(definedon=mesh.Boundaries(nm), bonus_intorder=6)
        if supg:
            hes = T.Operator("hesse")
            lap = hes[0] + hes[3]
            if self.axi:
                lap = lap + gT[0] / ngs.IfPos(r - 1e-300, r, 1e-300)
            gkx = grad_k or (ngs.CF(0.0), ngs.CF(0.0))
            resT = a * udT - (k * lap + gkx[0] * gT[0] + gkx[1] * gT[1]) - (Q if Q is not None else 0.0)
            bvec = (mer[0], mer[1])
            tau = tau_cf(bvec, a, k, 2, tau_rule, tau_scale, mesh)
            bf += tau * a * (mer[0] * gw[0] + mer[1] * gw[1]) * resT * rw * ngs.dx(bonus_intorder=6)
        self.bf, self.lf = bf, lf
        self.supg = supg
        # Dirichlet data
        comps = self.gf.components
        self.nodal_defect = 0.0
        for ci in range(nc):
            def gci(A, B, ci=ci):
                val = np.asarray(g_u(A, B), float)[..., ci]
                if self.axi and ci < 2:
                    val = np.where(np.abs(A) < 1e-14, 0.0, val)
                return val
            d, _, _ = set_nodal(comps[ci], Us[ci], mesh, gci, list(u_dir) + (["axis"] if self.axi and ci < 2 else []), 2)
            self.nodal_defect = max(self.nodal_defect, d)
        dT, self.Tnodes, self.Tvals = set_nodal(comps[nc + 1], VT, mesh, g_T, Tdir, 2)
        self.nodal_defect = max(self.nodal_defect, dT)
        self.Tdir = Tdir
        self.free = X.FreeDofs()
        self.history = []
        # discrete DUAL norm of the residual (as F2a): consistent r-weighted L2 mass matrices
        # of the velocity components, the pressure and the temperature over the rows actually solved (free DOFs; the
        # mean-value multiplier row excluded)
        mform = ngs.BilinearForm(X)
        mform += (sum(u[i] * v[i] for i in range(nc)) + p * q + T * w) * rw * ngs.dx(bonus_intorder=6)
        mform.Assemble()
        rows = ngs.BitArray(self.free)
        if mean_zero:
            off = sum(sp_.ndof for sp_ in Us) + Q1.ndof + VT.ndof
            rows[off] = False
        self._dual_rows = rows
        self._Minv = mform.mat.Inverse(rows, inverse="umfpack")

    # ---------------------------------------------------------------------------- nonlinear solve
    def residual_vec(self, vec=None):
        vec = vec if vec is not None else self.gf.vec
        res = vec.CreateVector()
        self.bf.Apply(vec, res)
        res.data -= self.lf.vec
        return res

    def free_norm(self, res):
        """Discrete dual norm ||R||_* = (R^T M^-1 R)^(1/2) over the solved rows (basis-free)."""
        r = res.CreateVector()
        r.data = res
        mask = np.array(list(self._dual_rows), bool)
        r.FV().NumPy()[~mask] = 0.0
        y = r.CreateVector()
        y.data = self._Minv * r
        return float(math.sqrt(max(float(np.dot(np.array(r.FV()), np.array(y.FV()))), 0.0)))

    def solve(self, rtol=1e-12, maxit=40, start="stokes", line_search=False, jac=None, damping_min=1 / 64):
        self.lf.Assemble()
        t0 = time.time()
        nc = self.nc
        if start == "stokes":
            # Newton's first iterate from the Dirichlet lift is the Stokes + conduction solution only if convection is
            # absent; we simply start Newton from the lift (u = data, T = data, p = 0)
            pass
        r0 = self.free_norm(self.residual_vec())
        self.r_lift = max(r0, 1e-300)
        self.history = [{"kind": "start", "residual": 1.0}]
        du = self.gf.vec.CreateVector()
        it = 0
        while it < maxit:
            res = self.residual_vec()
            rn = self.free_norm(res) / self.r_lift
            if rn < rtol:
                break
            if jac is None:
                self.bf.AssembleLinearization(self.gf.vec)
                A = self.bf.mat
            else:
                A = jac(self)
            inv = A.Inverse(self.free, inverse="umfpack")
            du.data = inv * res
            lamb = 1.0
            if line_search:
                old = self.gf.vec.CreateVector()
                old.data = self.gf.vec
                while True:
                    self.gf.vec.data = old - lamb * du
                    rnew = self.free_norm(self.residual_vec()) / self.r_lift
                    if rnew <= (1 - 1e-4 * lamb) * rn or lamb <= damping_min:
                        break
                    lamb /= 2
            else:
                self.gf.vec.data -= du
            it += 1
            self.history.append({"kind": "newton", "residual": self.free_norm(self.residual_vec()) / self.r_lift,
                                 "step": lamb})
            if not np.isfinite(self.history[-1]["residual"]) or self.history[-1]["residual"] > 1e8:
                break
        self.relative_residual = self.history[-1]["residual"]
        self.converged = self.relative_residual < max(rtol * 10, 1e-11)
        self.iterations = it
        self.seconds = time.time() - t0
        return self

    def evaluator(self):
        c = self.gf.components
        return NGEval(self.mesh, T=c[self.nc + 1], comps=list(c[:self.nc]), p=c[self.nc])

    def T_outward_flux(self, names):
        """Variationally consistent outward heat flux through Dirichlet-T boundaries: - R . w with w the coefficients of
        the nodal v_Gamma of the T space on the closure of `names` (the momentum and continuity rows are not touched)."""
        R = np.array(self.residual_vec().FV())
        w = np.zeros(self.X.ndof)
        offs = sum(sp_.ndof for sp_ in self.Us) + self.X.components[self.nc].ndof
        wT = nodal_indicator_weights(self.VT, self.mesh, names)
        w[offs:offs + self.VT.ndof] = wT
        return float(-(R * w).sum())

    def ndof(self):
        return self.X.ndof


def bous_from_field(mesh, data, form, supg=False, tau_rule="coth", mean_zero=True, gamma_graddiv=0.0,
                    tau_scale=1.0):
    cc = data["coords"]
    axi = data["geometry"] == "axisymmetric"
    fld = H.BousField(data)
    mu = cf(data["mu"]["sympy"], cc)
    k = cf(data["k"]["sympy"], cc)
    gk = tuple(cf(e["sympy"], cc) for e in data["grad_k"])
    f_u = ngs.CF(tuple(cf(e["sympy"], cc) for e in data["forms"][form]["f"]))
    Q = cf(data["Q"]["sympy"], cc)
    T_bcs = {bd["name"]: bd["type"] for bd in data["T_boundaries"]}
    robin, neumann = {}, {}
    for bd in data["T_boundaries"]:
        if bd["type"] == "robin":
            robin[bd["name"]] = (float(bd["h"]["value"]), cf(bd["T_ext"]["sympy"], cc))
        elif bd["type"] == "neumann":
            neumann[bd["name"]] = cf(bd["q_out"]["sympy"], cc)
    u_dir = [bd["name"] for bd in data["u_boundaries"] if bd["type"] == "dirichlet"]
    return BousProblem(mesh, data["geometry"], form, fld.rho0, fld.cp, fld.beta_g, fld.T_ref, mu, k, f_u, Q, u_dir,
                       fld.u, T_bcs, fld.T, mean_zero=mean_zero, supg=supg, tau_rule=tau_rule, robin=robin,
                       neumann=neumann, grad_k=gk, gamma_graddiv=gamma_graddiv, tau_scale=tau_scale), fld


def cavity_problem(mesh, Ra, Pr, form="lap", supg=False, geometry="planar", tau_rule="coth", tau_scale=1.0):
    """Differentially heated square cavity in the de Vahl Davis scaling (length L = 1, alpha = k/(rho0 c_p) = 1,
    rho0 = c_p = 1, mu = Pr, beta g = Ra Pr, Delta T = 1): T = +1/2 on x = 0 (hot), -1/2 on x = 1, adiabatic y = 0, 1;
    no slip; T_ref = 0; velocities in units of alpha / L.  Corner vertices of the hot / cold walls take the wall value
    (the Dirichlet walls include their end points)."""
    zero = ngs.CF((0.0, 0.0))
    gT = lambda A, B: np.where(np.abs(A) < 1e-12, 0.5, np.where(np.abs(A - 1) < 1e-12, -0.5, 0.0))  # noqa: E731
    return BousProblem(mesh, geometry, form, 1.0, 1.0, Ra * Pr, 0.0, ngs.CF(Pr), ngs.CF(1.0), zero, ngs.CF(0.0),
                       ["x0", "x1", "y0", "y1"], lambda A, B: np.zeros(np.shape(A) + (2,)),
                       {"x0": "dirichlet", "x1": "dirichlet", "y0": "neumann_zero", "y1": "neumann_zero"}, gT,
                       mean_zero=True, supg=supg, tau_rule=tau_rule, tau_scale=tau_scale)
