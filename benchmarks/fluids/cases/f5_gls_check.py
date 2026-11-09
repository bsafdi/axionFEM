"""Stabilized P1/P1 (group F5; GLS: SUPG + PSPG + LSIC): the check module's computations (runner.check_module "f5_gls_check").

numpy/scipy/stdlib only; imports the planar Stokes and steady Navier-Stokes check modules read-only (meshes, quadrature, norms, Newton/Taylor diagnostics).
Nothing here calls the solver: tau, the stabilized residual vector, the patch-test and cavity functionals are computed
here from raw nodal arrays.

THE DISCRETE PROBLEM.  Unknowns (u, p) continuous P1 on an affine triangle mesh
(nodal basis N_a), constant density rho, viscosity mu(x) >= 0 (analytic or a P1 nodal field, grad mu elementwise):

  R_u,a = sum_K int [ rho (u_t + (w.grad)u).N_a e_i + a_visc(mu; u)(N_a e_i) - p d_i N_a - f_i N_a
                      + tau_m rho ((w.grad N_a) R_i) + tau_c rho (div u) d_i N_a ]
  R_p,a = sum_K int [ (div u) N_a + (tau_p / rho) grad N_a . R ]
  R     = rho (u_t + (w.grad)u) + grad p - div(sigma_visc) - f      (strong, element-wise; grad of P1 constant)
  a_visc lap: mu grad u : grad v;   sym: mu (grad u + grad u^T) : grad v     (= 2 mu eps(u) : eps(v))
  div(sigma_visc) lap_i = sum_j d_j mu d_j u_i;  sym_i = sum_j d_j mu (d_j u_i + d_i u_j)  (d^2 u = 0 on P1, grad mu KEPT)
  w = u (Navier-Stokes) | a given field (Oseen) | 0 (Stokes: the convective and SUPG terms are absent)
  u_t = (u - u^{n-1}) / dt (BDF1) or (3u - 4u^{n-1} + u^{n-2}) / (2 dt) (BDF2, constant step); 0 if steady
  tau_m = [ (2/dt)^2 + CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)       SUPG ((2/dt)^2 only if dt is given)
  tau_p = [ CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)                   PSPG and LSIC: NO dt term (the small-dt failure of a dt-dependent PSPG weight, and D1)
  tau_c = 1 / (tau_p tr G)           (Bazilevs et al. 2007 eq. (65) has g.g, g_i = sum_k dxi_k/dx_i: that depends on the
                                      vertex numbering; tr G does not)
  G_ij = sum_k dz_k/dx_i dz_k/dx_j, z = coordinates of the EQUILATERAL reference triangle (0,0), (1,0), (1/2, sqrt(3)/2)
  (Bazilevs eq. (66) with xi = z): invariant under the numbering of the element's vertices, unlike G of the unit RIGHT
  triangle, for which G changes with which vertex is mapped to the right angle.  C_I = 36, CG = 1.
Every element integral uses the fixed rule RULE (Dunavant-type, degree RULE_DEG, 6 points; weights sum to 1/2).
"""
from __future__ import annotations

import math

import numpy as np

import f1a_stokes_check as C1
import f2a_ns_check as C2

RULE_DEG = 4
C_I = 36.0
CG = 1.0

# Strang-Fix / Dunavant degree-4, 6-point rule on the unit triangle (weights sum to 1/2)
_A1, _W1 = 0.445948490915965, 0.223381589678011
_A2, _W2 = 0.091576213509771, 0.109951743655322


def rule_points():
    pts, w = [], []
    for a, wt in ((_A1, _W1), (_A2, _W2)):
        b = 1 - 2 * a
        for p in ((a, a), (a, b), (b, a)):
            pts.append(p)
            w.append(wt / 2)                 # the tabulated weights sum to 1; the triangle has area 1/2
    return np.array(pts), np.array(w)


# ============================================================================================== geometry
def element_geometry(V, T):
    """Per element: area, grad N_a [ne,3,2], G [ne,2,2], g [ne,2] (reference triangle (0,0),(1,0),(0,1))."""
    P = V[T]
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=-1)        # [ne,2,2], columns = edge vectors
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    Ji = np.empty_like(J)
    Ji[:, 0, 0], Ji[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
    Ji[:, 1, 0], Ji[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det        # Ji = d xi / d x (rows xi_k, cols x_j)
    refgrad = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])            # d N_a / d xi
    gradN = np.einsum("ak,ekj->eaj", refgrad, Ji)
    Mq = np.array([[1.0, 0.5], [0.5, 1.0]])                               # R^T R, R: unit right triangle -> equilateral
    G = np.einsum("eki,kl,elj->eij", Ji, Mq, Ji)
    g = Ji.sum(axis=1)                                                    # (kept for reference; not used by tau)
    return {"det": det, "area": 0.5 * np.abs(det), "gradN": gradN, "G": G, "g": g}


def tau_fields(V, T, w_q, mu_q, rho, dt=None, CI=C_I, cg=CG, geo=None):
    """tau_m, tau_c at the quadrature points: w_q [ne,nq,2] the advecting velocity, mu_q [ne,nq] the viscosity."""
    geo = geo or element_geometry(V, T)
    G, g = geo["G"], geo["g"]
    wGw = np.einsum("eqi,eij,eqj->eq", w_q, G, w_q)
    GG = np.einsum("eij,eij->e", G, G)[:, None]
    nu = mu_q / rho
    s = cg * wGw + CI * nu ** 2 * GG
    tp = 1.0 / np.sqrt(s)                       # steady tau: PSPG and LSIC (no dt term)
    if dt is not None:
        s = s + (2.0 / dt) ** 2
    tm = 1.0 / np.sqrt(s)                       # SUPG tau (with the (2/dt)^2 term)
    trG = np.einsum("eii->e", G)[:, None]
    tc = 1.0 / (tp * trG)
    return tm, tp, tc


# ============================================================================================== the residual
def residual(V, T, u, p, rho, mu, f, form="sym", convection=True, advect=None, dt=None, hist=(), bdf=1, stab=True,
             CI=C_I, cg=CG, nu_nodal=None):
    """The residual vector (R_u [nv,2], R_p [nv]) at the nodal state (u [nv,2], p [nv]).
    mu: callable (x, y) -> (mu, grad_mu[...,2]) at the quadrature points, or None with nu_nodal = nodal mu (P1);
    f: callable (x, y) -> [...,2]; advect: None (the state), a callable (x, y) -> [...,2] (Oseen), or the string
    'zero' (Stokes: no convection and no SUPG); hist = (u^{n-1}[, u^{n-2}]) nodal arrays."""
    q, wq = rule_points()
    geo = element_geometry(V, T)
    ne = len(T)
    N = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], axis=-1)       # [nq,3]
    X = np.einsum("qa,eaj->eqj", N, V[T])
    gradN = geo["gradN"]
    W = geo["area"][:, None] * 2 * wq[None, :]                              # weights (weights sum to 1/2; detJ = 2 area)
    ue, pe = u[T], p[T]                                                     # [ne,3,2], [ne,3]
    uq = np.einsum("qa,eai->eqi", N, ue)
    Gu = np.einsum("eai,eaj->eij", ue, gradN)[:, None, :, :] * np.ones((1, len(q), 1, 1))   # d_j u_i (constant per element)
    gp = np.einsum("ea,eaj->ej", pe, gradN)[:, None, :] * np.ones((1, len(q), 1))
    div = Gu[..., 0, 0] + Gu[..., 1, 1]
    if nu_nodal is not None:
        mq = np.einsum("qa,ea->eq", N, nu_nodal[T])
        gm = np.einsum("ea,eaj->ej", nu_nodal[T], gradN)[:, None, :] * np.ones((1, len(q), 1))
    else:
        mq, gm = mu(X[..., 0], X[..., 1])
    fq = f(X[..., 0], X[..., 1])
    if isinstance(advect, str) and advect == "zero":
        wv = np.zeros_like(uq)
        conv = np.zeros_like(uq)
    else:
        wv = uq if advect is None else advect(X[..., 0], X[..., 1])
        conv = np.einsum("eqij,eqj->eqi", Gu, wv) if convection else np.zeros_like(uq)
    if dt is None:
        ut = np.zeros_like(uq)
    elif bdf == 1:
        u1 = np.einsum("qa,eai->eqi", N, hist[0][T])
        ut = (uq - u1) / dt
    else:
        u1 = np.einsum("qa,eai->eqi", N, hist[0][T])
        u2 = np.einsum("qa,eai->eqi", N, hist[1][T])
        ut = (3 * uq - 4 * u1 + u2) / (2 * dt)
    if form == "lap":
        sdiv = np.einsum("eqij,eqj->eqi", Gu, gm)
    else:
        sdiv = np.einsum("eqij,eqj->eqi", Gu + np.swapaxes(Gu, -1, -2), gm)
    Rm = rho * (ut + conv) + gp - sdiv - fq
    tm, tp, tc = tau_fields(V, T, wv, mq, rho, dt, CI, cg, geo)
    # Galerkin part
    Ru = np.zeros((len(V), 2))
    Rp = np.zeros(len(V))
    # test v = N_a e_i
    gal_inertia = rho * (ut + conv)[:, :, None, :] * N[None, :, :, None] - fq[:, :, None, :] * N[None, :, :, None]   # [e,q,a,i]
    if form == "lap":
        vis = np.einsum("eq,eqij,eaj->eqai", mq, Gu, gradN)
    else:
        vis = np.einsum("eq,eqij,eaj->eqai", mq, Gu + np.swapaxes(Gu, -1, -2), gradN)
    pres = -np.einsum("eq,eai->eqai", pe_q(N, pe), gradN)
    loc = gal_inertia + vis + pres
    if stab:
        sup = tm[..., None, None] * rho * np.einsum("eqj,eaj,eqi->eqai", wv, gradN, Rm)
        lsic = tc[..., None, None] * rho * div[..., None, None] * gradN[:, None, :, :]
        loc = loc + sup + lsic
    Ru_loc = np.einsum("eq,eqai->eai", W, loc)
    cont = div[..., None] * N[None, :, :]
    if stab:
        cont = cont + (tp / rho)[..., None] * np.einsum("eaj,eqj->eqa", gradN, Rm)
    Rp_loc = np.einsum("eq,eqa->ea", W, cont)
    np.add.at(Ru, T.ravel(), Ru_loc.reshape(-1, 2))
    np.add.at(Rp, T.ravel(), Rp_loc.ravel())
    return Ru, Rp


def pe_q(N, pe):
    return np.einsum("qa,ea->eq", N, pe)


# ============================================================================================== norms (re-exports and P1 helpers)
norms = C2.norms
load_field = C2.load_field
structured_mesh = C2.structured_mesh
distorted_unit = C2.distorted_unit
rect_mesh = C2.rect_mesh
verify_mesh = C2.verify_mesh
taylor_test = C2.taylor_test
newton_order = C2.newton_order
tri_rule = C2.tri_rule


def graded_mesh(nx, ny, ratio):
    """Structured (nx x ny) mesh on [0,1]^2 with a geometric stretching in y (cell height ratio 'ratio' between the
    last and the first row), every cell split along the (1,1) diagonal except the two corner cells (no all-boundary
    triangle).  Aspect ratios up to ratio * (ny/nx) ... used for the anisotropy checks of tau and the patch tests."""
    xs = np.linspace(0, 1, nx + 1)
    r = ratio ** (1.0 / max(ny - 1, 1))
    h = r ** np.arange(ny)
    ys = np.concatenate([[0], np.cumsum(h) / h.sum()])
    V = np.array([[a, b] for b in ys for a in xs])
    idx = lambda i, j: j * (nx + 1) + i  # noqa: E731
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if (i == nx - 1 and j == 0) or (i == 0 and j == ny - 1):
                T += [[a, b, d], [b, c, d]]
            else:
                T += [[a, b, c], [a, c, d]]
    return V, np.array(T)


# ============================================================================================== patch fields (P1-representable)
class Patch:
    """Exact solution representable in P1/P1: u = A (x, y) + b (div u = 0), p = c . x + p0, mu = mu0 (1 + m . x),
    rho; transient: u(t) = (1 + t) u_s, p(t) = (1 + t) p_s (u_t = u_s constant: BDF1 and BDF2 differentiate it exactly).
    The body force follows from the strong equations (quadratic polynomial, integrated exactly by the fixed rule)."""

    def __init__(self, rho=1.25, mu0=0.02, m=(0.3, -0.2), A=((0.4, 0.7), (0.5, -0.4)), b=(0.3, -0.2), c=(0.8, -0.6),
                 p0=0.1):
        self.rho, self.mu0, self.m = rho, mu0, np.array(m)
        self.A, self.b, self.c, self.p0 = np.array(A), np.array(b), np.array(c), p0
        assert abs(self.A[0, 0] + self.A[1, 1]) < 1e-14, "div u = 0"

    def u(self, x, y, t=0.0):
        s = (1.0 + t)
        return s * np.stack([self.A[0, 0] * x + self.A[0, 1] * y + self.b[0], self.A[1, 0] * x + self.A[1, 1] * y + self.b[1]], -1)

    def p(self, x, y, t=0.0):
        return (1.0 + t) * (self.c[0] * x + self.c[1] * y + self.p0)

    def mu(self, x, y):
        m = self.mu0 * (1 + self.m[0] * x + self.m[1] * y)
        gm = np.stack([self.mu0 * self.m[0] * np.ones_like(x), self.mu0 * self.m[1] * np.ones_like(x)], -1)
        return m, gm

    def f(self, form, convection=True, t=0.0, transient=False, advect_exact=True):
        def fn(x, y):
            uu = self.u(x, y, t)
            s = 1.0 + t
            conv = np.einsum("ij,...j->...i", s * self.A, uu) if convection else 0.0
            ut = np.stack([self.A[0, 0] * x + self.A[0, 1] * y + self.b[0], self.A[1, 0] * x + self.A[1, 1] * y + self.b[1]], -1) if transient else 0.0
            _, gm = self.mu(x, y)
            Gu = s * self.A
            sd = Gu @ np.array([self.mu0 * self.m[0], self.mu0 * self.m[1]]) if form == "lap" else \
                (Gu + Gu.T) @ np.array([self.mu0 * self.m[0], self.mu0 * self.m[1]])
            gp = s * self.c
            return self.rho * (ut + conv) + gp - sd * np.ones_like(uu)
        return fn


# ============================================================================================== P1 evaluator for the norms
class NodalEval:
    """Evaluator protocol of f2a_ns_check for nodal P1 data (u [nv,2], p [nv]) on (V, T); used by the self-tests of
    the check module and the NGSolve reference scripts, and as the reference adapter the case runner may use."""

    def __init__(self, V, T, u, p):
        self.V, self.T, self.u, self.p = V, T, u, p
        self.geo = element_geometry(V, T)
        self.n_elements = len(T)

    def _N(self, ref):
        return np.stack([1 - ref[:, 0] - ref[:, 1], ref[:, 0], ref[:, 1]], axis=-1)

    def geometry(self, ref):
        N = self._N(ref)
        return np.einsum("qa,eaj->eqj", N, self.V[self.T]), np.repeat(np.abs(self.geo["det"])[:, None], len(ref), 1)

    def velocity(self, ref):
        N = self._N(ref)
        ue = self.u[self.T]
        uq = np.einsum("qa,eai->eqi", N, ue)
        D = np.einsum("eai,eaj->eij", ue, self.geo["gradN"])[:, None] * np.ones((1, len(ref), 1, 1))
        return uq, D

    def pressure(self, ref):
        return np.einsum("qa,ea->eq", self._N(ref), self.p[self.T])

    def pressure_gradient(self, ref):
        return np.einsum("ea,eaj->ej", self.p[self.T], self.geo["gradN"])[:, None] * np.ones((1, len(ref), 1))


# ============================================================================================== inf-sup with PSPG (independent of any solver)
def stokes_matrices(V, T, mu, rho=1.0, stab=False, CI=C_I):
    """Dense P1/P1 Stokes blocks on the free (non-Dirichlet) velocity DOFs: A (vector H1-type: mu grad u : grad v),
    B (div), the PSPG matrix Cp (tau/rho grad q . grad p, tau_m with w = 0), and M_p.  For the stability estimate."""
    nv = len(V)
    geo = element_geometry(V, T)
    gradN, area = geo["gradN"], geo["area"]
    tau = 1.0 / np.sqrt(CI * (mu / rho) ** 2 * np.einsum("eij,eij->e", geo["G"], geo["G"]))
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((nv, 2 * nv))
    Cp = np.zeros((nv, nv))
    Mp = np.zeros((nv, nv))
    for e in range(len(T)):
        t = T[e]
        K = area[e] * gradN[e] @ gradN[e].T
        for i in range(2):
            A[np.ix_(2 * t + i, 2 * t + i)] += mu * K
        for a in range(3):
            for b in range(3):
                for i in range(2):
                    B[t[a], 2 * t[b] + i] += area[e] / 3.0 * gradN[e, b, i]
        Cp[np.ix_(t, t)] += tau[e] / rho * K
        Mp[np.ix_(t, t)] += area[e] / 12.0 * (np.ones((3, 3)) + np.eye(3))
    return A, B, Cp, Mp


def inf_sup_p1p1(V, T, mu=1.0, stab=False, zero_rel=1e-10):
    """Smallest nonzero eigenvalue of the (stabilized) pressure Schur complement S q = lambda M_p q,
    S = B A^-1 B^T + (Cp if stab).  Returns beta (sqrt of the smallest eigenvalue above zero_rel * max) and the number
    of zero modes (the constant is one).  Dirichlet velocity DOFs (all boundary nodes) removed."""
    import scipy.linalg as sl
    A, B, Cp, Mp = stokes_matrices(V, T, mu, 1.0, stab)
    bnd = np.unique(np.array(sorted({k for e in C1.boundary_edges(T) for k in e})))
    free = np.array([2 * k + i for k in range(len(V)) if k not in set(bnd.tolist()) for i in range(2)])
    Af, Bf = A[np.ix_(free, free)], B[:, free]
    S = Bf @ np.linalg.solve(Af, Bf.T)
    if stab:
        S = S + Cp
    lam = sl.eigh(S, Mp, eigvals_only=True)
    lam = np.sort(lam)
    mx = lam[-1]
    zero = int(np.sum(lam <= zero_rel * mx))
    pos = lam[lam > zero_rel * mx]
    return {"beta": float(math.sqrt(pos[0])) if len(pos) else 0.0, "zero_modes": zero, "lam_max": float(mx)}


# ============================================================================================== time-dependent exact field
class TimeField:
    """The transient MMS field (tools/fluids_mms/out/gls_tr_sq.json, sha256-verified by the runner's load) fixed at time t;
    the same accessor protocol as f2a_ns_check.Field (u, grad_u, p, mu, f)."""

    def __init__(self, data: dict, t: float):
        import scipy.special as sc
        self.geometry, self.coords, self.rho, self.t = "planar", ("x", "y"), float(data["rho"]), float(t)
        npd = data["np"]

        def comp(code):
            c = compile(code, "<mms>", "eval")

            def f(A, B):
                v = eval(c, {"__builtins__": {}, "np": np, "sc": sc}, {"x": A, "y": B, "t": self.t})
                return np.broadcast_to(np.asarray(v, dtype=float), np.shape(A))
            return f
        self._u = [comp(s) for s in npd["u"]]
        self._g = [[comp(s) for s in row] for row in npd["grad_u"]]
        self._p = comp(npd["p"])
        self._mu = comp(npd["mu"])
        self._f = {k: [comp(s) for s in v] for k, v in npd["f"].items()}

    def u(self, A, B):
        return np.stack([g(A, B) for g in self._u], -1)

    def grad_u(self, A, B):
        return np.stack([np.stack([g(A, B) for g in row], -1) for row in self._g], -2)

    def p(self, A, B):
        return self._p(A, B)

    def mu(self, A, B):
        return self._mu(A, B)

    def f(self, form):
        fs = self._f[form]
        return lambda A, B: np.stack([g(A, B) for g in fs], -1)


# ============================================================================================== tau pins
def tau_pin_defects(V, T, w_nodal, mu, rho, dt, reported, CI=C_I, cg=CG):
    """Relative defects of the solver-reported tau arrays against tau_fields.  w_nodal [nv,2] (advecting velocity, P1),
    mu: a constant or a nodal array [nv] (P1 viscosity), reported = {"tau_supg": [ne,nq], "tau_pspg": [ne,nq], "tau_lsic":
    [ne,nq]} at the fixed rule points (rule_points order).  max over elements and points of |reported - expected| /
    expected."""
    q, _ = rule_points()
    N = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], axis=-1)
    wq = np.einsum("qa,eai->eqi", N, w_nodal[T])
    mq = np.einsum("qa,ea->eq", N, np.asarray(mu)[T]) if np.ndim(mu) else np.full((len(T), len(q)), float(mu))
    tm, tp, tc = tau_fields(V, T, wq, mq, rho, dt, CI, cg)
    out = {}
    for key, ref in (("tau_supg", tm), ("tau_pspg", tp), ("tau_lsic", tc)):
        out[key] = float(np.max(np.abs(np.asarray(reported[key]) - ref) / ref))
    return out


# ============================================================================================== cavity functionals from nodal data
def cavity_quantities(V, T, u, nu, N):
    """Kinetic energy, viscous dissipation (sym form, mu = nu, rho = 1) and the extremes of u_x on x = 1/2 and of u_y on
    y = 1/2 of the P1 field (u [nv,2] in the fixed vertex order; the extremes of a continuous piecewise-linear function on
    a mesh line are attained at mesh vertices, so they are exact).  N even: the centre lines carry vertices."""
    ev = NodalEval(V, T, u, np.zeros(len(V)))
    f = C2.functionals(ev, "sym", nu)
    vx = np.where(np.abs(V[:, 0] - 0.5) < 1e-12)[0]
    hy = np.where(np.abs(V[:, 1] - 0.5) < 1e-12)[0]
    a = vx[np.argmin(u[vx, 0])]
    b, c = hy[np.argmax(u[hy, 1])], hy[np.argmin(u[hy, 1])]
    return {"Ekin": f["Ekin"], "Diss": f["Diss"], "ux_min_vc": float(u[a, 0]), "y_of_ux_min": float(V[a, 1]),
            "uy_max_hc": float(u[b, 1]), "x_of_uy_max": float(V[b, 0]), "uy_min_hc": float(u[c, 1]), "x_of_uy_min": float(V[c, 0])}


# ============================================================================================== fields with grad mu; parity states
class GlsField(C2.Field):
    """f2a_ns_check.Field plus the viscosity gradient (data["f5"]["grad_mu_np"]) and the body force without convection."""

    def __init__(self, data: dict):
        super().__init__(data)
        cc = self.coords
        self._gmu = [C2._compile({"numpy": s}, cc) for s in data["f5"]["grad_mu_np"]]
        self._f0 = {form: [C2._compile(s, cc) for s in fd["f_without_convection"]] for form, fd in data["forms"].items()}

    def mu_and_grad(self, A, B):
        return self.mu(A, B), np.stack([g(A, B) for g in self._gmu], -1)

    def f_no_convection(self, form):
        fs = self._f0[form]
        return lambda A, B: np.stack([g(A, B) for g in fs], -1)


def load_gls_field(entry: dict, root=None) -> GlsField:
    import hashlib
    import json
    from pathlib import Path
    root = Path(root) if root is not None else C2._repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return GlsField(json.loads(raw))


PARITY_STATES = {
    "steady_ns_distorted": dict(field="gls_sq_mix", mesh="dist1", mode="ns", form="sym"),
    "steady_ns_conv_graded": dict(field="gls_sq_conv", mesh="graded", mode="ns", form="sym"),
    "steady_stokes": dict(field="gls_sq_mix", mesh="dist1", mode="stokes", form="sym"),
    "steady_oseen": dict(field="gls_sq_mix", mesh="dist1", mode="oseen", form="sym"),
    "steady_nodal_mu_lap": dict(field="gls_sq_mix", mesh="dist1", mode="ns", form="lap", nodal_mu=True),
    "transient_bdf1": dict(field="gls_sq_mix", mesh="dist1", mode="ns", form="sym", dt=0.05, bdf=1),
    "transient_bdf2": dict(field="gls_sq_mix", mesh="dist1", mode="ns", form="sym", dt=0.05, bdf=2),
}


def parity_state(name: str, fields: dict) -> dict:
    """The fixed nodal state of the residual-parity case: fields = {"gls_sq_mix": GlsField, "gls_sq_conv": GlsField}.
    Returns the mesh, u, p (and BDF history), the physics (mode, form, rho, dt, bdf) and the callables the numpy residual needs.
    u = (0.5 sin(3x + y) + 0.1 g, 0.4 cos(2y - x) + 0.1 g'), p = 0.3 cos(2x) + 0.1 g'' with g, g', g'' the first three standard-normal
    vectors of numpy PCG64 default_rng(1); history u^{n-1} = 0.3 (cos x, sin y), u^{n-2} = 0.5 u^{n-1}."""
    spec = PARITY_STATES[name]
    fld = fields[spec["field"]]
    V, T = graded_mesh(12, 12, 1e3) if spec["mesh"] == "graded" else C2.distorted_unit(1)
    nv = len(V)
    rng = np.random.default_rng(1)
    u = np.stack([0.5 * np.sin(3 * V[:, 0] + V[:, 1]) + 0.1 * rng.standard_normal(nv),
                  0.4 * np.cos(2 * V[:, 1] - V[:, 0]) + 0.1 * rng.standard_normal(nv)], -1)
    p = 0.3 * np.cos(2 * V[:, 0]) + 0.1 * rng.standard_normal(nv)
    form, mode = spec["form"], spec["mode"]
    dt, bdf = spec.get("dt"), spec.get("bdf", 1)
    h1 = h2 = None
    if dt is not None:
        h1 = 0.3 * np.stack([np.cos(V[:, 0]), np.sin(V[:, 1])], -1)
        h2 = 0.5 * h1
    nu_nodal = fld.mu(V[:, 0], V[:, 1]) if spec.get("nodal_mu") else None
    f = fld.f_no_convection(form) if mode == "stokes" else fld.f(form)
    advect = {"ns": None, "stokes": "zero", "oseen": (lambda x, y: fld.u(x, y))}[mode]
    return {"V": V, "T": T, "u": u, "p": p, "rho": fld.rho, "mu": None if nu_nodal is not None else fld.mu_and_grad, "nu_nodal": nu_nodal,
            "f": f, "form": form, "convection": mode != "stokes", "advect": advect, "dt": dt, "bdf": bdf,
            "hist": () if dt is None else ((h1,) if bdf == 1 else (h1, h2)), "mode": mode}


def parity_reference(name: str, fields: dict):
    s = parity_state(name, fields)
    return residual(s["V"], s["T"], s["u"], s["p"], s["rho"], s["mu"], s["f"], form=s["form"], convection=s["convection"], advect=s["advect"],
                    dt=s["dt"], hist=s["hist"], bdf=s["bdf"], nu_nodal=s["nu_nodal"])


def parity_defect(name: str, fields: dict, Ru, Rp) -> float:
    """max over the velocity and pressure rows of |R_solver - R_check| / max |R_check| (rows of all vertices, no Dirichlet elimination)."""
    ru, rp = parity_reference(name, fields)
    num = max(np.abs(np.asarray(Ru) - ru).max(), np.abs(np.asarray(Rp) - rp).max())
    return float(num / max(np.abs(ru).max(), np.abs(rp).max()))


# ============================================================================================== fixed tau-pin states
TAU_STATES = ("diffusive", "convective", "variable_viscosity", "anisotropic", "transient_dt1e-2", "transient_dt1e-6")


def tau_state(name: str) -> dict:
    """The fixed states of case gls_tau_pins: mesh (V, T), advecting velocity w (nodal P1 [nv,2]), viscosity mu (a constant or nodal
    [nv]), rho, dt (None = steady)."""
    w_of = lambda V: np.stack([np.sin(3 * V[:, 0]), np.cos(2 * V[:, 1])], -1)  # noqa: E731
    if name == "diffusive":
        V, T = C2.structured_mesh(8)
        return {"V": V, "T": T, "w": np.zeros((len(V), 2)), "mu": 1.0, "rho": 1.25, "dt": None}
    if name == "convective":
        V, T = C2.structured_mesh(8)
        return {"V": V, "T": T, "w": w_of(V), "mu": 1e-9, "rho": 1.0, "dt": None}
    if name == "variable_viscosity":
        V, T = C2.distorted_unit(2)
        return {"V": V, "T": T, "w": w_of(V), "mu": 10.0 ** (-4 + 3 * V[:, 0]), "rho": 1.0, "dt": None}
    if name == "anisotropic":
        V, T = graded_mesh(8, 8, 1e4)
        return {"V": V, "T": T, "w": np.tile([1.0, 0.3], (len(V), 1)), "mu": 1e-3, "rho": 1.0, "dt": None}
    if name in ("transient_dt1e-2", "transient_dt1e-6"):
        V, T = C2.structured_mesh(8)
        return {"V": V, "T": T, "w": w_of(V), "mu": 0.05, "rho": 1.0, "dt": float(name.split("dt")[1])}
    raise KeyError(name)


def tau_state_defects(name: str, reported: dict) -> dict:
    s = tau_state(name)
    return tau_pin_defects(s["V"], s["T"], s["w"], s["mu"], s["rho"], s["dt"], reported)


def renumber_triangles(T):
    """Every triangle's vertex list cyclically permuted by (element index mod 3) and reversed for odd elements (orientation flips: tau must
    not care)."""
    T = np.asarray(T)
    out = T.copy()
    for e in range(len(T)):
        k = e % 3
        out[e] = np.roll(T[e], k)
        if e % 2:
            out[e] = out[e][::-1]
    return out


def tau_renumber_defect(rep_a: dict, rep_b: dict) -> float:
    """Largest relative difference between two reported tau sets of the same mesh with renumbered triangles: the quadrature points of a
    renumbered element are a permutation of the original ones (the 6-point rule is invariant), so the six values of every element are
    compared after sorting."""
    worst = 0.0
    for k in ("tau_supg", "tau_pspg", "tau_lsic"):
        a, b = np.sort(np.asarray(rep_a[k]), axis=1), np.sort(np.asarray(rep_b[k]), axis=1)
        worst = max(worst, float(np.max(np.abs(a - b) / a)))
    return worst


def tau_reference_arrays(V, T, w_nodal, mu, rho, dt):
    """The numpy tau arrays {tau_supg, tau_pspg, tau_lsic} [ne, 6] of f5_gls_check.tau_fields at the nodal state."""
    q, _ = rule_points()
    N = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], axis=-1)
    wq = np.einsum("qa,eai->eqi", N, w_nodal[T])
    mq = np.einsum("qa,ea->eq", N, np.asarray(mu)[T]) if np.ndim(mu) else np.full((len(T), len(q)), float(mu))
    tm, tp, tc = tau_fields(V, T, wq, mq, rho, dt)
    return {"tau_supg": tm, "tau_pspg": tp, "tau_lsic": tc}


def th_deviation_ratio(q64: float, q128: float, ref: float) -> float:
    """|q_h(N=64) - ref| / |q_h(N=128) - ref|: the shrinkage of the P1/P1 deviation from the (fixed, run.parameters) Taylor-Hood value between two refinements."""
    return abs(q64 - ref) / max(abs(q128 - ref), 1e-300)
