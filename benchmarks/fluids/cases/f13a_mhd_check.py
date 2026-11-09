"""Inductionless MHD (group F13a): the check module's error computation (runner.check_module "f13a_mhd_check").

It imports f1a_stokes_check (quadrature and the structured/distorted families) and f13a_mhd_series (closed forms and
series).  Nothing here calls the solver: every norm, identity and check value is computed from raw point
evaluations of the discrete solution.

Evaluator protocol (provided by the case runner), ref_pts on the reference triangle
(0,0), (1,0), (0,1), arrays over [n_elements, n_points]:
    ev.n_elements
    ev.geometry(ref_pts)   -> (x[ne, nq, 2], detJ[ne, nq])                 affine (straight-sided) elements
    ev.velocity(ref_pts)   -> (u[ne, nq, 3], grad_u[ne, nq, 3, 2])          u = (u_x, u_y, u_z); grad_u[..., i, j] = d u_i/d x_j
    ev.pressure(ref_pts)   -> p[ne, nq]
    ev.potential(ref_pts)  -> (phi[ne, nq], grad_phi[ne, nq, 2])

Conventions: J = sigma(-grad phi + E_z e_z + u x B); the Lorentz force is J x B; j_n = J.n
with the outward normal; Ha = B0 a sqrt(sigma/mu).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import f1a_stokes_check as F1
import f13a_mhd_series as S

QUAD_DEGREE = 10
LX_STRIP = 0.25                 # length of the Hartmann strip [0, LX] x [-1, 1]
BASE_N = 16                     # intervals per direction of the graded base meshes


# ============================================================================================== meshes
def graded_nodes(n: int, a: float, delta: float) -> np.ndarray:
    """n (even) intervals on [-a, a], symmetric, clustered at both ends so that the first interval is delta (a layer
    thickness), by the tanh map d(s) = a (1 + tanh(kappa (s - 1))/tanh(kappa)), s = 2j/n, of the distance from the
    wall; uniform when delta >= 2a/n.  kappa by bisection (deterministic)."""
    h = n // 2
    if not math.isfinite(delta) or delta >= a / h:
        s = np.linspace(0.0, 1.0, h + 1)
        d = a * s
    else:
        def first(k):
            return a * (1 + math.tanh(k * (1.0 / h - 1)) / math.tanh(k))
        lo, hi = 1e-8, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if first(mid) > delta:
                lo = mid
            else:
                hi = mid
        kap = 0.5 * (lo + hi)
        s = np.linspace(0.0, 1.0, h + 1)
        d = a * (1 + np.tanh(kap * (s - 1)) / math.tanh(kap))
        d[0], d[-1] = 0.0, a
    left = -a + d
    right = (a - d)[::-1]
    return np.r_[left[:-1], 0.0, right[1:]]


def graded_rect(xn, yn, k: int):
    """Tensor-product base mesh on the node sets xn, yn (cells split along the (1,1) diagonal, the two corner cells
    (x_max, y_min) and (x_min, y_max) the other way: Taylor-Hood corner condition), then k uniform red refinements
    (NESTED; the grading is fixed by the base, every refinement halves every edge)."""
    V, T = F1._structured(len(xn) - 1, len(yn) - 1)
    ix = np.rint(V[:, 0] * (len(xn) - 1)).astype(int)
    iy = np.rint(V[:, 1] * (len(yn) - 1)).astype(int)
    V = np.c_[np.asarray(xn)[ix], np.asarray(yn)[iy]]
    for _ in range(k):
        V, T = F1.red_refine(V, T)
    return V, T


def layer_widths(Ha: float):
    """(Hartmann layer, side layer) thicknesses a/Ha, a/sqrt(Ha) (a = 1); infinite for Ha = 0."""
    if Ha <= 0:
        return math.inf, math.inf
    return 1.0 / Ha, 1.0 / math.sqrt(Ha)


def hartmann_strip_mesh(Ha: float, k: int):
    """[0, LX] x [-1, 1]: 2 base intervals in x (uniform), BASE_N graded in y (first interval = a/Ha), k red
    refinements."""
    dH, _ = layer_widths(Ha)
    return graded_rect(np.linspace(0.0, LX_STRIP, 3), graded_nodes(BASE_N, 1.0, dH), k)


def duct_mesh(Ha: float, k: int):
    """[-1, 1]^2: BASE_N graded intervals per direction (x: side layers a/sqrt(Ha); y: Hartmann layers a/Ha), k red
    refinements."""
    dH, dS = layer_widths(Ha)
    return graded_rect(graded_nodes(BASE_N, 1.0, dS), graded_nodes(BASE_N, 1.0, dH), k)


def mms_mesh(family: str, level_param: int):
    if family == "structured":
        return F1.structured_mesh(level_param)
    if family == "distorted":
        return F1.distorted_mesh(level_param)
    raise ValueError(family)


verify_mesh = F1.verify_mesh
mesh_hash = F1.mesh_hash


# ============================================================================================== exact fields
def _compile(spec):
    return F1._compile(spec)


class MHDField:
    """Exact solution and data of one generated MMS field (tools/fluids_mms/out/mhd_*.json), sha256-verified."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self._u = [_compile(s) for s in data["u"]]
        self._g = [[_compile(s) for s in row] for row in data["grad_u"]]
        self.p = _compile(data["p"])
        self.phi = _compile(data["phi"])
        self._gphi = [_compile(s) for s in data["grad_phi"]]
        self.sigma = _compile(data["sigma"])
        self._B = [_compile(s) for s in data["B"]]
        self._J = [_compile(s) for s in data["J"]]
        self.q = _compile(data["q"])
        self._f = {rho: [_compile(s) for s in fr] for rho, fr in data["f"].items()}
        self._jn = {e: _compile(b["jn"]) for e, b in data["boundaries"].items()}
        self.phi_bc = {e: b["phi_bc"] for e, b in data["boundaries"].items()}
        self.constants = data["constants"]
        self.mu = float(self.constants["mu"])
        self.Ez = float(self.constants["E_z"])
        self.Gz = float(self.constants["G_z"])

    @staticmethod
    def _stack(fns, X, Y):
        return np.stack([f(X, Y) for f in fns], axis=-1)

    def u(self, X, Y):
        return self._stack(self._u, X, Y)

    def grad_u(self, X, Y):
        return np.stack([np.stack([self._g[i][j](X, Y) for j in range(2)], axis=-1) for i in range(3)], axis=-2)

    def grad_phi(self, X, Y):
        return self._stack(self._gphi, X, Y)

    def B(self, X, Y):
        return self._stack(self._B, X, Y)

    def J(self, X, Y):
        return self._stack(self._J, X, Y)

    def f(self, rho):
        key = str(float(rho)) if str(float(rho)) in self._f else str(int(rho))
        fns = self._f[key]
        return lambda X, Y: self._stack(fns, X, Y)

    def jn(self, edge):
        return self._jn[edge]


def load_field(entry: dict, root: Path | None = None) -> MHDField:
    root = Path(root) if root is not None else F1._repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return MHDField(json.loads(raw))


class HartmannExact:
    """Hartmann channel (A1) as an exact field on the strip: mode 'inplane' (u = (U,0,0), J = (0,0,J_z)) or 'duct'
    (u = (0,0,W), phi = -E_x x, J = (J_x,0,0)); uniform sigma, B = (0, B0, 0)."""

    def __init__(self, mu, sigma, B0, G, mode, circuit):
        self.h = S.Hartmann(mu, sigma, B0, G, 1.0, mode, circuit)
        self.mode = mode
        self.mu, self.sigma_v, self.B0 = mu, sigma, B0
        self.Ez = self.h.E if mode == "inplane" else 0.0

    def u(self, X, Y):
        U = self.h.U(Y)
        z = np.zeros_like(U)
        return np.stack([U, z, z] if self.mode == "inplane" else [z, z, U], axis=-1)

    def grad_u(self, X, Y):
        dU = self.h.dU(Y)
        G = np.zeros(np.shape(Y) + (3, 2))
        G[..., 0 if self.mode == "inplane" else 2, 1] = dU
        return G

    def p(self, X, Y):
        if self.mode == "inplane":
            return self.h.G * (LX_STRIP - X)
        return np.zeros_like(X)

    def phi(self, X, Y):
        return np.zeros_like(X) if self.mode == "inplane" else -self.h.E * X

    def grad_phi(self, X, Y):
        g = np.zeros(np.shape(X) + (2,))
        if self.mode == "duct":
            g[..., 0] = -self.h.E
        return g

    def sigma(self, X, Y):
        return np.full(np.shape(X), self.sigma_v)

    def B(self, X, Y):
        B = np.zeros(np.shape(X) + (3,))
        B[..., 1] = self.B0
        return B

    def J(self, X, Y):
        J = np.zeros(np.shape(X) + (3,))
        J[..., 2 if self.mode == "inplane" else 0] = self.h.current(Y)
        return J


class HartmannTwoLayerExact(HartmannExact):
    """In-plane Hartmann channel with sigma = s1 (y < 0), s2 (y > 0), short circuit (A1 variant 'twolayer'): the
    non-MMS case of the variable-conductivity row (the interface y = 0 is a mesh line of every level)."""

    def __init__(self, mu, s1, s2, B0, G):
        self.h = S.HartmannTwoLayer(mu, s1, s2, B0, G, 1.0)
        self.h.E = 0.0
        self.mode = "inplane"
        self.mu, self.B0, self.Ez = mu, B0, 0.0
        self.sigma_v = None

    def sigma(self, X, Y):
        return np.asarray(self.h.sigma(Y), float) + 0.0 * X

    def J(self, X, Y):
        J = np.zeros(np.shape(X) + (3,))
        J[..., 2] = self.h.sigma(Y) * self.B0 * self.h.U(Y)
        return J


# ============================================================================================== quadrature, J_h
def _weights(ev, deg=QUAD_DEGREE):
    q, w = F1.tri_rule(deg)
    xq, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    return q, xq, w[None, :] * detJ


def cross(a, b):
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def current_h(uh, gphih, sig, B, Ez):
    """J_h = sigma(-grad phi_h + E_z e_z + u_h x B) at the given points (the check module's own Ohm's law)."""
    uxB = cross(uh, B)
    J = sig[..., None] * uxB
    J[..., 0] -= sig * gphih[..., 0]
    J[..., 1] -= sig * gphih[..., 1]
    J[..., 2] += sig * Ez
    return J


def _mean(W, v):
    return float((W * v).sum() / W.sum())


# ============================================================================================== norms
def norms(ev, ex, pressure="mean_free", phi="determined", deg=QUAD_DEGREE) -> dict:
    """uL2, uH1 (all 6 gradient components), pL2 (mean-free or determined), phiL2, phiH1 (mean-free when the
    potential is determined only up to a constant), JL2 = ||J_h - J|| with J_h from the check module's Ohm's law
    (exact sigma, B, E_z), divL2, plus the magnitudes the round-off floors refer to."""
    q, xq, W = _weights(ev, deg)
    X, Y = xq[..., 0], xq[..., 1]
    uh, guh = ev.velocity(q)
    ph = ev.pressure(q)
    phih, gphih = ev.potential(q)
    ue, ge, pe = ex.u(X, Y), ex.grad_u(X, Y), ex.p(X, Y)
    phie, gphie = ex.phi(X, Y), ex.grad_phi(X, Y)
    sig, B = ex.sigma(X, Y), ex.B(X, Y)
    Jh = current_h(uh, gphih, sig, B, ex.Ez)
    Je = ex.J(X, Y)
    out = {"uL2": math.sqrt(float((W[..., None] * (uh - ue) ** 2).sum())),
           "uH1": math.sqrt(float((W[..., None, None] * (guh - ge) ** 2).sum())),
           "divL2": math.sqrt(float((W * (guh[..., 0, 0] + guh[..., 1, 1]) ** 2).sum())),
           "JL2": math.sqrt(float((W[..., None] * (Jh - Je) ** 2).sum())),
           "phiH1": math.sqrt(float((W[..., None] * (gphih - gphie) ** 2).sum())),
           "u_magnitude": float(np.max(np.linalg.norm(ue, axis=-1))),
           "J_magnitude": float(np.max(np.linalg.norm(Je, axis=-1))),
           "gradphi_magnitude": float(np.max(np.linalg.norm(gphie, axis=-1))),
           "area": float(W.sum())}
    if pressure == "mean_free":
        d = (ph - _mean(W, ph)) - (pe - _mean(W, pe))
        out["p_magnitude"] = float(np.max(np.abs(pe - _mean(W, pe))))
    else:
        d = ph - pe
        out["p_magnitude"] = float(np.max(np.abs(pe)))
    out["pL2"] = math.sqrt(float((W * d ** 2).sum()))
    if phi == "mean_free":
        d = (phih - _mean(W, phih)) - (phie - _mean(W, phie))
        out["phi_magnitude"] = float(np.max(np.abs(phie - _mean(W, phie))))
    else:
        d = phih - phie
        out["phi_magnitude"] = float(np.max(np.abs(phie)))
    out["phiL2"] = math.sqrt(float((W * d ** 2).sum()))
    out["uH1_magnitude"] = float(np.max(np.linalg.norm(ge.reshape(ge.shape[:-2] + (6,)), axis=-1)))
    return out


# ============================================================================================== conservation
def integrals(ev, sigma: float, B, Ez: float, mu: float, deg=QUAD_DEGREE) -> dict:
    """Uniform sigma and B: the discrete integrals of the conservation identities:
    net current int J_h (in-plane), Lorentz work W_L = int (J_h x B).u_h, Joule D_J = int |J_h|^2/sigma, axial current
    I_z = int J_h,z, viscous dissipation D_mu = mu |u_h|_1^2, flow rates int u_h, and int J_h.grad phi_h."""
    q, xq, W = _weights(ev, deg)
    uh, guh = ev.velocity(q)
    phih, gphih = ev.potential(q)
    Bv = np.broadcast_to(np.asarray(B, float), uh.shape)
    sig = np.full(W.shape, float(sigma))
    Jh = current_h(uh, gphih, sig, Bv, Ez)
    F = cross(Jh, Bv)
    jmag = np.linalg.norm(Jh, axis=-1)
    gmag = np.linalg.norm(gphih, axis=-1)
    uxb = np.linalg.norm(cross(uh, Bv)[..., :2], axis=-1)
    return {"net_Jx": float((W * Jh[..., 0]).sum()), "net_Jy": float((W * Jh[..., 1]).sum()),
            # scales of the two in-plane terms whose cancellation forms J (sigma grad phi, sigma u x B)
            "scale_lj": float((W * sigma * (gmag ** 2 + uxb ** 2)).sum()),
            "scale_net": float((W * sigma * (gmag + uxb)).sum()),
            "int_absJ": float((W * jmag).sum()),
            "lorentz_work": float((W[..., None] * F * uh).sum()),
            "lorentz_force": [float((W * F[..., i]).sum()) for i in range(3)],
            "joule": float((W * jmag ** 2).sum() / sigma),
            "I_z": float((W * Jh[..., 2]).sum()),
            "J_dot_gradphi": float((W * (Jh[..., 0] * gphih[..., 0] + Jh[..., 1] * gphih[..., 1])).sum()),
            "viscous": mu * float((W[..., None, None] * guh ** 2).sum()),
            "flow": [float((W * uh[..., i]).sum()) for i in range(3)],
            "area": float(W.sum())}


def lorentz_joule_defect(I: dict, Ez: float) -> float:
    """|W_L + D_J - E_z I_z| / int sigma(|grad phi_h|^2 + |(u_h x B)_perp|^2).  Pointwise (J x B).u =
    -J.(J/sigma + grad phi - E_z e_z), so the numerator is |int J_h.grad phi_h|, which vanishes for the Galerkin
    solution when phi_h is itself a test function (no source, insulating or homogeneous-Dirichlet potential walls).
    Revision 2: normalized by the CANCELLING terms, not by D_J, which shrinks with the core current at
    high Ha and amplified LU round-off by up to 1e4.  Returns 0 when the scale is 0 (B = 0: J_h = 0 exactly)."""
    if I["scale_lj"] == 0.0:
        return 0.0
    return abs(I["lorentz_work"] + I["joule"] - Ez * I["I_z"]) / I["scale_lj"]


def power_balance_defect(I: dict, G: float, axis: int = 2) -> float:
    """Ducts and strips with homogeneous Dirichlet velocity, f = 0 and u_perp = 0: testing the discrete momentum
    equation with v = u_h gives  mu |u_h|_1^2 = G int u_h,axis + W_L  exactly.  Returns the defect relative to the
    input power G int u_h."""
    Pin = G * I["flow"][axis]
    return abs(I["viscous"] - Pin - I["lorentz_work"]) / abs(Pin)


def net_current_rel(I: dict) -> float:
    """max(|int J_h,x|, |int J_h,y|) / int sigma(|grad phi_h| + |(u_h x B)_perp|): zero for the Galerkin solution
    with insulating walls everywhere, uniform sigma and no source (test functions psi = x, y are in P2).  Revision 2
: normalized by the cancelling terms; 0 when J_h = 0 identically (B = 0)."""
    if I["scale_net"] == 0.0:
        return 0.0
    return max(abs(I["net_Jx"]), abs(I["net_Jy"])) / I["scale_net"]


_P2_REF = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])


def _p2_grad_ref(q):
    """Gradients of the 6 P2 Lagrange basis functions at reference points q[nq, 2]: [nq, 6, 2] (order: 3 vertices,
    then the midpoints of edges 01, 12, 20)."""
    s, t = q[:, 0], q[:, 1]
    l0 = 1 - s - t
    g = np.zeros((len(q), 6, 2))
    g[:, 0] = np.c_[-(4 * l0 - 1), -(4 * l0 - 1)]
    g[:, 1] = np.c_[4 * s - 1, 0 * s]
    g[:, 2] = np.c_[0 * s, 4 * t - 1]
    g[:, 3] = np.c_[4 * (l0 - s), -4 * s]
    g[:, 4] = np.c_[4 * t, 4 * s]
    g[:, 5] = np.c_[-4 * t, 4 * (l0 - t)]
    return g


def weak_current_residual(ev, sigma_fn, B_fn, Ez: float, q_fn=None, jn_edges=None, dirichlet=None,
                          deg=QUAD_DEGREE) -> dict:
    """The discrete current-conservation residual recomputed by the check module, independently of the solver:
    for every P2 Lagrange basis function N_i of the potential space that is NOT on a potential-Dirichlet boundary,
        r_i = int J_h.grad N_i + int q N_i - int_{Gamma_jn} j_n N_i        (Galerkin: r_i = 0).
    dirichlet(xy) -> bool marks potential-Dirichlet nodes; jn_edges: list of (predicate(xy) -> bool, j_n(X, Y),
    normal).  Returns max |r_i| / max_i int |J_h||grad N_i| (round-off for exactly integrated data) and the count."""
    q, w = F1.tri_rule(deg)
    xq, detJ = ev.geometry(q)
    xv, _ = ev.geometry(_P2_REF)
    uh, _ = ev.velocity(q)
    _, gphih = ev.potential(q)
    X, Y = xq[..., 0], xq[..., 1]
    sig = np.broadcast_to(np.asarray(sigma_fn(X, Y), float), X.shape)
    Jh = current_h(uh, gphih, sig, B_fn(X, Y), Ez)[..., :2]
    W = w[None, :] * detJ
    # affine map: physical gradient = inv(Jac)^T ref gradient
    a0, a1, a2 = xv[:, 0], xv[:, 1], xv[:, 2]
    Jac = np.stack([a1 - a0, a2 - a0], axis=-1)                     # [ne, 2(x,y), 2(s,t)]
    Jinv = np.linalg.inv(Jac)                                        # [ne, 2(s,t), 2(x,y)]
    gref = _p2_grad_ref(q)                                           # [nq, 6, 2(s,t)]
    gphys = np.einsum("qks,esx->eqkx", gref, Jinv)                   # [ne, nq, 6, 2]
    contrib = np.einsum("eq,eqx,eqkx->ek", W, Jh, gphys)
    # normalized by the cancelling terms sigma(|grad phi_h| + |(u_h x B)_perp|)
    Bq = np.broadcast_to(np.asarray(B_fn(X, Y), float), uh.shape)
    canc = sig * (np.linalg.norm(gphih, axis=-1) + np.linalg.norm(cross(uh, Bq)[..., :2], axis=-1))
    mag = np.einsum("eq,eq,eqk->ek", W, canc, np.linalg.norm(gphys, axis=-1))
    if q_fn is not None:
        s, t = q[:, 0], q[:, 1]
        l0 = 1 - s - t
        Nref = np.c_[l0 * (2 * l0 - 1), s * (2 * s - 1), t * (2 * t - 1), 4 * s * l0, 4 * s * t, 4 * t * l0]
        qv = np.broadcast_to(np.asarray(q_fn(X, Y), float), X.shape)
        contrib += np.einsum("eq,eq,qk->ek", W, qv, Nref)
    keys = np.round(xv, 10) + 0.0
    index = {}
    gid = np.empty(keys.shape[:2], dtype=np.int64)
    for e in range(keys.shape[0]):
        for k in range(6):
            kk = (keys[e, k, 0], keys[e, k, 1])
            gid[e, k] = index.setdefault(kk, len(index))
    r = np.zeros(len(index))
    m = np.zeros(len(index))
    np.add.at(r, gid.ravel(), contrib.ravel())
    np.add.at(m, gid.ravel(), mag.ravel())
    coords = np.array(list(index.keys()))
    if jn_edges:
        gp, gw = np.polynomial.legendre.leggauss(12)
        sp_ = 0.5 * (gp + 1)
        for e in range(xv.shape[0]):
            for (i, j, mid) in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
                Pa, Pb, Pm = xv[e, i], xv[e, j], xv[e, mid]
                for pred, jfn, nrm in jn_edges:
                    if pred(Pa) and pred(Pb) and pred(Pm):
                        L = float(np.hypot(*(Pb - Pa)))
                        pts = Pa[None, :] + sp_[:, None] * (Pb - Pa)[None, :]
                        jv = np.asarray(jfn(pts[:, 0], pts[:, 1]), float)
                        Na = (1 - sp_) * (1 - 2 * sp_)
                        Nb = sp_ * (2 * sp_ - 1)
                        Nm = 4 * sp_ * (1 - sp_)
                        for node, Nv in ((gid[e, i], Na), (gid[e, j], Nb), (gid[e, mid], Nm)):
                            r[node] -= 0.5 * L * float(np.sum(gw * jv * Nv))
    free = np.ones(len(index), bool)
    if dirichlet is not None:
        free = np.array([not dirichlet(c) for c in coords])
    mmax = float(np.max(m[free]))
    return {"max_rel": float(np.max(np.abs(r[free])) / mmax) if mmax > 0 else 0.0, "n_free": int(free.sum()),
            "n_nodes": int(len(index))}


def weak_axial_momentum_residual(ev, mu: float, sigma: float, B, Gz: float, Ez: float = 0.0, dirichlet=None,
                                 deg=QUAD_DEGREE) -> dict:
    """The axial (z) momentum row rebuilt by the check module for
    every P2 basis function N_i that is not a velocity-Dirichlet node (dirichlet(xy) -> bool):
        r_i = mu int grad u_h,z . grad N_i - int (G_z + (J_h x B)_z) N_i          (Galerkin: r_i = 0)
    with J_h = sigma(-grad phi_h + E_z e_z + u_h x B) from the check module's own Ohm's law.  The z row has no
    pressure and no convection (z-invariance with u_perp = 0 in the duct cases), and for uniform sigma, B and G_z every
    integrand is a polynomial of degree <= 4 per element.  Together with weak_current_residual it certifies that the
    Lorentz force uses the current the potential row conserves.  Normalized locally by max_i int (|mu grad u_z .
    grad N_i| + |G_z N_i| + |(J_h x B)_z N_i|) (terms that do not cancel).  Valid for f = 0 and rho (u.grad)u_z = 0."""
    q, w = F1.tri_rule(deg)
    xq, detJ = ev.geometry(q)
    W = w[None, :] * detJ
    uh, guh = ev.velocity(q)
    _, gphih = ev.potential(q)
    Bv = np.broadcast_to(np.asarray(B, float), uh.shape)
    Jh = current_h(uh, gphih, np.full(W.shape, float(sigma)), Bv, Ez)
    Fz = cross(Jh, Bv)[..., 2]
    xv, _ = ev.geometry(_P2_REF)
    a0 = xv[:, 0]
    Jinv = np.linalg.inv(np.stack([xv[:, 1] - a0, xv[:, 2] - a0], -1))
    gphys = np.einsum("qks,esx->eqkx", _p2_grad_ref(q), Jinv)
    s_, t_ = q[:, 0], q[:, 1]
    l0 = 1 - s_ - t_
    Nref = np.c_[l0 * (2 * l0 - 1), s_ * (2 * s_ - 1), t_ * (2 * t_ - 1), 4 * s_ * l0, 4 * s_ * t_, 4 * t_ * l0]
    visc_q = mu * np.einsum("eq,eqx,eqkx->eqk", W, guh[..., 2, :], gphys)
    contrib = visc_q.sum(1) - np.einsum("eq,eq,qk->ek", W, Gz + Fz, Nref)
    mag = np.abs(visc_q).sum(1) + np.einsum("eq,eq,qk->ek", W, np.abs(Gz) + np.abs(Fz), np.abs(Nref))
    keys = np.round(xv, 10) + 0.0
    index, gid = {}, np.empty(keys.shape[:2], np.int64)
    for e in range(keys.shape[0]):
        for k in range(6):
            gid[e, k] = index.setdefault((keys[e, k, 0], keys[e, k, 1]), len(index))
    r = np.zeros(len(index))
    mg = np.zeros(len(index))
    np.add.at(r, gid.ravel(), contrib.ravel())
    np.add.at(mg, gid.ravel(), mag.ravel())
    coords = np.array(list(index.keys()))
    free = np.ones(len(index), bool) if dirichlet is None else np.array([not dirichlet(c) for c in coords])
    return {"max_rel": float(np.max(np.abs(r[free])) / np.max(mg[free])), "n_free": int(free.sum())}


def core_J_exact(kind: str, mu: float, sigma: float, B0: float, G: float):
    """Revision 2: the FIXED definition of the exact core current used by core_current_error:
    Hunt -> f13a_mhd_series.hunt_J_fast(Hunt(mu, sigma, B0, G, K=4000), X, Y, K=400) (Hartmann part in closed form);
    Shercliff -> f13a_mhd_series.shercliff_J_fast(Shercliff(mu, sigma, B0, G, K=8000), X, Y) (default K =
    4 m a/pi + 400).  Both agree with plain series sums and with an independent series to <= 7e-9 relative."""
    if kind == "hunt":
        h = S.Hunt(mu, sigma, B0, G, K=4000)
        return lambda X, Y: S.hunt_J_fast(h, X, Y, 400)
    if kind == "shercliff":
        sh = S.Shercliff(mu, sigma, B0, G, K=8000)
        return lambda X, Y: S.shercliff_J_fast(sh, X, Y)
    raise ValueError(kind)


# ============================================================================================== point and wall values
def point_value(ev, xy, what="w"):
    """Value of the discrete solution at a physical point that is a mesh VERTEX (all fixed probe points are
    vertices of every level): w = u_z, ux = u_x, phi."""
    xv, _ = ev.geometry(_P2_REF[:3])
    for v in range(3):
        d = np.hypot(xv[:, v, 0] - xy[0], xv[:, v, 1] - xy[1])
        e = int(np.argmin(d))
        if d[e] < 1e-12:
            ref = _P2_REF[v:v + 1]
            if what == "phi":
                return float(ev.potential(ref)[0][e, 0])
            u, _ = ev.velocity(ref)
            return float(u[e, 0, {"ux": 0, "uy": 1, "w": 2}[what]])
    raise ValueError(f"probe point {xy} is not a mesh vertex")


def wall_current_moment_direct(ev, sigma: float, B, a=1.0, L=1.0, n_gauss=10) -> float:
    """M_direct = int_{y=a} J_h.n (x/L) dx by Gauss quadrature on the boundary edges (one-sided traces of grad
    phi_h): the 'surface' current computation (converges like the gradient, O(h^2))."""
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    s = 0.5 * (gp + 1)
    tot = 0.0
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        Ri, Rj = _P2_REF[i], _P2_REF[j]
        ref = Ri[None, :] + s[:, None] * (Rj - Ri)[None, :]
        xq, _ = ev.geometry(ref)
        on = np.all(np.abs(xq[..., 1] - a) < 1e-12, axis=1)
        if not np.any(on):
            continue
        u, _ = ev.velocity(ref)
        _, gphi = ev.potential(ref)
        Bv = np.broadcast_to(np.asarray(B, float), u.shape)
        J = current_h(u, gphi, np.full(u.shape[:2], float(sigma)), Bv, 0.0)
        for e in np.where(on)[0]:
            Lk = abs(xq[e, -1, 0] - xq[e, 0, 0]) / abs(s[-1] - s[0])
            tot += 0.5 * Lk * float(np.sum(gw * J[e, :, 1] * xq[e, :, 0] / L))
    return tot


def wall_current_moment_variational(ev, sigma: float, B, a=1.0, L=1.0, deg=QUAD_DEGREE) -> float:
    """M_var = int_Omega J_h.grad chi, chi = (x/L)(y + a)/(2a) (a P2 function, = x/L on y = a, 0 on y = -a): the
    discrete normal current as the residual functional of the potential equation (Babuska-Miller).  It depends only
    on chi on the potential-Dirichlet walls (Galerkin), and for the exact J equals int_{y=a} J.n x/L dx (insulating
    side walls, no source).  Expected to superconverge relative to the direct trace."""
    q, xq, W = _weights(ev, deg)
    u, _ = ev.velocity(q)
    _, gphi = ev.potential(q)
    Bv = np.broadcast_to(np.asarray(B, float), u.shape)
    J = current_h(u, gphi, np.full(W.shape, float(sigma)), Bv, 0.0)
    X, Y = xq[..., 0], xq[..., 1]
    gx = (Y + a) / (2 * a * L)
    gy = X / (2 * a * L)
    return float((W * (J[..., 0] * gx + J[..., 1] * gy)).sum())


def core_current_error(ev, sigma: float, B, Jexact, half=(0.5, 0.5), deg=6) -> dict:
    """Spurious-current measure (Ni et al. 2007): ||J_h - J||_L2(core) / ||J||_L2(core) over the elements whose
    centroid lies in the core |x| < half[0], |y| < half[1]; J_h from the check module's Ohm's law, J from the series
    (Jexact(X, Y) -> (J_x, J_y))."""
    q, xq, W = _weights(ev, deg)
    cen, _ = ev.geometry(np.array([[1 / 3, 1 / 3]]))
    core = (np.abs(cen[:, 0, 0]) < half[0]) & (np.abs(cen[:, 0, 1]) < half[1])
    u, _ = ev.velocity(q)
    _, gphi = ev.potential(q)
    Bv = np.broadcast_to(np.asarray(B, float), u.shape)
    J = current_h(u, gphi, np.full(W.shape, float(sigma)), Bv, 0.0)[core][..., :2]
    X, Y = xq[core][..., 0], xq[core][..., 1]
    Jx, Jy = Jexact(X, Y)
    Je = np.stack([Jx, Jy], axis=-1)
    Wc = W[core]
    num = math.sqrt(float((Wc[..., None] * (J - Je) ** 2).sum()))
    den = math.sqrt(float((Wc[..., None] * Je ** 2).sum()))
    return {"rel": num / den, "abs": num, "norm": den, "n_core_elements": int(core.sum())}


def relative_difference(ev_a, ev_b, scale_phi_b: float = 1.0, deg=QUAD_DEGREE) -> dict:
    """||u_a - u_b|| / ||u_a|| and ||phi_a - s phi_b|| / ||phi_a|| (both potentials mean-free) on the same mesh:
    the scaling identity (sigma, B) -> (4 sigma, B/2) leaves u_h unchanged and halves phi_h (s = 2)."""
    q, xq, W = _weights(ev_a, deg)
    ua, _ = ev_a.velocity(q)
    ub, _ = ev_b.velocity(q)
    pa, _ = ev_a.potential(q)
    pb, _ = ev_b.potential(q)
    pa = pa - _mean(W, pa)
    pb = scale_phi_b * (pb - _mean(W, pb))
    out = {"u_rel": math.sqrt(float((W[..., None] * (ua - ub) ** 2).sum()) / float((W[..., None] * ua ** 2).sum()))}
    den = float((W * pa ** 2).sum())
    out["phi_rel"] = math.sqrt(float((W * (pa - pb) ** 2).sum()) / den) if den > 0 else math.sqrt(float((W * pb ** 2).sum()))
    return out


def potential_spread(ev, deg=4) -> dict:
    """max |phi_h - mean| and max |J_h,perp| proxies (|grad phi_h|) at quadrature points: B = 0 / in-plane Hartmann
    checks (the discrete potential must be constant to round-off)."""
    q, xq, W = _weights(ev, deg)
    ph, g = ev.potential(q)
    return {"phi_spread": float(np.max(np.abs(ph - _mean(W, ph)))), "gradphi_max": float(np.max(np.abs(g)))}
