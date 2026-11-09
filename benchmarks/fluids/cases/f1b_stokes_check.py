"""Axisymmetric Stokes with swirl (group F1b): the check module's error computation (runner.check_module "f1b_stokes_check").

It imports f1a_stokes_check for the mesh hash/verification and nodal-data helpers.  The case runner
(cases/f1b_stokes.py) builds the fixed meshes with the builders below, solves through the public fluids API, wraps
the discrete solution in an Evaluator and calls the functions here.  Nothing here calls the solver.

Evaluator protocol; meridional coordinates (r, z) = mesh (x, y), component order
(u_r, u_z, u_theta):
    ev.n_elements                -> int
    ev.geometry(ref_pts)         -> (x[ne, nq, 2], detJ[ne, nq])            element map (isoparametric if curved)
    ev.velocity(ref_pts)         -> (u[ne, nq, 3], grad_u[ne, nq, 3, 2])    grad_u[..., i, j] = d u_i / d (r, z)_j
    ev.pressure(ref_pts)         -> p[ne, nq]
ref_pts are points of the reference triangle (0,0), (1,0), (0,1); its P2 geometry nodes are
[v0, v1, v2, m01, m12, m20] at (0,0), (1,0), (0,1), (1/2,0), (1/2,1/2), (0,1/2).

Norms (axisymmetric cases use r-weighted norms and also report unweighted ones):
    ||e||_{L2_r}^2 = int |e|^2 r dr dz                                    (velocity: all three components)
    |e|_{H1_r}^2  = int (|grad e_r|^2 + e_r^2/r^2 + |grad e_z|^2 + |grad e_t|^2 + e_t^2/r^2) r dr dz   (V^1_1)
    ||e||_{L2}, |e|_{H1}: the same without the weight and without the hoop terms (plain meridional gradient)
    pressure mean-free: r-weighted mean for the r-weighted norm, plain mean for the plain norm
Physical 3D values carry 2 pi; every functional named *_3d includes it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from f1a_stokes_check import (_structured, red_refine, structured_mesh, distorted_mesh, mesh_hash, p2_nodes_hash,  # noqa: F401
                              verify_mesh, boundary_edges, all_boundary_triangles, dirichlet_nodal_defect,
                              dirichlet_shared_node_mismatch, tri_rule)

QUAD_DEGREE = 10          # error-norm quadrature: collapsed Gauss, exact to degree 12
FIELD_DIR = "tools/fluids_mms/out/f1b"
REF_NODES = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)


# ============================================================================================== exact fields
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compile(spec: dict):
    code = compile(spec["numpy"], "<mms expression>", "eval")

    def f(R, Z):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = eval(code, {"__builtins__": {}, "np": np, "numpy": np}, {"r": R, "z": Z})
        return np.broadcast_to(np.asarray(v, dtype=float), np.shape(R))
    return f


class Field:
    """Exact solution and data of one generated F1b field (tools/fluids_mms/out/f1b/<name>.json), sha256-verified."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self._u = [_compile(s) for s in data["u"]]
        self._g = [[_compile(s) for s in row] for row in data["grad_u"]]
        self.p = _compile(data["p"])
        self.mu = _compile(data["mu"])
        self._f = {form: [_compile(s) for s in fd["f"]] for form, fd in data["forms"].items()}
        self._sig = {form: [[_compile(s) for s in row] for row in fd["sigma"]] for form, fd in data["forms"].items()}
        self.constants = data["constants"]
        self.extra = data.get("extra", {})

    def u(self, R, Z):
        return np.stack([c(R, Z) for c in self._u], axis=-1)

    def grad_u(self, R, Z):
        return np.stack([np.stack([self._g[i][j](R, Z) for j in range(2)], axis=-1) for i in range(3)], axis=-2)

    def f(self, form):
        fs = self._f[form]
        return lambda R, Z: np.stack([c(R, Z) for c in fs], axis=-1)

    def sigma(self, form):
        s = self._sig[form]
        return lambda R, Z: np.stack([np.stack([s[i][j](R, Z) for j in range(3)], -1) for i in range(3)], -2)

    def traction(self, form):
        """g(R, Z, n) = sigma_form n for outward normals n[..., 2] = (n_r, n_z)."""
        s = self._sig[form]

        def g(R, Z, n):
            n = np.asarray(n, float)
            return np.stack([s[i][0](R, Z) * n[..., 0] + s[i][1](R, Z) * n[..., 1] for i in range(3)], axis=-1)
        return g


def load_field(entry: dict, root: Path | None = None) -> Field:
    root = Path(root) if root is not None else _repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return Field(json.loads(raw))


# ============================================================================================== meshes
def square_mesh(N: int):
    """MMS family on (r, z) in [0, 1]^2: f1a_stokes_check.structured_mesh(N) with x = r, y = z (N = 8 base, red
    refinements, corner cells (1,0) and (0,1) split the other way: no triangle with three boundary vertices,
    the axis counted as boundary)."""
    return structured_mesh(N)


def square_distorted(k: int):
    """f1a_stokes_check.distorted_mesh(k) with x = r: interior vertices only are moved, so the axis stays at r = 0."""
    return distorted_mesh(k)


def annulus_rect_mesh(N: int):
    """Couette family: structured_mesh(N) mapped to (r, z) in [1, 2] x [0, 1] (r = 1 + x)."""
    return structured_mesh(N, x0=1.0, x1=2.0)


def pipe_mesh(variant: str):
    """Hagen-Poiseuille patch meshes on (r, z) in [0, 1] x [0, 2]: 'coarse' 4 x 8 cells, 'fine' its red refinement,
    'dist_a', 'dist_b' interior vertices of the coarse mesh moved by 0.3 h U(-1,1)^2 (PCG64 seeds 20261101, 20261102).
    Corner cells split so that no triangle has three boundary vertices."""
    V, T = _structured(4, 8, 0.0, 1.0, 0.0, 2.0)
    if variant == "fine":
        return red_refine(V, T)
    if variant in ("dist_a", "dist_b"):
        seed = 20261101 if variant == "dist_a" else 20261102
        inner = np.where((V[:, 0] > 1e-12) & (V[:, 0] < 1 - 1e-12) & (V[:, 1] > 1e-12) & (V[:, 1] < 2 - 1e-12))[0]
        rng = np.random.Generator(np.random.PCG64(seed))
        V = V.copy()
        V[inner] += 0.3 * 0.25 * rng.uniform(-1, 1, (len(inner), 2))
        return V, T
    if variant == "coarse":
        return V, T
    raise ValueError(variant)


SPHERE_A, SPHERE_ROUT = 1.0, 4.0


def sphere_param_mesh(k: int, n_rho0: int = 2, n_th0: int = 4):
    """Parameter-space mesh of the sphere family: (s, t) in [0,1]^2, base n_rho0 x n_th0 cells (corner cells
    (1,0) and (0,1) split the other way), k red refinements.  Level k has N_rho = n_rho0 2^k, N_theta = n_th0 2^k."""
    V, T = _structured(n_rho0, n_th0, 0.0, 1.0, 0.0, 1.0)
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def _sphere_map(S, Tt):
    rho = SPHERE_A * (SPHERE_ROUT / SPHERE_A) ** S
    th = np.pi * Tt
    return np.c_[rho * np.sin(th), rho * np.cos(th)]


def sphere_mesh(k: int):
    """The meridional domain {1 <= rho <= 4, 0 <= theta <= pi} (r = rho sin theta, z = rho cos theta):
    rho = 4^s (geometric radial spacing), theta = pi t, on sphere_param_mesh(k).  Vertices and triangles; the
    isoparametric P2 geometry is sphere_p2_nodes(k).  Nested in the parameter space (as in F1a's A2)."""
    V, T = sphere_param_mesh(k)
    X = _sphere_map(V[:, 0], V[:, 1])
    X[np.abs(V[:, 1]) < 1e-14, 0] = 0.0          # exactly on the axis
    X[np.abs(V[:, 1] - 1) < 1e-14, 0] = 0.0
    return X, T[:, [0, 2, 1]]                    # the map (s, t) -> (r, z) reverses orientation: keep CCW


def sphere_body_data(R, Z, case: str):
    """Dirichlet data of the SPHERE attribute (rho = 1): the physical body condition, never the exact solution's
    extension (with the extension an affine sphere would be consistent and its geometry error invisible).
    case 'translation': u = 0; 'rotation': rigid rotation u = (0, 0, r) (Omega = 1).  The outer circle rho = 4
    takes the exact solution of the field."""
    R = np.asarray(R, float)
    z = np.zeros_like(R)
    if case == "translation":
        return np.stack([z, z, z], -1)
    if case == "rotation":
        return np.stack([z, z, R], -1)
    raise ValueError(case)


def sphere_p2_nodes(k: int):
    """X6[ne, 6, 2]: for every triangle of sphere_mesh(k) the 6 geometry nodes [v0, v1, v2, m01, m12, m20].  An edge
    with both end points on the sphere rho = 1 or both on rho = 4 has its midpoint ON that circle at the bisecting
    angle; every other edge (interior and axis edges) is straight (midpoint = mean of the end points)."""
    V, T = sphere_mesh(k)
    rad = np.hypot(V[:, 0], V[:, 1])
    X6 = np.empty((len(T), 6, 2))
    for e, t in enumerate(T):
        X6[e, :3] = V[t]
        for kk, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = V[t[i]], V[t[j]]
            ra, rb = rad[t[i]], rad[t[j]]
            circ = next((R for R in (SPHERE_A, SPHERE_ROUT) if abs(ra - R) < 1e-12 and abs(rb - R) < 1e-12), None)
            if circ is not None:
                ang = math.atan2(a[0] + b[0], a[1] + b[1])           # angle from the +z axis
                X6[e, 3 + kk] = (circ * math.sin(ang), circ * math.cos(ang))
            else:
                X6[e, 3 + kk] = 0.5 * (a + b)
    return X6


def tube_mesh(k: int, lam: float, L_over_R: float = 4.0, beta: float = 2.0, n_s0: int = 2, n_t0: int = 6):
    """Wall-corrected sphere case (lit_sphere_drag_tube_wall; L1): sphere of radius 1 on the
    axis of a cylinder of radius R = 1/lam, truncated at z = +-L, L = L_over_R R.  Parameter mesh (s, t) in [0,1]^2
    (base n_s0 x n_t0 cells, corner cells flipped, k red refinements); t -> polar angle theta piecewise linearly with
    breakpoints t = 1/n_t0 and 1 - 1/n_t0 at the corner angles theta_c = atan(R/L) and pi - theta_c (base-grid lines,
    so the cylinder corners are vertices at every level); the point is inner + g(s) (outer - inner) with inner on the
    sphere, outer where the ray at theta meets the end plane or the wall, g(s) = (e^(beta s) - 1)/(e^beta - 1).
    Returns (V, T, X6): edges on the sphere curved (midpoint on the circle at the bisecting angle), all others
    straight."""
    R = 1.0 / lam
    L = L_over_R * R
    tc = math.atan2(R, L)
    t1 = 1.0 / n_t0
    Vp, T = _structured(n_s0, n_t0, 0.0, 1.0, 0.0, 1.0)
    for _ in range(k):
        Vp, T = red_refine(Vp, T)
    S, Tt = Vp[:, 0], Vp[:, 1]
    th = np.where(Tt <= t1, Tt / t1 * tc, np.where(Tt >= 1 - t1, np.pi - (1 - Tt) / t1 * tc,
                                                   tc + (Tt - t1) / (1 - 2 * t1) * (np.pi - 2 * tc)))
    sn, cs = np.sin(th), np.cos(th)
    with np.errstate(divide="ignore"):
        lam_r = np.where(sn > 1e-300, R / np.maximum(sn, 1e-300), np.inf)
        lam_z = np.where(np.abs(cs) > 1e-300, L / np.maximum(np.abs(cs), 1e-300), np.inf)
    rho_out = np.minimum(lam_r, lam_z)
    g = (np.exp(beta * S) - 1) / (np.exp(beta) - 1)
    rho = 1.0 + g * (rho_out - 1.0)
    # outer boundary points exactly on the straight walls; inner on the sphere
    V = np.c_[rho * sn, rho * cs]
    V[np.abs(Tt) < 1e-14, 0] = 0.0
    V[np.abs(Tt - 1) < 1e-14, 0] = 0.0
    out = np.abs(S - 1) < 1e-14
    V[out & (Tt > t1 - 1e-14) & (Tt < 1 - t1 + 1e-14), 0] = R
    V[out & (Tt <= t1 + 1e-14), 1] = L
    V[out & (Tt >= 1 - t1 - 1e-14), 1] = -L
    T = T[:, [0, 2, 1]]
    rad = np.hypot(V[:, 0], V[:, 1])
    X6 = np.empty((len(T), 6, 2))
    for e, t in enumerate(T):
        X6[e, :3] = V[t]
        for kk, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = V[t[i]], V[t[j]]
            if abs(rad[t[i]] - 1) < 1e-12 and abs(rad[t[j]] - 1) < 1e-12:
                ang = math.atan2(a[0] + b[0], a[1] + b[1])
                X6[e, 3 + kk] = (math.sin(ang), math.cos(ang))
            else:
                X6[e, 3 + kk] = 0.5 * (a + b)
    return V, T, X6


# ============================================================================================== cavity lid (X case)
def cavity_lid(R, Z):
    """Dirichlet data of the swirling-lid cylinder cavity (xcode_stokes_axi_cavity_swirl), one smooth callable for
    every Dirichlet edge: g = (r (1 - r^2)^2 z^2, 0, 2 r (1 - r^2)^2 z^2) in (r, z, theta).  On the lid z = 1 it is
    (r (1-r^2)^2, 0, 2 r (1-r^2)^2); it vanishes on the wall r = 1 (to second order) and the bottom z = 0; u_r and
    u_theta vanish on the axis (odd in r)."""
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    a = R * (1 - R ** 2) ** 2 * Z ** 2
    return np.stack([a, np.zeros_like(a), 2 * a], axis=-1)


def kinetic_energy_r(ev) -> float:
    """(1/2) int |u_h|^2 r dr dz (per radian)."""
    q, xq, W = _weights(ev, 12)
    uh, _ = ev.velocity(q)
    return 0.5 * float((W * xq[..., 0] * (uh ** 2).sum(-1)).sum())


def dissipation_r(ev, form: str, mu: float = 1.0) -> float:
    """a_form(u_h, u_h) (per radian, constant mu): lap mu int (|grad u_h|^2 + (u_r^2 + u_t^2)/r^2) r dr dz; sym
    2 mu int eps(u_h) : eps(u_h) r dr dz with the hoop entries.  Degree-12 rule: polynomial integrands except on
    elements that touch the axis at ONE vertex (there u_h^2/r is rational)."""
    q, xq, W = _weights(ev, 12)
    R = xq[..., 0]
    uh, gh = ev.velocity(q)
    G = np.zeros(R.shape + (3, 3))
    G[..., :, :2] = gh
    G[..., 0, 2] = -uh[..., 2] / R
    G[..., 2, 2] = uh[..., 0] / R
    if form == "lap":
        return mu * float((W * R * (G ** 2).sum(axis=(-1, -2))).sum())
    E = 0.5 * (G + np.swapaxes(G, -1, -2))
    return 2 * mu * float((W * R * (E ** 2).sum(axis=(-1, -2))).sum())


# ============================================================================================== quadrature helpers
def _p2(q):
    xi, eta = q[:, 0], q[:, 1]
    l0, l1, l2 = 1 - xi - eta, xi, eta
    N = np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0], -1)
    dl = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    L = [l0, l1, l2]
    dN = np.zeros((len(q), 6, 2))
    for k in range(3):
        dN[:, k, :] = (4 * L[k] - 1)[:, None] * dl[k]
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        dN[:, 3 + k, :] = 4 * (L[a][:, None] * dl[b] + L[b][:, None] * dl[a])
    return N, dN


def element_nodes(ev):
    """The 6 geometry nodes of every element, read from the Evaluator's own map."""
    x, _ = ev.geometry(REF_NODES)
    return x


def _weights(ev, deg=QUAD_DEGREE):
    q, w = tri_rule(deg)
    xq, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    if np.any(xq[..., 0] <= 0):
        raise RuntimeError("a quadrature point on or left of the axis")
    return q, xq, w[None, :] * detJ


# ============================================================================================== norms
def norms(ev, fld: Field, pressure: str = "mean_free") -> dict:
    """Velocity and pressure errors in r-weighted and plain norms (module docstring), the swirl component alone,
    ||div_r u_h||_{L2_r}, and the magnitudes the round-off floors refer to."""
    q, xq, W = _weights(ev)
    R, Z = xq[..., 0], xq[..., 1]
    Wr = W * R
    uh, gh = ev.velocity(q)
    ph = ev.pressure(q)
    ue, ge = fld.u(R, Z), fld.grad_u(R, Z)
    pe = fld.p(R, Z)
    e, ge_ = uh - ue, gh - ge
    hoop = (e[..., 0] ** 2 + e[..., 2] ** 2) / R ** 2
    grad2 = (ge_ ** 2).sum(axis=(-1, -2))
    out = {
        "uL2r": math.sqrt(float((Wr[..., None] * e ** 2).sum())),
        "uH1r": math.sqrt(float((Wr * (grad2 + hoop)).sum())),
        "uL2": math.sqrt(float((W[..., None] * e ** 2).sum())),
        "uH1": math.sqrt(float((W * grad2).sum())),
        "utL2r": math.sqrt(float((Wr * e[..., 2] ** 2).sum())),
        "utH1r": math.sqrt(float((Wr * ((ge_[..., 2, :] ** 2).sum(-1) + e[..., 2] ** 2 / R ** 2)).sum())),
        "umL2r": math.sqrt(float((Wr * (e[..., 0] ** 2 + e[..., 1] ** 2)).sum())),
        "divL2r": math.sqrt(float((Wr * (gh[..., 0, 0] + uh[..., 0] / R + gh[..., 1, 1]) ** 2).sum())),
        "u_magnitude": float(np.max(np.sqrt((ue ** 2).sum(-1)))),
        "ut_magnitude": float(np.max(np.abs(ue[..., 2]))),
        "area_r": float(Wr.sum()), "area": float(W.sum()),
    }
    mr_h, mr = float((Wr * ph).sum() / Wr.sum()), float((Wr * pe).sum() / Wr.sum())
    m_h, m = float((W * ph).sum() / W.sum()), float((W * pe).sum() / W.sum())
    if pressure == "mean_free":
        out["pL2r"] = math.sqrt(float((Wr * ((ph - mr_h) - (pe - mr)) ** 2).sum()))
        out["pL2"] = math.sqrt(float((W * ((ph - m_h) - (pe - m)) ** 2).sum()))
        out["p_magnitude"] = float(np.max(np.abs(pe - mr)))
    elif pressure == "determined":
        out["pL2r"] = math.sqrt(float((Wr * (ph - pe) ** 2).sum()))
        out["pL2"] = math.sqrt(float((W * (ph - pe) ** 2).sum()))
        out["p_magnitude"] = float(np.max(np.abs(pe)))
    else:
        raise ValueError(pressure)
    out["pmean_r_h"], out["pmean_r"] = mr_h, mr
    return out


def discrete_norms(ev) -> dict:
    """||u_h||_{L2_r} per component group and ||p_h||_{L2_r} (patch tests whose exact value is zero)."""
    q, xq, W = _weights(ev)
    Wr = W * xq[..., 0]
    uh, _ = ev.velocity(q)
    ph = ev.pressure(q)
    return {"umL2r": math.sqrt(float((Wr * (uh[..., 0] ** 2 + uh[..., 1] ** 2)).sum())),
            "utL2r": math.sqrt(float((Wr * uh[..., 2] ** 2).sum())),
            "pL2r": math.sqrt(float((Wr * ph ** 2).sum()))}


def relative_l2r_difference(ev_a, ev_b, components=(0, 1, 2)) -> dict:
    """||u_a - u_b||_{L2_r} / ||u_a||_{L2_r} over the given velocity components, and the same for the
    r-weighted-mean-free pressures.  Same mesh."""
    q, xq, W = _weights(ev_a)
    Wr = W * xq[..., 0]
    c = list(components)
    ua, _ = ev_a.velocity(q)
    ub, _ = ev_b.velocity(q)
    pa, pb = ev_a.pressure(q), ev_b.pressure(q)
    pa0 = pa - (Wr * pa).sum() / Wr.sum()
    pb0 = pb - (Wr * pb).sum() / Wr.sum()
    du = float((Wr[..., None] * (ua[..., c] - ub[..., c]) ** 2).sum())
    nu = float((Wr[..., None] * ua[..., c] ** 2).sum())
    np_ = float((Wr * pa0 ** 2).sum())
    return {"u_rel": math.sqrt(du / nu) if nu > 0 else math.sqrt(du),
            "p_rel": math.sqrt(float((Wr * (pa0 - pb0) ** 2).sum()) / np_) if np_ > 0 else float("nan")}


# ============================================================================================== axis and continuity
def axis_nodal_max(points, values, scale: float) -> float:
    """max over velocity nodes ON THE AXIS (points[:, 0] == 0) of |u_r|, |u_theta|, divided by scale.  The strong
    axis condition gives exactly 0; a natural (missing) condition leaves O(h^?) values."""
    P = np.asarray(points, float)
    U = np.asarray(values, float)
    on = np.abs(P[:, 0]) < 1e-12
    if not np.any(on):
        raise RuntimeError("no axis node given")
    return float(np.max(np.abs(U[on][:, [0, 2]]))) / scale


def element_node_values(ev):
    """(X6[ne, 6, 2], U6[ne, 6, 3]): the P2 geometry nodes of every element and the discrete velocity there, read from
    the Evaluator at the reference nodes (independent of any node list the runner might supply)."""
    X6, _ = ev.geometry(REF_NODES)
    U6, _ = ev.velocity(REF_NODES)
    return X6, U6


def expected_axis_nodes(V, T) -> int:
    """Number of distinct P2 velocity nodes on r = 0 of the fixed mesh: axis vertices + one node per axis edge."""
    V = np.asarray(V, float)
    verts = {int(i) for e in boundary_edges(T) for i in e if abs(V[i, 0]) < 1e-12}
    edges = [e for e in boundary_edges(T) if abs(V[e[0], 0]) < 1e-12 and abs(V[e[1], 0]) < 1e-12]
    return len(verts) + len(edges)


def axis_nodal_max_ev(ev, V, T, scale: float) -> float:
    """Runner-independent axis check: every P2 velocity node with r = 0 is located from the Evaluator's own
    element map (vertices AND edge nodes); returns max |u_r|, |u_theta| there / scale.  Raises if the number of
    distinct axis nodes differs from the fixed mesh's (expected_axis_nodes), so a solver cannot pass by constraining
    (or reporting) only some of them."""
    X6, U6 = element_node_values(ev)
    on = np.abs(X6[..., 0]) < 1e-12
    keys = {tuple(np.round(p, 10) + 0.0) for p in X6[on]}
    want = expected_axis_nodes(V, T)
    if len(keys) != want:
        raise RuntimeError(f"axis nodes found {len(keys)} != fixed mesh {want}")
    return float(np.max(np.abs(U6[on][:, [0, 2]]))) / scale


def dirichlet_nodal_defect_ev(ev, edge_pred, g) -> float:
    """Runner-independent Dirichlet nodal-data check: for every element edge whose two end vertices and
    edge node satisfy edge_pred(Pa, Pm, Pb) (a Dirichlet attribute), the solution at its three P2 nodes against the
    fixed data g(R, Z) -> (..., 3): max |u_h - g| / max(max |g|, 1)."""
    X6, U6 = element_node_values(ev)
    pts, vals = [], []
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        for e in range(len(X6)):
            if edge_pred(X6[e, i], X6[e, 3 + k], X6[e, j]):
                for loc in (i, 3 + k, j):
                    pts.append(X6[e, loc])
                    vals.append(U6[e, loc])
    if not pts:
        raise RuntimeError("no Dirichlet edge matched")
    P = np.array(pts)
    gv = np.asarray(g(P[:, 0], P[:, 1]), float)
    return float(np.max(np.abs(np.array(vals) - gv)) / max(float(np.max(np.abs(gv))), 1.0))


def discrete_continuity_residual(ev, V) -> dict:
    """For every P1 pressure basis function q_k (hat function of mesh vertex k, V the fixed vertices):
        R_k   = int q_k (d_r u_r,h + u_r,h / r + d_z u_z,h) r dr dz      (the r q-tested continuity: 0 for a correct code)
        R0_k  = int q_k (same) dr dz                                     (q-tested, no weight: what an unweighted code zeroes)
    Returns max |R_k| and max |R0_k|, each relative to max_k int q_k |d_r(r u_r,h)| + |r d_z u_z,h| dr dz, and the
    per-vertex vector R.  Exact quadrature for affine elements (the integrands are polynomial)."""
    q, xq, W = _weights(ev, 12)
    R_ = xq[..., 0]
    uh, gh = ev.velocity(q)
    div_r = gh[..., 0, 0] * R_ + uh[..., 0] + gh[..., 1, 1] * R_          # r * div u_h
    scl = np.abs(gh[..., 0, 0] * R_ + uh[..., 0]) + np.abs(gh[..., 1, 1] * R_)
    xv, _ = ev.geometry(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    V = np.asarray(V, float)
    key = {tuple(np.round(v, 10) + 0.0): i for i, v in enumerate(V)}
    ids = np.array([[key[tuple(np.round(p, 10) + 0.0)] for p in el] for el in xv])
    lam = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], -1)       # P1 hats on the reference element
    Rk, R0k, Sk = np.zeros(len(V)), np.zeros(len(V)), np.zeros(len(V))
    np.add.at(Rk, ids.ravel(), np.einsum("eq,qk->ek", W * div_r, lam).ravel())
    np.add.at(R0k, ids.ravel(), np.einsum("eq,qk->ek", W * div_r / R_, lam).ravel())
    np.add.at(Sk, ids.ravel(), np.einsum("eq,qk->ek", W * scl, lam).ravel())
    s = float(Sk.max())
    return {"rq_max_rel": float(np.abs(Rk).max() / s), "q_max_rel": float(np.abs(R0k).max() / s), "R": Rk}


def flux_defect_r(edges, g):
    """r-weighted discrete flux of P2-interpolated Dirichlet data over straight boundary edges (a, m, b, n):
    int r g_h.n ds with g_h the P2 nodal interpolant, by Gauss (exact for the degree-3 integrand), and the reference
    int r |g.n| ds (20-point Gauss).  Returns (defect, reference)."""
    gp, gw = np.polynomial.legendre.leggauss(20)
    s = 0.5 * (gp + 1)
    gw = 0.5 * gw
    Nl = np.stack([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)], -1)
    tot, ref = 0.0, 0.0
    for (Pa, Pm, Pb, n) in edges:
        Pa, Pm, Pb, n = map(np.asarray, (Pa, Pm, Pb, n))
        L = float(np.hypot(*(Pb - Pa)))
        P = np.array([Pa, Pm, Pb])
        gn = np.asarray(g(P[:, 0], P[:, 1]))[:, :2] @ n
        xs = Pa[None, :] + s[:, None] * (Pb - Pa)[None, :]
        tot += L * float(np.sum(gw * xs[:, 0] * (Nl @ gn)))
        ref += L * float(np.sum(gw * xs[:, 0] * np.abs(np.asarray(g(xs[:, 0], xs[:, 1]))[:, :2] @ n)))
    return float(tot), float(ref)


# ============================================================================================== reactions
def body_nodes(X6, body):
    """on[ne, 6]: the element's P2 nodes that belong to the body.  A vertex belongs if body(vertex); an edge node
    belongs iff BOTH end vertices of its edge do (so the classification does not depend on where the solver put the
    edge node: on the curve for isoparametric geometry, on the chord for affine geometry)."""
    on = np.zeros(X6.shape[:2], bool)
    on[:, :3] = body(X6[:, :3, 0], X6[:, :3, 1])
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        on[:, 3 + k] = on[:, i] & on[:, j]
    return on


def _traction_term(ev, body, kind, traction_edges):
    """<g, w>_r over the boundary edges (straight or curved, P2 edge map from the element's geometry nodes) that
    satisfy one of the predicates in traction_edges [(pred(Pa, Pm, Pb) -> bool, g(R, Z, n) -> (..., 3)), ...]."""
    X6 = element_nodes(ev)
    on = body_nodes(X6, body)
    gp, gw = np.polynomial.legendre.leggauss(10)
    s = 0.5 * (gp + 1)
    gw = 0.5 * gw
    Nl = np.stack([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)], -1)
    dNl = np.stack([4 * s - 3, 4 - 8 * s, 4 * s - 1], -1)
    comp = {"force_z": 1, "torque_z": 2}[kind]
    tot = 0.0
    for e in np.where(on.any(axis=1))[0]:
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            loc = (i, 3 + k, j)
            if not on[e, list(loc)].any():
                continue
            Pa, Pm, Pb = X6[e, i], X6[e, 3 + k], X6[e, j]
            g = next((gg for pred, gg in traction_edges if pred(Pa, Pm, Pb)), None)
            if g is None:
                continue
            X = np.outer(Nl[:, 0], Pa) + np.outer(Nl[:, 1], Pm) + np.outer(Nl[:, 2], Pb)
            dX = np.outer(dNl[:, 0], Pa) + np.outer(dNl[:, 1], Pm) + np.outer(dNl[:, 2], Pb)
            jac = np.hypot(dX[:, 0], dX[:, 1])
            nrm = np.c_[dX[:, 1], -dX[:, 0]] / jac[:, None]
            if np.dot(Pm - X6[e, :3].mean(0), nrm[len(nrm) // 2]) < 0:
                nrm = -nrm
            coef = np.array([float(on[e, d]) * (X6[e, d, 0] if kind == "torque_z" else 1.0) for d in loc])
            wv = Nl @ coef
            gv = np.asarray(g(X[:, 0], X[:, 1], nrm), float)[:, comp]
            tot += float(np.sum(gw * jac * X[:, 0] * gv * wv))
    return tot


def reaction(ev, mu, body, kind: str, f=None, deg: int = 14, traction_edges=(), residual_form: str = "sym") -> float:
    """Babuska-Miller force/torque EXERTED BY THE FLUID ON THE BODY, per radian (multiply by 2 pi for 3D), computed
    here from the discrete solution (never from solver output), always with the symmetric (Cauchy) stress:
        F(w) = -[ a_sym(u_h, w) - (p_h, div_r w)_r - (f, w)_r ]
    with w the P2 function (on the fixed isoparametric geometry, read from ev.geometry at the reference nodes) equal
    to e_z ('force_z') or to r e_theta ('torque_z') at the velocity nodes on the body and 0 at all other nodes.
    body(R, Z) -> bool for VERTEX coordinates (edge nodes follow their edge: body_nodes).  Where the body's nodal functions reach a traction boundary edge (a
    body corner next to an inlet), the residual also subtracts <g, w>_r on that edge:
    traction_edges = [(pred(Pa, Pm, Pb), g(R, Z, n)), ...] with the fixed data g = sigma_form n.
    residual_form = "lap" uses mu grad u : grad w (the lap solution's own, Galerkin-consistent residual).
    Quadrature: degree `deg` here, not the solver's; on curved or 1/r integrands the difference from the solver's own
    residual is the solver's quadrature error, which bounds every balance check that uses it."""
    q, w = tri_rule(deg)
    N, dN = _p2(q)
    X6 = element_nodes(ev)                                              # (ne, 6, 2)
    on = body_nodes(X6, body)                                           # (ne, 6)
    sel = np.where(on.any(axis=1))[0]
    X6s = X6[sel]
    xq = np.einsum("qk,ekd->eqd", N, X6s)
    J = np.einsum("qkj,ekd->eqdj", dN, X6s)
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)
    dNx = np.einsum("qkj,eqjd->eqkd", dN, Ji)
    R, Z = xq[..., 0], xq[..., 1]
    Wr = w[None, :] * det * R
    comp = {"force_z": 1, "torque_z": 2}[kind]
    coef = on[sel].astype(float) * (X6s[..., 0] if kind == "torque_z" else 1.0)   # nodal values of w_comp
    wv = np.einsum("qk,ek->eq", N, coef)
    wg = np.einsum("eqkd,ek->eqd", dNx, coef)                           # d w_comp / d(r, z)
    # grad w (3x3, rows r, z, t): only row `comp` is nonzero, plus hoop entries for w_t: (r,t) = -w_t/r
    Gw = np.zeros(R.shape + (3, 3))
    Gw[..., comp, 0], Gw[..., comp, 1] = wg[..., 0], wg[..., 1]
    if comp == 2:
        Gw[..., 0, 2] = -wv / R
    # discrete solution on the same elements, at the same points (the Evaluator works on all elements)
    uh_all, gh_all = ev.velocity(q)
    ph_all = ev.pressure(q)
    uh, gh, ph = uh_all[sel], gh_all[sel], ph_all[sel]
    Gu = np.zeros(R.shape + (3, 3))
    Gu[..., :, :2] = gh
    Gu[..., 0, 2] = -uh[..., 2] / R
    Gu[..., 2, 2] = uh[..., 0] / R
    MU = mu(R, Z) * np.ones_like(R)
    if residual_form == "lap":
        # Galerkin-consistent residual of a LAP solution (pseudo-traction): physical only on no-slip bodies at rest
        #; never for moving walls (rotating sphere: 1/3 off)
        a = float((Wr * MU * (Gu * Gw).sum(axis=(-1, -2))).sum())
    else:
        eu = 0.5 * (Gu + np.swapaxes(Gu, -1, -2))
        ew = 0.5 * (Gw + np.swapaxes(Gw, -1, -2))
        a = float((Wr * 2 * MU * (eu * ew).sum(axis=(-1, -2))).sum())
    divw = Gw[..., 0, 0] + Gw[..., 1, 1] + Gw[..., 2, 2]
    b = float((Wr * ph * divw).sum())
    fw = 0.0
    if f is not None:
        fw = float((Wr * f(R, Z)[..., comp] * wv).sum())
    gw_ = _traction_term(ev, body, kind, traction_edges) if traction_edges else 0.0
    return -(a - b - fw - gw_)


def applied_boundary_load(ev, edge_pred, g, kind: str) -> float:
    """int_Gamma g_comp w r ds over the boundary edges matching edge_pred (curved edges by their P2 map), with
    w = 1 ('force_z', comp z) or w = r ('torque_z', comp theta): the force/torque the traction data apply, per
    radian.  g(R, Z, n) -> (..., 3) with the outward normal n."""
    X6 = element_nodes(ev)
    gp, gw = np.polynomial.legendre.leggauss(12)
    s_ = 0.5 * (gp + 1)
    gw = 0.5 * gw
    Nl = np.stack([(1 - s_) * (1 - 2 * s_), 4 * s_ * (1 - s_), s_ * (2 * s_ - 1)], -1)
    dNl = np.stack([4 * s_ - 3, 4 - 8 * s_, 4 * s_ - 1], -1)
    comp = {"force_z": 1, "torque_z": 2}[kind]
    tot = 0.0
    for e in range(len(X6)):
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            Pa, Pm, Pb = X6[e, i], X6[e, 3 + k], X6[e, j]
            if not edge_pred(Pa, Pm, Pb):
                continue
            X = np.outer(Nl[:, 0], Pa) + np.outer(Nl[:, 1], Pm) + np.outer(Nl[:, 2], Pb)
            dX = np.outer(dNl[:, 0], Pa) + np.outer(dNl[:, 1], Pm) + np.outer(dNl[:, 2], Pb)
            jac = np.hypot(dX[:, 0], dX[:, 1])
            nrm = np.c_[dX[:, 1], -dX[:, 0]] / jac[:, None]
            if np.dot(Pm - X6[e, :3].mean(0), nrm[len(nrm) // 2]) < 0:
                nrm = -nrm
            gv = np.asarray(g(X[:, 0], X[:, 1], nrm), float)[:, comp]
            w = X[:, 0] if kind == "torque_z" else 1.0
            tot += float(np.sum(gw * jac * X[:, 0] * gv * w))
    return tot


def area_r(ev) -> float:
    """int r dr dz of the discrete domain (geometry check: isoparametric order 4, affine order 2)."""
    _, xq, W = _weights(ev)
    return float((W * xq[..., 0]).sum())
