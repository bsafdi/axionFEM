"""Planar Stokes (group F1a): the check module's error computation (runner.check_module "f1a_stokes_check").

The case runner module (cases/f1a_stokes.py) builds the fixed meshes with the builders below, solves through the
public fluids API, wraps the discrete solution in an Evaluator and calls the functions here.  Nothing in this module
calls the solver: every norm, identity and check value is computed here from raw point evaluations.

Evaluator protocol (provided by the runner):
    ev.n_elements                        -> int
    ev.geometry(ref_pts)                 -> (x[ne, nq, 2], detJ[ne, nq])          element map, isoparametric if curved
    ev.velocity(ref_pts)                 -> (u[ne, nq, 2], grad_u[ne, nq, 2, 2])  grad_u[..., i, j] = d u_i / d x_j
    ev.pressure(ref_pts)                 -> p[ne, nq]
ref_pts are points of the reference triangle (0,0), (1,0), (0,1).

Conventions: G_ij = d u_i / d x_j; sigma_sym = 2 mu eps - p I; sigma_lap = mu G - p I; traction g = sigma_form n
with the outward normal; grad-div gamma (div u, div v) with gamma in viscosity units.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

QUAD_DEGREE = 10          # degree of the error-norm quadrature (exact for degree <= 12 with the collapsed rule)
FIELD_DIR = "tools/fluids_mms/out"


# ============================================================================================== exact fields
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compile(spec: dict):
    code = compile(spec["numpy"], "<mms expression>", "eval")

    def f(X, Y):
        v = eval(code, {"__builtins__": {}, "np": np, "numpy": np}, {"x": X, "y": Y})
        return np.broadcast_to(np.asarray(v, dtype=float), np.shape(X))
    return f


class Field:
    """Exact solution and data of one generated field (tools/fluids_mms/out/<name>.json), sha256-verified."""

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

    def u(self, X, Y):
        return np.stack([self._u[0](X, Y), self._u[1](X, Y)], axis=-1)

    def grad_u(self, X, Y):
        return np.stack([np.stack([self._g[i][j](X, Y) for j in range(2)], axis=-1) for i in range(2)], axis=-2)

    def f(self, form):
        fx, fy = self._f[form]
        return lambda X, Y: np.stack([fx(X, Y), fy(X, Y)], axis=-1)

    def traction(self, form, normal):
        s = self._sig[form]
        n0, n1 = normal
        return lambda X, Y: np.stack([s[i][0](X, Y) * n0 + s[i][1](X, Y) * n1 for i in range(2)], axis=-1)

    def extra_fn(self, key):
        return _compile(self.extra[key])


def load_field(entry: dict, root: Path | None = None) -> Field:
    """entry = {"path": "tools/fluids_mms/out/<name>.json", "sha256": ...} from the fixed run parameters."""
    root = Path(root) if root is not None else _repo_root()
    p = root / entry["path"]
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return Field(json.loads(raw))


# ============================================================================================== quadrature
def tri_rule(deg: int = QUAD_DEGREE):
    """Collapsed (Duffy) Gauss-Legendre rule on the reference triangle; exact for polynomials of degree <= 2n-2 with
    n = deg // 2 + 2 points per direction (deg 10 -> n = 7 -> exact to degree 12).  Weights sum to 1/2."""
    n = deg // 2 + 2
    g, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (g + 1)
    ws = 0.5 * w
    S, T = np.meshgrid(s, s, indexing="ij")
    WS, WT = np.meshgrid(ws, ws, indexing="ij")
    pts = np.c_[(S * (1 - T)).ravel(), T.ravel()]
    return pts, (WS * WT * (1 - T)).ravel()


# ============================================================================================== meshes
def _structured(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0, flip=True):
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    V = np.array([[a, b] for b in ys for a in xs])
    idx = lambda i, j: j * (nx + 1) + i  # noqa: E731
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if flip and ((i == nx - 1 and j == 0) or (i == 0 and j == ny - 1)):
                T += [[a, b, d], [b, c, d]]
            else:
                T += [[a, b, c], [a, c, d]]
    return V, np.array(T, dtype=np.int64)


def red_refine(V, T):
    V = [tuple(v) for v in np.asarray(V, float)]
    mids = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in mids:
            mids[key] = len(V)
            V.append(tuple(0.5 * (np.array(V[a]) + np.array(V[b]))))
        return mids[key]
    T2 = []
    for a, b, c in np.asarray(T):
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        T2 += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(V), np.array(T2, dtype=np.int64)


def structured_mesh(N: int, base: int = 8, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """The fixed structured family: the base mesh (base x base squares split along the (1,1) diagonal, except the
    cells at the corners (1,0) and (0,1), split the other way) refined uniformly (red) log2(N/base) times.  Nested;
    the flipped region of level k is a 2^k x 2^k block at each of those corners."""
    k = int(round(math.log2(N / base)))
    if base * 2 ** k != N:
        raise ValueError(f"N = {N} is not {base} * 2^k")
    V, T = _structured(base, base, x0, x1, y0, y1)
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def distorted_mesh(k: int, N0: int = 4, rel: float = 0.2, seed: int = 20261031):
    """Base N0 structured mesh, interior vertices displaced by rel * (1/N0) * U(-1,1)^2 drawn from
    numpy.random.Generator(PCG64(seed)) in vertex order, then k red refinements."""
    V, T = _structured(N0, N0)
    inner = np.where((V[:, 0] > 1e-12) & (V[:, 0] < 1 - 1e-12) & (V[:, 1] > 1e-12) & (V[:, 1] < 1 - 1e-12))[0]
    rng = np.random.Generator(np.random.PCG64(seed))
    V = V.copy()
    V[inner] += rel * (1.0 / N0) * rng.uniform(-1, 1, (len(inner), 2))
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def rotate(V, theta=math.pi / 6):
    c, s = math.cos(theta), math.sin(theta)
    return np.c_[V[:, 0] * c - V[:, 1] * s, V[:, 0] * s + V[:, 1] * c]


def polar_mesh(Nr: int, r1=1.0, r2=2.0, ntheta_per_nr: int = 8):
    """N_r x 8 N_r polar quadrilaterals, each split along the (i,j)-(i+1,j+1) diagonal.  Returns vertices and
    triangles; the P2 (isoparametric) nodes are polar_p2_nodes(Nr)."""
    Nt = ntheta_per_nr * Nr
    rs = np.linspace(r1, r2, Nr + 1)
    ts = 2 * np.pi * np.arange(Nt) / Nt
    V = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * Nt + (j % Nt)  # noqa: E731
    T = []
    for i in range(Nr):
        for j in range(Nt):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    return V, np.array(T, dtype=np.int64)


def polar_p2_nodes(Nr: int, r1=1.0, r2=2.0, ntheta_per_nr: int = 8):
    """The fixed isoparametric geometry of the A2 annulus: for every triangle of
    polar_mesh(Nr), its 6 geometry nodes [v0, v1, v2, m01, m12, m20].  A BOUNDARY edge (both end points on r = r1 or
    both on r = r2) has its midpoint on that circle at the bisecting angle; every other edge is straight (midpoint =
    mean of the end points).  Returns X6[ne, 6, 2]."""
    V, T = polar_mesh(Nr, r1, r2, ntheta_per_nr)
    rad = np.hypot(V[:, 0], V[:, 1])
    X6 = np.empty((len(T), 6, 2))
    for e, t in enumerate(T):
        X6[e, :3] = V[t]
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = V[t[i]], V[t[j]]
            ra, rb = rad[t[i]], rad[t[j]]
            on_circle = (abs(ra - r1) < 1e-12 and abs(rb - r1) < 1e-12) or (abs(ra - r2) < 1e-12 and abs(rb - r2) < 1e-12)
            if on_circle:
                ang = math.atan2(a[1] + b[1], a[0] + b[0])
                X6[e, 3 + k] = (ra * math.cos(ang), ra * math.sin(ang))
            else:
                X6[e, 3 + k] = 0.5 * (a + b)
    return X6


def p2_nodes_hash(X6, digits: int = 10) -> str:
    """Ordering-independent hash of the 6-node geometry of every element (vertices and edge nodes)."""
    X6 = np.round(np.asarray(X6, float), digits) + 0.0
    els = sorted(tuple(sorted(tuple(p) for p in el[:3])) + tuple(sorted(tuple(p) for p in el[3:])) for el in X6)
    return hashlib.sha256(repr(els).encode()).hexdigest()


def mesh_hash(V, T, digits: int = 10) -> str:
    """Connectivity-and-coordinates hash independent of vertex and element numbering: each triangle as the sorted
    tuple of its rounded vertex coordinates, the list sorted."""
    Vr = np.round(np.asarray(V, float), digits) + 0.0
    tris = sorted(tuple(sorted(tuple(Vr[k]) for k in t)) for t in np.asarray(T))
    return hashlib.sha256(repr(tris).encode()).hexdigest()


def boundary_edges(T):
    cnt = {}
    for t in np.asarray(T):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(t[i], t[j]), max(t[i], t[j]))
            cnt[key] = cnt.get(key, 0) + 1
    return [k for k, c in cnt.items() if c == 1]


def all_boundary_triangles(T) -> int:
    """Number of triangles with all three vertices on the boundary (the Taylor-Hood mesh condition)."""
    bv = {v for e in boundary_edges(T) for v in e}
    return int(sum(all(int(k) in bv for k in t) for t in np.asarray(T)))


def verify_mesh(V, T, V_ref, T_ref, V_coarser=None, X6=None, X6_ref=None) -> dict:
    """The solver's mesh against the fixed construction: element and vertex counts, coordinate/connectivity hash,
    the Taylor-Hood corner condition, and (if the next coarser fixed level is given) nestedness: every coarser
    vertex is a vertex of this level."""
    out = {"n_elements": int(len(T)), "n_vertices": int(len(V)),
           "counts_match": bool(len(T) == len(T_ref) and len(V) == len(V_ref)),
           "hash": mesh_hash(V, T), "hash_frozen": mesh_hash(V_ref, T_ref),
           "all_boundary_triangles": all_boundary_triangles(T)}
    out["hash_match"] = out["hash"] == out["hash_frozen"]
    if X6_ref is not None:      # curved (isoparametric) meshes: the edge nodes are part of the fixed geometry
        out["p2_hash"] = p2_nodes_hash(X6) if X6 is not None else None
        out["p2_hash_frozen"] = p2_nodes_hash(X6_ref)
        out["p2_hash_match"] = out["p2_hash"] == out["p2_hash_frozen"]
    if V_coarser is not None:
        here = {tuple(np.round(v, 10) + 0.0) for v in np.asarray(V)}
        out["nested"] = bool(all(tuple(np.round(v, 10) + 0.0) in here for v in np.asarray(V_coarser)))
    hmax = 0.0
    for t in np.asarray(T):
        P = np.asarray(V)[t]
        hmax = max(hmax, max(np.hypot(*(P[i] - P[j])) for i, j in ((0, 1), (1, 2), (2, 0))))
    out["h_max"] = float(hmax)
    return out


# ============================================================================================== norms
def _weights(ev, deg=QUAD_DEGREE):
    q, w = tri_rule(deg)
    xq, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    return q, xq, w[None, :] * detJ


def norms(ev, fld: Field, pressure: str = "mean_free", p_exact=None) -> dict:
    """uL2 = ||u - u_h||_L2, uH1 = |u - u_h|_H1 (all four gradient components), pL2 = ||p_h - p||_L2 (pressure
    'determined') or ||(p_h - mean p_h) - (p - mean p)||_L2 ('mean_free'), divL2 = ||div u_h||_L2, plus the
    magnitudes the round-off floors refer to and the pressure means.  p_exact overrides fld.p (periodic part)."""
    q, xq, W = _weights(ev)
    X, Y = xq[..., 0], xq[..., 1]
    uh, guh = ev.velocity(q)
    ph = ev.pressure(q)
    ue, ge = fld.u(X, Y), fld.grad_u(X, Y)
    pe = (p_exact or fld.p)(X, Y)
    area = W.sum()
    out = {"uL2": math.sqrt(float((W[..., None] * (uh - ue) ** 2).sum())),
           "uH1": math.sqrt(float((W[..., None, None] * (guh - ge) ** 2).sum())),
           "divL2": math.sqrt(float((W * (guh[..., 0, 0] + guh[..., 1, 1]) ** 2).sum())),
           "u_magnitude": float(np.max(np.hypot(ue[..., 0], ue[..., 1]))),
           "pmean_h": float((W * ph).sum() / area), "pmean": float((W * pe).sum() / area), "area": float(area)}
    if pressure == "mean_free":
        out["pL2"] = math.sqrt(float((W * ((ph - out["pmean_h"]) - (pe - out["pmean"])) ** 2).sum()))
        out["p_magnitude"] = float(np.max(np.abs(pe - out["pmean"])))
    elif pressure == "determined":
        out["pL2"] = math.sqrt(float((W * (ph - pe) ** 2).sum()))
        out["p_magnitude"] = float(np.max(np.abs(pe)))
    else:
        raise ValueError(pressure)
    out["pin_offset"] = abs(out["pmean_h"] - out["pmean"])
    return out


def discrete_norms(ev) -> dict:
    """||u_h||_L2, |u_h|_H1 and ||div u_h|| (no-flow test: the exact velocity is zero)."""
    q, xq, W = _weights(ev)
    uh, guh = ev.velocity(q)
    return {"uL2": math.sqrt(float((W[..., None] * uh ** 2).sum())),
            "uH1": math.sqrt(float((W[..., None, None] * guh ** 2).sum())),
            "divL2": math.sqrt(float((W * (guh[..., 0, 0] + guh[..., 1, 1]) ** 2).sum()))}


def relative_l2_difference(ev_a, ev_b) -> dict:
    """||u_a - u_b|| / ||u_a|| and ||p_a - p_b - mean|| / ||p_a - mean|| on the same mesh (identity checks)."""
    q, xq, W = _weights(ev_a)
    ua, _ = ev_a.velocity(q)
    ub, _ = ev_b.velocity(q)
    pa, pb = ev_a.pressure(q), ev_b.pressure(q)
    area = W.sum()
    pa0 = pa - (W * pa).sum() / area
    pb0 = pb - (W * pb).sum() / area
    return {"u_rel": math.sqrt(float((W[..., None] * (ua - ub) ** 2).sum()) / float((W[..., None] * ua ** 2).sum())),
            "p_rel": math.sqrt(float((W * (pa0 - pb0) ** 2).sum()) / float((W * pa0 ** 2).sum()))}


def energy_identity_defect(ev, nu: float, gamma: float, form: str, f) -> float:
    """No-flow test (u_h = 0 on the boundary, (p_h, div u_h) = 0 for the discrete pressure test functions):
        a_form(u_h, u_h) + gamma ||div u_h||^2 = (f, u_h)
    holds exactly for the correct discrete problem with nominal nu and gamma.  Returns |lhs - rhs| / |rhs|."""
    q, xq, W = _weights(ev, 14)
    uh, G = ev.velocity(q)
    if form == "lap":
        a = nu * float((W[..., None, None] * G ** 2).sum())
    elif form == "sym":
        eps = 0.5 * (G + np.swapaxes(G, -1, -2))
        a = 2 * nu * float((W[..., None, None] * eps ** 2).sum())
    else:
        raise ValueError(form)
    div = float((W * (G[..., 0, 0] + G[..., 1, 1]) ** 2).sum())
    fv = f(xq[..., 0], xq[..., 1])
    rhs = float((W[..., None] * fv * uh).sum())
    return abs(a + gamma * div - rhs) / abs(rhs)


def area(ev) -> float:
    _, _, W = _weights(ev)
    return float(W.sum())


# ============================================================================================== boundary quantities
def flux_defect(edges, g):
    """Discrete flux of P2-interpolated Dirichlet data: sum over straight boundary edges (a, m, b) given as
    coordinates, outward normal n, of |e|/6 (g.n(a) + 4 g.n(m) + g.n(b)) (the exact integral of the P2 interpolant
    = Simpson's rule), and the reference int |g.n| by 20-point Gauss.  Returns (defect, reference)."""
    gp, gw = np.polynomial.legendre.leggauss(20)
    s = 0.5 * (gp + 1)
    tot, ref = 0.0, 0.0
    for (Pa, Pm, Pb, n) in edges:
        Pa, Pm, Pb, n = map(np.asarray, (Pa, Pm, Pb, n))
        L = float(np.hypot(*(Pb - Pa)))
        P = np.array([Pa, Pm, Pb])
        gv = g(P[:, 0], P[:, 1]) @ n
        tot += L / 6 * (gv[0] + 4 * gv[1] + gv[2])
        xs = Pa[None, :] + s[:, None] * (Pb - Pa)[None, :]
        ref += L * 0.5 * float(np.sum(gw * np.abs(g(xs[:, 0], xs[:, 1]) @ n)))
    return float(tot), float(ref)


def slip_nodal_un(values, normals, scale, on_dirichlet=None) -> float:
    """max |u_h . n| / scale over slip/symmetry nodes; a node on two slip edges is checked against both normals.
    Nodes that also belong to a Dirichlet edge (on_dirichlet[i] True) are skipped: there the Dirichlet data win and
    dirichlet_nodal_defect covers them."""
    worst = 0.0
    for i, (u, ns) in enumerate(zip(values, normals)):
        if on_dirichlet is not None and on_dirichlet[i]:
            continue
        for n in ns:
            worst = max(worst, abs(float(np.dot(u, n))))
    return worst / scale


def dirichlet_nodal_defect(points, values, g) -> float:
    """Dirichlet data must be the P2 LAGRANGE NODAL INTERPOLANT of the fixed callable g (exact at the vertices and edge midpoints).  points[n, 2] are the Dirichlet velocity nodes of
    the discrete solution (vertices and edge nodes), values[n, 2] the solution there.  Returns
    max |u_h - g| / max(max |g|, 1): round-off (~1e-16) for nodal interpolation, ~1e-7 for projection-type data."""
    P = np.asarray(points, float)
    gv = g(P[:, 0], P[:, 1])
    return float(np.max(np.abs(np.asarray(values, float) - gv)) / max(float(np.max(np.abs(gv))), 1.0))


def dirichlet_shared_node_mismatch(points, callables) -> float:
    """Corner / shared-node rule: at a velocity node shared by several Dirichlet boundary
    attributes, the fixed data of those attributes must AGREE, so that the attribute order in the API cannot change
    the discrete data.  points[n, 2]: the shared nodes; callables: the per-attribute data g_a(X, Y) of the attributes
    meeting there (list of lists, one list per node).  Returns max_n max_{a,b} |g_a - g_b| / max(max |g|, 1)."""
    worst, scale = 0.0, 1.0
    for P, gs in zip(np.asarray(points, float), callables):
        vals = [np.asarray(g(np.array([P[0]]), np.array([P[1]])), float).reshape(-1) for g in gs]
        scale = max(scale, max(float(np.max(np.abs(v))) for v in vals))
        for a in vals:
            for b in vals:
                worst = max(worst, float(np.max(np.abs(a - b))))
    return worst / scale


# ============================================================================================== inf-sup
def inf_sup(A, B, Mp, zero_rel: float = 1e-10) -> dict:
    """Discrete inf-sup estimate: beta^2 = eigenvalues of B A^-1 B^T q = lambda Mp q on the free velocity DOFs
    (A the vector H1-seminorm Gram matrix with Dirichlet rows removed).  Dense (small meshes only).  Eigenvalues
    below zero_rel * lambda_max are zero modes; the constant is one of them.  Returns beta (smallest non-zero),
    the number of zero modes beyond the constant, and lambda_max."""
    import scipy.linalg as sla
    A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    B = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
    Mp = Mp.toarray() if hasattr(Mp, "toarray") else np.asarray(Mp)
    S = B @ np.linalg.solve(A, B.T)
    S = 0.5 * (S + S.T)
    lam = sla.eigh(S, Mp, eigvals_only=True)
    lmax = float(lam.max())
    nz = lam[lam > zero_rel * lmax]
    n_zero = int(np.sum(lam <= zero_rel * lmax))
    return {"beta": float(math.sqrt(nz.min())) if len(nz) else 0.0, "zero_modes_beyond_constant": max(n_zero - 1, 0),
            "lambda_max": lmax}
