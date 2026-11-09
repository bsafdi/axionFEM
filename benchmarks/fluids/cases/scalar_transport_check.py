"""Steady scalar transport (advection-diffusion-reaction, SUPG option, planar and axisymmetric): the check module's
error computation for the heat-transfer cases (group F4a).

It is SELF-CONTAINED on purpose: it imports only numpy and the standard library (no fluids or chemistry module), and
its data files are addressed by the case parameters' {path, sha256}, never by a hard-coded directory.
Nothing here calls a solver: every norm, extremum, flux and identity is computed from raw point evaluations and raw
vectors handed over by the runner.

PDE (a = rho c_p or 1, kappa = k or D, sigma = 0 in the fluids cases):
    convective   a b.grad T - div(kappa grad T) + sigma T = s
    conservative div(a b T - kappa grad T) + sigma T = s
Axisymmetric: (r, z) half-plane, measure r dr dz (the physical volume / 2 pi), T even in r.

Evaluator protocol (provided by the runner):
    ev.n_elements                    -> int
    ev.geometry(ref)                 -> (X[ne, nq, 2], detJ[ne, nq])   planar map (x, y) or (r, z); detJ WITHOUT r
    ev.scalar(ref)                   -> (T[ne, nq], dT[ne, nq, 2])     dT = (d/dc0, d/dc1) of the discrete scalar
ref = points of the reference triangle (0,0), (1,0), (0,1).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

QUAD_DEGREE = 10
_CHEB = np.polynomial.chebyshev


# ============================================================================================== quadrature
def tri_rule(deg: int = QUAD_DEGREE):
    """Collapsed (Duffy) Gauss-Legendre rule on the reference triangle, n = deg // 2 + 2 points per direction:
    exact for polynomials of degree <= 2n - 2 (deg 10 -> exact to degree 12).  Weights sum to 1/2."""
    n = deg // 2 + 2
    g, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (g + 1)
    ws = 0.5 * w
    S, T = np.meshgrid(s, s, indexing="ij")
    WS, WT = np.meshgrid(ws, ws, indexing="ij")
    return np.c_[(S * (1 - T)).ravel(), T.ravel()], (WS * WT * (1 - T)).ravel()


def line_rule(n: int = 8):
    g, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (g + 1), 0.5 * w


# The six P2 Lagrange nodes of the reference triangle (vertices, then edge midpoints) and the three local edges as
# (start, end) in reference coordinates.
P2_NODES = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
REF_EDGES = (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0)))


# ============================================================================================== meshes
def structured(nx, ny, box=(0.0, 1.0, 0.0, 1.0), flip=True):
    """nx x ny rectangles of the box split along the (1,1) diagonal, except the cells at the corners (c0max, c1min)
    and (c0min, c1max), split the other way (no triangle with all three vertices on the boundary: Taylor-Hood corner
    condition).  Identical construction to the fixed F1a builder (f1a_stokes_check._structured)."""
    x0, x1, y0, y1 = (float(v) for v in box)
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


def box_mesh(nx0: int, ny0: int, k: int, box=(0.0, 1.0, 0.0, 1.0)):
    """The nested box family: structured(nx0, ny0, box) refined uniformly (red) k times.  Unit square, N = 8 * 2^k:
    box_mesh(8, 8, k) (= f1a_stokes_check.structured_mesh(N)); channel/pipe [0,2] x [0,1] or [0,1] x [0,2]:
    box_mesh(16, 8, k, ...) / box_mesh(8, 16, k, ...)."""
    V, T = structured(nx0, ny0, box)
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def distorted_mesh(k: int, N0: int = 4, rel: float = 0.2, seed: int = 20261101):
    """Unit square, N0 x N0 base mesh with interior vertices displaced by rel / N0 * U(-1,1)^2 (numpy PCG64(seed)),
    refined (red) k times: nested, element Jacobians vary from element to element.  (Same construction as the F1a
    distorted family with its own seed.)"""
    V, T = structured(N0, N0)
    rng = np.random.default_rng(seed)
    V = V.copy()
    inner = (V[:, 0] > 1e-12) & (V[:, 0] < 1 - 1e-12) & (V[:, 1] > 1e-12) & (V[:, 1] < 1 - 1e-12)
    V[inner] += rel / N0 * rng.uniform(-1, 1, size=(int(inner.sum()), 2))
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def mesh_hash(V, T, digits: int = 10) -> str:
    h = hashlib.sha256()
    h.update(np.round(np.asarray(V, float), digits).tobytes())
    h.update(np.asarray(T, np.int64).tobytes())
    return h.hexdigest()


def all_boundary_triangles(T) -> int:
    cnt = {}
    for t in np.asarray(T):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(t[i], t[j]), max(t[i], t[j]))
            cnt[key] = cnt.get(key, 0) + 1
    bverts = {v for key, c in cnt.items() if c == 1 for v in key}
    return int(sum(all(v in bverts for v in t) for t in np.asarray(T)))


def verify_mesh(V, T, V_ref, T_ref, V_coarser=None) -> dict:
    """The solver's mesh against the fixed construction: counts, hash, corner condition, nestedness, h_max."""
    out = {"n_elements": int(len(T)), "n_vertices": int(len(V)),
           "counts_match": bool(len(T) == len(T_ref) and len(V) == len(V_ref)),
           "hash": mesh_hash(V, T), "hash_frozen": mesh_hash(V_ref, T_ref),
           "all_boundary_triangles": all_boundary_triangles(T)}
    out["hash_match"] = out["hash"] == out["hash_frozen"]
    if V_coarser is not None:
        here = {tuple(np.round(v, 10) + 0.0) for v in np.asarray(V)}
        out["nested"] = bool(all(tuple(np.round(v, 10) + 0.0) in here for v in np.asarray(V_coarser)))
    P = np.asarray(V)[np.asarray(T)]
    e = np.stack([np.hypot(*(P[:, i] - P[:, j]).T) for i, j in ((0, 1), (1, 2), (2, 0))], -1)
    out["h_max"] = float(e.max())
    return out


# ============================================================================================== data
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(entry: dict, root: Path | None = None) -> dict:
    """entry = {"path": <repository path>, "sha256": ...} from the fixed run parameters (any track's data dir)."""
    root = Path(root) if root is not None else _repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"case data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return json.loads(raw)


def compile_expr(spec: dict, coords):
    code = compile(spec["numpy"], "<case expression>", "eval")

    def f(A, B):
        with np.errstate(over="ignore", under="ignore"):
            v = eval(code, {"__builtins__": {}, "np": np, "numpy": np}, {coords[0]: A, coords[1]: B})
        return np.broadcast_to(np.asarray(v, dtype=float), np.shape(A))
    return f


class ScalarField:
    """Exact solution and data of a generated scalar field (kind scalar_mms | scalar_patch | scalar_layer |
    uniform_flux) or of a Graetz mode field (kind graetz_modes)."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self.kind = data["kind"]
        self.geometry = data["geometry"]
        self.coords = data["coords"]
        cc = self.coords
        self.a = float(data["a"]["value"])
        if self.kind == "graetz_modes":
            self.kappa_value = float(data["kappa"]["value"])
            self.modes = [(float(m["lambda"]), float(_rational(m["amplitude"])), np.array([float(v) for v in m["cheb"]]))
                          for m in data["modes"]]
            return
        self._T = compile_expr(data["T"], cc)
        self._gT = [compile_expr(s, cc) for s in data["grad_T"]]
        self._b = [compile_expr(s, cc) for s in data["b"]]
        self.kappa = compile_expr(data["kappa"], cc)
        self._s = {k: compile_expr(v, cc) for k, v in data["s"].items()}
        self.boundaries = {bd["name"]: bd for bd in data["boundaries"]}

    # ---- exact solution
    def T(self, A, B):
        if self.kind == "graetz_modes":
            t, s = (B, A) if self.geometry == "planar" else (A, B)
            return sum(amp * _CHEB.chebval(2 * t - 1, ch) * np.exp(-lm * s) for lm, amp, ch in self.modes)
        return self._T(A, B)

    def grad_T(self, A, B):
        if self.kind == "graetz_modes":
            t, s = (B, A) if self.geometry == "planar" else (A, B)
            dt = sum(amp * 2 * _CHEB.chebval(2 * t - 1, _CHEB.chebder(ch)) * np.exp(-lm * s) for lm, amp, ch in self.modes)
            ds = sum(-lm * amp * _CHEB.chebval(2 * t - 1, ch) * np.exp(-lm * s) for lm, amp, ch in self.modes)
            g = (ds, dt) if self.geometry == "planar" else (dt, ds)
            return np.stack([np.broadcast_to(g[0], np.shape(A)), np.broadcast_to(g[1], np.shape(A))], -1)
        return np.stack([g(A, B) for g in self._gT], -1)

    def b(self, A, B):
        if self.kind == "graetz_modes":
            if self.geometry == "planar":
                return np.stack([1.5 * (1 - B ** 2), 0 * B], -1)
            return np.stack([0 * A, 2.0 * (1 - A ** 2)], -1)
        return np.stack([g(A, B) for g in self._b], -1)

    def s(self, form="convective"):
        return self._s[form]

    def boundary_data(self, name: str, key: str):
        return compile_expr(self.boundaries[name][key], self.coords)


def _rational(s: str) -> float:
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def load_field(entry: dict, root: Path | None = None) -> ScalarField:
    return ScalarField(load_json(entry, root))


# ============================================================================================== norms
def _weights(ev, axisymmetric: bool, deg=QUAD_DEGREE):
    q, w = tri_rule(deg)
    X, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    W = w[None, :] * detJ
    Wr = W * X[..., 0] if axisymmetric else W
    return q, X, W, Wr


def element_mask(X, region):
    """region = None (whole domain) or [c0min, c0max, c1min, c1max]: the elements whose CENTROID lies inside
    (the fixed meshes align with the region's edges, so this is the region itself)."""
    if region is None:
        return np.ones(X.shape[0], bool)
    c = X.mean(axis=1)
    lo0, hi0, lo1, hi1 = region
    return (c[:, 0] >= lo0) & (c[:, 0] <= hi0) & (c[:, 1] >= lo1) & (c[:, 1] <= hi1)


def scalar_norms(ev, fld: ScalarField, axisymmetric: bool | None = None, region=None) -> dict:
    """TL2 = ||T - T_h||, TH1 = |T - T_h|_1 (both gradient components), Tstream = ||b.grad(T - T_h)|| (streamline
    derivative with the EXACT b), on the whole domain or a region (element centroids).  Axisymmetric: r-weighted
    (suffix w; the 3D norms / 2 pi) AND unweighted (suffix _unw).  Also the magnitudes for the round-off floors."""
    axi = (fld.geometry == "axisymmetric") if axisymmetric is None else axisymmetric
    q, X, W, Wr = _weights(ev, axi)
    m = element_mask(X, region)
    A, B = X[..., 0], X[..., 1]
    Th, dTh = ev.scalar(q)
    Te, dTe = fld.T(A, B), fld.grad_T(A, B)
    e, de = Th - Te, dTh - dTe
    bb = fld.b(A, B)
    st = (bb * de).sum(-1)
    out = {}
    for suf, Wt in ((("w" if axi else ""), Wr),) + ((("_unw", W),) if axi else ()):
        Wm = Wt[m]
        out["TL2" + suf] = math.sqrt(float((Wm * e[m] ** 2).sum()))
        out["TH1" + suf] = math.sqrt(float((Wm[..., None] * de[m] ** 2).sum()))
        out["Tstream" + suf] = math.sqrt(float((Wm * st[m] ** 2).sum()))
    out["T_magnitude"] = float(np.max(np.abs(Te[m])))
    out["gradT_magnitude"] = float(np.max(np.hypot(dTe[m][..., 0], dTe[m][..., 1])))
    out["Tstream_magnitude"] = float(np.max(np.abs((bb * dTe).sum(-1)[m])))
    out["measure"] = float(Wr[m].sum())
    # the solution_magnitude of every series: the max over the quadrature points
    # of the EXACT quantity whose error the norm measures (|T|, |grad T|, |b.grad T|), weighted or not alike
    mag = {"TL2": out["T_magnitude"], "TH1": out["gradT_magnitude"], "Tstream": out["Tstream_magnitude"]}
    out["magnitudes"] = {k: mag[k.replace("_unw", "").rstrip("w") if not k.endswith("_unw") else k[:-4]]
                         for k in out if k.startswith(("TL2", "TH1", "Tstream")) and not k.endswith("magnitude")}
    return out


def sampled_extrema(ev, region=None) -> dict:
    """max and min of T_h over the six P2 Lagrange nodes of every element (a SAMPLED extremum, declared as such:
    vertices and edge midpoints), optionally restricted to the elements of a region."""
    X, _ = ev.geometry(P2_NODES)
    Th, _ = ev.scalar(P2_NODES)
    m = element_mask(X, region)
    return {"max": float(Th[m].max()), "min": float(Th[m].min())}


def integral(ev, fn, axisymmetric: bool, region=None) -> float:
    """int fn(X, T_h, dT_h) dV over the domain or a region (axisymmetric: measure r dr dz)."""
    q, X, W, Wr = _weights(ev, axisymmetric)
    m = element_mask(X, region)
    Th, dTh = ev.scalar(q)
    return float((Wr * fn(X, Th, dTh))[m].sum())


def boundary_integral(ev, on_edge, fn, axisymmetric: bool, n1d: int = 8) -> float:
    """int over the boundary part {on_edge(X) true at every point of an element edge} of fn(X, T_h, dT_h, normal)
    ds (axisymmetric: r ds).  Elements and local edges are found from the element map alone (affine elements); the
    outward normal is that of the element edge."""
    t, w = line_rule(n1d)
    total = 0.0
    for (a0, a1) in REF_EDGES:
        a0, a1 = np.array(a0), np.array(a1)
        ref = a0[None, :] + t[:, None] * (a1 - a0)[None, :]
        X, _ = ev.geometry(ref)
        ok = np.all(on_edge(X), axis=1)
        if not ok.any():
            continue
        Th, dTh = ev.scalar(ref)
        Xe, _ = ev.geometry(np.stack([a0, a1]))
        tang = Xe[:, 1, :] - Xe[:, 0, :]
        L = np.hypot(tang[:, 0], tang[:, 1])
        Xc, _ = ev.geometry(np.array([[1 / 3, 1 / 3]]))
        nrm = np.stack([tang[:, 1], -tang[:, 0]], -1) / L[:, None]
        mid = 0.5 * (Xe[:, 0, :] + Xe[:, 1, :])
        flip = ((Xc[:, 0, :] - mid) * nrm).sum(-1) > 0
        nrm[flip] *= -1
        val = fn(X, Th, dTh, nrm[:, None, :])
        rw = X[..., 0] if axisymmetric else 1.0
        total += float((w[None, :] * L[:, None] * rw * val)[ok].sum())
    return total


# ============================================================================================== fluxes from raw vectors
def flux_from_residual(residual, dofs=None, weights=None) -> float:
    """Variationally consistent OUTWARD flux through a Dirichlet boundary part Gamma:
    Q_out = - R(T_h; v_Gamma) = - sum_i R_i w_i, with R_i = R(T_h; phi_i) the UNCONSTRAINED residual vector of EVERY
    degree of freedom (natural-boundary and SUPG terms included, Dirichlet rows NOT replaced) and w the coefficients of
    v_Gamma, the P2 (P1) Lagrange NODAL function equal to 1 at every node of the CLOSED part Gamma and 0 at every other
    node.  In a nodal (Lagrange) basis w is the indicator of Gamma's DOFs: pass `dofs`; in any other basis pass
    `weights`.  Exact for the exact solution: R(T; v_Gamma) = - int_Gamma q_out v_Gamma."""
    R = np.asarray(residual, float)
    if weights is not None:
        return float(-(R * np.asarray(weights, float)).sum())
    return float(-R[np.asarray(dofs, np.int64)].sum())


def nodal_dirichlet_defect(points, values, g) -> float:
    """max |T_h(node) - g(node)| over the Dirichlet nodes (P2 Lagrange nodal interpolation of the data)."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(values, float) - g(P[:, 0], P[:, 1]))))


def taylor_test(R0, Rfun, Jd, eps_list) -> dict:
    """Taylor test of a Jacobian: e(eps) = ||R(x + eps d) - R(x) - eps J(x) d|| for the fixed eps sequence (Rfun(eps)
    returns the residual vector at x + eps d on the free DOFs).  A correct Jacobian gives e ~ eps^2 (slope 2) until
    round-off; a wrong one gives slope 1.  Returns the errors, the pairwise slopes and the median slope over the
    pairs whose errors exceed 1e3 * machine epsilon * ||R0||."""
    R0 = np.asarray(R0, float)
    Jd = np.asarray(Jd, float)
    errs = [float(np.linalg.norm(np.asarray(Rfun(e), float) - R0 - e * Jd)) for e in eps_list]
    floor = 1e3 * np.finfo(float).eps * max(float(np.linalg.norm(R0)), float(np.linalg.norm(Jd)), 1e-300)
    slopes = []
    for i in range(len(eps_list) - 1):
        if errs[i + 1] > floor and errs[i] > floor:
            slopes.append(math.log(errs[i] / errs[i + 1]) / math.log(eps_list[i] / eps_list[i + 1]))
    med = float(np.median(slopes)) if slopes else float("nan")
    return {"errors": errs, "slopes": slopes, "median_slope": med, "n_slopes": len(slopes)}


# ============================================================================================== Nusselt / Sherwood
def bulk_and_wall_means(ev, fld: ScalarField, on_wall, axisymmetric: bool, region=None) -> dict:
    """Flow-weighted bulk mean T_b = int (b.e_flow) T_h dV / int (b.e_flow) dV (exact b, whole domain or region) and
    the wall mean T_w = int_wall T_h ds / int_wall ds (axisymmetric: r-weighted measures)."""
    flow = 0 if fld.geometry == "planar" else 1

    def ub(X, T, dT):
        return fld.b(X[..., 0], X[..., 1])[..., flow] * T

    def uu(X, T, dT):
        return fld.b(X[..., 0], X[..., 1])[..., flow]
    Tb = integral(ev, ub, axisymmetric, region) / integral(ev, uu, axisymmetric, region)
    Tw = (boundary_integral(ev, on_wall, lambda X, T, dT, n: T, axisymmetric)
          / boundary_integral(ev, on_wall, lambda X, T, dT, n: np.ones_like(T), axisymmetric))
    return {"T_bulk": float(Tb), "T_wall": float(Tw)}


def graetz_wall_flux_exact(fld: ScalarField, length: float = 2.0) -> float:
    """Exact outward diffusive flux through the wall t = 1 of a Graetz mode field over s in [0, length] (per radian in
    the pipe, where r = 1): -kappa sum_n A_n phi_n'(1) (1 - exp(-lam_n length)) / lam_n."""
    tot = 0.0
    for (lm, amp, ch) in fld.modes:
        dphi = 2 * _CHEB.chebval(1.0, _CHEB.chebder(ch))
        tot += -fld.kappa_value * amp * dphi * (1 - math.exp(-lm * length)) / lm
    return float(tot)


def exact_boundary_flux(ev, fld: ScalarField, on_edge, axisymmetric: bool) -> float:
    """int over a boundary part of the EXACT outward diffusive flux -kappa grad T . n (r ds in axisymmetry), by the
    same edge quadrature as boundary_integral (the solution_magnitude / reference of a variational-flux item)."""
    return boundary_integral(ev, on_edge, lambda X, T, dT, n: -fld.kappa(X[..., 0], X[..., 1])
                             * (fld.grad_T(X[..., 0], X[..., 1]) * n).sum(-1), axisymmetric)


def nusselt_uniform_flux(ev, fld: ScalarField, wall: str) -> float:
    """Nu = D q_in / (kappa (T_w - T_b)) of a uniform-wall-flux field (kind uniform_flux): q_in = - q_out of the wall
    data (heat entering the fluid), D = 4H (plates, hydraulic diameter, H = 1) or 2R (pipe, R = 1), means over the whole
    domain (T - A x is independent of x for the exact solution, so the domain means equal the cross-section ones)."""
    axi = fld.geometry == "axisymmetric"
    c = 0 if axi else 1
    on_wall = lambda X: np.abs(X[..., c] - 1.0) < 1e-12  # noqa: E731
    m = bulk_and_wall_means(ev, fld, on_wall, axi)
    qd = fld.boundaries[wall]["q_out"]
    q_out = float(compile_expr(qd, fld.coords)(np.array([0.5]), np.array([1.0]))[0]) if not axi else \
        float(compile_expr(qd, fld.coords)(np.array([1.0]), np.array([0.5]))[0])
    kap = float(fld.kappa(np.array([0.5]), np.array([0.5]))[0])
    D = 2.0 if axi else 4.0
    return D * (-q_out) / (kap * (m["T_wall"] - m["T_bulk"]))


# ============================================================================================== energy / mass balance
def natural_outflow(ev, parts, axisymmetric: bool) -> float:
    """Sum of the OUTWARD fluxes through the natural boundary parts, from the discrete solution and the fixed data:
    parts = [(on_edge, "neumann", q_out(A, B)) | (on_edge, "robin", h, T_ext(A, B))]; Neumann: int q_out ds;
    Robin: int h (T_h - T_ext) ds (axisymmetric: r ds).  Symmetry / axis / adiabatic parts contribute 0."""
    tot = 0.0
    for part in parts:
        if part[1] == "neumann":
            q = part[2]
            tot += boundary_integral(ev, part[0], lambda X, T, dT, n, q=q: q(X[..., 0], X[..., 1]), axisymmetric)
        elif part[1] == "robin":
            h, Te = part[2], part[3]
            tot += boundary_integral(ev, part[0], lambda X, T, dT, n, h=h, Te=Te: h * (T - Te(X[..., 0], X[..., 1])),
                                     axisymmetric)
        else:
            raise ValueError(part[1])
    return float(tot)


def edge_predicate(name: str, box):
    """on_edge(X) for a box edge name (x0, x1, y0, y1 / axis or r0 (the inner radius of an annulus), r1, z0, z1)."""
    c0a, c0b, c1a, c1b = (float(v) for v in box)
    ax, val = {"x0": (0, c0a), "x1": (0, c0b), "y0": (1, c1a), "y1": (1, c1b),
               "axis": (0, c0a), "r0": (0, c0a), "r1": (0, c0b), "z0": (1, c1a), "z1": (1, c1b)}[name]
    return lambda X: np.abs(X[..., ax] - val) < 1e-12


def balance_defect(q_dirichlet_out: float, natural_out: float, source: float, adv: float) -> float:
    """Discrete balance of the CONVECTIVE form tested with v = 1 (exact for any consistent discretization whose
    residual is the one solved, SUPG included since grad 1 = 0): q_D + natural_out - int s + int a b_h.grad T_h = 0,
    q_D the variational outward flux through the UNION of the Dirichlet parts (each DOF once).  Relative defect."""
    scale = max(abs(q_dirichlet_out), abs(natural_out), abs(source), abs(adv), 1e-300)
    return abs(q_dirichlet_out + natural_out - source + adv) / scale
