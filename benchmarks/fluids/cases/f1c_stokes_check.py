"""3D Stokes (group F1c): the check module's error computation (runner.check_module "f1c_stokes_check").

It imports numpy and the standard library only.  The case runner (cases/f1c_stokes.py) builds the fixed meshes with the
builders below, solves through the public fluids API on 1, 4 or 16 ranks, and hands each rank's part of the discrete
solution to the functions here through an Evaluator.  Nothing here calls the solver.

Evaluator protocol (one object per rank, or one for the whole mesh):
    ev.ids                     -> int array [m]: FIXED element indices (order of the builder) of the elements this
                                  evaluator covers (each element covered exactly once over all ranks)
    ev.geometry(ref)           -> (x[m, nq, 3], detJ[m, nq])     the element map actually used (isoparametric if curved)
    ev.velocity(ref)           -> (u[m, nq, 3], grad_u[m, nq, 3, 3])   grad_u[..., i, j] = d u_i / d x_j
    ev.pressure(ref)           -> p[m, nq]
ref are points of the reference cell in the basix/DOLFINx convention: tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1);
hexahedron [0,1]^3 with vertices in lexicographic order (x fastest).  detJ may be negative for left-handed vertex
orders; |detJ| is used.

Every functional returns PARTIAL SUMS (dicts of floats) over the evaluator's elements; `combine` adds the parts of
all ranks and finishes the norms.  So an MPI runner reduces sums, never gathers fields.

Geometry: every element's geometry nodes are fixed here: affine tetrahedra (4 vertices), trilinear (affine) cube
hexahedra (8 vertices), and the curved sphere elements as 10-node P2 tetrahedra / 27-node Q2 hexahedra whose nodes
are the images of the reference nodes (REF_P2_TET / REF_Q2_HEX, basix order) under the exact cubed-sphere map.
`geometry_defect` compares the evaluator's element map with this module's at the quadrature points: the solver must
use exactly the fixed geometry.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

QUAD_DEGREE_TET = 8       # collapsed Gauss-Legendre, 6 points per direction: exact to degree 8 on the tetrahedron
QUAD_POINTS_HEX = 5       # tensor Gauss-Legendre: exact to degree 9 per direction

REF_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
REF_P2_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                       [0, 0, .5], [0, .5, 0], [.5, 0, 0]], float)
REF_Q2_HEX = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
                       [.5, 0, 0], [0, .5, 0], [0, 0, .5], [1, .5, 0], [1, 0, .5], [.5, 1, 0], [0, 1, .5],
                       [1, 1, .5], [.5, 0, 1], [0, .5, 1], [1, .5, 1], [.5, 1, 1], [.5, .5, 0], [.5, 0, .5],
                       [0, .5, .5], [1, .5, .5], [.5, 1, .5], [.5, .5, 1], [.5, .5, .5]], float)
HEX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
HEX_FACES = [(0, 1, 3, 2), (0, 1, 5, 4), (0, 2, 6, 4), (1, 3, 7, 5), (2, 3, 7, 6), (4, 5, 7, 6)]
TET_FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FIELD_DIR = "tools/fluids_mms/out"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ============================================================================================== exact fields
def _compile(spec: dict):
    code = compile(spec["numpy"], "<mms expression>", "eval")

    def f(X, Y, Z):
        v = eval(code, {"__builtins__": {}, "np": np, "numpy": np}, {"x": X, "y": Y, "z": Z})
        return np.broadcast_to(np.asarray(v, dtype=float), np.shape(X))
    return f


class Field:
    """Exact solution and data of one generated field (tools/fluids_mms/out/f1c_<name>.json), sha256-verified, in
    PHYSICAL coordinates (the frame of a rotated field is applied here)."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self._u = [_compile(s) for s in data["u"]]
        self._g = [[_compile(s) for s in row] for row in data["grad_u"]]
        self._p = _compile(data["p"])
        self._mu = _compile(data["mu"])
        self._f = {form: [_compile(s) for s in fd["f"]] for form, fd in data["forms"].items()}
        self._sig = {form: [[_compile(s) for s in row] for row in fd["sigma"]] for form, fd in data["forms"].items()}
        self.constants = data["constants"]
        fr = data.get("frame")
        self.R = np.array(fr["R_float"]) if fr else np.eye(3)
        self.c = np.array(fr["c"]) if fr else np.zeros(3)
        self.rotated = fr is not None

    def local(self, P):
        return (np.asarray(P, float) - self.c) @ self.R          # x_loc = R^T (x - c), row vectors

    def _ev(self, fs, P):
        L = self.local(P)
        return np.stack([f(L[..., 0], L[..., 1], L[..., 2]) for f in fs], -1)

    def u(self, P):
        return self._ev(self._u, P) @ self.R.T

    def grad_u(self, P):
        L = self.local(P)
        G = np.stack([np.stack([self._g[i][j](L[..., 0], L[..., 1], L[..., 2]) for j in range(3)], -1)
                      for i in range(3)], -2)
        return self.R @ G @ self.R.T

    def p(self, P):
        L = self.local(P)
        return self._p(L[..., 0], L[..., 1], L[..., 2])

    def mu(self, P):
        L = self.local(P)
        return self._mu(L[..., 0], L[..., 1], L[..., 2])

    def f(self, form, P):
        return self._ev(self._f[form], P) @ self.R.T

    def sigma(self, form, P):
        L = self.local(P)
        S = np.stack([np.stack([self._sig[form][i][j](L[..., 0], L[..., 1], L[..., 2]) for j in range(3)], -1)
                      for i in range(3)], -2)
        return self.R @ S @ self.R.T

    def traction(self, form, P, n):
        """g = sigma_form n, n the outward unit normal (physical)."""
        return np.einsum("...ij,j->...i", self.sigma(form, P), np.asarray(n, float))


def load_field(entry: dict, root: Path | None = None) -> Field:
    root = Path(root) if root is not None else _repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return Field(json.loads(raw))


# ============================================================================================== duct series
DUCT = {"W": 1.0, "H": 0.5, "L": 2.0, "G": 2.0, "mu": 1.0}


def duct_u(xp, yp, W=1.0, H=0.5, G=2.0, mu=1.0, tol=1e-15, nmax=200001):
    """Fully developed flow in the rectangular duct [0, W] x [0, H] (axis z): -mu lap w = G, w = 0 on the walls.
    Returns (w, dw/dx, dw/dy).  Evaluated per point from whichever of the two equivalent series converges faster
    there, written about the duct centre (X = x - W/2, Y = y - H/2, a = W/2, b = H/2):
      x-series:  w = G/(2mu) (a^2 - X^2) - (16 G a^2/(mu pi^3)) sum_{k>=0} (-1)^k/n^3 cos(n pi X/2a)
                 cosh(n pi Y/2a)/cosh(n pi b/2a),  n = 2k+1
      y-series:  the same with (a, X) <-> (b, Y).
    Truncation: the k-th correction term is bounded by (16 G a^2/(mu pi^3)) n^-3 exp(-n pi d/2a), d = b - |Y| the
    distance to the far walls; the tail after n_K is bounded by the next term times 1/(1 - exp(-pi d/a)), and the sum
    stops when that bound is below tol * G a^2/mu.  Points ON a wall return the exact 0 (and the exact one-sided
    derivative of the other series).  Points within ~1e-5 of a duct EDGE (both distances tiny but nonzero) may exceed
    nmax and raise; the fixed quadrature points stay >= 2.3e-3 from the edges at h = 1/32.  The tail
    bound omits a factor <= 2 (cosh A / cosh B <= 2 exp(-(B - A))), immaterial at tol = 1e-15."""
    xp, yp = np.broadcast_arrays(np.asarray(xp, float), np.asarray(yp, float))
    shp = xp.shape
    Xc, Yc = xp.ravel() - W / 2, yp.ravel() - H / 2
    a, b = W / 2, H / 2
    w = np.empty_like(Xc)
    wx = np.empty_like(Xc)
    wy = np.empty_like(Xc)
    dx_far = b - np.abs(Yc)          # x-series converges with exp(-n pi dx_far / 2a)
    dy_far = a - np.abs(Xc)
    use_x = dx_far / a >= dy_far / b
    for sel, (A, B, P, Q, flip) in ((use_x, (a, b, Xc, Yc, False)), (~use_x, (b, a, Yc, Xc, True))):
        idx = np.where(sel)[0]
        if not len(idx):
            continue
        Pp, Qp = P[idx], Q[idx]
        d = np.maximum(B - np.abs(Qp), 0.0)
        amp = 16 * G * A * A / (mu * math.pi ** 3)
        ww = G / (2 * mu) * (A * A - Pp ** 2)
        wp = -G / mu * Pp
        wq = np.zeros_like(Pp)
        corner = d < 1e-12 * B          # only at the duct edges (corners of the section): w = 0, grad w = 0
        wp[corner] = 0.0
        active = ~corner
        k = 0
        while True:
            n = 2 * k + 1
            if n > nmax:
                raise RuntimeError("duct series did not converge (point too close to a corner)")
            ii = np.where(active)[0]
            if not len(ii):
                break
            kn = n * math.pi / (2 * A)
            # cosh(kn Q)/cosh(kn B) and sinh(kn Q)/cosh(kn B) without overflow
            e1 = np.exp(-kn * (B - Qp[ii]))
            e2 = np.exp(-kn * (B + Qp[ii]))
            den = 1 + np.exp(-2 * kn * B)
            ch = (e1 + e2) / den
            sh = (e1 - e2) / den
            sgn = -1.0 if k % 2 else 1.0
            c = amp * sgn / n ** 3
            ww[ii] -= c * np.cos(kn * Pp[ii]) * ch
            wp[ii] += c * kn * np.sin(kn * Pp[ii]) * ch
            wq[ii] -= c * kn * np.cos(kn * Pp[ii]) * sh
            # tail bound of the value and of the derivatives (one more power of kn)
            nn = n + 2
            dd = d[ii]
            r = np.exp(-math.pi * dd / A)
            nxt = amp / nn ** 3 * np.exp(-nn * math.pi * dd / (2 * A)) / np.maximum(1 - r, 1e-300)
            tail = nxt * max(1.0, nn * math.pi / (2 * A))
            done = tail < tol * G * A * A / mu
            active[ii[done]] = False
            k += 1
        w[idx] = ww
        if flip:
            wx[idx], wy[idx] = wq, wp
        else:
            wx[idx], wy[idx] = wp, wq
    return w.reshape(shp), wx.reshape(shp), wy.reshape(shp)


class DuctField:
    """Exact fully developed duct flow u = (0, 0, w(x, y)), p = G (L - z): the closed-form side of the series."""

    def __init__(self, W=1.0, H=0.5, L=2.0, G=2.0, mu=1.0):
        self.W, self.H, self.L, self.G, self.mu_ = W, H, L, G, mu
        self.name = "duct_series"

    def u(self, P):
        w, _, _ = duct_u(P[..., 0], P[..., 1], self.W, self.H, self.G, self.mu_)
        return np.stack([np.zeros_like(w), np.zeros_like(w), w], -1)

    def grad_u(self, P):
        _, wx, wy = duct_u(P[..., 0], P[..., 1], self.W, self.H, self.G, self.mu_)
        G = np.zeros(P.shape[:-1] + (3, 3))
        G[..., 2, 0], G[..., 2, 1] = wx, wy
        return G

    def p(self, P):
        return self.G * (self.L - P[..., 2])

    def mu(self, P):
        return np.full(P.shape[:-1], self.mu_)

    def sigma(self, form, P):
        G = self.grad_u(P)
        S = self.mu_ * (G + np.swapaxes(G, -1, -2) if form == "sym" else G)
        return S - self.p(P)[..., None, None] * np.eye(3)

    def traction(self, form, P, n):
        return np.einsum("...ij,j->...i", self.sigma(form, P), np.asarray(n, float))

    def f(self, form, P):
        return np.zeros(P.shape[:-1] + (3,))


def duct_inlet_flux(mesh, W=1.0, H=0.5, G=2.0, mu=1.0) -> float:
    """Exact integral over the inlet z = 0 of the P2 NODAL interpolant of the series profile (the flux the Dirichlet
    data impose).  For Taylor-Hood, testing continuity with q = z (in P1) gives (1/L) int u_h,z dV = this flux to
    round-off whenever the inflow is Dirichlet: the duct's discrete flux-balance check (2e-15 with an independent
    reference solver)."""
    fld = DuctField(W, H, 2.0, G, mu)
    bf = [f for f in mesh.boundary_faces() if np.all(np.abs(mesh.V[list(f)][:, 2]) < 1e-12)]
    t, _ = p2_face_flux_defect(mesh, fld.u, bf)
    return abs(t)


# ============================================================================================== quadrature
def tet_rule(deg: int = QUAD_DEGREE_TET):
    """Collapsed (Duffy) Gauss-Legendre rule on the reference tetrahedron: x = a(1-b)(1-c), y = b(1-c), z = c with
    Jacobian (1-b)(1-c)^2; n = ceil((deg+3)/2) points per direction makes it exact to degree deg.  Weights sum to 1/6."""
    n = math.ceil((deg + 3) / 2)
    g, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (g + 1), 0.5 * w
    A, B, C = np.meshgrid(s, s, s, indexing="ij")
    WA, WB, WC = np.meshgrid(ws, ws, ws, indexing="ij")
    pts = np.c_[(A * (1 - B) * (1 - C)).ravel(), (B * (1 - C)).ravel(), C.ravel()]
    return pts, (WA * WB * WC * (1 - B) * (1 - C) ** 2).ravel()


def hex_rule(n: int = QUAD_POINTS_HEX):
    g, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (g + 1), 0.5 * w
    A, B, C = np.meshgrid(s, s, s, indexing="ij")
    WA, WB, WC = np.meshgrid(ws, ws, ws, indexing="ij")
    return np.c_[A.ravel(), B.ravel(), C.ravel()], (WA * WB * WC).ravel()


def rule(kind: str):
    return tet_rule() if kind == "tet" else hex_rule()


# ============================================================================================== reference Lagrange bases
def _monomials(kind):
    if kind == "p2":
        return [(i, j, k) for i in range(3) for j in range(3) for k in range(3) if i + j + k <= 2]
    if kind == "q2":
        return [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    if kind == "p1":
        return [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if kind == "q1":
        return [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    raise ValueError(kind)


_BASIS_CACHE = {}


def _coeffs(kind):
    if kind not in _BASIS_CACHE:
        nodes = {"p2": REF_P2_TET, "q2": REF_Q2_HEX, "p1": REF_TET_VERTS, "q1": REF_Q2_HEX[:8]}[kind]
        mon = _monomials(kind)
        V = np.array([[x ** i * y ** j * z ** k for (i, j, k) in mon] for (x, y, z) in nodes])
        _BASIS_CACHE[kind] = (mon, np.linalg.inv(V))
    return _BASIS_CACHE[kind]


def lagrange(kind, ref):
    """Values [nq, nb] and reference gradients [nq, nb, 3] of the Lagrange basis whose nodes are REF_P2_TET (p2),
    REF_Q2_HEX (q2), the tet vertices (p1) or the hex vertices (q1), in that node order."""
    mon, Ci = _coeffs(kind)
    ref = np.asarray(ref, float)
    x, y, z = ref[:, 0], ref[:, 1], ref[:, 2]

    def pw(t, e):
        return t ** e if e > 0 else np.ones_like(t)

    def dpw(t, e):
        return e * t ** (e - 1) if e > 0 else np.zeros_like(t)
    M = np.stack([pw(x, i) * pw(y, j) * pw(z, k) for (i, j, k) in mon], -1)
    D = np.stack([np.stack([dpw(x, i) * pw(y, j) * pw(z, k), pw(x, i) * dpw(y, j) * pw(z, k),
                            pw(x, i) * pw(y, j) * dpw(z, k)], -1) for (i, j, k) in mon], -2)
    return M @ Ci, np.einsum("qmd,mb->qbd", D, Ci)


# ============================================================================================== meshes
class FrozenMesh:
    """kind 'tet' or 'hex'; V[nv, 3] vertices; E[ne, 4|8] vertex indices (basix vertex order); optional curved
    geometry nodes X[ne, 10|27, 3] (REF_P2_TET / REF_Q2_HEX order); boundary face attributes per face key."""

    def __init__(self, kind, V, E, X=None, family="", level=0, h_nominal=None, attr_fn=None):
        self.kind, self.V, self.E, self.X = kind, np.asarray(V, float), np.asarray(E, np.int64), X
        self.family, self.level, self.h_nominal = family, level, h_nominal
        self.attr_fn = attr_fn

    @property
    def n_elements(self):
        return len(self.E)

    def geometry_nodes(self):
        """10/27 geometry nodes per element: the fixed curved nodes, else the affine/trilinear interpolation."""
        if self.X is not None:
            return self.X
        ref = REF_P2_TET if self.kind == "tet" else REF_Q2_HEX
        phi, _ = lagrange("p1" if self.kind == "tet" else "q1", ref)
        return np.einsum("qv,evd->eqd", phi, self.V[self.E])

    def map(self, ref, ids=None):
        """(x[m, nq, 3], J[m, nq, 3, 3]) of the fixed element map at reference points."""
        ids = np.arange(self.n_elements) if ids is None else np.asarray(ids)
        if self.X is not None:
            phi, dphi = lagrange("p2" if self.kind == "tet" else "q2", ref)
            N = self.X[ids]
        else:
            phi, dphi = lagrange("p1" if self.kind == "tet" else "q1", ref)
            N = self.V[self.E[ids]]
        x = np.einsum("qb,ebd->eqd", phi, N)
        J = np.einsum("qbk,ebd->eqdk", dphi, N)
        return x, J

    def faces(self):
        """{sorted vertex tuple: [element ids]} over all element faces."""
        loc = TET_FACES if self.kind == "tet" else HEX_FACES
        out = {}
        for e, el in enumerate(self.E):
            for f in loc:
                out.setdefault(tuple(sorted(int(el[i]) for i in f)), []).append(e)
        return out

    def boundary_faces(self):
        return [k for k, v in self.faces().items() if len(v) == 1]


def _kuhn_tets_local():
    """The 6 Kuhn tetrahedra of the unit cube as paths 0 -> e_a -> e_a + e_b -> (1,1,1) (vertex order = path order)."""
    out = []
    for perm in itertools.permutations(range(3)):
        p = [np.zeros(3, int)]
        for ax in perm:
            q = p[-1].copy()
            q[ax] = 1
            p.append(q)
        out.append(p)
    return out


def kuhn_box(n, box=((0, 1), (0, 1), (0, 1)), octant=True):
    """Structured box mesh, n = (nx, ny, nz) cells, each split into the 6 Kuhn tetrahedra.  octant=True: in each
    half-box (octant about the box centre) the Kuhn diagonal is reflected so that it points to that octant's corner of
    the box; every cell's diagonal then runs from its corner nearest the box corner of its octant, and every
    tetrahedron has an interior vertex (the 3D Taylor-Hood mesh condition).  octant=False: the
    standard Kuhn mesh (all diagonals along (1,1,1)): six box corners then carry tetrahedra with all four vertices on
    the boundary (used only by the N1 mesh-condition case).  Needs even n per direction for octant=True.  Vertex
    order of each tetrahedron = the Kuhn path."""
    nx, ny, nz = n
    if octant and (nx % 2 or ny % 2 or nz % 2):
        raise ValueError("octant Kuhn meshes need even cell counts")
    xs = [np.linspace(b[0], b[1], m + 1) for b, m in zip(box, n)]
    Xg = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    vid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    paths = _kuhn_tets_local()
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s = [1, 1, 1]
                if octant:
                    s = [1 if 2 * i < nx else -1, 1 if 2 * j < ny else -1, 1 if 2 * k < nz else -1]
                for p in paths:
                    T.append([vid(i + (q[0] if s[0] > 0 else 1 - q[0]), j + (q[1] if s[1] > 0 else 1 - q[1]),
                                  k + (q[2] if s[2] > 0 else 1 - q[2])) for q in p])
    return Xg, np.array(T, np.int64)


def five_tet_cube(N):
    """N1 negative-control mesh (never an order-study family): each cell split into 5 tetrahedra (one central, four
    corner tetrahedra), the split reflected in x in odd cells (i + j + k odd) so that face diagonals match.  The
    cube corners whose cell has a corner tetrahedron there carry a tetrahedron with ALL SIX edges on the boundary
    (every P2 velocity node on the boundary): a certain Taylor-Hood violation."""
    V, _ = kuhn_box((N, N, N), octant=False)
    vid = lambda i, j, k: (i * (N + 1) + j) * (N + 1) + k  # noqa: E731
    even = [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [(1, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
            [(1, 0, 1), (1, 0, 0), (0, 0, 1), (1, 1, 1)], [(0, 1, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)],
            [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]]
    T = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                odd = (i + j + k) % 2
                for tet in even:
                    T.append([vid(i + (1 - a if odd else a), j + b, k + c) for a, b, c in tet])
    return FrozenMesh("tet", V, np.array(T, np.int64), family="five_tet", level=0, h_nominal=1.0 / N)


def hex_box(n, box=((0, 1), (0, 1), (0, 1))):
    nx, ny, nz = n
    xs = [np.linspace(b[0], b[1], m + 1) for b, m in zip(box, n)]
    Xg = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    vid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    H = [[vid(i + a, j + b, k + c) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
         for i in range(nx) for j in range(ny) for k in range(nz)]
    return Xg, np.array(H, np.int64)


def bey_refine(V, T):
    """Bey's red refinement (1995) of tetrahedra with ordered vertices (x0, x1, x2, x3) into 8 children; for Kuhn
    tetrahedra the children are the Kuhn tetrahedra of the halved grid (same reflected frame).  New vertices (edge
    midpoints) are appended in the order of the sorted unique edge list (deterministic)."""
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    E = np.concatenate([np.sort(T[:, [a, b]], axis=1) for a, b in pairs])
    uniq, inv = np.unique(E, axis=0, return_inverse=True)
    mid = len(V) + inv.reshape(len(pairs), len(T)).T                     # [ne, 6] in the order of `pairs`
    V2 = np.vstack([V, 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])])
    x0, x1, x2, x3 = T.T
    x01, x02, x03, x12, x13, x23 = mid.T
    kids = [(x0, x01, x02, x03), (x01, x1, x12, x13), (x02, x12, x2, x23), (x03, x13, x23, x3),
            (x01, x02, x03, x13), (x01, x02, x12, x13), (x02, x03, x13, x23), (x02, x12, x13, x23)]
    T2 = np.stack([np.stack(k, -1) for k in kids], 1).reshape(-1, 4)
    return V2, T2


def cube_tet(N, kind="octant"):
    """The fixed tetrahedral cube family 'kuhn': octant-reflected Kuhn mesh of N^3 cells (N in 4, 8, 16, 32).
    Nested: the level-N mesh is the Bey refinement of the level-N/2 mesh (checked by the self-tests)."""
    V, T = kuhn_box((N, N, N), octant=(kind == "octant"))
    return FrozenMesh("tet", V, T, family=f"kuhn_{kind}", level=int(round(math.log2(N / 4))), h_nominal=1.0 / N)


def cube_tet_distorted(k, N0=4, rel=0.2, seed=20261101):
    """The fixed distorted family: the octant Kuhn N0 = 4 base, every INTERIOR vertex displaced by
    rel * (1/N0) * U(-1,1)^3 (numpy Generator(PCG64(seed)), vertex order), then k Bey refinements (k = 0..3)."""
    V, T = kuhn_box((N0, N0, N0))
    inner = np.where(np.all((V > 1e-12) & (V < 1 - 1e-12), axis=1))[0]
    rng = np.random.Generator(np.random.PCG64(seed))
    V = V.copy()
    V[inner] += rel * (1.0 / N0) * rng.uniform(-1, 1, (len(inner), 3))
    for _ in range(k):
        V, T = bey_refine(V, T)
    return FrozenMesh("tet", V, T, family="kuhn_distorted", level=k, h_nominal=0.25 / 2 ** k)


def cube_hex(N):
    V, H = hex_box((N, N, N))
    return FrozenMesh("hex", V, H, family="hex", level=int(round(math.log2(N / 4))), h_nominal=1.0 / N)


def rotated_cube_tet(N, R):
    """T4 family: the octant Kuhn cube mesh mapped by x = R x_loc (R the field's frame, rational)."""
    m = cube_tet(N)
    return FrozenMesh("tet", m.V @ np.asarray(R, float).T, m.E, family="kuhn_rotated", level=m.level,
                      h_nominal=m.h_nominal)


def duct_tet(h):
    """P1 family: the duct [0, 1] x [0, 1/2] x [0, 2] (axis z), octant Kuhn cells of size h (h = 1/4 ... 1/32)."""
    n = (int(round(1 / h)), int(round(0.5 / h)), int(round(2 / h)))
    V, T = kuhn_box(n, box=((0, 1), (0, 0.5), (0, 2)))
    return FrozenMesh("tet", V, T, family="duct", level=int(round(math.log2(0.25 / h))), h_nominal=h)


# ---------------------------------------------------------------------------------------------- cubed-sphere shell
SPHERE = {"a": 1.0, "R": 4.0}


def _sphere_map(face, s1, s2, t, a=1.0, R=4.0):
    """Exact map of the cubed-sphere shell: face = (axis, sign); equiangular cube coordinates
    c1 = tan(pi s1/4), c2 = tan(pi s2/4) (s in [-1, 1]) on the two remaining global axes IN INCREASING ORDER;
    radius r = a (R/a)^t, t in [0, 1] (geometric radial spacing)."""
    ax, sg = face
    others = [d for d in range(3) if d != ax]
    c = np.zeros(np.broadcast(s1, s2, t).shape + (3,))
    c[..., ax] = sg
    c[..., others[0]] = np.tan(np.pi * np.asarray(s1) / 4)
    c[..., others[1]] = np.tan(np.pi * np.asarray(s2) / 4)
    rr = a * (R / a) ** np.asarray(t)
    return c / np.linalg.norm(c, axis=-1, keepdims=True) * rr[..., None]


def sphere_shell(n, kind="hex", a=1.0, R=4.0):
    """The fixed sphere family: 6 patches x n x n angular x n radial parametric hexahedra (n = 2, 4, 8, 16), each
    mapped exactly (_sphere_map).  kind 'hex': Q2 geometry, 27 nodes per element at the images of REF_Q2_HEX;
    kind 'tet': each parametric hexahedron split into its 6 Kuhn tetrahedra along the parametric diagonal from the
    all-low to the all-high corner (the patch parameters follow the global axes in increasing order, so the face
    diagonals of neighbouring patches coincide: conforming), P2 geometry, 10 nodes per element at the images of
    REF_P2_TET.  Vertex order: right-handed (|det J| > 0 with det J > 0).  h_nominal = 1/n (parametric)."""
    grid = np.linspace(-1, 1, n + 1)
    tg = np.linspace(0, 1, n + 1)
    verts, vkey = [], {}

    def vid(P):
        k = tuple(np.round(P, 9) + 0.0)
        if k not in vkey:
            vkey[k] = len(verts)
            verts.append(P)
        return vkey[k]
    E, X = [], []
    ref = REF_Q2_HEX if kind == "hex" else REF_P2_TET
    paths = _kuhn_tets_local()
    for ax in range(3):
        for sg in (-1, 1):
            face = (ax, sg)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        lo = np.array([grid[i], grid[j], tg[k]])
                        d = np.array([grid[i + 1] - grid[i], grid[j + 1] - grid[j], tg[k + 1] - tg[k]])
                        cells = []
                        if kind == "hex":
                            cells.append([np.array([c0, c1, c2]) for c2 in (0, 1) for c1 in (0, 1) for c0 in (0, 1)])
                        else:
                            cells += [[np.array(q) for q in p] for p in paths]
                        for corners in cells:
                            cp = [lo + d * c for c in corners]
                            Pv = [_sphere_map(face, *q, a=a, R=R) for q in cp]
                            # orientation from the mapped vertices
                            if kind == "hex":
                                J = np.array([Pv[1] - Pv[0], Pv[2] - Pv[0], Pv[4] - Pv[0]]).T
                            else:
                                J = np.array([Pv[1] - Pv[0], Pv[2] - Pv[0], Pv[3] - Pv[0]]).T
                            order = list(range(len(cp)))
                            if np.linalg.det(J) < 0:
                                # swap the first two parametric axes (hex) / two vertices (tet): same cell, other hand
                                order = [0, 2, 1, 3, 4, 6, 5, 7] if kind == "hex" else [0, 2, 1, 3]
                            cpo = [cp[o] for o in order]
                            E.append([vid(_sphere_map(face, *q, a=a, R=R)) for q in cpo])
                            # geometry nodes: parametric affine (tet) / trilinear (hex) map of the reference nodes
                            if kind == "hex":
                                phi, _ = lagrange("q1", ref)
                            else:
                                phi, _ = lagrange("p1", ref)
                            par = phi @ np.array(cpo)
                            X.append(_sphere_map(face, par[:, 0], par[:, 1], par[:, 2], a=a, R=R))
    return FrozenMesh(kind, np.array(verts), np.array(E, np.int64), X=np.array(X), family=f"sphere_{kind}",
                      level=int(round(math.log2(n / 2))), h_nominal=1.0 / n)


# ---------------------------------------------------------------------------------------------- mesh checks
def mesh_hash(mesh: FrozenMesh, digits: int = 9) -> str:
    """Numbering-independent hash: each element as the sorted tuple of its rounded geometry-node coordinates
    (vertices, and the curved nodes when present), the list sorted."""
    N = np.round(mesh.geometry_nodes() if mesh.X is not None else mesh.V[mesh.E], digits) + 0.0
    els = sorted(tuple(sorted(tuple(p) for p in el)) for el in N)
    return hashlib.sha256(repr(els).encode()).hexdigest()


def boundary_vertex_set(mesh: FrozenMesh):
    return {v for f in mesh.boundary_faces() for v in f}


_TE = np.array(TET_EDGES)
_FACE_EDGES = np.array([[3, 4, 5], [1, 2, 5], [0, 2, 4], [0, 1, 3]])   # edges (into TET_EDGES) of the face opposite v


def taylor_hood_condition(mesh: FrozenMesh) -> dict:
    """The 3D Taylor-Hood mesh condition, three counts for tetrahedral meshes:
    (a) tets_without_interior_vertex: tetrahedra with all four vertices on the boundary.  BOFFI'S HYPOTHESIS
        (Boffi 1997, as restated by Boffi-Brezzi-Fortin, CIME notes LNM 1939, Theorem th:htgen): every tetrahedron
        has at least one internal vertex -> P_k/P_{k-1} (k >= 2) satisfies the inf-sup condition;
    (b) tets_violating_weak_boffi: tetrahedra WITHOUT three edges off the boundary that are not coplanar (the
        weakening in the remark after that proof; an edge lies on the boundary iff it is an edge of a boundary face;
        three edges of a tetrahedron are coplanar iff they bound one face);
    (c) tets_all_p2_nodes_on_boundary: all six edges on the boundary (every P2 velocity node constrained): a CERTAIN
        spurious pressure mode, a diagnostic, not the condition.
    (a) and (b) are sufficient conditions.  Hexahedra: the number with all 8 vertices on the boundary (warned only)."""
    E = np.asarray(mesh.E, np.int64)
    bf = mesh.boundary_faces()
    nv = len(mesh.V)
    bvert = np.zeros(nv, bool)
    for f in bf:
        bvert[list(f)] = True
    out = {}
    if mesh.kind == "tet":
        bfa = np.array(bf, np.int64)
        be = np.concatenate([np.sort(bfa[:, [i, j]], axis=1) for i, j in ((0, 1), (0, 2), (1, 2))])
        bkey = np.unique(be[:, 0] * nv + be[:, 1])
        ed = np.sort(E[:, _TE], axis=2)
        inner = ~np.isin(ed[..., 0] * nv + ed[..., 1], bkey)
        n_in = inner.sum(axis=1)
        three_face = np.zeros(len(E), bool)
        for fe in _FACE_EDGES:
            three_face |= inner[:, fe].all(axis=1)
        ok_b = (n_in >= 4) | ((n_in == 3) & ~three_face)
        out["tets_without_interior_vertex"] = int(bvert[E].all(axis=1).sum())
        out["tets_violating_weak_boffi"] = int((~ok_b).sum())
        out["tets_all_p2_nodes_on_boundary"] = int((n_in == 0).sum())
    else:
        out["hexes_all_vertices_on_boundary"] = int(bvert[E].all(axis=1).sum())
    return out


def h_max(mesh: FrozenMesh) -> float:
    P = mesh.V[mesh.E]
    pairs = TET_EDGES if mesh.kind == "tet" else HEX_EDGES
    return float(max(np.max(np.linalg.norm(P[:, a] - P[:, b], axis=1)) for a, b in pairs))


def nested_in(fine: FrozenMesh, coarse: FrozenMesh) -> bool:
    """Every coarse vertex is a fine vertex, and (straight tetrahedra) every fine tetrahedron lies inside one coarse
    tetrahedron (barycentric test of its centroid and vertices), for nested families."""
    here = {tuple(np.round(v, 9) + 0.0) for v in fine.V}
    if not all(tuple(np.round(v, 9) + 0.0) in here for v in coarse.V):
        return False
    if fine.kind != "tet" or fine.X is not None:
        return True
    Pc = coarse.V[coarse.E]
    Minv = np.linalg.inv(np.stack([Pc[:, 1] - Pc[:, 0], Pc[:, 2] - Pc[:, 0], Pc[:, 3] - Pc[:, 0]], -1))
    cen_c = Pc.mean(1)
    Pf = fine.V[fine.E]
    cen = Pf.mean(1)
    # nearest coarse elements by centroid distance, check containment among a few candidates
    order = np.argsort(np.linalg.norm(cen[:, None, :] - cen_c[None, :, :], axis=-1), axis=1)[:, :8] \
        if len(cen) * len(cen_c) <= 4e7 else None
    if order is None:
        return True
    for e in range(len(Pf)):
        ok = False
        for c in order[e]:
            lam = (Minv[c] @ (Pf[e] - Pc[c, 0]).T).T
            l0 = 1 - lam.sum(1)
            if np.all(lam > -1e-9) and np.all(l0 > -1e-9):
                ok = True
                break
        if not ok:
            return False
    return True


def verify_mesh(mesh_solver: FrozenMesh, mesh_frozen: FrozenMesh, coarser: FrozenMesh | None = None) -> dict:
    """The solver's mesh against the fixed construction: counts, numbering-independent hash (geometry nodes
    included for curved meshes), the Taylor-Hood condition, h_max and nestedness with the next coarser level."""
    out = {"n_elements": int(mesh_solver.n_elements), "n_vertices": int(len(mesh_solver.V)),
           "counts_match": bool(mesh_solver.n_elements == mesh_frozen.n_elements and
                                len(mesh_solver.V) == len(mesh_frozen.V)),
           "hash": mesh_hash(mesh_solver), "hash_frozen": mesh_hash(mesh_frozen), "h_max": h_max(mesh_solver)}
    out["hash_match"] = out["hash"] == out["hash_frozen"]
    out.update(taylor_hood_condition(mesh_solver))
    if coarser is not None:
        out["nested"] = nested_in(mesh_solver, coarser)
    return out


# ============================================================================================== norms (partial sums)
def _empty(ev) -> bool:
    """A rank may own no element (coarse meshes on many ranks): its partial sums are zero and its maxima -inf."""
    return len(np.asarray(ev.ids)) == 0


def _qw(mesh: FrozenMesh, ev):
    q, w = rule(mesh.kind)
    x, detJ = ev.geometry(q)
    if np.any(np.abs(detJ) <= 0):
        raise RuntimeError("degenerate element map")
    return q, x, w[None, :] * np.abs(detJ)


def geometry_defect(mesh: FrozenMesh, ev) -> dict:
    """max |x_ev - x_frozen| at the quadrature points of the evaluator's elements (round-off for the fixed map) and
    max relative |detJ_ev - detJ_frozen|; partial (combine takes the max)."""
    if _empty(ev):
        return {"max:geometry_x": 0.0, "max:geometry_detJ_rel": 0.0}
    q, _ = rule(mesh.kind)
    xe, de = ev.geometry(q)
    xf, J = mesh.map(q, ev.ids)
    df = np.linalg.det(J)
    return {"max:geometry_x": float(np.max(np.abs(xe - xf))),
            "max:geometry_detJ_rel": float(np.max(np.abs(np.abs(de) - np.abs(df)) / np.abs(df)))}


def norm_sums(mesh: FrozenMesh, ev, fld, p_exact=None, p_shift: float = 0.0) -> dict:
    """Partial sums for uL2, uH1 (all nine gradient components), divL2, the pressure in both conventions
    (raw and mean-free: means from the same quadrature), and magnitudes; combine() finishes them.  Two-pass
    centering: p_shift = mean(p_h) - mean(p) from a first pass (pressure_mean_sums and the exact mean)
    makes pL2m round-off free; p_shift = 0 is the one-pass formula (floor ~ sqrt(eps) |mean difference|, harmless
    for the order items whose errors are >= 4e-5)."""
    if _empty(ev):
        return {k: 0.0 for k in ("sum:uL2sq", "sum:uH1sq", "sum:divsq", "sum:vol", "sum:ph", "sum:pe", "sum:ph2",
                                 "sum:pe2", "sum:phpe", "sum:pdiff2", "sum:uhz", "sum:uh2", "sum:pshiftres")} | \
            {"max:u_magnitude": -math.inf, "max:p_magnitude": -math.inf}
    q, xq, W = _qw(mesh, ev)
    uh, guh = ev.velocity(q)
    ph = ev.pressure(q)
    ue, ge = fld.u(xq), fld.grad_u(xq)
    pe = (p_exact or fld.p)(xq)
    return {"sum:uL2sq": float((W[..., None] * (uh - ue) ** 2).sum()),
            "sum:uH1sq": float((W[..., None, None] * (guh - ge) ** 2).sum()),
            "sum:divsq": float((W * np.trace(guh, axis1=-2, axis2=-1) ** 2).sum()),
            "sum:vol": float(W.sum()), "sum:ph": float((W * ph).sum()), "sum:pe": float((W * pe).sum()),
            "sum:ph2": float((W * ph * ph).sum()), "sum:pe2": float((W * pe * pe).sum()),
            "sum:phpe": float((W * ph * pe).sum()), "sum:pdiff2": float((W * (ph - pe - p_shift) ** 2).sum()),
            "sum:pshiftres": float((W * (ph - pe - p_shift)).sum()), "pshift": p_shift,
            "sum:uhz": float((W * uh[..., 2]).sum()), "sum:uh2": float((W[..., None] * uh ** 2).sum()),
            "max:u_magnitude": float(np.max(np.linalg.norm(ue, axis=-1))),
            "max:p_magnitude": float(np.max(np.abs(pe)))}


def combine(parts: list[dict]) -> dict:
    """Add the 'sum:' entries and take the max of the 'max:' entries over all parts, then finish the norms."""
    acc = {}
    for d in parts:
        for k, v in d.items():
            if k.startswith("max:"):
                acc[k] = max(acc.get(k, -math.inf), v)
            elif k == "pshift":
                acc[k] = v                     # the same shift on every rank
            else:
                acc[k] = acc.get(k, 0.0) + v
    out = {k.split(":", 1)[1]: v for k, v in acc.items() if k.startswith("max:")}
    if "sum:uL2sq" in acc:
        vol = acc["sum:vol"]
        shift = acc.get("pshift", 0.0)
        mh, me = acc["sum:ph"] / vol, acc["sum:pe"] / vol
        # ||(ph - mh) - (pe - me)||^2 = int (ph - pe - s)^2 - vol (mh - me - s)^2 for any shift s (two-pass: s ~ mh - me)
        rm = acc.get("sum:pshiftres", (mh - me - shift) * vol) / vol
        pm2 = acc["sum:pdiff2"] - vol * rm ** 2
        out.update({"uL2": math.sqrt(acc["sum:uL2sq"]), "uH1": math.sqrt(acc["sum:uH1sq"]),
                    "divL2": math.sqrt(acc["sum:divsq"]), "pL2": math.sqrt(max(acc["sum:pdiff2"], 0.0)) if not shift else None,
                    "pL2m": math.sqrt(max(pm2, 0.0)), "volume": vol, "pmean_h": mh, "pmean": me,
                    "uz_mean": acc["sum:uhz"] / vol, "uh_L2": math.sqrt(acc["sum:uh2"])})
    for k, v in acc.items():
        if k.startswith("sum:") and k[4:] not in out:
            out[k[4:]] = v
    return out


def pressure_mean_sums(mesh: FrozenMesh, ev) -> dict:
    """First pass of the two-pass pressure centering: partial sums of int p_h and |Omega|."""
    if _empty(ev):
        return {"sum:p": 0.0, "sum:vol": 0.0}
    q, xq, W = _qw(mesh, ev)
    return {"sum:p": float((W * ev.pressure(q)).sum()), "sum:vol": float(W.sum())}


def mean_of(parts: list[dict]) -> float:
    """The integral mean from pressure_mean_sums parts of all ranks."""
    return sum(d["sum:p"] for d in parts) / sum(d["sum:vol"] for d in parts)


def difference_sums(mesh: FrozenMesh, ev_a, ev_b, shift: float = 0.0) -> dict:
    """Partial sums of ||u_a - u_b||^2, ||u_a||^2 and of the pressure difference of two discrete solutions on the same
    fixed elements (identity checks, rank invariance).  TWO-PASS centering: pass
    shift = mean(p_a) - mean(p_b) from a first pass (pressure_mean_sums / mean_of); the sums then accumulate
    (p_a - p_b - shift), so a different pressure-constant normalization of the two runs costs no precision (the
    one-pass subtraction lost ~1e-8 per unit of constant offset).  ev_a and ev_b must cover the same ids."""
    if not np.array_equal(np.asarray(ev_a.ids), np.asarray(ev_b.ids)):
        raise RuntimeError("difference_sums needs both evaluators on the same fixed element ids")
    if _empty(ev_a):
        return {k: 0.0 for k in ("sum:du2", "sum:ua2", "sum:dg2", "sum:ga2", "sum:vol", "sum:pa", "sum:pb",
                                 "sum:dp2", "sum:pa2", "sum:dpm")} | {"shift": shift}
    q, xq, W = _qw(mesh, ev_a)
    ua, ga = ev_a.velocity(q)
    ub, gb = ev_b.velocity(q)
    pa, pb = ev_a.pressure(q), ev_b.pressure(q)
    d = pa - pb - shift
    return {"sum:du2": float((W[..., None] * (ua - ub) ** 2).sum()), "sum:ua2": float((W[..., None] * ua ** 2).sum()),
            "sum:dg2": float((W[..., None, None] * (ga - gb) ** 2).sum()),
            "sum:ga2": float((W[..., None, None] * ga ** 2).sum()),
            "sum:vol": float(W.sum()), "sum:pa": float((W * pa).sum()), "sum:pb": float((W * pb).sum()),
            "sum:dp2": float((W * d * d).sum()), "sum:dpm": float((W * d).sum()),
            "sum:pa2": float((W * pa * pa).sum()), "shift": shift}


def finish_difference(parts: list[dict]) -> dict:
    """u_rel, gradu_rel, p_rel (mean-free, relative to the mean-free ||p_a||) and the mean difference.  With the
    two-pass shift the residual mean sum:dpm/vol is round-off, so the subtraction is benign."""
    acc = {}
    for d in parts:
        for k, v in d.items():
            if k != "shift":
                acc[k] = acc.get(k, 0.0) + v
    vol = acc["sum:vol"]
    ma, mb = acc["sum:pa"] / vol, acc["sum:pb"] / vol
    rm = acc.get("sum:dpm", 0.0) / vol
    dp2 = acc["sum:dp2"] - vol * rm ** 2
    pa2 = acc["sum:pa2"] - vol * ma ** 2
    return {"u_rel": math.sqrt(acc["sum:du2"] / acc["sum:ua2"]), "gradu_rel": math.sqrt(acc["sum:dg2"] / acc["sum:ga2"]),
            "p_rel": math.sqrt(max(dp2, 0.0) / pa2), "p_mean_diff": abs(ma - mb),
            "residual_mean_after_shift": abs(rm)}


def partition_report(ids_per_rank: list, n_elements: int, n_ranks: int) -> dict:
    """Rank proof, from the fixed element ids each rank's evaluator covers: number of parts, empty parts,
    duplicated and missing ids.  A run that solved the whole problem on every rank has n_ranks parts each covering
    everything (duplicates); one that solved on one rank has empty parts."""
    parts = [np.asarray(x, np.int64).ravel() for x in ids_per_rank]
    allids = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    # a negative id would make np.bincount raise: count it as out of range instead
    inr = allids[(allids >= 0) & (allids < n_elements)]
    cnt = np.bincount(inr, minlength=n_elements) if len(inr) else np.zeros(n_elements, np.int64)
    return {"parts": len(parts), "parts_expected": int(n_ranks), "empty_parts": int(sum(len(x) == 0 for x in parts)),
            "duplicate_ids": int((cnt > 1).sum()), "missing_ids": int((cnt[:n_elements] == 0).sum()),
            "out_of_range_ids": int((allids >= n_elements).sum() + (allids < 0).sum())}


def energy_sums(ev, mesh: FrozenMesh, form: str, mu: float = 1.0) -> dict:
    """X1 functionals from the solution: partial sums of E_kin = 1/2 ||u_h||^2 and of the dissipation
    a_form(u_h, u_h) (lap: mu |grad u|^2; sym: 2 mu |eps|^2), with the check module's rule (exact for affine P2)."""
    if _empty(ev):
        return {"sum:ekin": 0.0, "sum:diss": 0.0}
    q, xq, W = _qw(mesh, ev)
    u, G = ev.velocity(q)
    if form == "sym":
        E = 0.5 * (G + np.swapaxes(G, -1, -2))
        dis = 2 * mu * (E * E).sum((-1, -2))
    else:
        dis = mu * (G * G).sum((-1, -2))
    return {"sum:ekin": float(0.5 * (W[..., None] * u * u).sum()), "sum:diss": float((W * dis).sum())}


def relative_deviation(values, reference) -> float:
    """max_i |values_i - reference_i| / |reference_i| (duct: the errors of an independent solve of the same discrete
    problem)."""
    v, f = np.asarray(values, float), np.asarray(reference, float)
    if v.shape != f.shape:
        return math.inf
    return float(np.max(np.abs(v - f) / np.abs(f)))


def boundary_node_sums(mesh: FrozenMesh, ev, dirichlet_preds=(), g=None, slip=()) -> dict:
    """Runner-independent nodal checks.  The velocity nodes of P2/Q2 elements are the geometry nodes; the
    solution is evaluated at the reference nodes (REF_P2_TET / REF_Q2_HEX) of the evaluator's own elements.
    - dirichlet_preds: predicates pred(X[n,3]) -> bool of the fixed Dirichlet faces; the max over those nodes of
      |u_h - g| / max(|g|, 1) ('max:dirichlet_nodal_defect');
    - slip: [(pred, normal)] for slip/symmetry faces; the max over their nodes NOT on a Dirichlet face of |u_h . n|
      ('max:slip_un'), each node against every slip face it lies on.
    Returns partial maxima and node counts (per rank, possibly repeated across elements)."""
    if _empty(ev):
        return {"max:dirichlet_nodal_defect": 0.0, "max:slip_un": 0.0, "sum:dirichlet_nodes": 0.0,
                "sum:slip_nodes": 0.0}
    ref = REF_P2_TET if mesh.kind == "tet" else REF_Q2_HEX
    N = mesh.geometry_nodes()[np.asarray(ev.ids)].reshape(-1, 3)
    u, _ = ev.velocity(ref)
    u = u.reshape(-1, 3)
    on_d = np.zeros(len(N), bool)
    for pr in dirichlet_preds:
        on_d |= np.asarray(pr(N), bool)
    out = {"max:dirichlet_nodal_defect": 0.0, "max:slip_un": 0.0, "sum:dirichlet_nodes": float(on_d.sum()),
           "sum:slip_nodes": 0.0}
    if g is not None and on_d.any():
        gv = g(N[on_d])
        out["max:dirichlet_nodal_defect"] = float(np.max(np.abs(u[on_d] - gv)) / max(float(np.max(np.abs(gv))), 1.0))
    for pr, n in slip:
        sel = np.asarray(pr(N), bool) & ~on_d
        out["sum:slip_nodes"] += float(sel.sum())
        if sel.any():
            out["max:slip_un"] = max(out["max:slip_un"], float(np.max(np.abs(u[sel] @ np.asarray(n, float)))))
    return out


# ============================================================================================== sphere drag (S1)
def reaction_test_function(mesh: FrozenMesh, ids, ref, radius=1.0, direction=(0.0, 0.0, 1.0), tol=1e-9):
    """The Babuska-Miller test function w_h: the P2 (tet) / Q2 (hex) Lagrange function equal to `direction` at every
    velocity node on the sphere r = radius and 0 at every other node (the velocity nodes ARE the geometry nodes of
    the isoparametric elements).  Returns (w[m, nq, 3], grad_w[m, nq, 3, 3]) on the fixed map."""
    kind = "p2" if mesh.kind == "tet" else "q2"
    phi, dphi = lagrange(kind, ref)
    N = mesh.geometry_nodes()[np.asarray(ids)]
    on = (np.abs(np.linalg.norm(N, axis=-1) - radius) < tol).astype(float)        # [m, nb]
    _, J = mesh.map(ref, ids)
    Jinv = np.linalg.inv(J)                                                          # [m, nq, 3, 3]
    s = np.einsum("qb,mb->mq", phi, on)
    ds_ref = np.einsum("qbk,mb->mqk", dphi, on)
    ds = np.einsum("mqk,mqkj->mqj", ds_ref, Jinv)
    d = np.asarray(direction, float)
    return s[..., None] * d, d[None, None, :, None] * ds[..., None, :]


def drag_sums(mesh: FrozenMesh, ev, mu: float = 1.0, form_residual: str = "sym") -> dict:
    """Partial sum of the Babuska-Miller drag F_z = -[a_form(u_h, w_h) - (p_h, div w_h)] (f = 0, w_h of
    reaction_test_function: no traction boundary touches the sphere).  form_residual 'sym' (Cauchy traction: the force) or 'lap' (the pseudo-traction residual: a negative control on curved walls)."""
    if _empty(ev):
        return {"sum:drag": 0.0}
    q, xq, W = _qw(mesh, ev)
    uh, G = ev.velocity(q)
    ph = ev.pressure(q)
    w, gw = reaction_test_function(mesh, ev.ids, q)
    if form_residual == "sym":
        a = mu * 0.5 * ((G + np.swapaxes(G, -1, -2)) * (gw + np.swapaxes(gw, -1, -2))).sum((-1, -2))
    else:
        a = mu * (G * gw).sum((-1, -2))
    divw = np.trace(gw, axis1=-2, axis2=-1)
    return {"sum:drag": float(-(W * (a - ph * divw)).sum())}


# ============================================================================================== boundary checks
def dirichlet_nodal_defect(points, values, g) -> float:
    """Dirichlet data must be the Lagrange NODAL interpolant of the fixed callable g:
    points[n, 3] the Dirichlet velocity nodes (vertices, edge nodes, and for Q2 face and cell nodes on the boundary),
    values[n, 3] the solution there.  Returns max |u_h - g| / max(max |g|, 1)."""
    P = np.asarray(points, float)
    gv = g(P)
    return float(np.max(np.abs(np.asarray(values, float) - gv)) / max(float(np.max(np.abs(gv))), 1.0))


def dirichlet_shared_node_mismatch(points, callables) -> float:
    """Shared-node rule: at a velocity node on several Dirichlet attributes the per-attribute data agree."""
    worst, scale = 0.0, 1.0
    for P, gs in zip(np.asarray(points, float), callables):
        vals = [np.asarray(g(P[None, :]), float).reshape(-1) for g in gs]
        scale = max(scale, max(float(np.max(np.abs(v))) for v in vals))
        for a in vals:
            for b in vals:
                worst = max(worst, float(np.max(np.abs(a - b))))
    return worst / scale


def slip_nodal_un(values, normals, scale, on_dirichlet=None) -> float:
    """max |u_h . n| / scale over slip/symmetry nodes; a node on two slip faces (the slip-symmetry edge of T4) is
    checked against BOTH normals; Dirichlet-owned nodes are skipped (Dirichlet wins)."""
    worst = 0.0
    for i, (u, ns) in enumerate(zip(values, normals)):
        if on_dirichlet is not None and on_dirichlet[i]:
            continue
        for n in ns:
            worst = max(worst, abs(float(np.dot(u, n))))
    return worst / scale


def p2_face_flux_defect(mesh: FrozenMesh, g, faces=None) -> tuple[float, float]:
    """Discrete flux of the P2 (tet faces) / Q2 (hex faces) NODAL interpolant of Dirichlet data g over the flat
    boundary faces of an affine mesh: the exact integral of the interpolant (triangle: the three edge midpoints with
    weight |T|/3 each, vertices 0; quadrilateral: tensor Simpson).  Returns (sum of the interpolant's g.n, the
    reference int |g.n| by the same rule).  The outward normal is oriented away from the mesh centroid: valid for
    star-shaped domains (cube, duct), NOT for the inner sphere of the shell (never used there)."""
    faces = mesh.boundary_faces() if faces is None else faces
    cen = mesh.V.mean(0)
    tot, ref = 0.0, 0.0
    for f in faces:
        P = mesh.V[list(f)]
        if len(f) == 3:
            nrm = np.cross(P[1] - P[0], P[2] - P[0])
            area = 0.5 * np.linalg.norm(nrm)
            n = nrm / np.linalg.norm(nrm)
            mids = np.array([(P[0] + P[1]) / 2, (P[1] + P[2]) / 2, (P[2] + P[0]) / 2])
            gn = g(mids) @ n
            wts = np.full(3, area / 3)
        else:
            # order the quad by angle in its plane
            c = P.mean(0)
            nrm = np.cross(P[1] - P[0], P[2] - P[0])
            n = nrm / np.linalg.norm(nrm)
            e1 = (P[0] - c) / np.linalg.norm(P[0] - c)
            e2 = np.cross(n, e1)
            ang = np.arctan2((P - c) @ e2, (P - c) @ e1)
            Q = P[np.argsort(ang)]
            s = np.array([0, .5, 1])
            pts, wts = [], []
            for i, a in enumerate(s):
                for j, b in enumerate(s):
                    pts.append(Q[0] + a * (Q[1] - Q[0]) + b * (Q[3] - Q[0]))
                    wts.append(np.array([1, 4, 1])[i] * np.array([1, 4, 1])[j] / 36)
            area = np.linalg.norm(Q[1] - Q[0]) * np.linalg.norm(Q[3] - Q[0])
            gn = g(np.array(pts)) @ n
            wts = np.array(wts) * area
        if n @ (P.mean(0) - cen) < 0:
            gn = -gn
        tot += float(wts @ gn)
        ref += float(wts @ np.abs(gn))
    return tot, ref


# ============================================================================================== inf-sup (N1)
def inf_sup(A, B, Mp, zero_rel: float = 1e-10) -> dict:
    """beta^2 = eigenvalues of B A^-1 B^T q = lambda Mp q on the free velocity DOFs (A the vector H1-seminorm Gram
    matrix with the Dirichlet rows removed); dense, numpy only (small meshes).  Eigenvalues below zero_rel * lambda_max
    are zero modes; the constant is one of them."""
    A = A.toarray() if hasattr(A, "toarray") else np.asarray(A, float)
    B = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
    Mp = Mp.toarray() if hasattr(Mp, "toarray") else np.asarray(Mp, float)
    S = B @ np.linalg.solve(A, B.T)
    S = 0.5 * (S + S.T)
    Lc = np.linalg.cholesky(Mp)
    Li = np.linalg.inv(Lc)
    lam = np.linalg.eigvalsh(Li @ S @ Li.T)
    lmax = float(lam.max())
    nz = lam[lam > zero_rel * lmax]
    n_zero = int(np.sum(lam <= zero_rel * lmax))
    return {"beta": float(math.sqrt(nz.min())) if len(nz) else 0.0,
            "zero_modes_beyond_constant": max(n_zero - 1, 0), "lambda_max": lmax}
