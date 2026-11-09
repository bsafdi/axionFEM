"""Deterministic triangle-mesh families of the current-distribution verification cases (the constructions described in
benchmarks/chem/registry/cases/chem_*_c3_*.json, ``run.parameters.meshes.*.construction``).

The construction texts name these functions (``meshes.rect``, ``meshes.annulus_sector``, ``meshes.disk_graded``),
which build the vertex sets and triangulations so that the vertex and connectivity hashes of every level match (they are checked in each case).  A mesh is kept in PARAMETRIC
coordinates ``X`` with an optional map ``geomap`` to physical space; P2 edge nodes of mapped families come from
the map (``edge_nodes``).

Boundary attributes (defaults): rect 1 x = x0, 2 x = x1, 3 y = y0, 4 y = y1; annulus_sector 1 inner arc, 2 outer arc,
3 th0 side, 4 th1 side; disk_graded 1 disk, 2 insulating plane, 3 axis, 4 outer arc.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class TriMesh:
    X: np.ndarray                 # parametric vertex coordinates (nv, 2)
    T: np.ndarray                 # triangles (ne, 3)
    eattr: np.ndarray
    B: np.ndarray                 # boundary segments (nb, 2)
    battr: np.ndarray
    h_nominal: float
    geomap: Optional[Callable] = None
    name: str = ""
    # P2 edge-node rule: None (straight), "map" (every edge node = map of the parametric midpoint),
    # "outer_arc" (disk: only the outer-arc edge node, the arc point at the mean polar angle)
    curved: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def vertices(self) -> np.ndarray:
        return self.X.copy() if self.geomap is None else self.geomap(self.X)

    def edge_nodes(self):
        """(edge vertex pairs (m, 2), physical node positions (m, 2)) of the P2 edge nodes that are not the straight
        midpoint of their chord; empty arrays for straight meshes."""
        if self.curved is None or self.geomap is None:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2))
        E = set()
        for t in self.T:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                E.add((min(a, b), max(a, b)))
        E = np.array(sorted(E), dtype=np.int32)
        A, Bp = self.X[E[:, 0]], self.X[E[:, 1]]
        if self.curved == "map":
            return E, self.geomap(0.5 * (A + Bp))
        if self.curved == "outer_arc":
            Rb = self.meta["Rb"]
            sel = (np.abs(A[:, 0] - Rb) < 1e-12 * Rb) & (np.abs(Bp[:, 0] - Rb) < 1e-12 * Rb)
            M = 0.5 * (A[sel] + Bp[sel])
            return E[sel], self.geomap(M)
        raise ValueError(f"unknown edge-node rule {self.curved!r}")


def _bfaces(T):
    cnt = Counter()
    for t in T:
        for a in range(3):
            cnt[tuple(sorted(np.delete(t, a)))] += 1
    return np.array([f for f, c in cnt.items() if c == 1], dtype=int)


def _orient(m: TriMesh):
    Xp = m.vertices()
    E = Xp[m.T[:, 1:]] - Xp[m.T[:, [0]]]
    det = E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
    bad = det < 0
    if np.any(bad):
        T = m.T.copy()
        T[bad, 0], T[bad, 1] = m.T[bad, 1], m.T[bad, 0]
        m.T = T


def rect(N, Ny=None, x0=0.0, x1=1.0, y0=0.0, y1=1.0, eattr_fn=None, geomap=None):
    """N x Ny squares, each split along the diagonal (i, j)-(i+1, j+1): triangles [a, b, c], [a, c, d]."""
    Ny = Ny or N
    xs, ys = np.linspace(x0, x1, N + 1), np.linspace(y0, y1, Ny + 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel()], 1)
    idx = lambda i, j: i * (Ny + 1) + j
    T = []
    for i in range(N):
        for j in range(Ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    T = np.array(T)
    B = _bfaces(T)
    mid = X[B].mean(1)
    tol = 1e-12 * max(x1 - x0, y1 - y0)
    battr = np.where(np.abs(mid[:, 0] - x0) < tol, 1, np.where(np.abs(mid[:, 0] - x1) < tol, 2,
                     np.where(np.abs(mid[:, 1] - y0) < tol, 3, 4)))
    cen = X[T].mean(1)
    eattr = np.ones(len(T), int) if eattr_fn is None else eattr_fn(cen)
    m = TriMesh(X, T, eattr, B, battr, h_nominal=(x1 - x0) / N, geomap=geomap, name=f"rect{N}x{Ny}")
    _orient(m)
    return m


def red_refine(m: TriMesh) -> TriMesh:
    """Uniform red refinement in parametric coordinates."""
    X, T = m.X, m.T
    edges, newX = {}, [*X]

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in edges:
            edges[k] = len(newX)
            newX.append(0.5 * (X[a] + X[b]))
        return edges[k]
    Tn, En = [], []
    for t, at in zip(T, m.eattr):
        a, b, c = t
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        Tn += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        En += [at] * 4
    Bn, BAn = [], []
    for f, at in zip(m.B, m.battr):
        c = mid(f[0], f[1])
        Bn += [[f[0], c], [c, f[1]]]
        BAn += [at, at]
    out = TriMesh(np.array(newX), np.array(Tn), np.array(En), np.array(Bn), np.array(BAn),
                  h_nominal=m.h_nominal / 2, geomap=m.geomap, name=m.name + "+r", curved=m.curved, meta=m.meta)
    _orient(out)
    return out


def distorted_rect(k, perturb=0.2, seed=20261031, N0=4, x1=1.0, y1=1.0, eattr_fn=None):
    """N0 x N0 base, interior vertices displaced by perturb (L/N0) U(-1,1)^2 (PCG64(seed)); vertices on x = L/2
    only in y; then k red refinements."""
    base = rect(N0, x1=x1, y1=y1, eattr_fn=eattr_fn)
    rng = np.random.Generator(np.random.PCG64(seed))
    X = base.X.copy()
    x0_, x1_ = X[:, 0].min(), X[:, 0].max()
    y0_, y1_ = X[:, 1].min(), X[:, 1].max()
    L = x1_ - x0_
    interior = (X[:, 0] > x0_ + 1e-12) & (X[:, 0] < x1_ - 1e-12) & (X[:, 1] > y0_ + 1e-12) & (X[:, 1] < y1_ - 1e-12)
    dX = perturb * (L / N0) * rng.uniform(-1, 1, (int(interior.sum()), 2))
    on_mid = np.abs(X[interior, 0] - 0.5 * (x0_ + x1_)) < 1e-12
    dX[on_mid, 0] = 0.0
    X[interior] += dX
    base.X = X
    m = base
    for _ in range(k):
        m = red_refine(m)
    m.h_nominal = L / N0 / 2 ** k
    m.name = f"dist{N0}+{k}"
    return m


def annulus_sector(Nr, Nt, r1, r2, th0, th1, axisym_meridian=False):
    """Polar map of rect(Nr, Nt) on the unit square: rho = r1 + s (r2 - r1), th = th0 + t (th1 - th0); planar
    x = rho cos th, y = rho sin th; meridian r = rho sin th, z = rho cos th.  P2 edge nodes by the exact map."""
    def gm(P):
        rho = r1 + P[:, 0] * (r2 - r1)
        th = th0 + P[:, 1] * (th1 - th0)
        if axisym_meridian:
            return np.stack([rho * np.sin(th), rho * np.cos(th)], 1)
        return np.stack([rho * np.cos(th), rho * np.sin(th)], 1)
    m = rect(Nr, Nt, geomap=gm)
    m.h_nominal = (r2 - r1) / Nr
    m.curved = "map"
    m.name = f"annulus{Nr}x{Nt}"
    _orient(m)
    return m


def disk_graded(n, r0=1.0, Rb=5.0, beta=1.0, Rg=None):
    """Quarter disk rho < Rb in the meridian for a disk electrode of radius r0 on z = 0: polar grid (2n radial
    intervals on [0, r0], 4n on [r0, Rb], 4n angular on [0, pi/2] from +z), fan at the origin, quads split along
    (i, j)-(i+1, j+1); grading x -> e + (x - e)(|x - e|/Rg)^(beta - 1) for |x - e| < Rg = r0/2 about e = (r0, 0).
    A z = 0 facet is disk (1) iff both vertices have r <= r0, else plane (2); 3 axis, 4 outer arc.  P2: only the
    outer-arc edge node is curved."""
    Rg = Rg if Rg is not None else 0.5 * r0
    n_in, n_out, n_th = 2 * n, 4 * n, 4 * n
    rho = np.concatenate([np.linspace(0, r0, n_in + 1), np.linspace(r0, Rb, n_out + 1)[1:]])
    th = np.linspace(0, 0.5 * np.pi, n_th + 1)
    X = [np.array([0.0, 0.0])]
    ids = {}
    for i in range(1, len(rho)):
        for j in range(n_th + 1):
            ids[(i, j)] = len(X)
            X.append(np.array([rho[i], th[j]]))
    X = np.array(X)
    T = [[0, ids[(1, j)], ids[(1, j + 1)]] for j in range(n_th)]
    for i in range(1, len(rho) - 1):
        for j in range(n_th):
            a, b, c, d = ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]
            T += [[a, b, c], [a, c, d]]
    T = np.array(T)
    e = np.array([r0, 0.0])

    def gm(P):
        x = np.stack([P[:, 0] * np.sin(P[:, 1]), P[:, 0] * np.cos(P[:, 1])], 1)
        if beta == 1.0:
            return x
        dvec = x - e
        dist = np.linalg.norm(dvec, axis=1)
        s = np.ones_like(dist)
        inside = (dist < Rg) & (dist > 0)
        s[inside] = (dist[inside] / Rg) ** (beta - 1.0)
        return e + dvec * s[:, None]
    B = _bfaces(T)
    battr = []
    for f in B:
        p = X[f]
        if np.all(np.abs(p[:, 1] - 0.5 * np.pi) < 1e-12) or (np.any(p[:, 0] == 0) and np.all(
                (np.abs(p[:, 1] - 0.5 * np.pi) < 1e-12) | (p[:, 0] == 0))):
            battr.append(1 if p[:, 0].max() <= r0 + 1e-12 else 2)
        elif np.all(np.abs(p[:, 0] - Rb) < 1e-12):
            battr.append(4)
        else:
            battr.append(3)
    m = TriMesh(X, T, np.ones(len(T), int), B, np.array(battr), h_nominal=r0 / n_in, geomap=gm,
                name=f"disk_n{n}_b{beta}", curved="outer_arc", meta=dict(r0=r0, Rb=Rb, beta=beta, Rg=Rg, n=n))
    _orient(m)
    return m
