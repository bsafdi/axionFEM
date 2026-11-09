"""3D tetrahedral mesh families of the current-distribution verification cases (the constructions described in the
case definitions).

* Kuhn box: the unit cube's N^3 cells, each split into the 6 Kuhn (Freudenthal) tetrahedra around the main diagonal
  (i,j,k) -> (i+1,j+1,k+1), one tetrahedron per permutation of the axes; vertices (i,j,k)/N in ij-order; cells
  cell-major / permutation-minor.  Nested under doubling.
* ``box(N, L)``: [0, L]^3; boundary attributes 1 x=0, 2 x=L, 3 y=0, 4 y=L, 5 z=0, 6 z=L.
* ``sector(N, "cyl3d" | "sph3d", order)``: the Kuhn box of [0, 1]^3 mapped to the cylinder sector (rho = r1 + s (r2 -
  r1), theta = t pi/2, z = u H) or the sphere sector (theta = pi/4 + t pi/4 polar, azimuth u pi/2); attributes 1 inner
  electrode, 2 outer electrode, 3 insulating walls.  P2 is isoparametric: edge nodes at gmap(parametric midpoint).

A boundary facet belongs to a face when ALL its geometry nodes lie on it (decided in the parametric cube, where the
mapped edge nodes of a facet lie on the face exactly when its vertices do).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

R1, R2, H_CYL = 0.005, 0.02, 0.01

_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def kuhn(N: int):
    """Unit-cube Kuhn family: (P (nv, 3) in [0, 1]^3, T (ne, 4) int64, I (nv, 3) integer lattice indices)."""
    xs = np.linspace(0.0, 1.0, N + 1)
    P = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3)
    I = np.stack(np.meshgrid(*(np.arange(N + 1),) * 3, indexing="ij"), -1).reshape(-1, 3)
    ii = np.stack(np.meshgrid(*(np.arange(N),) * 3, indexing="ij"), -1).reshape(-1, 3)

    def idx(v):
        return (v[:, 0] * (N + 1) + v[:, 1]) * (N + 1) + v[:, 2]
    per = []
    for p in itertools.permutations(range(3)):
        v = ii.copy()
        tet = [idx(v)]
        for ax in p:
            v = v.copy()
            v[:, ax] += 1
            tet.append(idx(v))
        per.append(np.stack(tet, 1))
    T = np.stack(per, 1).reshape(-1, 4)
    return P, T, I


def gmap(family: str):
    """The maps of the families (the same floating-point expressions as the reference generators)."""
    if family == "box":
        return lambda P: 0.01 * P
    if family == "sph3d":
        th0, th1, ph0, ph1 = 0.25 * math.pi, 0.5 * math.pi, 0.0, 0.5 * math.pi

        def g(P):
            rho = R1 + P[:, 0] * (R2 - R1)
            th = th0 + P[:, 1] * (th1 - th0)
            ph = ph0 + P[:, 2] * (ph1 - ph0)
            return np.stack([rho * np.sin(th) * np.cos(ph), rho * np.sin(th) * np.sin(ph), rho * np.cos(th)], 1)
        return g
    if family == "cyl3d":
        def g(P):
            rho = R1 + P[:, 0] * (R2 - R1)
            th = P[:, 1] * 0.5 * math.pi
            return np.stack([rho * np.cos(th), rho * np.sin(th), P[:, 2] * H_CYL], 1)
        return g
    raise ValueError(f"unknown 3D family {family!r}: use box, cyl3d or sph3d")


@dataclass
class TetMesh:
    vertices: np.ndarray
    T: np.ndarray
    eattr: np.ndarray
    B: np.ndarray
    battr: np.ndarray
    edge_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    edge_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    h_nominal: float = 0.0
    family: str = ""
    N: int = 0


def _boundary(T, I, N, kind: str):
    """Boundary facets (nb, 3) and attributes from the integer lattice indices of the parametric cube."""
    F = T[:, _FACES].reshape(-1, 3)                 # every tetrahedron face
    C = I[F]                                        # (nf, 3 vertices, 3 axes)
    attr = np.zeros(len(F), np.int32)
    for ax in range(3):
        on0 = np.all(C[:, :, ax] == 0, 1)
        on1 = np.all(C[:, :, ax] == N, 1)
        if kind == "box":
            attr[on0] = 2 * ax + 1
            attr[on1] = 2 * ax + 2
        else:
            attr[on0] = 1 if ax == 0 else 3
            attr[on1] = 2 if ax == 0 else 3
    keep = attr > 0
    return F[keep].astype(np.int32), attr[keep]


def box(N: int, L: float = 0.01) -> TetMesh:
    P, T, I = kuhn(N)
    B, ba = _boundary(T, I, N, "box")
    V = (L * P) if L != 0.01 else gmap("box")(P)
    return TetMesh(V, T.astype(np.int32), np.ones(len(T), np.int32), B, ba, h_nominal=L / N, family="box", N=N)


def sector(N: int, family: str, order: int = 1) -> TetMesh:
    P, T, I = kuhn(N)
    gm = gmap(family)
    B, ba = _boundary(T, I, N, "sector")
    m = TetMesh(gm(P), T.astype(np.int32), np.ones(len(T), np.int32), B, ba, h_nominal=(R2 - R1) / N, family=family,
                N=N)
    if order == 2:
        E = np.sort(T[:, _EDGES], axis=-1).reshape(-1, 2)
        uE = np.unique(E, axis=0)
        m.edge_v = uE.astype(np.int32)
        m.edge_xyz = gm(0.5 * (P[uE[:, 0]] + P[uE[:, 1]]))
    return m
