"""Export axionFEM geometries and meshes to COMSOL Multiphysics.

Three formats (docs/comsol_export.md has the details, the import recipes and the limitations):

``mphtxt``
    COMSOL's native sectionwise mesh text format (``Mesh`` class, version 4, file format 0.1;
    COMSOL Multiphysics Programming Reference Manual, "The COMSOL File Formats" -> "Mesh",
    "Selection" and "Mesh Element Numbering Conventions").  It is read by the base package
    (Mesh > Import, no add-on module).  The writer derives a complete geometric-entity
    partition from the finite-element mesh (domains, boundaries, edges, points) so that every
    axionFEM region, every tagged boundary group (outer sphere, symmetry planes, PEC holes, axis)
    and every region's boundary arrive in COMSOL as named selections.

``java``
    A COMSOL Java API model file (``Model.java``) that rebuilds the geometry from primitives
    (Block, Cylinder, Sphere, Cone, Torus, work-plane polygons, Revolve, Extrude, Booleans) with
    the design parameters as COMSOL parameters and one named selection per region (the
    "later regions win" painting order of axionFEM is reproduced with Difference selections).
    Compile with ``comsolcompile Model.java`` and run with
    ``comsol batch -inputfile Model.class``; it saves ``<name>.mph``.

``step``
    The conformal OpenCASCADE solids that axf-mesher builds (3D designs), written by gmsh as
    STEP, with a JSON map from solid to region.  Importing STEP in COMSOL needs the CAD Import
    Module (or LiveLink / Design Module).

Entry points: :func:`export_comsol` (also ``axf export-comsol`` from the Python console script),
:func:`gmsh_to_mphtxt`, :func:`read_gmsh`, :func:`read_mphtxt` and :func:`write_java`.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "GmshMesh",
    "export_comsol",
    "gmsh_to_mphtxt",
    "main",
    "read_gmsh",
    "read_mphtxt",
    "write_import_java",
    "write_java",
    "write_mphtxt",
    "comsol_expression",
]

# =====================================================================================
# gmsh .msh 2.2 reader (ASCII and binary; what axf-mesher writes)
# =====================================================================================

# gmsh element type -> (dimension, number of nodes, number of vertices)
_GMSH_TYPES = {
    15: (0, 1, 1),   # point
    1: (1, 2, 2),    # 2-node line
    8: (1, 3, 2),    # 3-node line
    2: (2, 3, 3),    # 3-node triangle
    9: (2, 6, 3),    # 6-node triangle
    4: (3, 4, 4),    # 4-node tetrahedron
    11: (3, 10, 4),  # 10-node tetrahedron
}
# local vertex pairs of the mid-edge nodes, in gmsh node order (gmsh reference manual, "Node ordering")
_GMSH_MID = {
    8: [(0, 1)],
    9: [(0, 1), (1, 2), (2, 0)],
    11: [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)],
}


@dataclass
class GmshMesh:
    """A gmsh 2.2 mesh: node coordinates and element blocks with their physical tags."""

    nodes: np.ndarray                      # (N, 3) coordinates, row = node index
    blocks: list = field(default_factory=list)  # [(gmsh_type, phys (n,), geom (n,), conn (n, k) node rows)]
    physical_names: dict = field(default_factory=dict)  # (dim, tag) -> name
    path: str = ""

    def elements(self, dim: int):
        """(gmsh_type, phys, conn) of every block of the given dimension."""
        return [(t, p, c) for t, p, g, c in self.blocks if _GMSH_TYPES[t][0] == dim]

    @property
    def dim(self) -> int:
        return max((_GMSH_TYPES[t][0] for t, *_ in self.blocks), default=0)


def _read_section_names(lines: list[str]) -> dict:
    names = {}
    n = int(lines[0])
    for ln in lines[1:1 + n]:
        m = re.match(r"\s*(\d+)\s+(-?\d+)\s+\"(.*)\"", ln)
        if m:
            names[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    return names


def read_gmsh(path: str | Path) -> GmshMesh:
    """Read a gmsh MSH 2.2 file (ASCII or binary), as written by axf-mesher.

    Element connectivity is returned as row indices into ``nodes`` (gmsh node ids may be
    non-contiguous).  Only the first two tags of each element (physical, elementary) are kept.
    """
    data = Path(path).read_bytes()
    pos = 0

    def line() -> bytes:
        nonlocal pos
        j = data.index(b"\n", pos)
        s = data[pos:j]
        pos = j + 1
        return s.strip()

    binary = False
    swap = "<"
    node_ids = coords = None
    blocks_raw: list = []
    names: dict = {}
    while pos < len(data):
        tag = line()
        if not tag:
            continue
        if tag == b"$MeshFormat":
            ver, ft, ds = line().split()
            if not ver.startswith(b"2"):
                raise ValueError(f"{path}: MSH format {ver.decode()} is not supported (axf-mesher writes 2.2)")
            binary = ft == b"1"
            if binary:
                one = data[pos:pos + 4]
                swap = "<" if struct.unpack("<i", one)[0] == 1 else ">"
                pos += 4
                line()  # rest of the line
            while line() != b"$EndMeshFormat":
                pass
        elif tag == b"$PhysicalNames":
            sec = []
            while True:
                s = line()
                if s == b"$EndPhysicalNames":
                    break
                sec.append(s.decode())
            names = _read_section_names(sec)
        elif tag == b"$Nodes":
            n = int(line())
            if binary:
                rec = np.dtype([("id", swap + "i4"), ("x", swap + "f8", (3,))])
                arr = np.frombuffer(data, dtype=rec, count=n, offset=pos)
                pos += n * rec.itemsize
                node_ids = arr["id"].astype(np.int64)
                coords = np.array(arr["x"], dtype=float)
                line()  # newline after the binary block
            else:
                end = data.index(b"$EndNodes", pos)
                tab = np.array(data[pos:end].split(), dtype=float).reshape(n, 4)
                pos = end
                node_ids = tab[:, 0].astype(np.int64)
                coords = tab[:, 1:4].copy()
            while line() != b"$EndNodes":
                pass
        elif tag == b"$Elements":
            ne = int(line())
            if binary:
                done = 0
                while done < ne:
                    etype, nfollow, ntags = struct.unpack(swap + "3i", data[pos:pos + 12])
                    pos += 12
                    if etype not in _GMSH_TYPES:
                        raise ValueError(f"{path}: gmsh element type {etype} is not supported")
                    nn = _GMSH_TYPES[etype][1]
                    width = 1 + ntags + nn
                    a = np.frombuffer(data, dtype=swap + "i4", count=nfollow * width, offset=pos).reshape(nfollow, width)
                    pos += 4 * nfollow * width
                    phys = a[:, 1] if ntags >= 1 else np.zeros(nfollow, int)
                    geom = a[:, 2] if ntags >= 2 else np.zeros(nfollow, int)
                    blocks_raw.append((etype, phys.astype(np.int64), geom.astype(np.int64), a[:, 1 + ntags:].astype(np.int64)))
                    done += nfollow
                line()
            else:
                rows: dict = {}
                for _ in range(ne):
                    v = line().split()
                    etype, ntags = int(v[1]), int(v[2])
                    if etype not in _GMSH_TYPES:
                        raise ValueError(f"{path}: gmsh element type {etype} is not supported")
                    rows.setdefault(etype, []).append([int(x) for x in v[3:3 + ntags][:2]] + [0] * max(0, 2 - ntags)
                                                      + [int(x) for x in v[3 + ntags:]])
                for etype, r in rows.items():
                    a = np.array(r, dtype=np.int64)
                    blocks_raw.append((etype, a[:, 0], a[:, 1], a[:, 2:]))
            while line() != b"$EndElements":
                pass
        else:  # unknown section: skip to its end
            end = b"$End" + tag[1:]
            while line() != end:
                pass
    if coords is None:
        raise ValueError(f"{path}: no $Nodes section")
    lut = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
    lut[node_ids] = np.arange(len(node_ids))
    blocks = [(t, p, g, lut[c]) for t, p, g, c in blocks_raw]
    for t, p, g, c in blocks:
        if (c < 0).any():
            raise ValueError(f"{path}: elements reference undefined nodes")
    return GmshMesh(nodes=coords, blocks=blocks, physical_names=names, path=str(path))


# =====================================================================================
# geometric-entity partition (domains / boundaries / edges / points) from a simplex mesh
# =====================================================================================

def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component labels (0..k-1, in order of first appearance) of a graph with edges i-j."""
    lab = np.arange(n, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    while len(i):
        li, lj = lab[i], lab[j]
        if (li == lj).all():
            break
        m = np.minimum(li, lj)
        np.minimum.at(lab, li, m)
        np.minimum.at(lab, lj, m)
        while True:  # pointer jumping
            nl = lab[lab]
            if (nl == lab).all():
                break
            lab = nl
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))  # relabel by first appearance (deterministic, mesh order)
    return order[inv.ravel()]


def _row_keys(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of a sorted-per-row integer array: (unique rows, inverse)."""
    a = np.ascontiguousarray(a, dtype=np.int64)
    if a.shape[1] <= 3 and (a.size == 0 or a.max() < (1 << 20)):
        k = np.zeros(len(a), dtype=np.int64)
        for c in range(a.shape[1]):
            k = (k << 21) | a[:, c]
        u, idx, inv = np.unique(k, return_index=True, return_inverse=True)
        return a[idx], inv.ravel()
    u, inv = np.unique(a, axis=0, return_inverse=True)
    return u, inv.ravel()


def _chains(segs: np.ndarray, key: np.ndarray, extra_vertices: Iterable[int] = ()) -> tuple[np.ndarray, list[int]]:
    """Split segments (m, 2) into chains between vertices; segments of one chain share `key`.

    A node is a vertex where the segment degree is not 2, where its two segments have different
    keys, or when listed in `extra_vertices`.  Closed loops without a vertex get two (COMSOL edges
    run between two distinct vertices).  Returns (chain id per segment, sorted vertex nodes).
    """
    m = len(segs)
    ent = np.full(m, -1, dtype=np.int64)
    if m == 0:
        return ent, sorted(set(int(v) for v in extra_vertices))
    adj: dict[int, list[int]] = {}
    for s, (a, b) in enumerate(segs.tolist()):
        adj.setdefault(a, []).append(s)
        adj.setdefault(b, []).append(s)
    vert = set(int(v) for v in extra_vertices)
    for nd, ss in adj.items():
        if len(ss) != 2 or key[ss[0]] != key[ss[1]]:
            vert.add(nd)
    nchain = 0

    def walk(start_node: int, s: int, cid: int) -> None:
        node = start_node
        while True:
            ent[s] = cid
            a, b = segs[s]
            node = int(b) if int(a) == node else int(a)
            if node in vert:
                return
            nxt = [t for t in adj[node] if ent[t] < 0]
            if not nxt:
                return
            s = nxt[0]

    for v in sorted(vert):
        for s in adj.get(v, []):
            if ent[s] < 0:
                walk(v, s, nchain)
                nchain += 1
    # closed loops: order the loop, put vertices at its start and half-way, walk again
    for s0 in range(m):
        if ent[s0] >= 0:
            continue
        loop_nodes, loop_segs = [int(segs[s0][0])], []
        s, node = s0, int(segs[s0][0])
        while True:
            loop_segs.append(s)
            ent[s] = -2
            a, b = segs[s]
            node = int(b) if int(a) == node else int(a)
            if node == loop_nodes[0]:
                break
            loop_nodes.append(node)
            nxt = [t for t in adj[node] if ent[t] == -1]
            if not nxt:
                break
            s = nxt[0]
        for t in loop_segs:
            ent[t] = -1
        v0, v1 = loop_nodes[0], loop_nodes[len(loop_nodes) // 2]
        vert.update((v0, v1))
        for v in (v0, v1):
            for t in adj[v]:
                if ent[t] < 0:
                    walk(v, t, nchain)
                    nchain += 1
    return ent, sorted(vert)


@dataclass
class _Partition:
    dim: int
    points: np.ndarray            # (N, dim)
    cells: np.ndarray             # (n, dim+1) vertex rows, positively oriented
    cell_dom: np.ndarray          # (n,) 1-based domain index
    dom_attr: np.ndarray          # (ndom,) physical attribute of each domain
    bfacets: np.ndarray           # (nb, dim) vertex rows of boundary facets
    bfacet_ent: np.ndarray        # (nb,) 0-based boundary index
    bnd_info: list                # per boundary: {"domains": (d0, d1) (0 = outside), "attr": tag or 0}
    edges: np.ndarray             # 3D: (ne, 2) feature-edge segments; 2D: unused (empty)
    edge_ent: np.ndarray
    vertices: np.ndarray          # vertex nodes (0-based point entity = position)
    mid: dict                     # (min, max) vertex pair -> mid-edge node (second-order meshes)


def _partition(mesh: GmshMesh, feature_angle: float | None = None) -> _Partition:
    dim = mesh.dim
    if dim not in (2, 3):
        raise ValueError(f"{mesh.path}: expected a 2D triangle or 3D tetrahedron mesh, found dimension {dim}")
    cblocks = mesh.elements(dim)
    cells = np.concatenate([c[:, :dim + 1] for t, p, c in cblocks])
    attr = np.concatenate([p for t, p, c in cblocks])
    pts = mesh.nodes[:, :dim]
    # mid-edge nodes of second-order cells (checked against the edge midpoints)
    mid: dict = {}
    for t, p, c in cblocks:
        if t in _GMSH_MID:
            pairs = _GMSH_MID[t]
            for k, (a, b) in enumerate(pairs):
                va, vb, vm = c[:, a], c[:, b], c[:, dim + 1 + k]
                h = np.linalg.norm(pts[va] - pts[vb], axis=1)
                off = np.linalg.norm(pts[vm] - 0.5 * (pts[va] + pts[vb]), axis=1)
                if (off > 0.35 * h).any():
                    raise ValueError(f"{mesh.path}: unexpected node order in gmsh element type {t}")
                lo, hi = np.minimum(va, vb), np.maximum(va, vb)
                mid.update(zip(zip(lo.tolist(), hi.tolist()), vm.tolist()))
    # positive orientation (COMSOL examples are counter-clockwise / positive volume)
    if dim == 2:
        d = pts[cells[:, 1]] - pts[cells[:, 0]]
        e = pts[cells[:, 2]] - pts[cells[:, 0]]
        det = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    else:
        d1 = pts[cells[:, 1]] - pts[cells[:, 0]]
        d2 = pts[cells[:, 2]] - pts[cells[:, 0]]
        d3 = pts[cells[:, 3]] - pts[cells[:, 0]]
        det = np.einsum("ij,ij->i", np.cross(d1, d2), d3)
    neg = det < 0
    cells = cells.copy()
    cells[neg, 1], cells[neg, 2] = cells[neg, 2].copy(), cells[neg, 1].copy()

    # facets of every cell and their neighbours
    loc = [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    nf = len(loc)
    allf = np.concatenate([cells[:, list(l)] for l in loc])
    owner = np.tile(np.arange(len(cells)), nf)
    ufac, inv = _row_keys(np.sort(allf, axis=1))
    cnt = np.bincount(inv, minlength=len(ufac))
    if cnt.max() > 2:
        raise ValueError(f"{mesh.path}: non-manifold mesh (a facet is shared by more than two elements)")
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    c0 = owner[order[starts]]
    c1 = np.where(cnt == 2, owner[order[np.minimum(starts + 1, len(order) - 1)]], -1)

    # domains: connected cells of one attribute (through shared facets)
    inner = cnt == 2
    same = inner & (attr[c0] == attr[np.maximum(c1, 0)])
    comp = _components(len(cells), c0[same], c1[same])
    cell_dom = comp + 1
    ndom = int(cell_dom.max())
    dom_attr = np.zeros(ndom, dtype=np.int64)
    dom_attr[cell_dom - 1] = attr

    # tagged facets from the file (outer sphere, symmetry planes, PEC holes, axis, sheets ...)
    ftag = np.zeros(len(ufac), dtype=np.int64)
    fb = mesh.elements(dim - 1)
    if fb:
        tf = np.concatenate([c[:, :dim] for t, p, c in fb])
        tp = np.concatenate([p for t, p, c in fb])
        allk, allinv = _row_keys(np.concatenate([ufac, np.sort(tf, axis=1)]))
        fidx = np.full(len(allk), -1, dtype=np.int64)
        fidx[allinv[:len(ufac)]] = np.arange(len(ufac))
        hit = fidx[allinv[len(ufac):]]
        if (hit < 0).any():
            raise ValueError(f"{mesh.path}: {(hit < 0).sum()} tagged boundary elements are not faces of the mesh")
        ftag[hit] = tp

    d0 = cell_dom[c0]
    d1 = np.where(c1 >= 0, cell_dom[np.maximum(c1, 0)], 0)
    isb = (cnt == 1) | (d0 != d1) | (ftag != 0)
    bidx = np.nonzero(isb)[0]
    bf = ufac[bidx]
    lo, hi = np.minimum(d0[bidx], d1[bidx]), np.maximum(d0[bidx], d1[bidx])
    gkey_rows = np.stack([lo, hi, ftag[bidx]], axis=1)
    gk, ginv = np.unique(gkey_rows, axis=0, return_inverse=True)
    ginv = ginv.ravel()

    # boundaries: connected facets of one key (through shared sub-facets: edges in 3D, points in 2D)
    if dim == 3:
        sub = np.concatenate([np.sort(bf[:, list(l)], axis=1) for l in [(0, 1), (1, 2), (0, 2)]])
        sowner = np.tile(np.arange(len(bf)), 3)
        su, sinv = _row_keys(sub)
        ecount = np.bincount(sinv, minlength=len(su))
        o = np.argsort(sinv, kind="stable")
        pa, pb = o[:-1], o[1:]
        same_edge = sinv[pa] == sinv[pb]
        pa, pb = sowner[pa][same_edge], sowner[pb][same_edge]  # facet pairs sharing an edge
        if feature_angle is not None and len(bf):
            x = pts[bf]
            nrm = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1)[:, None]
            smooth = np.abs(np.einsum("ij,ij->i", nrm[pa], nrm[pb])) >= math.cos(math.radians(feature_angle))
            pa, pb = pa[smooth], pb[smooth]
        sub_key = np.zeros(len(bf), dtype=np.int64)  # extra split of closed surfaces (below)
        for _ in range(3):
            key = ginv * (int(sub_key.max()) + 1) + sub_key
            link = key[pa] == key[pb]
            bent = _components(len(bf), pa[link], pb[link])
            nb = int(bent.max()) + 1 if len(bf) else 0
            # feature edges: edges of boundary facets that are not interior to exactly one boundary
            eb = bent[sowner]
            o2 = np.lexsort((eb, sinv))
            se, sb = sinv[o2], eb[o2]
            distinct = np.ones(len(o2), bool)
            distinct[1:] = (se[1:] != se[:-1]) | (sb[1:] != sb[:-1])
            nd = np.bincount(se[distinct], minlength=len(su))
            feat = (nd != 1) | (ecount != 2)
            # a boundary without any bounding edge (a closed surface, e.g. a sealed conductor wall or the
            # outer sphere) is cut in two along its longest extent, so that every COMSOL face has edges
            has_edge = np.zeros(nb, bool)
            has_edge[eb[feat[sinv]]] = True
            closed = np.nonzero(~has_edge)[0]
            if not len(closed):
                break
            cen = pts[bf].mean(axis=1)
            nxt = int(sub_key.max()) + 1
            for k in closed.tolist():
                fk = np.nonzero(bent == k)[0]
                ext = np.ptp(cen[fk], axis=0)
                ax = int(np.argmax(ext))
                cut = np.median(cen[fk, ax])
                side = cen[fk, ax] > cut
                if side.all() or not side.any():
                    continue
                sub_key[fk[side]] = nxt
                nxt += 1
        fe = np.nonzero(feat)[0]
        # key of a feature edge: the sorted set of incident boundaries
        sets: dict[int, list[int]] = {}
        for e_, b_ in zip(se[distinct].tolist(), sb[distinct].tolist()):
            if feat[e_]:
                sets.setdefault(e_, []).append(b_)
        keymap: dict = {}
        ekey = np.array([keymap.setdefault(tuple(sets[e_]), len(keymap)) for e_ in fe.tolist()], dtype=np.int64)
        edges = su[fe]
        edge_ent, verts = _chains(edges, ekey)
        bnd_info = []
        for k in range(nb):
            g = ginv[np.nonzero(bent == k)[0][0]]
            bnd_info.append({"domains": (int(gk[g][0]), int(gk[g][1])), "attr": int(gk[g][2])})
        return _Partition(dim, pts, cells, cell_dom, dom_attr, bf, bent, bnd_info, edges, edge_ent,
                          np.array(verts, dtype=np.int64), mid)
    # 2D: boundaries are chains of boundary edges between points
    bent, verts = _chains(bf, ginv)
    nb = int(bent.max()) + 1 if len(bf) else 0
    bnd_info = []
    for k in range(nb):
        g = ginv[np.nonzero(bent == k)[0][0]]
        bnd_info.append({"domains": (int(gk[g][0]), int(gk[g][1])), "attr": int(gk[g][2])})
    return _Partition(dim, pts, cells, cell_dom, dom_attr, bf, bent, bnd_info,
                      np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.array(verts, dtype=np.int64), mid)


# =====================================================================================
# .mphtxt writer / reader
# =====================================================================================

def _tok(s: str) -> str:
    """A COMSOL serialized string: '<length> <string>' (length in UTF-8 bytes)."""
    return f"{len(s.encode('utf-8'))} {s}"


def _clean_label(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip()) or "unnamed"


def _measure(P: _Partition) -> np.ndarray:
    """Volume (3D) or area (2D) of every domain."""
    x = P.points[P.cells]
    if P.dim == 2:
        d, e = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
        m = 0.5 * np.abs(d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0])
    else:
        m = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    return np.bincount(P.cell_dom - 1, weights=m, minlength=len(P.dom_attr))


def write_mphtxt(mesh: GmshMesh, path: str | Path, *, order: int = 1, names: dict | None = None,
                 boundary_meta: dict | None = None, description: str = "", axisymmetric: bool | None = None,
                 feature_angle: float | None = None) -> dict:
    """Write a gmsh mesh as a COMSOL .mphtxt mesh with named selections; returns the entity map.

    ``names`` maps physical tags to names (default: the file's $PhysicalNames).  ``order`` = 2
    writes second-order elements (edg2/tri2/tet2) when the gmsh mesh has them.  ``feature_angle``
    (degrees, 3D) also splits boundaries where neighbouring facets meet at a sharper angle, which
    gives CAD-like faces (a box wall becomes six faces).  A JSON sidecar
    ``<path>.entities.json`` records the entity numbering (also returned).
    """
    path = Path(path)
    P = _partition(mesh, feature_angle)
    dim = P.dim
    if axisymmetric is None:
        axisymmetric = dim == 2
    vol_names = {t: n for (d, t), n in mesh.physical_names.items() if d == dim}
    fac_names = {t: n for (d, t), n in mesh.physical_names.items() if d == dim - 1}
    if names:
        for t, n in names.items():
            (vol_names if int(t) in set(P.dom_attr.tolist()) else fac_names)[int(t)] = n
    meta = boundary_meta or {}
    second = order >= 2 and bool(P.mid)
    if order >= 2 and not P.mid:
        order = 1  # the gmsh mesh is linear

    # ---- element lists (vertex rows into P.points)
    def with_mid(conn: np.ndarray, pairs: list) -> np.ndarray:
        cols = [conn]
        for a, b in pairs:
            lo, hi = np.minimum(conn[:, a], conn[:, b]), np.maximum(conn[:, a], conn[:, b])
            cols.append(np.array([P.mid[(x, y)] for x, y in zip(lo.tolist(), hi.tolist())], dtype=np.int64)[:, None])
        return np.hstack(cols)

    # COMSOL local numbering (Programming Reference Manual, "Mesh Element Numbering Conventions"):
    # edg2: v0 v1 m01; tri2: v0 v1 v2 m01 m02 m12; tet2: v0 v1 v2 v3 m01 m02 m12 m03 m13 m23
    pairs = {"edg": [(0, 1)], "tri": [(0, 1), (0, 2), (1, 2)],
             "tet": [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]}
    types = []  # (name, conn, dom)
    types.append(("vtx", P.vertices[:, None], np.arange(len(P.vertices))))
    if dim == 3:
        types.append(("edg", P.edges, P.edge_ent))
        types.append(("tri", P.bfacets, P.bfacet_ent))
        types.append(("tet", P.cells, P.cell_dom))
    else:
        types.append(("edg", P.bfacets, P.bfacet_ent))
        types.append(("tri", P.cells, P.cell_dom))
    if second:
        types = [(n if n == "vtx" else n + "2", c if n == "vtx" else with_mid(c, pairs[n]), d) for n, c, d in types]

    used = np.unique(np.concatenate([c.ravel() for _, c, _ in types]))
    renum = np.full(len(P.points), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    xyz = P.points[used]

    # ---- selections
    ndom = len(P.dom_attr)
    nbnd = len(P.bnd_info)
    sels = []  # (label, dim, entities)
    attr_doms: dict[int, list[int]] = {}
    for k, a in enumerate(P.dom_attr.tolist()):
        attr_doms.setdefault(a, []).append(k + 1)
    taken = set()

    def add(label: str, sdim: int, ents: list[int]) -> None:
        label = _clean_label(label)
        base, i = label, 2
        while label in taken:
            label, i = f"{base}_{i}", i + 1
        taken.add(label)
        sels.append((label, sdim, sorted(ents)))

    region_label = {}
    for a in sorted(attr_doms):
        region_label[a] = vol_names.get(a, f"attr_{a}")
        add(region_label[a], dim, attr_doms[a])
    tag_bnds: dict[int, list[int]] = {}
    for k, b in enumerate(P.bnd_info):
        if b["attr"]:
            tag_bnds.setdefault(b["attr"], []).append(k)
    for a in sorted(tag_bnds):
        add(fac_names.get(a, f"boundary_attr_{a}"), dim - 1, tag_bnds[a])
    for a in sorted(tag_bnds):  # PEC holes (mesh.remove_pec): also name them after their conductor regions
        regs = (meta.get(a) or {}).get("regions")
        if regs:
            add("walls_of_" + "+".join(sorted(regs))[:80], dim - 1, tag_bnds[a])
    for a in sorted(attr_doms):  # every boundary of each region (interfaces included)
        ds = set(attr_doms[a])
        ents = [k for k, b in enumerate(P.bnd_info) if b["domains"][0] in ds or b["domains"][1] in ds]
        if ents:
            add(region_label[a] + "_boundary", dim - 1, ents)
    ext = [k for k, b in enumerate(P.bnd_info) if b["domains"][0] == 0]
    if ext:
        add("exterior_boundary", dim - 1, ext)

    # ---- write
    tag = "mesh1"
    stags = [f"sel{i + 1}" for i in range(len(sels))]
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        w = f.write
        w("# Created by axionFEM (python/axionfem/comsol.py); COMSOL native text mesh, file format 0.1\n")
        w("# Specification: COMSOL Multiphysics Programming Reference Manual, 'The COMSOL File Formats':\n")
        w("#   'File Structure', 'Mesh' (class version 4), 'Selection' (version 0),\n")
        w("#   'Mesh Element Numbering Conventions'.  Import: Mesh > Import > COMSOL Multiphysics file.\n")
        if description:
            for ln in description.splitlines():
                w(f"# {ln}\n")
        w(f"# Coordinates in metres; {'2D axisymmetric (r, z)' if (dim == 2 and axisymmetric) else f'{dim}D'}\n")
        w("# Major & minor version\n0 1\n")
        w(f"{1 + len(sels)} # number of tags\n# Tags\n")
        for t in [tag] + stags:
            w(_tok(t) + "\n")
        w(f"{1 + len(sels)} # number of types\n# Types\n")
        for _ in range(1 + len(sels)):
            w("3 obj\n")
        w("\n# --------- Object 0 ----------\n\n0 0 1\n4 Mesh # class\n4 # version\n")
        w(f"{dim} # sdim\n{len(xyz)} # number of mesh vertices\n0 # lowest mesh vertex index\n\n# Mesh vertex coordinates\n")
        np.savetxt(f, xyz, fmt="%.17g")
        w(f"\n{len(types)} # number of element types\n")
        for k, (name, conn, dom) in enumerate(types):
            w(f"\n# Type #{k}\n\n{_tok(name)} # type name\n\n\n{conn.shape[1]} # number of vertices per element\n")
            w(f"{len(conn)} # number of elements\n# Elements\n")
            if len(conn):
                np.savetxt(f, renum[conn], fmt="%d")
            w(f"\n{len(dom)} # number of geometric entity indices\n# Geometric entity indices\n")
            if len(dom):
                np.savetxt(f, np.asarray(dom, dtype=np.int64)[:, None], fmt="%d")
        for i, (label, sdim, ents) in enumerate(sels):
            w(f"\n# --------- Object {i + 1} ----------\n\n0 0 1\n9 Selection # class\n0 # version\n")
            w(f"{_tok(label)} # label\n{_tok(tag)} # mesh tag\n{sdim} # dimension\n{len(ents)} # number of entities\n# Entities\n")
            for e in ents:
                w(f"{e}\n")
    os.replace(tmp, path)

    meas = _measure(P)
    # a point inside each entity (centroid of one of its elements), to find it in COMSOL with a Ball Selection
    _, first_cell = np.unique(P.cell_dom, return_index=True)
    dprobe = [[float(v) for v in P.points[P.cells[c]].mean(axis=0)] for c in first_cell]
    _, first_f = np.unique(P.bfacet_ent, return_index=True)
    bprobe = [[float(v) for v in P.points[P.bfacets[f]].mean(axis=0)] for f in first_f]
    blo = np.full((nbnd, dim), np.inf)
    bhi = np.full((nbnd, dim), -np.inf)
    if len(P.bfacets):
        fp = P.points[P.bfacets]
        np.minimum.at(blo, P.bfacet_ent, fp.min(axis=1))
        np.maximum.at(bhi, P.bfacet_ent, fp.max(axis=1))
    mdesc = "area_m2" if dim == 2 else "volume_m3"
    info = {
        "file": str(path),
        "source": mesh.path,
        "sdim": dim,
        "axisymmetric": bool(dim == 2 and axisymmetric),
        "order": 2 if second else 1,
        "counts": {"mesh_vertices": int(len(xyz)), **{n: int(len(c)) for n, c, _ in types},
                   "domains": int(ndom), "boundaries": int(nbnd),
                   "edges": int(P.edge_ent.max() + 1) if (dim == 3 and len(P.edge_ent)) else (int(nbnd) if dim == 2 else 0),
                   "points": int(len(P.vertices))},
        "numbering": "domains are 1-based; boundaries, edges and points are 0-based (COMSOL mesh file convention); "
                     "COMSOL shows every index +1 in the GUI",
        "domains": [{"index": k + 1, "attr": int(a), "region": region_label[int(a)], mdesc: float(meas[k]),
                     "probe_point": dprobe[k]}
                    for k, a in enumerate(P.dom_attr.tolist())],
        "boundaries": [{"index": k, "domains": list(b["domains"]), "attr": b["attr"],
                        "group": fac_names.get(b["attr"]) if b["attr"] else None, "probe_point": bprobe[k],
                        "bbox": [blo[k].tolist(), bhi[k].tolist()],
                        **({"meta": meta[b["attr"]]} if b["attr"] in meta else {})}
                       for k, b in enumerate(P.bnd_info)],
        "selections": [{"label": l, "dim": d, "entities": e} for l, d, e in sels],
    }
    Path(str(path) + ".entities.json").write_text(json.dumps(info, indent=1))
    return info


def read_mphtxt(path: str | Path) -> dict:
    """Parse a COMSOL .mphtxt file (Mesh and Selection objects) for checks.

    Returns {"sdim", "vertices" (N, sdim), "types": {name: {"elements", "entities"}}, "selections": [...]}.
    """
    toks: list[str] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            toks.extend(ln.split())
    it = iter(toks)
    nx = lambda: next(it)  # noqa: E731

    def string() -> str:
        n = int(nx())
        s = nx()
        while len(s.encode("utf-8")) < n:
            s += " " + nx()
        return s

    major, minor = int(nx()), int(nx())
    if (major, minor) != (0, 1):
        raise ValueError(f"file format {major}.{minor}; expected 0.1")
    ntags = int(nx())
    tags = [string() for _ in range(ntags)]
    ntypes = int(nx())
    for _ in range(ntypes):
        string()
    out: dict = {"tags": tags, "types": {}, "selections": []}
    for _ in range(ntags):
        hdr = (nx(), nx(), nx())
        if hdr != ("0", "0", "1"):
            raise ValueError(f"bad object header {hdr}")
        cls = string()
        if cls == "Mesh":
            ver = int(nx())
            sdim = int(nx())
            npnt = int(nx())
            low = int(nx())
            v = np.array([float(nx()) for _ in range(npnt * sdim)]).reshape(npnt, sdim)
            out.update(version=ver, sdim=sdim, lowest=low, vertices=v)
            for _ in range(int(nx())):
                name = string()
                nep, ne = int(nx()), int(nx())
                el = np.array([int(nx()) for _ in range(nep * ne)], dtype=np.int64).reshape(ne, nep) - low
                nd = int(nx())
                dom = np.array([int(nx()) for _ in range(nd)], dtype=np.int64)
                out["types"][name] = {"elements": el, "entities": dom}
        elif cls == "Selection":
            int(nx())
            label = string()
            mtag = string()
            d = int(nx())
            n = int(nx())
            out["selections"].append({"label": label, "mesh": mtag, "dim": d, "entities": [int(nx()) for _ in range(n)]})
        else:
            raise ValueError(f"unsupported object class {cls}")
    return out


def gmsh_to_mphtxt(msh: str | Path, out: str | Path, *, order: int = 1, info_json: str | Path | None = None,
                   mesher_input: str | Path | None = None, feature_angle: float | None = None) -> dict:
    """Convert an axf-mesher .msh (and its mesh*_info.json / mesh*_in.json, found next to it by
    default) to .mphtxt; returns the entity map (also written to ``<out>.entities.json``)."""
    msh = Path(msh)
    stem = msh.with_suffix("")
    info_json = Path(info_json) if info_json else Path(str(stem) + "_info.json")
    mesher_input = Path(mesher_input) if mesher_input else Path(str(stem) + "_in.json")
    meta: dict = {}
    names: dict = {}
    desc = [f"source mesh: {msh.resolve()}"]
    if info_json.is_file():
        inf = json.loads(info_json.read_text())
        for g in inf.get("boundary_groups", []):
            meta[int(g["attr"])] = g
        if inf.get("pec_conductors"):
            for c in inf["pec_conductors"]:
                a = 1000 + int(c["body_id"])
                meta.setdefault(a, {})["regions"] = c["regions"]
        if inf.get("symmetry"):
            desc.append("symmetry clipping: " + json.dumps(inf["symmetry"]))
        if inf.get("removed_pec_volumes"):
            desc.append(f"{inf['removed_pec_volumes']} PEC volumes were removed by the mesher (mesh.remove_pec): "
                        "they are holes whose walls are the pec_body_* boundary selections")
    if mesher_input.is_file():
        mi = json.loads(mesher_input.read_text())
        for r in mi.get("regions", []):
            names[int(r["attr"])] = r["name"]
        for k, n in (("axis_attr", "axis"), ("outer_attr", "outer"), ("air_attr", "air"), ("shell_attr", "infinity_shell")):
            if k in mi:
                names.setdefault(int(mi[k]), n)
    mesh = read_gmsh(msh)
    return write_mphtxt(mesh, out, order=order, names=names, boundary_meta=meta, description="\n".join(desc),
                        feature_angle=feature_angle)


def write_import_java(mphtxt: str | Path, path: str | Path, *, sdim: int, axisymmetric: bool = False,
                      class_name: str | None = None, mph_name: str | None = None,
                      physics: dict | None = None, entities: dict | None = None) -> dict:
    """A small COMSOL Java program that imports an .mphtxt mesh (Mesh > Import) and saves an .mph.

    Compile with ``comsolcompile`` and run with ``comsol batch -inputfile <Class>.class``.  With ``physics`` (a
    :func:`axionfem.comsol_physics.physics_spec`) and ``entities`` (the entity map of the mesh) it also attaches the
    emw model to Explicit selections named like the mesh's selections and solves the imp/volt runs.
    """
    path = Path(path)
    cls = class_name or path.stem
    if not re.match(r"^[A-Za-z_]\w*$", cls):
        raise ValueError(f"'{cls}' is not a valid Java class name")
    mph = mph_name or (Path(mphtxt).stem + ".mph")
    fname = str(Path(mphtxt).resolve())
    if physics is not None:
        return _write_import_physics_java(Path(mphtxt), path, cls, mph, fname, sdim, physics, entities)
    src = f"""/*
 * {cls}.java -- import the axionFEM mesh {Path(mphtxt).name} into COMSOL (generated by axionfem.comsol).
 *   comsolcompile {cls}.java && comsol batch -inputfile {cls}.class     # -> {mph}
 * The mesh file holds named selections (regions, tagged boundaries); {Path(mphtxt).name}.entities.json maps
 * every domain/boundary index to its axionFEM region and gives a point inside each one. See docs/comsol_export.md.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;

public class {cls} {{

  public static Model run() {{
    Model model = ModelUtil.create("Model");
    model.label({json.dumps(mph)});
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", {sdim});
{'    model.component("comp1").geom("geom1").axisymmetric(true);' + chr(10) if axisymmetric else ''}    model.component("comp1").geom("geom1").lengthUnit("m");
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").create("imp1", "Import");
    model.component("comp1").mesh("mesh1").feature("imp1").set("filename", {json.dumps(fname)});
    model.component("comp1").mesh("mesh1").feature("imp1").importData();
    model.component("comp1").mesh("mesh1").run();
    return model;
  }}

  public static void main(String[] args) {{
    Model model = run();
    try {{
      model.save({json.dumps(mph)});
    }} catch (Exception e) {{
      throw new RuntimeException(e);
    }}
  }}
}}
"""
    path.write_text(src)
    return {"file": str(path), "class": cls, "mph": mph}


def _write_import_physics_java(mphtxt: Path, path: Path, cls: str, mph: str, fname: str, sdim: int,
                               spec: dict, entities: dict | None) -> dict:
    from . import comsol_physics as cp

    if sdim != 3:
        raise ValueError("physics export needs a 3D mesh")
    info = entities if entities is not None else json.loads(Path(str(mphtxt) + ".entities.json").read_text())
    sel = cp.import_selections(info, spec)
    files = {k: f'new java.io.File(System.getProperty("user.dir"), {json.dumps(cls + "_" + k + ".txt")}).getAbsolutePath()'
             for k in ("impedance", "voltage")}
    code = cp.physics_java(spec, sel["refs"], imp_file=files["impedance"], volt_file=files["voltage"])
    lines = ['model.component().create("comp1", true);',
             'model.component("comp1").geom().create("geom1", 3);',
             'model.component("comp1").geom("geom1").lengthUnit("m");']
    setup = [ln for ln in code["setup"] if 'mesh().create("mesh1")' not in ln]
    lines += setup[:1]   # Je_scale parameter
    lines += ['model.component("comp1").mesh().create("mesh1");',
              'model.component("comp1").mesh("mesh1").create("imp1", "Import");',
              f'model.component("comp1").mesh("mesh1").feature("imp1").set("filename", {json.dumps(fname)});',
              'model.component("comp1").mesh("mesh1").feature("imp1").importData();',
              'model.component("comp1").mesh("mesh1").run();']
    lines += cp.selection_java(sel["selections"])
    lines += setup[1:]
    notes = [f"Imports the axionFEM mesh {mphtxt.name} (Mesh > Import) and attaches the emw physics to Explicit selections",
             "built from its entity map (.entities.json); entity numbers: domains as in the file, boundaries = file index + 1",
             "(UNVERIFIED: check the domain volumes against entities.json once).", ""]
    notes += [f"approximation: {a}" for a in sel["approx"]] + [""] + cp.physics_notes(spec)
    imp = [x for x in code["imp"] if '.mesh("mesh1").run()' not in x]
    src = _java_class(cls, notes, [], [("    " + x) for x in lines], mph,
                      build=("Build and run (needs the RF Module; main() solves the imp and volt runs and writes "
                             f"{cls}_impedance.txt / {cls}_voltage.txt to the working directory):"),
                      solves=(imp, code["volt"]))
    path.write_text(src)
    return {"file": str(path), "class": cls, "mph": mph, "approximations": sel["approx"],
            "physics": _physics_summary(spec, sel["port_selection"], cls),
            "physics_selections": {lab: ents for _, lab, _, ents in sel["selections"]}}


# =====================================================================================
# expressions: axionFEM -> COMSOL
# =====================================================================================

# COMSOL built-in constants and the spatial/time variables of 2D-axisymmetric and 3D components
_COMSOL_RESERVED = {"pi", "i", "j", "eps", "inf", "Inf", "NaN", "nan", "x", "y", "z", "r", "phi", "t", "e",
                    "freq", "omega", "lambda", "true", "false"}
_FUNCS = {"sqrt", "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "abs", "min", "max", "pow"}


def comsol_param_name(name: str) -> str:
    """COMSOL-safe parameter name (names that COMSOL reserves for constants or coordinates get a prefix)."""
    n = re.sub(r"\W", "_", name)
    if n in _COMSOL_RESERVED or not re.match(r"[A-Za-z]", n):
        n = "par_" + n
    return n


_MU0 = 1.25663706212e-6        # H/m, CODATA 2018, as cpp/include/axionfem/units.hpp
_MAX_EXPR_DEPTH = 1000          # as kMaxExprDepth in cpp/src/expr.cpp
_NUMBER_HINT = "write numbers as 12, 1.25, .5, 2. or 1.5e-3 (one decimal point, digits in the exponent)"
_NUMBER_RE = re.compile(r"[0-9.]*(?:[eE][+-]?[0-9]+)?")


def _ieee(f, *a):
    """``f(*a)`` with the C library's IEEE results (inf, nan) instead of Python's exceptions."""
    try:
        return f(*a)
    except OverflowError:
        return math.inf
    except (ValueError, ZeroDivisionError):
        return math.nan


def _div(a, b):
    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _pow(b, e):
    """std::pow semantics: a negative base with a non-integer exponent is NaN (never complex)."""
    try:
        return math.pow(b, e)
    except ValueError:
        return math.inf if b == 0.0 and e < 0 else math.nan
    except OverflowError:
        odd = float(e).is_integer() and int(e) % 2 == 1
        return -math.inf if b < 0 and odd else math.inf


def _log(x):
    if x == 0.0:
        return -math.inf
    return _ieee(math.log, x)


def _log10(x):
    if x == 0.0:
        return -math.inf
    return _ieee(math.log10, x)


_FN1 = {"sqrt": lambda x: _ieee(math.sqrt, x), "exp": lambda x: _ieee(math.exp, x), "log": _log,
        "log10": _log10, "sin": lambda x: _ieee(math.sin, x), "cos": lambda x: _ieee(math.cos, x),
        "tan": lambda x: _ieee(math.tan, x), "asin": lambda x: _ieee(math.asin, x),
        "acos": lambda x: _ieee(math.acos, x), "atan": math.atan, "abs": abs}


class _ExprParser:
    """Recursive-descent parser with the grammar and arithmetic of cpp/src/expr.cpp; emits COMSOL text.

    Every unary minus, product, quotient, power and group is parenthesized; sums are left bare only at the top
    level, inside parentheses and as function arguments.  COMSOL documents ``^`` above unary minus but not the
    associativity of ``^``; with these parentheses neither matters.
    """

    def __init__(self, s: str, params: dict):
        self.s, self.i, self.p = s, 0, params
        self.depth = 0

    def fail(self, msg, hint="use numbers, parameter names, + - * / ^, parentheses and functions such as "
                             "sqrt, exp, log, min, max"):
        raise ValueError(f"cannot translate expression '{self.s}': {msg} (near position {self.i}); {hint}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def eat(self, c):
        self.ws()
        if self.s.startswith(c, self.i):
            self.i += len(c)
            return True
        return False

    def parse(self):
        import sys

        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 8 * _MAX_EXPR_DEPTH + 500))   # 5 frames per nesting level
        try:
            v, t = self.expr()
        finally:
            sys.setrecursionlimit(old)
        self.ws()
        if self.i != len(self.s):
            self.fail("trailing characters")
        if not math.isfinite(v):
            raise ValueError(f"expression '{self.s}' evaluated to a non-finite value; check for division by zero "
                             "or log of a non-positive number")
        return v, t

    def expr(self):
        v, t = self.term()
        while True:
            if self.eat("+"):
                v2, t2 = self.term()
                v, t = v + v2, f"{t}+{t2}"
            elif self.eat("-"):
                v2, t2 = self.term()
                v, t = v - v2, f"{t}-({t2})"
            else:
                return v, t

    def term(self):
        v, t = self.unary()
        while True:
            if self.eat("*"):
                v2, t2 = self.unary()
                v, t = v * v2, f"({t})*({t2})"
            elif self.eat("/"):
                v2, t2 = self.unary()
                v, t = _div(v, v2), f"({t})/({t2})"
            else:
                return v, t

    # standard precedence as in cpp/src/expr.cpp: unary minus binds looser than '^'
    # (-2^2 = -4), '^' is right-associative (2^3^2 = 2^9) and its exponent may be signed.
    # Every nesting level passes through unary(), which caps the depth as the core does.
    def unary(self):
        self.depth += 1
        if self.depth > _MAX_EXPR_DEPTH:
            self.fail(f"nesting deeper than {_MAX_EXPR_DEPTH} levels",
                      "simplify the expression, or split it into several parameters")
        if self.eat("-"):
            v, t = self.unary()
            out = -v, f"(-({t}))"
        elif self.eat("+"):
            out = self.unary()
        else:
            out = self.power()
        self.depth -= 1
        return out

    def power(self):
        v, t = self.primary()
        if self.eat("^"):
            v2, t2 = self.unary()
            return _pow(v, v2), f"({t})^({t2})"
        return v, t

    def number(self):
        tok = _NUMBER_RE.match(self.s, self.i).group(0)
        # strict, as the core: the whole token is one number, and it is 0 or a normal double
        if not re.fullmatch(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", tok):
            self.fail(f"malformed number '{tok}'", _NUMBER_HINT)
        v = float(tok)
        mantissa_zero = re.search(r"[1-9]", re.split(r"[eE]", tok)[0]) is None
        if math.isinf(v) or (not mantissa_zero and abs(v) < 2.2250738585072014e-308):
            self.fail(f"number '{tok}' is out of the range of double precision",
                      "numbers must be 0 or have a magnitude between 2.2e-308 and 1.7e308; check the exponent")
        self.i += len(tok)
        return v, tok

    def primary(self):
        self.ws()
        if self.i >= len(self.s):
            self.fail("unexpected end")
        if self.eat("("):
            v, t = self.expr()
            if not self.eat(")"):
                self.fail("missing ')'")
            return v, f"({t})"
        if self.s[self.i].isdigit() or self.s[self.i] == ".":
            return self.number()
        m = re.compile(r"[A-Za-z_]\w*").match(self.s, self.i)
        if not m:
            self.fail(f"unexpected character '{self.s[self.i]}'")
        name = m.group(0)
        self.i = m.end()
        if self.eat("("):
            args = []
            if not self.eat(")"):
                args = [self.expr()]
                while self.eat(","):
                    args.append(self.expr())
                if not self.eat(")"):
                    self.fail("missing ')' after function arguments")
            return self.call(name, args)
        if name in self.p:
            return float(self.p[name]), comsol_param_name(name)
        if name == "pi":
            return math.pi, "pi"
        if name == "e":
            return math.e, "exp(1)"
        if name == "mu0":
            return _MU0, "mu0_const"
        self.fail(f"unknown name '{name}'", "define it under \"parameters\" or use a number")

    def call(self, name, args):
        vs = [a[0] for a in args]
        ts = [a[1] for a in args]
        if name not in _FUNCS:
            self.fail(f"unknown function '{name}'")
        if name in ("min", "max"):
            if not vs:
                self.fail(f"{name}() needs arguments")
            for k, x in enumerate(vs):
                if math.isnan(x):
                    self.fail(f"{name}() argument {k + 1} is not a number (NaN)",
                              "check for sqrt, log, asin or acos of an out-of-range value, or 0/0, inside the "
                              "arguments")
            return (min(vs) if name == "min" else max(vs)), f"{name}({','.join(ts)})"
        n = 2 if name in ("pow", "atan2") else 1
        if len(vs) != n:
            self.fail(f"{name}() takes {n} argument(s)")
        if name == "pow":
            return _pow(vs[0], vs[1]), f"({ts[0]})^({ts[1]})"
        if name == "atan2":
            return math.atan2(vs[0], vs[1]), f"atan2({ts[0]},{ts[1]})"
        return _FN1[name](vs[0]), f"{name}({ts[0]})"


def comsol_expression(text: str, params: dict) -> tuple[float, str]:
    """Translate an axionFEM expression to COMSOL syntax; returns (value, COMSOL text).

    ``pow(a, b)`` becomes ``(a)^(b)``, ``e`` becomes ``exp(1)``, ``mu0`` becomes ``mu0_const``.  The value
    follows the core's grammar (standard precedence: ``-2^2`` is -4, ``2^3^2`` is 512) and arithmetic.  COMSOL
    documents ``^`` above unary minus but not the associativity of ``^``; every unary minus, product, quotient,
    power and group is parenthesized (sums stay bare only at the top level, in parentheses and as arguments),
    so neither matters.  Raises ValueError, with the expression and a hint, where the core would reject it.
    """
    return _ExprParser(str(text), params).parse()


# =====================================================================================
# COMSOL Java API model file
# =====================================================================================

def _num(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    r = repr(x)
    return r[:-2] if r.endswith(".0") else r


class _Q:
    """A number with the COMSOL text that produces it (a parameter expression or a literal)."""

    __slots__ = ("v", "t")

    def __init__(self, v: float, t: str | None = None):
        self.v = float(v)
        self.t = t if t is not None else _num(v)

    def _lit(self):
        return self.t == _num(self.v)

    def __add__(self, o):
        o = o if isinstance(o, _Q) else _Q(o)
        if self._lit() and o._lit():
            return _Q(self.v + o.v)
        return _Q(self.v + o.v, f"{self.t}+{o.t}")

    def __sub__(self, o):
        o = o if isinstance(o, _Q) else _Q(o)
        if self._lit() and o._lit():
            return _Q(self.v - o.v)
        return _Q(self.v - o.v, f"{self.t}-({o.t})")

    def __mul__(self, k):
        k = k if isinstance(k, _Q) else _Q(k)
        if self._lit() and k._lit():
            return _Q(self.v * k.v)
        return _Q(self.v * k.v, f"({self.t})*({k.t})")


def _pair(raw: Any, res: Any, params: dict) -> Any:
    """Walk the resolved geometry alongside the raw design: numbers become _Q carrying the raw expression."""
    if isinstance(res, dict):
        rd = raw if isinstance(raw, dict) else {}
        return {k: _pair(rd.get(k), v, params) for k, v in res.items()}
    if isinstance(res, list):
        rl = raw if isinstance(raw, list) and len(raw) == len(res) else [None] * len(res)
        return [_pair(a, b, params) for a, b in zip(rl, res)]
    if isinstance(res, bool) or not isinstance(res, (int, float)):
        return res
    if isinstance(raw, str):
        try:
            v, t = comsol_expression(raw, params)
            if abs(v - res) <= 1e-9 * max(1.0, abs(res)):
                return _Q(res, t)
        except (ValueError, ArithmeticError, TypeError, RecursionError):
            pass        # not translatable: write the number the core resolved
    return _Q(res)


def _fillet_polygon(P: list, F: list) -> tuple[list, list, list]:
    """Closed outline with per-vertex fillets as a rational Bezier polygon (as axf-mesher's polygon_wire).

    Returns (control points [(x, y)], degrees, weights) for COMSOL's BezierPolygon: lines are degree 1,
    each fillet arc a degree-2 rational segment (weights 1, cos(a/2), 1), exact circles.
    """
    n = len(P)
    C = []
    for i in range(n):
        px, py = P[(i - 1) % n]
        vx, vy = P[i]
        qx, qy = P[(i + 1) % n]
        ux, uy, wx, wy = px - vx, py - vy, qx - vx, qy - vy
        lu, lw = math.hypot(ux, uy), math.hypot(wx, wy)
        ux, uy, wx, wy = ux / lu, uy / lu, wx / lw, wy / lw
        ang = math.acos(max(-1.0, min(1.0, ux * wx + uy * wy)))
        f = F[i] if i < len(F) else 0.0
        if f <= 0 or ang < 1e-6 or abs(math.pi - ang) < 1e-6:
            C.append(((vx, vy), None, None))
            continue
        d = f / math.tan(ang / 2)
        C.append(((vx + d * ux, vy + d * uy), (vx, vy), (vx + d * wx, vy + d * wy), ang))
    pts, deg, w = [], [], []
    def same(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) < 1e-12 * (1 + abs(a[0]) + abs(a[1]))
    for i in range(n):
        c = C[i]
        if c[1] is None:
            a = b = c[0]
        else:
            a, b = c[0], c[2]
        if not pts:
            pts.append(a)
        elif not same(pts[-1], a):
            pts.append(a); deg.append(1); w += [1.0, 1.0]
        if c[1] is not None and not same(a, b):
            ang = c[3]
            pts += [c[1], b]
            deg.append(2)
            w += [1.0, math.sin(ang / 2), 1.0]  # arc of opening (pi - ang): weight cos((pi-ang)/2) = sin(ang/2)
    if not same(pts[-1], pts[0]):
        pts.append(pts[0]); deg.append(1); w += [1.0, 1.0]
    else:
        pts[-1] = pts[0]
    return pts, deg, w


class _Java:
    """Emits COMSOL Java API calls for one geometry sequence."""

    def __init__(self, comp: str = "comp1", geom: str = "geom1"):
        self.lines: list[str] = []
        self.n: dict[str, int] = {}
        self.G = f'model.component("{comp}").geom("{geom}")'
        self.approx: list[str] = []

    def tag(self, prefix: str) -> str:
        self.n[prefix] = self.n.get(prefix, 0) + 1
        return f"{prefix}{self.n[prefix]}"

    def w(self, s: str) -> None:
        self.lines.append("    " + s)

    @staticmethod
    def sarr(vals) -> str:
        return "new String[]{" + ", ".join(json.dumps(v.t if isinstance(v, _Q) else _num(v)) for v in vals) + "}"

    @staticmethod
    def darr(vals) -> str:
        return "new double[]{" + ", ".join(_num(v.v if isinstance(v, _Q) else v) for v in vals) + "}"

    def feat(self, seq: str, ftype: str, prefix: str, props: list[tuple[str, str]], label: str | None = None) -> str:
        t = self.tag(prefix)
        self.w(f'{seq}.create("{t}", "{ftype}");')
        for k, v in props:
            self.w(f'{seq}.feature("{t}").set("{k}", {v});')
        if label:
            self.w(f'{seq}.feature("{t}").label({json.dumps(label)});')
        return t

    def boolean(self, seq: str, kind: str, first: list[str], rest: list[str]) -> str:
        if kind == "union":
            t = self.feat(seq, "Union", "uni", [("intbnd", "false")])
            self.w(f'{seq}.feature("{t}").selection("input").set({", ".join(json.dumps(x) for x in first + rest)});')
        elif kind == "difference":
            t = self.feat(seq, "Difference", "dif", [("intbnd", "false")])
            self.w(f'{seq}.feature("{t}").selection("input").set({", ".join(json.dumps(x) for x in first)});')
            self.w(f'{seq}.feature("{t}").selection("input2").set({", ".join(json.dumps(x) for x in rest)});')
        else:
            t = self.feat(seq, "Intersection", "int", [("intbnd", "false")])
            self.w(f'{seq}.feature("{t}").selection("input").set({", ".join(json.dumps(x) for x in first + rest)});')
        return t

    # ---------------------------------------------------------------- 2D shapes (r, z) -> x, y
    def rect(self, seq, r0, r1, z0, z1, fillet: _Q | None, kind="solid") -> str:
        t = self.feat(seq, "Rectangle", "r", [("type", json.dumps(kind)), ("pos", self.sarr([r0, z0])),
                                              ("size", self.sarr([r1 - r0, z1 - z0]))])
        if fillet is not None and fillet.v > 0:
            f = self.feat(seq, "Fillet", "fil", [("radius", json.dumps(fillet.t))])
            self.w(f'{seq}.feature("{f}").selection("point").set("{t}", new int[]{{1, 2, 3, 4}});')
            return f
        return t

    def shape(self, seq: str, s: dict) -> str:
        (k, b), = s.items()
        if k == "rect":
            return self.rect(seq, b["r"][0], b["r"][1], b["z"][0], b["z"][1], b.get("fillet"))
        if k == "frame":
            f = b.get("fillet") or _Q(0)
            t = b["thickness"]
            outer = self.rect(seq, b["r"][0], b["r"][1], b["z"][0], b["z"][1], f)
            fin = _Q(max(f.v - t.v, 0.0), f"max({f.t}-({t.t}),0)") if f.v > 0 else None
            inner = self.rect(seq, b["r"][0] + t, b["r"][1] - t, b["z"][0] + t, b["z"][1] - t,
                              fin if (fin is not None and fin.v > 0) else None)
            return self.boolean(seq, "difference", [outer], [inner])
        if k == "polygon":
            P = [(p[0].v, p[1].v) for p in b["points"]]
            f0 = b.get("fillet").v if b.get("fillet") is not None else 0.0
            F = [x.v for x in b["fillets"]] if b.get("fillets") else [f0] * len(P)
            if any(x > 0 for x in F):
                pts, deg, wts = _fillet_polygon(P, F)
                xs = ", ".join(_num(p[0]) for p in pts)
                ys = ", ".join(_num(p[1]) for p in pts)
                return self.feat(seq, "BezierPolygon", "b", [
                    ("type", '"solid"'), ("p", f"new double[][]{{{{{xs}}}, {{{ys}}}}}"),
                    ("degree", "new int[]{" + ", ".join(str(d) for d in deg) + "}"),
                    ("w", "new double[]{" + ", ".join(_num(x) for x in wts) + "}")])
            rows = ", ".join("{" + json.dumps(p[0].t) + ", " + json.dumps(p[1].t) + "}" for p in b["points"])
            return self.feat(seq, "Polygon", "pol", [("type", '"solid"'), ("source", '"table"'),
                                                      ("table", f"new String[][]{{{rows}}}")])
        if k == "disk":
            return self.feat(seq, "Circle", "c", [("r", json.dumps(b["radius"].t)), ("pos", self.sarr(b["center"]))])
        if k in ("union", "difference"):
            tags = [self.shape(seq, c) for c in b]
            return self.boolean(seq, k, tags[:1], tags[1:])
        if k == "contour":
            raise ValueError("the 'contour' shape (exact elliptical arcs) is not exported to COMSOL yet; "
                             "export the faceted equivalent instead (srf designs: wall='polygon')")
        raise ValueError(f"unknown shape '{k}'")

    def path(self, seq: str, s: dict) -> list[str]:
        (k, b), = s.items()
        if k == "segment" or k == "polyline":
            pts = b if k == "segment" else b["points"]
            closed = k == "polyline" and bool(b.get("closed"))
            if closed:
                pts = list(pts) + [pts[0]]
            rows = ", ".join("{" + json.dumps(p[0].t) + ", " + json.dumps(p[1].t) + "}" for p in pts)
            return [self.feat(seq, "Polygon", "pol", [("type", '"open"'), ("source", '"table"'),
                                                       ("table", f"new String[][]{{{rows}}}")])]
        if k == "rect":
            return [self.rect(seq, b["r"][0], b["r"][1], b["z"][0], b["z"][1], b.get("fillet"), kind="curve")]
        if k == "arc":
            cx, cz, R = b["center"][0].v, b["center"][1].v, b["radius"].v
            a0, a1 = b["angles"][0].v, b["angles"][1].v
            nseg = max(1, int(math.ceil(abs(a1 - a0) / (math.pi / 2) - 1e-9)))
            pts, deg, wts = [], [], []
            for s_ in range(nseg):
                t0 = a0 + (a1 - a0) * s_ / nseg
                t1 = a0 + (a1 - a0) * (s_ + 1) / nseg
                h = (t1 - t0) / 2
                mid = (cx + R * math.cos(t0 + h) / math.cos(h), cz + R * math.sin(t0 + h) / math.cos(h))
                if not pts:
                    pts.append((cx + R * math.cos(t0), cz + R * math.sin(t0)))
                pts += [mid, (cx + R * math.cos(t1), cz + R * math.sin(t1))]
                deg.append(2)
                wts += [1.0, math.cos(h), 1.0]
            xs = ", ".join(_num(p[0]) for p in pts)
            ys = ", ".join(_num(p[1]) for p in pts)
            return [self.feat(seq, "BezierPolygon", "b", [
                ("type", '"open"'), ("p", f"new double[][]{{{{{xs}}}, {{{ys}}}}}"),
                ("degree", "new int[]{" + ", ".join(str(d) for d in deg) + "}"),
                ("w", "new double[]{" + ", ".join(_num(x) for x in wts) + "}")])]
        raise ValueError(f"unknown path '{k}'")

    # ---------------------------------------------------------------- 3D solids
    def workplane(self, plane: str, offset: _Q | None = None) -> tuple[str, str]:
        props = [("quickplane", json.dumps(plane)), ("unite", "true")]
        if offset is not None:
            props.append(("quickz" if plane == "xy" else "quicky", json.dumps(offset.t)))
        t = self.feat(self.G, "WorkPlane", "wp", props)
        return t, f'{self.G}.feature("{t}").geom()'

    def rotate_move(self, t: str, s: dict) -> str:
        if "rotate" in s:
            r = s["rotate"]
            o = r.get("origin") or [_Q(0)] * 3
            t2 = self.feat(self.G, "Rotate", "rot", [("rot", json.dumps(r["angle_deg"].t)), ("axistype", '"cartesian"'),
                                                      ("ax3", self.sarr(r["axis"])), ("pos", self.sarr(o))])
            self.w(f'{self.G}.feature("{t2}").selection("input").set("{t}");')
            t = t2
        if "translate" in s:
            d = s["translate"]
            t2 = self.feat(self.G, "Move", "mov", [("displx", json.dumps(d[0].t)), ("disply", json.dumps(d[1].t)),
                                                    ("displz", json.dumps(d[2].t))])
            self.w(f'{self.G}.feature("{t2}").selection("input").set("{t}");')
            t = t2
        return t

    def solid(self, s: dict) -> str:
        keys = [k for k in s if k not in ("rotate", "translate")]
        k = keys[0]
        b = s[k]
        G = self.G
        if k == "box":
            mn, mx = b["min"], b["max"]
            t = self.feat(G, "Block", "blk", [("base", '"corner"'), ("pos", self.sarr(mn)),
                                               ("size", self.sarr([mx[i] - mn[i] for i in range(3)]))])
        elif k in ("cylinder", "cone"):
            ax = b["axis"]
            h = math.sqrt(sum(a.v ** 2 for a in ax))
            props = [("pos", self.sarr(b["base"])), ("axistype", '"cartesian"'), ("ax3", self.sarr(ax)),
                     ("h", json.dumps(f"sqrt(({ax[0].t})^2+({ax[1].t})^2+({ax[2].t})^2)" if not all(a._lit() for a in ax) else _num(h)))]
            if k == "cylinder":
                t = self.feat(G, "Cylinder", "cyl", [("r", json.dumps(b["radius"].t))] + props)
            else:
                t = self.feat(G, "Cone", "cone", [("r", json.dumps(b["r1"].t)), ("specifytop", '"radius"'),
                                                   ("rtop", json.dumps(b["r2"].t))] + props)
        elif k == "sphere":
            t = self.feat(G, "Sphere", "sph", [("r", json.dumps(b["radius"].t)), ("pos", self.sarr(b["center"]))])
        elif k == "torus":
            t = self.feat(G, "Torus", "tor", [("rmaj", json.dumps(b["major"].t)), ("rmin", json.dumps(b["minor"].t)),
                                               ("pos", self.sarr(b["center"])), ("axistype", '"cartesian"'),
                                               ("ax3", self.sarr(b["axis"]))])
        elif k == "revolve":
            wp, seq = self.workplane("xz")
            self.shape(seq, b["profile"])
            a0, a1 = b["angles"]
            # work plane xz: local (x, y) = global (x, z), so the profile lies in the half-plane y = 0, x >= 0;
            # revolution about the global +z axis (axistype 3d), angles in degrees from that half-plane,
            # right-handed about +z as OpenCASCADE in axf-mesher (check the sense once in the GUI)
            t = self.feat(G, "Revolve", "rev", [("angtype", '"specang"'), ("angle1", json.dumps(a0.t)),
                                                 ("angle2", json.dumps(a1.t)), ("axistype", '"3d"'),
                                                 ("pos3", 'new String[]{"0", "0", "0"}'),
                                                 ("axis3", 'new String[]{"0", "0", "1"}')])
            self.w(f'{G}.feature("{t}").selection("input").set("{wp}");')
            c, a = b["center"], b["axis"]
            av = [x.v for x in a]
            n = math.sqrt(sum(x * x for x in av))
            ax_, ay_, az_ = av[0] / n, av[1] / n, av[2] / n
            if abs(ax_) + abs(ay_) > 1e-12 or az_ < 0:
                rx, ry = -ay_, ax_
                ang = math.degrees(math.acos(max(-1.0, min(1.0, az_))))
                if math.hypot(rx, ry) < 1e-12:
                    rx, ry, ang = 1.0, 0.0, 180.0
                t2 = self.feat(G, "Rotate", "rot", [("rot", json.dumps(_num(ang))), ("axistype", '"cartesian"'),
                                                     ("ax3", self.sarr([rx, ry, 0.0])), ("pos", 'new String[]{"0", "0", "0"}')])
                self.w(f'{G}.feature("{t2}").selection("input").set("{t}");')
                t = t2
            if any(x.v != 0 for x in c):
                t2 = self.feat(G, "Move", "mov", [("displx", json.dumps(c[0].t)), ("disply", json.dumps(c[1].t)),
                                                   ("displz", json.dumps(c[2].t))])
                self.w(f'{G}.feature("{t2}").selection("input").set("{t}");')
                t = t2
        elif k == "extrude":
            z0, z1 = b["z"]
            wp, seq = self.workplane("xy", z0)
            rows = ", ".join("{" + json.dumps(p[0].t) + ", " + json.dumps(p[1].t) + "}" for p in b["polygon"])
            self.feat(seq, "Polygon", "pol", [("type", '"solid"'), ("source", '"table"'), ("table", f"new String[][]{{{rows}}}")])
            t = self.feat(G, "Extrude", "ext", [("distance", self.sarr([z1 - z0]))])
            self.w(f'{G}.feature("{t}").selection("input").set("{wp}");')
        elif k == "step":
            self.approx.append(f"STEP solid '{b['file']}' is imported with the Import feature (needs the CAD Import Module)")
            t = self.feat(G, "Import", "imp", [("type", '"cad"'), ("filename", json.dumps(str(b["file"])))])
            sc = b.get("scale")
            if sc is not None and sc.v != 1:
                t2 = self.feat(G, "Scale", "sca", [("factor", json.dumps(sc.t)), ("pos", 'new String[]{"0", "0", "0"}')])
                self.w(f'{G}.feature("{t2}").selection("input").set("{t}");')
                t = t2
        elif k in ("union", "difference", "intersection"):
            tags = [self.solid(c) for c in b]
            t = self.boolean(G, k, tags[:1], tags[1:])
        else:
            raise ValueError(f"unknown solid '{k}'")
        return self.rotate_move(t, s)


def _contour_rz(bb: dict) -> np.ndarray:
    """(r, z) points of a resolved ``contour`` shape (its vertices and 65 points per elliptical arc)."""
    pts = [tuple(bb["start"])]
    for sg in bb["segments"]:
        if "line" in sg:
            pts.append(tuple(sg["line"]))
        else:
            e = sg["ellipse"]
            t = np.linspace(e["t"][0], e["t"][1], 65)
            pts += list(zip(e["center"][0] + e["axes"][0] * np.cos(t), e["center"][1] + e["axes"][1] * np.sin(t)))
    return np.array(pts, float)


def _bbox_solid(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rough axis-aligned bounding box of a resolved solid (plain numbers), for the default domain radius."""
    k = [x for x in s if x not in ("rotate", "translate")][0]
    b = s[k]
    if k == "box":
        lo, hi = np.array(b["min"], float), np.array(b["max"], float)
    elif k in ("cylinder", "cone"):
        p, a = np.array(b["base"], float), np.array(b["axis"], float)
        r = max(b.get("radius", 0), b.get("r1", 0), b.get("r2", 0))
        lo, hi = np.minimum(p, p + a) - r, np.maximum(p, p + a) + r
    elif k == "sphere":
        c = np.array(b["center"], float); lo, hi = c - b["radius"], c + b["radius"]
    elif k == "torus":
        c = np.array(b["center"], float); r = b["major"] + b["minor"]; lo, hi = c - r, c + r
    elif k == "revolve":
        def sb(sh):
            (kk, bb), = sh.items()
            if kk in ("rect", "frame"):
                return bb["r"][1], bb["z"][0], bb["z"][1]
            if kk == "polygon":
                return max(p[0] for p in bb["points"]), min(p[1] for p in bb["points"]), max(p[1] for p in bb["points"])
            if kk == "contour":
                P = _contour_rz(bb)
                return float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max())
            if kk == "disk":
                return bb["center"][0] + bb["radius"], bb["center"][1] - bb["radius"], bb["center"][1] + bb["radius"]
            rs = [sb(c) for c in bb]
            return max(x[0] for x in rs), min(x[1] for x in rs), max(x[2] for x in rs)
        R, z0, z1 = sb(b["profile"])
        c = np.array(b["center"], float)
        ext = max(R, abs(z0), abs(z1))
        lo, hi = c - ext, c + ext
    elif k == "extrude":
        P = np.array(b["polygon"], float)
        lo = np.array([P[:, 0].min(), P[:, 1].min(), min(b["z"])])
        hi = np.array([P[:, 0].max(), P[:, 1].max(), max(b["z"])])
    elif k in ("union", "difference", "intersection"):
        bs = [_bbox_solid(c) for c in (b if k == "union" else b[:1])]
        lo = np.min([x[0] for x in bs], axis=0); hi = np.max([x[1] for x in bs], axis=0)
    else:
        lo, hi = np.zeros(3), np.zeros(3)
    if "rotate" in s or "translate" in s:  # conservative: a sphere about the origin through the corners
        t = np.array(s.get("translate", [0, 0, 0]), float)
        o = np.array((s.get("rotate") or {}).get("origin", [0, 0, 0]), float)
        r = max(np.linalg.norm(lo - o), np.linalg.norm(hi - o))
        lo, hi = o - r + t, o + r + t
    return lo, hi


def _java_physics_selections(J: "_Java", G: str, spec: dict, final_sels: dict, adj_of: dict, region_box: dict,
                             sym_tags: list, dom_sel: str | None, Rb: float) -> dict:
    """Geometry selections that the emw physics is attached to (parametric Java route).

    Geometry-sequence selection features are referenced from the physics as ``geom1_<tag>``.
    """
    from .comsol_physics import port_face

    kinds = spec["region_kind"]
    approx: list[str] = []

    def union(label: str, dim: int, inputs: list[str]) -> str:
        t = J.tag("unisel")
        J.w(f'{G}.create("{t}", "UnionSelection");')
        J.w(f'{G}.feature("{t}").label({json.dumps(label)});')
        J.w(f'{G}.feature("{t}").set("entitydim", {dim});')
        J.w(f'{G}.feature("{t}").set("input", new String[]{{{", ".join(json.dumps(x) for x in inputs)}}});')
        return t

    def adjacent(label: str, inputs: list[str], interior: bool = True) -> str:
        t = J.tag("adjsel")
        J.w(f'{G}.create("{t}", "AdjacentSelection");')
        J.w(f'{G}.feature("{t}").label({json.dumps(label)});')
        J.w(f'{G}.feature("{t}").set("entitydim", 3);')
        J.w(f'{G}.feature("{t}").set("input", new String[]{{{", ".join(json.dumps(x) for x in inputs)}}});')
        J.w(f'{G}.feature("{t}").set("outputdim", 2);')
        J.w(f'{G}.feature("{t}").set("interior", {"true" if interior else "false"});')
        J.w(f'{G}.feature("{t}").set("exterior", true);')
        return t

    def difference(label: str, add: list[str], sub: list[str]) -> str:
        t = J.tag("difsel")
        J.w(f'{G}.create("{t}", "DifferenceSelection");')
        J.w(f'{G}.feature("{t}").label({json.dumps(label)});')
        J.w(f'{G}.feature("{t}").set("entitydim", 2);')
        J.w(f'{G}.feature("{t}").set("add", new String[]{{{", ".join(json.dumps(x) for x in add)}}});')
        if sub:
            J.w(f'{G}.feature("{t}").set("subtract", new String[]{{{", ".join(json.dumps(x) for x in sub)}}});')
        return t

    regs = [n for n in final_sels if n not in ("air", "infinity_shell")]
    pec = [n for n in regs if kinds.get(n) in ("superconductor", "conductor")]
    emw_in = [final_sels[n] for n in regs if n not in pec] + ([final_sels["air"]] if "air" in final_sels else [])
    emw = union("emw_domains", 3, emw_in)
    refs: dict = {"emw": "geom1_" + emw, "pec": [], "pmc": [], "scattering": None, "ports": [], "materials": []}
    pec_walls = None
    if pec:
        pd = union("pec_domains", 3, [final_sels[n] for n in pec])
        pec_walls = adjacent("pec_walls", [pd])
        refs["pec"].append("geom1_" + pec_walls)
    for p, t in sym_tags:
        (refs["pmc"] if p.get("type") == "pmc" else refs["pec"]).append("geom1_" + t)
    if spec["boundary"] == "radiation" and dom_sel:
        ext = adjacent("outer_boundary (exterior)", [dom_sel], interior=False)
        refs["scattering"] = "geom1_" + (difference("outer", [ext], [t for _, t in sym_tags]) if sym_tags else ext)
        approx.append("fullwave.boundary radiation: a first-order Scattering boundary condition on the domain sphere "
                      "(axionFEM: exact outgoing-wave exterior map); use boundary 'pec' for a like-for-like comparison")
    # materials: one per axionFEM material of the emw regions; air is vacuum
    by_mat: dict[str, list[str]] = {}
    for n in regs:
        if n not in pec:
            by_mat.setdefault(spec["region_material"].get(n, "vacuum"), []).append(final_sels[n])
    if "air" in final_sels:
        by_mat.setdefault("air", []).append(final_sels["air"])
    for m, sels in by_mat.items():
        refs["materials"].append((m, "geom1_" + (sels[0] if len(sels) == 1 else union("material " + m, 3, sels))))
    # ports
    port_sel = []
    for port in spec["ports"]:
        nm = port["region"]
        if nm not in final_sels:
            raise ValueError(f"port region '{nm}' is not in the exported model (clipped away by the symmetry planes?)")
        lo, hi, is_box = region_box.get(nm, (None, None, False))
        face = port_face(lo, hi, port["toward"]) if is_box else None
        if face is not None:
            ai = "xyz".index(face["normal_axis"])
            tol = 1e-6 * max(Rb, 1.0)
            t = J.tag("boxsel")
            J.w(f'{G}.create("{t}", "BoxSelection");')
            J.w(f'{G}.feature("{t}").label({json.dumps("port_" + port["name"])});')
            J.w(f'{G}.feature("{t}").set("entitydim", 2);')
            J.w(f'{G}.feature("{t}").set("condition", "inside");')
            for i, ax in enumerate("xyz"):
                a, b = (face["coord"] - tol, face["coord"] + tol) if i == ai else (lo[i] - tol, hi[i] + tol)
                J.w(f'{G}.feature("{t}").set("{ax}min", {json.dumps(_num(a))});')
                J.w(f'{G}.feature("{t}").set("{ax}max", {json.dumps(_num(b))});')
            port_sel.append(f"face {face['normal_axis']} = {face['coord']:.9g} of the gap box (current along "
                            f"{face['toward_axis']})")
        else:
            t = difference("port_" + port["name"], [adj_of[nm]], ([pec_walls] if pec_walls else []) + [x for _, x in sym_tags])
            port_sel.append("all gap boundaries except its PEC terminals and symmetry planes (UNVERIFIED: non-planar "
                            "lumped port)")
            approx.append(f"port '{port['name']}': region '{nm}' is not an axis-aligned box with a 'toward' direction; "
                          "the LumpedPort is put on all its non-PEC walls (check in COMSOL, or select one face)")
        refs["ports"].append("geom1_" + t)
    approx.append("ports: axionFEM feeds the current through the gap volume (Laplace profile); the COMSOL LumpedPort "
                  "carries it on one face of the gap, a modelling difference of order (gap size / loop size)")
    return {"refs": refs, "approx": approx, "port_selection": port_sel}


def _strip_q(x):
    if isinstance(x, _Q):
        return x.v
    if isinstance(x, dict):
        return {k: _strip_q(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_q(v) for v in x]
    return x


def write_java(resolved: dict, path: str | Path, *, raw: dict | None = None, class_name: str | None = None,
               mph_name: str | None = None, mesh_info: dict | None = None, build_domain: bool = True,
               physics: dict | None = None) -> dict:
    """Write a COMSOL Java API model file that rebuilds the geometry of a design.

    ``resolved``: the output of ``_core.resolve`` (``axf resolve``) or an axf-mesher input
    (``mesh*_in.json``: regions with resolved shapes/solids).  ``raw``: the design document, so that
    numbers given as parameter expressions stay expressions in COMSOL.  ``mesh_info``: the
    mesher report, which supplies the domain radius actually used.  ``physics``: a
    :func:`axionfem.comsol_physics.physics_spec` (3D only) to add the emw model, ports, source and
    studies.  Returns a summary (class name, selections, approximations).
    """
    path = Path(path)
    is3d = resolved.get("symmetry") == "3d" or resolved.get("dim") == 3 or any("solid" in r for r in resolved.get("regions", []))
    cls = class_name or path.stem
    if not re.match(r"^[A-Za-z_]\w*$", cls):
        raise ValueError(f"'{cls}' is not a valid Java class name; pass class_name= or rename the file")
    params = {k: float(v) for k, v in (resolved.get("parameters") or {}).items()}
    raw = raw or {}
    raw_regions = {r.get("name"): r for r in raw.get("regions", [])}
    raw_curves = {r.get("name"): r for r in raw.get("curves", [])}
    J = _Java()
    G = J.G
    name = resolved.get("name", raw.get("name", cls))
    mph = mph_name or (re.sub(r"[^\w.-]", "_", name) + ".mph")
    head = []
    # ---- parameters
    praw = raw.get("parameters") or {}
    plines = []
    for k, v in params.items():
        expr = praw.get(k, v)
        try:
            _, t = comsol_expression(expr, params) if isinstance(expr, str) else (v, _num(v))
        except (ValueError, ArithmeticError, TypeError, RecursionError):
            t = _num(v)     # not translatable: write the number the core resolved
        plines.append(f'model.param().set({json.dumps(comsol_param_name(k))}, {json.dumps(t)}, {json.dumps("axionFEM parameter " + k + " = " + _num(v))});')
    # ---- component / geometry
    J.w('model.component().create("comp1", true);')
    J.w(f'model.component("comp1").geom().create("geom1", {3 if is3d else 2});')
    if not is3d:
        J.w(f'{G}.axisymmetric(true);')
    J.w(f'{G}.lengthUnit("m");')
    regions = resolved.get("regions", [])
    region_box: dict[str, tuple] = {}
    sym_tags: list[tuple[dict, str]] = []
    adj_of: dict[str, str] = {}
    sel_of_region: dict[str, str] = {}
    region_kind = {}
    mesh = resolved.get("mesh") or {}
    sym = mesh.get("symmetry") or resolved.get("symmetry_planes") or (resolved.get("symmetry") if isinstance(resolved.get("symmetry"), list) else [])

    Rb = float(mesh.get("domain_radius") or 0) or float((mesh_info or {}).get("domain_radius") or 0)
    clip_lo = clip_hi = None
    if sym and is3d:
        Lc = 8 * max(Rb or 1.0, 1.0) + 1e3
        clip_lo, clip_hi = [-Lc] * 3, [Lc] * 3
        for p in sym:
            ai = "xyz".index(p["axis"])
            if p.get("side", 1) > 0:
                clip_lo[ai] = 0.0
            else:
                clip_hi[ai] = 0.0
        J.approx.append("symmetry: every object is intersected with the half-spaces " +
                        ", ".join(f"{p['axis']}{'>=' if p.get('side', 1) > 0 else '<='}0" for p in sym) +
                        " (one clipping Block and Intersection per object); the PEC/PMC conditions on the planes are left to the user "
                        "(BoxSelections 'sym_<name>_<type>')")

    def clip(t: str) -> str:
        if clip_lo is None:
            return t
        blk = J.feat(G, "Block", "blk", [("base", '"corner"'), ("pos", J.sarr(clip_lo)),
                                          ("size", J.sarr([clip_hi[i] - clip_lo[i] for i in range(3)]))])
        return J.boolean(G, "intersection", [t], [blk])

    def csel(label: str) -> str:
        t = J.tag("csel")
        J.w(f'{G}.selection().create("{t}", "CumulativeSelection");')
        J.w(f'{G}.selection("{t}").label({json.dumps(label + " (as drawn)")});')
        return t

    # ---- outer domain (air), drawn first
    dom_sel = None
    shell_sel = None
    if build_domain:
        if is3d:
            sol = [r["solid"] for r in regions]
            if not Rb:
                if sol:
                    los, his = zip(*[_bbox_solid(_strip_q(s)) for s in sol])
                    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
                    Rdev = max(np.linalg.norm([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))
                else:
                    Rdev = 1.0
                Rb = 1.6 * Rdev
                J.approx.append(f"domain radius not given: R_domain = 1.6 x (estimated device radius {Rdev:.6g} m), "
                                "as axf-mesher with the infinity shell; set mesh.domain_radius for the exact value")
            plines.append(f'model.param().set("R_domain", {json.dumps(_num(Rb))}, "radius of the air sphere (axionFEM mesh.domain_radius)");')
            dom_sel = csel("domain")
            if mesh.get("domain_solid"):
                t = clip(J.solid(_pair(None, mesh["domain_solid"], params)))
                J.w(f'{G}.feature("{t}").set("contributeto", "{dom_sel}");')
            else:
                t = clip(J.feat(G, "Sphere", "sph", [("r", '"R_domain"'), ("pos", 'new String[]{"0", "0", "0"}')], label="domain"))
                J.w(f'{G}.feature("{t}").set("contributeto", "{dom_sel}");')
                if not sym and not (resolved.get("fullwave") or {}).get("rf_only") and not physics:
                    plines.append('model.param().set("R_shell", "1.25*R_domain", "outer radius of the axionFEM infinity shell");')
                    shell_sel = csel("infinity_shell")
                    t = J.feat(G, "Sphere", "sph", [("r", '"R_shell"'), ("pos", 'new String[]{"0", "0", "0"}')], label="infinity_shell")
                    J.w(f'{G}.feature("{t}").set("contributeto", "{shell_sel}");')
        else:
            if not Rb:
                rmax = 0.0
                for r in regions:
                    def sb(sh):
                        (kk, bb), = sh.items()
                        if kk in ("rect", "frame"):
                            return max(abs(bb["r"][1]), abs(bb["z"][0]), abs(bb["z"][1]))
                        if kk == "polygon":
                            return max(max(abs(p[0]), abs(p[1])) for p in bb["points"])
                        if kk == "contour":
                            return float(np.abs(_contour_rz(bb)).max())
                        if kk == "disk":
                            return math.hypot(bb["center"][0], bb["center"][1]) + bb["radius"]
                        return max(sb(c) for c in bb)
                    rmax = max(rmax, sb(r["shape"]) * math.sqrt(2))
                Rb = 2.5 * (rmax or 1.0)
                J.approx.append(f"domain radius not given: 2.5 x estimated device radius ({Rb:.6g} m); "
                                "set mesh.domain_radius for the exact value")
            zc = float(mesh.get("domain_center_z") or 0)
            plines.append(f'model.param().set("R_domain", {json.dumps(_num(Rb))}, "radius of the outer half-disk (axionFEM mesh.domain_radius)");')
            plines.append(f'model.param().set("z_domain", {json.dumps(_num(zc))}, "z of the centre of the half-disk (mesh.domain_center_z)");')
            dom_sel = csel("domain")
            t = J.feat(G, "Circle", "c", [("r", '"R_domain"'), ("pos", 'new String[]{"0", "z_domain"}'),
                                           ("angle", '"180"'), ("rot", '"-90"')], label="domain")
            J.w(f'{G}.feature("{t}").set("contributeto", "{dom_sel}");')

    # ---- regions, in painting order
    for r in regions:
        nm = r["name"]
        rr = raw_regions.get(nm, {})
        if is3d:
            if clip_lo is not None:
                lo_, hi_ = _bbox_solid(r["solid"])
                if any(hi_[i] <= clip_lo[i] or lo_[i] >= clip_hi[i] for i in range(3)):
                    J.approx.append(f"region '{nm}' lies outside the symmetry-clipped model and is omitted")
                    continue
            s = _pair(rr.get("solid"), r["solid"], params)
            t = clip(J.solid(s))
            lo_, hi_ = _bbox_solid(r["solid"])
            if clip_lo is not None:
                lo_, hi_ = np.maximum(lo_, clip_lo), np.minimum(hi_, clip_hi)
            region_box[nm] = (lo_, hi_, set(r["solid"]) == {"box"})
        else:
            s = _pair(rr.get("shape"), r["shape"], params)
            t = J.shape(G, s)
        J.w(f'{G}.feature("{t}").label({json.dumps(nm)});')
        sel_of_region[nm] = csel(nm)
        J.w(f'{G}.feature("{t}").set("contributeto", "{sel_of_region[nm]}");')
        region_kind[nm] = r.get("kind")
    curve_sels = {}
    for c in resolved.get("curves", []):
        nm = c["name"]
        s = _pair(raw_curves.get(nm, {}).get("path"), c["path"], params)
        tags = J.path(G, s)
        curve_sels[nm] = csel(nm)
        for t in tags:
            J.w(f'{G}.feature("{t}").label({json.dumps(nm)});')
            J.w(f'{G}.feature("{t}").set("contributeto", "{curve_sels[nm]}");')

    # ---- symmetry planes (quarter / half models): boundary selections on the planes
    if sym and is3d:
        Lc = 4 * max(Rb, 1.0)
        for p in sym:
            a = p["axis"]
            t = J.tag("boxsel")
            J.w(f'{G}.create("{t}", "BoxSelection");')
            J.w(f'{G}.feature("{t}").label({json.dumps("sym_" + str(p.get("name", a)) + "_" + str(p.get("type", "")))});')
            sym_tags.append((p, t))
            J.w(f'{G}.feature("{t}").set("entitydim", 2);')
            J.w(f'{G}.feature("{t}").set("condition", "inside");')
            for ax in "xyz":
                lo_, hi_ = (-Lc, Lc) if ax != a else (-1e-6 * Lc, 1e-6 * Lc)
                J.w(f'{G}.feature("{t}").set("{ax}min", {json.dumps(_num(lo_))});')
                J.w(f'{G}.feature("{t}").set("{ax}max", {json.dumps(_num(hi_))});')

    # ---- finalize and region selections ("later regions win")
    J.w(f'{G}.feature("fin").set("action", "union");')
    final_sels = {}
    names = [r["name"] for r in regions if r["name"] in sel_of_region]
    D = 3 if is3d else 2
    # later[i]: union of the regions drawn after region i (a chain of Union selections, linear in size)
    later: dict[int, str | None] = {len(names) - 1: None}
    for i in range(len(names) - 2, -1, -1):
        nxt, rest = sel_of_region[names[i + 1]], later[i + 1]
        if rest is None:
            later[i] = nxt
        else:
            t = J.tag("unisel")
            J.w(f'{G}.create("{t}", "UnionSelection");')
            J.w(f'{G}.feature("{t}").label({json.dumps("drawn after " + names[i])});')
            J.w(f'{G}.feature("{t}").set("entitydim", {D});')
            J.w(f'{G}.feature("{t}").set("input", new String[]{{"{nxt}", "{rest}"}});')
            later[i] = t
    for i, nm in enumerate(names):
        t = J.tag("difsel")
        J.w(f'{G}.create("{t}", "DifferenceSelection");')
        J.w(f'{G}.feature("{t}").label({json.dumps(nm)});')
        J.w(f'{G}.feature("{t}").set("entitydim", {D});')
        J.w(f'{G}.feature("{t}").set("add", new String[]{{"{sel_of_region[nm]}"}});')
        if later[i]:
            J.w(f'{G}.feature("{t}").set("subtract", new String[]{{"{later[i]}"}});')
        final_sels[nm] = t
    if dom_sel:
        all_regions = None
        if names:
            if later[0] is None:
                all_regions = sel_of_region[names[0]]
            else:
                all_regions = J.tag("unisel")
                J.w(f'{G}.create("{all_regions}", "UnionSelection");')
                J.w(f'{G}.feature("{all_regions}").label("all regions (as drawn)");')
                J.w(f'{G}.feature("{all_regions}").set("entitydim", {D});')
                J.w(f'{G}.feature("{all_regions}").set("input", new String[]{{"{sel_of_region[names[0]]}", "{later[0]}"}});')
        t = J.tag("difsel")
        J.w(f'{G}.create("{t}", "DifferenceSelection");')
        J.w(f'{G}.feature("{t}").label("air");')
        J.w(f'{G}.feature("{t}").set("entitydim", {D});')
        J.w(f'{G}.feature("{t}").set("add", new String[]{{"{dom_sel}"}});')
        if all_regions:
            J.w(f'{G}.feature("{t}").set("subtract", new String[]{{"{all_regions}"}});')
        final_sels["air"] = t
        if shell_sel:
            t = J.tag("difsel")
            J.w(f'{G}.create("{t}", "DifferenceSelection");')
            J.w(f'{G}.feature("{t}").label("infinity_shell");')
            J.w(f'{G}.feature("{t}").set("entitydim", 3);')
            J.w(f'{G}.feature("{t}").set("add", new String[]{{"{shell_sel}"}});')
            J.w(f'{G}.feature("{t}").set("subtract", new String[]{{"{dom_sel}"}});')
            final_sels["infinity_shell"] = t
    for nm, t in list(final_sels.items()):
        a = J.tag("adjsel")
        adj_of[nm] = a
        J.w(f'{G}.create("{a}", "AdjacentSelection");')
        J.w(f'{G}.feature("{a}").label({json.dumps(nm + "_boundary")});')
        J.w(f'{G}.feature("{a}").set("entitydim", {D});')
        J.w(f'{G}.feature("{a}").set("input", new String[]{{"{t}"}});')
        J.w(f'{G}.feature("{a}").set("outputdim", {D - 1});')
        J.w(f'{G}.feature("{a}").set("interior", true);')
        J.w(f'{G}.feature("{a}").set("exterior", true);')
    for kind in ("superconductor", "conductor"):
        members = [final_sels[n] for n in names if region_kind.get(n) == kind]
        if members:
            t = J.tag("unisel")
            J.w(f'{G}.create("{t}", "UnionSelection");')
            J.w(f'{G}.feature("{t}").label({json.dumps("all_" + kind + "s")});')
            J.w(f'{G}.feature("{t}").set("entitydim", {D});')
            J.w(f'{G}.feature("{t}").set("input", new String[]{{{", ".join(json.dumps(m) for m in members)}}});')
    phys = None
    if physics is not None:
        if not is3d:
            raise ValueError("physics export needs a 3D design")
        phys = _java_physics_selections(J, G, physics, final_sels, adj_of, region_box, sym_tags, dom_sel, Rb)
    J.w(f"{G}.run();")
    if phys is not None:
        from . import comsol_physics as cp

        files = {k: f'new java.io.File(System.getProperty("user.dir"), {json.dumps(cls + "_" + k + ".txt")}).getAbsolutePath()'
                 for k in ("impedance", "voltage")}
        code = cp.physics_java(physics, phys["refs"], imp_file=files["impedance"], volt_file=files["voltage"])
        for ln in code["setup"]:
            J.w(ln)
        J.approx.extend(phys["approx"])

    pick = resolved.get("pickup") or raw.get("pickup") or {}
    notes = [
        f"axionFEM design: {name}  ({'3D' if is3d else '2D axisymmetric (r, z)'}; lengths in metres)",
        "Region selections reproduce axionFEM's painting order (later regions win) with Difference selections;",
        "'<region>_boundary' are the adjacent boundaries; 'all_superconductors' / 'all_conductors' group regions by kind.",
    ]
    if pick:
        notes.append("pickup: " + json.dumps({k: v for k, v in pick.items() if v}))
    if is3d and mesh.get("remove_pec"):
        notes.append("mesh.remove_pec: axionFEM removes conductor volumes and imposes PEC on their walls; here they are "
                     "ordinary domains (exclude them from the physics and use '<region>_boundary' for the PEC condition).")
    notes += [f"approximation: {a}" for a in J.approx]
    if phys is not None:
        from . import comsol_physics as cp

        notes += [""] + cp.physics_notes(physics)
    build = "Build and run (COMSOL installation, no extra modules for the geometry):"
    if phys is not None:
        build = ("Build and run (needs the RF Module for emw/LumpedPort; main() solves the imp and volt runs and writes "
                 f"{cls}_impedance.txt / {cls}_voltage.txt to the working directory):")
    src = _java_class(cls, notes, plines, J.lines, mph, build=build,
                      solves=(code["imp"], code["volt"]) if phys is not None else None)
    path.write_text(src)
    out = {"file": str(path), "class": cls, "mph": mph, "selections": sorted(final_sels),
           "parameters": list(params), "approximations": J.approx, "dimension": 3 if is3d else 2}
    if phys is not None:
        out["physics"] = _physics_summary(physics, phys["port_selection"], cls)
    return out


def _physics_summary(spec: dict, port_sel: list, cls: str) -> dict:
    return {"physics": spec["physics"], "frequencies_Hz": spec["frequencies_Hz"], "boundary": spec["boundary"],
            "ports": [dict(p, comsol_port=k + 1, active=(k == 0), selection=port_sel[k]) for k, p in enumerate(spec["ports"])],
            "source": {k: v for k, v in spec["source"].items() if k != "B_T"},
            "outputs": {"impedance": cls + "_impedance.txt", "voltage": cls + "_voltage.txt" if spec["source"].get("exported") else None},
            "warnings": spec["warnings"]}


def _java_class(cls: str, notes: list[str], plines: list[str], lines: list[str], mph: str, *, build: str,
                solves: tuple[list[str], list[str]] | None = None) -> str:
    """Java source of a model class: run() (split into run1, run2, ... to stay below Java's 64 kB method limit,
    as COMSOL's own Java export does) and main(), which also runs the imp/volt solves when given."""
    parts, cur, cost = [], [], 0
    for ln in lines:  # rough bytecode estimate: a call chain plus one constant load per literal
        c = 40 + 8 * (ln.count(",") + ln.count('"') // 2)
        if cur and cost + c > 30000:
            parts.append(cur)
            cur, cost = [], 0
        cur.append(ln)
        cost += c
    parts.append(cur)
    calls = "\n".join(f"    model = run{k + 1}(model);" for k in range(len(parts)))
    methods = "\n".join(f"""
  public static Model run{k + 1}(Model model) {{
{chr(10).join(part)}
    return model;
  }}""" for k, part in enumerate(parts))
    solve_methods, solve_calls = "", ""
    if solves is not None:
        imp, volt = solves
        solve_methods = f"""

  /** 'imp' run: port 1 driven, axion source off -> Zport_1 [freq_MHz, re, im]. */
  public static Model runImpedance(Model model) {{
{chr(10).join('    ' + x for x in imp)}
    return model;
  }}

  /** 'volt' run: axion source on, port 1 not driven (terminated by its Z_ref) -> Vport_1 [freq_MHz, re, im]. */
  public static Model runVoltage(Model model) {{
{chr(10).join('    ' + x for x in volt) if volt else '    // the axion source was not exported (see the header): no voltage run'}
    return model;
  }}"""
        solve_calls = "    model = runImpedance(model);\n    model = runVoltage(model);\n"
    return f"""/*
 * {cls}.java -- COMSOL Multiphysics model generated by axionFEM (python/axionfem/comsol.py).
 *
{chr(10).join((' * ' + n).rstrip() for n in notes)}
 *
 * {build}
 *   comsolcompile {cls}.java                  # -> {cls}.class
 *   comsol batch -inputfile {cls}.class       # -> {mph}
 * or open {cls}.class with File > Open in the COMSOL Desktop.
 * See docs/comsol_export.md.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;

public class {cls} {{

  public static Model run() {{
    Model model = ModelUtil.create("Model");
    model.label({json.dumps(mph)});
{chr(10).join('    ' + p for p in plines)}
{calls}
    return model;
  }}
{methods}{solve_methods}

  public static void main(String[] args) {{
    Model model = run();
{solve_calls}    try {{
      model.save({json.dumps(mph)});
    }} catch (Exception e) {{
      throw new RuntimeException(e);
    }}
  }}
}}
"""


# =====================================================================================
# high-level API
# =====================================================================================

def _as_doc(source) -> dict | None:
    from .design import Design

    if isinstance(source, Design):
        return source.doc
    if isinstance(source, dict):
        return source
    p = Path(source)
    if p.is_file() and p.suffix == ".json":
        doc = json.loads(p.read_text())
        if "regions" in doc and ("schema" in doc or "symmetry" in doc or "parameters" in doc or "name" in doc):
            if "attr" not in (doc["regions"][0] if doc["regions"] else {}):
                return doc
    return None


def _find_mesh(p: Path) -> Path | None:
    if p.is_file() and p.suffix == ".msh":
        return p
    if p.is_dir():
        for n in ("mesh3d.msh", "mesh.msh"):
            if (p / n).is_file():
                return p / n
    return None


def _core():
    from . import _core as c

    return c


def export_comsol(source, out: str | Path, *, format: str = "mphtxt", order: int = 1,
                  workdir: str | Path | None = None, mesher: str = "", verbose: int = 0,
                  class_name: str | None = None, feature_angle: float | None = None,
                  physics: str = "none") -> dict:
    """Export an axionFEM geometry or mesh to COMSOL.

    ``source``: a Design, a design dict, a design ``.json``, an axf-mesher ``.msh`` or a run directory
    (``axf solve -o DIR`` / ``axf mesh -o DIR``: holds ``mesh.msh`` or ``mesh3d.msh`` and ``design.json``).

    ``format``:
      * ``"mphtxt"`` -- COMSOL mesh with named selections (designs are meshed first with
        ``axf mesh`` into ``workdir``, default ``<out>_mesh/``); ``order=2`` keeps second-order nodes.
      * ``"java"`` -- COMSOL Java API model that rebuilds the geometry (needs the compiled core
        for designs, or a run directory with ``mesh*_in.json``).
      * ``"step"`` -- 3D designs: the OCC solids as STEP via axf-mesher, plus ``<out>.solids.json``.

    ``physics``: ``"none"`` (default: geometry and selections only) or ``"fullwave"`` (3D designs with a port
    pickup): the Java model, or for ``mphtxt`` the ``Import_<name>.java`` program, also gets the Electromagnetic
    Waves physics, materials, lumped ports, the axion source, a Frequency study and the Z/V exports
    (:mod:`axionfem.comsol_physics`).  The design is needed: a design, a design
    ``.json`` or a run directory with ``design.json``.

    Returns a summary dict (files written, entity counts, selections, approximations).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = format.lower()
    physics = (physics or "none").lower()
    from .comsol_physics import PHYSICS_CHOICES

    if physics not in PHYSICS_CHOICES:
        raise ValueError(f"unknown physics '{physics}'; use one of {PHYSICS_CHOICES} (an 'mqs' Magnetic Fields export "
                         "is not implemented)")
    if physics != "none" and fmt == "step":
        raise ValueError("physics is exported with --format java or mphtxt (Import_<name>.java), not step")
    doc = _as_doc(source) if not isinstance(source, (str, Path)) or Path(source).suffix == ".json" else None
    src_path = Path(source) if isinstance(source, (str, Path)) else None
    spec = None
    if physics != "none":
        pdoc = doc
        if pdoc is None and src_path is not None and src_path.is_dir() and (src_path / "design.json").is_file():
            pdoc = json.loads((src_path / "design.json").read_text())
        if pdoc is None:
            raise ValueError("physics export needs the design: give a design, a design .json or a run directory "
                             "that holds design.json")
        try:
            pres = json.loads(_core().resolve(json.dumps(pdoc)))
        except ImportError:
            pres = None
        from .comsol_physics import physics_spec

        spec = physics_spec(pdoc, pres, physics)
    if fmt == "mphtxt":
        msh = _find_mesh(src_path) if src_path is not None else None
        if msh is None:
            if doc is None:
                raise ValueError(f"cannot export {source!r}: expected a design, a design .json, a .msh or a run directory")
            wd = Path(workdir) if workdir else out.with_name(out.stem + "_mesh")
            r = json.loads(_core().mesh(json.dumps(doc), str(wd), int(verbose), mesher))
            msh = Path(r["mesh"])
        if spec is not None and feature_angle is None:
            feature_angle = 40.0   # split box walls into faces, so that a lumped port can sit on one face
        info = gmsh_to_mphtxt(msh, out, order=order, feature_angle=feature_angle)
        cls = "Import_" + re.sub(r"\W", "_", out.stem)
        imp = write_import_java(out, out.with_name(cls + ".java"), sdim=info["sdim"], axisymmetric=info["axisymmetric"],
                                physics=spec, entities=info)
        res = {"format": "mphtxt", "file": str(out), "entities": str(out) + ".entities.json", "source_mesh": str(msh),
               "import_java": imp["file"], "counts": info["counts"],
               "selections": [s["label"] for s in info["selections"]]}
        if spec is not None:
            res.update(physics=imp["physics"], approximations=imp["approximations"], feature_angle=feature_angle)
        return res
    if fmt == "java":
        raw = None
        mesh_info = None
        if doc is None and src_path is not None and src_path.is_dir():
            if (src_path / "design.json").is_file():
                doc = json.loads((src_path / "design.json").read_text())
            for n in ("mesh3d_info.json", "mesh_info.json"):
                if (src_path / n).is_file():
                    mesh_info = json.loads((src_path / n).read_text())
        resolved = None
        if doc is not None:
            raw = doc
            try:
                resolved = json.loads(_core().resolve(json.dumps(doc)))
                if doc.get("fullwave"):
                    resolved["fullwave"] = {"rf_only": bool(doc["fullwave"].get("rf_only"))}
            except ImportError:
                resolved = None
        if resolved is None:
            mi = None
            if src_path is not None and src_path.is_dir():
                for n in ("mesh3d_in.json", "mesh_in.json"):
                    if (src_path / n).is_file():
                        mi = json.loads((src_path / n).read_text())
                        break
            elif src_path is not None and src_path.name.endswith("_in.json"):
                mi = json.loads(src_path.read_text())
            if mi is None:
                raise ValueError("java export needs the compiled axionfem core (to resolve the design) or a run "
                                 "directory / mesh*_in.json with resolved geometry")
            resolved = {"name": (raw or {}).get("name", out.stem), "regions": mi["regions"], "curves": mi.get("curves", []),
                        "symmetry": "3d" if mi.get("dim") == 3 else "axisymmetric",
                        "mesh": {"domain_radius": mi.get("domain_radius", 0), "symmetry": mi.get("symmetry", []),
                                 "domain_center_z": mi.get("domain_center_z", 0),
                                 **({"domain_solid": mi["domain_solid"]} if mi.get("domain_solid") else {})},
                        "parameters": {}}
            if mi.get("dim") == 3 and float(mi.get("infinity_shell", 0) or 0) <= 1:
                resolved["fullwave"] = {"rf_only": True}
        s = write_java(resolved, out, raw=raw, class_name=class_name, mesh_info=mesh_info, physics=spec)
        s["format"] = "java"
        return s
    if fmt == "step":
        mi = None
        if src_path is not None and src_path.is_dir():
            if (src_path / "mesh3d_in.json").is_file():
                mi = json.loads((src_path / "mesh3d_in.json").read_text())
        if mi is None:
            if doc is None:
                raise ValueError("step export needs a 3D design or a 3D run directory (mesh3d_in.json)")
            if doc.get("symmetry") != "3d":
                raise ValueError("step export is for 3D designs (convert with axionfem.convert.to_3d first)")
            # mesher input of the design: write it with `axf mesh` would mesh too; build it from the resolved design
            res = json.loads(_core().resolve(json.dumps(doc)))
            mi = {"dim": 3, "regions": [{"name": r["name"], "attr": i + 2, "solid": r["solid"]} for i, r in enumerate(res["regions"])],
                  "domain_radius": res["mesh"].get("domain_radius", 0),
                  "infinity_shell": 0.0 if (doc.get("fullwave") or {}).get("rf_only") else 1.25,
                  "symmetry": res["mesh"].get("symmetry", []), "outer_attr": 901, "air_attr": 1, "shell_attr": 898}
            if res["mesh"].get("domain_solid"):
                mi["domain_solid"] = res["mesh"]["domain_solid"]
        from . import _paths

        exe = mesher or _paths.mesher_path()
        if not exe:
            raise RuntimeError(_paths.mesher_hint())
        mi = dict(mi)
        mi["step_output"] = str(out.resolve())
        mi["geometry_only"] = True
        mi["info"] = str(out.resolve()) + ".solids.json"
        mi["output"] = str(out.resolve()) + ".unused.msh"
        with tempfile.TemporaryDirectory() as td:
            ip = Path(td) / "step_in.json"
            ip.write_text(json.dumps(mi))
            env = dict(os.environ)
            _paths.prepare_environment()
            env.update({k: v for k, v in os.environ.items() if k == "LD_LIBRARY_PATH"})
            r = subprocess.run([exe, str(ip)], capture_output=True, text=True, env=env)
        info = json.loads(Path(mi["info"]).read_text()) if Path(mi["info"]).is_file() else {}
        if r.returncode != 0 or not info.get("ok"):
            raise RuntimeError(f"axf-mesher STEP export failed: {info.get('error', r.stderr[-2000:])}")
        return {"format": "step", "file": str(out), "solids": mi["info"], "n_solids": len(info["step"]["solids"])}
    raise ValueError(f"unknown format '{format}'; use mphtxt, java or step")


def main(argv: list[str] | None = None) -> int:
    """``axf export-comsol SOURCE -o OUT [--format mphtxt|java|step] [--order 1|2]``."""
    import argparse

    ap = argparse.ArgumentParser(prog="axf export-comsol",
                                 description="Export an axionFEM design, mesh or run directory to COMSOL "
                                             "(docs/comsol_export.md).")
    ap.add_argument("source", help="design .json, axf-mesher .msh, or a run directory (axf solve/mesh -o DIR)")
    ap.add_argument("-o", "--output", help="output file (default: <source stem>.<ext>)")
    ap.add_argument("--format", default="mphtxt", choices=["mphtxt", "java", "step"])
    ap.add_argument("--order", type=int, default=1, choices=[1, 2], help="mphtxt: element order (2 keeps mid-edge nodes)")
    ap.add_argument("--feature-angle", type=float, default=None,
                    help="mphtxt (3D): also split boundaries at facet angles sharper than this (degrees)")
    ap.add_argument("--workdir", help="where a design is meshed (default: <output>_mesh/)")
    ap.add_argument("--mesher", default="", help="path to axf-mesher")
    ap.add_argument("--class-name", help="java: class name (default: output file stem)")
    ap.add_argument("--physics", default="none", choices=["none", "fullwave"],
                    help="java / mphtxt (Import_<name>.java): also export the Electromagnetic Waves physics, ports, "
                         "axion source and imp/volt frequency studies (3D designs with a port pickup)")
    ap.add_argument("-v", "--verbose", type=int, default=0)
    a = ap.parse_args(argv)
    ext = {"mphtxt": ".mphtxt", "java": ".java", "step": ".step"}[a.format]
    src = Path(a.source)
    outp = Path(a.output) if a.output else (src / ("comsol" + ext) if src.is_dir() else src.with_suffix(ext))
    if a.format == "java" and not a.output and not a.class_name:
        outp = outp.with_name(re.sub(r"\W", "_", outp.stem) + ext)
    try:
        res = export_comsol(src, outp, format=a.format, order=a.order, workdir=a.workdir, mesher=a.mesher,
                            verbose=a.verbose, class_name=a.class_name, feature_angle=a.feature_angle,
                            physics=a.physics)
    except Exception as ex:  # noqa: BLE001  (CLI: report as JSON like axf)
        print(json.dumps({"ok": False, "error": str(ex)}, indent=2))
        return 1
    print(json.dumps({"ok": True, **res}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
