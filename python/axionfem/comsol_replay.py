"""Replay route of the COMSOL importer: rebuild the geometry sequence from axionFEM solids.  Beta.

The COMSOL geometry sequence (work planes with 2D primitives, Revolve/Extrude, 3D primitives, transforms and
booleans, Form Union) is replayed symbolically: every geometry object is a list of *cells* (the pieces that Form
Union and layers create), each an axionFEM solid in its own frame plus a rigid transform, with an exact
point-membership test.  Zero-thickness parts (revolved line segments, work-plane faces left in 3D) are *sheets*.

COMSOL's domain and boundary numbers cannot be reproduced, so the selections are mapped geometrically through
the mesh stored in the .mph: every COMSOL tetrahedron centroid is located in the replayed cells, which gives the
cell signature (set of cells containing it) of each COMSOL domain; domains of the replay with no COMSOL elements are
the domains COMSOL left out of the physics (PEC holes).  COMSOL boundary triangles are matched to sheets (distance
to the sheet surface).  Without a stored mesh the replay route is refused.

Approximations (reported): sheets become thin regions (``sheet_thickness``, default 1e-4 of the model size) on the
side where COMSOL meshed air; the lumped port becomes a volume port (axionFEM's uniform-conductivity gap feed) of
that thickness.  Use the mesh route for like-for-like results; the replay route gives an editable design.

Supported: WorkPlane (quick planes xy/yz/zx/yx/zy/xz with offset), Square, Rectangle (with layers), Circle
(full, no layers), Polygon (solid), LineSegment, Revolve (2D or 3D axis), Extrude (one distance), Block, Cylinder,
Sphere, Cone, Torus, Rotate, Move, Copy (single displacement), Union/Difference/Intersection, Form Union.
Anything else raises :class:`ReplayUnsupported` (the importer then falls back to the mesh route).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import comsol_mphbin as mb
from .comsol_model import ComsolModel, ExpressionError, Feature, evaluate

__all__ = ["ReplayUnsupported", "build_replay_route", "replay_geometry"]


class ReplayUnsupported(RuntimeError):
    """The geometry sequence uses a feature the replay does not cover."""


# ------------------------------------------------------------------------------------------------ transforms
def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _M(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def _apply(M, p):
    return p @ M[:3, :3].T + M[:3, 3]


def _inv(M):
    Ri = M[:3, :3].T
    return _M(Ri, -Ri @ M[:3, 3])


def _axis_angle(R):
    """Rotation matrix -> (axis, angle in degrees)."""
    c = (np.trace(R) - 1) / 2
    ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    if ang < 1e-12:
        return [0.0, 0.0, 1.0], 0.0
    if abs(ang - 180) < 1e-6:
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        ax = v[:, np.argmax(w)]
        return ax.tolist(), 180.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (ax / np.linalg.norm(ax)).tolist(), ang


def _place(solid: dict, M) -> dict:
    """axionFEM solid with the rigid transform M (rotate about the origin, then translate)."""
    if np.allclose(M, np.eye(4), atol=1e-15):
        return solid
    if abs(np.linalg.det(M[:3, :3]) - 1) > 1e-9:
        raise ReplayUnsupported("improper transform (mirror) on a solid")
    out = dict(solid)
    ax, ang = _axis_angle(M[:3, :3])
    if ang:
        out["rotate"] = {"axis": [float(x) for x in ax], "angle_deg": float(ang), "origin": [0.0, 0.0, 0.0]}
    t = M[:3, 3]
    if np.any(np.abs(t) > 0):
        out["translate"] = [float(x) for x in t]
    return out


# ------------------------------------------------------------------------------------------------ 2D shapes
def _poly_contains(pts, poly):
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / ((y2 - y1) if y2 != y1 else 1e-300) + x1)
        inside ^= c
    return inside


@dataclass
class Shape2:
    """A 2D region with an axionFEM shape description and an exact membership test (closed set, tol)."""
    shape: dict
    test: object                 # callable (n,2) -> bool
    bbox: tuple                  # (xmin, ymin, xmax, ymax)
    xs: list = field(default_factory=list)   # coordinates for the sampling grid
    ys: list = field(default_factory=list)

    def contains(self, p, tol=0.0):
        return self.test(np.atleast_2d(p), tol)


def rect2(x0, x1, y0, y1) -> Shape2:
    def t(p, tol=0.0):
        return (p[:, 0] >= x0 - tol) & (p[:, 0] <= x1 + tol) & (p[:, 1] >= y0 - tol) & (p[:, 1] <= y1 + tol)
    return Shape2({"rect": {"r": [x0, x1], "z": [y0, y1]}}, t, (x0, y0, x1, y1), [x0, x1], [y0, y1])


def poly2(points) -> Shape2:
    P = np.asarray(points, float)

    def t(p, tol=0.0):
        inside = _poly_contains(p, P)
        if tol > 0:
            inside |= _dist_to_polyline(p, P, closed=True) <= tol
        elif tol < 0:
            inside &= _dist_to_polyline(p, P, closed=True) > -tol
        return inside
    return Shape2({"polygon": {"points": P.tolist()}}, t, (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()),
                  P[:, 0].tolist(), P[:, 1].tolist())


def disk2(cx, cy, r) -> Shape2:
    def t(p, tol=0.0):
        return np.hypot(p[:, 0] - cx, p[:, 1] - cy) <= r + tol
    xs = [cx - r, cx, cx + r] + [cx + r * math.cos(a) for a in np.linspace(0, 2 * math.pi, 17)]
    ys = [cy - r, cy, cy + r] + [cy + r * math.sin(a) for a in np.linspace(0, 2 * math.pi, 17)]
    return Shape2({"disk": {"center": [cx, cy], "radius": r}}, t, (cx - r, cy - r, cx + r, cy + r), xs, ys)


def _dist_to_polyline(p, P, closed=False):
    d = np.full(len(p), np.inf)
    n = len(P)
    for i in range(n if closed else n - 1):
        a, b = P[i], P[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab) or 1e-300
        s = np.clip(((p - a) @ ab) / L2, 0, 1)
        q = a + s[:, None] * ab
        d = np.minimum(d, np.hypot(*(p - q).T))
    return d


def map_shape(s: Shape2, A, b) -> Shape2:
    """Affine image q -> A q + b of a 2D shape (A orthogonal)."""
    Ai = A.T

    def t(p, tol=0.0):
        return s.test((p - b) @ Ai.T, tol)
    sh = _map_json(s.shape, A, b)
    corners = np.array([[s.bbox[0], s.bbox[1]], [s.bbox[2], s.bbox[1]], [s.bbox[0], s.bbox[3]], [s.bbox[2], s.bbox[3]]])
    c2 = corners @ A.T + b
    grid = np.array([[x, y] for x in s.xs for y in s.ys]) @ A.T + b if s.xs and s.ys else c2
    return Shape2(sh, t, (c2[:, 0].min(), c2[:, 1].min(), c2[:, 0].max(), c2[:, 1].max()),
                  grid[:, 0].tolist(), grid[:, 1].tolist())


def _map_json(sh: dict, A, b) -> dict:
    k, v = next(iter(sh.items()))
    axis_aligned = np.allclose(np.abs(A), np.eye(2)) or np.allclose(np.abs(A), np.eye(2)[::-1])
    if k == "rect":
        (x0, x1), (y0, y1) = v["r"], v["z"]
        P = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ A.T + b
        if axis_aligned:
            return {"rect": {"r": [float(P[:, 0].min()), float(P[:, 0].max())],
                             "z": [float(P[:, 1].min()), float(P[:, 1].max())]}}
        return {"polygon": {"points": P.tolist()}}
    if k == "polygon":
        return {"polygon": {"points": (np.array(v["points"]) @ A.T + b).tolist()}}
    if k == "disk":
        c = np.array(v["center"]) @ A.T + b
        return {"disk": {"center": c.tolist(), "radius": v["radius"]}}
    if k in ("union", "difference"):
        return {k: [_map_json(x, A, b) for x in v]}
    raise ReplayUnsupported(f"cannot map 2D shape {k}")


def combine2(inside: list, outside: list) -> Shape2:
    """(∩ inside) − (∪ outside) as an axionFEM 2D shape (intersection A∩B = A − (A − B))."""
    base = inside[0]
    sh = base.shape
    for o in inside[1:]:
        sh = {"difference": [sh, {"difference": [sh, o.shape]}]}
    if outside:
        sh = {"difference": [sh] + [o.shape for o in outside]}

    def t(p, tol=0.0, _i=inside, _o=outside):
        r = np.ones(len(p), bool)
        for s in _i:
            r &= s.test(p, tol)
        for s in _o:
            r &= ~s.test(p, -tol)      # outside the others' interiors (their boundary belongs to this cell)
        return r
    bb = inside[0].bbox
    xs = sum((s.xs for s in inside + outside), [])
    ys = sum((s.ys for s in inside + outside), [])
    return Shape2(sh, t, bb, xs, ys)


@dataclass
class Seg2:
    """A 2D line segment (becomes a sheet)."""
    a: np.ndarray
    b: np.ndarray


# ------------------------------------------------------------------------------------------------ 3D cells
@dataclass
class Cell:
    name: str
    solid: dict                  # axionFEM solid in the cell frame
    M: np.ndarray                # cell frame -> global
    test: object                 # callable(points in cell frame, tol) -> bool
    samples: np.ndarray          # interior points in the cell frame
    bbox_local: np.ndarray       # (2,3) in the cell frame (corners transformed for the global bbox)
    sheet: dict | None = None    # sheets: {"dist": callable(local pts)->distance, "thicken": callable(t, side)->Cell}
    origin: str = ""

    def contains(self, P, tol=0.0):
        return self.test(_apply(_inv(self.M), P), tol)

    def gsamples(self):
        return _apply(self.M, self.samples)

    def gbbox(self):
        lo, hi = self.bbox_local
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        g = _apply(self.M, c)
        return g.min(axis=0), g.max(axis=0)

    def moved(self, M2, suffix=""):
        return Cell(self.name + suffix, self.solid, M2 @ self.M, self.test, self.samples, self.bbox_local,
                    self.sheet, self.origin)


@dataclass
class Obj:
    tag: str
    cells: list                  # 3D cells (solids and sheets)
    shapes2: list | None = None  # work-plane objects: [(Shape2|Seg2, label)]
    frame: tuple | None = None   # work plane: (origin, ex, ey)


def _grid_samples(shape: Shape2, extra=()):
    xs = sorted(set(round(x, 15) for x in shape.xs)) or [shape.bbox[0], shape.bbox[2]]
    ys = sorted(set(round(y, 15) for y in shape.ys)) or [shape.bbox[1], shape.bbox[3]]
    cx = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:])] or xs
    cy = [(a + b) / 2 for a, b in zip(ys[:-1], ys[1:])] or ys
    P = np.array([[x, y] for x in cx for y in cy] + list(extra))
    return P


# revolve: profile (r, h) about +z, angles a0..a1 (deg)
def revolve_cell(name, prof: Shape2, a0, a1, M, origin) -> Cell:
    full = (a1 - a0) >= 360 - 1e-9

    def t(p, tol=0.0):
        r = np.hypot(p[:, 0], p[:, 1])
        inside = prof.test(np.column_stack([r, p[:, 2]]), tol)
        if not full:
            th = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
            rel = np.mod(th - a0, 360.0)
            span = a1 - a0
            tol_deg = np.degrees(tol / np.maximum(r, 1e-300)) if tol else 0.0
            if tol >= 0:
                inside &= (rel <= span + tol_deg) | (rel >= 360 - tol_deg)
            else:
                inside &= (rel >= -tol_deg) & (rel <= span + tol_deg)
        return inside
    pr = _grid_samples(prof)
    pr = pr[prof.test(pr, 0.0)]
    if len(pr) == 0:
        raise ReplayUnsupported(f"{name}: no interior sample point in the profile")
    am = math.radians((a0 + a1) / 2)
    samples = np.column_stack([pr[:, 0] * math.cos(am), pr[:, 0] * math.sin(am), pr[:, 1]])
    rmax = prof.bbox[2]
    bb = np.array([[-rmax, -rmax, prof.bbox[1]], [rmax, rmax, prof.bbox[3]]])
    solid = {"revolve": {"profile": prof.shape, "center": [0, 0, 0], "axis": [0, 0, 1], "angles": [a0, a1]}}
    return Cell(name, solid, M, t, samples, bb, origin=origin)


def revolve_sheet(name, seg: Seg2, a0, a1, M, origin) -> Cell:
    a, b = seg.a, seg.b
    full = (a1 - a0) >= 360 - 1e-9

    def dist(p):
        r = np.hypot(p[:, 0], p[:, 1])
        d = _dist_to_polyline(np.column_stack([r, p[:, 2]]), np.array([a, b]))
        if not full:
            th = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
            rel = np.mod(th - a0, 360.0)
            out = (rel > a1 - a0) & (rel < 360)
            gap = np.minimum(rel - (a1 - a0), 360 - rel)
            d = np.where(out, np.hypot(d, np.radians(gap) * r), d)
        return d

    def thicken(t, side):
        tang = (b - a) / np.linalg.norm(b - a)
        n = np.array([tang[1], -tang[0]]) * side
        P = [a, b, b + t * n, a + t * n]
        if abs(tang[0]) < 1e-12 or abs(tang[1]) < 1e-12:
            xs, ys = [p[0] for p in P], [p[1] for p in P]
            prof = rect2(min(xs), max(xs), min(ys), max(ys))
        else:
            prof = poly2(P)
        return revolve_cell(name, prof, a0, a1, M, origin)

    am = math.radians((a0 + a1) / 2)
    mid = (a + b) / 2
    samples = np.array([[mid[0] * math.cos(am), mid[0] * math.sin(am), mid[1]]])
    rmax = max(a[0], b[0])
    bb = np.array([[-rmax, -rmax, min(a[1], b[1])], [rmax, rmax, max(a[1], b[1])]])
    c = Cell(name, {}, M, lambda p, tol=0.0: np.zeros(len(p), bool), samples, bb, origin=origin)
    c.sheet = {"dist": dist, "thicken": thicken, "normal_local": None, "kind": "revolved segment",
               "extent": float(np.linalg.norm(b - a))}
    return c


def planar_sheet(name, s2: Shape2, M, origin) -> Cell:
    """A 2D region left in 3D (work plane z = 0 in the cell frame)."""
    def dist(p):
        inside = s2.test(p[:, :2], 1e-9)
        return np.where(inside, np.abs(p[:, 2]), np.inf)

    def thicken(t, side):
        z0, z1 = (0.0, t) if side > 0 else (-t, 0.0)
        return extrude_cell(name, s2, z0, z1, M, origin)

    pr = _grid_samples(s2)
    pr = pr[s2.test(pr, 0.0)]
    samples = np.column_stack([pr, np.zeros(len(pr))])
    bb = np.array([[s2.bbox[0], s2.bbox[1], 0.0], [s2.bbox[2], s2.bbox[3], 0.0]])
    c = Cell(name, {}, M, lambda p, tol=0.0: np.zeros(len(p), bool), samples, bb, origin=origin)
    c.sheet = {"dist": dist, "thicken": thicken, "kind": "planar face",
               "extent": float(min(s2.bbox[2] - s2.bbox[0], s2.bbox[3] - s2.bbox[1]))}
    return c


def _shape_polygon(s2: Shape2):
    k, v = next(iter(s2.shape.items()))
    if k == "rect":
        (x0, x1), (y0, y1) = v["r"], v["z"]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if k == "polygon":
        return v["points"]
    return None


def extrude_cell(name, s2: Shape2, z0, z1, M, origin) -> Cell:
    def t(p, tol=0.0):
        return s2.test(p[:, :2], tol) & (p[:, 2] >= z0 - tol) & (p[:, 2] <= z1 + tol)
    pr = _grid_samples(s2)
    pr = pr[s2.test(pr, 0.0)]
    samples = np.column_stack([pr, np.full(len(pr), (z0 + z1) / 2)])
    bb = np.array([[s2.bbox[0], s2.bbox[1], z0], [s2.bbox[2], s2.bbox[3], z1]])
    return Cell(name, _extrude_json(s2.shape, z0, z1, name), M, t, samples, bb, origin=origin)


def _extrude_json(sh: dict, z0, z1, name) -> dict:
    """axionFEM solid of a 2D shape extruded from z0 to z1 (rect -> box, disk -> cylinder, booleans kept)."""
    k, v = next(iter(sh.items()))
    if k == "rect":
        (x0, x1), (y0, y1) = v["r"], v["z"]
        return {"box": {"min": [x0, y0, z0], "max": [x1, y1, z1]}}
    if k == "polygon":
        if any(len(q) > 2 for q in v["points"]) or v.get("fillet") or v.get("fillets"):
            raise ReplayUnsupported(f"{name}: filleted polygon extrusion")
        return {"extrude": {"polygon": v["points"], "z": [z0, z1]}}
    if k == "disk":
        c = v["center"]
        return {"cylinder": {"base": [c[0], c[1], z0], "axis": [0, 0, z1 - z0], "radius": v["radius"]}}
    if k in ("union", "difference"):
        return {k: [_extrude_json(x, z0, z1, name) for x in v]}
    raise ReplayUnsupported(f"{name}: cannot extrude a {k}")


def _shape_is_rect(s2):
    return "rect" in s2.shape


def box_cell(name, lo, hi, M, origin) -> Cell:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def t(p, tol=0.0):
        return np.all((p >= lo - tol) & (p <= hi + tol), axis=1)
    return Cell(name, {"box": {"min": lo.tolist(), "max": hi.tolist()}}, M, t, ((lo + hi) / 2)[None], np.array([lo, hi]),
                origin=origin)


def cylinder_cell(name, r, h, M, origin) -> Cell:
    def t(p, tol=0.0):
        return (np.hypot(p[:, 0], p[:, 1]) <= r + tol) & (p[:, 2] >= -tol) & (p[:, 2] <= h + tol)
    return Cell(name, {"cylinder": {"base": [0, 0, 0], "axis": [0, 0, h], "radius": r}}, M, t,
                np.array([[0, 0, h / 2], [r / 2, 0, h / 2]]), np.array([[-r, -r, 0], [r, r, h]]), origin=origin)


def sphere_cell(name, r, M, origin) -> Cell:
    def t(p, tol=0.0):
        return np.linalg.norm(p, axis=1) <= r + tol
    return Cell(name, {"sphere": {"center": [0, 0, 0], "radius": r}}, M, t, np.zeros((1, 3)),
                np.array([[-r] * 3, [r] * 3]), origin=origin)


def cone_cell(name, r1, r2, h, M, origin) -> Cell:
    def t(p, tol=0.0):
        z = p[:, 2]
        rr = r1 + (r2 - r1) * np.clip(z / h, 0, 1)
        return (np.hypot(p[:, 0], p[:, 1]) <= rr + tol) & (z >= -tol) & (z <= h + tol)
    R = max(r1, r2)
    return Cell(name, {"cone": {"base": [0, 0, 0], "axis": [0, 0, h], "r1": r1, "r2": r2}}, M, t,
                np.array([[0, 0, h / 2]]), np.array([[-R, -R, 0], [R, R, h]]), origin=origin)


def torus_cell(name, R, a, M, origin) -> Cell:
    def t(p, tol=0.0):
        return np.hypot(np.hypot(p[:, 0], p[:, 1]) - R, p[:, 2]) <= a + tol
    return Cell(name, {"torus": {"center": [0, 0, 0], "axis": [0, 0, 1], "major": R, "minor": a}}, M, t,
                np.array([[R, 0, 0]]), np.array([[-R - a, -R - a, -a], [R + a, R + a, a]]), origin=origin)


def bool_cell(name, kind, cells: list, origin) -> Cell:
    """A cell that is a boolean of other cells (each with its own transform); frame = identity."""
    def t(p, tol=0.0, _c=cells):
        if kind == "union":
            r = np.zeros(len(p), bool)
            for c in _c:
                r |= c.contains(p, tol)
        elif kind == "intersection":
            r = np.ones(len(p), bool)
            for c in _c:
                r &= c.contains(p, tol)
        else:
            r = _c[0].contains(p, tol)
            for c in _c[1:]:
                r &= ~c.contains(p, -tol)
        return r
    solid = {kind: [_place(c.solid, c.M) for c in cells]}
    smp = np.vstack([c.gsamples() for c in cells])
    los, his = zip(*(c.gbbox() for c in cells))
    bb = np.array([np.min(los, axis=0), np.max(his, axis=0)])
    keep = t(smp)
    return Cell(name, solid, np.eye(4), t, smp[keep] if keep.any() else smp, bb, origin=origin)


# ------------------------------------------------------------------------------------------------ feature values
class Values:
    """Property value of a geometry feature: user expression (props), else the built value in the key, else the
    COMSOL default."""

    def __init__(self, model: ComsolModel, f: Feature, unit_len=1.0):
        self.m, self.f, self.ul = model, f, unit_len

    def ev(self, x):
        return float(np.real(evaluate(x, self.m.param_values, None, self.m.functions)))

    def raw(self, name):
        return self.f.props.get(name)

    def num(self, name, default=None, length=False):
        v = self.raw(name)
        if isinstance(v, str) and v.strip() != "":
            try:
                return self.ev(v) * (self.ul if length else 1)
            except ExpressionError:
                pass
        if isinstance(v, list) and v and v[0]:
            return self.ev(v[0][0]) * (self.ul if length else 1)
        k = self._key_nums(name)
        if k:
            return k[0] * (self.ul if length else 1)
        if default is None:
            raise ReplayUnsupported(f"{self.f.tag}: no value for {name}")
        return default

    def vec(self, name, default=None, length=False):
        v = self.raw(name)
        out = None
        if isinstance(v, list) and v and v[0]:
            out = [self.ev(x) for x in v[0]]
        elif isinstance(v, str) and v.strip():
            parts = v.split()
            try:
                out = [self.ev(x) for x in parts]
            except ExpressionError:
                out = None
        if out is None:
            out = self._key_nums(name, len(default) if default is not None else None)
        if not out:
            if default is None:
                raise ReplayUnsupported(f"{self.f.tag}: no value for {name}")
            out = list(default)
        return [x * (self.ul if length else 1) for x in out]

    def word(self, name, default=""):
        v = self.raw(name)
        if isinstance(v, str) and v:
            return v
        key = self.f.key or ""
        m = re.search(re.escape(name) + r"(on|off)(?=[a-z]|$)", key)
        return m.group(1) if m else default

    def _key_nums(self, name, n=None):
        """Numbers after the first occurrence of ``name`` in the built-state key (keys concatenate name/value
        pairs; with ``n``, the first occurrence with n values, so that e.g. 'axis' skips 'revaxis' (4 values))."""
        key = self.f.key or ""
        for m in re.finditer(re.escape(name) + r"(-?\d[\d.]*(?:E-?\d+)?(?:\|-?\d[\d.]*(?:E-?\d+)?)*)\|?", key):
            vals = [float(x) for x in m.group(1).split("|") if x]
            if n is None or len(vals) == n:
                return vals
        return None

    def inputs(self, prop="input"):
        node = self.raw(prop)
        tags = []
        if node is not None and hasattr(node, "iter"):
            for o in node.iter("object"):
                fe = o.get("feature")
                if fe:
                    tags.append(fe.rstrip("/").split("/")[-1])
        if not tags:
            m = re.search(r"input(?:OBJECT)?\[([^\]]*)\]", self.f.key or "")
            if m:
                tags = [t.strip() for t in m.group(1).split(",") if t.strip()]
        return tags


_PLANES = {"xy": ([1, 0, 0], [0, 1, 0], 2), "yz": ([0, 1, 0], [0, 0, 1], 0), "zx": ([0, 0, 1], [1, 0, 0], 1),
           "yx": ([0, 1, 0], [1, 0, 0], 2), "zy": ([0, 0, 1], [0, 1, 0], 0), "xz": ([1, 0, 0], [0, 0, 1], 1)}


def _axis_vec(V: Values, prefix=""):
    at = V.word("axistype", "z")
    if at in ("x", "y", "z"):
        return np.eye(3)["xyz".index(at)]
    if at == "cartesian":
        return np.array(V.vec("ax3", [0, 0, 1]))
    if at == "spherical":
        th, ph = V.vec("ax2", [0, 0])
        th, ph = math.radians(th), math.radians(ph)
        return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    raise ReplayUnsupported(f"{V.f.tag}: axis type {at}")


def _frame_z_to(a):
    """Rotation taking +z to the unit vector a (minimal rotation, as axf-mesher)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    if s < 1e-14:
        return np.eye(3) if a[2] > 0 else _rot([1, 0, 0], 180)
    return _rot(v, math.degrees(math.atan2(s, z @ a)))


# ------------------------------------------------------------------------------------------------ replay
def _work_plane(model, f: Feature):
    V = Values(model, f)
    pt = V.word("planetype", "quick")
    if pt != "quick":
        raise ReplayUnsupported(f"{f.tag}: work plane type {pt} (only quick planes)")
    qp = V.word("quickplane", "xy")
    if qp not in _PLANES:
        raise ReplayUnsupported(f"{f.tag}: quick plane {qp}")
    ex, ey, k = _PLANES[qp]
    off = V.num("quick" + "xyz"[k], 0.0, length=True)
    if V.word("offsettype", "distance") != "distance":
        raise ReplayUnsupported(f"{f.tag}: work-plane offset type {V.word('offsettype')}")
    o = np.zeros(3)
    o[k] = off
    # 2D sub-sequence
    shapes = []
    for c in f.children:
        if c.disabled:
            continue
        shapes += _shape2d(model, c)
    if V.word("unite", "on") != "on":
        raise ReplayUnsupported(f"{f.tag}: work plane without 'unite' (2D objects kept separate)")
    return o, np.array(ex, float), np.array(ey, float), _partition2(shapes, f.tag)


def _layers(V: Values, x0, x1, y0, y1, label):
    lay = V.raw("layer")
    vals = []
    if isinstance(lay, list) and lay and lay[0]:
        vals = [V.ev(x) for x in lay[0]]
    elif V._key_nums("layer"):
        vals = V._key_nums("layer")
    if not vals:
        return [(rect2(x0, x1, y0, y1), label)]
    sides = {s: V.word("layer" + s, "on" if s == "bottom" else "off") for s in ("left", "right", "bottom", "top")}
    if not any(v == "on" for v in sides.values()):
        return [(rect2(x0, x1, y0, y1), label)]
    cum = np.cumsum(vals)
    xs = [x0] + ([x0 + c for c in cum] if sides["left"] == "on" else []) + \
         ([x1 - c for c in cum[::-1]] if sides["right"] == "on" else []) + [x1]
    ys = [y0] + ([y0 + c for c in cum] if sides["bottom"] == "on" else []) + \
         ([y1 - c for c in cum[::-1]] if sides["top"] == "on" else []) + [y1]
    xs, ys = sorted(set(xs)), sorted(set(ys))
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            out.append((rect2(xs[i], xs[i + 1], ys[j], ys[j + 1]), f"{label}[{i},{j}]"))
    return out


def _shape2d(model, f: Feature):
    V = Values(model, f)
    op = f.op
    if op in ("Square", "Rectangle"):
        if op == "Square":
            s = V.num("size", 1.0)
            w = h = s
        else:
            sz = V.vec("size", [1.0, 1.0])
            w, h = sz[0], sz[1]
        pos = V.vec("pos", [0.0, 0.0])
        if V.word("base", "corner") == "center":
            pos = [pos[0] - w / 2, pos[1] - h / 2]
        if abs(V.num("rot", 0.0)) > 1e-12:
            raise ReplayUnsupported(f"{f.tag}: rotated {op}")
        if V.word("type", "solid") != "solid":
            raise ReplayUnsupported(f"{f.tag}: {op} of type {V.word('type')}")
        return _layers(V, pos[0], pos[0] + w, pos[1], pos[1] + h, f.tag)
    if op == "Circle":
        r = V.num("r", 1.0)
        pos = V.vec("pos", [0.0, 0.0])
        if V.word("base", "center") != "center":
            pos = [pos[0] + r, pos[1] + r]
        if abs(V.num("angle", 360.0) - 360) > 1e-9 or V.raw("layer"):
            raise ReplayUnsupported(f"{f.tag}: circle sector or layers")
        return [(disk2(pos[0], pos[1], r), f.tag)]
    if op == "Polygon":
        xs = [V.ev(x) for x in str(V.raw("x") or "").split()]
        ys = [V.ev(y) for y in str(V.raw("y") or "").split()]
        if not xs or len(xs) != len(ys):
            raise ReplayUnsupported(f"{f.tag}: polygon coordinates not found")
        typ = V.word("type", "solid")
        if typ == "solid":
            return [(poly2(list(zip(xs, ys))), f.tag)]
        if typ == "open" and len(xs) == 2:
            return [(Seg2(np.array([xs[0], ys[0]]), np.array([xs[1], ys[1]])), f.tag)]
        raise ReplayUnsupported(f"{f.tag}: polygon type {typ}")
    if op == "LineSegment":
        if V.word("specify1", "coord") != "coord" or V.word("specify2", "coord") != "coord":
            raise ReplayUnsupported(f"{f.tag}: line segment endpoints from vertices")
        a = np.array(V.vec("coord1"))
        b = np.array(V.vec("coord2"))
        return [(Seg2(a, b), f.tag)]
    raise ReplayUnsupported(f"2D feature {f.tag} ({op}) is not supported by the replay")


def _partition2(shapes, label):
    """Form-union partition of the 2D solids of a work plane (segments are kept as they are)."""
    solids = [(s, l) for s, l in shapes if isinstance(s, Shape2)]
    segs = [(s, l) for s, l in shapes if isinstance(s, Seg2)]
    if not solids:
        return segs
    allx = sorted(set(sum((t.xs for t, _ in solids), [])))
    ally = sorted(set(sum((t.ys for t, _ in solids), [])))
    cx = [(a + b) / 2 for a, b in zip(allx[:-1], allx[1:])]
    cy = [(a + b) / 2 for a, b in zip(ally[:-1], ally[1:])]
    P = np.array([[x, y] for x in cx for y in cy])
    mem = np.array([s.test(P, 0.0) for s, _ in solids]).T   # (npts, nsolids)
    sig = {}
    for row in mem:
        if row.any():
            sig.setdefault(tuple(row), 0)
    cells = []
    for k, row in enumerate(sorted(sig)):
        ins = [solids[i][0] for i in range(len(solids)) if row[i]]
        outs = [solids[i][0] for i in range(len(solids)) if not row[i] and _bbox_overlap2(solids[i][0], ins)]
        name = "&".join(solids[i][1] for i in range(len(solids)) if row[i])
        cells.append((combine2(ins, outs) if (len(ins) > 1 or outs) else ins[0], f"{label}:{name}"))
    return cells + segs


def _bbox_overlap2(s, ins):
    for t in ins:
        if s.bbox[0] < t.bbox[2] and t.bbox[0] < s.bbox[2] and s.bbox[1] < t.bbox[3] and t.bbox[1] < s.bbox[3]:
            return True
    return False


def replay_geometry(model: ComsolModel, geom: str):
    """Replay a 3D geometry sequence.  Returns the list of final objects (after Form Union: their cells)."""
    g = model.geometries[geom]
    if g["sdim"] != 3:
        raise ReplayUnsupported("the replay route handles 3D geometries")
    ul = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6, "km": 1e3, "in": 0.0254, "ft": 0.3048}.get(g["lengthUnit"])
    if ul is None:
        raise ReplayUnsupported(f"length unit {g['lengthUnit']}")
    if g.get("angularUnit", "deg") != "deg":
        raise ReplayUnsupported(f"angular unit {g['angularUnit']}")
    objs: dict[str, Obj] = {}
    log = []
    final = None
    for f in g["features"]:
        if f.disabled:
            log.append((f.tag, "disabled"))
            continue
        V = Values(model, f, ul)
        op = f.op
        if op == "WorkPlane":
            o, ex, ey, shapes = _work_plane(model, f)
            if ul != 1.0:
                raise ReplayUnsupported("non-metre length unit with work planes")
            ez = np.cross(ex, ey)
            Mwp = _M(np.column_stack([ex, ey, ez]), o)
            cells = []
            for s, lab in shapes:
                if isinstance(s, Shape2):
                    cells.append(planar_sheet(lab, s, Mwp, f.tag))
            objs[f.tag] = Obj(f.tag, cells, shapes, (o, ex, ey))
        elif op in ("Revolve", "Extrude"):
            ins = V.inputs()
            if not ins:
                raise ReplayUnsupported(f"{f.tag}: no input objects")
            cells = []
            for it in ins:
                src = objs.pop(it, None)
                if src is None or src.shapes2 is None:
                    raise ReplayUnsupported(f"{f.tag}: input {it} is not a work-plane object")
                o, ex, ey = src.frame
                if op == "Revolve":
                    cells += _revolve(V, f, src, o, ex, ey)
                else:
                    cells += _extrude(V, f, src, o, ex, ey)
            objs[f.tag] = Obj(f.tag, cells)
        elif op in ("Block", "Cylinder", "Sphere", "Cone", "Torus"):
            objs[f.tag] = Obj(f.tag, [_primitive3(V, f, ul)])
        elif op in ("Rotate", "Move", "Copy"):
            ins = V.inputs()
            if op == "Rotate":
                ang = V.vec("rot", [0.0])
                if len(ang) != 1:
                    raise ReplayUnsupported(f"{f.tag}: several rotation angles (array of copies)")
                ax = _axis_vec(V)
                pos = np.array(V.vec("pos", [0, 0, 0], length=True))
                M2 = _M(None, pos) @ _M(_rot(ax, ang[0])) @ _M(None, -pos)
            else:
                d = [V.vec("displ" + c, [0.0], length=True) for c in "xyz"]
                if any(len(x) != 1 for x in d):
                    raise ReplayUnsupported(f"{f.tag}: several displacements (array of copies)")
                M2 = _M(None, [d[0][0], d[1][0], d[2][0]])
            keep = op == "Copy" or V.word("keep", "off") == "on"
            for it in ins:
                src = objs.get(it) if keep else objs.pop(it, None)
                if src is None:
                    raise ReplayUnsupported(f"{f.tag}: input {it} not found")
                if src.shapes2 is not None:
                    raise ReplayUnsupported(f"{f.tag}: transforming a work-plane object")
                # COMSOL names the output object after the feature (several inputs: tag(1), tag(2), ...)
                out_tag = f.tag if len(ins) == 1 else f"{f.tag}({ins.index(it) + 1})"
                objs[out_tag] = Obj(out_tag, [c.moved(M2) for c in src.cells])
        elif op in ("Union", "Difference", "Intersection"):
            ins = V.inputs()
            others = V.inputs("input2") if op == "Difference" else []
            A = [objs.pop(t) for t in ins if t in objs]
            B = [objs.pop(t) for t in others if t in objs]
            if len(A) != len(ins) or len(B) != len(others):
                raise ReplayUnsupported(f"{f.tag}: input objects not found")
            if any(c.sheet for o_ in A + B for c in o_.cells):
                raise ReplayUnsupported(f"{f.tag}: booleans with sheets")
            ca = [c for o_ in A for c in o_.cells]
            cb = [c for o_ in B for c in o_.cells]
            if op == "Union":
                if V.word("intbnd", "on") == "on":
                    cells = ca      # keeps interior boundaries: the cells stay (Form Union partitions them)
                else:
                    cells = [bool_cell(f.tag, "union", ca, f.tag)]
            elif op == "Intersection":
                cells = [bool_cell(f.tag, "intersection", [bool_cell(o_.tag, "union", o_.cells, o_.tag)
                                                         if len(o_.cells) > 1 else o_.cells[0] for o_ in A], f.tag)]
            else:
                cells = [bool_cell(f"{c.name}-", "difference", [c] + cb, f.tag) for c in ca]
            objs[f.tag] = Obj(f.tag, cells)
        elif op in ("Finalize",):
            act = V.word("action", "union")
            key = f.key or ""
            if "actionassembly" in key or act == "assembly":
                raise ReplayUnsupported("Form Assembly (identity pairs) is not supported")
            final = objs
        else:
            raise ReplayUnsupported(f"geometry feature {f.tag} ({op}) is not supported by the replay")
        log.append((f.tag, op))
    if final is None:
        final = objs
    for o_ in final.values():
        if o_.shapes2 is not None and o_.cells == []:
            raise ReplayUnsupported(f"work-plane object {o_.tag} with only curves left in 3D")
    return list(final.values()), log


def _revolve(V: Values, f, src: Obj, o, ex, ey):
    at = V.word("angtype", "specang")
    if at == "full":
        a0, a1 = 0.0, 360.0
    else:
        a0, a1 = V.num("angle1", 0.0), V.num("angle2", 360.0)
        a0, a1 = min(a0, a1), max(a0, a1)
    axt = V.word("axistype", "2d")
    cells = []
    for s, lab in src.shapes2:
        if axt == "2d":
            p = np.array(V.vec("pos", [0.0, 0.0]))
            d = np.array(V.vec("axis", [0.0, 1.0]))
            d = d / np.linalg.norm(d)
            P3 = o + p[0] * ex + p[1] * ey
            e3 = d[0] * ex + d[1] * ey
            perp2 = np.array([d[1], -d[0]])
            pts = _shape_points(s)
            side = np.sign(((pts - p) @ perp2).mean()) or 1.0
            e1 = side * (perp2[0] * ex + perp2[1] * ey)
            A = np.array([side * perp2, d])        # (r, h) = A (q - p)
            b = -A @ p
        else:
            P3 = np.array(V.vec("pos3", [0, 0, 0], length=True))
            e3 = np.array(V.vec("axis3", [0, 0, 1]))
            e3 = e3 / np.linalg.norm(e3)
            pts = _shape_points(s)
            X = o + pts[:, :1] * ex + pts[:, 1:2] * ey
            rel = X - P3
            h = rel @ e3
            radial = rel - np.outer(h, e3)
            rr = np.linalg.norm(radial, axis=1)
            k = int(np.argmax(rr))
            if rr[k] < 1e-14:
                raise ReplayUnsupported(f"{f.tag}: profile on the axis")
            e1 = radial[k] / rr[k]
            # the whole profile must lie in the half-plane spanned by e1, e3 (a meridian)
            n = np.cross(e3, e1)
            ex_n, ey_n = ex @ n, ey @ n
            if np.max(np.abs(rel @ n)) > 1e-9 * max(1.0, rr.max()):
                raise ReplayUnsupported(f"{f.tag}: 3D-axis revolve of a profile outside a meridian plane")
            # (r, h) of a work-plane point q: r = (o + q_x ex + q_y ey - P3) . e1, h = (...) . e3
            A = np.array([[ex @ e1, ey @ e1], [ex @ e3, ey @ e3]])
            b = np.array([(o - P3) @ e1, (o - P3) @ e3])
        e2 = np.cross(e3, e1)
        M = _M(np.column_stack([e1, e2, e3]), P3)
        if isinstance(s, Seg2):
            a2, b2 = A @ s.a + b, A @ s.b + b
            cells.append(revolve_sheet(f"{f.tag}:{lab}", Seg2(a2, b2), a0, a1, M, f.tag))
        else:
            prof = map_shape(s, A, b)
            if prof.bbox[0] < -1e-12:
                raise ReplayUnsupported(f"{f.tag}: profile {lab} crosses the revolution axis")
            cells.append(revolve_cell(f"{f.tag}:{lab}", prof, a0, a1, M, f.tag))
    return cells


def _shape_points(s):
    if isinstance(s, Seg2):
        return np.array([s.a, s.b])
    P = _grid_samples(s)
    P = P[s.test(P, 0.0)]
    return P if len(P) else np.array([[s.bbox[0], s.bbox[1]], [s.bbox[2], s.bbox[3]]])


def _extrude(V: Values, f, src: Obj, o, ex, ey):
    dist = V.vec("distance", [1.0])
    if len(dist) != 1:
        raise ReplayUnsupported(f"{f.tag}: several extrusion distances")
    for k, dflt in (("scale", 1.0), ("displ", 0.0), ("twist", 0.0)):
        v = V._key_nums(k)
        if v and any(abs(x - dflt) > 1e-12 for x in v):
            raise ReplayUnsupported(f"{f.tag}: extrusion with {k}")
    d = dist[0] * (-1 if V.word("reverse", "off") == "on" else 1)
    z0, z1 = min(0.0, d), max(0.0, d)
    ez = np.cross(ex, ey)
    M = _M(np.column_stack([ex, ey, ez]), o)
    cells = []
    for s, lab in src.shapes2:
        if isinstance(s, Seg2):
            raise ReplayUnsupported(f"{f.tag}: extruded segment (sheet) is not supported")
        cells.append(extrude_cell(f"{f.tag}:{lab}", s, z0, z1, M, f.tag))
    return cells


def _primitive3(V: Values, f, ul):
    op = f.op
    pos = np.array(V.vec("pos", [0, 0, 0], length=True))
    ax = _axis_vec(V)
    R = _frame_z_to(ax)
    rot = V.num("rot", 0.0)
    if abs(rot) > 1e-12:
        R = R @ _rot([0, 0, 1], rot)
    if V.raw("layer") and op in ("Block", "Cylinder"):
        lay = V.raw("layer")
        if isinstance(lay, list) and lay and lay[0] and any(V.ev(x) for x in lay[0]):
            raise ReplayUnsupported(f"{f.tag}: {op} with layers")
    if op == "Block":
        sz = np.array(V.vec("size", [1, 1, 1], length=True))
        lo = np.zeros(3) if V.word("base", "corner") == "corner" else -sz / 2
        return box_cell(f.tag, lo, lo + sz, _M(R, pos), f.tag)
    if op == "Cylinder":
        return cylinder_cell(f.tag, V.num("r", 1.0, True), V.num("h", 1.0, True), _M(R, pos), f.tag)
    if op == "Sphere":
        return sphere_cell(f.tag, V.num("r", 1.0, True), _M(None, pos), f.tag)
    if op == "Cone":
        r = V.num("r", 1.0, True)
        h = V.num("h", 1.0, True)
        if V.word("specifytop", "radius") == "radius":
            rt = V.num("rtop", 0.0, True)
        else:
            rt = r - h * math.tan(math.radians(V.num("ang", 0.0)))
        return cone_cell(f.tag, r, rt, h, _M(R, pos), f.tag)
    if op == "Torus":
        if abs(V.num("angle", 360.0) - 360) > 1e-9:
            raise ReplayUnsupported(f"{f.tag}: torus sector")
        return torus_cell(f.tag, V.num("rmaj", 1.0, True), V.num("rmin", 0.5, True), _M(R, pos), f.tag)
    raise ReplayUnsupported(f"{f.tag}: primitive {op}")


# ------------------------------------------------------------------------------------------------ route
@dataclass
class ReplayRoute:
    design: dict
    domain_map: dict


def build_replay_route(model: ComsolModel, report, out_dir: Path, stem: str, *, frequencies=None,
                       max_frequencies: int = 20, study=None, order=None, physics_tag=None,
                       sheet_thickness: float | None = None, check_geometry: bool = True) -> ReplayRoute:
    from .comsol_import import (ImportError_, _feat_desc, choose_frequencies, classify_source,
                                domain_materials, material_properties, study_frequencies, _port_geometry, _slug)
    emws = [p for p in model.physics if p["op"] == "ElectromagneticWaves" and not p["disabled"]]
    if physics_tag:
        emws = [p for p in emws if p["tag"] == physics_tag]
    if not emws:
        raise ReplayUnsupported("no emw interface")
    others = [p["tag"] for p in model.physics if not p["disabled"] and p is not emws[0]]
    if others:
        raise ReplayUnsupported(f"other physics interfaces {others} (reported by the mesh route)")
    ph = emws[0]
    if (ph["props"].get("SolveFor", "fullField")) != "fullField":
        raise ReplayUnsupported("scattered-field formulation")
    if (model.geometries[ph["geom"]].get("lengthUnit") or "m") != "m":
        raise ReplayUnsupported("non-metre geometry unit")
    geom = ph["geom"]
    objs, log = replay_geometry(model, geom)
    meshes = [m for m in model.meshes.values() if m["geom"] == geom and m["resource"]]
    if not meshes:
        raise ReplayUnsupported("no stored mesh to map the selections onto the replayed geometry")
    msq = meshes[0]
    mesh = mb.read_mesh(model.archive.read(msq["resource"]), msq["resource"])
    P = mesh.points
    tets, tdom = mesh.elements["tet"], mesh.entities["tet"]
    tris, tbnd = mesh.elements["tri"], mesh.entities["tri"] + 1
    L = float(np.ptp(P, axis=0).max())
    tol = 1e-7 * L
    cells = [c for o in objs for c in o.cells if not c.sheet]
    sheets = [c for o in objs for c in o.cells if c.sheet]
    if not cells:
        raise ReplayUnsupported("no solid cells after the geometry sequence")
    # ---- signatures of the COMSOL tetrahedra
    cen = P[tets].mean(axis=1)
    mem = np.array([c.contains(cen, tol) for c in cells]).T
    sig_of_tet = [tuple(np.nonzero(r)[0]) for r in mem]
    dom_sig = {}
    for d in np.unique(tdom):
        sel = np.nonzero(tdom == d)[0]
        sigs = {}
        for i in sel:
            sigs[sig_of_tet[i]] = sigs.get(sig_of_tet[i], 0) + 1
        best, nb = max(sigs.items(), key=lambda kv: kv[1])
        frac = nb / len(sel)
        if frac < 0.995 or best == ():
            raise ReplayUnsupported(f"COMSOL domain {d} does not map onto one replayed cell combination "
                                    f"({frac:.1%} of its elements in the best match {best})")
        dom_sig[int(d)] = best
    if len(set(dom_sig.values())) != len(dom_sig):
        raise ReplayUnsupported("two COMSOL domains map onto the same replayed cell combination")
    # replay signatures from the cells' own samples (finds the domains COMSOL did not mesh)
    smp = np.vstack([c.gsamples() for c in cells])
    mem2 = np.array([c.contains(smp, 0.0) for c in cells]).T
    all_sigs = {tuple(np.nonzero(r)[0]) for r in mem2 if r.any()} | set(dom_sig.values())
    sig_dom = {v: k for k, v in dom_sig.items()}
    # ---- physics, materials, frequencies (as in the mesh route)
    sel = ph["selection"]
    ndom = model.geometries[geom]["counts"].get(3) or int(tdom.max())
    if sel is None or sel.unknown:
        raise ReplayUnsupported("emw selection not understood")
    phys_doms = sorted(sel.entities) if sel.entities is not None else list(range(1, ndom + 1))
    frq = study_frequencies(model, report, study)
    fl = frequencies if frequencies is not None else choose_frequencies(frq["frequencies"], max_frequencies)
    dmat = domain_materials(model, geom, ndom, report)
    feats = ph["features"]
    wee_of = {}
    for f in feats:
        if f.op == "WaveEquationElectric" and not f.disabled:
            if f.selection is None or f.selection.unknown:
                raise ReplayUnsupported(f"{f.tag}: selection not understood")
            for d in (f.selection.entities if f.selection.entities is not None else phys_doms):
                wee_of[d] = f
    materials, regions, region_of_dom = {}, [], {}
    # domain cell solids
    t_sheet = sheet_thickness or 1e-4 * L

    def convex_contains(B: Cell, A: Cell) -> bool:
        """A inside B for a convex primitive B, tested on the corners of A's bounding box."""
        if not any(k in B.solid for k in ("box", "cylinder", "sphere", "cone")):
            return False
        lo, hi = A.gbbox()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        return bool(B.contains(corners, tol).all())

    def domain_solid(sig):
        ins = [cells[i] for i in sig]
        ins = [B for B in ins if not any(A is not B and convex_contains(B, A) for A in ins)] or ins
        outs = [c for j, c in enumerate(cells) if j not in sig and _bb_overlap(c, ins)]
        base = ins[0] if len(ins) == 1 else bool_cell("x", "intersection", ins, "")
        if outs:
            return {"difference": [_place(base.solid, base.M)] + [_place(c.solid, c.M) for c in outs]}
        return _place(base.solid, base.M)

    # sizes from the COMSOL mesh
    edge = np.linalg.norm(P[tets[:, 0]] - P[tets[:, 1]], axis=1)
    hdom = {int(d): float(np.median(edge[tdom == d])) for d in np.unique(tdom)}
    _a, _b, _c, _d = (P[tets[:, k]] for k in range(4))
    _tv = np.abs(np.einsum("ij,ij->i", np.cross(_b - _a, _c - _a), _d - _a)) / 6
    vol_d = {int(d): float(_tv[tdom == d].sum()) for d in np.unique(tdom)}
    vol_tot = float(_tv.sum())
    excluded_sigs = [s for s in all_sigs if s not in sig_dom or sig_dom[s] not in phys_doms]
    for s in sorted(all_sigs, key=lambda s: (s not in sig_dom, sig_dom.get(s, 0))):
        d = sig_dom.get(s)
        if d is not None and d in phys_doms:
            mat = dmat.get(d)
            props = material_properties(model, mat, wee_of.get(d), fl, report)
            key = (round(props["eps_r"], 12), round(props["tan_delta"], 14), round(props["mu_r"], 12),
                   round(props["sigma_S_m"], 9))
            if key == (1.0, 0.0, 1.0, 0.0):
                mname = "vacuum"
            else:
                spec_ = {"kind": "linear", "eps_r": props["eps_r"], "mu_r": props["mu_r"],
                         "tan_delta": props["tan_delta"], "sigma_S_m": props["sigma_S_m"]}
                mname = next((k for k, v in materials.items() if v == spec_), None)
                if mname is None:
                    base = "comsol_" + _slug((mat["name"] if mat else "m") + "_" + (mat["tag"] if mat else ""))
                    mname, k = base, 2
                    while mname in materials:
                        mname, k = f"{base}_{k}", k + 1
                    materials[mname] = spec_
            name = f"domain_{d}"
            sel_d = tdom == d
            small = vol_d.get(d, 0.0) < 1e-3 * vol_tot     # a small feature (e.g. a slot): refine its walls
            regions.append({"name": name, "material": mname, "solid": domain_solid(s),
                            "mesh_size": float(np.percentile(edge[sel_d], 75)),
                            **({"surface_size": float(np.percentile(edge[sel_d], 25))} if small else {}),
                            "notes": f"COMSOL domain {d}: cells {[cells[i].name for i in s]}"})
            region_of_dom[d] = name
            report.add("geometry", f"domain {d}", "mapped", f"region {name}",
                       f"replayed cells {[cells[i].name for i in s]}")
        else:
            nm = f"pec_{len([r for r in regions if r['name'].startswith('pec_')]) + 1}"
            regions.append({"name": nm, "material": "superconductor", "solid": domain_solid(s),
                            "notes": f"cells {[cells[i].name for i in s]}: "
                                     + ("outside the emw interface (COMSOL domain %s)" % d if d else
                                        "not meshed in COMSOL: outside the emw interface") + " -> PEC hole"})
            report.add("geometry", f"cells {[cells[i].name for i in s]}", "mapped", f"region {nm} (PEC, removed)",
                       "domain outside the emw interface: its surface is PEC in COMSOL")
    # ---- boundary features
    btype = {}
    lports, sources = [], []
    for f in feats:
        if f.disabled and f.op != "ExternalCurrentDensity":
            continue
        if f.selection is not None and f.selection.unknown:
            raise ReplayUnsupported(f"{_feat_desc(f)}: selection not understood")
        if any(not c.disabled and c.op != "UniformElement" for c in f.children):
            raise ReplayUnsupported(f"{_feat_desc(f)}: sub-features")
        ents = f.selection.entities if f.selection else None
        if f.op == "PerfectElectricConductor" and ents is None and not f.default:
            raise ReplayUnsupported(f"{_feat_desc(f)}: PEC on all boundaries")
        if f.op == "PerfectElectricConductor" and ents is not None:
            for b in ents:
                btype[b] = ("pec", f.tag)
        elif f.op == "LumpedPort":
            if f.p("PortType", "Uniform") != "Uniform":
                raise ReplayUnsupported(f"{f.tag}: port type {f.p('PortType')}")
            for b in ents or []:
                btype[b] = ("port", f.tag)
            lports.append(f)
        elif f.op == "ExternalCurrentDensity":
            sources.append(f)
        elif f.op in ("WaveEquationElectric", "init", "PerfectElectricConductor", "Continuity") or (f.disabled):
            pass
        else:
            raise ReplayUnsupported(f"{_feat_desc(f)} is not mapped by the replay route")
    if len(lports) > 1 or len([s_ for s_ in sources if not s_.disabled]) > 1:
        raise ReplayUnsupported("several lumped ports or current densities (reported by the mesh route)")
    # match typed boundaries to sheets
    # a triangle lies on a sheet when its three vertices do (COMSOL mesh vertices lie on the CAD surfaces; the
    # centroid of a flat triangle on a curved sheet does not)
    if sheets:
        sheet_dist = np.array([np.max(np.stack([s.sheet["dist"](_apply(_inv(s.M), P[tris[:, k]])) for k in range(3)]),
                                      axis=0) for s in sheets]).T
    else:
        sheet_dist = np.zeros((len(tris), 0))
    sheet_types = {}
    for b, (typ, ftag) in btype.items():
        sel_t = tbnd == b
        if not sel_t.any():
            continue
        if not sheets:
            raise ReplayUnsupported(f"boundary {b} ({ftag}) is not on a sheet: interior/solid-face boundary conditions "
                                    "are not mapped by the replay route")
        dmin = sheet_dist[sel_t]
        on = dmin <= 1e-6 * L
        if not on.all(axis=0).any():
            raise ReplayUnsupported(f"boundary {b} ({ftag}) does not lie on one replayed sheet")
        # the latest sheet containing the whole boundary
        k = int(np.nonzero(on.all(axis=0))[0][-1])
        sheet_types.setdefault(k, set()).add((typ, ftag, b))
    sheet_regions = []
    port_region, port_toward = None, None
    if sheet_thickness is None and lports:
        # thin enough not to dominate the port (a volume feed of thickness t stands for COMSOL's face port)
        pb = [b for b, (t_, _) in btype.items() if t_ == "port"]
        pg0 = _port_geometry(P, tris[np.isin(tbnd, pb)], tris, tbnd, btype, np.ones(len(tris), bool))
        t_sheet = min(t_sheet, 0.5 * min(pg0["length_m"], pg0["width_m"]))
    tet_cen_tree = None
    for k, s in enumerate(sheets):
        types = sheet_types.get(k, set())
        kinds = {t[0] for t in types}
        if not types:
            report.add("geometry", s.name, "ignored", detail="zero-thickness surface without a boundary condition "
                       "(continuity): dropped")
            continue
        if len(kinds) > 1:
            raise ReplayUnsupported(f"sheet {s.name} carries mixed boundary types {sorted(kinds)}")
        kind = kinds.pop()
        # side: the one where COMSOL meshed (emw) elements sit next to the sheet
        side = _sheet_side(s, t_sheet, P, tets, tdom, phys_doms)
        cell = s.sheet["thicken"](t_sheet, side)
        bnds = sorted(t[2] for t in types)
        if kind == "pec":
            nm = f"sheet_pec_{_slug(s.name)}"
            regions.append({"name": nm, "material": "superconductor", "surface_size": t_sheet,
                            "solid": _place(cell.solid, cell.M),
                            "notes": f"COMSOL zero-thickness PEC boundaries {bnds} ({s.name}), thickened by {t_sheet:g} m"})
            report.add("boundary", f"PEC boundaries {bnds}", "approximated", f"region {nm}",
                       f"zero-thickness sheet {s.name} -> thin PEC region {t_sheet:g} m thick (side {side:+d})")
        else:
            nm = f"port_{_slug(s.name)}"
            regions.append({"name": nm, "material": "vacuum", "surface_size": t_sheet / 2,
                            "solid": _place(cell.solid, cell.M),
                            "notes": f"COMSOL lumped port boundaries {bnds}, thickened by {t_sheet:g} m"})
            ptris = tris[np.isin(tbnd, bnds)]
            pg = _port_geometry(P, ptris, tris, tbnd, btype, np.ones(len(tris), bool))
            port_region, port_toward = nm, pg["toward"]
            report.add("boundary", f"lumped port boundaries {bnds}", "approximated", f"region {nm} (volume port)",
                       f"face port -> gap volume {t_sheet:g} m thick with axionFEM's uniform-conductivity feed")
    if lports and port_region is None:
        raise ReplayUnsupported("the lumped port is not on a replayed sheet")
    # ---- source
    src = None
    act = [s for s in sources if not s.disabled] or sources
    if act:
        s0 = act[0]
        if s0.selection is None or s0.selection.unknown:
            raise ReplayUnsupported(f"{s0.tag}: selection not understood")
        ents = s0.selection.entities if s0.selection.entities is not None else phys_doms
        ents = [d for d in ents if d in region_of_dom]
        sel_t = np.isin(tdom, ents)
        try:
            if (s0.p("coordinateSystem") or "GlobalSystem") != "GlobalSystem":
                raise ImportError_(f"Je in coordinate system {s0.p('coordinateSystem')}")
            src = classify_source(model, s0.params.get("Je"),
                                  np.vstack([P[tets[sel_t]].mean(axis=1), P[np.unique(tets[sel_t])]]))
            src.pop("exact")
            src["metadata"] = {"comsol_feature": s0.tag, "Je_A_m2": s0.params.get("Je"), "domains": ents}
        except (ImportError_, ExpressionError) as e:
            raise ReplayUnsupported(f"source {s0.tag}: {e}")
    # ---- domain solid: union of the top-level objects' cells
    env = [_place(c.solid, c.M) for c in cells]
    dom_solid = env[0] if len(env) == 1 else {"union": env}
    # air: the physics vacuum domain without source whose region is the largest -> leave to "air"
    fw = {"boundary": "pec", "rf_only": True}
    if fl:
        fw["frequencies"] = fl
    if src:
        fw["source"] = src
        fw["source_regions"] = [region_of_dom[d] for d in src["metadata"]["domains"]]
    design = {"schema": "axionfem.design/1", "name": _slug(Path(model.path).stem) + "_replay", "symmetry": "3d",
              "notes": f"Imported from {model.path} by axionfem.comsol_import, replay route (beta): geometry "
                       f"sequence {[t for t, _ in log]} rebuilt from axionFEM solids; zero-thickness boundaries "
                       f"thickened by {t_sheet:g} m. See import_report.md.",
              "regions": regions, "fullwave": fw,
              "mesh": {"order": order or 2, "size": float(np.percentile(edge, 90)), "surface_size": float(np.median(edge)),
                       "remove_pec": True, "domain_solid": dom_solid}}
    if materials:
        design["materials"] = materials
    if port_region:
        design["pickup"] = {"type": "port", "port": port_region, "toward": port_toward}
    for r in design["regions"]:
        if r.get("mesh_size") is None:
            r.pop("mesh_size", None)
    report.summary["replay"] = {"log": log, "cells": [c.name for c in cells], "sheets": [s.name for s in sheets],
                                "domain_signatures": {d: [cells[i].name for i in s] for d, s in dom_sig.items()},
                                "sheet_thickness_m": t_sheet}
    if check_geometry:
        _check_volumes(design, out_dir, stem, mesh, dom_sig, region_of_dom, report)
    return ReplayRoute(design, {d: region_of_dom.get(d) for d in dom_sig})


def _bb_overlap(c, ins):
    lo, hi = c.gbbox()
    for t in ins:
        l2, h2 = t.gbbox()
        if np.all(lo < h2) and np.all(l2 < hi):
            return True
    return False


def _sheet_side(s: Cell, t, P, tets, tdom, phys_doms):
    """+1 or -1: the side of the sheet whose probe points fall in COMSOL emw elements (air)."""
    from scipy.spatial import cKDTree
    keep = np.isin(tdom, phys_doms)
    T = tets[keep]
    cen = P[T].mean(axis=1)
    tree = cKDTree(cen)
    best, score = 1, -1
    for side in (1, -1):
        c = s.sheet["thicken"](t, side)
        probe = c.gsamples()
        n_in = 0
        for p in probe:
            idx = tree.query(p, k=min(16, len(cen)))[1]
            for i in np.atleast_1d(idx):
                if _in_tet(p, P[T[i]]):
                    n_in += 1
                    break
        if n_in > score:
            best, score = side, n_in
    return best


def _in_tet(p, v):
    A = np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
    try:
        l = np.linalg.solve(A, p - v[0])
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(l >= -1e-9) and l.sum() <= 1 + 1e-9)


def _check_volumes(design, out_dir: Path, stem, mesh, dom_sig, region_of_dom, report):
    """Build the replayed geometry with axf-mesher (no meshing) and compare region volumes with COMSOL's."""
    try:
        from .comsol import export_comsol
        out_dir.mkdir(parents=True, exist_ok=True)
        step = out_dir / f"{stem}_replay.step"
        res = export_comsol(design, str(step), format="step")
        info = json.loads(Path(res["solids"]).read_text())
    except Exception as e:  # noqa: BLE001
        report.add("geometry", "replay build check", "info", detail=f"axf-mesher geometry check not run: {e}")
        return
    vol = {}
    for s in info["step"]["solids"]:
        vol[s["region"]] = vol.get(s["region"], 0.0) + s["volume_m3"]
    P, T, D = mesh.points, mesh.elements["tet"], mesh.entities["tet"]
    a, b, c, d = (P[T[:, k]] for k in range(4))
    tv = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6
    rows = []
    for dom, reg in region_of_dom.items():
        vc = float(tv[D == dom].sum())
        va = vol.get(reg)
        rows.append({"domain": dom, "region": reg, "comsol_mesh_volume_m3": vc, "replay_volume_m3": va,
                     "relative_difference": (va - vc) / vc if va else None})
    report.summary["replay_volumes"] = rows
    worst = max((abs(r["relative_difference"]) for r in rows if r["relative_difference"] is not None), default=None)
    report.add("geometry", "replay vs COMSOL mesh volumes", "info",
               detail=f"largest relative volume difference {worst:.2e} (the COMSOL mesh is straight-sided)"
               if worst is not None else "no volumes")
    if worst is None or worst > 0.02:
        raise ReplayUnsupported(f"replayed region volumes differ from COMSOL's by {worst} (> 2 %)")
