"""Readers for COMSOL native binary data files (``*.mphbin``): meshes and geometry objects.  Beta.

Primary reference: the COMSOL Programming Reference Manual, chapter "The COMSOL File Formats", section
"Native Binary Data Files and Text Data Files":

- File structure (version 0.1, tags, types, objects with the header ``0 0 1``):
  https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.19.html
- Objects (version, unused integer, serialization type 1 = Serializable):
  https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.20.html
- Terminology (Boolean = 8-bit char, integer = 32-bit, double = 64-bit; matrices row-major):
  https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.21.html
- Binary file format (little-endian; a string is its length followed by one integer per character):
  https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.22.html
- Mesh (class version 4): https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.35.html
- Selection: https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.39.html
- Geom3 (class version 3): https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.32.html
  (Geom2 54.31, Geom1 54.30).

What the documents do not cover, and was established empirically on the five meshes of the 2211.00008 models
(COMSOL 5.6 saves, and a COMSOL 6.4 re-save of one of them; see docs/comsol_import.md):

The meshes stored inside ``.mph`` archives are **Mesh class version 7**, not the documented version 4.  Version 7
keeps the documented order of the fields (sdim, number of vertices, lowest index, coordinates, element types; per
type the name, vertices per element, elements, entity indices) and adds:

- two Boolean (1-byte) flags after sdim and one integer (=1) before the vertex count;
- matrices written as an outer count followed by one length-prefixed row per entry, i.e. ``double[d][np]`` is
  ``d, (np, np doubles) x d``; the element array is stored per local vertex, ``nep, (ne, ne ints) x nep``
  (the transpose of the documented ``integer[ne][nep]``);
- per element type, after the elements: ``(npar, nrows)`` and, if nrows > 0, the element parameter values (for
  ``edg`` an outer count + rows; for ``tri`` a variable layout that is skipped by locating the entity block);
- the entity block written as ``(ne, ne, ne ints)``;
- after it an up/down block: 0, or ``k, (ne, ne ints) x k`` (``tri``: the domain above and below each face,
  0 = outside/void).

Entity numbering follows the manual: domains (``tet`` in 3D, ``tri`` in 2D) are 1-based, boundaries, edges and
points are 0-based (the COMSOL GUI shows boundary/edge/point index + 1).  The reader cross-checks every block
(counts, index ranges, the entity block marker) and raises ``MphbinError`` on anything unexpected rather than
guessing.

Geometry objects: ``Geom3`` version 3 is decoded exactly as documented (vertices, parameter vertices, edges,
parameter edges, faces with their up/down domains).  The CAD representation ``Cad3Part`` (Parasolid kernel,
``geomrep = cadps``) is not documented and is reported as such.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = ["MphbinError", "ComsolMesh", "read_mesh", "Geom3", "read_geometry", "object_class"]

# vertices per element of COMSOL's linear element types (Mesh Element Numbering Conventions)
_NEP = {"vtx": 1, "edg": 2, "tri": 3, "quad": 4, "tet": 4, "pyr": 5, "prism": 6, "hex": 8,
        "edg2": 3, "tri2": 6, "quad2": 9, "tet2": 10, "pyr2": 14, "prism2": 18, "hex2": 27}
_DIM = {"vtx": 0, "edg": 1, "tri": 2, "quad": 2, "tet": 3, "pyr": 3, "prism": 3, "hex": 3,
        "edg2": 1, "tri2": 2, "quad2": 2, "tet2": 3, "pyr2": 3, "prism2": 3, "hex2": 3}


class MphbinError(ValueError):
    """The binary file does not have the expected structure."""


class _Reader:
    def __init__(self, b: bytes, what: str):
        self.b, self.p, self.what = b, 0, what

    def fail(self, msg: str):
        raise MphbinError(f"{self.what}: {msg} (at byte {self.p} of {len(self.b)})")

    def need(self, n: int):
        if n < 0 or self.p + n > len(self.b):
            self.fail(f"truncated: need {n} more bytes")

    def i32(self) -> int:
        self.need(4)
        v = struct.unpack_from("<i", self.b, self.p)[0]
        self.p += 4
        return v

    def f64(self) -> float:
        self.need(8)
        v = struct.unpack_from("<d", self.b, self.p)[0]
        self.p += 8
        return v

    def boolean(self) -> int:
        self.need(1)
        v = self.b[self.p]
        self.p += 1
        if v not in (0, 1):
            self.fail(f"Boolean must be 0 or 1, found {v}")
        return v

    def ints(self, n: int) -> np.ndarray:
        if n < 0:
            self.fail(f"negative array length {n}")
        self.need(4 * n)
        a = np.frombuffer(self.b, "<i4", n, self.p).astype(np.int64)
        self.p += 4 * n
        return a

    def doubles(self, n: int) -> np.ndarray:
        if n < 0:
            self.fail(f"negative array length {n}")
        self.need(8 * n)
        a = np.frombuffer(self.b, "<f8", n, self.p).copy()
        self.p += 8 * n
        return a

    def string(self) -> str:
        n = self.i32()
        if not 0 <= n <= 4096:
            self.fail(f"implausible string length {n}")
        return "".join(map(chr, self.ints(n)))

    def peek_i32(self, k: int = 0) -> int:
        if self.p + 4 * (k + 1) > len(self.b):
            return -(2 ** 31)
        return struct.unpack_from("<i", self.b, self.p + 4 * k)[0]


def _header(r: _Reader) -> tuple[list[str], str]:
    """File structure 0.1 (manual: File Structure, Objects): version, tags, types, the first object's class."""
    major, minor = r.i32(), r.i32()
    if (major, minor) != (0, 1):
        r.fail(f"unsupported file format version {major}.{minor} (expected 0.1)")
    tags = [r.string() for _ in range(r.i32())]
    _types = [r.string() for _ in range(r.i32())]
    v, _unused, typ = r.i32(), r.i32(), r.i32()
    if (v, typ) != (0, 1):
        r.fail(f"object header {v} {_unused} {typ} is not '0 0 1' (Serializable)")
    return tags, r.string()


def object_class(data: bytes) -> str:
    """Class name of the first object in a native binary file (``Mesh``, ``Geom3``, ``Cad3Part``, ...)."""
    return _header(_Reader(data, "mphbin"))[1]


@dataclass
class ComsolMesh:
    """A COMSOL mesh: vertex coordinates and, per element type, connectivity (0-based vertex indices), the
    geometric entity of each element (domains 1-based, boundaries/edges/points 0-based, as in the file) and,
    when stored, the up/down domains of each element."""
    sdim: int
    points: np.ndarray                      # (np, sdim)
    elements: dict = field(default_factory=dict)   # type -> (ne, nep) int array
    entities: dict = field(default_factory=dict)   # type -> (ne,) int array
    updown: dict = field(default_factory=dict)     # type -> (ne, 2) int array (domain up, domain down)
    version: int = 0
    lowest_index: int = 0

    @property
    def domain_type(self) -> str:
        return {3: "tet", 2: "tri"}.get(self.sdim, "edg")

    def summary(self) -> dict:
        out = {"sdim": self.sdim, "class_version": self.version, "vertices": int(len(self.points)), "types": {}}
        for t, el in self.elements.items():
            ent = self.entities[t]
            out["types"][t] = {"elements": int(len(el)),
                               "entities": [int(ent.min()), int(ent.max())] if len(ent) else [],
                               "distinct_entities": int(len(np.unique(ent)))}
        return out


def read_mesh(data: bytes, what: str = "mesh.mphbin") -> ComsolMesh:
    """Decode a COMSOL mesh object (class ``Mesh``, version 7 as saved in .mph archives; also version 4)."""
    r = _Reader(data, what)
    _tags, cls = _header(r)
    if cls != "Mesh":
        r.fail(f"first object is a {cls}, not a Mesh")
    version = r.i32()
    sdim = r.i32()
    if sdim == 0:
        return ComsolMesh(0, np.zeros((0, 0)), version=version)
    if not 1 <= sdim <= 3:
        r.fail(f"space dimension {sdim}")
    if version == 4:
        return _read_mesh_v4(r, sdim)
    if version != 7:
        r.fail(f"Mesh class version {version} is not supported (4 and 7 are)")
    r.boolean(), r.boolean()
    one = r.i32()
    if one != 1:
        r.fail(f"expected 1 before the vertex count, found {one}")
    npnt = r.i32()
    low = r.i32()
    nc = r.i32()
    if nc != sdim:
        r.fail(f"coordinate rows {nc} != sdim {sdim}")
    cols = []
    for _ in range(nc):
        n = r.i32()
        if n != npnt:
            r.fail(f"coordinate row of length {n} != number of vertices {npnt}")
        cols.append(r.doubles(n))
    pts = np.array(cols).T
    mesh = ComsolMesh(sdim, pts, version=version, lowest_index=low)
    ntypes = r.i32()
    if not 0 <= ntypes <= 32:
        r.fail(f"implausible number of element types {ntypes}")
    for _ in range(ntypes):
        name = r.string()
        if name not in _NEP:
            r.fail(f"unknown element type '{name}'")
        nep = r.i32()
        if nep != _NEP[name]:
            r.fail(f"{name}: {nep} vertices per element, expected {_NEP[name]}")
        rows = []
        for _k in range(nep):
            ne = r.i32()
            rows.append(r.ints(ne))
        if len({len(x) for x in rows}) > 1:
            r.fail(f"{name}: element rows of different lengths")
        el = np.array(rows).T.reshape(-1, nep) - low
        ne = len(el)
        if ne and (el.min() < 0 or el.max() >= npnt):
            r.fail(f"{name}: vertex index out of range")
        # element parameters: (npar, nrows), then (if nrows > 0) a type-dependent block; the entity block that
        # follows starts with the marker (ne, ne).  Skip to the first marker whose ne entries are valid indices.
        _npar, nrows = r.i32(), r.i32()
        if nrows:
            marker = struct.pack("<2i", ne, ne)
            q = r.p
            while True:
                k = r.b.find(marker, q)
                if k < 0:
                    r.fail(f"{name}: entity block not found")
                if k + 8 + 4 * ne <= len(r.b):
                    ent = np.frombuffer(r.b, "<i4", ne, k + 8)
                    if ne == 0 or (ent.min() >= 0 and ent.max() < 10_000_000):
                        r.p = k
                        break
                q = k + 4
        n1, n2 = r.i32(), r.i32()
        if n1 != ne or n2 != ne:
            r.fail(f"{name}: entity block ({n1}, {n2}) does not match {ne} elements")
        ent = r.ints(ne)
        nud = r.i32()
        ud = None
        if nud:
            if nud != 2:
                r.fail(f"{name}: up/down block with {nud} rows")
            ud = np.array([r.ints(r.i32()) for _ in range(nud)]).T
            if len(ud) != ne:
                r.fail(f"{name}: up/down rows of length {len(ud)} != {ne}")
        mesh.elements[name] = el.astype(np.int64)
        mesh.entities[name] = ent
        if ud is not None:
            mesh.updown[name] = ud
    return mesh


def _read_mesh_v4(r: _Reader, sdim: int) -> ComsolMesh:
    """Mesh class version 4 exactly as documented (row-major matrices without length prefixes)."""
    npnt = r.i32()
    low = r.i32()
    pts = r.doubles(sdim * npnt).reshape(npnt, sdim)
    mesh = ComsolMesh(sdim, pts, version=4, lowest_index=low)
    for _ in range(r.i32()):
        name = r.string()
        nep, ne = r.i32(), r.i32()
        el = r.ints(ne * nep).reshape(ne, nep) - low
        nd = r.i32()
        ent = r.ints(nd)
        mesh.elements[name] = el
        mesh.entities[name] = ent
    return mesh


# ------------------------------------------------------------------------------------------------ geometry
@dataclass
class Geom3:
    """A ``Geom3`` object (manual, Geom3 class version 3).  Indices as in the file: vertices/edges/faces are
    1-based in the edge and face records; ``face_updown`` holds the up and down domain of each face (0 = none)."""
    version: int
    type: int
    vertices: np.ndarray        # (nv, 3)
    edges: np.ndarray           # (ne, 2) 1-based vertex indices
    face_updown: np.ndarray     # (nf, 2)
    face_surface: np.ndarray    # (nf,)

    @property
    def ndomains(self) -> int:
        return int(self.face_updown.max()) if self.face_updown.size else 0

    def summary(self) -> dict:
        return {"class": "Geom3", "version": self.version, "type": self.type, "vertices": len(self.vertices),
                "edges": len(self.edges), "faces": len(self.face_updown), "domains": self.ndomains}


def read_geometry(data: bytes, what: str = "geometry.mphbin"):
    """Decode a geometry object.  Returns a ``Geom3`` for class Geom3 (version 3); for other classes
    (``Geom2``, ``Cad3Part``, ...) returns ``{"class": name, "decoded": False}``."""
    r = _Reader(data, what)
    _tags, cls = _header(r)
    if cls != "Geom3":
        return {"class": cls, "decoded": False}
    version, typ = r.i32(), r.i32()
    if version != 3:
        return {"class": cls, "version": version, "decoded": False}
    r.boolean()
    r.f64(), r.f64()
    nv = r.i32()
    V = np.zeros((nv, 3))
    for k in range(nv):
        V[k] = (r.f64(), r.f64(), r.f64())
        r.i32(), r.f64()
    for _ in range(r.i32()):                 # ParamVertexData
        r.i32(), r.f64(), r.f64(), r.i32(), r.i32(), r.f64()
    ne = r.i32()
    E = np.zeros((ne, 2), dtype=np.int64)
    for k in range(ne):
        E[k] = (r.i32(), r.i32())
        r.f64(), r.f64(), r.i32(), r.i32(), r.f64()
    for _ in range(r.i32()):                 # ParamEdgeData
        r.i32(), r.i32(), r.i32(), r.f64(), r.f64(), r.i32(), r.i32(), r.i32(), r.i32(), r.f64()
    nf = r.i32()
    F = np.zeros((nf, 2), dtype=np.int64)
    S = np.zeros(nf, dtype=np.int64)
    for k in range(nf):
        F[k] = (r.i32(), r.i32())
        S[k] = r.i32()
        r.f64()
    if nv and (E.size and (E.min() < 1 or E.max() > nv)):
        r.fail("edge vertex index out of range")
    if F.size and F.min() < 0:
        r.fail("negative face domain")
    return Geom3(version, typ, V, E, F, S)
