"""COMSOL export (python/axionfem/comsol.py, docs/comsol_export.md).

The .mphtxt writer is checked by round trips on synthetic gmsh meshes (no mesher needed): the file is
parsed back and compared with the source mesh (vertex and element counts, volume of every region,
tag sets, entity numbering, element orientation, second-order node placement), and the derived
entity partition is checked for the properties COMSOL needs (contiguous indices, every boundary
bounded by edges, every edge between points).  The Java writer is compiled with javac against the
stand-ins in tests/comsol_stubs (skipped without a JDK).  Tests that need the mesher are marked.
"""
import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import axionfem as af
from axionfem import comsol as cx

HERE = Path(__file__).resolve().parent
STUBS = HERE / "comsol_stubs"


# ------------------------------------------------------------------ synthetic gmsh meshes
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def cube_mesh(n=3, h=1.0):
    """n^3 cubes of side h split into 6 tets each; the centre cube is region 3, the x < h layer region 2,
    the rest region 1 (air).  Faces on z = 0 are tagged 910 ('sym'), all other outer faces 901 ('outer')."""
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    pts = np.array([[i * h, j * h, k * h] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)], float)
    tets, attr = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                # corner numbering c[4a+2b+d] -> binary (a b d) matches the Kuhn table with bits (x y z)
                corner = [c[(v >> 2) * 4 + ((v >> 1) & 1) * 2 + (v & 1)] for v in range(8)]
                lab = 3 if (i, j, k) == (n // 2,) * 3 else (2 if i == 0 else 1)
                for t in _KUHN:
                    tets.append([corner[v] for v in t])
                    attr.append(lab)
    tets = np.array(tets)
    # boundary faces
    faces = {}
    for t in tets:
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(t[list(f)]))
            faces[key] = faces.get(key, 0) + 1
    bnd, btag = [], []
    L = n * h
    for f, c in faces.items():
        if c == 1:
            x = pts[list(f)]
            bnd.append(list(f))
            btag.append(910 if np.allclose(x[:, 2], 0) else 901)
    return pts, tets, np.array(attr), np.array(bnd), np.array(btag), L


def write_msh(path, pts, blocks, names, binary=False):
    """blocks: [(gmsh_type, conn (n,k) 0-based, phys (n,))]; names: {(dim, tag): name}."""
    ne = sum(len(c) for _, c, _ in blocks)
    with open(path, "wb") as f:
        f.write(b"$MeshFormat\n")
        if binary:
            f.write(b"2.2 1 8\n" + struct.pack("<i", 1) + b"\n")
        else:
            f.write(b"2.2 0 8\n")
        f.write(b"$EndMeshFormat\n$PhysicalNames\n%d\n" % len(names))
        for (d, t), n in names.items():
            f.write(b'%d %d "%s"\n' % (d, t, n.encode()))
        f.write(b"$EndPhysicalNames\n$Nodes\n%d\n" % len(pts))
        P = np.zeros((len(pts), 3))
        P[:, :pts.shape[1]] = pts
        if binary:
            for i, p in enumerate(P):
                f.write(struct.pack("<iddd", i + 1, *p))
            f.write(b"\n")
        else:
            for i, p in enumerate(P):
                f.write(b"%d %.17g %.17g %.17g\n" % (i + 1, *p))
        f.write(b"$EndNodes\n$Elements\n%d\n" % ne)
        eid = 1
        for t, conn, phys in blocks:
            if binary:
                f.write(struct.pack("<3i", t, len(conn), 2))
            for c, p in zip(conn, phys):
                if binary:
                    f.write(struct.pack("<%di" % (3 + len(c)), eid, int(p), int(p), *[int(x) + 1 for x in c]))
                else:
                    f.write((" ".join(str(x) for x in [eid, t, 2, int(p), int(p)] + [int(x) + 1 for x in c]) + "\n").encode())
                eid += 1
        if binary:
            f.write(b"\n")
        f.write(b"$EndElements\n")


NAMES3 = {(3, 1): "air", (3, 2): "slab", (3, 3): "core", (2, 901): "outer", (2, 910): "sym"}


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    d = tmp_path_factory.mktemp("cube")
    pts, tets, attr, bnd, btag, L = cube_mesh()
    blocks = [(4, tets, attr), (2, bnd, btag)]
    write_msh(d / "a.msh", pts, blocks, NAMES3, binary=False)
    write_msh(d / "b.msh", pts, blocks, NAMES3, binary=True)
    return d, pts, tets, attr, bnd, btag, L


def _check_partition_3d(r):
    """COMSOL needs: contiguous entity indices, every boundary bounded by edges, edges between points."""
    tet, tri, edg, vtx = (r["types"][k] for k in ("tet", "tri", "edg", "vtx"))
    ndom = tet["entities"].max()
    assert sorted(set(tet["entities"].tolist())) == list(range(1, ndom + 1))
    nb = tri["entities"].max() + 1
    assert sorted(set(tri["entities"].tolist())) == list(range(nb))
    ne = edg["entities"].max() + 1
    assert sorted(set(edg["entities"].tolist())) == list(range(ne))
    assert sorted(vtx["entities"].tolist()) == list(range(len(vtx["entities"])))
    vnodes = set(vtx["elements"][:, 0].tolist())
    # each edge entity is a simple chain whose two ends are points
    for e in range(ne):
        segs = edg["elements"][edg["entities"] == e]
        deg = {}
        for a, b in segs.tolist():
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        ends = [v for v, d in deg.items() if d == 1]
        assert len(ends) == 2 and set(ends) <= vnodes, f"edge {e} is not a chain between two points"
        assert all(d <= 2 for d in deg.values())
        assert not (set(v for v, d in deg.items() if d == 2) & vnodes), f"a point lies inside edge {e}"
    # every boundary has at least one bounding edge
    eset = {tuple(sorted(s)) for s in edg["elements"].tolist()}
    for b in range(nb):
        f = tri["elements"][tri["entities"] == b]
        es = {tuple(sorted(p)) for t in f.tolist() for p in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))}
        assert es & eset, f"boundary {b} has no edges"
    # positive orientation of every tetrahedron
    x = r["vertices"][tet["elements"][:, :4]]
    det = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
    assert (det > 0).all()


@pytest.mark.parametrize("name", ["a.msh", "b.msh"])
def test_mphtxt_roundtrip_3d(cube, name, tmp_path):
    d, pts, tets, attr, bnd, btag, L = cube
    out = tmp_path / "cube.mphtxt"
    info = cx.gmsh_to_mphtxt(d / name, out)
    r = cx.read_mphtxt(out)
    assert r["sdim"] == 3 and r["version"] == 4 and r["lowest"] == 0
    assert r["tags"][0] == "mesh1"
    assert len(r["vertices"]) == len(pts)
    assert len(r["types"]["tet"]["elements"]) == len(tets)
    # coordinates survive exactly (17 significant digits)
    assert np.array_equal(np.sort(r["vertices"], axis=0), np.sort(pts, axis=0))
    _check_partition_3d(r)
    sels = {s["label"]: s for s in r["selections"]}
    assert {"air", "slab", "core", "outer", "sym", "core_boundary", "exterior_boundary"} <= set(sels)
    # volume of every region from the written file equals the source mesh
    tet = r["types"]["tet"]
    x = r["vertices"][tet["elements"]]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    for lab, a in (("air", 1), ("slab", 2), ("core", 3)):
        doms = set(sels[lab]["entities"])
        v_out = vol[np.isin(tet["entities"], list(doms))].sum()
        xs = pts[tets[attr == a]]
        v_in = np.abs(np.einsum("ij,ij->i", np.cross(xs[:, 1] - xs[:, 0], xs[:, 2] - xs[:, 0]), xs[:, 3] - xs[:, 0])).sum() / 6
        assert v_out == pytest.approx(v_in, rel=1e-12)
        assert sum(dd["volume_m3"] for dd in info["domains"] if dd["region"] == lab) == pytest.approx(v_in, rel=1e-12)
    assert sum(dd["volume_m3"] for dd in info["domains"]) == pytest.approx(L ** 3)
    # tagged boundary groups: same triangles as the source
    tri = r["types"]["tri"]
    for lab, tg in (("outer", 901), ("sym", 910)):
        got = {tuple(sorted(r["vertices"][t].round(12).ravel().tolist()))
               for t in tri["elements"][np.isin(tri["entities"], sels[lab]["entities"])]}
        want = {tuple(sorted(pts[t].round(12).ravel().tolist())) for t in bnd[btag == tg]}
        assert got == want
    # the core surface is closed (no edges of its own): it must have been cut into >= 2 faces
    assert len(sels["core_boundary"]["entities"]) >= 2
    ent = json.loads(Path(str(out) + ".entities.json").read_text())
    assert ent["counts"]["tet"] == len(tets)
    assert all(len(b["probe_point"]) == 3 for b in ent["boundaries"])


def test_ascii_and_binary_readers_agree(cube):
    d = cube[0]
    a, b = cx.read_gmsh(d / "a.msh"), cx.read_gmsh(d / "b.msh")
    assert np.array_equal(a.nodes, b.nodes)
    assert a.physical_names == b.physical_names == NAMES3
    for (ta, pa, ga, ca), (tb, pb, gb, cb) in zip(sorted(a.blocks, key=lambda x: x[0]), sorted(b.blocks, key=lambda x: x[0])):
        assert ta == tb and np.array_equal(pa, pb) and np.array_equal(ca, cb)


def test_second_order_numbering(cube, tmp_path):
    """tet10 input -> COMSOL tet2: mid nodes follow the COMSOL numbering convention."""
    d, pts, tets, attr, bnd, btag, L = cube
    # add mid-edge nodes in gmsh tet10 order: (0,1),(1,2),(2,0),(3,0),(3,2),(3,1)
    P = [p for p in pts]
    mid = {}

    def m(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(P)
            P.append(0.5 * (pts[a] + pts[b]))
        return mid[k]
    t10 = np.array([list(t) + [m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0]), m(t[3], t[0]), m(t[3], t[2]), m(t[3], t[1])]
                    for t in tets])
    b6 = np.array([list(t) + [m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0])] for t in bnd])
    write_msh(tmp_path / "q.msh", np.array(P), [(11, t10, attr), (9, b6, btag)], NAMES3, binary=True)
    info = cx.gmsh_to_mphtxt(tmp_path / "q.msh", tmp_path / "q.mphtxt", order=2)
    assert info["order"] == 2
    r = cx.read_mphtxt(tmp_path / "q.mphtxt")
    assert set(r["types"]) == {"vtx", "edg2", "tri2", "tet2"}
    X = r["vertices"]
    pairs = {"edg2": [(0, 1)], "tri2": [(0, 1), (0, 2), (1, 2)],
             "tet2": [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]}
    for name, pp in pairs.items():
        el = r["types"][name]["elements"]
        nv = {"edg2": 2, "tri2": 3, "tet2": 4}[name]
        for k, (a, b) in enumerate(pp):
            assert np.allclose(X[el[:, nv + k]], 0.5 * (X[el[:, a]] + X[el[:, b]]))
    assert len(X) == len(P)
    # linear export of the same mesh drops the mid nodes
    info1 = cx.gmsh_to_mphtxt(tmp_path / "q.msh", tmp_path / "q1.mphtxt", order=1)
    assert info1["counts"]["mesh_vertices"] == len(pts)


def test_feature_angle_splits_box_walls(cube, tmp_path):
    d = cube[0]
    base = cx.gmsh_to_mphtxt(d / "a.msh", tmp_path / "x.mphtxt")
    sharp = cx.gmsh_to_mphtxt(d / "a.msh", tmp_path / "y.mphtxt", feature_angle=30)
    nb = lambda inf, lab: len(next(s for s in inf["selections"] if s["label"] == lab)["entities"])  # noqa: E731
    assert nb(sharp, "core_boundary") == 6          # a cube: six faces
    assert nb(sharp, "outer") > nb(base, "outer")
    _check_partition_3d(cx.read_mphtxt(tmp_path / "y.mphtxt"))


def test_mphtxt_roundtrip_2d_axisymmetric(tmp_path):
    """(r, z) triangle mesh with an axis, an outer boundary and a zero-thickness sheet (interior tagged edges)."""
    n = 6
    xs = np.linspace(0, 1, n + 1)
    pts = np.array([[x, y] for y in xs for x in xs])
    idx = lambda i, j: j * (n + 1) + i  # noqa: E731
    tris, attr = [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            lab = 2 if (2 <= i < 4 and 2 <= j < 4) else 1
            tris += [[a, b, c], [a, c, d]] if (i + j) % 2 else [[a, c, b], [a, d, c]]  # mixed orientation
            attr += [lab, lab]
    tris, attr = np.array(tris), np.array(attr)
    edges, etag = [], []
    for j in range(n):
        edges.append([idx(0, j), idx(0, j + 1)]); etag.append(900)                      # axis r = 0
        edges.append([idx(n, j), idx(n, j + 1)]); etag.append(901)
    for i in range(n):
        edges.append([idx(i, 0), idx(i + 1, 0)]); etag.append(901)
        edges.append([idx(i, n), idx(i + 1, n)]); etag.append(901)
    for i in range(1, 3):                                                               # a sheet inside the air
        edges.append([idx(i, 5), idx(i + 1, 5)]); etag.append(1000)
    names = {(2, 1): "air", (2, 2): "coil", (1, 900): "axis", (1, 901): "outer", (1, 1000): "sheet"}
    write_msh(tmp_path / "m.msh", pts, [(2, tris, attr), (1, np.array(edges), np.array(etag))], names)
    info = cx.gmsh_to_mphtxt(tmp_path / "m.msh", tmp_path / "m.mphtxt")
    assert info["axisymmetric"]
    r = cx.read_mphtxt(tmp_path / "m.mphtxt")
    assert r["sdim"] == 2 and set(r["types"]) == {"vtx", "edg", "tri"}
    tri, edg, vtx = r["types"]["tri"], r["types"]["edg"], r["types"]["vtx"]
    x = r["vertices"][tri["elements"]]
    det = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0])
    assert (det > 0).all()
    sels = {s["label"]: s for s in r["selections"]}
    assert {"air", "coil", "axis", "outer", "sheet", "coil_boundary"} <= set(sels)
    area = {}
    for lab in ("air", "coil"):
        msk = np.isin(tri["entities"], sels[lab]["entities"])
        area[lab] = 0.5 * det[msk].sum()
    assert area["coil"] == pytest.approx(4 / 36) and area["air"] == pytest.approx(32 / 36)
    # the sheet is one boundary between two points, the coil outline is closed (split into 2 boundaries)
    assert len(sels["sheet"]["entities"]) == 1
    assert len(sels["coil_boundary"]["entities"]) == 2
    vnodes = set(vtx["elements"][:, 0].tolist())
    for e in range(edg["entities"].max() + 1):
        segs = edg["elements"][edg["entities"] == e]
        deg = {}
        for a, b in segs.tolist():
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        assert sorted(v for v, dd in deg.items() if dd == 1 and v in vnodes) == sorted(v for v, dd in deg.items() if dd == 1)
        assert sum(1 for dd in deg.values() if dd == 1) == 2


# ------------------------------------------------------------------ expressions and fillets
def test_expression_translation():
    p = {"R": 2.0, "H": 3.0, "x": 0.5}
    for text in ["pow(R, 2) + H/2", "-2^2", "sqrt(R*H) - max(R, H)", "exp(-0.5)*R", "2*pi*R", "e^2", "mu0*R",
                 "atan2(H, R) + log10(100) + abs(-R)", "x*R"]:
        v, t = cx.comsol_expression(text, p)
        # evaluate the COMSOL text with Python semantics (^ -> **, parameter names, constants)
        env = {"pi": math.pi, "exp": math.exp, "sqrt": math.sqrt, "max": max, "atan2": math.atan2,
               "log10": math.log10, "abs": abs, "mu0_const": 4e-7 * math.pi, "R": 2.0, "H": 3.0, "par_x": 0.5}
        assert eval(t.replace("^", "**"), env) == pytest.approx(v), (text, t)
    assert cx.comsol_expression("-2^2", {})[0] == -4.0         # standard precedence
    assert cx.comsol_param_name("x") == "par_x" and cx.comsol_param_name("Rc") == "Rc"
    with pytest.raises(ValueError):
        cx.comsol_expression("foo(1)", {})


# (axionFEM text, value with standard precedence); x = 1.5, z = 0.3, s = 0.7
_PRECEDENCE_CASES = [
    ("-2^2", -4.0), ("(-2)^2", 4.0), ("-(2^2)", -4.0), ("3*-2^2", -12.0),
    ("-x^2", -2.25), ("(-x)^2", 2.25), ("-(x^2)", -2.25), ("x^-2", 1 / 2.25), ("-x^-2", -1 / 2.25),
    ("2^-2", 0.25), ("-2^-2", -0.25), ("2^3^2", 512.0), ("(2^3)^2", 64.0), ("-2^3^2", -512.0),
    ("2^-3^2", 2.0 ** -9), ("e^-x^2", math.exp(-2.25)), ("exp(-z^2/(2*s^2))", math.exp(-0.09 / 0.98)),
    ("1 - -x^2", 3.25), ("2^-x*3", 3 * 2 ** -1.5), ("max(-x^2, -3) - 1", -3.25),
    ("atan2(-z^2, -x) + x - z^2^-1", math.atan2(-0.09, -1.5) + 1.5 - 0.3 ** 0.5),
    ("pow(-x^2, 3) + 2*(x - z)^2", -(2.25 ** 3) + 2 * 1.2 ** 2), ("min(1, 2, -3^2)", -9.0),
]

_COMSOL_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\S))")


class _ComsolModel:
    """Pratt parser of exported COMSOL text under one precedence model.

    COMSOL documents '^' above unary minus but not the associativity of '^'.  ``unary_tighter`` lets unary minus
    bind tighter than '^' instead; ``left_assoc`` makes '^' left-associative.  The exporter's parentheses must
    make all four models agree.
    """

    def __init__(self, text, env, unary_tighter, left_assoc):
        self.toks = []
        i = 0
        while i < len(text):
            m = _COMSOL_TOKEN.match(text, i)
            if m is None:
                break
            i = m.end()
            if m.group(1):
                self.toks.append(("num", float(m.group(1))))
            elif m.group(2):
                self.toks.append(("name", m.group(2)))
            elif m.group(3):
                self.toks.append(("op", m.group(3)))
        self.k = 0
        self.env = env
        self.ut = unary_tighter
        self.la = left_assoc

    def peek(self):
        return self.toks[self.k] if self.k < len(self.toks) else ("end", None)

    def take(self):
        tok = self.peek()
        self.k += 1
        return tok

    def parse(self):
        v = self.binary(0)
        assert self.peek()[0] == "end", self.toks[self.k:]
        return v

    # binding powers: + - 10, * / 20, unary 30 (50 when tighter than ^), ^ 40
    def prefix(self):
        tok = self.take()
        if tok[0] == "num":
            return tok[1]
        if tok == ("op", "("):
            v = self.binary(0)
            assert self.take() == ("op", ")")
            return v
        if tok in (("op", "-"), ("op", "+")):
            v = self.binary(50 if self.ut else 30)
            return -v if tok[1] == "-" else v
        assert tok[0] == "name", tok
        if self.peek() == ("op", "("):
            self.take()
            args = [self.binary(0)]
            while self.peek() == ("op", ","):
                self.take()
                args.append(self.binary(0))
            assert self.take() == ("op", ")")
            return self.env[tok[1]](*args)
        return self.env[tok[1]]

    def binary(self, rbp):
        left = self.prefix()
        while True:
            tok = self.peek()
            if tok[0] != "op" or tok[1] not in "+-*/^":
                return left
            lbp = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 40}[tok[1]]
            if lbp <= rbp:
                return left
            self.take()
            if tok[1] == "^":
                left = left ** self.binary(lbp if self.la else lbp - 1)
            else:
                right = self.binary(lbp)
                ops = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "*": lambda p, q: p * q,
                       "/": lambda p, q: p / q}
                left = ops[tok[1]](left, right)


def _comsol_values(text, env):
    """The exported text's value under the four COMSOL precedence models."""
    return [_ComsolModel(text, env, ut, la).parse() for ut in (False, True) for la in (False, True)]


def _assert_parenthesized(t):
    """Structure that makes the text precedence-independent: '^' only between ')' and '(', every unary minus
    written '(-(', and every binary minus followed by '(' (number exponents such as 1e-2 aside)."""
    assert re.search(r"[^)]\^|\^[^(]", t) is None, t
    u = re.sub(r"(?<=[\d.])[eE][+-](?=\d)", "E", t)
    for m in re.finditer("-", u):
        k = m.start()
        unary = k == 0 or u[k - 1] in "(,+-*/^"
        if unary:
            assert u[k - 1:k + 2] == "(-(", (t, k)
        else:
            assert u[k + 1:k + 2] == "(", (t, k)


@pytest.mark.parametrize("text,want", _PRECEDENCE_CASES)
def test_expression_precedence_round_trip(text, want):
    """Exported COMSOL text evaluates to axionFEM's value under every COMSOL precedence model, and the C++ core
    (through af.Design parameters) agrees with the exporter."""
    p = {"x": 1.5, "z": 0.3, "s": 0.7}
    v, t = cx.comsol_expression(text, p)
    assert v == pytest.approx(want, rel=1e-14), (text, t)
    _assert_parenthesized(t)
    env = {"exp": math.exp, "max": lambda *a: max(a), "min": lambda *a: min(a), "atan2": math.atan2,
           "par_x": 1.5, "par_z": 0.3, "s": 0.7}
    for got in _comsol_values(t, env):
        assert got == pytest.approx(v, rel=1e-14), (text, t)
    d = af.template("core")
    for k, val in p.items():
        d.add_param(k, val)
    d.add_param("q_expr", text)
    assert d.resolved_params()["q_expr"] == pytest.approx(v, rel=1e-14), (text, t)


def test_comsol_models_detect_missing_parentheses():
    """The four-model check fails on unparenthesized text, so the round-trip test guards the parentheses."""
    env = {"par_x": 1.5}
    assert len(set(_comsol_values("-par_x^2", env))) == 2
    assert len(set(_comsol_values("2^3^2", env))) == 2
    with pytest.raises(AssertionError):
        _assert_parenthesized("-par_x^2")


def _core_value(text, params=None):
    d = af.template("core")
    for k, val in (params or {}).items():
        d.add_param(k, val)
    d.add_param("q_expr", text)
    return d.resolved_params()["q_expr"]


@pytest.mark.parametrize("text,params", [
    ("max(0.01*R)", {"R": 2.0}), ("min(x)", {"x": 1.5}), ("0.01*R + 1/(1/0)", {"R": 2.0}),
    ("exp(-1/0) + 1", {}), ("1/exp(1000) + 2", {}), ("mu0^2/mu0", {}), ("sin(1/mu0)", {}),
    ("(-8)^2^1", {}), ("0e-400 + 0.0e+999 + 2.2250738585072014e-308", {}), ("atan(1/0) - pi/2 + 1", {}),
])
def test_exporter_arithmetic_matches_core(text, params):
    """The exporter evaluates as the core does: variadic min/max, IEEE intermediates, CODATA mu0."""
    v, _ = cx.comsol_expression(text, params)
    assert v == pytest.approx(_core_value(text, params), rel=1e-14, abs=0.0)


@pytest.mark.parametrize("text,needle", [
    ("1.2.5", "malformed number '1.2.5'"), ("1..5", "malformed number '1..5'"), (".", "malformed number '.'"),
    ("1e400", "number '1e400' is out of the range"), ("1e-310", "number '1e-310' is out of the range"), ("1e-400", "number '1e-400' is out of the range"),
    ("min(1, sqrt(-1))", "min() argument 2 is not a number"), ("min(sqrt(-1), 1)", "min() argument 1 is not a number"),
    ("(" * 1000 + "2" + ")" * 1000, "nesting deeper than 1000 levels"),
    ("2" + "^1" * 2000, "nesting deeper than 1000 levels"), ("-" * 5000 + "2", "nesting deeper than 1000 levels"),
    ("(-8)^(1/3)", "non-finite"),
], ids=lambda x: x if len(x) < 40 else f"{x[:12]}...({len(x)} chars)")
def test_rejected_expressions_agree(text, needle):
    """The core and the exporter reject the same expressions; the core's error names the path and the token."""
    with pytest.raises(ValueError, match=re.escape(needle)):
        cx.comsol_expression(text, {})
    with pytest.raises(af.AxionFEMError) as err:
        _core_value(text)
    assert needle in str(err.value) and "parameters/q_expr" in str(err.value)


def test_untranslatable_parameter_falls_back_to_the_number(tmp_path):
    """A parameter the exporter cannot translate is written as the number the core resolved."""
    d = af.template("toroid")
    d.set(tw="max(0.01*R)")
    cx.export_comsol(d, tmp_path / "Toroid.java", format="java")
    src = (tmp_path / "Toroid.java").read_text()
    assert '"max((0.01)*(R))"' in src
    d.set(tw="0.01*R + 1/(1/0)")
    cx.export_comsol(d, tmp_path / "Toroid2.java", format="java")
    assert 'model.param().set("tw", ' in (tmp_path / "Toroid2.java").read_text()


def test_fillet_polygon_is_exact_arc():
    pts, deg, w = cx._fillet_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], [0.5] * 4)
    assert deg.count(2) == 4 and deg.count(1) == 4
    assert len(pts) == sum(deg) + 1 and len(w) == sum(deg) + len(deg)
    assert pts[0] == pts[-1]
    # quadratic segments: control point at the corner, weight cos(45 deg)
    k = 0
    for dgr in deg:
        if dgr == 2:
            assert w[k + 1] == pytest.approx(math.sqrt(0.5))
        k += dgr + 1


# ------------------------------------------------------------------ Java model files
RESOLVED_2D = {
    "name": "demo2d", "symmetry": "axisymmetric", "parameters": {"R": 1.0, "t": 0.1},
    "regions": [
        {"name": "coil", "kind": "vacuum", "shape": {"rect": {"r": [0.5, 1.0], "z": [-0.5, 0.5], "fillet": 0.05}}},
        {"name": "sheath", "kind": "superconductor",
         "shape": {"frame": {"r": [0.4, 1.1], "z": [-0.6, 0.6], "thickness": 0.1, "fillet": 0.2}}},
        {"name": "port", "kind": "vacuum",
         "shape": {"polygon": {"points": [[1.1, -0.1], [1.2, -0.1], [1.2, 0.1], [1.1, 0.1]], "fillet": 0.01}}},
        {"name": "ring", "kind": "conductor", "shape": {"union": [{"disk": {"center": [2, 0], "radius": 0.1}},
                                                                    {"polygon": {"points": [[2, 0], [2.3, 0], [2.3, 0.2]], "fillet": 0}}]}},
    ],
    "curves": [{"name": "sheet", "kind": "superconductor", "path": {"arc": {"center": [0, 0], "radius": 3, "angles": [-1, 1]}}}],
    "mesh": {"domain_radius": 5.0, "domain_center_z": 0.0},
}
RAW_2D = {"parameters": {"R": 1.0, "t": "R/10"},
          "regions": [{"name": "coil", "shape": {"rect": {"r": ["R/2", "R"], "z": [-0.5, 0.5], "fillet": 0.05}}}]}
RESOLVED_3D = {
    "name": "demo3d", "symmetry": "3d", "parameters": {},
    "regions": [
        {"name": "sheath", "kind": "superconductor", "solid": {"revolve": {
            "profile": {"frame": {"r": [1, 2], "z": [-1, 1], "thickness": 0.05, "fillet": 0.2}},
            "angles": [2, 358], "center": [0, 0, 0], "axis": [0, 0, 1]}}},
        {"name": "case", "kind": "conductor", "solid": {"difference": [
            {"extrude": {"polygon": [[2.5, -0.2], [2.9, -0.2], [2.9, 0.2], [2.5, 0.2]], "z": [-0.3, 0.3]}},
            {"cylinder": {"base": [2.7, 0, -0.1], "axis": [0, 0, 0.2], "radius": 0.05}}]}},
        {"name": "pin", "kind": "vacuum", "solid": {"cone": {"base": [0, 0, 1.5], "axis": [0, 1, 0], "r1": 0.1, "r2": 0.05},
                                                     "rotate": {"axis": [0, 0, 1], "angle_deg": 30, "origin": [0, 0, 0]},
                                                     "translate": [0, 0, 0.1]}},
        {"name": "lump", "kind": "vacuum", "solid": {"union": [{"sphere": {"center": [0, 3, 0], "radius": 0.2}},
                                                                {"torus": {"center": [0, 0, 2], "axis": [1, 0, 0], "major": 0.5, "minor": 0.1}},
                                                                {"box": {"min": [-1, -1, -3], "max": [1, 1, -2.5]}}]}},
        {"name": "below", "kind": "vacuum", "solid": {"box": {"min": [-1, -1, -3], "max": [1, 1, -2.6]}}},
    ],
    "mesh": {"domain_radius": 6.0, "symmetry": [{"name": "mid", "axis": "z", "side": 1, "type": "pmc"}]},
    "pickup": {"type": "port", "port": "pin", "conductor": ""},
}


def _javac(files, outdir):
    if not shutil.which("javac"):
        pytest.skip("no JDK (javac) on PATH")
    r = subprocess.run(["javac", "-d", str(outdir), "-sourcepath", str(STUBS)] + [str(f) for f in files],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_java_2d_and_3d_compile(tmp_path):
    s2 = cx.write_java(RESOLVED_2D, tmp_path / "Demo2D.java", raw=RAW_2D)
    s3 = cx.write_java(RESOLVED_3D, tmp_path / "Demo3D.java")
    src2 = (tmp_path / "Demo2D.java").read_text()
    src3 = (tmp_path / "Demo3D.java").read_text()
    assert "axisymmetric(true)" in src2 and '"R/2"' not in src2 and '"(R)/(2)"' in src2   # expressions kept
    assert 'model.param().set("par_t", "(R)/(10)"' in src2
    assert "BezierPolygon" in src2 and "Fillet" in src2 and "DifferenceSelection" in src2
    assert set(s2["selections"]) == {"coil", "sheath", "port", "ring", "air"}
    assert "Revolve" in src3 and "Extrude" in src3 and "Intersection" in src3 and "sym_mid_pmc" in src3
    assert any("omitted" in a for a in s3["approximations"])          # 'below' lies at z < 0: clipped away
    assert "below" not in s3["selections"] and "infinity_shell" not in s3["selections"]
    _javac([tmp_path / "Demo2D.java", tmp_path / "Demo3D.java"], tmp_path)
    assert (tmp_path / "Demo3D.class").is_file()


def test_java_large_model_is_split(tmp_path):
    regs = [{"name": f"b{i}", "kind": "vacuum", "solid": {"box": {"min": [i, 0, 0], "max": [i + 0.5, 1, 1]}}}
            for i in range(300)]
    cx.write_java({"name": "big", "symmetry": "3d", "regions": regs, "mesh": {"domain_radius": 500}},
                  tmp_path / "Big.java")
    src = (tmp_path / "Big.java").read_text()
    assert "public static Model run2(Model model)" in src
    _javac([tmp_path / "Big.java"], tmp_path)


def test_import_java_compiles(cube, tmp_path):
    cx.gmsh_to_mphtxt(cube[0] / "a.msh", tmp_path / "c.mphtxt")
    cx.write_import_java(tmp_path / "c.mphtxt", tmp_path / "ImportC.java", sdim=3)
    _javac([tmp_path / "ImportC.java"], tmp_path)


# ------------------------------------------------------------------ with the compiled core and mesher
def _have_mesher():
    from axionfem import _paths

    return _paths.mesher_path() is not None


def test_resolve_and_java_from_template(tmp_path):
    import axionfem as af
    from axionfem import _core

    d = af.template("toroid")
    res = json.loads(_core.resolve(json.dumps(d.doc)))
    assert [r["name"] for r in res["regions"]] and all("shape" in r for r in res["regions"])
    s = cx.export_comsol(d, tmp_path / "Toroid.java", format="java")
    assert s["dimension"] == 2 and "air" in s["selections"]
    src = (tmp_path / "Toroid.java").read_text()
    for k in d.params:
        assert cx.comsol_param_name(k) in src
    _javac([tmp_path / "Toroid.java"], tmp_path)


@pytest.mark.skipif(not _have_mesher(), reason="needs axf-mesher")
def test_export_design_mphtxt_2d(tmp_path):
    import axionfem as af

    d = af.template("toroid")
    d.doc.setdefault("mesh", {}).update({"size": 0.4, "surface_size": 0.1})
    s = cx.export_comsol(d, tmp_path / "toroid.mphtxt", format="mphtxt", workdir=tmp_path / "m")
    r = cx.read_mphtxt(tmp_path / "toroid.mphtxt")
    labels = {x["label"] for x in r["selections"]}
    assert {"axis", "outer", "air"} <= labels
    assert {reg["name"] for reg in d.doc["regions"]} <= labels
    assert Path(s["import_java"]).is_file()


SMALL3D = {
    "schema": "axionfem.design/1", "name": "small3d", "symmetry": "3d", "parameters": {"a": 0.5, "t": "a/5"},
    "regions": [
        {"name": "shell", "material": "copper", "solid": {"difference": [
            {"box": {"min": ["-a", "-a", "-a"], "max": ["a", "a", "a"]}},
            {"box": {"min": ["t-a", "t-a", "t-a"], "max": ["a-t", "a-t", "a-t"]}}]}},
        {"name": "gap", "material": "vacuum", "solid": {"cylinder": {"base": [0, 0, "a-t"], "axis": [0, 0, "t"], "radius": 0.1}}},
    ],
    "mesh": {"order": 1, "size": 0.25, "surface_size": 0.1},
}


@pytest.mark.skipif(not _have_mesher(), reason="needs axf-mesher")
def test_export_design_mphtxt_3d(tmp_path):
    import axionfem as af

    if not af.build_info().get("engine_3d"):
        pytest.skip("no 3D engine in this build")
    s = cx.export_comsol(SMALL3D, tmp_path / "small.mphtxt", workdir=tmp_path / "m")
    info = json.loads(Path(s["entities"]).read_text())
    vol = {}
    for dd in info["domains"]:
        vol[dd["region"]] = vol.get(dd["region"], 0) + dd["volume_m3"]
    gap = math.pi * 0.1 ** 2 * 0.1
    assert vol["shell"] == pytest.approx(1 - 0.8 ** 3 - gap, rel=1e-3)   # exact up to the faceted cylinder
    assert vol["gap"] == pytest.approx(gap, rel=0.05)
    assert {"shell", "gap", "air", "outer", "gap_boundary"} <= set(s["selections"])
    _check_partition_3d(cx.read_mphtxt(tmp_path / "small.mphtxt"))
    st = cx.export_comsol(SMALL3D, tmp_path / "small.step", format="step")
    solids = json.loads(Path(st["solids"]).read_text())["step"]["solids"]
    assert {x["region"] for x in solids} >= {"shell", "gap", "air"}


# ------------------------------------------------------------------ physics (--physics fullwave)
from axionfem import comsol_physics as cph  # noqa: E402

# a copper square frame read out across one box gap (as benchmarks/designs/stage1_twoport.json), with a dielectric
# block, a second (inactive) port and a PMC symmetry plane z = 0
FRAME_DOC = {
    "schema": "axionfem.design/1", "name": "frame_port", "symmetry": "3d",
    "materials": {"lossy": {"kind": "vacuum", "eps_r": 4.0, "tan_delta": 0.01, "sigma_S_m": 0.5}},
    "regions": [
        {"name": "loop", "material": "copper", "solid": {"difference": [
            {"box": {"min": [-0.3, -0.05, -0.3], "max": [0.3, 0.05, 0.3]}},
            {"box": {"min": [-0.2, -0.06, -0.2], "max": [0.2, 0.06, 0.2]}},
            {"box": {"min": [-0.04, -0.06, 0.19], "max": [0.04, 0.06, 0.31]}}]}},
        {"name": "gap0", "material": "vacuum", "solid": {"box": {"min": [-0.04, -0.05, 0.2], "max": [0.04, -0.005, 0.3]}}},
        {"name": "gap1", "material": "vacuum", "solid": {"box": {"min": [-0.04, 0.005, 0.2], "max": [0.04, 0.05, 0.3]}}},
        {"name": "block", "material": "lossy", "solid": {"box": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}},
    ],
    "pickup": {"type": "port", "port": "gap0", "toward": [1, 0, 0]},
    "fullwave": {"frequencies": [1e6, 1.2e7], "boundary": "pec",
                 "ports": [{"name": "second", "region": "gap1", "toward": [1, 0, 0]}],
                 "source": {"kind": "uniform", "B_T": [1, 0.2, 0.3]}},
    "mesh": {"domain_radius": 0.7, "symmetry": [{"name": "mid", "axis": "z", "side": 1, "type": "pmc"}]},
}


def _frame_resolved():
    """What _core.resolve returns for FRAME_DOC (numbers only), so that these tests do not need the core."""
    mats = {k: dict(v) for k, v in cph._BUILTIN.items()}
    mats["lossy"] = dict(FRAME_DOC["materials"]["lossy"])
    regs = [dict(r, kind=mats[r["material"]]["kind"]) for r in FRAME_DOC["regions"]]
    return {"name": "frame_port", "symmetry": "3d", "parameters": {}, "regions": regs, "materials": mats,
            "mesh": dict(FRAME_DOC["mesh"]), "pickup": FRAME_DOC["pickup"]}


def _eval_comsol(expr, x, y, z, je_scale=1.0):
    """Evaluate one of the generated COMSOL expressions at a point (test helper)."""
    py = expr.replace("/1[m]", "").replace("^", "**").replace("&&", " and ")
    env = {"x": x, "y": y, "z": z, "sqrt": math.sqrt, "min": min, "max": max, "pi": math.pi, "Je_scale": je_scale,
           "if_": lambda c, a, b: a if c else b, "atan2": math.atan2, "mod": lambda a, b: a % b}
    py = re.sub(r"\bif\(", "if_(", py)
    return eval(py, {"__builtins__": {}}, env)  # noqa: S307 (generated by the exporter itself)


def test_physics_java_features_and_compile(tmp_path):
    spec = cph.physics_spec(FRAME_DOC, _frame_resolved())
    s = cx.write_java(_frame_resolved(), tmp_path / "FramePort.java", raw=FRAME_DOC, physics=spec)
    src = (tmp_path / "FramePort.java").read_text()
    G = 'model.component("comp1").geom("geom1")'
    P = 'model.component("comp1").physics("emw")'

    def tag_of(label):  # geometry selection feature (difsel, unisel, adjsel, boxsel) with this label
        import re
        m = re.search(r'feature\("([a-z]+sel\d+)"\)\.label\(' + re.escape(json.dumps(label)) + r"\)", src)
        assert m, label
        return m.group(1)

    def named(feature):
        import re
        m = re.search(re.escape(f'{P}.feature("{feature}").selection().named("') + r'geom1_(\w+)"', src)
        assert m, feature
        return m.group(1)

    assert f'model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");' in src
    emw = tag_of("emw_domains")
    assert f'{P}.selection().named("geom1_{emw}")' in src
    # emw domains: everything but the copper loop
    ins = src.split(f'{G}.feature("{emw}").set("input", new String[]{{')[1].split("}")[0]
    assert sorted(json.loads("[" + ins + "]")) == sorted(tag_of(n) for n in ("gap0", "gap1", "block", "air"))
    # PEC on the conductor walls, PMC on the symmetry plane
    assert named("pec2") == tag_of("pec_walls")
    assert f'{P}.create("pmc1", "PerfectMagneticConductor", 2);' in src and named("pmc1") == tag_of("sym_mid_pmc")
    # ports: lport1 on the outer face (y = -0.05) of gap0, lport2 on gap1 and inactive
    b1, b2 = named("lport1"), named("lport2")
    assert b1 == tag_of("port_pickup") and b2 == tag_of("port_second")
    assert f'{G}.feature("{b1}").set("ymin", "-0.050001")' in src.replace("-0.050001000000000004", "-0.050001")
    assert f'{G}.feature("{b1}").set("zmin", "0.199999")' in src            # clipped by z >= 0: unchanged here
    assert f'{P}.feature("lport2").active(false);' in src and f'{P}.feature("lport1").active(false);' not in src
    assert src.count('"LumpedPort", 2);') == 2
    # source on the emw domains, Je = Je_scale * B0
    assert named("ecd1") == emw
    assert '.set("Je", new String[]{"Je_scale*(1)", "Je_scale*(0.2)", "Je_scale*(0.3)"})' in src
    # material of the lossy block: sigma + omega eps0 eps_r tan(delta)
    assert '"0.5+2*pi*freq*epsilon0_const*4*0.01"' in src
    # study, solver (template sequence), exports and the two runs
    assert '"Frequency");' in src and 'set("plist", "1,12");' in src and 'set("punit", "MHz");' in src
    assert 'model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"1,12"});' in src
    assert "real(comp1.emw.Zport_1)" in src and "imag(comp1.emw.Vport_1)" in src
    assert "runImpedance(model)" in src and "runVoltage(model)" in src
    assert "UNVERIFIED" in src and "infinity_shell" not in s["selections"]
    assert s["physics"]["ports"][0]["selection"].startswith("face y = -0.05")
    _javac([tmp_path / "FramePort.java"], tmp_path)


def test_physics_source_normalization():
    """Je = Je_scale * B0 with B0 exactly axionFEM's source field (kappa = 1 A/m^2 per tesla)."""
    # uniform prescribed source
    u = cph.source_field(FRAME_DOC)
    assert u["exported"] and [float(_eval_comsol(e, 0.1, 0.2, 0.3)) for e in u["B_T"]] == [1.0, 0.2, 0.3]
    assert u["B_ref_T"] == pytest.approx(math.sqrt(1 + 0.04 + 0.09))
    # Biot-Savart segments: axionFEM's kernel (1e-7 = mu0/4pi) vs the textbook form mu0 I/(4 pi rho)(cos a1 - cos a2)
    segs = [[0, 0, -1, 0, 0, 1, 3.0], [0.5, 0, 0, 0.5, 0.4, 0.2, -2.0]]
    doc = dict(FRAME_DOC, fullwave=dict(FRAME_DOC["fullwave"], source={"kind": "segments", "segments_m_A": segs}))
    sg = cph.source_field(doc)
    assert sg["exported"] and sg["kind"] == "segments"
    p = np.array([0.13, -0.21, 0.37])
    ref = np.zeros(3)
    for s in segs:
        a, b, I = np.array(s[:3], float), np.array(s[3:6], float), s[6]
        d = (b - a) / np.linalg.norm(b - a)
        foot = a + d * np.dot(p - a, d)
        rho = np.linalg.norm(p - foot)
        c1, c2 = np.dot(p - a, d) / np.linalg.norm(p - a), np.dot(p - b, d) / np.linalg.norm(p - b)
        ref += 1e-7 * I / rho * (c1 - c2) * np.cross(d, (p - foot) / rho)
    got = np.array([_eval_comsol(e, *p) for e in sg["B_T"]])
    assert np.allclose(got, ref, rtol=1e-12, atol=0)
    # long-wire limit: B = mu0 I / (2 pi rho), azimuthal
    long = cph.source_field(dict(doc, fullwave={"source": {"kind": "segments", "segments_m_A": [[0, 0, -1e4, 0, 0, 1e4, 5.0]]}}))
    B = np.array([_eval_comsol(e, 0.25, 0.0, 0.1) for e in long["B_T"]])
    assert B[1] == pytest.approx(2e-7 * 5.0 / 0.25, rel=1e-9) and abs(B[0]) < 1e-15 and abs(B[2]) < 1e-15
    # too many segments: not exported, with a reason
    many = cph.source_field(dict(doc, fullwave={"source": {"kind": "segments", "segments_m_A": [segs[0]] * 101}}))
    assert not many["exported"] and "NOT exported" in many["warnings"][0]
    # ideal toroidal field of a poloidal frame winding: mu0 N I / (2 pi rho) phi-hat inside the frame's hole
    tor = {"symmetry": "3d", "windings": [{"region": "w", "type": "poloidal", "ampere_turns": 1e6}],
           "regions": [{"name": "w", "solid": {"revolve": {"profile": {"frame": {"r": [0.9, 2.1], "z": [-1.1, 1.1], "thickness": 0.1}},
                                                            "angles": [0, 360], "center": [0, 0, 0], "axis": [0, 0, 1]}}}]}
    t = cph.source_field(tor)
    assert t["exported"] and t["kind"] == "toroidal"
    for (x, y, z) in [(1.5, 0.0, 0.0), (0.0, 1.2, 0.5), (-1.0, -1.0, -0.9)]:
        rho = math.hypot(x, y)
        Bt = np.array([_eval_comsol(e, x, y, z) for e in t["B_T"]])
        assert np.allclose(Bt, cph.MU0 * 1e6 / (2 * math.pi * rho) * np.array([-y / rho, x / rho, 0]), rtol=1e-12)
    assert np.allclose([_eval_comsol(e, 3.0, 0.0, 0.0) for e in t["B_T"]], 0)     # outside the winding
    assert np.allclose([_eval_comsol(e, 1.5, 0.0, 1.05) for e in t["B_T"]][1], cph.MU0 * 1e6 / (2 * math.pi * 1.5) / 2)
    assert t["B_ref_T"] == pytest.approx(cph.MU0 * 1e6 / (2 * math.pi * 1.0))
    # toroidal_box prescribed source (cpp/src/prescribed_source.cpp): B0 phi_hat in the closed annular box, 0 outside
    box = {"kind": "toroidal_box", "B0_T": 10.0, "r_m": [0.525, 0.975], "z_m": [-0.225, 0.225]}
    for prof, scale in (("uniform", lambda rho: 1.0), ("inverse_r", lambda rho: 0.525 / rho)):
        tb = cph.source_field(dict(FRAME_DOC, fullwave=dict(FRAME_DOC["fullwave"], source=dict(box, profile=prof))))
        assert tb["exported"] and tb["kind"] == "toroidal_box" and tb["B_ref_T"] == 10.0
        for (x, y, z) in [(0.7, 0.0, 0.0), (0.0, -0.9, 0.2), (-0.6, 0.5, -0.1)]:
            rho = math.hypot(x, y)
            Bt = np.array([float(_eval_comsol(e, x, y, z)) for e in tb["B_T"]])
            assert np.allclose(Bt, 10.0 * scale(rho) * np.array([-y / rho, x / rho, 0]), rtol=1e-12)
        for (x, y, z) in [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.7, 0.0, 0.3)]:
            assert np.allclose([float(_eval_comsol(e, x, y, z)) for e in tb["B_T"]], 0)
    sec = cph.source_field(dict(FRAME_DOC, fullwave=dict(FRAME_DOC["fullwave"], source=dict(box, phi_deg=[0, 355]))))
    g, inn = math.radians(357.5), math.radians(300)
    assert np.allclose([float(_eval_comsol(e, 0.75 * math.cos(g), 0.75 * math.sin(g), 0)) for e in sec["B_T"]], 0)
    assert np.allclose([float(_eval_comsol(e, 0.75 * math.cos(inn), 0.75 * math.sin(inn), 0)) for e in sec["B_T"]],
                       [-10 * math.sin(inn), 10 * math.cos(inn), 0])
    # windings without a closed form: not exported, with a reason
    sol = cph.source_field(dict(tor, windings=[{"region": "w", "type": "azimuthal", "ampere_turns": 1}]))
    assert not sol["exported"] and "static solve" in sol["warnings"][0]


def test_physics_open_circuit_conversion():
    Z, Zref, Voc = 2.0 + 35.2j, 50.0, 1e-3 + 2e-3j
    Vload = Voc * Zref / (Zref + Z)                                         # Thevenin source loaded by Z_ref
    assert cph.open_circuit_voltage(Vload, Z, Zref) == pytest.approx(Voc)


def test_physics_import_route(cube, tmp_path):
    """mphtxt route: Explicit selections from the entity map; the port sits on one face of the box gap."""
    d = cube[0]
    info = cx.gmsh_to_mphtxt(d / "a.msh", tmp_path / "c.mphtxt", feature_angle=40)
    doc = {"symmetry": "3d", "regions": [{"name": "slab", "material": "copper", "solid": {"box": {"min": [0, 0, 0], "max": [1, 3, 3]}}},
                                         {"name": "core", "material": "vacuum", "solid": {"box": {"min": [1, 1, 1], "max": [2, 2, 2]}}}],
           "pickup": {"type": "port", "port": "core", "toward": [0, 0, 1]},
           "fullwave": {"frequencies": [1e6], "boundary": "pec", "source": {"kind": "uniform", "B_T": [0, 0, 1]}},
           "mesh": {"symmetry": [{"name": "sym", "axis": "z", "side": 1, "type": "pmc"}]}}
    spec = cph.physics_spec(doc)
    r = cx.write_import_java(tmp_path / "c.mphtxt", tmp_path / "ImpPhys.java", sdim=3, physics=spec, entities=info)
    sel = r["physics_selections"]
    by_index = {b["index"] + 1: b for b in info["boundaries"]}
    port = [by_index[i] for i in sel["port_pickup"]]
    # toward z; the thinnest other axis is a tie (x, y: 1 m) -> x, on the side farther from the origin: x = 2
    assert port and all(b["bbox"][0][0] == pytest.approx(2) and b["bbox"][1][0] == pytest.approx(2) for b in port)
    gap_dom = {dd["index"] for dd in info["domains"] if dd["region"] == "core"}
    wall_dom = {dd["index"] for dd in info["domains"] if dd["region"] == "slab"}
    assert all(set(b["domains"]) & gap_dom for b in port)
    assert set(sel["emw_domains"]) == {dd["index"] for dd in info["domains"]} - wall_dom
    assert gap_dom <= set(sel["emw_domains"])
    pmc = [by_index[i] for i in sel["pmc_symmetry"]]
    assert pmc and all(b["group"] == "sym" for b in pmc)
    pec = [by_index[i] for i in sel["pec_walls"]]
    assert any(set(b["domains"]) & wall_dom for b in pec) and any(b["group"] == "outer" for b in pec)
    src = (tmp_path / "ImpPhys.java").read_text()
    assert '"Explicit");' in src and 'mesh("mesh1").create("imp1", "Import")' in src and "runVoltage(model)" in src
    _javac([tmp_path / "ImpPhys.java"], tmp_path)


def test_physics_rejects_what_it_cannot_express():
    with pytest.raises(ValueError, match="3D designs only"):
        cph.physics_spec({"regions": [], "pickup": {"type": "port", "port": "g"}, "fullwave": {"frequencies": [1]}})
    with pytest.raises(ValueError, match="'port' pickup"):
        cph.physics_spec({"symmetry": "3d", "regions": [], "pickup": {"type": "azimuthal", "conductor": "s"}})
    with pytest.raises(ValueError, match="mqs"):
        cph.physics_spec(FRAME_DOC, physics="mqs")
    with pytest.raises(ValueError, match="frequencies"):
        cph.physics_spec(dict(FRAME_DOC, fullwave={}))


@pytest.mark.skipif(not _have_mesher(), reason="needs axf-mesher")
def test_physics_export_from_design(tmp_path):
    """export_comsol(..., physics='fullwave') on a real design, both routes (needs the core and mesher)."""
    import axionfem as af

    if not af.build_info().get("engine_3d"):
        pytest.skip("no 3D engine in this build")
    doc = json.loads((HERE.parent / "benchmarks/designs/stage1_twoport.json").read_text())
    doc["mesh"].update({"size": 0.4, "surface_size": 0.2})
    sj = cx.export_comsol(doc, tmp_path / "TwoPort.java", format="java", physics="fullwave")
    assert sj["physics"]["ports"][0]["selection"].startswith("face y = -0.05")
    _javac([tmp_path / "TwoPort.java"], tmp_path)
    sm = cx.export_comsol(doc, tmp_path / "two_port.mphtxt", format="mphtxt", workdir=tmp_path / "m", physics="fullwave")
    assert sm["physics"]["ports"][0]["selection"].startswith("face y = -0.05")
    _javac([Path(sm["import_java"])], tmp_path)
    with pytest.raises(ValueError, match="step"):
        cx.export_comsol(doc, tmp_path / "x.step", format="step", physics="fullwave")


def test_core_validates_toroidal_box_phi_deg():
    """The compiled core (what axionfem.Design / solve use) accepts the toroidal_box sector key phi_deg and rejects a
    bad one. A stale _core built before phi_deg existed fails here with '/fullwave/source: unknown key phi_deg'."""
    _core = pytest.importorskip("axionfem._core")
    doc = json.loads((Path(__file__).resolve().parents[1] / "examples/comsol_toroid/comsol_toroid_lossless.json").read_text())
    assert doc["fullwave"]["source"]["kind"] == "toroidal_box"
    doc["fullwave"]["source"]["phi_deg"] = [0, 355]
    assert json.loads(_core.validate(json.dumps(doc)))["ok"]
    doc["fullwave"]["source"]["phi_deg"] = [10, 5]
    with pytest.raises(Exception, match="phi_deg"):
        _core.validate(json.dumps(doc))


def test_physics_warns_about_wall_surface_resistance():
    """Conductor walls are exported as PEC (lossless). A wall material with rf data (the builtin copper: anomalous
    skin) is warned about: COMSOL's impedance BC is classical skin and is not exported."""
    spec = cph.physics_spec(FRAME_DOC, _frame_resolved())
    w = [x for x in spec["warnings"] if "rf surface-resistance data" in x]
    assert len(w) == 1 and "'copper'" in w[0] and "model 'anomalous'" in w[0] and "PerfectElectricConductor" in w[0]
    assert any("rf surface-resistance data" in x for x in cph.physics_notes(spec) if x.startswith("WARNING"))
    doc = json.loads(json.dumps(FRAME_DOC))
    doc["materials"]["pec"] = {"kind": "conductor"}
    for r in doc["regions"]:
        if r["material"] == "copper":
            r["material"] = "pec"
    res = _frame_resolved()
    res["materials"]["pec"] = {"kind": "conductor"}
    res["regions"] = [dict(r, material="pec") if r["material"] == "copper" else r for r in res["regions"]]
    assert not [x for x in cph.physics_spec(doc, res)["warnings"] if "rf surface-resistance data" in x]


def test_physics_wall_warning_quotes_the_rs_underestimate():
    """The percentages in the PEC-wall warning are 1 - R_skin/R_RS (how much a
    classical impedance BC underestimates the builtin copper R_s), computed here from the same rf blocks."""
    from axionfem import cavity as cv
    w = [x for x in cph.physics_spec(FRAME_DOC, _frame_resolved())["warnings"] if "rf surface-resistance data" in x][0]
    quoted = re.search(r"copper RRR 100: by ([\d.]+) % at 1 MHz, ([\d.]+) % at 100 MHz, ([\d.]+) % at 1 GHz", w)
    assert quoted, w
    cu = {"model": "anomalous", "rho_ohm_m": 1.7e-10, "rho_l_ohm_m2": 6.6e-16, "specularity": 0, "T_K": 0.03}
    for q, f in zip(quoted.groups(), (1e6, 1e8, 1e9)):
        under = 100 * (1 - cv.skin_Rs(f, 1.7e-10) / cv.surface_resistance(cu, f))
        assert float(q) == pytest.approx(under, abs=0.5), (f, under)
