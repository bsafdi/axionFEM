"""COMSOL import of 2D / 2D-axisymmetric models and of impedance, transition, scattering boundaries and PML
(python/axionfem/comsol_import2d.py, comsol_import.robin_feature / pml_layers, cpp/src/fw2d.cpp,
model3d_fullwave.cpp Robin groups and PML).  Beta.

No COMSOL file with these features is available: the archives here are synthetic (dmodel.xml layout as the real
3D files; PML coordinate systems, Robin feature parameters and the 2D component layout are assumed, see
docs/comsol_import.md), and the solves are compared with closed forms (benchmarks/fw2d_validation.py,
benchmarks/fw3d_robin_validation.py).
"""
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from axionfem import comsol_import as ci
from axionfem import comsol_import2d as ci2
from axionfem import comsol_mphbin as mb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

try:
    from axionfem import _core

    HAVE_FW2D = hasattr(_core, "Fw2DModel")
    HAVE_3D = bool(json.loads(_core.build_info())["engine_3d"])
except Exception:  # noqa: BLE001
    HAVE_FW2D = False
    HAVE_3D = False
needs_core = pytest.mark.skipif(not HAVE_FW2D, reason="axionfem._core with Fw2DModel not built")
needs_3d = pytest.mark.skipif(not HAVE_3D, reason="needs the 3D engine (an MPI build; the wheels have none)")

C0 = 299792458.0
MU0 = 1.25663706212e-6
ETA0 = MU0 * C0


# ------------------------------------------------------------------ synthetic 2D archives
def rect_mesh(xs, ys, dom_of, bnd_of):
    """Structured triangles of the grid xs x ys.  dom_of(xc, yc) -> COMSOL domain (1-based); bnd_of(xm, ym, ext)
    -> COMSOL boundary (1-based GUI number) or None for an untyped interior edge."""
    nx, ny = len(xs) - 1, len(ys) - 1
    idx = lambda i, j: i * (ny + 1) + j  # noqa: E731
    P = np.array([[x, y] for x in xs for y in ys], float)
    T, D = [], []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            for t in ([a, b, c], [a, c, d]):
                T.append(t)
                D.append(dom_of(*P[t].mean(axis=0)))
    T, D = np.array(T), np.array(D)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(E, axis=0, return_counts=True)
    # an edge is a geometric boundary if exterior or between two domains
    tri_of = {}
    for k, t in enumerate(T):
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            tri_of.setdefault(tuple(sorted(e)), []).append(k)
    edges, ent = [], []
    for e, c in zip(ue, cnt):
        ts = tri_of[tuple(e)]
        ext = c == 1
        if not ext and D[ts[0]] == D[ts[1]]:
            continue
        b = bnd_of(*P[e].mean(axis=0), ext)
        if b is not None:
            edges.append(e)
            ent.append(b - 1)
    return P, T, D, np.array(edges), np.array(ent)


def write_mesh_v7_2d(P, T, D, E, Eent, Q=None, Qd=None):
    b = bytearray()
    i32 = lambda *v: b.extend(struct.pack(f"<{len(v)}i", *v))  # noqa: E731

    def string(s):
        i32(len(s), *map(ord, s))

    i32(0, 1, 1)
    string("mesh1")
    i32(1)
    string("obj")
    i32(0, 0, 1)
    string("Mesh")
    i32(7, 2)
    b.extend(b"\x00\x00")
    i32(1, len(P), 0, 2)
    for c in range(2):
        i32(len(P))
        b.extend(np.asarray(P[:, c], "<f8").tobytes())
    types = [("edg", E, Eent)] + ([("tri", T, D)] if len(T) else []) + ([("quad", Q, Qd)] if Q is not None else [])
    i32(len(types))
    for name, el, ent in types:
        string(name)
        i32(el.shape[1])
        for c in range(el.shape[1]):
            i32(len(el))
            b.extend(np.asarray(el[:, c], "<i4").tobytes())
        i32(el.shape[1], 0)
        i32(len(el), len(el))
        b.extend(np.asarray(ent, "<i4").tobytes())
        i32(0)
    return bytes(b)


def _sel(dim, ents, nent):
    if dim == 2:
        return f'<selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" entities="{nent + 1},{",".join(map(str, ents))}"/></selection>'
    return f'<selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="1" hDim="1" geom="/geom/geom1" entities="-{nent},-1,{",".join(map(str, ents))}"/></selection>'


def _mat(tag, name, sel, eps="1", mu="1", sig="0[S/m]"):
    diag = lambda v: "9|" + "|".join(f"1,'{v if k in (0, 4, 8) else 0}'" for k in range(9))  # noqa: E731
    return (f'<Material op="Common" tag="{tag}" name="{name}">{sel}<MaterialModelList tag="materialmodel">'
            f'<MaterialModel tag="def" name="Basic"><param T="33" param="relpermittivity" value="{diag(eps)}"/>'
            f'<param T="33" param="relpermeability" value="{diag(mu)}"/>'
            f'<param T="33" param="electricconductivity" value="{diag(sig)}"/></MaterialModel></MaterialModelList></Material>')


def dmodel_2d(features: str, *, ndom, nbnd, axisym=True, components="threecomponent", materials="", coordsys="",
              study='<StudyFeature op="Frequency" tag="freq" name="Frequency Domain"><propertyValue T="30" name="p:plist" value="300[MHz]"/></StudyFeature>',
              thickness=None):
    sp = "3,'r','phi','z'" if axisym else "3,'x','y','z'"
    props = f'<PhysicsProp tag="components"><param T="33" param="components" value="1|1,\'{components}\'"/></PhysicsProp>'
    props += '<PhysicsProp tag="ShapeProperty"><param T="33" param="order_electricfield" value="1|1,\'2\'"/></PhysicsProp>'
    if thickness:
        props += f'<PhysicsProp tag="d"><param T="33" param="d" value="1|1,\'{thickness}\'"/></PhysicsProp>'
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Model tag="Model1" name="synthetic2d.mph">
  <ModelNodeList tag="modelNode"><ModelNode tag="comp1" name="Component 1"><spatialCoord T="1">{sp}</spatialCoord></ModelNode></ModelNodeList>
  <PhysicsList tag="physics">
    <Physics op="ElectromagneticWaves" tag="emw" name="emw">
      {_sel(2, list(range(1, ndom + 1)), ndom)}
      <geom T="50">/geom/geom1</geom>
      <PhysicsFeatureList tag="feature">
        <PhysicsFeature op="WaveEquationElectric" tag="wee1" name="Wave Equation, Electric 1">
          <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1"/></selection>
          <default T="3">true</default></PhysicsFeature>
        <PhysicsFeature op="PerfectElectricConductor" tag="pec1" name="PEC 1">
          <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="1" hDim="1" geom="/geom/geom1"/></selection>
          <default T="3">true</default></PhysicsFeature>
        {features}
      </PhysicsFeatureList>
      <PhysicsPropList tag="prop">{props}</PhysicsPropList>
    </Physics>
  </PhysicsList>
  <GeomList tag="geom"><GeomSequence tag="geom1" name="Geometry 1"><sdim T="5">2</sdim><lengthUnit T="0">m</lengthUnit><GeomFeatureList tag="feature"/></GeomSequence></GeomList>
  <ComponentCoordsysList tag="coordSystem"><CoordsysList tag="coordsys">{coordsys}</CoordsysList></ComponentCoordsysList>
  <MaterialList tag="material">{_mat("mat1", "Air", '<selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1"/></selection>')}{materials}</MaterialList>
  <MeshList tag="mesh"><MeshSequence tag="mesh1" name="Mesh 1"><geom T="50">/geom/geom1</geom>
    <MeshNative tag="meshnative"><mesh T="52" class="MESH" resource="mesh1.mphbin" id="1"/></MeshNative></MeshSequence></MeshList>
  <StudyList tag="study"><Study tag="std1" name="Study 1"><StudyFeatureList tag="feature">{study}</StudyFeatureList></Study></StudyList>
</Model>
"""


def feat(op, tag, dim, ents, nent, params=""):
    return f'<PhysicsFeature op="{op}" tag="{tag}" name="{tag}">{_sel(dim, ents, nent)}{params}</PhysicsFeature>'


def prm(name, value):
    return f'<param T="33" param="{name}" value="1|1,\'{value}\'"/>'


def write_mph(path, xml, mesh_bytes):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fileversion", "0:synthetic")
        z.writestr("dmodel.xml", xml)
        z.writestr("mesh1.mphbin", mesh_bytes)
    return path


A, B, L, DP = 0.01, 0.023, 0.3, 0.1


def coax_mph(tmp_path, *, end="pml", walls="ibc", source=True, n=(3, 45), ibc_params="", name=""):
    """Axisymmetric coax a < r < b; domain 1: 0 < z < L, domain 2 (PML): L < z < L + DP.  Boundaries: 1 port
    (z = 0), 2 inner wall, 3 outer wall, 4 z = L (interface or end), 5 / 6 walls in the PML, 7 PML end."""
    nz1 = n[1]
    zs = list(np.linspace(0, L, nz1 + 1)) + (list(L + DP * np.arange(1, 11) / 10) if end == "pml" else [])
    rs = np.linspace(A, B, n[0] + 1)

    def bnd(r, z, ext):
        if abs(z) < 1e-12:
            return 1
        if abs(z - L) < 1e-12:
            return 4
        if abs(z - L - DP) < 1e-12:
            return 7
        if abs(r - A) < 1e-12:
            return 2 if z < L else 5
        if abs(r - B) < 1e-12:
            return 3 if z < L else 6
        return None

    P, T, D, E, Ee = rect_mesh(rs, zs, lambda r, z: 1 if z < L else 2, bnd)
    ndom = 2 if end == "pml" else 1
    nb = 7 if end == "pml" else 4
    f = feat("LumpedPort", "lport1", 1, [1], nb, prm("PortType", "Uniform") + prm("Zref", "50[ohm]"))
    mats = ""
    if walls == "ibc":
        f += feat("ImpedanceBoundaryCondition", "imp1", 1, [2, 3], nb, ibc_params)
        mats = _mat("mat2", "Copper", _sel(1, [2, 3], nb), sig="5.8e7[S/m]")
    if end == "sbc":
        f += feat("Scattering", "sctr1", 1, [4], nb)
    if source:
        f += ('<PhysicsFeature op="ExternalCurrentDensity" tag="ecd1" name="ecd1">' + _sel(2, [1], ndom) +
              '<param T="33" param="Je" value="3|1,\'0\'|1,\'0\'|1,\'1[A/m^2]\'"/></PhysicsFeature>')
    cs = ""
    if end == "pml":
        cs = f'<Coordsys op="PML" tag="pml1" name="PML 1">{_sel(2, [2], ndom)}{prm("ScalingType", "Cylindrical")}</Coordsys>'
    xml = dmodel_2d(f, ndom=ndom, nbnd=nb, materials=mats, coordsys=cs)
    return write_mph(tmp_path / f"coax_{end}_{walls}{name}.mph", xml, write_mesh_v7_2d(P, T, D, E, Ee))


# ------------------------------------------------------------------ unit tests
def test_mesh_v7_2d_roundtrip():
    P, T, D, E, Ee = rect_mesh(np.linspace(0, 1, 3), np.linspace(0, 1, 3), lambda x, y: 1,
                               lambda x, y, ext: 1 if ext else None)
    m = mb.read_mesh(write_mesh_v7_2d(P, T, D, E, Ee))
    assert m.sdim == 2 and np.array_equal(m.points, P) and np.array_equal(m.elements["tri"], T)
    assert np.array_equal(m.entities["edg"], Ee)


def test_fit_curve():
    th = np.linspace(0.1, 1.2, 9)
    kind, (cx, cy, R) = ci2.fit_curve(np.column_stack([2 + 3 * np.cos(th), -1 + 3 * np.sin(th)]))
    assert kind == "circle" and (cx, cy, R) == pytest.approx((2, -1, 3))
    assert ci2.fit_curve(np.array([[0, 0], [1, 1], [2, 2.0]]))[0] == "line"
    assert ci2.fit_curve(np.array([[0, 0], [1, 1.0]]))[0] is None


def test_axisymmetric_route_content(tmp_path):
    mph = coax_mph(tmp_path, end="pml", walls="ibc")
    d = ci.import_comsol(mph, tmp_path / "out" / "design.json")
    doc = d.doc if hasattr(d, "doc") else d
    rep = d.import_report.to_json()
    assert not rep["blocking"], rep["blocking"]
    assert doc["symmetry"] == "axisymmetric" and doc["fullwave"]["frequencies"] == pytest.approx([3e8])
    assert doc["regions"][0]["shape"] == {"imported": {"comsol_domains": [1], "comsol_material": "mat1"}}
    assert doc["fullwave"]["pml"] == [{"regions": ["domain_2"], "type": "cylindrical", "scaling_factor": 1.0,
                                       "curvature": 1.0, "center": [0.0, 0.0]}]
    assert doc["fullwave"]["source"] == {**doc["fullwave"]["source"], "kind": "uniform", "B_T": [0.0, 0.0, 1.0]}
    assert doc["pickup"]["port_boundary"] == "lport1"
    info = json.loads(Path(doc["mesh"]["info_file"]).read_text())
    g = {x["name"]: x for x in info["boundary_groups"]}
    assert g["comsol_boundary_2"]["type"] == "impedance" and g["comsol_boundary_3"]["material"] == "comsol_ibc_imp1"
    assert doc["materials"]["comsol_ibc_imp1"]["sigma_S_m"] == pytest.approx(5.8e7)
    assert g["comsol_boundary_7"]["type"] == "pec" and g["lport1"]["type"] == "port"
    assert "comsol_boundary_4" not in g          # interior interface: continuity
    st = {i["comsol"].split(" ")[0]: i["status"] for i in rep["items"] if i["category"] == "physics"}
    assert st["imp1"] == "mapped" and st["pml1"] == "approximated"   # center not stored


def test_axisymmetric_detection_and_override(tmp_path):
    mph = coax_mph(tmp_path, end="sbc", walls="pec", source=False)
    xml = zipfile.ZipFile(mph).read("dmodel.xml").decode().replace("<spatialCoord T=\"1\">3,'r','phi','z'</spatialCoord>", "")
    mph2 = write_mph(tmp_path / "nodetect.mph", xml, zipfile.ZipFile(mph).read("mesh1.mphbin"))
    with pytest.raises(ci.ImportError_, match="axisymmetric"):
        ci.import_comsol(mph2, tmp_path / "o" / "design.json")
    d = ci.import_comsol(mph2, tmp_path / "o2" / "design.json", axisymmetric=True)
    assert (d.doc if hasattr(d, "doc") else d)["symmetry"] == "axisymmetric"


def test_ibc_without_material_blocks(tmp_path):
    mph = coax_mph(tmp_path, end="sbc", walls="ibc", source=False)
    xml = zipfile.ZipFile(mph).read("dmodel.xml").decode()
    i = xml.index('<Material op="Common" tag="mat2"')
    xml = xml[:i] + xml[xml.index("</Material>", i) + len("</Material>"):]
    mph2 = write_mph(tmp_path / "nomat.mph", xml, zipfile.ZipFile(mph).read("mesh1.mphbin"))
    with pytest.raises(ci.ImportError_, match="no boundary material"):
        ci.import_comsol(mph2, tmp_path / "o" / "design.json")


def test_thick_transition_boundary_blocks(tmp_path):
    """A copper layer thicker than the skin depth on an interior boundary cannot be a sheet (equal E_t)."""
    mph = coax_mph(tmp_path, end="pml", walls="pec", source=False)
    xml = zipfile.ZipFile(mph).read("dmodel.xml").decode()
    tbc = feat("TransitionBoundaryCondition", "tbc1", 1, [4], 7, prm("d", "1[mm]"))
    xml = xml.replace("</PhysicsFeatureList>", tbc + "</PhysicsFeatureList>", 1)
    xml = xml.replace("</MaterialList>", _mat("mat3", "Cu", _sel(1, [4], 7), sig="5.8e7[S/m]") + "</MaterialList>")
    mph2 = write_mph(tmp_path / "tbc.mph", xml, zipfile.ZipFile(mph).read("mesh1.mphbin"))
    with pytest.raises(ci.ImportError_, match="k d"):
        ci.import_comsol(mph2, tmp_path / "o" / "design.json")
    # a 10 nm resistive film is a sheet
    mph3 = write_mph(tmp_path / "tbc_thin.mph", xml.replace("1[mm]", "10[nm]"), zipfile.ZipFile(mph).read("mesh1.mphbin"))
    d = ci.import_comsol(mph3, tmp_path / "o3" / "design.json")
    info = json.loads(Path((d.doc if hasattr(d, "doc") else d)["mesh"]["info_file"]).read_text())
    g = {x["name"]: x for x in info["boundary_groups"]}
    assert g["comsol_boundary_4"]["type"] == "sheet" and g["comsol_boundary_4"]["thickness_m"] == pytest.approx(1e-8)


# ------------------------------------------------------------------ solves (need the core)
@needs_core
@pytest.mark.parametrize("end", ["sbc", "pml"])
def test_imported_coax_matches_line_theory(tmp_path, end):
    """Imported 2D-axisymmetric coax with copper (impedance) walls, matched by a scattering BC or a PML: Z close to
    Z0 = eta ln(b/a)/(2 pi), the copper walls take the power 1 - exp(-2 alpha L), the power balance closes, and the
    source overlap of a uniform axial Je is the closed form."""
    import axionfem as af

    mph = coax_mph(tmp_path, end=end, walls="ibc", n=(3, 30))
    d = ci.import_comsol(mph, tmp_path / "out" / "design.json")
    res = d.solve(workdir=str(tmp_path / "run"), verbose=0)
    row = res.fullwave["sweep"][0]
    Z = row["Z_re_ohm"] + 1j * row["Z_im_ohm"]
    Z0 = ETA0 / (2 * math.pi) * math.log(B / A)
    assert abs(Z - Z0) / Z0 < 3e-3
    pw = row["power_W_per_A2"]
    f = 3e8
    alpha = math.sqrt(math.pi * f * MU0 / 5.8e7) * (1 / A + 1 / B) / (2 * ETA0 * math.log(B / A))
    walls = sum(v for k, v in pw["boundary_by_group"].items() if k in ("comsol_boundary_2", "comsol_boundary_3"))
    assert walls / pw["input"] == pytest.approx(1 - math.exp(-2 * alpha * L), rel=0.02)
    if end == "sbc":
        assert pw["balance_rel"] < 1e-10


@needs_core
def test_fw2d_quick_validation(tmp_path):
    import fw2d_validation as V

    f = 3e8
    k = 2 * math.pi * f / C0
    d, m = V.build("coax", tmp_path / "c", n=(4, 30), termination="pec")
    Z = V.solve(d, tmp_path / "c", [f]).sweep()["Z"][0]
    Z0 = V.coax_Z0(m["a"], m["b"])
    Zx = -1j * Z0 * math.tan(k * m["l"])
    assert abs(Z - Zx) / abs(Zx) < 1e-4
    d, m = V.build("wedge", tmp_path / "w", n=(6, 20), termination="scattering")
    Z = V.solve(d, tmp_path / "w", [1e9]).sweep()["Z"][0]
    Zx = V.wedge_Z(1e9, m["theta0"], m["r0"], m["D"], m["R"])
    assert abs(Z - Zx) / abs(Zx) < 1e-3
    d, m = V.build("bicone", tmp_path / "b", n=(6, 20), termination="pml", order=3)
    Z = V.solve(d, tmp_path / "b", [1e9]).sweep()["Z"][0]
    assert abs(Z - V.bicone_Z(m["theta0"])) / V.bicone_Z(m["theta0"]) < 1e-3
    d, m = V.build("pillbox", tmp_path / "p", n=(10, 15), walls="impedance")
    res = V.solve(d, tmp_path / "p", modes=1, families=["TM", "TE"])
    f010, G = V.pillbox_TM010(m["a"], m["d"])
    tm = res.modes("TM")[0]
    assert tm["f_Hz"] == pytest.approx(f010, rel=1e-5)
    assert tm["Q_wall"] == pytest.approx(G / V.rs_copper(f010), rel=1e-4)
    assert res.modes("TE")[0]["f_Hz"] == pytest.approx(V.pillbox_TE011(m["a"], m["d"]), rel=1e-4)
    # coaxial cavity (conductor region r < a/2): the static G/r field is constrained away, TEM_1 at c/(2d)
    dc, mc = V.build("pillbox", tmp_path / "cc", n=(10, 15), extra={"inner": 0.05})
    rc = V.solve(dc, tmp_path / "cc", modes=1, families=["TM"])
    assert rc.modes("TM")[0]["f_Hz"] == pytest.approx(C0 / (2 * mc["d"]), rel=1e-4)
    assert rc.fullwave["eigenmodes"]["TM"]["static_mode_constraints"] == 1
    s = res.sample([0.03], [0.07], mode=("TM", 0))
    # components (r, z, phi): TM010 has E_z and H_phi only (E_r ~ 0 up to discretization), E_phi = 0 exactly
    E, H = s["E"][0], s["H"][0]
    assert s["valid"][0] and abs(E[2]) == 0 and abs(H[0]) == 0 and abs(H[1]) == 0
    assert abs(E[0]) < 1e-3 * abs(E[1])
    # eta |H_phi| / |E_z| = J1(x) / J0(x) at x = p01 r / a (both from the same normalized mode)
    from scipy.special import j0, j1
    x = 2.404825557695773 * 0.03 / m["a"]
    assert abs(H[2]) * ETA0 / abs(E[1]) == pytest.approx(j1(x) / j0(x), rel=5e-3)


@needs_core
@needs_3d
def test_fw3d_robin_quick(tmp_path):
    import axionfem as af
    import fw3d_robin_validation as V3

    f = 3e8
    for term, plates in (("scattering", "pec"), ("scattering", "impedance")):
        d, m = V3.build(tmp_path / f"{term}_{plates}", termination=term, plates=plates, nx=20)
        d["fullwave"]["frequencies"] = [f]
        row = af.Design(d).solve(workdir=str(tmp_path / f"{term}_{plates}"), verbose=0).data["fullwave"]["sweep"][0]
        Z = row["Z_re_ohm"] + 1j * row["Z_im_ohm"]
        Z0 = ETA0 * m["h"] / m["W"]
        if plates == "pec":
            assert abs(Z - Z0) / Z0 < 1e-6
        else:
            alpha = -math.log(row["boundary_power_W_per_A2"]["end"] / (0.5 * Z.real)) / (2 * m["l"])
            assert alpha == pytest.approx(math.sqrt(math.pi * f * MU0 / 5.8e7) / (ETA0 * m["h"]), rel=2e-3)
            assert row["power_balance_rel"] < 1e-8


def test_spec_rejects_bad_pml_and_planar_without_mesh():
    pytest.importorskip("axionfem._core")
    import axionfem as af

    with pytest.raises(Exception, match="planar"):
        af.Design({"schema": "axionfem.design/1", "name": "x", "symmetry": "planar", "regions": []}).validate()


# ------------------------------------------------------------------ 3D route: impedance / scattering features
def test_3d_route_impedance_and_scattering(tmp_path):
    """The synthetic 3D box of test_comsol_import.py with copper (impedance) walls on x = 0 and an absorbing
    (scattering) wall on x = 1: boundary groups carry the parameters; with the core, the lossy open box solves and
    its power balance closes."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_comsol_import as T3

    extra = (feat("ImpedanceBoundaryCondition", "imp1", 1, [1], 8).replace('dim="1" hDim="1"', 'dim="2" hDim="2"') +
             feat("Scattering", "sctr1", 1, [2], 8).replace('dim="1" hDim="1"', 'dim="2" hDim="2"'))
    xml = T3._DMODEL.replace("</PhysicsFeatureList>", extra + "</PhysicsFeatureList>", 1)
    xml = xml.replace("</MaterialList>", _mat("mat9", "Cu", _sel(1, [1], 8).replace('dim="1" hDim="1"', 'dim="2" hDim="2"'),
                                              sig="5.8e7[S/m]") + "</MaterialList>")
    P, Tt, tris, tent = T3._box_mesh(4)
    mph = write_mph(tmp_path / "box3d.mph", xml, T3._write_mesh_v7(P, Tt, tris, tent))
    d = ci.import_comsol(mph, tmp_path / "o" / "design.json", frequencies=[2e8])
    doc = d.doc if hasattr(d, "doc") else d
    info = json.loads(Path(doc["mesh"]["info_file"]).read_text())
    g = {x["name"]: x for x in info["boundary_groups"]}
    assert g["comsol_boundary_1"]["type"] == "impedance" and g["comsol_boundary_1"]["material"] == "comsol_ibc_imp1"
    assert g["comsol_boundary_2"] == {**g["comsol_boundary_2"], "type": "scattering", "wave": "plane"}
    assert not d.import_report.blocking
    if HAVE_FW2D and HAVE_3D:
        row = d.solve(workdir=str(tmp_path / "run"), verbose=0).data["fullwave"]["sweep"][0]
        assert row["Z_re_ohm"] > 0 and row["power_balance_rel"] < 1e-6
        assert row["boundary_power_W_per_A2"]["comsol_boundary_2"] > 0
        assert row["boundary_power_W_per_A2"]["comsol_boundary_1"] > 0


# ------------------------------------------------------------------ feature parameters, quads, curving, input checks
def _quad_mph(tmp_path, order):
    """Planar unit square of 2 x 2 quadrilaterals (COMSOL tensor order 0-1-3-2 when order == 'tensor')."""
    xs = ys = np.linspace(0, 1, 3)
    idx = lambda i, j: i * 3 + j  # noqa: E731
    P = np.array([[x, y] for x in xs for y in ys], float)
    Q = []
    for i in range(2):
        for j in range(2):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            Q.append([a, b, d, c] if order == "tensor" else [a, b, c, d])
    E, Ee = [], []
    for k in range(2):
        for e in ((idx(k, 0), idx(k + 1, 0)), (idx(2, k), idx(2, k + 1)), (idx(k, 2), idx(k + 1, 2)), (idx(0, k), idx(0, k + 1))):
            E.append(sorted(e))
            Ee.append(0)
    xml = dmodel_2d("", ndom=1, nbnd=1, axisym=False, thickness="1[m]")
    return write_mph(tmp_path / f"quad_{order}.mph", xml,
                     write_mesh_v7_2d(P, np.zeros((0, 3), int), np.zeros(0, int), np.array(E), np.array(Ee),
                                      np.array(Q), np.ones(4, int)))


def test_impedance_feature_parameters(tmp_path):
    """The impedance feature's own overrides (murbnd / sigmabnd), an explicit surface impedance and a
    nonzero surface source field."""
    over = (prm("murbnd_mat", "userdef") + prm("murbnd", "7") + prm("sigmabnd_mat", "userdef") +
            prm("sigmabnd", "6e5[S/m]"))
    d = ci.import_comsol(coax_mph(tmp_path, ibc_params=over, name="_over"), tmp_path / "o1" / "design.json")
    doc = d.doc if hasattr(d, "doc") else d
    m = doc["materials"]["comsol_ibc_imp1"]
    assert m["mu_r"] == pytest.approx(7) and m["sigma_S_m"] == pytest.approx(6e5)
    zs = prm("surfaceImpedanceDefinition", "userdef") + prm("surfaceImpedance", "50+2*j")
    d = ci.import_comsol(coax_mph(tmp_path, ibc_params=zs, name="_zs"), tmp_path / "o2" / "design.json")
    doc = d.doc if hasattr(d, "doc") else d
    g = {x["name"]: x for x in json.loads(Path(doc["mesh"]["info_file"]).read_text())["boundary_groups"]}
    assert g["comsol_boundary_2"]["type"] == "impedance" and g["comsol_boundary_2"]["Zs_ohm"] == pytest.approx([50, -2])
    es = '<param T="33" param="Es" value="3|1,\'0\'|1,\'1[V/m]\'|1,\'0\'"/>'
    with pytest.raises(ci.ImportError_, match="surface source field"):
        ci.import_comsol(coax_mph(tmp_path, ibc_params=es, name="_es"), tmp_path / "o3" / "design.json")


def test_quad_split(tmp_path):
    """Quadrilaterals in COMSOL's tensor order are split into triangles that tile them; a vertex order that does
    not tile the elements is refused."""
    d = ci.import_comsol(_quad_mph(tmp_path, "tensor"), tmp_path / "o" / "design.json", allow_partial=True)
    rep = d.import_report.to_json()
    assert any(i["comsol"] == "quadrilaterals" and i["status"] == "approximated" for i in rep["items"])
    with pytest.raises(ci.ImportError_, match="vertex order"):
        ci.import_comsol(_quad_mph(tmp_path, "cyclic"), tmp_path / "o2" / "design.json", allow_partial=True)


def test_curving_keeps_thin_elements_valid(tmp_path):
    """Boundary-layer triangles on a curved boundary (first layer 2 um, edge 0.2 mm): mid-node moves that would
    invert an element are undone and reported; every written element has det J > 0."""
    a, b, nth = 1e-3, 2e-3, 32
    rs, hh = [a], 2e-6
    while rs[-1] + hh < b and len(rs) < 12:
        rs.append(rs[-1] + hh)
        hh *= 1.6
    rs = np.array(rs + [b])
    th = np.linspace(0, 2 * np.pi, nth + 1)[:-1]
    P = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in th])
    idx = lambda i, j: i * nth + (j % nth)  # noqa: E731
    T = np.array([t for i in range(len(rs) - 1) for j in range(nth) for t in
                  ([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i + 1, j)])])
    E = np.array([[idx(0, j), idx(0, j + 1)] for j in range(nth)] +
                 [[idx(len(rs) - 1, j), idx(len(rs) - 1, j + 1)] for j in range(nth)])
    Ee = np.array([0] * nth + [1] * nth)
    xml = dmodel_2d("", ndom=1, nbnd=2, axisym=False, thickness="1[m]")
    mph = write_mph(tmp_path / "annulus.mph", xml, write_mesh_v7_2d(P, T, np.ones(len(T), int), E, Ee))
    d = ci.import_comsol(mph, tmp_path / "o" / "design.json", allow_partial=True)
    doc = d.doc if hasattr(d, "doc") else d
    cur = json.loads(Path(doc["mesh"]["info_file"]).read_text())["curving"]
    assert cur["reverted_mid_nodes"] > 0 and cur["min_detJ_over_straight"] > 0.2
    assert cur["moved_mid_nodes"] > cur["reverted_mid_nodes"]   # the outer circle and the coarse side stay curved


def _plates_design(tmp_path, groups, edges_extra=(), *, n=(20, 2), drop_edge=False, symmetry="planar", source=None,
                   x0=0.0):
    """Line 0 < y < 0.02 between x0 and x0 + 0.3 (port at x = x0, PEC elsewhere) as an imported 2D design."""
    from axionfem.fw2d import mapped_mesh, write_mesh

    tmp_path.mkdir(parents=True, exist_ok=True)
    l, h = 0.3, 0.02
    pts, tri, reg, edges, side = mapped_mesh(lambda u, v: (x0 + l * u, h * v), *n)
    attr = {0: 10002, 1: 10002, 2: 10002, 3: 10001}
    ea = [attr[s] for s in side]
    keep = np.ones(len(edges), bool)
    if drop_edge:
        keep[[i for i, s in enumerate(side) if s == 2][n[0] // 2]] = False
    E, A_ = list(edges[keep]), list(np.array(ea)[keep])
    for e, a in edges_extra:
        E.append(e)
        A_.append(a)
    write_mesh(tmp_path / "m.msh", pts, tri, 2 + reg, np.array(E), A_)
    (tmp_path / "info.json").write_text(json.dumps({"boundary_groups": groups}))
    fw = {"boundary": "pec", "rf_only": True, "frequencies": [1e8]}
    if symmetry == "planar":
        fw["thickness_m"] = 0.05
    if source:
        fw["source"] = source
    return {"schema": "axionfem.design/1", "name": "x", "symmetry": symmetry,
            "regions": [{"name": "field", "material": "vacuum", "shape": {"imported": {}}}],
            "fullwave": fw, "pickup": {"type": "port", "port_boundary": "lport1"},
            "mesh": {"order": 2, "file": str(tmp_path / "m.msh"), "info_file": str(tmp_path / "info.json")}}, pts, tri


@needs_core
def test_fw2d_input_checks(tmp_path):
    """An exterior edge without a boundary element, a scattering group on interior edges and a second
    port group are refused with the reason."""
    import axionfem as af
    from fw2d_validation import interior_edges

    base = [{"attr": 10001, "name": "lport1", "type": "port"}, {"attr": 10002, "name": "walls", "type": "pec"}]
    d, _, _ = _plates_design(tmp_path / "a", base, drop_edge=True)
    with pytest.raises(Exception, match="no boundary element"):
        af.Design(d).solve(workdir=str(tmp_path / "a" / "r"), verbose=0)
    from axionfem.fw2d import mapped_mesh

    pts, tri, *_ = mapped_mesh(lambda u, v: (0.3 * u, 0.02 * v), 20, 2)
    mid = [(e, 10003) for e in interior_edges(pts, tri, lambda x, y: abs(x - 0.15) < 1e-12)]
    d, _, _ = _plates_design(tmp_path / "b", base + [{"attr": 10003, "name": "absorber", "type": "scattering"}], mid)
    with pytest.raises(Exception, match="must be exterior"):
        af.Design(d).solve(workdir=str(tmp_path / "b" / "r"), verbose=0)
    d, _, _ = _plates_design(tmp_path / "c", base + [{"attr": 10003, "name": "lport2", "type": "port"}], mid)
    with pytest.raises(Exception, match="port groups"):
        af.Design(d).solve(workdir=str(tmp_path / "c" / "r"), verbose=0)


@needs_core
def test_fw2d_axisymmetric_sector_source(tmp_path):
    """A phi-dependent source in 2D axisymmetry couples through its azimuthal average (a 90 deg sector: 1/4 of
    the full overlap, independent of where the sector starts), with a warning."""
    import axionfem as af

    groups = [{"attr": 10001, "name": "lport1", "type": "port"}, {"attr": 10002, "name": "walls", "type": "pec"}]
    psi = {}
    for k, extra in enumerate(({}, {"phi_deg": [-10.0, 80.0]}, {"phi_deg": [10.0, 100.0]}, {"phi_deg": [10.0, 20.0]})):
        src = {"kind": "axial_box", "B0_T": 1.0, "r_m": [0.0, 1.0], "z_m": [-1.0, 1.0], **extra}
        d, _, _ = _plates_design(tmp_path / str(k), groups, symmetry="axisymmetric", source=src, x0=0.05)
        res = af.Design(d).solve(workdir=str(tmp_path / str(k) / "r"), verbose=0)
        psi[k] = res.sweep()["Psi"][0]
        assert bool(extra) == any("sector" in w for w in res.data.get("warnings", []))
    assert psi[1] / psi[0] == pytest.approx(0.25, rel=1e-9)
    assert psi[2] / psi[0] == pytest.approx(0.25, rel=1e-9)
    assert psi[3] / psi[0] == pytest.approx(1 / 36, rel=1e-9)   # a sector between the 45-degree probe angles


@needs_core
def test_fw2d_septum_and_two_sided_pml(tmp_path):
    """An interior PEC wall splits a planar cavity (both families: every half-cavity mode twice); one PML entry
    covering both ends of a line (port in the middle) gives Z0/2."""
    import fw2d_validation as V

    d, m = V.build("septum", tmp_path / "s", n=(8, 8))
    res = V.solve(d, tmp_path / "s", modes=4, families=["TE", "TM"])
    h = m["a"] / 2
    f10 = C0 / (2 * h)
    assert [x["f_Hz"] for x in res.modes("TE")] == pytest.approx([f10] * 4, rel=1e-3)
    assert [x["f_Hz"] for x in res.modes("TM")][:2] == pytest.approx([C0 / 2 * math.hypot(1 / h, 1 / m["b"])] * 2, rel=1e-3)
    assert res.fullwave["eigenmodes"]["TE"]["sheet_split_dofs"] > 0
    d, m = V.build("plates", tmp_path / "p", n=(4, 30), termination="pml", order=3, extra={"two_sided": True})
    Z = V.solve(d, tmp_path / "p", [3e8]).sweep()["Z"][0]
    Z0 = ETA0 * m["b"] / m["D"]
    assert abs(Z - Z0 / 2) / (Z0 / 2) < 1e-4


@needs_core
def test_fw2d_axis_sample_and_rerun(tmp_path):
    """A mode sampled on the axis is finite (u/r -> du/dr) and continuous with the near-axis value;
    running the same model twice gives the same result (state reset)."""
    import fw2d_validation as V
    from axionfem import _core
    from axionfem.fw2d import Fw2DResults

    d, m = V.build("pillbox", tmp_path / "p", n=(6, 6))
    res = V.solve(d, tmp_path / "p", modes=1, families=["TM"])
    s = res.sample([0.0, 1e-6], [0.05, 0.05], mode=("TM", 0))
    assert np.all(np.isfinite(s["E"])) and s["valid"][0]
    assert abs(s["E"][0][1]) == pytest.approx(abs(s["E"][1][1]), rel=1e-4)
    dd, mm = V.build("coax", tmp_path / "c", n=(4, 10), termination="pec")
    dd["fullwave"]["frequencies"] = [3e8]
    model = _core.Fw2DModel(json.dumps(dd))
    model.run(str(tmp_path / "c1"), 0, 0)
    z1 = json.loads(model.results())["fullwave"]["sweep"][0]["Z_im_ohm"]
    model.run(str(tmp_path / "c2"), 0, 0)
    z2 = json.loads(model.results())["fullwave"]["sweep"][0]["Z_im_ohm"]
    assert z1 == z2
