"""COMSOL .mph import (python/axionfem/comsol_import.py, docs/comsol_import.md).  Beta.

- the expression/unit evaluator and the dmodel.xml value/selection encodings;
- the native binary readers on a synthetic file (written with the layout established on real files) and on the
  2211.00008 .mph files (and a COMSOL 6.4 re-save of one) when they are present (skipped otherwise);
- the mesh route on a synthetic model (a PEC plate in a closed PEC box fed by a Uniform lumped port in the gap
  between the plate and the lid): design content, boundary groups, curving, and, with the core built, a small
  full-wave run (lossless: Re Z = 0, inductive at low frequency, order-1 vs order-2 agreement);
- the replay route geometry (cells, signatures, volumes) on the real toroid file.
"""
import json
import math
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from axionfem import comsol_model as cm
from axionfem import comsol_mphbin as mb
from axionfem import comsol_import as ci

# The real COMSOL files are not distributed. AXF_COMSOL_2211_DIR names a directory with the unpacked
# arXiv:2211.00008 models (Shielded_Toroid/, Shielded_Solenoid/, Toroid_With_Epoxy/) and AXF_COMSOL64_SOLENOID_MPH
# the shielded solenoid's impedance model re-saved by COMSOL 6.4; the tests that need a file skip when it is absent.
_UNSET = Path("/nonexistent-axf-comsol-files")
U = Path(os.environ["AXF_COMSOL_2211_DIR"]) if os.environ.get("AXF_COMSOL_2211_DIR") else _UNSET
TOROID = U / "Shielded_Toroid" / "Small_Shield.mph"
SOLENOID = U / "Shielded_Solenoid" / "Model_File.mph"
EPOXY = U / "Toroid_With_Epoxy" / "Small_Shield.mph"
SOLENOID64 = (Path(os.environ["AXF_COMSOL64_SOLENOID_MPH"]) if os.environ.get("AXF_COMSOL64_SOLENOID_MPH")
              else _UNSET)
needs = lambda p: pytest.mark.skipif(not p.is_file(), reason=f"{p.name} not available")  # noqa: E731


# ------------------------------------------------------------------ evaluator
@pytest.mark.parametrize("expr,val", [
    ("1[V]", 1.0), (".3[GHz]", 3e8), ("50[ohm]", 50.0), ("2^3", 8.0), ("1[A/m^2]", 1.0), ("5[mm]", 5e-3),
    ("10[kHz]*2", 2e4), ("sqrt(4)*pi", 2 * math.pi), ("90[deg]", math.pi / 2), ("1[mT]", 1e-3),
    ("if(2>1, 3, 4)", 3.0), ("1e-3[m]", 1e-3), ("2[uF]", 2e-6), ("1[kg/(m*s^2)]", 1.0)])
def test_evaluate_units(expr, val):
    assert cm.evaluate(expr) == pytest.approx(val)


def test_evaluate_variables_params_functions():
    x = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    v = cm.evaluate("(-y / sqrt(x^2 + y^2)) [A/m^2]", variables={"x": x[0], "y": x[1]})
    assert np.allclose(v, [0.0, -1.0])
    assert cm.evaluate("2*R+1", {"R": 3.0}) == 7.0
    assert cm.evaluate("f(freq/1[Hz])", variables={"freq": 5.0}, functions={"f": lambda t: 2 * t}) == 10.0
    with pytest.raises(cm.ExpressionError, match="unknown name 'q'"):
        cm.evaluate("q+1")
    with pytest.raises(cm.ExpressionError, match="model variable"):
        cm.evaluate("comp1.emw.Zport_1")


def test_parse_list_and_values():
    assert cm.parse_list("range(5[MHz],1[MHz],8[MHz])") == pytest.approx([5e6, 6e6, 7e6, 8e6])
    assert cm.parse_list("5 5.1 5.2", unit="MHz") == pytest.approx([5e6, 5.1e6, 5.2e6])
    assert cm.parse_list(".3[GHz]", unit="GHz") == pytest.approx([3e8])
    assert len(cm.parse_list("range(.1[MHz],.1[MHz],10[MHz])")) == 100
    assert cm.parse_value("3|1,'(-y / sqrt(x^2 + y^2)) [A/m^2]'|1,'b'|1,'0'") == [
        "(-y / sqrt(x^2 + y^2)) [A/m^2]", "b", "0"]
    assert cm.parse_value("1|1,'pA\\,T'") == ["pA,T"]
    assert cm.parse_matrix("3|2,'2e9','3.04'|2,'5e9','3.02'|2,'10e9','3.02'")[2] == ["10e9", "3.02"]


def test_explicit_selection_headers():
    from xml.etree.ElementTree import Element
    cnt, w = {}, []
    s = cm.parse_explicit(Element("explicit", dim="3", hDim="3", entities="11,1,6"), 3, cnt, w)
    assert s.entities == [1, 6] and cnt[3] == 10
    s = cm.parse_explicit(Element("explicit", dim="2", hDim="2", entities="-125,-1,121,122"), 3, cnt, w)
    assert s.entities == [121, 122] and cnt[2] == 125
    assert cm.parse_explicit(Element("explicit", dim="2"), 3).entities is None
    assert not w


# ------------------------------------------------------------------ synthetic .mph
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def _box_mesh(n):
    """Unit cube, n^3 cubes x 6 tets (n divisible by 4), domain 1.  Boundaries (GUI numbers): x=0 1, x=1 2,
    y=0 3, y=1 4, z=0 5, z=1 6; interior plate x=0.5, y 0.25..0.75, z 0..0.75 -> 7; port x=0.5, y 0.25..0.75,
    z 0.75..1 -> 8."""
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    P = np.array([[i, j, k] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)], float) / n
    T = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                T += [[c[q] for q in t] for t in _KUHN]
    T = np.array(T)
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    uf, cnt = np.unique(F, axis=0, return_counts=True)
    tris, ent = [], []
    eps = 1e-12
    for f, c in zip(uf, cnt):
        X = P[f]
        if c == 1:
            for ax in range(3):
                for side, val in ((0, 0.0), (1, 1.0)):
                    if np.all(np.abs(X[:, ax] - val) < eps):
                        tris.append(f)
                        ent.append(2 * ax + side)
        elif np.all(np.abs(X[:, 0] - 0.5) < eps) and np.all(X[:, 1] >= 0.25 - eps) and np.all(X[:, 1] <= 0.75 + eps):
            if np.all(X[:, 2] <= 0.75 + eps):
                tris.append(f)
                ent.append(6)
            else:
                tris.append(f)
                ent.append(7)
    return P, T, np.array(tris), np.array(ent)


def _write_mesh_v7(P, T, tris, tent):
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
    i32(7, 3)
    b.extend(b"\x00\x00")
    i32(1, len(P), 0, 3)
    for c in range(3):
        i32(len(P))
        b.extend(np.asarray(P[:, c], "<f8").tobytes())
    i32(2)
    for name, el, ent in (("tri", tris, tent), ("tet", T, np.ones(len(T), int))):
        string(name)
        i32(el.shape[1])
        for c in range(el.shape[1]):
            i32(len(el))
            b.extend(np.asarray(el[:, c], "<i4").tobytes())
        i32(el.shape[1], 0)                     # no element parameters
        i32(len(el), len(el))
        b.extend(np.asarray(ent, "<i4").tobytes())
        i32(0)                                   # no up/down
    return bytes(b)


_DMODEL = """<?xml version="1.0" encoding="UTF-8"?>
<Model tag="Model1" name="synthetic.mph">
  <ModelParam tag="param"><ModelParamGroupList tag="group"><ModelParamGroup tag="default" name="Parameters 1">
    <param T="33" param="f0" value="1|1,'2[MHz]'" descr="drive"/>
  </ModelParamGroup></ModelParamGroupList></ModelParam>
  <PhysicsList tag="physics">
    <Physics op="ElectromagneticWaves" tag="emw" name="emw">
      <selection T="32" selection="LocalSelection" selType="GEOMDIM"><explicit dim="3" hDim="3" geom="/geom/geom1" entities="2,1"/></selection>
      <geom T="50">/geom/geom1</geom>
      <PhysicsFeatureList tag="feature">
        <PhysicsFeature op="WaveEquationElectric" tag="wee1" name="Wave Equation, Electric 1">
          <selection T="32" selection="XDLocalSelection" flags="17" selType="GEOMDIM"><explicit dim="3" hDim="3" geom="/geom/geom1"/></selection>
          <param T="33" param="DisplacementFieldModel" value="1|1,'RelativePermittivity'"/>
          <default T="3">true</default>
        </PhysicsFeature>
        <PhysicsFeature op="PerfectElectricConductor" tag="pec1" name="PEC 1">
          <selection T="32" selection="XDLocalSelection" flags="17" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1"/></selection>
          <default T="3">true</default>
        </PhysicsFeature>
        <PhysicsFeature op="ExternalCurrentDensity" tag="ecd1" name="External Current Density 1">
          <entityFlags T="51">DISABLED</entityFlags>
          <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="3" hDim="3" geom="/geom/geom1" entities="2,1"/></selection>
          <param T="33" param="Je" value="3|1,'0'|1,'2[A/m^2]'|1,'0'"/>
        </PhysicsFeature>
        <PhysicsFeature op="PerfectElectricConductor" tag="pec2" name="PEC 2">
          <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" entities="-8,-1,7"/></selection>
        </PhysicsFeature>
        <PhysicsFeature op="LumpedPort" tag="lport1" name="Lumped Port 1">
          <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" entities="-8,-1,8"/></selection>
          <param T="33" param="PortType" value="1|1,'Uniform'"/>
          <param T="33" param="PortExcitation" value="1|1,'on'"/>
          <param T="33" param="Zref" value="1|1,'50[ohm]'"/>
        </PhysicsFeature>
      </PhysicsFeatureList>
      <PhysicsPropList tag="prop"><PhysicsProp tag="ShapeProperty"><param T="33" param="order_electricfield" value="1|1,'2'"/></PhysicsProp></PhysicsPropList>
    </Physics>
  </PhysicsList>
  <GeomList tag="geom"><GeomSequence tag="geom1" name="Geometry 1"><sdim T="5">3</sdim><lengthUnit T="0">m</lengthUnit><angularUnit T="0">deg</angularUnit>
    <GeomFeatureList tag="feature"/></GeomSequence></GeomList>
  <MaterialList tag="material">
    <Material op="Common" tag="mat1" name="Dielectric">
      <selection T="32" selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="3" hDim="3" geom="/geom/geom1"/></selection>
      <MaterialModelList tag="materialmodel"><MaterialModel tag="def" name="Basic">
        <param T="33" param="relpermittivity" value="9|1,'2.5'|1,'0'|1,'0'|1,'0'|1,'2.5'|1,'0'|1,'0'|1,'0'|1,'2.5'"/>
        <param T="33" param="relpermeability" value="9|1,'1'|1,'0'|1,'0'|1,'0'|1,'1'|1,'0'|1,'0'|1,'0'|1,'1'"/>
        <param T="33" param="electricconductivity" value="9|1,'0[S/m]'|1,'0'|1,'0'|1,'0'|1,'0[S/m]'|1,'0'|1,'0'|1,'0'|1,'0[S/m]'"/>
      </MaterialModel></MaterialModelList>
    </Material>
  </MaterialList>
  <MeshList tag="mesh"><MeshSequence tag="mesh1" name="Mesh 1"><geom T="50">/geom/geom1</geom>
    <MeshNative tag="meshnative"><numEntities T="6">8,12,8,1</numEntities><mesh T="52" class="MESH" resource="mesh1.mphbin" id="1"/></MeshNative>
  </MeshSequence></MeshList>
  <StudyList tag="study"><Study tag="std1" name="Study 1"><StudyFeatureList tag="feature">
    <StudyFeature op="Frequency" tag="freq" name="Frequency Domain">
      <propertyValue T="30" name="p:plist" value="f0 4[MHz]"/><propertyValue T="30" name="p:punit" value="MHz"/>
    </StudyFeature></StudyFeatureList></Study></StudyList>
</Model>
"""


def synthetic_mph(path: Path, n=4):
    P, T, tris, tent = _box_mesh(n)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fileversion", "0:synthetic")
        z.writestr("dmodel.xml", _DMODEL)
        z.writestr("mesh1.mphbin", _write_mesh_v7(P, T, tris, tent))
    return path


def test_synthetic_mesh_roundtrip(tmp_path):
    P, T, tris, tent = _box_mesh(4)
    m = mb.read_mesh(_write_mesh_v7(P, T, tris, tent))
    assert m.version == 7 and np.array_equal(m.points, P)
    assert np.array_equal(m.elements["tet"], T) and np.array_equal(m.entities["tri"], tent)
    assert mb.object_class(_write_mesh_v7(P, T, tris, tent)) == "Mesh"
    with pytest.raises(mb.MphbinError, match="truncated"):
        mb.read_mesh(_write_mesh_v7(P, T, tris, tent)[:-20])


def test_synthetic_mesh_route_design(tmp_path):
    mph = synthetic_mph(tmp_path / "synthetic.mph")
    d = ci.import_comsol(mph, tmp_path / "out" / "design.json")
    doc = d.doc if hasattr(d, "doc") else d
    rep = d.import_report.to_json()
    assert doc["fullwave"]["frequencies"] == pytest.approx([2e6, 4e6])          # parameter f0 + list in punit
    assert doc["materials"]["comsol_dielectric_mat1"]["eps_r"] == pytest.approx(2.5)
    assert doc["regions"][0]["solid"] == {"imported": {"comsol_domains": [1], "comsol_material": "mat1"}}
    assert doc["pickup"]["port_boundary"] == "lport1"
    assert doc["pickup"]["toward"] == pytest.approx([0, 0, 1])                    # from the plate up to the lid
    assert doc["fullwave"]["source"]["kind"] == "uniform"
    assert doc["fullwave"]["source"]["B_T"] == pytest.approx([0, 2, 0])
    info = json.loads(Path(doc["mesh"]["info_file"]).read_text())
    types = {g["name"]: g["type"] for g in info["boundary_groups"]}
    assert types["lport1"] == "port" and types["comsol_boundary_7"] == "pec"
    assert all(types[f"comsol_boundary_{b}"] == "pec" for b in range(1, 7))
    assert info["port"]["width_m"] == pytest.approx(0.5) and info["port"]["length_m"] == pytest.approx(0.25)
    assert not rep["blocking"]
    assert any(i["comsol"].startswith("ecd1") and "DISABLED" in i["detail"] for i in rep["items"])
    assert (tmp_path / "out" / "import_report.md").read_text().startswith("# COMSOL import report")


def test_unsupported_physics_blocks(tmp_path):
    mph = tmp_path / "bad.mph"
    extra = ('<PhysicsFeature op="ImpedanceBoundaryCondition" tag="imp1" name="Impedance 1"><selection T="32" '
             'selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" '
             'entities="-8,-1,1"/></selection></PhysicsFeature>\n      </PhysicsFeatureList>')
    xml = _DMODEL.replace("</PhysicsFeatureList>", extra, 1)
    P, T, tris, tent = _box_mesh(4)
    with zipfile.ZipFile(mph, "w") as z:
        z.writestr("dmodel.xml", xml)
        z.writestr("mesh1.mphbin", _write_mesh_v7(P, T, tris, tent))
    with pytest.raises(ci.ImportError_, match="ImpedanceBoundaryCondition"):
        ci.import_comsol(mph, tmp_path / "o" / "design.json")
    d = ci.import_comsol(mph, tmp_path / "o2" / "design.json", allow_partial=True)
    assert d.import_report.blocking


def test_curving_cylinder():
    """Mid-edge nodes of a faceted cylinder are moved onto it; seams between two faces stay on the surface."""
    n, m = 24, 4
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.linspace(0, 1, m + 1)
    P = np.array([[np.cos(t), np.sin(t), zz] for zz in z for t in th])
    tri, bnd = [], []
    for k in range(m):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            c, d_ = a + n, b + n
            face = 1 if i < n // 2 else 2
            tri += [[a, b, d_], [a, d_, c]]
            bnd += [face, face]
    cmid = ci.curve_midpoints(P, np.array(tri), np.array(bnd))
    r = np.array([np.hypot(*v[:2]) for v in cmid.values()])
    assert len(cmid) > 0 and np.allclose(r, 1.0, atol=1e-9)


# ------------------------------------------------------------------ real files
@needs(TOROID)
def test_toroid_model_tree():
    m = cm.load_model(TOROID)
    g = m.geometries["geom1"]
    assert g["counts"] == {2: 125, 3: 10}
    assert [f.tag for f in g["features"]] == ["wp1", "rev1", "rot1", "cyl1", "wp2", "rev2", "wp3", "rev3", "fin"]
    emw = m.physics[0]
    assert emw["selection"].entities == [1, 6]
    f = {x.tag: x for x in emw["features"]}
    assert f["ecd1"].disabled and f["ecd1"].selection.entities == [6]
    assert f["pec2"].selection.entities == [121, 122, 124, 125]
    assert f["lport1"].p("PortType") == "Uniform" and f["lport1"].p("Zref") == "50[ohm]"
    assert m.meshes["mesh1"]["resource"] == "mesh2.mphbin"


@needs(TOROID)
def test_toroid_mesh_decoder():
    z = zipfile.ZipFile(TOROID)
    m = mb.read_mesh(z.read("mesh2.mphbin"))
    assert len(m.points) == 26016 and len(m.elements["tet"]) == 132524
    assert set(np.unique(m.entities["tet"])) == {1, 6}
    tri = m.elements["tri"]
    port = m.points[tri[m.entities["tri"] == 122]].reshape(-1, 3)        # GUI boundary 123
    r = np.hypot(port[:, 0], port[:, 1])
    phi = np.degrees(np.arctan2(port[:, 1], port[:, 0])) % 360
    assert np.allclose(r, 1.0, atol=1e-9) and phi.min() > 356.99 and phi.max() < 357.02
    assert port[:, 2].min() >= 0.24 - 1e-12 and port[:, 2].max() <= 0.241 + 1e-12


@needs(SOLENOID)
def test_solenoid_geom3_decoder():
    g = mb.read_geometry(zipfile.ZipFile(SOLENOID).read("geometry1.mphbin"))
    assert (len(g.vertices), len(g.edges), len(g.face_updown), g.ndomains) == (74, 151, 84, 5)


@needs(SOLENOID64)
def test_solenoid_comsol64_resave_parses():
    m = cm.load_model(SOLENOID64)
    assert "6.4" in m.version
    f = {x.tag: x for x in m.physics[0]["features"]}
    assert f["lport1"].selection.entities == [79, 80] and f["ecd1"].disabled


@needs(TOROID)
def test_toroid_mesh_route(tmp_path):
    d = ci.import_comsol(TOROID, tmp_path / "t" / "design.json", frequencies=[5e6])
    doc = d.doc if hasattr(d, "doc") else d
    rep = d.import_report.to_json()
    assert [r["name"] for r in doc["regions"]] == ["domain_1", "domain_6"]
    src = doc["fullwave"]["source"]
    assert src["kind"] == "toroidal_box" and src["B0_T"] == pytest.approx(1.0) and src["profile"] == "uniform"
    assert doc["fullwave"]["source_regions"] == ["domain_6"]
    assert doc["pickup"]["toward"] == pytest.approx([0.05224881, 0.99863410, 0], abs=1e-6)
    v = rep["summary"]["domain_volumes_m3"]
    exact1 = math.pi * 1.5 ** 2 - math.pi * (1 - 0.25) * 0.5 * 355 / 360       # shield cylinder minus the toroid
    exact6 = math.pi * (0.975 ** 2 - 0.525 ** 2) * 0.45 * 355 / 360
    assert v[1]["curved"] == pytest.approx(exact1, rel=1e-5)
    assert v[6]["curved"] == pytest.approx(exact6, rel=1e-5)
    assert v[1]["straight"] < exact1 * (1 - 1e-3)                          # chords: the curving matters


@needs(TOROID)
def test_toroid_replay_signatures():
    from axionfem.comsol_replay import replay_geometry
    m = cm.load_model(TOROID)
    objs, log = replay_geometry(m, "geom1")
    cells = [c for o in objs for c in o.cells if not c.sheet]
    sheets = [c for o in objs for c in o.cells if c.sheet]
    assert len(cells) == 10 and len(sheets) == 2                 # 9 layer cells + the shield; bridge and port
    # a point in the toroid interior is in exactly the interior cell and the shield
    p = np.array([[0.75 * math.cos(1.0), 0.75 * math.sin(1.0), 0.25]])
    inside = [c.name for c in cells if c.contains(p)[0]]
    assert sorted(inside) == ["cyl1", "rev1:wp1:sq1[1,1]"]
    # the gap at phi in (355, 360) is outside every toroid cell
    q = np.array([[0.75 * math.cos(math.radians(357.5)), 0.75 * math.sin(math.radians(357.5)), 0.25]])
    assert [c.name for c in cells if c.contains(q)[0]] == ["cyl1"]


# ------------------------------------------------------------------ end to end (needs the core)
def _core_3d():
    try:
        import axionfem as af
        return af.build_info().get("engine_3d", False)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _core_3d(), reason="needs the 3D engine")
def test_synthetic_end_to_end(tmp_path):
    import axionfem as af
    mph = synthetic_mph(tmp_path / "synthetic.mph", n=4)
    out = {}
    for order in (1, 2):
        d = ci.import_comsol(mph, tmp_path / f"o{order}" / "design.json", frequencies=[1e6], order=order)
        r = d.solve(tmp_path / f"run{order}", verbose=0)
        fw = r.data["fullwave"]
        s = fw["sweep"][0]
        assert fw["ports"][0]["feed"].startswith("uniform surface current")
        assert fw["ports"][0]["width_m"] == pytest.approx(0.5)
        assert abs(s["Z_re_ohm"]) < 1e-9 * abs(s["Z_im_ohm"])           # lossless
        assert s["Z_im_ohm"] < 0                                          # inductive, e^{-i w t}
        out[order] = s
    L1 = -out[1]["Z_im_ohm"] / (2 * math.pi * 1e6)
    L2 = -out[2]["Z_im_ohm"] / (2 * math.pi * 1e6)
    assert L1 == pytest.approx(L2, rel=0.25)
    assert out[2]["Psi_abs"] > 0


# ------------------------------------------------------------------ blocking features, partial imports, units
def _variant(tmp_path, name, *subs):
    xml = _DMODEL
    for a, b in subs:
        assert a in xml, a
        xml = xml.replace(a, b, 1)
    P, T, tris, tent = _box_mesh(4)
    mph = tmp_path / f"{name}.mph"
    with zipfile.ZipFile(mph, "w") as z:
        z.writestr("dmodel.xml", xml)
        z.writestr("mesh1.mphbin", _write_mesh_v7(P, T, tris, tent))
    return mph


_ECD_SEL = '<explicit dim="3" hDim="3" geom="/geom/geom1" entities="2,1"/></selection>\n          <param T="33" param="Je"'


@pytest.mark.parametrize("name,subs,match", [
    ("unknown_selection", [(_ECD_SEL, '<named>sel9</named></selection>\n          <param T="33" param="Je"')],
     "selection"),
    # (enabled: an unmappable *disabled* source is reported without blocking, Z is unaffected)
    ("complex_je", [("value=\"3|1,'0'|1,'2[A/m^2]'|1,'0'\"", "value=\"3|1,'0'|1,'2*i[A/m^2]'|1,'0'\""),
                    ('<entityFlags T="51">DISABLED</entityFlags>', "")], "complex Je"),
    ("local_system", [("value=\"3|1,'0'|1,'2[A/m^2]'|1,'0'\"/>",
                       "value=\"3|1,'0'|1,'2[A/m^2]'|1,'0'\"/><param T=\"33\" param=\"coordinateSystem\" value=\"1|1,'sys2'\"/>")],
     "coordinate system"),
    ("missing_mu", [("""        <param T="33" param="relpermeability" value="9|1,'1'|1,'0'|1,'0'|1,'0'|1,'1'|1,'0'|1,'0'|1,'0'|1,'1'"/>\n""", "")],
     "relpermeability"),
    ("magnetic_losses", [("""<param T="33" param="DisplacementFieldModel" value="1|1,'RelativePermittivity'"/>""",
                          """<param T="33" param="DisplacementFieldModel" value="1|1,'RelativePermittivity'"/><param T="33" param="ConstitutiveRelationH" value="1|1,'MagneticLosses'"/>""")],
     "magnetic constitutive"),
])
def test_unmappable_features_block_the_import(tmp_path, name, subs, match):
    mph = _variant(tmp_path, name, *subs)
    with pytest.raises(ci.ImportError_, match=match):
        ci.import_comsol(mph, tmp_path / name / "design.json")
    assert not (tmp_path / name / "design.json").exists()


def test_no_design_written_when_blocking(tmp_path):
    extra = ('<PhysicsFeature op="SurfaceCurrentDensity" tag="scd1" name="S"><selection T="32" selection="XDLocalSelection" '
             'selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" entities="-8,-1,1"/></selection>'
             '</PhysicsFeature>\n      </PhysicsFeatureList>')
    mph = _variant(tmp_path, "scat", ("</PhysicsFeatureList>", extra))
    with pytest.raises(ci.ImportError_):
        ci.import_comsol(mph, tmp_path / "s" / "design.json")
    assert not (tmp_path / "s" / "design.json").exists() and (tmp_path / "s" / "import_report.md").exists()
    d = ci.import_comsol(mph, tmp_path / "s2" / "design.json", allow_partial=True)
    doc = d.doc if hasattr(d, "doc") else d
    assert doc["notes"].startswith("INCOMPLETE IMPORT")


def test_boundary_material_ignored(tmp_path):
    mph = _variant(tmp_path, "bmat", ('<Material op="Common" tag="mat1" name="Dielectric">\n      <selection T="32" '
                                      'selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="3" hDim="3" geom="/geom/geom1"/>',
                                      '<Material op="Common" tag="mat1" name="Dielectric">\n      <selection T="32" '
                                      'selection="XDLocalSelection" selType="GEOMDIM"><explicit dim="2" hDim="2" geom="/geom/geom1" entities="-8,-1,1"/>'))
    # the only material is on a boundary: the domain has no material -> blocked (COMSOL would not solve either)
    with pytest.raises(ci.ImportError_, match="no material"):
        ci.import_comsol(mph, tmp_path / "b" / "design.json")


def test_function_units_and_punit():
    f, _ = cm._interp_function([["1", "10"], ["2", "20"]], "const", "g")
    g = cm._with_units(f, ["MHz"], "mS/m")
    assert g(1.5e6) == pytest.approx(15e-3)
    assert cm.parse_list("f0 2", {"f0": 2.0}, unit="MHz") == pytest.approx([2e6, 2e6])          # dimensionless f0
    assert cm.parse_list("f0 2", {"f0": 2e6}, unit="MHz", unit_params={"f0"}) == pytest.approx([2e6, 2e6])
    assert cm.parse_list("range(1,1,f1)", {"f1": 3.0}, unit="MHz") == pytest.approx([1e6, 2e6, 3e6])


def test_cone_band_curving():
    n = 40
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.array([[r * np.cos(t), r * np.sin(t), z] for z, r in ((0.0, 1.0), (0.1, 0.5)) for t in th])
    tri = []
    for i in range(n):
        a, b = i, (i + 1) % n
        tri += [[a, b, b + n], [a, b + n, a + n]]
    cmid = ci.curve_midpoints(P, np.array(tri), np.ones(2 * n, int))
    err = [abs(np.hypot(m[0], m[1]) - (1.0 - 5 * m[2])) for m in cmid.values()]
    assert cmid and max(err) < 1e-9


def test_disabled_unmappable_source_does_not_block(tmp_path):
    """A disabled External Current Density (an impedance run) that cannot be mapped is reported, the import goes on
    (Z does not depend on it; no fullwave.source)."""
    mph = _variant(tmp_path, "cje", ("value=\"3|1,'0'|1,'2[A/m^2]'|1,'0'\"", "value=\"3|1,'0'|1,'2*i[A/m^2]'|1,'0'\""))
    d = ci.import_comsol(mph, tmp_path / "c" / "design.json")
    assert not d.import_report.blocking and "source" not in d.doc["fullwave"]
    assert any(i["comsol"].startswith("ecd1") and i["status"] == "unsupported" for i in d.import_report.items)
