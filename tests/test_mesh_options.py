"""Mesh and geometry options of the axisymmetric engine and the 3D mesher.

1. per-region and per-curve `surface_size` (and its interplay with `mesh_size` and the global sizes);
2. per-vertex polygon fillets;
3. `mesh.domain_center_z`: an off-origin domain, with the exterior (DtN) expansion about (0, z0).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import axionfem as af
from axionfem import _core

ROOT_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 1. mesh size control
def _ring_design(**mesh_extra):
    return {"schema": "axionfem.design/1", "name": "ring",
            "regions": [{"name": "coil", "material": "winding",
                         "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.1, 0.1]}}, **mesh_extra}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "mesh": {"order": 1, "size": 0.25, "surface_size": 0.25}}


def _two_region_design(global_hs=0.25, coil_hs=None, shield_hs=None):
    r = [{"name": "coil", "material": "winding", "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.2, 0.2]}}},
         {"name": "shield", "material": "nb", "shape": {"rect": {"r": [1.4, 1.6], "z": [-0.2, 0.2]}}}]
    if coil_hs:
        r[0]["surface_size"] = coil_hs
    if shield_hs:
        r[1]["surface_size"] = shield_hs
    return {"schema": "axionfem.design/1", "name": "two", "regions": r,
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "superconductors": [{"body": "shield", "azimuthal": {"flux": 0}}],
            "mesh": {"order": 1, "size": 0.25, "surface_size": global_hs}}


def _elements(doc, wd):
    return af.Design(doc).solve(workdir=wd, verbose=0).data["mesh"]["elements"]


def test_region_surface_size_matches_the_global_one_and_is_selective(tmp_path):
    """Per-region `surface_size` must do exactly what the global `mesh.surface_size` does when every
    refined region asks for it, and refine only the named region when only one does."""
    coarse = _elements(_two_region_design(), tmp_path / "coarse")
    glob = _elements(_two_region_design(global_hs=0.02), tmp_path / "glob")
    both = _elements(_two_region_design(coil_hs=0.02, shield_hs=0.02), tmp_path / "both")
    coil_only = _elements(_two_region_design(coil_hs=0.02), tmp_path / "coil")
    assert glob > coarse
    assert both == glob                       # same fields, same mesh
    assert coarse < coil_only < glob          # only the coil boundary was refined
    finer = _elements(_two_region_design(coil_hs=0.004), tmp_path / "finer")
    assert finer > 1.5 * coil_only            # and it keeps getting finer


def test_region_surface_size_overrides_the_global_one(tmp_path):
    """A region that names its own surface_size follows it, not mesh.surface_size."""
    a = _elements(_two_region_design(global_hs=0.02, coil_hs=0.02, shield_hs=0.02), tmp_path / "a")
    b = _elements(_two_region_design(global_hs=0.2, coil_hs=0.02, shield_hs=0.02), tmp_path / "b")
    assert a == b


def _sheet_design(global_hs=0.25, curve_hs=None):
    c = {"name": "disc", "material": "nb", "path": {"segment": [[0.2, 0.5], [0.8, 0.5]]}}
    if curve_hs:
        c["surface_size"] = curve_hs
    return {"schema": "axionfem.design/1", "name": "sheet",
            "regions": [{"name": "coil", "material": "winding",
                         "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.1, 0.1]}},
                         "surface_size": 0.25}],
            "curves": [c],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "superconductors": [{"body": "disc", "azimuthal": {"flux": 0}}],
            "mesh": {"order": 1, "size": 0.25, "surface_size": global_hs}}


def test_curve_surface_size_matches_the_global_one_and_keeps_refining(tmp_path):
    """A sheet with its own `surface_size` is meshed exactly as the global `mesh.surface_size`
    would, and ignores the global value."""
    coarse = _elements(_sheet_design(), tmp_path / "c")
    glob = _elements(_sheet_design(global_hs=0.02), tmp_path / "g")
    own = _elements(_sheet_design(curve_hs=0.02), tmp_path / "o")
    override = _elements(_sheet_design(global_hs=0.2, curve_hs=0.02), tmp_path / "ov")
    finer = _elements(_sheet_design(curve_hs=0.004), tmp_path / "f")
    assert glob > coarse
    assert own == glob
    assert override == own            # the curve's own value wins
    assert finer > 1.5 * own


def test_region_mesh_size_does_not_leak_out_through_surface_size(tmp_path):
    """A region that names both mesh_size and surface_size must refine only itself.  The per-region
    refinement is a gmsh Threshold field, which returns SizeMax at every point beyond DistMax and is
    not restricted to the region, so SizeMax must never undercut the device size -- otherwise the
    region's mesh_size caps the element size of the whole half-disk."""
    surface_only = _two_region_design(coil_hs=0.02)
    both = _two_region_design(coil_hs=0.02)
    both["regions"][0]["mesh_size"] = 0.02
    a = _elements(surface_only, tmp_path / "s")
    b = _elements(both, tmp_path / "b")
    assert b > a                   # the coil interior really is finer
    assert b < 3 * a, (a, b)       # but the rest of the domain is untouched (it was ~30x)


def test_touching_fillets_mesh(tmp_path):
    """Two adjacent corners filleted with half the edge length: their tangent points meet exactly at
    the midpoint of the shared edge, so the two arcs must share one point and no zero-length edge."""
    r = 0.3  # half of the 0.6 m edge between (1.0, -0.4) and (1.0, 0.4)
    d = af.Design(_poly_design([[0.4, -0.4], [1.0, -0.4, r], [1.0, 0.4, r], [0.4, 0.4]]))
    d.validate()
    res = d.solve(workdir=tmp_path / "touch", verbose=0)
    assert res.data["mesh"]["elements"] > 0
    assert _sharp_corners(res) == {(0.4, -0.4), (0.4, 0.4)}



def _mesher() -> str:
    return os.environ.get("AXF_MESHER") or str(ROOT_DIR / "build" / "axf-mesher")


def _mesh3d(tmp_path, tag: str, global_hs: float, region_hs: float | None = None) -> int:
    """Drive axf-mesher's 3D routine directly: cpp/src/model3d.cpp does not pass the per-region key to it, so
    the contract is tested at the mesher."""
    reg: dict = {"name": "slug", "attr": 2,
                 "solid": {"cylinder": {"base": [0, 0, -0.2], "axis": [0, 0, 0.4], "radius": 0.3}}}
    if region_hs:
        reg["surface_size"] = region_hs
    info = tmp_path / f"{tag}.json"
    inp = {"dim": 3, "domain_radius": 2.0, "size": 0.25, "surface_size": global_hs, "order": 1,
           "regions": [reg], "refine_regions": ["slug"], "outer_attr": 901, "air_attr": 1,
           "output": str(tmp_path / f"{tag}.msh"), "info": str(info), "verbose": 0}
    input_file = tmp_path / f"{tag}_in.json"
    input_file.write_text(json.dumps(inp))
    p = subprocess.run([_mesher(), str(input_file)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(info.read_text())
    assert got["ok"], got
    return int(got["elements_3d"])


def test_mesher_3d_honours_a_per_region_surface_size(tmp_path):
    coarse = _mesh3d(tmp_path, "coarse", 0.25)
    glob = _mesh3d(tmp_path, "glob", 0.05)
    own = _mesh3d(tmp_path, "own", 0.25, 0.05)
    fine = _mesh3d(tmp_path, "fine", 0.25, 0.02)
    assert glob > 2 * coarse
    # the same fields either way; gmsh's threaded 3D algorithm is not element-for-element
    # reproducible, so allow a few per cent rather than demanding equality as in 2D
    assert own == pytest.approx(glob, rel=0.03), (glob, own)
    assert fine > 2 * own


@pytest.mark.parametrize("bad,where", [
    ({"surface_size": -1}, "/regions/0/surface_size"),
    ({"mesh_size": -0.1}, "/regions/0/mesh_size"),
])
def test_negative_sizes_are_rejected(bad, where):
    with pytest.raises(Exception) as e:
        af.Design(_ring_design(**bad)).validate()
    assert where in str(e.value)


def test_surface_size_is_in_the_schema():
    s = json.loads(_core.schema())["properties"]
    assert "surface_size" in s["regions"]["items"]["properties"]
    assert "surface_size" in s["curves"]["items"]["properties"]
    assert "domain_center_z" in s["mesh"]["properties"]


# ---------------------------------------------------------------- 2. per-vertex polygon fillets
def _poly_design(points, fillets=None, fillet=None):
    poly: dict = {"points": points}
    if fillets is not None:
        poly["fillets"] = fillets
    if fillet is not None:
        poly["fillet"] = fillet
    return {"schema": "axionfem.design/1", "name": "poly",
            "regions": [{"name": "coil", "material": "winding",
                         "shape": {"rect": {"r": [1.4, 1.5], "z": [-0.5, 0.5]}}},
                        {"name": "body", "material": "nb", "shape": {"polygon": poly}}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "superconductors": [{"body": "body", "azimuthal": {"flux": 0}}],
            "mesh": {"order": 2, "size": 0.1, "surface_size": 0.02}}


SQUARE = [[0.4, -0.4], [1.0, -0.4], [1.0, 0.4], [0.4, 0.4]]


def _sharp_corners(res):
    """The convex sharp superconductor corners the solver detected, as (r, z) rounded to mm."""
    return {(round(c["r_m"], 3), round(c["z_m"], 3))
            for c in res.data["fields"]["superconductor_sharp_corners"]}


def test_per_vertex_fillets_round_only_the_chosen_corners(tmp_path):
    """The solver flags convex sharp superconductor corners.  Rounding two of the four corners of a square must
    leave exactly the other two flagged."""
    a = 0.15
    sharp = af.Design(_poly_design(SQUARE)).solve(workdir=tmp_path / "sharp", verbose=0)
    two = af.Design(_poly_design([[0.4, -0.4], [1.0, -0.4, a], [1.0, 0.4, a], [0.4, 0.4]])).solve(
        workdir=tmp_path / "two", verbose=0)
    four = af.Design(_poly_design(SQUARE, fillet=a)).solve(workdir=tmp_path / "four", verbose=0)
    assert _sharp_corners(sharp) == {(0.4, -0.4), (1.0, -0.4), (1.0, 0.4), (0.4, 0.4)}
    assert _sharp_corners(two) == {(0.4, -0.4), (0.4, 0.4)}   # the outer two were rounded
    assert _sharp_corners(four) == set()
    # rounding removes (1 - pi/4) a^2 of cross-section at each corner, so the body shrinks
    assert sharp.W != two.W != four.W


def test_fillets_list_equals_third_coordinate(tmp_path):
    a = 0.12
    by_point = af.Design(_poly_design(
        [[0.4, -0.4], [1.0, -0.4, a], [1.0, 0.4, a], [0.4, 0.4]])).solve(workdir=tmp_path / "p", verbose=0)
    by_list = af.Design(_poly_design(SQUARE, fillets=[0, a, a, 0])).solve(workdir=tmp_path / "l", verbose=0)
    assert by_point.W == pytest.approx(by_list.W, rel=1e-12)


def test_too_large_fillets_are_rejected_with_the_vertices_named():
    # the two radii together need more of the shared edge than it is long
    with pytest.raises(Exception) as e:
        af.Design(_poly_design([[0.4, -0.4], [1.0, -0.4, 0.45], [1.0, 0.4, 0.45], [0.4, 0.4]])).validate()
    msg = str(e.value)
    assert "fillet" in msg and "edge" in msg


def test_fillets_and_third_coordinate_cannot_be_mixed():
    with pytest.raises(Exception) as e:
        af.Design(_poly_design([[0.4, -0.4], [1.0, -0.4, 0.1], [1.0, 0.4], [0.4, 0.4]],
                               fillets=[0, 0.1, 0.1, 0])).validate()
    assert "not both" in str(e.value)


def test_fillets_length_must_match_the_points():
    with pytest.raises(Exception) as e:
        af.Design(_poly_design(SQUARE, fillets=[0.1, 0.1])).validate()
    assert "one per polygon point" in str(e.value)


# ---------------------------------------------------------------- 3. off-origin domain centre
def _shifted_toroid(z0):
    d = af.template("toroid", R=1.0, H=1.0).doc
    d["parameters"]["z0"] = z0
    for reg in d["regions"]:
        f = reg["shape"]["frame"]
        f["z"] = [f"({f['z'][0]})+z0", f"({f['z'][1]})+z0"]
    d["mesh"]["domain_center_z"] = "z0"
    return d


def test_domain_center_z_is_translation_invariant(tmp_path):
    """Shifting the whole device in z and the domain centre with it must not change the physics:
    without domain_center_z the outer boundary (and the DtN multipole origin) stays at z = 0 and the
    domain has to grow with the offset."""
    base = af.Design(_shifted_toroid(0.0)).solve(workdir=tmp_path / "z0", verbose=0)
    far = af.Design(_shifted_toroid(10.0)).solve(workdir=tmp_path / "z10", verbose=0)
    assert far.Psi == pytest.approx(base.Psi, rel=2e-6)
    assert far.L_p == pytest.approx(base.L_p, rel=2e-6)
    assert far.W == pytest.approx(base.W, rel=2e-6)
    # and the mesh is the same size: the domain followed the device
    assert far.data["mesh"]["dofs"] == pytest.approx(base.data["mesh"]["dofs"], rel=0.05)
    assert far.data["mesh"]["info"]["settings"]["domain_center_z"] == 10.0


def test_scene_sampling_and_render_follow_an_off_origin_device(tmp_path):
    """Everything downstream of the solve has to follow the device, not the origin: the scene's view
    window, point sampling and the PNG rasterizer."""
    base = af.Design(_shifted_toroid(0.0)).solve(workdir=tmp_path / "s0", verbose=0)
    far = af.Design(_shifted_toroid(12.0)).solve(workdir=tmp_path / "s12", verbose=0)
    v0, v1 = base.scene()["view"], far.scene()["view"]
    assert v1["r_max"] == pytest.approx(v0["r_max"], rel=1e-9)
    assert (v1["z_min"] + v1["z_max"]) / 2 == pytest.approx(12.0, abs=1e-6)
    assert (v1["z_max"] - v1["z_min"]) == pytest.approx(v0["z_max"] - v0["z_min"], rel=1e-9)
    # the same field at the same place inside the device
    b0 = abs(base.sample([0.8], [0.0])["Bphi"][0])
    b1 = abs(far.sample([0.8], [12.0])["Bphi"][0])
    assert b0 > 0.1 and b1 == pytest.approx(b0, rel=1e-9)
    png = far.png(tmp_path / "far.png")
    assert png.stat().st_size > 10_000


def test_domain_radius_check_uses_the_domain_centre():
    d = af.Design(_shifted_toroid(10.0))
    d.doc["mesh"]["domain_radius"] = 1.0  # smaller than the device, about (0, 10)
    with pytest.raises(Exception) as e:
        d.solve(workdir="unused", verbose=0)
    assert "domain_center_z" in str(e.value) or "domain centre" in str(e.value)
