"""Full-wave 3D conductor-surface quadrature export (fullwave.surface_quadrature_file).

The shorted parallel-plate TEM line of benchmarks/fw3d_robin_validation.py: PEC plates at y = 0 and y = h, PMC sides,
a surface port at x = 0 and a PEC short at x = l. The TEM field is exact on this straight mesh up to the axial
discretization: H = H_z(x) z between the plates, with the line current I(x) = cos(k (l - x))/cos(k l) per incoming
port ampere, so |B| = mu0 |I(x)|/W on the plates and mu0/(W |cos k l|) on the short, and n . B = 0 on every PEC face.

Checks: the selected groups (conductor PEC groups only: no port, PMC side or outer
boundary), the total area, the unit normals into the field region, the tangential field against the TEM closed form,
one file per frequency row, and the PML exclusion.
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import fw3d_robin_validation as robin  # noqa: E402  (the TEM line and its closed form)

MU0 = robin.MU0

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (MFEM/hypre build)")


def _read(path):
    raw = Path(path).read_bytes()
    npts, nports = struct.unpack("2d", raw[:16])
    npts = int(npts)
    nports = int(nports)
    body = np.frombuffer(raw[16:], dtype=np.float64)
    pts = body[:8 * npts].reshape(npts, 8)
    B = body[8 * npts:].reshape(nports, npts, 6)
    return pts, B


def test_tem_line_surface_quadrature(tmp_path):
    l, h, W = 0.3, 0.02, 0.04
    freqs = [5e7, 1e8]
    doc, geometry = robin.build(tmp_path, termination="pec", plates="pec", l=l, h=h, W=W, nx=12, order=2)
    doc["fullwave"]["frequencies"] = freqs
    doc["fullwave"]["surface_quadrature_file"] = str(tmp_path / "surf")
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    fw = data["fullwave"]
    sq = fw["surface_quadrature"]
    # plate_bottom 10003, plate_top 10004, end 10002; not the port 10001 and not the PMC sides
    assert sq["attributes"] == [10002, 10003, 10004]
    assert abs(sq["area_m2"] - (2 * l * W + h * W)) < 1e-12
    assert sq["skipped_faces_pml"] == 0 and sq["skipped_faces_pec_side"] == 0 and sq["skipped_faces_shell"] == 0
    assert sq["skipped_faces_interior"] == 0
    assert sum(sq["points_per_attribute"].values()) == sq["points"]
    for i, f in enumerate(freqs):
        row = fw["sweep"][i]
        assert row["surface_quadrature_file"].endswith(f"surf.f{i}.rank0.bin")
        pts, B = _read(row["surface_quadrature_file"])
        assert B.shape[0] == 1 and pts.shape[0] == sq["points"]
        x = pts[:, 0:3]
        n = pts[:, 3:6]
        attr = pts[:, 7].astype(int)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
        assert np.allclose(n[attr == 10003], [0, 1, 0], atol=1e-12)
        assert np.allclose(n[attr == 10004], [0, -1, 0], atol=1e-12)
        assert np.allclose(n[attr == 10002], [-1, 0, 0], atol=1e-12)
        Bc = B[0, :, 0:3] + 1j * B[0, :, 3:6]
        k = 2 * math.pi * f / robin.C0
        current = np.cos(k * (l - x[:, 0])) / math.cos(k * l)
        exact = MU0 * np.abs(current) / W
        mag = np.linalg.norm(Bc, axis=1)
        normal = np.abs(np.einsum("ij,ij->i", Bc, n))
        assert np.max(normal) < 1e-6 * np.max(mag)
        rel = np.abs(mag - exact) / exact
        assert np.max(rel) < 1e-2, np.max(rel)
        # the field is H_z alone: B points along z (1e-5 of |B| in B_x from the discretization)
        assert np.max(np.abs(Bc[:, 0])) < 1e-4 * np.max(mag)
        assert np.max(np.abs(Bc[:, 1])) < 1e-4 * np.max(mag)


def test_pml_faces_are_skipped(tmp_path):
    """The PML-terminated line: the plates inside the PML (group 10007) border PML elements only and are skipped."""
    doc, _ = robin.build(tmp_path, termination="pml", plates="pec", nx=6, order=1, pml_cells=2)
    doc["fullwave"]["frequencies"] = [3e8]
    doc["fullwave"]["surface_quadrature_file"] = str(tmp_path / "surf")
    sq = af.Design(doc).solve(tmp_path, verbose=0).data["fullwave"]["surface_quadrature"]
    assert 10007 in sq["attributes"]
    assert sq["skipped_faces_pml"] > 0
    pts, _ = _read(tmp_path / "surf.f0.rank0.bin")
    assert 10007 not in set(pts[:, 7].astype(int))
    assert abs(sq["area_m2"] - 2 * 0.3 * 0.04) < 1e-12   # plates up to x = l; the PML end short borders PML elements too


def test_generated_mesh_with_pec_hole(tmp_path):
    """A slit superconducting ring (tests/test_fullwave3d.py's loop) meshed with mesh.remove_pec, rf_only, in the
    closed PEC sphere of radius 1 m: the ring's walls are the mesher's group pec_body_0 (1000) and the sphere is the
    outer group (901). Both are conductors and both are exported; there is no infinity shell. Order 1, 2 s."""
    from axionfem.convert import to_3d
    a, h, slit, R = 0.3, 0.02, 6.0, 1.0
    doc2 = {"schema": "axionfem.design/1", "name": "loop",
            "regions": [{"name": "ring", "shape": {"rect": {"r": [a - h / 2, a + h / 2], "z": [-h / 2, h / 2]}},
                         "material": "superconductor"}],
            "superconductors": [{"body": "ring", "azimuthal_slit": True}],
            "pickup": {"type": "azimuthal", "conductor": "ring"},
            "fullwave": {"frequencies": [1e7], "rf_only": True, "boundary": "pec"},
            "mesh": {"order": 2, "size": 0.02, "surface_size": 0.006}}
    doc = to_3d(af.Design(doc2), pickup="slit", slit_deg=slit,
                mesh={"size": 0.2, "surface_size": 0.04, "order": 1, "domain_radius": R}).doc
    doc["mesh"]["remove_pec"] = True
    doc["pickup"]["toward"] = [0, 1, 0]      # the slit is at phi = 0: the gap current runs along y
    doc["fullwave"]["surface_quadrature_file"] = str(tmp_path / "sub" / "dir" / "surf")   # created at setup
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    sq = data["fullwave"]["surface_quadrature"]
    assert sq["attributes"] == [901, 1000]
    assert not any("surface_quadrature" in w for w in data["warnings"])
    assert sq["skipped_faces_shell"] == 0 and sq["skipped_faces_pml"] == 0 and sq["skipped_faces_pec_side"] == 0
    pts, B = _read(tmp_path / "sub" / "dir" / "surf.f0.rank0.bin")
    x = pts[:, 0:3]
    n = pts[:, 3:6]
    dA = pts[:, 6]
    attr = pts[:, 7].astype(int)
    ring = attr == 1000
    sphere = attr == 901
    assert ring.sum() + sphere.sum() == len(attr)
    # areas: the slit ring, 8 pi a h over (360 - slit) degrees plus the two slit faces; the faceted sphere
    ring_area = 8 * math.pi * a * h * (360 - slit) / 360 + 2 * h * h
    assert abs(dA[ring].sum() / ring_area - 1) < 2e-3
    assert abs(dA[sphere].sum() / (4 * math.pi * R * R) - 1) < 2e-2
    assert abs(sq["area_m2"] - dA.sum()) < 1e-12
    # unit normals out of the conductor into the field: away from the ring's cross-section centre (off the slit
    # faces), toward the centre on the sphere
    rho = np.hypot(x[:, 0], x[:, 1])
    centre = np.stack([a * x[:, 0] / rho, a * x[:, 1] / rho, np.zeros_like(rho)], axis=1)
    outward = np.einsum("ij,ij->i", n, x - centre)
    on_slit = np.abs(np.degrees(np.arctan2(x[:, 1], x[:, 0]))) < slit / 2 + 0.5
    assert np.all(outward[ring & ~on_slit] > 0)
    radial = np.einsum("ij,ij->i", n, x) / np.linalg.norm(x, axis=1)
    assert np.all(radial[sphere] < -0.95)       # flat facets of an order-1 sphere: within 15 degrees of radial
    # n . B = 0 on a perfect conductor: exact here (the normal curl of the vanishing tangential trace)
    Bc = B[0, :, 0:3] + 1j * B[0, :, 3:6]
    mag = np.linalg.norm(Bc, axis=1)
    assert np.all(mag[ring] > 0)
    assert np.max(np.abs(np.einsum("ij,ij->i", Bc, n)) / mag) < 1e-10


def test_axisymmetric_design_rejects_the_export():
    doc = {"schema": "axionfem.design/1", "name": "loop",
           "regions": [{"name": "ring", "shape": {"rect": {"r": [0.29, 0.31], "z": [-0.01, 0.01]}},
                        "material": "superconductor"},
                       {"name": "coil", "shape": {"rect": {"r": [0.1, 0.12], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "superconductors": [{"body": "ring", "azimuthal_slit": True}],
           "pickup": {"type": "azimuthal", "conductor": "ring"},
           "fullwave": {"frequencies": [1e7], "surface_quadrature_file": "surf"}}
    with pytest.raises(Exception, match="surface_quadrature_file is for 3D"):
        af.Design(doc).validate()


def test_export_needs_rf_only():
    """In a regular 3D design the conductors are volume elements, so nothing could be exported; the key
    is refused rather than producing empty files."""
    from axionfem.convert import to_3d
    doc2 = {"schema": "axionfem.design/1", "name": "loop",
            "regions": [{"name": "ring", "shape": {"rect": {"r": [0.29, 0.31], "z": [-0.01, 0.01]}},
                         "material": "superconductor"},
                        {"name": "coil", "shape": {"rect": {"r": [0.1, 0.12], "z": [-0.05, 0.05]}},
                         "material": "winding"}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
            "superconductors": [{"body": "ring", "azimuthal_slit": True}],
            "pickup": {"type": "azimuthal", "conductor": "ring"},
            "fullwave": {"frequencies": [1e7]},
            "mesh": {"order": 2, "size": 0.02, "surface_size": 0.006}}
    doc = to_3d(af.Design(doc2), pickup="slit", slit_deg=6.0).doc
    doc["fullwave"]["surface_quadrature_file"] = "surf"
    with pytest.raises(Exception, match="needs fullwave.rf_only"):
        af.Design(doc).validate()


def test_only_the_outer_wall_is_warned_about(tmp_path):
    """rf_only without mesh.remove_pec: the ring is a volume, only the closed sphere (901) is exported,
    with a warning."""
    from axionfem.convert import to_3d
    a, h = 0.3, 0.02
    doc2 = {"schema": "axionfem.design/1", "name": "loop",
            "regions": [{"name": "ring", "shape": {"rect": {"r": [a - h / 2, a + h / 2], "z": [-h / 2, h / 2]}},
                         "material": "superconductor"}],
            "superconductors": [{"body": "ring", "azimuthal_slit": True}],
            "pickup": {"type": "azimuthal", "conductor": "ring"},
            "fullwave": {"frequencies": [1e7], "rf_only": True, "boundary": "pec"},
            "mesh": {"order": 2, "size": 0.02, "surface_size": 0.006}}
    doc = to_3d(af.Design(doc2), pickup="slit", slit_deg=6.0,
                mesh={"size": 0.2, "surface_size": 0.04, "order": 1, "domain_radius": 1.0}).doc
    doc["pickup"]["toward"] = [0, 1, 0]
    doc["fullwave"]["surface_quadrature_file"] = str(tmp_path / "surf")
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    sq = data["fullwave"]["surface_quadrature"]
    assert sq["attributes"] == [901] and sq["points"] > 0
    assert any("only the outer boundary" in w for w in data["warnings"])


def test_unwritable_directory_is_refused_at_setup(tmp_path):
    """An existing directory that cannot be written is refused at setup (a probe file per
    rank), before any solve, and the message names the failing ranks."""
    import os
    ro = tmp_path / "readonly"
    ro.mkdir()
    os.chmod(ro, 0o555)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("running with permissions that ignore the directory mode")
        doc, _ = robin.build(tmp_path / "line", termination="pec", plates="pec", nx=6, order=1)
        doc["fullwave"]["frequencies"] = [1e8]
        doc["fullwave"]["surface_quadrature_file"] = str(ro / "surf")
        with pytest.raises(Exception, match=r"cannot write files with the prefix .*ranks 0"):
            af.Design(doc).solve(tmp_path / "line", verbose=0)
    finally:
        os.chmod(ro, 0o755)
