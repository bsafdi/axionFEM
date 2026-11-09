"""Unit tests of the helpers of benchmarks/srf/srf_vibration_mixing.py.  Fast, no solver."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from axionfem import srf

_P = Path(__file__).resolve().parents[1] / "benchmarks/srf/srf_vibration_mixing.py"
_spec = importlib.util.spec_from_file_location("srf_vibration_mixing", _P)
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)


def test_wall_points_weights_integrate_the_wall_area():
    """sum 2 pi r dl over the midpoint samples = the contour's wall area (exact: r is linear on each segment)."""
    con = srf.elliptical_cell_contour(axial_factor=0.86, ds=2e-3)
    for n in (1, 4):
        P, T, L = vm._wall_points(con["wall"], n_per_seg=n)
        assert float(np.sum(2 * math.pi * P[:, 0] * L)) == pytest.approx(con["A_polygon_m2"], rel=1e-12)
        assert np.allclose(np.hypot(T[:, 0], T[:, 1]), 1.0)
    assert con["params"]["tube_length"] == pytest.approx(0.100)


@pytest.mark.parametrize("theta", [0.0, 0.1, 0.7, math.pi / 2])
def test_m0_projection_recovers_the_mixing_angle(theta):
    """A synthetic mode E = cos(theta) TM + sin(theta) TE (each unit-normalized on the quadrature), built in Cartesian
    components at 8 azimuths, projects to (cos theta, sin theta); a cos 2phi admixture (m = 2) drops out."""
    rng = np.random.default_rng(3)
    npnt, phis = 50, (np.arange(8) + 0.5) * 2 * math.pi / 8
    pts = np.column_stack([rng.uniform(0.01, 0.2, npnt), rng.uniform(-0.1, 0.1, npnt)])
    w = pts[:, 0] * 1e-4
    Er, Ez, Ep = rng.normal(size=npnt), rng.normal(size=npnt), rng.normal(size=npnt)
    nTM, nTE = math.sqrt(np.sum(w * (Er ** 2 + Ez ** 2))), math.sqrt(np.sum(w * Ep ** 2))
    a, b = math.cos(theta) / nTM, math.sin(theta) / nTE
    E = np.zeros((npnt, len(phis), 3))
    for j, ph in enumerate(phis):
        er, ep, ez = a * Er + 0.3 * np.cos(2 * ph) * Ez, b * Ep, a * Ez
        E[:, j, 0] = er * math.cos(ph) - ep * math.sin(ph)
        E[:, j, 1] = er * math.sin(ph) + ep * math.cos(ph)
        E[:, j, 2] = ez
    cTM, cTE, nok = vm._m0_projection(E.reshape(-1, 3), pts, phis, w, (Er, Ez), Ep, np.ones(npnt * len(phis)))
    assert nok == npnt
    # the m = 2 part adds to the mode's own norm (nE) but not to the projections
    nE2 = 1 + 0.5 * 0.3 ** 2 * np.sum(w * Ez ** 2)
    assert cTM * math.sqrt(nE2) == pytest.approx(math.cos(theta), abs=1e-12)
    assert cTE * math.sqrt(nE2) == pytest.approx(math.sin(theta), abs=1e-12)
    assert math.atan2(abs(cTE), abs(cTM)) == pytest.approx(theta, abs=1e-12)


_MSH_ASCII = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 0.1 0.2 -0.3
3 -0.25 0.05 0.4
4 0.3 -0.1 0.2
$EndNodes
$Elements
1
1 4 2 1 1 1 2 3 4
$EndElements
"""


def _nodes(path):
    raw = Path(path).read_bytes()
    i0 = raw.index(b"$Nodes\n") + 7
    i1 = raw.index(b"\n", i0)
    n = int(raw[i0:i1])
    if raw[:raw.index(b"$EndMeshFormat")].split(b"\n")[1].split()[1] == b"1":
        dt = np.dtype([("id", "<i4"), ("xyz", "<f8", 3)])
        return np.frombuffer(raw, dtype=dt, count=n, offset=i1 + 1)["xyz"].copy(), raw
    lines = raw[i1 + 1:raw.index(b"$EndNodes")].decode().split("\n")
    return np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.strip()]), raw


def _binary_msh(xyz):
    dt = np.dtype([("id", "<i4"), ("xyz", "<f8", 3)])
    nodes = np.zeros(len(xyz), dt)
    nodes["id"], nodes["xyz"] = np.arange(1, len(xyz) + 1), xyz
    return (b"$MeshFormat\n2.2 1 8\n" + np.int32(1).tobytes() + b"\n$EndMeshFormat\n$Nodes\n"
            + f"{len(xyz)}\n".encode() + nodes.tobytes() + b"\n$EndNodes\n$Elements\n0\n$EndElements\n")


@pytest.mark.parametrize("binary", [False, True])
def test_affine_msh_identity_and_map(tmp_path, binary):
    """eps = 0 (with or without twist) leaves the nodes unchanged (bit-identical without twist); eps != 0 applies x(1+e), y(1-e), the same map as
    _map_points; the rest of the file is byte-identical."""
    src = tmp_path / "m.msh"
    if binary:
        xyz0 = np.array([[0, 0, 0], [0.1, 0.2, -0.3], [-0.25, 0.05, 0.4], [0.3, -0.1, 0.2]], float)
        src.write_bytes(_binary_msh(xyz0))
    else:
        src.write_text(_MSH_ASCII)
    x0, raw0 = _nodes(src)
    for twist in (0.0, 0.7854):
        dst = tmp_path / f"id_{twist}.msh"
        vm._affine_msh(src, dst, 0.0, twist, 0.5)
        x1, raw1 = _nodes(dst)
        assert np.allclose(x1, x0, atol=1e-15)
        if binary and twist == 0.0:
            assert raw1 == raw0          # bit-identical (with a twist the rotation round trip is exact to 1 ulp)
        e = 0.03
        vm._affine_msh(src, dst, e, twist, 0.5)
        x2, raw2 = _nodes(dst)
        assert np.allclose(x2, vm._map_points(x0, e, twist, 0.5), atol=1e-15)
        assert raw2[-40:] == raw0[-40:]
    x3, _ = _nodes(dst)   # last: e = 0.03, twist: at z = 0 the map is diag(1 + e, 1 - e)
    k = np.abs(x0[:, 2]) < 1e-15
    assert np.allclose(x3[k, 0], 1.03 * x0[k, 0]) and np.allclose(x3[k, 1], 0.97 * x0[k, 1])
