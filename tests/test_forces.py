"""Lorentz forces (axisymmetric): the winding self-force floor is removed, and Newton's-third-law closure is
reported."""
import pytest

import axionfem as af


def _loop_doc(with_sphere: bool):
    regions = [{"name": "coil", "shape": {"rect": {"r": [0.49, 0.51], "z": [-0.01, 0.01]}}, "material": "winding"}]
    if with_sphere:
        regions.append({"name": "ball", "shape": {"disk": {"center": [0, 0.5], "radius": 0.08}},
                        "material": "superconductor"})
    return {"schema": "axionfem.design/1", "name": "loopforce", "regions": regions,
            "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "mesh": {"order": 2, "size": 0.04, "surface_size": 0.008}}


def test_isolated_loop_has_no_net_force(tmp_path):
    r = af.Design(_loop_doc(False)).solve(tmp_path / "a", verbose=0)
    f = r.data["forces"]["coil"]
    hoop = abs(f["radial_force_per_radian_N"]) * 2 * 3.141592653589793
    assert abs(f["Fz_excluding_self_N"]) < 1e-9 * hoop
    assert abs(f["Fz_self_N"]) > 1e-8 * hoop   # the discrete residual that is removed


def test_loop_and_superconducting_sphere_forces_balance(tmp_path):
    r = af.Design(_loop_doc(True)).solve(tmp_path / "b", verbose=0)
    fc = r.data["forces"]["coil"]["Fz_excluding_self_N"]
    fs = r.data["forces"]["superconductor:ball"]["Fz_N"]
    print("coil", fc, "sphere", fs)
    assert fs > 0 and fc < 0                       # the diamagnetic sphere is repelled upwards
    assert fc == pytest.approx(-fs, rel=0.05)
    clos = r.data["forces"]["closure"]
    print(clos)
    assert abs(clos["sum_Fz_N"]) < 0.05 * abs(fs)
