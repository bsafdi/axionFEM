"""Persistent superconducting circuits: induced current from flux conservation."""
import json
import math

import pytest

import axionfem as af


def _design(passive):
    regions = [
        {"name": "sol", "shape": {"rect": {"r": [1.0, 1.05], "z": [-1, 1]}}, "material": "winding"},
        {"name": "pc", "shape": {"rect": {"r": [0.5, 0.52], "z": [-0.05, 0.05]}}, "material": "winding"},
    ]
    w = []
    circuits = []
    if passive == "persistent":
        w = [{"region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
             {"region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}]
        circuits = [{"name": "loop", "type": "persistent"}]
    elif passive == "self":
        w = [{"region": "pc", "type": "azimuthal", "ampere_turns": 100.0}]
    else:
        w = [{"region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
             {"name": "pc", "region": "pc", "type": "azimuthal", "ampere_turns": 0.0}]
    doc = {"schema": "axionfem.design/1", "name": f"circ_{passive}", "regions": regions, "windings": w,
           "mesh": {"order": 3, "size": 0.08, "surface_size": 0.01}}
    if circuits:
        doc["circuits"] = circuits
    return af.Design(doc)


def test_persistent_coil_current(tmp_path):
    rp = _design("persistent").solve(tmp_path / "p", verbose=0)
    rs = _design("self").solve(tmp_path / "s", verbose=0)
    rm = _design("mutual").solve(tmp_path / "m", verbose=0)
    circ = rp.data["static"]["poloidal"]["solver"]["circuits"][0]
    phi1 = rs.data["windings"]["pc"]["flux_per_turn_Wb"]          # per turn, coil current 1 A
    phis = rm.data["windings"]["pc"]["flux_per_turn_Wb"]          # per turn, from the solenoid
    expected = -phis / phi1
    assert abs(circ["flux_linkage_Wb"]) < 1e-9 * abs(100 * phis)
    assert circ["current_A"] == pytest.approx(expected, rel=1e-6)
    # the solenoid's force on the coil is radial only (centred): F_z ~ 0
    assert abs(rp.data["forces"]["pc"]["Fz_N"]) < 1e-4 * abs(rp.data["forces"]["pc"]["radial_force_per_radian_N"]) + 1e-9
