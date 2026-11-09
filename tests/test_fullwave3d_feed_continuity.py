"""Port feeds must be discretely divergence-free: each port's electric energy scales as f^2 down to 10 Hz.

The port current J = -grad V comes from a conduction solve in the port region, and its discrete divergence at the
free port nodes is that solve's residual. At low frequency the field amplifies the resulting charge by about
(k h)^-2: with the solve stopped at its PCG tolerance, the two-port frame's second port has a 1.2e-14 relative
residual and an electric energy 12.3 % high at 100 Hz in every solver path, while every gate passes. The conduction
solve is therefore refined to the rounding floor. What rounding still leaves, about 5e-16 of the right-hand side's
terms at free vertices and in the uncharged conductor rows, gives a 3.5 % field error at 10 Hz on the gauged path; its
default charge projection (AXF_FW_TC_CHARGE_PROJECTION=2) removes it. Here, on the gauged (tree-cotree) direct path:
U_e/f^2 at 10 Hz equals its 1 kHz value to 1e-8 (the physical change is 5e-10); and with the projection off, the
excess is there and the solver's own estimate (rhs_charge_field_fraction) predicts it, with a warning. About 1 min
on one rank. Set AXF_SLOW=1.
"""
import json
import os
from pathlib import Path

import pytest

import axionfem as af

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

DESIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"


def rhs_charges(row):
    """(solve, rhs_free_vertex_charge_relative) for every port solve and validation solve of a sweep row."""
    out = []
    for port in row["port_solutions"]:
        out.append((f"port_{port['port_index']}", port["solver"]["tree_cotree"]["rhs_free_vertex_charge_relative"]))
    for key, value in row.items():
        if key.endswith("_solver") and isinstance(value, dict) and "tree_cotree" in value:
            out.append((key, value["tree_cotree"]["rhs_free_vertex_charge_relative"]))
    assert len(out) >= len(row["port_solutions"]) > 0
    return out


def solve_two_port(tmp_path, monkeypatch, name, projection=None):
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.setenv("AXF_FW_GAUGE", "tree-cotree")
    if projection is None:
        monkeypatch.delenv("AXF_FW_TC_CHARGE_PROJECTION", raising=False)
    else:
        monkeypatch.setenv("AXF_FW_TC_CHARGE_PROJECTION", projection)
    doc = json.loads(DESIGN.read_text())
    doc["fullwave"]["frequencies"] = [1e1, 1e3]
    r = af.Design(doc).solve(tmp_path / name, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    return r.data


def low_frequency_excess(fw):
    """Per port: U_e/f^2 at 10 Hz against 1 kHz, minus 1."""
    sweep = {round(s["f_Hz"]): s for s in fw["sweep"]}
    out = []
    for j in range(2):
        lo = sweep[10]["energy_electric_J_per_A2"]["re"][j][j] / 1e1 ** 2
        hi = sweep[1000]["energy_electric_J_per_A2"]["re"][j][j] / 1e3 ** 2
        out.append(lo / hi - 1)
    return out


def test_port_feed_electric_energy_scales_as_f2(tmp_path, monkeypatch):
    data = solve_two_port(tmp_path, monkeypatch, "default")
    fw = data["fullwave"]
    assert "charge projection 2" in fw["solver_path"]
    # the physical symptom (unrefined feed: 0.123 at 100 Hz; 2e-3 at 10 Hz)
    for j, d in enumerate(low_frequency_excess(fw)):
        assert abs(d) < 1e-8, (j, d)
    for port in fw["ports"]:  # its first cause (unrefined feed: 1.2e-14 for port 2)
        assert port["conduction_relative_residual"] < 1e-16, port
    for row in fw["sweep"]:   # the right-hand side's charge is at rounding in every port and validation solve
        for name, q in rhs_charges(row):
            assert q < 1e-13, (row["f_Hz"], name, q)
        solves = [port["solver"]["tree_cotree"] for port in row["port_solutions"]]
        solves += [v["tree_cotree"] for k, v in row.items() if k.endswith("_solver") and isinstance(v, dict)
                   and "tree_cotree" in v]
        for tc in solves:     # every row is rounding, and the projection removes all of it (conductor rows too)
            assert tc["charged_free_vertex_rows"] == 0, tc
            assert tc["rhs_free_vertex_charge_relative_solved"] == 0.0
            assert tc["rhs_uncharged_conductor_charge_relative_solved"] == 0.0
            assert tc["rhs_charge_field_fraction_solved"] == 0.0
        # the load columns are projected like the feeds: the loaded field closes to rounding (unprojected: 4e-9)
        assert row["loaded_field_closure_relative_error"] < 1e-12, row["loaded_field_closure_relative_error"]
    assert not any("rounding-level charge" in w for w in data.get("warnings", []))


def test_unprojected_low_frequency_error_is_estimated(tmp_path, monkeypatch):
    data = solve_two_port(tmp_path, monkeypatch, "unprojected", projection="0")
    fw = data["fullwave"]
    excess = low_frequency_excess(fw)
    row = next(s for s in fw["sweep"] if round(s["f_Hz"]) == 10)
    for j, d in enumerate(excess):   # measured: d = 1.2e-3 and 5.0e-4 (the size depends on the rounding pattern)
        estimate = row["port_solutions"][j]["solver"]["tree_cotree"]["rhs_charge_field_fraction"]
        if abs(d) ** 0.5 > 1e-4:     # the field error is sqrt(d); the estimate must find it
            assert 0.5 < estimate / abs(d) ** 0.5 < 2, (j, estimate, d)
        else:                        # too small to compare: then the estimate must be small too
            assert estimate < 1e-3, (j, estimate, d)
        if estimate > 1e-3:
            assert any(f"port {j}: the right-hand side's rounding-level charge" in w for w in data.get("warnings", []))
