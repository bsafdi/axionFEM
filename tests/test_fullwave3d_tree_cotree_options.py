"""The gauged path's options: setup validation, and the charge projection never touching real source charge.

AXF_FW_TC_CHARGE_PROJECTION (default 2) zeroes the right-hand side's Gauss rows whose charge is rounding of their own
terms; a source that carries real charge keeps it. A toroidal_box source over a 120 degree sector ends inside the
domain, so its end faces carry charge O(1) of their terms: zeroing those rows would solve a different, tree-dependent
source. Here, on the two-port frame: bad values and use without the gauge are refused; with the sector source,
projections 0 and 2 give the same direct-source solve (its port voltages, to 1e-9) and Z, Psi agrees up to projection
0's own rounding error, and the source solve keeps its charged rows. About 3 min on one rank. Set AXF_SLOW=1.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import axionfem as af

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

DESIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"
SECTOR = {"kind": "toroidal_box", "B0_T": 1.0, "r_m": [0.35, 0.6], "z_m": [-0.2, 0.2], "phi_deg": [30, 150],
          "profile": "uniform"}


def ranks():
    return 4 if os.environ.get("AXF_MPI") else 1


def two_port(frequencies, source=None):
    doc = json.loads(DESIGN.read_text())
    doc["fullwave"]["frequencies"] = frequencies
    if source is not None:
        doc["fullwave"]["source"] = source
    return doc


def needs_strumpack():
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")


@pytest.mark.parametrize("env, message", [
    ({"AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "3"}, "must be 0, 1 or 2"),
    ({"AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_REFINE": "exact"}, "must be normwise or componentwise"),
    ({"AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "2"}, "apply to AXF_FW_GAUGE=tree-cotree only"),
    ({"AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_RESIDUAL": "quad"}, "must be extended or double"),
    ({"AXF_FW_GAUGE": "none", "AXF_FW_TC_RESIDUAL": "double"}, "AXF_FW_TC_RESIDUAL applies to AXF_FW_GAUGE=tree-cotree"),
    ({"AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_REFINE": "componentwise"}, "needs AXF_FW_TC_RESIDUAL=double"),
])
def test_bad_options_are_refused(tmp_path, monkeypatch, env, message):
    needs_strumpack()
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.delenv("AXF_FW_TC_CHARGE_PROJECTION", raising=False)
    monkeypatch.delenv("AXF_FW_TC_REFINE", raising=False)
    monkeypatch.delenv("AXF_FW_TC_RESIDUAL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(Exception, match=message):
        af.Design(two_port([1e6])).solve(tmp_path / "bad", verbose=0, mpi_ranks=ranks())


def test_refinement_residual_modes(tmp_path, monkeypatch):
    """The refinement residual: extended (default) evaluates it in long double from K, M and G_c and stops on the
    equilibrated correction; double is the plain double-precision residual. Both converge to the same field
    (Z to 1e-11), and the extended corrections reach rounding."""
    needs_strumpack()
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.setenv("AXF_FW_GAUGE", "tree-cotree")
    monkeypatch.delenv("AXF_FW_TC_CHARGE_PROJECTION", raising=False)
    monkeypatch.delenv("AXF_FW_TC_REFINE", raising=False)
    rows = {}
    for mode in ("extended", "double"):
        monkeypatch.setenv("AXF_FW_TC_RESIDUAL", mode)
        r = af.Design(two_port([1e5])).solve(tmp_path / mode, verbose=0, mpi_ranks=ranks())
        rows[mode] = r.data["fullwave"]["sweep"][0]
        assert f"{mode} residual" in r.data["fullwave"]["solver_path"]
    for port in rows["extended"]["port_solutions"]:
        tc = port["solver"]["tree_cotree"]
        assert tc["refinement_residual"] == "extended"
        assert tc["refinement_stop"] == "relative correction (equilibrated)"
        assert tc["refinement_residual_mantissa_bits"] > 53     # x86-64 long double
        assert tc["refinement_history_relative_correction"][-1] < 1e-14
    assert rows["double"]["port_solutions"][0]["solver"]["tree_cotree"]["refinement_residual"] == "double"
    za = np.array(rows["extended"]["Z_matrix_re_ohm"]) + 1j * np.array(rows["extended"]["Z_matrix_im_ohm"])
    zb = np.array(rows["double"]["Z_matrix_re_ohm"]) + 1j * np.array(rows["double"]["Z_matrix_im_ohm"])
    assert np.abs(za - zb).max() < 1e-11 * np.abs(za).max()


def test_projection_keeps_real_source_charge(tmp_path, monkeypatch):
    needs_strumpack()
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.setenv("AXF_FW_GAUGE", "tree-cotree")
    rows = {}
    for projection in ("0", "2"):
        monkeypatch.setenv("AXF_FW_TC_CHARGE_PROJECTION", projection)
        r = af.Design(two_port([1e3, 1e5], SECTOR)).solve(tmp_path / f"p{projection}", verbose=0, mpi_ranks=ranks())
        rows[projection] = r.data["fullwave"]["sweep"]
    def vector(row, key):
        return np.array(row[key]["re"]) + 1j * np.array(row[key]["im"])

    for a, b in zip(rows["0"], rows["2"]):
        # the direct source solve is where the projected source enters: its port voltages must not move (zeroing the
        # sector's charged rows would change them at O(1); measured: identical, the source has no rounding rows)
        va = vector(a, "direct_source_voltage")
        vb = vector(b, "direct_source_voltage")
        assert np.abs(va - vb).max() < 1e-9 * np.abs(va).max(), (a["f_Hz"], va, vb)
        assert b["source_reciprocity_relative_error"] < 1e-6, b["source_reciprocity_relative_error"]
        source = b["direct_source_solver"]["tree_cotree"]
        assert source["charged_free_vertex_rows"] > 0          # the sector's end faces (81 rows)
        assert source["rhs_free_vertex_charge_relative_solved"] == source["rhs_free_vertex_charge_relative"]   # kept
        za = np.array(a["Z_matrix_re_ohm"]) + 1j * np.array(a["Z_matrix_im_ohm"])
        zb = np.array(b["Z_matrix_re_ohm"]) + 1j * np.array(b["Z_matrix_im_ohm"])
        assert np.abs(za - zb).max() < 1e-9 * np.abs(za).max(), a["f_Hz"]
        # Psi overlaps the (unprojected) source with the port fields; with projection 0 those carry their rounding
        # charge's gradient field (3.6e-6 at 1 kHz), which a source with real charge sees: 1.7e-9 at 1 kHz, 1e-13 at
        # 100 kHz, falling like f^-2. That is projection 0's error; the bound only rules out an O(1) change.
        pa = vector(a, "Psi_vector")
        pb = vector(b, "Psi_vector")
        assert np.abs(pa - pb).max() < 1e-6 * np.abs(pa).max(), (a["f_Hz"], pa, pb)
        for port in b["port_solutions"]:                       # the feeds are charge-free: all their rows are rounding
            assert port["solver"]["tree_cotree"]["charged_free_vertex_rows"] == 0
