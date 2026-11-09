"""A port feed on curved elements must be discretely charge-free, like one on straight elements.

The port current comes from a conduction solve in the port region, normalized to the terminal current and integrated
into the full-wave right-hand side. If those three integrals use different quadratures, the feed is charge-free only on
straight elements, where every integrand is constant: on the curved (mesh order 2) cylinder port of the tree-cotree
capacitor (benchmarks/tree_cotree_capacitor.py) the free-vertex charge would be 2-3e-5 instead of rounding, and C would
carry a 2e-5 offset. One rule for all three (port_feed_rule) avoids it. Here, on the coarsest curved capacitor:
rounding-level charge in every port solve, and C(10 Hz) = C(1 kHz). About 2 min on one rank. Set AXF_SLOW=1.
"""
import importlib.util
import os
from pathlib import Path

import pytest

import axionfem as af

from test_fullwave3d_feed_continuity import rhs_charges

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "tree_cotree_capacitor.py"


def load_capacitor():
    spec = importlib.util.spec_from_file_location("tree_cotree_capacitor", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curved_port_feed_is_charge_free(tmp_path, monkeypatch):
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.setenv("AXF_FW_GAUGE", "tree-cotree")
    monkeypatch.setenv("AXF_FW_TC_CHARGE_PROJECTION", "0")   # the charge itself, not its projection
    cap = load_capacitor()
    doc = cap.design(0.02, 2, [1e1, 1e3])
    r = af.Design(doc).solve(tmp_path / "cap", order=1, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    fw = r.data["fullwave"]
    for row in fw["sweep"]:
        for name, q in rhs_charges(row):
            assert q < 1e-13, (row["f_Hz"], name, q)   # with mixed quadratures: 2-3e-5 for the curved port
        for port in row["port_solutions"]:
            tc = port["solver"]["tree_cotree"]
            assert tc["charged_free_vertex_rows"] == 0, (port["port_index"], tc["charged_free_vertex_rows"])
        # the capacitor port's terminals: the sphere and the shell, one of which may be the grounded root
        assert row["port_solutions"][2]["solver"]["tree_cotree"]["charged_conductor_rows"] >= 1
    lo, hi = fw["sweep"][0], fw["sweep"][1]
    C = [1 / (2 * 3.141592653589793 * row["f_Hz"] * row["Z_matrix_im_ohm"][2][2]) for row in (lo, hi)]
    assert abs(C[0] / C[1] - 1) < 1e-9, C
    assert 1.0 < C[0] / cap.exact_C() < 1.1, C[0] / cap.exact_C()   # coarse P1: above the closed form
