"""The two ungauged 3D full-wave solver paths on a closed model with an exact Z (seconds; quick pre-merge subset).

AMS-GMRES (the default) and plain STRUMPACK (AXF_FW_SOLVER=strumpack, no gauge) solve the shorted TEM line of
benchmarks/fullwave_indicator.py (PEC plates, PMC sides, a surface port at one end; Z = -i Z0 tan(k l)) at order 2, from
1 kHz to 100 MHz. Each must report its path and pass its own gate; the Gauss-law guard (on by default on both paths,
docs/physics/fullwave_gauss_guard.md) must remove the gradient error both leave at 1 kHz (2-4 % of the field); Z and the
exact per-frequency dZ/dw must match their closed forms to the discretization error (1e-7 and 4e-7 at 100 MHz), dZ/dw
the energy identity, and the two paths must agree to the solvers' tolerance. Otherwise plain STRUMPACK is exercised only
by the distributed-port cases (20 min each, AXF_SLOW). tests/test_dzdw_algebra.py checks the dZ/dw identity on a dense
model, and tests/test_fullwave3d_dzdw_terminations.py checks the engine's dZ/dw against a centred difference of Z with
Robin and PML terminations; here it meets the closed form on both ungauged paths. The gauged path refuses the PMC sides;
it has its own tests (tree-cotree options, feed continuity, curved port charge, the guard's reference)."""
import math
import sys
from pathlib import Path

import pytest

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import fullwave_indicator as FWI  # noqa: E402  (the TEM line's design and exact Z)
import fw3d_robin_validation as robin  # noqa: E402  (C0, ETA0)

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")

FREQUENCIES = [1e3, 1e6, 1e8]


def _exact_dzdw(f, geometry):
    """d/dw of Z = -i Z0 tan(w l / c): -i Z0 (l / c) / cos^2(k l)."""
    k = 2 * math.pi * f / robin.C0
    z0 = robin.ETA0 * geometry["h"] / geometry["W"]
    return -1j * z0 * (geometry["l"] / robin.C0) / math.cos(k * geometry["l"]) ** 2


def _solve(tmp_path, monkeypatch, solver):
    monkeypatch.setenv("AXF_FW_SOLVER", solver)
    for key in ("AXF_FW_GAUGE", "AXF_FW_GAUSS_GUARD", "AXF_FW_TC_CHARGE_PROJECTION"):
        monkeypatch.delenv(key, raising=False)
    doc, geometry = FWI.tem_design(tmp_path / solver, "pec", 2, 0, FREQUENCIES)
    return af.Design(doc).solve(tmp_path / solver, verbose=0).data["fullwave"], geometry


def test_ams_and_plain_strumpack_match_the_exact_tem_line(tmp_path, monkeypatch):
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    z = {}
    for solver, path in (("ams", "AMS-GMRES"), ("strumpack", "STRUMPACK")):
        fw, geometry = _solve(tmp_path, monkeypatch, solver)
        assert fw["solver_path"].startswith(path), fw["solver_path"]
        assert fw["gauss_law_guard"]["mode"] == "correct"
        z[solver] = []
        for row in fw["sweep"]:
            port = row["port_solutions"][0]["solver"]
            assert port["accepted"] is True
            guard = port["gauss_law_guard"]
            assert guard["applied"] is True
            assert guard["gauss_residual_after"] < 1e-14   # measured 3e-17 to 5e-16
            if row["f_Hz"] < 1e4:   # the low-frequency gradient error, removed
                assert 1e-3 < guard["field_fraction"] < 1e-1
                assert guard["gauss_residual_after"] < 1e-10 * guard["gauss_residual_before"]
            zh = complex(row["Z_re_ohm"], row["Z_im_ohm"])
            exact = FWI.tem_exact_z("pec", row["f_Hz"], geometry)
            assert abs(zh - exact) < 1e-6 * abs(exact), (solver, row["f_Hz"], zh, exact)
            dz = complex(row["dZ_domega_H"]["re"][0][0], row["dZ_domega_H"]["im"][0][0])
            dz_exact = _exact_dzdw(row["f_Hz"], geometry)
            assert abs(dz - dz_exact) < 3e-6 * abs(dz_exact), (solver, row["f_Hz"], dz, dz_exact)
            assert row["dZ_domega_vs_energy_identity_relative"] < 1e-9   # measured <= 7e-12
            z[solver].append(zh)
    for a, s in zip(z["ams"], z["strumpack"]):
        assert abs(a - s) < 1e-8 * abs(s), (a, s)
