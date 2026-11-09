"""The exact dZ/dw (`dZ_domega_H`) with frequency-dependent terminations (docs/physics/multiport_fullwave.md).

dZ/dw = Z^T/w - (i/w) e^T A'(w) e needs every w-dependent term of A in A': without the Robin boundary groups (impedance,
scattering, sheet) and the PML stretch, dZ/dw is wrong by 100 % on such models. On the parallel-plate TEM line of
benchmarks/fw3d_robin_validation.py (cases of benchmarks/dzdw_check.py, seconds per run) the engine's dZ/dw must equal a
centred difference of Z, relative to max(|difference|, |Z|/w): the shorted line (control), the absorbing end, copper
impedance plates, an impedance wall of fixed Z_s, a resistive sheet, and a PML with and without a fixed wavelength. The
full five-point Richardson check is benchmarks/dzdw_check.py."""
import math
import sys
from pathlib import Path

import pytest

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dzdw_check as DZ  # noqa: E402  (the TEM line cases)

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")

STEP = 1e-4


@pytest.mark.parametrize("case", DZ.TEM_CASES)
def test_dzdw_against_a_centred_difference(case, tmp_path):
    doc = DZ.tem_case(case, tmp_path)
    f0 = DZ.TEM_F0
    doc["fullwave"]["frequencies"] = [f0 * (1 - STEP), f0, f0 * (1 + STEP)]
    rows = sorted(af.Design(doc).solve(tmp_path, verbose=0).data["fullwave"]["sweep"], key=lambda r: r["f_Hz"])
    assert all(r["solver"]["accepted"] for r in rows)
    z = [complex(r["Z_re_ohm"], r["Z_im_ohm"]) for r in rows]
    w0 = 2 * math.pi * f0
    difference = (z[2] - z[0]) / (2 * STEP * w0)
    engine = complex(rows[1]["dZ_domega_H"]["re"][0][0], rows[1]["dZ_domega_H"]["im"][0][0])
    scale = max(abs(difference), abs(z[1]) / w0)
    assert abs(engine - difference) / scale < 1e-5, (case, engine, difference)
    # the scope names only the w-dependent terms the design has
    scope = rows[1]["dZ_domega_scope"]
    if case == "tem_pec":
        assert "A' also holds" not in scope
    elif case.startswith("tem_pml"):
        assert "PML layers" in scope and "Robin" not in scope
    else:
        assert "Robin boundary groups" in scope and "PML" not in scope
