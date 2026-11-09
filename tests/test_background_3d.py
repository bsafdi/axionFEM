"""The 3D MQS Psi with a uniform background, on a coarse model (default suite; the full-wave tests, including the
uniform background with the radiating boundary, are in tests/test_fullwave3d_background.py).

The model is benchmarks/fullwave3d_background.py's coax-in-a-solenoid with the coil removed (a mu_r = 1 region) in a
uniform 1 T background. The 3D MQS Psi must be the reciprocal form int (A_lab + A_bg) . B_pick: the primary form
int A_pick . B_lab cuts the background at the domain sphere and depends on the gauge, hence the mesh (it is kept as
checks.Psi_primary_form). The reciprocal form agrees with the axisymmetric engine's MQS Psi.
"""
import os
import sys
from pathlib import Path

import pytest

import axionfem as af

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fullwave3d_background as fb  # noqa: E402

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")

MESH = {"order": 2, "size": 0.15, "surface_size": 0.05}
UNIFORM = {"type": "uniform", "Bz": 1.0}


def test_uniform_background_mqs_psi_is_the_reciprocal_form(tmp_path):
    doc = fb.design(False, UNIFORM, fb.R1, (0.02,), mesh=MESH)
    doc.pop("fullwave")
    r = af.Design(doc).solve(tmp_path / "3d", verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    checks = r.data["checks"]
    psi = r.coupling["Psi_SI_H_T_m2"]
    primary = checks["Psi_primary_form"]
    print("3D MQS Psi", psi, "primary form / reported", primary["over_reported"], "warnings", r.data["warnings"])
    assert checks["Psi_reported_form"].startswith("reciprocal")
    assert psi == pytest.approx(abs(checks["Psi_A_lab_dot_B_pick"]), rel=1e-14)
    assert psi > 0   # oriented by the reported form
    assert primary["value_SI_H_T_m2"] == pytest.approx(primary["over_reported"] * psi, rel=1e-12)
    assert r.coupling["F_SI_T4_H2_m8"] == pytest.approx(psi ** 4 / r.coupling["L_p_H"] ** 2, rel=1e-12)
    if abs(primary["over_reported"] - 1) > 1e-3:
        assert any("primary form" in w for w in r.data["warnings"])
    # the axisymmetric engine reports the reciprocal form (its MQS Psi is int A_lab mu H_pick)
    d2 = fb.design(False, UNIFORM, fb.R1, (0.02,), axisymmetric=True)
    d2.pop("fullwave")
    axi = af.Design(d2).solve(tmp_path / "2d", verbose=0)
    print("axisymmetric MQS Psi", axi.coupling["Psi_SI_H_T_m2"])
    assert psi == pytest.approx(axi.coupling["Psi_SI_H_T_m2"], rel=1e-4)
