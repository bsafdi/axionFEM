"""3D engine on two MPI ranks against one (needs a launcher with >= 2 slots: AXF_MPI=1, e.g. scripts/slurm/tests.slurm)."""
import os

import pytest

import axionfem as af

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")


def _ring(r0, r1, z0, z1):
    return {"revolve": {"profile": {"rect": {"r": [r0, r1], "z": [z0, z1]}}}}


@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs >= 2 MPI slots; set AXF_MPI=1")
@pytest.mark.parametrize("order", [1, 2])  # order 2: face DOFs with orientation transformations
def test_two_ranks_match_one_rank(tmp_path, order):
    doc = {"schema": "axionfem.design/1", "name": "mpi2", "symmetry": "3d",
           "regions": [{"name": "coil", "solid": _ring(0.5, 0.6, -0.1, 0.1), "material": "winding"},
                       {"name": "ring", "solid": _ring(0.25, 0.3, -0.05, 0.05), "material": "superconductor"}],
           "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
           "superconductors": [{"body": "ring", "azimuthal_slit": True}],
           "pickup": {"type": "azimuthal", "conductor": "ring"},
           "mesh": {"size": 0.15 if order == 1 else 0.25, "surface_size": 0.06 if order == 1 else 0.09, "order": order}}
    d = af.Design(doc)
    r1 = d.solve(tmp_path / "np1", verbose=0)
    r2 = d.solve(tmp_path / "np2", verbose=0, mpi_ranks=2)
    assert r2.data["mesh"]["mpi_ranks"] == 2
    for k in ("L_p_H", "W_mag_J"):
        assert r2.coupling[k] == pytest.approx(r1.coupling[k], rel=1e-8), k
    # Psi can be a small difference in weakly coupled designs: compare on its Cauchy-Schwarz scale sqrt(2 L_p W)
    psi_scale = (r1.coupling["L_p_H"] * 2 * r1.coupling["W_mag_J"]) ** 0.5
    assert abs(r2.coupling["Psi_SI_H_T_m2"] - r1.coupling["Psi_SI_H_T_m2"]) < 1e-8 * psi_scale
    pts = ([0.1, 0.4, 0.7], [0.0, 0.05, -0.1], [0.0, 0.2, -0.3])
    s1, s2 = r1.sample(*pts), r2.sample(*pts)
    scale = max(abs(x) for x in s1["Bz"])
    for key in ("Bx", "By", "Bz", "pickup_Bz"):
        for a, b in zip(s1[key], s2[key]):
            assert abs(a - b) < 1e-7 * scale or abs(a - b) <= 1e-7 * abs(a)
    assert len(r2.scene()["regions"]) == len(r1.scene()["regions"])
    # gathered outputs (superconductor surface peaks, forces) must not depend on the rank count either
    for k, v in r1.data["fields"].get("superconductor_surface_T", {}).items():
        assert r2.data["fields"]["superconductor_surface_T"][k] == pytest.approx(v, rel=1e-6), k
    # pickup surface losses: Gamma only (the builtin superconductor has no rf block, so no Q and no T/Q)
    l1, l2 = r1.data["pickup"]["losses"], r2.data["pickup"]["losses"]
    assert set(l1["surfaces"]) == set(l2["surfaces"]) == {"ring"}
    assert "T_over_Q_total_K" not in l1 and "Q" not in l1["surfaces"]["ring"]
    for k, v in l1["surfaces"].items():
        assert l2["surfaces"][k]["Gamma"] == pytest.approx(v["Gamma"], rel=1e-6), k
        for reg, g in v["Gamma_by_field_region"].items():
            assert l2["surfaces"][k]["Gamma_by_field_region"][reg] == pytest.approx(g, rel=1e-6), (k, reg)
