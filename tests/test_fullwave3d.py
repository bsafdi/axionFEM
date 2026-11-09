"""3D full-wave (complex Nedelec E field, port feed, exact exterior) against thin-wire loop theory, the
axisymmetric solver and exact cavity modes, on coarse meshes (tight comparisons: benchmarks/fullwave3d_validation.py)."""
import math
import os
import sys
from pathlib import Path

import pytest

import axionfem as af
from axionfem.convert import to_3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from loop_theory import loop_L_R  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

C = 299792458.0


def _loop_doc(fw, a=0.3, h=0.02):
    """Superconducting ring (radius a, square cross-section h) read out across a slit; a small coil makes B_lab."""
    return {"schema": "axionfem.design/1", "name": "loop",
            "regions": [{"name": "ring", "shape": {"rect": {"r": [a - h / 2, a + h / 2], "z": [-h / 2, h / 2]}},
                         "material": "superconductor"},
                        {"name": "coil", "shape": {"rect": {"r": [0.1, 0.12], "z": [-0.05, 0.05]}}, "material": "winding"}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
            "superconductors": [{"body": "ring", "azimuthal_slit": True}],
            "pickup": {"type": "azimuthal", "conductor": "ring"},
            "fullwave": fw,
            "mesh": {"order": 2, "size": 0.02, "surface_size": 0.006}}


def test_small_loop_3d_matches_axisymmetric(tmp_path):
    a = 0.3
    kas = [0.02, 0.15]
    fw = {"frequencies": [ka * C / (2 * math.pi * a) for ka in kas]}
    d2 = af.Design(_loop_doc(fw))
    r2 = d2.solve(tmp_path / "ax", verbose=0)
    d3 = to_3d(d2, pickup="slit", slit_deg=6.0, mesh={"size": 0.12, "surface_size": 0.02, "order": 2})
    r3 = d3.solve(tmp_path / "3d", verbose=0)
    s2, s3 = r2.data["fullwave"]["sweep"], r3.data["fullwave"]["sweep"]
    mqs = r3.coupling
    print("3D:", [(x["L_H"], x["R_rad_ohm"], x["Psi_abs"], x["solver"]) for x in s3])
    print("2D:", [(x["L_H"], x["R_rad_ohm"], x["Psi_abs"]) for x in s2])
    print("3D MQS:", mqs["L_p_H"], mqs["Psi_SI_H_T_m2"])
    # low frequency: the 3D full-wave readout reduces to the 3D MQS port solution (Psi is ~0 by symmetry here:
    # poloidal static field, azimuthal readout; it is tested on the toroid in benchmarks/fullwave3d_validation.py).
    # At ka = 0.02 the gap-fed loop is not yet at its static limit: L/L_MQS - 1 = 1.8e-3, recorded in
    # docs/validation/fullwave3d.md (small loop table: 3D coarse and fine 1.0018, thin-wire theory 1.0018). It is
    # the reactance of the n >= 1 currents the slit drives (physical, not discretization: theory 1.0017274 for this
    # 6-degree slit vs 3D 1.0017594). A 1e-3 tolerance would be tighter than this offset; the bound is the recorded
    # value, and the ratio is checked against the theory itself below.
    assert s3[0]["L_H"] == pytest.approx(mqs["L_p_H"], rel=2e-3)
    # gap-fed loop vs thin-wire theory (Storer/Wu, all current modes): the slit also drives n >= 1 currents, so
    # L(omega) rises towards the first antiresonance and R exceeds the uniform-feed (n = 0) value of the 2D solver
    h = 0.02
    L_lo = loop_L_R(s3[0]["f_Hz"], a, 0.59 * h, gap_rad=math.radians(6.0))[0]
    Lt, Rt = loop_L_R(s3[1]["f_Hz"], a, 0.59 * h, gap_rad=math.radians(6.0))
    L0t = loop_L_R(1e-4 * C / (2 * math.pi * a), a, 0.59 * h, gap_rad=math.radians(6.0))[0]
    assert s3[0]["L_H"] / mqs["L_p_H"] - 1 == pytest.approx(L_lo / L0t - 1, rel=0.1)  # 1.76e-3 vs 1.73e-3
    assert s3[1]["R_rad_ohm"] == pytest.approx(Rt, rel=0.15)
    assert s3[1]["L_H"] / mqs["L_p_H"] == pytest.approx(Lt / L0t, rel=0.03)
    assert s3[1]["R_rad_ohm"] > s2[1]["R_rad_ohm"]
    assert s3[1]["solver"]["gmres_iterations"] < 1000
    # the RF field at the first sweep frequency is in the scene slices
    sl = r3.scene()["slices"][0]
    assert "rf_Emag" in sl["quantities"] and sl["rf_frequency_Hz"] == pytest.approx(s3[0]["f_Hz"])
    assert sl["ranges"]["rf_Emag"][1] > 0


@pytest.mark.parametrize("R", [1.0, 2.0])
def test_sphere_cavity_modes_3d(tmp_path, R):
    doc = {"schema": "axionfem.design/1", "name": "sph", "symmetry": "3d",
           "regions": [{"name": "coil", "solid": {"revolve": {"profile": {"rect": {"r": [0.3, 0.32], "z": [-0.01, 0.01]}}}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "fullwave": {"modes": 8},
           "mesh": {"order": 2, "size": 0.15, "surface_size": 0.05, "domain_radius": R}}
    r = af.Design(doc).solve(tmp_path / "s", verbose=0)
    xs = sorted(m["f_Hz"] * 2 * math.pi * R / C for m in r.data["fullwave"]["eigenmodes"]["modes"])
    print(xs)
    exact = [2.743707] * 3 + [3.870239] * 5   # TM_1 (3-fold), TM_2 (5-fold)
    diag = r.data["fullwave"]["eigenmodes"]
    assert diag["converged"] and diag["requested"] == diag["returned"] == len(exact)
    assert diag["max_relative_residual"] < diag["residual_tolerance"]
    assert diag["max_mass_overlap"] < 1e-4
    assert len(xs) == len(exact)
    for g, x in zip(xs, exact):
        assert g == pytest.approx(x, rel=2e-2)


def test_shell_cap_and_independent_magnetic_dipole(tmp_path, monkeypatch):
    """The old endpoint error changed R by 8.37%; energy identities missed it."""
    import hashlib
    d = to_3d(af.Design(_loop_doc({'frequencies':[.02*C/(2*math.pi*.3)], 'dtn_modes':8})),
              pickup='slit', slit_deg=4.,
              mesh={'size':.18,'surface_size':.028,'order':2,'domain_radius':.9})
    rows=[]; meshes=[]
    for cap in [3,5]:
        monkeypatch.setenv('AXF_FW_SHELL_FMAX',str(cap))
        wd=tmp_path/f'cap{cap}'
        r=d.solve(wd,verbose=0,mpi_ranks=4 if os.environ.get('AXF_MPI') else 1)
        row=r.fullwave['sweep'][0]
        assert row['solver']['relative_residual'] < 1e-6
        assert row['R_rad_ohm'] == pytest.approx(row['R_static_dipole_ohm'],rel=.01)
        rows.append(row)
        meshes.append(hashlib.sha256((wd/'mesh3d.msh').read_bytes()).hexdigest())
    assert meshes[0] == meshes[1]
    assert rows[0]['R_rad_ohm'] == pytest.approx(rows[1]['R_rad_ohm'],rel=1e-3)


@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="requires 16 MPI slots and AXF_MPI=1")
def test_diagnostic_reductions_include_ranks_without_local_support(tmp_path, monkeypatch):
    # This partition includes ranks with no R1 faces. An Allreduce made conditional on
    # local trace points would mismatch the next collective and abort MPI.
    monkeypatch.setenv('AXF_MPI_TIMEOUT','900')
    d=af.load(Path(__file__).resolve().parents[1]/'benchmarks/designs/solenoid_exterior.json')
    d.doc['mesh']['order']=1
    d.doc['fullwave']['frequencies']=d.doc['fullwave']['frequencies'][:1]
    d.doc.setdefault('materials',{})['winding']={'kind':'linear','eps_r':2.,'tan_delta':.001}
    r=d.solve(tmp_path/'interior_ranks',verbose=0,mpi_ranks=16)
    assert 0 < r.fullwave['dipole_trace_ranks'] < 16
    assert 0 < r.fullwave['bulk_loss_ranks'] < 16
    assert r.fullwave['sweep'][0]['R_bulk_ohm'] > 0
    assert r.fullwave['sweep'][0]['solver']['relative_residual'] < 1e-6
    moment_R=r.fullwave['sweep'][0]['R_static_dipole_ohm']
    assert math.isfinite(moment_R) and moment_R >= 0  # coax moment vanishes in the continuum
