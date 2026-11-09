"""Gauss-law guard on the ungauged full-wave paths (docs/physics/fullwave_gauss_guard.md; FwGaussGuard).

On the two-port frame at 100 Hz without validation loads, AMS accepts a field whose gradient part is 3-4 % wrong.
The guard's estimate must match that error, and its correction must bring the field
outputs to the gauged path's (the reference, exact there with its charge projection). Out of scope it must say so.
Benchmark: benchmarks/gauss_law_guard.py."""
import json
import math
import os
from pathlib import Path

import pytest

import axionfem as af

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

TWO_PORT = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"
GAUGED = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2",
          "AXF_FW_GAUSS_GUARD": "off"}
# the gauged reference path factorizes with STRUMPACK
needs_strumpack = pytest.mark.skipif(not af.build_info().get("strumpack"),
                                     reason="the gauged path (AXF_FW_SOLVER=strumpack) needs a STRUMPACK build")


def _ams(guard):
    return {"AXF_FW_SOLVER": "ams", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
            "AXF_FW_GAUSS_GUARD": guard}


def _solve(path, env, monkeypatch, f=100.0, loads=False):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    doc = json.loads(TWO_PORT.read_text())
    if not loads:   # the design's validation shunts and source check
        for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
            doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [f]
    return af.Design(doc).solve(path, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)


def _electric_energy(row):
    ue = row["energy_electric_J_per_A2"]["re"]
    return [ue[i][i] for i in range(len(ue))]


def _impedance(row):
    return [complex(a, b) for ra, ia in zip(row["Z_matrix_re_ohm"], row["Z_matrix_im_ohm"]) for a, b in zip(ra, ia)]


@needs_strumpack
def test_guard_estimates_and_removes_the_gradient_error(tmp_path, monkeypatch):
    ref = _solve(tmp_path / "gauged", GAUGED, monkeypatch).fullwave["sweep"][0]
    est = _solve(tmp_path / "estimate", _ams("estimate"), monkeypatch)
    cor = _solve(tmp_path / "correct", _ams("correct"), monkeypatch)
    ue_ref = _electric_energy(ref)
    # estimate only: the fields are as solved, and the estimate is their gradient error. The excess electric energy
    # of a gradient error M-orthogonal to the field is its squared norm, so sqrt(U_e / U_e_ref - 1) measures it.
    row = est.fullwave["sweep"][0]
    setup = est.fullwave["gauss_law_guard"]["setup"]
    assert setup["pec_sets"] == 2 and len(setup["floating_set_nodes"]) == 1   # the wall (grounded) and the frame
    assert setup["essential_edge_leak_relative"] == 0.0
    assert row["gauss_law_guard"]["operator_consistency_relative"] < 1e-12
    for port, ue in zip(row["port_solutions"], _electric_energy(row)):
        guard = port["solver"]["gauss_law_guard"]
        assert guard["applied"] is False
        measured = math.sqrt(ue / ue_ref[port["port_index"]] - 1)
        assert 1e-2 < guard["field_fraction"] < 1e-1   # 3-4 % at 100 Hz
        assert guard["field_fraction"] == pytest.approx(measured, rel=0.05)
    assert any("NOT corrected" in w for w in est.data["warnings"])
    # corrected: the field outputs are the gauged path's; Z (a gap port on one conductor) was right before and stays
    # right. The magnetic energy comes from the curl representative.
    row = cor.fullwave["sweep"][0]
    for port in row["port_solutions"]:
        guard = port["solver"]["gauss_law_guard"]
        assert guard["applied"] is True
        assert guard["pcg_relative_residual_max"] < 1e-10
        assert guard["gauss_residual_reduction"] < 1e-8   # what the S-solve achieved on the Gauss residual
        assert guard["gauss_residual_after"] < 1e-8 * guard["gauss_residual_before"]
    for mine, theirs in zip(_electric_energy(row), ue_ref):
        assert mine == pytest.approx(theirs, rel=1e-8)
    um = row["energy_magnetic_J_per_A2"]["re"]
    um_ref = ref["energy_magnetic_J_per_A2"]["re"]
    for k in range(len(um)):
        assert um[k][k] == pytest.approx(um_ref[k][k], rel=1e-8)
    assert "Gauss-law guard" in row["energy_magnetic_scope"]
    for mine, theirs in zip(_impedance(row), _impedance(ref)):
        assert abs(mine - theirs) < 1e-9 * abs(theirs)
    assert any("Gauss-law guard removed a gradient-field error" in w for w in cor.data["warnings"])


def test_guard_with_validation_loads(tmp_path, monkeypatch):
    """The validation shunts enter the correction by Woodbury, their Gauss rows projected like the right-hand side's:
    uncorrected, AMS's loaded field and its port-basis reconstruction disagree at 100 Hz (the gate refuses); corrected,
    they agree to the loaded voltages' own closure (unprojected load rows left 1.5e-8)."""
    with pytest.raises(Exception, match="direct loaded FEM field and port-basis reconstruction disagree"):
        _solve(tmp_path / "estimate", _ams("estimate"), monkeypatch, loads=True)
    row = _solve(tmp_path / "correct", _ams("correct"), monkeypatch, loads=True).fullwave["sweep"][0]
    assert row["loaded_field_closure_relative_error"] < 1e-9
    loaded = row["direct_loaded_solver"]["gauss_law_guard"]
    assert loaded["applied"] is True and loaded["loads"] == 2
    assert loaded["gauss_residual_reduction"] < 1e-8
    assert row["direct_source_solver"]["gauss_law_guard"]["applied"] is True


LOSSY_DIELECTRIC = {"kind": "linear", "eps_r": 2.0, "tan_delta": 1e-2}


def _lossy_doc(f, boundary="pec", material=LOSSY_DIELECTRIC, loads=False):
    """The two-port frame with a lossy block beside it (complex mass); by default a dielectric and no validation
    loads."""
    doc = json.loads(TWO_PORT.read_text())
    if not loads:
        for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
            doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [f]
    doc["fullwave"]["boundary"] = boundary
    doc["materials"] = {"lossy": material}
    doc["regions"].append({"name": "slab", "material": "lossy",
                           "solid": {"box": {"min": [0.45, -0.05, -0.05], "max": [0.55, 0.05, 0.05]}}})
    return doc


def test_guard_corrects_a_lossy_model(tmp_path, monkeypatch):
    """A lossy dielectric makes the mass complex. The guard solves the complex Gauss law (GMRES, preconditioned by the
    real S on M + Mi), and the corrected field follows the quasi-static laws: U_e / f^2 and R_bulk / f^3 at 100 Hz
    equal their 100 kHz values (where AMS's gradient error is ~1e-7) up to the (k R)^2 dispersion. Uncorrected, they do
    not."""
    ranks = 4 if os.environ.get("AXF_MPI") else 1
    runs = {}
    for name, guard, f in (("estimate", "estimate", 100.0), ("correct", "correct", 100.0), ("top", "correct", 1e5)):
        for key, value in _ams(guard).items():
            monkeypatch.setenv(key, value)
        runs[name] = af.Design(_lossy_doc(f)).solve(tmp_path / name, verbose=0, mpi_ranks=ranks).fullwave["sweep"][0]

    def law(row, f, top):
        ue = _electric_energy(row)
        ue_top = _electric_energy(runs["top"])
        return ([u / f ** 2 / (t / top ** 2) - 1 for u, t in zip(ue, ue_top)],
                row["R_bulk_ohm"] / f ** 3 / (runs["top"]["R_bulk_ohm"] / top ** 3) - 1)

    guard = runs["correct"]["port_solutions"][0]["solver"]["gauss_law_guard"]
    assert guard["applied"] is True and guard["lossy"] is True
    assert guard["pcg_relative_residual_max"] < 1e-10
    assert guard["gauss_residual_reduction"] < 1e-8
    assert 0 < guard["gmres_iterations"] < 200
    assert runs["correct"]["gauss_law_guard"]["loss_mass_consistency_relative"] < 1e-11
    ue_corrected, r_corrected = law(runs["correct"], 100.0, 1e5)
    ue_estimate, r_estimate = law(runs["estimate"], 100.0, 1e5)
    assert max(abs(x) for x in ue_corrected) < 1e-5 and abs(r_corrected) < 1e-5
    assert max(abs(x) for x in ue_estimate) > 1e-4   # the uncorrected gradient error (3-4 % of the field)


def test_guard_corrects_a_conducting_model(tmp_path, monkeypatch):
    """A resistive block (sigma 100 S/m; sigma / (omega eps0) = 1.8e10 at 100 Hz) beside the frame, with the design's
    validation shunts and source check: uncorrected, the loaded-field gate refuses (closure 5.5e-3); corrected, the
    closure is at the loaded voltages' own level, and the guard reports its floor near the conductor."""
    ranks = 4 if os.environ.get("AXF_MPI") else 1
    resistive = {"kind": "linear", "eps_r": 1.0, "sigma_S_m": 100.0}
    for key, value in _ams("estimate").items():
        monkeypatch.setenv(key, value)
    with pytest.raises(Exception, match="direct loaded FEM field and port-basis reconstruction disagree"):
        af.Design(_lossy_doc(100.0, material=resistive, loads=True)).solve(tmp_path / "estimate", verbose=0,
                                                                           mpi_ranks=ranks)
    for key, value in _ams("correct").items():
        monkeypatch.setenv(key, value)
    r = af.Design(_lossy_doc(100.0, material=resistive, loads=True)).solve(tmp_path / "correct", verbose=0,
                                                                          mpi_ranks=ranks)
    row = r.fullwave["sweep"][0]
    assert row["loaded_field_closure_relative_error"] < 1e-9
    frequency = row["gauss_law_guard"]
    assert frequency["lossy"] is True
    assert frequency["loss_contrast"] > 1e9
    assert frequency["loss_mass_consistency_relative"] < 1e-11
    loaded = row["direct_loaded_solver"]["gauss_law_guard"]
    assert loaded["applied"] is True and loaded["loads"] == 2
    assert loaded["pcg_relative_residual_max"] < 1e-10
    assert 0 < loaded["gmres_iterations"] < 200
    # the Gauss residual reduction is not gated here: under this contrast it measures the conductor's rows
    assert 0 <= loaded["floor_estimate"] <= 1e-8


def test_guard_not_applied_to_an_open_model_says_so(tmp_path, monkeypatch):
    """A radiating exterior is outside the guard's scope: the run records why and warns that its quasi-static field
    outputs are unchecked, instead of staying silent."""
    for key, value in _ams("correct").items():
        monkeypatch.setenv(key, value)
    doc = _lossy_doc(1e5, boundary="radiation")   # AMS converges there; k R ~ 1e-3, still quasi-static
    # a uniform source has no exterior expansion; a closed square filament loop away from the frame does
    corners = [(-0.1, 0.4, -0.1), (0.1, 0.4, -0.1), (0.1, 0.4, 0.1), (-0.1, 0.4, 0.1)]
    segments = [[*corners[k], *corners[(k + 1) % 4], 1.0] for k in range(4)]
    doc["fullwave"]["source"] = {"kind": "segments", "segments_m_A": segments}
    r = af.Design(doc).solve(tmp_path / "open", verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    assert "open boundary" in r.fullwave["gauss_law_guard"]["not_applied"]
    assert "gauss_law_guard" not in r.fullwave["sweep"][0]["solver"]
    assert any("Gauss-law guard was not applied (open boundary" in w for w in r.data["warnings"])


def test_guard_options_are_validated(tmp_path, monkeypatch):
    with pytest.raises(Exception, match="AXF_FW_GAUSS_GUARD must be correct, estimate or off"):
        _solve(tmp_path / "bad", _ams("sometimes"), monkeypatch)


@needs_strumpack
def test_guard_is_refused_on_the_gauged_path(tmp_path, monkeypatch):
    for mode in ("correct", "estimate"):
        with pytest.raises(Exception, match="AXF_FW_GAUSS_GUARD applies to the ungauged paths"):
            _solve(tmp_path / f"gauged_{mode}", {**GAUGED, "AXF_FW_GAUSS_GUARD": mode}, monkeypatch)
