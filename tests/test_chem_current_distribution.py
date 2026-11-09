"""Current distribution (axionfem.chem): Python API, mesh families against the case meshes' hashes, and the two short
1D current-distribution cases judged by benchmarks/fluids/acceptance.py (the full set runs with benchmarks/chem/run.py)."""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/chem/cases"
REG = ROOT / "benchmarks/chem/registry/cases"

chem = pytest.importorskip("axionfem.chem")
try:
    from axionfem._core import chem as _corechem  # noqa: F401
except Exception:  # pragma: no cover - a build without the chemistry bindings
    pytest.skip("axionfem._core has no chem submodule (rebuild)", allow_module_level=True)

from axionfem.chem import CurrentDistribution, Reaction, meshes as M  # noqa: E402


def _checks():
    sys.path.insert(0, str(CASES))
    try:
        return importlib.import_module("c3_checks")
    finally:
        sys.path.remove(str(CASES))


def _case(name):
    return json.loads((REG / f"{name}.json").read_text())


@pytest.mark.parametrize("case,key,builder", [
    ("chem_mms_c3_bv_2d", "p1_struct", lambda N: M.rect(N, x1=0.01, y1=0.01)),
    ("chem_mms_c3_bv_2d", "p1_dist", lambda N: M.distorted_rect(int(round(math.log2(N / 4))), x1=0.01, y1=0.01)),
    ("chem_mms_c3_annulus_2d", "p1", lambda N: M.annulus_sector(N, 2 * N, 0.005, 0.01, 0.0, 0.5 * math.pi)),
    ("chem_analytic_c3_disk_primary", "p1", lambda n: M.disk_graded(n, r0=0.001, Rb=0.005, beta=3.0)),
    ("chem_analytic_c3_disk_primary", "uniform_p1", lambda n: M.disk_graded(n, r0=0.001, Rb=0.005, beta=1.0)),
])
def test_mesh_families_match_the_case_hashes(case, key, builder):
    C = _checks()
    ms = _case(case)["run"]["parameters"]["meshes"][key]
    sizes = ms.get("N") or ms.get("n")
    for k in range(2):  # the two coarsest levels (seconds)
        m = builder(sizes[k])
        spec = ms["levels"][k]
        V = m.vertices()
        assert C.mesh_problems(spec, V, len(m.T)) == []
        assert C.connectivity_hash(V, m.T, spec["hash_scale"]) == spec["connectivity_hash"]


def test_bv_cell_potentiostatic_galvanostatic_consistent():
    L, H = 0.01, 0.005
    ra, rc = Reaction("bv", 20, 0.05, 0.3, 0.7), Reaction("bv", 5, -0.02, 0.6, 0.4)
    cd = CurrentDistribution(M.rect(8, 2, x1=L, y1=H), order=1)
    cd.set_conductivity({1: 1.0})
    cd.add_electrode("anode", [1], reactions=[ra], phi_s=0.6)
    cd.add_electrode("cathode", [2], reactions=[rc], phi_s=0.0)
    rep = cd.solve()
    assert rep["converged"] and rep["stop"] in ("stagnation", "zero")
    q = cd.boundary_quadrature([1], cd.boundary_degree)
    f = chem.FARADAY / (chem.GAS_CONSTANT * 298.15)
    eta = 0.6 - q["phi"] - 0.05
    I = float(np.sum(20 * (np.exp(0.3 * f * eta) - np.exp(-0.7 * f * eta)) * q["w"]))
    cg = CurrentDistribution(M.rect(8, 2, x1=L, y1=H), order=1)
    cg.set_conductivity({1: 1.0})
    cg.add_electrode("anode", [1], reactions=[ra], galvanostatic=True, current=I, phi_s=0.5)
    cg.add_electrode("cathode", [2], reactions=[rc], phi_s=0.0)
    assert cg.solve()["converged"]
    assert abs(cg.phi_s()[0] - 0.6) < 1e-10


def test_errors_say_what_is_wrong():
    with pytest.raises(ValueError, match="kind"):
        Reaction("butler", 1.0).as_tuple()
    with pytest.raises(ValueError, match="slope"):
        Reaction("tafel_a", 1.0).as_tuple()
    cd = CurrentDistribution(M.rect(2, x1=1.0, y1=1.0), order=1)
    with pytest.raises(Exception, match="no boundary element"):
        cd.add_electrode("x", [9], reactions=[Reaction("linear", 1.0)])


@pytest.mark.parametrize("name", ["chem_analytic_c3_cell_1d", "chem_analytic_c3_galvanic_1d"])
def test_short_cases_meet_their_criteria(name):
    sys.path.insert(0, str(CASES))
    sys.path.insert(0, str(ROOT / "benchmarks/fluids"))
    try:
        R = importlib.import_module("c3_current_distribution")
        H = importlib.import_module("acceptance")
    finally:
        sys.path.remove(str(CASES))
        sys.path.remove(str(ROOT / "benchmarks/fluids"))
    case = _case(name)
    res = getattr(R, case["run"]["runner"]["function"])(json.loads(json.dumps(case["run"]["parameters"])), "/nonexistent",
                                                         False)
    ev = H.evaluate(case, res)
    assert ev["criteria_met"], ev["headline"]


def _b1(V, **kw):
    cd = CurrentDistribution(M.rect(8, 2, x1=0.01, y1=0.005), order=1)
    cd.set_conductivity({1: 1.0})
    cd.add_electrode("anode", [1], reactions=[Reaction("bv", 10, 0.05)], phi_s=V, **kw)
    cd.add_electrode("cathode", [2], reactions=[Reaction("bv", 10, -0.02)], phi_s=0.0)
    return cd


def test_temperature_kept_by_every_evaluation():
    """The solve's temperature is used by relres/newton_update/residual afterwards."""
    cd = _b1(0.6)
    cd.solve(temperature=330.0)
    z = cd.state()
    assert cd.relres(z) < 1e-13
    assert np.max(np.abs(cd.newton_update(z) - z)) < 1e-12
    assert cd.temperature() == 330.0
    # set_temperature before solve is honoured (one source of truth for T)
    c2 = _b1(0.6)
    c2.set_temperature(330.0)
    c2.solve()
    assert np.max(np.abs(c2.state() - z)) < 1e-12
    assert c2.relres(c2.state()) < 1e-13
    c3 = _b1(0.6)
    c3.solve()
    assert c3.temperature() == 298.15
    assert np.max(np.abs(c3.state() - z)) > 1e-4


def test_invalid_input_refused():
    """No crash, no silent acceptance."""
    cd = _b1(0.5, galvanostatic=True, current=0.1)
    with pytest.raises(Exception, match="no state"):
        cd.phi_s()
    with pytest.raises(Exception, match="unknowns"):
        cd.residual(np.zeros(3))
    g = CurrentDistribution(M.rect(4, 2, x1=0.01, y1=0.005), order=1)
    g.set_conductivity({1: 1.0})
    g.add_electrode("a", [1], reactions=[Reaction("bv", 10)], galvanostatic=True, current=0.1, phi_s=0.5)
    g.add_electrode("c", [2], reactions=[Reaction("bv", 10)], galvanostatic=True, current=-0.1)
    with pytest.raises(Exception, match="no grounded electrode"):
        g.solve()
    with pytest.raises(Exception, match="i0 must be > 0"):
        _b1(0.5).add_electrode("x", [3], reactions=[Reaction("bv", -1.0)])


def test_large_cell_voltage_routes():
    """20 V from the fixed phi = 0 start does not converge (reported, raised by default); the documented
    routes (initial='primary', continuation with warm starts) do, and agree."""
    from axionfem.chem import ConvergenceError
    cd = _b1(20.0)
    with pytest.raises(ConvergenceError, match="initial='primary'"):
        cd.solve()
    assert cd.solve(initial="primary")["converged"]
    w = _b1(2.0)
    rep = w.solve_continuation(0, [2.0 + k for k in range(19)])   # steps of 1 V (envelope)
    assert rep["converged"]
    assert np.max(np.abs(w.state() - cd.state())) < 1e-10
