"""Global (differential-evolution) optimization: af.optimize(method="global").

Known optima found from a random start with no initial guess, on coarse meshes:
  * toroid with H = R: g peaks at y = 0.5675 (the long-limit value exp(-1/2) = 0.607 at H >> R);
  * long coaxial solenoid (H/R = 8): g peaks near the long-limit x = 0.534;
  * finite current-sheet solenoid: the cost metric at the cost-optimal size for a reference readout benchmark (a
    smooth objective with an interior optimum; benchmarks/records/cost_metric_optimum_reference.json).
Also: quantity() agrees with the sensitivity engine, determinism for a seed, infeasible points
(failing meshes, timeouts, constraints), bounds and log scaling, and the amplitude root for
nonlinear iron.  The finer known-optimum checks run with AXF_SLOW=1.
"""
from __future__ import annotations

import math
import os

import pytest

import axionfem as af
from axionfem.global_opt import _evaluate, quantity

SLOW = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="set AXF_SLOW=1 for the finer DE runs")
COARSE = {"order": 2, "size": "max(R, H)/12", "surface_size": "max(R, H)/40"}


def coarse(name, **params):
    d = af.template(name, **params)
    d.doc["mesh"] = dict(COARSE)
    return d


QUANTITIES = ["Psi", "L_p", "W_mag", "F", "g_factor", "A_cold", "device_radius", "B_peak", "sc_surface_peak",
              "cost", "cost_optimal", "W_at_req", "F_over_F_req", "amplitude_scale", "B_peak_at_req",
              "sc_surface_at_req"]


@pytest.mark.parametrize("name", ["toroid", "solenoid"])
def test_quantities_match_sensitivity_engine(tmp_path, name):
    res = coarse(name).solve(workdir=tmp_path / "s", verbose=0)
    s = res.sensitivities(params=["H"], objectives=QUANTITIES)["values"]
    for q in QUANTITIES:
        assert quantity(res, q) == pytest.approx(s[q], rel=1e-12, abs=1e-300), q
    assert quantity(res, lambda r: 2 * r.L_p) == pytest.approx(2 * s["L_p"])
    with pytest.raises(ValueError, match="unknown quantity"):
        quantity(res, "nope")


def test_toroid_hole_fraction_from_random_start(tmp_path):
    d = coarse("toroid", y=0.3)                       # the start value is not used by DE
    o = af.optimize(d, method="global", objective="g_factor", sense="max", params={"y": (0.25, 0.9)},
                    popsize=6, maxiter=12, seed=3, polish=False, workdir=tmp_path, verbose=0)
    assert o["ok"] and o["method"] == "global" and o["found_by"] == "de"
    assert o["parameters"]["y"] == pytest.approx(0.5675, abs=0.01)
    # history in af.optimize's format, every point inside the bounds
    assert len(o["history"]) == o["solves"] >= 6
    for h in o["history"]:
        assert 0.25 <= h["parameters"]["y"] <= 0.9
        assert set(h) >= {"parameters", "values", "caps", "stage", "generation"}


# the readout benchmark of benchmarks/m1_validation.py (REF_BENCH): round values
REF_BENCH = {"name": "reference", "Q": 1e6, "T": 0.1, "eta_A": 1.0}
REF_COST_OPTIMAL = 73.44   # the cost metric at the coarse optimum (regression value)


def test_solenoid_cost_optimum_and_powell_polish(tmp_path):
    d = coarse("solenoid", H=3.0, x=0.4, tw="0.002*R", g="0.001*R", tc="0.001*R", tp="0.002*R",
               benchmarks=[REF_BENCH])
    o = af.global_optimize(d, objective="cost_optimal", params={"H": (0.6, 4.0, "log"), "x": (0.35, 0.75)},
                           popsize=6, maxiter=10, seed=5, polish="powell", polish_maxfev=30,
                           workdir=tmp_path, verbose=0)
    assert o["ok"] and o["polish"]["method"] == "powell"
    assert o["parameters"]["H"] == pytest.approx(1.30, rel=0.06)
    assert o["parameters"]["x"] == pytest.approx(0.56, abs=0.02)
    assert o["value"] <= o["de_best"]["values"]["cost_optimal"]
    assert o["value"] == pytest.approx(REF_COST_OPTIMAL, rel=0.02)   # regression value of the cost metric


@SLOW
def test_long_solenoid_x(tmp_path):
    d = af.template("solenoid", x=0.42, H=8.0, R=1.0)
    d.doc["mesh"] = {"order": 2, "size": "max(R, H)/30", "surface_size": "0.004*R"}
    o = af.optimize(d, method="global", objective="g_factor", sense="max", params={"x": (0.3, 0.75)},
                    popsize=6, maxiter=15, seed=11, polish=True, workers=4, workdir=tmp_path, verbose=0)
    assert o["ok"] and o["polish"]["method"] == "adjoint"
    assert o["parameters"]["x"] == pytest.approx(0.5314, abs=0.005)    # adjoint-validation value; long limit 0.534


def test_deterministic_for_a_seed(tmp_path):
    d = coarse("toroid")
    kw = dict(method="global", objective="g_factor", sense="max", params={"y": (0.3, 0.8), "H": (1.0, 4.0)},
              popsize=3, maxiter=2, polish=False, verbose=0)
    a = af.optimize(d, seed=7, workdir=tmp_path / "a", **kw)
    b = af.optimize(d, seed=7, workdir=tmp_path / "b", **kw)
    c = af.optimize(d, seed=8, workdir=tmp_path / "c", **kw)
    pa = [(h["parameters"], h["values"]) for h in a["history"]]
    assert pa == [(h["parameters"], h["values"]) for h in b["history"]]
    assert [h["parameters"] for h in a["history"]] != [h["parameters"] for h in c["history"]]
    # a process pool gives the same search
    p = af.optimize(d, seed=7, workers=2, workdir=tmp_path / "p", **kw)
    assert pa == [(h["parameters"], h["values"]) for h in p["history"]]
    # ... and so does a serial run from a caller with several OpenMP threads (threads=1 is used, then restored)
    from axionfem import _core

    old = _core.get_threads()
    af.set_threads(4)
    try:
        t = af.optimize(d, seed=7, workdir=tmp_path / "t", **kw)
        assert _core.get_threads() == 4
    finally:
        af.set_threads(old)
    assert pa == [(h["parameters"], h["values"]) for h in t["history"]]
    # with the adjoint polish (run in the calling process): serial and pool agree from a 4-thread caller
    kwp = {**kw, "polish": True, "polish_max_iter": 3}
    af.set_threads(4)
    try:
        s1 = af.optimize(d, seed=7, workdir=tmp_path / "s1", **kwp)
        s2 = af.optimize(d, seed=7, workers=2, workdir=tmp_path / "s2", **kwp)
        assert _core.get_threads() == 4
    finally:
        af.set_threads(old)
    assert s1["polish"]["method"] == "adjoint" and s1["parameters"] == s2["parameters"]
    assert [(h["parameters"], h["values"]) for h in s1["history"]] == [(h["parameters"], h["values"])
                                                                      for h in s2["history"]]


def test_failed_meshes_are_infeasible_points(tmp_path):
    # y > 1 puts the inner radius outside the outer one: the design is invalid there
    d = coarse("toroid")
    o = af.optimize(d, method="global", objective="g_factor", sense="max", params={"y": (0.4, 1.6)},
                    popsize=8, maxiter=4, seed=2, polish=False, workdir=tmp_path, verbose=0)
    errs = [h for h in o["history"] if "error" in h]
    assert errs and all(not h["feasible"] for h in errs)
    assert o["ok"] and o["parameters"]["y"] < 1.0


def test_constraint_feasibility_rule(tmp_path):
    d = coarse("toroid")
    # g peaks at y = 0.57; L_p grows with y (the hole-mode inductance), so L_p >= L_p(y = 0.7) means y >= 0.7
    L07 = quantity(d.with_params(y=0.7).solve(workdir=tmp_path / "ref", verbose=0), "L_p")
    o = af.optimize(d, method="global", objective="g_factor", sense="max", params={"y": (0.3, 0.9)},
                    constraints=[("L_p", ">=", L07)], popsize=6, maxiter=10, seed=4, polish=False,
                    workdir=tmp_path / "run", verbose=0)
    assert o["ok"] and o["feasible"]
    assert o["parameters"]["y"] == pytest.approx(0.7, abs=0.02)
    assert o["values"]["L_p"] >= L07 * (1 - 1e-12)
    assert any(not h["feasible"] for h in o["history"])


def test_no_feasible_point(tmp_path):
    d = coarse("toroid")
    o = af.optimize(d, method="global", objective="g_factor", params={"y": (0.4, 0.8)},
                    constraints=[{"quantity": "L_p", "type": "<=", "value": 1e-30}], popsize=3, maxiter=3,
                    seed=0, workdir=tmp_path, verbose=0)
    assert not o["ok"] and not o["feasible"] and o["polish"].get("skipped")
    # an all-infeasible population is not "converged": DE runs every generation
    assert o["de"]["nit"] == 3 and not o["converged"]


def test_timeouts_become_failed_points(tmp_path):
    d = coarse("toroid")
    o = af.optimize(d, method="global", objective="g_factor", params={"y": (0.4, 0.8)}, popsize=2, maxiter=0,
                    seed=0, workers=2, timeout=1e-3, polish=False, workdir=tmp_path, verbose=0)
    assert not o["ok"] and o["history"] and all("timeout" in h["error"] for h in o["history"])
    assert o["parameters"] is None and o["constraints"] == [] and o["de"]["nit"] == 0


def test_bounds_and_arguments(tmp_path):
    d = coarse("toroid")
    with pytest.raises(ValueError, match="low < high"):
        af.global_optimize(d, params={"y": (0.8, 0.4)})
    with pytest.raises(ValueError, match="low > 0"):
        af.global_optimize(d, params={"y": (0.0, 0.8)}, log=["y"])
    with pytest.raises(KeyError, match="unknown parameters"):
        af.global_optimize(d, params={"nope": (0, 1)})
    with pytest.raises(ValueError, match="cannot also be optimized"):
        af.global_optimize(d, params={"NI": (1e5, 1e7)}, amplitude="NI")
    with pytest.raises(TypeError):
        af.optimize(d, "g_factor", {"y": (0.4, 0.8)}, None, method="global")
    # log-scaled search stays inside the bounds and hits them exactly at the unit-cube corners
    o = af.global_optimize(d, objective="g_factor", sense="max", params={"H": (0.5, 8.0, "log")},
                           popsize=4, maxiter=1, seed=1, polish=False, workdir=tmp_path, verbose=0)
    for h in o["history"]:
        assert 0.5 <= h["parameters"]["H"] <= 8.0


def test_nonlinear_amplitude_root(tmp_path):
    d = af.template("core", iron_t="0.3*Rc")
    d.doc["mesh"] = {"order": 2, "size": "max(b, H)/10", "surface_size": "min(w, gn)"}
    task = {"doc": d.doc, "parameters": {}, "quantities": ["cost", "F_over_F_req"], "benchmark": "",
            "amp_bench": "", "free": [], "order": 0, "workdir": str(tmp_path / "e"), "amplitude": "auto", "amplitude_tol": 1e-6,
            "amplitude_max_solves": 16, "keep_runs": False}
    r = _evaluate(task)
    assert "error" not in r, r.get("error")
    assert r["linear"] is False and r["amplitude"]["parameter"] == "NI"
    assert abs(r["values"]["F_over_F_req"] - 1) <= 1e-6
    assert r["amplitude"]["solves"] >= 2
    assert r["amplitude"]["solves"] <= 16
    assert "error" not in _evaluate({**task, "quantities": ["iron_mass"]})       # no root without a requirement quantity
    assert "also optimized" in _evaluate({**task, "free": ["NI"]})["error"]
    assert "amplitude must be solved" in _evaluate({**task, "amplitude": None})["error"]


def test_mcp_worker_job(tmp_path, capsys):
    pytest.importorskip("mcp")
    from axionfem import _mcp_worker, mcp_server

    assert "optimize_global" in _mcp_worker.JOBS and hasattr(mcp_server, "optimize_global")
    d = coarse("toroid")
    out = _mcp_worker.JOBS["optimize_global"]({"design": d.doc, "objective": "g_factor", "sense": "max",
                                              "free": {"y": [0.4, 0.8]}, "seed": 1, "popsize": 3, "maxiter": 1,
                                              "polish": False, "workdir": str(tmp_path)})
    assert out["ok"] and out["method"] == "global" and 0.4 <= out["parameters"]["y"] <= 0.8
    assert "AXF-PROGRESS" in capsys.readouterr().out


def _g_or_crash(res):
    """Objective that kills its worker process for y > 0.7 (a stand-in for a crashing mesher or solver)."""
    if res.design.resolved_params()["y"] > 0.7:
        os._exit(3)
    return res.g


def test_worker_crash_is_charged_to_its_point_and_powell_polish_with_pool(tmp_path):
    d = coarse("toroid")
    o = af.optimize(d, method="global", objective=_g_or_crash, sense="max", params={"y": (0.4, 0.9)},
                    popsize=4, maxiter=2, seed=6, workers=3, polish="powell", polish_maxfev=6,
                    workdir=tmp_path, verbose=0)
    de = [h for h in o["history"] if h["stage"] == "de"]
    for h in de:
        if h["parameters"]["y"] > 0.7:
            assert "worker process died" in h["error"]
        else:
            assert h["feasible"], h.get("error")            # neighbours of a crash are not tainted
    assert o["ok"] and o["polish"]["method"] == "powell" and o["parameters"]["y"] <= 0.7
    assert any(h["stage"] == "polish" for h in o["history"])


def test_precheck_and_argument_checks(tmp_path):
    d = coarse("toroid")
    o = af.optimize(d, method="global", objective="g_factor", sense="max", params={"y": (0.3, 0.9)},
                    precheck=lambda p: None if p["y"] <= 0.5 else "y above 0.5", popsize=4, maxiter=2, seed=1,
                    polish="powell", polish_maxfev=8, workdir=tmp_path, verbose=0)
    assert o["ok"] and o["parameters"]["y"] <= 0.5
    assert all(h["parameters"]["y"] <= 0.5 or h["error"].startswith("precheck") for h in o["history"])
    with pytest.raises(ValueError, match="polish must be"):
        af.optimize(d, method="global", params={"y": (0.3, 0.9)}, polish="lbfgs")
    with pytest.raises(ValueError, match="unknown quantity"):
        af.optimize(d, method="global", objective="cosst", params={"y": (0.3, 0.9)})


# ---------------------------------------------------------------- the driver with stand-in evaluations
# Unit tests with stand-ins for the FE evaluation (no mesh, no solve): failures, timeouts, crashes and thread counts.
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

import axionfem.global_opt as go  # noqa: E402


class _FakeRes:
    """g peaks at y = 0.57; L_p = y."""

    data: dict = {}

    def __init__(self, p):
        self.p = p

    @property
    def g(self):
        return 1.0 - (self.p.get("y", 0.5) - 0.57) ** 2

    @property
    def L_p(self):
        return self.p.get("y", 0.5)


def _fake_eval(task, seconds=0.0):
    """_evaluate's contract (keys, config errors, threads) without the FE solve."""
    from axionfem import _core

    p = dict(task["parameters"])
    rec = {"parameters": p}
    quantities = task["quantities"]
    if "quantities_pickle" in task:
        import pickle

        quantities = pickle.loads(task["quantities_pickle"])
    keys = task.get("keys") or [go._qname(q) for q in quantities]
    res = _FakeRes(p)
    named = {"g_factor": res.g, "L_p": res.L_p, "cost": 10 + (p.get("y", .5) - .3) ** 2}
    rec["values"] = {k: float(q(res)) if callable(q) else named[q.split(":")[0]] for k, q in zip(keys, quantities)}
    rec.update(caps={}, linear=True, threads_seen=_core.get_threads(), seconds=seconds)
    return rec


def _y(r):
    return r.L_p


def test_callables_with_the_same_name_keep_their_own_values(tmp_path, monkeypatch):
    monkeypatch.setattr(go, "_evaluate", _fake_eval)
    d = af.template("toroid")
    o = af.optimize(d, method="global", objective=lambda r: r.g, sense="max", params={"y": (0.3, 0.9)},
                    constraints=[(lambda r: r.L_p, ">=", 0.4)], popsize=6, maxiter=10, seed=0, polish=False,
                    workdir=tmp_path, verbose=0)
    assert o["objective"] == "<lambda>" and o["constraints"][0]["quantity"] == "<lambda>#2"
    y = o["parameters"]["y"]
    assert y < 0.7 and o["value"] == pytest.approx(1 - (y - 0.57) ** 2) and o["value"] >= 0.995  # g's peak
    assert o["values"]["<lambda>#2"] == y
    # a callable named like a named quantity does not overwrite it
    cost = types.FunctionType(_y.__code__, globals(), "cost")
    o = af.optimize(d, method="global", objective="cost", params={"y": (0.3, 0.9)},
                    constraints=[(cost, ">=", 0.5)], popsize=6, maxiter=10, seed=0, polish=False,
                    workdir=tmp_path, verbose=0)
    assert o["parameters"]["y"] == pytest.approx(0.5, abs=0.02)
    assert o["values"]["cost#2"] == o["parameters"]["y"] and o["value"] == pytest.approx(10.04, abs=0.01)


def test_precheck_numpy_bool_and_empty_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(go, "_evaluate", _fake_eval)
    d = af.template("toroid")
    kw = dict(method="global", objective="g_factor", sense="max", params={"y": (0.3, 0.9)}, popsize=4,
              maxiter=3, seed=1, polish=False, workdir=tmp_path, verbose=0)
    o = af.optimize(d, precheck=lambda p: np.float64(p["y"]) <= 0.8, **kw)            # numpy True accepts
    assert o["ok"] and all(h["parameters"]["y"] <= 0.8 or "precheck" in h["error"] for h in o["history"])
    assert any(h["feasible"] for h in o["history"])
    o = af.optimize(d, precheck=lambda p: "" if p["y"] > 0.5 else None, **kw)         # an empty reason rejects
    assert o["parameters"]["y"] <= 0.5
    assert all(h["parameters"]["y"] <= 0.5 or "precheck" in h["error"] for h in o["history"])
    with pytest.raises(TypeError, match="precheck must return"):
        af.optimize(d, precheck=lambda p: 1, **kw)


class _NonlinRes:
    def __init__(self, NI):
        self.data = {"linear": False, "coupling": {}, "geometry": {}, "fields": {},
                     "benchmarks": [{"name": "optimistic", "F_over_F_req": (NI / 1e6) ** 3, "amplitude_scale": 1.0,
                                     "cost": {"total_MCHF": 1.0}}]}

    def benchmark(self, b):
        if b != "optimistic":
            raise KeyError(f"no benchmark {b!r}; available: ['optimistic']")
        return self.data["benchmarks"][0]


@pytest.mark.parametrize("case", ["NI optimized", "unknown benchmark", "max solves"])
def test_configuration_errors_stop_at_the_first_point(tmp_path, monkeypatch, case):
    calls = []

    def fake_solve(design, task, wd):
        calls.append(1)
        return _NonlinRes(design.resolved_params()["NI"])

    monkeypatch.setattr(go, "_solve", fake_solve)
    d = af.template("core")
    kw = dict(method="global", objective="cost", params={"NI": (1e5, 1e7, "log")}, popsize=4, maxiter=5, seed=0,
              polish=False, workdir=tmp_path, verbose=0)
    if case == "NI optimized":
        with pytest.raises(ValueError, match="also optimized"):
            af.optimize(d, **kw)
    elif case == "unknown benchmark":
        with pytest.raises(ValueError, match="no benchmark 'nope'"):
            af.optimize(d, **{**kw, "objective": "cost:nope", "params": {"Rc": (0.5, 2.0)}})
    else:
        with pytest.raises(ValueError, match="amplitude_max_solves"):
            af.optimize(d, amplitude_max_solves=0, **{**kw, "params": {"Rc": (0.5, 2.0)}})
    assert len(calls) <= 1                       # not popsize * n * (maxiter + 1) points


def test_benchmark_summary_and_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(go, "_evaluate", _fake_eval)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    try:
        import global_opt_validation as gv
    finally:
        sys.path.pop(0)
    o = af.optimize(af.template("toroid"), method="global", objective="g_factor", sense="max",
                    params={"y": (0.25, 0.9)}, popsize=8, maxiter=20, seed=20261029, polish=False, workdir=tmp_path,
                    verbose=0)
    s = gv.summary(o)
    assert s["converged"] and s["message"].startswith("converged") and "callback" not in s["message"]
    assert s["de"]["scipy_message"] and o["solves"] == o["evaluations"] == len(o["history"])
    o = af.optimize(af.template("toroid"), method="global", objective="g_factor", sense="max",
                    params={"y": (0.25, 0.9)}, popsize=8, maxiter=1, tol=0, seed=1, polish=False, workdir=tmp_path,
                    verbose=0)
    assert not o["converged"] and o["message"].startswith("maximum number of generations")


def _threads_obj(res):
    from axionfem import _core

    return float(_core.get_threads())


def test_threads_reach_map_evaluations_and_the_polish(tmp_path, monkeypatch):
    import axionfem.sensitivity as sens
    from axionfem import _core

    seen = []

    def fake_local(start, **kw):
        seen.append(_core.get_threads())
        return {"ok": False, "history": []}

    def map_at_3_threads(fn, tasks):             # a map running elsewhere starts at its own thread count
        out = []
        for t in tasks:
            af.set_threads(3)
            out.append(fn(t))
        return out

    monkeypatch.setattr(go, "_solve_at_req", lambda d, task, wd: (_FakeRes(task["parameters"]), None))
    monkeypatch.setattr(go, "_caps", lambda res, b: {})
    monkeypatch.setattr(sens, "optimize", fake_local)
    old = _core.get_threads()
    af.set_threads(5)
    try:
        o = af.optimize(af.template("toroid"), method="global", objective=_threads_obj, params={"y": (0.3, 0.9)},
                        popsize=3, maxiter=1, seed=0, workers=map_at_3_threads, threads=1, polish=False,
                        workdir=tmp_path, verbose=0)
        assert {h["values"]["_threads_obj"] for h in o["history"]} == {1.0}
        monkeypatch.setattr(go, "_evaluate", _fake_eval)
        af.optimize(af.template("toroid"), method="global", objective="g_factor", sense="max",
                    params={"y": (0.3, 0.9)}, popsize=3, maxiter=1, seed=0, threads=2, polish="adjoint",
                    polish_maxfev=3, workdir=tmp_path, verbose=0)
        assert seen == [2] and _core.get_threads() == 5          # the polish ran on `threads`; caller restored
    finally:
        af.set_threads(old)


def test_fixed_sources_are_not_polished_by_the_adjoint(tmp_path, monkeypatch):
    import axionfem.sensitivity as sens

    def ev(task):   # a linear design with a background field: every DE point needed the amplitude root
        y = task["parameters"]["y"]
        return {"parameters": dict(task["parameters"]), "values": {"cost": 10 + (y - .5) ** 2}, "caps": {},
                "linear": True, "amplitude": {"parameter": "NI", "value": 2.0e6 * (1 + y), "F_over_F_req": 1.0,
                                              "solves": 2}}

    def local(start, **kw):     # would "win" by the rescaling approximation alone
        return {"ok": True, "history": [{"parameters": {"y": 0.5}, "values": {"cost": 9.9}, "caps": {}}]}

    monkeypatch.setattr(go, "_evaluate", ev)
    monkeypatch.setattr(sens, "optimize", local)
    o = af.optimize(af.template("toroid"), method="global", objective="cost", params={"y": (0.3, 0.9)}, popsize=3,
                    maxiter=3, seed=0, polish_maxfev=6, workdir=tmp_path, verbose=0)
    assert o["polish"]["method"] == "powell" and o["amplitude"] is not None
    assert af.Design(o["design"]).resolved_params()["NI"] == pytest.approx(o["amplitude"]["value"])
    assert o["solves"] == 2 * o["evaluations"]                    # the amplitude-root solves are counted
    o = af.optimize(af.template("toroid"), method="global", objective="cost", params={"y": (0.3, 0.9)}, popsize=3,
                    maxiter=1, seed=0, polish="adjoint", polish_maxfev=4, workdir=tmp_path, verbose=0)
    assert o["polish"]["method"] == "powell" and "amplitude root" in o["polish"]["fallback_reason"]


def test_adjoint_polish_is_replaced_when_it_cannot_fit_in_max_time(tmp_path, monkeypatch):
    import axionfem.sensitivity as sens

    monkeypatch.setattr(go, "_evaluate", lambda t: _fake_eval(t, seconds=100.0))
    monkeypatch.setattr(sens, "optimize", lambda *a, **k: pytest.fail("the adjoint polish must not start"))
    o = af.optimize(af.template("toroid"), method="global", objective="g_factor", sense="max",
                    params={"y": (0.3, 0.9)}, popsize=3, maxiter=2, seed=0, max_time=600, polish_maxfev=4,
                    workdir=tmp_path, verbose=0)
    assert o["polish"]["method"] == "powell" and "max_time" in o["polish"]["fallback_reason"]


def test_argument_validation(tmp_path):
    d = coarse("toroid")
    p = {"y": (0.3, 0.9)}
    for kw, match in [(dict(params=p, bounds=p), "not both"), (dict(params=p, x0={"H": 1.0}), "x0 names"),
                      (dict(params=p, x0={"y": 0.95}), "outside the bounds"),
                      (dict(params={"y": (0.3, 0.9, "log")}, init=[[0.5], [-1.0]]), "outside the bounds"),
                      (dict(params=p, init=[[0.5, 0.5]]), "one column per parameter"),
                      (dict(params=p, threads=0), "threads"), (dict(params=p, workers=True), "workers"),
                      (dict(params=p, popsize=0), "popsize"), (dict(params=p, tol=-1), "tol"),
                      (dict(params=p, amplitude_tol=0), "amplitude_tol"),
                      (dict(params=p, constraints=[("L_p", "<=")]), "constraint tuple"),
                      (dict(params=p, constraints=[{"type": "<=", "value": 1}]), "needs 'quantity'"),
                      (dict(params=p, precheck=lambda q: None, polish="adjoint"), "cannot apply the precheck")]:
        with pytest.raises(ValueError, match=match):
            af.global_optimize(d, objective="g_factor", workdir=tmp_path, verbose=0, **kw)
    # numpy integers are accepted as workers; the method may still be given positionally (local optimizer)
    import axionfem.sensitivity as sens

    got = {}
    orig = sens.optimize

    def fake(design, objective="cost", params=None, bounds=None, constraints=(), sense="min", method=None, **k):
        got.update(method=method, sense=sense)
        return {}

    try:
        sens.optimize = fake
        af.optimize(d, "g_factor", p, None, (), "max", "SLSQP")
        assert got == {"method": "SLSQP", "sense": "max"}
        af.optimize(d, "g_factor", p, method="L-BFGS-B")
        assert got["method"] == "L-BFGS-B"
    finally:
        sens.optimize = orig
    with pytest.warns(UserWarning, match="timeout"):
        with pytest.raises(ValueError, match="outside the bounds"):
            af.global_optimize(d, objective="g_factor", params=p, timeout=5, x0={"y": 2.0}, workers=np.int64(1),
                               workdir=tmp_path, verbose=0)


def test_newton_message_without_residual():
    class R:
        data = {"checks": [{"name": "static poloidal solve", "converged": False, "newton_iterations": 50,
                            "relative_residual": None}]}

    with pytest.raises(RuntimeError, match="did not converge .*relative residual None"):
        go._check_newton(R())


def test_huge_feasible_objective_is_a_failed_point(tmp_path, monkeypatch):
    monkeypatch.setattr(go, "_evaluate", lambda t: {**_fake_eval(t), "values": {"g_factor": 1e120}})
    o = af.optimize(af.template("toroid"), method="global", objective="g_factor", params={"y": (0.3, 0.9)},
                    popsize=2, maxiter=0, seed=0, polish=False, workdir=tmp_path, verbose=0)
    assert not o["ok"] and all("rescale" in h["error"] for h in o["history"])


# --- process-pool stand-ins (module level: the spawn workers import them from this file)
def _sleep_or_crash(task):
    if task["parameters"]["crash"]:
        time.sleep(1.0)
        os._exit(3)
    time.sleep(task["parameters"].get("sleep", 12.0))
    return {"parameters": task["parameters"], "values": {"g_factor": 1.0}}


class _Detected(Exception):
    pass


def test_crash_of_the_last_spawned_worker_is_detected_promptly():
    # CPython 3.12 spawns workers on demand and may see the newest one die only when a sibling finishes
    pool = go._Pool(3, 1, timeout=None, fn=_sleep_or_crash)
    t0 = time.monotonic()
    tasks = [{"parameters": {"crash": c, "sleep": 40.0}, "workdir": "/nonexistent", "keep_runs": True}
             for c in (False, False, True)]

    def spy():
        raise _Detected(time.monotonic() - t0)

    pool._new = spy                  # the restart after the crash: stop there and report when it happened
    try:
        with pytest.raises(_Detected) as e:
            pool.map(tasks)
    finally:
        pool._kill()
    assert e.value.args[0] < 30.0, e.value.args          # siblings take 40 s: detected before they finish


def _ghost_module_objective():
    m = types.ModuleType("ghost_module_not_importable")
    exec("def obj(res):\n    return 1.0\n", m.__dict__)
    sys.modules[m.__name__] = m                 # pickles by reference here, cannot be imported in a worker
    return m.obj


def test_callable_that_cannot_be_loaded_in_a_worker_is_a_configuration_error(tmp_path):
    obj = _ghost_module_objective()
    try:
        with pytest.raises(ValueError, match="cannot be loaded in the worker"):
            af.optimize(coarse("toroid"), method="global", objective=obj, params={"y": (0.3, 0.9)}, popsize=3,
                        maxiter=2, seed=0, workers=2, polish=False, workdir=tmp_path, verbose=0)
    finally:
        sys.modules.pop("ghost_module_not_importable", None)
