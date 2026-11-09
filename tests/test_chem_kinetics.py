"""Stiff chemical kinetics, reactors and time integration (axionfem.chem.kinetics, _core.chem): API unit tests, and a
selection of the short verification cases judged by benchmarks/fluids/acceptance.py (the full set runs with
benchmarks/chem/run.py)."""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/chem/cases"
REG = ROOT / "benchmarks/chem/registry/cases"

pytest.importorskip("axionfem.chem")
try:
    from axionfem._core import chem as _corechem
    _corechem.Mechanism  # noqa: B018
except Exception:  # pragma: no cover - a build without the kinetics bindings
    pytest.skip("axionfem._core.chem has no kinetics bindings (rebuild)", allow_module_level=True)

from axionfem.chem import kinetics as K  # noqa: E402


def _a_to_b(k=0.7):
    return K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": k}]})


# ------------------------------------------------------------------------------------------------------ kernel
@pytest.mark.parametrize("bad", ["f", "jac", "source"])
def test_wrong_sized_callbacks_are_refused(bad):
    """The sizes of f, jac and source are checked with a message naming the callback."""
    f = (lambda t, y: -y[:1]) if bad == "f" else (lambda t, y: -y)
    J = (lambda t, y: -np.eye(1)) if bad == "jac" else (lambda t, y: -np.eye(3))
    src = (lambda t: np.zeros(1)) if bad == "source" else None
    P = K.problem(3, f, J, source=src)
    with pytest.raises(ValueError, match=bad):
        K.grid_bdf(P, 1, [0.0, 0.1], [[1.0, 2.0, 3.0]])


def test_rates_and_jacobian_against_finite_differences():
    mech = K.mechanism({"species": ["A", "B", "C", "D"], "reactions": [
        {"reactants": {"A": 1, "B": 1}, "products": {"C": 1}, "A": 0.6, "reversible": True, "Kc": 2.5},
        {"reactants": {"C": 2}, "products": {"D": 1}, "A": 0.35, "Ea": 5e3, "n": 0.5},
        {"reactants": {"A": 1, "C": 1}, "products": {"D": 1}, "A": 0.3, "orders": {"A": 0.5, "C": 1.5}},
        {"reactants": {"D": 1}, "products": {"A": 1, "B": 1}, "A": 0.2, "reversible": True,
         "kr": {"A": 0.05, "n": 0.5, "Ea": 2e3}}]})
    c, T = np.array([1.1, 0.8, 0.6, 0.5]), 330.0
    w, J, dT = K.rates(mech, c, T)
    h = 1e-6
    Jfd = np.column_stack([(K.rates(mech, c + h * e, T)[0] - K.rates(mech, c - h * e, T)[0]) / (2 * h)
                           for e in np.eye(4)])
    assert np.max(np.abs(J - Jfd)) < 1e-8 * np.max(np.abs(J))
    dTfd = (K.rates(mech, c, T + 1e-3)[0] - K.rates(mech, c, T - 1e-3)[0]) / 2e-3
    assert np.max(np.abs(dT - dTfd)) < 1e-6 * np.max(np.abs(dT))
    S = mech.stoichiometry()
    q = mech.progress(c, T)[0]
    assert np.allclose(w, S @ q, rtol=0, atol=1e-15)


def test_noninteger_order_convention_at_negative_concentration():
    mech = K.mechanism({"species": ["A", "C", "E"], "reactions": [
        {"reactants": {"A": 1, "C": 1}, "products": {"E": 1}, "A": 1.0, "orders": {"A": 0.5, "C": 1.5}}]})
    w, J, _ = K.rates(mech, [-0.1, 0.5, 0.0])
    assert w[2] == 0.0 and J[2, 0] == 0.0


def test_units_are_converted_by_the_kernel():
    si = K.mechanism({"species": ["A", "B"], "reactions": [
        {"reactants": {"A": 1}, "products": {"B": 2}, "A": 0.8, "reversible": True, "Kc": 2.0}]})
    lm = K.mechanism({"species": ["A", "B"], "units": {"concentration": "mol/L", "time": "min", "energy": "cal/mol"},
                      "reactions": [{"reactants": {"A": 1}, "products": {"B": 2}, "A": 48.0, "reversible": True,
                                     "Kc": 0.002}]})
    c = np.array([0.9, 0.4])
    w_si = K.rates(si, c)[0]
    w_lm = K.rates(lm, lm.concentration_to_si(c * 1e-3))[0]
    assert np.allclose(w_si, w_lm, rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="unknown concentration unit"):
        K.mechanism({"species": ["A"], "units": {"concentration": "furlongs"}, "reactions": []})


def test_refusals():
    cyc = {"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"A": 1}, "products": {"B": 1}, "A": 1.2, "reversible": True, "Kc": 2.0},
        {"reactants": {"B": 1}, "products": {"C": 1}, "A": 0.7, "reversible": True, "Kc": 1.5},
        {"reactants": {"C": 1}, "products": {"A": 1}, "A": 0.9, "reversible": True, "Kc": 2.0 / 3.0}]}
    with pytest.raises(ValueError, match="Wegscheider"):
        K.mechanism(cyc)
    K.mechanism(cyc, allow_nonequilibrium_cycle=True)
    cyc["reactions"][2]["Kc"] = 1.0 / 3.0
    K.mechanism(cyc)
    with pytest.raises(ValueError, match="REVERSIBLE"):
        K.mechanism({"species": ["A", "B"], "reactions": [
            {"reactants": {"A": 1}, "products": {"B": 1}, "A": 1.0, "reversible": True, "Kc": 2.0,
             "orders": {"A": 0.5}}]})
    m = K.mechanism({"species": ["A", "B"], "reactions": [
        {"reactants": {"A": 1}, "products": {"B": 1}, "A": 1.0, "reversible": True, "Kc": 2.0, "dH": -1e4}]})
    with pytest.raises(ValueError, match="van 't Hoff"):
        K.adiabatic_liquid(m, 4e6)


# --------------------------------------------------------------------------------------------------- reactors
@pytest.mark.parametrize("which", ["cstr", "semibatch", "gas", "adiabatic", "pfr"])
def test_reactor_jacobians_taylor(which):
    m = K.mechanism({"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"A": 2}, "products": {"B": 1}, "A": 0.4, "Ea": 4e3, "dH": -2e4},
        {"reactants": {"A": 1, "B": 1}, "products": {"C": 1}, "A": 0.5, "Ea": 3e3, "dH": -1e4}]})
    if which == "cstr":
        P, x = K.cstr(m, 2.0, [1.0, 0.2, 0.0], T=320.0), np.array([0.9, 0.6, 0.4])
    elif which == "semibatch":
        P, x = K.semibatch(m, 0.3, [2.0, 0.5, 0.0], T=320.0), np.array([0.9, 0.6, 0.4, 2.3])
    elif which == "gas":
        P, x = K.const_p_gas(m, 500.0, 2e5), np.array([30.0, 20.0, 10.0])
    elif which == "adiabatic":
        P, x = K.adiabatic_liquid(m, 4e3), np.array([0.9, 0.6, 0.4, 330.0])
    else:
        P, x = K.pfr(m, 600.0, 0.05, 2.0), np.array([1.3, 1.1, 0.2])
    rng = np.random.default_rng(1)
    v = rng.standard_normal(len(x)) * np.abs(x)
    F0, Jv = P.rhs(0.0, x), P.jacobian(0.0, x) @ v
    r = [np.linalg.norm(P.rhs(0.0, x + e * v) - F0 - e * Jv) for e in (1e-2, 1e-3, 1e-4)]
    assert 1.9 < math.log(r[0] / r[2]) / math.log(100.0) < 2.1


def test_consistent_init_refuses_unreachable_states():
    """No consistent state -> an error, never inf or a non-solution."""
    es = K.equilibrium({"species": ["A", "B"], "reactions": []},
                       [{"reactants": {"A": 1}, "products": {"B": 2}, "K": 2.0}])
    assert np.allclose(es.consistent_init(np.array([1.0, 0.0])), [0.5, 1.0])
    for c0 in ([1.0, -0.5], [-1.0, 0.0]):
        with pytest.raises(RuntimeError, match="consistent_init"):
            es.consistent_init(np.array(c0))


def test_equilibrium_invariant_form_and_consistent_init():
    es = K.equilibrium({"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"B": 1}, "products": {"C": 1}, "A": 0.6}]}, [{"reactants": {"A": 1}, "products": {"B": 1},
                                                                    "K": 3.0}])
    assert np.allclose(es.W(), [[1, 1, 0], [0, 0, 1]])
    c0 = es.consistent_init(np.array([1.6, 0.2, 0.0]))
    assert np.allclose(c0, [0.45, 1.35, 0.0], rtol=1e-15, atol=1e-15)
    tr = K.adaptive(es.problem(), "onestep", 0.0, c0, [1.0, 4.0], 1e-8, 1e-10)
    assert max(abs(es.constraint(y)[0]) for y in tr.y) < 1e-12


# EquilibriumSystem.problem() returns a copy that must not share a GC-traversed callable with the system
_EQ_KIN = {"species": ["A", "B", "C"], "reactions": [{"reactants": {"B": 1}, "products": {"C": 1}, "A": 0.6}]}
_EQ_EQ = [{"reactants": {"A": 1}, "products": {"B": 1}, "K": 3.0}]
_EQ_Y = np.array([1.0, 3.0, 0.5])

_LEAK_PROGRAMS = {
    # a two-line program (no cycle at all)
    "plain": "S = K.equilibrium(kin, eq, source=lambda t: np.zeros(3)); P = S.problem()",
    # the system in a cycle through its source, the copy alive at exit
    "cycle": ("box = {}\n"
              "def src(t, box=box): return np.zeros(3)\n"
              "S = K.equilibrium(kin, eq, source=src); box['S'] = S; P = S.problem(); del S, src, box\n"
              "gc.collect(); P.rhs(0.0, np.array([1.0, 3.0, 0.5]))"),
    # the copy itself inside the cycle
    "copy_in_cycle": ("box = {}\n"
                      "def src(t, box=box): return np.zeros(3)\n"
                      "S = K.equilibrium(kin, eq, source=src); P = S.problem(); box['S'] = S; box['P'] = P\n"
                      "del S, P, src, box"),
}


@pytest.mark.parametrize("prog", sorted(_LEAK_PROGRAMS))
def test_equilibrium_problem_copy_exits_without_leak(prog):
    """S.problem() with a Python source leaked both objects at exit ("nanobind: leaked 2 instances").
    A fresh interpreter must exit 0 with no "leaked" line on stderr."""
    import subprocess
    code = ("import gc, numpy as np\nfrom axionfem.chem import kinetics as K\n"
            f"kin = {_EQ_KIN!r}\neq = {_EQ_EQ!r}\n" + _LEAK_PROGRAMS[prog] + "\nprint('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok", r.stdout
    assert "leaked" not in r.stderr, r.stderr


def test_equilibrium_problem_copy_outlives_the_system():
    """The copy keeps working (its source still called, the same values) after the system's name is
    deleted and the GC has run."""
    import gc
    src = lambda t: np.array([0.25, -0.5, 0.125]) * (1.0 + t)  # noqa: E731
    r_nosrc = K.equilibrium(_EQ_KIN, _EQ_EQ).problem().rhs(0.5, _EQ_Y)
    S = K.equilibrium(_EQ_KIN, _EQ_EQ, source=src)
    r_ref = S.problem().rhs(0.5, _EQ_Y)
    assert not np.allclose(r_ref, r_nosrc)            # the source contributes
    P = S.problem()
    del S, src
    for _ in range(3):
        gc.collect()
    assert P.has_source()
    assert np.array_equal(P.rhs(0.5, _EQ_Y), r_ref)
    c0 = K.equilibrium(_EQ_KIN, _EQ_EQ).consistent_init(np.array([1.6, 0.2, 0.0]))
    tr = K.adaptive(P, "onestep", 0.0, c0, [1.0], 1e-8, 1e-10)
    assert np.all(np.isfinite(tr.y[-1]))


@pytest.mark.parametrize("copy_in_cycle", [False, True])
def test_equilibrium_system_in_a_cycle_is_collected(copy_in_cycle):
    """A system in a garbage cycle (its source refers back to it) is kept alive by a live problem()
    copy, which stays usable, and is collected once the copy goes; with the copy inside the cycle the whole cycle is
    collected by one gc.collect().  The source function's weakref tells whether the system was freed."""
    import gc
    import weakref
    box = {}

    def src(t, box=box):
        return np.zeros(3)

    alive = weakref.ref(src)
    S = K.equilibrium(_EQ_KIN, _EQ_EQ, source=src)
    box["S"] = S
    P = S.problem()
    if copy_in_cycle:
        box["P"] = P
        del S, P, src, box
        gc.collect()
        assert alive() is None, "the cycle system -> source -> box -> copy -> system was not collected"
        return
    del S, src, box
    gc.collect()
    assert alive() is not None                        # the copy keeps the system (and its source) alive
    assert np.allclose(P.rhs(0.0, _EQ_Y), K.equilibrium(_EQ_KIN, _EQ_EQ).problem().rhs(0.0, _EQ_Y))
    del P
    gc.collect()
    assert alive() is None, "the system was not collected after its copy went"


def test_equilibrium_tp_clear_of_one_object_keeps_the_others_callable():
    """The copy owns its own reference (to the system), so tp_clear of the copy does not empty the
    system's source; tp_clear of the system (which the GC never runs while a copy lives) makes the copy refuse with an
    error instead of calling a null handle."""
    import ctypes
    Py_tp_clear = 51                                   # Include/typeslots.h

    def tp_clear(obj):
        get = ctypes.pythonapi.PyType_GetSlot
        get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_int]
        fn = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(get(type(obj), Py_tp_clear))   # keeps the GIL
        assert fn(obj) == 0

    calls = []
    S = K.equilibrium(_EQ_KIN, _EQ_EQ, source=lambda t: (calls.append(t), np.ones(3))[1])
    r_ref = S.problem().rhs(0.0, _EQ_Y)
    P = S.problem()
    tp_clear(P)
    assert np.array_equal(S.problem().rhs(0.0, _EQ_Y), r_ref)   # the system's source is intact
    assert np.array_equal(P.rhs(0.0, _EQ_Y), r_ref)              # and the copy still reaches it through S
    n = len(calls)
    Q = S.problem()
    tp_clear(S)
    with pytest.raises(RuntimeError, match="released by the garbage collector"):
        Q.rhs(0.0, _EQ_Y)
    assert len(calls) == n


# ------------------------------------------------------------------------------------------------- time core
def test_esdirk32_tableau_order_conditions():
    A, b, bh, c = (np.asarray(K.ESDIRK32.A), np.asarray(K.ESDIRK32.b), np.asarray(K.ESDIRK32.bhat),
                   np.asarray(K.ESDIRK32.c))
    assert np.allclose(A.sum(1), c, atol=1e-15)
    assert abs(b.sum() - 1) < 1e-15 and abs(b @ c - 0.5) < 1e-15
    assert abs(b @ c ** 2 - 1 / 3) < 1e-15 and abs(b @ A @ c - 1 / 6) < 1e-15
    assert abs(bh.sum() - 1) < 1e-15 and abs(bh @ c - 0.5) < 1e-14
    assert np.allclose(A[-1], b)                                   # stiffly accurate
    z = -1e8
    R = 1 + z * b @ np.linalg.solve(np.eye(4) - z * A, np.ones(4))
    assert abs(R) < 1e-7                                           # L-stable


def test_bdf_coefficients_variable_step():
    a = K._C.bdf_coefficients([1.0, 0.5, 0.0])
    assert np.allclose(a, [3.0, -4.0, 1.0])                        # (3 y2 - 4 y1 + y0)/(2h), h = 0.5
    t = [1.0, 0.75, 0.25]                                          # step ratio 1/2
    a = K._C.bdf_coefficients(t)
    for p in range(3):                                             # exact on quadratics
        assert abs(sum(ai * tj ** p for ai, tj in zip(a, t)) - (p * t[0] ** (p - 1) if p else 0.0)) < 1e-13


def test_orders_on_a_linear_problem():
    P = K.batch(_a_to_b())
    ex = lambda t: np.array([2 * math.exp(-0.7 * t), 2 - 2 * math.exp(-0.7 * t)])
    for meth, p in (("bdf1", 1), ("bdf2", 2), ("onestep", 3), ("bdf3", 3), ("bdf4", 4)):
        e = []
        for dt in (0.1, 0.05, 0.025):
            g = K.uniform_grid(0.0, 2.0, dt)
            if meth == "onestep":
                tr = K.grid_onestep(P, g, ex(0))
            else:
                k = int(meth[3])
                tr = K.grid_bdf(P, k, g, [ex(t) for t in g[:k]])
            e.append(abs(tr.y[-1][0] - ex(2.0)[0]))
        assert abs(math.log2(e[1] / e[2]) - p) < 0.2, (meth, e)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_bdf_dense_output_order(k):
    """The BDF-k dense output (the step's degree-k interpolant) has value order k off the grid."""
    P = K.problem(2, lambda t, y: np.array([y[1], -y[0]]), lambda t, y: np.array([[0.0, 1.0], [-1.0, 0.0]]))
    ex = lambda t: np.array([math.cos(t), -math.sin(t)])
    tq = [0.33, 0.71, 1.13]
    e = []
    for dt in (0.05, 0.025, 0.0125):
        g = K.uniform_grid(0.0, 1.5, dt)
        tr = K.grid_bdf(P, k, g, [ex(t) for t in g[:k]], newton_rtol=1e-14)
        e.append(max(float(np.max(np.abs(tr.dense(t) - ex(t)))) for t in tq))
    assert abs(math.log2(e[1] / e[2]) - k) < 0.3, (k, e)


def test_vsvo_past_vector_reallocation():
    """Regression: a variable-order run through many trajectory reallocations (the step's y is a copy);
    valgrind/ASan of this run is clean."""
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    P = K.problem(2, lambda t, y: A @ y, lambda t, y: A)
    tr = K.adaptive(P, "vsvo", 0.0, [1.0, 0.0], [30.0], 1e-9, 1e-11)
    assert tr.stats["steps"] > 600
    assert np.max(np.abs(tr.y[-1] - [math.cos(30.0), -math.sin(30.0)])) < 1e-6


def test_adaptive_stop_time_dense_and_events():
    P = K.batch(_a_to_b())
    for meth in ("bdf2", "onestep", "vsvo"):
        tr = K.adaptive(P, meth, 0.0, [2.0, 0.0], [0.5, 2.0, 5.0], 1e-8, 1e-10,
                        events=[lambda t, y: y[0] - 0.5])
        for to in (0.5, 2.0, 5.0):
            assert to in list(tr.t)
        assert tr.stats["converged"] and tr.stats["relative_residual"] < 1e-8
        assert abs(tr.events[0][1] - math.log(4) / 0.7) < 1e-4
        assert abs(tr.dense(1.234)[0] - 2 * math.exp(-0.7 * 1.234)) < 1e-5


def test_dae_robertson_stiff():
    k1, k2, k3 = 0.04, 3e7, 1e4
    f = lambda t, y: np.array([-k1 * y[0] + k3 * y[1] * y[2], k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                               y.sum() - 1.0])
    J = lambda t, y: np.array([[-k1, k3 * y[2], k3 * y[1]], [k1, -k3 * y[2] - 2 * k2 * y[1], -k3 * y[1]],
                               [1.0, 1.0, 1.0]])
    P = K.problem(3, f, J, mass=[1.0, 1.0, 0.0])
    for meth in ("bdf2", "onestep", "vsvo"):
        tr = K.adaptive(P, meth, 0.0, [1.0, 0.0, 0.0], [1e-2, 1.0, 1e4], 1e-6, [1e-8, 1e-14, 1e-8], h0=1e-8)
        assert abs(tr.y[-1].sum() - 1) < 1e-12
        assert tr.stats["converged"]


# --------------------------------------------------------------------------- the kinetics cases (group C1*), judged
def _case_defs():
    return {p.stem: json.loads(p.read_text()) for p in REG.glob("chem_*.json")
            if json.loads(p.read_text()).get("capability", "").startswith("C1")}


def _have_case_data():
    return (CASES / "c1_kinetics_check.py").exists() and (ROOT / "benchmarks/chem/casedata/c1/c1_fields.json").exists()


@pytest.mark.skipif(not _have_case_data(), reason="kinetics check module or case data missing")
@pytest.mark.parametrize("cid", ["chem_kernel_rates", "chem_kernel_units", "chem_jacobian_taylor",
                                 "chem_time_lstability", "chem_time_estimator", "chem_analytic_dae_equilibrium",
                                 "chem_conservation_network", "chem_stiff_robertson_modified",
                                 "chem_vo_stiff_robertson_dae_modified", "chem_vo_rtol_sweep"])
def test_kinetics_case_judged(cid):
    sys.path.insert(0, str(CASES))
    sys.path.insert(0, str(ROOT / "benchmarks/fluids"))
    try:
        R = importlib.import_module("c1_kinetics")
        H = importlib.import_module("acceptance")
    finally:
        sys.path.remove(str(CASES))
        sys.path.remove(str(ROOT / "benchmarks/fluids"))
    case = _case_defs()[cid]
    res = getattr(R, case["run"]["runner"]["function"])(json.loads(json.dumps(case["run"]["parameters"])), None, False)
    ev = H.evaluate(case, json.loads(json.dumps(res, default=float)))
    assert ev["criteria_met"], ev["headline"]
