"""Stiff chemical kinetics (beta): the kinetics kernel, 0D reactors and equilibrium, and the time core `axf::time`
(BDF1..5, ESDIRK3(2)4L[2]SA, PI control, index-1 DAE, dense output, events, variable-order BDF1..5).  The numerics
are C++ (cpp/src/chem/{kinetics,reactors,time_core}.cpp, ``_core.chem``); this module converts inputs and documents
the API.  Formulation and conventions: docs/guides/chemistry.md; the methods' properties are in ``DECLARATIONS``.

    from axionfem.chem import kinetics as K
    mech = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": 0.7}]})
    P = K.batch(mech)
    tr = K.adaptive(P, "onestep", 0.0, [2.0, 0.0], t_out=[1.0, 5.0], rtol=1e-8, atol=1e-10)
    tr.t, tr.y, tr.dense(0.3), tr.stats

Mechanism dictionaries: {"species": [...], "reactions": [{"reactants": {sp: nu'}, "products": {sp: nu''}, "A": .., "n":
.., "Ea": .., "reversible": bool, "Kc": .. | "kr": {"A", "n", "Ea"}, "orders": {sp: order}, "dH": ..}],
"units": {"concentration": "mol/m^3" | "mol/L" | "mol/cm^3", "time": "s" | "min", "energy": "J/mol" | "cal/mol" |
"kcal/mol" | "kJ/mol"}} (SI when omitted).  Every parameter is converted to SI by the kernel.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .. import _core

_C = _core.chem

GAS_CONSTANT = _C.GAS_CONSTANT_EXACT          # 8.31446261815324 J/(mol K), exact
ESDIRK32 = _C.tableau_esdirk32()              # ESDIRK3(2)4L[2]SA (Kennedy & Carpenter 2003)
THETA_HALF = _C.tableau_theta_half()          # theta = 1/2 (trapezoid) through the same one-step driver
NAN = float("nan")

# What the integrators do: method orders, error estimates, starters, dense output, norms (checked by the tests)
DECLARATIONS = {
    "onestep": {"method": "ESDIRK3(2)4L[2]SA", "p": 3, "stage_order_q": 2, "embedded_order": 2,
                "stiffly_accurate": True, "L_stable": True, "advances": "higher (order 3)",
                "estimate": "est = (embedded, order 2) - (advanced, order 3)", "estimator_limit": 1.0,
                "dense_output": "cubic Hermite on (y_n, y'_n, y_{n+1}, y'_{n+1}): value order 3; derivative order 2 "
                                "(not relied on)"},
    "bdf": {"form": "variable-coefficient (VC): Lagrange derivative on the actual past times",
            "estimate": "d = E (y_{n+1} - y^p), y^p the degree-k interpolant through t_n..t_{n-k}, "
                        "E = (1/alpha_0)/(1/alpha_0 + t_{n+1} - t_{n-k}); estimator limit 1",
            "starter": "BDF2: one BDF1 step (starting error O(h^2)); BDF3..5: ESDIRK3(2) with one Richardson "
                       "extrapolation per starting step (local error O(h^5)); exact-history start when given",
            "dense_output": "the step's interpolating polynomial of degree k (value order k)"},
    "error_norm": "weighted RMS over ALL components (differential and algebraic), w_i = atol_i + rtol max(|y_n|, |y_n+1|)",
    "positivity": "no clipping (not conservative); small negative excursions are left to the error control",
    "equilibrium": "reaction-invariant form, index 1: W c' = W(S_k q + s), 0 = prod c^nu'' - K prod c^nu'",
    "noninteger_orders": "max(c, 0)^o, derivative o c^(o-1) for c > 0 else 0; not regularized",
    "adaptive": "stop-time outputs (every output time is an accepted step end); PI control (0.7, 0.4) for bdf2 and "
                "onestep, I control with a hold/deadband for vsvo; safety 0.9",
}


def _arr(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=float))


# ------------------------------------------------------------------------------------------------------ kernel
def mechanism(d: dict, allow_nonequilibrium_cycle: bool = False):
    """Build the kernel (_core.chem.Mechanism) from a mechanism dictionary (see the module docstring)."""
    species = list(d["species"])
    idx = {s: i for i, s in enumerate(species)}
    rx = []
    for j, r in enumerate(d["reactions"]):
        def ix(name, what):
            if name not in idx:
                raise ValueError(f"reaction {j}: {what} species {name!r} is not in species {species}")
            return idx[name]
        k = _C.KineticReaction()
        k.reac = [ix(s, "reactant") for s in (r.get("reactants") or {})]
        k.nu_r = [float(v) for v in (r.get("reactants") or {}).values()]
        k.prod = [ix(s, "product") for s in (r.get("products") or {})]
        k.nu_p = [float(v) for v in (r.get("products") or {}).values()]
        k.kf = (float(r["A"]), float(r.get("n", 0.0)), float(r.get("Ea", 0.0)))
        k.reversible = bool(r.get("reversible", False))
        if r.get("Kc") is not None:
            k.has_Kc, k.Kc = True, float(r["Kc"])
        if r.get("kr") is not None:
            kr = r["kr"]
            k.has_kr, k.kr = True, (float(kr["A"]), float(kr.get("n", 0.0)), float(kr.get("Ea", 0.0)))
        if r.get("orders"):
            k.order_idx = [ix(s, "order") for s in r["orders"]]
            k.order_val = [float(v) for v in r["orders"].values()]
        k.dH = float(r.get("dH", 0.0))
        rx.append(k)
    u = d.get("units") or {}
    return _C.Mechanism(len(species), rx, u.get("concentration", "mol/m^3"), u.get("time", "s"),
                        u.get("energy", "J/mol"), bool(allow_nonequilibrium_cycle))


def rates(mech, c, T: Optional[float] = None):
    """(omega, d omega/dc, d omega/dT) in SI; c in mol/m^3 (use mech.concentration_to_si for other units)."""
    return mech.rates(_arr(c), NAN if T is None else float(T))


# ----------------------------------------------------------------------------------------------------- problems
def _vecfun(fn, n=None):
    def g(*a):
        v = np.asarray(fn(*a), dtype=float).reshape(-1)
        return np.ascontiguousarray(v)
    return g


def _matfun(fn):
    def g(t, y):
        return np.ascontiguousarray(np.asarray(fn(t, y), dtype=float))
    return g


def problem(n: int, f: Callable, jac: Callable, mass=None, source: Optional[Callable] = None):
    """Generic M y' = f(t, y) + s(t) for the time core (mass: None = identity, or an n x n matrix, possibly
    singular / non-diagonal; a 1-D mass is taken as the diagonal)."""
    M = None
    if mass is not None:
        M = np.asarray(mass, dtype=float)
        if M.ndim == 1:
            M = np.diag(M)
        if M.shape == (n, n) and np.array_equal(M, np.eye(n)):    # exactly the identity
            M = None
    return _C.TimeProblem(int(n), _vecfun(f), _matfun(jac), None if M is None else np.ascontiguousarray(M),
                          None if source is None else _vecfun(source))


def _src(source):
    return None if source is None else _vecfun(source)


def batch(mech, T=None, source=None):
    """Constant-volume batch: c' = omega(c, T) + s(t)."""
    return _C.batch_problem(mech, NAN if T is None else float(T), _src(source))


def cstr(mech, tau, c_in, T=None, source=None):
    """CSTR (constant V, density): c' = (c_in - c)/tau + omega + s."""
    return _C.cstr_problem(mech, NAN if T is None else float(T), float(tau), _arr(c_in), _src(source))


def semibatch(mech, q, c_feed, T=None, source=None):
    """Semibatch, y = [c, V]: c' = (q/V)(c_f - c) + omega + s, V' = q."""
    return _C.semibatch_problem(mech, NAN if T is None else float(T), float(q), _arr(c_feed), _src(source))


def const_p_gas(mech, T, P, source=None):
    """Isothermal constant-pressure ideal gas in moles: N' = V omega(N/V) + s, V = sum N R T / P."""
    return _C.const_p_gas_problem(mech, float(T), float(P), _src(source))


def gas_volume(T, P, N):
    """Volume of the constant-P gas reactor's state N (what the reactor uses: V = sum N R T / P)."""
    return float(_C.const_p_gas_volume(float(T), float(P), _arr(N)))


def adiabatic_liquid(mech, rho_cp, source=None):
    """Liquid adiabatic batch, y = [c, T]: rho c_p T' = sum_j (-dH_j) q_j (constant V and rho c_p)."""
    return _C.adiabatic_liquid_problem(mech, float(rho_cp), _src(source))


def pfr(mech, T, v0, F_tot0):
    """Isothermal isobaric ideal-gas PFR in reactor volume: dF/dV_r = omega(F/v), v = v0 F_tot / F_tot0."""
    return _C.pfr_problem(mech, NAN if T is None else float(T), float(v0), float(F_tot0))


def equilibrium(kinetic: dict, equilibria: Sequence[dict], T=None, source=None):
    """Equilibrium reactions in the reaction-invariant form (index 1).  kinetic: a mechanism dictionary (its
    species list defines the state; it may have no reactions); equilibria: [{"reactants": {...}, "products": {...},
    "K": K}] with K in (mol/m^3)^{sum nu}.  Returns _core.chem.EquilibriumSystem (.problem(), .consistent_init(c0),
    .constraint(c), .W())."""
    species = list(kinetic["species"])
    idx = {s: i for i, s in enumerate(species)}
    mech = mechanism(kinetic) if kinetic.get("reactions") else mechanism(dict(kinetic, reactions=[]))
    eqs = []
    for e in equilibria:
        r = _C.EquilibriumReaction()
        r.reac = [idx[s] for s in e["reactants"]]
        r.nu_r = [float(v) for v in e["reactants"].values()]
        r.prod = [idx[s] for s in e["products"]]
        r.nu_p = [float(v) for v in e["products"].values()]
        r.K = float(e["K"])
        eqs.append(r)
    return _C.EquilibriumSystem(mech, eqs, NAN if T is None else float(T), _src(source))


# -------------------------------------------------------------------------------------------------- integrators
def bdf_step(P, t_hist, y_hist, h, order, newton_rtol=1e-12, newton_max_iters=50, yp=None):
    """Single BDF step: histories in CHRONOLOGICAL order (oldest first, newest last); returns
    (y_{n+1}, est, newton) with est the raw local-error estimate of y_{n+1}."""
    th = [float(t) for t in t_hist][::-1]
    yh = [_arr(y) for y in y_hist][::-1]
    d = _C.bdf_advance(P, th, yh, float(h), int(order), float(newton_rtol), int(newton_max_iters),
                       None if yp is None else _arr(yp))
    if not d["ok"]:
        raise RuntimeError(f"bdf_step: Newton failed ({d['newton']})")
    return np.asarray(d["y"]), np.asarray(d["est"]), d["newton"]


def onestep_step(P, t, y, h, tableau=None, yp=None, newton_rtol=1e-12, newton_max_iters=50, full=False):
    """Single one-step (ESDIRK) step: returns (y_{n+1}, est, advances_higher, newton) with
    est = (lower) - (higher) as the controller uses it.  full=True returns the core's dict, which also holds the
    embedded (lower-order) solution itself ('y_embedded')."""
    tb = ESDIRK32 if tableau is None else tableau
    y = _arr(y)
    ypv = _C.consistent_derivative(P, float(t), y) if yp is None else _arr(yp)
    d = _C.rk_advance(P, tb, float(t), y, ypv, float(h), float(newton_rtol), int(newton_max_iters))
    if not d["ok"]:
        raise RuntimeError(f"onestep_step: Newton failed ({d['newton']})")
    if full:
        return d
    return np.asarray(d["y"]), np.asarray(d["est"]), bool(d["advances_higher"]), d["newton"]


def grid_bdf(P, order, times, y_start, newton_rtol=1e-12, newton_max_iters=50):
    """BDF-k on a prescribed grid (every accepted step end is a grid time).  y_start: the values at times[0..m-1]
    (m >= k: exact start; m = 1: self start with the declared starter)."""
    return _C.integrate_grid_bdf(P, int(order), [float(t) for t in times], [_arr(y) for y in y_start],
                                 float(newton_rtol), int(newton_max_iters))


def grid_onestep(P, times, y0, tableau=None, newton_rtol=1e-12, newton_max_iters=50):
    """One-step method (default ESDIRK3(2)4L[2]SA) on a prescribed grid."""
    return _C.integrate_grid_rk(P, ESDIRK32 if tableau is None else tableau, [float(t) for t in times], _arr(y0),
                                float(newton_rtol), int(newton_max_iters))


def uniform_grid(t0, t_end, dt):
    n = int(round((t_end - t0) / dt))
    if abs(n * dt - (t_end - t0)) > 1e-9 * max(1.0, abs(t_end)):
        raise ValueError(f"dt = {dt} does not divide [{t0}, {t_end}]")
    g = [t0 + i * dt for i in range(n + 1)]
    g[-1] = float(t_end)
    return g


def adaptive(P, method, t0, y0, t_out, rtol, atol, h0=0.0, max_order=5, events=(), tableau=None,
             newton_rtol=1e-10, newton_max_iters=25):
    """Adaptive run: method 'bdf2' (variable-step BDF2, BDF1 start), 'onestep' (ESDIRK3(2)) or 'vsvo' (own
    variable-order BDF1..max_order).  Stop-time outputs: every t_out is an accepted step end.  atol: scalar or per
    component.  events: callables g(t, y) whose sign changes are located on the dense output (tr.events)."""
    n = P.n
    a = np.full(n, float(atol)) if np.ndim(atol) == 0 else _arr(atol)
    evs = [(lambda t, y, g=g: float(g(t, np.asarray(y)))) for g in events]
    return _C.integrate_adaptive(P, str(method), float(t0), _arr(y0), [float(t) for t in t_out], float(rtol), a,
                                 float(h0), int(max_order), tableau, evs, float(newton_rtol), int(newton_max_iters),
                                 0.9, True)


def solver_entry(tr) -> dict:
    """The solver summary (convergence, residual, Newton and step counts) of a trajectory's statistics."""
    s = dict(tr.stats)
    n = max(int(s["solves"]), 1)
    return {"converged": bool(s["converged"]), "relative_residual": float(s["relative_residual"]),
            "iterations": int(s["newton_iters"]), "newton_iters": int(s["newton_iters"]),
            "newton_per_solve": float(s["newton_iters"]) / n, "steps": int(s["steps"]),
            "rejected": int(s["rejected"]), "newton_failures": int(s["newton_failures"]),
            "jac_evals": int(s["jac_evals"]), "stagnated_solves": int(s.get("stagnated", 0))}


__all__ = ["mechanism", "rates", "problem", "batch", "cstr", "semibatch", "const_p_gas", "adiabatic_liquid", "pfr",
           "gas_volume", "equilibrium", "bdf_step", "onestep_step", "grid_bdf", "grid_onestep", "uniform_grid", "adaptive",
           "solver_entry", "ESDIRK32", "THETA_HALF", "DECLARATIONS", "GAS_CONSTANT"]
