"""Shape sensitivities and gradient-based optimization of axisymmetric designs.

``Results.sensitivities`` and ``Design.gradient`` return the derivatives of the figures of
merit with respect to named design parameters, computed by a discrete adjoint on a morphed
mesh (see docs/optimization.md and cpp/include/axionfem/adjoint.hpp).  :func:`optimize`
drives a design to a local optimum with those gradients.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import _core


def objectives() -> dict:
    """The differentiable figures of merit with a one-line meaning each.

    A cost or requirement quantity may name a benchmark, e.g. ``"cost:optimistic"``.
    """
    return json.loads(_core.objectives())


def _sensitivities(model, params, objs, method, step, steps, benchmark, workdir, order) -> dict:
    return json.loads(model.sensitivities(params=list(params or []), objectives=list(objs or []), method=method,
                                          step=float(step), steps=json.dumps(steps or {}), benchmark=benchmark or "",
                                          workdir=str(workdir), order=int(order)))


# ---------------------------------------------------------------- optimization
class _Cache:
    """One solve + one gradient per distinct parameter vector, shared by f and jac."""

    def __init__(self, design, objective: str, params: Sequence[str], quantities: Sequence[str],
                 workdir: Path, order: int, step: float, method: str, benchmark: str, verbose: int,
                 sgn: float = 1.0):
        from .design import Design

        self.base = Design(design)
        self.objective = objective
        self.sgn = sgn
        self.params = list(params)
        self.quantities = list(dict.fromkeys([objective, *quantities]))
        self.workdir = Path(workdir)
        self.order = order
        self.step = step
        self.method = method
        self.benchmark = benchmark
        self.verbose = verbose
        self.key: tuple | None = None
        self.values: dict = {}
        self.grad: dict = {}
        self.history: list[dict] = []
        self.best: dict | None = None
        self.n_solve = 0

    def at(self, x: Sequence[float]) -> tuple[dict, dict]:
        key = tuple(float(v) for v in x)
        if key == self.key:
            return self.values, self.grad
        d = self.base.with_params(**{k: float(v) for k, v in zip(self.params, key)})
        rec: dict = {"parameters": dict(zip(self.params, key))}
        try:
            res = d.solve(workdir=self.workdir / "current", verbose=0)
            s = _sensitivities(res._model, self.params, self.quantities, self.method, self.step, None,
                              self.benchmark, self.workdir / "current", self.order)
            self.values = dict(s["values"])
            self.grad = {q: dict(s["gradient"][q]) for q in self.quantities}
            self.n_solve += 1
            rec["values"] = self.values
            rec["caps"] = _caps(res, self.benchmark)
            v = self.values[self.objective]
            if math.isfinite(v) and (self.best is None
                                     or self.sgn * v < self.sgn * self.best["values"][self.objective]):
                self.best = {"parameters": rec["parameters"], "values": self.values, "caps": rec["caps"]}
        except Exception as e:  # noqa: BLE001  (an infeasible shape must not stop the search)
            rec["error"] = str(e)
            self.values = {q: math.inf for q in self.quantities}
            self.grad = {q: {p: 0.0 for p in self.params} for q in self.quantities}
        self.key = key
        self.history.append(rec)
        if self.verbose:
            print(f"  [{len(self.history):3d}] " + "  ".join(f"{k}={v:.6g}" for k, v in rec["parameters"].items()) +
                  (f"  {self.objective}={rec['values'][self.objective]:.8g}" if "values" in rec else
                   f"  failed: {rec['error'][:60]}"))
        return self.values, self.grad


def _caps(res, benchmark: str) -> dict:
    """Field-cap status at the requirement (cost.sc_wall_cap / cost.B_peak_cap)."""
    try:
        b = res.benchmark(benchmark) if benchmark else res.data["benchmarks"][0]
    except (KeyError, IndexError):
        return {}
    return b.get("caps_at_req", {})


def optimize(design, objective: str = "cost", params: dict[str, tuple[float, float]] | None = None,
             bounds: dict[str, tuple[float, float]] | None = None, constraints: Iterable = (),
             sense: str = "min", method: str | None = None, max_iter: int = 30, tol: float = 1e-6,
             step: float = 1e-4, gradient: str = "adjoint", benchmark: str = "", workdir: str | Path | None = None,
             order: int = 0, verbose: int = 1, callback: Callable | None = None) -> dict:
    """Minimize (or maximize) a figure of merit over design parameters, using adjoint gradients.

    ``params`` (or ``bounds``) maps parameter names to ``(low, high)``; the search runs in units of
    the starting values, so all parameters are O(1).  ``constraints`` holds entries
    ``{"quantity": "F", "type": ">=", "value": 1e-14}`` (or tuples ``("F", ">=", 1e-14)``) over any
    differentiable quantity (:func:`objectives`); with constraints the optimizer is SLSQP, without
    them L-BFGS-B.  ``sense`` is "min" or "max".

    Field caps (``cost.sc_wall_cap``, ``cost.B_peak_cap``) are reported for every iterate under
    ``caps`` and for the optimum; to enforce one, add a constraint on ``sc_surface_at_req`` or
    ``B_peak_at_req`` (their derivatives come from finite differences of a maximum, so they are
    noisier than the smooth quantities).

    Returns a dict with the optimum (parameters, all requested quantities, field caps), the history
    of every iterate and the optimized design document under ``design``.
    """
    import numpy as np
    from scipy.optimize import minimize

    from .design import Design

    base = Design(design)
    given = params if params is not None else bounds
    if not isinstance(given, dict) or not given:
        raise ValueError("give params={'name': (low, high), ...} with at least one parameter and its bounds")
    bounds = {k: tuple(v) for k, v in given.items()}
    for k, v in bounds.items():
        if len(v) != 2 or not v[0] < v[1]:
            raise ValueError(f"bounds for '{k}' must be (low, high) with low < high")
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    cons = []
    for c in constraints:
        if isinstance(c, (tuple, list)):
            c = {"quantity": c[0], "type": c[1], "value": c[2]}
        if c["type"] not in ("<=", ">="):
            raise ValueError("constraint type must be '<=' or '>='")
        cons.append(dict(c))
    names = list(bounds)
    p0 = base.resolved_params()
    missing = [n for n in names if n not in p0]
    if missing:
        raise KeyError(f"unknown parameters {missing}; known: {sorted(p0)}")
    x0 = np.array([p0[n] for n in names], float)
    if np.any(x0 == 0):
        raise ValueError("parameters that start at zero cannot be scaled; give a nonzero starting value")
    lo = np.array([bounds[n][0] for n in names], float) / x0
    hi = np.array([bounds[n][1] for n in names], float) / x0
    wd = Path(workdir) if workdir else Path("runs") / f"{base.name}__optimize"
    sgn = 1.0 if sense == "min" else -1.0
    cache = _Cache(base, objective, names, [c["quantity"] for c in cons], wd, order, step, gradient, benchmark,
                   verbose, sgn)

    v0, _ = cache.at(x0)
    if not math.isfinite(v0[objective]):
        raise RuntimeError(f"the starting design could not be evaluated: {cache.history[-1].get('error')}")
    scale = abs(v0[objective]) or 1.0

    def unscale(u):
        return np.asarray(u, float) * x0

    def f(u):
        val, _ = cache.at(unscale(u))
        return sgn * val[objective] / scale if math.isfinite(val[objective]) else 1e30

    def jac(u):
        _, g = cache.at(unscale(u))
        return np.array([sgn * g[objective][n] * x0[i] / scale for i, n in enumerate(names)])

    def con_fun(c):
        s = 1.0 if c["type"] == ">=" else -1.0
        cs = abs(c["value"]) or 1.0

        def g_(u):
            val, _ = cache.at(unscale(u))
            v = val[c["quantity"]]
            return s * (v - c["value"]) / cs if math.isfinite(v) else -1e6  # an unsolvable shape is infeasible

        def dg(u):
            _, gr = cache.at(unscale(u))
            return np.array([s * gr[c["quantity"]][n] * x0[i] / cs for i, n in enumerate(names)])

        return {"type": "ineq", "fun": g_, "jac": dg}

    algo = method or ("SLSQP" if cons else "L-BFGS-B")
    if verbose:
        print(f"optimize: {sense} {objective} over {names} with {algo}"
              + (f" and {len(cons)} constraint(s)" if cons else ""))
    r = minimize(f, np.ones(len(names)), jac=jac, method=algo, bounds=list(zip(lo, hi)),
                 constraints=[con_fun(c) for c in cons] if cons else (),
                 options={"maxiter": max_iter, "ftol": tol} if algo in ("SLSQP", "L-BFGS-B") else {"maxiter": max_iter},
                 callback=callback)
    xopt = unscale(r.x)
    vals, _ = cache.at(xopt)
    best = {n: float(v) for n, v in zip(names, xopt)}
    d_opt = base.with_params(**best)
    out = {"ok": bool(np.isfinite(vals[objective])),
           "objective": objective, "sense": sense, "value": vals[objective],
           "start_value": v0[objective], "parameters": best, "start_parameters": {n: p0[n] for n in names},
           "values": vals, "caps": cache.history[-1].get("caps", {}),
           "constraints": [{**c, "value_at_optimum": vals.get(c["quantity"])} for c in cons],
           "iterations": int(getattr(r, "nit", 0)), "solves": cache.n_solve, "converged": bool(r.success),
           "message": str(r.message), "design": d_opt.doc,
           "history": [{k: v for k, v in h.items() if k != "results"} for h in cache.history]}
    if cache.best is not None and sgn * cache.best["values"][objective] < sgn * vals[objective]:
        out["note"] = "the last iterate is worse than the best one seen; 'best_seen' holds the better design"
        out["best_seen"] = {"parameters": cache.best["parameters"], "values": cache.best["values"]}
    return out
