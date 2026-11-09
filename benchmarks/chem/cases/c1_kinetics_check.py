"""Kinetics / time-integration check module: the error computations of the C1 cases (`runner.check_module`).

Pure functions of the solver's RAW output (time grid, states, step log, rates and Jacobians returned by the public
API) and of exact/reference data; no solver calls, no verdicts (the judge decides).  Lint: this module does not
use the words the case-module lint forbids.

Contract with the runner (runner side):
  sol = {"t": [...], "y": [[...]...], "names": [...], "dt": float|None, "dense": callable|None,
         "solver": {"converged": bool, "relative_residual": float, "newton_iters": int, "steps": int,
                    "rejected": int, "jac_evals": int}}
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------------------------------------------
def component_scales(exact, t_grid, comps=None):
    """S_i = max_t |exact_i(t)| on the given grid (fixed with the case parameters; reported, not recomputed by the
    solver)."""
    Y = np.array([exact(t) for t in t_grid], float)
    S = np.max(np.abs(Y), axis=0)
    return S if comps is None else S[comps]


def max_norm_error(t, Y, exact, scales, comps=None):
    """max over grid points n and components i of |y_i^n - y_i(t_n)| / S_i  (normalised discrete L-infinity in time).
    The exact solution is evaluated at the solver's own time points; nothing is interpolated."""
    Y = np.asarray(Y, float)
    E = np.array([Y[n] - exact(tn) for n, tn in enumerate(t)])
    if comps is not None:
        E = E[:, comps]
    return float(np.max(np.abs(E) / np.asarray(scales)))


def end_error_signed(t, Y, exact, comp, scale):
    return float((np.asarray(Y)[-1][comp] - exact(t[-1])[comp]) / scale)


def order_series(levels, errors, magnitude=1.0):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


def run_entry(level, dt, solver, seconds=None):
    return {"level": level, "dt": float(dt) if dt is not None else None, "mesh": {}, "solver": dict(solver),
            "seconds": seconds}


def numpy_function(codes, n=None):
    """Build f(t[, y]) from the numpy code strings of benchmarks/chem/casedata/c1/c1_fields.json (no SymPy needed to run a case).
    The strings are evaluated with numpy as the only name; no builtins."""
    env = {"__builtins__": {}, "numpy": np}
    if n is None:
        fs = [eval("lambda t: " + c, env) for c in codes]          # noqa: S307 (expressions from the case data files)
        return lambda t: np.array([float(f(t)) for f in fs])
    args = ", ".join(f"y{i}" for i in range(n))
    fs = [eval(f"lambda t, {args}: " + c, env) for c in codes]       # noqa: S307
    return lambda t, y: np.array([float(f(t, *y)) for f in fs])


def final_time_error(Y_end, y_exact_end, scales):
    """max_i |y_i(T) - y_i,exact(T)| / S_i at the final time (insensitive to the starter's h^3 transient)."""
    return float(np.max(np.abs(np.asarray(Y_end, float) - np.asarray(y_exact_end, float)) / np.asarray(scales, float)))


def pfr_conversion(Vr, v0, k):
    """X(V_r) of the isothermal PFR A -> 2B, pure-A ideal-gas feed: root of (v0/k)(2 ln(1/(1-X)) - X) = V_r (Brent;
    the left side is increasing in X on [0, 1))."""
    from scipy.optimize import brentq
    if Vr <= 0:
        return 0.0
    g = lambda X: (v0 / k) * (2.0 * math.log(1.0 / (1.0 - X)) - X) - Vr
    return float(brentq(g, 0.0, 1.0 - 1e-15, xtol=1e-15, rtol=4 * np.finfo(float).eps))


def linear_cycle_exact(rate_matrix, c0, t):
    """c(t) = expm(K t) c0 (scipy.linalg.expm; K and c0 from c1_fields.json)."""
    from scipy.linalg import expm
    return expm(np.asarray(rate_matrix, float) * t) @ np.asarray(c0, float)


def lower_member_error(y_new, est, y_exact_new, advances_higher):
    """True local error of the LOWER-order member of an embedded pair, from the solver's (y_new, est) with the fixed
    sign convention est = (lower) - (higher)."""
    e_adv = np.asarray(y_new, float) - np.asarray(y_exact_new, float)
    return e_adv + np.asarray(est, float) if advances_higher else e_adv


# ---------------------------------------------------------------------------------------------------------------
# reference comparisons (fixed times; dense output or exact hits)
# ---------------------------------------------------------------------------------------------------------------
def rel_errors_at(t_ref, Y_ref, Y_sol, floors):
    """|y - y_ref| / max(|y_ref|, floor_i) per fixed time and component; the solver must return values AT t_ref
    (output times or dense output; the runner states which)."""
    Y_ref, Y_sol = np.asarray(Y_ref, float), np.asarray(Y_sol, float)
    return np.abs(Y_sol - Y_ref) / np.maximum(np.abs(Y_ref), np.asarray(floors, float))


def peak_from_dense(dense, comp, t_lo, t_hi, tol=1e-12):
    """Location and value of the maximum of component comp of the solver's dense output on [t_lo, t_hi] by golden
    section (the dense output must be unimodal there; the check module asserts the bracket is interior)."""
    g = (math.sqrt(5) - 1) / 2
    a, b = t_lo, t_hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = dense(c)[comp], dense(d)[comp]
    while b - a > tol * max(1.0, abs(b)):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = dense(c)[comp]
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = dense(d)[comp]
    tm = 0.5 * (a + b)
    if not (t_lo < tm < t_hi) or min(tm - t_lo, t_hi - tm) < 1e-6 * (t_hi - t_lo):
        raise ValueError("peak at the bracket edge: the dense output is not unimodal on the bracket")
    return tm, float(dense(tm)[comp])


def extremum_by_rhs_root(dense, rhs, comp, t_lo, t_hi, n_scan=400):
    """Location of an interior extremum of component comp as the ROOT of rhs(t, y_dense(t))[comp] (like the event checks; uses the right-hand side, not the interpolant's slope, so a dense output that is accurate in value but not in derivative is not penalised).  Scans for the sign change, then Brent; returns
    (t_ext, y_dense(t_ext)[comp])."""
    from scipy.optimize import brentq
    ts = np.linspace(t_lo, t_hi, n_scan + 1)
    g = [float(rhs(t, dense(t))[comp]) for t in ts]
    for a, b, ga, gb in zip(ts[:-1], ts[1:], g[:-1], g[1:]):
        if ga == 0.0:
            return float(a), float(dense(a)[comp])
        if ga * gb < 0:
            tr = brentq(lambda s: float(rhs(s, dense(s))[comp]), a, b, xtol=1e-15, rtol=4 * np.finfo(float).eps)
            return float(tr), float(dense(tr)[comp])
    raise ValueError("no sign change of the right-hand side component on the bracket")


# ---------------------------------------------------------------------------------------------------------------
# prescribed grids and output policy
# ---------------------------------------------------------------------------------------------------------------
def cyclic_grid_times(t_end, n_steps, step_pattern):
    """The fixed cyclic step-ratio grid: steps proportional to step_pattern repeated n_steps/len(pattern) times,
    scaled to sum to t_end; returns t_0 = 0, ..., t_N (N + 1 times)."""
    pat = [float(x) for x in step_pattern]
    if n_steps % len(pat):
        raise ValueError("n_steps is not a multiple of the pattern length")
    p = np.array(pat * (n_steps // len(pat)), float)
    h = p / p.sum() * float(t_end)
    return np.concatenate([[0.0], np.cumsum(h)])


def grid_adherence(t_steps, t_end, n_steps, step_pattern):
    """Did the run follow the fixed grid?  t_steps: the solver's accepted step-end times INCLUDING t_0 (what the
    integrator actually stepped to, not output/interpolation times).  Returns
      grid_dev     = max_n |t_n - t_n^grid| / t_end   (inf if the number of steps differs from n_steps),
      dt_max_ratio = max_n h_n / h_{n-1},  h_ratio_max_min = max h / min h,  steps = len(t_steps) - 1."""
    t = np.asarray(t_steps, float)
    g = cyclic_grid_times(t_end, n_steps, step_pattern)
    h = np.diff(t)
    out = {"steps": int(len(t) - 1),
           "dt_max_ratio": float(np.max(h[1:] / h[:-1])) if len(h) > 1 else float("nan"),
           "h_ratio_max_min": float(h.max() / h.min()) if len(h) else float("nan")}
    out["grid_dev"] = float(np.max(np.abs(t - g)) / float(t_end)) if len(t) == len(g) else float("inf")
    return out


def grid_adherence_series(level_results):
    """Aggregate grid_adherence() over the levels of one series (the quantities the fixed checks read):
      <series>_grid_dev_max     = np.max of the levels' grid_dev,
      <series>_dt_max_ratio_min = np.min of the levels' dt_max_ratio.
    np.max/np.min PROPAGATE NaN (Python's max/min silently drop a NaN that is not first), so a non-finite level makes
    the aggregate non-finite and the check FAIL; the runner must not aggregate by hand (nit 1)."""
    if not level_results:
        return {"grid_dev_max": float("inf"), "dt_max_ratio_min": float("nan")}
    return {"grid_dev_max": float(np.max([float(r["grid_dev"]) for r in level_results])),
            "dt_max_ratio_min": float(np.min([float(r["dt_max_ratio"]) for r in level_results]))}


def outputs_at_step_ends(t_steps, t_out, rel=8 * np.finfo(float).eps):
    """1.0 if every fixed output time is an accepted step end (stop-time mode: the solver stepped exactly to it),
    else 0.0.  Interpolated outputs (dense output between step ends) give 0.0."""
    ts = np.asarray(t_steps, float)
    for to in np.atleast_1d(np.asarray(t_out, float)):
        if not np.any(np.abs(ts - to) <= rel * max(1.0, abs(to))):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------------------------------------------
# invariants, positivity
# ---------------------------------------------------------------------------------------------------------------
def invariant_drift(Y, W, scale=None):
    """max_n |W (y^n - y^0)| / scale  for the linear invariants W (rows); scale = max |W y^0| unless given."""
    Y = np.asarray(Y, float)
    W = np.atleast_2d(np.asarray(W, float))
    v = Y @ W.T
    s = scale if scale is not None else max(float(np.max(np.abs(v[0]))), 1e-300)
    return float(np.max(np.abs(v - v[0])) / s)


def min_scaled(Y, atol):
    """min over n, i of y_i^n / atol_i  (positivity: must be >= -POS_ATOL_FACTOR)."""
    return float(np.min(np.asarray(Y, float) / np.asarray(atol, float)))


# ---------------------------------------------------------------------------------------------------------------
# error-estimator effectivity, tolerance studies
# ---------------------------------------------------------------------------------------------------------------
def estimate_ratio(est, true_local_error):
    """(d . e)/(e . e): projection of the solver's local-error estimate on the true local error (1 for an
    asymptotically correct estimator; sign errors give negative values)."""
    d, e = np.asarray(est, float), np.asarray(true_local_error, float)
    return float(d @ e / (e @ e))


def loglog_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


# ---------------------------------------------------------------------------------------------------------------
# Jacobian Taylor test
# ---------------------------------------------------------------------------------------------------------------
def taylor_remainders(F, J, x0, v, eps_list):
    """r(eps) = ||F(x0 + eps v) - F(x0) - eps J(x0) v||_2 and the fitted slope over eps_list (2 for a correct
    Jacobian, 1 for a wrong one).  F and J are the solver's own rate function and Jacobian."""
    F0 = np.asarray(F(x0), float)
    Jv = np.asarray(J(x0), float) @ v
    r = [float(np.linalg.norm(np.asarray(F(x0 + e * v), float) - F0 - e * Jv)) for e in eps_list]
    return r, loglog_slope(eps_list, r)


# ---------------------------------------------------------------------------------------------------------------
# self-test (python -B c1_kinetics_check.py)
# ---------------------------------------------------------------------------------------------------------------
def selftest():
    rep = {}
    # 1. max_norm_error on a known perturbation
    ex = lambda t: np.array([math.exp(-t), 2.0])
    t = np.linspace(0, 1, 11)
    Y = np.array([ex(x) + np.array([1e-3 * x, 0]) for x in t])
    assert abs(max_norm_error(t, Y, ex, [1.0, 2.0]) - 1e-3) < 1e-15
    rep["max_norm_error"] = "ok"
    # 2. peak finder on a known function
    tm, v = peak_from_dense(lambda s: np.array([s * math.exp(-s)]), 0, 0.1, 5.0)
    assert abs(tm - 1.0) < 1e-8 and abs(v - math.exp(-1)) < 1e-14
    rep["peak_from_dense"] = "ok"
    # 3. invariant drift
    Y = np.array([[1, 0, 0], [0.5, 0.25, 0.25 + 1e-12]])
    assert abs(invariant_drift(Y, [[1, 1, 1]]) - 1e-12) < 1e-16
    rep["invariant_drift"] = "ok"
    # 4. Taylor test on F = sin, correct and wrong Jacobian
    F = lambda x: np.sin(x)
    J = lambda x: np.diag(np.cos(x))
    Jw = lambda x: np.diag(np.cos(x)) * 1.01
    x0, v = np.array([0.3, 1.1]), np.array([0.7, -0.4])
    _, s = taylor_remainders(F, J, x0, v, [1e-2, 1e-3, 1e-4, 1e-5])
    _, sw = taylor_remainders(F, Jw, x0, v, [1e-2, 1e-3, 1e-4, 1e-5])
    assert abs(s - 2) < 0.05 and abs(sw - 1) < 0.1
    rep["taylor"] = {"correct": s, "wrong_by_1pct": sw}
    # 5. estimate ratio
    assert estimate_ratio([2, 0], [1, 0]) == 2 and estimate_ratio([-1, 0], [1, 0]) == -1
    rep["estimate_ratio"] = "ok"
    # 6. slope
    assert abs(loglog_slope([1e-4, 1e-6, 1e-8], [3e-4 ** 0.8, 3e-6 ** 0.8, 3e-8 ** 0.8]) - 0.8) < 1e-12
    rep["loglog_slope"] = "ok"
    # 7. numpy_function round trip
    f = numpy_function(["numpy.sin(t) + y0*y1", "y1**2"], 2)
    assert np.allclose(f(0.5, [2.0, 3.0]), [math.sin(0.5) + 6.0, 9.0])
    rep["numpy_function"] = "ok"
    # 8. pfr_conversion inverts V_r(X)
    X = pfr_conversion(0.05, 0.05, 0.9)
    assert abs((0.05 / 0.9) * (2 * math.log(1 / (1 - X)) - X) - 0.05) < 1e-14
    rep["pfr_conversion"] = X
    # 9. linear_cycle_exact: conserves the sum for a column-sum-zero K
    K = [[-1.0, 0.5], [1.0, -0.5]]
    assert abs(linear_cycle_exact(K, [1.0, 0.0], 0.7).sum() - 1.0) < 1e-14
    rep["linear_cycle_exact"] = "ok"
    # 10. lower_member_error sign convention
    assert np.allclose(lower_member_error([1.0], [0.1], [0.95], True), [0.15])
    rep["lower_member_error"] = "ok"
    # 11. extremum_by_rhs_root on y = t e^{-t}: y' = (1 - t) e^{-t}
    tm, v = extremum_by_rhs_root(lambda s: np.array([s * math.exp(-s)]), lambda s, y: np.array([(1 - s) * math.exp(-s)]),
                                 0, 0.1, 5.0)
    assert abs(tm - 1.0) < 1e-12 and abs(v - math.exp(-1)) < 1e-14
    rep["extremum_by_rhs_root"] = "ok"
    # 12. grid_adherence: the fixed grid itself passes; a uniform grid of the same N does not
    g = cyclic_grid_times(2.0, 20, [1.0, 1.25, 0.8, 1.1, 0.9])
    a = grid_adherence(g, 2.0, 20, [1.0, 1.25, 0.8, 1.1, 0.9])
    assert a["grid_dev"] == 0.0 and abs(a["dt_max_ratio"] - 1.375) < 1e-12 and abs(a["h_ratio_max_min"] - 1.5625) < 1e-12
    u = grid_adherence(np.linspace(0, 2.0, 21), 2.0, 20, [1.0, 1.25, 0.8, 1.1, 0.9])
    assert u["grid_dev"] > 1e-3 and abs(u["dt_max_ratio"] - 1.0) < 1e-12
    assert grid_adherence(g[:-1], 2.0, 20, [1.0, 1.25, 0.8, 1.1, 0.9])["grid_dev"] == float("inf")
    rep["grid_adherence"] = "ok"
    # 12b. grid_adherence_series propagates NaN (Python max would drop it)
    agg = grid_adherence_series([a, {"grid_dev": float("nan"), "dt_max_ratio": float("nan")}])
    assert math.isnan(agg["grid_dev_max"]) and math.isnan(agg["dt_max_ratio_min"])
    agg = grid_adherence_series([a, u])
    assert agg["grid_dev_max"] == u["grid_dev"] and agg["dt_max_ratio_min"] == u["dt_max_ratio"]
    rep["grid_adherence_series"] = "ok"
    # 13. outputs_at_step_ends
    assert outputs_at_step_ends([0, 0.5, 1.0, 2.0], [1.0, 2.0]) == 1.0 and outputs_at_step_ends([0, 0.5, 2.0], [1.0]) == 0.0
    rep["outputs_at_step_ends"] = "ok"
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
