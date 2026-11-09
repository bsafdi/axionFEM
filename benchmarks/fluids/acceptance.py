"""Acceptance criteria of the fluids and chemistry verification cases.

``evaluate(case, result)`` judges the raw numbers a case runner returns (``{"runs", "series", "quantities"}``)
against the case definition's ``reference`` and ``acceptance`` blocks and returns one verdict per item plus a
summary (``criteria_met``, ``headline``).  Item kinds: ``convergence`` (observed orders of an error series under
refinement), ``functional_convergence`` (Richardson extrapolation and grid-convergence index of a functional),
``comparisons`` (a quantity against a reference value: closed form, published value or another code's result),
``checks`` (identities and bounds), ``solver_sensitivity`` / ``quadrature_sensitivity`` (a quantity barely moves when
the solver or the quadrature is tightened), ``tolerance_behaviour`` (adaptive integrators) and ``solver`` (every run
converged).  Each verdict column (``self_consistency``, ``exact_solution``, ``literature``, ``cross_code``) means only
what it says; a case that tests no literature quantity says nothing about agreement with the literature.

Pure Python (numpy); it never imports the compiled module.  The runner is ``benchmarks/fluids/run.py``; the case
layout is described in ``benchmarks/fluids/README.md``.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

EPS = 2.220446049250313e-16
IDENTITY_MAX_C = 1e3              # a round-off identity bound c * eps * scale with c <= 1e3
FEATURE_TYPES = ("peak", "crossing", "argmax_derivative")
MIN_REFINEMENT_RATIO = 1.3       # Celik et al. 2008
MAX_DOWNWARD_DRIFT = 0.1         # |p_i - p_expected| growing by more than this toward fine meshes: hidden other order
MIN_SPAN = 4.0                   # h_coarse / h_fine over the fitted levels
MIN_FLOOR_REL = 1e-13            # a round-off floor below this is no floor
SOLUTION_SCALE_FACTOR = 10.0     # solution_scale within this factor of the magnitude the case reports
GCI_ORDER_WINDOW_MAX = 0.5       # admissible observed order: formal +- at most this
GCI_MAX_FRACTION = 1.0           # the GCI term may add at most this fraction of the recipe's fixed allowance
MIN_LEVELS_ORDER = 4             # mms/analytic order studies (an item may set min_levels)
NESTED_MAX_ORDER_TOLERANCE = 0.15   # at 0.3 a q = p - 1/2 term can hide 64 % of the finest error
UNSTRUCTURED_MAX_ORDER_TOLERANCE = 0.3   # analytic and unstructured order items
VERDICT_OF_LEVEL = {"mms": "self_consistency", "conservation": "self_consistency", "analytic": "exact_solution",
                    "literature": "literature", "cross_code": "cross_code"}
VERDICT_KEYS = ("self_consistency", "exact_solution", "literature", "cross_code")
LEVELS = ("mms", "analytic", "literature", "conservation", "cross_code")


def get_path(d, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(dotted)
    return cur


def ref_allowance_fixed(tol: dict | None, scale: float) -> float:
    tol = tol or {}
    return float(tol.get("abs", 0.0)) + float(tol.get("rel", 0.0)) * abs(scale)


def comparison_tolerance(case: dict, item: dict) -> dict:
    """The tolerance recipe of a comparison: the reference block's tolerance_recipe[reference] (prescribed at
    extraction) if present, else the comparison's own tolerance (closed-form references)."""
    recipe = ((case.get("reference") or {}).get("tolerance_recipe") or {}).get(item.get("reference"))
    return recipe if recipe is not None else (item.get("tolerance") or {})


def pairwise_orders(x: Iterable[float], err: Iterable[float]) -> list[float]:
    """Orders between successive levels in the given order (coarse to fine): ln(e_i/e_{i+1}) / ln(x_i/x_{i+1})."""
    x, e = [float(v) for v in x], [abs(float(v)) for v in err]
    out = []
    for i in range(len(x) - 1):
        if e[i] <= 0 or e[i + 1] <= 0 or x[i] == x[i + 1]:
            out.append(float("nan"))
        else:
            out.append(math.log(e[i] / e[i + 1]) / math.log(x[i] / x[i + 1]))
    return out


def fit_order(x: Iterable[float], err: Iterable[float], last: int | None = None) -> dict:
    """Least-squares slope of ln|e| against ln x over the `last` finest levels (input coarse to fine)."""
    pts = list(zip(map(float, x), (abs(float(v)) for v in err)))
    if last:
        pts = pts[-last:]
    if len(pts) < 2 or any(p[1] <= 0 or p[0] <= 0 for p in pts):
        return {"order": float("nan"), "stderr": float("nan"), "n": len(pts)}
    xs, ys = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    se = float("nan")
    if len(xs) > 2:
        s2 = float(np.sum((ys - A @ coef) ** 2)) / (len(xs) - 2)
        se = math.sqrt(s2 / float(np.sum((xs - xs.mean()) ** 2)))
    return {"order": float(coef[0]), "stderr": se, "n": len(xs)}


def series_abscissa(spec: dict, levels: list, by_level: dict) -> tuple[list[float], list[float], list[str]]:
    """(nominal sequence, abscissa values, problems) from the runs that each series level names."""
    ab = spec.get("abscissa") or {}
    kind, dim = ab.get("kind"), ab.get("dim")
    nominal, xs, probs = [], [], []
    for lv in levels:
        run = by_level.get(lv)
        if run is None:
            probs.append(f"level '{lv}' has no run entry (every level must report its solve)")
            continue
        mesh = run.get("mesh") or {}
        nominal.append(mesh.get("h_nominal") if kind != "dt" else run.get("dt"))
        if kind in ("h_nominal", "h_max", "h_eff"):
            v = mesh.get(kind)
        elif kind == "dofs":
            d = mesh.get("dofs_field")
            v = float(d) ** (-1.0 / float(dim)) if d and dim else None
        elif kind == "dt":
            v = run.get("dt")
        else:
            probs.append(f"abscissa kind '{kind}' unknown")
            v = None
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            probs.append(f"level '{lv}': abscissa {kind} missing or not positive in its run entry")
        else:
            xs.append(float(v))
    return nominal, xs, probs


def mesh_family_problems(spec: dict, series: dict | None, by_level: dict) -> list[str]:
    """an MMS order study runs on a NESTED mesh family (uniform refinement of one coarse mesh: mesh
    noise far below 1 %) with order_tolerance <= 0.15, and every level's run reports the same mesh.family and
    consecutive mesh.refinement_level values.  Unstructured families are allowed for other levels only."""
    mf = spec.get("mesh_family") or {}
    fam = mf.get("kind")
    if (spec.get("abscissa") or {}).get("kind") == "dt":
        return []
    errs = []
    if fam == "layer_adapted" and mf.get("log_factor"):
        errs.append("a layer-adapted family with a log factor has no power-law order: use functional_convergence "
                    "(GCI) on integral quantities only")
    if spec.get("level") == "mms" and fam != "nested":
        errs.append("an MMS order study needs mesh_family.kind 'nested'")
    if (spec.get("level") == "mms" or fam == "nested") and float(spec["order_tolerance"]) > NESTED_MAX_ORDER_TOLERANCE:
        errs.append(f"order_tolerance {spec['order_tolerance']} > {NESTED_MAX_ORDER_TOLERANCE} on a nested family: "
                    "too coarse to see a q = p - 1/2 term")
    elif float(spec["order_tolerance"]) > UNSTRUCTURED_MAX_ORDER_TOLERANCE:
        errs.append(f"order_tolerance {spec['order_tolerance']} > {UNSTRUCTURED_MAX_ORDER_TOLERANCE}: at that width a "
                    "q = p - 1/2 term hides most of the finest error")
    if fam == "nested" and mf.get("dim") and isinstance(series, dict) and isinstance(series.get("levels"), list):
        # a nested family multiplies the element count by r^d per level (mislabelled h)
        runs = [by_level.get(lv) or {} for lv in series["levels"]]
        ns = [(r.get("mesh") or {}).get("n_elements") for r in runs]
        hs = [(r.get("mesh") or {}).get("h_nominal") for r in runs]
        if not all(isinstance(n, int) and n > 0 for n in ns) or not all(isinstance(h, (int, float)) and h > 0
                                                                          for h in hs):
            errs.append("nested family with a declared dim: every run reports mesh.n_elements and mesh.h_nominal")
        else:
            for i in range(len(ns) - 1):
                want = (hs[i] / hs[i + 1]) ** int(mf["dim"])
                if abs(ns[i + 1] / ns[i] - want) > 1e-6 * want:
                    errs.append(f"nested family: n_elements {ns[i]} -> {ns[i + 1]} is not (h ratio)^dim = {want:.4g} "
                                "times more: the h labels do not describe the meshes")
                    break
    if fam == "nested" and isinstance(series, dict) and isinstance(series.get("levels"), list):
        meshes = [((by_level.get(lv) or {}).get("mesh") or {}) for lv in series["levels"]]
        fams = {m.get("family") for m in meshes}
        lvls = [m.get("refinement_level") for m in meshes]
        if len(fams) != 1 or None in fams or "" in fams:
            errs.append(f"nested family: the runs report mesh.family {sorted(map(str, fams))}, not one family id")
        if not all(isinstance(v, int) for v in lvls) or any(b != a + 1 for a, b in zip(lvls, lvls[1:])):
            errs.append(f"nested family: mesh.refinement_level {lvls} are not consecutive integers coarse to fine")
    return errs


def assess_convergence(series: dict | None, spec: dict, by_level: dict, params: dict, quick: bool = False) -> dict:
    """Observed order against the prescribed expectation.

    PASS needs all of: the series levels are exactly the prescribed sequence (run.parameters[levels_from]) in order; every
    level's run reports the declared abscissa; refinement ratios >= min_refinement_ratio; >= min_levels; |errors|
    strictly decreasing and above the round-off floor floor_rel * solution_scale; signed errors keep their sign over
    the finest three; the least-squares order over the finest fit_points AND the finest pairwise order inside
    [expected - tol, expected + tol] (upper bound max_order if superconvergence was declared); the fit's standard
    error <= tol; the fitted levels span h_coarse/h_fine >= min_span (default 4); and the distance |p_i - expected|
    of the pairwise orders does not GROW toward fine meshes by more than max_downward_drift (finest distance minus
    the smallest earlier distance).  The trend test is two-sided: a lower-order term hides
    as falling orders when it has the same sign as the leading term and as RISING orders when it has the opposite
    sign; a healthy higher-order term makes the distance shrink, which passes.  Resolution (measured in the
    tests): a lower-order component of about 17-20 % of the finest error passes with four levels at r = 2.

    The round-off floor needs floor_rel >= 1e-13 and solution_scale within 10x of the magnitude the case reports
    as series.solution_magnitude; max_order <= expected + 1."""
    p_exp, tol = float(spec["expected_order"]), float(spec["order_tolerance"])
    out = {"series": spec["series"], "expected_order": p_exp, "order_tolerance": tol,
           "abscissa": spec.get("abscissa"), "reasons": []}
    R = out["reasons"]
    R += mesh_family_problems(spec, series, by_level)
    if not isinstance(series, dict):
        R.append("series missing from the case result")
        out["verdict"] = "FAIL"
        return out
    levels, err = series.get("levels"), series.get("error")
    if not isinstance(levels, list) or not isinstance(err, list) or len(levels) != len(err) or not levels:
        R.append("series needs equal-length non-empty 'levels' and 'error' lists")
        out["verdict"] = "FAIL"
        return out
    try:
        e_signed = [float(v) for v in err]
    except (TypeError, ValueError):
        R.append("errors are not numbers")
        out["verdict"] = "FAIL"
        return out
    nominal, x, probs = series_abscissa(spec, levels, by_level)
    R += probs
    try:
        prescribed = list(get_path(params, spec["levels_from"]))
    except KeyError:
        prescribed = None
        R.append(f"levels_from '{spec.get('levels_from')}' not found in the run parameters")
    if prescribed is not None:
        want = prescribed[:len(nominal)] if quick else prescribed
        if [float(v) for v in nominal if v is not None] != [float(v) for v in want]:
            R.append(f"series levels {nominal} are not the prescribed sequence {prescribed}")
    out.update({"x": x, "error": e_signed, "levels": levels})
    if probs or len(x) != len(e_signed):
        out["verdict"] = "FAIL"
        return out
    e = [abs(v) for v in e_signed]
    if any(not math.isfinite(v) for v in e):
        R.append("non-finite error")
    if any(b >= a for a, b in zip(x, x[1:])):
        R.append(f"abscissa {x} is not strictly decreasing coarse to fine")
    ratios = [a / b for a, b in zip(x, x[1:]) if b > 0]
    rmin = float(spec.get("min_refinement_ratio", MIN_REFINEMENT_RATIO))
    if any(r < rmin for r in ratios):
        R.append(f"refinement ratios {[round(r, 3) for r in ratios]} below {rmin} (Celik et al.)")
    nmin = int(spec.get("min_levels", MIN_LEVELS_ORDER))
    if len(e) < nmin:
        R.append(f"only {len(e)} levels, need {nmin}")
    if any(b >= a for a, b in zip(e, e[1:])):
        R.append("|error| does not decrease strictly under refinement")
    floor_rel, scale = float(spec["floor_rel"]), abs(float(spec["solution_scale"]))
    if floor_rel < MIN_FLOOR_REL:
        R.append(f"floor_rel {floor_rel:g} < {MIN_FLOOR_REL:g}: below double-precision round-off")
    mag = series.get("solution_magnitude")
    try:
        mag = float(mag)
    except (TypeError, ValueError):
        mag = None
    if mag is None or not math.isfinite(mag) or mag <= 0:
        R.append("the series reports no positive solution_magnitude (the scale the round-off floor is relative to)")
    elif not (mag / SOLUTION_SCALE_FACTOR <= scale <= mag * SOLUTION_SCALE_FACTOR):
        R.append(f"solution_scale {scale:g} is not within {SOLUTION_SCALE_FACTOR:g}x of the reported solution "
                 f"magnitude {mag:g}: the round-off floor would be misplaced")
    out["solution_magnitude"] = mag
    floor = floor_rel * scale
    out["floor"] = floor
    at_floor = [i for i, v in enumerate(e) if v <= 10 * floor]
    if at_floor:
        R.append(f"errors at levels {at_floor} are within 10x of the round-off floor {floor:.3e}: no order measurable")
    tail = e_signed[-3:]
    if any(v != 0 for v in tail) and len({math.copysign(1, v) for v in tail if v != 0}) > 1:
        R.append("the signed error changes sign over the finest three levels (oscillatory: no order)")
    pw = pairwise_orders(x, e)
    out["pairwise_orders"] = pw
    f = fit_order(x, e, int(spec.get("fit_points", 3)))
    out["fit"] = f
    p_fit = f["order"]
    p_pair = pw[-1] if pw else float("nan")
    out["observed_order"] = p_fit if spec.get("estimator", "fit") == "fit" else p_pair
    out["finest_pair_order"] = p_pair
    upper = float(spec["max_order"]) if spec.get("allow_superconvergence") else p_exp + tol
    if spec.get("allow_superconvergence") and upper > p_exp + 1.0:
        R.append(f"max_order {upper:g} > expected + 1: superconvergence by more than one order is not a declaration "
                 "but an open bound")
    k_fit = int(spec.get("fit_points", 3))
    if len(x) >= 2:
        xs_fit = x[-k_fit:] if k_fit <= len(x) else x
        span = xs_fit[0] / xs_fit[-1] if xs_fit[-1] > 0 else float("nan")
        out["span"] = span
        smin = float(spec.get("min_span", MIN_SPAN))
        if not span >= smin:
            R.append(f"the fitted levels span only h_coarse/h_fine = {span:.3g} < {smin:g}: too short to resolve a "
                     "hidden lower-order term")
    for name, p in (("fitted", p_fit), ("finest-pair", p_pair)):
        if not math.isfinite(p):
            R.append(f"{name} order is not finite")
        elif p < p_exp - tol:
            R.append(f"{name} order {p:.3f} below {p_exp} - {tol}")
        elif p > upper:
            R.append(f"{name} order {p:.3f} above {upper:g}: superconvergence not declared before running (the test "
                     "may not measure what it claims)" if not spec.get("allow_superconvergence")
                     else f"{name} order {p:.3f} above the declared max_order {upper:g}")
    if math.isfinite(f["stderr"]) and f["stderr"] > tol:
        R.append(f"fit standard error {f['stderr']:.3f} > tolerance {tol}: the data do not determine the order")
    fin = [p for p in pw if math.isfinite(p)]
    if len(fin) >= 2:
        # with declared superconvergence, the distance is to the band [expected, max_order]
        lo_b, hi_b = (p_exp, upper) if spec.get("allow_superconvergence") else (p_exp, p_exp)
        dist = [max(lo_b - p, 0.0, p - hi_b) for p in fin]
        growth = dist[-1] - min(dist[:-1])
        out["distance_growth"] = growth
        out["drift"] = fin[-1] - max(fin[:-1])          # kept for display
        md = float(spec.get("max_downward_drift", MAX_DOWNWARD_DRIFT))
        if growth > md:
            R.append(f"pairwise orders move AWAY from {p_exp} toward fine meshes (|p - p_expected| grows by "
                     f"{growth:.3f} > {md}; pairwise {[round(p, 3) for p in fin]}): a term of another order is "
                     "emerging, or the sequence is pre-asymptotic")
    R += step_ratio_problems(spec, levels, by_level) + temporal_contamination_problems(spec, series, e)
    out["verdict"] = "PASS" if not R else "FAIL"
    return out


def step_ratio_problems(spec: dict, levels: list, by_level: dict) -> list[str]:
    """a variable-step order study (abscissa dt, prescribed step_function) stays inside the prescribed
    zero-stability bound: every level's run reports dt_max_ratio (the largest ratio of successive steps)."""
    if spec.get("max_step_ratio") is None:
        return []
    bound = float(spec["max_step_ratio"])
    out = []
    for lv in levels:
        v = (by_level.get(lv) or {}).get("dt_max_ratio")
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            out.append(f"level '{lv}': dt_max_ratio missing")
        elif float(v) > bound:
            out.append(f"level '{lv}': step ratio {float(v):.4g} > the prescribed zero-stability bound {bound:g}")
    return out


def temporal_contamination_problems(spec: dict, series: dict, e: list) -> list[str]:
    """a spatial order study of a transient problem reports, at the finest
    mesh, the result with dt and dt/2; their difference is at most `frac` (prescribed, <= 0.1) of the finest spatial
    error, or the 'spatial' order is partly temporal."""
    tc = spec.get("temporal_contamination")
    if not tc:
        return []
    chk = series.get("temporal_check") if isinstance(series, dict) else None
    if not isinstance(chk, dict) or not all(isinstance(chk.get(k), (int, float)) for k in ("value_dt",
                                                                                          "value_dt_half")):
        return ["temporal_check {value_dt, value_dt_half} at the finest mesh missing"]
    d = abs(float(chk["value_dt"]) - float(chk["value_dt_half"]))
    lim = min(float(tc.get("frac", 0.1)), 0.1) * (e[-1] if e else 0.0)
    return [] if d <= lim else [f"dt vs dt/2 at the finest mesh differ by {d:.3g} > {lim:.3g} (frac of the finest "
                                "spatial error): the spatial study is contaminated by the time error"]


# Celik et al. 2008 three-mesh Richardson/GCI, a copy of python/axionfem/convergence.py, so that the criteria do not
# depend on the library under test,
# and the verdict must not change with it.  Kept identical in behaviour; tests compare the two on sample data.
RICHARDSON_MIN_ORDER = 0.5
RICHARDSON_MAX_ORDER = 8.0


def _monotone_order(h1: float, h2: float, h3: float, ratio: float, p_max: float) -> float | None:
    """The root p of (h3^p - h2^p) / (h2^p - h1^p) = ratio (> 0), increasing in p; None if it is not in (0, p_max]."""
    def lhs(p):
        return (h3 ** p - h2 ** p) / (h2 ** p - h1 ** p)
    if ratio <= math.log(h3 / h2) / math.log(h2 / h1):   # the limit p -> 0: no positive root
        return None
    hi = p_max
    if lhs(hi) < ratio:
        return math.inf
    lo = 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lhs(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _oscillatory_order(r21: float, r32: float, ratio: float, iterations: int = 200) -> float | None:
    """Celik's fixed point for s = -1; None unless it converges."""
    try:
        p = abs(math.log(abs(ratio))) / math.log(r21)
        for _ in range(iterations):
            p_new = abs(math.log(abs(ratio)) + math.log((r21 ** p + 1) / (r32 ** p + 1))) / math.log(r21)
            if abs(p_new - p) < 1e-12 * max(1.0, p):
                return p_new
            p = p_new
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def richardson(h: list[float], values: list[float], formal_order: float | None = None,
               noise: float | None = None) -> dict:
    """Observed order, extrapolated value and fine-mesh GCI from three meshes (in any order).

    ``formal_order``: the quantity's formal order. The order used is then min(observed, formal), and the safety
    factor is 3 when they differ by more than 10 % (Roache), else 1.25; an observed order above twice the formal
    one is flagged in ``behaviour`` but still gives this conservative GCI. Without it, orders above RICHARDSON_MAX_ORDER are
    "unreliable order" (no GCI). ``noise``: an absolute level below which a difference counts as zero (default 1e3
    machine epsilons of the largest value). ``order`` is None when the observed order exceeds RICHARDSON_MAX_ORDER
    (``order_exceeds``).
    """
    if len(h) != 3 or len(values) != 3:
        raise ValueError("richardson needs exactly three meshes")
    (h1, f1), (h2, f2), (h3, f3) = sorted(zip(h, values))
    if not (0 < h1 < h2 < h3):
        raise ValueError("the three mesh sizes must be positive and distinct")
    r21 = h2 / h1
    r32 = h3 / h2
    e21 = f2 - f1
    e32 = f3 - f2
    floor = noise if noise is not None else 1e3 * 2.2e-16 * max(abs(f1), abs(f2), abs(f3))
    out = {"h": [h1, h2, h3], "values": [f1, f2, f3], "order": None, "extrapolated": None, "gci_fine": None,
           "extrapolation_error": None}

    def done(behaviour, p=None):
        out.update({"behaviour": behaviour, "order": p})
        return out

    small21 = abs(e21) <= floor
    small32 = abs(e32) <= floor
    if small21 and small32:
        out.update({"extrapolated": f1, "gci_fine": 0.0, "extrapolation_error": 0.0})
        return done("unchanged")
    if small21 or small32:
        return done("no usable order (one difference at the noise floor)")
    ratio = e32 / e21
    if ratio > 0:
        behaviour = "monotone"
        p = _monotone_order(h1, h2, h3, ratio, RICHARDSON_MAX_ORDER)
        if p is None:
            return done("no usable order")
    else:
        behaviour = "oscillatory"
        p = _oscillatory_order(r21, r32, ratio)
        if p is None:
            return done("iteration did not converge")
    observed = p if math.isfinite(p) else None    # above RICHARDSON_MAX_ORDER the bisection reports infinity: no number
    if p < RICHARDSON_MIN_ORDER:
        return done("no usable order", observed)
    if formal_order:
        # Roache: the formal order caps the order used and the safety factor rises to 3 when they disagree; an
        # observed order far above the formal one is flagged but still gives this conservative GCI
        p_use = min(p, formal_order)
        safety = 3.0 if abs(p - formal_order) / formal_order > 0.1 else 1.25
        if p > 2 * formal_order:
            behaviour += ", observed order far above formal"
    else:
        if p > RICHARDSON_MAX_ORDER:
            out["order_exceeds"] = RICHARDSON_MAX_ORDER
            return done("unreliable order", observed)
        p_use = p
        safety = 1.25
    extrapolated = (r21 ** p_use * f1 - f2) / (r21 ** p_use - 1)
    approx = abs(e21 / f1) if f1 != 0 else math.inf
    out.update({"extrapolated": extrapolated, "gci_fine": safety * approx / (r21 ** p_use - 1),
                "extrapolation_error": abs((extrapolated - f1) / extrapolated) if extrapolated != 0 else None,
                "safety_factor": safety, "order_used": p_use})
    if observed is None:
        out["order_exceeds"] = RICHARDSON_MAX_ORDER
    return done(behaviour, observed)


def richardson_series(h: Iterable[float], values: Iterable[float], formal_order: float | None = None) -> dict:
    """Three-mesh Richardson/GCI (Celik et al. 2008), a self-contained copy."""
    return richardson(list(h), list(values), formal_order=formal_order)


def assess_functional(series: dict | None, spec: dict, by_level: dict, params: dict, quick: bool = False) -> dict:
    """functional_convergence: a quantity on >= 3 prescribed levels must converge monotonically with an order in
    the prescribed range and a fine-mesh GCI <= gci_cap_rel; the GCI then measures our discretisation error for
    comparisons whose recipe adds it."""
    out = {"id": spec["id"], "series": spec["series"], "reasons": []}
    R = out["reasons"]
    if not isinstance(series, dict) or not isinstance(series.get("levels"), list) or \
            not isinstance(series.get("values"), list) or len(series["levels"]) != len(series["values"]):
        R.append("series with 'levels' and 'values' missing from the case result")
        out["verdict"] = "FAIL"
        return out
    nominal, x, probs = series_abscissa(spec, series["levels"], by_level)
    R += probs
    try:
        prescribed = list(get_path(params, spec["levels_from"]))
        want = prescribed[:len(nominal)] if quick else prescribed
        if [float(v) for v in nominal if v is not None] != [float(v) for v in want]:
            R.append(f"series levels {nominal} are not the prescribed sequence {prescribed}")
    except KeyError:
        R.append(f"levels_from '{spec.get('levels_from')}' not found in the run parameters")
    vals = [float(v) for v in series["values"]]
    if len(x) < 3 or len(x) != len(vals):
        R.append("needs at least 3 levels with abscissae")
        out["verdict"] = "FAIL"
        return out
    r = richardson_series(x[-3:], vals[-3:], formal_order=spec.get("formal_order"))
    out["richardson"] = r
    lo, hi = spec["order_range"]
    if not str(r.get("behaviour", "")).startswith("monotone"):
        R.append(f"behaviour '{r.get('behaviour')}' (needs monotone convergence)")
    if r.get("order") is None or not lo <= r["order"] <= hi:
        R.append(f"observed order {r.get('order')} outside the prescribed range [{lo}, {hi}]")
    g = r.get("gci_fine")
    if g is None or g > float(spec["gci_cap_rel"]):
        R.append(f"fine-mesh GCI {g} above the cap {spec['gci_cap_rel']}")
    out["gci_fine"] = g
    out["verdict"] = "PASS" if not R else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------- comparisons
def _allowance(tolerance: dict, scale: float) -> float:
    """allowance = abs + rel * |scale|."""
    if not tolerance or not any(k in tolerance for k in ("abs", "rel")):
        raise ValueError("tolerance needs 'abs' and/or 'rel'")
    for k in ("abs", "rel"):
        if k in tolerance and not (math.isfinite(float(tolerance[k])) and float(tolerance[k]) >= 0):
            raise ValueError(f"tolerance.{k} must be finite and >= 0")
    return ref_allowance_fixed(tolerance, scale)


def compare(value, reference: dict, tolerance: dict, gci: float | None = None,
            gci_max_fraction: float | None = None) -> dict:
    """A scalar against {"value": v, "uncertainty": u} or {"interval": [lo, hi]} (not both).

    Pass: |value - v| <= allowance + u, or lo - allowance <= value <= hi + allowance.  allowance = abs + rel*|ref|
    (+ gci*|value| if the recipe adds the measured GCI, capped at gci_max_fraction times the fixed allowance: a
    slowly converging, less accurate solution may not earn an unbounded allowance)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {"value": value, "verdict": "FAIL", "reasons": ["the computed value is not a number"]}
    has_v, has_i = reference.get("value") is not None, reference.get("interval") is not None
    if has_v == has_i:
        raise ValueError("reference needs exactly one of 'value' or 'interval'")
    extra = (gci or 0.0) * abs(value) if math.isfinite(value) else 0.0

    def capped(fixed):
        return min(extra, gci_max_fraction * fixed) if gci_max_fraction is not None else extra
    if has_i:
        lo, hi = map(float, reference["interval"])
        if not lo < hi:
            raise ValueError(f"reference interval [{lo}, {hi}] must have lo < hi")
        mid = 0.5 * (lo + hi)
        fixed = _allowance(tolerance, mid)
        allow = fixed + capped(fixed)
        ok = math.isfinite(value) and (lo - allow) <= value <= (hi + allow)
        dev = 0.0 if lo <= value <= hi else (value - hi if value > hi else value - lo)
        return {"value": value, "reference_interval": [lo, hi], "allowance": allow, "deviation": dev,
                "relative_deviation": dev / mid if mid else float("nan"), "verdict": "PASS" if ok else "FAIL"}
    ref = float(reference["value"])
    unc = float(reference.get("uncertainty") or 0.0)
    fixed = _allowance(tolerance, ref)
    allow = fixed + capped(fixed)
    dev = value - ref
    ok = math.isfinite(value) and abs(dev) <= allow + unc
    return {"value": value, "reference": ref, "reference_uncertainty": unc, "allowance": allow, "deviation": dev,
            "relative_deviation": dev / ref if ref else float("nan"), "verdict": "PASS" if ok else "FAIL"}


def compare_profile(result, reference: dict, tolerance: dict) -> dict:
    """A profile (e.g. a centreline velocity table): the case returns {"coords": [...], "values": [...]} at exactly the
    reference coordinates; every point must satisfy |v_i - r_i| <= abs + rel|r_i| + u_i."""
    rc = list(map(float, (reference.get("coords") or {}).get("values") or []))
    rv = list(map(float, reference.get("values") or []))
    if not isinstance(result, dict) or "coords" not in result or "values" not in result:
        return {"verdict": "FAIL", "reasons": ["profile quantity must be {'coords': [...], 'values': [...]}"]}
    try:
        c, v = list(map(float, result["coords"])), list(map(float, result["values"]))
    except (TypeError, ValueError):
        return {"verdict": "FAIL", "reasons": ["profile coords/values are not numbers"]}
    ctol = float(reference.get("coord_tol", 1e-12))
    if len(c) != len(rc) or any(abs(a - b) > ctol * max(1.0, abs(b)) for a, b in zip(c, rc)):
        return {"verdict": "FAIL", "reasons": ["profile evaluated at other coordinates than the reference's"]}
    u = reference.get("uncertainty") or 0.0
    us = list(map(float, u)) if isinstance(u, list) else [float(u)] * len(rv)
    devs, allows, bad = [], [], []
    for i, (a, b, ui) in enumerate(zip(v, rv, us)):
        al = _allowance(tolerance, b) + ui
        d = a - b
        devs.append(d)
        allows.append(al)
        if not (math.isfinite(a) and abs(d) <= al):
            bad.append(i)
    return {"values": v, "reference_values": rv, "coords": c, "deviation": devs, "allowance": allows,
            "max_abs_deviation": max(map(abs, devs)) if devs else None,
            "rms_deviation": math.sqrt(sum(d * d for d in devs) / len(devs)) if devs else None,
            "failing_points": bad, "verdict": "PASS" if not bad else "FAIL"}


def periodic_statistics(t, y, definition: dict) -> dict:
    """Statistics of a time-periodic signal, computed here from the case's raw signal.

    definition (prescribed with the reference): statistic in mean | amplitude | period | frequency | strouhal | max | min,
    t_start, n_periods_min, samples_per_period_min, periodicity_tol_rel; for strouhal, length and velocity.  Periods
    from upward crossings of the window mean (linear interpolation); max/min per period with a parabolic fit through
    the discrete extremum and its neighbours; statistics averaged over the last n_periods_min periods."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    out = {"reasons": []}
    R = out["reasons"]
    if t.ndim != 1 or t.shape != y.shape or t.size < 8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        R.append("signal needs finite 't' and 'signal' arrays of equal length >= 8")
        return out
    if np.any(np.diff(t) <= 0):
        R.append("time must increase strictly")
        return out
    m = t >= float(definition.get("t_start", t[0]))
    t, y = t[m], y[m]
    n_req = int(definition["n_periods_min"])
    mean = float(np.trapezoid(y, t) / (t[-1] - t[0])) if t.size > 1 else float("nan")
    for _ in range(2):
        s = y - mean
        idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        cr = t[idx] - s[idx] * (t[idx + 1] - t[idx]) / (s[idx + 1] - s[idx])
        if cr.size < n_req + 1:
            R.append(f"only {max(cr.size - 1, 0)} full periods after t_start, need {n_req}")
            return out
        a, b = cr[-n_req - 1], cr[-1]
        w = (t >= a) & (t <= b)
        mean = float(np.trapezoid(y[w], t[w]) / (t[w][-1] - t[w][0]))
    periods = np.diff(cr[-n_req - 1:])
    T = float(np.mean(periods))
    spread = float((periods.max() - periods.min()) / T)
    if spread > float(definition.get("periodicity_tol_rel", 0.01)):
        R.append(f"period varies by {spread:.3g} (relative) over the window: not periodic yet")
    dtmax = float(np.max(np.diff(t[(t >= cr[-n_req - 1]) & (t <= cr[-1])])))
    spp = T / dtmax
    if spp < float(definition.get("samples_per_period_min", 20)):
        R.append(f"{spp:.1f} samples per period < {definition.get('samples_per_period_min', 20)}")
    maxs, mins, means = [], [], []
    for k in range(n_req):
        w = np.where((t >= cr[-n_req - 1 + k]) & (t <= cr[-n_req + k]))[0]
        means.append(float(np.trapezoid(y[w], t[w]) / (t[w][-1] - t[w][0])) if w.size > 1 else float(y[w][0]))
        for store, fn in ((maxs, np.argmax), (mins, np.argmin)):
            j = w[fn(y[w])]
            if 0 < j < t.size - 1:
                x3, y3 = t[j - 1:j + 2], y[j - 1:j + 2]
                c = np.polyfit(x3 - x3[1], y3, 2)
                store.append(float(c[2] - c[1] ** 2 / (4 * c[0])) if c[0] != 0 else float(y[j]))
            else:
                store.append(float(y[j]))
    # stationarity of the signal itself, not only of its period: a decaying or growing
    # amplitude, or a drifting mean, makes "max" or "amplitude" a transient number
    amps = np.subtract(maxs, mins) / 2
    A = float(np.mean(amps))
    ref_scale = A if A > 0 else max(abs(mean), 1e-300)
    tol_p = float(definition.get("periodicity_tol_rel", 0.01))
    for name, arr in (("maxima", maxs), ("minima", mins), ("means", means)):
        sp = (max(arr) - min(arr)) / ref_scale
        out[f"period_{name}_spread_rel"] = sp
        if sp > tol_p:
            R.append(f"per-period {name} vary by {sp:.3g} of the amplitude over the window (> {tol_p}): not "
                     "stationary yet (decaying, growing or drifting)")
    if n_req >= 2:
        slope = float(np.polyfit(np.arange(n_req, dtype=float), amps, 1)[0])
        trend = abs(slope) * (n_req - 1) / ref_scale
        out["amplitude_trend_rel"] = trend
        if trend > tol_p:
            R.append(f"the per-period amplitude trends by {trend:.3g} (relative) over the window (> {tol_p})")
    out.update({"mean": mean, "period": T, "frequency": 1.0 / T, "max": float(np.mean(maxs)),
                "min": float(np.mean(mins)), "amplitude": float(np.mean(np.subtract(maxs, mins)) / 2),
                "periods_used": n_req, "period_spread_rel": spread, "samples_per_period": spp})
    if definition.get("statistic") == "strouhal":
        out["strouhal"] = float(definition["length"]) / (float(definition["velocity"]) * T)
    stat = definition.get("statistic")
    if stat not in out:
        R.append(f"unknown statistic '{stat}'")
    else:
        out["value"] = out[stat]
    return out


def check_identity(qty: dict, spec: dict) -> dict:
    """an identity (a balance that holds exactly in exact arithmetic) is checked to
    round-off, |value| <= c * eps * scale, or to the Newton tolerance, |value| <= c * newton_rtol * scale.  `scale`
    is PRESCRIBED; the case reports the magnitude it measured (scale_quantity), which must be within 10x of it; for
    newton_tol the case reports the tolerance it used, which must equal the prescribed newton_rtol."""
    q = spec["quantity"]
    R = []
    try:
        v = abs(float(qty[q]))
    except (KeyError, TypeError, ValueError):
        return {"quantity": q, "verdict": "FAIL", "reasons": ["quantity missing or not a number"]}
    c, scale = float(spec["c"]), abs(float(spec["scale"]))
    if c > IDENTITY_MAX_C:
        R.append(f"c = {c:g} > {IDENTITY_MAX_C:g}: not a round-off bound")
    mag = qty.get(spec.get("scale_quantity")) if spec.get("scale_quantity") else None
    if not isinstance(mag, (int, float)) or not math.isfinite(float(mag)) or float(mag) <= 0:
        R.append(f"the case reports no positive magnitude '{spec.get('scale_quantity')}' for the prescribed scale")
    elif not (float(mag) / SOLUTION_SCALE_FACTOR <= scale <= float(mag) * SOLUTION_SCALE_FACTOR):
        R.append(f"prescribed scale {scale:g} is not within {SOLUTION_SCALE_FACTOR:g}x of the reported magnitude "
                 f"{float(mag):g}")
    if spec.get("bound_kind") == "newton_tol":
        rt = qty.get(spec.get("rtol_quantity"))
        if not isinstance(rt, (int, float)) or float(rt) != float(spec["newton_rtol"]):
            R.append(f"the run's Newton tolerance {rt} is not the prescribed newton_rtol {spec.get('newton_rtol')}")
        bound = c * float(spec["newton_rtol"]) * scale
    else:
        bound = c * EPS * scale
    if not (math.isfinite(v) and v <= bound):
        R.append(f"|{q}| = {v:.3e} > {bound:.3e}")
    return {"quantity": q, "kind": "identity", "value": v, "bound": bound, "verdict": "PASS" if not R else "FAIL",
            "reasons": R}


def check_two_sided(qty: dict, spec: dict) -> dict:
    """lo - tol*scale <= min and max <= hi + tol*scale over the declared evaluation set
    (dofs, quadrature points or both: P2 undershoots between nodes).  The case reports {set: {min, max}} (or a list
    of such, one per run); a missing set FAILs."""
    q = spec.get("quantity") or spec.get("field")
    val = qty.get(q)
    sets = {"dofs": ["dofs"], "quadrature": ["quadrature"], "both": ["dofs", "quadrature"]}.get(spec.get("eval_set"))
    if sets is None:
        return {"quantity": q, "verdict": "FAIL", "reasons": ["eval_set must be dofs, quadrature or both"]}
    items = val if isinstance(val, list) else [val]
    tol = float(spec.get("tol", 0.0)) * abs(float(spec.get("scale", 0.0)))
    R, mins, maxs = [], [], []
    for i, it in enumerate(items):
        for st in sets:
            d = it.get(st) if isinstance(it, dict) else None
            if not isinstance(d, dict) or not all(isinstance(d.get(k), (int, float)) for k in ("min", "max")):
                R.append(f"run {i}: no {{min, max}} over the '{st}' set")
                continue
            mins.append(float(d["min"]))
            maxs.append(float(d["max"]))
            if "lo" in spec and not float(d["min"]) >= float(spec["lo"]) - tol:
                R.append(f"run {i}: min over {st} {float(d['min']):.6g} < lo {spec['lo']} - {tol:.3g}")
            if "hi" in spec and not float(d["max"]) <= float(spec["hi"]) + tol:
                R.append(f"run {i}: max over {st} {float(d['max']):.6g} > hi {spec['hi']} + {tol:.3g}")
    return {"quantity": q, "kind": "bounds", "min": min(mins) if mins else None, "max": max(maxs) if maxs else None,
            "verdict": "PASS" if not R and mins else "FAIL", "reasons": R or ([] if mins else ["no values"])}


def _parabolic_peak(t, y, j):
    if 0 < j < len(t) - 1:
        c = np.polyfit(t[j - 1:j + 2] - t[j], y[j - 1:j + 2], 2)
        if c[0] != 0:
            x = -c[1] / (2 * c[0])
            return float(t[j] + x), float(c[2] - c[1] ** 2 / (4 * c[0]))
    return float(t[j]), float(y[j])


def extract_feature(sig: dict, definition: dict) -> dict:
    """one feature extraction, applied to our raw signal and to the cross-code's.
    peak: location and value by a parabola through the discrete maximum in the window, with at least
    min_samples_per_width samples above half height; crossing: first crossing of `level` in `direction`
    (linear or cubic interpolation); argmax_derivative: the maximum of the derivative SUPPLIED with the signal from
    the right-hand side (never finite differences of samples)."""
    out = {"reasons": []}
    R = out["reasons"]
    try:
        t = np.asarray(sig["t"], float)
        y = np.asarray(sig["signal"] if "signal" in sig else sig["y"], float)
    except (KeyError, TypeError, ValueError):
        R.append("signal needs 't' and 'signal' arrays")
        return out
    if t.ndim != 1 or t.shape != y.shape or t.size < 5 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(y)):
        R.append("signal needs finite, equal-length arrays with strictly increasing t (>= 5 samples)")
        return out
    w0, w1 = definition.get("window") or [t[0], t[-1]]
    m = (t >= float(w0)) & (t <= float(w1))
    idx = np.where(m)[0]
    if idx.size < 3:
        R.append("fewer than 3 samples in the window")
        return out
    typ = definition.get("type")
    if typ == "peak":
        j = int(idx[np.argmax(y[idx])])
        loc, val = _parabolic_peak(t, y, j)
        base = float(np.min(y[idx]))
        half = base + 0.5 * (y[j] - base)
        k0 = j
        while k0 > 0 and y[k0 - 1] >= half:
            k0 -= 1
        k1 = j
        while k1 < len(y) - 1 and y[k1 + 1] >= half:
            k1 += 1
        n = k1 - k0 + 1
        need = int(definition.get("min_samples_per_width", 5))
        out.update(location=loc, value=val, samples_per_width=n)
        if n < need:
            R.append(f"{n} samples above half height < {need}: too coarse to locate the peak")
    elif typ == "crossing":
        lev, dirn = float(definition["level"]), definition.get("direction", "any")
        s = y - lev
        hit = None
        for i in idx[:-1]:
            if i + 1 > idx[-1]:
                break
            up, down = s[i] < 0 <= s[i + 1], s[i] > 0 >= s[i + 1]
            if (dirn in ("up", "any") and up) or (dirn in ("down", "any") and down):
                hit = int(i)
                break
        if hit is None:
            R.append(f"no {dirn} crossing of {lev} in the window")
            return out
        x = t[hit] - s[hit] * (t[hit + 1] - t[hit]) / (s[hit + 1] - s[hit])
        if definition.get("interp") == "cubic" and 1 <= hit < len(t) - 2:
            tt, ss = t[hit - 1:hit + 3], s[hit - 1:hit + 3]
            roots = np.roots(np.polyfit(tt - t[hit], ss, 3))
            real = [float(r.real) + t[hit] for r in roots if abs(r.imag) < 1e-12 and t[hit] <= r.real + t[hit]
                    <= t[hit + 1]]
            if real:
                x = real[0]
        out.update(location=float(x), value=lev)
    elif typ == "argmax_derivative":
        if sig.get("derivative") is None or sig.get("derivative_source") in (None, "finite_difference"):
            R.append("argmax_derivative needs 'derivative' evaluated from the right-hand side with derivative_source "
                     "!= 'finite_difference'")
            return out
        dy = np.asarray(sig["derivative"], float)
        if dy.shape != t.shape:
            R.append("derivative must have the length of t")
            return out
        j = int(idx[np.argmax(dy[idx])])
        loc, val = _parabolic_peak(t, dy, j)
        out.update(location=loc, value=val)
    else:
        R.append(f"unknown feature type {typ!r} (one of {FEATURE_TYPES})")
        return out
    out["output"] = out.get(definition.get("output", "location"))
    return out


def compare_species(val, q: dict, tol: dict) -> dict:
    """a per-species quantity against the reference's species values, one tolerance recipe for
    the family; species declared not_compared in stage 1 are skipped."""
    if not isinstance(val, dict) or list(val.get("species") or []) != list(q.get("species") or []):
        return {"verdict": "FAIL", "reasons": [f"species quantity must be {{'species': {q.get('species')}, "
                                               "'values': [...]}} in the reference's order"]}
    skip = set(q.get("not_compared") or [])
    per, bad = {}, []
    for name, a, b in zip(q["species"], val.get("values") or [], q.get("values") or []):
        if name in skip:
            continue
        c = compare(a, {"value": b}, tol)
        per[name] = c
        if c["verdict"] != "PASS":
            bad.append(name)
    return {"per_species": per, "failing_species": bad, "verdict": "PASS" if per and not bad else "FAIL"}


def compare_complex(val, q: dict, tol: dict) -> dict:
    """|Z - Z_ref| <= abs + rel |Z_ref| + u, on the modulus of the complex difference."""
    try:
        z = complex(float(val["re"]), float(val["im"]))
    except (KeyError, TypeError, ValueError):
        return {"verdict": "FAIL", "reasons": ["a complex quantity must be {'re': x, 'im': y} (a real value is not a "
                                               "comparison of a complex quantity)"]}
    zr = complex(float(q["value"]["re"]), float(q["value"]["im"]))
    allow = _allowance(tol, abs(zr)) + float(q.get("uncertainty") or 0.0)
    d = abs(z - zr)
    return {"value": {"re": z.real, "im": z.imag}, "reference": {"re": zr.real, "im": zr.imag}, "deviation": d,
            "allowance": allow, "verdict": "PASS" if math.isfinite(d) and d <= allow else "FAIL"}


def assess_tolerance_behaviour(series, spec: dict, params: dict) -> dict:
    """an adaptive integrator over the prescribed tolerance sweep.  Reliability:
    err <= C_hi * tol at every tolerance; efficiency: err >= C_lo * tol at the tightest k tolerances, or steps <=
    steps_bound there; an optional slope band for log err against log tol.  No monotonicity rule (errors of adaptive
    integrators are not monotone in tol)."""
    out = {"id": spec.get("id"), "series": spec.get("series"), "reasons": []}
    R = out["reasons"]
    try:
        tols = [float(v) for v in get_path(params, spec["tols_from"])]
    except (KeyError, TypeError, ValueError):
        R.append(f"tols_from '{spec.get('tols_from')}' is not a numeric list in the run parameters")
        out["verdict"] = "FAIL"
        return out
    if not isinstance(series, dict) or [float(v) for v in series.get("tols") or []] != tols or \
            len(series.get("error") or []) != len(tols):
        R.append(f"series must report 'tols' equal to the prescribed {tols} and one 'error' per tolerance")
        out["verdict"] = "FAIL"
        return out
    err = [abs(float(v)) for v in series["error"]]
    hi, lo, k = float(spec["C_hi"]), float(spec["C_lo"]), int(spec.get("tightest_k", 2))
    bad = [t for t, e in zip(tols, err) if not e <= hi * t]
    if bad:
        R.append(f"reliability: err > {hi:g} * tol at tol {bad[:4]}")
    order = sorted(range(len(tols)), key=lambda i: tols[i])[:k]
    steps = series.get("steps") or []
    ineff = [tols[i] for i in order if not err[i] >= lo * tols[i]]
    if ineff:
        sb = spec.get("steps_bound")
        if sb is None or len(steps) != len(tols) or any(float(steps[i]) > float(sb) for i in order):
            R.append(f"efficiency: err < {lo:g} * tol at the tightest tolerances {ineff} (an estimator that "
                     "over-estimates the error wastes steps)")
    if spec.get("slope_band"):
        f = fit_order(tols, err)
        a, b = spec["slope_band"]
        out["slope"] = f["order"]
        if not (math.isfinite(f["order"]) and a <= f["order"] <= b):
            R.append(f"slope d log err / d log tol = {f['order']:.3f} outside [{a}, {b}]")
    out.update(tols=tols, error=err, verdict="PASS" if not R else "FAIL")
    return out


def check_bound(value, spec: dict) -> dict:
    """{"min": a, "max": b, "abs_max": c}: every bound given applies; at least one is required."""
    bounds = {k: spec[k] for k in ("min", "max", "abs_max") if k in spec}
    if not bounds:
        raise ValueError("check needs at least one of 'min', 'max', 'abs_max'")
    import numbers
    # `max: 0` must not pass a reported -1 or the string "0".  Only a real
    # number is a value (never a string, a bool or None), it must be finite, and a COUNT (`count: true`, or a check
    # whose only bound is `max: 0`) must be a non-negative integer.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return {"quantity": spec["quantity"], "value": repr(value), **bounds, "verdict": "FAIL",
                "reasons": [f"not a number ({type(value).__name__}): a check needs a finite int or float"]}
    value = float(value)
    ok = math.isfinite(value)
    reasons = [] if ok else ["not finite"]
    if spec.get("count") or (set(bounds) == {"max"} and float(bounds["max"]) == 0):
        if ok and (value < 0 or not value.is_integer()):
            ok = False
            reasons.append(f"a count must be a non-negative integer, got {value:g}")
    if "min" in bounds:
        ok = ok and value >= float(bounds["min"])
    if "max" in bounds:
        ok = ok and value <= float(bounds["max"])
    if "abs_max" in bounds:
        ok = ok and abs(value) <= float(bounds["abs_max"])
    return {"quantity": spec["quantity"], "value": value, **bounds, "verdict": "PASS" if ok else "FAIL",
            **({"reasons": reasons} if reasons else {})}


def solver_problems(runs: list, spec: dict) -> list[str]:
    """Every solve must report convergence and a residual within the prescribed bound."""
    if not runs:
        return ["the case reported no solves (runs is empty): nothing shows that a solver ran"]
    key, mx = spec.get("residual_key", "relative_residual"), float(spec["max_residual"])
    errs = []
    for r in runs:
        lab = r.get("level") or r.get("label") or "?"
        s = r.get("solver")
        if not isinstance(s, dict):
            errs.append(f"run {lab}: no solver diagnostics")
            continue
        if s.get("converged") is not True:
            errs.append(f"run {lab}: solver did not converge (converged={s.get('converged')})")
        res = s.get(key)
        if not isinstance(res, (int, float)) or not math.isfinite(float(res)) or float(res) > mx:
            errs.append(f"run {lab}: {key} = {res} exceeds the prescribed bound {mx} (or is missing)")
    return errs


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluate(case: dict, result: dict, quick: bool = False) -> dict:
    """Verdicts of the case's acceptance items.  Every item is evaluated separately; an item that cannot be evaluated
    (a malformed quantity) is a FAIL with the reason, never a crash."""
    acc = case.get("acceptance") or {}
    ref = case.get("reference") or {}
    rq = ref.get("quantities") or {}
    params = (case.get("run") or {}).get("parameters") or {}
    runs = result.get("runs") or []
    by_level = {r.get("level"): r for r in runs if isinstance(r, dict) and r.get("level") is not None}
    series = result.get("series") or {}
    qty = result.get("quantities") or {}
    verdict = {k: "NOT_TESTED" for k in VERDICT_KEYS}

    def mark(level, ok, informative=False):
        if informative:
            return
        key = VERDICT_OF_LEVEL[level]
        if not ok:
            verdict[key] = "FAIL"
        elif verdict[key] == "NOT_TESTED":
            verdict[key] = "PASS"

    def guarded(fn, label):
        try:
            return fn()
        except Exception as e:     # noqa: BLE001  -- a malformed result is a FAIL, not a crash
            return {"verdict": "FAIL", "reasons": [f"could not evaluate {label}: {type(e).__name__}: {e}"]}

    conv, fconv, comps, checks, sens = [], [], [], [], []
    for spec in acc.get("convergence", []) or []:
        a = guarded(lambda: assess_convergence(series.get(spec["series"]), spec, by_level, params, quick),
                    spec.get("series"))
        a.update({"series": spec["series"], "level": spec["level"]})
        conv.append(a)
        mark(spec["level"], a["verdict"] == "PASS")
    gci = {}
    for spec in acc.get("functional_convergence", []) or []:
        a = guarded(lambda: assess_functional(series.get(spec["series"]), spec, by_level, params, quick),
                    spec.get("id"))
        a.update({"id": spec["id"], "level": spec["level"]})
        fconv.append(a)
        gci[spec["id"]] = a if a["verdict"] == "PASS" else None
        mark(spec["level"], a["verdict"] == "PASS")
    for spec in acc.get("comparisons", []) or []:
        def one():
            q = rq.get(spec["reference"])
            if q is None:
                return {"verdict": "FAIL", "reasons": [f"no reference '{spec['reference']}' in the reference block"]}
            if spec["quantity"] not in qty:
                return {"verdict": "FAIL", "reasons": ["quantity missing from the case result"]}
            tol = comparison_tolerance(case, spec)
            g, gfrac = None, None
            if tol.get("gci"):
                rcp = tol["gci"]
                if not isinstance(rcp, dict):
                    return {"verdict": "FAIL", "reasons": ["the recipe's gci must be an object {formal_order, "
                                                           "order_window, cap_rel, max_fraction}"]}
                fa = gci.get(spec.get("gci_from"))
                if fa is None:
                    return {"verdict": "FAIL", "reasons": [f"the recipe adds the GCI of '{spec.get('gci_from')}', "
                                                           "which did not pass"]}
                rr = fa.get("richardson") or {}
                p_obs, gf = rr.get("order"), fa.get("gci_fine")
                po, win = float(rcp["formal_order"]), min(float(rcp["order_window"]), GCI_ORDER_WINDOW_MAX)
                why = []
                if p_obs is None or not abs(float(p_obs) - po) <= win:
                    why.append(f"observed order {p_obs} outside the recipe's window {po} +- {win}")
                if gf is None or not float(gf) <= float(rcp["cap_rel"]):
                    why.append(f"fine-mesh GCI {gf} above the recipe's cap {rcp['cap_rel']}")
                if why:
                    return {"verdict": "FAIL", "reasons": [f"GCI of '{spec.get('gci_from')}' not admissible under "
                                                           f"the prescribed recipe: {'; '.join(why)}"]}
                g = float(gf)
                gfrac = min(float(rcp["max_fraction"]), GCI_MAX_FRACTION)
            tol_fixed = {k: tol[k] for k in ("abs", "rel") if k in tol}
            if not tol_fixed:
                tol_fixed = {"abs": 0.0}
            val = qty[spec["quantity"]]
            kind = q.get("kind", "scalar")
            if kind == "profile":
                if g is not None:
                    return {"verdict": "FAIL", "reasons": ["a profile comparison cannot add a GCI"]}
                return compare_profile(val, q, tol_fixed)
            if kind == "periodic":
                if not isinstance(val, dict):
                    return {"verdict": "FAIL", "reasons": ["periodic quantity must be {'t': [...], 'signal': [...]}"]}
                st = periodic_statistics(val.get("t"), val.get("signal"), q["definition"])
                if st["reasons"]:
                    return {"verdict": "FAIL", "reasons": st["reasons"], "statistics": st}
                c = compare(st["value"], q, tol_fixed, g, gfrac)
                c["statistics"] = {k: v for k, v in st.items() if k != "reasons"}
                return c
            if kind == "feature":
                ours = extract_feature(val if isinstance(val, dict) else {}, q["definition"])
                theirs = extract_feature(q.get("signal") or {}, q["definition"])
                if ours["reasons"] or theirs["reasons"]:
                    return {"verdict": "FAIL", "reasons": ours["reasons"] + [f"reference signal: {r}" for r in
                                                                             theirs["reasons"]]}
                c = compare(ours["output"], {"value": theirs["output"], "uncertainty": q.get("uncertainty")},
                            tol_fixed, g, gfrac)
                c["feature"] = {"ours": {k: v for k, v in ours.items() if k != "reasons"},
                                "reference": {k: v for k, v in theirs.items() if k != "reasons"}}
                return c
            if kind == "species":
                return compare_species(val, q, tol_fixed)
            if kind == "complex":
                return compare_complex(val, q, tol_fixed)
            if kind == "point":
                if not isinstance(val, dict) or "value" not in val or "at" not in val:
                    return {"verdict": "FAIL", "reasons": ["point functional must be {'value': v, 'at': {...}}"]}
                at_tol = float(q.get("at_tol", 1e-12))
                want = q["at"]
                if set(val["at"]) != set(want) or any(abs(float(val["at"][k]) - float(want[k])) > at_tol
                                                      for k in want):
                    return {"verdict": "FAIL", "reasons": [f"evaluated at {val['at']}, the reference is at {want} "
                                                           f"({q.get('frame', 'no frame given')})"]}
                return compare(val["value"], q, tol_fixed, g, gfrac)
            return compare(val, q, tol_fixed, g, gfrac)
        c = guarded(one, spec.get("quantity"))
        c.update({"quantity": spec["quantity"], "reference_key": spec["reference"], "level": spec["level"],
                  "tolerance": comparison_tolerance(case, spec), "informative": bool(spec.get("informative")),
                  "source": (rq.get(spec["reference"]) or {}).get("source"),
                  "locator": (rq.get(spec["reference"]) or {}).get("locator")})
        comps.append(c)
        mark(spec["level"], c["verdict"] == "PASS", spec.get("informative", False))
    for spec in acc.get("checks", []) or []:
        q = spec["quantity"]
        if spec.get("kind") == "identity":
            c = guarded(lambda: check_identity(qty, spec), q)
        elif spec.get("kind") == "bounds":
            c = guarded(lambda: check_two_sided(qty, spec), q)
        else:
            c = guarded(lambda: check_bound(qty[q], spec) if q in qty else
                        {"quantity": q, "verdict": "FAIL", "reasons": ["quantity missing from the case result"]}, q)
        c.update({"quantity": q, "level": spec["level"]})
        checks.append(c)
        mark(spec["level"], c["verdict"] == "PASS")
    for spec in acc.get("solver_sensitivity", []) or []:
        def one_s():
            a, b = float(qty[spec["quantity"]]), float(qty[spec["tight_quantity"]])
            d = abs(a - b)
            lims = []                    # every bound given applies
            if "abs_max" in spec:
                lims.append(float(spec["abs_max"]))
            if "rel_max" in spec:
                lims.append(float(spec["rel_max"]) * abs(b))
            if not lims:
                raise ValueError("solver_sensitivity needs abs_max and/or rel_max")
            lim = min(lims)
            return {"value": a, "tight_value": b, "difference": d, "bound": lim,
                    "verdict": "PASS" if math.isfinite(d) and d <= lim else "FAIL"}
        c = guarded(one_s, spec.get("quantity"))
        c.update({"quantity": spec["quantity"], "level": spec["level"]})
        sens.append(c)
        mark(spec["level"], c["verdict"] == "PASS")
    extra = {}
    for spec in acc.get("tolerance_behaviour", []) or []:
        a = guarded(lambda: assess_tolerance_behaviour(series.get(spec["series"]), spec, params), spec.get("id"))
        a.update({"id": spec.get("id"), "level": spec["level"]})
        extra.setdefault("tolerance_behaviour", []).append(a)
        mark(spec["level"], a["verdict"] == "PASS")
    for spec in acc.get("quadrature_sensitivity", []) or []:
        def one_q():
            a, b = float(qty[spec["quantity"]]), float(qty[spec["doubled_quantity"]])
            lims = [float(spec["abs_max"])] if "abs_max" in spec else []
            lims += [float(spec["rel_max"]) * abs(b)] if "rel_max" in spec else []
            if not lims:
                raise ValueError("quadrature_sensitivity needs abs_max and/or rel_max")
            d = abs(a - b)
            return {"value": a, "doubled_value": b, "difference": d, "bound": min(lims),
                    "verdict": "PASS" if math.isfinite(d) and d <= min(lims) else "FAIL"}
        c = guarded(one_q, spec.get("quantity"))
        c.update({"quantity": spec["quantity"], "level": spec["level"]})
        extra.setdefault("quadrature_sensitivity", []).append(c)
        mark(spec["level"], c["verdict"] == "PASS")
    solv = guarded(lambda: {"problems": solver_problems(runs, acc.get("solver") or {})}, "solver")
    solver_issues = solv.get("problems", solv.get("reasons", []))
    tested = [v for v in verdict.values() if v != "NOT_TESTED"]
    all_pass = bool(tested) and all(v == "PASS" for v in tested) and not solver_issues
    xs = [x for v in extra.values() for x in v]
    fails = [f"{x.get('series') or x.get('quantity') or x.get('id')}" for x in conv + fconv + comps + checks + sens + xs
             if x.get("verdict") != "PASS" and not x.get("informative")]
    if solver_issues:
        fails.append("solver")
    out = {"convergence": conv, "functional_convergence": fconv, "comparisons": comps, "checks": checks,
           "solver_sensitivity": sens, "solver_problems": solver_issues, "verdicts": verdict,
           "criteria_met": all_pass,
           "headline": "all criteria met" if all_pass else f"FAILED: {', '.join(fails) or 'nothing tested'}"}
    if extra:              # only when used, so that the result of a case without these items keeps its shape 
        out["extra"] = extra
    return out


# ----------------------------------------------------------------- MMS / exact-solution consistency (finite differences)
def _fd(f: Callable, x: np.ndarray, i: int, h: float) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[..., i] = h
    return (f(x + dx) - f(x - dx)) / (2 * h)


def gradient_fd(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of a scalar field f(x) (x: (..., d)); returns (..., d)."""
    x = np.asarray(x, float)
    return np.stack([_fd(f, x, i, h) for i in range(x.shape[-1])], axis=-1)


def laplacian_fd(f: Callable, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, float)
    out = -2 * x.shape[-1] * f(x)
    for i in range(x.shape[-1]):
        dx = np.zeros_like(x)
        dx[..., i] = h
        out = out + f(x + dx) + f(x - dx)
    return out / h ** 2


def divergence_fd(u: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """div u for a vector field u(x) -> (..., d)."""
    x = np.asarray(x, float)
    return sum(_fd(lambda y, i=i: u(y)[..., i], x, i, h) for i in range(x.shape[-1]))


def ns_residual_fd(u: Callable, p: Callable, f: Callable, nu: float, x: np.ndarray, convective: bool = True,
                   h: float = 1e-4) -> dict:
    """Pointwise residuals of steady incompressible Navier-Stokes (Stokes if convective=False),
        (u.grad)u + grad p - nu lap u - f   and   div u,
    by central differences.  Use it to check a hand-derived MMS forcing before trusting an MMS case (a wrong
    source gives a clean but wrong convergence study).  Returns max-norms scaled by the largest term."""
    x = np.asarray(x, float)
    d = x.shape[-1]
    U = u(x)
    mom = -f(x) + gradient_fd(p, x, h)
    scale = np.max(np.abs(gradient_fd(p, x, h)))
    for i in range(d):
        ui = (lambda y, i=i: u(y)[..., i])
        lap = laplacian_fd(ui, x, h)
        mom[..., i] -= nu * lap
        scale = max(scale, np.max(np.abs(nu * lap)))
        if convective:
            conv = np.sum(U * gradient_fd(ui, x, h), axis=-1)
            mom[..., i] += conv
            scale = max(scale, np.max(np.abs(conv)))
    div = divergence_fd(u, x, h)
    return {"momentum_max": float(np.max(np.abs(mom))), "divergence_max": float(np.max(np.abs(div))),
            "scale": float(scale), "momentum_rel": float(np.max(np.abs(mom)) / max(scale, 1e-300))}



# ---------------------------------------------------------------------------------------------------- extraction
