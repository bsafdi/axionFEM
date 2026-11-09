"""Global (differential-evolution) design optimization.

:func:`global_optimize` searches a box of named design parameters with scipy's
``differential_evolution``.  Every evaluation is one mesh and one solve of the design at those
parameters; figures of merit that refer to a noise benchmark (``cost``, ``cost_optimal``,
``W_at_req``, ...) are taken at the amplitude that reaches F = F_req, the same convention as the
gradient-based :func:`axionfem.sensitivity.optimize`.  For designs with nonlinear iron, where a
rescaling of the sources is not exact, the amplitude parameter is solved for (log-secant on F) so
that F = F_req holds to ``amplitude_tol`` before the quantities are read.

Shapes that fail to mesh or solve, and evaluations that exceed ``timeout``, are infeasible points,
not errors.  Constraints use the feasibility rule (Deb 2000): a feasible point beats any infeasible
one, infeasible points are ranked by their total relative violation, failed points rank last.  The
best point is then polished locally, with the adjoint-gradient :func:`axionfem.optimize` when the
design supports it (linear materials, fixed mesh topology) and otherwise with bounded Powell.

:func:`optimize` is the package-level entry point: ``method="global"`` selects this module, every
other value is passed to the gradient-based optimizer unchanged.
"""
from __future__ import annotations

import inspect
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Ranks of non-feasible energies seen by differential_evolution (see energy() in global_optimize).  A feasible
# objective must satisfy |f| < _MAX_OBJECTIVE so that it always ranks before every infeasible point.
_INFEASIBLE = 1e100
_FAILED = 1e150
_MAX_OBJECTIVE = 1e99


class _ConfigError(ValueError):
    """An error of the run's configuration (not of the point): the search stops at the first one."""


# ---------------------------------------------------------------- quantities
def _split(name: str, default_bench: str) -> tuple[str, str]:
    base, _, bench = name.partition(":")
    return base, bench or default_bench


def _bench(res, bench: str) -> dict:
    if bench:
        try:
            return res.benchmark(bench)
        except KeyError as e:
            raise _ConfigError(str(e).strip("'\"")) from None
    b = res.data.get("benchmarks", [])
    if not b:
        raise _ConfigError("the design has no benchmarks, so a requirement quantity cannot be evaluated")
    return b[0]


def quantity(res, name, benchmark: str = "") -> float:
    """Value of a figure of merit (any name of :func:`axionfem.objectives`) from a solved model.

    Mirrors the closed forms of the sensitivity engine (``cpp/src/adjoint.cpp``,
    ``objective_value``) and reads the solver's post-processing for the benchmark quantities, so
    the values agree with ``Results.sensitivities(...)["values"]`` (checked in
    tests/test_global_opt.py).  ``name`` may also be a callable ``f(results) -> float``.
    """
    if callable(name):
        return float(name(res))
    base, bench = _split(str(name), benchmark)
    d = res.data
    c, g, f = d.get("coupling", {}), d.get("geometry", {}), d.get("fields", {})
    prim = {"Psi": c.get("Psi_SI_H_T_m2", 0.0), "L_p": c.get("L_p_H", 0.0), "W_mag": c.get("W_mag_J", 0.0),
            "A_cold": g.get("A_cold_m2", 0.0), "iron_mass": g.get("iron_mass_kg", 0.0),
            "device_radius": c.get("device_radius_m", 0.0), "B_peak": f.get("B_max_T", 0.0),
            "sc_surface_peak": max([0.0, *f.get("superconductor_surface_regular_T", {}).values()])}
    if base in prim:
        return float(prim[base])
    Psi, Lp, W, Rd = prim["Psi"], prim["L_p"], prim["W_mag"], prim["device_radius"]
    F = Psi ** 4 / Lp ** 2 if Lp > 0 else 0.0
    if base == "F":
        return F
    if base == "g_factor":
        mu0 = 1.25663706212e-6
        return Psi * Psi / (Lp * W * Rd * Rd) / mu0 ** 2 if (Lp > 0 and W > 0 and Rd > 0) else 0.0
    b = None
    if base in ("cost", "cost_optimal", "W_at_req", "F_over_F_req", "amplitude_scale", "sc_surface_at_req",
                "B_peak_at_req"):
        b = _bench(res, bench)
    if base == "F_over_F_req":
        return float(b.get("F_over_F_req", 0.0))
    if b is not None and "amplitude_scale" not in b:
        raise ValueError(f"the design has F = 0 (no coupling or no pickup), so '{name}' cannot be evaluated")
    if base == "amplitude_scale":
        return float(b["amplitude_scale"])
    if base == "W_at_req":
        return float(b["W_mag_at_req_J"])
    if base == "B_peak_at_req":
        return float(b["B_max_at_req_T"])
    if base == "sc_surface_at_req":
        return float(max([0.0, *b.get("superconductor_surface_regular_at_req_T", {}).values()]))
    if base == "cost":
        return float(b["cost"]["total_MCHF"])
    if base == "cost_optimal":
        opt = b.get("cost_optimal_size")
        if not opt:
            raise _ConfigError("'cost_optimal' is reported only for iron-free designs; use 'cost'")
        return float(opt["cost"]["total_MCHF"])
    from .sensitivity import objectives

    raise _ConfigError(f"unknown quantity {name!r}; available: {sorted(objectives())} (or a callable)")


def _qname(q) -> str:
    return q if isinstance(q, str) else getattr(q, "__name__", "custom")


def _quantity_keys(quantities: list) -> list[str]:
    """One distinct key per quantity, used for ``values``: a named quantity is its own key; a callable gets its
    ``__name__`` (``"custom"`` without one), suffixed ``#2``, ``#3`` ... when that key is already taken (two lambdas,
    two ``functools.partial`` objects, or a callable named like a quantity)."""
    keys: list[str] = []
    taken = {q for q in quantities if isinstance(q, str)}
    for q in quantities:
        if isinstance(q, str):
            keys.append(q)
            continue
        k, i = _qname(q), 2
        while k in taken:
            k, i = f"{_qname(q)}#{i}", i + 1
        taken.add(k)
        keys.append(k)
    return keys


# ---------------------------------------------------------------- one evaluation (picklable)
# quantities that refer to a noise benchmark (read at the amplitude that reaches F = F_req)
REQUIREMENT = ("cost", "cost_optimal", "W_at_req", "F_over_F_req", "amplitude_scale", "sc_surface_at_req",
               "B_peak_at_req")


def _bench_name(res, bench: str) -> str:
    return bench or (res.data.get("benchmarks") or [{}])[0].get("name", "")


def _check_newton(res) -> None:
    """A non-converged nonlinear solve is a failed evaluation, not a value (the solver only warns)."""
    for c in res.data.get("checks", []):
        if c.get("name") == "static poloidal solve" and c.get("converged") is False:
            rr = c.get("relative_residual")
            rr = f"{rr:.3g}" if isinstance(rr, (int, float)) else str(rr)
            raise RuntimeError(f"static poloidal Newton solve did not converge ({c.get('newton_iterations')} "
                               f"iterations, relative residual {rr})")


def _solve(design, task: dict, wd: Path):
    res = design.solve(workdir=wd, order=task["order"], verbose=0, self_force=False, force_samples=0)
    _check_newton(res)
    return res


def _solve_at_req(design, task: dict, wd: Path):
    """Solve; when a requirement quantity is asked for and amplitude scaling is not exact (nonlinear iron,
    or fixed sources such as a background field), iterate the amplitude parameter until F = F_req."""
    res = _solve(design, task, wd)
    req = [q for q in task["quantities"] if callable(q) or _split(str(q), "")[0] in REQUIREMENT]
    if not req:
        return res, None
    bench = _bench_name(res, task["amp_bench"])
    named = [q for q in req if not callable(q)]
    try:
        entry = _bench(res, bench)
    except _ConfigError:
        if not named and res.data.get("linear", True):
            return res, None                   # callables only, no benchmark, exact scaling: nothing to solve for
        raise
    if res.data.get("linear", True) and "note" not in entry:
        return res, None                       # exact rescaling of the post-processing
    benches = {_bench_name(res, _split(str(q), task["benchmark"])[1]) for q in named}
    if benches - {bench}:
        raise _ConfigError(f"amplitude scaling is not exact for this design, so all requirement quantities must "
                           f"refer to one benchmark; got {sorted(benches | {bench})}")
    par = task["amplitude"]
    p0 = design.resolved_params()
    if par == "auto":
        par = "NI" if "NI" in p0 else None
    if not par or par not in p0:
        raise _ConfigError("amplitude scaling is not exact for this design (nonlinear iron or fixed sources), so the "
                           "amplitude must be solved for F = F_req: give amplitude='<parameter that scales every "
                           "source>' (e.g. 'NI')")
    if par in task.get("free", ()):
        raise _ConfigError(f"the amplitude parameter {par!r} is also optimized; it is solved for F = F_req, so "
                           f"remove it from params or give amplitude=<another parameter>")
    x, hist = float(p0[par]), []
    if not x > 0:
        raise ValueError(f"amplitude parameter {par!r} must be positive (got {x})")
    nmax, tol = int(task["amplitude_max_solves"]), task["amplitude_tol"]
    if nmax < 1:
        raise _ConfigError("amplitude_max_solves must be at least 1")
    for it in range(nmax):
        r = float(_bench(res, bench).get("F_over_F_req", 0.0))
        hist.append((x, r))
        if not (r > 0 and math.isfinite(r)):
            raise ValueError(f"F/F_req = {r} at {par} = {x}: cannot solve for the amplitude")
        if abs(r - 1.0) <= tol:
            return res, {"parameter": par, "value": x, "F_over_F_req": r, "solves": len(hist), "benchmark": bench}
        if it == nmax - 1:
            break
        if len(hist) >= 2 and hist[-1][1] != hist[-2][1] and hist[-1][0] != hist[-2][0]:
            (x0, r0), (x1, r1) = hist[-2], hist[-1]
            slope = math.log(r1 / r0) / math.log(x1 / x0)          # d log F / d log amplitude
            slope = min(max(slope, 0.5), 12.0)
        else:
            slope = 4.0                                             # F ~ amplitude^4 without saturation
        x = x * r ** (-1.0 / slope)
        design = design.with_params(**{par: x})
        res = _solve(design, task, wd)
    raise RuntimeError(f"amplitude root for F = F_req did not converge in {nmax} solves "
                       f"(last F/F_req = {hist[-1][1]:.6g} at {par} = {hist[-1][0]:.6g})")


def _evaluate(task: dict) -> dict:
    """Mesh, solve and read the requested quantities at one parameter vector (runs in a worker)."""
    from .design import Design

    t0 = time.monotonic()
    rec: dict = {"parameters": dict(task["parameters"])}
    wd = Path(task["workdir"])
    try:
        if task.get("threads"):
            from . import set_threads

            set_threads(int(task["threads"]))     # also in a map run elsewhere (spawn pool, SLURM array)
        quantities = task["quantities"]
        if "quantities_pickle" in task:            # callables, unpickled here so that a failure is reported as such
            import pickle

            try:
                quantities = pickle.loads(task["quantities_pickle"])
            except Exception as e:  # noqa: BLE001
                raise _ConfigError(f"a callable objective/constraint cannot be loaded in the worker process "
                                   f"({type(e).__name__}: {e}); define it at module level of an importable module "
                                   f"(not in __main__ of a notebook or 'python -c'), or use workers=1") from None
        task = {**task, "quantities": quantities}
        keys = task.get("keys") or [_qname(q) for q in quantities]
        d = Design(task["doc"]).with_params(**task["parameters"])
        res, amp = _solve_at_req(d, task, wd)
        bench = task["benchmark"]
        rec["values"] = {k: quantity(res, q, bench) for k, q in zip(keys, quantities)}
        rec["caps"] = _caps(res, task["amp_bench"])
        rec["linear"] = bool(res.data.get("linear", True))
        if amp:
            rec["amplitude"] = amp
    except Exception as e:  # noqa: BLE001  (an infeasible shape must not stop the search)
        rec.pop("values", None)
        rec["error"] = f"{type(e).__name__}: {e}"
        if isinstance(e, _ConfigError):
            rec["config_error"] = str(e)
    rec["seconds"] = round(time.monotonic() - t0, 3)
    if not task.get("keep_runs"):
        shutil.rmtree(wd, ignore_errors=True)
    return rec


def _caps(res, benchmark: str) -> dict:
    try:
        b = _bench(res, benchmark)
    except (KeyError, ValueError, IndexError):
        return {}
    return b.get("caps_at_req", {})


def _init_worker(threads: int) -> None:
    from . import set_threads

    set_threads(threads)


def _descendants(pid: int) -> list[int]:
    """All descendant process ids (Linux /proc), e.g. an axf-mesher started by a worker."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as fh:
                    kids = [int(c) for c in fh.read().split()]
                out += kids
                todo += kids
        except OSError:
            pass
    return out


# ---------------------------------------------------------------- process pool with timeouts
class _Pool:
    """Persistent spawn-context worker pool; a hung or crashed evaluation becomes a failed record.

    When a worker dies, every point that was running is re-run alone, so that the crash is charged to the
    point that caused it and not to its neighbours."""

    def __init__(self, n: int, threads: int, timeout: float | None, fn: Callable | None = None):
        self.n, self.threads, self.timeout = n, threads, timeout
        self.fn = fn or _evaluate          # module-level, picklable (tests substitute a stand-in)
        self.ex = None
        self._new()

    def _new(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self.ex = ProcessPoolExecutor(max_workers=self.n, mp_context=multiprocessing.get_context("spawn"),
                                      initializer=_init_worker, initargs=(self.threads,))

    def _kill(self):
        import signal

        procs = list((getattr(self.ex, "_processes", None) or {}).values())
        for p in procs:
            for c in _descendants(p.pid):        # the mesher is a child of the worker: do not orphan it
                try:
                    os.kill(c, signal.SIGKILL)
                except OSError:
                    pass
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        self.ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.join(5)
            except Exception:  # noqa: BLE001
                pass

    def _dead_worker(self) -> bool:
        for p in list((getattr(self.ex, "_processes", None) or {}).values()):
            try:
                if p.exitcode is not None:
                    return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None

    def map(self, tasks: list[dict]) -> list[dict]:
        from collections import deque
        from concurrent.futures import FIRST_COMPLETED, wait
        from concurrent.futures.process import BrokenProcessPool

        out: list[dict | None] = [None] * len(tasks)
        pending = deque(range(len(tasks)))
        solo: deque = deque()          # points that were running when a worker died: re-run one at a time
        running: dict = {}

        def over(i, r) -> bool:
            """A finished evaluation that itself ran longer than the timeout is charged as one."""
            if self.timeout and float(r.get("seconds") or 0.0) > self.timeout:
                fail(i, f"evaluation exceeded timeout = {self.timeout:g} s")
                return True
            return False

        def fail(i, msg):
            out[i] = {"parameters": dict(tasks[i]["parameters"]), "error": msg}
            if not tasks[i].get("keep_runs"):            # the killed worker could not clean up its workdir
                shutil.rmtree(tasks[i]["workdir"], ignore_errors=True)

        def submit(i, alone):
            try:
                running[self.ex.submit(self.fn, tasks[i])] = (i, time.monotonic(), alone)
            except BrokenProcessPool:
                self._kill()
                self._new()
                running[self.ex.submit(self.fn, tasks[i])] = (i, time.monotonic(), alone)

        while pending or running or solo:
            if solo:
                if not running:
                    submit(solo.popleft(), True)
            else:
                while pending and len(running) < self.n:
                    submit(pending.popleft(), False)
            done, _ = wait(list(running), timeout=1.0, return_when=FIRST_COMPLETED)
            broken = False
            for fut in done:
                i, _t, alone = running.pop(fut)
                try:
                    out[i] = fut.result()
                    over(i, out[i])
                except BrokenProcessPool:
                    broken = True
                    if alone:
                        fail(i, "the worker process died on this point (crash in the mesher or solver, or out of "
                                "memory)")
                    else:
                        solo.append(i)
                except Exception as e:  # noqa: BLE001  (e.g. an objective that cannot be unpickled)
                    fail(i, f"{type(e).__name__}: {e}")
                if out[i] is not None and out[i].get("config_error"):   # same for every point: stop now
                    self._kill()
                    self._new()
                    raise _ConfigError(out[i]["config_error"])
            if not broken and running and self._dead_worker():
                # CPython may notice a dead worker only when a sibling finishes: check the processes directly
                broken = True
                for f, (i, _t, alone) in list(running.items()):
                    if alone and not f.done():
                        fail(i, "the worker process died on this point (crash in the mesher or solver, or out of "
                                "memory)")
                        running.pop(f)
            late = []
            if self.timeout:
                now = time.monotonic()
                late = [f for f, (_i, t, _a) in running.items() if now - t > self.timeout and not f.done()]
                for f in late:
                    fail(running[f][0], f"evaluation exceeded timeout = {self.timeout:g} s")
            if broken or late:
                # the pool is unusable (or a worker is stuck): restart it; points still running are re-run
                for f, (i, _t, alone) in running.items():
                    if f in late:
                        continue
                    if f.done():                     # finished meanwhile: keep its result if it has one
                        try:
                            out[i] = f.result()
                            over(i, out[i])
                            continue
                        except Exception:  # noqa: BLE001
                            pass
                    (solo if broken else pending).appendleft(i)
                running = {}
                self._kill()
                self._new()
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------- the optimizer
def _parse_params(params, log) -> tuple[list[str], list[tuple[float, float]], list[bool]]:
    if not isinstance(params, dict) or not params:
        raise ValueError("give params={'name': (low, high), ...} with at least one parameter and its bounds")
    log = set(log or ())
    names, bnds, islog = [], [], []
    for k, v in params.items():
        v = tuple(v)
        if len(v) == 3:
            if v[2] not in ("log", "lin"):
                raise ValueError(f"bounds for '{k}': the third entry must be 'log' or 'lin', got {v[2]!r}")
            if v[2] == "log":
                log.add(k)
            v = v[:2]
        if len(v) != 2:
            raise ValueError(f"bounds for '{k}' must be (low, high) or (low, high, 'log')")
        lo, hi = float(v[0]), float(v[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"bounds for '{k}' must be finite with low < high, got ({lo}, {hi})")
        if k in log and lo <= 0:
            raise ValueError(f"log-scaled parameter '{k}' needs low > 0, got {lo}")
        names.append(k)
        bnds.append((lo, hi))
    unknown = log - set(names)
    if unknown:
        raise ValueError(f"log= names parameters that are not optimized: {sorted(unknown)}")
    return names, bnds, [n in log for n in names]


def _parse_constraints(constraints) -> list[dict]:
    out = []
    for c in constraints or ():
        if isinstance(c, (tuple, list)):
            if len(c) != 3:
                raise ValueError(f"a constraint tuple is (quantity, '<=' or '>=', value), got {c!r}")
            c = {"quantity": c[0], "type": c[1], "value": c[2]}
        c = dict(c)
        if "quantity" not in c or "value" not in c:
            raise ValueError(f"a constraint needs 'quantity', 'type' and 'value', got {c!r}")
        if c.get("type") not in ("<=", ">="):
            raise ValueError("constraint type must be '<=' or '>='")
        c["value"] = float(c["value"])
        out.append(c)
    return out


class _Stop(Exception):
    """Raised inside the Powell polish when ``max_time`` is exhausted."""


class _Abort(Exception):
    """Carries an error out of scipy's map call (scipy re-raises a TypeError/ValueError there as RuntimeError)."""


def _population_size(popsize: int, n: int, init) -> int:
    """Population size scipy uses: max(5, popsize * n), rounded up to a power of 2 for Sobol."""
    if not isinstance(init, str):
        return len(init)
    m = max(5, popsize * n)
    if init == "sobol":
        m = 1 << (m - 1).bit_length()
    return m


def global_optimize(design, objective: Any = "cost", params: dict | None = None, *,
                    bounds: dict | None = None, constraints: Iterable = (), sense: str = "min",
                    log: Sequence[str] = (), seed: int | None = 0, popsize: int = 15, maxiter: int = 50,
                    tol: float = 0.01, atol: float = 0.0, mutation: Any = (0.5, 1.0), recombination: float = 0.7,
                    strategy: str = "best1bin", init: Any = "latinhypercube", x0: dict | None = None,
                    workers: Any = 1, threads: int = 1, timeout: float | None = None,
                    max_time: float | None = None, polish: Any = True, polish_max_iter: int = 30,
                    polish_maxfev: int = 200, benchmark: str = "", amplitude: str | None = "auto",
                    amplitude_tol: float = 1e-6, amplitude_max_solves: int = 16,
                    workdir: str | Path | None = None, order: int = 0, keep_runs: bool = False,
                    precheck: Callable | None = None, verbose: int = 1, callback: Callable | None = None) -> dict:
    """Global minimization (or maximization) of a figure of merit by differential evolution.

    ``params`` (or its alias ``bounds``; not both) maps parameter names to ``(low, high)`` or
    ``(low, high, "log")``; ``log`` lists parameters searched in log space (use it for sizes and ratios
    spanning decades).  ``objective`` is a quantity name (:func:`axionfem.objectives`, e.g. ``"cost"``,
    ``"cost_optimal:conservative"``, ``"g_factor"``) or a picklable callable ``f(results) -> float``
    (importable by name in a worker when ``workers`` > 1 or a map is given).  ``constraints`` holds
    ``{"quantity": "sc_surface_at_req", "type": "<=", "value": 0.093}`` entries (or 3-tuples); they are
    enforced by the feasibility rule, not by a penalty weight.  ``values`` are keyed by the quantity name;
    a callable by its ``__name__`` (``"custom"`` without one), suffixed ``#2``, ``#3`` ... when that key is
    already taken (two lambdas, or a callable named like a quantity).  A feasible objective must have
    ``|f| < 1e99`` (larger values are failed points: rescale).

    DE settings are scipy's: the population has ``max(5, popsize * len(params))`` members (rounded up to
    a power of 2 for ``init="sobol"``; an ``init`` array is given in parameter units, one row per
    member, inside the bounds), at most ``maxiter`` generations after the initial one.  The search stops
    when the whole population is feasible and ``std(f) <= atol + tol * |mean(f)|`` (scipy >= 1.12; older
    scipy runs all ``maxiter`` generations).  It is deterministic for a given ``seed``: every evaluation
    (serial, pool or map) and the polish run on ``threads`` OpenMP threads (default 1; the calling
    process is set to it and restored on return), so that the solver's parallel sums do not depend on
    the thread count.  ``workers`` > 1 evaluates each generation on a process pool (``-1``: the CPUs this
    process may use, divided by ``threads``); a callable ``workers(fn, tasks) -> records`` (a map)
    distributes evaluations elsewhere, e.g. a SLURM array.  Scripts that use a process pool must guard
    their entry point with ``if __name__ == "__main__":``.

    Failure handling.  A shape that fails to mesh or solve is an infeasible point.  With a process pool
    (``workers`` > 1) also: an evaluation slower than ``timeout`` [s] (killed when still running
    ``timeout`` s after submission, which includes a worker's start-up after a restart, then the pool is
    restarted and the other running points are re-run; a finished one whose own run time exceeds
    ``timeout`` is charged too) and a worker that dies (the points running with it are re-run one at a time, so the
    crash is charged to the point that caused it).  In serial mode (``workers=1``) ``timeout`` is
    ignored (with a warning) and a hard crash of the solver ends the run.  Nothing is saved before the
    run returns.  An error of the configuration (an unknown benchmark, ``cost_optimal`` on an iron
    design, a nonlinear design without an amplitude parameter, ``NI`` optimized while it is the
    amplitude, a callable that cannot be loaded in a worker, ...) is raised as ``ValueError`` at the
    first evaluation that shows it, instead of failing every point.

    ``max_time`` [s] is checked after every generation (not inside one, nor during the initial
    population): the search then stops, the polish is skipped if the time is exhausted, the Powell
    polish stops when it is, and the adjoint polish (which cannot be interrupted; at most
    ``polish_max_iter`` iterations) is replaced by Powell unless ``7 (polish_max_iter + 1)`` evaluations
    of the DE's mean duration fit in the remaining time (an iteration takes several solves).  ``max_time``
    is thus a soft bound: the last generation, a Powell evaluation, or an adjoint polish that needs more
    solves than this estimate can overrun it.  ``callback(generation, best_record)`` returning
    True stops the search too.

    ``precheck(parameters)`` rejects a point before it is meshed: a returned string (the reason, also an
    empty one), ``False`` (also ``numpy.False_``) or a raised exception rejects it, ``None`` or ``True``
    (also ``numpy.True_``) accepts it; any other return value is an error.  Use it for cheap geometric
    rules (wall thickness, corner radii).  ``polish="adjoint"`` cannot apply it and is refused with a
    precheck.

    Requirement quantities (``cost``, ``cost_optimal``, ``W_at_req``, ``sc_surface_at_req``, ...) are
    read at F = F_req: by the solver's exact rescaling for linear designs, as by
    :func:`axionfem.optimize`; when the rescaling is not exact (nonlinear iron, or fixed sources such
    as a background field) the parameter ``amplitude`` (default ``"NI"``) is solved for F = F_req of the
    objective's benchmark to ``amplitude_tol`` (log-secant, at most ``amplitude_max_solves`` solves), and
    all requirement quantities must then refer to that benchmark.  A nonlinear solve whose Newton
    iteration did not converge is a failed point.  Known limitation: the amplitude at which the other
    quantities are read differs by case.  When the root runs (a requirement quantity or any callable
    is asked for on a nonlinear design or one with fixed sources), every quantity and callable -- also
    ``B_peak``, ``W_mag``, ``sc_surface_peak`` -- is read at the root amplitude; on a linear design
    without fixed sources, non-requirement quantities and callables see the design's own ``NI``.  A
    constraint such as ``("B_peak", "<=", 6)`` therefore means different things in the two cases; use
    the ``*_at_req`` quantities for limits at the operating point.  A callable on a nonlinear design
    needs the amplitude parameter and costs the ~3 solves of the root.

    ``polish``: ``True``/``"auto"`` refines the best point with the adjoint-gradient optimizer when
    the design is linear without fixed sources (no amplitude root), the objective and constraints are
    named quantities, no ``precheck`` is given and ``order`` is 0, else with bounded Powell;
    ``"adjoint"`` (falling back to Powell if the adjoint optimizer fails or the point needed the
    amplitude root), ``"powell"`` or ``False`` choose.  The polished point replaces the DE point only
    if it is feasible and better.

    Returns the structure of :func:`axionfem.optimize` (``parameters``, ``values``, ``caps``,
    ``constraints``, ``design``, ``history`` of every evaluation with ``stage``, ``generation``,
    ``energy`` and ``feasible``), plus ``feasible``, ``de`` (generations, evaluations, convergence,
    scipy's own message), ``de_best``, ``polish`` and ``found_by``.  ``converged`` is the DE convergence
    test above and ``message`` says how the search ended.  ``evaluations`` is ``len(history)`` (DE and
    Powell points, adjoint iterates, precheck rejections); ``solves`` counts the FE solves (one per
    evaluated point, plus the amplitude-root and adjoint-polish solves).  The default ``workdir`` is a
    new directory per run under ``runs/<design>__global/``.
    """
    import pickle

    import numpy as np
    from scipy.optimize import differential_evolution, minimize

    from . import _core, set_threads
    from .design import Design
    from .sensitivity import objectives as _catalog

    t_start = time.monotonic()
    base = Design(design)
    if params is not None and bounds is not None:
        raise ValueError("give params= or bounds= (the same thing), not both")
    given = params if params is not None else bounds
    names, bnds, islog = _parse_params(given, log)
    if isinstance(workers, bool) or not (callable(workers) or isinstance(workers, (int, np.integer))):
        raise ValueError("workers must be a positive integer, -1, or a map-like callable workers(fn, tasks)")
    if not callable(workers):
        workers = int(workers)
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    threads = int(threads)
    if isinstance(popsize, bool) or not isinstance(popsize, (int, np.integer)) or popsize < 1:
        raise ValueError(f"popsize must be a positive integer, got {popsize!r}")
    for nm, v, ok, need in (("tol", tol, tol >= 0, ">= 0"), ("atol", atol, atol >= 0, ">= 0"),
                            ("amplitude_tol", amplitude_tol, amplitude_tol > 0, "> 0"),
                            ("amplitude_max_solves", amplitude_max_solves, amplitude_max_solves >= 1, ">= 1")):
        if not (ok and math.isfinite(v)):
            raise ValueError(f"{nm} must be finite and {need}, got {v!r}")
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    if polish not in (True, False, None, "auto", "adjoint", "powell"):
        raise ValueError(f"polish must be True, False, 'auto', 'adjoint' or 'powell', got {polish!r}")
    cons = _parse_constraints(constraints)
    known = set(_catalog())
    for q in [objective, *[c["quantity"] for c in cons]]:
        if not callable(q) and _split(str(q), "")[0] not in known:
            raise ValueError(f"unknown quantity {q!r}; available: {sorted(known)} (or a callable)")
    p0 = base.resolved_params()
    missing = [n for n in names if n not in p0]
    if missing:
        raise KeyError(f"unknown parameters {missing}; known: {sorted(p0)}")
    if amplitude and amplitude != "auto" and amplitude not in p0:
        raise KeyError(f"amplitude parameter {amplitude!r} is not a parameter of the design; known: {sorted(p0)}")
    if amplitude and amplitude != "auto" and amplitude in names:
        raise ValueError(f"the amplitude parameter {amplitude!r} cannot also be optimized; it is solved for F = F_req")
    if polish == "adjoint" and precheck is not None:
        raise ValueError("polish='adjoint' cannot apply the precheck before meshing (the adjoint optimizer meshes its "
                         "iterates itself); use polish='powell' or 'auto' with a precheck")
    sgn = 1.0 if sense == "min" else -1.0
    quantities = list(dict.fromkeys([objective, *[c["quantity"] for c in cons]]))
    qkeys = _quantity_keys(quantities)
    key_of = {id(q): k for q, k in zip(quantities, qkeys)}

    def qkey(q) -> str:
        return q if isinstance(q, str) else key_of[id(q)]

    oname = qkey(objective)
    amp_bench = _split(objective, benchmark)[1] if isinstance(objective, str) else benchmark
    # default: a directory of its own per run (concurrent runs of one design must not share e000000...)
    wd = Path(workdir) if workdir else (Path("runs") / f"{base.name}__global"
                                        / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{os.urandom(2).hex()}")
    wd.mkdir(parents=True, exist_ok=True)
    lo = np.array([math.log(b[0]) if g else b[0] for b, g in zip(bnds, islog)])
    hi = np.array([math.log(b[1]) if g else b[1] for b, g in zip(bnds, islog)])

    def to_params(u) -> dict:
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        out = {}
        for i, n in enumerate(names):
            if u[i] == 0.0:                   # exact endpoints (exp(log(x)) can differ from x in the last bit)
                x = bnds[i][0]
            elif u[i] == 1.0:
                x = bnds[i][1]
            else:
                v = lo[i] + u[i] * (hi[i] - lo[i])
                x = math.exp(v) if islog[i] else v
            out[n] = float(x)
        return out

    def to_unit(p: dict, what: str = "the polish start"):
        for n, (a, b) in zip(names, bnds):
            x = float(p[n])
            eps = 1e-12 * max(abs(a), abs(b), 1.0)   # adjoint iterates can overshoot an active bound in the last bit
            if not (math.isfinite(x) and a - eps <= x <= b + eps):
                raise ValueError(f"{what}: {n} = {x} is outside the bounds ({a}, {b})")
        v = np.array([math.log(p[n]) if g else p[n] for n, g in zip(names, islog)], float)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    if workers == -1:
        try:
            ncpu = len(os.sched_getaffinity(0))
        except AttributeError:
            ncpu = os.cpu_count() or 1
        workers = max(ncpu // threads, 1)
    if isinstance(workers, int) and workers < 1:
        raise ValueError("workers must be a positive integer, -1, or a map-like callable workers(fn, tasks)")
    use_pool = isinstance(workers, int) and workers > 1
    if timeout is not None and not use_pool:
        import warnings

        warnings.warn("timeout is applied only with a process pool (workers > 1); it is ignored here", stacklevel=2)
    callables = [q for q in quantities if callable(q)]
    q_pickle = None
    if callables and not (isinstance(workers, int) and workers == 1):
        try:
            q_pickle = pickle.dumps(quantities)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"a callable objective/constraint cannot be sent to a worker process ({e}); define it "
                             f"at module level of an importable module, or use workers=1") from e
    if x0 is not None:
        extra = set(x0) - set(names)
        if extra:
            raise ValueError(f"x0 names parameters that are not optimized: {sorted(extra)}")
    if not isinstance(init, str):
        rows = np.atleast_2d(np.asarray(init, float))
        if rows.ndim != 2 or rows.shape[1] != len(names):
            raise ValueError(f"an init array needs one column per parameter ({names}), got shape {rows.shape}")
        init = np.array([to_unit(dict(zip(names, row)), f"init row {i}") for i, row in enumerate(rows)])
    npop = _population_size(popsize, len(names), init)

    cache: dict[tuple, dict] = {}
    history: list[dict] = []
    state = {"gen": -1, "stage": "de", "n": 0, "best": None, "stop": False, "converged": False,
             "adjoint_solves": 0}

    def violation(rec) -> float:
        v = 0.0
        for c in cons:
            q = rec["values"].get(qkey(c["quantity"]))
            if q is None or not math.isfinite(q):
                return 1e40
            s = abs(c["value"]) or 1.0
            v += max(0.0, (q - c["value"]) / s) if c["type"] == "<=" else max(0.0, (c["value"] - q) / s)
        return min(v, 1e40)

    def energy(rec) -> float:
        if "values" not in rec:
            return _FAILED
        f = rec["values"].get(oname)
        if f is None or not math.isfinite(f):
            return _FAILED
        v = violation(rec)
        return sgn * f if v <= 0 else _INFEASIBLE * (1.0 + v)

    def rejected(p) -> str | None:
        if precheck is None:
            return None
        try:
            why = precheck(dict(p))
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if why is None:
            return None
        if isinstance(why, str):                       # a reason, even an empty one, rejects
            return why or "precheck returned an empty reason"
        if isinstance(why, (bool, np.bool_)):          # also numpy comparisons such as np.hypot(...) > t
            return None if bool(why) else "precheck returned False"
        raise TypeError(f"precheck must return None/True (accept), or False/a reason string (reject); got "
                        f"{type(why).__name__} {why!r}")

    def task_of(p: dict, idx: int) -> dict:
        t = {"doc": base.doc, "parameters": p, "quantities": quantities, "keys": qkeys, "benchmark": benchmark,
             "amp_bench": amp_bench, "free": names, "order": order, "workdir": str(wd / f"e{idx:06d}"),
             "amplitude": amplitude, "amplitude_tol": amplitude_tol, "threads": threads,
             "amplitude_max_solves": amplitude_max_solves, "keep_runs": keep_runs}
        if q_pickle is not None:      # callables travel as bytes and are unpickled inside the worker's try
            t["quantities"] = [q if isinstance(q, str) else None for q in quantities]
            t["quantities_pickle"] = q_pickle
        return t

    def record(r, k):
        r = dict(r, stage=state["stage"], generation=state["gen"], evaluation=len(history))
        f = r.get("values", {}).get(oname) if "values" in r else None
        if f is not None and math.isfinite(f) and abs(f) >= _MAX_OBJECTIVE:
            r.pop("values")
            r["error"] = f"objective {oname} = {f:.6g}: |f| >= {_MAX_OBJECTIVE:g} is not supported (rescale it)"
        r["energy"] = energy(r)
        r["feasible"] = r["energy"] < _INFEASIBLE
        if k is not None:
            cache[k] = r
        history.append(r)
        b = state["best"]
        if r["feasible"] and (b is None or r["energy"] < b["energy"]):
            state["best"] = r
        return r

    def run_batch(us) -> list[float]:
        if state["stage"] == "polish" and max_time is not None and time.monotonic() - t_start > max_time:
            raise _Stop
        ps = [to_params(u) for u in us]
        keys = [tuple(p[n] for n in names) for p in ps]
        todo, seen = [], set()
        for p, k in zip(ps, keys):
            if k in cache or k in seen:
                continue
            seen.add(k)
            why = rejected(p)
            if why:
                record({"parameters": dict(p), "error": f"precheck: {why}", "seconds": 0.0}, k)
                continue
            todo.append((k, task_of(p, state["n"])))
            state["n"] += 1
        if todo:
            tasks = [t for _, t in todo]
            if pool is not None:
                recs = pool.map(tasks)
            elif callable(workers):
                recs = list(workers(_evaluate, tasks))
                if len(recs) != len(tasks):
                    raise RuntimeError(f"the workers map returned {len(recs)} records for {len(tasks)} tasks")
            else:
                recs = []
                for t in tasks:
                    recs.append(_evaluate(t))
                    if recs[-1].get("config_error"):
                        break
            for r in recs:
                if r.get("config_error"):        # the same error at every point: stop instead of spending the budget
                    raise _ConfigError(r["config_error"])
            for (k, _t), r in zip(todo, recs):
                record(r, k)
        return [cache[k]["energy"] for k in keys]

    def de_map(_func, population):
        state["gen"] += 1
        h0 = len(history)
        pop = [np.asarray(u, float) for u in population]
        try:
            out = run_batch(pop)
        except (TypeError, ValueError) as e:
            raise _Abort(e) from e
        if verbose:
            b = state["best"]
            new = history[h0:]
            nf = sum(1 for e in out if e < _INFEASIBLE)
            nfail = sum(1 for h in new if "error" in h)
            msg = (f"  gen {state['gen']:3d}: {len(pop)} trials, {nf} feasible, {nfail} failed, evaluations "
                   f"{len(history)}, " + (f"best {oname} = {sgn * b['energy']:.8g}" if b else "no feasible point yet")
                   + f"  [{time.monotonic() - t_start:.0f} s]")
            if nfail:
                msg += "\n      e.g. " + next(h["error"] for h in new if "error" in h)[:200]
            print(msg, flush=True)
        return out

    def check_stop(energies) -> bool:
        if energies is not None:
            e = np.asarray(energies, float)
            if np.all(e < _INFEASIBLE) and np.std(e) <= atol + tol * abs(np.mean(e)):
                state["converged"] = True
                return True
        if callback is not None and callback(state["gen"], state["best"]):
            state["stop"] = True
        if max_time is not None and time.monotonic() - t_start > max_time:
            state["stop"] = True
        return bool(state["stop"])

    import scipy

    if tuple(int(v) for v in scipy.__version__.split(".")[:2]) >= (1, 12):  # the population energies are passed
        def de_callback(intermediate_result):
            return check_stop(intermediate_result.population_energies)
    else:                                     # older scipy: no convergence test (runs maxiter generations)
        def de_callback(xk, convergence=None):
            return check_stop(None)

    n = len(names)
    kw: dict = {}
    sig = inspect.signature(differential_evolution).parameters
    if "rng" in sig:
        kw["rng"] = np.random.default_rng(seed) if seed is not None else None
    else:  # scipy < 1.15
        kw["seed"] = seed
    if x0 is not None:
        kw["x0"] = to_unit({**{k: p0[k] for k in names}, **x0}, "x0")
    if verbose:
        print(f"global_optimize: {sense} {oname} over {names} by differential evolution "
              f"(population {npop}, up to {maxiter} generations, <= {npop * (maxiter + 1)} evaluations, "
              f"workers {workers if not callable(workers) else 'map'})"
              + (f", {len(cons)} constraint(s)" if cons else ""), flush=True)
    # the calling process runs on `threads` too (serial evaluations, the polish), and is restored afterwards
    threads_before = _core.get_threads()
    set_threads(threads)
    pool = _Pool(int(workers), threads, timeout) if use_pool else None
    pol: dict = {"method": None}
    try:
        # scipy's own convergence test is disabled (atol < 0) and replaced by check_stop: with the rank offsets of
        # infeasible/failed points it would declare an all-failed or all-infeasible population converged
        try:
            de = differential_evolution(lambda u: 0.0, [(0.0, 1.0)] * n, strategy=strategy, maxiter=maxiter,
                                        popsize=popsize, tol=0.0, atol=-1.0, mutation=mutation,
                                        recombination=recombination, init=init, polish=False, updating="deferred",
                                        workers=de_map, callback=de_callback, **kw)
        except _Abort as a:
            raise a.args[0] from None
        de_best = state["best"]
        if de_best is None:   # nothing feasible: report the least-violating point
            ok = [h for h in history if "values" in h]
            de_best = min(ok, key=lambda h: h["energy"]) if ok else None
        de_info = {"nit": int(de.nit), "nfev": int(de.nfev), "converged": state["converged"],
                   "stopped_by_time_or_callback": state["stop"], "scipy_message": str(de.message),
                   "best_value": de_best["values"].get(oname) if de_best and "values" in de_best else None,
                   "evaluations": len(history), "population": npop, "seed": seed, "popsize": popsize,
                   "maxiter": maxiter, "tol": tol, "atol": atol, "strategy": strategy,
                   "mutation": list(mutation) if isinstance(mutation, (tuple, list)) else mutation,
                   "recombination": recombination, "seconds": round(time.monotonic() - t_start, 1)}

        # -------------------------------------------------------- local polish
        mode = "auto" if polish is True else polish
        out_of_time = max_time is not None and time.monotonic() - t_start > max_time
        if mode and de_best is not None and de_best.get("feasible") and out_of_time:
            pol = {"method": None, "skipped": "max_time exhausted"}
        elif mode and de_best is not None and de_best.get("feasible"):
            state["stage"], state["gen"] = "polish", -1
            # the adjoint optimizer reads the solver's rescaled requirement quantities: not where the DE solved for
            # the amplitude (nonlinear iron or fixed sources), which would mix two conventions
            amp_root = bool(de_best.get("amplitude"))
            use_adjoint = (mode == "adjoint" and not amp_root) or (
                mode == "auto" and isinstance(objective, str) and de_best.get("linear", True) and not amp_root
                and precheck is None and order == 0 and all(isinstance(c["quantity"], str) for c in cons))
            pre_reason = "the DE point needed the amplitude root (fixed sources)" if (
                mode == "adjoint" and amp_root) else None
            # the adjoint polish cannot be stopped by max_time: start it only if its worst case fits
            secs = [h["seconds"] for h in history if h.get("stage") == "de" and h.get("seconds")]
            # an L-BFGS-B/SLSQP iteration takes several solves (line search): the record's long solenoid took 20
            # solves in 3 iterations, so the budget is 7 per iteration
            worst = 7 * (polish_max_iter + 1) * (sum(secs) / len(secs) if secs else 0.0)
            if use_adjoint and max_time is not None and time.monotonic() - t_start + worst > max_time:
                use_adjoint = False
                pre_reason = (f"max_time: the adjoint polish (~{7 * (polish_max_iter + 1)} solves, ~{worst:.0f} s) "
                              f"would not fit in the remaining time")
            t_pol = time.monotonic()
            done = False
            pol = {"method": None, "fallback_reason": pre_reason} if pre_reason else pol
            if use_adjoint:
                try:
                    from .sensitivity import optimize as _local

                    lims = dict(zip(names, bnds))

                    start = base.with_params(**de_best["parameters"])
                    r = _local(start, objective=objective, params={k: b for k, b in zip(names, bnds)},
                               constraints=[dict(c) for c in cons], sense=sense, max_iter=polish_max_iter,
                               benchmark=benchmark, workdir=wd / "polish_adjoint", order=order,
                               verbose=max(verbose - 1, 0))
                    for h in r.get("history", []):
                        h = {**h, "method": "adjoint"}
                        if "parameters" in h:     # u*p0 with u at the bound b/p0 can land one ulp outside it
                            h["parameters"] = {k: (min(max(v, lims[k][0]), lims[k][1]) if k in lims else v)
                                               for k, v in h["parameters"].items()}
                        why = rejected(h["parameters"])
                        if why:
                            h = {k: v for k, v in h.items() if k != "values"}
                            h["error"] = f"precheck: {why}"
                        record(h, None)
                    state["adjoint_solves"] = int(r.get("solves") or 0)
                    pol = {"method": "adjoint", "converged": r.get("converged"), "message": r.get("message"),
                           "iterations": r.get("iterations"), "solves": r.get("solves")}
                    done = bool(r.get("ok"))
                    if not done:
                        pol["fallback_reason"] = "adjoint optimizer did not return a valid optimum"
                except Exception as e:  # noqa: BLE001  (e.g. topology-changing parameters: fall back)
                    pol = {"method": "adjoint", "error": f"{type(e).__name__}: {e}"}
            if not done:
                u0 = to_unit(state["best"]["parameters"])
                reason = pol.get("error") or pol.get("fallback_reason")
                try:
                    r = minimize(lambda u: run_batch([u])[0], u0, method="Powell", bounds=[(0.0, 1.0)] * n,
                                 options={"maxfev": polish_maxfev, "xtol": 1e-4, "ftol": 1e-7})
                    pol = {"method": "powell", "converged": bool(r.success), "message": str(r.message),
                           "evaluations": int(r.nfev)}
                except _Stop:
                    pol = {"method": "powell", "converged": False, "message": "stopped: max_time exhausted"}
                if reason:
                    pol["fallback_reason"] = reason
            pol["seconds"] = round(time.monotonic() - t_pol, 1)
        elif mode and de_best is not None:
            pol = {"method": None, "skipped": "no feasible point to polish"}
    finally:
        if pool is not None:
            pool.close()
        if threads_before is not None:
            set_threads(threads_before)

    best = state["best"] or de_best
    # FE solves: one per evaluated point (precheck rejections: none), the amplitude-root solves, the adjoint polish's
    n_solves = sum((h.get("amplitude") or {}).get("solves", 1) for h in history
                   if h.get("method") != "adjoint" and not str(h.get("error", "")).startswith("precheck"))
    n_solves += state["adjoint_solves"]
    if de_info["converged"]:
        msg = "converged: the population is feasible and std(f) <= atol + tol * |mean(f)|"
    elif state["stop"]:
        msg = "stopped by max_time or the callback"
    else:
        msg = f"maximum number of generations ({maxiter}) reached without convergence"
    common = {"method": "global", "objective": oname, "sense": sense, "start_parameters": None,
              "start_value": None, "iterations": de_info["nit"], "solves": n_solves, "evaluations": len(history),
              "converged": de_info["converged"], "message": msg, "de": de_info,
              "polish": pol, "seconds": round(time.monotonic() - t_start, 1),
              "history": [{k: v for k, v in h.items() if k != "results"} for h in history]}
    if best is None:
        return {**common, "ok": False, "feasible": False, "value": None, "parameters": None, "values": {},
                "caps": {}, "amplitude": None, "found_by": None, "de_best": None, "design": None,
                "constraints": [{**{k: qkey(v) if k == "quantity" else v for k, v in c.items()},
                                 "value_at_optimum": None} for c in cons],
                "message": "no evaluation succeeded"}
    vals = best["values"]
    pbest = {k: float(best["parameters"][k]) for k in names}
    d_opt = base.with_params(**pbest)
    if best.get("amplitude"):
        d_opt = d_opt.with_params(**{best["amplitude"]["parameter"]: best["amplitude"]["value"]})
    return {**common, "ok": bool(best.get("feasible")), "feasible": bool(best.get("feasible")),
            "value": vals.get(oname), "parameters": pbest, "values": vals, "caps": best.get("caps", {}),
            "amplitude": best.get("amplitude"), "found_by": best.get("stage"), "design": d_opt.doc,
            "constraints": [{**{k: qkey(v) if k == "quantity" else v for k, v in c.items()},
                             "value_at_optimum": vals.get(qkey(c["quantity"]))} for c in cons],
            "de_best": {"parameters": de_best["parameters"], "values": de_best["values"]}
            if de_best and "values" in de_best else None}


def optimize(design, objective: Any = "cost", params: dict | None = None, *args, method: str | None = None,
             **kw) -> dict:
    """Optimize a design: ``method="global"`` runs :func:`global_optimize` (differential evolution);
    any other ``method`` (default: L-BFGS-B, or SLSQP with constraints) runs the adjoint-gradient
    local optimizer :func:`axionfem.sensitivity.optimize` with the same arguments."""
    from .sensitivity import optimize as _local

    if method is None and args:
        # positional arguments follow sensitivity.optimize's signature (bounds, constraints, sense, method, ...)
        method = inspect.signature(_local).bind_partial(design, objective, params, *args).arguments.get("method")
        given_positionally = True
    else:
        given_positionally = False
    if method == "global":
        if args:
            raise TypeError("method='global' takes its options by keyword")
        return global_optimize(design, objective=objective, params=params, **kw)
    if given_positionally:
        return _local(design, objective, params, *args, **kw)
    return _local(design, objective, params, *args, method=method, **kw)

