"""Worker process that runs one MCP job (solve / sweep / optimize / sensitivities) out of process.

The MCP server (``axionfem.mcp_server``) spawns this module so that a long solve can report
progress and be killed:

* progress goes to stdout as ``AXF-PROGRESS {json}`` lines (``{"progress", "total", "message"}``);
* the solver's own log (``[axf] ...`` on stderr) is merged into the same stream by the parent and
  relayed as progress messages;
* the result is written to the file named by the job's ``result`` key, as
  ``{"ok": true, ...}`` or ``{"ok": false, "error": ..., "where"?, "hint"?}``.

usage: python -m axionfem._mcp_worker job.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import axionfem as af

# Fractions of a single solve reached when the C++ engine logs these messages.  Progress is
# clamped to be monotone, so an unrecognised line just repeats the last fraction with a new message.
STAGES: list[tuple[str, float]] = [
    ("mesh:", 0.25),
    ("solving static poloidal field", 0.32),
    ("solving static toroidal field", 0.45),
    ("solving pickup mode", 0.60),
    ("full-wave", 0.80),
]


def stage_fraction(message: str) -> float | None:
    for prefix, frac in STAGES:
        if message.startswith(prefix):
            return frac
    return None


def progress(value: float, message: str, total: float | None = 1.0) -> None:
    rec = {"progress": value, "message": message}
    if total is not None:
        rec["total"] = total
    sys.stdout.write("AXF-PROGRESS " + json.dumps(rec) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- shared helpers
def compact(res: "af.Results") -> dict:
    """The part of a results document worth returning to an agent."""
    data = res.data
    out = {"coupling": data.get("coupling"), "fields": data.get("fields"), "forces": data.get("forces"),
           "geometry": data.get("geometry"), "warnings": data.get("warnings"), "checks": data.get("checks"),
           "benchmarks": data.get("benchmarks"),
           "mesh": {"elements": data["mesh"]["elements"], "dofs": data["mesh"]["dofs"], "order": data["mesh"]["order"]},
           "timing": data.get("timing")}
    forces = out.get("forces")
    if isinstance(forces, dict) and "distributions" in forces:
        # the sample arrays can be large: summarise them and point at the full results file
        out["forces"] = dict(forces)
        out["forces"]["distributions"] = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
            for k, v in forces["distributions"].items()}
        out["forces"]["distributions_note"] = (
            "per-sample arrays (r_m, z_m, f_r/f_z, traction) are in the full results file; "
            "plot them with axionfem.plots.force_distribution(results)")
    p = dict(data.get("pickup") or {})
    p.pop("solver", None)
    out["pickup"] = p
    if data.get("conductors"):
        out["conductors"] = data["conductors"]
    h = data.get("hts")
    if h:
        # the step history can be long: keep the regions, the summary and the final state
        out["hts"] = {k: v for k, v in h.items() if k != "steps"}
        out["hts"]["steps_summary"] = [
            {"t_s": s["t_s"], "scale": s["scale"], "cumulative_loss_J": s["cumulative_loss_J"],
             "regions": s["regions"], "field_of_view": s.get("field_of_view")}
            for s in h.get("steps", [])]
    is_cavity = "cavity" in (data.get("fullwave") or {})
    if is_cavity and not data.get("pickup"):   # an eigenmode design: the lumped coupling and scan figures are empty
        for k in ("coupling", "benchmarks"):
            out.pop(k, None)
    if is_cavity:
        try:
            out["cavity_modes"] = [{k: v for k, v in m.items() if k not in ("residual_true",)}
                                   for m in res.cavity_modes()]
        except Exception as e:  # noqa: BLE001
            out["cavity_modes_error"] = str(e)
    fw = data.get("fullwave")
    if fw:
        f = {k: v for k, v in fw.items() if k not in ("sweep", "eigenmodes")}
        f["sweep"] = [{k: v for k, v in row.items() if k != "solver"} for row in fw.get("sweep", [])]
        f["eigenmodes"] = {fam: [{"f_Hz": m["f_Hz"], "k_R": m["k_R"], "energy_fraction": m["energy_fraction"]}
                                 for m in e["modes"]] for fam, e in fw.get("eigenmodes", {}).items()}
        v = mqs_validity(f["sweep"])
        if v:
            f["mqs_validity"] = v
        out["fullwave"] = f
    return out


def mqs_validity(sweep: list[dict], tolerances=(0.01, 0.05, 0.10)) -> dict | None:
    """Lowest swept frequency at which Psi, L_p and F = Psi^4/L_p^2 depart from their MQS values by more
    than each tolerance (log-log interpolated between sweep points); None = not reached below the top of the
    sweep.  A quantity that already departs at the lowest swept frequency is listed under exceeded_at_start."""
    rows = sorted((r for r in sweep if r.get("f_Hz")), key=lambda r: r["f_Hz"])

    def ratio(r, q):
        p, l = r.get("Psi_over_MQS"), r.get("L_over_MQS")
        if q == "Psi":
            return p
        if q == "L_p":
            return l
        return None if p is None or l is None or l == 0 else p ** 4 / l ** 2

    out: dict = {}
    start: list[str] = []
    top = None
    for q in ("Psi", "L_p", "F"):
        pts = [(r["f_Hz"], abs(ratio(r, q) - 1)) for r in rows if ratio(r, q) is not None]
        if len(pts) < 2:
            continue
        top = pts[-1][0]
        out[q] = {}
        for tol in tolerances:
            hit = None
            for i, (f, d) in enumerate(pts):
                if d > tol:
                    if i == 0:
                        hit = f
                        start.append(f"{q} {tol:.0%}")
                    else:
                        (f0, d0), lf0, lf1 = pts[i - 1], math.log(pts[i - 1][0]), math.log(f)
                        if d0 > 0:   # log-log interpolation (the departure grows as a power of f)
                            t = (math.log(tol) - math.log(d0)) / (math.log(d) - math.log(d0))
                        else:        # from exactly zero: linear in the deviation
                            t = tol / d
                        hit = math.exp(lf0 + min(max(t, 0.0), 1.0) * (lf1 - lf0))
                    break
            out[q][f"{tol:.0%}"] = hit
    if not out:
        return None
    out["note"] = ("frequency [Hz] above which |X(f)/X_MQS - 1| exceeds the tolerance; null = not reached up to "
                   f"the top of the sweep ({top:.4g} Hz)")
    if start:
        out["exceeded_at_start"] = start
        out["note"] += "; exceeded_at_start: already exceeded at the lowest swept frequency (the limit is lower)"
    return out


def metric(res: "af.Results", which: str, benchmark: str) -> float:
    """Objective value; designs that exceed cost.sc_wall_cap / cost.B_peak_cap are penalised by
    (1 + 10 * violation), violation = max(field/cap - 1, 0) at the size the metric refers to."""
    b = res.benchmark(benchmark)
    opt = b.get("cost_optimal_size")
    if which == "cost":
        val, caps = b["cost"]["total_MCHF"], b.get("caps_at_req", {})
    elif which == "cost_optimal":
        val = opt["cost"]["total_MCHF"] if opt else b["cost"]["total_MCHF"]
        caps = (opt or {}).get("caps", b.get("caps_at_req", {}))
    elif which == "W":
        val, caps = b["W_mag_at_req_J"], b.get("caps_at_req", {})
    elif which == "minus_g":
        return -res.g
    else:
        raise ValueError("metric must be cost, cost_optimal, W or minus_g")
    return val * (1 + 10 * caps.get("violation", 0.0))


def safe(v: Any) -> str:
    """Directory-safe text for a parameter value (expressions may contain '/', '*', ...)."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", str(v))[:60]


# ---------------------------------------------------------------- jobs
def job_solve(job: dict) -> dict:
    d = af.Design(job["design"])
    wd = Path(job["workdir"])
    progress(0.05, "meshing and solving")
    t0 = time.time()
    res = d.solve(workdir=wd, order=int(job.get("order", 0)), verbose=1,
                  self_force=bool(job.get("self_force", True)),
                  force_samples=int(job.get("force_samples", 200)))
    out = {"ok": True, "name": job["name"], "seconds": round(time.time() - t0, 2), **compact(res),
           "files": {"results": str(wd / "results.json")}}
    if job.get("scene", True):
        progress(0.92, "writing the viewer scene")
        out["files"]["scene"] = str(res.save_scene(wd / "scene.json"))
        try:
            out["files"]["html"] = str(res.export_html(wd / f"{job['name']}.html"))
        except Exception as e:  # noqa: BLE001  (viewer assets missing)
            out["files"]["html_error"] = str(e)
    return out


def job_sweep(job: dict) -> dict:
    base = af.Design(job["design"])
    bench = job["benchmark"]
    values = job["values"]
    param = job["parameter"]
    root = Path(job["workdir"])
    rows = []
    for i, v in enumerate(values):
        progress(i / len(values), f"{param} = {v}  ({i + 1}/{len(values)})", 1.0)
        d = base.with_params(**{param: v})
        res = d.solve(workdir=root / f"{param}_{safe(v)}", verbose=1,
                      self_force=bool(job.get("self_force", True)))
        b = res.benchmark(bench)
        rows.append({param: v, "Psi": res.Psi, "L_p": res.L_p, "g": res.g,
                     "F_over_F_req": b.get("F_over_F_req"), "W_at_req_J": b.get("W_mag_at_req_J"),
                     "B_max_at_req_T": b.get("B_max_at_req_T"),
                     "sc_surface_at_req_T": b.get("superconductor_surface_at_req_T"),
                     "cost_MCHF": b.get("cost", {}).get("total_MCHF"),
                     "cost_optimal_MCHF": (b.get("cost_optimal_size") or {}).get("cost", {}).get("total_MCHF")})
    return {"ok": True, "benchmark": bench, "rows": rows}


def job_optimize(job: dict) -> dict:
    import numpy as np
    from scipy.optimize import minimize

    base = af.Design(job["design"])
    free = job["free"]
    keys = list(free)
    lo = np.array([free[k][0] for k in keys], float)
    hi = np.array([free[k][1] for k in keys], float)
    max_evals = int(job["max_evals"])
    hist: list[dict] = []
    root = Path(job["workdir"])

    def f(u):
        x = lo + np.clip(u, 0, 1) * (hi - lo)
        progress(min(len(hist) / max_evals, 0.99),
                 "evaluation %d/%d: %s" % (len(hist) + 1, max_evals,
                                           ", ".join(f"{k}={v:.4g}" for k, v in zip(keys, x))))
        d = base.with_params(**{k: float(v) for k, v in zip(keys, x)})
        try:
            # the objective does not use forces, and the run directory is reused every evaluation
            res = d.solve(workdir=root / "current", verbose=1, self_force=False, force_samples=0)
            val = metric(res, job["metric"], job["benchmark"])
        except Exception as e:  # noqa: BLE001
            hist.append({"x": dict(zip(keys, x.tolist())), "error": str(e)})
            return 1e30
        hist.append({"x": dict(zip(keys, x.tolist())), "value": val})
        return val

    u0 = np.full(len(keys), 0.5)
    resolved = base.resolved_params()
    for i, k in enumerate(keys):
        v = resolved.get(k)
        if v is not None and hi[i] > lo[i]:
            u0[i] = min(max((v - lo[i]) / (hi[i] - lo[i]), 0.0), 1.0)
    r = minimize(f, u0, method="Powell", bounds=[(0.0, 1.0)] * len(keys),
                 options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-4})
    good = [h for h in hist if "value" in h]
    if not good:
        raise RuntimeError("every evaluation failed; see the history: " + json.dumps(hist[-5:]))
    best = min(good, key=lambda h: h["value"])
    return {"ok": True, "best": best, "evaluations": len(hist), "history": hist[-20:],
            "converged": bool(r.success)}


def job_sensitivities(job: dict) -> dict:
    d = af.Design(job["design"])
    order = int(job.get("order", 0))
    progress(0.05, "meshing and solving the nominal design")
    t0 = time.time()
    res = d.solve(workdir=Path(job["workdir"]), order=order, verbose=1)
    progress(0.6, "differentiating")
    s = res.sensitivities(params=job.get("parameters") or None, objectives=job.get("quantities") or None,
                          method=job.get("method", "adjoint"), step=float(job.get("step", 1e-4)),
                          benchmark=job.get("benchmark", ""), order=order)
    s["seconds"] = round(time.time() - t0, 2)
    s["diagnostics"] = {k: v for k, v in s.get("diagnostics", {}).items() if k != "morph"}
    return {"ok": True, "name": job["name"], **s}


def job_optimize_adjoint(job: dict) -> dict:
    max_iter = int(job["max_iter"])
    done = [0]

    def iterate(*_args, **_kw):  # scipy calls this once per accepted iteration, with xk
        done[0] += 1
        progress(min(done[0] / (max_iter + 1), 0.99), f"iteration {done[0]} of at most {max_iter}")

    return af.optimize(af.Design(job["design"]), objective=job["objective"],
                       params={k: tuple(v) for k, v in job["free"].items()},
                       constraints=job.get("constraints") or (), sense=job.get("sense", "min"),
                       max_iter=max_iter, benchmark=job.get("benchmark", ""),
                       workdir=Path(job["workdir"]), order=int(job.get("order", 0)), verbose=0,
                       callback=iterate)


def job_optimize_global(job: dict) -> dict:
    maxiter = int(job["maxiter"])

    def generation(gen, best):  # called once per DE generation
        progress(min(gen / max(maxiter, 1), 0.95),
                 f"generation {gen} of at most {maxiter}" + (
                     f": best {job['objective']} = {best['values'][job['objective']]:.6g}" if best else ""))
        return False

    free = {k: tuple(v) for k, v in job["free"].items()}
    return af.optimize(af.Design(job["design"]), method="global", objective=job["objective"], params=free,
                       constraints=job.get("constraints") or (), sense=job.get("sense", "min"),
                       seed=job.get("seed", 0), popsize=int(job.get("popsize", 15)), maxiter=maxiter,
                       workers=int(job.get("workers", 1)), polish=job.get("polish", True),
                       benchmark=job.get("benchmark", ""), workdir=Path(job["workdir"]), verbose=0,
                       callback=generation)


JOBS = {"solve": job_solve, "sweep": job_sweep, "optimize": job_optimize,
        "sensitivities": job_sensitivities, "optimize_adjoint": job_optimize_adjoint,
        "optimize_global": job_optimize_global}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(job["result"])
    af.set_threads(int(job.get("threads", 4)))
    progress(0.01, "starting " + job["kind"])
    try:
        out = JOBS[job["kind"]](job)
        progress(1.0, "done")
        rc = 0
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        out = {"ok": False, "error": msg}
        m = re.search(r"\[at ([^\]]*)\]", msg)
        if m:
            out["where"] = m.group(1)
        if "Hint:" in msg:
            out["hint"] = msg.split("Hint:", 1)[1].strip()
        rc = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
