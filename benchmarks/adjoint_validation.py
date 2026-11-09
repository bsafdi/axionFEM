"""Validation of the adjoint shape sensitivities.

    python benchmarks/adjoint_validation.py [workdir] [--quick]

Four independent checks:
  1. source scaling: for linear media every figure of merit is an exact power of the
     ampere-turns (Psi^1, L_p^0, W^2, F^4, cost^0), so the derivative is known in closed form;
  2. dilation: all template lengths are proportional to R and H (Rc and H for CORE), so the
     figures of merit are homogeneous and Euler's identity R dJ/dR + H dJ/dH = n J holds
     exactly for the discrete problem as well;
  3. adjoint vs central differences on the same morphed mesh (same discrete functional);
  4. adjoint vs central differences with full remeshing, at several step sizes: the difference
     is remeshing noise and shrinks as the step grows.
Three optimizations then check the gradients end to end against known optima.

Writes benchmarks/records/adjoint_validation.json and docs/validation/adjoint.md.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records" / "adjoint_validation.json"
# the same optimum from a derivative-free search (36 solves, order 3)
DFS_REC = ROOT / "benchmarks" / "records" / "cost_metric_optimum_reference.json"
# the readout benchmark of the cost case: the same as m1_validation.REF_BENCH (round values; the optimum shape does not
# depend on it, the absolute size does)
REF_BENCH = {"name": "reference", "Q": 1e6, "T": 0.1, "eta_A": 1.0}
OUT = ROOT / "docs" / "validation" / "adjoint.md"

OBJ = ["Psi", "L_p", "W_mag", "F", "cost", "cost_optimal"]
SOURCE_DEGREE = {"Psi": 1, "L_p": 0, "W_mag": 2, "F": 4, "cost": 0, "cost_optimal": 0}
# the cost at the cost-optimal size is scale invariant, hence degree 0
DILATION_DEGREE = {"Psi": 2, "L_p": 1, "W_mag": 1, "F": 6, "cost_optimal": 0}
CASES = {
    "toroid": {"params": ["R", "H", "y", "tw", "g", "ts", "f", "NI"], "size": ["R", "H"]},
    "solenoid": {"params": ["R", "H", "x", "tw", "g", "tc", "tp", "NI"], "size": ["R", "H"]},
    "core": {"params": ["Rc", "H", "b", "h", "w", "gn", "NI"], "size": ["Rc", "H"]},
}
QUICK_MESH = {"toroid": {"size": "max(R, H)/12", "surface_size": "max(R, H)/40"},
              "solenoid": {"size": "max(R, H)/12", "surface_size": "max(R, H)/40"},
              "core": {"size": "max(b, H)/12", "surface_size": "min(w, gn)"}}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def run(name: str, wd: Path, quick: bool) -> dict:
    d = af.template(name)
    if quick:
        d.doc["mesh"] = {**d.doc["mesh"], **QUICK_MESH[name]}
    t0 = time.time()
    res = d.solve(workdir=wd / name, verbose=0)
    solve_s = time.time() - t0
    params = CASES[name]["params"]
    out = {"template": name, "parameters": res.data["design"]["parameters"], "quick": quick,
           "mesh": {k: res.data["mesh"][k] for k in ("elements", "dofs", "order")},
           "solve_seconds": solve_s}

    sens = {}
    for method, step in [("adjoint", 1e-4), ("adjoint", 1e-3), ("fd_morph", 1e-4), ("fd_morph", 1e-3),
                         ("fd_morph", 1e-2), ("fd_remesh", 1e-4), ("fd_remesh", 1e-3), ("fd_remesh", 1e-2)]:
        key = f"{method}@{step:g}"
        t0 = time.time()
        sens[key] = res.sensitivities(params=params, objectives=OBJ, method=method, step=step)
        sens[key]["wall_seconds"] = time.time() - t0
        print(f"  {name:9s} {key:16s} {sens[key]['wall_seconds']:6.1f} s", flush=True)
    out["values"] = sens["adjoint@0.0001"]["values"]
    out["gradient"] = {k: v["gradient"] for k, v in sens.items()}
    out["relative"] = sens["adjoint@0.0001"]["relative"]
    out["seconds"] = {k: v["wall_seconds"] for k, v in sens.items()}
    out["morph"] = sens["adjoint@0.0001"]["diagnostics"]["morph"][params[0]]
    out["closed_form_check"] = sens["adjoint@0.0001"]["diagnostics"]["checks"]["closed_form_vs_post_processing"]

    # 1. source scaling
    NI = out["parameters"]["NI"]
    out["source"] = []
    for o, n in SOURCE_DEGREE.items():
        J = out["values"][o]
        for key in ("adjoint@0.0001", "fd_morph@0.0001"):
            dJ = out["gradient"][key][o]["NI"]
            exact = n * J / NI
            out["source"].append({"objective": o, "method": key, "degree": n, "d": dJ, "exact": exact,
                                  "rel": rel(dJ, exact) if n else abs(dJ) * NI / max(abs(J), 1e-300)})
    # 2. dilation
    a, b = CASES[name]["size"]
    out["dilation"] = []
    for o, n in DILATION_DEGREE.items():
        for key in out["gradient"]:
            g = out["gradient"][key][o]
            lhs = out["parameters"][a] * g[a] + out["parameters"][b] * g[b]
            rhs = n * out["values"][o]
            out["dilation"].append({"objective": o, "method": key, "degree": n, "lhs": lhs, "rhs": rhs,
                                    "rel": rel(lhs, rhs) if n else abs(lhs) / abs(out["values"][o])})
    # 3./4. adjoint vs finite differences, parameter by parameter
    out["compare"] = []
    for o in OBJ:
        J = out["values"][o]
        for p in params:
            da = out["gradient"]["adjoint@0.0001"][o][p]
            row = {"objective": o, "parameter": p, "adjoint": da,
                   "log_derivative": da * out["parameters"][p] / J if J else 0.0}
            # noise floor: a logarithmic derivative below 1e-5 is not resolvable
            floor = 1e-5 * abs(J) / max(abs(out["parameters"][p]), 1e-300)
            row["below_noise_floor"] = abs(da) < floor
            for key in out["gradient"]:
                if key == "adjoint@0.0001":
                    continue
                df = out["gradient"][key][o][p]
                row[key] = df
                row["rel_" + key] = rel(da, df)
            out["compare"].append(row)
    return out


def record_opt(case: str, o: dict, seconds: float, reference: dict | None = None) -> dict:
    r = {"case": case, "objective": o["objective"], "sense": o["sense"], "start": o["start_parameters"],
         "optimum": o["parameters"], "start_value": o["start_value"], "value": o["value"],
         "iterations": o["iterations"], "solves": o["solves"], "converged": o["converged"], "seconds": seconds}
    if reference:
        r["reference"] = reference
    return r


def optimization(wd: Path, quick: bool) -> list[dict]:
    """Optimizations driven by the adjoint gradients, against known optima."""
    runs = []
    # 1. the long-limit geometric factor of a toroid peaks at y = exp(-1/2) (g_tor = y^2 ln(1/y))
    d = af.template("toroid", y=0.45)
    d.doc["mesh"] = {**d.doc["mesh"], **QUICK_MESH["toroid"]} if quick else d.doc["mesh"]
    t0 = time.time()
    o = af.optimize(d, objective="g_factor", sense="max", params={"y": (0.25, 0.9)}, max_iter=20,
                    workdir=wd / "opt_toroid", verbose=0)
    runs.append(record_opt("toroid with H = R: maximise g over y (start 0.45)", o, time.time() - t0,
                           {"y_long_limit": math.exp(-0.5)}))
    # 2. the long-limit geometric factor of a coaxial solenoid peaks at x = 0.534
    #    (g_sol = (1 - x^2)^2/(4 ln 1/x))
    d2 = af.template("solenoid", x=0.42, H=8.0, R=1.0)
    d2.doc["mesh"] = {"order": 2, "size": "max(R, H)/30", "surface_size": "0.004*R"}
    t0 = time.time()
    o2 = af.optimize(d2, objective="g_factor", sense="max", params={"x": (0.3, 0.75)}, max_iter=20,
                     workdir=wd / "opt_solenoid_x", verbose=0)
    runs.append(record_opt("long solenoid (H/R = 8): maximise g over x (start 0.42)", o2, time.time() - t0,
                           {"x_long_limit": 0.534}))
    # 3. the optimum of the cost metric at the cost-optimal size of a finite current-sheet solenoid, found with
    #    gradients; the reference is the same optimum from the derivative-free (Nelder-Mead) search of
    #    m1_validation.py (record cost_metric_optimum_reference.json). The cost metric is a scaling metric, used here
    #    only as a smooth objective with an interior optimum.
    d3 = af.template("solenoid", R=1.0, H=1.3, x=0.55, tw="0.002*R", g="0.001*R", tc="0.001*R", tp="0.002*R",
                     mesh={"order": 2 if quick else 3, "size": "max(R,H)/30", "surface_size": "0.002*R"},
                     benchmarks=[REF_BENCH])
    t0 = time.time()
    o3 = af.optimize(d3, objective="cost_optimal", params={"H": (0.6, 4.0), "x": (0.35, 0.75)}, max_iter=25,
                     benchmark=REF_BENCH["name"], workdir=wd / "opt_solenoid_cost", verbose=0)
    res = af.Design(o3["design"]).solve(workdir=wd / "opt_solenoid_cost" / "best", verbose=0)
    c = res.benchmark(REF_BENCH["name"])["cost_optimal_size"]
    got = {"R_m": c["device_radius_m"], "H_m": o3["parameters"]["H"] * c["length_scale"],
           "x": o3["parameters"]["x"], "W_MJ": c["W_mag_J"] / 1e6, "cost_MCHF": c["cost"]["total_MCHF"],
           "length_scale": c["length_scale"]}
    ref = None
    if DFS_REC.exists():
        j = json.loads(DFS_REC.read_text())
        ref = {k: j["computed"][k] for k in ("R_m", "H_m", "x", "W_MJ")}
        ref["cost_MCHF"] = j["cost_MCHF"]
        ref["source"] = "derivative-free (Nelder-Mead) optimum, benchmarks/records/cost_metric_optimum_reference.json"
    rec = record_opt("finite current-sheet solenoid: minimise the cost metric at the cost-optimal size over H and x",
                     o3, time.time() - t0, ref)
    rec["computed"] = got
    if ref:
        rec["rel_dev"] = {k: abs(got[k] - v) / v for k, v in ref.items() if isinstance(v, (int, float))}
    runs.append(rec)
    return runs


def fmt(x, f="{:.6g}"):
    return "-" if x is None else f.format(x)


def step_dependence(t: dict) -> tuple[float, str, str]:
    """The adjoint's own sensitivity to the finite-difference step it uses internally (1e-4 vs 1e-3),
    over the entries that are above the noise floor; returns the worst one and where it sits."""
    worst, where = 0.0, ("", "")
    for r in t["compare"]:
        if r["below_noise_floor"]:
            continue
        o, p = r["objective"], r["parameter"]
        a, b = t["gradient"]["adjoint@0.0001"][o][p], t["gradient"]["adjoint@0.001"][o][p]
        if rel(a, b) > worst:
            worst, where = rel(a, b), (o, p)
    return worst, where[0], where[1]


def report(rec: dict) -> str:
    L = ["# Adjoint shape sensitivities: validation", "",
         "Generated by `benchmarks/adjoint_validation.py` "
         f"({'coarse' if rec['quick'] else 'default'} meshes).",
         "", "Method and conventions: `docs/optimization.md`. All derivatives are of the *discrete* figures of",
         "merit at a fixed mesh topology; `fd_remesh` builds a new mesh for every evaluation, so the",
         "difference from the adjoint is remeshing noise, not an error of the adjoint.", ""]
    for name, t in rec["cases"].items():
        p = t["parameters"]
        L += [f"## {name}", "",
              f"{t['mesh']['elements']} triangles, {t['mesh']['dofs']} DOFs, order {t['mesh']['order']}; "
              f"solve {t['solve_seconds']:.1f} s. "
              + ", ".join(f"{k} = {fmt(v, '{:.4g}')}" for k, v in p.items() if k in CASES[name]["params"]), "",
              "Figures of merit: " + ", ".join(f"{k} = {fmt(v)}" for k, v in t["values"].items()
                                               if not k.startswith("cost")), "",
              "Cost of one gradient (all %d parameters, 5 objectives):" % len(CASES[name]["params"]), "",
              "| method | wall time [s] | per solve |", "|---|---|---|"]
        for k, v in t["seconds"].items():
            L.append(f"| {k} | {v:.1f} | {v / t['solve_seconds']:.1f} |")
        L += ["", "### 1. Source scaling (exact: J ~ NI^n)", "",
              "| quantity | n | dJ/dNI | exact | rel |", "|---|---|---|---|---|"]
        for r in t["source"]:
            if r["method"] != "adjoint@0.0001":
                continue
            L.append(f"| {r['objective']} | {r['degree']} | {fmt(r['d'])} | {fmt(r['exact'])} | "
                     f"{r['rel']:.1e} |")
        a, b = CASES[name]["size"]
        L += ["", f"### 2. Dilation (exact: {a} dJ/d{a} + {b} dJ/d{b} = n J)", "",
              "| quantity | n | adjoint | fd_morph | fd_remesh (1e-4) | fd_remesh (1e-2) |", "|---|---|---|---|---|---|"]
        for o in DILATION_DEGREE:
            row = {r["method"]: r for r in t["dilation"] if r["objective"] == o}
            L.append(f"| {o} | {DILATION_DEGREE[o]} | " + " | ".join(
                f"{row[k]['rel']:.1e}" for k in ("adjoint@0.0001", "fd_morph@0.0001", "fd_remesh@0.0001",
                                                 "fd_remesh@0.01")) + " |")
        L += ["", "(relative difference between the two sides of the identity)", "",
              "### 3./4. Adjoint vs finite differences", "",
              "`d log J / d log p` is the dimensionless sensitivity; the five comparison columns are relative",
              "differences |adjoint - FD| / max(|adjoint|, |FD|) at the given finite-difference step. Rows marked",
              "* have a derivative below the noise floor (|d log J / d log p| < 1e-5, i.e. below the solver's own",
              "discretization error), where the relative difference is meaningless.", "",
              "| quantity | p | dJ/dp | d log J / d log p | fd_morph 1e-4 | fd_morph 1e-3 | fd_remesh 1e-4 | fd_remesh 1e-3 | fd_remesh 1e-2 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in t["compare"]:
            star = "*" if r["below_noise_floor"] else ""
            L.append(f"| {r['objective']}{star} | {r['parameter']} | {fmt(r['adjoint'])} | "
                     f"{fmt(r['log_derivative'], '{:+.3f}')} | {r['rel_fd_morph@0.0001']:.1e} | "
                     f"{r['rel_fd_morph@0.001']:.1e} | {r['rel_fd_remesh@0.0001']:.1e} | "
                     f"{r['rel_fd_remesh@0.001']:.1e} | {r['rel_fd_remesh@0.01']:.1e} |")
        L += ["", f"Mesh morphing for `{CASES[name]['params'][0]}`: " +
              ", ".join(f"{k} = {v}" for k, v in t["morph"].items() if k != "domain_radius_m") + ".",
              ""]
        sd, so_, sp = step_dependence(t)
        L += [f"Closed-form post-processing check {t['closed_form_check']:.1e}. The adjoint's own sensitivity to "
              f"the step it differences the assembly with (1e-4 vs 1e-3) is at most {sd:.1e} relative, on "
              f"d{so_}/d{sp}" + (" (a millimetre-scale feature)" if sp in ("g", "ts", "tc", "tp", "gn", "w", "tw", "f")
                                 else "") + ".", ""]
    L += ["## Gradient-based optimization", ""]
    for r in rec["optimization"]:
        cost = r["objective"].startswith("cost")   # the cost metric: its change, not its value
        change = (f"changes by {r['value'] / r['start_value'] - 1:+.1e} (relative)" if cost
                  else f"{fmt(r['start_value'])} -> {fmt(r['value'])}")
        ref = {k: v for k, v in (r.get("reference") or {}).items()
               if isinstance(v, (int, float)) and not k.startswith("cost")}
        # the cost metric's optimum (shape and size) is not quoted: only the optimizers' agreement
        L += [f"- **{r['case']}**: {r['objective']} {change} in "
              f"{r['iterations']} iterations ({r['solves']} solves, {r['seconds']:.0f} s, converged "
              f"{r['converged']})"
              + ("" if cost else "; " + ", ".join(f"{k}: {fmt(v, '{:.4g}')} -> {fmt(r['optimum'][k], '{:.4g}')}"
                                               for k, v in r["start"].items()))
              + (("; reference " + ", ".join(f"{k} = {fmt(v, '{:.4g}')}" for k, v in ref.items()))
                 if ref and "computed" not in r else "")]
        if "computed" in r and r.get("reference"):
            nm = len(json.loads(DFS_REC.read_text()).get("history", [])) if DFS_REC.exists() else "-"
            L += ["", "  | quantity at the optimum | adjoint vs Nelder-Mead |", "  |---|---|"]
            for k in ref:
                L.append(f"  | {k} | {100 * r['rel_dev'][k]:.2f} % |")
            L += ["", f"  The reference is the same optimum found with a derivative-free (Nelder-Mead) search ({nm} solves,",
                  "  order 3, `benchmarks/records/cost_metric_optimum_reference.json`): the adjoint run agrees with it to ~3 digits",
                  f"  in {r['solves']} solves, and the cost metric at the two optima differs by "
                  f"{r['rel_dev'].get('cost_MCHF', float('nan')):.1e} relative. The cost metric is a scaling metric for",
                  "  comparing designs, not a construction estimate; it serves here only as a smooth objective with an",
                  "  interior optimum, so its optimum itself is not quoted.", ""]
    L += ["", "## Reading these numbers", "",
          "- The source-scaling identities hold to 1e-12-1e-10: the sources enter the right-hand side only,",
          "  and the adjoint picks that up essentially exactly.",
          "- The dilation identity holds to 1e-11-1e-5 for the adjoint and to 1e-12-1e-7 for `fd_morph`.",
          "  The adjoint's residual term is a difference of two large, nearly cancelling contributions, so",
          "  it carries more round-off than a plain difference of functionals; the gap is the round-off of",
          "  the finite differences of the *assembly*, not an error of the adjoint relation.",
          "- `fd_morph` and the adjoint differ by 1e-9-1e-6 relative for the parameters that set the overall",
          "  size, and by up to ~1e-3 for parameters of millimetre-scale features (a wall thickness, a gap).",
          "  There the default relative step is a micrometre: the perturbed functionals differ in their last",
          "  few digits and the finite difference is round-off limited, while the adjoint is not. Raising the",
          "  step (the `fd_morph 1e-3` column, or `steps={'gn': 1e-5}`) removes the discrepancy and shows",
          "  that the adjoint, not the finite difference, is the stable estimate.",
          "- `cost_optimal` is scale invariant (dilation degree 0), so its derivative with respect to a single",
          "  size parameter is a small difference of large contributions - for the toroid dC*/dR = -0.194 and",
          "  dC*/dH = +0.194 cancel to 1e-5 of each other. Those entries inherit a cancellation factor of 50-100",
          "  and are the least accurate in the table (6e-3 for the toroid, 3e-2 for CORE between the two internal",
          "  finite-difference steps), although the primitives behind them are good to 1e-6. The descent",
          "  direction is unaffected (the cost-optimum run above lands on the derivative-free optimum), but a single such number",
          "  should not be read to three digits.",
          "- `fd_remesh` differs by 1e-4-1e-2 at a step of 1e-4 and improves roughly in proportion to the",
          "  step, which identifies the difference as remeshing noise (a fresh mesh changes the discrete",
          "  figures of merit by ~1e-5 relative) rather than a bias of the adjoint.",
          "- Derivatives that vanish identically are physics, not failure: a copper coax is transparent to",
          "  the static field (dW/dx = 0 for the solenoid), L_p does not depend on the ampere-turns, and",
          "  the cost at the requirement is invariant under a rescaling of the sources. Those rows are",
          "  marked * and their relative differences are meaningless.", ""]
    return "\n".join(L) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    wd = Path(args[0] if args else "adjoint_validation_runs")
    wd.mkdir(parents=True, exist_ok=True)
    if "--report-only" in sys.argv:  # rewrite the document from the stored record
        OUT.write_text(report(json.loads(REC.read_text())))
        print(OUT)
        return
    rec = {"version": af.version(), "date": time.strftime("%Y-%m-%d"), "quick": quick, "cases": {}}
    for name in CASES:
        print(name, flush=True)
        rec["cases"][name] = run(name, wd, quick)
    rec["optimization"] = optimization(wd, quick)
    REC.write_text(json.dumps(rec, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report(rec))
    print(OUT)


if __name__ == "__main__":
    main()
