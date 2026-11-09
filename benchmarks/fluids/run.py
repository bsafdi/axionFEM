#!/usr/bin/env python3
"""Run the fluids and chemistry verification cases and check their acceptance criteria (beta modules).

    python benchmarks/fluids/run.py list [--track fluids|chem|all] [--cost short|long]
    python benchmarks/fluids/run.py show CASE                    # the case definition's criteria, one line each
    python benchmarks/fluids/run.py run CASE [CASE ...] [--quick] [--out DIR] [--jobs N] [--timeout S]
    python benchmarks/fluids/run.py run --all [--track ...] [--cost short|long] [--quick] [--out DIR] [--jobs N]
    python benchmarks/fluids/run.py report DIR [DIR ...] [--track fluids|chem]   # Markdown table of saved results

A case is a JSON file in benchmarks/<track>/registry/cases/: the runner function that produces the raw numbers
(``run.runner``, a function of benchmarks/<track>/cases/<module>.py), its parameters (``run.parameters``: meshes,
levels, data files with their sha256), the reference values (``reference``) and the acceptance items
(``acceptance``).  ``run`` calls the runner, judges its result with benchmarks/fluids/acceptance.py and prints one
line per case; ``--out`` also writes the raw result and the item-by-item evaluation as JSON.

``--quick`` runs the first two refinement levels of every series only: a smoke test of the whole pipeline in a
fraction of the time.  Convergence and Richardson items need more levels, so a quick run usually reports them as
failed; its exit status only reflects errors (a case that crashed or timed out).  A full run (no ``--quick``) exits
with status 1 unless every case meets all its criteria.  ``cost`` short cases take seconds to a few minutes on one
core; long cases take minutes to hours, and some need MPI (see benchmarks/fluids/README.md).

``report`` reads the JSON results that ``run --out`` wrote (several directories may be given; a full result is
preferred to a quick one) and prints one Markdown table per case group with the measured numbers: observed orders
against the expected ones, the largest comparison deviation relative to its allowance, the number of checks.  A
closing paragraph gives the totals (cases run in full and in quick mode, criteria met, and the numbers of observed
orders, Richardson estimates, comparisons and checks in the full runs).  The tables and totals in
docs/validation/fluids.md and docs/validation/chemistry.md are its output.

The fluids library is built only in MPI builds (``axionfem._core._fluids``); several cases need it, the chemistry
cases need ``axionfem._core.chem``.  When ``axionfem`` is not importable, the in-place module under python/ of this
checkout is used.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRACKS = {"fluids": ROOT / "benchmarks" / "fluids", "chem": ROOT / "benchmarks" / "chem"}


def case_dir(track: str) -> Path:
    return TRACKS[track] / "registry" / "cases"


def all_cases(track: str = "all") -> list[tuple[str, dict]]:
    out = []
    for t in (TRACKS if track == "all" else [track]):
        for f in sorted(case_dir(t).glob("*.json")):
            out.append((t, json.loads(f.read_text())))
    return out


def find_case(cid: str) -> tuple[str, dict]:
    for t in TRACKS:
        f = case_dir(t) / f"{cid}.json"
        if f.is_file():
            return t, json.loads(f.read_text())
    raise SystemExit(f"no case '{cid}' in benchmarks/fluids or benchmarks/chem (see `run.py list`)")


def jsonable(x):
    if isinstance(x, dict) or hasattr(x, "items"):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    return repr(x)


def _import_paths(track: str) -> None:
    paths = [str(HERE), str(TRACKS[track] / "cases"), str(TRACKS["fluids"] / "cases")]
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import axionfem  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "python"))


def run_one(track: str, case: dict, quick: bool, out: Path | None) -> dict:
    """Run one case in this process: runner, then the acceptance check.  Returns the summary dict."""
    _import_paths(track)
    import acceptance
    r = case["run"]["runner"]
    mod = importlib.import_module(r["module"])
    fn = getattr(mod, r["function"])
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"axf_case_{case['id']}_") as wd:
        result = jsonable(fn(json.loads(json.dumps(case["run"]["parameters"])), Path(wd), bool(quick)))
    wall = time.time() - t0
    ev = acceptance.evaluate(case, result, quick=quick)
    summary = {"case": case["id"], "track": track, "quick": bool(quick), "wall_s": wall,
               "criteria_met": bool(ev["criteria_met"]), "headline": ev["headline"], "verdicts": ev["verdicts"]}
    if out is not None:
        try:
            from axionfem import _core
            build = json.loads(_core.build_info())
            build.get("library", {}).pop("build_dir", None)   # an absolute path: not written
        except Exception:  # noqa: BLE001
            build = None
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{case['id']}.json").write_text(json.dumps(jsonable(
            {**summary, "build": build, "result": result, "evaluation": ev}), indent=1))
    return summary


def _one_subprocess(track: str, cid: str, quick: bool, out: Path, timeout: float) -> dict:
    argv = [sys.executable, str(Path(__file__).resolve()), "_one", cid, "--out", str(out)] + (["--quick"] if quick else [])
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"case": cid, "track": track, "status": "TIMEOUT", "wall_s": time.time() - t0,
                "headline": f"timed out after {timeout:.0f} s"}
    log = out / f"{cid}.log"
    log.write_text(p.stdout + p.stderr)
    f = out / f"{cid}.json"
    if p.returncode != 0 or not f.is_file():
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
        return {"case": cid, "track": track, "status": "ERROR", "wall_s": time.time() - t0,
                "headline": f"{tail[0][:200]} (log: {log})"}
    s = json.loads(f.read_text())
    s["status"] = "PASS" if s["criteria_met"] else ("SMOKE" if quick else "FAIL")
    return s


def cmd_list(a) -> int:
    rows = [(t, c) for t, c in all_cases(a.track) if not a.cost or c["run"]["runner"].get("cost") == a.cost]
    for t, c in rows:
        r = c["run"]["runner"]
        print(f"{c['id']:44s} {t:6s} {c['capability']:5s} {r.get('cost', ''):5s} {','.join(c['levels']):28s} "
              f"{r['module']}.{r['function']}")
    print(f"{len(rows)} cases")
    return 0


def cmd_show(a) -> int:
    track, c = find_case(a.case)
    print(f"{c['id']} ({track}, {c['capability']}): {c['title']}")
    print(f"  physics: {c.get('physics', '')}")
    acc = c.get("acceptance") or {}
    for spec in acc.get("convergence", []) or []:
        print(f"  convergence  {spec['series']}: expected order {spec.get('expected_order')} "
              f"+- {spec.get('order_tolerance', '')} [{spec['level']}]")
    for spec in acc.get("functional_convergence", []) or []:
        print(f"  functional   {spec.get('id')}: order in {spec.get('order_range')} [{spec['level']}]")
    ref = (c.get("reference") or {}).get("quantities") or {}
    for spec in acc.get("comparisons", []) or []:
        q = ref.get(spec["reference"], {})
        print(f"  comparison   {spec['quantity']} vs {spec['reference']} = {q.get('value', q.get('kind'))} "
              f"(source {q.get('source')}) tol {spec.get('tolerance', '')} [{spec['level']}]")
    for spec in acc.get("checks", []) or []:
        bounds = {k: spec[k] for k in ("min", "max", "abs_max", "c", "lo", "hi") if k in spec}
        print(f"  check        {spec['quantity']} {spec.get('kind', 'bound')} {bounds} [{spec['level']}]")
    for key in ("solver_sensitivity", "quadrature_sensitivity", "tolerance_behaviour"):
        for spec in acc.get(key, []) or []:
            print(f"  {key:12s} {spec.get('quantity') or spec.get('id')} [{spec['level']}]")
    return 0


def cmd_run(a) -> int:
    if a.all or a.cost:
        sel = [(t, c) for t, c in all_cases(a.track) if not a.cost or c["run"]["runner"].get("cost") == a.cost]
    elif a.cases:
        sel = [find_case(cid) for cid in a.cases]
    else:
        raise SystemExit("name one or more cases, or pass --all / --cost")
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="axf_cases_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(sel)} case(s){' (quick)' if a.quick else ''}; results in {out}")
    jobs = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for t, c in sel:
            timeout = a.timeout or float(c["run"]["runner"].get("timeout_s") or 3600)
            jobs.append(ex.submit(_one_subprocess, t, c["id"], a.quick, out, timeout))
        res = []
        for j in cf.as_completed(jobs):
            s = j.result()
            res.append(s)
            print(f"{s['status']:8s} {s['case']:44s} {s['wall_s']:8.1f} s  {s['headline'][:150]}", flush=True)
    counts = {k: sum(1 for s in res if s["status"] == k) for k in ("PASS", "SMOKE", "FAIL", "ERROR", "TIMEOUT")}
    print("summary: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    (out / "summary.json").write_text(json.dumps(jsonable(sorted(res, key=lambda s: s["case"])), indent=1))
    bad = counts["ERROR"] + counts["TIMEOUT"] + (0 if a.quick else counts["FAIL"])
    return 1 if bad else 0


GROUPS = {"F1a": "Taylor-Hood Stokes, planar", "F1b": "Taylor-Hood Stokes, axisymmetric with swirl",
          "F1c": "Taylor-Hood Stokes, 3D (MPI)", "F2a": "steady Navier-Stokes", "F3a": "transient Navier-Stokes",
          "F4a": "scalar transport and Boussinesq", "F5": "stabilized P1/P1 (GLS)", "F13a": "inductionless MHD",
          "C1a": "time integrators", "C1b": "kinetics kernel", "C1c": "reactors and equilibrium",
          "C1d": "variable-order BDF", "C3a": "current distribution, 1D, planar and axisymmetric",
          "C3b": "current distribution, 3D (MPI) and porous electrode"}


def _group_key(cap: str):
    m = re.match(r"([A-Za-z]+)(\d+)(.*)", cap)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (cap, 0, "")


def _fmt(x, nd=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return str(x)
    return f"{x:.{nd}f}" if 1e-2 <= abs(x) < 1e3 or x == 0 else f"{x:.1e}"


def measured(ev: dict) -> str:
    """One-cell summary of an evaluation: what was measured and how it compares with the criteria."""
    parts = []
    conv = [c for c in ev.get("convergence", []) if c.get("observed_order") is not None]
    if conv:
        d = [float(c["observed_order"]) - float(c["expected_order"]) for c in conv
             if isinstance(c.get("observed_order"), (int, float)) and math.isfinite(float(c["observed_order"]))]
        exp = sorted({_fmt(c["expected_order"], 1) for c in conv})
        if len(conv) <= 2:
            parts.append("; ".join(f"order {_fmt(c['observed_order'])} (expected {_fmt(c['expected_order'], 1)})"
                                   for c in conv))
        elif d:
            parts.append(f"{len(conv)} orders (expected {', '.join(exp)}), observed minus expected "
                         f"{_fmt(min(d))} to {_fmt(max(d))}")
    fc = ev.get("functional_convergence", [])
    if fc:
        orders = [f.get("richardson", {}).get("order") for f in fc]
        orders = [float(o) for o in orders if isinstance(o, (int, float)) and math.isfinite(float(o))]
        parts.append(f"{len(fc)} Richardson/GCI" + (f" (orders {_fmt(min(orders))} to {_fmt(max(orders))})"
                                                       if orders else ""))
    comps = [c for c in ev.get("comparisons", []) if not c.get("informative")]
    if comps:
        r = [abs(float(c["deviation"])) / float(c["allowance"]) for c in comps
             if isinstance(c.get("deviation"), (int, float)) and isinstance(c.get("allowance"), (int, float))
             and float(c["allowance"]) > 0]
        srcs = sorted({str(c.get("source")) for c in comps if c.get("source")})
        parts.append(f"{len(comps)} comparison{'s' * (len(comps) > 1)} ({', '.join(srcs)})"
                     + (f", largest deviation {_fmt(max(r))} of the allowance" if r else ""))
    n = len(ev.get("checks", [])) + len(ev.get("solver_sensitivity", [])) + \
        sum(len(v) for v in (ev.get("extra") or {}).values())
    if n:
        parts.append(f"{n} check{'s' * (n > 1)}")
    return "; ".join(parts) or "-"


def item_counts(ev: dict) -> dict:
    """Numbers of acceptance items of an evaluation, counted as in `measured`."""
    return {"orders": sum(1 for c in ev.get("convergence", []) if c.get("observed_order") is not None),
            "richardson": len(ev.get("functional_convergence", [])),
            "comparisons": sum(1 for c in ev.get("comparisons", []) if not c.get("informative")),
            "checks": len(ev.get("checks", [])) + len(ev.get("solver_sensitivity", []))
            + sum(len(v) for v in (ev.get("extra") or {}).values())}


def cmd_report(a) -> int:
    found = {}
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name == "summary.json":
                continue
            try:
                r = json.loads(f.read_text())
            except ValueError:
                continue
            if "evaluation" not in r or "case" not in r:
                continue
            old = found.get(r["case"])
            if old is None or (old.get("quick") and not r.get("quick")):
                found[r["case"]] = r
    groups = {}
    for t, c in all_cases(a.track):
        groups.setdefault(c["capability"], []).append(c)
    counts = {"full": 0, "quick": 0, "not run": 0, "met": 0}
    items = {"orders": 0, "richardson": 0, "comparisons": 0, "checks": 0}       # over the full runs
    for cap in sorted(groups, key=_group_key):
        title = GROUPS.get(cap, cap)
        print(f"\n### {title[0].upper() + title[1:]}\n")
        print("| case | evidence | cost | run | result | measured |")
        print("|---|---|---|---|---|---|")
        for c in groups[cap]:
            r = found.get(c["id"])
            if r is None:
                mode, res, meas = "not run", "-", "-"
            else:
                ev = r["evaluation"]
                mode = "quick" if r.get("quick") else "full"
                if ev.get("criteria_met"):
                    res = "all criteria met" + (" (on two levels)" if r.get("quick") else "")
                    counts["met"] += mode == "full"
                elif r.get("quick"):
                    res = "smoke only"
                else:
                    res = ev.get("headline", "FAILED")
                meas = measured(ev)
                if mode == "full":
                    for k, v in item_counts(ev).items():
                        items[k] += v
            counts[mode] += 1
            print(f"| `{c['id']}` | {', '.join(c['levels'])} | {c['run']['runner'].get('cost', '')} | {mode} | "
                  f"{res.replace('|', '/')} | {meas.replace('|', '/')} |")
    print(f"\n{sum(len(v) for v in groups.values())} cases: {counts['full']} run in full ({counts['met']} meet all "
          f"their criteria), {counts['quick']} run in quick mode only, {counts['not run']} not run. The full runs "
          f"contain {items['orders']} observed orders of convergence, {items['richardson']} Richardson/GCI estimates, "
          f"{items['comparisons']} comparisons with reference values and {items['checks']} checks.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list the cases")
    p.add_argument("--track", choices=("fluids", "chem", "all"), default="all")
    p.add_argument("--cost", choices=("short", "long"))
    p = sub.add_parser("show", help="print a case's acceptance items")
    p.add_argument("case")
    p = sub.add_parser("run", help="run cases and check their criteria")
    p.add_argument("cases", nargs="*")
    p.add_argument("--all", action="store_true", help="every case of --track")
    p.add_argument("--track", choices=("fluids", "chem", "all"), default="all")
    p.add_argument("--cost", choices=("short", "long"), help="every case of this cost class")
    p.add_argument("--quick", action="store_true", help="first two levels only (smoke test)")
    p.add_argument("--out", help="directory for the JSON results (default: a new temporary directory)")
    p.add_argument("--jobs", type=int, default=1, help="cases run in parallel (default 1)")
    p.add_argument("--timeout", type=float, help="seconds per case (default: the case's timeout_s)")
    p = sub.add_parser("report", help="Markdown summary of saved results (run --out directories)")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--track", choices=("fluids", "chem", "all"), default="all")
    p = sub.add_parser("_one", help=argparse.SUPPRESS)
    p.add_argument("case")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list(a)
    if a.cmd == "show":
        return cmd_show(a)
    if a.cmd == "report":
        return cmd_report(a)
    if a.cmd == "_one":
        track, c = find_case(a.case)
        s = run_one(track, c, a.quick, Path(a.out))
        print(json.dumps({k: s[k] for k in ("case", "criteria_met", "headline")}))
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
