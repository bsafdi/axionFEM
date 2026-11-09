"""The fluids and chemistry verification cases (beta): case definitions, the acceptance check and the runner.

Fast: the case files are only loaded and checked for consistency, the acceptance check is exercised on synthetic
results, and two chemistry cases that take milliseconds run end to end through benchmarks/fluids/run.py.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FLUIDS = ROOT / "benchmarks" / "fluids"
TRACKS = {"fluids": FLUIDS, "chem": ROOT / "benchmarks" / "chem"}
LEVELS = {"mms", "analytic", "literature", "conservation", "cross_code"}
ITEM_KINDS = {"convergence", "functional_convergence", "comparisons", "checks", "solver_sensitivity",
              "quadrature_sensitivity", "tolerance_behaviour", "solver"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load(FLUIDS / "acceptance.py", "acceptance_under_test")
RUN = _load(FLUIDS / "run.py", "run_under_test")


def _cases():
    out = []
    for track, d in TRACKS.items():
        for f in sorted((d / "registry" / "cases").glob("*.json")):
            out.append((track, f))
    return out


CASES = _cases()


def test_there_are_cases_for_both_tracks():
    tracks = {t for t, _ in CASES}
    assert tracks == {"fluids", "chem"}
    assert len(CASES) >= 100


@pytest.mark.parametrize("track,path", CASES, ids=[p.stem for _, p in CASES])
def test_case_definition_is_complete(track, path):
    c = json.loads(path.read_text())
    assert c["id"] == path.stem
    assert set(c) <= {"id", "capability", "title", "physics", "levels", "run", "reference", "acceptance"}
    assert c["title"] and c["capability"]
    assert set(c["levels"]) <= LEVELS and c["levels"]
    r = c["run"]["runner"]
    assert set(r) <= {"module", "function", "check_module", "cost", "timeout_s"}
    assert r["cost"] in ("short", "long") and float(r["timeout_s"]) > 0
    src = (TRACKS[track] / "cases" / f"{r['module']}.py").read_text()
    assert re.search(rf"^(def {re.escape(r['function'])}\b|{re.escape(r['function'])} = )", src, re.M), r
    if r.get("check_module"):
        assert (TRACKS[track] / "cases" / f"{r['check_module']}.py").is_file()
    acc = c["acceptance"]
    assert ITEM_KINDS & set(acc), "no acceptance item"
    refs = (c.get("reference") or {}).get("quantities") or {}
    for kind in ITEM_KINDS - {"solver"}:
        for item in acc.get(kind, []) or []:
            assert item["level"] in LEVELS, (kind, item)
            if kind == "comparisons":
                assert item["reference"] in refs, item["reference"]


def _data_entries(o):
    if isinstance(o, dict):
        if isinstance(o.get("path"), str) and isinstance(o.get("sha256"), str):
            yield o
        for v in o.values():
            yield from _data_entries(v)
    elif isinstance(o, list):
        for v in o:
            yield from _data_entries(v)


def test_every_data_file_a_case_names_exists_with_its_sha256():
    seen = {}
    for _, path in CASES:
        for e in _data_entries(json.loads(path.read_text())["run"]["parameters"]):
            seen[e["path"]] = e["sha256"]
    assert seen, "the cases name no data files"
    for rel, sha in sorted(seen.items()):
        f = ROOT / rel
        assert f.is_file(), rel
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, rel


# ------------------------------------------------------------------------------------- the acceptance check
def _conv_case(expected=2.0, tol=0.15):
    levels = [8, 16, 32, 64]
    return {"id": "synthetic", "levels": ["mms"],
            "run": {"parameters": {"meshes": {"h_nominal": [1.0 / n for n in levels]}}},
            "reference": {"quantities": {"one": {"kind": "scalar", "value": 1.0, "uncertainty": 0.0}}},
            "acceptance": {"convergence": [{"series": "uL2", "level": "mms", "expected_order": expected,
                                            "order_tolerance": tol, "levels_from": "meshes.h_nominal",
                                            "abscissa": {"kind": "h_nominal", "dim": 2},
                                            "mesh_family": {"kind": "nested"}, "estimator": "fit",
                                            "fit_points": 3, "min_levels": 4, "floor_rel": 1e-10,
                                            "solution_scale": 3.0, "min_refinement_ratio": 1.9,
                                            "max_downward_drift": 0.1, "min_span": 4}],
                           "solver": {"max_residual": 1e-10, "residual_key": "relative_residual"},
                           "comparisons": [{"quantity": "q", "reference": "one", "level": "analytic",
                                            "tolerance": {"rel": 1e-6}}],
                           "checks": [{"quantity": "defect", "level": "conservation", "max": 1e-12}]}}


def _conv_result(order=2.0, q=1.0, defect=0.0, converged=True):
    hs = [1.0 / n for n in (8, 16, 32, 64)]
    runs = [{"level": f"L{i}", "mesh": {"h_nominal": h, "family": "f", "refinement_level": i, "n_elements": 2 * 4 ** (i + 3)},
             "solver": {"converged": converged, "relative_residual": 1e-14}} for i, h in enumerate(hs)]
    return {"runs": runs, "series": {"uL2": {"levels": [r["level"] for r in runs], "solution_magnitude": 3.0,
                                              "error": [3.0 * h ** order for h in hs]}},
            "quantities": {"q": q, "defect": defect}}


def test_acceptance_passes_a_correct_result():
    ev = A.evaluate(_conv_case(), _conv_result())
    assert ev["criteria_met"], ev["headline"]
    assert ev["verdicts"]["self_consistency"] == "PASS" and ev["verdicts"]["exact_solution"] == "PASS"
    assert ev["verdicts"]["literature"] == "NOT_TESTED" and ev["verdicts"]["cross_code"] == "NOT_TESTED"


@pytest.mark.parametrize("bad", [dict(order=1.0), dict(q=1.01), dict(defect=1e-6), dict(converged=False)])
def test_acceptance_fails_a_wrong_result(bad):
    ev = A.evaluate(_conv_case(), _conv_result(**bad))
    assert not ev["criteria_met"]
    assert ev["headline"].startswith("FAILED")


def test_acceptance_reports_a_malformed_result_instead_of_crashing():
    res = _conv_result()
    res["series"]["uL2"]["error"] = ["not a number"] * 4
    del res["quantities"]["q"]
    ev = A.evaluate(_conv_case(), res)
    assert not ev["criteria_met"]
    assert any(c["verdict"] == "FAIL" for c in ev["convergence"] + ev["comparisons"])


def test_quick_mode_judges_the_first_levels_only():
    case, res = _conv_case(), _conv_result()
    for k in ("levels", "error"):
        res["series"]["uL2"][k] = res["series"]["uL2"][k][:2]
    res["runs"] = res["runs"][:2]
    ev = A.evaluate(case, res, quick=True)
    assert ev["convergence"][0]["verdict"] == "FAIL"    # two levels are too few for an order study
    assert ev["comparisons"][0]["verdict"] == "PASS" and ev["checks"][0]["verdict"] == "PASS"


def test_fit_order_and_pairwise_orders():
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    e = 2.0 * h ** 3
    assert abs(A.fit_order(h, e)["order"] - 3.0) < 1e-12
    assert np.allclose(A.pairwise_orders(h, e), 3.0)


def test_richardson_recovers_the_order_and_the_limit():
    h = [0.4, 0.2, 0.1]
    v = [1.0 + 0.5 * x ** 2 for x in h]
    r = A.richardson(h, v)
    assert abs(r["order"] - 2.0) < 1e-9 and abs(r["extrapolated"] - 1.0) < 1e-9


# ------------------------------------------------------------------------------------------------ the runner
def test_runner_lists_every_case():
    out = subprocess.run([sys.executable, str(FLUIDS / "run.py"), "list"], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == f"{len(CASES)} cases"
    out = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "chem" / "run.py"), "list"], capture_output=True,
                         text=True, check=True)
    assert all(" chem " in ln for ln in out.stdout.strip().splitlines()[:-1])


def test_runner_shows_a_case():
    out = subprocess.run([sys.executable, str(FLUIDS / "run.py"), "show", "analytic_couette_2d"], capture_output=True,
                         text=True, check=True)
    assert "comparison" in out.stdout and "analytic_couette_2d" in out.stdout


def test_runner_refuses_an_unknown_case():
    out = subprocess.run([sys.executable, str(FLUIDS / "run.py"), "show", "no_such_case"], capture_output=True,
                         text=True)
    assert out.returncode != 0 and "no case" in out.stderr + out.stdout


def test_runner_runs_two_short_chemistry_cases(tmp_path):
    pytest.importorskip("axionfem.chem")
    argv = [sys.executable, str(FLUIDS / "run.py"), "run", "chem_kernel_rates", "chem_time_lstability",
            "--out", str(tmp_path), "--jobs", "2"]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [s["case"] for s in summary] == ["chem_kernel_rates", "chem_time_lstability"]
    assert all(s["status"] == "PASS" for s in summary)
    res = json.loads((tmp_path / "chem_kernel_rates.json").read_text())
    assert res["criteria_met"] and res["evaluation"]["criteria_met"] and res["result"]["quantities"]
    assert not any(math.isnan(v) for v in res["result"]["quantities"].values() if isinstance(v, float))
    rep = subprocess.run([sys.executable, str(FLUIDS / "run.py"), "report", str(tmp_path), "--track", "chem"],
                         capture_output=True, text=True, check=True).stdout
    row = next(ln for ln in rep.splitlines() if ln.startswith("| `chem_kernel_rates`"))
    assert "| full | all criteria met |" in row and "comparisons" in row
    assert "2 run in full (2 meet all their criteria)" in rep


@pytest.mark.parametrize("track,page", [("fluids", "fluids.md"), ("chem", "chemistry.md")])
def test_validation_page_lists_every_case(track, page):
    text = (ROOT / "docs" / "validation" / page).read_text()
    missing = [p.stem for t, p in CASES if t == track and f"`{p.stem}`" not in text]
    assert not missing, f"docs/validation/{page} does not list {missing} (regenerate it with run.py report)"
