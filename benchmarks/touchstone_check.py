"""Touchstone export of full-wave port results against closed forms and the engine's own checks.

    python benchmarks/touchstone_check.py [CASE ...]      (default: every case)
    (as a batch job: the same command inside the allocation)

Cases (record: benchmarks/records/touchstone_check.json; files: runs/touchstone_check/):

- tem_short: the shorted parallel-plate TEM line (3D, surface port, order 2) with an adaptive sweep over 100-400 MHz.
  Its exact Z is -i Zc tan(kl) in the engine's exp(-i omega t), so S = (j Zc tan(kl) - z0)/(j Zc tan(kl) + z0) in the
  file's exp(+j omega t): |S| = 1 and the phase turns clockwise. Compares the .s1p of the solved rows and of the dense
  model curve with it (S stays bounded through the quarter-wave pole, unlike Z).
- coax2d_lossy: the axisymmetric sealed coax with a lossy dielectric (tan delta 1e-3) and two shorted stubs, the
  closed form of tests/test_fullwave.py; |S| < 1.
- stage1_adaptive: the two-port frame (benchmarks/designs/stage1_twoport.json) with an adaptive sweep over
  115-150 MHz: reciprocity and passivity of the 2 x 2 S from the rows and from the dense curve.
- records: the committed multiport record of the two-port frame (lossless, reversed port, lossy, floating port).
- skrf: when scikit-rf is importable, every file written above read back by scikit-rf and compared.
"""
from __future__ import annotations

import cmath
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

import axionfem as af  # noqa: E402
from axionfem import touchstone as ts  # noqa: E402

OUT = ROOT / "runs" / "touchstone_check"
RECORD = ROOT / "benchmarks" / "records" / "touchstone_check.json"
C0 = 299792458.0
ETA = 376.730313668
Z0_REF = 50.0
RANKS = int(os.environ.get("SLURM_NTASKS", "1"))


def _s_exact(z_engine) -> np.ndarray:
    z = np.conj(np.asarray(z_engine, dtype=complex))
    return (z - Z0_REF) / (z + Z0_REF)


def _summary(rep: dict) -> dict:
    keep = ("path", "kind", "ports", "n_frequencies", "f_min_Hz", "f_max_Hz", "sigma_max", "passivity_violations_Hz",
            "reciprocity_error", "source", "warnings")
    out = {k: rep[k] for k in keep if k in rep}
    out["path"] = str(Path(rep["path"]).relative_to(ROOT))
    return out


def _quiet_exports(data: dict, wd: Path, stem: str, n: int) -> dict:
    """Z and Y exports of the same results (rows and, when present, the dense curve): the warnings each raises, the
    largest estimated relative error of Z, the Y check and the frequencies it flags (ordinary
    exports must be quiet unless something is wrong)."""
    out = {}
    dense_opts = (False, True) if (data["fullwave"].get("adaptive") or {}).get("dense") else (False,)
    for dense in dense_opts:
        for kind in ("Z", "Y"):
            tag = f"{'dense' if dense else 'rows'}_{kind}"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rep = ts.write_touchstone(data, wd / f"{stem}_{tag}.s{n}p", kind=kind, dense=dense)
            out[tag] = {"python_warnings": [str(x.message) for x in w],
                        "Z_relative_error_max": rep["Z_relative_error_max"],
                        "Z_poorly_determined": len(rep["Z_poorly_determined_Hz"]),
                        "Y_check": rep["Y_check"], "Y_ill_determined_Hz": rep["Y_ill_determined_Hz"]}
    return out


def case_tem_short() -> dict:
    import fullwave_indicator as FWI

    wd = OUT / "tem_short"
    doc, geometry = FWI.tem_design(wd, "pec", 2, 0, [1e8], nx=6)
    doc["fullwave"].pop("frequencies")
    doc["fullwave"].update({"adaptive": {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-5, "spacing": "linear"},
                            "modes": 2})
    t0 = time.time()
    data = af.Design(doc).solve(wd, verbose=0, mpi_ranks=RANKS).data
    out = {"solve_seconds": time.time() - t0, "geometry": geometry,
           "adaptive": {k: data["fullwave"]["adaptive"].get(k) for k in ("converged", "solves", "estimated_error")}}
    for dense in (False, True):
        tag = "dense" if dense else "rows"
        rep = ts.write_touchstone(data, wd / f"tem_short_{tag}.s1p", dense=dense)
        back = ts.read_touchstone(rep["path"])
        f = back["f_Hz"]
        exact = _s_exact([FWI.tem_exact_z("pec", x, geometry) for x in f])
        S = back["S"][:, 0, 0]
        pole = C0 / (4 * geometry["l"])
        below = f < pole
        away = np.abs(f / pole - 1) > 0.05
        out[tag] = {**_summary(rep),
                    "max_abs_S_minus_exact": float(np.max(np.abs(S - exact))),
                    "max_abs_S_minus_exact_away_from_pole": float(np.max(np.abs(S - exact)[away])),
                    "max_abs_1_minus_abs_S": float(np.max(np.abs(1 - np.abs(S)))),
                    "Im_S_positive_below_pole": bool(np.all(S[below & (f < 0.5 * pole)].imag > 0)),
                    "first": {"f_Hz": float(f[0]), "S": [S[0].real, S[0].imag],
                              "S_exact": [exact[0].real, exact[0].imag]}}
    out["other_kinds"] = _quiet_exports(data, wd, "tem_short", 1)
    return out


def case_coax2d_lossy() -> dict:
    eps_r, tand = 2.2, 1e-3
    a1, b1, t, g, l1, l2 = 0.05, 0.15, 0.02, 0.005, 0.6, 0.3
    freqs = [1e6, 2e7, 5e7, 8e7, 1e8, 1.2e8]
    body = {"union": [{"rect": {"r": [0.0, a1], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [g + l1, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [-l2 - t, -l2]}}]}
    regions = [{"name": "coax", "shape": body, "material": "superconductor"},
               {"name": "fill1", "shape": {"rect": {"r": [a1, b1], "z": [g, g + l1]}}, "material": "diel"},
               {"name": "fill2", "shape": {"rect": {"r": [a1, b1], "z": [-l2, 0.0]}}, "material": "diel"},
               {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [0.0, g]}}, "material": "vacuum"},
               {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}}, "material": "winding"}]
    doc = {"schema": "axionfem.design/1", "name": "coax2d_lossy", "regions": regions,
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.3]},
           "fullwave": {"frequencies": freqs},
           "materials": {"diel": {"kind": "linear", "eps_r": eps_r, "tan_delta": tand}},
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.0075}}
    wd = OUT / "coax2d_lossy"
    t0 = time.time()
    data = af.Design(doc).solve(wd, verbose=0).data
    rep = ts.write_touchstone(data, wd / "coax2d_lossy.s1p")
    back = ts.read_touchstone(rep["path"])
    ec = eps_r * (1 + 1j * tand)
    Zc = ETA / (2 * math.pi) * math.log(b1 / a1) / cmath.sqrt(ec)
    Zv = ETA / (2 * math.pi) * math.log(b1 / a1)
    exact = []
    for f in back["f_Hz"]:
        k, k0 = 2 * math.pi * f / C0 * cmath.sqrt(ec), 2 * math.pi * f / C0

        def stub(length):   # engine convention, as in tests/test_fullwave.py
            ZL = -1j * Zc * cmath.tan(k * length)
            tt = cmath.tan(k0 * g / 2)
            return Zv * (ZL - 1j * Zv * tt) / (Zv - 1j * ZL * tt)
        Z1, Z2 = stub(l1), stub(l2)
        exact.append(Z1 * Z2 / (Z1 + Z2))
    S = back["S"][:, 0, 0]
    Se = _s_exact(exact)
    return {**_summary(rep), "solve_seconds": time.time() - t0,
            "max_abs_S_minus_exact": float(np.max(np.abs(S - Se))),
            "max_relative_Z_error": float(np.max(np.abs(back["Z"][:, 0, 0] / np.conj(exact) - 1))),
            "abs_S": np.abs(S).tolist(), "abs_S_exact": np.abs(Se).tolist(),
            "all_abs_S_below_1": bool(np.all(np.abs(S) < 1))}


def case_stage1_adaptive() -> dict:
    doc = json.loads((ROOT / "benchmarks" / "designs" / "stage1_twoport.json").read_text())
    fw = doc["fullwave"]
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    fw["adaptive"] = {"f_min_Hz": 1.15e8, "f_max_Hz": 1.5e8, "tolerance": 1e-3, "spacing": "linear"}
    wd = OUT / "stage1_adaptive"
    t0 = time.time()
    data = af.Design(doc).solve(wd, verbose=0, mpi_ranks=RANKS).data
    out = {"solve_seconds": time.time() - t0,
           "adaptive": {k: data["fullwave"]["adaptive"].get(k) for k in ("converged", "solves", "estimated_error")},
           "self_resonances_Hz": [r.get("f_Hz") for r in data["fullwave"].get("self_resonances", [])]}
    for dense in (False, True):
        tag = "dense" if dense else "rows"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ts.write_touchstone(data, wd / f"stage1_{tag}.s2p", dense=dense)
        S = ts.read_touchstone(rep["path"])["S"]
        out[tag] = {**_summary(rep), "max_S_asymmetry": float(np.max(ts.reciprocity_error(S))),
                    "max_sigma_minus_1": float(np.max(ts.passivity(S)["excess"])),
                    "python_warnings": [str(x.message) for x in w]}
    out["other_kinds"] = _quiet_exports(data, wd, "stage1", 2)
    return out


def case_records() -> dict:
    rec = json.loads((ROOT / "benchmarks" / "records" / "multiport_stage1.json").read_text())
    out = {}
    for case in ("multiport", "reversed", "lossy", "floating_port"):
        rows = rec["cases"][case]["sweep"]
        res = {"design": {"name": f"stage1_{case}"}, "fullwave": {"sweep": rows}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ts.write_touchstone(res, OUT / "records" / f"stage1_{case}.s2p")
        S = ts.read_touchstone(rep["path"])["S"]
        out[case] = {**_summary(rep), "max_S_asymmetry": float(np.max(ts.reciprocity_error(S))),
                     "sigma_max_per_f": ts.passivity(S)["sigma_max"].tolist(),
                     "min_eig_1_minus_SHS": [float(np.linalg.eigvalsh(np.eye(2) - M.conj().T @ M).min()) for M in S],
                     "python_warnings": [str(x.message) for x in w]}
    return out


def case_skrf(files: list[str]) -> dict:
    try:
        import skrf
    except ImportError:
        return {"available": False}
    out = {"available": True, "version": skrf.__version__, "files": {}}
    for p in files:
        mine = ts.read_touchstone(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = skrf.Network(p)
        out["files"][Path(p).name] = {
            "max_abs_S_difference": float(np.max(np.abs(net.s - mine["S"]))),
            "max_relative_Z_difference": float(np.max(np.abs(net.z - mine["Z"]) / np.abs(mine["Z"]).max())),
            "f_equal": bool(np.allclose(net.f, mine["f_Hz"], rtol=1e-15)),
            "skrf_reciprocal": bool(net.is_reciprocal(tol=1e-9)) if net.nports > 1 else None}
    return out


CASES = {"tem_short": case_tem_short, "coax2d_lossy": case_coax2d_lossy, "stage1_adaptive": case_stage1_adaptive,
         "records": case_records}


# the gates of the record (also checked on the committed record by tests/test_touchstone.py)
def evaluate(record: dict) -> list[str]:
    """The failed checks of a record (empty: every gate holds)."""
    fails = []
    c = record.get("cases", {})

    def gate(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    for name in list(CASES):
        gate(name in c, f"case {name} missing")
    if "tem_short" in c:
        for tag in ("rows", "dense"):
            x = c["tem_short"][tag]
            gate(x["max_abs_S_minus_exact"] < 5e-5, f"tem_short {tag}: |S - S_exact| {x['max_abs_S_minus_exact']}")
            gate(x["max_abs_1_minus_abs_S"] < 1e-9, f"tem_short {tag}: ||S| - 1| {x['max_abs_1_minus_abs_S']}")
            gate(x["Im_S_positive_below_pole"], f"tem_short {tag}: Im S11 sign below the pole")
            gate(not x["passivity_violations_Hz"], f"tem_short {tag}: passivity violations")
    if "coax2d_lossy" in c:
        x = c["coax2d_lossy"]
        gate(x["all_abs_S_below_1"], "coax2d_lossy: |S| >= 1 somewhere")
        gate(x["max_abs_S_minus_exact"] < 5e-3, f"coax2d_lossy: |S - S_exact| {x['max_abs_S_minus_exact']}")
    for name in ("tem_short", "stage1_adaptive"):
        for tag, x in (c.get(name, {}).get("other_kinds") or {}).items():
            gate(not x["python_warnings"], f"{name} {tag}: warnings on an ordinary export: {x['python_warnings']}")
    if "stage1_adaptive" in c:
        for tag in ("rows", "dense"):
            x = c["stage1_adaptive"][tag]
            gate(x["max_sigma_minus_1"] < ts.PASSIVITY_TOLERANCE, f"stage1_adaptive {tag}: sigma - 1")
            gate(x["max_S_asymmetry"] < 1e-8, f"stage1_adaptive {tag}: S asymmetry {x['max_S_asymmetry']}")
    if "records" in c:
        for case in ("multiport", "reversed", "floating_port"):
            x = c["records"][case]
            gate(max(x["sigma_max_per_f"]) - 1 < ts.PASSIVITY_TOLERANCE, f"records {case}: sigma - 1")
        gate(min(c["records"]["lossy"]["min_eig_1_minus_SHS"]) > 0, "records lossy: 1 - S^H S not positive")
    return fails + skrf_gate(record)[0]


def skrf_gate(record: dict) -> tuple[list[str], list[str]]:
    """The scikit-rf gate: (failures, skipped).  scikit-rf is optional, so a run without it records the skip instead
    of passing silently; with it, every file this run wrote must be read to an identical S."""
    sk = record.get("cases", {}).get("skrf")
    if sk is None:
        return [], ["skrf: case not run"]
    if not sk.get("available"):
        return [], ["skrf: scikit-rf not importable, files not cross-checked"]
    files = sk.get("files") or {}
    fails = []
    if not files or len(files) != sk.get("expected_files", len(files)):
        fails.append(f"skrf: read {len(files)} files, expected {sk.get('expected_files')}")
    for fname, x in files.items():
        if not (x["max_abs_S_difference"] == 0.0 and x["max_relative_Z_difference"] < 1e-10 and x["f_equal"]):
            fails.append(f"skrf {fname}: differs")
    return fails, []


def _written(obj) -> list[str]:
    """The Touchstone files a case reports (its 'path' entries)."""
    if isinstance(obj, dict):
        out = [str(ROOT / obj["path"])] if isinstance(obj.get("path"), str) and obj["path"].endswith("p") else []
        for v in obj.values():
            out += _written(v)
        return out
    return []


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    names = argv or list(CASES) + ["skrf"]
    # a full run writes a fresh record; a partial run replaces only its cases, each stamped with its own run
    record = json.loads(RECORD.read_text()) if argv and RECORD.exists() else {}
    stamp = {"date": time.strftime("%Y-%m-%d"), "git_head": _git_head(),
             "version": af.version()}
    record.update({"check": "Touchstone export (axionfem.touchstone) against closed forms and the engine's checks",
                   "ranks": RANKS, "z0_ohm": Z0_REF})
    if not argv:
        record.update(stamp)   # a partial run stamps only its own cases
    record.setdefault("cases", {})
    written: list[str] = []
    for name in names:
        if name == "skrf":
            continue
        print(f"[touchstone_check] {name}", flush=True)
        out = CASES[name]()
        out.update(stamp)
        record["cases"][name] = out
        written += _written(out)
        RECORD.write_text(json.dumps(record, indent=1) + "\n")
    if "skrf" in names:
        # only the files this run wrote, never leftovers in runs/
        files = sorted(set(written))
        record["cases"]["skrf"] = {**case_skrf(files), "expected_files": len(files), **stamp}
    fails = evaluate(record)
    record["gates_failed"] = fails
    record["gates_skipped"] = skrf_gate(record)[1]
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1)[:6000])
    skipped = record["gates_skipped"]
    print("TOUCHSTONE GATES " + ("OK" if not fails else "FAILED: " + "; ".join(fails))
          + (f" (skipped: {'; '.join(skipped)})" if skipped else ""), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
