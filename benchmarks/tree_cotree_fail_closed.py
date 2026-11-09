"""Tree-cotree validation, run 6: do the ungauged paths fail closed at low frequency without validation loads?

    python benchmarks/tree_cotree_fail_closed.py run [--ranks N]
    python benchmarks/tree_cotree_fail_closed.py report

Run 6: the ungauged paths at 100 Hz and 1 kHz with no
validation loads and no source check; if no gate fires, a mandatory guard on field outputs is needed. On the two-port
frame with both checks configured (run 1), AMS refused 10 Hz-1 kHz and plain STRUMPACK 10-100 Hz, the refusals coming
from the validation loads and a failed factorization, and plain STRUMPACK accepted 1 kHz with a 5e-4 field error.

Here: the same frame (benchmarks/designs/stage1_twoport.json) without `source_check` and `validation_shunt_R_ohm`, at
10 Hz, 100 Hz and 1 kHz, on AMS, plain STRUMPACK and the gauged path (the reference; its default charge projection
makes it exact to the f^2 law there, run 1), one design per frequency. Recorded per path and frequency: refused or
not (and the gate), the warnings, and Z, Psi and U_e against the gauged path. Writes
benchmarks/records/tree_cotree_fail_closed.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

RUNS = Path(os.environ.get("AXF_TC_FC_RUNS", ROOT / "runs" / "tree_cotree_fail_closed")).resolve()
OUT = ROOT / "benchmarks" / "records" / "tree_cotree_fail_closed.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
FREQUENCIES = [1e1, 1e2, 1e3]
CONFIGS = {   # each sets every variable it relies on (os.environ carries over between runs)
    "tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2"},
    "ams": {"AXF_FW_SOLVER": "ams", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0"},
    "strumpack": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_STRUMPACK_DIST": "0",
                  "AXF_FW_TC_CHARGE_PROJECTION": "0"},
}


def design(frequency: float) -> dict:
    doc = json.loads(STAGE1.read_text())
    doc["name"] = f"tree_cotree_fail_closed_{frequency:g}Hz"
    for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
        doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [frequency]
    return doc


def solve(name: str, doc: dict, env: dict, ranks: int) -> None:
    import axionfem as af
    for k, v in env.items():
        os.environ[k] = v
    t0 = time.time()
    try:
        af.Design(doc).solve(RUNS / name, verbose=0, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (refusals are the point: the report records them)
        print(f"{name} refused: {type(exc).__name__}: {exc}", flush=True)
        (RUNS / name).mkdir(parents=True, exist_ok=True)
        (RUNS / name / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name}
    (RUNS / name / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def run(ranks: int) -> None:
    for config, env in CONFIGS.items():
        for f in FREQUENCIES:
            name = f"{config}_f{f:g}"
            if (RUNS / name / "results.json").exists() or (RUNS / name / "error.txt").exists():
                print(f"{name}: kept", flush=True)
                continue
            solve(name, design(f), env, ranks)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


def provenance() -> dict:
    import axionfem as af
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": file_record(exe), "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")),
            "python_core": file_record(af._core.__file__), "driver": file_record(__file__),
            "date": time.strftime("%Y-%m-%d")}


def rel(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


def row_values(row: dict) -> dict:
    z = np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)
    psi = np.array(row["Psi_vector"]["re"], dtype=float) + 1j * np.array(row["Psi_vector"]["im"], dtype=float)
    ue = np.diag(np.array(row["energy_electric_J_per_A2"]["re"])) if row.get("energy_electric_J_per_A2") else None
    return {"Z": z, "Psi": psi, "U_e": ue}


def report() -> None:
    rec = {"check": "tree_cotree_fail_closed", "provenance": provenance(), "frequencies_Hz": FREQUENCIES, "runs": []}
    reference = {}
    for f in FREQUENCIES:
        res = RUNS / f"tree_cotree_f{f:g}" / "results.json"
        if res.exists():
            reference[f] = row_values(json.loads(res.read_text())["fullwave"]["sweep"][0])
    for config in CONFIGS:
        for f in FREQUENCIES:
            run_dir = RUNS / f"{config}_f{f:g}"
            entry = {"config": config, "f_Hz": f}
            if (run_dir / "results.json").exists():
                d = json.loads((run_dir / "results.json").read_text())
                row = d["fullwave"]["sweep"][0]
                entry["refused"] = False
                entry["warnings"] = d.get("warnings", [])
                entry["relative_residual"] = (row.get("solver") or {}).get("relative_residual")
                entry["status"] = d["fullwave"].get("status")
                if f in reference and config != "tree_cotree":
                    v = row_values(row)
                    r = reference[f]
                    entry["Z_vs_gauged"] = rel(v["Z"], r["Z"])
                    entry["Psi_vs_gauged"] = rel(v["Psi"], r["Psi"])
                    if v["U_e"] is not None and r["U_e"] is not None:
                        entry["U_e_vs_gauged"] = [float(x / y - 1) for x, y in zip(v["U_e"], r["U_e"])]
                        entry["field_error"] = [float(np.sqrt(abs(x / y - 1))) for x, y in zip(v["U_e"], r["U_e"])]
            else:
                entry["refused"] = True
                err = run_dir / "error.txt"
                entry["error"] = err.read_text().strip() if err.exists() else "missing"
                rej = run_dir / "fullwave_rejected.json"
                if rej.exists():
                    entry["rejection"] = json.loads(rej.read_text())["primary_row"].get("rejection")
            rec["runs"].append(entry)
            summary = "refused: " + entry.get("error", "")[:120] if entry["refused"] else (
                f"accepted; Z {entry.get('Z_vs_gauged', float('nan')):.1e}, Psi {entry.get('Psi_vs_gauged', float('nan')):.1e}, "
                f"field error {entry.get('field_error')}; warnings {len(entry['warnings'])}")
            print(f"{config} {f:g} Hz: {summary}", flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    # the committed record only from the default run directory (a trial with AXF_TC_FC_RUNS writes its own copy)
    if RUNS == (ROOT / "runs" / "tree_cotree_fail_closed").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_TC_FC_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        run(int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4")))
    report()


if __name__ == "__main__":
    main()
