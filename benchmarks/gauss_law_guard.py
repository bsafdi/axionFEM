"""Gauss-law guard on the ungauged full-wave paths (AXF_FW_GAUSS_GUARD; FwGaussGuard in cpp/src/model3d_fullwave.cpp).

    python benchmarks/gauss_law_guard.py run [--ranks N] [--parts A,B,C]
    python benchmarks/gauss_law_guard.py report

Run 6 of the tree-cotree validation (benchmarks/tree_cotree_fail_closed.py) found the ungauged paths (AMS, plain STRUMPACK)
accepting 10 Hz-1 kHz on the two-port frame with field errors up to x13-17 when no validation loads are configured.
The guard evaluates the discrete Gauss-law residual of every ungauged solve, estimates from it the gradient part of
the field error (field_fraction, energy norm) and, by default, removes it. Here:

A. fields: the run-6 design (no validation loads, no source check), fields exported, at 10 Hz, 100 Hz, 1 kHz and
   100 kHz, on AMS and plain STRUMPACK with the guard estimating only (the fields as before the guard) and
   correcting, against the gauged path (the reference). Per solve: the guard's estimate and the measured field
   error of each port's field (relative l2 norm of the exported DOF vectors, same mesh and partition), and Z, Psi
   and U_e against the reference.
B. gates: the run-1 design (validation loads and the source check) at 100 Hz and 1 kHz on AMS and plain STRUMPACK,
   estimate and correct: which validation gates refuse, and the loaded-field closure.
C. order 2: the run-6 design on an order-2 mesh (the gauged path is lowest order only), AMS estimate and correct
   from 10 Hz to 100 kHz: U_e / f^2 against its 100 kHz value (the quasi-static law), and the estimate.
D. several floating conductors: the tree-cotree run-4 topology design (five floating PEC sets, a genus-1 torus, a
   thin plate; validation shunts and the source check) at 100 Hz and 1 kHz against the gauged path.
E. charged ports and loads: the run-2 capacitor with shunts on all three ports at 1 and 10 MHz. Its third port
   bridges two floating conductors: Z of that port carries the gradient field itself, and its load column has real
   conductor charge (the Woodbury term with W != 0).
F-I: lossy closed models (complex mass; AMS; no gauged reference, so quasi-static laws and the loaded-field closure):
F. the two-port frame with a dielectric slab (tan delta 1e-2), validation shunts and the source check;
G. a resistive block (sigma 100 S/m, sigma / (omega eps0) = 1.8e10 at 100 Hz) instead;
H. a copper block (1.0e16 at 100 Hz): the correction's floor near a good conductor and its floor_estimate;
I. the part-E capacitor with a conducting gap: a charged load column with loss (the complex Woodbury term).

The committed record is written only from the default run directory (AXF_GUARD_RUNS unset).

Writes benchmarks/records/gauss_law_guard.json.
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

RUNS = Path(os.environ.get("AXF_GUARD_RUNS", ROOT / "runs" / "gauss_law_guard")).resolve()
OUT = ROOT / "benchmarks" / "records" / "gauss_law_guard.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"

GAUGED = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2",
          "AXF_FW_GAUSS_GUARD": "off"}


def ungauged(solver: str, guard: str) -> dict:
    env = {"AXF_FW_SOLVER": solver, "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
           "AXF_FW_GAUSS_GUARD": guard}
    if solver == "strumpack":
        env["AXF_FW_STRUMPACK_DIST"] = "0"
    return env


# each configuration sets every variable it relies on (os.environ carries over between solves)
PARTS = {
    "A": {"frequencies": [1e1, 1e2, 1e3, 1e5], "loads": False, "order": 1, "export": True,
          "configs": {"tree_cotree": GAUGED,
                      "ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct"),
                      "strumpack_estimate": ungauged("strumpack", "estimate"),
                      "strumpack_correct": ungauged("strumpack", "correct")}},
    "B": {"frequencies": [1e2, 1e3], "loads": True, "order": 1, "export": False,
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct"),
                      "strumpack_estimate": ungauged("strumpack", "estimate"),
                      "strumpack_correct": ungauged("strumpack", "correct")}},
    "C": {"frequencies": [1e1, 1e2, 1e3, 1e4, 1e5], "loads": False, "order": 2, "export": False,
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct")}},
    # the tree-cotree run-4 topology design (frame, two floating spheres, a floating torus, a 2 mm plate, a rod on the
    # wall; validation shunts and the source check): five floating PEC sets
    "D": {"model": "topology", "frequencies": [1e2, 1e3], "export": True,
          "configs": {"tree_cotree": GAUGED, "ams_estimate": ungauged("ams", "estimate"),
                      "ams_correct": ungauged("ams", "correct"), "strumpack_correct": ungauged("strumpack", "correct")}},
    # the run-2 capacitor (straight, s = 0.01) with shunts on all three ports: its third port bridges two floating
    # conductors, so the feed and its load column carry real conductor charge (W = G_c^T U != 0 in the Woodbury
    # term), at frequencies where the ungauged paths pass their gates
    "E": {"model": "capacitor", "frequencies": [1e6, 1e7], "export": True,
          "configs": {"tree_cotree": GAUGED, "strumpack_estimate": ungauged("strumpack", "estimate"),
                      "strumpack_correct": ungauged("strumpack", "correct"), "ams_correct": ungauged("ams", "correct")}},
    # lossy closed models (complex mass; AMS only, no gauged reference): the two-port frame with a dielectric slab
    # (eps_r 2, tan delta 1e-2) beside it, validation shunts and the source check. Checks: U_e / f^2 and R_bulk / f^3
    # against their 100 kHz values (the quasi-static laws), and the loaded-field closure.
    "F": {"model": "stage1", "loads": True, "order": 1, "export": False, "slab": "dielectric",
          "frequencies": [1e1, 1e2, 1e3, 1e5],
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct")}},
    # a resistive block (sigma 100 S/m: sigma / (omega eps0) = 1.8e10 at 100 Hz) instead: the complex S-solve's
    # robustness to a large loss, the gates and the closure
    "G": {"model": "stage1", "loads": True, "order": 1, "export": False, "slab": "resistive",
          "frequencies": [1e2, 1e4],
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct")}},
    # the same block of copper (sigma / (omega eps0) = 1.0e16 at 100 Hz): the floor of the correction near a good
    # conductor, its floor_estimate and warning, and the skin effect
    # clearing it at 10 kHz
    "H": {"model": "stage1", "loads": True, "order": 1, "export": False, "slab": "copper",
          "frequencies": [1e2, 1e4],
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct")}},
    # the part-E capacitor with a conducting gap (sigma 1e-3 S/m: loss ratio ~2 at 10 MHz): a charged load column with
    # loss, i.e. the complex Woodbury term with W != 0
    "I": {"model": "capacitor", "gap_sigma": 1e-3, "frequencies": [1e6, 1e7], "export": False,
          "configs": {"ams_estimate": ungauged("ams", "estimate"), "ams_correct": ungauged("ams", "correct")}},
}
SLABS = {"dielectric": {"kind": "linear", "eps_r": 2.0, "tan_delta": 1e-2},
         "resistive": {"kind": "linear", "eps_r": 1.0, "sigma_S_m": 100.0},
         "copper": {"kind": "linear", "eps_r": 1.0, "sigma_S_m": 5.8e7}}


def design(part: str, frequency: float, run_dir: Path) -> tuple[dict, dict]:
    """The design document and the extra solve() arguments of one run."""
    spec = PARTS[part]
    model = spec.get("model", "stage1")
    kwargs = {}
    if model == "topology":
        import tree_cotree_topology
        doc = tree_cotree_topology.design(frequency)
    elif model == "capacitor":
        import tree_cotree_capacitor
        doc = tree_cotree_capacitor.design(0.01, 1, [frequency])
        doc["fullwave"]["validation_shunt_R_ohm"] = [10, 17, 50]
        if spec.get("gap_sigma"):   # a conducting capacitor gap
            doc.setdefault("materials", {})["gap_loss"] = {"kind": "linear", "eps_r": 1.0,
                                                          "sigma_S_m": spec["gap_sigma"]}
            for region in doc["regions"]:
                if region["name"] == "cap_gap":
                    region["material"] = "gap_loss"
        kwargs = {"order": 1}
    else:
        doc = json.loads(STAGE1.read_text())
        if not spec["loads"]:
            for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
                doc["fullwave"].pop(key, None)
        doc["fullwave"]["frequencies"] = [frequency]
        doc["mesh"]["order"] = spec["order"]
        if spec.get("slab"):   # a lossy block beside the frame, away from the ports
            doc["materials"] = {"slab": SLABS[spec["slab"]]}
            doc["regions"].append({"name": "slab", "material": "slab",
                                   "solid": {"box": {"min": [0.45, -0.05, -0.05], "max": [0.55, 0.05, 0.05]}}})
    doc["name"] = f"gauss_law_guard_{part}_{frequency:g}Hz"
    if spec["export"]:
        doc["fullwave"]["field_export_directory"] = str(run_dir / "fields")
    return doc, kwargs


def solve(part: str, name: str, frequency: float, env: dict, ranks: int) -> None:
    import axionfem as af
    for k, v in env.items():
        os.environ[k] = v
    run_dir = RUNS / part / name
    t0 = time.time()
    try:
        doc, kwargs = design(part, frequency, run_dir)
        af.Design(doc).solve(run_dir, verbose=0, mpi_ranks=ranks, **kwargs)
        print(f"{part} {name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (refusals are recorded, not fatal)
        print(f"{part} {name} refused: {type(exc).__name__}: {exc}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name,
                "seconds": time.time() - t0}
    (run_dir / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def run(ranks: int, parts: list[str]) -> None:
    for part in parts:
        for config, env in PARTS[part]["configs"].items():
            for f in PARTS[part]["frequencies"]:
                name = f"{config}_f{f:g}"
                run_dir = RUNS / part / name
                if (run_dir / "results.json").exists() or (run_dir / "error.txt").exists():
                    print(f"{part} {name}: kept", flush=True)
                    continue
                solve(part, name, f, env, ranks)


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
    um = np.diag(np.array(row["energy_magnetic_J_per_A2"]["re"])) if row.get("energy_magnetic_J_per_A2") else None
    return {"Z": z, "Psi": psi, "U_e": ue, "U_m": um}


def guard_solves(row: dict) -> dict:
    """The guard's statistics of every solve in a sweep row: each port, then each validation solve."""
    out = {}
    for port in row.get("port_solutions", []):
        stats = (port.get("solver") or {}).get("gauss_law_guard")
        if stats:
            out[f"port_{port['port_index']}"] = stats
    for key, value in row.items():
        if key.endswith("_solver") and isinstance(value, dict) and value.get("gauss_law_guard"):
            out[key] = value["gauss_law_guard"]
    return out


def read_gf(path: Path) -> np.ndarray:
    """The values of an MFEM GridFunction text file (header, blank line, one value per line)."""
    lines = path.read_text().splitlines()
    start = lines.index("") + 1
    return np.array([float(x) for x in lines[start:] if x.strip()])


def port_field(run_dir: Path, port: int) -> np.ndarray | None:
    files = sorted((run_dir / "fields").glob(f"f_0_port_{port}_rank_*_re.gf"),
                   key=lambda p: int(p.name.split("_rank_")[1].split("_")[0]))
    if not files:
        return None
    parts = []
    for re_file in files:
        im_file = re_file.with_name(re_file.name.replace("_re.gf", "_im.gf"))
        parts.append(read_gf(re_file))
        parts.append(read_gf(im_file))
    return np.concatenate(parts)


def entry_for(part: str, config: str, f: float, reference: dict) -> dict:
    run_dir = RUNS / part / f"{config}_f{f:g}"
    entry = {"part": part, "config": config, "f_Hz": f}
    env_file = run_dir / "env.json"
    if env_file.exists():
        entry["env"] = json.loads(env_file.read_text())
    if not (run_dir / "results.json").exists():
        entry["refused"] = True
        err = run_dir / "error.txt"
        entry["error"] = err.read_text().strip() if err.exists() else "missing"
        rejected = run_dir / "fullwave_rejected.json"
        if not rejected.exists():   # with a field export directory the refused row's diagnostics go there
            rejected = run_dir / "fields" / "f_0_rejected.json"
        if rejected.exists():
            row = json.loads(rejected.read_text())["primary_row"]
            entry["rejection"] = row.get("rejection")
            entry["guard"] = guard_solves(row)
            for key in ("loaded_field_closure_relative_error", "loaded_closure_relative_error",
                        "source_reciprocity_relative_error"):
                if key in row:
                    entry[key] = row[key]
        return entry
    d = json.loads((run_dir / "results.json").read_text())
    fw = d["fullwave"]
    row = fw["sweep"][0]
    entry["refused"] = False
    entry["warnings"] = d.get("warnings", [])
    entry["relative_residual"] = (row.get("solver") or {}).get("relative_residual")
    entry["guard_setup"] = fw.get("gauss_law_guard")
    entry["guard_frequency"] = row.get("gauss_law_guard")
    entry["guard"] = guard_solves(row)
    entry["mesh_sha256"] = sha256(run_dir / "mesh3d.msh") if (run_dir / "mesh3d.msh").exists() else None
    for key in ("loaded_field_closure_relative_error", "loaded_closure_relative_error",
                "source_reciprocity_relative_error"):
        if key in row:
            entry[key] = row[key]
    v = row_values(row)
    entry["U_e_J_per_A2"] = [float(x) for x in v["U_e"]] if v["U_e"] is not None else None
    if row.get("R_bulk_ohm") is not None:
        entry["R_bulk_ohm"] = row["R_bulk_ohm"]
    if f in reference and not config.startswith("tree_cotree"):
        r = reference[f]
        entry["Z_vs_gauged"] = rel(v["Z"], r["values"]["Z"])
        # entry by entry (a charged port's Z can be small next to the others), with entries below 1e-9 of the largest
        # (near-null couplings) compared against that floor
        zr = r["values"]["Z"]
        floor = 1e-9 * np.abs(zr).max()
        entry["Z_entrywise_vs_gauged"] = float(np.max(np.abs(v["Z"] - zr) / np.maximum(np.abs(zr), floor)))
        entry["Z_diagonal_vs_gauged"] = [float(abs(v["Z"][k, k] - zr[k, k]) / abs(zr[k, k])) for k in range(len(zr))]
        entry["Psi_vs_gauged"] = rel(v["Psi"], r["values"]["Psi"])
        for key in ("U_e", "U_m"):
            if v[key] is not None and r["values"][key] is not None:
                entry[f"{key}_vs_gauged"] = [float(x / y - 1) for x, y in zip(v[key], r["values"][key])]
        entry["same_mesh_as_gauged"] = entry["mesh_sha256"] == r["mesh_sha256"]
        errors = []
        for port in range(len(v["Z"])):
            mine = port_field(run_dir, port)
            theirs = port_field(r["dir"], port)
            if mine is None or theirs is None or mine.shape != theirs.shape or not entry["same_mesh_as_gauged"]:
                errors.append(None)
                continue
            errors.append(float(np.linalg.norm(mine - theirs) / np.linalg.norm(theirs)))
        entry["field_error_vs_gauged"] = errors
    return entry


def report() -> None:
    rec = {"check": "gauss_law_guard", "provenance": provenance(),
           "parts": {p: {"model": s.get("model", "stage1"), "frequencies_Hz": s["frequencies"],
                         "validation_loads": s.get("loads", True), "order": s.get("order", 1), "configs": s["configs"]}
                     for p, s in PARTS.items()},
           "runs": []}
    for part, spec in PARTS.items():
        reference = {}
        for f in spec["frequencies"]:
            ref_dir = RUNS / part / f"tree_cotree_f{f:g}"
            res = ref_dir / "results.json"
            if res.exists():
                row = json.loads(res.read_text())["fullwave"]["sweep"][0]
                reference[f] = {"values": row_values(row), "dir": ref_dir,
                                "mesh_sha256": sha256(ref_dir / "mesh3d.msh")}
        for config in spec["configs"]:
            for f in spec["frequencies"]:
                entry = entry_for(part, config, f, reference)
                rec["runs"].append(entry)
                if entry["refused"]:
                    summary = "refused: " + entry.get("error", "")[:110]
                else:
                    fractions = [s.get("field_fraction") for s in entry.get("guard", {}).values()]
                    summary = (f"accepted; guard estimate {fractions}; field error {entry.get('field_error_vs_gauged')}; "
                               f"Z {entry.get('Z_vs_gauged', float('nan')):.1e}, U_e {entry.get('U_e_vs_gauged')}")
                print(f"{part} {config} {f:g} Hz: {summary}", flush=True)
        if part in ("C", "F"):   # U_e / f^2 (and R_bulk / f^3) against the corrected value at the highest frequency
            rows = [e for e in rec["runs"] if e["part"] == part and not e["refused"]]
            tops = [e for e in rows if e["config"] == "ams_correct"]
            top = max(tops, key=lambda e: e["f_Hz"]) if tops else None
            for e in rows:
                if top and e["U_e_J_per_A2"] and top["U_e_J_per_A2"]:
                    e["U_e_over_f2_vs_top"] = [float((x / e["f_Hz"] ** 2) / (y / top["f_Hz"] ** 2) - 1)
                                               for x, y in zip(e["U_e_J_per_A2"], top["U_e_J_per_A2"])]
                    print(f"{part} {e['config']} {e['f_Hz']:g} Hz: U_e/f^2 vs {top['f_Hz']:g} Hz {e['U_e_over_f2_vs_top']}",
                          flush=True)
                if top and e.get("R_bulk_ohm") and top.get("R_bulk_ohm"):
                    e["R_bulk_over_f3_vs_top"] = float((e["R_bulk_ohm"] / e["f_Hz"] ** 3)
                                                       / (top["R_bulk_ohm"] / top["f_Hz"] ** 3) - 1)
                    print(f"{part} {e['config']} {e['f_Hz']:g} Hz: R_bulk/f^3 vs {top['f_Hz']:g} Hz "
                          f"{e['R_bulk_over_f3_vs_top']:.3e}", flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    # the committed record only from the default run directory (a trial with AXF_GUARD_RUNS writes its own copy only)
    if RUNS == (ROOT / "runs" / "gauss_law_guard").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_GUARD_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
        parts = a[a.index("--parts") + 1].split(",") if "--parts" in a else list(PARTS)
        run(ranks, parts)
    report()


if __name__ == "__main__":
    main()
