"""Tree-cotree validation, run 4: a topology set on the gauged path, against plain STRUMPACK and the f^2 law.

    python benchmarks/tree_cotree_topology.py run [--ranks N]
    python benchmarks/tree_cotree_topology.py report

Run 4: two floating conductors, a conductor touching the
wall, a genus-1 closed loop and an internal PEC sheet. Z, Psi, U_e and U_m must agree with plain STRUMPACK to
<= 1e-10 at 100 kHz-10 MHz, and the U_e/f^2 test must hold down to 10 Hz.

One design: the two-port frame (benchmarks/designs/stage1_twoport.json: two gap ports, uniform source, source check,
shunt-loaded validation, PEC sphere of 0.7 m) plus, all copper (perfect at RF) and clear of the frame:
  float_a, float_b  two floating spheres (radius 0.08 m) at (0, 0.35, +-0.35);
  rod               a cylinder (radius 0.05 m) along -y from y = -0.45 m, cut by the 0.7 m ball so that its end lies
                    on the wall: a conductor touching the grounded wall;
  ring              a floating torus about z at z = -0.5 m (major radius 0.12 m, minor 0.02 m): genus 1;
  sheet             a plate 2 mm thick at x = 0.35-0.5 m: the internal sheet (the 3D mesher has no zero-thickness
                    conductor, so a thin plate stands in for it).
Gauged path (AXF_FW_GAUGE=tree-cotree, default charge projection 2) at 10 Hz-10 MHz, plain STRUMPACK at 100 kHz,
1 MHz and 10 MHz; one design per frequency, so that a refused frequency does not abort the others (refused rows are
kept). The U_e test is run 1's (benchmarks/tree_cotree_validation.py, energy_protocol). Writes
benchmarks/records/tree_cotree_topology.json.
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
sys.path.insert(0, str(HERE))

RUNS = Path(os.environ.get("AXF_TC_TOPO_RUNS", ROOT / "runs" / "tree_cotree_topology")).resolve()
OUT = ROOT / "benchmarks" / "records" / "tree_cotree_topology.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
FREQUENCIES = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
HIGH = [1e5, 1e6, 1e7]
CRITERION_HIGH = 1e-10
CONFIGS = {
    "tree_cotree": ({"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}, FREQUENCIES),
    "strumpack": ({"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_STRUMPACK_DIST": "0"}, HIGH),
}


FEATURES = {   # region name -> (solid, surface size)
    "float_a": ({"sphere": {"center": [0, 0.35, 0.35], "radius": 0.08}}, 0.04),
    "float_b": ({"sphere": {"center": [0, 0.35, -0.35], "radius": 0.08}}, 0.04),
    "rod": ({"intersection": [{"cylinder": {"base": [0, -0.45, 0], "axis": [0, -0.35, 0], "radius": 0.05}},
                              {"sphere": {"center": [0, 0, 0], "radius": 0.7}}]}, 0.04),
    "ring": ({"torus": {"center": [0, 0, -0.5], "axis": [0, 0, 1], "major": 0.12, "minor": 0.02}}, 0.02),
    "sheet": ({"box": {"min": [0.35, -0.1, -0.001], "max": [0.5, 0.1, 0.001]}}, 0.03),
}
# diagnostic subsets: each feature alone (the floating pair together), at the first two HIGH frequencies
SUBSETS = {"floating": ["float_a", "float_b"], "rod": ["rod"], "ring": ["ring"], "sheet": ["sheet"]}


def design(frequency: float, features=tuple(FEATURES)) -> dict:
    doc = json.loads(STAGE1.read_text())
    doc["name"] = f"tree_cotree_topology_{frequency:g}Hz"
    for name in features:
        solid, surface = FEATURES[name]
        doc["regions"].append({"name": name, "material": "copper", "solid": solid, "surface_size": surface})
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
    except Exception as exc:  # noqa: BLE001  (the report records what is missing)
        print(f"{name} failed: {type(exc).__name__}: {exc}", flush=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name}
    (RUNS / name / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def run_name(config: str, f: float) -> str:
    return f"{config}_f{f:g}"


def run(ranks: int) -> None:
    for config, (env, freqs) in CONFIGS.items():
        for f in freqs:
            if (RUNS / run_name(config, f) / "results.json").exists():
                print(f"{run_name(config, f)}: kept", flush=True)
                continue
            solve(run_name(config, f), design(f), env, ranks)


DIAGNOSE = {   # name -> (features, environment); the gauged path's default projection unless set
    **{f"{subset}_{config}": (features, CONFIGS[config][0]) for subset, features in SUBSETS.items()
       for config in CONFIGS},
    "all_tree_cotree_projection0": (tuple(FEATURES), {**CONFIGS["tree_cotree"][0], "AXF_FW_TC_CHARGE_PROJECTION": "0"}),
}


def diagnose_env(env: dict) -> dict:
    """The projection set explicitly (os.environ carries over between runs): the default 2 on the gauged path unless
    the variant sets it, 0 (the only value allowed) without the gauge."""
    if env.get("AXF_FW_GAUGE") == "tree-cotree":
        return {"AXF_FW_TC_CHARGE_PROJECTION": "2", **env}
    return {**env, "AXF_FW_TC_CHARGE_PROJECTION": "0"}


def run_diagnose(ranks: int) -> None:
    """Which feature makes Psi and the source reciprocity worse at 100 kHz-1 MHz (each alone, both paths; and the
    full set without the charge projection)."""
    for name, (features, env) in DIAGNOSE.items():
        for f in HIGH[:2]:
            run_dir = f"diag_{name}_f{f:g}"
            if (RUNS / run_dir / "results.json").exists():
                continue
            solve(run_dir, design(f, features), diagnose_env(env), ranks)


def diagnosis() -> list[dict]:
    """The diagnostic runs, if present: per subset and frequency, Psi and Z of the gauged path against plain STRUMPACK
    and both source reciprocities; and the full set without the projection against both full runs."""
    out = []
    for f in HIGH[:2]:
        for subset in SUBSETS:
            g = load_row(f"diag_{subset}_tree_cotree_f{f:g}")[0]
            p = load_row(f"diag_{subset}_strumpack_f{f:g}")[0]
            if g is None or p is None:
                continue
            out.append({"f_Hz": f, "features": SUBSETS[subset], "Psi": max_rel(cvec(g["Psi_vector"]), cvec(p["Psi_vector"])),
                        "Z": max_rel(zmat(g), zmat(p)), "source_reciprocity_gauged": g["source_reciprocity_relative_error"],
                        "source_reciprocity_plain": p["source_reciprocity_relative_error"]})
        g0 = load_row(f"diag_all_tree_cotree_projection0_f{f:g}")[0]
        full = load_row(run_name("tree_cotree", f))[0]
        plain = load_row(run_name("strumpack", f))[0]
        if g0 is not None and full is not None and plain is not None:
            out.append({"f_Hz": f, "features": list(FEATURES), "projection": 0,
                        "Psi_vs_plain": max_rel(cvec(g0["Psi_vector"]), cvec(plain["Psi_vector"])),
                        "Psi_vs_projection2": max_rel(cvec(g0["Psi_vector"]), cvec(full["Psi_vector"])),
                        "source_reciprocity_gauged": g0["source_reciprocity_relative_error"]})
    return out


def report_diagnose() -> None:
    out = {}
    for f in HIGH[:2]:
        for name in DIAGNOSE:
            row, _, refused = load_row(f"diag_{name}_f{f:g}")
            out[f"{name}_f{f:g}"] = row if row is not None else refused
    for f in HIGH[:2]:
        for subset in list(SUBSETS) + ["all"]:
            g = out.get(f"{subset}_tree_cotree_f{f:g}") if subset != "all" else out.get(f"all_tree_cotree_projection0_f{f:g}")
            p = out.get(f"{subset}_strumpack_f{f:g}") if subset != "all" else None
            if subset == "all":
                full = load_row(run_name("tree_cotree", f))[0]
                plain = load_row(run_name("strumpack", f))[0]
                if g is None or full is None or plain is None:
                    continue
                print(f"{f:g} Hz all features, projection 0: dPsi vs plain {max_rel(cvec(g['Psi_vector']), cvec(plain['Psi_vector'])):.1e}, "
                      f"vs projection 2 {max_rel(cvec(g['Psi_vector']), cvec(full['Psi_vector'])):.1e}; source reciprocity "
                      f"{g['source_reciprocity_relative_error']:.1e}", flush=True)
                continue
            if g is None or p is None:
                print(f"{f:g} Hz {subset}: missing", flush=True)
                continue
            print(f"{f:g} Hz {subset}: dPsi {max_rel(cvec(g['Psi_vector']), cvec(p['Psi_vector'])):.1e}, dZ "
                  f"{max_rel(zmat(g), zmat(p)):.1e}; source reciprocity gauged {g['source_reciprocity_relative_error']:.1e}, "
                  f"plain {p['source_reciprocity_relative_error']:.1e}", flush=True)


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


def load_row(name: str) -> tuple[dict | None, dict | None, dict | None]:
    """(sweep row, results, refused row) of one per-frequency run."""
    d = RUNS / name
    if (d / "results.json").exists():
        res = json.loads((d / "results.json").read_text())
        return res["fullwave"]["sweep"][0], res, None
    if (d / "fullwave_rejected.json").exists():
        return None, None, json.loads((d / "fullwave_rejected.json").read_text())["primary_row"]
    return None, None, None


def cvec(x: dict) -> np.ndarray:
    return np.array(x["re"], dtype=float) + 1j * np.array(x["im"], dtype=float)


def zmat(row: dict) -> np.ndarray:
    return np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)


def max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def report() -> None:
    import tree_cotree_validation as V
    rec = {"check": "tree_cotree_topology", "provenance": provenance(), "frequencies_Hz": FREQUENCIES,
           "high_frequencies_Hz": HIGH, "criterion_high": CRITERION_HIGH, "configs": {}, "comparison": [],
           "topology": None}
    for config, (env, freqs) in CONFIGS.items():
        cfg = {"env": env, "sweep": [], "refused": {}, "missing": []}
        for f in freqs:
            row, res, refused = load_row(run_name(config, f))
            if row is not None:
                row = dict(row)
                row["mesh_sha256"] = sha256(RUNS / run_name(config, f) / "mesh3d.msh")
                row["dofs"] = res["mesh"]["dofs"]
                cfg["sweep"].append(row)
                if config == "tree_cotree" and rec["topology"] is None:
                    rec["topology"] = res["fullwave"].get("tree_cotree")
                    rec["solver_path"] = res["fullwave"].get("solver_path")
            elif refused is not None:
                refused.pop("field_basis_files", None)
                cfg["refused"][f"{f:g}"] = refused
            else:
                cfg["missing"].append(f)
        rec["configs"][config] = cfg
    gauged = {r["f_Hz"]: r for r in rec["configs"]["tree_cotree"]["sweep"]}
    for f in HIGH:
        g = gauged.get(f)
        rows = [r for r in rec["configs"]["strumpack"]["sweep"] if r["f_Hz"] == f]
        refused = rec["configs"]["strumpack"]["refused"].get(f"{f:g}")
        p = rows[0] if rows else refused
        if g is None or p is None:
            rec["comparison"].append({"f_Hz": f, "missing": True})
            continue
        entry = {"f_Hz": f, "plain_refused": None if rows else p.get("rejection"),
                 "Z": max_rel(zmat(p), zmat(g)), "Psi": max_rel(cvec(p["Psi_vector"]), cvec(g["Psi_vector"]))}
        for key in ("energy_electric_J_per_A2", "energy_magnetic_J_per_A2"):
            if p.get(key) and g.get(key):
                entry[key] = max_rel(np.diag(np.array(p[key]["re"])), np.diag(np.array(g[key]["re"])))
        entry["passes"] = entry["plain_refused"] is None and max(v for k, v in entry.items()
                                                                    if k in ("Z", "Psi", "energy_electric_J_per_A2",
                                                                             "energy_magnetic_J_per_A2")) <= CRITERION_HIGH
        rec["comparison"].append(entry)
        print(f"{f:g} Hz vs plain STRUMPACK: " + ", ".join(f"{k} {v:.1e}" for k, v in entry.items()
                                                          if isinstance(v, float) and k != "f_Hz")
              + f"; passes {entry['passes']}" + (f" (plain refused: {entry['plain_refused']})" if entry["plain_refused"] else ""),
              flush=True)
    proto = V.energy_protocol({"configs": {"tree_cotree": rec["configs"]["tree_cotree"]}})["tree_cotree"]
    rec["energy_protocol"] = proto
    for t in proto.get("tests", []):
        print(f"U_e test {t['f_Hz']:g} Hz: d {t['U_e_deviation'][0]:+.2e} / {t['U_e_deviation'][1]:+.2e}, "
              f"U_e22/U_e11 {t['U_e22_over_U_e11_deviation']:+.2e}, passes {t['passes']}", flush=True)
    charges = []
    for r in rec["configs"]["tree_cotree"]["sweep"]:
        for name, tc in V.gauged_solves(r).items():
            charges.append({"f_Hz": r["f_Hz"], "solve": name, "charged_free_vertex_rows": tc.get("charged_free_vertex_rows"),
                            "charged_conductor_rows": tc.get("charged_conductor_rows"),
                            "rhs_free_vertex_charge_relative": tc.get("rhs_free_vertex_charge_relative"),
                            "rhs_charge_field_fraction_solved": tc.get("rhs_charge_field_fraction_solved")})
    rec["gauged_charges"] = charges
    backward = []
    for r in rec["configs"]["tree_cotree"]["sweep"]:
        for name, tc in V.gauged_solves(r).items():
            backward.append({"f_Hz": r["f_Hz"], "solve": name, **(tc.get("componentwise_backward_error") or {}),
                             "gauss_residual_field_fraction": tc.get("gauss_residual_field_fraction")})
    rec["gauged_backward_errors"] = backward
    rec["diagnosis"] = diagnosis()
    print("topology:", json.dumps(rec["topology"]), flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    # the committed record only from the default run directory (a trial with AXF_TC_TOPO_RUNS writes its own copy)
    if RUNS == (ROOT / "runs" / "tree_cotree_topology").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_TC_TOPO_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report", "diagnose"):
        sys.exit(__doc__.split("\n\n")[1])
    ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
    if a[0] == "diagnose":
        RUNS.mkdir(parents=True, exist_ok=True)
        run_diagnose(ranks)
        report_diagnose()
        return
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        run(ranks)
    report()


if __name__ == "__main__":
    main()
