"""Tree-cotree validation, run 5: the gauged path at scale.

    python benchmarks/tree_cotree_scale.py run [--ranks N] [--configs tree_cotree,ams]
    python benchmarks/tree_cotree_scale.py report

Run 5: a closed-PEC ND1 model with >= 5e5 free
unknowns (an ABRA toroid model) at 1-100 kHz: memory, time, 32-bit headroom, curl_representative_norm_fraction, and
agreement with AMS at 100 kHz.

Model: the fiducial toroid of Benabou et al. 2022 (the `toroid` template, as benchmarks/fullwave3d_validation.py), in
3D with a real 2-degree slit as the readout port (axionfem.convert.to_3d), lowest order, closed by the PEC sphere.
The whole toroid, not a quarter: the scale is what is tested, and PEC symmetry planes would only restrict the
excitation. One sweep per configuration at 1, 10 and 100 kHz, so the gauged setup (gather, contraction, tree) is paid
once and the factorization once per frequency:
- tree_cotree: the gauged STRUMPACK path (the reference at low frequency);
- ams: AMS-GMRES with the Gauss-law guard on (the default).
Recorded per frequency: unknowns, free edges, the gathered and factored nonzeros against 2^31 - 1, peak memory (max and
sum over ranks), setup / factorization / solve times, the gauged path's curl_representative_norm_fraction and
residuals, and Z, Psi, U_e and U_m of each path against the other. Writes benchmarks/records/tree_cotree_scale.json.
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

RUNS = Path(os.environ.get("AXF_TC_SCALE_RUNS", ROOT / "runs" / "tree_cotree_scale")).resolve()
OUT = ROOT / "benchmarks" / "records" / "tree_cotree_scale.json"
FREQUENCIES = [1e3, 1e4, 1e5]
MESH = {"order": 1, "size": 0.13, "surface_size": 0.02}
INT32 = 2**31 - 1
CONFIGS = {   # each sets every variable it relies on (os.environ carries over between runs)
    "tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2",
                    "AXF_FW_STRUMPACK_DIST": "0", "AXF_FW_GAUSS_GUARD": "off"},
    "ams": {"AXF_FW_SOLVER": "ams", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
            "AXF_FW_GAUSS_GUARD": "correct"},
}


def design():
    import axionfem as af
    from axionfem.convert import to_3d
    d2 = af.template("toroid", R=1.0, y=0.5, H=0.5, tw=0.02, g=0.01, ts=0.025, f=0.03, NI=2.5e7,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.01})
    d3 = to_3d(d2, pickup="slit", slit_deg=2.0, mesh=dict(MESH))
    d3.doc["name"] = "tree_cotree_scale"
    d3.doc["fullwave"] = {"frequencies": FREQUENCIES, "boundary": "pec"}
    return d3


def solve(config: str, ranks: int) -> None:
    env = CONFIGS[config]
    for k, v in env.items():
        os.environ[k] = v
    run_dir = RUNS / config
    t0 = time.time()
    try:
        design().solve(run_dir, verbose=1, mpi_ranks=ranks)
        print(f"{config}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (refusals are recorded, not fatal)
        print(f"{config} refused: {type(exc).__name__}: {exc}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name,
                "ranks": ranks, "seconds": time.time() - t0}
    (run_dir / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


def provenance() -> dict:
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": exe, "driver": str(Path(__file__).relative_to(ROOT)), "date": time.strftime("%Y-%m-%d")}


def matrices(row: dict) -> dict:
    z = np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)
    psi = np.array(row["Psi_vector"]["re"], dtype=float) + 1j * np.array(row["Psi_vector"]["im"], dtype=float)
    out = {"Z": z, "Psi": psi}
    for key, name in (("energy_electric_J_per_A2", "U_e"), ("energy_magnetic_J_per_A2", "U_m")):
        if row.get(key):
            out[name] = np.diag(np.array(row[key]["re"], dtype=float))
    return out


def rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def summary(config: str) -> dict:
    run_dir = RUNS / config
    entry = {"config": config}
    env_file = run_dir / "env.json"
    if env_file.exists():
        entry["env"] = json.loads(env_file.read_text())
    res = run_dir / "results.json"
    if not res.exists():
        err = run_dir / "error.txt"
        entry["error"] = err.read_text().strip() if err.exists() else "missing"
        rejected = run_dir / "fullwave_rejected.json"
        if rejected.exists():
            entry["rejected_row"] = json.loads(rejected.read_text())["primary_row"]
        return entry
    d = json.loads(res.read_text())
    fw = d["fullwave"]
    entry["dofs"] = d["mesh"].get("dofs")
    entry["mesh"] = {k: v for k, v in d["mesh"].items() if isinstance(v, (int, float, str))}
    entry["mesh_sha256"] = sha256(run_dir / "mesh3d.msh") if (run_dir / "mesh3d.msh").exists() else None
    entry["solver_path"] = fw.get("solver_path")
    entry["tree_cotree_topology"] = fw.get("tree_cotree")
    entry["gauss_law_guard"] = fw.get("gauss_law_guard")
    entry["energy_operator_setup_seconds"] = fw.get("energy_operator_setup_seconds")
    entry["fullwave_seconds"] = fw.get("seconds")
    entry["warnings"] = d.get("warnings", [])
    rows = []
    for row in fw["sweep"]:
        r = {"f_Hz": row["f_Hz"], "L_H": row.get("L_H"), "Psi_abs": row.get("Psi_abs"),
             "solver_setup_seconds": row.get("solver_setup_seconds"), "port_solves_seconds": row.get("port_solves_seconds"),
             "frequency_total_seconds": row.get("frequency_total_seconds"),
             "peak_rss_max_rank_bytes": row.get("peak_rss_max_rank_bytes"),
             "peak_rss_sum_rank_bytes": row.get("peak_rss_sum_rank_bytes"),
             "solver": {k: v for k, v in (row.get("solver") or {}).items() if k != "tree_cotree"}}
        fact = row.get("direct_factorization") or {}
        if fact:
            r["factorization_seconds"] = fact.get("seconds")
            factor = fact.get("factor") or {}
            r["factor"] = {k: factor.get(k) for k in ("factor_nonzeros", "matrix_nonzeros", "backend", "inertia")}
            if factor.get("factor_nonzeros"):
                r["factor_nonzeros_over_int32"] = factor["factor_nonzeros"] / INT32
            if factor.get("matrix_nonzeros"):
                r["matrix_nonzeros_over_int32"] = factor["matrix_nonzeros"] / INT32
        tc = (row.get("solver") or {}).get("tree_cotree")
        if tc:
            r["tree_cotree"] = {k: tc.get(k) for k in (
                "curl_representative_norm_fraction", "refinement_steps", "split_true_relative_residual",
                "curl_block_relative_residual", "gauss_block_relative_residual", "componentwise_backward_error",
                "rhs_charge_field_fraction", "charged_free_vertex_rows", "charged_conductor_rows")}
        guard = (row.get("solver") or {}).get("gauss_law_guard")
        if guard:
            r["gauss_law_guard"] = {k: guard.get(k) for k in (
                "applied", "field_fraction", "gauss_residual_reduction", "pcg_relative_residual_max", "seconds")}
        r["values"] = {k: v.tolist() if not np.iscomplexobj(v) else {"re": v.real.tolist(), "im": v.imag.tolist()}
                       for k, v in matrices(row).items()}
        rows.append(r)
    entry["rows"] = rows
    return entry


def report() -> None:
    rec = {"check": "tree_cotree_scale", "provenance": provenance(), "frequencies_Hz": FREQUENCIES, "mesh": MESH,
           "configs": {c: summary(c) for c in CONFIGS}}
    topo = (rec["configs"]["tree_cotree"].get("tree_cotree_topology") or {})
    if topo:
        rec["int32_headroom"] = {"global_edges": topo.get("global_edges"), "free_edges": topo.get("free_edges"),
                                 "gathered_K_nonzeros_over_int32": (topo.get("gathered_K_nonzeros") or 0) / INT32,
                                 "gathered_M_nonzeros_over_int32": (topo.get("gathered_M_nonzeros") or 0) / INT32}
    a = {r["f_Hz"]: matrices_from(r) for r in rec["configs"]["ams"].get("rows", [])}
    g = {r["f_Hz"]: matrices_from(r) for r in rec["configs"]["tree_cotree"].get("rows", [])}
    rec["ams_vs_gauged"] = {}
    for f in FREQUENCIES:
        if f in a and f in g:
            cmp = {"Z": rel(a[f]["Z"], g[f]["Z"]), "Psi": rel(a[f]["Psi"], g[f]["Psi"])}
            for key in ("U_e", "U_m"):
                if key in a[f] and key in g[f]:
                    cmp[key] = [float(x / y - 1) for x, y in zip(a[f][key], g[f][key])]
            rec["ams_vs_gauged"][str(f)] = cmp
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    # the committed record only from the default run directory (a trial with AXF_TC_SCALE_RUNS writes its own copy)
    if RUNS == (ROOT / "runs" / "tree_cotree_scale").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    for config, entry in rec["configs"].items():
        if "rows" not in entry:
            print(config, "refused:", entry.get("error", "")[:200])
            continue
        print(config, "dofs", entry["dofs"])
        for r in entry["rows"]:
            print(f"  {r['f_Hz']:g} Hz: setup {r['solver_setup_seconds']:.0f} s, factor {r.get('factorization_seconds')}, "
                  f"solves {r['port_solves_seconds']:.0f} s, rss max {r['peak_rss_max_rank_bytes'] / 1e9:.1f} GB "
                  f"sum {r['peak_rss_sum_rank_bytes'] / 1e9:.1f} GB, tc {r.get('tree_cotree')}, guard {r.get('gauss_law_guard')}")
    print("int32:", rec.get("int32_headroom"))
    print("ams vs gauged:", json.dumps(rec["ams_vs_gauged"]))
    print("wrote", OUT, flush=True)


def matrices_from(r: dict) -> dict:
    out = {}
    for k, v in r["values"].items():
        out[k] = np.array(v["re"]) + 1j * np.array(v["im"]) if isinstance(v, dict) else np.array(v)
    return out


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "8"))
        configs = a[a.index("--configs") + 1].split(",") if "--configs" in a else list(CONFIGS)
        for config in configs:
            if (RUNS / config / "results.json").exists():
                print(config, "kept", flush=True)
                continue
            solve(config, ranks)
    report()


if __name__ == "__main__":
    main()
