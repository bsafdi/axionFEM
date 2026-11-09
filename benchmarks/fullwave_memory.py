"""Memory across a long 3D full-wave sweep: the resident set per frequency on every solver path.

    python benchmarks/fullwave_memory.py run --case mid|run5 --path PATH [--variant NAME] [--ranks N] [--tag T]
                                             [--nfreq N] [--descending]
    python benchmarks/fullwave_memory.py compare RUN_A RUN_B
    python benchmarks/fullwave_memory.py report

Tree-cotree run 5 (benchmarks/tree_cotree_scale.py) saw the gauged path's rank-0 peak memory grow by about 1 GB per
frequency over three frequencies. This driver runs one sweep of 20 frequencies (1-100 kHz, log-spaced) on run 5's model
(the fiducial toroid of Benabou et al. 2022 with a 2-degree slit port, lowest order, closed by the PEC sphere) and reads
each frequency's `memory` record (cpp/src/model3d_fullwave.cpp): VmRSS at the start of the frequency (what the setup and
the earlier frequencies left) and at its end, and VmHWM (the peak so far), each as the maximum over the ranks, their sum
and rank zero's. Cases: `mid` (a coarser mesh, minutes per sweep) and `run5` (run 5's mesh, 871k free unknowns). Every
run meshes its own design (a mesh import would need fullwave.rf_only, without the static solve and Psi); axf-mesher's
meshes are byte-identical run to run, and the record keeps each run's mesh hash.

Paths (each sets every variable it relies on):
- ams: AMS-GMRES with the Gauss-law guard (the default path);
- strumpack: the real factor gathered to rank zero (AXF_FW_STRUMPACK_DIST=0), with the guard;
- strumpack_dist: the factor distributed over the ranks (AXF_FW_STRUMPACK_DIST=1; a build against an MPI STRUMPACK);
- tree_cotree: the gauged path (factor on rank zero).
Variants switch one per-frequency feature or the allocator (VARIANTS). Growth per frequency: the least-squares slope of
the start-of-frequency resident set over frequencies 2..N (the first one starts before any factor exists), and of the
end-of-frequency peak. `compare` checks that two runs agree in every output except timings and memory (bit for bit).
Writes benchmarks/records/fullwave_memory.json from the runs under runs/fullwave_memory (AXF_FW_MEMORY_RUNS).
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

RUNS = Path(os.environ.get("AXF_FW_MEMORY_RUNS", ROOT / "runs" / "fullwave_memory")).resolve()
OUT = ROOT / "benchmarks" / "records" / "fullwave_memory.json"
FREQUENCIES = [float(f) for f in np.logspace(3, 5, 20)]
CASES = {
    "mid": {"order": 1, "size": 0.2, "surface_size": 0.04},
    "run5": {"order": 1, "size": 0.13, "surface_size": 0.02},   # benchmarks/tree_cotree_scale.py MESH
}
PATHS = {
    "ams": {"AXF_FW_SOLVER": "ams", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
            "AXF_FW_GAUSS_GUARD": "correct"},
    "strumpack": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
                  "AXF_FW_STRUMPACK_DIST": "0", "AXF_FW_GAUSS_GUARD": "correct"},
    "strumpack_dist": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
                       "AXF_FW_STRUMPACK_DIST": "1", "AXF_FW_GAUSS_GUARD": "correct"},
    "tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2",
                    "AXF_FW_STRUMPACK_DIST": "0", "AXF_FW_GAUSS_GUARD": "off"},
}
# every variable a variant may set, so that one run's choice does not leak into the next (os.environ carries over)
VARIANT_KEYS = ("AXF_FW_ERROR_INDICATOR", "AXF_FW_MALLOC_TRIM", "AXF_FW_MALLOC_THRESHOLDS", "AXF_FW_FREQUENCY_PEAK",
                "MALLOC_ARENA_MAX", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "LD_PRELOAD", "HEAPPROFILE")
VARIANTS = {
    "base": {},
    "indicator_off": {"AXF_FW_ERROR_INDICATOR": "off"},
    "guard_off": {"AXF_FW_GAUSS_GUARD": "off"},
    "export": {},   # fullwave.field_export_directory (design key)
    # AXF_FW_MALLOC_TRIM=1 trims the heap at every frequency (a diagnostic); the default is factor, and the
    # per-frequency diagnostic is frequency
    "trim": {"AXF_FW_MALLOC_TRIM": "1"},
    "peak": {"AXF_FW_FREQUENCY_PEAK": "1"},   # each row's peak is that frequency's own (not the running maximum)
    "trimoff_peak": {"AXF_FW_MALLOC_TRIM": "off", "AXF_FW_FREQUENCY_PEAK": "1"},   # the trim switched off
    "trimfreq_peak": {"AXF_FW_MALLOC_TRIM": "frequency", "AXF_FW_FREQUENCY_PEAK": "1"},
    # AXF_FW_MALLOC_THRESHOLDS, fixed (default) or glibc. "peak" is the default (the trim and the fixed
    # thresholds); these are the alternatives, all with per-frequency peaks
    "glibc_peak": {"AXF_FW_MALLOC_TRIM": "off", "AXF_FW_MALLOC_THRESHOLDS": "glibc", "AXF_FW_FREQUENCY_PEAK": "1"},
    "trimonly_peak": {"AXF_FW_MALLOC_TRIM": "factor", "AXF_FW_MALLOC_THRESHOLDS": "glibc", "AXF_FW_FREQUENCY_PEAK": "1"},
    "thresholdsonly_peak": {"AXF_FW_MALLOC_TRIM": "off", "AXF_FW_MALLOC_THRESHOLDS": "fixed",
                            "AXF_FW_FREQUENCY_PEAK": "1"},
    "arena1trim_peak": {"MALLOC_ARENA_MAX": "1", "AXF_FW_MALLOC_TRIM": "factor", "AXF_FW_MALLOC_THRESHOLDS": "glibc",
                        "AXF_FW_FREQUENCY_PEAK": "1"},
    "trim_peak": {"AXF_FW_MALLOC_TRIM": "1", "AXF_FW_FREQUENCY_PEAK": "1"},
    "arena1": {"MALLOC_ARENA_MAX": "1"},
    "mmap_fixed": {"MALLOC_MMAP_THRESHOLD_": str(128 * 1024), "MALLOC_TRIM_THRESHOLD_": str(128 * 1024)},
    "tcmalloc": {"LD_PRELOAD": "/usr/lib64/libtcmalloc.so.4"},   # gperftools' allocator instead of glibc's
    "heapprofile": {},   # AXF_EXE wrapped by scripts/axf_heapprofile.sh (in-use heap per rank)
    "massif": {},        # AXF_EXE wrapped by scripts/axf_massif.sh (valgrind)
    "arena1_peak": {"MALLOC_ARENA_MAX": "1", "AXF_FW_FREQUENCY_PEAK": "1"},
    "trim_arena1_peak": {"AXF_FW_MALLOC_TRIM": "1", "MALLOC_ARENA_MAX": "1", "AXF_FW_FREQUENCY_PEAK": "1"},
    "nothp_peak": {"AXF_FW_FREQUENCY_PEAK": "1"},        # AXF_EXE wrapped by scripts/axf_nothp.py (no huge pages)
    "nothp_trim_peak": {"AXF_FW_MALLOC_TRIM": "1", "AXF_FW_FREQUENCY_PEAK": "1"},   # the same, with the trim
}
WRAPPERS = {"heapprofile": "axf_heapprofile.sh", "massif": "axf_massif.sh", "nothp_peak": "axf_nothp.py",
            "nothp_trim_peak": "axf_nothp.py"}


def design(case: str, export_dir: Path | None = None, nfreq: int = len(FREQUENCIES), descending: bool = False):
    import axionfem as af
    from axionfem.convert import to_3d
    d2 = af.template("toroid", R=1.0, y=0.5, H=0.5, tw=0.02, g=0.01, ts=0.025, f=0.03, NI=2.5e7,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.01})
    d3 = to_3d(d2, pickup="slit", slit_deg=2.0, mesh=dict(CASES[case]))
    d3.doc["name"] = f"fullwave_memory_{case}"
    frequencies = [float(f) for f in np.logspace(3, 5, nfreq)]
    if descending:   # separates a drift with the number of frequencies from one with the frequency
        frequencies.reverse()
    d3.doc["fullwave"] = {"frequencies": frequencies, "boundary": "pec"}
    if export_dir is not None:
        d3.doc["fullwave"]["field_export_directory"] = str(export_dir)
    return d3


def run_name(case: str, path: str, variant: str, tag: str) -> str:
    return "_".join([case, path, variant] + ([tag] if tag else []))


def solve(case: str, path: str, variant: str, ranks: int, tag: str, nfreq: int, descending: bool = False) -> None:
    env = {**PATHS[path], **VARIANTS[variant]}
    for k in VARIANT_KEYS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    run_dir = RUNS / run_name(case, path, variant, tag)
    export_dir = run_dir / "fields" if variant == "export" else None
    exe = os.environ.get("AXF_EXE", "")
    if variant in WRAPPERS:   # the wrapper runs the real executable (under a heap profiler, or without huge pages)
        os.environ["AXF_REAL_EXE"] = exe
        os.environ["AXF_HEAP_DIR"] = str(run_dir / "heap")
        os.environ["AXF_EXE"] = str(ROOT / "scripts" / WRAPPERS[variant])
        env.update({k: os.environ[k] for k in ("AXF_REAL_EXE", "AXF_HEAP_DIR", "AXF_EXE")})
    t0 = time.time()
    try:
        design(case, export_dir, nfreq, descending).solve(run_dir, verbose=1, mpi_ranks=ranks)
        print(f"{run_dir.name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (refusals are recorded, not fatal)
        print(f"{run_dir.name} refused: {type(exc).__name__}: {exc}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    finally:
        os.environ["AXF_EXE"] = exe
    run_dir.mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name,
                "ranks": ranks, "threads": os.environ.get("OMP_NUM_THREADS", ""), "seconds": time.time() - t0}
    (run_dir / "env.json").write_text(json.dumps({"case": case, "path": path, "variant": variant, "tag": tag,
                                                  "env": env, "produced_by": produced}) + "\n")


def slope(y: list[float]) -> float | None:
    """Least-squares growth per frequency of y over its index (bytes per frequency)."""
    if len(y) < 3:
        return None
    return float(np.polyfit(np.arange(len(y)), np.asarray(y, dtype=float), 1)[0])


SERIES = ("rss_rank0", "rss_sum", "heap_held_rank0", "heap_held_sum", "heap_free_rank0", "heap_free_sum",
          "rss_anon_sum", "rss_shmem_sum")


def summary(run_dir: Path) -> dict:
    """One run: its environment and job, and per frequency the memory record's main series (rank 0 and the sum over
    the ranks) with the metrics of the validation note:
    - peak_growth: the largest end-of-frequency peak of frequencies 2..N minus frequency 1's (with
      AXF_FW_FREQUENCY_PEAK=1 each frequency's own peak; otherwise the running maximum);
    - start_2, start_N: the resident set at the start of frequency 2 and N (after one factor was built and freed);
    - held_2, held_N: the heap the program holds there (glibc), and its least-squares slope over 2..N."""
    entry = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    res = run_dir / "results.json"
    if not res.exists():
        err = run_dir / "error.txt"
        entry["error"] = err.read_text().strip() if err.exists() else "missing"
        return entry
    d = json.loads(res.read_text())
    fw = d["fullwave"]
    entry["dofs"] = d["mesh"].get("dofs")
    entry["elements"] = d["mesh"].get("elements")
    entry["solver_path"] = fw.get("solver_path")
    entry["fullwave_seconds"] = fw.get("seconds")
    entry["memory_after_sweep"] = (fw.get("memory") or {}).get("after_sweep", fw.get("memory_after_sweep"))
    entry["malloc_trim"] = (fw.get("memory") or {}).get("malloc_trim")
    entry["mesh_sha256"] = sha256(run_dir / "mesh3d.msh") if (run_dir / "mesh3d.msh").exists() else None
    rows = [r for r in fw["sweep"] if "memory" in r]
    if not rows:
        entry["error"] = "no per-frequency memory record (a build before the record)"
        return entry
    entry["peak_scope"] = rows[0]["memory"].get("peak_scope", "the run so far")
    series = {}
    for where in ("frequency_start", "frequency_start_after_malloc_trim", "frequency_end"):
        if where not in rows[0]["memory"]:
            continue
        keys = SERIES + (("peak_rss_rank0", "peak_rss_sum") if where == "frequency_end" else ())
        for key in keys:
            if f"{key}_bytes" in rows[0]["memory"][where]:
                series[f"{where}/{key}"] = [int(r["memory"][where][f"{key}_bytes"]) for r in rows]
    entry["series_bytes"] = series
    entry["f_Hz"] = [r["f_Hz"] for r in rows]
    fact = [r.get("direct_factorization") or {} for r in rows]
    nnz = [f.get("factor_nonzeros") or (f.get("factor") or {}).get("factor_nonzeros") for f in fact]
    if any(nnz):
        entry["factor_nonzeros"] = nnz[0]
    seconds = [r.get("frequency_total_seconds") or 0.0 for r in rows]
    metrics = {"seconds_per_frequency": sum(seconds) / len(seconds)}
    for who in ("rank0", "sum"):
        peak = series[f"frequency_end/peak_rss_{who}"]
        metrics[f"peak_1_{who}"] = peak[0]
        metrics[f"peak_growth_{who}"] = max(peak[1:]) - peak[0]
        start = series[f"frequency_start/rss_{who}"]
        metrics[f"start_2_{who}"] = start[1]
        metrics[f"start_N_{who}"] = start[-1]
        trimmed = series.get(f"frequency_start_after_malloc_trim/rss_{who}")
        if trimmed:
            metrics[f"trimmed_start_2_{who}"] = trimmed[1]
            metrics[f"trimmed_start_N_{who}"] = trimmed[-1]
        held = series.get(f"frequency_start/heap_held_{who}")
        if held and held[0] > 0:
            metrics[f"held_2_{who}"] = held[1]
            metrics[f"held_N_{who}"] = held[-1]
            metrics[f"held_slope_{who}_bytes_per_frequency"] = slope(held[1:])
    entry["metrics_bytes"] = metrics
    return entry


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}/{i}")
    else:
        yield path, x


VOLATILE = ("seconds", "rss", "memory", "/timing/")


def kept(path: str, value) -> bool:
    """Not a timing, a memory figure or a file path (the run directories differ)."""
    if any(s in path for s in VOLATILE) or path.endswith("_s"):   # *_s: timings of the static solves
        return False
    return not (isinstance(value, str) and (value.startswith("/") or str(RUNS) in value))


def compare(a: Path, b: Path) -> dict:
    """Every leaf of the two results.json except timings, memory and paths: identical (bit for bit)."""
    da = {p: v for p, v in leaves(json.loads((a / "results.json").read_text())) if kept(p, v)}
    db = {p: v for p, v in leaves(json.loads((b / "results.json").read_text())) if kept(p, v)}
    only_a = sorted(set(da) - set(db))
    only_b = sorted(set(db) - set(da))
    different = [p for p in da if p in db and da[p] != db[p]]
    return {"a": str(a), "b": str(b), "compared_leaves": len(set(da) & set(db)), "different": len(different),
            "first_differences": [(p, da[p], db[p]) for p in different[:10]], "only_a": only_a[:10],
            "only_b": only_b[:10]}


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
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "driver": str(Path(__file__).relative_to(ROOT)), "date": time.strftime("%Y-%m-%d")}


# bit identity of every output but timings and memory: a build without the allocator settings and the memory
# record (b1) against builds with them (b5, and b8 at the same 8 x 8 layout); at run 5's size, a build without the
# allocator settings (b4) against builds with them (b6, b8)
BIT_IDENTITY_PAIRS = [(f"mid_{p}_base_b1", f"mid_{p}_peak_b5") for p in PATHS] + [
    (f"mid_{p}_base_b1", f"mid_{p}_peak_b8_8x8") for p in PATHS] + [
    (f"run5_{p}_peak_b4", f"run5_{p}_peak_{b}") for p in ("strumpack", "tree_cotree") for b in ("b6", "b8_8x8")]


def report() -> None:
    runs = {}
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "env.json").exists()):
        runs[run_dir.name] = summary(run_dir)
    identity = {}
    for a, b in BIT_IDENTITY_PAIRS:
        if (RUNS / a / "results.json").exists() and (RUNS / b / "results.json").exists():
            identity[f"{a} vs {b}"] = compare(RUNS / a, RUNS / b)
    rec = {"check": "fullwave_memory", "provenance": provenance(), "cases": CASES, "paths": PATHS,
           "variants": VARIANTS, "bit_identity": identity, "runs": runs}
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    if RUNS == (ROOT / "runs" / "fullwave_memory").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    gb = 1e9
    print(f"{'run':40s} {'s/f':>5s} | peak1 growth (rank0, sum) | start 2 -> N (rank0, sum) | held 2 -> N (sum)  [GB]")
    for name, e in runs.items():
        if "metrics_bytes" not in e:
            print(f"{name:40s} --", e.get("error", "")[:150])
            continue
        m = e["metrics_bytes"]
        held = f"{m['held_2_sum'] / gb:6.3f} -> {m['held_N_sum'] / gb:6.3f}" if "held_2_sum" in m else "      -"
        print(f"{name:40s} {m['seconds_per_frequency']:5.1f} | {m['peak_1_rank0'] / gb:5.2f} {m['peak_growth_rank0'] / gb:+5.2f} "
              f"{m['peak_1_sum'] / gb:6.2f} {m['peak_growth_sum'] / gb:+5.2f} | {m['start_2_rank0'] / gb:5.2f} -> "
              f"{m['start_N_rank0'] / gb:5.2f}  {m['start_2_sum'] / gb:6.2f} -> {m['start_N_sum'] / gb:6.2f} | {held}")
    for pair, c in identity.items():
        print(f"{pair}: {c['compared_leaves']} leaves compared, {c['different']} different, only in one: "
              f"{len(c['only_a']) + len(c['only_b'])}")
    print("wrote", RUNS / "record.json", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "compare", "report"):
        sys.exit(__doc__.split("\n\n")[1])

    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default

    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        ranks = int(opt("--ranks", os.environ.get("SLURM_NTASKS", "8")))
        for path in opt("--path", "ams").split(","):
            for variant in opt("--variant", "base").split(","):
                solve(opt("--case", "mid"), path, variant, ranks, opt("--tag", ""), int(opt("--nfreq", len(FREQUENCIES))),
                      "--descending" in a)
    elif a[0] == "compare":
        print(json.dumps(compare(Path(a[1]), Path(a[2])), indent=1, default=str))
    else:
        report()


if __name__ == "__main__":
    main()
