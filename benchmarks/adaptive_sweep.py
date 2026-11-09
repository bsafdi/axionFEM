"""The adaptive 3D full-wave sweep (`fullwave.adaptive`, docs/physics/adaptive_sweep.md): validation.

    python benchmarks/adaptive_sweep.py run CASE BUILD [NRANKS]        solve one case with BUILD (axf, axf-mesher)
    python benchmarks/adaptive_sweep.py invariance OLD_BUILD NEW_BUILD [NRANKS]
    python benchmarks/adaptive_sweep.py gap NAME BUILD [NRANKS]         adaptive and dense on one gap-validation case
    python benchmarks/adaptive_sweep.py samples                         the fitted solves of the solved runs, as text
                                                                         for benchmarks/adaptive_sweep_trials.cpp
    python benchmarks/adaptive_sweep.py report                          analyse what is solved, write the record

Cases (runs/adaptive_sweep/CASE):
  cavity          benchmarks/designs/stage1_twoport.json (a copper frame with two gap ports, uniform source) in its
                  closed PEC sphere (R1 0.7 m), adaptive over 100-270 MHz, with the 16 lowest eigenmodes of the same
                  mesh in the same run. The discrete Z(omega) and Psi(omega) are rational with poles at the discrete
                  eigenfrequencies of the closed structure (port open), so every in-band pole of the model must sit
                  on an eigenfrequency, up to the fit and the eigensolver tolerance.
  cavity_lossy    the same with a dielectric slab (eps_r 2, tan delta 1e-3, box +-0.1 m): complex poles. Reference
                  Q = 1 / (tan delta p_e), p_e the slab's share of the lossless eigenmode's electric energy (first
                  order in tan delta: agreement to O(tan delta) is expected).
  cavity_check, cavity_lossy_check
                  fixed lists at the midpoints between the adaptive solves and at band points away from them: the
                  model against direct solves it never saw.
  core_adaptive   DMRadio-Core reduced (benchmarks/core_readout_ports.py c360, the full-circumference uniform feed) on
                  a coarse mesh (order 2, h 0.32 m, surface 0.05 m), radiation boundary, adaptive over 70-140 MHz.
  core_dense_K    the same mesh, 61 log-spaced frequencies (the 4-point sweep's points among them), in 4 parallel
                  parts K = 0..3 (each also solves f_max, so that every part has the same mesh and exterior map).
                  Reported: the model against the 61 direct solves (Z, Psi), and the scan time (core_readout_ports'
                  integrand with L_eff = (w dX/dw + X) / (2 w)) from the dense solves, the 4-point subset, the
                  adaptive solves alone and the adaptive model's dense curve.
Invariance (runs/adaptive_sweep/invariance/{old,new}/NAME, or $AXF_ADAPTIVE_INVARIANCE_DIR/{old,new}/NAME): fixed-list
designs solved by a build without the adaptive sweep and by this one; every output value is compared, timings and memory records
excepted (keys with seconds, rss, timing, date, path, directory, file or memory, or ending in _s).
  stage1          stage1_twoport.json as is (1 and 12 MHz, PEC, two ports, source_check, shunt loads)
  stage1_lossy    with the slab, 3 frequencies (bulk loss)
  stage1_radiating  radiation boundary, 3 frequencies near 50 MHz (exterior map and its dZ/dw)
  stage1_gauged   stage1 with AXF_FW_SOLVER=strumpack AXF_FW_GAUGE=tree-cotree
Gap validation (runs/adaptive_sweep/gap_NAME/{adaptive,dense}): on one small mesh
per case an adaptive sweep (tolerance 1e-3, linear, max_solves 30) and a dense fixed list of 41 frequencies; the
model against every dense solve.
  tree_cotree     the two-port frame in its PEC sphere, STRUMPACK with the tree-cotree gauge, 115-150 MHz
  pml, robin_absorbing, robin_impedance
                  the TEM line (nx 12, order 2) with a PML end, an absorbing (scattering) end, and copper impedance
                  plates with an absorbing end, 100-300 MHz
  four_ports      the two-port frame with its slot split into 4 gap ports, 115-150 MHz
  refusal         the shorted TEM line with a validation shunt and GMRES capped at 32 iterations (one rank), so that
                  the frequency next to the quarter-wave pole is refused; the dense list runs uncapped
High Q (runs/adaptive_sweep/tem_tand_T, `run tem_tand_T`): the shorted TEM line filled
with a dielectric of tan delta T = 0, 1e-11, 1e-10, 1e-9 (Q = 1 / T), adaptive over 100-400 MHz: the pole's class and
Q against 1 / T.
Writes benchmarks/records/adaptive_sweep.json (with the commit, the dirty flag, the source hashes and the builds).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
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

STAGE1 = HERE / "designs" / "stage1_twoport.json"
RUNS = ROOT / "runs" / "adaptive_sweep"
OUT = HERE / "records" / "adaptive_sweep.json"
SOURCES = ["cpp/src/model3d_fullwave.cpp", "cpp/src/rational_fit.cpp", "cpp/src/spec.cpp"]
CAVITY_BAND = (100e6, 270e6)
CORE_BAND = (70e6, 140e6)
TOLERANCE = 1e-3
SLAB = {"kind": "linear", "mu_r": 1.0, "eps_r": 2.0, "tan_delta": 1e-3}
CORE_MESH = {"order": 2, "size": 0.32, "surface_size": 0.05}
DENSE_POINTS = 61
# "memory": the per-frequency memory records, as volatile as the timings
TIMING_WORDS = ("seconds", "rss", "timing", "date", "path", "directory", "file", "memory")
# AXF_ADAPTIVE_INVARIANCE_DIR: a rerun of the invariance check elsewhere, leaving the record's inputs as they are
INVARIANCE = Path(os.environ.get("AXF_ADAPTIVE_INVARIANCE_DIR") or RUNS / "invariance")


# ------------------------------------------------------------------ designs

def stage1(lossy: bool = False) -> dict:
    doc = json.loads(STAGE1.read_text())
    if lossy:
        doc["materials"] = {"slab_dielectric": SLAB}
        doc["regions"].append({"name": "slab", "material": "slab_dielectric",
                               "solid": {"box": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}})
    return doc


def cavity(lossy: bool) -> dict:
    doc = stage1(lossy)
    fw = doc["fullwave"]
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    fw["adaptive"] = {"f_min_Hz": CAVITY_BAND[0], "f_max_Hz": CAVITY_BAND[1], "tolerance": TOLERANCE,
                      "max_solves": 60, "spacing": "linear"}
    fw["modes"] = 16
    doc["name"] = "adaptive_sweep_cavity" + ("_lossy" if lossy else "")
    return doc


def cavity_check(lossy: bool) -> dict:
    """Fixed list: the midpoints between consecutive adaptive solves, and 8 band points (largest distance to them)."""
    src = RUNS / ("cavity_lossy" if lossy else "cavity") / "results.json"
    ad = json.loads(src.read_text())["fullwave"]["adaptive"]
    solved = sorted(ad["solved_frequencies_Hz"])
    freqs = [0.5 * (a + b) for a, b in zip(solved, solved[1:])]
    grid = np.linspace(*CAVITY_BAND, 401)[1:-1]
    for _ in range(8):
        taken = np.array(solved + freqs)
        d = np.min(np.abs(grid[:, None] - taken[None, :]), axis=1)
        freqs.append(float(grid[int(np.argmax(d))]))
    doc = stage1(lossy)
    fw = doc["fullwave"]
    for key in ("source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    fw["frequencies"] = sorted(freqs)
    # the adaptive run's mesh: its sizing follows the largest frequency (f_max of the band)
    fw["frequencies"].append(CAVITY_BAND[1])
    doc["name"] = "adaptive_sweep_cavity_check" + ("_lossy" if lossy else "")
    return doc


def core(freqs: dict) -> dict:
    import axionfem  # noqa: F401  (first: core_readout_ports puts python/ ahead of a package copy on PYTHONPATH)
    import core_readout_ports as CR
    d = CR.design("c360", "fast", {"f_min": CORE_BAND[0], "f_max": CORE_BAND[1], "points": 4,
                                   "boundary": "radiation"})
    doc = copy.deepcopy(d.doc)
    doc["mesh"] = dict(CORE_MESH)
    doc["fullwave"] = {"boundary": "radiation", **freqs}
    doc["name"] = doc["name"] + "_coarse"
    return doc


def core_adaptive() -> dict:
    return core({"adaptive": {"f_min_Hz": CORE_BAND[0], "f_max_Hz": CORE_BAND[1], "tolerance": TOLERANCE,
                              "max_solves": 30, "spacing": "log"}})


def dense_frequencies() -> list[float]:
    a, b = CORE_BAND
    f = [a * (b / a) ** (i / (DENSE_POINTS - 1)) for i in range(DENSE_POINTS)]
    f[-1] = b   # exactly the band end, as the adaptive run's last initial point
    return f


def core_dense(part: int) -> dict:
    """Every DENSE_PARTS-th of the 61 frequencies, starting at `part`, in parallel jobs. Each part also solves f_max:
    the mesh sizing and the exterior map's multipole order follow the largest frequency, so every part has the same
    mesh and operator as the adaptive run."""
    freqs = dense_frequencies()[part::DENSE_PARTS]
    if freqs[-1] != CORE_BAND[1]:
        freqs.append(CORE_BAND[1])
    return core({"frequencies": freqs})


DENSE_PARTS = 4
CASES = {"cavity": lambda: cavity(False), "cavity_lossy": lambda: cavity(True),
         "cavity_check": lambda: cavity_check(False), "cavity_lossy_check": lambda: cavity_check(True),
         "core_adaptive": core_adaptive,
         "core_probe": lambda: core({"frequencies": [CORE_BAND[1]]})}   # the cost of one frequency
for _k in range(DENSE_PARTS):
    CASES[f"core_dense_{_k}"] = (lambda k: lambda: core_dense(k))(_k)


def invariance_designs() -> dict:
    out = {"stage1": stage1()}
    lossy = stage1(True)
    for key in ("source_check", "validation_shunt_R_ohm"):
        lossy["fullwave"].pop(key, None)
    lossy["fullwave"]["frequencies"] = [5e6, 12e6, 30e6]
    out["stage1_lossy"] = lossy
    rad = stage1()
    for key in ("source", "source_check", "validation_shunt_R_ohm"):
        rad["fullwave"].pop(key, None)
    rad["fullwave"]["boundary"] = "radiation"
    rad["fullwave"]["frequencies"] = [45e6, 50e6, 55e6]
    out["stage1_radiating"] = rad
    out["stage1_gauged"] = stage1()
    return out


INVARIANCE_ENV = {"stage1_gauged": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}}


# ------------------------------------------------------------------ gap validation
# Each case: an adaptive sweep and a dense fixed list on the same mesh; the model against every dense solve.
GAP_TEM_BAND = (1e8, 3e8)
GAP_STAGE1_BAND = (1.15e8, 1.5e8)
GAP_DENSE = 41


def tem(wd: Path, termination: str, fullwave: dict, impedance_plates: bool = False) -> dict:
    """The TEM line of benchmarks/fw3d_robin_validation.py (nx 12, order 2), its mesh written to wd."""
    import fw3d_robin_validation as robin
    import fullwave_indicator as fwi
    if impedance_plates:
        doc, _ = robin.build(wd, termination=termination, plates="impedance", nx=12, order=2)
    else:
        doc, _ = fwi.tem_design(wd, termination, 2, 0, [1e8], nx=12)
    doc["mesh"]["refinements"] = 0
    doc["fullwave"].pop("frequencies", None)
    doc["fullwave"].update(fullwave)
    return doc


def stage1_ports(nports: int) -> dict:
    """The two-port frame with its slot split into `nports` gap ports (y slices of 0.02 m, 0.005 m apart), all bridging the one
    cut in the frame, in the closed PEC sphere."""
    doc = stage1()
    regions = [r for r in doc["regions"] if not r["name"].startswith("gap")]
    edges = np.linspace(-0.05, 0.05, nports + 1)
    names = []
    for k in range(nports):
        lo = float(edges[k]) + (0.0025 if k > 0 else 0.0)
        hi = float(edges[k + 1]) - (0.0025 if k < nports - 1 else 0.0)
        names.append(f"gap{k}")
        regions.append({"name": f"gap{k}", "material": "vacuum",
                        "solid": {"box": {"min": [-0.04, lo, 0.2], "max": [0.04, hi, 0.3]}}})
    doc["regions"] = regions
    doc["pickup"]["port"] = "gap0"
    fw = doc["fullwave"]
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    fw["ports"] = [{"name": f"port{k}", "region": f"gap{k}", "toward": [1, 0, 0]} for k in range(1, nports)]
    doc["name"] = f"adaptive_sweep_stage1_{nports}_ports"
    return doc


def stage1_two_ports() -> dict:
    doc = stage1()
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        doc["fullwave"].pop(key, None)
    return doc


def gap_adaptive_settings(band: tuple[float, float]) -> dict:
    return {"adaptive": {"f_min_Hz": band[0], "f_max_Hz": band[1], "tolerance": TOLERANCE, "spacing": "linear",
                         "max_solves": 30}}


def gap_dense_settings(band: tuple[float, float]) -> dict:
    return {"frequencies": [float(f) for f in np.linspace(band[0], band[1], GAP_DENSE)]}


def gap_case(name: str, part: str, wd: Path) -> tuple[dict, dict]:
    """(design, extra environment) of one gap case; part is "adaptive" or "dense"."""
    tem_band = gap_adaptive_settings(GAP_TEM_BAND) if part == "adaptive" else gap_dense_settings(GAP_TEM_BAND)
    st_band = gap_adaptive_settings(GAP_STAGE1_BAND) if part == "adaptive" else gap_dense_settings(GAP_STAGE1_BAND)
    if name == "tree_cotree":
        doc = stage1_two_ports()
        doc["fullwave"].update(st_band)
        return doc, {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}
    if name == "pml":
        return tem(wd, "pml", tem_band), {}
    if name == "robin_absorbing":
        return tem(wd, "scattering", tem_band), {}
    if name == "robin_impedance":
        return tem(wd, "scattering", tem_band, impedance_plates=True), {}
    if name == "four_ports":
        doc = stage1_ports(4)
        doc["fullwave"].update(st_band)
        return doc, {}
    if name == "refusal":
        # GMRES capped at 32 iterations, with a validation shunt whose gate refuses a frequency next to the line's
        # quarter-wave pole; the dense reference runs uncapped (the gate
        # would stop a list). On four ranks the block preconditioner needs more iterations, and the cap refused every
        # initial point: the all-refused path, not a refusal mid-sweep.
        doc = tem(wd, "pec", tem_band)
        doc["fullwave"]["validation_shunt_R_ohm"] = [50.0]
        return doc, ({"AXF_FW_MAX_ITER": "32"} if part == "adaptive" else {})
    raise KeyError(name)


GAP_CASES = ("tree_cotree", "pml", "robin_absorbing", "robin_impedance", "four_ports", "refusal")
GAP_RANKS = {"refusal": 1}   # cases whose behaviour depends on the rank count (otherwise the job's)


def run_gap(name: str, build: Path, nranks: int) -> None:
    for part in ("adaptive", "dense"):
        wd = RUNS / f"gap_{name}" / part
        wd.mkdir(parents=True, exist_ok=True)
        doc, env = gap_case(name, part, wd)
        solve(wd, doc, build, GAP_RANKS.get(name, nranks), env)


def gap_report() -> dict:
    out = {}
    for name in GAP_CASES:
        a = RUNS / f"gap_{name}" / "adaptive" / "results.json"
        d = RUNS / f"gap_{name}" / "dense" / "results.json"
        if not (a.exists() and d.exists()):
            out[name] = {"status": "not solved"}
            continue
        ra = json.loads(a.read_text())
        rd = json.loads(d.read_text())
        ad = ra["fullwave"]["adaptive"]
        info = json.loads((a.parent / "run_info.json").read_text())
        row = {"build": info.get("build"), "ranks": info.get("ranks"),
               "env": info.get("env"),
               "ports": len(ra["fullwave"]["ports"]), "converged": ad["converged"], "stop_reason": ad["stop_reason"],
               "solves": ad["solves"], "refused": [{"f_Hz": r["f_Hz"], "reason": r["reason"][:160]} for r in ad["refused"]],
               "support_points": len(ad["model"]["support_Hz"]) if ad["model"] else 0,
               "hermite": ad["hermite"], "fit_consistent": ad["fit_consistent"],
               "estimated_error": ad["estimated_error"], "dense_solves": len(rd["fullwave"]["sweep"]),
               "self_resonances": [{"f_Hz": r["f_Hz"], "Q": r["Q"]} for r in ra["fullwave"]["self_resonances"]],
               "pole_classes": {}}
        for p in ad["poles"]:
            row["pole_classes"][p["class"]] = row["pole_classes"].get(p["class"], 0) + 1
        if ad["model"]:
            row["model_vs_dense"] = model_errors(ra, accepted(rd["fullwave"]["sweep"]), accepted(ra["fullwave"]["sweep"]))
        out[name] = row
    return out


# ------------------------------------------------------------------ high Q
# The shorted TEM line filled with a dielectric of loss tangent tan delta: one pole near 250 MHz with Q = 1 / tan delta
# (0: lossless). How a lossless or very high-Q pole is classed, and its Q against 1 / tan delta.
HIGHQ_TAN_DELTA = ("0", "1e-11", "1e-10", "1e-9")


def tem_tand(tand: str) -> dict:
    wd = RUNS / f"tem_tand_{tand}"
    wd.mkdir(parents=True, exist_ok=True)
    band = {"adaptive": {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": TOLERANCE, "spacing": "linear",
                         "max_solves": 30}}
    doc = tem(wd, "pec", band)
    if float(tand) > 0:
        doc["materials"]["fill"] = {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "tan_delta": float(tand)}
        for region in doc["regions"]:
            region["material"] = "fill"
    doc["name"] = f"adaptive_sweep_tem_tand_{tand}"
    return doc


for _t in HIGHQ_TAN_DELTA:
    CASES[f"tem_tand_{_t}"] = (lambda t: lambda: tem_tand(t))(_t)


def highq_report() -> dict:
    out = {}
    for tand in HIGHQ_TAN_DELTA:
        res = load(f"tem_tand_{tand}")
        if res is None:
            out[tand] = {"status": "not solved"}
            continue
        ad = res["fullwave"]["adaptive"]
        t = float(tand)
        out[tand] = {"converged": ad["converged"],
                     "stop_reason": ad["stop_reason"], "solves": ad["solves"],
                     "Q_expected": (1.0 / t) if t > 0 else None,
                     "poles": [{"f_Hz": q["f_Hz"], "im_Hz": q["pole_im_Hz"], "uncertainty_Hz": q.get("pole_uncertainty_Hz"),
                                "class": q["class"], "Q": q["Q"], "Q_uncertainty": q.get("Q_uncertainty"),
                                "Q_min": q.get("Q_min"),
                                "Q_over_expected": (q["Q"] * t) if (q["Q"] and t > 0) else None}
                               for q in ad["poles"]],
                     "self_resonances": len(res["fullwave"]["self_resonances"])}
    return out


# ------------------------------------------------------------------ samples for the trial harness
# One line per fitted solve: f nports, Z (row-major) and Psi as re im pairs, then dZ/df = 2 pi dZ/dw (row-major) as re
# im pairs (benchmarks/adaptive_sweep_trials.cpp refitnoise).
SAMPLE_RUNS = {"four_ports": "gap_four_ports/adaptive", "tree_cotree": "gap_tree_cotree/adaptive", "cavity": "cavity",
               "cavity_lossy": "cavity_lossy", **{f"tem_tand_{t}": f"tem_tand_{t}" for t in HIGHQ_TAN_DELTA}}


def write_samples() -> None:
    out_dir = RUNS / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, run in SAMPLE_RUNS.items():
        path = RUNS / run / "results.json"
        if not path.exists():
            continue
        fw = json.loads(path.read_text())["fullwave"]
        nports = len(fw["ports"])
        lines = []
        for row in accepted(fw["sweep"]):
            vals = [repr(float(row["f_Hz"])), str(nports)]
            for j in range(nports):
                for k in range(nports):
                    vals.append(repr(float(row["Z_matrix_re_ohm"][j][k])))
                    vals.append(repr(float(row["Z_matrix_im_ohm"][j][k])))
            for k in range(nports):
                vals.append(repr(float(row["Psi_vector"]["re"][k])))
                vals.append(repr(float(row["Psi_vector"]["im"][k])))
            for j in range(nports):
                for k in range(nports):
                    vals.append(repr(2 * math.pi * float(row["dZ_domega_H"]["re"][j][k])))
                    vals.append(repr(2 * math.pi * float(row["dZ_domega_H"]["im"][j][k])))
            lines.append(" ".join(vals))
        (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(name, len(lines), "samples")


# ------------------------------------------------------------------ solving

def _mpi_launcher(nranks: int) -> list:
    """The launcher prefix for nranks > 1, by the rule of axionfem.Design.solve: $AXF_MPI_LAUNCHER, else srun inside
    a batch job when srun is on PATH, else mpirun, else mpiexec."""
    if nranks <= 1:
        return []
    import shlex
    import shutil
    cmd = os.environ.get("AXF_MPI_LAUNCHER")
    if not cmd:
        if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
            cmd = "srun"
        else:
            cmd = "mpirun" if shutil.which("mpirun") or not shutil.which("mpiexec") else "mpiexec"
    return shlex.split(cmd) + ["-n", str(nranks)]


def solve(wd: Path, doc: dict, build: Path, nranks: int, env_extra: dict | None = None) -> dict:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "results.json").unlink(missing_ok=True)
    (wd / "design.json").write_text(json.dumps(doc, indent=1))
    env = dict(os.environ, AXF_MESHER=str(build / "axf-mesher"), **(env_extra or {}))
    launcher = _mpi_launcher(nranks)
    t0 = time.time()
    p = subprocess.run(launcher + [str(build / "axf"), "solve", str(wd / "design.json"), "-o", str(wd), "-v", "1"],
                       capture_output=True, text=True, env=env)
    (wd / "solve.log").write_text(p.stdout + "\n" + p.stderr)
    info = {"returncode": p.returncode, "seconds": time.time() - t0, "build": Path(build).name,
            "ranks": nranks, "env": env_extra or {}}
    (wd / "run_info.json").write_text(json.dumps(info, indent=1))
    print(wd.name, "returncode", p.returncode, f"{info['seconds']:.0f} s", flush=True)
    return info


# ------------------------------------------------------------------ analysis helpers

def load(name: str) -> dict | None:
    p = RUNS / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_info(name: str) -> dict:
    p = RUNS / name / "run_info.json"
    return json.loads(p.read_text()) if p.exists() else {}


def model_of(results: dict):
    import axionfem.circuit as ck
    return ck.fullwave_model(results)


def outputs(row: dict) -> np.ndarray:
    """Z (row-major) then Psi, as the adaptive fit orders them."""
    Z = np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)
    Psi = np.array(row["Psi_vector"]["re"], dtype=float) + 1j * np.array(row["Psi_vector"]["im"], dtype=float)
    return np.concatenate([Z.ravel(), Psi])


def floors(ad: dict, rows: list[dict]) -> np.ndarray:
    """The adaptive fit's error floor per output (medians over the fitted solves): floor_fraction x
    sqrt(median |Z_jj| median |Z_kk|) for Z[j][k], not below 1e-6 of the Z group's (median largest |Z|); floor_fraction
    x the median largest |Psi| for Psi. A group that is zero throughout (Psi without an axion source) is inactive,
    as in the fit: floor +inf."""
    n = len(rows[0]["Psi_vector"]["re"])
    vals = np.array([np.abs(outputs(r)) for r in rows])
    frac = ad["settings"]["floor_fraction"]
    gz = np.median(vals[:, :n * n].max(axis=1))
    gp = np.median(vals[:, n * n:].max(axis=1))
    med = np.median(vals, axis=0)
    out = []
    for j in range(n):
        for k in range(n):
            out.append(max(frac * math.sqrt(med[j * n + j] * med[k * n + k]), 1e-6 * gz))
    psi_floor = frac * gp if gp > 0 else math.inf
    return np.array(out + [psi_floor] * n)


def model_errors(results: dict, check_rows: list[dict], fitted_rows: list[dict]) -> dict:
    """The model against direct solves: the fit's error measure, Z and Psi separately, and the worst frequency."""
    ad = results["fullwave"]["adaptive"]
    model = model_of(results)
    n = len(check_rows[0]["Psi_vector"]["re"])
    fl = floors(ad, fitted_rows)
    worst = {"all": 0.0, "Z": 0.0, "Psi": 0.0}
    at = None
    for r in check_rows:
        m = model(r["f_Hz"])
        pred = np.concatenate([m["Z_matrix"].ravel(), m["Psi_vector"].ravel()])
        ref = outputs(r)
        e = np.abs(pred - ref) / np.maximum(np.abs(ref), fl)
        if e.max() > worst["all"]:
            worst["all"] = float(e.max())
            at = r["f_Hz"]
        worst["Z"] = max(worst["Z"], float(e[:n * n].max()))
        worst["Psi"] = max(worst["Psi"], float(e[n * n:].max()))
    if not np.isfinite(fl[n * n:]).any():
        worst["Psi"] = None   # no axion source: Psi is zero and not fitted
    return {"max_relative_error": worst, "at_Hz": at, "checked_frequencies": len(check_rows),
            "tolerance": ad["settings"]["tolerance"]}


def accepted(rows: list[dict]) -> list[dict]:
    return [r for r in rows if all(p["solver"]["accepted"] for p in r["port_solutions"])]


# ------------------------------------------------------------------ cavity

def cavity_report(lossy: bool) -> dict:
    name = "cavity_lossy" if lossy else "cavity"
    res = load(name)
    if res is None:
        return {"status": "not solved"}
    fw = res["fullwave"]
    ad = fw["adaptive"]
    modes = [m for m in fw.get("eigenmodes", {}).get("modes", []) if m.get("converged")]
    eig = np.array([m["f_Hz"] for m in modes])
    out = {"build": run_info(name).get("build"), "solves": ad["solves"],
           "converged": ad["converged"], "stop_reason": ad["stop_reason"], "estimated_error": ad["estimated_error"],
           "hermite": ad["hermite"], "fit_consistent": ad["fit_consistent"],
           "last_prediction_error": ad["last_prediction_error"], "support_points": len(ad["model"]["support_Hz"]),
           "refused": ad["refused"], "eigenmodes_converged": fw.get("eigenmodes", {}).get("converged"),
           "eigenmodes_in_band": [float(f) for f in eig if CAVITY_BAND[0] <= f <= CAVITY_BAND[1]]}
    out["self_resonances"] = len(fw["self_resonances"])
    poles = []
    for r in ad["poles"]:   # every in-band pole, with its class
        f = r["f_Hz"]
        k = int(np.argmin(np.abs(eig - f))) if len(eig) else None
        row = {"pole_f_Hz": f, "pole_im_Hz": r["pole_im_Hz"], "Q": r["Q"], "class": r["class"],
               "relative_strength": r["relative_strength"],
               "contribution_one_step_away": r["contribution_one_step_away"],
               "uncertainty_Hz": r.get("pole_uncertainty_Hz"), "Q_uncertainty": r.get("Q_uncertainty"),
               "Q_min": r.get("Q_min"), "reproduced_by_refits": r.get("reproduced_by_refits")}
        if k is not None:
            row["eigenmode_f_Hz"] = float(eig[k])
            row["relative_difference"] = (f - eig[k]) / eig[k]
            if lossy:
                frac = modes[k].get("energy_fraction", {}).get("slab")
                if frac:
                    row["slab_energy_fraction"] = frac
                    row["Q_reference"] = 1.0 / (SLAB["tan_delta"] * frac)
                    if r["Q"]:
                        row["Q_relative_difference"] = r["Q"] / row["Q_reference"] - 1.0
        poles.append(row)
    out["poles"] = poles
    if poles:
        out["max_abs_pole_vs_eigenmode"] = max(abs(p.get("relative_difference", np.inf)) for p in poles)
        if lossy:
            qd = [abs(p["Q_relative_difference"]) for p in poles if "Q_relative_difference" in p]
            out["max_abs_Q_vs_reference"] = max(qd) if qd else None
    # eigenmodes in the band without a pole: uncoupled to every port and to the source, or missed
    unmatched = [float(f) for f in out["eigenmodes_in_band"]
                 if not any(abs(p["pole_f_Hz"] - f) < 1e-6 * f for p in poles)]
    out["eigenmodes_without_pole"] = unmatched
    check = load(name + "_check")
    if check is not None:
        rows = [r for r in accepted(check["fullwave"]["sweep"]) if r["f_Hz"] < CAVITY_BAND[1]]
        out["check"] = model_errors(res, rows, accepted(fw["sweep"]))
    return out


# ------------------------------------------------------------------ core

def core_scan(points: list[tuple[float, complex, complex, complex]]) -> float | None:
    """Scan time (years, the reduced design's) by core_readout_ports' integrand, trapezoid in ln f, from
    (f, Z, dZ/dw, Psi) of the single port."""
    import core_readout_ports as CR
    rr = []
    for f, Z, dZ, Psi in points:
        w = 2 * math.pi * f
        X = -Z.imag
        Le = (w * (-dZ.imag) + X) / (2 * w)
        rr.append({"f_Hz": f, "F_eff": (abs(Psi) ** 4 / Le ** 2) if Le > 0 else None})
    return CR.scan_time_yr(rr)


def row_point(r: dict) -> tuple[float, complex, complex, complex]:
    return (r["f_Hz"], complex(r["Z_re_ohm"], r["Z_im_ohm"]),
            complex(r["dZ_domega_H"]["re"][0][0], r["dZ_domega_H"]["im"][0][0]), complex(r["Psi_re"], r["Psi_im"]))


def core_report() -> dict:
    ada = load("core_adaptive")
    parts = [load(f"core_dense_{k}") for k in range(DENSE_PARTS)]
    names = ["core_adaptive"] + [f"core_dense_{k}" for k in range(DENSE_PARTS)]
    out = {"builds": {name: run_info(name).get("build") for name in names},
           "band_Hz": list(CORE_BAND), "mesh": CORE_MESH}
    if ada is None or any(p is None for p in parts):
        out["status"] = "not solved"
        return out
    ad = ada["fullwave"]["adaptive"]
    # the parts' rows, f_max once (every part solved it; the rows must agree, as the parts share mesh and operator)
    rows = {}
    for p in parts:
        for r in p["fullwave"]["sweep"]:
            rows.setdefault(r["f_Hz"], []).append(r)
    fmax_rows = rows[CORE_BAND[1]]
    out["dense_parts_f_max_spread"] = max(abs(complex(r["Z_re_ohm"], r["Z_im_ohm"]) /
                                              complex(fmax_rows[0]["Z_re_ohm"], fmax_rows[0]["Z_im_ohm"]) - 1)
                                          for r in fmax_rows)
    dense_sweep = [v[0] for _, v in sorted(rows.items())]
    drows = accepted(dense_sweep)
    out.update({"adaptive_solves": ad["solves"], "dense_solves": len(dense_sweep),
                "dense_accepted": len(drows), "converged": ad["converged"], "stop_reason": ad["stop_reason"],
                "estimated_error": ad["estimated_error"], "last_prediction_error": ad["last_prediction_error"],
                "support_points": len(ad["model"]["support_Hz"]),
                "self_resonances": ada["fullwave"]["self_resonances"]})
    out["model_vs_dense"] = model_errors(ada, drows, accepted(ada["fullwave"]["sweep"]))
    # scan time: dense solves, their 4-point subset, the adaptive solves, the adaptive model's dense curve
    dpts = [row_point(r) for r in sorted(drows, key=lambda r: r["f_Hz"])]
    sub = [dpts[i] for i in range(0, len(dpts), (len(dpts) - 1) // 3)] if len(dpts) == DENSE_POINTS else None
    apts = [row_point(r) for r in sorted(accepted(ada["fullwave"]["sweep"]), key=lambda r: r["f_Hz"])]
    import axionfem.circuit as ck
    dd = ck.fullwave_dense(ada)
    mpts = [(float(f), complex(z), complex(dz[0][0]), complex(p))
            for f, z, dz, p in zip(dd["f_Hz"], dd["Z"], dd["dZ_domega_H"], dd["Psi"])]
    # the model at the dense solves' 61 frequencies, its derivative by a centred difference: the same quadrature as
    # the dense reference, so the difference is the model's alone
    model = model_of(ada)

    def model_point(f: float) -> tuple[float, complex, complex, complex]:
        h = 1e-6 * f
        m = model(f)
        dz = (model(f + h)["Z_matrix"][0, 0] - model(f - h)["Z_matrix"][0, 0]) / (2 * h) / (2 * math.pi)
        return (f, m["Z_matrix"][0, 0], dz, m["Psi_vector"][0])
    m61 = [model_point(p[0]) for p in dpts]
    t_dense = core_scan(dpts)
    scans = {"dense_61_solves": t_dense, "four_point_subset": core_scan(sub) if sub else None,
             "adaptive_solves_only": core_scan(apts), "adaptive_model_dense": core_scan(mpts),
             "adaptive_model_at_the_61": core_scan(m61)}
    out["scan_time_yr"] = scans
    out["scan_time_relative_to_dense"] = {k: (v / t_dense - 1.0) if (v and t_dense) else None
                                          for k, v in scans.items()}
    # L_eff of the model (its derivative) against the dense solves' exact dZ/dw
    worst = 0.0
    for (f, Z, dZ, _), (_, Zm, dZm, _) in zip(dpts, m61):
        w = 2 * math.pi * f
        Le = (w * (-dZ.imag) - Z.imag) / (2 * w)
        Lm = (w * (-dZm.imag) - Zm.imag) / (2 * w)
        worst = max(worst, abs(Lm / Le - 1))
    out["L_eff_model_vs_dense_max_relative"] = worst
    return out


# ------------------------------------------------------------------ invariance

def flatten(x, prefix: str = ""):
    if isinstance(x, dict):
        for k, v in x.items():
            if any(word in k.lower() for word in TIMING_WORDS) or k.endswith("_s"):   # timings (setup_s, solve_s, ...)
                continue
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}/{i}")
    else:
        yield prefix, x


def compare(a: dict, b: dict) -> dict:
    fa = dict(flatten(a))
    fb = dict(flatten(b))
    differ = []
    numbers = 0
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if isinstance(va, float) or isinstance(vb, float):
            numbers += 1
        same = va == vb or (isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb))
        if not same:
            differ.append({"key": k, "old": va, "new": vb})
    return {"values_compared": len(set(fa) | set(fb)), "floats_compared": numbers, "differences": len(differ),
            "first_differences": differ[:20]}


def invariance(old: Path, new: Path, nranks: int) -> None:
    for name, doc in invariance_designs().items():
        for tag, build in (("old", old), ("new", new)):
            solve(INVARIANCE / tag / name, doc, build, nranks, INVARIANCE_ENV.get(name))


def invariance_report() -> dict:
    out = {}
    for name in invariance_designs():
        a = INVARIANCE / "old" / name / "results.json"
        b = INVARIANCE / "new" / name / "results.json"
        if not (a.exists() and b.exists()):
            out[name] = {"status": "not solved"}
            continue
        ra = json.loads(a.read_text())
        rb = json.loads(b.read_text())
        c = compare(ra, rb)
        info_a = json.loads((a.parent / "run_info.json").read_text())
        info_b = json.loads((b.parent / "run_info.json").read_text())
        c.update({"old_build": info_a["build"], "new_build": info_b["build"],
                  "frequencies": len(rb.get("fullwave", {}).get("sweep", [])),
                  "returncodes": [info_a["returncode"], info_b["returncode"]]})
        out[name] = c
    return out


# ------------------------------------------------------------------ record

def provenance() -> dict:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, env=env).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES}}


# What each build directory used by the record was built from (the record identifies the binary).
BUILDS = {   # build directory name -> free-text description of the committed record's builds
    "build1": "a build without the adaptive sweep (invariance reference)",
    "build5": "a build with a change to the adaptive dense output (the Core's dense parts only: fixed lists, "
              "bit-identical across builds by the invariance check)",
    "build9": "an intermediate build",
    "build14": "the Core and invariance runs (the pole uncertainty's offset term of build16 changes neither)",
    "build16": "adds the model's offset in the pole uncertainty; the gap, cavity and high-Q runs",
}


def report() -> None:
    rec = {"check": "adaptive_sweep", "date": time.strftime("%Y-%m-%d %H:%M"), "provenance": provenance(),
           "builds": BUILDS,
           "tolerance": TOLERANCE, "cavity_band_Hz": list(CAVITY_BAND),
           "cavity": cavity_report(False), "cavity_lossy": cavity_report(True), "core": core_report(),
           "invariance": invariance_report(), "gap_validation": gap_report(), "high_q": highq_report()}
    OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, indent=1, default=float)[:6000])
    print("wrote", OUT)


def main() -> None:
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    if a[0] == "run":
        build = Path(a[2]).resolve()
        nranks = int(a[3]) if len(a) > 3 else 4
        solve(RUNS / a[1], CASES[a[1]](), build, nranks)
    elif a[0] == "gap":
        build = Path(a[2]).resolve()
        nranks = int(a[3]) if len(a) > 3 else 4
        run_gap(a[1], build, nranks)
    elif a[0] == "samples":
        write_samples()
    elif a[0] == "invariance":
        invariance(Path(a[1]).resolve(), Path(a[2]).resolve(), int(a[3]) if len(a) > 3 else 4)
    elif a[0] == "report":
        report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
