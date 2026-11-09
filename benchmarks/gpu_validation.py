"""The CUDA build's 3D solves on a GPU (AXF_DEVICE=cuda) against the CPU build: same designs, same meshes.

    python benchmarks/gpu_validation.py run --name NAME --exe AXF --deps PREFIX [--device cpu|cuda] [--ranks N]
                                            [--cases a,b] [--threads T]
    python benchmarks/gpu_validation.py report

The only GPU code path in axionFEM is hypre (AMS for the 3D MQS and full-wave H(curl) solves, AME for the cavity
eigenmodes, BoomerAMG for the H1 solves of the Gauss-law guard, the port feeds and the divergence cleaning), selected
at run time by AXF_DEVICE (docs/install.md, "GPU builds"). Each `run` solves every case with `axf solve --save` of one
build (--exe, linked against --deps) on one device, as one configuration NAME. The configurations of the committed
record (the build of each run is in its run.json and in the record: exe, its sha256 and libaxionfem's):

- `cpu`:              the CPU build under test (third_party/install) on one core, the reference of the small cases;
- `main`:             a second CPU build (of the main line) on the same node, in the same job, compared bit for bit;
- `cuda_host`:        the CUDA build with AXF_DEVICE=cpu (hypre built with CUDA, executing on the host), same node;
- `cuda_device_a40`, `cuda_device_h100`: the CUDA build with AXF_DEVICE=cuda on one GPU and one core;
- `cpu_a40`:          the CPU build on one core of an A40 node, the reference and the timing baseline of the order-2
                      cases; `cpu_r16_a40`, `cpu_r14_h100`: on 16 (14) MPI ranks of the GPU nodes, for the timings;
- `*_repeat`:         the same configuration again: `report` also compares it with the first run, bit for bit;
- `buildN_*`:         runs with other CUDA builds of the code, named after their build directory (`buildcuda`,
                      `buildcuda3`, `4`, `6`), with variants `gpuams` (MQS AMS with
                      MFEM's GPU options) and `fwgpuams` / `fwcpuams` (full-wave AMS with MFEM's GPU / host options),
                      and earlier CPU builds (`build1`, `build2`).

Every run meshes with the same mesher ($AXF_MESHER), and `report` checks that the meshes are byte-identical. The
saved fields of every run are sampled afterwards by this interpreter's axionfem package (PYTHONPATH: the CPU build's),
so that the only difference between two configurations is the solve.

Cases (CASES below): the coarse 3D solenoid (MQS, infinity shell: L_p, Psi, W, field samples), the Stage 1 frame
(MQS pickup and a four-frequency AMS-GMRES full-wave sweep with the Gauss-law guard, field probes), a saturated iron
core (Newton: a new AMS setup per step), three cavities with skin-effect wall losses (the TM010 pillbox with the AME
and the shift-invert eigensolvers, and with a retained conductor rod), Stage 1 at one frequency on the STRUMPACK
full-wave paths (AXF_FW_SOLVER=strumpack, without and with AXF_FW_GAUGE=tree-cotree), and larger order-2 versions of
the solenoid (on two meshes) and of Stage 1 for timing (ND2_CASES), and Stage 1 at order 2 with a weakly conducting
block (INDICATOR_CASES: the order-2 error indicator's complex-eps projection, GMRES with PRESB).

Every solve and sample pins OpenBLAS's kernels, OPENBLAS_CORETYPE=Haswell (AXF_GPUVAL_OPENBLAS_CORETYPE overrides):
MFEM builds its bases through LAPACK, and OpenBLAS's run-time kernel choice moves their last bits and the order-2
error indicator with them (OPENBLAS_CORETYPE below). run.json records the pin; runs without it record none.

`report` compares every configuration against the CPU build on one rank (`cpu`, else `cpu_a40`) on each case: the key
quantities (L_p, Psi, W, Z, dZ/domega, Psi vector, energies, field samples and probes; per cavity mode f, C, G, Q0,
Q_wall, the walls' G and the peak wall fields; in INDICATOR_CASES the error indicator's total, magnetic and electric
parts per port) as relative differences, and lists any key quantity the run lacks;
every other numeric output of results.json as a census; flags a run pinned to other OpenBLAS kernels than its
reference (openblas_coretype_mismatch); every leaf bit for bit (equal or not, and the keys only one
side has); iteration counts side by side, and the timings. Writes benchmarks/records/gpu_validation.json when the run
directory is the default one (AXF_GPUVAL_RUNS unset), keeping the record's provenance when no run is newer than it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

RUNS = Path(os.environ.get("AXF_GPUVAL_RUNS", ROOT / "runs" / "gpu_validation")).resolve()
OUT = ROOT / "benchmarks" / "records" / "gpu_validation.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
REFERENCES = ("cpu", "cpu_a40", "cpu_h100")   # the first one present: the CPU build on one rank

# field samples of the MQS cases (static field and pickup field per ampere), [m]
SOLENOID_POINTS = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.45, 0.5), (0.6, 0.6, -0.4), (0.25, -0.2, 0.9),
                   (1.2, 0.0, 0.0), (0.0, 0.0, 1.4), (-0.8, 0.9, 0.3), (0.45, 0.2, -1.1), (2.0, 0.5, 0.5)]
STAGE1_POINTS = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.1), (0.0, 0.1, 0.0), (0.25, 0.1, 0.0), (0.4, 0.0, 0.0),
                 (0.0, 0.0, 0.25), (0.0, 0.03, 0.25), (0.5, 0.2, -0.3), (0.0, 0.3, 0.5), (-0.35, -0.2, 0.35)]
IRON_POINTS = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.1), (0.45, 0.0, 0.0), (0.0, 0.55, 0.2), (0.7, 0.3, -0.4),
               (0.0, 0.0, 0.5), (1.0, 0.0, 0.0)]


def solenoid(order: int, size: float, surface: float) -> dict:
    """The solenoid template in 3D (tests/test_3d.py::test_solenoid_3d_vs_2d): a coax pickup in a solenoid,
    exact open boundary (infinity shell)."""
    import axionfem as af
    from axionfem.convert import to_3d
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    return to_3d(d2, mesh={"size": size, "surface_size": surface, "order": order}).doc


def stage1(order: int, frequencies: list[float]) -> dict:
    """The Stage 1 frame with two slit ports: the MQS pickup, then the AMS-GMRES full-wave sweep (PEC box, Gauss-law
    guard in its default mode, validation shunts and the source check of the committed design)."""
    doc = json.loads(STAGE1.read_text())
    doc["mesh"]["order"] = order
    doc["fullwave"]["frequencies"] = frequencies
    doc["fullwave"]["field_probes"] = [list(p) for p in STAGE1_POINTS]
    return doc


# a weakly conducting block (sigma / omega eps0 = 25 at 12 MHz, 300 at 1 MHz) in the loop's opening of the Stage 1
# frame, clear of the loop and of the slit: the conductivity of tests/test_fullwave3d_error_indicator.py's conducting
# line
LOSSY_BLOCK = {"name": "lossy_block", "material": "lossy",
               "solid": {"box": {"min": [-0.15, -0.04, -0.15], "max": [0.15, 0.04, 0.1]}}}
LOSSY_MATERIAL = {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": 0.0167}


def stage1_lossy(order: int, frequencies: list[float]) -> dict:
    """The Stage 1 frame at order 2 with a weakly conducting block in the loop's opening (0.0167 S/m): the field
    materials' loss phases differ (vacuum against about 88 degrees at 12 MHz), so the order-2 error indicator's
    gradient projection takes its complex-eps path (GMRES with PRESB: the real and imaginary parts and their sum
    assembled by hypre, the sum's diagonal read by stored_diagonal, BoomerAMG of the sum on the GPU). The copper loop
    is outside the field (rows of every projection operator empty there), the imaginary part is zero in the vacuum
    (empty rows of that block), and eps changes with the frequency (the projection is reassembled at each one): the
    places where hypre on a GPU leaves rows empty (docs/validation/gpu.md, fixes 2, 4 and 7)."""
    doc = stage1(order, frequencies)
    doc["regions"].append(dict(LOSSY_BLOCK))
    doc.setdefault("materials", {})["lossy"] = dict(LOSSY_MATERIAL)
    return doc


def iron() -> dict:
    """A saturated iron core in a coil (tests/test_3d.py::test_saturated_iron_newton_3d): Newton, with a new
    operator and AMS setup at every step."""
    import axionfem as af
    from axionfem.convert import to_3d
    doc = {"schema": "axionfem.design/1", "name": "fe",
           "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.2, 0.2]}}, "material": "winding"},
                       {"name": "core", "shape": {"rect": {"r": [0.0, 0.4], "z": [-0.3, 0.3]}}, "material": "iron"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 3e6}],
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.01}}
    return to_3d(af.Design(doc), mesh={"size": 0.3, "surface_size": 0.15, "order": 1}).doc


CAVITY_WALL = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 4.0}}


def pillbox_ame() -> dict:
    """The TM010 pillbox of tests/test_cavity3d.py with skin-effect walls, solved by hypre's AME eigensolver (LOBPCG
    with AMS): f, the form factor and the wall losses, whose wall-current solve once lost them on a GPU."""
    from axionfem import cavity as cv
    return cv.cylinder_design_3d(0.1, 0.2, h=0.02, modes=1, wall_material=CAVITY_WALL).doc


def pillbox_si() -> dict:
    """The same pillbox with the shift-invert eigensolver (STRUMPACK LU of A - sigma M on the host; hypre for the
    projections), target 1.1 GHz."""
    from axionfem import cavity as cv
    return cv.cylinder_design_3d(0.1, 0.2, h=0.02, modes=1, target_f_Hz=1.1e9, wall_material=CAVITY_WALL).doc


def pillbox_rod_si() -> dict:
    """The pillbox with an off-centre retained conductor rod (wall faces between the cavity and a conductor volume),
    shift-invert at 0.6 GHz."""
    from axionfem import cavity as cv
    rod = {"radius": 0.015, "gap": 0.02, "x": 0.045, "hole": False}
    return cv.cylinder_design_3d(0.1, 0.2, h=0.02, modes=1, rod=rod, target_f_Hz=0.6e9, wall_material=CAVITY_WALL).doc


CASES = {
    "solenoid": {"design": lambda: solenoid(1, 0.4, 0.13), "points": SOLENOID_POINTS},
    "stage1": {"design": lambda: stage1(1, [1e3, 1e5, 1e6, 1.2e7]), "points": STAGE1_POINTS},
    "iron": {"design": iron, "points": IRON_POINTS},
    "pillbox_ame": {"design": pillbox_ame, "points": []},
    "pillbox_si": {"design": pillbox_si, "points": []},
    "pillbox_rod_si": {"design": pillbox_rod_si, "points": []},
    # the Stage 1 frame at one frequency on the STRUMPACK full-wave paths: hypre assembles (and on a GPU applies the
    # projections), STRUMPACK factors on the host
    "stage1_direct": {"design": lambda: stage1(1, [1e6]), "points": STAGE1_POINTS,
                      "env": {"AXF_FW_SOLVER": "strumpack"},
                      "description": "Stage 1 at 1 MHz, the STRUMPACK direct full-wave solve (AXF_FW_SOLVER=strumpack)"},
    "stage1_tc": {"design": lambda: stage1(1, [1e5]), "points": STAGE1_POINTS,
                  "env": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"},
                  "description": "Stage 1 at 0.1 MHz, the tree-cotree gauged STRUMPACK solve (AXF_FW_GAUGE=tree-cotree)"},
    "solenoid_o2": {"design": lambda: solenoid(2, 0.4, 0.13), "points": SOLENOID_POINTS},
    "stage1_o2": {"design": lambda: stage1(2, [1e5, 1e6, 1.2e7]), "points": STAGE1_POINTS},
    "solenoid_o2_fine": {"design": lambda: solenoid(2, 0.25, 0.08), "points": SOLENOID_POINTS},
    # the order-2 error indicator's complex-eps projection (PRESB): the indicator's estimates per port are key
    # quantities, and the projection's iterations are listed (a projection that returned at once on a GPU gave an
    # electric part 79 times too small without a warning: fix 7)
    "stage1_o2_lossy": {"design": lambda: stage1_lossy(2, [1e6, 1.2e7]), "points": STAGE1_POINTS, "indicator": True,
                        "description": "Stage 1 at order 2, 1 and 12 MHz, with a weakly conducting block (0.0167 S/m) "
                                       "in the loop's opening: the order-2 error indicator's complex-eps gradient "
                                       "projection (GMRES with PRESB)"},
}
DEFAULT_CASES = ["solenoid", "stage1", "iron", "pillbox_ame", "pillbox_si", "pillbox_rod_si", "stage1_direct",
                 "stage1_tc"]
ND2_CASES = ["solenoid_o2", "stage1_o2", "solenoid_o2_fine"]
# the cases whose report compares the error indicator (their "indicator" flag; run them with `time`,
# AXF_GPUVAL_TIME_CASES=stage1_o2_lossy)
INDICATOR_CASES = [c for c in CASES if CASES[c].get("indicator")]
# caveats that belong with the record (report copies them into it)
RECORD_NOTES = {
    "order2_indicator": "the order-2 error indicator values of stage1_o2 and stage1_o2_lossy (and of the other order-2 "
                        "runs) were measured before the indicator's residual problems moved to the "
                        "canonical complement (docs/physics/fullwave_error_indicator.md): "
                        "they were not rerun. The GPU-CPU comparisons stand: both devices ran the same code, the "
                        "local solves run on the host in both builds, and without the tree the host effect is 1e-11",
}
# The OpenBLAS kernel family of every solve and sample. MFEM builds the ND and H1 bases through LAPACK, and OpenBLAS
# picks its kernels at run time: AVX-512 hosts (Xeon Gold) get SkylakeX, AVX2 hosts (EPYC 7713P and 7742) Zen, and the
# last bits of the bases differ, which moves the order-2 indicator. Haswell runs on every node and gives
# Zen's bits, so runs on different hosts compare like with like. Not in scripts/env.sh: it costs BLAS speed on
# AVX-512 hosts. AXF_GPUVAL_OPENBLAS_CORETYPE overrides it.
OPENBLAS_CORETYPE = os.environ.get("AXF_GPUVAL_OPENBLAS_CORETYPE", "Haswell")


# ---------------------------------------------------------------- run
def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return out.stdout.strip().splitlines()[0] + (f" (CUDA_VISIBLE_DEVICES={visible})" if visible else "")
    except Exception:  # noqa: BLE001
        return ""


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def solve_env(deps: str, device: str, threads: int) -> dict:
    env = dict(os.environ)
    lib = f"{deps}/lib:{deps}/lib64"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["AXF_DEVICE"] = device
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_CORETYPE"] = OPENBLAS_CORETYPE
    env["SLURM_EXPORT_ENV"] = "ALL"
    for key in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_TRES_PER_TASK"):
        env.pop(key, None)
    return env


def sample(run_dir: Path, points: list) -> None:
    """Sample the saved fields with the CPU build (this interpreter's axionfem package)."""
    if not points:
        return
    code = ("import json, sys\n"
            "import axionfem as af\n"
            "pts = json.loads(sys.argv[2])\n"
            "r = af.load_solution(sys.argv[1])\n"
            "s = r.sample([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])\n"
            "print(json.dumps({k: list(map(float, v)) for k, v in s.items()}))\n")
    env = dict(os.environ)
    env["AXF_DEVICE"] = "cpu"
    env["OPENBLAS_CORETYPE"] = OPENBLAS_CORETYPE
    out = subprocess.run([sys.executable, "-c", code, str(run_dir / "fields"), json.dumps(points)],
                         capture_output=True, text=True, env=env)
    if out.returncode != 0:
        (run_dir / "samples_error.txt").write_text(out.stdout + "\n" + out.stderr)
        return
    samples = json.loads(out.stdout.strip().splitlines()[-1])
    (run_dir / "samples.json").write_text(json.dumps({"points_m": points, **samples}) + "\n")


def run_case(case: str, name: str, exe: str, deps: str, device: str, ranks: int, threads: int) -> None:
    run_dir = RUNS / case / name
    if (run_dir / "run.json").exists():
        print(f"{case} {name}: kept", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = CASES[case]["design"]()
    doc["name"] = f"gpuval_{case}"
    design = run_dir / "design.json"
    design.write_text(json.dumps(doc, indent=1) + "\n")
    cmd = [exe, "solve", str(design), "-o", str(run_dir), "-v", "2", "--save"]
    if ranks > 1:
        cmd = ["srun", "-n", str(ranks), "--cpus-per-task", str(threads), "--cpu-bind=cores"] + cmd
    env = solve_env(deps, device, threads)
    env.update(CASES[case].get("env", {}))
    trace = None
    if device != "cpu":   # GPU utilization and memory, twice a second
        trace = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits", "-lms", "500", "-f", str(run_dir / "gpu_trace.csv")])
    t0 = time.time()
    with open(run_dir / "solve.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    if trace is not None:
        trace.terminate()
        trace.wait()
    record = {"case": case, "name": name, "exe": str(Path(*Path(exe).parts[-2:])), "exe_sha256": sha256(Path(exe)),
              "libaxionfem_sha256": sha256(Path(exe).parent / "libaxionfem.so"), "deps": deps, "device": device,
              "ranks": ranks, "threads": threads, "command": cmd, "returncode": proc.returncode, "wall_seconds": wall,
              "solver_env": {k: v for k, v in env.items() if k.startswith(("AXF_AMS_", "AXF_FW_", "AXF_SOLVER"))},
              "openblas_coretype": env["OPENBLAS_CORETYPE"],
              "cpu": cpu_name(), "gpu": gpu_name() if device != "cpu" else "",
              "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if proc.returncode == 0 and (run_dir / "fields").exists():
        t1 = time.time()
        sample(run_dir, [list(p) for p in CASES[case]["points"]])
        record["sample_seconds"] = time.time() - t1
    (run_dir / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{case} {name}: exit {proc.returncode}, {wall:.1f} s", flush=True)


# ---------------------------------------------------------------- report
def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
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


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def provenance(runs: dict) -> dict:
    diff = git("diff", "HEAD")
    exes = sorted({r["run"]["exe"] for case in runs.values() for r in case.values()})
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "executables": {e: {"axf": file_record(e), "libaxionfem": file_record(str(Path(e).parent / "libaxionfem.so"))}
                            for e in exes},
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")), "driver": file_record(__file__),
            "date": time.strftime("%Y-%m-%d")}


# results.json leaves that are timings, memory or bookkeeping, not results. Matched on the leaf's own name (its last
# key, array indices dropped), so that a result under a key such as losses/walls/outer or peak_wall_fields is compared
SKIP_LEAF = re.compile(r"(seconds|_s$|rss|bytes|elapsed|^date$|mtime|sha256|^hypre_device$|^mpi_ranks$)", re.I)
# whole subtrees that are bookkeeping: the engine's clocks, and solver options (set, not computed)
SKIP_TREE = re.compile(r"(^|/)timings?(/|\[|$)|/solver/options/|/solver_settings/ams_options/", re.I)


def skipped(key: str) -> bool:
    leaf = re.sub(r"\[\d+\]", "", key.rsplit("/", 1)[-1])
    return bool(SKIP_LEAF.search(leaf) or SKIP_TREE.search(key))


# Set apart as solver diagnostics: what measures the solve itself rather than the solution, and differs between
# preconditioners by construction: iteration and step counts, residuals and backward errors, preconditioned norms;
# and the consistency checks that are zero (or one) in exact arithmetic and so carry the solver's noise: reciprocity,
# hermiticity, closure and the dZ/domega energy identity, operator consistency, Gram minimum eigenvalues and condition
# numbers of the port sources, leaks and imbalances, the tree-cotree right-hand side's charges (1e-16 of the load) and
# its check of K on discrete gradients (zero in exact arithmetic, 6e-8 of omega^2 M in floating point), an
# eigenvector's divergence and mass overlaps, Ritz estimates
DIAGNOSTIC = re.compile(r"(iteration|residual|backward_error|field_fraction|consistency|reciprocity|hermiticity|"
                        r"identity|cancellation|refinement_steps|its$|_its|setups|assemblies|rhs_solves|leak|"
                        r"gram_min|closure|imbalance|floor|preconditioned_.*norm|rounding_charge|relative_last_step|"
                        r"rhs_norm|label_exchange|mass_overlap|divergence_fraction|ritz|rayleigh|lanczos_steps|"
                        r"/solves$|restarts|charge_relative|gradient_block_check)",
                        re.I)
# zero in exact arithmetic for a lossless model: compared relative to the impedance scale max |Z| (ohm), not to itself
ZERO_LOSSLESS = re.compile(r"resistance_(min|max)_eigenvalue_ohm$")


def flatten(x, prefix: str = "") -> dict:
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(x, bool):
        pass
    elif isinstance(x, (int, float)):
        out[prefix] = float(x)
    return out


def census(a: dict, b: dict) -> dict:
    """Every numeric output of two results.json files, leaf by leaf: |a - b| / max(|a|, |b|), outside the timings, and
    |a - b| relative to the largest magnitude under the same key (an array's entries scaled by the array, a field probe
    by the largest probe value): the second is the norm-wise measure, the first flags small entries of an array.
    Counted apart: solver diagnostics (DIAGNOSTIC above: iteration counts, residuals, preconditioned norms, and the
    consistency checks that are zero in exact arithmetic), and leaves at the round-off of their neighbours (|b| <
    1e-12 of the largest magnitude under the same key): an entry that is zero in exact arithmetic carries only solver
    noise. A lossless model's resistance eigenvalues (zero in exact arithmetic) are compared relative to max |Z|."""
    fa = flatten(a)
    fb = flatten(b)
    keys = [k for k in fb if k in fa and not skipped(k)]
    scale = {}
    for k in keys:
        base = re.sub(r"\[\d+\]", "[]", k)
        scale[base] = max(scale.get(base, 0.0), abs(fb[k]), abs(fa[k]))
    zscale = max((abs(v) for k, v in fb.items() if "Z_matrix" in k), default=0.0)
    scaled = []   # |a - b| / (largest magnitude under the same key): the norm-wise measure of an array
    rows = []
    diag = []
    noise = 0
    for k in keys:
        x, y = fa[k], fb[k]
        if ZERO_LOSSLESS.search(k) and zscale > 0:   # relative to the impedance, not to itself
            rows.append((abs(x - y) / zscale, k, x, y))
            scaled.append((abs(x - y) / zscale, k, x, y))
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            if not (math.isnan(x) and math.isnan(y)) and x != y:
                rows.append((math.inf, k, x, y))
            continue
        m = max(abs(x), abs(y))
        r = abs(x - y) / m if m > 0 else 0.0
        base = re.sub(r"\[\d+\]", "[]", k)
        if DIAGNOSTIC.search(k):
            diag.append((r, k, x, y))
        elif m < 1e-12 * scale[base]:
            noise += 1
        else:
            rows.append((r, k, x, y))
            scaled.append((abs(x - y) / scale[base] if scale[base] > 0 else 0.0, k, x, y))
    rows.sort(reverse=True)
    scaled.sort(reverse=True)
    counts = {f"above_{t:g}": sum(1 for r in rows if r[0] > t) for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)}
    return {"compared": len(rows), "solver_diagnostics": len(diag), "round_off_entries": noise,
            "missing_in_run": sorted(k for k in fb if k not in fa and not skipped(k))[:20],
            "max_relative_difference": rows[0][0] if rows else 0.0,
            "max_difference_relative_to_key_scale": scaled[0][0] if scaled else 0.0, **counts,
            "worst": [{"key": k, "run": x, "reference": y, "relative": r} for r, k, x, y in rows[:12]],
            "worst_relative_to_key_scale": [{"key": k, "run": x, "reference": y, "relative": r}
                                            for r, k, x, y in scaled[:6]]}


def leaves(x, prefix: str = "") -> dict:
    """Every leaf of a results.json (numbers, strings, booleans, nulls), for a bit-for-bit comparison."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(leaves(v, f"{prefix}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(leaves(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def bit_identity(a: dict, b: dict) -> dict:
    """a against b leaf by leaf, outside the timings and bookkeeping (skipped()): numbers by their bits (Python's
    float equality on the parsed JSON, which round-trips doubles exactly), strings and booleans as they are. Strings
    that are absolute paths (the run directories) are left out. Keys present in one of the two only are listed apart:
    between main and this branch they are the keys the branch adds."""
    la = {k: v for k, v in leaves(a).items() if not skipped(k) and not (isinstance(v, str) and v.startswith("/"))}
    lb = {k: v for k, v in leaves(b).items() if not skipped(k) and not (isinstance(v, str) and v.startswith("/"))}
    common = [k for k in lb if k in la]
    differ = [k for k in common if not (la[k] == lb[k] or (la[k] != la[k] and lb[k] != lb[k]))]
    return {"identical": not differ, "compared": len(common), "differing": len(differ), "differing_keys": differ[:20],
            "only_in_run": sorted(k for k in la if k not in lb)[:30],
            "only_in_reference": sorted(k for k in lb if k not in la)[:30]}


def rel(a, b) -> float | None:
    """Norm-wise relative difference max|a - b| / max|b| (None when either is missing)."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        return None
    m = np.abs(b).max()
    return float(np.abs(a - b).max() / m) if m > 0 else float(np.abs(a - b).max())


def cmatrix(block) -> np.ndarray | None:
    if not block:
        return None
    return np.array(block["re"], dtype=float) + 1j * np.array(block["im"], dtype=float)


def probes_of(row: dict, field: str) -> np.ndarray | None:
    """The field-probe samples of one field ("E_V_m_per_A" or "B_T_per_A") of a sweep row: every port, real and
    imaginary parts, in a fixed order. E and B are compared apart: their units differ by orders of magnitude."""
    found = []
    for port in row.get("field_probe_port_fields") or []:
        block = port.get(field) or {}
        for part in ("re", "im"):
            vals = flatten(block.get(part, []))
            found.extend(vals[k] for k in sorted(vals))
    return np.array(found) if found else None


def indicator_parts(row: dict) -> np.ndarray | None:
    """The error indicator of a sweep row per port: [total, magnetic part, electric part] (None if a port has none)."""
    parts = []
    for port in row.get("port_solutions", []):
        ind = port.get("error_indicator") or {}
        values = [ind.get(k) for k in ("estimated_relative_error", "magnetic_relative_error",
                                       "electric_relative_error")]
        if None in values:
            return None
        parts.append(values)
    return np.array(parts) if parts else None


def key_values(d: dict, samples: dict | None, indicator: bool = False) -> dict:
    """The quantities compared one by one: MQS (L_p, Psi, W, pickup energy, samples), each full-wave frequency, and
    each cavity mode (f, form factor, G, Q0, Q_wall, each wall's G, the peak wall fields). With INDICATOR, also the
    error indicator's parts per port and frequency, each part compared as its own quantity."""
    c = d.get("coupling", {})
    out = {"L_p_H": c.get("L_p_H"), "Psi_SI_H_T_m2": c.get("Psi_SI_H_T_m2"), "W_mag_J": c.get("W_mag_J"),
           "pickup_energy_J": (d.get("pickup") or {}).get("energy_J"),
           "static_energy_J": (d.get("static") or {}).get("energy_J")}
    if samples:
        out["samples_B_lab_T"] = np.array([samples[k] for k in ("Bx", "By", "Bz")])
        out["samples_pickup_B_T_per_A"] = np.array([samples[k] for k in ("pickup_Bx", "pickup_By", "pickup_Bz")])
    fw = d.get("fullwave") or {}
    for row in fw.get("sweep", []):
        f = f"{row['f_Hz']:g}Hz"
        z = np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)
        out[f"{f}/Z_ohm"] = z
        out[f"{f}/dZ_domega_H"] = cmatrix(row.get("dZ_domega_H"))
        out[f"{f}/Psi_vector"] = cmatrix(row.get("Psi_vector"))
        out[f"{f}/L_H"] = row.get("L_H")
        out[f"{f}/energy_magnetic_J_per_A2"] = cmatrix(row.get("energy_magnetic_J_per_A2"))
        out[f"{f}/energy_electric_J_per_A2"] = cmatrix(row.get("energy_electric_J_per_A2"))
        out[f"{f}/field_probes_E"] = probes_of(row, "E_V_m_per_A")
        out[f"{f}/field_probes_B"] = probes_of(row, "B_T_per_A")
        if indicator:
            parts = indicator_parts(row)
            for j, part in enumerate(("total", "magnetic", "electric")):
                out[f"{f}/indicator_{part}"] = parts[:, j] if parts is not None else None
    for i, mode in enumerate((fw.get("eigenmodes") or {}).get("modes", [])):
        m = f"mode{i}"
        out[f"{m}/f_Hz"] = mode.get("f_Hz")
        out[f"{m}/C"] = (mode.get("form_factor") or {}).get("C")
        losses = mode.get("losses") or {}
        for key in ("G_ohm", "Q0", "Q_wall"):
            out[f"{m}/{key}"] = losses.get(key)
        walls = losses.get("walls") or {}
        out[f"{m}/walls_G_ohm"] = [walls[b].get("G_ohm") for b in sorted(walls)] if walls else None
        if out[f"{m}/walls_G_ohm"] is not None and None in out[f"{m}/walls_G_ohm"]:
            out[f"{m}/walls_G_ohm"] = None
        peak = mode.get("peak_wall_fields") or {}
        out[f"{m}/B_t_max_flux"] = peak.get("B_t_max_flux")
        out[f"{m}/E_max"] = peak.get("E_max")
    return out


def iterations(d: dict, indicator: bool = False) -> dict:
    """Iteration counts; with INDICATOR also the error indicator's gradient projection per port and frequency (PCG on
    the |eps| path, GMRES with PRESB on the complex-eps path)."""
    out = {}
    solver = d.get("solver") or {}
    if solver.get("solves") is not None:
        out["mqs_pcg_per_solve"] = [s.get("iterations") for s in solver["solves"]]
    nl = (d.get("static") or {}).get("nonlinear") or {}
    if nl:
        out["newton_iterations"] = nl.get("newton_iterations")
    fw = d.get("fullwave") or {}
    if fw:
        out["fullwave_gmres_total"] = fw.get("gmres_iterations_total")
        per = {}
        guard = {}
        for row in fw.get("sweep", []):
            f = f"{row['f_Hz']:g}Hz"
            solves = [p.get("solver") or {} for p in row.get("port_solutions", [])]
            solves += [v for k, v in sorted(row.items()) if k.endswith("_solver") and isinstance(v, dict)]
            per[f] = [s.get("gmres_iterations") for s in solves]
            guard[f] = [(s.get("gauss_law_guard") or {}).get("pcg_iterations_max") for s in solves]
        if fw.get("sweep"):
            out["fullwave_gmres_per_solve"] = per
            out["guard_pcg_max_per_solve"] = guard
        if indicator and fw.get("sweep"):
            out["indicator_projection_per_port"] = {
                f"{row['f_Hz']:g}Hz": [((p.get("error_indicator") or {}).get("gradient") or {}).get("iterations")
                                       for p in row.get("port_solutions", [])] for row in fw["sweep"]}
        eig = fw.get("eigenmodes") or {}
        if eig:
            es = eig.get("solver") or {}
            out["eigensolver"] = {k: es.get(k) for k in ("method", "solves", "lanczos_steps") if es.get(k) is not None}
            out["cavity_wall_current_pcg_per_mode"] = [((m.get("losses") or {}).get("wall_current_solve") or {})
                                                       .get("pcg_iterations") for m in eig.get("modes", [])]
    return out


def timings(d: dict, run: dict) -> dict:
    """The wall clock of the `axf solve` process (meshing, solving, writing results and the saved fields: the shared
    file system stalled for minutes on a few writes) and the engine's own clocks: `compute_seconds` = the MQS run's
    total (mesh, assembly, solves, post-processing) plus the full-wave analysis."""
    out = {"wall_seconds": run.get("wall_seconds")}
    total = (d.get("timing") or {}).get("total_s")
    if total is not None:
        out["compute_seconds"] = total + ((d.get("fullwave") or {}).get("seconds") or 0.0)
    solver = d.get("solver") or {}
    if solver.get("solves"):
        out["mqs_setup_seconds"] = sum(s.get("setup_s", 0.0) for s in solver["solves"])
        out["mqs_solve_seconds"] = sum(s.get("solve_s", 0.0) for s in solver["solves"])
    fw = d.get("fullwave") or {}
    if fw:
        out["fullwave_seconds"] = fw.get("seconds")
        out["fullwave_setup_before_frequencies_seconds"] = fw.get("setup_before_frequencies_seconds")
        out["fullwave_solver_setup_seconds"] = sum(r.get("solver_setup_seconds", 0.0) for r in fw.get("sweep", []))
        out["fullwave_port_solves_seconds"] = sum(r.get("port_solves_seconds", 0.0) for r in fw.get("sweep", []))
    out["mesh_seconds"] = ((d.get("mesh") or {}).get("info") or {}).get("seconds")
    return out


def gpu_trace(run_dir: Path) -> dict | None:
    path = run_dir / "gpu_trace.csv"
    if not path.exists():
        return None
    util = []
    mem = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                util.append(float(parts[1]))
                mem.append(float(parts[2]))
            except ValueError:
                continue
    if not util:
        return None
    return {"samples": len(util), "utilization_mean_percent": float(np.mean(util)),
            "utilization_busy_fraction": float(np.mean(np.array(util) > 0)), "memory_peak_MiB": float(max(mem))}


def load_runs() -> dict:
    runs = {}
    for case in CASES:
        for run_file in sorted((RUNS / case).glob("*/run.json")):
            run_dir = run_file.parent
            entry = {"run": json.loads(run_file.read_text()), "dir": run_dir}
            res = run_dir / "results.json"
            entry["results"] = json.loads(res.read_text()) if res.exists() else None
            smp = run_dir / "samples.json"
            entry["samples"] = json.loads(smp.read_text()) if smp.exists() else None
            entry["mesh_sha256"] = sha256(run_dir / "mesh3d.msh")
            runs.setdefault(case, {})[run_dir.name] = entry
    return runs


def kept_provenance() -> dict | None:
    """The committed record's provenance, when no run of the default run directory is newer than the record: a report
    that only adds notes or reformats then keeps the commit, executables and date the results came from instead of
    the tree and builds of report time."""
    if RUNS != (ROOT / "runs" / "gpu_validation").resolve() or not OUT.exists():
        return None
    newest = max((f.stat().st_mtime for f in RUNS.glob("*/*/run.json")), default=None)
    if newest is None or newest > OUT.stat().st_mtime:
        return None
    kept = json.loads(OUT.read_text()).get("provenance")
    if kept:
        print("no run is newer than", OUT.name, "- its provenance is kept", flush=True)
    return kept


def report() -> None:
    runs = load_runs()
    rec = {"check": "gpu_validation", "provenance": kept_provenance() or provenance(runs), "notes": RECORD_NOTES,
           "cases": {c: {"description": CASES[c].get("description") or (CASES[c]["design"].__doc__ or "").strip()
                         or None, "solver_env": CASES[c].get("env")} for c in CASES},
           "results": {}}
    for case, by_name in runs.items():
        indicator = bool(CASES[case].get("indicator"))
        ref_name = next((n for n in REFERENCES if n in by_name), None)
        ref = by_name.get(ref_name) if ref_name else None
        case_rec = {"reference": ref_name}
        for name, entry in sorted(by_name.items()):
            run = entry["run"]
            d = entry["results"]
            item = {"run": {k: run.get(k) for k in ("device", "ranks", "threads", "returncode", "cpu", "gpu",
                                                     "date", "exe", "exe_sha256", "libaxionfem_sha256",
                                                     "solver_env", "renamed_from", "note")},
                    "build": Path(run.get("exe", "")).parent.name,
                    "mesh_sha256": entry["mesh_sha256"], "ok": d is not None and run.get("returncode") == 0}
            if "openblas_coretype" in run:   # pinned (OPENBLAS_CORETYPE above); earlier runs: unset
                item["run"]["openblas_coretype"] = run["openblas_coretype"]
            if d is None:
                log = entry["dir"] / "solve.log"
                item["log_tail"] = log.read_text().splitlines()[-25:] if log.exists() else []
                case_rec[name] = item
                print(f"{case} {name}: FAILED (exit {run.get('returncode')})", flush=True)
                continue
            item["dofs"] = (d.get("mesh") or {}).get("dofs")
            item["warnings"] = d.get("warnings", [])
            item["iterations"] = iterations(d, indicator)
            item["timings"] = timings(d, run)
            item["gpu_trace"] = gpu_trace(entry["dir"])
            if ref is not None and ref["results"] is not None and name != ref_name:
                item["same_mesh_as_reference"] = entry["mesh_sha256"] == ref["mesh_sha256"]
                # the OpenBLAS kernels (OPENBLAS_CORETYPE): unpinned runs record none. A run pinned unlike its
                # reference compares two kernel families, and the order-2 indicator moves with them
                kernels = (run.get("openblas_coretype"), ref["run"].get("openblas_coretype"))
                if kernels[0] != kernels[1]:
                    item["openblas_coretype_mismatch"] = {"run": kernels[0], "reference": kernels[1]}
                    print(f"{case} {name}: OPENBLAS_CORETYPE {kernels[0]} against the reference's {kernels[1]}: "
                          "rerun the reference with the same pin to compare like with like", flush=True)
                mine = key_values(d, entry["samples"], indicator)
                theirs = key_values(ref["results"], ref["samples"], indicator)
                item["vs_reference"] = {k: rel(mine[k], theirs.get(k)) for k in mine
                                        if mine[k] is not None and theirs.get(k) is not None}
                item["key_values_missing_in_run"] = [k for k in theirs if theirs[k] is not None and mine.get(k) is None]
                item["census_vs_reference"] = census(d, ref["results"])
                item["bit_identity_vs_reference"] = bit_identity(d, ref["results"])
                worst = max((v for v in item["vs_reference"].values() if v is not None), default=0.0)
                print(f"{case} {name}: same mesh {item['same_mesh_as_reference']}, key quantities max rel diff "
                      f"{worst:.2e}, census max {item['census_vs_reference']['max_relative_difference']:.2e}, "
                      f"bit-identical {item['bit_identity_vs_reference']['identical']}, "
                      f"key values missing {item['key_values_missing_in_run']}, "
                      f"wall {run.get('wall_seconds') or 0.0:.1f} s, iterations {item['iterations']}", flush=True)
            else:
                print(f"{case} {name}: wall {run.get('wall_seconds') or 0.0:.1f} s, iterations {item['iterations']}", flush=True)
            first = by_name.get(name[:-len("_repeat")]) if name.endswith("_repeat") else None
            if first is not None and first["results"] is not None:   # the same configuration run again
                c = census(d, first["results"])
                c["bit_identity"] = bit_identity(d, first["results"])
                c["bit_identical_numbers"] = c["bit_identity"]["identical"]
                item["census_vs_first_run"] = c
                print(f"{case} {name}: against the first run, bit-identical {c['bit_identical_numbers']}, max "
                      f"{c['max_difference_relative_to_key_scale']:.2e}", flush=True)
            case_rec[name] = item
        rec["results"][case] = case_rec
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    if RUNS == (ROOT / "runs" / "gpu_validation").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_GPUVAL_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        def opt(flag, default=None):
            return a[a.index(flag) + 1] if flag in a else default
        name = opt("--name")
        exe = opt("--exe")
        deps = opt("--deps")
        if not (name and exe and deps):
            sys.exit("run needs --name, --exe and --deps")
        cases = opt("--cases", ",".join(DEFAULT_CASES)).split(",")
        for case in cases:
            run_case(case, name, exe, deps, opt("--device", "cpu"), int(opt("--ranks", "1")), int(opt("--threads", "1")))
        return
    report()


if __name__ == "__main__":
    main()
