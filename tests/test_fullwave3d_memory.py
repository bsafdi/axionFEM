"""Memory across a 3D full-wave sweep (docs/validation/fullwave_memory.md, benchmarks/fullwave_memory.py).

Every frequency row carries a `memory` record: the resident set (VmRSS, VmHWM) and glibc's heap, held by the program
and kept free, at the start and end of the frequency, and rank 0's right after a direct factorization. On the direct
paths the factor dominates memory, and each frequency builds a new one. glibc kept a freed factor's memory resident: in
its bins, and in the top chunks of the OpenMP threads' arenas, more of it the more threads per rank. axionFEM hands it
back (AXF_FW_MALLOC_TRIM=factor and AXF_FW_MALLOC_THRESHOLDS=fixed, the defaults).

On a short sweep of the plain path (the two-port frame, a 415 MB factor) and of the gauged path (a coarse slit
toroid, a 347 MB factor), each at 8 and 32 OpenMP threads per rank, two gates on rank 0's resident
memory that the program does not hold (VmRSS minus glibc's heap held), in factors:
- released factors: at the later frequency starts, against the last start before any factor was freed. That is
  frequency 1 on the plain path, which frees its factor at the end of every frequency, and frequency 2 on the gauged
  path, which keeps its factor (held) until it builds the next one; so on the gauged path this gate sees growth from the
  second frequency on, not one factorization's work memory;
- the factorization's work memory: right after every factorization (after its trim), against the start of
  frequency 1. This one sees whether the work memory goes back right after the factorization, on both paths.
Each gate is < 0.3 factor, and each holds only where a negative control, the same sweep with glibc's behaviour
(AXF_FW_MALLOC_TRIM=off, AXF_FW_MALLOC_THRESHOLDS=glibc), shows at least 0.4 factor; where neither does (another
allocator, or one that returns memory itself) the gates cannot tell and the test skips. The control's own record must
show glibc's settings, so that a broken opt-out fails instead of skipping. No leak: the heap the program holds at the
frequency starts does not grow, up to the sweep rows.
Calibration: in the module's constant block (GATE, CONTROL_MINIMUM).
Every sweep runs in a fresh process (MPI ranks, or the axf executable in a subprocess), so that memory freed by an
earlier sweep or test cannot hide the retention, with the thread count pinned through OMP_NUM_THREADS (not through
`threads=`, which would also set the pytest process's own OpenMP threads). glibc only: other allocators report no heap
and the test skips.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

import axionfem as af
from axionfem._paths import axf_path

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

TWO_PORT = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"
PATHS = {
    "strumpack": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_TC_CHARGE_PROJECTION": "0",
                  "AXF_FW_STRUMPACK_DIST": "0", "AXF_FW_GAUSS_GUARD": "correct"},
    "tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "2",
                    "AXF_FW_STRUMPACK_DIST": "0", "AXF_FW_GAUSS_GUARD": "off"},
}
SETTINGS = {"default": {}, "glibc": {"AXF_FW_MALLOC_TRIM": "off", "AXF_FW_MALLOC_THRESHOLDS": "glibc"}}
# Both gates in factors. Calibration (2 ranks, 8 and 32 threads, with and without MPI): the default 0.015-0.190
# (released) and -0.40 to 0.11 (work); glibc's behaviour 0.435-3.64 and 0.61-3.51; the fixed thresholds with every trim
# off (a lost trim) 0.545-1.73 and 0.75-1.32. The gate sits near the geometric mean of the largest default and the
# smallest failure (0.29); the control must clear it by a margin.
GATE = 0.3
CONTROL_MINIMUM = 0.4
# the last frequency start before any factor was freed (see the module docstring)
REFERENCE = {"strumpack": 0, "tree_cotree": 1}


def _doc(path):
    if path == "strumpack":   # the two-port frame: 109k unknowns, a 415 MB plain factor
        doc = json.loads(TWO_PORT.read_text())
        for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
            doc["fullwave"].pop(key, None)
        doc["fullwave"]["frequencies"] = [1e5, 2e5, 4e5, 8e5, 1.6e6, 3.2e6]
        return doc
    from axionfem.convert import to_3d
    d2 = af.template("toroid", R=1.0, y=0.5, H=0.5, tw=0.02, g=0.01, ts=0.025, f=0.03, NI=2.5e7,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.01})
    d3 = to_3d(d2, pickup="slit", slit_deg=2.0, mesh={"order": 1, "size": 0.3, "surface_size": 0.08})
    d3.doc["fullwave"] = {"frequencies": [1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4], "boundary": "pec"}
    return d3.doc


def _sweep(wd, path, setting, threads, monkeypatch):
    """One sweep in a fresh process with `threads` OpenMP threads per rank; its fullwave results."""
    for key in ("AXF_FW_MALLOC_TRIM", "AXF_FW_MALLOC_THRESHOLDS", "AXF_FW_FREQUENCY_PEAK", "MALLOC_ARENA_MAX",
                "MALLOC_TRIM_THRESHOLD_", "MALLOC_MMAP_THRESHOLD_", "GLIBC_TUNABLES", "LD_PRELOAD"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {**PATHS[path], **SETTINGS[setting]}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))   # the ranks' threads; the pytest process's stay as they are
    doc = _doc(path)
    if os.environ.get("AXF_MPI"):   # srun or mpirun starts fresh rank processes, with this environment
        return af.Design(doc).solve(wd, verbose=0, mpi_ranks=2).data["fullwave"]
    exe = axf_path()
    if exe is None:
        pytest.skip("needs the axf executable (AXF_EXE) to run a sweep in a fresh process")
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "design.json").write_text(json.dumps(doc))
    run = subprocess.run([exe, "solve", str(wd / "design.json"), "-o", str(wd), "-v", "0"], env=dict(os.environ),
                         capture_output=True, text=True, timeout=3600)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads((wd / "results.json").read_text())["fullwave"]


def _measures(fw, path):
    rows = fw["sweep"]
    starts = [r["memory"]["frequency_start"] for r in rows]
    if not all(s["heap_held_rank0_bytes"] > 0 for s in starts):
        pytest.skip("no glibc heap figures (another allocator)")
    factor = rows[0]["direct_factorization"]
    factor_bytes = 8.0 * (factor.get("factor_nonzeros") or factor["factor"]["factor_nonzeros"])
    loose = [s["rss_rank0_bytes"] - s["heap_held_rank0_bytes"] for s in starts]
    after = [r["memory"]["after_factorization_rank0"] for r in rows]
    loose_after = [a["rss_bytes"] - a["heap_held_bytes"] for a in after]
    held = [s["heap_held_sum_bytes"] for s in starts[1:]]
    ref = REFERENCE[path]
    return {"factor": factor_bytes,
            "released": (max(loose[ref + 1:]) - loose[ref]) / factor_bytes,
            "work": (max(loose_after) - loose[0]) / factor_bytes,
            "loose_MB": [round(x / 1e6) for x in loose], "loose_after_MB": [round(x / 1e6) for x in loose_after],
            "held_growth": held[-1] - held[0], "frequencies": len(rows)}


@pytest.mark.parametrize("threads", [8, 32])
@pytest.mark.parametrize("path", list(PATHS))
def test_a_released_factor_goes_back_to_the_system(tmp_path, monkeypatch, path, threads):
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    fixed = _sweep(tmp_path / "default", path, "default", threads, monkeypatch)
    assert fixed["memory"]["malloc_trim"] == "factor"
    assert fixed["memory"]["malloc_thresholds_mode"] == "fixed"
    assert fixed["memory"]["malloc_thresholds_applied_ranks"] >= 1
    fix = _measures(fixed, path)
    glibc = _sweep(tmp_path / "glibc", path, "glibc", threads, monkeypatch)
    # the control's own record: glibc's settings, or a broken opt-out would turn the test into a skip
    assert glibc["memory"]["malloc_trim"] == "off"
    assert glibc["memory"]["malloc_thresholds_mode"] == "glibc"
    assert glibc["memory"]["malloc_thresholds_applied_ranks"] == 0
    control = _measures(glibc, path)
    print(f"{path}, {threads} threads per rank: factor {fix['factor'] / 1e6:.0f} MB; resident not held (rank 0, MB) at "
          f"the frequency starts: fix {fix['loose_MB']}, glibc {control['loose_MB']}; after the factorizations: fix "
          f"{fix['loose_after_MB']}, glibc {control['loose_after_MB']}; released {fix['released']:.3f} against "
          f"{control['released']:.3f}, work {fix['work']:.3f} against {control['work']:.3f} factors")
    # no leak, either way: the sweep rows add well under 1 MB per frequency; a leak of 1 % of the factor per frequency
    # would add about 4 MB (plain) or 3 MB (gauged) per frequency
    for name, measure in (("fix", fix), ("glibc", control)):
        limit = 1e6 * (measure["frequencies"] - 2) + 0.002 * measure["factor"]
        assert measure["held_growth"] < limit, (name, path, threads, measure)
    gated = [gate for gate in ("released", "work") if control[gate] >= CONTROL_MINIMUM]
    if not gated:
        pytest.skip(f"glibc's behaviour kept only {control['released']:.2f} and {control['work']:.2f} factors resident "
                    f"here: the gates cannot tell")
    for gate in gated:
        assert fix[gate] < GATE, (gate, path, threads, fix, control)


def _two_port_one_frequency():
    doc = json.loads(TWO_PORT.read_text())
    for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
        doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [1e6]
    return doc


@pytest.mark.parametrize("key, value, message", [
    ("AXF_FW_MALLOC_TRIM", "1", "AXF_FW_MALLOC_TRIM must be factor, frequency or off"),
    ("AXF_FW_MALLOC_THRESHOLDS", "on", "AXF_FW_MALLOC_THRESHOLDS must be fixed or glibc"),
    ("AXF_FW_FREQUENCY_PEAK", "yes", "AXF_FW_FREQUENCY_PEAK must be 0 or 1"),
])
def test_bad_allocator_options_are_refused(tmp_path, monkeypatch, key, value, message):
    monkeypatch.setenv(key, value)
    with pytest.raises(Exception, match=message):
        af.Design(_two_port_one_frequency()).solve(tmp_path / key, verbose=0,
                                                 mpi_ranks=2 if os.environ.get("AXF_MPI") else 1)


def test_a_rejected_frequency_keeps_its_memory_record(tmp_path, monkeypatch):
    """A frequency refused by a validation gate (here: a two-port solve capped at one GMRES iteration) is written to
    fullwave_rejected.json with its memory record, where the memory picture matters for a diagnosis."""
    for key, value in PATHS["strumpack"].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AXF_FW_SOLVER", "ams")
    monkeypatch.setenv("AXF_FW_MAX_ITER", "1")
    monkeypatch.setenv("AXF_FW_FREQUENCY_PEAK", "1")
    wd = tmp_path / "rejected"
    with pytest.raises(Exception, match="did not converge"):
        af.Design(_two_port_one_frequency()).solve(wd, verbose=0, mpi_ranks=2 if os.environ.get("AXF_MPI") else 1)
    diagnostic = json.loads((wd / "fullwave_rejected.json").read_text())
    row = diagnostic["primary_row"]
    assert row["status"] == "provisional_rejected"
    memory = row["memory"]
    assert memory["peak_scope"].startswith("this frequency")
    for where in ("frequency_start", "frequency_end"):
        assert memory[where]["rss_rank0_bytes"] > 0
        assert memory[where]["peak_rss_rank0_bytes"] >= memory[where]["rss_rank0_bytes"]
    assert diagnostic["fullwave"]["memory"]["malloc_trim"] in ("factor", "unavailable (not glibc)")


def test_the_record_says_which_thresholds_were_in_force(tmp_path, monkeypatch):
    """In one process, as a Python session runs them: a direct sweep sets the
    thresholds for itself and relaxes them when it ends; a shift-invert eigenmode run records them too; a later run
    with the opt-out says what an earlier analysis left in force instead of claiming glibc's dynamic thresholds."""
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    for key in ("AXF_FW_MALLOC_TRIM", "AXF_FW_MALLOC_THRESHOLDS", "MALLOC_TRIM_THRESHOLD_", "MALLOC_MMAP_THRESHOLD_",
                "GLIBC_TUNABLES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in PATHS["strumpack"].items():
        monkeypatch.setenv(key, value)
    direct = af.Design(_two_port_one_frequency()).solve(tmp_path / "direct", verbose=0).data["fullwave"]["memory"]
    if direct["malloc_trim"] == "unavailable (not glibc)":
        pytest.skip("not glibc's malloc")
    assert direct["malloc_thresholds_mode"] == "fixed"
    assert direct["malloc_thresholds_applied_ranks"] == 1
    assert "after it" in direct["malloc_thresholds"]
    eigen_doc = _two_port_one_frequency()
    eigen_doc["fullwave"]["frequencies"] = []
    eigen_doc["fullwave"]["modes"] = 2
    eigen_doc["fullwave"]["target_f_Hz"] = 2.0e8
    eigen = af.Design(eigen_doc).solve(tmp_path / "eigen", verbose=0).data["fullwave"]
    assert eigen["memory"]["malloc_thresholds_applied_ranks"] == 1
    assert "after_sweep" not in eigen["memory"]
    monkeypatch.setenv("AXF_FW_MALLOC_THRESHOLDS", "glibc")
    monkeypatch.setenv("AXF_FW_SOLVER", "ams")
    optout = af.Design(_two_port_one_frequency()).solve(tmp_path / "optout", verbose=0).data["fullwave"]["memory"]
    assert optout["malloc_thresholds_mode"] == "glibc"
    assert optout["malloc_thresholds_applied_ranks"] == 0
    assert "left by an earlier full-wave analysis" in optout["malloc_thresholds_before"]
    assert optout["malloc_thresholds"].startswith("not set by this analysis")
