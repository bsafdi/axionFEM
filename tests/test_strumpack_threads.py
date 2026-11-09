"""STRUMPACK called from more than one thread of a process.

STRUMPACK 8.0.0 appends a timer to a process-wide std::list per OpenMP thread number after every reorder, factor
and solve. The vector of lists is sized once, to omp_get_max_threads() when the library is loaded, and is neither
locked nor emptied. Before the guard in cpp/src/sym_solver.cpp, `Results.torsion(solver={"backend": "strumpack",
"parallel_frequencies": 2})` with OMP_NUM_THREADS=1 at load wrote past that vector from OpenMP thread 1 and the
process died with SIGSEGV (at the latest in its exit handlers); concurrent poloidal solves from Python threads
raced on list 0. Each case runs in a child process, so a crash fails the test instead of killing pytest; the
guard's counters (`_core.strumpack_guard_stats`) are checked after the exit status. cpp/tests/test_strumpack_threads
checks the same guard from C++.

The guard has two modes (`build_info()["strumpack_guard"]`): "concurrent" (STRUMPACK 8.0.x: frequency threads
factorize at the same time, each with its own timer list) and "serialized" (any other STRUMPACK version, or a build
configured with -DAXF_STRUMPACK_SERIALIZE=ON: one STRUMPACK call at a time, frequency-parallel STRUMPACK sweeps run
serially, STRUMPACK's internals are not read, so the timer counters are null). The expectations follow the mode.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import axionfem as af

pytestmark = pytest.mark.skipif(not af.build_info().get("strumpack"), reason="build without STRUMPACK")
MODE = af.build_info().get("strumpack_guard")
SERIALIZED = MODE == "serialized"
NOTICE = "every STRUMPACK call in this process runs alone"

TESTS = Path(__file__).resolve().parent
PKG = Path(af.__file__).resolve().parents[1]


def _run_child(code: str, tmp_path: Path, omp_threads: int, extra_env: dict | None = None) -> str:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG), str(TESTS)])
    env["TMPDIR"] = str(tmp_path)
    env.update(extra_env or {})
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(code))
    p = subprocess.run([sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, text=True,
                       timeout=600)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"child exited with {p.returncode} (negative: killed by that signal)\n{out[-3000:]}"
    assert "CHILD OK" in p.stdout, out[-3000:]
    # the serialized guard's one-time warning (C++ stderr): once per process, and only in that mode
    assert out.count(NOTICE) == (1 if SERIALIZED else 0), out[-3000:]
    return p.stdout


def _tagged(out: str, tag: str):
    return json.loads(next(line.split(" ", 1)[1] for line in out.splitlines() if line.startswith(tag + " ")))


_STATS = """
        def stats():
            f = getattr(_core, "strumpack_guard_stats", None)
            return json.loads(f(True)) if f else None
"""


def test_frequency_parallel_strumpack_with_one_thread_at_load(tmp_path):
    """One OpenMP thread at load, two concurrent frequency threads: the solve must not crash."""
    out = _run_child("""
        import json
        import sys
        from pathlib import Path

        from axionfem import _core
        from test_torsion_highfreq import _tube_design
        """ + _STATS + """
        res = _tube_design(0.5, 0.01, 0.1, 0.002, curve=True).solve(Path(sys.argv[1]) / "tube", verbose=0)
        loads = [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": 1.0}]
        freqs = [1.1e5, 2.2e5, 3.3e5, 4.4e5]
        ref = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs, solver={"backend": "eigen"})["harmonic"]
        stats()
        if hasattr(_core, "strumpack_guard_test_hold"):
            _core.strumpack_guard_test_hold(30)        # each call holds its locks 30 ms: the two threads overlap
        for k in range(3):
            h = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs,
                            solver={"backend": "strumpack", "parallel_frequencies": 2})["harmonic"]
            for x, y in zip(h["response"], ref["response"]):
                assert abs(x["Y_re"] / y["Y_re"] - 1) < 1e-9 and abs(x["Y_im"] / y["Y_im"] - 1) < 1e-9
        print("TIMING", json.dumps(h["timing"]))
        print("STATS", json.dumps(stats()))
        print("CHILD OK", flush=True)
    """, tmp_path, omp_threads=1)
    # checked here, after the exit status: an unprotected path makes the child die with SIGSEGV (in its exit handlers)
    timing = _tagged(out, "TIMING")
    st = _tagged(out, "STATS")
    assert st is not None, "this build has no strumpack_guard_stats"
    assert st["mode"] == MODE, (st, MODE)
    assert st["max_same_thread_number"] == 1, st
    assert st["max_concurrent_during_life"] == 1, st   # solvers built and destroyed alone
    if SERIALIZED:   # the sweep ran one frequency at a time, and says so
        assert timing["backend"] == "strumpack" and timing["parallel_frequencies"] == 1, timing
        assert timing["parallel_frequencies_requested"] == 2 and timing["strumpack_guard"] == "serialized", timing
        assert st["max_concurrent"] == 1, st
        assert st["timer_entries"] is None and st["timer_lists"] is None, st   # internals not read
    else:
        assert timing["backend"] == "strumpack" and timing["parallel_frequencies"] == 2, timing
        assert "parallel_frequencies_requested" not in timing, timing
        assert st["timer_entries"] == 0 and st["timer_lists"] >= 2, st
        assert st["max_concurrent"] == 2, st          # the frequency threads solved concurrently


def test_concurrent_poloidal_strumpack_solves_from_python_threads(tmp_path):
    """Design.solve releases the GIL: Python threads (all OpenMP thread number 0) factorize with STRUMPACK at the
    same time, and the guard must run their STRUMPACK calls one at a time (each call held 30 ms, so unguarded
    calls would overlap)."""
    out = _run_child("""
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from axionfem import _core
        from test_torsion_highfreq import _tube_design
        """ + _STATS + """
        root = Path(sys.argv[1])

        def flux(k):
            res = _tube_design(0.5, 0.01, 0.1, 0.004).solve(root / f"s{k}", verbose=0)
            s = res.data["static"]["poloidal"]
            assert s["solver"]["backend"] == "strumpack", s["solver"]
            return s["energy_J"]

        ref = flux(-1)
        stats()
        if hasattr(_core, "strumpack_guard_test_hold"):
            _core.strumpack_guard_test_hold(30)
        with ThreadPoolExecutor(3) as ex:
            vals = list(ex.map(flux, range(9)))
        assert all(abs(v / ref - 1) < 1e-10 for v in vals), (ref, vals)
        print("STATS", json.dumps(stats()))
        print("CHILD OK", flush=True)
    """, tmp_path, omp_threads=2, extra_env={"AXF_STRUMPACK_MIN_DOFS": "0"})
    st = _tagged(out, "STATS")
    assert st is not None, "this build has no strumpack_guard_stats"
    assert st["max_same_thread_number"] == 1, st
    if SERIALIZED:
        assert st["max_concurrent"] == 1 and st["timer_entries"] is None, st
    else:
        assert st["timer_entries"] == 0, st


def test_fluids_lu_goes_through_the_guard(tmp_path):
    """The fluids SparseLU (STRUMPACK by default in a STRUMPACK build) goes through the same guard as the EM solvers:
    alone, its calls are counted and leave no timer entries; on one Python thread while EM factorizations run on
    another (both are OpenMP thread number 0, each call held 30 ms), the calls run one at a time and both results are
    right."""
    pytest.importorskip("axionfem.fluids")
    out = _run_child("""
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import numpy as np

        from axionfem import _core, fluids
        from test_fluids_api import channel_mesh
        from test_torsion_highfreq import _tube_design
        """ + _STATS + """
        root = Path(sys.argv[1])
        mesh = channel_mesh(nx=8, ny=4)
        prof = lambda x, y: (4 * y * (1 - y) if x < 1e-12 else 0.0, 0.0)
        bcs = [{"attribute": 1, "kind": "dirichlet", "value": prof}, {"attribute": 2, "kind": "do_nothing"},
               {"attribute": 3, "kind": "dirichlet"}]
        el = np.arange(mesh.n_elements, dtype=np.int32)
        ref_pts = np.array([[0.2, 0.3], [0.6, 0.1]])

        def poiseuille():                    # exact in Taylor-Hood: velocity 4 y (1 - y), pressure 8 nu (2 - x)
            sol = fluids.solve_stokes(mesh, 0.5, None, bcs, form="laplacian")
            d = sol.diagnostics
            assert d["lu_backend"] == "strumpack" and d["converged"], d
            x, _ = sol.geometry(el, ref_pts)
            u, _ = sol.velocity(el, ref_pts)
            assert np.allclose(u[..., 0], 4 * x[..., 1] * (1 - x[..., 1]), atol=1e-10)
            assert np.allclose(sol.pressure(el, ref_pts), 4 * (2.0 - x[..., 0]), atol=1e-9)

        def flux(k):
            res = _tube_design(0.5, 0.01, 0.1, 0.004).solve(root / f"s{k}", verbose=0)
            s = res.data["static"]["poloidal"]
            assert s["solver"]["backend"] == "strumpack", s["solver"]
            return s["energy_J"]

        stats()
        for _ in range(2):
            poiseuille()
        print("ALONE", json.dumps(stats()))
        ref = flux(-1)
        stats()
        if hasattr(_core, "strumpack_guard_test_hold"):
            _core.strumpack_guard_test_hold(30)
        with ThreadPoolExecutor(1) as ex:
            em = ex.submit(lambda: [flux(k) for k in range(3)])
            for _ in range(3):
                poiseuille()
            vals = em.result()
        assert all(abs(v / ref - 1) < 1e-10 for v in vals), (ref, vals)
        print("MIXED", json.dumps(stats()))
        print("CHILD OK", flush=True)
    """, tmp_path, omp_threads=2, extra_env={"AXF_STRUMPACK_MIN_DOFS": "0"})
    alone = _tagged(out, "ALONE")
    assert alone is not None, "this build has no strumpack_guard_stats"
    assert alone["max_concurrent"] == 1, alone           # 0 when its calls were not counted
    assert alone["max_concurrent_during_life"] == 1, alone
    mixed = _tagged(out, "MIXED")
    assert mixed["max_same_thread_number"] == 1, mixed
    if SERIALIZED:
        assert mixed["max_concurrent"] == 1 and alone["timer_entries"] is None, (alone, mixed)
    else:
        assert alone["timer_entries"] == 0, alone        # 8 per solve when the fluids LU bypassed the guard
        assert mixed["timer_entries"] == 0, mixed


def test_build_info_reports_the_guard_mode():
    """The concurrent guard reads STRUMPACK 8.0 internals: any other version must get the serialized one (an 8.0
    build may be serialized too, with -DAXF_STRUMPACK_SERIALIZE=ON)."""
    import json
    from axionfem import _core
    info = af.build_info()
    assert MODE in ("concurrent", "serialized"), info
    if not info["strumpack_version"].startswith("8.0."):
        assert MODE == "serialized", info
    st = json.loads(_core.strumpack_guard_stats(False))
    assert st["mode"] == MODE, st
    assert (st["timer_lists"] is None) == SERIALIZED, st
    assert (_core.strumpack_timer_entries() is None) == SERIALIZED
