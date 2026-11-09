"""MCP server: progress reporting and cancellation of a running solve, and the guide's description of them.

Each test drives axionfem.mcp_server over stdio in a subprocess, as an MCP client does.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")   # the `mcp` extra (the server module imports it)

import axionfem as af  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN_VENV = Path(os.environ.get("AXF_VENV_PYTHON", sys.executable))


def _mcp_env(ws: Path) -> dict:
    # the directory of the axionfem imported here (the source tree's python/, or an installed package's site-packages)
    env = {"AXF_WORKSPACE": str(ws), "AXF_THREADS": "2", "PYTHONPATH": str(Path(af.__file__).resolve().parents[1]),
           "PYTHONDONTWRITEBYTECODE": "1"}
    for k in ("LD_LIBRARY_PATH", "AXF_MESHER", "PATH", "HOME", "OMP_NUM_THREADS", "AXIONFEM_DEPS"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


# a design small enough to solve in a few seconds but slow enough to catch mid-flight
FAST_DESIGN = {"schema": "axionfem.design/1", "name": "mcpfast",
               "regions": [{"name": "coil", "material": "winding",
                            "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.2, 0.2]}}},
                           {"name": "shield", "material": "nb",
                            "shape": {"frame": {"r": [0.5, 1.5], "z": [-0.6, 0.6], "thickness": 0.02,
                                                "fillet": 0.05}}}],
               "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
               "superconductors": [{"body": "shield", "azimuthal": {"net_current": 0},
                                    "azimuthal_slit": True}],
               "pickup": {"type": "azimuthal", "conductor": "shield"},
               "mesh": {"order": 2, "size": 0.04, "surface_size": 0.01}}


def _run_mcp(coro_factory, ws: Path, timeout: float = 600.0):
    import asyncio

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=str(MAIN_VENV), args=["-m", "axionfem.mcp_server"],
                                   env=_mcp_env(ws))

    async def go():
        # errlog explicitly: stdio_client's default (sys.stderr) is bound when mcp.client.stdio is first imported,
        # which can be under another test's capsys (test_global_opt.py::test_mcp_worker_job); Popen then fails
        # with io.UnsupportedOperation: fileno
        async with stdio_client(params, errlog=sys.__stderr__) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await coro_factory(s)

    return asyncio.run(asyncio.wait_for(go(), timeout))


def _payload(res):
    return json.loads(res.content[0].text)


@pytest.fixture(scope="module")
def mcp_ws(tmp_path_factory):
    ws = tmp_path_factory.mktemp("mcpws")
    (ws / "designs").mkdir(parents=True, exist_ok=True)
    (ws / "designs" / "fast.json").write_text(json.dumps(FAST_DESIGN))
    return ws


def test_mcp_solve_reports_progress(mcp_ws):
    """A solve must stream progress instead of going silent."""
    seen: list[tuple[float, float | None, str | None]] = []

    async def body(s):
        async def on_progress(progress, total, message):
            seen.append((progress, total, message))

        res = await s.call_tool("solve_design", {"name": "fast", "scene": False},
                                progress_callback=on_progress)
        return _payload(res)

    out = _run_mcp(body, mcp_ws)
    assert out["ok"] is True, out
    # Psi is zero for this design (a poloidal lab field with an azimuthal readout is orthogonal);
    # what matters here is that a real solve came back through the worker
    assert out["coupling"]["W_mag_J"] > 0
    assert out["mesh"]["dofs"] > 1000
    assert len(seen) >= 4, seen
    values = [p for p, _, _ in seen]
    assert values == sorted(values), values           # progress never goes backwards
    assert values[-1] == pytest.approx(1.0)
    msgs = " | ".join(m or "" for _, _, m in seen)
    assert "mesh" in msgs and "solving" in msgs, msgs  # the engine's own stages come through
    # the compacted reply summarises the distributions instead of shipping every sample
    dists = out["forces"]["distributions"]
    assert dists and all(not isinstance(v, list) for d in dists.values() for v in d.values())
    assert "distributions_note" in out["forces"]


def test_mcp_cancel_job_stops_a_running_solve(mcp_ws):
    import asyncio

    async def body(s):
        solve = asyncio.create_task(s.call_tool(
            "solve_design", {"name": "fast", "order": 4, "scene": True}))
        job_id = None
        for _ in range(300):                      # wait for the worker to appear
            await asyncio.sleep(0.1)
            jobs = _payload(await s.call_tool("list_jobs", {}))["jobs"]
            if jobs:
                job_id = jobs[0]["job_id"]
                break
            if solve.done():
                break
        assert job_id, "the solve finished before it could be cancelled"
        cancelled = _payload(await s.call_tool("cancel_job", {"job_id": job_id}))
        out = _payload(await solve)
        # the session must still work afterwards
        still_there = _payload(await s.call_tool("list_jobs", {}))
        nosuch = _payload(await s.call_tool("cancel_job", {"job_id": "no-such-job"}))
        return cancelled, out, still_there, nosuch, job_id

    cancelled, out, still_there, nosuch, job_id = _run_mcp(body, mcp_ws)
    assert cancelled["ok"] is True and job_id in cancelled["cancelled"]
    assert out["ok"] is False and out["cancelled"] is True
    assert "cancel" in out["error"]
    assert still_there["jobs"] == []              # the registry is cleaned up
    assert nosuch["ok"] is False and "no running job" in nosuch["error"]


def test_mcp_client_cancellation_kills_the_worker(mcp_ws):
    """A client that cancels the request (here by a read timeout, which makes the library send
    notifications/cancelled) must leave no worker behind and no broken session."""
    import asyncio

    async def body(s):
        try:
            await s.call_tool("solve_design", {"name": "fast", "order": 4, "scene": True},
                              read_timeout_seconds=2.0)
            timed_out = False
        except Exception:  # noqa: BLE001  (MCPError REQUEST_TIMEOUT)
            timed_out = True
        # the handler's finally clause kills the worker and clears the registry
        for _ in range(100):
            await asyncio.sleep(0.1)
            jobs = _payload(await s.call_tool("list_jobs", {}))["jobs"]
            if not jobs:
                break
        return timed_out, jobs, _payload(await s.call_tool("list_jobs", {}))

    timed_out, jobs, still = _run_mcp(body, mcp_ws)
    assert timed_out, "the solve finished inside the timeout; make the design slower"
    assert jobs == [], jobs
    assert still["jobs"] == []


def test_the_plot_helper_named_in_the_mcp_reply_exists():
    """The compacted MCP reply tells the agent which function to call; it must be the real one."""
    from axionfem import _mcp_worker, mcp_server, plots

    assert hasattr(plots, "force_distribution")
    for text in (mcp_server.GUIDE, _mcp_worker.compact.__doc__ or "",
                 Path(_mcp_worker.__file__).read_text()):
        assert "plot_force_distribution" not in text
    assert "axionfem.plots.force_distribution" in mcp_server.GUIDE


def test_mcp_guide_documents_progress_and_cancellation():
    from axionfem import mcp_server

    guide = mcp_server.GUIDE
    for word in ("progress", "cancel_job", "list_jobs", "worker subprocess",
                 "Fz_excluding_self_N", "forces.closure", "forces.distributions",
                 "domain_center_z", "surface_size", "fillets"):
        assert word in guide, word
    names = {t for t in dir(mcp_server) if not t.startswith("_")}
    assert {"cancel_job", "list_jobs", "solve_design", "sweep", "optimize"} <= names
