"""End-to-end MCP smoke test: start the server over stdio and drive it like an agent.

Covers the tool surface, progress notifications and cancellation (the automated versions of the
last two are in tests/test_mcp_progress.py).

    .venv/bin/python tests/mcp_smoke.py <workspace>
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main(ws: str):
    # the interpreter that has axionfem: the repo's own venv, or whatever is running this script
    # (a git worktree has no .venv of its own)
    python = Path(os.environ.get("AXF_VENV_PYTHON") or (ROOT / ".venv/bin/python"))
    if not python.exists():
        python = Path(sys.executable)
    params = StdioServerParameters(command=str(python), args=["-m", "axionfem.mcp_server"],
                                   env={"AXF_WORKSPACE": ws, "AXF_THREADS": "4", "PYTHONPATH": str(ROOT / "python"),
                                        "PYTHONDONTWRITEBYTECODE": "1",
                                        "LD_LIBRARY_PATH": os.environ.get(
                                            "LD_LIBRARY_PATH", str(ROOT / "third_party/install/lib")),
                                        **{k: os.environ[k] for k in ("AXF_MESHER", "PATH", "HOME")
                                           if k in os.environ}})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("tools:", names)
            res = await s.call_tool("create_design", {"template": "core", "name": "c1", "parameters": {"Rc": 1.2}})
            print("create:", json.loads(res.content[0].text)["ok"])
            seen = []

            async def on_progress(progress, total, message):
                seen.append((progress, message))

            res = await s.call_tool("solve_design", {"name": "c1"}, progress_callback=on_progress)
            out = json.loads(res.content[0].text)
            print("solve:", out["ok"], "g =", round(out["coupling"]["g_factor"], 4), "files:", sorted(out["files"]))
            print(f"progress: {len(seen)} notifications, last {seen[-1] if seen else None}")
            print("  stages:", " -> ".join(f"{p:.2f} {m}" for p, m in seen[:6]))

            # cancellation: start a slow solve, find it with list_jobs, stop it with cancel_job
            slow = asyncio.create_task(s.call_tool("solve_design", {"name": "c1", "order": 4}))
            job = None
            for _ in range(300):
                await asyncio.sleep(0.1)
                jobs = json.loads((await s.call_tool("list_jobs", {})).content[0].text)["jobs"]
                if jobs:
                    job = jobs[0]
                    break
                if slow.done():
                    break
            print("running job:", job)
            if job:
                print("cancel:", json.loads((await s.call_tool("cancel_job", {"job_id": job["job_id"]})).content[0].text))
            cancelled = json.loads((await slow).content[0].text)
            print("cancelled solve returned:", {k: cancelled.get(k) for k in ("ok", "cancelled", "error")})
            print("jobs after:", json.loads((await s.call_tool("list_jobs", {})).content[0].text))
            res = await s.call_tool("render_design", {"name": "c1", "view": "section"})
            kinds = [c.type for c in res.content]
            print("render content types:", kinds, "bytes:", len(res.content[0].data) if kinds and kinds[0] == "image" else 0)
            bad = await s.call_tool("set_parameters", {"name": "c1", "parameters": {"nope": 1}})
            print("bad edit:", bad.content[0].text[:120])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
