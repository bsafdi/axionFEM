"""Execute the example notebooks in place and store them without run-specific metadata.

    python scripts/docs/execute_notebooks.py                       # every notebook of the documentation site
    python scripts/docs/execute_notebooks.py examples/srf_demo.ipynb [--timeout 3600] [--jobs 4]
    python scripts/docs/execute_notebooks.py --audit-only          # only list suspicious output lines

Each notebook runs with its own directory as the working directory, so its relative links and paths resolve as they do
on the site, and with the inline matplotlib backend (do not set MPLBACKEND=Agg: figures are then not stored). After a
notebook has run, the script removes what would make the stored copy specific to one machine or one run: interactive
widget views (outputs with the mime type application/vnd.jupyter.widget-view+json, and metadata.widgets) and the
per-cell execution timestamps that nbclient records. A notebook that raises is left unchanged on disk.

The script then audits the stored outputs and lists the lines that contain an absolute path, the user or host name,
a traceback or a wall-clock timing, for review before committing.

Requirements: `pip install nbclient nbformat ipykernel` in the environment that has axionfem (the kernel runs
`python3` from that environment). The 3D cells need a build with MPI (and STRUMPACK for the SRF 3D cross-check); on a
cluster, run this inside a batch job so that the notebooks' MPI solves can start their ranks. Set SOURCE_DATE_EPOCH
for reproducible dates in the SVG drawings.
"""
from __future__ import annotations

import argparse
import getpass
import re
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


def site_notebooks() -> list[Path]:
    """The notebooks listed in scripts/docs/gen_pages.py (SITE_NOTEBOOKS)."""
    text = (ROOT / "scripts" / "docs" / "gen_pages.py").read_text()
    block = text[text.index("SITE_NOTEBOOKS = ["):]
    block = block[:block.index("]")]
    return [ROOT / rel for rel in re.findall(r'"([^"]+\.ipynb)"', block)]


def strip(nb) -> None:
    """Remove widget views, widget state and execution timestamps (in place)."""
    nb.metadata.pop("widgets", None)
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code":
            cell.outputs = [o for o in cell.outputs if WIDGET_MIME not in o.get("data", {})]


def execute(path: Path, timeout: int) -> tuple[str, float, str]:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3", coalesce_streams=True,
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    except Exception as e:  # noqa: BLE001 - report and keep the stored copy
        return str(path), time.time() - t0, f"FAILED: {type(e).__name__}: {str(e)[-2000:]}"
    strip(nb)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return str(path), time.time() - t0, "ok"


def audit(path: Path) -> list[str]:
    """Output lines that look machine- or run-specific."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    names = {n for n in (getpass.getuser(), socket.gethostname().split(".")[0]) if n and len(n) > 2}
    pats = [("path", re.compile(r"(?<![\w.])(/home/|/Users/|/tmp/|/var/|/scratch|/mnt/|/opt/|/usr/|/proj|"
                                r"[A-Za-z]:\\)")),
            ("traceback", re.compile(r"Traceback|Error:")),
            ("timing", re.compile(r"\b\d+(\.\d+)?\s?(s|sec|seconds|ms|min|minutes)\b(?! per)"))]
    hits = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for o in cell.outputs:
            texts = []
            if "text" in o:
                texts.append(o["text"])
            for mime in ("text/plain", "text/html", "text/markdown"):
                if mime in o.get("data", {}):
                    texts.append(o["data"][mime])
            if o.get("output_type") == "error":
                texts.append("\n".join(o.get("traceback", [])))
            for t in texts:
                t = "".join(t) if isinstance(t, list) else t
                for line in t.splitlines():
                    kinds = [k for k, p in pats if p.search(line)] + [f"name {n}" for n in names if n in line]
                    if kinds:
                        hits.append(f"{path.name} cell {i} [{', '.join(kinds)}]: {line.strip()[:160]}")
        if WIDGET_MIME in str(cell.outputs):
            hits.append(f"{path.name} cell {i}: widget view left in the outputs")
    if "widgets" in nb.metadata:
        hits.append(f"{path.name}: metadata.widgets present")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebooks", nargs="*", type=Path, help="default: the notebooks of the documentation site")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    ap.add_argument("--jobs", type=int, default=1, help="notebooks executed at the same time")
    ap.add_argument("--audit-only", action="store_true", help="do not execute; only audit the stored outputs")
    a = ap.parse_args()
    paths = [p.resolve() for p in a.notebooks] or site_notebooks()
    failed = []
    if not a.audit_only:
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for name, seconds, status in ex.map(execute, paths, [a.timeout] * len(paths)):
                print(f"{Path(name).relative_to(ROOT)}: {status} ({seconds:.0f} s)", flush=True)
                if status != "ok":
                    failed.append(name)
    for p in paths:
        for h in audit(p):
            print("audit:", h)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
