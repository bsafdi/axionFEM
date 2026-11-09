#!/usr/bin/env python3
"""Run the chemistry verification cases (beta): benchmarks/fluids/run.py with --track chem as the default.

    python benchmarks/chem/run.py list
    python benchmarks/chem/run.py run --all --cost short
    python benchmarks/chem/run.py run chem_kernel_rates --out results/
    python benchmarks/chem/run.py report results/

See benchmarks/chem/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fluids"))
import run as _run  # noqa: E402  (benchmarks/fluids/run.py)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] in ("list", "run", "report") and "--track" not in argv:
        argv = argv[:1] + ["--track", "chem"] + argv[1:]
    sys.exit(_run.main(argv))
