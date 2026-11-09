"""Console script ``axf``: runs the C++ command line shipped in the wheel.

``cpp/apps/axf.cpp`` is installed as ``axionfem/bin/axf``.  This wrapper puts it
on ``PATH`` through the entry point in pyproject.toml and replaces the Python
process with it, so exit codes, signals and streams behave exactly as they do
when the binary is called directly.

``axf export-comsol ...`` is handled in Python (:mod:`axionfem.comsol`, docs/comsol_export.md)
before the C++ command line is looked for, and so is ``axf import-comsol ...``
(:mod:`axionfem.comsol_import`, docs/comsol_import.md), and ``axf export-touchstone results.json ...``
(:mod:`axionfem.touchstone`, docs/physics/touchstone.md).

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import os
import sys

from . import _paths


def axf_path() -> str | None:
    """The ``axf`` executable behind the console script: $AXF_BIN, else the search of :func:`_paths.axf_path`
    ($AXF_EXE, the installed package, a development build)."""
    return os.environ.get("AXF_BIN") or _paths.axf_path()


def main(argv: list[str] | None = None) -> int:
    args_in = list(argv) if argv is not None else sys.argv[1:]
    if args_in and args_in[0] == "export-comsol":
        from .comsol import main as export_main

        return export_main(args_in[1:])
    if args_in and args_in[0] == "import-comsol":
        from .comsol_import import main as import_main

        return import_main(args_in[1:])
    if args_in and args_in[0] == "export-touchstone":
        from .touchstone import main as touchstone_main

        return touchstone_main(args_in[1:])
    exe = axf_path()
    if exe is None:
        sys.stderr.write(
            "axf: the compiled command line was not found.\n"
            "  Expected axionfem/bin/axf inside the installed package, or build/axf in a\n"
            "  source checkout. Set $AXF_BIN to its path, or reinstall axionFEM\n"
            "  (see docs/install.md).\n"
        )
        return 127
    _paths.prepare_environment()  # loader path for the bundled libraries and gmsh
    args = [exe] + args_in
    os.execv(exe, args)  # does not return
