"""Generate the virtual pages of the documentation site (run by mkdocs-gen-files).

* ``examples/*.ipynb`` -- the example notebooks listed in SITE_NOTEBOOKS (rendered with their stored
                         outputs, not executed, by mkdocs-jupyter): one walkthrough per guide area, each
                         listed in the ``nav`` of mkdocs.yml. A listed notebook that does not exist is
                         reported as a warning, so ``mkdocs build --strict`` fails on it.

Links inside these pages are written for their real location in the
repository; scripts/docs/hooks.py rewrites them for the
site.  The mapping virtual page -> real file is ``REAL_SOURCE`` in hooks.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mkdocs.plugins.axionfem.gen_pages")

SITE_NOTEBOOKS = [
    "examples/quickstart.ipynb",
    "examples/lumped_element.ipynb",
    "examples/fullwave.ipynb",
    "examples/readout_squid.ipynb",
    "examples/cavity_haloscope.ipynb",
    "examples/srf_heterodyne.ipynb",
    "examples/mechanics_thermal.ipynb",
    "examples/reach_optimization.ipynb",
    "examples/fluids_beta.ipynb",
    "examples/chemistry_beta.ipynb",
]


def _notebooks() -> None:
    for rel in SITE_NOTEBOOKS:
        nb = ROOT / rel
        if not nb.is_file():
            log.warning("gen_pages: %s is listed in SITE_NOTEBOOKS but does not exist", rel)
            continue
        rel = nb.relative_to(ROOT).as_posix()      # examples/..., the same path in the site
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(nb.read_bytes())
        mkdocs_gen_files.set_edit_path(rel, f"../{rel}")


_notebooks()
