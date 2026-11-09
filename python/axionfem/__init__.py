"""axionFEM: a finite-element package for designing and evaluating axion dark-matter detectors.

Quick start::

    import axionfem as af
    d = af.template("toroid", R=2.0, H=2.0)   # parametric design
    d.set(y=0.6)                             # edit parameters
    res = d.solve()                          # mesh + solve (C++)
    print(res.summary())                     # Psi, L_p, F/F_req, fields, cost metric
    res.export_html("toroid.html")           # interactive 3D viewer

The design format is documented in docs/design_format.md; ``af.schema()``
returns its JSON Schema and ``af.templates()`` lists parametric templates.
"""
from __future__ import annotations

import os as _os

from . import _paths  # noqa: F401  (locates the mesher before the core loads)
from ._core import AxionFEMError, version
from .design import Design, load, schema, template, templates
from .results import Results, Results3D, conductors, jc, load_solution, requirement
from . import circuit  # readout-circuit networks
from ._beta import BetaWarning
from .sensitivity import objectives
from .global_opt import global_optimize, optimize

__version__ = version()

__all__ = [
    "AxionFEMError",
    "BetaWarning",
    "Design",
    "Results",
    "Results3D",
    "build_info",
    "conductors",
    "import_comsol",
    "jc",
    "load",
    "load_solution",
    "objectives",
    "optimize",
    "global_optimize",
    "requirement",
    "schema",
    "set_threads",
    "show",
    "template",
    "templates",
    "version",
]


def build_info() -> dict:
    """What this build can do: MPI and the 3D engine, OpenMP, GPU backend, MFEM.

    The sequential build (also the configuration of the planned wheels) has no MPI, so
    ``build_info()["engine_3d"]`` is False there and 3D designs raise an error that says how to get a
    build with it. See docs/guides/install.md.
    """
    import json as _json

    from ._core import build_info as _bi

    return _json.loads(_bi())


def show(scene, **kw):
    """Display a scene (or a ``Results``) in a Jupyter notebook.

    ``Results.show()`` is the usual entry point; this helps with a saved
    ``scene.json``.  See :mod:`axionfem.widget`.
    """
    from .widget import show as _show

    return _show(scene, **kw)


def import_comsol(path, out=None, **kw):
    """Import a COMSOL ``.mph`` model as a :class:`Design` (beta; docs/comsol_import.md).

    See :func:`axionfem.comsol_import.import_comsol` for the options; the import report is attached as
    ``design.import_report``."""
    from .comsol_import import import_comsol as _imp

    return _imp(path, out, **kw)


def set_threads(n: int) -> None:
    """Number of OpenMP threads used by the C++ core, for the calling thread only: OpenMP keeps the count per OS
    thread, so Python worker threads keep the count of `OMP_NUM_THREADS` (or call this themselves)."""
    from ._core import set_threads as _st

    _st(int(n))


if "AXF_THREADS" in _os.environ:
    set_threads(int(_os.environ["AXF_THREADS"]))
