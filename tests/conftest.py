"""Session-wide guard: worker processes import the same axionfem package as the test session.

Several benchmark modules that tests load put the source tree ``<repo>/python`` at the front of ``sys.path``, so that
they also run from a plain checkout. After a non-editable ``pip install``, that directory holds the Python sources
but no compiled ``_core``. The test process itself has already imported the installed package, but child processes
started with the ``spawn`` method (the worker pool of ``axionfem.global_opt``) inherit ``sys.path`` and would import
the source tree and fail. Before every test, any other directory that provides an ``axionfem`` package is moved behind
the one the session imported. Editable installs and in-place builds (``PYTHONPATH=python``) import ``<repo>/python``
itself, so nothing moves for them.
"""
import os
import sys


def _keep_imported_axionfem_first():
    mod = sys.modules.get("axionfem")
    path = getattr(mod, "__file__", None) if mod is not None else None
    if not path:
        return
    home = os.path.realpath(os.path.dirname(os.path.dirname(path)))
    shadows = []
    for p in sys.path:
        try:
            real = os.path.realpath(p or os.getcwd())
        except (OSError, ValueError):
            continue
        if real != home and os.path.isfile(os.path.join(real, "axionfem", "__init__.py")):
            shadows.append(p)
    if shadows:
        sys.path[:] = [p for p in sys.path if p not in shadows] + shadows


def pytest_collection_finish(session):
    _keep_imported_axionfem_first()


def pytest_runtest_setup(item):
    _keep_imported_axionfem_first()
