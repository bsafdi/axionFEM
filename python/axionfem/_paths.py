"""Locate helper executables and shared libraries for the C++ core.

Two things have to be found at import time:

* ``axf-mesher``, the separate program that meshes designs.  It links the GPL
  gmsh library and is therefore kept apart from everything else (docs/install.md, "Licensing of each wheel");
* the gmsh library itself.  The wheels do **not** ship it: ``axf-mesher`` is
  built against the headers and library of gmsh's own PyPI wheel
  (``pip install gmsh``, the same upstream 4.15 binaries as the SDK), and this
  module puts that directory on the loader path of the mesher child process.
  See docs/install.md, "Licensing of each wheel".

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IS_MAC = sys.platform == "darwin"
_LIB_PATH_VAR = "DYLD_LIBRARY_PATH" if _IS_MAC else "LD_LIBRARY_PATH"


def _candidates():
    yield _HERE / "bin" / "axf-mesher"                  # installed wheel
    yield _HERE.parent.parent / "build" / "axf-mesher"  # in-place development build


def axf_path() -> str | None:
    """The axf command-line executable (multi-rank 3D solves)."""
    if os.environ.get("AXF_EXE"):
        return os.environ["AXF_EXE"]
    for c in (_HERE / "bin" / "axf", _HERE.parent.parent / "build" / "axf"):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def mesher_path() -> str | None:
    """Path to ``axf-mesher``, or None when it is not installed."""
    if os.environ.get("AXF_MESHER"):
        return os.environ["AXF_MESHER"]
    for c in _candidates():
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _prefixes():
    """Directories that may hold a gmsh installation, most specific first."""
    env = os.environ.get("AXF_GMSH_ROOT")
    if env:
        yield Path(env)
    # The gmsh wheel installs its data files into the interpreter's data scheme:
    # <prefix>/lib/libgmsh.so.4.15, <prefix>/include/gmsh.h, <prefix>/bin/gmsh.
    for key in ("data", "platlib"):
        try:
            p = sysconfig.get_paths().get(key)
        except Exception:  # pragma: no cover - defensive
            p = None
        if p:
            yield Path(p)
    yield Path(sys.prefix)
    yield Path(sys.base_prefix)
    # Source checkouts: the pinned SDK from scripts/build_deps.sh.
    deps = os.environ.get("AXIONFEM_DEPS")
    if deps:
        yield Path(deps) / "gmsh-sdk"
    yield _HERE.parent.parent / "third_party" / "install" / "gmsh-sdk"


def gmsh_lib_dir() -> str | None:
    """Directory holding ``libgmsh`` (from the gmsh wheel or the SDK), or None."""
    patterns = ("libgmsh*.dylib",) if _IS_MAC else ("libgmsh.so*",)
    seen = set()
    for prefix in _prefixes():
        for sub in ("lib", "lib64", ""):
            d = (prefix / sub) if sub else prefix
            key = str(d)
            if key in seen:
                continue
            seen.add(key)
            if not d.is_dir():
                continue
            for pat in patterns:
                if any(d.glob(pat)):
                    return str(d)
    return None


def mesher_hint() -> str:
    """What to tell the user when meshing is not available."""
    return (
        "axf-mesher was not found. It is the separate program that runs gmsh "
        "(a GPL library, kept out of libaxionfem; see docs/install.md, \"Licensing of each wheel\").\n"
        "  * installed from a wheel: run `pip install gmsh` -- the wheel ships the mesher "
        "but not the GPL library it needs;\n"
        "  * source checkout: build the project (the `axf-mesher` target) and make sure "
        "gmsh is found, or set $AXF_MESHER to the executable.\n"
        "  See docs/install.md."
    )



def default_mpi_launcher() -> str:
    """The command that starts a multi-rank run: $AXF_MPI_LAUNCHER, else ``srun`` inside a SLURM job when srun is on
    PATH (it is not, e.g., inside a container that a SLURM job started), else ``mpirun``, else ``mpiexec``."""
    import shutil

    explicit = os.environ.get("AXF_MPI_LAUNCHER")
    if explicit:
        return explicit
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        return "srun"
    return "mpirun" if shutil.which("mpirun") or not shutil.which("mpiexec") else "mpiexec"


def prepare_environment() -> dict:
    """Put the bundled libraries and gmsh on the loader path for child processes.

    Changing ``LD_LIBRARY_PATH``/``DYLD_LIBRARY_PATH`` does not affect this
    process (the loader read it at exec time), but ``axf-mesher`` is a *child*
    process and inherits ``os.environ``, which is exactly what needs it.
    Returns a small report, which is useful when diagnosing an installation.
    """
    report: dict = {}
    extra = []
    libs = _HERE / "_libs"          # bundled MFEM and METIS (and hypre, STRUMPACK when a build bundles them)
    if libs.is_dir():
        extra.append(str(libs))
    m = mesher_path()
    gmsh_dir = gmsh_lib_dir()
    report["gmsh_lib_dir"] = gmsh_dir
    # The packaged mesher (axionfem/bin/axf-mesher) does not record where gmsh's wheel put libgmsh, so that directory
    # goes on the loader path. A mesher from a CMake build has the libgmsh it was linked against (the SDK's headless
    # build, or a gmsh wheel) in its RUNPATH; LD_LIBRARY_PATH would take precedence and could substitute another
    # libgmsh, e.g. the gmsh wheel's GUI build, which needs X11/OpenGL libraries.
    packaged = m is None or Path(m).resolve() == (_HERE / "bin" / "axf-mesher").resolve()
    report["gmsh_on_loader_path"] = bool(gmsh_dir) and packaged
    if gmsh_dir and packaged:
        extra.append(gmsh_dir)
    if extra:
        current = os.environ.get(_LIB_PATH_VAR, "")
        parts = [p for p in current.split(os.pathsep) if p]
        for p in extra:
            if p not in parts:
                parts.insert(0, p)
        os.environ[_LIB_PATH_VAR] = os.pathsep.join(parts)
    report[_LIB_PATH_VAR] = os.environ.get(_LIB_PATH_VAR, "")

    # Single-process MPI (Python sessions): avoid InfiniBand/UCX initialization
    # on cluster nodes.  Harmless in a build without MPI.
    os.environ.setdefault("OMPI_MCA_pml", "ob1")
    os.environ.setdefault("OMPI_MCA_btl", "self,vader")

    report["mesher"] = m
    if m and "AXF_MESHER" not in os.environ:
        os.environ["AXF_MESHER"] = m
    return report


_report = prepare_environment()
