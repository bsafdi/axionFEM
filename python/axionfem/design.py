"""Design documents: load, build from templates, edit, validate, solve."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:          # annotations only (results imports design)
    from .results import Results

# One line of the pretty-printed JSON `axf` writes (nlohmann dump(2)): an optional key, then a bracket, a closing
# bracket or a scalar, an optional comma. Strings hold no raw newlines, so every line of the document matches.
_STR = r'"(?:[^"\\]|\\.)*"'
_JSON_LINE = re.compile(r"^\s*(?:" + _STR + r"\s*:\s*)?(?:[{\[]|[}\]],?|(?:" + _STR +
                        r"|-?\d[\d.eE+\-]*|true|false|null|\{\}|\[\])\s*,?)\s*$")


def _is_axf_result(obj: Any) -> bool:
    """`axf solve`'s summary: {"ok": true, "results": "<path>", ...} or {"ok": false, "error": "...", ...}."""
    if not isinstance(obj, dict) or not isinstance(obj.get("ok"), bool):
        return False
    return isinstance(obj.get("results" if obj["ok"] else "error"), str)


def _pretty_document(lines: list[str], first: int) -> str | None:
    """The pretty-printed JSON document that opens at lines[first] ("{" at column 0), skipping foreign lines.

    nlohmann's dump(2) puts one key or value per line, indented by two spaces per level of nesting. A line belongs
    to the document only if it has that form (_JSON_LINE) and exactly the indentation its depth requires; other
    lines (warnings) are skipped. The document is rejected (None) rather than accepted without a member when a
    skipped line could be one of its own lines displaced or damaged by a foreign fragment spliced into it (srun
    forwards output in 1 KiB pieces): a line of JSON form at the wrong indentation, or a line at the required
    indentation that starts like a member (a quote). None also if the document does not close.
    """
    if lines[first] != "{":
        return None
    kept = ["{"]
    depth = 1
    for line in lines[first + 1:]:
        body = line.lstrip(" ")
        closing = body[:1] in ("}", "]")
        indent = 2 * (depth - 1 if closing else depth)
        json_form = bool(_JSON_LINE.match(line))
        if not json_form or len(line) - len(body) != indent:
            if json_form or (len(line) - len(body) == indent and body[:1] == '"'):
                return None
            continue
        kept.append(line)
        if closing:
            depth -= 1
        if body.rstrip().endswith(("{", "[")):
            depth += 1
        if depth == 0:
            return "\n".join(kept)
    return None


def _parse_axf_output(stdout: str) -> dict | None:
    """The summary object of an `axf solve` run from its stdout, or None.

    MPI launchers and transports (UCX, OpenMPI, PMIx, srun) may print warnings into the same stream: before, after,
    or, from other ranks, between the lines of rank 0's JSON. Warnings may contain
    braces and even small JSON objects, so an object is accepted only if it opens with "{" at column 0 and is the
    summary (_is_axf_result); errors printed by every rank before MPI starts give several copies, and the first is
    taken. Pass 1 decodes the raw stream at each such line. Pass 2 rebuilds each pretty-printed document without the
    foreign lines inside it (_pretty_document). A summary cut short, or with a member damaged by a spliced
    fragment, is not accepted. `axf solve` also writes the summary to DIR/axf_status.json, which Design.solve reads
    first (_read_axf_status); this parser is the fallback.
    """
    dec = json.JSONDecoder()
    for m in re.finditer(r"(?m)^\{", stdout):
        try:
            obj = dec.raw_decode(stdout, m.start())[0]
        except ValueError:
            continue
        if _is_axf_result(obj):
            return obj
    # split at newlines only: str.splitlines would also split at U+2028 or U+0085 inside strings (raw UTF-8)
    lines = [line[:-1] if line.endswith("\r") else line for line in stdout.split("\n")]
    for k in range(len(lines)):
        doc = _pretty_document(lines, k)
        if doc is None:
            continue
        try:
            obj = json.loads(doc)
        except ValueError:
            continue
        if _is_axf_result(obj):
            return obj
    return None


def _read_axf_status(path: Path) -> dict | None:
    """The summary `axf solve` wrote to DIR/axf_status.json (rank 0, atomically), or None."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if _is_axf_result(obj) else None


def schema() -> dict:
    """JSON Schema of the design format (axionfem.design/1)."""
    return json.loads(_core.schema())


def templates() -> list[dict]:
    """Parametric templates with their parameters (default, units, meaning)."""
    return json.loads(_core.templates())


def template(name: str, **params: Any) -> "Design":
    """Instantiate a parametric template, overriding parameter defaults.

    Values may be numbers or expression strings over other parameters,
    e.g. ``template("core", Rc=1.5, b="2*Rc")``.
    """
    return Design(json.loads(_core.template(name, json.dumps(params))))


def load(path: str | Path) -> "Design":
    """Load a design from a JSON file."""
    return Design(json.loads(Path(path).read_text()))


class Design:
    """A design document (dict) with helpers.

    The document is the single source of truth; this class never hides state.
    ``d.doc`` is the underlying dict and may be edited directly.
    """

    def __init__(self, doc: dict):
        if isinstance(doc, Design):
            doc = doc.doc
        self.doc: dict = copy.deepcopy(doc)

    # ------------------------------------------------------------ basics
    @property
    def name(self) -> str:
        return self.doc.get("name", "design")

    @property
    def params(self) -> dict:
        """Parameter definitions (numbers or expressions)."""
        return self.doc.setdefault("parameters", {})

    def resolved_params(self) -> dict:
        """Numerical values of all parameters (expressions evaluated)."""
        return json.loads(_core.validate(json.dumps(self.doc)))["parameters"]

    def set(self, **params: Any) -> "Design":
        """Change existing parameters in place; returns self.

        Raises KeyError for unknown names (use ``add_param`` to create one).
        """
        for k, v in params.items():
            if k not in self.params:
                known = ", ".join(sorted(self.params))
                raise KeyError(f"unknown parameter '{k}'; known parameters: {known}")
            self.params[k] = v
        return self

    def add_param(self, name: str, value: Any) -> "Design":
        self.params[name] = value
        return self

    def with_params(self, **params: Any) -> "Design":
        """A modified copy."""
        return Design(self.doc).set(**params)

    def copy(self) -> "Design":
        return Design(self.doc)

    def validate(self) -> dict:
        """Parse and check the design in C++; raises AxionFEMError with a hint on problems."""
        return json.loads(_core.validate(json.dumps(self.doc)))

    # ------------------------------------------------------------ io
    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.doc, indent=indent)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json())
        return p

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Design({self.name!r}; {ps})"

    # ------------------------------------------------------------ solve
    def solve(self, workdir: str | Path | None = None, order: int = 0, verbose: int = 1,
              threads: int | None = None, mpi_ranks: int = 1, launcher: str | None = None,
              self_force: bool = True, force_samples: int = 200) -> "Results":
        """Mesh and solve; returns Results.

        workdir defaults to ``runs/<name>``.  ``order`` overrides mesh.order.
        3D designs: ``mpi_ranks > 1`` runs ``axf solve --save`` on that many MPI ranks (``launcher``: e.g.
        "srun" or "mpirun"; default: $AXF_MPI_LAUNCHER, else srun inside a SLURM allocation when srun is on PATH,
        else mpirun, else mpiexec) and
        loads the saved solution, so sampling and scenes work as after a single-process solve.

        ``self_force`` estimates (and, for linear designs, removes) the numerical self-force of each
        powered azimuthal winding, at the price of one extra linear solve per such winding
        (axisymmetric designs only).

        ``force_samples`` is the number of samples per body in ``forces.distributions``.  The merge
        that produces them preserves the integral, so the net forces do not depend on it; it only
        sets how finely the load is resolved and how big ``results.json`` gets (about 160 bytes per
        sample per body).  0 keeps the net forces and the closure check but writes no arrays.
        """
        from .results import Results

        if threads:
            from . import set_threads

            set_threads(threads)
        wd = Path(workdir) if workdir else Path("runs") / self.name
        if self.doc.get("symmetry", "axisymmetric") == "xsec":  # z-uniform cavity cross-section (axionfem.cavity_xsec)
            from .cavity_xsec import XsecResults

            if mpi_ranks > 1:
                raise ValueError("mpi_ranks > 1 is for 3D designs; the cross-section solver runs in one process")
            modelx = _core.XsecModel(json.dumps(self.doc))
            modelx.run(str(wd), int(order), int(verbose), "")
            return XsecResults(modelx, self, wd)
        if self.doc.get("symmetry", "axisymmetric") in ("axisymmetric", "planar") and \
                "file" in (self.doc.get("mesh") or {}):  # 2D full wave on an imported mesh (COMSOL 2D import)
            from .fw2d import Fw2DResults

            if mpi_ranks > 1:
                raise ValueError("mpi_ranks > 1 is for 3D designs; the 2D imported-mesh solver runs in one process")
            model2 = _core.Fw2DModel(json.dumps(self.doc))
            model2.run(str(wd), int(order), int(verbose))
            return Fw2DResults(model2, self, wd)
        if self.doc.get("symmetry", "axisymmetric") == "3d":
            from .results import Results3D

            if mpi_ranks > 1:
                return self._solve_mpi(wd, order, verbose, mpi_ranks, launcher, threads)
            model3 = _core.Model3D(json.dumps(self.doc))
            model3.run(str(wd), int(order), int(verbose), "")
            return Results3D(model3, self, wd)
        if mpi_ranks > 1:
            raise ValueError("mpi_ranks > 1 is for 3D designs; the axisymmetric engine runs in one process")
        model = _core.Model(json.dumps(self.doc))
        model.run(str(wd), int(order), int(verbose), "", bool(self_force), int(force_samples))
        return Results(model, self, wd)

    # ------------------------------------------------------------ sensitivities
    def gradient(self, params: Any = None, objectives: Any = None, method: str = "adjoint", step: float = 1e-4,
                 steps: dict | None = None, benchmark: str = "", workdir: str | Path | None = None,
                 order: int = 0, verbose: int = 0) -> dict:
        """Solve and return the derivatives of the figures of merit with respect to parameters.

        Shorthand for ``self.solve(...).sensitivities(...)``; see
        :meth:`axionfem.Results.sensitivities` for the arguments and the returned dict.
        """
        res = self.solve(workdir=workdir, order=order, verbose=verbose)
        return res.sensitivities(params=params, objectives=objectives, method=method, step=step, steps=steps,
                                 benchmark=benchmark, order=order)

    def optimize(self, objective: str = "cost", params: dict | None = None, **kw) -> dict:
        """Optimize this design; see :func:`axionfem.optimize` (``method="global"``: differential evolution)."""
        from .global_opt import optimize as _opt

        return _opt(self, objective=objective, params=params, **kw)

    def _solve_mpi(self, wd: Path, order: int, verbose: int, nranks: int, launcher: str | None,
                   threads: int | None = None) -> "Results":
        import os
        import shlex
        import shutil
        import subprocess
        import uuid

        from ._paths import axf_path, default_mpi_launcher
        from .results import Results3D

        exe = axf_path()
        if exe is None:
            raise RuntimeError("cannot find the axf executable (build target axf, or set AXF_EXE)")
        launcher = launcher or default_mpi_launcher()
        wd.mkdir(parents=True, exist_ok=True)
        dpath = wd / "design.json"
        dpath.write_text(json.dumps(self.doc, indent=1))
        # rank 0's summary goes to a status file of this launch only, so that concurrent solves in one workdir do not
        # read each other's
        status = wd / f"axf_status.{uuid.uuid4().hex}.json"
        cmd = shlex.split(launcher) + ["-n", str(nranks), exe, "solve", str(dpath), "-o", str(wd), "-v", str(verbose),
                                       "--save", "--status-file", str(status)]
        if order:
            cmd += ["--order", str(order)]
        # single node: keep the shared-memory transport (avoids InfiniBand/UCX setup); several nodes need the
        # network transports, so drop the single-process settings of this Python session
        multi_node = int(os.environ.get("SLURM_NNODES", "1")) > 1
        env = {k: v for k, v in os.environ.items() if not (multi_node and k in ("OMPI_MCA_pml", "OMPI_MCA_btl"))}
        if "SLURM_MEM_PER_NODE" in env:  # srun refuses mutually exclusive SLURM_MEM_PER_* settings
            env.pop("SLURM_MEM_PER_CPU", None)
            env.pop("SLURM_MEM_PER_GPU", None)
        env.pop("SLURM_TRES_PER_TASK", None)  # may contradict SLURM_CPUS_PER_TASK in a nested srun
        env["SLURM_EXPORT_ENV"] = "ALL"  # srun passes this environment to the ranks (sites may default to NONE)
        if threads:
            env["OMP_NUM_THREADS"] = str(int(threads))
        timeout = float(os.environ.get("AXF_MPI_TIMEOUT", "0")) or None  # seconds; unset: no limit
        log = wd / "mpi_solve.log"
        # fields/ of an earlier run must not pass for this run's (exit 0 is accepted below only with fields/ written)
        if (wd / "fields").is_dir():
            shutil.rmtree(wd / "fields")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as ex:
            log.write_text(str(ex.stdout or "") + "\n" + str(ex.stderr or ""))
            raise _core.AxionFEMError(f"MPI solve exceeded AXF_MPI_TIMEOUT = {timeout} s; see {log}")
        log.write_text(proc.stdout + "\n" + proc.stderr)
        # rank 0's summary: this launch's status file, else stdout among any launcher/transport warnings
        # (_parse_axf_output). Exit 0 with fields/ written (by this launch: an older one was removed) is a success even
        # if neither can be read (a split line must not turn a finished solve into a failure).
        out = _read_axf_status(status) or _parse_axf_output(proc.stdout)
        try:
            status.replace(wd / "axf_status.json")  # the name `axf solve` uses by default
        except OSError:
            pass
        fields = wd / "fields"
        if out is None and proc.returncode == 0 and fields.is_dir():
            out = {"ok": True, "results": str(wd / "results.json")}
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        stderr_tail = f". Last lines of stderr:\n{tail}" if tail else ""
        if out is None:
            raise _core.AxionFEMError(f"MPI solve failed (exit {proc.returncode}); see {log}{stderr_tail}")
        if not out.get("ok", False):
            err = out.get("error", "MPI solve failed")
            if out.get("hint"):
                err += f" (hint: {out['hint']})"
            raise _core.AxionFEMError(err)
        if proc.returncode != 0:
            # rank 0 finished (it writes the summary after saving the fields), then the launch failed: another rank,
            # or MPI/UCX teardown. Not accepted silently.
            saved = (f"; rank 0 reported success and saved the solution in {fields} (axionfem._core.Model3D.load); check "
                     f"the log before using it") if fields.is_dir() else ""
            raise _core.AxionFEMError(f"MPI solve failed (exit {proc.returncode}) after rank 0 reported success; see "
                                      f"{log}{saved}{stderr_tail}")
        return Results3D(_core.Model3D.load(str(fields)), self, wd)
