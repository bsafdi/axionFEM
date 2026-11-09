"""MCP server exposing axionFEM to AI agents (Model Context Protocol, stdio).

Run:   python -m axionfem.mcp_server            (workspace: $AXF_WORKSPACE or ./axf_workspace)
Register it with any MCP client as a stdio server (command ``<venv>/bin/python``, arguments
``-m axionfem.mcp_server``); client-specific examples are in docs/guides/mcp.md.

Designs live as JSON files in <workspace>/designs/<name>.json; solves write
<workspace>/runs/<name>/{results.json, scene.json, <name>.html}.  Every tool
returns JSON; errors carry 'where' and 'hint' so the agent can fix the input.

Solves, sweeps and optimizations run in a worker subprocess
(``axionfem._mcp_worker``): they report MCP progress notifications while they
run, can be cancelled with ``cancel_job`` (or by the client's
``notifications/cancelled``), and cannot take the server down with them.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anyio

try:
    from mcp.server.mcpserver import Context, Image, MCPServer
except ImportError as _e:  # mcp 1.x has no mcp.server.mcpserver (it was called FastMCP there)
    raise ImportError(
        "the axionFEM MCP server needs the 'mcp' package, version 2.0 or newer: "
        "pip install -U 'mcp>=2.0' (the package's 'mcp' extra, axionfem[mcp], installs the same)") from _e

import axionfem as af

WORKSPACE = Path(os.environ.get("AXF_WORKSPACE", "axf_workspace")).resolve()
DESIGNS = WORKSPACE / "designs"
RUNS = WORKSPACE / "runs"
JOBDIR = WORKSPACE / "jobs"
THREADS = int(os.environ.get("AXF_THREADS", "4"))

GUIDE = """\
axionFEM agent guide
====================
Purpose: design and evaluate lumped-element axion dark-matter detectors (toroid / ABRACADABRA,
solenoid / DMRadio-m3, CORE / DMRadio-Core, custom) with a C++ finite-element solver; also
cavity haloscopes, SRF heterodyne detectors, dc SQUID readout chains and COMSOL files (last section).

Workflow
1. list_templates -> create_design(template, name, parameters) or put_design(name, design).
2. solve_design(name) -> Psi (signal coupling), L_p (pickup inductance), F = Psi^4/L_p^2,
   stored energy W, peak fields, superconductor surface fields, forces, and for each noise
   benchmark: F/F_req, the field-amplitude scale needed to reach the target coupling,
   the cost metric (MCHF; a scaling metric for comparing designs, not a construction estimate) and the
   cost-optimal size.  In a linear design every source scales
   together: the drive needed at the requirement is (ampere-turns) x amplitude_scale; at the
   cost-optimal size every length is multiplied by length_scale and the peak field is B_max_T,
   so the drive there is (ampere-turns) x length_scale x B_max_T / fields.B_max_T.
3. render_design(name) -> PNG image to look at; the HTML viewer path is returned by solve_design.
4. set_parameters / sweep / sensitivities to explore (set_parameters(..., save_as=new_name) keeps the
   original and saves a variant), optimize_design (gradient-based), optimize_global (differential
   evolution over the whole box, no start needed) or optimize (derivative-free Powell) to improve.

Physics conventions (SI unless stated)
- Psi = (1/I) int A_I . B_lab d^3x [H T m^2]; natural units Psi_nat = Psi/mu0 [T m^3].
  Axion flux Phi_a = kappa_a Psi, kappa_a = g_agg sqrt(2 rho_DM).
- Scan requirement F = Psi^4/L_p^2 >= F_req (optimal resonant scan of Chaudhuri et al.,
  default: DFSZ over 0.5-4.1 neV in 3 years, an illustrative target; a design's scan section sets its own).  Benchmarks (illustrative presets; give a design its own
  {name, Q, T, eta_A} for a real readout): optimistic (Q=2e7, T=10 mK, eta_A=0.1), conservative
  (Q=1e6, T=30 mK, eta_A=10), conservative_solenoid (Q=1e5).
- Geometry is axisymmetric in the (r, z) half-plane, r >= 0, metres.  Lengths may be
  expressions of named parameters.  Later regions override earlier ones where they overlap.
- Materials: vacuum/air, winding (stranded coil, vacuum-like), nb/superconductor (Meissner),
  copper/conductor (transparent to static field, perfect conductor at RF; builtin copper R_s: anomalous skin, RRR 100; copper_classical: classical skin), iron (warm, B-H law).
- Windings: type 'azimuthal' (solenoid/CORE coil) or 'poloidal' (toroidal winding, ring-shaped
  region).  Pickup: 'azimuthal' (slit superconducting sheath, toroid) or 'poloidal' (readout
  port region spanning a gap between conductors: coax, CORE neck).
- Superconductor states: azimuthal {net_current|flux}, poloidal {net_current|flux},
  azimuthal_slit.  RF pickup modes conserve fluxoid automatically.
- Check superconductor surface fields against Nb limits (~86 mT with margin, 172 mT bare)
  after scaling to the requirement (benchmarks[*].superconductor_surface_regular_at_req_T; set
  cost.sc_wall_cap / cost.B_peak_cap to have caps checked and penalised in optimize).
- Iron is nonlinear (B-H law) in static fields; RF problems use its secant permeability.
- Persistent circuits ({"type": "persistent", "flux_linkage": total linkage}) model passive
  cancellation coils; a background field ({"type": "uniform"|"grid"}) models an external magnet.
- Regions can be switched by a parameter: "enabled": "iron_t" (CORE iron caps).
- Conductors: give a winding "conductor": "nbti" | "nb3sn" | "rebco" (or your own under the design's
  "conductors") and solve_design reports its critical-current margins: engineering and superconductor
  current density (fill_factor), peak |B| and the field angle to the tape, Jc at the worst point of the
  winding, the load-line fraction I/Ic and short-sample scale, and the temperature margin -- also at the
  required amplitude and the cost-optimal size. list_conductors() shows the built-in critical surfaces;
  cost.load_line_cap penalises designs over a given fraction of Jc in optimize.
- HTS screening currents: set_hts(name, regions, conductor, ...) adds a critical-state (E-J power law)
  time-dependent analysis of the designated superconducting regions (the static solve keeps the ideal
  Meissner model). Results under 'hts': screening-current distribution and profiles, the
  screening-current-induced field (SCIF) at the field-of-view points, field quality, trapped flux after
  a ramp-down, AC loss per cycle, and Psi recomputed with the critical-state field.
- Shape sensitivities: sensitivities(name) gives dJ/dp and d log J / d log p for every design
  parameter (adjoint on a morphed mesh; exact for the discrete model, no remeshing noise), and
  optimize_design minimizes/maximizes any of those quantities under bounds and constraints.
  Derivatives are total: parameters written as expressions of p (a template's g = "0.005*R")
  move with p, so dJ/dR is a scale-up of everything defined relative to R.
  Zero geometric derivatives are physical (e.g. a copper coax is invisible to the static field);
  a parameter that only sets a mesh size has no derivative at fixed mesh.
- Beyond MQS: set_fullwave(name, f_min, f_max, points, modes) adds a time-harmonic Maxwell analysis
  (axisymmetric): Z(omega) = R_rad - i omega L(omega), Psi(omega), Q_rad = omega L/R_rad, F(omega) and
  self-resonances. MQS numbers hold while kR << 1 (corrections ~ (kR)^2); check the sweep when the
  device is not small compared with the wavelength (kR = 2 pi f R / c).  fullwave.mqs_validity gives,
  for Psi, L_p and F, the frequency above which each departs from its MQS value by 1%, 5%, 10%.
- 3D: make_3d(name) revolves an axisymmetric design into a full 3D design (or write one with
  "symmetry": "3d" and OpenCASCADE solids). solve_design/render_design work the same; 3D scenes
  show a quarter cutaway with the field on the two cut planes.
- Forces: per winding Fz and the hoop load per radian; per superconductor the Maxwell pressure.
  For a powered azimuthal winding, Fz_self_N is the numerical self-force (exactly zero in theory,
  so it is the error bar on Fz_N) and Fz_excluding_self_N is the force from every other source --
  the number to quote for a linear design.  forces.closure is the Newton's-third-law sum over all
  bodies, which must vanish (not when a background field is present: its source is outside the
  model).  forces.distributions holds the distributed load along each superconductor / iron /
  magnetizable surface (traction in Pa, force per unit meridional length in N/m) and the Lorentz
  force density inside each winding; the per-sample arrays are in the full results file and plot
  with axionfem.plots.force_distribution.  A poloidal (toroid) winding feels no net axial force
  from its own field at all, so it needs no self-force estimate.
- Mesh control: mesh.size/surface_size globally, mesh_size/surface_size per region or curve
  (surface_size refines at that body's own boundary), mesh.domain_center_z to centre the outer
  boundary on a device far from z = 0, and per-vertex polygon fillets ([r, z, fillet] points or a
  parallel "fillets" list).
- Long jobs: solve_design, sweep, sensitivities, optimize_design, optimize_global and optimize each run in a
  worker subprocess and send progress notifications.  list_jobs shows what is running;
  cancel_job(job_id) stops it (cancel_job() stops everything).  Cancelling the MCP request itself
  also kills the worker.
ALWAYS read `warnings` and `checks` in the solve result: Newton non-convergence, circuit
residuals, sharp-corner peaks and approximations are reported there, never silently.
Results are only as good as the mesh: compare order 2 vs 3 (solve_design(order=3)) before
quoting numbers.  Sharp convex corners on superconductors give peak fields that grow under
refinement (flagged; fields.superconductor_surface_regular_T excludes them): use fillets.
"""

from ._mcp_modules import GUIDE as _MODULES_GUIDE  # noqa: E402

GUIDE = GUIDE + _MODULES_GUIDE

server = MCPServer(name="axionfem", title="axionFEM detector design", version=af.version(), instructions=GUIDE)

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")   # no leading dot: no "..", no hidden files


def _err(e: Exception) -> dict:
    msg = str(e)
    out = {"ok": False, "error": msg}
    m = re.search(r"\[at ([^\]]*)\]", msg)
    if m:
        out["where"] = m.group(1)
    if "Hint:" in msg:
        out["hint"] = msg.split("Hint:", 1)[1].strip()
    return out


def _path(name: str) -> Path:
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid design name {name!r}: use letters, digits, '_', '-', '.' (not first), at most 80")
    return DESIGNS / f"{name}.json"


def _load(name: str) -> af.Design:
    p = _path(name)
    if not p.exists():
        have = sorted(x.stem for x in DESIGNS.glob("*.json"))
        raise FileNotFoundError(f"no design named {name!r}; existing designs: {have}")
    return af.load(p)


def _save(name: str, d: af.Design) -> dict:
    d.doc["name"] = name
    info = d.validate()
    DESIGNS.mkdir(parents=True, exist_ok=True)
    d.save(_path(name))
    return info


# ---------------------------------------------------------------- worker jobs
class _Job:
    """A running worker subprocess."""

    def __init__(self, job_id: str, kind: str, name: str):
        self.id = job_id
        self.kind = kind
        self.name = name
        self.started = time.time()
        self.process: Any = None
        self.cancelled = False
        self.tail: list[str] = []   # the worker's non-progress output, for a crash report
        self.log: Any = None        # open file the same lines are teed to
        self.progress = 0.0
        self.cap = 0.1          # progress may be nudged this far on messages without their own value
        self.message = "starting"

    def info(self) -> dict:
        return {"job_id": self.id, "kind": self.kind, "design": self.name,
                "seconds": round(time.time() - self.started, 1),
                "progress": round(self.progress, 4), "message": self.message,
                "cancelled": self.cancelled}

    def signal(self, sig: int) -> bool:
        """Signal the whole worker process group (the solver spawns axf-mesher)."""
        p = self.process
        if p is None or p.returncode is not None:
            return False
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                p.send_signal(sig)
            except Exception:  # noqa: BLE001
                return False
        return True


_JOBS: dict[str, _Job] = {}
_JOB_SEQ = itertools.count(1)


async def _report(ctx: Context | None, job: _Job, value: float | None, message: str,
                  total: float | None) -> None:
    """Send one progress notification.  MCP progress must not go backwards, so a message that
    carries no fraction of its own only nudges it, and never past the stage it belongs to."""
    if value is None:
        value = min(job.progress + 0.005, job.cap)
    else:
        job.cap = min(max(job.cap, value + 0.08), 0.99)
    value = min(max(value, job.progress), 1.0)
    job.progress, job.message = value, message
    if ctx is None:
        return
    try:
        await ctx.report_progress(value, total, message)
    except Exception:  # noqa: BLE001  (a dead client must not fail the job)
        pass


async def _pump(ctx: Context | None, job: _Job) -> None:
    """Relay the worker's stdout (its progress records and the solver's [axf] log) as progress."""
    from ._mcp_worker import stage_fraction

    buf = b""
    async for chunk in job.process.stdout:
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            if line.startswith("AXF-PROGRESS "):
                try:
                    rec = json.loads(line[len("AXF-PROGRESS "):])
                except json.JSONDecodeError:
                    continue
                await _report(ctx, job, float(rec.get("progress", 0.0)), str(rec.get("message", "")),
                              rec.get("total"))
            elif line.startswith("[axf] "):
                msg = line[len("[axf] "):]
                await _report(ctx, job, stage_fraction(msg), msg, 1.0)
            else:
                # anything else is the worker dying: a traceback, an MFEM abort, an OOM message
                job.tail.append(line)
                del job.tail[:-40]
                if job.log is not None:
                    job.log.write(line + "\n")
                    job.log.flush()


async def _run_job(ctx: Context | None, kind: str, name: str, payload: dict) -> dict:
    """Run one job in a worker subprocess, streaming progress; returns its result document."""
    JOBDIR.mkdir(parents=True, exist_ok=True)
    jid = f"{kind}-{name}-{next(_JOB_SEQ)}"
    job_file, result_file = JOBDIR / f"{jid}.job.json", JOBDIR / f"{jid}.result.json"
    result_file.unlink(missing_ok=True)
    job_file.write_text(json.dumps({**payload, "kind": kind, "name": name, "result": str(result_file),
                                    "threads": THREADS}))
    job = _Job(jid, kind, name)
    _JOBS[jid] = job
    logf = JOBDIR / f"{jid}.log"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        job.log = logf.open("w")
        job.process = await anyio.open_process(
            [sys.executable, "-u", "-m", "axionfem._mcp_worker", str(job_file)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
        # a cancel_job that arrived while the process was starting could not signal it yet
        if job.cancelled:
            job.signal(signal.SIGTERM)
        try:
            await _pump(ctx, job)
            await job.process.wait()
        finally:
            # client cancellation, cancel_job, or a normal exit: never leave the worker behind
            with anyio.CancelScope(shield=True):
                if job.process.returncode is None:
                    job.signal(signal.SIGTERM)
                    with anyio.move_on_after(5):
                        await job.process.wait()
                if job.process.returncode is None:
                    job.signal(signal.SIGKILL)
                    with anyio.move_on_after(5):
                        await job.process.wait()  # reap it, and close the stdout transport
                await job.process.aclose()
        rc = job.process.returncode
        if job.cancelled:
            return {"ok": False, "cancelled": True, "job_id": jid,
                    "error": f"{kind} of {name!r} was cancelled after {time.time() - job.started:.1f} s"}
        if result_file.exists():
            out = json.loads(result_file.read_text())
            out.setdefault("job_id", jid)
            return out
        detail = " ".join(job.tail[-8:])[:1200]
        return {"ok": False, "job_id": jid,
                "error": f"the {kind} worker exited with status {rc} without writing a result"
                         + (f": {detail}" if detail else ""),
                "hint": f"the worker's full output is in {logf}"}
    finally:
        if job.log is not None:
            job.log.close()
        _JOBS.pop(jid, None)


@server.tool()
def axionfem_guide() -> str:
    """How to use these tools and the physics conventions. Read this first."""
    return GUIDE


@server.tool()
def list_templates() -> dict:
    """Parametric detector templates with parameter defaults, units and meanings."""
    return {"ok": True, "templates": af.templates()}


@server.tool()
def design_schema() -> dict:
    """JSON Schema of the design document (axionfem.design/1)."""
    return af.schema()


@server.tool()
def list_designs() -> dict:
    """Designs in the workspace with their parameters (an empty list in a new workspace)."""
    out = []
    for p in sorted(DESIGNS.glob("*.json")):
        try:
            d = af.load(p)
            out.append({"name": p.stem, "parameters": d.params,
                        "solved": (RUNS / p.stem / "results.json").exists()})
        except Exception as e:  # noqa: BLE001
            out.append({"name": p.stem, "error": str(e)})
    return {"ok": True, "workspace": str(WORKSPACE), "designs": out}


@server.tool()
def create_design(template: str, name: str, parameters: dict[str, Any] | None = None) -> dict:
    """Create a design from a template (see list_templates) and save it under `name`.
    `parameters` overrides template defaults; values are numbers or expressions of other parameters."""
    try:
        d = af.template(template, **(parameters or {}))
        info = _save(name, d)
        return {"ok": True, "name": name, "design": d.doc, "resolved_parameters": info["parameters"]}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def get_design(name: str) -> dict:
    """The full design document."""
    try:
        return {"ok": True, "design": _load(name).doc}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def put_design(name: str, design: dict) -> dict:
    """Create or replace a design from a full design document (validated before saving)."""
    try:
        d = af.Design(design)
        info = _save(name, d)
        return {"ok": True, "name": name, "resolved_parameters": info["parameters"]}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def set_parameters(name: str, parameters: dict[str, Any], save_as: str | None = None) -> dict:
    """Change parameters of a saved design (numbers or expressions); validates and saves.
    save_as: write the changed design under this new name and leave `name` unchanged (a variant)."""
    try:
        d = _load(name)
        d.set(**parameters)
        out_name = save_as or name
        info = _save(out_name, d)
        return {"ok": True, "name": out_name, "parameters": d.params, "resolved_parameters": info["parameters"]}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def set_fullwave(name: str, f_min: float = 0.0, f_max: float = 0.0, points: int = 20, frequencies: list[float] | None = None,
                 boundary: str = "radiation", modes: int = 0) -> dict:
    """Request a full-wave analysis beyond magnetoquasistatics (axisymmetric designs): readout impedance
    Z(omega) = R_rad - i omega L(omega), coupling Psi(omega) (-> MQS Psi as omega -> 0), Q_rad, F(omega) and,
    with modes > 0, eigenmodes in a closed conducting sphere. Give frequencies [Hz] or f_min/f_max/points
    (log-spaced). boundary: 'radiation' (open, exact) or 'pec' (closed sphere). All zero removes the
    section. Then call solve_design; results appear under 'fullwave'."""
    try:
        d = _load(name)
        if not frequencies and f_max <= 0 and modes <= 0:
            d.doc.pop("fullwave", None)
        else:
            fw: dict = {"boundary": boundary}
            if frequencies:
                fw["frequencies"] = [float(x) for x in frequencies]
            elif f_max > 0:
                fw.update({"f_min": float(f_min if f_min > 0 else f_max), "f_max": float(f_max), "points": int(points)})
            if modes > 0:
                fw["modes"] = int(modes)
            d.doc["fullwave"] = fw
        _save(name, d)
        return {"ok": True, "name": name, "fullwave": d.doc.get("fullwave")}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def list_conductors() -> dict:
    """Built-in superconductor critical surfaces Jc(B, T) ("nbti" Bottura fit, "nb3sn" Durham/ITER
    scaling at fixed strain, "rebco" anisotropic Kim) with their parameters, operating temperature and
    a few Jc values in a perpendicular field. A design may override them or add its own under
    "conductors" (law nbti_bottura | nb3sn_durham | rebco_kim | table)."""
    return af.conductors()


@server.tool()
def set_hts(name: str, regions: list[str] | None = None, conductor: str = "rebco",
            initial: str = "zero_field_cooled", t_ramp: float = 1.0, t_hold: float = 0.0,
            ramp_down: bool = True, cycles: int = 1, steps_per_segment: int = 20,
            field_of_view: list[list[float]] | None = None) -> dict:
    """Request a critical-state (screening-current) analysis of HTS regions (axisymmetric designs).
    `regions` are superconducting regions treated as power-law conductors (E = E0 (|J|/Jc)^n); the
    static and pickup solves keep the ideal Meissner model, so this is opt-in and additive. The drive
    waveform multiplies all winding currents and the background field by scale(t): ramp 0 -> 1 in
    t_ramp [s], hold t_hold, ramp back to 0 (ramp_down), repeated `cycles` times. `initial`:
    'zero_field_cooled' (cooled in zero field, must start at scale 0) or 'field_cooled' (cooled with
    the drive on, no screening currents at t0). `field_of_view` are (r, z) points where the
    screening-current-induced field is reported. regions=None removes the section. Then solve_design;
    results appear under 'hts'."""
    try:
        d = _load(name)
        if not regions:
            d.doc.pop("hts", None)
        else:
            hts: dict = {"regions": [{"region": r, "conductor": conductor, "initial": initial} for r in regions],
                         "drive": {"t_ramp": float(t_ramp), "t_hold": float(t_hold),
                                   "ramp_down": bool(ramp_down), "cycles": int(cycles)},
                         "steps_per_segment": int(steps_per_segment)}
            if field_of_view:
                hts["field_of_view"] = [[float(p[0]), float(p[1])] for p in field_of_view]
            d.doc["hts"] = hts
        _save(name, d)
        return {"ok": True, "name": name, "hts": d.doc.get("hts")}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
async def solve_design(name: str, ctx: Context, order: int = 0, scene: bool = True,
                       self_force: bool = True, force_samples: int = 200) -> dict:
    """Mesh and solve a saved design. Returns coupling (Psi, L_p, F, W, g), peak fields, forces,
    per-benchmark F/F_req, required amplitude, the cost metric (a scaling metric, not a construction estimate)
    and the cost-optimal size, and output paths.
    order: finite-element order override (0 = design default; use 3 to check convergence).
    self_force: estimate and remove the numerical self-force of each powered azimuthal winding
    (one extra linear solve per winding; turn it off for the largest meshes).
    force_samples: samples per body in forces.distributions (0 = net forces and the closure check
    only).  The merge preserves the integral, so the net forces never depend on it; it sets how
    finely the distributed load is resolved and how big results.json gets.
    Runs in a worker subprocess: it reports progress and can be stopped with cancel_job."""
    try:
        d = _load(name)
        return await _run_job(ctx, "solve", name,
                              {"design": d.doc, "workdir": str(RUNS / name), "order": int(order),
                               "scene": bool(scene), "self_force": bool(self_force),
                               "force_samples": int(force_samples)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def make_3d(name: str, save_as: str | None = None, slit_deg: float = 4.0, mesh: dict[str, Any] | None = None,
            pickup: str = "ideal") -> dict:
    """Convert a saved axisymmetric design into a full 3D design (every region revolved). A slit-sheath
    pickup stays closed and carries the readout current around the axis (pickup='ideal', the exact 3D
    counterpart of the axisymmetric model) or gets a real slit of slit_deg fed by a port (pickup='slit',
    for studying the slit itself). The 3D solver is slower than the axisymmetric engine: start coarse,
    e.g. mesh={"size": 0.3, "order": 1}, compare with the axisymmetric result, then refine (order 2 is
    the validated order). Saved as `save_as` (default <name>_3d)."""
    try:
        from .convert import to_3d

        d3 = to_3d(_load(name), slit_deg=slit_deg, mesh=mesh, pickup=pickup)
        out = save_as or f"{name}_3d"
        info = _save(out, d3)
        return {"ok": True, "saved_as": out, "regions": [r["name"] for r in d3.doc["regions"]],
                "pickup": d3.doc.get("pickup"), "validate": info}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def render_design(name: str, view: str = "section", quantity: str = "Bmag", log_scale: bool = False):
    """PNG image of a solved design. view: 'section' (meridional cut, mirrored) or '3d' (cutaway).
    quantity: Bmag, Br, Bz, Bphi, psi, pickup_Bmag."""
    try:
        scene_path = RUNS / name / "scene.json"
        if not scene_path.exists():
            return {"ok": False, "error": f"design {name!r} has no solved scene", "hint": "call solve_design first"}
        from .render import render_png  # C++ rasterizer binding

        png = render_png(json.loads(scene_path.read_text()), view=view, quantity=quantity, log_scale=log_scale)
        out = RUNS / name / f"{name}_{view}_{quantity}.png"
        out.write_bytes(png)
        return Image(data=png, format="png")
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def requirement(benchmark: str = "optimistic", scan: dict[str, Any] | None = None) -> dict:
    """F_req for a benchmark ('optimistic', 'conservative', 'conservative_solenoid' or a dict
    {name,Q,T,eta_A}); scan overrides f_min, f_max [Hz], live_time [yr], model, C_QCD, rho_DM, Q_axion, Theta95."""
    try:
        return {"ok": True, **af.requirement(benchmark, **(scan or {}))}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
async def sweep(name: str, parameter: str, values: list[Any], ctx: Context,
                benchmark: str = "optimistic") -> dict:
    """Solve the design for several values of one parameter; returns a table of
    Psi, L_p, g, F/F_req, W at requirement, peak fields, SC surface peaks and cost.
    Runs in a worker subprocess: it reports progress per value and cancel_job stops it."""
    try:
        base = _load(name)
        return await _run_job(ctx, "sweep", name,
                              {"design": base.doc, "parameter": parameter, "values": list(values),
                               "benchmark": benchmark, "workdir": str(RUNS / f"{name}__sweep")})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
async def optimize(name: str, free: dict[str, list[float]], ctx: Context, metric: str = "cost_optimal",
                   benchmark: str = "optimistic", max_evals: int = 40, save_as: str | None = None) -> dict:
    """Minimize a metric over parameters within bounds (bounded Powell).
    free: {parameter: [low, high]}.  metric: 'cost_optimal' (cost at the cost-optimal size,
    iron-free designs), 'cost', 'W' (stored energy at requirement) or 'minus_g'.
    Cost metrics are penalised when the design's cost.sc_wall_cap or cost.B_peak_cap is exceeded.
    Saves the best design as `save_as` (default <name>_opt).
    Runs in a worker subprocess: it reports progress per evaluation and cancel_job stops it."""
    try:
        base = _load(name)
        out = await _run_job(ctx, "optimize", name,
                             {"design": base.doc, "free": free, "metric": metric, "benchmark": benchmark,
                              "max_evals": int(max_evals), "workdir": str(RUNS / f"{name}__opt")})
        if out.get("ok"):
            out_name = save_as or f"{name}_opt"
            _save(out_name, base.with_params(**out["best"]["x"]))
            out["saved_as"] = out_name
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
async def sensitivities(name: str, ctx: Context, parameters: list[str] | None = None,
                        quantities: list[str] | None = None, method: str = "adjoint", step: float = 1e-4,
                        benchmark: str = "", order: int = 0) -> dict:
    """Derivatives of the figures of merit with respect to design parameters (shape sensitivities).

    Solves the design and returns dJ/dp for each quantity and parameter, plus the dimensionless
    logarithmic derivatives d log J / d log p ('relative'), which say directly which parameter is
    worth changing and in which direction. `parameters` defaults to all of them, `quantities` to
    Psi, L_p, W_mag, F and cost; available quantities are listed by list_objectives (cost-like ones
    accept a benchmark, e.g. 'cost:optimistic').  method: 'adjoint' (default; axisymmetric linear
    designs), 'fd_morph' or 'fd_remesh' (finite differences; the only option for nonlinear iron or
    for parameters that switch regions on and off).
    Runs in a worker subprocess: it reports progress and can be stopped with cancel_job."""
    try:
        d = _load(name)
        return await _run_job(ctx, "sensitivities", name,
                              {"design": d.doc, "workdir": str(RUNS / name), "parameters": parameters or [],
                               "quantities": quantities or [], "method": method, "step": float(step),
                               "benchmark": benchmark, "order": int(order)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def list_objectives() -> dict:
    """Quantities that sensitivities and optimize_design can use as objectives or constraints."""
    return {"ok": True, "quantities": af.objectives(),
            "note": "cost, cost_optimal, W_at_req, F_over_F_req, amplitude_scale, sc_surface_at_req and "
                    "B_peak_at_req refer to a noise benchmark and accept 'quantity:benchmark'"}


@server.tool()
async def optimize_design(name: str, ctx: Context, objective: str = "cost",
                          free: dict[str, list[float]] | None = None, constraints: list[dict] | None = None,
                          sense: str = "min", max_iter: int = 30, benchmark: str = "",
                          save_as: str | None = None) -> dict:
    """Gradient-based optimization (adjoint sensitivities, scipy L-BFGS-B / SLSQP).

    free: {parameter: [low, high]}.  objective: any quantity from list_objectives, e.g. 'cost'
    (cost metric at the requirement, i.e. the design is scaled to reach F_req first),
    'cost_optimal', 'F', 'g_factor'.  constraints: [{"quantity": "W_at_req", "type": "<=",
    "value": 1.0e8}, ...] over any differentiable quantity; use sc_surface_at_req / B_peak_at_req
    to enforce field caps.  sense: 'min' or 'max'.  Much cheaper than `optimize` (derivative-free
    Powell) and finds smooth optima; needs an axisymmetric design with linear materials.
    Saves the optimized design as `save_as` (default <name>_opt).
    Runs in a worker subprocess: it reports progress per iteration and cancel_job stops it."""
    try:
        base = _load(name)
        if not free:
            return {"ok": False, "error": "give free={'parameter': [low, high], ...}"}
        out = await _run_job(ctx, "optimize_adjoint", name,
                             {"design": base.doc, "objective": objective, "free": free,
                              "constraints": constraints or [], "sense": sense, "max_iter": int(max_iter),
                              "benchmark": benchmark, "workdir": str(RUNS / f"{name}__optimize")})
        if out.get("ok"):
            out_name = save_as or f"{name}_opt"
            _save(out_name, af.Design(out["design"]))
            out["saved_as"] = out_name
            out.pop("design", None)
            out["history"] = out["history"][-20:]
        return {"name": name, **out}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
async def optimize_global(name: str, free: dict[str, list[Any]], ctx: Context, objective: str = "cost",
                          constraints: list[dict] | None = None, sense: str = "min", benchmark: str = "",
                          seed: int = 0, popsize: int = 15, maxiter: int = 30, workers: int = 1,
                          polish: bool = True, save_as: str | None = None) -> dict:
    """Global optimization by differential evolution (af.optimize(method="global")).

    free: {parameter: [low, high]} or [low, high, "log"] for a log-scaled search.  objective and
    constraints as for optimize_design (any quantity of list_objectives; constraints are enforced by
    the feasibility rule).  About max(5, popsize * len(free)) * (maxiter + 1) evaluations: use it when the cost
    landscape may have several minima or no good starting point is known; it is deterministic for a
    seed (each evaluation runs on one OpenMP thread).  workers > 1 solves each generation in parallel.
    The best point is polished with the adjoint optimizer when the design is linear, else with bounded
    Powell.  Works for nonlinear iron: the amplitude parameter NI is solved for F = F_req at every point
    (about 3 solves per point).  Saves the best design as
    `save_as` (default <name>_opt).  Runs in a worker subprocess (progress per generation; cancel_job
    stops it)."""
    try:
        base = _load(name)
        if not free:
            return {"ok": False, "error": "give free={'parameter': [low, high], ...}"}
        out = await _run_job(ctx, "optimize_global", name,
                             {"design": base.doc, "objective": objective, "free": free,
                              "constraints": constraints or [], "sense": sense, "benchmark": benchmark,
                              "seed": int(seed), "popsize": int(popsize), "maxiter": int(maxiter),
                              "workers": int(workers), "polish": bool(polish),
                              "workdir": str(RUNS / f"{name}__global" / (time.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex()))})
        if out.get("ok"):
            out_name = save_as or f"{name}_opt"
            _save(out_name, af.Design(out["design"]))
            out["saved_as"] = out_name
        out.pop("design", None)
        if "history" in out:
            out["history"] = out["history"][-20:]
        return {"name": name, **out}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@server.tool()
def list_jobs() -> dict:
    """Solves, sweeps and optimizations running right now, with their progress and job_id."""
    return {"ok": True, "jobs": [j.info() for j in list(_JOBS.values())]}


@server.tool()
def cancel_job(job_id: str = "") -> dict:
    """Stop a running solve / sweep / optimize (see list_jobs for job_id).
    With no job_id every running job is stopped. The tool call that started the job returns
    {"ok": false, "cancelled": true}."""
    # this tool runs on a worker thread while _run_job runs on the event loop, so look the job up
    # once instead of testing and then indexing (it may finish in between)
    one = _JOBS.get(job_id) if job_id else None
    jobs = [one] if one is not None else ([] if job_id else list(_JOBS.values()))
    if job_id and not jobs:
        return {"ok": False, "error": f"no running job with job_id {job_id!r}",
                "hint": "call list_jobs; a job that already finished cannot be cancelled",
                "running": [j.id for j in _JOBS.values()]}
    stopped = []
    for j in jobs:
        j.cancelled = True   # _run_job honours this even if the worker has not started yet
        j.signal(signal.SIGTERM)
        stopped.append(j.id)
    return {"ok": True, "cancelled": stopped}


class _Workspace:
    """The workspace as the module tools see it (reads the globals, so tests can redirect them)."""

    workspace = property(lambda self: WORKSPACE)
    runs = property(lambda self: RUNS)

    @staticmethod
    def check_name(name: str) -> Path:
        return _path(name)

    @staticmethod
    def load(name: str) -> af.Design:
        return _load(name)

    @staticmethod
    def save(name: str, d: af.Design) -> dict:
        return _save(name, d)


# cavity haloscope, SRF heterodyne, dc SQUID and COMSOL tools (axionfem._mcp_modules)
from ._mcp_modules import register as _register_modules  # noqa: E402

_register_modules(server, _Workspace(), _err)


_viewer_thread: threading.Thread | None = None


@server.tool()
def start_viewer(port: int = 8765) -> dict:
    """Start the live 3D viewer server for the workspace (reloads when designs are re-solved).
    On a cluster, forward the port: ssh -L PORT:localhost:PORT <host>."""
    global _viewer_thread
    try:
        import socket

        from .viewer import serve

        if _viewer_thread is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
                if sk.connect_ex(("127.0.0.1", port)) == 0:
                    return {"ok": False, "error": f"port {port} is already in use",
                            "hint": "choose another port, e.g. start_viewer(port=8766)"}
            _viewer_thread = threading.Thread(target=serve, args=(str(RUNS),), kwargs={"port": port}, daemon=True)
            _viewer_thread.start()
        return {"ok": True, "url": f"http://127.0.0.1:{port}/"}
    except Exception as e:  # noqa: BLE001
        return _err(e)


def main() -> None:
    DESIGNS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    JOBDIR.mkdir(parents=True, exist_ok=True)
    af.set_threads(THREADS)
    try:
        server.run("stdio")
    finally:
        for j in list(_JOBS.values()):  # never leave a worker running after the server stops
            j.cancelled = True
            j.signal(signal.SIGKILL)


if __name__ == "__main__":
    main()
