"""F4a heat transfer in fluids: the case runner (set-up, solve, Evaluator).

Every norm, extremum, flux, identity and Taylor-test number comes from the check modules f4a_heat_check
(imported as H) and scalar_transport_check (S, shared with chemistry C2).  This module only builds the fixed meshes with
S's builders, verifies the solver's mesh against them, solves through the public API (axionfem.fluids.solve_heat and
solve_boussinesq: the shared scalar-transport operator and the Boussinesq solver) and wraps each discrete solution in an
Evaluator (the protocol of the check modules' docstrings).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two levels only.

Failed solves are results (a solve that misses its gate, or whose factorization throws, is recorded in runs[] with
converged = false and only the quantities that need its fields are skipped: the judges the case FAIL on the
solver gate, not UNPROVEN).

Conventions: flux data of the API are INWARD (g = n.kappa grad T), this runner
converts the design's OUTWARD q_out (g = -q_out; total flux j_out likewise).  Boundary attributes of every rectangle:
1 bottom (y = y0 / z = z0), 2 right (x = x1 / r = r1), 3 top, 4 left (x = x0 / r = r0 / the axis r = 0).  Axisymmetric
Boussinesq: the API's internal component order is (u_r, u_z, u_theta), the check modules' (u_r, u_theta, u_z): PERM
permutes both ways.  Every cavity case reports BOTH Nu_hot and Nu_cold; the flux rule is the
design's: q_in = -Q_out of the variational flux (Nu_cold = + Q_out of the cold wall).  SUPG: the one rule shakib_hK
(the runner records its name in every run).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

import f4a_heat_check as H
import scalar_transport_check as S
from axionfem import fluids

FORM = {"lap": "laplacian", "sym": "stress"}
TOL = 1e-9
SIDES = {"bottom": 1, "right": 2, "top": 3, "left": 4}
SIDE_OF = {"x0": 4, "r0": 4, "axis": 4, "x1": 2, "r1": 2, "y0": 1, "z0": 1, "y1": 3, "z1": 3}
NORMAL = {"x0": (-1, 0), "r0": (-1, 0), "x1": (1, 0), "r1": (1, 0), "y0": (0, -1), "z0": (0, -1), "y1": (0, 1),
          "z1": (0, 1)}
PERM = [0, 2, 1]          # (u_r, u_z, u_t) <-> (u_r, u_t, u_z)
UNIT_BOX = (0.0, 1.0, 0.0, 1.0)
TAU_RULE = "shakib_hK"
BOX = (0, 1, 0, 1)


# ============================================================================================== identity of the core
_CORE: dict = {}


def core_identity() -> dict:
    """This process's compiled core: path, sha256, git sha."""
    if not _CORE:
        from axionfem import _core
        f = os.path.realpath(_core.__file__)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        try:
            git = json.loads(_core.build_info()).get("git_sha")
        except Exception:   # noqa: BLE001
            git = None
        _CORE.update(core_file=f, core_sha256=h.hexdigest(), git_sha=git)
    return dict(_CORE)


def _levels(seq, quick, n=2):
    seq = list(seq)
    return seq[:n] if quick else seq


# ============================================================================================== meshes
def boundary_edges(T):
    cnt = {}
    for t in np.asarray(T):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(t[i], t[j]), max(t[i], t[j]))
            cnt[key] = cnt.get(key, 0) + 1
    return [k for k, c in cnt.items() if c == 1]


def rect_classify(x0, x1, y0, y1):
    def f(x, y):
        if abs(y - y0) < TOL:
            return SIDES["bottom"]
        if abs(x - x1) < TOL:
            return SIDES["right"]
        if abs(y - y1) < TOL:
            return SIDES["top"]
        if abs(x - x0) < TOL:
            return SIDES["left"]
        raise RuntimeError(f"boundary edge at ({x}, {y}) is on no side of the rectangle")
    return f


def build_mesh(family, k, box=UNIT_BOX, axi=False):
    """(V, T, h_nominal, classify) of a fixed family: unit (k = 0..4), distorted (k = 1..5), annulus, channel / pipe."""
    if family == "unit":
        V, T = S.box_mesh(8, 8, k)
        return V, T, 0.125 / 2 ** k, rect_classify(*UNIT_BOX)
    if family == "distorted":
        V, T = S.distorted_mesh(k)
        return V, T, 0.25 / 2 ** k, rect_classify(*UNIT_BOX)
    if family == "annulus":
        V, T = S.box_mesh(4, 8, k, box)
        return V, T, 0.125 / 2 ** k, rect_classify(*box)
    if family in ("channel", "pipe"):
        V, T = S.box_mesh(8, 16, k, box) if axi else S.box_mesh(16, 8, k, box)
        return V, T, 0.125 / 2 ** k, rect_classify(*box)
    raise ValueError(family)


def make_mesh(V, T, classify):
    V = np.ascontiguousarray(V, dtype=float)
    T = np.ascontiguousarray(T, dtype=np.int32)
    seg = np.array(boundary_edges(T), dtype=np.int32)
    att = np.array([classify(*(0.5 * (V[a] + V[b]))) for a, b in seg], dtype=np.int32)
    return fluids.Mesh.from_arrays(V, T, seg, att), seg, att


def verify(mesh, V_ref, T_ref, V_coarser=None):
    v = S.verify_mesh(mesh.vertices(), mesh.elements(), V_ref, T_ref, V_coarser=V_coarser)
    bad = [k for k in ("counts_match", "hash_match", "nested") if k in v and not v[k]]
    if bad:
        raise RuntimeError(f"the solver mesh does not match the fixed construction: {bad} ({v})")
    return v


# ============================================================================================== evaluators, data
class Evaluator:
    """The discrete solution in the check modules' protocol (one MPI rank): scalar T (ScalarTransportSolution.scalar or
    BoussinesqSolution.temperature), and for Boussinesq velocity / pressure."""

    def __init__(self, sol, axi=False, bous=False):
        self.sol, self.axi, self.bous = sol, axi, bous
        self.n_elements = int(sol.n_elements)
        self._el = np.arange(self.n_elements, dtype=np.int32)

    @staticmethod
    def _ref(ref_pts):
        return np.ascontiguousarray(ref_pts, dtype=float)

    def geometry(self, ref_pts):
        return self.sol.geometry(self._el, self._ref(ref_pts))

    def scalar(self, ref_pts):
        f = self.sol.temperature if self.bous else self.sol.scalar
        t, g = f(self._el, self._ref(ref_pts))
        return np.asarray(t), np.asarray(g)

    def velocity(self, ref_pts):
        u, g = self.sol.velocity(self._el, self._ref(ref_pts))
        u, g = np.asarray(u), np.asarray(g)
        if self.axi:
            u, g = u[..., PERM], g[..., PERM, :]
        return u, g

    def pressure(self, ref_pts):
        return np.asarray(self.sol.pressure(self._el, self._ref(ref_pts)))


def check_data(fn, axi):
    """The fixed velocity data in the check module's order, with the shared-node axis rule (u_r = u_t = 0 at r = 0,
    which the exact data have as their limit; their numpy spelling is not evaluated there for those components)."""
    if not axi:
        return fn

    def g(X, Y):
        X = np.asarray(X, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a = np.array(fn(X, np.asarray(Y, float)), dtype=float, copy=True)
        on = np.abs(X) < 1e-12
        a[on, 0] = 0.0
        a[on, 1] = 0.0
        return a
    return g


def to_api(fn, axi):
    if fn is None or not axi:
        return fn
    return lambda X, Y: np.ascontiguousarray(np.asarray(fn(X, Y), float)[..., PERM])


def series(levels, errors, magnitude):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


class Acc:
    def __init__(self):
        self.q = {}

    def max(self, k, v):
        self.q[k] = max(self.q.get(k, -math.inf), float(v))

    def min(self, k, v):
        self.q[k] = min(self.q.get(k, math.inf), float(v))


def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v if isinstance(v, (bool, int)) else (bool(v) if isinstance(v, np.bool_) else int(v))
    if v is None or isinstance(v, str):
        return v
    return str(v)


# ============================================================================================== one case's state
class Failed:
    """A solve that raised (a failed factorization, a refused configuration): a result, never a crash."""

    def __init__(self, why):
        self.why = why
        self.diagnostics = {"converged": False, "relative_residual": float("nan"), "failure": why}
        self.history = []
        self.n_elements = 0


class Run:
    """Runs, series, quantities and notes of one case; every solve goes through solve_scalar / solve_bous."""

    def __init__(self, params, quick):
        self.P, self.quick = params, quick
        self.ident = core_identity()
        self.runs, self.ser, self.acc, self.notes = [], {}, Acc(), []
        self.failed = []

    # ---- scalar transport
    def solve_scalar(self, level, mesh, info, h, fld, bcs, *, order=2, supg=False, form="convective", family=None,
                     refinement_level=None, rtol=1e-12, maxit=4):
        axi = fld.geometry == "axisymmetric"
        kap, gk = kappa_fns(fld)
        vel = None if fld.kind == "conduction" else (lambda X, Y: fld.b(X, Y))
        cons = form != "convective"
        div = (lambda X, Y: 0.0 * np.asarray(X, float)) if cons and vel is not None else None
        src = None if fld.kind == "graetz_modes" else fld.s("conservative" if cons else "convective")
        t0 = time.time()
        try:
            sol = fluids.solve_heat(mesh, kap, capacity=fld.a, velocity=vel, source=src, bcs=bcs,
                                    geometry="axisymmetric" if axi else "planar", order=order, form=form, supg=supg,
                                    grad_kappa=gk, div_velocity=div, rtol=rtol, maxit=maxit)
        except Exception as exc:   # noqa: BLE001
            sol = Failed(f"{type(exc).__name__}: {exc}")
        sec = time.time() - t0
        e = run_entry(level, sol, sec, h, info, family, refinement_level, fld, order, supg, form, axi, False)
        self.runs.append(e)
        ok = bool(sol.diagnostics["converged"])
        if not ok:
            self.failed.append(level)
        return sol, ok

    # ---- Boussinesq
    def solve_bous(self, level, mesh, info, h, fld, form, *, supg=False, family=None, refinement_level=None,
                   newton_rtol=1e-12, newton_maxit=30, line_search=False, continuation=None, params=None):
        axi = fld.geometry == "axisymmetric"
        vb, tb = bous_bcs(fld, axi)
        gk = grad_fn(fld.data["grad_k"], fld.coords)
        kw = {}
        if continuation:
            kw.update(continuation_ra=list(continuation), ra=float(continuation[-1]))
        t0 = time.time()
        try:
            sol = fluids.solve_boussinesq(
                mesh, fld.mu, fld.rho0, fld.cp, fld.k, fld.beta_g, T_ref=fld.T_ref, gravity="-z" if axi else "-y",
                velocity_bcs=vb, temperature_bcs=tb, force=to_api(fld.f(form), axi), heat_source=fld.Q,
                form=FORM[form], geometry="axisymmetric" if axi else "planar", supg=supg, line_search=line_search,
                newton_rtol=newton_rtol, newton_maxit=newton_maxit, grad_k=gk, **kw)
        except Exception as exc:   # noqa: BLE001
            sol = Failed(f"{type(exc).__name__}: {exc}")
        sec = time.time() - t0
        e = run_entry(level, sol, sec, h, info, family, refinement_level, fld, 2, supg, "convective", axi, True,
                      viscous_form=form)
        self.runs.append(e)
        ok = bool(sol.diagnostics["converged"])
        if not ok:
            self.failed.append(level)
        return sol, ok

    def cavity(self, level, mesh, info, h, Ra, Pr, form, supg, *, family=None, refinement_level=None,
               newton_rtol=1e-12, newton_maxit=30):
        """The differentially heated cavity (de Vahl Davis scaling) with the Ra continuation 1e3, 1e4, ... below Ra
        (line search allowed): returns (solution, ok, record) with BOTH Nusselt numbers."""
        p = fluids.natconv_cavity_parameters(Ra, Pr)
        zero2 = lambda X, Y: np.zeros((np.asarray(X).shape[0], 2))   # noqa: E731
        vb = [dict(attribute=a, kind="dirichlet", value=zero2) for a in (1, 2, 3, 4)]
        tb = [dict(attribute=4, kind="dirichlet", value=0.5), dict(attribute=2, kind="dirichlet", value=-0.5),
              dict(attribute=1, kind="zero"), dict(attribute=3, kind="zero")]
        kw = {}
        if Ra > 0:
            sched = [r for r in (1e3, 1e4, 1e5, 1e6, 1e7) if r < Ra] + [Ra]
            kw.update(continuation_ra=sched, ra=float(Ra))
        t0 = time.time()
        try:
            sol = fluids.solve_boussinesq(
                mesh, p["mu"], p["rho0"], p["cp"], p["k"], p["beta_g"], T_ref=0.0, gravity="-y", velocity_bcs=vb,
                temperature_bcs=tb, form=FORM[form], supg=supg, line_search=True, newton_rtol=newton_rtol,
                newton_maxit=newton_maxit, **kw)
        except Exception as exc:   # noqa: BLE001
            sol = Failed(f"{type(exc).__name__}: {exc}")
        sec = time.time() - t0
        e = run_entry(level, sol, sec, h, info, family, refinement_level, None, 2, supg, "convective", False, True,
                      viscous_form=form)
        ok = bool(sol.diagnostics["converged"])
        rec = None
        if ok:
            ev = Evaluator(sol, False, True)
            rec = H.cavity_functionals(ev, False)
            R_T = np.asarray(sol.full_residual_T())
            q_hot = flux_of(self, sol, R_T, [4])
            q_cold = flux_of(self, sol, R_T, [2])
            rec.update({"Q_out_hot": q_hot, "Q_out_cold": q_cold, "Nu_hot": H.nusselt(-q_hot, 1.0, 1.0),
                        "Nu_cold": H.nusselt(q_cold, 1.0, 1.0), "Ra": Ra, "Pr": Pr})
            e["cavity"] = {k: float(v) for k, v in rec.items()}
        else:
            self.failed.append(level)
        self.runs.append(e)
        return sol, ok, rec

    def result(self):
        me = self.ident
        if self.failed:
            self.notes.append(f"{len(self.failed)} solve(s) missed the gate (recorded; their field quantities skipped): "
                              + ", ".join(self.failed[:12]) + (" ..." if len(self.failed) > 12 else ""))
        self.notes.append(f"SUPG tau rule: {TAU_RULE} (the only rule; Shakib, h_K = longest edge, C = 1, pointwise at "
                          "the assembly quadrature points)")
        self.notes.append(f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]})")
        return {"runs": self.runs, "series": self.ser, "quantities": self.acc.q, "notes": " ".join(self.notes)}


def run_entry(level, sol, sec, h, info, family, refinement_level, fld, order, supg, form, axi, bous, viscous_form=None):
    d = sol.diagnostics
    g = d.get
    hist = [_jsonable(dict(x)) for x in getattr(sol, "history", [])]
    e = {"level": level,
         "mesh": {"h_nominal": h, "h_max": (info or {}).get("h_max"), "n_elements": int(sol.n_elements),
                  "dofs_field": int(g("dofs_field", 0) or 0), "files": [], "hash": (info or {}).get("hash"),
                  "all_boundary_triangles": (info or {}).get("all_boundary_triangles")},
         "solver": {"converged": bool(g("converged")), "relative_residual": float(g("relative_residual", float("nan"))),
                    "residual_lift": g("residual_lift"), "iterations": int(g("iterations", -1) or -1),
                    "history": hist, "newton_steps": sum(1 for x in hist if x.get("kind") == "newton"),
                    "kind": "boussinesq" if bous else "scalar_transport", "geometry": "axisymmetric" if axi else "planar",
                    "viscous_form": viscous_form, "advective_form": form, "temperature_order": order,
                    "supg": bool(supg), "tau_rule": TAU_RULE if supg else "none", "failure": g("failure"),
                    "lu_backend": g("lu_backend"), "lu_residual": g("lu_residual"),
                    "nullspace": g("nullspace"), "nullspace_detected": bool(g("nullspace_detected", False)),
                    "line_search_min_step_accepts": g("line_search_min_step_accepts"),
                    "buoyancy_scales": _jsonable(g("buoyancy_scales")), "start": g("start"),
                    "quad_orders": _jsonable(g("quad_orders")), "quad_order": g("quad_order"), "ranks": g("ranks")},
         "seconds": sec}
    if family is not None:
        e["mesh"]["family"] = family
        e["mesh"]["refinement_level"] = refinement_level
    if bous:
        e["mesh"]["dofs_velocity"] = int(g("dofs_velocity", 0) or 0)
        e["mesh"]["dofs_pressure"] = int(g("dofs_pressure", 0) or 0)
        e["mesh"]["dofs_temperature"] = int(g("dofs_temperature", 0) or 0)
    return e


# ============================================================================================== data helpers
def grad_fn(specs, coords):
    fs = [S.compile_expr(e, coords) for e in specs]
    return lambda X, Y: np.stack([np.broadcast_to(f(X, Y), np.shape(X)) for f in fs], -1)


def kappa_fns(fld):
    """(kappa, grad kappa): a number for the constant kappa of the Graetz fields; callables otherwise."""
    if fld.kind == "graetz_modes":
        return fld.kappa_value, None
    return (lambda X, Y: fld.kappa(X, Y)), grad_fn(fld.data["grad_kappa"], fld.coords)


def scalar_bcs(fld, form="convective"):
    """The API's boundary conditions of a scalar field from its fixed boundary list: Dirichlet = the exact T at the
    nodes; flux data INWARD g = -q_out (convective) / -j_out (conservative_flux, total flux); Robin h, T_ext (for the
    conservative form: T_ext_conservative, the total-flux Robin); symmetry -> zero; axis -> axis."""
    cc = fld.coords
    cons = form != "convective"
    bcs = []
    for bd in fld.data["boundaries"]:
        a, t, nm = SIDE_OF[bd["name"]], bd["type"], bd["name"]
        if t == "dirichlet":
            bcs.append(dict(attribute=a, kind="dirichlet", value=fld.T))
        elif t == "axis":
            bcs.append(dict(attribute=a, kind="axis"))
        elif t == "symmetry":
            bcs.append(dict(attribute=a, kind="zero"))
        elif t == "neumann":
            key = "j_out" if cons else "q_out"
            if key in bd:
                q = S.compile_expr(bd[key], cc)
                bcs.append(dict(attribute=a, kind="total_flux" if cons else "flux",
                                value=lambda X, Y, q=q: -np.asarray(q(X, Y), float)))
            else:   # the exact outward diffusive flux of an analytic field (Graetz outlet)
                n = NORMAL[nm]
                bcs.append(dict(attribute=a, kind="flux", value=lambda X, Y, n=n: fld.kappa_value * (
                    fld.grad_T(X, Y) * np.array(n, float)).sum(-1)))
        elif t == "robin":
            key = "T_ext_conservative" if cons and "T_ext_conservative" in bd else "T_ext"
            bcs.append(dict(attribute=a, kind="robin", h=float(bd["h"]["value"]),
                            T_ext=S.compile_expr(bd[key], cc)))
        else:
            raise ValueError(f"boundary type {t}")
    return bcs


def bous_bcs(fld, axi):
    """(velocity BCs, temperature BCs) of a Boussinesq field."""
    cc = fld.coords
    g_api = to_api(check_data(fld.u, axi), axi)
    vb = []
    for bd in fld.data["u_boundaries"]:
        a = SIDE_OF[bd["name"]]
        vb.append(dict(attribute=a, kind="axis") if bd["type"] == "axis" else
                  dict(attribute=a, kind="dirichlet", value=g_api))
    tb = []
    for bd in fld.data["T_boundaries"]:
        a, t = SIDE_OF[bd["name"]], bd["type"]
        if t == "dirichlet":
            tb.append(dict(attribute=a, kind="dirichlet", value=fld.T))
        elif t == "axis":
            tb.append(dict(attribute=a, kind="axis"))
        elif t == "neumann_zero":
            tb.append(dict(attribute=a, kind="zero"))
        elif t == "neumann":
            q = S.compile_expr(bd["q_out"], cc)
            tb.append(dict(attribute=a, kind="flux", value=lambda X, Y, q=q: -np.asarray(q(X, Y), float)))
        elif t == "robin":
            tb.append(dict(attribute=a, kind="robin", h=float(bd["h"]["value"]), T_ext=S.compile_expr(bd["T_ext"], cc)))
        else:
            raise ValueError(f"temperature boundary type {t}")
    return vb, tb


def dirichlet_attrs(boundaries):
    return sorted({SIDE_OF[b["name"]] for b in boundaries if b["type"] == "dirichlet"})


def flux_of(R, sol, resid, attrs):
    """The variational OUTWARD flux through the closed boundary part `attrs`: -R . w (scalar_transport_check)."""
    return S.flux_from_residual(resid, weights=np.asarray(sol.boundary_weights(list(attrs))))


def nodal_scalar(R, sol, fld, attrs, bous=False, key="dirichlet_nodal_data_defect_max"):
    P, U = (sol.temperature_boundary_nodes(list(attrs)) if bous else sol.boundary_nodes(list(attrs)))
    if len(P):
        R.acc.max(key, S.nodal_dirichlet_defect(np.asarray(P, float), np.asarray(U, float), fld.T))


def nodal_velocity(R, sol, fld, attrs, axi, key="dirichlet_nodal_data_defect_max"):
    P, U = sol.boundary_nodes(list(attrs))
    P, U = np.asarray(P, float), np.asarray(U, float)
    if len(P):
        if axi:
            U = U[:, PERM]
        g = check_data(fld.u, axi)(P[:, 0], P[:, 1])
        R.acc.max(key, float(np.max(np.abs(U - g))))


def put_series(R, prefix, rows, keys=None):
    """rows: [(level, norms)] of the converged levels; one series per error norm of the check module."""
    if not rows:
        return
    ks = keys or [k for k in rows[0][1] if k.startswith(("TL2", "TH1", "Tstream", "uL2", "uH1", "pL2", "divL2"))
                  and not k.endswith("magnitude") and k != "magnitudes"]
    for k in ks:
        if all(k in n for _, n in rows):
            R.ser[f"{prefix}_{k}"] = series([lv for lv, _ in rows], [n[k] for _, n in rows],
                                            rows[-1][1]["magnitudes"][k])


# ============================================================================================== scalar families
def scalar_family(R, fld, family, prefix, bcs, *, order=2, supg=False, form="convective", ks=None, box=UNIT_BOX,
                  axi=False, keep=False, extra=None):
    """Solve one fixed family (levels ks); norms of the converged levels; nodal checks.  Returns [(level, norms, sol, ev)]."""
    out, rows = [], []
    meshes = []
    for i, k in enumerate(ks):
        V, T, h, cl = build_mesh(family, k, box, axi)
        meshes.append((V, T, h, cl))
    for i, (V, T, h, cl) in enumerate(meshes):
        mesh, seg, att = make_mesh(V, T, cl)
        info = verify(mesh, V, T, meshes[i - 1][0] if i else None)
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve_scalar(lvl, mesh, info, h, fld, bcs, order=order, supg=supg, form=form, family=prefix,
                                 refinement_level=i)
        if not ok:
            continue
        ev = Evaluator(sol, axi, False)
        n = S.scalar_norms(ev, fld)
        nodal_scalar(R, sol, fld, dirichlet_attrs(fld.data["boundaries"]))
        if extra:
            n.update(extra(sol, ev, fld))
        rows.append((lvl, n))
        out.append((lvl, n, sol, ev))
    put_series(R, prefix, rows)
    return out


def load_scalar(P, key):
    return S.load_field(P["fields"][key])


def p2_nodal_error(ev, fld):
    X, _ = ev.geometry(S.P2_NODES)
    Th, _ = ev.scalar(S.P2_NODES)
    return float(np.max(np.abs(Th - fld.T(X[..., 0], X[..., 1]))))


def patch_checks(R, P, axi, fields, name_pairs):
    """PATCH TEST (SUPG consistency): max over unit N = 8, N = 32 and distorted k = 1 of max |T_h - T| at the P2 nodes;
    name_pairs: [(quantity, field key, supg, form)]."""
    runs = [("unit", 0), ("unit", 2), ("distorted", 1)]
    if R.quick:
        runs = runs[:1]
    for quantity, key, supg, form in name_pairs:
        fld = S.load_field(P["patch_fields"][key])
        bcs = scalar_bcs(fld, form)
        for fam, k in runs:
            V, T, h, cl = build_mesh(fam, k)
            mesh, seg, att = make_mesh(V, T, cl)
            info = verify(mesh, V, T)
            sol, ok = R.solve_scalar(f"patch_{key}_{'supg' if supg else 'gal'}_{form}_{fam}{k}", mesh, info, h, fld, bcs,
                                     supg=supg, form=form)
            if ok:
                R.acc.max(quantity, p2_nodal_error(Evaluator(sol, axi, False), fld))
                nodal_scalar(R, sol, fld, dirichlet_attrs(fld.data["boundaries"]))


def balance_checks(R, P, axi, quantity, key, supg):
    """The discrete balance identity (v = 1) on unit N = 8, 16, 32: q_D + natural_out - int s + int a b.grad T_h = 0."""
    fld = load_scalar(P, key)
    bcs = scalar_bcs(fld)
    dn = dirichlet_attrs(fld.data["boundaries"])
    for i, k in enumerate(_levels([0, 1, 2], R.quick)):
        V, T, h, cl = build_mesh("unit", k)
        mesh, seg, att = make_mesh(V, T, cl)
        info = verify(mesh, V, T)
        sol, ok = R.solve_scalar(f"balance_{key}_{'supg' if supg else 'gal'}_N{8 * 2 ** k}", mesh, info, h, fld, bcs,
                                 supg=supg)
        if not ok:
            continue
        ev = Evaluator(sol, axi, False)
        qD = flux_of(R, sol, np.asarray(sol.residual_full()), dn)
        parts = []
        for bd in fld.data["boundaries"]:
            on = S.edge_predicate(bd["name"], BOX)
            if bd["type"] == "neumann":
                parts.append((on, "neumann", S.compile_expr(bd["q_out"], fld.coords)))
            elif bd["type"] == "robin":
                parts.append((on, "robin", float(bd["h"]["value"]), S.compile_expr(bd["T_ext"], fld.coords)))
        nat = S.natural_outflow(ev, parts, axi)
        src = S.integral(ev, lambda X, T, dT: fld.s("convective")(X[..., 0], X[..., 1]), axi)
        adv = S.integral(ev, lambda X, T, dT: fld.a * (fld.b(X[..., 0], X[..., 1]) * dT).sum(-1), axi)
        R.acc.max(quantity, S.balance_defect(qD, nat, src, adv))


# ---------------------------------------------------------------------------------------------- S1 / S2
def _scalar_mms(params, quick, axi):
    R = Run(params, quick)
    P = params
    ks_unit = _levels([0, 1, 2, 3, 4], quick)
    ks_dist = _levels([1, 2, 3, 4, 5], quick)
    sfx = "w" if axi else ""
    fl = {pe: load_scalar(P, pe) for pe in ("pe1", "pe1e6")}
    bc = {pe: scalar_bcs(fl[pe]) for pe in fl}
    keep = {}

    def fam(pe, kind, family, ks, **kw):
        tag = {"unit": "unit", "distorted": "dist"}[family]
        prefix = f"{pe}_{kind}_{tag}"
        if kind == "gal":
            return scalar_family(R, fl[pe], family, prefix, bc[pe], ks=ks, axi=axi, **kw)
        return scalar_family(R, fl[pe], family, prefix, bc[pe], ks=ks, axi=axi, supg=True, **kw)
    keep["pe1_gal_unit"] = fam("pe1", "gal", "unit", ks_unit)
    fam("pe1", "gal", "distorted", ks_dist)
    fam("pe1", "supg", "unit", ks_unit)
    s_unit = fam("pe1e6", "supg", "unit", ks_unit)
    s_dist = fam("pe1e6", "supg", "distorted", ks_dist)
    # conservative form (total-flux natural data) at Pe ~ 1 with SUPG; P1 temperature
    fc = load_scalar(P, "pe1")
    scalar_family(R, fc, "unit", "pe1_cons_supg_unit", scalar_bcs(fc, "conservative_flux"), ks=ks_unit, axi=axi,
                  supg=True, form="conservative_flux")
    scalar_family(R, fl["pe1e6"], "unit", "p1_pe1e6_supg_unit", bc["pe1e6"], ks=ks_unit, axi=axi, supg=True, order=1)
    if not axi:
        scalar_family(R, fl["pe1"], "unit", "p1_pe1_gal_unit", bc["pe1"], ks=ks_unit, axi=axi, order=1)
    # finest-level bounds of the axisymmetric Pe 1e6 SUPG runs (orders are not proposed there)
    if axi:
        for key, nk in (("pe1e6_supg_Tstreamw_finest_max", "Tstreamw"), ("pe1e6_supg_TL2w_finest_max", "TL2w")):
            for runs in (s_unit, s_dist):
                if runs:
                    R.acc.max(key, runs[-1][1][nk])
    # patches
    pairs = [("patch_pe1e6_gal_nodal_err_max", "pe1e6", False, "convective"),
             ("patch_pe1e6_supg_nodal_err_max", "pe1e6", True, "convective"),
             ("patch_pe1e2_gal_nodal_err_max", "pe1e2", False, "convective"),
             ("patch_pe1e2_supg_nodal_err_max", "pe1e2", True, "convective")]
    if not axi:
        pairs.append(("patch_pe1e2_cons_supg_nodal_err_max", "pe1e2", True, "conservative_flux"))
    patch_checks(R, P, axi, None, pairs)
    balance_checks(R, P, axi, "balance_pe1_defect_max", "pe1", False)
    balance_checks(R, P, axi, "balance_pe1e6_defect_max", "pe1e6", True)
    # solver sensitivity: the finest pe1 Galerkin error at the default tolerance vs a tightened solve
    g = keep["pe1_gal_unit"]
    if g and not quick:
        lvl, n, sol, ev = g[-1]
        key = f"pe1_gal_unit_TL2{sfx}_finest"
        R.acc.q[key] = n["TL2" + sfx]
        V, T, h, cl = build_mesh("unit", ks_unit[-1])
        mesh, seg, att = make_mesh(V, T, cl)
        info = verify(mesh, V, T, build_mesh("unit", ks_unit[-2])[0])
        solT, okT = R.solve_scalar("tight_pe1_gal_unit_finest", mesh, info, h, fl["pe1"], bc["pe1"], rtol=1e-14, maxit=4)
        if okT:
            R.acc.q[key + "_tight"] = S.scalar_norms(Evaluator(solT, axi, False), fl["pe1"])["TL2" + sfx]
    return R.result()


def scalar_mms_planar(params, workdir, quick):
    """S1 mms_advdiff_2d: planar scalar transport MMS (Pe ~ 1 Galerkin/SUPG, Pe ~ 1e6 SUPG, unit and distorted
    families, conservative total-flux form, P1 temperature), SUPG patch tests, balance identities."""
    return _scalar_mms(params, quick, False)


def scalar_mms_axi(params, workdir, quick):
    """S2 mms_advdiff_axi: the axisymmetric counterpart (Robin face z = 1: the r weight of the Robin term; hoop term)."""
    return _scalar_mms(params, quick, True)


# ---------------------------------------------------------------------------------------------- L1
def scalar_layer(params, workdir, quick):
    """L1 analytic_advdiff_layer_2d: the unresolved outflow layer, P2 and P1 SUPG: sampled extrema over the levels and the
    interior L2 error of the finest level."""
    R = Run(params, quick)
    fld = S.load_field(params["field"])
    region = list(params["interior_region"])
    bcs = scalar_bcs(fld)
    for order, tag in ((2, "p2"), (1, "p1")):
        out = scalar_family(R, fld, "unit", f"{tag}_supg_unit", bcs, ks=_levels([0, 1, 2, 3, 4], quick), supg=True,
                            order=order)
        if not out:
            continue
        for lvl, n, sol, ev in out:
            ex = S.sampled_extrema(ev)
            R.acc.max(f"{tag}_supg_sampled_max_over_levels", ex["max"])
            R.acc.min(f"{tag}_supg_sampled_min_over_levels", ex["min"])
        R.acc.q[f"{tag}_supg_interior_TL2_finest"] = S.scalar_norms(out[-1][3], fld, region=region)["TL2"]
    return R.result()


# ---------------------------------------------------------------------------------------------- G1 / G2
def _wall_flux_extra(wall, length=2.0):
    def f(sol, ev, fld):
        return {}
    return f


def _graetz(params, quick, axi):
    R = Run(params, quick)
    P = params
    box = (0, 1, 0, 2) if axi else (0, 2, 0, 1)
    famname = "pipe" if axi else "channel"
    sfx = "w" if axi else ""
    ks = _levels([0, 1, 2, 3, 4], quick)
    modes = S.load_field(P["modes_field"])
    flux = S.load_field(P["uniform_flux_field"])
    wall = "r1" if axi else "y1"
    wattr = SIDE_OF[wall]
    q_ex = S.graetz_wall_flux_exact(modes, 2.0)
    nu_ex = float(flux.data["nusselt_closed_form"]["value"])
    for supg, tag in ((False, "gal"), (True, "supg")):
        # modes, with the variational wall flux against the exact flux
        rows = scalar_family(R, modes, famname, f"modes_{tag}", scalar_bcs(modes), ks=ks, supg=supg, box=box, axi=axi)
        errs = []
        for lvl, n, sol, ev in rows:
            qh = flux_of(R, sol, np.asarray(sol.residual_full()), [wattr])
            errs.append(abs(qh - q_ex))
        if rows:
            if not axi:
                R.ser[f"modes_{tag}_wall_flux_err"] = series([r[0] for r in rows], errs, abs(q_ex))
            else:
                R.acc.q[f"modes_{tag}_wall_flux_err_finest"] = errs[-1]
        # uniform wall flux: Nusselt number
        rows = scalar_family(R, flux, famname, f"flux_{tag}", scalar_bcs(flux), ks=ks, supg=supg, box=box, axi=axi)
        if rows:
            nus = [S.nusselt_uniform_flux(ev, flux, wall) for lvl, n, sol, ev in rows]
            R.ser[f"flux_{tag}_Nu_err"] = series([r[0] for r in rows], [abs(v - nu_ex) for v in nus], abs(nu_ex))
    if axi:   # Robin-wall modes (the non-MMS Robin cell), Galerkin
        rob = S.load_field(P["robin_modes_field"])
        scalar_family(R, rob, famname, "robin_gal", scalar_bcs(rob), ks=ks, box=box, axi=axi)
    return R.result()


def graetz_planar(params, workdir, quick):
    """G1 analytic_graetz_planar: Graetz modes with axial conduction (variational wall flux order 4) and the uniform-flux
    Nusselt number, Galerkin and SUPG."""
    return _graetz(params, quick, False)


def graetz_pipe(params, workdir, quick):
    """G2 analytic_graetz_pipe: the pipe (axisymmetric) Graetz modes, uniform flux, and Robin-wall modes (Bi = 2)."""
    return _graetz(params, quick, True)


# ---------------------------------------------------------------------------------------------- A1
def conduction_annulus(params, workdir, quick):
    """A1 analytic_conduction_annulus_axi: variable conductivity in a hollow cylinder, inner-wall flux exactly 1."""
    R = Run(params, quick)
    fld = S.load_field(params["field"])
    box = (0.5, 1.0, 0.0, 1.0)
    ks = _levels([0, 1, 2, 3, 4], quick)
    rows = scalar_family(R, fld, "annulus", "gal", scalar_bcs(fld), ks=ks, box=box, axi=True)
    if rows:
        errs = [abs(flux_of(R, sol, np.asarray(sol.residual_full()), [SIDE_OF["r0"]]) - 1.0)
                for lvl, n, sol, ev in rows]
        R.ser["gal_inner_flux_err"] = series([r[0] for r in rows], errs, 1.0)
    return R.result()


# ============================================================================================== Boussinesq families
def bous_family(R, fld, form, family, prefix, *, supg=False, ks=None, axi=False, pressure="mean_free", hydro=False,
                nc_key=True):
    """Solve one fixed family with the Boussinesq solver; norms of the converged levels; nodal checks."""
    meshes = [build_mesh(family, k) for k in ks]
    out, rows = [], []
    for i, (V, T, h, cl) in enumerate(meshes):
        mesh, seg, att = make_mesh(V, T, cl)
        info = verify(mesh, V, T, meshes[i - 1][0] if i else None)
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve_bous(lvl, mesh, info, h, fld, form, supg=supg, family=prefix, refinement_level=i)
        if nc_key and ok:
            R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        if not ok:
            continue
        ev = Evaluator(sol, axi, True)
        n = H.flow_norms(ev, fld, pressure)
        if hydro:
            dvn = H.discrete_velocity_norms(ev, axi, u_scale=H.hydrostatic_velocity_scale(fld))
            n["spurious"] = dvn
        nodal_scalar(R, sol, fld, dirichlet_attrs(fld.data["T_boundaries"]), bous=True)
        nodal_velocity(R, sol, fld, dirichlet_attrs(fld.data["u_boundaries"]), axi)
        rows.append((lvl, n))
        out.append((lvl, n, sol, ev, h, V, T))
    if not hydro:
        put_series(R, prefix, rows)
    return out, rows


def _bous_mms(params, quick, axi):
    R = Run(params, quick)
    P = params
    fld = H.load_bous(P["field"])
    sfx = "w" if axi else ""
    ks_unit = _levels([0, 1, 2, 3, 4], quick)
    ks_dist = _levels([1, 2, 3, 4, 5], quick)
    keep = {}
    n32 = 2   # unit N = 32 (index 2)
    for form in P.get("viscous_forms", ["lap", "sym"]):
        for supg, tag in ((False, "gal"), (True, "supg")):
            out, rows = bous_family(R, fld, form, "unit", f"{form}_{tag}_unit", supg=supg, ks=ks_unit, axi=axi)
            keep[(form, tag)] = out
        if not axi:
            bous_family(R, fld, form, "distorted", f"{form}_gal_dist", ks=ks_dist, axi=axi)
    # Newton iterations from the lift at N = 32 (full steps, exact Jacobian): the runs above ARE those solves
    if not quick:
        for r in R.runs:
            lv = (r["mesh"].get("family") or "")
            if lv.endswith("_unit") and r["mesh"].get("refinement_level") == n32:
                steps = r["solver"]["newton_steps"] if r["solver"]["converged"] else 31 + 1
                R.acc.max("newton_iterations_from_lift_max", steps)
    # solver sensitivity: the finest lap Galerkin uL2 at newton_rtol vs the tight value
    g = keep.get(("lap", "gal")) or []
    if g and not quick:
        lvl, n, sol, ev, h, V, T = g[-1]
        key = f"lap_gal_unit_uL2{sfx}_finest"
        R.acc.q[key] = n["uL2" + sfx]
        mesh, seg, att = make_mesh(V, T, build_mesh("unit", ks_unit[-1])[3])
        info = verify(mesh, V, T)
        solT, okT = R.solve_bous("tight_lap_gal_unit_finest", mesh, info, h, fld, "lap", newton_rtol=1e-14)
        if okT:
            R.acc.q[key + "_tight"] = H.flow_norms(Evaluator(solT, axi, True), fld)["uL2" + sfx]
    return R.result()


def bous_mms_planar(params, workdir, quick):
    """B1 mms_boussinesq_2d: coupled planar Boussinesq MMS (lap/sym x Galerkin/SUPG; distorted lap/sym Galerkin)."""
    return _bous_mms(params, quick, False)


def bous_mms_axi(params, workdir, quick):
    """B2 mms_boussinesq_axi: the axisymmetric counterpart (swirl; Robin face z = 1)."""
    return _bous_mms(params, quick, True)


# ---------------------------------------------------------------------------------------------- H1
def hydrostatic(params, workdir, quick):
    """H1 analytic_hydrostatic_noflow: stratified (pL2m order 2, spurious Taylor-Hood velocity) and uniform-T identities,
    planar and axisymmetric, lap and sym."""
    R = Run(params, quick)
    P = params
    ks = _levels([0, 1, 2, 3, 4], quick)
    for geo, axi in (("sq", False), ("axi", True)):
        sfx = "w" if axi else ""
        for kind in ("strat", "uniform"):
            fld = H.load_bous(P["fields"][f"hydro_{geo}_{kind}"])
            for form in P.get("viscous_forms", ["lap", "sym"]):
                prefix = f"{kind}_{geo}_{form}"
                out, rows = bous_family(R, fld, form, "unit", prefix, ks=ks, axi=axi, hydro=True)
                if not out:
                    continue
                lv = [o[0] for o in out]
                if kind == "strat":
                    R.ser[f"{prefix}_pL2m{sfx}"] = series(lv, [o[1]["pL2m" + sfx] for o in out],
                                                          out[-1][1]["magnitudes"]["pL2m" + sfx])
                    for nk in ("uL2", "uH1"):
                        R.ser[f"{prefix}_spurious_{nk}"] = series(lv, [o[1]["spurious"][nk] for o in out],
                                                                  out[-1][1]["spurious"]["magnitudes"][nk])
                    R.acc.q[f"{prefix}_TL2{sfx}_finest"] = out[-1][1]["TL2" + sfx]
                else:
                    R.acc.max(f"{prefix}_u_L2_max_over_levels", max(o[1]["spurious"]["uL2"] for o in out))
                    R.acc.max(f"{prefix}_pL2m_max_over_levels", max(o[1]["pL2m" + sfx] for o in out))
                    R.acc.max(f"{prefix}_TL2_max_over_levels", max(o[1]["TL2" + sfx] for o in out))
    return R.result()


# ---------------------------------------------------------------------------------------------- C1 / X1
def _cavity_mesh(N):
    k = int(round(math.log2(N / 8)))
    V, T, h, cl = build_mesh("unit", k)
    mesh, seg, att = make_mesh(V, T, cl)
    info = verify(mesh, V, T)
    return mesh, info, h


def _ratio(a, b):
    return float(a / b) if b != 0.0 else 0.0


def cavity_small_ra(params, workdir, quick):
    """C1 analytic_conduction_limit_cavity: Ra = 0 (Nu = 1, u = 0), the small-Ra scalings, and the point symmetry
    Nu_hot = Nu_cold at Ra = 1e5."""
    R = Run(params, quick)
    Pr = float(params.get("Pr", 0.71))
    N0 = 16 if quick else 32
    mesh, info, h = _cavity_mesh(N0)
    for form in ("lap", "sym"):
        sol, ok, rec = R.cavity(f"ra0_{form}_N{N0}", mesh, info, h, 0.0, Pr, form, False)
        if ok:
            R.acc.max("ra0_nu_minus_one_abs_max", max(abs(rec["Nu_hot"] - 1.0), abs(rec["Nu_cold"] - 1.0)))
            R.acc.max("ra0_umax_nodal", rec["umax_nodal"])
    recs = {}
    for Ra in (10.0, 100.0):
        sol, ok, rec = R.cavity(f"small_ra_{int(Ra)}_N{N0}", mesh, info, h, Ra, Pr, "lap", False)
        if ok:
            recs[Ra] = rec
    if 10.0 in recs and 100.0 in recs:
        # a vanishing denominator (no convection at all) is a ratio of 0: out of its window
        R.acc.q["small_ra_nu_ratio"] = _ratio(recs[100.0]["Nu_hot"] - 1.0, recs[10.0]["Nu_hot"] - 1.0)
        R.acc.q["small_ra_umax_ratio"] = _ratio(recs[100.0]["umax_nodal"], recs[10.0]["umax_nodal"])
        R.acc.q["small_ra_vleft_ra100"] = recs[100.0]["vleft"]
        R.acc.q["small_ra_nu_minus_one_ra10"] = recs[10.0]["Nu_hot"] - 1.0
    for N in _levels([16, 32], quick, 1):
        m2, i2, h2 = _cavity_mesh(N)
        for form in ("lap", "sym"):
            for supg in (False, True):
                sol, ok, rec = R.cavity(f"ra1e5_{form}_{'supg' if supg else 'gal'}_N{N}", m2, i2, h2, 1e5, Pr, form, supg)
                if ok:
                    R.acc.max("ra1e5_nu_hot_minus_cold_abs_max", abs(rec["Nu_hot"] - rec["Nu_cold"]))
    return R.result()


def xcode_natconv(params, workdir, quick):
    """X1 xcode_natconv_ra1e5: the differentially heated cavity at Ra = 1e5, Pr = 0.71, Galerkin, N = 32 and 64, lap and
    sym: the functionals Ekin, Diss, Tsq, vT, vleft, Nu_hot, Nu_cold compared with the cross-code reference."""
    R = Run(params, quick)
    Ra, Pr = float(params["Ra"]), float(params["Pr"])
    for N in _levels(params["N"], quick, 1):
        mesh, info, h = _cavity_mesh(N)
        for form in params["viscous_forms"]:
            sol, ok, rec = R.cavity(f"x1_{form}_N{N}", mesh, info, h, Ra, Pr, form, False)
            if ok:
                for name in ("Ekin", "Diss", "Tsq", "vT", "vleft", "Nu_hot", "Nu_cold"):
                    R.acc.q[f"{form}_{name}_N{N}"] = float(rec[name])
    return R.result()


# ---------------------------------------------------------------------------------------------- E1
def bous_balance(params, workdir, quick):
    """E1 cons_heat_balance: the discrete energy balance of the coupled system (v = 1) on the MMS fields."""
    R = Run(params, quick)
    P = params
    for key, axi in (("bous_sq", False), ("bous_axi", True)):
        fld = H.load_bous(P["fields"][key])
        dn = dirichlet_attrs(fld.data["T_boundaries"])
        parts_spec = [bd for bd in fld.data["T_boundaries"] if bd["type"] in ("neumann", "robin")]
        for form in ("lap", "sym"):
            for supg in (False, True):
                for i, N in enumerate(_levels(P["levels_N"], quick)):
                    k = int(round(math.log2(N / 8)))
                    V, T, h, cl = build_mesh("unit", k)
                    mesh, seg, att = make_mesh(V, T, cl)
                    info = verify(mesh, V, T)
                    sol, ok = R.solve_bous(f"{key}_{form}_{'supg' if supg else 'gal'}_N{N}", mesh, info, h, fld, form,
                                           supg=supg)
                    if not ok:
                        continue
                    ev = Evaluator(sol, axi, True)
                    qD = flux_of(R, sol, np.asarray(sol.full_residual_T()), dn)
                    parts = []
                    for bd in parts_spec:
                        on = S.edge_predicate(bd["name"], BOX)
                        if bd["type"] == "neumann":
                            parts.append((on, "neumann", S.compile_expr(bd["q_out"], fld.coords)))
                        else:
                            parts.append((on, "robin", float(bd["h"]["value"]), S.compile_expr(bd["T_ext"], fld.coords)))
                    nat = S.natural_outflow(ev, parts, axi)
                    src = S.integral(ev, lambda X, T, dT: fld.Q(X[..., 0], X[..., 1]), axi)
                    adv = H.cavity_functionals(ev, axi, a=fld.rho0 * fld.cp)["adv"]
                    R.acc.max(f"{key}_balance_defect_max", S.balance_defect(qD, nat, src, adv))
    return R.result()


# ---------------------------------------------------------------------------------------------- J1
def bous_taylor(params, workdir, quick):
    """J1 mms_boussinesq_newton_jacobian: Taylor tests of the coupled Jacobian (every block, with and without SUPG)."""
    R = Run(params, quick)
    P = params
    eps = [float(e) for e in P["eps"]]
    for key, axi in (("bous_sq", False), ("bous_axi", True)):
        fld = H.load_bous(P["fields"][key])
        V, T, h, cl = build_mesh("unit", 0)
        mesh, seg, att = make_mesh(V, T, cl)
        info = verify(mesh, V, T)
        for form in ("lap", "sym"):
            for supg, tag in ((False, "gal"), (True, "supg")):
                name = f"taylor_{key}_{form}_{tag}"
                sol, ok = R.solve_bous(f"{key}_{form}_{tag}_N{P['N']}", mesh, info, h, fld, form, supg=supg)
                if not ok:
                    continue
                x0 = np.asarray(sol.state())
                nu_, npr, nT = sol.layout()
                rng = np.random.default_rng(1)
                x = x0 + 0.05 * rng.standard_normal(len(x0))
                d = rng.standard_normal(len(x0))
                R0 = np.asarray(sol.residual(x))
                Jd = np.asarray(sol.jacobian_action(x, d))
                res = S.taylor_test(R0, lambda e, x=x, d=d, sol=sol: np.asarray(sol.residual(x + e * d)), Jd, eps)
                R.acc.q[f"{name}_median_slope"] = res["median_slope"]
                R.acc.q[f"{name}_n_slopes"] = res["n_slopes"]
    return R.result()
