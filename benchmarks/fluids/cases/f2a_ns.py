"""Steady Navier-Stokes cases (planar and axisymmetric with swirl): set-up, solve, Evaluator.

Every norm, identity, Newton-history and Taylor-test number is computed by the check module f2a_ns_check (imported as
C; it re-exports the planar Stokes mesh builders and nodal checks).  This module builds the case meshes with C's
builders, verifies the solver's mesh against them, solves through the public API axionfem.fluids.solve_navier_stokes
and wraps each discrete solution in an Evaluator (C's protocol).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two
levels only.

Failed solves are results: a solve that misses its gate is recorded in runs[] (converged, relative_residual, the
history) and only the quantities that need its fields are skipped, so the acceptance check reports the case as failed
(solver gate).

Conventions: form "lap" = mu grad u : grad v, "sym" = 2 mu eps:eps;
convection "standard" rho ((w.grad) u, v), "skew" (Temam) + (rho/2)((div w) u, v).  Axisymmetric: the API's
internal component order is (u_r, u_z, u_theta) for data, force and outputs; the check module's is
(u_r, u_theta, u_z): this runner permutes (PERM, an involution) both ways.  Boundary attributes of every rectangle:
1 bottom (y = y0 / z = z0), 2 right (x = x1 / r = r1), 3 top, 4 left (x = x0 / the axis r = 0 when on the axis).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

import f2a_ns_check as C
from axionfem import fluids

FORM = {"lap": "laplacian", "sym": "stress"}
TOL = 1e-9
SIDES = {"bottom": 1, "right": 2, "top": 3, "left": 4}
PERM = [0, 2, 1]          # (u_r, u_z, u_t) <-> (u_r, u_t, u_z)
N_PICARD = 3              # run block nonlinear_solver.start: Stokes, then 3 Picard steps, then Newton
PTC_RELEASE_DT = 1e12     # C1 run.parameters.ptc.release: "pure Newton once dt > 1e12" (fixed as text)


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


def _nl(params):
    ns = params.get("nonlinear_solver") or {}
    return float(ns.get("newton_rtol", 1e-12)), int(ns.get("newton_maxit", 30))


def tight_rtol(params):
    """The tight newton_rtol of M1's solver-sensitivity run: the FIXED run parameter
    run.parameters.solver_sensitivity.tight_newton_rtol (1e-13). There is no fallback: a missing key
    is an error, never the production newton_rtol (a tight run at the production tolerance would show nothing)."""
    try:
        return float(params["solver_sensitivity"]["tight_newton_rtol"])
    except (KeyError, TypeError) as exc:
        raise KeyError("run.parameters.solver_sensitivity.tight_newton_rtol is missing: M1's tight run has no "
                       "tolerance to run at (no fallback by design)") from exc


# ============================================================================================== evaluator, data
class Evaluator:
    """The discrete solution in the check module's protocol (one MPI rank)."""

    def __init__(self, sol, axi):
        self.sol, self.axi = sol, axi
        self.n_elements = int(sol.n_elements)
        self._el = np.arange(self.n_elements, dtype=np.int32)

    @staticmethod
    def _ref(ref_pts):
        return np.ascontiguousarray(ref_pts, dtype=float)

    def geometry(self, ref_pts):
        return self.sol.geometry(self._el, self._ref(ref_pts))

    def velocity(self, ref_pts):
        u, g = self.sol.velocity(self._el, self._ref(ref_pts))
        u, g = np.asarray(u), np.asarray(g)
        if self.axi:
            u, g = u[..., PERM], g[..., PERM, :]
        return u, g

    def pressure(self, ref_pts):
        return np.asarray(self.sol.pressure(self._el, self._ref(ref_pts)))


def check_data(fn, axi):
    """The fixed data in the check module's order, with the shared-node axis rule (u_r = u_t = 0 at r = 0, which the
    exact data have as their limit; their numpy spelling is not evaluated there for those components)."""
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
    """A check-order vector callable as the API wants it (internal component order in axisymmetry)."""
    if fn is None or not axi:
        return fn
    return lambda X, Y: np.ascontiguousarray(np.asarray(fn(X, Y), float)[..., PERM])


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


UNIT = rect_classify(0.0, 1.0, 0.0, 1.0)


def make_mesh(V, T, classify):
    V = np.ascontiguousarray(V, dtype=float)
    T = np.ascontiguousarray(T, dtype=np.int32)
    seg = np.array(C.boundary_edges(T), dtype=np.int32)
    att = np.array([classify(*(0.5 * (V[a] + V[b]))) for a, b in seg], dtype=np.int32)
    return fluids.Mesh.from_arrays(V, T, seg, att), seg, att


def verify(mesh, V_ref, T_ref, V_coarser=None):
    v = C.verify_mesh(mesh.vertices(), mesh.elements(), V_ref, T_ref, V_coarser=V_coarser)
    bad = [k for k in ("counts_match", "hash_match", "nested") if k in v and not v[k]]
    if bad:
        raise RuntimeError(f"the solver mesh does not match the fixed construction: {bad} ({v})")
    return v


def bc(attr, kind, value=None):
    d = {"attribute": attr, "kind": kind}
    if value is not None:
        d["value"] = value
    return d


class Acc:
    def __init__(self):
        self.q = {}

    def max(self, k, v):
        self.q[k] = max(self.q.get(k, -math.inf), float(v))

    def min(self, k, v):
        self.q[k] = min(self.q.get(k, math.inf), float(v))


def series(levels, errors, magnitude):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


# ============================================================================================== one case's state
class Run:
    """Runs, series, quantities and notes of one case; every solve goes through solve()."""

    def __init__(self, params, quick):
        self.P, self.quick = params, quick
        self.ident = core_identity()
        self.rtol, self.maxit = _nl(params)
        self.runs, self.ser, self.acc, self.notes = [], {}, Acc(), []
        self.failed = []

    def solve(self, level, mesh, info, h, mu, rho, force, bcs, form, conv, axi, family=None, refinement_level=None,
              **kw):
        opts = dict(form=FORM[form], convection=conv, geometry="axisymmetric" if axi else "planar",
                    vectorized=True, lu_threads=1, newton_rtol=self.rtol, newton_maxit=self.maxit,
                    n_picard=N_PICARD)
        opts.update(kw)
        t0 = time.time()
        sol = fluids.solve_navier_stokes(mesh, mu, rho, force, bcs, **opts)
        sec = time.time() - t0
        e = run_entry(level, sol, sec, h, info, family, refinement_level, form, conv, axi, opts)
        self.runs.append(e)
        ok = bool(sol.diagnostics["converged"])
        if not ok:
            self.failed.append(level)
        return sol, ok

    def result(self):
        me = self.ident
        if self.failed:
            self.notes.append(f"{len(self.failed)} solve(s) missed the gate (recorded; their field quantities skipped): "
                              + ", ".join(self.failed[:12]) + (" ..." if len(self.failed) > 12 else ""))
        self.notes.append(f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]})")
        return {"runs": self.runs, "series": self.ser, "quantities": self.acc.q, "notes": " ".join(self.notes)}


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


def run_entry(level, sol, sec, h, info, family, refinement_level, form, conv, axi, opts):
    d = sol.diagnostics
    g = d.get
    hist = [_jsonable(dict(x)) for x in sol.history]
    e = {"level": level,
         "mesh": {"h_nominal": h, "h_max": (info or {}).get("h_max"), "n_elements": int(sol.n_elements),
                  "dofs_field": int(g("dofs_velocity", 0)), "dofs_pressure": int(g("dofs_pressure", 0)),
                  "dofs_total": int(g("dofs_field", 0)), "files": [], "hash": (info or {}).get("hash"),
                  "all_boundary_triangles": (info or {}).get("all_boundary_triangles")},
         "solver": {"converged": bool(g("converged")), "relative_residual": float(g("relative_residual")),
                    "residual_lift": g("residual_lift"), "iterations": int(g("iterations", -1)),
                    "history": hist, "newton_steps": sum(1 for x in hist if x.get("kind") == "newton"),
                    "strategy": g("strategy"), "convection": g("convection", conv), "viscous_form": form,
                    "geometry": "axisymmetric" if axi else "planar", "kind": g("solver"),
                    "lu_backend": g("lu_backend"), "nullspace": g("nullspace"),
                    "nullspace_detected": bool(g("nullspace_detected")), "kernel_defect": g("kernel_defect"),
                    "flux_defect": g("flux_defect"), "flux_defect_abs": g("flux_defect_abs"),
                    "line_search_min_step_accepts": g("line_search_min_step_accepts"),
                    "quad_orders": _jsonable(g("quad_orders")), "ranks": g("ranks"), "threads": g("threads"),
                    "lu_threads": g("lu_threads"), "options": {k: _jsonable(v) for k, v in opts.items()}},
         "seconds": sec}
    if family is not None:
        e["mesh"]["family"] = family
        e["mesh"]["refinement_level"] = refinement_level
    return e


# ============================================================================================== shared pieces
def nodal_dirichlet(R, sol, attrs, g_check, axi, key="dirichlet_nodal_data_defect_max"):
    P, U = sol.boundary_nodes(list(attrs))
    U = np.asarray(U, float)
    if axi:
        U = U[:, PERM]
    if len(P):
        R.acc.max(key, C.dirichlet_nodal_defect(np.asarray(P, float), U, g_check))


def axis_nodal(R, sol, axis_attr, umag):
    P, U = sol.boundary_nodes([axis_attr])
    U = np.asarray(U, float)
    v = float(np.max(np.abs(U[:, [0, 2]]))) if len(U) else 0.0    # internal (u_r, u_z, u_t)
    R.acc.max("axis_nodal_ur_ut_max", v / max(float(umag), 1e-300))


def shared_mismatch(R, V, seg, att, data):
    """Mismatch of the per-attribute Dirichlet data at vertices where several Dirichlet attributes meet."""
    owners = {}
    for (i, j), a in zip(seg, att):
        if int(a) in data:
            for v in (int(i), int(j)):
                owners.setdefault(v, set()).add(int(a))
    shared = [(v, sorted(s)) for v, s in owners.items() if len(s) > 1]
    if shared:
        pts = np.array([V[v] for v, _ in shared])
        R.acc.max("dirichlet_shared_node_mismatch_max",
                  C.dirichlet_shared_node_mismatch(pts, [[data[a] for a in s] for _, s in shared]))
    else:
        R.acc.max("dirichlet_shared_node_mismatch_max", 0.0)


def mag_of(key, n):
    return n["p_magnitude"] if key.startswith("p") else n["u_magnitude"]


def put_series(R, prefix, rows, names):
    """rows: [(level, norms)] of the converged levels; names: {series suffix: norms key}."""
    if not rows:
        return
    for suf, key in names.items():
        R.ser[f"{prefix}_{suf}"] = series([lv for lv, _ in rows], [n[key] for _, n in rows],
                                          max(mag_of(key, n) for _, n in rows))


def family_meshes(fam, P, quick):
    """[(V, T, V_coarser, h, idx, tag)] of a fixed family named in run.parameters.meshes."""
    m = P["meshes"][fam]
    out = []
    if fam == "structured":
        Ns = _levels(m["N"], quick)
        for i, N in enumerate(Ns):
            V, T = C.structured_mesh(N)
            Vc = C.structured_mesh(Ns[i - 1])[0] if i else None
            out.append((V, T, Vc, m["h_nominal"][i], i, f"N{N}"))
    elif fam == "distorted":
        ks = _levels(m["refinements"], quick)
        for i, k in enumerate(ks):
            V, T = C.distorted_unit(k)
            Vc = C.distorted_unit(ks[i - 1])[0] if i else None
            out.append((V, T, Vc, m["h_nominal"][i], i, f"k{k}"))
    elif fam == "kovasznay_rect":
        ks = _levels(m["k"], quick)
        for i, k in enumerate(ks):
            V, T = C.rect_mesh(6, 8, k, -0.5, 1.0, -0.5, 1.5)
            Vc = C.rect_mesh(6, 8, ks[i - 1], -0.5, 1.0, -0.5, 1.5)[0] if i else None
            out.append((V, T, Vc, m["h_nominal"][i], i, f"k{k}"))
    elif fam == "couette_rect":
        ks = _levels(m["k"], quick)
        for i, k in enumerate(ks):
            V, T = C.rect_mesh(4, 4, k, 0.5, 1.0, 0.0, 0.5)
            Vc = C.rect_mesh(4, 4, ks[i - 1], 0.5, 1.0, 0.0, 0.5)[0] if i else None
            out.append((V, T, Vc, m["h_nominal"][i], i, f"k{k}"))
    else:
        raise ValueError(fam)
    return out


CLASSIFY = {"structured": UNIT, "distorted": UNIT, "kovasznay_rect": rect_classify(-0.5, 1.0, -0.5, 1.5),
            "couette_rect": rect_classify(0.5, 1.0, 0.0, 0.5)}
FAMTAG = {"structured": "struct", "distorted": "dist", "kovasznay_rect": "kov", "couette_rect": "cc"}


def field_mu(P, F):
    """A number when the fixed parameters give one (constant mu), else the field's callable."""
    mu = P.get("mu")
    return float(mu) if isinstance(mu, (int, float)) else F.mu


def field_force(F, form, axi):
    if F.data.get("exact_ns"):
        return None          # exact NS solutions: f = 0 (checked symbolically by the generator)
    return to_api(F.f(form), axi)


# ============================================================================================== MMS / analytic families
def _family(R, F, fam, form, conv, axi, dirichlet_sides, axis_side=None, pressure="mean_free", names=None):
    """Solve one (form, conv) over one fixed family; norms of the converged levels; nodal checks.  Returns
    [(level, norms, sol, ev)] of the converged levels."""
    P = R.P
    rho = float(P.get("rho", F.rho))
    mu = field_mu(P, F)
    g_chk = check_data(F.u, axi)
    g_api = to_api(g_chk, axi)
    bcs = [bc(SIDES[s], "dirichlet", g_api) for s in dirichlet_sides]
    if axis_side:
        bcs.append(bc(SIDES[axis_side], "axis"))
    tag = FAMTAG[fam]
    prefix = f"{form}_{conv}_{tag}"
    out, rows = [], []
    for (V, T, Vc, h, i, lab) in family_meshes(fam, P, R.quick):
        mesh, seg, att = make_mesh(V, T, CLASSIFY[fam])
        info = verify(mesh, V, T, Vc)
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve(lvl, mesh, info, h, mu, rho, field_force(F, form, axi), bcs, form, conv, axi,
                          family=prefix, refinement_level=i)
        if pressure == "mean_free":
            R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        else:
            R.acc.max("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        shared_mismatch(R, V, seg, att, {SIDES[s]: g_chk for s in dirichlet_sides})
        if not ok:
            continue
        ev = Evaluator(sol, axi)
        n = C.norms(ev, F, pressure)
        nodal_dirichlet(R, sol, [SIDES[s] for s in dirichlet_sides], g_chk, axi)
        if axis_side:
            axis_nodal(R, sol, SIDES[axis_side], n["u_magnitude"])
        rows.append((lvl, n))
        out.append((lvl, n, sol, ev, mesh, info, h, V, T))
    if names:
        put_series(R, prefix, rows, names)
    return out


def _newton_check(R, F, fam, level_index, forms, axi, dirichlet_sides, axis_side=None, tag="newton_check"):
    """Full Newton steps from the Stokes solution (no Picard, no line search) on one fixed level: the number of
    Newton steps per (form, conv) and the residual list [after the Stokes step, after each Newton step]."""
    P = R.P
    rho = float(P.get("rho", F.rho))
    mu = field_mu(P, F)
    g_api = to_api(check_data(F.u, axi), axi)
    bcs = [bc(SIDES[s], "dirichlet", g_api) for s in dirichlet_sides]
    if axis_side:
        bcs.append(bc(SIDES[axis_side], "axis"))
    meshes = family_meshes(fam, P, False)
    V, T, Vc, h, i, lab = meshes[level_index]
    mesh, seg, att = make_mesh(V, T, CLASSIFY[fam])
    info = verify(mesh, V, T, Vc)
    out = {}
    for form, conv in forms:
        sol, ok = R.solve(f"{tag}_{form}_{conv}_{lab}", mesh, info, h, mu, rho, field_force(F, form, axi), bcs, form,
                          conv, axi, n_picard=0, line_search=False)
        hist = list(sol.history)
        res = [float(x["residual"]) for x in hist]
        nsteps = sum(1 for x in hist if x["kind"] == "newton")
        out[(form, conv)] = {"residuals": res, "newton_steps": nsteps, "converged": ok}
        R.acc.max("newton_iterations_from_stokes_max", nsteps if ok else max(nsteps, R.maxit + 1))
    return out


STD_NAMES_P = {"uL2": "uL2", "uH1": "uH1", "pL2m": "pL2", "divL2": "divL2"}
STD_NAMES_AXI = {"uL2w": "uL2", "uH1w": "uH1", "pL2mw": "pL2", "divL2w": "divL2", "uL2_unw": "uL2_unw",
                 "uH1_unw": "uH1_unw", "pL2m_unw": "pL2_unw"}
COMBOS = [("lap", "standard"), ("sym", "standard"), ("lap", "skew"), ("sym", "skew")]
PLANAR_D = ("bottom", "right", "top", "left")
AXI_D = ("bottom", "right", "top")


def _combos(P):
    return [(f, c) for c in P.get("convective_forms", ["standard", "skew"]) for f in P.get("viscous_forms", ["lap", "sym"])
            if (f, c) in COMBOS]


# ---------------------------------------------------------------------------------------------- M1
def mms_planar(params, workdir, quick):
    """M1 mms_ns_steady_2d: all-Dirichlet planar MMS, variable mu, rho = 5/4; structured (4 form pairs) and distorted
    (lap/standard uH1, pL2m; sym/standard and sym/skew uL2, uH1, pL2m) families; Newton from Stokes at N = 32;
    Picard-only vs Newton at N = 32 (lap/standard); the tight solver-sensitivity run at the finest level."""
    R = Run(params, quick)
    P = params
    F = C.load_field(P["mms_data"])
    keep = {}
    for form, conv in _combos(P):
        out = _family(R, F, "structured", form, conv, False, PLANAR_D, names=STD_NAMES_P)
        keep[(form, conv)] = out
    dist = {("lap", "standard"): {"uH1": "uH1", "pL2m": "pL2"},
            ("sym", "standard"): {"uL2": "uL2", "uH1": "uH1", "pL2m": "pL2"},
            ("sym", "skew"): {"uL2": "uL2", "uH1": "uH1", "pL2m": "pL2"}}
    for (form, conv), names in dist.items():
        if (form, conv) in _combos(P):
            _family(R, F, "distorted", form, conv, False, PLANAR_D, names=names)
    # Newton from the Stokes solution, N = 32 (index of N = 32 in the structured family)
    nc = P.get("newton_check") or {}
    Ns = list(P["meshes"]["structured"]["N"])
    N_nc = int(nc.get("N", 32))
    if not quick and N_nc in Ns:
        forms = [tuple(x) for x in nc.get("forms", COMBOS)]
        _newton_check(R, F, "structured", Ns.index(N_nc), forms, False, PLANAR_D)
    # Picard only vs Newton (lap, standard), N = 32
    pc = P.get("picard_check") or {}
    N_pc = int(pc.get("N", 32))
    pform = tuple(pc.get("form", ("lap", "standard")))
    if not quick and N_pc in Ns:
        V, T, Vc, h, i, lab = family_meshes("structured", P, False)[Ns.index(N_pc)]
        mesh, seg, att = make_mesh(V, T, UNIT)
        info = verify(mesh, V, T, Vc)
        g_api = check_data(F.u, False)
        bcs = [bc(SIDES[s], "dirichlet", g_api) for s in PLANAR_D]
        solP, okP = R.solve(f"picard_only_{pform[0]}_{pform[1]}_N{N_pc}", mesh, info, h, field_mu(P, F),
                            float(P.get("rho", F.rho)), F.f(pform[0]), bcs, pform[0], pform[1], False,
                            strategy="picard")
        newton = next((x for x in keep.get(pform, []) if x[0] == f"{pform[0]}_{pform[1]}_struct_L{i}"), None)
        if okP and newton is not None:
            R.acc.max("picard_vs_newton_u_rel", C.relative_difference(newton[3], Evaluator(solP, False))["u_rel"])
    # solver sensitivity: finest sym/standard uL2 at newton_rtol vs the tight value
    ss = keep.get(("sym", "standard")) or []
    if ss:
        lvl, n, sol, ev, mesh, info, h, V, T = ss[-1]
        if lvl == f"sym_standard_struct_L{len(family_meshes('structured', P, quick)) - 1}":
            R.acc.q["sym_standard_struct_uL2_finest"] = n["uL2"]
            tr = tight_rtol(P)
            g_api = check_data(F.u, False)
            bcs = [bc(SIDES[s], "dirichlet", g_api) for s in PLANAR_D]
            solT, okT = R.solve("sym_standard_struct_tight", mesh, info, h, field_mu(P, F), float(P.get("rho", F.rho)),
                                F.f("sym"), bcs, "sym", "standard", False, newton_rtol=tr)
            if okT:
                R.acc.q["sym_standard_struct_uL2_finest_tight"] = C.norms(Evaluator(solT, False), F)["uL2"]
            # the tight run's own final relative residual (a check of the fixed acceptance bounds it by tight_rtol);
            # the run is in runs[] (level sym_standard_struct_tight), so the solver gate applies to it too
            R.acc.q["sym_standard_struct_finest_tight_relative_residual"] = float(
                solT.diagnostics["relative_residual"])
            R.notes.append(f"solver sensitivity: tight newton_rtol {tr:g} (run.parameters.solver_sensitivity."
                           f"tight_newton_rtol), final relative residual "
                           f"{float(solT.diagnostics['relative_residual']):.3g}")
    return R.result()


# ---------------------------------------------------------------------------------------------- M2
def mms_planar_traction(params, workdir, quick):
    """M2 mms_ns_steady_2d_traction: traction sigma_form n = g on x = 1 (u.n of both signs), Dirichlet elsewhere;
    pressure determined (raw comparison)."""
    R = Run(params, quick)
    F = C.load_field(params["mms_data"])
    names = {"uL2": "uL2", "uH1": "uH1", "pL2": "pL2", "divL2": "divL2"}
    for form, conv in _combos(params):
        _family_traction(R, F, form, conv, names)
    R.notes.append("Traction sigma_form n = g on x = 1 for both convective forms (no convective boundary term).")
    return R.result()


def _family_traction(R, F, form, conv, names):
    """M2 helper: the traction edge is the field's boundary 'x1' (attribute right); Dirichlet x0, y0, y1."""
    P = R.P
    rho = float(P.get("rho", F.rho))
    g = F.u
    dsides = ("bottom", "top", "left")
    bcs = [bc(SIDES[s], "dirichlet", g) for s in dsides] + [bc(SIDES["right"], "traction", F.traction(form, "x1"))]
    prefix = f"{form}_{conv}_struct"
    rows = []
    for (V, T, Vc, h, i, lab) in family_meshes("structured", P, R.quick):
        mesh, seg, att = make_mesh(V, T, UNIT)
        info = verify(mesh, V, T, Vc)
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve(lvl, mesh, info, h, F.mu, rho, F.f(form), bcs, form, conv, False, family=prefix,
                          refinement_level=i)
        R.acc.max("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        shared_mismatch(R, V, seg, att, {SIDES[s]: g for s in dsides})
        if not ok:
            continue
        n = C.norms(Evaluator(sol, False), F, "determined")
        nodal_dirichlet(R, sol, [SIDES[s] for s in dsides], g, False)
        rows.append((lvl, n))
    put_series(R, prefix, rows, names)


# ---------------------------------------------------------------------------------------------- M3
def mms_axi(params, workdir, quick):
    """M3 mms_ns_axi: axisymmetric with swirl, axis r = 0 (u_r = u_t = 0 strongly), Dirichlet r = 1, z = 0, 1;
    r-weighted and unweighted norms; distorted lap/standard, sym/skew, sym/standard; Newton from Stokes at N = 32."""
    R = Run(params, quick)
    P = params
    F = C.load_field(P["mms_data"])
    for form, conv in _combos(P):
        _family(R, F, "structured", form, conv, True, AXI_D, axis_side="left", names=STD_NAMES_AXI)
    dn = {"uL2w": "uL2", "uH1w": "uH1", "pL2mw": "pL2"}
    for form, conv in (("lap", "standard"), ("sym", "skew"), ("sym", "standard")):
        if (form, conv) in _combos(P):
            _family(R, F, "distorted", form, conv, True, AXI_D, axis_side="left", names=dn)
    nc = P.get("newton_check") or {}
    Ns = list(P["meshes"]["structured"]["N"])
    N_nc = int(nc.get("N", 32))
    if not quick and N_nc in Ns:
        _newton_check(R, F, "structured", Ns.index(N_nc), [tuple(x) for x in nc.get("forms", COMBOS)], True, AXI_D,
                      axis_side="left")
    R.notes.append("Axisymmetric: components permuted from the API's (u_r, u_z, u_theta) to the check module's "
                   "(u_r, u_theta, u_z); axis nodes from the solution's 'axis' attribute.")
    return R.result()


# ---------------------------------------------------------------------------------------------- A1
def kovasznay(params, workdir, quick):
    """A1 analytic_kovasznay_re40: exact NS solution, all-Dirichlet; Newton order and count at k = 2."""
    R = Run(params, quick)
    P = params
    F = C.load_field(P["mms_data"])
    for form, conv in [tuple(v) for v in P["variants"]]:
        _family(R, F, "kovasznay_rect", form, conv, False, PLANAR_D, names=STD_NAMES_P)
    nc = P.get("newton_check") or {}
    ks = list(P["meshes"]["kovasznay_rect"]["k"])
    k_nc = 2
    try:
        k_nc = int(str(nc.get("mesh", "k = 2")).split("=")[-1])
    except ValueError:
        pass
    if not quick and k_nc in ks:
        out = _newton_check(R, F, "kovasznay_rect", ks.index(k_nc), [tuple(x) for x in nc.get("forms", COMBOS[:2])],
                            False, PLANAR_D)
        for (form, conv), o in out.items():
            no = C.newton_order(o["residuals"])
            med = no["p_median"] if o["converged"] and math.isfinite(no["p_median"]) else 0.0
            R.acc.min("newton_order_p_median", med)
    return R.result()


# ---------------------------------------------------------------------------------------------- A2, A3
def burgers_axi(params, workdir, quick):
    """A2 analytic_burgers_vortex_axi: exact axisymmetric NS solution with swirl, axis, Dirichlet elsewhere."""
    R = Run(params, quick)
    P = params
    F = C.load_field(P["mms_data"])
    names = {"uL2w": "uL2", "uH1w": "uH1", "pL2mw": "pL2", "uL2_unw": "uL2_unw", "pL2m_unw": "pL2_unw"}
    for form, conv in [tuple(v) for v in P["variants"]]:
        _family(R, F, "structured", form, conv, True, AXI_D, axis_side="left", names=names)
    return R.result()


def couette_axi(params, workdir, quick):
    """A3 analytic_circular_couette_ns_axi: circular Couette between r = 1/2 and 1 (no axis), all-Dirichlet; pressure
    items only."""
    R = Run(params, quick)
    P = params
    F = C.load_field(P["mms_data"])
    names = {"pL2mw": "pL2", "pL2m_unw": "pL2_unw"}
    for form, conv in [tuple(v) for v in P["variants"]]:
        _family(R, F, "couette_rect", form, conv, True, PLANAR_D, names=names)
    return R.result()


# ---------------------------------------------------------------------------------------------- E1
def energy_identity(params, workdir, quick):
    """E1 cons_ns_energy_identity: discrete energy identities, planar N = 8 and axisymmetric N = 32, four form pairs;
    positive controls (other convective form, other viscous form, swirl pair)."""
    R = Run(params, quick)
    P = params
    mu = float(P["mu"])
    rho = float(P["rho"])
    for geo, axi in (("planar", False), ("axisymmetric", True)):
        F = C.load_field(P["fields"][geo])
        N = int(P["meshes"][geo]["N"][0])
        h = float(P["meshes"][geo]["h_nominal"][0])
        V, T = C.structured_mesh(N)
        mesh, seg, att = make_mesh(V, T, UNIT)
        info = verify(mesh, V, T)
        dsides = AXI_D if axi else PLANAR_D
        zero = check_data(F.u, axi)
        bcs = [bc(SIDES[s], "dirichlet", to_api(zero, axi)) for s in dsides]
        if axi:
            bcs.append(bc(SIDES["left"], "axis"))
        tag = "axi" if axi else "planar"
        for form, conv in [tuple(v) for v in P["variants"]]:
            f_chk = F.f(form)
            sol, ok = R.solve(f"{tag}_{form}_{conv}_N{N}", mesh, info, h, mu, rho, to_api(f_chk, axi), bcs, form,
                              conv, axi)
            if not ok:
                continue
            ev = Evaluator(sol, axi)
            e = C.energy_identity(ev, form, conv, rho, mu, f_chk, axisymmetric=axi)
            k = f"{tag}_{form}_{conv}"
            R.acc.q[f"energy_identity_defect_{k}"] = e["defect"]
            R.acc.q[f"energy_identity_other_convective_defect_{k}"] = e["defect_other_convective_form"]
            R.acc.q[f"energy_identity_other_viscous_defect_{k}"] = e["defect_other_viscous_form"]
            if axi:
                R.acc.q[f"swirl_pair_power_rel_{k}"] = abs(C.swirl_pair_power(ev, rho)) / abs(e["power_input"])
            nodal_dirichlet(R, sol, [SIDES[s] for s in dsides], zero, axi)
    return R.result()


# ---------------------------------------------------------------------------------------------- J1
def jacobian_taylor(params, workdir, quick):
    """J1 mms_ns_newton_jacobian: Taylor tests of R(x) = 0's Jacobian from the public residual/Jacobian-action API at
    x = Newton solution (N = 8) + 0.1 xi, d = xi' (xi, xi' standard normal over the free DOFs, a fresh
    PCG64(seed) generator per configuration: first xi, then xi'); Euclidean norms of the raw residual vectors (as
    in a standard Taylor test)."""
    R = Run(params, quick)
    P = params
    tay = P["taylor"]
    eps = [float(e) for e in tay["eps"]]
    seed = int(tay["seed"])
    N = int(P["meshes"]["structured"]["N"][0])
    h = float(P["meshes"]["structured"]["h_nominal"][0])
    for geo, axi in (("planar", False), ("axisymmetric", True)):
        F = C.load_field(P["fields"][geo])
        V, T = C.structured_mesh(N)
        mesh, seg, att = make_mesh(V, T, UNIT)
        info = verify(mesh, V, T)
        dsides = AXI_D if axi else PLANAR_D
        g_api = to_api(check_data(F.u, axi), axi)
        bcs = [bc(SIDES[s], "dirichlet", g_api) for s in dsides]
        if axi:
            bcs.append(bc(SIDES["left"], "axis"))
        tag = "axi" if axi else "planar"
        for form, conv in [tuple(v) for v in P["variants"]]:
            sol, ok = R.solve(f"{tag}_{form}_{conv}_N{N}", mesh, info, h, F.mu, F.rho, to_api(F.f(form), axi), bcs,
                              form, conv, axi)
            if not ok:
                continue
            x0 = np.asarray(sol.state(), float)
            rng = np.random.Generator(np.random.PCG64(seed))
            x = x0 + 0.1 * rng.standard_normal(x0.size)
            d = rng.standard_normal(x0.size)
            R0 = np.asarray(sol.residual(x), float)
            Jd = np.asarray(sol.jacobian_action(x, d), float)
            Rs = [np.asarray(sol.residual(x + e * d), float) for e in eps]
            t = C.taylor_test(R0, Rs, Jd, eps)
            k = f"{tag}_{form}_{conv}"
            R.acc.q[f"taylor_slope_r2_{k}"] = t["slope_r2"]
            R.acc.q[f"taylor_r2_over_eps2_spread_{k}"] = t["r2_over_eps2_spread"]
            R.acc.q[f"taylor_slope_r1_{k}"] = t["slope_r1"]
            R.runs[-1]["taylor"] = {kk: t[kk] for kk in ("slope_r1", "slope_r2", "r2_over_eps2_spread",
                                                          "r2_over_r1_min_eps", "r1", "r2")}
            R.runs[-1]["taylor"]["n_free"] = int(x0.size)
            R.runs[-1]["taylor"]["layout"] = list(map(int, sol.layout()))
    return R.result()


# ---------------------------------------------------------------------------------------------- cavities
def lid_data():
    """Singular lid: u = (1, 0) at every lid node except the lid corner vertices (wall value 0), u = 0 elsewhere."""
    def g(X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        top = (np.abs(Y - 1.0) < 1e-12) & (X > 1e-12) & (X < 1.0 - 1e-12)
        return np.stack([np.where(top, 1.0, 0.0), 0.0 * X], axis=-1)
    return g


def _cavity(R, N, Re_list, form="lap", conv="standard", label=None, **kw):
    V, T = C.structured_mesh(N)
    mesh, seg, att = make_mesh(V, T, UNIT)
    info = verify(mesh, V, T)
    g = lid_data()
    bcs = [bc(SIDES[s], "dirichlet", g) for s in PLANAR_D]
    Re = float(Re_list[-1])
    opts = dict(n_picard=0)
    if len(Re_list) > 1:
        opts.update(continuation_re=[float(r) for r in Re_list], re=Re)
    opts.update(kw)
    sol, ok = R.solve(label, mesh, info, 1.0 / N, 1.0 / Re, 1.0, None, bcs, form, conv, False, **opts)
    shared_mismatch(R, V, seg, att, {SIDES[s]: g for s in PLANAR_D})
    if ok:
        nodal_dirichlet(R, sol, [SIDES[s] for s in PLANAR_D], g, False)
    return sol, ok


def cavity_continuation(params, workdir, quick):
    """C1 selfconv_ns_cavity_continuation: singular-lid cavity, Re = 1000, lap/standard, N = 32, 64: two Re-
    continuation schedules and PTC from the Stokes solution reach the same steady state; Newton from Stokes with the
    dual-norm Armijo line search at Re_ls on N = 32."""
    R = Run(params, quick)
    P = params
    Ns = _levels(P["meshes"]["structured"]["N"], quick, 1)
    rc = P["re_continuation"]
    ptc = P["ptc"]
    for N in Ns:
        sa, oa = _cavity(R, N, rc["A"], label=f"cont_A_N{N}")
        sb, ob = _cavity(R, N, rc["B"], label=f"cont_B_N{N}")
        sp, op = _cavity(R, N, [rc["A"][-1]], label=f"ptc_N{N}", strategy="ptc", ptc_dt0=float(ptc["dt0"]),
                         ptc_release_dt=PTC_RELEASE_DT)
        if oa and ob:
            R.acc.q[f"continuation_schedules_u_rel_N{N}"] = C.relative_difference(Evaluator(sa, False),
                                                                                  Evaluator(sb, False))["u_rel"]
        if oa and op:
            d = C.relative_difference(Evaluator(sa, False), Evaluator(sp, False))
            R.acc.q[f"ptc_vs_continuation_u_rel_N{N}"] = d["u_rel"]
            R.acc.q[f"ptc_vs_continuation_p_rel_N{N}"] = d["p_rel"]
    ls = P["line_search"]
    Re_ls = float(ls["Re_ls"])
    N_ls = 32
    sl, ol = _cavity(R, N_ls, [Re_ls], label=f"line_search_Re{int(Re_ls)}_N{N_ls}", line_search=True)
    hist = list(sl.history)
    r = [float(x["residual"]) for x in hist]
    steps = [float(x["step"]) for x in hist[1:]]
    R.acc.q["line_search_armijo_violations"] = C.line_search_violations(r, steps)
    R.acc.q["line_search_run_final_residual"] = float(sl.diagnostics["relative_residual"])
    R.notes.append(f"Line-search run: Newton from the Stokes solution at Re = {Re_ls:g}, N = {N_ls}, "
                   f"{len(steps)} steps, minimum-step acceptances {sl.diagnostics.get('line_search_min_step_accepts')}.")
    return R.result()


def xcode_cavity(params, workdir, quick):
    """X1 xcode_cavity_re1000: singular-lid cavity at Re = 1000 by Re continuation, N = 64, 128, four form pairs;
    Ekin, Diss, Enst (f2a_ns_check.functionals) against the fixed cross-code reference."""
    R = Run(params, quick)
    P = params
    Ns = _levels(P["meshes"]["structured"]["N"], quick, 1)
    mu = float(P["mu"])
    for N in Ns:
        for form, conv in [tuple(v) for v in P["variants"]]:
            sol, ok = _cavity(R, N, P["continuation"], form=form, conv=conv, label=f"{form}_{conv}_N{N}")
            if not ok:
                continue
            fun = C.functionals(Evaluator(sol, False), form, mu)
            for k in ("Ekin", "Diss", "Enst"):
                R.acc.q[f"{form}_{conv}_{k}_N{N}"] = fun[k]
    return R.result()
