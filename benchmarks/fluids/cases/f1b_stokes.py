"""F1b axisymmetric Stokes with swirl: the case runner (set-up, solve, Evaluator).  Every norm, reaction,
residual, energy and check value is computed by the check module f1b_stokes_check (imported as C); this
module only builds the fixed meshes with C's builders, verifies the solver's mesh against them, solves through the
public API axionfem.fluids (geometry="axisymmetric") and wraps each discrete solution in an Evaluator (C's protocol:
n_elements, geometry, velocity, pressure; meridional (r, z) = mesh (x, y), components (u_r, u_z, u_theta)).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first levels only.

Conventions: form "lap" = mu grad u : grad v r, "sym" = 2 mu eps:eps r;
traction data g = sigma_form n of the fixed field with the OUTWARD GEOMETRIC normal of the (possibly curved,
isoparametric) boundary at each quadrature point (the API's normal_dependent traction); the axis carries
u_r = u_theta = 0 strongly at every P2 node (the solver's "axis" condition).  The axis and Dirichlet nodal checks
locate their nodes from the Evaluator (C.axis_nodal_max_ev, C.dirichlet_nodal_defect_ev): no node list from here.
"""
from __future__ import annotations

import functools
import math
import os
import time

import numpy as np

import f1b_stokes_check as C
from axionfem import fluids

FORM = {"lap": "laplacian", "sym": "stress"}
TOL = 1e-9
# boundary attributes of the rectangular meridional domains (square, annulus, pipe)
RECT = {"z0": 1, "rmax": 2, "zmax": 3, "rmin": 4}
# sphere family
SPH = {"axis_top": 1, "axis_bottom": 2, "sphere": 3, "outer": 4}


# ============================================================================================== small helpers
def _levels(seq, quick, n=2):
    seq = list(seq)
    return seq[:n] if quick else seq


class Evaluator:
    """The discrete solution in the check module's protocol (one MPI rank)."""

    def __init__(self, sol):
        self.sol = sol
        self.n_elements = int(sol.n_elements)
        self._el = np.arange(self.n_elements, dtype=np.int32)

    @staticmethod
    def _ref(ref_pts):
        return np.ascontiguousarray(ref_pts, dtype=float)

    def geometry(self, ref_pts):
        return self.sol.geometry(self._el, self._ref(ref_pts))

    def velocity(self, ref_pts):
        return self.sol.velocity(self._el, self._ref(ref_pts))

    def pressure(self, ref_pts):
        return self.sol.pressure(self._el, self._ref(ref_pts))

    def p_at_vertex(self, V, T, point):
        """p_h at a vertex of the fixed mesh (V, T) (evaluated in one element that has it)."""
        k = int(np.argmin(np.hypot(V[:, 0] - point[0], V[:, 1] - point[1])))
        if math.hypot(*(V[k] - np.asarray(point, float))) > 1e-12:
            raise RuntimeError(f"{point} is not a vertex of the mesh")
        e = int(np.nonzero((np.asarray(T) == k).any(axis=1))[0][0])
        loc = int(np.nonzero(np.asarray(T)[e] == k)[0][0])
        ref = np.array([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][loc]])
        return float(self.sol.pressure(np.array([e], dtype=np.int32), ref)[0, 0])


def rect_classify(r0, r1, z0, z1):
    def f(x, y):
        if abs(y - z0) < TOL:
            return RECT["z0"]
        if abs(x - r1) < TOL:
            return RECT["rmax"]
        if abs(y - z1) < TOL:
            return RECT["zmax"]
        if abs(x - r0) < TOL:
            return RECT["rmin"]
        raise RuntimeError(f"boundary edge at ({x}, {y}) is on no side of the rectangle")
    return f


def sphere_classify(x, y):
    """Edge midpoints (of the straight chords): the axis above/below the sphere, the sphere rho = 1, the outer rho = 4."""
    if abs(x) < 1e-12:
        return SPH["axis_top"] if y > 0 else SPH["axis_bottom"]
    rho = math.hypot(x, y)
    if rho < 2.5:
        return SPH["sphere"]
    return SPH["outer"]


def make_mesh(V, T, classify, X6=None):
    V = np.ascontiguousarray(V, dtype=float)
    T = np.ascontiguousarray(T, dtype=np.int32)
    edges = C.boundary_edges(T)
    seg = np.array(edges, dtype=np.int32)
    att = np.array([classify(*(0.5 * (V[a] + V[b]))) for a, b in edges], dtype=np.int32)
    m = fluids.Mesh.from_arrays(V, T, seg, att, p2_nodes=None if X6 is None else np.ascontiguousarray(X6, float))
    return m, seg, att


def verify(mesh, V_ref, T_ref, V_coarser=None, X6_ref=None):
    """The solver's mesh against the fixed construction (counts, hash, nesting, P2 nodes); raises on mismatch."""
    v = C.verify_mesh(mesh.vertices(), mesh.elements(), V_ref, T_ref, V_coarser=V_coarser,
                      X6=None if X6_ref is None else mesh.p2_nodes(), X6_ref=X6_ref)
    bad = [k for k in ("counts_match", "hash_match", "nested", "p2_hash_match") if k in v and not v[k]]
    if bad:
        raise RuntimeError(f"the solver mesh does not match the fixed construction: {bad} ({v})")
    return v


class NotConverged(Exception):
    """A solve that did not pass its own gate: the case stops and reports that run (the solver gate then fails)."""

    def __init__(self, entry):
        super().__init__(f"solve {entry['level']} did not converge (relative residual "
                         f"{entry['solver']['relative_residual']:.3e})")
        self.entry = entry


def solve(mesh, mu, f, bcs, form, **kw):
    t0 = time.time()
    sol = fluids.solve_stokes(mesh, mu, f, bcs, form=FORM[form], geometry="axisymmetric", vectorized=True, lu_threads=1, **kw)
    sec = time.time() - t0
    if not sol.diagnostics["converged"]:
        raise NotConverged(run_entry("not_converged", sol, sec, None, {"n_elements": int(sol.n_elements)}))
    return sol, sec


def case(fn):
    """Every case function: a solve that fails its own gate ends the case with that run reported."""

    @functools.wraps(fn)
    def wrapped(params, workdir, quick):
        try:
            return fn(params, workdir, quick)
        except NotConverged as e:
            return {"runs": [e.entry], "series": {}, "quantities": {}, "notes": f"stopped: {e}"}
    return wrapped


def run_entry(level, sol, sec, h, mesh_info, family=None, refinement_level=None, extra=None):
    d = sol.diagnostics
    e = {"level": level,
         "mesh": {"h_nominal": h, "h_max": mesh_info.get("h_max"), "n_elements": int(mesh_info["n_elements"]),
                  "dofs_field": int(d["dofs_velocity"]), "dofs_pressure": int(d["dofs_pressure"]),
                  "dofs_total": int(d["dofs_field"]), "files": [],
                  "hash": mesh_info.get("hash"), "all_boundary_triangles": mesh_info.get("all_boundary_triangles")},
         "solver": {"converged": bool(d["converged"]), "relative_residual": float(d["relative_residual"]),
                    "iterations": int(d["iterations"]), "kind": d["solver"], "nullspace": d["nullspace"],
                    "nullspace_detected": bool(d["nullspace_detected"]), "kernel_defect": d["kernel_defect"],
                    "flux_defect": d["flux_defect"], "flux_defect_abs": d["flux_defect_abs"],
                    "residual_momentum": d["residual_momentum"], "residual_continuity": d["residual_continuity"],
                    "lu_residual_before_refinement": d["lu_residual_before_refinement"],
                    "lu_refine_steps": d["lu_refine_steps"], "grad_div": d["grad_div"], "ranks": d["ranks"],
                    "threads": d["threads"], "lu_threads": d["lu_threads"],
                    "geometry": "axisymmetric"},
         "seconds": sec}
    if family is not None:
        e["mesh"]["family"] = family
        e["mesh"]["refinement_level"] = refinement_level
    if extra:
        e.update(extra)
    return e


class Acc:
    """max/min accumulators for check quantities."""

    def __init__(self):
        self.q = {}

    def max(self, k, v):
        self.q[k] = max(self.q.get(k, -math.inf), float(v))

    def min(self, k, v):
        self.q[k] = min(self.q.get(k, math.inf), float(v))


def series(levels, errors, magnitude):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


def vseries(levels, values):
    return {"levels": list(levels), "values": [float(v) for v in values]}


def tr_value(field, form):
    """Normal-dependent traction data g(R, Z, n_r, n_z) = sigma_form n of the fixed field."""
    g = field.traction(form)
    return lambda X, Y, NX, NY: g(X, Y, np.stack([NX, NY], axis=-1))


def bc(attr, kind, value=None, normal_dependent=False):
    d = {"attribute": attr, "kind": kind}
    if value is not None:
        d["value"] = value
    if normal_dependent:
        d["normal_dependent"] = True
    return d


def nodal_checks(acc, ev, V, T, scale, dirichlet_pred, g):
    """Axis nodal check (if the mesh touches r = 0) and the Dirichlet nodal-data defect, both located by the check
    module from the Evaluator.  dirichlet_pred(Pa, Pm, Pb): the edge belongs to a Dirichlet attribute (by the
    fixed geometry of its end vertices); g(R, Z): the fixed data on those edges."""
    if np.any(np.abs(np.asarray(V)[:, 0]) < 1e-12):
        acc.max("axis_nodal_max_rel", C.axis_nodal_max_ev(ev, V, T, scale))
    acc.max("dirichlet_nodal_data_defect_max", C.dirichlet_nodal_defect_ev(ev, dirichlet_pred, g))


def boundary_edge_pred(V, T, side_of_edge, kinds):
    """Edge predicate for C.dirichlet_nodal_defect_ev: (Pa, Pm, Pb) is a boundary edge of (V, T) whose side
    (side_of_edge(midpoint of the vertices)) is one of `kinds`."""
    vid = {tuple(np.round(v, 10) + 0.0): k for k, v in enumerate(np.asarray(V))}
    bset = {(min(i, j), max(i, j)) for (i, j) in C.boundary_edges(T)}

    def pred(Pa, Pm, Pb):
        ka, kb = vid.get(tuple(np.round(Pa, 10) + 0.0)), vid.get(tuple(np.round(Pb, 10) + 0.0))
        if ka is None or kb is None or (min(ka, kb), max(ka, kb)) not in bset:
            return False
        return side_of_edge(*(0.5 * (np.asarray(Pa) + np.asarray(Pb)))) in kinds
    return pred


def shared_mismatch(acc, V, seg, att, data):
    """Mismatch of the per-attribute Dirichlet data at vertices where several Dirichlet attributes meet."""
    owners = {}
    for (i, j), a in zip(seg, att):
        if int(a) in data:
            for v in (int(i), int(j)):
                owners.setdefault(v, set()).add(int(a))
    shared = [(v, sorted(s)) for v, s in owners.items() if len(s) > 1]
    if shared:
        pts = np.array([V[v] for v, _ in shared])
        acc.max("dirichlet_shared_node_mismatch_max",
                C.dirichlet_shared_node_mismatch(pts, [[data[a] for a in s] for _, s in shared]))
    else:
        acc.max("dirichlet_shared_node_mismatch_max", 0.0)


def outward_rect_edges(V, T, r0, r1, z0, z1, skip_axis=True):
    """(Pa, Pm, Pb, n) per straight boundary edge of a rectangle, for C.flux_defect_r (axis edges skipped)."""
    out = []
    for (i, j) in C.boundary_edges(T):
        Pa, Pb = V[i], V[j]
        if skip_axis and abs(Pa[0]) < 1e-12 and abs(Pb[0]) < 1e-12:
            continue
        side = rect_classify(r0, r1, z0, z1)(*(0.5 * (Pa + Pb)))
        n = {RECT["z0"]: (0.0, -1.0), RECT["rmax"]: (1.0, 0.0), RECT["zmax"]: (0.0, 1.0),
             RECT["rmin"]: (-1.0, 0.0)}[side]
        out.append((Pa, 0.5 * (Pa + Pb), Pb, np.array(n)))
    return out


def mag_key(k):
    return "ut_magnitude" if k.startswith("ut") else ("p_magnitude" if k.startswith("p") else "u_magnitude")


def put_series(ser, prefix, lv, rows, keys):
    for k in keys:
        ser[f"{prefix}_{k}"] = series(lv, [r[k] for r in rows], max(r[mag_key(k)] for r in rows))


def square_meshes(msh, quick):
    """[(V, T, V_coarser, h, idx)] of the structured family."""
    Ns = _levels(msh["structured"]["N"], quick)
    out = []
    for i, N in enumerate(Ns):
        V, T = C.square_mesh(N)
        out.append((V, T, C.square_mesh(Ns[i - 1])[0] if i else None, msh["structured"]["h_nominal"][i], i, N))
    return out


SQUARE = rect_classify(0.0, 1.0, 0.0, 1.0)


# ============================================================================================== M1
@case
def mms_dirichlet(params, workdir, quick):
    """M1: all-Dirichlet + axis, swirl, variable mu, both forms, structured and distorted families; flux defect solver
    vs check; the identity sym(gamma=0) == lap(gamma=mu) at constant mu; the tightened-solver sensitivity."""
    P = params
    F = C.load_field(P["mms_data"])
    Fc = C.load_field(P["mms_data_identity"])
    msh = P["meshes"]
    runs, ser, acc, notes = [], {}, Acc(), []
    ks = _levels(msh["distorted"]["refinements"], quick)
    dpred_cache = {}

    def bcs(field):
        return [bc(RECT["rmin"], "axis")] + [bc(RECT[s], "dirichlet", field.u) for s in ("z0", "rmax", "zmax")]

    def do(form, V, T, Vc, h, fam, idx, tag, field=F, gamma=0.0, tight=False):
        mesh, seg, att = make_mesh(V, T, SQUARE)
        info = verify(mesh, V, T, Vc)
        sol, sec = solve(mesh, field.mu, field.f(form), bcs(field), form, grad_div=gamma,
                         min_refine_steps=2 if tight else 0)
        lvl = f"{tag}_L{idx}"
        runs.append(run_entry(lvl, sol, sec, h, info, family=fam, refinement_level=idx))
        acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        ev = Evaluator(sol)
        key = (len(V), fam)
        if key not in dpred_cache:
            dpred_cache[key] = boundary_edge_pred(V, T, SQUARE, {RECT["z0"], RECT["rmax"], RECT["zmax"]})
        shared_mismatch(acc, V, seg, att, {RECT[s]: field.u for s in ("z0", "rmax", "zmax")})
        return lvl, ev, sol, dpred_cache[key]

    for form in P["forms"]:
        lv, rows = [], []
        for (V, T, Vc, h, i, N) in square_meshes(msh, quick):
            lvl, ev, sol, dp = do(form, V, T, Vc, h, f"{form}_struct", i, f"{form}_struct")
            n = C.norms(ev, F, "mean_free")
            nodal_checks(acc, ev, V, T, n["u_magnitude"], dp, F.u)
            if N <= 16:  # the solver's r-weighted flux defect against the check module's
                d, _ = C.flux_defect_r(outward_rect_edges(V, T, 0, 1, 0, 1), F.u)
                acc.max("flux_defect_solver_vs_check_rel", abs(sol.diagnostics["flux_defect_abs"] - abs(d)) / abs(d))
            lv.append(lvl)
            rows.append(n)
        put_series(ser, f"{form}_struct", lv, rows,
                   ["uL2r", "uH1r", "pL2r", "uL2", "uH1", "pL2", "utL2r", "utH1r", "divL2r"])
        if form == "sym":
            acc.q["sym_struct_uL2r_finest"] = rows[-1]["uL2r"]
        lv, rows = [], []
        dd = msh["distorted"]
        for i, k in enumerate(ks):
            V, T = C.square_distorted(k)
            Vc = C.square_distorted(ks[i - 1])[0] if i else None
            lvl, ev, sol, dp = do(form, V, T, Vc, dd["h_nominal"][i], f"{form}_dist", i, f"{form}_dist")
            n = C.norms(ev, F, "mean_free")
            nodal_checks(acc, ev, V, T, n["u_magnitude"], dp, F.u)
            lv.append(lvl)
            rows.append(n)
        put_series(ser, f"{form}_dist", lv, rows, ["uL2r", "uH1r", "pL2r"])
    # tightened direct solve (2 extra LU refinement steps) at the finest structured level
    if "sym" in P["forms"]:
        V, T, Vc, h, i, N = square_meshes(msh, quick)[-1]
        _, ev, _, _ = do("sym", V, T, Vc, h, "sym_struct_tight", i, "sym_struct_tight", tight=True)
        acc.q["sym_struct_uL2r_finest_tight"] = C.norms(ev, F, "mean_free")["uL2r"]
    # identity: sym (gamma = 0) == lap (gamma = mu) at constant mu, all Dirichlet + axis
    idn = P["identity"]
    mu0 = float(idn["mu"])
    Ns = _levels(msh["structured"]["N"], quick)
    for N in idn["N"]:
        if quick and N > Ns[-1]:
            continue
        V, T = C.square_mesh(N)
        _, es, _, dp = do("sym", V, T, None, 1.0 / N, None, 0, f"identity_sym_N{N}", field=Fc)
        _, el, _, _ = do("lap", V, T, None, 1.0 / N, None, 0, f"identity_lap_gd_N{N}", field=Fc, gamma=mu0)
        r = C.relative_l2r_difference(es, el)
        acc.max("identity_sym_vs_lap_graddiv_mu_u_rel_max", r["u_rel"])
        acc.max("identity_sym_vs_lap_graddiv_mu_p_rel_max", r["p_rel"])
        nodal_checks(acc, es, V, T, C.norms(es, Fc, "mean_free")["u_magnitude"], dp, Fc.u)
    notes.append("Enclosed (axis + Dirichlet): the continuity right-hand side is projected onto 1-perp (the r-weighted "
                 "flux defect of the interpolated data is reported per run); the mean-value multiplier removes the "
                 "constant; relative_residual is that of the consistent system.")
    return {"runs": runs, "series": ser, "quantities": acc.q, "notes": " ".join(notes)}


# ============================================================================================== M2, M3
def _open_family(P, quick, fields_forms, bcs_of, prefix_of, keys_of, dir_sides):
    msh = P["meshes"]
    runs, ser, acc = [], {}, Acc()
    for field, form, tag in fields_forms:
        lv, rows = [], []
        for (V, T, Vc, h, i, N) in square_meshes(msh, quick):
            mesh, seg, att = make_mesh(V, T, SQUARE)
            info = verify(mesh, V, T, Vc)
            sol, sec = solve(mesh, field.mu, field.f(form), bcs_of(field, form), form)
            lvl = f"{prefix_of(form, tag)}_L{i}"
            runs.append(run_entry(lvl, sol, sec, h, info, family=prefix_of(form, tag), refinement_level=i))
            acc.max("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
            ev = Evaluator(sol)
            n = C.norms(ev, field, "determined")
            nodal_checks(acc, ev, V, T, n["u_magnitude"], boundary_edge_pred(V, T, SQUARE, dir_sides), field.u)
            shared_mismatch(acc, V, seg, att, {s: field.u for s in dir_sides})
            lv.append(lvl)
            rows.append(n)
        put_series(ser, prefix_of(form, tag), lv, rows, keys_of(tag))
    return {"runs": runs, "series": ser, "quantities": acc.q}


@case
def mms_traction(params, workdir, quick):
    """M2: traction sigma_form n on r = 1 and z = 1, Dirichlet on z = 0, axis; variable mu and mu = 1."""
    P = params
    F = C.load_field(P["mms_data"])
    Fc = C.load_field(P["mms_data_constmu"])

    def bcs(field, form):
        return [bc(RECT["rmin"], "axis"), bc(RECT["z0"], "dirichlet", field.u),
                bc(RECT["rmax"], "traction", tr_value(field, form), True),
                bc(RECT["zmax"], "traction", tr_value(field, form), True)]
    ff = [(fld, form, tag) for fld, tag in ((F, ""), (Fc, "constmu")) for form in P["forms"]]
    out = _open_family(P, quick, ff, bcs,
                       lambda form, tag: f"{form}_struct" + (f"_{tag}" if tag else ""),
                       lambda tag: ["uL2r", "uH1r", "pL2r", "uL2", "pL2", "utL2r"] if not tag
                       else ["uL2r", "pL2r", "utL2r"], {RECT["z0"]})
    out["notes"] = "Traction data with the geometric outward normal; pressure determined (raw comparison)."
    return out


@case
def mms_do_nothing(params, workdir, quick):
    """M3: each form's do-nothing on z = 1 (its own field), Dirichlet on r = 1 and z = 0, axis."""
    P = params
    Fs = {form: C.load_field(P["mms_data"][form]) for form in P["forms"]}

    def bcs(field, form):
        return [bc(RECT["rmin"], "axis"), bc(RECT["z0"], "dirichlet", field.u), bc(RECT["rmax"], "dirichlet", field.u),
                bc(RECT["zmax"], "do_nothing")]
    ff = [(Fs[form], form, "dn") for form in P["forms"]]
    out = _open_family(P, quick, ff, bcs, lambda form, tag: f"{form}_struct_dn",
                       lambda tag: ["uL2r", "uH1r", "pL2r", "utL2r"], {RECT["z0"], RECT["rmax"]})
    out["notes"] = "Do-nothing of the solved form on z = 1; pressure determined (raw comparison)."
    return out


# ============================================================================================== M4
@case
def mms_axis(params, workdir, quick):
    """M4: flux-free field; orders, the r q-tested discrete continuity residual, swirl/meridional decoupling, and the
    pressure point at an interior vertex and at an axis vertex against the mean-value treatment."""
    P = params
    F = C.load_field(P["mms_data"])
    msh = P["meshes"]
    runs, ser, acc = [], {}, Acc()
    dec_levels = set(P["decoupling_levels"])

    def bcs(g):
        return [bc(RECT["rmin"], "axis")] + [bc(RECT[s], "dirichlet", g) for s in ("z0", "rmax", "zmax")]

    for form in P["forms"]:
        lv, rows = [], []
        for (V, T, Vc, h, i, N) in square_meshes(msh, quick):
            mesh, seg, att = make_mesh(V, T, SQUARE)
            info = verify(mesh, V, T, Vc)
            sol, sec = solve(mesh, F.mu, F.f(form), bcs(F.u), form)
            lvl = f"{form}_struct_ff_L{i}"
            runs.append(run_entry(lvl, sol, sec, h, info, family=f"{form}_struct_ff", refinement_level=i))
            acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
            ev = Evaluator(sol)
            n = C.norms(ev, F, "mean_free")
            dp = boundary_edge_pred(V, T, SQUARE, {RECT["z0"], RECT["rmax"], RECT["zmax"]})
            nodal_checks(acc, ev, V, T, n["u_magnitude"], dp, F.u)
            shared_mismatch(acc, V, seg, att, {RECT[s]: F.u for s in ("z0", "rmax", "zmax")})
            cr = C.discrete_continuity_residual(ev, V)
            acc.max("continuity_rq_max_rel", cr["rq_max_rel"])
            acc.max("continuity_q_unweighted_max_rel_info", cr["q_max_rel"])
            if N in dec_levels:
                fm = lambda R, Z, _f=F.f(form): _f(R, Z) * np.array([1.0, 1.0, 0.0])  # noqa: E731
                ft = lambda R, Z, _f=F.f(form): _f(R, Z) * np.array([0.0, 0.0, 1.0])  # noqa: E731
                gm = lambda R, Z: F.u(R, Z) * np.array([1.0, 1.0, 0.0])               # noqa: E731
                gt = lambda R, Z: F.u(R, Z) * np.array([0.0, 0.0, 1.0])               # noqa: E731
                st, s1 = solve(mesh, F.mu, ft, bcs(gt), form)
                runs.append(run_entry(f"{form}_swirl_only_N{N}", st, s1, h, info))
                sm, s2 = solve(mesh, F.mu, fm, bcs(gm), form)
                runs.append(run_entry(f"{form}_meridional_only_N{N}", sm, s2, h, info))
                et, em = Evaluator(st), Evaluator(sm)
                acc.max("swirl_decoupling_u_rel", C.relative_l2r_difference(ev, et, (2,))["u_rel"])
                dm = C.relative_l2r_difference(ev, em, (0, 1))
                acc.max("meridional_decoupling_u_rel", dm["u_rel"])
                acc.max("meridional_decoupling_p_rel", dm["p_rel"])
                dn = C.discrete_norms(et)
                acc.max("swirl_only_meridional_abs", dn["umL2r"] + dn["pL2r"])
                for tag, x0 in (("interior", (0.5, 0.5)), ("axis", (0.0, 0.5))):
                    pval = float(F.p(np.array(x0[0]), np.array(x0[1])))
                    sp, s3 = solve(mesh, F.mu, F.f(form), bcs(F.u), form,
                                   pressure_point={"x0": x0, "value": pval})
                    runs.append(run_entry(f"{form}_pin_{tag}_N{N}", sp, s3, h, info))
                    epp = Evaluator(sp)
                    d = C.relative_l2r_difference(ev, epp)
                    acc.max(f"pin_{tag}_vs_mean_u_rel", d["u_rel"])
                    acc.max(f"pin_{tag}_vs_mean_p_rel", d["p_rel"])
                    acc.max("pin_value_dev_max", abs(epp.p_at_vertex(V, T, x0) - pval))
            lv.append(lvl)
            rows.append(n)
        put_series(ser, f"{form}_struct_ff", lv, rows, ["uL2r", "uH1r", "pL2r", "utL2r"])
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "continuity_rq_max_rel: the check module's r q-tested residual at every level; decoupling and "
                     "pressure points at the fixed decoupling levels."}


# ============================================================================================== P1
@case
def poiseuille_pipe(params, workdir, quick):
    """P1: Hagen-Poiseuille patch tests (exact in P2 x P1) on four meshes; wall-force balance; positive control."""
    P = params
    F = C.load_field(P["mms_data"])
    acc = Acc()
    runs = []
    cls = rect_classify(0.0, 1.0, 0.0, 2.0)
    wall = lambda R, Z: np.abs(R - 1.0) < 1e-12  # noqa: E731
    inlet = lambda a, m, b: abs(a[1]) < 1e-12 and abs(m[1]) < 1e-12 and abs(b[1]) < 1e-12        # noqa: E731
    outlet = lambda a, m, b: all(abs(p[1] - 2.0) < 1e-12 for p in (a, m, b))                        # noqa: E731

    def variant(name):
        form = "sym" if name.startswith("sym") or name.startswith("control") else "lap"
        ax, w = bc(RECT["rmin"], "axis"), bc(RECT["rmax"], "dirichlet", F.u)
        if name in ("lap_dn", "control_sym_dn"):
            return form, [ax, w, bc(RECT["z0"], "dirichlet", F.u), bc(RECT["zmax"], "do_nothing")], None, []
        if name == "sym_tr":
            return form, [ax, w, bc(RECT["z0"], "dirichlet", F.u),
                          bc(RECT["zmax"], "traction", tr_value(F, form), True)], None, [(outlet, F.traction(form))]
        if name in ("lap_pd", "sym_pd"):
            return form, [ax, w, bc(RECT["z0"], "traction", tr_value(F, form), True),
                          bc(RECT["zmax"], "traction", tr_value(F, form), True)], None, \
                [(inlet, F.traction(form)), (outlet, F.traction(form))]
        if name == "lap_pin":
            return form, [ax, w, bc(RECT["z0"], "dirichlet", F.u), bc(RECT["zmax"], "dirichlet", F.u)], \
                {"x0": (0.0, 1.0), "value": 2.0}, []
        raise ValueError(name)
    q = {}
    variants = ["lap_dn", "sym_tr", "lap_pd", "sym_pd", "lap_pin", "control_sym_dn"]
    for name in variants:
        form, bcs, pp, tr = variant(name)
        wu = wp = wf = 0.0
        dir_sides = {b["attribute"] for b in bcs if b["kind"] == "dirichlet"}
        for mv in _levels(P["meshes"]["patch"]["variants"], quick):
            V, T = C.pipe_mesh(mv)
            mesh, seg, att = make_mesh(V, T, cls)
            info = verify(mesh, V, T)
            sol, sec = solve(mesh, F.mu, F.f(form), bcs, form, pressure_point=pp)
            runs.append(run_entry(f"{name}_{mv}", sol, sec, None, info))
            ev = Evaluator(sol)
            n = C.norms(ev, F, "determined")
            wu = max(wu, n["uL2r"] / F.constants["u_L2r"])
            wp = max(wp, n["pL2r"] / F.constants["p_max_abs_sampled"])
            fz = C.reaction(ev, F.mu, wall, "force_z", traction_edges=tr)
            wf = max(wf, abs(fz - 2.0) / 2.0)
            nodal_checks(acc, ev, V, T, 1.0, boundary_edge_pred(V, T, cls, dir_sides), F.u)
            acc.max(f"{name}_nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        q[f"{name}_u_rel"], q[f"{name}_p_rel"], q[f"{name}_wall_force_rel"] = wu, wp, wf
    q["control_sym_dn_dev_rel"] = q.pop("control_sym_dn_u_rel")
    acc.q.update(q)
    return {"runs": runs, "series": {}, "quantities": acc.q,
            "notes": "u/p relative errors are the worst over the four patch meshes; wall force by the check module's "
                     "Babuska-Miller reaction (traction edges subtracted where the wall's corner functions reach them)."}


# ============================================================================================== P2
@case
def circular_couette(params, workdir, quick):
    """P2: rigid rotation patch; Dirichlet-driven and shear-driven circular Couette (orders, torque order 4, torque
    balance, zero meridional flow)."""
    P = params
    msh = P["meshes"]
    runs, ser, acc = [], {}, Acc()
    cls = rect_classify(1.0, 2.0, 0.0, 1.0)
    inner = lambda R, Z: np.abs(R - 1.0) < 1e-12  # noqa: E731

    def bcs(field, form, outer):
        o = bc(RECT["rmax"], "dirichlet", field.u) if outer == "dirichlet" else \
            bc(RECT["rmax"], "traction", tr_value(field, form), True)
        return [bc(RECT["rmin"], "dirichlet", field.u), o, bc(RECT["z0"], "do_nothing"), bc(RECT["zmax"], "do_nothing")]

    def num(spec):
        return float(eval(spec["numpy"], {"np": np, "__builtins__": {}}))
    Fr = C.load_field(P["mms_data"]["couette_rotation"])
    for form in P["forms"]:
        wu, wt = 0.0, 0.0
        for N in _levels(msh["patch"]["N"], quick):
            V, T = C.annulus_rect_mesh(N)
            mesh, seg, att = make_mesh(V, T, cls)
            info = verify(mesh, V, T)
            sol, sec = solve(mesh, Fr.mu, Fr.f(form), bcs(Fr, form, "dirichlet"), form)
            runs.append(run_entry(f"rot_{form}_N{N}", sol, sec, 1.0 / N, info))
            ev = Evaluator(sol)
            n = C.norms(ev, Fr, "determined")
            wu = max(wu, n["uL2r"] / Fr.constants["u_L2r"] + n["pL2r"])
            wt = max(wt, abs(C.reaction(ev, Fr.mu, inner, "torque_z")))
        acc.q[f"rot_{form}_u_rel"], acc.q[f"rot_{form}_torque_abs"] = wu, wt
    Ns = _levels(msh["structured"]["N"], quick)
    nt = len(_levels(msh["torque"]["N"], quick))
    for name, outer, pre in (("couette_circular", "dirichlet", "cou"), ("couette_traction", "traction", "tau")):
        F = C.load_field(P["mms_data"][name])
        T_ex = num(F.data["extra"]["torque_inner_per_radian"])
        for form in P["forms"]:
            lv, rows = [], []
            for i, N in enumerate(Ns):
                V, T = C.annulus_rect_mesh(N)
                Vc = C.annulus_rect_mesh(Ns[i - 1])[0] if i else None
                mesh, seg, att = make_mesh(V, T, cls)
                info = verify(mesh, V, T, Vc)
                sol, sec = solve(mesh, F.mu, F.f(form), bcs(F, form, outer), form)
                lvl = f"{form}_{pre}_L{i}"
                runs.append(run_entry(lvl, sol, sec, msh["structured"]["h_nominal"][i], info, family=f"{form}_{pre}",
                                      refinement_level=i))
                ev = Evaluator(sol)
                n = C.norms(ev, F, "determined")
                tq = C.reaction(ev, F.mu, inner, "torque_z")
                n["torque_err"], n["torque"] = abs(tq - T_ex) / abs(T_ex), tq
                dn = C.discrete_norms(ev)
                n["meridional_abs"] = dn["umL2r"] + dn["pL2r"]
                dsides = {RECT["rmin"]} | ({RECT["rmax"]} if outer == "dirichlet" else set())
                nodal_checks(acc, ev, V, T, n["u_magnitude"], boundary_edge_pred(V, T, cls, dsides), F.u)
                if pre == "tau":
                    applied = num(F.data["extra"]["applied_torque_per_radian"])
                    acc.max(f"tau_{form}_torque_balance_rel", abs(tq - applied) / abs(applied))
                lv.append(lvl)
                rows.append(n)
            put_series(ser, f"{form}_{pre}", lv, rows, ["utL2r", "utH1r"])
            ser[f"{form}_{pre}_torque_err"] = series(lv[:nt], [r["torque_err"] for r in rows[:nt]], 1.0)
            ser[f"{form}_{pre}_torque"] = vseries(lv, [r["torque"] for r in rows])
            acc.max(f"{pre}_{form}_meridional_max", max(r["meridional_abs"] for r in rows))
            acc.q[f"{pre}_{form}_torque_finest"] = rows[-1]["torque"]
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "Torques by the check module's Babuska-Miller reaction (r e_theta on the inner cylinder); the "
                     "torque series use the first four structured levels (meshes.torque)."}


# ============================================================================================== S1
@case
def sphere(params, workdir, quick):
    """S1: Stokes flow past a translating / rotating sphere with the exact far field on rho = 4 (Dirichlet), body data
    on the sphere; isoparametric P2 geometry; drag/torque order 4; curved traction variants (transtr, rottr)."""
    P = params
    msh = P["meshes"]["sphere"]
    runs, ser, acc = [], {}, Acc()
    ks = _levels(msh["k"], quick)
    hs = msh["h_nominal"][:len(ks)]
    on_sphere = lambda R, Z: np.abs(np.hypot(R, Z) - 1.0) < 1e-9  # noqa: E731
    # an outer edge by its two VERTICES (exact in every geometry; the edge node is on the circle only when curved)
    outer_edge = lambda A, M_, B: all(abs(math.hypot(*Pt) - 4.0) < 1e-9 for Pt in (A, B))  # noqa: E731

    def num(spec):
        return float(eval(spec["numpy"], {"np": np, "__builtins__": {}}))
    for name, pre, key, kind in (("sphere_translation", "trans", "drag_z_per_radian", "force_z"),
                                 ("sphere_rotation", "rot", "torque_z_per_radian", "torque_z")):
        F = C.load_field(P["mms_data"][name])
        ex = num(F.data["extra"][key])
        body = (lambda R, Z, _c="translation" if pre == "trans" else "rotation": C.sphere_body_data(R, Z, _c))
        data = (lambda R, Z, _b=body, _F=F: np.where((np.hypot(R, Z) < 2.5)[..., None], _b(R, Z), _F.u(R, Z)))
        for variant in ("dir", "tr"):
            vpre = pre if variant == "dir" else pre + "tr"
            for form in P["forms"]:
                lv, rows = [], []
                for i, k in enumerate(ks):
                    V, T = C.sphere_mesh(k)
                    X6 = C.sphere_p2_nodes(k)
                    Vc = C.sphere_mesh(ks[i - 1])[0] if i else None
                    mesh, seg, att = make_mesh(V, T, sphere_classify, X6)
                    info = verify(mesh, V, T, Vc, X6_ref=X6)
                    outer = bc(SPH["outer"], "dirichlet", F.u) if variant == "dir" else \
                        bc(SPH["outer"], "traction", tr_value(F, form), True)
                    bcs = [bc(SPH["axis_top"], "axis"), bc(SPH["axis_bottom"], "axis"),
                           bc(SPH["sphere"], "dirichlet", body), outer]
                    sol, sec = solve(mesh, F.mu, F.f(form), bcs, form)
                    lvl = f"{form}_{vpre}_L{i}"
                    runs.append(run_entry(lvl, sol, sec, hs[i], info, family=f"{form}_{vpre}", refinement_level=i))
                    ev = Evaluator(sol)
                    dsides = {SPH["sphere"]} | ({SPH["outer"]} if variant == "dir" else set())
                    dp = boundary_edge_pred(V, T, sphere_classify, dsides)
                    if variant == "dir":
                        n = C.norms(ev, F, "mean_free")
                        acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
                        fv = C.reaction(ev, F.mu, on_sphere, kind)
                        n["fun_err"], n["fun"] = abs(fv - ex) / abs(ex), fv
                        if pre == "trans" and form == "lap":
                            fl = C.reaction(ev, F.mu, on_sphere, kind, residual_form="lap")
                            n["fun_lres_err"] = abs(fl - ex) / abs(ex)
                        n["area_err"] = abs(C.area_r(ev) - 42.0)   # int r dr dz = 2 (4^3 - 1)/3 = 42
                        shared_mismatch(acc, V, seg, att, {SPH["sphere"]: body, SPH["outer"]: F.u})
                    else:
                        n = C.norms(ev, F, "determined")
                        acc.max("nullspace_detected_tr", int(sol.diagnostics["nullspace_detected"]))
                        if form == "sym":
                            fr = C.reaction(ev, F.mu, on_sphere, kind)
                            ap = C.applied_boundary_load(ev, outer_edge, F.traction("sym"), kind)
                            acc.max(f"{vpre}_sym_balance_rel", abs(fr - ap) / abs(ap))
                    nodal_checks(acc, ev, V, T, n["u_magnitude"], dp, data)
                    lv.append(lvl)
                    rows.append(n)
                if variant == "dir":
                    keys = ["uL2r", "uH1r", "pL2r"] if pre == "trans" else ["utL2r", "utH1r"]
                    put_series(ser, f"{form}_{pre}", lv, rows, keys)
                    fk = "drag" if pre == "trans" else "torque"
                    ser[f"{form}_{pre}_{fk}_err"] = series(lv, [r["fun_err"] for r in rows], 1.0)
                    ser[f"{form}_{pre}_{fk}"] = vseries(lv, [r["fun"] for r in rows])
                    acc.q[f"{form}_{pre}_{fk}_3d_finest"] = 2 * math.pi * rows[-1]["fun"]
                    if pre == "trans" and form == "lap":
                        ser["lap_trans_drag_lres_err"] = series(lv, [r["fun_lres_err"] for r in rows], 1.0)
                        ser["geometry_area_r_err"] = series(lv, [r["area_err"] for r in rows], 42.0)
                else:
                    keys = ["uL2r", "uH1r", "pL2r"] if pre == "trans" else ["utL2r", "utH1r"]
                    put_series(ser, f"{form}_{vpre}", lv, rows, keys)
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "Body data on the sphere, exact solution (or its sigma_form n with the geometric normal of the "
                     "isoparametric circle) on rho = 4; drag/torque by the check module's reaction; 3D values x 2 pi."}


# ============================================================================================== X1
@case
def xcode_cavity(params, workdir, quick):
    """X1: the swirling-lid cylinder cavity (mu = 1, no forcing): E_kin and the dissipation per form and level."""
    P = params
    runs, acc = [], Acc()
    Ns = _levels(P["meshes"]["structured"]["N"], quick, 1)
    for form in P["forms"]:
        for N in Ns:
            V, T = C.square_mesh(N)
            mesh, seg, att = make_mesh(V, T, SQUARE)
            info = verify(mesh, V, T)
            bcs = [bc(RECT["rmin"], "axis")] + [bc(RECT[s], "dirichlet", C.cavity_lid) for s in ("z0", "rmax", "zmax")]
            sol, sec = solve(mesh, 1.0, None, bcs, form)
            runs.append(run_entry(f"{form}_N{N}", sol, sec, 1.0 / N, info))
            ev = Evaluator(sol)
            nodal_checks(acc, ev, V, T, 1.0,
                         boundary_edge_pred(V, T, SQUARE, {RECT["z0"], RECT["rmax"], RECT["zmax"]}), C.cavity_lid)
            acc.q[f"{form}_Ekin_N{N}"] = C.kinetic_energy_r(ev)
            acc.q[f"{form}_Diss_N{N}"] = C.dissipation_r(ev, form)
    return {"runs": runs, "series": {}, "quantities": acc.q,
            "notes": "E_kin = (1/2) int |u_h|^2 r, Diss = a_form(u_h, u_h) per radian, by the check module."}
