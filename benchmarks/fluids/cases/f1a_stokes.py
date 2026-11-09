"""F1a planar Stokes: the case runner (set-up, solve, Evaluator).  Every norm, identity and check value is
computed by the check module f1a_stokes_check (imported as C); this module only builds the fixed meshes
with C's builders, verifies the solver's mesh against them, solves through the public API axionfem.fluids and
wraps each discrete solution in an Evaluator (C's protocol: n_elements, geometry, velocity, pressure).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first levels only.

Conventions: form "lap" = Laplacian (mu grad u), "sym" = symmetric stress (2 mu eps);
traction data g = sigma_form n with the outward normal; grad-div gamma in viscosity units.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

import f1a_stokes_check as C
from axionfem import fluids

FORM = {"lap": "laplacian", "sym": "stress"}
SQ = {"y0": 1, "x1": 2, "y1": 3, "x0": 4}          # unit square / rectangle edges -> boundary attributes
TOL = 1e-9


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

    def _ref(self, ref_pts):
        return np.ascontiguousarray(ref_pts, dtype=float)

    def geometry(self, ref_pts):
        return self.sol.geometry(self._el, self._ref(ref_pts))

    def velocity(self, ref_pts):
        return self.sol.velocity(self._el, self._ref(ref_pts))

    def pressure(self, ref_pts):
        return self.sol.pressure(self._el, self._ref(ref_pts))

    def at_vertex(self, mesh, point):
        """(u, p) at a mesh vertex (evaluated in one element that has it)."""
        V, T = mesh.vertices(), mesh.elements()
        k = int(np.argmin(np.hypot(V[:, 0] - point[0], V[:, 1] - point[1])))
        if math.hypot(*(V[k] - np.asarray(point, float))) > 1e-12:
            raise RuntimeError(f"{point} is not a vertex of the mesh")
        e = int(np.nonzero((T == k).any(axis=1))[0][0])
        loc = int(np.nonzero(T[e] == k)[0][0])
        ref = np.array([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][loc]])
        el = np.array([e], dtype=np.int32)
        u, _ = self.sol.velocity(el, ref)
        return u[0, 0], float(self.sol.pressure(el, ref)[0, 0])


def boundary_of(V, T, classify):
    """(segments (nb, 2), attributes (nb,)) of the boundary edges of (V, T), classify(midpoint) -> attribute."""
    edges = C.boundary_edges(T)
    seg = np.array(edges, dtype=np.int32)
    att = np.array([classify(*(0.5 * (V[a] + V[b]))) for a, b in edges], dtype=np.int32)
    return seg, att


def rect_classify(x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    def f(x, y):
        if abs(y - y0) < TOL:
            return SQ["y0"]
        if abs(x - x1) < TOL:
            return SQ["x1"]
        if abs(y - y1) < TOL:
            return SQ["y1"]
        if abs(x - x0) < TOL:
            return SQ["x0"]
        raise RuntimeError(f"boundary edge at ({x}, {y}) is on no side of the rectangle")
    return f


def make_mesh(V, T, classify, X6=None):
    V = np.ascontiguousarray(V, dtype=float)
    T = np.ascontiguousarray(T, dtype=np.int32)
    seg, att = boundary_of(V, T, classify)
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
    sol = fluids.solve_stokes(mesh, mu, f, bcs, form=FORM[form], vectorized=True, lu_threads=1,
                              **kw)
    sec = time.time() - t0
    if not sol.diagnostics["converged"]:
        raise NotConverged(run_entry("not_converged", sol, sec, None, {"n_elements": int(sol.n_elements)}))
    return sol, sec


def case(fn):
    """Every case function: a solve that fails its own gate ends the case with that run reported."""
    import functools

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
                    "lu_refine_steps": d["lu_refine_steps"], "slip_nodes": d["slip_nodes"],
                    "slip_corner_nodes": d["slip_corner_nodes"],
                    "periodic_pressure_gradient": list(d["periodic_pressure_gradient"]),
                    "grad_div": d["grad_div"], "ranks": d["ranks"], "threads": d["threads"],
                    "lu_threads": d["lu_threads"],
                    "velocity_nullspace": d["velocity_nullspace"]},
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


def dirichlet_checks(acc, sol, data, V, seg, att):
    """Nodal-interpolation defect per Dirichlet attribute (data: attribute -> callable) and the shared-node mismatch
    of the data at vertices where several Dirichlet attributes meet."""
    for a, g in data.items():
        P, U = sol.boundary_nodes([a])
        if len(P):
            acc.max("dirichlet_nodal_data_defect_max", C.dirichlet_nodal_defect(P, U, g))
    owners = {}
    for (i, j), a in zip(seg, att):
        if int(a) in data:
            for v in (int(i), int(j)):
                owners.setdefault(v, set()).add(int(a))
    shared = [(v, sorted(s)) for v, s in owners.items() if len(s) > 1]
    if shared:
        pts = np.array([V[v] for v, _ in shared])
        cal = [[data[a] for a in s] for _, s in shared]
        acc.max("dirichlet_shared_node_mismatch_max", C.dirichlet_shared_node_mismatch(pts, cal))
    else:
        acc.max("dirichlet_shared_node_mismatch_max", 0.0)


def straight_edges(V, T, attrs_of_edge=None):
    """(Pa, Pm, Pb, outward n) for every boundary edge (straight), for C.flux_defect."""
    out = []
    cent = {}
    for e, t in enumerate(np.asarray(T)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            cent[(min(t[i], t[j]), max(t[i], t[j]))] = V[t].mean(axis=0)
    for a, b in C.boundary_edges(T):
        Pa, Pb = V[a], V[b]
        d = Pb - Pa
        n = np.array([d[1], -d[0]]) / math.hypot(*d)
        if np.dot(n, 0.5 * (Pa + Pb) - cent[(a, b)]) < 0:
            n = -n
        out.append((Pa, 0.5 * (Pa + Pb), Pb, n))
    return out


def l2(ev, fn, vdim=2):
    """||fn||_L2 by the check module's quadrature (fn(X, Y) -> (..., vdim) or (...))."""
    q, w = C.tri_rule(C.QUAD_DEGREE)
    xq, detJ = ev.geometry(q)
    W = w[None, :] * detJ
    v = np.asarray(fn(xq[..., 0], xq[..., 1]), float)
    if v.ndim == W.ndim + 1:
        return math.sqrt(float((W[..., None] * v ** 2).sum()))
    return math.sqrt(float((W * v ** 2).sum()))


def ph_l2(ev):
    q, w = C.tri_rule(C.QUAD_DEGREE)
    _, detJ = ev.geometry(q)
    return math.sqrt(float(((w[None, :] * detJ) * ev.pressure(q) ** 2).sum()))


def dissipation(ev, form, mu=1.0, gamma=0.0):
    """a_form(u_h, u_h) + gamma ||div u_h||^2 by the check module's quadrature (constant mu)."""
    q, w = C.tri_rule(C.QUAD_DEGREE)
    _, detJ = ev.geometry(q)
    W = w[None, :] * detJ
    _, G = ev.velocity(q)
    if form == "lap":
        a = mu * float((W[..., None, None] * G ** 2).sum())
    else:
        eps = 0.5 * (G + np.swapaxes(G, -1, -2))
        a = 2 * mu * float((W[..., None, None] * eps ** 2).sum())
    div = float((W * (G[..., 0, 0] + G[..., 1, 1]) ** 2).sum())
    return a + gamma * div


def series(levels, errors, magnitude):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


def const(v):
    return lambda X, Y: np.full(np.shape(X), float(v))


def zero_vec(X, Y):
    return np.zeros(np.shape(X) + (2,))


def all_dirichlet(g):
    return [{"attribute": a, "kind": "dirichlet", "value": g} for a in (1, 2, 3, 4)]


class Pointwise:
    """A field whose fixed numpy spellings are scalar-only (Python conditionals, e.g. the two-layer Couette):
    the same expressions evaluated point by point, so that arrays can be passed."""

    def __init__(self, F):
        self.F = F
        self.constants = F.constants

    @staticmethod
    def _pw(fn, shape_tail):
        def g(X, Y):
            X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
            out = np.array([np.asarray(fn(np.float64(x), np.float64(y)), float)
                            for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape + shape_tail)
            return out
        return g

    def u(self, X, Y):
        return self._pw(self.F.u, (2,))(X, Y)

    def grad_u(self, X, Y):
        return self._pw(self.F.grad_u, (2, 2))(X, Y)

    def p(self, X, Y):
        return self._pw(self.F.p, ())(X, Y)

    def mu(self, X, Y):
        return self._pw(self.F.mu, ())(X, Y)

    def f(self, form):
        return self._pw(self.F.f(form), (2,))

    def traction(self, form, normal):
        return self._pw(self.F.traction(form, normal), (2,))


def mesh_struct(N):
    return C.structured_mesh(N)


# ============================================================================================== T1
@case
def mms_dirichlet(params, workdir, quick):
    """T1: all-Dirichlet, variable mu, both forms, structured and distorted families, grad-div gamma = 1, the
    identity sym(gamma=0) == lap(gamma=mu) at constant mu, the tightened-solver sensitivity."""
    P = params
    F = C.load_field(P["mms_data"])
    Fc = C.load_field(P["mms_data_constmu"])
    msh = P["meshes"]
    runs, ser, acc, notes = [], {}, Acc(), []
    Ns = _levels(msh["structured"]["N"], quick)
    ks = _levels(msh["distorted"]["refinements"], quick)
    gammas = [float(g) for g in P["grad_div_gamma"]]

    def do(form, fam, idx, gamma, tag, tight=False, field=F, mu=None):
        if fam == "struct":
            N = Ns[idx]
            V, T = mesh_struct(N)
            Vc = mesh_struct(Ns[idx - 1])[0] if idx > 0 else None
            h = msh["structured"]["h_nominal"][idx]
        else:
            k = ks[idx]
            V, T = C.distorted_mesh(k, N0=msh["distorted"]["N0"], rel=msh["distorted"]["perturb_rel"],
                                    seed=msh["distorted"]["seed"])
            Vc = C.distorted_mesh(ks[idx - 1], N0=msh["distorted"]["N0"], rel=msh["distorted"]["perturb_rel"],
                                  seed=msh["distorted"]["seed"])[0] if idx > 0 else None
            h = msh["distorted"]["h_nominal"][idx]
        mesh, seg, att = make_mesh(V, T, rect_classify())
        info = verify(mesh, V, T, Vc)
        sol, sec = solve(mesh, field.mu if mu is None else mu, field.f(form), all_dirichlet(field.u), form,
                         grad_div=gamma, min_refine_steps=2 if tight else 0)
        lvl = f"{tag}_{fam}_L{idx}"
        runs.append(run_entry(lvl, sol, sec, h, info, family=f"{fam}" + ("_gd1" if gamma else "") + ("_tight" if tight else ""),
                              refinement_level=idx))
        acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        dirichlet_checks(acc, sol, {a: field.u for a in (1, 2, 3, 4)}, V, seg, att)
        d, ref = C.flux_defect(straight_edges(V, T), field.u)
        acc.max("discrete_flux_defect_rel_max", abs(d) / ref)
        return lvl, Evaluator(sol), sol

    for form in P["forms"]:
        for fam in ("struct", "dist"):
            lv, rows = [], []
            for i in range(len(Ns) if fam == "struct" else len(ks)):
                lvl, ev, _ = do(form, fam, i, 0.0, f"{form}")
                lv.append(lvl)
                rows.append(C.norms(ev, F, "mean_free"))
            mag_u = max(r["u_magnitude"] for r in rows)
            mag_p = max(r["p_magnitude"] for r in rows)
            for k, key, mag in (("uL2", "uL2", mag_u), ("uH1", "uH1", mag_u), ("pL2m", "pL2", mag_p),
                                ("divL2", "divL2", mag_u)):
                ser[f"{form}_{fam}_{k}"] = series(lv, [r[key] for r in rows], mag)
            if fam == "struct" and form == "sym":
                acc.q["sym_struct_uL2_finest"] = rows[-1]["uL2"]
        g1 = [g for g in gammas if g != 0.0]
        for g in g1:
            lv, rows = [], []
            for i in range(len(Ns)):
                lvl, ev, _ = do(form, "struct", i, g, f"{form}_gd{g:g}")
                lv.append(lvl)
                rows.append(C.norms(ev, F, "mean_free"))
            mag_u = max(r["u_magnitude"] for r in rows)
            mag_p = max(r["p_magnitude"] for r in rows)
            for k, key, mag in (("uL2", "uL2", mag_u), ("uH1", "uH1", mag_u), ("pL2m", "pL2", mag_p)):
                ser[f"{form}_struct_gd{g:g}_{k}"] = series(lv, [r[key] for r in rows], mag)
    # tightened direct solve (2 extra LU refinement steps) at the finest structured level
    if "sym" in P["forms"]:
        _, ev, _ = do("sym", "struct", len(Ns) - 1, 0.0, "sym_tight", tight=True)
        acc.q["sym_struct_uL2_finest_tight"] = C.norms(ev, F, "mean_free")["uL2"]
    # identity: sym (gamma = 0) == lap (gamma = mu) at constant mu, all Dirichlet
    idn = P["identity"]
    mu0 = float(idn["mu"])
    for N in idn["N"]:
        if quick and N > Ns[-1]:
            continue
        V, T = mesh_struct(N)
        mesh, seg, att = make_mesh(V, T, rect_classify())
        info = verify(mesh, V, T)
        h = 1.0 / N
        s_sym, t1 = solve(mesh, Fc.mu, Fc.f("sym"), all_dirichlet(Fc.u), "sym", grad_div=0.0)
        s_lap, t2 = solve(mesh, Fc.mu, Fc.f("lap"), all_dirichlet(Fc.u), "lap", grad_div=mu0)
        runs.append(run_entry(f"identity_sym_N{N}", s_sym, t1, h, info))
        runs.append(run_entry(f"identity_lap_gd_N{N}", s_lap, t2, h, info))
        r = C.relative_l2_difference(Evaluator(s_sym), Evaluator(s_lap))
        acc.max("identity_sym_vs_lap_graddiv_mu_u_rel_max", r["u_rel"])
        acc.max("identity_sym_vs_lap_graddiv_mu_p_rel_max", r["p_rel"])
        dirichlet_checks(acc, s_sym, {a: Fc.u for a in (1, 2, 3, 4)}, V, seg, att)
    notes.append("The continuity right-hand side is projected onto 1-perp (flux defect reported per run); the "
                 "mean-value multiplier removes the constant; relative_residual is that of the consistent system.")
    return {"runs": runs, "series": ser, "quantities": acc.q, "notes": " ".join(notes)}


# ============================================================================================== T2
@case
def mms_traction(params, workdir, quick):
    """T2: Dirichlet on x = 0 and y = 0, inhomogeneous traction g = sigma_form n on x = 1 and y = 1."""
    P = params
    F = C.load_field(P["mms_data"])
    Fc = C.load_field(P["mms_data_constmu"])
    msh = P["meshes"]
    runs, ser, acc = [], {}, Acc()
    Ns = _levels(msh["structured"]["N"], quick)
    ks = _levels(msh["distorted"]["refinements"], quick)

    def bcs(field, form):
        return [{"attribute": SQ["x0"], "kind": "dirichlet", "value": field.u},
                {"attribute": SQ["y0"], "kind": "dirichlet", "value": field.u},
                {"attribute": SQ["x1"], "kind": "traction", "value": field.traction(form, (1.0, 0.0))},
                {"attribute": SQ["y1"], "kind": "traction", "value": field.traction(form, (0.0, 1.0))}]

    def family(form, fam, field, tag, keys):
        lv, rows = [], []
        n = len(Ns) if fam == "struct" else len(ks)
        for i in range(n):
            if fam == "struct":
                V, T = mesh_struct(Ns[i])
                Vc = mesh_struct(Ns[i - 1])[0] if i else None
                h = msh["structured"]["h_nominal"][i]
            else:
                dd = msh["distorted"]
                V, T = C.distorted_mesh(ks[i], N0=dd["N0"], rel=dd["perturb_rel"], seed=dd["seed"])
                Vc = C.distorted_mesh(ks[i - 1], N0=dd["N0"], rel=dd["perturb_rel"], seed=dd["seed"])[0] if i else None
                h = dd["h_nominal"][i]
            mesh, seg, att = make_mesh(V, T, rect_classify())
            info = verify(mesh, V, T, Vc)
            sol, sec = solve(mesh, field.mu, field.f(form), bcs(field, form), form)
            lvl = f"{tag}_L{i}"
            runs.append(run_entry(lvl, sol, sec, h, info, family=tag, refinement_level=i))
            acc.max("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
            dirichlet_checks(acc, sol, {SQ["x0"]: field.u, SQ["y0"]: field.u}, V, seg, att)
            lv.append(lvl)
            rows.append(C.norms(Evaluator(sol), field, "determined"))
        mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
        for k in keys:
            key = "pL2" if k.startswith("p") else k
            ser[f"{tag}_{k}"] = series(lv, [r[key] for r in rows], mp if k.startswith("p") else mu_)

    for form in P["forms"]:
        family(form, "struct", F, f"{form}_struct", ("uL2", "uH1", "pL2"))
        family(form, "dist", F, f"{form}_dist", ("uL2", "uH1", "pL2"))
        family(form, "struct", Fc, f"{form}_struct_constmu", ("uL2", "pL2"))
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "traction data g = sigma_form n (outward n) on x = 1 and y = 1; pressure determined (raw)."}


# ============================================================================================== T3
@case
def mms_do_nothing(params, workdir, quick):
    """T3: the form's own homogeneous natural condition on x = 1 (do-nothing), Dirichlet elsewhere."""
    P = params
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    runs, ser, acc = [], {}, Acc()
    for form in P["forms"]:
        F = C.load_field(P["mms_data"][form])
        bcs = [{"attribute": a, "kind": "dirichlet", "value": F.u} for a in (SQ["x0"], SQ["y0"], SQ["y1"])] + \
              [{"attribute": SQ["x1"], "kind": "do_nothing"}]
        lv, rows = [], []
        for i, N in enumerate(Ns):
            V, T = mesh_struct(N)
            mesh, seg, att = make_mesh(V, T, rect_classify())
            info = verify(mesh, V, T, mesh_struct(Ns[i - 1])[0] if i else None)
            sol, sec = solve(mesh, F.mu, F.f(form), bcs, form)
            lvl = f"{form}_dn_L{i}"
            runs.append(run_entry(lvl, sol, sec, msh["h_nominal"][i], info, family=f"{form}_dn", refinement_level=i))
            acc.max("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
            dirichlet_checks(acc, sol, {a: F.u for a in (SQ["x0"], SQ["y0"], SQ["y1"])}, V, seg, att)
            lv.append(lvl)
            rows.append(C.norms(Evaluator(sol), F, "determined"))
        mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
        for k in ("uL2", "uH1", "pL2"):
            ser[f"{form}_struct_dn_{k}"] = series(lv, [r[k] for r in rows], mp if k == "pL2" else mu_)
    return {"runs": runs, "series": ser, "quantities": acc.q, "notes": "do-nothing = the form's natural condition."}


# ============================================================================================== T4
@case
def mms_slip(params, workdir, quick):
    """T4: rotated unit square (30 deg), slip on local t = 0, symmetry on local s = 1, Dirichlet on s = 0, t = 1.
    The slip normals are the API default (outward normal of the straight edges)."""
    P = params
    F = C.load_field(P["mms_data"])
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    th = math.pi / 6
    c, s_ = math.cos(th), math.sin(th)
    ATT = {"t0": 1, "s1": 2, "t1": 3, "s0": 4}
    NRM = {"t0": (s_, -c), "s1": (c, s_), "s0": (-c, -s_), "t1": (-s_, c)}

    def classify(x, y):
        s, t = x * c + y * s_, -x * s_ + y * c
        for k, v in (("t0", abs(t)), ("s1", abs(s - 1)), ("t1", abs(t - 1)), ("s0", abs(s))):
            if v < TOL:
                return ATT[k]
        raise RuntimeError("edge on no side")
    kinds = P["boundaries"]
    runs, ser, acc = [], {}, Acc()
    for form in P["forms"]:
        bcs = []
        for side, kind in kinds.items():
            b = {"attribute": ATT[side], "kind": kind}
            if kind == "dirichlet":
                b["value"] = F.u
            bcs.append(b)
        dir_sides = [k for k, v in kinds.items() if v == "dirichlet"]
        slip_sides = [k for k, v in kinds.items() if v in ("slip", "symmetry")]
        lv, rows = [], []
        for i, N in enumerate(Ns):
            V0, T = mesh_struct(N)
            V = C.rotate(V0, th)
            mesh, seg, att = make_mesh(V, T, classify)
            Vc = C.rotate(mesh_struct(Ns[i - 1])[0], th) if i else None
            info = verify(mesh, V, T, Vc)
            sol, sec = solve(mesh, F.mu, F.f(form), bcs, form)
            lvl = f"{form}_L{i}"
            runs.append(run_entry(lvl, sol, sec, msh["h_nominal"][i], info, family=f"{form}_rotated",
                                  refinement_level=i))
            acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
            dirichlet_checks(acc, sol, {ATT[k]: F.u for k in dir_sides}, V, seg, att)
            ev = Evaluator(sol)
            r = C.norms(ev, F, "mean_free")
            # u.n at every slip/symmetry node, against every slip normal of the edges it lies on
            Pn, Un = sol.boundary_nodes([ATT[k] for k in slip_sides])
            normals, on_dir = [], []
            for x, y in Pn:
                ss, tt = x * c + y * s_, -x * s_ + y * c
                loc = {"t0": abs(tt), "s1": abs(ss - 1), "t1": abs(tt - 1), "s0": abs(ss)}
                normals.append([NRM[k] for k in slip_sides if loc[k] < TOL])
                on_dir.append(any(loc[k] < TOL for k in dir_sides))
            acc.max("slip_un_nodal_max_rel", C.slip_nodal_un(Un, normals, r["u_magnitude"], on_dir))
            lv.append(lvl)
            rows.append(r)
        mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
        for k, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2")):
            ser[f"{form}_struct_{k}"] = series(lv, [r[key] for r in rows], mp if k == "pL2m" else mu_)
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "strong u.n = 0 by nodal rotation with the geometric (exact) edge normals; Dirichlet wins at "
                     "shared nodes; mean-free pressure."}


# ============================================================================================== T5
def _q_value(F, P):
    q = float(np.asarray(F.extra_fn("flow_rate_Q")(np.zeros(1), np.zeros(1))).ravel()[0])
    spelled = P.get("flow_rate_Q")
    if isinstance(spelled, str):
        alt = float(eval(spelled, {"__builtins__": {}}, {"sin": math.sin, "cos": math.cos, "pi": math.pi}))
        if abs(alt - q) > 1e-14 * max(1.0, abs(q)):
            raise RuntimeError(f"flow rate Q of the field {q} != the parameter {spelled} = {alt}")
    return q


@case
def mms_periodic(params, workdir, quick):
    """T5: periodic in x (period 1) with the pressure drop G = 2 prescribed (dp) or the flow rate Q prescribed and
    G_h computed (q); walls y = 0, 1 Dirichlet.  Pressure compared on its periodic part."""
    P = params
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    G = float(P["pressure_drop_per_period"])
    runs, ser, acc = [], {}, Acc()
    for variant in ("dp", "q"):
        F = C.load_field(P["mms_data"][variant])
        pper = F.extra_fn("p_periodic")
        Q = _q_value(F, P)
        per = {"attribute": SQ["x0"], "kind": "periodic", "partner": SQ["x1"], "translation": (1.0, 0.0)}
        per.update({"pressure_drop": G * 1.0} if variant == "dp" else {"flow_rate": Q})
        bcs = [per] + [{"attribute": a, "kind": "dirichlet", "value": F.u} for a in (SQ["y0"], SQ["y1"])]
        for form in P["forms"]:
            lv, rows = [], []
            for i, N in enumerate(Ns):
                V, T = mesh_struct(N)
                mesh, seg, att = make_mesh(V, T, rect_classify())
                info = verify(mesh, V, T, mesh_struct(Ns[i - 1])[0] if i else None)
                sol, sec = solve(mesh, F.mu, F.f(form), bcs, form)
                lvl = f"{form}_{variant}_L{i}"
                runs.append(run_entry(lvl, sol, sec, msh["h_nominal"][i], info, family=f"{form}_{variant}",
                                      refinement_level=i))
                acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
                dirichlet_checks(acc, sol, {a: F.u for a in (SQ["y0"], SQ["y1"])}, V, seg, att)
                pg = sol.diagnostics["periodic_pressure_gradient"]
                Gh = float(pg[0]) if len(pg) else float("nan")   # nan: the solver reported no periodic pair
                runs[-1]["G_h"] = Gh
                if variant == "q" and i == len(Ns) - 1:
                    acc.max("q_G_rel_err_finest_max", abs(Gh - G) / abs(G) if math.isfinite(Gh) else math.inf)
                lv.append(lvl)
                rows.append(C.norms(Evaluator(sol), F, "mean_free", p_exact=pper))
            mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
            for k, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2")):
                ser[f"{form}_struct_{variant}_{k}"] = series(lv, [r[key] for r in rows], mp if k == "pL2m" else mu_)
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "periodic identification of x = 0 and x = 1 (make_periodic_mesh); dp: body force G e_x with "
                     "p = p_per - G x; q: one multiplier for int_Omega u_x = Q (unit period), G_h its value."}


# ============================================================================================== T6
@case
def mms_pressure_point(params, workdir, quick):
    """T6: all-Dirichlet flux-free data, pressure point p_h(x0) = p(x0) at x0 = (1/2, 1/4)."""
    P = params
    F = C.load_field(P["mms_data"])
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    x0 = tuple(float(v) for v in P["pressure_point"]["x"])
    p0 = float(np.asarray(F.extra_fn("p_at_point")(np.zeros(1), np.zeros(1))).ravel()[0])
    p0_direct = float(F.p(np.array([x0[0]]), np.array([x0[1]]))[0])
    if abs(p0 - p0_direct) > 1e-12 * max(1.0, abs(p0)):
        raise RuntimeError(f"p_at_point {p0} != p(x0) {p0_direct}")
    pp = {"x0": x0, "value": p0}
    runs, ser, acc = [], {}, Acc()
    for form in P["forms"]:
        lv, rows = [], []
        for i, N in enumerate(Ns):
            V, T = mesh_struct(N)
            mesh, seg, att = make_mesh(V, T, rect_classify())
            info = verify(mesh, V, T, mesh_struct(Ns[i - 1])[0] if i else None)
            sol, sec = solve(mesh, F.mu, F.f(form), all_dirichlet(F.u), form, pressure_point=pp)
            lvl = f"{form}_pt_L{i}"
            runs.append(run_entry(lvl, sol, sec, msh["h_nominal"][i], info, family=f"{form}_pt", refinement_level=i))
            dirichlet_checks(acc, sol, {a: F.u for a in (1, 2, 3, 4)}, V, seg, att)
            ev = Evaluator(sol)
            r = C.norms(ev, F, "mean_free")
            _, p_at = ev.at_vertex(mesh, x0)
            acc.max("pin_value_dev_max", abs(p_at - p0))
            if i == len(Ns) - 1:
                acc.max("pin_offset_finest_max", r["pin_offset"])
            # the same problem with a mean-zero pressure: the velocity must not change
            s0, sec0 = solve(mesh, F.mu, F.f(form), all_dirichlet(F.u), form)
            runs.append(run_entry(f"{form}_meanzero_L{i}", s0, sec0, msh["h_nominal"][i], info))
            acc.max("pin_vs_meanzero_u_rel_max", C.relative_l2_difference(Evaluator(s0), ev)["u_rel"])
            lv.append(lvl)
            rows.append(r)
        mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
        for k, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2")):
            ser[f"{form}_struct_pt_{k}"] = series(lv, [r[key] for r in rows], mp if k == "pL2m" else mu_)
    # the pressure point with an open boundary must be refused
    V, T = mesh_struct(Ns[0])
    mesh, _, _ = make_mesh(V, T, rect_classify())
    bcs = [{"attribute": a, "kind": "dirichlet", "value": F.u} for a in (1, 3, 4)] + \
          [{"attribute": SQ["x1"], "kind": "traction", "value": F.traction("sym", (1.0, 0.0))}]
    try:
        solve(mesh, F.mu, F.f("sym"), bcs, "sym", pressure_point=pp)
        acc.q["pin_with_open_boundary_refused"] = 0
    except fluids.FluidsError:
        acc.q["pin_with_open_boundary_refused"] = 1
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "pressure point = COMSOL pressure point constraint (raw pressure); p_h(x0) evaluated at the "
                     "vertex x0 by the Evaluator."}


# ============================================================================================== P1 Poiseuille
def _channel_meshes(P, quick):
    """The four patch meshes of the channel [0, L] x [0, H]: structured Nx x Ny (corner cells flipped) and the
    distorted ones (interior vertices of the 8 x 4 base moved by perturb_rel * (L / Nx) * U(-1, 1)^2, PCG64(seed),
    then red refinements), all with C's builders."""
    ch = P["channel"]
    L, H = float(ch["L"]), float(ch["H"])
    out = []
    for m in P["meshes"]:
        if m["family"] == "structured":
            V, T = C._structured(int(m["Nx"]), int(m["Ny"]), 0.0, L, 0.0, H)
            out.append((f"struct_{m['Nx']}x{m['Ny']}", V, T))
        else:
            nx, ny = (int(v.split("=")[1]) for v in str(m["N0"]).split(","))
            V, T = C._structured(nx, ny, 0.0, L, 0.0, H)
            inner = np.where((V[:, 0] > 1e-12) & (V[:, 0] < L - 1e-12) & (V[:, 1] > 1e-12) & (V[:, 1] < H - 1e-12))[0]
            rng = np.random.Generator(np.random.PCG64(int(m["seed"])))
            V = V.copy()
            V[inner] += float(m["perturb_rel"]) * (L / nx) * rng.uniform(-1, 1, (len(inner), 2))
            for _ in range(int(m["refinements"])):
                V, T = C.red_refine(V, T)
            out.append((f"dist_s{m['seed']}_r{m['refinements']}", V, T))
    return out[:2] if quick else out


def _rel(ev, F, mean_free=False):
    r = C.norms(ev, F, "mean_free" if mean_free else "determined")
    un = l2(ev, F.u)
    pn = l2(ev, F.p, 1)
    return r["uL2"] / un, (r["pL2"] / pn if pn > 0 else r["pL2"]), r


def boundary_flux(sol, mesh, attrs_of_edge, seg, att, which):
    """int u_h . n over the boundary edges whose attribute is in `which` (outward n), Gauss-Legendre 6 points."""
    V, T = mesh.vertices(), mesh.elements()
    edge_el = {}
    for e, t in enumerate(T):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge_el[(min(t[i], t[j]), max(t[i], t[j]))] = (e, i, j)
    ref = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    g, w = np.polynomial.legendre.leggauss(6)
    s = 0.5 * (g + 1)
    tot = 0.0
    for (a, b), at in zip(seg, att):
        if int(at) not in which:
            continue
        e, i, j = edge_el[(min(a, b), max(a, b))]
        ra, rb = np.array(ref[i]), np.array(ref[j])
        pts = ra[None, :] + s[:, None] * (rb - ra)[None, :]
        u, _ = sol.velocity(np.array([e], np.int32), np.ascontiguousarray(pts))
        Pa, Pb = V[T[e][i]], V[T[e][j]]
        d = Pb - Pa
        n = np.array([d[1], -d[0]])
        cen = V[T[e]].mean(axis=0)
        if np.dot(n, 0.5 * (Pa + Pb) - cen) < 0:
            n = -n
        tot += float(np.sum(0.5 * w * (u[0] @ n)))   # |e| cancels: n has length |e|
    return tot


@case
def poiseuille_patch(params, workdir, quick):
    """P1: Poiseuille u = (y(1-y), 0), p = 2(2 - x) on [0, 2] x [0, 1] (mu = 1, G = 2), exact in P2/P1: every
    variant must reproduce it to round-off on every patch mesh."""
    P = params
    ch = P["channel"]
    L, H, G = float(ch["L"]), float(ch["H"]), float(ch["G"])
    Fi = C.load_field(P["mms_data"]["poiseuille_inflow_dirichlet"])
    Fp = C.load_field(P["mms_data"]["poiseuille_pressure_driven"])
    Fh = C.load_field(P["mms_data"]["poiseuille_half_symmetry"])
    Fr = C.load_field(P["mms_data"]["plug_rotated"])
    mu = float(ch["mu"])
    runs, acc = [], Acc()
    zero = zero_vec
    th = math.pi / 6
    c, s_ = math.cos(th), math.sin(th)

    def go(tag, V, T, classify, field, form, bcs, **kw):
        mesh, seg, att = make_mesh(V, T, classify)
        info = verify(mesh, V, T)
        sol, sec = solve(mesh, mu, field.f(form), bcs, form, **kw)
        runs.append(run_entry(tag, sol, sec, None, info))
        return mesh, seg, att, sol, Evaluator(sol)

    for mtag, V, T in _channel_meshes(P, quick):
        rc = rect_classify(0.0, L, 0.0, H)
        walls = [{"attribute": SQ["y0"], "kind": "dirichlet", "value": zero},
                 {"attribute": SQ["y1"], "kind": "dirichlet", "value": zero}]
        inflow = {"attribute": SQ["x0"], "kind": "dirichlet", "value": Fi.u}
        dirdata = {SQ["x0"]: Fi.u, SQ["y0"]: zero, SQ["y1"]: zero}
        # lap_dn
        mesh, seg, att, sol, ev = go(f"lap_dn_{mtag}", V, T, rc, Fi, "lap",
                                     [inflow, {"attribute": SQ["x1"], "kind": "do_nothing"}] + walls)
        ur, pr, _ = _rel(ev, Fi)
        acc.max("lap_dn_u_rel", ur)
        acc.max("lap_dn_p_rel", pr)
        acc.max("lap_dn_nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        dirichlet_checks(acc, sol, dirdata, V, seg, att)
        fin = boundary_flux(sol, mesh, None, seg, att, {SQ["x0"]})
        fout = boundary_flux(sol, mesh, None, seg, att, {SQ["x1"]})
        fw = boundary_flux(sol, mesh, None, seg, att, {SQ["y0"], SQ["y1"]})
        acc.max("lap_dn_flux_imbalance_rel", abs(fin + fout + fw) / abs(fin))
        # sym_tr: exact outlet traction (0, mu U'(y))
        _, _, _, sol, ev = go(f"sym_tr_{mtag}", V, T, rc, Fi, "sym",
                              [inflow, {"attribute": SQ["x1"], "kind": "traction",
                                        "value": Fi.traction("sym", (1.0, 0.0))}] + walls)
        ur, pr, _ = _rel(ev, Fi)
        acc.max("sym_tr_u_rel", ur)
        acc.max("sym_tr_p_rel", pr)
        # control: sym with do-nothing is NOT Poiseuille
        _, _, _, sol, ev = go(f"control_sym_dn_{mtag}", V, T, rc, Fi, "sym",
                              [inflow, {"attribute": SQ["x1"], "kind": "do_nothing"}] + walls)
        acc.min("control_sym_dn_dev_rel", _rel(ev, Fi)[0])
        # pressure-driven: traction on inlet and outlet
        for form in ("lap", "sym"):
            _, _, _, sol, ev = go(f"{form}_pd_{mtag}", V, T, rc, Fp, form,
                                  [{"attribute": SQ["x0"], "kind": "traction", "value": Fp.traction(form, (-1.0, 0.0))},
                                   {"attribute": SQ["x1"], "kind": "traction", "value": Fp.traction(form, (1.0, 0.0))}]
                                  + walls)
            acc.max(f"{form}_pd_u_rel", _rel(ev, Fp)[0])
        # periodic, prescribed drop and prescribed flow rate
        for form in ("lap", "sym"):
            for var in ("dp", "q"):
                per = {"attribute": SQ["x0"], "kind": "periodic", "partner": SQ["x1"], "translation": (L, 0.0)}
                Q = G * H ** 3 / (12 * mu)
                per.update({"pressure_drop": G * L} if var == "dp" else {"flow_rate": Q})
                _, _, _, sol, ev = go(f"{form}_per_{var}_{mtag}", V, T, rc, Fp, form, [per] + walls)
                if var == "dp":
                    acc.max(f"{form}_per_dp_u_rel", _rel(ev, Fp, True)[0])
                else:
                    pg = sol.diagnostics["periodic_pressure_gradient"]
                    Gh = float(pg[0]) if len(pg) else float("nan")
                    acc.max(f"{form}_per_q_G_rel", abs(Gh - G) / G if math.isfinite(Gh) else math.inf)
        # lap_pin: all Dirichlet, pressure point at the vertex nearest (L/2, H/2) with the exact p there
        Vm = mesh.vertices()
        k = int(np.argmin(np.hypot(Vm[:, 0] - L / 2, Vm[:, 1] - H / 2)))
        xp = (float(Vm[k, 0]), float(Vm[k, 1]))
        ppv = float(Fi.p(np.array([xp[0]]), np.array([xp[1]]))[0])
        mesh, seg, att, sol, ev = go(f"lap_pin_{mtag}", V, T, rc, Fi, "lap",
                                     [inflow, {"attribute": SQ["x1"], "kind": "dirichlet", "value": Fi.u}] + walls,
                                     pressure_point={"x0": xp, "value": ppv})
        ur, pr, _ = _rel(ev, Fi)
        acc.max("lap_pin_u_rel", ur)
        acc.max("lap_pin_p_rel", pr)
        acc.min("lap_pin_nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        dirichlet_checks(acc, sol, {**dirdata, SQ["x1"]: Fi.u}, V, seg, att)
        # half channel, symmetry on y = H/2
        Vh = V.copy()
        Vh[:, 1] *= 0.5
        _, seg, att, sol, ev = go(f"lap_half_{mtag}", Vh, T, rect_classify(0.0, L, 0.0, H / 2), Fh, "lap",
                                  [{"attribute": SQ["x0"], "kind": "dirichlet", "value": Fh.u},
                                   {"attribute": SQ["x1"], "kind": "do_nothing"},
                                   {"attribute": SQ["y0"], "kind": "dirichlet", "value": zero},
                                   {"attribute": SQ["y1"], "kind": "symmetry"}])
        acc.max("lap_half_u_rel", _rel(ev, Fh)[0])
        dirichlet_checks(acc, sol, {SQ["x0"]: Fh.u, SQ["y0"]: zero}, Vh, seg, att)
        # plug flow in the rotated channel, slip walls
        Vr = C.rotate(V, th)

        def rclass(x, y):
            s, t = x * c + y * s_, -x * s_ + y * c
            for a, v in ((1, abs(t)), (2, abs(s - L)), (3, abs(t - H)), (4, abs(s))):
                if v < TOL:
                    return a
            raise RuntimeError("edge on no side")
        for form in ("lap", "sym"):
            _, seg, att, sol, ev = go(f"plug_{form}_{mtag}", Vr, T, rclass, Fr, form,
                                      [{"attribute": 4, "kind": "dirichlet", "value": Fr.u},
                                       {"attribute": 2, "kind": "do_nothing"},
                                       {"attribute": 1, "kind": "slip"}, {"attribute": 3, "kind": "slip"}])
            acc.max(f"plug_{form}_u_rel", _rel(ev, Fr)[0])
    return {"runs": runs, "series": {}, "quantities": acc.q,
            "notes": "lap_pin: pressure point at the vertex nearest (1, 1/2) (on the distorted meshes (1, 1/2) is not "
                     "a vertex) with the exact pressure there; periodic period L = 2."}


# ============================================================================================== P2 Couette
@case
def couette_patch(params, workdir, quick):
    """P2: plane Couette u = (y, 0), p = 0 (periodic and inflow/outflow variants) and the traction-driven two-layer
    Couette with mu = 1 / 3 and the interface on a mesh line."""
    P = params
    Fper = C.load_field(P["mms_data"]["couette_periodic"])
    Fout = C.load_field(P["mms_data"]["couette_outflow"])
    F2 = Pointwise(C.load_field(P["mms_data"]["couette_two_layer"]))   # scalar-only spellings
    u_top_exact, u_mid_exact = 2.0 / 3.0, 0.5
    runs, acc = [], Acc()
    meshes = []
    for m in P["meshes"]:
        if m["family"] == "structured":
            V, T = C._structured(int(m["N"]), int(m["N"]))
            meshes.append((f"struct_{m['N']}", V, T, True))
        else:
            V, T = C.distorted_mesh(int(m["refinements"]), N0=int(m["N0"]), rel=float(m["perturb_rel"]),
                                    seed=int(m["seed"]))
            meshes.append((f"dist_s{m['seed']}", V, T, False))
    if quick:
        meshes = meshes[:1]

    def go(tag, V, T, field, form, bcs, mu=1.0):
        mesh, seg, att = make_mesh(V, T, rect_classify())
        info = verify(mesh, V, T)
        sol, sec = solve(mesh, mu, field.f(form), bcs, form)
        runs.append(run_entry(tag, sol, sec, None, info))
        return mesh, seg, att, sol, Evaluator(sol)

    for mtag, V, T, periodic_ok in meshes:
        if periodic_ok:
            for form in ("lap", "sym"):
                per = {"attribute": SQ["x0"], "kind": "periodic", "partner": SQ["x1"], "translation": (1.0, 0.0)}
                mesh, seg, att, sol, ev = go(f"{form}_per_{mtag}", V, T, Fper, form,
                                             [per, {"attribute": SQ["y0"], "kind": "dirichlet", "value": Fper.u},
                                              {"attribute": SQ["y1"], "kind": "dirichlet", "value": Fper.u}])
                r = C.norms(ev, Fper, "mean_free")
                acc.max(f"{form}_per_u_rel", r["uL2"] / l2(ev, Fper.u))
                acc.max(f"{form}_per_p_abs", ph_l2(ev))
                dirichlet_checks(acc, sol, {SQ["y0"]: Fper.u, SQ["y1"]: Fper.u}, V, seg, att)
                # two-layer: shear traction on y = 1, mu = 1 below / 3 above y = 1/2, periodic in x
                mesh, seg, att, sol, ev = go(f"{form}_two_layer_{mtag}", V, T, F2, form,
                                             [per, {"attribute": SQ["y0"], "kind": "dirichlet", "value": F2.u},
                                              {"attribute": SQ["y1"], "kind": "traction",
                                               "value": F2.traction(form, (0.0, 1.0))}], mu=F2.mu)
                r = C.norms(ev, F2, "determined")
                acc.max(f"{form}_two_layer_u_rel", r["uL2"] / l2(ev, F2.u))
                acc.max(f"{form}_two_layer_p_abs", ph_l2(ev))
                ut, _ = ev.at_vertex(mesh, (0.5, 1.0))
                um, _ = ev.at_vertex(mesh, (0.5, 0.5))
                for key, val, ex in ((f"{form}_two_layer_u_top", ut[0], u_top_exact),
                                     (f"{form}_two_layer_u_mid", um[0], u_mid_exact)):
                    if key not in acc.q or abs(val - ex) > abs(acc.q[key] - ex):
                        acc.q[key] = float(val)
                dirichlet_checks(acc, sol, {SQ["y0"]: F2.u}, V, seg, att)
        # inflow / outflow variants (every mesh)
        out_bcs = [{"attribute": a, "kind": "dirichlet", "value": Fout.u} for a in (SQ["x0"], SQ["y0"], SQ["y1"])]
        mesh, seg, att, sol, ev = go(f"out_lap_{mtag}", V, T, Fout, "lap",
                                     out_bcs + [{"attribute": SQ["x1"], "kind": "do_nothing"}])
        acc.max("out_lap_u_rel", C.norms(ev, Fout, "determined")["uL2"] / l2(ev, Fout.u))
        dirichlet_checks(acc, sol, {a: Fout.u for a in (SQ["x0"], SQ["y0"], SQ["y1"])}, V, seg, att)
        _, _, _, sol, ev = go(f"out_sym_{mtag}", V, T, Fout, "sym",
                              out_bcs + [{"attribute": SQ["x1"], "kind": "traction",
                                          "value": Fout.traction("sym", (1.0, 0.0))}])
        acc.max("out_sym_u_rel", C.norms(ev, Fout, "determined")["uL2"] / l2(ev, Fout.u))
        _, _, _, sol, ev = go(f"control_sym_dn_{mtag}", V, T, Fout, "sym",
                              out_bcs + [{"attribute": SQ["x1"], "kind": "do_nothing"}])
        acc.min("control_sym_dn_dev_rel", C.norms(ev, Fout, "determined")["uL2"] / l2(ev, Fout.u))
    return {"runs": runs, "series": {}, "quantities": acc.q,
            "notes": "u_top / u_mid: u_x at the vertices (1/2, 1) and (1/2, 1/2), the value farthest from the exact "
                     "one over the meshes; p_abs = ||p_h||_L2 (exact p = 0)."}


# ============================================================================================== A1 no-flow
@case
def noflow(params, workdir, quick):
    """A1: u = 0, p = phi, f = grad phi, Dirichlet u = 0: the spurious velocity of Taylor-Hood, nu-scaling,
    the grad-div energy identity and the grad-div ratios."""
    P = params
    F = C.load_field(P["mms_data"])
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    f = F.f("lap")
    zero = zero_vec
    bcs = all_dirichlet(zero)
    runs, ser, acc = [], {}, Acc()
    cache = {}

    def go(N, form, nu, gamma, tag, refinement_level=None, family=None, h=None):
        key = (N, form, nu, gamma)
        if key in cache:
            return cache[key]
        if N not in cache:
            V, T = mesh_struct(N)
            mesh, seg, att = make_mesh(V, T, rect_classify())
            info = verify(mesh, V, T)
            cache[N] = (mesh, info)
        mesh, info = cache[N]
        sol, sec = solve(mesh, nu, f, bcs, form, grad_div=gamma)
        runs.append(run_entry(tag, sol, sec, h if h is not None else 1.0 / N, info, family=family,
                              refinement_level=refinement_level))
        cache[key] = Evaluator(sol)
        return cache[key]

    nu0 = 1.0
    for form in P["forms"]:
        lv, rows = [], []
        for i, N in enumerate(Ns):
            ev = go(N, form, nu0, 0.0, f"{form}_noflow_L{i}", i, f"{form}_noflow", msh["h_nominal"][i])
            if i:
                verify(cache[N][0], *mesh_struct(N), mesh_struct(Ns[i - 1])[0])
            dn = C.discrete_norms(ev)
            r = C.norms(ev, F, "mean_free")
            lv.append(f"{form}_noflow_L{i}")
            rows.append({"uL2": dn["uL2"], "uH1": dn["uH1"], "pL2": r["pL2"], "pmag": r["p_magnitude"]})
        mag = max(r["pmag"] for r in rows) / nu0
        for k, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2")):
            ser[f"{form}_noflow_{k}"] = series(lv, [r[key] for r in rows], mag)
    # nu-scaling: nu ||u_h(nu)|| and p_h independent of nu
    Nn = 32 if not quick else Ns[-1]
    for form in P["forms"]:
        e1 = go(Nn, form, 1.0, 0.0, f"{form}_nu1_N{Nn}")
        u1 = C.discrete_norms(e1)["uL2"]
        for nu in P["nu"]:
            nu = float(nu)
            if nu == 1.0:
                continue
            en = go(Nn, form, nu, 0.0, f"{form}_nu{nu:g}_N{Nn}")
            acc.max("nu_scaling_dev_rel_max", abs(nu * C.discrete_norms(en)["uL2"] - u1) / u1)
            acc.max("p_nu_independence_dev_rel_max", C.relative_l2_difference(e1, en)["p_rel"])
    # identity sym(gamma = 0) == lap(gamma = nu)
    es = go(Nn, "sym", 1.0, 0.0, f"sym_nu1_N{Nn}")
    el = go(Nn, "lap", 1.0, 1.0, f"lap_gd1_N{Nn}")
    acc.max("identity_sym_vs_lap_graddiv_nu_u_rel", C.relative_l2_difference(es, el)["u_rel"])
    # grad-div energy identity
    ei = P["energy_identity"]
    Ne = int(ei["N"]) if not quick else Ns[-1]
    for nu in ei["nu"]:
        for g in ei["gamma"]:
            for form in ei["forms"]:
                ev = go(Ne, form, float(nu), float(g), f"{form}_energy_nu{float(nu):g}_g{float(g):g}")
                acc.max("graddiv_energy_identity_defect_max",
                        C.energy_identity_defect(ev, float(nu), float(g), form, f))
    # grad-div ratios R(gamma) = |u_h(gamma)|_1 / |u_h(0)|_1 (lap, nu = 1)
    gd = P["grad_div"]
    Ng = int(gd["N"]) if not quick else Ns[-1]
    H1 = {}
    for g in gd["gamma"]:
        H1[float(g)] = C.discrete_norms(go(Ng, "lap", float(gd["nu"]), float(g), f"lap_ratio_g{float(g):g}"))["uH1"]
    gs = sorted(H1)
    R = {g: H1[g] / H1[0.0] for g in gs}
    acc.q["graddiv_ratio_g10"] = R.get(10.0, float("nan"))
    acc.q["graddiv_ratio_g1000"] = R.get(1000.0, float("nan"))
    acc.q["graddiv_monotone"] = int(all(R[b] < R[a] for a, b in zip(gs, gs[1:])))
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": f"grad-div ratios {R}; f = grad phi independent of nu."}


# ============================================================================================== A2 annulus
@case
def annulus_slip(params, workdir, quick):
    """A2: annulus 1 < r < 2 on isoparametric P2 polar meshes (6-node geometry input), Dirichlet on r = 1, free slip
    on r = 2 with the exact radial nodal normal; MMS per form, the lap closed form, the sym rigid rotation."""
    P = params
    msh = P["meshes"]["polar"]
    Nrs = _levels(msh["N_r"], quick)
    hs = msh["h_nominal"]
    runs, ser, acc = [], {}, Acc()

    def classify(x, y):   # chord midpoint of a boundary edge: r = R cos(pi / N_theta) with R = 1 or 2
        r = math.hypot(x, y)
        if r < 1.5:
            return 1
        return 2

    def radial(X, Y):
        R = np.hypot(X, Y)
        return np.stack([X / R, Y / R], axis=-1)
    meshes = {}

    def mesh_of(i):
        if i not in meshes:
            Nr = Nrs[i]
            V, T = C.polar_mesh(Nr)
            X6 = C.polar_p2_nodes(Nr)
            mesh, seg, att = make_mesh(V, T, classify, X6=X6)
            info = verify(mesh, V, T, C.polar_mesh(Nrs[i - 1])[0] if i else None, X6_ref=X6)
            meshes[i] = (mesh, seg, att, info, V)
        return meshes[i]

    def go(i, field, form, tag, family=None):
        mesh, seg, att, info, V = mesh_of(i)
        bcs = [{"attribute": 1, "kind": "dirichlet", "value": field.u},
               {"attribute": 2, "kind": "slip", "normal": radial}]
        sol, sec = solve(mesh, field.mu, field.f(form), bcs, form)
        runs.append(run_entry(tag, sol, sec, hs[i], info, family=family, refinement_level=i if family else None))
        dirichlet_checks(acc, sol, {1: field.u}, V, seg, att)
        return Evaluator(sol)

    for form in P["forms"]:
        F = C.load_field(P["mms_data"][f"mms_{form}"])
        lv, rows, areas = [], [], []
        for i in range(len(Nrs)):
            ev = go(i, F, form, f"{form}_mms_L{i}", f"{form}_polar")
            lv.append(f"{form}_mms_L{i}")
            rows.append(C.norms(ev, F, "mean_free"))
            areas.append(C.area(ev) - 3 * math.pi)
        mu_, mp = max(r["u_magnitude"] for r in rows), max(r["p_magnitude"] for r in rows)
        for k, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2")):
            ser[f"{form}_annulus_mms_{k}"] = series(lv, [r[key] for r in rows], mp if k == "pL2m" else mu_)
        if form == "lap":
            ser["geometry_area_err"] = series(lv, areas, 3 * math.pi)
    if "lap" in P["forms"]:
        F = C.load_field(P["mms_data"]["closed_lap"])
        lv, rows = [], []
        for i in range(len(Nrs)):
            ev = go(i, F, "lap", f"lap_closed_L{i}", "lap_closed_polar")
            lv.append(f"lap_closed_L{i}")
            rows.append(C.norms(ev, F, "mean_free"))
        ser["lap_annulus_closed_uL2"] = series(lv, [r["uL2"] for r in rows], max(r["u_magnitude"] for r in rows))
    if "sym" in P["forms"]:
        F = C.load_field(P["mms_data"]["closed_sym"])
        for i in range(len(Nrs)):
            ev = go(i, F, "sym", f"sym_rigid_L{i}")
            acc.max("sym_rigid_rotation_u_rel_max", C.norms(ev, F, "mean_free")["uL2"] / l2(ev, F.u))
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "isoparametric P2 geometry from polar_p2_nodes (verified by hash); slip normal x/|x| at the "
                     "nodes; geometry_area_err = |Omega_h| - 3 pi (signed) on the lap MMS meshes."}


# ============================================================================================== C1, X1 cavity
def _lid(X, Y):
    X = np.asarray(X, float)
    on = np.abs(np.asarray(Y, float) - 1.0) < 1e-12
    return np.stack([np.where(on, 16 * X ** 2 * (1 - X) ** 2, 0.0), np.zeros_like(X)], axis=-1)


def _cavity(N, form, acc, runs, tag, mu, pp=None, family=None, refinement_level=None, h=None, mesh_cache=None,
            Vc=None):
    V, T = mesh_struct(N)
    mesh, seg, att = make_mesh(V, T, rect_classify())
    info = verify(mesh, V, T, Vc)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": (_lid if a == SQ["y1"] else zero_vec)} for a in (1, 2, 3, 4)]
    sol, sec = solve(mesh, mu, None, bcs, form, pressure_point=pp)
    runs.append(run_entry(tag, sol, sec, h if h is not None else 1.0 / N, info, family=family,
                          refinement_level=refinement_level))
    dirichlet_checks(acc, sol, {a: (_lid if a == SQ["y1"] else zero_vec) for a in (1, 2, 3, 4)}, V, seg, att)
    return sol, Evaluator(sol)


@case
def cavity_regularized(params, workdir, quick):
    """C1: regularized lid u = (16 x^2 (1-x)^2, 0), mu = 1: kinetic energy and dissipation per form (functional
    convergence), the two forms' energies, pin invariance, dissipation = lid power (Babuska-Miller)."""
    P = params
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    mu = float(P["mu"])
    runs, ser, acc = [], {}, Acc()
    E = {}
    for form in P["forms"]:
        lv, ek, ds = [], [], []
        for i, N in enumerate(Ns):
            sol, ev = _cavity(N, form, acc, runs, f"{form}_cavity_L{i}", mu, family=f"{form}_cavity",
                              refinement_level=i, h=msh["h_nominal"][i],
                              Vc=mesh_struct(Ns[i - 1])[0] if i else None)
            e = 0.5 * C.discrete_norms(ev)["uL2"] ** 2
            d = dissipation(ev, form, mu)
            pw = sol.boundary_power([SQ["y1"]])
            acc.max("diss_equals_lid_power_rel", abs(d - pw) / abs(d))
            lv.append(f"{form}_cavity_L{i}")
            ek.append(e)
            ds.append(d)
        ser[f"{form}_cavity_Ekin"] = {"levels": lv, "values": ek}
        ser[f"{form}_cavity_Diss"] = {"levels": lv, "values": ds}
        E[form] = ek[-1]
    if "lap" in E and "sym" in E:
        acc.q["forms_Ekin_dev_rel_finest"] = abs(E["lap"] - E["sym"]) / abs(E["sym"])
    pin = P["pressure"]["pin_probe"]
    Np = int(pin["N"]) if not quick else Ns[0]
    for form in P["forms"]:
        _, e0 = _cavity(Np, form, acc, runs, f"{form}_meanzero_N{Np}", mu)
        _, e1 = _cavity(Np, form, acc, runs, f"{form}_pin_N{Np}", mu,
                        pp={"x0": tuple(float(v) for v in pin["x"]), "value": 0.0})
        acc.max("pin_vs_meanzero_u_rel", C.relative_l2_difference(e0, e1)["u_rel"])
    return {"runs": runs, "series": ser, "quantities": acc.q,
            "notes": "Ekin = (1/2)||u_h||^2 and Diss = a_form(u_h, u_h) by the check module's quadrature; lid power "
                     "from the residual of the solved form at the lid nodes (StokesSolution.boundary_power)."}


@case
def xcode_cavity(params, workdir, quick):
    """X1: the regularized cavity at N = 64, 128 (both forms), E_kin and dissipation, for the comparison with the
    NGSolve values of the reference block."""
    P = params
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick, 1)
    mu = float(P["mu"])
    runs, acc = [], Acc()
    for form in P["forms"]:
        for i, N in enumerate(Ns):
            sol, ev = _cavity(N, form, acc, runs, f"{form}_xcode_N{N}", mu, h=msh["h_nominal"][i])
            acc.q[f"{form}_Ekin_N{N}"] = 0.5 * C.discrete_norms(ev)["uL2"] ** 2
            acc.q[f"{form}_Diss_N{N}"] = dissipation(ev, form, mu)
    return {"runs": runs, "series": {}, "quantities": acc.q,
            "notes": "lid by P2 nodal interpolation; mean-zero pressure (multiplier)."}


# ============================================================================================== N1 P1/P1 control
def _free_blocks(sol):
    import scipy.sparse as sp
    b = sol.blocks()

    def m(d):
        return sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    A, B, Mp = m(b["A"]), m(b["B"]), m(b["Mp"])
    ess = np.zeros(A.shape[0], bool)
    ess[np.asarray(b["essential_velocity_dofs"], int)] = True
    free = np.nonzero(~ess)[0]
    return A[free][:, free], B[:, free], Mp


@case
def p1p1_control(params, workdir, quick):
    """N1: discrete inf-sup constants of P2/P1 and unstabilized P1/P1 (C.inf_sup on the free velocity DOFs, nu = 1
    Laplacian form: A = the vector H1-seminorm Gram), and the Taylor-Hood corner mesh check of the API."""
    P = params
    msh = P["meshes"]["structured"]
    Ns = _levels(msh["N"], quick)
    F = C.load_field({"path": "tools/fluids_mms/out/sq_dirichlet.json",
                      "sha256": "ca166b6b9467630217e56fe1ab00f4fd9edf8ee890783b52f7f17758a7c5e931"})
    runs, acc, notes = [], Acc(), []
    beta = {}
    for i, N in enumerate(Ns):
        V, T = mesh_struct(N)
        mesh, seg, att = make_mesh(V, T, rect_classify())
        info = verify(mesh, V, T, mesh_struct(Ns[i - 1])[0] if i else None)
        acc.min("mesh_check_accepts_structured", int(len(mesh.taylor_hood_check()["all_boundary_vertices"]) == 0))
        bcs = all_dirichlet(zero_vec)
        th = fluids.assemble_stokes(mesh, 1.0, bcs, form="laplacian")
        r = C.inf_sup(*_free_blocks(th))
        beta[("TH", N)] = r["beta"]
        acc.max("TH_zero_modes_beyond_constant_max", r["zero_modes_beyond_constant"])
        p11 = fluids.assemble_stokes(mesh, 1.0, bcs, form="laplacian", order=1, pressure_order=1,
                                     allow_unstable_pair=True)
        r1 = C.inf_sup(*_free_blocks(p11))
        beta[("P1P1", N)] = r1["beta"]
        if N == 32 or (quick and i == len(Ns) - 1):
            acc.q["P1P1_zero_modes_beyond_constant_N32"] = r1["zero_modes_beyond_constant"]
        notes.append(f"N={N}: beta_TH {r['beta']:.4g} (zero modes beyond the constant {r['zero_modes_beyond_constant']}),"
                     f" beta_P1P1 {r1['beta']:.4g} ({r1['zero_modes_beyond_constant']})")
        # the Taylor-Hood MMS solve (a real solve: the run entry of this level)
        sol, sec = solve(mesh, F.mu, F.f("lap"), all_dirichlet(F.u), "lap")
        runs.append(run_entry(f"TH_mms_N{N}", sol, sec, msh["h_nominal"][i], info, family="TH_mms",
                              refinement_level=i))
        try:   # informative only: the unstabilized P1/P1 solve may be singular
            s1, _ = solve(mesh, F.mu, F.f("lap"), all_dirichlet(F.u), "lap", order=1, pressure_order=1,
                          allow_unstable_pair=True)
            if s1.diagnostics["converged"]:
                rr = C.norms(Evaluator(s1), F, "mean_free")
                notes.append(f"P1/P1 MMS N={N}: uL2 {rr['uL2']:.3e} pL2m {rr['pL2']:.3e} (informative)")
            else:
                notes.append(f"P1/P1 MMS N={N}: not converged (residual {s1.diagnostics['relative_residual']:.2e})")
        except (fluids.FluidsError, NotConverged) as e:
            notes.append(f"P1/P1 MMS N={N}: not usable ({str(e)[:120]})")
    Nf, N0 = Ns[-1], Ns[0]
    acc.q["beta_TH_N32"] = beta[("TH", Nf)]
    acc.q["beta_TH_ratio_N32_N8"] = beta[("TH", Nf)] / beta[("TH", N0)]
    acc.q["beta_P1P1_over_TH_N32"] = beta[("P1P1", Nf)] / beta[("TH", Nf)]
    Vb, Tb = C._structured(16, 16, flip=False)
    mb, _, _ = make_mesh(Vb, Tb, rect_classify())
    acc.q["mesh_check_rejects_bad_corner"] = int(len(mb.taylor_hood_check()["all_boundary_vertices"]) > 0)
    return {"runs": runs, "series": {}, "quantities": acc.q, "notes": "; ".join(notes)}
