"""F13a inductionless MHD: the case runner (set-up, solve, Evaluator) for the eight fixed cases.

Every norm, identity and functional is computed by the check modules (f13a_mhd_check, f13a_annulus_check,
f13a_suction_check, f1a_stokes_check; imported as C / AC / SC / F1).  This module only builds the fixed meshes with
their builders, verifies the solver's mesh against them, solves through axionfem.fluids.solve_navier_stokes(...,
geometry="planar25d", mhd=...) and wraps each discrete solution in an Evaluator (the check module's protocol).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two levels only.

Boundary attributes of every rectangle: 1 bottom (y = y0), 2 right (x = x1), 3 top (y = y1), 4 left (x = x0); annuli:
1 inner circle, 2 outer circle.  Components are (u_x, u_y, u_z) everywhere.  Physical units: rho, mu, sigma separate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

import f13a_annulus_check as AC
import f13a_mhd_check as C
import f13a_suction_check as SC
import f1a_stokes_check as F1
from axionfem import fluids

MU, SIGMA, G, A_ = 0.5, 2.0, 1.0, 1.0       # analytic cases (run.parameters.physical of A1-A3, X1)
TOL = 1e-9
SIDES = {"bottom": 1, "right": 2, "top": 3, "left": 4}
# stopping tolerance of the duct solves (aspect ratios ~1e3, Ha^2 = 1e6: the backward-error floor of the direct solve in the
# SCALED dual norm is ~1e-11 at Ha = 1000); the gates are 1e-9 (A2, A3) and 1e-10 (X1) on the
# UNSCALED EUCLIDEAN residual (`relative_residual_unscaled_euclid`)
DUCT_RTOL = 1e-10
# A4 (gate 1e-10): the dual-norm residual of the 131k-element radial m = 5 levels is 1.3e-12 (growing with refinement), above the
# 1e-12 default: stopping there ran 33 pointless refinement steps (45 min per solve)
ANNULUS_RTOL = 1e-11
EDGE_ATTR = {"y0": 1, "x1": 2, "y1": 3, "x0": 4}


# ============================================================================================== identity of the core
_CORE: dict = {}


def core_identity() -> dict:
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


# ============================================================================================== evaluator, meshes
class Evaluator:
    """The discrete solution in f13a_mhd_check's protocol (one MPI rank)."""

    def __init__(self, sol):
        self.sol = sol
        self.n_elements = int(sol.n_elements)
        self._el = np.arange(self.n_elements, dtype=np.int32)

    @staticmethod
    def _ref(r):
        return np.ascontiguousarray(r, dtype=float)

    def geometry(self, ref_pts):
        return self.sol.geometry(self._el, self._ref(ref_pts))

    def velocity(self, ref_pts):
        u, g = self.sol.velocity(self._el, self._ref(ref_pts))
        return np.asarray(u), np.asarray(g)

    def pressure(self, ref_pts):
        return np.asarray(self.sol.pressure(self._el, self._ref(ref_pts)))

    def potential(self, ref_pts):
        p, g = self.sol.potential(self._el, self._ref(ref_pts))
        return np.asarray(p), np.asarray(g)


def rect_classify(x0, x1, y0, y1):
    def f(x, y):
        if abs(y - y0) < TOL:
            return 1
        if abs(x - x1) < TOL:
            return 2
        if abs(y - y1) < TOL:
            return 3
        if abs(x - x0) < TOL:
            return 4
        raise RuntimeError(f"boundary edge at ({x}, {y}) is on no side of the rectangle")
    return f


def circle_classify(x, y):
    # the midpoint of a boundary chord lies slightly inside its circle: classify by the nearer circle
    r = math.hypot(x, y)
    return 1 if abs(r - AC.PARAMS["r1"]) < abs(r - AC.PARAMS["r2"]) else 2


def make_mesh(V, T, classify, X6=None):
    V = np.ascontiguousarray(V, dtype=float)
    T = np.ascontiguousarray(T, dtype=np.int32)
    seg = np.array(F1.boundary_edges(T), dtype=np.int32)
    att = np.array([classify(*(0.5 * (V[a] + V[b]))) for a, b in seg], dtype=np.int32)
    if X6 is None:
        return fluids.Mesh.from_arrays(V, T, seg, att), seg, att
    return fluids.Mesh.from_arrays(V, T, seg, att, p2_nodes=np.ascontiguousarray(X6, float)), seg, att


def verify(mesh, V_ref, T_ref, V_coarser=None, X6=None):
    v = F1.verify_mesh(mesh.vertices(), mesh.elements(), V_ref, T_ref, V_coarser=V_coarser,
                       X6=(mesh.p2_nodes() if X6 is not None else None), X6_ref=X6)
    bad = [k for k in ("counts_match", "hash_match", "nested", "p2_hash_match") if k in v and not v[k]]
    if bad:
        raise RuntimeError(f"the solver mesh does not match the fixed construction: {bad} ({v})")
    return v


def zero3(X, Y):
    z = np.zeros(np.shape(X))
    return np.stack([z, z, z], axis=-1)


def vec3(x, y, z):
    def f(X, Y):
        o = np.ones(np.shape(X))
        return np.stack([x * o, y * o, z * o], axis=-1)
    return f


def const_B(b):
    def f(X, Y):
        o = np.ones(np.shape(X))
        return np.stack([b[0] * o, b[1] * o, b[2] * o], axis=-1)
    return f


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


def unscaled_euclid_residual(sol):
    """The UNSCALED EUCLIDEAN relative residual of the MHD system the solver solved (the measure in which the
    solver-residual criteria of the MHD cases are set): the Euclidean norm of the physical residual over the free
    DOFs, the potential rows multiplied back by the solver's potential-row scale (`mhd_row_scale`), relative to the
    same norm at the lift (the zero state).  `relative_residual` (the solver's scaled dual norm) is reported unchanged."""
    d = sol.diagnostics
    x = np.asarray(sol.state(), float)
    na = int(d["dofs_potential"]) - int(d["phi_dirichlet_dofs"]) - (1 if int(d["phi_pinned_dof"]) >= 0 else 0)
    s = float(d["mhd_row_scale"])

    def nrm(R):
        R = np.asarray(R, float)
        return math.sqrt(float(np.dot(R[:len(R) - na], R[:len(R) - na])) + (s * s) * float(np.dot(R[len(R) - na:], R[len(R) - na:])))

    den = nrm(sol.residual(list(np.zeros_like(x))))
    return nrm(sol.residual(list(x))) / den if den > 0 else 0.0


def _safe_euclid(sol):
    try:
        v = unscaled_euclid_residual(sol)
    except Exception:       # an unsolved/failed solve: no number (the gate then fails with "missing")
        return None
    return v if math.isfinite(v) else None


def run_entry(level, sol, sec, h, info, family, refinement_level, opts):
    d = sol.diagnostics
    g = d.get
    hist = [_jsonable(dict(x)) for x in sol.history]
    e = {"level": level,
         "mesh": {"h_nominal": h, "h_max": (info or {}).get("h_max"), "n_elements": int(sol.n_elements),
                  "dofs_field": int(g("dofs_velocity", 0)), "dofs_pressure": int(g("dofs_pressure", 0)),
                  "dofs_potential": int(g("dofs_potential", 0)), "dofs_total": int(g("dofs_field", 0)) + int(g("dofs_potential", 0)),
                  "files": [], "hash": (info or {}).get("hash"),
                  "all_boundary_triangles": (info or {}).get("all_boundary_triangles")},
         "solver": {"converged": bool(g("converged")), "relative_residual": float(g("relative_residual")),
                    "relative_residual_dual_scaled": float(g("relative_residual")),
                    "relative_residual_unscaled_euclid": _safe_euclid(sol),
                    "mhd_row_scale": g("mhd_row_scale"), "mhd_phi_scale": g("mhd_phi_scale"),
                    "residual_lift": g("residual_lift"), "iterations": int(g("iterations", -1)),
                    "history": hist, "newton_steps": sum(1 for x in hist if x.get("kind") == "newton"),
                    "strategy": g("strategy"), "rho": g("rho"), "viscous_form": "laplacian",
                    "geometry": "planar25d", "kind": g("solver"), "lu_backend": g("lu_backend"),
                    "nullspace": g("nullspace"), "nullspace_detected": bool(g("nullspace_detected")),
                    "phi_nullspace_detected": bool(g("phi_nullspace_detected")),
                    "phi_kernel_defect": g("phi_kernel_defect"), "phi_flux_defect_abs": g("phi_flux_defect_abs"),
                    "phi_pinned_dof": g("phi_pinned_dof"), "potential_order": g("potential_order"),
                    "mhd_quad_order": g("mhd_quad_order"), "mhd_aux_unknowns": g("mhd_aux_unknowns"),
                    "current_form": g("current_form"), "kernel_defect": g("kernel_defect"),
                    "flux_defect": g("flux_defect"), "flux_defect_abs": g("flux_defect_abs"),
                    "line_search_min_step_accepts": g("line_search_min_step_accepts"),
                    "quad_orders": _jsonable(g("quad_orders")), "ranks": g("ranks"), "threads": g("threads"),
                    "lu_threads": g("lu_threads"), "options": {k: _jsonable(v) for k, v in opts.items() if k not in ("mhd",)}},
         "seconds": sec}
    if family is not None:
        e["mesh"]["family"] = family
        e["mesh"]["refinement_level"] = refinement_level
    return e


class Run:
    """Runs, series, quantities and notes of one case; every solve goes through solve()."""

    def __init__(self, params, quick):
        self.P, self.quick = params, quick
        self.ident = core_identity()
        self.runs, self.ser, self.acc, self.notes = [], {}, Acc(), []
        self.failed = []
        self.rtol = 1e-12

    def solve(self, level, mesh, info, h, mu, rho, force, bcs, mhd, family=None, refinement_level=None, **kw):
        opts = dict(form="laplacian", geometry="planar25d", vectorized=True, lu_threads=1,
                    newton_rtol=self.rtol, newton_maxit=30, n_picard=3)
        opts.update(kw)
        t0 = time.time()
        sol = fluids.solve_navier_stokes(mesh, mu, rho, force, bcs, mhd=mhd, **opts)
        sec = time.time() - t0
        self.runs.append(run_entry(level, sol, sec, h, info, family, refinement_level, opts))
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


def bc(attr, kind, value=None):
    d = {"attribute": attr, "kind": kind}
    if value is not None:
        d["value"] = value
    return d


def el(attr, kind, value=None):
    d = {"attribute": attr, "kind": kind}
    if value is not None:
        d["value"] = value
    return d


def mhd_opts(sigma, B, Ez, electric, q=None):
    d = {"sigma": sigma, "B": B, "E_z": float(Ez), "electric": electric, "potential_order": 2}
    if q is not None:
        d["q"] = q
    return d


SUFFIX = {"uL2": "uL2", "uH1": "uH1", "pL2m": "pL2", "phiL2": "phiL2", "phiL2m": "phiL2", "phiH1": "phiH1",
          "JL2": "JL2", "divL2": "divL2"}
MAG = {"uL2": "u_magnitude", "uH1": "u_magnitude", "pL2": "p_magnitude", "phiL2": "phi_magnitude",
       "phiH1": "phi_magnitude", "JL2": "J_magnitude", "divL2": "u_magnitude"}


def put_series(R, prefix, rows, suffixes):
    """rows: [(level, norms)] of the converged levels."""
    if not rows:
        return
    for suf in suffixes:
        key = SUFFIX[suf]
        R.ser[f"{prefix}_{suf}"] = series([lv for lv, _ in rows], [n[key] for _, n in rows], rows[-1][1][MAG[key]])


def shared_mismatch(R, V, seg, att, data):
    owners = {}
    for (i, j), a in zip(seg, att):
        if int(a) in data:
            for v in (int(i), int(j)):
                owners.setdefault(v, set()).add(int(a))
    shared = [(v, sorted(s)) for v, s in owners.items() if len(s) > 1]
    if shared:
        pts = np.array([V[v] for v, _ in shared])
        R.acc.max("dirichlet_shared_node_mismatch_max",
                  F1.dirichlet_shared_node_mismatch(pts, [[data[a] for a in s] for _, s in shared]))
    else:
        R.acc.max("dirichlet_shared_node_mismatch_max", 0.0)


def nodal_defects(R, sol, vel_attrs, g_u, phi_attr=None, g_phi=None):
    P, U = sol.boundary_nodes(list(vel_attrs))
    if len(P):
        R.acc.max("dirichlet_nodal_defect_max", F1.dirichlet_nodal_defect(np.asarray(P, float), np.asarray(U, float), g_u))
    if phi_attr:
        P, U = sol.potential_boundary_nodes(list(phi_attr))
        if len(P):
            R.acc.max("dirichlet_nodal_defect_max",
                      F1.dirichlet_nodal_defect(np.asarray(P, float), np.asarray(U, float), g_phi))


# ============================================================================================== MMS (T1, T2)
def _mms_force(fld, rho):
    f = fld.f(rho)

    def g(X, Y):
        a = np.array(f(X, Y), dtype=float, copy=True)
        a[..., 2] += fld.Gz
        return a
    return g


def mms_mixed(params, workdir, quick):
    R = Run(params, quick)
    fld = C.load_field(params["mms_data"])
    vel = (1, 2, 3, 4)

    def electric():
        out = []
        for e, b in fld.phi_bc.items():
            a = EDGE_ATTR[e]
            if b == "dirichlet_phi":
                out.append(el(a, "potential", fld.phi))
            elif b == "jn":
                out.append(el(a, "normal_current", fld.jn(e)))
            else:
                out.append(el(a, "insulating"))
        return out
    dir_attrs = [EDGE_ATTR[e] for e, b in fld.phi_bc.items() if b == "dirichlet_phi"]
    finest = None
    for variant, rho_name, fam in (("stokes", "stokes", "structured"), ("ns", "ns", "structured"),
                                   ("stokes", "stokes", "distorted")):
        rho = params["rho"][rho_name]
        out = _mms_family2(R, fld, fam, rho, variant, electric, vel, dir_attrs, quick, params)
        if variant == "stokes" and fam == "structured" and out:
            finest = out[-1]
    # solver sensitivity: the finest structured Stokes level again with the tight tolerance
    if finest is not None:
        lvl, n, sol, ev, mesh, info, h, V, T = finest
        R.acc.max("stokes_struct_uL2_finest", n["uL2"])
        tight, ok = R.solve("stokes_struct_tight", mesh, info, h, fld.mu, 0.0, _mms_force(fld, 0.0),
                            [bc(a, "dirichlet", fld.u) for a in vel],
                            mhd_opts(fld.sigma, fld.B, fld.Ez, electric(), q=fld.q), family="tight",
                            refinement_level=0, newton_rtol=1e-14, newton_maxit=5)
        if ok:
            R.acc.max("stokes_struct_uL2_finest_tight", C.norms(Evaluator(tight), fld, "mean_free", "determined")["uL2"])
    return R.result()


def _mms_family2(R, fld, fam, rho, variant, electric, vel, dir_attrs, quick, params):
    """One (rho, family) MMS sweep of T1: series prefix {variant}_{struct|dist}."""
    m = params["meshes"][fam]
    ks = _levels(m["N"] if fam == "structured" else m["refinements"], quick)
    prefix = f"{variant}_{'struct' if fam == 'structured' else 'dist'}"
    rows, out = [], []
    for i, kk in enumerate(ks):
        V, T = C.mms_mesh(fam, kk)
        Vc = C.mms_mesh(fam, ks[i - 1])[0] if i else None
        mesh, seg, att = make_mesh(V, T, rect_classify(0.0, 1.0, 0.0, 1.0))
        info = verify(mesh, V, T, Vc)
        h = m["h_nominal"][i]
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve(lvl, mesh, info, h, fld.mu, float(rho), _mms_force(fld, rho),
                          [bc(a, "dirichlet", fld.u) for a in vel],
                          mhd_opts(fld.sigma, fld.B, fld.Ez, electric(), q=fld.q), family=prefix, refinement_level=i)
        d = sol.diagnostics
        R.acc.min("nullspace_detected", int(d["nullspace_detected"]))
        R.acc.max("phi_nullspace_detected", int(d["phi_nullspace_detected"]))
        shared_mismatch(R, V, seg, att, {a: fld.u for a in vel})
        if not ok:
            continue
        ev = Evaluator(sol)
        n = C.norms(ev, fld, pressure="mean_free", phi="determined")
        nodal_defects(R, sol, vel, fld.u, dir_attrs, fld.phi)
        rows.append((lvl, n))
        out.append((lvl, n, sol, ev, mesh, info, h, V, T))
    put_series(R, prefix, rows, ["uL2", "uH1", "pL2m", "phiL2", "phiH1", "JL2", "divL2"])
    return out


def mms_insulating(params, workdir, quick):
    R = Run(params, quick)
    fld = C.load_field(params["mms_data"])
    vel = (1, 2, 3, 4)
    Bv = [float(c) for c in (0.0, 1.2, 0.8)]
    m = params["meshes"]["structured"]
    ks = _levels(m["N"], quick)
    for variant in ("stokes", "ns"):
        rho = float(params["rho"][variant])
        prefix = f"{variant}_struct"
        rows = []
        for i, N in enumerate(ks):
            V, T = C.mms_mesh("structured", N)
            Vc = C.mms_mesh("structured", ks[i - 1])[0] if i else None
            mesh, seg, att = make_mesh(V, T, rect_classify(0.0, 1.0, 0.0, 1.0))
            info = verify(mesh, V, T, Vc)
            lvl = f"{prefix}_L{i}"
            sol, ok = R.solve(lvl, mesh, info, m["h_nominal"][i], fld.mu, rho, _mms_force(fld, rho),
                              [bc(a, "dirichlet", fld.u) for a in vel],
                              mhd_opts(fld.sigma, fld.B, fld.Ez, [el(a, "insulating") for a in vel], q=fld.q),
                              family=prefix, refinement_level=i)
            d = sol.diagnostics
            R.acc.min("nullspace_detected", int(d["nullspace_detected"]))
            R.acc.min("phi_nullspace_detected", int(d["phi_nullspace_detected"]))
            if not ok:
                continue
            ev = Evaluator(sol)
            n = C.norms(ev, fld, pressure="mean_free", phi="mean_free")
            nodal_defects(R, sol, vel, fld.u)
            I = C.integrals(ev, 1.6, Bv, fld.Ez, fld.mu)
            R.acc.max("lorentz_joule_defect_max", C.lorentz_joule_defect(I, fld.Ez))
            R.acc.max("net_current_rel_max", C.net_current_rel(I))
            R.acc.max("weak_current_residual_rel_max", C.weak_current_residual(ev, fld.sigma, fld.B, fld.Ez)["max_rel"])
            rows.append((lvl, n))
        put_series(R, prefix, rows, ["uL2", "uH1", "pL2m", "phiL2m", "phiH1", "JL2"])
    return R.result()


# ============================================================================================== A1 Hartmann
def _strip_mesh(Ha, k):
    V, T = C.hartmann_strip_mesh(Ha, k)
    Vc = C.hartmann_strip_mesh(Ha, k - 1)[0] if k else None
    return V, T, Vc


def _hartmann_row(R, key, Ha, k, i, mode, circuit, phi_bc, h, family):
    """One Hartmann solve (in-plane or duct variant); returns (row dict, ok)."""
    B0 = Ha / (A_ * math.sqrt(SIGMA / MU))
    ex = C.HartmannExact(MU, SIGMA, B0, G, mode, circuit)
    V, T, Vc = _strip_mesh(Ha, k)
    mesh, seg, att = make_mesh(V, T, rect_classify(0.0, C.LX_STRIP, -1.0, 1.0))
    info = verify(mesh, V, T, Vc)
    lvl = f"{family}_L{i}"
    z = None
    if mode == "inplane":
        bcs = [bc(1, "dirichlet"), bc(3, "dirichlet"), bc(4, "traction", vec3(G * C.LX_STRIP, 0.0, 0.0)),
               bc(2, "do_nothing")]
        electric = [el(a, "insulating") for a in (1, 2, 3, 4)]
        force, Ez = None, ex.h.E
    else:
        Wd = lambda X, Y: np.asarray(ex.u(X, Y), float)   # noqa: E731
        bcs = [bc(1, "dirichlet"), bc(3, "dirichlet"), bc(2, "dirichlet", Wd), bc(4, "dirichlet", Wd)]
        force, Ez = vec3(0.0, 0.0, G), 0.0
        if phi_bc == "jn":
            def jx(X, Y):
                return SIGMA * (ex.h.E - B0 * ex.h.U(Y)) + 0.0 * X
            electric = [el(4, "normal_current", lambda X, Y: -jx(X, Y)), el(2, "normal_current", jx),
                        el(1, "insulating"), el(3, "insulating")]
        else:
            electric = [el(4, "potential", lambda X, Y: 0.0 * X),
                        el(2, "potential", lambda X, Y: -ex.h.E * C.LX_STRIP + 0.0 * X),
                        el(1, "insulating"), el(3, "insulating")]
    sol, ok = R.solve(lvl, mesh, info, h, MU, 0.0, force, bcs, mhd_opts(float(SIGMA), const_B([0.0, B0, 0.0]), Ez, electric),
                      family=family, refinement_level=i)
    row = {"ok": ok, "lvl": lvl, "Ha": Ha, "ex": ex, "sol": sol, "B0": B0}
    d = sol.diagnostics
    row["p_null"] = int(d["nullspace_detected"])
    row["phi_null"] = int(d["phi_nullspace_detected"])
    if not ok:
        return row
    ev = Evaluator(sol)
    row["ev"] = ev
    row["norms"] = C.norms(ev, ex, pressure="determined" if mode == "inplane" else "mean_free",
                           phi="mean_free" if (mode == "inplane" or phi_bc == "jn") else "determined")
    I = C.integrals(ev, SIGMA, [0.0, B0, 0.0], Ez, MU)
    axis = 0 if mode == "inplane" else 2
    row["Q_h"] = I["flow"][axis] / C.LX_STRIP
    row["Q"] = ex.h.Q
    row["I_z_h"] = I["I_z"] / C.LX_STRIP
    row["potential"] = C.potential_spread(ev)
    if mode == "duct":
        row["axial"] = C.weak_axial_momentum_residual(ev, MU, SIGMA, [0.0, B0, 0.0], G, dirichlet=lambda P: (abs(P[0]) < 1e-12 or abs(P[0] - C.LX_STRIP) < 1e-12
                                                                                      or abs(abs(P[1]) - 1) < 1e-12))["max_rel"]
    return row


A1_SERIES = {
    "inplane_short_Ha1": ["uL2", "uH1"], "inplane_short_Ha10": ["uL2", "uH1"], "inplane_short_Ha100": ["uL2", "uH1"],
    "inplane_short_Ha1000": ["uH1"], "twolayer_Ha10": ["uL2", "uH1", "JL2"],
    "duct_open_Ha10": ["uL2", "uH1", "JL2"], "duct_jn_Ha10": ["uL2", "uH1", "JL2"],
    "duct_open_Ha100": ["uL2", "uH1", "JL2"], "duct_jn_Ha100": ["uL2", "uH1", "JL2"],
}


def hartmann(params, workdir, quick):
    R = Run(params, quick)
    strip = params["meshes"]["strip"]
    hs = strip["h_nominal"]
    ks = _levels(strip["k"], quick)
    jobs = []
    for Ha in strip["Ha"]:
        jobs.append(("inplane", "short", "dirichlet", Ha))
    for Ha in (10.0, 100.0):
        jobs += [("inplane", "open", "dirichlet", Ha), ("duct", "open", "dirichlet", Ha), ("duct", "open", "jn", Ha)]
    finest = {}
    for (mode, circ, pbc, Ha) in jobs:
        key = f"{mode}_{'jn' if pbc == 'jn' else circ}_Ha{Ha:g}"
        rows = []
        for i, k in enumerate(ks):
            r = _hartmann_row(R, key, Ha, k, i, mode, circ, pbc, hs[i], key)
            if mode == "inplane":
                R.acc.max("nullspace_detected_inplane", r["p_null"])
                R.acc.min("phi_nullspace_detected_inplane", r["phi_null"])
            elif pbc == "jn":
                R.acc.min("phi_nullspace_detected_duct_jn", r["phi_null"])
            else:
                R.acc.max("phi_nullspace_detected_duct", r["phi_null"])
            if not r["ok"]:
                continue
            if mode == "inplane":
                R.acc.max("inplane_phi_spread_max", max(r["potential"]["phi_spread"], r["potential"]["gradphi_max"]))
            else:
                R.acc.max("axial_momentum_residual_max", r["axial"])
            rows.append(r)
        if not rows:
            continue
        put_series_rows(R, key, rows)
        f = rows[-1]
        R.acc.q[f"{key}_Q_h_finest"] = f["Q_h"]
        finest[key] = f
        if key.startswith("inplane_open_Ha"):
            R.acc.q[f"{key}_net_axial_current_rel_finest"] = abs(f["I_z_h"]) / (SIGMA * (Ha / 2) * f["Q"])
    # two-layer channel
    rows = []
    B0 = 10.0 / (A_ * math.sqrt(SIGMA / MU))
    ex2 = C.HartmannTwoLayerExact(MU, SIGMA, 0.5, B0, G)
    for i, k in enumerate(ks):
        V, T, Vc = _strip_mesh(10.0, k)
        mesh, seg, att = make_mesh(V, T, rect_classify(0.0, C.LX_STRIP, -1.0, 1.0))
        info = verify(mesh, V, T, Vc)
        lvl = f"twolayer_Ha10_L{i}"
        sol, ok = R.solve(lvl, mesh, info, hs[i], MU, 0.0, None,
                          [bc(1, "dirichlet"), bc(3, "dirichlet"), bc(4, "traction", vec3(G * C.LX_STRIP, 0.0, 0.0)),
                           bc(2, "do_nothing")],
                          mhd_opts(lambda X, Y: ex2.sigma(X, Y), const_B([0.0, B0, 0.0]), 0.0,
                                   [el(a, "insulating") for a in (1, 2, 3, 4)]),
                          family="twolayer_Ha10", refinement_level=i)
        if not ok:
            continue
        ev = Evaluator(sol)
        n = C.norms(ev, ex2, pressure="determined", phi="mean_free")
        q, xq, W = C._weights(ev)
        uh, _ = ev.velocity(q)
        rows.append({"lvl": lvl, "norms": n, "Q_h": float((W * uh[..., 0]).sum()) / C.LX_STRIP})
    if rows:
        put_series_rows(R, "twolayer_Ha10", rows)
        R.acc.q["twolayer_Ha10_Q_h_finest"] = rows[-1]["Q_h"]
    # Ha = 0 patch test (uniform strip, k = 1)
    r = _hartmann_row(R, "patch", 0.0, 1, 0, "inplane", "short", "dirichlet", 0.03125, "patch_Ha0")
    if r["ok"]:
        R.acc.q["Ha0_patch_u_rel"] = r["norms"]["uL2"] / r["norms"]["u_magnitude"]
        R.acc.q["Ha0_patch_p_rel"] = r["norms"]["pL2"] / r["norms"]["p_magnitude"]
    rel = []
    for Ha in (10.0, 100.0, 1000.0):
        f = finest.get(f"inplane_short_Ha{Ha:g}")
        if f:
            rel.append(f["norms"]["uL2"] / f["norms"]["u_magnitude"])
    if len(rel) == 3:
        R.acc.q["layer_robustness_ratio"] = max(rel) / min(rel)
    return R.result()


def put_series_rows(R, key, rows):
    names = A1_SERIES.get(key, [])
    if names:
        put_series(R, key, [(r["lvl"], r["norms"]) for r in rows], names)


# ============================================================================================== A2 / A3 / X1 ducts
def _duct_row(R, kind, Ha, k, i, sigma=SIGMA, B0=None, extra=False, h=None, family=None):
    B0 = Ha / (A_ * math.sqrt(SIGMA / MU)) if B0 is None else B0
    V, T = C.duct_mesh(Ha, k)
    Vc = C.duct_mesh(Ha, k - 1)[0] if k else None
    mesh, seg, att = make_mesh(V, T, rect_classify(-1.0, 1.0, -1.0, 1.0))
    info = verify(mesh, V, T, Vc)
    lvl = f"{family or kind}_Ha{Ha:g}_L{i}" if B0 is None or sigma == SIGMA else f"{family or kind}_scaled_L{i}"
    if kind == "hunt":
        electric = [el(1, "potential", lambda X, Y: 0.0 * X), el(3, "potential", lambda X, Y: 0.0 * X),
                    el(2, "insulating"), el(4, "insulating")]
    else:
        electric = [el(a, "insulating") for a in (1, 2, 3, 4)]
    sol, ok = R.solve(lvl, mesh, info, h if h is not None else 1.0 / (C.BASE_N * 2 ** k), MU, 0.0, vec3(0.0, 0.0, G),
                      [bc(a, "dirichlet") for a in (1, 2, 3, 4)],
                      mhd_opts(float(sigma), const_B([0.0, B0, 0.0]), 0.0, electric),
                      family=family or f"{kind}_Ha{Ha:g}", refinement_level=i)
    row = {"ok": ok, "Ha": Ha, "k": k, "lvl": lvl, "sol": sol, "phi_null": int(sol.diagnostics["phi_nullspace_detected"])}
    if not ok:
        return row, None
    ev = Evaluator(sol)
    I = C.integrals(ev, sigma, [0.0, B0, 0.0], 0.0, MU)
    row.update({"Q_h": I["flow"][2], "w_center_h": C.point_value(ev, (0.0, 0.0), "w"),
                "lorentz_joule_defect": C.lorentz_joule_defect(I, 0.0) if I["joule"] > 0 else 0.0,
                "power_balance_defect": C.power_balance_defect(I, G),
                "net_current_rel": C.net_current_rel(I) if I["int_absJ"] > 0 else 0.0,
                "potential": C.potential_spread(ev)})
    dir_pred = (lambda P: abs(abs(P[1]) - 1) < 1e-12) if kind == "hunt" else None
    if B0 > 0:
        row["weak_residual"] = C.weak_current_residual(
            ev, lambda X, Y: np.full(np.shape(X), sigma),
            lambda X, Y: np.broadcast_to(np.array([0.0, B0, 0.0]), np.shape(X) + (3,)), 0.0, dirichlet=dir_pred)["max_rel"]
    if kind == "hunt":
        row["phi_side_h"] = C.point_value(ev, (1.0, 0.0), "phi")
        row["M_direct_h"] = C.wall_current_moment_direct(ev, sigma, [0.0, B0, 0.0])
        row["M_var_h"] = C.wall_current_moment_variational(ev, sigma, [0.0, B0, 0.0])
    else:
        row["dphi_side_h"] = C.point_value(ev, (1.0, 0.0), "phi") - C.point_value(ev, (-1.0, 0.0), "phi")
    row["axial"] = C.weak_axial_momentum_residual(
        ev, MU, sigma, [0.0, B0, 0.0], G, dirichlet=lambda P: abs(abs(P[0]) - 1) < 1e-12 or abs(abs(P[1]) - 1) < 1e-12)["max_rel"]
    if extra and B0 > 0:
        row["core"] = C.core_current_error(ev, sigma, [0.0, B0, 0.0], C.core_J_exact(kind, MU, sigma, B0, G))["rel"]
    return row, ev


def _duct_case(params, quick, kind):
    R = Run(params, quick)
    R.rtol = DUCT_RTOL
    duct = params["meshes"]["duct"]
    hs = duct["h_nominal"]
    ks = _levels(duct["k"], quick)
    by_ha = {}
    for Ha in params["Ha"]:
        rows = []
        for i, k in enumerate(ks):
            last = i == len(ks) - 1
            r, _ = _duct_row(R, kind, Ha, k, i, extra=(last and Ha in (100.0, 1000.0)), h=hs[i])
            R.acc.min("phi_nullspace_detected", r["phi_null"]) if kind == "shercliff" else \
                R.acc.max("phi_nullspace_detected", r["phi_null"])
            if r["ok"]:
                rows.append(r)
        by_ha[Ha] = rows
        if not rows:
            continue
        f = rows[-1]
        tag = f"Ha{Ha:g}"
        for name in ("Q_h", "w_center_h", "phi_side_h", "dphi_side_h", "M_var_h", "M_direct_h"):
            if name in f and not (name == "dphi_side_h" and Ha == 0.0):
                R.acc.q[f"{tag}_{name}_finest"] = f[name]
        for r in rows:
            R.acc.max("power_balance_defect_max", r["power_balance_defect"])
            R.acc.max("axial_momentum_residual_max", r["axial"])
            if Ha > 0:
                R.acc.max("lorentz_joule_defect_max", r["lorentz_joule_defect"])
                if kind == "shercliff":      # (Hunt: the current closes through the conducting walls, not a check)
                    R.acc.max("net_current_rel_max", r["net_current_rel"])
                R.acc.max("weak_current_residual_rel_Ha1000_max" if Ha == 1000.0 else "weak_current_residual_rel_max",
                          r["weak_residual"])
            else:
                R.acc.max("Ha0_phi_spread_max", max(r["potential"]["phi_spread"], r["potential"]["gradphi_max"]))
        if "core" in f:
            R.acc.q[f"{tag}_core_current_rel_finest"] = f["core"]
    # (sigma, B0) -> (4 sigma, B0/2) scaling identity at Ha = 100, k = 1
    Ha, k = 100.0, 1
    B0 = Ha / (A_ * math.sqrt(SIGMA / MU))
    r1, e1 = _duct_row(R, kind, Ha, k, 0, family=f"{kind}_scale_a")
    r2, e2 = _duct_row(R, kind, Ha, k, 0, sigma=4 * SIGMA, B0=B0 / 2, family=f"{kind}_scale_b")
    if e1 is not None and e2 is not None:
        s = C.relative_difference(e1, e2, scale_phi_b=2.0)
        R.acc.q["scaling_sigma_B_u_rel"] = s["u_rel"]
        R.acc.q["scaling_sigma_B_phi_rel"] = s["phi_rel"]
    return R.result()


def shercliff_duct(params, workdir, quick):
    return _duct_case(params, quick, "shercliff")


def hunt_duct(params, workdir, quick):
    return _duct_case(params, quick, "hunt")


def xcode_duct(params, workdir, quick):
    R = Run(params, quick)
    R.rtol = DUCT_RTOL
    duct = params["meshes"]["duct"]
    for kind in ("hunt", "shercliff"):
        for i, k in enumerate(duct["k"]):
            r, _ = _duct_row(R, kind, 100.0, k, i, h=duct["h_nominal"][i], family=f"x_{kind}")
            if not r["ok"]:
                continue
            tag = f"{kind}_Ha100_k{k}"
            names = ("M_var_h", "Q_h", "phi_side_h", "w_center_h") if kind == "hunt" else ("Q_h", "dphi_side_h", "w_center_h")
            for nm in names:
                R.acc.q[f"{tag}_{nm}"] = r[nm]
    return R.result()


# ============================================================================================== A4 annulus
def annulus_varB(params, workdir, quick):
    R = Run(params, quick)
    R.rtol = ANNULUS_RTOL
    ks = _levels(params["meshes"]["annulus"]["k"], quick)
    hs = params["meshes"]["annulus"]["h_nominal"]
    for variant, var in AC.VARIANTS.items():
        ex = AC.exact(variant)
        Cc = var["C"]
        if var["field"] == "radial":
            def Bf(X, Y, Cc=Cc):
                r2 = X * X + Y * Y
                return np.stack([Cc * X / r2, Cc * Y / r2, 0 * X], axis=-1)
        else:
            def Bf(X, Y, Cc=Cc):
                r2 = X * X + Y * Y
                return np.stack([-Cc * Y / r2, Cc * X / r2, 0 * X], axis=-1)
        Wv = float(var.get("W", 0.0))
        pot = var["potential"]
        if pot == "grounded":
            electric = [el(1, "potential", lambda X, Y: 0.0 * X), el(2, "potential", lambda X, Y: 0.0 * X)]
        elif pot == "dirichlet":
            electric = [el(1, "potential", lambda X, Y: 0.0 * X), el(2, "potential", lambda X, Y, V=var["V"]: V + 0.0 * X)]
        else:
            electric = [el(1, "insulating"), el(2, "insulating")]
        rows = []
        prefix = variant
        for i, k in enumerate(ks):
            V, T, X6 = AC.annulus_mesh(variant, k)
            Vc = AC.annulus_mesh(variant, ks[i - 1])[0] if i else None
            mesh, seg, att = make_mesh(V, T, lambda x, y: circle_classify(x, y), X6)
            info = verify(mesh, V, T, Vc, X6=X6)
            lvl = f"{prefix}_L{i}"
            sol, ok = R.solve(lvl, mesh, info, hs[i], AC.PARAMS["mu"], 0.0, vec3(0.0, 0.0, AC.PARAMS["G"]),
                              [bc(1, "dirichlet", vec3(0.0, 0.0, Wv)), bc(2, "dirichlet")],
                              mhd_opts(float(AC.PARAMS["sigma"]), Bf, var["E_z"], electric),
                              family=prefix, refinement_level=i)
            d = sol.diagnostics
            R.acc.min("nullspace_detected", int(d["nullspace_detected"]))
            # per-level flags are aggregated over ALL levels: max for the flags the fixed items require to be
            # 0 (potential Dirichlet walls), min for those required to be 1 (insulating walls)
            if variant == "azimuthal_cond" or pot == "grounded":
                R.acc.max(f"phi_nullspace_detected_{variant}", int(d["phi_nullspace_detected"]))
            else:
                R.acc.min(f"phi_nullspace_detected_{variant}", int(d["phi_nullspace_detected"]))
            if not ok:
                continue
            ev = Evaluator(sol)
            n = C.norms(ev, ex, pressure="mean_free", phi="mean_free" if d["phi_nullspace_detected"] else "determined")
            row = {"lvl": lvl, "norms": n, "Q_h": AC.flow_rate(ev), "potential": AC.potential_spread(ev)}
            if var["field"] == "radial":
                row["L_h"] = AC.angular_momentum(ev)
            elif pot == "dirichlet":
                row["I_var_h"] = AC.wall_current_variational(ev, ex)
                row["I_direct_h"] = AC.wall_current_direct(ev, ex, X6)
            rows.append(row)
        if not rows:
            continue
        if var["field"] == "radial":
            suf = ["uL2", "uH1", "JL2"]
        elif pot == "dirichlet":
            suf = ["uL2", "uH1", "pL2m", "phiL2", "phiH1", "JL2"]
        else:
            suf = ["uL2", "uH1", "pL2m", "phiL2m", "phiH1", "JL2"]
        put_series(R, prefix, [(r["lvl"], r["norms"]) for r in rows], suf)
        f = rows[-1]
        R.acc.q[f"{variant}_Q_h_finest"] = f["Q_h"]
        if "L_h" in f:
            R.acc.q[f"{variant}_L_h_finest"] = f["L_h"]
            R.acc.q[f"{variant}_phi_spread_finest"] = f["potential"]["phi_spread"]
        if "I_var_h" in f:
            R.acc.q[f"{variant}_I_var_h_finest"] = f["I_var_h"]
            R.acc.q[f"{variant}_I_direct_h_finest"] = f["I_direct_h"]
    return R.result()


# ============================================================================================== A5 suction
def hartmann_suction(params, workdir, quick):
    R = Run(params, quick)
    strip = params["meshes"]["strip"]
    ks = _levels(strip["k"], quick)
    hs = strip["h_nominal"]
    R.rtol = 1e-12
    for variant, v in SC.VARIANTS.items():
        ex = SC.exact(variant)
        mode, Ha, Vs = v["mode"], v["Ha"], v["V"]
        B0 = ex.B0
        rows = []
        for i, k in enumerate(ks):
            Vv, T = SC.mesh(variant, k)
            Vc = SC.mesh(variant, k - 1)[0] if k else None
            mesh, seg, att = make_mesh(Vv, T, rect_classify(0.0, C.LX_STRIP, -1.0, 1.0))
            info = verify(mesh, Vv, T, Vc)
            lvl = f"{variant}_L{i}"
            wall = vec3(0.0, Vs, 0.0)
            if mode == "inplane":
                bcs = [bc(1, "dirichlet", wall), bc(3, "dirichlet", wall),
                       bc(4, "traction", vec3(SC.PARAMS["G"] * C.LX_STRIP, 0.0, 0.0)), bc(2, "do_nothing")]
                electric = [el(a, "insulating") for a in (1, 2, 3, 4)]
                force = None
            else:
                Wd = lambda X, Y, ex=ex: np.asarray(ex.u(X, Y), float)   # noqa: E731
                bcs = [bc(1, "dirichlet", wall), bc(3, "dirichlet", wall), bc(2, "dirichlet", Wd), bc(4, "dirichlet", Wd)]
                electric = [el(4, "potential", lambda X, Y: 0.0 * X), el(2, "potential", lambda X, Y: 0.0 * X),
                            el(1, "insulating"), el(3, "insulating")]
                force = vec3(0.0, 0.0, SC.PARAMS["G"])
            sol, ok = R.solve(lvl, mesh, info, hs[i], SC.PARAMS["mu"], SC.PARAMS["rho"], force, bcs,
                              mhd_opts(float(SC.PARAMS["sigma"]), const_B([0.0, B0, 0.0]), 0.0, electric),
                              family=variant, refinement_level=i)
            d = sol.diagnostics
            # per level, aggregated over all levels: inplane requires phi = 1 / p = 0, duct phi = 0 / p = 1
            (R.acc.min if mode == "inplane" else R.acc.max)(f"phi_nullspace_detected_{variant}", int(d["phi_nullspace_detected"]))
            (R.acc.max if mode == "inplane" else R.acc.min)(f"nullspace_detected_{variant}", int(d["nullspace_detected"]))
            if not ok:
                continue
            ev = Evaluator(sol)
            n = C.norms(ev, ex, pressure="determined" if mode == "inplane" else "mean_free",
                        phi="mean_free" if mode == "inplane" else "determined")
            rows.append({"lvl": lvl, "norms": n, "Q_h": SC.flow_rate(ev, mode)})
        if rows:
            put_series(R, variant, [(r["lvl"], r["norms"]) for r in rows], ["uL2", "uH1", "JL2"])
            R.acc.q[f"{variant}_Q_h_finest"] = rows[-1]["Q_h"]
    return R.result()
