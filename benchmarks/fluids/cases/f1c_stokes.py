"""F1c 3D Stokes: the case runner (set-up, distributed solve, Evaluator).  Every norm, identity, functional and
check value is computed by the check module f1c_stokes_check (imported as C); this module builds the fixed
meshes with C's builders, verifies the solver's mesh against them, solves through the public 3D fluids API
(axionfem.fluids.solve_stokes_3d) on 16 ranks of one node (cases/fluids_mpi.launch, worker cases/f1c_mpi_worker.py)
and hands each rank's part of the solution to C through an Evaluator; the parent adds the partial sums of the ranks.

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two levels only.

Rank proof: every run entry reports comm_size (the solve's communicator), shared_dofs
(the solver's shared/ghost DOFs) and a digest of the fixed element ids of each rank's evaluator; the three
quantities comm_size_mismatch_runs, partition_failures and shared_dofs_min_runs are computed from the full id lists
exactly as the reference rule tests/test_fluids_stokes_3d_check.py::rank_proof_quantities does (the full lists are not
stored in the result: up to 2 x 10^5 ids per run).

Conventions: form "lap" = mu grad u, "sym" = 2 mu eps(u); traction data g = sigma_form n with
the outward normal; grad-div gamma in viscosity units.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import f1c_mpi_worker  # noqa: F401  (the MPI child; imported here so that a missing module fails early)
import f1c_stokes_check as C

HERE = Path(__file__).resolve().parent
WORKER = HERE / "f1c_mpi_worker.py"
FORM = {"lap": "laplacian", "sym": "stress"}
RANKS = 16
FACE_TOL = 1e-12        # fixed node predicate (run.parameters.boundary_faces.predicate)
SPHERE_TOL = 1e-9


# ============================================================================================== small helpers
def plain(o):
    """JSON default: numpy scalars/arrays and read-only mappings."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "keys"):
        return {k: o[k] for k in o.keys()}
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _levels(seq, quick, n=2):
    seq = list(seq)
    return seq[:n] if quick else seq


def _js(o):
    """A plain-JSON copy of fixed parameters (read-only mappings)."""
    return json.loads(json.dumps(o, default=plain))


# ============================================================================================== meshes and faces
_MESH_CACHE: dict = {}


def build_mesh(spec: dict) -> C.FrozenMesh:
    """The fixed mesh of a level, from the check module's builders only."""
    key = json.dumps(spec, sort_keys=True)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    fam = spec["family"]
    if fam == "kuhn":
        m = C.cube_tet(int(spec["N"]))
    elif fam == "standard":
        m = C.cube_tet(int(spec["N"]), "standard")
    elif fam == "five":
        m = C.five_tet_cube(int(spec["N"]))
    elif fam == "dist":
        m = C.cube_tet_distorted(int(spec["k"]), N0=int(spec["N0"]), rel=float(spec["rel"]), seed=int(spec["seed"]))
    elif fam == "hex":
        m = C.cube_hex(int(spec["N"]))
    elif fam == "rot":
        m = C.rotated_cube_tet(int(spec["N"]), np.array(spec["R"], float))
    elif fam == "duct":
        m = C.duct_tet(float(spec["h"]))
    elif fam in ("sphere_hex", "sphere_tet"):
        m = C.sphere_shell(int(spec["n"]), fam.split("_")[1])
    else:
        raise ValueError(f"unknown mesh family {fam}")
    _MESH_CACHE.clear()          # one level at a time (memory)
    _MESH_CACHE[key] = m
    return m


class Faces:
    """The named boundary faces of a family: attribute, outward normal (physical), node predicate."""

    def __init__(self, spec: dict):
        fam = spec["family"]
        self.R = np.array(spec["R"], float) if fam == "rot" else np.eye(3)
        self.sphere = fam.startswith("sphere")
        if self.sphere:
            self.names = {"body": (1, 1.0), "outer": (2, 4.0)}
        else:
            hi = (1.0, 0.5, 2.0) if fam == "duct" else (1.0, 1.0, 1.0)
            self.names = {}
            for ax, c in enumerate("xyz"):
                self.names[c + "0"] = (2 * ax + 1, (ax, 0.0))
                self.names[c + "1"] = (2 * ax + 2, (ax, hi[ax]))

    def attr(self, name):
        return self.names[name][0]

    def local(self, X):
        return np.asarray(X, float) @ self.R           # x_loc = R^T x (row vectors)

    def on(self, name, X, tol=None):
        X = np.asarray(X, float)
        if self.sphere:
            return np.abs(np.linalg.norm(X, axis=-1) - self.names[name][1]) < (tol or SPHERE_TOL)
        ax, val = self.names[name][1]
        return np.abs(self.local(X)[..., ax] - val) < (tol or FACE_TOL)

    def pred(self, name):
        return lambda X: self.on(name, X)

    def normal(self, name):
        """Outward unit normal of a planar face (physical frame)."""
        ax, val = self.names[name][1]
        n = np.zeros(3)
        n[ax] = 1.0 if val > 0 else -1.0
        return self.R @ n

    def classify(self, P):
        """Attribute of a boundary face from its vertex coordinates P[fv, 3]: the face all of whose vertices lie on it."""
        hit = [a for nm, (a, _) in self.names.items() if np.all(self.on(nm, P, 1e-9))]
        if len(hit) != 1:
            raise RuntimeError(f"boundary face {P.tolist()} lies on {len(hit)} named faces")
        return hit[0]


def boundary_faces(fm: C.FrozenMesh, faces: Faces):
    """(faces (nb, 3|4) int32 in cyclic order, attributes (nb,)) of the exterior faces of the fixed mesh."""
    loc = C.TET_FACES if fm.kind == "tet" else C.HEX_FACES
    E = np.asarray(fm.E, np.int64)
    allf = np.concatenate([E[:, list(f)] for f in loc])
    key = np.sort(allf, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bf = allf[cnt[inv.ravel()] == 1]
    att = np.array([faces.classify(fm.V[f]) for f in bf], np.int32)
    return np.ascontiguousarray(bf, np.int32), att


# ============================================================================================== evaluators
class Ev:
    """This rank's part of a discrete solution in the check module's protocol (ids, geometry, velocity, pressure)."""

    def __init__(self, sol):
        self.sol = sol
        self.ids = np.asarray(sol.element_ids(), dtype=np.int64)
        self._c = {}

    def _get(self, kind, ref):
        ref = np.ascontiguousarray(ref, float)
        key = (kind, ref.shape, ref.tobytes())
        if key not in self._c:
            self._c[key] = getattr(self.sol, kind)(ref)
        return self._c[key]

    def geometry(self, ref):
        return self._get("geometry", ref)

    def velocity(self, ref):
        return self._get("velocity", ref)

    def pressure(self, ref):
        return self._get("pressure", ref)

    def clear(self):
        self._c = {}


class DataEv:
    """A recorded solution (R1: the one-rank reference at the check module's quadrature points), restricted to ids."""

    def __init__(self, data, ids, ref):
        self.ids = np.asarray(ids, np.int64)
        self._ref = np.asarray(ref, float)
        self._d = data

    def _chk(self, ref):
        if not np.array_equal(np.asarray(ref, float), self._ref):
            raise RuntimeError("the recorded reference solution exists at the fixed quadrature points only")

    def geometry(self, ref):
        self._chk(ref)
        return self._d["x"][self.ids], self._d["detJ"][self.ids]

    def velocity(self, ref):
        self._chk(ref)
        return self._d["u"][self.ids], self._d["G"][self.ids]

    def pressure(self, ref):
        self._chk(ref)
        return self._d["p"][self.ids]


# ============================================================================================== MPI launch
_TASKS = [0]


def _launch(argv, ranks, workdir):
    import fluids_mpi                      # the multi-rank launcher (benchmarks/fluids/cases)
    return fluids_mpi.launch(argv, ranks, workdir)


def run_mpi(task: dict, ranks: int, workdir) -> list:
    wd = Path(workdir).resolve() / "f1c_mpi"
    wd.mkdir(parents=True, exist_ok=True)
    _TASKS[0] += 1
    tp = wd / f"task{_TASKS[0]:04d}.json"
    task = dict(task, sys_path=[str(p) for p in sys.path if p], ranks=ranks)
    tp.write_text(json.dumps(task, default=plain))
    err = None
    try:
        _launch([sys.executable, "-I", "-S", "-B", str(WORKER), str(tp)], ranks, workdir)
    except RuntimeError as e:
        err = e
    outs = []
    for r in range(ranks):
        f = Path(f"{tp}.rank{r}.json")
        if f.is_file():
            outs.append(json.loads(f.read_text()))
            f.unlink()
        else:
            outs.append(None)
    bad = [o for o in outs if o is not None and not o.get("ok")]
    if bad:
        raise RuntimeError(f"MPI worker failed: {bad[0]['error']}\n{bad[0].get('traceback', '')[-2500:]}")
    if err is not None or any(o is None for o in outs):
        raise RuntimeError(f"MPI worker: {err or 'a rank wrote no result'}")
    tp.unlink()
    return outs


# ============================================================================================== worker side
def _fluids():
    from axionfem import fluids
    return fluids


def _allsum(vals):
    return _fluids()._core._fluids.allreduce_sum(np.ascontiguousarray(vals, float))


def _field(task, key):
    src = task["fields"][key]
    if src == "duct":
        d = task["duct"]
        return C.DuctField(float(d["W"]), float(d["H"]), float(d["L"]), float(d["G"]), float(d["mu"]))
    return C.load_field(src)


def _xyz(fn):
    """A function of points P[..., 3] as a solver callable f(X, Y, Z)."""
    return lambda X, Y, Z: fn(np.stack([X, Y, Z], -1))


def _const_vec(v):
    v = np.asarray(v, float)
    return lambda P: np.broadcast_to(v, np.shape(P)[:-1] + (3,)).copy()


def _bc_data(kind, data, fld, form, faces, face):
    """The fixed data of one boundary condition as a function of points."""
    if kind in ("do_nothing", "slip", "symmetry"):
        return None
    if data == "u":
        return fld.u
    if data == "zero":
        return _const_vec((0.0, 0.0, 0.0))
    if isinstance(data, list):
        return _const_vec(data)
    if data == "traction":                       # sigma_form n of the field, n the outward normal of a planar face
        n = faces.normal(face)
        return lambda P: fld.traction(form, P, n)
    if data == "radial_traction":                # sigma_form x/|x| on the sphere r = 4 (outward)
        return lambda P: np.einsum("...ij,...j->...i", fld.sigma(form, P),
                                   np.asarray(P, float) / np.linalg.norm(P, axis=-1, keepdims=True))
    raise ValueError(f"unknown boundary data {data!r}")


def _pressure_shift(fm, ev, fld):
    """Two-pass centering: mean(p_h) - mean(p) over all ranks."""
    if len(ev.ids):
        q, xq, W = C._qw(fm, ev)
        parts = [float((W * ev.pressure(q)).sum()), float(W.sum()), float((W * fld.p(xq)).sum())]
    else:
        parts = [0.0, 0.0, 0.0]
    s = _allsum(parts)
    return (s[0] - s[2]) / s[1]


def _pmean(fm, ev):
    d = C.pressure_mean_sums(fm, ev)
    s = _allsum([d["sum:p"], d["sum:vol"]])
    return s[0] / s[1]


def _verify(mesh, fm, task):
    """Rank 0: the solver's mesh against the fixed construction and the API's Taylor-Hood counts."""
    sm = C.FrozenMesh(fm.kind, mesh.vertices(), np.asarray(mesh.elements(), np.int64),
                      X=(np.asarray(mesh.geometry_nodes()) if fm.X is not None else None))
    coarser = build_mesh_nocache(task["coarser"]) if task.get("coarser") else None
    v = C.verify_mesh(sm, fm, coarser)
    fails = [k for k in ("counts_match", "hash_match", "nested") if k in v and not v[k]]
    if fm.kind == "tet" and (v.get("tets_without_interior_vertex") or v.get("tets_violating_weak_boffi")):
        fails.append("boffi")
    api = dict(mesh.taylor_hood_check())
    chk = C.taylor_hood_condition(fm)
    dis = [k for k in chk if int(api.get(k, -1)) != int(chk[k])]
    return {"verify": {k: v[k] for k in v}, "fails": fails, "api_th": {k: api[k] for k in api}, "check_th": chk,
            "th_disagree": dis}


def build_mesh_nocache(spec):
    saved = dict(_MESH_CACHE)
    try:
        return build_mesh(spec)
    finally:
        _MESH_CACHE.clear()
        _MESH_CACHE.update(saved)


def _shared_node_mismatch(fm, faces, preds_data):
    """Shared-node rule: at a geometry node on two or more Dirichlet faces, the per-face data agree."""
    N = np.unique(np.round(fm.geometry_nodes().reshape(-1, 3), 13), axis=0)
    on = np.stack([faces.on(nm, N) for nm, _ in preds_data], -1)
    sel = on.sum(-1) >= 2
    if not sel.any():
        return 0.0
    pts = N[sel]
    cal = [[(lambda P, g=g: g(P)) for (nm, g), o in zip(preds_data, row) if o] for row in on[sel]]
    return C.dirichlet_shared_node_mismatch(pts, cal)


def worker(task: dict) -> dict:
    """One MPI rank: build the level's fixed mesh, solve every job, return this rank's partial sums."""
    fluids = _fluids()
    rank = int(fluids._core._fluids.world_rank())
    spec = task["mesh"]
    fm = build_mesh(spec)
    faces = Faces(spec)
    bf, ba = boundary_faces(fm, faces)
    nodes = None if fm.X is None else np.ascontiguousarray(fm.X, float)
    mesh = fluids.Mesh3D.from_arrays(np.ascontiguousarray(fm.V, float), np.ascontiguousarray(fm.E, np.int32), bf, ba,
                                     nodes=nodes)
    out = {"rank": rank, "jobs": {}, "n_elements": int(fm.n_elements)}
    if rank == 0 and task.get("verify"):
        out["verify"] = _verify(mesh, fm, task)
    if task.get("special") == "inf_sup":
        out["inf_sup"] = _inf_sup_worker(mesh, fm, task)
    fields = {}
    keep = {}
    for job in task.get("jobs", []):
        fk = job["field"]
        if fk not in fields:
            fields[fk] = _field(task, fk)
        fld = fields[fk]
        form = job["form"]
        bcs, preds_data = [], []
        for b in job["bcs"]:
            g = _bc_data(b["kind"], b.get("data"), fld, form, faces, b["face"])
            e = {"attribute": faces.attr(b["face"]), "kind": b["kind"]}
            if g is not None:
                e["value"] = _xyz(g)
            if b["kind"] == "dirichlet":
                preds_data.append((b["face"], g))
            bcs.append(e)
        mu = job.get("mu", "field")
        mu_c = _xyz(fld.mu) if mu == "field" else float(mu)
        force = _xyz(lambda P, f=fld, fo=form: f.f(fo, P)) if job.get("force") else None
        t0 = time.time()
        try:
            sol = fluids.solve_stokes_3d(mesh, mu_c, force, bcs, form=FORM[form],
                                         grad_div=float(job.get("gamma", 0.0)),
                                         min_refine_steps=int(job.get("min_refine_steps", 0)),
                                         solver=job.get("solver", "direct"), rtol=float(job.get("rtol", 1e-12)))
        except fluids.FluidsError as e:     # collective (a refused or failed factorization): a failed run, reported
            res = {"ids": [], "seconds": time.time() - t0, "error": str(e)[:2000]}
            if rank == 0:
                res["diag"] = _failed_diag(str(e))
            out["jobs"][job["tag"]] = res
            continue
        sec = time.time() - t0
        d = dict(sol.diagnostics)
        res = {"ids": np.asarray(sol.element_ids()).tolist(), "seconds": sec}
        if rank == 0:
            res["diag"] = d
        if not d["converged"]:
            out["jobs"][job["tag"]] = res
            continue
        ev = Ev(sol)
        ev_ = job.get("eval", {})
        if ev_.get("norms"):
            shift = _pressure_shift(fm, ev, fld) if ev_.get("two_pass") else 0.0
            res["norms"] = C.norm_sums(fm, ev, fld, p_shift=shift)
        if ev_.get("geom"):
            res["geom"] = C.geometry_defect(fm, ev)
        for i, bn in enumerate(ev_.get("bnodes", [])):
            g = fld.u if bn.get("data", "u") == "u" else _const_vec((0.0, 0.0, 0.0))
            res[f"bnodes{i}"] = C.boundary_node_sums(
                fm, ev, dirichlet_preds=[faces.pred(nm) for nm in bn.get("dirichlet", [])], g=g,
                slip=[(faces.pred(nm), faces.normal(nm)) for nm in bn.get("slip", [])])
        for fr in ev_.get("drag", []):
            res[f"drag_{fr}"] = C.drag_sums(fm, ev, mu=float(ev_["drag_mu"]), form_residual=fr)
        if ev_.get("reaction"):
            r = sol.reaction([faces.attr(nm) for nm in ev_["reaction"]], "stress")
            res["reaction"] = list(r["force"])
        if ev_.get("energy"):
            res["energy"] = C.energy_sums(ev, fm, form, mu=float(ev_["energy_mu"]))
        if ev_.get("flux") and rank == 0:
            dfl = [b["face"] for b in job["bcs"] if b["kind"] == "dirichlet"]
            bfaces = [tuple(int(v) for v in f) for f, a in zip(bf, ba)
                      if a in {faces.attr(nm) for nm in dfl}]
            res["flux_check"] = list(C.p2_face_flux_defect(fm, fld.u, bfaces))
        if ev_.get("inlet_flux") and rank == 0:
            res["inlet_flux"] = C.duct_inlet_flux(fm, float(task["duct"]["W"]), float(task["duct"]["H"]),
                                                  float(task["duct"]["G"]), float(task["duct"]["mu"]))
        if ev_.get("shared_nodes") and rank == 0:
            res["shared_node_mismatch"] = _shared_node_mismatch(fm, faces, preds_data)
        if ev_.get("dump"):                    # R1: the reference (one rank) at the fixed quadrature points
            q, _ = C.rule(fm.kind)
            x, detJ = ev.geometry(q)
            u, G = ev.velocity(q)
            np.savez(ev_["dump"], ids=ev.ids, x=x, detJ=detJ, u=u, G=G, p=ev.pressure(q), pmean=_pmean(fm, ev))
        if ev_.get("diff_ref"):                # R1: this solve against the recorded one-rank reference
            q, _ = C.rule(fm.kind)
            data = np.load(ev_["diff_ref"], mmap_mode="r")
            order = np.argsort(np.asarray(data["ids"]))
            ref = {k: np.asarray(data[k])[order] for k in ("x", "detJ", "u", "G", "p")}
            evb = DataEv(ref, ev.ids, q)
            shift = _pmean(fm, ev) - float(data["pmean"])
            res["diff"] = C.difference_sums(fm, ev, evb, shift=shift)
        if job.get("diff_with"):               # identity: this solve against an earlier one of the same task
            ea = keep[job["diff_with"]]
            shift = _pmean(fm, ea) - _pmean(fm, ev)
            res["diff"] = C.difference_sums(fm, ea, ev, shift=shift)
        if job.get("keep"):
            keep[job["tag"]] = ev
        else:
            ev.clear()
        out["jobs"][job["tag"]] = res
    return out


def _failed_diag(msg):
    """The diagnostics of a solve the library refused or could not factorize (reported as a failed run)."""
    d = {k: None for k in ("iterations", "solver", "nullspace", "kernel_defect", "flux_defect", "flux_defect_abs",
                           "residual_momentum", "residual_continuity", "lu_residual_before_refinement",
                           "lu_refine_steps", "lu_krylov_iterations", "lu_krylov_fallback", "lu_matching", "lu_matching_retry", "lu_backend", "slip_nodes",
                           "slip_corner_nodes", "grad_div", "ranks",
                           "threads", "lu_threads")}
    d.update(converged=False, relative_residual=1e300, nullspace_detected=False,
             comm_size=0, shared_dofs=0, dofs_velocity=0, dofs_pressure=0, dofs_field=0, error=msg[:2000])
    return d


def _csr_dense(d):
    n, m = (int(v) for v in d["shape"])
    A = np.zeros((n, m))
    ip, ix, dv = np.asarray(d["indptr"]), np.asarray(d["indices"]), np.asarray(d["data"])
    for i in range(n):
        A[i, ix[ip[i]:ip[i + 1]]] += dv[ip[i]:ip[i + 1]]
    return A


def _inf_sup_worker(mesh, fm, task):
    """N1 (one rank): the dense inf-sup of the assembled blocks (A: vector H1 seminorm Gram = the Laplacian viscous
    block with nu = 1 on the free DOFs; B; M_p) for Taylor-Hood and, when asked, P1/P1."""
    fluids = _fluids()
    faces = Faces(task["mesh"])
    bcs = [{"attribute": faces.attr(nm), "kind": "dirichlet"} for nm in faces.names]
    out = {}
    for name, po, vo in (("th", -1, 2), ("p1p1", 1, 1)):
        if name not in task["pairs"]:
            continue
        s = fluids.assemble_stokes_3d(mesh, 1.0, bcs, form="laplacian", order=vo, pressure_order=po,
                                      allow_unstable_pair=(name == "p1p1"))
        blk = s.blocks()
        ess = np.asarray(blk["essential_velocity_dofs"], int)
        A = _csr_dense(blk["A"])
        B = _csr_dense(blk["B"])
        Mp = _csr_dense(blk["Mp"])
        free = np.setdiff1d(np.arange(A.shape[0]), ess)
        r = C.inf_sup(A[np.ix_(free, free)], B[:, free], Mp, zero_rel=float(task["zero_rel"]))
        out[name] = r
    return out


# ============================================================================================== parent side
def _combine_job(outs, tag):
    parts = [o["jobs"][tag] for o in outs]
    r0 = parts[0]
    cj = {"ids_per_rank": [p["ids"] for p in parts], "diag": r0.get("diag"), "seconds": max(p["seconds"] for p in parts)}
    for key in ("norms", "geom", "energy"):
        if key in r0:
            cj[key] = C.combine([p[key] for p in parts])
    for k in r0:
        if k.startswith("bnodes"):
            cj[k] = C.combine([p[k] for p in parts])
        if k.startswith("drag_"):
            cj[k] = sum(p[k]["sum:drag"] for p in parts)
    if "diff" in r0:
        cj["diff"] = C.finish_difference([p["diff"] for p in parts])
    for k in ("reaction", "flux_check", "inlet_flux", "shared_node_mismatch"):
        if k in r0:
            cj[k] = r0[k]
    return cj


def _digest(ids_per_rank):
    out = []
    for ids in ids_per_rank:
        a = np.asarray(ids, np.int64)
        out.append({"n": int(len(a)), "min": int(a.min()) if len(a) else None, "max": int(a.max()) if len(a) else None,
                    "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]})
    return out


def _union_digest(ids_per_rank, n_elements):
    """Re-checkable partition proof: the sha256 prefix of the SORTED union of every
    rank's ids next to that of range(n_elements). A reader re-derives `partition_failures` from the result: the two
    must be equal and n == n_elements (equal: no duplicated, missing or out-of-range id, also when one id is
    duplicated and another missing inside a rank, which leaves the per-rank n/min/max unchanged); per-rank `n` gives
    the number of parts and the empty ones."""
    allids = np.sort(np.concatenate([np.asarray(a, np.int64).ravel() for a in ids_per_rank] or [np.zeros(0, np.int64)]))
    want = np.arange(int(n_elements), dtype=np.int64)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]  # noqa: E731
    return {"n": int(len(allids)), "sha256": h(allids), "expected_n": int(n_elements), "expected_sha256": h(want),
            "per_rank_n": [int(len(a)) for a in ids_per_rank]}


class Collector:
    """Run entries, series, quantities and the rank-proof inputs of one case."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.runs, self.series, self.q, self.notes = [], {}, {}, []
        self.proof = []

    def max(self, k, v):
        self.q[k] = max(self.q.get(k, -math.inf), float(v))

    def min(self, k, v):
        self.q[k] = min(self.q.get(k, math.inf), float(v))

    def run(self, tag, cj, level_info, family=None, refinement_level=None, extra=None, proof=True):
        d = cj["diag"]
        e = {"level": tag,
             "mesh": {"h_nominal": level_info.get("h"), "h_max": level_info.get("h_max"),
                      "n_elements": int(level_info["n_elements"]), "dofs_field": int(d["dofs_velocity"]),
                      "dofs_pressure": int(d["dofs_pressure"]), "dofs_total": int(d["dofs_field"]), "files": [],
                      "hash": level_info.get("hash")},
             "solver": {k: d.get(k) for k in ("converged", "relative_residual", "iterations", "solver", "nullspace",
                                         "nullspace_detected", "kernel_defect", "flux_defect", "flux_defect_abs",
                                         "residual_momentum", "residual_continuity", "lu_residual_before_refinement",
                                         "lu_refine_steps", "lu_backend", "slip_nodes", "slip_corner_nodes",
                                         "grad_div", "ranks", "threads", "lu_threads",
                                         "essential_velocity_dofs", "dofs_velocity")},
             "seconds": cj["seconds"], "comm_size": int(d["comm_size"]), "shared_dofs": int(d["shared_dofs"]),
             "evaluator_ids": _digest(cj["ids_per_rank"]),
             "evaluator_ids_union": _union_digest(cj["ids_per_rank"], level_info["n_elements"])}
        e["solver"]["kind"] = d["solver"]
        # a solve finished by the GMRES fallback says so (the label is then "direct+gmres:<backend>"). The
        # iteration count is stored as a float: it may differ between builds.
        e["solver"]["lu_krylov_fallback"] = bool(d.get("lu_krylov_fallback"))
        e["solver"]["lu_matching"] = d.get("lu_matching")
        e["solver"]["lu_matching_retry"] = d.get("lu_matching_retry")
        # the retry's reason and the matching-free attempt's plain residual and GMRES
        # iterations are kept (the retry overwrites the solver's statistics). Floats on purpose, as lu_krylov_iterations.
        e["solver"]["lu_matching_retry_reason"] = d.get("lu_matching_retry_reason") or ""
        e["solver"]["lu_first_attempt_residual_before"] = float(d.get("lu_first_attempt_residual_before", -1.0))
        e["solver"]["lu_first_attempt_krylov_iterations"] = float(d.get("lu_first_attempt_krylov_iterations") or 0)
        e["solver"]["lu_krylov_iterations"] = float(d.get("lu_krylov_iterations") or 0)
        if family is not None:
            e["mesh"]["family"] = family
            e["mesh"]["refinement_level"] = refinement_level
        if extra:
            e.update(extra)
        self.runs.append(e)
        if proof:
            self.proof.append({"level": tag, "comm_size": int(d["comm_size"]), "shared_dofs": int(d["shared_dofs"]),
                               "ids_per_rank": cj["ids_per_rank"], "n_elements": int(level_info["n_elements"])})
        return e

    def rank_proof(self):
        """THE REFERENCE RULE (tests/test_fluids_stokes_3d_check.py::rank_proof_quantities), on the full id lists."""
        runs, n = self.proof, self.ranks
        mismatch = sum(int(r["comm_size"]) != n for r in runs)
        failed = set()
        for r in runs:
            rep = C.partition_report(r["ids_per_rank"], r["n_elements"], n)
            if (rep["parts"] != n or rep["duplicate_ids"] or rep["missing_ids"] or rep["out_of_range_ids"]
                    or (rep["empty_parts"] and r["n_elements"] >= n)):
                failed.add(r["level"])
        self.q["comm_size_mismatch_runs"] = mismatch
        self.q["partition_failures"] = len(failed)
        self.q["shared_dofs_min_runs"] = min(int(r["shared_dofs"]) for r in runs) if runs else 0

    def level_checks(self, outs, info):
        """mesh_verify_failures / api_th_check_disagreements (counted per level) and the level info for the runs."""
        v = outs[0].get("verify")
        if v is None:
            return
        self.q["mesh_verify_failures"] = self.q.get("mesh_verify_failures", 0) + int(bool(v["fails"]))
        self.q["api_th_check_disagreements"] = self.q.get("api_th_check_disagreements", 0) + int(bool(v["th_disagree"]))
        info["h_max"] = v["verify"].get("h_max")
        info["hash"] = v["verify"].get("hash")
        if v["fails"] or v["th_disagree"]:
            self.notes.append(f"level {info.get('label')}: verify fails {v['fails']}, TH disagreements "
                              f"{v['th_disagree']} (api {v['api_th']}, check {v['check_th']})")

    def result(self):
        fb = [r["level"] for r in self.runs if r["solver"].get("lu_krylov_fallback")]
        if fb:
            self.notes.append(f"{len(fb)} run(s) were finished by the LU's GMRES fallback (solver label "
                              f"'direct+gmres:<backend>', solver.lu_krylov_iterations): {', '.join(fb)}")
        return {"runs": self.runs, "series": self.series, "quantities": self.q, "notes": " ".join(self.notes)}


def _series(levels, errors, magnitude):
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


def _fseries(levels, values):
    return {"levels": list(levels), "values": [float(v) for v in values]}


def _common_checks(col, cj, dirichlet_key="bnodes0"):
    """Per run: solver nullspace, nodal Dirichlet data, geometry defect."""
    if dirichlet_key in cj:
        col.max("dirichlet_nodal_data_defect_max", cj[dirichlet_key]["dirichlet_nodal_defect"])
    if "geom" in cj:
        col.max("geometry_defect_max", cj["geom"]["geometry_x"])
    if "shared_node_mismatch" in cj:
        col.max("dirichlet_shared_node_mismatch_max", cj["shared_node_mismatch"])


def _task(spec, coarser, fields, jobs, verify=True, **kw):
    t = {"mesh": spec, "coarser": coarser, "verify": verify, "fields": fields, "jobs": jobs}
    t.update(kw)
    return t


def _ev(norms=True, two_pass=False, geom=True, bnodes=None, **kw):
    e = {"norms": norms, "two_pass": two_pass, "geom": geom, "bnodes": bnodes or []}
    e.update(kw)
    return e


def _dbc(names, data="u"):
    return [{"face": nm, "kind": "dirichlet", "data": data} for nm in names]


ALL6 = ["x0", "x1", "y0", "y1", "z0", "z1"]


def _solved(cj):
    return cj["diag"] is not None and bool(cj["diag"]["converged"])


class NotConverged(Exception):
    pass


def _need(col, cj, tag):
    if not _solved(cj):
        raise NotConverged(tag)


def case(fn):
    """A solve that fails its own gate ends the case with the runs so far (the solver gate then fails)."""
    import functools

    @functools.wraps(fn)
    def wrapped(params, workdir, quick):
        col = Collector(RANKS)
        try:
            fn(params, workdir, quick, col)
        except NotConverged as e:
            col.notes.append(f"stopped: solve {e} did not converge")
        return col.result()
    return wrapped


def _add_run(col, outs, tag, info, family=None, rl=None, **kw):
    cj = _combine_job(outs, tag)
    if cj["diag"] is None:
        raise RuntimeError(f"no diagnostics for {tag}")
    col.run(tag, cj, info, family, rl, **kw)
    col.min("nullspace_detected_min_all", int(cj["diag"]["nullspace_detected"]))
    _need(col, cj, tag)
    return cj


# ============================================================================================== T1
def _cube_specs(P, fam, quick):
    m = P["meshes"]
    if fam == "kuhn":
        return [({"family": "kuhn", "N": int(N)}, h) for N, h in zip(_levels(m["kuhn"]["N"], quick),
                                                                     m["kuhn"]["h_nominal"])]
    if fam == "hex":
        return [({"family": "hex", "N": int(N)}, h) for N, h in zip(_levels(m["hex"]["N"], quick), m["hex"]["h_nominal"])]
    if fam == "dist":
        d = m["kuhn_distorted"]
        return [({"family": "dist", "k": int(k), "N0": 4, "rel": float(d["perturb_rel"]), "seed": int(d["seed"])}, h)
                for k, h in zip(_levels(d["k"], quick), d["h_nominal"])]
    if fam == "rot":
        d = m["kuhn_rotated"]
        return [({"family": "rot", "N": int(N), "R": P["_R"]}, h) for N, h in zip(_levels(d["N"], quick), d["h_nominal"])]
    raise ValueError(fam)


FAMILY_ID = {"kuhn": "kuhn_octant", "dist": "kuhn_distorted", "hex": "hex", "rot": "kuhn_rotated"}


def _level_loop(col, P, workdir, fam, quick, fields, jobs_of_level, after_level, magnitudes=None):
    specs = _cube_specs(P, fam, quick)
    for i, (spec, h) in enumerate(specs):
        outs = run_mpi(_task(spec, specs[i - 1][0] if i else None, fields, jobs_of_level(i, len(specs))),
                       RANKS, workdir)
        info = {"h": float(h), "label": f"{fam}_L{i}"}
        col.level_checks(outs, info)
        info["n_elements"] = int(outs[0]["n_elements"])
        after_level(i, outs, info)


@case
def mms_dirichlet_3d(params, workdir, quick, col):
    """T1: all-Dirichlet cube, variable mu, both forms; kuhn / distorted / hex; identity; solver sensitivity."""
    P = _js(params)
    fields = {"vel": P["mms_data_velocity"], "pre": P["mms_data_pressure"], "id": P["mms_data_identity"]}
    forms = P["forms"]
    rows = {}
    for fam in ("kuhn", "dist", "hex"):
        def jobs(i, n, fam=fam):
            js = []
            for form in forms:
                for fk, pre in (("vel", "u"), ("pre", "p")):
                    js.append({"tag": f"{pre}_{form}_{fam}_L{i}", "field": fk, "form": form, "bcs": _dbc(ALL6),
                               "force": True,
                               "eval": _ev(two_pass=True, bnodes=[{"dirichlet": ALL6}], flux=(fk == "vel"),
                                           shared_nodes=(fk == "vel" and form == forms[0]))})
            if fam == "kuhn" and i == n - 1 and "sym" in forms:
                js.append({"tag": "u_sym_kuhn_tight", "field": "vel", "form": "sym", "bcs": _dbc(ALL6), "force": True,
                           "min_refine_steps": 2, "eval": _ev(two_pass=True, geom=False)})
            return js

        def after(i, outs, info, fam=fam):
            for form in forms:
                for fk, pre in (("vel", "u"), ("pre", "p")):
                    tag = f"{pre}_{form}_{fam}_L{i}"
                    cj = _add_run(col, outs, tag, info, FAMILY_ID[fam], i)
                    _common_checks(col, cj)
                    rows.setdefault((pre, form, fam), []).append((tag, cj["norms"]))
                    if fk == "vel":
                        api = float(cj["diag"]["flux_defect_abs"])
                        tot, ref = cj["flux_check"]
                        col.max("flux_defect_api_vs_check_rel_max", abs(api - abs(tot)) / ref)
            if "u_sym_kuhn_tight" in outs[0]["jobs"]:
                cj = _add_run(col, outs, "u_sym_kuhn_tight", info, "kuhn_octant_tight", i)
                col.q["u_sym_kuhn_uL2_finest_tight"] = cj["norms"]["uL2"]
        _level_loop(col, P, workdir, fam, quick, fields, jobs, after)
    for (pre, form, fam), rr in rows.items():
        lv = [t for t, _ in rr]
        if pre == "u":
            mag = max(n["u_magnitude"] for _, n in rr)
            for k in ("uL2", "uH1", "divL2"):
                col.series[f"u_{form}_{fam}_{k}"] = _series(lv, [n[k] for _, n in rr], mag)
            if form == "sym" and fam == "kuhn":
                col.q["u_sym_kuhn_uL2_finest"] = rr[-1][1]["uL2"]
        else:
            mag = max(n["p_magnitude"] for _, n in rr)
            col.series[f"p_{form}_{fam}_pL2m"] = _series(lv, [n["pL2m"] for _, n in rr], mag)
    # identity: sym (gamma = 0) == lap (gamma = mu), constant mu = 0.37, all Dirichlet, direct path
    idn = P["identity"]
    mu0 = float(idn["mu"])
    for N in idn["N"]:
        if quick and int(N) > 8:
            continue
        spec = {"family": "kuhn", "N": int(N)}
        jobs = [{"tag": f"identity_sym_N{N}", "field": "id", "form": "sym", "bcs": _dbc(ALL6), "force": True,
                 "gamma": 0.0, "keep": True, "eval": _ev(norms=False, bnodes=[{"dirichlet": ALL6}])},
                {"tag": f"identity_lap_gd_N{N}", "field": "id", "form": "lap", "bcs": _dbc(ALL6), "force": True,
                 "gamma": mu0, "diff_with": f"identity_sym_N{N}", "eval": _ev(norms=False, bnodes=[{"dirichlet": ALL6}])}]
        outs = run_mpi(_task(spec, None, fields, jobs, verify=False), RANKS, workdir)
        info = {"h": 1.0 / int(N), "n_elements": 6 * int(N) ** 3}
        _add_run(col, outs, jobs[0]["tag"], info)
        cj = _add_run(col, outs, jobs[1]["tag"], info)
        col.max("identity_sym_vs_lap_graddiv_mu_u_rel_max", cj["diff"]["u_rel"])
        col.max("identity_sym_vs_lap_graddiv_mu_p_rel_max", cj["diff"]["p_rel"])
        _common_checks(col, cj)
    col.q["nullspace_detected"] = col.q.pop("nullspace_detected_min_all")
    col.rank_proof()


# ============================================================================================== T2
@case
def mms_traction_3d(params, workdir, quick, col):
    """T2: Dirichlet on x0, y0, z0; inhomogeneous traction sigma_form n on x1, y1, z1 (kuhn and hex)."""
    P = _js(params)
    fields = {"vel": P["mms_data_velocity"], "pre": P["mms_data_pressure"], "cmu": P["mms_data_constmu"]}
    dirf, trf = P["boundary_faces"]["dirichlet"], P["boundary_faces"]["traction"]
    bcs = _dbc(dirf) + [{"face": nm, "kind": "traction", "data": "traction"} for nm in trf]
    rows, ns = {}, []
    for fam in ("kuhn", "hex"):
        def jobs(i, n, fam=fam):
            return [{"tag": f"{key}_{form}_{fam}_L{i}", "field": fk, "form": form, "bcs": bcs, "force": True,
                     "eval": _ev(bnodes=[{"dirichlet": dirf}], shared_nodes=(fk == "vel" and form == "lap"))}
                    for form in P["forms"] for fk, key in (("vel", "u"), ("pre", "p"), ("cmu", "constmu_u"))]

        def after(i, outs, info, fam=fam):
            for form in P["forms"]:
                for fk, key in (("vel", "u"), ("pre", "p"), ("cmu", "constmu_u")):
                    tag = f"{key}_{form}_{fam}_L{i}"
                    cj = _add_run(col, outs, tag, info, FAMILY_ID[fam], i)
                    _common_checks(col, cj)
                    ns.append(int(cj["diag"]["nullspace_detected"]))
                    rows.setdefault((key, form, fam), []).append((tag, cj["norms"]))
        _level_loop(col, P, workdir, fam, quick, fields, jobs, after)
    for (key, form, fam), rr in rows.items():
        lv = [t for t, _ in rr]
        if key == "u":
            mag = max(n["u_magnitude"] for _, n in rr)
            for k in ("uL2", "uH1"):
                col.series[f"u_{form}_{fam}_{k}"] = _series(lv, [n[k] for _, n in rr], mag)
        elif key == "constmu_u":
            mag = max(n["u_magnitude"] for _, n in rr)
            col.series[f"constmu_u_{form}_{fam}_uL2"] = _series(lv, [n["uL2"] for _, n in rr], mag)
        else:
            mag = max(n["p_magnitude"] for _, n in rr)
            col.series[f"p_{form}_{fam}_pL2"] = _series(lv, [n["pL2"] for _, n in rr], mag)
    col.q.pop("nullspace_detected_min_all", None)
    col.q["nullspace_detected"] = max(ns)
    col.rank_proof()


# ============================================================================================== T3
@case
def mms_do_nothing_3d(params, workdir, quick, col):
    """T3: do-nothing on x1 in each form's own sense (velocity field per form; pressure field in both forms)."""
    P = _js(params)
    fields = {"vel_lap": P["mms_data_velocity_lap"], "vel_sym": P["mms_data_velocity_sym"],
              "pre": P["mms_data_pressure"]}
    dirf = P["boundary_faces"]["dirichlet"]
    bcs = _dbc(dirf) + [{"face": nm, "kind": "do_nothing"} for nm in P["boundary_faces"]["do_nothing"]]
    rows, ns = {}, []

    def jobs(i, n):
        js = []
        for form in P["forms"]:
            for fk, key in ((f"vel_{form}", "u"), ("pre", "p")):
                js.append({"tag": f"{key}_{form}_kuhn_L{i}", "field": fk, "form": form, "bcs": bcs, "force": True,
                           "eval": _ev(bnodes=[{"dirichlet": dirf}], shared_nodes=(key == "u"))})
        return js

    def after(i, outs, info):
        for form in P["forms"]:
            for key in ("u", "p"):
                tag = f"{key}_{form}_kuhn_L{i}"
                cj = _add_run(col, outs, tag, info, "kuhn_octant", i)
                _common_checks(col, cj)
                ns.append(int(cj["diag"]["nullspace_detected"]))
                rows.setdefault((key, form), []).append((tag, cj["norms"]))
    _level_loop(col, P, workdir, "kuhn", quick, fields, jobs, after)
    for (key, form), rr in rows.items():
        lv = [t for t, _ in rr]
        if key == "u":
            mag = max(n["u_magnitude"] for _, n in rr)
            for k in ("uL2", "uH1"):
                col.series[f"u_{form}_kuhn_{k}"] = _series(lv, [n[k] for _, n in rr], mag)
        else:
            mag = max(n["p_magnitude"] for _, n in rr)
            col.series[f"p_{form}_kuhn_pL2"] = _series(lv, [n["pL2"] for _, n in rr], mag)
    col.q.pop("nullspace_detected_min_all", None)
    col.q["nullspace_detected"] = max(ns)
    col.rank_proof()


# ============================================================================================== T4
@case
def mms_slip_3d(params, workdir, quick, col):
    """T4: rotated cube; slip on local z0, symmetry on local x1 meeting along an edge; Dirichlet elsewhere."""
    P = _js(params)
    Fv = C.load_field(P["mms_data_velocity"])
    P["_R"] = np.asarray(Fv.data["frame"]["R_float"], float).tolist()
    fields = {"vel": P["mms_data_velocity"], "pre": P["mms_data_pressure"], "patch": P["slip_patch"]["field"]}
    bfc = P["boundary_faces"]
    dirf, slipf, symf = bfc["dirichlet"], bfc["slip"], bfc["symmetry"]
    bcs = (_dbc(dirf) + [{"face": nm, "kind": "slip"} for nm in slipf] +
           [{"face": nm, "kind": "symmetry"} for nm in symf])
    bn = [{"dirichlet": dirf, "slip": slipf + symf}]
    rows = {}

    def jobs(i, n):
        js = [{"tag": f"{key}_{form}_rot_L{i}", "field": fk, "form": form, "bcs": bcs, "force": True,
               "eval": _ev(two_pass=True, bnodes=bn, shared_nodes=(fk == "vel"))}
              for form in P["forms"] for fk, key in (("vel", "u"), ("pre", "p"))]
        return js

    def after(i, outs, info):
        for form in P["forms"]:
            for fk, key in (("vel", "u"), ("pre", "p")):
                tag = f"{key}_{form}_rot_L{i}"
                cj = _add_run(col, outs, tag, info, "kuhn_rotated", i)
                _common_checks(col, cj)
                if fk == "vel":
                    col.max("slip_un_nodal_max_rel", cj["bnodes0"]["slip_un"] / cj["norms"]["u_magnitude"])
                rows.setdefault((key, form), []).append((tag, cj["norms"]))
    _level_loop(col, P, workdir, "rot", quick, fields, jobs, after)
    for (key, form), rr in rows.items():
        lv = [t for t, _ in rr]
        if key == "u":
            mag = max(n["u_magnitude"] for _, n in rr)
            for k in ("uL2", "uH1"):
                col.series[f"u_{form}_rot_{k}"] = _series(lv, [n[k] for _, n in rr], mag)
        else:
            mag = max(n["p_magnitude"] for _, n in rr)
            col.series[f"p_{form}_rot_pL2m"] = _series(lv, [n["pL2m"] for _, n in rr], mag)
    # non-MMS cell: uniform flow along the slip-symmetry edge (u in P2: round-off), N = 4, both forms
    sp = P["slip_patch"]
    spec = {"family": "rot", "N": int(sp["N"]), "R": P["_R"]}
    jobs_p = [{"tag": f"patch_{form}_N{sp['N']}", "field": "patch", "form": form, "bcs": bcs, "force": True,
               "eval": _ev(two_pass=True, bnodes=bn)} for form in sp["forms"]]
    outs = run_mpi(_task(spec, None, fields, jobs_p, verify=False), RANKS, workdir)
    info = {"h": 1.0 / int(sp["N"]), "n_elements": 6 * int(sp["N"]) ** 3}
    for j in jobs_p:
        cj = _add_run(col, outs, j["tag"], info)
        _common_checks(col, cj)
        col.max("slip_patch_u_rel_max", cj["norms"]["uL2"] / cj["norms"]["uh_L2"])
    col.q["nullspace_detected"] = col.q.pop("nullspace_detected_min_all")
    col.rank_proof()


# ============================================================================================== P1
@case
def duct_poiseuille_3d(params, workdir, quick, col):
    """P1: rectangular duct, series solution; four variants per level; deviations from the fixed calibrated errors,
    flux balance, positive control, the lap_pd flow-rate functional."""
    P = _js(params)
    du = P["duct"]
    G, L = float(du["G"]), float(du["L"])
    fields = {"duct": "duct"}
    walls = ["x0", "x1", "y0", "y1"]
    var = {
        "lap_dn": ("lap", _dbc(walls, "zero") + _dbc(["z0"], "u") + [{"face": "z1", "kind": "do_nothing"}],
                   walls + ["z0"]),
        "lap_pd": ("lap", _dbc(walls, "zero") + [{"face": "z0", "kind": "traction", "data": [0.0, 0.0, G * L]},
                                                 {"face": "z1", "kind": "traction", "data": [0.0, 0.0, 0.0]}], walls),
        "sym_dir": ("sym", _dbc(walls, "zero") + _dbc(["z0", "z1"], "u"), walls + ["z0", "z1"]),
        "sym_dn_control": ("sym", _dbc(walls, "zero") + _dbc(["z0"], "u") + [{"face": "z1", "kind": "do_nothing"}],
                           walls + ["z0"]),
    }
    hs = _levels(P["meshes"]["duct"]["h_nominal"], quick)
    rows = {v: [] for v in var}
    for i, h in enumerate(hs):
        spec = {"family": "duct", "h": float(h)}
        jobs = [{"tag": f"{v}_L{i}", "field": "duct", "form": fo, "bcs": bc, "force": False, "mu": float(du["mu"]),
                 "eval": _ev(geom=False, bnodes=[{"dirichlet": dn}], inlet_flux=(v == "lap_dn"))}
                for v, (fo, bc, dn) in var.items()]
        outs = run_mpi(_task(spec, {"family": "duct", "h": float(hs[i - 1])} if i else None, fields, jobs,
                             duct=du), RANKS, workdir)
        info = {"h": float(h), "label": f"duct_L{i}"}
        col.level_checks(outs, info)
        info["n_elements"] = int(outs[0]["n_elements"])
        inflow = None
        for v in var:
            cj = _add_run(col, outs, f"{v}_L{i}", info, "duct", i)
            _common_checks(col, cj)
            if "inlet_flux" in cj:
                inflow = cj["inlet_flux"]
            rows[v].append((f"{v}_L{i}", cj))
        for v in ("lap_dn", "sym_dir", "sym_dn_control"):
            Qh = rows[v][-1][1]["norms"]["uhz"] / L
            col.max("duct_flux_balance_rel_max", abs(Qh - inflow) / inflow)
    cal = du["calibrated_errors"]
    for v in ("lap_dn", "lap_pd", "sym_dir"):
        for k in ("uL2", "uH1"):
            vals = [cj["norms"][k] for _, cj in rows[v]]
            col.q[f"duct_{v}_{k}_rel_dev_max"] = C.relative_deviation(vals, cal[f"{v}_{k}"][:len(vals)])
    col.q["sym_dn_control_dev_rel"] = min(cj["norms"]["uL2"] / cj["norms"]["uh_L2"] for _, cj in rows["sym_dn_control"])
    for v, key, agg in (("lap_dn", "nullspace_detected_lap_dn", max), ("lap_pd", "nullspace_detected_lap_pd", max),
                        ("sym_dir", "nullspace_detected_sym_dir", min)):
        col.q[key] = agg(int(cj["diag"]["nullspace_detected"]) for _, cj in rows[v])
    Q = [cj["norms"]["uhz"] / L for _, cj in rows["lap_pd"]]
    col.series["duct_lap_pd_Q"] = _fseries([t for t, _ in rows["lap_pd"]], Q)
    col.q["duct_lap_pd_Q_finest"] = Q[-1]
    col.q.pop("nullspace_detected_min_all", None)
    col.q.pop("geometry_defect_max", None)
    col.rank_proof()


# ============================================================================================== S1
@case
def sphere_drag_3d(params, workdir, quick, col):
    """S1: Stokes flow past a sphere in the shell 1 < r < 4 (hex Q2 / tet P2 curved geometry): unbounded field (both
    forms), concentric field (sym), the curved-traction variant S1T (both forms); drag functionals; API reaction."""
    P = _js(params)
    fields = {"unb": P["fields"]["unbounded"], "conc": P["fields"]["concentric"]}
    mu = 37.0 / 100.0
    vol_exact = 4.0 / 3.0 * math.pi * (4.0 ** 3 - 1.0)
    dir_out = _dbc(["body"], "zero") + _dbc(["outer"], "u")
    trac = _dbc(["body"], "zero") + [{"face": "outer", "kind": "traction", "data": "radial_traction"}]
    variants = [("unbounded_sym", "unb", "sym", dir_out, True), ("unbounded_lap", "unb", "lap", dir_out, True),
                ("concentric_sym", "conc", "sym", dir_out, True), ("traction_sym", "unb", "sym", trac, False),
                ("traction_lap", "unb", "lap", trac, False)]
    for fam in ("hex", "tet"):
        mf = P["meshes"][f"sphere_{fam}"]
        ns_, hs = _levels(mf["n"], quick), mf["h_nominal"]
        rows = {}
        for i, n in enumerate(ns_):
            spec = {"family": f"sphere_{fam}", "n": int(n)}
            jobs = []
            for name, fk, form, bc, dirichlet_outer in variants:
                dn = ["body", "outer"] if dirichlet_outer else ["body"]
                ev = _ev(two_pass=dirichlet_outer, bnodes=[{"dirichlet": dn}], drag=["sym", "lap"], drag_mu=mu,
                         reaction=(["body"] if form == "sym" else None))
                jobs.append({"tag": f"{fam}_{name}_L{i}", "field": fk, "form": form, "bcs": bc, "force": False,
                             "mu": mu, "eval": ev})
            outs = run_mpi(_task(spec, {"family": f"sphere_{fam}", "n": int(ns_[i - 1])} if i else None, fields,
                                 jobs), RANKS, workdir)
            info = {"h": float(hs[i]), "label": f"sphere_{fam}_L{i}"}
            col.level_checks(outs, info)
            info["n_elements"] = int(outs[0]["n_elements"])
            for name, fk, form, bc, dirichlet_outer in variants:
                tag = f"{fam}_{name}_L{i}"
                cj = _add_run(col, outs, tag, info, f"sphere_{fam}", i)
                _common_checks(col, cj)
                if dirichlet_outer:
                    col.min("nullspace_detected_dirichlet", int(cj["diag"]["nullspace_detected"]))
                rows.setdefault(name, []).append((tag, cj))
        lv = lambda name: [t for t, _ in rows[name]]  # noqa: E731
        for name in ("unbounded_sym", "unbounded_lap"):
            rr = rows[name]
            mag_u = max(cj["norms"]["u_magnitude"] for _, cj in rr)
            mag_p = max(cj["norms"]["p_magnitude"] for _, cj in rr)
            for k in ("uL2", "uH1"):
                col.series[f"{fam}_{name}_{k}"] = _series(lv(name), [cj["norms"][k] for _, cj in rr], mag_u)
            col.series[f"{fam}_{name}_pL2m"] = _series(lv(name), [cj["norms"]["pL2m"] for _, cj in rr], mag_p)
        for name in ("traction_sym", "traction_lap"):
            rr = rows[name]
            mag_u = max(cj["norms"]["u_magnitude"] for _, cj in rr)
            mag_p = max(cj["norms"]["p_magnitude"] for _, cj in rr)
            for k in ("uL2", "uH1"):
                col.series[f"{fam}_{name}_{k}"] = _series(lv(name), [cj["norms"][k] for _, cj in rr], mag_u)
            col.series[f"{fam}_{name}_pL2"] = _series(lv(name), [cj["norms"]["pL2"] for _, cj in rr], mag_p)
        rr = rows["concentric_sym"]
        col.series[f"{fam}_concentric_sym_uL2"] = _series(lv("concentric_sym"), [cj["norms"]["uL2"] for _, cj in rr],
                                                          max(cj["norms"]["u_magnitude"] for _, cj in rr))
        rr = rows["unbounded_sym"]
        col.series[f"{fam}_geometry_volume_err"] = _series(
            lv("unbounded_sym"), [abs(cj["norms"]["volume"] - vol_exact) for _, cj in rr], vol_exact)
        # drag functionals (check module, from the solution) and the API reaction at the finest level
        dr = {"unbounded_drag": ("unbounded_sym", "drag_sym"), "unbounded_lap_Lres_drag": ("unbounded_lap", "drag_lap"),
              "traction_drag": ("traction_sym", "drag_sym"), "concentric_drag": ("concentric_sym", "drag_sym")}
        for key, (name, dk) in dr.items():
            vals = [cj[dk] for _, cj in rows[name]]
            col.series[f"{fam}_{key}"] = _fseries(lv(name), vals)
            if key != "unbounded_lap_Lres_drag":
                col.q[f"{fam}_{key}_finest"] = vals[-1]
        for name in ("unbounded_sym", "concentric_sym", "traction_sym"):
            _, cj = rows[name][-1]
            api, chk = float(cj["reaction"][2]), float(cj["drag_sym"])
            col.max("api_reaction_vs_check_drag_rel_finest", abs(api - chk) / abs(chk))
            col.notes.append(f"{fam} {name} finest: API reaction {api:.12g}, check drag {chk:.12g}")
    col.q["nullspace_detected"] = col.q.pop("nullspace_detected_dirichlet")
    col.q.pop("nullspace_detected_min_all", None)
    col.rank_proof()


# ============================================================================================== R1
@case
def rank_invariance_3d(params, workdir, quick, col):
    """R1: the identity field (u and p O(1), mu = 0.37), sym form, direct path, on 1, 4 and 16 ranks; compared at the
    fixed quadrature points by element id against the one-rank solve.  Two problems (amendment TD1): "dirichlet"
    (kuhn N = 16, all six faces Dirichlet) and "mixed" (kuhn N = 4, Dirichlet on x0, y0, z0 and the exact traction on
    x1, y1, z1: the only configuration in which an essential DOF can lose its owner's boundary element).
    The global essential-DOF count must not depend on the rank count (`essential_dofs_spread`)."""
    P = _js(params)
    fields = {"id": P["mms_data"]}
    col.ranks = None
    problems = [("dirichlet", int(P["mesh"]["N"]) if not quick else 8, _dbc(ALL6))]
    mixed = P.get("mixed_variant")
    if mixed:
        faces_d, faces_t = mixed["boundary_faces"]["dirichlet"], mixed["boundary_faces"]["traction"]
        problems.append(("mixed", int(mixed["mesh"]["N"]),
                         _dbc(faces_d) + [{"face": nm, "kind": "traction", "data": "traction"} for nm in faces_t]))
    nullspace = None
    for pname, N, bcs in problems:
        spec = {"family": "kuhn", "N": N}
        ref = Path(workdir) / f"f1c_r1_reference_{pname}.npz"
        nel = 6 * N ** 3
        base = {"field": "id", "form": "sym", "bcs": bcs, "force": True, "gamma": 0.0}
        uL2, ess = {}, {}
        for np_ in [int(r) for r in P["ranks"]]:
            ev = _ev(two_pass=True, geom=False)
            if np_ == 1:
                ev["dump"] = str(ref)
            else:
                ev["diff_ref"] = str(ref)
            job = dict(base, tag=f"{pname}_ranks{np_}", eval=ev)
            outs = run_mpi(_task(spec, None, fields, [job], verify=False), np_, workdir)
            cj = _add_run(col, outs, job["tag"], {"h": 1.0 / N, "n_elements": nel}, proof=False)
            d = cj["diag"]
            rep = C.partition_report(cj["ids_per_rank"], nel, np_)
            bad = (rep["parts"] != np_ or rep["duplicate_ids"] or rep["missing_ids"] or rep["out_of_range_ids"] or
                   (rep["empty_parts"] and nel >= np_))
            col.q["partition_failures"] = col.q.get("partition_failures", 0) + int(bool(bad))
            col.q["comm_size_mismatch"] = col.q.get("comm_size_mismatch", 0) + int(int(d["comm_size"]) != np_)
            col.notes.append(f"{pname}, {np_} ranks: partition {rep}, comm_size {d['comm_size']}, shared DOFs "
                             f"{d['shared_dofs']}, essential velocity DOFs {d['essential_velocity_dofs']}")
            ess[np_] = int(d["essential_velocity_dofs"])
            if np_ > 1:
                col.min("shared_dofs_min_multirank", int(d["shared_dofs"]))
                col.max("u_rel_max_vs_1rank", cj["diff"]["u_rel"])
                col.max("p_rel_max_vs_1rank", cj["diff"]["p_rel"])
            uL2[np_] = cj["norms"]["uL2"]
        for np_, v in uL2.items():
            if np_ != 1:
                col.max("uL2_err_rel_change_max", abs(v - uL2[1]) / uL2[1])
        col.max("essential_dofs_spread", max(ess.values()) - min(ess.values()))
        if nullspace is None:                  # the all-Dirichlet problem only: the mixed one is open (no nullspace)
            nullspace = col.q.pop("nullspace_detected_min_all")
        else:
            col.q.pop("nullspace_detected_min_all", None)
        if ref.is_file():
            ref.unlink()
    col.q["nullspace_detected_all_ranks"] = nullspace


# ============================================================================================== X1
@case
def xcode_cavity_3d(params, workdir, quick, col):
    """X1: lid-driven cube (regularized lid, mu = 1, f = 0), kuhn N = 8, 16, both forms: E_kin and dissipation."""
    P = _js(params)
    fields = {"lid": P["lid_data"]}
    others = ["x0", "x1", "y0", "y1", "z0"]
    bcs = _dbc(others, "zero") + _dbc(["z1"], "u")
    for N, h in zip(_levels(P["meshes"]["kuhn"]["N"], quick), P["meshes"]["kuhn"]["h_nominal"]):
        spec = {"family": "kuhn", "N": int(N)}
        jobs = [{"tag": f"{form}_N{N}", "field": "lid", "form": form, "bcs": bcs, "force": False, "mu": 1.0,
                 "eval": _ev(norms=False, geom=False, bnodes=[{"dirichlet": ["z1"]},
                                                               {"dirichlet": others, "data": "zero"}],
                             energy=True, energy_mu=1.0)} for form in P["forms"]]
        outs = run_mpi(_task(spec, None, fields, jobs, verify=False), RANKS, workdir)
        info = {"h": float(h), "n_elements": 6 * int(N) ** 3}
        for j in jobs:
            cj = _add_run(col, outs, j["tag"], info)
            col.max("dirichlet_nodal_data_defect_max", cj["bnodes0"]["dirichlet_nodal_defect"])
            col.max("dirichlet_nodal_data_defect_max", cj["bnodes1"]["dirichlet_nodal_defect"])
            col.q[f"{j['form']}_Ekin_N{N}"] = cj["energy"]["ekin"]
            col.q[f"{j['form']}_Diss_N{N}"] = cj["energy"]["diss"]
    col.q.pop("nullspace_detected_min_all", None)
    col.rank_proof()


# ============================================================================================== N1
@case
def taylor_hood_control_3d(params, workdir, quick, col):
    """N1: dense inf-sup of Taylor-Hood and P1/P1 on the octant, standard-Kuhn and five-tet unit-cube meshes (N = 2,
    4, 8, one rank); the API's three mesh-condition counts against the check module's; one TH solve (solver gate)."""
    P = _js(params)
    Ns = _levels(P["meshes"]["N"], quick)
    zr = float(P["zero_mode_threshold_rel"])
    beta, th_oct_modes, th_five_modes, p1_modes = {}, [], [], []
    agree, flags = True, True
    for fam in ("kuhn", "standard", "five"):
        for N in Ns:
            pairs = ["p1p1"] + (["th"] if fam in ("kuhn", "five") else [])
            t = _task({"family": fam, "N": int(N)}, None, {}, [], verify=True, special="inf_sup", pairs=pairs,
                      zero_rel=zr)
            outs = run_mpi(t, 1, workdir)
            v, r = outs[0]["verify"], outs[0]["inf_sup"]
            api, chk = v["api_th"], v["check_th"]
            agree = agree and not v["th_disagree"]
            if fam in ("standard", "five"):
                flags = flags and api["tets_without_interior_vertex"] > 0 and api["tets_violating_weak_boffi"] > 0
                if fam == "five":
                    flags = flags and api["tets_all_p2_nodes_on_boundary"] > 0
            col.notes.append(f"{fam} N={N}: api {api}, check {chk}, inf-sup {r}")
            p1_modes.append(r["p1p1"]["zero_modes_beyond_constant"])
            if fam == "kuhn":
                th_oct_modes.append(r["th"]["zero_modes_beyond_constant"])
                beta[int(N)] = r["th"]["beta"]
            if fam == "five":
                th_five_modes.append(r["th"]["zero_modes_beyond_constant"])
    col.q["th_octant_zero_modes_beyond_constant_max"] = max(th_oct_modes)
    col.q["th_octant_beta_min"] = min(beta.values())
    if 4 in beta and 8 in beta:
        col.q["th_octant_beta_ratio_8_over_4"] = beta[8] / beta[4]
    col.q["th_five_zero_modes_beyond_constant_min"] = min(th_five_modes)
    col.q["api_flags_violations"] = int(flags)
    col.q["api_counts_agree_all_meshes"] = int(agree)
    col.q["p1p1_zero_modes_beyond_constant_min"] = min(p1_modes)
    # the solver gate: one direct Taylor-Hood solve of the T1 velocity field on the octant N = 4 mesh (one rank)
    fields = {"vel": P.get("mms_data_velocity") or {"path": "tools/fluids_mms/out/f1c_cube_u.json",
                                                    "sha256": "9c9d6db55fafac937098e56bb591e9a9dc7f79de95df98cdce51d49431c448d0"}}
    job = {"tag": "th_solve_kuhn_N4", "field": "vel", "form": "lap", "bcs": _dbc(ALL6), "force": True,
           "eval": _ev(norms=True, geom=False)}
    outs = run_mpi(_task({"family": "kuhn", "N": 4}, None, fields, [job], verify=False), 1, workdir)
    _add_run(col, outs, job["tag"], {"h": 0.25, "n_elements": 384}, proof=False)
    col.q.pop("nullspace_detected_min_all", None)
