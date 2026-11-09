"""F3a transient Navier-Stokes: the case runner (set-up, time integration, Evaluators).

Every norm, identity, effectivity and step statistic comes from the check module f3a_transient_check (FC;
it re-exports the fixed F1a mesh builders and nodal Dirichlet check).  This module builds the fixed meshes and time
grids with FC's builders, integrates through the public API axionfem.fluids.TransientNavierStokes (design) and wraps the stored time levels in Evaluators.

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two levels of every study only.

Conventions: axisymmetric: the API's component order is (u_r, u_z, u_theta);
the check module's is (u_r, u_theta, u_z): PERM (an involution) converts both ways.  Boundary attributes of every
rectangle: 1 bottom, 2 right, 3 top, 4 left (the axis r = 0 when axisymmetric).  Failed solves are results: a run
that misses its gate is recorded (converged False) and the quantities that need its fields are still computed from the
last state reached, so the acceptance check judges the solver gate.
"""
from __future__ import annotations

import math
import time

import numpy as np

import f3a_transient_check as FC
from axionfem import fluids
from f2a_ns import PERM, Acc, _jsonable, core_identity, make_mesh, rect_classify

FORM = {"lap": "laplacian", "sym": "stress"}
UNIT = rect_classify(0.0, 1.0, 0.0, 1.0)


# ============================================================================================== data
def check_fn(fn, axi):
    """f(X, Y, t) in the check module's order as the API wants it: the axis rule (u_r = u_theta = 0 at r = 0, the limit
    of the exact data) and the component permutation in axisymmetry."""
    def g(X, Y, t):
        X = np.asarray(X, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a = np.array(fn(X, np.asarray(Y, float), t), dtype=float, copy=True)
        if axi:
            on = np.abs(X) < 1e-12
            a[on, 0] = 0.0
            a[on, 1] = 0.0
            a = a[..., PERM]
        return np.ascontiguousarray(a)
    return g


def check_g(fn, axi, t):
    """The fixed data at time t as g(X, Y) in the check order (for the nodal Dirichlet check)."""
    def g(X, Y):
        X = np.asarray(X, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a = np.array(fn(X, np.asarray(Y, float), t), dtype=float, copy=True)
        if axi:
            on = np.abs(X) < 1e-12
            a[on, 0] = 0.0
            a[on, 1] = 0.0
        return a
    return g


def bc(attr, kind, value=None):
    d = {"attribute": attr, "kind": kind}
    if value is not None:
        d["value"] = value
    return d


class Evaluator:
    """A stored time level in the check module's protocol (one MPI rank)."""

    def __init__(self, snap, axi):
        self.snap, self.axi = snap, axi
        self.n_elements = int(snap.n_elements)
        self._el = np.arange(self.n_elements, dtype=np.int32)

    def geometry(self, ref_pts):
        return self.snap.geometry(self._el, np.ascontiguousarray(ref_pts, dtype=float))

    def velocity(self, ref_pts):
        u, g = self.snap.velocity(self._el, np.ascontiguousarray(ref_pts, dtype=float))
        u, g = np.asarray(u), np.asarray(g)
        if self.axi:
            u, g = u[..., PERM], g[..., PERM, :]
        return u, g

    def pressure(self, ref_pts):
        return np.asarray(self.snap.pressure(self._el, np.ascontiguousarray(ref_pts, dtype=float)))


# ============================================================================================== one case's state
class Rec:
    """Runs, series, quantities and notes of one case."""

    def __init__(self, params, quick):
        self.P, self.quick = params, quick
        self.ident = core_identity()
        self.runs, self.ser, self.acc, self.notes = [], {}, Acc(), []
        self.failed = []
        self.t0 = time.time()

    def levels(self, seq, n=2):
        seq = list(seq)
        return seq[:n] if self.quick else seq

    def entry(self, label, ts, form, axi, *, dt=None, dt_max_ratio=None, mesh=None, seconds=0.0, options=None,
              stats=None):
        d = ts.diagnostics
        st = stats if stats is not None else ts.stats
        ok = bool(st["all_converged"])
        if not ok:
            self.failed.append(label)
        e = {"level": label,
             "mesh": dict(mesh or {}),
             "solver": {"converged": ok, "relative_residual": float(st["max_relative_residual"]),
                        "iterations": int(st["newton_iterations"]), "solves": int(st["solves"]),
                        "failure": st.get("failure") or None, "strategy": d.get("strategy"),
                        "convection": d.get("convection"), "viscous_form": form,
                        "geometry": "axisymmetric" if axi else "planar", "kind": d.get("solver"),
                        "lu_backend": d.get("lu_backend"), "lu_fallbacks": int(st.get("lu_fallbacks", 0)),
                        "lu_fallback_backend": st.get("lu_fallback_backend") or None, "nullspace": d.get("nullspace"),
                        "nullspace_detected": bool(d.get("nullspace_detected")), "flux_defect": d.get("flux_defect"),
                        "quad_orders": _jsonable(d.get("quad_orders")), "ranks": d.get("ranks"),
                        "threads": d.get("threads"), "lu_threads": d.get("lu_threads"),
                        "options": {k: _jsonable(v) for k, v in (options or {}).items()}},
             "seconds": float(seconds)}
        if dt is not None:
            e["dt"] = float(dt)
        if dt_max_ratio is not None:
            e["dt_max_ratio"] = float(dt_max_ratio)
        self.runs.append(e)
        return e

    def add(self, name, label, err, mag):
        s = self.ser.setdefault(name, {"levels": [], "error": [], "solution_magnitude": float(mag)})
        s["levels"].append(label)
        s["error"].append(float(err))
        s["solution_magnitude"] = float(mag)

    def result(self):
        me = self.ident
        fb = [(r["level"], r["solver"]["lu_fallbacks"], r["solver"]["lu_fallback_backend"]) for r in self.runs
              if r["solver"].get("lu_fallbacks")]
        if fb:
            self.notes.append(f"SOLVER FALLBACK: {sum(x[1] for x in fb)} direct solve(s) in {len(fb)} run(s) were repeated "
                              f"after the first LU returned a residual above 1e-6 (last backend {fb[-1][2]}): " +
                              ", ".join(f"{x[0]}:{x[1]}" for x in fb[:8]) + (" ..." if len(fb) > 8 else ""))
        if self.failed:
            self.notes.append(f"{len(self.failed)} integration(s) missed the gate (recorded): "
                              + ", ".join(self.failed[:12]) + (" ..." if len(self.failed) > 12 else ""))
        self.notes.append(f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]})")
        return {"runs": self.runs, "series": self.ser, "quantities": self.acc.q, "notes": " ".join(self.notes)}


def verify(mesh, V, T, V_coarser=None):
    v = FC.verify_mesh(mesh.vertices(), mesh.elements(), V, T, V_coarser=V_coarser)
    bad = [k for k in ("counts_match", "hash_match", "nested") if k in v and not v[k]]
    if bad:
        raise RuntimeError(f"the solver mesh does not match the fixed construction: {bad} ({v})")
    return v


def mesh_entry(V, T, h, family=None, level=None):
    m = {"h_nominal": float(h), "n_elements": int(len(T)), "hash": FC.mesh_hash(V, T),
         "all_boundary_triangles": int(FC.all_boundary_triangles(T))}
    if family is not None:
        m["family"] = family
        m["refinement_level"] = int(level)
    return m


def new_solver(mesh, fld, form, scheme, order, force, bcs, newton_rtol=1e-12, **kw):
    axi = fld.axi
    return fluids.TransientNavierStokes(
        mesh, fld.mu, fld.rho, force, bcs, form=FORM[form], geometry="axisymmetric" if axi else "planar",
        scheme=scheme, max_order=order, newton_rtol=newton_rtol, vectorized=True, lu_threads=1, **kw)


def u0_fn(fld, axi, t=0.0):
    f = check_fn(fld.u, axi)
    return lambda X, Y: f(X, Y, t)


def dirichlet_defect(R, snap, attrs, fld, axi, t, key="dirichlet_nodal_data_defect_max", data=None):
    P, U = snap.boundary_nodes(list(attrs))
    U = np.asarray(U, float)
    if axi:
        U = U[:, PERM]
    if len(P):
        R.acc.max(key, FC.dirichlet_nodal_defect(np.asarray(P, float), U, check_g(data or fld.u, axi, t)))


def standard_bcs(fld, axi):
    """Dirichlet data g(x, y, t) on every side (the axis on the left when axisymmetric)."""
    g = check_fn(fld.u, axi)
    bcs = [bc(a, "dirichlet", g) for a in (1, 2, 3)]
    bcs.append(bc(4, "axis") if axi else bc(4, "dirichlet", g))
    return bcs, ((1, 2, 3) if axi else (1, 2, 3, 4))


def integrate(ts, grid, step_cb=None):
    """Advance over the time grid; returns (reached, times). step_cb(n, t_n, ts) after every accepted step."""
    times = [float(grid[0])]
    for n in range(1, len(grid)):
        r = ts.advance(float(grid[n] - grid[n - 1]))
        if not r["converged"]:
            return False, times
        times.append(float(r["t"]))
        if step_cb:
            step_cb(n, times[-1], ts)
    return True, times


def ratio_of(times):
    """Largest ratio of successive steps of the steps actually taken (1.0 if fewer than two: a run whose first step
    missed its gate is a result, not a crash)."""
    return FC.step_stats(times)["dt_max_ratio"] if len(times) >= 2 else 1.0


# ============================================================================================== T1
def temporal_mms(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    T_end, Ns = float(P["T"]), R.levels(P["N_steps"])
    V, Tr = FC.structured_mesh(P["mesh"]["N"])
    for geom in ("planar", "axisymmetric"):
        axi = geom == "axisymmetric"
        fld = FC.load_field(P["mms_data"][geom])
        mesh, seg, att = make_mesh(V, Tr, UNIT)
        verify(mesh, V, Tr)
        g = "axi" if axi else "planar"
        bcs, dattrs = standard_bcs(fld, axi)
        for form in P["forms"]:
            force = check_fn(fld.f(form), axi)
            for var in P["variants"]:
                order, scheme, grid = int(var["order"]), var["scheme"], var["grid"]
                prefix = f"{g}_{form}_bdf{order}_{scheme}_{grid}"
                spec = P["step_function"][grid]
                ts = new_solver(mesh, fld, form, scheme, order, force, bcs, newton_rtol=float(P["newton_rtol"]))
                for N in Ns:
                    t0 = time.time()
                    tg = FC.time_grid(spec, T_end, N)
                    ts.initialize(u0_fn(fld, axi), 0.0, True)
                    ts.reset_stats()
                    emax = {"u": 0.0}
                    key = "uL2w" if axi else "uL2"

                    def cb(n, t, s):
                        e = FC.errors(Evaluator(s.snapshot(), axi), fld, t)
                        emax["u"] = max(emax["u"], e[key])
                    ok, times = integrate(ts, tg, cb)
                    snap = ts.snapshot()
                    eT = FC.errors(Evaluator(snap, axi), fld, times[-1])
                    label = f"{prefix}_N{N}"
                    R.entry(label, ts, form, axi, dt=T_end / N, dt_max_ratio=ratio_of(times),
                            mesh=mesh_entry(V, Tr, 1.0 / P["mesh"]["N"]), seconds=time.time() - t0,
                            options={"order": order, "scheme": scheme, "grid": grid, "N": N,
                                     "newton_rtol": P["newton_rtol"]})
                    if axi:
                        R.add(f"{prefix}_uL2w_T", label, eT["uL2w"], eT["u_magnitude"])
                        R.add(f"{prefix}_uL2wmax", label, emax["u"], eT["u_magnitude"])
                        R.add(f"{prefix}_pL2mw_T", label, eT["pL2w"], eT["p_magnitude"])
                        R.add(f"{prefix}_uL2_unw_T", label, eT["uL2_unw"], eT["u_magnitude"])
                        R.add(f"{prefix}_pL2m_unw_T", label, eT["pL2_unw"], eT["p_magnitude"])
                    else:
                        R.add(f"{prefix}_uL2_T", label, eT["uL2"], eT["u_magnitude"])
                        R.add(f"{prefix}_uL2max", label, emax["u"], eT["u_magnitude"])
                        R.add(f"{prefix}_pL2m_T", label, eT["pL2"], eT["p_magnitude"])
                    dirichlet_defect(R, snap, dattrs, fld, axi, times[-1])
                if geom == "planar" and form == "lap" and order == 2 and scheme == "newton" and grid == "uniform":
                    sens = P["sensitivity_run"]
                    N = Ns[-1]
                    R.acc.q["planar_lap_bdf2_newton_uniform_uL2_T_finest"] = \
                        R.ser[f"{prefix}_uL2_T"]["error"][-1]
                    tight = new_solver(mesh, fld, form, scheme, order, force, bcs,
                                       newton_rtol=float(sens["newton_rtol_tight"]))
                    tight.initialize(u0_fn(fld, axi), 0.0, True)
                    tight.reset_stats()
                    ok, times = integrate(tight, FC.time_grid(spec, T_end, N))
                    eT = FC.errors(Evaluator(tight.snapshot(), axi), fld, times[-1])
                    R.acc.q["planar_lap_bdf2_newton_uniform_uL2_T_finest_tight"] = eT["uL2"]
                    R.entry(f"{prefix}_N{N}_tight", tight, form, axi, dt=T_end / N, mesh=mesh_entry(V, Tr, 0.125),
                            options={"newton_rtol": sens["newton_rtol_tight"]})
    return R.result()


# ============================================================================================== T2
def _hist_times(tn, h, k, omega=None):
    """Times of the history, newest first: uniform spacing h, or (varhist) the steps before t_{n-1} equal h / omega."""
    ts = [tn - h]
    hh = h if omega is None else h / omega
    for _ in range(k):
        ts.append(ts[-1] - hh)
    return ts


def _single_step(R, ts, fld, axi, tn, h, order, omega=None):
    times = _hist_times(tn, h, order, omega)
    ts.set_history(times, [u0_fn(fld, axi, t) for t in times])
    ts.reset_stats()
    t0 = time.time()
    r = ts.step_trial(h, order)
    ev_est = Evaluator(ts.estimate_snapshot(), axi)
    ev_un = Evaluator(ts.trial_snapshot(), axi)
    eff = FC.effectivity(ev_est, ev_un, fld, tn)
    return r, eff, ev_un, time.time() - t0


def error_control(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    fld = FC.load_field(P["data"])
    V, Tr = FC.structured_mesh(P["mesh"]["N"])
    mesh, seg, att = make_mesh(V, Tr, UNIT)
    verify(mesh, V, Tr)
    bcs, dattrs = standard_bcs(fld, False)
    form = P["forms"][0]
    force = check_fn(fld.f(form), False)
    mk = lambda nr=P["newton_rtol"]: new_solver(mesh, fld, form, P["scheme"], 2, force, bcs,
                                                newton_rtol=float(nr))
    E = P["effectivity"]
    tn = float(E["t_n"])
    ts = mk()
    effs = {}
    for k in (1, 2):
        for h in E["h"][str(k)]:
            r, eff, ev_un, sec = _single_step(R, ts, fld, False, tn, h, k)
            effs[(k, h)] = eff["effectivity"]
            R.acc.q[f"effectivity_bdf{k}_h{h:g}"] = eff["effectivity"]
            R.entry(f"effectivity_bdf{k}_h{h:g}", ts, form, False, dt=h, mesh=mesh_entry(V, Tr, 0.125), seconds=sec,
                    options={"order": k, "t_n": tn})
            if h == E["h"][str(k)][0]:
                R.acc.q[f"est_norm_ratio_planar_bdf{k}"] = r["est_norm"] / eff["est_L2"]
            dirichlet_defect(R, ts.trial_snapshot(), dattrs, fld, False, tn)
        hs = E["h"][str(k)]
        R.acc.q[f"effectivity_stability_bdf{k}"] = effs[(k, hs[1])] / effs[(k, hs[0])]
    vh = E["varhist"]
    for om in vh["omegas"]:
        r, eff, ev_un, sec = _single_step(R, ts, fld, False, tn, vh["h"], 2, omega=om)
        R.entry(f"varhist_om{om:.1f}", ts, form, False, dt=vh["h"], mesh=mesh_entry(V, Tr, 0.125), seconds=sec,
                options={"order": 2, "omega": om})
        R.acc.q[f"effectivity_varhist_ratio_om{om:.1f}"] = eff["effectivity"] / effs[(2, vh["h"])]
    # axisymmetric reported-norm checks
    A = E["axisymmetric"]
    fa = FC.load_field(A["data"])
    Va, Ta = FC.structured_mesh(A["mesh"]["N"][0])
    mesha, _, _ = make_mesh(Va, Ta, UNIT)
    bcsa, _ = standard_bcs(fa, True)
    forca = check_fn(fa.f(form), True)
    tsa = new_solver(mesha, fa, form, P["scheme"], 2, forca, bcsa, newton_rtol=float(P["newton_rtol"]))
    for k in (1, 2):
        h = A["h"][str(k)][0]
        r, eff, ev_un, sec = _single_step(R, tsa, fa, True, float(A["t_n"]), h, k)
        R.acc.q[f"est_norm_ratio_axi_bdf{k}"] = r["est_norm"] / eff["est_L2"]
        R.entry(f"axi_est_norm_bdf{k}", tsa, form, True, dt=h, mesh=mesh_entry(Va, Ta, 1.0 / A["mesh"]["N"][0]),
                seconds=sec, options={"order": k})
    # tolerance sweep
    S = P["rtol_sweep"]
    errs, steps, ratios = [], [], []
    for rtol in S["rtols"]:
        t0 = time.time()
        tsw = mk()
        tsw.initialize(u0_fn(fld, False), 0.0, True)
        tsw.reset_stats()
        res = tsw.adapt(float(P["T"]), float(rtol), float(S["S_u"]), float(S["h0"]))
        e = FC.errors(Evaluator(tsw.snapshot(), False), fld, res["times"][-1])
        errs.append(e["uL2"] / S["S_u"])
        steps.append(len(res["times"]) - 1)
        ratios.append(res["dt_max_ratio"])
        R.entry(f"sweep_rtol{rtol:g}", tsw, form, False, dt=res["dt_max"] or None, dt_max_ratio=res["dt_max_ratio"],
                mesh=mesh_entry(V, Tr, 0.125), seconds=time.time() - t0,
                options={"rtol": rtol, "steps": steps[-1], "rejected": res["rejected"], "reached": res["reached"]})
        dirichlet_defect(R, tsw.snapshot(), dattrs, fld, False, res["times"][-1])
        if not res["reached"]:
            R.notes.append(f"the sweep at rtol {rtol:g} did not reach T")
    R.ser["rtol_sweep"] = {"tols": [float(t) for t in S["rtols"]], "error": errs, "steps": steps}
    R.acc.q["rtol_sweep_dt_max_ratio"] = max(ratios)
    J = P["rejection_run"]
    tsr = mk()
    tsr.initialize(u0_fn(fld, False), 0.0, True)
    tsr.reset_stats()
    res = tsr.adapt(float(P["T"]), float(J["rtol"]), float(S["S_u"]), float(J["h0"]))
    R.acc.q["rejected_steps_from_large_h0"] = res["rejected"]
    R.entry("rejection_run", tsr, form, False, dt_max_ratio=res["dt_max_ratio"], mesh=mesh_entry(V, Tr, 0.125),
            options={"rtol": J["rtol"], "h0": J["h0"], "rejected": res["rejected"], "steps": len(res["times"]) - 1})
    return R.result()


# ============================================================================================== T7
def initial_projection(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    fld = FC.load_field(P["data"])
    fam = P["meshes"]["structured"]
    Ns = R.levels(fam["N"], 4)
    tm = P["time"]
    bcs, dattrs = standard_bcs(fld, False)
    prev = None
    for lev, N in enumerate(Ns):
        V, Tr = FC.structured_mesh(N)
        mesh, seg, att = make_mesh(V, Tr, UNIT)
        verify(mesh, V, Tr, V_coarser=prev)
        prev = V
        for form in P["forms"]:
            t0 = time.time()
            force = check_fn(fld.f(form), False)
            ts = new_solver(mesh, fld, form, tm["scheme"], int(tm["order"]), force, bcs,
                            newton_rtol=float(P["newton_rtol"]))
            ts.initialize(u0_fn(fld, False), 0.0, True)
            ts.reset_stats()
            snap0 = ts.snapshot()
            ev0 = Evaluator(snap0, False)
            e0 = FC.errors(ev0, fld, 0.0)
            dd = FC.discrete_divergence(ev0, V, False)
            R.acc.max("initial_div_dual_rel_max", dd["div_dual_rel"])
            dirichlet_defect(R, snap0, dattrs, fld, False, 0.0, key="dirichlet_nodal_data_defect_t0_max")
            ok, times = integrate(ts, [0.0] + [tm["dt"] * (i + 1) for i in range(int(tm["steps"]))])
            dirichlet_defect(R, ts.snapshot(), dattrs, fld, False, times[-1])
            label = f"{form}_N{N}"
            R.entry(label, ts, form, False, mesh=mesh_entry(V, Tr, 1.0 / N, "structured", lev),
                    seconds=time.time() - t0, options={"order": tm["order"], "dt": tm["dt"], "steps": tm["steps"]})
            R.add(f"{form}_u0L2", label, e0["uL2"], e0["u_magnitude"])
            R.add(f"{form}_u0H1", label, e0["uH1"], e0["u_magnitude"])
    return R.result()


# ============================================================================================== Taylor-Green
def taylor_green(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    fld = FC.load_field(P["data"])
    fam = P["meshes"]["rect"]
    ks = R.levels(fam["k"])
    T_end, N = float(P["T"]), int(P["N_steps"])
    bcs, dattrs = standard_bcs(fld, False)
    order = int(P["order"])
    kin = {}
    for var in P["variants"]:
        form, scheme = var["form"], var["scheme"]
        prefix = f"{form}_{scheme}"
        prev = None
        for lev, k in enumerate(ks):
            V, Tr, h = FC.family_mesh({"builder": "rect", "nx": fam["nx"], "ny": fam["ny"], "box": fam["box"],
                                       "k": fam["k"], "cell": 0.25}, fam["k"].index(k))
            mesh, seg, att = make_mesh(V, Tr, UNIT)
            verify(mesh, V, Tr, V_coarser=prev)
            prev = V
            runs = [(N, f"{prefix}_k{k}")]
            if lev == len(ks) - 1:
                runs.append((2 * N, f"{prefix}_k{k}_halfdt"))
            vals = {}
            for NN, label in runs:
                t0 = time.time()
                ts = new_solver(mesh, fld, form, scheme, order, None, bcs, newton_rtol=float(P["newton_rtol"]))
                ts.initialize(u0_fn(fld, False), 0.0, True)
                ts.reset_stats()
                ok, times = integrate(ts, np.linspace(0.0, T_end, NN + 1))
                snap = ts.snapshot()
                ev = Evaluator(snap, False)
                e = FC.errors(ev, fld, times[-1])
                R.entry(label, ts, form, False, dt=T_end / NN, mesh=mesh_entry(V, Tr, h, "rect", lev),
                        seconds=time.time() - t0, options={"order": order, "scheme": scheme, "N": NN})
                dirichlet_defect(R, snap, dattrs, fld, False, times[-1])
                vals[NN] = e
                if NN == N:
                    for nm, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2"), ("divL2", "divL2")):
                        mag = e["p_magnitude"] if nm == "pL2m" else e["u_magnitude"]
                        R.add(f"{prefix}_{nm}", f"{prefix}_k{k}", e[key], mag)
                    if lev == len(ks) - 1:
                        kin[prefix] = abs(FC.kinetic_energy(ev, fld.rho, False) -
                                          fld.rho / 4.0 * math.exp(-4 * math.pi ** 2 * fld.mu / fld.rho * times[-1]
                                                                   )) / (fld.rho / 4.0 * math.exp(
                                              -4 * math.pi ** 2 * fld.mu / fld.rho * times[-1]))
            if lev == len(ks) - 1:
                for nm, key in (("uL2", "uL2"), ("uH1", "uH1"), ("pL2m", "pL2"), ("divL2", "divL2")):
                    R.ser[f"{prefix}_{nm}"]["temporal_check"] = {"value_dt": vals[N][key],
                                                                 "value_dt_half": vals[2 * N][key]}
    R.acc.q["kinetic_energy_rel_err_T_finest"] = max(kin.values())
    return R.result()


# ============================================================================================== Womersley
def _wom(geom, kind):
    def fn(params, workdir, quick):
        return womersley(params, quick, geom, kind)
    return fn


def _wom_setup(R, fld, geom, form, scheme, order, V, Tr, newton_rtol):
    axi = geom == "axi"
    box = (0.0, 1.0, 0.0, 1.0) if axi else (0.0, 1.0, -1.0, 1.0)
    mesh, seg, att = make_mesh(V, Tr, rect_classify(*box))
    verify(mesh, V, Tr)
    if axi:
        bcs = [bc(1, "traction", check_fn(fld.traction(form, (0.0, -1.0)), True)), bc(2, "dirichlet"),
               bc(3, "traction", check_fn(fld.traction(form, (0.0, 1.0)), True)), bc(4, "axis")]
        dattrs = (2,)
    else:
        bcs = [bc(1, "dirichlet"), bc(2, "traction", check_fn(fld.traction(form, (1.0, 0.0)), False)),
               bc(3, "dirichlet"), bc(4, "traction", check_fn(fld.traction(form, (-1.0, 0.0)), False))]
        dattrs = (1, 3)
    ts = new_solver(mesh, fld, form, scheme, order, None, bcs, newton_rtol=newton_rtol)
    return ts, mesh, dattrs


def _wom_zero(X, Y, t):
    return np.zeros((np.shape(X)[0], 1))


def womersley(params, quick, geom, kind):
    R = Rec(params, quick)
    P = params
    axi = geom == "axi"
    fld = FC.load_field(P["data"])
    uk, hk, unw = ("uL2w", "uH1w", "uL2_unw") if axi else ("uL2", "uH1", None)
    rn = float(P["newton_rtol"])
    zero_data = lambda X, Y, t: np.zeros((np.shape(X)[0], 3 if axi else 2))
    if kind == "space":
        fam = P["meshes"]["space"]
        S = P["space_study"]
        ks = R.levels(fam["k"])
        T_end, N = float(S["T"]), int(S["N_steps"])
        prev = None
        for lev, k in enumerate(ks):
            V, Tr = FC.rect_mesh(fam["nx"], fam["ny"], k, *fam["box"])
            for form in P["forms"]:
                runs = [(N, f"space_{form}_k{k}")]
                if lev == len(ks) - 1:
                    runs.append((2 * N, f"space_{form}_k{k}_halfdt"))
                vals = {}
                for NN, label in runs:
                    t0 = time.time()
                    ts, mesh, dattrs = _wom_setup(R, fld, geom, form, "newton", int(S["order"]), V, Tr, rn)
                    if NN == N and form == P["forms"][0]:
                        verify(mesh, V, Tr, V_coarser=prev)
                    ts.initialize(u0_fn(fld, axi), 0.0, True)
                    ts.reset_stats()
                    ok, times = integrate(ts, np.linspace(0.0, T_end, NN + 1))
                    snap = ts.snapshot()
                    e = FC.errors(Evaluator(snap, axi), fld, times[-1])
                    R.entry(label, ts, form, axi, dt=T_end / NN, mesh=mesh_entry(V, Tr, fam["h_nominal"][lev], "rect", lev),
                            seconds=time.time() - t0, options={"N": NN, "scheme": "newton", "k": k})
                    dirichlet_defect(R, snap, dattrs, fld, axi, times[-1], data=zero_data)
                    vals[NN] = e
                    if NN == N:
                        R.add(f"space_{form}_{uk}", label, e[uk], e["u_magnitude"])
                        R.add(f"space_{form}_{hk}", label, e[hk], e["u_magnitude"])
                        if unw:
                            R.add(f"space_{form}_{unw}", label, e[unw], e["u_magnitude"])
                if lev == len(ks) - 1:
                    for nm in [uk, hk] + ([unw] if unw else []):
                        R.ser[f"space_{form}_{nm}"]["temporal_check"] = {"value_dt": vals[N][nm],
                                                                         "value_dt_half": vals[2 * N][nm]}
            prev = V
        return R.result()
    # time study
    ts_p = P["time_study"]
    T_end = float(ts_p["T"])
    Ns = R.levels(ts_p["N_steps"])
    k = P["meshes"]["time"]["k"]
    kc = P["meshes"]["time"]["check_mesh_k"]
    nx, ny, box = (2, 2, (0.0, 1.0, 0.0, 1.0)) if axi else (2, 4, (0.0, 1.0, -1.0, 1.0))
    V, Tr = FC.rect_mesh(nx, ny, k, *box)
    fin = {}
    for form in P["forms"]:
        for order, scheme in ts_p["variants"]:
            prefix = f"time_{form}_bdf{order}_{scheme}"
            for N in Ns:
                t0 = time.time()
                ts, mesh, dattrs = _wom_setup(R, fld, geom, form, scheme, order, V, Tr, rn)
                ts.initialize(u0_fn(fld, axi), 0.0, True)
                ts.reset_stats()
                emax = {"u": 0.0}

                def cb(n, t, s):
                    emax["u"] = max(emax["u"], FC.errors(Evaluator(s.snapshot(), axi), fld, t)[uk])
                ok, times = integrate(ts, np.linspace(0.0, T_end, N + 1), cb)
                snap = ts.snapshot()
                e = FC.errors(Evaluator(snap, axi), fld, times[-1])
                label = f"{prefix}_N{N}"
                R.entry(label, ts, form, axi, dt=T_end / N, mesh=mesh_entry(V, Tr, 0.5 / 2 ** k),
                        seconds=time.time() - t0, options={"order": order, "scheme": scheme, "N": N})
                dirichlet_defect(R, snap, dattrs, fld, axi, times[-1], data=zero_data)
                R.add(f"{prefix}_uL2_T", label, e[uk], e["u_magnitude"])
                R.add(f"{prefix}_uL2max", label, emax["u"], e["u_magnitude"])
                if order == 2 and scheme == "newton" and N == Ns[-1]:
                    fin[form] = e[uk]
    # contamination: the finest dt on mesh k + 1 (BDF2 / Newton), |e_T(k) - e_T(k+1)| / e_T(k)
    V1, T1 = FC.rect_mesh(nx, ny, kc, *box)
    for form in P["forms"]:
        N = Ns[-1]
        t0 = time.time()
        ts, mesh, dattrs = _wom_setup(R, fld, geom, form, "newton", 2, V1, T1, rn)
        ts.initialize(u0_fn(fld, axi), 0.0, True)
        ts.reset_stats()
        ok, times = integrate(ts, np.linspace(0.0, T_end, N + 1))
        e1 = FC.errors(Evaluator(ts.snapshot(), axi), fld, times[-1])[uk]
        R.entry(f"contamination_{form}_k{kc}_N{N}", ts, form, axi, dt=T_end / N,
                mesh=mesh_entry(V1, T1, 0.5 / 2 ** kc), seconds=time.time() - t0,
                options={"order": 2, "scheme": "newton", "N": N, "k": kc})
        R.acc.q[f"time_study_spatial_contamination_{form}_bdf2_newton"] = abs(fin[form] - e1) / fin[form]
    return R.result()


womersley_2d_space = _wom("2d", "space")
womersley_2d_time = _wom("2d", "time")
womersley_axi_space = _wom("axi", "space")
womersley_axi_time = _wom("axi", "time")


# ============================================================================================== energy budget
def energy_budget(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    T_end, N = float(P["T"]), int(P["N_steps"])
    for geom in ("planar", "axisymmetric"):
        axi = geom == "axisymmetric"
        g = "axi" if axi else "planar"
        fld = FC.load_field(P["data"][geom])
        Nm = P["meshes"][geom]["N"]
        V, Tr = FC.structured_mesh(Nm)
        mesh, seg, att = make_mesh(V, Tr, UNIT)
        verify(mesh, V, Tr)
        bcs, dattrs = standard_bcs(fld, axi)
        for form in P["forms"]:
            force = check_fn(fld.f(form), axi)
            for var in P["variants"]:
                order, scheme, grid = int(var["order"]), var["scheme"], var["grid"]
                prefix = f"{g}_{form}_bdf{order}_{scheme}_{grid}"
                t0 = time.time()
                tg = FC.time_grid(P["grids"][grid], T_end, N)
                ts = new_solver(mesh, fld, form, scheme, order, force, bcs, newton_rtol=float(P["newton_rtol"]))
                ts.initialize(u0_fn(fld, axi), 0.0, True)
                ts.reset_stats()
                evs = [Evaluator(ts.snapshot(), axi)]
                ok, times = integrate(ts, tg, lambda n, t, s: evs.append(Evaluator(s.snapshot(), axi)))
                R.entry(prefix, ts, form, axi, dt=T_end / N, dt_max_ratio=ratio_of(times),
                        mesh=mesh_entry(V, Tr, 1.0 / Nm), seconds=time.time() - t0,
                        options={"order": order, "scheme": scheme, "grid": grid, "N": N})
                dirichlet_defect(R, ts.snapshot(), dattrs, fld, axi, times[-1])
                if ok:
                    res = FC.run_energy_identity(evs, np.asarray(times), order, fld, form, scheme)
                    R.acc.q[f"{prefix}_identity_defect_max"] = res["defect_rel_max"]
                    R.acc.q[f"{prefix}_positive_control_min"] = min(res["controls_min"].values())
    return R.result()


# ============================================================================================== X1
def xcode_cavity(params, workdir, quick):
    R = Rec(params, quick)
    P = params
    fld = FC.load_field(P["data"])
    Ns = R.levels(P["meshes"]["structured"]["N"], 1)
    dt = float(P["dt"])
    bcs = [bc(a, "dirichlet", check_fn(fld.u, False)) for a in (1, 2, 3, 4)]
    for N in Ns:
        V, Tr = FC.structured_mesh(N)
        mesh, seg, att = make_mesh(V, Tr, UNIT)
        verify(mesh, V, Tr)
        for form, scheme in P["variants"]:
            for tt in P["times"]:
                t0 = time.time()
                ts = new_solver(mesh, fld, form, scheme, 2, None, bcs, newton_rtol=float(P["newton_rtol"]))
                ts.initialize(u0_fn(fld, False), 0.0, True)
                ts.reset_stats()
                nst = int(round(tt / dt))
                ok, times = integrate(ts, [dt * i for i in range(nst + 1)])
                snap = ts.snapshot()
                fl = FC.flow_functionals(Evaluator(snap, False), fld.mu, form)
                tag = f"{form}_{scheme}"
                for q in ("Ekin", "Enst", "Diss"):
                    R.acc.q[f"{tag}_{q}_t{int(tt)}_N{N}"] = fl[q]
                R.entry(f"{tag}_t{int(tt)}_N{N}", ts, form, False, dt=dt, mesh=mesh_entry(V, Tr, 1.0 / N),
                        seconds=time.time() - t0, options={"order": 2, "scheme": scheme, "N": N, "t_end": tt})
                dirichlet_defect(R, snap, (1, 2, 3, 4), fld, False, times[-1])
    return R.result()


temporal_mms.__doc__ = "T1 mms_ns_transient_temporal"
