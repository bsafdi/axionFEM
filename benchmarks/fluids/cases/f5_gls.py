"""F5 stabilized P1/P1 (SUPG + PSPG + LSIC) planar runner: the cases (set-up, solve, hand the nodal arrays over).

Every norm, tau value, residual comparison, Taylor slope, patch error and cavity functional comes from the check module's
module f5_gls_check (imported as C5; it re-exports the fixed F1a/F2a meshes, norms and diagnostics).  This module only
builds the fixed meshes with the fixed builders, solves through the public API axionfem.fluids.gls_problem and hands
nodal arrays to the check module.  Nothing here recomputes a quantity of the check module, and the solver is never given
anything but the fixed data (the case code reads `params` only).

Case contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.  quick=True: the first two levels only.

Failed solves are results (convention reused): a solve that misses its gate is recorded in runs[]
(converged, relative_residual, history) and only the quantities that need its fields are skipped, so the
acceptance check fails the case on the solver gate.  The negative control is the one exception: its Galerkin and
SUPG+LSIC-only solves are expected to fail; they are reported under result["variant_runs"] (never in runs[]), and a
variant that did not converge counts as REFUSED (pressure error 1e300, the fixed failure convention).

Conventions: form "lap" = mu grad u : grad v -> API "laplacian", "sym" = 2 mu eps:eps -> API "stress"; dynamic units.
The nodal P1 arrays are in the fixed vertex order of the check module's builders; the Dirichlet data are the P1 nodal
interpolant of the fixed callable at the boundary vertices (boundary vertices from f1a_stokes_check.boundary_edges).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import f1a_stokes_check as C1
import f2a_ns as F2          # the steady Navier-Stokes runner's helpers (core identity, run entry, series)
import f2a_ns_check as C2
import f5_gls_check as C5
from axionfem import fluids

FORM = {"lap": "laplacian", "sym": "stress"}
MODE = {"ns": "navier_stokes", "stokes": "stokes", "oseen": "oseen"}
NAMES = {"uL2": "uL2", "uH1": "uH1", "pL2": "pL2", "divL2": "divL2"}
ROOT = Path(__file__).resolve().parents[3]


# ============================================================================================== state of one case
class Run:
    """Runs, series, quantities and notes of one case; every solve goes through solve()."""

    def __init__(self, params, quick):
        self.P, self.quick = params, quick
        self.ident = F2.core_identity()
        ns = params.get("nonlinear_solver") or {}
        self.rtol = float(ns.get("newton_rtol", 1e-12))
        self.maxit = int(ns.get("newton_maxit", 40))
        self.runs, self.ser, self.acc, self.notes, self.failed = [], {}, F2.Acc(), [], []
        self.variant_runs = []

    def solve(self, level, V, T, info, h, *, mode, form, family=None, refinement_level=None, record=True, **kw):
        """One P1/P1 GLS solve.  Returns (solution, converged).  record=False: the run goes to variant_runs."""
        opts = dict(mode=MODE[mode], form=FORM[form], lu_threads=1, newton_rtol=self.rtol, newton_maxit=self.maxit)
        opts.update(kw)
        t0 = time.time()
        sol = fluids.gls_problem(V, T, solve=True, **opts)
        sec = time.time() - t0
        shown = {k: v for k, v in opts.items() if k not in ("mu", "mu_nodal", "force", "advect", "history",
                                                            "dirichlet")}
        e = F2.run_entry(level, sol, sec, h, info, family, refinement_level, form, mode, False, shown)
        e["solver"]["stabilized_relative_residual"] = e["solver"]["relative_residual"]
        e["solver"]["geometry"] = "planar"
        (self.runs if record else self.variant_runs).append(e)
        ok = bool(sol.diagnostics["converged"])
        if not ok and record:
            self.failed.append(level)
        return sol, ok

    def result(self):
        me = self.ident
        if self.failed:
            self.notes.append(f"{len(self.failed)} solve(s) missed the gate (recorded; their field quantities skipped): "
                              + ", ".join(self.failed[:12]) + (" ..." if len(self.failed) > 12 else ""))
        self.notes.append(f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]})")
        out = {"runs": self.runs, "series": self.ser, "quantities": self.acc.q, "notes": " ".join(self.notes)}
        if self.variant_runs:
            out["variant_runs"] = self.variant_runs
        return out


# ============================================================================================== helpers
def boundary_vertices(T):
    return np.array(sorted({int(v) for e in C1.boundary_edges(T) for v in e}), dtype=np.int32)


def dirichlet_of(V, T, g):
    """(nodes, values): the nodal P1 interpolant of the fixed callable g(X, Y) -> [..., 2] at the boundary vertices."""
    b = boundary_vertices(T)
    return b, np.asarray(g(V[b, 0], V[b, 1]), float)


def mesh_info(V, T, Vc=None):
    return C5.verify_mesh(V, T, V, T, V_coarser=Vc)


def nodal_defect(R, V, T, sol, g):
    b = boundary_vertices(T)
    R.acc.max("dirichlet_nodal_data_defect_max", C1.dirichlet_nodal_defect(V[b], sol.u()[b], g))


def mean_weights(V, T):
    """c_a = int N_a (the weights of the solver's mean-free pressure)."""
    geo = C5.element_geometry(V, T)
    c = np.zeros(len(V))
    np.add.at(c, T.ravel(), np.repeat(geo["area"] / 3.0, 3))
    return c


def load_data(entry):
    """sha256-verified JSON of a fixed data file (run.parameters)."""
    raw = (ROOT / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return json.loads(raw)


def mms_family(R, F, fam, prefix, *, mode, form, rho, force, mu_kind="analytic", advect=None, scales=None, g=None,
               names=NAMES):
    """Solve one (mode, form, viscosity kind) over one fixed family; norms of the converged levels; nodal checks.
    Returns [(level, norms)]."""
    g = g or F.u
    rows = []
    for (V, T, Vc, h, i, lab) in F2.family_meshes(fam, R.P, R.quick):
        info = mesh_info(V, T, Vc)
        b, vals = dirichlet_of(V, T, g)
        kw = dict(rho=rho, force=force, dirichlet=(b, vals))
        if mu_kind == "nodal":
            kw["mu_nodal"] = F.mu(V[:, 0], V[:, 1])
        elif callable(getattr(F, "mu_and_grad", None)):
            kw["mu"] = F.mu_and_grad
        else:
            kw["mu"] = R.P["mu_value"]
        if advect is not None:
            kw["advect"] = advect
        if scales:
            kw["viscosity_scales"] = list(scales)
        lvl = f"{prefix}_L{i}"
        sol, ok = R.solve(lvl, V, T, info, h, mode=mode, form=form, family=prefix, refinement_level=i, **kw)
        R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        if not ok:
            continue
        n = C5.norms(C5.NodalEval(V, T, sol.u(), sol.p()), F, "mean_free")
        nodal_defect(R, V, T, sol, g)
        rows.append((lvl, n))
    F2.put_series(R, prefix, rows, names)
    return rows


def load_fields(P):
    """The fixed GlsField(s) of run.parameters.mms_data: one entry (dict with a path) or several named entries."""
    d = P["mms_data"]
    if "path" in d:
        return C5.load_gls_field(d)
    return {k: C5.load_gls_field(v) for k, v in d.items()}


# ============================================================================================== C1 patch
def patch_consistency(params, workdir, quick):
    """C1 patch_gls_consistency: an exact P1/P1 solution is reproduced to round-off (Stokes, Oseen, Navier-Stokes; both
    viscous forms; distorted and strongly graded meshes; steady and transient BDF1/BDF2 steps from the exact history)."""
    R = Run(params, quick)
    pt = C5.Patch()
    meshes = {"distorted_k2": C2.distorted_unit(2), "graded": C5.graded_mesh(8, 8, 100.0)}
    tr = params["transient"]
    dt, nsteps = float(tr["dt"]), int(tr["steps"])
    for mname, (V, T) in meshes.items():
        info = mesh_info(V, T)
        b = boundary_vertices(T)
        c = mean_weights(V, T)

        def errors(sol, t):
            uex = pt.u(V[:, 0], V[:, 1], t)
            pex = pt.p(V[:, 0], V[:, 1], t)
            pex = pex - float(c @ pex) / float(c.sum())
            return float(np.max(np.abs(sol.u() - uex))), float(np.max(np.abs(sol.p() - pex)))

        for form in params["viscous_forms"]:
            for mode in params["modes"]:
                f = pt.f(form, convection=(mode != "stokes"))
                kw = dict(mu=pt.mu, rho=pt.rho, force=f, dirichlet=(b, pt.u(V[b, 0], V[b, 1])))
                if mode == "oseen":
                    kw["advect"] = lambda x, y: pt.u(x, y)
                lvl = f"steady_{mname}_{mode}_{form}"
                sol, ok = R.solve(lvl, V, T, info, None, mode=mode, form=form, **kw)
                R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
                if not ok:
                    continue
                eu, ep = errors(sol, 0.0)
                R.acc.max("patch_err_u_steady_max", eu)
                R.acc.max("patch_err_p_steady_max", ep)
                nodal_defect(R, V, T, sol, lambda x, y: pt.u(x, y))
            for bdf in tr["bdf"]:
                for k in range(1, nsteps + 1):
                    t = k * dt
                    hist = [pt.u(V[:, 0], V[:, 1], (k - j) * dt) for j in range(1, int(bdf) + 1)]
                    f = pt.f(form, convection=True, t=t, transient=True)
                    kw = dict(mu=pt.mu, rho=pt.rho, force=f, dirichlet=(b, pt.u(V[b, 0], V[b, 1], t)), dt=dt,
                              bdf=int(bdf), history=hist)
                    lvl = f"transient_{mname}_bdf{bdf}_{form}_step{k}"
                    sol, ok = R.solve(lvl, V, T, info, None, mode="ns", form=form, **kw)
                    R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
                    if not ok:
                        continue
                    eu, ep = errors(sol, t)
                    R.acc.max("patch_err_u_transient_max", eu)
                    R.acc.max("patch_err_p_transient_max", ep)
                    nodal_defect(R, V, T, sol, lambda x, y, t=t: pt.u(x, y, t))
    R.notes.append("Patch: every steady and transient problem has a P1/P1 exact solution; errors are absolute maxima over "
                   "the vertices (pressure compared mean-free in the c-weighted sense of the solver).")
    return R.result()


# ============================================================================================== R1 residual parity + Taylor
def residual_jacobian(params, workdir, quick):
    """R1 gls_residual_jacobian: the residual vector of the solver at the seven fixed states against the numpy assembly
    (f5_gls_check.parity_defect), and Taylor tests of the stabilized residual (the exact Jacobian: tau differentiated)."""
    R = Run(params, quick)
    fields = {k: C5.load_gls_field(v) for k, v in params["fields"].items()}
    for name in params["states"]["names"]:
        s = C5.parity_state(name, fields)
        mode = {"ns": "navier_stokes"}.get(s["mode"], s["mode"])
        kw = dict(mu=s["mu"]) if s["nu_nodal"] is None else dict(mu_nodal=s["nu_nodal"])
        prob = fluids.gls_problem(s["V"], s["T"], rho=s["rho"], force=s["f"], advect=s["advect"] if mode == "oseen" else None,
                                  mode=mode, form=FORM[s["form"]], dt=float(s["dt"] or 0.0), bdf=s["bdf"],
                                  history=s["hist"], solve=False, lu_threads=1, **kw)
        Ru, Rp = prob.residual_full(s["u"], s["p"])
        R.acc.q[f"parity_rel_{name}"] = C5.parity_defect(name, fields, Ru, Rp)
    tay = params["taylor"]
    eps = [float(e) for e in tay["eps"]]
    seed = int(tay["seed"])
    for tag, fn, scales in (("steady_conv", "gls_sq_conv", [200.0, 60.0, 20.0, 6.0, 2.0, 1.0]),
                            ("steady_mix", "gls_sq_mix", [4.0, 1.0])):
        F = fields[fn]
        V, T = C2.structured_mesh(8)
        info = mesh_info(V, T)
        b, vals = dirichlet_of(V, T, F.u)
        sol, ok = R.solve(f"taylor_{tag}", V, T, info, 0.125, mode="ns", form="sym", mu=F.mu_and_grad, rho=F.rho,
                          force=F.f("sym"), dirichlet=(b, vals), viscosity_scales=scales)
        R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
        if not ok:
            continue
        x0 = sol.state()
        rng = np.random.Generator(np.random.PCG64(seed))
        xi = rng.standard_normal(x0.size)
        d = rng.standard_normal(x0.size)
        x = x0 + 0.1 * xi * float(np.abs(x0).max())
        R0 = sol.residual(x)
        Jd = sol.jacobian_action(x, d)
        Rs = [sol.residual(x + e * d) for e in eps]
        t = C5.taylor_test(R0, Rs, Jd, eps)
        R.acc.q[f"taylor_slope_r2_{tag}"] = t["slope_r2"]
        R.acc.q[f"taylor_slope_r1_{tag}"] = t["slope_r1"]
        R.runs[-1]["taylor"] = {kk: t[kk] for kk in ("slope_r1", "slope_r2", "r2_over_eps2_spread", "r2_over_r1_min_eps",
                                                      "r1", "r2")}
        R.runs[-1]["taylor"]["n_free"] = int(x0.size)
        R.runs[-1]["taylor"]["layout"] = list(map(int, sol.layout()))
    return R.result()


# ============================================================================================== T1 tau pins
def tau_pins(params, workdir, quick):
    """T1 gls_tau_pins: tau_supg, tau_pspg, tau_lsic reported by the solver (tau_at: the routine the assembly calls) at the
    pinned rule points of the six fixed states against f5_gls_check.tau_fields; dt independence of tau_pspg, the small-dt
    limit of tau_supg, and the vertex-renumbering invariance."""
    R = Run(params, quick)

    def problem(s, T=None):
        mu = s["mu"]
        kw = dict(mu_nodal=mu) if np.ndim(mu) else dict(mu=float(mu))
        dt = s["dt"]
        hist = (np.zeros((len(s["V"]), 2)),) if dt else ()
        return fluids.gls_problem(s["V"], s["T"] if T is None else T, rho=s["rho"], mode="navier_stokes", form="stress",
                                  dt=float(dt or 0.0), bdf=1, history=hist, solve=False, lu_threads=1,
                                  **kw)

    rep = {}
    for name in C5.TAU_STATES:
        s = C5.tau_state(name)
        rep[name] = problem(s).tau_at(s["w"])
        for key, v in C5.tau_state_defects(name, rep[name]).items():
            R.acc.q[f"{key}_defect_{name}"] = v
    a, b = rep["transient_dt1e-2"]["tau_pspg"], rep["transient_dt1e-6"]["tau_pspg"]
    R.acc.q["tau_pspg_dt_independence"] = float(np.max(np.abs(a - b) / np.abs(a)))
    s6 = rep["transient_dt1e-6"]["tau_supg"]
    R.acc.q["tau_supg_small_dt_limit"] = float(np.max(np.abs(s6 / (1e-6 / 2.0) - 1.0)))
    s = C5.tau_state("variable_viscosity")
    Tr = C5.renumber_triangles(s["T"])
    R.acc.q["tau_vertex_renumbering_defect"] = C5.tau_renumber_defect(rep["variable_viscosity"],
                                                                      problem(s, Tr).tau_at(s["w"]))
    # solver-gate witnesses (nothing is compared): a Stokes solve and a transient Navier-Stokes step on the fixed N = 8 mesh
    V, T = C2.structured_mesh(8)
    info = mesh_info(V, T)
    b_nodes = boundary_vertices(T)
    zero = (b_nodes, np.zeros((len(b_nodes), 2)))
    sol, ok = R.solve("witness_stokes_N8", V, T, info, 0.125, mode="stokes", form="sym", mu=1.0, rho=1.25,
                      force=lambda x, y: np.stack([np.sin(3 * x) * y, np.cos(2 * y) * x], -1), dirichlet=zero)
    R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
    sol, ok = R.solve("witness_transient_dt1e-2_N8", V, T, info, 0.125, mode="ns", form="sym", mu=0.05, rho=1.0,
                      force=lambda x, y: np.stack([np.sin(3 * x) * y, np.cos(2 * y) * x], -1), dirichlet=zero,
                      dt=1e-2, bdf=1, history=(np.zeros((len(V), 2)),))
    R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
    R.notes.append("tau arrays come from the solver's tau_at (same routine as the assembly); the two solves in runs[] only "
                   "witness the solver gate.")
    return R.result()


# ============================================================================================== M1 Stokes
def mms_stokes(params, workdir, quick):
    """M1 mms_gls_stokes_2d: the gls_sq_mix field without convection (w = 0: PSPG + LSIC active), both viscous forms,
    variable analytic mu and its P1 nodal interpolant, structured and distorted families."""
    R = Run(params, quick)
    F = C5.load_gls_field(params["mms_data"])
    rho = F.rho
    for form in ("lap", "sym"):
        mms_family(R, F, "structured", f"{form}_struct", mode="stokes", form=form, rho=rho, force=F.f_no_convection(form))
    mms_family(R, F, "structured", "sym_nodalmu_struct", mode="stokes", form="sym", rho=rho,
               force=F.f_no_convection("sym"), mu_kind="nodal")
    for form in ("lap", "sym"):
        mms_family(R, F, "distorted", f"{form}_dist", mode="stokes", form=form, rho=rho, force=F.f_no_convection(form))
    return R.result()


# ============================================================================================== M2 Oseen and Navier-Stokes
def mms_ns(params, workdir, quick):
    """M2 mms_gls_ns_2d: Navier-Stokes in the diffusive, mixed and convection-dominated regimes (viscosity continuation
    in the last), both viscous forms in the mixed regime, distorted and nodal-mu variants, and Oseen (w = the exact field)."""
    R = Run(params, quick)
    fl = {k: C5.load_gls_field(v) for k, v in params["mms_data"].items()}
    cont = [200.0, 80.0, 30.0, 12.0, 5.0, 2.0, 1.0]
    D, M, Cv = fl["gls_sq_diff"], fl["gls_sq_mix"], fl["gls_sq_conv"]
    rho = M.rho
    mms_family(R, D, "structured", "diff_sym", mode="ns", form="sym", rho=rho, force=D.f("sym"))
    for form in ("lap", "sym"):
        mms_family(R, M, "structured", f"mix_{form}", mode="ns", form=form, rho=rho, force=M.f(form))
    mms_family(R, M, "distorted", "mixdist_sym", mode="ns", form="sym", rho=rho, force=M.f("sym"))
    mms_family(R, M, "structured", "mixnodal_sym", mode="ns", form="sym", rho=rho, force=M.f("sym"), mu_kind="nodal")
    mms_family(R, Cv, "structured", "conv_sym", mode="ns", form="sym", rho=rho, force=Cv.f("sym"), scales=cont)
    mms_family(R, M, "structured", "oseen_mix_sym", mode="oseen", form="sym", rho=rho, force=M.f("sym"),
               advect=lambda x, y: M.u(x, y))
    mms_family(R, Cv, "structured", "oseen_conv_sym", mode="oseen", form="sym", rho=rho, force=Cv.f("sym"),
               advect=lambda x, y: Cv.u(x, y))
    R.notes.append(f"convection-dominated regime: viscosity continuation {cont} (any path is allowed; the solution is that of "
                   "the steady discrete equations)")
    return R.result()


# ============================================================================================== A1 Kovasznay
def kovasznay(params, workdir, quick):
    """A1 analytic_gls_kovasznay_re40: the exact NS solution (f = 0): every stabilization term must vanish on it."""
    R = Run(params, quick)
    F = C5.load_gls_field(params["mms_data"])
    R.P = dict(params, mu_value=1.0 / 40.0)
    force = None                      # exact NS solution: f = 0 (the generator checked it symbolically)
    for form in params["viscous_forms"]:
        mms_family(R, F, "kovasznay_rect", f"{form}_kov", mode="ns", form=form, rho=float(params["rho"]), force=force,
                   scales=[4.0, 2.0, 1.0])
    return R.result()


# ============================================================================================== N1 negative control
def control_no_pspg(params, workdir, quick):
    """N1 control_gls_no_pspg: the inf-sup estimates of the check module (independent of the solver) and the solver-level
    Stokes MMS without PSPG (and with SUPG + LSIC only), against the scheme.  Every variant reports the MEASURED pressure
    error of its returned iterate (converged or not; the unstabilized systems are singular and return garbage); 1e300 only
    if the solve raises or the iterate is not finite.  Each variant's gate residual and LU residual are recorded as
    quantities.  Nothing is rescued."""
    R = Run(params, quick)
    Ns = list(params["meshes"]["structured"]["N"])
    beta = {}
    for N in (Ns[:2] if quick else Ns):
        V, T = C2.structured_mesh(N)
        st = C5.inf_sup_p1p1(V, T, 1.0, stab=True)
        beta[N] = st["beta"]
        R.acc.q[f"infsup_pspg_zero_modes_beyond_constant_N{N}"] = st["zero_modes"] - 1
        R.acc.q[f"infsup_beta_pspg_N{N}"] = st["beta"]
    if 8 in beta and 32 in beta:
        R.acc.q["infsup_beta_pspg_ratio_N32_N8"] = beta[32] / beta[8]
    V, T = C2.structured_mesh(16)
    g16 = C5.inf_sup_p1p1(V, T, 1.0, stab=False)
    R.acc.q["infsup_galerkin_zero_modes_beyond_constant_N16"] = g16["zero_modes"] - 1
    if 16 in beta:
        R.acc.q["infsup_galerkin_over_pspg_beta_N16"] = g16["beta"] / beta[16]
    mm = params["mms"]
    F = C5.load_gls_field(mm["field"])
    variants = {"pspg": dict(), "galerkin": dict(stabilization={"supg": False, "pspg": False, "lsic": False}),
                "supg_lsic_only": dict(stabilization={"supg": True, "pspg": False, "lsic": True})}
    for N in (mm["N"][:1] if quick else mm["N"]):
        V, T = C2.structured_mesh(N)
        info = mesh_info(V, T)
        b, vals = dirichlet_of(V, T, F.u)
        err = {}
        for vn, vkw in variants.items():
            lvl = f"control_{vn}_N{N}"
            try:
                sol, ok = R.solve(lvl, V, T, info, 1.0 / N, mode="stokes", form="sym", mu=F.mu_and_grad, rho=F.rho,
                                  force=F.f_no_convection("sym"), dirichlet=(b, vals), record=(vn == "pspg"), **vkw)
            except Exception as exc:     # noqa: BLE001  -- a raise is a result of this control: the sentinel 1e300
                err[vn] = 1e300
                R.variant_runs.append({"level": lvl, "refused": f"{type(exc).__name__}: {exc}"})
                R.notes.append(f"{lvl}: the solve RAISED ({type(exc).__name__}); pressure error reported as 1e300")
                continue
            d = sol.diagnostics
            R.acc.q[f"{vn}_gate_N{N}"] = float(d["relative_residual"])
            R.acc.q[f"{vn}_lu_residual_N{N}"] = float(d["lu_residual"])
            R.acc.q[f"{vn}_converged_N{N}"] = float(ok)
            if vn == "pspg":
                R.acc.q[f"stabilized_solve_converged_N{N}"] = float(ok)
            # the MEASURED pressure error of the returned iterate, converged or not (the fixed convention, clarified:
            # 1e300 only when the solve raises or the iterate is not finite)
            with np.errstate(all="ignore"):
                try:
                    pe = float(C5.norms(C5.NodalEval(V, T, sol.u(allow_unconverged=True),
                                                     sol.p(allow_unconverged=True)), F, "mean_free")["pL2"])
                except Exception:        # noqa: BLE001
                    pe = float("nan")
            err[vn] = pe if math.isfinite(pe) else 1e300
            R.acc.q[f"{vn}_pL2_N{N}"] = err[vn]
        e0 = err.get("pspg", float("nan"))
        R.acc.q[f"pressure_error_ratio_galerkin_over_pspg_N{N}"] = err.get("galerkin", 1e300) / e0
        R.acc.q[f"pressure_error_ratio_supg_lsic_over_pspg_N{N}"] = err.get("supg_lsic_only", 1e300) / e0
        R.notes.append(f"N = {N}: MEASURED pressure error of the returned iterate: pspg {e0:.3e}, galerkin "
                       f"{err.get('galerkin'):.3e}, supg+lsic only {err.get('supg_lsic_only'):.3e} (gate residuals "
                       f"{R.acc.q.get(f'galerkin_gate_N{N}'):.2e} / {R.acc.q.get(f'supg_lsic_only_gate_N{N}'):.2e}; "
                       f"1e300 only if a solve raises or the iterate is not finite)")
    return R.result()


# ============================================================================================== D1 small dt
def transient_small_dt(params, workdir, quick):
    """D1 mms_gls_transient_small_dt: BDF2, 4 steps from the discretely divergence-free steady state at t = 0, error
    against the exact field at t = 4 dt; the small-dt PSPG check."""
    R = Run(params, quick)
    data = load_data(params["mms_data"])
    rho = float(data["rho"])
    gx = [compile(s, "<grad_mu>", "eval") for s in data["grad_mu"]]
    ns_ = {"__builtins__": {}, "cos": np.cos, "sin": np.sin, "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi}

    def mu_and_grad(x, y):
        fld = C5.TimeField(data, 0.0)
        g = np.stack([np.broadcast_to(np.asarray(eval(c, ns_, {"x": x, "y": y}), float), np.shape(x)) for c in gx], -1)
        return fld.mu(x, y), g

    dts = [float(d) for d in params["dts"]]
    nstep = int(params["steps"])
    Ns = list(params["meshes"]["structured"]["N"])
    for N in (Ns[:1] if quick else Ns):
        V, T = C2.structured_mesh(N)
        info = mesh_info(V, T)
        h = 1.0 / N
        f0 = C5.TimeField(data, 0.0)
        b, vals = dirichlet_of(V, T, f0.u)
        sol0, ok0 = R.solve(f"steady_t0_N{N}", V, T, info, h, mode="ns", form="sym", mu=mu_and_grad, rho=rho,
                            force=f0.f("sym"), dirichlet=(b, vals))
        R.acc.min("nullspace_detected", int(sol0.diagnostics["nullspace_detected"]))
        if not ok0:
            continue
        u0 = sol0.u()
        errs = {}
        for dt in (dts[:2] if quick else dts):
            up, upp = u0, u0
            good, p_last = True, None
            for k in range(1, nstep + 1):
                t = k * dt
                ft = C5.TimeField(data, t)
                b, vals = dirichlet_of(V, T, ft.u)
                sol, ok = R.solve(f"step_N{N}_dt{dt:g}_k{k}", V, T, info, h, mode="ns", form="sym", mu=mu_and_grad,
                                  rho=rho, force=ft.f("sym"), dirichlet=(b, vals), dt=dt, bdf=2, history=(up, upp))
                R.acc.min("nullspace_detected", int(sol.diagnostics["nullspace_detected"]))
                if not ok:
                    good = False
                    break
                upp, up, p_last = up, sol.u(), sol.p()
            if good:
                n = C5.norms(C5.NodalEval(V, T, up, p_last), C5.TimeField(data, nstep * dt), "mean_free")
                errs[dt] = {k: n[k] for k in ("uL2", "uH1", "pL2")}
        if 0.1 in errs:
            for key in ("pL2", "uL2", "uH1"):
                grow = [errs[d][key] / errs[0.1][key] for d in errs if d != 0.1 and d <= 1e-2 + 1e-15]
                if grow:
                    R.acc.q[f"small_dt_growth_{key}_N{N}"] = max(grow)
        if N == 16 and 1e-5 in errs:
            R.acc.q["small_dt_pL2_absolute_N16_dt1e-5"] = errs[1e-5]["pL2"]
        R.notes.append(f"N = {N}: pL2(dt) = " + ", ".join(f"{d:g}: {errs[d]['pL2']:.3e}" for d in errs))
    return R.result()


# ============================================================================================== X1, X2 cavities
def _re_stages(cont, Re):
    st = [float(r) for r in cont if float(r) < Re] + [float(Re)]
    return st


def _cavity_quantities(R, V, T, sol, form, mu, N, Re, prefix, keys):
    ev = C5.NodalEval(V, T, sol.u(), sol.p())
    # run.parameters.quantities: f5_gls_check.cavity_quantities is the definition (Ekin, Diss, extremes; its Diss is the SYM
    # form of the state, whichever viscous form was solved); Enst is form-independent (f2a_ns_check.functionals)
    fun = C2.functionals(ev, "sym", mu)
    cq = C5.cavity_quantities(V, T, sol.u(), mu, N)
    vals = {"Ekin": cq["Ekin"], "Diss": cq["Diss"], "Enst": fun["Enst"], "ux_min_vc": cq["ux_min_vc"],
            "uy_max_hc": cq["uy_max_hc"], "uy_min_hc": cq["uy_min_hc"]}
    for k in keys:
        R.acc.q[f"{prefix}_{k}_N{N}"] = vals[k]
    return vals


def xcode_cavity(params, workdir, quick):
    """X1 xcode_gls_cavity: the pinned discrete problem on the singular-lid cavity (corner rule: lid corner vertices take
    the wall value 0), Re = 100 and 1000 by Re continuation, N = 32, 64, both viscous forms."""
    R = Run(params, quick)
    lid = F2.lid_data()
    for N in (params["meshes"]["structured"]["N"][:1] if quick else params["meshes"]["structured"]["N"]):
        V, T = C2.structured_mesh(N)
        info = mesh_info(V, T)
        b, vals = dirichlet_of(V, T, lid)
        for Re in params["Re"]:
            mu = 1.0 / float(Re)
            for (form,) in params["variants"]:
                sol, ok = R.solve(f"{form}_Re{Re}_N{N}", V, T, info, 1.0 / N, mode="ns", form=form, mu=mu, rho=1.0,
                                  dirichlet=(b, vals), continuation_re=_re_stages(params["continuation"], Re), re=float(Re))
                if not ok:
                    continue
                nodal_defect(R, V, T, sol, lid)
                _cavity_quantities(R, V, T, sol, form, mu, N, Re, f"{form}_Re{Re}",
                                   ("Ekin", "Diss", "Enst", "ux_min_vc", "uy_max_hc", "uy_min_hc"))
    return R.result()


def xcode_th(params, workdir, quick):
    """X2 xcode_gls_vs_taylor_hood: P1/P1 GLS on the REGULARIZED-lid cavity (u_x = 16 x^2 (1 - x)^2) at N = 32, 64, 128
    against the fixed Taylor-Hood N = 128 reference; the deviation ratio checks."""
    R = Run(params, quick)

    def lid(X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        top = np.abs(Y - 1.0) < 1e-12
        return np.stack([np.where(top, 16.0 * X ** 2 * (1.0 - X) ** 2, 0.0), 0.0 * X], axis=-1)

    Ns = list(params["meshes"]["structured"]["N"])
    store = {}
    for N in (Ns[:1] if quick else Ns):
        V, T = C2.structured_mesh(N)
        info = mesh_info(V, T)
        b, vals = dirichlet_of(V, T, lid)
        for Re in params["Re"]:
            mu = 1.0 / float(Re)
            sol, ok = R.solve(f"reg_Re{Re}_N{N}", V, T, info, 1.0 / N, mode="ns", form="sym", mu=mu, rho=1.0,
                              dirichlet=(b, vals), continuation_re=_re_stages([10.0, 30.0, 100.0, 200.0, 400.0, 700.0], Re),
                              re=float(Re))
            if not ok:
                continue
            nodal_defect(R, V, T, sol, lid)
            q = _cavity_quantities(R, V, T, sol, "sym", mu, N, Re, f"gls_Re{Re}",
                                   ("Ekin", "Diss", "ux_min_vc", "uy_max_hc", "uy_min_hc"))
            store[(Re, N)] = q
    ref = params["th_reference_for_ratio_checks"]
    for Re in params["Re"]:
        if (Re, 64) in store and (Re, 128) in store:
            for k in ("Ekin", "Diss", "ux_min_vc", "uy_max_hc", "uy_min_hc"):
                R.acc.q[f"gls_th_deviation_ratio_{k}_Re{Re}"] = C5.th_deviation_ratio(
                    store[(Re, 64)][k], store[(Re, 128)][k], float(ref[f"Re{Re}_{k}"]))
    return R.result()
