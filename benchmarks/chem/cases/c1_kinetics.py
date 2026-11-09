"""Kinetics, reactor and time-integration case runner (setup and solve; every error, adherence, invariant and
effectivity computation is in the check module c1_kinetics_check).

Contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.
Problems are built from the fixed case data (benchmarks/chem/casedata/c1, sha256 checked) and solved with the public
API axionfem.chem.kinetics (C++ core _core.chem: kinetics kernel, reactors, axf::time).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import c1_kinetics_check as C
from axionfem.chem import kinetics as K

REPO = Path(__file__).resolve().parents[3]
NAN = float("nan")


# ------------------------------------------------------------------------------------------------------- data
def _load(spec):
    p = REPO / spec["path"]
    raw = p.read_bytes()
    if spec.get("sha256") and hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise RuntimeError(f"{spec['path']}: sha256 differs from the fixed value")
    return json.loads(raw)


_FIELDS = {}


def _fields(params):
    key = params["fields"]["path"]
    if key not in _FIELDS:
        _FIELDS[key] = _load(params["fields"])
    return _FIELDS[key]


def _num(v):
    """Rational strings of the fixed parameters ('7/10', '[1, 1/5]') to floats."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.startswith("["):
        return [_num(x) for x in s[1:-1].split(",")]
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _fun_t(codes):
    return C.numpy_function(codes)


def _fun_ty(codes, n):
    return C.numpy_function(codes, n)


def _jac(rows, n):
    fs = [C.numpy_function(r, n) for r in rows]
    return lambda t, y: np.array([f(t, y) for f in fs])


_SYMENV = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt, "log": np.log, "pi": np.pi,
           "__builtins__": {}}


def _forcing(fields, name):
    """MMS source s(t) from the fixed forcing expressions (plain arithmetic of sin/cos/exp/sqrt; evaluated with numpy,
    no SymPy needed to run a case)."""
    fs = [eval("lambda t: " + s, _SYMENV) for s in fields["mms_forcing_sympy"][name]]   # noqa: S307 (fixed case data)
    return lambda t: np.array([float(f(t)) for f in fs])


def generic(fields, name):
    """(problem, exact) of a manufactured generic-interface problem (M1..M4, M6, M1_batch) from c1_fields.json."""
    g = fields["generic_problems"][name]
    n = int(g["n"])
    f = _fun_ty(g["rhs_numpy"], n)
    J = _jac(g["jac_numpy"], n)
    return K.problem(n, f, J, mass=g.get("mass")), _fun_t(g["exact_numpy"])


# reactor-API problems; every parameter is read from the fixed case data
def _m4_system(fields, source=None):
    """M4: A <=> B equilibrium (K), B -> C (k), C -> A (k3) from c1_fields.json mechanisms.M4."""
    m = fields["mechanisms"]["M4"]
    kin = {"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"B": 1}, "products": {"C": 1}, "A": _num(m["k"])},
        {"reactants": {"C": 1}, "products": {"A": 1}, "A": _num(m["k3"])}]}
    return K.equilibrium(kin, [{"reactants": {"A": 1}, "products": {"B": 1}, "K": _num(m["K"])}], source=source)


def _m6_mech():
    return _load({"path": "benchmarks/chem/casedata/c1/c1_kernel_ref.json"})["reference"]["M6"]["mechanism"]


def _m6_params(fields):
    """M6 semibatch (q, c_f, V_0) read off the fixed generic M6 problem: V' = q is its last row, and its Jacobian
    column d/dV is -(q/V^2)(c_f - c), so c_f = c - J[:, V] V^2 / q at any state (y = [c, V])."""
    g = fields["generic_problems"]["M6"]
    n = int(g["n"])
    f = _fun_ty(g["rhs_numpy"], n)
    J = _jac(g["jac_numpy"], n)
    y = np.array([0.9, 0.6, 0.4, 2.3])
    q = float(f(0.0, y)[-1])
    cf = y[:-1] - J(0.0, y)[:-1, -1] * y[-1] ** 2 / q
    V0 = float(_fun_t(g["exact_numpy"])(0.0)[-1])
    return q, [float(round(v, 12)) for v in cf], V0


def reactor(fields, name, with_source=True):
    """(problem, exact) of the manufactured problem assembled by the kernel and reactor API."""
    M = fields["mechanisms"]
    src = _forcing(fields, name) if with_source else None
    ex = _fun_t(fields["generic_problems"][name]["exact_numpy"])
    if name == "M1":
        P = K.cstr(K.mechanism(M["M1"]), M["M1_cstr"]["tau"], M["M1_cstr"]["cin"], source=src)
    elif name == "M2":
        P = K.adiabatic_liquid(K.mechanism(M["M2"]), M["M2_rhocp"], source=src)
    elif name == "M3":
        P = K.const_p_gas(K.mechanism(M["M3"]), M["M3_T"], M["M3_P"], source=src)
    elif name == "M4":
        P = _m4_system(fields, src).problem()
    elif name == "M6":
        q, cf, _ = _m6_params(fields)
        P = K.semibatch(K.mechanism(_m6_mech()), q, cf, source=src)
    else:
        raise ValueError(name)
    return P, ex


# ---------------------------------------------------------------------------------------------- run helpers
def _entry(level, dt, tr=None, solver=None, seconds=None, **extra):
    s = K.solver_entry(tr) if tr is not None else solver
    e = C.run_entry(level, dt, s, seconds)
    e.update(extra)
    return e


def _no_solve_entry(label):
    return C.run_entry(label, None, {"converged": True, "relative_residual": 0.0, "iterations": 0}, 0.0)


def _newton(params):
    n = params.get("newton") or {}
    if not isinstance(n, dict):
        n = {}
    return float(n.get("rtol", 1e-12)), int(n.get("max_iters", 50))


def _uniform(t_end, dt):
    return K.uniform_grid(0.0, t_end, dt)


def _fixed(P, method, times, exact, nr, nm, start="exact", y0=None):
    """Fixed-step / prescribed-grid run.  method: bdf1..bdf5 or onestep."""
    y0 = exact(times[0]) if y0 is None else np.asarray(y0, float)
    if method == "onestep":
        return K.grid_onestep(P, times, y0, newton_rtol=nr, newton_max_iters=nm)
    k = int(method[3:])
    ys = [y0] + ([exact(t) for t in times[1:k]] if start == "exact" else [])
    return K.grid_bdf(P, k, times, ys, newton_rtol=nr, newton_max_iters=nm)


def _Y(tr):
    return np.asarray(tr.y)


def _at(tr, t_out):
    """Values AT the fixed output times (stop-time mode: every output time is an accepted step end)."""
    ts = np.asarray(tr.t)
    out = []
    for to in t_out:
        i = int(np.argmin(np.abs(ts - to)))
        if abs(ts[i] - to) > 8 * np.finfo(float).eps * max(1.0, abs(to)):
            raise RuntimeError(f"output time {to} is not an accepted step end")
        out.append(_Y(tr)[i])
    return np.array(out)


def _tag(t):
    return f"{float(t):g}"


# ------------------------------------------------------------------------------------------ build identity
_CORE = {}


def core_identity() -> dict:
    """This process's compiled core: path, sha256, git sha."""
    if not _CORE:
        import os
        from axionfem import _core
        f = os.path.realpath(_core.__file__)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        _CORE.update(core_file=os.path.basename(f), core_sha256=h.hexdigest(),
                     git_sha=json.loads(_core.build_info()).get("git_sha"))
    return dict(_CORE)


class _Out:
    def __init__(self):
        self.ident = core_identity()
        self.runs, self.series, self.q, self.notes = [], {}, {}, []
        self.ose = []      # outputs_at_step_ends per adaptive run

    def adaptive_outputs(self, tr, t_out):
        self.ose.append(C.outputs_at_step_ends(tr.t, t_out))

    def result(self):
        if self.ose:
            self.q["outputs_at_step_ends_min"] = float(np.min(self.ose))
        me = self.ident
        self.notes.append(f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]})")
        return {"runs": self.runs, "series": self.series, "quantities": self.q, "notes": self.notes}


def _scales(params, name):
    return np.asarray(params["scales"][name], float)


def _order_study(o, sname, P, exact, method, dts, t_end, scales, nr, nm, start="exact", err="max"):
    errs = []
    labels = []
    for i, dt in enumerate(dts):
        t0 = time.time()
        tr = _fixed(P, method, _uniform(t_end, dt), exact, nr, nm, start)
        if err == "max":
            e = C.max_norm_error(tr.t, _Y(tr), exact, scales)
        else:   # final time
            e = C.final_time_error(_Y(tr)[-1], exact(tr.t[-1]), scales)
        lab = f"{sname}_L{i}"
        o.runs.append(_entry(lab, dt, tr, seconds=time.time() - t0))
        errs.append(e)
        labels.append(lab)
    o.series[sname] = C.order_series(labels, errs, 1.0)
    return errs


# =============================================================================================================
# C1a
# =============================================================================================================
def time_mms_fixed(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    nr, nm = _newton(params)
    T = float(params["t_end"])
    for name in params["problems"]:
        P, ex = generic(F, name)
        sc = _scales(params, name)
        dl = params["dt_levels"]
        _order_study(o, f"{name}_bdf1", P, ex, "bdf1", dl["bdf1"], T, sc, nr, nm)
        _order_study(o, f"{name}_bdf2_exact", P, ex, "bdf2", dl["bdf2"], T, sc, nr, nm, start="exact")
        _order_study(o, f"{name}_bdf2_self", P, ex, "bdf2", dl["bdf2"], T, sc, nr, nm, start="self")
        _order_study(o, f"{name}_onestep", P, ex, "onestep", dl["onestep"], T, sc, nr, nm)
    o.notes.append("generic interface (numpy RHS/Jacobian from c1_fields.json); BDF2 self start = one BDF1 step")
    return o.result()


def _cyclic(o, sname, P, ex, method, params, sc, nr, nm, k=None):
    g = params["grid"]
    T, pat = float(g["t_end"]), g["step_pattern"]
    errs, labels, adh = [], [], []
    for i, (N, dt) in enumerate(zip(g["N"], params["nominal_dt"])):
        t0 = time.time()
        times = list(C.cyclic_grid_times(T, int(N), pat))
        tr = _fixed(P, method, times, ex, nr, nm, start="exact")
        a = C.grid_adherence(tr.t, T, int(N), pat)
        adh.append(a)
        lab = f"{sname}_L{i}"
        o.runs.append(_entry(lab, dt, tr, seconds=time.time() - t0, dt_max_ratio=a["dt_max_ratio"],
                             h_ratio_max_min=a["h_ratio_max_min"], grid_dev=a["grid_dev"],
                             h_max=float(np.max(np.diff(tr.t))), h_min=float(np.min(np.diff(tr.t)))))
        errs.append(C.max_norm_error(tr.t, _Y(tr), ex, sc))
        labels.append(lab)
    o.series[sname] = C.order_series(labels, errs, 1.0)
    s = C.grid_adherence_series(adh)
    o.q[f"{sname}_grid_dev_max"] = s["grid_dev_max"]
    o.q[f"{sname}_dt_max_ratio_min"] = s["dt_max_ratio_min"]


def time_varstep(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    nr, nm = _newton(params)
    for name in params["problems"]:
        P, ex = generic(F, name)
        sc = _scales(params, name)
        _cyclic(o, f"{name}_cyclic_bdf2", P, ex, "bdf2", params, sc, nr, nm)
        _cyclic(o, f"{name}_cyclic_onestep", P, ex, "onestep", params, sc, nr, nm)
    o.notes.append("prescribed cyclic grids (accepted step ends = grid times); BDF2 variable-coefficient, exact start "
                   "(y at t_0, t_1 from the exact solution)")
    return o.result()


def _pr(fields, lam):
    gm = fields["generic_problems"]["M5"]
    g = C.numpy_function([gm["g_numpy"]])
    dg = C.numpy_function([gm["dg_numpy"]])
    f = lambda t, y: lam * (y - g(t)) + dg(t)
    J = lambda t, y: np.array([[lam]])
    return K.problem(1, f, J), g


def _pr_series(o, sname, P, g, method, dts, t_end, nr, nm):
    gmax = float(np.max(np.abs([g(t)[0] for t in np.linspace(0.0, t_end, 2001)])))
    _order_study(o, sname, P, g, method, dts, t_end, [gmax], nr, nm, start="exact")


def time_prothero_robinson(params, workdir=None, quick=False):
    o = _Out()
    pr = params["pr"]
    F = _load({"path": "benchmarks/chem/casedata/c1/c1_fields.json"})
    T = float(pr["t_end"])
    for regime, lam in pr["lambdas"].items():
        P, g = _pr(F, float(lam))
        for m in ("bdf1", "bdf2", "onestep"):
            _pr_series(o, f"pr_{regime}_{m}", P, g, m, pr["dt_levels"], T, 1e-12, 50)
    o.notes.append("error max_n |y_n - g(t_n)| / max|g| (max over 2001 points of [0, t_end]); exact start")
    return o.result()


def _a3_problem(fields, which="A3"):
    p = fields["parameters"]["A3"]
    if which == "A3":
        k1, k2 = _num(p["k1"]), _num(p["k2"])
    else:
        k1 = k2 = _num(p["kdeg"])
    mech = K.mechanism({"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"A": 1}, "products": {"B": 1}, "A": k1}, {"reactants": {"B": 1}, "products": {"C": 1}, "A": k2}]})
    cf = fields["closed_forms"][which if which == "A3" else "A3_deg"]
    ex = C.numpy_function([cf["A"]["numpy"], cf["B"]["numpy"], cf["C"]["numpy"]])
    return K.batch(mech), ex, [_num(p["A0"]), 0.0, 0.0]


def _robertson(params, form="ode"):
    rp = _load(params["params"]) if "params" in params else None
    rhs = _load(params["rhs"])
    n = len(rhs["rhs_numpy"])
    f = _fun_ty(rhs["rhs_numpy"], n)
    J = _jac(rhs["jac_numpy"], n)
    if form == "dae":
        # algebraic last component: 0 = sum y - 1 (the invariant of the ODE form)
        f0, J0 = f, J
        f = lambda t, y: np.concatenate([f0(t, y)[:-1], [np.sum(y) - 1.0]])
        J = lambda t, y: np.vstack([J0(t, y)[:-1], np.ones(n)])
        return K.problem(n, f, J, mass=[1.0] * (n - 1) + [0.0]), rhs, rp
    return K.problem(n, f, J), rhs, rp


def time_estimator(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    ef = params["effectivity"]
    P, ex = generic(F, ef["problem"])
    ts = float(ef["t_star"])
    for q in (1, 2):
        for tag, h in (("h", float(ef["h"][str(q)])), ("2h", 2 * float(ef["h"][str(q)]))):
            th = [ts - j * h for j in range(q, -1, -1)]
            y, d, nw = K.bdf_step(P, th, [ex(t) for t in th], h, q, newton_rtol=1e-14)
            o.q[f"effectivity_bdf{q}_{tag}"] = C.estimate_ratio(d, y - ex(ts + h))
            o.runs.append(_entry(f"eff_bdf{q}_{tag}", h, solver={"converged": bool(nw["converged"]),
                                                                    "relative_residual": nw["relative_residual"],
                                                                    "iterations": nw["iterations"]}))
    # variable history: steps h/omega^2, h/omega, then the new step h (step ratio omega)
    om, h = float(ef["varhist_step_ratio"]), float(ef["h"]["2"])
    th = [ts - h / om - h / om / om, ts - h / om, ts]
    y, d, nw = K.bdf_step(P, th, [ex(t) for t in th], h, 2, newton_rtol=1e-14)
    o.q["effectivity_bdf2_varhist"] = C.estimate_ratio(d, y - ex(ts + h))
    o.runs.append(_entry("eff_bdf2_varhist", h, solver={"converged": bool(nw["converged"]),
                                                         "relative_residual": nw["relative_residual"],
                                                         "iterations": nw["iterations"]}))
    for tag, h in (("h", float(ef["h"]["onestep"])), ("2h", 2 * float(ef["h"]["onestep"]))):
        d = K.onestep_step(P, ts, ex(ts), h, newton_rtol=1e-14, full=True)
        y, est, adv, nw = np.asarray(d["y"]), np.asarray(d["est"]), bool(d["advances_higher"]), d["newton"]
        # e: the true local error of the LOWER member, rebuilt by the check module from the two members the step
        # produced (embedded minus advanced), independent of the estimate the controller reports
        e_low = C.lower_member_error(y, np.asarray(d["y_embedded"]) - y, ex(ts + h), adv)
        o.q[f"effectivity_onestep_{tag}"] = C.estimate_ratio(est, e_low)
        o.runs.append(_entry(f"eff_onestep_{tag}", h, solver={"converged": bool(nw["converged"]),
                                                               "relative_residual": nw["relative_residual"],
                                                               "iterations": nw["iterations"]}))
    # local-error order of the embedded member alone (h, 2h, 4h): 3 for the declared order-2 embedding
    hs = [float(ef["h"]["onestep"]) * m for m in (1, 2, 4)]
    emb = []
    for h in hs:
        d = K.onestep_step(P, ts, ex(ts), h, newton_rtol=1e-14, full=True)
        emb.append(float(np.max(np.abs((np.asarray(d["y_embedded"]) - ex(ts + h)) / _scales(params, ef["problem"])))))
    o.q["onestep_embedded_local_order"] = C.loglog_slope(hs, emb)
    # tolerance behaviour on A3
    rs = params["rtol_sweep"]
    P3, ex3, y03 = _a3_problem(F)
    Tn = float(rs["t_end"])
    t_out = list(np.linspace(0.0, Tn, int(rs["n_out"]) + 1)[1:])
    X = np.array([ex3(t) for t in t_out])
    for meth in rs["methods"]:
        cls = "bdf2" if meth.startswith("bdf2") else "onestep"
        errs, steps = [], []
        for rtol in rs["rtols"]:
            t0 = time.time()
            tr = K.adaptive(P3, cls, 0.0, y03, t_out, float(rtol), float(rtol))
            o.adaptive_outputs(tr, t_out)
            errs.append(float(np.max(np.abs(_at(tr, t_out) - X))))
            steps.append(int(tr.stats["steps"]))
            o.runs.append(_entry(f"rtol_{cls}_{rtol:g}", None, tr, seconds=time.time() - t0))
        o.series[f"rtol_sweep_{cls}"] = {"tols": [float(r) for r in rs["rtols"]], "error": errs, "steps": steps}
    # forced rejections (Robertson class from h0 = 1)
    rj = params["rejections"]
    rp = _load(rj["params"])
    Pr, _, _ = _robertson({"rhs": {"path": "benchmarks/chem/casedata/c1/robertson_modified_rhs_numpy.json"}})
    for cls in ("bdf2", "onestep"):
        t_out = [float(rj["t_end"])]
        tr = K.adaptive(Pr, cls, 0.0, rp["y0"], t_out, float(rj["rtol"]), rp["abs_floor"], h0=float(rj["h0"]))
        o.adaptive_outputs(tr, t_out)
        o.q[f"rejections_{cls}"] = int(tr.stats["rejected"])
        o.runs.append(_entry(f"rejections_{cls}", None, tr))
    o.notes.append("effectivity from exact histories (newest at t_star); rejections: atol = the Robertson abs_floor")
    return o.result()


def time_dense_events(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    dn = params["dense"]
    tq = [float(t) for t in dn["t_query"]]
    for name in dn["problems"]:
        P, ex = generic(F, name)
        sc = _scales(params, name)
        T = 2.0
        for m in ("bdf1", "bdf2", "onestep"):
            errs, labels = [], []
            for i, dt in enumerate(dn["dt_levels"][m]):
                t0 = time.time()
                tr = _fixed(P, m, _uniform(T, dt), ex, 1e-12, 50, start="exact")
                e = max(float(np.max(np.abs(tr.dense(t) - ex(t)) / sc)) for t in tq)
                lab = f"dense_{name}_{m}_L{i}"
                o.runs.append(_entry(lab, dt, tr, seconds=time.time() - t0))
                errs.append(e)
                labels.append(lab)
            o.series[f"dense_{name}_{m}"] = C.order_series(labels, errs, 1.0)
    ev = params["events"]
    p1 = F["parameters"]["A1"]
    A0, k = _num(p1["A0"]), _num(p1["k"])
    thr = float(ev["A1"]["c_threshold"])
    m1 = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": k}]})
    P1 = K.batch(m1)
    tc = math.log(A0 / thr) / k
    P3, _, y03 = _a3_problem(F)
    tstar = float(C.numpy_function([F["closed_forms"]["A3"]["t_star"]["numpy"]])(0.0)[0])
    for cls, p in (("bdf2", 2), ("onestep", 3)):
        scaled = []
        for rtol in ev["rtols"]:
            rtol = float(rtol)
            for P, y0, T, g, texact in ((P1, [A0, 0.0], 5.0, lambda t, y: y[0] - thr, tc),
                                        (P3, y03, 10.0, lambda t, y, PP=P3: PP.rhs(t, np.asarray(y, float))[1], tstar)):
                tr = K.adaptive(P, cls, 0.0, y0, [T], rtol, 1e-2 * rtol, events=[g])
                o.adaptive_outputs(tr, [T])
                o.runs.append(_entry(f"event_{cls}_{rtol:g}_{len(scaled)}", None, tr))
                if not tr.events:
                    scaled.append(float("inf"))
                    continue
                te = tr.events[0][1]
                scaled.append(abs(te - texact) / texact / rtol ** (p / (p + 1)))
        o.q[f"event_scaled_max_{cls}"] = float(np.max(scaled))
    o.notes.append("dense output at off-grid times on fixed steps (exact start for BDF2); events located by the "
                   "core (Brent on the dense output of the step where g changes sign)")
    return o.result()


def time_lstability(params, workdir=None, quick=False):
    o = _Out()
    lam, delta, dt, n = float(params["lam"]), float(params["delta"]), float(params["dt"]), int(params["n_steps"])
    f = lambda t, y: np.array([lam * (y[0] - math.cos(t)) - math.sin(t)])
    J = lambda t, y: np.array([[lam]])
    P = K.problem(1, f, J)
    ex = lambda t: np.array([math.cos(t)])
    tr = K.grid_bdf(P, 1, [0.0, dt], [[1.0 + delta]])
    o.q["damping_bdf1"] = abs(tr.y[-1][0] - math.cos(dt)) / delta
    o.runs.append(_entry("bdf1", dt, tr))
    times = [i * dt for i in range(n + 2)]
    tr = K.grid_bdf(P, 2, times, [[1.0], [math.cos(dt) + delta]])
    o.q["damping_bdf2"] = abs(tr.y[-1][0] - math.cos(times[-1])) / delta
    o.runs.append(_entry("bdf2", dt, tr))
    tr = K.grid_onestep(P, [0.0, dt], [1.0 + delta])
    o.q["damping_onestep"] = abs(tr.y[-1][0] - math.cos(dt)) / delta
    o.runs.append(_entry("onestep", dt, tr))
    times = [i * dt for i in range(n + 1)]
    tr = K.grid_onestep(P, times, [1.0 + delta], tableau=K.THETA_HALF)
    e = np.array([tr.y[i][0] - math.cos(t) for i, t in enumerate(tr.t)])
    o.q["theta_half_persistence"] = abs(e[-1]) / delta
    o.q["theta_half_sign_alternates"] = int(bool(np.all(np.sign(e[1:-1]) * np.sign(e[2:]) < 0)))
    sha = hashlib.sha256(np.ascontiguousarray(np.asarray(K.THETA_HALF.A, dtype=np.float64)).tobytes()).hexdigest()
    o.q["theta_half_same_driver"] = int(sha == params["theta_half_tableau_sha256"])
    o.q["theta_half_tableau_sha256"] = sha
    o.q["onestep_tableau_sha256"] = hashlib.sha256(
        np.ascontiguousarray(np.asarray(K.ESDIRK32.A, dtype=np.float64)).tobytes()).hexdigest()
    o.runs.append(_entry("theta_half", dt, tr))
    o.notes.append("theta = 1/2 control: the trapezoid tableau through the same one-step driver (grid_onestep)")
    return o.result()


# ---------------------------------------------------------------------------------------- stiff cross-code
def _stiff(params, classes, keyed):
    o = _Out()
    form = params.get("form", "ode")
    P, rhs, rp = _robertson(params, form)
    rp = rp or {}
    t_out = [float(t) for t in rp.get("t_out", rhs.get("t_out"))]
    y0 = rp.get("y0", rhs.get("y0"))
    atol = np.asarray(params["atol"], float)
    names = rp.get("variables", [f"y{i + 1}" for i in range(P.n)])
    nps, mins, drifts = [], [], []
    for cls in classes:
        t0 = time.time()
        tr = K.adaptive(P, cls, float(rp.get("t0", 0.0)), y0, t_out, float(params["rtol"]), atol,
                        h0=float(params.get("h0", 0.0)))
        o.adaptive_outputs(tr, t_out)
        o.runs.append(_entry(cls, None, tr, seconds=time.time() - t0))
        Y = _at(tr, t_out)
        suf = f"__{cls}" if keyed else ""
        for j, t in enumerate(t_out):
            for i, nm in enumerate(names):
                o.q[f"{nm}_t{_tag(t)}{suf}"] = float(Y[j, i])
        s = K.solver_entry(tr)
        nps.append(s["newton_per_solve"])
        mins.append(C.min_scaled(_Y(tr), atol))
        if rp.get("invariant") is not None:
            drifts.append(C.invariant_drift(_Y(tr), [rp["invariant"]], scale=1.0))
    o.q["newton_per_solve_mean"] = float(np.max(nps))
    o.q["min_scaled"] = float(np.min(mins))
    if drifts:
        o.q["invariant_drift"] = float(np.max(drifts))
        o.q["invariant_scale"] = float(abs(np.sum(y0)))
    o.notes.append(f"{form} form; stop-time outputs at the fixed times; methods {classes}")
    return o.result()


def stiff_robertson_modified(params, workdir=None, quick=False):
    return _stiff(params, ["bdf2", "onestep"], True)


def stiff_robertson_dae_modified(params, workdir=None, quick=False):
    return _stiff(params, ["bdf2", "onestep"], True)


def stiff_vdp_modified(params, workdir=None, quick=False):
    return _stiff(params, ["bdf2", "onestep"], True)


def vo_stiff_robertson_modified(params, workdir=None, quick=False):
    return _stiff(params, ["vsvo"], False)


def vo_stiff_robertson_dae_modified(params, workdir=None, quick=False):
    return _stiff(params, ["vsvo"], False)


def vo_stiff_vdp_modified(params, workdir=None, quick=False):
    return _stiff(params, ["vsvo"], False)


# =============================================================================================================
# C1b
# =============================================================================================================
def kernel_rates(params, workdir=None, quick=False):
    o = _Out()
    ref = _load(params["reference"])["reference"]
    jdev, tdev = 0.0, 0.0
    for name, r in ref.items():
        mech = K.mechanism(r["mechanism"])
        w, J, dT = K.rates(mech, r["c"], r["T"])
        for i, s in enumerate(r["mechanism"]["species"]):
            o.q[f"omega_{name}_{s}"] = float(w[i])
        Jr = np.asarray(r["domega_dc"], float)
        jdev = max(jdev, float(np.max(np.abs(J - Jr)) / np.max(np.abs(Jr))))
        if r.get("domega_dT") is not None:
            Tr = np.asarray(r["domega_dT"], float)
            tdev = max(tdev, float(np.max(np.abs(dT - Tr)) / np.max(np.abs(Tr))))
    o.q["jacobian_max_rel_dev"] = jdev
    o.q["dRdT_max_rel_dev"] = tdev
    # refusals (allow_nonequilibrium_cycle = false, the default)
    bad = json.loads(json.dumps(ref["A11"]["mechanism"]))
    bad["reactions"][2]["Kc"] = 2.0 * float(bad["reactions"][2]["Kc"])
    tests = {
        "refused_inconsistent_cycle": lambda: K.mechanism(bad),
        "refused_orders_on_reversible": lambda: K.mechanism({"species": ["A", "B"], "reactions": [
            {"reactants": {"A": 1}, "products": {"B": 1}, "A": 1.0, "reversible": True, "Kc": 2.0,
             "orders": {"A": 0.5}}]}),
        "refused_kc_dH_nonisothermal": lambda: K.adiabatic_liquid(K.mechanism({"species": ["A", "B"], "reactions": [
            {"reactants": {"A": 1}, "products": {"B": 1}, "A": 1.0, "Ea": 1000.0, "reversible": True, "Kc": 2.0,
             "dH": -1e4}]}), 4e6),
    }
    for qn, fn in tests.items():
        try:
            fn()
            o.q[qn] = 0
            o.notes.append(f"{qn}: NOT refused")
        except ValueError as e:
            o.q[qn] = 1
            o.notes.append(f"{qn}: {e}")
    o.runs.append(_no_solve_entry("kernel"))
    return o.result()


def kernel_units(params, workdir=None, quick=False):
    o = _Out()
    var = _load(params["variants"])["variants"]
    for name, v in var.items():
        mech = K.mechanism(v["mechanism"])
        c = mech.concentration_to_si(np.asarray(v["c"], float))
        w, _, _ = K.rates(mech, c, v["T"])
        for i, s in enumerate(v["mechanism"]["species"]):
            o.q[f"omegaSI_{name}_{s}"] = float(w[i])
    o.runs.append(_no_solve_entry("kernel_units"))
    o.notes.append("the kernel converts the declared units; concentrations through Mechanism.concentration_to_si")
    return o.result()


def jacobian_taylor(params, workdir=None, quick=False):
    o = _Out()
    F = _load({"path": "benchmarks/chem/casedata/c1/c1_fields.json"})
    M = F["mechanisms"]
    st = params["states"]
    eps = [float(e) for e in params["eps"]]
    rng = np.random.default_rng(int(params["seed"]))
    a10 = F["parameters"]["A10"]
    probs = {
        "M1": K.cstr(K.mechanism(M["M1"]), M["M1_cstr"]["tau"], M["M1_cstr"]["cin"]),
        "M2": K.adiabatic_liquid(K.mechanism(M["M2"]), M["M2_rhocp"]),
        "M3": K.const_p_gas(K.mechanism(M["M3"]), M["M3_T"], M["M3_P"]),
        "A10": K.pfr(K.mechanism({"species": ["A", "B"], "reactions": [
            {"reactants": {"A": 1}, "products": {"B": 2}, "A": _num(a10["k"])}]}), None, _num(a10["v0"]),
            _num(a10["FA0"])),
        "M6": K.semibatch(K.mechanism(_m6_mech()), *_m6_params(F)[:2]),
    }
    for name, P in probs.items():
        x0 = np.asarray(st[name], float)
        Ff = lambda x, P=P: P.rhs(0.0, np.asarray(x, float))
        jac_f = lambda x, P=P: P.jacobian(0.0, np.asarray(x, float))
        v = rng.standard_normal(len(x0)) * np.abs(x0)
        _, s = C.taylor_remainders(Ff, jac_f, x0, v, eps)
        o.q[f"taylor_slope_{name}" if name != "M2" else "taylor_slope_M2_cT"] = s
        if name == "M2":
            vT = np.zeros(len(x0))
            vT[-1] = x0[-1]
            _, sT = C.taylor_remainders(Ff, jac_f, x0, vT, eps)
            o.q["taylor_slope_M2_T"] = sT
    o.runs.append(_no_solve_entry("taylor"))
    o.notes.append("direction v = N(0,1) x |x0| per component (seed fixed); M2 T-only: v = T e_T")
    return o.result()


# =============================================================================================================
# C1c
# =============================================================================================================
def mms_reactors(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    nr, nm = _newton(params)
    T = float(params["t_end"])
    for name in params["problems"]:
        P, ex = reactor(F, name)
        sc = _scales(params, name)
        dl = params["dt_levels"]
        _order_study(o, f"reactor_{name}_bdf2_self", P, ex, "bdf2", dl["bdf2"], T, sc, nr, nm, start="self")
        _order_study(o, f"reactor_{name}_onestep", P, ex, "onestep", dl["onestep"], T, sc, nr, nm)
    o.notes.append("M1 CSTR, M2 liquid adiabatic, M3 constant-P gas, M4 invariant-form equilibrium, M6 semibatch, "
                   "assembled by the kernel and reactor API with the additive source term for manufactured solutions")
    return o.result()


def _adaptive_cmp(o, P, y0, t_out, rtol, atol, prefix_vals, cls, extra_runs_label):
    t0 = time.time()
    tr = K.adaptive(P, cls, 0.0, y0, t_out, rtol, atol)
    o.adaptive_outputs(tr, t_out)
    o.runs.append(_entry(f"{extra_runs_label}_{cls}", None, tr, seconds=time.time() - t0))
    return tr, _at(tr, t_out)


def _final_order(o, sname, P, ex, method, dts, scales, y0=None):
    T = 16 * float(dts[0])
    errs, labels = [], []
    for i, dt in enumerate(dts):
        t0 = time.time()
        tr = _fixed(P, method, _uniform(T, dt), ex, 1e-12, 50, start="exact", y0=y0)
        errs.append(C.final_time_error(_Y(tr)[-1], ex(tr.t[-1]), scales))
        lab = f"{sname}_L{i}"
        o.runs.append(_entry(lab, dt, tr, seconds=time.time() - t0))
        labels.append(lab)
    o.series[sname] = C.order_series(labels, errs, 1.0)


def _cf(F, key, comps):
    cf = F["closed_forms"][key]
    return C.numpy_function([cf[c]["numpy"] for c in comps])


def analytic_batch_kinetics(params, workdir=None, quick=False):
    o = _Out()
    F = _load({"path": "benchmarks/chem/casedata/c1/c1_fields.json"})
    cs = params["cases"]
    ad = params["adaptive"]
    rtol = float(ad["rtol"])
    sc = params["scales"]
    mk = lambda rx, sp: K.mechanism({"species": sp, "reactions": rx})
    A1, A2, A2m, A3, A4, A11 = (cs[k] for k in ("A1", "A2", "A2m", "A3", "A4", "A11"))
    probs = {
        "A1": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A1["k"])}], ["A", "B"]),
               [_num(A1["A0"]), 0.0], _cf(F, "A1", "AB"), [0.5, 2.0, 5.0]),
        "A2_kr": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A2["kf"]), "reversible": True,
                       "kr": {"A": _num(A2["kr"])}}], ["A", "B"]), [_num(A2["A0"]), 0.0], _cf(F, "A2_kr", "AB"),
                  [0.5, 2.0, 5.0]),
        "A2_Kc": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A2["kf"]), "reversible": True,
                       "Kc": _num(A2["Kc"])}], ["A", "B"]), [_num(A2["A0"]), 0.0], _cf(F, "A2_Kc", "AB"),
                  [0.5, 2.0, 5.0]),
        "A2m": (mk([{"reactants": {"A": 1}, "products": {"B": 2}, "A": _num(A2m["kf"]), "reversible": True,
                     "Kc": _num(A2m["Kc"])}], ["A", "B"]), [_num(A2m["A0"]), 0.0], _cf(F, "A2m", "AB"),
                [0.5, 2.0, 5.0, 40.0]),
        "A11": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A11["k1"]), "reversible": True,
                     "Kc": _num(A11["K1"])},
                    {"reactants": {"B": 1}, "products": {"C": 1}, "A": _num(A11["k2"]), "reversible": True,
                     "Kc": _num(A11["K2"])},
                    {"reactants": {"C": 1}, "products": {"A": 1}, "A": _num(A11["k3"]), "reversible": True,
                     "Kc": 1.0 / (_num(A11["K1"]) * _num(A11["K2"]))}], ["A", "B", "C"]),
                _num(A11["c0"]), None, [0.5, 2.0, 40.0]),
        "A3": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A3["k1"])},
                   {"reactants": {"B": 1}, "products": {"C": 1}, "A": _num(A3["k2"])}], ["A", "B", "C"]),
               [_num(A3["A0"]), 0.0, 0.0], _cf(F, "A3", "ABC"), [1.0, 3.0, 10.0]),
        "A3_deg": (mk([{"reactants": {"A": 1}, "products": {"B": 1}, "A": _num(A3["kdeg"])},
                       {"reactants": {"B": 1}, "products": {"C": 1}, "A": _num(A3["kdeg"])}], ["A", "B", "C"]),
                   [_num(A3["A0"]), 0.0, 0.0], _cf(F, "A3_deg", "ABC"), [1.0, 3.0, 10.0]),
        "A4": (mk([{"reactants": {"A": 1, "B": 1}, "products": {"C": 1}, "A": _num(A4["k"])}], ["A", "B", "C"]),
               [_num(A4["A0"]), _num(A4["B0"]), 0.0], _cf(F, "A4", "ABC"), [0.5, 2.0, 5.0]),
    }
    Kmat = np.array([[_num(x) for x in r] for r in F["A11_rate_matrix"]])
    c11 = np.array(_num(A11["c0"]))
    probs["A11"] = probs["A11"][:2] + (lambda t: C.linear_cycle_exact(Kmat, c11, t),) + probs["A11"][3:]
    ex3 = {}
    for name, (mech, y0, ex, t_out) in probs.items():
        P = K.batch(mech)
        S = np.asarray(sc[name], float)
        atol = 1e-10 * S
        comps = "ABC"[:len(y0)]
        dts = params["dt_levels"][name]
        _final_order(o, f"{name}_bdf2_order", P, ex, "bdf2", dts, S, y0=y0)
        _final_order(o, f"{name}_onestep_order", P, ex, "onestep", dts, S, y0=y0)
        key = name.replace("A3_deg", "A3deg")
        for cls in ("bdf2", "onestep"):
            tr, Y = _adaptive_cmp(o, P, y0, t_out, rtol, atol, None, cls, name)
            for j, t in enumerate(t_out):
                for i, c in enumerate(comps):
                    o.q[f"{name}_{c}_t{_tag(t)}__{cls}"] = float(Y[j, i])
            if name in ("A3", "A3_deg"):
                rhs = lambda t, y, P=P: P.rhs(t, np.asarray(y, float))
                ts_, bm = C.extremum_by_rhs_root(tr.dense, rhs, 1, 0.2, 5.0)
                o.q[f"{key}_t_star__{cls}"] = ts_
                if name == "A3":
                    o.q[f"A3_B_max__{cls}"] = bm
            if name in ("A3", "A4") and cls == "bdf2":
                ex3[name] = tr
            if name == "A3":
                W = [[1, 1, 1]]
            elif name == "A4":
                W = [[1, 0, 1], [0, 1, 1]]
            else:
                W = None
            if W is not None:
                Y_all = _Y(tr)
                d = float(np.max(np.abs((Y_all - Y_all[0]) @ np.asarray(W, float).T)))
                o.q[f"{name}_invariant_drift_abs"] = max(o.q.get(f"{name}_invariant_drift_abs", 0.0), d)
                o.q[f"{name}_invariant_scale"] = float(np.max(np.abs(np.asarray(W, float) @ Y_all[0])))
                o.q[f"{name}_min_scaled"] = min(o.q.get(f"{name}_min_scaled", np.inf), C.min_scaled(Y_all, atol))
            if name == "A2m" and cls == "onestep":
                yA, yB = Y[-1]
                o.q["A2m_Q_over_K_minus_1"] = yB ** 2 / yA / _num(A2m["Kc"]) - 1.0
            if name == "A11" and cls == "onestep":
                a, b, c = Y[-1]
                o.q["A11_ratio1_minus_1"] = b / (a * _num(A11["K1"])) - 1.0
                o.q["A11_ratio2_minus_1"] = c / (b * _num(A11["K2"])) - 1.0
    o.notes.append("fixed-step orders on [0, T], T = 16 x the coarsest dt (T/16 .. T/256), exact start, error at T; "
                   "adaptive values at stop-time outputs; t* = root of f_B on the dense output (bracket [0.2, 5]); "
                   "approach-to-K and cycle ratios from the one-step run at t = 40")
    return o.result()


def analytic_reactors(params, workdir=None, quick=False):
    o = _Out()
    F = _load({"path": "benchmarks/chem/casedata/c1/c1_fields.json"})
    cs = params["cases"]
    rtol = float(params["adaptive"]["rtol"])
    sc = params["scales"]
    t_out = [0.5, 2.0, 5.0]
    # A6: CSTR, feed of A only, A -> B
    a6 = cs["A6"]
    m6 = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1},
                                                             "A": _num(a6["k"])}]})
    P6 = K.cstr(m6, _num(a6["tau"]), [_num(a6["cin"]), 0.0])
    y6 = [_num(a6["c0"]), 0.0]
    ex6 = _cf(F, "A6", "AB")
    # A7: semibatch, y = [c_A, c_B, V]
    a7 = cs["A7"]
    m7 = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1},
                                                             "A": _num(a7["k"])}]})
    P7 = K.semibatch(m7, _num(a7["q"]), [_num(a7["cf"]), 0.0])
    y7 = [_num(a7["c0"]), 0.0, _num(a7["V0"])]
    ex7 = _cf(F, "A7", ["A", "B", "V"])
    # A8: constant-P gas A -> 2B in moles, V0 = 1 m^3 (N0 = c0 V0); compared in concentrations
    a8 = cs["A8"]
    m8 = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 2},
                                                             "A": _num(a8["k"])}]})
    T8, P8_ = _num(a8["T"]), _num(a8["P"])
    N8 = np.array([_num(a8["cA0"]), 0.0])
    P8 = K.const_p_gas(m8, T8, P8_)
    RTP = K.GAS_CONSTANT * T8 / P8_
    V80 = N8.sum() * RTP
    exc8 = _cf(F, "A8", "AB")
    vrel8 = _cf(F, "A8", ["V_rel"])
    ex8N = lambda t: exc8(t) * vrel8(t)[0] * V80     # moles = c V

    def conc8(Y):
        Y = np.atleast_2d(Y)
        V = np.array([K.gas_volume(T8, P8_, y) for y in Y])     # the reactor's own volume
        return Y / V[:, None], V / V80

    for name, P, y0, ex, S in (("A6", P6, y6, ex6, sc["A6"]), ("A7", P7, y7, ex7, sc["A7"]),
                               ("A8", P8, N8, ex8N, None)):
        dts = params["dt_levels"][name]
        if name == "A8":
            T = 16 * float(dts[0])
            for m in ("bdf2", "onestep"):
                errs, labels = [], []
                for i, dt in enumerate(dts):
                    tr = _fixed(P, m, _uniform(T, dt), ex, 1e-12, 50, start="exact")
                    c, _ = conc8(_Y(tr)[-1])
                    errs.append(C.final_time_error(c[0], exc8(T), sc["A8"]))
                    lab = f"A8_{m}_order_L{i}"
                    o.runs.append(_entry(lab, dt, tr))
                    labels.append(lab)
                o.series[f"A8_{m}_order"] = C.order_series(labels, errs, 1.0)
        else:
            for m in ("bdf2", "onestep"):
                _final_order(o, f"{name}_{m}_order", P, ex, m, dts, np.asarray(S, float), y0=y0)
        Sabs = np.asarray(sc[name], float) if name != "A8" else np.array([_num(a8["cA0"]), 2 * _num(a8["cA0"])])
        atol = 1e-10 * (Sabs if name != "A8" else Sabs * V80)
        for cls in ("bdf2", "onestep"):
            tr, Y = _adaptive_cmp(o, P, y0, t_out, rtol, atol, None, cls, name)
            if name == "A8":
                c, vr = conc8(Y)
                for j, t in enumerate(t_out):
                    o.q[f"A8_A_t{_tag(t)}__{cls}"] = float(c[j, 0])
                    o.q[f"A8_B_t{_tag(t)}__{cls}"] = float(c[j, 1])
                o.q[f"A8_V_rel_t2__{cls}"] = float(vr[1])
            else:
                for j, t in enumerate(t_out):
                    o.q[f"{name}_A_t{_tag(t)}__{cls}"] = float(Y[j, 0])
                    o.q[f"{name}_B_t{_tag(t)}__{cls}"] = float(Y[j, 1])
    # A6 steady states: steady mode (Newton on the reactor right-hand side)
    o.q["A6_ss_first"] = float(_steady(P6, [_num(a6["cin"]), 0.0])[0])
    m6b = K.mechanism({"species": ["A"], "reactions": [{"reactants": {"A": 1}, "products": {}, "A": _num(a6["k2"]),
                                                        "orders": {"A": 2}}]})
    o.q["A6_ss_second"] = float(_steady(K.cstr(m6b, _num(a6["tau"]), [_num(a6["cin"])]), [_num(a6["cin"])])[0])
    o.runs.append(_no_solve_entry("A6_steady"))
    # A10: PFR in reactor volume, X = 1 - F_A/F_A0
    a10 = cs["A10"]
    k10, FA0, v0 = _num(a10["k"]), _num(a10["FA0"]), _num(a10["v0"])
    m10 = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 2}, "A": k10}]})
    P10 = K.pfr(m10, None, v0, FA0)
    X = lambda V: C.pfr_conversion(V, v0, k10)
    ex10 = lambda V: np.array([FA0 * (1 - X(V)), 2 * FA0 * X(V)])
    dts = params["dt_levels"]["A10"]
    T10 = 16 * float(dts[0])
    for m in ("bdf2", "onestep"):
        errs, labels = [], []
        for i, dt in enumerate(dts):
            tr = _fixed(P10, m, _uniform(T10, dt), ex10, 1e-12, 50, start="exact")
            errs.append(abs((1 - _Y(tr)[-1][0] / FA0) - X(T10)))
            lab = f"A10_{m}_order_L{i}"
            o.runs.append(_entry(lab, dt, tr))
            labels.append(lab)
        o.series[f"A10_{m}_order"] = C.order_series(labels, errs, 1.0)
    vo = [0.02, 0.05, 0.1, 0.2]
    for cls in ("bdf2", "onestep"):
        tr, Y = _adaptive_cmp(o, P10, [FA0, 0.0], vo, rtol, 1e-12, None, cls, "A10")
        for j, V in enumerate(vo):
            o.q[f"A10_X_V{_tag(V)}__{cls}"] = float(1 - Y[j, 0] / FA0)
    o.notes.append("A8 in moles with V0 = 1 m^3, compared as c = N/V; A6 steady states by Newton on the CSTR "
                   "right-hand side (steady mode); A10 independent variable = reactor volume; fixed-step orders on "
                   "[0, 16 dt_0], exact start, error at the end")
    return o.result()


def _steady(P, c0, tol=1e-15):
    """Steady mode: Newton on the reactor right-hand side; refuses when it does not converge."""
    c = np.asarray(c0, float)
    for _ in range(100):
        r = P.rhs(0.0, c)
        d = np.linalg.solve(P.jacobian(0.0, c), -r)
        c = c + d
        if np.max(np.abs(d)) <= tol * max(1.0, np.max(np.abs(c))):
            r = P.rhs(0.0, c)
            if np.all(np.isfinite(c)) and np.max(np.abs(r)) <= 1e-12 * max(1.0, np.max(np.abs(c))):
                return c
            break
    raise RuntimeError(f"steady state: Newton did not converge from {list(c0)} (last state {list(c)})")


def analytic_adiabatic(params, workdir=None, quick=False):
    o = _Out()
    p = params["case"]
    A, Ea, dH, rcp, A0, T0 = (_num(p[k]) for k in ("A", "Ea", "dH", "rhocp", "A0", "T0"))
    mech = K.mechanism({"species": ["A", "B"], "reactions": [{"reactants": {"A": 1}, "products": {"B": 1}, "A": A,
                                                               "Ea": Ea, "dH": dH}]})
    P = K.adiabatic_liquid(mech, rcp)
    dTad = -dH * A0 / rcp
    t_out = [float(t) for t in params["t_out"]]
    ad = params["adaptive"]
    drift, scale = 0.0, 0.0
    for cls in ("bdf2", "onestep"):
        tr, Y = _adaptive_cmp(o, P, [A0, 0.0, T0], t_out, float(ad["rtol"]), np.asarray(ad["atol"], float), None,
                              cls, "A5")
        for j, t in enumerate(t_out):
            o.q[f"A5_X_t{_tag(t)}__{cls}"] = float(1 - Y[j, 0] / A0)
        Ya = _Y(tr)
        inv = (Ya[:, 2] - T0) - dTad * (1 - Ya[:, 0] / A0)
        drift = max(drift, float(np.max(np.abs(inv))))
        scale = max(scale, float(np.max(np.abs(Ya[:, 2]))))
    o.q["A5_invariant_abs"] = drift
    o.q["A5_invariant_scale"] = scale
    o.notes.append("state [c_A, c_B, T]; X = 1 - c_A/c_A0; invariant (T - T0) - dT_ad X over every accepted step; "
                   "its scale = max T")
    return o.result()


def analytic_dae(params, workdir=None, quick=False):
    o = _Out()
    F = _load({"path": "benchmarks/chem/casedata/c1/c1_fields.json"})
    p = params["case"]
    Kq, k, A0, B0 = (_num(p[x]) for x in ("K", "k", "A0", "B0"))
    sysm = K.equilibrium({"species": ["A", "B", "C"], "reactions": [
        {"reactants": {"B": 1}, "products": {"C": 1}, "A": k}]},
        [{"reactants": {"A": 1}, "products": {"B": 1}, "K": Kq}])
    P = sysm.problem()
    c0 = sysm.consistent_init(np.array([A0, B0, 0.0]))
    o.q["A9_A_init"], o.q["A9_B_init"] = float(c0[0]), float(c0[1])
    ex = _cf(F, "A9", "ABC")
    s0 = A0 + B0
    dts = params["dt_levels"]
    T = 16 * float(dts[0])
    for m in ("bdf1", "bdf2", "onestep"):
        errs, labels = [], []
        for i, dt in enumerate(dts):
            tr = _fixed(P, m, _uniform(T, dt), ex, 1e-12, 50, start="exact", y0=c0)
            errs.append(C.final_time_error(_Y(tr)[-1], ex(T), [s0] * 3))
            lab = f"A9_{m}_order_L{i}"
            o.runs.append(_entry(lab, dt, tr))
            labels.append(lab)
        o.series[f"A9_{m}_order"] = C.order_series(labels, errs, 1.0)
    ad = params["adaptive"]
    t_out = [1.0, 2.0, 4.0]
    cres = 0.0
    for cls in ("bdf2", "onestep"):
        tr, Y = _adaptive_cmp(o, P, c0, t_out, float(ad["rtol"]), float(ad["atol"]), None, cls, "A9")
        for j, t in enumerate(t_out):
            for i, c in enumerate("ABC"):
                o.q[f"A9_{c}_t{_tag(t)}__{cls}"] = float(Y[j, i])
        for y in _Y(tr):
            cres = max(cres, abs(float(sysm.constraint(y)[0])) / s0)
    o.q["A9_constraint_residual_max"] = cres
    o.notes.append("consistent initialization keeps W c (s = c_A + c_B) and solves the equilibrium; the fixed-step "
                   "orders start from the projected state; error / s0 at T = 16 dt_0")
    return o.result()


def conservation_network(params, workdir=None, quick=False):
    o = _Out()
    nw = params["network"]
    mech = K.mechanism(nw["mechanism"])
    P = K.batch(mech)
    E = np.array([[nw["element_counts"][s] for s in nw["mechanism"]["species"]]], float)
    c0 = np.asarray(nw["c0"], float)
    T = float(nw["t_end"])
    fx = nw["fixed"]
    tr = _fixed(P, fx["method"], _uniform(T, float(fx["dt"])), None, 1e-10, 25, start="self", y0=c0)
    Y = _Y(tr)
    o.q["element_drift_fixed_abs"] = float(np.max(np.abs((Y - c0) @ E.T)))
    o.q["element_scale"] = float(abs((E @ c0)[0]))
    o.runs.append(_entry("network_fixed", float(fx["dt"]), tr))
    ad = nw["adaptive"]
    for meth in ad["methods"]:
        cls = "bdf2" if meth.startswith("bdf2") else "onestep"
        tr = K.adaptive(P, cls, 0.0, c0, [T], float(ad["rtol"]), float(ad["atol"]))
        o.adaptive_outputs(tr, [T])
        Y = _Y(tr)
        o.q[f"element_drift_adaptive_{cls}_abs"] = float(np.max(np.abs((Y - c0) @ E.T)))
        o.runs.append(_entry(f"network_{cls}", None, tr))
    ps = params["positivity"]
    mp = K.mechanism(ps["mechanism"])
    Pp = K.batch(mp)
    p0 = np.asarray(ps["c0"], float)
    drift, mins = 0.0, np.inf
    for meth in ps["methods"]:
        cls = "bdf2" if meth.startswith("bdf2") else "onestep"
        for rtol in ps["rtols"]:
            atol = 1e-3 * float(rtol)
            tr = K.adaptive(Pp, cls, 0.0, p0, [float(ps["t_end"])], float(rtol), atol, h0=float(ps["h0"]))
            o.adaptive_outputs(tr, [float(ps["t_end"])])
            Y = _Y(tr)
            drift = max(drift, float(np.max(np.abs(Y.sum(axis=1) - p0.sum()))))
            mins = min(mins, C.min_scaled(Y, np.full(3, atol)))
            o.runs.append(_entry(f"positivity_{cls}_{float(rtol):g}", None, tr))
    o.q["positivity_sum_drift_max"] = drift
    o.q["positivity_sum_scale"] = float(p0.sum())
    o.q["positivity_min_scaled"] = float(mins)
    o.notes.append("production Newton defaults (relative residual 1e-10); fixed BDF2 self-started; no clipping")
    return o.result()


def cross_cantera(params, workdir=None, quick=False):
    o = _Out()
    mech = K.mechanism(params["mechanism"])
    sp = params["mechanism"]["species"]
    c0 = np.asarray(params["c0"], float)
    T = float(params["T"])
    t_out = [float(t) for t in params["t_out"]]
    ad = params["adaptive"]
    Pg = K.GAS_CONSTANT * T * c0.sum()          # the pressure of the initial state (ideal gas)
    for r in params["reactors"]:
        for cls in ("bdf2", "onestep"):
            t0 = time.time()
            if r == "const_v":
                P = K.batch(mech, T=T)
                tr = K.adaptive(P, cls, 0.0, c0, t_out, float(ad["rtol"]), float(ad["atol"]))
                Y = _at(tr, t_out)
            else:
                P = K.const_p_gas(mech, T, Pg)         # moles with V0 = 1 m^3
                tr = K.adaptive(P, cls, 0.0, c0, t_out, float(ad["rtol"]), float(ad["atol"]))
                N = _at(tr, t_out)
                Y = N / np.array([K.gas_volume(T, Pg, n_) for n_ in N])[:, None]   # the reactor's own volume
            o.adaptive_outputs(tr, t_out)
            o.runs.append(_entry(f"{r}_{cls}", None, tr, seconds=time.time() - t0))
            for j, t in enumerate(t_out):
                for i, s in enumerate(sp):
                    o.q[f"{r}_{s}_t{_tag(t)}__{cls}"] = float(Y[j, i])
    o.notes.append("isothermal 600 K; constant P = c_tot,0 R T (moles, V0 = 1 m^3), compared as concentrations")
    return o.result()


# =============================================================================================================
# C1d (own BDF3-5)
# =============================================================================================================
def vo_bdf345_fixed(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    T = float(params["t_end"])
    for name in params["problems"]:
        P, ex = generic(F, name)
        sc = _scales(params, name)
        for k in (3, 4, 5):
            if k == 5 and name in ("M2", "M4"):
                continue
            for start in ("exact", "self"):
                _order_study(o, f"{name}_bdf{k}_{start}", P, ex, f"bdf{k}", params["dt_levels"][f"bdf{k}"], T, sc,
                             1e-13, 50, start=start)
    o.notes.append("BDF3-5 constant step; self start = ESDIRK3(2) with one Richardson extrapolation per starting "
                   "step (local error O(h^5)); BDF5 on M2, M4 not run (reported-only in the design)")
    return o.result()


def vo_own_bdf345(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    for name in ("M1", "M2", "M4", "M6"):
        P, ex = generic(F, name)
        sc = _scales(params, name)
        for k in (3, 4):
            _cyclic(o, f"{name}_cyclic_bdf{k}", P, ex, f"bdf{k}", params, sc, 1e-13, 50)
    pr = params["pr"]
    P, g = _pr(F, float(pr["lambda"]))
    for k in (3, 4, 5):
        _pr_series(o, f"pr_stiff_real_bdf{k}", P, g, f"bdf{k}", pr["dt_levels"], float(pr["t_end"]), 1e-13, 50)
    # complex-spectrum control: y' = A y, eigenvalues r(-cos phi +- i sin phi), h r = hr
    cs = params["cspec"]
    r, phi = float(cs["r"]), math.radians(float(cs["phi_deg"]))
    a, b = -r * math.cos(phi), r * math.sin(phi)
    Am = np.array([[a, b], [-b, a]])
    Pc = K.problem(2, lambda t, y: Am @ y, lambda t, y: Am)
    h = float(cs["hr"]) / r
    n = int(cs["n_steps"])
    delta = float(cs["delta"])
    for k in (1, 2, 3, 4, 5):
        times = [i * h for i in range(n + 1)]
        # exact history of the decaying solution plus a perturbation delta in the newest start value
        from scipy.linalg import expm
        y0 = np.array([1.0, 0.0])
        ys = [expm(Am * t) @ y0 for t in times[:k]]
        tr0 = K.grid_bdf(Pc, k, times, ys, newton_rtol=1e-14)
        ysd = [y.copy() for y in ys]
        ysd[-1] = ysd[-1] + delta * np.array([1.0, 0.0])
        tr = K.grid_bdf(Pc, k, times, ysd, newton_rtol=1e-14)
        # growth of the perturbation itself: perturbed minus unperturbed run from the same exact history
        e_end = np.linalg.norm(_Y(tr)[-1] - _Y(tr0)[-1])
        o.q[f"cspec_bdf{k}_amplification"] = float(e_end / delta)
        o.runs.append(_entry(f"cspec_bdf{k}", h, tr))
    ef = params["effectivity"]
    Pm, exm = generic(F, ef["problem"])
    ts = float(ef["t_star"])
    for k in (3, 4, 5):
        for tag, hh in (("h", float(ef["h"][str(k)])), ("2h", 2 * float(ef["h"][str(k)]))):
            th = [ts - j * hh for j in range(k, -1, -1)]
            y, d, nw = K.bdf_step(Pm, th, [exm(t) for t in th], hh, k, newton_rtol=1e-14)
            o.q[f"effectivity_bdf{k}_{tag}"] = C.estimate_ratio(d, y - exm(ts + hh))
            o.runs.append(_entry(f"eff_bdf{k}_{tag}", hh, solver={"converged": bool(nw["converged"]),
                                                                   "relative_residual": nw["relative_residual"],
                                                                   "iterations": nw["iterations"]}))
    o.notes.append("complex-spectrum control: |y_n(perturbed) - y_n(unperturbed)|/delta after n_steps, the perturbation "
                   "delta added to the newest exact start value (amplification > 1: growth)")
    return o.result()


def vo_rtol_sweep(params, workdir=None, quick=False):
    o = _Out()
    F = _fields(params)
    rs = params["rtol_sweep"]
    P3, ex3, y03 = _a3_problem(F)
    Tn = float(rs["t_end"])
    t_out = list(np.linspace(0.0, Tn, int(rs["n_out"]) + 1)[1:])
    X = np.array([ex3(t) for t in t_out])
    errs, steps, drift = [], [], 0.0
    for rtol in rs["rtols"]:
        tr = K.adaptive(P3, "vsvo", 0.0, y03, t_out, float(rtol), float(rtol))
        o.adaptive_outputs(tr, t_out)
        errs.append(float(np.max(np.abs(_at(tr, t_out) - X))))
        steps.append(int(tr.stats["steps"]))
        Y = _Y(tr)
        drift = max(drift, float(np.max(np.abs(Y.sum(axis=1) - Y[0].sum())) / Y[0].sum()))
        o.runs.append(_entry(f"vsvo_{float(rtol):g}", None, tr))
    o.series["rtol_sweep_vsvo"] = {"tols": [float(r) for r in rs["rtols"]], "error": errs, "steps": steps}
    o.q["A3_invariant_drift_rel"] = drift
    return o.result()
