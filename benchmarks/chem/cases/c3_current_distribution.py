"""C3a case runner (set-up and solve; the error computation is in the check module c3_checks).

Contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.
Every function reads only its fixed ``params`` (run.parameters of its case) and the fixed case data under
benchmarks/chem/casedata (sha256 checked by c3_checks.load_json).  The solver is axionfem.chem.CurrentDistribution
(C++ axf::chem::CurrentDistribution).  Every current, balance, measure, error norm, Taylor statistic and hash is
computed by c3_checks from the solver's RAW data (quadrature points, weights without r, field values and gradients,
the full residual vector, mesh vertices and elements); this module only drives the solver and names the numbers.

Every solve uses the fixed Newton criterion (run.parameters.newton): stagnation below `stag`, at most `max_iter`
iterations, limiter 10 RT/F, halving line search to 1/64, exponent clip 200.  A solve that does not converge raises
unless it is a level run (then the run entry carries converged = false and the judge's solver gate fails it).
Every mesh is checked against the fixed vertex hash, connectivity hash and element count where the case definitions
them; a mismatch raises (ERROR), never a silent substitute.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import c3_checks as C
from axionfem.chem import CurrentDistribution, Reaction, meshes as M

REPO = Path(__file__).resolve().parents[3]
LC = 0.01
F_CONST, R_CONST, T_K = 96485.33212, 8.314462618, 298.15
TWO_PI = 2 * math.pi


# ------------------------------------------------------------------------------------------------ helpers
def _num(v):
    return C._num(v)


def _newton(params):
    n = params["newton"]
    return dict(stag=float(n["stag"]), max_iter=int(n["max_iter"]), gate=1e-12, limiter_rt_f=10.0, ls_min=1 / 64,
                clip=200.0, temperature=T_K, initial="zero", check=False)   # fixed start phi = 0; the gate judges


def _rx(lst):
    return [Reaction.from_dict(r) for r in lst]


def _solver_entry(rep):
    return dict(converged=bool(rep["converged"]), relative_residual=float(rep["relative_residual"]),
                iterations=int(rep["iterations"]), residual_history=[float(v) for v in rep["residual_history"]],
                stop=rep["stop"], limited_steps=int(rep["limited_steps"]), clamp_active=bool(rep["clamp_active"]),
                field_residual=float(rep["field_residual"]), constraint_residual=float(rep["constraint_residual"]),
                linear_solver=rep["linear_solver"], factor_nonzeros=int(rep["factor_nonzeros"]))


def _require(rep, what):
    if not rep["converged"]:
        raise RuntimeError(f"{what}: Newton did not converge (stop {rep['stop']}, relres {rep['relative_residual']:.3e},"
                           f" clamp {rep['clamp_active']}, history {rep['residual_history']})")


def _run_entry(level, cd, t0, family=None, k=None, h_nominal=None):
    m = cd.mesh_stats()
    if h_nominal is not None:
        m["h_nominal"] = float(h_nominal)
    if family is not None:
        m["family"] = family
        m["refinement_level"] = int(k)
    return dict(level=level, mesh=m, solver=_solver_entry(cd.report), seconds=time.time() - t0)


def _check_mesh(cd, spec, problems):
    """Vertex hash, connectivity hash and element count of the SOLVER's mesh against the fixed level."""
    V, E = cd.mesh_vertices(), cd.mesh_elements()
    p = C.mesh_problems(spec, V, len(E))
    if "connectivity_hash" in spec:
        h = C.connectivity_hash(V, E, spec["hash_scale"])
        if h != spec["connectivity_hash"]:
            p.append(f"mesh {spec['name']}: connectivity hash {h} != fixed {spec['connectivity_hash']}")
    if p:
        raise RuntimeError("; ".join(p))
    problems.append(spec["name"])


def _bq(cd, attrs, degree=None):
    return cd.boundary_quadrature([int(a) for a in attrs], int(degree or cd.boundary_degree))


def _kin_current(cd, attrs, phi_s, reactions, axi, g=None, weight=None):
    q = _bq(cd, attrs)
    return C.electrode_current(q["x"], q["w"], q["phi"], phi_s, reactions, axi, g=g, weight=weight)


def _flux_current(cd, attrs, kappa, axi):
    """Strong flux of one-sided element gradients: int kappa grad phi_h . n_out dA (3D total in axisymmetry)."""
    q = _bq(cd, attrs)
    x = q["x"]
    dens = kappa(x) * np.sum(q["grad"] * q["normal"], 1)
    w = q["w"] * (x[:, 0] * TWO_PI if axi else 1.0)
    return float(np.sum(dens * w))


def _measure(cd, attrs, axi):
    q = _bq(cd, attrs)
    return C.electrode_measure(q["x"], q["w"], axi)


def _react_current(cd, k, axi):
    """Reaction (variational) current of electrode k: the full residual without its own term, summed over its dofs."""
    return C.reaction_current(cd.residual_full(k), cd.electrode_dofs(k), axi)


def _variational_density(cd, k, rs):
    """Current density on electrode k by the variational flux: M_e lam = R_e (boundary mass of the electrode, same
    r weight as the residual), then lam interpolated along r at the radii rs."""
    import scipy.sparse.linalg as spla
    dofs = np.asarray(cd.electrode_dofs(k), int)
    attrs = cd.electrode_attrs(k)
    Mb = cd.boundary_mass(attrs).tocsr()[dofs][:, dofs].tocsc()
    R = np.asarray(cd.residual_full(k))[dofs]
    lam = spla.spsolve(Mb, R)
    r = np.asarray(cd.dof_coordinates())[dofs, 0]
    o = np.argsort(r)
    return [float(np.interp(p, r[o], lam[o])) for p in rs]


def _cd(mesh, order, axi, extra=0, curved=True):
    q = {"domain": 2 * order + 4 + extra, "boundary": 2 * order + 4 + extra}
    return CurrentDistribution(mesh, order=order, axisymmetric=axi, quadrature=q, curved=curved)


def _add(cd, name, attrs, **kw):
    return cd.add_electrode(name, attrs, **kw)


# ------------------------------------------------------------------------------------------------ MMS (BV fields)
def _mms_mesh(field, family, N):
    if field == "bvann":
        return M.annulus_sector(N, 2 * N, 0.005, 0.01, 0.0, 0.5 * math.pi)
    if family == "struct":
        return M.rect(N, x1=LC, y1=LC)
    if family == "dist":
        return M.distorted_rect(int(round(math.log2(N / 4))), x1=LC, y1=LC)
    raise ValueError(family)


def _mms_solve(data, variant, order, mesh, newton, galvanostatic=False, extra=0, one_more=True):
    """One manufactured-solution solve; returns raw-derived numbers (all through c3_checks)."""
    axi = bool(data["axisymmetric"])
    fns = C.mms_functions(data, variant)
    V = data["variants"][variant]
    t0 = time.time()
    cd = _cd(mesh, order, axi, extra)
    cd.set_conductivity(fns["kappa"])
    cd.set_source(fns["f"])
    for a, fc in data["faces"].items():
        if fc["role"] == "insulating":
            cd.set_insulator_flux(int(a), C.np_fn(fc["numpy"]["flux_kappa_dphi_dn_out"], data["coords"]))
    Iex = {r: _num(data["galvanostatic_I_exact"][variant][r]) for r in ("anode", "cathode")}
    for role in ("anode", "cathode"):
        e = V[role]
        galv = galvanostatic and role == "anode"
        _add(cd, role, [int(e["attr"])], reactions=_rx(e["reactions"]), galvanostatic=galv,
             phi_s=float(e["phi_s_float"]) + (0.02 if galv else 0.0), current=Iex[role] if galv else 0.0,
             g=fns[f"g_{role}"])
    rep = cd.solve(**newton)
    out = dict(cd=cd, t0=t0, rep=rep, axi=axi, Iex=Iex)
    k = order
    dq = cd.domain_quadrature(2 * k + 8 + extra)
    out["errors"] = C.field_errors(dq["x"], dq["w"], dq["phi"], dq["grad"], fns, axi)
    phis = cd.phi_s()
    out["phi_s"] = phis
    out["phi_s_exact"] = [float(V["anode"]["phi_s_float"]), float(V["cathode"]["phi_s_float"])]
    attrs = {r: [int(V[r]["attr"])] for r in ("anode", "cathode")}
    rxs = {r: V[r]["reactions"] for r in ("anode", "cathode")}

    def currents():
        ph = cd.phi_s()
        return {r: _kin_current(cd, attrs[r], ph[i], rxs[r], axi, g=fns[f"g_{r}"])
                for i, r in enumerate(("anode", "cathode"))}
    out["I_kin"] = currents()
    out["I_flux"] = _flux_current(cd, attrs["anode"], fns["kappa"], axi)
    # balance: sum_e I_kin,e + Q_insulators + F_volume (solver's own quadrature), through the check module
    loads = []
    for a, fc in data["faces"].items():
        if fc["role"] == "insulating":
            q = _bq(cd, [int(a)])
            loads.append(C.boundary_load(q["x"], q["w"], C.np_fn(fc["numpy"]["flux_kappa_dphi_dn_out"], data["coords"]),
                                         axi))
    dqa = cd.domain_quadrature(cd.domain_degree)
    loads.append(C.volume_load(dqa["x"], dqa["w"], fns["f"], axi))
    Is = list(out["I_kin"].values())
    out["balance"] = abs(sum(Is) + sum(loads))
    out["balance_terms"] = sum(abs(v) for v in Is) + sum(abs(v) for v in loads)
    out["measures"] = [_measure(cd, attrs[r], axi) for r in ("anode", "cathode")]
    if axi:
        out["measures_exact"] = [math.pi * LC ** 2] * 2
    elif data["field"] == "bvann":
        out["measures_exact"] = [0.5 * math.pi * 0.005, 0.5 * math.pi * 0.01]
    else:
        out["measures_exact"] = [LC, LC]
    wfun = (lambda X: (1 - (X[:, 0] / LC) ** 2) ** 2)
    if axi:
        out["J_w"] = _kin_current(cd, attrs["anode"], phis[0], rxs["anode"], axi, g=fns["g_anode"], weight=wfun)
        if variant in data.get("weighted_I_exact", {}):
            out["J_w_exact"] = _num(data["weighted_I_exact"][variant])
    out["window_ratio"] = C.newton_window_ratio(rep["residual_history"])
    if one_more:
        z = cd.state()
        z1 = cd.newton_update(z)
        cd.set_state(z1)
        out["I_kin_plus1"] = currents()["anode"]
        out["phi_s_plus1"] = cd.phi_s()[0]
        if axi:
            out["J_w_plus1"] = _kin_current(cd, attrs["anode"], cd.phi_s()[0], rxs["anode"], axi, g=fns["g_anode"],
                                            weight=wfun)
        cd.set_state(z)
    return out


def _mms_groups(params, data, variants, groups, galvanostatic=False, quad2x=(), check_meshes=True):
    """groups: (group name, order, family, mesh key in params.meshes, what) — the series of each group."""
    runs, series, q, notes = [], {}, {}, []
    newton = _newton(params)
    field = data["field"]
    checked = []
    for v in variants:
        for gname, order, family, mkey, what in groups:
            ms = params["meshes"][mkey]
            Ns = ms.get("N") or ms.get("n")
            lv = []
            acc = {k: [] for k in ("L2", "H1", "L2w", "H1w", "Ika", "Ikc", "Ifa", "phis", "Jw", "bal", "terms", "mdev")}
            for k, N in enumerate(Ns):
                mesh = _mms_mesh(field, family, N)
                o = _mms_solve(data, v, order, mesh, newton, galvanostatic=galvanostatic,
                               one_more=(k == len(Ns) - 1))
                cd = o["cd"]
                if check_meshes and "levels" in ms:
                    _check_mesh(cd, ms["levels"][k], checked)
                level = f"{v}_{gname}_N{N}"
                lv.append(level)
                runs.append(_run_entry(level, cd, o["t0"], ms["family"], k, ms["h_nominal"][k]))
                e = o["errors"]
                acc["L2"].append(e["L2"]); acc["H1"].append(e["H1"])
                acc["L2w"].append(e.get("L2_w", float("nan"))); acc["H1w"].append(e.get("H1_w", float("nan")))
                acc["Ika"].append(o["I_kin"]["anode"] - o["Iex"]["anode"])
                acc["Ikc"].append(o["I_kin"]["cathode"] - o["Iex"]["cathode"])
                acc["Ifa"].append(o["I_flux"] - o["Iex"]["anode"])
                acc["phis"].append(o["phi_s"][0] - o["phi_s_exact"][0])
                if "J_w_exact" in o:
                    acc["Jw"].append(o["J_w"] - o["J_w_exact"])
                acc["bal"].append(o["balance"]); acc["terms"].append(o["balance_terms"])
                acc["mdev"].append(max(abs(a - b) for a, b in zip(o["measures"], o["measures_exact"])))
                if k == len(Ns) - 1:
                    p_ = f"{v}_{gname}"
                    Ia = o["Iex"]["anode"]
                    q[p_ + "_newton_window_ratio"] = o["window_ratio"]
                    q[p_ + "_newton_stop"] = o["rep"]["stop"]
                    q[p_ + "_Iflux_rel_finest"] = abs(acc["Ifa"][-1]) / abs(Ia)
                    q[p_ + "_Ikin_finest"] = o["I_kin"]["anode"]
                    q[p_ + "_Ikin_finest_plus1"] = o["I_kin_plus1"]
                    if "J_w" in o:
                        q[p_ + "_Jw_finest"] = o["J_w"]
                        q[p_ + "_Jw_finest_plus1"] = o["J_w_plus1"]
                    q[p_ + "_phis_finest"] = o["phi_s"][0]
                    q[p_ + "_phis_finest_plus1"] = o["phi_s_plus1"]
                    q[p_ + "_Ikin_finest_err_rel"] = abs(acc["Ika"][-1]) / abs(Ia)
                    if gname in quad2x:
                        o2 = _mms_solve(data, v, order, mesh, newton, galvanostatic=galvanostatic, extra=8,
                                        one_more=False)
                        _require(o2["rep"], f"{level} with the doubled quadrature")
                        q[p_ + "_Ikin_finest_quad2x"] = o2["I_kin"]["anode"]
                del o, cd
            p = f"{v}_{gname}"
            Iabs = abs(_num(data["galvanostatic_I_exact"][v]["anode"]))
            nrm = _exact_norms(data, v)
            if "L2" in what:
                if data["axisymmetric"]:
                    series[p + "_L2w"] = C.order_series(lv, acc["L2w"], nrm["L2_w"])
                    series[p + "_H1w"] = C.order_series(lv, acc["H1w"], nrm["H1_w"])
                series[p + "_L2"] = C.order_series(lv, acc["L2"], nrm["L2"])
                series[p + "_H1"] = C.order_series(lv, acc["H1"], nrm["H1"])
            if "I" in what:
                series[p + "_Ikin_anode"] = C.order_series(lv, acc["Ika"], Iabs)
                series[p + "_Ikin_cathode"] = C.order_series(lv, acc["Ikc"], Iabs)
            if "Iflux" in what:
                series[p + "_Iflux_anode"] = C.order_series(lv, acc["Ifa"], Iabs)
            if "phis" in what:
                series[p + "_phis"] = C.order_series(lv, acc["phis"],
                                                     abs(float(data["variants"][v]["anode"]["phi_s_float"])))
            if "Jw" in what and acc["Jw"]:
                series[p + "_Jw_anode"] = C.order_series(lv, acc["Jw"], abs(_num(data["weighted_I_exact"][v])))
            q[p + "_balance_abs_max"] = float(max(acc["bal"]))
            q[p + "_I_scale"] = float(max(acc["terms"]))
            q[p + "_I_exact_abs"] = Iabs
            q[p + "_measure_dev_max"] = float(max(acc["mdev"]))
            q[p + "_measure_scale"] = float(max(o_meas for o_meas in _measures_exact(data)))
    notes.append(f"meshes verified by vertex/connectivity hash and element count: {len(checked)} levels")
    return runs, series, q, notes


def _measures_exact(data):
    if data["axisymmetric"]:
        return [math.pi * LC ** 2] * 2
    if data["field"] == "bvann":
        return [0.5 * math.pi * 0.005, 0.5 * math.pi * 0.01]
    return [LC, LC]


_NORMS = {}


def _exact_norms(data, variant=None):
    """Norms of the manufactured solution (solution_magnitude of the error series), by quadrature of the exact
    field on a fine P2 mesh of the domain (the solver's own quadrature data, solution-independent)."""
    key = data["field"]
    if key in _NORMS:
        return _NORMS[key]
    if key == "bvann":
        m = M.annulus_sector(64, 128, 0.005, 0.01, 0.0, 0.5 * math.pi)
    else:
        m = M.rect(64, x1=LC, y1=LC)
    cd = _cd(m, 2, bool(data["axisymmetric"]))
    cd.set_conductivity({1: 1.0})
    _add(cd, "ground", [1], primary=True, phi_s=0.0)   # only the quadrature points and weights are used
    cd.set_state(cd.initial_state())
    dq = cd.domain_quadrature(10)
    X, W = dq["x"], dq["w"]
    fns = C.mms_functions(data)
    u, g = fns["phi"](X), fns["grad"](X)
    out = {"L2": math.sqrt(np.sum(u * u * W)), "H1": math.sqrt(np.sum(np.sum(g * g, 1) * W))}
    if data["axisymmetric"]:
        out["L2_w"] = math.sqrt(np.sum(u * u * W * X[:, 0]))
        out["H1_w"] = math.sqrt(np.sum(np.sum(g * g, 1) * W * X[:, 0]))
    _NORMS[key] = out
    return out


def _data(params):
    f = params["field"]
    return C.load_json(REPO / f["path"], f["sha256"])


def _taylor(params, data, variant):
    """Jacobian Taylor test per run.parameters.taylor: converged galvanostatic state (P2, N = 8) + 0.01 N(0,1),
    direction N(0,1) (PCG64(7), perturbation first); remainders |R(z + e v) - R(z) - e J v|_2 incl. the bordered row."""
    tp = params["taylor"]
    newton = _newton(params)
    o = _mms_solve(data, variant, 2, M.rect(8, x1=LC, y1=LC), newton, galvanostatic=True, one_more=False)
    _require(o["rep"], f"Taylor state {variant}")
    cd = o["cd"]
    z = np.asarray(cd.state())
    rng = np.random.Generator(np.random.PCG64(7))
    z0 = z + float(tp["perturbation_scale"]) * rng.standard_normal(z.size)
    v = float(tp["direction_scale"]) * rng.standard_normal(z.size)
    r0 = np.asarray(cd.residual(z0))
    Jv = cd.jacobian(z0) @ v
    eps = [float(e) for e in tp["eps"]]
    rem = [float(np.linalg.norm(np.asarray(cd.residual(z0 + e * v)) - r0 - e * Jv)) for e in eps]
    t = C.taylor_test(eps, rem)
    return t["min"], rem


# ------------------------------------------------------------------------------------------------ case functions
def mms_bv_2d(params, workdir, quick=False):
    data = _data(params)
    variants = list(params["variants"])
    runs, series, q, notes = _mms_groups(params, data, variants,
                                         [("p1_struct", 1, "struct", "p1_struct", "L2 I Iflux"),
                                          ("p2_struct", 2, "struct", "p2_struct", "L2 I")], quad2x=("p2_struct",))
    r2, s2, q2, n2 = _mms_groups(params, data, ["bv_sym", "bv_asym"],
                                 [("p1_dist", 1, "dist", "p1_dist", "L2"), ("p2_dist", 2, "dist", "p2_dist", "L2")])
    runs += r2; series.update(s2); q.update(q2); notes += n2
    for v in params["taylor"]["variants"]:
        mn, rem = _taylor(params, data, v)
        q[f"taylor_{v}_min_slope"] = mn
        q[f"taylor_{v}_remainders"] = rem
    return dict(runs=runs, series=series, quantities=q, notes=notes)


def mms_annulus(params, workdir, quick=False):
    data = _data(params)
    runs, series, q, notes = _mms_groups(params, data, ["bv_sym", "bv_asym"],
                                         [("p1_ann", 1, "struct", "p1", "L2 I"), ("p2_ann", 2, "struct", "p2", "L2 I")])
    return dict(runs=runs, series=series, quantities=q, notes=notes)


def mms_galvanostatic(params, workdir, quick=False):
    data = _data(params)
    runs, series, q, notes = _mms_groups(params, data, ["bv_sym", "multi", "tafel"],
                                         [("p1_struct", 1, "struct", "p1_struct", "L2 phis"),
                                          ("p2_struct", 2, "struct", "p2_struct", "L2 phis")], galvanostatic=True)
    return dict(runs=runs, series=series, quantities=q, notes=notes)


def mms_axi(params, workdir, quick=False):
    data = _data(params)
    runs, series, q, notes = _mms_groups(params, data, ["bv_sym", "bv_asym", "tafel"],
                                         [("p1_struct", 1, "struct", "p1_struct", "L2 I"),
                                          ("p2_struct", 2, "struct", "p2_struct", "L2"),
                                          ("p2_cur", 2, "struct", "p2_axi_current", "I Jw")])
    r2, s2, q2, n2 = _mms_groups(params, data, ["bv_asym"],
                                 [("galv_p1", 1, "struct", "p1_struct", "phis"),
                                  ("galv_p2", 2, "struct", "p2_axi_current", "phis")], galvanostatic=True)
    runs += r2; series.update(s2); q.update(q2); notes += n2
    return dict(runs=runs, series=series, quantities=q, notes=notes)


def mms_kappa_jump(params, workdir, quick=False):
    data = _data(params)
    newton = _newton(params)
    c = data["coords"]
    nf = data["numpy"]
    fL, fR = C.np_fn(nf["f_left"], c), C.np_fn(nf["f_right"], c)
    left = lambda X: X[:, 0] < 0.5 * LC
    src = lambda X: np.where(left(X), fL(X), fR(X))
    fns = C.mms_functions(data)
    kap = {1: _num(data["kappa_left"]), 2: _num(data["kappa_right"])}
    kfun = lambda X: np.where(left(X), kap[1], kap[2])
    eattr = lambda cen: np.where(cen[:, 0] < 0.5 * LC, 1, 2)
    Iex = {k: _num(v) for k, v in data["primary_I_exact"].items()}
    ea, ec = data["electrodes"]["anode"], data["electrodes"]["cathode"]
    nrm = None
    runs, series, q, checked = [], {}, {}, []

    def mesh_of(family, N):
        if family == "struct":
            return M.rect(N, x1=LC, y1=LC, eattr_fn=eattr)
        return M.distorted_rect(int(round(math.log2(N / 4))), x1=LC, y1=LC, eattr_fn=eattr)

    def solve(mesh, order, galv=False):
        t0 = time.time()
        cd = _cd(mesh, order, False)
        cd.set_conductivity(kap)
        cd.set_source(src)
        for a, fc in data["faces"].items():
            ql, qr = C.np_fn(fc["numpy"]["flux_left"], c), C.np_fn(fc["numpy"]["flux_right"], c)
            cd.set_insulator_flux(int(a), (lambda ql, qr: (lambda X: np.where(left(X), ql(X), qr(X))))(ql, qr))
        _add(cd, "anode", [int(ea["attr"])], primary=True, E_eq=_num(ea["E_eq"]), galvanostatic=galv,
             phi_s=_num(ea["phi_s"]) + (0.05 if galv else 0.0), current=Iex["anode"] if galv else 0.0)
        _add(cd, "cathode", [int(ec["attr"])], primary=True, E_eq=_num(ec["E_eq"]), phi_s=_num(ec["phi_s"]))
        cd.solve(**newton)
        return cd, t0

    for gname, order, family in (("p1_struct", 1, "struct"), ("p2_struct", 2, "struct"), ("p1_dist", 1, "dist"),
                                 ("p2_dist", 2, "dist")):
        ms = params["meshes"][gname]
        L2, H1, lv, Ir, md = [], [], [], [], []
        for k, N in enumerate(ms["N"]):
            cd, t0 = solve(mesh_of(family, N), order)
            if "levels" in ms:
                _check_mesh(cd, ms["levels"][k], checked)
            level = f"{gname}_N{N}"
            lv.append(level)
            runs.append(_run_entry(level, cd, t0, ms["family"], k, ms["h_nominal"][k]))
            dq = cd.domain_quadrature(2 * order + 8)
            e = C.field_errors(dq["x"], dq["w"], dq["phi"], dq["grad"], fns, False)
            L2.append(e["L2"]); H1.append(e["H1"])
            for i, role in enumerate(("anode", "cathode")):
                Ir.append(abs(_react_current(cd, i, False) - Iex[role]) / abs(Iex[role]))
            md.append(max(abs(_measure(cd, [int(x["attr"])], False) - LC) for x in (ea, ec)))
        if nrm is None:
            nrm = _exact_norms_jump(data, fns)
        series[gname + "_L2"] = C.order_series(lv, L2, nrm["L2"])
        series[gname + "_H1"] = C.order_series(lv, H1, nrm["H1"])
        q[gname + "_Ireact_rel_max"] = float(max(Ir))
        q[gname + "_measure_dev_max"] = float(max(md))
        q[gname + "_measure_scale"] = LC
    # galvanostatic primary anode (floating-conductor tie)
    for gname, order in (("p1_struct", 1), ("p2_struct", 2)):
        e = []
        for N in params["meshes"][gname]["N"]:
            cd, _ = solve(mesh_of("struct", N), order, galv=True)
            _require(cd.report, f"galvanostatic {gname} N={N}")
            e.append(abs(cd.phi_s()[0] - _num(ea["phi_s"])))
        q[gname + "_galv_phis_abs_max"] = float(max(e))
    return dict(runs=runs, series=series, quantities=q,
                notes=[f"meshes verified by hash: {len(checked)} levels"])


def _exact_norms_jump(data, fns):
    m = M.rect(64, x1=LC, y1=LC, eattr_fn=lambda cen: np.where(cen[:, 0] < 0.5 * LC, 1, 2))
    cd = _cd(m, 2, False)
    cd.set_conductivity({1: 1.0, 2: 1.0})
    _add(cd, "ground", [1], primary=True, phi_s=0.0)   # only the quadrature points and weights are used
    cd.set_state(cd.initial_state())
    dq = cd.domain_quadrature(10)
    X, W = dq["x"], dq["w"]
    u, g = fns["phi"](X), fns["grad"](X)
    return {"L2": math.sqrt(np.sum(u * u * W)), "H1": math.sqrt(np.sum(np.sum(g * g, 1) * W))}


# ------------------------------------------------------------------------------------------------ B1 cell
def _cell_solve(params, variant, newton, order=1, scale=1.0):
    cell = params["cell"]
    spec = params["variants"][variant]
    L, H = _num(cell["L"]) * scale, _num(cell["H"]) * scale
    two = spec["layers"] == "two"
    eattr = (lambda cen: np.where(cen[:, 0] < 0.5 * L, 1, 2)) if two else None
    mesh = M.rect(8, 2, x1=L, y1=H, eattr_fn=eattr)
    kap = {1: _num(cell["kappa_two"][0]), 2: _num(cell["kappa_two"][1])} if two else {1: _num(cell["kappa"])}
    t0 = time.time()
    cd = _cd(mesh, order, False)
    cd.set_conductivity(kap)
    rxa, rxc = _rx(spec["anode"]), _rx(spec["cathode"])
    for r in rxa + rxc:
        r.i0 /= scale
    if spec["control"] == "galvanostatic":
        I = _num(spec["i"]) * H / scale
        _add(cd, "anode", [1], reactions=rxa, galvanostatic=True, current=I, phi_s=0.5)
    else:
        _add(cd, "anode", [1], reactions=rxa, phi_s=_num(spec["V"]))
    _add(cd, "cathode", [2], reactions=rxc, phi_s=0.0)
    rep = cd.solve(**newton)
    _require(rep, f"cell_1d {variant} (order {order}, length scale {scale})")
    ph = cd.phi_s()
    raw = lambda rx: [dict(kind=r.kind, i0=r.i0, E_eq=r.E_eq, alpha_a=r.alpha_a, alpha_c=r.alpha_c, A=r.A) for r in rx]
    Ia = _kin_current(cd, [1], ph[0], raw(rxa), False)
    Ic = _kin_current(cd, [2], ph[1], raw(rxc), False)
    return dict(cd=cd, t0=t0, I=Ia, Ic=Ic, V=ph[0] - ph[1], measure=_measure(cd, [1], False), H=H, L=L)


def cell_1d(params, workdir, quick=False):
    newton = _newton(params)
    q, runs = {}, []
    for v in params["variants"]:
        o = _cell_solve(params, v, newton)
        runs.append(_run_entry(v, o["cd"], o["t0"], h_nominal=o["L"] / 8))
        q[f"{v}_I"] = o["I"]
        q[f"{v}_V"] = o["V"]
        q[f"{v}_balance_abs"] = abs(o["I"] + o["Ic"])
        q[f"{v}_I_scale"] = abs(o["I"]) + abs(o["Ic"])
        q[f"{v}_measure_dev"] = abs(o["measure"] - o["H"])
    q["measure_scale"] = _num(params["cell"]["H"])
    s = float(params["break_scale"])
    for v in ("sym_pot", "asym_pot", "irrev_pot", "twolayer_galv"):
        o = _cell_solve(params, v, newton, scale=s)
        runs.append(_run_entry(f"{v}_break", o["cd"], o["t0"], h_nominal=o["L"] / 8))
        q[f"{v}_break_I"] = o["I"]
        q[f"{v}_break_V"] = o["V"]
    o = _cell_solve(params, "asym_pot", newton, order=2)
    runs.append(_run_entry("asym_pot_p2", o["cd"], o["t0"], h_nominal=o["L"] / 8))
    q["asym_pot_p2_I"] = o["I"]
    return dict(runs=runs, series={}, quantities=q, notes=["B1 cell: rect(8, 2), P1 (P2 for asym_pot_p2)"])


# ------------------------------------------------------------------------------------------------ galvanic couple, B5
def galvanic_1d(params, workdir, quick=False):
    newton = _newton(params)
    g = params["galvanic"]
    L, H = _num(g["L"]), _num(g["H"])
    q, runs = {}, []
    for k in g["kappas"]:
        t0 = time.time()
        cd = _cd(M.rect(8, 2, x1=L, y1=H), 1, False)
        cd.set_conductivity({1: _num(k)})
        _add(cd, "left", [1], reactions=_rx(g["left"]), phi_s=0.0)
        _add(cd, "right", [2], reactions=_rx(g["right"]), phi_s=0.0)
        cd.solve(**newton)
        runs.append(_run_entry(f"kappa_{k}", cd, t0, h_nominal=L / 8))
        Il = _kin_current(cd, [1], 0.0, g["left"], False)
        Ir = _kin_current(cd, [2], 0.0, g["right"], False)
        bq = _bq(cd, [1])
        q[f"kappa_{k}_I"] = Il
        q[f"kappa_{k}_phi_left"] = float(np.sum(bq["phi"] * bq["w"]) / np.sum(bq["w"]))
        q[f"kappa_{k}_balance_abs"] = abs(Il + Ir)
        q[f"kappa_{k}_I_scale"] = abs(Il) + abs(Ir)
    mx = params["mixed"]
    t0 = time.time()
    cd = _cd(M.rect(4, x1=LC, y1=LC), 1, False)
    cd.set_conductivity({1: 1.0})
    _add(cd, "metal", [1], reactions=_rx([mx["anode"], mx["cathode"]]), phi_s=0.0)
    cd.solve(**newton)
    runs.append(_run_entry("mixed", cd, t0, h_nominal=LC / 4))
    phi = np.asarray(cd.phi_dofs())
    q["mixed_E_corr"] = float(-np.mean(phi))
    q["mixed_phi_spread"] = float(np.ptp(phi))
    return dict(runs=runs, series={}, quantities=q,
                notes=["mixed potential from the fixed default guess phi = 0"])


# ------------------------------------------------------------------------------------------------ B2/B3 concentric
# BV_SYM_* / BV_ASYM_* are named, not listed, in run.parameters.kinetics; their values are those of the cell_1d
# variants sym_pot / asym_pot in the run parameters of chem_analytic_c3_cell_1d.
_KIN = {
    "bv_sym": ([dict(kind="bv", i0="10", E_eq="0.05", alpha_a="0.5", alpha_c="0.5")],
               [dict(kind="bv", i0="10", E_eq="-0.02", alpha_a="0.5", alpha_c="0.5")]),
    "bv_asym": ([dict(kind="bv", i0="20", E_eq="0.05", alpha_a="0.3", alpha_c="0.7")],
                [dict(kind="bv", i0="5", E_eq="-0.02", alpha_a="0.6", alpha_c="0.4")]),
}


def concentric(params, workdir, quick=False):
    newton = _newton(params)
    g = params["geometry"]
    r1, r2, kap = _num(g["r1"]), _num(g["r2"]), _num(params["kappa"])
    runs, series, q, checked = [], {}, {}, []
    for geom in ("planar_cyl", "axi_sph", "axi_cyl"):
        axi = geom != "planar_cyl"
        for kin in ("primary", "bv_sym", "bv_asym"):
            for order in (1, 2):
                ms = params["meshes"][f"{geom}_p{order}"]
                lv, vals, fvals, bal = [], [], [], []
                for k, N in enumerate(ms["N"]):
                    if geom == "planar_cyl":
                        mesh = M.annulus_sector(N, 2 * N, r1, r2, 0.0, 0.5 * math.pi)
                    elif geom == "axi_sph":
                        mesh = M.annulus_sector(N, 4 * N, r1, r2, 0.0, math.pi, axisym_meridian=True)
                    else:
                        mesh = M.rect(N, N, x0=r1, x1=r2, y0=0.0, y1=0.01)
                    t0 = time.time()
                    cd = _cd(mesh, order, axi)
                    cd.set_conductivity({1: kap})
                    if kin == "primary":
                        _add(cd, "anode", [1], primary=True, phi_s=float(params["V_primary"]))
                        _add(cd, "cathode", [2], primary=True, phi_s=0.0)
                    else:
                        A, Cc = _KIN[kin]
                        _add(cd, "anode", [1], reactions=_rx(A), phi_s=float(params["V_secondary"]))
                        _add(cd, "cathode", [2], reactions=_rx(Cc), phi_s=0.0)
                    cd.solve(**newton)
                    if "levels" in ms:
                        _check_mesh(cd, ms["levels"][k], checked)
                    level = f"{geom}_{kin}_p{order}_N{N}"
                    lv.append(level)
                    runs.append(_run_entry(level, cd, t0, ms["family"], k, ms["h_nominal"][k]))
                    if kin == "primary":
                        Ia, Ic = _react_current(cd, 0, axi), _react_current(cd, 1, axi)
                    else:
                        A, Cc = _KIN[kin]
                        ph = cd.phi_s()
                        Ia = _kin_current(cd, [1], ph[0], A, axi)
                        Ic = _kin_current(cd, [2], ph[1], Cc, axi)
                    vals.append(Ia)
                    fvals.append(_flux_current(cd, [1], lambda X: np.full(len(X), kap), axi))
                    bal.append(abs(Ia + Ic))
                    del cd
                p = f"{geom}_{kin}_p{order}"
                series[p + "_I"] = C.functional_series(lv, vals)
                series[p + "_Iflux"] = C.functional_series(lv, fvals)
                q[p + "_I_finest"] = vals[-1]
                q[p + "_balance_abs_max"] = max(bal)
                q[p + "_I_scale"] = abs(vals[-1])
    return dict(runs=runs, series=series, quantities=q, notes=[f"meshes verified by hash: {len(checked)} levels"])


# ------------------------------------------------------------------------------------------------ B4 disks
def _disk_solve(disk, exterior_modes, n, beta, order, kinetics=None, i0=None, kappa=None):
    r0, V = _num(disk["r0"]), _num(disk["V"])
    kap = _num(disk["kappa"]) if kappa is None else kappa
    Rb = _num(disk["Rb_over_r0"]) * r0
    mesh = M.disk_graded(n, r0=r0, Rb=Rb, beta=beta)
    t0 = time.time()
    cd = _cd(mesh, order, True)
    cd.set_conductivity({1: kap})
    if kinetics is None:
        _add(cd, "disk", [1], primary=True, phi_s=V)
    else:
        _add(cd, "disk", [1], reactions=[Reaction("linear", i0=i0, E_eq=0.0, alpha_a=0.5, alpha_c=0.5)], phi_s=V)
    cd.set_exterior_halfspace(4, Rb, exterior_modes, kap)
    return cd, t0, dict(r0=r0, V=V, kappa=kap, Rb=Rb)


DTN_MODES = 12   # run.parameters.exterior: modes n = 0..11


def disk_primary(params, workdir, quick=False):
    newton = _newton(params)
    disk = params["disk"]
    pts = [float(p) for p in params["points_r_over_r0"]]
    runs, series, q, checked = [], {}, {}, []
    r0, V, kap = _num(disk["r0"]), _num(disk["V"]), _num(disk["kappa"])
    Iex = 4 * kap * r0 * V
    for name, order in (("p1", 1), ("p2", 2), ("uniform_p1", 1), ("uniform_p2", 2)):
        ms = params["meshes"][name]
        lv, e, ef, vals = [], [], [], []
        for k, n in enumerate(ms["n"]):
            cd, t0, g = _disk_solve(disk, DTN_MODES, n, float(ms["beta"]), order)
            cd.solve(**newton)
            _check_mesh(cd, ms["levels"][k], checked)
            level = f"{name}_n{n}"
            lv.append(level)
            runs.append(_run_entry(level, cd, t0, ms["family"], k, ms["h_nominal"][k]))
            I = _react_current(cd, 0, True)
            e.append(I / Iex - 1)
            ef.append(_flux_current(cd, [1], lambda X: np.full(len(X), kap), True) / Iex - 1)
            vals.append(I)
            q[f"{name}_n{n}"] = I
            q[f"{name}_measure_dev_n{n}"] = abs(_measure(cd, [1], True) - math.pi * r0 ** 2)
            q[f"{name}_measure_scale"] = math.pi * r0 ** 2
            if k == len(ms["n"]) - 1:
                dens = _variational_density(cd, 0, [p * r0 for p in pts])
                for p, dv in zip(pts, dens):
                    q[f"{name}_density_var_rel_r{p}"] = dv / (2 * kap * V / (math.pi * r0 * math.sqrt(1 - p * p))) - 1
                q[f"{name}_I_react"] = I
            del cd
        series[name + "_I_rel"] = C.order_series(lv, e, 1.0)
        series[name + "_Iflux_rel"] = C.order_series(lv, ef, 1.0)
        series[name + "_I"] = C.functional_series(lv, vals)
        q[name + "_from_above"] = float(min(e))
        q[name + "_measure_dev_max"] = max(q[f"{name}_measure_dev_n{n}"] for n in ms["n"])
    return dict(runs=runs, series=series, quantities=q,
                notes=[f"meshes verified by hash: {len(checked)} levels; half-space even-mode DtN, {DTN_MODES} modes"])


def disk_secondary(params, workdir, quick=False):
    newton = _newton(params)
    disk = params["disk"]
    pts = [float(p) for p in params["points_r_over_r0"]]
    ms = params["meshes"]["p2"]
    runs, series, q, checked = [], {}, {}, []
    r0, V, kap = _num(disk["r0"]), _num(disk["V"]), _num(disk["kappa"])
    lin = [dict(kind="linear", i0=None, E_eq="0", alpha_a="0.5", alpha_c="0.5")]
    for J in params["Js"]:
        i0 = float(params["i0_A_per_m2"][J])
        rx = [dict(lin[0], i0=i0)]
        lv, vals = [], []
        for k, n in enumerate(ms["n"]):
            cd, t0, g = _disk_solve(disk, DTN_MODES, n, float(ms["beta"]), 2, kinetics="linear", i0=i0)
            cd.solve(**newton)
            _check_mesh(cd, ms["levels"][k], checked)
            level = f"J{J}_n{n}"
            lv.append(level)
            runs.append(_run_entry(level, cd, t0, ms["family"], k, ms["h_nominal"][k]))
            I = _kin_current(cd, [1], V, rx, True)
            vals.append(I / (4 * kap * r0 * V))
            q[f"J{J}_n{n}"] = vals[-1]
            q["p2_measure_dev_max"] = max(q.get("p2_measure_dev_max", 0.0), abs(_measure(cd, [1], True) - math.pi * r0 ** 2))
            q["p2_measure_scale"] = math.pi * r0 ** 2
            if k == len(ms["n"]) - 1:
                Iavg = I / (math.pi * r0 ** 2)
                for p, dv in zip(pts, _variational_density(cd, 0, [p * r0 for p in pts])):
                    q[f"J{J}_i_over_iavg_r{p}"] = dv / Iavg
            del cd
        series[f"J{J}_I_over_primary"] = C.functional_series(lv, vals)
        q[f"J{J}_I_over_primary"] = vals[-1]
    return dict(runs=runs, series=series, quantities=q,
                notes=[f"meshes verified by hash: {len(checked)} levels; i0 fixed per J (no conversion)"])
