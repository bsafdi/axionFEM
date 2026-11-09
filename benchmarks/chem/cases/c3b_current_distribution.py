"""C3b case runner (set-up and solve; the error computation is in the check modules c3_checks and
c3b_checks).

Contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.
Every function reads only its fixed ``params`` and the fixed case data under benchmarks/chem/casedata (sha256
checked by c3_checks.load_json).

* 3D cases (mms_3d, concentric_3d): every level runs on exactly 4 MPI ranks (single-node mpirun on this node; the
  1-rank runs and the Taylor test on 1 rank) through ``c3b_mpi_worker.py``; each rank reduces its own raw data through
  the check modules and this module combines the per-rank numbers (sums; c3b_checks.combine_error_sums).  The
  launch goes through ``fluids_mpi.launch`` (benchmarks/fluids/cases); never a serial fallback.
* porous_1d: axionfem.chem.PorousElectrode (serial, login scale).

Meshes are checked against the fixed vertex hash, connectivity hash, element count (and the P2 geometry-node hash)
inside the run; a mismatch raises (ERROR).  Every solve uses the fixed Newton criterion (run.parameters.newton).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import c3_checks as C
import c3b_checks as CB

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORKER = HERE / "c3b_mpi_worker.py"
RANKS = 4
R1, R2, H_CYL = 0.005, 0.02, 0.01
MEASURE_NMAX = 16      # the electrode-measure identity on N <= 16 (run.parameters / acceptance basis)


# ------------------------------------------------------------------------------------------------ MPI launch
def _launch(argv, ranks, workdir):
    """A multi-rank launch through the shared launcher ``fluids_mpi`` (benchmarks/fluids/cases)."""
    import fluids_mpi
    return fluids_mpi.launch(argv, ranks, workdir)


_TASKS = [0]


def _plain(o):
    if hasattr(o, "keys"):
        return {k: o[k] for k in o.keys()}
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _run_mpi(task: dict, ranks: int, workdir) -> list:
    wd = Path(workdir) / "c3b_mpi"
    wd.mkdir(parents=True, exist_ok=True)
    _TASKS[0] += 1
    tp = wd / f"task{_TASKS[0]:04d}.json"
    task = dict(task, sys_path=[str(p) for p in sys.path if p])
    tp.write_text(json.dumps(task, default=_plain))   # params are read-only mappings
    _launch([sys.executable, "-I", "-S", "-B", str(WORKER), str(tp)], ranks, workdir)
    outs = []
    for r in range(ranks):
        f = Path(f"{tp}.rank{r}.json")
        if not f.is_file():
            raise RuntimeError(f"MPI worker rank {r} wrote no result ({f.name})")
        outs.append(json.loads(f.read_text()))
        f.unlink()
    tp.unlink()
    _check_ranks(outs)
    return outs


# ----------------------------------------------------------------------------------------------------- build identity
_CORE = {}


def core_identity() -> dict:
    """This process's compiled core: path, sha256, git sha."""
    if not _CORE:
        import hashlib
        from axionfem import _core
        f = os.path.realpath(_core.__file__)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        _CORE.update(core_file=os.path.basename(f), core_sha256=h.hexdigest(),
                     git_sha=json.loads(_core.build_info()).get("git_sha"))
    return dict(_CORE)


def _check_ranks(outs):
    """Every rank ran the same compiled core as this process."""
    me = core_identity()
    for o in outs:
        b = o.get("build") or {}
        for k in ("core_sha256", "git_sha"):
            if b.get(k) != me[k]:
                raise RuntimeError(f"MPI rank {o.get('rank')} reports {k} = {b.get(k)!r}, this process {me[k]!r}: the "
                                   "ranks must run the core the case was started with")


def _build_note():
    me = core_identity()
    return (f"compiled core {me['core_sha256'][:16]} (git {str(me['git_sha'])[:12]}), checked identical on every MPI "
            f"rank")


def _newton(params):
    n = params["newton"]
    return dict(stag=float(n["stag"]), max_iter=int(n["max_iter"]))


def _solver_entry(rep):
    return dict(converged=bool(rep["converged"]), relative_residual=float(rep["relative_residual"]),
                iterations=int(rep["iterations"]), residual_history=[float(v) for v in rep["residual_history"]],
                stop=rep["stop"], limited_steps=int(rep["limited_steps"]), clamp_active=bool(rep["clamp_active"]),
                field_residual=float(rep["field_residual"]), constraint_residual=float(rep["constraint_residual"]),
                linear_solver=rep["linear_solver"], cg_iterations_max=int(rep.get("cg_iterations_max", 0)))


def _check_mesh(m, spec, geometry=False):
    """The solver's serial mesh (rank 0) against the fixed level: hashes and element count; ERROR on mismatch."""
    p = []
    for key in ("vertex_hash", "connectivity_hash") + (("geometry_node_hash",) if geometry else ()):
        if key in spec and m.get(key) != spec[key]:
            p.append(f"mesh {spec['name']}: {key} {m.get(key)} != fixed {spec[key]}")
    if int(m["n_elements"]) != int(spec["n_elements"]) or int(m["serial_elements"]) != int(spec["n_elements"]):
        p.append(f"mesh {spec['name']}: {m['n_elements']} elements, fixed {spec['n_elements']}")
    if p:
        raise RuntimeError("; ".join(p))


def _mesh_entry(m, h_nominal, family, k, ranks_expected):
    if int(m["ranks"]) != ranks_expected:
        raise RuntimeError(f"the run saw {m['ranks']} MPI ranks, expected {ranks_expected}")
    return dict(h_nominal=float(h_nominal), h_max=float(m["h_max"]),
                h_eff=float((m["volume"] / m["n_elements"]) ** (1 / 3)), n_elements=int(m["n_elements"]),
                dofs_field=int(m["dofs_field"]), dofs_total=int(m["dofs_field"]), family=family,
                refinement_level=int(k), ranks=int(m["ranks"]), owned_cells_min=int(min(m["owned_cells_per_rank"])))


def _sum(outs, key, sub=None):
    return float(sum((o[key][sub] if sub is not None else o[key]) for o in outs))


# ------------------------------------------------------------------------------------------------ MMS 3D
def _bv3d_norms(data):
    """L2 and H1 norms of the bv3d exact solution (solution_magnitude of the error series) by tensor Gauss-Legendre
    quadrature on [0, L]^3 of the check module's numpy spellings."""
    L = C._num(data["L"])
    xg, wg = np.polynomial.legendre.leggauss(60)
    x, w = 0.5 * L * (xg + 1), 0.5 * L * wg
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    W = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    f = C.mms_functions(data)
    u, g = f["phi"](X), f["grad"](X)
    return {"L2": math.sqrt(float(np.sum(u * u * W))), "H1": math.sqrt(float(np.sum(np.sum(g * g, 1) * W)))}


def _mms_task(params, kind, v, k, N, spec, one_more):
    f = params["field"]
    return dict(kind=kind, variant=v, order=k, N=N, data_path=str(REPO / f["path"]), data_sha256=f["sha256"],
                newton=_newton(params), one_more=one_more, hash=dict(scale=spec["hash_scale"]))


def _mms_level(outs):
    """Combine the per-rank numbers of one level (sums of per-rank check-module results)."""
    o0 = outs[0]
    err = CB.combine_error_sums([c for o in outs for c in o["error_chunks"]])
    Ia, Ic = _sum(outs, "I_kin", "anode"), _sum(outs, "I_kin", "cathode")
    loads = [sum(o["loads"][i] for o in outs) for i in range(len(o0["loads"]))]
    meas = [sum(o["measures"][i] for o in outs) for i in range(2)]
    res = dict(rep=o0["report"], mesh=o0["mesh"], L2=err["L2"], H1=err["H1"], Ia=Ia, Ic=Ic,
               balance=abs(Ia + Ic + sum(loads)), terms=abs(Ia) + abs(Ic) + sum(abs(x) for x in loads),
               measures=meas, phi_s=o0["phi_s"][0], seconds=max(o["seconds"] for o in outs),
               I_flux_anode=_sum(outs, "I_flux_anode"))
    if "I_kin_plus1" in o0:
        res["Ia_plus1"] = _sum(outs, "I_kin_plus1", "anode")
        res["phi_s_plus1"] = o0["phi_s_plus1"][0]
    return res


def mms_3d(params, workdir, quick=False):
    data = C.load_json(REPO / params["field"]["path"], params["field"]["sha256"])
    nrm = _bv3d_norms(data)
    runs, series, q, notes = [], {}, {}, []
    ranks_seen, owned_min = [], []
    Iex = {v: {r: C._num(data["galvanostatic_I_exact"][v][r]) for r in ("anode", "cathode")} for v in ("bv_sym",
                                                                                                         "bv_asym")}
    L2c = C._num(data["L"]) ** 2
    finest = {}
    for v in params["variants"]:
        for g, k in (("p1_3d", 1), ("p2_3d", 2)):
            ms = params["meshes"][g]
            lv, L2, H1, Ia, Ic, bal, terms, mdev = [], [], [], [], [], [], [], []
            for j, (N, spec) in enumerate(zip(ms["N"], ms["levels"])):
                last = j == len(ms["N"]) - 1
                o = _mms_level(_run_mpi(_mms_task(params, "mms", v, k, N, spec, last), RANKS, workdir))
                _check_mesh(o["mesh"], spec)
                level = f"{v}_{g}_N{N}"
                lv.append(level)
                me = _mesh_entry(o["mesh"], ms["h_nominal"][j], ms["family"], j, RANKS)
                ranks_seen.append(me["ranks"]), owned_min.append(me["owned_cells_min"])
                runs.append(dict(level=level, mesh=me, solver=_solver_entry(o["rep"]), seconds=o["seconds"]))
                L2.append(o["L2"]), H1.append(o["H1"])
                Ia.append(o["Ia"] - Iex[v]["anode"]), Ic.append(o["Ic"] - Iex[v]["cathode"])
                bal.append(o["balance"]), terms.append(o["terms"])
                if N <= MEASURE_NMAX:
                    mdev.append(max(abs(m - L2c) for m in o["measures"]))
                if last:
                    finest[(v, g)] = o
            p = f"{v}_{g}"
            o = finest[(v, g)]
            series[p + "_L2"] = C.order_series(lv, L2, nrm["L2"])
            series[p + "_H1"] = C.order_series(lv, H1, nrm["H1"])
            series[p + "_Ikin_anode"] = C.order_series(lv, Ia, abs(Iex[v]["anode"]))
            series[p + "_Ikin_cathode"] = C.order_series(lv, Ic, abs(Iex[v]["cathode"]))
            q[p + "_balance_abs_max"] = float(max(bal))
            q[p + "_I_scale"] = float(max(terms))
            q[p + "_measure_dev_max"] = float(max(mdev))
            q[p + "_measure_scale"] = float(L2c)
            q[p + "_newton_window_ratio"] = C.newton_window_ratio(o["rep"]["residual_history"])
            q[p + "_Ikin_finest"] = o["Ia"]
            q[p + "_Ikin_finest_plus1"] = o["Ia_plus1"]
            q[p + "_Ikin_finest_err"] = abs(Ia[-1])
    # Taylor tests (serial, P2 N = 8)
    tp = params["taylor"]
    for v in tp["variants"]:
        t = dict(kind="taylor", variant=v, order=2, N=8, data_path=str(REPO / params["field"]["path"]),
                 data_sha256=params["field"]["sha256"], newton=_newton(params), taylor=tp)
        o = _run_mpi(t, 1, workdir)[0]
        q[f"taylor_{v}_min_slope"] = float(o["min_slope"])
        q[f"taylor_{v}_remainders"] = o["remainders"]
    # galvanostatic bv_asym anode at the exact 3D total (bordered row reduced over all ranks)
    v = "bv_asym"
    galv_finest = {}
    for g, k in (("p1_3d", 1), ("p2_3d", 2)):
        ms = params["meshes"][g]
        lv, L2, ph, bal, terms, cons = [], [], [], [], [], []
        phx = C._num(data["variants"][v]["anode"]["phi_s_float"])
        for j, (N, spec) in enumerate(zip(ms["N"], ms["levels"])):
            last = j == len(ms["N"]) - 1
            o = _mms_level(_run_mpi(_mms_task(params, "galv", v, k, N, spec, last), RANKS, workdir))
            _check_mesh(o["mesh"], spec)
            level = f"bv_asym_galv_{g}_N{N}"
            lv.append(level)
            me = _mesh_entry(o["mesh"], ms["h_nominal"][j], ms["family"], j, RANKS)
            ranks_seen.append(me["ranks"]), owned_min.append(me["owned_cells_min"])
            runs.append(dict(level=level, mesh=me, solver=_solver_entry(o["rep"]), seconds=o["seconds"]))
            L2.append(o["L2"])
            ph.append(o["phi_s"] - phx)
            bal.append(o["balance"]), terms.append(o["terms"])
            cons.append(abs(o["Ia"] - Iex[v]["anode"]) / abs(Iex[v]["anode"]))
            if last:
                galv_finest[g] = o
        p = f"bv_asym_galv_{g}"
        o = galv_finest[g]
        series[p + "_phis"] = C.order_series(lv, ph, abs(phx))
        series[p + "_L2"] = C.order_series(lv, L2, nrm["L2"])
        q[p + "_balance_abs_max"] = float(max(bal))
        q[p + "_I_scale"] = float(max(terms))
        q[p + "_phis_finest"] = o["phi_s"]
        q[p + "_phis_finest_plus1"] = o["phi_s_plus1"]
        q[p + "_phis_finest_err"] = abs(ph[-1])
        q[p + "_constraint_rel_max"] = float(max(cons))
    q["ranks_4_min"], q["ranks_4_max"] = int(min(ranks_seen)), int(max(ranks_seen))
    q["owned_cells_min"] = int(min(owned_min))
    # the finest P2 bv_asym level, potentiostatic and galvanostatic, on 1 rank
    spec = params["meshes"]["p2_3d"]["levels"][-1]
    N = params["meshes"]["p2_3d"]["N"][-1]
    a1 = _mms_level(_run_mpi(_mms_task(params, "mms", "bv_asym", 2, N, spec, False), 1, workdir))
    g1 = _mms_level(_run_mpi(_mms_task(params, "galv", "bv_asym", 2, N, spec, False), 1, workdir))
    for o in (a1, g1):
        _check_mesh(o["mesh"], spec)
    a4, g4 = finest[("bv_asym", "p2_3d")], galv_finest["p2_3d"]
    q["ranks_1_max"] = int(max(a1["mesh"]["ranks"], g1["mesh"]["ranks"]))
    q["rank_invariance_L2_rel"] = max(abs(a1["L2"] / a4["L2"] - 1), abs(g1["L2"] / g4["L2"] - 1))
    q["rank_invariance_I_abs"] = abs(a1["Ia"] - a4["Ia"])
    q["rank_invariance_phis_abs"] = abs(g1["phi_s"] - g4["phi_s"])
    notes.append(_build_note())
    notes.append(f"every level on {RANKS} MPI ranks (single-node mpirun), the rank-invariance runs and the Taylor "
                 "tests on 1 rank; meshes verified by vertex/connectivity hash and element count")
    return dict(runs=runs, series=series, quantities=q, notes=notes)


# ------------------------------------------------------------------------------------------------ concentric 3D
def _area_exact(geom):
    if geom == "sph3d":
        return R1 ** 2 * (0.5 * math.pi) * math.cos(0.25 * math.pi)
    return R1 * 0.5 * math.pi * H_CYL


def concentric_3d(params, workdir, quick=False):
    kin_p = params["kinetics"]
    kappa = C._num(params["geometry"]["kappa"])
    runs, series, q, notes = [], {}, {}, []
    bv = {"inner": kin_p["bv_sym"]["inner (anode)"], "outer": kin_p["bv_sym"]["outer (cathode)"]}
    for geom in ("cyl3d", "sph3d"):
        for kin in ("primary", "bv_sym"):
            for k in (1, 2):
                ms = params["meshes"][f"{geom}_p{k}"]
                lv, vals, area, bal, scl = [], [], [], [], []
                for j, (N, spec) in enumerate(zip(ms["N"], ms["levels"])):
                    last = j == len(ms["N"]) - 1
                    task = dict(kind="conc", geom=geom, kin=kin, order=k, N=N, kappa=kappa, primary_phi=[0.2, 0.0],
                                bv=bv, newton=_newton(params), one_more=last and kin != "primary",
                                hash=dict(scale=spec["hash_scale"], geometry=k == 2))
                    outs = _run_mpi(task, RANKS, workdir)
                    o0 = outs[0]
                    _check_mesh(o0["mesh"], spec, geometry=k == 2)
                    I, Ic = _sum(outs, "I"), _sum(outs, "I_cathode")
                    meas = _sum(outs, "measure")
                    level = f"{geom}_{kin}_p{k}_N{N}"
                    lv.append(level)
                    runs.append(dict(level=level, mesh=_mesh_entry(o0["mesh"], ms["h_nominal"][j], ms["family"], j,
                                                                   RANKS),
                                     solver=_solver_entry(o0["report"]), seconds=max(o["seconds"] for o in outs)))
                    vals.append(I)
                    q[f"{geom}_{kin}_p{k}_N{N}_I"] = I
                    area.append(meas - _area_exact(geom))
                    bal.append(abs(I + Ic))
                    scl.append(abs(I) + abs(Ic))
                    if last:
                        p = f"{geom}_{kin}_p{k}"
                        q[p + "_I_finest"] = I
                        q[p + "_Iflux_finest"] = _sum(outs, "I_flux")
                        if "I_plus1" in o0:
                            q[p + "_I_finest_plus1"] = _sum(outs, "I_plus1")
                p = f"{geom}_{kin}_p{k}"
                series[p + "_I"] = C.functional_series(lv, vals)
                if kin == "primary":
                    series[f"{geom}_p{k}_area"] = C.order_series(lv, area, _area_exact(geom))
                q[p + "_balance_abs_max"] = float(max(bal))
                q[p + "_I_scale"] = float(max(scl))
    notes.append(_build_note())
    notes.append(f"every level on {RANKS} MPI ranks; P1 affine, P2 isoparametric (geometry-node hash verified)")
    return dict(runs=runs, series=series, quantities=q, notes=notes)


# ------------------------------------------------------------------------------------------------ porous 1D
def _porous_norms(pv, I):
    """L2/H1 norms of the closed-form fields (c3b_checks.exact) by composite Gauss-Legendre quadrature."""
    l = C._num(pv["l"])
    e1, d1, e2, d2, _ = CB.exact(pv, I)
    xg, wg = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(0, l, 201)
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * xg[None, :]).ravel()
    w = (0.5 * np.diff(edges)[:, None] * wg[None, :]).ravel()
    return (math.sqrt(float(np.sum((e1(x) ** 2 + e2(x) ** 2) * w))),
            math.sqrt(float(np.sum((d1(x) ** 2 + d2(x) ** 2) * w))))


def porous_1d(params, workdir, quick=False):
    from axionfem.chem import PorousElectrode, Reaction
    newton = dict(_newton(params), gate=1e-12, limiter_rt_f=10.0, ls_min=1 / 64, clip=200.0, temperature=298.15,
                  check=False)
    runs, series, q, notes = [], {}, {}, []
    for v, pv in params["variants"].items():
        pot = pv["control"] == "potentiostatic"
        R = CB.resistance(pv)
        I_ex = C._num(pv["V"]) / R if pot else C._num(pv["I"])
        V_ex = C._num(pv["V"]) if pot else CB.exact(pv, I_ex)[4]
        L2n, H1n = _porous_norms(pv, I_ex)
        rx = Reaction("linear", i0=C._num(pv["i0"]), E_eq=0.0, alpha_a=C._num(pv["alpha_a"]),
                      alpha_c=C._num(pv["alpha_c"]))
        for k in (1, 2):
            ms = params["meshes"][f"p{k}"]
            lv, eV, eL, eH, ex, co, sc, sp = [], [], [], [], [], [], [], []
            for j, N in enumerate(ms["N"]):
                t0 = time.time()
                pe = PorousElectrode(C._num(pv["l"]), N, order=k, degree=2 * k + 4)
                pe.setup(C._num(pv["sigma"]), C._num(pv["kappa"]), C._num(pv["a_v"]), rx,
                         I=None if pot else C._num(pv["I"]), V=C._num(pv["V"]) if pot else None)
                rep = pe.solve(**newton)
                z = np.asarray(pe.state())
                nn = pe.num_dofs()
                res = np.asarray(pe.residual_full())
                Vh = float(z[0] - z[-1])
                Ih = -C.reaction_current(res, [pe.separator_row()], False) if pot else C._num(pv["I"])
                Qe = np.asarray(pe.quadrature(2 * k + 8))
                err = CB.porous_field_errors(Qe[:, 0], Qe[:, 1], Qe[:, 2], Qe[:, 3], Qe[:, 4], Qe[:, 5], pv, I_ex)
                Qa = np.asarray(pe.quadrature(2 * k + 4))
                exch = CB.exchange_total(Qa[:, 0], Qa[:, 1], Qa[:, 2], Qa[:, 4], pv)
                coll = C.reaction_current(res, [pe.collector_row()], False)
                terms = float(pe.abs_action_sum()) + abs(Ih)
                idn = CB.identities(exch, coll, Ih, terms)
                xn = np.asarray(pe.node_x())
                spread = CB.strip_spread(xn, z[:nn], z[nn:], xn)
                level = f"{v}_p{k}_N{N}"
                lv.append(level)
                runs.append(dict(level=level, mesh=dict(h_nominal=float(ms["h_nominal"][j]), h_max=float(pe.h()),
                                                        h_eff=float(pe.h()), n_elements=int(N), dofs_field=2 * nn,
                                                        dofs_total=2 * nn, family=ms["family"], refinement_level=j),
                                 solver=_solver_entry(dict(rep, cg_iterations_max=0)), seconds=time.time() - t0))
                eV.append((Ih - I_ex) if pot else (Vh - V_ex))
                eL.append(err["L2"]), eH.append(err["H1"])
                ex.append(idn["exchange_abs"]), co.append(idn["collector_abs"]), sc.append(idn["terms"])
                sp.append(spread)
            p = f"{v}_p{k}"
            key = "I" if pot else "V"
            series[f"{p}_{key}"] = C.order_series(lv, eV, abs(I_ex if pot else V_ex))
            series[p + "_L2"] = C.order_series(lv, eL, L2n)
            series[p + "_H1"] = C.order_series(lv, eH, H1n)
            q[f"{p}_{key}_finest"] = Ih if pot else Vh
            q[p + "_exchange_abs_max"] = float(max(ex))
            q[p + "_collector_abs_max"] = float(max(co))
            q[p + "_terms"] = float(max(sc))
            q[p + "_strip_spread"] = float(max(sp))
            q[p + "_R_finest"] = Vh / Ih
    notes.append(_build_note())
    notes.append("1D interval elements (no strip): strip_spread is 0 by construction (c3b_checks.strip_spread on "
                 "the 1D lattice)")
    return dict(runs=runs, series=series, quantities=q, notes=notes)
