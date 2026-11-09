"""C3b MPI worker, launched by c3b_current_distribution.py on several ranks of one node (fluids_mpi.launch).

    <MPI launcher> -n R python -I -S -B c3b_mpi_worker.py TASK.json

Every rank builds the same mesh, the solver (axionfem.chem.CurrentDistribution3D) partitions it, and each rank
reduces ITS OWN raw data through the fixed check modules (c3_checks / c3b_checks: currents, loads, measures, flux
currents, per-chunk error sums, reaction currents from the owned residual rows) and writes them to
TASK.rank<r>.json.  The parent combines the per-rank numbers (sums; c3b_checks.combine_error_sums) and names them.
Nothing here judges; no reference value is visible.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

TASK = json.loads(Path(sys.argv[1]).read_text())
sys.path[:] = TASK["sys_path"]

import numpy as np                                                    # noqa: E402

import c3_checks as C                                                 # noqa: E402
import c3b_checks as CB                                               # noqa: E402
from axionfem.chem import CurrentDistribution3D, Reaction, meshes3d  # noqa: E402

CHUNK = 20000          # elements per error-sum chunk
R1, R2, H_CYL = 0.005, 0.02, 0.01


def _newton():
    n = TASK["newton"]
    return dict(stag=float(n["stag"]), max_iter=int(n["max_iter"]), gate=1e-12, limiter_rt_f=10.0, ls_min=1 / 64,
                clip=200.0, temperature=298.15, initial="zero", check=False)


def _one_more(cd, rep):
    """The state after one more undamped Newton step, or None when the solve did not converge (the judge's solver gate
    fails that level; a Jacobian the linear solver cannot handle is reported, not raised)."""
    if not rep["converged"]:
        return None
    try:
        return np.asarray(cd.newton_update())
    except RuntimeError:
        return None


def _bq(cd, attrs, deg=None):
    return cd.boundary_quadrature([int(a) for a in attrs], int(deg or cd.boundary_degree))


def _error_chunks(cd, degree, phi, grad):
    ne = cd.local_elements()
    out = []
    for e0 in range(0, max(ne, 1), CHUNK):
        q = cd.domain_quadrature(degree, e0, min(ne, e0 + CHUNK))
        out.append(CB.field_error_sums(q["x"], q["w"], q["phi"], q["grad"], phi, grad))
    return out


def _volume_load(cd, f):
    ne, tot = cd.local_elements(), 0.0
    for e0 in range(0, max(ne, 1), CHUNK):
        q = cd.domain_quadrature(cd.domain_degree, e0, min(ne, e0 + CHUNK))
        tot += C.volume_load(q["x"], q["w"], f, False)
    return tot


def _mesh_info(cd, mesh):
    out = dict(n_elements=int(cd.global_elements()), dofs_field=int(cd.global_true_dofs()),
               h_max=float(cd.h_max()), volume=float(cd.volume()), ranks=int(cd.ranks()),
               owned_cells_per_rank=[int(v) for v in cd.owned_cells_per_rank()])
    if cd.rank() == 0 and TASK.get("hash"):
        h = TASK["hash"]
        V, E = np.asarray(cd.mesh_vertices()), np.asarray(cd.mesh_elements())
        out["vertex_hash"] = C.vertex_hash(V, h["scale"])
        out["connectivity_hash"] = C.connectivity_hash(V, E, h["scale"])
        out["serial_elements"] = int(len(E))
        if h.get("geometry"):
            out["geometry_node_hash"] = C.vertex_hash(np.asarray(cd.geometry_nodes()), h["scale"])
    return out


def run_mms(galv: bool):
    t0 = time.time()
    data = C.load_json(TASK["data_path"], TASK["data_sha256"])
    v, k, N = TASK["variant"], int(TASK["order"]), int(TASK["N"])
    fns = C.mms_functions(data, v)
    var = data["variants"][v]
    mesh = meshes3d.box(N)
    cd = CurrentDistribution3D(mesh, order=k)
    cd.set_conductivity(fns["kappa"])
    cd.set_source(fns["f"])
    ins = [(int(a), C.np_fn(fc["numpy"]["flux_kappa_dphi_dn_out"], data["coords"]))
           for a, fc in data["faces"].items() if fc["role"] == "insulating"]
    for a, q in ins:
        cd.set_insulator_flux(a, q)
    Iex = {r: C._num(data["galvanostatic_I_exact"][v][r]) for r in ("anode", "cathode")}
    for role in ("anode", "cathode"):
        e = var[role]
        g = galv and role == "anode"
        cd.add_electrode(role, [int(e["attr"])], reactions=[Reaction.from_dict(r) for r in e["reactions"]],
                         galvanostatic=g, phi_s=float(e["phi_s_float"]) + (0.02 if g else 0.0),
                         current=Iex[role] if g else 0.0, g=fns[f"g_{role}"])
    rep = cd.solve(**_newton())
    ts = time.time()
    out = dict(report=rep, mesh=_mesh_info(cd, mesh), solve_seconds=ts - t0, rank=cd.rank())
    out["error_chunks"] = _error_chunks(cd, 2 * k + 8, fns["phi"], fns["grad"])
    attrs = {r: [int(var[r]["attr"])] for r in ("anode", "cathode")}
    rxs = {r: var[r]["reactions"] for r in ("anode", "cathode")}

    def currents():
        ph = cd.phi_s()
        res = {}
        for i, r in enumerate(("anode", "cathode")):
            q = _bq(cd, attrs[r])
            res[r] = C.electrode_current(q["x"], q["w"], q["phi"], ph[i], rxs[r], False, g=fns[f"g_{r}"])
        return res
    out["I_kin"] = currents()
    loads = []
    for a, qf in ins:
        q = _bq(cd, [a])
        loads.append(C.boundary_load(q["x"], q["w"], qf, False))
    loads.append(_volume_load(cd, fns["f"]))
    out["loads"] = loads
    out["measures"] = [C.electrode_measure(_bq(cd, attrs[r])["x"], _bq(cd, attrs[r])["w"], False)
                       for r in ("anode", "cathode")]
    qa = _bq(cd, attrs["anode"])
    out["I_flux_anode"] = CB.flux_current(qa["x"], qa["w"], qa["grad"], qa["normal"], fns["kappa"])
    out["phi_s"] = cd.phi_s()
    if TASK.get("one_more"):
        z, z1 = np.asarray(cd.state()), _one_more(cd, rep)
        if z1 is None:
            out["I_kin_plus1"] = {"anode": float("nan"), "cathode": float("nan")}
            out["phi_s_plus1"] = [float("nan")] * len(out["phi_s"])
        else:
            cd.set_state(z1)
            out["I_kin_plus1"] = currents()
            out["phi_s_plus1"] = cd.phi_s()
            cd.set_state(z)
    out["diagnostics"] = dict(cd.diagnostics())
    out["seconds"] = time.time() - t0
    return out


def run_conc():
    t0 = time.time()
    geom, kin, k, N = TASK["geom"], TASK["kin"], int(TASK["order"]), int(TASK["N"])
    mesh = meshes3d.sector(N, geom, k)
    cd = CurrentDistribution3D(mesh, order=k)
    cd.set_conductivity(float(TASK["kappa"]))
    if kin == "primary":
        cd.add_electrode("inner", [1], primary=True, phi_s=float(TASK["primary_phi"][0]))
        cd.add_electrode("outer", [2], primary=True, phi_s=float(TASK["primary_phi"][1]))
    else:
        for name, a in (("inner", 1), ("outer", 2)):
            e = TASK["bv"][name]
            cd.add_electrode(name, [a], reactions=[Reaction.from_dict(r) for r in e["reactions"]],
                             phi_s=float(e["phi_s"]))
    rep = cd.solve(**_newton())
    out = dict(report=rep, mesh=_mesh_info(cd, mesh), rank=cd.rank())

    def currents():
        if kin == "primary":
            res = [C.reaction_current(np.asarray(cd.residual_full(-1)), cd.attr_true_dofs([a]), False) for a in (1, 2)]
        else:
            res = []
            for name, a in (("inner", 1), ("outer", 2)):
                q = _bq(cd, [a])
                e = TASK["bv"][name]
                res.append(C.electrode_current(q["x"], q["w"], q["phi"], float(e["phi_s"]), e["reactions"], False))
        return res
    out["I"], out["I_cathode"] = currents()
    q1 = _bq(cd, [1])
    out["I_flux"] = CB.flux_current(q1["x"], q1["w"], q1["grad"], q1["normal"], float(TASK["kappa"]))
    out["measure"] = C.electrode_measure(q1["x"], q1["w"], False)
    if TASK.get("one_more") and kin != "primary":
        z, z1 = np.asarray(cd.state()), _one_more(cd, rep)
        if z1 is None:
            out["I_plus1"] = float("nan")
        else:
            cd.set_state(z1)
            out["I_plus1"] = currents()[0]
            cd.set_state(z)
    out["diagnostics"] = dict(cd.diagnostics())
    out["seconds"] = time.time() - t0
    return out


def run_taylor():
    """run.parameters.taylor: converged potentiostatic P2 N = 8 state + 0.01 N(0,1), direction N(0,1) (PCG64(7),
    perturbation first) over the solver's dof order (serial), remainders |R(z + e v) - R(z) - e J v|_2."""
    data = C.load_json(TASK["data_path"], TASK["data_sha256"])
    v = TASK["variant"]
    tp = TASK["taylor"]
    fns = C.mms_functions(data, v)
    var = data["variants"][v]
    cd = CurrentDistribution3D(meshes3d.box(int(TASK["N"])), order=int(TASK["order"]))
    if cd.ranks() != 1:
        raise RuntimeError("the Taylor test is a serial run")
    cd.set_conductivity(fns["kappa"])
    cd.set_source(fns["f"])
    for a, fc in data["faces"].items():
        if fc["role"] == "insulating":
            cd.set_insulator_flux(int(a), C.np_fn(fc["numpy"]["flux_kappa_dphi_dn_out"], data["coords"]))
    for role in ("anode", "cathode"):
        e = var[role]
        cd.add_electrode(role, [int(e["attr"])], reactions=[Reaction.from_dict(r) for r in e["reactions"]],
                         phi_s=float(e["phi_s_float"]), g=fns[f"g_{role}"])
    rep = cd.solve(**_newton())
    z = np.asarray(cd.state())
    rng = np.random.Generator(np.random.PCG64(7))
    z0 = z + float(tp["perturbation_scale"]) * rng.standard_normal(z.size)
    d = float(tp["direction_scale"]) * rng.standard_normal(z.size)
    r0 = np.asarray(cd.residual(z0))
    Jd = np.asarray(cd.jacobian_apply(z0, d))
    eps = [float(e) for e in tp["eps"]]
    rem = [float(np.linalg.norm(np.asarray(cd.residual(z0 + e * d)) - r0 - e * Jd)) for e in eps]
    t = C.taylor_test(eps, rem)
    return dict(report=rep, remainders=rem, min_slope=t["min"], slopes=t["pairwise"], rank=cd.rank())


def main():
    import c3b_current_distribution as RUN   # the runner's build-identity rules (one implementation)
    ident = RUN.core_identity()
    kind = TASK["kind"]
    res = {"mms": lambda: run_mms(False), "galv": lambda: run_mms(True), "conc": run_conc,
           "taylor": run_taylor}[kind]()
    res["build"] = ident
    Path(f"{sys.argv[1]}.rank{res['rank']}.json").write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
