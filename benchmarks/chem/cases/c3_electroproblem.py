"""Runner of chem_crosscode_c3_electroproblem_primary (C3a): the PRIMARY
concentric axisymmetric problems compared per level with axi::ElectroProblem (eps -> kappa), whose values are the
reference values of the case definition (computed out of band; never read here).

A module of its own: it imports only the check module c3_checks and axionfem.chem.

Contract: fn(params, workdir, quick) -> {"runs", "series", "quantities", "notes"}.
Reads only its fixed run.parameters.  Every solve uses the fixed Newton criterion (run.parameters.newton); a level
that does not converge carries converged = false and the judge's solver gate fails it.  Every mesh is checked against
the fixed vertex hash, connectivity hash and element count (a mismatch raises: ERROR).
"""
from __future__ import annotations

import math
import time

import numpy as np

import c3_checks as C
from axionfem.chem import CurrentDistribution, meshes as M

T_K = 298.15
TWO_PI = 2 * math.pi


def _num(v):
    return C._num(v)


def _newton(params):
    n = params["newton"]
    return dict(stag=float(n["stag"]), max_iter=int(n["max_iter"]), gate=1e-12, limiter_rt_f=10.0, ls_min=1 / 64,
                clip=200.0, temperature=T_K, initial="zero", check=False)   # fixed start phi = 0; the gate judges


def _run_entry(level, cd, t0, family, k, h_nominal):
    m = cd.mesh_stats()
    m.update(h_nominal=float(h_nominal), family=family, refinement_level=int(k))
    rep = cd.report
    solver = dict(converged=bool(rep["converged"]), relative_residual=float(rep["relative_residual"]),
                  iterations=int(rep["iterations"]), residual_history=[float(v) for v in rep["residual_history"]],
                  stop=rep["stop"], limited_steps=int(rep["limited_steps"]), clamp_active=bool(rep["clamp_active"]),
                  field_residual=float(rep["field_residual"]), constraint_residual=float(rep["constraint_residual"]),
                  linear_solver=rep["linear_solver"], factor_nonzeros=int(rep["factor_nonzeros"]))
    return dict(level=level, mesh=m, solver=solver, seconds=time.time() - t0)


def _check_mesh(cd, spec):
    """Vertex hash, connectivity hash and element count of the SOLVER's mesh against the fixed level."""
    V, E = cd.mesh_vertices(), cd.mesh_elements()
    p = C.mesh_problems(spec, V, len(E))
    h = C.connectivity_hash(V, E, spec["hash_scale"])
    if h != spec["connectivity_hash"]:
        p.append(f"mesh {spec['name']}: connectivity hash {h} != fixed {spec['connectivity_hash']}")
    if p:
        raise RuntimeError("; ".join(p))


def electroproblem_primary(params, workdir, quick=False):
    """Per level and variant (pot, galv) the numbers ElectroProblem computes on the same discrete problem (the
    references): reaction currents (c3_checks.reaction_current), the strong-flux anode current on the facet rule of
    degree 2k + 6, int phi dV on the domain rule of degree 2k + 4, the floating anode's phi_s; plus the discretization
    certificates of run.parameters.discretization (domain points per element, P2 edge-node radii)."""
    newton = _newton(params)
    g, disc = params["geometry"], params["discretization"]
    r1, r2, H = _num(g["r1"]), _num(g["r2"]), _num(g["H_cyl"])
    kap = _num(params["kappa"])
    an, ca = params["electrodes"]["anode"], params["electrodes"]["cathode"]
    a_attr, c_attr = [int(an["attr"])], [int(ca["attr"])]
    I_galv = _num(params["galvanostatic"]["current"])
    runs, q, checked = [], {}, set()
    for fam in ("axi_sph_p1", "axi_sph_p2", "axi_cyl_p1", "axi_cyl_p2"):
        ms = params["meshes"][fam]
        order = 2 if fam.endswith("p2") else 1
        deg_dom, deg_flux = int(disc["domain_degree"][f"p{order}"]), int(disc["facet_degree_flux"][f"p{order}"])
        npe, node_dev = [], []
        for k, N in enumerate(ms["N"]):
            for var in ("pot", "galv"):
                if fam.startswith("axi_sph"):
                    mesh = M.annulus_sector(N, 4 * N, r1, r2, 0.0, math.pi, axisym_meridian=True)
                else:
                    mesh = M.rect(N, N, x0=r1, x1=r2, y0=0.0, y1=H)
                t0 = time.time()
                cd = CurrentDistribution(mesh, order=order, axisymmetric=True,
                                         quadrature={"domain": deg_dom, "boundary": 2 * order + 4})
                cd.set_conductivity({1: kap})
                if var == "pot":
                    cd.add_electrode("anode", a_attr, primary=True, E_eq=_num(an["E_eq"]), phi_s=_num(an["phi_s"]))
                else:
                    cd.add_electrode("anode", a_attr, primary=True, E_eq=_num(an["E_eq"]), phi_s=_num(an["phi_s"]),
                                     galvanostatic=True, current=I_galv)
                cd.add_electrode("cathode", c_attr, primary=True, E_eq=_num(ca["E_eq"]), phi_s=_num(ca["phi_s"]))
                cd.solve(**newton)
                _check_mesh(cd, ms["levels"][k])
                checked.add(ms["levels"][k]["name"])
                lv = f"{fam}_{var}_N{N}"
                runs.append(_run_entry(lv, cd, t0, ms["family"], k, ms["h_nominal"][k]))
                npe.append(cd.domain_points().shape[0] / cd.num_elements())
                if order == 2 and fam.startswith("axi_sph"):
                    rho = np.linalg.norm(np.asarray(cd.dof_coordinates()), axis=1)
                    step = (r2 - r1) / (2 * N)        # every P2 node of the fixed rule lies on rho = r1 + i step
                    node_dev.append(float(np.max(np.abs(rho - (r1 + np.rint((rho - r1) / step) * step)))))
                dq = cd.domain_quadrature(deg_dom)    # int phi dV on the FIXED degree
                q[f"{lv}_phi_mean"] = float(np.sum(np.asarray(dq["w"]) * np.asarray(dq["x"])[:, 0]
                                                   * np.asarray(dq["phi"])) * TWO_PI)
                q[f"{lv}_I_cathode"] = C.reaction_current(cd.residual_full(1), cd.electrode_dofs(1), True)
                if var == "pot":
                    q[f"{lv}_I_anode"] = C.reaction_current(cd.residual_full(0), cd.electrode_dofs(0), True)
                    bq = cd.boundary_quadrature(a_attr, deg_flux)
                    dens = kap * np.sum(np.asarray(bq["grad"]) * np.asarray(bq["normal"]), 1)
                    q[f"{lv}_Iflux_anode"] = float(np.sum(dens * np.asarray(bq["w"]) * np.asarray(bq["x"])[:, 0])
                                                   * TWO_PI)
                else:
                    q[f"{lv}_phi_s_anode"] = float(cd.phi_s()[0])
                del cd
        q[f"{fam}_domain_points_per_element_min"] = float(min(npe))
        q[f"{fam}_domain_points_per_element_max"] = float(max(npe))
        if node_dev:
            q[f"{fam}_p2_node_radius_dev_max"] = max(node_dev)
    return dict(runs=runs, series={}, quantities=q,
                notes=[f"meshes verified by hash: {len(checked)} distinct levels (each solved twice: pot, galv); "
                       f"primary only; kappa = {kap}"])
