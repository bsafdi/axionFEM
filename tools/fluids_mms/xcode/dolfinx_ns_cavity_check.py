"""Inter-code agreement for the F2a cavity cross-code reference: the SAME discrete problem as
ngsolve_ns_cavity_reference.py, solved with FEniCSx/DOLFINx 0.11, a separate code
(UFL forms with automatic differentiation for the Jacobian, PETSc SNES Newton with backtracking line search, MUMPS LU).

Only the fixed mesh builder (f1a_stokes_check.structured_mesh, read-only) is shared.  Lid: u = (1, 0) at every
lid node except the two lid corner vertices (wall value 0), P2 nodal interpolation; pressure fixed at the vertex
(0, 0) (the velocity, hence every compared functional, does not depend on the pressure constant).  Output: the same
quantities for N in argv, both viscous forms, standard and skew (Temam) convection.

    UCX_TLS=self,sm fenicsx/bin/python -B tools/fluids_mms/xcode/dolfinx_ns_cavity_check.py OUT.json 32 64
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "fluids" / "cases"))
import f1a_stokes_check as C1  # noqa: E402  (fixed mesh builder only)


def solve(N, conv, form="lap", schedule=(100, 200, 400, 700, 1000)):
    V0, T0 = C1.structured_mesh(N)
    ce = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, np.asarray(T0, np.int64), ce, np.asarray(V0, np.float64))
    P2 = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    P1 = basix.ufl.element("Lagrange", "triangle", 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    Vs, _ = W.sub(0).collapse()
    Qs, _ = W.sub(1).collapse()
    fdim = msh.topology.dim - 1
    bfac = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))
    g = dolfinx.fem.Function(Vs)

    def lid(x):
        top = (np.abs(x[1] - 1) < 1e-12) & (x[0] > 1e-12) & (x[0] < 1 - 1e-12)
        return np.stack([np.where(top, 1.0, 0.0), 0 * x[0]])
    g.interpolate(lid)
    bdofs = dolfinx.fem.locate_dofs_topological((W.sub(0), Vs), fdim, bfac)
    bc_u = dolfinx.fem.dirichletbc(g, bdofs, W.sub(0))
    p0 = dolfinx.fem.Function(Qs)
    pdofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Qs), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    bc_p = dolfinx.fem.dirichletbc(p0, pdofs, W.sub(1))
    w = dolfinx.fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    mu = dolfinx.fem.Constant(msh, 1.0 / schedule[0])
    adv = ufl.dot(ufl.grad(u), u)
    visc = (ufl.inner(ufl.grad(u), ufl.grad(v)) if form == "lap" else 2 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))))
    F = (mu * visc + ufl.inner(adv, v) - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx
    if conv == "skew":
        F += 0.5 * ufl.div(u) * ufl.inner(u, v) * ufl.dx
    problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=[bc_u, bc_p],
                                                 petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                                                                "snes_rtol": 1e-13, "snes_atol": 1e-14,
                                                                "snes_max_it": 60, "ksp_type": "preonly",
                                                                "pc_factor_mat_solver_type": "mumps",
                                                                "pc_type": "lu"},
                                                 petsc_options_prefix=f"cav{N}{conv}{form}_")
    its = {}
    for Re in schedule:
        mu.value = 1.0 / Re
        problem.solve()
        snes = problem.solver
        its[str(Re)] = int(snes.getIterationNumber())
        if snes.getConvergedReason() <= 0:
            raise RuntimeError(f"SNES did not converge at Re = {Re}: reason {snes.getConvergedReason()}")
    uh = w.sub(0).collapse()

    def integ(e):
        return float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e * ufl.dx)))
    om = uh[1].dx(0) - uh[0].dx(1)
    G = ufl.grad(uh)
    diss = integ(ufl.inner(G, G)) if form == "lap" else 2 * integ(ufl.inner(ufl.sym(G), ufl.sym(G)))
    return {"Ekin": 0.5 * integ(ufl.inner(uh, uh)), "Diss": 1e-3 * diss, "Enst": integ(om * om)}, its


def main():
    out = Path(sys.argv[1])
    Ns = [int(a) for a in sys.argv[2:]] or [32, 64]
    t0 = time.time()
    res = {"code": "DOLFINx", "version": dolfinx.__version__, "quantities": {}, "iterations": {}}
    for form in ("lap", "sym"):
        for conv in ("standard", "skew"):
            for N in Ns:
                q, its = solve(N, conv, form)
                for k, v in q.items():
                    res["quantities"][f"{form}_{conv}_{k}_N{N}"] = v
                res["iterations"][f"{form}_{conv}_N{N}"] = its
                print(f"dolfinx {form}/{conv} N={N}: {q} its {its} ({time.time() - t0:.0f}s)", flush=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
