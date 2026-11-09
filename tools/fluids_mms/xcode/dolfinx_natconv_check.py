"""Inter-code agreement for the F4a natural-convection cross-code reference: the SAME discrete problem as
ngsolve_natconv_reference.py, solved with FEniCSx/DOLFINx 0.11, a separate code (UFL
forms, automatic differentiation for the Jacobian, PETSc SNES Newton with backtracking line search, MUMPS LU).

Only the fixed mesh builder (scalar_transport_check.box_mesh, read-only) is shared.  Problem: differentially heated square cavity, de Vahl Davis scaling (rho0 = c_p = k = 1, mu = Pr, beta g = Ra Pr), Taylor-
Hood P2/P1 + P2 temperature, Galerkin (no SUPG), standard convection, T = +1/2 on x = 0 and -1/2 on x = 1 (P2 nodal
interpolation; the corner vertices belong to the hot / cold walls), adiabatic y = 0, 1, no slip; pressure fixed at the
vertex (0, 0) (every compared functional is independent of the pressure constant); Ra continuation 1e3, 1e4, 1e5.
Output: Ekin, Diss, Tsq, vT, vleft, Nu_hot, Nu_cold (variationally consistent wall fluxes: the T-equation residual
tested with the P2 nodal indicator of the wall), for N in argv and both viscous forms.

    UCX_TLS=self,sm fenicsx/bin/python -B tools/fluids_mms/xcode/dolfinx_natconv_check.py OUT.json 32 64
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
import scalar_transport_check as S  # noqa: E402  (fixed mesh builder only)


def solve(N, form="lap", Ra=1e5, Pr=0.71, schedule=(1e3, 1e4, 1e5)):
    k = int(round(np.log2(N / 8)))
    V0, T0 = S.box_mesh(8, 8, k)
    ce = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, np.asarray(T0, np.int64), ce, np.asarray(V0, np.float64))
    P2v = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    P1 = basix.ufl.element("Lagrange", "triangle", 1)
    P2 = basix.ufl.element("Lagrange", "triangle", 2)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2v, P1, P2]))
    Vs, _ = W.sub(0).collapse()
    Qs, _ = W.sub(1).collapse()
    Ts, _ = W.sub(2).collapse()
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, fdim + 1)
    bfac = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))
    g0 = dolfinx.fem.Function(Vs)
    bdofs = dolfinx.fem.locate_dofs_topological((W.sub(0), Vs), fdim, bfac)
    bc_u = dolfinx.fem.dirichletbc(g0, bdofs, W.sub(0))
    p0 = dolfinx.fem.Function(Qs)
    pdofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Qs), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    bc_p = dolfinx.fem.dirichletbc(p0, pdofs, W.sub(1))
    hot = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    cold = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    gT = dolfinx.fem.Function(Ts)
    gT.interpolate(lambda x: np.where(np.isclose(x[0], 0.0), 0.5, np.where(np.isclose(x[0], 1.0), -0.5, 0.0)))
    hdofs = dolfinx.fem.locate_dofs_topological((W.sub(2), Ts), fdim, np.concatenate([hot, cold]))
    bc_T = dolfinx.fem.dirichletbc(gT, hdofs, W.sub(2))
    w = dolfinx.fem.Function(W)
    u, p, T = ufl.split(w)
    v, q, s = ufl.TestFunctions(W)
    bg = dolfinx.fem.Constant(msh, schedule[0] * Pr)
    mu = Pr
    G = ufl.grad(u)
    visc = (ufl.inner(G, ufl.grad(v)) if form == "lap" else 2 * ufl.inner(ufl.sym(G), ufl.sym(ufl.grad(v))))
    ey = ufl.as_vector((0.0, 1.0))
    F = ((mu * visc + ufl.inner(G * u, v) - p * ufl.div(v) - q * ufl.div(u) - bg * T * ufl.inner(ey, v)
          + ufl.inner(u, ufl.grad(T)) * s + ufl.inner(ufl.grad(T), ufl.grad(s))) * ufl.dx)
    problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=[bc_u, bc_p, bc_T],
                                                 petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                                                                "snes_rtol": 1e-13, "snes_atol": 1e-13,
                                                                "snes_max_it": 60, "ksp_type": "preonly",
                                                                "pc_factor_mat_solver_type": "mumps", "pc_type": "lu"},
                                                 petsc_options_prefix=f"nc{N}{form}_")
    its = {}
    for R in schedule:
        bg.value = R * Pr
        problem.solve()
        snes = problem.solver
        its[str(R)] = int(snes.getIterationNumber())
        if snes.getConvergedReason() <= 0:
            raise RuntimeError(f"SNES did not converge at Ra = {R}: reason {snes.getConvergedReason()}")
    uh = w.sub(0).collapse()
    Th = w.sub(2).collapse()

    def integ(e, dx=ufl.dx):
        return float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e * dx)))
    Gh = ufl.grad(uh)
    x = ufl.SpatialCoordinate(msh)
    left = ufl.conditional(ufl.lt(x[0], 0.5), 1.0, 0.0)
    out = {"Ekin": 0.5 * integ(ufl.inner(uh, uh)), "Diss": integ(ufl.inner(Gh, Gh)), "Tsq": integ(Th * Th),
           "vT": integ(uh[1] * Th), "vleft": integ(uh[1] * left)}
    for name, facets in (("hot", hot), ("cold", cold)):
        ind = dolfinx.fem.Function(Ts)
        dd = dolfinx.fem.locate_dofs_topological(Ts, fdim, facets)
        ind.x.array[:] = 0.0
        ind.x.array[dd] = 1.0
        R = integ(ufl.inner(uh, ufl.grad(Th)) * ind + ufl.inner(ufl.grad(Th), ufl.grad(ind)))
        out[f"Q_out_{name}"] = -R
    out["Nu_hot"] = -out["Q_out_hot"]
    out["Nu_cold"] = out["Q_out_cold"]
    return out, its


def main():
    out = Path(sys.argv[1])
    Ns = [int(a) for a in sys.argv[2:]] or [32, 64]
    t0 = time.time()
    res = {"code": "DOLFINx", "version": dolfinx.__version__, "quantities": {}, "iterations": {}}
    for form in ("lap", "sym"):
        for N in Ns:
            qv, its = solve(N, form)
            for k, v in qv.items():
                res["quantities"][f"{form}_{k}_N{N}"] = v
            res["iterations"][f"{form}_N{N}"] = its
            print(f"dolfinx {form} N={N}: {qv} its {its} ({time.time() - t0:.0f}s)", flush=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
