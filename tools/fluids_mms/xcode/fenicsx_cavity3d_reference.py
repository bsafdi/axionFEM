"""Out-of-band PRIMARY cross-code reference for F1c X1 (xcode_stokes_cavity_reg_3d): DOLFINx 0.11 (conda-forge), regularized-lid Stokes cavity in the unit cube.

Produced independently of the axionFEM code, before any axionFEM run; self-contained (imports only the mesh
builder of the check module).  The second code is NGSolve (ngsolve_cavity3d_reference.py); the two
must agree to round-off.

Discrete problem (X1):
  * mesh: f1c_stokes_check.cube_tet(N), octant Kuhn, N = 8, 16;
  * Taylor-Hood P2/P1, both viscous forms, mu = 1, gamma = 0, f = 0;
  * Dirichlet u = (256 x^2 (1-x)^2 y^2 (1-y)^2, 0, 0) on z = 1, 0 elsewhere, by P2 Lagrange NODAL interpolation
    (Function.interpolate at the Lagrange points; the lid data vanish on the lid's edges, so wall and lid agree);
  * enclosed: pressure rows of the rhs projected onto 1-perp (zero here: the data are flux-free), one pressure DOF
    pinned at the fixed vertex nearest the centre (consistent system, so the velocity is the multiplier one);
    MUMPS LU; E_kin = 1/2 ||u_h||^2 and dissipation a_form(u_h, u_h) with degree-4 quadrature (exact).

    mpiexec -launcher fork -n NP fenicsx/bin/python -B tools/fluids_mms/xcode/fenicsx_cavity3d_reference.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem import petsc as fpetsc

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f1c_stokes_check as C  # noqa: E402

COMM = MPI.COMM_WORLD


def lid(Xp):
    g = 256 * Xp[0] ** 2 * (1 - Xp[0]) ** 2 * Xp[1] ** 2 * (1 - Xp[1]) ** 2
    g = np.where(np.abs(Xp[2] - 1) < 1e-12, g, 0.0)
    return np.stack([g, 0 * g, 0 * g])


def solve(N, form):
    fm = C.cube_tet(N)
    e = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    cells, x = (fm.E, fm.V) if COMM.rank == 0 else (np.empty((0, 4), np.int64), np.empty((0, 3)))
    msh = dmesh.create_mesh(COMM, cells, ufl.Mesh(e), x)
    W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,)),
                                                        basix.ufl.element("Lagrange", "tetrahedron", 1)]))
    V, _ = W.sub(0).collapse()
    Q, qmap = W.sub(1).collapse()
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    G, Gv = ufl.grad(u), ufl.grad(v)
    visc = 2 * ufl.inner(ufl.sym(G), ufl.sym(Gv)) if form == "sym" else ufl.inner(G, Gv)
    a = fem.form((visc - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx)
    L = fem.form(ufl.inner(fem.Constant(msh, np.zeros(3)), v) * ufl.dx)
    msh.topology.create_connectivity(2, 3)
    fc = dmesh.exterior_facet_indices(msh.topology)
    gh = fem.Function(V)
    gh.interpolate(lid)
    bcu = fem.dirichletbc(gh, fem.locate_dofs_topological((W.sub(0), V), 2, fc), W.sub(0))
    pc_ = fm.V[np.argmin(np.linalg.norm(fm.V - 0.5, axis=1))]
    pd = fem.locate_dofs_geometrical((W.sub(1), Q), lambda Xp: np.linalg.norm(Xp.T - pc_, axis=1) < 1e-9)
    bcp = fem.dirichletbc(fem.Function(Q), pd, W.sub(1))
    A = fpetsc.assemble_matrix(a, bcs=[bcu, bcp])
    A.assemble()
    b = fpetsc.assemble_vector(L)
    fpetsc.apply_lifting(b, [a], bcs=[[bcu]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    nown = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    qm = np.asarray(qmap, np.int64).ravel()
    own_p = qm[qm < nown]
    s = COMM.allreduce(float(b.array[own_p].sum()), op=MPI.SUM)
    n = COMM.allreduce(len(own_p), op=MPI.SUM)
    b.array[own_p] -= s / n
    fpetsc.set_bc(b, [bcu, bcp])
    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    wh = fem.Function(W)
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    r = b.duplicate()
    A.mult(wh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    uh = wh.sub(0)
    dx = ufl.Measure("dx", msh, metadata={"quadrature_degree": 4})
    Gh = ufl.grad(uh)
    ek = COMM.allreduce(fem.assemble_scalar(fem.form(0.5 * ufl.inner(uh, uh) * dx)), op=MPI.SUM)
    dform = (2 * ufl.inner(ufl.sym(Gh), ufl.sym(Gh)) if form == "sym" else ufl.inner(Gh, Gh)) * dx
    di = COMM.allreduce(fem.assemble_scalar(fem.form(dform)), op=MPI.SUM)
    return {"Ekin": float(ek), "Diss": float(di), "relres": float(r.norm() / b.norm()),
            "flux_sum_projected": s, "dofs": int(W.dofmap.index_map.size_global)}


def main():
    out = Path(sys.argv[1])
    t0 = time.time()
    q, diag = {}, {}
    for form in ("lap", "sym"):
        for N in (8, 16):
            r = solve(N, form)
            q[f"{form}_Ekin_N{N}"] = r["Ekin"]
            q[f"{form}_Diss_N{N}"] = r["Diss"]
            diag[f"{form}_N{N}"] = {k: r[k] for k in ("relres", "flux_sum_projected", "dofs")}
            if COMM.rank == 0:
                print(f"{form} N={N}: Ekin {r['Ekin']!r} Diss {r['Diss']!r} relres {r['relres']:.1e}", flush=True)
    if COMM.rank == 0:
        out.write_text(json.dumps({"format": "axionfem.fluids.xcode_output/1", "code": "DOLFINx",
                                   "version": dolfinx.__version__, "ranks": COMM.size,
                                   "case": "xcode_stokes_cavity_reg_3d", "quantities": q, "diagnostics": diag},
                                  indent=1, sort_keys=True) + "\n")
        print(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
