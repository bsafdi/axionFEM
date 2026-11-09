"""Second independent code for the F13a cross-code reference (out of band): FEniCSx (dolfinx 0.11).

The discrete problem of xcode_mhd_duct_2d (X1): Shercliff and Hunt square ducts, Ha = 100,
mu = 0.5, sigma = 2, G = 1, B = (0, B0, 0), on f13a_mhd_check.duct_mesh(Ha, k) (fixed builder; vertex/triangle
arrays read from the JSON written by ngsolve_mhd_duct_reference.py so FEniCSx needs no import of the check module).
Unknowns w = u_z in P2 and phi in P2 (the in-plane velocity and the pressure are identically zero/constant in the
discrete 2.5D problem: their equations have zero data), Galerkin:
    mu (grad w, grad v) + sigma B0 ((grad phi)_x + B0 w, v) = (G, v)
    sigma (grad phi, grad psi) + sigma B0 (w, psi_x) = 0       [J = sigma(-grad phi - B0 w e_x); -(J, grad psi) = 0]
Hunt: phi = 0 on y = +-1; Shercliff: insulating everywhere, phi pinned at one vertex (the system is compatible, so
the solution equals the mean-value-constrained one up to a constant; only potential DIFFERENCES are compared).
MUMPS LU.  Outputs Q_h = int w_h, w_h(0,0), phi_h(1,0) [- phi_h(-1,0)], M_var = int J_h.grad chi.

    fenicsx/bin/python -I tools/fluids_mms/xcode/fenicsx_mhd_duct_check.py MESHES.json OUT.json
"""
import json
import sys

import numpy as np
from mpi4py import MPI
import basix.ufl
import dolfinx
import ufl
from dolfinx import fem, mesh as dmesh, geometry
from dolfinx.fem.petsc import LinearProblem

MU, SIGMA, G = 0.5, 2.0, 1.0


def evaluate(f, pts, msh):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    p3 = np.c_[pts, np.zeros(len(pts))]
    coll = geometry.compute_collisions_points(tree, p3)
    cells = geometry.compute_colliding_cells(msh, coll, p3)
    out = []
    for i in range(len(pts)):
        c = cells.links(i)
        out.append(float(f.eval(p3[i:i + 1], c[:1])[0]))
    return out


def run(V, T, kind, Ha):
    B0 = Ha / np.sqrt(SIGMA / MU)
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dmesh.create_mesh(MPI.COMM_WORLD, np.asarray(T, np.int64), ufl.Mesh(el), np.asarray(V, float))
    P2 = basix.ufl.element("Lagrange", "triangle", 2)
    W = fem.functionspace(msh, basix.ufl.mixed_element([P2, P2]))
    (w, ph), (v, ps) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    s, b = fem.Constant(msh, SIGMA), fem.Constant(msh, B0)
    a = (MU * ufl.inner(ufl.grad(w), ufl.grad(v)) + s * b * (ph.dx(0) + b * w) * v
         + s * ufl.inner(ufl.grad(ph), ufl.grad(ps)) + s * b * w * ps.dx(0)) * ufl.dx
    L = fem.Constant(msh, G) * v * ufl.dx
    Ws, _ = W.sub(0).collapse()
    Ps, _ = W.sub(1).collapse()
    fdim = msh.topology.dim - 1
    walls = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(np.abs(x[0]), 1) | np.isclose(np.abs(x[1]), 1))
    zw = fem.Function(Ws)
    bcs = [fem.dirichletbc(zw, fem.locate_dofs_topological((W.sub(0), Ws), fdim, walls), W.sub(0))]
    zp = fem.Function(Ps)
    if kind == "hunt":
        hw = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(np.abs(x[1]), 1))
        bcs.append(fem.dirichletbc(zp, fem.locate_dofs_topological((W.sub(1), Ps), fdim, hw), W.sub(1)))
    else:
        pin = fem.locate_dofs_geometrical((W.sub(1), Ps), lambda x: np.isclose(x[0], -1) & np.isclose(x[1], -1))
        bcs.append(fem.dirichletbc(zp, pin, W.sub(1)))
    prob = LinearProblem(a, L, bcs=bcs, petsc_options_prefix=f"mhd_{kind}_",
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    sol = prob.solve()
    wh, phh = sol.sub(0).collapse(), sol.sub(1).collapse()
    Q = msh.comm.allreduce(fem.assemble_scalar(fem.form(wh * ufl.dx)), op=MPI.SUM)
    x = ufl.SpatialCoordinate(msh)
    Jx = s * (-phh.dx(0) - b * wh)
    Jy = -s * phh.dx(1)
    Mv = fem.assemble_scalar(fem.form((Jx * (x[1] + 1) / 2 + Jy * x[0] / 2) * ufl.dx))
    wc = evaluate(wh, np.array([[0.0, 0.0]]), msh)[0]
    pr, pl = evaluate(phh, np.array([[1.0, 0.0], [-1.0, 0.0]]), msh)
    out = {"Q_h": float(Q), "w_center_h": wc}
    if kind == "hunt":
        out.update({"phi_side_h": pr, "M_var_h": float(Mv)})
    else:
        out["dphi_side_h"] = pr - pl
    return out


def main():
    meshes = json.load(open(sys.argv[1]))
    res = {"code": "FEniCSx/dolfinx", "version": dolfinx.__version__, "quantities": {}}
    for key, m in meshes["meshes"].items():
        kind, Ha = m["kind"], m["Ha"]
        r = run(m["V"], m["T"], kind, Ha)
        for q, val in r.items():
            res["quantities"][f"{key}_{q}"] = val
        print(key, r, flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
