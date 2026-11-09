"""3D Taylor–Hood Stokes (beta): meshes from arrays, distributed solve, distributed evaluation by input element id.

    from axionfem import fluids
    mesh = fluids.Mesh3D.from_arrays(V, E, faces, face_attr, nodes=None)   # tets (ne,4) or hexes (ne,8, basix order)
    sol = fluids.solve_stokes_3d(mesh, nu, force, bcs, form="stress")      # collective over MPI_COMM_WORLD
    ids = sol.element_ids()                                                  # this rank's input element ids
    x, detJ = sol.geometry(ref); u, grad_u = sol.velocity(ref); p = sol.pressure(ref)   # INPUT reference points

Every callable is vectorized: f(X, Y, Z) with 1-D arrays, returning (n,) (scalars) or (n, 3) / (3, n) / a sequence
of three arrays (vectors). Boundary conditions: one per boundary attribute, {"attribute", "kind", "value"/"normal"}
with kind dirichlet (P2/Q2 nodal interpolation of value), do_nothing, traction (value g = sigma_form n, a vector field
of x only: the solver never computes a normal for it), slip / symmetry (u.n = 0 strongly with the consistent normal of
the attribute; at a node on two slip/symmetry attributes both constraints, u along the common edge; three: u = 0;
Dirichlet wins at shared nodes). Curved order-2 geometry: `nodes` (ne, 10 | 27, 3) in the basix reference-node order.
The solve partitions the mesh with METIS over MPI_COMM_WORLD (deterministic); diagnostics["comm_size"] and
["shared_dofs"] show that the solve was distributed. Units are kinematic.
"""
from __future__ import annotations

import numpy as np

from .. import _core

_f = _core._fluids
Mesh3D = _f.Mesh3D
StokesSolution3D = _f.StokesSolution3D


def _vec3(fn, vdim):
    """Normalize a vectorized callable f(X, Y, Z) to a contiguous float64 array (n,) or (n, vdim)."""
    if fn is None or not callable(fn):
        return fn

    def g(X, Y, Z):
        X = np.asarray(X, float)
        n = X.shape[0]
        r = fn(X, np.asarray(Y, float), np.asarray(Z, float))
        if vdim == 1:
            a = np.broadcast_to(np.asarray(r, float), (n,))
        else:
            if isinstance(r, (tuple, list)):
                r = np.stack([np.broadcast_to(np.asarray(c, float), (n,)) for c in r], axis=-1)
            a = np.asarray(r, float)
            if a.shape == (vdim, n) and n != vdim:
                a = a.T
            a = np.broadcast_to(a, (n, vdim))
        return np.ascontiguousarray(a, dtype=float)
    return g


def default_lu_3d() -> str:
    """The distributed STRUMPACK when the build has it, else the shared-memory one, else Eigen."""
    for b in ("strumpack-mpi", "strumpack"):
        if _f.lu_backend_available(b):
            return b
    return "eigen"


def solve_stokes_3d(mesh, nu, force=None, bcs=(), *, form="stress", solver="direct", lu=None, nullspace="auto",
                    order=2, rtol=1e-10, direct_tol=1e-10, grad_div=0.0, pressure_order=-1,
                    allow_unstable_pair=False, min_refine_steps=0, solve=True, lu_threads=1,
                    slip_corner_angle_deg=30.0, lu_krylov_fallback=True, lu_matching=False):
    """Solve 3D Taylor–Hood Stokes (collective). form "stress" (2 nu eps:eps) or "laplacian" (nu grad:grad), variable
    nu in both; grad_div gamma (viscosity units). Never raises on non-convergence (diagnostics["converged"]).

    lu_krylov_fallback (default True): when the direct LU's iterative refinement stagnates above its target (a poor
    static-pivoting factorization of a 3D saddle point), restarted GMRES preconditioned by the same LU finishes the
    solve, stopping on the residual the gate measures. diagnostics["lu_krylov_fallback"] is then True,
    ["lu_krylov_iterations"] the count and ["solver"] reads "direct+gmres:<backend>". Pass False for a strictly
    direct solve (the gate then decides). The 2D and axisymmetric solvers have no such fallback.

    lu_matching (default False): STRUMPACK's MC64 matching/scaling. On 3D Taylor-Hood saddle points the matching
    permutes the structure and leaves a plain-solve residual of 0.1 to 18 on the largest levels (measured at N = 32);
    without it the plain residual is round-off (5e-16) and the factorization is faster. When the matching-free
    factorization fails or misses the gate, the solve is repeated with the matching
    (diagnostics["lu_matching_retry"] True, ["lu_matching"] says which one produced the result). True: the 2D
    configuration (matching only)."""
    bcs = [dict(b) for b in bcs]
    for b in bcs:
        for k in ("value", "normal"):
            if callable(b.get(k)):
                b[k] = _vec3(b[k], 3)
    return _f.solve_stokes_3d(mesh, _vec3(nu, 1), _vec3(force, 3), bcs, form, solver, lu or default_lu_3d(),
                              nullspace, int(order), float(rtol), float(direct_tol), float(grad_div),
                              int(pressure_order), bool(allow_unstable_pair), int(min_refine_steps), bool(solve),
                              int(lu_threads), float(slip_corner_angle_deg), bool(lu_krylov_fallback), bool(lu_matching))


def assemble_stokes_3d(mesh, nu, bcs=(), *, form="stress", order=2, pressure_order=-1, allow_unstable_pair=False,
                       lu=None):
    """Assemble without solving (blocks() for inf-sup estimates on one rank)."""
    return solve_stokes_3d(mesh, nu, None, bcs, form=form, order=order, pressure_order=pressure_order,
                           allow_unstable_pair=allow_unstable_pair, lu=lu, solve=False)


__all__ = ["Mesh3D", "StokesSolution3D", "solve_stokes_3d", "assemble_stokes_3d", "default_lu_3d"]
