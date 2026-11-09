"""Out-of-band cross-code reference for F5 xcode_gls_cavity: NGSolve, singular-lid cavity, the PINNED P1/P1 GLS discrete problem.

Discrete problem (fixed in xcode_gls_cavity run.parameters): f5_gls_check.residual formulation, rho = 1,
mu = 1/Re constant, both viscous forms, Re = 100 and 1000, nested structured meshes N = 32, 64
(f1a_stokes_check.structured_mesh), lid u = (1, 0) at every lid vertex EXCEPT the lid corner vertices (0,1), (1,1), which take the
wall value 0 (the F2a corner rule, P1 nodal); u = 0 on the other walls; mean-zero pressure; Re continuation 10, 30, 100, 200, 400,
700, 1000.  Quantities (f5_gls_check.cavity_quantities from the nodal arrays): Ekin, Diss (a_form(u_h,u_h)), Enst, min of u_x on
x = 1/2, max and min of u_y on y = 1/2.

Independent second implementation: the converged NGSolve state is substituted into the numpy residual f5_gls_check.residual;
the interior velocity rows and all pressure rows must vanish (diagnostics.max_numpy_residual_rel).  Self-consistency: the N = 32
cases re-solved with another continuation schedule and a tighter tolerance.

    python -B tools/fluids_mms/xcode/ngsolve_gls_cavity_reference.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "fluids_mms"))
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import numpy as np  # noqa: E402
import ngsolve as ngs  # noqa: E402
import f5_gls_check as C  # noqa: E402
import ngs_gls as NG  # noqa: E402
from ngs_ns import ng_mesh, box_namer  # noqa: E402


def lid(X, Y):
    out = np.zeros((len(X), 2))
    on = (np.abs(Y - 1.0) < 1e-12) & (X > 1e-12) & (X < 1 - 1e-12)
    out[on, 0] = 1.0
    return out


def solve(N, Re, form, path, rtol=1e-12):
    V, T = C.structured_mesh(N)
    mesh = ng_mesh(V, T, box_namer(0, 1, 0, 1))
    mup = ngs.Parameter(1.0 / path[0])
    prob = NG.GLS(mesh, 1.0, ngs.CF(mup), ngs.CF((0.0, 0.0)), ngs.CF((0.0, 0.0)), ["x0", "x1", "y0", "y1"], lid, form=form, mean_zero=True)
    r0 = prob.r0()
    hist = []
    for r in path:
        mup.Set(1.0 / r)
        prob.newton(rtol=rtol if r == path[-1] else 1e-8, r0=r0, history=hist)
    ux, uy, p = prob.uvec()
    u = np.stack([ux, uy], -1)
    nv = len(V)
    # numpy residual of the converged state: interior velocity rows and pressure rows
    Ru, Rp = C.residual(V, T, u, p, 1.0, lambda x, y: (np.full_like(x, 1.0 / Re), np.zeros(x.shape + (2,))),
                        lambda x, y: np.zeros(x.shape + (2,)), form=form)
    interior = np.ones(nv, bool)
    for a, b in C.C1.boundary_edges(T):
        interior[[a, b]] = False
    # normalization: the same rows of the residual at the nodal lift (boundary data, zero interior velocity, p = 0)
    ul = np.zeros_like(u)
    bd = ~interior
    ul[bd] = u[bd]
    L_u, L_p = C.residual(V, T, ul, np.zeros(nv), 1.0, lambda x, y: (np.full_like(x, 1.0 / Re), np.zeros(x.shape + (2,))),
                          lambda x, y: np.zeros(x.shape + (2,)), form=form)
    scale = max(np.abs(L_u[interior]).max(), np.abs(L_p).max())
    return u, p, V, T, max(np.abs(Ru[interior]).max(), np.abs(Rp).max()) / scale, hist[-1]["residual"]


def main(out):
    Q, diag = {}, {"numpy_residual": {}, "solver_residual": {}}
    t0 = time.time()
    for Re in (100, 1000):
        path = [r for r in (10, 30, 100, 200, 400, 700, 1000) if r <= Re]
        for form in ("lap", "sym"):
            for N in (32, 64):
                u, p, V, T, nres, sres = solve(N, Re, form, path)
                q = C.cavity_quantities(V, T, u, 1.0 / Re, N)
                for k in ("Ekin", "Diss", "ux_min_vc", "uy_max_hc", "uy_min_hc"):
                    Q[f"{form}_Re{Re}_{k}_N{N}"] = q[k]
                ev = C.NodalEval(V, T, u, p)
                Q[f"{form}_Re{Re}_Enst_N{N}"] = C.C2.functionals(ev, form, 1.0 / Re)["Enst"]
                diag["numpy_residual"][f"{form}_Re{Re}_N{N}"] = nres
                diag["solver_residual"][f"{form}_Re{Re}_N{N}"] = sres
                print(form, Re, N, "%.1fs" % (time.time() - t0), nres, sres, flush=True)
    # self-consistency: another continuation schedule, tighter tolerance (N = 32)
    selfc = {}
    for Re in (100, 1000):
        for form in ("lap", "sym"):
            path = [20, 60, 100] if Re == 100 else [25, 60, 150, 300, 600, 1000]
            u, p, V, T, nres, sres = solve(32, Re, form, path, rtol=1e-13)
            q = C.cavity_quantities(V, T, u, 1.0 / Re, 32)
            for k in ("Ekin", "Diss", "ux_min_vc", "uy_max_hc", "uy_min_hc"):
                selfc[f"{form}_Re{Re}_{k}_N32"] = abs(q[k] - Q[f"{form}_Re{Re}_{k}_N32"]) / abs(Q[f"{form}_Re{Re}_{k}_N32"])
    diag["max_self_consistency_rel"] = max(selfc.values())
    diag["max_numpy_residual_rel"] = max(diag["numpy_residual"].values())
    res = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__, "case": "xcode_gls_cavity",
           "quantities": dict(sorted(Q.items())), "diagnostics": diag}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
