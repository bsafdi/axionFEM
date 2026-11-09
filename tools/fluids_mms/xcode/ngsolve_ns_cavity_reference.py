"""Out-of-band cross-code reference for F2a xcode_cavity_re1000: NGSolve, singular-lid cavity, Re = 1000.

Produced independently of the axionFEM code, before any axionFEM run.

Discrete problem (fixed in xcode_cavity_re1000 run.parameters):
  * mesh: f1a_stokes_check.structured_mesh(N), N = 64, 128 (fixed builder, imported read-only via f2a_ns_check);
  * Taylor-Hood P2/P1, rho = 1, mu = 1/1000, f = 0, both viscous forms, both convective forms (standard,
    skew = Temam), mean-zero pressure by a scalar multiplier;
  * lid: u = (1, 0) at every lid node (vertices and edge midpoints) EXCEPT the two lid corner vertices (0,1) and
    (1,1), which carry the wall value 0 (fixed corner rule); u = 0 on the other walls; P2 nodal interpolation;
  * Re continuation 100, 200, 400, 700, 1000 (Newton with Armijo line search at each step), relative residual of the
    steady system <= 1e-12 at Re = 1000;
  * quantities by f2a_ns_check.functionals (degree-10 rule): Ekin = (1/2)||u_h||^2, Diss = a_form(u_h, u_h),
    Enst = ||omega_h||^2.
Self-consistency diagnostics (stored, not compared): at N = 64 the lap/standard solution is recomputed with the
hand-written Jacobian and a 1e-13 Newton tolerance, and with the continuation schedule
250, 500, 1000; the relative changes of the quantities bound the precision claimed for the reference.

    python -B tools/fluids_mms/xcode/ngsolve_ns_cavity_reference.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ngs_ns as S  # noqa: E402
import ngsolve as ngs  # noqa: E402
import numpy as np  # noqa: E402

C = S.C


def problem(N, form, conv):
    V, T = C.structured_mesh(N)
    mesh = S.ng_mesh(V, T, S.box_namer(0, 1, 0, 1))

    def g(A, B):
        top = (np.abs(B - 1) < 1e-12) & (A > 1e-12) & (A < 1 - 1e-12)
        return np.stack([np.where(top, 1.0, 0.0), 0 * A], -1)
    mu = ngs.Parameter(1e-2)
    prob = S.NSProblem(mesh, "planar", mu, 1.0, None, ["x0", "x1", "y0", "y1"], g, form=form, conv=conv,
                       mean_zero=True)
    return prob, mu


def solve(N, form, conv, schedule=(100, 200, 400, 700, 1000), rtol=1e-12, symbolic=True):
    prob, mu = problem(N, form, conv)
    prob.picard(1)
    hist = {}
    for Re in schedule:
        mu.Set(1.0 / Re)
        h = prob.newton(rtol=rtol, r0=prob.r0(), maxit=40, symbolic=symbolic)
        hist[str(Re)] = [x["residual"] for x in h]
    q = C.functionals(prob.evaluator(), form, 1e-3)
    return q, hist, prob


def _task(t):
    """One independent solve (a separate process: the variants share nothing)."""
    ngs.SetNumThreads(1)
    kind, N, form, conv = t
    t0 = time.time()
    if kind == "ref":
        q, hist, prob = solve(N, form, conv)
        return t, q, {"final_residual": hist["1000"][-1], "newton_history": hist, "nodal_defect": prob.nodal_defect,
                      "ndof": prob.ndof(), "seconds": round(time.time() - t0, 1)}
    if kind == "hand_jacobian_rtol1e-13":
        q, hist, prob = solve(N, form, conv, rtol=1e-13, symbolic=False)
    else:
        q, hist, prob = solve(N, form, conv, schedule=(250, 500, 1000))
    return t, q, {"final_residual": hist["1000"][-1]}


def main():
    out_path = Path(sys.argv[1])
    t0 = time.time()
    tasks = [("ref", N, f, c) for N in (128, 64) for f in ("lap", "sym") for c in ("standard", "skew")]
    tasks += [("hand_jacobian_rtol1e-13", 64, "lap", "standard"), ("schedule_250_500_1000", 64, "lap", "standard")]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        results = pool.map(_task, tasks)
    quantities, diag, var = {}, {}, {}
    for (kind, N, form, conv), q, d in results:
        if kind == "ref":
            for k, v in q.items():
                quantities[f"{form}_{conv}_{k}_N{N}"] = v
            diag[f"{form}_{conv}_N{N}"] = d
            print(f"{form}/{conv} N={N}: {q} res {d['final_residual']:.1e} ({d['seconds']}s)", flush=True)
    base = {k: quantities[f"lap_standard_{k}_N64"] for k in ("Ekin", "Diss", "Enst")}
    for (kind, N, form, conv), q, d in results:
        if kind != "ref":
            var[kind] = {k: abs(q[k] - base[k]) / abs(base[k]) for k in base}
    diag["self_consistency_rel"] = var
    diag["max_self_consistency_rel"] = max(v for dd in var.values() for v in dd.values())
    out = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__,
           "case": "xcode_cavity_re1000", "quantities": dict(sorted(quantities.items())), "diagnostics": diag}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({time.time() - t0:.0f} s); self-consistency {diag['max_self_consistency_rel']:.1e}")


if __name__ == "__main__":
    main()
