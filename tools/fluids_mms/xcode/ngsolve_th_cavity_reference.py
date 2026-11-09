"""Cross-code reference for xcode_gls_vs_taylor_hood (out of band): NGSolve Taylor-Hood P2/P1 (the steady
Navier-Stokes solver tools/fluids_mms/ngs_ns.py), regularized-lid cavity (u_x = 16 x^2 (1 - x)^2 on y = 1, P2 nodal
interpolation), rho = 1, mu = 1/Re, Re = 100 and 1000, structured meshes N = 64 and 128
(f1a_stokes_check.structured_mesh), sym viscous form, standard convection, Re continuation 10, 30, 100, 200, 400, 700,
1000.  Quantities: Ekin, Diss, min u_x on x = 1/2, max / min u_y on y = 1/2 (the P2 field sampled on 4001 points per
centre line).  The reference quantity th_Re<Re>_<q>_N<N> carries the N = 128 value for every GLS level N (one recipe
per comparison); the N = 64 values and the uncertainty |q(128) - q(64)| are in `diagnostics`.

    python -B tools/fluids_mms/xcode/ngsolve_th_cavity_reference.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "fluids_mms"))
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import ngsolve as ngs  # noqa: E402
import f5_gls_check as C  # noqa: E402
import ngs_ns as NS  # noqa: E402


def lid_regular(X, Y):
    out = np.zeros((len(X), 2))
    on = np.abs(Y - 1.0) < 1e-12
    out[on, 0] = 16 * X[on] ** 2 * (1 - X[on]) ** 2
    return out


def cav_quantities(ev, mesh, comps):
    f = C.C2.functionals(ev, "sym", 1.0)         # Ekin, Diss with mu = 1 (scaled by the caller), Enst
    ys = np.linspace(0, 1, 4001)
    ux = np.array([comps[0](mesh(0.5, y)) for y in ys])
    xs = np.linspace(0, 1, 4001)
    uy = np.array([comps[1](mesh(x, 0.5)) for x in xs])
    return {"Ekin": f["Ekin"], "Diss_mu1": f["Diss"], "Enst": f["Enst"], "ux_min_vc": float(ux.min()),
            "y_of_ux_min": float(ys[ux.argmin()]), "uy_max_hc": float(uy.max()), "x_of_uy_max": float(xs[uy.argmax()]),
            "uy_min_hc": float(uy.min()), "x_of_uy_min": float(xs[uy.argmin()])}


def cav_th(N, Re, lid):
    """Taylor-Hood solution of the cavity at Re: two Picard steps at Re = 10, then Newton along the Re path."""
    V, T = C.structured_mesh(N)
    mesh = NS.ng_mesh(V, T, NS.box_namer(0, 1, 0, 1))
    mup = ngs.Parameter(1.0 / 10)
    prob = NS.NSProblem(mesh, "planar", ngs.CF(mup), 1.0, None, ["x0", "x1", "y0", "y1"], lid, form="sym",
                        conv="standard", mean_zero=True)
    hist = []
    r0 = prob.r0()
    prob.picard(2, history=hist)
    path = [10.0, 30.0, 100.0] if Re <= 100 else [10.0, 30.0, 100.0, 200.0, 400.0, 700.0, 1000.0]
    path = [r for r in path if r <= Re] + ([Re] if Re not in path else [])
    for r in path:
        mup.Set(1.0 / r)
        prob.newton(rtol=1e-11 if r == Re else 1e-8, r0=r0, history=hist)
    comps = list(prob.gf.components[:2])
    q = cav_quantities(prob.evaluator(), mesh, comps)
    q["Diss"] = q.pop("Diss_mu1") / Re
    return q


def main(out):
    raw, t0 = {}, time.time()
    for Re in (100, 1000):
        for N in (64, 128):
            raw[f"Re{Re}_N{N}"] = cav_th(N, Re, lid_regular)
            print(Re, N, "%.0fs" % (time.time() - t0), flush=True)
    Q, unc = {}, {}
    for Re in (100, 1000):
        for q in ("Ekin", "Diss", "ux_min_vc", "uy_max_hc", "uy_min_hc"):
            ref = raw[f"Re{Re}_N128"][q]
            unc[f"Re{Re}_{q}"] = abs(ref - raw[f"Re{Re}_N64"][q])
            for N in (32, 64, 128):
                Q[f"th_Re{Re}_{q}_N{N}"] = ref
    res = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__,
           "case": "xcode_gls_vs_taylor_hood", "quantities": dict(sorted(Q.items())),
           "diagnostics": {"raw": raw, "uncertainty_abs": unc}}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
