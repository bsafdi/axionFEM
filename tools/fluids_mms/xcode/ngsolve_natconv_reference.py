"""Natural-convection cross-code reference (out of band; reference values of xcode_natconv_ra1e5): the fixed discrete
problem of that case solved with NGSolve, written independently of axionFEM (tools/fluids_mms/ngs_heat.py,
cavity_problem): differentially heated square cavity, Taylor-Hood P2/P1 + P2 temperature on the structured meshes
scalar_transport_check.box_mesh(8, 8, k), Ra continuation 1e3, 1e4, 1e5 at Pr = 0.71, Newton to a dual-norm residual
of 1e-12, N = 32, 64, both viscous forms.  The quantities are evaluated by the fixed check module f4a_heat_check
(cavity_functionals) and the consistent wall fluxes from the residual vector (vertex-hat weights of the walls).  The
DOLFINx script dolfinx_natconv_check.py recomputes them independently.

    python -B tools/fluids_mms/xcode/ngsolve_natconv_reference.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ngs_heat as NH  # noqa: E402
import ngsolve as ngs  # noqa: E402

S, H = NH.S, NH.H
KEYS = ("Ekin", "Diss", "Tsq", "vT", "vleft", "Nu_hot", "Nu_cold")


def cavity_run(N_, Ra, Pr=0.71, form="lap"):
    """Steady cavity at (Ra, Pr) on the N_ x N_ structured mesh, reached by Ra continuation from 1e3."""
    k = {8: 0, 16: 1, 32: 2, 64: 3, 128: 4}[N_]
    V, T = S.box_mesh(8, 8, k)
    m = NH.make_mesh(V, T, (0, 1, 0, 1), False)
    sched = [Ra] if Ra <= 1e3 else [r for r in (1e3, 1e4, 1e5, 1e6, 1e7) if r < Ra] + [Ra]
    prev = None
    hist = []
    for R in sched:
        pr = NH.cavity_problem(m, R, Pr, form)
        if prev is not None:
            pr.gf.vec.data = prev.gf.vec
        pr.solve(rtol=1e-12, line_search=True)
        hist.append({"Ra": R, "newton": H.newton_iterations(pr.history), "res": pr.relative_residual})
        prev = pr
    fun = H.cavity_functionals(pr.evaluator())
    q_hot, q_cold = pr.T_outward_flux(["x0"]), pr.T_outward_flux(["x1"])
    fun.update({"Q_out_hot": q_hot, "Q_out_cold": q_cold, "Nu_hot": H.nusselt(-q_hot, 1.0, 1.0),
                "Nu_cold": H.nusselt(q_cold, 1.0, 1.0),
                "balance_defect": H.balance_identity([q_hot, q_cold], fun["adv"]),
                "relative_residual": pr.relative_residual, "converged": pr.converged, "continuation": hist})
    return fun


def main():
    out = Path(sys.argv[1])
    t0 = time.time()
    res = {"code": "NGSolve", "version": ngs.__version__, "quantities": {}, "details": {}}
    for form in ("lap", "sym"):
        for N in (32, 64):
            r = cavity_run(N, 1e5, 0.71, form=form)
            for k in KEYS:
                res["quantities"][f"{form}_{k}_N{N}"] = r[k]
            res["details"][f"{form}_N{N}"] = {"relative_residual": r["relative_residual"],
                                              "continuation": r["continuation"], "balance_defect": r["balance_defect"]}
            print(f"ngsolve {form} N={N}: " + ", ".join(f"{k}={r[k]:.15g}" for k in KEYS) + f" ({time.time() - t0:.0f}s)",
                  flush=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
