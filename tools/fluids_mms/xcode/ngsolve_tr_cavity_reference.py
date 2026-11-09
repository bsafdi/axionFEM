"""F3a cross-code reference X1 (out of band): the start-up lid-driven cavity of field f3a_tr_cavity_x1
(regularized lid, smoothstep ramp, Re = 400) with the NGSolve reference solver (tools/fluids_mms/ngs_tr.py): BDF2 with
one BDF1 starter step at dt = 1/50, u(0) = 0, P2/P1 on f1a_stokes_check.structured_mesh(N), N = 32, 64; Newton per
step to the dual-norm relative residual 1e-13 (lap and sym) and the semi-implicit scheme (lap).  Quantities (computed
by the fixed check module f3a_transient_check.flow_functionals): Ekin, Enst, Diss at t = 1 and t = 2.

    python -B tools/fluids_mms/xcode/ngsolve_tr_cavity_reference.py OUT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ngs_tr as R  # noqa: E402

C = R.C
NT_PER_UNIT = 50
NS = (32, 64)
VARIANTS = (("lap", "newton"), ("sym", "newton"), ("lap", "semi"))


def main(out):
    p = R.ROOT / "tools/fluids_mms/out/f3a_tr_cavity_x1.json"
    F = C.load_field({"path": "tools/fluids_mms/out/f3a_tr_cavity_x1.json",
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    res = {"code": "NGSolve", "quantities": {}, "solver": {}}
    import ngsolve
    res["version"] = ngsolve.__version__
    for form, scheme in VARIANTS:
        for N in NS:
            m, *_ = R.mesh_for(F, {"builder": "structured", "N": [N]}, 0)
            for T_end in (1.0, 2.0):
                prob = R.TRProblem(m, F, form)
                ts = C.time_grid({"kind": "uniform"}, T_end, int(round(T_end * NT_PER_UNIT)))
                t0 = time.time()
                ev, d = R.run_fixed(prob, ts, 2, scheme, store="last", newton_rtol=1e-13)
                q = C.flow_functionals(ev[-1], F.mu, form)
                for k, v in q.items():
                    res["quantities"][f"{form}_{scheme}_{k}_t{int(T_end)}_N{N}"] = v
                res["solver"][f"{form}_{scheme}_t{int(T_end)}_N{N}"] = {
                    "relative_residual": d["relative_residual"], "converged": d["converged"],
                    "iterations": d["iterations"]}
                print(f"ngsolve x1 {form}/{scheme} N={N} T={T_end}: {q} res {d['relative_residual']:.1e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
