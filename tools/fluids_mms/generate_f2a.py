"""Generate the F2a Navier-Stokes MMS / exact-solution data and run their self-checks.

    python3 -B tools/fluids_mms/generate_f2a.py [--only NAME ...]

Writes tools/fluids_mms/out/<field>.json (sympy / C / numpy spellings, forcing per viscous form, the forcing
without convection / centrifugal / Coriolis term (the negative controls of the self-checks), constants) and
tools/fluids_mms/out/f2a_selfcheck_report.json.  Exit status 1 if any self-check fails.  Self-checks
(f2a_fields.selfcheck): div u = 0; forcing by two independent routes per viscous form (planar: tensor vs index
notation, symbolic; axisymmetric: cylindrical formulas vs a 3D Cartesian embedding, 60 digits); the emitted strings
parsed back; exact NS solutions (Kovasznay, Burgers) have f = 0 symbolically; a 4th-order finite-difference residual
from u, p, mu, rho only (3D Cartesian for the axisymmetric fields), with teeth (forcing without convection, the
other viscous form, without the centrifugal / Coriolis term); non-Beltrami (the convective term is not a gradient);
parity and axis regularity; u.n of both signs on the traction edge.  Kovasznay additionally through
ns_residual_fd(convective=True) of benchmarks/fluids/acceptance.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import f2a_fields as F  # noqa: E402


def harness_kovasznay(fld, em):
    sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "fluids"))
    import acceptance as harness  # noqa: E402  (benchmarks/fluids/acceptance.py)
    c = fld.coords
    uf = [F._lam(e, c) for e in fld.u]
    pf = F._lam(fld.p, c)
    a, b = F.sample_box(fld, n=11, margin=0.05)
    P = np.stack([a, b], -1)
    return harness.ns_residual_fd(
        lambda Q: np.stack([uf[0](Q[..., 0], Q[..., 1]), uf[1](Q[..., 0], Q[..., 1])], -1),
        lambda Q: pf(Q[..., 0], Q[..., 1]), lambda Q: np.zeros(Q.shape), float(fld.mu), P, convective=True)["momentum_rel"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rp = out / "f2a_selfcheck_report.json"
    report = json.loads(rp.read_text()).get("fields", {}) if (a.only and rp.exists()) else {}
    status = 0
    for mk in F.ALL:
        fld = mk()
        if a.only and fld.name not in a.only:
            continue
        t0 = time.time()
        print(f"== {fld.name}", flush=True)
        em = F.emit(fld)
        try:
            rep = F.selfcheck(fld, em)
            if fld.name.startswith("ns_kovasznay"):
                h = harness_kovasznay(fld, em)
                rep["harness_ns_residual_fd"] = h
                if not h < 1e-5:
                    raise F.CheckFailed(f"harness.ns_residual_fd = {h}")
                print(f"  ok  harness.ns_residual_fd(convective=True) = {h:.1e}")
            em["selfcheck"] = {"status": "pass", **rep}
        except F.CheckFailed as exc:
            print(f"  FAIL {exc}")
            em["selfcheck"] = {"status": "fail", "error": str(exc)}
            status = 1
        report[fld.name] = em["selfcheck"]
        F.write_json(out / f"{fld.name}.json", em)
        print(f"   ({time.time() - t0:.0f} s)", flush=True)
    F.write_json(rp, {"generator": "tools/fluids_mms/generate_f2a.py", "sympy": sp.__version__,
                      "numpy": np.__version__, "fields": report})
    return status


if __name__ == "__main__":
    sys.exit(main())
