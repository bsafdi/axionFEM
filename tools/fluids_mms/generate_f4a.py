"""Generate the F4a scalar-transport / Boussinesq / Graetz data and run their self-checks.

    python3 -B tools/fluids_mms/generate_f4a.py [--only NAME ...]

Writes tools/fluids_mms/out/<field>.json and tools/fluids_mms/out/f4a_selfcheck_report.json.  Exit status 1 if any
self-check fails.  Checks (f4a_fields.selfcheck_*): div b = 0 / div u = 0; scalar forcing by two routes (vector
calculus vs product rule; axisymmetric also vs a 3D Cartesian embedding, 60 digits); emitted strings parse back and
evaluate; 4th-order FD residual from the primitive fields only, with teeth (no advection, kappa doubled, no buoyancy,
buoyancy sign flipped, energy without advection); parity and axis regularity; Neumann/Robin edges are outflow edges
(MMS); Robin data consistent; hydrostatic balance; Graetz modes: ODE residual and phi(1) = 0 to 1e-30 (50 digits),
Chebyshev data = series, full-PDE FD residual from the emitted data, teeth.  Uniform-flux fields: the closed-form
Nusselt numbers derived symbolically are recorded in the field data.
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
import mpmath as mp  # noqa: E402

import f4a_fields as F  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true", help="print the field names and exit")
    ap.add_argument("--no-report", action="store_true", help="do not touch f4a_selfcheck_report.json (parallel runs)")
    ap.add_argument("--merge-report", action="store_true", help="rebuild the report from the field files and exit")
    a = ap.parse_args(argv)
    if a.list:
        jobs = [m() for m in F.ALL_SCALAR + F.ALL_BOUS + F.ALL_GRAETZ]
        print(" ".join(j.name for j in jobs))
        return 0
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rp = out / "f4a_selfcheck_report.json"
    if a.merge_report:
        names = [m().name for m in F.ALL_SCALAR + F.ALL_BOUS + F.ALL_GRAETZ]
        rep = {n: json.loads((out / f"{n}.json").read_text()).get("selfcheck", {"status": "missing"}) for n in names
               if (out / f"{n}.json").exists()}
        F.write_json(rp, {"generator": "tools/fluids_mms/generate_f4a.py", "sympy": sp.__version__,
                          "mpmath": mp.__version__, "numpy": np.__version__, "fields": rep})
        bad = [n for n, v in rep.items() if v.get("status") != "pass"] + [n for n in names if n not in rep]
        print("report merged;", "FAIL/missing: " + " ".join(bad) if bad else "all fields pass")
        return 1 if bad else 0
    report = json.loads(rp.read_text()).get("fields", {}) if (a.only and rp.exists()) else {}
    status = 0
    jobs = [("scalar", m) for m in F.ALL_SCALAR] + [("bous", m) for m in F.ALL_BOUS] + [("graetz", m) for m in F.ALL_GRAETZ]
    for kind, mk in jobs:
        fld = mk()
        if a.only and fld.name not in a.only:
            continue
        t0 = time.time()
        print(f"== {fld.name}", flush=True)
        try:
            if kind == "scalar":
                em = F.emit_scalar(fld)
                if fld.kind == "uniform_flux":
                    nu = F.nusselt_uniform_flux(fld)
                    em["nusselt_closed_form"] = {"sympy": str(nu), "value": float(nu),
                                                 "definition": F.nusselt_uniform_flux.__doc__.strip()}
                    print(f"  ok  closed-form Nu = {nu} = {float(nu):.15g}")
                rep = F.selfcheck_scalar(fld, em)
            elif kind == "bous":
                em = F.emit_bous(fld)
                rep = F.selfcheck_bous(fld, em)
            else:
                em = F.emit_graetz(fld)
                rep = F.selfcheck_graetz(fld, em)
            em["selfcheck"] = {"status": "pass", **rep}
        except F.CheckFailed as exc:
            print(f"  FAIL {exc}")
            em = locals().get("em", {"name": fld.name})
            em["selfcheck"] = {"status": "fail", "error": str(exc)}
            status = 1
        report[fld.name] = em["selfcheck"]
        em["generator"] = "tools/fluids_mms/generate_f4a.py (f4a_fields.py)"
        F.write_json(out / f"{fld.name}.json", em)
        print(f"   ({time.time() - t0:.0f} s)", flush=True)
    if not a.no_report:
        F.write_json(rp, {"generator": "tools/fluids_mms/generate_f4a.py", "sympy": sp.__version__,
                          "mpmath": mp.__version__, "numpy": np.__version__, "fields": report})
    return status


if __name__ == "__main__":
    sys.exit(main())
