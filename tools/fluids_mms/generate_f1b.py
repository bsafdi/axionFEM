"""Generate the F1b (axisymmetric Stokes with swirl) MMS / closed-form data and run the self-checks.

    python3 -B tools/fluids_mms/generate_f1b.py [--only NAME ...] [--light]

Writes tools/fluids_mms/out/f1b/<field>.json (sympy / C / numpy spellings in the variables r, z; components
(u_r, u_z, u_theta); per-form forcing, stress and boundary tractions; constants) and
tools/fluids_mms/out/f1b/selfcheck_report.json.  Exit status 1 if any self-check fails.  Self-checks
(axi_stokes.selfcheck): div u = 0; parity; forcing by two independent routes (cylindrical tensor calculus and the 3D
Cartesian field) per form with gamma 0 and 1; emitted strings parse back and the C spelling agrees; a 3D Cartesian
finite-difference residual per form from u, p, mu only; the weak form (r-weighted, hoop terms) against the strong
forcing with parity-respecting test functions; the forms differ where needed; BC identities; zero net r-weighted
flux; flux-free data where claimed.
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

import axi_stokes as S  # noqa: E402
import f1b_fields as F  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--light", action="store_true", help="skip the weak-form route (C)")
    a = ap.parse_args(argv)
    out = HERE / "out" / "f1b"
    out.mkdir(parents=True, exist_ok=True)
    report, failed = {}, []
    for mk in F.ALL:
        fld = mk()
        if a.only and fld.name not in a.only:
            continue
        t0 = time.time()
        print(f"[{fld.name}]", flush=True)
        em = S.emit(fld)
        try:
            rep = S.selfcheck(fld, em, heavy=not a.light)
            rep["status"] = "pass"
        except S.CheckFailed as exc:
            print("  FAIL", exc)
            rep = {"status": "FAIL", "error": str(exc)}
            failed.append(fld.name)
        em["selfcheck"] = json.loads(json.dumps(rep, default=str))
        rep["seconds"] = round(time.time() - t0, 1)
        dest = out
        if "parity" in fld.name:            # negative controls: not written to out/
            dest = HERE / "negative_controls" / "f1b"
            dest.mkdir(parents=True, exist_ok=True)
        S.write_json(dest / f"{fld.name}.json", em)
        report[fld.name] = rep
        print(f"  ({rep['seconds']} s)", flush=True)
    prev = {}
    rp = out / "selfcheck_report.json"
    if a.only and rp.exists():
        prev = json.loads(rp.read_text()).get("fields", {})
    prev.update(report)
    S.write_json(rp, {"sympy": sp.__version__, "numpy": np.__version__, "fields": prev})
    print("FAILED:" if failed else "all self-checks passed", failed or "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
