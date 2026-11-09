"""Generate the F1a MMS / closed-form data and run the self-checks.

    python3 -B tools/fluids_mms/generate.py [--only NAME ...] [--no-harness-fd]

Writes tools/fluids_mms/out/<field>.json (expressions in sympy / C / numpy spelling, boundary data per form,
constants) and tools/fluids_mms/out/selfcheck_report.json.  Exit status 1 if any self-check fails.
Self-checks (stokes2d.selfcheck): div u = 0; forcing by two independent derivations per viscous form and with
grad-div; the EMITTED forcing string, parsed back, gives zero PDE residual symbolically; the C spelling evaluates
to the same numbers; a finite-difference residual per form built from u, p, mu only; the two forms differ where
the design needs them to; BC identities (slip/symmetry u.n = 0 and zero tangential traction, do-nothing of one
form and not of the other, traction data differ between forms); zero net boundary flux; periodicity.
For constant-mu Laplacian fields, also the finite-difference residual ns_residual_fd of benchmarks/fluids/acceptance.py.
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

import stokes2d as S  # noqa: E402
import f1a_fields as F  # noqa: E402


def harness_fd(fld):
    """Cross-check with harness.ns_residual_fd (Laplacian, constant mu only)."""
    if fld.mu.free_symbols or "lap" not in fld.forms:
        return None
    root = HERE.parents[1]
    sys.path.insert(0, str(root / "benchmarks" / "fluids"))
    try:
        import acceptance as harness  # noqa: WPS433  (benchmarks/fluids/acceptance.py)
    except Exception as exc:  # pragma: no cover
        return f"harness not importable: {exc}"
    nu = float(fld.mu)
    uf = [S._np_lambda(c) for c in fld.u]
    pf = S._np_lambda(fld.p)
    em = S.emit(fld)
    ff = [S._np_lambda(sp.sympify(s["sympy"], locals={"x": S.x, "y": S.y})) for s in em["forms"]["lap"]["f"]]
    xs, ys = fld.sample()
    pts = np.stack([xs, ys], axis=-1)[:: max(1, len(xs) // 100)]
    r = harness.ns_residual_fd(lambda P: np.stack([uf[0](P[..., 0], P[..., 1]), uf[1](P[..., 0], P[..., 1])], -1),
                               lambda P: pf(P[..., 0], P[..., 1]),
                               lambda P: np.stack([ff[0](P[..., 0], P[..., 1]), ff[1](P[..., 0], P[..., 1])], -1),
                               nu, pts, convective=False)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-harness-fd", action="store_true")
    a = ap.parse_args(argv)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    report, failed = {}, []
    for mk in F.ALL:
        fld = mk()
        if a.only and fld.name not in a.only:
            continue
        t0 = time.time()
        print(f"[{fld.name}]")
        em = S.emit(fld)
        try:
            rep = S.selfcheck(fld, em)
            if not a.no_harness_fd:
                h = harness_fd(fld)
                if h is not None:
                    rep["harness_ns_residual_fd"] = h
                    # momentum_rel degenerates to 1 when every term is ~0 (f = 0, linear u): then judge the
                    # absolute FD residual (central differences with h = 1e-4 leave ~1e-8 of round-off)
                    ok = isinstance(h, dict) and (h["momentum_rel"] < 1e-5 or h["momentum_max"] < 1e-6) \
                        and h["divergence_max"] < 1e-6
                    print(f"  {'ok ' if ok else 'BAD'} harness.ns_residual_fd: {h}")
                    if not ok:
                        raise S.CheckFailed(f"[{fld.name}] harness.ns_residual_fd {h}")
            rep["status"] = "pass"
        except S.CheckFailed as exc:
            print("  FAIL", exc)
            rep = {"status": "FAIL", "error": str(exc)}
            failed.append(fld.name)
        em["selfcheck"] = dict(rep)            # deterministic content (the file's sha256 is fixed in run blocks)
        rep["seconds"] = round(time.time() - t0, 1)
        S.write_json(out / f"{fld.name}.json", em)
        report[fld.name] = rep
    prev = {}
    rp = out / "selfcheck_report.json"
    if a.only and rp.exists():
        prev = json.loads(rp.read_text()).get("fields", {})      # merge, never nest
    prev.update(report)
    S.write_json(rp, {"sympy": sp.__version__, "numpy": np.__version__, "fields": prev})
    print("FAILED:" if failed else "all self-checks passed", failed or "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
