"""Write the F3a field data files tools/fluids_mms/out/f3a_tr_*.json (exported case data) and the self-check report.

    python3 -B tools/fluids_mms/generate_f3a.py [name ...]

Every field passes its self-checks (f3a_fields.selfcheck) or nothing is written for it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f3a_fields as F  # noqa: E402

OUT = HERE / "out"


def main(argv):
    OUT.mkdir(exist_ok=True)
    want = set(argv)
    reports = {}
    rp = OUT / "f3a_selfcheck_report.json"
    if rp.exists():
        reports = json.loads(rp.read_text()).get("fields", {})
    bad = []
    for mk in F.ALL:
        fld = mk()
        if want and fld.name not in want:
            continue
        print(f"== {fld.name}", flush=True)
        em = F.emit(fld)
        rep = F.selfcheck(fld, em)
        em["selfcheck"] = {"status": rep["status"], "checks": rep["checks"]}
        reports[fld.name] = rep
        if rep["status"] != "pass":
            bad.append(fld.name)
            continue
        F.write_json(OUT / f"{fld.name}.json", em)
    import sympy, numpy, scipy, mpmath
    F.write_json(rp, {"versions": {"sympy": sympy.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
                                   "mpmath": mpmath.__version__}, "fields": reports})
    if bad:
        raise SystemExit(f"self-check FAILED: {bad}")
    print("all fields pass")


if __name__ == "__main__":
    main(sys.argv[1:])
