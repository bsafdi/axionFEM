"""Stern et al. 1603.06990 design B form-factor convention (beta).

    python benchmarks/cavity/repro/stern_B_variants.py

The reproduction on docs/validation/cavity_xsec.md finds: designs A, C, D, F, G agree with the digitized Fig. 6 C when C is
normalized by the full cylinder (C_full, 0.3-1.6 % rms), design B agrees with the vacuum-area C instead (4.5 %; 4 of
5 points <= 0.2 %), and design E (Table I post diameter 0.375 R) is matched by 0.275 R (likely Table I typo).
Question: is B, like E, a Table I misprint (a geometry for which C_full matches together with the Lambda range and
Q0), or a genuinely different normalization? Discrete candidates (no continuous fit): the Table I row; post
diameters 0.25 / 0.275 / 0.325 / 0.35 R; post-location radii (R_MOV = R_STAT) 0.413 / 0.454 R (the A and C values); the A and C rows themselves
(a legend swap). Each is swept like benchmarks/cavity/xsec_stern.py and compared along Lambda
with the digitized Fig. 6a C (both conventions) and Q0.
Writes benchmarks/records/cavity_repro_stern_B_variants.json.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# xsec_stern reads its reference file and work directory from sys.argv at import
sys.argv = [sys.argv[0], str(ROOT / "benchmarks/cavity/refs/stern_multipost_1603.json"),
            os.environ.get("AXF_REPRO_WORK", str(ROOT / "runs" / "cavity_repro")) + "/stern_B_variants"]
sys.path.insert(0, str(ROOT / "benchmarks/cavity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_stern as st  # noqa: E402
import _common as cm  # noqa: E402

CANDS = {"table": [2, 2, 0.425, 0.425, 0.300], "dia0.25": [2, 2, 0.425, 0.425, 0.25],
         "dia0.275": [2, 2, 0.425, 0.425, 0.275], "dia0.325": [2, 2, 0.425, 0.425, 0.325],
         "dia0.35": [2, 2, 0.425, 0.425, 0.35], "r0.413": [2, 2, 0.413, 0.413, 0.3], "r0.454": [2, 2, 0.454, 0.454, 0.3],
         "A_row": [2, 2, 0.413, 0.413, 0.35], "C_row": [2, 2, 0.454, 0.454, 0.35]}


def main():
    pc = st.digitized("Fig. 6a (middle panel)", "design B")
    pq = st.digitized("Fig. 6a (top panel)", "design B")
    rms = lambda c: float(np.sqrt(np.mean([(e["ours"] / e["paper"] - 1) ** 2 for e in c]))) if c else None  # noqa: E731
    out = {"paper_Lambda_range": [min(p[0] for p in pc), max(p[0] for p in pc)], "candidates": {}}
    for lab, row in CANDS.items():
        st.DESIGNS["B"] = row
        st.WD = Path(sys.argv[2]) / lab
        rows = st.sweep_design("B")
        o = {"row": row, "Lambda": [min(r["Lambda"] for r in rows), max(r["Lambda"] for r in rows)],
             "rms_C_vac": rms(st.compare(rows, pc, "C")), "rms_C_full": rms(st.compare(rows, pc, "C_full")),
             "rms_Q0": rms(st.compare(rows, pq, "Q0")), "n_cmp": len(st.compare(rows, pc, "C")),
             "pointwise_C_vac": st.compare(rows, pc, "C"), "pointwise_C_full": st.compare(rows, pc, "C_full")}
        print(lab, {k: v for k, v in o.items() if not k.startswith("pointwise")}, flush=True)
        out["candidates"][lab] = o
    cm.write_record("stern_B_variants", out)


if __name__ == "__main__":
    main()
