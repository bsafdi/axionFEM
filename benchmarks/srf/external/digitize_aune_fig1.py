"""Digitize Fig. 1 of Aune et al., PRST-AB 3, 092001 (2000) [arXiv:physics/0003011v1, page 5]: R_s of a 9-cell
TESLA cavity (1.3 GHz) vs T_c/T, open circles, with the drawn lines R_BCS and R_res = 3 nOhm.

The figure is vector graphics: the markers are read exactly from the PDF drawing commands (circle bounding-box
centres), not from pixels. Axis calibration from the frame and tick marks: x = 2 at 228.59 pt, 7 at 410.14 pt
(major ticks 2..7 checked); log10 R_s = 0 at y = 278.26 pt, 3 at y = 77.23 pt. Check: the horizontal R_res line
(y = 246.1 pt) reads 3.02 nOhm (labelled 3 nOhm).

    python -I digitize_aune_fig1.py <path to physics_0003011.pdf>   -> data/aune2000_fig1_rs.csv
"""
import sys
from pathlib import Path

import pymupdf

X0, X1, Y0, Y3 = 228.59, 410.14, 278.26, 77.23


def x2v(x):
    return 2 + 5 * (x - X0) / (X1 - X0)


def y2v(y):
    return 10 ** (3 * (Y0 - y) / (Y0 - Y3))


def main(pdf):
    page = pymupdf.open(pdf)[4]
    pts, lines, segs = [], [], []
    for d in page.get_drawings():
        r, kinds = d["rect"], [i[0] for i in d["items"]]
        inside = X0 < r.x0 < X1 and Y3 < r.y0 < Y0
        if kinds == ["c"] * 4 and inside and 2 < r.width < 4:
            pts.append((x2v(0.5 * (r.x0 + r.x1)), y2v(0.5 * (r.y0 + r.y1))))
        if inside and r.height < 0.5 and r.width > 50:
            lines.append(y2v(0.5 * (r.y0 + r.y1)))
        if kinds == ["l"] * 5 and X0 - 1 < r.x0 and r.x1 < X1 + 1 and r.width > 5 and r.height > 5:
            q = [it[2] for it in d["items"]]          # stroked as a thin parallelogram: use its long edge
            segs.append(((x2v(d["items"][1][1].x), y2v(d["items"][1][1].y)), (x2v(q[1].x), y2v(q[1].y))))
    pts.sort()
    out = Path(__file__).parent / "data/aune2000_fig1_rs.csv"
    with open(out, "w") as f:
        f.write("# Aune et al. PRST-AB 3 092001 (2000) Fig. 1 (arXiv:physics/0003011v1 p.5): 9-cell TESLA cavity, "
                "1.3 GHz; digitized from PDF vector markers by digitize_aune_fig1.py\n")
        f.write(f"# R_res line reads {', '.join(f'{v:.3f}' for v in lines)} nOhm (labelled 3 nOhm)\n")
        for a, b in sorted(segs):
            f.write(f"# R_BCS line segment: ({a[0]:.4f}, {a[1]:.4f}) -> ({b[0]:.4f}, {b[1]:.4f}) [Tc/T, nOhm]\n")
        f.write("Tc_over_T,Rs_nOhm\n")
        for x, y in pts:
            f.write(f"{x:.4f},{y:.4f}\n")
    print(len(pts), "points;", "R_res line(s):", lines)


if __name__ == "__main__":
    main(sys.argv[1])
