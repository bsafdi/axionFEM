"""Digitize arXiv:2004.02754 (Alesini et al., QUAX hollow-sapphire resonator) Fig. 3:
radial profiles of E_z (top) and H_phi (bottom) of the pseudo-TM030 mode at mid-height (HFSS).

Usage: python3 -I quax_hollow_sapphire_2004_digitize.py /path/to/2004.02754.pdf
Fig. 3 is an embedded raster. Render at 600 dpi; axes calibrated on the tick marks
(E panel x: 0 mm at px 215.5, 29 mm at 1481.5; H panel x is offset by +3 px).
Curves: the SOLID line (with shells) is traced; the DOTTED/dashed line is the analytic empty-cavity
TM030 (E_z = J0(j03 r/R), H_phi = A J1(j03 r/R)) and is masked out (+-10 px band around the analytic
curve). The thick horizontal line at 0 and the shaded-band edges are masked out too. Dark pixels:
column scans (runs <= 12 px) + row scans (runs <= 12 px, for steep parts) -> run centres.
Also extracts the shaded sapphire bands (blue vertical edge lines).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.special import j0, j1  # noqa: E402

import quax_raster_common as rc  # noqa: E402

PDF = sys.argv[1]
PAGE, CLIP, DPI = 4, (212.17, 124.79, 399.07, 323.23), 600
a, _ = rc.render(PDF, PAGE, CLIP, DPI)
lum = a.mean(2)
J03 = 8.653727912911013
P = {
    "E": dict(x=[(215.5, 0.0), (1481.5, 29.0)],
              y=[(32.5, 1.0), (141.5, 0.75), (251, 0.5), (360.5, 0.25), (467.5, 0.0), (577.5, -0.25), (686.5, -0.5)],
              rows=(36, 683), zero=(461, 480), analytic=lambda r: j0(J03 * r / 29.0)),
    "H": dict(x=[(218.5, 0.0), (1484.5, 29.0)],
              y=[(851.5, 2.0), (961.5, 1.5), (1070.5, 1.0), (1180, 0.5), (1287.5, 0.0), (1396.5, -0.5), (1506.5, -1.0)],
              rows=(855, 1503), zero=(1279, 1298), analytic=lambda r: 2.664 * j1(J03 * r / 29.0)),
}
BAND_COLS = [(680, 689), (774, 783), (1082, 1091), (1125, 1134)]
ANNOT = (700, 40, 1130, 400)  # 'Sapphire cylindrical shells' text + arrows (E panel only)


def trace(k):
    p = P[k]
    mx, bx, rx = rc.lin_cal(p["x"])
    my, by, ry = rc.lin_cal(p["y"])
    r0, r1 = p["rows"]
    dark = lum < 150
    m = np.zeros_like(dark)
    m[r0:r1, 224:1478] = dark[r0:r1, 224:1478]
    m[p["zero"][0]:p["zero"][1], :] = False
    for c0, c1 in BAND_COLS:
        m[:, c0:c1] = False
    if k == "E":
        x0, y0, x1, y1 = ANNOT
        m[y0:y1, x0:x1] = False
    # mask analytic (dashed) curve
    xs = np.arange(m.shape[1])
    ya = (p["analytic"](mx * xs + bx) - by) / my
    yy = np.arange(m.shape[0])[:, None]
    m &= ~(np.abs(yy - ya[None, :]) <= 10)
    pts = []
    for x in range(m.shape[1]):
        rr = np.where(m[:, x])[0]
        if len(rr):
            for s in np.split(rr, np.where(np.diff(rr) > 1)[0] + 1):
                if len(s) <= 12:
                    pts.append((float(x), float(s.mean())))
    for y in range(r0, r1):
        cc = np.where(m[y])[0]
        if len(cc):
            for s in np.split(cc, np.where(np.diff(cc) > 1)[0] + 1):
                if len(s) <= 12:
                    pts.append((float(s.mean()), float(y)))
    # decimate: one point per ~4 px of arc
    pts.sort()
    keep = []
    for q in pts:
        if not keep or (q[0] - keep[-1][0]) ** 2 + (q[1] - keep[-1][1]) ** 2 >= 16:
            keep.append(q)
    vals = [[round(mx * x + bx, 3), round(my * y + by, 4)] for x, y in keep]
    return keep, vals, (mx, bx, rx), (my, by, ry), ya


out = {k: trace(k) for k in P}
mxE, bxE, _ = out["E"][2]
bands_mm = [round(mxE * (0.5 * (c0 + c1)) + bxE, 2) for c0, c1 in BAND_COLS]

rc.save_png(rc.downsample(a, 2), os.path.join(rc.REFS, "quax_hollow_sapphire_2004_fig3.png"))
groups = [(out[k][0], (255, 0, 255), "x") for k in P]
lines = [("h", p0, (0, 180, 0)) for k in P for p0, _ in P[k]["y"]] + [("v", p0, (0, 180, 0)) for p0, _ in P["E"]["x"]]
tmp = os.path.join(rc.REFS, "_tmp2004.png")
rc.overlay(a, groups, tmp, lines, r=3, t=1)
import fitz  # noqa: E402
pm = fitz.Pixmap(tmp)
b = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
rc.save_png(rc.downsample(b, 2), os.path.join(rc.REFS, "quax_hollow_sapphire_2004_fig3_overlay.png"))
os.remove(tmp)


def dig(k):
    _, _, (mx, bx, rx), (my, by, ry), _ = out[k]
    return {
        "method": "raster dark-pixel trace at 600 dpi (embedded bitmap): column + row run centres of the solid line; "
                  "analytic empty-cavity dotted curve, thick zero line, band edges and annotation masked; axes "
                  "calibrated on tick marks",
        "page": PAGE + 1, "clip_bbox_pt": list(CLIP), "dpi": DPI,
        "axis_calibration": {"x": [list(v) for v in P[k]["x"]], "y": [list(v) for v in P[k]["y"]], "log": False,
                             "fit_rms_residual": {"x": rx, "y": ry}},
        "uncertainty_x": round(3 * abs(mx), 3),
        "uncertainty_y": round(3 * abs(my), 4),
        "uncertainty_basis": "+-3 px at 600 dpi (line width ~4 px, source bitmap ~374 dpi). Gaps where the solid "
                             "line crosses the dotted analytic curve or the zero line.",
        "figure_png": "quax_hollow_sapphire_2004_fig3.png",
        "overlay_png": "quax_hollow_sapphire_2004_fig3_overlay.png",
        "script": "digitize/quax_hollow_sapphire_2004_digitize.py",
    }


R_ = {"label": "radial coordinate r (mid-height)", "unit": "mm"}
curves = [
    ("reference_values", {"source": "Fig. 3 (top)", "description": "E_z(r) of pseudo-TM030 with sapphire shells "
                          "(solid line), normalized to E_z(0)=1 [V/m as labelled]",
                          "x": R_, "y": {"label": "E_z (normalized)", "unit": "V/m (arb.)"},
                          "series": [{"label": "with sapphire shells (HFSS, solid)", "points": out["E"][1]}],
                          "digitization": dig("E")}),
    ("reference_values", {"source": "Fig. 3 (bottom)", "description": "H_phi(r) of pseudo-TM030 with sapphire shells "
                          "(solid line), same normalization as top panel",
                          "x": R_, "y": {"label": "H_phi", "unit": "mA/m (arb.)"},
                          "series": [{"label": "with sapphire shells (HFSS, solid)", "points": out["H"][1]}],
                          "digitization": dig("H")}),
]
rc.update_curves(os.path.join(rc.REFS, "quax_hollow_sapphire_2004.json"), curves)
print("bands (mm, edge-line centres):", bands_mm)
for k in P:
    print(k, len(out[k][1]), out[k][1][::25])

# reference crops of the dimension sources (read by eye, not digitized): Fig. 4 table, Fig. 10 drawing
for pg, clip, name in ((5, (160, 115, 450, 275), "fig4"), (11, (200, 320, 420, 550), "fig10")):
    c, _ = rc.render(PDF, pg, clip, 250)
    rc.save_png(c, os.path.join(rc.REFS, "quax_hollow_sapphire_2004_%s.png" % name))
