"""Digitize arXiv:2201.04223 (Di Vora et al., QUAX two-shell sapphire resonator) Fig. 3.

Usage: python3 -I quax_two_shell_2201_digitize.py /path/to/2201.04223.pdf
Fig. 3 is an embedded raster (sliced into strips). Method: render the figure clip
at 600 dpi, calibrate both axes (main and inset) by least squares on the light-gray
gridline pixel positions, detect markers by color mask:
  red filled diamonds  = simulation (Table I values; inset f shifted by +121 MHz? see caption)
  black '+'            = measured, cooling run 1
  blue hollow diamonds = measured, cooling run 2
  light-gray '+'       = measured points affected by mode mixing (shown greyed in the paper)
Writes REFS/quax_two_shell_2201_fig3.png, ..._fig3_overlay.png and updates the curves in
REFS/quax_two_shell_2201.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import ndimage as nd  # noqa: E402

import quax_raster_common as rc  # noqa: E402

PDF = sys.argv[1]
PAGE, CLIP, DPI = 4, (315, 300, 560, 505), 600
a, _ = rc.render(PDF, PAGE, CLIP, DPI)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
red = (R > 180) & (G < 90) & (B < 90)
blue = (B > 140) & (R < 120) & (B - R > 60)
black = (R < 90) & (G < 90) & (B < 90)
gray = (abs(R - G) < 8) & (abs(G - B) < 8) & (R > 150) & (R < 215)

# gridline pixel positions (600 dpi, clip origin), read from 1-px gray (228,228,228) lines
MX = [(208, 0), (497, 50), (786, 100), (1075, 150), (1363.5, 200), (1652.5, 250), (1941.5, 300)]
MY = [(182, 11e6), (299, 10e6), (416, 9e6), (533, 8e6), (650, 7e6), (767, 6e6), (884.5, 5e6),
      (1001, 4e6), (1118, 3e6), (1235.5, 2e6), (1352.5, 1e6)]
IX = [(1100.5, 0), (1238, 50), (1374, 100), (1511, 150), (1648, 200), (1784, 250), (1921, 300)]
IY = [(246, 10.350), (331.5, 10.345), (417, 10.340), (502, 10.335), (587, 10.330), (672.5, 10.325)]
cal = {k: rc.lin_cal(v) for k, v in dict(MX=MX, MY=MY, IX=IX, IY=IY).items()}

MAIN = (212, 135, 1966, 1455)
INSET = (1103, 160, 1920, 700)


def in_main(p):
    x, y = p
    return MAIN[0] < x < MAIN[2] and MAIN[1] < y < MAIN[3] and not (x > 885 and y < 812)


def in_inset(p):
    x, y = p
    return INSET[0] < x < INSET[2] and INSET[1] < y < INSET[3]


def diamonds(mask):
    m = nd.binary_dilation(mask, iterations=1)
    m = nd.binary_fill_holes(m)
    st = nd.generate_binary_structure(2, 1)
    core = nd.binary_erosion(m, structure=st, iterations=6)
    lab, n = nd.label(core)
    return [(float(x), float(y)) for y, x in nd.center_of_mass(core, lab, range(1, n + 1))]


pts = {
    "red": rc.blobs(red, min_px=60),
    "black": rc.crosses(black, L=17),
    "gray": rc.crosses(gray, L=15),
    "blue": rc.ring_template_peaks(nd.binary_dilation(blue, iterations=1), 11, 15, thr=0.5, nms=8),
}


def tov(p, kx, ky):
    (mx, bx, _), (my, by, _) = cal[kx], cal[ky]
    return [round(mx * p[0] + bx, 2), float("%.6g" % (my * p[1] + by))]


def series(key, region, kx, ky):
    s = [tov(p, kx, ky) for p in pts[key] if region(p)]
    return sorted(s)


main = {k: series(k, in_main, "MX", "MY") for k in pts}
ins = {k: series(k, in_inset, "IX", "IY") for k in ("red", "black")}

# PNGs: crop at 600 dpi downsampled x2 (=300 dpi) for the figure; overlay at full 600 dpi then x2
fig_png = os.path.join(rc.REFS, "quax_two_shell_2201_fig3.png")
ov_png = os.path.join(rc.REFS, "quax_two_shell_2201_fig3_overlay.png")
rc.save_png(rc.downsample(a, 2), fig_png)
groups = []
for key, col, shp in [("red", (0, 200, 0), "s"), ("black", (255, 0, 255), "x"),
                      ("blue", (255, 140, 0), "s"), ("gray", (0, 160, 255), "x")]:
    groups.append(([p for p in pts[key] if in_main(p) or (key in ("red", "black") and in_inset(p))], col, shp))
lines = [("v", p, (0, 120, 255)) for p, _ in MX + IX] + [("h", p, (0, 120, 255)) for p, _ in MY + IY]
tmp = os.path.join(rc.REFS, "_tmp_overlay.png")
rc.overlay(a, groups, tmp, lines)
import fitz  # noqa: E402
pm = fitz.Pixmap(tmp)
b = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
rc.save_png(rc.downsample(b, 2), ov_png)
os.remove(tmp)

px_main_x, px_main_y = abs(cal["MX"][0]), abs(cal["MY"][0])
px_in_x, px_in_y = abs(cal["IX"][0]), abs(cal["IY"][0])


def dig(kx, ky, ux, uy, basis):
    return {
        "method": "raster color mask at 600 dpi (figure is an embedded bitmap); marker centroids "
                  "(filled diamonds: connected components; '+' markers: pixels with full horizontal and "
                  "vertical arm runs; hollow diamonds: hole-filled, eroded cores); linear axes "
                  "least-squares calibrated on gridline pixels",
        "page": PAGE + 1, "clip_bbox_pt": list(CLIP), "dpi": DPI,
        "axis_calibration": {"x": [list(v) for v in dict(MX=MX, IX=IX)[kx]],
                             "y": [list(v) for v in dict(MY=MY, IY=IY)[ky]],
                             "log": False,
                             "fit_rms_residual": {"x": cal[kx][2], "y": cal[ky][2]}},
        "uncertainty_x": ux, "uncertainty_y": uy, "uncertainty_basis": basis,
        "figure_png": "quax_two_shell_2201_fig3.png",
        "overlay_png": "quax_two_shell_2201_fig3_overlay.png",
        "script": "digitize/quax_two_shell_2201_digitize.py",
    }


u_main = dig("MX", "MY", round(3 * px_main_x, 2), float("%.2g" % (3 * px_main_y)),
             "+-3 px at 600 dpi (gridline calibration <1 px, marker centroid ~2 px; overlapping markers "
             "can merge or drop: point COUNT is not exact)")
u_in = dig("IX", "IY", round(3 * px_in_x, 2), float("%.2g" % (3 * px_in_y)),
           "+-3 px at 600 dpi; in the flat low-T region crosses overlap heavily, so some are missing")

X = {"label": "T", "unit": "K"}
Q = {"label": "Q0 (unloaded)", "unit": "1"}
F = {"label": "f(TM030)", "unit": "GHz"}
curves = [
    ("reference_values", {
        "source": "Fig. 3 (main)", "description": "simulated Q0 vs T (red diamonds; same as Table I Q0 column)",
        "x": X, "y": Q, "series": [{"label": "simulation (HFSS)", "points": main["red"]}],
        "digitization": u_main}),
    ("reference_values", {
        "source": "Fig. 3 (inset)",
        "description": "simulated f(TM030) vs T, SHIFTED by the authors to match measurement at 4 K (caption: "
                       "'shifted by 121 MHz'); unshifted values are Table I f column",
        "x": X, "y": F, "series": [{"label": "simulation, shifted (red diamonds)", "points": ins["red"]}],
        "digitization": u_in}),
    ("measured_values", {
        "source": "Fig. 3 (main)", "description": "measured Q0 vs T, two cooling runs + greyed points",
        "x": X, "y": Q, "series": [
            {"label": "measured run A (black +)", "points": main["black"]},
            {"label": "measured run B (light-blue hollow diamonds)", "points": main["blue"]},
            {"label": "measured, greyed (mode-mixing affected, light-gray +)", "points": main["gray"]}],
        "digitization": u_main}),
    ("measured_values", {
        "source": "Fig. 3 (inset)", "description": "measured f(TM030) vs T",
        "x": X, "y": F, "series": [{"label": "measured (black +)", "points": ins["black"]}],
        "digitization": u_in}),
]
rc.update_curves(os.path.join(rc.REFS, "quax_two_shell_2201.json"), curves)
for k in main:
    print("main", k, len(main[k]))
for k in ins:
    print("inset", k, len(ins[k]), ins[k][:3])
print("cal", cal)
print("red main", main["red"])

# reference crop of the dimension source (read by eye): Fig. 1
c, _ = rc.render(PDF, 2, (60, 40, 560, 230), 200)
rc.save_png(c, os.path.join(rc.REFS, "quax_two_shell_2201_fig1.png"))
