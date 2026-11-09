"""Digitize arXiv:2410.07774 (Di Vora et al., QUAX single-shell clamshell resonator) Fig. 7 (a)-(d).

Usage: python3 -I quax_clamshell_2410_digitize.py /path/to/2410.07774.pdf
Fig. 7 is an embedded raster. Render at 600 dpi; common x axis (opening angle, deg)
calibrated on the 9 vertical gridlines; per-panel y axes on gridlines/frame ticks
(panel b is log10). Markers:
  blue filled circles (simulation, HFSS, 0.1 deg steps)   -> eroded blue mask, centroids
  red open circles (experiment; x rescaled by the authors) -> ring-template correlation
  red 'x' (simulated mode-crossing points, panels c, d)   -> red connected components
Writes REFS/quax_clamshell_2410_fig7.png, ..._fig7_overlay.png and updates the json curves.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import ndimage as nd  # noqa: E402

import quax_raster_common as rc  # noqa: E402

PDF = sys.argv[1]
PAGE, CLIP, DPI = 4, (316, 193, 563, 429), 600
a, _ = rc.render(PDF, PAGE, CLIP, DPI)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
blue = (B > 180) & (R < 90) & (G < 90)
red = (R > 180) & (G < 110) & (B < 110)

X = [(292, 0.0), (503, 0.25), (713.5, 0.5), (924.5, 0.75), (1135.5, 1.0), (1345.5, 1.25),
     (1556.5, 1.5), (1766.5, 1.75), (1977.5, 2.0)]
PANELS = {  # name: (y-frame top, y-frame bottom, y calibration pairs, log?)
    "a": (142.5, 506, [(142.5, 11.05), (323.5, 11.00), (504.5, 10.95)], False),
    "b": (577, 943, [(684.5, 6.0), (836.5, 5.0)], True),
    "c": (1015.5, 1380, [(1088.5, 0.4), (1234, 0.2), (1379.5, 0.0)], False),
    "d": (1452, 1816, [(1512.5, 5.0e4), (1661.5, 2.5e4), (1811, 0.0)], False),
}
cx = rc.lin_cal(X)
cal = {k: rc.lin_cal(v[2]) for k, v in PANELS.items()}

# simulation: filled blue circles; erode away the 5-px connecting line.
# Red open rings / thin 'x' do not survive the erosion on their own.
disk = np.hypot(*np.mgrid[-5:6, -5:6]) <= 5
filled = nd.binary_fill_holes(blue | red)  # a red ring drawn over a blue disc becomes one filled disc
core = nd.binary_erosion(filled, structure=disk)
cand = rc.blobs(core, min_px=5)
yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]


def blue_frac(p, r=9):
    x, y = int(round(p[0])), int(round(p[1]))
    w = (np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r)
    return float(blue[y - r:y + r + 1, x - r:x + r + 1][w].mean())


blue_pts = [p for p in cand if blue_frac(p) > 0.25]  # isolated red open rings fill to white/red discs
red_ring = rc.ring_template_peaks(red, 10, 10, thr=0.45, nms=10)  # circle = diamond-ish ring tolerance
red_x = rc.blobs(red, min_px=60)


def vertex_points(k):
    """Polyline-vertex method: the simulated markers sit on the 0.1-deg grid and the connecting line passes
    through their centres. For each grid angle fit straight lines to the blue line centroid on both sides
    (14-68 px away, i.e. clear of the marker discs and of red markers) and evaluate at the grid x."""
    t, b, _, _ = PANELS[k]
    rows = slice(int(t) + 3, int(b) - 2)
    cen = {}
    for x in range(296, 1974):
        col = blue[rows, x]
        if k in "abcd":
            col = col.copy()
            if x > 1450:  # legend: drop its rows
                col[:135] = False
        rr = np.where(col)[0]
        if len(rr) and not red[rows, x][max(rr.min() - 15, 0):rr.max() + 15].any():
            if rr.max() - rr.min() < 60:
                cen[x] = rr.mean() + int(t) + 3
    out = []
    for i in range(21):
        xi = X[0][0] + i * (X[-1][0] - X[0][0]) / 20.0
        ests = []
        for sgn in (-1, 1):
            xs = [x for x in cen if 14 <= sgn * (x - xi) <= 68]
            if len(xs) >= 15:
                c1 = np.polyfit(xs, [cen[x] for x in xs], 1)
                ests.append(np.polyval(c1, xi))
        if ests:
            out.append((xi, float(np.mean(ests)), float(np.ptp(ests)) if len(ests) == 2 else None))
    return out


def panel_of(p, legend_free=True):
    x, y = p
    for k, (t, b, _, _) in PANELS.items():
        if t + 4 < y < b - 2 and 280 < x < 1990:
            if legend_free and x > 1450 and y < t + 135:
                return None  # legend region
            return k
    return None


def val(p, k):
    m, b0, _ = cal[k]
    y = m * p[1] + b0
    if PANELS[k][3]:
        y = 10 ** y
    return [round(cx[0] * p[0] + cx[1], 4), float("%.5g" % y)]


ser = {}
for name, pts in (("sim", blue_pts), ("exp", red_ring), ("xcross", red_x)):
    for p in pts:
        k = panel_of(p)
        if k is None:
            continue
        ser.setdefault((name, k), []).append(p)
# red 'x' only in c,d ; red open circles only in a,b (the ring detector also fires on 'x')
for k in ("c", "d"):
    ser.pop(("exp", k), None)
for k in ("a", "b"):
    ser.pop(("xcross", k), None)

# The last experimental ring in panel (a) (~2.0 deg) is clipped by the right frame,
# so the ring template misses it. Recover it from the left arc: centre = arc-left-edge + ring radius
# (radius measured on the complete rings, ~10 px at 600 dpi), y = midpoint of the arc's vertical extent.
RING_R = 10.0
ys_, xs_ = np.nonzero(red[int(PANELS["a"][0]):int(PANELS["a"][1]), 1950:])
xs_, ys_ = xs_ + 1950, ys_ + int(PANELS["a"][0])
CLIPPED_RING = []
if len(xs_) and not any(p[0] > 1950 for p in ser.get(("exp", "a"), [])):
    CLIPPED_RING = [(xs_.min() + RING_R, 0.5 * (ys_.min() + ys_.max()))]

rc.save_png(rc.downsample(a, 2), os.path.join(rc.REFS, "quax_clamshell_2410_fig7.png"))
groups = []
for k in PANELS:
    sk = [0.2, 0.7] if k in "cd" else []
    groups.append(([(x, y) for x, y, _ in vertex_points(k) if round(cx[0] * x + cx[1], 1) not in sk], (255, 160, 0), "s"))
groups += [(ser.get(("exp", k), []), (0, 200, 0), "x") for k in "ab"]
groups += [(CLIPPED_RING, (255, 0, 255), "x")]  # manually recovered clipped ring (magenta)
groups += [(ser.get(("xcross", k), []), (0, 200, 0), "s") for k in "cd"]
lines = [("v", p, (0, 200, 200)) for p, _ in X] + [("h", p, (0, 200, 200)) for k in PANELS for p, _ in PANELS[k][2]]
tmp = os.path.join(rc.REFS, "_tmp_overlay2410.png")
rc.overlay(a, groups, tmp, lines)
import fitz  # noqa: E402
pm = fitz.Pixmap(tmp)
b = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
rc.save_png(rc.downsample(b, 2), os.path.join(rc.REFS, "quax_clamshell_2410_fig7_overlay.png"))
os.remove(tmp)

pxx = abs(cx[0])
ylab = {"a": ({"label": "f(pseudo-TM030)", "unit": "GHz"}, 3 * abs(cal["a"][0])),
        "b": ({"label": "Q0 (unloaded)", "unit": "1"}, None),
        "c": ({"label": "C030", "unit": "1"}, 3 * abs(cal["c"][0])),
        "d": ({"label": "F = (C V)^2 Q", "unit": "L^2 (axis label '(Lt)^2')"}, 3 * abs(cal["d"][0]))}


def dig(k):
    uy = ylab[k][1]
    return {
        "method": "raster color mask at 600 dpi (embedded bitmap); filled circles: blue mask eroded by r=5 px disk "
                  "(removes connecting line), centroids; open circles: ring-template correlation; 'x': connected "
                  "components; linear x and per-panel y calibration on gridline/frame pixels" +
                  (" (log10 y)" if PANELS[k][3] else ""),
        "page": PAGE + 1, "clip_bbox_pt": list(CLIP), "dpi": DPI,
        "axis_calibration": {"x": [list(v) for v in X], "y": [list(v) for v in PANELS[k][2]],
                             "log": PANELS[k][3],
                             "fit_rms_residual": {"x": cx[2], "y": cal[k][2]}},
        "uncertainty_x": round(3 * pxx, 4),
        "uncertainty_y": ("x/÷ %.3f (3 px in log10)" % (10 ** (3 * abs(cal[k][0])))) if uy is None else float("%.2g" % uy),
        "uncertainty_basis": "+-3 px at 600 dpi (calibration <1 px + marker centroid ~2 px)",
        "figure_png": "quax_clamshell_2410_fig7.png", "overlay_png": "quax_clamshell_2410_fig7_overlay.png",
        "script": "digitize/quax_clamshell_2410_digitize.py",
    }


XA = {"label": "clamshell opening angle", "unit": "deg"}
curves = []
VSPREAD = {}
for k, desc in (("a", "simulated f vs opening angle"), ("b", "simulated Q0 vs opening angle"),
                ("c", "simulated C030 vs opening angle"), ("d", "simulated figure of merit F vs opening angle")):
    vp = vertex_points(k)
    skip = [0.2, 0.7] if k in "cd" else []  # no simulated marker there (replaced by red 'x' crossing points)
    prim = [[round(cx[0] * x + cx[1], 1), val((x, y), k)[1]] for x, y, _ in vp if round(cx[0] * x + cx[1], 1) not in skip]
    lab = "simulation (HFSS, blue filled circles); polyline-vertex method, x snapped to the stated 0.1-deg grid"
    if k == "a":
        # At 0.4 and 0.8 deg the vertex method is 1-2 MHz off the marker centroid, and an
        # independent re-digitization agrees with the centroid -> use the centroid value there.
        cen = {round(v[0], 1): v[1] for v in (val(p, k) for p in ser.get(("sim", k), []))}
        for q in prim:
            if q[0] in (0.4, 0.8) and q[0] in cen:
                q[1] = cen[q[0]]
        lab += "; at 0.4 and 0.8 deg the marker-centroid value is used"
    s = [{"label": lab, "points": prim},
         {"label": "cross-check: marker-centroid method (subset where disc detected)",
          "points": sorted(val(p, k) for p in ser.get(("sim", k), []))}]
    VSPREAD[k] = max([d for _, _, d in vp if d is not None] or [0])
    if k in "cd":
        s.append({"label": "simulated mode-crossing points (red x)",
                  "points": sorted(val(p, k) for p in ser.get(("xcross", k), []))})
    curves.append(("reference_values", {"source": "Fig. 7(%s)" % k, "description": desc, "x": XA,
                                         "y": ylab[k][0], "series": s, "digitization": dig(k)}))
for k, desc in (("a", "measured f vs opening angle (x axis rescaled by authors from motor steps)"),
                ("b", "measured Q0 vs opening angle (x axis rescaled by authors from motor steps)")):
    curves.append(("measured_values", {"source": "Fig. 7(%s)" % k, "description": desc, "x": XA, "y": ylab[k][0],
                                       "series": [{"label": "experiment (red open circles)" +
                                                   ("; last point (~2.0 deg) recovered from a ring clipped by the "
                                                    "right frame (left arc + ring radius; uncertainty ~2x stated)"
                                                    if k == "a" and CLIPPED_RING else ""),
                                                   "points": sorted(val(p, k) for p in ser.get(("exp", k), []) +
                                                                    (CLIPPED_RING if k == "a" else []))}],
                                       "digitization": dig(k)}))
rc.update_curves(os.path.join(rc.REFS, "quax_clamshell_2410.json"), curves)
print("max left/right vertex disagreement px:", VSPREAD)
for k in PANELS:
    print(k, [val((x, y), k) for x, y, _ in vertex_points(k)])
for key in sorted(ser):
    print(key, len(ser[key]), [val(p, key[1]) for p in sorted(ser[key])][:25])

# reference crop of Table I/II region is text; save Fig. 1 (model, gap) for context
c, _ = rc.render(PDF, 1, (315, 50, 572, 235), 200)
rc.save_png(c, os.path.join(rc.REFS, "quax_clamshell_2410_fig1.png"))
