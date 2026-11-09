"""Digitize Simanovskaia et al. 2020 (arXiv:2006.01248) Fig. 3 and Fig. 6.

Usage:  python3 -I haystac_rod_2006_digitize.py /path/to/2006.01248.pdf

Both figures are embedded raster images (no vector paths), so we extract the embedded
bitmaps at native resolution (Fig. 3: 1000x1000 px; Fig. 6a: 1650x900; 6b: 750x900; 6c: 750x870)
and calibrate the axes from the matplotlib grid lines (grey, RGB 176) whose pixel positions are
detected automatically and checked against the expected tick values.

Fig. 3 (mode map, black dots):
  * every dark pixel run in every pixel column of the plot area -> (theta, f) point cloud;
  * the TM010-like branch is the lowest steeply-tuning curve (caption: "TM010-like mode
    frequencies correspond to the lowest-frequency TM mode at all rod rotations").  It is
    extracted by snapping, column by column, to the dark run nearest (<= 3 px) a PCHIP guide
    curve through hand-placed anchor points (ANCHORS below, chosen by eye on zoomed crops and
    checked on the overlay).  Columns where no run lies within 3 px (dot gaps at crossings) are
    left out.  The snapped points are then binned to 1 deg (median).
  * theta <= 23 deg: two parallel steep lines 14-40 MHz apart are traced
    SEPARATELY on individual '+' marker centroids, each snapped (<= 7 MHz) to its own guide
    (LOWER_ANCHORS = TM010-like, UPPER_ANCHORS = upper parallel line); see the constants below.
    Also writes _fig3_overlay_zoom.png (0-30 deg; red = lower, green = upper, blue = tail).
Fig. 6 (scatter markers, 4 designs, colour coded):
  * per colour mask, the clean isolated marker shape is used as a template; binary erosion of
    the colour mask with the template gives the marker centres even where markers of the same
    colour overlap.  Markers partially hidden by another colour or clipped by the axes frame are
    recovered from their visible bounding box and flagged (larger uncertainty).

Writes <REFS>/haystac_rod_2006_fig3.png, _fig3_overlay.png, _fig6{a,b,c}.png, _fig6{a,b,c}_overlay.png
and <REFS>/digitize/haystac_rod_2006_curves.json; if <REFS>/haystac_rod_2006.json exists, its
reference_values.curves is replaced by the new extraction.
"""
import json
import os
import sys

import numpy as np  # numpy/scipy before fitz: fitz first pulls in an older libstdc++
from scipy import ndimage as nd
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.dirname(HERE)
NAME = "haystac_rod_2006"

# (theta_deg, f_GHz) guide anchors for the TM010-like branch of Fig. 3.
ANCHORS = [[0, 5.86], [10, 5.62], [20, 5.3], [30, 4.95], [40, 4.72], [50, 4.5], [60, 4.33],
           [70, 4.17], [80, 4.03], [90, 3.9], [100, 3.79], [110, 3.7], [120, 3.63], [130, 3.57],
           [140, 3.53], [150, 3.5], [160, 3.475], [170, 3.46], [180, 3.455]]

# Near theta = 0 Fig. 3 has TWO parallel steep lines 13-40 MHz apart; the
# 1-deg median binning above hopped between them.  For theta <= TWO_BRANCH_MAX the two lines are now
# traced separately on individual dot centroids (each simulated point is a 5-px '+' marker), each by
# snapping to its own PCHIP guide (anchors read off labelled zooms of the dot centroids; tolerance
# SNAP_TOL_GHZ ~ 2 px, i.e. far less than the 13-40 MHz line separation, so a point can only join
# the line it lies on).  Caption: TM010-like = lowest-frequency TM mode -> LOWER line.
# Between ~14.5 and ~19.5 deg the lower line passes a dense cluster of avoided crossings with flat
# intruder lines (5.30-5.48 GHz); following the caption rule the lowest steep chain is taken
# (15.9/5.366 -> 17.5/5.329 -> 19.4/5.272), but the assignment there is uncertain at +/-0.03 GHz.
TWO_BRANCH_MAX = 23.0
SNAP_TOL_GHZ = 0.007
LOWER_ANCHORS = [[0.3, 5.835], [1, 5.8282], [2, 5.8115], [4, 5.7612], [5, 5.7311], [6, 5.698],
                 [7, 5.6674], [8, 5.6373], [9.5, 5.5871], [11, 5.5368], [12.5, 5.49], [14.5, 5.4297],
                 [16, 5.366], [17.5, 5.329], [19.5, 5.2723], [21, 5.232], [22, 5.2086], [23, 5.1852]]
UPPER_ANCHORS = [[0.3, 5.848], [1, 5.8449], [1.5, 5.8416], [2, 5.8315], [4, 5.788], [6, 5.7277],
                 [8, 5.6674], [10, 5.6038], [12, 5.5402], [14, 5.4799]]
AMBIGUOUS_RANGES = [[14.5, 19.5]]


def dot_centroids(L, x_axis, cx, cy, max_area=9):
    """(theta, f, area) of isolated '+' markers; the frame column(s) at theta = 0 are masked out."""
    m = L < 150
    m[:, :int(round(x_axis)) + 2] = False
    lab, n = nd.label(m)
    idx = np.arange(1, n + 1)
    cs = np.array(nd.center_of_mass(m, lab, idx))
    ar = np.array(nd.sum(m, lab, idx))
    keep = ar <= max_area
    return np.c_[np.polyval(cx, cs[keep, 1]), np.polyval(cy, cs[keep, 0]), ar[keep]]


def snap_branch(D, anchors, tol=SNAP_TOL_GHZ):
    """Dot centroids within tol of the PCHIP guide; one point per 0.5-deg simulation step (nearest)."""
    A = np.array(anchors, float)
    g = PchipInterpolator(A[:, 0], A[:, 1])
    sel = (D[:, 0] >= A[0, 0] - 0.25) & (D[:, 0] <= A[-1, 0] + 0.25)
    P = D[sel]
    P = P[np.abs(P[:, 1] - g(np.clip(P[:, 0], A[0, 0], A[-1, 0]))) < tol]
    P = P[np.argsort(P[:, 0])]
    out = []
    for t, fv, _ in P:
        if out and t - out[-1][0] < 0.3:      # same simulation step (raster jitter): keep nearer to guide
            if abs(fv - g(t)) < abs(out[-1][1] - g(out[-1][0])):
                out[-1] = [t, fv]
            continue
        out.append([t, fv])
    return [[round(float(t), 2), round(float(fv), 4)] for t, fv in out]


def page_images(doc, pno):
    """Return [(bbox, PIL image)] of the embedded images on page pno, native resolution."""
    out = []
    for info in doc[pno].get_image_info(xrefs=True):
        pix = fitz.Pixmap(doc, info["xref"])
        if pix.n - pix.alpha > 3 or pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((info["bbox"], img))
    return out


def find_page(doc, caption):
    for i, p in enumerate(doc):
        if caption in p.get_text() and p.get_images():
            return i
    raise RuntimeError(caption)


def grid_lines(a, axis, frac=0.3):
    """Pixel centres of grey (176) matplotlib grid lines along an axis (0 = rows, 1 = cols)."""
    g = (np.abs(a[..., 0] - 176) < 6) & (np.abs(a[..., 1] - 176) < 6) & (np.abs(a[..., 2] - 176) < 6)
    s = g.sum(axis=1 - axis) if axis == 0 else g.sum(axis=0)
    n = a.shape[1] if axis == 0 else a.shape[0]
    idx = np.where(s > frac * n)[0]
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [float(gr.mean()) for gr in groups]


def calib(pix, vals):
    pix, vals = np.asarray(pix, float), np.asarray(vals, float)
    c = np.polyfit(pix, vals, 1)
    resid = np.max(np.abs(np.polyval(c, pix) - vals)) if len(pix) > 2 else 0.0
    return c, resid


# ---------------------------------------------------------------- Fig. 3
def fig3(doc):
    pno = find_page(doc, "FIG. 3.")
    (bbox, img), = page_images(doc, pno)
    img.save(os.path.join(REFS, f"{NAME}_fig3.png"))
    a = np.asarray(img).astype(int)
    cols = [c for c in grid_lines(a, 1) if 100 < c < 950]
    rows = [r for r in grid_lines(a, 0) if 40 < r < 910]
    assert len(cols) == 8 and len(rows) == 5, (cols, rows)
    cx, rx = calib(cols, np.arange(20, 180, 20))      # theta = cx[0]*px + cx[1]
    cy, ry = calib(rows, [5.5, 5.0, 4.5, 4.0, 3.5])   # f = cy[0]*py + cy[1]
    L = a.mean(axis=2)
    # plot frame (black axis lines) -> data window [0,180] x [3,6]
    x0 = int(round((0 - cx[1]) / cx[0])) + 3
    x1 = int(round((180 - cx[1]) / cx[0])) - 3
    y0 = int(round((6 - cy[1]) / cy[0])) + 3
    y1 = int(round((3 - cy[1]) / cy[0])) - 3
    cloud = []
    for x in range(x0, x1 + 1):
        idx = np.where(L[y0:y1 + 1, x] < 150)[0]
        if len(idx) == 0:
            continue
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            cloud.append((x, y0 + r.mean(), len(r)))
    cloud = np.array(cloud)
    A = np.array(ANCHORS, float)
    guide = PchipInterpolator((A[:, 0] - cx[1]) / cx[0], (A[:, 1] - cy[1]) / cy[0])
    snap = []
    for x in range(x0, x1 + 1):
        s = cloud[cloud[:, 0] == x]
        if not len(s):
            continue
        d = np.abs(s[:, 1] - guide(x))
        j = d.argmin()
        if d[j] <= 3.0:
            snap.append((x, s[j, 1]))
    snap = np.array(snap)
    th = np.polyval(cx, snap[:, 0])
    f = np.polyval(cy, snap[:, 1])
    branch = []
    for t in range(0, 181):
        m = np.abs(th - t) <= 0.5
        if m.sum() >= 2:
            branch.append([t, round(float(np.median(f[m])), 4)])
    # two-branch structure near theta = 0
    x_axis = (0 - cx[1]) / cx[0]
    D = dot_centroids(L, x_axis, cx, cy)
    lower = snap_branch(D, LOWER_ANCHORS)
    upper = snap_branch(D, UPPER_ANCHORS)
    branch_tail = [p for p in branch if p[0] > TWO_BRANCH_MAX]
    tm010 = lower + branch_tail
    # decimated full cloud: pixel columns nearest every 2 deg
    allpts = []
    for t in range(0, 181, 2):
        x = int(round((t - cx[1]) / cx[0]))
        x = min(max(x, x0), x1)
        for _, y, n in cloud[cloud[:, 0] == x]:
            allpts.append([t, round(float(np.polyval(cy, y)), 4)])
    # overlay
    ov = img.convert("RGB").copy()
    d = ImageDraw.Draw(ov)
    for x, y in snap:
        if np.polyval(cx, x) > TWO_BRANCH_MAX:
            d.point((x, y), fill=(0, 150, 255))
    for t, fv in allpts:
        X = (t - cx[1]) / cx[0]
        Y = (fv - cy[1]) / cy[0]
        d.point((X, Y), fill=(255, 140, 0))
    for pts, col in ((branch_tail, (0, 120, 255)), (lower, (230, 0, 0)), (upper, (0, 160, 0))):
        for t, fv in pts:
            X = (t - cx[1]) / cx[0]
            Y = (fv - cy[1]) / cy[0]
            d.ellipse((X - 3, Y - 3, X + 3, Y + 3), outline=col, width=1)
    for c in cols:
        d.line((c, y1 + 6, c, y1 + 14), fill=(0, 200, 0), width=2)
    for r in rows:
        d.line((x0 - 14, r, x0 - 6, r), fill=(0, 200, 0), width=2)
    ov.save(os.path.join(REFS, f"{NAME}_fig3_overlay.png"))
    # zoom of the two-branch region (0-30 deg, 4.95-5.90 GHz), 4x nearest-neighbour
    zx0 = int(x_axis) - 4
    zx1 = int((30 - cx[1]) / cx[0])
    zy0 = int((5.90 - cy[1]) / cy[0])
    zy1 = int((4.95 - cy[1]) / cy[0])
    zoom = ov.crop((zx0, zy0, zx1, zy1))
    zoom.resize((zoom.width * 4, zoom.height * 4), Image.NEAREST).save(
        os.path.join(REFS, f"{NAME}_fig3_overlay_zoom.png"))
    dx, dy = abs(cx[0]), abs(cy[0])
    dig = {
        "method": "raster: embedded 1000x1000 bitmap extracted at native resolution via pymupdf; "
                  "axes calibrated by linear fit to the 8 vertical / 5 horizontal grey grid lines; "
                  "dark-pixel runs per pixel column; TM010-like branch = runs snapped (<=3 px) to a "
                  "PCHIP guide through hand-placed anchors, binned to 1 deg (median)",
        "page": pno + 1, "clip_bbox_pt": [round(v, 1) for v in bbox],
        "axis_calibration": {"x": [[round(c, 1), v] for c, v in zip(cols, range(20, 180, 20))],
                             "y": [[round(r, 1), v] for r, v in zip(rows, [5.5, 5.0, 4.5, 4.0, 3.5])],
                             "log": False,
                             "max_calibration_residual": {"deg": round(float(rx), 3), "GHz": round(float(ry), 5)}},
        "px_size": {"deg_per_px": round(dx, 4), "GHz_per_px": round(dy, 5)},
        "uncertainty_x": 0.5, "uncertainty_y": 0.008,
        "uncertainty_basis": "dot diameter ~2-3 px and 0.5 deg simulation step ~2.4 px -> +/-0.5 deg; "
                             "+/-2 px dot centroid + calibration -> +/-0.008 GHz on isolated stretches. "
                             "Near avoided crossings and for theta > ~160 deg (branch merges with flat "
                             "intruder lines near 3.44-3.46 GHz) the branch assignment itself is "
                             "uncertain at the +/-0.02 GHz level.",
        "anchors": ANCHORS,
        "figure_png": f"{NAME}_fig3.png", "overlay_png": f"{NAME}_fig3_overlay.png",
        "script": f"digitize/{NAME}_digitize.py",
    }
    dig2 = dict(dig)
    dig2.update({
        "method": "raster, as above, but for theta <= %.0f deg on individual marker centroids: every "
                  "isolated 5-px '+' marker (connected dark component, area <= 9 px, theta = 0 frame column "
                  "masked) whose centroid lies within %.3f GHz (~2 px) of the branch's own PCHIP guide; one "
                  "point per 0.5-deg simulation step; theta is the raw centroid (raster-quantized, so steps "
                  "read e.g. 1.44/1.86/2.47 deg). Markers merged with flat intruder lines are skipped (gaps). "
                  "For theta > %.0f deg the 1-deg median binning of the dark-run cloud (original method) is kept."
                  % (TWO_BRANCH_MAX, SNAP_TOL_GHZ, TWO_BRANCH_MAX),
        "uncertainty_x": 0.15, "uncertainty_y": 0.004,
        "uncertainty_basis": "two-branch region: +/-1 px marker centroid -> +/-0.2 deg (raster quantization of "
                             "the 0.5-deg step), +/-0.0034 GHz + calibration 0.0013 GHz -> +/-0.004 GHz per point; "
                             "branch assignment unambiguous for theta < 14.5 deg (lines 13-40 MHz apart, "
                             "tolerance 7 MHz), ambiguous at the +/-0.03 GHz level for 14.5-19.5 deg (avoided "
                             "crossings); theta > 23 deg: as the original method (+/-0.5 deg, +/-0.008 GHz; "
                             "+/-0.02 GHz near crossings and theta > 160 deg)",
        "anchors_lower": LOWER_ANCHORS, "anchors_upper": UPPER_ANCHORS, "anchors_tail": ANCHORS,
        "ambiguous_theta_ranges_deg": AMBIGUOUS_RANGES,
        "overlay_zoom_png": f"{NAME}_fig3_overlay_zoom.png",
        "overlay_legend": "red circles = lower (TM010-like) line, theta <= 23 deg; green circles = upper "
                          "parallel line; blue circles = TM010-like branch theta > 23 deg (1-deg bins); "
                          "orange dots = full cloud sample",
    })
    f0 = {
        "lower_TM010_like_GHz": 5.835, "upper_parallel_GHz": 5.849, "uncertainty_GHz": 0.004,
        "basis": "markers at theta = 0 are hidden under the frame line; their right arms are visible at "
                 "f = 5.835 (lower) and 5.850 (upper), and the 0.5-deg markers read 5.8349 / 5.8464-5.8483: "
                 "both lines are flat at the top (f even in theta), so f(0) = f(0.5 deg) within 1 px. "
                 "Branch separation grows 14 -> 27 -> 33 -> ~40 MHz at 0 / 4 / 9.5 / 14 deg.",
        "benchmark_target": "f_TM010-like(theta = 0) = 5.835 GHz on the lower line; quote 5.83-5.85 GHz with "
                            "+/-0.015 GHz target tolerance (line assignment, axles, CST mesh unknown)",
    }
    return [
        {"source": "2006.01248 Fig. 3", "description": "Simulated (CST MWS eigenmode) TM010-like branch of the "
         "one-rod HAYSTAC Phase I/II cavity vs rod rotation (0 deg = rod concentric, max frequency). Near "
         "theta = 0 the figure has two parallel steep lines (13-40 MHz apart); they are recorded as two "
         "separate series. Series 0 = lower line = TM010-like (caption: lowest-frequency TM mode); series 1 = "
         "upper parallel line (not TM010-like; identity unknown, plausibly TM011-like: the coax TM011 offset "
         "sqrt(f^2 + (c/2H)^2) - f is ~30 MHz).",
         "x": {"label": "rod rotation theta", "unit": "deg"}, "y": {"label": "frequency", "unit": "GHz"},
         "series": [{"label": "TM010-like (lower of the two parallel lines; lowest steeply tuning TM branch)",
                     "points": tm010,
                     "note": "theta <= 23 deg: marker centroids (0.01-deg abscissae, raster-quantized 0.5-deg "
                             "steps); theta >= 24: 1-deg median bins. 14.5-19.5 deg ambiguous (+/-0.03 GHz)."},
                    {"label": "upper parallel steep line (NOT TM010-like), theta 0.3-14 deg",
                     "points": upper,
                     "note": "traced only where unambiguous; merges into the crossing cluster beyond ~14 deg"}],
         "f_theta0": f0,
         "digitization": dig2},
        {"source": "2006.01248 Fig. 3", "description": "Full simulated mode map point cloud (all eigenmodes "
         "between 3 and 6 GHz), sampled at the pixel column nearest every 2 deg; one point per dark-pixel "
         "run, so near-degenerate modes (< ~3 px = 0.01 GHz apart) are merged into one point",
         "x": {"label": "rod rotation theta", "unit": "deg"}, "y": {"label": "frequency", "unit": "GHz"},
         "series": [{"label": "all modes (unassigned)", "points": allpts}],
         "digitization": dict(dig, method=dig["method"].split("; TM010")[0] + "; every dark run kept "
                              "(no mode assignment)", uncertainty_y=0.008)},
    ]


# ---------------------------------------------------------------- Fig. 6
COLORS = {
    "1x2.00in OD (red triangles)": lambda R, G, B: (R > 180) & (G < 90) & (B < 90),
    "1x2.46in OD (black stars)": lambda R, G, B: (R < 70) & (G < 70) & (B < 70),
    "7x0.625in OD (blue squares)": lambda R, G, B: (B > 180) & (R < 90) & (G < 90),
    "7x0.875in OD (green circles)": lambda R, G, B: (G > 90) & (R < 70) & (B < 70),
}


def markers(a, mask, frame, legend=None):
    """Return list of (x, y, flag) marker centres in pixel coords."""
    inside = np.zeros(mask.shape, bool)
    x0, y0, x1, y1 = frame
    inside[y0 + 4:y1 - 3, x0 + 4:x1 - 3] = True
    if legend is not None:
        lx0, ly0, lx1, ly1 = legend
        inside[ly0:ly1, lx0:lx1] = False
    m = mask & inside
    lab, n = nd.label(m)
    objs = nd.find_objects(lab)
    areas = [(lab[s] == i + 1).sum() for i, s in enumerate(objs)]
    big = [i for i, ar in enumerate(areas) if ar >= 60]
    if not big:
        return []
    # template = the most common (modal) clean component
    shapes = {}
    for i in big:
        s = objs[i]
        key = (s[0].stop - s[0].start, s[1].stop - s[1].start, areas[i])
        shapes.setdefault(key, []).append(i)
    key = max(shapes, key=lambda k: len(shapes[k]))
    s = objs[shapes[key][0]]
    tmpl = lab[s] == shapes[key][0] + 1
    th, tw = tmpl.shape
    er = nd.binary_erosion(m, structure=tmpl, origin=(0, 0), border_value=0)
    # binary_erosion centres the structure: offset is (th//2, tw//2) -> bbox centre correction
    elab, en = nd.label(er)
    out = []
    covered = np.zeros_like(m)
    for j, es in enumerate(nd.find_objects(elab)):
        ys, xs = np.where(elab[es] == j + 1)
        cy_ = es[0].start + ys.mean()
        cx_ = es[1].start + xs.mean()
        # convert erosion anchor (structure centre) to bbox centre of template
        cy_c = cy_ - (th // 2) + (th - 1) / 2
        cx_c = cx_ - (tw // 2) + (tw - 1) / 2
        out.append((cx_c, cy_c, "ok"))
        yy0 = int(round(cy_c - (th - 1) / 2)); xx0 = int(round(cx_c - (tw - 1) / 2))
        covered[max(yy0 - 1, 0):yy0 + th + 1, max(xx0 - 1, 0):xx0 + tw + 1] = True
    # leftovers: occluded or clipped markers
    rest = m & ~covered
    # group fragments of one hidden marker (dilate only for grouping), bbox from raw pixels
    rl, rn = nd.label(nd.binary_dilation(rest, iterations=4))
    rl = np.where(rest, rl, 0)
    for k in range(1, rn + 1):
        ys_, xs_ = np.where(rl == k)
        if len(ys_) < 20:
            continue
        rs = (slice(ys_.min(), ys_.max() + 1), slice(xs_.min(), xs_.max() + 1))
        hh = rs[0].stop - rs[0].start
        ww = rs[1].stop - rs[1].start
        yc = (rs[0].start + rs[0].stop - 1) / 2
        xc = (rs[1].start + rs[1].stop - 1) / 2
        flag = "partial"
        if rs[1].stop >= x1 - 5:      # clipped by right frame: anchor on left edge
            xc = rs[1].start + (tw - 1) / 2
            flag = "clipped_right"
        if hh > 1.5 * th:             # two clipped markers stacked
            out.append((xc, rs[0].start + (th - 1) / 2, flag))
            out.append((xc, rs[0].stop - 1 - (th - 1) / 2, flag))
            continue
        if ww < tw - 1 and flag == "partial":   # occluded from one side: keep the full-height edge
            # assume the occluder sits on the side with the larger gap to the template width
            xc = rs[1].stop - 1 - (tw - 1) / 2
        if hh < th - 1 and flag == "partial":
            yc = rs[0].start + (th - 1) / 2 if rs[0].start > 0 else yc
        out.append((xc, yc, flag))
    return out


def fig6(doc):
    pno = find_page(doc, "FIG. 6.")
    ims = page_images(doc, pno)
    # (a) = top wide image; (b) = lower-left; (c) = lower-right (caption: a FoM, b Q, c C)
    ims.sort(key=lambda t: (round(t[0][1] / 100), t[0][0]))
    panels = {"a": ims[0], "b": ims[1], "c": ims[2]}
    spec = {
        "a": dict(ylabel="figure of merit Q C^2 V^2", yunit="m^6", yscale=1e-2,
                  yticks=[2.0, 1.5, 1.0, 0.5, 0.0], xticks=[5, 6, 7, 8, 9]),
        "b": dict(ylabel="quality factor Q", yunit="1", yscale=1e4,
                  yticks=[1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2], xticks=[6, 8]),
        "c": dict(ylabel="form factor C", yunit="1", yscale=1.0,
                  yticks=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3], xticks=[6, 8]),
    }
    curves = []
    for key, (bbox, img) in panels.items():
        sp = spec[key]
        img.save(os.path.join(REFS, f"{NAME}_fig6{key}.png"))
        a = np.asarray(img).astype(int)
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        blk = (R < 70) & (G < 70) & (B < 70)
        fc = np.where(blk.sum(0) > 0.5 * a.shape[0] * 0.5)[0]
        fr = np.where(blk.sum(1) > 0.5 * a.shape[1] * 0.5)[0]
        frame = (int(fc.min()), int(fr.min()), int(fc.max()), int(fr.max()))
        cols = [c for c in grid_lines(a, 1, 0.25) if frame[0] + 5 < c < frame[2] - 5]
        rows = [r for r in grid_lines(a, 0, 0.25) if frame[1] + 5 < r < frame[3] - 5]
        assert len(cols) == len(sp["xticks"]) and len(rows) == len(sp["yticks"]), (key, cols, rows)
        cx, rx = calib(cols + [frame[0] + 1, frame[2] - 1], sp["xticks"] + [4, 10])
        cy, ry = calib(rows, sp["yticks"])
        legend = (frame[0] + int(0.67 * (frame[2] - frame[0])), frame[1], frame[2], frame[1] + 290) if key == "a" else None
        ov = img.convert("RGB").copy()
        d = ImageDraw.Draw(ov)
        series = []
        flags_all = {}
        for lab_, fn in COLORS.items():
            pts = markers(a, fn(R, G, B), frame, legend)
            pts.sort()
            P = []
            for x, y, fl in pts:
                xv = float(np.polyval(cx, x))
                yv = float(np.polyval(cy, y)) * sp["yscale"]
                P.append([round(xv, 3), float(f"{yv:.4g}")])
                if fl != "ok":
                    flags_all.setdefault(lab_, []).append([round(xv, 3), fl])
                col = (255, 0, 255) if fl == "ok" else (255, 150, 0)
                d.line((x - 7, y, x + 7, y), fill=col, width=1)
                d.line((x, y - 7, x, y + 7), fill=col, width=1)
                d.ellipse((x - 10, y - 10, x + 10, y + 10), outline=col, width=1)
            series.append({"label": lab_, "points": P})
        for c in cols:
            d.line((c, frame[3] + 3, c, frame[3] + 12), fill=(0, 200, 0), width=2)
        for r in rows:
            d.line((frame[0] - 12, r, frame[0] - 3, r), fill=(0, 200, 0), width=2)
        ov.save(os.path.join(REFS, f"{NAME}_fig6{key}_overlay.png"))
        ux = round(abs(cx[0]) * 2, 3)
        uy = abs(cy[0]) * 2 * sp["yscale"]
        curves.append({
            "source": f"2006.01248 Fig. 6{key}",
            "description": f"Simulated (CST MWS) TM010-like {sp['ylabel']} vs frequency for four rod designs "
                           "(0.01 in end gaps, alumina axles; one point per simulated rod position)",
            "x": {"label": "frequency", "unit": "GHz"},
            "y": {"label": sp["ylabel"], "unit": sp["yunit"]},
            "series": series,
            "digitization": {
                "method": "raster: embedded bitmap at native resolution; axes from grey grid lines + frame; "
                          "marker centres by binary erosion of per-colour masks with the clean marker shape "
                          "as template (separates overlapping same-colour markers); occluded / frame-clipped "
                          "markers from their visible bounding box (flagged)",
                "page": pno + 1, "clip_bbox_pt": [round(v, 1) for v in bbox],
                "axis_calibration": {"x": [[round(c, 1), v] for c, v in zip(cols, sp["xticks"])],
                                     "y": [[round(r, 1), v] for r, v in zip(rows, sp["yticks"])],
                                     "y_multiplier": sp["yscale"], "log": False,
                                     "max_calibration_residual": {"x": round(float(rx), 4), "y": round(float(ry), 4)}},
                "uncertainty_x": ux, "uncertainty_y": float(f"{uy:.3g}"),
                "uncertainty_basis": "+/-2 px on marker centre (template fit) + calibration; flagged "
                                     "(occluded/clipped) markers: about 2x larger",
                "flagged_markers": flags_all,
                "figure_png": f"{NAME}_fig6{key}.png", "overlay_png": f"{NAME}_fig6{key}_overlay.png",
                "script": f"digitize/{NAME}_digitize.py",
            },
        })
    return curves


def dump(obj, path):
    """json with short numeric / string lists kept on one line."""
    import re
    txt = json.dumps(obj, indent=1)
    pat = re.compile(r"\[\s*((?:-?[\d.eE+-]+|null|\"[^\"\n]*\")(?:,\s*(?:-?[\d.eE+-]+|null|\"[^\"\n]*\"))*)\s*\]")
    txt = pat.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", txt)
    json.loads(txt)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def main(pdf):
    doc = fitz.open(pdf)
    curves = fig3(doc) + fig6(doc)
    dump(curves, os.path.join(HERE, f"{NAME}_curves.json"))
    main_json = os.path.join(REFS, f"{NAME}.json")
    if os.path.exists(main_json):
        with open(main_json) as fh:
            J = json.load(fh)
        J["reference_values"]["curves"] = curves
        dump(J, main_json)
    for c in curves:
        print(c["source"], [(s["label"], len(s["points"])) for s in c["series"]])


if __name__ == "__main__":
    main(sys.argv[1])
