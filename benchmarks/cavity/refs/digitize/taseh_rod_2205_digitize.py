"""Digitize TASEH arXiv:2205.01477 Figs. 5, 7, 8 (raster figures embedded in the PDF).

Usage:  python3 -I taseh_rod_2205_digitize.py <path/to/2205.01477.pdf> [refs_dir]

All figures in 2205.01477 are embedded bitmaps (MATLAB exports), so we extract the
embedded image at native resolution (no resampling), calibrate the axes from the
tick marks (pixel positions measured on the frame colour (38,38,38), listed below),
segment the markers by colour, separate touching markers by binary erosion, and take
the centroid of each marker. The simulation curve of Fig. 7 is traced column by column.
The script writes <name>_figN.png (native crop), <name>_figN_overlay.png, and updates
the "curves" entries of <refs_dir>/taseh_rod_2205.json in place.
"""
import json
import os
import sys

import numpy as np  # import numpy/scipy BEFORE fitz: pymupdf can pull in an older libstdc++
from PIL import Image, ImageDraw
from scipy import ndimage as nd
import fitz

NAME = "taseh_rod_2205"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = "digitize/taseh_rod_2205_digitize.py"

# (page index, image index on page sorted by area desc) for each figure
FIGS = {"fig5": (4, 0), "fig7": (6, 1), "fig8": (7, 0)}

# Axis calibrations: list of (pixel, value) from tick-mark centres (measured on the
# native bitmap; tick runs on the frame colour). Linear fits are used.
CAL = {
    "fig5a": {"x": [(174.5, 0), (322, 30), (471, 60), (619.5, 90), (769, 120), (918.5, 150), (1065.5, 180)],
              "y": [(33, 5.5), (228.5, 5.0), (425, 4.5)]},
    "fig5b": {"x": "fig5a", "y": [(517, 2.0), (664, 1.5)]},
    "fig5c": {"x": "fig5a", "y": [(757.5, 0.65), (906, 0.55)]},
    "fig7a": {"x": [(266.5, 0), (535, 1), (804, 2)], "y": [(634.5, 0), (355, 50), (76, 100)]},
    "fig7b": {"x": [(1232, 2), (1457, 3), (1680, 4), (1903, 5)],
              "y": [(633, 0), (494, 200), (355, 400), (215, 600), (76, 800)]},
    "fig8a": {"x": [(181, 0), (327, 30), (471.5, 60), (616.5, 90), (761.5, 120), (906.5, 150), (1051.5, 180)],
              "y": [(94.5, 5.1), (246.5, 5.0), (399, 4.9), (551, 4.8), (703.5, 4.7), (856.5, 4.6), (1008.5, 4.5)]},
    "fig8b": {"x": [(1272, 50), (1395.5, 60), (1517.5, 70), (1639.5, 80), (1761.5, 90), (1883.5, 100),
                    (2006, 110), (2126.5, 120)],
              "y": [(94.5, 7.5), (246, 7.0), (399, 6.5), (552, 6.0), (704, 5.5), (856.5, 5.0), (1008.5, 4.5)]},
}

# Plot-interior boxes (x0, y0, x1, y1) in pixels, inset from the frame, plus exclusion boxes
# (legends, panel labels, text).
BOX = {
    "fig5a": (164, 24, 1076, 416), "fig5b": (164, 493, 1076, 688), "fig5c": (164, 762, 1076, 901),
    "fig7a": (271, 81, 933, 629), "fig7b": (1237, 81, 1899, 628),  # line/dots: black<25 excludes frame+ticks (38-grey)
    "fig8a": (170, 88, 1062, 1015), "fig8b": (1262, 88, 2136, 1015),
}
EXCL = {
    "fig5a": [(700, 320, 860, 380), (960, 45, 1060, 100)],   # 'TM010' text, '(a)'
    "fig5b": [(960, 495, 1060, 540)], "fig5c": [(960, 765, 1060, 810)],
    "fig7a": [(276, 115, 780, 265)], "fig7b": [(1253, 115, 1757, 265)],
    "fig8a": [(405, 772, 1024, 984)], "fig8b": [(1338, 755, 2092, 964)],
}


def fit(pairs):
    p = np.array(pairs, float)
    A = np.vstack([p[:, 0], np.ones(len(p))]).T
    coef, res, *_ = np.linalg.lstsq(A, p[:, 1], rcond=None)
    resid = p[:, 1] - A @ coef
    return coef, float(np.max(np.abs(resid))) if len(p) > 2 else 0.0


def calib(panel):
    c = CAL[panel]
    cx = CAL[c["x"]]["x"] if isinstance(c["x"], str) else c["x"]
    (ax, bx), rx = fit(cx)
    (ay, by), ry = fit(c["y"])
    return (lambda px: ax * px + bx), (lambda py: ay * py + by), dict(
        x=[list(t) for t in cx], y=[list(t) for t in c["y"]], dx_per_px=ax, dy_per_px=ay,
        max_tick_residual_x=rx, max_tick_residual_y=ry, log=False)


def load(pdf, fig):
    doc = fitz.open(pdf)
    page_i, img_i = FIGS[fig]
    page = doc[page_i]
    imgs = sorted(page.get_images(full=True), key=lambda im: -im[2] * im[3])
    pm = fitz.Pixmap(doc, imgs[img_i][0])
    if pm.colorspace and pm.colorspace.n != 3:
        pm = fitz.Pixmap(fitz.csRGB, pm)
    if pm.alpha:
        pm = fitz.Pixmap(pm, 0)
    arr = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].copy()
    return arr, page_i + 1, [round(v, 2) for v in page.get_image_rects(imgs[img_i][0])[0]]


def panel_mask(mask, panel):
    m = np.zeros_like(mask)
    x0, y0, x1, y1 = BOX[panel]
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    for (a, b, c, d) in EXCL.get(panel, []):
        m[b:d, a:c] = False
    return m


def color_mask(a, rgb, tol):
    return np.abs(a.astype(int) - np.array(rgb)).max(2) < tol


def centroids(mask, erode, amin, amax=1e9, split_wide=None):
    """Centroids of markers; erosion separates touching markers. split_wide=(spacing_px)
    splits a still-merged blob along x into round(width/spacing) pieces."""
    st = nd.generate_binary_structure(2, 1)
    m = nd.binary_erosion(mask, st, iterations=erode) if erode else mask
    lab, n = nd.label(m)
    out = []
    for i, sl in enumerate(nd.find_objects(lab), start=1):
        comp = lab[sl] == i
        area = comp.sum()
        if area < amin or area > amax:
            continue
        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        w = xs.max() - xs.min() + 1
        k = 1 if split_wide is None else max(1, int(round(w / split_wide)))
        if k == 1:
            out.append((xs.mean(), ys.mean()))
        else:
            edges = np.linspace(xs.min(), xs.max() + 1, k + 1)
            for j in range(k):
                s = (xs >= edges[j]) & (xs < edges[j + 1])
                if s.sum():
                    out.append((xs[s].mean(), ys[s].mean()))
    return sorted(out)


def grid_phase(mask, xs, search=4.0):
    """Sub-pixel shift of a regular x grid that maximises the marker-mask column profile
    (absorbs the ~1 px bias of the tick-based x calibration)."""
    prof = mask.sum(0).astype(float)
    prof = nd.gaussian_filter1d(prof, 1.5)
    best = (-1, 0.0)
    for d in np.arange(-search, search + 1e-9, 0.1):
        v = np.interp(np.asarray(xs) + d, np.arange(len(prof)), prof).sum()
        if v > best[0]:
            best = (v, d)
    return best[1]


def grid_runs(mask, xs, halfw=2, diam=None, minrun=3):
    """For each grid column x: vertical runs of marker pixels within x+-halfw; each run's
    centre is one marker (runs longer than 1.5*diam are split evenly)."""
    out = []
    for n, x in enumerate(xs):
        xi = int(round(x))
        col = mask[:, xi - halfw:xi + halfw + 1].any(1)
        if not col.any() and n in (0, len(xs) - 1):
            # end markers are half hidden under the axes frame: sample 4-6 px inside it
            sh = 5 if n == 0 else -5
            col = mask[:, xi + sh - 1:xi + sh + 2].any(1)
        ys = np.nonzero(col)[0]
        if len(ys) == 0:
            out.append((x, []))
            continue
        runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
        cents = []
        for r in runs:
            if len(r) < minrun:
                continue
            k = 1 if diam is None else max(1, int(round(len(r) / diam)))
            edges = np.linspace(r[0], r[-1] + 1, k + 1)
            for j in range(k):
                cents.append(0.5 * (edges[j] + edges[j + 1]) - 0.5)
        out.append((x, cents))
    return out


def trace_line(mask):
    """Column-wise centre of a thin line."""
    pts = []
    for x in range(mask.shape[1]):
        ys = np.nonzero(mask[:, x])[0]
        if len(ys) == 0:
            continue
        # take the longest contiguous run (the line), ignore isolated specks
        runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
        r = max(runs, key=len)
        if len(r) >= 3:
            pts.append((x + 0.0, r.mean()))
    return pts


def to_data(pts, fx, fy, nd_x=3, nd_y=5):
    return [[round(float(fx(x)), nd_x), round(float(fy(y)), nd_y)] for x, y in pts]


def overlay(arr, series, cal_panels, path, scale=1.0):
    im = Image.fromarray(arr).convert("RGB")
    dr = ImageDraw.Draw(im)
    for pts, col, shape in series:
        for x, y in pts:
            r = 4
            if shape == "x":
                dr.line([x - r, y - r, x + r, y + r], fill=col, width=2)
                dr.line([x - r, y + r, x + r, y - r], fill=col, width=2)
            elif shape == "+":
                dr.line([x - r, y, x + r, y], fill=col, width=2)
                dr.line([x, y - r, x, y + r], fill=col, width=2)
            else:
                dr.point([(x, y)], fill=col)
                dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=col)
    for panel in cal_panels:
        c = CAL[panel]
        cx = CAL[c["x"]]["x"] if isinstance(c["x"], str) else c["x"]
        x0, y0, x1, y1 = BOX[panel]
        for px, _ in cx:  # calibration ticks: magenta marks just below the frame
            dr.line([px, y1 + 6, px, y1 + 18], fill=(255, 0, 255), width=2)
        for py, _ in c["y"]:
            dr.line([x0 - 18, py, x0 - 6, py], fill=(255, 0, 255), width=2)
    im.save(path, optimize=True)


def main():
    pdf = sys.argv[1]
    refs = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(HERE)
    curves, meas = [], []

    def dig(fig, page, bbox, panel, cal, method, ux, uy, basis):
        return {"method": method, "page": page, "clip_bbox_pt": bbox,
                "image": "embedded bitmap extracted at native resolution (no resampling)",
                "axis_calibration": cal, "uncertainty_x": ux, "uncertainty_y": uy,
                "uncertainty_basis": basis, "figure_png": f"{NAME}_{fig}.png",
                "overlay_png": f"{NAME}_{fig}_overlay.png", "script": SCRIPT}

    # ---------------- Fig. 5 ----------------
    # The HFSS sweep is on a regular theta grid (2.5 deg for f and Q0, 5 deg for C; verified
    # from the marker autocorrelation and isolated-marker centroids). We sample each grid
    # column (+-2 px), fit a common sub-pixel x phase, and take the centre of each vertical
    # run of marker pixels. Reported theta values are the grid values.
    a, page, bbox = load(pdf, "fig5")
    Image.fromarray(a).save(os.path.join(refs, f"{NAME}_fig5.png"), optimize=True)
    red = (a[:, :, 0] > 170) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)
    blk = a.max(2) < 25          # pure-black markers; the axes frame is (38,38,38)
    fx, fya, cala = calib("fig5a")
    _, fyb, calb = calib("fig5b")
    _, fyc, calc = calib("fig5c")
    inv = lambda th: (th - fx(0)) / (fx(1) - fx(0))
    g25 = np.arange(0, 180.01, 2.5)
    g5 = np.arange(0, 180.01, 5.0)

    def single(mask, panel, grid, diam):
        m = panel_mask(mask, panel)
        xs = inv(grid)
        d = grid_phase(m, xs)
        pts, th = [], []
        for (x, ys), t in zip(grid_runs(m, xs + d, 2, None), grid):
            if ys:
                pts.append((x, ys[0] if len(ys) == 1 else max(ys, key=lambda y: m[int(round(y)), int(round(x))])))
                th.append(t)
        return pts, th, d

    tm, th_tm, d_tm = single(red, "fig5a", g25, 16)
    q0, th_q0, d_q0 = single(blk, "fig5b", g25, 13)
    cc, th_cc, d_cc = single(blk, "fig5c", g5, 13)
    mo = panel_mask(blk, "fig5a")
    d_mo = grid_phase(mo, inv(g25))
    other, th_o = [], []
    for (x, ys), t in zip(grid_runs(mo, inv(g25) + d_mo, 2, 13), g25):
        for y in ys:
            other.append((x, y))
            th_o.append(t)
    print("x phase shifts (px):", d_tm, d_q0, d_cc, d_mo)
    overlay(a, [(tm, (0, 200, 255), "x"), (other, (0, 220, 0), "+"), (q0, (0, 220, 0), "+"),
                (cc, (0, 220, 0), "+")], ["fig5a", "fig5b", "fig5c"],
            os.path.join(refs, f"{NAME}_fig5_overlay.png"))

    def pts_on_grid(pts, th, fy, ndy):
        return [[float(t), round(float(fy(y)), ndy)] for (x, y), t in zip(pts, th)]

    m5 = ("raster, grid sampling: theta grid 2.5 deg (f, Q0) / 5 deg (C) verified from marker spacing; "
          "colour mask (red r>170,g<100,b<100 / black max<25, frame is 38-grey), column window +-2 px, "
          "common sub-pixel x phase fitted, y = centre of the vertical marker run")
    curves.append({"source": "2205.01477 Fig. 5(a)", "description": "HFSS RT eigenfrequency of the TM010-like mode (red diamonds) vs rod angle",
                   "x": {"label": "theta", "unit": "deg"}, "y": {"label": "f_c", "unit": "GHz"},
                   "series": [{"label": "TM010 (HFSS, RT)", "points": pts_on_grid(tm, th_tm, fya, 4)}],
                   "digitization": dig("fig5", page, bbox, "fig5a", cala, m5, 0.0, 0.002,
                                       "theta exact on the 2.5-deg grid; y: tick-fit + run-centre ~0.5 px, 1 px = 0.00256 GHz; text end points 4.662/4.951 GHz as check")})
    curves.append({"source": "2205.01477 Fig. 5(a)", "description": "other eigenmodes in the mode map (black dots), not assigned to branches; points closer than one marker diameter in f are merged/split by run length",
                   "x": {"label": "theta", "unit": "deg"}, "y": {"label": "f", "unit": "GHz"},
                   "series": [{"label": "other modes (unlabelled in paper)", "points": pts_on_grid(other, th_o, fya, 4)}],
                   "digitization": dig("fig5", page, bbox, "fig5a", cala, m5, 0.0, 0.004,
                                       "as TM010; near-degenerate branches (5.12/5.15 GHz merging above ~150 deg) only resolved to ~1 marker (0.03 GHz)")})
    curves.append({"source": "2205.01477 Fig. 5(b)", "description": "HFSS RT unloaded Q0 of TM010 vs rod angle",
                   "x": {"label": "theta", "unit": "deg"}, "y": {"label": "Q0", "unit": "1e4"},
                   "series": [{"label": "Q0 TM010 (HFSS, RT), units of 1e4", "points": pts_on_grid(q0, th_q0, fyb, 4)}],
                   "digitization": dig("fig5", page, bbox, "fig5b", calb, m5, 0.0, 0.004,
                                       "1 px = 0.0034e4; only two y ticks (2.0 at 517 px, 1.5 at 664 px); ~1 px total")})
    curves.append({"source": "2205.01477 Fig. 5(c)", "description": "HFSS form factor C010 (Eq. 3, uniform axial B) vs rod angle",
                   "x": {"label": "theta", "unit": "deg"}, "y": {"label": "C010", "unit": ""},
                   "series": [{"label": "C010 (HFSS)", "points": pts_on_grid(cc, th_cc, fyc, 4)}],
                   "digitization": dig("fig5", page, bbox, "fig5c", calc, m5, 0.0, 0.001,
                                       "1 px = 0.00067; y calibrated on the frame-edge ticks 0.65/0.55 only; ~1 px total. Paper's own mesh systematic on C is 1%")})

    # ---------------- Fig. 7 ----------------
    a, page, bbox = load(pdf, "fig7")
    Image.fromarray(a).save(os.path.join(refs, f"{NAME}_fig7.png"), optimize=True)
    red = color_mask(a, (254, 0, 0), 90)
    blk = a.max(2) < 25          # line core; frame/ticks are (38,38,38)
    out_series = []
    for panel, ax_lab, scale in [("fig7a", "d1", 1), ("fig7b", "d2", 1)]:
        fx, fy, cal = calib(panel)
        line = trace_line(panel_mask(blk & ~nd.binary_dilation(red, iterations=2), panel))
        line = line[::4]
        dots = centroids(panel_mask(red, panel), erode=2, amin=10)
        out_series.append((line, (0, 200, 255), "dot"))
        out_series.append((dots, (0, 220, 0), "+"))
        probe = ax_lab[-1]
        curves.append({"source": f"2205.01477 Fig. 7({'a' if probe == '1' else 'b'})",
                       "description": f"HFSS RT coupling rate kappa{probe}/2pi of probe {probe} vs insertion depth {ax_lab}, theta = 85 deg (spline through HFSS points as plotted)",
                       "x": {"label": ax_lab, "unit": "mm"}, "y": {"label": f"kappa{probe}/2pi", "unit": "kHz"},
                       "series": [{"label": "Simulation (black line)", "points": to_data(line, fx, fy, 4, 3)}],
                       "digitization": dig("fig7", page, bbox, panel, cal,
                                           "raster: black line traced column-wise (centre of longest run), every 4th column kept",
                                           0.005 if probe == "1" else 0.006, 0.4 if probe == "1" else 3.0,
                                           "line width ~5 px; 1 px = 0.0037 mm/0.18 kHz (a), 0.0045 mm/1.44 kHz (b)")})
        meas.append({"source": f"2205.01477 Fig. 7({'a' if probe == '1' else 'b'})",
                     "description": f"RT measured kappa{probe}/2pi vs {ax_lab}; ALREADY shifted by {'-0.13' if probe == '1' else '+0.64'} mm in the figure (caption)",
                     "x": {"label": ax_lab, "unit": "mm"}, "y": {"label": f"kappa{probe}/2pi", "unit": "kHz"},
                     "series": [{"label": "RT measurement (red dots)", "points": to_data(dots, fx, fy, 4, 3)}],
                     "digitization": dig("fig7", page, bbox, panel, cal, "raster: red mask, centroid", 0.005, 0.5 if probe == "1" else 3.0,
                                         "marker centroid ~0.5 px")})
    overlay(a, out_series, ["fig7a", "fig7b"], os.path.join(refs, f"{NAME}_fig7_overlay.png"))

    # ---------------- Fig. 8 ----------------
    a, page, bbox = load(pdf, "fig8")
    Image.fromarray(a).save(os.path.join(refs, f"{NAME}_fig8.png"), optimize=True)
    blue = color_mask(a, (0, 114, 190), 70)
    orng = color_mask(a, (217, 83, 25), 60)
    yell = color_mask(a, (237, 177, 30), 60)
    ov = []
    for panel, ylab, yunit, ndy, (t0, t1) in [("fig8a", "f_c", "GHz", 4, (0, 180)),
                                              ("fig8b", "Q0 (scaled, see label)", "1e4", 3, (50, 120))]:
        fx, fy, cal = calib(panel)
        inv = lambda th: (th - fx(0)) / (fx(1) - fx(0))
        res = {}
        for key, m, step in [("yel", yell, 2.5), ("blu", blue, 5.0), ("org", orng, 2.5)]:
            grid = np.arange(t0, t1 + 1e-9, step)
            mm = panel_mask(m, panel)
            d = grid_phase(mm, inv(grid))
            pts, th = [], []
            for (x, ys), t in zip(grid_runs(mm, inv(grid) + d, 2, None), grid):
                if ys:
                    pts.append((x, ys[int(np.argmin([abs(y - np.median(ys)) for y in ys]))] if len(ys) > 1 else ys[0]))
                    th.append(float(t))
            res[key] = (pts, th)
        ov += [(res["yel"][0], (0, 0, 0), "+"), (res["blu"][0], (255, 0, 255), "x"), (res["org"][0], (0, 160, 0), "x")]
        conv = lambda k: [[t, round(float(fy(y)), ndy)] for (x, y), t in zip(*res[k])]
        sub = panel[-1]
        meth = ("raster, grid sampling (sim and RT data every 2.5 deg, base-temp data every 5 deg, verified from marker centroids); "
                "colour masks yellow~(237,177,30) / blue~(0,114,190) / orange~(217,83,25) on the grey band; column window +-2 px; "
                "y = centre of the vertical marker run. Blue/orange markers are partly hidden under the yellow dots (yellow drawn last)")
        uy = 0.0015 if sub == "a" else 0.01
        basis = "1 px = 0.00066 GHz (a), 0.0033e4 (b); occluded measured markers biased by up to ~3 px"
        scale_note = "" if sub == "a" else " (plotted value = 3.2 x simulated Q0; divide by 3.2 for the HFSS RT Q0)"
        curves.append({"source": f"2205.01477 Fig. 8({sub})", "description": f"RT HFSS simulation (yellow dots){scale_note}",
                       "x": {"label": "theta", "unit": "deg"}, "y": {"label": ylab, "unit": yunit},
                       "series": [{"label": "Simulation at room temp" + (" (x3.2)" if sub == "b" else ""), "points": conv("yel")}],
                       "digitization": dig("fig8", page, bbox, panel, cal, meth, 0.0, uy, basis)})
        meas.append({"source": f"2205.01477 Fig. 8({sub})", "description": "measured TM010" + (" Q0; RT series multiplied by 3.5 in the plot, base-temp series unscaled" if sub == "b" else " frequency"),
                     "x": {"label": "theta", "unit": "deg"}, "y": {"label": ylab, "unit": yunit},
                     "series": [{"label": "Measurement at base temp (Tc = 155 mK)", "points": conv("blu")},
                                {"label": "Measurement at room temp" + (" (x3.5)" if sub == "b" else ""), "points": conv("org")}],
                     "digitization": dig("fig8", page, bbox, panel, cal, meth, 0.0, 3 * uy, basis)})
    overlay(a, ov, ["fig8a", "fig8b"], os.path.join(refs, f"{NAME}_fig8_overlay.png"))


    jpath = os.path.join(refs, f"{NAME}.json")
    with open(jpath) as f:
        J = json.load(f)
    J["reference_values"]["curves"] = curves
    J["measured_values"]["curves"] = meas
    with open(jpath, "w") as f:
        json.dump(J, f, indent=1)
        f.write("\n")
    for c in curves + meas:
        for s in c["series"]:
            print(c["source"], s["label"], len(s["points"]))


if __name__ == "__main__":
    main()
