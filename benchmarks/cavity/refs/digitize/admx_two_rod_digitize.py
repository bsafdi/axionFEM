"""Digitize the ADMX two-rod (Run 1A/1B) cavity figures.

Usage:
  python3 -I admx_two_rod_digitize.py PRL_1910.08638.pdf ANA1B_2010.06183.pdf DD_2010.00169.pdf ANA1A_2312.16668.pdf

All figures are embedded bitmaps (extracted at native resolution); 2010.00169 Fig. 5 and
2312.16668 Fig. 8a also carry hand-placed VECTOR annotations (rod circles, pivot dots, arm lines)
whose exact page coordinates are read with PyMuPDF get_drawings().

  1. 1910.08638 Fig. 2 (= 2010.06183 Fig. 16, same CST render at 2540 px): cavity outline (outer
     white boundary + inner rim edge, Kasa circle fits), rod-centre marker dots, pivot rings at the
     far ends of the alumina armatures -> rod positions, pivot radius, arm length, rod angles.
  2. 2010.00169 Fig. 5 top right (773 MHz, rods at 180 / 270 deg): annotation geometry.
  3. 2312.16668 Fig. 8a (656.4 MHz, rods at 120 / 0 deg): rods + outline from the raster, pivots
     from the annotation dots.
  4. 2312.16668 Fig. 8c simulated S12 mode map (rod 1 at the wall, rod 2 angle 90-155 deg):
     inverse 'inferno' colour map, per-column peaks (prominence 0.04), branches by guide curves.
  5. 2312.16668 Fig. 4c simulated C010 vs f (COMSOL per the text) with its 6% band; measured Q_L.
  6. 2010.06183 Fig. 8 simulated C vs f (CST), Run 1B.

Writes PNG crops/overlays into the refs directory and digitize/admx_two_rod_digitized.json.
"""
import json
import os
import sys

import numpy as np  # numpy/scipy before fitz (libstdc++ load order)
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.dirname(HERE)
NAME = "admx_two_rod"
R_MM = 210.0          # cavity inner radius (2408.03444 p.52: inner diameter 42 cm)
RROD_MM = 25.4        # rod radius (2010.06183 p.3: 50.8 mm diameter)


def P(name):
    return os.path.join(REFS, f"{NAME}_{name}")


def xref_png(pdf, xref):
    d = fitz.open(pdf)
    pix = fitz.Pixmap(d, xref)
    if pix.n - pix.alpha >= 4 or pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()
    return a


def kasa(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.c_[xs, ys, np.ones_like(xs)]
    c, *_ = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx ** 2 + cy ** 2)
    res = np.hypot(xs - cx, ys - cy) - r
    return float(cx), float(cy), float(r), float(res.std())


def circle(dr, c, r, col, w=2):
    dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=col, width=w)


def cross(dr, c, s, col, w=2):
    dr.line([c[0] - s, c[1], c[0] + s, c[1]], fill=col, width=w)
    dr.line([c[0], c[1] - s, c[0], c[1] + s], fill=col, width=w)


def ang_ccw_from_outward(pivot, rod, centre):
    """Rod angle theta (deg, counter-clockwise as seen in the image, y up) measured from the
    radially outward direction at the pivot (theta = 0: rod at the wall, 180: toward the centre)."""
    out = np.array([pivot[0] - centre[0], -(pivot[1] - centre[1])])
    arm = np.array([rod[0] - pivot[0], -(rod[1] - pivot[1])])
    t = np.degrees(np.arctan2(arm[1], arm[0]) - np.arctan2(out[1], out[0]))
    return float(t % 360.0)


# ---------------------------------------------------------------- 1. Run 1B Fig. 2 / Fig. 16
def fig2_measure(a, panels, tag):
    L = a.astype(int).sum(2)

    def outer(cx0, cy0, rmin, rmax):
        xs, ys = [], []
        for t in np.linspace(0, 2 * np.pi, 720, endpoint=False):
            for rr in np.arange(rmin, rmax, 0.5):
                x = cx0 + rr * np.cos(t); y = cy0 + rr * np.sin(t)
                if L[int(round(y)), int(round(x))] > 735:
                    xs.append(x); ys.append(y); break
        return kasa(xs, ys)

    def inner(cx0, cy0, rmin, rmax):
        xs, ys = [], []
        for t in np.linspace(0, 2 * np.pi, 720, endpoint=False):
            for rr in np.arange(rmax, rmin, -0.5):
                x = cx0 + rr * np.cos(t); y = cy0 + rr * np.sin(t)
                p = a[int(round(y)), int(round(x))].astype(int)
                if L[int(round(y)), int(round(x))] <= 735 and not (
                        p[0] > 140 and p[1] > 140 and abs(p[0] - p[1]) < 20 and p[2] > 95):
                    xs.append(x); ys.append(y); break
        return kasa(xs, ys)

    def dot(x0, y0, hw):
        sub = L[y0 - hw:y0 + hw, x0 - hw:x0 + hw]
        m = sub < sub.min() + 40
        yy, xx = np.nonzero(m)
        return float(xx.mean() + x0 - hw), float(yy.mean() + y0 - hw)

    def ring(x0, y0, hw, thr):
        sub = L[y0 - hw:y0 + hw, x0 - hw:x0 + hw]
        blk = a[y0 - hw:y0 + hw, x0 - hw:x0 + hw].astype(int)
        sat = blk.max(2) - blk.min(2)
        yy, xx = np.nonzero((sub > thr) & (sat < 70))
        return kasa(xx + x0 - hw, yy + y0 - hw)

    out = {}
    for pn, g in panels.items():
        s = g["scale"]
        oc = outer(g["c"][0], g["c"][1], 420 * s, 560 * s)
        ic = inner(oc[0], oc[1], 400 * s, 500 * s)
        cen = (oc[0], oc[1])
        rods = [dot(int(round(x)), int(round(y)), max(6, int(14 * s))) for (x, y) in g["rods"]]
        pivs = [ring(int(round(x)), int(round(y)), max(9, int(19 * s)), g["ring_thr"])[:3] for (x, y) in g["pivots"]]
        res = {"outer_circle_px": oc, "inner_rim_circle_px": ic, "rods_px": rods,
               "pivots_px": [list(p[:2]) for p in pivs], "pivot_ring_radius_px": [p[2] for p in pivs]}
        for sc_name, Rpx in (("inner_rim_edge_is_wall", ic[2]), ("outer_rim_edge_is_wall", oc[2])):
            mm = R_MM / Rpx
            geo = []
            for k in range(2):
                rod, piv = np.array(rods[k]), np.array(pivs[k][:2])
                geo.append({
                    "rod_xy_mm": [(rod[0] - cen[0]) * mm, -(rod[1] - cen[1]) * mm],
                    "rod_r_mm": float(np.hypot(*(rod - cen)) * mm),
                    "pivot_xy_mm": [(piv[0] - cen[0]) * mm, -(piv[1] - cen[1]) * mm],
                    "pivot_r_mm": float(np.hypot(*(piv - cen)) * mm),
                    "arm_mm": float(np.hypot(*(rod - piv)) * mm),
                    "theta_ccw_deg": ang_ccw_from_outward(piv, rod, cen)})
            res[sc_name] = {"mm_per_px": mm, "rods": geo}
        out[pn] = res
    return out


def fig2_overlay(a, meas, path, scale_key="outer_rim_edge_is_wall"):
    im = Image.fromarray(a).convert("RGB")
    dr = ImageDraw.Draw(im)
    for pn, m in meas.items():
        oc, ic = m["outer_circle_px"], m["inner_rim_circle_px"]
        circle(dr, oc[:2], oc[2], (255, 0, 255), 2)
        circle(dr, ic[:2], ic[2], (0, 0, 0), 1)
        cross(dr, oc[:2], 10, (0, 0, 0), 2)
        mm = m[scale_key]["mm_per_px"]
        for k in range(2):
            rod, piv = m["rods_px"][k], m["pivots_px"][k]
            dr.line([piv[0], piv[1], rod[0], rod[1]], fill=(255, 0, 0), width=3)
            cross(dr, piv, 8, (255, 0, 0), 3)
            circle(dr, rod, RROD_MM / mm, (255, 0, 0), 3)
            cross(dr, rod, 6, (0, 0, 0), 2)
        # pivot circle of radius r_p (mean of two pivots)
        rp = np.mean([np.hypot(p[0] - oc[0], p[1] - oc[1]) for p in m["pivots_px"]])
        circle(dr, oc[:2], rp, (255, 255, 255), 1)
    im.save(path)


# ---------------------------------------------------------------- 2. 2010.00169 Fig. 5 annotations
def dd_fig5(pdf):
    d = fitz.open(pdf)
    p = d[5]
    clip = fitz.Rect(170, 55, 290, 160)
    circles = []
    for dr in p.get_drawings():
        if not dr["rect"].intersects(clip):
            continue
        r = dr["rect"]
        if dr.get("fill") == (0.0, 0.0, 0.0) and abs(r.width - r.height) < 0.05 and 10 < r.width < 12:
            circles.append(("rod", (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width / 2))
        if dr.get("color") == (0.0, 0.0, 0.0) and dr.get("fill") is None and 80 < r.width < 90:
            circles.append(("outline", (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width / 2))
    # pivots: corner points of the dashed angle arcs (items 13/14 in the stream): the arc closing
    # segments end at the arc centre.  Read them from the dashed paths.
    piv = []
    for dr in p.get_drawings():
        if dr["rect"].intersects(clip) and dr.get("dashes") and dr["dashes"] not in ("[] 0",) and dr.get("color") == (1.0, 1.0, 1.0):
            ls = [it for it in dr["items"] if it[0] == "l"]
            # the shared vertex of the two straight segments is the arc centre (pivot)
            pts = [(round(q.x, 3), round(q.y, 3)) for it in ls for q in it[1:]]
            from collections import Counter
            cnt = Counter(pts)
            piv.append(max(cnt, key=cnt.get))
    piv = sorted(piv)
    rods = sorted([c for c in circles if c[0] == "rod"], key=lambda c: c[1])
    outl = [c for c in circles if c[0] == "outline"][0]
    cen = (outl[1], outl[2])
    mm = R_MM / outl[3]
    res = {"outline_pt": outl[1:], "rods_pt": [c[1:] for c in rods], "pivots_pt": piv, "mm_per_pt": mm,
           "labels": {"left": "180 deg", "right": "270 deg", "f": "773 MHz"}}
    geo = []
    for k in range(2):
        rod = np.array(rods[k][1:3]); pv = np.array(piv[k])
        geo.append({"rod_r_mm": float(np.hypot(*(rod - cen)) * mm), "pivot_r_mm": float(np.hypot(*(pv - cen)) * mm),
                    "arm_mm": float(np.hypot(*(rod - pv)) * mm), "rod_radius_drawn_mm": rods[k][3] * mm,
                    "theta_ccw_deg": ang_ccw_from_outward(pv, rod, cen)})
    res["geometry"] = geo
    # crop + overlay at 600 dpi
    z = 600 / 72
    pm = p.get_pixmap(dpi=600, clip=clip)
    im = Image.frombytes("RGB", (pm.w, pm.h), pm.samples)
    im.save(P("dd_fig5.png"))
    dr = ImageDraw.Draw(im)
    T = lambda x, y: ((x - clip.x0) * z, (y - clip.y0) * z)
    circle(dr, T(*cen), outl[3] * z, (255, 0, 255), 3)
    cross(dr, T(*cen), 15, (255, 0, 255), 3)
    for k in range(2):
        circle(dr, T(*rods[k][1:3]), rods[k][3] * z, (255, 0, 0), 3)
        cross(dr, T(*piv[k]), 12, (255, 0, 0), 3)
        circle(dr, T(*rods[k][1:3]), RROD_MM / mm * z, (0, 255, 0), 2)
    im.save(P("dd_fig5_overlay.png"))
    return res


# ---------------------------------------------------------------- 3. 2312.16668 Fig. 8a
def a1_fig8a(pdf):
    a = xref_png(pdf, 608)
    L = a.astype(int).sum(2)
    img_rect = (54.0, 52.0760498046875, 169.88943481445312, 142.03114318847656)
    s = a.shape[1] / (img_rect[2] - img_rect[0])
    # cavity outline: dark outline pixels; search outward along rays from a rough centre
    c0 = (234.0, 247.0)
    xs, ys = [], []
    for t in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        prof = []
        for rr in np.arange(200, 240, 0.5):
            x = c0[0] + rr * np.cos(t); y = c0[1] + rr * np.sin(t)
            if 0 <= x < a.shape[1] - 1 and 0 <= y < a.shape[0] - 1:
                prof.append((L[int(round(y)), int(round(x))], x, y))
        if prof:
            v, x, y = min(prof)
            if v < 200:
                xs.append(x); ys.append(y)
    oc = kasa(xs, ys)
    # rods: dark outline rings, fit within boxes
    rods = []
    for (bx, by) in ((137, 316), (426, 257)):
        yy, xx = np.nonzero(L[by - 32:by + 32, bx - 32:bx + 32] < 150)
        rods.append(kasa(xx + bx - 32, yy + by - 32))
    # annotation pivot dots (page pt) -> raster px
    piv_pt = [(71.606, 99.62), (121.189, 99.067)]   # tiny black dots (get_drawings items 12, 15)
    rodann_pt = [(78.931, 109.438), (131.307, 99.067)]
    T = lambda x, y: ((x - img_rect[0]) * s, (y - img_rect[1]) * s)
    piv = [T(*q) for q in piv_pt]
    rodann = [T(*q) for q in rodann_pt]
    cen = oc[:2]
    mm = R_MM / oc[2]
    geo = []
    for k in range(2):
        rod = np.array(rods[k][:2]); pv = np.array(piv[k])
        geo.append({"rod_r_mm": float(np.hypot(*(rod - cen)) * mm), "rod_radius_mm": rods[k][2] * mm,
                    "pivot_r_mm_annotation": float(np.hypot(*(pv - cen)) * mm),
                    "arm_mm_annotation": float(np.hypot(*(rod - pv)) * mm),
                    "annotation_rod_offset_mm": float(np.hypot(*(np.array(rodann[k]) - rod)) * mm),
                    "theta_ccw_deg_annotation": ang_ccw_from_outward(pv, rod, cen)})
    im = Image.fromarray(a)
    im.save(P("1a_fig8a.png"))
    dr = ImageDraw.Draw(im)
    circle(dr, cen, oc[2], (255, 0, 255), 2)
    for k in range(2):
        circle(dr, rods[k][:2], rods[k][2], (255, 0, 255), 2)
        cross(dr, piv[k], 5, (0, 0, 0), 2)
        dr.line([piv[k][0], piv[k][1], rods[k][0], rods[k][1]], fill=(0, 0, 0), width=1)
    im.save(P("1a_fig8a_overlay.png"))
    return {"outline_px": oc, "rods_px": rods, "pivots_px_from_annotation": piv, "mm_per_px": mm,
            "labels": {"left": "120 deg", "right": "0 deg", "f": "656.4 MHz"}, "geometry": geo,
            "note": "rods and outline from the CST-style raster (style/colour bar are CST's although the text says COMSOL); pivots from hand-placed vector dots"}


# ---------------------------------------------------------------- 4. 2312.16668 Fig. 8c mode map
FIG8C = {"xpx": [59, 121, 184, 246, 308, 371, 433], "xval": [90, 100, 110, 120, 130, 140, 150],
         "ypx": [15, 77, 138, 200, 262, 323, 385], "yval": [0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60],
         "plot": (60, 463, 16, 384)}
BR8C = {  # guides (deg, GHz) and snap tolerance (GHz)
    "TM010c (bracketed Run 1A branch; 2312.16668: 'TM010-like ... bracketed in black')":
        ([[90, 0.6365], [100, 0.6395], [110, 0.648], [120, 0.659], [130, 0.6705], [135, 0.677],
          [140, 0.684], [145, 0.6955], [150, 0.7055], [155, 0.712]], 0.0025),
    "upper steep line (parallel to TM010c, ~16-19 MHz above: TM011-like by the empty-cavity offset)":
        ([[90, 0.643], [100, 0.653], [110, 0.6645], [120, 0.678], [130, 0.6915], [140, 0.705],
          [150, 0.719]], 0.0025),
    "lowest line (rises 617 -> 624 MHz, then flat; anti-crossing partner of TM010c near 90-100 deg)":
        ([[90, 0.617], [100, 0.622], [110, 0.624], [130, 0.6235], [155, 0.6225]], 0.002),
    "top-left line (703 -> 719 MHz, 90-111 deg)":
        ([[90, 0.703], [100, 0.709], [111, 0.7185]], 0.0025),
}


def a1_fig8c(pdf):
    from matplotlib import colormaps
    a = xref_png(pdf, 609)
    cx = np.polyfit(FIG8C["xpx"], FIG8C["xval"], 1)
    cy = np.polyfit(FIG8C["ypx"], FIG8C["yval"], 1)
    rx = np.max(np.abs(np.polyval(cx, FIG8C["xpx"]) - FIG8C["xval"]))
    ry = np.max(np.abs(np.polyval(cy, FIG8C["ypx"]) - FIG8C["yval"]))
    x0, x1, y0, y1 = FIG8C["plot"]
    lut = colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255
    sub = a[y0:y1 + 1, x0:x1 + 1].astype(float)
    dd = ((sub[:, :, None, :] - lut[None, None]) ** 2).sum(-1)
    v = dd.argmin(-1) / 255.0
    cerr = np.sqrt(dd.min(-1))
    pts = []
    for j in range(v.shape[1]):
        pk, _ = find_peaks(v[:, j], prominence=0.04)
        for q in pk:
            # sub-pixel parabolic refinement
            if 0 < q < v.shape[0] - 1:
                y_m, y_0, y_p = v[q - 1, j], v[q, j], v[q + 1, j]
                den = y_m - 2 * y_0 + y_p
                dq = 0.5 * (y_m - y_p) / den if den != 0 else 0.0
            else:
                dq = 0.0
            pts.append((float(np.polyval(cx, j + x0)), float(np.polyval(cy, q + dq + y0)), float(v[q, j])))
    pts = np.array(pts)
    series = []
    used = np.zeros(len(pts), bool)
    for lab, (anc, tol) in BR8C.items():
        anc = np.array(anc)
        g = PchipInterpolator(anc[:, 0], anc[:, 1], extrapolate=False)
        gy = g(pts[:, 0])
        m = np.isfinite(gy) & (np.abs(pts[:, 1] - gy) < tol)
        # one point per column: nearest to guide
        sel = []
        for th in np.unique(pts[m, 0]):
            k = np.nonzero(m & (pts[:, 0] == th))[0]
            sel.append(k[np.argmin(np.abs(pts[k, 1] - gy[k]))])
        used[sel] = True
        s_pts = pts[sel]
        # bin to 1 deg (median) for the JSON
        b = np.round(s_pts[:, 0])
        binned = [[float(t), round(float(np.median(s_pts[b == t, 1])), 5)] for t in np.unique(b)]
        series.append({"label": lab, "points": binned, "n_raw": int(len(sel)), "guide": anc.tolist(),
                       "snap_tol_GHz": tol})
    others = pts[~used]
    im = Image.fromarray(a).resize((a.shape[1] * 2, a.shape[0] * 2), Image.NEAREST)
    im.save(P("1a_fig8c.png"))
    dr = ImageDraw.Draw(im)
    inv_x = lambda t: (np.polyval(np.polyfit(FIG8C["xval"], FIG8C["xpx"], 1), t)) * 2
    inv_y = lambda f: (np.polyval(np.polyfit(FIG8C["yval"], FIG8C["ypx"], 1), f)) * 2
    cols = [(0, 255, 0), (0, 200, 255), (255, 255, 255), (255, 0, 255)]
    for s_, c in zip(series, cols):
        for t, f in s_["points"]:
            X, Y = inv_x(t), inv_y(f)
            dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], outline=c, width=2)
    for t, f, _ in others[::2]:
        X, Y = inv_x(t), inv_y(f)
        dr.point((X, Y), fill=(0, 0, 0))
    im.save(P("1a_fig8c_overlay.png"))
    cal = {"x": [[p, val] for p, val in zip(FIG8C["xpx"], FIG8C["xval"])],
           "y": [[p, val] for p, val in zip(FIG8C["ypx"], FIG8C["yval"])],
           "max_calibration_residual": {"deg": float(rx), "GHz": float(ry)},
           "deg_per_px": float(cx[0]), "GHz_per_px": float(-cy[0]),
           "colormap_inverse_rms_rgb_err_median": float(np.median(cerr))}
    return {"series": series, "unassigned_peaks": [[round(t, 2), round(f, 5)] for t, f, _ in others],
            "axis_calibration": cal}


# ---------------------------------------------------------------- 5/6. C(f) curves
def line_trace(a, mask, frame, cx, cy, xstep=1):
    x0, x1, y0, y1 = frame
    out = []
    for x in range(x0 + 3, x1 - 2, xstep):
        yy = np.nonzero(mask[y0 + 3:y1 - 2, x])[0]
        if len(yy) == 0:
            continue
        # take the largest connected run
        runs = np.split(yy, np.nonzero(np.diff(yy) > 1)[0] + 1)
        r = max(runs, key=len)
        out.append([float(np.polyval(cx, x)), float(np.polyval(cy, r.mean() + y0 + 3)), int(len(r))])
    return np.array(out)


def rdp(P_, eps):
    if len(P_) < 3:
        return P_
    a, b = P_[0], P_[-1]
    e = b - a
    q = P_ - a
    d = np.abs(e[0] * q[:, 1] - e[1] * q[:, 0]) / (np.hypot(*(b - a)) + 1e-12)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(P_[:i + 1], eps)[:-1], rdp(P_[i:], eps)])
    return np.vstack([a, b])


def a1_fig4c(pdf):
    a = xref_png(pdf, 416).astype(int)  # solid-line layer (417 is the dashed copy printed on the page)
    a417 = xref_png(pdf, 417).astype(int)
    frame = (329, 1698, 38, 1312)
    cx = np.polyfit([328.5, 633.0, 937.0, 1241.5, 1546.0], [640, 650, 660, 670, 680], 1)
    cyL = np.polyfit([148.0, 323.0, 497.5, 672.0, 847.0, 1021.5, 1196.0], [0.400, 0.375, 0.350, 0.325, 0.300, 0.275, 0.250], 1)
    cyR = np.polyfit([130.0, 328.0, 526.0, 723.5, 921.0, 1119.0], [35000, 30000, 25000, 20000, 15000, 10000], 1)
    R_, G_, B_ = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    blue = (B_ > 200) & (R_ < 90) & (G_ < 90)
    band = (B_ > 220) & (R_ > 170) & (R_ < 215) & (G_ > 170) & (G_ < 215)
    red = (R_ > 200) & (G_ < 80) & (B_ < 80)
    C = line_trace(a, blue, frame, cx, cyL)
    Q = line_trace(a, red, frame, cx, cyR)
    # band edges per column: topmost/bottommost band-or-line pixel in the band column set
    bandall = band | blue
    edges = []
    x0, x1, y0, y1 = frame
    for x in range(x0 + 3, x1 - 2, 4):
        yy = np.nonzero(bandall[y0 + 3:y1 - 2, x])[0]
        if len(yy) < 5:
            continue
        edges.append([float(np.polyval(cx, x)), float(np.polyval(cyL, yy.max() + y0 + 3)), float(np.polyval(cyL, yy.min() + y0 + 3))])
    # dashed layer cross-check
    b417 = (a417[:, :, 2] > 200) & (a417[:, :, 0] < 90) & (a417[:, :, 1] < 90)
    C417 = line_trace(a417, b417, frame, cx, cyL)
    common = np.intersect1d(np.round(C[:, 0], 3), np.round(C417[:, 0], 3))
    dC = [abs(np.interp(t, C[:, 0], C[:, 1]) - np.interp(t, C417[:, 0], C417[:, 1])) for t in common]
    # 1-MHz bins
    def bins(D, step=1.0, nd=4):
        b = np.round(D[:, 0] / step) * step
        return [[float(t), round(float(np.median(D[b == t, 1])), nd)] for t in np.unique(b)]
    Cb, Qb = bins(C), bins(Q, nd=0)
    eb = np.array(edges)
    bb = np.round(eb[:, 0])
    Eb = [[float(t), round(float(np.median(eb[bb == t, 1])), 4), round(float(np.median(eb[bb == t, 2])), 4)] for t in np.unique(bb)]
    im = Image.fromarray(a.astype(np.uint8)).resize((1024, 768))
    Image.fromarray(a417.astype(np.uint8)).resize((1024, 768)).save(P("1a_fig4c.png"))
    dr = ImageDraw.Draw(im)
    ix = np.polyfit([640, 650, 660, 670, 680], [328.5, 633.0, 937.0, 1241.5, 1546.0], 1)
    iyL = np.polyfit([0.400, 0.375, 0.350, 0.325, 0.300, 0.275, 0.250], [148.0, 323.0, 497.5, 672.0, 847.0, 1021.5, 1196.0], 1)
    iyR = np.polyfit([35000, 30000, 25000, 20000, 15000, 10000], [130.0, 328.0, 526.0, 723.5, 921.0, 1119.0], 1)
    for t, c in Cb:
        X, Y = np.polyval(ix, t) / 2, np.polyval(iyL, c) / 2
        dr.ellipse([X - 4, Y - 4, X + 4, Y + 4], outline=(0, 160, 0), width=2)
    for t, lo, hi in Eb[::2]:
        for c in (lo, hi):
            X, Y = np.polyval(ix, t) / 2, np.polyval(iyL, c) / 2
            dr.ellipse([X - 2, Y - 2, X + 2, Y + 2], fill=(255, 140, 0))
    for t, q in Qb:
        X, Y = np.polyval(ix, t) / 2, np.polyval(iyR, q) / 2
        dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], outline=(0, 0, 0), width=1)
    im.save(P("1a_fig4c_overlay.png"))
    return {"C_sim": Cb, "C_band_lo_hi": Eb, "QL_measured": Qb,
            "dashed_vs_solid_layer_max_abs_diff_C": float(max(dC)) if dC else None,
            "axis_calibration": {"x": [[328.5, 640], [633.0, 650], [937.0, 660], [1241.5, 670], [1546.0, 680]],
                                  "y_left": [[148.0, 0.400], [1196.0, 0.250]], "y_right": [[130.0, 35000], [1119.0, 10000]],
                                  "MHz_per_px": float(cx[0]), "C_per_px": float(-cyL[0]), "Q_per_px": float(-cyR[0])}}


def ana_fig8(pdf):
    a = xref_png(pdf, 318).astype(int)
    frame = (262, 1889, 33, 939)
    xt = [363.0, 584.0, 805.0, 1026.0, 1247.0, 1468.0, 1689.0]
    xv = [680, 700, 720, 740, 760, 780, 800]
    yt = [147.0, 313.0, 480.0, 646.0, 812.0]
    yv = [0.45, 0.40, 0.35, 0.30, 0.25]
    cx = np.polyfit(xt, xv, 1); cy = np.polyfit(yt, yv, 1)
    R_, G_, B_ = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    m = (np.abs(R_ - 31) < 45) & (np.abs(G_ - 119) < 45) & (np.abs(B_ - 180) < 45) & (B_ > R_ + 80)
    D = line_trace(a, m, frame, cx, cy)
    # vertices of the piecewise-linear CST curve (Ramer-Douglas-Peucker in pixel space, eps 3 px; vertices < 1 MHz apart merged)
    pxy = np.c_[np.polyval(np.polyfit(xv, xt, 1), D[:, 0]), np.polyval(np.polyfit(yv, yt, 1), D[:, 1])]
    V = rdp(pxy, 3.0)
    keep = [0] + [k for k in range(1, len(V)) if abs(V[k, 0] - V[k - 1, 0]) > 1.0 / abs(cx[0])]
    V = V[keep]
    # local-extremum vertices: the column-median of a thick line is biased toward the inside of a
    # sharp V; matplotlib's round joins put the outer edge of the stroke lw/2 beyond the vertex, so
    # vertex_y = extreme stroke pixel -/+ lw/2 (lw = median vertical stroke thickness on flat parts).
    lw = float(np.median(D[:, 2]))
    V = V.copy()
    refined = []
    for k in range(1, len(V) - 1):
        lo = V[k, 1] > V[k - 1, 1] and V[k, 1] > V[k + 1, 1]   # minimum of C (max image y)
        hi = V[k, 1] < V[k - 1, 1] and V[k, 1] < V[k + 1, 1]
        if not (lo or hi):
            continue
        xs_ = range(int(V[k, 0]) - 6, int(V[k, 0]) + 7)
        ys_ = [np.nonzero(m[frame[2] + 3:frame[3] - 2, x])[0] + frame[2] + 3 for x in xs_]
        ys_ = np.concatenate([y for y in ys_ if len(y)])
        y_new = (ys_.max() - lw / 2) if lo else (ys_.min() + lw / 2)
        refined.append([round(float(np.polyval(cx, V[k, 0])), 2), round(float(np.polyval(cy, V[k, 1])), 4),
                        round(float(np.polyval(cy, y_new)), 4)])
        V[k, 1] = y_new
    verts = [[round(float(np.polyval(cx, x)), 2), round(float(np.polyval(cy, y)), 4)] for x, y in V]
    b = np.round(D[:, 0])
    dense = [[float(t), round(float(np.median(D[b == t, 1])), 4)] for t in np.unique(b)]
    im = Image.fromarray(a.astype(np.uint8))
    im.resize((961, 560)).save(P("1banalysis_fig8.png"))
    dr = ImageDraw.Draw(im)
    for x, y in V:
        dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 0, 0), width=4)
    for t, c in dense[::2]:
        X = np.polyval(np.polyfit(xv, xt, 1), t); Y = np.polyval(np.polyfit(yv, yt, 1), c)
        dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=(0, 0, 0))
    im.resize((961, 560)).save(P("1banalysis_fig8_overlay.png"))
    return {"vertices": verts, "dense_1MHz": dense, "stroke_px": lw,
            "extremum_vertex_refinement_f_Cmedian_Crefined": refined,
            "axis_calibration": {"x": [[p, q] for p, q in zip(xt, xv)], "y": [[p, q] for p, q in zip(yt, yv)],
                                  "MHz_per_px": float(cx[0]), "C_per_px": float(-cy[0])}}


def main():
    prl, ana, dd, a1 = sys.argv[1:5]
    out = {}
    # 1. Fig. 2 (PRL, 1188 px) and Fig. 16 (2010.06183, 2540 px; identical render)
    a16 = xref_png(ana, 405)
    p16 = {"symmetric": {"c": (613, 734), "scale": 1.0, "ring_thr": 650,
                         "rods": [(387, 860), (838, 610)], "pivots": [(327, 736), (901, 730)]},
           "anti_symmetric": {"c": (1640, 734), "scale": 1.0, "ring_thr": 650,
                              "rods": [(1438, 834), (1833, 834)], "pivots": [(1350, 730), (1922, 728)]}}
    m16 = fig2_measure(a16, p16, "fig16")
    fig2_overlay(a16, m16, P("1banalysis_fig16_overlay.png"))
    a2 = xref_png(prl, 75)
    Image.fromarray(a2).save(P("1b_fig2.png"))
    # PRL image: same render at a different crop/scale; map Fig. 16 guesses by the outer-circle fit
    # PRL image: same render, different crop and scale (outer radius 258.7 px vs 494.4 px, scale
    # 0.5233).  Starting guesses = Fig. 16 positions mapped by that scale about the panel centres
    # (read off the PRL raster); every position is then re-fitted on the PRL pixels.
    p2 = {"symmetric": {"c": (269, 354), "scale": 0.5233, "ring_thr": 640,
                        "rods": [(150, 420), (388, 288)], "pivots": [(119, 356), (418, 352)]},
          "anti_symmetric": {"c": (804, 351), "scale": 0.5233, "ring_thr": 640,
                             "rods": [(701, 407), (907, 407)], "pivots": [(655, 352), (953, 352)]}}
    m2 = fig2_measure(a2, p2, "fig2")
    fig2_overlay(a2, m2, P("1b_fig2_overlay.png"))
    out["run1b_fig2"] = {"source": "1910.08638 Fig. 2 (embedded 1188x703 px)", "panels": m2}
    out["run1b_analysis_fig16"] = {"source": "2010.06183 Fig. 16 (same render, embedded 2540x1458 px)", "panels": m16}
    out["dd_fig5"] = dd_fig5(dd)
    out["run1a_fig8a"] = a1_fig8a(a1)
    out["run1a_fig8c"] = a1_fig8c(a1)
    out["run1a_fig4c"] = a1_fig4c(a1)
    out["run1b_analysis_fig8"] = ana_fig8(ana)
    json.dump(out, open(os.path.join(HERE, f"{NAME}_digitized.json"), "w"), indent=1)
    print(json.dumps({k: (v if k in ("dd_fig5",) else "...") for k, v in out.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
