"""Shared helpers for raster-figure digitization of cavity reference figures.

Used by the per-benchmark *_digitize.py scripts. Pure functions: render a PDF
clip, build color masks, find marker centroids, calibrate linear axes from
gridline pixel positions, and draw verification overlays.
"""
import json
import os

import fitz  # PyMuPDF
import numpy as np
from scipy import ndimage as nd


def render(pdf, page, clip, dpi):
    """Render page `page` (0-based) restricted to `clip` (pt) at `dpi`; return HxWx3 int array + pixmap."""
    doc = fitz.open(pdf)
    pm = doc[page].get_pixmap(dpi=dpi, clip=fitz.Rect(*clip))
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
    return a, pm


def lin_cal(pairs):
    """pairs = [(pixel, value), ...] -> (slope, intercept, rms residual in value units) of least-squares fit."""
    p = np.array(pairs, float)
    A = np.vstack([p[:, 0], np.ones(len(p))]).T
    (m, b), *_ = np.linalg.lstsq(A, p[:, 1], rcond=None)
    res = p[:, 1] - (m * p[:, 0] + b)
    return float(m), float(b), float(np.sqrt(np.mean(res ** 2)))


def blobs(mask, min_px=20, max_px=None, dilate=0, box=None):
    """Centroids (x, y) of connected components of `mask` within box=(x0,y0,x1,y1)."""
    m = mask.copy()
    if box is not None:
        x0, y0, x1, y1 = box
        z = np.zeros_like(m)
        z[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        m = z
    if dilate:
        m = nd.binary_dilation(m, iterations=dilate)
    lab, n = nd.label(m)
    out = []
    for i in range(1, n + 1):
        cnt = int((lab == i).sum())
        if cnt < min_px or (max_px and cnt > max_px):
            continue
        cy, cx = nd.center_of_mass(lab == i)
        out.append((float(cx), float(cy)))
    return out


def crosses(mask, L=17, box=None):
    """Centres (x, y) of '+' markers: pixels whose horizontal AND vertical L-window are fully inside mask."""
    m = mask.astype(float)
    k = np.ones(L)
    h = nd.convolve1d(m, k, axis=1, mode="constant") >= L - 0.5
    v = nd.convolve1d(m, k, axis=0, mode="constant") >= L - 0.5
    q = h & v
    if box is not None:
        x0, y0, x1, y1 = box
        z = np.zeros_like(q)
        z[y0:y1, x0:x1] = q[y0:y1, x0:x1]
        q = z
    lab, n = nd.label(q, structure=np.ones((3, 3)))
    if n == 0:
        return []
    c = nd.center_of_mass(q, lab, range(1, n + 1))
    return [(float(x), float(y)) for y, x in c]


def save_png(a, path):
    pm = fitz.Pixmap(fitz.csRGB, a.shape[1], a.shape[0], np.ascontiguousarray(a.astype(np.uint8)).tobytes(), False)
    pm.save(path)


def overlay(a, groups, path, cal_lines=(), r=11, t=3):
    """Draw markers on a copy of `a`. groups = [(points_px, (r,g,b), shape)], shape 's' (square) or 'x'.
    cal_lines = [('h'|'v', pixel, (r,g,b))] dotted calibration lines."""
    o = a.copy()
    H, W = o.shape[:2]
    for kind, pix, col in cal_lines:
        p = int(round(pix))
        if kind == "h" and 0 <= p < H:
            o[p, ::4] = col
        if kind == "v" and 0 <= p < W:
            o[::4, p] = col
    for pts, col, shape in groups:
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if shape == "s":
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) > r - t:
                            X, Y = xi + dx, yi + dy
                            if 0 <= X < W and 0 <= Y < H:
                                o[Y, X] = col
            else:
                for d in range(-r, r + 1):
                    for w in range(-(t // 2), t // 2 + 1):
                        for X, Y in ((xi + d + w, yi + d), (xi + d + w, yi - d)):
                            if 0 <= X < W and 0 <= Y < H:
                                o[Y, X] = col
    save_png(o, path)


def ring_template_peaks(mask, half_w, half_h, thr=0.55, nms=8, box=None):
    """Hollow-diamond detector: fraction of a diamond ring (|x|/hw+|y|/hh in [0.8,1.1]) covered by mask;
    returns local maxima (x, y) above thr."""
    yy, xx = np.mgrid[-half_h - 2:half_h + 3, -half_w - 2:half_w + 3]
    rr = np.abs(xx) / half_w + np.abs(yy) / half_h
    T = ((rr > 0.8) & (rr < 1.1)).astype(float)
    T /= T.sum()
    sc = nd.correlate(mask.astype(float), T, mode="constant")
    mx = nd.maximum_filter(sc, size=2 * nms + 1)
    pk = (sc == mx) & (sc > thr)
    if box is not None:
        x0, y0, x1, y1 = box
        z = np.zeros_like(pk)
        z[y0:y1, x0:x1] = pk[y0:y1, x0:x1]
        pk = z
    lab, n = nd.label(pk, structure=np.ones((3, 3)))
    if n == 0:
        return []
    c = nd.center_of_mass(pk, lab, range(1, n + 1))
    pts = sorted(((float(x), float(y), float(sc[int(round(y)), int(round(x))])) for y, x in c), key=lambda t: -t[2])
    keep = []  # greedy non-maximum suppression (plateaus/ties give several maxima per marker)
    for x, y, v in pts:
        if all((x - u) ** 2 + (y - w) ** 2 > (1.5 * nms) ** 2 for u, w in keep):
            keep.append((x, y))
    return keep


def downsample(a, f):
    """Box-average downsample by integer factor f (for modest PNG sizes)."""
    H, W = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:H, :W].reshape(H // f, f, W // f, f, 3).mean((1, 3)).astype(int)


def update_curves(json_path, curves):
    """Replace/insert curves (matched by 'source' + 'description') in reference_values.curves / measured_values.curves."""
    with open(json_path) as f:
        J = json.load(f)
    for sect, cv in curves:
        lst = J.setdefault(sect, {}).setdefault("curves", [])
        lst[:] = [c for c in lst if not (c["source"] == cv["source"] and c["description"] == cv["description"])]
        lst.append(cv)
    with open(json_path, "w") as f:
        json.dump(J, f, indent=1)
        f.write("\n")


REFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
