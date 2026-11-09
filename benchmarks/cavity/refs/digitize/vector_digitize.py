"""Vector-path digitization helpers for figures embedded as PDF vector graphics.

Works on MATLAB/matplotlib-style plots: an axes frame made of straight lines, short tick
marks on the frame, numeric tick labels as text, and data curves as long polylines
with a distinct stroke colour.  Coordinates are taken directly from the PDF content
stream (page.get_drawings()), so the only digitization error is the axis calibration
(least-squares fit of tick positions to tick-label values; residual reported).

Usage from a per-benchmark script (run with `python3 -I script.py <pdf>`; the script adds
its own directory to sys.path before importing this module).
"""
import re

import numpy as np  # import before fitz (libstdc++ load order)
import fitz  # PyMuPDF

NUM = re.compile(r"^[-−]?\d+(\.\d+)?$")


def _num(s):
    s = s.strip().replace("−", "-")
    return float(s) if NUM.match(s) else None


def segments(drawing):
    """List of (p0, p1) line segments of a drawing (lines and curve chords)."""
    segs = []
    for it in drawing["items"]:
        if it[0] == "l":
            segs.append((it[1], it[2]))
        elif it[0] == "c":
            segs.append((it[1], it[4]))
    return segs


def polyline(drawing):
    """Ordered vertex list of a polyline drawing (consecutive duplicates removed)."""
    pts = []
    for p0, p1 in segments(drawing):
        for p in (p0, p1):
            if not pts or abs(pts[-1][0] - p.x) > 1e-6 or abs(pts[-1][1] - p.y) > 1e-6:
                pts.append((p.x, p.y))
    return np.array(pts)


def axis_calibration(page, frame, tick_len_max=3.0, label_gap=14.0, use_grid=False):
    """Calibrate a linear axes frame.

    frame: fitz.Rect of the axes box (x0,y0,x1,y1) in PDF points.
    Returns dict with x/y fit (value = a*pt + b), tick pairs and RMS residuals.
    """
    xt, yt = [], []
    for dr in page.get_drawings():
        for p0, p1 in segments(dr):
            # short vertical tick on bottom frame
            if abs(p0.x - p1.x) < 1e-3 and abs(p0.y - p1.y) <= tick_len_max + 1e-3 and abs(p0.y - p1.y) > 0.1:
                if frame.x0 - 0.5 <= p0.x <= frame.x1 + 0.5 and min(abs(p0.y - frame.y1), abs(p1.y - frame.y1)) < 0.2:
                    xt.append(p0.x)
            if abs(p0.y - p1.y) < 1e-3 and abs(p0.x - p1.x) <= tick_len_max + 1e-3 and abs(p0.x - p1.x) > 0.1:
                if frame.y0 - 0.5 <= p0.y <= frame.y1 + 0.5 and min(abs(p0.x - frame.x0), abs(p1.x - frame.x0)) < 0.2:
                    yt.append(p0.y)
            # full-span grid lines (used when the plot has a grid instead of / besides ticks)
            if use_grid and abs(p0.x - p1.x) < 1e-3 and abs(abs(p0.y - p1.y) - frame.height) < 0.5:
                xt.append(p0.x)
            if use_grid and abs(p0.y - p1.y) < 1e-3 and abs(abs(p0.x - p1.x) - frame.width) < 0.5:
                yt.append(p0.y)
    xt = sorted(set(round(v, 2) for v in xt))
    yt = sorted(set(round(v, 2) for v in yt))
    spans = []
    for b in page.get_text("dict", clip=fitz.Rect(frame.x0 - 40, frame.y0 - 10, frame.x1 + 15, frame.y1 + label_gap + 6))["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                v = _num(s["text"])
                if v is not None:
                    r = fitz.Rect(s["bbox"])
                    spans.append((v, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r))
    xpairs, ypairs = [], []
    for x in xt:
        cands = [s for s in spans if s[3].y0 >= frame.y1 - 1 and s[3].y0 <= frame.y1 + label_gap and abs(s[1] - x) < 8]
        if cands:
            c = min(cands, key=lambda s: abs(s[1] - x))
            xpairs.append((x, c[0]))
    for y in yt:
        cands = [s for s in spans if s[3].x1 <= frame.x0 + 1 and s[3].x1 >= frame.x0 - 40 and abs(s[2] - y) < 5]
        if cands:
            c = min(cands, key=lambda s: abs(s[2] - y))
            ypairs.append((y, c[0]))
    out = {}
    for key, pairs in (("x", xpairs), ("y", ypairs)):
        if len(pairs) < 2:
            raise RuntimeError(f"not enough {key} tick/label pairs: {pairs}")
        P = np.array(pairs)
        A = np.vstack([P[:, 0], np.ones(len(P))]).T
        (a, b), *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
        res = P[:, 1] - (a * P[:, 0] + b)
        out[key] = {"a": a, "b": b, "pairs": [[float(p), float(v)] for p, v in pairs],
                    "rms_residual": float(np.sqrt(np.mean(res ** 2)))}
    return out


def to_data(cal, pts):
    pts = np.asarray(pts, float)
    return np.column_stack([cal["x"]["a"] * pts[:, 0] + cal["x"]["b"], cal["y"]["a"] * pts[:, 1] + cal["y"]["b"]])


def curves_in(page, frame, color, tol=0.08, min_items=10, dashes=None):
    """Polylines inside the frame with stroke colour ~color (and optional dash filter)."""
    out = []
    for dr in page.get_drawings():
        c = dr.get("color")
        if c is None or len(dr["items"]) < min_items:
            continue
        if max(abs(a - b) for a, b in zip(c, color)) > tol:
            continue
        r = dr["rect"]
        if not (r.x0 >= frame.x0 - 2 and r.x1 <= frame.x1 + 2 and r.y0 >= frame.y0 - 2 and r.y1 <= frame.y1 + 2):
            continue
        if dashes is not None and (dr.get("dashes") or "") != dashes and not dashes(dr.get("dashes")):
            continue
        out.append(dr)
    return out


def thin(xy, nmax=300, keep_extrema=True):
    """Downsample a dense polyline to <= nmax points while keeping local extrema in y."""
    xy = np.asarray(xy)
    if len(xy) <= nmax:
        return xy
    idx = set(np.linspace(0, len(xy) - 1, nmax).round().astype(int))
    if keep_extrema:
        y = xy[:, 1]
        for i in range(1, len(y) - 1):
            if (y[i] > y[i - 1] and y[i] >= y[i + 1]) or (y[i] < y[i - 1] and y[i] <= y[i + 1]):
                idx.add(i)
    return xy[sorted(idx)]


def render_crop(page, clip, path, dpi=250):
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    pix.save(path)
    return dpi


def overlay(page, clip, cal, series, path, dpi=250):
    """Render crop and draw extracted points (data coords) as markers; ticks marked in green."""
    from PIL import Image, ImageDraw

    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(img)
    s = dpi / 72.0
    cols = [(255, 140, 0), (0, 170, 0), (200, 0, 200), (0, 160, 160), (120, 60, 0)]

    def topix(x, y):
        X = ((x - cal["x"]["b"]) / cal["x"]["a"] - clip.x0) * s
        Y = ((y - cal["y"]["b"]) / cal["y"]["a"] - clip.y0) * s
        return X, Y

    for k, (label, pts) in enumerate(series):
        c = cols[k % len(cols)]
        for x, y in pts:
            X, Y = topix(x, y)
            dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], outline=c, width=1)
    for p, v in cal["x"]["pairs"]:
        X = (p - clip.x0) * s
        dr.line([X, img.height - 6, X, img.height], fill=(0, 200, 0), width=2)
    for p, v in cal["y"]["pairs"]:
        Y = (p - clip.y0) * s
        dr.line([0, Y, 6, Y], fill=(0, 200, 0), width=2)
    img.save(path)
