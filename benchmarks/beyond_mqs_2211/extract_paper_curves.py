#!/usr/bin/env python
"""Extract the plotted data of arXiv:2211.00008 (Benabou, Foster, Kahn, Safdi, Salemi 2022) from its vector figures.

The figures in the arXiv source are matplotlib PDFs, so the curves are stored as vector paths. This script recovers
them exactly instead of digitizing by eye:

1. The plot areas are the axes background rectangles (filled, unstroked).
2. Each axis is calibrated from its tick labels (text spans; a "10" followed by a raised, smaller span is 10^k),
   each label matched to the nearest major tick mark. Linear or log scale is detected from the label values, and
   the fit residual (PDF points) is recorded.
3. Every stroked path inside a plot area becomes a curve (vertices mapped to data coordinates, style kept), and
   every filled polygon becomes a region. Legend samples (short horizontal strokes next to text) give each style
   its legend label.

Usage:
    python extract_paper_curves.py [--src DIR] [--out paper_curves]

``--src`` is the unpacked arXiv source (it must contain Plots/*.pdf); without it, the source is downloaded from
arXiv into ./paper/ (git-ignored: papers are not redistributed, docs/references.md). Output: one JSON per figure PDF.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
ARXIV = "2211.00008"
NUM = re.compile(r"^[−\-]?\d+(\.\d+)?$")


def fetch_source(dest: Path) -> Path:
    if (dest / "Plots").is_dir():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(f"https://arxiv.org/e-print/{ARXIV}", timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts:
                tf.extract(m, dest)
    return dest


def _num(t: str) -> float:
    return float(t.replace("−", "-"))


def tick_labels(page) -> list[dict]:
    """Numeric labels with their centre; '10' + raised superscript -> 10^k."""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                if t:
                    spans.append({"t": t, "bb": s["bbox"], "size": s["size"]})
    out, used = [], set()
    for i, s in enumerate(spans):
        if i in used or not NUM.match(s["t"]):
            continue
        x0, y0, x1, y1 = s["bb"]
        val = _num(s["t"])
        # superscript: the next span starts at x1, is smaller and raised
        if s["t"] == "10":
            exp, j, xe = "", i + 1, x1
            while j < len(spans):  # superscript spans: adjacent, smaller, raised ("−", "17" may be separate)
                n = spans[j]
                if abs(n["bb"][0] - xe) < 1.0 and n["size"] < 0.85 * s["size"] and n["bb"][1] < y0 and \
                        re.fullmatch(r"[−\-]?\d*", n["t"]):
                    exp += n["t"]
                    xe = n["bb"][2]
                    used.add(j)
                    j += 1
                else:
                    break
            if exp and NUM.match(exp):
                val = 10.0 ** _num(exp)
                x1 = xe
        out.append({"value": val, "cx": 0.5 * (x0 + x1), "cy": 0.5 * (y0 + y1), "bbox": [x0, y0, x1, y1],
                    "text": s["t"]})
    return out


def axes_rects(page) -> list[pymupdf.Rect]:
    """Axes background patches: filled, unstroked rectangles at least 80 pt on a side, not inside another one."""
    rects = []
    for d in page.get_drawings():
        it = d["items"]
        if len(it) == 1 and it[0][0] == "re" and d.get("color") is None and d.get("fill") is not None:
            r = it[0][1]
            if r.width > 80 and r.height > 80:
                rects.append(pymupdf.Rect(r))
    rects = [r for r in rects if not any(o != r and o.contains(r) for o in rects)]
    # a figure's own background is larger than everything and contains the axes: drop rects containing others
    rects = [r for r in rects if not any(o != r and r.contains(o) for o in rects)]
    return sorted(rects, key=lambda r: (round(r.y0), r.x0))


def major_ticks(page, r, top: bool = False) -> tuple[list[float], list[float]]:
    """x positions of vertical ticks on the bottom (or top) edge, y positions of horizontal ticks on the left edge."""
    xs, ys = [], []
    edge = r.y0 if top else r.y1
    for d in page.get_drawings():
        it = d["items"]
        if len(it) != 1 or it[0][0] != "l" or d.get("color") is None:
            continue
        p, q = it[0][1], it[0][2]
        L = math.hypot(q.x - p.x, q.y - p.y)
        if not 3.0 <= L <= 8.0:
            continue
        if abs(p.x - q.x) < 0.01 and r.x0 - 0.5 <= p.x <= r.x1 + 0.5 and min(abs(p.y - edge), abs(q.y - edge)) < 0.5:
            xs.append(p.x)
        if abs(p.y - q.y) < 0.01 and r.y0 - 0.5 <= p.y <= r.y1 + 0.5 and min(abs(p.x - r.x0), abs(q.x - r.x0)) < 0.5:
            ys.append(p.y)
    return sorted(set(round(v, 3) for v in xs)), sorted(set(round(v, 3) for v in ys))


def minor_ticks(page, r, axis: str, top: bool = False) -> list[float]:
    """Positions of short (1-3 pt) ticks on the bottom/top (axis 'x') or left (axis 'y') edge."""
    out = []
    edge = r.y0 if top else r.y1
    for d in page.get_drawings():
        it = d["items"]
        if len(it) != 1 or it[0][0] != "l" or d.get("color") is None:
            continue
        p, q = it[0][1], it[0][2]
        L = math.hypot(q.x - p.x, q.y - p.y)
        if not 1.0 <= L < 3.0:
            continue
        if axis == "x" and abs(p.x - q.x) < 0.01 and min(abs(p.y - edge), abs(q.y - edge)) < 0.5:
            out.append(p.x)
        if axis == "y" and abs(p.y - q.y) < 0.01 and min(abs(p.x - r.x0), abs(q.x - r.x0)) < 0.5:
            out.append(p.y)
    return sorted(set(round(v, 3) for v in out))


def _log_from_minors(major: list[float], minor: list[float]) -> bool | None:
    """Log axis if the minor ticks between two majors are unevenly spaced (2..9 decade subdivisions)."""
    if len(major) < 2:
        return None
    a, b = sorted(major)[:2]
    m = [v for v in minor if a < v < b]
    if len(m) < 3:
        return None
    gaps = np.diff([a] + m + [b])
    return bool(gaps.max() / gaps.min() > 1.2)


def calibrate(labels, ticks, coord: str, log_hint: bool | None = None):
    """Fit value = f(pdf coordinate) from labels matched to ticks; log scale if the values are geometric."""
    pairs = []
    for lab in labels:
        c = lab[coord]
        if not ticks:
            break
        t = min(ticks, key=lambda v: abs(v - c))
        if abs(t - c) < 12:
            pairs.append((t, lab["value"]))
    pairs = sorted(set(pairs))
    if len(pairs) < 2:
        return None
    pos = np.array([p for p, _ in pairs])
    val = np.array([v for _, v in pairs])
    logscale = bool(np.all(val > 0) and len(val) >= 3 and np.allclose(np.diff(np.log10(val)),
                                                                      np.diff(np.log10(val)).mean(), rtol=1e-6, atol=0)
                    and not np.allclose(np.diff(val), np.diff(val).mean(), rtol=1e-6, atol=0))
    if log_hint is not None and np.all(val > 0):
        logscale = log_hint
    y = np.log10(val) if logscale else val
    A = np.vstack([np.ones_like(pos), pos]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid_pt = np.abs((A @ coef - y) / coef[1]).max()
    return {"scale": "log" if logscale else "linear", "a": float(coef[0]), "b": float(coef[1]),
            "max_residual_pt": float(resid_pt), "ticks": [[float(p), float(v)] for p, v in pairs]}


def to_data(cal, p):
    v = cal["a"] + cal["b"] * np.asarray(p, float)
    return 10.0 ** v if cal["scale"] == "log" else v


def path_points(items):
    pts = []
    for it in items:
        k = it[0]
        if k == "l":
            seg = [it[1], it[2]]
        elif k == "c":
            seg = [it[1], it[4]]
        elif k == "re":
            r = it[1]
            seg = [pymupdf.Point(r.x0, r.y0), pymupdf.Point(r.x1, r.y0), pymupdf.Point(r.x1, r.y1),
                   pymupdf.Point(r.x0, r.y1), pymupdf.Point(r.x0, r.y0)]
        elif k == "qu":
            q = it[1]
            seg = [q.ul, q.ur, q.lr, q.ll, q.ul]
        else:
            continue
        for s in seg:
            if not pts or (abs(pts[-1][0] - s.x) > 1e-9 or abs(pts[-1][1] - s.y) > 1e-9):
                pts.append((s.x, s.y))
    return pts


def _style(d):
    return {"color": [round(c, 4) for c in d["color"]] if d.get("color") else None,
            "width": d.get("width"), "dashes": d.get("dashes"),
            "fill": [round(c, 4) for c in d["fill"]] if d.get("fill") else None,
            "fill_opacity": d.get("fill_opacity"), "stroke_opacity": d.get("stroke_opacity")}


def _is_hatch(d) -> bool:
    """matplotlib hatch patterns: stroked paths of many short diagonal segments or quads, no fill."""
    kinds = {it[0] for it in d["items"]}
    if d.get("color") is None:
        return False
    if "qu" in kinds and d.get("fill") is None:
        return True
    if len(d["items"]) >= 20 and kinds <= {"l"}:
        dx = np.array([abs(it[2].x - it[1].x) for it in d["items"]])
        dy = np.array([abs(it[2].y - it[1].y) for it in d["items"]])
        L = np.hypot(dx, dy)
        # every segment an exact 45-degree diagonal (clipped hatch lines vary in length; data curves are never this regular)
        return bool(np.all(np.abs(dx - dy) <= 0.02 * L + 1e-6))
    return False


def _skey(s):
    return (tuple(s["color"] or ()), s["width"], s["dashes"])


def legend(page) -> dict:
    """Legend samples: short horizontal stroked lines; the label is the text on the same line to their right."""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t and not NUM.match(t):
                spans.append((t, ln["bbox"]))
    out = {}
    for d in page.get_drawings():
        it = d["items"]
        if len(it) != 1 or it[0][0] != "l" or d.get("color") is None or d.get("fill") is not None:
            continue
        p, q = it[0][1], it[0][2]
        if abs(p.y - q.y) > 0.01 or not 15 <= abs(q.x - p.x) <= 40:
            continue
        xr = max(p.x, q.x)
        cands = [(bb[0] - xr, t) for t, bb in spans if 0 < bb[0] - xr < 20 and bb[1] - 2 <= p.y <= bb[3] + 2]
        if cands:
            out.setdefault(_skey(_style(d)), min(cands)[1])
    return out


def raster_bands(page, r, cx, dpi: int = 600) -> list[dict]:
    """Shaded vertical bands (pattern-filled axvspan regions, which are not plain vector paths): rasterize the
    panel and find runs of light grey along rows near the top and bottom of the plot area, away from curves. A run
    must appear in both rows (bands span the full height). Edges are accurate to one pixel (72/dpi pt)."""
    pix = page.get_pixmap(dpi=dpi, clip=r, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(int)
    grey = (np.abs(a[..., 0] - a[..., 1]) < 6) & (np.abs(a[..., 1] - a[..., 2]) < 6) & (a[..., 0] > 180) & (a[..., 0] < 245)
    rows = [int(pix.height * f) for f in (0.03, 0.05, 0.95, 0.97)]
    runs = []
    for y in rows:
        g = grey[y]
        # light grey or hatch strokes: smooth the row so a hatched band reads as one run
        k = max(1, dpi // 36)
        g = np.convolve(g.astype(float), np.ones(2 * k + 1), "same") > 0
        edges = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        runs.append([(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] > 3 * k])
    bands = []
    for (x0, x1) in runs[0]:
        hits = sum(any(abs(a0 - x0) <= 3 * k and abs(a1 - x1) <= 3 * k for a0, a1 in rr) for rr in runs[1:])
        if hits >= 2:
            px = [r.x0 + (x0 + k) * 72 / dpi, r.x0 + (x1 - k) * 72 / dpi]  # undo the smoothing dilation
            bands.append({"x": [float(v) for v in to_data(cx, px)], "method": f"raster {dpi} dpi"})
    return bands


def extract(pdf: Path) -> dict:
    page = pymupdf.open(pdf)[0]
    labels = tick_labels(page)
    leg = legend(page)
    texts = [("".join(sp["text"] for sp in ln["spans"]), ln["bbox"]) for b in page.get_text("dict")["blocks"]
             for ln in b.get("lines", []) if "".join(sp["text"] for sp in ln["spans"]).strip()]
    drawings = page.get_drawings()
    panels, pending = [], []
    rects = axes_rects(page)
    # panels with their own x labels first, so that shared-axis panels can borrow their calibration
    rects.sort(key=lambda r: -max(r.y1, 0))
    for r in rects:
        xt, yt = major_ticks(page, r)
        xl = [l for l in labels if r.x0 - 15 <= l["cx"] <= r.x1 + 15 and r.y1 < l["cy"] < r.y1 + 40]
        yl = [l for l in labels if r.x0 - 20 <= l["bbox"][2] <= r.x0 + 1 and r.y0 - 15 <= l["cy"] <= r.y1 + 15]
        cx = calibrate(xl, xt, "cx", _log_from_minors(xt, minor_ticks(page, r, "x")))
        cy = calibrate(yl, yt, "cy", _log_from_minors(yt, minor_ticks(page, r, "y")))
        if cx is None:  # shared x axis: borrow the calibration of a panel with the same horizontal extent
            for q in panels:
                if q.get("x") and abs(q["bbox_pt"][0] - r.x0) < 0.5 and abs(q["bbox_pt"][2] - r.x1) < 0.5:
                    cx = dict(q["x"], shared_from_panel=panels.index(q))
            if cx is None:
                pending.append(r)
        xt2, _ = major_ticks(page, r, top=True)
        xl2 = [l for l in labels if r.x0 - 15 <= l["cx"] <= r.x1 + 15 and r.y0 - 40 < l["cy"] < r.y0]
        cx2 = calibrate(xl2, xt2, "cx", _log_from_minors(xt2, minor_ticks(page, r, "x", top=True)))
        pan = {"bbox_pt": [r.x0, r.y0, r.x1, r.y1], "x": cx, "y": cy, "x_top": cx2, "curves": [], "regions": [],
               "hatch_bands": []}
        if cx is None or cy is None:
            pan["error"] = "axis calibration failed (no tick labels matched)"
            panels.append(pan)
            continue
        pan["xlim"] = [float(v) for v in to_data(cx, [r.x0, r.x1])]
        pan["shaded_bands"] = raster_bands(page, r, cx)
        if cx2:
            pan["xlim_top"] = [float(v) for v in to_data(cx2, [r.x0, r.x1])]
        pan["ylim"] = [float(v) for v in to_data(cy, [r.y1, r.y0])]
        for d in drawings:
            pts = path_points(d["items"])
            if len(pts) < 2:
                continue
            P = np.array(pts)
            inside = (P[:, 0] >= r.x0 - 0.5) & (P[:, 0] <= r.x1 + 0.5) & (P[:, 1] >= r.y0 - 0.5) & (P[:, 1] <= r.y1 + 0.5)
            if not inside.any():
                continue
            st = _style(d)
            if _is_hatch(d):  # hatched bands: keep their x extent (clipped to the panel) and colour
                x0, x1 = max(P[:, 0].min(), r.x0), min(P[:, 0].max(), r.x1)
                if x1 > x0:
                    pan["hatch_bands"].append({"x": sorted(float(v) for v in to_data(cx, [x0, x1])),
                                               "color": st["color"]})
                continue
            if len(pts) == 2 and abs(P[0, 1] - P[1, 1]) < 0.01 and 15 <= abs(P[0, 0] - P[1, 0]) <= 40 and \
                    _skey(st) in leg:
                continue  # legend sample
            # skip the axes patch, spines, ticks and tiny marks
            dr = d["rect"]
            if dr.width < 1.0 and dr.height < 9 or dr.height < 1.0 and dr.width < 9:
                continue
            if st["color"] is None and st["fill"] is not None and len(d["items"]) == 1 and d["items"][0][0] == "re" \
                    and pymupdf.Rect(d["items"][0][1]).contains(r):
                continue
            if d.get("fill") is not None and dr.width < 40 and dr.height < 20 and any(
                    0 < bb[0] - dr.x1 < 20 and bb[1] - 4 <= 0.5 * (dr.y0 + dr.y1) <= bb[3] + 4 for _, bb in texts):
                continue  # legend swatch
            rec = {"style": st, "n": len(pts), "x": to_data(cx, P[:, 0]).tolist(), "y": to_data(cy, P[:, 1]).tolist(),
                   "inside_fraction": float(inside.mean())}
            if d.get("fill") is not None and d.get("type") in ("f", "fs"):
                pan["regions"].append(rec)
            elif st["color"] is not None:
                if len(pts) == 2 and (abs(P[0, 0] - P[1, 0]) < 0.01 and abs(P[0, 0] - r.x0) < 0.6 or
                                      abs(P[0, 1] - P[1, 1]) < 0.01 and (abs(P[0, 1] - r.y0) < 0.6 or abs(P[0, 1] - r.y1) < 0.6)):
                    continue  # spines
                lab = leg.get(_skey(st))
                if lab:
                    rec["label"] = lab
                pan["curves"].append(rec)
        panels.append(pan)
    panels.sort(key=lambda p: (p["bbox_pt"][1], p["bbox_pt"][0]))
    return {"file": pdf.name, "page_pt": [page.rect.width, page.rect.height], "legend": {str(k): v for k, v in leg.items()},
            "panels": panels}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=HERE / "paper")
    ap.add_argument("--out", type=Path, default=HERE / "paper_curves")
    a = ap.parse_args()
    src = fetch_source(a.src)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pdf in sorted((src / "Plots").glob("*.pdf")):
        res = extract(pdf)
        (a.out / (pdf.stem + ".json")).write_text(json.dumps(res) + "\n")
        summary[pdf.stem] = [{"x": p["x"] and (p["x"]["scale"], round(p["x"]["max_residual_pt"], 4)),
                              "y": p["y"] and (p["y"]["scale"], round(p["y"]["max_residual_pt"], 4)),
                              "xlim": p.get("xlim"), "ylim": p.get("ylim"), "xlim_top": p.get("xlim_top"),
                              "curves": len(p["curves"]), "regions": len(p["regions"]),
                              "labels": sorted({c.get("label") for c in p["curves"] if c.get("label")}),
                              "error": p.get("error")} for p in res["panels"]]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
