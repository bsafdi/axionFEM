"""Digitize Jeong, Youn, Kim, arXiv:2205.01319 Figs. 5 and 6 (C and normalized C^2 V^2 Q vs partition gap Rg/Rc).

Usage:  python3 -I capp_multicell_2205_digitize.py <path/to/2205.01319.pdf> [<out_dir>]

Both figures are embedded 2400x2250 RGB bitmaps (matplotlib output, pages 13-14).  We use the native bitmap.
Calibration: detected axis frame + tick marks (x: frame = 0 and 0.10 / 0.30; y: ticks every 0.2).
Markers: the simulated points sit on a 0.01 grid in Rg/Rc (plus 0.15..0.25 step 0.02 for the 4-cell set).
For each series and nominal x we take the pixels of the series colour in a +-30 px column window, remove
thin lines (error bars, caps, connecting lines) with a binary opening (disk r=7), and take the bounding-box
centre of the remaining blob closest to the expected location -> marker centre.  Error-bar ends: extremes
of the series-coloured pixels in a +-3 px strip around the marker x, contiguous with the marker allowing
<=20 px gaps (other series drawn on top can interrupt the bar).  Fig. 5: partial/hidden markers re-measured
from their unoccluded top edge (recover_partial).  Dashed "ideal" curves: Fig. 5 column-wise
tracking of thin runs, sampled every 4 px; Fig. 6 continuity-constrained edge tracking (track_dashed: the 2-cell trace
would otherwise jump onto the solid Monte Carlo line).  Overlay: black circles = marker
centres, cyan squares = re-measured partial/hidden markers, magenta = error-bar ends and dashed-curve samples,
cyan vertical ticks = truncation points of the Fig. 6 dashed traces.
Outputs <out>/capp_multicell_2205_fig5.png, _fig6.png, *_overlay.png and digitize/capp_multicell_2205_digitized.json.
"""
import sys, os, json
import numpy as np                      # numpy before fitz (libstdc++ load-order clash)
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import fitz

PDF = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

X0PX, X1PX = 397.0, 2235.0
TOP = dict(y0px=991.0, px_per_unit=(991.0 - 233.0) / 0.6)      # C panel: 0 at 991, 0.6 at 233
BOT = dict(y0px=1951.0, px_per_unit=(1951.0 - 1084.0) / 1.0)    # C^2V^2Q panel: 0 at 1951, 1.0 at 1084
TAB = {'blue': (31, 119, 180), 'orange': (255, 127, 14), 'green': (44, 160, 44), 'red': (214, 39, 40),
       'pure_red': (255, 0, 0), 'pure_green': (0, 128, 0), 'pure_blue': (0, 0, 255), 'black': (0, 0, 0)}

FIGS = {
    'fig5': dict(page=12, xmax=0.10,
                 series={'tol_0.01pct': ('blue', np.arange(1, 11) / 100), 'tol_0.03pct': ('orange', np.arange(1, 11) / 100),
                         'tol_0.05pct': ('green', np.arange(1, 11) / 100), 'tol_0.10pct': ('red', np.arange(1, 11) / 100)},
                 dashed={'ideal_2cell': 'black'}),
    'fig6': dict(page=13, xmax=0.30,
                 series={'2cell_tol_0.05pct': ('pure_red', np.arange(1, 11) / 100),
                         '4cell_tol_0.05pct': ('pure_green', np.array([0.15, 0.17, 0.19, 0.21, 0.23, 0.25])),
                         '6cell_tol_0.05pct': ('pure_blue', np.array([0.24, 0.25, 0.26, 0.27, 0.28]))},
                 dashed={'ideal_2cell': 'pure_red', 'ideal_4cell': 'pure_green', 'ideal_6cell': 'pure_blue'}),
}
# legend boxes to ignore (native px)
LEGEND = {'fig5': [(1790, 430, 2215, 965)], 'fig6': [(1270, 1600, 2215, 1930)]}


def cmask(a, rgb, tol=70):
    return np.sqrt(((a - np.array(rgb)) ** 2).sum(2)) < tol


# Fig. 6 dashed curves: the three dashed "ideal" curves overlap each
# other and the solid Monte Carlo lines, so a dash run is accepted only through an edge that is known to be
# unoccluded for that curve (z-order: red < green < blue, solid MC lines and markers above all dashes), and only
# within TOL px of a linear prediction from the last accepted points (continuity constraint).  Dash thickness is
# 6 px, so centre = top edge + 2.5 = bottom edge - 2.5.  Marker blobs (all series) are masked out (dilated 4 px).
# xmax: hard truncation where the dash merges with the solid MC line (beyond it the curve is not separable).
TRACK2 = {'fig6': {
    ('C', 'ideal_2cell'):          dict(y0=0.6455, modes=('bottom',), xmax=0.060),   # red sliver under green dash
    ('C', 'ideal_4cell'):          dict(y0=0.648, modes=('centre', 'top'), xmax=None),
    ('C', 'ideal_6cell'):          dict(y0=0.628, modes=('centre', 'top'), xmax=None),
    ('C2V2Q_norm', 'ideal_2cell'): dict(y0=1.0, modes=('centre', 'top'), xmax=0.090),  # merges with MC line beyond
    ('C2V2Q_norm', 'ideal_4cell'): dict(y0=1.0, modes=('centre', 'bottom'), xmax=None),  # under blue for x < 0.1
    ('C2V2Q_norm', 'ideal_6cell'): dict(y0=1.0, modes=('centre', 'top'), xmax=0.270, steep=True),  # coincides with MC line beyond
}}
TOL, MSKIP, MAXGAP, NFIT = 2.5, 34, 140, 10


def track_dashed(m, cfg, mk, val, cal):
    """Continuity-constrained tracking; returns [(x_px, y_px, mode), ...]."""
    y0px = cal['y0px'] - cfg['y0'] * cal['px_per_unit']
    pts = []; lastx = None
    xend = X1PX - 2 if cfg.get('xmax') is None else X0PX + cfg['xmax'] * (X1PX - X0PX) / 0.30
    for xp in range(int(X0PX) + 6, int(xend), 4):
        if any(abs(xp - q) < MSKIP for q in mk): continue
        if lastx is not None and xp - lastx > MAXGAP: break
        if len(pts) < 3:
            pred = pts[-1][1] if pts else y0px
        else:
            xs = np.array([p[0] for p in pts[-NFIT:]], float); ys = np.array([p[1] for p in pts[-NFIT:]], float)
            if cfg.get('steep'):                                 # local fit (last <= 60 px) for the steep 6-cell drop
                xs, ys = xs[xs > xp - 60], ys[xs > xp - 60]
            pred = (np.polyval(np.polyfit(xs, ys, 1), xp) if len(xs) > 1 and np.ptp(xs) > 0
                    else (ys[-1] if len(ys) else pts[-1][1]))
        col_ = np.nonzero(m[:, xp])[0]
        if len(col_) == 0: continue
        runs = np.split(col_, np.nonzero(np.diff(col_) > 1)[0] + 1)
        cands = []
        for r in runs:
            th = r[-1] - r[0] + 1
            if 'centre' in cfg['modes'] and 5 <= th <= 7: cands.append((r.mean(), 'centre'))
            if 'top' in cfg['modes']: cands.append((r[0] + 2.5, 'top'))
            if 'bottom' in cfg['modes']: cands.append((r[-1] - 2.5, 'bottom'))
        if not cands: continue
        y, mode = min(cands, key=lambda c: abs(c[0] - pred))
        slope = 0.0 if len(pts) < 3 else abs(pts[-1][1] - pts[-3][1]) / max(pts[-1][0] - pts[-3][0], 1)
        tol = 6 if len(pts) < 3 else TOL + (2.0 * slope * 4 if cfg.get('steep') else 0.0)   # steep: allow curvature lag
        if abs(y - pred) > tol: continue
        pts.append((xp, float(y), mode)); lastx = xp
    return pts


# Partial / hidden markers.  Series are drawn in tolerance order and every later
# series lies below the earlier ones, so the TOP edge of each marker is the unoccluded one.  A marker whose
# opened blob is clipped (bbox height > 5 px below the series median) or missing is re-measured from its top-edge profile: template = top
# profile of a full-size marker of the same series (relative to its bbox centre, |dx| = 5..12 px, the bar
# column |dx| <= 4 excluded); cy = median(y_top(dx) - template(dx)); accepted if >= 8 columns and MAD <= 1.5 px.
# Recovered/corrected points carry a note and a per-point uncertainty of 5 px.
DXS = [d for d in range(-12, 13) if abs(d) > 4]


def top_profile(m, cx, y_from, y_to):
    prof = {}
    for d in DXS:
        c = np.nonzero(m[int(y_from):int(y_to), int(round(cx)) + d])[0]
        if len(c): prof[d] = int(c[0]) + int(y_from)
    return prof


def recover_partial(pts, m, xs, tox, val, dr, ylo):
    hs = [q['_hw'][0] for q in pts if q.get('y') is not None]
    if not hs:
        for q in pts: q.pop('_hw', None); q.pop('_bar', None)
        return
    hmed = float(np.median(hs))
    full = [q for q in pts if q.get('y') is not None and q['_hw'][0] >= hmed - 1 and q['_hw'][1] <= hmed + 3]
    # template: the full-size marker with the most complete top profile in a tight window (excludes caps above)
    best = None
    for q in full:
        rcx, rcy = q['px']
        t = {d: y - rcy for d, y in top_profile(m, rcx, rcy - hmed / 2 - 3, rcy + 5).items()}
        if best is None or len(t) > len(best): best = t
    tmpl = best or {}
    known = [(q['x'], q['px'][1]) for q in pts if q.get('y') is not None]
    for q in pts:
        hw = q.pop('_hw', None); bar = q.pop('_bar', None)
        partial = q.get('y') is not None and hw[0] < hmed - 5
        if not (partial or q.get('y') is None): continue
        if partial:
            cx, guess = q['px']
        else:
            cx = tox(q['x'])
            # guess: the series' own solid line through the marker position (median y of the line 22-30 px either side)
            ys_line = []
            for d in (-30, -26, -22, 22, 26, 30):
                c = np.nonzero(m[ylo:, int(cx) + d])[0]
                if len(c):
                    runs = np.split(c, np.nonzero(np.diff(c) > 1)[0] + 1)
                    ys_line.append(min(runs, key=lambda r: abs(r.mean() + ylo - (known[-1][1] if known else 0)) if False else r[0]).mean() + ylo)
            if not ys_line: continue
            guess = float(np.median(ys_line))
        prof = top_profile(m, cx, guess - 32, guess + 4)
        diffs = [prof[d] - tmpl[d] for d in prof if d in tmpl]
        ok = len(diffs) >= 8
        if ok:
            cy = float(np.median(diffs)); mad = float(np.median(np.abs(np.array(diffs) - cy)))
            ok = mad <= 1.5 and abs(cy - guess) <= 14
        if not ok:
            if partial:     # keep the bbox value but flag it and inflate its uncertainty (half the marker size)
                q['uncertainty_y_px'] = round(hmed / 2)
                q['note'] = 'partial marker (clipped blob %dx%d px); bbox centre, top-edge fit failed' % hw
            continue
        old = q.get('y')
        q['y'] = round(val(cy), 4); q['px'] = [round(cx, 1), round(cy, 1)]
        q['x_from_px'] = round(q['x'], 4) if old is None else q.get('x_from_px')
        q['uncertainty_y_px'] = 5
        q['note'] = (('partial marker (clipped blob %dx%d px; bbox centre gave %.4f)' % (hw[0], hw[1], old)) if partial
                     else 'hidden marker recovered') + '; centre from the unoccluded top-edge profile (template fit, %d columns, MAD %.1f px)' % (len(diffs), mad)
        if partial and bar is not None:
            q['err_hi'] = round(val(bar[0]) - val(cy), 4); q['err_lo'] = round(val(cy) - val(bar[1]), 4)
        dr.rectangle([cx - 11, cy - 11, cx + 11, cy + 11], outline=(0, 200, 200), width=3)


def main():
    doc = fitz.open(PDF)
    res = {}
    disk = (lambda r: (np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r))(7)
    for fig, spec in FIGS.items():
        pg = doc[spec['page']]; xref = pg.get_images(full=True)[0][0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3: pix = fitz.Pixmap(fitz.csRGB, pix)
        png = os.path.join(OUT, f'capp_multicell_2205_{fig}.png'); pix.save(png)
        img = Image.open(png).convert('RGB'); a = np.asarray(img).astype(float)
        for (x0, y0, x1, y1) in LEGEND[fig]: a[y0:y1, x0:x1] = 255
        ov = img.copy(); dr = ImageDraw.Draw(ov)
        sx = (X1PX - X0PX) / spec['xmax']
        tox = lambda v: X0PX + v * sx
        out = {'calibration': {'x': [[X0PX, 0.0], [X1PX, spec['xmax']]],
                               'y_C': [[TOP['y0px'], 0.0], [TOP['y0px'] - 0.6 * TOP['px_per_unit'], 0.6]],
                               'y_C2V2Q': [[BOT['y0px'], 0.0], [BOT['y0px'] - BOT['px_per_unit'], 1.0]]},
               'series': {}, 'ideal_curves': {}}
        for panel, cal, (ylo, yhi) in (('C', TOP, (81, 991)), ('C2V2Q_norm', BOT, (1041, 1951))):
            val = lambda py: (cal['y0px'] - py) / cal['px_per_unit']
            for name, (col, xs) in spec['series'].items():
                m = cmask(a, TAB[col]); m[:ylo + 2] = False; m[yhi - 2:] = False
                op = ndi.binary_opening(m, structure=disk)
                lab, n = ndi.label(op)
                pts = []
                for xv in xs:
                    xp = tox(xv)
                    cands = []
                    for i in range(1, n + 1):
                        ys_, xs_ = np.nonzero(lab[:, int(xp) - 40:int(xp) + 41] == i)
                        if len(ys_) < 60: continue
                        sl = ndi.find_objects((lab == i).astype(int))[0]
                        cx = 0.5 * (sl[1].start + sl[1].stop - 1); cy = 0.5 * (sl[0].start + sl[0].stop - 1)
                        if abs(cx - xp) < 14: cands.append((abs(cx - xp), cx, cy, sl))
                    if not cands:
                        pts.append(dict(x=float(xv), y=None, note='marker occluded by series drawn on top; value within one marker radius of the visible marker')); continue
                    _, cx, cy, sl = min(cands)
                    # error bar: strip +-3 px, contiguous through gaps <= 20 px
                    strip = m[:, int(round(cx)) - 3:int(round(cx)) + 4].any(1)
                    up = int(cy); gap = 0; y = int(cy)
                    while y > ylo and gap <= 20:
                        y -= 1; gap = 0 if strip[y] else gap + 1
                        if strip[y]: up = y
                    lo = int(cy); gap = 0; y = int(cy)
                    while y < yhi and gap <= 20:
                        y += 1; gap = 0 if strip[y] else gap + 1
                        if strip[y]: lo = y
                    half = (sl[0].stop - sl[0].start) / 2
                    has_bar = (cy - up) > half + 12 or (lo - cy) > half + 12   # else bar hidden inside marker (< ~0.015)
                    pts.append(dict(x=float(xv), y=round(val(cy), 4), px=[round(cx, 1), round(cy, 1)],
                                    x_from_px=round((cx - X0PX) / sx, 4),
                                    err_hi=round(val(up) - val(cy), 4) if has_bar else None,
                                    err_lo=round(val(cy) - val(lo), 4) if has_bar else None,
                                    _hw=(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), _bar=(up, lo) if has_bar else None))
                    dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(0, 0, 0), width=3)
                    if has_bar:
                        dr.line([cx + 14, up, cx + 30, up], fill=(255, 0, 255), width=3)
                        dr.line([cx + 14, lo, cx + 30, lo], fill=(255, 0, 255), width=3)
                if fig == 'fig5':      # Fig. 6: dashed curves are drawn above the markers -> top edges not clean
                    recover_partial(pts, m, xs, tox, val, dr, ylo)
                else:
                    for q in pts: q.pop('_hw', None); q.pop('_bar', None)
                out['series'].setdefault(name, {})[panel] = pts
            # dashed ideal curves
            if fig in TRACK2:
                for name, col in spec['dashed'].items():
                    cfg = TRACK2[fig][(panel, name)]
                    m = cmask(a, TAB[col], 80); m[:ylo + 2] = False; m[yhi - 2:] = False
                    m[:, :int(X0PX) + 4] = False
                    # remove marker blobs of every series (opening with the r=7 disk keeps only markers), dilated 4 px
                    blobs = np.zeros_like(m)
                    for (c2, _) in spec['series'].values():
                        blobs |= ndi.binary_opening(cmask(a, TAB[c2]), structure=disk)
                    m &= ~ndi.binary_dilation(blobs, iterations=4)
                    px_pts = track_dashed(m, cfg, [], val, cal)
                    out['ideal_curves'].setdefault(name, {})[panel] = [[round((xp - X0PX) / sx, 5), round(val(y), 4), mode]
                                                                       for xp, y, mode in px_pts]
                    for xp, y, _ in px_pts:
                        dr.ellipse([xp - 2, y - 2, xp + 2, y + 2], fill=(255, 0, 255))
                    if cfg.get('xmax') is not None:
                        xe = tox(cfg['xmax']); dr.line([xe, y - 25, xe, y + 25], fill=(0, 200, 200), width=3)
                continue
            # dashed ideal curves (Fig. 5): thin runs (<= 9 px tall) of the dashed colour, tracked from the left
            for name, col in spec['dashed'].items():
                m = cmask(a, TAB[col], 80 if col != 'black' else 120); m[:ylo + 2] = False; m[yhi - 2:] = False
                m[:, :int(X0PX) + 4] = False
                # remove marker/errorbar blobs of this colour (they are wider than the dash line)
                prev = None; curve = []
                mk = [tox(v) for (c2, xs2) in spec['series'].values() if c2 == col for v in xs2]
                for xp in range(int(X0PX) + 6, int(X1PX) - 2, 4):
                    if any(abs(xp - q) < 22 for q in mk): continue      # skip columns crossing markers
                    col_ = np.nonzero(m[:, xp])[0]
                    if len(col_) == 0: continue
                    runs = np.split(col_, np.nonzero(np.diff(col_) > 1)[0] + 1)
                    runs = [r for r in runs if len(r) <= 9]
                    if not runs: continue
                    ys = [r.mean() for r in runs]
                    if prev is None:
                        j = int(np.argmin([abs(val(y) - (0.65 if panel == 'C' else 1.0)) for y in ys]))
                    else:
                        j = int(np.argmin([abs(y - prev) for y in ys]))
                        if abs(ys[j] - prev) > 25: continue
                    prev = ys[j]; curve.append([round((xp - X0PX) / sx, 5), round(val(ys[j]), 4)])
                    dr.point([(xp, ys[j])], fill=(255, 0, 255)); dr.ellipse([xp - 2, ys[j] - 2, xp + 2, ys[j] + 2], fill=(255, 0, 255))
                out['ideal_curves'].setdefault(name, {})[panel] = curve
        for xp in (X0PX, X1PX):
            dr.line([xp, 1951, xp, 1990], fill=(0, 200, 200), width=5)
        for yp in (233, 991, 1084, 1951):
            dr.line([350, yp, 397, yp], fill=(0, 200, 200), width=5)
        ov.save(os.path.join(OUT, f'capp_multicell_2205_{fig}_overlay.png'))
        res[fig] = out
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capp_multicell_2205_digitized.json'), 'w'), indent=1)
    for fig in res:
        for s, d in res[fig]['series'].items():
            for p, pts in d.items():
                print(fig, s, p, [(q['x'], q['y'], q.get('err_hi'), q.get('err_lo'), q.get('x_from_px')) for q in pts])
        for s, d in res[fig]['ideal_curves'].items():
            for p, c in d.items():
                print(fig, s, p, 'n=%d' % len(c), c[::25])


if __name__ == '__main__':
    main()
