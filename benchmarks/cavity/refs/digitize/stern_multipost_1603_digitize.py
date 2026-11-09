"""Digitize Stern et al. 2015 (arXiv:1603.06990) Figs. 6 and 7: Q0, C, figure of merit vs Lambda = f*R.

Usage:  python3 -I stern_multipost_1603_digitize.py <path/to/1603.06990.pdf> [<out_dir>]

The figures are embedded RGB bitmaps (1270x1582 px, pages 7 and 8).  We extract the native bitmap
(no resampling), calibrate each of the 12 panels from the detected axis frame + tick marks, detect the
simulation markers (hollow circle/diamond/triangle/square/asterisk) with an angular ring-coverage filter
on the non-white mask, assign each marker to a series by the dominant hue around it, then apply a short
list of manual corrections (documented below; each was checked on the overlay).  Writes
  digitize/stern_multipost_1603_digitized.json   (points, calibration)
  <out>/stern_multipost_1603_fig6.png, _fig7.png (native bitmaps)
  <out>/stern_multipost_1603_fig6_overlay.png, _fig7_overlay.png
Only the markers (simulation points) are digitized; the dashed curves are the authors' spline guides.
"""
import sys, os, json
import numpy as np                      # import numpy before fitz (libstdc++ load-order clash)
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import fitz

PDF = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

CLASSES = {'blue': (185, 235), 'green': (70, 145), 'red': (330, 380), 'purple': (250, 310), 'brown': (15, 55)}
DRAW = {'blue': (0, 0, 255), 'green': (0, 160, 0), 'red': (255, 0, 0), 'purple': (150, 0, 200), 'brown': (120, 60, 0)}

# figure -> (page index, {column: {colour: design}}, exclusion boxes per (column,row) in native px)
FIGS = {
    'fig6': dict(page=6, series={0: {'blue': 'A', 'green': 'B', 'red': 'C'},
                                  1: {'blue': 'D', 'green': 'E', 'red': 'F', 'purple': 'G', 'brown': 'H'}},
                 exclude={(0, 0): [(122, 18, 335, 108), (400, 25, 590, 212)], (0, 1): [(122, 530, 335, 625)],
                          (0, 2): [(122, 1040, 335, 1128)],
                          (1, 0): [(782, 18, 990, 162), (1058, 25, 1250, 212)], (1, 1): [(782, 530, 990, 678)],
                          (1, 2): [(782, 1040, 990, 1190)]}),
    'fig7': dict(page=7, series={0: {'blue': 'I', 'green': 'J', 'red': 'K'},
                                  1: {'blue': 'L', 'green': 'M', 'red': 'N'}},
                 exclude={(0, 0): [(122, 18, 335, 108), (400, 22, 590, 212)], (0, 1): [(122, 528, 335, 622)],
                          (0, 2): [(122, 1035, 335, 1125)],
                          (1, 0): [(782, 108, 1000, 202), (1058, 165, 1250, 350)], (1, 1): [(782, 528, 1000, 618)],
                          (1, 2): [(782, 1035, 1000, 1125)]}),
}
YVAL = [(20000.0, 10000.0), (1.0, 0.0), (2.5, 1.0)]       # (top frame, bottom frame) values per row
YNAME = ['Q0', 'C', 'FoM']
XT = (0.25, 0.30)                                           # labelled x ticks used for calibration

# Manual corrections after inspecting overlays: (fig, col, row) -> list of ops
#  ('add', colour, x_px, y_px)  marker missed by the detector (centre read off a 8x zoom)
#  ('del', x_px, y_px)          spurious detection (line crossing / dash-dot clutter) nearest to (x,y)
#  ('set', x_px, y_px, colour)  re-assign colour of detection nearest to (x,y)
MANUAL = {}
MANUAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stern_multipost_1603_manual.json')
if os.path.exists(MANUAL_PATH):
    MANUAL = {(k.split(',')[0], int(k.split(',')[1]), int(k.split(',')[2])): v
              for k, v in json.load(open(MANUAL_PATH)).items() if not k.startswith('_')}


def hue_sat(a):
    mx = a.max(2); mn = a.min(2); s = (mx - mn) / np.maximum(mx, 1)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]; d = np.maximum(mx - mn, 1e-9)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, s


def ring_coverage(mask, r0=4.0, r1=8.5, nsec=24):
    R = int(np.ceil(r1)); yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(xx, yy); th = np.arctan2(yy, xx)
    band = (rr >= r0) & (rr <= r1); cov = np.zeros(mask.shape)
    for k in range(nsec):
        a0 = -np.pi + 2 * np.pi * k / nsec
        ker = (band & (th >= a0) & (th < a0 + 2 * np.pi / nsec)).astype(float)
        cov += ndi.correlate(mask.astype(float), ker, mode='constant') > 0
    return cov


def groups(idx):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 2: out[-1].append(i)
        else: out.append([i])
    return [float(np.mean(g)) for g in out]


def calibrate(dark, col):
    W = dark.shape[1]; x0, x1 = (110, 615) if col == 0 else (770, W)
    rows = groups(np.where(dark[:, x0 + 25:x1 - 25].sum(1) > 0.6 * (x1 - x0 - 50))[0])
    cols = groups(np.where(dark[30:470, x0:x1].sum(0) > 380)[0] + x0)
    L, R = cols[0], cols[-1]
    assert len(rows) == 6, rows
    b = int(round(rows[5]))
    t = groups(np.where(dark[b - 14:b - 2, int(L) + 4:int(R) - 4].sum(0) >= 8)[0] + int(L) + 4)
    assert len(t) == 2, t
    return dict(L=L, R=R, rows=rows, xt=t)


def classify(h, s, nonwhite, cx, cy):
    """Dominant series hue among saturated pixels within 9 px of (cx, cy)."""
    hh = np.where(h < 20, h + 360, h)
    Y, X = int(round(cy)), int(round(cx)); sub = (slice(Y - 9, Y + 10), slice(X - 9, X + 10))
    sel = (s[sub] > 0.15) & nonwhite[sub]
    cnt = {}
    for k, (l, u) in CLASSES.items():
        hv = hh[sub] if k == 'red' else h[sub]
        cnt[k] = int((sel & (hv >= l) & (hv < u)).sum())
    return max(cnt, key=cnt.get)


def detect(a, h, s, nonwhite, box, excl, thr=19):
    """Generic marker detection: fraction of 24 angular sectors of the annulus 4 <= r <= 8.5 px that
    contain non-white pixels; local maxima (9x9, after 3x3 smoothing) with >= thr sectors.  A dashed
    line through a point covers ~4-8 sectors, a closed marker outline >= 19."""
    x0, y0, x1, y1 = box
    inner = np.zeros(nonwhite.shape, bool); inner[y0:y1, x0:x1] = True
    for (a0, b0, a1, b1) in excl: inner[b0:b1, a0:a1] = False
    cov = ring_coverage(inner & nonwhite)
    sm = ndi.uniform_filter(cov, 3)
    pk = (sm == ndi.maximum_filter(sm, size=9)) & (cov >= thr) & inner
    lab, n = ndi.label(pk)
    out = []
    for c in ndi.center_of_mass(pk, lab, range(1, n + 1)):
        if all(np.hypot(c[1] - o[0], c[0] - o[1]) > 5 for o in out):
            out.append([c[1], c[0], classify(h, s, nonwhite, c[1], c[0])])
    return out, cov


def refine(cov, x, y, r=4):
    """Move a hand-placed seed to the coverage-weighted centre of the best ring response within r px."""
    Y, X = int(round(y)), int(round(x))
    win = cov[Y - r:Y + r + 1, X - r:X + r + 1]
    m = win >= win.max() - 1
    ys, xs = np.nonzero(m)
    return X - r + xs.mean(), Y - r + ys.mean()


XCHK = []


def main():
    doc = fitz.open(PDF)
    result = {}
    for fig, spec in FIGS.items():
        pg = doc[spec['page']]
        xref = pg.get_images(full=True)[0][0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3: pix = fitz.Pixmap(fitz.csRGB, pix)
        png = os.path.join(OUT, f'stern_multipost_1603_{fig}.png'); pix.save(png)
        img = Image.open(png).convert('RGB'); a = np.asarray(img).astype(float)
        h, s = hue_sat(a); nonwhite = a.sum(2) < 600; dark = a.sum(2) < 400
        ov = img.copy(); dr = ImageDraw.Draw(ov)
        result[fig] = {'native_px': [img.width, img.height], 'panels': {}}
        for col in (0, 1):
            cal = calibrate(dark, col)
            (x25, x30) = cal['xt']; ax = (XT[1] - XT[0]) / (x30 - x25)
            fx = lambda px: XT[0] + (px - x25) * ax
            for row in range(3):
                top, bot = cal['rows'][2 * row], cal['rows'][2 * row + 1]
                vt, vb = YVAL[row]
                fy = lambda py: vb + (py - bot) * (vt - vb) / (top - bot)
                box = (int(cal['L']) + 4, int(top) + 4, int(cal['R']) - 3, int(bot) - 3)
                auto, _ = detect(a, h, s, nonwhite, box, spec['exclude'][(col, row)])
                seeds = MANUAL[(fig, col, row)]
                ser = {}
                inv = {v: k for k, v in spec['series'][col].items()}
                for d in sorted(seeds):
                    c = inv[d]
                    for pt in seeds[d]:
                        px, py = pt[0], pt[1]; occ = len(pt) > 2
                        # cross-check against the automatic ring detector (independent estimate)
                        dd = [np.hypot(q[0] - px, q[1] - py) for q in auto]
                        j = int(np.argmin(dd)) if dd else -1
                        chk = [round(auto[j][0] - px, 1), round(auto[j][1] - py, 1)] if j >= 0 and dd[j] < 4 else None
                        if chk: XCHK.append(chk)
                        ser.setdefault(d, []).append(dict(x=round(fx(px), 5), y=round(fy(py), 1 if row == 0 else 4),
                                                         px=[px, py], occluded=occ, auto_minus_manual_px=chk))
                        r = 6; dr.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 0) if not occ else (255, 0, 255), width=1)
                        dr.line([px - 9, py, px + 9, py], fill=DRAW[c], width=1); dr.line([px, py - 9, px, py + 9], fill=DRAW[c], width=1)
                        dr.text((px + 7, py - 14), d, fill=DRAW[c])
                # calibration marks
                for xp in (x25, x30): dr.line([xp, bot - 25, xp, bot], fill=(255, 0, 255), width=2)
                for yp in (top, bot): dr.line([cal['L'], yp, cal['L'] + 25, yp], fill=(255, 0, 255), width=2)
                result[fig]['panels'][f'{col},{row}'] = dict(
                    quantity=YNAME[row], design_points=ser,
                    calibration=dict(x=[[x25, XT[0]], [x30, XT[1]]], y=[[top, vt], [bot, vb]],
                                     px_per_unit_x=1 / ax, px_per_unit_y=(bot - top) / (vt - vb)))
        ov.save(os.path.join(OUT, f'stern_multipost_1603_{fig}_overlay.png'))
    json.dump(result, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stern_multipost_1603_digitized.json'), 'w'), indent=1)
    for fig in result:
        for k, p in result[fig]['panels'].items():
            print(fig, k, p['quantity'], {d: len(v) for d, v in p['design_points'].items()})
    X = np.array(XCHK)
    print('auto-vs-manual cross-check: n=%d  rms dx=%.2f dy=%.2f px  max |d|=%.2f px' % (
        len(X), np.sqrt((X[:, 0] ** 2).mean()), np.sqrt((X[:, 1] ** 2).mean()), np.abs(X).max()))


if __name__ == '__main__':
    main()
