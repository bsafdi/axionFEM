"""Regenerate docs/images/cavity_modes.png, the cavity eigenmode gallery of docs/validation/cavity_2d.md.

    python docs/images/make_cavity_figure.py [workdir]

Solves a copper pillbox (a = 0.1 m, L = 0.2 m) and a coaxial cavity (centred rod a = 0.05 m spanning both end
caps, wall b = 0.2 m, L = 0.4 m), axisymmetric, a few seconds each, and renders |E| of four eigenmodes on the
meridional section, mirrored about the axis, with field lines (TM: E lines; TE: H lines): pillbox TM010, TM020,
TE011 and the coaxial TM010-like mode. The field panels come from the C++ renderer (Results.plot_mode); the
renderer's small title band is cropped and replaced by a header with f, the form factor C (uniform axial B0) and
Q0 (normal-skin copper walls, rho = 1.7e-8 ohm m, i.e. room temperature), all from the solve.
Needs the built package, Pillow and matplotlib (for its DejaVu fonts).
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from axionfem import cavity as cv

out = Path(__file__).resolve().parent / "cavity_modes.png"
work = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/readme_cavity")
cu = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 300.0}}
A, L = 0.1, 0.2

pill = cv.pillbox_design(A, L, h=0.004, modes=8, wall_material=cu).solve(str(work / "pillbox"), verbose=0)
coax = cv.coax_design(0.05, 0.2, 2 * L, h=0.004, modes=4, wall_material=cu).solve(str(work / "coax"), verbose=0)


def pick(res, fam, f_ref):
    """The solved mode of family `fam` nearest the analytic frequency f_ref (checked to 1e-4)."""
    row = min((r for r in res.cavity_modes() if r["family"] == fam), key=lambda r: abs(r["f_Hz"] - f_ref))
    assert abs(row["f_Hz"] / f_ref - 1) < 1e-4, (fam, row["f_Hz"], f_ref)
    return row


ref = {(m["family"], m["n"], m["p"]): m["f_Hz"] for m in cv.pillbox_modes(A, L, 4e9)}
panels = [(pill, pick(pill, "TM", ref[("TM", 1, 0)]), "Pillbox TM010", "E lines"),
          (pill, pick(pill, "TM", ref[("TM", 2, 0)]), "Pillbox TM020", "E lines"),
          (pill, pick(pill, "TE", ref[("TE", 1, 1)]), "Pillbox TE011", "H lines"),
          (coax, pick(coax, "TM", cv.coax_tm_mode(0.05, 0.2)["f_Hz"]), "Coaxial (centred rod) TM010-like", "E lines")]

font = lambda bold, size: ImageFont.truetype(  # noqa: E731
    font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)
F_HEAD, F_SUB = font(True, 44), font(False, 36)
W, H = 1000, 1000


def crop_title(im):
    """Drop the renderer's title and subtitle: everything above the last text row before the plot frame
    (the first row that is mostly frame colour), searched in the left 60 % (the colour-bar label is on the right)."""
    a = np.asarray(im.convert("L"), dtype=float)
    dark = a < 90
    frame = int(np.argmax(dark.sum(axis=1) > 0.3 * a.shape[1]))
    left = (a[:frame, : int(0.6 * a.shape[1])] < 200).any(axis=1)
    last_text = int(np.nonzero(left)[0].max()) if left.any() else 0
    return im.crop((0, last_text + 6, im.width, im.height))


tiles = []
for k, (res, row, name, lines) in enumerate(panels):
    p = res.plot_mode(row, work / f"panel{k}.png", quantity="Emag", label=name, annotate=False, view_scale=1.08,
                      contours=8 if row["family"] == "TM" else 12, title=" ", width=W, height=H)
    body = crop_title(Image.open(p).convert("RGB"))
    mant, ex = f"{row['Q0']:.2e}".split("e")
    q0 = mant + " \u00d7 10" + "".join("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"[int(c)]
                                       for c in str(int(ex)))
    sub = f"f = {row['f_Hz'] / 1e9:.4f} GHz    C = {abs(row['C']):.3f}    Q\u2080 = {q0}"
    tiles.append((name, sub, body))
    print(name, "|", sub, "|", lines)

head = 128
th = head + max(b.height for _, _, b in tiles)
fig = Image.new("RGB", (2 * W, 2 * th + 110), "white")
d = ImageDraw.Draw(fig)
for k, (name, sub, body) in enumerate(tiles):
    x0, y0 = (k % 2) * W, (k // 2) * th
    d.text((x0 + 28, y0 + 14), name, font=F_HEAD, fill=(20, 20, 24))
    d.text((x0 + 28, y0 + 70), sub, font=F_SUB, fill=(60, 60, 66))
    fig.paste(body, (x0, y0 + head + (th - head - body.height) // 2))
note = ("|E| of each eigenmode on the meridional section (mirrored about the axis), normalized to "
        "\u222b\u03b5\u1d63|E|\u00b2dV = 1;\nwhite: E field lines (TM) or H field lines (TE). "
        "C for a uniform axial B\u2080; Q\u2080 for room-temperature copper walls (normal skin effect).")
d.multiline_text((28, 2 * th + 14), note, font=font(False, 30), fill=(60, 60, 66), spacing=12)
fig = fig.resize((fig.width * 7 // 10, fig.height * 7 // 10), Image.LANCZOS)
fig.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
print(out, fig.size, out.stat().st_size // 1024, "KB")
