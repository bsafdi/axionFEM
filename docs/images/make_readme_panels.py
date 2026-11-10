"""Regenerate the README panels of axionFEM (docs/images/readme_*.png), with all text typeset by LaTeX.

    python docs/images/make_readme_panels.py [--out docs/images] [--work runs/readme_panels]
           [--only cavity,srf,readout,chain]

Panels (``--only`` names, output files):

* ``cavity``   ``readme_cavity_offcentre.png``: |E_z| of the TM010-like mode on the transverse cross-section of a
  copper cavity with an off-centre copper tuning rod (R = 210 mm, one 50.8 mm rod; the geometry of the homepage
  cavity card, ``CAV`` and ``_cavity_design`` of make_homepage_figures.py), solved here with the cross-section
  eigensolver (``"symmetry": "xsec"``). The title quotes the mode's frequency f and form factor C from this solve.
* ``srf``      ``readme_srf_cylinder.png``: overlap density E1.B0 of the degenerate TE012 pump and TM013 signal of the
  60 L closed cylinder of R. Lasenby, Phys. Rev. D 102, 015008 (2020), arXiv:1912.11056, on the meridional section
  at the degenerate length, from the axisymmetric eigensolver (the SRF stage of make_homepage_figures.py:
  ``SRF``, ``_srf_solve`` and ``_srf_overlap`` at its finer mesh). The title quotes f and the overlap factor eta_10
  from this solve.
* ``readout``  ``readme_readout_circuit.png``: the series readout loop of ``axionfem.circuit.series_readout`` with
  every optional series element, drawn by ``Readout.draw(annotate=False)`` and checked against its circuit
  (``axionfem.draw.check_drawing``).
* ``chain``    ``readme_squid_chain.png``: the dc SQUID chain ``squid.chain("C1+low+RT").draw(annotate=False)``.

The two schematics are the package's own drawings. Only labels are moved or retyped for the LaTeX rendering
(``readout_drawing``, ``chain_drawing``): the readout's SQUID noise notes go inside the empty readout loop, L_sq
moves off the junction crosses, M and M_IN2 are set as math, the resistors read R(T) on one line, the module name is
set in typewriter, and the chain's long stage notes become short stage labels.

Style: Computer Modern text through matplotlib's ``text.usetex``, inward ticks on all four sides with minor ticks,
inferno for field magnitudes and a diverging map symmetric about zero for signed fields; white background. The
field panels are 1200 px wide; the schematics are rendered at 300 dpi, trimmed, and resampled to 1200 px wide.

Needs matplotlib with a working LaTeX toolchain for ``text.usetex`` (``latex`` and ``dvipng`` on the PATH, with the
type1cm, type1ec and cm-super packages), NumPy, SciPy and Pillow; the cavity and SRF panels need a build with Python
bindings and the mesher (``axf-mesher``, or ``AXF_MESHER`` pointing to it); the schematics need ``schemdraw``
(>= 0.19; the ``draw`` extra). The PNGs carry no metadata, so for a given build and TeX installation they are
byte-identical between runs. Runtime: one to two minutes on one node.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:
    import axionfem  # noqa: F401
except ImportError:                       # source checkout without an install
    sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PANELS = ("cavity", "srf", "readout", "chain")
WIDTH_PX = 1200          # width of every README panel
FS = 8.0                 # base font size [pt] of the field panels
CMAP = "inferno"         # field magnitudes
CMAP_SIGNED = "RdBu_r"   # signed fields, symmetric about zero
METAL = "#a4a7ad"        # conductors drawn in a field panel (the tuning rod)
EDGE = "0.55"            # cavity walls: mid grey reads against the dark field and the white page
FS_READOUT = 30          # font size of the readout drawing at its native scale
FS_CHAIN = 21            # font size of the SQUID-chain drawing at its native scale
DPI_CIRCUIT = 300        # resolution of the native-scale schematic renders


# ============================================================================================ style and output
def latex_style():
    """Fixed rcParams: LaTeX text in Computer Modern, sizes for a figure about 3.3 in wide, inward ticks on all four
    sides with minor ticks."""
    plt.rcdefaults()
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS - 1,
        "ytick.labelsize": FS - 1, "axes.linewidth": 0.6, "axes.labelpad": 2.0, "axes.titlepad": 4.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "standard",
        "path.simplify": True,
    })


def save_png(im: Image.Image, path: Path) -> Path:
    """Write an RGB PNG without metadata, losslessly optimized."""
    im.convert("RGB").save(path, optimize=True)
    return path


def fig_image(fig, dpi) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def colorbar(fig, mappable, cax, label, ticks):
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.set_label(label)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.5)
    return cb


def homepage_module():
    """make_homepage_figures.py (next to this script): the geometry and the solve helpers of its cavity and SRF
    cards, so that the README panels show the same designs."""
    spec = importlib.util.spec_from_file_location("make_homepage_figures", HERE / "make_homepage_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mode_tex(name: str) -> str:
    """'TM010' -> 'TM$_{010}$'."""
    return f"{name[:2]}$_{{{name[2:]}}}$"


# ============================================================================================ cavity
def cavity_field(work: Path, n_grid=501) -> dict:
    """|E_z| of the TM010-like mode (the TM mode with the largest form factor) of the homepage cavity design at the
    homepage card's finer mesh, sampled on an n_grid x n_grid grid over the cross-section (NaN in metal and outside)."""
    hp = homepage_module()
    c = hp.CAV
    R = c["R_m"]
    d = hp._cavity_design(R / 100)
    res = d.solve(work / "cavity", verbose=0)
    tm = sorted((m for m in res.cavity_modes() if m["family"] == "TM"), key=lambda m: m["f_Hz"])
    m = max(tm, key=lambda m: m["C"])
    if m is not tm[0]:
        raise RuntimeError("the TM mode with the largest form factor is not the lowest TM mode")
    P = res.cavity_mode_probe(m, x=(-R, R, n_grid), y=(-R, R, n_grid))
    th = math.radians(c["angle_deg"])
    rod = (c["pivot_m"] + c["arm_m"] * math.cos(th), c["arm_m"] * math.sin(th))
    print(f"cavity: TM010-like f = {m['f_Hz']:.7g} Hz, C = {m['C']:.7g}, rod centre {rod}")
    return {"x": P["X"][0], "y": P["Y"][:, 0], "absEz": np.abs(np.asarray(P["E_z"], float)), "R": R,
            "rod_center": rod, "rod_radius": c["rod_radius_m"], "f_Hz": m["f_Hz"], "C": m["C"]}


def cavity_panel(F: dict, out: Path) -> Path:
    from matplotlib.patches import Polygon

    latex_style()
    W, H = 3.3, 2.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.17, 0.15, 0.62, 0.62 * W / H])
    cax = fig.add_axes([0.81, 0.15, 0.025, 0.62 * W / H])
    x, y = F["x"] * 100, F["y"] * 100
    E = F["absEz"] / np.nanmax(F["absEz"])
    im = ax.imshow(E, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap=CMAP, vmin=0, vmax=1,
                   interpolation="bilinear", rasterized=True)
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    rc, a = np.array(F["rod_center"]) * 100, F["rod_radius"] * 100
    ax.add_patch(Polygon(np.c_[rc[0] + a * np.cos(t), rc[1] + a * np.sin(t)], closed=True, fc=METAL, ec="white",
                         lw=0.6))
    R = F["R"] * 100
    ax.plot(R * np.cos(t), R * np.sin(t), color=EDGE, lw=0.9)
    ax.set_xlim(-R * 1.04, R * 1.04)
    ax.set_ylim(-R * 1.04, R * 1.04)
    ax.set_aspect("equal")
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xlabel(r"$x$ [cm]")
    ax.set_ylabel(r"$y$ [cm]")
    ax.set_title(r"TM$_{010}$-like: $f=%.1f$ MHz, $C=%.2f$" % (F["f_Hz"] / 1e6, F["C"]), pad=4)
    colorbar(fig, im, cax, r"$|E_z|/|E_z|_{\max}$", [0, 0.5, 1])
    return save_png(fig_image(fig, WIDTH_PX / W), out)


# ============================================================================================ SRF
def srf_field(work: Path) -> dict:
    """Overlap density of the homepage SRF design (the 60 L Lasenby cylinder at its TE012/TM013 degeneracy) at the
    homepage card's finer mesh, on a grid over the meridional section mirrored about the axis (NaN outside)."""
    hp = homepage_module()
    if hp.SRF["design"] != "lasenby_60L":
        raise RuntimeError(f"make_homepage_figures.SRF selects {hp.SRF['design']!r}; this panel shows the 60 L "
                           "cylinder")
    ctx = SimpleNamespace(work=work)
    d, res, o = hp._srf_solve(ctx, 40, True)
    ov, rr, zz, tris, _ = hp._srf_overlap(o)
    W = np.asarray(d.srf_contour["wall"])                                # (r, z) [m]
    zmax = float(np.abs(W[:, 1]).max()) * 100
    rmax = float(W[:, 0].max()) * 100
    zg, rg = np.linspace(-zmax, zmax, 943), np.linspace(0.0, rmax, 401)
    Zg, Rg = np.meshgrid(zg, rg)
    G = hp.lattice_interpolator(zz * 100, rr * 100, tris, ov, rmax)(Zg, Rg)
    G = np.vstack([G[:0:-1], G])
    print(f"srf: {o['pump']}/{o['signal']} f = {o['f_pump_Hz']:.7g} / {o['f_signal_Hz']:.7g} Hz, "
          f"eta_10 = {o['eta_10']:.7g}, L = {d.srf_contour['length']:.6g} m, peak {np.nanmax(np.abs(ov)):.4g}")
    return {"G": G, "zmax": zmax, "rmax": rmax, "dz": zg[1] - zg[0], "dr": rg[1] - rg[0],
            "lim": float(np.nanmax(np.abs(ov))), "pump": o["pump"], "signal": o["signal"],
            "f_Hz": o["f_pump_Hz"], "eta_10": o["eta_10"]}


def srf_panel(F: dict, out: Path) -> Path:
    from matplotlib.patches import Rectangle

    latex_style()
    zmax, rmax, lim = F["zmax"], F["rmax"], F["lim"]
    W = 3.3
    aw = 0.62 * W                                  # axes width [in], as in the cavity panel
    ah = aw * rmax / zmax                          # equal aspect over the same 4 % margins
    bottom, top = 0.41, 0.33
    H = bottom + ah + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.17, bottom / H, 0.62, ah / H])
    cax = fig.add_axes([0.81, bottom / H, 0.025, ah / H])
    hz, hr = 0.5 * F["dz"], 0.5 * F["dr"]          # the grid points are pixel centres
    im = ax.imshow(F["G"], origin="lower", extent=(-zmax - hz, zmax + hz, -rmax - hr, rmax + hr), cmap=CMAP_SIGNED,
                   vmin=-lim, vmax=lim, interpolation="bilinear", rasterized=True)
    ax.add_patch(Rectangle((-zmax, -rmax), 2 * zmax, 2 * rmax, fill=False, ec=EDGE, lw=0.9))
    ax.axhline(0.0, color="0.35", lw=0.5, ls=(0, (6, 2, 1, 2)))
    ax.set_xlim(-zmax * 1.08, zmax * 1.08)
    ax.set_ylim(-rmax * 1.08, rmax * 1.08)
    ax.set_aspect("equal")
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xlabel(r"$z$ [cm]")
    ax.set_ylabel(r"$x$ [cm]")
    ax.set_title(r"%s/%s: $f=%.3f$ GHz, $\eta_{10}=%.2f$" % (mode_tex(F["pump"]), mode_tex(F["signal"]),
                                                          F["f_Hz"] / 1e9, F["eta_10"]), pad=4)
    k = int(math.floor(lim))
    colorbar(fig, im, cax, r"$\tilde{\mathbf{E}}_1\cdot\tilde{\mathbf{B}}_0$", [-k, 0, k])
    return save_png(fig_image(fig, WIDTH_PX / W), out)


# ============================================================================================ schematics
def trim(im: Image.Image, pad=8) -> Image.Image:
    """Crop uniform white margins."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.nonzero(a < 245)
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width),
                    min(ys.max() + pad, im.height)))


def _find_label(d, text):
    """The Label element of drawing ``d`` whose text is ``text`` (exactly one)."""
    hits = [e for e in d.elements if type(e).__name__ == "Label" and e._userlabels and e._userlabels[0].label == text]
    if len(hits) != 1:
        raise RuntimeError(f"expected one label {text!r} in the drawing, found {len(hits)}")
    return hits[0]


def _move_label(d, elm, text, xy, new=None, **kw):
    """Replace the Label element ``text`` by one at ``xy`` (schemdraw places elements when they are added)."""
    e = _find_label(d, text)
    lb = e._userlabels[0]
    d.elements.remove(e)
    args = dict(color=lb.color, fontsize=lb.fontsize, halign=lb.halign, valign=lb.valign)
    args.update(kw)
    d.add(elm.Label().at(xy).label(new if new is not None else text, **args))


def _relabel(d, text, new):
    """Replace element label ``text`` by ``new`` in place (same anchor and alignment)."""
    hits = [g for e in d.elements for g in e.segments if getattr(g, "text", None) == text]
    if len(hits) != 1:
        raise RuntimeError(f"expected one element label {text!r}, found {len(hits)}")
    hits[0].text = new


def _detach_label(d, elm, text, dxy, **kw):
    """Draw element label ``text`` as a free Label at the element's centre + ``dxy`` (clear of its symbol)."""
    hits = [(e, lb) for e in d.elements for lb in getattr(e, "_userlabels", []) if lb.label == text]
    if len(hits) != 1:
        raise RuntimeError(f"expected one element label {text!r}, found {len(hits)}")
    e, lb = hits[0]
    e._userlabels.remove(lb)       # labels are placed (as text segments) when the element is added: drop those
    parts = {text, *text.split("\n")}
    n0 = len(e.segments)
    e.segments = [g for g in e.segments if getattr(g, "text", None) not in parts]
    if len(e.segments) == n0:
        raise RuntimeError(f"no text segment {text!r} on its element")
    cx, cy = e.center
    d.add(elm.Label().at((cx + dxy[0], cy + dxy[1])).label(text, color=lb.color, fontsize=lb.fontsize, **kw))


@contextlib.contextmanager
def draw_font(D, fs, note_step=None):
    """Set axionfem.draw's font size (and the line step of its note blocks) for one drawing; restore them after."""
    fs0, defaults0 = D.FS, D.note_lines.__defaults__
    D.FS = fs
    D.note_lines.__defaults__ = (fs - 2, note_step if note_step is not None else defaults0[1], defaults0[2])
    try:
        yield
    finally:
        D.FS, D.note_lines.__defaults__ = fs0, defaults0


def readout_drawing(D, elm):
    """The series readout loop with LaTeX-friendly labels (see the module docstring)."""
    from axionfem import circuit

    r = circuit.series_readout(L_p_H=1.4e-6, Q=1e6, f_Hz=1e6, T_K=0.03, L_lead_H=50e-9, R_lead_ohm=1e-4,
                               T_lead_K=4.0, tan_delta=1e-6, L_in_H=2e-9, R_in_ohm=1e-5)
    d = r.draw(annotate=False)
    errs = D.check_drawing(d, r)
    if errs:
        raise RuntimeError(f"readout drawing does not match its circuit: {errs}")
    jj = [e for e in d.elements if type(e).__name__ == "Josephson"]
    xj = max(e._userparams["at"].x for e in jj)                # right junction of the SQUID
    sq = _find_label(d, r"$L_{\rm sq}$")._userparams["at"]
    _move_label(d, elm, r"$L_{\rm sq}$", (xj + 0.3, sq.y), halign="left", valign="center")
    m = _find_label(d, "M")._userparams["at"]
    _move_label(d, elm, "M", (m.x, m.y), new=r"$M$")
    notes = [e for e in d.elements if type(e).__name__ == "Label" and e._userparams["at"].y < 0.0]
    for e in notes:
        d.elements.remove(e)
    # the SQUID notes inside the loop, which spans x in [0.3, x_Lin] and y in [0, ~9]: a centred block
    x_in = max(e._userparams["at"].x for e in d.elements if type(e).__name__ == "Dot")
    yc, step = 5.4, 1.25
    for k, e in enumerate(notes):
        lb = e._userlabels[0]
        d.add(elm.Label().at((0.5 * (0.3 + x_in), yc + step * (len(notes) / 2 - k))).label(
            lb.label, color=lb.color, fontsize=lb.fontsize, halign="center", valign="top"))
    _detach_label(d, elm, r"$R_{\rm loss}$" + "\n$T$", (-0.75, 0.0), halign="right", valign="center")
    # schemdraw sizes the canvas from estimated text extents: keep the canvas of the drawing with its original
    # labels, so that the relabels below (all inside it) leave everything else in place
    bbox0 = d.get_bbox()
    rl = r"$R_{\rm loss}$" + "\n$T$"
    _move_label(d, elm, rl, _find_label(d, rl)._userparams["at"], new=r"$R_{\rm loss}(T)$")
    _relabel(d, r"$R_{\rm lead}$" + "\n$T$", r"$R_{\rm lead}(T)$")
    _relabel(d, r"$R_{\rm in}$" + "\n$T$", r"$R_{\rm in}(T)$")
    # the module name in typewriter: the wider label moves left by its extra width (0.353 unit), so that its right
    # edge, which bounds the drawing, stays in place, and up by the 0.2 unit by which the typewriter line lowers the
    # bottom-aligned label (keeps its gap to the amplifier)
    lat = _find_label(d, "later stages\n(squid.Chain)")._userparams["at"]
    _move_label(d, elm, "later stages\n(squid.Chain)", (lat.x - 0.353, lat.y + 0.2),
                new="later stages\n(\\texttt{squid.Chain})")
    d.get_bbox = lambda: bbox0
    d.config(lw=3.0)
    return d


def chain_drawing(D, elm):
    """The C1+low+RT dc SQUID chain with short stage labels (see the module docstring)."""
    from axionfem import squid

    ch = squid.chain("C1+low+RT")
    c = ch.draw(annotate=False)
    notes = [e for e in c.elements if type(e).__name__ == "Label" and e._userparams.get("at").y < -2.0]
    for e in notes:
        c.elements.remove(e)
    xs = sorted({round(e._userparams["at"].x, 3) for e in notes})
    s2 = ch.second
    short = [("1st stage", "dc SQUID"), ("2nd stage", rf"SQUID array (${s2.N_ser}\times{s2.N_par}$)"),
             ("RT preamp", "")]
    if len(xs) != len(short):
        raise RuntimeError(f"expected {len(short)} stage-note columns in the chain drawing, found {len(xs)}")
    for x, lines in zip(xs, short):
        for k, ln in enumerate(s for s in lines if s):
            c.add(elm.Label().at((x, -2.4 - 0.8 * k)).label(ln, color=D.SQC, fontsize=D.FS, halign="center",
                                                          valign="top"))
    sq = _find_label(c, r"$L_{\rm sq}$")._userparams["at"]
    _move_label(c, elm, r"$L_{\rm sq}$", (sq.x + 0.7 + 0.3, sq.y), halign="left", valign="center")
    m2 = _find_label(c, r"$M_{IN2}$")._userparams["at"]
    _move_label(c, elm, r"$M_{IN2}$", (m2.x, m2.y), new=r"$M_{\rm IN2}$")
    _detach_label(c, elm, r"$R_{\rm lead}(T_{\rm lead})$", (0.0, 1.05), halign="center", valign="bottom")
    c.config(lw=2.4)
    return c


def render_drawing(d, work: Path, name: str) -> Image.Image:
    """Render a schemdraw drawing at its native scale (LaTeX text) to a trimmed RGB image."""
    p = work / f"{name}.png"
    d.save(str(p), dpi=DPI_CIRCUIT)
    plt.close("all")
    return trim(Image.open(p).convert("RGB"))


def circuit_panel(kind: str, work: Path, out: Path) -> Path:
    from axionfem import draw as D

    latex_style()
    _, elm = D._schemdraw()
    if kind == "readout":
        with draw_font(D, FS_READOUT, note_step=1.25):
            im = render_drawing(readout_drawing(D, elm), work, "readout")
    else:
        with draw_font(D, FS_CHAIN):
            im = render_drawing(chain_drawing(D, elm), work, "chain")
    h = round(im.height * WIDTH_PX / im.width)
    return save_png(im.resize((WIDTH_PX, h), Image.LANCZOS), out)


# ============================================================================================ driver
OUTPUTS = {"cavity": "readme_cavity_offcentre.png", "srf": "readme_srf_cylinder.png",
           "readout": "readme_readout_circuit.png", "chain": "readme_squid_chain.png"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--work", default="runs/readme_panels")
    ap.add_argument("--only", default=",".join(PANELS))
    a = ap.parse_args(argv)
    only = [n for n in a.only.split(",") if n]
    unknown = sorted(set(only) - set(PANELS))
    if unknown:
        ap.error(f"unknown panel(s) {unknown}; choose from {list(PANELS)}")
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    for name in PANELS:
        if name not in only:
            continue
        p = out / OUTPUTS[name]
        if name == "cavity":
            cavity_panel(cavity_field(work), p)
        elif name == "srf":
            srf_panel(srf_field(work), p)
        else:
            circuit_panel(name, work, p)
        with Image.open(p) as im:
            print(f"{name}: {p} {im.size[0]} x {im.size[1]}, {round(p.stat().st_size / 1024)} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
