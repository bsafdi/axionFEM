"""Regenerate the homepage graphics of axionFEM (docs/images/home/ and docs/images/logo/).

    python docs/images/make_homepage_figures.py [--out docs/images/home] [--work runs/homepage]
           [--only hero,cavity,srf,readout,mech,reach,fluids,chem,social,logo] [--themes light,dark]
           [--skip-unavailable] [--check]

Every picture is axionFEM output: the solves run here, and the numbers that captions quote come from the same
run. Each figure function returns a manifest entry with the design and its parameters, the solver settings, a
convergence check of every quoted number (the number at two discretizations and their relative difference), the
numbers a caption may quote, the wall time and the figure's status (validated or beta, from ``STATUS``).
``manifest.json`` collects the entries together with the build features of ``axionfem.build_info()``, the axionFEM
version and the SHA-256 of this script; ``captions.md`` is generated from the manifest (pages include its sections
with pymdownx.snippets, e.g. ``--8<-- "docs/images/home/captions.md:hero"``), so no caption number is typed by hand.

Figures (``--only`` names, output stems):

* ``hero``    ``hero_toroid_{light,dark}.png`` (2400 x 1000): the quick-start toroid, 3D cutaway rendered by the
  C++ renderer: static |B| on the cut planes, pickup-mode field lines in the bore.
* ``cavity``  ``card_cavity``: E_z of the three lowest TM modes of a copper cavity with an off-centre copper rod
  (cross-section solver), checked against the method of particular solutions.
* ``srf``     ``card_srf``: overlap density E1.B0 of the degenerate TE012 pump and TM013 signal of the published
  60 L closed cylinder of Lasenby (arXiv:1912.11056), checked against the closed form (``SRF`` selects the design).
* ``readout`` ``card_readout``: noise of a resonant readout of the quick-start toroid with a published dc SQUID
  chain, referred to the pickup EMF.
* ``mech``    ``card_mech``: von Mises stress of an illustrative solenoid winding under its own Lorentz force, drawn
  by the C++ renderer as a 3D cutaway.
* ``reach``   ``card_reach``: exclusion limits of published haloscope runs recomputed from their published
  parameters (a few inferred or assumed where a paper does not state them), next to the published values.
* ``fluids``  ``card_fluids`` (needs an MPI build): natural convection in a differentially heated square cavity.
* ``chem``    ``card_chem``: primary current distribution on a disk electrode.
* ``social``  ``social_preview.png`` (1280 x 640): the dark hero and the wordmark.
* ``logo``    ``../logo/``: icon, wordmarks and favicons (SVG, with PNG renderings).

Cards are 1200 x 800 px with transparent backgrounds and theme-specific ink (light: #1f2430, dark: #e6e8ec); the
hero uses the renderer backgrounds (#ffffff, #13151a). Colour maps: viridis for magnitudes, coolwarm (symmetric
about zero) for signed fields. Card text is typeset by LaTeX (matplotlib's ``text.usetex``, Computer Modern), and
plot axes have inward ticks on all four sides with minor ticks; the logo files and the social preview keep their
DejaVu Sans lettering (drawn as paths, without LaTeX). Output is deterministic for a given build and TeX
installation (fixed rcParams, no PNG metadata, the byte-deterministic C++ renderer): images and captions are
byte-identical between runs. In the manifest the ``seconds`` fields change, and a different OpenMP thread count can
change the last digit of relative differences that sit at the round-off level (about 1e-9).

Needs a build with Python bindings and the mesher (``axf-mesher``), NumPy, SciPy, matplotlib and Pillow, and a
LaTeX toolchain for matplotlib's ``text.usetex`` (``latex`` and ``dvipng`` on the PATH, with the type1cm, type1ec
and cm-super packages); ``fluids`` needs an MPI build (``axionfem.build_info()["fluids"]``). ``--skip-unavailable``
skips, with a message, the figures whose module this build lacks instead of stopping. ``--check`` only verifies that
every file in the manifest exists and that the manifest version equals ``axionfem.__version__``. Runtime on one node:
a few minutes; not run in CI.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:
    import axionfem as af
except ImportError:                       # source checkout without an install
    sys.path.insert(0, str(ROOT / "python"))
    import axionfem as af

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from axionfem.render import render_png  # noqa: E402

FIGURES = ("hero", "cavity", "srf", "readout", "mech", "reach", "fluids", "chem", "social", "logo")
DPI = 200
CARD_IN = (6.0, 4.0)                      # 1200 x 800 px at DPI
HERO_PX = (2400, 1000)
SOCIAL_PX = (1280, 640)
# light / dark ink, the C++ renderer backgrounds, and the notebooks' categorical palette
THEME = {
    "light": {"ink": "#1f2430", "muted": "#5b6170", "faint": "#c9ccd3", "grid": "#e4e3df", "bg": "#ffffff",
              "grey": "#6f6e69", "metal": "#b4b8c0", "metal_edge": "#5b6170"},
    "dark": {"ink": "#e6e8ec", "muted": "#a3a8b3", "faint": "#454a55", "grid": "#2c313a", "bg": "#13151a",
             "grey": "#a8a7a2", "metal": "#6b717d", "metal_edge": "#c3c7cf"},
}
BLUE, ORANGE, GREEN, GOLD, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
# lines drawn over a coolwarm or light field (nodal lines, streamlines): one dark grey in both themes, because the
# middle of coolwarm is light grey in either theme
FIELD_LINE = "#2b2f38"
MU0 = 1.25663706212e-6                   # H/m, as in the C++ core


# ============================================================================================ helpers
def sig(x, n=7):
    """Round to n significant digits for the manifest (stable text across platforms)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.{n}g}")


def rel(a, b):
    """Relative difference (a - b)/|b|."""
    return (a - b) / abs(b) if b else float("nan")


def check_row(name, coarse, fine, label_coarse, label_fine):
    """One convergence-check row: the quantity at two discretizations and their relative difference."""
    return {"quantity": name, label_coarse: sig(coarse), label_fine: sig(fine), "rel_diff": sig(rel(coarse, fine), 3)}


def style(theme):
    """Fixed rcParams for every matplotlib card: LaTeX text in Computer Modern, inward ticks on all four sides with
    minor ticks."""
    t = THEME[theme]
    plt.rcdefaults()
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"], "font.size": 14,
        "text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.edgecolor": t["muted"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink"],
        "ytick.labelcolor": t["ink"], "axes.facecolor": "none", "figure.facecolor": "none",
        "savefig.facecolor": "none", "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "legend.frameon": False, "axes.labelsize": 15, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.pad": 5, "ytick.major.pad": 5, "axes.linewidth": 1.0, "svg.hashsalt": "axionfem",
        "path.simplify": True,
    })
    return t


def save_card(fig, path):
    """1200 x 800 transparent PNG, no metadata, losslessly optimized."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True, metadata={"Software": None})
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue()))
    assert im.size == (int(CARD_IN[0] * DPI), int(CARD_IN[1] * DPI)), im.size
    im.save(path, optimize=True)
    return path


def colorbar(fig, mappable, rect, label, t, ticks=None, fmt=None):
    """A slim vertical colour bar with units."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks, format=fmt)
    cb.minorticks_off()
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor(t["muted"])
    cb.ax.tick_params(labelsize=13, length=4, width=0.8, direction="in", color=t["muted"],
                      labelcolor=t["ink"])
    cb.set_label(label, fontsize=14, color=t["ink"], labelpad=6)
    return cb


def scale_bar(ax, x0, y0, length, label, t, lw=2.5, dy=0.0):
    """A thin horizontal scale bar starting at (x0, y0) in data units, label centred below."""
    ax.plot([x0, x0 + length], [y0, y0], color=t["ink"], lw=lw, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - dy, label, ha="center", va="top", fontsize=14, color=t["ink"])


def field_image(ax, F, extent, **kw):
    """imshow of a sampled field: interpolated in data space and then colour-mapped, so every pixel of the field
    takes one of the colour map's 256 colours. That looks the same as interpolating colours and keeps the PNG
    small (smooth colour gradients compress poorly)."""
    return ax.imshow(F, extent=extent, origin="lower", interpolation="bilinear", interpolation_stage="data",
                     rasterized=True, **kw)


def lattice_interpolator(x, y, tris, values, length):
    """Linear interpolator of values on a mode-field lattice. The lattice repeats the points shared by neighbouring
    elements; they are merged first (to 1e-9 of `length`). NaN outside the lattice."""
    import matplotlib.tri as mtri

    x, y = np.asarray(x, float), np.asarray(y, float)
    key = np.round(np.c_[x, y] / (length * 1e-9)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    tu = mtri.Triangulation(x[first], y[first], inv.ravel()[np.asarray(tris)])
    f = mtri.LinearTriInterpolator(tu, np.asarray(values, float)[first])
    return lambda X, Y: np.ma.filled(f(X, Y), np.nan)


def png_kb(path):
    return round(Path(path).stat().st_size / 1024)


def build_info():
    """The keys of axionfem.build_info() that describe the build's features (no paths, host data or commit
    hashes)."""
    bi = af.build_info()
    bi = json.loads(bi) if isinstance(bi, str) else dict(bi)
    keep = ("version", "source_dirty", "git_dirty", "mfem", "mpi", "openmp", "metis", "strumpack",
            "strumpack_version", "fluids", "engine_3d", "device", "gpu_backend")
    return {k: bi[k] for k in keep if k in bi}


def script_sha256():
    """SHA-256 of this script: ties each figure of the manifest to the generator that drew it."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ============================================================================================ hero
def _set_quantity(scene, name, values):
    q = scene["field"]["quantities"][name]
    q["data"] = base64.b64encode(np.ascontiguousarray(values, dtype=np.float32).tobytes()).decode()


def _get_quantity(scene, name):
    q = scene["field"]["quantities"][name]
    return np.frombuffer(base64.b64decode(q["data"]), dtype=np.float32).copy()


def crop_render(im, bg, margin=0.035):
    """Deterministic crop of a renderer image (annotate=False) to the drawn object: drops the subtitle band at the
    top, the colour bar at the right and the axis label above the body; returns the RGB crop with a margin of
    background around the object."""
    a = np.asarray(im.convert("RGB")).astype(int)
    on = np.abs(a - np.array(bg)).max(axis=2) > 12
    H, W = on.shape
    rows = on.any(axis=1)
    r0 = int(np.argmax(rows))                         # first row of the subtitle text
    r1 = r0 + int(np.argmax(~rows[r0:]))              # first blank row after it
    on[:r1] = False
    # column clusters separated by wide gaps: the object is the widest (the colour bar is narrow)
    cols = on.any(axis=0)
    runs, start = [], None
    for x in range(W + 1):
        c = cols[x] if x < W else False
        if c and start is None:
            start = x
        if not c and start is not None:
            runs.append([start, x])
            start = None
    groups = [runs[0]]
    for s, e in runs[1:]:
        if s - groups[-1][1] < 0.02 * W:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    c0, c1 = max(groups, key=lambda g: g[1] - g[0])
    on[:, :c0] = False
    on[:, c1:] = False
    span = np.zeros(H, int)
    for y in range(H):
        xs = np.nonzero(on[y])[0]
        if xs.size:
            span[y] = xs[-1] - xs[0] + 1
    body_top = int(np.argmax(span > 0.2 * (c1 - c0)))
    axis_w = max(6, W // 400)                         # the axis line is narrower than this
    cap_top = body_top
    while cap_top > 0 and span[cap_top - 1] > axis_w:  # up through the top of the body
        cap_top -= 1
    label = np.nonzero(span[:cap_top] > axis_w)[0]     # the axis label above it
    top = int(label.max()) + 2 if label.size else int(np.argmax(span > 0))
    on[:top] = False
    ys = np.nonzero(on.any(axis=1))[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    m = int(margin * (y1 - y0))
    box = (c0 - m, y0 - m, c1 + m, y1 + m)
    out = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), tuple(bg))
    src = Image.fromarray(a.astype(np.uint8)).crop((c0, top, c1, y1))
    out.paste(src, (m, m + (top - y0)))
    return out


def fit_canvas(im, size, bg, fill=0.96):
    """Scale an object crop to fit `fill` of the canvas height (or width) and centre it on a background canvas."""
    W, H = size
    s = min(fill * H / im.height, fill * W / im.width)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    out = Image.new("RGB", size, tuple(bg))
    out.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return out


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


HERO_VIEW = {"cutaway_deg": 280.0, "elev_deg": 20.0, "azim_deg": -50.0}


def fig_hero(ctx):
    d = af.template("toroid", R=2.0, H=2.0)
    d.set(y=0.6)
    res, rows = {}, {}
    for order in (2, 3):
        r = d.solve(ctx.work / f"hero_order{order}", order=order, verbose=0)
        res[order] = r
        c = r.coupling
        rows[order] = {"Psi_SI_H_T_m2": r.Psi, "Psi_natural_T_m3": c["Psi_natural_T_m3"], "L_p_H": r.L_p,
                       "F_natural_T4_m10": c["F_natural_T4_m10"], "B_max_T": r.B_max,
                       "dofs": r.data["mesh"]["dofs"]}
    r = res[3]
    ctx.toroid = {"order2": rows[2], "order3": rows[3], "doc": d.doc}
    scene = r.scene(subdiv=2, contours=16)
    B = _get_quantity(scene, "Bmag")
    peak = float(np.nanmax(B))
    zero = B < 1e-9 * peak                       # the static field vanishes outside the toroid volume
    masked_max = float(B[zero].max()) if zero.any() else 0.0
    B[zero] = np.nan
    _set_quantity(scene, "Bmag", B)
    files = []
    for theme in ctx.themes:
        bg = (255, 255, 255) if theme == "light" else (19, 21, 26)
        png = render_png(scene, view="3d", quantity="Bmag", field_lines=True, annotate=False, title=" ",
                         width=3200, height=1400, supersample=4, theme=theme, **HERO_VIEW)
        im = fit_canvas(crop_render(Image.open(io.BytesIO(png)), bg), HERO_PX, bg, fill=0.97)
        p = ctx.out / f"hero_toroid_{theme}.png"
        im.save(p, optimize=True)
        files.append(p)
        if theme == "dark":
            ctx.hero_dark = im                   # reused by the social preview
    p = d.resolved_params()
    return {
        "title": "Toroidal lumped-element detector",
        "files": files,
        "design": {"template": "toroid", "parameters": {k: sig(v) for k, v in p.items()},
                   "note": "the quick-start toroid: af.template('toroid', R=2.0, H=2.0).set(y=0.6)"},
        "solve": {"kind": "axisymmetric magnetoquasistatic", "order": 3, "mesh": d.doc.get("mesh"),
                  "dofs": rows[3]["dofs"]},
        "render": {"renderer": "C++ (axionfem.render)", "view": "3d", "quantity": "Bmag", **HERO_VIEW,
                   "scene": {"subdiv": 2, "contours": 16},
                   "mask": "|B| below 1e-9 of its peak (the field-free bore and exterior) is left uncoloured",
                   "masked_max_over_peak": sig(masked_max / peak, 3)},
        "check": [check_row("Psi [H T m^2]", rows[2]["Psi_SI_H_T_m2"], rows[3]["Psi_SI_H_T_m2"], "order2", "order3"),
                  check_row("L_p [H]", rows[2]["L_p_H"], rows[3]["L_p_H"], "order2", "order3"),
                  check_row("B_max [T]", rows[2]["B_max_T"], rows[3]["B_max_T"], "order2", "order3")],
        "numbers": {"R_m": sig(p["R"]), "H_m": sig(p["H"]), "NI_A": sig(p["NI"]),
                    "Psi_SI_H_T_m2": sig(rows[3]["Psi_SI_H_T_m2"]), "L_p_H": sig(rows[3]["L_p_H"]),
                    "B_max_T": sig(rows[3]["B_max_T"])},
    }


# ============================================================================================ cavity
CAV = {"R_m": 0.210, "rod_radius_m": 0.0254, "pivot_m": 0.122, "arm_m": 0.057, "angle_deg": 120.0, "length_m": 1.010}


def _cavity_design(h):
    from axionfem import cavity_xsec as cx

    R = CAV["R_m"]
    regions = [{"name": "cavity", "shape": {"disk": {"center": [0.0, 0.0], "radius": R}}, "material": "vacuum"},
               {"name": "rod", "shape": {"disk": {"pivot": [CAV["pivot_m"], 0.0], "arm": CAV["arm_m"],
                                                   "angle_deg": CAV["angle_deg"], "radius": CAV["rod_radius_m"]}},
                "material": "copper"}]
    return cx.xsec_design(regions, name="cavity_offcentre_rod", wall_material="copper", length=CAV["length_m"],
                          modes=6, h=h, surface_size=R / 250)


def fig_cavity(ctx):
    from axionfem import cavity_xsec as cx

    R = CAV["R_m"]
    sols = {}
    for n in (70, 100):
        d = _cavity_design(R / n)
        r = d.solve(ctx.work / f"cavity_h{n}", verbose=0)
        modes = sorted((m for m in r.cavity_modes() if m["family"] == "TM"), key=lambda m: m["f_Hz"])
        sols[n] = (d, r, modes, r.data["mesh"]["dofs"])
    d, r, modes, dofs = sols[100]
    th = math.radians(CAV["angle_deg"])
    rc = (CAV["pivot_m"] + CAV["arm_m"] * math.cos(th), CAV["arm_m"] * math.sin(th))
    ref = cx.eccentric_rod_tm_mode(R, CAV["rod_radius_m"], math.hypot(*rc))
    show = modes[:3]
    import matplotlib.tri as mtri

    fields = []
    g = np.linspace(-R, R, 801)
    GX, GY = np.meshgrid(g, g)
    for m in show:
        F = r.cavity_mode_field(m, subdiv=2)            # sign: TM modes with int E_z dA > 0
        E = np.asarray(F["E_z"], float)
        # E_z vanishes on every conductor surface, so the zero contour would also trace the wall and the rod: the
        # triangles within R/150 of either are left out of the nodal-line triangulation
        x, y = np.asarray(F["x"], float), np.asarray(F["y"], float)
        near = (R - np.hypot(x, y) < R / 150) | (np.hypot(x - rc[0], y - rc[1]) - CAV["rod_radius_m"] < R / 150)
        tri = mtri.Triangulation(x, y, F["tris"], mask=near[np.asarray(F["tris"])].any(axis=1))
        Eg = lattice_interpolator(x, y, F["tris"], E, R)(GX, GY)   # NaN in the rod and outside
        # inside the wall, NaN only marks the rod (and slivers next to the curved conductors): E_z = 0 there, which
        # also leaves no transparent pixels between the field and the rod outline
        Eg[np.isnan(Eg) & (np.hypot(GX, GY) < R)] = 0.0
        fields.append((tri, E / np.nanmax(np.abs(E)), Eg / np.nanmax(np.abs(E))))
    from matplotlib.patches import Circle

    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        big = fig.add_axes([0.02, 0.1, 0.555, 0.83])
        small = [fig.add_axes([0.59, 0.525, 0.255, 0.385]), fig.add_axes([0.59, 0.1, 0.255, 0.385])]
        for ax, (tri, E, Eg) in zip([big] + small, fields):
            field_image(ax, Eg, (-R, R, -R, R), cmap="coolwarm", vmin=-1, vmax=1)
            ax.tricontour(tri, E, levels=[0.0], colors=[FIELD_LINE], linewidths=1.1, alpha=0.8)
            ax.add_patch(Circle(rc, CAV["rod_radius_m"], fc=t["metal"], ec=t["metal_edge"], lw=1.0, zorder=3))
            ax.add_patch(Circle((0.0, 0.0), R, fill=False, ec=t["ink"], lw=1.6, zorder=4))
            ax.set_xlim(-1.03 * R, 1.03 * R)
            ax.set_ylim(-1.03 * R, 1.03 * R)
            ax.set_aspect("equal")
            ax.axis("off")
        scale_bar(big, -R, -1.02 * R, 0.1, "10 cm", t, dy=0.012)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1, 1))
        cb = colorbar(fig, sm, [0.87, 0.2, 0.016, 0.6], "", t, ticks=[-1, 0, 1])
        cb.ax.set_yticklabels([r"$-$max", "0", "max"])
        cb.ax.set_title("$E_z$", fontsize=15, color=t["ink"], pad=8)
        p = save_card(fig, ctx.out / f"card_cavity_{theme}.png")
        files.append(p)
    c70, c100 = sols[70][2], sols[100][2]
    check = [check_row(f"f mode {i} [Hz]", c70[i]["f_Hz"], c100[i]["f_Hz"], "h=R/70", "h=R/100") for i in range(3)]
    check += [check_row("C mode 0", c70[0]["C"], c100[0]["C"], "h=R/70", "h=R/100")]
    check += [{"quantity": "TM010-like vs method of particular solutions", "f_mps_Hz": sig(ref["f_Hz"]),
               "C_mps": sig(ref["C"]), "rel_diff_f": sig(rel(show[0]["f_Hz"], ref["f_Hz"]), 3),
               "rel_diff_C": sig(rel(show[0]["C"], ref["C"]), 3)}]
    return {
        "title": "Cavity haloscope",
        "files": files,
        "design": {"builder": "axionfem.cavity_xsec.xsec_design", "parameters": {k: sig(v) for k, v in CAV.items()},
                   "rod_center_m": sig(list(rc)), "walls": "builtin copper (cold copper, anomalous skin effect)",
                   "note": "illustrative rod position in a 210 mm copper cavity"},
        "solve": {"kind": "cross-section eigenmodes (symmetry xsec)", "order": 2, "h_m": sig(R / 100),
                  "surface_size_m": sig(R / 250), "dofs": dofs, "modes": 6},
        "check": check,
        "numbers": {"R_mm": sig(R * 1e3), "rod_diameter_mm": sig(2e3 * CAV["rod_radius_m"]),
                    "f_Hz": [sig(m["f_Hz"]) for m in show], "C": [sig(m["C"], 4) for m in show],
                    "Q0": [sig(m["Q0"], 4) for m in show], "f_mps_Hz": sig(ref["f_Hz"]), "C_mps": sig(ref["C"])},
    }


# ============================================================================================ SRF
# SRF card. Default: the 60 L closed cylinder of R. Lasenby, Phys. Rev. D 102, 015008 (2020), arXiv:1912.11056
# (R = 20 cm, length at the TE012/TM013 degeneracy, d/R = 2.355; C01 = eta_10^2 = 0.19 there), which the closed
# forms of axionfem.srf.cylinder_pair check. "projectx" draws instead the Project X 650 MHz beta = 0.9 cell with
# its axial length scaled by srf.elliptical_cell_design's default axial_factor (0.86), which makes its TM020 and
# TE011 modes degenerate.
SRF = {"design": "lasenby_60L", "R_m": 0.20, "pair": "TE012/TM013", "published_C01": 0.19,
       "source": "R. Lasenby, Phys. Rev. D 102, 015008 (2020), arXiv:1912.11056 (the 60 L cylinder)"}


def _srf_overlap(o):
    """Overlap density E1.B0 / (E1_rms B0_rms) of a figures_of_merit(..., with_fields=True) result on the lattice
    of the mode fields, with the sign that makes its volume average (eta_10) positive; also the lattice."""
    P, S = o["fields"]["pump"], o["fields"]["signal"]
    V = o["V_m3"]
    B0 = o["pump_peaks"]["B_rms"]                                    # sqrt(V^-1 int |B_pump|^2)
    E1 = math.sqrt(o["checks"]["int_E1^2 (1 expected)"] / V)         # sqrt(V^-1 int |E_signal|^2)
    ov = sum(np.asarray(S[e]) * (MU0 * np.asarray(P[h])) for e, h in (("E_r", "H_r"), ("E_phi", "H_phi"),
                                                                     ("E_z", "H_z"))) / (E1 * B0)
    rr, zz = np.asarray(P["r"]), np.asarray(P["z"])
    ov *= np.sign(np.sum(ov * rr))
    return ov, rr, zz, P["tris"], P


def _srf_solve(ctx, n, with_fields):
    """Solve the SRF design with mesh size (equator) radius / n at order 3; returns (design, result, figures of
    merit)."""
    from axionfem import srf

    if SRF["design"] == "projectx":
        req = srf.PROJECTX_650_B09["equator_radius"]
        d = srf.elliptical_cell_design(h=req / n, order=3, modes=6)
        r = d.solve(ctx.work / f"srf_projectx_h{n}", verbose=0)
        return d, r, srf.figures_of_merit(r, with_fields=with_fields)
    R = SRF["R_m"]
    pump, signal = SRF["pair"].split("/")
    d = srf.cylinder_pair_design(SRF["pair"], R, h=R / n, order=3, modes=8)
    r = d.solve(ctx.work / f"srf_cylinder_h{n}", verbose=0)
    return d, r, srf.figures_of_merit(r, pump, signal, with_fields=with_fields)


def fig_srf(ctx):
    import matplotlib.tri as mtri

    from axionfem import srf

    cyl = SRF["design"] != "projectx"
    meshes = (30, 40)
    sols = {n: _srf_solve(ctx, n, n == meshes[-1]) for n in meshes}
    d, r, o = sols[meshes[-1]]
    dofs = r.data["mesh"]["dofs"]
    con = d.srf_contour
    ov, rr, zz, tris, P = _srf_overlap(o)
    lim = float(np.nanmax(np.abs(ov)))
    W = np.asarray(con["wall"])                                      # (r, z) [m]
    zmax = float(np.abs(W[:, 1]).max()) * 100
    rmax = float(W[:, 0].max()) * 100
    # the meridional section on a grid of about one point per pixel, mirrored about the axis (NaN outside)
    zg, rg = np.linspace(-zmax, zmax, 901), np.linspace(0.0, rmax, 401)
    Zg, Rg = np.meshgrid(zg, rg)
    G = lattice_interpolator(zz * 100, rr * 100, tris, ov, rmax)(Zg, Rg)
    G = np.vstack([G[:0:-1], G])
    hz, hr = 0.5 * (zg[1] - zg[0]), 0.5 * (rg[1] - rg[0])             # the grid points are pixel centres
    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.0, 0.015, 0.86, 0.97])
        im = field_image(ax, G, (-zmax - hz, zmax + hz, -rmax - hr, rmax + hr), cmap="coolwarm", vmin=-lim, vmax=lim)
        for s in (1, -1):
            if not cyl:                                               # the sign change of the TM020 pump field
                tc = mtri.Triangulation(zz * 100, s * rr * 100, tris, mask=rr[tris].min(axis=1) < 0.03 * rr.max())
                ax.tricontour(tc, MU0 * np.asarray(P["H_phi"]), levels=[0.0], colors=[GREEN], linewidths=2.0)
            ax.plot(W[:, 1] * 100, s * W[:, 0] * 100, color=t["ink"], lw=2.0, solid_joinstyle="round",
                    solid_capstyle="round")
        ax.axhline(0, color=t["muted"], lw=1.0, ls=(0, (8, 3, 1.5, 3)))
        ax.set_aspect("equal")
        ax.axis("off")
        if cyl:
            ax.set_xlim(-zmax - 1.0, zmax + 1.0)
            ax.set_ylim(-rmax - 4.6, rmax + 1.6)
            scale_bar(ax, -zmax, -rmax - 1.6, 10.0, "10 cm", t, dy=0.6)
        else:
            ax.set_xlim(-zmax, zmax)
            ax.set_ylim(-21.5, 21.5)
            scale_bar(ax, -zmax + 0.5, -16.5, 10.0, "10 cm", t, dy=0.6)
        ticks = list(range(-int(lim), int(lim) + 1))
        cb = colorbar(fig, im, [0.875, 0.2, 0.016, 0.6], "", t, ticks=ticks)
        cb.ax.set_title(r"$E_1\!\cdot\!B_0$", fontsize=15, color=t["ink"], pad=8)
        files.append(save_card(fig, ctx.out / f"card_srf_{theme}.png"))
    a, b = sols[meshes[0]][2], sols[meshes[-1]][2]
    lab = ("h=R/30", "h=R/40") if cyl else ("h=Req/30", "h=Req/40")
    check = [check_row(k, a[k], b[k], *lab) for k in ("eta_10", "f_pump_Hz", "f_signal_Hz")]
    num = {"eta_10": sig(b["eta_10"], 4), "eta_10_squared": sig(b["eta_10"] ** 2, 4), "f_pump_Hz": sig(b["f_pump_Hz"]),
           "f_signal_Hz": sig(b["f_signal_Hz"]), "splitting_Hz": sig(b["splitting_Hz"], 3), "pump": b["pump"],
           "signal": b["signal"], "overlap_peak": sig(lim, 3)}
    if cyl:
        cf = srf.cylinder_pair(SRF["pair"], n_wall=1501, n_vol=601)
        R, L = SRF["R_m"], float(con["length"])
        f_cf = con["params"]["f_degenerate_Hz"]
        design = {"builder": "axionfem.srf.cylinder_pair_design", "pair": SRF["pair"], "source": SRF["source"],
                  "R_m": sig(R), "L_m": sig(L), "L_over_R": sig(cf["L_over_R"]), "V_m3": sig(math.pi * R * R * L),
                  "wall": "closed cylinder (pillbox)"}
        check += [{"quantity": "eta_10 and f against the closed forms (axionfem.srf.cylinder_pair)",
                   "eta_10_closed_form": sig(cf["eta_10"]), "f_closed_form_Hz": sig(f_cf),
                   "rel_diff_eta": sig(rel(b["eta_10"], cf["eta_10"]), 3),
                   "rel_diff_f_pump": sig(rel(b["f_pump_Hz"], f_cf), 3),
                   "rel_diff_f_signal": sig(rel(b["f_signal_Hz"], f_cf), 3)},
                  {"quantity": "eta_10^2 against the published C01", "published_C01": SRF["published_C01"],
                   "eta_10_squared": sig(b["eta_10"] ** 2, 4)}]
        num.update({"R_cm": sig(R * 100), "L_cm": sig(L * 100, 4), "L_over_R": sig(cf["L_over_R"], 4),
                    "V_L": sig(math.pi * R * R * L * 1e3, 3), "published_C01": SRF["published_C01"],
                    "eta_10_closed_form": sig(cf["eta_10"], 4)})
    else:
        design = {"builder": "axionfem.srf.elliptical_cell_design", "cell": "Project X 650 MHz beta = 0.9 regular cell",
                  "axial_factor": sig(con["params"]["axial_factor"]), "wall": "smooth (exact elliptical arcs)"}
    return {
        "title": "SRF heterodyne cavity",
        "files": files,
        "design": design,
        "solve": {"kind": "axisymmetric eigenmodes", "order": 3, "h_m": sig(con["equator_radius"] / meshes[-1]),
                  "dofs": dofs, "modes": 8 if cyl else 6},
        "check": check,
        "numbers": num,
    }


# ============================================================================================ readout
# sign: the sign s of the correlation between SQUID imprecision and backaction (squid.Chain.sign). The source gives
# only its magnitude, so s is a convention: s = -1 flips the sign of the correlation term, which moves the minimum
# of the total noise to the other side of f0. The card draws s = +1; the manifest records the s = -1 numbers too.
READOUT = {"f0_Hz": 1e6, "Q": 1e6, "T_K": 0.03, "L_in_H": 2e-9, "chain": "C1+low+RT", "sign": 1.0}
# label colours of the readout curves (the line colours, darkened or lightened for text contrast >= 4.4:1)
RO_LABEL = {"light": {"imprecision": BLUE, "thermal": "#c4501d", "backaction": VIOLET},
            "dark": {"imprecision": "#6aa5ea", "thermal": ORANGE, "backaction": "#9d92ee"}}


def _readout(L_p, Psi, sign):
    from axionfem import circuit, squid

    p = READOUT
    ch = squid.chain(p["chain"], sign=sign)
    r = circuit.series_readout(L_p_H=L_p, Psi_SI=Psi, Q=p["Q"], f_Hz=p["f0_Hz"], T_K=p["T_K"], L_in_H=p["L_in_H"],
                               M_H=1e-9)
    r.squid = ch.squid_for_resonator(p["f0_Hz"], p["Q"], L_p + p["L_in_H"], p["T_K"])
    return ch, r


def _band(r, f0):
    """Detunings (Hz) below and above f0 where the SQUID imprecision, referred to the pickup EMF, equals the
    thermal noise of the pickup loss: the edges of the thermal-noise-limited band."""
    from scipy.optimize import brentq

    def g(x):
        n = r.noise_psd(2 * math.pi * (f0 + x))
        return n["N_V_imprecision_V2_s"] - n["N_V_thermal_V2_s"]
    span = 1e4 * f0 / READOUT["Q"]
    return -brentq(lambda x: g(-x), 0.0, span), brentq(g, 0.0, span)


RO_KEYS = ("N_V_thermal_V2_s", "N_V_backaction_V2_s", "N_V_imprecision_V2_s", "N_V_correlation_V2_s", "N_V_V2_s")


def fig_readout(ctx):
    tor = ctx.toroid
    p = READOUT
    f0 = p["f0_Hz"]
    out = {}
    for k, sign in (("order2", p["sign"]), ("order3", p["sign"]), ("order3_opposite_sign", -p["sign"])):
        src = k.split("_")[0]
        ch, r = _readout(tor[src]["L_p_H"], tor[src]["Psi_SI_H_T_m2"], sign)
        eq = r.equivalent(f_Hz=f0)
        lo, hi = _band(r, f0)
        x = np.linspace(-500.0, 500.0, 2001)
        N = {kk: np.array([r.noise_psd(2 * math.pi * (f0 + xi))[kk] for xi in x]) for kk in RO_KEYS}
        th0 = r.noise_psd(2 * math.pi * f0)["N_V_thermal_V2_s"]
        i = int(np.argmin(N["N_V_V2_s"]))
        out[k] = {"ch": ch, "r": r, "eq": eq, "lo": lo, "hi": hi, "x": x, "N": N, "th0": th0,
                  "x_min": float(x[i]), "N_min_over_thermal": float(N["N_V_V2_s"][i] / th0)}
    o = out["order3"]
    x, N, th0 = o["x"], o["N"], o["th0"]
    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.115, 0.165, 0.855, 0.8])
        ax.semilogy(x, N["N_V_imprecision_V2_s"] / th0, color=BLUE, lw=2.2, zorder=3)
        ax.semilogy(x, N["N_V_thermal_V2_s"] / th0, color=ORANGE, lw=2.2, zorder=3)
        ax.semilogy(x, N["N_V_backaction_V2_s"] / th0, color=VIOLET, lw=2.2, zorder=3)
        ax.semilogy(x, N["N_V_V2_s"] / th0, color=t["ink"], lw=3.0, zorder=4)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0.25, 80)
        ax.set_xticks([-400, -200, 0, 200, 400])
        ax.set_yticks([1, 10], ["1", "10"])
        ax.set_xlabel(r"detuning from resonance, $f - f_0$ [Hz]")
        ax.set_ylabel("noise / thermal noise")
        ax.grid(True, which="major", axis="both")
        lc = RO_LABEL[theme]
        ax.text(-490, 1.12, "thermal (loss at $T$)", ha="left", va="bottom", fontsize=14, color=lc["thermal"])
        ax.text(-490, 0.77, "SQUID backaction", ha="left", va="top", fontsize=14, color=lc["backaction"])
        j = np.searchsorted(x, 485.0)                    # above the right branch, where it runs above the total
        ax.text(x[j], N["N_V_imprecision_V2_s"][j] / th0 * 1.3, "SQUID imprecision", ha="right",
                va="bottom", fontsize=14, color=lc["imprecision"])
        j = np.searchsorted(x, -330.0)
        ax.text(x[j] + 10, N["N_V_V2_s"][j] / th0 * 1.25, "total", ha="left", va="bottom", fontsize=14,
                color=t["ink"])
        files.append(save_card(fig, ctx.out / f"card_readout_{theme}.png"))
    a, b = out["order2"], out["order3"]
    check = [check_row("thermal-limited band [Hz]", a["hi"] - a["lo"], b["hi"] - b["lo"], "toroid order2",
                       "toroid order3"),
             check_row("minimum total / thermal", a["N_min_over_thermal"], b["N_min_over_thermal"], "toroid order2",
                       "toroid order3"),
             check_row("loaded Q", a["eq"]["Q_loaded"], b["eq"]["Q_loaded"], "toroid order2", "toroid order3")]
    m = out["order3_opposite_sign"]
    check.append({"quantity": "correlation sign s (convention): the drawn s against the opposite s",
                  "s_drawn": p["sign"], "s_opposite": -p["sign"],
                  "x_min_Hz": {"drawn": sig(b["x_min"], 3), "opposite": sig(m["x_min"], 3)},
                  "N_min_over_thermal": {"drawn": sig(b["N_min_over_thermal"], 4),
                                         "opposite": sig(m["N_min_over_thermal"], 4)},
                  "band_Hz": {"drawn": [sig(b["lo"], 3), sig(b["hi"], 3)],
                              "opposite": [sig(m["lo"], 3), sig(m["hi"], 3)]},
                  "eta": {"drawn": sig(float(b["ch"].eta(f0)), 4), "opposite": sig(float(m["ch"].eta(f0)), 4)}})
    ch, r, eq = b["ch"], b["r"], b["eq"]
    return {
        "title": "Readout and SQUID chains",
        "files": files,
        "design": {"pickup": "the hero toroid (L_p and Psi of the third-order solve)",
                   "readout": "axionfem.circuit.series_readout: pickup with its loss, tuning capacitor, "
                              "SQUID input coil",
                   "parameters": {k: (sig(v) if not isinstance(v, str) else v) for k, v in p.items()},
                   "squid": "axionfem.squid.chain('C1+low+RT') of arXiv:2504.20398 at its scan-optimal mutual "
                            "inductance",
                   "sign": "s = +1 or -1, the sign of the imprecision-backaction correlation; the source gives "
                           "only its magnitude, so the side of resonance on which the total noise has its "
                           "minimum is a convention (parameters.sign is the s drawn)"},
        "solve": {"kind": "circuit noise model (no discretization); toroid inputs from the hero solve",
                  "plotted": "noise referred to an EMF in the pickup, divided by the thermal noise of the pickup loss"},
        "check": check,
        "numbers": {"f0_Hz": sig(f0), "Q": sig(p["Q"]), "T_K": sig(p["T_K"]), "eta": sig(float(ch.eta(f0)), 4),
                    "M_H": sig(r.squid.M_H, 4), "band_Hz": sig(b["hi"] - b["lo"], 3), "band_lo_Hz": sig(b["lo"], 3),
                    "band_hi_Hz": sig(b["hi"], 3), "linewidth_Hz": sig(f0 / eq["Q_loaded"], 3),
                    "backaction_over_thermal": sig(float(N["N_V_backaction_V2_s"][1000] / th0), 3),
                    "x_min_Hz": sig(b["x_min"], 3), "x_min_opposite_sign_Hz": sig(m["x_min"], 3),
                    "sign": p["sign"], "N_min_over_thermal": sig(b["N_min_over_thermal"], 3)},
    }


# ============================================================================================ mechanics
SOLENOID = {"R": 0.30, "tw": 0.12, "H": 0.48, "NI": 3.2e6}
MECH_VIEW = {"cutaway_deg": 270.0, "elev_deg": 22.0, "azim_deg": -55.0}


def _solenoid_design(size):
    return af.Design({
        "schema": "axionfem.design/1", "name": "solenoid_winding_stress",
        "description": "Illustrative thick solenoid winding loaded by its own Lorentz force (smeared winding "
                       "material).",
        "parameters": dict(SOLENOID),
        "mesh": {"order": 2, "size": size, "surface_size": "tw/30"},
        "regions": [{"name": "winding", "material": "winding",
                     "shape": {"rect": {"r": ["R", "R+tw"], "z": ["-H/2", "H/2"]}}}],
        "windings": [{"name": "coil", "region": "winding", "type": "azimuthal", "ampere_turns": "NI"}],
        "mechanics": {}})


def color_to_alpha(im, bg, edge=2):
    """RGBA image of a renderer crop: the uniform background bg becomes transparent, the object stays opaque, and
    only a band of `edge` px along the object's outline gets the colour-to-alpha treatment (the smallest alpha that
    reproduces the antialiased pixel over bg)."""
    from scipy import ndimage

    a = np.asarray(im.convert("RGB")).astype(float) / 255.0
    b = np.array(bg, float) / 255.0
    obj = np.abs(a - b).max(axis=2) > 2.5 / 255
    core = ndimage.binary_erosion(obj, iterations=edge)
    lim = np.where(a >= b, 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ach = np.where(np.abs(lim - b) > 1e-9, (a - b) / (lim - b), 0.0)
        alpha = np.clip(ach.max(axis=2), 0.0, 1.0)
        rgb = np.where(alpha[..., None] > 1e-6, (a - b) / alpha[..., None] + b, 0.0)
    alpha = np.where(core, 1.0, np.where(obj, alpha, 0.0))
    rgb = np.where(core[..., None], a, rgb)
    out = np.dstack([np.clip(rgb, 0, 1), alpha])
    return Image.fromarray(np.round(out * 255).astype(np.uint8), "RGBA")


def lines_over_ground(scene, opts, colour):
    """The light-theme render of `scene`, with the field lines that cross the open background recoloured.

    The renderer draws static field lines white over a viridis quantity, and in the light theme its background is
    white too, so the lines that leave the cut planes (through the bore) would vanish. Their coverage is measured
    on the dark-theme render, where the background is dark: the background pixels of the render without lines,
    and there the excess of the render with lines over the background. Those pixels of the light render are
    repainted as `colour` with that coverage; everything drawn over a surface is left as the renderer drew it."""
    def rgb(png):
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(float)
    light = rgb(render_png(scene, theme="light", field_lines=True, **opts))
    dark = rgb(render_png(scene, theme="dark", field_lines=True, **opts))
    bare = rgb(render_png(scene, theme="dark", field_lines=False, **opts))
    bg_dark = np.array(hex_rgb(THEME["dark"]["bg"]), float)
    ground = np.abs(bare - bg_dark).max(axis=2) <= 1.0
    cover = np.clip(((dark - bg_dark) / (255.0 - bg_dark)).max(axis=2), 0.0, 1.0) * ground
    c = np.array(hex_rgb(colour), float)
    out = light * (1.0 - cover[..., None]) + c * cover[..., None]
    out = np.where(ground[..., None], out, light)
    return Image.fromarray(np.round(out).astype(np.uint8), "RGB")


def fig_mech(ctx):
    rows = {}
    for size, order in (("0.01", 2), ("0.01", 3)):
        r = _solenoid_design(size).solve(ctx.work / f"mech_solenoid_o{order}", order=order, verbose=0)
        w = r.mechanics["regions"]["winding"]
        rows[order] = {"res": r, "vm_max_Pa": w["von_mises_max_Pa"], "hoop_max_Pa": w["sigma_hoop_Pa"]["max"],
                       "u_max_m": w["u_max_m"], "B_max_T": r.B_max, "dofs": r.data["mesh"]["dofs"],
                       "vm_at": w["von_mises_max_location"]}
    r = rows[3]["res"]
    scene = r.scene(subdiv=2, contours=14)
    V = np.frombuffer(base64.b64decode(scene["field"]["vertices"]["data"]), dtype=np.float32).reshape(-1, 2)
    m = r.sample_mech(V[:, 0].astype(float), V[:, 1].astype(float))
    vm = np.asarray(m["von_mises"], float) / 1e6                    # NaN outside the winding
    vmax = 10.0 * math.ceil(np.nanmax(vm) / 10.0)
    q = dict(scene["field"]["quantities"]["Bmag"])
    scene["field"]["quantities"]["von_mises"] = q
    _set_quantity(scene, "von_mises", vm)
    scene["field"]["units"]["von_mises"] = "MPa"
    scene["field"]["ranges"]["von_mises"] = [0.0, vmax]
    files = []
    for theme in ctx.themes:
        t = style(theme)
        bg = (255, 255, 255) if theme == "light" else (19, 21, 26)
        opts = dict(view="3d", quantity="von_mises", annotate=False, title=" ", width=2400, height=1600,
                    supersample=4, vmin=0.0, vmax=vmax, **MECH_VIEW)
        if theme == "light":
            im = lines_over_ground(scene, opts, t["muted"])
        else:
            im = Image.open(io.BytesIO(render_png(scene, theme=theme, field_lines=True, **opts)))
        obj = color_to_alpha(crop_render(im, bg, margin=0.01), bg)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.0, 0.0, 0.84, 1.0])
        ax.imshow(np.asarray(obj), interpolation="lanczos")
        ax.set_xlim(-0.04 * obj.width, 1.04 * obj.width)
        ax.set_ylim(1.02 * obj.height, -0.02 * obj.height)
        ax.set_aspect("equal")
        ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, vmax))
        cb = colorbar(fig, sm, [0.865, 0.2, 0.016, 0.6], "", t, ticks=np.arange(0, vmax + 1, 10))
        cb.ax.set_title("MPa", fontsize=14, color=t["ink"], pad=8)
        files.append(save_card(fig, ctx.out / f"card_mech_{theme}.png"))
    a, b = rows[2], rows[3]
    check = [check_row("peak von Mises stress [Pa]", a["vm_max_Pa"], b["vm_max_Pa"], "order2", "order3"),
             check_row("peak hoop stress [Pa]", a["hoop_max_Pa"], b["hoop_max_Pa"], "order2", "order3"),
             check_row("peak |B| [T]", a["B_max_T"], b["B_max_T"], "order2", "order3")]
    return {
        "title": "Mechanics",
        "files": files,
        "design": {"doc": "solenoid_winding_stress (built in this script)",
                   "parameters": {k: sig(v) for k, v in SOLENOID.items()},
                   "material": "builtin 'winding' (smeared isotropic impregnated coil: E = 80 GPa, nu = 0.3)",
                   "note": "an illustrative winding, free-standing; loads: its own Lorentz force J x B"},
        "solve": {"kind": "axisymmetric magnetostatics + small-strain elasticity", "order": 3, "dofs": b["dofs"]},
        "render": {"renderer": "C++ (axionfem.render)", "view": "3d", **MECH_VIEW,
                   "quantity": "von Mises stress sampled from the mechanics solution on the scene mesh (MPa)",
                   "lines": "static field lines (contours of psi = r A_phi): white over the cut planes and walls; "
                            "in the light variant, where they cross the open bore, the theme's muted ink",
                   "vmax_MPa": vmax},
        "check": check,
        "numbers": {"R_m": sig(SOLENOID["R"]), "tw_m": sig(SOLENOID["tw"]), "H_m": sig(SOLENOID["H"]),
                    "NI_A": sig(SOLENOID["NI"]), "B_max_T": sig(b["B_max_T"], 3),
                    "vm_max_MPa": sig(b["vm_max_Pa"] / 1e6, 3),
                    "hoop_max_MPa": sig(b["hoop_max_Pa"] / 1e6, 3), "u_max_mm": sig(b["u_max_m"] * 1e3, 3),
                    "vm_max_at_m": {k: sig(v, 3) for k, v in b["vm_at"].items()}},
    }


# ============================================================================================ reach
# Family labels: (family, runs that place it, x factor, ha, y factor, va). The anchor is the geometric mean of those
# runs' masses and of their published and recomputed limits, so every label sits beside its own points and outside
# the QCD band; a family whose runs lie far apart (QUAX) gets one label per run.
REACH_LABELS = [("ADMX", ("ADMX_1A",), 1.12, "left", 0.85, "center"),
                ("CAPP", ("CAPP_12TB",), 1.11, "left", 0.62, "center"),
                ("HAYSTAC", ("HAYSTAC_2021", "HAYSTAC_2023"), 0.88, "right", 1.0, "center"),
                ("TASEH", ("TASEH_2022",), 0.9, "right", 1.0, "center"),
                ("ORGAN", ("ORGAN_Q_2025",), 1.0, "center", 1.5, "bottom"),
                ("QUAX", ("QUAX_2019",), 1.0, "center", 1.5, "bottom"),
                ("QUAX", ("QUAX_2021",), 0.9, "right", 1.0, "center")]
REACH_FAMILY = {"ADMX": BLUE, "HAYSTAC": ORANGE, "CAPP": GREEN, "TASEH": GOLD, "QUAX": VIOLET, "ORGAN": "#d1495b"}


def fig_reach(ctx):
    from axionfem import haloscope as hs

    from axionfem.haloscope_presets import REPRODUCIBLE

    rows = []
    for name in REPRODUCIBLE:                         # runs whose published parameters fix the scan rate
        L = hs.preset_limit(name)
        fam = name.split("_")[0]
        rows.append({"name": name, "family": fam, "label": name.replace("_", " "), "f_Hz": L["f_Hz"],
                     "m_ueV": hs.freq_to_mass_eV(L["f_Hz"]) * 1e6, "g": L["g_GeV"], "g_pub": L["g_published_GeV"],
                     "ratio": L["ratio"]})
    rows.sort(key=lambda r: r["m_ueV"])
    m = np.logspace(np.log10(2.0), np.log10(50.0), 200)
    gk = np.array([abs(hs.g_axion(x * 1e-6, "KSVZ")) for x in m])
    gd = np.array([abs(hs.g_axion(x * 1e-6, "DFSZ")) for x in m])
    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.155, 0.15, 0.805, 0.81])
        ax.fill_between(m, gd, gk, color=t["grid"], lw=0, zorder=0)
        ax.loglog(m, gk, color=t["grey"], lw=1.2, zorder=1)
        ax.loglog(m, gd, color=t["grey"], lw=1.2, ls=(0, (5, 3)), zorder=1)
        for r in rows:
            c = REACH_FAMILY.get(r["family"], t["grey"])
            ax.loglog([r["m_ueV"]], [r["g_pub"]], "o", ms=13, mfc="none", mec=c, mew=2.0, zorder=3)
            ax.loglog([r["m_ueV"]], [r["g"]], "o", ms=6.5, mfc=c, mec=c, zorder=4)
        ax.set_xlim(2, 50)
        ax.set_ylim(1e-16, 3e-12)
        ax.set_xlabel(r"axion mass $m_a$ [$\mu$eV]")
        ax.set_ylabel(r"$|g_{a\gamma\gamma}|$ [GeV$^{-1}$]")
        ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="major")
        k30 = int(np.searchsorted(m, 30.0))
        ax.text(30.0, gk[k30] * 1.5, "KSVZ", color=t["muted"], fontsize=13, va="bottom")
        ax.text(30.0, gd[k30] / 1.5, "DFSZ", color=t["muted"], fontsize=13, va="top")
        by_name = {r["name"]: r for r in rows}
        missing = sorted({r["family"] for r in rows} - {lab[0] for lab in REACH_LABELS})
        if missing:
            raise RuntimeError(f"reach card: no label placement for {missing}; add them to REACH_LABELS")
        for fam, names, dx, ha, fy, va in REACH_LABELS:
            rs = [by_name[n] for n in names if n in by_name]
            if not rs:
                continue
            mm = math.exp(np.mean([math.log(r["m_ueV"]) for r in rs]))
            gg = math.exp(np.mean([math.log(r[k]) for r in rs for k in ("g", "g_pub")]))
            if va == "bottom":                    # above the points: clear the larger marker
                gg = max(max(r["g"], r["g_pub"]) for r in rs)
            ax.text(mm * dx, gg * fy, fam, ha=ha, va=va, fontsize=13.5, color=t["ink"])
        ax.plot([2.45], [1.25e-12], "o", ms=13, mfc="none", mec=t["muted"], mew=2.0)
        ax.plot([2.45], [3.4e-13], "o", ms=6.5, mfc=t["muted"], mec=t["muted"])
        ax.text(2.75, 1.25e-12, "published limit", va="center", fontsize=13.5, color=t["ink"])
        ax.text(2.75, 3.4e-13, "recomputed by axionFEM", va="center", fontsize=13.5, color=t["ink"])
        files.append(save_card(fig, ctx.out / f"card_reach_{theme}.png"))
    ratios = [r["ratio"] for r in rows]
    return {
        "title": "Reach across mass",
        "files": files,
        "design": {"source": "axionfem.haloscope presets: published cavity, magnet, receiver and scan parameters of "
                             "each run; a few are inferred from stated numbers or assumed where a paper does not "
                             "state them (marked INFERRED / ASSUMED in axionfem.haloscope_presets)",
                   "runs": [r["name"] for r in rows]},
        "solve": {"kind": "haloscope sensitivity model (closed form; no FEM)"},
        "check": [{"quantity": "recomputed / published limit", "per_run": {r["name"]: sig(r["ratio"], 3) for r in rows},
                   "note": "no discretization: the comparison with the published limit is the check"}],
        "numbers": {"n_runs": len(rows), "ratio_min": sig(min(ratios), 3), "ratio_max": sig(max(ratios), 3),
                    "rows": [{k: (sig(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]},
    }


# ============================================================================================ fluids (beta)
NATCONV = {"Ra": 1e5, "Pr": 0.71, "meshes": (48, 64), "grading": "cosine (Chebyshev) spacing toward the walls"}


def _cavity_mesh(n):
    """Structured triangle mesh of the unit square, cosine-graded toward the walls; boundary attributes 1 bottom,
    2 right, 3 top, 4 left. Cell (i, j) holds triangles 2(j n + i) = [a, b, c] and 2(j n + i) + 1 = [a, c, d]."""
    from axionfem import fluids

    xs = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
    V = np.array([[x, y] for y in xs for x in xs])
    vid = lambda i, j: j * (n + 1) + i  # noqa: E731
    tri, seg, att = [], [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tri += [[a, b, c], [a, c, d]]
    for i in range(n):
        seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, n), vid(i, n)]]
        att += [1, 3]
    for j in range(n):
        seg += [[vid(n, j), vid(n, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [2, 4]
    m = fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32), np.array(seg, dtype=np.int32),
                                np.array(att, dtype=np.int32))
    return m, xs


def _cavity_sample(sol, xs, X, Y):
    """Temperature and velocity of the solution at points (X, Y) of the structured mesh with nodes xs."""
    n = len(xs) - 1
    x, y = X.ravel(), Y.ravel()
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 1)
    j = np.clip(np.searchsorted(xs, y, side="right") - 1, 0, n - 1)
    s = (x - xs[i]) / (xs[i + 1] - xs[i])
    t = (y - xs[j]) / (xs[j + 1] - xs[j])
    upper = t > s
    el = 2 * (j * n + i) + upper
    ref = np.where(upper[:, None], np.c_[s, t - s], np.c_[s - t, t])   # reference coordinates in [a,b,c] / [a,c,d]
    T = np.empty(x.size)
    U = np.empty((x.size, 2))
    order = np.argsort(el, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(el[order]))[0] + 1, x.size]
    for k0, k1 in zip(bounds[:-1], bounds[1:]):
        idx = order[k0:k1]
        e = np.array([el[idx[0]]], dtype=np.int32)
        r = np.ascontiguousarray(ref[idx])
        T[idx] = np.asarray(sol.temperature(e, r)[0])[0]
        U[idx] = np.asarray(sol.velocity(e, r)[0])[0]
    return T.reshape(X.shape), U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape)


def fig_fluids(ctx):
    from axionfem import fluids

    p = fluids.natconv_cavity_parameters(NATCONV["Ra"], NATCONV["Pr"])
    zero2 = lambda X, Y: np.zeros((np.asarray(X).shape[0], 2))  # noqa: E731
    vbcs = [dict(attribute=a, kind="dirichlet", value=zero2) for a in (1, 2, 3, 4)]
    tbcs = [dict(attribute=4, kind="dirichlet", value=0.5), dict(attribute=2, kind="dirichlet", value=-0.5),
            dict(attribute=1, kind="zero"), dict(attribute=3, kind="zero")]
    rows = {}
    for n in NATCONV["meshes"]:
        m, xs = _cavity_mesh(n)
        sol = fluids.solve_boussinesq(m, p["mu"], p["rho0"], p["cp"], p["k"], p["beta_g"], T_ref=0.0, gravity="-y",
                                      velocity_bcs=vbcs, temperature_bcs=tbcs, line_search=True,
                                      continuation_ra=[1e3, 1e4, 1e5], ra=NATCONV["Ra"])
        dg = sol.diagnostics
        if not dg["converged"]:
            raise RuntimeError(f"natural convection on the {n}x{n} mesh did not converge: {dg.get('failure')}")
        rows[n] = {"sol": sol, "xs": xs, "Nu_hot": -float(sol.surface_flux([4])),   # inward flux: Nu > 0
                   "Nu_cold": float(sol.surface_flux([2])),
                   "dofs": int(dg["dofs_field"]), "newton": int(dg["iterations"]),
                   "residual": float(dg["relative_residual"])}
    nf = NATCONV["meshes"][-1]
    sol, xs = rows[nf]["sol"], rows[nf]["xs"]
    g = np.linspace(0.0, 1.0, 481)
    X, Y = np.meshgrid(g, g)
    T, U, Vv = _cavity_sample(sol, xs, X, Y)
    from scipy.integrate import cumulative_trapezoid

    psi = cumulative_trapezoid(U, g, axis=0, initial=0.0)          # u = d psi/dy, psi = 0 on the walls
    umax = float(np.max(np.hypot(U, Vv)))
    pmin = float(psi.min())
    levels = pmin * np.linspace(0.92, 0.08, 7)
    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.155, 0.04, 0.613, 0.92])
        field_image(ax, T, (0, 1, 0, 1), cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.contour(X, Y, psi, levels=np.sort(levels), colors=[FIELD_LINE], linewidths=1.3, alpha=0.85,
                   linestyles="solid")
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=t["ink"], lw=2.0)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.text(0.125, 0.5, "hot wall", rotation=90, ha="center", va="center", fontsize=14, color=t["ink"])
        fig.text(0.798, 0.5, "cold wall", rotation=270, ha="center", va="center", fontsize=14, color=t["ink"])
        ax.annotate("", xy=(-0.105, 0.04), xytext=(-0.105, 0.24), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=t["ink"], lw=1.6, mutation_scale=14))
        ax.text(-0.085, 0.14, "$g$", fontsize=15, color=t["ink"], va="center", ha="left")
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-0.5, 0.5))
        cb = colorbar(fig, sm, [0.87, 0.2, 0.016, 0.6], "", t, ticks=[-0.5, 0, 0.5])
        cb.ax.set_yticklabels([r"$-0.5$", "0", "0.5"])
        cb.ax.set_title(r"$(T-T_0)/\Delta T$", fontsize=14, color=t["ink"], pad=8)
        files.append(save_card(fig, ctx.out / f"card_fluids_{theme}.png"))
    a, b = rows[NATCONV["meshes"][0]], rows[nf]
    lab = [f"{n}x{n}" for n in NATCONV["meshes"]]
    check = [check_row("Nu (hot wall)", a["Nu_hot"], b["Nu_hot"], lab[0], lab[1]),
             {"quantity": "Nu hot vs cold wall (energy balance)", "Nu_hot": sig(b["Nu_hot"]),
              "Nu_cold": sig(b["Nu_cold"]),
              "rel_diff": sig(rel(b["Nu_hot"], b["Nu_cold"]), 3)}]
    return {
        "title": "Fluids",
        "files": files,
        "design": {"problem": "differentially heated square cavity, de Vahl Davis scaling (natconv_cavity_parameters)",
                   "parameters": {"Ra": NATCONV["Ra"], "Pr": NATCONV["Pr"]}, "mesh": NATCONV["grading"],
                   "boundary": "no slip; left wall T0 + dT/2, right wall T0 - dT/2, top and bottom adiabatic"},
        "solve": {"kind": "steady Boussinesq, Taylor-Hood P2/P1 with P2 temperature, Newton with Ra continuation "
                          "1e3 -> 1e4 -> 1e5, one MPI rank", "mesh": f"{nf}x{nf}", "dofs": b["dofs"],
                  "newton_iterations": b["newton"], "relative_residual": sig(b["residual"], 3)},
        "check": check,
        "numbers": {"Ra": NATCONV["Ra"], "Pr": NATCONV["Pr"], "Nu": sig(b["Nu_hot"], 4),
                    "u_max_alpha_over_L": sig(umax, 3),
                    "streamlines": "contours of the stream function at 8 % ... 92 % of its extremum"},
    }


# ============================================================================================ chemistry (beta)
DISK = {"r0_m": 1e-3, "V": 0.1, "kappa_S_m": 1.0, "Rb_over_r0": 5.0, "beta": 3.0, "order": 2, "meshes": (8, 16)}


def _disk_primary(n):
    from axionfem.chem import CurrentDistribution, meshes

    r0, V, kap = DISK["r0_m"], DISK["V"], DISK["kappa_S_m"]
    cd = CurrentDistribution(meshes.disk_graded(n, r0=r0, Rb=DISK["Rb_over_r0"] * r0, beta=DISK["beta"]),
                             order=DISK["order"], axisymmetric=True)
    cd.set_conductivity({1: kap})
    cd.add_electrode("disk", [1], primary=True, phi_s=V)
    cd.set_exterior_halfspace(4, DISK["Rb_over_r0"] * r0, 12, kap)
    cd.solve(temperature=298.15)
    res = np.asarray(cd.residual_full(0))
    current = float(res[np.asarray(cd.electrode_dofs(0), int)].sum()) * 2 * math.pi
    return cd, current


def fig_chem(ctx):
    import matplotlib.tri as mtri

    r0, V, kap = DISK["r0_m"], DISK["V"], DISK["kappa_S_m"]
    I_newman = 4 * kap * r0 * V
    rows = {}
    for n in DISK["meshes"]:
        cd, I = _disk_primary(n)
        rows[n] = {"cd": cd, "I": I, "err": I / I_newman - 1.0, "dofs": int(cd.num_dofs())}
    cd = rows[DISK["meshes"][-1]]["cd"]
    X = np.asarray(cd.dof_coordinates()) / r0
    phi = np.asarray(cd.phi_dofs()) / V
    tri = mtri.Triangulation(X[:, 0], X[:, 1])
    # current lines: contours of the Stokes stream function psi(r, z) = int_0^r i_z r' dr' (i = -kappa grad phi)
    interp = mtri.CubicTriInterpolator(tri, phi, kind="geom")
    rg = np.linspace(0.0, 4.0, 801)
    zg = np.linspace(0.0, 4.0, 801)
    Rg, Zg = np.meshgrid(rg, zg)
    _, dz = interp.gradient(Rg, Zg)
    iz = -np.ma.filled(dz, np.nan)
    from scipy.integrate import cumulative_trapezoid

    psi = cumulative_trapezoid(np.nan_to_num(iz) * Rg, rg, axis=1, initial=0.0)
    ptot = float(np.nanmax(psi[1, :]))
    levels = ptot * (np.arange(1, 10) / 10.0)
    xg = np.linspace(-3.4, 3.4, 1361)
    zz = np.linspace(0.0, 3.6, 721)
    XX, ZZ = np.meshgrid(xg, zz)
    lin = mtri.LinearTriInterpolator(tri, phi)
    PH = np.ma.filled(lin(np.abs(XX), ZZ), np.nan)
    files = []
    for theme in ctx.themes:
        t = style(theme)
        fig = plt.figure(figsize=CARD_IN)
        ax = fig.add_axes([0.025, 0.0, 0.835, 1.0])
        im = field_image(ax, PH, (xg[0], xg[-1], zz[0], zz[-1]), cmap="viridis", vmin=0, vmax=1)
        for s in (1, -1):
            ax.contour(s * Rg, Zg, psi, levels=levels, colors=["white"], linewidths=1.2, alpha=0.9)
        ax.fill_between([xg[0], xg[-1]], -0.4, 0.0, color=t["faint"], lw=0)
        ax.plot([xg[0], xg[-1]], [0, 0], color=t["muted"], lw=1.5)
        ax.plot([-1, 1], [0, 0], color=ORANGE, lw=7, solid_capstyle="butt")
        ax.text(0, -0.2, "disk electrode", ha="center", va="center", fontsize=13.5, color=t["ink"])
        ax.text(2.2, -0.2, "insulator", ha="center", va="center", fontsize=13.5, color=t["ink"])
        ax.set_xlim(xg[0], xg[-1])
        ax.set_ylim(-0.4, 3.6)
        ax.set_aspect("equal")
        ax.axis("off")
        cb = colorbar(fig, im, [0.9, 0.2, 0.016, 0.6], "", t, ticks=[0, 0.5, 1])
        cb.ax.set_title(r"$\phi_l/V$", fontsize=14, color=t["ink"], pad=8)
        files.append(save_card(fig, ctx.out / f"card_chem_{theme}.png"))
    a, b = rows[DISK["meshes"][0]], rows[DISK["meshes"][-1]]
    lab = [f"n={n}" for n in DISK["meshes"]]
    return {
        "title": "Chemistry",
        "files": files,
        "design": {"problem": "primary current distribution on a disk electrode in an insulating plane (Newman)",
                   "parameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DISK.items()},
                   "mesh": "axionfem.chem.meshes.disk_graded, graded toward the disk edge; half-space exterior map"},
        "solve": {"kind": "axisymmetric electrolyte potential, P2", "dofs": b["dofs"]},
        "check": [check_row("I [A]", a["I"], b["I"], lab[0], lab[1]),
                  {"quantity": "I vs Newman 4 kappa r0 V", "I_newman_A": sig(I_newman), "rel_diff": sig(b["err"], 3)}],
        "numbers": {"I_over_newman_minus_1": sig(b["err"], 3), "lines": "equal steps of current between lines"},
    }


# ============================================================================================ logo, favicon, social
# The icon is the hero in miniature: the quick-start toroid cut open through its axis. The two cut faces carry the
# static field |B| on viridis, falling as 1/r from the inner leg (r = yR) to the outer leg (r = R), y = 0.6 (the
# colours are those of the hero; the bore is drawn narrower than yR so the cut faces read at small sizes); the top
# annulus and the far inner wall give the body; the dash-dot line is the symmetry axis. Units: a 64 x 64 box.
ICON = {"cx": 32.0, "Ro": 28.0, "Ri": 14.0, "k": 0.30, "yt": 17.0, "yb": 52.0, "y": 0.6}
SHEATH, WALL_DARK, WALL_LIGHT = "#8796ac", "#2f3845", "#5d697b"
LOGO_INK = {"light": "#1f2430", "dark": "#e6e8ec"}


def _viridis_hex(v):
    r, g, b, _ = plt.get_cmap("viridis")(float(v))
    return "#{:02x}{:02x}{:02x}".format(round(255 * r), round(255 * g), round(255 * b))


def _slab_stops(n=5):
    """(offset, colour) stops from the inner leg (offset 0) to the outer leg (1): viridis(B/B_max), B ~ 1/r."""
    y = ICON["y"]
    return [(t, _viridis_hex(y / (y + t * (1 - y)))) for t in np.linspace(0.0, 1.0, n)]


def _icon_svg(ink, *, axis=True, sw=1.3, ink_class=False, prefix="axf"):
    c = ICON
    cx, Ro, Ri, k, yt, yb = c["cx"], c["Ro"], c["Ri"], c["k"], c["yt"], c["yb"]
    stops = "".join(f'<stop offset="{o:.3f}" stop-color="{col}"/>' for o, col in _slab_stops())
    stroke = 'class="ink"' if ink_class else f'stroke="{ink}"'
    f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")  # noqa: E731
    parts = [
        "<defs>",
        f'<linearGradient id="{prefix}-l" x1="1" y1="0" x2="0" y2="0">{stops}</linearGradient>',
        f'<linearGradient id="{prefix}-r" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>',
        f'<linearGradient id="{prefix}-w" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{WALL_DARK}"/>'
        f'<stop offset="1" stop-color="{WALL_LIGHT}"/></linearGradient>',
        "</defs>",
        # top annulus of the far half
        f'<path d="M{f(cx - Ro)} {f(yt)}A{f(Ro)} {f(k * Ro)} 0 0 1 {f(cx + Ro)} {f(yt)}L{f(cx + Ri)} {f(yt)}'
        f'A{f(Ri)} {f(k * Ri)} 0 0 0 {f(cx - Ri)} {f(yt)}Z" fill="{SHEATH}"/>',
        # far inner wall of the bore
        f'<path d="M{f(cx - Ri)} {f(yt)}A{f(Ri)} {f(k * Ri)} 0 0 1 {f(cx + Ri)} {f(yt)}L{f(cx + Ri)} {f(yb)}'
        f'A{f(Ri)} {f(k * Ri)} 0 0 0 {f(cx - Ri)} {f(yb)}Z" fill="url(#{prefix}-w)"/>',
        # the two cut faces
        f'<rect x="{f(cx - Ro)}" y="{f(yt)}" width="{f(Ro - Ri)}" height="{f(yb - yt)}" fill="url(#{prefix}-l)"/>',
        f'<rect x="{f(cx + Ri)}" y="{f(yt)}" width="{f(Ro - Ri)}" height="{f(yb - yt)}" fill="url(#{prefix}-r)"/>',
        # outlines
        f'<g fill="none" {stroke} stroke-width="{f(sw)}" stroke-linejoin="round" stroke-linecap="round">',
        f'<path d="M{f(cx - Ro)} {f(yb)}V{f(yt)}A{f(Ro)} {f(k * Ro)} 0 0 1 {f(cx + Ro)} {f(yt)}V{f(yb)}Z"/>',
        f'<path d="M{f(cx - Ri)} {f(yb)}V{f(yt)}A{f(Ri)} {f(k * Ri)} 0 0 1 {f(cx + Ri)} {f(yt)}V{f(yb)}"/>',
        f'<path d="M{f(cx - Ri)} {f(yb)}A{f(Ri)} {f(k * Ri)} 0 0 1 {f(cx + Ri)} {f(yb)}"/>',
        "</g>",
    ]
    if axis:
        a = 'class="ink"' if ink_class else f'stroke="{ink}"'
        parts.append(f'<line x1="{f(cx)}" y1="5" x2="{f(cx)}" y2="59" {a} stroke-width="0.9" '
                     f'stroke-dasharray="6 2.2 1.2 2.2" stroke-linecap="butt"/>')
    return "".join(parts)


def _text_path_d(text, weight, size, x0, y0):
    """SVG path data of `text` in DejaVu Sans (matplotlib's bundled font) with its baseline at (x0, y0)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MPath
    from matplotlib.textpath import TextPath

    tp = TextPath((0, 0), text, size=size, prop=FontProperties(family="DejaVu Sans", weight=weight))
    out = []
    f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")  # noqa: E731
    P = lambda x, y: f"{f(x0 + x)} {f(y0 - y)}"  # noqa: E731
    for verts, code in tp.iter_segments(simplify=False, curves=True):
        v = list(zip(verts[0::2], verts[1::2]))
        if code == MPath.MOVETO:
            out.append("M" + P(*v[0]))
        elif code == MPath.LINETO:
            out.append("L" + P(*v[0]))
        elif code == MPath.CURVE3:
            out.append("Q" + P(*v[0]) + " " + P(*v[1]))
        elif code == MPath.CURVE4:
            out.append("C" + P(*v[0]) + " " + P(*v[1]) + " " + P(*v[2]))
        elif code == MPath.CLOSEPOLY:
            out.append("Z")
    return "".join(out)


def _text_width(text, weight, size):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path

    w, _, _ = text_to_path.get_text_width_height_descent(text, FontProperties(family="DejaVu Sans", weight=weight,
                                                                             size=size), ismath=False)
    return w


def _wordmark_svg(theme):
    ink = LOGO_INK[theme]
    size, x0, base = 38.0, 78.0, 45.0
    w1 = _text_width("axion", "normal", size)
    w2 = _text_width("FEM", "bold", size)
    W = math.ceil(x0 + w1 + w2 + 4)
    d1 = _text_path_d("axion", "normal", size, x0, base)
    d2 = _text_path_d("FEM", "bold", size, x0 + w1 + 0.5, base)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 64" width="{W * 3}" height="192" '
            f'role="img" aria-label="axionFEM"><title>axionFEM</title>{_icon_svg(ink, prefix="axf-" + theme)}'
            f'<path d="{d1}" fill="{ink}"/><path d="{d2}" fill="{ink}"/></svg>\n'), W


def _icon_file_svg(*, axis=True, sw=1.7):
    """The icon as a standalone SVG whose ink follows the viewer's colour scheme."""
    style_ = (f"<style>.ink{{stroke:{LOGO_INK['light']}}}@media (prefers-color-scheme: dark)"
              f"{{.ink{{stroke:{LOGO_INK['dark']}}}}}</style>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" '
            f'aria-label="axionFEM">{style_}{_icon_svg(None, axis=axis, sw=sw, ink_class=True)}</svg>\n')


def _draw_icon(ax, ink, x0=0.0, y0=0.0, s=1.0, axis=True, sw_pt=None):
    """Draw the icon with matplotlib into data coordinates (icon units, y down) offset by (x0, y0), scale s."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath

    c = ICON
    cx, Ro, Ri, k, yt, yb = c["cx"], c["Ro"], c["Ri"], c["k"], c["yt"], c["yb"]
    T = lambda pts: [(x0 + s * x, y0 + s * y) for x, y in pts]  # noqa: E731

    def arc(R, y, a0, a1, n=96):
        th = np.linspace(a0, a1, n)
        return [(cx + R * math.cos(t), y - k * R * math.sin(t)) for t in th]
    sw = sw_pt if sw_pt is not None else 1.3 * s * 0.75
    ann = arc(Ro, yt, math.pi, 0) + arc(Ri, yt, 0, math.pi)
    ax.add_patch(PathPatch(MPath(T(ann), closed=True), fc=SHEATH, ec="none", zorder=1))
    wall = arc(Ri, yt, math.pi, 0) + arc(Ri, yb, 0, math.pi)
    wp = PathPatch(MPath(T(wall), closed=True), fc="none", ec="none", zorder=1)
    ax.add_patch(wp)
    gw = np.linspace(0, 1, 256)[None, :]
    from matplotlib.colors import LinearSegmentedColormap

    cw = LinearSegmentedColormap.from_list("wall", [WALL_DARK, WALL_LIGHT])
    im = ax.imshow(gw, cmap=cw, extent=(x0 + s * (cx - Ri), x0 + s * (cx + Ri), y0 + s * yb, y0 + s * (yt - k * Ri)),
                   aspect="auto", interpolation="bilinear", zorder=1)
    im.set_clip_path(wp)
    cs = LinearSegmentedColormap.from_list("slab", [(o, col) for o, col in _slab_stops()])
    for side in (-1, 1):
        xa, xb = (cx - Ro, cx - Ri) if side < 0 else (cx + Ri, cx + Ro)
        rp = Rectangle((x0 + s * xa, y0 + s * yt), s * (xb - xa), s * (yb - yt), fc="none", ec="none", zorder=2)
        ax.add_patch(rp)
        g = np.linspace(0, 1, 256)[None, :]
        img = g[:, ::-1] if side < 0 else g                # offset 0 (inner leg) next to the bore
        im = ax.imshow(img, cmap=cs, extent=(x0 + s * xa, x0 + s * xb, y0 + s * yb, y0 + s * yt), aspect="auto",
                       interpolation="bilinear", zorder=2)
        im.set_clip_path(rp)
    lw = dict(color=ink, lw=sw, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    out = [(cx - Ro, yb), (cx - Ro, yt)] + arc(Ro, yt, math.pi, 0) + [(cx + Ro, yb), (cx - Ro, yb)]
    ax.plot(*zip(*T(out)), **lw)
    inn = [(cx - Ri, yb), (cx - Ri, yt)] + arc(Ri, yt, math.pi, 0) + [(cx + Ri, yb)]
    ax.plot(*zip(*T(inn)), **lw)
    ax.plot(*zip(*T(arc(Ri, yb, math.pi, 0))), **lw)
    if axis:
        ax.plot(*zip(*T([(cx, 5), (cx, 59)])), color=ink, lw=sw * 0.7, ls=(0, (6, 2.2, 1.2, 2.2)), zorder=4)


def _icon_png(px, ink, path, *, axis=True, pad=0.0, bg=None, sw_scale=1.0):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    m = 64 * pad
    ax.set_xlim(-m, 64 + m)
    ax.set_ylim(64 + m, -m)
    ax.axis("off")
    if bg:
        fig.patch.set_facecolor(bg)
    _draw_icon(ax, ink, axis=axis, sw_pt=1.3 * sw_scale * px / (64 + 2 * m) * 0.72)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=bg is None, facecolor=bg or "none", metadata={"Software": None})
    plt.close(fig)
    Image.open(io.BytesIO(buf.getvalue())).save(path, optimize=True)
    return path


def fig_logo(ctx):
    d = ctx.out.parent / "logo"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    (d / "axionfem_icon.svg").write_text(_icon_file_svg())
    (d / "favicon.svg").write_text(_icon_file_svg(axis=False, sw=2.4))
    files += [d / "axionfem_icon.svg", d / "favicon.svg"]
    for theme in ("light", "dark"):
        svg, W = _wordmark_svg(theme)
        p = d / f"axionfem_wordmark_{theme}.svg"
        p.write_text(svg)
        files.append(p)
    files.append(_icon_png(512, LOGO_INK["light"], d / "axionfem_icon_512.png"))
    files.append(_icon_png(32, LOGO_INK["light"], d / "favicon-32.png", axis=False, sw_scale=1.85))
    files.append(_icon_png(180, LOGO_INK["light"], d / "apple-touch-icon.png", pad=0.12, bg="#ffffff"))
    return {"title": "Logo and favicon", "files": files,
            "design": {"concept": "the quick-start toroid cut open through its axis: the cut faces carry |B| ~ 1/r on "
                                  "viridis (inner leg r = yR to outer leg r = R, y = 0.6)",
                       "slab_stops": [[sig(o, 3), col] for o, col in _slab_stops()],
                       "font": "DejaVu Sans (bundled with matplotlib), converted to paths"},
            "check": [], "numbers": {}}


def fig_social(ctx):
    W, H = SOCIAL_PX
    bg = THEME["dark"]["bg"]
    ink = LOGO_INK["dark"]
    hero = ctx.hero_dark
    a = np.asarray(hero)
    on = np.abs(a.astype(int) - np.array(hex_rgb(bg))).max(axis=2) > 10
    ys, xs = np.nonzero(on)
    obj = hero.crop((int(xs.min()) - 8, int(ys.min()) - 8, int(xs.max()) + 8, int(ys.max()) + 8))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    axh = fig.add_axes([0.505, 0.075, 0.47, 0.85])
    axh.imshow(np.asarray(obj), interpolation="lanczos")
    axh.set_aspect("equal")
    axh.axis("off")
    axl = fig.add_axes([0.065, 0.0, 0.43, 1.0])
    axl.set_xlim(0, 430)
    axl.set_ylim(640, 0)
    axl.axis("off")
    _draw_icon(axl, ink, x0=-4, y0=118, s=2.3, sw_pt=2.2)
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    size = 60
    w1 = _text_width("axion", "normal", size)
    for txt, wt, x in (("axion", "normal", 0.0), ("FEM", "bold", w1 + 0.8)):
        tp = TextPath((0, 0), txt, size=size, prop=FontProperties(family="DejaVu Sans", weight=wt))
        tr = Affine2D().scale(1, -1).translate(x, 350)
        axl.add_patch(PathPatch(tr.transform_path(tp), fc=ink, ec="none"))
    tag = ("Finite-element simulation\nfor axion dark-matter detectors")
    axl.text(2, 395, tag, fontsize=21, color=THEME["dark"]["muted"], va="top", ha="left", linespacing=1.35,
             family="DejaVu Sans", usetex=False)
    p = ctx.out / "social_preview.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=bg, metadata={"Software": None})
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    assert im.size == (W, H), im.size
    im.save(p, optimize=True)
    return {"title": "Social preview", "files": [p],
            "design": {"composition": "dark hero (hero_toroid_dark.png) with the icon, the wordmark and the tagline"},
            "check": [], "numbers": {}}


# ============================================================================================ captions
def fmt(x, digits=3):
    """A number for a caption: d significant digits, x 10^n in Unicode for large/small values."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -3 <= e <= 4:
        dec = max(digits - 1 - e, 0)
        return f"{x:.{dec}f}"
    m = x / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{digits - 1}f} × 10{sup}"


def max_rel(entry):
    """Largest |relative difference| of the convergence rows of a manifest entry."""
    vals = [abs(r[k]) for r in entry.get("check", []) for k in r if k.startswith("rel_diff")]
    return max(vals) if vals else None


def mesh_rel(entry):
    """Largest |relative difference| of the two-discretization rows only (rows with a plain ``rel_diff``)."""
    vals = [abs(r["rel_diff"]) for r in entry.get("check", []) if "rel_diff" in r]
    return max(vals) if vals else None


CAPTIONS = {}


def caption(name):
    def deco(fn):
        CAPTIONS[name] = fn
        return fn
    return deco


def signed(x, fmt_spec="+.0f"):
    """A signed number for a caption, with a typographic minus."""
    return format(x, fmt_spec).replace("-", "\u2212")


def frel(x):
    """A relative difference for a caption, one significant digit (e.g. 4 x 10^-6)."""
    return fmt(abs(x), 1)


@caption("hero")
def _cap_hero(e):
    n = e["numbers"]
    R, H = n["R_m"], n["H_m"]
    dims = f"R = H = {fmt(R, 2)} m" if R == H else f"R = {fmt(R, 2)} m, H = {fmt(H, 2)} m"
    return (f"The toroidal lumped-element detector of the quick start ({dims}, "
            f"NI = {fmt(n['NI_A'], 2)} A turns), solved axisymmetrically and drawn by axionFEM's renderer as a 3D "
            f"cutaway. Colour: the static field |B| on the cut planes (viridis, 0 to {fmt(n['B_max_T'])} T), which "
            f"is confined to the toroid volume; where it vanishes the planes are left uncoloured. Pink: field lines "
            f"of the pickup mode in the bore. Signal coupling \u03a8 = {fmt(n['Psi_SI_H_T_m2'], 4)} H T m\u00b2, "
            f"pickup inductance L_p = {fmt(n['L_p_H'] * 1e6, 4)} \u00b5H, with third-order elements; second-order "
            f"elements agree to {frel(max_rel(e))}.")


@caption("cavity")
def _cap_cavity(e):
    n = e["numbers"]
    f, C = n["f_Hz"], n["C"]
    Cs = " and ".join("0" if abs(c) < 1e-9 else fmt(c, 2) for c in C[1:])
    return (f"Cavity haloscope: E_z of the three lowest TM modes of a copper cavity of {fmt(n['R_mm'], 3)} mm "
            f"radius with a {fmt(n['rod_diameter_mm'], 3)} mm copper tuning rod at an illustrative off-centre "
            f"position (cross-section eigensolver; each mode scaled to its largest |E_z|; dark lines: nodal lines). "
            f"Large: the TM\u2080\u2081\u2080-like mode, f = {fmt(f[0] / 1e6, 4)} MHz, form factor "
            f"C = {fmt(C[0], 3)}, which an independent method-of-particular-solutions calculation reproduces to "
            f"{frel(e['check'][-1]['rel_diff_f'])} in f and {frel(e['check'][-1]['rel_diff_C'])} in C. Small: the "
            f"next two TM modes ({fmt(f[1] / 1e6, 4)} and {fmt(f[2] / 1e6, 4)} MHz); their sign changes give "
            f"C = {Cs}.")


@caption("srf")
def _cap_srf(e):
    n = e["numbers"]
    if e["design"]["builder"].endswith("cylinder_pair_design"):
        cf = next(r for r in e["check"] if "rel_diff_eta" in r)
        p, s = n["pump"], n["signal"]
        return (f"SRF heterodyne cavity: the 60 L closed cylinder of Lasenby (arXiv:1912.11056), radius "
                f"{fmt(n['R_cm'], 2)} cm, with the length {fmt(n['L_cm'], 4)} cm (d/R = {fmt(n['L_over_R'], 4)}) at "
                f"which the {p} pump and the {s} signal are degenerate ({fmt(n['V_L'], 3)} L, "
                f"{fmt(n['f_pump_Hz'] / 1e9, 5)} GHz). Colour: their overlap density E\u2081\xb7B\u2080 on the "
                f"meridional section, in units of the product of the rms fields (axisymmetric eigensolver). Its "
                f"volume average gives the overlap factor \u03b7\u2081\u2080 = {fmt(n['eta_10'], 4)}, so "
                f"\u03b7\u2081\u2080\xb2 = {fmt(n['eta_10_squared'], 3)} against the published "
                f"C\u2080\u2081 = {fmt(n['published_C01'], 2)}; the closed form agrees to {frel(cf['rel_diff_eta'])} "
                f"and two meshes agree to {frel(mesh_rel(e))}.")
    return (f"SRF heterodyne cavity: the Project X 650 MHz \u03b2 = 0.9 elliptical cell with its axial length scaled "
            f"by {fmt(e['design']['axial_factor'], 2)}, which brings the {n['pump']} pump and the {n['signal']} signal "
            f"to {fmt(n['f_pump_Hz'] / 1e9, 5)} GHz, {fmt(n['splitting_Hz'] / 1e3, 3)} kHz apart. Colour: their "
            f"overlap density E\u2081\u00b7B\u2080 on the meridional section, in units of the product of the rms "
            f"fields (axisymmetric eigensolver); green: the sign change of the pump field. Its volume average gives "
            f"the overlap factor "
            f"\u03b7\u2081\u2080 = {fmt(n['eta_10'], 3)}; two meshes agree to {frel(max_rel(e))}.")


@caption("readout")
def _cap_readout(e):
    n = e["numbers"]
    return (f"Readout and SQUID chain: noise of a resonant readout of the quick-start toroid (L_p and \u03a8 of the "
            f"toroid solve in the same run), tuned to f\u2080 = {fmt(n['f0_Hz'] / 1e6, 1)} MHz with "
            f"Q = {fmt(n['Q'], 1)} at {fmt(n['T_K'] * 1e3, 2)} mK (illustrative values) and read by the C1+low+RT dc "
            f"SQUID chain of Ankel et al. (arXiv:2504.20398) at its scan-optimal coupling; the chain's noise "
            f"parameter, its minimum noise temperature in units of \u0127\u03c9/2k_B, is "
            f"\u03b7 = {fmt(n['eta'], 3)}. Each "
            f"noise is referred to an EMF in the pickup and divided by the thermal noise of the pickup loss. SQUID "
            f"backaction adds {fmt(n['backaction_over_thermal'], 2)} of it; the SQUID imprecision stays below it "
            f"over {fmt(n['band_Hz'], 3)} Hz, about {fmt(n['band_Hz'] / n['linewidth_Hz'], 3)} linewidths. The "
            f"correlation between imprecision and backaction moves the minimum of the total "
            f"{abs(n['x_min_Hz']):.0f} Hz off resonance. The source gives the size of that correlation but not its "
            f"sign s, so the side is a convention: the card shows s = {signed(n['sign'])} (minimum at "
            f"{signed(n['x_min_Hz'])} Hz); s = {signed(-n['sign'])} puts it at "
            f"{signed(n['x_min_opposite_sign_Hz'])} Hz.")


@caption("mech")
def _cap_mech(e):
    n = e["numbers"]
    return (f"Mechanics: an illustrative solenoid winding (inner radius {fmt(n['R_m'], 2)} m, "
            f"{fmt(n['tw_m'] * 100, 2)} cm thick, {fmt(n['H_m'], 2)} m tall, {fmt(n['NI_A'], 2)} A turns, peak field "
            f"{fmt(n['B_max_T'], 3)} T) "
            f"loaded by its own Lorentz force. Colour: von Mises stress from the axisymmetric elasticity solve, drawn "
            f"by axionFEM's renderer on the cut planes of a 3D cutaway; lines: static field lines. The stress peaks "
            f"at the inner midplane, {fmt(n['vm_max_MPa'], 3)} MPa (hoop stress {fmt(n['hoop_max_MPa'], 3)} MPa); "
            f"third- and second-order elements agree to {frel(max_rel(e))}. The same structural model takes "
            f"cooldown contraction of bonded parts, and a steady thermal model with k(T) gives heat loads.")


@caption("reach")
def _cap_reach(e):
    n = e["numbers"]
    fams = ", ".join(dict.fromkeys(r["family"] for r in n["rows"]))     # in order of mass
    return (f"Reach across mass: exclusion limits of {n['n_runs']} published cavity-haloscope runs ({fams}), "
            f"recomputed by axionfem.haloscope (filled) from each run's published cavity, magnet, receiver and "
            f"scan parameters, with a few values inferred or assumed where a paper does not state them (listed in "
            f"axionfem.haloscope_presets), next to the published limits (open). The recomputed limits lie between "
            f"{fmt(n['ratio_min'], 2)} and {fmt(n['ratio_max'], 3)} times the published ones. Band: QCD axion "
            f"models between the DFSZ and KSVZ couplings.")


@caption("fluids")
def _cap_fluids(e):
    n = e["numbers"]
    return (f"Fluids: natural convection in a differentially heated square cavity, Ra = {fmt(n['Ra'], 1)}, "
            f"Pr = {fmt(n['Pr'], 2)} (steady Boussinesq, Taylor\u2013Hood elements, Newton with Rayleigh-number "
            f"continuation, one MPI rank). Colour: temperature; lines: streamlines (contours of the stream "
            f"function). Average Nusselt number on the hot wall Nu = {fmt(n['Nu'], 4)}; two meshes agree to "
            f"{frel(e['check'][0]['rel_diff'])}.")


@caption("chem")
def _cap_chem(e):
    n = e["numbers"]
    return (f"Chemistry: primary current distribution on a disk electrode in an insulating plane (axisymmetric "
            f"electrolyte potential with a half-space exterior). Colour: electrolyte potential \u03c6_l/V; white: "
            f"current lines, equal current between neighbours, crowding at the edge of the disk where the current "
            f"density is singular. The total current matches Newman's closed form 4\u03bar\u2080V to "
            f"{fmt(abs(n['I_over_newman_minus_1']), 2)}.")     # two digits (one would round e.g. 1.37e-4 to 1e-4)


def write_captions(man, path):
    lines = ["<!-- Generated by docs/images/make_homepage_figures.py from manifest.json; do not edit. -->", ""]
    for name in FIGURES:
        e = man["figures"].get(name)
        if not e or name not in CAPTIONS:
            continue
        text = CAPTIONS[name](e)
        if e.get("status") == "beta":
            text = "**Beta.** " + text
        lines += [f"<!-- --8<-- [start:{name}] -->", text, f"<!-- --8<-- [end:{name}] -->", ""]
    path.write_text("\n".join(lines))


# ============================================================================================ driver
STATUS = {
    # Release status of the feature each figure shows: "validated" or "beta"; a beta caption starts with "Beta.".
    # Fluids and chemistry are beta features, and the modules behind the cavity (cavity, cavity_xsec), SRF (srf),
    # readout (squid) and reach (haloscope_presets) figures are not declared validated, so those figures are beta.
    "hero": "validated", "cavity": "beta", "srf": "beta", "readout": "beta", "mech": "validated",
    "reach": "beta", "fluids": "beta", "chem": "beta",
}


class Ctx:
    def __init__(self, out, work, themes):
        self.out, self.work, self.themes = out, work, themes
        self.toroid = None
        self.hero_dark = None


def checkout_dirty(*outputs):
    """True when tracked files other than generated ones (this script's default output folders and `outputs`) have
    uncommitted changes, i.e. when the outputs match no committed source; None outside a git checkout."""
    import subprocess

    excl = []
    for p in (HERE / "home", HERE / "logo") + outputs:
        p = Path(p).resolve()
        if p.is_relative_to(ROOT):
            excl.append(f":(exclude){p.relative_to(ROOT).as_posix()}")
    try:
        return subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--", "."]
                              + excl, capture_output=True, text=True, check=True).stdout.strip() != ""
    except Exception:  # noqa: BLE001 - not a checkout
        return None


def check(out):
    man = json.loads((out / "manifest.json").read_text())
    bad = []
    if man.get("axionfem_version") != af.__version__:
        bad.append(f"manifest version {man.get('axionfem_version')} != axionfem {af.__version__}")
    for name, e in man["figures"].items():
        for f in e.get("files", []):
            if not (out / f).exists():
                bad.append(f"{name}: missing {f}")
    if not (out / "captions.md").exists():
        bad.append("missing captions.md")
    for b in bad:
        print("FAIL", b)
    print("check:", "ok" if not bad else f"{len(bad)} problem(s)")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "home"))
    ap.add_argument("--work", default="runs/homepage")
    ap.add_argument("--only", default=",".join(FIGURES))
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--skip-unavailable", action="store_true",
                    help="skip, with a message, figures whose module this build lacks (fluids needs an MPI build)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.check:
        return check(out)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(out, work, [t for t in a.themes.split(",") if t])
    only = [n for n in a.only.split(",") if n]
    unknown = sorted(set(only) - set(FIGURES))
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    mpath = out / "manifest.json"
    # figures not asked for keep their entries from an existing manifest; the top level is written afresh
    figs = json.loads(mpath.read_text()).get("figures", {}) if mpath.exists() else {}
    bi = build_info()
    provenance = {"axionfem_version": af.__version__, "generator_sha256": script_sha256()}
    dirty = checkout_dirty(out, out.parent / "logo")
    if ("readout" in only or "social" in only) and "hero" not in only:
        only = ["hero"] + only              # the readout card and the social preview reuse the hero solve
    for name in FIGURES:
        if name not in only:
            continue
        if name == "fluids" and not bi.get("fluids"):
            msg = "fluids: this build has no fluids module (needs an MPI build)"
            if a.skip_unavailable:
                print("skip", msg)
                continue
            raise SystemExit(msg + "; use --skip-unavailable to skip it")
        t0 = time.time()
        e = globals()[f"fig_{name}"](ctx)
        e["seconds"] = round(time.time() - t0, 1)
        if name in STATUS:
            e["status"] = STATUS[name]
        e["generated_with"] = dict(provenance, checkout_dirty=dirty)
        e["files"] = [Path(os.path.relpath(Path(f).resolve(), out.resolve())).as_posix() for f in e["files"]]
        figs[name] = e
        sizes = ", ".join(f"{Path(f).name} {png_kb(out / f)} KB" for f in e["files"] if f.endswith(".png"))
        print(f"{name}: {e['seconds']} s; {sizes}", flush=True)
    man = {"schema": "axionfem.homepage_figures/1", "axionfem_version": af.__version__, "build_info": bi,
           "checkout_dirty": dirty, "generator": "docs/images/make_homepage_figures.py",
           "generator_sha256": provenance["generator_sha256"],
           "figures": {k: figs[k] for k in FIGURES if k in figs}}
    mpath.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    write_captions(man, out / "captions.md")
    print("wrote", mpath, "and", out / "captions.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
