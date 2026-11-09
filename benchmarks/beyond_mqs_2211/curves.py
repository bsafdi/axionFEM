"""Read the curves extracted from arXiv:2211.00008 (paper_curves/*.json) and redraw them.

``load(stem)`` returns the figure record; ``draw_panel(ax, panel)`` redraws one panel's curves and regions in data
coordinates with the paper's styles (used to verify the extraction against the original and as the background of
the axionFEM overlays).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURVES = HERE / "paper_curves"


def load(stem: str) -> dict:
    return json.loads((CURVES / f"{stem}.json").read_text())


def labelled(panel: dict, label: str) -> list[dict]:
    return [c for c in panel["curves"] if c.get("label") == label]


def to_gray(color, *, fill: bool = False):
    """Grey shade for a paper colour (overlay backgrounds: paper curves grey, axionFEM in maroon/black).
    Distinct paper colours stay distinguishable: black -> 0.35, maroon -> 0.6, mid grey -> 0.75, others by luminance."""
    if color is None:
        return "0.88" if fill else "0.5"
    if isinstance(color, str):
        from matplotlib.colors import to_rgb
        color = to_rgb(color)
    r, g, b = (list(color) + [0, 0, 0])[:3]
    if fill:
        return "0.88"
    if max(r, g, b) < 0.05:
        return "0.35"
    if r > 0.4 and g < 0.1 and b < 0.1:
        return "0.6"
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    return f"{min(0.85, 0.55 + 0.4 * lum):.2f}"


def draw_panel(ax, panel: dict, *, alpha: float = 1.0, lw_scale: float = 0.5, regions: bool = True,
               only_labelled: bool = False, gray: bool = False):
    if regions:
        for r in panel["regions"]:
            fc = to_gray(r["style"]["fill"], fill=True) if gray else (r["style"]["fill"] or (0.8, 0.8, 0.8))
            op = r["style"].get("fill_opacity") or 1.0
            ax.fill(r["x"], r["y"], color=fc, alpha=alpha * op, lw=0, zorder=0)
        for b in panel.get("shaded_bands", []):
            ax.axvspan(*b["x"], color="0.88", lw=0, zorder=0)
        seen = set()
        for b in panel.get("hatch_bands", []):
            key = tuple(round(v, 6) for v in b["x"])
            if key not in seen:
                seen.add(key)
                ax.axvspan(*b["x"], facecolor="none", edgecolor=to_gray(b["color"]) if gray else b["color"], hatch="xx", lw=0, alpha=0.5 * alpha, zorder=0)
    for c in panel["curves"]:
        if only_labelled and not c.get("label"):
            continue
        st = c["style"]
        ls = "-"
        if st.get("dashes") and st["dashes"].strip().startswith("[") and st["dashes"].strip() != "[] 0":
            nums = [float(v) for v in st["dashes"].strip().split("]")[0].strip("[ ").split()]
            if nums:
                ls = (0, [v / max(st["width"] or 1, 0.5) for v in nums])
        ax.plot(c["x"], c["y"], color=to_gray(st["color"]) if gray else st["color"], lw=(st["width"] or 1) * lw_scale, ls=ls, alpha=alpha, zorder=2)
    ax.set_xscale(panel["x"]["scale"])
    ax.set_yscale(panel["y"]["scale"])
    ax.set_xlim(panel["xlim"])
    ax.set_ylim(panel["ylim"])
