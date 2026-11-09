"""PNG snapshots of solved designs (C++ software rasterizer; no GPU or browser needed).

    png = render_png(scene, view="section", quantity="Bmag")   # bytes
    save_png(res.scene(), "core.png", view="3d")
"""
from __future__ import annotations

import json
from pathlib import Path

from . import _core


def render_png(scene: dict | str | Path, **options) -> bytes:
    """Render a scene (dict or path to scene.json).  Options: view ('section' | '3d'),
    quantity (Bmag, Br, Bz, Bphi, psi, pickup_Bmag, pickup_psi, none), log_scale, width, height,
    field_lines, annotate, cutaway_deg, elev_deg, azim_deg, theme ('light' | 'dark'), vmin, vmax, title."""
    if isinstance(scene, (str, Path)):
        scene = json.loads(Path(scene).read_text())
    return _core.render_png(json.dumps(scene), json.dumps(options))


def save_png(scene: dict | str | Path, path: str | Path, **options) -> Path:
    p = Path(path)
    p.write_bytes(render_png(scene, **options))
    return p
