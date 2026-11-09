"""A closed, rounded niobium box with a full-circumference slit in its outer wall, read out through all or part of it.

The geometry is a CORE-type pickup, a toroidal cavity whose readout gap runs around its outer wall (as in DMRadio-Core,
arXiv:2604.16602): the wall mid-surface is the box a < r < b, |z| < h/2 with rounded corners, cut at the outer-wall
midplane by a slit with rounded lips. The wall thickness and the slit opening are both `wall`, and the corner radius
is at least 2 x wall. The coil and any iron are left out. benchmarks/fullwave_indicator.py and
benchmarks/fullwave_surface_losses.py use the small box `mini` (r = 0.3-0.9 m, h = 0.8 m, wall and slit 0.1 m).

Configurations: c360 (a uniform, 360-degree feed) and n1 (one 30-degree port; the rest of the slit is an open gap).
`design` gives the 3D design, `design2d` the uniform feed as an axisymmetric design (the 2D twin).
"""
from __future__ import annotations

import math

# wall mid-surface [m]: inner and outer radius, height, corner radius
BOXES = {
    "mini": {"a": 0.3, "b": 0.9, "h": 0.8, "corner": 0.1},
}
CONFIGS = {"c360": None, "n1": 30.0}


def _rect(r0, r1, z0, z1, fillet=None):
    s = {"rect": {"r": [r0, r1], "z": [z0, z1]}}
    if fillet is not None:
        s["rect"]["fillet"] = fillet
    return s


def profiles(box: dict, wall: float) -> tuple[dict, dict, float]:
    """The slotted wall and the port profile (r-z) for wall thickness and slit opening d = wall."""
    a, b, h, d = box["a"], box["b"], box["h"], wall
    rho = max(box["corner"], 2 * d)
    gs = 2 * d                                  # slot height between the lip centres; opening gs - d = d
    outer = _rect(a - d / 2, b + d / 2, -(h + d) / 2, (h + d) / 2, rho + d / 2)
    inner = _rect(a + d / 2, b - d / 2, -(h - d) / 2, (h - d) / 2, rho - d / 2)
    shape = {"union": [{"difference": [outer, inner, _rect(b - d, b + d, -gs / 2, gs / 2)]},
                       {"disk": {"center": [b, gs / 2], "radius": d / 2}},
                       {"disk": {"center": [b, -gs / 2], "radius": d / 2}}]}
    port = _rect(b - d / 4, b + d / 4, -gs / 2, gs / 2)
    return shape, port, rho


def design(box_name: str, config: str, wall: float, geometry_order: int | None = None) -> dict:
    """The 3D design of box `box_name` fed uniformly (c360) or across one 30-degree port (n1)."""
    box = BOXES[box_name]
    shape, port, rho = profiles(box, wall)
    width = CONFIGS[config]
    slit = wall / 2   # surface size of the slit (port and gap) regions
    regions = []
    if width is None:
        regions.append({"name": "port", "material": "vacuum", "solid": {"revolve": {"profile": port}},
                        "surface_size": slit})
    else:
        regions.append({"name": "port", "material": "vacuum",
                        "solid": {"revolve": {"profile": port, "angles": [-width / 2, width / 2]}},
                        "surface_size": slit, "notes": f"readout port: {width:g} degrees of the slit"})
        regions.append({"name": "gap", "material": "vacuum",
                        "solid": {"revolve": {"profile": port, "angles": [width / 2, 360 - width / 2]}},
                        "surface_size": slit, "notes": "open, undriven part of the slit"})
    coarse = min(box["b"] - box["a"], box["h"]) / 6      # the cavity field (1/r) is smooth on the box scale
    # a thin wall's two faces are triangulated separately, and chords whose sagitta h^2/(8 R) approaches the wall
    # thickness cross (HXT "segment and facet intersect"), with R a fillet wider than an element or the inner wall's
    # revolution radius: h <= sqrt(R d) keeps the sagitta at d/8
    if rho > coarse:
        coarse = min(coarse, math.sqrt(rho * wall))
    coarse = min(coarse, math.sqrt((box["a"] - wall / 2) * wall))
    regions.append({"name": "wall", "material": "nb", "solid": {"revolve": {"profile": shape}},
                    "surface_size": coarse})
    return {
        "schema": "axionfem.design/1",
        "name": f"slotted_box_{box_name}_{config}_wall{wall * 100:g}cm",
        "description": (f"CORE-type slotted pickup box ({box_name}): closed rounded Nb wall "
                        f"(modelled {wall * 100:g} cm, corner {rho:.3f} m) with a full-circumference outer-wall slit, "
                        + ("fed uniformly (360-degree port)" if width is None else
                           f"read out across {width:g} degrees, the rest of the slit open")
                        + "; no coil, no iron (benchmarks/slotted_box.py)"),
        "symmetry": "3d",
        "regions": regions,
        "superconductors": [{"body": "wall", "azimuthal": {"net_current": 0}}],
        "pickup": {"type": "port", "port": "port", "toward": [0.0, 0.0, 1.0]},   # the current crosses the slit along z
        "mesh": {"order": 2, "size": max(box["h"], box["b"] - box["a"]) / 6, "surface_size": wall / 2,
                 # 1: straight-sided elements (order-2 fields)
                 **({"geometry_order": geometry_order} if geometry_order is not None else {})},
    }


def design2d(box_name: str, wall: float, surface: float) -> dict:
    """The uniform-feed box (c360) as an axisymmetric design, at wall-surface size `surface`: the 2D twin of the 3D
    design. 2D needs a magnet: a weak coil outside the box (vacuum kind, no effect on the pickup)."""
    box = BOXES[box_name]
    shape, port, rho = profiles(box, wall)
    a, b, h = box["a"], box["b"], box["h"]
    return {
        "schema": "axionfem.design/1",
        "name": f"slotted_box_{box_name}_2d_wall{wall * 100:g}cm",
        "description": (f"CORE-type slotted pickup box ({box_name}), uniform feed, axisymmetric: the 2D twin of "
                        f"the 3D c360 design (benchmarks/slotted_box.py)"),
        "regions": [{"name": "port", "material": "vacuum", "shape": port},
                    {"name": "wall", "material": "nb", "shape": shape},
                    {"name": "coil", "material": "winding", "shape": _rect(b + 1.0, b + 1.1, -0.05, 0.05)}],
        "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
        "pickup": {"type": "poloidal", "port": "port"},
        "mesh": {"order": 2, "size": max(h, b - a) / 6, "surface_size": surface},
    }
