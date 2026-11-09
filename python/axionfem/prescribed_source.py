"""Prescribed finite-pack racetrack source for 3D full-wave quadrature.  Beta (3D prescribed sources).

The segment kernel is exact for straight filaments. This generator approximates
curved arcs by polygons and a continuous rectangular pack by Gauss integration;
both resolutions must converge before calling its finite-pack field resolved.
It does not include individual insulated turns, joints, leads, screening or iron.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _contour(a, b, h, corner, arc_segments):
    """Clockwise in (r,z), yielding +B_phi inside a meridional loop."""
    centers = ((b-corner, -h/2+corner, -math.pi/2),
               (b-corner, h/2-corner, 0),
               (a+corner, h/2-corner, math.pi/2),
               (a+corner, -h/2+corner, math.pi))
    points = []
    for r, z, angle in centers:
        theta = np.linspace(angle, angle+math.pi/2, arc_segments+1)
        points.extend(np.column_stack((r+corner*np.cos(theta), z+corner*np.sin(theta))))
    # Consecutive arc endpoints are connected by one exact straight segment.
    return np.asarray(points, dtype=float)[::-1]


def racetrack_source(detector_spec: Mapping, *, quadrature_order=3,
                     arc_segments_per_quarter=12):
    """Return a source object for a set of racetrack coils (beta).

    ``detector_spec`` is a dict ``{"coils": {"list": [coil, ...]}}``; each coil supplies ``phi_rad``,
    ``inner_leg_r``, ``outer_leg_r``, ``height``, ``corner_radius``, ``pack_normal``, ``pack_azimuthal`` and
    ``ampere_turns``. Lengths are metres. The coils are used exactly as given (no scaling is applied).
    The output can be assigned to ``fullwave.source`` or written to a JSON file.
    """
    order = _integer(quadrature_order, "quadrature_order")
    arcs = _integer(arc_segments_per_quarter, "arc_segments_per_quarter")
    if not isinstance(detector_spec, Mapping) or "coils" not in detector_spec:
        raise ValueError("detector spec must contain the renderer's coils object")
    data = detector_spec["coils"]
    coils = data.get("list", [])
    if not isinstance(coils, list) or not coils:
        raise ValueError("coils.list must be a nonempty explicit list of finite packs")
    if "N" in data and _integer(data["N"], "coils.N") != len(coils):
        raise ValueError("coils.N must match the number of explicit coils.list entries")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    blocks = []
    for index, coil in enumerate(coils):
        prefix = f"coils.list[{index}]"
        a, b, h, corner, normal, tangent = (
            _positive(coil[name], f"{prefix}.{name}") for name in
            ("inner_leg_r", "outer_leg_r", "height", "corner_radius", "pack_normal", "pack_azimuthal"))
        current, phi = float(coil["ampere_turns"]), float(coil["phi_rad"])
        if not math.isfinite(current) or not math.isfinite(phi):
            raise ValueError(f"{prefix}: ampere_turns and phi_rad must be finite")
        if b <= a or 2*corner >= min(b-a, h):
            raise ValueError(f"{prefix}: rounded rectangle requires outer > inner and 2*corner < min(width,height)")
        if normal/2 >= min(a, corner):
            raise ValueError(f"{prefix}: pack_normal/2 must be less than inner radius and corner radius")
        cosine, sine = math.cos(phi), math.sin(phi)
        for node_u, weight_u in zip(nodes, weights/2):
            u = node_u*normal/2
            rz = _contour(a-u, b+u, h+2*u, corner+u, arcs)
            for node_v, weight_v in zip(nodes, weights/2):
                v = node_v*tangent/2
                xyz = np.column_stack((rz[:, 0]*cosine-v*sine,
                                       rz[:, 0]*sine+v*cosine, rz[:, 1]))
                blocks.append(np.column_stack((xyz, np.roll(xyz, -1, axis=0),
                                               np.full(len(xyz), current*weight_u*weight_v))))
    segments = np.concatenate(blocks)
    return {
        "kind": "segments",
        "segments_m_A": segments.tolist(),
        "metadata": {
            "generator": "axionfem.prescribed_source.racetrack_source",
            "coil_count": len(coils),
            "cross_section_quadrature_order": order,
            "arc_segments_per_quarter": arcs,
            "pack_model": "continuous uniform rectangular current pack; individual turns and leads unresolved",
            "static_state": "prescribed vacuum field; field-cooled frozen-field approximation",
            "numerical_status": "source quadrature and polygon arcs must be converged separately from the RF mesh",
            "units": "endpoints m; filament currents A; field T",
        },
    }
