"""SRF (heterodyne) cavities: elliptical-cell and cylinder mode pairs, degeneracy tuning and figures of merit.

Beta (docs/srf/README.md). Geometry and eigenmode side of the SRF heterodyne sensitivity model
(:mod:`axionfem.heterodyne`): a pump mode B0 (frequency omega_0) and a signal mode E1 made degenerate, with

- xi = omega_0 V^(1/3) / c                         (omega_0 = xi V^(-1/3) in natural units),
- eta_10 = |int E1* . B0 dV| / sqrt(int |E1|^2 dV int |B0|^2 dV)   (signal E, pump B),
- B0 = spatial rms of the pump |B| over the cavity volume V (B0^2 = V^-1 int |B|^2),
- B_pk / B0 (largest |B| of the pump on the wall; also reported: |B| at the equator, B_equator_over_B0), E_pk / (c B0) (largest |E| of the pump on the wall; "E_pk in
  units of B0" with c = 1),
- A / V^(2/3) (wall area including beam-tube walls and their end caps),
- G = omega mu0 int |H|^2 dV / oint |H_t|^2 dA = Z0 (omega/c) int |B|^2 dV / oint |B_t|^2 dA (all walls): the
  dimensionless geometry factor (c = 1) times Z0, so that G is in ohm and Q = G / R_s.

Axisymmetric (m = 0) modes come from the 2D cavity solver (``fullwave`` + ``cavity`` sections); 3D cross-checks from
the 3D cavity solver (``elliptical_cell_design_3d``). Closed forms for cylinders: ``cylinder_pair``.
Lengths are in metres. The default cell (Project X 650 MHz beta = 0.9 regular cell) is ``PROJECTX_650_B09``; its
default axial factor 0.86 is the value at which the TM020 pump and the TE011 signal of the cell with 50 mm x 100 mm
beam tubes are degenerate, rounded (:func:`tune_degeneracy` gives 0.86003 on the faceted wall; with the smooth wall
the pair is split by 20 kHz, 1.4e-5 f0, at 0.86; docs/srf/README.md, validation table).
"""
from __future__ import annotations

import math
from typing import Any

from .cavity import C_LIGHT, MU0, Z0, _design

__all__ = ["PROJECTX_650_B09", "elliptical_cell_contour", "contour_points", "azimuthal_wall_peaks", "elliptical_cell_design", "elliptical_cell_design_3d",
           "classify_modes", "find_mode", "figures_of_merit", "figures_of_merit_3d", "tune_degeneracy",
           "cylinder_mode_fields", "cylinder_pair", "cylinder_pair_design", "cylinder_pair_geometry", "eta_cylinder", "CYLINDER_PAIRS"]

# Project X 650 MHz beta = 0.9 regular cell [m] (S. Holmes et al., Project X reference design; INSPIRE Holmes:2013vpa)
PROJECTX_650_B09 = {"iris_radius": 0.050, "equator_radius": 0.2003, "half_length": 0.1038,
                    "iris_axes": (0.018, 0.038), "equator_axes": (0.0825, 0.084)}


# ------------------------------------------------------------------------------------------------ geometry
def _half_cell(Ri, Req, L, a, b, A, B):
    """Half-cell profile from the iris plane (z = 0, r = Ri) to the equator plane (z = L, r = Req).
    Iris ellipse: centre (0, Ri + b), axial semi-axis a, radial b (its interior is metal: the iris nose);
    equator ellipse: centre (L, Req - B), axial A, radial B (its interior is cavity). The two arcs are joined by
    their internal common tangent (the line has the iris ellipse on one side and the equator ellipse on the
    other). Returns the parametrisations and the tangency parameters (t*, p*)."""
    import numpy as np
    from scipy.optimize import brentq

    ci, ce = np.array([0.0, Ri + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a * np.sin(t), -b * np.cos(t)])  # noqa: E731  t = 0: iris tip (0, Ri)
    T1 = lambda t: np.array([a * np.cos(t), b * np.sin(t)])  # noqa: E731
    P2 = lambda p: ce + np.array([-A * np.cos(p), B * np.sin(p)])  # noqa: E731  p = pi/2: equator (L, Req)
    ptan = lambda d: math.atan2(B * d[0], A * d[1])  # noqa: E731  point of ellipse 2 with tangent || d

    def F(t):
        d = T1(t)
        w = P2(ptan(d)) - P1(t)
        return d[0] * w[1] - d[1] * w[0]

    ts = np.linspace(1e-6, math.pi - 1e-6, 4001)
    v = np.array([F(t) for t in ts])
    cands = []
    for i in np.nonzero(v[:-1] * v[1:] < 0)[0]:
        t = brentq(F, ts[i], ts[i + 1], xtol=1e-15)
        p = ptan(T1(t))
        q1, q2, d = P1(t), P2(p), T1(t)
        side = lambda x: d[0] * (x[1] - q1[1]) - d[1] * (x[0] - q1[0])  # noqa: E731
        # internal tangent: the two centres on opposite sides; arcs inside the half-cell; the line rises
        if side(ci) * side(ce) < 0 and -math.pi / 2 < p < math.pi / 2 and 0 <= q1[0] <= q2[0] <= L and q2[1] > q1[1]:
            cands.append((t, p))
    if len(cands) != 1:
        raise ValueError(f"elliptical cell: expected one internal common tangent of the iris and equator ellipses, "
                         f"found {len(cands)} (Ri={Ri}, Req={Req}, L={L}, iris axes ({a}, {b}), equator axes "
                         f"({A}, {B})); the ellipses overlap or the parameters do not describe a cell")
    return P1, P2, cands[0]


def contour_points(shape: dict, n_arc: int = 400):
    """Points (r, z) along a ``contour`` shape (start, then every segment; ``n_arc`` points per elliptical arc),
    without the closing segment back to the start. A numpy array (n, 2)."""
    import numpy as np

    c = shape["contour"] if "contour" in shape else shape
    out = [tuple(c["start"])]
    for sg in c["segments"]:
        if "line" in sg:
            out.append(tuple(sg["line"]))
        else:
            e = sg["ellipse"]
            tt = np.linspace(e["t"][0], e["t"][1], n_arc + 1)[1:]
            out += list(zip(e["center"][0] + e["axes"][0] * np.cos(tt), e["center"][1] + e["axes"][1] * np.sin(tt)))
    return np.array(out, float)


def _contour_volume_area(shape: dict) -> tuple[float, float]:
    """Volume of revolution (signed, the closing segment included) and wall area (the closing segment excluded: it is
    the axis for the cell) of a ``contour`` shape, exact for lines and to ~1e-15 for the arcs (64-point Gauss-Legendre
    per arc piece of at most pi/8)."""
    import numpy as np

    c = shape["contour"] if "contour" in shape else shape
    xg, wg = np.polynomial.legendre.leggauss(64)
    V = A = 0.0
    cur = tuple(c["start"])

    def line(p0, p1):
        (r0, z0), (r1, z1) = p0, p1
        return (math.pi * (r1 * r1 + r1 * r0 + r0 * r0) / 3 * (z1 - z0),
                math.pi * (r1 + r0) * math.hypot(z1 - z0, r1 - r0))
    for sg in c["segments"]:
        if "line" in sg:
            dv, da = line(cur, tuple(sg["line"]))
            cur = tuple(sg["line"])
        else:
            e = sg["ellipse"]
            (rc, zc), (ar, az), (t0, t1) = e["center"], e["axes"], e["t"]
            npc = max(1, int(math.ceil(abs(t1 - t0) / (math.pi / 8))))
            dv = da = 0.0
            for k in range(npc):
                ta, tb = t0 + (t1 - t0) * k / npc, t0 + (t1 - t0) * (k + 1) / npc
                t = 0.5 * (ta + tb) + 0.5 * (tb - ta) * xg
                ww = 0.5 * (tb - ta) * wg
                r = rc + ar * np.cos(t)
                dr, dz = -ar * np.sin(t), az * np.cos(t)
                dv += float(np.sum(ww * math.pi * r * r * dz))
                da += float(np.sum(np.abs(ww) * 2 * math.pi * r * np.hypot(dr, dz)))
            cur = (rc + ar * math.cos(t1), zc + az * math.sin(t1))
        V += dv
        A += da
    dv, _ = line(cur, tuple(c["start"]))
    return V + dv, A


def elliptical_cell_contour(cell: dict | None = None, *, axial_factor: float = 1.0, scale: float = 1.0,
                            tube_radius: float | None = None, tube_length: float = 0.100, ds: float = 1e-3,
                            ds_min: float | None = None, angle_tol: float = 0.05) -> dict:
    """Interior wall of a single elliptical cell closed by beam-tube stubs, centred on z = 0 (equator plane).

    ``cell``: {"iris_radius", "equator_radius", "half_length", "iris_axes": (axial, radial), "equator_axes":
    (axial, radial)} [m] (default ``PROJECTX_650_B09``). ``axial_factor`` scales the half-cell length and both
    axial semi-axes (0.86 makes TM020/TE011 degenerate, see the module docstring); ``scale`` scales everything
    uniformly (after the axial factor; also the tubes and ``ds``). Tubes: radius ``tube_radius`` (default the iris radius), length ``tube_length``, closed by
    flat caps; tube_length = 0 closes the cell at the iris planes.
    ``ds``: largest spacing of the polyline vertices on the arcs (before scaling); locally the spacing is
    angle_tol * rho (rho = curvature radius; the wall turns by <= ``angle_tol`` rad per chord), but not below
    ``ds_min`` (default ds/20, before scaling). The polygon's vertices are weak field singularities (a convex
    corner of 180 deg + theta): the pointwise wall E of the pump converges only as theta -> 0 (docs/srf/README.md).

    Returns {"points": [(r, z), ...] the closed cavity polygon (axis included, counter-clockwise), "wall": the wall
    polyline from (0, -z_end) to (0, z_end) (axis excluded), "shape": the exact wall as a ``contour`` shape (lines and
    elliptical arcs; the mesher builds the arcs exactly, so the wall is smooth), "V_m3", "A_m2" (of the exact wall;
    wall area incl. tube caps), "V_polygon_m3", "A_polygon_m2" (of the polygon),
    "length", "equator_radius", "tangent_angle_deg" (the usual SRF "wall angle": the tangent line's angle from the
    radial direction, atan(dz/dr)),
    "params" (the scaled geometry)}."""
    import numpy as np

    c = dict(PROJECTX_650_B09 if cell is None else cell)
    s, f = float(scale), float(axial_factor)
    Ri, Req = s * c["iris_radius"], s * c["equator_radius"]
    L = s * f * c["half_length"]
    a, b = s * f * c["iris_axes"][0], s * c["iris_axes"][1]
    A, B = s * f * c["equator_axes"][0], s * c["equator_axes"][1]
    rt = s * (tube_radius if tube_radius is not None else c["iris_radius"])
    lt = s * tube_length
    ds = s * ds
    dmin = s * ds_min if ds_min is not None else ds / 20
    if rt > Ri + 1e-12:
        raise ValueError(f"tube_radius {rt} is larger than the iris radius {Ri}: the tube would cut the iris")
    P1, P2, (t1, p1) = _half_cell(Ri, Req, L, a, b, A, B)

    def arc(P, u0, u1):
        # vertex spacing g = clamp(angle_tol * rho, dmin, ds) with rho the local curvature radius: every chord turns
        # the wall by <= angle_tol (the polygon's corners are weak field singularities: keep them shallow)
        uu = np.linspace(u0, u1, 40001)
        pts = np.array([P(u) for u in uu])
        d1 = np.gradient(pts, uu, axis=0)
        d2 = np.gradient(d1, uu, axis=0)
        sp = np.hypot(d1[:, 0], d1[:, 1])
        rho = sp ** 3 / np.maximum(np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]), 1e-300)
        g = np.clip(angle_tol * rho, dmin, ds)
        sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        cnt = np.concatenate([[0], np.cumsum(np.diff(sl) * 0.5 * (1 / g[1:] + 1 / g[:-1]))])  # int ds / g
        n = max(4, int(math.ceil(cnt[-1])))
        return np.array([P(u) for u in np.interp(np.linspace(0, cnt[-1], n + 1), cnt, uu)])

    iris = arc(P1, 0.0, t1)
    eq = arc(P2, p1, math.pi / 2)
    half = np.vstack([iris, eq[0:]])          # (z, r), z from 0 (iris) to L (equator); tangent line implicit
    zr = np.column_stack([half[:, 0] - L, half[:, 1]])  # equator at z = 0, left half z in [-L, 0]
    left = zr
    right = np.column_stack([-zr[::-1, 0], zr[::-1, 1]])[1:]
    prof = np.vstack([left, right])           # (z, r) from left iris to right iris
    wall = []
    zend = L + lt
    if lt > 0:
        wall += [(0.0, -zend), (rt, -zend)]
        if rt < Ri - 1e-12:
            wall += [(rt, -L)]
    else:
        wall += [(0.0, -L)]
    wall += [(r, z) for z, r in prof]
    if lt > 0:
        if rt < Ri - 1e-12:
            wall += [(rt, L)]
        wall += [(rt, zend), (0.0, zend)]
    else:
        wall += [(0.0, L)]
    # drop duplicate consecutive points
    w2 = [wall[0]]
    for p in wall[1:]:
        if math.hypot(p[0] - w2[-1][0], p[1] - w2[-1][1]) > 1e-12 * s:
            w2.append(p)
    wall = w2
    q1, q2 = P1(t1), P2(p1)
    # the exact wall as a "contour" shape (lines and elliptical arcs, point(tau) = c + (ar cos tau, az sin tau)):
    # left iris arc tau = pi - t (t: iris parameter from the tip), equator arc tau = p - pi/2 (both halves in one
    # arc), right iris arc the mirror image. The mesher builds the arcs exactly (smooth wall, docs/srf/README.md)
    segs, cur = [], None

    def line(P):
        nonlocal cur
        if cur is None or math.hypot(P[0] - cur[0], P[1] - cur[1]) > 1e-12 * s:
            segs.append({"line": [float(P[0]), float(P[1])]})
            cur = (float(P[0]), float(P[1]))

    def ell(c0, ax, ta, tb):
        nonlocal cur
        segs.append({"ellipse": {"center": [float(c0[0]), float(c0[1])], "axes": [float(ax[0]), float(ax[1])],
                                 "t": [float(ta), float(tb)]}})
        cur = (c0[0] + ax[0] * math.cos(tb), c0[1] + ax[1] * math.sin(tb))

    start = (0.0, -zend) if lt > 0 else (0.0, -L)
    cur = start
    if lt > 0:
        line((rt, -zend))
        if rt < Ri - 1e-12:
            line((rt, -L))
    line((Ri, -L))
    ell((Ri + b, -L), (b, a), math.pi, math.pi - t1)                       # left iris arc, tip -> tangent point
    line((q2[1], q2[0] - L))                                                 # tangent line
    ell((Req - B, 0.0), (B, A), p1 - math.pi / 2, math.pi / 2 - p1)         # equator arc, both halves
    line((q1[1], L - q1[0]))                                                 # tangent line (right)
    ell((Ri + b, L), (b, a), t1 - math.pi, -math.pi)                        # right iris arc, tangent point -> tip
    if lt > 0:
        if rt < Ri - 1e-12:
            line((rt, L))
        line((rt, zend))
        line((0.0, zend))
    else:
        line((0.0, L))
    shape = {"contour": {"start": [start[0], start[1]], "segments": segs}}
    V, Aw = _contour_volume_area(shape)
    W = np.array(wall)
    rp, zp = W[:, 0], W[:, 1]
    V_poly = float(np.sum(np.pi * (rp[1:] ** 2 + rp[1:] * rp[:-1] + rp[:-1] ** 2) / 3 * np.diff(zp)))
    A_poly = float(np.sum(np.pi * (rp[1:] + rp[:-1]) * np.hypot(np.diff(zp), np.diff(rp))))
    ang = math.degrees(math.atan2(q2[0] - q1[0], q2[1] - q1[1]))
    # counter-clockwise polygon in (r, z): wall goes up in z at r > 0 ... the axis closes it from (0, zend) down
    pts = [(float(x), float(y)) for x, y in wall]
    return {"points": pts, "wall": pts, "shape": shape, "V_m3": V, "A_m2": Aw, "V_polygon_m3": V_poly,
            "A_polygon_m2": A_poly, "length": 2 * zend, "z_end": zend,
            "equator_radius": Req, "tangent_angle_deg": ang,
            "params": {"iris_radius": Ri, "equator_radius": Req, "half_length": L, "iris_axes": (a, b),
                       "equator_axes": (A, B), "tube_radius": rt, "tube_length": lt, "axial_factor": f,
                       "scale": s, "tangent_points_zr": [(float(q1[0] - L), float(q1[1])),
                                                         (float(q2[0] - L), float(q2[1]))]}}


def _wall_shape(con: dict, wall: str) -> dict:
    if wall == "smooth":
        return con["shape"]
    if wall == "polygon":
        return {"polygon": {"points": [list(p) for p in con["points"]]}}
    raise ValueError(f"wall = {wall!r}: use 'smooth' (exact elliptical arcs) or 'polygon' (faceted, angle_tol)")


def elliptical_cell_design(cell: dict | None = None, *, axial_factor: float = 0.86, scale: float = 1.0,
                           tube_radius: float | None = None, tube_length: float = 0.100, h: float | None = None,
                           wall_size: float | None = None, order: int = 2, modes: int = 8,
                           families=("TE", "TM"), target_f_Hz: float | None = 1.42e9,
                           wall_material: dict | None = None, t: float | None = None, angle_tol: float = 0.01,
                           wall: str = "smooth"):
    """Axisymmetric design of the elliptical cell (``elliptical_cell_contour``) inside a conducting wall of
    thickness ``t`` (default 5 % of the equator radius): cavity region (``mesh_size`` h, default Req/30) and wall
    surface mesh size ``wall_size`` (default h/3; the mesher also refines by curvature). With ``wall="polygon"`` the
    wall is the polyline (vertex spacing <= ``wall_size``, denser near the iris by ``angle_tol``, see
    :func:`elliptical_cell_contour`) and the mesh follows its vertices. ``target_f_Hz`` is scaled by
    1/scale. ``wall_material`` as in :func:`axionfem.cavity.pillbox_design` (default lossless, G still reported).
    The design carries the contour in ``design.srf_contour`` (used by :func:`figures_of_merit`).
    ``wall``: "smooth" (default) meshes the exact wall (elliptical arcs, ``contour`` shape) with
    curved second-order elements on it; "polygon" meshes the polyline (``angle_tol``), whose vertices are weak field
    singularities (docs/srf/README.md, "Wall representation")."""
    c0 = PROJECTX_650_B09 if cell is None else cell
    Req0 = c0["equator_radius"]
    h0 = h / scale if h is not None else Req0 / 30
    ws0 = wall_size / scale if wall_size is not None else h0 / 3
    con = elliptical_cell_contour(cell, axial_factor=axial_factor, scale=scale, tube_radius=tube_radius,
                                  tube_length=tube_length, ds=ws0, angle_tol=angle_tol)
    Req, ze = con["equator_radius"], con["z_end"]
    t = t or 0.05 * Req
    hh, ws = h0 * scale, ws0 * scale
    regions = [{"name": "wall", "shape": {"rect": {"r": [0.0, Req + t], "z": [-ze - t, ze + t]}},
                "material": "wall_cu"},
               {"name": "cavity", "shape": _wall_shape(con, wall),
                "material": "vacuum", "mesh_size": hh, "surface_size": ws}]
    R = 1.5 * math.hypot(Req + t, ze + t)
    d = _design("srf_elliptical_cell", regions, wall_material=wall_material, fill=None, h=hh, order=order, Bz=1.0,
                modes=modes, families=families, target_f_Hz=(target_f_Hz / scale if target_f_Hz else None),
                interior_point=(0.5 * Req, 0.0), domain_radius=R, surface_size=ws)
    d.srf_contour = con
    d.srf_wall = wall
    return d


def elliptical_cell_design_3d(cell: dict | None = None, *, axial_factor: float = 0.86, scale: float = 1.0,
                              tube_radius: float | None = None, tube_length: float = 0.100, h: float | None = None,
                              wall_size: float | None = None, order: int = 2, modes: int = 12,
                              target_f_Hz: float | None = 1.42e9, eigensolver: dict | None = None,
                              wall_material: dict | None = None, mode_overlaps: bool = True,
                              angle_tol: float = 0.1, wall: str = "smooth", curvature_points: int | None = None,
                              wall_samples: bool | None = None):
    """3D model of the same cell (no mirror planes: the pump TM0n0 and signal TE0np live in different mirror
    sectors, and eta vanishes identically under any mirror symmetry): the closed polygon revolved about z over
    360 degrees is the cavity region and the PEC domain wall. Shift-invert about ``target_f_Hz``/scale.
    ``h``: element size (default Req/12), ``wall_size``: surface size and polygon spacing (default h/2).
    ``wall``: "smooth" (default): the exact wall (elliptical arcs) revolved, with curved
    second-order elements on it; "polygon": the polyline (``angle_tol``) revolved, a faceted wall whose narrow conical
    strips near the iris force poorly shaped elements (docs/srf/README.md, "Wall representation").
    ``curvature_points`` (smooth wall): elements per 2 pi of wall turning (gmsh curvature sizing), default
    max(16, 1.5 R_eq/h), so that the iris nose (curvature radius 6 mm) is refined together with h.
    ``wall_samples`` (default: with the smooth wall): ``cavity.wall_samples``, the per-mode wall quadrature points that
    :func:`figures_of_merit_3d` turns into the 2D-comparable wall peaks (:func:`azimuthal_wall_peaks`); about 5 MB
    per mode at h = R_eq/8."""
    from .cavity import _design3d

    c0 = PROJECTX_650_B09 if cell is None else cell
    Req0 = c0["equator_radius"]
    h0 = h / scale if h is not None else Req0 / 12
    ws0 = wall_size / scale if wall_size is not None else h0 / 2
    con = elliptical_cell_contour(cell, axial_factor=axial_factor, scale=scale, tube_radius=tube_radius,
                                  tube_length=tube_length, ds=ws0, angle_tol=angle_tol)
    solid = {"revolve": {"profile": _wall_shape(con, wall),
                         "angles": [0.0, 360.0], "center": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}}
    # center/axis explicit: mesh.domain_solid reaches the mesher unresolved (no defaults filled in)
    regions = [{"name": "cavity", "material": "vacuum", "solid": solid}]
    Req = con["equator_radius"]
    ip = (0.41 * Req, 0.23 * Req, 0.0123 * Req)
    es = dict(eigensolver) if eigensolver else ({"method": "shift_invert"} if target_f_Hz else None)
    d = _design3d("srf_elliptical_cell_3d", regions, solid, ip, h=h0 * scale, order=order, modes=modes,
                  target_f_Hz=(target_f_Hz / scale if target_f_Hz else None), eigensolver=es, symmetry=None,
                  B_T=(0.0, 0.0, 1.0), wall_material=wall_material, losses=True, mode_overlaps=mode_overlaps,
                  surface_size=ws0 * scale, remove_pec=False)
    if wall == "smooth":
        # grade the volume from the curvature-refined wall (the iris nose) and remove slivers: a fine wall mesh next to
        # a coarse interior otherwise leaves flat wall elements, whose normal E (the whole wall E) is poorly resolved
        cp = curvature_points if curvature_points is not None else max(16, int(math.ceil(1.5 * Req0 / h0 - 1e-9)))
        d.doc["mesh"].update(extend_from_boundary=1, optimize_netgen=True, curvature_points=int(cp))
    if wall_samples if wall_samples is not None else wall == "smooth":  # per-mode wall points (azimuthal_wall_peaks)
        d.doc["cavity"]["wall_samples"] = True
    d.srf_contour = con
    d.srf_wall = wall
    return d


# ------------------------------------------------------------------------------------- mode identification
def _sign_changes(v, rel=2e-3):
    import numpy as np

    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0
    v = v[np.abs(v) > rel * np.abs(v).max()]
    return int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))


def classify_modes(res, contour: dict | None = None, n_samples: int = 801) -> list[dict]:
    """Label the m = 0 modes of an axisymmetric cavity result as TM0np / TE0np by their nodal lines (closed
    cylinder convention: TM H_phi ~ J1(kc r) cos(p pi z/L), TE E_phi ~ J1(kc r) sin(p pi z/L)):
    n - 1 = sign changes of the primary field (TM: H_phi, TE: E_phi) along the radius, p = sign changes along z
    (TM), p - 1 (TE); n - 1 is counted on the radial line through the largest |field|, p on the axial line through the largest
    |field| of that radial line. Rows of ``res.cavity_modes()`` with "name" (e.g. "TM020"), "n", "p", "z_parity" (+1
    even, -1 odd primary field). A heuristic for cell-like (single-cell) cavities: check the labels of new
    geometries (the mode plots of docs/srf)."""
    import numpy as np

    rows = res.cavity_modes()
    if contour is not None:
        W = np.array(contour["wall"])
        rmax = float(W[:, 0].max())
        z0, z1 = float(W[:, 1].min()), float(W[:, 1].max())
    else:
        F = res.cavity_mode_field(0)
        rmax, z0, z1 = float(F["r"].max()), float(F["z"].min()), float(F["z"].max())
    out = []
    for row in rows:
        key = "H_phi" if row["family"] == "TM" else "E_phi"
        F = res.cavity_mode_field(row, subdiv=1)
        j = int(np.nanargmax(np.abs(F[key])))
        # radial line through the largest |field| (the midplane for z-even modes, off it for z-odd ones), then
        # the axial line at the radius of the largest |field| on that radial line
        rr = np.linspace(1e-6 * rmax, rmax * (1 - 1e-6), n_samples)
        sr = res.cavity_mode_sample(row, rr, np.full_like(rr, F["z"][j]))[key]
        i = int(np.nanargmax(np.abs(sr)))
        zz = np.linspace(z0 + 1e-6 * (z1 - z0), z1 - 1e-6 * (z1 - z0), n_samples)
        sz = res.cavity_mode_sample(row, np.full_like(zz, rr[i]), zz)[key]
        n = _sign_changes(sr) + 1
        pc = _sign_changes(sz)
        p = pc if row["family"] == "TM" else pc + 1
        ok = np.isfinite(sz)
        fz = sz[ok]
        par = 1 if np.nansum(fz * fz[::-1]) >= 0 else -1
        out.append({**row, "n": n, "p": p, "z_parity": par, "name": f"{row['family']}0{n}{p}"})
    return out


def find_mode(res, name: str, contour: dict | None = None, labels: list | None = None) -> dict:
    """The cavity_modes() row labelled ``name`` (e.g. "TM020") by :func:`classify_modes`; error if absent or
    ambiguous."""
    labels = labels if labels is not None else classify_modes(res, contour)
    hit = [r for r in labels if r["name"] == name]
    if len(hit) != 1:
        raise ValueError(f"mode {name}: {len(hit)} matches among {[r['name'] for r in labels]} (f = "
                         f"{[round(r['f_Hz'] / 1e9, 5) for r in labels]} GHz); widen/shift fullwave.modes or "
                         f"target_f_Hz, or pick the row by index")
    return hit[0]


# --------------------------------------------------------------------------------------- figures of merit
def _tri_product_integral(r, z, tris, f, g):
    """sum over sub-triangles of int f g 2 pi r dA with f, g, r linear on each sub-triangle (exact for that)."""
    import numpy as np

    P = np.column_stack([r, z])[tris]                  # (nt, 3, 2)
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    F, Gv, R = f[tris], g[tris], r[tris]
    # int l_i l_j l_k dA = 2 A i!j!k!/(i+j+k+2)! over the barycentrics: sum_{ijk} F_i G_j R_k w_ijk
    s = np.zeros(len(tris))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                cnt = np.bincount([i, j, k], minlength=3)
                w = math.factorial(cnt[0]) * math.factorial(cnt[1]) * math.factorial(cnt[2]) / math.factorial(5)
                s += F[:, i] * Gv[:, j] * R[:, k] * w
    return float(np.sum(2 * area * s) * 2 * math.pi)


def figures_of_merit(res, pump="TM020", signal="TE011", *, contour: dict | None = None, subdiv: int = 4,
                     with_fields: bool = False) -> dict:
    """Heterodyne figures of merit of an axisymmetric cavity result (see the module docstring for definitions).

    ``pump``/``signal``: mode labels for :func:`find_mode`, or rows of ``res.cavity_modes()``. ``contour``: the
    cell contour (``design.srf_contour``; default ``res.design.srf_contour``): mode labels, equator and A.
    Wall peaks: the solver's ``peak_wall_fields``, the largest normal |E| and tangential |B| over
    the Gauss points of the wall faces evaluated in the cavity element (never at mesh vertices), the same field the
    3D peaks estimate (:func:`figures_of_merit_3d`); face-rms variants and the tangential |E| (a PEC check) too. The
    peak ratios use the solver's B_rms, the equator values the lattice B0 (they agree to ~3e-5).
    Integrals: sub-triangle lattice of each element (``subdiv``) with the fields linear on each sub-triangle
    (converges as (h/subdiv)^2; the eigenvectors themselves converge with the mesh).
    Returns xi (pump, signal), f, the frequency splitting, eta_10, B_pk/B0 and E_pk/(c B0) of the pump (and of the
    signal, normalized by the signal's own rms B), A/V^(2/3), G (solver), V, and with ``with_fields`` the mode
    fields (``cavity_mode_field`` dicts) under "fields"."""
    import numpy as np

    con = contour if contour is not None else getattr(res.design, "srf_contour", None)
    labels = classify_modes(res, con)
    pr = pump if isinstance(pump, dict) else find_mode(res, pump, con, labels)
    sr = signal if isinstance(signal, dict) else find_mode(res, signal, con, labels)
    pr = next((x for x in labels if x["family"] == pr["family"] and x["index"] == pr["index"]), pr)
    sr = next((x for x in labels if x["family"] == sr["family"] and x["index"] == sr["index"]), sr)
    if pr["family"] == sr["family"]:
        raise ValueError("pump and signal of the same family (TM/TE) have eta = 0 identically for m = 0 modes")
    Fp = res.cavity_mode_field(pr, subdiv=subdiv)
    Fs = res.cavity_mode_field(sr, subdiv=subdiv)
    if not (np.array_equal(Fp["r"], Fs["r"]) and np.array_equal(Fp["z"], Fs["z"])):
        raise RuntimeError("pump and signal lattices differ (internal)")
    r, z, tris = Fp["r"], Fp["z"], Fp["tris"]
    V = res.fullwave["cavity"]["volume_m3"]
    I = lambda f, g: _tri_product_integral(r, z, tris, f, g)  # noqa: E731
    # B = mu0 H (vacuum); the solver normalization int |E|^2 = 1, int |H|^2 = 1/Z0^2 is re-measured here
    Bp = {k: MU0 * Fp[k] for k in ("H_r", "H_phi", "H_z")}
    Bs = {k: MU0 * Fs[k] for k in ("H_r", "H_phi", "H_z")}
    BB = sum(I(Bp[k], Bp[k]) for k in Bp)
    BBs = sum(I(Bs[k], Bs[k]) for k in Bs)
    EE = sum(I(Fs[k], Fs[k]) for k in ("E_r", "E_phi", "E_z"))
    EEp = sum(I(Fp[k], Fp[k]) for k in ("E_r", "E_phi", "E_z"))
    EB = sum(I(Fs[e], Bp[b]) for e, b in (("E_r", "H_r"), ("E_phi", "H_phi"), ("E_z", "H_z")))
    eta = abs(EB) / math.sqrt(EE * BB)
    # wall peaks: the solver's wall quadrature points (Gauss points of the wall faces, never mesh
    # vertices; normal |E| and tangential |B| evaluated in the cavity element), the definition of the 3D solver
    # (figures_of_merit_3d). The equator values are sampled just inside the wall at the equator.
    if con is None:
        raise ValueError("figures_of_merit needs the wall contour (design.srf_contour or contour=)")
    blocks = res.fullwave["eigenmodes"]
    Req = con["equator_radius"]

    def peaks(row, F, BBv):
        pw = blocks[row["family"]]["modes"][int(row["index"])].get("peak_wall_fields")
        if not pw or "Bpk_over_Brms" not in pw:
            raise RuntimeError("figures_of_merit: the result has no peak_wall_fields (the design needs "
                               "cavity.losses)")
        B0 = math.sqrt(BBv / V)
        q = res.cavity_mode_sample(row, np.array([Req * (1 - 1e-7)]), np.array([0.0]))
        Beq = MU0 * math.sqrt(sum(float(q[k][0]) ** 2 for k in ("H_r", "H_phi", "H_z")))
        Eeq = math.sqrt(sum(float(q[k][0]) ** 2 for k in ("E_r", "E_phi", "E_z")))
        return {"B_rms": B0, "Bpk_over_B0": float(pw["Bpk_over_Brms"]), "Bpk_at_rz": tuple(pw["B_t_max_at_rz"]),
                "Epk_over_cB0": float(pw["cEpk_over_Brms"]), "Epk_at_rz": tuple(pw["E_max_at_rz"]),
                "Bpk_over_B0_face_rms": float(pw["Bpk_face_rms_over_Brms"]),
                "Epk_over_cB0_face_rms": float(pw["cEpk_face_rms_over_Brms"]),
                "Et_max_over_cB0": float(pw["cEt_max_over_Brms"]), "n_wall_points": int(pw["n_points"]),
                "B_equator_over_B0": Beq / B0, "E_equator_over_cB0": Eeq / (C_LIGHT * B0)}

    pk, sk = peaks(pr, Fp, BB), peaks(sr, Fs, BBs)
    fp, fs = pr["f_Hz"], sr["f_Hz"]
    A = con["A_m2"]
    out = {"pump": pr["name"], "signal": sr["name"], "f_pump_Hz": fp, "f_signal_Hz": fs,
           "splitting_Hz": fs - fp, "V_m3": V, "V_contour_m3": con["V_m3"], "A_m2": A,
           "A_over_V23": A / V ** (2 / 3),
           "xi_pump": 2 * math.pi * fp / C_LIGHT * V ** (1 / 3), "xi_signal": 2 * math.pi * fs / C_LIGHT * V ** (1 / 3),
           "eta_10": eta, "eta_10_squared": eta * eta,
           "Bpk_over_B0": pk["Bpk_over_B0"], "Epk_over_cB0": pk["Epk_over_cB0"],
           "B_equator_over_B0": pk["B_equator_over_B0"], "pump_peaks": pk,
           "signal_peaks": sk, "G_pump_ohm": pr["G_ohm"], "G_signal_ohm": sr["G_ohm"],
           "checks": {"int_E1^2 (1 expected)": EE, "int_E0^2 (1 expected)": EEp,
                      "c^2 int B0^2 (1 expected at resonance)": BB * C_LIGHT ** 2,
                      "c^2 int B1^2 (1 expected)": BBs * C_LIGHT ** 2,
                      "n_wall_points": pk["n_wall_points"], "subdiv": subdiv,
                      "volume_contour_vs_solver": con["V_m3"] / V - 1}}
    if with_fields:
        out["fields"] = {"pump": Fp, "signal": Fs}
    return out


def _contour_arclength(shape: dict, r, z, rel_step: float = 2e-5):
    """Arclength s along a ``contour`` shape of the nearest point to each (r, z) (dense sampling, spacing rel_step x
    the length), the distance to it, and the total length."""
    import numpy as np
    from scipy.spatial import cKDTree

    C = contour_points(shape, n_arc=4000)
    L0 = float(np.hypot(*np.diff(C, axis=0).T).sum())
    step = rel_step * L0
    P = [C[:1]]
    for p0, p1 in zip(C[:-1], C[1:]):
        m = max(1, int(np.ceil(np.hypot(*(p1 - p0)) / step)))
        P.append(p0 + (p1 - p0) * np.linspace(0, 1, m + 1)[1:, None])
    P = np.vstack(P)
    sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    d, j = cKDTree(P).query(np.column_stack([r, z]))
    return sl[j], d, float(sl[-1])


def _contour_point_at(shape: dict, s_m: float):
    """(r, z) of the point at arclength ``s_m`` along a ``contour`` shape (dense sampling, as _contour_arclength)."""
    import numpy as np

    C = contour_points(shape, n_arc=4000)
    sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    return (float(np.interp(s_m, sl, C[:, 0])), float(np.interp(s_m, sl, C[:, 1])))


def _peak_fit(s, v, w, s0, W, min_points=12):
    """dA-weighted least-squares quadratic v ~ a + b u + c u^2 (u = s - s0) over |u| <= W, re-centred once on its
    vertex; returns (peak, s_peak, scatter-only standard error, points used), or None when there are fewer than
    ``min_points`` points, the fit is not concave, or its vertex falls outside the window (not a peak)."""
    import numpy as np

    for it in range(4):  # re-centre on the vertex (by at most W per step) until it lies inside the window
        m = np.abs(s - s0) <= W
        if m.sum() < max(min_points, 4):
            return None
        u, vv, ww = s[m] - s0, v[m], w[m]
        A = np.column_stack([np.ones_like(u), u, u * u]) * np.sqrt(ww)[:, None]
        coef, *_ = np.linalg.lstsq(A, vv * np.sqrt(ww), rcond=None)
        a0, b0, c0 = coef
        if not c0 < 0:
            return None
        us = -b0 / (2 * c0)
        if abs(us) <= 0.5 * W and it > 0:
            break
        s0 = s0 + float(np.clip(us, -W, W))
    else:
        return None
    s0 = s0 + us
    fit = a0 + b0 * u + c0 * u * u
    res2 = float(np.sum(ww * (vv - fit) ** 2) / np.sum(ww))
    neff = float(np.sum(ww) ** 2 / np.sum(ww * ww))
    pk = float(a0 + b0 * us + c0 * us * us)
    return pk, float(s0), float(np.sqrt(res2 / max(neff - 3, 1.0))), int(m.sum())


def azimuthal_wall_peaks(peak_wall_fields: dict, contour: dict, bin_width: float | None = None,
                         min_points: int = 10, fit_halfwidth: float | None = None) -> dict | None:
    """Wall peaks of an m = 0 mode of a 3D solution, comparable with the 2D (meridional) ones.

    The wall quadrature points (``cavity.wall_samples``) are placed on the meridional contour by their arclength s
    (nearest point of the exact contour). For an axisymmetric mode the exact wall field is a function of s only; the
    3D solution scatters around it by a few % from point to point (a single-point maximum over thousands of points is
    therefore biased high). In each mirror half of the cell:

    1. coarse location: the largest dA-weighted azimuthal rms over bins of width ``bin_width`` (default 0.5 mm x
       R_eq/0.2003 m) holding >= ``min_points`` points;
    2. value: a dA-weighted least-squares quadratic in s through all points within ``fit_halfwidth`` (default 1.5 mm
       x R_eq/0.2003 m) of it, re-centred once on its vertex; the peak is the vertex value.

    If a half has no concave peak inside the window (broad or noisy maxima), its binned maximum is used and
    ``*_fit_ok`` is False. The reported peak is the mean of the two halves (their difference, ``*_halves``, is a direct measure of the 3D
    noise), with the fit's scatter-only standard error (the samples are correlated within elements, so it is a lower
    bound). Quantities: |E| (``Epk_over_cB0``, the same pointwise field as the 2D
    peak), the consistent-flux |E_n| (``Epk_flux_over_cB0``), |B_t| (``Bpk_over_B0``) and the flux |B_t|
    (``Bpk_flux_over_B0``), in units of (c) B_rms; also the binned maxima (``*_bin_max``). Raises if samples lie
    farther than 1e-2 R_eq from the contour (a deformed or non-matching mesh). None without a samples file."""
    import numpy as np

    path = peak_wall_fields.get("samples_file")
    if not path or not peak_wall_fields.get("B_rms"):
        return None
    D = np.fromfile(path, dtype=np.float64)
    n_exp = peak_wall_fields.get("samples_count")
    if D.size % 8 or (n_exp is not None and D.size != 8 * int(n_exp)):
        raise ValueError(f"azimuthal_wall_peaks: {path} holds {D.size} doubles, expected 8 x {n_exp}: the file belongs "
                         "to another solve (wall-sample files are rewritten by every solve in the workdir)")
    D = D.reshape(-1, 8)
    x, y, z, dA = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    r = np.hypot(x, y)
    s, dist, Ltot = _contour_arclength(contour["shape"], r, z)
    Req = contour["equator_radius"]
    if dist.max() > 1e-2 * Req:  # the curved (quadratic) faces deviate from the surface by up to ~1e-3 R_eq
        raise ValueError(f"azimuthal_wall_peaks: wall samples up to {dist.max():.3g} m from the contour (> 1e-2 R_eq): "
                         "the mesh is not the revolved contour (deformed or a different design)")
    scale = Req / PROJECTX_650_B09["equator_radius"]
    w = bin_width if bin_width is not None else 5e-4 * scale
    W = fit_halfwidth if fit_halfwidth is not None else 1.5e-3 * scale
    Brms = float(peak_wall_fields["B_rms"])
    out = {"bin_width_m": w, "fit_halfwidth_m": W, "n_samples": int(len(r)),
           "max_distance_to_contour_m": float(dist.max())}
    halves = (s < 0.5 * Ltot, s >= 0.5 * Ltot)
    for key, col, norm in (("Epk_over_cB0", 4, C_LIGHT * Brms), ("Epk_flux_over_cB0", 5, C_LIGHT * Brms),
                           ("Bpk_over_B0", 6, Brms), ("Bpk_flux_over_B0", 7, Brms)):
        stem = key.split("_over")[0]
        v = D[:, col] / norm
        hv, binmax, se, locs = [], [], [], []
        fit_ok = True
        for hm in halves:
            sh, vh, wh = s[hm], v[hm], dA[hm]
            b = np.floor(sh / w).astype(np.int64)
            ub, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
            rms = np.sqrt(np.bincount(inv, wh * vh * vh) / np.maximum(np.bincount(inv, wh), 1e-300))
            rms[cnt < min_points] = -1
            if not (rms > 0).any():
                return None
            i = int(np.argmax(rms))
            binmax.append(float(rms[i]))
            f = _peak_fit(sh, vh, wh, (ub[i] + 0.5) * w, W, min_points)
            if f is None:  # no concave peak inside the window: keep the binned maximum, flagged
                fit_ok = False
                f = (float(rms[i]), float((ub[i] + 0.5) * w), float("nan"), int(cnt[i]))
            hv.append(f[0])
            se.append(f[2])
            locs.append(_contour_point_at(contour["shape"], f[1]))
        out[key] = float(np.mean(hv))
        out[key + "_fit_ok"] = fit_ok
        out[key + "_halves"] = hv
        out[key + "_stderr"] = float(np.hypot(*se) / 2)
        out[key + "_bin_max"] = float(max(binmax))
        out[stem + "_at_rz"] = locs[int(np.argmax(hv))]
    return out


def figures_of_merit_3d(res, f_pump_Hz: float, f_signal_Hz: float, contour: dict | None = None) -> dict:
    """Pick the pump (E_phi fraction ~ 0: TM0n0-like) and signal (E_phi fraction ~ 1: TE0np-like) 3D modes
    nearest to the given (2D) frequencies and return their figures of merit from the 3D results: f, xi, eta_10
    (``overlaps.eta[signal][pump]``), G, B_pk/B0 and E_pk/(c B0), and the E_phi fractions as the identification
    evidence. Wall peaks: with the exact contour and ``cavity.wall_samples`` (the default of
    :func:`elliptical_cell_design_3d`), :func:`azimuthal_wall_peaks` (largest azimuthal rms of the wall quadrature
    values in a meridional bin), comparable with the 2D peaks; otherwise, and always under ``*_pointwise_max``, the
    largest single quadrature-point value (``peak_wall_fields``), which the solution's azimuthal scatter biases high."""
    fw = res.fullwave
    modes = fw["eigenmodes"]["modes"]
    V = fw["cavity"]["volume_m3"]

    def pick(f, want_te):
        def ok(m):
            fr = m.get("E_azimuthal_fraction", 0.5)
            return fr > 0.99 if want_te else fr < 0.01
        cands = [(abs(m["f_Hz"] - f), i) for i, m in enumerate(modes) if ok(m)]
        if not cands:
            raise ValueError("no 3D mode with the required E_phi fraction (TE > 0.99, TM < 0.01) near "
                             f"{f / 1e9:.6f} GHz: request more modes, or (exact degeneracy can mix the pair in 3D) "
                             "detune the cell slightly")
        return min(cands)[1]

    ip, is_ = pick(f_pump_Hz, False), pick(f_signal_Hz, True)
    mp, ms = modes[ip], modes[is_]
    eta = fw["eigenmodes"].get("overlaps", {}).get("eta")
    pkp, pks = mp.get("peak_wall_fields", {}), ms.get("peak_wall_fields", {})
    out = {"i_pump": ip, "i_signal": is_, "f_pump_Hz": mp["f_Hz"], "f_signal_Hz": ms["f_Hz"],
           "splitting_Hz": ms["f_Hz"] - mp["f_Hz"], "V_m3": V,
           "xi_pump": 2 * math.pi * mp["f_Hz"] / C_LIGHT * V ** (1 / 3),
           "xi_signal": 2 * math.pi * ms["f_Hz"] / C_LIGHT * V ** (1 / 3),
           "eta_10": eta[is_][ip] if eta else None,
           "eta_01_transpose": eta[ip][is_] if eta else None,
           "G_pump_ohm": mp.get("losses", {}).get("G_ohm"), "G_signal_ohm": ms.get("losses", {}).get("G_ohm"),
           # largest single wall quadrature point (biased high by the solution's azimuthal scatter: the extreme of
           # thousands of samples; see "azimuthal" below for the definition comparable with 2D)
           "Bpk_over_B0": pkp.get("Bpk_over_Brms"),
           "Epk_over_cB0": pkp.get("cEpk_over_Brms"),
           "Bpk_over_B0_pointwise_max": pkp.get("Bpk_over_Brms"),
           "Epk_over_cB0_pointwise_max": pkp.get("cEpk_over_Brms"),
           "Epk_over_cB0_flux_max": pkp.get("cEpk_flux_over_Brms"),
           "Epk_over_cB0_flux_face_rms": pkp.get("cEpk_flux_face_rms_over_Brms"),
           "Bpk_over_B0_flux": pkp.get("Bpk_flux_over_Brms"),
           "Bpk_over_B0_flux_face_rms": pkp.get("Bpk_flux_face_rms_over_Brms"),
           "Bpk_over_B0_face_rms": pkp.get("Bpk_face_rms_over_Brms"),
           "Epk_over_cB0_face_rms": pkp.get("cEpk_face_rms_over_Brms"),
           "Bpk_at_xyz": pkp.get("B_t_max_at"),
           "Epk_at_xyz": pkp.get("E_max_at"), "Epk_flux_at_xyz": pkp.get("E_n_max_flux_at"),
           "E_azimuthal_fraction": (mp.get("E_azimuthal_fraction"), ms.get("E_azimuthal_fraction")),
           "residual_true": (mp.get("residual_true"), ms.get("residual_true")),
           "G_flux_pointwise_disagreement": (
               max((w.get("G_flux_pointwise_disagreement", 0) for w in mp.get("losses", {}).get("walls", {}).values()),
                   default=None),
               max((w.get("G_flux_pointwise_disagreement", 0) for w in ms.get("losses", {}).get("walls", {}).values()),
                   default=None))}
    out["peak_definition"] = "pointwise_max"
    if contour is not None:  # A and V of the exact contour; V_m3 above is the meshed volume
        out["A_over_V23"] = contour["A_m2"] / contour["V_m3"] ** (2 / 3)
        out["V_contour_m3"] = contour["V_m3"]
        if "shape" in contour:
            az = azimuthal_wall_peaks(pkp, contour)
            if az is None and pkp.get("samples_file"):
                import warnings
                warnings.warn("figures_of_merit_3d: the azimuthal wall-peak fit failed (too few points or no concave "
                              "peak); Bpk_over_B0 / Epk_over_cB0 are the single-point maxima (peak_definition = "
                              "'pointwise_max')", stacklevel=2)
            if az is not None:
                # the 2D-comparable definition (docs/srf/README.md, "Wall peaks"): the largest azimuthal rms of the
                # wall quadrature values in a meridional bin; the pointwise maxima stay under *_pointwise_max
                out["azimuthal"] = az
                out["Bpk_over_B0"] = az["Bpk_over_B0"]
                out["Epk_over_cB0"] = az["Epk_over_cB0"]
                out["Bpk_at_rz"], out["Epk_at_rz"] = az["Bpk_at_rz"], az["Epk_at_rz"]
                out["peak_definition"] = "azimuthal"

    return out


def tune_degeneracy(pump: str = "TM020", signal: str = "TE011", *, cell: dict | None = None,
                    bracket=(0.80, 0.92), tol: float = 1e-5, design_kw: dict | None = None,
                    workdir=None, verbose: bool = False) -> dict:
    """Axial factor s (scaling the half-cell length and the axial semi-axes) at which the 2D pump and signal
    frequencies coincide: secant/Brent on Delta f(s) = f_signal - f_pump, each evaluation one axisymmetric solve.
    Every evaluation re-meshes, so Delta f(s) carries mesh noise at the level of the discretization error (a few
    kHz at h = Req/40, order 3, i.e. ~1e-5 in s): a ``tol`` below that only converges onto the noisy root.
    Returns {"axial_factor", "f_Hz", "history": [(s, f_pump, f_signal)], "design_kw"}."""
    import tempfile

    from scipy.optimize import brentq

    kw = dict(design_kw or {})
    hist = []
    base = workdir or tempfile.mkdtemp(prefix="srf_tune_")

    def df(s):
        d = elliptical_cell_design(cell, axial_factor=s, **kw)
        r = d.solve(f"{base}/s_{s:.9f}", verbose=0)
        lab = classify_modes(r, d.srf_contour)
        fp, fs = find_mode(r, pump, labels=lab)["f_Hz"], find_mode(r, signal, labels=lab)["f_Hz"]
        hist.append((s, fp, fs))
        if verbose:
            print(f"[srf] axial factor {s:.7f}: f_pump {fp / 1e9:.7f} GHz, f_signal {fs / 1e9:.7f} GHz")
        return fs - fp

    lo, hi = df(bracket[0]), df(bracket[1])  # labels checked (find_mode raises) before the root search
    if lo * hi > 0:
        raise ValueError(f"tune_degeneracy: f_{signal} - f_{pump} has the same sign at both ends of the bracket "
                         f"{bracket} ({lo:.4g} Hz, {hi:.4g} Hz); widen it")
    s = brentq(df, bracket[0], bracket[1], xtol=tol, rtol=1e-12)
    last = min(hist, key=lambda x: abs(x[0] - s))
    return {"axial_factor": s, "f_Hz": 0.5 * (last[1] + last[2]), "history": hist, "design_kw": kw}


# --------------------------------------------------------------------------------------- closed cylinders
CYLINDER_PAIRS = {  # name: (pump (family, n, p), signal (family, n, p)) -- Berlin:2019ahk and Lasenby:2019prg
    "TM020/TE011": (("TM", 2, 0), ("TE", 1, 1)),
    "TM030/TE021": (("TM", 3, 0), ("TE", 2, 1)),
    "TE012/TM013": (("TE", 1, 2), ("TM", 1, 3)),
}


def _kc(fam, n, a):
    from scipy.special import jn_zeros, jnp_zeros

    return (jn_zeros(0, n)[-1] if fam == "TM" else jnp_zeros(0, n)[-1]) / a


def cylinder_pair_geometry(pump, signal) -> dict:
    """L/R at which the two m = 0 modes of a closed cylinder are degenerate (k^2 = kc^2 + (p pi/L)^2), and
    xi = k V^(1/3) at that ratio (R = 1)."""
    (fp, np_, pp), (fs, ns, ps) = pump, signal
    k1, k2 = _kc(fp, np_, 1.0), _kc(fs, ns, 1.0)
    num = (ps * ps - pp * pp) * math.pi ** 2
    den = k1 * k1 - k2 * k2
    if num * den <= 0:
        raise ValueError(f"{pump} and {signal} are never degenerate in a closed cylinder")
    L = math.sqrt(num / den)
    k = math.hypot(k1, pp * math.pi / L)
    return {"L_over_R": L, "kR": k, "xi": k * (math.pi * L) ** (1 / 3)}


def cylinder_mode_fields(fam: str, n: int, p: int, a: float, L: float):
    """Closed-cylinder m = 0 mode fields as functions of (r, z), z in [0, L], natural units (c = 1, omega = k),
    up to a common factor and phase: returns (k, E(r, z) -> (E_r, E_phi, E_z), B(r, z) -> (B_r, B_phi, B_z)).
    TM0np: E_z = J0(kc r) cos(bz), E_r = (b/kc) J1(kc r) sin(bz), B_phi = (k/kc) J1(kc r) cos(bz);
    TE0np: E_phi = J1(kc r) sin(bz), B_r = -(b/k) J1(kc r) cos(bz), B_z = (kc/k) J0(kc r) sin(bz)
    (B = curl E / (i omega); the common factor 1/i is dropped)."""
    import numpy as np
    from scipy.special import j0, j1

    kc, b = _kc(fam, n, a), p * math.pi / L
    k = math.hypot(kc, b)
    z0 = lambda r: np.zeros_like(np.asarray(r, float) + 0.0)  # noqa: E731
    if fam == "TM":
        E = lambda r, z: ((b / kc) * j1(kc * r) * np.sin(b * z), z0(r * z), j0(kc * r) * np.cos(b * z))  # noqa: E731
        B = lambda r, z: (z0(r * z), (k / kc) * j1(kc * r) * np.cos(b * z), z0(r * z))  # noqa: E731
    else:
        E = lambda r, z: (z0(r * z), j1(kc * r) * np.sin(b * z), z0(r * z))  # noqa: E731
        B = lambda r, z: (-(b / k) * j1(kc * r) * np.cos(b * z), z0(r * z), (kc / k) * j0(kc * r) * np.sin(b * z))  # noqa: E731
    return k, E, B


def eta_cylinder(a: float, L: float, pump, signal) -> float:
    """eta_10 = |int E_signal . B_pump| / sqrt(int |E_signal|^2 int |B_pump|^2) for m = 0 modes of a closed vacuum
    cylinder (radius a, length L); ``pump``/``signal`` = (family, n, p). Separable radial x axial integrals
    (scipy quad, rel. 1e-13)."""
    from scipy.integrate import quad
    from scipy.special import j0, j1

    (fp, np_, pp), (fs, ns, ps) = pump, signal
    q = lambda f, hi: quad(f, 0, hi, epsabs=1e-15, epsrel=1e-13, limit=400)[0]  # noqa: E731
    k0c, b0 = _kc(fp, np_, a), pp * math.pi / L
    k1c, b1 = _kc(fs, ns, a), ps * math.pi / L
    k0 = math.hypot(k0c, b0)
    Rr = lambda f, g: q(lambda r: f(r) * g(r) * r, a)  # noqa: E731
    Zz = lambda f, g: q(lambda z: f(z) * g(z), L)  # noqa: E731
    J0_ = lambda kk: (lambda r: j0(kk * r))  # noqa: E731
    J1_ = lambda kk: (lambda r: j1(kk * r))  # noqa: E731
    S = lambda bb: (lambda z: math.sin(bb * z))  # noqa: E731
    Cc = lambda bb: (lambda z: math.cos(bb * z))  # noqa: E731
    if fs == "TE" and fp == "TM":   # E_phi (signal) . B_phi (pump)
        EB = Rr(J1_(k1c), J1_(k0c)) * Zz(S(b1), Cc(b0)) * (k0 / k0c)
        EE = Rr(J1_(k1c), J1_(k1c)) * Zz(S(b1), S(b1))
        BB = Rr(J1_(k0c), J1_(k0c)) * Zz(Cc(b0), Cc(b0)) * (k0 / k0c) ** 2
    elif fs == "TM" and fp == "TE":  # (E_r, E_z) . (B_r, B_z)
        EB = ((b1 / k1c) * (-(b0 / k0)) * Rr(J1_(k1c), J1_(k0c)) * Zz(S(b1), Cc(b0))
              + (k0c / k0) * Rr(J0_(k1c), J0_(k0c)) * Zz(Cc(b1), S(b0)))
        EE = (b1 / k1c) ** 2 * Rr(J1_(k1c), J1_(k1c)) * Zz(S(b1), S(b1)) + Rr(J0_(k1c), J0_(k1c)) * Zz(Cc(b1), Cc(b1))
        BB = (b0 / k0) ** 2 * Rr(J1_(k0c), J1_(k0c)) * Zz(Cc(b0), Cc(b0)) + (k0c / k0) ** 2 * Rr(J0_(k0c), J0_(k0c)) * Zz(S(b0), S(b0))
    else:
        return 0.0
    return abs(EB) / math.sqrt(EE * BB)


def cylinder_pair_design(pair: str = "TM020/TE011", a: float = 0.1, *, h: float | None = None, order: int = 2,
                         modes: int = 6, wall_material: dict | None = None):
    """Axisymmetric closed cylinder (radius a) at the degenerate aspect ratio of ``pair`` (a key of
    ``CYLINDER_PAIRS`` or ((fam, n, p), (fam, n, p))), with the wall contour attached for
    :func:`figures_of_merit` (labels "TM020" etc.). Built on :func:`axionfem.cavity.pillbox_design`."""
    from .cavity import pillbox_design

    pump, signal = CYLINDER_PAIRS[pair] if isinstance(pair, str) else pair
    g = cylinder_pair_geometry(pump, signal)
    L = g["L_over_R"] * a
    f = C_LIGHT * g["kR"] / a / (2 * math.pi)
    d = pillbox_design(a, L, h=h or a / 20, order=order, modes=modes, target_f_Hz=f, wall_material=wall_material,
                       families=("TE", "TM"))
    wall = [(0.0, -L / 2), (a, -L / 2), (a, L / 2), (0.0, L / 2)]
    d.srf_contour = {"points": wall, "wall": wall, "V_m3": math.pi * a * a * L,
                     "A_m2": 2 * math.pi * a * L + 2 * math.pi * a * a, "length": L, "z_end": L / 2,
                     "equator_radius": a, "params": {"pair": pair, "L": L, "a": a, "f_degenerate_Hz": f}}
    return d


def cylinder_pair(pair: str = "TM020/TE011", *, n_wall: int = 4001, n_vol: int = 1201) -> dict:
    """Closed-form figures of merit of a degenerate m = 0 mode pair of a closed vacuum cylinder (R = 1, L at
    degeneracy; all quantities are scale-free): L/R, xi, eta_10, B_pk/B0 and E_pk/(c B0) of the pump (wall
    maxima on a fine grid of ``n_wall`` points per wall), A/V^(2/3), G of pump and signal (ohm; one R_s on all
    walls)."""
    import numpy as np
    pump, signal = CYLINDER_PAIRS[pair] if isinstance(pair, str) else pair
    g = cylinder_pair_geometry(pump, signal)
    a, L = 1.0, g["L_over_R"]
    V, A = math.pi * a * a * L, 2 * math.pi * a * L + 2 * math.pi * a * a
    out = {"pair": pair, "pump": pump, "signal": signal, "L_over_R": L, "xi": g["xi"],
           "eta_10": eta_cylinder(a, L, pump, signal), "A_over_V23": A / V ** (2 / 3)}
    for role, (fam, n, p) in (("pump", pump), ("signal", signal)):
        k, E, B = cylinder_mode_fields(fam, n, p, a, L)
        zz = np.linspace(0, L, n_wall)
        rr = np.linspace(0, a, n_wall)
        # int |B|^2 dV on an n_vol^2 Simpson grid (smooth integrand)
        from scipy.integrate import simpson
        rv, zv = np.linspace(0, a, n_vol), np.linspace(0, L, n_vol)
        Rg, Zg = np.meshgrid(rv, zv)
        Bv = sum(np.asarray(c) ** 2 for c in B(Rg, Zg))
        intB = 2 * math.pi * simpson(simpson(Bv * Rg, x=rv, axis=1), x=zv)
        B0 = math.sqrt(intB / V)
        # wall: barrel r = a (all z), caps z = 0, L (all r)
        bar = [np.asarray(c) for c in B(np.full_like(zz, a), zz)]
        cap0 = [np.asarray(c) for c in B(rr, np.zeros_like(rr))]
        cap1 = [np.asarray(c) for c in B(rr, np.full_like(rr, L))]
        Bw = max(np.sqrt(sum(c ** 2 for c in w)).max() for w in (bar, cap0, cap1))
        Ew = max(np.sqrt(sum(np.asarray(c) ** 2 for c in w)).max()
                 for w in (E(np.full_like(zz, a), zz), E(rr, np.zeros_like(rr)), E(rr, np.full_like(rr, L))))
        # G = Z0 k int|B|^2 / oint |B_t|^2 (B_n = 0 on a PEC wall)
        Sbar = 2 * math.pi * a * np.trapezoid(sum(c ** 2 for c in bar), zz)
        Scap = sum(2 * math.pi * np.trapezoid(sum(c ** 2 for c in w) * rr, rr) for w in (cap0, cap1))
        out[role] = {"mode": f"{fam}0{n}{p}", "kR": k, "B_rms": B0, "Bpk_over_B0": float(Bw / B0),
                     "Epk_over_cB0": float(Ew / B0), "G_ohm": Z0 * k * intB / (Sbar + Scap)}
    out["Bpk_over_B0"], out["Epk_over_cB0"] = out["pump"]["Bpk_over_B0"], out["pump"]["Epk_over_cB0"]
    out["G_pump_ohm"], out["G_signal_ohm"] = out["pump"]["G_ohm"], out["signal"]["G_ohm"]
    return out
