"""Mesh convergence of computed quantities: three-mesh Richardson extrapolation and h-studies.

    from axionfem.convergence import richardson, h_study, adapt_convergence, refine_and_compare

``richardson(h, values)`` takes three meshes (any refinement ratios) and returns the observed order, the extrapolated
value and the grid convergence index (GCI) of the finest mesh, after Celik et al. (J. Fluids Eng. 130, 078001, 2008).
For monotone data the order is the root of the exact three-mesh equation (bisection); with ``formal_order`` the GCI
uses Roache's conservative variant. ``h_study`` solves a design with its device-region mesh sizes scaled by each
factor and extrapolates L_p, Psi, W_mag and each surface's loss factor Gamma. ``adapt_convergence`` applies the same to
the three finest cycles of ``mesh.adapt``, with h = dofs^(-1/d); the adapted meshes are not geometrically similar, so
that estimate is a heuristic. ``refine_and_compare`` solves a small 3D full-wave design on its mesh and on uniform
refinements of it, and compares each mesh's error indicator with the observed error of L per frequency.

The extrapolation assumes the three meshes are in the asymptotic range. ``behaviour`` says what was found: ``monotone``
or ``oscillatory`` with a usable order, or why there is no extrapolation (``unchanged``, ``no usable order``,
``unreliable order``, ``iteration did not converge``). On pre-asymptotic meshes the GCI is an estimate, not a bound.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable

SIZE_KEYS = ("size", "surface_size", "mesh_size", "far_size")
MIN_ORDER = 0.5    # observed orders below this are reported as "no usable order", without extrapolation
MAX_ORDER = 8.0    # without formal_order, observed orders above this are reported as "unreliable order"


def _monotone_order(h1: float, h2: float, h3: float, ratio: float, p_max: float) -> float | None:
    """The root p of (h3^p - h2^p) / (h2^p - h1^p) = ratio (> 0), increasing in p; None if it is not in (0, p_max]."""
    def lhs(p):
        return (h3 ** p - h2 ** p) / (h2 ** p - h1 ** p)
    if ratio <= math.log(h3 / h2) / math.log(h2 / h1):   # the limit p -> 0: no positive root
        return None
    hi = p_max
    if lhs(hi) < ratio:
        return math.inf
    lo = 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lhs(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _oscillatory_order(r21: float, r32: float, ratio: float, iterations: int = 200) -> float | None:
    """Celik's fixed point for s = -1; None unless it converges."""
    try:
        p = abs(math.log(abs(ratio))) / math.log(r21)
        for _ in range(iterations):
            p_new = abs(math.log(abs(ratio)) + math.log((r21 ** p + 1) / (r32 ** p + 1))) / math.log(r21)
            if abs(p_new - p) < 1e-12 * max(1.0, p):
                return p_new
            p = p_new
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def richardson(h: list[float], values: list[float], formal_order: float | None = None,
               noise: float | None = None) -> dict:
    """Observed order, extrapolated value and fine-mesh GCI from three meshes (in any order).

    ``formal_order``: the quantity's formal order. The order used is then min(observed, formal), and the safety
    factor is 3 when they differ by more than 10 % (Roache), else 1.25; an observed order above twice the formal
    one is flagged in ``behaviour`` but still gives this conservative GCI. Without it, orders above MAX_ORDER are
    "unreliable order" (no GCI). ``noise``: an absolute level below which a difference counts as zero (default 1e3
    machine epsilons of the largest value). ``order`` is None when the observed order exceeds MAX_ORDER
    (``order_exceeds``).
    """
    if len(h) != 3 or len(values) != 3:
        raise ValueError("richardson needs exactly three meshes")
    (h1, f1), (h2, f2), (h3, f3) = sorted(zip(h, values))
    if not (0 < h1 < h2 < h3):
        raise ValueError("the three mesh sizes must be positive and distinct")
    r21 = h2 / h1
    r32 = h3 / h2
    e21 = f2 - f1
    e32 = f3 - f2
    floor = noise if noise is not None else 1e3 * 2.2e-16 * max(abs(f1), abs(f2), abs(f3))
    out = {"h": [h1, h2, h3], "values": [f1, f2, f3], "order": None, "extrapolated": None, "gci_fine": None,
           "extrapolation_error": None}

    def done(behaviour, p=None):
        out.update({"behaviour": behaviour, "order": p})
        return out

    small21 = abs(e21) <= floor
    small32 = abs(e32) <= floor
    if small21 and small32:
        out.update({"extrapolated": f1, "gci_fine": 0.0, "extrapolation_error": 0.0})
        return done("unchanged")
    if small21 or small32:
        return done("no usable order (one difference at the noise floor)")
    ratio = e32 / e21
    if ratio > 0:
        behaviour = "monotone"
        p = _monotone_order(h1, h2, h3, ratio, MAX_ORDER)
        if p is None:
            return done("no usable order")
    else:
        behaviour = "oscillatory"
        p = _oscillatory_order(r21, r32, ratio)
        if p is None:
            return done("iteration did not converge")
    observed = p if math.isfinite(p) else None    # above MAX_ORDER the bisection reports infinity: no number
    if p < MIN_ORDER:
        return done("no usable order", observed)
    if formal_order:
        # Roache: the formal order caps the order used and the safety factor rises to 3 when they disagree; an
        # observed order far above the formal one is flagged but still gives this conservative GCI
        p_use = min(p, formal_order)
        safety = 3.0 if abs(p - formal_order) / formal_order > 0.1 else 1.25
        if p > 2 * formal_order:
            behaviour += ", observed order far above formal"
    else:
        if p > MAX_ORDER:
            out["order_exceeds"] = MAX_ORDER
            return done("unreliable order", observed)
        p_use = p
        safety = 1.25
    extrapolated = (r21 ** p_use * f1 - f2) / (r21 ** p_use - 1)
    approx = abs(e21 / f1) if f1 != 0 else math.inf
    out.update({"extrapolated": extrapolated, "gci_fine": safety * approx / (r21 ** p_use - 1),
                "extrapolation_error": abs((extrapolated - f1) / extrapolated) if extrapolated != 0 else None,
                "safety_factor": safety, "order_used": p_use})
    if observed is None:
        out["order_exceeds"] = MAX_ORDER
    return done(behaviour, observed)


def _scale(value, factor: float):
    if isinstance(value, (int, float)):
        return value * factor
    return f"({value})*{factor!r}"   # a parameter expression


def scaled_design(design, factor: float):
    """A copy of the design with its device-region mesh sizes multiplied by ``factor``: ``mesh.size``,
    ``surface_size`` and ``far_size`` (if given), every region's and curve's sizes, and the 3D curvature sizing
    (``curvature_points`` divided by the factor). The far field's automatic size and the 2D curvature sizing are not
    scaled."""
    from .design import Design
    doc = copy.deepcopy(design.doc if hasattr(design, "doc") else design)
    mesh = doc.setdefault("mesh", {})
    if "size" not in mesh:
        raise ValueError("h_study needs an explicit mesh.size to scale")
    if "file" in mesh or "adapt" in mesh:
        raise ValueError("h_study scales generated meshes: remove mesh.file / mesh.adapt")
    for key in SIZE_KEYS:
        if key in mesh:
            mesh[key] = _scale(mesh[key], factor)
    if doc.get("symmetry") == "3d":
        mesh["curvature_points"] = math.ceil(mesh.get("curvature_points", 16) / factor)
    for item in doc.get("regions", []) + doc.get("curves", []):
        for key in SIZE_KEYS:
            if key in item:
                item[key] = _scale(item[key], factor)
    return Design(doc)


def default_quantities(data: dict) -> dict:
    """L_p, Psi and W_mag, and the loss factor Gamma of every pickup surface (pickup.losses), from results data."""
    c = data.get("coupling", {})
    q = {k: c[k] for k in ("L_p_H", "Psi_SI_H_T_m2", "W_mag_J") if k in c}
    for surface, v in data.get("pickup", {}).get("losses", {}).get("surfaces", {}).items():
        q[f"Gamma:{surface}"] = v["Gamma"]
    return q


def _dimension(data: dict) -> int:
    return 3 if data.get("design", {}).get("symmetry") == "3d" else 2


def h_study(design, factors: list[float], workdir: str | Path, *, mpi_ranks: int = 1, verbose: int = 0,
            extract: dict[str, Callable] | None = None, formal_order: float | dict | None = None) -> dict:
    """Solve the design with its mesh sizes scaled by each factor (1 = as given, 0.5 = twice as fine) and extrapolate.

    ``extract`` maps extra quantity names to functions of the Results. With three or more factors the three finest
    give ``extrapolation[name]`` (see ``richardson``), with h = the factor. ``formal_order`` may be one number or
    per quantity, e.g. {"L_p_H": 4, "Psi_SI_H_T_m2": 2, "Gamma": 2} for order-2 elements (energies converge at
    2p). ``dof_ratios`` compares the effective
    refinement (dofs^(1/d)) with the factor ratios; ``warnings`` flags a mismatch above 20 %, where the far field or
    the curvature sizing, which are not scaled in full, set the mesh.
    """
    rows = []
    for f in factors:
        r = scaled_design(design, f).solve(Path(workdir) / f"h_{f:g}", verbose=verbose, mpi_ranks=mpi_ranks)
        q = default_quantities(r.data)
        for name, fn in (extract or {}).items():
            q[name] = fn(r)
        rows.append({"factor": f, "dofs": r.data.get("mesh", {}).get("dofs"), "dimension": _dimension(r.data),
                     "quantities": q})
    out = {"rows": rows, "extrapolation": _extrapolate([(row["factor"], row["quantities"]) for row in rows],
                                                       formal_order),
           "dof_ratios": [], "warnings": []}
    ordered = sorted(rows, key=lambda row: row["factor"])
    for fine, coarse in zip(ordered, ordered[1:]):
        if not (fine["dofs"] and coarse["dofs"]):
            continue
        nominal = coarse["factor"] / fine["factor"]
        effective = (fine["dofs"] / coarse["dofs"]) ** (1.0 / fine["dimension"])
        out["dof_ratios"].append({"factors": [fine["factor"], coarse["factor"]], "nominal": nominal,
                                  "effective": effective})
        if abs(effective / nominal - 1) > 0.2:
            out["warnings"].append(f"factors {fine['factor']:g}/{coarse['factor']:g}: the mesh refined by "
                                   f"{effective:.3g} (dofs^(1/d)) for a nominal {nominal:.3g}; the unscaled far field "
                                   "or curvature sizing sets part of the mesh, so h = factor is approximate")
    return out


def adapt_convergence(results, dimension: int | None = None, formal_order: float | dict | None = None) -> dict:
    """Richardson estimates from the three finest cycles of a mesh.adapt history (h = dofs^(-1/d); heuristic).
    ``dimension`` defaults to 3 for 3D designs and 2 otherwise; cycles with repeated dofs are skipped."""
    data = results.data if hasattr(results, "data") else results
    d = dimension or _dimension(data)
    points = []
    seen = set()
    for row in data.get("adapt", {}).get("history", []):
        if row["dofs"] in seen:
            continue
        seen.add(row["dofs"])
        q = {k: row[k] for k in ("L_p_H", "Psi_SI_H_T_m2", "W_mag_J") if k in row}
        for surface, g in row.get("Gamma", {}).items():
            q[f"Gamma:{surface}"] = g
        points.append((row["dofs"] ** (-1.0 / d), q))
    out = _extrapolate(points, formal_order)
    for v in out.values():
        v["note"] = "adapted meshes are not geometrically similar: h = dofs^(-1/d) makes this a heuristic estimate"
    return out


def fullwave_rows(data: dict) -> list[dict]:
    """Port 0's full-wave quantities per sweep frequency, from results data: f_Hz, L_H, Z_re_ohm, Psi_abs and, for 3D
    runs, the error indicator (``error_indicator``, docs/physics/fullwave_error_indicator.md)."""
    rows = []
    for row in data.get("fullwave", {}).get("sweep", []):
        out = {k: row[k] for k in ("f_Hz", "L_H", "Z_re_ohm", "Z_im_ohm", "Psi_abs") if k in row}
        if "error_indicator" in row:
            out["error_indicator"] = row["error_indicator"]
        rows.append(out)
    return rows


def refine_and_compare(design, workdir: str | Path, levels: int = 3, *, mpi_ranks: int = 1, verbose: int = 0) -> dict:
    """Refine and compare, for small 3D full-wave designs: solve on the design's mesh and on ``levels - 1`` uniform
    refinements of it (``mesh.refinements``), and per sweep frequency compare each mesh's error indicator with the
    observed error of L.

    Uniform refinement halves h and keeps the element geometry, so the differences between the meshes are the field's
    discretization error alone, which is what the indicator estimates (a generated mesh at a smaller size also changes
    the geometry's approximation). The reference is the Richardson extrapolation of L over the three finest meshes
    (h = 2^-level, observed order), else the finest mesh's L, whose own row then has no observed error. The
    indicator's estimate is ``L_relative_error_estimate`` (|Z_h - Z| <= ||E - E_h||^2 / omega per ampere);
    ``effectivity`` = estimate / observed should stay roughly constant under refinement. ``changes`` compares the
    change of L between consecutive meshes with the one the signed estimates predict
    (``L_signed_relative_error_estimate``), which needs no reference; at element order 2 and above the signed
    estimate is withheld (no prediction) where the magnetic and electric errors are within a factor 8 of each other,
    since they then nearly cancel in L (docs/physics/fullwave_error_indicator.md). Frequencies are paired by f_Hz; a frequency missing from one mesh's
    sweep is refused. The cost grows about 8x per level.
    """
    from .design import Design
    doc = copy.deepcopy(design.doc if hasattr(design, "doc") else design)
    if doc.get("symmetry") != "3d" or not doc.get("fullwave", {}).get("frequencies"):
        hint = ""
        if doc.get("fullwave", {}).get("adaptive"):
            hint = (" (an adaptive sweep chooses different frequencies on each mesh, so they could not be paired: list "
                    "the frequencies to compare, e.g. its fullwave.adaptive.solved_frequencies_Hz)")
        raise ValueError("refine_and_compare needs a 3D design with fullwave.frequencies" + hint)
    mesh = doc.setdefault("mesh", {})
    if "adapt" in mesh:
        raise ValueError("refine_and_compare refines a fixed mesh: remove mesh.adapt")
    if levels < 2:
        raise ValueError("refine_and_compare needs at least two levels")
    base = int(mesh.get("refinements", 0))
    meshes = []
    for level in range(levels):
        mesh["refinements"] = base + level
        r = Design(copy.deepcopy(doc)).solve(Path(workdir) / f"refine_{base + level}", verbose=verbose,
                                             mpi_ranks=mpi_ranks)
        meshes.append({"refinements": base + level, "h": 2.0 ** -level, "dofs": r.data.get("mesh", {}).get("dofs"),
                       "elements": r.data.get("mesh", {}).get("elements"), "rows": fullwave_rows(r.data)})
    out = {"levels": [{k: m[k] for k in ("refinements", "dofs", "elements")} for m in meshes], "frequencies": []}
    # the rows of one frequency on every mesh, paired by f_Hz (a frequency rejected on one mesh drops out of its sweep)
    by_frequency = [{row["f_Hz"]: row for row in m["rows"]} for m in meshes]
    frequencies = sorted(by_frequency[0])
    for m, found in zip(meshes, by_frequency):
        if sorted(found) != frequencies:
            raise ValueError(f"refine_and_compare: the sweep at refinements {m['refinements']} has the frequencies "
                             f"{sorted(found)}, the first mesh {frequencies}; a frequency missing on one mesh "
                             "cannot be compared")
    for f in frequencies:
        for m, found in zip(meshes, by_frequency):
            m["row"] = found[f]
        first = meshes[0]["row"]
        values = [m["row"]["L_H"] for m in meshes]
        reference = values[-1]
        extrapolation = None
        if levels >= 3:
            extrapolation = richardson([m["h"] for m in meshes[-3:]], values[-3:])
            if extrapolation["extrapolated"] is not None:
                reference = extrapolation["extrapolated"]
        rows = []
        for m, value in zip(meshes, values):
            indicator = m["row"].get("error_indicator", {})
            observed = abs(value / reference - 1) if reference else None
            if extrapolation is None or extrapolation["extrapolated"] is None:
                observed = None if m is meshes[-1] else observed
            estimate = indicator.get("L_relative_error_estimate")   # None when withheld (JSON null)
            rows.append({"refinements": m["refinements"], "dofs": m["dofs"], "L_H": value,
                         "observed_relative_error": observed, "L_relative_error_estimate": estimate,
                         "L_signed_relative_error_estimate": indicator.get("L_signed_relative_error_estimate"),
                         "estimated_relative_error": indicator.get("estimated_relative_error"),
                         "effectivity": estimate / observed if estimate is not None and observed else None})
        # between consecutive meshes: the change of L against the signed estimates' prediction of it
        # (L_h = L (1 + s), so L_fine / L_coarse - 1 = s_fine - s_coarse), which needs no reference
        changes = []
        for coarse, fine in zip(rows, rows[1:]):
            s0 = coarse["L_signed_relative_error_estimate"]
            s1 = fine["L_signed_relative_error_estimate"]
            changes.append({"refinements": [coarse["refinements"], fine["refinements"]],
                            "L_change_relative": fine["L_H"] / coarse["L_H"] - 1,
                            "predicted_change": s1 - s0 if s0 is not None and s1 is not None else None})
        out["frequencies"].append({"f_Hz": first["f_Hz"], "L_reference_H": reference,
                                   "L_extrapolation": extrapolation, "rows": rows, "changes": changes})
    return out


def _formal(formal_order, name: str):
    """A number for every quantity, or per quantity: {"L_p_H": 4, "Gamma": 2} (a key before ':' covers a family)."""
    if isinstance(formal_order, dict):
        return formal_order.get(name, formal_order.get(name.split(":")[0]))
    return formal_order


def _extrapolate(points: list[tuple[float, dict]], formal_order=None) -> dict:
    if len(points) < 3:
        return {}
    finest = sorted(points, key=lambda x: x[0])[:3]
    names = set(finest[0][1]).intersection(*(set(q) for _, q in finest[1:]))
    return {name: richardson([h for h, _ in finest], [q[name] for _, q in finest],
                             formal_order=_formal(formal_order, name))
            for name in sorted(names)}
