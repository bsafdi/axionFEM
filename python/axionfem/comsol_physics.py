"""Physics for the COMSOL export: an Electromagnetic Waves (``emw``) model of an axionFEM 3D full-wave problem.

``axf export-comsol design.json -o M.java --format java --physics fullwave`` (or ``export_comsol(...,
physics="fullwave")``) adds to the generated model what axionFEM solves in ``fullwave``:

* materials (vacuum, eps_r, mu_r, sigma and tan(delta) of the non-PEC media);
* superconductors and ``conductor`` materials as perfect electric conductors: their domains are left out of
  the emw interface and their walls get a PerfectElectricConductor;
* one LumpedPort per axionFEM gap port (the pickup is port 1, ``fullwave.ports`` follow; only port 1 is
  active, so the others are open circuits exactly as in axionFEM's Z-matrix definition);
* symmetry planes (``mesh.symmetry``) as PerfectElectricConductor / PerfectMagneticConductor;
* the axion source as an ExternalCurrentDensity Je = Je_scale * B0, where B0 [T] is axionFEM's laboratory
  field (``fullwave.source``: uniform or a few Biot-Savart segments; or the ideal field of poloidal
  ``frame`` windings, B_phi = mu0 N I f / (2 pi rho));
* a Frequency study over the design's ``fullwave`` frequencies, solved twice by ``main()``: "imp" (port 1
  driven, Je off: Z) and "volt" (Je on, port 1 not driven: V), each exported with an EvaluationGroup to a text
  file whose rows are ``freq_MHz  re  im`` (the source_impedance.txt / source_voltage.txt convention).

Feature type names and property keys are copied from a model saved by COMSOL 6.4 as Java (a 3D impedance
study; not distributed).  Calls that the template
does not contain are marked ``UNVERIFIED`` in the generated Java and listed in docs/comsol_export.md.

Conventions (docs/comsol_export.md, "Physics"):

* axionFEM uses exp(-i omega t), COMSOL exp(+j omega t): COMSOL's complex numbers are the complex conjugates
  of axionFEM's (axionFEM Z = R - i omega L, COMSOL Z = R + j omega L).
* With Je_scale = 1 the source is Je [A/m^2] = B0 [T] numerically, i.e. axionFEM's unit source coefficient
  kappa = 1 A m^-2 T^-1: COMSOL's open-circuit port voltage is conj(V_a / kappa) = conj(-i omega Psi).
  For a "|J| = 1 A/m^2" normalisation set Je_scale = 1 / B_ref.
* COMSOL terminates a port that is not driven with its reference impedance (50 ohm by default).  The "volt"
  run therefore gives the loaded voltage; :func:`open_circuit_voltage` converts it.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

__all__ = ["PHYSICS_CHOICES", "physics_spec", "source_field", "open_circuit_voltage", "read_eval_table"]

PHYSICS_CHOICES = ("none", "fullwave")
PEC_KINDS = ("superconductor", "conductor")
MU0 = 1.25663706212e-6      # cpp/include/axionfem/units.hpp
MAX_SEGMENTS = 100          # Biot-Savart segments written inline into the Je expression (Java string limit)
Z_REF_DEFAULT = 50.0        # COMSOL's default lumped-port reference impedance [ohm]

# built-in materials (cpp/src/materials.cpp) for when the compiled core is not available
_BUILTIN = {
    "vacuum": {"kind": "vacuum"}, "air": {"kind": "vacuum"}, "winding": {"kind": "vacuum"},
    "superconductor": {"kind": "superconductor"}, "nb": {"kind": "superconductor"},
    "copper": {"kind": "conductor", "rf_model": "anomalous"}, "copper_classical": {"kind": "conductor", "rf_model": "skin"},
    "iron": {"kind": "iron", "mu_r": 1000.0}, "iron_knee": {"kind": "iron", "mu_r": 1000.0},
}

_U = "UNVERIFIED"  # marker of API calls not present in the COMSOL 6.4 template


def _num(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    r = repr(x)
    return r[:-2] if r.endswith(".0") else r


def _val(v, params) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    from .comsol import comsol_expression

    return comsol_expression(str(v), params)[0]


# =====================================================================================
# the axion source: B0 as a COMSOL expression
# =====================================================================================
# Coordinates are made dimensionless (x/1[m]) so that the expressions are plain numbers in SI; Je multiplies
# them by the dimensionless parameter Je_scale (A/m^2 per tesla).
_X, _Y, _Z = "(x/1[m])", "(y/1[m])", "(z/1[m])"


def _segments_expr(segs) -> list[str]:
    """B of straight filaments [ax,ay,az,bx,by,bz,I], with axionFEM's kernel (cpp/src/prescribed_source.cpp):
    B = 1e-7 I (a+b) / (a b (a b + u.v)) (d x u), u = p - a, v = p - b, d = b - a."""
    out = [[], [], []]
    for s in segs:
        ax, ay, az, bx, by, bz, cur = (float(t) for t in s)
        if cur == 0:
            continue
        dx, dy, dz = bx - ax, by - ay, bz - az
        ux, uy, uz = f"({_X}-({_num(ax)}))", f"({_Y}-({_num(ay)}))", f"({_Z}-({_num(az)}))"
        vx, vy, vz = f"({_X}-({_num(bx)}))", f"({_Y}-({_num(by)}))", f"({_Z}-({_num(bz)}))"
        a = f"sqrt({ux}^2+{uy}^2+{uz}^2)"
        b = f"sqrt({vx}^2+{vy}^2+{vz}^2)"
        uv = f"({ux}*{vx}+{uy}*{vy}+{uz}*{vz})"
        fac = f"{_num(1e-7 * cur)}*({a}+{b})/({a}*{b}*({a}*{b}+{uv}))"
        cross = (f"({_num(dy)}*{uz}-({_num(dz)})*{uy})", f"({_num(dz)}*{ux}-({_num(dx)})*{uz})",
                 f"({_num(dx)}*{uy}-({_num(dy)})*{ux})")
        for k in range(3):
            out[k].append(f"{fac}*{cross[k]}")
    return ["+".join(t) if t else "0" for t in out]


def _toroidal_box_expr(src: dict) -> list[str]:
    """kind=toroidal_box of cpp/src/prescribed_source.cpp: B = B0 phi_hat (profile uniform) or B0 r0/rho phi_hat
    (inverse_r) for r0 <= rho <= r1, z0 <= z <= z1; zero elsewhere (COMSOL if(), the same closed box)."""
    B0 = float(src["B0_T"])
    (r0, r1), (z0, z1) = (float(v) for v in src["r_m"]), (float(v) for v in src["z_m"])
    rho2 = f"({_X}^2+{_Y}^2)"
    rho = f"sqrt({rho2})"
    inside = (f"({rho}>={_num(r0)})&&({rho}<={_num(r1)})&&({_Z}>={_num(z0)})&&({_Z}<={_num(z1)})")
    if "phi_deg" in src and float(src["phi_deg"][1]) - float(src["phi_deg"][0]) < 360:
        p0, p1 = (float(v) for v in src["phi_deg"])
        # (atan2 in degrees - p0) mod 360 <= p1 - p0
        rel = f"mod(atan2({_Y},{_X})*180/pi-({_num(p0)})+720,360)"
        inside += f"&&({rel}<={_num(p1 - p0)})"
    # B_phi / rho: the Cartesian components are -B_phi y/rho and B_phi x/rho
    c = f"{_num(B0 * r0)}/{rho2}" if src.get("profile", "uniform") == "inverse_r" else f"{_num(B0)}/{rho}"
    return [f"if({inside},-{c}*{_Y},0)", f"if({inside},{c}*{_X},0)", "0"]


def _frame_of(region: dict):
    """(r0, r1, z0, z1, t) of a 'frame' region: 2D shape, or a 3D revolve about the global z axis."""
    if "shape" in region:
        sh = region["shape"]
    else:
        so = region.get("solid") or {}
        if set(so) - {"revolve"}:
            return None
        rv = so["revolve"]
        if [float(c) for c in rv.get("center", [0, 0, 0])] != [0, 0, 0] or \
                [float(c) for c in rv.get("axis", [0, 0, 1])] != [0, 0, 1]:
            return None
        sh = rv["profile"]
    if set(sh) != {"frame"}:
        return None
    f = sh["frame"]
    return float(f["r"][0]), float(f["r"][1]), float(f["z"][0]), float(f["z"][1]), float(f["thickness"])


def _toroidal_expr(frames) -> list[str]:
    """Ideal field of axisymmetric poloidal windings on 'frame' regions: B_phi = mu0 N I f / (2 pi rho), with f the
    fraction of the winding current enclosed by the circle through (rho, z): 1 inside the frame's hole, 0 outside
    the frame, linear across each leg (uniform current in the legs; approximate in the corners)."""
    rho = f"sqrt({_X}^2+{_Y}^2)"
    terms = []
    for (r0, r1, z0, z1, t, NI) in frames:
        legs = [f"({rho}-({_num(r0)}))", f"(({_num(r1)})-{rho})", f"({_Z}-({_num(z0)}))", f"(({_num(z1)})-{_Z})"]
        f = f"min(max(min(min({legs[0]},{legs[1]}),min({legs[2]},{legs[3]}))/({_num(t)}),0),1)"
        terms.append(f"{_num(MU0 * NI / (2 * math.pi))}*{f}")
    c = "(" + "+".join(terms) + f")/({_X}^2+{_Y}^2)"
    return [f"-{c}*{_Y}", f"{c}*{_X}", "0"]


def source_field(doc: dict, resolved: dict | None = None) -> dict:
    """The laboratory field B0 that axionFEM's 3D full-wave source overlap uses, as COMSOL expressions.

    Returns {"exported": bool, "kind", "B_T": [Bx, By, Bz] (COMSOL expressions in tesla, dimensionless
    coordinates), "B_ref_T" (a reference |B0| for the npy |J| = 1 A/m^2 convention), "description", "warnings"}.
    """
    fw = doc.get("fullwave") or {}
    params = {k: float(v) for k, v in ((resolved or {}).get("parameters") or {}).items()}
    src = fw.get("source")
    if src is not None:
        if "file" in src:
            nav = int(src.get("azimuthal_average_samples", 0) or 0)
            src = json.loads(Path(src["file"]).read_text())
            if nav:
                src = dict(src, azimuthal_average_samples=nav)
        if int(src.get("azimuthal_average_samples", 0) or 0):
            return {"exported": False, "kind": src.get("kind"),
                    "description": "fullwave.source with azimuthal_average_samples",
                    "warnings": ["source NOT exported: the azimuthally averaged prescribed source has no closed form; "
                                 "Je is 0 and only the impedance run is meaningful"]}
        if src.get("kind") == "uniform":
            B = [float(v) for v in src["B_T"]]
            return {"exported": True, "kind": "uniform", "B_T": [_num(v) for v in B], "B_ref_T": float(np.linalg.norm(B)),
                    "description": f"fullwave.source uniform B0 = {B} T (exact)", "warnings": []}
        if src.get("kind") == "segments":
            segs = src["segments_m_A"]
            if len(segs) > MAX_SEGMENTS:
                return {"exported": False, "kind": "segments",
                        "description": f"fullwave.source with {len(segs)} Biot-Savart segments",
                        "warnings": [f"source NOT exported: {len(segs)} Biot-Savart segments exceed the {MAX_SEGMENTS} that "
                                     "fit into an inline Je expression; a gridded B0 interpolation function would be "
                                     "needed. Je is 0 and only the impedance run is meaningful"]}
            return {"exported": True, "kind": "segments", "B_T": _segments_expr(segs), "B_ref_T": None,
                    "description": f"fullwave.source: {len(segs)} straight filaments, exact finite-wire Biot-Savart "
                                   "(axionFEM's kernel)", "warnings": []}
        if src.get("kind") == "toroidal_box":
            prof = src.get("profile", "uniform")
            return {"exported": True, "kind": "toroidal_box", "B_T": _toroidal_box_expr(src),
                    "B_ref_T": abs(float(src["B0_T"])),
                    "description": f"fullwave.source toroidal_box: B_phi = {src['B0_T']} T"
                                   + (" * r0/rho" if prof == "inverse_r" else "")
                                   + f" for rho in {list(src['r_m'])} m, z in {list(src['z_m'])} m"
                                   + (f", phi in {list(src['phi_deg'])} deg" if "phi_deg" in src else "")
                                   + ", 0 outside (exact)",
                    "warnings": []}
        return {"exported": False, "kind": src.get("kind"), "description": "unknown fullwave.source kind",
                "warnings": [f"source NOT exported: unknown fullwave.source kind {src.get('kind')!r}"]}
    # no prescribed source: B0 is axionFEM's static solve of the windings
    regions = {r["name"]: r for r in ((resolved or {}).get("regions") or doc.get("regions", []))}
    frames, bad = [], []
    for w in doc.get("windings", []):
        reg = regions.get(w.get("region"), {})
        fr = _frame_of(reg) if w.get("type") == "poloidal" and "ampere_turns" in w else None
        if fr is None:
            bad.append(w.get("name") or w.get("region"))
            continue
        frames.append((*fr, _val(w["ampere_turns"], params)))
    if bad or not frames:
        why = (f"windings {bad} are not poloidal 'frame' windings with ampere_turns" if bad else "the design has no windings")
        return {"exported": False, "kind": "static", "description": "B0 from axionFEM's static solve",
                "warnings": [f"source NOT exported: {why}; axionFEM takes B0 from its static solve, which has no closed "
                             "form here (give fullwave.source, or set Je in COMSOL, e.g. from an mf study). Je is 0 and "
                             "only the impedance run is meaningful"]}
    r_in = min(fr[0] + fr[4] for fr in frames)
    Bref = abs(sum(MU0 * fr[5] / (2 * math.pi) for fr in frames)) / r_in
    return {"exported": True, "kind": "toroidal", "B_T": _toroidal_expr(frames), "B_ref_T": Bref,
            "description": "ideal toroidal field of poloidal frame windings, B_phi = mu0 N I f/(2 pi rho) "
                           f"(B_ref = B_phi at the inner radius of the field box, rho = {r_in:.6g} m)",
            "warnings": ["source: the ideal toroidal field ignores superconductor screening and non-axisymmetric "
                         "features; axionFEM's B0 is its static solve (compare the MQS Psi first)"]}


# =====================================================================================
# what the physics needs from the design
# =====================================================================================

def _frequencies(fw: dict, params: dict) -> list[float]:
    if fw.get("frequencies"):
        return [_val(f, params) for f in fw["frequencies"]]
    if "f_min" in fw or "f_max" in fw:
        a, b = _val(fw["f_min"], params), _val(fw["f_max"], params)
        n = int(fw.get("points", 20))
        return [a] if n == 1 else [a * (b / a) ** (i / (n - 1)) for i in range(n)]
    return []


def physics_spec(doc: dict, resolved: dict | None = None, physics: str = "fullwave") -> dict:
    """Collect the physics of an axionFEM 3D full-wave design for the COMSOL export (raises ValueError if the
    design cannot be expressed: 2D, no port pickup, no frequencies)."""
    if physics != "fullwave":
        raise ValueError(f"physics '{physics}' is not supported; use one of {PHYSICS_CHOICES} "
                         "(an 'mqs' Magnetic Fields export is not implemented, see docs/comsol_export.md)")
    resolved = resolved or {}
    is3d = doc.get("symmetry") == "3d" or resolved.get("symmetry") == "3d"
    if not is3d:
        raise ValueError("--physics fullwave exports 3D designs only (a 2D-axisymmetric emw model with an ideal slit "
                         "has no lumped-port counterpart); convert with axionfem.convert.to_3d(design, pickup='slit')")
    pick = doc.get("pickup") or {}
    if pick.get("type") != "port":
        raise ValueError("--physics fullwave needs a 'port' pickup (a gap region) for the COMSOL LumpedPort; "
                         f"pickup is {pick!r}. Use axionfem.convert.to_3d(design, pickup='slit') for slit pickups")
    fw = doc.get("fullwave") or {}
    params = {k: float(v) for k, v in (resolved.get("parameters") or {}).items()}
    freqs = _frequencies(fw, params)
    if not freqs:
        raise ValueError("the design has no fullwave frequencies (fullwave.frequencies or f_min/f_max/points)")
    ports = [{"name": "pickup", "region": pick["port"], "toward": pick.get("toward")}]
    for p in fw.get("ports", []):
        ports.append({"name": p["name"], "region": p["region"], "toward": p.get("toward")})
    mats = dict(_BUILTIN)
    for k, v in (resolved.get("materials") or {}).items():
        mats[k] = v
    for k, v in (doc.get("materials") or {}).items():
        if isinstance(v, dict) and k not in (resolved.get("materials") or {}):
            mats[k] = v
    region_mat, region_kind = {}, {}
    for r in resolved.get("regions") or doc.get("regions", []):
        m = r.get("material", "vacuum")
        region_mat[r["name"]] = m
        region_kind[r["name"]] = r.get("kind") or mats.get(m, {}).get("kind", "vacuum")
    for p in ports:
        if p["region"] not in region_mat:
            raise ValueError(f"port region '{p['region']}' is not a region of the design")
    mesh = resolved.get("mesh") or doc.get("mesh") or {}
    src = source_field(doc, resolved)
    warnings = list(src.get("warnings", []))
    # conductor walls are exported as PEC: their rf surface resistance (R_s) is not part of the COMSOL model
    for m in sorted({region_mat[r] for r, k in region_kind.items() if k in PEC_KINDS}):
        rf_model = mats.get(m, {}).get("rf_model") or (mats.get(m, {}).get("rf") or {}).get("model")
        if rf_model:
            warnings.append(
                f"material '{m}' has rf surface-resistance data (model '{rf_model}') but its walls are exported as "
                "PerfectElectricConductor (lossless): the COMSOL Re Zport has no wall loss. COMSOL's "
                "ImpedanceBoundaryCondition (sigma) is classical skin; for an 'anomalous' wall it would underestimate "
                "R_s (copper RRR 100: by 2.4 % at 1 MHz, 23 % at 100 MHz, 40 % at 1 GHz). The axionFEM 3D full-wave Z has no "
                "wall loss either, so the comparison stays like-for-like; to add wall loss in COMSOL use a "
                "frequency-dependent surface impedance from axionfem.cavity.surface_resistance "
                "(docs/physics/surface_resistance.md)")
    return {"physics": "fullwave", "frequencies_Hz": freqs, "boundary": fw.get("boundary", "radiation"),
            "ports": ports, "source": src, "materials": mats, "region_material": region_mat,
            "region_kind": region_kind, "symmetry": list(mesh.get("symmetry") or []),
            "remove_pec": bool(mesh.get("remove_pec")), "warnings": warnings}


def material_props(m: dict) -> dict:
    """COMSOL 'def' property values of an axionFEM non-PEC material (eps_r, mu_r, sigma + omega eps0 eps_r tan_d)."""
    eps, mu = float(m.get("eps_r", 1.0)), float(m.get("mu_r", 1.0))
    sig, tan = float(m.get("sigma_S_m", 0.0)), float(m.get("tan_delta", 0.0))
    s = _num(sig)
    if tan:
        s = f"{_num(sig)}+2*pi*freq*epsilon0_const*{_num(eps)}*{_num(tan)}"
    return {"relpermittivity": _num(eps), "relpermeability": _num(mu), "electricconductivity": s,
            "frequency_dependent": bool(tan)}


# =====================================================================================
# port faces
# =====================================================================================

def port_face(lo, hi, toward) -> dict | None:
    """Face of an axis-aligned box gap for the COMSOL LumpedPort: the face that contains the current direction
    (``toward``) and is normal to the gap's thinnest other axis, on the side farther from the origin (the outer
    face of a detector).  None without ``toward``."""
    if toward is None:
        return None
    t = np.abs(np.asarray(toward, float))
    if not t.any():
        return None
    ta = int(np.argmax(t))
    ext = np.asarray(hi, float) - np.asarray(lo, float)
    others = [i for i in range(3) if i != ta]
    na = min(others, key=lambda i: ext[i])
    side = hi[na] if abs(hi[na]) >= abs(lo[na]) else lo[na]
    return {"normal_axis": "xyz"[na], "coord": float(side), "lo": [float(v) for v in lo], "hi": [float(v) for v in hi],
            "toward_axis": "xyz"[ta]}


def import_selections(info: dict, spec: dict) -> dict:
    """Explicit component selections for the .mphtxt import route, from the entity map (``entities.json``).

    COMSOL API entity numbers: domains as in the file (1-based), boundaries = file index + 1 (UNVERIFIED: COMSOL keeps
    the file's entity numbering on import).  Returns {"selections": [(tag, label, dim, [numbers])], "refs", "approx",
    "port_selection"}.
    """
    kinds = spec["region_kind"]
    doms = info["domains"]
    bnds = info["boundaries"]
    pec_dom = {d["index"] for d in doms if kinds.get(d["region"]) in PEC_KINDS}
    shell = {d["index"] for d in doms if d["region"] == "infinity_shell"}
    emw = [d["index"] for d in doms if d["index"] not in pec_dom | shell]
    emw_set = set(emw)
    sels: list = []
    approx: list[str] = []

    def add(label, dim, ents):
        tag = f"sel{len(sels) + 1}"
        sels.append((tag, label, dim, sorted(int(e) for e in ents)))
        return tag

    refs: dict = {"emw": add("emw_domains", 3, emw), "pec": [], "pmc": [], "scattering": None, "ports": [], "materials": []}
    sym_type = {str(p.get("name")): p.get("type", "pec") for p in spec["symmetry"]}

    def meta_type(b):
        return (b.get("meta") or {}).get("type") or sym_type.get(b.get("group"))

    pec_b, pmc_b, outer_b = [], [], []
    for b in bnds:
        d0, d1 = b["domains"]
        if not (d0 in emw_set or d1 in emw_set):
            continue
        other = d1 if d0 in emw_set else d0
        if other in emw_set:
            continue                                    # interior interface of the emw domains
        if b.get("group") in sym_type or meta_type(b) == "pmc":
            (pmc_b if meta_type(b) == "pmc" else pec_b).append(b["index"] + 1)
        elif other in pec_dom or b.get("group", "") and str(b.get("group")).startswith("pec_body"):
            pec_b.append(b["index"] + 1)                # conductor walls (kept or removed conductor volumes)
        else:
            outer_b.append(b["index"] + 1)              # outer sphere / domain_solid wall / infinity-shell interface
    if spec["boundary"] == "radiation":
        if outer_b:
            refs["scattering"] = add("outer (scattering)", 2, outer_b)
        approx.append("fullwave.boundary radiation: a first-order Scattering boundary condition on the outer boundary of "
                      "the air domain (axionFEM: exact outgoing-wave exterior map); use boundary 'pec' to compare")
    else:
        pec_b += outer_b
    if pec_b:
        refs["pec"].append(add("pec_walls", 2, pec_b))
    if pmc_b:
        refs["pmc"].append(add("pmc_symmetry", 2, pmc_b))
    by_mat: dict[str, list[int]] = {}
    for d in doms:
        if d["index"] in emw_set:
            m = "air" if d["region"] == "air" else spec["region_material"].get(d["region"], "vacuum")
            by_mat.setdefault(m, []).append(d["index"])
    for m, ds in by_mat.items():
        refs["materials"].append((m, add("material " + m, 3, ds)))
    port_sel = []
    for port in spec["ports"]:
        gd = {d["index"] for d in doms if d["region"] == port["region"]}
        if not gd:
            raise ValueError(f"port region '{port['region']}' has no domain in the mesh")
        # a distributed port (several disjoint gap pieces) has no single COMSOL LumpedPort
        # equivalent: the box-face selection below would put the port on one piece only
        comp = {g: g for g in gd}

        def root(g):
            while comp[g] != g:
                g = comp[g]
            return g

        for b in bnds:
            d0, d1 = b["domains"]
            if d0 in gd and d1 in gd:
                comp[root(d0)] = root(d1)
        pieces = len({root(g) for g in gd})
        if pieces > 1:
            raise ValueError(f"port '{port['name']}': region '{port['region']}' is a distributed port of {pieces} "
                             "disjoint pieces, which the COMSOL export does not support; give each piece its own "
                             "region and port and combine them in COMSOL (an ideal common bus only for pieces that a "
                             "symmetry permutes)")
        adj = [b for b in bnds if b["domains"][0] in gd or b["domains"][1] in gd]
        cand = [b for b in adj if not b["attr"] and not ({*b["domains"]} - gd) & (pec_dom | {0})]
        lo = np.min([b["bbox"][0] for b in adj], axis=0)
        hi = np.max([b["bbox"][1] for b in adj], axis=0)
        face = port_face(lo, hi, port["toward"])
        pick = []
        if face is not None:
            ai = "xyz".index(face["normal_axis"])
            tol = 1e-6 * max(float(np.max(hi - lo)), 1e-9) + 1e-9
            pick = [b for b in cand if abs(b["bbox"][0][ai] - face["coord"]) <= tol and abs(b["bbox"][1][ai] - face["coord"]) <= tol]
        if pick:
            port_sel.append(f"face {face['normal_axis']} = {face['coord']:.9g} of the gap (current along {face['toward_axis']})")
        else:
            pick = cand
            port_sel.append("all gap walls that face non-PEC domains (UNVERIFIED: non-planar lumped port)")
            approx.append(f"port '{port['name']}': no planar gap face containing 'toward' was found (export with "
                          "--feature-angle to split the gap walls); the LumpedPort is put on all its non-PEC walls")
        if not pick:
            raise ValueError(f"port region '{port['region']}' has no wall towards a non-PEC domain")
        refs["ports"].append(add("port_" + port["name"], 2, [b["index"] + 1 for b in pick]))
    approx.append("ports: axionFEM feeds the current through the gap volume (Laplace profile); the COMSOL LumpedPort "
                  "carries it on one face of the gap, a modelling difference of order (gap size / loop size)")
    if shell:
        approx.append("the infinity-shell domain is left out of the emw interface")
    return {"selections": sels, "refs": refs, "approx": approx, "port_selection": port_sel}


def selection_java(sels) -> list[str]:
    """Explicit component selections (template: selection().create(.., "Explicit"), geom(dim), set(ints))."""
    out = []
    for tag, label, dim, ents in sels:
        S = f'{C}.selection("{tag}")'
        out.append(f'{C}.selection().create("{tag}", "Explicit");')
        out.append(f'{S}.label({json.dumps(label)});')
        out.append(f'{S}.geom({dim});')
        for i in range(0, max(len(ents), 1), 2000):
            chunk = ents[i:i + 2000]
            out.append(f'{S}.{"set" if i == 0 else "add"}(new int[]{{{", ".join(str(e) for e in chunk)}}});'
                       + ("" if i == 0 else f"  // {_U}: add(int[])"))
    return out


# =====================================================================================
# Java emission (shared by the parametric Java route and the .mphtxt import route)
# =====================================================================================

C = 'model.component("comp1")'
P = C + '.physics("emw")'


def _sarr(v) -> str:
    return "new String[]{" + ", ".join(json.dumps(x) for x in v) + "}"


def _diag(v: str) -> str:
    return _sarr([v, "0", "0", "0", v, "0", "0", "0", v])


def physics_java(spec: dict, refs: dict, *, imp_file: str, volt_file: str) -> dict:
    """Java statements for the emw model.

    ``refs``: selection references (strings accepted by ``selection().named(...)``):
    ``emw`` (domains of the emw interface), ``materials`` [(material name, ref)], ``pec`` [refs], ``pmc`` [refs],
    ``scattering`` ref or None, ``ports`` [ref per spec port].
    Returns {"setup": [...], "imp": [...], "volt": [...]} (the setup goes into run(); imp/volt are the two solves).
    """
    L = []
    w = L.append
    src = spec["source"]
    f_mhz = ",".join(_num(round(f / 1e6, 12)) for f in spec["frequencies_Hz"])
    w('model.param().set("Je_scale", "1", "Je [A/m^2] per tesla of axionFEM\'s B0 (1: axionFEM kappa = 1 A/m^2/T; '
      '1/B_ref: the npy |J| = 1 A/m^2 convention)");')
    w(f'{C}.mesh().create("mesh1");')
    w(f'{C}.physics().create("emw", "ElectromagneticWaves", "geom1");')
    w(f'{P}.selection().named({json.dumps(refs["emw"])});  // {_U}: selection().named')
    # materials
    for k, (mname, ref) in enumerate(refs["materials"]):
        m = spec["materials"].get(mname, {"kind": "vacuum"})
        pr = material_props(m)
        t = f"mat{k + 1}"
        M = f'{C}.material("{t}")'
        w(f'{C}.material().create("{t}", "Common");')
        w(f'{M}.label({json.dumps("axionFEM " + mname)});')
        for key in ("relpermittivity", "relpermeability", "electricconductivity"):
            w(f'{M}.propertyGroup("def").set("{key}", {_diag(pr[key])});')
        if pr["frequency_dependent"]:
            w(f'{M}.propertyGroup("def").addInput("frequency");  // {_U}: addInput on "def" (template: on LossTangentDF)')
        w(f'{M}.selection().named({json.dumps(ref)});  // {_U}: selection().named')
    # boundary conditions (COMSOL's default pec1 already covers every exterior boundary of the emw domains)
    for k, ref in enumerate(refs["pec"]):
        t = f"pec{k + 2}"
        w(f'{P}.create("{t}", "PerfectElectricConductor", 2);')
        w(f'{P}.feature("{t}").selection().named({json.dumps(ref)});  // {_U}: selection().named')
    for k, ref in enumerate(refs["pmc"]):
        t = f"pmc{k + 1}"
        w(f'{P}.create("{t}", "PerfectMagneticConductor", 2);  // {_U}: feature type PerfectMagneticConductor')
        w(f'{P}.feature("{t}").selection().named({json.dumps(ref)});  // {_U}: selection().named')
    if refs.get("scattering"):
        w(f'{P}.create("sctr1", "Scattering", 2);  // {_U}: feature type Scattering (first-order absorbing, approximate)')
        w(f'{P}.feature("sctr1").selection().named({json.dumps(refs["scattering"])});  // {_U}: selection().named')
    for k, (port, ref) in enumerate(zip(spec["ports"], refs["ports"])):
        t = f"lport{k + 1}"
        w(f'{P}.create("{t}", "LumpedPort", 2);')
        w(f'{P}.feature("{t}").label({json.dumps("port " + str(k + 1) + ": axionFEM " + port["name"] + " (" + port["region"] + ")")});')
        w(f'{P}.feature("{t}").selection().named({json.dumps(ref)});  // {_U}: selection().named')
        w(f'{P}.feature("{t}").set("PortExcitation", "off");')
        if k > 0:
            w(f'{P}.feature("{t}").active(false);  // open circuit, as in axionFEM\'s Z matrix; activate to load it with Z_ref')
    # axion source
    w(f'{P}.create("ecd1", "ExternalCurrentDensity", 3);')
    w(f'{P}.feature("ecd1").label("axion effective current Je = Je_scale*B0");')
    w(f'{P}.feature("ecd1").selection().named({json.dumps(refs["emw"])});  // {_U}: selection().named')
    if src.get("exported"):
        je = [f"Je_scale*({b})" for b in src["B_T"]]
        w(f'{P}.feature("ecd1").set("Je", {_sarr(je)});  // {_U}: String[] form (template: new int[]{{0, 0, 1}})')
    else:
        w(f'{P}.feature("ecd1").set("Je", new int[]{{0, 0, 0}});  // source NOT exported: set Je yourself')
    # study and solver (copied from the template)
    S = 'model.study("std1")'
    w('model.study().create("std1");')
    w(f'{S}.create("freq", "Frequency");')
    w(f'{S}.feature("freq").set("solnum", "auto");')
    w(f'{S}.feature("freq").set("notsolnum", "auto");')
    w(f'{S}.feature("freq").activate("emw", true);')
    w(f'{S}.feature("freq").set("punit", "MHz");')
    w(f'{S}.feature("freq").set("plist", {json.dumps(f_mhz)});')
    O = 'model.sol("sol1")'
    s1 = f'{O}.feature("s1")'
    i1 = f'{s1}.feature("i1")'
    mg = f'{i1}.feature("mg1")'
    for ln in [
        'model.sol().create("sol1");', f'{O}.study("std1");',
        f'{O}.create("st1", "StudyStep");', f'{O}.feature("st1").set("study", "std1");',
        f'{O}.feature("st1").set("studystep", "freq");', f'{O}.create("v1", "Variables");',
        f'{O}.feature("v1").set("control", "freq");', f'{O}.create("s1", "Stationary");',
        f'{s1}.set("stol", 0.01);', f'{s1}.create("p1", "Parametric");', f'{s1}.feature().remove("pDef");',
        f'{s1}.feature("p1").set("pname", new String[]{{"freq"}});',
        f'{s1}.feature("p1").set("plistarr", new String[]{{{json.dumps(f_mhz)}}});',
        f'{s1}.feature("p1").set("punit", new String[]{{"MHz"}});',
        f'{s1}.feature("p1").set("pcontinuationmode", "no");', f'{s1}.feature("p1").set("preusesol", "auto");',
        f'{s1}.feature("p1").set("pdistrib", "off");', f'{s1}.feature("p1").set("plot", "off");',
        f'{s1}.feature("p1").set("probesel", "all");', f'{s1}.feature("p1").set("probes", new String[]{{}});',
        f'{s1}.feature("p1").set("control", "freq");', f'{s1}.set("control", "freq");',
        f'{s1}.feature("aDef").set("complexfun", true);', f'{s1}.create("fc1", "FullyCoupled");',
        f'{s1}.create("i1", "Iterative");', f'{i1}.set("linsolver", "gmres");', f'{i1}.set("prefuntype", "right");',
        f'{i1}.set("itrestart", "300");', f'{i1}.label("Suggested Iterative Solver (emw)");',
        f'{i1}.create("mg1", "Multigrid");', f'{mg}.set("iter", "1");',
        f'{mg}.feature("pr").create("sv1", "SORVector");', f'{mg}.feature("pr").feature("sv1").set("prefun", "sorvec");',
        f'{mg}.feature("pr").feature("sv1").set("sorvecdof", new String[]{{"comp1_E"}});',
        f'{mg}.feature("po").create("sv1", "SORVector");', f'{mg}.feature("po").feature("sv1").set("prefun", "soruvec");',
        f'{mg}.feature("po").feature("sv1").set("sorvecdof", new String[]{{"comp1_E"}});',
        f'{mg}.feature("cs").create("d1", "Direct");', f'{mg}.feature("cs").feature("d1").set("linsolver", "pardiso");',
        f'{s1}.feature("fc1").set("linsolver", "i1");', f'{s1}.feature().remove("fcDef");', f'{O}.attach("std1");',
    ]:
        w(ln)
    # evaluation groups: [freq_MHz, re, im] text files (template: eg1 with storetable onfile)
    for eg, var, fname, lab in (("eg1", "Zport_1", imp_file, "port 1 impedance (imp run)"),
                                ("eg2", "Vport_1", volt_file, "port 1 voltage (volt run, port loaded by Z_ref)")):
        E = f'model.result().evaluationGroup("{eg}")'
        w(f'model.result().evaluationGroup().create("{eg}", "EvaluationGroup");')
        w(f'{E}.label({json.dumps(lab)});')
        w(f'{E}.set("storetable", "onfile");')
        w(f'{E}.set("filename", {fname});')
        w(f'{E}.create("gev1", "EvalGlobal");')
        w(f'{E}.feature("gev1").setIndex("expr", "real(comp1.emw.{var})", 0);')
        w(f'{E}.feature("gev1").setIndex("expr", "imag(comp1.emw.{var})", 1);')
    imp = [
        f'{P}.feature("ecd1").active(false);',
        f'{P}.feature("lport1").set("PortExcitation", "on");',
        f'{C}.mesh("mesh1").run();',
        'model.sol("sol1").runAll();',
        'model.result().evaluationGroup("eg1").run();',
        f'model.result().evaluationGroup("eg1").save({imp_file});',
    ]
    volt = [
        f'{P}.feature("ecd1").active(true);',
        f'{P}.feature("lport1").set("PortExcitation", "off");',
        'model.sol("sol1").runAll();',
        'model.result().evaluationGroup("eg2").run();',
        f'model.result().evaluationGroup("eg2").save({volt_file});',
    ]
    return {"setup": L, "imp": imp, "volt": volt if src.get("exported") else []}


def physics_notes(spec: dict) -> list[str]:
    src = spec["source"]
    f = ", ".join(f"{x / 1e6:.6g}" for x in spec["frequencies_Hz"])
    notes = [
        "PHYSICS (--physics fullwave): Electromagnetic Waves, Frequency Domain (emw), exp(+j omega t): complex values are",
        "  the complex conjugates of axionFEM's (exp(-i omega t)); axionFEM Z = R - i omega L, COMSOL Z = R + j omega L.",
        f"  frequencies [MHz]: {f}; outer boundary: {spec['boundary']} (axionFEM fullwave.boundary)",
        "  ports: " + "; ".join(f"{k + 1} = {p['name']} ({p['region']}){'' if k == 0 else ' [inactive: open]'}"
                                for k, p in enumerate(spec["ports"])),
        f"  axion source Je = Je_scale*B0: {src.get('description')}",
        "  main(): 'imp' run (port 1 driven, Je off) -> Zport_1; 'volt' run (Je on, port 1 not driven, so loaded by its",
        "  reference impedance, 50 ohm by default) -> Vport_1. Open-circuit V = Vport (Zport + Z_ref)/Z_ref.",
        "  Calls marked UNVERIFIED are not in the COMSOL 6.4 template this export copies (docs/comsol_export.md).",
    ]
    notes += [f"WARNING: {x}" for x in spec["warnings"]]
    return notes


# =====================================================================================
# reading COMSOL's output and converting it
# =====================================================================================

def read_eval_table(path) -> np.ndarray:
    """Read an EvaluationGroup text file (header lines start with %) as a (3, N) array [freq_MHz, re, im]."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        rows.append([float(t) for t in s.replace(",", " ").split()[:3]])
    return np.array(rows, float).T


def open_circuit_voltage(V_port, Z_port, Z_ref: float = Z_REF_DEFAULT):
    """Open-circuit voltage from COMSOL's 'volt' run, where port 1 is terminated by Z_ref (both in COMSOL's
    exp(+j omega t) convention): V_oc = V_port (Z_port + Z_ref) / Z_ref."""
    return np.asarray(V_port) * (np.asarray(Z_port) + Z_ref) / Z_ref
