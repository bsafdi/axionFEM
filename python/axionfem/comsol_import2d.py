"""Mesh route for COMSOL 2D and 2D-axisymmetric Electromagnetic Waves models.  Beta.

Called by :func:`axionfem.comsol_import.import_comsol` when the emw interface lives on a 2D geometry.  The stored
COMSOL mesh (triangles; quadrilaterals are split) is written as a second-order gmsh mesh whose boundary mid-nodes
lie on fitted lines/circles, and the design runs the 2D imported-mesh engine (cpp/src/fw2d.cpp; ``"symmetry":
"axisymmetric"`` with ``mesh.file`` or ``"symmetry": "planar"``).  Conventions as the 3D route (docs/comsol_import.md):
axionFEM's Z is the complex conjugate of COMSOL's Zport; J_e [A/m^2] -> B0 [T] of ``fullwave.source``.

Families: the in-plane E family is driven by lumped ports (axisymmetric TM = (E_r, E_z); planar TE = (E_x, E_y));
the out-of-plane family (axisymmetric TE = E_phi; planar TM = E_z) has eigenmodes only.  COMSOL 2D-axisymmetric
coordinates are (r, z) with vector components (r, phi, z).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import comsol_mphbin as mb
from .comsol_import import (BOUNDARY_ATTR0, PORT_ATTR0, REGION_ATTR0, ImportError_, ImportReport, MeshRoute,
                            _feat_desc, _num, _slug, boundary_materials, choose_frequencies, classify_source,
                            domain_materials, material_properties, pml_layers, robin_feature, sel_entities,
                            study_frequencies)
from .comsol_model import ComsolModel, ExpressionError


def _components(ph: dict) -> str:
    v = str(ph["props"].get("components") or ph["props"].get("Components") or "").lower()
    if "out" in v:
        return "outofplane"
    if "in" in v and "three" not in v:
        return "inplane"
    if "three" in v:
        return "threecomponent"
    return ""


def fit_curve(X: np.ndarray, rtol: float = 2e-4):
    """Line or circle through the 2D points X: ("line", None), ("circle", (cx, cy, R)) or (None, reason)."""
    if len(X) < 3:
        return None, "two vertices (a single edge): the curve is not determined"
    c = X.mean(axis=0)
    sc = float(np.abs(X - c).max()) or 1.0
    Y = (X - c) / sc
    sv = np.linalg.svd(Y - Y.mean(axis=0), compute_uv=False)
    if sv[-1] <= 1e-9 * sv[0] * math.sqrt(len(Y)):
        return "line", None
    A = np.column_stack([Y[:, 0], Y[:, 1], np.ones(len(Y))])
    sol, *_ = np.linalg.lstsq(A, -(Y ** 2).sum(axis=1), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    R2 = cx * cx + cy * cy - sol[2]
    if R2 <= 0:
        return None, "no circle"
    R = math.sqrt(R2)
    res = np.abs(np.hypot(Y[:, 0] - cx, Y[:, 1] - cy) - R).max()
    if res > rtol:
        return None, f"neither a line nor a circle (circle residual {res * sc:.2e} m)"
    return "circle", (c[0] + sc * cx, c[1] + sc * cy, sc * R)


def build_mesh_route_2d(model: ComsolModel, report: ImportReport, out_dir: Path, stem: str, ph: dict, *,
                        frequencies=None, max_frequencies: int = 20, study: str | None = None, order: int | None = None,
                        ports_toward=None, curved: bool = True, mesh_resource: str | None = None,
                        axisymmetric: bool | None = None) -> MeshRoute:
    from .fw2d import write_mesh

    t0 = time.time()
    geom = ph["geom"]
    g = model.geometries[geom]
    axi = axisymmetric if axisymmetric is not None else model.axisymmetric(geom)
    if axi is None:
        raise ImportError_(f"geometry {geom} is 2D but the importer cannot tell whether it is axisymmetric (no "
                           "spatial coordinates in the model node); pass axisymmetric=True/False "
                           "(--axisymmetric / --planar)")
    report.add("geometry", f"{geom} (2D)", "mapped", "symmetry " + ("axisymmetric" if axi else "planar"),
               "COMSOL 2D " + ("axisymmetric component (r, phi, z)" if axi else "component (x, y)") +
               (" (from the user)" if axisymmetric is not None else " (from the component's spatial coordinates)"))
    if (g.get("lengthUnit") or "m") != "m":
        raise ImportError_(f"geometry {geom} uses the length unit {g['lengthUnit']}; re-save the model with length unit m")
    if ph["props"].get("SolveFor", "fullField") != "fullField":
        raise ImportError_(f"emw {ph['tag']} solves for the scattered field: only the full-field formulation is mapped")
    comps = _components(ph)
    report.add("physics", f"{ph['tag']} field components", "info",
               detail=f"electric field components solved for: {comps or 'not stored (assumed three-component)'}")
    comps = comps or "threecomponent"
    for key, v in ph["props"].items():
        kl = key.lower()
        if any(t in kl for t in ("mfloquet", "azimuthalmode", "modenumber", "outofplanewavenumber", "kz")):
            try:
                val = float(np.real(_num(model, v))) if isinstance(v, str) else 0.0
            except ExpressionError:
                val = float("nan")
            if val != 0:
                raise ImportError_(f"emw {ph['tag']}: {key} = {v}: only m = 0 (axisymmetric) / k_z = 0 (planar) fields "
                                   "are supported")
    thickness = 1.0
    if not axi:
        dv = ph["props"].get("d")
        if dv is not None:
            try:
                thickness = float(np.real(_num(model, dv)))
            except ExpressionError:
                raise ImportError_(f"emw {ph['tag']}: out-of-plane thickness d = {dv} could not be evaluated")
        report.add("physics", f"{ph['tag']} thickness", "mapped", "fullwave.thickness_m",
                   f"out-of-plane thickness D = {thickness:g} m (3D quantities are per D)")
    # ---- mesh
    meshes = [m for m in model.meshes.values() if m["geom"] == geom and m["resource"]]
    if not meshes:
        raise ImportError_(f"no stored mesh for geometry {geom}: save the .mph with its mesh (Mesh > Build All)")
    msq = dict(meshes[0])
    if mesh_resource:
        msq["resource"] = mesh_resource
    mesh = mb.read_mesh(model.archive.read(msq["resource"]), msq["resource"])
    report.add("mesh", f"{msq['tag']} -> {msq['resource']}", "mapped", "mesh.file",
               f"decoded Mesh v{mesh.version}: {len(mesh.points)} vertices, " +
               ", ".join(f"{len(v)} {k}" for k, v in mesh.elements.items()))
    if mesh.sdim != 2:
        raise ImportError_(f"mesh {msq['resource']} has space dimension {mesh.sdim}, expected 2")
    for k in mesh.elements:
        if k not in ("vtx", "edg", "tri", "quad"):
            raise ImportError_(f"2D mesh element type {k} is not supported (linear triangles and quadrilaterals)")
    P = mesh.points
    tris = [mesh.elements.get("tri", np.zeros((0, 3), int))]
    tdom = [mesh.entities.get("tri", np.zeros(0, int))]
    if "quad" in mesh.elements:   # COMSOL quad numbering (0,1,3,2) counter-clockwise is 0-1-3-2: split along 0-3
        q = mesh.elements["quad"]
        # COMSOL's quadrilateral vertex order is 0-1-3-2 around the element (tensor numbering): split along 0-3,
        # and check that the two triangles tile the quadrilateral (equal |area| sum to the shoelace area)
        cyc = P[q[:, [0, 1, 3, 2]]]
        shoe = 0.5 * np.abs(np.einsum("ni,ni->n", cyc[:, :, 0], np.roll(cyc[:, :, 1], -1, axis=1))
                            - np.einsum("ni,ni->n", cyc[:, :, 1], np.roll(cyc[:, :, 0], -1, axis=1)))

        def tri_area(t):
            a = P[t]
            return 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                                - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))
        if not np.allclose(tri_area(q[:, [0, 1, 3]]) + tri_area(q[:, [0, 3, 2]]), shoe, rtol=1e-9, atol=0):
            raise ImportError_("quadrilateral elements: the assumed COMSOL vertex order (0-1-3-2 around the element) "
                               "does not tile the elements; report this file")
        tris += [q[:, [0, 1, 3]], q[:, [0, 3, 2]]]
        tdom += [mesh.entities["quad"]] * 2
        report.add("mesh", "quadrilaterals", "approximated", "triangles",
                   f"{len(q)} quadrilaterals split into two triangles each (same nodes; the field space differs "
                   "from COMSOL's quadrilateral elements at discretization level)")
    T = np.concatenate(tris)
    Td = np.concatenate(tdom)
    E = mesh.elements.get("edg", np.zeros((0, 2), int))
    Eb = mesh.entities.get("edg", np.zeros(0, int)) + 1   # GUI boundary numbers
    counts = dict(g["counts"])
    ndom = counts.get(2) or (msq["numEntities"][2] if msq["numEntities"] else int(Td.max()))
    nbnd = counts.get(1) or (msq["numEntities"][1] if msq["numEntities"] else int(Eb.max()))
    phys_doms = sorted(sel_entities(ph["selection"], range(1, ndom + 1), f"emw interface {ph['tag']}"))
    missing = [d for d in phys_doms if d not in set(np.unique(Td).tolist())]
    if missing:
        raise ImportError_(f"emw domains {missing} have no elements in the stored mesh")
    excluded = sorted(set(range(1, ndom + 1)) - set(phys_doms))
    keep = np.isin(Td, phys_doms)
    T, Td = T[keep], Td[keep]
    a = P[T]
    area = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    if np.any(np.abs(area) < 1e-300):
        raise ImportError_("degenerate triangles in the stored mesh")
    T[area < 0] = T[area < 0][:, [0, 2, 1]]
    if axi and P[:, 0].min() < -1e-9 * np.ptp(P):
        raise ImportError_("axisymmetric mesh with r < 0 vertices")
    # ---- edge adjacency
    nv = len(P)
    te = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    tk = te[:, 0].astype(np.int64) * nv + te[:, 1]
    uk, cnt = np.unique(tk, return_counts=True)
    Es = np.sort(E, axis=1)
    ek = Es[:, 0].astype(np.int64) * nv + Es[:, 1]
    pos = np.clip(np.searchsorted(uk, ek), 0, max(len(uk) - 1, 0))
    eadj = np.where(uk[pos] == ek, cnt[pos], 0)
    if not np.isin(uk[cnt == 1], ek).all():
        raise ImportError_("exterior edges of the emw domains without a boundary element in the stored mesh")
    bset_ext = set(int(b) for b in np.unique(Eb[eadj == 1]))
    bset_int = set(int(b) for b in np.unique(Eb[eadj == 2]))
    bset_all = bset_ext | bset_int
    # ---- features
    frq = study_frequencies(model, report, study)
    fl = frequencies if frequencies is not None else choose_frequencies(frq["frequencies"], max_frequencies)
    btype: dict = {b: ("pec", "pec1 (default: exterior)") for b in bset_ext}
    if axi:   # boundaries on the symmetry axis (COMSOL's Axial Symmetry, added by default)
        tol = 1e-9 * max(np.ptp(P), 1e-300)
        for b in bset_ext:
            if np.all(np.abs(P[np.unique(E[Eb == b])][:, 0]) <= tol):
                btype[b] = ("axis", "axi1 (Axial Symmetry)")
    dmat = domain_materials(model, geom, ndom, report, sdim=2)
    bmat = boundary_materials(model, geom, 2, nbnd)
    wee_of, gparams, materials, lports, sources = {}, {}, {}, [], []
    for f in ph["features"]:
        if f.op == "WaveEquationElectric" and not f.disabled:
            for d in sel_entities(f.selection, phys_doms, _feat_desc(f)):
                wee_of[d] = f
    for f in ph["features"]:
        desc = _feat_desc(f)
        if f.disabled and f.op != "ExternalCurrentDensity":
            report.add("physics", desc, "ignored", detail="disabled in the saved model")
            continue
        if f.selection is not None and f.selection.unknown:
            raise ImportError_(f"{desc}: its selection is not an explicit or named selection the importer understands")
        ents = f.selection.entities if f.selection else None
        if f.op in ("WaveEquationElectric",):
            report.add("physics", desc, "mapped", "materials", "constitutive relations per domain (see materials)")
        elif f.op in ("init", "InitialValues"):
            report.add("physics", desc, "ignored", detail="initial values do not affect a linear frequency-domain solve")
        elif f.op in ("AxialSymmetry", "Axisymmetry"):
            report.add("physics", desc, "mapped", "boundary_groups type axis", "boundaries on r = 0")
        elif f.op == "PerfectElectricConductor":
            if ents is None and f.default:
                report.add("physics", desc, "mapped", "boundary_groups type pec", "all exterior boundaries")
                continue
            for b in (ents if ents is not None else sorted(bset_all)):
                if btype.get(b, ("",))[0] != "axis":
                    btype[b] = ("pec", f.tag)
            report.add("physics", desc, "mapped", "boundary_groups type pec", f"boundaries {ents}")
        elif f.op == "PerfectMagneticConductor":
            pm = ents if ents is not None else sorted(bset_ext)
            if set(pm) & (bset_int - bset_ext):
                raise ImportError_(f"{desc}: PMC on interior boundaries is not supported")
            for b in pm:
                btype[b] = ("pmc", f.tag)
            report.add("physics", desc, "mapped", "boundary_groups type pmc", f"boundaries {pm}")
        elif f.op == "LumpedPort":
            if ents is None:
                raise ImportError_(f"{desc}: lumped port without an explicit boundary list")
            ptype = f.p("PortType", "Uniform")
            if ptype not in ("Uniform",) and not (axi and ptype == "Coaxial"):
                report.add("physics", desc, "unsupported", detail=f"PortType {ptype}: only Uniform lumped ports "
                           "(and Coaxial ones in 2D axisymmetry) map to the 2D boundary port; User defined 2D ports "
                           "have the field out of the plane", changes_results=True)
                continue
            for b in ents:
                btype[b] = ("port", f.tag)
            lports.append(f)
        elif f.op == "ExternalCurrentDensity":
            sources.append(f)
        elif f.op in ("ImpedanceBoundaryCondition", "Impedance", "TransitionBoundaryCondition", "Scattering",
                      "ScatteringBoundaryCondition"):
            if ents is None:
                ents = sorted(bset_ext) if f.op != "TransitionBoundaryCondition" else sorted(bset_int)
            try:
                gparams[f.tag] = robin_feature(model, f, ents, bset_ext, bset_int, bmat, fl, report, materials)
            except (ImportError_, ExpressionError) as e:
                report.add("physics", desc, "unsupported", detail=str(e), changes_results=True)
                continue
            if gparams[f.tag]["type"] == "scattering" and gparams[f.tag]["wave"] == "spherical" and not axi:
                report.add("physics", desc, "unsupported", detail="spherical scattered wave in a planar 2D model",
                           changes_results=True)
                continue
            for b in ents:
                btype[b] = (gparams[f.tag]["type"], f.tag)
        elif f.op in ("Continuity",):
            report.add("physics", desc, "ignored", detail="the imported mesh is conforming (no identity pairs)")
        else:
            report.add("physics", desc, "unsupported", detail=f"emw feature {f.op} is not mapped by the 2D route",
                       changes_results=True)
    # ---- materials and regions
    region_of, regions, mat_key = {}, [], {}
    for d in phys_doms:
        mat = dmat.get(d)
        try:
            props = material_properties(model, mat, wee_of.get(d), fl, report)
        except (ImportError_, ExpressionError) as e:
            raise ImportError_(f"domain {d}: {e}") from e
        key = (round(props["eps_r"], 12), round(props["tan_delta"], 14), round(props["mu_r"], 12),
               round(props["sigma_S_m"], 9))
        if key == (1.0, 0.0, 1.0, 0.0):
            mname = "vacuum"
        else:
            if key not in mat_key:
                nm = "comsol_" + _slug((mat["name"] if mat else "material") + "_" + (mat["tag"] if mat else ""))
                base, k = nm, 2
                while nm in materials:
                    nm, k = f"{base}_{k}", k + 1
                mat_key[key] = nm
                materials[nm] = {"kind": "linear", "eps_r": props["eps_r"], "mu_r": props["mu_r"],
                                 "tan_delta": props["tan_delta"], "sigma_S_m": props["sigma_S_m"]}
            mname = mat_key[key]
        status = "approximated" if props["frequency_dependence"] else "mapped"
        report.add("material", f"domain {d} <- {(mat['tag'] + ' (' + mat['name'] + ')') if mat else 'no material'}",
                   status, mname, f"eps_r {props['eps_r']:.6g}, tan_delta {props['tan_delta']:.3g}, mu_r "
                   f"{props['mu_r']:.6g}, sigma {props['sigma_S_m']:.3g} S/m ({props['model']})",
                   changes_results=False)
        region_of[d] = f"domain_{d}"
        regions.append({"name": f"domain_{d}", "material": mname,
                        "shape": {"imported": {"comsol_domains": [d], "comsol_material": mat["tag"] if mat else None}}})
    attr_of_dom = {d: REGION_ATTR0 + i for i, d in enumerate(phys_doms)}
    # ---- source
    source, source_regions = None, []
    chosen = [s for s in sources if not s.disabled] or sources
    if len(chosen) > 1:
        report.add("physics", ", ".join(s.tag for s in chosen[1:]), "unsupported",
                   detail="several External Current Density features: only the first is mapped", changes_results=True)
    in_plane = comps in ("inplane", "threecomponent")
    if chosen:
        s = chosen[0]
        je = s.params.get("Je")
        ents = [d for d in sel_entities(s.selection, phys_doms, _feat_desc(s)) if d in phys_doms]
        try:
            if not ents:
                raise ImportError_("no emw domain in its selection")
            sel = np.isin(Td, ents)
            pts = np.vstack([P[T[sel]].mean(axis=1), P[np.unique(T[sel])]])
            if axi:
                src = _classify_axisymmetric(model, je, pts)
            else:
                src = classify_source(model, je, np.column_stack([pts, np.zeros(len(pts))]))
                src.pop("exact", None)
            fam_in = (src["kind"] == "uniform" and (src["B_T"][0] != 0 or (src["B_T"][2] != 0 if axi else src["B_T"][1] != 0))) \
                or (src["kind"] == "toroidal_box" and not axi)
            source = {**src, "metadata": {"comsol_feature": s.tag, "Je_A_m2": je, "domains": ents,
                                          "normalization": "B0 [T] = Je [A/m^2] / (1 A m^-2 T^-1)",
                                          "components": "(r, phi, z) at phi = 0" if axi else "(x, y, z)"}}
            source_regions = [region_of[d] for d in ents]
            report.add("physics", _feat_desc(s), "mapped", f"fullwave.source ({src['kind']}) + source_regions",
                       f"Je = {je} on domains {ents}" + ("" if fam_in else " — out-of-plane: it does not couple to the "
                       "in-plane (port) family, Psi = 0") + (" — DISABLED in the saved model; mapped anyway" if s.disabled else ""))
        except (ImportError_, ExpressionError) as e:
            # a disabled source (an impedance run) does not change Z: report it, do not block the import
            report.add("physics", _feat_desc(s), "unsupported", detail=str(e) + (" (disabled in the saved model: "
                       "Z is unaffected, no Psi)" if s.disabled else ""), changes_results=not s.disabled)
    # ---- groups and port
    groups = []
    for b in sorted(bset_all):
        t = btype.get(b)
        if t is None or t[0] == "port":
            continue
        groups.append({"attr": BOUNDARY_ATTR0 + b, "name": f"comsol_boundary_{b}", "type": t[0], "comsol_boundary": b,
                       "feature": t[1], "exterior": b in bset_ext, **gparams.get(t[1], {})})
    eattr = BOUNDARY_ATTR0 + Eb
    pickup, port_info = None, None
    if len(lports) > 1:
        report.add("physics", ", ".join(p.tag for p in lports[1:]), "unsupported",
                   detail="several lumped ports: only the first is imported", changes_results=True)
    if lports:
        lp = lports[0]
        pb = [b for b in lp.selection.entities if btype.get(b, ("",))[0] == "port"]
        eattr = np.where(np.isin(Eb, pb), PORT_ATTR0 + 1, eattr)
        groups.append({"attr": PORT_ATTR0 + 1, "name": lp.tag, "type": "port", "comsol_boundaries": pb, "feature": lp.tag})
        try:
            zref = float(np.real(_num(model, lp.p("Zref", "50[ohm]"))))
        except ExpressionError:
            zref = 50.0
        mask = np.isin(Eb, pb) & (eadj > 0)
        ends = np.unique(E[mask], return_counts=True)
        pend = ends[0][ends[1] == 1]
        port_info = {"comsol_boundaries": pb, "Zref_ohm": zref, "PortExcitation": lp.p("PortExcitation"),
                     "edges": int(mask.sum()), "ends_m": P[pend].tolist(),
                     "length_m": float(np.linalg.norm(P[E[mask][:, 1]] - P[E[mask][:, 0]], axis=1).sum())}
        if not in_plane:
            report.add("physics", _feat_desc(lp), "unsupported", detail="lumped port with out-of-plane field "
                       "components only: axionFEM's 2D ports drive the in-plane family", changes_results=True)
        else:
            toward = list(ports_toward) if ports_toward is not None else None
            pickup = {"type": "port", "port_boundary": lp.tag,
                      "notes": f"COMSOL {lp.tag}: Uniform lumped port on boundaries {pb}, Zref {zref:g} ohm"}
            if toward:
                pickup["toward"] = toward + [0.0] * (3 - len(toward))
            ptype = lp.p("PortType", "Uniform")
            pe = P[E[mask]]
            rspan = float(np.ptp(pe[:, :, 0])) if len(pe) else 0.0
            const_r = (not axi) or rspan <= 1e-9 * max(port_info["length_m"], 1e-300)
            status = "mapped" if (ptype == "Coaxial" or const_r) else "approximated"
            detail = (f"{ptype} lumped port on boundaries {pb}, length {port_info['length_m']:.4g} m: current I "
                      + ("uniform around the circumference, K = I/(2 pi r) t" if axi else "K = I/D t") +
                      f", V = -int t.E dl. Zref {zref:g} ohm is not part of axionFEM's Z")
            if status == "approximated":
                detail += ("; the port edge spans r = " + f"{pe[:, :, 0].min():.4g}..{pe[:, :, 0].max():.4g} m: axionFEM "
                           "keeps the current per circle uniform (K ~ 1/r, as COMSOL's Coaxial port), COMSOL's Uniform "
                           "port definition there is not verified")
            report.add("physics", _feat_desc(lp), status, "pickup.port_boundary (2D boundary port)", detail)
    elif fl:
        report.add("physics", "(no lumped port)", "unsupported", detail="a 2D frequency sweep needs a lumped port",
                   changes_results=True)
    # ---- mesh with curved boundaries
    keep_e = eadj > 0
    E2, A2, B2 = E[keep_e], eattr[keep_e], Eb[keep_e]
    used = np.unique(np.concatenate([T.ravel(), E2.ravel()]))
    remap = -np.ones(nv, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts = P[used]
    T2, E2 = remap[T], remap[E2]
    # unique edges of the triangles, mid nodes
    te = np.stack([np.sort(T2[:, [0, 1]], axis=1), np.sort(T2[:, [1, 2]], axis=1), np.sort(T2[:, [2, 0]], axis=1)], axis=1)
    keys = te[:, :, 0].astype(np.int64) * len(pts) + te[:, :, 1]
    uk, inv = np.unique(keys.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 3)
    mid = 0.5 * (pts[uk // len(pts)] + pts[uk % len(pts)])
    curving = {"lines": 0, "circles": 0, "unfitted": {}, "moved_mid_nodes": 0, "max_displacement_m": 0.0}
    if curved:
        for b in np.unique(B2):
            sel = B2 == b
            V = pts[np.unique(E2[sel])]
            kind, fit = fit_curve(V)
            if kind == "line":
                curving["lines"] += 1
                continue
            if kind is None:
                curving["unfitted"][int(b)] = fit
                continue
            curving["circles"] += 1
            cx, cy, R = fit
            for e in E2[sel]:
                k = np.searchsorted(uk, min(e) * len(pts) + max(e))
                m0 = mid[k].copy()
                d = m0 - (cx, cy)
                m1 = np.array([cx, cy]) + R * d / np.linalg.norm(d)
                le = np.linalg.norm(pts[e[1]] - pts[e[0]])
                if np.linalg.norm(m1 - m0) <= 0.15 * le:
                    mid[k] = m1
                    curving["moved_mid_nodes"] += 1
                    curving["max_displacement_m"] = max(curving["max_displacement_m"], float(np.linalg.norm(m1 - m0)))
        # element validity: a mid-node moved by delta towards the opposite vertex inverts a thin triangle (boundary
        # layers: delta = le^2 / (8 R) can exceed its height / 4). Undo moves where det J at the 6 nodes or the centroid
        # falls below 0.2 x the straight value (straight edges there; reported).
        straight_mid = 0.5 * (pts[uk // len(pts)] + pts[uk % len(pts)])
        reverted = set()
        for _ in range(4):
            tri6_ = np.column_stack([T2, len(pts) + inv])
            ratio = _min_detj_ratio(np.vstack([pts, mid]), tri6_)
            bad = np.nonzero(ratio < 0.2)[0]
            if not len(bad):
                break
            for t in bad:
                for k in inv[t]:
                    if not np.array_equal(mid[k], straight_mid[k]):
                        mid[k] = straight_mid[k]
                        reverted.add(int(k))
        curving["reverted_mid_nodes"] = len(reverted)
        curving["min_detJ_over_straight"] = float(_min_detj_ratio(np.vstack([pts, mid]),
                                                                  np.column_stack([T2, len(pts) + inv])).min())
        report.add("mesh", "curved geometry", "approximated", "second-order triangles",
                   f"boundary mid-nodes moved onto fitted circles ({curving['circles']} boundaries, "
                   f"{curving['moved_mid_nodes']} nodes, {len(reverted)} moves undone where they would have distorted "
                   f"or inverted a thin element; min det J / straight {curving['min_detJ_over_straight']:.3g}); "
                   f"{curving['lines']} straight boundaries; unfitted: {curving['unfitted'] or 'none'}")
        if reverted:
            report.warnings.append(f"curving: {len(reverted)} boundary mid-node moves undone (thin elements next to "
                                   "curved boundaries); those edges stay straight")
    allp = np.vstack([pts, mid])
    tri6 = np.column_stack([T2, len(pts) + inv])
    worst = float(_min_detj_ratio(allp, tri6).min())
    if worst <= 0:
        raise ImportError_(f"the written second-order mesh has an inverted triangle (det J ratio {worst:.3g})")
    ekey = np.sort(E2, axis=1)
    eidx = np.searchsorted(uk, ekey[:, 0].astype(np.int64) * len(pts) + ekey[:, 1])
    line3 = np.column_stack([E2, len(pts) + eidx])
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = (out_dir / f"{stem}.msh").resolve()
    info_path = (out_dir / f"{stem}_mesh_info.json").resolve()
    write_mesh(mesh_path, allp, tri6, np.vectorize(attr_of_dom.get)(Td), line3, A2)
    # domain areas (straight / curved via the second-order map, 6-point rule)
    a6 = _areas(allp, tri6)
    report.summary["domain_areas_m2"] = {d: float(a6[Td == d].sum()) for d in phys_doms}
    info = {"ok": True, "imported_from": str(model.path), "importer": "axionfem.comsol_import2d (mesh route)",
            "symmetry": "axisymmetric" if axi else "planar", "elements_2d": int(len(T2)),
            "boundary_elements": int(len(E2)), "vertices": int(len(pts)), "seconds": time.time() - t0,
            "boundary_groups": groups,
            "groups": [{"dim": 2, "attr": attr_of_dom[d], "name": region_of[d], "comsol_domain": d} for d in phys_doms],
            "comsol": {"version": model.version, "mesh_sequence": msq["tag"], "resource": msq["resource"],
                       "domains": ndom, "boundaries": nbnd, "emw_domains": phys_doms, "excluded_domains": excluded},
            "port": port_info, "curving": curving}
    info_path.write_text(json.dumps(info, indent=1, default=float))
    # ---- design
    shape_order = ph["props"].get("order_electricfield")
    mo = order or (int(float(shape_order)) if shape_order else 2)
    fw = {"boundary": "pec", "rf_only": True}
    pml = pml_layers(model, geom, 2, phys_doms, region_of, report, axisym=axi)
    if pml:
        fw["pml"] = pml
    if fl and pickup:
        fw["frequencies"] = fl
    if frq["modes"]:
        if pml or any(gr["type"] in ("scattering", "sheet") for gr in groups):
            report.add("study", "eigenfrequency", "unsupported", detail="eigenmodes with PML, scattering or transition "
                       "boundaries are complex; axionFEM's 2D eigenmodes are for closed lossless models",
                       changes_results=True)
        else:
            fw["modes"] = frq["modes"]
            fam = {"outofplane": ["TE" if axi else "TM"], "inplane": ["TM" if axi else "TE"]}.get(comps)
            if fam:
                fw["families"] = fam
            if frq["mode_shift_Hz"]:
                fw["target_f_Hz"] = frq["mode_shift_Hz"]
    if source and pickup:
        fw["source"] = source
        fw["source_regions"] = source_regions
    if not axi:
        fw["thickness_m"] = thickness
    design = {"schema": "axionfem.design/1", "name": _slug(Path(model.path).stem),
              "symmetry": "axisymmetric" if axi else "planar",
              "notes": (f"Imported from {model.path} (COMSOL {model.version}) by axionfem.comsol_import, 2D mesh route "
                        f"(beta): COMSOL's stored mesh {msq['resource']} ({len(T2)} triangles on emw domains "
                        f"{phys_doms}); boundary conditions as boundary groups in {info_path.name}. See import_report.md."),
              "regions": regions, "fullwave": fw, "mesh": {"order": mo, "file": str(mesh_path), "info_file": str(info_path)}}
    if materials:
        design["materials"] = materials
    if pickup:
        design["pickup"] = pickup
    report.add("mesh", "element order", "mapped", f"mesh.order {mo}", f"COMSOL emw shape order {shape_order}")
    for mf in msq["features"]:
        report.add("mesh", _feat_desc(mf), "ignored", detail="mesh-generation setting; the stored mesh is used as is")
    report.summary.update({"comsol_version": model.version, "symmetry": design["symmetry"], "domains": ndom,
                           "boundaries": nbnd, "emw_domains": phys_doms, "triangles": int(len(T2)),
                           "boundary_types": {b: btype[b][0] for b in sorted(btype)}, "frequencies_Hz": fl,
                           "studies": frq["studies"], "port": port_info, "source": source, "field_components": comps})
    return MeshRoute(design, info, mesh_path, info_path)


def _dN_tri6(u, v):
    l0 = 1 - u - v
    return np.array([[-(4 * l0 - 1), -(4 * l0 - 1)], [4 * u - 1, 0], [0, 4 * v - 1],
                     [4 * (l0 - u), -4 * u], [4 * v, 4 * u], [-4 * v, 4 * (l0 - v)]])


def _min_detj_ratio(allp, tri6):
    """Per triangle: min over the 6 nodes and the centroid of det J of the quadratic map / det J of the straight
    triangle (1 for straight elements; <= 0: inverted)."""
    X = allp[tri6]
    a = X[:, :3]
    det_lin = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    out = np.full(len(tri6), np.inf)
    for u, v in ((0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5), (1 / 3, 1 / 3)):
        J = np.einsum("nkd,ke->nde", X, _dN_tri6(u, v))
        out = np.minimum(out, np.linalg.det(J) / det_lin)
    return out


def _areas(allp, tri6):
    """Areas of second-order triangles (6-point degree-4 rule on the quadratic map)."""
    r = np.array([[0.445948490915965, 0.445948490915965], [0.445948490915965, 0.108103018168070],
                  [0.108103018168070, 0.445948490915965], [0.091576213509771, 0.091576213509771],
                  [0.091576213509771, 0.816847572980459], [0.816847572980459, 0.091576213509771]])
    w = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2
    X = allp[tri6]
    out = np.zeros(len(tri6))
    for (u, v), wq in zip(r, w):
        l0 = 1 - u - v
        dN = np.array([[-(4 * l0 - 1), -(4 * l0 - 1)], [4 * u - 1, 0], [0, 4 * v - 1],
                       [4 * (l0 - u), -4 * u], [4 * v, 4 * u], [-4 * v, 4 * (l0 - v)]])
        J = np.einsum("nkd,ke->nde", X, dN)
        out += wq * np.linalg.det(J)
    return out


def _classify_axisymmetric(model, je, pts):
    """Je = (J_r, J_phi, J_z) expressions (COMSOL 2D-axisymmetric components) at points (r, z) -> a prescribed
    source evaluated at (r, 0, z): uniform J_z ("uniform" along z), or J_phi uniform / proportional to 1/r
    ("toroidal_box").  Anything else (J_r, mixed components, other profiles) raises ImportError_."""
    if je is None or len(je) != 3:
        raise ImportError_(f"Je must have three components (got {je})")
    r, z = pts[:, 0], pts[:, 1]
    var = {"r": r, "z": z, "R": r, "Z": z, "x": r, "y": np.zeros_like(r)}
    J = []
    for c in je:
        v = np.asarray(_num(model, c, var))
        if np.iscomplexobj(v) and np.any(np.abs(np.imag(v)) > 1e-12 * max(np.abs(v).max(), 1e-300)):
            raise ImportError_(f"complex Je component '{c}' is not representable by a real prescribed source")
        J.append(np.broadcast_to(np.real(v), r.shape).astype(float))
    Jr, Jp, Jz = J
    scale = max(np.abs(np.array(J)).max(), 1e-300)
    tol = 1e-9 * scale
    if np.abs(Jr).max() > tol:
        raise ImportError_("Je has a radial component; only J_z (uniform) or J_phi (uniform or 1/r) map to sources")
    if np.abs(Jp).max() <= tol:
        if np.all(np.abs(Jz - Jz[0]) <= tol):
            return {"kind": "uniform", "B_T": [0.0, 0.0, float(Jz[0])]}
        raise ImportError_("non-uniform J_z is not representable")
    if np.abs(Jz).max() > tol:
        raise ImportError_("Je with both J_phi and J_z: only one family per source is mapped")
    rmin, rmax, zmin, zmax = float(r.min()), float(r.max()), float(z.min()), float(z.max())
    pad = 1e-6 * max(rmax, zmax - zmin)
    if np.all(np.abs(Jp - Jp[0]) <= 1e-7 * scale):
        return {"kind": "toroidal_box", "B0_T": float(Jp[0]), "r_m": [max(rmin - pad, 0.0), rmax + pad],
                "z_m": [zmin - pad, zmax + pad], "profile": "uniform"}
    c = Jp * r
    if r.min() > 0 and np.all(np.abs(c - c[0]) <= 1e-7 * np.abs(c).max()):
        r0 = max(rmin - pad, 1e-9)
        return {"kind": "toroidal_box", "B0_T": float(c[0] / r0), "r_m": [r0, rmax + pad], "z_m": [zmin - pad, zmax + pad],
                "profile": "inverse_r"}
    raise ImportError_("J_phi with a profile other than uniform or 1/r is not representable")
