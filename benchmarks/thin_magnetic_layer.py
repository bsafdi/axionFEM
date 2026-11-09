"""Validation of the thin magnetic layer (docs/physics/thin_magnetic_layer.md).

Cases (docs/validation/thin_magnetic_layer.md):

1. a spherical shell in a uniform background field: the thin-layer FE result against the
   model's own closed form (the 2 x 2 system of the physics note), against the exact thick
   shell, and against the same shell meshed through its thickness as a linear region;
   h-convergence and the error vs d/R;
2. a solid linear sphere in a uniform background (the background-coupling fix);
3. an open long tube and a closed can in an axial background: thin layer vs meshed shell, and
   the infinite-tube relation for the wall flux at mid-length;
4. the inductance L_p of a superconducting pickup loop inside a closed can: thin vs meshed.

Lengths in m, fields in T. Usage:
    python benchmarks/thin_magnetic_layer.py [--quick] [--out FILE] [--work DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import axionfem as af

MU0 = 4e-7 * math.pi
HALF_PI = 0.5 * math.pi
R_SPHERE = 1.0      # mid-surface radius of every shell (m)
MU_R = 1000.0       # relative permeability of the FE cases
D_TUBE = 0.005      # wall thickness of the tube, can and pickup cases (m)
R_AXIS = 1e-6       # "on axis" sampling radius (m)
DEFAULT_OUT = Path(__file__).parent / "records" / "thin_magnetic_layer.json"

# ---------------------------------------------------------------- closed forms


def shield_exact(mu_r: float, a: float, b: float) -> float:
    """Exact B0/B_in of a spherical shell (inner a, outer b) in a uniform field.

    Jackson, Classical Electrodynamics, Problem 5.12."""
    num = (2 * mu_r + 1) * (mu_r + 2) - 2 * (a / b) ** 3 * (mu_r - 1) ** 2
    return num / (9 * mu_r)


def shield_model(mu_r: float, R: float, d: float) -> float:
    """B0/B_in of the thin-layer model on a sphere of radius R in vacuum.

    The 2 x 2 system of the physics note, unknowns x = m/R^3 and B_i at B0 = 1, with
    P = (mu - mu0) d and c = (d/4)(nu - nu0)/R for each half strip."""
    P = (mu_r - 1) * MU0 * d
    c = 0.25 * d * (1 / (mu_r * MU0) - 1 / MU0) / R
    # out: -(1 - x)/mu0 + (R/P)(1/2 + x - B_i/2) + 2c (1 + 2x) = 0
    # in :  B_i/mu0 - (R/P)(1/2 + x - B_i/2) + 2c B_i = 0
    A = np.array([[1 / MU0 + R / P + 4 * c, -R / (2 * P)],
                  [-R / P, 1 / MU0 + R / (2 * P) + 2 * c]])
    rhs = np.array([1 / MU0 - R / (2 * P) - 2 * c, R / (2 * P)])
    x, Bi = np.linalg.solve(A, rhs)
    return 1.0 / Bi


def shield_first_order(mu_r: float, R: float, d: float) -> float:
    """The common first-order term 1 + (2/3)(d/R)(mu_r - 1)^2/mu_r."""
    return 1 + 2.0 / 3.0 * d / R * (mu_r - 1) ** 2 / mu_r


def model_error_law(mu_r: float, dR: float) -> float:
    """Leading relative error of the model's S - 1 against the exact shell:
    (S_model - 1)/(S_exact - 1) - 1 = -(13/12 + 1/(2 mu_r)) (d/R)^2 + O((d/R)^3)."""
    return -(13.0 / 12.0 + 0.5 / mu_r) * dR ** 2


def coated_sc_axis_Bz(mu_r: float, a: float, b: float, z: float) -> float:
    """B_z/B0 on the axis at height z > b outside a superconducting ball of radius a coated by a
    shell a < rho < b of permeability mu_r, in a uniform field B0 (exact, three regions).

    A_phi = sin(theta) f(rho): shell f = B rho + C/rho^2, outside f = rho/2 + D/rho^2; f(a) = 0
    (the superconductor), A and H_t continuous at b. On the axis B_z = B0 (1 + 2 D/z^3)."""
    M = np.array([[a, 1 / a ** 2, 0.0],
                  [b, 1 / b ** 2, -1 / b ** 2],
                  [2 * b / mu_r, -1 / (mu_r * b ** 2), 1 / b ** 2]])
    _, _, D = np.linalg.solve(M, np.array([0.0, b / 2, b]))
    return 1 + 2 * D / z ** 3


def core_shell_Bin(mu_c: float, mu_r: float, a: float, b: float) -> float:
    """B_in/B0 inside a linear ball (mu_c, radius a) coated by a shell (mu_r, a < rho < b) in a
    uniform field (exact, three regions; magnetic scalar potential, H0 = 1)."""
    M = np.array([[0.0, b, 1 / b ** 2, 1 / b ** 2],
                  [0.0, mu_r, -2 * mu_r / b ** 3, -2 / b ** 3],
                  [a, -a, -1 / a ** 2, 0.0],
                  [mu_c, -mu_r, 2 * mu_r / a ** 3, 0.0]])
    A, _, _, _ = np.linalg.solve(M, np.array([b, 1.0, 0.0, 0.0]))
    return mu_c * A


def solid_sphere_reaction_energy(R: float, mu_r: float, B0: float) -> float:
    """Energy (J) of the reaction field of a linear ball in a uniform field B0, as the engine
    reports it with a background: (1/2) int nu |B - B_bg|^2 dV over all space (nu of the medium)."""
    k = (mu_r - 1) / (mu_r + 2) * B0 * R ** 3          # exterior dipole strength
    Br_in = 2 * (mu_r - 1) / (mu_r + 2) * B0           # interior reaction field
    w_out = 8 * math.pi * k ** 2 / (3 * R ** 3) / (2 * MU0)
    w_in = Br_in ** 2 / (2 * mu_r * MU0) * 4 * math.pi * R ** 3 / 3
    return w_out + w_in


# ---------------------------------------------------------------- designs


def material(mu_r: float) -> dict:
    return {"shield": {"kind": "linear", "mu_r": mu_r}}


def mesh_block(order: int, size: float, surface: float, domain: float) -> dict:
    return {"order": order, "size": size, "surface_size": surface, "domain_radius": domain}


def sphere_layer(R: float, d: float, mu_r: float, h: float, order: int = 2) -> dict:
    """Thin layer on the sphere of radius R in a uniform 1 T background."""
    arc = {"arc": {"center": [0, 0], "radius": R, "angles": [-HALF_PI, HALF_PI]}}
    curve = {"name": "shell", "material": "shield", "thickness": d, "path": arc,
             "surface_size": h}
    return {"schema": "axionfem.design/1", "name": "sphere_layer",
            "materials": material(mu_r), "regions": [], "curves": [curve],
            "background": {"type": "uniform", "Bz": 1.0},
            "mesh": mesh_block(order, 4 * h, h, 3 * R)}


def annulus(a: float, b: float) -> dict:
    """Contour of the spherical shell a < rho < b in the meridional half plane."""
    outer = {"ellipse": {"center": [0, 0], "axes": [b, b], "t": [-HALF_PI, HALF_PI]}}
    inner = {"ellipse": {"center": [0, 0], "axes": [a, a], "t": [HALF_PI, -HALF_PI]}}
    segments = [outer, {"line": [0, a]}, inner, {"line": [0, -b]}]
    return {"contour": {"start": [0, -b], "segments": segments}}


def sphere_meshed(R: float, d: float, mu_r: float, h: float, order: int = 2) -> dict:
    """The same shell meshed through its thickness as a linear region."""
    a, b = R - d / 2, R + d / 2
    region = {"name": "shell", "shape": annulus(a, b), "material": "shield",
              "mesh_size": h, "surface_size": h}
    return {"schema": "axionfem.design/1", "name": "sphere_meshed",
            "materials": material(mu_r), "regions": [region],
            "background": {"type": "uniform", "Bz": 1.0},
            "mesh": mesh_block(order, max(4 * h, 0.1 * R), h, 3 * R)}


def solid_sphere(R: float, mu_r: float, h: float) -> dict:
    """A solid linear sphere in a uniform 1 T background (B_in = 3 mu_r/(mu_r + 2))."""
    shape = ball(R)
    return {"schema": "axionfem.design/1", "name": "solid_sphere",
            "materials": material(mu_r),
            "regions": [{"name": "ball", "shape": shape, "material": "shield"}],
            "background": {"type": "uniform", "Bz": 1.0},
            "mesh": mesh_block(2, h, h / 2, 3 * R)}


def tube(R: float, L: float, d: float, mu_r: float, h: float, meshed: bool,
         closed: bool) -> dict:
    """Open tube (closed=False) or can with end caps (closed=True), axis z, centred at 0."""
    z0, z1 = -L / 2, L / 2
    doc = {"schema": "axionfem.design/1", "name": "tube", "materials": material(mu_r),
           "regions": [], "background": {"type": "uniform", "Bz": 1.0},
           "mesh": mesh_block(2, max(4 * h, 0.1 * R), h, 1.6 * L)}
    if not meshed:
        pts = [[0, z0], [R, z0], [R, z1], [0, z1]] if closed else [[R, z0], [R, z1]]
        path = {"polyline": {"points": pts, "closed": False}}
        doc["curves"] = [{"name": "wall", "material": "shield", "thickness": d,
                          "surface_size": h, "path": path}]
        return doc
    a, b = R - d / 2, R + d / 2
    if closed:
        lo, hi = z0 + d / 2, z1 - d / 2
        pts = [[0, z0 - d / 2], [b, z0 - d / 2], [b, z1 + d / 2], [0, z1 + d / 2],
               [0, hi], [a, hi], [a, lo], [0, lo]]
        shape = {"polygon": {"points": pts}}
    else:
        shape = {"rect": {"r": [a, b], "z": [z0, z1]}}
    doc["regions"] = [{"name": "wall", "shape": shape, "material": "shield",
                       "mesh_size": h, "surface_size": h}]
    return doc


def pickup_in_can(R: float, L: float, d: float, mu_r: float, h: float, shield: str) -> dict:
    """A superconducting pickup loop (a small nb torus) in a closed can.

    shield is none, thin or meshed. The static field is a weak azimuthal winding (a static
    solve needs a source); L_p is what is compared."""
    if shield == "none":
        doc = {"schema": "axionfem.design/1", "materials": material(mu_r), "regions": [],
               "mesh": mesh_block(2, max(4 * h, 0.1 * R), h, 1.6 * L)}
    else:
        doc = tube(R, L, d, mu_r, h, shield == "meshed", True)
        doc.pop("background")
    doc["name"] = "pickup_in_can"
    loop = {"name": "loop", "shape": {"disk": {"center": [0.5 * R, 0.0], "radius": 0.02 * R}},
            "material": "nb", "surface_size": 0.005 * R}
    zc = 0.3 * L / 2
    coil = {"name": "coil", "shape": {"rect": {"r": [0.3 * R, 0.32 * R], "z": [zc - 0.01, zc]}},
            "material": "winding"}
    doc["regions"] = doc["regions"] + [loop, coil]
    doc["windings"] = [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1.0}]
    doc["pickup"] = {"type": "azimuthal", "conductor": "loop"}
    doc["superconductors"] = [{"body": "loop", "azimuthal": {"net_current": 0.0}}]
    return doc


def ball(R: float) -> dict:
    half_disk = {"ellipse": {"center": [0, 0], "axes": [R, R], "t": [-HALF_PI, HALF_PI]}}
    return {"contour": {"start": [0, -R], "segments": [half_disk, {"line": [0, -R]}]}}


def coated_sc_sphere(R: float, d: float, mu_r: float, h: float, order: int,
                     jacket_first: bool) -> dict:
    """A layer on the surface of a superconducting (nb) ball in 1 T. A vacuum 'jacket' annulus
    (R to 1.2 R) listed before or after the ball changes which side keeps the original DOF numbers; the answer must not depend on it."""
    doc = sphere_layer(R, d, mu_r, h, order)
    sc = {"name": "sc", "material": "nb", "shape": ball(R)}
    jacket = {"name": "jacket", "material": "vacuum", "shape": annulus(R, 1.2 * R)}
    doc["regions"] = [jacket, sc] if jacket_first else [sc, jacket]
    return doc


def layer_on_core(R: float, d: float, mu_r: float, mu_c: float, h: float, order: int = 3) -> dict:
    """A layer on the surface of a linear ball (mu_c) in 1 T."""
    doc = sphere_layer(R, d, mu_r, h, order)
    doc["materials"]["core"] = {"kind": "linear", "mu_r": mu_c}
    doc["regions"] = [{"name": "core", "material": "core", "shape": ball(R)}]
    return doc


def can_with_sheet(R: float, L: float, d: float, mu_r: float, h: float, meshed: bool,
                   r_end: float, z_s: float = 0.3) -> dict:
    """A closed can in 1 T with an annular superconducting sheet at z = z_s from r = 0.4 R to r_end
    (r_end = R: ends on the thin wall; r_end > R: crosses it)."""
    doc = tube(R, L, d, mu_r, h, meshed, True)
    sheet = {"name": "disk", "material": "superconductor", "surface_size": h,
             "path": {"polyline": {"points": [[0.4 * R, z_s], [r_end, z_s]]}}}
    doc["curves"] = doc.get("curves", []) + [sheet]
    return doc


def hybrid_can(R: float, L: float, d: float, mu_r: float, h: float) -> dict:
    """A thin wall ending on the inner faces of meshed magnetic caps:
    the caps meet the wall only at its ends, so this is refused."""
    doc = tube(R, L - d, d, mu_r, h, False, False)
    z1 = L / 2
    caps = []
    for k, sgn in ((0, -1), (1, 1)):
        rect = {"r": [0.0, R + d / 2], "z": [sgn * z1 - d / 2, sgn * z1 + d / 2]}
        caps.append({"name": f"cap{k}", "material": "shield", "mesh_size": d / 2,
                     "surface_size": d / 2, "shape": {"rect": rect}})
    doc["regions"] = caps
    return doc


def sc_sheet(name: str, pts: list, h: float) -> dict:
    return {"name": name, "material": "superconductor", "surface_size": h,
            "path": {"polyline": {"points": pts}}}


def two_sheet_can(R: float, L: float, d: float, mu_r: float, h: float, meshed: bool,
                  z_s: float = 0.3) -> dict:
    """A closed can in 1 T with two annular superconducting sheets at z = z_s, one inside (0.4 R to
    the wall) and one outside (the wall to 1.6 R): two bodies, which must stay apart across the
    layer."""
    doc = tube(R, L, d, mu_r, h, meshed, True)
    r_in, r_out = (R - d / 2, R + d / 2) if meshed else (R, R)
    sheets = [sc_sheet("inner", [[0.4 * R, z_s], [r_in, z_s]], h),
              sc_sheet("outer", [[r_out, z_s], [1.6 * R, z_s]], h)]
    doc["curves"] = doc.get("curves", []) + sheets
    return doc


def flanged_ball(Rb: float, d: float, mu_r: float, h: float, meshed: bool) -> dict:
    """An nb ball (radius Rb) with a layer on its surface and a superconducting flange from the
    layer out to r = 1.2 m at z = 0, in 1 T (the flange is a second body)."""
    doc = {"schema": "axionfem.design/1", "name": "flanged_ball", "materials": material(mu_r),
           "background": {"type": "uniform", "Bz": 1.0}, "mesh": mesh_block(2, 0.1, h, 3.0)}
    if meshed:
        doc["regions"] = [{"name": "coat", "material": "shield", "mesh_size": h, "surface_size": h,
                           "shape": annulus(Rb - d / 2, Rb + d / 2)},
                          {"name": "sc", "material": "nb", "shape": ball(Rb - d / 2)}]
        r_in = Rb + d / 2
    else:
        arc = {"arc": {"center": [0, 0], "radius": Rb, "angles": [-HALF_PI, HALF_PI]}}
        doc["regions"] = [{"name": "sc", "material": "nb", "shape": ball(Rb)}]
        doc["curves"] = [{"name": "coat", "material": "shield", "thickness": d, "surface_size": h,
                          "path": arc}]
        r_in = Rb
    doc["curves"] = doc.get("curves", []) + [sc_sheet("flange", [[r_in, 0.0], [1.2, 0.0]], h)]
    return doc


def partial_coating(R: float, d: float, mu_r: float, mu_c: float, h: float, meshed: bool) -> dict:
    """A coating on the upper hemisphere of a linear ball (mu_c) in 1 T, order 3: the layer runs
    along the core up to its end at the equator, which is accepted."""
    mats = {"shield": {"kind": "linear", "mu_r": mu_r}, "core": {"kind": "linear", "mu_r": mu_c}}
    doc = {"schema": "axionfem.design/1", "name": "partial_coating", "materials": mats,
           "background": {"type": "uniform", "Bz": 1.0},
           "regions": [{"name": "core", "material": "core", "shape": ball(R)}]}
    if meshed:
        a, b = R - d / 2, R + d / 2
        outer = {"ellipse": {"center": [0, 0], "axes": [b, b], "t": [HALF_PI, 0.0]}}
        inner = {"ellipse": {"center": [0, 0], "axes": [a, a], "t": [0.0, HALF_PI]}}
        segs = [outer, {"line": [a, 0.0]}, inner, {"line": [0, b]}]
        shell = {"contour": {"start": [0, b], "segments": segs}}
        doc["regions"].append({"name": "coat", "material": "shield", "shape": shell,
                               "mesh_size": h, "surface_size": h})
        doc["mesh"] = mesh_block(3, 0.2, 0.05, 3 * R)
    else:
        arc = {"arc": {"center": [0, 0], "radius": R, "angles": [0.0, HALF_PI]}}
        doc["curves"] = [{"name": "coat", "material": "shield", "thickness": d, "surface_size": h,
                          "path": arc}]
        doc["mesh"] = mesh_block(3, 0.2, h, 3 * R)
    return doc


def hybrid_can_with_collar(mu_collar: float, h: float = 0.05) -> dict:
    """The hybrid can (refused) plus a weakly magnetic collar, 2 cm x 5 cm, outside the wall's last
    5 cm at each end: the collar runs along the end, the caps are still in front of it, so this is
    refused too."""
    doc = hybrid_can(R_SPHERE, 4.0, D_TUBE, MU_R, h)
    doc["materials"]["ss"] = {"kind": "linear", "mu_r": mu_collar}
    z_end = 2.0 - D_TUBE / 2
    for k, sgn in ((0, -1), (1, 1)):
        z = sorted([sgn * (z_end - 0.05), sgn * z_end])
        doc["regions"].append({"name": f"collar{k}", "material": "ss",
                               "shape": {"rect": {"r": [R_SPHERE, R_SPHERE + 0.02], "z": z}}})
    return doc


def plate_can(h: float, as_sheet: bool) -> dict:
    """The closed can with an nb plate crossing the wall at z = 0.3, as a region (r 0.4-1.2,
    z 0.295-0.305) or as a sheet: either way it cuts the wall."""
    doc = tube(R_SPHERE, 2.0, D_TUBE, MU_R, h, False, True)
    if as_sheet:
        doc["curves"].append(sc_sheet("plate", [[0.4, 0.3], [1.2, 0.3]], h))
    else:
        doc["regions"] = [{"name": "plate", "material": "nb", "surface_size": 0.01,
                           "shape": {"rect": {"r": [0.4, 1.2], "z": [0.295, 0.305]}}}]
    return doc


def crossed_pickup(mu_layer: float | None) -> dict:
    """An nb pickup ring (r 0.45-0.55, |z| < 0.02) driven by a weak winding, crossed by a 1 mm layer
    along r = 0.5; None: no layer."""
    doc = {"schema": "axionfem.design/1", "name": "crossed_pickup", "materials": material(MU_R),
           "mesh": mesh_block(2, 0.2, 0.02, 3.0),
           "regions": [{"name": "loop", "material": "nb", "surface_size": 0.005,
                        "shape": {"rect": {"r": [0.45, 0.55], "z": [-0.02, 0.02]}}},
                       {"name": "coil", "material": "winding",
                        "shape": {"rect": {"r": [0.2, 0.22], "z": [0.3, 0.32]}}}],
           "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "pickup": {"type": "azimuthal", "conductor": "loop"},
           "superconductors": [{"body": "loop", "azimuthal": {"net_current": 0.0}}]}
    if mu_layer is not None:
        doc["materials"]["weak"] = {"kind": "linear", "mu_r": mu_layer}
        path = {"polyline": {"points": [[0.5, -0.3], [0.5, 0.3]]}}
        doc["curves"] = [{"name": "tube", "material": "weak", "thickness": 0.001,
                          "surface_size": 0.02, "path": path}]
    return doc


# ---------------------------------------------------------------- helpers


def run(doc: dict, workdir: Path) -> af.Results:
    workdir.mkdir(parents=True, exist_ok=True)
    return af.Design(doc).solve(workdir=str(workdir), verbose=0)


def center_Bz(res) -> float:
    """B_z at the centre, sampled just off the axis (point location can fail exactly on it)."""
    return float(res.sample([R_AXIS], [0.0])["Bz"][0])


def layer_info(res, key: str = "static") -> dict:
    if key == "static":
        return res.data["static"]["poloidal"]["thin_layers"][0]
    return res.data["pickup"]["thin_layers"][0]


def static_dofs(res) -> int:
    return res.data["static"]["poloidal"]["solver"]["dofs"]


# ---------------------------------------------------------------- cases


def sphere_h_convergence(work: Path, quick: bool) -> dict:
    """FE thin layer -> the model's closed form under refinement, orders 2 and 3.

    The mesh next to the layer must stay a few layer thicknesses thick (the half-strip term is
    negative), so the finest meshes are run at d/R = 1e-3; refused meshes are recorded."""
    R = R_SPHERE
    conv = {}
    for d in (0.01, 0.001):
        S_mod = shield_model(MU_R, R, d)
        S_ex = shield_exact(MU_R, R - d / 2, R + d / 2)
        rows = []
        hs = [0.05, 0.025, 0.0125, 0.00625] if not quick else [0.05, 0.025]
        for order in (2, 3):
            for h in hs:
                tag = f"sph_d{d}_h{h}_p{order}"
                try:
                    res = run(sphere_layer(R, d, MU_R, h, order), work / tag)
                except af.AxionFEMError as err:
                    rows.append({"order": order, "h": h, "error": str(err)[:160]})
                    print(f"{tag}: refused ({str(err)[:80]})", flush=True)
                    continue
                S = 1.0 / center_Bz(res)
                li = layer_info(res)
                rows.append({"order": order, "h": h, "dofs": static_dofs(res),
                             "layer_faces": li["faces"], "S": S,
                             "rel_err_vs_model": S / S_mod - 1,
                             "rel_err_vs_exact": S / S_ex - 1,
                             "min_h_perp_over_d": li["min_h_perp_over_d"],
                             "B_layer_max_T": li["B_layer_max_T"],
                             "split_dofs": li["split_dofs"]})
                print(f"{tag}: S={S:.9f} model {S_mod:.9f} exact {S_ex:.9f} "
                      f"err(model) {S / S_mod - 1:+.2e}", flush=True)
        conv[str(d)] = {"R": R, "d": d, "mu_r": MU_R, "S_model": S_mod, "S_exact": S_ex,
                        "S_first_order": shield_first_order(MU_R, R, d), "rows": rows}
    return conv


def sphere_vs_d(work: Path, quick: bool) -> list:
    """Error vs d/R: closed forms for all d; the FE thin layer for d/R <= 0.01 (where the mesh
    can stay several d thick); the meshed shell for d/R >= 0.01."""
    R = R_SPHERE
    rows = []
    for mr in (100.0, 1000.0, 1.0e4):
        for dR in (0.1, 0.03, 0.01, 0.003, 0.001):
            dd = dR * R
            S_mod = shield_model(mr, R, dd)
            S_ex = shield_exact(mr, R - dd / 2, R + dd / 2)
            S_1 = shield_first_order(mr, R, dd)
            row = {"mu_r": mr, "d_over_R": dR, "S_exact": S_ex, "S_model": S_mod,
                   "S_first_order": S_1, "model_rel_err": S_mod / S_ex - 1,
                   "model_rel_err_S_minus_1": (S_mod - 1) / (S_ex - 1) - 1,
                   "first_order_rel_err": S_1 / S_ex - 1}
            if mr == MU_R and dR <= 0.01:
                h = 0.1 if dR > 0.005 else 0.05
                res = run(sphere_layer(R, dd, mr, h, 3), work / f"sph_d{dR}")
                row["S_thin_FE"] = 1.0 / center_Bz(res)
                row["thin_FE_h"] = h
                row["thin_FE_min_h_perp_over_d"] = layer_info(res)["min_h_perp_over_d"]
                row["thin_FE_rel_err_vs_exact"] = row["S_thin_FE"] / S_ex - 1
                row["thin_FE_rel_err_vs_model"] = row["S_thin_FE"] / S_mod - 1
            if mr == MU_R and dR >= 0.01 and not (quick and dR < 0.03):
                for hm in ([dd / 2, dd / 4] if not quick else [dd / 2]):
                    resm = run(sphere_meshed(R, dd, mr, hm, 2), work / f"sphm_d{dR}_h{hm}")
                    Sm = 1.0 / center_Bz(resm)
                    row.setdefault("meshed", []).append(
                        {"h": hm, "S": Sm, "rel_err_vs_exact": Sm / S_ex - 1,
                         "dofs": static_dofs(resm)})
            meshed = [m["S"] for m in row.get("meshed", [])]
            print(f"mu_r={mr} d/R={dR}: exact {S_ex:.6f} model {S_mod:.6f} "
                  f"(rel {S_mod / S_ex - 1:+.2e}) thin FE {row.get('S_thin_FE')} "
                  f"meshed {meshed}", flush=True)
            rows.append(row)
    return rows


def tubes(work: Path, quick: bool) -> dict:
    """Open tube and closed can in an axial 1 T background: thin layer vs meshed wall."""
    R = R_SPHERE
    out_all = {}
    for closed in (False, True):
        L = 4.0 if closed else 20.0
        name = "can" if closed else "open_tube"
        out = {"R": R, "L": L, "d": D_TUBE, "mu_r": MU_R}
        pts_r = [R_AXIS, 0.5, 0.9, 1.5, 0.5]
        pts_z = [0.0, 0.0, 0.0, 0.0, 0.45 * L]
        h_thin = [0.1, 0.05, 0.025] if not quick else [0.1]
        h_mesh = [D_TUBE / 2, D_TUBE / 4] if not quick else [D_TUBE / 2]
        for kind, hh in (("thin", h_thin), ("meshed", h_mesh)):
            for h in hh:
                doc = tube(R, L, D_TUBE, MU_R, h, kind == "meshed", closed)
                res = run(doc, work / f"{name}_{kind}_{h}")
                smp = res.sample(pts_r, pts_z)
                row = {"h": h, "points_rz": list(zip(pts_r, pts_z)), "Bz": smp["Bz"].tolist(),
                       "Br": smp["Br"].tolist()}
                if kind == "thin":
                    row["layer"] = layer_info(res)
                    if not closed:
                        # infinite tube: excess wall flux = 2 pi R (mu_r - 1) d B_inside
                        Bin = float(res.sample([0.98 * R], [0.0])["Bz"][0])
                        row["excess_flux_infinite_tube_Wb"] = (
                            2 * math.pi * R * (MU_R - 1) * D_TUBE * Bin)
                out.setdefault(kind, []).append(row)
                print(f"{name} {kind} h={h}: Bz {np.round(smp['Bz'], 6)}", flush=True)
        out_all[name] = out
    return out_all


def pickup(work: Path, quick: bool) -> dict:
    """L_p of a superconducting loop in a closed can: no shield, thin layer, meshed wall."""
    out = {}
    cases = (("none", 0.05), ("thin", 0.1), ("thin", 0.05), ("thin", 0.025),
             ("meshed", D_TUBE / 2), ("meshed", D_TUBE / 4))
    for shield, h in cases:
        if quick and h in (0.025, D_TUBE / 4):
            continue
        doc = pickup_in_can(R_SPHERE, 2.0, D_TUBE, MU_R, h, shield)
        res = run(doc, work / f"pick_{shield}_{h}")
        pk = res.data["pickup"]
        out.setdefault(shield, []).append({"h": h, "d": D_TUBE, "mu_r": MU_R,
                                           "L_p_H": pk["L_p_H"],
                                           "L_p_from_flux_H": pk["L_p_from_flux_H"]})
        print(f"pickup {shield} h={h}: L_p {pk['L_p_H']:.6e}", flush=True)
    return out


def junctions(work: Path, quick: bool) -> dict:
    """Layers meeting superconductors and magnetic regions."""
    out: dict = {}
    R, d = R_SPHERE, 0.01
    z = 1.5
    ex = coated_sc_axis_Bz(MU_R, R - d / 2, R + d / 2, z)
    rows = []
    for h in ([0.1, 0.05] if not quick else [0.1]):
        for jf in (False, True):
            res = run(coated_sc_sphere(R, d, MU_R, h, 3, jf), work / f"coated_{h}_{jf}")
            bz = float(res.sample([R_AXIS], [z])["Bz"][0])
            rows.append({"h": h, "jacket_first": jf, "Bz": bz, "exact": ex,
                         "rel_err_B_minus_1": (bz - 1) / (ex - 1) - 1})
            print(f"coated sc h={h} jacket_first={jf}: Bz {bz:.7f} exact {ex:.7f}", flush=True)
    out["coated_superconductor"] = {"R": R, "d": d, "mu_r": MU_R, "z": z, "rows": rows}

    rows = []
    for mu_c in (5.0, 50.0):
        for dd in (0.01, 0.003):
            for h in ([0.1, 0.05] if not quick else [0.1]):
                res = run(layer_on_core(R, dd, MU_R, mu_c, h), work / f"core_{mu_c}_{dd}_{h}")
                b = center_Bz(res)
                bex = core_shell_Bin(mu_c, MU_R, R - dd / 2, R + dd / 2)
                rows.append({"mu_c": mu_c, "d": dd, "h": h, "B_in": b, "exact": bex,
                             "rel_err": b / bex - 1,
                             "unshielded_core": core_shell_Bin(mu_c, mu_c, R, R)})
                print(f"core mu_c={mu_c} d={dd} h={h}: B_in {b:.8f} exact {bex:.8f} "
                      f"rel {b / bex - 1:+.2e}", flush=True)
    out["layer_on_linear_core"] = rows

    pr, pz = [R_AXIS, 0.5, 0.7, 0.7, 0.5], [0.0, 0.0, 0.5, 0.1, 0.9]
    rows = []
    cases = [("thin", "touch", R, 0.05), ("thin", "cross", 1.2, 0.05),
             ("thin", "gap2cm", R - 0.02, 0.05), ("meshed", "touch", R - D_TUBE / 2, D_TUBE / 2),
             ("meshed", "cross", 1.2, D_TUBE / 2), ("meshed", "gap2cm", R - 0.02, D_TUBE / 2)]
    for kind, tag, r_end, h in cases:
        doc = can_with_sheet(R, 2.0, D_TUBE, MU_R, h, kind == "meshed", r_end)
        res = run(doc, work / f"sheetcan_{kind}_{tag}")
        smp = res.sample(pr, pz)
        rows.append({"wall": kind, "sheet": tag, "r_end": r_end, "h": h,
                     "points_rz": list(zip(pr, pz)), "Bz": smp["Bz"].tolist(),
                     "Br": smp["Br"].tolist()})
        print(f"sheet can {kind} {tag}: Bz {np.round(smp['Bz'], 5)} Br {np.round(smp['Br'], 5)}",
              flush=True)
    out["can_with_sheet"] = rows

    pts = ([R_AXIS, 0.5, 0.7, 0.7], [0.0, 0.0, 0.5, 0.1])
    rows = []
    for kind, h in (("thin", 0.05), ("meshed", D_TUBE / 2)):
        doc = two_sheet_can(R, 2.0, D_TUBE, MU_R, h, kind == "meshed")
        res = run(doc, work / f"twosheet_{kind}")
        smp = res.sample(*pts)
        rows.append({"wall": kind, "h": h, "points_rz": list(zip(*pts)), "Bz": smp["Bz"].tolist(),
                     "Br": smp["Br"].tolist(),
                     "bodies": [b["names"] for b in res.data["static"]["poloidal"]["bodies"]]})
        print(f"two-sheet can {kind}: Bz {np.round(smp['Bz'], 5)} bodies {rows[-1]['bodies']}",
              flush=True)
    out["two_sheet_can"] = rows
    pts = ([R_AXIS, 0.3, 0.7, 0.55], [0.8, 0.6, 0.2, 0.05])
    rows = []
    for kind, h in (("thin", 0.025), ("meshed", 0.0025)):
        res = run(flanged_ball(0.5, 0.005, MU_R, h, kind == "meshed"), work / f"flange_{kind}")
        smp = res.sample(*pts)
        rows.append({"coating": kind, "h": h, "points_rz": list(zip(*pts)),
                     "Bz": smp["Bz"].tolist(), "Br": smp["Br"].tolist(),
                     "bodies": [b["names"] for b in res.data["static"]["poloidal"]["bodies"]]})
        print(f"flanged ball {kind}: Bz {np.round(smp['Bz'], 5)} bodies {rows[-1]['bodies']}",
              flush=True)
    out["flanged_ball"] = rows
    pts = ([R_AXIS, R_AXIS, 0.5, 0.5, R_AXIS, 1.3, 0.8], [0.0, 0.6, 0.3, -0.3, 1.5, 0.2, -0.9])
    rows = []
    for mu_c in (50.0, 5.0):
        for kind, h in (("thin", 0.1), ("meshed", 0.005), ("meshed", 0.0025)):
            res = run(partial_coating(R, 0.01, MU_R, mu_c, h, kind == "meshed"),
                      work / f"partial_{mu_c}_{kind}_{h}")
            smp = res.sample(*pts)
            rows.append({"mu_c": mu_c, "coating": kind, "h": h, "points_rz": list(zip(*pts)),
                         "Bz": smp["Bz"].tolist(), "Br": smp["Br"].tolist()})
            print(f"partial coating mu_c={mu_c} {kind} h={h}: Bz {np.round(smp['Bz'], 5)}",
                  flush=True)
    out["partial_coating"] = rows

    rows = []
    for tag, doc in (("region", plate_can(0.05, False)), ("sheet", plate_can(0.05, True))):
        res = run(doc, work / f"plate_{tag}")
        rows.append({"plate": tag, "Bz_center": center_Bz(res),
                     "bodies": [b["names"] for b in res.data["static"]["poloidal"]["bodies"]]})
        print(f"plate crossing the can, as a {tag}: Bz(0, 0) {rows[-1]['Bz_center']:.5f} "
              f"bodies {rows[-1]['bodies']}", flush=True)
    out["plate_crossing_can"] = rows
    rows = []
    for mu in (None, 1.5, 1000.0):
        res = run(crossed_pickup(mu), work / f"crossed_pickup_{mu}")
        pk = res.data["pickup"]
        rows.append({"mu_layer": mu, "L_p_H": pk["L_p_H"],
                     "L_p_from_flux_H": pk["L_p_from_flux_H"]})
        print(f"crossed pickup ring, layer mu_r {mu}: L_p {pk['L_p_H']:.6e} "
              f"from flux {pk['L_p_from_flux_H']:.6e}", flush=True)
    out["crossed_pickup_ring"] = rows
    try:
        run(hybrid_can_with_collar(1.05), work / "hybrid_collar")
        out["hybrid_can_with_collar"] = {"refused": False}
    except af.AxionFEMError as err:
        out["hybrid_can_with_collar"] = {"refused": True, "error": str(err)[:300]}
    print("hybrid can with a mu_r = 1.05 collar:", out["hybrid_can_with_collar"], flush=True)

    try:
        run(hybrid_can(R, 4.0, D_TUBE, MU_R, 0.05), work / "hybrid")
        out["hybrid_can"] = {"refused": False}
    except af.AxionFEMError as err:
        out["hybrid_can"] = {"refused": True, "error": str(err)[:300]}
    print("hybrid can:", out["hybrid_can"], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = Path(args.work or tempfile.mkdtemp(prefix="thinlayer_", dir=os.environ.get("TMPDIR")))
    rec: dict = {"script": "benchmarks/thin_magnetic_layer.py", "quick": args.quick, "cases": {}}
    t0 = time.time()

    rec["cases"]["sphere_h_convergence"] = sphere_h_convergence(work, args.quick)
    rec["cases"]["sphere_vs_d_over_R"] = sphere_vs_d(work, args.quick)

    res = run(solid_sphere(R_SPHERE, 10.0, 0.1), work / "solid")
    rec["cases"]["solid_sphere_mu10"] = {"Bz_center": center_Bz(res), "exact": 30.0 / 12.0}
    print("solid sphere mu_r=10:", center_Bz(res), "exact 2.5", flush=True)

    rec["cases"].update(tubes(work, args.quick))
    rec["cases"].update(junctions(work, args.quick))
    rec["cases"]["pickup_in_can"] = pickup(work, args.quick)

    rec["seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
