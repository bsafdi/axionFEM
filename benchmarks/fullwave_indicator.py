"""The 3D full-wave error indicator: does its estimate track the
actual error under refinement?

    python benchmarks/fullwave_indicator.py tem
    python benchmarks/fullwave_indicator.py cost [--ranks N]
    python benchmarks/fullwave_indicator.py mini CONFIG [--closed] [--levels 2] [--ranks N]
    python benchmarks/fullwave_indicator.py stage1 ORDER [--levels 3] [--ranks N]
    python benchmarks/fullwave_indicator.py coax [--levels 2] [--ranks N]
    python benchmarks/fullwave_indicator.py probe crack|ridge|crackopen ORDER [--levels 4] [--ranks N]
    python benchmarks/fullwave_indicator.py layer lossless|tand001|tand01|sigma [--levels 4] [--ranks N]
    python benchmarks/fullwave_indicator.py pmcplane [--levels 3] [--ranks N]
    python benchmarks/fullwave_indicator.py contrast cu|mid pec|pmc [--levels 4] [--ranks N]
    python benchmarks/fullwave_indicator.py contrast_partition cu|mid pec|pmc [--refinements 2] [--rank-list 1,3,4]
    python benchmarks/fullwave_indicator.py mini2d | coax2d
    python benchmarks/fullwave_indicator.py tem_exact          (a development build only, see below)
    python benchmarks/fullwave_indicator.py report [--write] [--development DIR | --development-from SOURCE]
                                                   [--provenance FILE]

The indicator (docs/physics/fullwave_error_indicator.md) estimates the energy-norm error of every port field; at order
2 and above also a(e, e) with sign. For the port's own impedance Z_h - Z = a(e, e) / (i omega) per ampere; its
estimates relative to omega L are L_relative_error_estimate (unsigned) and L_signed_relative_error_estimate, compared
here with the observed relative error of L. AXF_FWI_RUNS sets the runs directory (default runs/fullwave_indicator).

tem: the parallel-plate TEM line of benchmarks/fw3d_robin_validation.py, whose Z is exact, on its mesh and on uniform
refinements, at orders 1 and 2: shorted (5:1 cells, and cubic cells), with the port on an interior plane between two
shorted stubs (a cut), with an interior resistive sheet and a matched end, and PML-terminated. Runs in one process (a
few minutes).
mini: a small CORE-like box (the slotted box `mini` of benchmarks/slotted_box.py: r = 0.3-0.9 m, H = 0.8 m,
wall and slit d = 0.1 m), CONFIG c360 (uniform feed) or n1 (one 30-degree port), on a coarse mesh (131k tetrahedra)
and on one uniform refinement of it (axionfem.convergence.refine_and_compare, levels 2: the finer mesh is the
reference). Uniform refinement keeps the element geometry, so the differences are the field's discretization error
alone. --closed: inside a closed sphere instead of the radiating exterior (no termination error in the change of L).
mini2d: the c360 box axisymmetric (2D full wave on the exact geometry) at four wall-surface sizes, whose L adds the 3D
meshes' geometric error to the comparison. Frequencies 3 MHz (k R = 0.06, L within 0.6 % of its static value) and
20 MHz (k R = 0.4, half-way to the slit's LC resonance near 40 MHz: L/L_3MHz = 1.33, so the electric energy matters);
radiating exterior in 2D and 3D.
stage1: the two-port frame (benchmarks/designs/stage1_twoport.json without its validation loads; a sharp-edged frame,
so fields singular at its edges) at element order ORDER, refined the same way; 100 kHz and 10 MHz, closed sphere.
coax: a curved geometry new in this study, the shorted coax of benchmarks/fullwave3d_background.py revolved to 3D with
its annular 360-degree port (coax_design), closed sphere, kR = 0.02 and 0.4, refined the same way; coax2d its
axisymmetric twin at four surface sizes.
probe: singular lines (singular_line_design), a shorted parallel-plate
line 4 cm long with a crack (a PEC fin) or a 270-degree ridge (a perfect-conductor block) on its far half, or the
crack with an open end (crackopen: the error is electric), at 100 MHz and 1.2 GHz (crackopen: 100 MHz), on four
uniformly refined meshes. The reference is the order-2 series' Richardson limit at the theoretical rate (1 at a
crack, 4/3 at a 270-degree edge).
layer: the 5:1 shorted TEM line with its lower half filled (layered_line_design: lossless eps_r 4, tan delta 0.01
and 0.1, or a conductor with eps''/eps' = 3), order 2, 100 MHz, four uniformly refined meshes: a material
interface across which the loss phase of eps changes.
pmcplane: the 5:1 TEM line with and without its interior faces on y = h/2 as a pmc group, which the solver ignores
(pmc_plane_design), order 2, 100 MHz, three meshes.
contrast: a high-contrast line (contrast_design): a conductor meshed as a field region (cu:
5.8e7 S/m; mid: 5.8e3 S/m) under a lossy dielectric, shorted (pec) or open (pmc), order 2, 100 MHz, on uniformly
refined meshes; contrast_partition runs one mesh on several partitions (the complex projection's stopping test).
tem_exact: the shorted TEM lines at orders 1 and 2 with the exact magnetic and electric error energies against the
analytic TEM field, beside every estimator variant; it needs a development build that exposes every estimator variant
(AXF_FWI_TEM and the dev_residual key are not part of this release) and writes tem/tem_exact.json.
report prints the comparison; with --write it writes benchmarks/records/fullwave_indicator.json, and refuses when the
runs are missing. --development DIR adds the development runs (a development build with every estimator variant side
by side, the recovery included at order 2, and tem_exact), which the released code cannot regenerate; --development-from SOURCE
(a git commit, whose committed record is read, or a record file; not with --development) copies that section unchanged
when the runs are gone, marked `carried_over` with its source and sha256; --provenance FILE adds a JSON of builds and
job IDs.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
if not os.environ.get("PYTHONPATH"):   # else the package on PYTHONPATH (e.g. runs/pkgN with another build's module)
    sys.path.insert(0, str(ROOT / "python"))

import axionfem  # noqa: E402,F401
import slotted_box  # noqa: E402

RUNS = Path(os.environ.get("AXF_FWI_RUNS", ROOT / "runs" / "fullwave_indicator")).resolve()
OUT = ROOT / "benchmarks" / "records" / "fullwave_indicator.json"

MINI_WALL = 0.1
# a coarse base mesh (131k tetrahedra), since each uniform refinement multiplies it by 8
MINI_MESH = {"size": 0.2, "surface_size": 0.1, "curvature_points": 6}
MINI_REGION_SIZES = {"wall": 0.15, "port": 0.1, "gap": 0.1}
MINI_FREQUENCIES = [3e6, 2e7]
STAGE1_FREQUENCIES = [1e5, 1e7]


# ------------------------------------------------------------------ the TEM line (exact Z)
TEM_FREQUENCIES = [1e8, 2e8, 3e8]
TEM_ISO_FREQUENCIES = [1e8, 3e8, 6e8]
TEM_SHEET = {"i": 3, "sigma": 1e4, "d": 1e-6}


def tem_design(wd: Path, termination: str, order: int, refinements: int, frequencies, sheet=None, nx=6,
               W=0.04) -> tuple[dict, dict]:
    """The TEM line of benchmarks/fw3d_robin_validation.py (PEC plates, PMC sides, surface port at x = 0): shorted
    ("pec"), matched ("scattering", with an optional interior resistive sheet) or PML-terminated ("pml")."""
    import fw3d_robin_validation as robin
    kw = {"termination": termination, "plates": "pec", "nx": nx, "order": order, "W": W}
    if termination == "pml":
        kw["pml_cells"] = 2
    if sheet:
        kw["sheet"] = sheet
    doc, geometry = robin.build(wd, **kw)
    doc["fullwave"]["frequencies"] = list(frequencies)
    doc["mesh"]["refinements"] = refinements
    return doc, geometry


def tem_interior_port_design(wd: Path, order: int, refinements: int, frequencies, l_left=0.1, l_right=0.2, nl=2,
                             nr=4, h=0.02, W=0.04) -> tuple[dict, dict]:
    """The TEM line with its surface port on the interior plane x = 0, shorted at x = -l_left and x = l_right: two
    shorted stubs in parallel. The port's face carries the readout current, so tangential H jumps across it: a cut of
    the recovery."""
    import numpy as np
    import fw3d_robin_validation as robin
    wd.mkdir(parents=True, exist_ok=True)
    xs = np.concatenate([np.linspace(-l_left, 0, nl + 1), np.linspace(0, l_right, nr + 1)[1:]])
    P, T, tris, lab = robin.box_mesh(xs, np.linspace(0, h, 3), np.linspace(0, W, 3))
    attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004, 4: 10005, 5: 10006, 10 + nl: 10008}
    keep = (lab < 10) | (lab == 10 + nl)
    tri_attr = np.array([attr[int(b)] for b in lab[keep]])
    groups = [{"attr": 10008, "name": "lport1", "type": "port"},
              {"attr": 10001, "name": "end_left", "type": "pec"}, {"attr": 10002, "name": "end_right", "type": "pec"},
              {"attr": 10003, "name": "plate_bottom", "type": "pec"},
              {"attr": 10004, "name": "plate_top", "type": "pec"},
              {"attr": 10005, "name": "side_1", "type": "pmc"}, {"attr": 10006, "name": "side_2", "type": "pmc"}]
    mesh = wd / "plates3d.msh"
    robin.write_msh(mesh, P, T, np.full(len(T), 2), tris[keep], tri_attr)
    info = wd / "plates3d_mesh_info.json"
    info.write_text(json.dumps({"ok": True, "boundary_groups": groups, "shell_attr": -1,
                                "groups": [{"dim": 3, "attr": 2, "name": "field"}],
                                "domain_center": [(l_right - l_left) / 2, h / 2, W / 2],
                                "device_radius": l_left + l_right, "domain_radius": l_left + l_right}))
    doc = {"schema": "axionfem.design/1", "name": "plates3d_interior_port", "symmetry": "3d",
           "regions": [{"name": "field", "material": "vacuum", "solid": {"imported": {}}}],
           "pickup": {"type": "port", "port_boundary": "lport1", "toward": [0, 1, 0]},
           "fullwave": {"boundary": "pec", "rf_only": True, "frequencies": list(frequencies)},
           "mesh": {"order": order, "file": str(mesh), "info_file": str(info), "refinements": refinements}}
    return doc, {"l_left": l_left, "l_right": l_right, "h": h, "W": W}


def tem_exact_z(case: str, f: float, geometry: dict) -> complex:
    """The exact Z of a TEM line case (lossless, TEM: the mesh's straight geometry is exact)."""
    import fw3d_robin_validation as robin
    k = 2 * math.pi * f / robin.C0
    z0 = robin.ETA0 * geometry["h"] / geometry["W"]
    if case == "pec":
        return -1j * z0 * math.tan(k * geometry["l"])
    if case == "interior_port":
        left = -1j * z0 * math.tan(k * geometry["l_left"])
        right = -1j * z0 * math.tan(k * geometry["l_right"])
        return 1 / (1 / left + 1 / right)
    raise ValueError(case)


def tem_row(row: dict, z_exact: complex | None) -> dict:
    """One frequency of a TEM run: Z, the exact Z, the true errors of Z and L, and the indicator against them."""
    ind = row.get("error_indicator", {})
    out = {"f_Hz": row["f_Hz"], "Z": [row["Z_re_ohm"], row["Z_im_ohm"]],
           "indicator": {k: ind.get(k) for k in ("estimated_relative_error", "magnetic_relative_error",
                                                 "electric_relative_error", "electric_norm_share",
                                                 "Z_error_estimate_ohm", "L_relative_error_estimate",
                                                 "L_signed_relative_error_estimate", "Z_L_withheld",
                                                 "L_signed_withheld", "largest_element")}}
    if z_exact is None:
        return out
    w = 2 * math.pi * row["f_Hz"]
    zh = complex(row["Z_re_ohm"], row["Z_im_ohm"])
    out["Z_exact"] = [z_exact.real, z_exact.imag]
    out["Z_error_abs"] = abs(zh - z_exact)
    L_exact = -z_exact.imag / w
    out["L_true_signed_relative_error"] = row["L_H"] / L_exact - 1 if L_exact > 0 else None
    estimate = ind.get("Z_error_estimate_ohm")
    if estimate is not None and out["Z_error_abs"] > 0:
        out["Z_effectivity"] = estimate / out["Z_error_abs"]
    true = out["L_true_signed_relative_error"]
    if true:
        if ind.get("L_relative_error_estimate") is not None:
            out["L_effectivity"] = ind["L_relative_error_estimate"] / abs(true)
        if ind.get("L_signed_relative_error_estimate") is not None:
            out["L_signed_ratio"] = ind["L_signed_relative_error_estimate"] / true
    return out


# ------------------------------------------------------------------ probes: singular lines
PROBE_FREQUENCIES = {"crack": [1e8, 1.2e9], "ridge": [1e8, 1.2e9], "crackopen": [1e8]}
# the theoretical rate of L (or of Im Z for the open crack) in h: 1 next to a crack, 4/3 next to a 270-degree edge
PROBE_RATE = {"crack": 1.0, "ridge": 4.0 / 3.0, "crackopen": 1.0}
LAYER_MATERIALS = {
    "lossless": {"kind": "linear", "eps_r": 4.0, "mu_r": 1.0},
    "tand001": {"kind": "linear", "eps_r": 4.0, "mu_r": 1.0, "tan_delta": 0.01},
    "tand01": {"kind": "linear", "eps_r": 4.0, "mu_r": 1.0, "tan_delta": 0.1},
    "sigma": {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": 0.0167},
}


def singular_line_design(wd: Path, kind: str, order: int, frequencies, l=0.04, h=0.02, W=0.04, x0=0.01, n=4,
                         fill=None) -> dict:
    """A shorted parallel-plate line (PEC plates y = 0, h; PMC sides z = 0, W; surface port on x = 0) with a sharp
    feature on the span x >= x0, y <= h/2:
    - crack: a zero-thickness PEC fin (an interior pec group) on z = W/2, so crack edges (fields r^-1/2);
    - ridge: a meshed perfect-conductor block on W/4 <= z <= 3W/4, so 270-degree re-entrant edges (r^-1/3);
    - crackopen: the crack with an open (PMC) end x = l, a capacitive stub whose error is electric.
    Kuhn tetrahedra of an n^3 grid; refined uniformly by mesh.refinements. FILL (a LAYER_MATERIALS tag): the upper half
    y > h/2 filled with that material (region "fill"); with a lossy fill and the ridge, the gradient projection takes
    its complex-eps path on a mesh with elements outside the field (tests/test_gpu.py)."""
    import numpy as np
    import fw3d_robin_validation as robin
    wd.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0, l, n + 1)
    ys = np.linspace(0, h, n + 1)
    zs = np.linspace(0, W, n + 1)
    index = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    P = np.array([[x, y, z] for x in xs for y in ys for z in zs], float)
    T = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [index(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                T += [[c[q] for q in t] for t in robin._KUHN]
    T = np.array(T)
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    faces, count = np.unique(F, axis=0, return_counts=True)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    centres = P[T].mean(axis=1)
    tet_attr = np.full(len(T), 2)
    if kind == "ridge":
        inside = (centres[:, 2] > W / 4) & (centres[:, 2] < 3 * W / 4) & (centres[:, 1] < h / 2) & (centres[:, 0] > x0)
        tet_attr = np.where(inside, 3, 2)
    if fill is not None:
        tet_attr = np.where(centres[:, 1] > h / 2, 4, tet_attr)
    outer = {(0, 0): 10001, (0, 1): 10002, (1, 0): 10003, (1, 1): 10004, (2, 0): 10005, (2, 1): 10006}
    tris = []
    tri_attr = []
    tol = 1e-12
    for f, c in zip(faces, count):
        X = P[f]
        if c == 1:
            for ax in range(3):
                for side, value in ((0, lo[ax]), (1, hi[ax])):
                    if np.all(np.abs(X[:, ax] - value) < tol):
                        tris.append(f)
                        tri_attr.append(outer[(ax, side)])
        elif kind in ("crack", "crackopen"):
            if (np.all(np.abs(X[:, 2] - W / 2) < tol) and np.all(X[:, 1] <= h / 2 + tol)
                    and np.all(X[:, 0] >= x0 - tol)):
                tris.append(f)
                tri_attr.append(10009)
    groups = [{"attr": 10001, "name": "lport1", "type": "port"},
              {"attr": 10002, "name": "end", "type": "pmc" if kind == "crackopen" else "pec"},
              {"attr": 10003, "name": "plate_bottom", "type": "pec"},
              {"attr": 10004, "name": "plate_top", "type": "pec"},
              {"attr": 10005, "name": "side_1", "type": "pmc"}, {"attr": 10006, "name": "side_2", "type": "pmc"}]
    if kind in ("crack", "crackopen"):
        groups.append({"attr": 10009, "name": "fin", "type": "pec"})
    mesh = wd / "line.msh"
    robin.write_msh(mesh, P, T, tet_attr, np.array(tris), np.array(tri_attr))
    regions = [{"name": "field", "material": "vacuum", "solid": {"imported": {}}}]
    volume_groups = [{"dim": 3, "attr": 2, "name": "field"}]
    doc = {"schema": "axionfem.design/1", "name": f"fullwave_indicator_probe_{kind}", "symmetry": "3d",
           "regions": regions, "pickup": {"type": "port", "port_boundary": "lport1", "toward": [0, 1, 0]},
           "fullwave": {"boundary": "pec", "rf_only": True, "frequencies": list(frequencies)},
           "mesh": {"order": order, "file": str(mesh), "refinements": 0}}
    if kind == "ridge":
        regions.append({"name": "block", "material": "pecblock", "solid": {"imported": {}}})
        doc["materials"] = {"pecblock": {"kind": "conductor"}}
        volume_groups.append({"dim": 3, "attr": 3, "name": "block"})
    if fill is not None:
        regions.append({"name": "fill", "material": "fillmat", "solid": {"imported": {}}})
        doc.setdefault("materials", {})["fillmat"] = dict(LAYER_MATERIALS[fill])
        volume_groups.append({"dim": 3, "attr": 4, "name": "fill"})
        doc["name"] += f"_fill_{fill}"
    info = wd / "line_mesh_info.json"
    info.write_text(json.dumps({"ok": True, "boundary_groups": groups, "shell_attr": -1, "groups": volume_groups,
                                "domain_center": [l / 2, h / 2, W / 2], "device_radius": l, "domain_radius": l}))
    doc["mesh"]["info_file"] = str(info)
    return doc



# The ridge line of the order-2 indicator's stored-number checks (tests/test_fullwave3d_error_indicator.py, on CPU
# nodes; tests/test_gpu.py, the CPU half of the GPU comparison): singular_line_design("ridge", order 2, 0.3 and 1 GHz,
# one uniform refinement, 21424 DOFs), lossless or with its upper half weakly conducting (FILL "sigma"), the gradient
# projection solved to RIDGE_LINE_TOL with OpenBLAS's kernels pinned to RIDGE_LINE_CORETYPE.
RIDGE_LINE_FREQUENCIES = [3e8, 1e9]
RIDGE_LINE_TOL = "1e-10"
RIDGE_LINE_CORETYPE = "Haswell"
# The CPU build's indicator there, per fill and frequency: [total, magnetic part, electric part]. Two runs of the CPU
# build (the residual problems on the canonical complement) on nodes with the same CPU model gave the same bits. The
# OpenBLAS kernel family moves them by at most 1.2e-11 (Zen gives Haswell's bits; SkylakeX and Sandybridge differ in the last digits), and a build
# with another compiler on another CPU (GCC 14, AMD Zen 3) by 1.15e-11. The earlier numbers (the tree complement) were
# 3.7e-4 (0.3 GHz) and 4.9e-3 (1 GHz) higher in the magnetic part. Refresh these from a CPU run of the same design in
# the same environment (ridge_line_design, OPENBLAS_CORETYPE, AXF_FW_INDICATOR_TOL) whenever the indicator changes on
# purpose; compiler, MFEM, hypre, Eigen or the BLAS can each move the last bits.
# The checks compare to RIDGE_LINE_STORED_REL relative: the gradient projection is solved only to a relative residual of
# RIDGE_LINE_TOL, so round-off from another build can move the indicator by up to about that much (1.2e-11 measured,
# above), while every intended change of the indicator so far moved it by 3.7e-4 or more.
RIDGE_LINE_STORED_REL = float(RIDGE_LINE_TOL)
RIDGE_LINE_CPU = {"sigma": {3e8: [0.042260268065290166, 0.04070319795927715, 0.08624094905108938],
                            1e9: [0.048582952163426577, 0.04237473448466212, 0.07076548238998352]},
                  None: {3e8: [0.041246652022512036, 0.04068985113051931, 0.06700320474007705],
                         1e9: [0.047812991756670024, 0.042371526042686264, 0.06845348318067056]}}


def ridge_line_design(wd: Path, fill=None) -> dict:
    """The ridge line of RIDGE_LINE_CPU: the 270-degree-edge probe at order 2 on one uniform refinement, 0.3 and 1 GHz;
    FILL a LAYER_MATERIALS tag for the upper half (None: lossless)."""
    doc = singular_line_design(wd, "ridge", 2, RIDGE_LINE_FREQUENCIES, fill=fill)
    doc["mesh"]["refinements"] = 1
    return doc

def layered_line_design(wd: Path, tag: str, frequencies, order=2, l=0.3, h=0.02, W=0.04, nx=6) -> dict:
    """The 5:1 shorted TEM line with its lower half (y < h/2) filled by LAYER_MATERIALS[tag]: a material interface
    across which the loss phase of eps changes (lossy fills)."""
    import numpy as np
    import fw3d_robin_validation as robin
    wd.mkdir(parents=True, exist_ok=True)
    P, T, tris, lab = robin.box_mesh(np.linspace(0, l, nx + 1), np.linspace(0, h, 3), np.linspace(0, W, 3))
    centres = P[T].mean(axis=1)
    tet_attr = np.where(centres[:, 1] < h / 2, 3, 2)
    attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004, 4: 10005, 5: 10006}
    keep = lab < 10
    groups = [{"attr": 10001, "name": "lport1", "type": "port"}, {"attr": 10002, "name": "end", "type": "pec"},
              {"attr": 10003, "name": "plate_bottom", "type": "pec"},
              {"attr": 10004, "name": "plate_top", "type": "pec"},
              {"attr": 10005, "name": "side_1", "type": "pmc"}, {"attr": 10006, "name": "side_2", "type": "pmc"}]
    mesh = wd / "plates3d.msh"
    robin.write_msh(mesh, P, T, tet_attr, tris[keep], np.array([attr[int(b)] for b in lab[keep]]))
    info = wd / "plates3d_mesh_info.json"
    info.write_text(json.dumps({"ok": True, "boundary_groups": groups, "shell_attr": -1,
                                "groups": [{"dim": 3, "attr": 2, "name": "field"},
                                           {"dim": 3, "attr": 3, "name": "fill"}],
                                "domain_center": [l / 2, h / 2, W / 2], "device_radius": l, "domain_radius": l}))
    return {"schema": "axionfem.design/1", "name": f"fullwave_indicator_layer_{tag}", "symmetry": "3d",
            "regions": [{"name": "field", "material": "vacuum", "solid": {"imported": {}}},
                        {"name": "fill", "material": "fillmat", "solid": {"imported": {}}}],
            "materials": {"fillmat": LAYER_MATERIALS[tag]},
            "pickup": {"type": "port", "port_boundary": "lport1", "toward": [0, 1, 0]},
            "fullwave": {"boundary": "pec", "rf_only": True, "frequencies": list(frequencies)},
            "mesh": {"order": order, "file": str(mesh), "info_file": str(info), "refinements": 0}}



CONTRAST_SIGMA = {"cu": 5.8e7, "mid": 5.8e3}


def _plate_line(wd: Path, name: str, xs, ys, zs, layers, materials: dict, end: str, order: int, frequencies,
                interior_pmc_y=None) -> dict:
    """A parallel-plate line on a Kuhn grid: PEC plates y = 0, h; PMC sides z = 0, W; surface port x = 0; end x = l of
    type END (pec: shorted, pmc: open). layers: [(y_lo, y_hi, region, material)] by element centroid, else vacuum.
    interior_pmc_y: the interior faces on that plane as a pmc group (the solver imposes nothing there)."""
    import numpy as np
    import fw3d_robin_validation as robin
    wd.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    index = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    P = np.array([[x, y, z] for x in xs for y in ys for z in zs], float)
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [index(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                T += [[c[q] for q in tet] for tet in robin._KUHN]
    T = np.array(T)
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    faces, count = np.unique(F, axis=0, return_counts=True)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    centres = P[T].mean(axis=1)
    names = ["field"]
    material_of = {"field": "vacuum"}
    tet_attr = np.full(len(T), 2)
    for y0, y1, region, material in layers:
        names.append(region)
        material_of[region] = material
        inside = (centres[:, 1] > y0) & (centres[:, 1] < y1) & (tet_attr == 2)
        tet_attr = np.where(inside, 1 + len(names), tet_attr)
    outer = {(0, 0): 10001, (0, 1): 10002, (1, 0): 10003, (1, 1): 10004, (2, 0): 10005, (2, 1): 10006}
    tris = []
    tri_attr = []
    tol = 1e-12
    for f, c in zip(faces, count):
        X = P[f]
        if c == 1:
            for ax in range(3):
                for side, value in ((0, lo[ax]), (1, hi[ax])):
                    if np.all(np.abs(X[:, ax] - value) < tol):
                        tris.append(f)
                        tri_attr.append(outer[(ax, side)])
        elif interior_pmc_y is not None and np.all(np.abs(X[:, 1] - interior_pmc_y) < tol):
            tris.append(f)
            tri_attr.append(10009)
    groups = [{"attr": 10001, "name": "lport1", "type": "port"}, {"attr": 10002, "name": "end", "type": end},
              {"attr": 10003, "name": "plate_bottom", "type": "pec"},
              {"attr": 10004, "name": "plate_top", "type": "pec"},
              {"attr": 10005, "name": "side_1", "type": "pmc"}, {"attr": 10006, "name": "side_2", "type": "pmc"}]
    if interior_pmc_y is not None:
        groups.append({"attr": 10009, "name": "interior_plane", "type": "pmc"})
    mesh = wd / "line.msh"
    robin.write_msh(mesh, P, T, tet_attr, np.array(tris), np.array(tri_attr))
    info = wd / "line_mesh_info.json"
    l, h, W = (float(v) for v in hi)
    info.write_text(json.dumps({"ok": True, "boundary_groups": groups, "shell_attr": -1,
                                "groups": [{"dim": 3, "attr": 2 + k, "name": r} for k, r in enumerate(names)],
                                "domain_center": [l / 2, h / 2, W / 2], "device_radius": l, "domain_radius": l}))
    doc = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d",
           "regions": [{"name": r, "material": material_of[r], "solid": {"imported": {}}} for r in names],
           "pickup": {"type": "port", "port_boundary": "lport1", "toward": [0, 1, 0]},
           "fullwave": {"boundary": "pec", "rf_only": True, "frequencies": list(frequencies)},
           "mesh": {"order": order, "file": str(mesh), "info_file": str(info), "refinements": 0}}
    if materials:
        doc["materials"] = dict(materials)
    return doc


def pmc_plane_design(wd: Path, plane: bool, order: int, frequencies) -> dict:
    """The 5:1 shorted TEM line (l = 0.3 m, h = 2 cm, W = 4 cm, 6 x 2 x 2 cells) with or without its interior faces on
    y = h/2 as a pmc group. The solver imposes nothing there, so Z is the same with and without; the indicator must be
    too (zero traction there would make the estimate 1500-12700 times too high)."""
    import numpy as np
    return _plate_line(wd, f"fullwave_indicator_pmcplane_{'on' if plane else 'off'}", np.linspace(0, 0.3, 7),
                       np.linspace(0, 0.02, 3), np.linspace(0, 0.04, 3), [], {}, "pec", order, frequencies,
                       interior_pmc_y=0.01 if plane else None)


def contrast_design(wd: Path, cond: str, end: str, order: int, frequencies) -> dict:
    """The high-contrast line: l = 0.1 m, h = 2 cm, W = 4 cm; layers in y: 0-5 mm a conductor meshed
    as a field region (cu: 5.8e7 S/m, sigma / omega eps0 = 1e10 at 100 MHz; mid: 5.8e3 S/m, 1e6), 5-10 mm eps_r 4 with
    tan delta 0.02, vacuum above; END pec (shorted) or pmc (open)."""
    import numpy as np
    layers = [(0.0, 0.005, "cond", "condmat"), (0.005, 0.01, "diel", "dielmat")]
    materials = {"condmat": {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": CONTRAST_SIGMA[cond]},
                 "dielmat": {"kind": "linear", "eps_r": 4.0, "mu_r": 1.0, "tan_delta": 0.02}}
    return _plate_line(wd, f"fullwave_indicator_contrast_{cond}_{end}", np.linspace(0, 0.1, 5),
                       [0.0, 0.005, 0.01, 0.015, 0.02], np.linspace(0, 0.04, 3), layers, materials, end, order,
                       frequencies)


def run_contrast_partition(cond: str, end: str, refinements: int, rank_list: list[int]) -> None:
    """The high-contrast line on one mesh, on several partitions: the projection's partition sensitivity."""
    import axionfem as af
    base = RUNS / f"contrast_partition_{cond}_{end}"
    doc = contrast_design(base / "mesh", cond, end, 2, [1e8])
    doc["mesh"]["refinements"] = refinements
    for ranks in rank_list:
        af.Design(doc).solve(base / f"n{ranks}", verbose=1, mpi_ranks=ranks)


def run_tem() -> None:
    import axionfem as af
    cases = []
    for order, levels in ((1, (0, 1, 2, 3)), (2, (0, 1, 2))):
        cases += [("pec", order, n) for n in levels]
        cases += [("pec_iso", order, n) for n in levels[:3]]
        cases += [("interior_port", order, n) for n in levels]
    cases += [("sheet", 2, n) for n in (0, 1, 2)]
    cases += [("pml", 2, n) for n in (0, 1)]
    out = []
    for case, order, n in cases:
        wd = RUNS / "tem" / f"{case}_p{order}_r{n}"
        if case == "interior_port":
            doc, geometry = tem_interior_port_design(wd, order, n, TEM_FREQUENCIES[:2])
        elif case == "pec_iso":   # cubic Kuhn cells: dx = dy = dz = 1 cm
            doc, geometry = tem_design(wd, "pec", order, n, TEM_ISO_FREQUENCIES, nx=30, W=0.02)
        elif case == "sheet":
            doc, geometry = tem_design(wd, "scattering", order, n, TEM_FREQUENCIES, sheet=TEM_SHEET)
        else:
            doc, geometry = tem_design(wd, case, order, n, TEM_FREQUENCIES)
        t0 = time.time()
        r = af.Design(doc).solve(wd, verbose=0)
        exact_case = "pec" if case in ("pec", "pec_iso") else case
        rows = [tem_row(row, tem_exact_z(exact_case, row["f_Hz"], geometry) if exact_case in ("pec", "interior_port")
                        else None) for row in r.data["fullwave"]["sweep"]]
        entry = {"case": case, "order": order, "refinements": n, "dofs": r.data["mesh"]["dofs"],
                 "describe": r.data["fullwave"].get("error_indicator"), "rows": rows, "seconds": time.time() - t0}
        out.append(entry)
        print(f"tem {case} p{order} r{n}: dofs {entry['dofs']}",
              [(x["f_Hz"], x.get("L_effectivity"), x.get("L_signed_ratio"),
                x["indicator"].get("estimated_relative_error")) for x in rows], flush=True)
    (RUNS / "tem").mkdir(parents=True, exist_ok=True)
    (RUNS / "tem" / "tem_runs.json").write_text(json.dumps(out, indent=1) + "\n")


def run_tem_exact() -> None:
    """The shorted lines (5:1 and cubic cells) at orders 1 and 2, two meshes each, with the exact magnetic and electric
    error energies (AXF_FWI_TEM: the analytic TEM field, in the development build's indicator) beside every estimator
    variant (dev_residual). Writes tem/tem_exact.json."""
    import axionfem as af
    out = []
    for case, nx, W, frequencies in (("pec", 6, 0.04, [1e8, 2e8]), ("pec_iso", 30, 0.02, [1e8, 6e8])):
        for order in (1, 2):
            for n in (0, 1):
                wd = RUNS / "tem_exact" / f"{case}_p{order}_r{n}"
                doc, geometry = tem_design(wd, "pec", order, n, frequencies, nx=nx, W=W)
                os.environ["AXF_FWI_TEM"] = f"{geometry['l']},{geometry['h']},{geometry['W']}"
                try:
                    data = af.Design(doc).solve(wd, verbose=0).data
                finally:
                    os.environ.pop("AXF_FWI_TEM", None)
                for row in data["fullwave"]["sweep"]:
                    dev = row["error_indicator"].get("dev_residual")
                    if dev is None:
                        raise SystemExit("tem_exact needs a development build with every estimator variant: no dev_residual")
                    exact = min(dev["tem_exact"].values(), key=lambda p: p[0] + p[1])
                    tot = [sum(v[c] for v in dev["regions"].values()) for c in range(15)]
                    entry = {"case": case, "order": order, "refinements": n, "f_Hz": row["f_Hz"],
                             "dofs": data["mesh"]["dofs"], "exact_magnetic": exact[0], "exact_electric": exact[1],
                             "recovery": [tot[13], tot[14]], "residual_magnetic": tot[0],
                             "gradient_full_local": tot[4], "gradient_surplus_local": tot[7],
                             "gradient_global": tot[1] + tot[10]}
                    entry["ratios"] = {"recovery_magnetic": tot[13] / exact[0],
                                       "recovery_electric": tot[14] / exact[1],
                                       "residual_magnetic": tot[0] / exact[0],
                                       "gradient_full_local": tot[4] / exact[1],
                                       "gradient_surplus_local": tot[7] / exact[1],
                                       "gradient_global": (tot[1] + tot[10]) / exact[1]}
                    out.append(entry)
                    print(json.dumps({k: entry[k] for k in ("case", "order", "refinements", "f_Hz", "ratios")}),
                          flush=True)
    (RUNS / "tem").mkdir(parents=True, exist_ok=True)
    (RUNS / "tem" / "tem_exact.json").write_text(json.dumps(out, indent=1) + "\n")


COST_PATHS = {"ams_guard": {}, "plain_strumpack": {"AXF_FW_SOLVER": "strumpack"},
              "gauged": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}}


def run_cost(ranks: int) -> None:
    """The indicator's share of each frequency's time on the two-port frame (orders 1 and 2), per solver path (the
    tree-cotree path is lowest order only)."""
    import axionfem as af
    out = []
    for order in (1, 2):
        for path, env in COST_PATHS.items():
            if path == "gauged" and order > 1:
                continue
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                wd = RUNS / "cost" / f"stage1_p{order}_{path}"
                data = af.Design(stage1_design(order)).solve(wd, verbose=0, mpi_ranks=ranks).data
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            for row in data["fullwave"]["sweep"]:
                seconds = sum(p["error_indicator"]["seconds"] for p in row["port_solutions"])
                out.append({"order": order, "path": path, "ranks": ranks, "f_Hz": row["f_Hz"],
                            "dofs": data["mesh"]["dofs"], "indicator_seconds": seconds,
                            "frequency_seconds": row["frequency_total_seconds"],
                            "share": seconds / row["frequency_total_seconds"],
                            "setup_seconds": data["fullwave"]["error_indicator"]["setup_seconds"]})
                print(json.dumps(out[-1]), flush=True)
    (RUNS / "cost").mkdir(parents=True, exist_ok=True)
    (RUNS / "cost" / "cost.json").write_text(json.dumps(out, indent=1) + "\n")


def mini_design(config: str, closed: bool = False) -> dict:
    """The small CORE-like box; closed: inside a closed sphere (PEC at R1) instead of the radiating exterior, so that
    the change of L under refinement holds no termination error (which the indicator leaves out)."""
    doc = slotted_box.design("mini", config, MINI_WALL)
    doc["name"] = f"fullwave_indicator_mini_{config}" + ("_closed" if closed else "")
    doc["fullwave"] = {"frequencies": MINI_FREQUENCIES}
    if closed:
        doc["fullwave"]["boundary"] = "pec"
    doc["mesh"].update(MINI_MESH)
    for region in doc["regions"]:
        region["surface_size"] = MINI_REGION_SIZES[region["name"]]
    return doc


def mini2d_design(surface: float) -> dict:
    doc = slotted_box.design2d("mini", MINI_WALL, surface)
    doc["fullwave"] = {"frequencies": MINI_FREQUENCIES}
    return doc


COAX_MESH = {"order": 2, "size": 0.12, "surface_size": 0.04}
COAX_KR = (0.02, 0.4)


def coax_design(axisymmetric: bool = False) -> dict:
    """A new curved geometry: the shorted coax of benchmarks/fullwave3d_background.py (the solenoid template's
    DMRadio-m^3 topology without its coil, the coil space kept as a mu_r = 1 region) revolved to 3D, read out across
    its annular bottom gap (a 360-degree port), inside a closed sphere of R1 = 0.45 m; kR = 0.02 and 0.4. Curved
    cylindrical and toroidal walls, sharp rectangular wall edges. The axisymmetric twin is the same design in 2D, with
    the coil as its winding (the 2D engine needs a magnet; the coil as a winding and as a mu_r = 1 region give the same
    Z, benchmarks/fullwave3d_background.py part A)."""
    import fullwave3d_background as bg
    doc = bg.design(axisymmetric, None, bg.R1, COAX_KR, boundary="pec", axisymmetric=axisymmetric, mesh=COAX_MESH)
    doc["name"] = "fullwave_indicator_coax" + ("_2d" if axisymmetric else "")
    return doc


def coax2d_design(surface: float) -> dict:
    doc = coax_design(axisymmetric=True)
    doc.setdefault("mesh", {})["surface_size"] = surface
    return doc


def stage1_design(order: int) -> dict:
    """The two-port frame (two gap ports, sharp-edged conductor, closed sphere): fields singular at the edges."""
    doc = json.loads((ROOT / "benchmarks" / "designs" / "stage1_twoport.json").read_text())
    for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
        doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = STAGE1_FREQUENCIES
    doc["mesh"]["order"] = order
    return doc


def run_refine(name: str, doc: dict, levels: int, ranks: int) -> None:
    from axionfem.convergence import refine_and_compare
    wd = RUNS / name
    t0 = time.time()
    if levels == 1:   # the base mesh alone (e.g. to compare indicator variants on it)
        axionfem.Design(doc).solve(wd / "refine_0", verbose=1, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
        return
    out = refine_and_compare(doc, wd, levels, mpi_ranks=ranks, verbose=1)
    out["seconds"] = time.time() - t0
    (wd / "refine_and_compare.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def run_2d(name: str, design, wall: float) -> None:
    import axionfem as af
    for k in (2, 4, 8, 16):
        wd = RUNS / f"{name}_s{k}"
        t0 = time.time()
        r = af.Design(design(wall / k)).solve(wd, verbose=0)
        rows = [(row["f_Hz"], row["L_H"]) for row in r.data["fullwave"]["sweep"]]
        print(f"{name} surface wall/{k}: {rows}, {time.time() - t0:.0f} s", flush=True)


def _rows(path: Path) -> list[dict] | None:
    res = path / "results.json"
    if not res.exists():
        return None
    from axionfem.convergence import fullwave_rows
    d = json.loads(res.read_text())
    rows = fullwave_rows(d)
    for row in rows:
        row["dofs"] = d.get("mesh", {}).get("dofs")
        row["elements"] = d.get("mesh", {}).get("elements")
        row["L_p_MQS_H"] = d.get("coupling", {}).get("L_p_H")
    return rows


def _reference_2d(name: str) -> dict:
    """The finest 2D L per frequency, and its change from the next finest."""
    finest = _rows(RUNS / f"{name}_s16")
    second = _rows(RUNS / f"{name}_s8")
    if not finest:
        return {}
    out = {}
    for j, row in enumerate(finest):
        change = abs(second[j]["L_H"] / row["L_H"] - 1) if second else None
        out[row["f_Hz"]] = {"L_H": row["L_H"], "change_from_s8": change}
    return out


def _signed(indicator: dict, order: int = 2) -> float | None:
    """The signed estimate of (L_h - L) / L for a lossless model, -(eta_m^2 - eta_e^2) / (omega^2 L): the magnetic and
    electric errors enter L with opposite signs (docs/physics/fullwave_error_indicator.md). Formed from the reported
    parts: eta_m^2 = (1 - s) m^2 and eta_e^2 = s e^2 of ||E||^2, with omega^2 L = (eta_m^2 + eta_e^2) / L_estimate.
    Current runs report it themselves; this fallback, for older runs, withholds it at order 2 where the two
    parts are within a factor 3 of each other (the rule of those runs; the released code uses a factor 8)."""
    if "L_signed_relative_error_estimate" in indicator or "L_signed_withheld" in indicator:
        return indicator.get("L_signed_relative_error_estimate")
    if indicator.get("L_relative_error_estimate") is None:
        return None
    s = indicator["electric_norm_share"]
    magnetic = (1 - s) * indicator["magnetic_relative_error"] ** 2
    electric = s * indicator["electric_relative_error"] ** 2
    if order >= 2 and abs(magnetic - electric) < 0.5 * (magnetic + electric):
        return None
    return -(magnetic - electric) / (magnetic + electric) * indicator["L_relative_error_estimate"]


def _brief(indicator: dict) -> dict:
    keys = ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error", "electric_norm_share",
            "L_relative_error_estimate", "surface_relative_error", "recovery_magnetic_relative_error",
            "Z_signed_error_estimate_ohm", "gradient", "withheld")
    out = {k: indicator[k] for k in keys if k in indicator}
    out["L_signed_relative_error_estimate"] = _signed(indicator)
    out["by_region"] = indicator.get("by_region", {})
    if "largest_element" in indicator:
        out["largest_element"] = indicator["largest_element"]
    return out


REQUIRED = ["tem/tem_runs.json", "cost/cost.json", "stage1_p1/refine_and_compare.json", "stage1_p2/refine_and_compare.json",
            "probe_crack_p1/refine_and_compare.json", "probe_crack_p2/refine_and_compare.json",
            "probe_ridge_p1/refine_and_compare.json", "probe_ridge_p2/refine_and_compare.json",
            "probe_crackopen_p2/refine_and_compare.json", "layer_lossless/refine_and_compare.json",
            "layer_tand001/refine_and_compare.json", "layer_tand01/refine_and_compare.json",
            "layer_sigma/refine_and_compare.json", "pmcplane_on/refine_and_compare.json",
            "pmcplane_off/refine_and_compare.json", "contrast_mid_pmc/refine_and_compare.json",
            "contrast_partition_cu_pmc/n4/results.json",
            "mini_c360/refine_and_compare.json", "mini_n1/refine_and_compare.json",
            "mini_c360_closed/refine_and_compare.json", "coax/refine_and_compare.json", "mini2d_s16/results.json",
            "coax2d_s16/results.json"]


def _development(dev: Path) -> dict:
    """The development runs (a development build, every estimator variant under dev_residual): per study and frequency the
    change of L between the two meshes against the change each variant's signed estimate predicts. Variants: recovery
    (the order-1 estimator, at order 2), residual + full local gradient, residual + surplus gradient, residual +
    recovery's electric part, residual + global gradient (the final estimator)."""
    def variants(ind: dict) -> dict:
        d = ind["dev_residual"]
        n = len(next(iter(d["regions"].values())))
        t = [sum(v[c] for v in d["regions"].values()) for c in range(n)]
        w2L = d["omega"] ** 2 * d["L_H"]
        zz_m, zz_e = t[-2], t[-1]
        out = {"recovery": -(zz_m - zz_e) / w2L, "residual_full_local_gradient": (t[2] + t[5]) / w2L,
               "residual_surplus_gradient": (t[2] + t[8]) / w2L, "residual_recovery_electric": (t[2] + zz_e) / w2L}
        if n == 15:
            out["residual_global_gradient"] = (t[2] + t[11]) / w2L
        return out

    out = {"note": "development runs of a build that exposes every estimator variant side by side; the released "
                   "code reports the residual + global gradient variant only, so these are not regenerable from it",
           "studies": {}}
    for study in sorted(dev.glob("*")):
        levels = sorted(study.glob("refine_[0-9]*"))
        if len(levels) < 2:
            continue
        sweeps = [json.loads((lv / "results.json").read_text())["fullwave"]["sweep"] for lv in levels[:2]]
        rows = []
        for a, b in zip(*sweeps):
            if "dev_residual" not in a["error_indicator"] or "dev_residual" not in b["error_indicator"]:
                continue
            va, vb = variants(a["error_indicator"]), variants(b["error_indicator"])
            change = b["L_H"] / a["L_H"] - 1
            rows.append({"f_Hz": a["f_Hz"], "L_change_relative": change,
                         "predicted_over_observed": {k: (vb[k] - va[k]) / change for k in va if k in vb}})
        out["studies"][study.name] = rows
    exact = dev / "tem" / "tem_exact.json"
    if exact.exists():
        out["tem_exact"] = json.loads(exact.read_text())
    for name, rows in out["studies"].items():
        for row in rows:
            print("development", name, row["f_Hz"], {k: round(v, 3) for k, v in row["predicted_over_observed"].items()})
    return out


def _richardson(values: list[float], rate: float) -> float:
    """The limit from the two finest values at a given rate in h (uniform refinement halves h)."""
    return values[-1] + (values[-1] - values[-2]) / (2 ** rate - 1)


def _probes(rec: dict) -> dict:
    """The crack, the 270-degree ridge and the open crack: per mesh the estimate against the error of L (of Im Z for
    the open crack) at the Richardson limit of the order-2 series at the theoretical rate (1 and 4/3)."""
    out = {}
    for name, study in sorted(rec["refine_and_compare"].items()):
        if not name.startswith("probe_"):
            continue
        kind = name.split("_")[1]
        order2 = rec["refine_and_compare"].get(f"probe_{kind}_p2")
        if not order2:
            continue
        entry = {"rate": PROBE_RATE[kind], "frequencies": []}
        for freq, ref_freq in zip(study["frequencies"], order2["frequencies"]):
            refine = [int(r["refinements"]) for r in ref_freq["rows"]]
            rows_ref = [next(x for x in _rows(RUNS / f"probe_{kind}_p2" / f"refine_{n}")
                             if x["f_Hz"] == ref_freq["f_Hz"]) for n in refine]
            reference_L = _richardson([r["L_H"] for r in rows_ref], PROBE_RATE[kind])
            reference_X = _richardson([-r["Z_im_ohm"] for r in rows_ref], PROBE_RATE[kind])
            rows = []
            for row in freq["rows"]:
                full = next(x for x in _rows(RUNS / name / f"refine_{row['refinements']}") if x["f_Hz"] == freq["f_Hz"])
                ind = full.get("error_indicator", {})
                true = row["L_H"] / reference_L - 1
                signed = ind.get("L_signed_relative_error_estimate")
                unsigned = ind.get("L_relative_error_estimate")
                item = {"refinements": row["refinements"], "dofs": row["dofs"], "L_true_relative_error": true,
                        "signed_over_true": signed / true if signed is not None else None,
                        "unsigned_over_true": unsigned / abs(true) if unsigned is not None else None}
                dz = ind.get("Z_signed_error_estimate_ohm")
                if kind == "crackopen" and dz is not None:   # the error is electric: Im Z (reactance X = -Im Z)
                    item["Z_im_signed_over_true"] = dz["im"] / (full["Z_im_ohm"] + reference_X)
                if ind.get("recovery_magnetic_relative_error") is not None:
                    item["recovery_over_residual_magnetic"] = (ind["recovery_magnetic_relative_error"] /
                                                               ind["magnetic_relative_error"]) ** 2
                rows.append(item)
                print(f"probe {name} f {freq['f_Hz']:.3g} r{row['refinements']}:", json.dumps(item))
            entry["frequencies"].append({"f_Hz": freq["f_Hz"], "L_reference_H": reference_L, "rows": rows})
        out[name] = entry
    return out


def _layers(rec: dict) -> dict:
    """The half-filled lines: per mesh the electric part and the signed estimates, against the lossless line on the
    same mesh (the same field up to the loss); between consecutive meshes the change of Z against the change of the
    signed Z estimate (complex, no reference); and per mesh the signed Z estimate against Z_h - Z at the Richardson
    limit of the three finest meshes at their observed rate (Z converges slowly here: about 1.8 times per level on the
    first meshes)."""
    out = {}
    lossless = rec["refine_and_compare"].get("layer_lossless")
    for name, study in sorted(rec["refine_and_compare"].items()):
        if not name.startswith("layer_"):
            continue
        entry = {"frequencies": []}
        for k, freq in enumerate(study["frequencies"]):
            fulls = [next(x for x in _rows(RUNS / name / f"refine_{r['refinements']}") if x["f_Hz"] == freq["f_Hz"])
                     for r in freq["rows"]]
            Zs = [complex(x["Z_re_ohm"], x["Z_im_ohm"]) for x in fulls]
            limit = None
            rate = None
            if len(Zs) >= 3 and abs(Zs[-1] - Zs[-2]) > 0:
                ratio = abs(Zs[-2] - Zs[-3]) / abs(Zs[-1] - Zs[-2])
                rate = math.log2(ratio)
                limit = Zs[-1] + (Zs[-1] - Zs[-2]) / (ratio - 1)
            base = None
            if lossless and name != "layer_lossless":
                base = [next(x for x in _rows(RUNS / "layer_lossless" / f"refine_{r['refinements']}")
                             if x["f_Hz"] == freq["f_Hz"]) for r in lossless["frequencies"][k]["rows"]]
            rows = []
            for j, (row, full, Z) in enumerate(zip(freq["rows"], fulls, Zs)):
                ind = full.get("error_indicator", {})
                dz = ind.get("Z_signed_error_estimate_ohm")
                item = {"refinements": row["refinements"], "dofs": row["dofs"],
                        "electric_relative_error": ind.get("electric_relative_error"),
                        "magnetic_relative_error": ind.get("magnetic_relative_error"),
                        "L_signed_relative_error_estimate": ind.get("L_signed_relative_error_estimate"),
                        "Z_signed_error_estimate_ohm": dz, "gradient": ind.get("gradient")}
                if base and j < len(base) and ind.get("electric_relative_error"):
                    item["electric_over_lossless"] = (ind["electric_relative_error"] /
                                                      base[j]["error_indicator"]["electric_relative_error"])
                if limit is not None:
                    true = Z - limit
                    item["Z_h_minus_Z_ohm"] = [true.real, true.imag]
                    if dz is not None and abs(true) > 0:
                        q = complex(dz["re"], dz["im"]) / true
                        item["Z_signed_over_true"] = [q.real, q.imag]
                if j + 1 < len(fulls):
                    nxt = fulls[j + 1].get("error_indicator", {}).get("Z_signed_error_estimate_ohm")
                    change = Zs[j + 1] - Z
                    item["Z_change_to_next_ohm"] = [change.real, change.imag]
                    if dz is not None and nxt is not None and abs(change) > 0:
                        q = (complex(nxt["re"], nxt["im"]) - complex(dz["re"], dz["im"])) / change
                        item["predicted_over_observed_change"] = [q.real, q.imag]
                rows.append(item)
                print(f"layer {name} f {freq['f_Hz']:.3g} r{row['refinements']}:", json.dumps(item))
            entry["frequencies"].append({"f_Hz": freq["f_Hz"], "Z_observed_rate": rate,
                                         "Z_reference_ohm": [limit.real, limit.imag] if limit is not None else None,
                                         "rows": rows})
        out[name] = entry
    return out



def _pmc_plane(rec: dict) -> dict:
    """The TEM line with and without an interior pmc plane, per mesh: Z and the estimates (they must agree)."""
    on = rec["refine_and_compare"].get("pmcplane_on")
    off = rec["refine_and_compare"].get("pmcplane_off")
    if not (on and off):
        return {}
    rows = []
    for fa, fb in zip(on["frequencies"], off["frequencies"]):
        for a, b in zip(fa["rows"], fb["rows"]):
            ra = next(x for x in _rows(RUNS / "pmcplane_on" / f"refine_{a['refinements']}") if x["f_Hz"] == fa["f_Hz"])
            rb = next(x for x in _rows(RUNS / "pmcplane_off" / f"refine_{b['refinements']}") if x["f_Hz"] == fb["f_Hz"])
            ia, ib = ra["error_indicator"], rb["error_indicator"]
            item = {"f_Hz": fa["f_Hz"], "refinements": a["refinements"],
                    "Z_difference_ohm": abs(complex(ra["Z_re_ohm"], ra["Z_im_ohm"]) -
                                            complex(rb["Z_re_ohm"], rb["Z_im_ohm"])),
                    "eta_with_plane": ia.get("estimated_relative_error"),
                    "eta_without": ib.get("estimated_relative_error"),
                    "L_signed_with_plane": ia.get("L_signed_relative_error_estimate"),
                    "L_signed_without": ib.get("L_signed_relative_error_estimate")}
            rows.append(item)
            print("pmc plane", json.dumps(item))
    return {"rows": rows}


def _contrast(rec: dict) -> dict:
    """The high-contrast lines: per mesh the electric part, the projection's iterations and the indicator's share of
    the frequency; on one mesh the electric part on several partitions."""
    out = {}
    for name, study in sorted(rec["refine_and_compare"].items()):
        if not name.startswith("contrast_"):
            continue
        rows = []
        for freq in study["frequencies"]:
            for row in freq["rows"]:
                full = next(x for x in _rows(RUNS / name / f"refine_{row['refinements']}") if x["f_Hz"] == freq["f_Hz"])
                ind = full.get("error_indicator", {})
                item = {"f_Hz": freq["f_Hz"], "refinements": row["refinements"], "dofs": row["dofs"],
                        "withheld": ind.get("withheld"), "electric_relative_error": ind.get("electric_relative_error"),
                        "magnetic_relative_error": ind.get("magnetic_relative_error"),
                        "Z_signed_error_estimate_ohm": ind.get("Z_signed_error_estimate_ohm"),
                        "gradient": ind.get("gradient"),
                        "indicator_share": (ind["seconds"] / full["frequency_total_seconds"]
                                            if "seconds" in ind and full.get("frequency_total_seconds") else None)}
                rows.append(item)
                print(f"contrast {name} r{row['refinements']}:", json.dumps(item))
        out[name] = {"rows": rows}
    for path in sorted(RUNS.glob("contrast_partition_*")):
        runs = {}
        for run in sorted(path.glob("n*")):
            res = run / "results.json"
            if not res.exists():
                continue
            row = json.loads(res.read_text())["fullwave"]["sweep"][0]
            ind = row["error_indicator"]
            runs[run.name] = {"accepted": row.get("solver", {}).get("accepted"),
                              "electric_relative_error": ind.get("electric_relative_error"),
                              "estimated_relative_error": ind.get("estimated_relative_error"),
                              "Z_signed_error_estimate_ohm": ind.get("Z_signed_error_estimate_ohm"),
                              "gradient": ind.get("gradient"), "withheld": ind.get("withheld")}
        values = [r["electric_relative_error"] for r in runs.values() if r["electric_relative_error"]]
        spread = (max(values) - min(values)) / max(values) if len(values) > 1 else None
        out[path.name] = {"runs": runs, "electric_spread": spread}
        print(path.name, "electric part's spread over the partitions:", spread)
    return out


def _per_mesh_vs_2d(rec: dict, references: dict) -> dict:
    """Per mesh, the signed estimate against the difference from the 2D twin. That difference also holds the curved
    geometry's own error (about 1e-5 on the coax, 4e-4 on the mini box at 3 MHz), so these are indicative only."""
    out = {}
    for name, ref in references.items():
        study = rec["refine_and_compare"].get(name)
        if not study:
            continue
        out[name] = {"note": "the 3D-2D difference includes the curved geometry's error, which the estimate does "
                             "not see",
                     "frequencies": []}
        for freq in study["frequencies"]:
            if freq["f_Hz"] not in ref:
                continue
            rows = []
            for row in freq["rows"]:
                diff = row["L_H"] / ref[freq["f_Hz"]]["L_H"] - 1
                s = row["L_signed_relative_error_estimate"]
                rows.append({"refinements": row["refinements"], "L_minus_2d_relative": diff,
                             "signed_over_difference": s / diff if s is not None and diff else None})
            out[name]["frequencies"].append({"f_Hz": freq["f_Hz"], "rows": rows})
    return out


def _tem_summary() -> list[dict]:
    path = RUNS / "tem" / "tem_runs.json"
    if not path.exists():
        return []
    runs = json.loads(path.read_text())
    for entry in runs:
        for row in entry["rows"]:
            print(f"tem {entry['case']} p{entry['order']} r{entry['refinements']} f {row['f_Hz']:.3g}:"
                  f" L effectivity {row.get('L_effectivity')}, signed {row.get('L_signed_ratio')},"
                  f" Z effectivity {row.get('Z_effectivity')}, eta {row['indicator'].get('estimated_relative_error')}")
    return runs


def _development_from(source: str) -> dict:
    """The development section of a previous record, unchanged, with where it came from: SOURCE is a git commit (the
    committed record there) or a record file."""
    import hashlib
    path = Path(source)
    if path.is_file():
        text = path.read_text()
        origin = f"the record file {path.name}"
    else:
        rel = OUT.relative_to(ROOT).as_posix()
        commit = subprocess.run(["git", "rev-parse", "--short", source], cwd=ROOT, capture_output=True, text=True)
        shown = subprocess.run(["git", "show", f"{source}:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if commit.returncode or shown.returncode:
            raise SystemExit(f"--development-from {source}: neither a record file nor a commit with {rel}")
        text = shown.stdout
        origin = f"the record committed at {commit.stdout.strip()} ({rel})"
    out = json.loads(text)["development"]
    out["carried_over"] = (f"copied unchanged by `report --development-from` from {origin}, sha256 "
                           f"{hashlib.sha256(text.encode()).hexdigest()}: the development runs "
                           "were not rerun. Their residual problems are the final code's formulation "
                           "(the penalized system is the range(Q) Galerkin problem), so this section checks it "
                           "independently")
    return out


def report(write: bool, development: Path | None = None, provenance: Path | None = None,
           development_from: str | None = None) -> None:
    rec = {"check": "fullwave_indicator", "date": time.strftime("%Y-%m-%d"), "tem": _tem_summary(),
           "refine_and_compare": {}}
    cost = RUNS / "cost" / "cost.json"
    rec["cost"] = json.loads(cost.read_text()) if cost.exists() else []
    for row in rec["cost"]:
        print(f"cost order {row['order']} {row['path']} f {row['f_Hz']:.3g}:"
              f" {100 * row['share']:.1f} % of the frequency")
    ref_mini = _reference_2d("mini2d")
    ref_coax = _reference_2d("coax2d")
    rec["coax2d_reference"] = {str(k): v for k, v in ref_coax.items()}
    rec["mini2d_reference"] = {str(k): v for k, v in ref_mini.items()}
    for path in sorted(RUNS.glob("*/refine_and_compare.json")):
        name = path.parent.name
        if name.endswith("_alt2"):   # the development build's recovery-order experiment, below
            continue
        study = json.loads(path.read_text())
        for freq in study["frequencies"]:
            f = freq["f_Hz"]
            for row in freq["rows"]:
                rows = _rows(RUNS / name / f"refine_{row['refinements']}")
                ind = next(x for x in rows if x["f_Hz"] == f).get("error_indicator", {})
                row["indicator"] = _brief(ind)
                ref2d = {"mini_c360": ref_mini, "coax": ref_coax}.get(name, {})
                if f in ref2d:   # the radiating box and the coax: the same design in 2D
                    row["relative_error_vs_2d"] = abs(row["L_H"] / ref2d[f]["L_H"] - 1)
            ref2d = {"mini_c360": ref_mini, "coax": ref_coax}.get(name, {})
            if f in ref2d:
                freq["reference_vs_2d"] = abs(freq["L_reference_H"] / ref2d[f]["L_H"] - 1)
            # between consecutive meshes: the change of L against what the estimates predict for it, signed
            # (L_h = L (1 + s), so L_fine / L_coarse - 1 = s_fine - s_coarse) and as the drop of the unsigned estimate
            freq["level_changes"] = []
            for coarse, fine in zip(freq["rows"], freq["rows"][1:]):
                s0 = coarse["indicator"].get("L_signed_relative_error_estimate")
                s1 = fine["indicator"].get("L_signed_relative_error_estimate")
                e0 = coarse["L_relative_error_estimate"]
                e1 = fine["L_relative_error_estimate"]
                change = fine["L_H"] / coarse["L_H"] - 1
                signed_change = s1 - s0 if s0 is not None and s1 is not None else None
                freq["level_changes"].append({
                    "refinements": [coarse["refinements"], fine["refinements"]],
                    "L_change_relative": change,
                    "signed_estimate_change": signed_change,
                    "predicted_over_observed": signed_change / change if signed_change is not None and change else None,
                    "estimate_drop": e0 - e1 if e0 is not None and e1 is not None else None,
                    "estimate_ratio": e0 / e1 if e0 and e1 else None})
                print(f"{name} f {f:.3g}: change of L {change:.4g}, predicted {signed_change}")
        rec["refine_and_compare"][name] = study
        for freq in study["frequencies"]:
            for row in freq["rows"]:
                print(f"{name} f {freq['f_Hz']:.3g} ref {row['refinements']} dofs {row['dofs']}: L {row['L_H']:.8e}"
                      f" observed {row['observed_relative_error']} estimate {row['L_relative_error_estimate']}"
                      f" effectivity {row['effectivity']} vs2d {row.get('relative_error_vs_2d')}")
    # the two-port frame against a Richardson limit at its asymptotic order 4/3 (L near convex 90-degree edges) instead
    # of the fitted 1.16: the effectivities depend on that choice
    p1 = rec["refine_and_compare"].get("stage1_p1")
    if p1:
        rec["stage1_order_4_3"] = {}
        for freq in p1["frequencies"]:
            L = [row["L_H"] for row in freq["rows"]]
            limit = L[-1] + (L[-1] - L[-2]) / (2 ** (4 / 3) - 1)
            entry = {"L_limit_H": limit}
            for name in ("stage1_p1", "stage1_p2"):
                study = rec["refine_and_compare"].get(name, {"frequencies": []})
                for f in study["frequencies"]:
                    if f["f_Hz"] == freq["f_Hz"]:
                        entry[name] = [row["L_relative_error_estimate"] / abs(row["L_H"] / limit - 1)
                                       for row in f["rows"]]
                        entry[name + "_signed"] = [
                            row["L_signed_relative_error_estimate"] / (row["L_H"] / limit - 1)
                            if row["L_signed_relative_error_estimate"] is not None else None for row in f["rows"]]
            rec["stage1_order_4_3"][str(freq["f_Hz"])] = entry
            print("stage1 at order 4/3,", freq["f_Hz"], entry)
    rec["probes"] = _probes(rec)
    rec["layered_lines"] = _layers(rec)
    rec["pmc_plane"] = _pmc_plane(rec)
    rec["contrast"] = _contrast(rec)
    rec["per_mesh_vs_2d"] = _per_mesh_vs_2d(rec, {"mini_c360": ref_mini, "coax": ref_coax})
    if development:
        rec["development"] = _development(development)
    elif development_from:   # a record's development section, unchanged (its runs are gone; see the docstring)
        rec["development"] = _development_from(development_from)
    if provenance:
        rec["provenance"] = json.loads(provenance.read_text())
    if not write:
        print("not written (report --write writes", OUT, ")")
        return
    missing = [p for p in REQUIRED if not (RUNS / p).exists()]
    if missing:
        raise SystemExit(f"report --write: runs missing under {RUNS}: {missing}; the record is left as it is")
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", OUT)


USAGE = ("usage: fullwave_indicator.py tem | tem_exact | cost | mini CONFIG [--closed] | stage1 ORDER | coax | "
         "probe crack|ridge|crackopen ORDER | layer lossless|tand001|tand01|sigma | pmcplane | "
         "contrast cu|mid pec|pmc | contrast_partition cu|mid pec|pmc | "
         "mini2d | coax2d | report [--write] [--development DIR] [--provenance FILE] | "
         "mesh mini CONFIG (see the module docstring)")


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("tem", "tem_exact", "cost", "mini", "stage1", "mini2d", "coax", "coax2d",
                             "probe", "layer", "pmcplane", "contrast", "contrast_partition", "report", "mesh"):
        raise SystemExit(USAGE)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    ranks = int(opt("--ranks", "8"))
    tag = f"_{opt('--tag', '')}" if "--tag" in a else ""
    if a[0] == "tem":
        run_tem()
    elif a[0] == "tem_exact":
        run_tem_exact()
    elif a[0] == "cost":
        run_cost(ranks)
    elif a[0] == "mini":
        closed = "--closed" in a
        run_refine(f"mini_{a[1]}{'_closed' if closed else ''}{tag}", mini_design(a[1], closed),
                   int(opt("--levels", "2")), ranks)
    elif a[0] == "stage1":
        run_refine(f"stage1_p{a[1]}{tag}", stage1_design(int(a[1])), int(opt("--levels", "3")), ranks)
    elif a[0] == "probe":   # probe KIND ORDER: singular lines, four uniformly refined meshes
        kind = a[1]
        order = int(a[2])
        doc = singular_line_design(RUNS / "probe_meshes" / f"{kind}_p{order}", kind, order, PROBE_FREQUENCIES[kind])
        run_refine(f"probe_{kind}_p{order}", doc, int(opt("--levels", "4")), ranks)
    elif a[0] == "layer":   # layer TAG: the half-filled lossy lines, four uniformly refined meshes, 100 MHz
        doc = layered_line_design(RUNS / "layer_meshes" / a[1], a[1], [1e8])
        run_refine(f"layer_{a[1]}", doc, int(opt("--levels", "4")), ranks)
    elif a[0] == "pmcplane":   # the TEM line with and without an interior pmc plane, order 2, 100 MHz
        for plane in (True, False):
            name = f"pmcplane_{'on' if plane else 'off'}"
            run_refine(name, pmc_plane_design(RUNS / "pmcplane_meshes" / name, plane, 2, [1e8]),
                       int(opt("--levels", "3")), ranks)
    elif a[0] == "contrast":   # contrast cu|mid pec|pmc: the high-contrast line on uniformly refined meshes, 100 MHz
        doc = contrast_design(RUNS / "contrast_meshes" / f"{a[1]}_{a[2]}", a[1], a[2], 2, [1e8])
        run_refine(f"contrast_{a[1]}_{a[2]}", doc, int(opt("--levels", "4")), ranks)
    elif a[0] == "contrast_partition":   # contrast_partition cu|mid pec|pmc --refinements N --rank-list 1,3,4
        run_contrast_partition(a[1], a[2], int(opt("--refinements", "2")),
                               [int(v) for v in opt("--rank-list", "1,3,4").split(",")])
    elif a[0] == "coax":
        run_refine(f"coax{tag}", coax_design(), int(opt("--levels", "2")), ranks)
    elif a[0] == "mini2d":
        run_2d("mini2d", mini2d_design, MINI_WALL)
    elif a[0] == "coax2d":
        run_2d("coax2d", coax2d_design, COAX_MESH["surface_size"])
    elif a[0] == "mesh":   # mesh-only trial: mesh mini CONFIG (the axf of $AXF_EXE)
        doc = mini_design(a[2])
        wd = RUNS / f"mesh_{a[1]}_{a[2]}"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "design.json").write_text(json.dumps(doc, indent=1))
        exe = os.environ.get("AXF_EXE", str(ROOT / "build" / "axf"))
        r = subprocess.run([exe, "mesh", str(wd / "design.json"), "-o", str(wd), "-v", "1"], capture_output=True,
                           text=True)
        print((r.stdout + r.stderr)[-2000:], flush=True)
    else:
        development = Path(opt("--development", "")) if "--development" in a else None
        provenance = Path(opt("--provenance", "")) if "--provenance" in a else None
        development_from = opt("--development-from", "") if "--development-from" in a else None
        if development and development_from:
            raise SystemExit("report: --development DIR and --development-from SOURCE exclude each other")
        report("--write" in a, development, provenance, development_from)


if __name__ == "__main__":
    main()
