"""Full-wave surface losses of the 3D readout (fullwave sweep rows: surface_Gamma, R_loss_ohm, Q_loss, Q_loss_tuned).

    python benchmarks/fullwave_surface_losses.py coax LEVEL [--ranks N]     (3D sealed coax, mesh level 0, 1, 2, ...)
    python benchmarks/fullwave_surface_losses.py coax2d                     (the same design, axisymmetric engine)
    python benchmarks/fullwave_surface_losses.py solenoid [--ranks N]       (solenoid template, 3D)
    python benchmarks/fullwave_surface_losses.py solenoid2d
    python benchmarks/fullwave_surface_losses.py slotted mini CONFIG [--ranks N] [--geometry-order 1] [--boundary B]
                                                          (the slotted box; c360 or n1)
    python benchmarks/fullwave_surface_losses.py coax_straight [--ranks N]  (level 0 on straight-sided elements, 1 MHz)
    python benchmarks/fullwave_surface_losses.py coax_paths                 (the order-1 coax through four solve paths)
    python benchmarks/fullwave_surface_losses.py mqs_identity TAG [--ranks N]   (pickup.losses of three MQS designs)
    python benchmarks/fullwave_surface_losses.py report [--write]

coax: a sealed coaxial cavity (inner radius a, outer b, copper_classical walls of thickness t) cut by a radial port slab
of height g between two shorted stubs of lengths l1 and l2. The slab's feed J_r ~ 1/r, uniform in z, is the TEM
profile, and the end walls are flat, so the field is exactly TEM at every frequency: H = I(z) / (2 pi r) with
I'' + k^2 I = -i_s'(z), i_s = I0 / g on the slab and I' = 0 at the shorts. Per unit port current
    Gamma_inner = int |I|^2 dz / (2 pi a),  Gamma_outer = int |I|^2 dz / (2 pi b),
    Gamma_top/bottom = |I(end)|^2 ln(b/a) / (2 pi),
Z = -(1/g) int_slab V dz with V = (I' + i_s) / (i omega C'), L_eff = (L - Im dZ/domega) / 2 and the tuned
Q = omega L_eff / (R_s Gamma) (copper_classical: R_s = sqrt(pi f mu0 rho)). The sweep runs from the MQS limit to 1.8 %
below the TEM_1 pole (the half wave of the whole length, where Z and Gamma diverge) and above it. Levels refine the
surface size by 1/sqrt(2) each (h-convergence of Gamma(f)); coax2d is the 2D engine's twin.
solenoid: the solenoid template (rf_losses_3d.py's parameters and order-2 mesh) with full wave at kR = 1e-3 and 1, 10
and 30 MHz: the MQS limit per surface against pickup.losses of the same run, and the 3D against 2D Gamma(f).
slotted: the CORE-type slotted box `mini` of benchmarks/slotted_box.py (the error indicator's small box; the record
keys and run directories of these cases are labelled `abra`) with an rf block on the Nb wall, at 3 and 20 MHz, uniform feed (c360) and one 30-degree
port (n1): the MQS limit of an edge- and port-dominated Gamma. --geometry-order 1 runs straight-sided elements;
--boundary radiation the open exterior, which the n1 port's field reaches through the open slit.
coax_straight: the static limit when both engines integrate polynomials exactly (geometry order 1). coax_paths: the
default solve (AMS, Gauss-law guard: the curl of the guard's representative), the guard off (the curl of the solver's
E), plain STRUMPACK and the tree-cotree gauge (the curl of S_c e), in this process. mqs_identity TAG: the MQS
pickup.losses of three designs (magnetoresistance at order 1, order 2, the mini box's 30-degree port) with the build of
AXF_EXE, into runs/.../mqs_identity_TAG; report compares the tags b1 (a build in which the MQS engine has its own
loss code) and b5 (the shared code) bit for bit, with the largest relative difference.
report prints the comparisons; --write writes benchmarks/records/fullwave_surface_losses.json with the runs' builds
(recorded per run in run.json).
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

import numpy as np  # noqa: E402

import axionfem as af  # noqa: E402
import slotted_box  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

RUNS = ROOT / "runs" / "fullwave_surface_losses"
OUT = ROOT / "benchmarks" / "records" / "fullwave_surface_losses.json"
C = 299792458.0
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C * C)
RHO_CU = 1.7e-10   # copper_classical

COAX = {"a": 0.08, "b": 0.2, "t": 0.04, "g": 0.02, "l1": 0.6, "l2": 0.3}
COAX_F1 = C / (2 * (COAX["l1"] + COAX["l2"] + COAX["g"]))   # TEM_1: 162.93 MHz
COAX_FREQUENCIES = [1e6, 5e7, 1e8, 1.4e8, 1.55e8, 1.6e8, 1.75e8]
COAX_RADIUS = 0.65               # the closed domain sphere
COAX_SIZE = 0.3                  # away from the walls (the exterior is field-free)
COAX_SURFACE = 0.04              # level 0; level n: 0.04 / sqrt(2)^n
# the 2D twin at order 3: within 1e-5 of the closed form at 160 MHz, where order 2 at this size is 6.6e-3 low (the
# discrete pole sits 8e-5 high)
COAX_2D = {"order": 3, "size": 0.015, "surface_size": 0.0075}

SOLENOID = {"R": 1.0, "H": 2.0, "x": 0.534, "tw": 0.05, "g": 0.03, "tc": 0.03, "tp": 0.05}   # rf_losses_3d.py
SOLENOID_3D = {"order": 2, "size": 0.4, "surface_size": 0.13}
SOLENOID_2D = {"order": 2, "size": 0.05, "surface_size": 0.015}
SOLENOID_MHZ = [1.0, 10.0, 30.0]

ABRA_RF = {"model": "power", "Rs_ohm": 1e-8, "f_ref": 1e6, "exponent": 2, "T_K": 0.01}
ABRA = {"mini": {"wall": 0.1, "frequencies": [3e6, 2e7], "radius": None}}
MINI_MESH = {"size": 0.2, "surface_size": 0.1, "curvature_points": 6}   # fullwave_indicator.py
MINI_REGION_SIZES = {"wall": 0.15, "port": 0.1, "gap": 0.1}


# ------------------------------------------------------------------ the sealed coax and its TEM closed form
def _rect(r0, r1, z0, z1):
    return {"rect": {"r": [r0, r1], "z": [z0, z1]}}


def coax_doc(frequencies, mesh) -> dict:
    """Centred on z = 0 (the domain sphere stays small); closed by a PEC sphere of radius COAX_RADIUS, whose
    exterior modes (above 201 MHz for the empty sphere) the sealed coax does not excite."""
    a, b, t, g, l1, l2 = (COAX[k] for k in ("a", "b", "t", "g", "l1", "l2"))
    bottom = -(l1 + l2 + g) / 2
    slab = bottom + l2
    top = slab + g + l1
    return {
        "schema": "axionfem.design/1", "name": "fw_losses_coax",
        "description": "sealed copper coax, radial port slab between two shorted stubs (benchmarks/"
                       "fullwave_surface_losses.py): the exact TEM line at every frequency",
        "regions": [{"name": "inner", "shape": _rect(0.0, a, bottom - t, top + t), "material": "copper_classical"},
                    {"name": "outer", "shape": _rect(b, b + t, bottom - t, top + t), "material": "copper_classical"},
                    {"name": "top", "shape": _rect(a, b, top, top + t), "material": "copper_classical"},
                    {"name": "bottom", "shape": _rect(a, b, bottom - t, bottom), "material": "copper_classical"},
                    {"name": "port", "shape": _rect(a, b, slab, slab + g), "material": "vacuum"},
                    {"name": "coil", "shape": _rect(0.28, 0.38, -0.1, 0.1), "material": "winding",
                     "mesh_size": 0.1}],
        "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
        "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.3]},
        "fullwave": {"frequencies": list(frequencies), "boundary": "pec"},
        "mesh": {**mesh, "domain_radius": COAX_RADIUS},
    }


def coax_surface(level: int) -> float:
    return COAX_SURFACE / math.sqrt(2.0) ** level


def coax_design3d(level: int):
    d2 = af.Design(coax_doc(COAX_FREQUENCIES, COAX_2D))
    # curvature_points 8: the surface size sets the faces on the inner conductor at every level (12.6 to 36 per turn;
    # the default 16 per turn would override level 0 and 1 there, and the levels would not refine by sqrt 2)
    return to_3d(d2, mesh={"order": 2, "size": COAX_SIZE, "surface_size": coax_surface(level), "curvature_points": 8})


def tem_current(f: float, n: int = 64):
    """The TEM line current I(z) per unit port current at f (lossless), on Gauss-Legendre nodes of each segment.
    Returns (segments [(z, w, I, dI)], I(bottom), I(top), Z)."""
    a, b, g, l1, l2 = (COAX[k] for k in ("a", "b", "g", "l1", "l2"))
    w = 2 * math.pi * f
    k = w / C
    Cp = 2 * math.pi * EPS0 / math.log(b / a)
    i_s = 1.0 / g
    # unknowns: A (stub 2: A cos k(z + l2)), C, D (slab: C cos kz + D sin kz), B (stub 1: B cos k(g + l1 - z))
    M = np.array([
        [math.cos(k * l2), -1.0, 0.0, 0.0],                                   # I continuous at 0
        [k * math.sin(k * l2), 0.0, k, 0.0],                                  # I'(0+) - I'(0-) = -i_s
        [0.0, math.cos(k * g), math.sin(k * g), -math.cos(k * l1)],          # I continuous at g
        [0.0, -k * math.sin(k * g), k * math.cos(k * g), -k * math.sin(k * l1)],   # I'(g+) - I'(g-) = +i_s
    ])
    rhs = np.array([0.0, -i_s, 0.0, -i_s])
    A, Cc, D, B = np.linalg.solve(M, rhs)
    x, wq = np.polynomial.legendre.leggauss(n)
    segs = []
    for z0, z1, fn, dfn in (
            (-l2, 0.0, lambda z: A * np.cos(k * (z + l2)), lambda z: -A * k * np.sin(k * (z + l2))),
            (0.0, g, lambda z: Cc * np.cos(k * z) + D * np.sin(k * z),
             lambda z: -Cc * k * np.sin(k * z) + D * k * np.cos(k * z)),
            (g, g + l1, lambda z: B * np.cos(k * (g + l1 - z)), lambda z: B * k * np.sin(k * (g + l1 - z)))):
        z = 0.5 * (z1 - z0) * x + 0.5 * (z1 + z0)
        segs.append((z, 0.5 * (z1 - z0) * wq, fn(z), dfn(z)))
    # Z = -(1/g) int_slab V dz, V = (I' + i_s) / (i omega C')
    z, ws, _, dI = segs[1]
    V = (dI + i_s) / (1j * w * Cp)
    Z = -np.sum(ws * V) / g
    return segs, A, B, Z


def tem_static() -> dict:
    """The k -> 0 limit of the closed form (MQS): the port current splits so that int I dz = 0 (V = 0 at both shorts),
    I = I2 on stub 2, I2 - z/g on the slab, I2 - 1 on stub 1, with I2 = (l1 + g/2) / (l1 + l2 + g)."""
    a, b, g, l1, l2 = (COAX[k] for k in ("a", "b", "g", "l1", "l2"))
    I2 = (l1 + g / 2) / (l1 + l2 + g)
    int_all = I2 ** 2 * l2 + g * (I2 ** 2 - I2 + 1.0 / 3.0) + (1 - I2) ** 2 * l1
    return {"Gamma": {"inner": int_all / (2 * math.pi * a), "outer": int_all / (2 * math.pi * b),
                      "top": (1 - I2) ** 2 * math.log(b / a) / (2 * math.pi),
                      "bottom": I2 ** 2 * math.log(b / a) / (2 * math.pi)}}


def tem_closed_form(f: float) -> dict:
    a, b = COAX["a"], COAX["b"]
    segs, I_bottom, I_top, Z = tem_current(f)
    int_all = sum(float(np.sum(ws * np.abs(I) ** 2)) for _, ws, I, _ in segs)
    int_port = float(np.sum(segs[1][1] * np.abs(segs[1][2]) ** 2))
    w = 2 * math.pi * f
    h = 1e-6 * f   # central difference of the analytic Z for dZ/domega (relative error ~1e-11)
    dZ = (tem_current(f + h)[3] - tem_current(f - h)[3]) / (2 * 2 * math.pi * h)
    L = -Z.imag / w
    L_eff = 0.5 * (L - dZ.imag)
    G = {"inner": int_all / (2 * math.pi * a), "outer": int_all / (2 * math.pi * b),
         "top": abs(I_top) ** 2 * math.log(b / a) / (2 * math.pi),
         "bottom": abs(I_bottom) ** 2 * math.log(b / a) / (2 * math.pi)}
    port = {"inner": int_port / (2 * math.pi * a), "outer": int_port / (2 * math.pi * b)}
    Rs = math.sqrt(math.pi * f * MU0 * RHO_CU)
    R_loss = Rs * sum(G.values())
    out = {"f_Hz": f, "Gamma": G, "Gamma_port_region": port, "Z_im_ohm": Z.imag, "L_H": L, "L_eff_H": L_eff,
           "R_s_ohm": Rs, "R_loss_ohm": R_loss}
    if L > 0:
        out["Q_loss"] = w * L / R_loss
        if L_eff > 0:
            out["Q_loss_tuned"] = w * L_eff / R_loss
    return out


# ------------------------------------------------------------------ runs
def _solve(design, wd: Path, ranks: int | None) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    meta = {"exe": Path(os.environ.get("AXF_EXE") or "").name,
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
            "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "cpp", "python"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
            "ranks": ranks, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    try:
        if ranks:
            design.solve(wd, verbose=1, mpi_ranks=ranks)
        else:
            design.solve(wd, verbose=0)
    except Exception as exc:  # noqa: BLE001  (results.json is written by the C++ solver)
        print("solve raised:", type(exc).__name__, exc, flush=True)
        meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["seconds"] = time.time() - t0
    (wd / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{wd.name}: {meta['seconds']:.0f} s", flush=True)


def run_coax(level: int, ranks: int) -> None:
    _solve(coax_design3d(level), RUNS / f"coax_l{level}", ranks)


def run_coax2d() -> None:
    _solve(af.Design(coax_doc(COAX_FREQUENCIES, COAX_2D)), RUNS / "coax_2d", None)


def solenoid_doc(mesh2d: dict) -> dict:
    d = af.template("solenoid", **SOLENOID, mesh=mesh2d)
    doc = d.doc
    R = 1.0 + SOLENOID["tw"]   # the device radius (outer radius of all regions): the winding's outside
    doc["fullwave"] = {"frequencies": [1e-3 * C / (2 * math.pi * R)] + [m * 1e6 for m in SOLENOID_MHZ]}
    return doc


def run_solenoid(ranks: int) -> None:
    d2 = af.Design(solenoid_doc(SOLENOID_2D))
    _solve(to_3d(d2, mesh=SOLENOID_3D), RUNS / "solenoid_3d", ranks)


def run_solenoid2d() -> None:
    _solve(af.Design(solenoid_doc(SOLENOID_2D)), RUNS / "solenoid_2d", None)


def abra_doc(box: str, config: str, geometry_order: int | None = None, boundary: str = "pec") -> dict:
    """geometry_order 1: straight-sided elements with order-2 fields (the MQS and full-wave operators then integrate
    polynomials exactly, so their static limits share one discrete problem). boundary "radiation": the open exterior
    of the MQS reference, which a partial port's field reaches through the open slit (a closed sphere lowers it)."""
    spec = ABRA[box]
    doc = slotted_box.design(box, config, spec["wall"], geometry_order=geometry_order)
    doc["name"] = f"fw_losses_abra_{box}_{config}"
    doc.setdefault("materials", {})["nb_rf"] = {"kind": "superconductor", "rf": dict(ABRA_RF)}
    for region in doc["regions"]:
        if region["material"] == "nb":
            region["material"] = "nb_rf"
    doc["fullwave"] = {"frequencies": list(spec["frequencies"]), "boundary": boundary}
    if box == "mini":
        doc["mesh"].update(MINI_MESH)
        for region in doc["regions"]:
            region["surface_size"] = MINI_REGION_SIZES[region["name"]]
    if spec["radius"]:
        doc["mesh"]["domain_radius"] = spec["radius"]
    return doc


def run_coax_straight(ranks: int) -> None:
    d = coax_design3d(0)
    d.doc["mesh"]["geometry_order"] = 1
    d.doc["fullwave"]["frequencies"] = [1e6]
    _solve(d, RUNS / "coax_straight", ranks)


COAX_PATHS = {"default": {},
              "guard_off": {"AXF_FW_GAUSS_GUARD": "off"},
              "strumpack": {"AXF_FW_SOLVER": "strumpack"},
              "tree_cotree": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree"}}


def run_coax_paths() -> None:
    """The order-1 coax at 0.1, 1 and 100 MHz through each solve path, in this process (the engine reads the variables
    when the analysis starts)."""
    for name, env in COAX_PATHS.items():
        for k in ("AXF_FW_GAUSS_GUARD", "AXF_FW_SOLVER", "AXF_FW_GAUGE"):
            os.environ.pop(k, None)
        os.environ.update(env)
        d = to_3d(af.Design(coax_doc([1e5, 1e6, 1e8], COAX_2D)), mesh={"order": 1, "size": 0.3, "surface_size": 0.05})
        _solve(d, RUNS / f"coax_paths_{name}", None)


def mqs_identity_designs():
    """Three MQS designs whose pickup.losses exercise the shared face loop: the solenoid template with a Kohler
    magnetoresistance at order 1, the template at order 2, and the small slotted box with one 30-degree port."""
    d2 = af.template("solenoid", **SOLENOID, mesh=SOLENOID_2D)
    d2.doc.setdefault("materials", {})["cu_mr"] = {
        "kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-10, "T_K": 0.03,
                                    "magnetoresistance": {"model": "kohler", "B0_T": 0.3}}}
    for region in d2.doc["regions"]:
        if region["material"] == "copper":
            region["material"] = "cu_mr"
    yield "solenoid_mr_o1", to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 1})
    plain = af.template("solenoid", **SOLENOID)
    yield "solenoid_o2", to_3d(plain, mesh={"size": 0.4, "surface_size": 0.13, "order": 2})
    doc = slotted_box.design("mini", "n1", ABRA["mini"]["wall"])
    doc["mesh"].update(MINI_MESH)
    for region in doc["regions"]:
        region["surface_size"] = MINI_REGION_SIZES[region["name"]]
    yield "abra_mini_n1", af.Design(doc)


def run_mqs_identity(tag: str, ranks: int) -> None:
    for name, d in mqs_identity_designs():
        _solve(d, RUNS / f"mqs_identity_{tag}" / name, ranks)


def run_abra(box: str, config: str, ranks: int, geometry_order: int | None = None, boundary: str = "pec") -> None:
    tag = (f"_g{geometry_order}" if geometry_order else "") + ("_rad" if boundary == "radiation" else "")
    _solve(af.Design(abra_doc(box, config, geometry_order, boundary)), RUNS / f"abra_{box}_{config}{tag}", ranks)


# ------------------------------------------------------------------ report
def _load(name: str) -> tuple[dict | None, dict]:
    wd = RUNS / name
    res = wd / "results.json"
    meta = json.loads((wd / "run.json").read_text()) if (wd / "run.json").exists() else {}
    return (json.loads(res.read_text()) if res.exists() else None), meta


def _provenance(d: dict, meta: dict) -> dict:
    return {"commit": meta.get("commit"), "dirty": meta.get("dirty"),
            "exe": meta.get("exe"), "ranks": meta.get("ranks"), "seconds": meta.get("seconds"),
            "dofs": d.get("mesh", {}).get("dofs"), "elements": d.get("mesh", {}).get("elements"),
            "order": d.get("mesh", {}).get("order")}


def _rel(a, b):
    return a / b - 1.0 if b else float("nan")


def coax_report() -> dict:
    out = {"geometry": COAX, "TEM_1_Hz": COAX_F1, "closed_form": [tem_closed_form(f) for f in COAX_FREQUENCIES],
           "levels": {}}
    cf = {row["f_Hz"]: row for row in out["closed_form"]}
    names = [p.name for p in sorted(RUNS.glob("coax_l*"))] + ["coax_2d"]
    for name in names:
        d, meta = _load(name)
        if d is None:
            continue
        mqs = {k: v["Gamma"] for k, v in d["pickup"]["losses"]["surfaces"].items()}
        cf0 = tem_static()
        rows = []
        for row in d["fullwave"]["sweep"]:
            f = row["f_Hz"]
            ref = cf[min(cf, key=lambda x: abs(x - f))]
            G = row["surface_Gamma"]
            r = {"f_Hz": f, "Gamma": G,
                 "Gamma_over_closed_form": {k: G[k] / ref["Gamma"][k] for k in ref["Gamma"]},
                 "total_over_closed_form": sum(G[k] for k in ref["Gamma"]) / sum(ref["Gamma"].values()),
                 "L_over_closed_form": row["L_H"] / ref["L_H"],
                 "R_loss_over_closed_form": row.get("R_loss_ohm", float("nan")) / ref["R_loss_ohm"]}
            for key in ("Q_loss", "Q_loss_tuned"):
                if key in row and key in ref:
                    r[f"{key}_over_closed_form"] = row[key] / ref[key]
            if "L_eff_H" in row:
                r["L_eff_over_closed_form"] = row["L_eff_H"] / ref["L_eff_H"]
                if "energy_magnetic_J_per_A2" in row:   # Foster: L_eff = 4 W_m for a closed lossless structure
                    Wm = row["energy_magnetic_J_per_A2"]["re"][0][0]
                    r["L_eff_over_4Wm"] = row["L_eff_H"] / (4 * Wm)
            if "surface_losses" in row:
                port = {k: v["Gamma_by_field_region"].get("port", 0.0)
                        for k, v in row["surface_losses"]["surfaces"].items() if k in ("inner", "outer")}
                r["port_region_over_closed_form"] = {k: port[k] / ref["Gamma_port_region"][k] for k in port}
                r["H_normal_share"] = row["surface_losses"]["H_normal_share"]
                r["R_loss_over_abs_X"] = row.get("R_loss_over_abs_X")
            ind = row.get("error_indicator", {})
            if "surface_relative_error" in ind:
                r["indicator_surface_relative_error"] = ind["surface_relative_error"]
            # the MQS limit on the same mesh: Gamma(f) / Gamma_MQS against the closed form's own (kL)^2 change
            r["over_MQS_minus_closed_form_ratio"] = {
                k: G[k] / mqs[k] - ref["Gamma"][k] / cf0["Gamma"][k] for k in ref["Gamma"] if k in mqs}
            rows.append(r)
        out["levels"][name] = {"provenance": _provenance(d, meta), "rows": rows,
                               "Gamma_MQS": mqs, "Gamma_MQS_over_closed_form_static":
                                   {k: mqs[k] / cf0["Gamma"][k] for k in cf0["Gamma"] if k in mqs},
                               "summary": d["fullwave"].get("surface_losses")}
        for r in rows:
            print(name, f"f {r['f_Hz'] / 1e6:8.2f} MHz",
                  "Gamma/cf", {k: f"{v - 1:+.2e}" for k, v in r["Gamma_over_closed_form"].items()},
                  f"L/cf {r['L_over_closed_form'] - 1:+.2e}",
                  f"Qt/cf {r.get('Q_loss_tuned_over_closed_form', float('nan')) - 1:+.2e}", flush=True)
    # observed convergence order of each surface's Gamma between the levels (surface size ratio sqrt 2)
    levels = sorted(k for k in out["levels"] if k.startswith("coax_l"))
    rates = {}
    for fine, coarse in zip(levels[1:], levels[:-1]):
        rf, rc = out["levels"][fine]["rows"], out["levels"][coarse]["rows"]
        per = []
        for a, b in zip(rc, rf):
            per.append({"f_Hz": a["f_Hz"], "rate": {k: math.log(abs(a["Gamma_over_closed_form"][k] - 1) /
                                                                 abs(b["Gamma_over_closed_form"][k] - 1))
                                                    / math.log(math.sqrt(2.0))
                                                    for k in a["Gamma_over_closed_form"]
                                                    if abs(b["Gamma_over_closed_form"][k] - 1) > 0}})
        rates[f"{coarse}->{fine}"] = per
    out["observed_rates"] = rates
    # Richardson extrapolation of each surface's Gamma(f) from the three finest levels (axionfem.convergence)
    if len(levels) >= 3:
        from axionfem.convergence import richardson
        three = levels[-3:]
        h = [coax_surface(int(name.removeprefix("coax_l"))) for name in three]
        extrapolation = []
        for i, f in enumerate(COAX_FREQUENCIES):
            ref = cf[f]
            per = {"f_Hz": f, "levels": three}
            for k in ref["Gamma"]:
                values = [out["levels"][name]["rows"][i]["Gamma"][k] for name in three]
                r = richardson(h, values)
                # the formal order 2 from the two finest levels (h ratio sqrt 2: G + (G - G_coarse) / (2 - 1))
                order2 = 2 * values[-1] - values[-2]
                per[k] = {"finest_over_closed_form": values[-1] / ref["Gamma"][k] - 1, "observed_order": r["order"],
                          "behaviour": r.get("behaviour"),
                          "extrapolated_over_closed_form": (r["extrapolated"] / ref["Gamma"][k] - 1
                                                            if r["extrapolated"] else None),
                          "gci_fine": r["gci_fine"],
                          "order2_extrapolated_over_closed_form": order2 / ref["Gamma"][k] - 1}
            extrapolation.append(per)
            print("richardson", f"{f / 1e6:7.2f} MHz",
                  {k: (round(per[k]["observed_order"] or float("nan"), 2),
                       f"{per[k]['finest_over_closed_form']:+.1e}",
                       f"{per[k]['order2_extrapolated_over_closed_form']:+.1e}")
                   for k in ref["Gamma"]}, flush=True)
        out["richardson"] = extrapolation
    return out


def solenoid_report() -> dict:
    out = {}
    d3, m3 = _load("solenoid_3d")
    d2, m2 = _load("solenoid_2d")
    if d3 is None:
        return out
    mqs3 = {k: v["Gamma"] for k, v in d3["pickup"]["losses"]["surfaces"].items()}
    rows = []
    for i, row in enumerate(d3["fullwave"]["sweep"]):
        r = {"f_Hz": row["f_Hz"], "k_R": row["k_R"], "Gamma": row["surface_Gamma"],
             "over_MQS_same_run": {k: row["surface_Gamma"][k] / mqs3[k] - 1 for k in mqs3},
             "R_loss_ohm": row.get("R_loss_ohm"), "Q_loss": row.get("Q_loss"), "Q_loss_tuned": row.get("Q_loss_tuned"),
             "L_over_MQS": row.get("L_over_MQS")}
        if d2 is not None:
            row2 = d2["fullwave"]["sweep"][i]
            r["over_2D"] = {k: row["surface_Gamma"][k] / row2["surface_Gamma"][k] - 1 for k in row2["surface_Gamma"]}
            r["R_loss_over_2D"] = row.get("R_loss_ohm", float("nan")) / row2.get("R_loss_ohm", float("nan")) - 1
            r["L_over_2D"] = row["L_H"] / row2["L_H"] - 1
        rows.append(r)
        print("solenoid", f"f {row['f_Hz'] / 1e6:9.4f} MHz kR {row['k_R']:.3g}",
              "FW/MQS-1", {k: f"{v:+.2e}" for k, v in r["over_MQS_same_run"].items()},
              "3D/2D-1", {k: f"{v:+.2e}" for k, v in r.get("over_2D", {}).items()}, flush=True)
    out = {"params": SOLENOID, "mesh3d": SOLENOID_3D, "mesh2d": SOLENOID_2D, "provenance": _provenance(d3, m3),
           "provenance_2d": _provenance(d2, m2) if d2 else None, "Gamma_MQS": mqs3, "rows": rows,
           "summary": d3["fullwave"].get("surface_losses")}
    return out


def abra_report() -> dict:
    out = {}
    for wd in sorted(RUNS.glob("abra_*")):
        d, meta = _load(wd.name)
        if d is None:
            continue
        mqs = d["pickup"]["losses"]["surfaces"]
        rows = []
        for row in d["fullwave"]["sweep"]:
            r = {"f_Hz": row["f_Hz"], "k_R": row["k_R"], "Gamma": row["surface_Gamma"],
                 "over_MQS_same_run": {k: row["surface_Gamma"][k] / mqs[k]["Gamma"] - 1 for k in mqs},
                 "by_region_over_MQS": {k: {reg: v / mqs[k]["Gamma_by_field_region"][reg] - 1
                                            for reg, v in row["surface_losses"]["surfaces"][k]
                                            ["Gamma_by_field_region"].items()} for k in mqs},
                 "L_over_MQS": row.get("L_over_MQS"), "Q_loss": row.get("Q_loss"),
                 "Q_loss_tuned": row.get("Q_loss_tuned"), "R_loss_ohm": row.get("R_loss_ohm")}
            ind = row.get("error_indicator", {})
            if "surface_relative_error" in ind:
                r["indicator_surface_relative_error"] = ind["surface_relative_error"]
            rows.append(r)
            print(wd.name, f"f {row['f_Hz'] / 1e6:.3f} MHz kR {row['k_R']:.3g}",
                  "FW/MQS-1", {k: f"{v:+.2e}" for k, v in r["over_MQS_same_run"].items()},
                  "by region", r["by_region_over_MQS"], flush=True)
        out[wd.name] = {"provenance": _provenance(d, meta), "Gamma_MQS": {k: v["Gamma"] for k, v in mqs.items()},
                        "rows": rows, "summary": d["fullwave"].get("surface_losses")}
    return out


def extra_report() -> dict:
    """coax_straight, coax_paths and mqs_identity (b1 against b3), when their runs are there."""
    out = {}
    d, meta = _load("coax_straight")
    if d is not None:
        mqs = {k: v["Gamma"] for k, v in d["pickup"]["losses"]["surfaces"].items()}
        static = tem_static()["Gamma"]
        row = d["fullwave"]["sweep"][0]
        ref = tem_closed_form(row["f_Hz"])["Gamma"]
        out["coax_straight_order2_mqs_limit"] = {
            "provenance": _provenance(d, meta), "f_Hz": row["f_Hz"],
            "over_MQS_minus_closed_form_ratio": {k: row["surface_Gamma"][k] / mqs[k] - ref[k] / static[k] for k in ref},
            "Gamma_over_closed_form": {k: row["surface_Gamma"][k] / ref[k] - 1 for k in ref}}
    runs = {name: _load(f"coax_paths_{name}") for name in COAX_PATHS}
    if all(d is not None for d, _ in runs.values()):
        base = runs["default"][0]["fullwave"]["sweep"]
        worst = {}
        for name, (d, _) in runs.items():
            dev = 0.0
            for r, r0 in zip(d["fullwave"]["sweep"], base):
                values = [r["surface_Gamma"][k] / r0["surface_Gamma"][k] - 1 for k in r0["surface_Gamma"]]
                values += [r["R_loss_ohm"] / r0["R_loss_ohm"] - 1, r["Q_loss_tuned"] / r0["Q_loss_tuned"] - 1]
                dev = max(dev, max(abs(v) for v in values))
            worst[name] = dev
        out["coax_order1_solve_paths"] = {
            "provenance": {name: _provenance(d, meta) for name, (d, meta) in runs.items()},
            "solver_path": {name: d["fullwave"]["solver_path"] for name, (d, _) in runs.items()},
            "frequencies_Hz": [r["f_Hz"] for r in base], "max_relative_deviation_from_default": worst}
    # pickup.losses before the shared helpers (b1) against the final build (MQS_IDENTITY_TAG): bit for bit, and the
    # largest relative difference of any number (the magnetoresistance weight is Rs_factor(B), which
    # can differ from R_s(1 Hz, B) / R_s(1 Hz) by a rounding)
    a_dir, b_dir = RUNS / "mqs_identity_b1", RUNS / f"mqs_identity_{MQS_IDENTITY_TAG}"
    if a_dir.exists() and b_dir.exists():
        same = {}
        largest = {}
        provenance = {}
        for name, _ in mqs_identity_designs():
            a, ma = _load(f"mqs_identity_b1/{name}")
            b, mb = _load(f"mqs_identity_{MQS_IDENTITY_TAG}/{name}")
            if a is None or b is None:
                continue
            same[name] = a["pickup"]["losses"] == b["pickup"]["losses"]
            largest[name] = _largest_relative_difference(a["pickup"]["losses"], b["pickup"]["losses"])
            provenance[name] = {"b1": _provenance(a, ma), MQS_IDENTITY_TAG: _provenance(b, mb)}
        out["mqs_pickup_losses_bit_identical"] = {"against": MQS_IDENTITY_TAG, "designs": same,
                                                 "largest_relative_difference": largest, "provenance": provenance}
    return out


MQS_IDENTITY_TAG = "b5"   # the final build of the comparison


def _largest_relative_difference(a, b) -> float:
    """The largest |a - b| / max(|a|, |b|) over the numbers of two JSON trees of the same shape (inf if they differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_largest_relative_difference(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max([_largest_relative_difference(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        scale = max(abs(a), abs(b))
        return abs(a - b) / scale if scale > 0 else 0.0
    return 0.0 if a == b else float("inf")


def report(write: bool) -> None:
    rec = {"check": "fullwave_surface_losses", "date": time.strftime("%Y-%m-%d"),
           "coax": coax_report(), "solenoid": solenoid_report(), "abra": abra_report(), "extra": extra_report()}
    if write:
        OUT.write_text(json.dumps(rec, indent=1) + "\n")
        print("wrote", OUT)


def main() -> None:
    a = sys.argv[1:]
    ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
    if a[0] == "coax":
        run_coax(int(a[1]), ranks)
    elif a[0] == "coax2d":
        run_coax2d()
    elif a[0] == "solenoid":
        run_solenoid(ranks)
    elif a[0] == "solenoid2d":
        run_solenoid2d()
    elif a[0] == "coax_straight":
        run_coax_straight(ranks)
    elif a[0] == "coax_paths":
        run_coax_paths()
    elif a[0] == "mqs_identity":
        run_mqs_identity(a[1], ranks)
    elif a[0] == "slotted":
        run_abra(a[1], a[2], ranks, int(a[a.index("--geometry-order") + 1]) if "--geometry-order" in a else None,
                 a[a.index("--boundary") + 1] if "--boundary" in a else "pec")
    elif a[0] == "report":
        report("--write" in a)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
