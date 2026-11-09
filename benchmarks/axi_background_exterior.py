"""The axisymmetric engine's exterior term for an imposed grid background (cpp/src/model.cpp, axi::BackgroundExterior).

    python benchmarks/axi_background_exterior.py run --tag after [--parts A,R,M,N,Q,C,U,I,F] [--only NAME,...]
        [--ranks N]
    python benchmarks/axi_background_exterior.py report

The axisymmetric TM sweep couples a background in the reciprocal form (1/I) int A_bg mu H dV. It
includes the part outside the domain sphere Rb for a grid background, int_{rho > Rb} A_bg mu0 H' dV, from the outgoing
multipoles of H' and the moments of the grid that the 3D sweep uses (the grid up to its inscribed sphere, its exterior
multipoles beyond; docs/physics/fullwave_background.md). Designs and frequencies are those of the 3D validation
(benchmarks/fullwave3d_background.py): a shorted coax read out across its bottom gap inside a small solenoid
(R = 0.2 m, H = 0.3 m), its field as a grid background (the coil region kept, as vacuum), or a 1.8 m magnet of the same
radii and current density; kR = 0.02, 0.4, 0.8 with R = 0.24 m.

A. 2D against 3D: the 3D Psi(omega) of the committed record benchmarks/records/fullwave3d_background.json against the 2D
   runs, before (tag "before": a build without the term) and after; also the coil as the 2D winding (its exterior term
   from static multipoles) on the same mesh, and the long magnet as a 2D winding inside a domain that encloses it
   (R1 = 1.2 m), the two independent references.
R. Independence of the domain radius: the coil grid, the long magnet grid and the coil winding (control) at R1 = 0.36
   to 1.2 m.
M. The MQS limit: the exterior term against kR from 0.0025 to 0.32 (Re ~ (kR)^2, Im ~ (kR)^3 expected).
N. Convergence in the multipole order (fullwave.dtn_modes: the exterior map and the exterior term together).
Q. The background moments' radial panels (AXF_FW_BG_PANELS).
C. A grid that does not reach past the domain sphere (the coil's grid cut to r <= 0.4 m, |z| <= 0.4 m): at R1 = 0.6 m
   it ends inside the sphere, at R1 = 0.45 m it is cut at its box; labels, warnings and Psi against the full grid and
   the winding.
U. Uniform backgrounds (1 T): the background's own exterior term stays omitted (it diverges); A_lab's
   term (windings, superconductor reactions) is included, as in 3D. How far
   Psi(omega) moves between R1 = 0.45 and 0.675 m (a superconducting disk, superconducting coax walls, the coil as the
   design's winding; a CORE-like cavity at R1 = 0.8 and 1.2 m), A_lab's term, its agreement with the same field given
   as a 1 T grid, and the change against a build without the term, which must be exactly that term.
V. The disk design in a uniform 1 T in 3D (part F's coarse mesh), against 2D with and without the exterior term.
I. Invariance: designs without a background, uniform backgrounds, a closed grid design, a TE sweep with a uniform
   background and eigenmode-only runs, with both builds; every leaf of results.json compared, numbers and strings apart
   (numbers bit-identical expected; run with OMP_NUM_THREADS=1).
F. The 3D sweep after moving its background moments to axi::BackgroundExterior: a coarse 3D grid design with both
   builds (bit-identical expected; --ranks MPI ranks).

Run directories: runs/axi_background_exterior/<tag>/<name> (AXF_BG2D_RUNS overrides the root). Writes
benchmarks/records/axi_background_exterior.json.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence
sys.path.insert(0, str(HERE))

import fullwave3d_background as fb  # noqa: E402

RUNS = Path(os.environ.get("AXF_BG2D_RUNS", ROOT / "runs" / "axi_background_exterior")).resolve()
OUT = ROOT / "benchmarks" / "records" / "axi_background_exterior.json"
REC3D = ROOT / "benchmarks" / "records" / "fullwave3d_background.json"

KR = fb.KR
R1 = fb.R1
R1_SCAN = (0.36, 0.45, 0.54, 0.675, 0.9, 1.2)
KR_MQS = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
DTN_SCAN = (4, 6, 8, 12, 16, 24, 40, 80, 160)
PANEL_SCAN = (16, 32, 64, 128, 256)
KR_CORE = (0.02, 0.2, 0.4)   # part U's CORE-like cavity (R = 0.502 m)
LONG_R1 = 1.2           # a domain that encloses the long magnet (it reaches rho = 0.93 m)
MESH3D = {"order": 2, "size": 0.09, "surface_size": 0.03}   # part F (the slow tests' coarse 3D mesh)


# ---------------------------------------------------------------- grids (cached: the analytic field is slow)
def grid_specs() -> dict:
    return {"coil": dict(extent_r=2.0, extent_z=2.0),
            "coil_half": dict(extent_r=1.0, extent_z=1.0),
            "long": dict(extent_r=2.0, extent_z=3.0, z1=-fb.LONG_HALF, z2=fb.LONG_HALF, fine_z=1.1),
            "long_narrow": dict(extent_r=0.8, extent_z=3.0, z1=-fb.LONG_HALF, z2=fb.LONG_HALF, fine_z=1.1),
            "coil_coarse": dict(extent_r=2.0, extent_z=2.0, h_fine=0.004, h=0.02)}


def grid(name: str) -> dict:
    path = RUNS / "grids" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    block = fb.grid(**grid_specs()[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")   # two jobs may build the same grid
    tmp.write_text(json.dumps(block))
    os.replace(tmp, path)
    print(f"grid {name}: {len(block['r'])} x {len(block['z'])} nodes, {time.time() - t0:.0f} s", flush=True)
    return block


# ---------------------------------------------------------------- designs
def axi(background: str | None, domain_radius: float, kr: tuple, boundary: str = "radiation",
        dtn_modes: int | None = None, plug: bool = False) -> dict:
    """The 3D validation's model on the axisymmetric engine; background: a grid name, "uniform", or None (the coil as
    the design's winding)."""
    bg = None
    if background == "uniform":
        bg = {"type": "uniform", "Bz": 1.0}
    elif background is not None:
        bg = grid(background)
    doc = fb.design(background is None, bg, domain_radius, kr, boundary=boundary, axisymmetric=True, plug=plug)
    if dtn_modes is not None:
        doc["fullwave"]["dtn_modes"] = dtn_modes
    return doc


def axi_doc(background: dict, domain_radius: float) -> dict:
    """The model with a given background block, kR = 0.02, 0.4, 0.8."""
    return fb.design(False, background, domain_radius, KR, axisymmetric=True)


def long_winding(domain_radius: float, kr: tuple) -> dict:
    """The long magnet (1.8 m, the coil's radii and current density) as the 2D design's winding."""
    doc = fb.design(True, None, domain_radius, kr, axisymmetric=True)
    doc = copy.deepcopy(doc)
    for region in doc["regions"]:
        if region["name"] == "winding":
            region["shape"] = {"rect": {"r": [fb.R, fb.R + fb.TW], "z": [-fb.LONG_HALF, fb.LONG_HALF]}}
    for winding in doc["windings"]:
        winding["ampere_turns"] = fb.NI * 2 * fb.LONG_HALF / fb.H
    return doc


def toroid_te(background: dict | None = None) -> dict:
    """A TE (azimuthal readout) sweep: a path the change must leave alone (a poloidal background does not couple)."""
    import axionfem as af
    d = af.template("toroid", R=2, H=2)
    doc = d.doc
    doc.pop("benchmarks", None)
    if background is not None:
        doc["background"] = background
    doc["fullwave"] = {"frequencies": [1.0e6, 1.0e7]}
    return doc


def crop(g: dict, r_max: float, z_lo: float, z_hi: float) -> dict:
    """The grid cut to r <= r_max, z_lo <= z <= z_hi (the field beyond is then taken as zero)."""
    nr = len(g["r"])
    ir = [i for i, r in enumerate(g["r"]) if r <= r_max + 1e-12]
    iz = [j for j, z in enumerate(g["z"]) if z_lo - 1e-12 <= z <= z_hi + 1e-12]
    return {"type": "grid", "r": [g["r"][i] for i in ir], "z": [g["z"][j] for j in iz],
            "Br": [g["Br"][j * nr + i] for j in iz for i in ir],
            "Bz_grid": [g["Bz_grid"][j * nr + i] for j in iz for i in ir]}


def uniform_grid(extent: float = 2.0, h: float = 0.05) -> dict:
    """1 T along z as a grid over r <= extent, |z| <= extent: psi = r^2/2 is exact on it, so the domain integral is the
    uniform background's, and the engine reports the static field's own exterior term (which it omits for "uniform")."""
    nr = int(round(extent / h)) + 1
    nz = int(round(2 * extent / h)) + 1
    r = [extent * i / (nr - 1) for i in range(nr)]
    z = [-extent + 2 * extent * j / (nz - 1) for j in range(nz)]
    return {"type": "grid", "r": r, "z": z, "Br": [0.0] * (nr * nz), "Bz_grid": [1.0] * (nr * nz)}


def uniform_design(kind: str, domain_radius: float, as_grid: bool = False) -> dict:
    """A uniform 1 T background with a reaction field or windings: "plug" (a superconducting disk above the coax),
    "sccoax" (superconducting coax walls), "windings" (the coil as the design's winding)."""
    bg = uniform_grid() if as_grid else {"type": "uniform", "Bz": 1.0}
    if kind == "windings":
        doc = fb.design(True, None, domain_radius, KR, axisymmetric=True)
        doc["background"] = bg
        return doc
    doc = fb.design(False, bg, domain_radius, KR, axisymmetric=True, plug=kind == "plug")
    if kind == "sccoax":
        for region in doc["regions"]:
            if region["name"] in ("coax_inner", "coax_outer", "coax_short"):
                region["material"] = "superconductor"
    return doc


def core_uniform(domain_radius: float, as_grid: bool = False) -> dict:
    """A CORE-like cavity (`core` template, Rc = 0.2 m, H = 0.4 m; the coils as mu_r = 1 regions) in a uniform 1 T,
    kR = 0.02, 0.2, 0.4 with R = 0.502 m."""
    import axionfem as af
    d = af.template("core", Rc=0.2, H=0.4)
    doc = copy.deepcopy(d.doc)
    doc.pop("benchmarks", None)
    doc["windings"] = []
    doc.setdefault("materials", {})["coil_space"] = {"kind": "linear", "mu_r": 1.0}
    for region in doc["regions"]:
        if region["name"] in ("coil_upper", "coil_lower"):
            region["material"] = "coil_space"
    doc.setdefault("mesh", {})["domain_radius"] = domain_radius
    doc["background"] = uniform_grid(extent=3.0) if as_grid else {"type": "uniform", "Bz": 1.0}
    r_dev = 0.2 * 2.5 + 0.2 * 0.01
    doc["fullwave"] = {"frequencies": [k * fb.C / (2 * math.pi * r_dev) for k in KR_CORE], "boundary": "radiation"}
    return doc


def eigen_only(background: str | None) -> dict:
    """Eigenmodes only (no sweep), with a TM pickup: no Psi(omega), so no sweep labels."""
    doc = axi(background, R1, (0.4,))
    doc["fullwave"] = {"modes": 2, "boundary": "radiation"}
    return doc


def models() -> dict:
    """name -> (part, builder, ranks, env); env: extra environment for the solve"""
    m = {}
    # A: the 3D validation's models (names as in its record), and the two 2D references
    m["A_grid"] = ("A", lambda: axi("coil", R1, KR), 1, {})
    m["A_grid_half"] = ("A", lambda: axi("coil_half", R1, KR), 1, {})
    m["A_grid_R1x1.5"] = ("A", lambda: axi("coil", 1.5 * R1, (0.4, 0.8)), 1, {})
    m["A_windings"] = ("A", lambda: axi(None, R1, KR), 1, {})
    m["B_long"] = ("A", lambda: axi("long", R1, KR), 1, {})
    m["B_long_R1x1.5"] = ("A", lambda: axi("long", 1.5 * R1, KR), 1, {})
    m["B_long_narrow_grid"] = ("A", lambda: axi("long_narrow", R1, (0.8,)), 1, {})
    m["B_long_windings_R1.2"] = ("A", lambda: long_winding(LONG_R1, KR), 1, {})
    # a superconducting disk next to the coax screens the coil's field: a reaction field (its own exterior term, from
    # A_lab without the background) next to the grid's
    m["P_plug_grid"] = ("A", lambda: axi("coil", R1, KR, plug=True), 1, {})
    m["P_plug_grid_R1x1.5"] = ("A", lambda: axi("coil", 1.5 * R1, KR, plug=True), 1, {})
    m["P_plug_windings"] = ("A", lambda: axi(None, R1, KR, plug=True), 1, {})
    # R: the domain radius
    for rb in R1_SCAN:
        m[f"R_coil_{rb}"] = ("R", lambda rb=rb: axi("coil", rb, KR), 1, {})
        m[f"R_long_{rb}"] = ("R", lambda rb=rb: axi("long", rb, KR), 1, {})
        m[f"R_windings_{rb}"] = ("R", lambda rb=rb: axi(None, rb, KR), 1, {})
    # M: the MQS limit
    m["M_long"] = ("M", lambda: axi("long", R1, KR_MQS), 1, {})
    m["M_coil"] = ("M", lambda: axi("coil", R1, KR_MQS), 1, {})
    m["M_windings"] = ("M", lambda: axi(None, R1, KR_MQS), 1, {})
    # N: the multipole order
    for n in DTN_SCAN:
        m[f"N_long_{n}"] = ("N", lambda n=n: axi("long", R1, (0.4, 0.8), dtn_modes=n), 1, {})
    # Q: the moments' radial panels
    for p in PANEL_SCAN:
        m[f"Q_long_{p}"] = ("Q", lambda: axi("long", R1, (0.02, 0.8)), 1, {"AXF_FW_BG_PANELS": str(p)})
    # C: grids that do not reach past the domain sphere
    m["C_inside_R0.6"] = ("C", lambda: axi_doc(crop(grid("coil"), 0.4, -0.4, 0.4), 0.6), 1, {})
    m["C_full_R0.6"] = ("C", lambda: axi("coil", 0.6, KR), 1, {})
    m["C_windings_R0.6"] = ("C", lambda: axi(None, 0.6, KR), 1, {})
    m["C_cut_R0.45"] = ("C", lambda: axi_doc(crop(grid("coil"), 0.4, -0.4, 0.4), R1), 1, {})
    # U: uniform backgrounds
    for kind in ("plug", "sccoax", "windings"):
        for rb in (R1, 1.5 * R1):
            m[f"U_{kind}_{rb:g}"] = ("U", lambda kind=kind, rb=rb: uniform_design(kind, rb), 1, {})
        m[f"UG_{kind}_{R1:g}"] = ("U", lambda kind=kind: uniform_design(kind, R1, as_grid=True), 1, {})
    for rb in (0.8, 1.2):
        m[f"U_core_{rb:g}"] = ("U", lambda rb=rb: core_uniform(rb), 1, {})
    m["UG_core_0.8"] = ("U", lambda: core_uniform(0.8, as_grid=True), 1, {})
    # V: the disk design in a uniform 1 T in 3D
    m["V_uniform_plug_3d"] = ("V", lambda: fb.design(False, {"type": "uniform", "Bz": 1.0}, R1, KR, mesh=MESH3D,
                                                     plug=True), None, {})
    # I: invariance (both builds)
    m["I_windings"] = ("I", lambda: axi(None, R1, KR), 1, {})
    m["I_windings_plug"] = ("I", lambda: axi(None, R1, KR, plug=True), 1, {})
    m["I_toroid_te"] = ("I", toroid_te, 1, {})
    m["I_uniform"] = ("I", lambda: axi("uniform", R1, KR), 1, {})
    m["I_uniform_plug"] = ("I", lambda: axi("uniform", R1, KR, plug=True), 1, {})
    m["I_uniform_pec"] = ("I", lambda: axi("uniform", R1, KR, boundary="pec"), 1, {})
    m["I_grid_pec"] = ("I", lambda: axi("coil", R1, KR, boundary="pec"), 1, {})
    m["I_toroid_te_uniform"] = ("I", lambda: toroid_te({"type": "uniform", "Bz": 1.0}), 1, {})
    m["I_eigen_windings"] = ("I", lambda: eigen_only(None), 1, {})
    m["I_eigen_grid"] = ("I", lambda: eigen_only("coil"), 1, {})
    # F: the 3D sweep (moments moved to axi::BackgroundExterior)
    m["F_grid_3d"] = ("F", lambda: fb.design(False, grid("coil_coarse"), R1, (0.02, 0.8), mesh=MESH3D), None, {})
    return m


def solve(name: str, builder, ranks: int, env: dict, run_dir: Path) -> None:
    import axionfem as af
    t0 = time.time()
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        af.Design(builder()).solve(run_dir, verbose=0, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.1f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (recorded, not fatal)
        print(f"{name} failed: {type(exc).__name__}: {exc}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name,
                "python_core": Path(af._core.__file__).name, "ranks": ranks, "env": env, "seconds": time.time() - t0,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps({"produced_by": produced}) + "\n")


def run(tag: str, parts: list[str], only: list[str] | None, ranks: int) -> None:
    for name, (part, builder, own_ranks, env) in models().items():
        if part not in parts or (only and name not in only):
            continue
        run_dir = RUNS / tag / name
        if (run_dir / "results.json").exists() or (run_dir / "error.txt").exists():
            print(f"{tag}/{name}: kept", flush=True)
            continue
        solve(name, builder, own_ranks or ranks, env, run_dir)


# ---------------------------------------------------------------- report
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


def cx(d: dict | None) -> complex | None:
    return complex(d["re"], d["im"]) if d else None


def load(tag: str, name: str) -> dict | None:
    path = RUNS / tag / name / "results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    fw = res.get("fullwave", {})
    rows = []
    for row in fw.get("sweep", []):
        bg = row.get("Psi_background") or {}
        rows.append({"f_Hz": row["f_Hz"], "k_R": row["k_R"], "Psi": complex(row["Psi_re"], row["Psi_im"]),
                     "Psi_exterior": cx(row.get("Psi_exterior")), "Psi_bg_volume": cx(bg.get("volume")),
                     "Psi_bg_exterior": cx(bg.get("exterior")),
                     "bg_last_order_fraction": bg.get("exterior_last_order_fraction"),
                     "L_H": row["L_H"]})
    env = json.loads((RUNS / tag / name / "env.json").read_text()) if (RUNS / tag / name / "env.json").exists() else {}
    return {"rows": rows, "Psi_MQS": res["coupling"]["Psi_SI_H_T_m2"], "L_MQS": res["coupling"]["L_p_H"],
            "dofs": res.get("mesh", {}).get("dofs"), "background": fw.get("background"),
            "exterior_term": fw.get("exterior_term"), "exterior_term_background": fw.get("exterior_term_background"),
            "warnings": res.get("warnings", []), "produced_by": env.get("produced_by", {})}


def jc(z: complex | None):
    return None if z is None else [z.real, z.imag]


def summary(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k != "rows"}
    out["rows"] = [{k: (jc(v) if isinstance(v, complex) else v) for k, v in r.items()} for r in m["rows"]]
    return out


def at_f(m: dict, f: float) -> dict:
    return min(m["rows"], key=lambda r: abs(r["f_Hz"] - f))


def at(m: dict, kr: float) -> dict:
    return at_f(m, fb.f_of(kr))


def rel(a: complex, b: complex) -> float:
    return abs(a - b) / abs(b)


def bg_ext(row: dict) -> complex:
    return row["Psi_bg_exterior"] if row["Psi_bg_exterior"] is not None else 0j


def load_3d() -> dict:
    rec = json.loads(REC3D.read_text())
    out = {}
    for name, m in rec["models"].items():
        rows = [{"f_Hz": r["f_Hz"], "k_R": r["k_R"], "Psi": complex(*r["Psi"]),
                 "Psi_bg_exterior": complex(*r["Psi_bg_exterior"]) if r["Psi_bg_exterior"] else None,
                 "L_H": r["L_H"]} for r in m["rows"]]
        out[name] = {"rows": rows}
    return out


def leaves(obj, prefix=""):
    """path -> value for every number, string, boolean and null in a JSON tree."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(leaves(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def run_specific(path: str, value) -> bool:
    """Timings, memory peaks, the 3D memory records (`memory` in the sweep rows and in `fullwave`;
    docs/validation/fullwave_memory.md) and the run's own file paths, which differ between any two runs."""
    last = path.rsplit("/", 1)[-1]
    if "seconds" in last or path.startswith("/timing/") or "peak_rss" in last:
        return True
    if "/memory/" in path:
        return True
    if path.startswith("/solver/solves[") and last.endswith("_s"):   # 3D solver phases (setup_s, solve_s, clean_s)
        return True
    return isinstance(value, str) and value.startswith("/")


def compare_results(a_path: Path, b_path: Path) -> dict:
    """Every leaf of two results.json (timings and file paths aside): those that differ (numbers and the rest
    separately), and those in only one."""
    a = leaves(json.loads(a_path.read_text()))
    b = leaves(json.loads(b_path.read_text()))
    differ = []
    differ_numbers = []
    compared = 0
    compared_numbers = 0
    for k in sorted(set(a) & set(b)):
        if run_specific(k, a[k]):
            continue
        compared += 1
        number = isinstance(a[k], (int, float)) and not isinstance(a[k], bool)
        if number:
            compared_numbers += 1
        if a[k] != b[k]:
            differ.append(k)
            if number:
                differ_numbers.append(k)
    only_a = sorted(k for k in set(a) - set(b) if not run_specific(k, a[k]))
    only_b = sorted(k for k in set(b) - set(a) if not run_specific(k, b[k]))
    return {"compared_leaves": compared, "compared_numbers": compared_numbers,
            "n_differing_numbers": len(differ_numbers), "differing_numbers": differ_numbers[:40],
            "n_differing": len(differ), "differing": [{"leaf": k, "before": a[k], "after": b[k]} for k in differ[:10]],
            "n_only_before": len(only_a), "only_before": only_a[:40],
            "n_only_after": len(only_b), "only_after": only_b[:40]}


def report() -> None:
    names = list(models())
    after = {n: load("after", n) for n in names}
    before = {n: load("before", n) for n in names}
    d3 = load_3d()
    rec = {"check": "axi_background_exterior",
           "provenance": {"git_describe": git("describe", "--always", "--dirty").strip(),
                          "git_head": git("rev-parse", "HEAD").strip(),
                          "driver": file_record(__file__), "date": time.strftime("%Y-%m-%d"),
                                                    "reference_3d": {"record": str(REC3D.relative_to(ROOT)), "sha256": sha256(REC3D)}},
           "R1_m": R1, "design": {"R": fb.R, "H": fb.H, "x": fb.X, "tw": fb.TW, "g": fb.G, "tc": fb.TC,
                                  "tp": fb.TP, "NI": fb.NI, "R_dev": fb.R_DEV, "long_half_m": fb.LONG_HALF}}
    builds = {}
    for tag, m in (("before", before), ("after", after)):
        prod = [x["produced_by"] for x in m.values() if x and x.get("produced_by")]
        if prod:
            exe = prod[0].get("AXF_EXE", "")
            builds[tag] = {"AXF_EXE": file_record(exe),
                           "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
                           "python_core": file_record(prod[0].get("python_core", ""))}
    rec["provenance"]["builds"] = builds
    rec["models_after"] = {n: summary(v) for n, v in after.items() if v}
    rec["models_before"] = {n: summary(v) for n, v in before.items() if v and not n.startswith("I_")}
    res = {}

    # A: 2D against 3D (before and after), and the 2D references
    pairs = [("A_grid", "A_grid"), ("A_grid_half", "A_grid_half"), ("A_grid_R1x1.5", "A_grid_R1x1.5"),
             ("A_windings", "A_windings"), ("B_long", "B_long"), ("B_long_R1x1.5", "B_long_R1x1.5"),
             ("B_long_narrow_grid", "B_long_narrow_grid")]
    table = []
    for n2, n3 in pairs:
        a, b, t = after.get(n2), before.get(n2), d3.get(n3)
        if not (a and t):
            continue
        for r3 in t["rows"]:
            ra = at_f(a, r3["f_Hz"])
            entry = {"model": n2, "k_R": r3["k_R"], "Psi_3d": jc(r3["Psi"]), "Psi_2d_after": jc(ra["Psi"]),
                     "after_vs_3d": rel(ra["Psi"], r3["Psi"]), "L_2d_vs_3d": ra["L_H"] / r3["L_H"] - 1,
                     "exterior_2d": jc(ra["Psi_bg_exterior"]), "exterior_3d": jc(r3["Psi_bg_exterior"]),
                     "exterior_over_Psi_2d": abs(bg_ext(ra)) / abs(ra["Psi"])}
            if r3["Psi_bg_exterior"] is not None and abs(r3["Psi_bg_exterior"]) > 0:
                entry["exterior_2d_vs_3d"] = rel(bg_ext(ra), r3["Psi_bg_exterior"])
            if b:
                rb = at_f(b, r3["f_Hz"])
                entry["Psi_2d_before"] = jc(rb["Psi"])
                entry["before_vs_3d"] = rel(rb["Psi"], r3["Psi"])
                entry["after_minus_before_minus_exterior"] = abs(ra["Psi"] - rb["Psi"] - bg_ext(ra)) / abs(ra["Psi"])
            table.append(entry)
    res["A_2d_vs_3d"] = table
    g, w = after.get("A_grid"), after.get("A_windings")
    if g and w:   # the same mesh: the grid's exterior term against the winding's static multipoles
        res["A_coil_grid_vs_2d_winding"] = [
            {"k_R": kr, "Psi": rel(at(g, kr)["Psi"], at(w, kr)["Psi"]),
             "exterior": rel(bg_ext(at(g, kr)), at(w, kr)["Psi_exterior"]),
             "L": at(g, kr)["L_H"] / at(w, kr)["L_H"] - 1} for kr in KR]
        gb = before.get("A_grid")
        if gb:
            res["A_coil_grid_vs_2d_winding"] = [dict(e, Psi_before=rel(at(gb, e["k_R"])["Psi"], at(w, e["k_R"])["Psi"]))
                                                for e in res["A_coil_grid_vs_2d_winding"]]
    lw = after.get("B_long_windings_R1.2")
    if lw:
        out = []
        for n2 in ("B_long", "B_long_R1x1.5"):
            x, xb = after.get(n2), before.get(n2)
            if not x:
                continue
            for kr in KR:
                e = {"model": n2, "k_R": kr, "Psi": rel(at(x, kr)["Psi"], at(lw, kr)["Psi"]),
                     "L": at(x, kr)["L_H"] / at(lw, kr)["L_H"] - 1}
                if xb:
                    e["Psi_before"] = rel(at(xb, kr)["Psi"], at(lw, kr)["Psi"])
                out.append(e)
        res["A_long_grid_vs_2d_long_winding_R1.2"] = out
    pg, pw, pb = after.get("P_plug_grid"), after.get("P_plug_windings"), after.get("P_plug_grid_R1x1.5")
    if pg and pw:   # the reaction field's term (A_lab alone) next to the grid's, against the winding (one A_lab)
        out = []
        for kr in KR:
            g1, w1 = at(pg, kr), at(pw, kr)
            e = {"k_R": kr, "Psi": rel(g1["Psi"], w1["Psi"]),
                 "reaction_exterior_over_Psi": abs(g1["Psi_exterior"]) / abs(g1["Psi"]),
                 "background_exterior_over_Psi": abs(bg_ext(g1)) / abs(g1["Psi"]),
                 "sum_of_exteriors_vs_winding_exterior": rel(g1["Psi_exterior"] + bg_ext(g1), w1["Psi_exterior"])}
            if pb:
                g2 = at(pb, kr)
                e["R1x1.5"] = rel(g2["Psi"], g1["Psi"])
                e["R1x1.5_without_exteriors"] = rel(g2["Psi"] - g2["Psi_exterior"] - bg_ext(g2),
                                                    g1["Psi"] - g1["Psi_exterior"] - bg_ext(g1))
            gb = before.get("P_plug_grid")
            if gb:
                e["Psi_before"] = rel(at(gb, kr)["Psi"], w1["Psi"])
            out.append(e)
        res["A_plug_grid_vs_2d_winding"] = out
    narrow = after.get("B_long_narrow_grid")
    if narrow:
        res["A_narrow_grid"] = {"exterior": (narrow["background"] or {}).get("exterior"),
                                "warned": any("inscribed" in s for s in narrow["warnings"])}

    # R: the domain radius
    for fam in ("coil", "long", "windings"):
        ref = after.get(f"R_{fam}_{R1_SCAN[-1]}")
        if not ref:
            continue
        rows = []
        for rb in R1_SCAN:
            x = after.get(f"R_{fam}_{rb}")
            if not x:
                continue
            for kr in KR:
                r, r0 = at(x, kr), at(ref, kr)
                tail = bg_ext(r) if fam != "windings" else r["Psi_exterior"]
                tail0 = bg_ext(r0) if fam != "windings" else r0["Psi_exterior"]
                rows.append({"R1": rb, "k_R": kr, "Psi": jc(r["Psi"]), "Psi_vs_largest_R1": rel(r["Psi"], r0["Psi"]),
                             "without_exterior_vs_largest_R1": rel(r["Psi"] - tail, r0["Psi"] - tail0),
                             "exterior_over_Psi": abs(tail) / abs(r["Psi"]),
                             "L_vs_largest_R1": r["L_H"] / r0["L_H"] - 1})
        res[f"R_{fam}"] = rows

    # M: the MQS limit
    for fam in ("long", "coil", "windings"):
        x = after.get(f"M_{fam}")
        if not x:
            continue
        rows = []
        for r in x["rows"]:
            kr = r["k_R"]
            t = bg_ext(r) if fam != "windings" else r["Psi_exterior"]
            rows.append({"k_R": kr, "exterior_over_Psi_MQS": abs(t) / abs(x["Psi_MQS"]),
                         "re_over_kR2_Psi_MQS": t.real / kr ** 2 / x["Psi_MQS"],
                         "im_over_kR3_Psi_MQS": t.imag / kr ** 3 / x["Psi_MQS"],
                         "Psi_over_MQS_minus_1": abs(r["Psi"]) / abs(x["Psi_MQS"]) - 1})
        for i in range(1, len(rows)):
            a, b = rows[i - 1], rows[i]
            a_t = a["re_over_kR2_Psi_MQS"] * a["k_R"] ** 2
            b_t = b["re_over_kR2_Psi_MQS"] * b["k_R"] ** 2
            b["re_order"] = math.log(abs(b_t / a_t)) / math.log(b["k_R"] / a["k_R"])
            a_i = a["im_over_kR3_Psi_MQS"] * a["k_R"] ** 3
            b_i = b["im_over_kR3_Psi_MQS"] * b["k_R"] ** 3
            if a_i != 0 and b_i != 0:
                b["im_order"] = math.log(abs(b_i / a_i)) / math.log(b["k_R"] / a["k_R"])
        res[f"M_{fam}"] = rows

    # N: the multipole order
    ref = after.get(f"N_long_{DTN_SCAN[-1]}")
    if ref:
        rows = []
        for n in DTN_SCAN:
            x = after.get(f"N_long_{n}")
            if not x:
                continue
            for kr in (0.4, 0.8):
                r, r0 = at(x, kr), at(ref, kr)
                rows.append({"dtn_modes": n, "k_R": kr, "Psi_vs_largest": rel(r["Psi"], r0["Psi"]),
                             "exterior_vs_largest": rel(bg_ext(r), bg_ext(r0)),
                             "L_vs_largest": r["L_H"] / r0["L_H"] - 1,
                             "last_order_fraction": r["bg_last_order_fraction"]})
        res["N_long"] = rows

    # Q: the radial panels
    ref = after.get(f"Q_long_{PANEL_SCAN[-1]}")
    if ref:
        rows = []
        for p in PANEL_SCAN:
            x = after.get(f"Q_long_{p}")
            if not x:
                continue
            ext = (x["background"] or {}).get("exterior", {})
            for kr in (0.02, 0.8):
                r, r0 = at(x, kr), at(ref, kr)
                rows.append({"panels": p, "k_R": kr, "exterior_vs_256": rel(bg_ext(r), bg_ext(r0)),
                             "Psi_vs_256": rel(r["Psi"], r0["Psi"]),
                             "quadrature_relative_change_doubled_panels":
                                 ext.get("quadrature_relative_change_doubled_panels")})
        res["Q_long"] = rows

    # C: grids that do not reach past the domain sphere
    ci = after.get("C_inside_R0.6")
    cf = after.get("C_full_R0.6")
    cw = after.get("C_windings_R0.6")
    cc = after.get("C_cut_R0.45")
    if ci and cf and cw:
        out = {"inside_R0.6": {"exterior_term_background": ci["exterior_term_background"],
                               "background": {k: v for k, v in (ci["background"] or {}).items() if k != "form"},
                               "warnings": ci["warnings"],
                               "before_warnings": (before.get("C_inside_R0.6") or {}).get("warnings")},
               "rows": []}
        for kr in KR:
            a, b, w = at(ci, kr), at(cf, kr), at(cw, kr)
            out["rows"].append({"k_R": kr, "Psi_cut_grid": jc(a["Psi"]), "Psi_full_grid": jc(b["Psi"]),
                                "cut_vs_full": rel(a["Psi"], b["Psi"]), "full_vs_winding": rel(b["Psi"], w["Psi"])})
        if cc:
            out["cut_R0.45"] = {"exterior_term_background": cc["exterior_term_background"],
                                "background": {k: v for k, v in (cc["background"] or {}).items() if k != "form"},
                                "warnings": cc["warnings"]}
        res["C_grid_coverage"] = out

    # U: uniform backgrounds; A_lab's term is included (the "before" build omitted it)
    urows = []
    for kind, radii, krs in (("plug", (R1, 1.5 * R1), KR), ("sccoax", (R1, 1.5 * R1), KR),
                             ("windings", (R1, 1.5 * R1), KR), ("core", (0.8, 1.2), KR_CORE)):
        x0 = after.get(f"U_{kind}_{radii[0]:g}")
        x1 = after.get(f"U_{kind}_{radii[1]:g}")
        xg = after.get(f"UG_{kind}_{radii[0]:g}")
        xb = before.get(f"U_{kind}_{radii[0]:g}")
        if not (x0 and x1):
            continue
        for row0 in x0["rows"]:
            f = row0["f_Hz"]
            a = row0
            b = at_f(x1, f)
            lab = a["Psi_exterior"] if a["Psi_exterior"] is not None else 0j
            e = {"design": kind, "R1": list(radii), "k_R_device": row0["k_R"], "f_Hz": f,
                 "R1_change": rel(b["Psi"], a["Psi"]), "A_lab_term_over_Psi": abs(lab) / abs(a["Psi"])}
            if xg:   # the same field as a 1 T grid: the same domain integral and reaction
                g = at_f(xg, f)
                lab_g = g["Psi_exterior"] if g["Psi_exterior"] is not None else 0j
                e["A_lab_term_vs_uniform_as_grid"] = rel(lab, lab_g) if abs(lab_g) > 0 else None
                e["grid_minus_its_background_term_vs_uniform"] = rel(g["Psi"] - bg_ext(g), a["Psi"])
            if xb:   # build1 omitted A_lab's term: the change must be exactly that term
                c = at_f(xb, f)
                e["change_vs_before"] = rel(a["Psi"], c["Psi"])
                e["after_minus_before_minus_term"] = abs(a["Psi"] - c["Psi"] - lab) / abs(a["Psi"])
                b_before = at_f(before[f"U_{kind}_{radii[1]:g}"], f) if before.get(f"U_{kind}_{radii[1]:g}") else None
                if b_before:
                    e["R1_change_before"] = rel(b_before["Psi"], c["Psi"])
            urows.append(e)
    res["U_uniform"] = urows

    # V: the disk design in a uniform 1 T, 3D against 2D before and after
    v3 = after.get("V_uniform_plug_3d")
    v2 = after.get(f"U_plug_{R1:g}")
    v2b = before.get(f"U_plug_{R1:g}")
    if v3 and v2:
        rows = []
        for r3 in v3["rows"]:
            a = at_f(v2, r3["f_Hz"])
            e = {"k_R": r3["k_R"], "Psi_3d": jc(r3["Psi"]), "Psi_2d": jc(a["Psi"]),
                 "after_vs_3d": rel(a["Psi"], r3["Psi"]), "L_3d_over_L_2d_minus_1": r3["L_H"] / a["L_H"] - 1}
            if v2b:
                e["before_vs_3d"] = rel(at_f(v2b, r3["f_Hz"])["Psi"], r3["Psi"])
            rows.append(e)
        res["V_uniform_plug_2d_vs_3d"] = {"rows": rows, "Psi_MQS_3d": v3["Psi_MQS"], "Psi_MQS_2d": v2["Psi_MQS"],
                                          "dofs_3d": v3["dofs"]}

    # I and F: invariance (every leaf of results.json, before against after). I_uniform_plug changes by design
    # (A_lab's term, the disk's reaction): checked as exactly that term
    inv = {}
    for n in names:
        if not (n.startswith("I_") or n.startswith("F_")):
            continue
        pa = RUNS / "before" / n / "results.json"
        pb = RUNS / "after" / n / "results.json"
        if pa.exists() and pb.exists():
            inv[n] = compare_results(pa, pb)
    res["I_invariance"] = inv
    xa, xb = after.get("I_uniform_plug"), before.get("I_uniform_plug")
    if xa and xb:
        res["I_uniform_plug_change"] = [
            {"k_R": ra["k_R"], "change": rel(ra["Psi"], rb["Psi"]),
             "A_lab_term_over_Psi": abs(ra["Psi_exterior"] or 0j) / abs(ra["Psi"]),
             "after_minus_before_minus_term": abs(ra["Psi"] - rb["Psi"] - (ra["Psi_exterior"] or 0j)) / abs(ra["Psi"]),
             "L_unchanged": ra["L_H"] == rb["L_H"]}
            for ra, rb in zip(xa["rows"], xb["rows"])]

    rec["results"] = res
    print(json.dumps(res, indent=1, default=float), flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    if RUNS == (ROOT / "runs" / "axi_background_exterior").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_BG2D_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        tag = a[a.index("--tag") + 1] if "--tag" in a else "after"
        parts = a[a.index("--parts") + 1].split(",") if "--parts" in a else ["A", "R", "M", "N", "Q", "C", "U", "I"]
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
        (RUNS / tag).mkdir(parents=True, exist_ok=True)
        run(tag, parts, only, ranks)
        return
    report()


if __name__ == "__main__":
    main()
