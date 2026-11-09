"""An imposed background in 3D full-wave sweeps (cpp/src/model3d_fullwave.cpp: BackgroundA; the exterior term:
axi::BackgroundExterior in cpp/src/axi_fullwave.cpp, shared with the axisymmetric engine).

    python benchmarks/fullwave3d_background.py run [--ranks N] [--parts A,B,C,D,E] [--only NAME,...]
    python benchmarks/fullwave3d_background.py report

A 3D sweep couples an imposed `background` (axisymmetric about z: uniform, or an (r, z) grid) in the reciprocal form
Psi_bg(omega) = (1/(i omega I)) int curl E . A_bg dV with A_bg = (psi / r) phi_hat: over the domain sphere from the
finite-element field, and, with a radiating boundary, outside it from the TM m = 0 exterior multipoles of the readout
field: the grid up to the sphere inscribed in its box, its exterior multipoles beyond
(docs/physics/fullwave_background.md; docs/validation/fullwave3d_background.md). The model is a small copy of the
`solenoid` template (the DMRadio-m^3 topology: a shorted coax inside a solenoid, read out across its bottom gap),
revolved to 3D with the annular port; R1 = 0.45 m unless stated.

A. The solenoid's field two ways on one mesh: as the design's winding (B_lab = curl A_lab, the validated
   E . B_lab path with its multipole exterior term) and as a grid background sampled from the analytic field of the
   same coil (the coil region kept, as vacuum), at kR = 0.02, 0.4 and 0.8. Also: the grid case's MQS limit against
   its own 3D MQS Psi (both reciprocal forms); a grid of half the extent (continued from 1 m instead of
   2 m); the grid case at R1 = 0.675 m (R1-independence, which needs the exterior term).
B. A long magnet (1.8 m, four times R1) as the background around the same pickup, as DMRadio-Core's 2.95 m magnet
   would be: its field crosses R1, so only the exterior term makes Psi independent of R1. R1 = 0.45 and 0.675 m;
   and a grid whose inscribed sphere cuts the magnet, which the continuation check must flag.
C. MPI: the grid case of A at kR = 0.8 on two ranks (A runs on the eight of the job; one rank ran out of 96 GB).
D. A uniform background (1 T): with a closed sphere (boundary "pec") its MQS limit; with the radiating boundary the
   omitted exterior term and its warning.
E. The axisymmetric engine on the same designs (its uniform annular feed is the 3D models' 360-degree port): the
   winding (with its exterior term) against A's winding model, and the grid and uniform backgrounds, which it couples
   in the same reciprocal form inside its domain (no exterior term in 2D), against the 3D volume part and D.

Writes benchmarks/records/fullwave3d_background.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

RUNS = Path(os.environ.get("AXF_BG_RUNS", ROOT / "runs" / "fullwave3d_background")).resolve()
OUT = ROOT / "benchmarks" / "records" / "fullwave3d_background.json"

MU0 = 4e-7 * math.pi
C = 299792458.0
# the solenoid template, small: coil [R, R + TW] x [-H/2, H/2]; coax walls at X R and R - G (thickness TC); port TP
R, H, X, TW, G, TC, TP = 0.2, 0.3, 0.5, 0.04, 0.02, 0.02, 0.02
NI = 1.0e6
R_DEV = R + TW          # coupling.device_radius_m: the outer radius of all regions
R1 = 0.45
MESH = {"order": 2, "size": 0.06, "surface_size": 0.02}
KR = (0.02, 0.4, 0.8)
LONG_HALF = 0.9         # half-length of part B's magnet (same radii and current density as the coil)


def f_of(kr: float) -> float:
    return kr * C / (2 * math.pi * R_DEV)


# ---------------------------------------------------------------- the coil's field (analytic)
def _cel(kc, p, c, s):
    """Bulirsch's cel(kc, p, c, s) = int_0^{pi/2} (c cos^2 t + s sin^2 t) / ((cos^2 t + p sin^2 t)
    sqrt(cos^2 t + kc^2 sin^2 t)) dt (p > 0), as c R_F(0, kc^2, 1) + (s - p c) R_J(0, kc^2, 1, p) / 3."""
    from scipy.special import elliprf, elliprj
    k2 = kc * kc
    return c * elliprf(0.0, k2, 1.0) + (s - p * c) * elliprj(0.0, k2, 1.0, p) / 3.0


def sheet_field(r, z, a, b, K):
    """Br, Bz of an ideal solenoid sheet of radius a over -b <= z <= b with surface current K (A/m), right-handed
    about +z (Derby and Olbert, Am. J. Phys. 78, 229 (2010), Eqs. 14-15)."""
    B0 = MU0 * K / math.pi
    gamma = (a - r) / (a + r)
    br = 0.0
    bz = 0.0
    for zs, sign in ((z + b, 1.0), (z - b, -1.0)):
        den = np.sqrt(zs * zs + (r + a) ** 2)
        kc = np.sqrt(zs * zs + (a - r) ** 2) / den
        br = br + sign * (a / den) * _cel(kc, 1.0, 1.0, -1.0)
        bz = bz + sign * (zs / den) * _cel(kc, gamma * gamma, 1.0, gamma)
    return B0 * br, B0 * a / (a + r) * bz


def coil_field(r, z, a1=R, a2=R + TW, z1=-H / 2, z2=H / 2, ampere_turns=NI, nodes=24):
    """Br, Bz of a coil with uniform azimuthal current density over [a1, a2] x [z1, z2] (total ampere_turns): the
    sheets integrated over the radius by Gauss-Legendre, split at a = r inside the winding (where Bz kinks)."""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    r, z = np.broadcast_arrays(r, z)
    J = ampere_turns / ((a2 - a1) * (z2 - z1))
    b = 0.5 * (z2 - z1)
    zc = 0.5 * (z1 + z2)
    x, w = np.polynomial.legendre.leggauss(nodes)
    Br = np.zeros(r.shape)
    Bz = np.zeros(r.shape)
    split = np.clip(r, a1, a2)
    for lo, hi in ((np.full(r.shape, a1), split), (split, np.full(r.shape, a2))):
        length = hi - lo
        live = length > 1e-9 * a2   # a node on the winding's edge (up to round-off) leaves one piece empty
        for xi, wi in zip(x, w):
            a = np.where(live, lo + 0.5 * (xi + 1.0) * length, a2 + 1.0)   # a dummy radius where the piece is empty
            br, bz = sheet_field(r, z - zc, a, b, J)
            Br += np.where(live, 0.5 * wi * length * br, 0.0)
            Bz += np.where(live, 0.5 * wi * length * bz, 0.0)
    return Br, Bz


def loop_field(r, z, a, z0, current):
    """Br, Bz of one circular loop (radius a at height z0), for the self-check."""
    from scipy.special import ellipe, ellipk
    d = z - z0
    m = 4 * a * r / ((a + r) ** 2 + d * d)
    K = ellipk(m)
    E = ellipe(m)
    q = (a + r) ** 2 + d * d
    p = (a - r) ** 2 + d * d
    bz = MU0 * current / (2 * math.pi * np.sqrt(q)) * (K + (a * a - r * r - d * d) / p * E)
    br = MU0 * current * d / (2 * math.pi * r * np.sqrt(q)) * (-K + (a * a + r * r + d * d) / p * E)
    return br, bz


def field_self_check() -> dict:
    """coil_field against the on-axis formula, against loop superposition away from the winding, and its divergence."""
    zs = np.linspace(-1.0, 1.0, 41)
    _, bz = coil_field(np.zeros_like(zs), zs)
    J = NI / (TW * H)

    def on_axis(zz):  # int over a of mu0 J / 2 [(z + b) / sqrt((z + b)^2 + a^2) - (z - b) / ...]
        x, w = np.polynomial.legendre.leggauss(40)
        out = 0.0
        for xi, wi in zip(x, w):
            a = R + 0.5 * (xi + 1) * TW
            out += 0.5 * TW * wi * MU0 * J / 2 * ((zz + H / 2) / np.hypot(zz + H / 2, a) - (zz - H / 2) / np.hypot(zz - H / 2, a))
        return out
    axis_err = float(np.max(np.abs(bz - on_axis(zs))) / np.max(np.abs(on_axis(zs))))
    pts = [(0.1, 0.05), (0.3, 0.0), (0.22, 0.4), (0.5, -0.3), (1.2, 0.7)]
    x, w = np.polynomial.legendre.leggauss(40)
    loop_err = 0.0
    for rr, zz in pts:
        br, bz1 = coil_field(rr, zz)
        sr = 0.0
        sz = 0.0
        for xa, wa in zip(x, w):
            for xz, wz in zip(x, w):
                a = R + 0.5 * (xa + 1) * TW
                z0 = 0.5 * xz * H
                lr, lz = loop_field(rr, zz, a, z0, J * 0.25 * wa * wz * TW * H)
                sr += lr
                sz += lz
        loop_err = max(loop_err, float(np.hypot(br - sr, bz1 - sz) / np.hypot(sr, sz)))
    h = 1e-5
    div = 0.0
    for rr, zz in pts + [(0.21, 0.0), (0.23, 0.1)]:   # the last two inside the winding
        brp, _ = coil_field(rr + h, zz)
        brm, _ = coil_field(rr - h, zz)
        br0, _ = coil_field(rr, zz)
        _, bzp = coil_field(rr, zz + h)
        _, bzm = coil_field(rr, zz - h)
        d = (brp - brm) / (2 * h) + br0 / rr + (bzp - bzm) / (2 * h)
        div = max(div, float(abs(d) / (abs(J) * MU0)))
    return {"on_axis_relative_error": axis_err, "loop_superposition_relative_error": loop_err,
            "divergence_over_mu0_J": div}


def grid(extent_r: float, extent_z: float, z1=-H / 2, z2=H / 2, fine_z: float = 0.3, h_fine: float = 0.002,
         h: float = 0.01) -> dict:
    from axionfem.background import graded_nodes, grid_block
    rn = graded_nodes(0.0, extent_r, h, fine=[(0.0, 0.4, h_fine)])
    zn = graded_nodes(-extent_z, extent_z, h, fine=[(-fine_z, fine_z, h_fine)])
    turns = NI * (z2 - z1) / H   # the coil's current density over the given length
    return grid_block(lambda rr, zz: coil_field(rr, zz, z1=z1, z2=z2, ampere_turns=turns), rn, zn,
                      source=f"analytic coil [{R}, {R + TW}] x [{z1}, {z2}] m, {turns:g} ampere-turns",
                      notes="benchmarks/fullwave3d_background.py")


# ---------------------------------------------------------------- designs
def design(winding: bool, background: dict | None, domain_radius: float, kr: tuple, boundary: str = "radiation",
           axisymmetric: bool = False, mesh: dict | None = None, plug: bool = False):
    """The model; plug adds a superconducting disk (r <= 0.15 m, 0.19 <= z <= 0.22 m) above the coax, which screens
    the coil's field: a reaction field (A_lab != 0) next to the background, and a perfect conductor at RF."""
    import axionfem as af
    from axionfem.convert import to_3d
    d = af.template("solenoid", R=R, H=H, x=X, tw=TW, g=G, tc=TC, tp=TP, NI=NI)
    if plug:
        d.doc["regions"].append({"name": "plug", "shape": {"rect": {"r": [0.0, 0.15], "z": [0.19, 0.22]}},
                                 "material": "superconductor"})
    if not winding:
        # the coil region is kept, as a linear material with mu_r = 1 (vacuum, statically and at RF): the mesher
        # refines non-vacuum regions as it refines windings, so every model meshes the same geometry the same way
        d.doc["windings"] = []
        d.doc.setdefault("materials", {})["coil_space"] = {"kind": "linear", "mu_r": 1.0}
        for region in d.doc["regions"]:
            if region["name"] == "winding":
                region["material"] = "coil_space"
    d.doc.pop("benchmarks", None)
    if axisymmetric:   # the axisymmetric engine's mesh (its own defaults, finer than the 3D one)
        d.doc.setdefault("mesh", {})["domain_radius"] = domain_radius
        doc = d.doc
    else:
        doc = to_3d(d, mesh={**(mesh or MESH), "domain_radius": domain_radius}).doc
    if background is not None:
        doc["background"] = background
    doc["fullwave"] = {"frequencies": [f_of(k) for k in kr], "boundary": boundary}
    return doc


def models() -> dict:
    """name -> (part, builder, ranks or None for the job's ranks)"""
    full = lambda: grid(2.0, 2.0)                                      # noqa: E731
    half = lambda: grid(1.0, 1.0)                                      # noqa: E731
    long = lambda: grid(2.0, 3.0, z1=-LONG_HALF, z2=LONG_HALF, fine_z=1.1)  # noqa: E731
    uniform = lambda: {"type": "uniform", "Bz": 1.0}                  # noqa: E731
    return {
        "A_grid": ("A", lambda: design(False, full(), R1, KR), None),
        "A_windings": ("A", lambda: design(True, None, R1, KR), None),
        "A_grid_half": ("A", lambda: design(False, half(), R1, KR), None),
        "A_grid_R1x1.5": ("A", lambda: design(False, full(), 1.5 * R1, (0.4, 0.8)), None),
        "B_long": ("B", lambda: design(False, long(), R1, KR), None),
        "B_long_R1x1.5": ("B", lambda: design(False, long(), 1.5 * R1, KR), None),
        # a grid whose inscribed sphere (0.8 m) cuts the magnet (to 0.93 m): the continuation check must say so
        "B_long_narrow_grid": ("B", lambda: design(False, grid(0.8, 3.0, z1=-LONG_HALF, z2=LONG_HALF, fine_z=1.1),
                                                   R1, (0.8,)), None),
        "C_grid_2ranks": ("C", lambda: design(False, full(), R1, (0.8,)), 2),
        "D_uniform_pec": ("D", lambda: design(False, uniform(), R1, (0.02,), boundary="pec"), None),
        "D_uniform_radiation": ("D", lambda: design(False, uniform(), R1, (0.02,)), None),
        # the axisymmetric engine (uniform annular feed, as the 3D models' 360-degree port): the winding with its
        # exterior term, and the grid and uniform backgrounds in the same reciprocal form (exterior omitted in 2D)
        "E_axi_windings": ("E", lambda: design(True, None, R1, KR, axisymmetric=True), 1),
        "E_axi_grid": ("E", lambda: design(False, full(), R1, KR, axisymmetric=True), 1),
        "E_axi_uniform_pec": ("E", lambda: design(False, uniform(), R1, (0.02,), boundary="pec", axisymmetric=True), 1),
    }


def solve(name: str, builder, ranks: int) -> None:
    import axionfem as af
    run_dir = RUNS / name
    t0 = time.time()
    try:
        af.Design(builder()).solve(run_dir, verbose=1, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (recorded, not fatal)
        print(f"{name} failed: {type(exc).__name__}: {exc}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name,
                "ranks": ranks, "seconds": time.time() - t0}
    (run_dir / "env.json").write_text(json.dumps({"produced_by": produced}) + "\n")


def run(ranks: int, parts: list[str], only: list[str] | None = None) -> None:
    for name, (part, builder, own_ranks) in models().items():
        if part not in parts or (only and name not in only):
            continue
        if (RUNS / name / "results.json").exists() or (RUNS / name / "error.txt").exists():
            print(f"{name}: kept", flush=True)
            continue
        solve(name, builder, own_ranks or ranks)


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


def provenance() -> dict:
    import axionfem as af
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": file_record(exe), "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")),
            "python_core": file_record(af._core.__file__), "driver": file_record(__file__),
            "date": time.strftime("%Y-%m-%d")}


def cx(d: dict | None) -> complex | None:
    return complex(d["re"], d["im"]) if d else None


def mqs_forms(res: dict, checks: dict) -> tuple[float, float | None]:
    """The 3D MQS Psi in its primary form (int A_pick . B_lab) and its reciprocal form, both oriented. A current
    run with a background reports the reciprocal form in coupling and keeps the primary one in
    checks.Psi_primary_form; older results report the primary form in coupling and the reciprocal one (unoriented) in
    checks.Psi_A_lab_dot_B_pick."""
    reported = res["coupling"]["Psi_SI_H_T_m2"]
    if checks.get("Psi_reported_form") == "reciprocal":
        return checks["Psi_primary_form"]["value_SI_H_T_m2"], reported
    return reported, checks.get("Psi_A_lab_dot_B_pick")


def load(name: str) -> dict | None:
    path = RUNS / name / "results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    fw = res.get("fullwave", {})
    checks = res.get("checks") if isinstance(res.get("checks"), dict) else {}   # a list in axisymmetric results
    rows = []
    for row in fw.get("sweep", []):
        bg = row.get("Psi_background") or {}
        solver = row.get("solver") or {}
        rows.append({"f_Hz": row["f_Hz"], "k_R": row["k_R"], "Psi": complex(row["Psi_re"], row["Psi_im"]),
                     "Psi_exterior": cx(row.get("Psi_exterior")), "Psi_bg_volume": cx(bg.get("volume")),
                     "Psi_bg_exterior": cx(bg.get("exterior")),
                     "bg_last_order_fraction": bg.get("exterior_last_order_fraction"),
                     "Z": complex(row["Z_re_ohm"], row["Z_im_ohm"]), "L_H": row["L_H"],
                     "residual": solver.get("relative_residual"), "gmres": solver.get("gmres_iterations")})
    env = json.loads((RUNS / name / "env.json").read_text()) if (RUNS / name / "env.json").exists() else {}
    psi_primary, psi_reciprocal = mqs_forms(res, checks)
    return {"rows": rows, "Psi_MQS": psi_primary, "L_MQS": res["coupling"]["L_p_H"],
            "Psi_MQS_reciprocal": psi_reciprocal,
            "dofs": res.get("mesh", {}).get("dofs"), "background": fw.get("background"),
            "exterior_term": fw.get("exterior_term"), "exterior_term_background": fw.get("exterior_term_background"),
            "warnings": res.get("warnings", []), "produced_by": env.get("produced_by", {})}


def jc(z: complex | None):
    return None if z is None else [z.real, z.imag]


def summary(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k != "rows"}
    out["rows"] = [{k: (jc(v) if isinstance(v, complex) else v) for k, v in r.items()} for r in m["rows"]]
    return out


def at(m: dict, kr: float) -> dict:
    return min(m["rows"], key=lambda r: abs(r["k_R"] - kr))


def rel(a: complex, b: complex) -> float:
    return abs(a - b) / abs(b)


def report() -> None:
    names = list(models())
    m = {n: load(n) for n in names}
    rec = {"check": "fullwave3d_background", "provenance": provenance(), "R1_m": R1, "mesh": MESH,
           "design": {"R": R, "H": H, "x": X, "tw": TW, "g": G, "tc": TC, "tp": TP, "NI": NI, "R_dev": R_DEV},
           "coil_field_self_check": field_self_check(), "models": {n: summary(v) for n, v in m.items() if v}}
    print("coil field self-check:", rec["coil_field_self_check"], flush=True)
    res = {}
    w, g = m["A_windings"], m["A_grid"]
    if w and g:
        res["A_windings_vs_grid"] = []
        for kr in KR:
            a, b = at(w, kr), at(g, kr)
            res["A_windings_vs_grid"].append({"k_R": b["k_R"], "Psi_grid_vs_windings": rel(b["Psi"], a["Psi"]),
                                              "Z_grid_vs_windings": rel(b["Z"], a["Z"]),
                                              "Psi_windings": jc(a["Psi"]), "Psi_grid": jc(b["Psi"]),
                                              "bg_exterior_over_Psi": abs(b["Psi_bg_exterior"]) / abs(b["Psi"])})
        lo = at(g, 0.02)
        res["A_grid_mqs_limit"] = {"Psi_FW_over_MQS_reciprocal": abs(lo["Psi"]) / abs(g["Psi_MQS_reciprocal"]),
                                   "Psi_FW_over_MQS": abs(lo["Psi"]) / abs(g["Psi_MQS"]),
                                   "MQS_reciprocal_over_MQS": g["Psi_MQS_reciprocal"] / g["Psi_MQS"],
                                   "windings_Psi_FW_over_MQS": abs(at(w, 0.02)["Psi"]) / abs(w["Psi_MQS"])}
    h = m["A_grid_half"]
    if g and h:   # the grid's extent (continuation from 1 m instead of 2 m)
        res["A_grid_extent"] = [{"k_R": kr, "Psi": rel(at(h, kr)["Psi"], at(g, kr)["Psi"]),
                                 "bg_exterior": rel(at(h, kr)["Psi_bg_exterior"], at(g, kr)["Psi_bg_exterior"])}
                                for kr in KR]
    ew = m["E_axi_windings"]
    if g and ew:   # the same exterior integral, int A_coil . B_I outside R1, from the axisymmetric engine's multipoles
        res["A_grid_exterior_vs_axisymmetric_windings_exterior"] = [
            {"k_R": kr, "bg_exterior_3d": jc(at(g, kr)["Psi_bg_exterior"]), "exterior_2d": jc(at(ew, kr)["Psi_exterior"]),
             "relative": rel(at(g, kr)["Psi_bg_exterior"], at(ew, kr)["Psi_exterior"])} for kr in KR]
    narrow = m["B_long_narrow_grid"]
    if narrow:
        res["B_long_narrow_grid"] = {"exterior": (narrow["background"] or {}).get("exterior"),
                                     "warned": any("inscribed" in s for s in narrow["warnings"])}
    big = m["A_grid_R1x1.5"]
    if g and big:
        res["A_grid_R1_independence"] = [
            {"k_R": kr, "Psi": rel(at(big, kr)["Psi"], at(g, kr)["Psi"]),
             "Psi_without_bg_exterior": rel(at(big, kr)["Psi"] - at(big, kr)["Psi_bg_exterior"],
                                            at(g, kr)["Psi"] - at(g, kr)["Psi_bg_exterior"])} for kr in (0.4, 0.8)]
    b1, b2 = m["B_long"], m["B_long_R1x1.5"]
    if b1 and b2:
        res["B_long_R1_independence"] = [
            {"k_R": kr, "Psi": rel(at(b2, kr)["Psi"], at(b1, kr)["Psi"]),
             "Psi_without_bg_exterior": rel(at(b2, kr)["Psi"] - at(b2, kr)["Psi_bg_exterior"],
                                            at(b1, kr)["Psi"] - at(b1, kr)["Psi_bg_exterior"]),
             "bg_exterior_over_Psi": [abs(at(x, kr)["Psi_bg_exterior"]) / abs(at(x, kr)["Psi"]) for x in (b1, b2)]}
            for kr in KR]
        res["B_long_mqs_limit"] = {
            name: {"Psi_FW_over_MQS_reciprocal": abs(at(x, 0.02)["Psi"]) / abs(x["Psi_MQS_reciprocal"]),
                   "Psi_FW_over_MQS": abs(at(x, 0.02)["Psi"]) / abs(x["Psi_MQS"]),
                   "MQS_reciprocal_over_MQS": x["Psi_MQS_reciprocal"] / x["Psi_MQS"]}
            for name, x in (("R1", b1), ("R1x1.5", b2))}
    one = m["C_grid_2ranks"]
    if g and one:
        res["C_ranks"] = {"Psi_2ranks_vs_job_ranks": rel(at(one, 0.8)["Psi"], at(g, 0.8)["Psi"]),
                          "Z_2ranks_vs_job_ranks": rel(at(one, 0.8)["Z"], at(g, 0.8)["Z"]),
                          "ranks": [one["produced_by"].get("ranks"), g["produced_by"].get("ranks")]}
    for name in ("D_uniform_pec", "D_uniform_radiation"):
        x = m[name]
        if x:
            lo = x["rows"][0]
            res[name] = {"Psi_FW_over_MQS_reciprocal": abs(lo["Psi"]) / abs(x["Psi_MQS_reciprocal"]),
                         "Psi_FW_over_MQS": abs(lo["Psi"]) / abs(x["Psi_MQS"]),
                         "exterior_term_background": x["exterior_term_background"],
                         "warned": any("uniform" in s and "exterior" in s for s in x["warnings"])}
    # the axisymmetric engine: windings against windings (totals), and its background coupling (interior only, the
    # same reciprocal form) against the 3D volume part
    ew, eg, eu = m["E_axi_windings"], m["E_axi_grid"], m["E_axi_uniform_pec"]
    if ew and w:
        res["E_windings_3d_vs_axisymmetric"] = [{"k_R": kr, "Psi": rel(at(w, kr)["Psi"], at(ew, kr)["Psi"]),
                                                 "L": at(w, kr)["L_H"] / at(ew, kr)["L_H"] - 1} for kr in KR]
    if eg and g:
        res["E_grid_3d_volume_vs_axisymmetric"] = [{"k_R": kr, "Psi": rel(at(g, kr)["Psi_bg_volume"], at(eg, kr)["Psi"])}
                                                   for kr in KR]
    if eu and m["D_uniform_pec"]:
        x = m["D_uniform_pec"]
        res["E_uniform_pec_3d_vs_axisymmetric"] = {"Psi_FW": rel(x["rows"][0]["Psi"], eu["rows"][0]["Psi"]),
                                                   "Psi_MQS_axisymmetric": eu["Psi_MQS"],
                                                   "Psi_MQS_3d": x["Psi_MQS"],
                                                   "Psi_MQS_3d_reciprocal": x["Psi_MQS_reciprocal"]}
    rec["results"] = res
    print(json.dumps(res, indent=1, default=float), flush=True)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    if RUNS == (ROOT / "runs" / "fullwave3d_background").resolve():
        OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", OUT, flush=True)
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_BG_RUNS is set)", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    if a[0] == "run":
        RUNS.mkdir(parents=True, exist_ok=True)
        ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
        parts = a[a.index("--parts") + 1].split(",") if "--parts" in a else ["A", "B", "C", "D", "E"]
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        run(ranks, parts, only)
    report()


if __name__ == "__main__":
    main()
