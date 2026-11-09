"""Shared helpers for the cavity reproductions. beta.

Every benchmark builds axisymmetric designs from its reference JSON (benchmarks/cavity/refs/<name>.json), solves
them with the existing `fullwave` + `cavity` sections (2D, m = 0), and compares with the published values.

Geometry convention used here: lengths in metres, the cavity is a closed body of revolution described by its
meridional outline (r >= 0, z), centred on z = 0. The conducting wall is one `conductor` region (a rectangle
that encloses the cavity outline with margin `t`) painted first; the cavity interior (vacuum) is painted on top,
then the dielectric parts (later regions win). Everything is inside the closed PEC sphere (`boundary: pec`);
the room between the wall and the sphere is excluded by the cavity section.

Run artefacts: every solve writes its results.json under $AXF_REPRO_WORK (default
runs/cavity_repro/<benchmark>/<tag> in the repository), and solve() reuses a finished
results.json whose design hash matches (so plotting/re-analysis does not recompute).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import time
from pathlib import Path

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6
Z0 = MU0 * C_LIGHT
ROOT = Path(__file__).resolve().parents[3]
REFS = ROOT / "benchmarks" / "cavity" / "refs"
RECS = ROOT / "benchmarks" / "records"
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AXF_REPRO_WORK", ROOT / "runs" / "cavity_repro"))
CHI01 = 2.404825557695773
CHI02 = 5.520078110286311
CHI03 = 8.653727912911013


def ref(name: str) -> dict:
    return json.loads((REFS / f"{name}.json").read_text())


def skin(sigma: float, T_K: float = 4.0) -> dict:
    """`conductor` with the classical skin-effect rf model for conductivity sigma [S/m]."""
    return {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.0 / sigma, "T_K": T_K}}


def const_Rs(Rs: float, T_K: float = 4.0) -> dict:
    """`conductor` with a frequency-independent R_s (rf model `power`, exponent 0)."""
    return {"kind": "conductor", "rf": {"model": "power", "Rs_ohm": Rs, "f_ref": 1e10, "exponent": 0.0, "T_K": T_K}}


def Rs_skin(f: float, sigma: float) -> float:
    return math.sqrt(math.pi * f * MU0 / sigma)


def design(name: str, outline: list, *, h: float, order: int = 2, wall: dict | None = None,
           dielectrics: list | None = None, conductors: list | None = None, t: float | None = None,
           modes: int = 6, families=("TM",), target_f_Hz: float | None = None, interior_point=None,
           surface_size: float | None = None, refine: list | None = None, Bz: float = 1.0):
    """Closed cavity whose interior is the polygon `outline` ([[r, z], ...], counter-clockwise, on/after r = 0).

    dielectrics: [{"name", "shape" (design SHAPE), "eps_r", "tan_delta", "mesh_size"?}] painted over the interior.
    conductors: [{"name", "shape"}] extra conducting parts inside the cavity (rods), same wall material.
    refine: extra vacuum regions [{"name", "shape", "mesh_size"}] painted over the interior (local refinement).
    """
    from axionfem.design import Design

    rs = [p[0] for p in outline]
    zs = [p[1] for p in outline]
    R, z0, z1 = max(rs), min(zs), max(zs)
    t = t or 0.1 * R
    mats = {"wall_cu": wall or {"kind": "conductor"}}
    regions = [{"name": "wall", "shape": {"rect": {"r": [0.0, R + t], "z": [z0 - t, z1 + t]}}, "material": "wall_cu",
                "mesh_size": max(h, t / 2)},
               {"name": "cavity", "shape": {"polygon": {"points": [list(p) for p in outline]}}, "material": "vacuum",
                "mesh_size": h}]
    for rg in refine or []:
        regions.append({"name": rg["name"], "shape": rg["shape"], "material": "vacuum", "mesh_size": rg["mesh_size"]})
    for dl in dielectrics or []:
        mname = f"mat_{dl['name']}"
        mats[mname] = {"kind": "linear", "eps_r": dl["eps_r"]}
        if dl.get("tan_delta"):
            mats[mname]["tan_delta"] = dl["tan_delta"]
        regions.append({"name": dl["name"], "shape": dl["shape"], "material": mname,
                        "mesh_size": dl.get("mesh_size", h)})
    for cd in conductors or []:
        regions.append({"name": cd["name"], "shape": cd["shape"], "material": "wall_cu",
                        "mesh_size": cd.get("mesh_size", h)})
    fw = {"modes": modes, "families": list(families), "boundary": "pec"}
    if target_f_Hz:
        fw["target_f_Hz"] = target_f_Hz
    # the mesher caps every element at min(Rd/6, pi Rd/96): keep that cap above h
    Rd = max(1.3 * math.hypot(R + t, max(abs(z0), abs(z1)) + t), 32 * h)
    ip = interior_point or [0.5 * R, 0.5 * (z0 + z1)]
    doc = {"schema": "axionfem.design/1", "name": name, "materials": mats, "regions": regions,
           "background": {"type": "uniform", "Bz": Bz}, "fullwave": fw,
           "cavity": {"interior_point": list(ip)},
           "mesh": {"order": order, "size": max(h, Rd / 20), "surface_size": surface_size or h, "domain_radius": Rd}}
    return Design(doc)


def clip_z(outline: list, zmin: float, zmax: float) -> list:
    """Part of the polygon `outline` with zmin <= z <= zmax (Sutherland-Hodgman against the two half-planes), for
    refinement zones that must not paint vacuum over the wall."""
    def clip(pts, inside, cut):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(cut(a, b))
                out.append(b)
            elif ia:
                out.append(cut(a, b))
        return out

    def cutz(z0):
        def f(a, b):
            s = (z0 - a[1]) / (b[1] - a[1])
            return [a[0] + s * (b[0] - a[0]), z0]
        return f
    pts = clip([list(p) for p in outline], lambda p: p[1] >= zmin, cutz(zmin))
    pts = clip(pts, lambda p: p[1] <= zmax, cutz(zmax))
    res = []
    for q in pts:
        if not res or abs(res[-1][0] - q[0]) + abs(res[-1][1] - q[1]) > 1e-12:
            res.append(q)
    if len(res) > 1 and abs(res[0][0] - res[-1][0]) + abs(res[0][1] - res[-1][1]) < 1e-12:
        res.pop()
    return res


def _key(doc: dict) -> str:
    return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def _solve_one(args):
    doc, wd = args
    from axionfem.design import Design

    wd = Path(wd)
    stamp = wd / "design_key.txt"
    res = wd / "results.json"
    key = _key(doc)
    if res.exists() and stamp.exists() and stamp.read_text().strip() == key:
        data = json.loads(res.read_text())
        return data, json.loads((wd / "wall_seconds.txt").read_text()) if (wd / "wall_seconds.txt").exists() else None
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = Design(doc).solve(wd, verbose=0)
    dt = time.time() - t0
    data = r.data
    if not res.exists():
        res.write_text(json.dumps(data))
    stamp.write_text(key)
    (wd / "wall_seconds.txt").write_text(json.dumps(dt))
    return data, dt


def solve_many(jobs: list, bench: str, workers: int | None = None) -> list:
    """jobs: [(tag, Design)]; returns [(tag, results dict, seconds)] in input order (parallel processes)."""
    workers = workers or int(os.environ.get("AXF_REPRO_WORKERS", "1"))
    args = [(d.doc, str(WORK / bench / tag)) for tag, d in jobs]
    if workers <= 1 or len(args) == 1:
        outs = [_solve_one(a) for a in args]
    else:
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(args))) as ex:
            outs = list(ex.map(_solve_one, args))
    return [(tag, data, dt) for (tag, _), (data, dt) in zip(jobs, outs)]


def modes(data: dict, fam: str = "TM") -> list:
    return data["fullwave"]["eigenmodes"][fam]["modes"]


def mode_row(data: dict, m: dict, V_full: float | None = None, diel_regions=()) -> dict:
    """Flat summary of one eigenmode: f, C (eps-weighted, cavity volume = conductors excluded), C_full (V_full),
    losses (G, Q_wall, Q_dielectric, Q0), electric filling of the given dielectric regions."""
    cav = data["fullwave"]["cavity"]
    ff, lo = m.get("form_factor", {}), m.get("losses", {})
    V = cav["volume_m3"]
    row = {"f_Hz": m["f_Hz"], "C": ff.get("C"), "V_m3": V, "residual_true": m.get("residual_true")}
    if V_full and ff.get("C") is not None:
        row["C_full"] = ff["C"] * V / V_full
        row["V_full_m3"] = V_full
    if lo:
        row.update({"G_ohm": lo.get("G_ohm"), "Q_wall": lo.get("Q_wall"), "Q_dielectric": lo.get("Q_dielectric"),
                    "Q0": lo.get("Q0")})
        pe = 0.0
        for rg in diel_regions:
            d = lo.get("dielectric", {}).get(rg)
            if d:
                pe += d["electric_energy_fraction"]
        if diel_regions:
            row["p_e"] = pe
    if "energy_fraction" in m and diel_regions:
        row["energy_fraction"] = {k: v for k, v in m["energy_fraction"].items()}
    return row


def dofs(data: dict) -> int:
    return data["mesh"]["dofs"]


def richardson(hs: list, ys: list, p: float) -> float | None:
    """Extrapolate y(h) = y0 + a h^p from the two finest points."""
    if len(hs) < 2:
        return None
    (h1, y1), (h2, y2) = sorted(zip(hs, ys))[:2]
    r = (h2 / h1) ** p
    return (r * y1 - y2) / (r - 1)


def observed_order(hs: list, ys: list) -> float | None:
    """Observed convergence order from three meshes with a constant refinement ratio (h sorted descending)."""
    if len(hs) < 3:
        return None
    s = sorted(zip(hs, ys), reverse=True)[-3:]
    (h0, y0), (h1, y1), (h2, y2) = s
    d1, d2 = y0 - y1, y1 - y2
    if d1 == 0 or d2 == 0 or d1 / d2 <= 0:
        return None
    return math.log(abs(d1 / d2)) / math.log(h0 / h1)


def write_record(name: str, rec: dict) -> Path:
    rec = dict(rec)
    rec.setdefault("status", "beta")
    p = RECS / f"cavity_repro_{name}.json"
    p.write_text(json.dumps(rec, indent=1, default=float))
    return p


# ---------------------------------------------------------------------------------------------------------------
# 3D helpers (RADES filters, CAPP sector cells). beta.

def rf_skin(sigma: float, T_K: float = 4.0) -> dict:
    """`conductor` material with the classical skin-effect rf model (same as `skin`)."""
    return skin(sigma, T_K)


def design3d(name: str, domain: dict, *, h: float, order: int = 2, modes: int = 8, target_f_Hz: float,
             wall: dict, B_T=(0.0, 0.0, 1.0), symmetry: dict | None = None, refine: list | None = None,
             conductors: list | None = None, interior_point, probes: list | None = None,
             surface_size: float | None = None, extra_materials: dict | None = None, remove_pec: bool = False):
    """Closed 3D cavity whose interior is the solid `domain` (PEC domain wall with material `wall`).

    symmetry: {"x": "pec"|"pmc", ...} mirror planes through the origin (the model keeps the + side).
    refine: [{"name", "solid", "mesh_size"}] vacuum regions inside the cavity (local refinement).
    conductors: [{"name", "solid", "material"}] conductor regions (holes with remove_pec)."""
    from axionfem.design import Design

    regions = [{"name": "cavity", "material": "vacuum", "solid": domain}]
    for rg in refine or []:
        regions.append({"name": rg["name"], "material": "vacuum", "solid": rg["solid"], "mesh_size": rg["mesh_size"]})
    mats = {"wall_cu": wall}
    mats.update(extra_materials or {})
    for cd in conductors or []:
        regions.append({"name": cd["name"], "material": cd.get("material", "wall_cu"), "solid": cd["solid"]})
    fw = {"rf_only": True, "boundary": "pec", "modes": modes, "target_f_Hz": target_f_Hz,
          "eigensolver": {"method": "shift_invert"}, "source": {"kind": "uniform", "B_T": list(B_T)}}
    if probes:
        fw["field_probes"] = [list(p) for p in probes]
    mesh = {"order": order, "size": h, "surface_size": surface_size or h, "domain_solid": domain}
    if symmetry:
        mesh["symmetry"] = [{"name": f"sym_{ax}", "axis": ax, "type": t} for ax, t in symmetry.items()]
    if remove_pec:
        mesh["remove_pec"] = True
    doc = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d", "materials": mats, "regions": regions,
           "fullwave": fw, "cavity": {"interior_point": list(interior_point), "wall_material": "wall_cu",
                                      "losses": True},
           "mesh": mesh}
    return Design(doc)


def _solve_one3d(args):
    doc, wd, ranks = args
    from axionfem.design import Design

    wd = Path(wd)
    stamp = wd / "design_key.txt"
    res = wd / "results.json"
    key = _key(doc)
    if res.exists() and stamp.exists() and stamp.read_text().strip() == key:
        data = json.loads(res.read_text())
        return data, json.loads((wd / "wall_seconds.txt").read_text()) if (wd / "wall_seconds.txt").exists() else None
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = Design(doc).solve(wd, verbose=0, mpi_ranks=ranks)
    dt = time.time() - t0
    data = r.data
    if not res.exists():
        res.write_text(json.dumps(data))
    stamp.write_text(key)
    (wd / "wall_seconds.txt").write_text(json.dumps(dt))
    return data, dt


def solve_many3d(jobs: list, bench: str, ranks: int | None = None, parallel: int | None = None) -> list:
    """jobs: [(tag, Design)] 3D; each solve on `ranks` MPI ranks ($AXF_REPRO_RANKS, default 1), `parallel` solves at
    a time ($AXF_REPRO_WORKERS, default 1). Cached like solve_many."""
    ranks = ranks or int(os.environ.get("AXF_REPRO_RANKS", "1"))
    parallel = parallel or int(os.environ.get("AXF_REPRO_WORKERS", "1"))
    args = [(d.doc, str(WORK / bench / tag), ranks) for tag, d in jobs]
    if parallel <= 1 or len(args) == 1:
        outs = [_solve_one3d(a) for a in args]
    elif ranks > 1:  # each solve is its own MPI subprocess (axf solve): threads only wait on it
        import concurrent.futures as cf2
        with cf2.ThreadPoolExecutor(max_workers=min(parallel, len(args))) as ex:
            outs = list(ex.map(_solve_one3d, args))
    else:  # in-process solves must not share one process (global log level, MPI_COMM_WORLD): separate processes
        with cf.ProcessPoolExecutor(max_workers=min(parallel, len(args))) as ex:
            outs = list(ex.map(_solve_one3d, args))
    return [(tag, data, dt) for (tag, _), (data, dt) in zip(jobs, outs)]


def modes3d(data: dict) -> list:
    return data["fullwave"]["eigenmodes"]["modes"]


def rounded_box_xz(x0, x1, y0, y1, z0, z1, r: float) -> dict:
    """Box whose four edges parallel to y are rounded with radius r (r = 0: plain box)."""
    if r <= 0:
        return {"box": {"min": [x0, y0, z0], "max": [x1, y1, z1]}}
    parts = [{"box": {"min": [x0, y0, z0 + r], "max": [x1, y1, z1 - r]}},
             {"box": {"min": [x0 + r, y0, z0], "max": [x1 - r, y1, z1]}}]
    for xc in (x0 + r, x1 - r):
        for zc in (z0 + r, z1 - r):
            parts.append({"cylinder": {"base": [xc, y0, zc], "axis": [0.0, y1 - y0, 0.0], "radius": r}})
    return {"union": parts}
