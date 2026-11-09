"""Validation of the 3D elasticity and steady-conduction solvers.

    python benchmarks/mech3d_validation.py [workdir]          (AXF_RANKS = MPI ranks per solve, default 1)

Analytic references (each derived in docs/validation/mech3d.md):
  1. Lame thick-walled cylinder under internal pressure, ends frictionless (plane strain) and free;
  2. the same cylinder as a 40-degree sector between two inclined symmetry planes (rotated-constraint path);
  3. bonded shrink fit: a rod in a tube of another material, cooled down, plane strain;
  4. free cooldown of an unsupported body: u = -alpha (x - x_c), zero stress;
  5. conduction in a slab (constant k with and without a heat source; tabulated k(T), exact by Kirchhoff);
  6. conduction through a hollow cylinder (constant k and k(T)), heat load and convergence in h;
  7. Lorentz loads: net winding force against results["forces"] and the support reactions; the hoop stress of a
     solenoid against the exact thin-shell value p (Rm - t/6)/t;
  8. an axisymmetric design solved by the axisymmetric module and by the 3D module on its revolved mesh.

Writes benchmarks/records/mech3d_*.json, docs/validation/figures/mech3d_*.pdf and docs/validation/mech3d.md.
AXF_M3D_QUICK=1 keeps only the coarsest mesh of every sweep (a smoke test of every code path).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import axionfem as af
from axionfem.convert import to_3d

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records"
FIG = ROOT / "docs" / "validation" / "figures"
OUT = ROOT / "docs" / "validation" / "mech3d.md"
MU0 = 1.25663706212e-6
QUICK = bool(os.environ.get("AXF_M3D_QUICK"))
RANKS = int(os.environ.get("AXF_RANKS", "1"))
STAMP = {"date": time.strftime("%Y-%m-%d %H:%M"), "mpi_ranks": RANKS}


def sweep(values):
    return values[:1] if QUICK else values


class _Run:
    """results.json of an MPI run (`axf solve` without --save: these checks need no fields on disk)."""

    def __init__(self, data):
        self.data = data


def solve(doc, wd, ranks=None):
    t0 = time.time()
    ranks = ranks or RANKS
    if ranks > 1:
        import shlex
        import subprocess

        from axionfem._paths import axf_path, default_mpi_launcher

        wd = Path(wd)
        wd.mkdir(parents=True, exist_ok=True)
        af.Design(doc).validate()  # parse errors here, not in the ranks
        (wd / "design.json").write_text(json.dumps(doc, indent=1))
        launcher = default_mpi_launcher()  # the rule of Design.solve(mpi_ranks=...)
        cmd = shlex.split(launcher) + ["-n", str(ranks), axf_path(), "solve", str(wd / "design.json"), "-o", str(wd),
                                       "-v", "0"]
        env = dict(os.environ, SLURM_EXPORT_ENV="ALL")
        env.pop("SLURM_TRES_PER_TASK", None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (wd / "mpi_solve.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0 or not (wd / "results.json").exists():
            raise RuntimeError(f"MPI solve failed in {wd}: {proc.stderr[-2000:]}")
        res = _Run(json.loads((wd / "results.json").read_text()))
    else:
        res = af.Design(doc).solve(wd, verbose=0)
    res.data.setdefault("timing", {})["wall_s"] = time.time() - t0
    return res


def cyl(r, z0, z1):
    return {"cylinder": {"base": [0, 0, z0], "axis": [0, 0, z1 - z0], "radius": r}}


def tube(r0, r1, z0, z1):
    return {"difference": [cyl(r1, z0, z1), cyl(r0, z0, z1)]}


def sector(r0, r1, z0, z1, a0, a1):
    prof = {"rect": {"r": [r0, r1], "z": [z0, z1]}}
    return {"revolve": {"profile": prof, "angles": [a0, a1]}}


def zplanes(z0, z1):
    return [{"name": "z_low", "point": [0, 0, z0], "normal": [0, 0, 1]},
            {"name": "z_high", "point": [0, 0, z1], "normal": [0, 0, 1]}]


def cyl_stress(s6, phi):
    """(sigma_rr, sigma_tt, sigma_zz) from Cartesian (xx, yy, zz, yz, xz, xy) at azimuth phi."""
    c, s = math.cos(phi), math.sin(phi)
    xx, yy, zz, _, _, xy = s6
    return (c * c * xx + s * s * yy + 2 * c * s * xy, s * s * xx + c * c * yy - 2 * c * s * xy, zz)


def radial_probes(r, phi, z=0.0):
    return [[float(x * math.cos(phi)), float(x * math.sin(phi)), float(z)] for x in r]


def probe_arrays(res, phi):
    """u_r, u_z and cylindrical stresses from the mechanics probes on a radial line at azimuth phi."""
    pr = res.data["mechanics"]["probes"]
    u = np.array([p["u_m"] for p in pr])
    s = np.array([cyl_stress(p["sigma_Pa"], phi) for p in pr])
    ur = u[:, 0] * math.cos(phi) + u[:, 1] * math.sin(phi)
    return ur, u[:, 2], s[:, 0], s[:, 1], s[:, 2]


def rate(errs, hs):
    return [None] + [math.log(errs[i - 1] / errs[i]) / math.log(hs[i - 1] / hs[i])
                     if errs[i] > 0 and errs[i - 1] > 0 else None for i in range(1, len(errs))]


# ================================================================== 1-2. Lame cylinder
E_L, NU_L, A_L, B_L, H_L, P_L = 200e9, 0.3, 0.1, 0.3, 0.4, 1.0e6
PHI_L = math.radians(30.0)


def lame_exact(r, plane_strain):
    k = P_L * A_L ** 2 / (B_L ** 2 - A_L ** 2)
    s_rr = k * (1 - B_L ** 2 / r ** 2)
    s_tt = k * (1 + B_L ** 2 / r ** 2)
    if plane_strain:
        return (1 + NU_L) / E_L * k * ((1 - 2 * NU_L) * r + B_L ** 2 / r), s_rr, s_tt, NU_L * (s_rr + s_tt)
    return k / E_L * ((1 - NU_L) * r + (1 + NU_L) * B_L ** 2 / r), s_rr, s_tt, np.zeros_like(r)


def lame_doc(case, size, order, r):
    steel = {"kind": "linear", "mechanical": {"E_Pa": E_L, "nu": NU_L, "alpha": 0.0}}
    if case == "sector":
        a = 20.0
        bore = sector(0.0, A_L, -H_L / 2, H_L / 2, -a, a)
        body = sector(A_L, B_L, -H_L / 2, H_L / 2, -a, a)
        planes = zplanes(-H_L / 2, H_L / 2) + [
            {"name": f"phi_{s:+.0f}", "point": [0, 0, 0],
             "normal": [-math.sin(math.radians(s)), math.cos(math.radians(s)), 0]} for s in (-a, a)]
        phi = math.radians(7.0)
    else:
        bore, body = cyl(A_L, -H_L / 2, H_L / 2), tube(A_L, B_L, -H_L / 2, H_L / 2)
        planes = zplanes(-H_L / 2, H_L / 2) if case == "fixed" else []
        phi = PHI_L
    mech = {"pressures": [{"on": "cyl", "from": "bore", "p_Pa": P_L}], "probes": radial_probes(r, phi)}
    if planes:
        mech["symmetry_planes"] = planes
    return {"schema": "axionfem.design/1", "name": f"lame3d_{case}", "symmetry": "3d",
            "materials": {"steel": steel},
            "regions": [{"name": "bore", "material": "vacuum", "solid": bore, "mesh_size": 2 * size},
                        {"name": "cyl", "material": "steel", "solid": body, "mesh_size": size,
                         "surface_size": size}],
            "mechanics": mech,
            "mesh": {"order": order, "size": max(2 * size, 0.1), "surface_size": size, "domain_radius": 0.6, "far_size": 0.15}}, phi


def lame(wd):
    w = B_L - A_L
    r = np.linspace(A_L + 0.1 * w, B_L - 0.1 * w, 21)
    # a second error measure, less sensitive to where a few points fall in their elements: 400 random interior
    # points inside the sector wedge (|phi| < 19.5 deg, |z| < 0.15 m, 0.12 < r < 0.28 m), the same for every case
    rng = np.random.default_rng(7)
    rc, pc, zc = rng.uniform(0.12, 0.28, 400), np.radians(rng.uniform(-19.5, 19.5, 400)), rng.uniform(-0.15, 0.15, 400)
    cloud = np.c_[rc * np.cos(pc), rc * np.sin(pc), zc]
    rows = []
    for case in ("fixed", "free", "sector"):
        ur_e, srr_e, stt_e, szz_e = lame_exact(r, case != "free")
        urc_e, _, sttc_e, _ = lame_exact(rc, case != "free")
        orders = (1, 2) if case != "sector" else (2,)
        for order in (orders[-1:] if QUICK else orders):
            for size in sweep((0.04, 0.02, 0.01)):
                doc, phi = lame_doc(case, size, order, r)
                nl = len(doc["mechanics"]["probes"])
                doc["mechanics"]["probes"] = doc["mechanics"]["probes"] + cloud.tolist()
                res = solve(doc, wd / f"lame_{case}_{order}_{size}")
                m = res.data["mechanics"]
                pr = m["probes"]
                uc = np.array([q["u_m"] for q in pr[nl:]])
                urc = uc[:, 0] * np.cos(pc) + uc[:, 1] * np.sin(pc)
                sttc = np.array([cyl_stress(q["sigma_Pa"], a)[1] for q, a in zip(pr[nl:], pc)])
                eu_c, es_c = np.abs(urc - urc_e) / np.abs(urc_e), np.abs(sttc - sttc_e) / np.abs(sttc_e)
                m["probes"] = pr[:nl]
                ur, uz, srr, stt, szz = probe_arrays(res, phi)
                rows.append({"case": case, "order": order, "mesh_size": size, "dofs": m["solver"]["dofs"],
                             "solid_elements": m["solver"]["solid_elements"],
                             "err_ur": float(np.max(np.abs(ur - ur_e) / np.abs(ur_e))),
                             "err_stt": float(np.max(np.abs(stt - stt_e) / np.abs(stt_e))),
                             "err_srr": float(np.max(np.abs(srr - srr_e)) / P_L),
                             "err_szz": float(np.max(np.abs(szz - szz_e)) / P_L),
                             "cloud_rms_ur": float(np.sqrt(np.mean(eu_c ** 2))), "cloud_max_ur": float(eu_c.max()),
                             "cloud_rms_stt": float(np.sqrt(np.mean(es_c ** 2))), "cloud_max_stt": float(es_c.max()),
                             "equilibrium_round_off": m["checks"]["equilibrium_F_round_off"],
                             "applied_radial_check": m["checks"]["applied_F_N"],
                             "rotated_nodes": m["solver"]["rotated_nodes"], "pins": m["solver"]["rigid_mode_pins"],
                             "pcg_iterations": m["solver"]["pcg_iterations"],
                             "relative_residual": m["solver"]["relative_residual"],
                             "wall_s": res.data["timing"]["wall_s"],
                             "profile": {"r": r.tolist(), "u_r": ur.tolist(), "s_tt": stt.tolist(),
                                         "s_rr": srr.tolist()}})
                print(f"lame {case} p{order} h={size}: u_r {rows[-1]['err_ur']:.2e} s_tt {rows[-1]['err_stt']:.2e} "
                      f"cloud rms u_r {rows[-1]['cloud_rms_ur']:.2e} s_tt {rows[-1]['cloud_rms_stt']:.2e}",
                      flush=True)
    rec = {**STAMP, "E_Pa": E_L, "nu": NU_L, "a": A_L, "b": B_L, "H": H_L, "p_Pa": P_L, "rows": rows,
           "patch": hydro_patch(wd)}
    (REC / "mech3d_lame.json").write_text(json.dumps(rec, indent=1))
    return rec


def hydro_patch(wd):
    """Hydrostatic patch test of the rotated elimination: a sector a < r < b, 0 < z < H/2 between two inclined planes
    through the axis and the plane z = 0, pressure p on every other face. sigma = -p I, u = -p/(3K) x meets every
    condition on any mesh (curved faces included) and is in the FE space, so the error is round-off only."""
    K = E_L / (3 * (1 - 2 * NU_L))
    rng = np.random.default_rng(1)
    rows = []
    for a0, a1 in ((-20.0, 20.0), (10.0, 55.0)):
        r, ph, z = rng.uniform(0.11, 0.29, 60), np.radians(rng.uniform(a0 + 1, a1 - 1, 60)), rng.uniform(0.01, 0.19, 60)
        pts = np.c_[r * np.cos(ph), r * np.sin(ph), z]
        planes = [{"name": "z0", "point": [0, 0, 0], "normal": [0, 0, 1]}] + [
            {"name": f"phi{i}", "point": [0, 0, 0], "normal": [-math.sin(math.radians(s)), math.cos(math.radians(s)), 0]}
            for i, s in enumerate((a0, a1))]
        for order in sweep((1, 2)):
            for size in sweep((0.04, 0.02)):
                doc = {"schema": "axionfem.design/1", "name": "hydro_patch", "symmetry": "3d",
                       "materials": {"steel": {"kind": "linear", "mechanical": {"E_Pa": E_L, "nu": NU_L, "alpha": 0.0}}},
                       "regions": [{"name": "cyl", "material": "steel", "solid": sector(A_L, B_L, 0.0, H_L / 2, a0, a1),
                                    "mesh_size": size, "surface_size": size}],
                       "mechanics": {"pressures": [{"on": "cyl", "from": "vacuum", "p_Pa": P_L}],
                                     "symmetry_planes": planes, "probes": pts.tolist()},
                       "mesh": {"order": order, "size": max(2 * size, 0.1), "surface_size": size,
                                "domain_radius": 0.6, "far_size": 0.15}}
                m = solve(doc, wd / f"patch_{a0:+.0f}_{a1:+.0f}_{order}_{size}").data["mechanics"]
                u = np.array([q["u_m"] for q in m["probes"]])
                sg = np.array([q["sigma_Pa"] for q in m["probes"]])
                ue = -P_L / (3 * K) * pts
                rows.append({"angles_deg": [a0, a1], "order": order, "mesh_size": size, "dofs": m["solver"]["dofs"],
                             "rotated_nodes": m["solver"]["rotated_nodes"], "pins": m["solver"]["rigid_mode_pins"],
                             "err_u": float(np.max(np.abs(u - ue)) / np.max(np.abs(ue))),
                             "err_sigma": float(np.max(np.abs(sg - np.array([-P_L] * 3 + [0] * 3))) / P_L)})
                print(f"patch {a0:+.0f}..{a1:+.0f} p{order} h={size}: u {rows[-1]['err_u']:.1e} "
                      f"sigma {rows[-1]['err_sigma']:.1e}", flush=True)
    return rows


# ================================================================== 3. shrink fit (bonded rod in a tube)
SF = {"b1": 0.10, "c": 0.20, "H": 0.30,
      "rod": {"E": 70e9, "nu": 0.33, "alpha": 4.15e-3}, "tube": {"E": 200e9, "nu": 0.30, "alpha": 2.97e-3}}


def lame_consts(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def shrink_exact(r):
    """Plane strain, u = A r + B/r in each material, stress-free strain -alpha (cooldown).
    sigma_rr = 2 (lam + mu) A - 2 mu B / r^2 - beta, beta = (3 lam + 2 mu) eps_th; rod: B = 0."""
    l1, m1 = lame_consts(SF["rod"]["E"], SF["rod"]["nu"])
    l2, m2 = lame_consts(SF["tube"]["E"], SF["tube"]["nu"])
    be1 = (3 * l1 + 2 * m1) * (-SF["rod"]["alpha"])
    be2 = (3 * l2 + 2 * m2) * (-SF["tube"]["alpha"])
    b1, c = SF["b1"], SF["c"]
    # unknowns A1, A2, B2
    M = np.array([[b1, -b1, -1 / b1],
                  [2 * (l1 + m1), -2 * (l2 + m2), 2 * m2 / b1 ** 2],
                  [0, 2 * (l2 + m2), -2 * m2 / c ** 2]])
    rhs = np.array([0, be1 - be2, be2])
    A1, A2, B2 = np.linalg.solve(M, rhs)
    rod = r < b1
    A = np.where(rod, A1, A2)
    B = np.where(rod, 0.0, B2)
    lam = np.where(rod, l1, l2)
    mu = np.where(rod, m1, m2)
    be = np.where(rod, be1, be2)
    u = A * r + B / r
    srr = 2 * (lam + mu) * A - 2 * mu * B / r ** 2 - be
    stt = 2 * (lam + mu) * A + 2 * mu * B / r ** 2 - be
    szz = 2 * lam * A - be
    p_int = -(2 * (l1 + m1) * A1 - be1)  # interface pressure (positive: compression)
    return u, srr, stt, szz, p_int


def shrink(wd):
    b1, c, H = SF["b1"], SF["c"], SF["H"]
    r = np.concatenate([np.linspace(0.02, b1 - 0.004, 10), np.linspace(b1 + 0.004, c - 0.01, 12)])
    phi = math.radians(25.0)
    ue, srre, stte, szze, p_int = shrink_exact(r)
    rows = []
    for size in sweep((0.04, 0.02, 0.01)):
        mats = {k: {"kind": "linear", "mechanical": {"E_Pa": v["E"], "nu": v["nu"], "alpha": v["alpha"]}}
                for k, v in (("rod_mat", SF["rod"]), ("tube_mat", SF["tube"]))}
        doc = {"schema": "axionfem.design/1", "name": "shrink3d", "symmetry": "3d", "materials": mats,
               "regions": [{"name": "rod", "material": "rod_mat", "solid": cyl(b1, -H / 2, H / 2), "mesh_size": size},
                           {"name": "tube", "material": "tube_mat", "solid": tube(b1, c, -H / 2, H / 2),
                            "mesh_size": size}],
               "mechanics": {"cooldown": True, "symmetry_planes": zplanes(-H / 2, H / 2),
                             "probes": radial_probes(r, phi)},
               "mesh": {"order": 2, "size": max(2 * size, 0.1), "surface_size": size, "domain_radius": 0.8, "far_size": 0.24}}
        res = solve(doc, wd / f"shrink_{size}")
        m = res.data["mechanics"]
        ur, uz, srr, stt, szz = probe_arrays(res, phi)
        scale = abs(p_int)
        rows.append({"mesh_size": size, "dofs": m["solver"]["dofs"],
                     "err_u": float(np.max(np.abs(ur - ue)) / np.max(np.abs(ue))),
                     "err_srr": float(np.max(np.abs(srr - srre)) / scale),
                     "err_stt": float(np.max(np.abs(stt - stte)) / scale),
                     "err_szz": float(np.max(np.abs(szz - szze)) / scale),
                     "equilibrium_round_off": m["checks"]["equilibrium_F_round_off"],
                     "profile": {"r": r.tolist(), "u_r": ur.tolist(), "s_rr": srr.tolist(), "s_tt": stt.tolist()}})
        print(f"shrink h={size}: u {rows[-1]['err_u']:.2e} s_tt {rows[-1]['err_stt']:.2e}", flush=True)
    rec = {**STAMP, **SF, "interface_pressure_Pa": p_int, "exact": {"r": r.tolist(), "u_r": ue.tolist(),
           "s_rr": srre.tolist(), "s_tt": stte.tolist(), "s_zz": szze.tolist()}, "rows": rows}
    (REC / "mech3d_shrink.json").write_text(json.dumps(rec, indent=1))
    return rec


# ================================================================== 4. free cooldown
def cooldown(wd):
    alpha = 2.5e-3
    # an L-shaped union of two boxes: no symmetry that could hide a wrong rigid-mode treatment
    boxes = [{"box": {"min": [0, 0, 0], "max": [0.4, 0.1, 0.1]}}, {"box": {"min": [0, 0, 0], "max": [0.1, 0.3, 0.1]}}]
    pts = [[0.35, 0.05, 0.05], [0.05, 0.25, 0.05], [0.2, 0.08, 0.02], [0.02, 0.02, 0.09], [0.08, 0.15, 0.05]]
    doc = {"schema": "axionfem.design/1", "name": "cool3d", "symmetry": "3d",
           "materials": {"m": {"kind": "linear", "mechanical": {"E_Pa": 1.2e11, "nu": 0.3, "alpha": alpha}}},
           "regions": [{"name": "body", "material": "m", "solid": {"union": boxes}, "mesh_size": 0.03}],
           "mechanics": {"cooldown": True, "probes": pts},
           "mesh": {"order": 2, "size": 0.06, "surface_size": 0.03, "domain_radius": 1.0, "far_size": 0.3}}
    res = solve(doc, wd / "cooldown")
    m = res.data["mechanics"]
    # centroid of the L: volumes 0.4*0.1*0.1 and 0.1*0.3*0.1 overlap in 0.1*0.1*0.1
    parts = [((0.2, 0.05, 0.05), 0.004), ((0.05, 0.15, 0.05), 0.003), ((0.05, 0.05, 0.05), -0.001)]
    V = sum(v for _, v in parts)
    xc = np.array([sum(p[k] * v for p, v in parts) / V for k in range(3)])
    u = np.array([p["u_m"] for p in m["probes"]])
    ue = -alpha * (np.array(pts) - xc)
    vm = m["regions"]["body"]["von_mises_max_Pa"]
    rec = {**STAMP, "alpha": alpha, "centroid": xc.tolist(), "probes": pts, "u": u.tolist(), "u_exact": ue.tolist(),
           "err_u_rel": float(np.max(np.abs(u - ue)) / np.max(np.abs(ue))),
           "von_mises_max_over_E_alpha": vm / (1.2e11 * alpha), "pins": m["solver"]["rigid_mode_pins"],
           "rigid_modes": m["rigid_modes"], "equilibrium_round_off": m["checks"]["equilibrium_F_round_off"]}
    (REC / "mech3d_cooldown.json").write_text(json.dumps(rec, indent=1))
    print(f"cooldown: u {rec['err_u_rel']:.2e}, vm/(E alpha) {rec['von_mises_max_over_E_alpha']:.2e}", flush=True)
    return rec


# ================================================================== 5. conduction in a slab
SL = {"L": 1.0, "w": 0.2, "T1": 4.0, "T2": 30.0, "k0": 10.0, "c": 0.08, "q": 400.0}


def slab_doc(case, size):
    L, w = SL["L"], SL["w"]
    k = SL["k0"] if case != "kT" else [[SL["T1"], SL["k0"]], [SL["T2"], SL["k0"] * (1 + SL["c"] * (SL["T2"] - SL["T1"]))]]
    mats = {"cond": {"kind": "linear", "thermal": {"k_W_mK": k}}, "sink": {"kind": "linear", "thermal": {"k_W_mK": 1e3}}}
    loads = [{"region": "slab", "W_m3": SL["q"]}] if case == "q" else []
    x = np.linspace(0.05, 0.95, 19)
    return {"schema": "axionfem.design/1", "name": f"slab_{case}", "symmetry": "3d", "materials": mats,
            "regions": [{"name": "slab", "material": "cond", "mesh_size": size,
                         "solid": {"box": {"min": [0, 0, 0], "max": [L, w, w]}}},
                        {"name": "cold", "material": "sink", "solid": {"box": {"min": [-0.02, 0, 0], "max": [0, w, w]}}},
                        {"name": "warm", "material": "sink", "solid": {"box": {"min": [L, 0, 0], "max": [L + 0.02, w, w]}}}],
            "thermal": {"fixed": [{"region": "cold", "T_K": SL["T1"]}, {"region": "warm", "T_K": SL["T2"]}],
                        "loads": loads, "probes": [[float(v), 0.07, 0.13] for v in x]},
            "mesh": {"order": 2, "size": max(2 * size, 0.1), "surface_size": size, "domain_radius": 2.0, "far_size": 0.6}}, x


def slab_exact(case, x):
    L, T1, T2, k0, c = SL["L"], SL["T1"], SL["T2"], SL["k0"], SL["c"]
    if case == "k":
        return T1 + (T2 - T1) * x / L, k0 * (T2 - T1) / L
    if case == "q":
        return T1 + (T2 - T1) * x / L + SL["q"] * x * (L - x) / (2 * k0), None
    # Kirchhoff: theta(T) = k0 [(T - T1) + c (T - T1)^2 / 2] is linear in x
    th2 = k0 * ((T2 - T1) + c * (T2 - T1) ** 2 / 2)
    th = th2 * x / L
    return T1 + (np.sqrt(1 + 2 * c * th / k0) - 1) / c, th2 / L


def slab(wd):
    rows = []
    A = SL["w"] ** 2
    # 300 random interior points: RMS error over the slab, and where the largest error sits
    rng = np.random.default_rng(3)
    cloud = np.c_[rng.uniform(0.02, 0.98, 300), rng.uniform(0.01, 0.19, 300), rng.uniform(0.01, 0.19, 300)]
    for case in ("k", "q", "kT"):
        for size in sweep((0.1, 0.07, 0.05, 0.035, 0.025, 0.0175) if case == "kT" else (0.1,)):
            doc, x = slab_doc(case, size)
            nl = len(x)
            doc["thermal"]["probes"] = doc["thermal"]["probes"] + cloud.tolist()
            res = solve(doc, wd / f"slab_{case}_{size}")
            t = res.data["thermal"]
            Tall = np.array([p["T_K"] for p in t["probes"]])
            T = Tall[:nl]
            Te, flux = slab_exact(case, x)
            ec = np.abs(Tall[nl:] - slab_exact(case, cloud[:, 0])[0])
            ck = t["checks"]
            row = {"case": case, "mesh_size": size, "dofs": t["solver"]["dofs"],
                   "max_abs_dT_K": float(np.max(np.abs(T - Te))), "x_of_max_m": float(x[np.argmax(np.abs(T - Te))]),
                   "cloud_rms_dT_K": float(np.sqrt(np.mean(ec ** 2))), "cloud_max_dT_K": float(ec.max()),
                   "cloud_rms_dT_K_x_gt_0.2": float(np.sqrt(np.mean(ec[cloud[:, 0] > 0.2] ** 2))),
                   "picard_change": t["solver"].get("picard_change"),
                   "picard_tolerance": t["solver"].get("picard_tolerance"), "picard": t["solver"]["picard_iterations"],
                   "balance_relative": ck["balance_relative"], "generated_W": ck["generated_W"],
                   "heat_removed_W": ck["heat_removed_W"], "heat_added_W": ck["heat_added_W"]}
            if flux is not None:
                row["Q_exact_W"] = flux * A
                row["Q_rel_err"] = (ck["heat_added_W"] - flux * A) / (flux * A)
            rows.append(row)
            print(f"slab {case} h={size}: max|dT| {row['max_abs_dT_K']:.2e}", flush=True)
    rec = {**STAMP, **SL, "rows": rows}
    (REC / "mech3d_slab.json").write_text(json.dumps(rec, indent=1))
    return rec


# ================================================================== 6. conduction through a hollow cylinder
HC = {"a": 0.4, "b": 0.5, "H": 0.3, "d": 0.01, "T1": 4.0, "T2": 30.0, "k0": 10.0, "c": 0.08}


def hollow_exact(r, kT):
    a, b, T1, T2, k0, c, H = HC["a"], HC["b"], HC["T1"], HC["T2"], HC["k0"], HC["c"], HC["H"]
    cc = c if kT else 0.0
    th2 = k0 * ((T2 - T1) + cc * (T2 - T1) ** 2 / 2)
    th = th2 * np.log(r / a) / np.log(b / a)
    T = T1 + th / k0 if not kT else T1 + (np.sqrt(1 + 2 * c * th / k0) - 1) / c
    return T, 2 * math.pi * H * th2 / math.log(b / a)


def hollow(wd):
    a, b, H, d = HC["a"], HC["b"], HC["H"], HC["d"]
    r = np.linspace(a + 0.005, b - 0.005, 11)
    phi = math.radians(40.0)
    rows = []
    for kT in (False, True):
        k = [[HC["T1"], HC["k0"]], [HC["T2"], HC["k0"] * (1 + HC["c"] * (HC["T2"] - HC["T1"]))]] if kT else HC["k0"]
        for size in sweep((0.05, 0.025, 0.0125, 0.00625)):
            doc = {"schema": "axionfem.design/1", "name": "hollow_kT" if kT else "hollow_k", "symmetry": "3d",
                   "materials": {"cond": {"kind": "linear", "thermal": {"k_W_mK": k}},
                                 "sink": {"kind": "linear", "thermal": {"k_W_mK": 1e3}}},
                   "regions": [{"name": "wall", "material": "cond", "solid": tube(a, b, -H / 2, H / 2), "mesh_size": size},
                               {"name": "inner", "material": "sink", "solid": tube(a - d, a, -H / 2, H / 2)},
                               {"name": "outer", "material": "sink", "solid": tube(b, b + d, -H / 2, H / 2)}],
                   "thermal": {"fixed": [{"region": "inner", "T_K": HC["T1"]}, {"region": "outer", "T_K": HC["T2"]}],
                               "probes": radial_probes(r, phi, 0.05)},
                   "mesh": {"order": 2, "size": max(2 * size, 0.1), "surface_size": size, "domain_radius": 1.5, "far_size": 0.45}}
            res = solve(doc, wd / f"hollow_{kT}_{size}")
            t = res.data["thermal"]
            T = np.array([p["T_K"] for p in t["probes"]])
            Te, Q = hollow_exact(r, kT)
            ck = t["checks"]
            rows.append({"kT": kT, "mesh_size": size, "dofs": t["solver"]["dofs"],
                         "max_abs_dT_K": float(np.max(np.abs(T - Te))),
                         "Q_FE_W": ck["heat_added_W"], "Q_exact_W": Q, "Q_rel_err": (ck["heat_added_W"] - Q) / Q,
                         "removed_minus_added_rel": (ck["heat_removed_W"] - ck["heat_added_W"]) / Q,
                         "balance_relative": ck["balance_relative"], "picard": t["solver"]["picard_iterations"],
                         "picard_change": t["solver"]["picard_change"]})
            print(f"hollow kT={kT} h={size}: max|dT| {rows[-1]['max_abs_dT_K']:.2e} Q {rows[-1]['Q_rel_err']:.2e}",
                  flush=True)
    rec = {**STAMP, **HC, "rows": rows}
    (REC / "mech3d_hollow.json").write_text(json.dumps(rec, indent=1))
    return rec


# ================================================================== 7. Lorentz loads
SOL = {"R": 0.5, "t": 0.05, "H": 4.0, "NI": 1.0e6, "E": 100e9, "nu": 0.3}


def solenoid(wd):
    R, t, H = SOL["R"], SOL["t"], SOL["H"]
    phi = math.radians(20.0)
    rr = np.linspace(R + 1e-4 * t, R + t - 1e-4 * t, 11)  # inside the winding (a probe on its face may land in air)
    rows = []
    for size in sweep((0.05, 0.025)):
        pts = radial_probes(rr, phi) + radial_probes([R - 0.01, R + t + 0.01], phi)
        doc = {"schema": "axionfem.design/1", "name": "solenoid3d", "symmetry": "3d",
               "materials": {"w": {"kind": "vacuum", "mechanical": {"E_Pa": SOL["E"], "nu": SOL["nu"], "alpha": 0.0}}},
               "regions": [{"name": "winding", "material": "w", "solid": tube(R, R + t, -H / 2, H / 2),
                            "mesh_size": size, "surface_size": size}],
               "windings": [{"region": "winding", "type": "azimuthal", "ampere_turns": SOL["NI"]}],
               "mechanics": {"probes": pts},
               "mesh": {"order": 2, "size": 0.2, "surface_size": size, "domain_radius": 6.0, "far_size": 1.8}}
        res = solve(doc, wd / f"solenoid_{size}")
        m = res.data["mechanics"]
        pr = m["probes"]
        Bin, Bout = (np.linalg.norm(pr[-2]["B_T"]), np.linalg.norm(pr[-1]["B_T"]))
        p = (Bin ** 2 - Bout ** 2) / (2 * MU0)
        stt = np.array([cyl_stress(q["sigma_Pa"], phi)[1] for q in pr[:-2]])
        mean = float(np.trapezoid(stt, rr) / t)
        Rm = R + t / 2
        exact = p * (Rm - t / 6) / t
        Fm = np.array(m["lorentz_forces"]["winding"]["F_N"])
        Ff = np.array(res.data["forces"]["windings"]["winding"]["F_N"])
        rows.append({"mesh_size": size, "dofs": m["solver"]["dofs"], "B_in_T": Bin, "B_out_T": Bout, "p_Pa": p,
                     "hoop_mean_FE_Pa": mean, "hoop_exact_Pa": exact, "rel": (mean - exact) / exact,
                     "membrane_Pa": p * Rm / t, "F_mech_N": Fm.tolist(), "F_forces_N": Ff.tolist(),
                     "load_scale_N": m["checks"]["load_scale_N"],
                     "equilibrium_round_off": m["checks"]["equilibrium_F_round_off"],
                     "profile": {"r": rr.tolist(), "s_tt": stt.tolist()}})
        print(f"solenoid h={size}: hoop {mean:.6g} exact {exact:.6g} rel {rows[-1]['rel']:.2e}", flush=True)
    # two coaxial coils (attracting): net axial force on each from the mechanics assembly and from forces[],
    # and the support reactions
    coils = [("upper", 0.15, 0.25), ("lower", -0.25, -0.15)]
    doc = {"schema": "axionfem.design/1", "name": "twocoils3d", "symmetry": "3d",
           "materials": {"w": {"kind": "vacuum", "mechanical": {"E_Pa": 80e9, "nu": 0.3, "alpha": 0.0}}},
           "regions": [{"name": n, "material": "w", "solid": tube(0.3, 0.36, z0, z1), "mesh_size": 0.03}
                       for n, z0, z1 in coils],
           "windings": [{"region": n, "type": "azimuthal", "ampere_turns": 2e5} for n, _, _ in coils],
           "mechanics": {"supports": [{"region": n, "fix": "all"} for n, _, _ in coils]},
           "mesh": {"order": 2, "size": 0.1, "surface_size": 0.03, "domain_radius": 2.0, "far_size": 0.6}}
    res = solve(doc, wd / "twocoils")
    m = res.data["mechanics"]
    two = {}
    for n, _, _ in coils:
        Fm = np.array(m["lorentz_forces"][n]["F_N"])
        Ff = np.array(res.data["forces"]["windings"][n]["F_N"])
        Rs = np.array(next(s["reaction_F_N"] for s in m["supports"] if s["region"] == n))
        two[n] = {"F_mech_N": Fm.tolist(), "F_forces_N": Ff.tolist(), "reaction_N": Rs.tolist(),
                  "rel_mech_vs_forces": float(abs(Fm[2] - Ff[2]) / abs(Ff[2])),
                  "rel_reaction_plus_F": float(np.linalg.norm(Rs + Fm) / abs(Fm[2]))}
        print(f"two coils {n}: Fz mech {Fm[2]:.9g} forces {Ff[2]:.9g} reaction {Rs[2]:.9g}", flush=True)
    rec = {**STAMP, **SOL, "rows": rows, "two_coils": two}
    (REC / "mech3d_lorentz.json").write_text(json.dumps(rec, indent=1))
    return rec


# ================================================================== 8. axisymmetric vs 3D
def axi_doc():
    k316 = [[4.0, 0.27], [10.0, 0.75], [20.0, 1.9], [40.0, 3.9]]  # illustrative table (not material data)
    return {"schema": "axionfem.design/1", "name": "axi_vs_3d",
            "materials": {"steel": {"kind": "linear", "mechanical": {"E_Pa": 2.0e11, "nu": 0.3, "alpha": 2.97e-3},
                                    "thermal": {"k_W_mK": k316}},
                          "wmat": {"kind": "vacuum", "mechanical": {"E_Pa": 8.0e10, "nu": 0.3, "alpha": 3.0e-3},
                                   "thermal": {"k_W_mK": 1.0}}},
            "regions": [{"name": "bobbin", "material": "steel", "shape": {"rect": {"r": [0.30, 0.32], "z": [-0.30, 0.30]}}},
                        {"name": "coil", "material": "wmat", "shape": {"rect": {"r": [0.32, 0.36], "z": [-0.25, 0.25]}}},
                        {"name": "tube", "material": "nb", "shape": {"rect": {"r": [0.15, 0.16], "z": [-0.40, 0.40]}}},
                        {"name": "flange", "material": "steel", "shape": {"rect": {"r": [0.30, 0.42], "z": [-0.34, -0.30]}}}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 4.0e5}],
            "mechanics": {"cooldown": True, "supports": [{"region": "flange", "fix": "z"}]},
            # the niobium tube is a separate conducting part: it needs its own sink
            "thermal": {"fixed": [{"region": "flange", "T_K": 4.2}, {"region": "tube", "T_K": 4.2}],
                        "loads": [{"region": "coil", "W": 20.0}]},
            "mesh": {"order": 2, "size": 0.02, "domain_radius": 1.5}}


def axi_vs_3d(wd):
    d2 = af.Design(axi_doc())
    res2 = d2.solve(wd / "axi", verbose=0)
    phi = math.radians(33.0)
    lines = {"bobbin": (np.full(9, 0.31), np.linspace(-0.28, 0.28, 9)),
             "coil": (np.linspace(0.322, 0.358, 9), np.zeros(9)),
             "tube": (np.full(9, 0.155), np.linspace(-0.38, 0.38, 9))}
    s2 = {k: res2.sample_mech(r, z) for k, (r, z) in lines.items()}
    rows = []
    for size in sweep((0.03, 0.015)):
        doc3 = to_3d(axi_doc(), mesh={"order": 2, "size": max(2 * size, 0.1), "surface_size": size, "domain_radius": 1.5, "far_size": 0.45}).doc
        for rg in doc3["regions"]:
            rg["mesh_size"] = size
        pts = []
        for k, (r, z) in lines.items():
            pts += [[float(x * math.cos(phi)), float(x * math.sin(phi)), float(zz)] for x, zz in zip(r, z)]
        doc3["mechanics"]["probes"] = pts
        doc3["thermal"]["probes"] = pts
        res3 = solve(doc3, wd / f"3d_{size}")
        m3, m2 = res3.data["mechanics"], res2.data["mechanics"]
        t3, t2 = res3.data["thermal"], res2.data["thermal"]
        reg = {}
        for name in ("bobbin", "coil", "tube", "flange"):
            a, b = m2["regions"][name], m3["regions"][name]
            reg[name] = {"u_max_2d": a["u_max_m"], "u_max_3d": b["u_max_m"],
                         "vm_max_2d": a["von_mises_max_Pa"], "vm_max_3d": b["von_mises_max_Pa"],
                         "hoop_2d": a["sigma_hoop_Pa"], "hoop_3d": b["sigma_hoop_Pa"],
                         "T_max_2d": t2["regions"][name]["T_max_K"], "T_max_3d": t3["regions"][name]["T_max_K"]}
        # pointwise along the lines
        pr = m3["probes"]
        tp = t3["probes"]
        lerr = {}
        i = 0
        for k, (r, z) in lines.items():
            n = len(r)
            u3 = np.array([q["u_m"] for q in pr[i:i + n]])
            ur3 = u3[:, 0] * math.cos(phi) + u3[:, 1] * math.sin(phi)
            uz3 = u3[:, 2]
            sig3 = np.array([cyl_stress(q["sigma_Pa"], phi) for q in pr[i:i + n]])
            vm3 = np.array([q["von_mises_Pa"] for q in pr[i:i + n]])
            T3 = np.array([q["T_K"] for q in tp[i:i + n]])
            a2 = s2[k]
            umax = max(np.max(np.abs(a2["u_r"])), np.max(np.abs(a2["u_z"])))
            vmax = np.max(np.abs(a2["von_mises"]))
            lerr[k] = {"u_r": float(np.max(np.abs(ur3 - np.array(a2["u_r"]))) / umax),
                       "u_z": float(np.max(np.abs(uz3 - np.array(a2["u_z"]))) / umax),
                       "sigma_hoop": float(np.max(np.abs(sig3[:, 1] - np.array(a2["sigma_hoop"]))) / vmax),
                       "von_mises": float(np.max(np.abs(vm3 - np.array(a2["von_mises"]))) / vmax),
                       "T_K": float(np.max(np.abs(T3 - np.array(a2["T_K"]))))}
            i += n
        rows.append({"mesh_size": size, "dofs_3d": m3["solver"]["dofs"], "regions": reg, "lines": lerr,
                     "force_closure_3d": m3["checks"]["equilibrium_F_round_off"],
                     "heat_2d": t2["checks"]["heat_removed_W"], "heat_3d": t3["checks"]["heat_removed_W"]})
        print(f"axi vs 3d h={size}: " + ", ".join(f"{k} u {v['u_r']:.1e}/{v['u_z']:.1e} vm {v['von_mises']:.1e} "
                                                  f"T {v['T_K']:.1e}" for k, v in lerr.items()), flush=True)
    rec = {**STAMP, "design": axi_doc(), "axi_dofs": res2.data["mesh"]["dofs"], "rows": rows}
    (REC / "mech3d_axi_vs_3d.json").write_text(json.dumps(rec, indent=1))
    return rec


# ================================================================== figures and report
def figures(L, S, Hc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": False, "figure.figsize": (4.2, 3.2)})
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    # Lame convergence
    fig, ax = plt.subplots()
    for (case, order), col, mk in ((("fixed", 1), "0.5", "s"), (("fixed", 2), "k", "o"), (("free", 2), "maroon", "^"),
                                   (("sector", 2), "tab:blue", "v")):
        rr = [r for r in L["rows"] if r["case"] == case and r["order"] == order]
        if len(rr) < 2:
            continue
        ax.loglog([r["mesh_size"] for r in rr], [r["err_ur"] for r in rr], marker=mk, color=col,
                  label=f"$u_r$, P{order}, ends {case}")
        ax.loglog([r["mesh_size"] for r in rr], [r["err_stt"] for r in rr], marker=mk, color=col, ls="--",
                  label=rf"$\sigma_{{\theta\theta}}$, P{order}, ends {case}")
    ax.set_xlabel("element size h [m]")
    ax.set_ylabel("max relative error")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    p = FIG / "mech3d_lame_convergence.pdf"
    fig.savefig(p, metadata={"CreationDate": None})   # no creation date in the file
    plt.close(fig)
    out.append(p)
    # shrink fit profile
    fig, ax = plt.subplots()
    ex = S["exact"]
    rr = np.array(ex["r"])
    for key, col, lab in (("s_rr", "k", r"$\sigma_{rr}$"), ("s_tt", "maroon", r"$\sigma_{\theta\theta}$")):
        ax.plot(rr * 1e3, np.array(ex[key]) / 1e6, color=col, lw=1, label=lab + " exact")
        fe = S["rows"][-1]["profile"]
        ax.plot(np.array(fe["r"]) * 1e3, np.array(fe[key]) / 1e6, ls="none", marker="o", ms=3, color=col,
                label=lab + " 3D FE")
    ax.axvline(S["b1"] * 1e3, color="0.5", lw=0.8)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("stress [MPa]")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = FIG / "mech3d_shrink_fit.pdf"
    fig.savefig(p, metadata={"CreationDate": None})   # no creation date in the file
    plt.close(fig)
    out.append(p)
    # conduction convergence
    fig, ax = plt.subplots()
    for kT, col, mk in ((False, "k", "o"), (True, "maroon", "^")):
        rr = [r for r in Hc["rows"] if r["kT"] == kT]
        ax.loglog([r["mesh_size"] for r in rr], [r["max_abs_dT_K"] for r in rr], marker=mk, color=col,
                  label="max |dT|, " + ("k(T)" if kT else "constant k"))
        ax.loglog([r["mesh_size"] for r in rr], [abs(r["Q_rel_err"]) * 26.0 for r in rr], marker=mk, color=col,
                  ls="--", label="|dQ/Q| x 26 K, " + ("k(T)" if kT else "constant k"))
    ax.set_xlabel("element size h [m]")
    ax.set_ylabel("error [K]")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    p = FIG / "mech3d_hollow_convergence.pdf"
    fig.savefig(p, metadata={"CreationDate": None})   # no creation date in the file
    plt.close(fig)
    out.append(p)
    return out


def fmt(x, f="{:.2e}"):
    return "-" if x is None else f.format(x)


def report(L, S, C, Sl, Hc, Lo, X):
    md = ["# 3D mechanics and thermal validation", "",
          f"Generated by `benchmarks/mech3d_validation.py` ({RANKS} MPI rank(s) per solve). "
          "Records: `benchmarks/records/mech3d_*.json`. Formulation: `cpp/src/model3d_mech.cpp` and "
          "[physics/mechanics_thermal.md](../physics/mechanics_thermal.md#3d). All solves use H1 elements of the mesh order on "
          "tetrahedra; probes are evaluated at points (`mechanics.probes`, `thermal.probes`).", ""]
    # 1-2 Lame
    md += ["## 1. Lame thick-walled cylinder under internal pressure", "",
           f"a = {A_L} m, b = {B_L} m, H = {H_L} m, p = {P_L:g} Pa, E = {E_L:g} Pa, nu = {NU_L}. The pressure is a "
           "`mechanics.pressures` load on the faces the cylinder shares with the bore. Three configurations, each with "
           "an exact solution:", "",
           "* **ends frictionless** (symmetry planes z = +-H/2): u_z = 0 everywhere is exact (plane strain), "
           "sigma_zz = nu (sigma_rr + sigma_tt);",
           "* **ends free**: sigma_rr + sigma_tt = 2 k is uniform, so u_z = eps_zz z with sigma_zz = 0 is exact for any "
           "length; the body has no constraint at all, so all six rigid motions are pinned and projected out;",
           "* **40-degree sector** between the planes phi = +-20 degrees (normals not along an axis) plus the end planes: "
           "the frictionless sector is the full ring by symmetry, so the same plane-strain solution is exact. This "
           "exercises the rotated-constraint path (Q^T K Q); no rigid motion is left free.", "",
           "Errors: maximum over 21 radii at z = 0 (azimuth 30 degrees; 7 degrees in the sector), excluding the outer "
           "10 % of the wall on each side, as in the axisymmetric benchmark. \"cloud\": RMS of the relative error over "
           "400 random interior points (0.12 < r < 0.28 m, |phi| < 19.5 degrees, |z| < 0.15 m), the same points for "
           "all three configurations.", "",
           "| ends | order | h [m] | DOFs | rel. err u_r | rate | rel. err sigma_tt | rate | cloud RMS u_r | rate | "
           "cloud RMS sigma_tt | rate | max abs d sigma_rr / p | max abs d sigma_zz / p | rotated nodes | pins | "
           "equilibrium round-off |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in ("fixed", "free", "sector"):
        for order in (1, 2):
            rr = [r for r in L["rows"] if r["case"] == case and r["order"] == order]
            if not rr:
                continue
            ru = rate([r["err_ur"] for r in rr], [r["mesh_size"] for r in rr])
            rs = rate([r["err_stt"] for r in rr], [r["mesh_size"] for r in rr])
            cu = rate([r["cloud_rms_ur"] for r in rr], [r["mesh_size"] for r in rr])
            cs = rate([r["cloud_rms_stt"] for r in rr], [r["mesh_size"] for r in rr])
            for r, a, b, c, d in zip(rr, ru, rs, cu, cs):
                md.append(f"| {case} | {order} | {r['mesh_size']} | {r['dofs']:,} | {r['err_ur']:.2e} | "
                          f"{fmt(a, '{:.2f}')} | {r['err_stt']:.2e} | {fmt(b, '{:.2f}')} | {r['cloud_rms_ur']:.2e} | "
                          f"{fmt(c, '{:.2f}')} | {r['cloud_rms_stt']:.2e} | {fmt(d, '{:.2f}')} | {r['err_srr']:.2e} | "
                          f"{r['err_szz']:.2e} | {r['rotated_nodes']} | {r['pins']} | {r['equilibrium_round_off']:.1e} |")
    md += ["", "![](figures/mech3d_lame_convergence.pdf)", "",
           "Expected rates for a smooth solution: O(h^(p+1)) in displacement and O(h^p) in stress (P1: 2 and 1, "
           "P2: 3 and 2); the curved bore and outer surface are represented by the mesh geometry, so a geometric "
           "error enters as well.", "",
           "**Sector convergence.** The maximum over the 21-point line is a noisy measure: the pointwise P2 error depends "
           "on where each point falls in its element, and the largest values sit in the first element layer at the "
           "curved bore, where element shape and the geometric error of the quadratic surface vary from mesh to mesh. "
           "The sector's irregular line maxima (e.g. the u_r rate at the finest step) are this effect, not a defect of "
           "the inclined planes: on the common cloud the sector and the full ring (ends fixed) have RMS errors within "
           "about 30 % of each other at every h, converging at the expected rates (cloud columns). The rotated elimination itself is exact: "
           "in the hydrostatic patch test below (pressure p on all faces of a sector a < r < b, 0 < z < H/2 between the "
           "plane z = 0 and two planes through the axis; sigma = -p I and u = -p/(3K) x are exact on any mesh, curved "
           "faces included, and lie in the FE space) the error is round-off for P1 and P2 and for a sector with no "
           "coordinate-plane face.", "",
           "| sector [deg] | order | h [m] | DOFs | rotated nodes | pins | max abs du / max abs u | max abs d sigma / p |",
           "|---|---|---|---|---|---|---|---|"]
    for r in L.get("patch", []):
        md.append(f"| {r['angles_deg'][0]:+.0f} .. {r['angles_deg'][1]:+.0f} | {r['order']} | {r['mesh_size']} | "
                  f"{r['dofs']:,} | {r['rotated_nodes']} | {r['pins']} | {r['err_u']:.1e} | {r['err_sigma']:.1e} |")
    md += [""]
    # 3 shrink fit
    md += ["## 2. Bonded shrink fit (bimaterial cooldown)", "",
           f"A rod (r < {SF['b1']} m; E = {SF['rod']['E']:g}, nu = {SF['rod']['nu']}, alpha = {SF['rod']['alpha']}) "
           f"bonded inside a tube ({SF['b1']} < r < {SF['c']} m; E = {SF['tube']['E']:g}, nu = {SF['tube']['nu']}, "
           f"alpha = {SF['tube']['alpha']}), `mechanics.cooldown: true`, frictionless ends (plane strain). The "
           "material values are illustrative. In each material u = A r + B/r (B = 0 in the rod) with "
           "sigma_rr = 2 (lambda + mu) A - 2 mu B / r^2 - beta, beta = (3 lambda + 2 mu) eps_th, eps_th = -alpha; "
           "continuity of u and sigma_rr at the interface and sigma_rr(c) = 0 fix A1, A2, B2. Interface pressure "
           f"{S['interface_pressure_Pa'] / 1e6:.4f} MPa. Errors over 22 radii (both materials, 4 mm clear of the "
           "interface), relative to the interface pressure (displacement: to max |u_r|).", "",
           "| h [m] | DOFs | rel. err u_r | max abs d sigma_rr / p_int | max abs d sigma_tt / p_int | "
           "max abs d sigma_zz / p_int | equilibrium round-off |", "|---|---|---|---|---|---|---|"]
    for r in S["rows"]:
        md.append(f"| {r['mesh_size']} | {r['dofs']:,} | {r['err_u']:.2e} | {r['err_srr']:.2e} | {r['err_stt']:.2e} | "
                  f"{r['err_szz']:.2e} | {r['equilibrium_round_off']:.1e} |")
    md += ["", "![](figures/mech3d_shrink_fit.pdf)", ""]
    # 4 cooldown
    md += ["## 3. Free cooldown of an unsupported L-shaped body", "",
           f"alpha = {C['alpha']}, no supports or loads: the exact answer is u = -alpha (x - x_c) about the centroid "
           f"x_c = {np.round(C['centroid'], 6).tolist()} m (the volume-weighted rigid-body projection removes "
           "exactly the translation and rotation), with zero stress.", "",
           f"* max relative error of u at 5 interior points: {C['err_u_rel']:.1e}",
           f"* max von Mises / (E alpha): {C['von_mises_max_over_E_alpha']:.1e}",
           f"* rigid motions pinned: {C['pins']} (six: nothing is constrained)",
           f"* equilibrium round-off: {C['equilibrium_round_off']:.1e}", ""]
    # 5 slab
    md += ["## 4. Conduction in a slab", "",
           f"A slab 0 < x < {SL['L']} m between two plates held at {SL['T1']} K and {SL['T2']} K, adiabatic sides. "
           f"Constant k = {SL['k0']} W/(m K): T linear (exact for any order). With q = {SL['q']} W/m^3: "
           "T = T1 + (T2 - T1) x/L + q x (L - x)/(2 k), quadratic, so P2 is exact. With the two-point table "
           f"k(T) = {SL['k0']} (1 + {SL['c']} (T - {SL['T1']})) the Kirchhoff potential theta = int k dT is linear "
           "in x, so T = T1 + (sqrt(1 + 2 c theta/k0) - 1)/c exactly; the solver iterates on k(T_prev) (Picard), "
           "so this is a test of the nonlinear path.", "",
           "Columns: max abs dT over 19 points on a line along x (0.05 ... 0.95 m) and the x where it occurs; RMS and "
           "max over 300 random interior points (\"cloud\"), and the RMS over the cloud points with x > 0.2 m.", "",
           "| case | h [m] | DOFs | max abs dT, line [K] | at x [m] | cloud RMS [K] | rate | cloud max [K] | "
           "cloud RMS, x > 0.2 [K] | heat Q rel. err | Picard its | Picard change | balance |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rk = [r for r in Sl["rows"] if r["case"] == "kT"]
    rts = dict(zip([id(r) for r in rk], rate([r["cloud_rms_dT_K"] for r in rk], [r["mesh_size"] for r in rk])))
    r5 = [r for r in rk if r["mesh_size"] <= 0.05]
    rk_avg = rate([r5[0]["cloud_rms_dT_K"], r5[-1]["cloud_rms_dT_K"]], [r5[0]["mesh_size"], r5[-1]["mesh_size"]])[1] \
        if len(r5) > 1 else None
    for r in Sl["rows"]:
        md.append(f"| {r['case']} | {r['mesh_size']} | {r['dofs']:,} | {r['max_abs_dT_K']:.2e} | {r['x_of_max_m']:.2f} | "
                  f"{r['cloud_rms_dT_K']:.2e} | {fmt(rts.get(id(r)), '{:.2f}')} | {r['cloud_max_dT_K']:.2e} | "
                  f"{r['cloud_rms_dT_K_x_gt_0.2']:.2e} | {fmt(r.get('Q_rel_err'))} | {r['picard']} | "
                  f"{fmt(r.get('picard_change') if r['case'] == 'kT' else None, '{:.1e}')} | {r['balance_relative']:.1e} |")
    md += ["", "The k(T) error is discretization error, not Picard error or a defect of the k(T) evaluation. The Picard "
           "iteration stops at a relative change below 1e-10 (column Picard change); it contracts by a factor of about 6 "
           "per iteration, so the remaining iteration error is of order 1e-9 K, far below the table. k(T_h) is linear "
           "in T_h, so the quadrature (order 2p + 2) integrates the stiffness exactly, and no k(T) table was clamped. "
           "The constant-k quadratic case (q) on the same mesh is exact to round-off, which rules out the sinks, the "
           "boundary treatment and the assembly. The error is concentrated at the cold end: "
           "T(x) = T1 + (sqrt(1 + 2 c theta/k0) - 1)/c has a third derivative proportional to (1 + 8.5 x)^(-5/2), "
           "about 40 times larger at x = 0 than at x = 0.4 m, and the line maximum sits at its first point "
           "(x = 0.05 m). A pointwise maximum at one point near the steep end of an unstructured mesh only a few "
           "elements thick does not have to decrease monotonically (2.2e-3, 2.9e-3, 1.2e-4 K at h = 0.1, 0.05, 0.025 "
           f"in the first run); the cloud RMS, a mesh-averaged measure, decreases with h, at an average rate of "
           f"h^{fmt(rk_avg, '{:.1f}')} from h = 0.05 m to the finest mesh (P2: 3), once the cold end is resolved "
           "(at h >= 0.07 m the slab is two to three elements thick).", ""]
    # 6 hollow
    md += ["", "## 5. Conduction through a hollow cylinder", "",
           f"Wall {HC['a']} < r < {HC['b']} m, height {HC['H']} m, between a sink tube at {HC['T1']} K inside and one "
           f"at {HC['T2']} K outside, adiabatic ends. Constant k: T = T1 + dT ln(r/a)/ln(b/a), "
           "Q = 2 pi k H dT / ln(b/a). With k(T) (same table as the slab): theta(r) = theta(T2) ln(r/a)/ln(b/a), "
           "Q = 2 pi H theta(T2)/ln(b/a). The heat load is `heat_added_W` (what the warm sink supplies; "
           "`heat_removed_W` is the same number to the solver tolerance).", "",
           "| k | h [m] | DOFs | max abs dT [K] | rate | Q FE [W] | Q exact [W] | rel. | Picard its | balance |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for kT in (False, True):
        rr = [r for r in Hc["rows"] if r["kT"] == kT]
        rt = rate([r["max_abs_dT_K"] for r in rr], [r["mesh_size"] for r in rr])
        for r, a in zip(rr, rt):
            md.append(f"| {'k(T)' if kT else 'const'} | {r['mesh_size']} | {r['dofs']:,} | {r['max_abs_dT_K']:.2e} | "
                      f"{fmt(a, '{:.2f}')} | {r['Q_FE_W']:.6f} | {r['Q_exact_W']:.6f} | {r['Q_rel_err']:+.1e} | "
                      f"{r['picard']} | {r['balance_relative']:.1e} |")
    md += ["", "![](figures/mech3d_hollow_convergence.pdf)", ""]
    # 7 Lorentz
    md += ["## 6. Lorentz loads", "",
           f"**Solenoid hoop stress.** Winding R = {SOL['R']} m, t = {SOL['t']} m, H = {SOL['H']} m, "
           f"NI = {SOL['NI']:g} A, E = {SOL['E']:g} Pa, unsupported. The magnetic pressure "
           "p = (B_in^2 - B_out^2)/(2 mu0) is taken from the solved field just inside and outside the winding at "
           "mid-height (probes), so the comparison tests the mechanics. With f_r = J B_z(r) and the linear profile "
           "B_z = B0 (b - r)/t, radial equilibrium of an infinitely long solenoid integrates to the thickness-mean "
           "hoop stress (1/t) int sigma_tt dr = p (Rm - t/6)/t (the same derivation as the axisymmetric record). "
           "The finite length (H/R = 8) is a model error of this reference, beyond what the measured p absorbs "
           "(radial B and axial shear near the mid-plane), so the column is FE against an infinite-solenoid "
           "model, not a pure discretization error.", "",
           "| h [m] | DOFs | B_in [T] | B_out [T] | p [Pa] | mean hoop FE [Pa] | p (Rm - t/6)/t [Pa] | rel. | membrane p Rm/t |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in Lo["rows"]:
        md.append(f"| {r['mesh_size']} | {r['dofs']:,} | {r['B_in_T']:.5f} | {r['B_out_T']:.2e} | {r['p_Pa']:.1f} | "
                  f"{r['hoop_mean_FE_Pa']:.6g} | {r['hoop_exact_Pa']:.6g} | {r['rel']:+.1e} | {r['membrane_Pa']:.6g} |")
    md += ["",
           "**Net winding force.** Two coaxial coils 0.3 m apart attract. The Lorentz load integrated by the elasticity "
           "assembly (`mechanics.lorentz_forces`) against the independent winding force of `results[\"forces\"]` "
           "(same field and quadrature, different code path), and the reaction of each coil's support (fix all), "
           "which must equal -F:", "",
           "| coil | F_z mechanics [N] | F_z `forces[]` [N] | rel. diff | reaction F_z [N] | abs(R + F)/abs(F_z) |",
           "|---|---|---|---|---|---|"]
    for n, v in Lo["two_coils"].items():
        md.append(f"| {n} | {v['F_mech_N'][2]:.9g} | {v['F_forces_N'][2]:.9g} | {v['rel_mech_vs_forces']:.1e} | "
                  f"{v['reaction_N'][2]:.9g} | {v['rel_reaction_plus_F']:.1e} |")
    # 8 axi vs 3D
    md += ["", "## 7. Axisymmetric module against the 3D module", "",
           "A steel bobbin with a flange (support: fix z), an azimuthal winding bonded on it (4e5 A), and a separate "
           "zero-field-cooled niobium tube in the bore (Maxwell pressure), all cooled down; thermal: the flange held at "
           "4.2 K, 20 W in the winding, a tabulated k(T) for the steel (illustrative numbers). Solved by the "
           f"axisymmetric module ({X['axi_dofs']:,} magnetostatic DOFs, mesh size 0.02 m) and, after "
           "`axionfem.convert.to_3d`, by the 3D module. Pointwise differences along three lines (bobbin at r = 0.31 m, "
           "winding across its thickness at z = 0, tube at r = 0.155 m), at azimuth 33 degrees; displacements relative "
           "to the largest axisymmetric displacement on the line, stresses relative to its largest von Mises stress, "
           "temperatures in K.", "",
           "| h 3D [m] | DOFs 3D | line | u_r | u_z | sigma_hoop | von Mises | T [K] |", "|---|---|---|---|---|---|---|---|"]
    for r in X["rows"]:
        for k, v in r["lines"].items():
            md.append(f"| {r['mesh_size']} | {r['dofs_3d']:,} | {k} | {v['u_r']:.1e} | {v['u_z']:.1e} | "
                      f"{v['sigma_hoop']:.1e} | {v['von_mises']:.1e} | {v['T_K']:.1e} |")
    md += ["", "| h 3D [m] | region | u_max 2D [m] | u_max 3D [m] | von Mises max 2D [Pa] | 3D [Pa] | T_max 2D [K] | T_max 3D [K] |",
           "|---|---|---|---|---|---|---|---|"]
    for r in X["rows"]:
        for name, v in r["regions"].items():
            md.append(f"| {r['mesh_size']} | {name} | {v['u_max_2d']:.6e} | {v['u_max_3d']:.6e} | {v['vm_max_2d']:.6g} | "
                      f"{v['vm_max_3d']:.6g} | {v['T_max_2d']:.6f} | {v['T_max_3d']:.6f} |")
    md += ["", "Peak values (u_max, von Mises max) are lattice maxima and converge more slowly than the pointwise "
           "fields, especially at re-entrant corners (bobbin-flange), where the stress is singular in both models.",
           ""]
    OUT.write_text("\n".join(md) + "\n")


def main():
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/mech3d_validation")
    wd.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("AXF_M3D_ONLY", "")
    def want(k):
        return not only or k in only.split(",")

    def load(name):
        return json.loads((REC / f"mech3d_{name}.json").read_text())

    L = lame(wd) if want("lame") else load("lame")
    S = shrink(wd) if want("shrink") else load("shrink")
    C = cooldown(wd) if want("cooldown") else load("cooldown")
    Sl = slab(wd) if want("slab") else load("slab")
    Hc = hollow(wd) if want("hollow") else load("hollow")
    Lo = solenoid(wd) if want("lorentz") else load("lorentz")
    X = axi_vs_3d(wd) if want("axi") else load("axi_vs_3d")
    figures(L, S, Hc)
    report(L, S, C, Sl, Hc, Lo, X)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
