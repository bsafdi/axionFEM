"""Slater pump-signal mode mixing of the SRF cell by out-of-round wall error (beta; docs/srf/vibrations.md).

Pump-signal coupling by a wall displacement u_n (first-order Slater perturbation theory) and the resulting
vibration requirement,

    g = (w/4U~) oint dA u_n (B0.B1/mu0 - eps0 E0.E1) = (w/2)(u/L_c) eta_mix,     L_c = V^(1/3),
    S_u^(1/2) <= (2 sqrt2 L_c/eta_mix) (eps n_i hbar/(U_0 Q_i))^(1/2),
    eta_mix = c_eff delta_s/R_eq  (delta_s: rms out-of-roundness per octave at k_el).

For any body of revolution the TM0 pump and TE0 signal have B0.B1 = E0.E1 = 0 pointwise, so g = 0 at first order
in the wall displacement; the coupling is bilinear in a static out-of-round error and the vibration.  Two numbers:

1. ``ceff`` (2D): c_eff for a *random* figure error driven by a *diffuse* displacement field,
       c_eff = (ln 2)^(-1/2) R_eq L_c sqrt(pi I_4),   I_4 = oint (B0t B1t)^2 dA,
   both modes normalized to int |B|^2 dV = 1, evaluated with the 2D wall fields of axionfem.srf (Slater-type
   perturbation theory from the unperturbed fields; no deformed solve is needed for it).
2. ``m2`` (3D, MPI): a *coherent* m = 2 deformation.  The 3D mesh of the cell is mapped affinely,
   x -> (1 + e) x, y -> (1 - e) y (an elliptical out-of-round cell, wall displacement e r cos 2phi, e = delta/R_eq at
   the equator; with --twist the ellipse axes rotate along z, a chiral control), and re-solved with field probes.
   Each eigenmode's E is projected (after a phi average: the m = 0 part) on the 2D TM020 and TE011 fields; theta is the
   TE content of the TM-like mode (and vice versa), and |W| = Delta omega_pair sin theta cos theta.  If W = w e^2 omega,
   a same-pattern vibration u = de R_eq on a static e gives g = 2 w e de omega, eta_mix = 4 w e L_c/R_eq and
   c_m2 = 4 w L_c/R_eq.  The same mesh is used for every e (only node coordinates move).  Result (docs/srf/vibrations.md):
   the plain ellipse does not mix the pair at all (mirror selection rule: TM0-like modes are even, TE0-like modes odd
   under every axial mirror plane the deformed cell keeps, C2v here); the twisted ellipse does.  The rule covers a
   static error plus a vibration that keep a common axial mirror plane (e.g. the same (r, z) profile); a generic
   vibration pattern (different (r, z) profile, chiral sum with a z-odd part) couples the pair at O(e de omega), as the
   twisted control shows.

    python benchmarks/srf/srf_vibration_mixing.py m2 --n 12 --order 3 --ranks 16 --eps 0.01 0.02 0.04   (MPI)
    python benchmarks/srf/srf_vibration_mixing.py m2 --n 8 --order 2 --ranks 16 --eps 0.02 0.04 --twist 0.7854
    python benchmarks/srf/srf_vibration_mixing.py ceff
    python benchmarks/srf/srf_vibration_mixing.py report      # requirement with the model-A benchmarks

Writes benchmarks/srf/results/srf_vibration.json (keys ceff, m2, report).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from axionfem import srf  # noqa: E402

OUT = ROOT / "benchmarks/srf/results/srf_vibration.json"
REQ = srf.PROJECTX_650_B09["equator_radius"]
MU0 = srf.MU0
HBAR = 1.054571817e-34
KB = 1.380649e-23
EPS = 0.1          # eps in the requirement on S_u^(1/2) (module docstring)


def _load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _store(key, val):
    d = _load()
    d[key] = val
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1, default=float) + "\n")


# ------------------------------------------------------------------------------------------- 1. c_eff (2D)
def _wall_points(wall, n_per_seg=4):
    W = np.asarray(wall, float)
    P, T, L = [], [], []
    for a, b in zip(W[:-1], W[1:]):
        ln = float(np.hypot(*(b - a)))
        if ln <= 0:
            continue
        t = (b - a) / ln
        for u in (np.arange(n_per_seg) + 0.5) / n_per_seg:
            P.append(a + u * (b - a))
            T.append(t)
            L.append(ln / n_per_seg)
    return np.array(P), np.array(T), np.array(L)


def _tangential_B(res, row, P, T, fam):
    """|B_t| just inside the wall: per point, the first of the offsets 1e-6, 1e-5, 1e-4 R_eq along +-normal that
    lands inside the mesh (the polygon's convex corners at the iris need the larger offsets)."""
    N = np.column_stack([-T[:, 1], T[:, 0]])
    out = np.full(len(P), np.nan)
    for off in (1e-6, 1e-5, 1e-4):
        for sg in (1, -1):
            todo = ~np.isfinite(out)
            if not todo.any():
                break
            Q = P[todo] + sg * off * REQ * N[todo]
            s = res.cavity_mode_sample(row, Q[:, 0], Q[:, 1])
            if fam == "TM":
                v = MU0 * np.abs(s["H_phi"])
            else:
                v = MU0 * np.abs(s["H_r"] * T[todo, 0] + s["H_z"] * T[todo, 1])
            out[np.flatnonzero(todo)] = v
    bad = ~np.isfinite(out)
    if bad.mean() > 2e-3:
        raise RuntimeError(f"wall sampling: {int(bad.sum())} of {len(out)} points outside the mesh")
    if bad.any():   # a few corner points (tube/cap junctions): take the neighbors' values along the wall
        i = np.arange(len(out))
        out[bad] = np.interp(i[bad], i[~bad], out[~bad])
    return out


def ceff_one(n, order, work):
    from axionfem.srf import _tri_product_integral
    # the faceted wall (as for the recorded c_eff): the wall is sampled along the polygon, 1e-6 R_eq inside
    d = srf.elliptical_cell_design(axial_factor=0.86, h=REQ / n, order=order, modes=6, angle_tol=0.4 / n,
                                   wall="polygon")
    r = d.solve(f"{work}/ceff_o{order}_n{n}", verbose=0)
    con = d.srf_contour
    lab = srf.classify_modes(r, con)
    pr, sr = srf.find_mode(r, "TM020", labels=lab), srf.find_mode(r, "TE011", labels=lab)
    V = r.fullwave["cavity"]["volume_m3"]
    norms = {}
    for nm, row in (("pump", pr), ("signal", sr)):
        F = r.cavity_mode_field(row, subdiv=4)
        norms[nm] = sum(_tri_product_integral(F["r"], F["z"], F["tris"], MU0 * F[k], MU0 * F[k])
                        for k in ("H_r", "H_phi", "H_z"))   # int |B|^2 dV at the solver normalization
    P, T, L = _wall_points(con["wall"])
    B0 = _tangential_B(r, pr, P, T, "TM") / math.sqrt(norms["pump"])
    B1 = _tangential_B(r, sr, P, T, "TE") / math.sqrt(norms["signal"])
    dA = 2 * math.pi * P[:, 0] * L
    q = (B0 * B1) ** 2 * dA
    cell = np.abs(P[:, 1]) <= con["z_end"] - con["params"]["tube_length"] + 1e-12
    Lc = V ** (1 / 3)
    I4, I4_cell = float(q.sum()), float(q[cell].sum())
    per_oct = 1 / math.sqrt(math.log(2))
    c = per_oct * REQ * Lc * math.sqrt(math.pi * I4)
    return {"h_over_Req": 1 / n, "order": order, "V_m3": V, "L_c_m": Lc, "R_eq_m": REQ, "A_wall_m2": float(dA.sum()),
            "A_contour_m2": con["A_polygon_m2"], "A_exact_m2": con["A_m2"], "I4_m-4": I4, "I4_cell_only_m-4": I4_cell, "per_octave_factor": per_oct,
            "c_eff": c, "c_eff_per_unit_lnk": c / per_oct,
            "c_eff_cell_only": per_oct * REQ * Lc * math.sqrt(math.pi * I4_cell),
            # G-factor-average closed form of the reference (B_t^2 -> its wall average): sqrt(pi A <B0^2><B1^2>)
            "c_eff_G_average": per_oct * REQ * Lc * math.sqrt(
                math.pi * float((B0 ** 2 * dA).sum()) * float((B1 ** 2 * dA).sum()) / float(dA.sum())),
            "f_pump_Hz": pr["f_Hz"], "f_signal_Hz": sr["f_Hz"]}


def stage_ceff(a):
    with tempfile.TemporaryDirectory(prefix="srf_vib_") as work:
        rows = [ceff_one(n, 3, work) for n in (20, 40, 80)]
    for x in rows:
        print({k: x[k] for k in ("h_over_Req", "c_eff", "c_eff_cell_only", "c_eff_G_average", "I4_m-4")}, flush=True)
    _store("ceff", {"method": "c_eff = (ln 2)^(-1/2) R_eq L_c sqrt(pi I_4) with axionFEM 2D wall fields "
                              "(order 3, angle_tol 0.4/n, wall sampled 1e-6 R_eq inside, 4 points per segment)",
                    "rows": rows, "c_eff": rows[-1]["c_eff"]})


# --------------------------------------------------------------------------------------------- 2. m = 2 (3D)
def _affine_msh(src, dst, e, twist=0.0, z_end=1.0):
    """Copy a gmsh MSH 2.2 file (ASCII or binary, as axf mesh writes it) with node coordinates
    x -> (1 + e) x, y -> (1 - e) y; with ``twist`` the ellipse axes are rotated by twist * z/z_end [rad]
    (R(a) diag(1+e, 1-e) R(-a), a chiral deformation with no mirror plane through the axis)."""

    def amap(xyz):
        x, y, z = xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2]
        a = twist * z / z_end
        c, s_ = np.cos(a), np.sin(a)
        u, v = c * x + s_ * y, -s_ * x + c * y
        u, v = u * (1 + e), v * (1 - e)
        xyz[:, 0], xyz[:, 1] = c * u - s_ * v, s_ * u + c * v
        return xyz
    raw = Path(src).read_bytes()
    head = raw[:raw.index(b"$EndMeshFormat")]
    binary = head.split(b"\n")[1].split()[1] == b"1"
    i0 = raw.index(b"$Nodes\n") + len(b"$Nodes\n")
    i1 = raw.index(b"\n", i0)
    n = int(raw[i0:i1])
    if binary:
        dt = np.dtype([("id", "<i4"), ("xyz", "<f8", 3)])
        start = i1 + 1
        nodes = np.frombuffer(raw, dtype=dt, count=n, offset=start).copy()
        nodes["xyz"] = amap(nodes["xyz"])
        out = raw[:start] + nodes.tobytes() + raw[start + n * dt.itemsize:]
    else:
        j = raw.index(b"$EndNodes", i1)
        lines = raw[i1 + 1:j].decode().split("\n")
        body = []
        for ln in lines:
            if not ln.strip():
                continue
            p = ln.split()
            q = amap(np.array([[float(p[1]), float(p[2]), float(p[3])]]))[0]
            body.append(f"{p[0]} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g}")
        out = raw[:i1 + 1] + ("\n".join(body) + "\n").encode() + raw[j:]
    Path(dst).write_bytes(out)


def _map_points(xyz, e, twist, z_end):
    """The node map of :func:`_affine_msh` applied to points."""
    xyz = np.array(xyz, float)
    a = twist * xyz[:, 2] / z_end
    c, s_ = np.cos(a), np.sin(a)
    u, v = c * xyz[:, 0] + s_ * xyz[:, 1], -s_ * xyz[:, 0] + c * xyz[:, 1]
    u, v = u * (1 + e), v * (1 - e)
    xyz[:, 0], xyz[:, 1] = c * u - s_ * v, s_ * u + c * v
    return xyz


def _meridional_grid(con, n_r=28, n_phi=8):
    """Probe set: a uniform (r, z) grid inside the wall polygon, at least 0.25 grid steps from it (the faceted 3D
    wall differs from the polygon), times n_phi azimuths; weights r dr dz (the m = 0 projection averages over phi)."""
    from matplotlib.path import Path as MPath
    W = np.asarray(con["points"], float)
    dr = REQ / n_r
    rr = np.arange(0.5, n_r) * dr
    zz = np.arange(-con["z_end"] + 0.5 * dr, con["z_end"], dr)
    R, Z = np.meshgrid(rr, zz)
    pts = np.column_stack([R.ravel(), Z.ravel()])
    poly = np.vstack([W, [[0.0, W[-1, 1]], [0.0, W[0, 1]]]])
    pts = pts[MPath(poly).contains_points(pts)]
    d2 = np.full(len(pts), np.inf)
    for p0, p1 in zip(W[:-1], W[1:]):
        t = np.clip(((pts - p0) @ (p1 - p0)) / max(float((p1 - p0) @ (p1 - p0)), 1e-300), 0, 1)
        d2 = np.minimum(d2, ((pts - (p0 + t[:, None] * (p1 - p0))) ** 2).sum(1))
    pts = pts[d2 > (0.25 * dr) ** 2]
    phis = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
    return pts, phis, pts[:, 0] * dr * dr


def _m0_projection(E, pts, phis, w, ref_TM, ref_TE, valid):
    """m = 0 parts of a 3D mode's E (probes ordered point-major, phi-minor) projected on the 2D TM020 (E_r, E_z) and
    TE011 (E_phi) fields: (c_TM, c_TE), each a direction cosine (reference norm and the mode's own norm on the same
    quadrature), and the number of meridional points with all azimuths valid."""
    npnt, nph = len(pts), len(phis)
    E = np.nan_to_num(np.asarray(E, float)).reshape(npnt, nph, 3)
    ok = np.asarray(valid, bool).reshape(npnt, nph).all(1)
    c, s_ = np.cos(phis), np.sin(phis)
    Er = (E[:, :, 0] * c + E[:, :, 1] * s_).mean(1)
    Ep = (-E[:, :, 0] * s_ + E[:, :, 1] * c).mean(1)
    Ez = E[:, :, 2].mean(1)
    wt = w * ok
    nTM = math.sqrt(float(np.sum(wt * (ref_TM[0] ** 2 + ref_TM[1] ** 2))))
    nTE = math.sqrt(float(np.sum(wt * ref_TE ** 2)))
    nE = math.sqrt(float(np.sum(wt[:, None] * (E ** 2).sum(2)) / nph))
    cTM = float(np.sum(wt * (Er * ref_TM[0] + Ez * ref_TM[1]))) / nTM / nE
    cTE = float(np.sum(wt * Ep * ref_TE)) / nTE / nE
    return cTM, cTE, int(ok.sum())


def stage_m2(a):
    work = Path(a.workdir or tempfile.mkdtemp(prefix="srf_m2_"))
    work.mkdir(parents=True, exist_ok=True)
    # 2D reference fields (TM020, TE011) on the probe grid
    d2 = srf.elliptical_cell_design(axial_factor=0.86, h=REQ / 40, order=3, modes=6, wall="polygon")
    r2 = d2.solve(work / "ref2d", verbose=0)
    con = d2.srf_contour
    lab = srf.classify_modes(r2, con)
    pr, sr = srf.find_mode(r2, "TM020", labels=lab), srf.find_mode(r2, "TE011", labels=lab)
    pts, phis, w = _meridional_grid(con)
    sp = r2.cavity_mode_sample(pr, pts[:, 0], pts[:, 1])
    ss = r2.cavity_mode_sample(sr, pts[:, 0], pts[:, 1])
    ref_TM = (np.nan_to_num(sp["E_r"]), np.nan_to_num(sp["E_z"]))
    ref_TE = np.nan_to_num(ss["E_phi"])
    f_ref = 0.5 * (pr["f_Hz"] + sr["f_Hz"])
    d = srf.elliptical_cell_design_3d(axial_factor=0.86, h=REQ / a.n, order=a.order, modes=a.modes,
                                      target_f_Hz=f_ref, wall="polygon")  # as recorded (vibrations.md)
    mdir = work / "mesh"
    mdir.mkdir(parents=True, exist_ok=True)
    d.save(mdir / "design.json")
    axf = os.environ.get("AXF_EXE", "axf")
    subprocess.run([axf, "mesh", str(mdir / "design.json"), "-o", str(mdir)], check=True)
    msh = mdir / "mesh3d.msh"
    if not msh.exists():
        raise FileNotFoundError(f"{msh} not written by axf mesh; files: {sorted(p.name for p in mdir.iterdir())}")
    info = mdir / "mesh3d_info.json"
    rows = []
    for e in [0.0] + list(a.eps):
        m = work / f"e{e:+.4f}.msh"
        _affine_msh(msh, m, e, a.twist, con["z_end"])
        shutil.copy(info, work / f"e{e:+.4f}_info.json")
        dd = srf.elliptical_cell_design_3d(axial_factor=0.86, h=REQ / a.n, order=a.order, modes=a.modes,
                                           target_f_Hz=f_ref, wall="polygon")
        dd.doc["mesh"]["file"] = str(m)
        dd.doc["mesh"]["info_file"] = str(work / f"e{e:+.4f}_info.json")
        dd.doc["mesh"].pop("domain_solid", None)
        pq = np.array([[rr * math.cos(ph), rr * math.sin(ph), zz] for rr, zz in pts for ph in phis])
        dd.doc["fullwave"]["field_probes"] = _map_points(pq, e, a.twist, con["z_end"]).tolist()
        r = dd.solve(work / f"run_e{e:+.4f}", verbose=0, mpi_ranks=a.ranks)
        fw = r.fullwave
        modes = fw["eigenmodes"]["modes"]
        valid = fw["eigenmodes"]["field_probes"]["valid"]
        near = sorted(range(len(modes)), key=lambda i: abs(modes[i]["f_Hz"] - f_ref))[:2]
        proj = {i: _m0_projection(modes[i]["probes"]["E"], pts, phis, w, ref_TM, ref_TE, valid) for i in near}
        ia, ib = sorted(near, key=lambda i: -abs(proj[i][0]))          # TM-like first
        (aTM, aTE, nok), (bTM, bTE, _) = proj[ia], proj[ib]
        th_a = math.atan2(abs(aTE), abs(aTM))      # TE content of the TM-like mode
        th_b = math.atan2(abs(bTM), abs(bTE))      # TM content of the TE-like mode
        fa, fb = modes[ia]["f_Hz"], modes[ib]["f_Hz"]
        dw = 2 * math.pi * abs(fb - fa)
        wbar = math.pi * (fa + fb)
        Wa = dw * abs(math.sin(th_a) * math.cos(th_a)) / wbar
        Wb = dw * abs(math.sin(th_b) * math.cos(th_b)) / wbar
        row = {"e": e, "f_TM_like_Hz": fa, "f_TE_like_Hz": fb, "proj_TM_like": [aTM, aTE], "proj_TE_like": [bTM, bTE],
               "theta_TM_like": th_a, "theta_TE_like": th_b, "W_over_omega": Wa, "W_over_omega_from_TE_like": Wb,
               "w": (0.5 * (Wa + Wb) / e ** 2) if e else None,
               "Ephi_frac": [modes[ia].get("E_azimuthal_fraction"), modes[ib].get("E_azimuthal_fraction")],
               "V_m3": fw["cavity"]["volume_m3"], "n_points_valid": nok, "n_points": len(pts),
               "residuals": [modes[i].get("residual_true") for i in (ia, ib)],
               "spectrum_Hz": [m_["f_Hz"] for m_ in modes]}
        rows.append(row)
        print({k: v for k, v in row.items() if k != "spectrum_Hz"}, flush=True)
    Lc = rows[0]["V_m3"] ** (1 / 3)
    ws = [x["w"] for x in rows if x["w"]]
    out = {"method": "3D affine m = 2 deformation of one mesh (x(1+e), y(1-e)); W = (pair splitting) x "
                     "sin(theta) cos(theta), theta from the m = 0 projection of each eigenmode's E on the 2D TM020 / "
                     "TE011 fields (probes on a meridional grid x 8 azimuths); c_m2 = 4 w L_c/R_eq",
           "h_over_Req": 1 / a.n, "order": a.order, "rows": rows, "w_values": ws,
           "L_c_m": Lc, "R_eq_m": REQ, "c_m2_all": [4 * w * Lc / REQ for w in ws]}
    old = _load().get("m2", {})
    out["twist_rad"] = a.twist
    old[f"n{a.n}_o{a.order}" + (f"_twist{a.twist:g}" if a.twist else "")] = out
    _store("m2", old)
    print("c_m2:", out["c_m2_all"])


# ------------------------------------------------------------------------------------------- 3. requirement
def stage_report(a):
    """The vibration requirement S_u^(1/2) at eta_mix = 1 for the model-A benchmarks of benchmarks/srf/results/model_a_geometry.json (the
    computed geometry presets)."""
    mag = json.loads((ROOT / "benchmarks/srf/results/model_a_geometry.json").read_text())
    res = mag["results"]
    from axionfem import heterodyne as H
    rec = H._load_record(str(H.PROJECTX_RECORD))
    V_cell = rec["figures_of_merit"]["V_m3"]       # the unit cell (R_eq = REQ) the model-A volumes scale
    if mag.get("record", {}).get("fingerprint") != rec.get("fingerprint"):
        raise RuntimeError("model_a_geometry.json was computed from another geometry record; rerun "
                           "benchmarks/srf/model_a_geometry.py first")
    d = _load()
    c = d.get("ceff", {}).get("c_eff")
    out = {}
    for key, per in res.items():
        sc = H.SCENARIOS_A[key]
        for geo, stats in per.items():
            for stat, r in stats.items():
                V, f0, U0, Qi = r["V_m3"], r["f0_Hz"], r["U0_J"], r["Q_i"]
                n_i = float(H.occupation(f0, sc.T_K))
                Lc = V ** (1 / 3)
                req = 2 * math.sqrt(2) * Lc * math.sqrt(EPS * n_i * HBAR / (U0 * Qi))
                R_eq = REQ * (V / V_cell) ** (1 / 3)
                out[f"{key}/{geo}/{stat}"] = {"V_m3": V, "f0_Hz": f0, "L_c_m": Lc, "R_eq_m": R_eq, "n_i": n_i,
                                              "S_u_req_eta1_m_rtHz": req,
                                              "S_u_req_at_delta_1um_m_rtHz": (req / (c * 1e-6 / R_eq)) if c else None}
    _store("report", {"eps": EPS, "c_eff_used": c, "V_cell_m3": V_cell,
                      "record_fingerprint": rec.get("fingerprint"), "rows": out})
    for k, v in out.items():
        print(k, f"{v['S_u_req_eta1_m_rtHz']:.3g}", f"{v['S_u_req_at_delta_1um_m_rtHz']:.3g}" if c else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ceff", "m2", "report"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--twist", type=float, default=0.0, help="m2: ellipse-axis rotation at z_end [rad] (chiral control)")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    {"ceff": stage_ceff, "m2": stage_m2, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
