"""External check 4 (beta): first-order wall-deformation perturbation theory vs direct 3D solves of a deformed
closed cylinder (the general 3D path; the cylinder wall does not use the elliptical-cell mesher).

Cavity: PEC cylinder R = 0.1 m, d = 2.354884 R (TE012/TM013 degenerate, Lasenby 1912.11056). One 3D mesh, mapped
x -> (1+e) x, y -> (1-e) y (benchmarks/srf/srf_vibration_mixing._affine_msh): barrel displacement e R cos 2phi to
first order, flattening f = (a-b)/a = 2e/(1+e). Each e is re-solved with E-field probes on rings (r, z, 16 azimuths).

Compared with first-order (Slater) perturbation theory, derived here and checked against an independent derivation
(tests/test_srf_external.py):
 (a) the splitting of the m = 1 doublets (degenerate in the round cylinder):
     TM_11p:  |dw|/w = e (x11/kR)^2,                     x11 = 3.8317 (J1 zero)
     TE_11p:  |dw|/w = e (x'11/kR)^2 (x'11^2+1)/(x'11^2-1), x'11 = 1.8412 (J1' zero)
     for TM111, TE113, TM112;
 (b) the deformed TE012 mode's barrel E_r (zero in the round cylinder): the m = 2 TE field with the same k_c and
     beta, fixed by E'_phi(R) = -delta dE_phi/dr(R), E'_z(R) = 0, has |E'_r(R)| = |E'_phi(R)|, so
         E'_r(R, phi, z) = e R A k_c J0(x01') sin 2phi sin(beta z)   (E_phi0 = A J1(k_c r) sin(beta z), x01' = 3.8317)
     i.e. |E'_r(R)| / (c B_barrel,max) = e kR = (f/2) kR. Inside: E'_r(r) = E'_r(R) (R/r) J2(k_c r)/J2(k_c R).
     Lasenby Eq. 23 quotes E_r ~ 40 kV/m (f/1e-3) sin 2phi sin(2 pi z/d) at max wall |B| = 0.2 T.
 (c) the m = 0 pair TE012/TM013: no first-order shift (the m = 0 part of cos 2phi vanishes) and no mixing at any
     order (both axial mirror planes are kept: docs/srf/vibrations.md Sec. 2).

    python benchmarks/srf/external/slater_ellipse.py --n 10 --order 3 --ranks 16 --eps 0.0025 0.005 0.01 0.02  (MPI)
-> benchmarks/srf/results/external_slater.json
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks/srf"))
from axionfem import srf  # noqa: E402
from axionfem.cavity import C_LIGHT, cylinder_design_3d  # noqa: E402
from srf_vibration_mixing import _affine_msh, _map_points  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import jsonstore  # noqa: E402

OUT = ROOT / "benchmarks/srf/results/external_slater.json"
R = 0.1
LR = srf.cylinder_pair_geometry(("TE", 1, 2), ("TM", 1, 3))["L_over_R"]
D = LR * R
X11, XP11, XP01 = 3.831705970207512, 1.841183780931897, 3.831705970207512  # J1 zero, J1' zero, J0' (= -J1) zero
NPHI = 16


def kR(x, p):
    return math.hypot(x, p * math.pi / LR)


def predictions():
    """Slater coefficients c in |dw|/w = c e for the m = 1 doublets, and the TE012 barrel E_r coefficient."""
    out = {}
    for name, x, p, te in (("TM111", X11, 1, False), ("TM112", X11, 2, False), ("TE113", XP11, 3, True)):
        k = kR(x, p)
        c = (x / k) ** 2 * ((x * x + 1) / (x * x - 1) if te else 1.0)
        out[name] = {"kR": k, "f_Hz": C_LIGHT * k / R / (2 * math.pi), "c_split": c}
    k = kR(XP01, 2)
    beta = 2 * math.pi / LR
    from scipy.special import j0, j1
    rr = np.linspace(0, 1, 20001)
    cap_max = float((beta / k) * np.max(np.abs(j1(XP01 * rr))))   # c|B| on the caps / A  (B_r = (beta/w) A J1)
    bar_max = float((XP01 / k) * abs(j0(XP01)))                     # c|B| on the barrel / A (B_z = (k_c/w) A J0)
    Er_over_A = XP01 * abs(j0(XP01))                                # |E'_r(R)| / (e A)   (R = 1 units)
    out["TE012_Er"] = {"kR": k, "Er_over_cBbarrel_per_e": Er_over_A / bar_max, "Er_over_cBwallmax_per_e":
                       Er_over_A / max(cap_max, bar_max), "cB_cap_max_over_A": cap_max, "cB_barrel_max_over_A": bar_max,
                       "lasenby_Er_over_cBwallmax_per_f": 40e3 / (C_LIGHT * 0.2) / 1e-3}
    return out


def probe_set():
    rr = np.array([0.35, 0.6, 0.8, 0.9, 0.95])
    zz = (np.arange(24) + 0.5) / 24 * D - D / 2
    ph = np.arange(NPHI) * 2 * math.pi / NPHI
    pts = np.array([[r * R * math.cos(p), r * R * math.sin(p), z] for r in rr for z in zz for p in ph])
    return rr, zz, ph, pts


def decompose(E, xyz):
    """Cartesian E at physical points -> cylindrical (E_r, E_phi, E_z) with the physical azimuth."""
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([E[:, 0] * c + E[:, 1] * s, -E[:, 0] * s + E[:, 1] * c, E[:, 2]]), phi


def m_content(Ec, nr, nz):
    """fraction of sum |E|^2 per azimuthal order m = 0..NPHI/2 (DFT over the 16 azimuths, all components)."""
    A = Ec.reshape(nr, nz, NPHI, 3)
    F = np.fft.rfft(A, axis=2)
    w = (np.abs(F) ** 2).sum(axis=(0, 1, 3))
    w[1:-1] *= 2
    return w / w.sum()


def fit_te012(Ec, phi, xyz, nr, nz, rr):
    """A from the m = 0 E_phi (J1(k_c r) sin(beta(z + d/2))); then the sin2phi / cos2phi parts of E_r on each ring
    vs the first-order profile (R/r) J2(k_c r)/J2(k_c R) sin(beta(z + d/2))."""
    from scipy.special import j1, jv
    kc, beta = XP01 / R, 2 * math.pi / D
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]
    shape0 = j1(kc * r) * np.sin(beta * (z + D / 2))
    A = float(np.sum(Ec[:, 1] * shape0) / np.sum(shape0 ** 2))
    prof = (R / r) * jv(2, kc * r) / jv(2, kc * R) * np.sin(beta * (z + D / 2))
    s2, c2 = np.sin(2 * phi) * prof, np.cos(2 * phi) * prof
    M = np.column_stack([s2, c2])
    coef, *_ = np.linalg.lstsq(M, Ec[:, 0], rcond=None)
    resid = Ec[:, 0] - M @ coef
    # per-ring amplitudes (profile check)
    ring = []
    for i, rv in enumerate(rr):
        sl = slice(i * nz * NPHI, (i + 1) * nz * NPHI)
        cc, *_ = np.linalg.lstsq(M[sl], Ec[sl, 0], rcond=None)
        ring.append({"r_over_R": float(rv), "sin2phi_coef": float(cc[0]), "cos2phi_coef": float(cc[1])})
    return {"A": A, "Er_sin2phi_at_R": float(coef[0]), "Er_cos2phi_at_R": float(coef[1]),
            "Er_fit_rel_resid": float(np.linalg.norm(resid) / max(np.linalg.norm(Ec[:, 0]), 1e-300)),
            "Ephi_m0_rel_resid": float(np.linalg.norm(Ec[:, 1] - A * shape0) / np.linalg.norm(Ec[:, 1])), "rings": ring}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--modes", type=int, default=20)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02])
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    pred = predictions()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="slater_"))
    work.mkdir(parents=True, exist_ok=True)
    ft = C_LIGHT * 4.45 / R / (2 * math.pi)
    mk = lambda: cylinder_design_3d(R, D, h=R / a.n, order=a.order, modes=a.modes, target_f_Hz=ft,  # noqa: E731
                                    eigensolver={"method": "shift_invert"}, losses=False)
    d = mk()
    mdir = work / "mesh"
    mdir.mkdir(exist_ok=True)
    d.save(mdir / "design.json")
    subprocess.run([os.environ.get("AXF_EXE", "axf"), "mesh", str(mdir / "design.json"), "-o", str(mdir)], check=True)
    msh, info = mdir / "mesh3d.msh", mdir / "mesh3d_info.json"
    rr, zz, ph, pts = probe_set()
    nr, nz = len(rr), len(zz)
    rows = []
    for e in [0.0] + list(a.eps):
        m = work / f"e{e:+.4f}.msh"
        _affine_msh(msh, m, e, 0.0, 1.0)
        shutil.copy(info, work / f"e{e:+.4f}_info.json")
        dd = mk()
        dd.doc["mesh"]["file"] = str(m)
        dd.doc["mesh"]["info_file"] = str(work / f"e{e:+.4f}_info.json")
        dd.doc["mesh"].pop("domain_solid", None)
        xyz = _map_points(pts, e, 0.0, 1.0)
        dd.doc["fullwave"]["field_probes"] = xyz.tolist()
        r = dd.solve(work / f"run_e{e:+.4f}", verbose=0, mpi_ranks=a.ranks)
        modes = r.fullwave["eigenmodes"]["modes"]
        valid = np.asarray(r.fullwave["eigenmodes"]["field_probes"]["valid"], bool)
        info_m = []
        for i, md in enumerate(modes):
            if not valid.all():
                raise RuntimeError(f"{int((~valid).sum())} field probes outside the mesh (e = {e})")
            E = np.asarray(md["probes"]["E"], float)
            Ec, phi = decompose(E, xyz)
            mc = m_content(Ec, nr, nz)
            info_m.append({"i": i, "f_Hz": md["f_Hz"], "kR": 2 * math.pi * md["f_Hz"] * R / C_LIGHT,
                           "m_dominant": int(np.argmax(mc)), "m_content": mc[:5].tolist(),
                           # E_phi fraction over all azimuthal orders (key name kept for the records): for the TM013-like
                           # mode it grows as e^2 from its own deformation field, so it bounds the TE012 mixing
                           "E_phi_fraction_m0": float(np.sum(Ec[:, 1] ** 2) / np.sum(Ec ** 2)),
                           "residual": md.get("residual_true")})
        row = {"e": e, "f_flattening": 2 * e / (1 + e), "n_valid": int(valid.sum()), "n_probes": len(valid),
               "modes": info_m, "doublets": {}}
        for name in ("TM111", "TM112", "TE113"):
            f0 = pred[name]["f_Hz"]
            cand = sorted([x for x in info_m if x["m_dominant"] == 1], key=lambda x: abs(x["f_Hz"] - f0))[:2]
            if len(cand) == 2:
                fa, fb = sorted(x["f_Hz"] for x in cand)
                sp = (fb - fa) / (0.5 * (fa + fb))
                row["doublets"][name] = {"f_Hz": [fa, fb], "split_rel": sp, "c_measured": sp / e if e else None,
                                         "c_slater": pred[name]["c_split"], "f_round_Hz": f0}
        # TE012-like: m = 0 dominant with the largest E_phi fraction near f(TE012)
        fte = C_LIGHT * pred["TE012_Er"]["kR"] / R / (2 * math.pi)
        c0 = [x for x in info_m if x["m_dominant"] == 0 and abs(x["f_Hz"] / fte - 1) < 0.01]
        if c0:
            te = max(c0, key=lambda x: x["E_phi_fraction_m0"])
            tm = [x for x in c0 if x is not te]
            E = np.nan_to_num(np.asarray(modes[te["i"]]["probes"]["E"], float))
            Ec, phi = decompose(E, xyz)
            fit = fit_te012(Ec, phi, xyz, nr, nz, rr)
            cb = pred["TE012_Er"]["cB_barrel_max_over_A"] * abs(fit["A"])
            cw = max(pred["TE012_Er"]["cB_cap_max_over_A"], pred["TE012_Er"]["cB_barrel_max_over_A"]) * abs(fit["A"])
            row["TE012"] = {"f_Hz": te["f_Hz"], "E_phi_fraction": te["E_phi_fraction_m0"], **fit,
                            "Er_over_cBbarrel": abs(fit["Er_sin2phi_at_R"]) / cb,
                            "Er_over_cBwallmax": abs(fit["Er_sin2phi_at_R"]) / cw,
                            "Er_over_cBbarrel_per_e": abs(fit["Er_sin2phi_at_R"]) / cb / e if e else None,
                            "TM013_like": [{"f_Hz": x["f_Hz"], "E_phi_fraction": x["E_phi_fraction_m0"]} for x in tm]}
        rows.append(row)
        print(json.dumps({"e": e, "doublets": {k: (v["c_measured"], v["c_slater"], v["split_rel"]) for k, v in row["doublets"].items()},
                          "TE012": {k: row.get("TE012", {}).get(k) for k in ("f_Hz", "Er_over_cBbarrel", "Er_over_cBbarrel_per_e",
                                                                            "Er_cos2phi_at_R", "Er_fit_rel_resid", "E_phi_fraction")}},
                         default=float), flush=True)
    jsonstore.update(OUT, f"n{a.n}_o{a.order}", {"predictions": pred, "R_m": R, "d_over_R": LR, "rows": rows})


if __name__ == "__main__":
    main()
