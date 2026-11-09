"""External check 1 (beta): TESLA 1.3 GHz elliptical cells against published design tables.

References (downloaded, not committed; quote locations):
- Aune et al., PRST-AB 3, 092001 (2000), arXiv:physics/0003011v1. Table 3 (p. 12): half-cell shape parameters
  [mm] for midcup / endcup 1 / endcup 2: R_eq 103.3/103.3/103.3, R_iris 35/39/39, R_arc (circular equator arc)
  42.0/40.3/42, iris ellipse a (horizontal = axial) 12/10/9, b (vertical = radial) 19/13.5/12.8, length l
  57.7/56.0/57.0. Table 2 (p. 11): 9-cell pi mode 1300 MHz, cell-to-cell coupling 1.87 %, geometry factor
  270 Ohm, R/Q 518 Ohm with R = V^2/(2P) (footnote a: the other convention R = V^2/P gives twice that),
  E_peak/E_acc 2.0, B_peak/E_acc 4.26 mT/(MV/m), active length 1.038 m. Footnote 6 (p. 10):
  f_n = f0 / sqrt(1 + 2 k_cell cos(n pi/9)).
- Juntong & Jones, arXiv:0910.1305 Table 1 (p. 3) (also 0906.3592 Table 1): single mid-cell (Superfish) TESLA
  (R_i 35, R_eq 103.3, A = B = 42, a 12, b 19 mm): f_s 1.30116 GHz, bandwidth 24.32 MHz, k_c 1.89 %, E_s/E_a 2.18,
  B_s/E_a 4.18 mT/(MV/m); LL (R_i 30, R_eq 98.58, A 50.052, B 36.50, a 7.6, b 10 mm): 1.30008 GHz, 19.74 MHz,
  1.53 %, 2.42, 3.64. LL half-cell length 57.652 mm: 0906.3592 p. 1 ("the cell length is fixed at half the
  operating wavelength (L = lambda/2 = 115.304 mm)", stated for their optimisation); equivalently a + A =
  7.6 + 50.052 = 57.652 mm for LL (vertical wall between the arcs). 0910.1305 p. 2: Superfish vs HFSS differ by 1.7 %, 8.2 %,
  1.3 % in k_c, E_s/E_a, B_s/E_a.

Geometry not in the references (assumptions, stated): beam tubes of radius 39 mm (the end-cup iris radius) and
length `--tube` (default 0.2 m; sensitivity checked), flat ends. The contours are built by
`axionfem.srf._half_cell` (iris ellipse + equator ellipse joined by their internal common tangent; a circle arc is
the ellipse A = B = R_arc).

Mid-cell (stage ``cell``): the iris-centred domain [-l, l] (two half cups joined at the iris, z = 0) closed by flat
PEC planes at the two equators. Both the 0 mode (E_z even about the iris plane) and the pi mode (odd) of the infinite
periodic chain satisfy E_t = 0 on the equator (cell-centre) planes, so they are the two lowest TM0 modes of this
domain. Figures of merit for the pi mode: E_acc = |int E_z(0, z) e^{i w z/c} dz| / (2l) over the domain (one cell
length). The window matters at the 1e-3 level: the integrand is (2l)-periodic only if 2kl = pi exactly, and for TESLA
2kl - pi = 5e-3 rad, so V over the iris-centred window and over the iris-to-iris cell (from the pi-mode continuation
E(z + 2l) = -E(z)) differ by ~0.16 %; both are reported (*_iris_to_iris). U = (eps0/2) int|E|^2,
R/Q = V^2/(w U) (linac convention; Aune's convention is half of it), E_pk, B_pk = largest |E|, mu0|H_phi| on the
real cell wall (the equator planes are excluded), G = w mu0 int|H|^2 / int_wall |H_phi|^2, k_c = 2(f_pi - f_0)/(f_pi + f_0).

9-cell cavity (stage ``ninecell``): tube | endcup1 | 7.5 mid cells | endcup2 | tube (cell 1 = endcup 1 + midcup,
cells 2-8 two midcups, cell 9 = midcup + endcup 2), all PEC. The nine TM010-passband modes, pi mode figures as above
with V_acc = |int E_z e^{iwz/c} dz| over the whole axis (tubes included) and E_acc = V / (9 x 2 x 57.7 mm) (Aune's active length 1.038 m ~ 9 x 115.4 mm), field flatness (cell-peak E_z
spread), k_cell from a least-squares fit of footnote 6 to the nine frequencies.

    python benchmarks/srf/external/tesla.py cell --n 20 40 80
    python benchmarks/srf/external/tesla.py ninecell --n 20 40 [--tube 0.2]
    python benchmarks/srf/external/tesla.py cell3d --n 8 --order 2 --ranks 16      (MPI; one 3D cross-check)
-> benchmarks/srf/results/external_tesla.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from axionfem import srf  # noqa: E402
from axionfem.cavity import C_LIGHT, MU0, _design  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import jsonstore  # noqa: E402

EPS0 = 1 / (MU0 * C_LIGHT ** 2)
OUT = ROOT / "benchmarks/srf/results/external_tesla.json"
MM = 1e-3
# Aune et al. 2000 Table 3 [m]: (R_iris, R_eq, l, a (axial iris), b (radial iris), A = B = R_arc)
TESLA = {"mid": dict(Ri=35 * MM, Req=103.3 * MM, L=57.7 * MM, a=12 * MM, b=19 * MM, A=42.0 * MM, B=42.0 * MM),
         "end1": dict(Ri=39 * MM, Req=103.3 * MM, L=56.0 * MM, a=10 * MM, b=13.5 * MM, A=40.3 * MM, B=40.3 * MM),
         "end2": dict(Ri=39 * MM, Req=103.3 * MM, L=57.0 * MM, a=9 * MM, b=12.8 * MM, A=42.0 * MM, B=42.0 * MM)}
# Juntong & Jones 0910.1305 Table 1 (mid cells); LL half-length: 0906.3592 p. 1, "L = lambda/2 = 115.304 mm"
LL = dict(Ri=30 * MM, Req=98.58 * MM, L=57.652 * MM, a=7.6 * MM, b=10 * MM, A=50.052 * MM, B=36.50 * MM)
REF_CELL = {"TESLA": {"f_pi_GHz": 1.30116, "bandwidth_MHz": 24.32, "kc_pct": 1.89, "Epk_Eacc": 2.18, "Bpk_Eacc_mT_MVm": 4.18},
            "LL": {"f_pi_GHz": 1.30008, "bandwidth_MHz": 19.74, "kc_pct": 1.53, "Epk_Eacc": 2.42, "Bpk_Eacc_mT_MVm": 3.64}}
REF_9 = {"f_pi_MHz": 1300.0, "k_cell_pct": 1.87, "G_ohm": 270.0, "R_over_Q_aune_ohm": 518.0, "R_over_Q_linac_ohm": 1036.0,
         "Epk_Eacc": 2.0, "Bpk_Eacc_mT_MVm": 4.26, "active_length_m": 1.038}


def half_profile(p, angle_tol, ds, dmin):
    """Half-cup wall from the iris (z = 0, r = Ri) to the equator (z = L, r = Req), as (z, r) vertices; the vertex
    spacing turns the wall by <= angle_tol per chord (as srf.elliptical_cell_contour)."""
    P1, P2, (t1, p1) = srf._half_cell(p["Ri"], p["Req"], p["L"], p["a"], p["b"], p["A"], p["B"])

    def arc(P, u0, u1):
        uu = np.linspace(u0, u1, 20001)
        pts = np.array([P(u) for u in uu])
        d1 = np.gradient(pts, uu, axis=0)
        d2 = np.gradient(d1, uu, axis=0)
        sp = np.hypot(d1[:, 0], d1[:, 1])
        rho = sp ** 3 / np.maximum(np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]), 1e-300)
        g = np.clip(angle_tol * rho, dmin, ds)
        sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        cnt = np.concatenate([[0], np.cumsum(np.diff(sl) * 0.5 * (1 / g[1:] + 1 / g[:-1]))])
        n = max(4, int(math.ceil(cnt[-1])))
        return np.array([P(u) for u in np.interp(np.linspace(0, cnt[-1], n + 1), cnt, uu)])

    return np.vstack([arc(P1, 0.0, t1), arc(P2, p1, math.pi / 2)])   # (z, r)


def _dedup(pts, tol):
    out = [pts[0]]
    for q in pts[1:]:
        if math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > tol:
            out.append(q)
    return out


def chain(cups, angle_tol, ds, tube_r=None, tube_len=0.0):
    """Wall polyline (r, z) of a chain of half cups. ``cups``: list of (params, orientation), orientation +1 = iris
    then equator along +z, -1 = equator then iris. Starts at z = 0 (after the left tube, if any)."""
    z0, wall = 0.0, []
    if tube_r:
        wall += [(0.0, -tube_len), (tube_r, -tube_len), (tube_r, 0.0)]
    for p, o in cups:
        hp = half_profile(p, angle_tol, ds, ds / 20)
        if o > 0:
            seg = [(r, z0 + z) for z, r in hp]
        else:
            seg = [(r, z0 + p["L"] - z) for z, r in hp[::-1]]
        wall += seg
        z0 += p["L"]
    if tube_r:
        wall += [(tube_r, z0), (tube_r, z0 + tube_len), (0.0, z0 + tube_len)]
    return _dedup(wall, 1e-12), z0


def design_from_wall(name, poly, Req, zlo, zhi, h, order, modes, target, ws):
    t = 0.05 * Req
    regions = [{"name": "wall", "shape": {"rect": {"r": [0.0, Req + t], "z": [zlo - t, zhi + t]}}, "material": "wall_cu"},
               {"name": "cavity", "shape": {"polygon": {"points": [list(q) for q in poly]}}, "material": "vacuum",
                "mesh_size": h, "surface_size": ws}]
    R = 1.5 * math.hypot(Req + t, max(abs(zlo), abs(zhi)) + t)
    return _design(name, regions, wall_material=None, fill=None, h=h, order=order, Bz=1.0, modes=modes,
                   families=("TM",), target_f_Hz=target, interior_point=(0.2 * Req, 0.5 * (zlo + zhi)),
                   domain_radius=R, surface_size=ws)


def wall_samples(res, row, wall, Req, n_per_seg=4):
    """|E| and mu0|H_phi| sampled just inside each wall segment (offsets 1e-6..1e-4 Req along +-normal), with the
    area elements 2 pi r ds."""
    W = np.asarray(wall, float)
    P, T, dA = [], [], []
    for a, b in zip(W[:-1], W[1:]):
        ln = float(np.hypot(*(b - a)))
        if ln <= 0:
            continue
        for u in (np.arange(n_per_seg) + 0.5) / n_per_seg:
            q = a + u * (b - a)
            P.append(q)
            T.append((b - a) / ln)
            dA.append(2 * math.pi * q[0] * ln / n_per_seg)
    P, T, dA = np.array(P), np.array(T), np.array(dA)
    N = np.column_stack([-T[:, 1], T[:, 0]])
    E = np.full(len(P), np.nan)
    B = np.full(len(P), np.nan)
    for off in (1e-6, 1e-5, 1e-4):
        for sg in (1, -1):
            todo = ~np.isfinite(E)
            if not todo.any():
                break
            Q = P[todo] + sg * off * Req * N[todo]
            s = res.cavity_mode_sample(row, Q[:, 0], Q[:, 1])
            E[todo] = np.sqrt(s["E_r"] ** 2 + s["E_z"] ** 2 + s["E_phi"] ** 2)
            B[todo] = MU0 * np.abs(s["H_phi"])
    bad = ~np.isfinite(E)
    if bad.mean() > 2e-3:
        raise RuntimeError(f"wall sampling: {int(bad.sum())} of {len(E)} points outside the mesh")
    if bad.any():
        i = np.arange(len(E))
        E[bad] = np.interp(i[bad], i[~bad], E[~bad])
        B[bad] = np.interp(i[bad], i[~bad], B[~bad])
    return P, E, B, dA


def mode_figures(res, row, wall, Req, zlo, zhi, L_acc, alt=None):
    """pi-mode figures (see module docstring). Field normalization: re-measured int |E|^2 and int |H|^2.
    V over [zlo, zhi], E_acc = V/L_acc. ``alt``: a second V window, "iris_to_iris" (single cell, by the pi-mode
    continuation) or (z0, z1) (e.g. the 9-cell active length without the tube fringe fields); figures *_alt."""
    from axionfem.srf import _tri_product_integral
    F = res.cavity_mode_field(row, subdiv=4)
    I = lambda f, g: _tri_product_integral(F["r"], F["z"], F["tris"], f, g)  # noqa: E731
    EE = sum(I(F[k], F[k]) for k in ("E_r", "E_z", "E_phi"))
    HH = sum(I(F[k], F[k]) for k in ("H_r", "H_phi", "H_z"))
    w = 2 * math.pi * row["f_Hz"]
    pad = 1e-9 * (zhi - zlo)   # the end points lie on PEC planes / caps: sample strictly inside
    zz = np.linspace(zlo + pad, zhi - pad, 8001)
    ez = res.cavity_mode_sample(row, np.full_like(zz, 1e-7 * Req), zz)["E_z"]
    if not np.all(np.isfinite(ez)):
        raise RuntimeError(f"on-axis E_z: {int(np.sum(~np.isfinite(ez)))} samples outside the mesh")
    k = w / C_LIGHT
    Vacc = abs(np.trapezoid(ez * np.exp(1j * k * zz), zz))
    Lacc = L_acc
    Eacc = Vacc / Lacc
    Valt = None
    if alt == "iris_to_iris":   # [0, 2l] = [0, l] + continuation of [-l, 0] shifted by 2l with a sign flip
        ph = ez * np.exp(1j * k * zz)
        pos, neg = zz >= 0, zz <= 0
        Valt = abs(np.trapezoid(ph[pos], zz[pos]) - np.exp(2j * k * zhi) * np.trapezoid(ph[neg], zz[neg]))
    elif alt is not None:
        sel = (zz >= alt[0]) & (zz <= alt[1])
        Valt = abs(np.trapezoid((ez * np.exp(1j * k * zz))[sel], zz[sel]))
    U = 0.5 * EPS0 * EE
    P, E, B, dA = wall_samples(res, row, wall, Req)
    G = w * MU0 * HH / float(np.sum((B / MU0) ** 2 * dA))
    ie, ib = int(np.argmax(E)), int(np.argmax(B))
    return {"f_Hz": row["f_Hz"], "V_acc": Vacc, "L_acc_m": Lacc, "E_acc": Eacc, "U_J": U,
            "R_over_Q_linac_ohm": Vacc ** 2 / (w * U), "R_over_Q_aune_ohm": Vacc ** 2 / (2 * w * U),
            "Epk_over_Eacc": float(E[ie] / Eacc), "Bpk_over_Eacc_mT_per_MVm": float(B[ib] / Eacc * 1e9),
            "Epk_at_rz": P[ie].tolist(), "Bpk_at_rz": P[ib].tolist(), "G_ohm": G, "G_solver_all_pec_ohm": row.get("G_ohm"),
            "energy_check_eps0E2_over_mu0H2": EPS0 * EE / (MU0 * HH), "ez_axis": (zz, ez),
            **({} if Valt is None else {"alt_window": alt, "V_acc_alt": Valt, "R_over_Q_linac_ohm_alt": Valt ** 2 / (w * U),
                                        "R_over_Q_aune_ohm_alt": Valt ** 2 / (2 * w * U),
                                        "Epk_over_Eacc_alt": float(E[ie] / (Valt / Lacc)),
                                        "Bpk_over_Eacc_mT_per_MVm_alt": float(B[ib] / (Valt / Lacc) * 1e9)})}


def _pick_pass_band(rows, n):
    rows = sorted(rows, key=lambda r: r["f_Hz"])
    return rows[:n]


def run_cell(name, p, n, order, work, angle_scale=0.4):
    Req = p["Req"]
    h = Req / n
    at = angle_scale / n
    right, _ = chain([(p, +1)], at, h / 3)
    # iris-centred domain: (r, z) from the left equator plane through the iris (z = 0) to the right equator plane
    left = [(r, -z) for r, z in right[::-1]]
    wall = _dedup(left + right[1:], 1e-12)
    L = p["L"]
    poly = [(0.0, -L)] + wall + [(0.0, L)]
    d = design_from_wall(f"cell_{name}", poly, Req, -L, L, h, order, 6, 1.3e9, h / 3)
    r = d.solve(f"{work}/{name}_n{n}_o{order}", verbose=0)
    rows = [x for x in r.cavity_modes() if x["family"] == "TM"]
    lo = _pick_pass_band(rows, 2)
    figs = []
    for row in lo:
        zz = np.linspace(-0.9 * L, 0.9 * L, 401)
        ez = np.nan_to_num(r.cavity_mode_sample(row, np.full_like(zz, 1e-7 * Req), zz)["E_z"])
        parity = 1 if np.sum(ez * ez[::-1]) > 0 else -1
        figs.append((parity, row))
    zero = [row for par, row in figs if par > 0]
    pi = [row for par, row in figs if par < 0]
    if len(zero) != 1 or len(pi) != 1:
        raise RuntimeError(f"{name}: could not identify the 0 and pi modes among {[(p_, x['f_Hz']) for p_, x in figs]}")
    m = mode_figures(r, pi[0], wall, Req, -L, L, 2 * L, alt="iris_to_iris")
    m.pop("ez_axis")
    f0, fp = zero[0]["f_Hz"], pi[0]["f_Hz"]
    return {"cell": name, "h_over_Req": 1 / n, "order": order, "angle_tol": at, "f_0_Hz": f0, "f_pi_Hz": fp,
            "bandwidth_Hz": fp - f0, "kc_pct": 200 * (fp - f0) / (fp + f0), **m}


def stage_cell(a):
    out = {}
    with tempfile.TemporaryDirectory(prefix="tesla_") as work:
        for name, p in (("TESLA", TESLA["mid"]), ("LL", LL)):
            rows = []
            for n in a.n:
                x = run_cell(name, p, n, a.order, work)
                rows.append(x)
                print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()
                             if k in ("h_over_Req", "f_0_Hz", "f_pi_Hz", "kc_pct", "Epk_over_Eacc",
                                      "Bpk_over_Eacc_mT_per_MVm", "G_ohm", "R_over_Q_linac_ohm")}, flush=True)
            out[name] = {"reference": REF_CELL[name], "rows": rows}
    _store("cell", {"order": a.order, **out})


def build_ninecell(angle_tol, ds, tube_len):
    m, e1, e2 = TESLA["mid"], TESLA["end1"], TESLA["end2"]
    cups = [(e1, +1), (m, -1)] + [(m, +1), (m, -1)] * 7 + [(m, +1), (e2, -1)]
    wall, ztot = chain(cups, angle_tol, ds, tube_r=39 * MM, tube_len=tube_len)
    return wall, ztot


def stage_ninecell(a):
    rows = []
    with tempfile.TemporaryDirectory(prefix="tesla9_") as work:
        for n in a.n:
            Req = TESLA["mid"]["Req"]
            h = Req / n
            wall, ztot = build_ninecell(0.4 / n, h / 3, a.tube)
            poly = wall  # the wall starts and ends on the axis
            d = design_from_wall("tesla9", poly, Req, -a.tube, ztot + a.tube, h, a.order, 14, 1.29e9, h / 3)
            r = d.solve(f"{work}/n{n}", verbose=0, **({"mpi_ranks": a.ranks} if a.ranks > 1 else {}))
            tm = sorted([x for x in r.cavity_modes() if x["family"] == "TM"], key=lambda x: x["f_Hz"])
            band = [x for x in tm if 1.2e9 < x["f_Hz"] < 1.32e9]
            fs = np.array([x["f_Hz"] for x in band])
            if len(band) != 9:
                raise RuntimeError(f"9-cell: {len(band)} TM modes in 1.2-1.32 GHz: {fs}")
            # footnote 6: f_n = f0/sqrt(1 + 2k cos(n pi/9)), n = 1..9 (n = 9: pi mode, highest)
            from scipy.optimize import least_squares
            nn = np.arange(1, 10)
            sol = least_squares(lambda q: q[0] / np.sqrt(1 + 2 * q[1] * np.cos(nn * np.pi / 9)) - fs, [fs.mean(), 0.0187])
            pi = band[-1]
            # V_acc over the whole axis (cells + beam-tube fringe fields), E_acc = V/(9 x 115.4 mm)
            m = mode_figures(r, pi, wall, Req, -a.tube, ztot + a.tube, 9 * 2 * 57.7 * MM, alt=(0.0, ztot))
            zz, ez = m.pop("ez_axis")
            # cell peaks of |E_z| on axis (field flatness)
            edges = np.concatenate([[0.0], np.cumsum([56.0 + 57.7] + [115.4] * 7 + [57.7 + 57.0])]) * MM
            peaks = [float(np.max(np.abs(ez[(zz >= edges[i]) & (zz <= edges[i + 1])]))) for i in range(9)]
            flat = 1 - np.std(peaks) / np.mean(peaks)
            row = {"h_over_Req": 1 / n, "order": a.order, "tube_m": a.tube, "passband_Hz": fs.tolist(),
                   "k_cell_fit_pct": 100 * sol.x[1], "f0_fit_Hz": sol.x[0],
                   "k_cell_fit_rms_resid_Hz": float(np.sqrt(np.mean(sol.fun ** 2))),
                   "cell_peak_Ez": peaks, "field_flatness": float(flat),
                   "field_flatness_min_over_max": float(min(peaks) / max(peaks)), **m}
            rows.append(row)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()
                   if k in ("h_over_Req", "f_Hz", "k_cell_fit_pct", "Epk_over_Eacc", "Bpk_over_Eacc_mT_per_MVm",
                            "G_ohm", "G_solver_all_pec_ohm", "R_over_Q_linac_ohm", "R_over_Q_aune_ohm", "field_flatness")}, flush=True)
    _store("ninecell", {f"tube{a.tube:g}_o{a.order}": {"reference": REF_9, "rows": rows}}, merge=True)


def stage_cell3d(a):
    """One 3D cross-check: the TESLA mid-cell iris-centred domain revolved (the 3D path, faceted wall), pi and 0
    modes, f and G against the 2D values of the same domain."""
    from axionfem.cavity import _design3d
    p = TESLA["mid"]
    Req, L = p["Req"], p["L"]
    h = Req / a.n
    right, _ = chain([(p, +1)], 0.1, h / 2)
    left = [(r, -z) for r, z in right[::-1]]
    wall = _dedup(left + right[1:], 1e-12)
    poly = [(0.0, -L)] + wall + [(0.0, L)]
    solid = {"revolve": {"profile": {"polygon": {"points": [list(q) for q in poly]}}, "angles": [0.0, 360.0],
                         "center": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}}
    regions = [{"name": "cavity", "material": "vacuum", "solid": solid}]
    zp = np.linspace(-0.98 * L, 0.98 * L, 401)
    d = _design3d("tesla_cell_3d", regions, solid, (0.17 * Req, 0.11 * Req, 0.31 * L), h=h, order=a.order,
                  modes=6, target_f_Hz=1.29e9, eigensolver={"method": "shift_invert"}, symmetry=None,
                  B_T=(0.0, 0.0, 1.0), wall_material=None, losses=True, mode_overlaps=False, surface_size=h / 2,
                  remove_pec=False)
    d.doc["fullwave"]["field_probes"] = [[0.0, 0.0, float(z)] for z in zp]
    work = Path(a.workdir or tempfile.mkdtemp(prefix="tesla3d_"))
    r = d.solve(work / f"n{a.n}_o{a.order}", verbose=0, mpi_ranks=a.ranks)
    modes = r.fullwave["eigenmodes"]["modes"]
    rows = []
    for i, m in enumerate(modes):
        E = np.nan_to_num(np.asarray(m["probes"]["E"], float))
        ez = E[:, 2]
        w = 2 * math.pi * m["f_Hz"]
        Vacc = abs(np.trapezoid(ez * np.exp(1j * w / C_LIGHT * zp), zp))
        pk = m.get("peak_wall_fields", {})
        rows.append({"i": i, "f_Hz": m["f_Hz"], "E_azimuthal_fraction": m.get("E_azimuthal_fraction"),
                     "axis_Ez_parity": float(np.sum(ez * ez[::-1]) / max(np.sum(ez * ez), 1e-300)),
                     "G_solver_all_pec_ohm": m.get("losses", {}).get("G_ohm"), "V_axis_rel": Vacc,
                     "peak_wall_fields": pk})
        print(rows[-1]["f_Hz"], rows[-1]["E_azimuthal_fraction"], rows[-1]["axis_Ez_parity"], rows[-1]["G_solver_all_pec_ohm"], flush=True)
    # matching 2D solve of the same polygon (all-PEC G, comparable to the 3D solver's G)
    with tempfile.TemporaryDirectory(prefix="tesla3d2d_") as w2:
        d2 = design_from_wall("cell_tesla_2d", poly, Req, -L, L, Req / 40, 3, 6, 1.3e9, Req / 120)
        r2 = d2.solve(f"{w2}/x", verbose=0)
        ref2 = [{"f_Hz": x["f_Hz"], "G_solver_all_pec_ohm": x["G_ohm"]} for x in r2.cavity_modes() if x["family"] == "TM"]
    _store("cell3d", {f"n{a.n}_o{a.order}": {"rows": rows, "ref2d_same_polygon_angle_tol_0.1": ref2,
                               "note": "3D wall polygon angle_tol 0.1 (faceted 3D wall); 2D reference "
                                       "uses the same polygon at h = Req/40, order 3"}}, merge=True)


def _load():
    return jsonstore.load(OUT)


def _store(key, val, merge=False):
    jsonstore.update(OUT, key, val, merge=merge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cell", "ninecell", "cell3d"])
    ap.add_argument("--n", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--tube", type=float, default=0.2)
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    if a.stage == "cell3d":
        a.n = a.n[0]
    {"cell": stage_cell, "ninecell": stage_ninecell, "cell3d": stage_cell3d}[a.stage](a)


if __name__ == "__main__":
    main()
