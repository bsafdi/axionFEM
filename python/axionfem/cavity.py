"""Cavity haloscopes (axisymmetric m = 0 modes): design builders and closed-form references.

Beta. The solver side is the design section
``cavity`` (docs/design_format.md, "Cavity haloscopes"): the eigenmodes of ``fullwave`` restricted to the
cavity selected by ``cavity.interior_point``, with per-mode form factor C, geometry factor G = Q R_s and
Q0 (wall and dielectric losses). ``Results.cavity_modes()`` flattens them into rows.

Closed forms (derived independently of the solver; used as test references):

- pillbox (radius a, length L): TM0np  k^2 = (chi_0n/a)^2 + (p pi/L)^2, p >= 0;
  TE0np  k^2 = (chi'_0n/a)^2 + (p pi/L)^2, p >= 1 (chi'_0n = zeros of J1);
  C(TM0n0) = 4/chi_0n^2, C = 0 otherwise (axial uniform B0);
  G(TM0n0) = Z0 chi_0n L / (2 (a + L));  G(TM0np, p > 0) = Z0 k a L / (2 (L + 2 a));  G(TE0np) = Z0 k^3 a L / (2 (kc^2 L + 2 a beta^2)) (Pozar, Microwave Engineering, Sec. 6.4, TE_nml with n = 0).
- coaxial cavity (rod a spanning both end caps, outer wall b): lowest coupled mode J0(kb)Y0(ka) = Y0(kb)J0(ka),
  C over the vacuum volume pi (b^2 - a^2) L.
"""
from __future__ import annotations

import math
from typing import Any

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6  # H/m (CODATA 2018, as cpp/include/axionfem/units.hpp)
Z0 = MU0 * C_LIGHT  # free-space impedance [ohm]

__all__ = ["pillbox_design", "coax_design", "pillbox_modes", "coax_tm_mode", "sleeve_tm_mode", "skin_Rs",
           "surface_resistance"]


def skin_Rs(f_Hz: float, rho_ohm_m: float) -> float:
    """Normal-skin surface resistance R_s = sqrt(pi f mu0 rho) [ohm] (the ``rf`` model "skin")."""
    return math.sqrt(math.pi * f_Hz * MU0 * rho_ohm_m)


def surface_resistance(rf: dict, f_Hz, *, flux_flow_part: bool = False):
    """R_s(f) [ohm] of an ``rf`` block of the design format (any model: skin, power, table, anomalous, bcs, with an
    optional flux_flow term), evaluated by the same C++ code as the solvers (arguments checked by the design parser).
    Scalar or sequence f_Hz. With ``flux_flow_part=True`` returns (R_s, flux_flow part of R_s). The flux_flow part here
    needs an explicit ``B_T`` and uses the fixed ``orientation`` factor, or 1 (current perpendicular to the vortices, the
    upper bound); the cavity wall losses weight it per wall point by sin^2(J_rf, B0) instead."""
    import json

    import numpy as np

    from . import _core

    f = np.atleast_1d(np.asarray(f_Hz, dtype=float))
    Rs, ff = _core.surface_resistance(json.dumps(rf), [float(x) for x in f])
    if np.ndim(f_Hz) == 0:
        Rs, ff = float(Rs[0]), float(ff[0])
    return (Rs, ff) if flux_flow_part else Rs


def _design(name: str, regions: list, *, wall_material: dict | None, fill: dict | None, h: float, order: int,
            Bz: float, modes: int, families, target_f_Hz, interior_point, domain_radius: float,
            surface_size: float | None) -> "Any":
    from .design import Design

    mats: dict = {"wall_cu": wall_material or {"kind": "conductor"}}
    if fill is not None:
        mats["fill"] = dict(fill)
    fw: dict = {"modes": modes, "families": list(families), "boundary": "pec"}
    if target_f_Hz:
        fw["target_f_Hz"] = target_f_Hz
    doc = {"schema": "axionfem.design/1", "name": name, "materials": mats, "regions": regions,
           "background": {"type": "uniform", "Bz": Bz},
           "fullwave": fw, "cavity": {"interior_point": list(interior_point)},
           # the cavity region carries mesh_size = h; the room outside the wall is excluded, so it stays coarse
           "mesh": {"order": order, "size": max(h, domain_radius / 20), "surface_size": surface_size or h,
                    "domain_radius": domain_radius}}
    if wall_material is None:  # the default wall is lossless on purpose (Q0 = Q_dielectric, flagged), as in 3D
        doc["cavity"]["allow_lossless_walls"] = True
    return Design(doc)


def pillbox_design(a: float = 0.1, L: float = 0.2, *, t: float | None = None, h: float | None = None,
                   order: int = 2, wall_material: dict | None = None, fill: dict | None = None, Bz: float = 1.0,
                   modes: int = 6, families=("TE", "TM"), target_f_Hz: float | None = None,
                   domain_radius: float | None = None):
    """Closed right cylinder (inner radius a, length L, centred on z = 0) with a conducting wall of thickness t
    (default a/10) inside the closed outer sphere, a uniform axial background field Bz, and the cavity section.
    ``fill``: material of the interior (e.g. ``{"kind": "linear", "eps_r": 2, "tan_delta": 1e-4}``), vacuum
    by default. ``wall_material`` defaults to a perfect ``conductor`` without rf data, with
    ``cavity.allow_lossless_walls`` set (lossless on purpose; Q0 = Q_dielectric if any). ``domain_radius``: the
    closed outer sphere (default 1.5x the device); the mesher caps elements at about pi R/96, so a larger
    sphere is needed for coarse cavity meshes (convergence studies)."""
    t = t or a / 10
    h = h or a / 5
    wall = {"union": [{"rect": {"r": [a, a + t], "z": [-L / 2 - t, L / 2 + t]}},
                      {"rect": {"r": [0.0, a], "z": [L / 2, L / 2 + t]}},
                      {"rect": {"r": [0.0, a], "z": [-L / 2 - t, -L / 2]}}]}
    regions = [{"name": "cavity", "shape": {"rect": {"r": [0.0, a], "z": [-L / 2, L / 2]}},
                "material": "fill" if fill is not None else "vacuum", "mesh_size": h},
               {"name": "wall", "shape": wall, "material": "wall_cu"}]
    R = domain_radius or 1.5 * math.hypot(a + t, L / 2 + t)
    return _design("pillbox", regions, wall_material=wall_material, fill=fill, h=h, order=order, Bz=Bz,
                   modes=modes, families=families, target_f_Hz=target_f_Hz, interior_point=(a / 2, 0.0),
                   domain_radius=R, surface_size=None)


def coax_design(a: float = 0.05, b: float = 0.2, L: float = 0.2, *, t: float | None = None, h: float | None = None,
                order: int = 2, wall_material: dict | None = None, Bz: float = 1.0, modes: int = 4,
                families=("TM",), target_f_Hz: float | None = None, domain_radius: float | None = None):
    """Coaxial cavity: conducting rod of radius a on the axis spanning both end caps, outer wall radius b,
    length L; the rod, caps and barrel form one conducting body."""
    t = t or b / 10
    h = h or (b - a) / 6
    wall = {"union": [{"rect": {"r": [0.0, a], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [b, b + t], "z": [-L / 2 - t, L / 2 + t]}},
                      {"rect": {"r": [0.0, b], "z": [L / 2, L / 2 + t]}},
                      {"rect": {"r": [0.0, b], "z": [-L / 2 - t, -L / 2]}}]}
    regions = [{"name": "cavity", "shape": {"rect": {"r": [a, b], "z": [-L / 2, L / 2]}}, "material": "vacuum",
                "mesh_size": h},
               {"name": "wall", "shape": wall, "material": "wall_cu"}]
    R = domain_radius or 1.5 * math.hypot(b + t, L / 2 + t)
    return _design("coax", regions, wall_material=wall_material, fill=None, h=h, order=order, Bz=Bz,
                   modes=modes, families=families, target_f_Hz=target_f_Hz, interior_point=((a + b) / 2, 0.0),
                   domain_radius=R, surface_size=None)


def pillbox_modes(a: float, L: float, f_max: float, eps_r: float = 1.0) -> list[dict]:
    """Analytic m = 0 modes of a closed cylinder below f_max: family ("TM" or "TE"), n, p, f_Hz, C (axial uniform
    B0, eps-weighted; 0 except TM0n0), G_ohm (Q R_s with one R_s on all walls; vacuum filling only)."""
    from scipy.special import jn_zeros

    kmax = 2 * math.pi * f_max * math.sqrt(eps_r) / C_LIGHT
    nmax = int(kmax * a / math.pi) + 3
    out = []
    for fam, zeros in (("TM", jn_zeros(0, nmax)), ("TE", jn_zeros(1, nmax))):
        for n, x in enumerate(zeros, start=1):
            kc = x / a
            for p in range(0 if fam == "TM" else 1, int(kmax * L / math.pi) + 2):
                beta = p * math.pi / L
                k = math.hypot(kc, beta)
                if k > kmax:
                    break
                f = C_LIGHT * k / (2 * math.pi * math.sqrt(eps_r))
                if fam == "TM":
                    C = 4 / x ** 2 if p == 0 else 0.0
                    # TM0np: H_phi ~ J1(kc r) cos(beta z); int |H|^2 = pi L a^2 J1^2 / (2 or 1), barrel 2 pi a L J1^2
                    # (/2 for p > 0), caps 2 pi a^2 J1^2:  G = Z0 k a L / (2 (L + a)) (p = 0), / (2 (L + 2 a)) (p > 0)
                    G = Z0 * k * a * L / (2 * (L + (a if p == 0 else 2 * a)))
                else:
                    C = 0.0
                    G = Z0 * k ** 3 * a * L / (2 * (kc ** 2 * L + 2 * a * beta ** 2))
                out.append({"family": fam, "n": n, "p": p, "f_Hz": f, "C": C, "G_ohm": G})
    out.sort(key=lambda m: m["f_Hz"])
    return out


def coax_tm_mode(a: float, b: float) -> dict:
    """Lowest z-independent TM mode of a coaxial cavity (rod radius a spanning both end caps, outer radius b):
    J0(kb)Y0(ka) - Y0(kb)J0(ka) = 0; C over the vacuum volume pi (b^2 - a^2) L (closed-form reference)."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1, y0, y1

    u = lambda k: j0(k * b) * y0(k * a) - y0(k * b) * j0(k * a)  # noqa: E731
    # first sign change above 0 on a fine grid, then Brent
    ks = [i * (20.0 / (b - a)) / 4000 for i in range(1, 4001)]
    k0 = next(k1 for k1, k2 in zip(ks, ks[1:]) if u(k1) * u(k2) < 0)
    k = brentq(u, k0, k0 + (20.0 / (b - a)) / 4000, xtol=1e-15, rtol=1e-15)
    v = lambda r: j1(k * r) * y0(k * a) - y1(k * r) * j0(k * a)  # noqa: E731
    C = 4 * (b * v(b) - a * v(a)) ** 2 / (k ** 2 * (b ** 2 - a ** 2) * (b ** 2 * v(b) ** 2 - a ** 2 * v(a) ** 2))
    return {"k": k, "kb": k * b, "f_Hz": C_LIGHT * k / (2 * math.pi), "C": C}


def sleeve_tm_mode(b: float, r1: float, r2: float, eps: float) -> dict:
    """Lowest z-independent TM mode of a cylinder (radius b) with a coaxial dielectric sleeve r1 < r < r2
    (relative permittivity eps; r1 = 0 is a central rod, r2 = b a lining on the wall): E_z and dE_z/dr continuous
    (mu = mu0), E_z(b) = 0. Returns k, f, the eps-weighted C over the full cylinder volume and
    ``fe_sleeve``, the fraction of the electric energy in the sleeve (1/Q_d = tan(delta) fe_sleeve)."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import j0, j1, y0, y1

    if not (0 <= r1 < r2 <= b) or not eps > 0:
        raise ValueError(f"sleeve_tm_mode needs 0 <= r1 < r2 <= b and eps > 0 (got b={b}, r1={r1}, r2={r2}, eps={eps})")
    # layers [lo, hi, refractive index]; the innermost holds J0 only (regular on the axis), the others J0 and Y0
    layers = []
    for lo, hi, n in ((0.0, r1, 1.0), (r1, r2, math.sqrt(eps)), (r2, b, 1.0)):
        if hi > lo:
            layers.append((lo, hi, n))
    nl = len(layers)
    nu = 2 * nl - 1  # unknowns: a_0; (a_i, c_i) for i >= 1

    def cols(i):
        return [0] if i == 0 else [2 * i - 1, 2 * i]

    def basis(i, k, r):  # (E_z, dE_z/dr) of the layer's basis functions at r
        q = k * layers[i][2]
        out = [(j0(q * r), -q * j1(q * r))]
        if i > 0:
            out.append((y0(q * r), -q * y1(q * r)))
        return out

    def mat(k):
        A = np.zeros((nu, nu))
        row = 0
        for i in range(nl - 1):  # continuity of E_z and dE_z/dr at layers[i].hi
            r = layers[i][1]
            for sgn, j in ((1.0, i), (-1.0, i + 1)):
                for c, (v, dv) in zip(cols(j), basis(j, k, r)):
                    A[row, c] += sgn * v
                    A[row + 1, c] += sgn * dv
            row += 2
        for c, (v, _) in zip(cols(nl - 1), basis(nl - 1, k, b)):  # E_z(b) = 0
            A[row, c] = v
        return A

    def det(k):
        A = mat(k)
        return np.linalg.det(A / np.abs(A).max(axis=1)[:, None])

    kmax = 2.4048255577 / b  # the filled mode lies below the empty one
    ks = np.linspace(1e-3 * kmax, kmax * (1 + 1e-9), 4000)
    dv = [det(k) for k in ks]
    i = next((i for i in range(len(ks) - 1) if dv[i] * dv[i + 1] < 0), None)
    if i is None:
        raise ValueError("sleeve_tm_mode: no root of the matching determinant below the empty-cylinder TM010")
    k = brentq(det, ks[i], ks[i + 1], xtol=1e-14, rtol=1e-15)
    c = np.linalg.svd(mat(k))[2][-1]

    def ez(r):
        for i, (lo, hi, _) in enumerate(layers):
            if r <= hi or i == nl - 1:
                return sum(cc * bv[0] for cc, bv in zip((c[j] for j in cols(i)), basis(i, k, r)))

    pts = [x for x in (r1, r2) if 0 < x < b]
    kw = dict(points=pts or None, epsabs=0, epsrel=1e-13, limit=200)
    I1 = quad(lambda r: ez(r) * r, 0, b, **kw)[0]
    wt = lambda r: eps if r1 <= r < r2 else 1.0  # noqa: E731
    I2 = quad(lambda r: wt(r) * ez(r) ** 2 * r, 0, b, **kw)[0]
    Is = quad(lambda r: eps * ez(r) ** 2 * r, r1, r2, epsabs=0, epsrel=1e-13, limit=200)[0]
    # C = (2 pi L I1)^2 / (pi b^2 L * 2 pi L I2) = 2 I1^2 / (b^2 I2)
    return {"k": k, "f_Hz": C_LIGHT * k / (2 * math.pi), "C": 2 * I1 ** 2 / (b * b * I2), "fe_sleeve": Is / I2}


# ----------------------------------------------------------------------------------------------------------------
# 3D cavities (beta). The design section ``cavity`` of a 3D design post-processes the
# eigenmodes of a closed PEC model (``fullwave.rf_only`` with ``mesh.domain_solid`` as the cavity wall, tuning rods
# as PEC holes with ``mesh.remove_pec``); B0 for the form factor is ``fullwave.source`` (the applied field).
# ----------------------------------------------------------------------------------------------------------------
def _sym_parity(fam: str, m: int, p: int, kind: str, axis: str) -> int:
    """Parity of the field of a cylinder mode (axis z, centred on z = 0) under the mirror x -> -x ("x"),
    y -> -y ("y") or z -> -z ("z"): +1 if E(Rx) = R E(x) (allowed by a PMC plane), -1 if E(Rx) = -R E(x) (PEC).
    TM: E_z ~ J_m(.) {cos, sin}(m phi) cos(p pi (z + L/2)/L); TE: H_z ~ J_m(.) {cos, sin}(m phi) sin(p pi (z + L/2)/L)
    (H is a pseudovector: its parity sign is the opposite of E's for the same function)."""
    if axis == "y":    # phi -> -phi; the z components are tangential
        f = 1 if kind == "cos" else -1
        return f if fam == "TM" else -f
    if axis == "x":    # phi -> pi - phi; tangential z components
        f = (-1) ** m if kind == "cos" else (-1) ** (m + 1)
        return f if fam == "TM" else -f
    # z -> -z: the z components are normal (E_n parity = +1 for E(Rx) = -R E(x))
    if fam == "TM":
        return -((-1) ** p)          # E_z parity (-1)^p; normal component: E symmetric <=> E_z odd
    return (-1) ** (p + 1)           # H_z parity (-1)^(p+1); normal H even <=> E symmetric


def cylinder_modes_3d(a: float, L: float, f_max: float, symmetry: dict | None = None) -> list[dict]:
    """All TM_mnp / TE_mnp modes of a closed vacuum cylinder (radius a, length L) below f_max, each m > 0 member of a
    degenerate pair listed separately (``kind`` "cos"/"sin" of m phi; for TM the E_z dependence, for TE the H_z one).
    ``symmetry``: {"x": "pec"|"pmc", "y": ..., "z": ...} keeps the modes a model with those mirror planes (through the
    centre) supports. C (axial uniform B0): 4/x_0n^2 for TM0n0, else 0; G_ohm = Q R_s for TM0n0 (a + L walls)."""
    from scipy.special import jn_zeros, jnp_zeros

    kmax = 2 * math.pi * f_max / C_LIGHT
    out = []
    m = 0
    while True:
        added = False
        for fam in ("TM", "TE"):
            nmax = int(kmax * a / math.pi) + 3
            zeros = jn_zeros(m, nmax) if fam == "TM" else jnp_zeros(m, nmax)
            for n, x in enumerate(zeros, start=1):
                kc = x / a
                if kc > kmax:
                    break
                for p in range(0 if fam == "TM" else 1, int(kmax * L / math.pi) + 2):
                    k = math.hypot(kc, p * math.pi / L)
                    if k > kmax:
                        break
                    for kind in (("cos",) if m == 0 else ("cos", "sin")):
                        if symmetry and any(_sym_parity(fam, m, p, kind, ax) != (1 if t == "pmc" else -1)
                                            for ax, t in symmetry.items()):
                            continue
                        C = 4 / x ** 2 if (fam == "TM" and m == 0 and p == 0) else 0.0
                        G = Z0 * x * L / (2 * (a + L)) if (fam == "TM" and m == 0 and p == 0) else None
                        out.append({"family": fam, "m": m, "n": n, "p": p, "kind": kind, "f_Hz": C_LIGHT * k / (2 * math.pi),
                                    "C": C, "G_ohm": G, "name": f"{fam}{m}{n}{p}" + ("" if m == 0 else f"({kind})")})
                        added = True
        if not added and jnp_zeros(m, 1)[0] / a > kmax and jn_zeros(m, 1)[0] / a > kmax:
            break
        m += 1
    out.sort(key=lambda r: r["f_Hz"])
    return out


def box_modes(a: float, b: float, d: float, f_max: float) -> list[dict]:
    """Modes of a closed vacuum box 0 < x < a, 0 < y < b, 0 < z < d below f_max (TM/TE relative to z):
    TM_mnp m, n >= 1, p >= 0; TE_mnp m, n >= 0 (not both 0), p >= 1. C (uniform B0 along z) = 64/(pi^4 m^2 n^2) for
    TM_mn0 with m, n odd, else 0; G_ohm = Q R_s for TM_mn0 (closed-form reference)."""
    out = []
    kmax = 2 * math.pi * f_max / C_LIGHT
    M = int(kmax * max(a, b, d) / math.pi) + 2
    for m in range(0, M + 1):
        for n in range(0, M + 1):
            for p in range(0, M + 1):
                k = math.pi * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
                if k > kmax or k == 0:
                    continue
                for fam in ("TM", "TE"):
                    if fam == "TM" and (m == 0 or n == 0):
                        continue
                    if fam == "TE" and ((m == 0 and n == 0) or p == 0):
                        continue
                    C = G = None
                    if fam == "TM" and p == 0:
                        C = 64 / (math.pi ** 4 * m * m * n * n) if (m % 2 and n % 2) else 0.0
                        al, ga = m * math.pi / a, n * math.pi / b
                        G = Z0 * k / (2 / d + 4 * al ** 2 / (k ** 2 * a) + 4 * ga ** 2 / (k ** 2 * b))
                    elif fam == "TE" or p > 0:
                        C = 0.0
                    out.append({"family": fam, "m": m, "n": n, "p": p, "f_Hz": C_LIGHT * k / (2 * math.pi), "C": C,
                                "G_ohm": G, "name": f"{fam}{m}{n}{p}"})
    out.sort(key=lambda r: r["f_Hz"])
    return out


def _design3d(name: str, regions: list, domain: dict, interior_point, *, h: float, order: int, modes: int,
              target_f_Hz, eigensolver, symmetry, B_T, wall_material, losses, mode_overlaps, surface_size,
              remove_pec: bool):
    from .design import Design

    fw: dict = {"rf_only": True, "boundary": "pec", "modes": modes, "source": {"kind": "uniform", "B_T": list(B_T)}}
    if target_f_Hz:
        fw["target_f_Hz"] = target_f_Hz
    if eigensolver:
        fw["eigensolver"] = dict(eigensolver)
    mesh: dict = {"order": order, "size": h, "surface_size": surface_size or h, "domain_solid": domain}
    if remove_pec:
        mesh["remove_pec"] = True
    if symmetry:
        mesh["symmetry"] = [{"name": f"sym_{ax}", "axis": ax, "type": t} for ax, t in symmetry.items()]
    doc = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d",
           "materials": {"wall_cu": wall_material or {"kind": "conductor"}},
           "regions": regions, "fullwave": fw,
           "cavity": {"interior_point": list(interior_point), "wall_material": "wall_cu", "losses": losses,
                      "mode_overlaps": mode_overlaps},
           "mesh": mesh}
    if wall_material is None:  # explicit: the default wall is lossless on purpose (Q0 = Q_dielectric, flagged)
        doc["cavity"]["allow_lossless_walls"] = True
    return Design(doc)


def cylinder_design_3d(a: float = 0.1, L: float = 0.2, *, h: float | None = None, order: int = 2, modes: int = 6,
                       target_f_Hz: float | None = None, eigensolver: dict | None = None,
                       symmetry: dict | None = None, rod: dict | None = None, B_T=(0.0, 0.0, 1.0),
                       wall_material: dict | None = None, losses: bool = True, mode_overlaps: bool = False,
                       surface_size: float | None = None, interior_point=None, sleeve: dict | None = None):
    """3D closed cylinder cavity (radius a, length L, axis z, centred on z = 0): the PEC wall is the domain
    boundary (mesh.domain_solid), the interior one vacuum region. ``rod``: {"radius": r, "gap": g (0: touches both
    caps), "x": .., "y": .., "material": {...}}: a conducting rod meshed as a PEC hole (mesh.remove_pec), with
    end gaps g at both caps (a floating conductor when g > 0); "hole": False keeps it as a meshed conductor
    volume. ``symmetry``: {"x": "pec"|"pmc", "y": ...} mirror planes through the axis (the model keeps x >= 0 /
    y >= 0). B0: uniform ``B_T`` (fullwave.source).
    ``rod`` local refinement: "surface_size" (the rod surface) and "gap_size": element size inside two small vacuum
    regions ``rod_gap_lo``/``rod_gap_hi`` (cylinders of radius r + m from each cap to m beyond the rod tip, m =
    "refine_margin", default r) that resolve the gap field and the tip edges (docs/validation/cavity_3d.md,
    floating rod); "gap_size" also sets the rod's surface_size unless that is given.
    ``wall_material``: material of the wall (and rod). Default: a perfect ``conductor`` without rf data, with
    ``cavity.allow_lossless_walls`` set, so the walls are lossless on purpose and Q0 (if reported) is the dielectric
    Q alone (``losses.walls_lossless_by_request``); pass an rf material (e.g. ``{"kind": "conductor", "rf":
    {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 4}}``) for wall losses.
    ``sleeve``: {"r1": .., "r2": .., "material": {...}}: a coaxial dielectric tube r1 < r < r2 over the full length."""
    h = h or a / 5
    cyl = {"cylinder": {"base": [0.0, 0.0, -L / 2], "axis": [0.0, 0.0, L], "radius": a}}
    regions = [{"name": "cavity", "material": "vacuum", "solid": cyl}]
    if interior_point is None:
        r0 = 0.5 * a if not rod else 0.5 * (a + rod["radius"] + math.hypot(rod.get("x", 0.0), rod.get("y", 0.0)))
        interior_point = (0.83 * r0, 0.557 * r0, 0.0123 * L)
    if sleeve:
        tube = {"difference": [{"cylinder": {"base": [0.0, 0.0, -L / 2], "axis": [0.0, 0.0, L], "radius": sleeve["r2"]}},
                               {"cylinder": {"base": [0.0, 0.0, -L / 2], "axis": [0.0, 0.0, L], "radius": sleeve["r1"]}}]}
        regions.append({"name": "sleeve", "material": "sleeve_mat", "solid": tube})
    if rod:
        g = rod.get("gap", 0.0)
        x0, y0, r = rod.get("x", 0.0), rod.get("y", 0.0), rod["radius"]
        if rod.get("gap_size") and g > 0:  # listed before the rod: the rod owns the overlap
            m = rod.get("refine_margin", r)
            if math.hypot(x0, y0) + r + m >= a:
                raise ValueError("rod refine_margin: the gap regions would reach the cavity wall")
            for nm, z0 in (("rod_gap_lo", -L / 2), ("rod_gap_hi", L / 2 - g - m)):
                regions.append({"name": nm, "material": "vacuum", "mesh_size": rod["gap_size"],
                                "solid": {"cylinder": {"base": [x0, y0, z0], "axis": [0.0, 0.0, g + m],
                                                       "radius": r + m}}})
        regions.append({"name": "rod", "material": "rod_mat" if rod.get("material") else "wall_cu",
                        "solid": {"cylinder": {"base": [x0, y0, -L / 2 + g],
                                               "axis": [0.0, 0.0, L - 2 * g], "radius": r}}})
        ss = rod.get("surface_size") or (rod.get("gap_size") if g > 0 else None)
        if ss:
            regions[-1]["surface_size"] = ss
    d = _design3d("cylinder3d", regions, cyl, interior_point, h=h, order=order, modes=modes, target_f_Hz=target_f_Hz,
                  eigensolver=eigensolver, symmetry=symmetry, B_T=B_T, wall_material=wall_material, losses=losses,
                  mode_overlaps=mode_overlaps, surface_size=surface_size,
                  remove_pec=bool(rod) and rod.get("hole", True))
    if rod and rod.get("material"):
        d.doc["materials"]["rod_mat"] = dict(rod["material"])
    if sleeve:
        d.doc["materials"]["sleeve_mat"] = dict(sleeve["material"])
    return d


def dielectric_rod_tm_mode(a: float, r0: float, eps: float, L: float) -> dict:
    """Lowest z-independent TM mode of a closed cylinder (radius a, length L) with a coaxial dielectric rod of radius
    r0 (relative permittivity eps) over the full length: E_z = J0(k n r) inside, B J0(k r) + C Y0(k r) outside, E_z
    and dE_z/dr continuous at r0, E_z(a) = 0. Returns f, the eps-weighted C over the full volume, the electric
    energy fraction in the rod (1/Q_d = tan(delta) * fraction) and G = Q R_s (barrel and both caps)."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import j0, j1, y0, y1

    if not (0 < r0 < a) or not eps > 0:
        raise ValueError("dielectric_rod_tm_mode needs 0 < r0 < a and eps > 0")
    n = math.sqrt(eps)

    def parts(k):
        M = np.array([[j0(k * r0), y0(k * r0)], [-k * j1(k * r0), -k * y1(k * r0)]])
        return np.linalg.solve(M, np.array([j0(k * n * r0), -k * n * j1(k * n * r0)]))

    def g(k):
        B, C = parts(k)
        return B * j0(k * a) + C * y0(k * a)

    kmax = 2.4048255577 / a * (1 + 1e-9)  # the loaded mode lies below the empty TM010
    ks = np.linspace(1e-3 * kmax, kmax, 4000)
    v = [g(k) for k in ks]
    i = next((i for i in range(len(v) - 1) if v[i] * v[i + 1] < 0), None)
    if i is None:
        raise ValueError("dielectric_rod_tm_mode: no root below the empty-cylinder TM010")
    k = brentq(g, ks[i], ks[i + 1], xtol=1e-15, rtol=1e-15)
    B, C = parts(k)
    q = lambda f, lo, hi: quad(f, lo, hi, epsabs=0, epsrel=1e-13, limit=200)[0]  # noqa: E731
    Ei = lambda r: j0(k * n * r)  # noqa: E731
    Eo = lambda r: B * j0(k * r) + C * y0(k * r)  # noqa: E731
    dEi = lambda r: -k * n * j1(k * n * r)  # noqa: E731
    dEo = lambda r: -k * (B * j1(k * r) + C * y1(k * r))  # noqa: E731
    I1 = q(lambda r: Ei(r) * r, 0, r0) + q(lambda r: Eo(r) * r, r0, a)
    We_in, We_out = eps * q(lambda r: Ei(r) ** 2 * r, 0, r0), q(lambda r: Eo(r) ** 2 * r, r0, a)
    Hv = q(lambda r: dEi(r) ** 2 * r, 0, r0) + q(lambda r: dEo(r) ** 2 * r, r0, a)
    # G = omega mu0 int |H|^2 / oint |H_t|^2, H_phi ~ dE_z/dr: volume L Hv, barrel L a dEo(a)^2, caps 2 Hv
    G = Z0 * k * L * Hv / (L * a * dEo(a) ** 2 + 2 * Hv)
    return {"k": k, "f_Hz": C_LIGHT * k / (2 * math.pi), "C": 2 * I1 ** 2 / (a * a * (We_in + We_out)),
            "fe_rod": We_in / (We_in + We_out), "G_ohm": float(G)}


def floating_rod_design(a: float = 0.1, L: float = 0.2, rod_radius: float = 0.015, gap: float = 0.02, *,
                        t: float | None = None, h: float | None = None, rod_size: float | None = None, order: int = 2,
                        wall_material: dict | None = None, Bz: float = 1.0, modes: int = 4, families=("TM",),
                        target_f_Hz: float | None = None, domain_radius: float = 0.5):
    """Axisymmetric (2D) pillbox (radius a, length L, centred on z = 0) with a floating conducting rod of radius
    ``rod_radius`` on the axis and end gaps ``gap`` at both caps: the reference for the 3D floating tuning rod
    (cylinder_design_3d(rod={"radius": r, "gap": g})). ``h``: element size in the cavity; ``rod_size``: element size
    at the rod surface (tips and edges; default h/4). V excludes the rod, as in 3D."""
    t = t or a / 10
    h = h or a / 40
    rod_size = rod_size or h / 4
    ra, g = rod_radius, gap
    wall = {"union": [{"rect": {"r": [a, a + t], "z": [-L / 2 - t, L / 2 + t]}},
                      {"rect": {"r": [0.0, a], "z": [L / 2, L / 2 + t]}},
                      {"rect": {"r": [0.0, a], "z": [-L / 2 - t, -L / 2]}}]}
    cav = {"union": [{"rect": {"r": [ra, a], "z": [-L / 2, L / 2]}},
                     {"rect": {"r": [0.0, ra], "z": [-L / 2, -L / 2 + g]}},
                     {"rect": {"r": [0.0, ra], "z": [L / 2 - g, L / 2]}}]}
    regions = [{"name": "cavity", "shape": cav, "material": "vacuum", "mesh_size": h},
               {"name": "wall", "shape": wall, "material": "wall_cu"},
               {"name": "rod", "shape": {"rect": {"r": [0.0, ra], "z": [-L / 2 + g, L / 2 - g]}},
                "material": "wall_cu", "surface_size": rod_size}]
    return _design("floating_rod", regions, wall_material=wall_material, fill=None, h=h, order=order, Bz=Bz,
                   modes=modes, families=families, target_f_Hz=target_f_Hz, interior_point=((a + ra) / 2, 0.0),
                   domain_radius=domain_radius, surface_size=None)


def eta_tm010_te01p(a: float, L: float, p: int = 1) -> float:
    """Heterodyne overlap eta = |int E_TM010 . B_TE01p| / sqrt(int |E|^2 int |B|^2) of a closed vacuum cylinder
    (TM010: E_z = J0(chi01 r/a); TE01p: H_z = J0(kc r) sin(beta z'), H_r = -(beta/kc) J1(kc r) cos(beta z'),
    kc = chi'01/a, beta = p pi/L, z' in [0, L])."""
    from scipy.integrate import quad
    from scipy.special import j0, j1, jn_zeros, jnp_zeros

    k1, kc, beta = jn_zeros(0, 1)[0] / a, jnp_zeros(0, 1)[0] / a, p * math.pi / L
    q = lambda f: quad(f, 0, a, epsabs=0, epsrel=1e-13, limit=200)[0]  # noqa: E731
    EB = 2 * math.pi * q(lambda r: j0(k1 * r) * j0(kc * r) * r) * (1 - (-1) ** p) * L / (p * math.pi)
    EE = 2 * math.pi * L * q(lambda r: j0(k1 * r) ** 2 * r)
    BB = 2 * math.pi * (L / 2) * (q(lambda r: j0(kc * r) ** 2 * r) + (beta / kc) ** 2 * q(lambda r: j1(kc * r) ** 2 * r))
    return abs(EB) / math.sqrt(EE * BB)


def box_design_3d(a: float = 0.2, b: float = 0.15, d: float = 0.1, *, h: float | None = None, order: int = 2,
                  modes: int = 6, target_f_Hz: float | None = None, eigensolver: dict | None = None,
                  B_T=(0.0, 0.0, 1.0), wall_material: dict | None = None, losses: bool = True,
                  mode_overlaps: bool = False, surface_size: float | None = None):
    """3D closed rectangular cavity 0 < x < a, 0 < y < b, 0 < z < d (PEC domain wall), uniform B0 = ``B_T``."""
    h = h or min(a, b, d) / 5
    box = {"box": {"min": [0.0, 0.0, 0.0], "max": [a, b, d]}}
    regions = [{"name": "cavity", "material": "vacuum", "solid": box}]
    return _design3d("box3d", regions, box, (0.437 * a, 0.519 * b, 0.471 * d), h=h, order=order, modes=modes,
                     target_f_Hz=target_f_Hz, eigensolver=eigensolver, symmetry=None, B_T=B_T,
                     wall_material=wall_material, losses=losses, mode_overlaps=mode_overlaps,
                     surface_size=surface_size, remove_pec=False)


__all__ += ["cylinder_modes_3d", "box_modes", "cylinder_design_3d", "box_design_3d", "eta_tm010_te01p",
            "floating_rod_design", "dielectric_rod_tm_mode"]

# Tuning sweeps with mode tracking for the 2D, cross-section and 3D backends (beta);
# implementation in axionfem.cavity_track.
from .cavity_track import (Branch, ModeStep, Tracking, adaptive_track, mode_step, probe_grid_2d,  # noqa: E402
                           probe_points_3d, sweep, track, two_level_steps)

__all__ += ["sweep", "track", "adaptive_track", "mode_step", "ModeStep", "Tracking", "Branch", "probe_grid_2d",
            "probe_points_3d", "two_level_steps"]
