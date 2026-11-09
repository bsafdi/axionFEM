"""Analytic validation of the 2D imported-mesh full-wave engine (cpp/src/fw2d.cpp) and its boundary conditions.
beta (COMSOL 2D import, docs/comsol_import.md "2D models").

No COMSOL file with 2D geometry, impedance/transition/scattering boundaries or PML is distributed with axionFEM,
so these features are validated against closed forms on structured meshes written in the importer's format
(axionfem.fw2d.mapped_mesh / write_mesh: second-order triangles, mid nodes on the exact curves):

  coax      axisymmetric TEM line a < r < b, 0 < z < l, port across z = 0 (TM family):
            PEC short Z = -i Z0 tan(k l); plane scattering BC or PML: Z = Z0 = eta ln(b/a) / (2 pi);
            impedance (copper) walls: attenuation alpha = R_s (1/a + 1/b) / (2 eta ln(b/a)).
  bicone    axisymmetric biconical line (cones at theta0 and pi - theta0, r0 < rho < R), port on rho = r0:
            spherical TEM, Z = (eta / pi) ln cot(theta0 / 2) with a scattering BC or a spherical PML at R.
  plates    planar parallel plates (gap h, thickness D), port across x = 0: Z = eta h / D (SBC / PML).
  wedge     planar wedge |phi| < alpha, r0 < rho < R, port on rho = r0: outgoing cylindrical wave, Z from Hankel
            functions; cylindrical PML (exact) and cylindrical scattering BC (exact 1D radial solution with the
            Robin condition).
  pillbox   axisymmetric closed cylinder: TM010 / TE011 frequencies; impedance walls: perturbative Q vs
            G / R_s (TM010: G = p01 eta / (2 (1 + a/d))), and the driven Q (resonance fit of Z through a wall gap).

    python benchmarks/fw2d_validation.py [--out benchmarks/records/fw2d_validation.json] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

C0 = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = MU0 * C0
SIGMA_CU = 5.8e7


def _design(name, symmetry, regions, mesh, info, fw, pickup=None, materials=None):
    fw = dict(fw)
    order = fw.pop("_order", 2)
    d = {"schema": "axionfem.design/1", "name": name, "symmetry": symmetry,
         "regions": [{"name": r, "material": m, "shape": {"imported": {}}} for r, m in regions],
         "fullwave": {"boundary": "pec", "rf_only": True, **fw},
         "mesh": {"order": order, "file": str(mesh), "info_file": str(info)}}
    if pickup:
        d["pickup"] = pickup
    if materials:
        d["materials"] = materials
    return d


def interior_edges(pts, tri, pred):
    """line3 edges [a, b, mid] of the triangles whose two vertices satisfy pred(x1, x2) (deduplicated)."""
    seen, out = set(), []
    for t in tri:
        for a, b, m in ((t[0], t[1], t[3]), (t[1], t[2], t[4]), (t[2], t[0], t[5])):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            if pred(*pts[a]) and pred(*pts[b]):
                seen.add(key)
                out.append([a, b, m])
    return np.array(out, dtype=int).reshape(-1, 3)


def build(case: str, wd: Path, *, n=(8, 40), order=2, termination="scattering", walls="pec", length=None,
          pml_cells=20, f=None, extra=None):
    """Write mesh + info + design for one case; returns (design dict, meta dict)."""
    from axionfem.fw2d import mapped_mesh, write_mesh

    wd.mkdir(parents=True, exist_ok=True)
    meta = {"case": case, "termination": termination, "walls": walls, "n": list(n), "order": order}
    materials = {"copper_ibc": {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": SIGMA_CU}}
    pml = termination == "pml"
    if case in ("coax", "plates"):
        a, b = (0.01, 0.023) if case == "coax" else (0.0, 0.02)
        l = length or 0.3
        dp = 0.1 if pml else 0.0
        nz = n[1] + (pml_cells if pml else 0)
        L = l + dp
        vl = n[1] / nz  # element boundary at the end of the line (PML interface)

        def zz(v):
            return np.where(v <= vl, l * v / vl, l + dp * (v - vl) / max(1 - vl, 1e-300))

        two = case == "plates" and bool((extra or {}).get("two_sided"))  # PML at both ends, port at x = 0
        if case == "coax":
            mp = lambda u, v: (a + (b - a) * u, zz(v))  # noqa: E731
        elif two:
            # x from -(l + dp) to l + dp: mirror of zz; the PML regions at both ends are ONE layer entry
            nz = 2 * nz
            mp = lambda u, v: (np.where(v >= 0.5, zz(2 * v - 1), -zz(1 - 2 * v)), b * u)  # noqa: E731
        else:
            mp = lambda u, v: (zz(v), b * u)  # noqa: E731   (x along the line, y across the gap)
        reg_uv = (lambda uc, vc: int(abs(2 * vc - 1) > vl)) if two else (lambda uc, vc: int(vc > vl))
        pts, tri, reg, edges, side = mapped_mesh(mp, n[0], nz, region_uv=reg_uv)
        # sides: 0 v=0 (port), 1 u=1 (outer wall / top plate), 2 v=1 (end), 3 u=0 (inner wall / bottom plate)
        attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004}
        groups = [{"attr": 10001, "name": "lport1", "type": "port"}]
        wtype = {"type": "pec"} if walls == "pec" else {"type": "impedance", "material": "copper_ibc"}
        groups += [{"attr": 10002, "name": "wall_outer", **wtype}, {"attr": 10004, "name": "wall_inner", **wtype}]
        if termination == "pec" or pml:
            groups.append({"attr": 10003, "name": "end", "type": "pec"})
        else:
            groups.append({"attr": 10003, "name": "end", "type": "scattering", "wave": "plane"})
        # walls next to the PML: keep PEC in the PML (an impedance wall inside the layer is fine but unphysical)
        if pml and walls != "pec":
            edge_attr = []
            for e, s in zip(edges, side):
                x = pts[e[:2]].mean(axis=0)
                inl = (x[1] if case == "coax" else x[0]) > l + 1e-12
                edge_attr.append(10005 if (s in (1, 3) and inl) else attr[s])
            groups.append({"attr": 10005, "name": "wall_pml", "type": "pec"})
        else:
            edge_attr = [attr[s] for s in side]
        edges, edge_attr = list(edges), list(edge_attr)
        if two:  # the port is the interior edge chain at x = 0; both ends are PEC behind the layers
            edge_attr = [10003 if a_ == 10001 else a_ for a_ in edge_attr]
            for e in interior_edges(pts, tri, lambda x, y: abs(x) < 1e-12):
                edges.append(e)
                edge_attr.append(10001)
        sheet = (extra or {}).get("sheet")
        if sheet:  # a thin resistive layer across the line at x = sheet["x"]
            for e in interior_edges(pts, tri, lambda x, y: abs(x - sheet["x"]) < 1e-12):
                edges.append(e)
                edge_attr.append(10006)
            materials["film"] = {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": sheet["sigma"]}
            groups.append({"attr": 10006, "name": "film", "type": "sheet", "material": "film", "thickness_m": sheet["d"]})
        edges = np.array(edges)
        regions = [("field", "vacuum")] + ([("pml", "vacuum")] if pml else [])
        fw = {"_order": order}
        if pml:
            fw["pml"] = [{"regions": ["pml"], "type": "cylindrical" if case == "coax" else "cartesian"}]
        if (extra or {}).get("source"):
            fw["source"] = {"kind": "uniform", "B_T": [0.0, extra["source"], 0.0]}
            fw["source_regions"] = ["field"]
        meta.update(a=a, b=b, l=l, two_sided=two)
    elif case in ("bicone", "wedge"):
        th0 = math.radians(30.0) if case == "bicone" else math.radians(20.0)
        r0, R = 0.05, 0.6
        dp = 0.25 if pml else 0.0
        nrad = n[1] + (pml_cells if pml else 0)
        Rt = R + dp
        vR = n[1] / nrad  # element boundary at rho = R

        def radius(v):
            return np.where(v <= vR, r0 + (R - r0) * v / vR, R + dp * (v - vR) / max(1 - vR, 1e-300))
        if case == "bicone":
            # u: theta from th0 to pi - th0; v: rho from r0 to Rt; (r, z) = rho (sin, cos)
            def mp(u, v):
                th = th0 + (math.pi - 2 * th0) * u
                rho = radius(v)
                return rho * np.sin(th), rho * np.cos(th)
        else:
            def mp(u, v):
                ph = -th0 + 2 * th0 * u
                rho = radius(v)
                return rho * np.cos(ph), rho * np.sin(ph)
        pts, tri, reg, edges, side = mapped_mesh(mp, n[0], nrad, region_uv=lambda uc, vc: int(vc > vR))
        attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004}
        groups = [{"attr": 10001, "name": "lport1", "type": "port"},
                  {"attr": 10002, "name": "cone_2", "type": "pec"}, {"attr": 10004, "name": "cone_1", "type": "pec"}]
        if pml or termination == "pec":
            groups.append({"attr": 10003, "name": "end", "type": "pec"})
        else:
            groups.append({"attr": 10003, "name": "end", "type": "scattering",
                           "wave": "spherical" if case == "bicone" else "cylindrical", "center": [0.0, 0.0]})
        edge_attr = [attr[s] for s in side]
        regions = [("field", "vacuum")] + ([("pml", "vacuum")] if pml else [])
        fw = {"_order": order}
        if pml:
            fw["pml"] = [{"regions": ["pml"], "type": "spherical" if case == "bicone" else "cylindrical",
                          "center": [0.0, 0.0]}]
        meta.update(theta0=th0, r0=r0, R=R)
    elif case == "pillbox":
        a, dlen = 0.1, 0.15
        gap = (extra or {}).get("gap", 0.0)          # driven: a wall gap (port) at r = a, z in [z0, z0 + gap]
        mp = lambda u, v: (a * u, dlen * v)  # noqa: E731
        inner = (extra or {}).get("inner", 0.0)  # a conductor region r < inner (coaxial cavity)
        pts, tri, reg, edges, side = mapped_mesh(mp, n[0], n[1], region_uv=lambda uc, vc: int(uc < inner / a))
        wtype = {"type": "pec"} if walls == "pec" else {"type": "impedance", "material": "copper_ibc"}
        groups = [{"attr": 10001, "name": "axis", "type": "axis"}, {"attr": 10002, "name": "wall", **wtype}]
        edge_attr = []
        z0 = 0.5 * dlen - 0.5 * gap
        for e, s in zip(edges, side):
            x = pts[e[:2]].mean(axis=0)
            if s == 3:
                edge_attr.append(10001)
            elif s == 1 and gap > 0 and z0 - 1e-12 < x[1] < z0 + gap + 1e-12:
                edge_attr.append(10003)
            else:
                edge_attr.append(10002)
        if gap > 0:
            groups.append({"attr": 10003, "name": "lport1", "type": "port"})
        regions = [("cavity", "vacuum")] + ([("inner", "copper")] if inner > 0 else [])
        fw = {"_order": order}
        meta.update(a=a, d=dlen, gap=gap, inner=inner)
    elif case == "radial":
        # axisymmetric radial line between plates z = 0 and z = h, r0 < r < R (+ PML), port at r = r0 (TM, E_z):
        # cylindrical outgoing wave; E_t on the end r = R is axial (curvature term + nu / (2 R))
        r0, R, h = 0.05, 0.6, 0.05
        dp = 0.25 if pml else 0.0
        nrad = n[1] + (pml_cells if pml else 0)
        vR = n[1] / nrad
        rr = lambda v: np.where(v <= vR, r0 + (R - r0) * v / vR, R + dp * (v - vR) / max(1 - vR, 1e-300))  # noqa: E731
        mp = lambda u, v: (rr(v), h * u)  # noqa: E731
        pts, tri, reg, edges, side = mapped_mesh(mp, n[0], nrad, region_uv=lambda uc, vc: int(vc > vR))
        attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004}
        groups = [{"attr": 10001, "name": "lport1", "type": "port"},
                  {"attr": 10002, "name": "top", "type": "pec"}, {"attr": 10004, "name": "bottom", "type": "pec"}]
        groups.append({"attr": 10003, "name": "end", "type": "pec"} if (pml or termination == "pec") else
                      {"attr": 10003, "name": "end", "type": "scattering", "wave": "cylindrical"})
        edge_attr = [attr[s_] for s_ in side]
        regions = [("field", "vacuum")] + ([("pml", "vacuum")] if pml else [])
        fw = {"_order": order}
        if pml:
            fw["pml"] = [{"regions": ["pml"], "type": "cylindrical"}]
        meta.update(r0=r0, R=R, h=h)
    elif case == "septum":
        # planar PEC rectangle a x b with a PEC septum at x = a/2 (interior conducting sheet): eigenmodes
        a, b = 0.2, 0.1
        pts, tri, reg, edges, side = mapped_mesh(lambda u, v: (a * u, b * v), n[0], n[1])
        groups = [{"attr": 10001, "name": "wall", "type": "pec"}, {"attr": 10002, "name": "septum", "type": "pec"}]
        edges, edge_attr = list(edges), [10001] * len(edges)
        for e in interior_edges(pts, tri, lambda x, y: abs(x - a / 2) < 1e-12):
            edges.append(e)
            edge_attr.append(10002)
        edges = np.array(edges)
        regions = [("cavity", "vacuum")]
        fw = {"_order": order}
        meta.update(a=a, b=b)
    else:
        raise ValueError(case)
    used = sorted(set(edge_attr))
    groups = [g for g in groups if g["attr"] in used]
    mesh = wd / f"{case}.msh"
    write_mesh(mesh, pts, tri, 2 + reg, edges, edge_attr)
    info = wd / f"{case}_mesh_info.json"
    info.write_text(json.dumps({"boundary_groups": groups}))
    sym = "planar" if case in ("plates", "wedge", "septum") else "axisymmetric"
    pickup = {"type": "port", "port_boundary": "lport1"} if any(g["type"] == "port" for g in groups) else None
    d = _design(case, sym, regions, mesh, info, fw, pickup, materials)
    if sym == "planar":
        d["fullwave"]["thickness_m"] = 0.05
        meta["D"] = 0.05
    return d, meta


def solve(d, wd, frequencies=None, modes=0, families=None):
    import axionfem as af

    d = json.loads(json.dumps(d))
    if frequencies is not None:
        d["fullwave"]["frequencies"] = list(frequencies)
    if modes:
        d["fullwave"]["modes"] = modes
        if families:
            d["fullwave"]["families"] = families
    return af.Design(d).solve(workdir=str(wd), verbose=0)


# ------------------------------------------------------------------------------------------------ closed forms
def coax_Z0(a, b, eta=ETA0):
    return eta / (2 * math.pi) * math.log(b / a)


def rs_copper(f):
    return math.sqrt(math.pi * f * MU0 / SIGMA_CU)


def coax_alpha(a, b, f):
    return rs_copper(f) * (1 / a + 1 / b) / (2 * ETA0 * math.log(b / a))


def bicone_Z(th0):
    return ETA0 / math.pi * math.log(1 / math.tan(th0 / 2))


def wedge_Z(f, alpha, r0, D, R=None):
    """Planar wedge, outgoing cylindrical wave (R None) or with the first-order cylindrical scattering condition at
    rho = R (H_z = (1/eta - i/(2 omega mu R)) E_phi there)."""
    from scipy.special import hankel1, hankel2, h1vp, h2vp

    k = 2 * math.pi * f / C0
    w = 2 * math.pi * f
    if R is None:
        A, B = 1.0, 0.0
    else:
        # H = A H1 + B H2, E_phi = (1/(i omega eps)) dH/drho
        g = 1 / ETA0 - 1j / (2 * w * MU0 * R)
        c1 = hankel1(0, k * R) - g * k * h1vp(0, k * R) / (1j * w * EPS0)
        c2 = hankel2(0, k * R) - g * k * h2vp(0, k * R) / (1j * w * EPS0)
        A, B = 1.0, -c1 / c2
    H = A * hankel1(0, k * r0) + B * hankel2(0, k * r0)
    dH = k * (A * h1vp(0, k * r0) + B * h2vp(0, k * r0))
    E = dH / (1j * w * EPS0)
    # H_z(r0) = -I/D (outward normal -rho), V = -int t.E dl = -2 alpha r0 E_phi  ->  Z = 2 alpha r0 D E / H... sign:
    return -2 * alpha * r0 * E / H / D * (-1)


def radial_Z(f, r0, h, R=None):
    """Axisymmetric radial line (TM: E_z, H_phi), port across the gap h at r = r0: Z = -(h/(2 pi r0)) E_z / H_phi with
    H_phi = A H1(kr) + B H1^(2)(kr), E_z = i eta (A H0 + B H0^(2)); outgoing (R None) or the first-order cylindrical
    scattering condition at R: i omega H_phi = beta E_z, beta = -i omega/eta + 1/(2 mu R)."""
    from scipy.special import hankel1, hankel2

    k, w = 2 * math.pi * f / C0, 2 * math.pi * f
    if R is None:
        A, B = 1.0, 0.0
    else:
        beta = -1j * w / ETA0 + 1 / (2 * MU0 * R)
        c1 = 1j * w * hankel1(1, k * R) - beta * 1j * ETA0 * hankel1(0, k * R)
        c2 = 1j * w * hankel2(1, k * R) - beta * 1j * ETA0 * hankel2(0, k * R)
        A, B = 1.0, -c1 / c2
    Ez = 1j * ETA0 * (A * hankel1(0, k * r0) + B * hankel2(0, k * r0))
    Hp = A * hankel1(1, k * r0) + B * hankel2(1, k * r0)
    return -(h / (2 * math.pi * r0)) * Ez / Hp


def line_Z(Z0, ZL, k, L):
    """Input impedance of a line of length L terminated by ZL (e^{-i omega t})."""
    t = math.tan(k * L)
    return Z0 * (ZL - 1j * Z0 * t) / (Z0 - 1j * ZL * t)


def pillbox_TE011_Q(a, d, f):
    """Pozar's TE_011 copper Q of a closed cylinder (all walls R_s)."""
    p = 3.831705970207512
    k = 2 * math.pi * f / C0
    beta = math.pi / d
    return (k * a) ** 3 * ETA0 * a * d / (4 * p * p * rs_copper(f)) / (a * d / 2 + (beta * a * a / p) ** 2)


def pillbox_TM010(a, d):
    p01 = 2.404825557695773
    f = p01 * C0 / (2 * math.pi * a)
    G = p01 * ETA0 / (2 * (1 + a / d))
    return f, G


def pillbox_TE011(a, d):
    p11 = 3.831705970207512
    k = math.hypot(p11 / a, math.pi / d)
    return k * C0 / (2 * math.pi)


# ------------------------------------------------------------------------------------------------ runs
def run(out: Path | None, quick: bool = False):
    rec = {"schema": "axionfem.fw2d_validation/1", "beta": True, "cases": []}
    tmp = Path(tempfile.mkdtemp(prefix="fw2d_val_"))
    nmul = 1 if quick else 2

    def add(**kw):
        rec["cases"].append(kw)
        print(json.dumps(kw, default=float))

    # coax
    f = 3e8
    k = 2 * math.pi * f / C0
    for term in ("pec", "scattering", "pml"):
        d, m = build("coax", tmp / f"coax_{term}", n=(4 * nmul, 30 * nmul), termination=term)
        r = solve(d, tmp / f"coax_{term}", [f]).sweep()
        Z0 = coax_Z0(m["a"], m["b"])
        Zx = -1j * Z0 * math.tan(k * m["l"]) if term == "pec" else Z0
        add(case="coax", termination=term, f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=[Zx.real, Zx.imag],
            rel_err=abs(r["Z"][0] - Zx) / abs(Zx))
    # coax with copper walls, matched end: attenuation from the power that reaches the end
    d, m = build("coax", tmp / "coax_ibc", n=(4 * nmul, 30 * nmul), termination="scattering", walls="impedance")
    res = solve(d, tmp / "coax_ibc", [f])
    row = res.fullwave["sweep"][0]["power_W_per_A2"]
    Pin, Pend = row["input"], row["boundary_by_group"]["end"]
    alpha = -math.log(Pend / Pin) / (2 * m["l"])
    add(case="coax_ibc", f_Hz=f, alpha_fem=alpha, alpha_exact=coax_alpha(m["a"], m["b"], f),
        rel_err=abs(alpha / coax_alpha(m["a"], m["b"], f) - 1), balance_rel=row["balance_rel"])
    # bicone
    f = 1e9
    for term in ("scattering", "pml"):
        d, m = build("bicone", tmp / f"bicone_{term}", n=(6 * nmul, 20 * nmul), termination=term)
        r = solve(d, tmp / f"bicone_{term}", [f]).sweep()
        Zx = bicone_Z(m["theta0"])
        add(case="bicone", termination=term, f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=Zx,
            rel_err=abs(r["Z"][0] - Zx) / Zx)
    # plates
    f = 3e8
    for term in ("scattering", "pml"):
        d, m = build("plates", tmp / f"plates_{term}", n=(4 * nmul, 30 * nmul), termination=term)
        r = solve(d, tmp / f"plates_{term}", [f]).sweep()
        Zx = ETA0 * m["b"] / m["D"]
        add(case="plates", termination=term, f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=Zx,
            rel_err=abs(r["Z"][0] - Zx) / Zx)
    # wedge
    f = 1e9
    for term in ("scattering", "pml"):
        d, m = build("wedge", tmp / f"wedge_{term}", n=(6 * nmul, 20 * nmul), termination=term)
        r = solve(d, tmp / f"wedge_{term}", [f]).sweep()
        Zx = wedge_Z(f, m["theta0"], m["r0"], m["D"], None if term == "pml" else m["R"])
        Zinf = wedge_Z(f, m["theta0"], m["r0"], m["D"])
        add(case="wedge", termination=term, f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=[Zx.real, Zx.imag],
            Z_outgoing=[Zinf.real, Zinf.imag], rel_err=abs(r["Z"][0] - Zx) / abs(Zx))
    # pillbox eigenmodes + perturbative Q
    d, m = build("pillbox", tmp / "pill", n=(10 * nmul, 15 * nmul), walls="impedance")
    m_p = m
    res = solve(d, tmp / "pill", modes=2, families=["TM", "TE"])
    f010, G = pillbox_TM010(m["a"], m["d"])
    tm = res.modes("TM")[0]
    te = res.modes("TE")[0]
    Qx = G / rs_copper(f010)
    add(case="pillbox_TM010", f_Hz=tm["f_Hz"], f_exact=f010, rel_err_f=abs(tm["f_Hz"] / f010 - 1), Q_wall=tm["Q_wall"],
        Q_exact=Qx, rel_err_Q=abs(tm["Q_wall"] / Qx - 1))
    fte = pillbox_TE011(m["a"], m["d"])
    add(case="pillbox_TE011", f_Hz=te["f_Hz"], f_exact=fte, rel_err_f=abs(te["f_Hz"] / fte - 1))
    # coaxial cavity: a conductor region r < a/2 (PEC elements): TM_00p (TEM) modes f = p c / (2 d); the dual H
    # family keeps the natural condition on the conductor surface (and its static G/r mode is dropped)
    d, m = build("pillbox", tmp / "coaxcav", n=(10 * nmul, 15 * nmul), extra={"inner": 0.05})
    res = solve(d, tmp / "coaxcav", modes=1, families=["TM"])
    fx = C0 / (2 * m["d"])
    add(case="coaxial_cavity_TEM1", f_Hz=res.modes("TM")[0]["f_Hz"], f_exact=fx,
        rel_err_f=abs(res.modes("TM")[0]["f_Hz"] / fx - 1),
        static_mode_constraints=res.fullwave["eigenmodes"]["TM"]["static_mode_constraints"])
    # driven Q through a small wall gap (open-circuit current drive: no port loading)
    # The gap (length g on the side wall, where the TM010 wall current is uniform in z) is lossless: the
    # perturbative Q of the same geometry is Q_wall / (1 - (g/d) / (1 + a/d)).
    gap = 0.15 / (15 * nmul)
    d, m = build("pillbox", tmp / "pill_drv", n=(10 * nmul, 15 * nmul), walls="impedance", extra={"gap": gap})
    f0 = tm["f_Hz"]
    bw = f0 / Qx
    fs = f0 + bw * np.linspace(-2, 2, 9)
    res = solve(d, tmp / "pill_drv", fs)
    r = res.sweep()
    gap = res.fullwave["ports"][0]["length_m"]  # the edges actually selected
    fit = fit_resonance(r["f_Hz"], r["Z"])
    Qexp = tm["Q_wall"] / (1 - (gap / m["d"]) / (1 + m["a"] / m["d"]))
    add(case="pillbox_driven_Q", gap_m=gap, f0_fit=fit["f0"], Q_fit=fit["Q"], Q_perturbative=tm["Q_wall"],
        Q_perturbative_gap_corrected=Qexp, f0_perturbative=f0, rel_diff=abs(fit["Q"] / Qexp - 1), fit_residual=fit["residual"])
    # TE011 wall Q (out-of-plane family: wall current by consistent-flux recovery)
    add(case="pillbox_TE011_Q", Q_wall=te.get("Q_wall"), Q_exact=pillbox_TE011_Q(m_p["a"], m_p["d"], te["f_Hz"]),
        rel_err_Q=abs(te["Q_wall"] / pillbox_TE011_Q(m_p["a"], m_p["d"], te["f_Hz"]) - 1))
    # thin resistive sheet across a matched parallel-plate line (sheet boundary, E_t continuous)
    f = 3e8
    k = 2 * math.pi * f / C0
    sigma, dfilm = 1e4, 1e-6
    d, m = build("plates", tmp / "plates_sheet", n=(4 * nmul, 30 * nmul), termination="scattering",
                 extra={"sheet": {"x": 0.15, "sigma": sigma, "d": dfilm}})
    r = solve(d, tmp / "plates_sheet", [f]).sweep()
    Z0 = ETA0 * m["b"] / m["D"]
    ec = EPS0 + 1j * sigma / (2 * math.pi * f)
    km, etam = 2 * math.pi * f * np.sqrt(MU0 * ec), np.sqrt(MU0 / ec)
    G = -2j / etam * np.tan(km * dfilm / 2) * m["D"] / m["b"]
    Zx = line_Z(Z0, Z0 / (1 + G * Z0), k, 0.15)
    add(case="plates_sheet", f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=[Zx.real, Zx.imag],
        rel_err=abs(r["Z"][0] - Zx) / abs(Zx))
    # source overlap Psi: uniform B0 = B y on the matched line, V_oc closed form (sign included)
    Bs = 1.0
    d, m = build("plates", tmp / "plates_psi", n=(4 * nmul, 30 * nmul), termination="scattering", extra={"source": Bs})
    r = solve(d, tmp / "plates_psi", [f]).sweep()
    w = 2 * math.pi * f
    Psix = -(Bs * m["D"] * Z0 / (1j * w)) * (np.exp(1j * k * m["l"]) - 1) / (1j * k)
    add(case="plates_Psi", f_Hz=f, Psi=[r["Psi"][0].real, r["Psi"][0].imag], Psi_exact=[Psix.real, Psix.imag],
        rel_err=abs(r["Psi"][0] - Psix) / abs(Psix))
    # PML at both ends in one layer entry, port in the middle: two matched lines in parallel, Z = Z0 / 2
    d, m = build("plates", tmp / "plates_pml2", n=(4 * nmul, 30 * nmul), termination="pml", order=3,
                 extra={"two_sided": True})
    r = solve(d, tmp / "plates_pml2", [f]).sweep()
    add(case="plates_pml_two_sided", f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag], Z_exact=Z0 / 2,
        rel_err=abs(r["Z"][0] - Z0 / 2) / (Z0 / 2))
    # axisymmetric radial line: cylindrical scattering condition (E_t axial) and radial cylindrical PML
    f = 1e9
    for term, o in (("scattering", 2), ("pml", 3)):
        d, m = build("radial", tmp / f"radial_{term}", n=(4 * nmul, 30 * nmul), termination=term, order=o)
        r = solve(d, tmp / f"radial_{term}", [f]).sweep()
        Zx = radial_Z(f, m["r0"], m["h"], None if term == "pml" else m["R"])
        add(case="radial_line", termination=term, order=o, f_Hz=f, Z=[r["Z"][0].real, r["Z"][0].imag],
            Z_exact=[Zx.real, Zx.imag], rel_err=abs(r["Z"][0] - Zx) / abs(Zx))
    # interior PEC septum (dual H family: split DOFs) and its E family
    d, m = build("septum", tmp / "septum", n=(8 * nmul, 8 * nmul))
    res = solve(d, tmp / "septum", modes=5, families=["TE", "TM"])
    te_f = [x["f_Hz"] for x in res.modes("TE")]
    tm_f = [x["f_Hz"] for x in res.modes("TM")]
    h = m["a"] / 2
    exact_te = sorted([C0 / 2 * math.hypot(p / h, q / m["b"]) for p in range(0, 4) for q in range(0, 4) if p + q > 0])
    exact_te = sorted(exact_te + exact_te)[:5]              # two halves: every mode twice
    exact_tm = sorted([C0 / 2 * math.hypot(p / h, q / m["b"]) for p in range(1, 4) for q in range(1, 4)])
    exact_tm = sorted(exact_tm + exact_tm)[:5]
    add(case="septum_eigen", TE_f_Hz=te_f, TE_exact=exact_te, TM_f_Hz=tm_f, TM_exact=exact_tm,
        rel_err_TE=max(abs(a_ / b_ - 1) for a_, b_ in zip(te_f, exact_te)),
        rel_err_TM=max(abs(a_ / b_ - 1) for a_, b_ in zip(tm_f, exact_tm)),
        sheet_split_dofs=res.fullwave["eigenmodes"]["TE"]["sheet_split_dofs"])
    # PML layer resolution (the layer is strongly stretched, |s| ~ 1 + F lambda / d): cells across it and order
    for case, f, n in (("coax", 3e8, (4, 30)), ("bicone", 1e9, (6, 30)), ("wedge", 1e9, (6, 30))):
        rows = []
        for pc in ((10, 20) if quick else (10, 20, 40)):
            for o in (2, 3):
                d, m = build(case, tmp / f"pmlconv_{case}_{pc}_{o}", n=n, termination="pml", pml_cells=pc, order=o)
                Z = solve(d, tmp / f"pmlconv_{case}_{pc}_{o}", [f]).sweep()["Z"][0]
                Zx = (coax_Z0(m["a"], m["b"]) if case == "coax" else bicone_Z(m["theta0"]) if case == "bicone"
                      else wedge_Z(f, m["theta0"], m["r0"], m["D"]))
                rows.append({"pml_cells": pc, "order": o, "rel_err": abs(Z - Zx) / abs(Zx)})
        add(case=f"{case}_pml_convergence", f_Hz=f, rows=rows)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def fit_resonance(f, Z):
    """Series resonance of a current-driven wall gap: Z(w) = R + i X(w) with X linear near its zero w_z;
    Q = -(w_z / (2 R)) dX/dw (the series-resonator Q, = omega U / P at the zero for a small background)."""
    f = np.asarray(f, float)
    Z = np.asarray(Z)
    c = np.polyfit(f - f.mean(), Z.imag, 2)
    roots = np.roots(c)
    roots = roots[np.isreal(roots)].real
    fz = f.mean() + roots[np.argmin(np.abs(roots))]
    slope = np.polyval(np.polyder(c), fz - f.mean())
    R = float(np.interp(fz, f, Z.real))
    Q = -fz * slope / (2 * R)
    res = float(np.abs(np.polyval(c, f - f.mean()) - Z.imag).max() / np.abs(Z.imag).max())
    return {"f0": float(fz), "Q": float(Q), "R_ohm": R, "residual": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "records" / "fw2d_validation.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if run(a.out, a.quick) else 1)
