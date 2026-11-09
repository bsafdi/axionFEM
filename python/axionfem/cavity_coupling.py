"""Receiver coupling of a cavity mode through a coaxial antenna port (beta).

Beta: the first use of this module in a process emits one :class:`axionfem.BetaWarning`.

A coaxial line enters a cavity wall; its inner conductor continues into the cavity as an electric probe
(monopole) of insertion depth ``h``.  The line is lossless and its far end is a lumped port with the
reference impedance ``Z0`` of the receiver (default: the line's characteristic impedance, i.e. a matched
receiver).  The external quality factor Q_ext = omega U / P_ext, the coupling beta = Q0 / Q_ext and the loaded
Q_L = Q0 / (1 + beta) follow in two independent ways (docs/cavity/coupling.md):

(a) **driven**: the full-wave port impedance Z(omega) of the lossy structure (Leontovich walls in the operator)
    on a frequency grid near the mode, fitted by one pole plus a quadratic background.  The pole of Z is the
    open-port (unloaded) complex resonance, Q0 = Re w / (-2 Im w); the zero of Z + Z0 near it is the loaded
    resonance, Q_L likewise; beta = Q0 / Q_L - 1.

(b) **eigenmode expansion**: two lossless eigenmode solves of the same mesh, port open (omega_o) and port
    short-circuited (omega_s), the open-port mode's port voltage v = -int t.e dl (e real, int eps e^2 dV = 1) and
    its perturbative wall Q0.  Near omega_o the port impedance is Z = -i v^2 / (2 (omega_o - omega)) + i Y with
    the non-resonant reactance Y = v^2 / (2 (omega_o - omega_s)); a load Z0 gives
        Q_ext = omega_o (Z0^2 + Y^2) / (v^2 Z0),   loaded shift  d omega = -v^2 Y / (2 (Z0^2 + Y^2)).

Both agree to O(1/Q) (the background's variation across the resonance, the perturbative wall loss).  The
geometry is the axisymmetric pillbox with an on-axis probe (TM0np modes), solved by the 2D imported-mesh
engine (``Fw2DModel``, :mod:`axionfem.fw2d`) on a structured second-order mesh written here.

Time dependence e^{-i omega t}; Z = R - i omega L for an inductor (axionFEM convention).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._beta import warn_beta

C0 = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = MU0 * C0
P01 = 2.404825557695773  # first zero of J0

__all__ = ["ProbePillbox", "probe_pillbox_design", "coax_Z0", "solve_probe_pillbox", "expansion_coupling",
           "fit_port_resonance", "driven_coupling", "loaded_port_power", "coupling_study", "one_port_coupling",
           "sweep_grid", "line_resonances", "coupling_flags",
           "probe_coupling_estimates",
           "to_haloscope_cavity"]


def coax_Z0(a: float, b: float, eps_r: float = 1.0) -> float:
    """Characteristic impedance of a coax filled with a dielectric eps_r: eta0 ln(b/a) / (2 pi sqrt(eps_r))."""
    return ETA0 / (2 * math.pi * math.sqrt(eps_r)) * math.log(b / a)


# ------------------------------------------------------------------------------------------------ geometry + mesh
@dataclass
class ProbePillbox:
    """Pillbox (radius R, length L, z in [0, L]) with a coax (inner radius a, outer b, z in [-coax_len, 0]) on
    the axis through the bottom plate; the inner conductor protrudes to z = h (h = 0: flush).  The coax
    (z < 0) is filled with a lossless dielectric ``coax_eps_r`` that ends flush with the plate.

    Walls: the cavity walls and the probe are Leontovich (impedance) walls of conductivity ``sigma_S_m``
    (``probe_lossy=False``: the probe is PEC); the coax walls are PEC (lossless line).  ``sigma_S_m=None``:
    every wall PEC (no losses; eigenmodes only).

    Mesh: tensor grid graded towards r = a, b and z = 0, h (the re-entrant corners) with element sizes from
    ``h_min`` (default min(a, b - a)/3; also at the port plane z = -coax_len) growing by ``growth`` per element up to ``h_max`` (default R/12), all
    divided by ``refine``; second-order triangles."""

    R: float = 0.1
    L: float = 0.15
    a: float = 0.002
    b: float = 0.0046
    h: float = 0.01
    coax_len: float = 0.03
    coax_eps_r: float = 1.0
    sigma_S_m: float | None = 5.8e7
    probe_lossy: bool = True
    B_T: float = 1.0
    order: int = 3
    refine: float = 1.0
    h_min: float | None = None
    h_max: float | None = None
    growth: float = 0.25

    def __post_init__(self):
        warn_beta("axionfem.cavity_coupling", stacklevel=4)
        if not (0 < self.a < self.b < self.R):
            raise ValueError(f"ProbePillbox: need 0 < a < b < R, got a={self.a}, b={self.b}, R={self.R}")
        if not (0 <= self.h < self.L):
            raise ValueError(f"ProbePillbox: need 0 <= h < L, got h={self.h}, L={self.L}")
        if self.coax_len <= 0:
            raise ValueError("ProbePillbox: coax_len must be > 0")
        if self.coax_eps_r < 1.0:
            raise ValueError("ProbePillbox: coax_eps_r must be >= 1")

    @property
    def Z_c(self) -> float:
        return coax_Z0(self.a, self.b, self.coax_eps_r)

    @property
    def cavity_volume(self) -> float:
        """Volume of the cavity region (z > 0), probe excluded."""
        return math.pi * self.R ** 2 * self.L - math.pi * self.a ** 2 * self.h

    def f_tm010_empty(self) -> float:
        return P01 * C0 / (2 * math.pi * self.R)


def _graded(x0: float, x1: float, feats, hmin: float, hmax: float, g: float) -> np.ndarray:
    """Nodes on [x0, x1] with spacing ~ min(hmax, hmin + g * distance to the nearest feature)."""
    xs = np.linspace(x0, x1, 4001)
    d = np.min(np.abs(xs[:, None] - np.asarray(list(feats), float)[None, :]), axis=1) if feats else np.full_like(xs, np.inf)
    size = np.minimum(hmax, hmin + g * d)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (1 / size[1:] + 1 / size[:-1]) * np.diff(xs))])
    n = max(1, int(math.ceil(phi[-1] - 1e-9)))
    return np.interp(np.linspace(0, phi[-1], n + 1), phi, xs)


def _axis_nodes(breaks, feats, hmin, hmax, g):
    out = [breaks[0]]
    for x0, x1 in zip(breaks[:-1], breaks[1:]):
        out.extend(_graded(x0, x1, feats, hmin, hmax, g)[1:])
    return np.array(out)


def probe_pillbox_mesh(geo: ProbePillbox):
    """Structured second-order triangle mesh.  Returns (pts, tri6, tri_region (0 cavity, 1 coax), edges (k, 3),
    edge_group names (k,), meta)."""
    s = geo.refine
    hmin = (geo.h_min if geo.h_min is not None else min(geo.a, geo.b - geo.a) / 3.0) / s
    hmax = (geo.h_max if geo.h_max is not None else geo.R / 12.0) / s
    rb = [0.0, geo.a, geo.b, geo.R]
    zb = [-geo.coax_len, 0.0] + ([geo.h] if geo.h > 0 else []) + [geo.L]
    rn = _axis_nodes(rb, [geo.a, geo.b], hmin, hmax, geo.growth)
    zn = _axis_nodes(zb, [-geo.coax_len, 0.0, geo.h], hmin, hmax, geo.growth)   # the port plane too (v)
    nr, nz = len(rn) - 1, len(zn) - 1
    rc = 0.5 * (rn[1:] + rn[:-1])
    zc = 0.5 * (zn[1:] + zn[:-1])

    def kept(i, j):
        if i < 0 or j < 0 or i >= nr or j >= nz:
            return False
        r, z = rc[i], zc[j]
        if r < geo.a and z < geo.h:      # inner conductor (coax) and probe
            return False
        if z < 0 and r > geo.b:          # outer conductor
            return False
        return True

    vid, mid, pts = {}, {}, []

    def v(i, j):
        k = (i, j)
        if k not in vid:
            vid[k] = len(pts)
            pts.append((rn[i], zn[j]))
        return vid[k]

    def m(p, q):
        k = (min(p, q), max(p, q))
        if k not in mid:
            mid[k] = len(pts)
            pts.append(tuple(0.5 * (np.array(pts[p]) + np.array(pts[q]))))
        return mid[k]

    tris, reg = [], []
    for i in range(nr):
        for j in range(nz):
            if not kept(i, j):
                continue
            a_, b_, c_, d_ = v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)
            for t in ((a_, b_, c_), (a_, c_, d_)):
                tris.append([t[0], t[1], t[2], m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0])])
                reg.append(0 if zc[j] > 0 else 1)
    edges, groups = [], []
    tol = 1e-12 * max(geo.R, geo.L)
    for i in range(nr):
        for j in range(nz):
            if not kept(i, j):
                continue
            for (di, dj), (p, q) in (((0, -1), ((i, j), (i + 1, j))), ((1, 0), ((i + 1, j), (i + 1, j + 1))),
                                     ((0, 1), ((i + 1, j + 1), (i, j + 1))), ((-1, 0), ((i, j + 1), (i, j)))):
                if kept(i + di, j + dj):
                    continue
                pa, pb = v(*p), v(*q)
                rm, zm = 0.5 * (pts[pa][0] + pts[pb][0]), 0.5 * (pts[pa][1] + pts[pb][1])
                vertical = di != 0
                if vertical and abs(rm) < tol:
                    g = "axis"
                elif not vertical and abs(zm + geo.coax_len) < tol:
                    g = "port"
                elif zm < 0:
                    g = "coax_wall"
                elif (vertical and abs(rm - geo.a) < tol and zm < geo.h) or \
                        (not vertical and abs(zm - geo.h) < tol and rm < geo.a):
                    g = "probe"
                else:
                    g = "wall"
                edges.append([pa, pb, m(pa, pb)])
                groups.append(g)
    tri = np.array(tris, int)
    P = np.array(pts, float)
    pp = P[tri[:, :3]]
    area = (pp[:, 1, 0] - pp[:, 0, 0]) * (pp[:, 2, 1] - pp[:, 0, 1]) - (pp[:, 2, 0] - pp[:, 0, 0]) * (pp[:, 1, 1] - pp[:, 0, 1])
    neg = area < 0
    tri[neg] = tri[neg][:, [0, 2, 1, 5, 4, 3]]
    meta = {"r_nodes": rn.tolist(), "z_nodes": zn.tolist(), "triangles": len(tri), "nodes": len(P),
            "h_min_m": hmin, "h_max_m": hmax}
    return P, tri, np.array(reg, int), np.array(edges, int), np.array(groups), meta


_GROUP_ATTR = {"axis": 10001, "port": 10002, "coax_wall": 10003, "probe": 10004, "wall": 10005}


def probe_pillbox_design(geo: ProbePillbox, workdir, *, port: str = "open", frequencies=None, modes: int = 0,
                         target_f_Hz: float | None = None, name: str = "probe_pillbox") -> tuple[dict, dict]:
    """Write the mesh, its info file and the design for ``geo`` into ``workdir``.

    ``port``: "open" (lumped port at z = -coax_len, the pickup; open circuit in the eigenproblem) or "short"
    (that boundary is PEC, no pickup: eigenmodes only).  Returns (design dict, mesh meta)."""
    from .fw2d import write_mesh

    if port not in ("open", "short"):
        raise ValueError("port must be 'open' or 'short'")
    if port == "short" and frequencies is not None and len(frequencies):
        raise ValueError("a short-circuited port has no driven sweep")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    P, tri, reg, edges, groups, meta = probe_pillbox_mesh(geo)
    attr = [_GROUP_ATTR[g] for g in groups]
    write_mesh(wd / f"{name}.msh", P, tri, 2 + reg, edges, attr)
    lossy = geo.sigma_S_m is not None
    wall = {"type": "impedance", "material": "wall_metal"} if lossy else {"type": "pec"}
    gl = [{"attr": 10001, "name": "axis", "type": "axis"},
          {"attr": 10002, "name": "port", "type": "port" if port == "open" else "pec"},
          {"attr": 10003, "name": "coax_wall", "type": "pec"},
          {"attr": 10004, "name": "probe", **(wall if geo.probe_lossy else {"type": "pec"})},
          {"attr": 10005, "name": "wall", **wall}]
    used = set(attr)
    gl = [g for g in gl if g["attr"] in used]
    (wd / f"{name}_mesh_info.json").write_text(json.dumps({"ok": True, "boundary_groups": gl}))
    fw = {"boundary": "pec", "rf_only": True,
          "source": {"kind": "uniform", "B_T": [0.0, 0.0, float(geo.B_T)]}, "source_regions": ["cavity"]}
    if frequencies is not None and len(frequencies):
        fw["frequencies"] = [float(f) for f in frequencies]
    if modes:
        fw["modes"] = int(modes)
        fw["families"] = ["TM"]
        if target_f_Hz:
            fw["target_f_Hz"] = float(target_f_Hz)
    d = {"schema": "axionfem.design/1", "name": name, "symmetry": "axisymmetric",
         "regions": [{"name": "cavity", "material": "vacuum", "shape": {"imported": {}}},
                     {"name": "coax", "material": "coax_fill" if geo.coax_eps_r != 1.0 else "vacuum",
                      "shape": {"imported": {}}}],
         "fullwave": fw,
         "mesh": {"order": int(geo.order), "file": str(wd / f"{name}.msh"),
                  "info_file": str(wd / f"{name}_mesh_info.json")}}
    mats = {}
    if lossy:
        mats["wall_metal"] = {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": float(geo.sigma_S_m)}
    if geo.coax_eps_r != 1.0:
        mats["coax_fill"] = {"kind": "linear", "eps_r": float(geo.coax_eps_r), "mu_r": 1.0}
    if mats:
        d["materials"] = mats
    if port == "open":
        d["pickup"] = {"type": "port", "port_boundary": "port", "toward": [1.0, 0.0, 0.0]}
    return d, meta


def solve_probe_pillbox(geo: ProbePillbox, workdir, **kw):
    """Build and solve; returns (Fw2DResults, mesh meta)."""
    import axionfem as af

    d, meta = probe_pillbox_design(geo, workdir, **kw)
    return af.Design(d).solve(workdir=str(workdir), verbose=0), meta


# ------------------------------------------------------------------------------------------------ quadrature
def _tri_rule():
    """Degree-6 symmetric 12-point rule on the reference triangle (Dunavant), barycentric, weights sum 1."""
    a = [(0.873821971016996, 0.063089014491502, 0.063089014491502, 0.050844906370207),
         (0.501426509658179, 0.249286745170910, 0.249286745170910, 0.116786275726379)]
    L, W = [], []
    for l1, l2, l3, w in a:
        for p in {(l1, l2, l3), (l2, l1, l3), (l2, l3, l1)}:
            L.append(p)
            W.append(w)
    for l1, l2, l3, w in [(0.636502499121399, 0.310352451033785, 0.053145049844816, 0.082851075618374)]:
        for p in ((l1, l2, l3), (l1, l3, l2), (l2, l1, l3), (l2, l3, l1), (l3, l1, l2), (l3, l2, l1)):
            L.append(p)
            W.append(w)
    return np.array(L), np.array(W)


def _cell_points(rn, zn, kept):
    """Quadrature points (r, z) and weights (2 pi r dA) over the kept rectangles (two triangles each)."""
    Lb, Wb = _tri_rule()
    R, Z, Wt = [], [], []
    for i in range(len(rn) - 1):
        for j in range(len(zn) - 1):
            if not kept(i, j):
                continue
            c = np.array([[rn[i], zn[j]], [rn[i + 1], zn[j]], [rn[i + 1], zn[j + 1]], [rn[i], zn[j + 1]]])
            A = (rn[i + 1] - rn[i]) * (zn[j + 1] - zn[j]) / 2
            for t in ((0, 1, 2), (0, 2, 3)):
                X = Lb @ c[list(t)]
                R.append(X[:, 0])
                Z.append(X[:, 1])
                Wt.append(Wb * A * 2 * math.pi * X[:, 0])
    return np.concatenate(R), np.concatenate(Z), np.concatenate(Wt)


# ------------------------------------------------------------------------------------------------ (b) expansion
def _modes_TM(res) -> list[dict]:
    ms = res.modes("TM")
    if not ms:
        raise RuntimeError("no TM eigenmodes in the result")
    return ms


def _mode_index_near(res, f_ref: float) -> int:
    fs = [m["f_Hz"] for m in _modes_TM(res)]
    return int(np.argmin(np.abs(np.array(fs) - f_ref)))


def _dominant(x: np.ndarray) -> np.ndarray:
    """The real or imaginary part of a sampled eigenfield, whichever carries it (the other is round-off)."""
    return x.imag if np.abs(x.imag).sum() >= np.abs(x.real).sum() else x.real


def _sample_checked(res, r, z, mode, what):
    s = res.sample(r, z, mode=mode)
    if not np.all(s["valid"]):
        raise RuntimeError(f"{what}: {int((~np.asarray(s['valid'])).sum())} sample points fell outside the mesh")
    return s


def line_resonances(geo: ProbePillbox, f_max: float) -> dict:
    """Resonances of the coax stub below f_max, with the electrical length l_e = coax_len sqrt(eps_r) + h (the
    probe extends the line; an estimate): ``open_open`` n c/(2 l_e) (n >= 1; the open-port problem with a
    capacitive, open-like probe end) and ``open_short`` (2n+1) c/(4 l_e) (relevant to the short-port problem)."""
    le = geo.coax_len * math.sqrt(geo.coax_eps_r) + geo.h
    oo, os_ = [], []
    n = 1
    while n * C0 / (4 * le) <= f_max:
        (os_ if n % 2 else oo).append(n * C0 / (4 * le))
        n += 1
    return {"open_open": oo, "open_short": os_}


def expansion_coupling(geo: ProbePillbox, res_open, res_short, meta: dict, *, Z0: float | None = None,
                       f_ref: float | None = None, n_gauss: int = 6) -> dict:
    """Eigenmode-expansion coupling (b) from the open-port and short-port eigenmode results of the same mesh.

    Returns f_open, f_short, Q0 (perturbative wall Q of the open-port mode), v (port voltage of the real mode
    normalized int eps e^2 dV = 1 over the whole structure, U = 1/2 J), v_TEM (the same from the coax TEM standing
    wave, a check), I_short (port current of the short-port mode), Y (non-resonant port reactance, Z_bg = i Y, from
    the open/short pair) and Y_from_fields = |v / I_short| (Hartung-Haebel), Q_ext, beta, Q_L, f_loaded, the source
    overlap O = int_cavity e . B0 dV, C (form factor, cavity volume V, mode energy over the whole structure) and
    C_cavity_energy (mode energy over the cavity only), coax_energy_fraction, and the normalization check.

    The short-port partner of the open-port mode is the short-port mode whose |Y| (frequency pair) best matches the
    field estimate |v / I_short| of the same short mode (Foster: zeros and poles interlace, so the partner may lie
    on either side); all frequencies are recorded."""
    Z0 = geo.Z_c if Z0 is None else float(Z0)
    f_ref = geo.f_tm010_empty() if f_ref is None else float(f_ref)
    exw = []                                  # expansion warnings
    rn, zn = np.array(meta["r_nodes"]), np.array(meta["z_nodes"])
    rc, zc = 0.5 * (rn[1:] + rn[:-1]), 0.5 * (zn[1:] + zn[:-1])

    def kept(i, j):
        r, z = rc[i], zc[j]
        return not ((r < geo.a and z < geo.h) or (z < 0 and r > geo.b))

    # real mode e: the sampled E of the dual (H real) family is i x real; take the dominant part
    R, Z, W = _cell_points(rn, zn, kept)
    epsw = np.where(Z < 0, geo.coax_eps_r, 1.0) * EPS0
    cav = Z > 0

    def mode_overlap(i):
        E = _sample_checked(res_open, R, Z, ("TM", i), "mode quadrature")["E"]
        use_imag = bool(np.abs(E.imag).sum() >= np.abs(E.real).sum())
        phase = E.imag if use_imag else E.real
        dens = W * epsw * (phase[:, 0] ** 2 + phase[:, 1] ** 2)
        norm = float(np.sum(dens))
        e = phase / math.sqrt(norm)
        O = float(np.sum(W[cav] * e[cav, 1] * geo.B_T))   # B0 = B_T z-hat in the cavity region (source_regions)
        return {"use_imag": use_imag, "norm": norm, "coax_frac": float(np.sum(dens[~cav]) / norm), "O": O}

    # the haloscope mode: among the open-port modes within 30 % of f_ref, the one with the largest |O| (B0 overlap);
    # a line-like mode near a stub resonance has a small overlap and is not picked by frequency alone
    modes_o = _modes_TM(res_open)
    cand = [i for i, m in enumerate(modes_o) if abs(m["f_Hz"] / f_ref - 1) < 0.3] or [_mode_index_near(res_open, f_ref)]
    ovl = {i: mode_overlap(i) for i in cand}
    io = max(cand, key=lambda i: abs(ovl[i]["O"]))
    if io != _mode_index_near(res_open, f_ref):
        exw.append(f"the open-port mode with the largest B0 overlap ({modes_o[io]['f_Hz']:.6g} Hz) is not the one "
                   f"nearest f_ref ({f_ref:.6g} Hz)")
    mo = modes_o[io]
    wo = 2 * math.pi * mo["f_Hz"]
    use_imag, norm, coax_frac, O = (ovl[io][k] for k in ("use_imag", "norm", "coax_frac", "O"))
    V = geo.cavity_volume
    C = O * O * EPS0 / (geo.B_T ** 2 * V)
    # port voltage v = -int_a^b e_r dr just inside the port (composite Gauss on the mesh r-intervals)
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    rr, ww = [], []
    for r0, r1 in zip(rn[:-1], rn[1:]):
        if r0 >= geo.a - 1e-15 and r1 <= geo.b + 1e-15:
            rr.append(0.5 * (r0 + r1) + 0.5 * (r1 - r0) * xg)
            ww.append(0.5 * (r1 - r0) * wg)
    rr, ww = np.concatenate(rr), np.concatenate(ww)
    zp = -geo.coax_len + 1e-9 * geo.L
    Ep = _sample_checked(res_open, rr, np.full_like(rr, zp), ("TM", io), "port line")["E"]
    ep = (Ep.imag if use_imag else Ep.real) / math.sqrt(norm)
    v = float(-np.sum(ww * ep[:, 0]))
    # check: TEM standing wave in the lower half of the coax, r H_phi = A sin(k (z - z_port)); v = 2 pi Z_c A
    k = wo * math.sqrt(geo.coax_eps_r) / C0
    zs = np.linspace(-geo.coax_len, -0.5 * geo.coax_len, 9)[1:]
    rs = np.full_like(zs, math.sqrt(geo.a * geo.b))
    Hs = _sample_checked(res_open, rs, zs, ("TM", io), "coax TEM samples")["H"]
    hphi = _dominant(Hs[:, 2]) / math.sqrt(norm)
    sk = np.sin(k * (zs + geo.coax_len))
    A = float(np.dot(sk, rs * hphi) / np.dot(sk, sk))
    # the dual family's E = curl H / (-i omega eps): E_r = d_z H_phi / (i omega eps) -> real-basis e_r sign:
    v_tem = 2 * math.pi * geo.Z_c * A * (1.0 if use_imag else -1.0)
    # short-port partner: every short mode gives Y from the pair (lemma) and |v / I_s| from its port current
    # (Hartung & Haebel PAC 1993: Q_ext = Q_V + Q_I = w Z0 / v^2 + w / (Z0 I_s^2), i.e. Y = |v / I_s|)
    cands = []
    for i_s, m in enumerate(_modes_TM(res_short)):
        ws = 2 * math.pi * m["f_Hz"]
        if ws == wo:
            continue
        hs_p = _dominant(_sample_checked(res_short, rr, np.full_like(rr, zp), ("TM", i_s), "short port line")["H"][:, 2])
        I_s = float(np.sum(ww * 2 * math.pi * rr * hs_p) / (geo.b - geo.a))
        Ysec = v * v / (2 * (wo - ws))
        Yf = abs(v / I_s) if I_s != 0 else math.inf
        cands.append({"index": i_s, "f_Hz": m["f_Hz"], "w": ws, "I_s": I_s, "Y": Ysec, "Y_fields": Yf,
                      "mismatch": abs(math.log(abs(Ysec) / Yf)) if Yf not in (0, math.inf) else math.inf,
                      "residual": m.get("residual_true")})
    if not cands:
        raise RuntimeError("no short-port mode to pair with the open-port mode")
    best = min(cands, key=lambda c: c["mismatch"])
    ws, I_s = best["w"], best["I_s"]
    Qext_hh = wo * Z0 / (v * v) + ws / (Z0 * I_s * I_s)
    one = one_port_coupling(wo, ws, v, Z0)
    Y = one["Y_ohm"]
    Q0 = mo.get("Q_wall")
    Qext = one["Q_ext"]
    dw = one["d_omega_loaded"]
    method_Y = "open/short frequency pair"
    if abs(ws - wo) / wo < 1e-9:
        # the open/short shift is at the eigensolver's precision (extremely weak coupling): use the field estimate
        exw.append(f"open/short shift {abs(ws - wo) / wo:.2g} unresolved: Y and Q_ext from |v/I_short| "
                   "(Hartung-Haebel) instead of the frequency pair")
        Y = math.copysign(best["Y_fields"], wo - ws)
        Qext = Qext_hh
        dw = -v * v * Y / (2 * (Z0 ** 2 + Y ** 2))
        method_Y = "Hartung-Haebel |v/I_short|"
    if best["mismatch"] > 0.5:
        exw.append(f"no short-port mode pairs consistently with the open-port mode (best |ln(Y/Y_fields)| = "
                   f"{best['mismatch']:.2f}); request more modes or check the geometry")
    # Leontovich walls of a good conductor have X_s = R_s: the complex shift of the open-port resonance is
    # -(1 + i) omega / (2 Q0) (perturbation theory), so the lossy open-port pole sits at omega_o (1 - 1/(2 Q0))
    wshift = -wo / (2 * Q0) if (Q0 and geo.sigma_S_m is not None) else 0.0
    out = {"method": "eigenmode expansion (open/short port eigenmodes)", "Y_method": method_Y, "warnings": exw,
           "f_open_Hz": mo["f_Hz"],
           "f_short_Hz": best["f_Hz"], "residual_open": mo.get("residual_true"), "residual_short": best["residual"],
           "open_modes_Hz": [m["f_Hz"] for m in _modes_TM(res_open)],
           "short_modes_Hz": [m["f_Hz"] for m in _modes_TM(res_short)],
           "short_candidates": [{k: c[k] for k in ("f_Hz", "Y", "Y_fields", "mismatch")} for c in cands],
           "v": v, "v_TEM": v_tem, "I_short": I_s, "Y_ohm": Y, "Y_from_fields_ohm": best["Y_fields"],
           "Q_ext_hartung_haebel": Qext_hh, "Z0_ohm": Z0, "Q_ext": Qext, "f_loaded_Hz": (wo + wshift + dw) / (2 * math.pi),
           "f_pole_Hz": (wo + wshift) / (2 * math.pi), "wall_reactance_shift_Hz": wshift / (2 * math.pi),
           "overlap_O": O, "C": C, "C_cavity_energy": C / (1.0 - coax_frac), "coax_energy_fraction": coax_frac,
           "V_m3": V, "norm_int_eps_e2_raw": norm, "Q0": Q0, "wall_power_W": mo.get("wall_power_W"),
           "line_resonances_Hz": line_resonances(geo, 2 * mo["f_Hz"])}
    if Q0:
        out["beta"] = Q0 / Qext
        out["Q_L"] = 1.0 / (1.0 / Q0 + 1.0 / Qext)
    return out


# ------------------------------------------------------------------------------------------------ (a) driven fit
def fit_port_resonance(f_Hz, Z, Z0: float, *, background: int = 2, max_iter: int = 100) -> dict:
    """Fit Z(w) = sum_{k<=background} c_k x^k + K / (x - s), x = w/w_ref - 1, to a driven port impedance near one
    resonance (linearized least squares, then damped Gauss-Newton on the complex residual).

    Returns the open-port pole w_p = Re w_p - i gamma0 (Q0 = Re w_p / (2 gamma0)), the loaded zero w_L of Z + Z0
    (decay gamma_L, Q_L = Re w_L / (2 gamma_L)), gamma_ext = gamma_L - gamma0, beta = gamma_ext / gamma0 (no
    reference frequency), Q_ext = Re w_p / (2 gamma_ext), the relative rms residual (scaled by max |Z| and by
    Z0, the scale near the loaded zero), ``converged``, ``inside_grid`` (pole and loaded zero within the sampled band) and ``warnings``
    (non-positive decay rates, a second root of Z + Z0 as close to the pole as the chosen one, extrapolation).
    Q_L and Q0/(1 + beta) differ by Re w_L / Re w_p - 1, the relative loaded shift (O(1/Q_ext))."""
    f = np.asarray(f_Hz, float)
    Z = np.asarray(Z, complex)
    if len(f) < background + 4:
        raise ValueError(f"fit_port_resonance: need at least {background + 4} points, got {len(f)}")
    wr = 2 * math.pi * f.mean()
    x = 2 * math.pi * f / wr - 1.0
    nb = background + 1
    # Z (x - s) = poly(x)(x - s) + K  ->  Z x = s Z + q(x), deg q = background + 1 (linear in s and q's coeffs)
    A = np.column_stack([Z] + [x ** k for k in range(nb + 1)])
    sol = np.linalg.lstsq(A, Z * x, rcond=None)[0]
    s = sol[0]
    q = sol[1:]
    # recover poly c (deg nb-1) and K from q(x) = c(x)(x - s) + K by synthetic division
    cdesc, rem = np.polydiv(q[::-1], np.array([1.0, -s]))
    c = cdesc[::-1]
    K = rem[-1] if len(rem) else 0.0
    p = np.concatenate([c, [K, s]])

    def model(p, x):
        c, K, s = p[:nb], p[nb], p[nb + 1]
        return sum(c[k] * x ** k for k in range(nb)) + K / (x - s)

    def jac(p, x):
        c, K, s = p[:nb], p[nb], p[nb + 1]
        return np.column_stack([x ** k for k in range(nb)] + [1 / (x - s), K / (x - s) ** 2])

    def cost(p):
        return float(np.sum(np.abs(Z - model(p, x)) ** 2))

    converged, ls_failed, cst = False, False, cost(p)
    for _ in range(max_iter):
        dp = np.linalg.lstsq(jac(p, x), Z - model(p, x), rcond=None)[0]
        t, accepted = 1.0, False
        while t > 1e-6:                    # backtracking: accept the first step that does not increase the cost
            pn = p + t * dp
            cn = cost(pn)
            if cn <= cst * (1 + 1e-14):
                accepted = True
                break
            t *= 0.5
        if not accepted:                   # no descent step: keep p; converged only if already at round-off
            ls_failed = np.max(np.abs(dp) / (1e-300 + np.abs(p))) > 1e-8
            converged = not ls_failed
            break
        step = np.abs(t * dp) / (1e-300 + np.abs(p))
        p, improve, cst = pn, (cst - cn) / max(cst, 1e-300), cn
        if np.max(step) < 1e-10 or improve < 1e-14:
            converged = True
            break
    c, K, s = p[:nb], p[nb], p[nb + 1]
    res_abs = Z - model(p, x)
    resid = float(np.sqrt(np.mean(np.abs(res_abs) ** 2)) / np.abs(Z).max())
    resid_L = float(np.sqrt(np.mean(np.abs(res_abs) ** 2)) / abs(Z0))   # the scale of Z near the loaded zero
    # loaded zero of Z + Z0: (c(x) + Z0)(x - s) + K = 0, polynomial of degree nb; root nearest s
    cz = c.copy().astype(complex)
    cz[0] += Z0
    poly = np.polymul(cz[::-1], np.array([1.0, -s]))
    poly[-1] += K
    roots = np.roots(poly)
    order = np.argsort(np.abs(roots - s))
    xL = roots[order[0]]
    warnings = []
    if len(roots) > 1 and np.abs(roots[order[1]] - s) < 3 * np.abs(xL - s):
        warnings.append("a second zero of Z + Z0 lies within 3x the distance of the chosen loaded zero")
    wp, wL = wr * (1 + s), wr * (1 + xL)
    g0, gL = -wp.imag, -wL.imag          # decay rates of the open-port pole and of the loaded resonance
    gext = gL - g0
    if not (g0 > 0 and gext > 0):
        warnings.append(f"non-physical decay rates gamma0 = {g0:.4g}, gamma_ext = {gext:.4g} (wrong root or bad fit)")
    # each of the pole and the loaded zero must have a sample within one of its own half-widths (|Im|): the grid
    # is two windows, and a root in the gap between them (or beyond) is extrapolated
    near = lambda r: bool(np.min(np.abs(x - r.real)) <= max(abs(r.imag), 1e-300))  # noqa: E731
    inside = near(s) and near(xL)
    if not inside:
        warnings.append("the fitted pole or loaded zero has no sample within one of its half-widths (extrapolated)")
    tol_fit = max(1e-3, 0.05 * 2 * gL / wL.real)          # max(1e-3, 0.05/Q_L)
    if resid_L > tol_fit:
        warnings.append(f"rms fit misfit {resid_L:.2g} Z0 exceeds max(1e-3, 0.05/Q_L) Z0 = {tol_fit:.2g} Z0 "
                        "(Z0 sets the scale near the loaded zero)")
    if not converged:
        warnings.append("Gauss-Newton did not converge" + (" (line search found no descent step)" if ls_failed else ""))
    Q0 = wp.real / (2 * g0)
    QL = wL.real / (2 * gL)
    Qext = wp.real / (2 * gext)          # same reference frequency as Q0 (Re of the open-port pole)
    return {"method": "driven port impedance, one pole + polynomial background", "f_pole_Hz": wp.real / (2 * math.pi),
            "Q0": Q0, "f_loaded_Hz": wL.real / (2 * math.pi), "Q_L": QL, "Q_ext": Qext, "beta": gext / g0,
            "gamma0_per_s": g0, "gammaL_per_s": gL,
            "Z0_ohm": Z0, "residue_ohm": [complex(K * wr).real, complex(K * wr).imag],
            "background_ohm": [[complex(ck).real, complex(ck).imag] for ck in c], "fit_residual_rel": resid,
            "fit_residual_rel_loaded": resid_L, "converged": converged, "inside_grid": inside,
            "warnings": warnings, "points": len(f)}


def loaded_port_power(f_Hz, Z, Psi, Z0: float) -> np.ndarray:
    """Power [W] into a load Z0 at the port for the source of the sweep (V_oc = -i omega Psi per unit source
    strength): P = |V_oc|^2 Z0 / (2 |Z + Z0|^2)."""
    w = 2 * math.pi * np.asarray(f_Hz, float)
    Voc = -1j * w * np.asarray(Psi)
    return np.abs(Voc) ** 2 * Z0 / (2 * np.abs(np.asarray(Z) + Z0) ** 2)


def sweep_grid(f_loaded: float, Q_L: float, f_pole: float | None = None, Q0: float | None = None, *,
               span: float = 3.0, points: int = 13) -> np.ndarray:
    """Driven-sweep frequencies: ``points`` over +-``span`` loaded half-widths about ``f_loaded`` and, when given,
    ``points`` over +-``span`` unloaded half-widths about the open-port pole ``f_pole`` (union, sorted)."""
    fs = f_loaded * (1 + span * np.linspace(-1, 1, points) / (2 * Q_L))
    if f_pole and Q0:
        fs = np.concatenate([fs, f_pole * (1 + span * np.linspace(-1, 1, points) / (2 * Q0))])
    return np.unique(fs)


def driven_coupling(geo: ProbePillbox, workdir, f_loaded: float, Q_L_guess: float, *, Z0: float | None = None,
                    span: float = 3.0, points: int = 13, f_pole: float | None = None,
                    Q0_guess: float | None = None) -> tuple[dict, dict]:
    """Driven sweep (a) on :func:`sweep_grid` (the loaded window about ``f_loaded`` and the unloaded window about
    the open-port pole ``f_pole``), then :func:`fit_port_resonance`.  Returns (fit, sweep arrays)."""
    Z0 = geo.Z_c if Z0 is None else float(Z0)
    fs = sweep_grid(f_loaded, Q_L_guess, f_pole, Q0_guess, span=span, points=points)
    res, _ = solve_probe_pillbox(geo, workdir, port="open", frequencies=fs)
    sw = res.sweep()
    sw["result"] = res
    fit = fit_port_resonance(sw["f_Hz"], sw["Z"], Z0)
    return fit, sw


# ------------------------------------------------------------------------------------------------ study driver
def coupling_flags(out: dict) -> tuple[list[str], list[str]]:
    """Consistency checks of a :func:`coupling_study` result -> (warnings, notes).

    Warnings (the result should not be used as is): a half-wave (open/open) resonance of the coax stub (electrical
    length coax_len sqrt(eps_r) + h) within 30 % of the mode (the open-port mode then mixes cavity and line); the driven fit's own warnings;
    Q_L of the two routes or the loaded power vs beta/(1+beta) Q_L O^2/(2 w) differing by more than 3/Q_L + 5e-3;
    the two estimates of Y (open/short frequency pair, |v/I_short|) by more than 10/Q_L + 5e-3 (both carry O(1/Q)
    errors; the record shows ~5/Q_L); engine warnings (except sweep residuals < 1e-5 near the pole, a note); the
    expansion's own warnings (unresolved open/short shift, inconsistent pairing).  Notes: coax energy fraction of the open-port mode above 1 % (C is then the
    whole-structure value; C_cavity_energy is the cavity-only one)."""
    import re

    ex = out["expansion"]
    W, N = [], []
    # engine warnings: a sweep point next to the open-port pole has a nearly singular operator and the direct
    # solver reports relative residuals of 1e-8..1e-6 there ("ill-conditioned"); the fit-quality gate covers their
    # effect, so residuals below 1e-5 are notes (summarized), everything else is a warning
    ill = []
    fit0 = out.get("driven") or {}
    for m in out.get("engine_warnings", []):
        r = re.search(r"^driven sweep solve: .* at f = ([0-9.eE+-]+) Hz: relative residual ([0-9.eE+-]+) \(ill-conditioned\)", m)
        near_pole = False
        if r and fit0.get("Q0"):
            # within 4 unloaded half-widths of the fitted open-port pole
            near_pole = abs(float(r.group(1)) / fit0["f_pole_Hz"] - 1) <= 4.0 / (2 * fit0["Q0"])
        if r and near_pole and float(r.group(2)) < 1e-5:
            ill.append(float(r.group(2)))
        else:
            W.append(m)
    if ill:
        N.append(f"{len(ill)} driven-sweep points near the open-port pole with solver relative residual up to "
                 f"{max(ill):.2g} (ill-conditioned near resonance; covered by the fit-quality gate)")
    W += [f"expansion: {m}" for m in ex.get("warnings", [])]
    near = [f for f in ex["line_resonances_Hz"]["open_open"] if abs(f / ex["f_open_Hz"] - 1) < 0.30]
    if near:
        W.append(f"coax stub half-wave resonance(s) {[round(f) for f in near]} Hz within 30 % of the mode "
                 f"({ex['f_open_Hz']:.6g} Hz): the open-port mode mixes with the line; Q0, beta and C refer to that "
                 "mixed mode (shorten the line)")
    if ex["coax_energy_fraction"] > 1e-2:
        N.append(f"{ex['coax_energy_fraction']:.3g} of the open-port mode energy is in the coax (C uses the whole-"
                 "structure normalization, consistent with the Q's; C_cavity_energy is the cavity-only value)")
    fit = out.get("driven")
    if fit:
        W += [f"driven fit: {m}" for m in fit["warnings"]]
        QL = fit["Q_L"]
        checks = (("Q_L driven vs expansion", out["comparison"]["Q_L"], 3.0),
                  ("loaded power vs beta/(1+beta) Q_L O^2/(2 w)", out["loaded_power"]["ratio"] - 1.0, 3.0),
                  ("Y (open/short pair) vs |v/I_short|", abs(ex["Y_ohm"]) / ex["Y_from_fields_ohm"] - 1.0, 10.0))
        for name, val, c in checks:
            tol = c / QL + 5e-3
            if abs(val) > tol:
                W.append(f"{name}: relative difference {val:+.3g} exceeds {c:g}/Q_L + 5e-3 = {tol:.3g}")
    return W, N



def coupling_study(geo: ProbePillbox, workdir, *, Z0: float | None = None, modes: int = 3, points: int = 13,
                   span: float = 3.0, warn: bool = True) -> dict:
    """(b) expansion from the open/short eigenmodes, then (a) the driven fit (the expansion only places the grid:
    loaded window about its loaded resonance, unloaded window about its open-port pole; the fit itself is
    independent of it), the driven-vs-expansion comparison of Psi(omega) near the resonance, and the loaded port
    power at the loaded resonance vs the single-mode formula P = beta/(1+beta) Q_L O^2 / (2 omega) (unit source).

    ``warnings`` lists failed consistency checks and ``notes`` informative ones (:func:`coupling_flags`); warnings
    are also issued as RuntimeWarning when ``warn``."""
    import warnings as _w

    wd = Path(workdir)
    f_ref = geo.f_tm010_empty()
    ro, meta = solve_probe_pillbox(geo, wd / "open", port="open", modes=modes, target_f_Hz=f_ref)
    rs, _ = solve_probe_pillbox(geo, wd / "short", port="short", modes=modes, target_f_Hz=f_ref)
    ex = expansion_coupling(geo, ro, rs, meta, Z0=Z0)
    engine = [f"{name} solve: {w}" for name, r in (("open-port eigen", ro), ("short-port eigen", rs))
              for w in r.data.get("warnings", [])]
    out = {"geometry": asdict(geo), "Z_c_ohm": geo.Z_c, "mesh": {k: meta[k] for k in ("triangles", "nodes", "h_min_m", "h_max_m")},
           "expansion": ex}
    if ex.get("Q0"):
        fit, sw = driven_coupling(geo, wd / "driven", ex["f_loaded_Hz"], ex["Q_L"], Z0=ex["Z0_ohm"], span=span,
                                  points=points, f_pole=ex["f_pole_Hz"], Q0_guess=ex["Q0"])
        if not fit["inside_grid"] and fit["Q0"] > 0 and fit["Q_L"] > 0:
            # the windows placed from the expansion missed a root (coarse mesh, strong coupling): sweep once more
            # with both windows centred on the fitted pole and loaded zero
            fit, sw = driven_coupling(geo, wd / "driven2", fit["f_loaded_Hz"], fit["Q_L"], Z0=ex["Z0_ohm"],
                                      span=span, points=points, f_pole=fit["f_pole_Hz"], Q0_guess=fit["Q0"])
            fit["resweeps"] = 1
        out["driven"] = fit
        # Psi: driven vs single-mode expansion Psi = -v O / (w_c^2 - w^2), w_c = w_o (1 - (1 + i)/(2 Q0))
        w = 2 * math.pi * sw["f_Hz"]
        wo = 2 * math.pi * ex["f_open_Hz"]
        wc = wo * (1 - (1 + 1j) / (2 * ex["Q0"]))
        Psi_exp = -ex["v"] * ex["overlap_O"] / (wc ** 2 - w ** 2)
        rel = np.abs(sw["Psi"] - Psi_exp) / np.abs(sw["Psi"]).max()
        # near the loaded resonance (what the receiver sees) vs near the open-port pole (where Psi of a current
        # drive peaks and a frequency error of a fraction of f/Q0 dominates)
        # loaded window, without the points of the unloaded window about the open-port pole
        near_L = np.abs(sw["f_Hz"] - ex["f_loaded_Hz"]) <= span * ex["f_loaded_Hz"] / (2 * ex["Q_L"]) * (1 + 1e-9)
        near_p = np.abs(sw["f_Hz"] - ex["f_pole_Hz"]) <= span * ex["f_pole_Hz"] / (2 * ex["Q0"]) * (1 + 1e-9)
        relL = np.abs(sw["Psi"] / Psi_exp - 1.0)[near_L & ~near_p]
        out["Psi_check"] = {"max_rel_diff_vs_peak": float(rel.max()),
                            "max_rel_diff_loaded_window": float(relL.max()) if relL.size else None,
                            "f_Hz": sw["f_Hz"].tolist(),
                            "Psi_driven": [[z.real, z.imag] for z in sw["Psi"]],
                            "Psi_expansion": [[z.real, z.imag] for z in Psi_exp]}
        # loaded power at the loaded resonance (one extra point) vs the single-mode haloscope form
        rL, _ = solve_probe_pillbox(geo, wd / "loaded_peak", port="open", frequencies=[fit["f_loaded_Hz"]])
        engine += [f"{name} solve: {w}" for name, r in (("driven sweep", sw["result"]), ("loaded-peak", rL))
                   for w in r.data.get("warnings", [])]
        s1 = rL.sweep()
        P_fe = float(loaded_port_power(s1["f_Hz"], s1["Z"], s1["Psi"], ex["Z0_ohm"])[0])
        wL = 2 * math.pi * fit["f_loaded_Hz"]
        P_tot = fit["Q_L"] * ex["overlap_O"] ** 2 / (2 * wL)
        eta = fit["beta"] / (1 + fit["beta"])
        out["loaded_power"] = {"f_Hz": fit["f_loaded_Hz"], "P_fe_W": P_fe, "P_single_mode_W": eta * P_tot,
                               "ratio": P_fe / (eta * P_tot), "eta_c_fit": eta, "eta_c_from_power": P_fe / P_tot,
                               "note": "unit source J = B0 (A/m^2 per T); P_single_mode = beta/(1+beta) Q_L O^2/(2 omega); "
                                       "eta_c_from_power = P_fe / (Q_L O^2/(2 omega))"}
        out["comparison"] = {k: (fit[k] / ex[k] - 1.0) for k in ("Q0", "Q_L", "Q_ext", "beta")}
        out["comparison"]["f_loaded"] = fit["f_loaded_Hz"] / ex["f_loaded_Hz"] - 1.0
        out["comparison"]["f_pole"] = fit["f_pole_Hz"] / ex["f_pole_Hz"] - 1.0
        out["comparison"]["one_over_Q_L"] = 1.0 / ex["Q_L"]
        out["comparison"]["residue_over_iv2_2"] = (complex(*fit["residue_ohm"]) / (0.5j * ex["v"] ** 2) - 1.0).real
    out["engine_warnings"] = engine
    W, N = coupling_flags(out)
    out["warnings"], out["notes"] = W, N
    if warn:
        for m in W:
            _w.warn(f"cavity_coupling: {m}", RuntimeWarning, stacklevel=2)
    return out


# ------------------------------------------------------------------------------------------------ references
def one_port_coupling(w_open: float, w_short: float, v: float, Z0: float) -> dict:
    """Single-mode one-port lemma (docs/cavity/coupling.md, "Eigenmode expansion").

    Lossless port impedance near the open-port resonance w_open: Z = -i v^2 / (2 (w_open - w)) + i Y, with the
    background Y from the short-circuit resonance (Z(w_short) = 0): Y = v^2 / (2 (w_open - w_short)) (secant: the
    background's variation between w_short and w_open is neglected, a relative O(1/Q_ext) error).  A real load Z0
    gives the loaded complex frequency w_open - v^2 (Y + i Z0) / (2 (Z0^2 + Y^2)) (plus the intrinsic loss), so
        Q_ext = w_open (Z0^2 + Y^2) / (v^2 Z0),   d omega_loaded = -v^2 Y / (2 (Z0^2 + Y^2))."""
    if w_open == w_short:
        raise ValueError("open- and short-port resonances coincide: no coupling resolved (refine the mesh)")
    Y = v * v / (2 * (w_open - w_short))
    return {"Y_ohm": Y, "Q_ext": w_open * (Z0 ** 2 + Y ** 2) / (v * v * Z0),
            "d_omega_loaded": -v * v * Y / (2 * (Z0 ** 2 + Y ** 2))}


def probe_coupling_estimates(geo: ProbePillbox, Q0: float, *, Z0: float | None = None) -> dict:
    """Closed-form beta of the on-axis TM010 probe (an independent analytic estimate, docs/cavity/coupling.md,
    "Small-probe law and flush probe"), trend references only; the suffixes of the output keys name the formulas:

    * h > 0, slender rod (a, b << h << R, L; leading-log, relative error O(1/ln(h/a))): "2d"
      beta = Q0 w Z0 C_p^2 l_eff^2 / (eps0 V_m (1 + (w C_p Z0)^2)), C_p = 2 pi eps0 h / ln(h/a), l_eff = h/2,
      V_m = pi R^2 L J1(p01)^2; and its high-reactance limit "2e" beta ~ h^4 / ln^2(h/a).
    * h = 0, flush coax (TEM aperture potential, approximate): "3c"
      beta = Q0 Z0 w eps0 pi (b^2 - a^2)^2 / (4 R^2 L J1(p01)^2 ln^2(b/a)).
    The eps_r of the coax fill does not enter except through Z0 (the line impedance).  w = the empty TM010."""
    from scipy.special import j1

    Z0 = geo.Z_c if Z0 is None else float(Z0)
    w = 2 * math.pi * geo.f_tm010_empty()
    Vm = math.pi * geo.R ** 2 * geo.L * j1(P01) ** 2
    out = {"Z0_ohm": Z0, "Q0": Q0}
    if geo.h > 0:
        Lam = math.log(geo.h / geo.a)
        Cp = 2 * math.pi * EPS0 * geo.h / Lam
        leff = geo.h / 2
        b2d = Q0 * w * Z0 * Cp ** 2 * leff ** 2 / (EPS0 * Vm * (1 + (w * Cp * Z0) ** 2))
        b2e = Q0 * math.pi * EPS0 * w * Z0 * geo.h ** 4 / (geo.R ** 2 * geo.L * j1(P01) ** 2 * Lam ** 2)
        out.update(beta_slender_2d=b2d, Q_ext_slender_2d=Q0 / b2d, beta_slender_2e=b2e, Q_ext_slender_2e=Q0 / b2e,
                   C_probe_F=Cp, ln_h_over_a=Lam)
    else:
        b3 = Q0 * Z0 * w * EPS0 * math.pi * (geo.b ** 2 - geo.a ** 2) ** 2 / (
            4 * geo.R ** 2 * geo.L * j1(P01) ** 2 * math.log(geo.b / geo.a) ** 2)
        out.update(beta_flush_TEM_3c=b3, Q_ext_flush_TEM_3c=Q0 / b3)
    return out


def to_haloscope_cavity(coupling: dict, B_T: float, *, source: str = "driven", V_m3: float | None = None,
                        C: float | None = None, **kw):
    """A :class:`axionfem.haloscope.Cavity` from a :func:`coupling_study` result: Q0 and beta from ``source``
    ("driven" or "expansion"), C and V from the expansion (the eigenmode) unless given.  Use it with
    :func:`axionfem.haloscope.signal_power` / :func:`axionfem.haloscope.scan_rate`.

    C is normalized with the open-port mode energy over the whole structure, the energy Q0, Q_ext and Q_L refer to,
    so beta/(1+beta) Q_L C B^2 V reproduces the full-wave port power (``loaded_power.ratio``); with a short line
    (coax_energy_fraction << 1) this is the usual cavity C.  Results with ``warnings`` are refused unless
    ``allow_warnings=True``."""
    from .haloscope import Cavity

    if source not in ("driven", "expansion"):
        raise ValueError(f"source must be 'driven' or 'expansion', got {source!r}")
    allow = kw.pop("allow_warnings", False)
    if coupling.get("warnings") and not allow:
        raise ValueError("coupling result has warnings (pass allow_warnings=True to use it anyway): "
                         + "; ".join(coupling["warnings"]))
    ex = coupling["expansion"]
    src = coupling.get(source)
    if src is None:
        raise KeyError(f"coupling result has no '{source}' block")
    return Cavity(B_T=B_T, V_m3=V_m3 if V_m3 is not None else ex["V_m3"], C=C if C is not None else ex["C"],
                  Q0=src["Q0"], beta=src["beta"], **kw)
