"""Figures of the SRF heterodyne cell (written to runs/srf_figures/*.pdf, *.png; not committed). Beta.

Stages (data first, then plots; the data files go to runs/srf_figures/data/):
  python benchmarks/srf/make_srf_figures.py data      # 2D solves: cell + cylinder fields, tuning sweep (minutes)
  python benchmarks/srf/make_srf_figures.py data3d --ranks 16   # 3D cell with field probes on z = 0 (MPI)
  python benchmarks/srf/make_srf_figures.py plot [--style STYLE.py]   # optional matplotlib style module

Figures (inferno for magnitudes, RdBu_r for signed fields):
  srf_xsec           transverse (x, y) section at the equator plane z = 0: (a) pump TM020 B_phi (signed, circulation
                     arrows, sign-reversal radius), (b) signal TE011 E_phi, (c) overlap density E1.B0, (d) radial
                     profiles: 2D axisymmetric (lines) vs the 3D solution sliced at z = 0 (markers), max difference;
  srf_overlap_meridional  signed overlap density E1.B0 on the meridional (z, r) plane, cell vs the TM020/TE011
                     closed cylinder (eta = 0.437), with the pump's sign-change line;
  srf_tuning         TM020 / TE011 frequencies vs the axial shortening factor (unscaled cell) and the splitting
                     (f_TE011 - f_TM020)/f vs the factor, with a 2 MHz splitting at 1.4 GHz for comparison;
  srf_wall_profile   pump (and signal) |B_t| along the wall contour vs arc length, B_pk, the pump's field node on
                     the wall (pickup location) and the loss density 2 pi r |B_t|^2 (geometry factor G) with the
                     arc that carries 80 % of the pump loss.
Overlap density: E1.B0 / (E1_rms B0_rms), whose volume average is +-eta_10 (rms over the cavity volume).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from axionfem import srf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "srf_figures"
DATA = OUT / "data"
REQ = srf.PROJECTX_650_B09["equator_radius"]
MU0, C = srf.MU0, srf.C_LIGHT
NX = 81                      # (x, y) grid of the transverse section
W1, W2 = 3.4, 7.0            # single- and double-column widths [in]
FS = 8.0
NODE_COLOR = "#1b7837"       # the pump's sign-change (node) line


# ------------------------------------------------------------------------------------------------- data (2D)
def _solve_cell(work, n=40, order=3, s=0.86):
    # the committed figures were made with the faceted wall (angle_tol 0.01); the wall profiles sample that polyline
    # (1e-6 R_eq inside); the quoted peaks are the solver's wall Gauss points
    d = srf.elliptical_cell_design(axial_factor=s, h=REQ / n, order=order, modes=6, wall="polygon")
    r = d.solve(f"{work}/cell_{s}_{n}_{order}", verbose=0)
    return d, r


def _rms(F, keys, V):
    from axionfem.srf import _tri_product_integral
    return math.sqrt(sum(_tri_product_integral(F["r"], F["z"], F["tris"], F[k], F[k]) for k in keys) / V)


def _wall_profile(res, row, wall, fam, n_per_seg=6):
    """Fields just inside the wall polyline: arc length s, (r, z), tangential |B_t| (vacuum: mu0 |H_t|) and the
    signed B_phi (TM) at points spread along each segment."""
    W = np.asarray(wall, float)
    pts, ss, tang = [], [], []
    s0 = 0.0
    for a, b in zip(W[:-1], W[1:]):
        L = float(np.hypot(*(b - a)))
        if L <= 0:
            continue
        t = (b - a) / L
        for u in (np.arange(n_per_seg) + 0.5) / n_per_seg:
            pts.append(a + u * (b - a))
            ss.append(s0 + u * L)
            tang.append(t)
        s0 += L
    P, T = np.array(pts), np.array(tang)
    Nrm = np.column_stack([-T[:, 1], T[:, 0]])
    eps = 1e-6 * REQ
    best = None
    for sg in (1, -1):
        Q = P + sg * eps * Nrm
        smp = res.cavity_mode_sample(row, Q[:, 0], Q[:, 1])
        ok = np.isfinite(smp["H_phi"]).mean()
        if best is None or ok > best[0]:
            best = (ok, Q, smp)
    _, Q, smp = best
    if fam == "TM":
        Bt = MU0 * np.abs(smp["H_phi"])
        signed = MU0 * smp["H_phi"]
    else:
        Bt = MU0 * np.abs(smp["H_r"] * T[:, 0] + smp["H_z"] * T[:, 1])
        signed = MU0 * (smp["H_r"] * T[:, 0] + smp["H_z"] * T[:, 1])
    E = np.sqrt(smp["E_r"] ** 2 + smp["E_phi"] ** 2 + smp["E_z"] ** 2)
    return {"s": np.array(ss), "r": Q[:, 0], "z": Q[:, 1], "Bt": Bt, "B_signed": signed, "E": E}


def stage_data(args):
    work = tempfile.mkdtemp(prefix="srf_fig_")
    DATA.mkdir(parents=True, exist_ok=True)
    d, r = _solve_cell(work)
    con = d.srf_contour
    o = srf.figures_of_merit(r, with_fields=True)
    lab = srf.classify_modes(r, con)
    pr, sr = srf.find_mode(r, "TM020", labels=lab), srf.find_mode(r, "TE011", labels=lab)
    V = o["V_m3"]
    P, S = o["fields"]["pump"], o["fields"]["signal"]
    B0 = _rms(P, ("H_r", "H_phi", "H_z"), V) * MU0
    E1 = _rms(S, ("E_r", "E_phi", "E_z"), V)
    # meridional lattice (signed overlap density)
    mer = {"r": P["r"], "z": P["z"], "tris": P["tris"], "Bphi": MU0 * P["H_phi"] / B0, "Ephi": S["E_phi"] / E1}
    # transverse section z = 0
    x = np.linspace(-REQ, REQ, NX)
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    sp = r.cavity_mode_sample(pr, R.ravel(), np.zeros(R.size))
    ss = r.cavity_mode_sample(sr, R.ravel(), np.zeros(R.size))
    rr = np.linspace(1e-6 * REQ, REQ * (1 - 1e-6), 801)
    lp = r.cavity_mode_sample(pr, rr, np.zeros_like(rr))
    ls = r.cavity_mode_sample(sr, rr, np.zeros_like(rr))
    Bl, El = MU0 * lp["H_phi"] / B0, ls["E_phi"] / E1
    i0 = np.nonzero(np.diff(np.sign(Bl[np.isfinite(Bl)])))[0]
    r_node = float(np.interp(0, [Bl[i0[0] + 1], Bl[i0[0]]], [rr[i0[0] + 1], rr[i0[0]]])) if len(i0) else None
    # wall profiles (pump TM, signal TE)
    wp = _wall_profile(r, pr, con["wall"], "TM")
    ws = _wall_profile(r, sr, con["wall"], "TE")
    # cylinder pair (TM020/TE011)
    dc = srf.cylinder_pair_design("TM020/TE011", 0.1, h=0.0025, order=3, modes=6)
    rc = dc.solve(f"{work}/cyl", verbose=0)
    oc = srf.figures_of_merit(rc, "TM020", "TE011", with_fields=True)
    Pc, Sc = oc["fields"]["pump"], oc["fields"]["signal"]
    Bc, Ec = _rms(Pc, ("H_r", "H_phi", "H_z"), oc["V_m3"]) * MU0, _rms(Sc, ("E_r", "E_phi", "E_z"), oc["V_m3"])
    np.savez_compressed(DATA / "cell2d.npz", mer_r=mer["r"], mer_z=mer["z"], mer_tris=mer["tris"],
                        mer_Bphi=mer["Bphi"], mer_Ephi=mer["Ephi"], X=X, Y=Y,
                        Bphi_xy=(MU0 * sp["H_phi"] / B0).reshape(X.shape), Ephi_xy=(ss["E_phi"] / E1).reshape(X.shape),
                        line_r=rr, line_Bphi=Bl, line_Ephi=El, wall=np.array(con["wall"]),
                        wp_s=wp["s"], wp_r=wp["r"], wp_z=wp["z"], wp_Bt=wp["Bt"] / B0, wp_Bsigned=wp["B_signed"] / B0,
                        wp_E=wp["E"] / (C * B0), ws_Bt=ws["Bt"] / (_rms(S, ("H_r", "H_phi", "H_z"), V) * MU0),
                        cyl_r=Pc["r"], cyl_z=Pc["z"], cyl_tris=Pc["tris"], cyl_Bphi=MU0 * Pc["H_phi"] / Bc,
                        cyl_Ephi=Sc["E_phi"] / Ec, cyl_wall=np.array(dc.srf_contour["wall"]))
    meta = {k: o[k] for k in ("f_pump_Hz", "f_signal_Hz", "eta_10", "xi_pump", "Bpk_over_B0", "Epk_over_cB0",
                              "B_equator_over_B0", "G_pump_ohm", "G_signal_ohm", "A_over_V23", "V_m3")}
    meta.update(r_node_z0_m=r_node, pump_peaks=o["pump_peaks"], cyl_eta=oc["eta_10"], cyl_L=dc.srf_contour["length"],
                B0_rms=B0, E1_rms=E1, mesh="h = Req/40, order 3, angle_tol 0.01",
                eta_sign=float(np.sign(np.sum(mer["Bphi"] * mer["Ephi"] * mer["r"]))))
    # tuning sweep (unscaled cell; h = Req/30, order 3)
    sweep = []
    for s in np.round(np.arange(0.78, 0.941, 0.01), 3):
        dd = srf.elliptical_cell_design(axial_factor=float(s), h=REQ / 30, order=3, modes=6, wall="polygon")
        rs = dd.solve(f"{work}/tune_{s}", verbose=0)
        lb = srf.classify_modes(rs, dd.srf_contour)
        row = {"s": float(s)}
        for nm in ("TM020", "TE011", "TM011", "TM010", "TM021", "TE021"):
            hit = [m for m in lb if m["name"] == nm]
            row[nm] = hit[0]["f_Hz"] if len(hit) == 1 else None
        row["V_m3"] = dd.srf_contour["V_m3"]
        sweep.append(row)
        print("tune", row, flush=True)
    tu = srf.tune_degeneracy(design_kw={"h": REQ / 40, "order": 3, "modes": 6}, bracket=(0.84, 0.88), tol=1e-7,
                             workdir=f"{work}/tuner")
    meta["tuning"] = {"sweep": sweep, "crossing": {"axial_factor": tu["axial_factor"], "f_Hz": tu["f_Hz"]}}
    (DATA / "cell2d.json").write_text(json.dumps(meta, indent=1, default=float))
    print("wrote", DATA / "cell2d.npz", DATA / "cell2d.json")


# ------------------------------------------------------------------------------------------------- data (3D)
def stage_data3d(args):
    """3D cell (h = Req/n3, order o3) with field probes on the z = 0 plane: the (x, y) grid of the figure and two
    radial lines (along x and along the diagonal)."""
    DATA.mkdir(parents=True, exist_ok=True)
    meta2 = json.loads((DATA / "cell2d.json").read_text())
    x = np.linspace(-REQ, REQ, NX)
    X, Y = np.meshgrid(x, x)
    m = np.hypot(X, Y) < REQ * 0.999
    rr = np.linspace(0.005, 0.995, 100) * REQ
    pts = [[float(a), float(b), 0.0] for a, b in zip(X[m], Y[m])]
    pts += [[float(t), 0.0, 0.0] for t in rr] + [[float(t / math.sqrt(2)), float(t / math.sqrt(2)), 0.0] for t in rr]
    d = srf.elliptical_cell_design_3d(axial_factor=0.86, h=REQ / args.n3, order=args.o3, modes=16,
                                      target_f_Hz=meta2["f_pump_Hz"], wall="polygon")
    d.doc["fullwave"]["field_probes"] = pts
    r = d.solve(tempfile.mkdtemp(prefix="srf_fig3d_"), verbose=0, mpi_ranks=args.ranks)
    o = srf.figures_of_merit_3d(r, meta2["f_pump_Hz"], meta2["f_signal_Hz"], contour=d.srf_contour)
    modes = r.fullwave["eigenmodes"]["modes"]
    valid = np.array(r.fullwave["eigenmodes"]["field_probes"]["valid"], bool)
    Bp = np.array(modes[o["i_pump"]]["probes"]["B"])
    Es = np.array(modes[o["i_signal"]]["probes"]["E"])
    Pq = np.array(pts)
    phi = np.column_stack([-Pq[:, 1], Pq[:, 0]]) / np.maximum(np.hypot(Pq[:, 0], Pq[:, 1]), 1e-300)[:, None]
    V = r.fullwave["cavity"]["volume_m3"]
    # normalization as in 2D: int |E|^2 = 1 over the cavity -> E_rms = V^-1/2; B rms of the pump from the solver
    B0 = modes[o["i_pump"]]["peak_wall_fields"]["B_rms"]
    E1 = 1 / math.sqrt(V)
    np.savez_compressed(DATA / "cell3d_z0.npz", pts=Pq, valid=valid, Bphi=(Bp[:, :2] * phi).sum(1) / B0,
                        Ephi=(Es[:, :2] * phi).sum(1) / E1, B_all=Bp / B0, E_all=Es / E1, n_grid=int(m.sum()),
                        n_line=len(rr))
    (DATA / "cell3d.json").write_text(json.dumps({**{k: v for k, v in o.items()}, "h_over_Req": 1 / args.n3,
                                                  "order": args.o3, "V_m3": V}, indent=1, default=float))
    print("wrote", DATA / "cell3d_z0.npz")


# ------------------------------------------------------------------------------------------------------ plots
def _style(path):
    if path:
        spec = importlib.util.spec_from_file_location("plotting_defaults", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS - 1,
                         "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
                         "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.labelpad": 2.0,
                         "savefig.facecolor": "white", "lines.linewidth": 1.0})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / f"{name}.pdf")


def _polar(D, key_line, rmax=REQ, nr=200, nt=256):
    """The z = 0 section of an m = 0 field on a polar grid (exact: the field depends on r only), filling the disk
    r < rmax without the ragged edge of a Cartesian grid."""
    r = np.linspace(0, rmax, nr)
    t = np.linspace(0, 2 * np.pi, nt)
    R, T = np.meshgrid(r, t)
    F = np.interp(R, D["line_r"], np.nan_to_num(D[key_line]))
    return R * np.cos(T), R * np.sin(T), F


def _sym(v):
    m = np.nanmax(np.abs(v))
    return -m, m


def plot_xsec(D, meta, D3, meta3):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    X, Y, Bp = _polar(D, "line_Bphi")
    Es = _polar(D, "line_Ephi")[2]
    ov = Bp * Es * meta["eta_sign"]
    rn = meta["r_node_z0_m"]
    cm = 100.0  # m -> cm
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.42)
    th = np.linspace(0, 2 * np.pi, 400)
    titles = [r"(a) pump TM$_{020}$" "\n" r"$\tilde B_{0\phi}/B_0$", r"(b) signal TE$_{011}$" "\n" r"$\tilde E_{1\phi}/E_{1,\rm rms}$",
              r"(c) overlap density" "\n" r"$\tilde{\mathbf E}_1\!\cdot\!\tilde{\mathbf B}_0/(E_{1,\rm rms}B_0)$"]
    for k, (F, t) in enumerate(zip((Bp, Es, ov), titles)):
        ax = fig.add_subplot(gs[0, k])
        lo, hi = _sym(F)
        im = ax.pcolormesh(X * cm, Y * cm, F, cmap="RdBu_r", norm=TwoSlopeNorm(0, lo, hi), shading="gouraud",
                           rasterized=True)
        ax.plot(REQ * cm * np.cos(th), REQ * cm * np.sin(th), color="0.15", lw=0.9)
        if rn and k in (0, 2):
            ax.plot(rn * cm * np.cos(th), rn * cm * np.sin(th), color=NODE_COLOR, lw=0.9)
        if k < 2:  # circulation arrows: phi-hat times the sign (and size) of the field
            for rad, na in ((0.5 * (rn or REQ / 2), 6), (0.5 * (rn + REQ) if rn else 0.75 * REQ, 10)):
                ang = np.linspace(0, 2 * np.pi, na + 1)[:-1] + np.pi / na
                v = np.interp(rad, D["line_r"], D["line_Bphi"] if k == 0 else D["line_Ephi"])
                ux, uy = -np.sin(ang) * np.sign(v), np.cos(ang) * np.sign(v)
                ax.quiver(rad * cm * np.cos(ang), rad * cm * np.sin(ang), ux, uy, color="k", scale=18, width=0.012,
                          headwidth=3.5, headlength=3.5, headaxislength=3, pivot="mid")
        ax.set_aspect("equal")
        ax.set_xlim(-REQ * cm * 1.05, REQ * cm * 1.05)
        ax.set_ylim(-REQ * cm * 1.05, REQ * cm * 1.05)
        ax.set_xlabel(r"$x$ [cm]")
        if k == 0:
            ax.set_ylabel(r"$y$ [cm]")
        ax.set_title(t, fontsize=FS - 0.5, pad=3)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2, aspect=16)
        cb.ax.tick_params(labelsize=FS - 2, length=1.5)
        cb.outline.set_linewidth(0.4)
    ax = fig.add_subplot(gs[0, 3])
    ax.set_box_aspect(1.0)
    rr = D["line_r"] * cm
    ax.plot(rr, D["line_Bphi"], color="#2166ac", label=r"$\tilde B_{0\phi}/B_0$ (2D)")
    ax.plot(rr, D["line_Ephi"], color="#b2182b", label=r"$\tilde E_{1\phi}/E_{1,\rm rms}$ (2D)")
    txt = ""
    if D3 is not None:
        ng, nl = int(D3["n_grid"]), int(D3["n_line"])
        P = D3["pts"]
        for sl, mk in ((slice(ng, ng + nl), "o"), (slice(ng + nl, ng + 2 * nl), "s")):
            rq = np.hypot(P[sl, 0], P[sl, 1]) * cm
            ok = D3["valid"][sl]
            sb = np.sign(np.nansum(D3["Bphi"][sl][ok] * np.interp(rq[ok], rr, D["line_Bphi"])))
            se = np.sign(np.nansum(D3["Ephi"][sl][ok] * np.interp(rq[ok], rr, D["line_Ephi"])))
            ax.plot(rq[ok][::5], sb * D3["Bphi"][sl][ok][::5], mk, ms=2.2, mfc="none", mec="#2166ac", mew=0.5)
            ax.plot(rq[ok][::5], se * D3["Ephi"][sl][ok][::5], mk, ms=2.2, mfc="none", mec="#b2182b", mew=0.5)
        ax.plot([], [], "o", ms=2.2, mfc="none", mec="0.3", mew=0.5, label=r"3D, along $x$")
        ax.plot([], [], "s", ms=2.2, mfc="none", mec="0.3", mew=0.5, label=r"3D, along $x=y$")
        txt = ("\n" rf"3D (h = R$_{{\rm eq}}$/{1 / meta3['h_over_Req']:.0f}, order {meta3['order']}) vs 2D on the plane: max "
               rf"$|\Delta|$ = {100 * meta3['xsec_maxdiff_B']:.2f}\% ($\tilde B_{{0\phi}}$), "
               rf"{100 * meta3['xsec_maxdiff_E']:.2f}\% ($\tilde E_{{1\phi}}$) of the peak")
    ax.axhline(0, color="0.6", lw=0.5)
    if rn:
        ax.axvline(rn * cm, color=NODE_COLOR, lw=0.8)
    ax.set_xlim(0, REQ * cm)
    ax.set_xlabel(r"$r$ [cm] ($z=0$)")
    ax.set_title(r"(d) radial profiles" "\n" r"2D (lines) vs 3D (markers)", fontsize=FS - 0.5, pad=3)
    ax.legend(loc="lower left", fontsize=FS - 2.5, frameon=False, handlelength=1.4)
    fig.text(0.5, -0.01, rf"$f_0 = {meta['f_pump_Hz'] / 1e9:.4f}$ GHz (unscaled cell), $\eta_{{10}} = "
             rf"{meta['eta_10']:.3f}$; green: $\tilde B_{{0\phi}} = 0$ at $r = {rn * cm:.1f}$ cm" + txt, ha="center",
             va="top", fontsize=FS - 1)
    _save(fig, "srf_xsec")
    plt.close(fig)


def plot_meridional(D, meta):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    cm = 100.0
    # the cylinder is drawn scaled to the cell's pump frequency (lengths x k_cyl / k_cell)
    k_cell = 2 * np.pi * meta["f_pump_Hz"] / C
    sc = (5.520078110286311 / float(D["cyl_wall"][:, 0].max())) / k_cell
    ext = [float(np.ptp(D["wall"][:, 1])), sc * float(np.ptp(D["cyl_wall"][:, 1]))]
    hgt = 2.9
    wid = hgt * 0.80 * sum(ext) / 0.42 + 1.1   # equal-aspect panels of height 0.42 m (in plot units), + colorbar
    fig, axs = plt.subplots(1, 2, figsize=(min(wid, W2), hgt), gridspec_kw={"width_ratios": ext, "wspace": 0.06})
    sets = [("mer", "wall", meta["eta_10"], r"(a) elliptical cell" "\n", 1.0),
            ("cyl", "cyl_wall", meta["cyl_eta"], r"(b) cylinder, $L/R = 0.791$" "\n", sc)]
    lim = max(np.nanmax(np.abs(D["mer_Bphi"] * D["mer_Ephi"])), np.nanmax(np.abs(D["cyl_Bphi"] * D["cyl_Ephi"])))
    for ax, (key, wk, eta, title, fac) in zip(axs, sets):
        r, z, tr = fac * D[f"{key}_r"], fac * D[f"{key}_z"], D[f"{key}_tris"]
        Bp, Ep = D[f"{key}_Bphi"], D[f"{key}_Ephi"]
        sg = np.sign(np.sum(Bp * Ep * r))
        ov = sg * Bp * Ep
        for mir in (1, -1):
            t = mtri.Triangulation(z * cm, mir * r * cm, tr)
            im = ax.tripcolor(t, ov, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="gouraud",
                              rasterized=True)
            tc = mtri.Triangulation(z * cm, mir * r * cm, tr, mask=r[tr].min(axis=1) < 0.03 * r.max())
            ax.tricontour(tc, Bp, levels=[0.0], colors=NODE_COLOR, linestyles="-", linewidths=0.9)
        Wl = fac * D[wk]
        ax.plot(Wl[:, 1] * cm, Wl[:, 0] * cm, color="0.15", lw=0.9)
        ax.plot(Wl[:, 1] * cm, -Wl[:, 0] * cm, color="0.15", lw=0.9)
        ax.axhline(0, color="0.5", lw=0.4, ls="-.")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$z$ [cm]")
        ax.set_title(title + rf"$\eta_{{10}} = {eta:.3f}$", fontsize=FS - 0.5, pad=3)
        ax.set_ylim(-21, 21)
    axs[1].tick_params(labelleft=False)
    axs[0].set_ylabel(r"$r$ [cm]")
    cb = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.03, pad=0.015, aspect=25)
    cb.set_label(r"$\tilde{\mathbf E}_1\!\cdot\!\tilde{\mathbf B}_0/(E_{1,\rm rms}B_0)$, volume average $\eta_{10}$",
                 fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 2)
    fig.text(0.45, 0.0, r"TM$_{020}$ pump / TE$_{011}$ signal; green: sign change of the pump $\tilde B_{0\phi}$;" "\n"
             r"the cylinder is scaled to the cell's frequency", ha="center", va="top", fontsize=FS - 1)
    _save(fig, "srf_overlap_meridional")
    plt.close(fig)


def plot_tuning(meta):
    import matplotlib.pyplot as plt

    sw = meta["tuning"]["sweep"]
    s = np.array([r["s"] for r in sw])
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"wspace": 0.32})
    ax = axs[0]
    for nm, col, ls in (("TM020", "#2166ac", "-"), ("TE011", "#b2182b", "-"), ("TM011", "0.55", "--"),
                        ("TM021", "0.55", ":"), ("TE021", "0.55", "-.")):
        f = np.array([np.nan if r[nm] is None else r[nm] for r in sw]) / 1e9
        lab = {"TM020": r"TM$_{020}$ (pump)", "TE011": r"TE$_{011}$ (signal)"}.get(nm, nm[:2] + r"$_{" + nm[2:] + "}$")
        ax.plot(s, f, ls, color=col, label=lab, lw=1.2 if nm in ("TM020", "TE011") else 0.8)
    cr = meta["tuning"]["crossing"]
    ax.plot([cr["axial_factor"]], [cr["f_Hz"] / 1e9], "o", ms=4, mfc="none", mec="k", mew=0.7)
    ax.annotate(rf"$s = {cr['axial_factor']:.4f}$, ${cr['f_Hz'] / 1e9:.4f}$ GHz", (cr["axial_factor"], cr["f_Hz"] / 1e9),
                xytext=(-70, 22), textcoords="offset points", fontsize=FS - 1.5,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.3"})
    ax.set_xlabel(r"axial shortening factor $s$")
    ax.set_ylabel(r"$f$ [GHz] (unscaled cell)")
    ax.set_ylim(1.2, 2.3)
    ax.legend(loc="upper right", fontsize=FS - 2, frameon=False, ncol=1)
    ax.set_title(r"(a) mode frequencies", fontsize=FS - 0.5, pad=3)
    ax = axs[1]
    fp = np.array([r["TM020"] for r in sw])
    fs = np.array([r["TE011"] for r in sw])
    split = (fs - fp) / fp * 100
    ax.plot(s, split, color="k", lw=1.2, label=r"$(f_{\rm TE_{011}} - f_{\rm TM_{020}})/f_0$")
    band = 2e6 / 1.4e9 * 100
    ax.axhspan(0, band, color="0.2", alpha=0.35, lw=0, label=rf"a 2 MHz splitting at 1.4 GHz (${band:.2f}\%$)")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(cr["axial_factor"], color="0.5", lw=0.5, ls=":")
    # local slope of the splitting at the crossing
    k = np.argmin(np.abs(s - cr["axial_factor"]))
    sl = np.polyfit(s[max(k - 2, 0):k + 3], split[max(k - 2, 0):k + 3], 1)[0]
    ax.text(0.03, 0.04, rf"slope at crossing: ${sl:.1f}\%$ per unit $s$" + "\n"
            + rf"$\Delta s$ for a 1\% change: ${1.0 / abs(sl):.4f}$",
            transform=ax.transAxes, fontsize=FS - 2, va="bottom")
    ax.set_xlabel(r"axial shortening factor $s$")
    ax.set_ylabel(r"splitting [\% of $f_0$]")
    ax.set_ylim(-8, 8)
    ax.legend(loc="upper right", fontsize=FS - 2.5, frameon=False)
    ax.set_title(r"(b) pump--signal splitting", fontsize=FS - 0.5, pad=3)
    _save(fig, "srf_tuning")
    plt.close(fig)
    return sl


def plot_wall(D, meta):
    import matplotlib.pyplot as plt

    cm = 100.0
    s, Bt, Bsg, E = D["wp_s"] * cm, D["wp_Bt"], D["wp_Bsigned"], D["wp_E"]
    r, z = D["wp_r"], D["wp_z"]
    fig, axs = plt.subplots(2, 1, figsize=(W1 * 1.4, 3.5), sharex=True, gridspec_kw={"hspace": 0.08,
                                                                                         "height_ratios": [1.3, 1]})
    ax = axs[0]
    ax.plot(s, Bt, color="#2166ac", label=r"pump TM$_{020}$ $|\tilde B_t|/B_0$")
    ax.plot(s, D["ws_Bt"], color="#b2182b", lw=0.8, label=r"signal TE$_{011}$ $|\tilde B_t|/B_{1,\rm rms}$")
    ax.plot(s, E, color="#e08214", lw=0.8, ls="--", label=r"pump $|\tilde E|/(cB_0)$")
    i = int(np.argmax(Bt))
    ax.plot(s[i], Bt[i], "o", ms=4, mfc="none", mec="k", mew=0.7)
    ax.annotate(rf"$B_{{\rm pk}} = {Bt[i]:.2f}\,B_0$", (s[i], Bt[i]), xytext=(4, 4), textcoords="offset points",
                fontsize=FS - 1.5)
    ie = int(np.argmax(E))
    ie = int(np.argmax(np.where(s < s[np.argmax(r)], E, -1)))  # the left one (mirror symmetric)
    ax.annotate(rf"$E_{{\rm pk}} = {E[ie]:.2f}\,cB_0$", (s[ie], E[ie]), xytext=(-62, -4), textcoords="offset points",
                fontsize=FS - 1.5)
    # field node of the pump on the wall (pickup location): sign changes of B_phi away from the axis caps
    ch = np.nonzero((np.sign(Bsg[1:]) != np.sign(Bsg[:-1])) & (r[1:] > 0.3 * REQ))[0]
    for j in ch:
        sn = float(np.interp(0, sorted([Bsg[j], Bsg[j + 1]]), [s[j], s[j + 1]] if Bsg[j] < Bsg[j + 1] else [s[j + 1], s[j]]))
        for a in axs:
            a.axvline(sn, color=NODE_COLOR, lw=0.8)
    if len(ch):
        ax.text(s[ch[-1]] + 0.5, 0.15, "pump node\n(pickup)", fontsize=FS - 2.5, va="bottom")
    # landmarks: equator, iris
    seq = s[np.argmax(r)]
    ax.set_ylabel(r"wall field")
    ax.legend(loc="upper center", fontsize=FS - 2.5, frameon=False, ncol=3, columnspacing=0.8, handlelength=1.6)
    ax.set_ylim(0, 2.55)
    ax = axs[1]
    ds = np.gradient(s)
    q = 2 * np.pi * r * Bt ** 2
    q /= np.sum(q * ds)
    qs = 2 * np.pi * r * D["ws_Bt"] ** 2
    qs /= np.sum(qs * ds)
    ax.plot(s, q, color="#2166ac", label=r"pump loss density $2\pi r|\tilde B_t|^2$")
    ax.plot(s, qs, color="#b2182b", lw=0.8, label=r"signal")
    # smallest set of wall points that carries 80 % of the pump loss (threshold on the density)
    o = np.argsort(q)[::-1]
    cum = np.cumsum((q * ds)[o])
    thr = q[o][np.searchsorted(cum, 0.8)]
    ax.fill_between(s, 0, q, where=q >= thr, color="#2166ac", alpha=0.25, lw=0, label=r"80\% of the pump loss")
    for a in axs:
        a.axvline(seq, color="0.6", lw=0.5, ls="-.")
    axs[0].text(seq + 0.3, 1.75, "equator", fontsize=FS - 2.5)
    ax.set_xlabel(r"arc length along the wall $s$ [cm] (from the left beam-tube cap)")
    ax.set_ylabel(r"normalized loss density [cm$^{-1}$]")
    ax.legend(loc="upper left", fontsize=FS - 2.5, frameon=False)
    ax.set_xlim(s.min(), s.max())
    axs[0].set_title(rf"wall fields of the cell; $G_{{\rm pump}} = {meta['G_pump_ohm']:.0f}\,\Omega$, "
                     rf"$G_{{\rm signal}} = {meta['G_signal_ohm']:.0f}\,\Omega$", fontsize=FS - 0.5, pad=3)
    _save(fig, "srf_wall_profile")
    plt.close(fig)


def stage_plot(args):
    _style(args.style)
    D = np.load(DATA / "cell2d.npz")
    meta = json.loads((DATA / "cell2d.json").read_text())
    D3 = meta3 = None
    if (DATA / "cell3d_z0.npz").exists():
        D3 = np.load(DATA / "cell3d_z0.npz")
        meta3 = json.loads((DATA / "cell3d.json").read_text())
        # max difference on the (x, y) grid, 2D (revolved) vs 3D, per mode up to the global sign, in units of the
        # 2D maximum on the plane
        ng = int(D3["n_grid"])
        P, ok = D3["pts"][:ng], D3["valid"][:ng]
        rq = np.hypot(P[:, 0], P[:, 1])
        for k2, k3, key in (("line_Bphi", "Bphi", "B"), ("line_Ephi", "Ephi", "E")):
            ref = np.interp(rq, D["line_r"], D[k2])
            v = D3[k3][:ng]
            sg = np.sign(np.sum(ref[ok] * v[ok]))
            meta3[f"xsec_maxdiff_{key}"] = float(np.max(np.abs(sg * v[ok] - ref[ok])) / np.nanmax(np.abs(D[k2])))
        (DATA / "cell3d.json").write_text(json.dumps(meta3, indent=1, default=float))
        print("3D vs 2D on z = 0: max diff B", meta3["xsec_maxdiff_B"], "E", meta3["xsec_maxdiff_E"])
    plot_xsec(D, meta, D3, meta3)
    plot_meridional(D, meta)
    sl = plot_tuning(meta)
    plot_wall(D, meta)
    print("tuning slope [% per unit s]", sl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("data", "data3d", "plot"))
    ap.add_argument("--style", default=None, help="optional matplotlib style module, run before plotting")
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--n3", type=float, default=16)
    ap.add_argument("--o3", type=int, default=3)
    a = ap.parse_args()
    {"data": stage_data, "data3d": stage_data3d, "plot": stage_plot}[a.stage](a)


if __name__ == "__main__":
    main()
