"""Publication-style figures (matplotlib): reach, signal/noise PSD, field cross-section.

All plotting is optional (matplotlib imported lazily); numbers come from the C++
core.  Functions return the matplotlib Figure.
"""
from __future__ import annotations

import base64
from typing import Sequence

import numpy as np

from . import scan


def _plt():
    import matplotlib.pyplot as plt  # no backend switch: headless runs fall back to Agg, notebooks keep inline

    return plt


def _decode(a: dict) -> np.ndarray:
    dt = {"float32": np.float32, "uint32": np.uint32}[a["dtype"]]
    return np.frombuffer(base64.b64decode(a["data"]), dtype=dt).reshape(a["shape"])


def reach(results_list: Sequence, labels: Sequence[str] | None = None, benchmark: str = "optimistic",
          amplitude: str = "as_designed", live_time_yr: float = 3.0, ax=None, models=("DFSZ", "KSVZ"), **kw):
    """Expected 95% reach g_95(m_a) for one or more solved designs, with QCD lines."""
    plt = _plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, res in enumerate(results_list):
        rc = res.reach(benchmark, amplitude=amplitude, live_time_yr=live_time_yr, **kw)
        lab = labels[i] if labels else res.design.name
        ax.loglog(rc["m_eV"] * 1e9, rc["g95_GeV"], lw=2, label=f"{lab} ({benchmark})")
    m = np.geomspace(0.3e-9, 6e-9, 200)
    for model, ls in zip(models, ("--", ":")):
        ax.loglog(m * 1e9, scan.g_qcd(m, model), "k" + ls, lw=1, label=model)
    ax.set_xlabel(r"$m_a$ [neV]")
    ax.set_ylabel(r"$|g_{a\gamma\gamma}|$ [GeV$^{-1}$] (expected 95% limit)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    return fig or ax.figure


def psd(res, f0_Hz: float, benchmark: str = "optimistic", amplitude: str = "required", fig=None, **kw):
    """Readout-current PSDs (Foster normalization) at a tuning f0: left, noise components across the
    sensitivity band; right, zoom on the axion line (DFSZ coupling) against the total noise."""
    plt = _plt()
    Psi = res.Psi * (res.benchmark(benchmark)["amplitude_scale"] if amplitude == "required" else 1.0)
    p = scan.psd(Psi, res.L_p, f0_Hz, benchmark, **kw)
    if fig is None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    else:
        a1, a2 = fig.subplots(1, 2)
    f = p["f_Hz"]
    df = f - f0_Hz
    a1.semilogy(df, p["lambda_B_A2_s"], "k", lw=1.6, label=r"total noise $\lambda_B$")
    Z2 = p["N_V_V2_s"] / np.maximum(p["lambda_B_A2_s"], 1e-300)  # |Z|^2
    a1.semilogy(df, p["SV_th_V2_s"] / Z2, "C0", lw=1, label="thermal")
    a1.semilogy(df, p["SV_ba_V2_s"] / Z2, "C2", lw=1, label="SQUID backaction")
    a1.semilogy(df, p["SI_imp_A2_s"], "C1", lw=1, label="SQUID imprecision")
    a1.set_xlabel(r"$f - f_0$ [Hz]")
    a1.set_ylabel(r"current PSD [A$^2$ s]")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title(f"{res.design.name}: f0 = {f0_Hz/1e3:.2f} kHz, {benchmark}", fontsize=9)
    sel = p["lambda_a_A2_s"] > 0
    fa = f[sel].min() if sel.any() else f0_Hz
    m = (f >= fa - 2e-6 * fa) & (f <= fa + 1e-5 * fa)
    a2.semilogy((f[m] - fa) * 1e3, p["N_V_V2_s"][m], "k", lw=1.6, label=r"noise $N_V$ (referred to EMF)")
    sv = np.where(p["S_Va_V2_s"][m] > 0, p["S_Va_V2_s"][m], np.nan)
    a2.semilogy((f[m] - fa) * 1e3, sv, "C3", lw=1.6, label=r"axion EMF $S_{V,a}$ (SHM line)")
    a2.set_xlabel(r"$f - f_a$ [mHz]")
    a2.set_ylabel(r"voltage PSD [V$^2$ s]")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title(f"axion line, m_a = f0, g = DFSZ ({amplitude} amplitude)", fontsize=9)
    fig.tight_layout()
    return fig


def section(scene: dict, quantity: str = "Bmag", log: bool = False, ax=None, lines: bool = True, cmap=None):
    """Meridional cross-section (mirrored to r < 0) colored by a field quantity."""
    plt = _plt()
    import matplotlib.tri as mtri

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
    fld = scene["field"]
    V = _decode(fld["vertices"]).astype(float)
    I = _decode(fld["indices"]).astype(int)
    q = _decode(fld["quantities"][quantity]).astype(float).ravel()
    signed = quantity in ("Br", "Bz", "Bphi", "psi", "pickup_psi")
    if log:
        q = np.log10(np.maximum(np.abs(q), 1e-12))
    vmax = np.nanmax(np.abs(q)) if signed else None
    cm = cmap or ("RdBu_r" if signed else "viridis")
    for sgn in (1, -1):
        tri = mtri.Triangulation(sgn * V[:, 0], V[:, 1], I)
        tp = ax.tripcolor(tri, q, shading="gouraud", cmap=cm, vmin=-vmax if signed else None, vmax=vmax)
    for reg in scene["regions"]:
        if reg["role"] in ("vacuum",):
            continue
        RV = _decode(reg["triangles"]["vertices"]).astype(float)
        RI = _decode(reg["triangles"]["indices"]).astype(int)
        if reg["role"] in ("superconductor", "iron", "conductor"):
            for sgn in (1, -1):
                ax.tripcolor(mtri.Triangulation(sgn * RV[:, 0], RV[:, 1], RI), np.zeros(len(RV)),
                             cmap="Greys", vmin=-1, vmax=3, shading="flat")
        for o in reg["outline"]:
            P = _decode(o).astype(float)
            for sgn in (1, -1):
                ax.plot(sgn * P[:, 0], P[:, 1], color=reg["color"], lw=0.8)
    if lines:
        for key, col in (("static", "w"), ("pickup", "C1")):
            seg = _decode(scene["field_lines"][key]).astype(float) if key in scene.get("field_lines", {}) else None
            if seg is not None and len(seg):
                from matplotlib.collections import LineCollection

                S = seg.reshape(-1, 2, 2)
                for sgn in (1, -1):
                    S2 = S.copy()
                    S2[:, :, 0] *= sgn
                    ax.add_collection(LineCollection(S2, colors=col, linewidths=0.4, alpha=0.8))
    ax.axvline(0, color="k", lw=0.6, ls="-.")
    v = scene["view"]
    ax.set_xlim(-v["r_max"], v["r_max"])
    ax.set_ylim(v["z_min"], v["z_max"])
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    unit = scene["field"]["units"].get(quantity, "")
    cb = plt.colorbar(tp, ax=ax, shrink=0.85)
    cb.set_label(("log10 " if log else "") + f"{quantity} [{unit}]")
    ax.set_title(scene["name"], fontsize=9)
    return fig or ax.figure


def fullwave(results_list: Sequence, labels: Sequence[str] | None = None, x: str = "f", fig=None):
    """Beyond-MQS response from `results.fullwave.sweep` (designs solved with a `fullwave` section).

    Three panels against frequency (x="f", MHz) or kR (x="kR"): |Psi(omega)|/Psi_MQS, L(omega)/L_MQS (sign
    changes mark self-resonances; capacitive points are drawn hollow) and the quality factors Q_rad, Q_bulk
    and Q_total where available."""
    plt = _plt()
    if not isinstance(results_list, (list, tuple)):
        results_list = [results_list]
    labels = labels or [getattr(getattr(r, "design", None), "name", f"design {i}") for i, r in enumerate(results_list)]
    fig = fig or plt.figure(figsize=(7.2, 8.0))
    axs = fig.subplots(3, 1, sharex=True)
    for res, lab in zip(results_list, labels):
        rows = res.fullwave.get("sweep", []) if hasattr(res, "fullwave") else res["fullwave"]["sweep"]
        if not rows:
            continue
        xs = np.array([r["f_Hz"] / 1e6 if x == "f" else r["k_R"] for r in rows])
        psi = np.array([r.get("Psi_over_MQS", np.nan) for r in rows], dtype=float)
        lr = np.array([r.get("L_over_MQS", np.nan) for r in rows], dtype=float)
        line, = axs[0].plot(xs, psi, "-o", ms=3, label=lab)
        c = line.get_color()
        axs[1].plot(xs, np.abs(lr), "-", color=c)
        ind, cap = lr > 0, lr <= 0
        axs[1].plot(xs[ind], np.abs(lr[ind]), "o", ms=3, color=c)
        axs[1].plot(xs[cap], np.abs(lr[cap]), "o", ms=4, mfc="none", color=c)
        for key, sty in (("Q_rad", "-"), ("Q_bulk", "--"), ("Q_total", ":")):
            q = np.array([r.get(key, np.nan) or np.nan for r in rows], dtype=float)
            if np.isfinite(q).any():
                axs[2].plot(xs, q, sty, color=c, label=f"{lab}: {key}")
    axs[0].set_ylabel(r"$|\Psi(\omega)|/\Psi_{\rm MQS}$")
    axs[1].set_ylabel(r"$|L(\omega)|/L_{\rm MQS}$ (hollow: capacitive)")
    axs[2].set_ylabel("Q")
    for a in axs:
        a.set_xscale("log")
        a.grid(True, which="both", alpha=0.3)
    axs[1].set_yscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("frequency [MHz]" if x == "f" else "kR")
    axs[0].legend(fontsize=8)
    if axs[2].lines:
        axs[2].legend(fontsize=7)


def force_distribution(res, bodies: Sequence[str] | None = None, quantity: str = "f_z",
                       axes=None, annotate: bool = True):
    """Distributed Lorentz / Maxwell load of a solved axisymmetric design (`forces.distributions`).

    Two panels: the load against arc length (surfaces) or z (winding volumes), and the load drawn as
    arrows on the meridional (r, z) outline of each body.

    quantity: 'f_z' | 'f_r' (force per unit meridional length, N/m, for surfaces; per cell, N, for
    winding volumes), 'traction_z' | 'traction_r' (Pa, surfaces only), or
    'density_z' | 'density_r' (N/m^3, winding volumes only).

    bodies: which entries of forces.distributions to show (default: all).  Note that a winding has
    both a 'volume' entry (the J x B density -- the accurate one) and a '<name>:boundary' surface
    entry, whose net force is a small difference of large tractions.

    Needs the full results document (`Results.data`), not the compacted MCP reply.
    """
    plt = _plt()
    data = res.data if hasattr(res, "data") else res
    dists = (data.get("forces") or {}).get("distributions") or {}
    if not dists:
        raise ValueError("this result has no forces.distributions; solve an axisymmetric design with "
                         "at least one superconductor, iron body or winding")
    names = list(bodies) if bodies else list(dists)
    missing = [n for n in names if n not in dists]
    if missing:
        raise KeyError(f"no distribution for {missing}; available: {list(dists)}")
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    else:
        fig = axes[0].figure
    keys = {"f_z": ("f_z_N_per_m", "f_z_N"), "f_r": ("f_r_N_per_m", "f_r_N"),
            "traction_z": ("traction_z_Pa", None), "traction_r": ("traction_r_Pa", None),
            "density_z": (None, "force_density_z_N_per_m3"), "density_r": (None, "force_density_r_N_per_m3")}
    if quantity not in keys:
        raise ValueError(f"quantity must be one of {sorted(keys)}")
    ax0, ax1 = axes[0], axes[1]
    units = {"f_z": "N/m or N", "f_r": "N/m or N", "traction_z": "Pa", "traction_r": "Pa",
             "density_z": "N/m^3", "density_r": "N/m^3"}
    drawn: list[tuple[int, str]] = []
    for i, name in enumerate(names):
        d = dists[name]
        surface = d.get("geometry", "surface") == "surface"
        key = keys[quantity][0 if surface else 1]
        if key is None or key not in d:
            continue
        v = np.asarray(d[key], float)
        r, z = np.asarray(d["r_m"], float), np.asarray(d["z_m"], float)
        x = np.asarray(d["arclength_m"], float) if surface else z
        order = np.argsort(x)
        lab = f"{name} [{d['kind']}, {d.get('geometry', 'surface')}]"
        if annotate:
            lab += f"  Fz = {d['Fz_N']:.4g} N"
        ax0.plot(x[order], v[order], lw=1.2, label=lab, color=f"C{i}")
        ax1.plot(r, z, ".", ms=2, color=f"C{i}", label=lab)
        drawn.append((i, name))
    # arrows: both components, scaled per body (only for the bodies panel 0 could show)
    for i, name in drawn:
        d = dists[name]
        surface = d.get("geometry", "surface") == "surface"
        kr = keys["f_r"][0 if surface else 1]
        kz = keys["f_z"][0 if surface else 1]
        if kr not in d or kz not in d:
            continue
        r, z = np.asarray(d["r_m"], float), np.asarray(d["z_m"], float)
        fr, fz = np.asarray(d[kr], float), np.asarray(d[kz], float)
        n = max(1, len(r) // 60)
        span = max(float(np.ptp(r)), float(np.ptp(z)), 1e-12)
        norm = max(float(np.max(np.hypot(fr, fz))), 1e-300)
        s = 0.25 * span / norm
        ax1.quiver(r[::n], z[::n], fr[::n] * s, fz[::n] * s, color=f"C{i}",
                   angles="xy", scale_units="xy", scale=1, width=0.004)
    ax0.axhline(0, color="k", lw=0.6)
    ax0.set_xlabel("arc length along the surface [m]  /  z [m] for winding volumes")
    ax0.set_ylabel(f"{quantity} [{units[quantity]}]")
    ax0.legend(fontsize=7)
    ax0.set_title("distributed load", fontsize=9)
    ax1.set_xlabel("r [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_aspect("equal")
    ax1.set_title("load direction (arrows scaled per body)", fontsize=9)
    fig.tight_layout()
    return fig


def hts(res, fig=None):
    """Critical-state history from `results.hts` (designs solved with an `hts` section).

    Four panels: the drive waveform, the screening current and its magnetic moment, the
    screening-current-induced field at the field-of-view points, and the cumulative dissipation; plus the
    final screening-current profile J(r)/Jc of the first region."""
    plt = _plt()
    h = res.hts if hasattr(res, "hts") else res["hts"]
    steps = h.get("steps", [])
    if not steps:
        raise ValueError("this result has no critical-state history (add an 'hts' section and solve)")
    fig = fig or plt.figure(figsize=(7.2, 9.0))
    axs = fig.subplots(5, 1)
    t = np.array([s["t_s"] for s in steps])
    axs[0].plot(t, [s["scale"] for s in steps], "-o", ms=3)
    axs[0].set_ylabel("drive scale")
    for i, reg in enumerate(h["regions"]):
        name = reg["region"]
        axs[1].plot(t, [s["regions"][i]["screening_current_A"] for s in steps], "-", label=f"{name}: I")
        axs[1].plot(t, [s["regions"][i]["moment_Am2"] for s in steps], "--", label=f"{name}: m")
    axs[1].set_ylabel("screening current [A],\nmoment [A m$^2$]")
    axs[1].legend(fontsize=7)
    fov = steps[-1].get("field_of_view", [])
    for k, p in enumerate(fov):
        axs[2].plot(t, [s["field_of_view"][k]["scif_Bz_T"] for s in steps], "-",
                    label=f"({p['r_m']:.3g}, {p['z_m']:.3g}) m")
    axs[2].set_ylabel(r"SCIF $B_z$ [T]")
    if fov:
        axs[2].legend(fontsize=7)
    axs[3].plot(t, [s["cumulative_loss_J"] for s in steps], "-")
    axs[3].set_ylabel("dissipation [J]")
    axs[3].set_xlabel("time [s]")
    prof = h.get("final", {}).get("current_profiles", [])
    if prof:
        pts = prof[0]["points"]
        r = np.array([p["r_m"] for p in pts])
        jc = np.array([p["Jc_A_m2"] for p in pts])
        j = np.array([p["J_A_m2"] for p in pts])
        axs[4].plot(r, np.where(jc > 0, j / np.where(jc > 0, jc, 1), 0.0), "-o", ms=3)
        axs[4].set_ylabel(r"$J/J_c$ (final)")
        axs[4].set_xlabel(f"r [m] at z = {prof[0]['z_m']:.4g} m")
    for a in axs:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
