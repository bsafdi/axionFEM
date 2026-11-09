"""Validation of the axisymmetric m = 0 torsional dynamics.

    python benchmarks/torsion.py [workdir]

Closed-form references
----------------------
1. *Rigid rotation.*  u_phi = c r has both shear strains zero, so it is an exact zero mode of the
   stiffness operator: the lowest eigenfrequency of an unsupported body must be 0.

2. *Free-free torsional modes of a cylinder or annulus (exact, non-dispersive).*  Substituting
   u_phi = r g(z) in the weak form gives -mu g'' = rho omega^2 g with g'(0) = g'(L) = 0, hence

       f_n = n c_T/(2 L),   c_T = sqrt(mu/rho),  n = 1, 2, ...

   for any inner/outer radius, exactly (the r dependence drops out of both the stiffness and the
   mass form).  This is the classical fundamental torsional branch of a rod.

3. *Radial shear ("torsional") modes of a disc.*  For z-uniform
   u_phi(r) the radial equation is u'' + u'/r - u/r^2 + k^2 u = 0, so u = J_1(k r), and the
   traction-free edge sigma_r phi = mu(u' - u/r) = 0 at r = a requires

       k a J_0(k a) = 2 J_1(k a),      f = k c_T/(2 pi),

   exact for any thickness (the z-uniform modes satisfy the traction-free faces identically).

4. *Drive-point mobility of a long tube against the T(0,1) waveguide (asymptotic).*  A body force
   g(r, z) = p(z)/h uniform through the wall of a long thin tube (mean circumference b, areal mass
   rho_s = rho h, shear stiffness S = mu h) drives torsional waves away from the drive point; the
   band mean of Re[Y] tends to the infinite-waveguide value, Skudrzyk's mean-value theorem. As
   eta -> 0 that is (b |P(k)|)^2 Y_inf, Y_inf = 1/(2 b sqrt(S rho_s)), P(k) the axial transform of p.
   At finite eta it is the damped infinite waveguide,
       Re Y_d(w) = T^2/(pi J mu) int_0^inf |P(k)|^2 w eta k^2/((k^2 - k0^2)^2 + eta^2 k^4) dk,
   T = 2 pi int w(r) r^2 dr, J = 2 pi int r^3 dr, k0 = w/c_T, which differs from the lossless value at
   first order in eta with a sign set by the shape of |P(k)|^2.
   The comparison must use the transform of the load the FEM *applies*: the P_p interpolant of the
   nodal values. For a step load the edge nodes decide whether the applied
   load is W_z + h_e/3 or W_z - h_e/3 wide (the record's earlier "+15 %"). So:
   4a. rect ring loads, compared with the nominal sinc asymptote (history)
       and with the asymptote of the applied load, |P(k)|^2 from the FE participations of the even
       modes, Re Y(k_n) = c_n^2 L/(4 c_T);
   4b. a smooth Gaussian ring load (sigma_z = 0.1 m) against its closed-form asymptote, the
       validation proper.

5. *Validity range of the T(0,1) asymptote*.  It is used here only below the first
   thickness-shear cutoff, f < c_T/(2 t), and for loads uniform through the wall (the conservative
   statement; below the cutoff only T(0,1) propagates, so other radial profiles work with their own
   torque, but that is not validated here).  A traction concentrated at a
   surface drives the T(0, n >= 2) branches above their cutoffs and Re Y exceeds the T(0,1) value
   by several times: a short thick tube (a = 0.5 m, t = 0.04 m, cutoff 39 kHz), Gaussian in z,
   uniform vs surface-concentrated through the wall, against the T(0,1) value of each profile,
   Re Y = (2 pi int w r^2 dr)^2 |P(k)|^2/(2 rho c_T J), J = 2 pi int r^3 dr.  The asymptote must
   not be used for surface loads above c_T/(2 t) (0.156 MHz for a 1 cm wall).

Convention: e^{-i omega t}, (K (1 - i eta) - omega^2 M) u = F, Y = -i omega q (Re Y is
the same under either convention).

Residuals: the record keeps every mode's relative residual; the report shows the largest over the
elastic modes (index >= 1). The rigid mode's relative residual is 0/0 and carries no information.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, special

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records"
OUT = ROOT / "docs" / "validation" / "torsion.md"
STAMP = {"date": time.strftime("%Y-%m-%d %H:%M")}


def _stamp():
    """Date, source and build of the run, read at its start.

    `commit`/`source_dirty`: the checkout's HEAD and whether it has tracked changes or untracked files under the
    source paths; `build.source_commit`/`build.source_dirty`: the same, embedded in the compiled module when it was
    built. A stale build shows as build.source_commit != commit. Outside a git checkout both checkout fields are
    "unknown"."""
    import subprocess
    out = dict(STAMP)
    paths = ["cpp", "python", "benchmarks", "tests", "CMakeLists.txt"]
    try:
        out["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                       text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--"] + paths,
                               capture_output=True, text=True, check=True).stdout.strip()
        out["source_dirty"] = dirty != ""
    except (OSError, subprocess.CalledProcessError):
        out["commit"] = "unknown"
        out["source_dirty"] = "unknown"
    out["build"] = af.build_info()
    out["build"].get("library", {}).pop("build_dir", None)   # an absolute path: not written
    out["build_matches_checkout"] = out["build"].get("source_commit") == out["commit"]
    return out

E_S, NU_S, RHO_S = 200e9, 0.3, 7900.0
MU_S = E_S / (2 * (1 + NU_S))
CT_S = math.sqrt(MU_S / RHO_S)


def _mat():
    return {"steel": {"kind": "linear", "density": RHO_S,
                      "mechanical": {"E_Pa": E_S, "nu": NU_S, "alpha": 0.0}}}


def _dummy_coil(r0=40.0):
    return ({"name": "coil", "shape": {"rect": {"r": [r0, r0 + 0.05], "z": [-0.05, 0.05]}},
             "material": "winding"},
            [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}])


def tube_design(a, t, L, size, order=2):
    coil, wind = _dummy_coil()
    return af.Design({
        "schema": "axionfem.design/1", "name": "torsion_tube", "materials": _mat(),
        "regions": [
            {"name": "tube", "shape": {"rect": {"r": [a, a + t], "z": [-L / 2, L / 2]}},
             "material": "steel", "mesh_size": size},
            coil],
        "windings": wind,
        "mesh": {"order": order, "size": max(L, a) / 6.0, "domain_radius": 3.0 * max(L, a, 40.0)}})


def disc_design(a, h, size, order=2):
    coil, wind = _dummy_coil()
    return af.Design({
        "schema": "axionfem.design/1", "name": "torsion_disc", "materials": _mat(),
        "regions": [
            {"name": "disc", "shape": {"rect": {"r": [0.0, a], "z": [-h / 2, h / 2]}},
             "material": "steel", "mesh_size": size},
            coil],
        "windings": wind,
        "mesh": {"order": order, "size": a / 6.0, "domain_radius": 3.0 * max(a, 40.0)}})


def disc_roots(n=4):
    """Roots of x J_0(x) = 2 J_1(x) (excluding x = 0)."""
    f = lambda x: x * special.jv(0, x) - 2.0 * special.jv(1, x)
    out, x = [], 1.0
    while len(out) < n:
        a, b = x, x + 0.1
        while f(a) * f(b) > 0 and b < 200:
            a, b = b, b + 0.1
        if b >= 200:
            break
        out.append(optimize.brentq(f, a, b, xtol=1e-13))
        x = b + 0.2
    return out


def case_tube(wd: Path, rows):
    a, t, L = 1.0, 0.02, 4.0
    exact = [0.0] + [n * CT_S / (2 * L) for n in range(1, 6)]
    for order in (1, 2):
        for size in (0.20, 0.10, 0.05):
            d = tube_design(a, t, L, size, order)
            res = d.solve(wd / f"tube_{order}_{size}", verbose=0)
            tr = res.torsion(nmodes=6)
            f = tr["modes"]["f_Hz"]
            rows.append(dict(case="tube_free_free", order=order, mesh_size=size,
                             dofs=res.data["mesh"]["dofs"],
                             free_dofs=tr["diagnostics"]["free_dofs"],
                             mass_kg=tr["diagnostics"]["solid_mass_kg"],
                             mass_exact_kg=RHO_S * math.pi * ((a + t) ** 2 - a ** 2) * L,
                             f_Hz=f, f_exact_Hz=exact,
                             rigid_f_Hz=f[0], rigid_over_f1=f[0] / exact[1],
                             rel_err=[abs(f[i] / exact[i] - 1) for i in range(1, min(len(f), 6))],
                             residual=tr["modes"]["residual"],
                             max_elastic_residual=max(tr["modes"]["residual"][1:]),
                             inertia=tr["modes"]["diagnostics"]["inertia"],
                             warnings=tr["warnings"]))
    return rows


def case_disc(wd: Path, rows):
    a, h = 1.0, 0.02
    xs = disc_roots(4)
    exact = [x * CT_S / (2 * math.pi * a) for x in xs]
    for order in (1, 2):
        for size in ((0.10, 0.05, 0.025) if order == 1 else (0.10, 0.05)):
            d = disc_design(a, h, size, order)
            res = d.solve(wd / f"disc_{order}_{size}", verbose=0)
            tr = res.torsion(nmodes=5)
            f = tr["modes"]["f_Hz"]
            rows.append(dict(case="thin_disc_free_edge", order=order, mesh_size=size,
                             dofs=res.data["mesh"]["dofs"],
                             free_dofs=tr["diagnostics"]["free_dofs"],
                             f_Hz=f, f_exact_Hz=exact, roots=xs,
                             rigid_f_Hz=f[0],
                             rel_err=[abs(f[i + 1] / exact[i] - 1)
                                      for i in range(min(len(f) - 1, len(exact)))],
                             residual=tr["modes"]["residual"],
                             max_elastic_residual=max(tr["modes"]["residual"][1:]),
                             inertia=tr["modes"]["diagnostics"]["inertia"],
                             warnings=tr["warnings"]))
    return rows


def _tube_wall_moments(a, t):
    I2 = ((a + t) ** 3 - a ** 3) / 3                 # int r^2 dr over the wall
    I3 = ((a + t) ** 4 - a ** 4) / 4                 # int r^3 dr over the wall
    return I2, 2 * math.pi * I3                      # J = 2 pi int r^3 dr


def _thick_asym(P2, a, t):
    """T(0,1) infinite-waveguide Re Y for g = p(z)/t uniform through the wall, |P(k)|^2 = P2."""
    I2, J = _tube_wall_moments(a, t)
    return (2 * math.pi * I2 / t) ** 2 * P2 / (2 * RHO_S * CT_S * J)


def damped_waveguide_re_y(P2fun, T, J, eta, freqs):
    """Re Y_d of the damped infinite T(0,1) waveguide, one quad per frequency:
    T^2/(pi J mu) int_0^inf |P(k)|^2 w eta k^2/((k^2 - k0^2)^2 + eta^2 k^4) dk."""
    from scipy import integrate
    out = []
    for f in freqs:
        w = 2 * math.pi * f
        k0 = w / CT_S
        fun = lambda k: P2fun(k) * w * eta * k * k / ((k * k - k0 * k0) ** 2 + eta * eta * k ** 4)  # noqa: E731
        width = max(eta * k0, 1e-6 * k0)
        pts = [k0 - 5 * width, k0, k0 + 5 * width]
        val = integrate.quad(fun, 0.0, 4 * k0, points=[p for p in pts if 0 < p < 4 * k0], limit=400)[0]
        val += integrate.quad(fun, 4 * k0, np.inf, limit=200)[0]
        out.append(T * T / (math.pi * J * MU_S) * val)
    return np.array(out)


def case_mobility(wd: Path, rows):
    """Ring loads on a long tube vs the infinite-waveguide mobility (4a rect, 4b Gaussian).

    Skudrzyk's theorem is about the *frequency average*: the mean of Re[Y] over a band containing
    many modes equals the infinite-structure value.  At an individual frequency the finite tube is
    resonant and Re[Y] swings by orders of magnitude.  The centred load drives the even modes only,
    so the modal overlap of the participating modes is eta f/(2 df).
    """
    a, t, L = 1.0, 0.01, 20.0
    h = t
    rho_s = RHO_S * h
    S = MU_S * h
    b = 2 * math.pi * (a + t / 2)
    Y_inf = 1.0 / (2.0 * b * math.sqrt(S * rho_s))
    I2, J = _tube_wall_moments(a, t)
    df = CT_S / (2 * L)                      # mode spacing of the free-free tube
    res = tube_design(a, t, L, 0.05, 2).solve(wd / "mobility", verbose=0)
    tr0 = res.torsion(dofs=True)
    r = np.array(tr0["dof_r"])
    z = np.array(tr0["dof_z"])
    wall = (r >= a - 1e-9) & (r <= a + t + 1e-9)
    bands = ((2e3, 4e3), (4e3, 8e3))
    k_top = 2 * math.pi * bands[-1][1] / CT_S
    nmodes = int(k_top * L / math.pi) + 8    # every mode up to the top of the last band, and a margin
    loads = []
    for wl in (0.2, 0.4):
        g = np.where(np.abs(z) <= wl / 2, 1.0 / h, 0.0)
        loads.append(("rect", wl, np.where(wall, g, 0.0)))
    sig = 0.1
    loads.append(("gauss", sig, np.where(wall, np.exp(-z ** 2 / (2 * sig * sig)) / h, 0.0)))
    for kind, wl, g in loads:
        md = res.torsion(nmodes=nmodes, load=g)
        c = np.array(md["modes"]["participation"])
        n = np.arange(len(c))
        even = (n % 2 == 0) & (n > 0)
        k_even = n[even] * math.pi / L
        asym_even = c[even] ** 2 * L / (4 * CT_S)        # the applied load's T(0,1) Re Y at k_n
        applied = md["load"]
        if kind == "rect":
            W = b * wl                                   # int g dA over the nominal ring
            torque_nominal = 2 * math.pi * I2 / h * wl
        else:
            W = b * wl * math.sqrt(2 * math.pi)
            torque_nominal = 2 * math.pi * I2 / h * wl * math.sqrt(2 * math.pi)
        for eta in (3e-3, 3e-2):
            for f0, f1 in bands:
                freqs = list(np.linspace(f0, f1, 241))
                hr = res.torsion(load=g, freqs_Hz=freqs, eta=eta)["harmonic"]["response"]
                Y = np.array([x["Y_re"] for x in hr])
                fm = math.sqrt(f0 * f1)
                kz = 2 * math.pi * np.array(freqs) / CT_S
                asym_applied = np.interp(kz, k_even, asym_even)
                row = dict(case="ring_load_mobility" if kind == "rect" else "gauss_load_mobility",
                           ring_width_m=wl, eta=eta, band_Hz=[f0, f1], n_freqs=len(freqs),
                           b_m=b, W=W, mode_spacing_Hz=df, modal_overlap=eta * fm / df,
                           modal_overlap_participating=eta * fm / (2 * df),
                           mean_Re_Y=float(np.mean(Y)),
                           torque_applied_over_nominal=applied["torque_N_m"] / torque_nominal,
                           applied_load=applied,
                           ratio_mean_over_applied=float(np.mean(Y) / np.mean(asym_applied)))
                if kind == "rect":
                    kappa2 = np.sinc(kz * wl / 2 / math.pi) ** 2    # np.sinc(x) = sin(pi x)/(pi x)
                    row.update(Y_inf_scaled=W ** 2 * Y_inf,
                               ratio_mean=float(np.mean(Y) / (W ** 2 * Y_inf)),
                               ratio_min=float(np.min(Y) / (W ** 2 * Y_inf)),
                               ratio_max=float(np.max(Y) / (W ** 2 * Y_inf)),
                               lambda_over_width=CT_S / fm / wl,
                               kappa2_mean=float(np.mean(kappa2)),
                               ratio_mean_over_kappa2=float(np.mean(Y) / (W ** 2 * Y_inf)
                                                            / np.mean(kappa2)),
                               applied_over_nominal_asymptote=float(
                                   np.mean(asym_applied) / (W ** 2 * Y_inf * np.mean(kappa2))))
                else:
                    P2 = 2 * math.pi * wl * wl * np.exp(-(kz * wl) ** 2)
                    closed = _thick_asym(P2, a, t)
                    damped = damped_waveguide_re_y(
                        lambda k, s_=wl: 2 * math.pi * s_ * s_ * np.exp(-(k * s_) ** 2),
                        2 * math.pi * I2 / h, J, eta, freqs)
                    row.update(sigma_z_m=wl,
                               ratio_mean_over_closed_form=float(np.mean(Y) / np.mean(closed)),
                               damped_over_lossless=float(np.mean(damped) / np.mean(closed)),
                               ratio_mean_over_damped=float(np.mean(Y) / np.mean(damped)),
                               ratio_min=float(np.min(Y / closed)),
                               ratio_max=float(np.max(Y / closed)))
                rows.append(row)
    return rows


def case_surface_load(wd: Path, rows):
    """5. The T(0,1) asymptote above the first thickness-shear cutoff."""
    from scipy import integrate
    a, t, L = 0.5, 0.04, 0.6
    bo = a + t
    fun = lambda f: (special.jv(2, 2 * math.pi * f / CT_S * a) * special.yv(2, 2 * math.pi * f / CT_S * bo)  # noqa
                     - special.jv(2, 2 * math.pi * f / CT_S * bo) * special.yv(2, 2 * math.pi * f / CT_S * a))
    fc = optimize.brentq(fun, 0.7 * CT_S / (2 * t), 1.3 * CT_S / (2 * t))
    coil, wind = _dummy_coil(1.5)
    doc = {"schema": "axionfem.design/1", "name": "short_tube", "materials": _mat(),
           "regions": [{"name": "tube", "shape": {"rect": {"r": [a, bo], "z": [-L / 2, L / 2]}},
                        "material": "steel", "mesh_size": 0.002}, coil],
           "windings": wind, "mesh": {"order": 2, "size": 0.5, "domain_radius": 4.0}}
    res = af.Design(doc).solve(wd / "surface_load", verbose=0)
    tr0 = res.torsion(dofs=True)
    r = np.array(tr0["dof_r"])
    z = np.array(tr0["dof_z"])
    wall = (r >= a - 1e-9) & (r <= bo + 1e-9)
    sz = 0.005
    delta = 0.003
    profiles = {"uniform": lambda x: np.ones_like(x) / t,
                "surface": lambda x: np.exp(-(x - a) / delta) / (delta * (1 - math.exp(-t / delta)))}
    J = 2 * math.pi * (bo ** 4 - a ** 4) / 4
    freqs = [10e3, 25e3, 40e3, 60e3, 80e3, 100e3, 120e3]
    k = 2 * math.pi * np.array(freqs) / CT_S
    P2 = 2 * math.pi * sz * sz * np.exp(-(k * sz) ** 2)
    eta = 0.1
    for name, w in profiles.items():
        g = np.where(wall, w(r) * np.exp(-z ** 2 / (2 * sz * sz)), 0.0)
        tq = integrate.quad(lambda x: w(np.array([x]))[0] * x * x, a, bo)[0]
        asym = (2 * math.pi * tq) ** 2 * P2 / (2 * RHO_S * CT_S * J)
        hr = res.torsion(load=g, freqs_Hz=freqs, eta=eta)["harmonic"]["response"]
        Y = np.array([x["Y_re"] for x in hr])
        rows.append(dict(case="surface_load_cutoff", profile=name, a_m=a, t_m=t, L_m=L, eta=eta,
                         sigma_z_m=sz, surface_efold_m=delta if name == "surface" else None,
                         cutoff_T02_Hz=fc, c_T_over_2t_Hz=CT_S / (2 * t),
                         free_dofs=tr0["diagnostics"]["free_dofs"], f_Hz=freqs,
                         ratio_to_T01=[float(x) for x in Y / asym]))
    return rows


def main():
    wd = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "torsion")
    wd.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()                     # at the start of the run
    rows = []
    case_tube(wd, rows)
    case_disc(wd, rows)
    case_mobility(wd, rows)
    case_surface_load(wd, rows)
    rec = dict(stamp=stamp, material=dict(E_Pa=E_S, nu=NU_S, rho=RHO_S, mu_Pa=MU_S,
                                          c_T_m_s=CT_S), rows=rows)
    (REC / "torsion.json").write_text(json.dumps(rec, indent=1))
    # report
    lines = ["# Validation: axisymmetric m = 0 torsional dynamics", "",
             '!!! warning "Beta"', "    The axisymmetric torsional analysis (`Results.torsion`) is beta.", "",
             "Generated by `benchmarks/torsion.py`; the record stores the commit it was made from.", "",
             f"Steel: E = {E_S:.3g} Pa, nu = {NU_S}, rho = {RHO_S} kg/m^3, "
             f"mu = {MU_S:.4g} Pa, c_T = {CT_S:.5g} m/s.", ""]
    lines += ["## 1-2. Free-free tube (exact f_n = n c_T/2L) and the rigid-rotation zero mode", "",
              "| order | h | free DOFs | f_rigid/f_1 | max rel. err. f_1..f_5 | max elastic residual |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if r["case"] != "tube_free_free":
            continue
        lines.append(f"| {r['order']} | {r['mesh_size']} | {r['free_dofs']} | "
                     f"{r['rigid_over_f1']:.2e} | {max(r['rel_err']):.3e} | "
                     f"{r['max_elastic_residual']:.2e} |")
    lines += ["", "## 3. Thin disc, free edge (exact: x J_0(x) = 2 J_1(x))", "",
              "| order | h | free DOFs | max rel. err. | max elastic residual |", "|---|---|---|---|---|"]
    for r in rows:
        if r["case"] != "thin_disc_free_edge":
            continue
        lines.append(f"| {r['order']} | {r['mesh_size']} | {r['free_dofs']} | "
                     f"{max(r['rel_err']):.3e} | {r['max_elastic_residual']:.2e} |")
    lines += ["", "## 4. Long-tube mobility vs the T(0,1) infinite waveguide", "",
              "Band mean of Re[Y] over 241 frequencies against the infinite-waveguide asymptote of",
              "the load. Overlap: modal overlap eta f/(2 df) of the participating (even) modes, the",
              "centred load driving only those. Asymptote of the *applied* load: |P(k)|^2 from the FE",
              "participations, Re Y(k_n) = c_n^2 L/(4 c_T), interpolated in k.",
              "", "### 4a. Rect ring loads", "",
              "The nominal asymptote is W^2 Y_inf <kappa^2>, kappa = sinc(k W_z/2). The applied load is",
              "the P2 interpolant of the nodal step: its torque differs from the nominal by +-h_e/(3 W_z)",
              "depending on whether the edge nodes pass |z| <= W_z/2 by rounding (here they do), which",
              "explains a +15 % excess over the nominal asymptote.", "",
              "| ring width | eta | band [Hz] | overlap | applied torque/nominal | mean/nominal asymptote "
              "| applied/nominal asymptote | **mean/applied asymptote** |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["case"] != "ring_load_mobility":
            continue
        lines.append(f"| {r['ring_width_m']} | {r['eta']} | "
                     f"{r['band_Hz'][0]:.0f}-{r['band_Hz'][1]:.0f} | "
                     f"{r['modal_overlap_participating']:.2f} | {r['torque_applied_over_nominal']:.4f} | "
                     f"{r['ratio_mean_over_kappa2']:.3f} | {r['applied_over_nominal_asymptote']:.3f} | "
                     f"**{r['ratio_mean_over_applied']:.3f}** |")
    lines += ["", "### 4b. Gaussian ring load (closed-form transform): the validation", "",
              "g = exp(-z^2/2 sigma_z^2)/t on the wall, sigma_z = 0.1 m, |P(k)|^2 = 2 pi sigma_z^2",
              "exp(-k^2 sigma_z^2), thick-wall T(0,1) formula. Lossless: the eta -> 0 asymptote. Damped:",
              "the damped infinite waveguide Re Y_d at the row's eta, the limit of the band mean at finite",
              "eta; it differs from the lossless value at first order in eta.", "",
              "| sigma_z | eta | band [Hz] | overlap | applied torque/closed form | mean/lossless "
              "| damped/lossless | **mean/damped** | min | max |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["case"] != "gauss_load_mobility":
            continue
        lines.append(f"| {r['sigma_z_m']} | {r['eta']} | "
                     f"{r['band_Hz'][0]:.0f}-{r['band_Hz'][1]:.0f} | "
                     f"{r['modal_overlap_participating']:.2f} | {r['torque_applied_over_nominal']:.6f} | "
                     f"{r['ratio_mean_over_closed_form']:.3f} | {r['damped_over_lossless']:.4f} | "
                     f"**{r['ratio_mean_over_damped']:.4f}** | {r['ratio_min']:.3g} | {r['ratio_max']:.3g} |")
    surf = [r for r in rows if r["case"] == "surface_load_cutoff"]
    if surf:
        s0 = surf[0]
        lines += ["", "## 5. Validity range: surface loads above the thickness-shear cutoff", "",
                  f"Tube a = {s0['a_m']} m, t = {s0['t_m']} m, L = {s0['L_m']} m, eta = {s0['eta']}, "
                  f"Gaussian in z (sigma_z = {s0['sigma_z_m']} m). T(0,2) cutoff "
                  f"{s0['cutoff_T02_Hz'] / 1e3:.2f} kHz (c_T/2t = {s0['c_T_over_2t_Hz'] / 1e3:.2f} kHz).",
                  "Re[Y] over the T(0,1) asymptote of the same radial profile. Below about 30 kHz both",
                  "rows follow the modes (overlap < 1). The asymptote holds for the load uniform through",
                  "the wall; for a surface traction above the cutoff it is several times too low.", "",
                  "| profile | " + " | ".join(f"{f / 1e3:.0f} kHz" for f in s0["f_Hz"]) + " |",
                  "|---|" + "---|" * len(s0["f_Hz"])]
        for r in surf:
            lines.append(f"| {r['profile']} | " + " | ".join(f"{x:.2f}" for x in r["ratio_to_T01"]) + " |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", REC / "torsion.json", "and", OUT)


if __name__ == "__main__":
    main()
