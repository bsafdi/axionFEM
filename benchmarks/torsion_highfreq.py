"""Validation of the high-frequency torsional mobility: the multi-branch waveguide (A) and the
wall-resolved FEM harmonic path (B).  docs/validation/torsion_highfreq.md.

    python benchmarks/torsion_highfreq.py SECTION [workdir]      SECTION: A0 B1 perf X demo report

References
----------
* A: `axionfem.torsion_waveguide.Tube.mobility`, the damped infinite tube summed over every torsional
  branch T(0, n) (exact Bessel radial shapes, complex axial wavenumbers).
* F: `Tube.finite_tube_mobility`, the exact free-free finite tube (separable modes
  psi_n(r) cos(m pi (z + L/2)/L)), the same structure the FEM solves: B must equal F up to
  discretisation; F - A is the finite-length / modal-overlap scatter.
* B: `Results.torsion(loads=..., band=...)`, loads applied exactly at quadrature points.

Sections
--------
auto  the measured auto choice (backend, concurrency, memory) on a thin wall and a compact cross-section.
A0  A and F against the nodal-load FEM path on a thick tube (a = 0.5,
    t = 0.04, L = 0.6 m, eta = 0.1): uniform and surface-concentrated volume loads, 5-130 kHz.
B1  mesh convergence of B against F (band means and pointwise) vs elements per shear wavelength for
    P1-P3: the thresholds of the mesh-resolution guard.
perf  cost per frequency vs DOF (Eigen SparseLU vs STRUMPACK, frequencies in parallel).
X   cross-validation A vs B vs F on uniform tubes: several a, t, eta; volume and surface loads;
    bands below and above several cutoffs, up to 1 MHz for a 1 cm wall.
demo  B on a non-uniform structure (tube with a flange) with convergence, against A of the bare tube.
report  writes docs/validation/torsion_highfreq.md from the records.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import axionfem as af
from axionfem.torsion_waveguide import Gaussian, Tube, TubeLoad

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records"
OUT = ROOT / "docs" / "validation" / "torsion_highfreq.md"
E_S, NU_S, RHO_S = 200e9, 0.3, 7900.0
MU_S = E_S / (2 * (1 + NU_S))
CT_S = math.sqrt(MU_S / RHO_S)


def _stamp():
    out = {"date": time.strftime("%Y-%m-%d %H:%M")}
    try:
        out["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                       text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        out["worktree_clean"] = dirty == ""
    except (OSError, subprocess.CalledProcessError):
        out["commit"] = None
    out["build"] = af.build_info()
    out["build"].get("library", {}).pop("build_dir", None)   # an absolute path: not written
    return out


def _save(name, rows, extra=None):
    REC.mkdir(parents=True, exist_ok=True)
    rec = {"stamp": _stamp(), "material": {"E_Pa": E_S, "nu": NU_S, "rho": RHO_S, "c_T_m_s": CT_S},
           "rows": rows}
    rec.update(extra or {})
    (REC / f"torsion_highfreq_{name}.json").write_text(json.dumps(rec, indent=1))
    print("wrote", REC / f"torsion_highfreq_{name}.json", flush=True)


def _mat():
    return {"steel": {"kind": "linear", "density": RHO_S,
                      "mechanical": {"E_Pa": E_S, "nu": NU_S, "alpha": 0.0}}}


def tube_design(a, t, L, h, order, flange=None, name="tube"):
    """Free-free tube r in [a, a + t], z in [-L/2, L/2], meshed with size h; a vacuum `bore` region
    r in [0, a] (its faces with the tube are the inner wall) and a thin vacuum `skin` outside (the
    outer wall).  flange = (r_out, thickness): an annular flange at the z = +L/2 end."""
    regions = [{"name": "tube", "shape": {"rect": {"r": [a, a + t], "z": [-L / 2, L / 2]}},
                "material": "steel", "mesh_size": h}]
    if a > 0:
        regions.append({"name": "bore", "shape": {"rect": {"r": [0.0, a], "z": [-L / 2, L / 2]}},
                        "material": "vacuum", "mesh_size": max(h * 8, a / 4)})
    regions.append({"name": "skin", "shape": {"rect": {"r": [a + t, a + t + 0.2 * t], "z": [-L / 2, L / 2]}},
                    "material": "vacuum", "mesh_size": max(h * 2, 0.2 * t)})
    if flange:
        ro, tf = flange
        regions.append({"name": "flange", "shape": {"rect": {"r": [a + 1.2 * t, ro], "z": [L / 2 - tf, L / 2]}},
                        "material": "steel", "mesh_size": h})
        # the skin is cut where the flange joins the tube: the flange is painted after it
        regions.append({"name": "collar", "shape": {"rect": {"r": [a + t, a + 1.2 * t], "z": [L / 2 - tf, L / 2]}},
                        "material": "steel", "mesh_size": h})
    rc = 1.5 * max(a + t, (flange[0] if flange else 0.0)) + 0.2
    regions.append({"name": "coil", "shape": {"rect": {"r": [rc, rc + 0.01], "z": [-0.01, 0.01]}},
                    "material": "winding"})
    ext = max(L, a + t)
    return af.Design({"schema": "axionfem.design/1", "name": name, "materials": _mat(), "regions": regions,
                      "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
                      "mesh": {"order": order, "size": ext / 3, "domain_radius": 2.5 * ext + rc}})


def gauss_expr(sigma, z0=0.0):
    return f"exp(-((z - ({z0!r}))^2)/(2*{sigma!r}^2))"


def fem_loads(kind, sigma, z0=0.0, t=None):
    """B's `loads` for a Gaussian axial profile: inner/outer traction (1 Pa peak) or volume (1 N/m^3 peak)."""
    ex = gauss_expr(sigma, z0)
    if kind == "inner":
        return [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": ex}]
    if kind == "outer":
        return [{"surface": {"on": "tube", "from": "skin"}, "traction_Pa": ex}]
    if kind == "volume":
        return [{"region": "tube", "g_N_m3": ex}]
    if kind == "both":         # both walls, the case whose branch tail alternates with parity
        return fem_loads("inner", sigma, z0) + fem_loads("outer", sigma, z0)
    raise ValueError(kind)


def analytic_load(kind, sigma, z0=0.0):
    ax = Gaussian(sigma, z0)
    return {"inner": TubeLoad(ax, inner=1.0), "outer": TubeLoad(ax, outer=1.0),
            "volume": TubeLoad(ax, volume=1.0), "both": TubeLoad(ax, inner=1.0, outer=1.0)}[kind]


def trap_mean(f, y):
    return float(np.trapezoid(y, f) / (f[-1] - f[0]))


def solve_tube(wd, a, t, L, h, order, **kw):
    t0 = time.time()
    res = tube_design(a, t, L, h, order, **kw).solve(wd, verbose=0)
    return res, time.time() - t0


# ------------------------------------------------------------------------------------------------ A0
def section_A0(wd):
    a, t, L, eta, sz, delta = 0.5, 0.04, 0.6, 0.1, 0.005, 0.003
    bo = a + t
    doc = {"schema": "axionfem.design/1", "name": "short_tube", "materials": _mat(),
           "regions": [{"name": "tube", "shape": {"rect": {"r": [a, bo], "z": [-L / 2, L / 2]}},
                        "material": "steel", "mesh_size": 0.002},
                       {"name": "coil", "shape": {"rect": {"r": [1.5, 1.55], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mesh": {"order": 2, "size": 0.5, "domain_radius": 4.0}}
    res = af.Design(doc).solve(wd / "A0", verbose=0)
    tr0 = res.torsion(dofs=True)
    r = np.array(tr0["dof_r"])
    z = np.array(tr0["dof_z"])
    wall = (r >= a - 1e-9) & (r <= bo + 1e-9)
    prof = {"uniform": lambda x: np.ones_like(x) / t,
            "surface": lambda x: np.exp(-(x - a) / delta) / (delta * (1 - math.exp(-t / delta)))}
    tube = Tube(a, t, RHO_S, E=E_S, nu=NU_S)
    freqs = np.linspace(5e3, 130e3, 251)
    rows = []
    for name, w in prof.items():
        g = np.where(wall, w(r) * np.exp(-z ** 2 / (2 * sz * sz)), 0.0)
        hr = res.torsion(load=g, freqs_Hz=list(freqs), eta=eta)["harmonic"]["response"]
        Yb = np.array([x["Y_re"] for x in hr])
        ld = TubeLoad(Gaussian(sz), volume=w)
        A = tube.mobility(freqs, ld, eta)
        F = tube.finite_tube_mobility(freqs, ld, eta, L).real
        T01 = A.Y_branch[0].real
        for f0, f1 in [(5e3, 35e3), (40e3, 75e3), (80e3, 115e3), (40e3, 130e3)]:
            s = (freqs >= f0) & (freqs <= f1)
            rows.append(dict(profile=name, band_Hz=[f0, f1], fem_over_A=trap_mean(freqs[s], Yb[s]) /
                             trap_mean(freqs[s], A.Y.real[s]),
                             finite_over_A=trap_mean(freqs[s], F[s]) / trap_mean(freqs[s], A.Y.real[s]),
                             fem_over_finite=trap_mean(freqs[s], Yb[s]) / trap_mean(freqs[s], F[s]),
                             A_over_T01=trap_mean(freqs[s], A.Y.real[s]) / trap_mean(freqs[s], T01[s])))
        rows.append(dict(profile=name, pointwise_max_fem_over_finite_minus_1=float(np.max(np.abs(Yb / F - 1))),
                         n_branches=len(A.kappa), cutoffs_Hz=list(tube.cutoffs_Hz(4))))
        print(name, rows[-5:], flush=True)
    _save("A0", rows, {"case": dict(a=a, t=t, L=L, eta=eta, sigma_z=sz, efold=delta, order=2, h=0.002,
                                    free_dofs=tr0["diagnostics"]["free_dofs"])})


# ------------------------------------------------------------------------------------------------ B1
def section_B1(wd):
    """Band-mean and pointwise error of B against F vs elements per shear wavelength at f_max."""
    a, t, L, sz = 0.5, 0.01, 0.2, 0.002
    fmax = 1.0e6
    lam = CT_S / fmax
    tube = Tube(a, t, RHO_S, E=E_S, nu=NU_S)
    cases = {1: (6, 10, 16, 24, 32), 2: (2, 3, 4, 6, 8, 12), 3: (1.5, 2, 3, 4, 6)}
    bands = [(0.8e6, 1.0e6)]
    rows = []
    for eta in (0.01, 0.003):
        for kind in ("inner", "volume"):
            ld = analytic_load(kind, sz)
            for f0, f1 in bands:
                n = 401 if eta < 0.005 else 201
                freqs = np.linspace(f0, f1, n)
                F = tube.finite_tube_mobility(freqs, ld, eta, L).real
                F2 = tube.finite_tube_mobility(freqs, ld, eta, L,
                                               beta_max=2 * (4 * 2 * math.pi * f1 / CT_S + 40 / sz)).real
                for order, Ns in cases.items():
                    for N in Ns:
                        if eta < 0.005 and kind == "volume":
                            continue
                        h = lam / N
                        res, tsolve = solve_tube(wd / f"B1_p{order}_N{N}", a, t, L, h, order)
                        tr = res.torsion(loads=fem_loads(kind, sz), eta=eta,
                                         band={"f_Hz": [f0, f1], "n": n}, mesh_check="off",
                                         solver={"backend": "auto", "parallel_frequencies": 8})
                        hm = tr["harmonic"]
                        Yb = np.array([x["Y_re"] for x in hm["response"]])
                        row = dict(eta=eta, load=kind, band_Hz=[f0, f1], order=order, N_target=N, h_m=h,
                                   free_dofs=hm["timing"]["free_dofs"],
                                   N_min=hm["resolution"]["elements_per_wavelength_min"],
                                   N_median=hm["resolution"]["per_region"]["tube"]["median"],
                                   band_mean_B=hm["band"]["mean_Re_Y"], band_mean_F=trap_mean(freqs, F),
                                   band_mean_F_truncation=abs(trap_mean(freqs, F2) / trap_mean(freqs, F) - 1),
                                   band_err=hm["band"]["mean_Re_Y"] / trap_mean(freqs, F) - 1,
                                   pointwise_median_err=float(np.median(np.abs(Yb / F - 1))),
                                   pointwise_max_err=float(np.max(np.abs(Yb / F - 1))),
                                   applied_force_N=tr["loads"]["total"]["force_N"],
                                   seconds_per_frequency=hm["timing"]["effective_seconds_per_frequency"],
                                   backend=hm["timing"]["backend"], mesh_solve_s=tsolve)
                        rows.append(row)
                        print(json.dumps({k: row[k] for k in ("eta", "load", "order", "N_target", "N_min",
                                                               "free_dofs", "band_err", "pointwise_median_err",
                                                               "pointwise_max_err")}), flush=True)
                        _save("B1", rows)
    # the analytic applied force of the Gaussian: inner 2 pi a sqrt(2 pi) sz, volume 2 pi I1 sqrt(2 pi) sz
    _save("B1", rows, {"case": dict(a=a, t=t, L=L, sigma_z=sz, f_max=fmax, lambda_T_m=lam,
                                    force_inner_exact_N=2 * math.pi * a * math.sqrt(2 * math.pi) * sz,
                                    force_volume_exact_N=math.pi * ((a + t) ** 2 - a ** 2) *
                                    math.sqrt(2 * math.pi) * sz)})


# ------------------------------------------------------------------------------------------------ perf
def section_perf(wd, big=False):
    """Seconds per frequency vs free DOFs; Eigen SparseLU vs STRUMPACK; frequencies in parallel.

    big: the supplementary run at 0.3-1.1 M DOF with Eigen SparseLU in parallel (record perf_big)."""
    a, t, sz = 0.5, 0.01, 0.002
    lam = CT_S / 1e6
    h = lam / 6
    nthreads = af.get_threads() if hasattr(af, "get_threads") else None
    rows = []
    name = "perf_big" if big else "perf"
    for L in ((1.5, 4.0, 8.0) if big else (0.05, 0.15, 0.5, 1.5, 4.0, 8.0)):
        res, tsolve = solve_tube(wd / f"perf_L{L}", a, t, L, h, 2)
        freqs = list(np.linspace(0.9e6, 1.0e6, 16 if big else 8))
        configs = [("strumpack", 1), ("strumpack", 4), ("strumpack", 8)]
        nfree = res.torsion(dofs=False, nmodes=0)["diagnostics"]["free_dofs"]
        if nfree <= 4e5:
            configs = [("eigen", 1), ("eigen", 8)] + configs
        if big:
            configs = [("eigen", 8), ("eigen", 16), ("strumpack", 2)]
        for be, p in configs:
            t0 = time.time()
            tr = res.torsion(loads=fem_loads("inner", sz), eta=0.01, freqs_Hz=freqs, mesh_check="off",
                             solver={"backend": be, "parallel_frequencies": p})
            wall = time.time() - t0
            tm = tr["harmonic"]["timing"]
            Y = [x["Y_re"] for x in tr["harmonic"]["response"]]
            row = dict(L=L, free_dofs=tm["free_dofs"], matrix_nonzeros=tm["matrix_nonzeros"],
                       factor_nonzeros=tm["factor_nonzeros"], backend=tm["backend"],
                       parallel_frequencies=tm["parallel_frequencies"], parallel_frequencies_requested=p,
                       threads_per_frequency=tm["threads_per_frequency"], n_freqs=len(freqs),
                       mean_seconds_per_frequency=tm["mean_seconds_per_frequency"],
                       effective_seconds_per_frequency=tm["effective_seconds_per_frequency"],
                       call_wall_s=wall, setup_s=tm["setup_s"], Y_re=Y, threads=nthreads, mesh_solve_s=tsolve)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("free_dofs", "backend", "parallel_frequencies",
                                                   "threads_per_frequency", "mean_seconds_per_frequency",
                                                   "effective_seconds_per_frequency", "call_wall_s",
                                                   "factor_nonzeros")}), flush=True)
            _save(name, rows)
    _save(name, rows)


# ------------------------------------------------------------------------------------------------ X
def section_X(wd):
    """A vs B vs F on uniform tubes, band means; B at the guard's 1 % resolution or finer."""
    tubes = [dict(a=0.5, t=0.01, L=0.5), dict(a=0.1, t=0.01, L=0.5), dict(a=1.0, t=0.02, L=0.5)]
    sz = 0.003
    rows = []
    for tb in tubes:
        a, t, L = tb["a"], tb["t"], tb["L"]
        tube = Tube(a, t, RHO_S, E=E_S, nu=NU_S)
        fc = tube.cutoffs_Hz(8)
        fmax = 1.0e6 * 0.01 / t
        bands = [(0.3 * fc[1], 0.9 * fc[1]), (1.05 * fc[1], 0.95 * fc[2]), (1.05 * fc[2], 0.95 * fc[4]),
                 (0.6 * fmax, fmax)]
        h = CT_S / fmax / 8          # P2 at 8 elements per wavelength at f_max (guard: >= 6)
        res, tsolve = solve_tube(wd / f"X_a{a}_t{t}", a, t, L, h, 2)
        for eta in (0.01, 0.03, 0.1):
            for kind in ("volume", "inner", "outer", "both"):
                if kind == "outer" and eta != 0.03:
                    continue
                if kind == "both" and eta == 0.01:
                    continue
                ld = analytic_load(kind, sz)
                for f0, f1 in bands:
                    n = int(2 * math.ceil((f1 - f0) / (0.25 * eta * f0) / 2)) + 1
                    n = max(n, 101)
                    freqs = np.linspace(f0, f1, n)
                    A = tube.mobility(freqs, ld, eta)
                    F = tube.finite_tube_mobility(freqs, ld, eta, L).real
                    tr = res.torsion(loads=fem_loads(kind, sz), eta=eta,
                                     band={"f_Hz": [f0, f1], "n": n, "length_m": L},
                                     solver={"backend": "eigen", "parallel_frequencies": 16})
                    hm = tr["harmonic"]
                    Yb = np.array([x["Y_re"] for x in hm["response"]])
                    mA, mF, mB = trap_mean(freqs, A.Y.real), trap_mean(freqs, F), hm["band"]["mean_Re_Y"]
                    # participating modal overlap: sum over propagating branches of eta f L/v_g (even modes)
                    fm = 0.5 * (f0 + f1)
                    kz2 = (2 * math.pi * fm / CT_S) ** 2 - tube.kappa(kappa_max=2 * math.pi * fm / CT_S) ** 2
                    vg = CT_S ** 2 * np.sqrt(np.maximum(kz2, 1e-30)) / (2 * math.pi * fm)
                    overlap = float(eta * fm * np.sum(L / vg))
                    T01 = trap_mean(freqs, A.Y_branch[0].real)
                    row = dict(a=a, t=t, L=L, eta=eta, load=kind, band_Hz=[f0, f1], n=n,
                               n_propagating=int(np.sum(fc < fm)), overlap=overlap,
                               free_dofs=hm["timing"]["free_dofs"],
                               N_min=hm["resolution"]["elements_per_wavelength_min"],
                               mean_A=mA, mean_F=mF, mean_B=mB, B_over_A=mB / mA, F_over_A=mF / mA, B_over_F=mB / mF,
                               B_quadrature_unc=hm["band"]["quadrature_uncertainty"] / mB,
                               B_ripple_sem=hm["band"]["ripple_sem"] / mB,
                               B_edge_estimate=hm["band"]["edge_estimate"] / mB,
                               A_tail_share=float(np.max(np.abs(A.tail.real) / np.abs(A.Y.real))),
                               A_over_T01=mA / T01, pointwise_max_B_over_F=float(np.max(np.abs(Yb / F - 1))),
                               A_truncation=float(np.max(A.truncation)), n_branches_A=len(A.kappa))
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("a", "t", "eta", "load", "band_Hz", "n_propagating",
                                                           "overlap", "B_over_A", "F_over_A", "B_over_F",
                                                           "B_ripple_sem", "B_edge_estimate", "A_tail_share")}),
                          flush=True)
                    _save("X", rows)
    _save("X", rows, {"case": dict(sigma_z=sz, tubes=tubes, resolution="P2, 8 elements per wavelength at f_max")})


# ------------------------------------------------------------------------------------------------ demo
def section_demo(wd):
    """A tube with an annular flange at one end, loaded on the inner wall near the junction."""
    a, t, L, sz = 0.5, 0.01, 0.5, 0.003
    flange = (0.6, 0.01)
    z0 = L / 2 - 0.02
    tube = Tube(a, t, RHO_S, E=E_S, nu=NU_S)
    fc = tube.cutoffs_Hz(6)
    bands = [(0.3 * fc[1], 0.9 * fc[1]), (1.05 * fc[1], 0.95 * fc[2]), (0.6e6, 1.0e6)]
    rows = []
    eta = 0.03
    lam = CT_S / 1e6
    # the bare finite tube with the same load position: F (exact) and B (P2, N = 8)
    loads_bare = [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": gauss_expr(sz, z0)}]
    res_bare, _ = solve_tube(wd / "demo_bare_p2_N8", a, t, L, lam / 8, 2, name="tube_bare")
    bare = {}
    for f0, f1 in bands:
        n = max(int(2 * math.ceil((f1 - f0) / (0.25 * eta * f0) / 2)) + 1, 101)
        freqs = np.linspace(f0, f1, n)
        Fb = tube.finite_tube_mobility(freqs, TubeLoad(Gaussian(sz, z0), inner=1.0), eta, L).real
        hb = res_bare.torsion(loads=loads_bare, eta=eta, band={"f_Hz": [f0, f1], "n": n},
                              solver={"backend": "eigen", "parallel_frequencies": 16})["harmonic"]
        bare[(f0, f1)] = dict(F=trap_mean(freqs, Fb), B=hb["band"]["mean_Re_Y"])
    for order, N in ((2, 4), (2, 6), (2, 8), (2, 12), (3, 6)):
        res, tsolve = solve_tube(wd / f"demo_p{order}_N{N}", a, t, L, lam / N, order, flange=flange,
                                 name="tube_flange")
        loads = [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": gauss_expr(sz, z0)}]
        for f0, f1 in bands:
            n = int(2 * math.ceil((f1 - f0) / (0.25 * eta * f0) / 2)) + 1
            n = max(n, 101)
            freqs = np.linspace(f0, f1, n)
            tr = res.torsion(loads=loads, eta=eta, band={"f_Hz": [f0, f1], "n": n}, mesh_check="warn",
                             solver={"backend": "eigen", "parallel_frequencies": 16})
            hm = tr["harmonic"]
            A = tube.mobility(freqs, TubeLoad(Gaussian(sz, z0), inner=1.0), eta)
            row = dict(order=order, N_target=N, free_dofs=hm["timing"]["free_dofs"],
                       N_min=hm["resolution"]["elements_per_wavelength_min"], band_Hz=[f0, f1], n=n, eta=eta,
                       mean_B=hm["band"]["mean_Re_Y"], quadrature_unc=hm["band"]["quadrature_uncertainty"],
                       ripple_sem=hm["band"]["ripple_sem"], mean_A_bare_tube=trap_mean(freqs, A.Y.real),
                       mean_F_bare_tube=bare[(f0, f1)]["F"], mean_B_bare_tube=bare[(f0, f1)]["B"],
                       Y_re=[x["Y_re"] for x in hm["response"]], f_Hz=list(freqs),
                       seconds_per_frequency=hm["timing"]["effective_seconds_per_frequency"],
                       warnings=tr["warnings"])
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("order", "N_target", "free_dofs", "band_Hz", "mean_B",
                                                   "mean_A_bare_tube", "mean_F_bare_tube", "mean_B_bare_tube")}),
                  flush=True)
            _save("demo", rows)
    _save("demo", rows, {"case": dict(a=a, t=t, L=L, flange_r_out=flange[0], flange_t=flange[1],
                                      sigma_z=sz, z0=z0, eta=eta)})


# ------------------------------------------------------------------------------------------------ auto
def _auto_design(kind, h):
    if kind == "compact":
        reg = [{"name": "block", "shape": {"rect": {"r": [0.2, 0.3], "z": [-0.05, 0.05]}}, "material": "steel",
                "mesh_size": h}]
    else:
        reg = [{"name": "block", "shape": {"rect": {"r": [0.5, 0.51], "z": [-1.0, 1.0]}}, "material": "steel",
                "mesh_size": h}]
    reg.append({"name": "coil", "shape": {"rect": {"r": [1.0, 1.01], "z": [-0.01, 0.01]}}, "material": "winding"})
    return af.Design({"schema": "axionfem.design/1", "name": kind, "materials": _mat(), "regions": reg,
                      "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
                      "mesh": {"order": 2, "size": 0.3, "domain_radius": 5.0}})


def _hwm_gb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1]) / 1048576
    return float("nan")


def section_auto_one(wd, kind, h):
    """One geometry in its own process (peak RSS per case): auto's probes, choice and cost."""
    res = _auto_design(kind, h).solve(wd / f"auto_{kind}_{h}", verbose=0)
    freqs = list(np.linspace(1.9e5, 2.1e5, 16))
    t0 = time.time()
    tr = res.torsion(loads=[{"region": "block", "g_N_m3": 1.0}], eta=0.01, freqs_Hz=freqs, mesh_check="off",
                     solver={"backend": "auto", "parallel_frequencies": "auto"})
    tm = tr["harmonic"]["timing"]
    print("AUTO_ROW " + json.dumps(dict(geometry=kind, h=h, free_dofs=tm["free_dofs"], choice=tm["parallel_choice"],
                                        effective_seconds_per_frequency=tm["effective_seconds_per_frequency"],
                                        call_wall_s=time.time() - t0, peak_rss_GB=_hwm_gb(),
                                        Y_re=[x["Y_re"] for x in tr["harmonic"]["response"]])), flush=True)


def section_auto(wd):
    import os
    rows = []
    for kind, h in (("thin", 0.00052), ("compact", 0.00037)):
        p = subprocess.run([sys.executable, __file__, "auto_one", str(wd), kind, str(h)], capture_output=True,
                           text=True)
        print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("AUTO_ROW ")]
        if not line:
            raise RuntimeError(f"auto_one {kind} failed (exit {p.returncode})")
        rows.append(json.loads(line[0][len("AUTO_ROW "):]))
        _save("auto", rows)
    mem = os.environ.get("SLURM_MEM_PER_NODE") or (os.environ.get("SLURM_MEM_PER_CPU", "?") + " MB per CPU")
    _save("auto", rows, {"case": dict(job_mem=f"{mem} MB" if mem.isdigit() else mem,
                                      threads=os.environ.get("SLURM_CPUS_PER_TASK"))})


# ------------------------------------------------------------------------------------------------ report
def _load(name):
    p = REC / f"torsion_highfreq_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def section_report(_wd):
    L = ["# High-frequency torsional mobility: multi-branch waveguide (A) and wall-resolved FEM (B)", "",
         '!!! warning "Beta"',
         "    `axionfem.torsion_waveguide` and the high-frequency torsional mobility (`Results.torsion_band`) are beta.",
         "",
         "Generated by `benchmarks/torsion_highfreq.py report` from `benchmarks/records/torsion_highfreq_*.json`.",
         "Physics: [torsion_highfreq.md](../physics/torsion_highfreq.md). A = damped infinite tube, all",
         "branches T(0, n); F = exact finite free-free tube (separable modal sum), the structure the FEM solves;",
         "B = FEM with exact quadrature-point loads. Band means are trapezoidal means of Re Y over the band.",
         f"Steel: E = {E_S:.3g} Pa, nu = {NU_S}, rho = {RHO_S} kg/m^3, c_T = {CT_S:.5g} m/s.", ""]
    r = _load("A0")
    if r:
        c = r["case"]
        L += ["## A0. A and F against the nodal-load FEM path (a thick tube)", "",
              f"a = {c['a']} m, t = {c['t']} m, L = {c['L']} m, eta = {c['eta']}, P2, h = {c['h']} m "
              f"({c['free_dofs']} free DOFs), Gaussian in z (sigma = {c['sigma_z']} m), nodal loads: uniform",
              f"through the wall, or concentrated at the inner surface (e-fold {c['efold']} m).", "",
              "| profile | band [kHz] | FEM/A | F/A | FEM/F | A/T(0,1) |", "|---|---|---|---|---|---|"]
        for x in r["rows"]:
            if "band_Hz" in x:
                L.append(f"| {x['profile']} | {x['band_Hz'][0] / 1e3:.0f}-{x['band_Hz'][1] / 1e3:.0f} | "
                         f"{x['fem_over_A']:.3f} | {x['finite_over_A']:.3f} | {x['fem_over_finite']:.4f} | "
                         f"{x['A_over_T01']:.2f} |")
        for x in r["rows"]:
            if "pointwise_max_fem_over_finite_minus_1" in x:
                L.append(f"\n{x['profile']}: max pointwise |FEM/F - 1| = "
                         f"{x['pointwise_max_fem_over_finite_minus_1']:.1e}; A used {x['n_branches']} branches.")
        L.append("")
    r = _load("B1")
    if r:
        c = r["case"]
        L += ["## B1. Mesh convergence of B against F: the resolution guard", "",
              f"a = {c['a']} m, t = {c['t']} m, L = {c['L']} m, Gaussian sigma_z = {c['sigma_z']} m, band "
              f"0.8-1.0 MHz (lambda_T = {c['lambda_T_m'] * 1e3:.2f} mm at 1 MHz). N_min = elements per shear",
              "wavelength at f_max, as the guard measures it (h = longest edge). Errors relative to F.", "",
              "| eta | load | order | N target | N_min | free DOFs | band-mean err | pointwise median | "
              "pointwise max | s/freq |", "|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['eta']} | {x['load']} | P{x['order']} | {x['N_target']} | {x['N_min']:.2f} | "
                     f"{x['free_dofs']} | {x['band_err']:+.2e} | {x['pointwise_median_err']:.1e} | "
                     f"{x['pointwise_max_err']:.1e} | {x['seconds_per_frequency']:.2f} |")
        tr = max(x["band_mean_F_truncation"] for x in r["rows"])
        L += ["", f"F's truncation (beta_max doubled): <= {tr:.1e} in band mean.", ""]
    r = _load("perf")
    rb = _load("perf_big")
    if r and rb:
        r["rows"] = r["rows"] + rb["rows"]
    if r:
        thr = r["stamp"]["build"].get("threads")
        L += ["## Cost per frequency vs DOF", "",
              "P2 tube (a = 0.5, t = 0.01, 6 elements per wavelength at 1 MHz: a thin wall), 8 (16) frequencies near",
              f"1 MHz, one node ({thr} threads). s/freq (effective) = wall time of the sweep / number of",
              "frequencies. A compact cross-section behaves differently: section auto below.", "",
              "| free DOFs | factor nnz | backend | parallel freqs | threads/freq | s/freq (each) | "
              "s/freq (effective) |", "|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['free_dofs']} | {x['factor_nonzeros']:.3g} | {x['backend']} | "
                     f"{x['parallel_frequencies']} | {x['threads_per_frequency']} | "
                     f"{x['mean_seconds_per_frequency']:.3g} | {x['effective_seconds_per_frequency']:.3g} |")
        L.append("")
    r = _load("auto")
    if r:
        L += ["## auto: the measured choice of backend and concurrency (opt-in)", "",
              f"`solver.parallel_frequencies = \"auto\"` on a thin wall and on a compact cross-section, in a job with "
              f"{r['case']['job_mem']} and {r['case']['threads']} threads. The default of `Results.torsion_band` is "
              "serial STRUMPACK. auto reads the job's memory first, solves one frequency with STRUMPACK (its "
              "numerical factorization timed apart from the analysis), probes Eigen SparseLU only where it can pay "
              "off and fits, and picks the faster. Peak RSS from the process's high-water mark.", "",
              "| geometry | free DOFs | memory available [GB] (source) | STRUMPACK factor [s] | STRUMPACK fill/DOF | "
              "Eigen probe | Eigen fill/DOF | Eigen kB/DOF/solve | chosen | p | s/freq (effective) | peak RSS [GB] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            c = x["choice"]
            pe = c["probe_eigen_s"]
            probe = f"{pe:.3g} s" if pe is not None else "skipped: " + c["eigen_probe"]["reason"]
            fe = c["probe_eigen_factor_nnz_per_dof"]
            mem = c["memory"]
            L.append(f"| {x['geometry']} | {x['free_dofs']} | {mem['available_bytes'] / 1e9:.1f} ({mem['source']}) | "
                     f"{c['probe_strumpack_factor_s']:.3g} | {c['probe_strumpack_factor_nnz_per_dof']:.0f} | {probe} | "
                     f"{'-' if fe is None else f'{fe:.0f}'} | "
                     f"{c['eigen_memory_per_solve_bytes'] / x['free_dofs'] / 1e3:.2f} | {c['chosen_backend']} | "
                     f"{c['chosen_p']} | {x['effective_seconds_per_frequency']:.3g} | {x['peak_rss_GB']:.2f} |")
        L.append("")
    r = _load("X")
    if r:
        L += ["## X. Cross-validation A vs B vs F on uniform tubes", "",
              f"L = 0.5 m, Gaussian sigma_z = {r['case']['sigma_z']} m centred, {r['case']['resolution']}. "
              "overlap = participating modal overlap sum_n eta f L/v_g,n (even modes); n_prop = propagating",
              "branches at the band centre. Band diagnostics of B (relative to its mean): quad = quadrature",
              "uncertainty, ripple = ripple_sem, edge = edge_estimate (band.length_m = L).", "",
              "| a | t | eta | load | band [kHz] | n_prop | overlap | B/A | F/A | B/F | quad / ripple / edge | "
              "A/T(0,1) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['a']} | {x['t']} | {x['eta']} | {x['load']} | "
                     f"{x['band_Hz'][0] / 1e3:.0f}-{x['band_Hz'][1] / 1e3:.0f} | {x['n_propagating']} | "
                     f"{x['overlap']:.2g} | {x['B_over_A']:.3f} | {x['F_over_A']:.3f} | {x['B_over_F']:.4f} | "
                     f"{x['B_quadrature_unc']:.1e} / {x['B_ripple_sem']:.1e} / {x['B_edge_estimate']:.1e} | "
                     f"{x['A_over_T01']:.3g} |")
        rows = r["rows"]
        hi = [x for x in rows if x["overlap"] >= 1]
        vol = [x for x in rows if x["load"] == "volume"]
        srf = [x for x in rows if x["load"] != "volume"]
        mx = lambda xs, k: max(abs(x[k] - 1) for x in xs)  # noqa: E731
        big = [x for x in rows if abs(x["F_over_A"] - 1) > 1e-2]

        def cover(k):
            ok = [x[k] >= abs(x["F_over_A"] - 1) for x in rows]
            okb = [x[k] >= abs(x["F_over_A"] - 1) for x in big]
            ratio = np.median([x[k] / max(abs(x["F_over_A"] - 1), 1e-12) for x in rows])
            return f"{sum(ok)} of {len(rows)} rows ({sum(okb)} of the {len(big)} with |F/A - 1| > 1 %), median ratio {ratio:.0f}"
        L += ["", "Summary (computed from the rows above):",
              f"- B vs F (discretisation): max |B/F - 1| = {mx(vol, 'B_over_F'):.1e} (volume loads), "
              f"{mx(srf, 'B_over_F'):.1e} (surface tractions);",
              f"- F vs A (finite length): max |F/A - 1| = {mx(rows, 'F_over_A'):.1e} over all rows, "
              f"{mx(hi, 'F_over_A'):.1e} at participating overlap >= 1;",
              f"- B vs A: max |B/A - 1| = {mx(rows, 'B_over_A'):.1e} over all rows, {mx(hi, 'B_over_A'):.1e} "
              "at overlap >= 1;",
              f"- edge_estimate >= |F/A - 1| in {cover('B_edge_estimate')};",
              f"- ripple_sem >= |F/A - 1| in {cover('B_ripple_sem')}: a ripple diagnostic, neither a bound nor an "
              "estimate of the finite-length scatter;",
              f"- A's analytic evanescent tail is at most {max(x['A_tail_share'] for x in rows):.1e} of Re Y "
              f"(the last branch kept: at most {max(x['A_truncation'] for x in rows):.1e}).", ""]
    r = _load("demo")
    if r:
        c = r["case"]
        L += ["## Non-uniform structure: tube with a flange", "",
              f"Tube a = {c['a']}, t = {c['t']}, L = {c['L']} m with an annular steel flange (r to {c['flange_r_out']} m,",
              f"{c['flange_t']} m thick) at z = +L/2; inner-wall Gaussian traction (sigma_z = {c['sigma_z']} m) at "
              f"z0 = {c['z0']} m, eta = {c['eta']}. The flange's own effect is B(flange) against the bare finite tube "
              "with the same load position (F exact; B at P2, N = 8). A (infinite tube) is for scale only.",
              "", "| order | N target | N_min | free DOFs | band [kHz] | mean Re Y (B) | B quad / ripple | "
              "B / F bare | B / B bare | B / A |", "|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| P{x['order']} | {x['N_target']} | {x['N_min']:.2f} | {x['free_dofs']} | "
                     f"{x['band_Hz'][0] / 1e3:.0f}-{x['band_Hz'][1] / 1e3:.0f} | {x['mean_B']:.4e} | "
                     f"{x['quadrature_unc'] / x['mean_B']:.1e} / {x['ripple_sem'] / x['mean_B']:.1e} | "
                     f"{x['mean_B'] / x['mean_F_bare_tube']:.4f} | {x['mean_B'] / x['mean_B_bare_tube']:.4f} | "
                     f"{x['mean_B'] / x['mean_A_bare_tube']:.3f} |")
        L += ["", "Bare tube with the same load position: F/A and B/F per band:", ""]
        seen = set()
        for x in r["rows"]:
            b = tuple(x["band_Hz"])
            if b in seen:
                continue
            seen.add(b)
            L.append(f"- {b[0] / 1e3:.0f}-{b[1] / 1e3:.0f} kHz: F/A = {x['mean_F_bare_tube'] / x['mean_A_bare_tube']:.4f}, "
                     f"B/F = {x['mean_B_bare_tube'] / x['mean_F_bare_tube']:.6f}")
        ref = {tuple(x["band_Hz"]): x["mean_B"] for x in r["rows"] if x["order"] == 3}
        L += ["", "Convergence: band mean relative to the P3 mesh (the finest), per band (N_min below the P2 warn "
              "threshold 3.5 marked *):", ""]
        for x in r["rows"]:
            b = tuple(x["band_Hz"])
            if b in ref and x["order"] == 2:
                star = " *" if x["N_min"] < 3.5 else ""
                L.append(f"- P2, N = {x['N_target']}, {b[0] / 1e3:.0f}-{b[1] / 1e3:.0f} kHz (N_min {x['N_min']:.2f}): "
                         f"{x['mean_B'] / ref[b] - 1:+.1e}{star}")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


def main():
    sec = sys.argv[1]
    wd = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "runs" / "torsion_highfreq")
    wd.mkdir(parents=True, exist_ok=True)
    if sec == "auto_one":
        section_auto_one(wd, sys.argv[3], float(sys.argv[4]))
        return
    sections = {"A0": section_A0, "B1": section_B1, "perf": section_perf,
                "perf_big": lambda w: section_perf(w, True), "X": section_X, "demo": section_demo,
                "auto": section_auto, "report": section_report}
    sections[sec](wd)


if __name__ == "__main__":
    main()
