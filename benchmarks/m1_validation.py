"""Validation of the axisymmetric engine against independent results.

Each check writes a record to benchmarks/records/<name>.json with the inputs,
the computed numbers, the reference, the deviation and pass/fail.

    python benchmarks/m1_validation.py all            # every check
    python benchmarks/m1_validation.py solenoid_field coax_inductance ...

References
----------
- thick finite solenoid on-axis field: closed form (elementary).
- coax inductance: closed form for the cavity plus the linear-profile port.
- toroid hole inductance L_out H/[pi (yR)^2] at H/R = 10, 5, 2, 1 for y = e^-1/2,
  no outer shield: 0.92, 0.85, 0.71, 0.55, quoted to two decimals (values computed with an independent code that is
  not distributed with axionFEM; they are not published).
- toroid/CORE duality at finite H.
- long-limit geometric factors (g_sol = (1-x^2)^2/(4 ln 1/x), g_tor = y^2 ln(1/y)),
  tested against exact references for the finite wall/winding geometry.
- finite current-sheet solenoid: the derivative-free (Nelder-Mead) optimum of the cost metric at the cost-optimal size
  over H/R and x, for the reference readout benchmark REF_BENCH below; the reference of benchmarks/adjoint_validation.py
  and global_opt_validation.py; checked for an interior optimum and the closed-form magnet/cryostat ratio
  p_area/p_energy of the cost-optimal size.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import axionfem as af  # noqa: E402

MU0 = 1.25663706212e-6
REC = ROOT / "benchmarks" / "records"
# Readout benchmark of the optimizer validations (here, in adjoint_validation.py and in global_opt_validation.py):
# round values (an SQL-limited amplifier, eta_A = 1), chosen only to give the scan requirement a definite value. The
# cost-optimal shape (H/R, x) does not depend on the benchmark; the absolute size and the cost metric do.
REF_BENCH = {"name": "reference", "Q": 1e6, "T": 0.1, "eta_A": 1.0}
RUNS = ROOT / "runs" / "validation"


def record(name: str, data: dict) -> dict:
    REC.mkdir(parents=True, exist_ok=True)
    data = {"check": name, "axionfem_version": af.version(), "date": time.strftime("%Y-%m-%d %H:%M"), **data}
    (REC / f"{name}.json").write_text(json.dumps(data, indent=2))
    status = "PASS" if data.get("pass") else "FAIL"
    print(f"[{status}] {name}: {data.get('headline', '')}", flush=True)
    return data


def design(name: str, params: dict, regions: list, **sections) -> af.Design:
    doc = {"schema": "axionfem.design/1", "name": name, "parameters": params, "regions": regions}
    doc.update(sections)
    return af.Design(doc)


# ---------------------------------------------------------------- 1. finite thick solenoid field
def check_solenoid_field(order: int = 3) -> dict:
    R1, R2, L, NI = 1.0, 1.1, 2.0, 1.0e6
    J = NI / ((R2 - R1) * L)

    def f(u):
        return u * np.log((R2 + np.sqrt(R2**2 + u**2)) / (R1 + np.sqrt(R1**2 + u**2)))

    def Bz_axis(z):
        return MU0 * J / 2 * (f(z + L / 2) - f(z - L / 2))

    zs = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 4.0])
    rows = []
    worst = 0.0
    for dr in (1.5, 2.5, 5.0):
        d = design("solenoid_field", {"R1": R1, "R2": R2, "L": L, "NI": NI},
                   [{"name": "coil", "shape": {"rect": {"r": ["R1", "R2"], "z": ["-L/2", "L/2"]}}, "material": "winding"}],
                   windings=[{"region": "coil", "type": "azimuthal", "ampere_turns": "NI"}],
                   mesh={"order": order, "size": 0.08, "surface_size": 0.02,
                         "domain_radius": dr * math.hypot(R2, L / 2)})
        res = d.solve(RUNS / f"solenoid_field_dr{dr}", verbose=0)
        Rb = res.data["mesh"]["info"]["settings"].get("domain_radius", dr * math.hypot(R2, L / 2))
        zin = zs[np.abs(zs) < 0.95 * Rb]  # sample only inside the computational domain
        s = res.sample(np.zeros_like(zin) + 1e-9, zin)
        num = np.asarray(s["Bz"])
        ref = Bz_axis(zin)
        err = np.abs(num - ref) / np.abs(Bz_axis(0.0))
        if not np.all(np.isfinite(err)):
            raise RuntimeError(f"non-finite field samples at domain radius factor {dr}")
        worst = max(worst, float(err.max()))
        rows.append({"domain_radius_factor": dr, "z": zin.tolist(), "Bz_fem": num.tolist(), "Bz_exact": ref.tolist(),
                     "max_error_rel_center": float(err.max()), "W_J": res.W,
                     "dofs": res.data["mesh"]["dofs"]})
    return record("solenoid_field", {"rows": rows, "tolerance": 2e-4, "pass": worst < 2e-4,
                                     "headline": f"max |dBz|/Bz(0) = {worst:.2e} over in-domain z and domain radii 1.5-5"})


# ---------------------------------------------------------------- 2. coax inductance
def check_coax_inductance() -> dict:
    a, b, h, tp, t = 0.4, 0.9, 3.0, 0.02, 0.01
    d = design("coax_L", {"a": a, "b": b, "h": h, "tp": tp, "t": t},
               [{"name": "coil", "shape": {"rect": {"r": [1.0, 1.05], "z": ["-h/2", "h/2"]}}, "material": "winding"},
                {"name": "inner", "shape": {"rect": {"r": ["a-t", "a"], "z": ["-h/2", "h/2"]}}, "material": "copper"},
                {"name": "outer", "shape": {"rect": {"r": ["b", "b+t"], "z": ["-h/2", "h/2"]}}, "material": "copper"},
                {"name": "short", "shape": {"rect": {"r": ["a-t", "b+t"], "z": ["h/2-t", "h/2"]}}, "material": "copper"},
                {"name": "port", "shape": {"rect": {"r": ["a", "b"], "z": ["-h/2", "-h/2+tp"]}}, "material": "vacuum"}],
               windings=[{"region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
               pickup={"type": "poloidal", "port": "port"},
               mesh={"order": 2, "size": 0.1, "surface_size": 0.01})
    res = d.solve(RUNS / "coax_L", verbose=0)
    h_cav = h - t - tp
    L_exact = MU0 / (2 * math.pi) * math.log(b / a) * (h_cav + tp / 3)
    err = abs(res.L_p - L_exact) / L_exact
    return record("coax_inductance", {"L_fem_H": res.L_p, "L_exact_H": L_exact, "rel_error": err, "tolerance": 1e-6,
                                      "pass": err < 1e-6, "headline": f"L_p relative error {err:.1e}"})


# ---------------------------------------------------------------- 3. toroid hole inductance
def toroid_shell(HR: float, y: float = math.exp(-0.5), t: float = 2e-3, order: int = 3, tag: str = "") -> af.Results:
    R = 1.0
    d = design(f"toroid_shell_{HR}{tag}", {"R": R, "y": y, "H": HR * R, "t": t * R},
               [{"name": "winding", "shape": {"frame": {"r": ["y*R+2*t", "R-2*t"], "z": ["-H/2+2*t", "H/2-2*t"],
                                                        "thickness": "t"}}, "material": "winding"},
                {"name": "sheath", "shape": {"frame": {"r": ["y*R", "R"], "z": ["-H/2", "H/2"], "thickness": "t"}},
                 "material": "nb"}],
               windings=[{"region": "winding", "type": "poloidal", "ampere_turns": 1e6}],
               superconductors=[{"body": "sheath", "azimuthal_slit": True}],
               pickup={"type": "azimuthal", "conductor": "sheath"},
               mesh={"order": order, "size": f"max(R,H)/40", "surface_size": "t"})
    return d.solve(RUNS / d.name, verbose=0)


def check_toroid_Lout() -> dict:
    ref = {10: 0.92, 5: 0.85, 2: 0.71, 1: 0.55}   # from an independent code that is not distributed (module docstring)
    y = math.exp(-0.5)
    rows, worst = [], 0.0
    for HR, r in ref.items():
        res = toroid_shell(HR)
        ratio = res.L_p * HR / (MU0 * math.pi * y**2)  # R = 1
        dev = abs(ratio - r)
        worst = max(worst, dev)
        rows.append({"H_over_R": HR, "Lout_ratio_fem": ratio, "reference_2dp": r, "abs_dev": dev,
                     "L_p_H": res.L_p, "g": res.g})
    return record("toroid_Lout", {"rows": rows, "tolerance_abs": 0.006, "pass": worst < 0.006,
                                  "headline": f"L_out H/(pi (yR)^2) agrees with the reference values to {worst:.3f} (refs quoted to 2 dp)"})


# ---------------------------------------------------------------- 4. toroid/CORE duality
def _duality_pair(HR: float, t: float):
    y = math.exp(-0.5)
    tor = toroid_shell(HR, y=y, t=t, tag=f"_dual_t{t}")
    R, H = 1.0, HR
    # CORE dual: SC box = same shell (outer boundary [yR, R] x [-H/2, H/2]); thin solenoid just inside the
    # hole against the inner wall, split around a neck in the inner wall at z = 0.
    d = design(f"core_dual_t{t}", {"R": R, "y": y, "H": H, "t": t, "gn": 2 * t},
               [{"name": "coil_u", "shape": {"rect": {"r": ["y*R-t", "y*R"], "z": ["gn/2", "H/2"]}}, "material": "winding"},
                {"name": "coil_l", "shape": {"rect": {"r": ["y*R-t", "y*R"], "z": ["-H/2", "-gn/2"]}}, "material": "winding"},
                {"name": "box", "shape": {"difference": [{"frame": {"r": ["y*R", "R"], "z": ["-H/2", "H/2"], "thickness": "t"}},
                                                         {"rect": {"r": ["y*R", "y*R+t"], "z": ["-gn/2", "gn/2"]}}]},
                 "material": "nb"},
                {"name": "neck", "shape": {"rect": {"r": ["y*R", "y*R+t"], "z": ["-gn/2", "gn/2"]}}, "material": "vacuum"}],
               windings=[{"region": "coil_u", "type": "azimuthal", "ampere_turns": 5e5},
                         {"region": "coil_l", "type": "azimuthal", "ampere_turns": 5e5}],
               superconductors=[{"body": "box", "azimuthal": {"net_current": 0}}],
               pickup={"type": "poloidal", "port": "neck"},
               mesh={"order": 3, "size": "max(R,H)/40", "surface_size": "t"})
    core = d.solve(RUNS / d.name, verbose=0)
    return tor.g, core.g


def check_duality(HR: float = 1.0) -> dict:
    """Exact duality holds for ideal (zero-thickness) shells with windings on the shell.  With wall/winding
    thickness t the two finite models differ at O(t/R) (insets of the field boxes), so the deviation must
    shrink linearly with t and extrapolate to zero."""
    rows = []
    for t in (4e-3, 2e-3, 1e-3):
        gt, gc = _duality_pair(HR, t)
        rows.append({"t_over_R": t, "g_toroid": gt, "g_core": gc, "rel_dev": (gc - gt) / gt})
    # linear extrapolation of rel_dev to t = 0 from the two thinnest
    d1, d2 = rows[-2]["rel_dev"], rows[-1]["rel_dev"]
    t1, t2 = rows[-2]["t_over_R"], rows[-1]["t_over_R"]
    extrap = d2 - (d1 - d2) / (t1 - t2) * t2
    slope_ok = abs(rows[0]["rel_dev"]) > abs(rows[1]["rel_dev"]) > abs(rows[2]["rel_dev"])
    ok = slope_ok and abs(extrap) < 2e-3
    return record("toroid_core_duality", {"H_over_R": HR, "rows": rows, "extrapolated_rel_dev_t0": extrap, "pass": ok,
                                          "headline": "rel. dev. " + ", ".join(f"{r['rel_dev']:.2e}" for r in rows)
                                                      + f" at t/R = 4e-3, 2e-3, 1e-3; extrapolated to t=0: {extrap:.1e}"})


# ---------------------------------------------------------------- 5. long-limit geometric factors
def check_long_limits() -> dict:
    """Long-limit geometric factors against exact references for the actual geometry.

    Toroid (toroid_shell): the sheath interior is tied, so Psi = L_p Phi_box and
    g = L_out(H) y^2 F^2/Q with F = int g(r)/r dr, Q = int g(r)^2/r dr over the radial current-fraction profile
    g(r) (uniform J in the vertical winding legs, 1 in the field box); L_out(H) = L_p H/(mu0 pi (yR)^2) is
    taken from the same solve (checked against the reference values in toroid_Lout). Remaining corrections: top/bottom
    legs and corners, O(t/H).

    Solenoid (template, coax pickup): long-limit g for the template's radii,
    g_inf = (b^2 - a^2)^2 / (4 ln(b/a) R_eff^2 R_out^2), a = xR, b = R - g - tc,
    pi R_eff^2 = pi R^2 + (winding-thickness energy), R_out = R + tw; finite-H corrections are O(R/H) and are
    removed by extrapolation in 1/H.  For tw, g, tc -> 0 this is (1 - x^2)^2/(4 ln 1/x).
    """
    from scipy import integrate

    y = math.exp(-0.5)
    rows_t = []
    for HR in (5, 10, 20):
        t = 2e-3
        res = toroid_shell(HR, t=t, tag="_long")
        r0, r1 = y + 2 * t, y + 3 * t           # inner vertical leg
        s0, s1 = 1 - 3 * t, 1 - 2 * t           # outer vertical leg
        gin = lambda r: (r * r - r0 * r0) / (r1 * r1 - r0 * r0)  # noqa: E731
        gout = lambda r: (s1 * s1 - r * r) / (s1 * s1 - s0 * s0)  # noqa: E731
        F = integrate.quad(lambda r: gin(r) / r, r0, r1)[0] + math.log(s0 / r1) + integrate.quad(lambda r: gout(r) / r, s0, s1)[0]
        Q = integrate.quad(lambda r: gin(r) ** 2 / r, r0, r1)[0] + math.log(s0 / r1) + integrate.quad(lambda r: gout(r) ** 2 / r, s0, s1)[0]
        Lout = res.L_p * HR / (MU0 * math.pi * y ** 2)
        g_ref = Lout * y ** 2 * F ** 2 / Q
        rows_t.append({"H_over_R": HR, "g_fem": res.g, "L_out_fem": Lout, "g_ref": g_ref, "rel_dev": res.g / g_ref - 1,
                       "g_paper_long": y ** 2 * math.log(1 / y)})
    rows_s = []
    x, tw, gg, tc = 0.534, 0.01, 0.005, 0.005
    a, b, R = x, 1 - gg - tc, 1.0
    eff = R ** 2 + 2 * integrate.quad(lambda r: ((R + tw - r) / tw) ** 2 * r, R, R + tw)[0]  # R_eff^2
    g_inf = (b * b - a * a) ** 2 / (4 * math.log(b / a) * eff * (R + tw) ** 2)
    for HR in (10, 20, 40):
        d = af.template("solenoid", R=1.0, H=HR, x=x, tw=tw, g=gg, tc=tc,
                        mesh={"order": 3, "size": "max(R, H)/40", "surface_size": "min(tc, g)"})
        res = d.solve(RUNS / f"long_solenoid_{HR}", verbose=0)
        rows_s.append({"H_over_R": HR, "g_fem": res.g, "g_inf": g_inf, "rel_dev": res.g / g_inf - 1,
                       "g_paper": (1 - x * x) ** 2 / (4 * math.log(1 / x))})
    # Richardson in 1/H from the two longest solenoids
    d1, d2 = rows_s[-2]["rel_dev"], rows_s[-1]["rel_dev"]
    sol_extrap = 2 * d2 - d1
    tor_worst = max(abs(r["rel_dev"]) for r in rows_t)
    ok = tor_worst < 2e-3 and abs(sol_extrap) < 2e-3 and abs(rows_s[-1]["rel_dev"]) < abs(rows_s[0]["rel_dev"])
    return record("long_limits", {"toroid": rows_t, "solenoid": rows_s, "solenoid_extrapolated_rel_dev": sol_extrap,
                                  "pass": ok,
                                  "headline": f"toroid g vs L_out y^2 F^2/Q: max |dev| {tor_worst:.1e} (H/R 5-20); "
                                              f"solenoid g vs exact long limit: " +
                                              ", ".join(f"{r['rel_dev']:+.2e}" for r in rows_s) +
                                              f" at H/R 10, 20, 40, extrapolated {sol_extrap:+.1e}"})


# ---------------------------------------------------------------- 6. finite current-sheet solenoid, cost-metric optimum
def check_solenoid_optimum(max_evals: int = 36) -> dict:
    """Derivative-free reference optimum of the cost metric at the cost-optimal size (readout benchmark REF_BENCH) over
    H/R and x of a finite current-sheet solenoid, by Nelder-Mead on full solves (order 3). The cost metric is a scaling
    metric for comparing designs, not a construction estimate; here it is only the objective that
    benchmarks/adjoint_validation.py and global_opt_validation.py also optimize, and whose optimum they compare with
    this one. Checks: the optimum lies inside the search box, and at the cost-optimal size the magnet and cryostat terms
    stand in the closed-form ratio p_area/p_energy (d/dl [a (W/l^2)^pE + a kappa (A l^2)^pA] = 0)."""
    from scipy.optimize import minimize

    def build(hr, x):
        return af.template("solenoid", R=1.0, H=hr, x=x, tw="0.002*R", g="0.001*R", tc="0.001*R", tp="0.002*R",
                           mesh={"order": 3, "size": "max(R,H)/40", "surface_size": "0.002*R"},
                           benchmarks=[REF_BENCH])

    hist = []

    def f(p):
        hr, x = float(p[0]), float(p[1])
        if not (0.5 <= hr <= 4 and 0.3 <= x <= 0.8):
            return 1e9
        res = build(hr, x).solve(RUNS / "sol_opt" / "cur", verbose=0)
        b = res.benchmark(REF_BENCH["name"])
        c = b["cost_optimal_size"]
        hist.append({"H_over_R": hr, "x": x, "cost_MCHF": c["cost"]["total_MCHF"], "R_m": c["device_radius_m"],
                     "scale": c["length_scale"], "W_J": c["W_mag_J"], "B_max_T": c["B_max_T"], "g": res.g,
                     "magnet_over_cryostat": c["cost"]["magnet_MCHF"] / c["cost"]["cryostat_MCHF"]})
        return c["cost"]["total_MCHF"]

    r = minimize(f, [1.3, 0.55], method="Nelder-Mead",
                 options={"maxfev": max_evals, "xatol": 2e-3, "fatol": 1e-4, "initial_simplex": [[1.3, 0.55], [1.6, 0.55], [1.3, 0.6]]})
    best = min(hist, key=lambda h: h["cost_MCHF"])
    # center field at the optimum: rebuild at the optimal scale and sample
    hr, x, l = best["H_over_R"], best["x"], best["scale"]
    res = build(hr, x).solve(RUNS / "sol_opt" / "best", verbose=0)
    b = res.benchmark(REF_BENCH["name"])
    s_amp = b["amplitude_scale"] * l ** (-2.5)
    Bc = float(res.sample([1e-9], [0.0])["Bz"][0]) * s_amp
    R_opt = 1.0 * l  # winding radius scales with l
    H_opt = hr * R_opt
    got = {"R_m": R_opt, "H_m": H_opt, "x": x, "B_center_T": Bc, "W_MJ": best["W_J"] / 1e6}
    cost = build(hr, x).doc.get("cost") or {}
    ratio_expected = cost.get("p_area", 0.8) / cost.get("p_energy", 0.7)
    ratio_dev = abs(best["magnet_over_cryostat"] / ratio_expected - 1)
    interior = 0.5 < hr < 4 and 0.3 < x < 0.8
    ok = bool(interior and ratio_dev < 1e-6)
    return record("cost_metric_optimum_reference", {"benchmark": REF_BENCH, "computed": got,
                                            "cost_MCHF": best["cost_MCHF"],
                                            "magnet_over_cryostat": best["magnet_over_cryostat"],
                                            "magnet_over_cryostat_expected": ratio_expected,
                                            "magnet_over_cryostat_rel_dev": ratio_dev,
                                            "nelder_mead": {"success": bool(r.success), "nfev": int(r.nfev),
                                                            "message": str(r.message)},
                                            "history": hist, "pass": ok,
                                            "note": "the cost metric is a scaling metric, not a construction estimate; "
                                                    "it is flat in H/R near the optimum",
                                            "headline": f"R={R_opt:.2f} m, H={H_opt:.2f} m, x={x:.3f}, Bc={Bc:.2f} T, "
                                                        f"W={best['W_J'] / 1e6:.0f} MJ, magnet/cryostat "
                                                        f"{best['magnet_over_cryostat']:.6f} (expected {ratio_expected:.6f})"})


# ---------------------------------------------------------------- 7. superconducting sphere in a uniform field
def check_sc_sphere() -> dict:
    """Exact: surface field 1.5 B0 sin(theta); on-axis Bz = B0 (1 - a^3/z^3) for |z| > a.
    Exercises the background-field (reduced) formulation and superconductor exclusion."""
    a, B0 = 0.5, 1.0
    d = design("sc_sphere", {"a": a}, [{"name": "sphere", "shape": {"disk": {"center": [0, 0], "radius": "a"}}, "material": "nb"}],
               background={"type": "uniform", "Bz": B0},
               mesh={"order": 3, "size": 0.05, "surface_size": 0.01, "domain_radius": 2.0})
    res = d.solve(RUNS / "sc_sphere", verbose=0)
    peak = res.data["fields"]["superconductor_surface_T"]["sphere"]
    z = np.array([0.6, 0.8, 1.0, 1.5])
    bz = np.asarray(res.sample(np.full_like(z, 1e-9), z)["Bz"])
    ref = B0 * (1 - a**3 / z**3)
    th = np.linspace(0.1, np.pi - 0.1, 9)
    s2 = res.sample(1.0001 * a * np.sin(th), 1.0001 * a * np.cos(th))
    bt = np.hypot(s2["Br"], s2["Bz"])
    e1 = abs(peak - 1.5 * B0) / (1.5 * B0)
    e2 = float(np.max(np.abs(bz - ref)))
    e3 = float(np.max(np.abs(bt - 1.5 * B0 * np.sin(th))))
    ok = e1 < 1e-4 and e2 < 2e-4 and e3 < 5e-4
    return record("sc_sphere", {"surface_peak_T": peak, "axis_Bz": bz.tolist(), "axis_exact": ref.tolist(),
                                "surface_profile_max_abs_err_T": e3, "pass": ok,
                                "headline": f"peak {peak:.7f} T (exact 1.5), axis err {e2:.1e} T, surface profile err {e3:.1e} T"})


ALL = {"solenoid_field": check_solenoid_field, "coax_inductance": check_coax_inductance,
       "toroid_Lout": check_toroid_Lout, "duality": check_duality, "long_limits": check_long_limits,
       "solenoid_optimum": check_solenoid_optimum, "sc_sphere": check_sc_sphere}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(ALL)
    out = [ALL[n]() for n in names]
    sys.exit(0 if all(o.get("pass") for o in out) else 1)
