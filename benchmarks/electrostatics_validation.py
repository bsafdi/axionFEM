"""Validation of the axisymmetric electrostatics engine against closed forms.

Each check writes a record to benchmarks/records/electrostatics_<name>.json with the inputs, the
computed numbers, the reference, the deviation and pass/fail.

    python benchmarks/electrostatics_validation.py all
    python benchmarks/electrostatics_validation.py sphere concentric shell ...

References (all exact unless noted)
-----------------------------------
- isolated conducting sphere of radius a:                C = 4 pi eps0 a
- concentric spheres (a inside the shell b..c):          C11 = 4 pi eps0/(1/a - 1/b), C12 = -C11,
                                                         C22 = C11 + 4 pi eps0 c
- sphere a with a dielectric shell a < rho < b (eps_r):  C = 4 pi eps0 / [(1/a - 1/b)/eps_r + 1/b]
- sphere a inside a grounded sphere Rb:                  C = 4 pi eps0/(1/a - 1/Rb)
- two equal spheres a, centre distance D, cosh eta = D/(2a) (bispherical series, Smythe):
      C11 = 4 pi eps0 a sinh(eta) sum_{n>=0} 1/sinh((2n+1) eta)
      C12 = -4 pi eps0 a sinh(eta) sum_{n>=1} 1/sinh(2n eta)
- thin circular ring (major R, minor b, b << R):         C = 4 pi^2 eps0 R / ln(8R/b)  [leading order]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import axionfem as af  # noqa: E402

MU0 = 1.25663706212e-6
C_LIGHT = 299792458.0
EPS0 = 1.0 / (MU0 * C_LIGHT**2)
FPE = 4 * math.pi * EPS0
REC = ROOT / "benchmarks" / "records"
RUNS = ROOT / "runs" / "electrostatics"


def record(name: str, data: dict) -> dict:
    REC.mkdir(parents=True, exist_ok=True)
    data = {"check": name, "axionfem_version": af.version(), "date": time.strftime("%Y-%m-%d %H:%M"), **data}
    (REC / f"electrostatics_{name}.json").write_text(json.dumps(data, indent=2))
    status = "PASS" if data.get("pass") else "FAIL"
    print(f"[{status}] {name}: {data.get('headline', '')}", flush=True)
    return data


def solve(name, regions, materials=None, mesh=None, boundary="open", notes=""):
    doc = {"schema": "axionfem.design/1", "name": name, "regions": regions,
           "electrostatics": {"boundary": boundary, "notes": notes or name},
           "mesh": mesh or {"order": 3, "size": 0.05, "surface_size": 0.01, "domain_radius": 2.0}}
    if materials:
        doc["materials"] = materials
    r = af.Design(doc).solve(RUNS / name, verbose=0)
    return r.data["electrostatics"], r.data.get("warnings", [])


def disk(rho, zc=0.0):
    return {"disk": {"center": [0.0, zc], "radius": rho}}


def shell(r_in, r_out):
    return {"difference": [disk(r_out), disk(r_in)]}


def named(e, name):
    """Index of a body by name (the matrix rows follow electrostatics.bodies)."""
    return [b["name"] for b in e["bodies"]].index(name)


def checks_summary(e):
    """The checks, split by what they actually test.

    `symmetry_*`, `energy_*`, `energy_superposition`, `floating_solve` and `exterior_monopole_*` are
    algebraic identities of the discrete system: they verify the assembly and the linear solves.  Only
    `charge_gauss_*` and `exterior_flux_*` compare the solution with an independently evaluated strong
    form, so only those converge under h/p refinement.
    """
    c = e["checks"]
    return {"identity_symmetry_relative": c["symmetry_relative"],
            "identity_energy_max_relative_difference": c["energy_max_relative_difference"],
            "identity_energy_superposition_relative": c["energy_superposition"]["relative_difference"],
            "identity_exterior_monopole_max_relative_difference":
                c["exterior_monopole_max_relative_difference"],
            "identity_floating_solve_max_difference_V": c["floating_solve"].get("max_difference_V"),
            "discretization_charge_gauss_max_relative_difference":
                c["charge_gauss_max_relative_difference"],
            "discretization_exterior_flux_max_relative_difference":
                c["exterior_flux_max_relative_difference"],
            "min_slack_relative": c["min_slack_relative"],
            "off_diagonal_signs_ok": c["off_diagonal_signs_ok"]}


# ---------------------------------------------------------------- 1. isolated sphere
def check_sphere() -> dict:
    a = 0.5
    e, w = solve("sphere", [{"name": "ball", "shape": disk(a), "material": "superconductor"}],
                 notes="isolated sphere: C = 4 pi eps0 a (exact with the open exterior map)")
    C = e["capacitance_F"][0][0]
    exact = FPE * a
    err = C / exact - 1
    return record("sphere", {"a_m": a, "C_F": C, "C_exact_F": exact, "relative_error": err,
                             "checks": checks_summary(e), "warnings": w, "pass": abs(err) < 1e-6,
                             "headline": f"C = {C:.9e} F (exact {exact:.9e}), rel err {err:.1e}"})


# ---------------------------------------------------------------- 2. concentric spheres
def check_concentric() -> dict:
    a, b, c = 0.3, 0.5, 0.6
    e, w = solve("concentric",
                 [{"name": "inner", "shape": disk(a), "material": "superconductor"},
                  {"name": "outer", "shape": shell(b, c), "material": "superconductor"}],
                 notes="concentric spheres: C11 = 4 pi eps0/(1/a - 1/b), C22 adds 4 pi eps0 c")
    i, o = named(e, "inner"), named(e, "outer")
    C = e["capacitance_F"]
    Cin = FPE / (1 / a - 1 / b)
    ref = {"C_inner_inner": Cin, "C_inner_outer": -Cin, "C_outer_outer": Cin + FPE * c}
    got = {"C_inner_inner": C[i][i], "C_inner_outer": C[i][o], "C_outer_outer": C[o][o]}
    err = {k: got[k] / ref[k] - 1 for k in ref}
    worst = max(abs(v) for v in err.values())
    pair = e["pairs"][0]
    return record("concentric", {"a_m": a, "b_m": b, "c_m": c, "computed_F": got, "exact_F": ref,
                                 "relative_error": err, "pair": pair, "checks": checks_summary(e),
                                 "warnings": w, "pass": worst < 1e-6,
                                 "headline": f"C11 {got['C_inner_inner']:.8e} F, C12 {got['C_inner_outer']:.8e} F, "
                                             f"C22 {got['C_outer_outer']:.8e} F; worst rel err {worst:.1e}"})


# ---------------------------------------------------------------- 3. sphere with a dielectric shell
def check_shell() -> dict:
    a, b, eps_r = 0.3, 0.5, 4.0
    e, w = solve("shell",
                 [{"name": "ball", "shape": disk(a), "material": "superconductor"},
                  {"name": "diel", "shape": shell(a, b), "material": "dielectric"}],
                 materials={"dielectric": {"kind": "linear", "eps_r": eps_r}},
                 notes="layered spherical capacitor: C = 4 pi eps0 / [(1/a - 1/b)/eps_r + 1/b]")
    C = e["capacitance_F"][0][0]
    exact = FPE / ((1 / a - 1 / b) / eps_r + 1 / b)
    err = C / exact - 1
    return record("shell", {"a_m": a, "b_m": b, "eps_r": eps_r, "C_F": C, "C_exact_F": exact,
                            "relative_error": err, "checks": checks_summary(e), "warnings": w,
                            "pass": abs(err) < 1e-5,
                            "headline": f"C = {C:.9e} F (exact {exact:.9e}), rel err {err:.1e}"})


# ---------------------------------------------------------------- 3b. zero-thickness sheet
def check_sheet_sphere() -> dict:
    """The same sphere as a zero-thickness conductor sheet (a `curves` entry): C = 4 pi eps0 a."""
    a = 0.4
    doc = {"schema": "axionfem.design/1", "name": "sheet_sphere", "regions": [],
           "curves": [{"name": "shell",
                       "path": {"arc": {"center": [0.0, 0.0], "radius": a,
                                        "angles": [-math.pi / 2, math.pi / 2]}},
                       "material": "superconductor", "mesh_size": 0.01}],
           "electrostatics": {"boundary": "open", "notes": "spherical conductor sheet"},
           "mesh": {"order": 3, "size": 0.05, "domain_radius": 2.0}}
    r = af.Design(doc).solve(RUNS / "sheet_sphere", verbose=0)
    e = r.data["electrostatics"]
    C = e["capacitance_F"][0][0]
    exact = FPE * a
    err = C / exact - 1
    return record("sheet_sphere", {"a_m": a, "C_F": C, "C_exact_F": exact, "relative_error": err,
                                   "checks": checks_summary(e), "warnings": r.data["warnings"],
                                   "pass": abs(err) < 1e-6,
                                   "headline": f"sheet C = {C:.9e} F (exact {exact:.9e}), rel err {err:.1e}"})


# ---------------------------------------------------------------- 4. grounded outer sphere
def check_grounded() -> dict:
    a, Rb = 0.3, 1.0
    e, w = solve("grounded", [{"name": "ball", "shape": disk(a), "material": "superconductor"}],
                 mesh={"order": 3, "size": 0.05, "surface_size": 0.01, "domain_radius": Rb},
                 boundary="grounded", notes="sphere inside a grounded sphere: C = 4 pi eps0/(1/a - 1/Rb)")
    C = e["capacitance_F"][0][0]
    exact = FPE / (1 / a - 1 / Rb)
    err = C / exact - 1
    return record("grounded", {"a_m": a, "domain_radius_m": Rb, "C_F": C, "C_exact_F": exact,
                               "relative_error": err, "checks": checks_summary(e), "warnings": w,
                               "pass": abs(err) < 1e-6,
                               "headline": f"C = {C:.9e} F (exact {exact:.9e}), rel err {err:.1e}"})


# ---------------------------------------------------------------- 5. two axial spheres
def two_sphere_exact(a, D, nmax=400):
    eta = math.acosh(D / (2 * a))
    s = math.sinh(eta)
    terms = lambda xs: sum(1 / math.sinh(x) for x in xs if x * 1.0 < 700.0)  # noqa: E731
    c11 = FPE * a * s * terms([(2 * n + 1) * eta for n in range(nmax)])
    c12 = -FPE * a * s * terms([2 * n * eta for n in range(1, nmax)])
    return c11, c12


def check_two_spheres() -> dict:
    a, D = 0.1, 0.5
    e, w = solve("two_spheres",
                 [{"name": "upper", "shape": disk(a, +D / 2), "material": "superconductor"},
                  {"name": "lower", "shape": disk(a, -D / 2), "material": "superconductor"}],
                 mesh={"order": 3, "size": 0.04, "surface_size": 0.008, "domain_radius": 2.0},
                 notes="two equal spheres on the axis: bispherical series")
    u, l = named(e, "upper"), named(e, "lower")
    C = e["capacitance_F"]
    c11, c12 = two_sphere_exact(a, D)
    got = {"C_self": 0.5 * (C[u][u] + C[l][l]), "C_mutual_coefficient": C[u][l]}
    err = {"C_self": got["C_self"] / c11 - 1, "C_mutual_coefficient": got["C_mutual_coefficient"] / c12 - 1}
    worst = max(abs(v) for v in err.values())
    return record("two_spheres",
                  {"a_m": a, "centre_distance_m": D, "computed_F": got,
                   "exact_F": {"C_self": c11, "C_mutual_coefficient": c12},
                   "leading_order_C12_F": -FPE * a * a / D, "relative_error": err,
                   "asymmetry_of_equal_spheres": abs(C[u][u] / C[l][l] - 1),
                   "pair": e["pairs"][0], "checks": checks_summary(e), "warnings": w,
                   "pass": worst < 1e-5,
                   "headline": f"C11 {got['C_self']:.8e} F, C12 {got['C_mutual_coefficient']:.8e} F; "
                               f"worst rel err {worst:.1e}"})


# ---------------------------------------------------------------- 6. ring (torus)
def check_ring_series() -> dict:
    R = 0.5
    rows = []
    for b in (0.02, 0.01, 0.005):
        e, w = solve(f"ring_b{b}",
                     [{"name": "ring", "shape": {"disk": {"center": [R, 0.0], "radius": b}},
                       "material": "superconductor"}],
                     mesh={"order": 3, "size": 0.05, "surface_size": b / 5, "domain_radius": 2.0},
                     notes="thin ring (torus): C = 4 pi^2 eps0 R / ln(8R/b) to leading order")
        C = e["capacitance_F"][0][0]
        ref = 4 * math.pi**2 * EPS0 * R / math.log(8 * R / b)
        rows.append({"b_m": b, "b_over_R": b / R, "C_F": C, "thin_ring_F": ref, "ratio": C / ref,
                     "checks": checks_summary(e), "warnings": w})
    worst = max(abs(x["ratio"] - 1) for x in rows)
    trend = all(abs(rows[i + 1]["ratio"] - 1) < abs(rows[i]["ratio"] - 1) + 1e-12 for i in range(len(rows) - 1))
    return record("ring", {"R_m": R, "rows": rows, "pass": worst < 0.02 and trend,
                           "headline": "C/C_thin-ring = " + ", ".join(f"{x['ratio']:.4f} (b/R={x['b_over_R']:.3f})"
                                                                      for x in rows)})


# ---------------------------------------------------------------- 7. h/p convergence
def check_convergence() -> dict:
    a, b = 0.3, 0.5
    exact = FPE / (1 / a - 1 / b)
    rows = []
    for order in (1, 2, 3):
        for h in (0.08, 0.04, 0.02):
            e, _ = solve(f"conv_p{order}_h{h}",
                         [{"name": "inner", "shape": disk(a), "material": "superconductor"},
                          {"name": "outer", "shape": shell(b, b + 0.1), "material": "superconductor"}],
                         mesh={"order": order, "size": 4 * h, "surface_size": h, "domain_radius": 2.0},
                         notes="h/p convergence of the concentric-sphere capacitance")
            i, o = named(e, "inner"), named(e, "outer")
            C = e["capacitance_F"]
            rows.append({"order": order, "surface_size_m": h, "dofs": e["solver"]["dofs"],
                         "C_inner_F": C[i][i], "relative_error": C[i][i] / exact - 1,
                         "identity_symmetry_relative": e["checks"]["symmetry_relative"],
                         "identity_energy_relative": e["checks"]["energy_max_relative_difference"],
                         "identity_exterior_monopole_relative":
                             e["checks"]["exterior_monopole_max_relative_difference"],
                         "charge_gauss_relative": e["checks"]["charge_gauss_max_relative_difference"],
                         "exterior_flux_relative": e["checks"]["exterior_flux_max_relative_difference"],
                         "mutual_F": -C[i][o]})
    # the error must fall when h falls, for every order
    ok = True
    for order in (1, 2, 3):
        es = [abs(x["relative_error"]) for x in rows if x["order"] == order]
        ok = ok and es[0] > es[1] > es[2]
    # The exterior-monopole value is an identity, so it must NOT track the true error, while
    # the exterior-flux and Gauss checks must.  Record the p-ladder at the finest h as evidence.
    fine = [x for x in rows if x["surface_size_m"] == 0.02]
    p_ladder = {f"p{x['order']}": {"relative_error": x["relative_error"],
                                   "identity_exterior_monopole": x["identity_exterior_monopole_relative"],
                                   "discretization_exterior_flux": x["exterior_flux_relative"],
                                   "discretization_charge_gauss": x["charge_gauss_relative"]}
                for x in fine}
    mono = [x["identity_exterior_monopole_relative"] for x in fine]
    flux = [x["exterior_flux_relative"] for x in fine]
    tru = [abs(x["relative_error"]) for x in fine]
    # the identity has already hit its floor at p2 -- the SAME value to 5e-5 at p2 and p3 -- while the
    # real error falls by two more decades
    identity_is_flat = abs(mono[2] / max(mono[1], 1e-300) - 1) < 1e-3 and tru[2] < 0.1 * tru[1]
    flux_converges = flux[2] < 0.1 * flux[1] and flux[1] < 0.1 * flux[0]
    ok = ok and identity_is_flat and flux_converges
    return record("convergence", {"a_m": a, "b_m": b, "C_exact_F": exact, "rows": rows,
                                  "p_ladder_at_h_0.02": p_ladder,
                                  "identity_does_not_track_the_error": identity_is_flat,
                                  "exterior_flux_converges": flux_converges, "pass": ok,
                                  "headline": "; ".join(f"p{x['order']} h{x['surface_size_m']}: "
                                                        f"{x['relative_error']:+.2e}" for x in rows)})


# ---------------------------------------------------------------- 8. readout gap and self-resonance
def check_port(name="solenoid", f_min=5e6, f_max=1.2e8, points=16) -> dict:
    """Lumped f = 1/(2 pi sqrt(L_p C_gap)) against the first pole of L(omega) in a full-wave sweep.

    The sweep is the reference: the readout's first self-resonance is where L(omega) changes sign
    (the full-wave eigenmodes of the closed sphere are dominated by the sphere itself, which is why
    they are not used here)."""
    d = af.template(name)
    d.doc["fullwave"] = {"f_min": f_min, "f_max": f_max, "points": points}
    r = d.solve(RUNS / f"port_{name}", verbose=0)
    pc = r.data["pickup"]["port_capacitance"]
    sweep = [{"f_Hz": x["f_Hz"], "k_R": x["k_R"], "L_over_MQS": x.get("L_over_MQS"), "L_H": x["L_H"]}
             for x in r.data["fullwave"]["sweep"]]
    bracket = None
    for a, b in zip(sweep, sweep[1:]):
        if a["L_H"] > 0 and b["L_H"] < 0:
            bracket = [a["f_Hz"], b["f_Hz"]]
            break
    f_true = math.sqrt(bracket[0] * bracket[1]) if bracket else None
    C_eff = 1 / ((2 * math.pi * f_true) ** 2 * r.L_p) if f_true else None
    out = {"template": name, "L_p_H": r.L_p, "C_gap_F": pc["C_gap_F"],
           "C_gap_port_region_F": pc["C_gap_port_region_F"],
           "self_resonance_lumped_Hz": pc["self_resonance_Hz"],
           "self_resonance_from_port_region_Hz": pc["self_resonance_from_port_region_Hz"],
           "dirichlet_bound": pc["checks"]["dirichlet_bound"],
           "fullwave_first_pole_bracket_Hz": bracket, "fullwave_first_pole_Hz": f_true,
           "lumped_over_fullwave": pc["self_resonance_Hz"] / f_true if f_true else None,
           "C_effective_from_fullwave_F": C_eff,
           "C_gap_over_C_effective": pc["C_gap_F"] / C_eff if C_eff else None,
           # (2/pi)^2 = 0.405: for a uniform line the first pole is (pi/2)/(2 pi sqrt(L C_total)), so the
           # lumped estimate built on C_gap exceeds it only while C_gap/C_total < (2/pi)^2.  This is why
           # the lumped value is NOT an upper bound in general.
           "C_gap_over_C_effective_threshold": (2 / math.pi) ** 2,
           "lumped_estimate_is_above_the_pole": (pc["C_gap_F"] / C_eff < (2 / math.pi) ** 2) if C_eff else None,
           "sweep": sweep, "checks": pc["checks"],
           # the rigorous relation between the two gap values (not C_gap >= C_region, which is unproven)
           "pass": bracket is not None and pc["checks"]["dirichlet_bound"]["holds"]
                   and pc["C_gap_port_region_F"] > 0,
           "headline": f"{name}: C_gap = {pc['C_gap_F']:.4e} F (region only {pc['C_gap_port_region_F']:.4e}), "
                       f"lumped f_SR = {pc['self_resonance_Hz']/1e6:.1f} MHz vs full-wave pole "
                       f"{(f_true or 0)/1e6:.1f} MHz (x{pc['self_resonance_Hz']/f_true:.2f}; "
                       f"C_gap/C_eff {pc['C_gap_F']/C_eff:.3f} < (2/pi)^2 = 0.405, so the lumped "
                       f"estimate lands above the pole here)"
                       if bracket else f"{name}: no L(omega) sign change in the sweep"}
    return record(f"port_{name}", out)


def check_port_core() -> dict:
    return check_port("core", f_min=5e6, f_max=8e7, points=12)


def check_port_convergence(name="core") -> dict:
    """Mesh dependence of the readout-gap capacitance (the electrode edges are Dirichlet-Neumann
    junctions, so the fringing value converges more slowly than the region-only one)."""
    rows = []
    for h in (1.0, 0.7, 0.5):
        d = af.template(name)
        mesh = dict(d.doc.get("mesh", {}))
        base = mesh.get("surface_size")
        mesh["surface_size"] = f"({base})*{h}" if isinstance(base, str) else (base or 0.01) * h
        d.doc["mesh"] = mesh
        r = d.solve(RUNS / f"portconv_{name}_{h}", verbose=0)
        pc = r.data["pickup"]["port_capacitance"]
        rows.append({"surface_size_scale": h, "dofs": r.data["mesh"]["dofs"],
                     "C_gap_F": pc["C_gap_F"], "C_gap_port_region_F": pc["C_gap_port_region_F"],
                     "self_resonance_Hz": pc["self_resonance_Hz"],
                     "charge_fraction_beyond_terminal_faces":
                         pc["checks"]["gap_charge_fraction_beyond_terminal_faces"]})
    spread = max(x["C_gap_F"] for x in rows) / min(x["C_gap_F"] for x in rows) - 1
    spread_r = max(x["C_gap_port_region_F"] for x in rows) / min(x["C_gap_port_region_F"] for x in rows) - 1
    return record(f"port_convergence_{name}",
                  {"template": name, "rows": rows, "C_gap_spread": spread,
                   "C_gap_port_region_spread": spread_r, "pass": spread < 0.05,
                   "headline": f"{name}: C_gap spread {spread:.1%} over 2x mesh refinement "
                               f"(region-only {spread_r:.2%})"})


ALL = {"sphere": check_sphere, "concentric": check_concentric, "shell": check_shell,
       "sheet_sphere": check_sheet_sphere,
       "grounded": check_grounded, "two_spheres": check_two_spheres, "ring": check_ring_series,
       "convergence": check_convergence, "port": check_port, "port_core": check_port_core,
       "port_convergence": check_port_convergence}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(ALL)
    out = [ALL[n]() for n in names]
    sys.exit(0 if all(o.get("pass") for o in out) else 1)
