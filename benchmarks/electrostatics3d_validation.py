"""Validation of the 3D electrostatics engine against closed forms, the axisymmetric solver and
published values.

Each check writes a record to benchmarks/records/electrostatics3d_<name>.json with the inputs, the
computed numbers, the reference, the deviation and pass/fail.

    python benchmarks/electrostatics3d_validation.py all
    python benchmarks/electrostatics3d_validation.py sphere concentric cube ...

References
----------
- isolated conducting sphere of radius a:                C = 4 pi eps0 a                        (exact)
- concentric spheres (a inside the shell b..c):           C11 = 4 pi eps0/(1/a - 1/b), C12 = -C11,
                                                          C22 = C11 + 4 pi eps0 c               (exact)
- sphere a with a dielectric shell a < rho < b (eps_r):   C = 4 pi eps0/[(1/a - 1/b)/eps_r + 1/b] (exact)
- sphere a inside a grounded sphere Rb:                   C = 4 pi eps0/(1/a - 1/Rb)             (exact)
- two equal spheres a, centre distance D, cosh eta = D/(2a) (bispherical series, Smythe):
      C11 = 4 pi eps0 a sinh(eta) sum_{n>=0} 1/sinh((2n+1) eta)
      C12 = -4 pi eps0 a sinh(eta) sum_{n>=1} 1/sinh(2n eta)                                    (exact)
  The series depends only on a and D, so the SAME reference holds for a pair that is not aligned with
  any coordinate axis -- a genuinely non-axisymmetric case with an exact answer (check `offset`).
- cube of side L:  C = 0.66067813 * 4 pi eps0 * L  (published value; the independent determinations in the
  literature -- Read 1997 (0.6606785), Mansfield/Douglas/Garboczi 2001, Hwang & Mascagni 2004 -- agree with
  it to about 1e-6 relative, far below the discretization error of a 3D finite-element mesh).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import axionfem as af  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

MU0 = 1.25663706212e-6
C_LIGHT = 299792458.0
EPS0 = 1.0 / (MU0 * C_LIGHT**2)
FPE = 4 * math.pi * EPS0
CUBE_FACTOR = 0.66067813  # C / (4 pi eps0 L) of a cube of side L
REC = ROOT / "benchmarks" / "records"
RUNS = ROOT / "runs" / "electrostatics3d"


def record(name: str, data: dict) -> dict:
    REC.mkdir(parents=True, exist_ok=True)
    data = {"check": name, "axionfem_version": af.version(), "date": time.strftime("%Y-%m-%d %H:%M"), **data}
    (REC / f"electrostatics3d_{name}.json").write_text(json.dumps(data, indent=2))
    status = "PASS" if data.get("pass") else "FAIL"
    print(f"[{status}] {name}: {data.get('headline', '')}", flush=True)
    return data


# ------------------------------------------------------------------ geometry helpers
def ball(r, center=(0, 0, 0)):
    return {"sphere": {"center": list(center), "radius": r}}


def hollow(r_in, r_out, center=(0, 0, 0)):
    return {"difference": [ball(r_out, center), ball(r_in, center)]}


def box(L, center=(0, 0, 0)):
    h = 0.5 * L
    return {"box": {"min": [center[i] - h for i in range(3)], "max": [center[i] + h for i in range(3)]}}


DEFAULT_MESH = {"order": 2, "size": 0.12, "surface_size": 0.04, "domain_radius": 1.6}


def design(name, regions, materials=None, mesh=None, boundary="open", notes=""):
    doc = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d", "regions": regions,
           "electrostatics": {"boundary": boundary, "notes": notes or name},
           "mesh": {**DEFAULT_MESH, **(mesh or {})}}
    if materials:
        doc["materials"] = materials
    return af.Design(doc)


def solve(name, regions, materials=None, mesh=None, boundary="open", notes="", ranks=1, wd=None):
    d = design(name, regions, materials, mesh, boundary, notes)
    r = d.solve(wd or (RUNS / name), verbose=0, mpi_ranks=ranks)
    return r.data["electrostatics"], r.data.get("warnings", [])


def named(e, name):
    """Index of a body by name (the matrix rows follow electrostatics.bodies)."""
    return [b["name"] for b in e["bodies"]].index(name)


def checks_summary(e):
    c = e["checks"]
    return {"symmetry_relative": c["symmetry_relative"],
            "energy_max_relative_difference": c["energy_max_relative_difference"],
            "charge_gauss_max_relative_difference": c["charge_gauss_max_relative_difference"],
            "exterior_monopole_max_relative_difference": c["exterior_monopole_max_relative_difference"],
            "energy_superposition_relative": c["energy_superposition"]["relative_difference"],
            "mixed_potential_max_relative_difference": c["mixed_potential_solve"]["max_relative_difference"],
            "floating_max_relative_charge": c["floating_solve"].get("max_relative_floating_charge"),
            "min_slack_relative": c["min_slack_relative"],
            "off_diagonal_signs_ok": c["off_diagonal_signs_ok"],
            "max_relative_residual": e["solver"]["max_relative_residual"],
            "dofs": e["solver"]["dofs"]}


def _coth_series(eta, odd):
    """sum_k 1/sinh(k eta) over k = 1, 3, 5, ... (odd) or k = 2, 4, 6, ... (even).

    The terms fall off like exp(-k eta), so the sum is converged long before k eta reaches 700, where
    math.sinh would overflow."""
    total, k = 0.0, (1 if odd else 2)
    while k * eta < 700:
        total += 1.0 / math.sinh(k * eta)
        k += 2
    return total


def bispherical(a, D):
    """Exact C11, C12 of two equal spheres of radius a whose centres are D apart."""
    eta = math.acosh(D / (2 * a))
    s = math.sinh(eta)
    return FPE * a * s * _coth_series(eta, True), -FPE * a * s * _coth_series(eta, False)


# ---------------------------------------------------------------- 1. isolated sphere
def check_sphere() -> dict:
    a = 0.5
    e, w = solve("sphere", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
                 mesh={"surface_size": 0.04, "size": 0.15, "domain_radius": 1.5},
                 notes="isolated sphere: C = 4 pi eps0 a (the monopole escapes through the infinity shell)")
    C = e["capacitance_F"][0][0]
    exact = FPE * a
    err = C / exact - 1
    return record("sphere", {"a_m": a, "C_F": C, "C_exact_F": exact, "relative_error": err,
                             "checks": checks_summary(e), "warnings": w, "pass": abs(err) < 2e-3,
                             "headline": f"C = {C:.8e} F (exact {exact:.8e}), rel err {err:+.2e}"})


# ---------------------------------------------------------------- 1b. the exterior map does its job
def check_exterior() -> dict:
    """The strongest test of the open boundary: with an exact exterior map the capacitance cannot depend on
    where the domain ends. The same sphere is solved at three domain radii (same surface mesh), and against
    a grounded boundary at the same radii, which does depend on them (C = 4 pi eps0/(1/a - 1/Rb))."""
    a = 0.5
    rows = []
    for Rb in (1.0, 1.5, 2.5):
        e, w = solve("exterior", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
                     mesh={"order": 2, "surface_size": 0.05, "size": 0.15, "domain_radius": Rb},
                     wd=RUNS / f"exterior_open_{Rb}",
                     notes="domain-radius independence of the open boundary")
        eg, _ = solve("exterior", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
                      boundary="grounded",
                      mesh={"order": 2, "surface_size": 0.05, "size": 0.15, "domain_radius": Rb},
                      wd=RUNS / f"exterior_grounded_{Rb}",
                      notes="grounded boundary at the same radius, for contrast")
        rows.append({"domain_radius_m": Rb, "C_open_F": e["capacitance_F"][0][0],
                     "relative_error_open": e["capacitance_F"][0][0] / (FPE * a) - 1,
                     "C_grounded_F": eg["capacitance_F"][0][0],
                     "C_grounded_exact_F": FPE / (1 / a - 1 / Rb),
                     "relative_error_grounded": eg["capacitance_F"][0][0] / (FPE / (1 / a - 1 / Rb)) - 1,
                     "exterior_flux_check": e["checks"]["exterior_monopole_max_relative_difference"],
                     "dofs": e["solver"]["dofs"], "warnings": w})
    Co = [r["C_open_F"] for r in rows]
    spread = (max(Co) - min(Co)) / (FPE * a)
    gspread = (max(r["C_grounded_F"] for r in rows) - min(r["C_grounded_F"] for r in rows)) / (FPE * a)
    return record("exterior", {"a_m": a, "rows": rows, "open_spread_relative": spread,
                               "grounded_spread_relative": gspread,
                               "pass": spread < 2e-3 and gspread > 0.3,
                               "headline": f"open: C varies by {spread:.2e} of 4 pi eps0 a over "
                                           f"R_b = 1.0-2.5 m (grounded: {gspread:.2f})"})


# ---------------------------------------------------------------- 2. concentric spheres
def check_concentric() -> dict:
    a, b, c = 0.3, 0.5, 0.6
    e, w = solve("concentric",
                 [{"name": "inner", "solid": ball(a), "material": "superconductor"},
                  {"name": "outer", "solid": hollow(b, c), "material": "superconductor"}],
                 mesh={"surface_size": 0.04, "size": 0.12, "domain_radius": 1.6},
                 notes="concentric spheres: C11 = 4 pi eps0/(1/a - 1/b), C22 adds 4 pi eps0 c")
    i, o = named(e, "inner"), named(e, "outer")
    C = e["capacitance_F"]
    Cin = FPE / (1 / a - 1 / b)
    ref = {"C_inner_inner": Cin, "C_inner_outer": -Cin, "C_outer_outer": Cin + FPE * c}
    got = {"C_inner_inner": C[i][i], "C_inner_outer": C[i][o], "C_outer_outer": C[o][o]}
    err = {k: got[k] / ref[k] - 1 for k in ref}
    worst = max(abs(v) for v in err.values())
    return record("concentric", {"a_m": a, "b_m": b, "c_m": c, "computed_F": got, "exact_F": ref,
                                 "relative_error": err, "pairs": e["pairs"], "checks": checks_summary(e),
                                 "warnings": w, "pass": worst < 5e-3,
                                 "headline": f"C11 {got['C_inner_inner']:.6e}, C12 {got['C_inner_outer']:.6e}, "
                                             f"C22 {got['C_outer_outer']:.6e} F; worst rel err {worst:+.2e}"})


# ---------------------------------------------------------------- 3. layered dielectric
def check_dielectric() -> dict:
    a, b, eps_r = 0.3, 0.5, 4.0
    e, w = solve("dielectric",
                 [{"name": "ball", "solid": ball(a), "material": "superconductor"},
                  {"name": "diel", "solid": hollow(a, b), "material": "dielectric"}],
                 materials={"dielectric": {"kind": "linear", "eps_r": eps_r}},
                 mesh={"surface_size": 0.04, "size": 0.12, "domain_radius": 1.6},
                 notes="layered spherical capacitor: C = 4 pi eps0/[(1/a - 1/b)/eps_r + 1/b]")
    C = e["capacitance_F"][0][0]
    exact = FPE / ((1 / a - 1 / b) / eps_r + 1 / b)
    err = C / exact - 1
    return record("dielectric", {"a_m": a, "b_m": b, "eps_r": eps_r, "C_F": C, "C_exact_F": exact,
                                 "relative_error": err, "checks": checks_summary(e), "warnings": w,
                                 "pass": abs(err) < 5e-3,
                                 "headline": f"C = {C:.8e} F (exact {exact:.8e}), rel err {err:+.2e}"})


# ---------------------------------------------------------------- 4. grounded outer sphere
def check_grounded() -> dict:
    a, Rb = 0.3, 1.0
    e, w = solve("grounded", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
                 boundary="grounded", mesh={"surface_size": 0.04, "size": 0.12, "domain_radius": Rb},
                 notes="sphere inside a grounded sphere: C = 4 pi eps0/(1/a - 1/Rb)")
    C = e["capacitance_F"][0][0]
    exact = FPE / (1 / a - 1 / Rb)
    err = C / exact - 1
    return record("grounded", {"a_m": a, "domain_radius_m": Rb, "C_F": C, "C_exact_F": exact,
                               "relative_error": err, "checks": checks_summary(e), "warnings": w,
                               "pass": abs(err) < 5e-3,
                               "headline": f"C = {C:.8e} F (exact {exact:.8e}), rel err {err:+.2e}"})


# ---------------------------------------------------------------- 5. two spheres on the z axis
def _two_sphere_case(name, a, D, centers, notes):
    e, w = solve(name,
                 [{"name": "s1", "solid": ball(a, centers[0]), "material": "superconductor"},
                  {"name": "s2", "solid": ball(a, centers[1]), "material": "superconductor"}],
                 mesh={"surface_size": 0.016, "size": 0.09, "domain_radius": 1.2}, notes=notes)
    i, j = named(e, "s1"), named(e, "s2")
    C = e["capacitance_F"]
    c11, c12 = bispherical(a, D)
    got = {"C_11": C[i][i], "C_22": C[j][j], "C_12": C[i][j]}
    err = {"C_11": got["C_11"] / c11 - 1, "C_22": got["C_22"] / c11 - 1, "C_12": got["C_12"] / c12 - 1}
    worst = max(abs(v) for v in err.values())
    return record(name, {"a_m": a, "centre_distance_m": D, "centers_m": [list(c) for c in centers],
                         "computed_F": got, "exact_F": {"C_11": c11, "C_12": c12},
                         "relative_error": err, "C11_over_C22_minus_1": got["C_11"] / got["C_22"] - 1,
                         "pairs": e["pairs"], "checks": checks_summary(e), "warnings": w,
                         "pass": worst < 1e-2,
                         "headline": f"C11 {got['C_11']:.6e} (exact {c11:.6e}), C12 {got['C_12']:.6e} "
                                     f"(exact {c12:.6e}); worst rel err {worst:+.2e}"})


def check_two_spheres() -> dict:
    a, D = 0.1, 0.5
    return _two_sphere_case("two_spheres", a, D, [(0, 0, -D / 2), (0, 0, D / 2)],
                            "two equal spheres on the z axis (an axisymmetric configuration solved in 3D)")


# ---------------------------------------------------------------- 6. two offset spheres (non-axisymmetric)
def check_offset() -> dict:
    """The same pair, rotated off every coordinate axis and displaced from the origin: the bispherical
    series depends only on (a, D), so the reference is unchanged while the configuration has no symmetry
    axis through the domain centre."""
    a, D = 0.1, 0.5
    u = [1.0 / math.sqrt(3.0)] * 3
    o = (0.13, -0.09, 0.07)
    c1 = tuple(o[k] - 0.5 * D * u[k] for k in range(3))
    c2 = tuple(o[k] + 0.5 * D * u[k] for k in range(3))
    return _two_sphere_case("offset", a, D, [c1, c2],
                            "two equal spheres along (1,1,1)/sqrt(3), centre of the pair off the origin: "
                            "genuinely non-axisymmetric, same exact bispherical reference")


# ---------------------------------------------------------------- 7. cube (published value)
def check_cube() -> dict:
    L = 0.4
    out = {}
    for h in (0.04, 0.028, 0.02):
        e, w = solve("cube", [{"name": "cube", "solid": box(L), "material": "superconductor"}],
                     mesh={"surface_size": h, "size": 4 * h, "domain_radius": 1.2},
                     wd=RUNS / f"cube_{h}",
                     notes="unit cube: C = 0.66067813 * 4 pi eps0 * side (published)")
        C = e["capacitance_F"][0][0]
        out[str(h)] = {"surface_size_m": h, "C_F": C, "C_over_4pi_eps0_L": C / (FPE * L),
                       "relative_error": C / (CUBE_FACTOR * FPE * L) - 1, "dofs": e["solver"]["dofs"],
                       "checks": checks_summary(e), "warnings": w}
    fine = out[str(0.02)]
    err = fine["relative_error"]
    return record("cube", {"side_m": L, "published_factor": CUBE_FACTOR, "mesh_series": out,
                           "relative_error": err, "pass": abs(err) < 5e-3,
                           "headline": f"C/(4 pi eps0 L) = {fine['C_over_4pi_eps0_L']:.7f} "
                                       f"(published {CUBE_FACTOR}), rel err {err:+.2e}"})


# ---------------------------------------------------------------- 8. h convergence
def check_convergence() -> dict:
    a, b, c = 0.3, 0.5, 0.6
    Cin = FPE / (1 / a - 1 / b)
    rows = []
    for h in (0.09, 0.06, 0.04):
        for order in (1, 2):
            e, w = solve("concentric", [{"name": "inner", "solid": ball(a), "material": "superconductor"},
                                        {"name": "outer", "solid": hollow(b, c), "material": "superconductor"}],
                         mesh={"order": order, "surface_size": h, "size": 2.5 * h, "domain_radius": 1.6},
                         wd=RUNS / f"conv_{order}_{h}", notes="h/p convergence of the concentric capacitor")
            i = named(e, "inner")
            C = e["capacitance_F"][i][i]
            ch = e["checks"]
            rows.append({"order": order, "surface_size_m": h, "dofs": e["solver"]["dofs"], "C_inner_F": C,
                         "relative_error": C / Cin - 1,
                         "gauss": ch["charge_gauss_max_relative_difference"],
                         "monopole": ch["exterior_monopole_max_relative_difference"],
                         "energy": ch["energy_max_relative_difference"]})
            print(f"      order {order} h {h}: dofs {e['solver']['dofs']:7d} C {C:.8e} "
                  f"rel {C / Cin - 1:+.2e} gauss {ch['charge_gauss_max_relative_difference']:.1e}", flush=True)
    fine = [r for r in rows if r["order"] == 2][-1]
    coarse = [r for r in rows if r["order"] == 1][0]
    monotone = abs(fine["relative_error"]) < abs(coarse["relative_error"])
    return record("convergence", {"a_m": a, "b_m": b, "c_m": c, "C_exact_F": Cin, "rows": rows,
                                  "pass": monotone and abs(fine["relative_error"]) < 2e-3,
                                  "headline": f"order {coarse['order']} h {coarse['surface_size_m']}: "
                                              f"{coarse['relative_error']:+.2e} -> order {fine['order']} "
                                              f"h {fine['surface_size_m']}: {fine['relative_error']:+.2e}"})


# ---------------------------------------------------------------- 9. 3D vs the axisymmetric engine
def check_vs_2d() -> dict:
    """A revolved axisymmetric design (axionfem.convert.to_3d): the same capacitance matrix."""
    def disk(rho, zc=0.0):
        return {"disk": {"center": [0.0, zc], "radius": rho}}

    doc = {"schema": "axionfem.design/1", "name": "ring_can",
           "materials": {"support": {"kind": "linear", "eps_r": 3.0}},
           "regions": [
               {"name": "can", "shape": {"difference": [{"rect": {"r": [0.0, 0.5], "z": [-0.5, 0.5]}},
                                                        {"rect": {"r": [0.0, 0.45], "z": [-0.45, 0.45]}}]},
                "material": "superconductor"},
               {"name": "ring", "shape": {"rect": {"r": [0.25, 0.3], "z": [-0.03, 0.03]}},
                "material": "superconductor"},
               {"name": "support", "shape": {"rect": {"r": [0.3, 0.45], "z": [-0.03, 0.03]}},
                "material": "support"}],
           "electrostatics": {"boundary": "open", "notes": "pickup ring in a shield can with a dielectric support"},
           "mesh": {"order": 3, "size": 0.05, "surface_size": 0.012, "domain_radius": 1.5}}
    d2 = af.Design(doc)
    r2 = d2.solve(RUNS / "vs2d_axi", verbose=0)
    e2 = r2.data["electrostatics"]
    d3 = to_3d(d2, mesh={"order": 2, "size": 0.1, "surface_size": 0.04, "domain_radius": 1.5})
    r3 = d3.solve(RUNS / "vs2d_3d", verbose=0)
    e3 = r3.data["electrostatics"]
    rows, worst = [], 0.0
    for nm in ("can", "ring"):
        for nm2 in ("can", "ring"):
            a2 = e2["capacitance_F"][named(e2, nm)][named(e2, nm2)]
            a3 = e3["capacitance_F"][named(e3, nm)][named(e3, nm2)]
            rel = a3 / a2 - 1 if a2 != 0 else 0.0
            scale = max(abs(e2["capacitance_F"][named(e2, "can")][named(e2, "can")]), 1e-300)
            rel_scaled = (a3 - a2) / scale
            rows.append({"i": nm, "j": nm2, "axi_F": a2, "three_d_F": a3, "relative_difference": rel,
                         "difference_over_C_can_can": rel_scaled})
            worst = max(worst, abs(rel_scaled))
    return record("vs_2d", {"rows": rows, "worst_difference_over_C_max": worst,
                            "axi": {"dofs": e2["solver"]["dofs"], "checks": checks_summary(e2)
                                    if "mixed_potential_solve" in e2["checks"] else None},
                            "three_d": {"dofs": e3["solver"]["dofs"], "checks": checks_summary(e3)},
                            "warnings_3d": r3.data.get("warnings", []),
                            "pass": worst < 2e-2,
                            "headline": f"largest 3D-vs-2D matrix difference {worst:.2e} of C(can, can)"})


# ---------------------------------------------------------------- 10. rank independence
def check_ranks() -> dict:
    a, b, c = 0.3, 0.5, 0.6
    regions = [{"name": "inner", "solid": ball(a), "material": "superconductor"},
               {"name": "outer", "solid": hollow(b, c), "material": "superconductor"}]
    want = [int(n) for n in os.environ.get("AXF_RANKS", "1,4,8").split(",")]
    res = {}
    for n in want:
        e, w = solve("ranks", regions, mesh={"surface_size": 0.04, "size": 0.12, "domain_radius": 1.6},
                     ranks=n, wd=RUNS / f"ranks_{n}", notes="rank independence of the capacitance matrix")
        res[n] = e
    ref = res[want[0]]
    scale = max(abs(x) for row in ref["capacitance_F"] for x in row)
    rows, worst = [], 0.0
    for n in want[1:]:
        d = max(abs(res[n]["capacitance_F"][i][j] - ref["capacitance_F"][i][j])
                for i in range(len(ref["capacitance_F"])) for j in range(len(ref["capacitance_F"])))
        rows.append({"ranks": n, "max_absolute_difference_F": d, "relative": d / scale,
                     "mpi_ranks_reported": res[n]["solver"]["mpi_ranks"]})
        worst = max(worst, d / scale)
    return record("ranks", {"ranks": want, "capacitance_F": {str(n): res[n]["capacitance_F"] for n in want},
                            "rows": rows, "worst_relative_difference": worst, "pass": worst < 1e-8,
                            "headline": f"ranks {want}: largest relative difference of the matrix {worst:.2e}"})


# ---------------------------------------------------------------- 11. readout-port gap capacitance
def check_port() -> dict:
    """A split superconducting ring with a vacuum port filling the gap: the gap capacitance and the
    self-resonance bound, compared with the parallel-plate estimate eps0 A / g of the gap."""
    R, w_, t, g = 0.3, 0.05, 0.05, 0.02
    ang = math.degrees(g / R)  # opening angle of the gap
    ring = {"revolve": {"profile": {"rect": {"r": [R - w_ / 2, R + w_ / 2], "z": [-t / 2, t / 2]}},
                        "angles": [ang / 2, 360 - ang / 2]}}
    port = {"revolve": {"profile": {"rect": {"r": [R - w_ / 2, R + w_ / 2], "z": [-t / 2, t / 2]}},
                        "angles": [-ang / 2, ang / 2]}}
    doc = {"schema": "axionfem.design/1", "name": "port_gap", "symmetry": "3d",
           "regions": [{"name": "coil", "solid": {"revolve": {"profile": {"rect": {"r": [0.6, 0.65],
                                                                                  "z": [-0.1, 0.1]}}}},
                        "material": "winding", "mesh_size": 0.05},
                       {"name": "ring", "solid": ring, "material": "superconductor", "mesh_size": 0.02},
                       {"name": "gap", "solid": port, "material": "vacuum", "mesh_size": 0.008}],
           "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
           "pickup": {"type": "port", "port": "gap"},
           "electrostatics": {},  # 3D: the readout-gap analysis runs with an electrostatics section
           "mesh": {"order": 2, "size": 0.15, "surface_size": 0.035, "domain_radius": 1.6}}
    r = af.Design(doc).solve(RUNS / "port_gap", verbose=0)
    pc = r.data["pickup"]["port_capacitance"]
    plate = EPS0 * (w_ * t) / g
    ratio_region = pc["C_gap_port_region_F"] / plate
    ok = (0.5 < ratio_region < 1.5 and pc["C_gap_F"] >= pc["C_gap_port_region_F"]
          and pc["terminals_on_one_body"])
    return record("port", {"ring_R_m": R, "cross_section_m": [w_, t], "gap_m": g,
                           "parallel_plate_F": plate, "port_capacitance": pc, "L_p_H": r.L_p,
                           "region_over_parallel_plate": ratio_region,
                           "warnings": r.data.get("warnings", []), "pass": bool(ok),
                           "headline": f"C_gap = {pc['C_gap_F']:.4e} F (region only "
                                       f"{pc['C_gap_port_region_F']:.4e} F, parallel plate {plate:.4e} F), "
                                       f"f_SR = {pc.get('self_resonance_Hz', 0):.4e} Hz"})


CHECKS = {"sphere": check_sphere, "exterior": check_exterior, "concentric": check_concentric,
          "dielectric": check_dielectric, "grounded": check_grounded, "two_spheres": check_two_spheres,
          "offset": check_offset, "cube": check_cube, "convergence": check_convergence,
          "vs_2d": check_vs_2d, "ranks": check_ranks, "port": check_port}


def main(argv):
    names = list(CHECKS) if not argv or argv[0] == "all" else argv
    bad = [n for n in names if n not in CHECKS]
    if bad:
        raise SystemExit(f"unknown check(s) {bad}; available: {list(CHECKS)}")
    results = {}
    for n in names:
        t0 = time.time()
        try:
            results[n] = CHECKS[n]()
        except Exception as exc:  # a failing case must not hide the others
            results[n] = record(n, {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                                    "headline": f"ERROR {type(exc).__name__}: {exc}"})
        print(f"      ({time.time() - t0:.1f} s)", flush=True)
    failed = [n for n, r in results.items() if not r.get("pass")]
    print(f"\n{len(names) - len(failed)}/{len(names)} checks passed"
          + (f"; failed: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
