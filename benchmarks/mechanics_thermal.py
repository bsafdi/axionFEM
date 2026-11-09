"""Validation of the axisymmetric elasticity and steady-conduction solvers.

    python benchmarks/mechanics_thermal.py [workdir]

Writes benchmarks/records/mechanics_{lame,solenoid,sc_pressure,cooldown,templates}.json,
benchmarks/records/thermal_{uniform,kT}.json and docs/validation/mechanics_thermal.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records"
OUT = ROOT / "docs" / "validation" / "mechanics_thermal.md"
MU0 = 1.25663706212e-6
STAMP = {"axionfem_version": af.__version__ if hasattr(af, "__version__") else "0.1.0",
         "date": time.strftime("%Y-%m-%d %H:%M")}
# AXF_MT_QUICK=1 keeps only the coarsest mesh of every sweep: a smoke test of all code paths.
QUICK = bool(os.environ.get("AXF_MT_QUICK"))


def sweep(values):
    return values[:1] if QUICK else values

# ---------------------------------------------------------------- Lamé thick-walled cylinder
E_L, NU_L = 200e9, 0.3
A_L, B_L, H_L, P_L = 0.1, 0.3, 4.0, 1.0e6


def lame_design(fix_z, size, order):
    doc = {"schema": "axionfem.design/1", "name": "lame",
           "materials": {"steel": {"kind": "linear", "mechanical": {"E_Pa": E_L, "nu": NU_L, "alpha": 0.0}}},
           "regions": [
               {"name": "bore", "shape": {"rect": {"r": [0.0, A_L], "z": [-H_L / 2, H_L / 2]}}, "material": "vacuum"},
               {"name": "cyl", "shape": {"rect": {"r": [A_L, B_L], "z": [-H_L / 2, H_L / 2]}}, "material": "steel",
                "mesh_size": size},
               {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"pressures": [{"on": "cyl", "from": "bore", "p_Pa": P_L}]},
           "mesh": {"order": order, "size": 0.3, "domain_radius": 12.0}}
    if fix_z:
        doc["mechanics"]["supports"] = [{"region": "cyl", "fix": "z"}]
    return af.Design(doc)


def lame_exact(r, plane_strain):
    k = P_L * A_L ** 2 / (B_L ** 2 - A_L ** 2)
    s_rr = k * (1 - B_L ** 2 / r ** 2)
    s_tt = k * (1 + B_L ** 2 / r ** 2)
    if plane_strain:
        return (1 + NU_L) / E_L * k * ((1 - 2 * NU_L) * r + B_L ** 2 / r), s_rr, s_tt, NU_L * (s_rr + s_tt)
    return k / E_L * ((1 - NU_L) * r + (1 + NU_L) * B_L ** 2 / r), s_rr, s_tt, np.zeros_like(r)


def lame(wd: Path):
    w = B_L - A_L
    r = np.linspace(A_L + 0.1 * w, B_L - 0.1 * w, 21)
    rows = []
    for fix_z in (True, False):
        ur_e, srr_e, stt_e, szz_e = lame_exact(r, fix_z)
        for order in sweep((1, 2)):
            for size in sweep((0.08, 0.04, 0.02, 0.01)):
                d = lame_design(fix_z, size, order)
                res = d.solve(wd / f"lame_{fix_z}_{order}_{size}", verbose=0)
                s = res.sample_mech(r, np.zeros_like(r))
                ck = res.mechanics["checks"]
                rows.append({
                    "ends": "fixed (plane strain)" if fix_z else "free",
                    "order": order, "mesh_size": size,
                    "dofs": res.data["mesh"]["dofs"],
                    "solid_elements": res.mechanics["solver"]["solid_elements"],
                    "err_u_r": float(np.max(np.abs(s["u_r"] / ur_e - 1))),
                    "err_hoop": float(np.max(np.abs(s["sigma_hoop"] / stt_e - 1))),
                    "err_sigma_rr_over_p": float(np.max(np.abs(s["sigma_rr"] - srr_e)) / P_L),
                    "err_sigma_zz_over_p": float(np.max(np.abs(s["sigma_zz"] - szz_e)) / P_L),
                    "equilibrium_Fz_round_off": ck["equilibrium_Fz_round_off"],
                    "applied_radial_N_per_rad": ck["applied_radial_N_per_rad"],
                    "applied_radial_exact": P_L * A_L * H_L,
                    "seconds": res.data["timing"]["mechanics_thermal_s"]})
    # A halved h whose mesh does not actually refine (fewer than 4x the elements) would show up as an apparent
    # error floor that is not a method limit.  Record the element ratio of each step so a stalled ladder is
    # visible in the record and in the report.
    for i, x in enumerate(rows):
        prev = rows[i - 1] if i > 0 and rows[i - 1]["ends"] == x["ends"] \
            and rows[i - 1]["order"] == x["order"] else None
        x["element_ratio_vs_previous"] = (x["solid_elements"] / prev["solid_elements"]) if prev else None
        x["is_a_real_refinement"] = (x["element_ratio_vs_previous"] > 2.0) if prev else None
    out = dict(STAMP, check="mechanics_lame",
               geometry={"a_m": A_L, "b_m": B_L, "length_m": H_L, "p_Pa": P_L, "E_Pa": E_L, "nu": NU_L},
               reference="Lamé: sigma_rr/tt = p a^2/(b^2-a^2) (1 -/+ b^2/r^2); plane strain u_r = "
                         "(1+nu)/E k ((1-2nu) r + b^2/r), sigma_zz = nu (s_rr + s_tt); free ends sigma_zz = 0, "
                         "u_r = k/E ((1-nu) r + (1+nu) b^2/r)",
               rows=rows)
    (REC / "mechanics_lame.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- solenoid winding hoop stress
def solenoid(wd: Path):
    R, Hs, NI = 0.5, 10.0, 1.0e6
    rows = []
    for t in sweep((0.02, 0.01, 0.005, 0.0025)):
        doc = {"schema": "axionfem.design/1", "name": "sol",
               "materials": {"coil": {"kind": "vacuum", "mechanical": {"E_Pa": 100e9, "nu": 0.3}}},
               "regions": [{"name": "w", "shape": {"rect": {"r": [R, R + t], "z": [-Hs / 2, Hs / 2]}},
                            "material": "coil", "mesh_size": t}],
               "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
               "mechanics": {},
               "mesh": {"order": 2, "size": 0.5, "domain_radius": 30.0}}
        res = af.Design(doc).solve(wd / f"sol_{t}", verbose=0)
        f = res.sample([R - 1e-5, R + t + 1e-5], [0.0, 0.0])
        p = (f["Bz"][0] ** 2 - f["Bz"][1] ** 2) / (2 * MU0)
        rm = R + t / 2
        s = res.sample_mech(np.linspace(R + 0.1 * t, R + 0.9 * t, 9), np.zeros(9))
        hoop, szz = float(np.mean(s["sigma_hoop"])), float(np.mean(s["sigma_zz"]))
        ck = res.mechanics["checks"]
        # Exact thickness-mean hoop stress for this load, not a membrane approximation: radial
        # equilibrium with f_r = J B_z(r) and the linear field profile B_z = B0 (b - r)/t gives
        #   (1/t) int_a^b sigma_tt dr = (B0^2/mu0)(Rm/2 - t/12)/t = p (Rm - t/6)/t.
        hoop_exact = p * (rm - t / 6) / t
        rows.append({"t_over_R": t / R, "t_m": t, "B_in_T": abs(float(f["Bz"][0])),
                     "B_out_T": abs(float(f["Bz"][1])), "p_Pa": p,
                     "hoop_Pa": hoop,
                     "hoop_exact_distributed_Pa": hoop_exact,
                     "hoop_exact_rel": hoop / hoop_exact - 1,
                     "hoop_thin_shell_Pa": p * rm / t, "hoop_rel": hoop / (p * rm / t) - 1,
                     "sigma_zz_Pa": szz, "sigma_zz_ref_Pa": -p * R ** 2 / (2 * rm * t),
                     "sigma_zz_rel": szz / (-p * R ** 2 / (2 * rm * t)) - 1,
                     "lorentz_radial_N_per_rad": ck["applied_radial_N_per_rad"],
                     "force_postprocessing_N_per_rad": res.data["forces"]["w"]["radial_force_per_radian_N"],
                     "equilibrium_Fz_round_off": ck["equilibrium_Fz_round_off"]})
    out = dict(STAMP, check="mechanics_solenoid_hoop",
               geometry={"R_m": R, "H_m": Hs, "ampere_turns": NI},
               reference="hoop: EXACT thickness mean for this load, p (Rm - t/6)/t, from radial equilibrium "
                         "with f_r = J B_z(r) and B_z = B0 (b - r)/t (p = (B_in^2 - B_out^2)/2 mu0); the "
                         "membrane value p Rm/t is kept as the leading-order comparison. sigma_zz: the two "
                         "halves attract with p pi R^2, so sigma_zz = -p R^2/(2 Rm t) -- a MEMBRANE value "
                         "whose O(t/R) error the table shows (it halves as t halves). It is not replaced by "
                         "an axial-resultant identity, because the area integral of the recovered sigma_zz is "
                         "dominated by the stress recovery right at the two loaded surfaces; and the residual "
                         "is not an end effect: it is 1.79e-2 at H/R = 20, 40 and 80 for t/R = 0.02.",
               rows=rows)
    (REC / "mechanics_solenoid.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- superconductor surface pressure
def sc_pressure(wd: Path):
    a, Ls, Hs, NI = 0.5, 8.0, 12.0, 1.0e6
    rows = []
    for t in sweep((0.02, 0.01, 0.005)):
        doc = {"schema": "axionfem.design/1", "name": "shell",
               "materials": {"nbm": {"kind": "superconductor", "mechanical": {"E_Pa": 105e9, "nu": 0.4}}},
               "regions": [{"name": "sh", "shape": {"rect": {"r": [a, a + t], "z": [-Ls / 2, Ls / 2]}},
                            "material": "nbm", "mesh_size": t},
                           {"name": "w", "shape": {"rect": {"r": [0.8, 0.81], "z": [-Hs / 2, Hs / 2]}},
                            "material": "winding"}],
               "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
               "superconductors": [{"body": "sh", "azimuthal": {"flux": 0.0}}],
               "mechanics": {},
               "mesh": {"order": 2, "size": 0.5, "surface_size": t, "domain_radius": 40.0}}
        res = af.Design(doc).solve(wd / f"sc_{t}", verbose=0)
        f = res.sample([a - 1e-5, a + t + 1e-5], [0.0, 0.0])
        p = (f["Bz"][1] ** 2 - f["Bz"][0] ** 2) / (2 * MU0)
        rm = a + t / 2
        b = a + t
        rr = np.linspace(a + 0.1 * t, a + 0.9 * t, 9)
        s = res.sample_mech(rr, np.zeros(9))
        hoop = float(np.mean(s["sigma_hoop"]))
        ck = res.mechanics["checks"]
        # EXACT thick-wall Lame, external pressure p on r = b with free ends: k = -p b^2/(b^2 - a^2),
        # sigma_tt = k (1 + a^2/r^2), sigma_rr = k (1 - a^2/r^2).  Its thickness mean is -p b/t exactly
        # (int_a^b (1 + a^2/r^2) dr = (b - a)(1 + a/b) = t (a + b)/b), i.e. the membrane value with the
        # OUTER radius, not Rm -- which is why the Rm form was 0.5-2 % off and looked like a method error.
        k = -p * b ** 2 / (b ** 2 - a ** 2)
        stt_e = k * (1 + a ** 2 / rr ** 2)
        srr_e = k * (1 - a ** 2 / rr ** 2)
        rows.append({"t_over_R": t / a, "t_m": t, "B_bore_T": abs(float(f["Bz"][0])),
                     "B_outside_T": abs(float(f["Bz"][1])), "p_Pa": p,
                     "err_hoop_exact_thick_wall":
                         float(np.max(np.abs(np.asarray(s["sigma_hoop"]) / stt_e - 1))),
                     "err_sigma_rr_over_p":
                         float(np.max(np.abs(np.asarray(s["sigma_rr"]) - srr_e)) / p),
                     "hoop_exact_mean_Pa": -p * b / t,
                     "hoop_exact_mean_rel": hoop / (-p * b / t) - 1,
                     "hoop_Pa": hoop, "hoop_thin_shell_Pa": -p * rm / t, "hoop_rel": hoop / (-p * rm / t) - 1,
                     "applied_Fz_N": ck["applied_Fz_N"],
                     "force_postprocessing_Fz_N": res.data["forces"]["superconductor:sh"]["Fz_N"],
                     "Fz_rel": ck["applied_Fz_N"] / res.data["forces"]["superconductor:sh"]["Fz_N"] - 1,
                     "equilibrium_Fz_round_off": ck["equilibrium_Fz_round_off"]})
    out = dict(STAMP, check="mechanics_sc_pressure",
               geometry={"a_m": a, "shell_length_m": Ls, "solenoid_length_m": Hs, "ampere_turns": NI},
               reference="zero-field-cooled tube inside a solenoid: external Maxwell pressure p = B^2/2 mu0. "
                         "EXACT thick-wall Lame with free ends, k = -p b^2/(b^2 - a^2): sigma_tt = k(1 + a^2/r^2), "
                         "sigma_rr = k(1 - a^2/r^2), thickness mean -p b/t exactly. The membrane form -p Rm/t is "
                         "kept as the leading-order comparison. The axial load is cross-checked against the "
                         "independent surface-force post-processing in results['forces']",
               rows=rows)
    (REC / "mechanics_sc_pressure.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------- zero-thickness superconducting sheet on a structure
def sc_sheet(wd: Path):
    """A steel tube with a zero-thickness superconducting skin on its outer surface, inside a solenoid.

    The sheet carries the screening current, so its Maxwell surface traction acts
    on the tube it is bonded to; without that load the tube would report zero hoop stress.  Reference: the exact thick-wall Lamé solution for the external pressure p = B_out^2/2 mu0.
    """
    a, Ls, Hs, NI = 0.5, 8.0, 12.0, 1.0e6
    rows = []
    for t in sweep((0.02, 0.01, 0.005)):
        b = a + t
        doc = {"schema": "axionfem.design/1", "name": "skin",
               "materials": {"st": {"kind": "linear", "mechanical": {"E_Pa": 200e9, "nu": 0.3}}},
               "regions": [{"name": "tube", "shape": {"rect": {"r": [a, b], "z": [-Ls / 2, Ls / 2]}},
                            "material": "st", "mesh_size": t},
                           {"name": "w", "shape": {"rect": {"r": [0.8, 0.81], "z": [-Hs / 2, Hs / 2]}},
                            "material": "winding"}],
               "curves": [{"name": "skin", "path": {"segment": [[b, -Ls / 2], [b, Ls / 2]]},
                           "material": "nb"}],
               "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
               "superconductors": [{"body": "skin", "azimuthal": {"flux": 0.0}}],
               "mechanics": {},
               "mesh": {"order": 2, "size": 0.5, "surface_size": t, "domain_radius": 40.0}}
        res = af.Design(doc).solve(wd / f"skin_{t}", verbose=0)
        f = res.sample([a + 0.5 * t, b + 0.01], [0.0, 0.0])
        B_in, B_out = float(f["Bz"][0]), float(f["Bz"][1])
        p = (B_out ** 2 - B_in ** 2) / (2 * MU0)
        rr = np.linspace(a + 0.1 * t, b - 0.1 * t, 9)
        sm = res.sample_mech(rr, np.zeros_like(rr))
        hoop = float(np.mean(sm["sigma_hoop"]))
        k = -p * b ** 2 / (b ** 2 - a ** 2)
        stt_e = k * (1 + a ** 2 / rr ** 2)
        srr_e = k * (1 - a ** 2 / rr ** 2)
        ck = res.mechanics["checks"]
        rows.append({"t_over_R": t / a, "t_m": t, "B_screened_T": abs(B_in), "B_outside_T": abs(B_out),
                     "p_Pa": p, "sheets_loaded": res.mechanics["magnetic_pressure_sheets"],
                     "err_hoop_exact_thick_wall":
                         float(np.max(np.abs(np.asarray(sm["sigma_hoop"]) / stt_e - 1))),
                     "err_sigma_rr_over_p":
                         float(np.max(np.abs(np.asarray(sm["sigma_rr"]) - srr_e)) / p),
                     "hoop_Pa": hoop, "hoop_exact_mean_Pa": -p * b / t,
                     "hoop_exact_mean_rel": hoop / (-p * b / t) - 1,
                     "maxwell_traction_Bn_over_B_max": ck["maxwell_traction_Bn_over_B_max"],
                     "applied_radial_N_per_rad": ck["applied_radial_N_per_rad"]})
    out = dict(STAMP, check="mechanics_sc_sheet",
               geometry={"tube_r_in_m": a, "tube_length_m": Ls, "solenoid_length_m": Hs,
                         "ampere_turns": NI, "sheet_at": "outer surface of the tube"},
               reference="the sheet screens the tube, so the tube carries the external pressure "
                         "p = (B_out^2 - B_in^2)/2 mu0; exact thick-wall Lamé with free ends, "
                         "k = -p b^2/(b^2 - a^2), thickness mean -p b/t",
               rows=rows)
    (REC / "mechanics_sc_sheet.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- free cooldown
def cooldown(wd: Path):
    alpha = 2.5e-3
    doc = {"schema": "axionfem.design/1", "name": "ring",
           "materials": {"m": {"kind": "linear", "mechanical": {"E_Pa": 120e9, "nu": 0.33, "alpha": alpha}}},
           "regions": [{"name": "ring", "shape": {"rect": {"r": [0.4, 0.6], "z": [-0.1, 0.1]}},
                        "material": "m", "mesh_size": 0.02},
                       {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"cooldown": True},
           "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}
    res = af.Design(doc).solve(wd / "ring", verbose=0)
    r = np.array([0.41, 0.45, 0.5, 0.55, 0.59, 0.47, 0.53])
    z = np.array([0.01, 0.03, 0.05, -0.08, 0.09, -0.02, 0.07])
    s = res.sample_mech(r, z)
    reg = res.mechanics["regions"]["ring"]
    out = dict(STAMP, check="mechanics_cooldown",
               geometry={"r_m": [0.4, 0.6], "z_m": [-0.1, 0.1], "E_Pa": 120e9, "nu": 0.33, "alpha": alpha},
               reference="a free body contracting uniformly: u = -alpha x, zero stress",
               max_rel_err_u_r=float(np.max(np.abs(s["u_r"] / (-alpha * r) - 1))),
               max_rel_err_u_z=float(np.max(np.abs(s["u_z"] / (-alpha * z) - 1))),
               von_mises_max_Pa=reg["von_mises_max_Pa"],
               von_mises_over_E_alpha=reg["von_mises_max_Pa"] / (120e9 * alpha),
               u_max_m=reg["u_max_m"], u_max_exact_m=alpha * math.hypot(0.6, 0.1),
               equilibrium_Fz_round_off=res.mechanics["checks"]["equilibrium_Fz_round_off"])
    (REC / "mechanics_cooldown.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- templates
def templates(wd: Path):
    rows = {}
    for name in sweep(("toroid", "solenoid", "core")):
        d = af.template(name)
        d.doc["mechanics"] = {"cooldown": True}
        res = d.solve(wd / f"tpl_{name}", verbose=0)
        ck = res.mechanics["checks"]
        # "distributions" and "closure" are reserved keys of results["forces"] and are
        # not bodies, so they have no Fz_N.
        f = {k: v for k, v in res.data["forces"].items() if k not in ("distributions", "closure")}
        # results["forces"] gives F_z for windings and superconductors but only the radial resultant of
        # windings, so the axial sum is the complete cross-check and the radial one covers windings only.
        fz_sum = sum(v["Fz_N"] for v in f.values())
        fr_wind = sum(v["radial_force_per_radian_N"] for k, v in f.items()
                      if not k.startswith("superconductor:"))
        rows[name] = {
            "Psi_SI": res.Psi, "L_p_H": res.L_p, "W_mag_J": res.W,
            "lorentz_windings": res.mechanics["lorentz_windings"],
            "magnetic_pressure_bodies": res.mechanics["magnetic_pressure_bodies"],
            "sc_surface_field_T": res.data["fields"]["superconductor_surface_regular_T"],
            "applied_Fz_N": ck["applied_Fz_N"],
            "forces_Fz_sum_N": fz_sum,
            "Fz_residual_N": ck["applied_Fz_N"] - fz_sum,
            "applied_radial_N_per_rad": ck["applied_radial_N_per_rad"],
            "winding_radial_N_per_rad": fr_wind,
            "sc_radial_N_per_rad": ck["applied_radial_N_per_rad"] - fr_wind,
            "equilibrium_Fz_round_off": ck["equilibrium_Fz_round_off"],
            "regions": {k: {"u_max_m": v["u_max_m"], "von_mises_max_Pa": v["von_mises_max_Pa"],
                            "sigma_hoop_Pa": v["sigma_hoop_Pa"], "E_Pa": v["E_Pa"],
                            "thermal_strain": v["thermal_strain"]}
                        for k, v in res.mechanics["regions"].items()},
            "seconds": res.data["timing"]["mechanics_thermal_s"],
            "warnings": res.data["warnings"]}
    out = dict(STAMP, check="mechanics_templates",
               note="default template parameters with cooldown; built-in material defaults (winding = smeared "
                    "isotropic composite, nb = niobium). The Lorentz load is cross-checked against the "
                    "independent winding force in results['forces'].",
               templates=rows)
    (REC / "mechanics_templates.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- thermal: uniform generation
T0, Q, RP, RS, HT = 4.0, 5000.0, 0.4, 0.5, 1.0
VAC = {"kind": "vacuum"}


def thermal_design(mat, size, order, loads=None):
    return af.Design({
        "schema": "axionfem.design/1", "name": "th",
        "materials": {"m": mat, "cm": VAC},
        "regions": [{"name": "core", "shape": {"rect": {"r": [0.0, RP], "z": [-HT / 2, HT / 2]}},
                     "material": "m", "mesh_size": size},
                    {"name": "sink", "shape": {"rect": {"r": [RP, RS], "z": [-HT / 2, HT / 2]}},
                     "material": "m", "mesh_size": size},
                    {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}}, "material": "cm"}],
        "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
        "thermal": {"fixed": [{"region": "sink", "T_K": T0}],
                    "loads": loads if loads is not None else [{"region": "core", "W_m3": Q}]},
        "mesh": {"order": order, "size": 0.3, "domain_radius": 12.0}})


def thermal_uniform(wd: Path):
    k = 12.0
    r = np.linspace(0.0, RP - 1e-9, 41)
    exact = T0 + Q * (RP ** 2 - r ** 2) / (4 * k)
    rows = []
    for order in sweep((1, 2, 3)):
        for size in sweep((0.08, 0.04, 0.02)):
            res = thermal_design({"kind": "linear", "thermal": {"k_W_mK": k}}, size, order).solve(
                wd / f"thu_{order}_{size}", verbose=0)
            s = res.sample_mech(r, np.zeros_like(r))
            ck = res.thermal["checks"]
            rows.append({"order": order, "mesh_size": size, "dofs": res.data["mesh"]["dofs"],
                         "max_abs_err_K": float(np.max(np.abs(s["T_K"] - exact))),
                         "T_max_K": res.thermal["regions"]["core"]["T_max_K"],
                         "T_max_exact_K": T0 + Q * RP ** 2 / (4 * k),
                         "generated_W": ck["generated_W"], "outflow_W": ck["outflow_W"],
                         "balance_relative": ck["balance_relative"],
                         "seconds": res.data["timing"]["mechanics_thermal_s"]})
    out = dict(STAMP, check="thermal_uniform",
               geometry={"R_generating_m": RP, "R_sink_m": RS, "height_m": HT, "q_W_m3": Q, "k_W_mK": k, "T0_K": T0},
               reference="T = T0 + q (R^2 - r^2)/(4 k); total power q pi R^2 h = "
                         f"{Q * math.pi * RP ** 2 * HT:.6f} W",
               rows=rows)
    (REC / "thermal_uniform.json").write_text(json.dumps(out, indent=1))
    return out


def thermal_kT(wd: Path):
    k0, c = 10.0, 0.08
    table = [[T0, k0], [T0 + 30.0, k0 * (1 + c * 30.0)]]
    r = np.linspace(0.0, RP - 1e-9, 41)
    theta = Q * (RP ** 2 - r ** 2) / 4.0
    exact = T0 + (np.sqrt(1 + 2 * c * theta / k0) - 1) / c
    rise = float(exact[0] - T0)
    rows = []
    for size in sweep((0.08, 0.04, 0.02, 0.01)):
        res = thermal_design({"kind": "linear", "thermal": {"k_W_mK": table}}, size, 2).solve(
            wd / f"thk_{size}", verbose=0)
        s = res.sample_mech(r, np.zeros_like(r))
        ck, sol = res.thermal["checks"], res.thermal["solver"]
        rows.append({"mesh_size": size, "dofs": res.data["mesh"]["dofs"],
                     "max_abs_err_K": float(np.max(np.abs(s["T_K"] - exact))),
                     "max_rel_err_of_rise": float(np.max(np.abs(s["T_K"] - exact))) / rise,
                     "T_max_K": float(s["T_K"][0]), "T_max_exact_K": float(exact[0]),
                     "picard_iterations": sol["picard_iterations"], "picard_change": sol["picard_change"],
                     "balance_relative": ck["balance_relative"],
                     "seconds": res.data["timing"]["mechanics_thermal_s"]})
    out = dict(STAMP, check="thermal_kT",
               geometry={"R_generating_m": RP, "R_sink_m": RS, "height_m": HT, "q_W_m3": Q, "T0_K": T0},
               material={"k0_W_mK": k0, "dk_dT_over_k0": c, "table": table},
               reference="Kirchhoff: theta = int_T0^T k dT = q (R^2 - r^2)/4 solves the linear Poisson problem; "
                         "for k = k0 (1 + c (T - T0)), T = T0 + (sqrt(1 + 2 c theta/k0) - 1)/c",
               temperature_rise_K=rise, rows=rows)
    (REC / "thermal_kT.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------- conduction between two prescribed temperatures
def thermal_two_sinks(wd: Path):
    """A wall carried between a cold and a warm sink: the exact radial conduction 2 pi k H dT / ln(b/a).

    `outflow_W` is the NET reaction over all prescribed-temperature degrees of
    freedom, so it cancels here and the reported heat balance was meaningless (3.4e290).  The split
    `heat_removed_W` / `heat_added_W` is the cold-stage load, which is the number a cryostat design needs.
    """
    k, H = 10.0, 1.0
    a, b = 0.4, 0.5
    T_cold, T_warm = 4.0, 300.0
    rows = []
    for size in sweep((0.04, 0.02, 0.01)):
        doc = {"schema": "axionfem.design/1", "name": "two_sinks",
               "materials": {"m": {"kind": "linear", "thermal": {"k_W_mK": k}}, "cm": {"kind": "vacuum"}},
               "regions": [{"name": "cold", "shape": {"rect": {"r": [0.3, a], "z": [-H / 2, H / 2]}},
                            "material": "m", "mesh_size": size},
                           {"name": "wall", "shape": {"rect": {"r": [a, b], "z": [-H / 2, H / 2]}},
                            "material": "m", "mesh_size": size},
                           {"name": "warm", "shape": {"rect": {"r": [b, 0.6], "z": [-H / 2, H / 2]}},
                            "material": "m", "mesh_size": size},
                           {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                            "material": "cm"}],
               "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
               "thermal": {"fixed": [{"region": "cold", "T_K": T_cold}, {"region": "warm", "T_K": T_warm}]},
               "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}
        res = af.Design(doc).solve(wd / f"two_sinks_{size}", verbose=0)
        ck = res.thermal["checks"]
        exact = 2 * math.pi * k * H * (T_warm - T_cold) / math.log(b / a)
        rr = np.linspace(a + 0.005, b - 0.005, 9)
        sm = res.sample_mech(rr, np.zeros_like(rr))
        T_exact = T_cold + (T_warm - T_cold) * np.log(rr / a) / math.log(b / a)
        rows.append({"mesh_size": size, "dofs": res.data["mesh"]["dofs"],
                     "heat_removed_W": ck["heat_removed_W"], "heat_added_W": ck["heat_added_W"],
                     "exact_W": exact, "heat_removed_rel": ck["heat_removed_W"] / exact - 1,
                     "net_outflow_W": ck["outflow_W"], "generated_W": ck["generated_W"],
                     "balance_relative": ck["balance_relative"],
                     "max_abs_err_K": float(np.max(np.abs(np.asarray(sm["T_K"]) - T_exact)))})
    out = dict(STAMP, check="thermal_two_sinks",
               geometry={"wall_r_in_m": a, "wall_r_out_m": b, "height_m": H, "k_W_mK": k,
                         "T_cold_K": T_cold, "T_warm_K": T_warm},
               reference="steady radial conduction between two isothermal cylinders: "
                         "Q = 2 pi k H (T_warm - T_cold)/ln(b/a), T(r) = T_cold + dT ln(r/a)/ln(b/a)",
               exact_W=2 * math.pi * k * H * (T_warm - T_cold) / math.log(b / a), rows=rows)
    (REC / "thermal_two_sinks.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- report
def report(rec):
    L = ["# Mechanics and thermal validation (axisymmetric)", "",
         "Generated by `benchmarks/mechanics_thermal.py`. Records: `benchmarks/records/mechanics_*.json`, "
         "`benchmarks/records/thermal_*.json`. Formulation and decisions: "
         "[physics/mechanics_thermal.md](../physics/mechanics_thermal.md).", ""]

    r = rec["lame"]
    g = r["geometry"]
    L += ["## 1. Lamé thick-walled cylinder under internal pressure", "",
          f"a = {g['a_m']} m, b = {g['b_m']} m, length {g['length_m']} m, p = {g['p_Pa']:g} Pa, "
          f"E = {g['E_Pa']:g} Pa, nu = {g['nu']}. The pressure is applied with "
          "`mechanics.pressures` on the faces that the cylinder shares with the bore region.", "",
          "Two exact configurations:", "",
          "* **ends fixed** (`supports: [{region: cyl, fix: z}]`): u_z = 0 everywhere is the exact plane-strain "
          "solution, with sigma_zz = nu (sigma_rr + sigma_tt) = "
          f"{2 * g['nu'] * g['p_Pa'] * g['a_m'] ** 2 / (g['b_m'] ** 2 - g['a_m'] ** 2):g} Pa carried by the end faces;",
          "* **ends free**: sigma_rr + sigma_tt is uniform, so eps_zz is uniform and u_z = eps_zz z with "
          "sigma_zz = 0 is exact for any length.", "",
          "Errors are the maximum over 21 radii at mid-height, excluding the outer 10 % of the wall on each side "
          "(the stress recovery is least accurate right at a loaded surface). This is the mesh-convergence table.", "",
          "| ends | order | h [m] | DOFs | elements | elem. ratio | max rel. err u_r | "
          "max rel. err sigma_tt | max \\|d sigma_rr\\|/p | max \\|d sigma_zz\\|/p | equilibrium round-off |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        ratio = x.get("element_ratio_vs_previous")
        L.append(f"| {x['ends']} | {x['order']} | {x['mesh_size']:g} | {x['dofs']:,} | {x['solid_elements']:,} | "
                 f"{'-' if ratio is None else format(ratio, '.2f')} | "
                 f"{x['err_u_r']:.2e} | {x['err_hoop']:.2e} | {x['err_sigma_rr_over_p']:.2e} | "
                 f"{x['err_sigma_zz_over_p']:.2e} | {x['equilibrium_Fz_round_off']:.1e} |")
    ex = r["rows"][0]
    stalled = [x for x in r["rows"] if x.get("is_a_real_refinement") is False]
    L += ["", "`elem. ratio` is the element count divided by the previous row's, so that a halved `h` that "
          "does not actually refine the mesh is visible (it should be about 4).",
          "", f"The applied radial resultant is {ex['applied_radial_N_per_rad']:.6f} N/rad against the exact "
          f"p a L = {ex['applied_radial_exact']:.6f} N/rad in every run.", ""]
    if stalled:
        L += [f"**{len(stalled)} row(s) are not a genuine refinement** (element ratio below 2): "
              + ", ".join(f"{x['ends']} p{x['order']} h={x['mesh_size']:g}" for x in stalled) + ".", ""]

    r = rec["solenoid"]
    L += ["## 2. Long thin solenoid: hoop stress from the Lorentz body force", "",
          f"Winding inner radius {r['geometry']['R_m']} m, height {r['geometry']['H_m']} m, "
          f"{r['geometry']['ampere_turns']:g} ampere-turns, E = 100 GPa, nu = 0.3, no supports.",
          "The magnetic pressure p = (B_in^2 - B_out^2)/2 mu0 is taken from the solved field just inside and just "
          "outside the winding at mid-height, so the comparison tests the mechanics only. A thin shell carries "
          "the exact thickness-mean hoop stress for this load (**not** a membrane approximation): radial "
          "equilibrium with f_r = J B_z(r) and the linear profile B_z = B0 (b - r)/t integrates to "
          "(1/t) int sigma_tt dr = p (Rm - t/6)/t. The membrane value p Rm/t is kept beside it as the "
          "leading-order comparison. The two halves attract with p pi R^2, giving sigma_zz = "
          "-p R^2/(2 Rm t) at mid-height -- that one IS a membrane value, and its O(t/R) error is what the "
          "last column shows (it halves as t halves).", "",
          "| t/R | p [Pa] | hoop FE [Pa] | exact p(Rm-t/6)/t [Pa] | rel. | membrane p Rm/t | rel. | "
          "sigma_zz FE [Pa] | -p R^2/(2 Rm t) [Pa] | rel. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['t_over_R']:g} | {x['p_Pa']:.5g} | {x['hoop_Pa']:.6g} | "
                 f"{x['hoop_exact_distributed_Pa']:.6g} | {x['hoop_exact_rel']:+.2e} | "
                 f"{x['hoop_thin_shell_Pa']:.6g} | {x['hoop_rel']:+.2e} | "
                 f"{x['sigma_zz_Pa']:.6g} | {x['sigma_zz_ref_Pa']:.6g} | "
                 f"{x['sigma_zz_rel']:+.2e} |")
    x = r["rows"][0]
    L += ["", "The radial resultant of the body force integrated by the elasticity assembly agrees with the "
          "independent winding force of `results[\"forces\"]` to "
          f"{abs(x['lorentz_radial_N_per_rad'] / x['force_postprocessing_N_per_rad'] - 1):.1e} "
          "(same quadrature, different code path).", "",
          "The sigma_zz residual is a thickness effect, not an end effect: at t/R = 0.02 it is 1.79e-2 for "
          "H/R = 20, 40 and 80 alike, while it halves with t at fixed H/R. It is not replaced by the exact "
          "axial-resultant identity int sigma_zz dA = -F_z, because an area-weighted integral of the "
          "recovered sigma_zz across the full wall is dominated by the stress recovery at the two loaded "
          "surfaces (several per cent), which is the quantity this table deliberately excludes elsewhere.", ""]

    r = rec["sc_pressure"]
    L += ["## 3. Superconducting tube under the Maxwell surface pressure", "",
          f"A zero-field-cooled niobium tube (inner radius {r['geometry']['a_m']} m, length "
          f"{r['geometry']['shell_length_m']} m) inside a longer solenoid. The tube screens its bore "
          "(B_bore < 1e-9 T), so it feels the external pressure p = B^2/2 mu0. The reference is the **exact** "
          "thick-wall Lamé solution for an external pressure with free ends, k = -p b^2/(b^2 - a^2), "
          "sigma_tt = k (1 + a^2/r^2), sigma_rr = k (1 - a^2/r^2), whose thickness mean is exactly -p b/t. "
          "The membrane form -p Rm/t is kept beside it: using Rm instead of the outer radius b is what made "
          "this comparison look 0.5-2 % off.", "",
          "| t/R | B outside [T] | p [Pa] | max rel. err sigma_tt | max \\|d sigma_rr\\|/p | "
          "mean hoop FE [Pa] | exact -p b/t [Pa] | rel. | membrane -p Rm/t | rel. | "
          "F_z assembly [N] | F_z from `forces[]` [N] | rel. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['t_over_R']:g} | {x['B_outside_T']:.5g} | {x['p_Pa']:.5g} | "
                 f"{x['err_hoop_exact_thick_wall']:.2e} | {x['err_sigma_rr_over_p']:.2e} | "
                 f"{x['hoop_Pa']:.6g} | {x['hoop_exact_mean_Pa']:.6g} | {x['hoop_exact_mean_rel']:+.2e} | "
                 f"{x['hoop_thin_shell_Pa']:.6g} | {x['hoop_rel']:+.2e} | {x['applied_Fz_N']:.6g} | "
                 f"{x['force_postprocessing_Fz_N']:.6g} | {x['Fz_rel']:+.1e} |")
    L += ["", "The axial column is the independent cross-check: the same surface integral computed by the "
          "Maxwell-pressure post-processing of `cpp/src/model.cpp` (different quadrature order).", ""]

    r = rec["sc_sheet"]
    g = r["geometry"]
    L += ["## 3b. Zero-thickness superconducting sheet: the load it transmits", "",
          f"The same solenoid with a **steel** tube (inner radius {g['tube_r_in_m']} m, E = 200 GPa, "
          "nu = 0.3) carrying a zero-thickness superconducting skin (a `curves` entry) on its outer "
          "surface. The skin screens the wall, so the tube feels the full external pressure "
          "p = (B_out^2 - B_in^2)/2 mu0; the reference is again the exact thick-wall Lamé solution. The "
          "sheet itself has no thickness and so no stress of its own, but its surface traction is bonded "
          "into the tube -- before this fix the tube reported hoop stress exactly 0.", "",
          "`|B.n|/|B|` is the size of the (B.n)B part of the Maxwell traction that the implementation "
          "drops; it vanishes identically on an ideal superconductor, and the column shows how close the "
          "discrete field gets.", "",
          "| t/R | B screened [T] | B outside [T] | p [Pa] | max rel. err sigma_tt | "
          "max \\|d sigma_rr\\|/p | mean hoop FE [Pa] | exact -p b/t [Pa] | rel. | \\|B.n\\|/\\|B\\| |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['t_over_R']:g} | {x['B_screened_T']:.3g} | {x['B_outside_T']:.5g} | "
                 f"{x['p_Pa']:.5g} | {x['err_hoop_exact_thick_wall']:.2e} | "
                 f"{x['err_sigma_rr_over_p']:.2e} | {x['hoop_Pa']:.6g} | "
                 f"{x['hoop_exact_mean_Pa']:.6g} | {x['hoop_exact_mean_rel']:+.2e} | "
                 f"{x['maxwell_traction_Bn_over_B_max']:.1e} |")
    L += [""]

    r = rec["cooldown"]
    L += ["## 4. Uniform cooldown of a free ring", "",
          f"A free rectangular ring (r = {r['geometry']['r_m']}, z = {r['geometry']['z_m']} m), "
          f"alpha = {r['geometry']['alpha']:g}, `mechanics.cooldown: true`, no supports or loads. The exact "
          "answer is a uniform contraction u = -alpha x with zero stress.", "",
          f"* max rel. error on u_r: {r['max_rel_err_u_r']:.1e}; on u_z: {r['max_rel_err_u_z']:.1e}",
          f"* max von Mises {r['von_mises_max_Pa']:.3g} Pa = {r['von_mises_over_E_alpha']:.1e} of E alpha "
          f"({r['geometry']['E_Pa'] * r['geometry']['alpha']:.3g} Pa)",
          f"* max \\|u\\| {r['u_max_m']:.8g} m against alpha \\|x\\|_max = {r['u_max_exact_m']:.8g} m",
          f"* equilibrium round-off {r['equilibrium_Fz_round_off']:.1e}", "",
          "u_z is reported relative to the component's volume-weighted mean, which for this symmetric ring is "
          "z = 0.", ""]

    r = rec["templates"]
    L += ["## 5. The templates with their built-in material defaults", "",
          "Default parameters, `mechanics: {cooldown: true}`, built-in `winding` (smeared isotropic composite, "
          "E = 80 GPa, alpha = 3.0e-3), `nb` (E = 105 GPa, alpha = 1.43e-3) and `copper` (E = 130 GPa, "
          "alpha = 3.26e-3). These are engineering sanity numbers, not a design: the structures are "
          "unsupported, so only the cooldown and the magnetic loads act.", "",
          "`results[\"forces\"]` reports F_z for windings and superconductors but only the radial resultant "
          "of windings, so the axial sum is the complete cross-check of the load assembly; the radial residual "
          "is the superconductor surface pressure, which `forces` does not report.", "",
          "| template | applied F_z [N] | sum forces[] F_z [N] | residual [N] | applied radial [N/rad] | "
          "windings [N/rad] | superconductors [N/rad] | equilibrium round-off |",
          "|---|---|---|---|---|---|---|---|"]
    for name, t in r["templates"].items():
        L.append(f"| {name} | {t['applied_Fz_N']:.6g} | {t['forces_Fz_sum_N']:.6g} | {t['Fz_residual_N']:+.1e} | "
                 f"{t['applied_radial_N_per_rad']:.6g} | {t['winding_radial_N_per_rad']:.6g} | "
                 f"{t['sc_radial_N_per_rad']:.4g} | {t['equilibrium_Fz_round_off']:.1e} |")
    L += ["", "| template | loaded windings | pressurised bodies | superconductor surface field [T] |",
          "|---|---|---|---|"]
    for name, t in r["templates"].items():
        L.append(f"| {name} | {', '.join(t['lorentz_windings']) or '-'} | "
                 f"{', '.join(t['magnetic_pressure_bodies']) or '-'} | "
                 f"{', '.join(f'{k}: {v:.4g}' for k, v in t['sc_surface_field_T'].items()) or '-'} |")
    L += ["", "| template | region | max \\|u\\| [mm] | max von Mises [MPa] | hoop min/max [MPa] |",
          "|---|---|---|---|---|"]
    for name, t in r["templates"].items():
        for rg, v in t["regions"].items():
            L.append(f"| {name} | {rg} | {v['u_max_m'] * 1e3:.4g} | {v['von_mises_max_Pa'] / 1e6:.4g} | "
                     f"{v['sigma_hoop_Pa']['min'] / 1e6:.4g} / {v['sigma_hoop_Pa']['max'] / 1e6:.4g} |")
    L += ["", "The toroid's sheath carries essentially no magnetic load: a toroidal field is confined inside the "
          "winding, so the gap between winding and sheath is field free. Its displacement is the cooldown "
          "contraction. The solenoid template's coax walls are copper, i.e. transparent to the static field, so "
          "they too see only the cooldown.", "",
          "**The CORE row is a warning, not a result.** At the template's default 1e6 ampere-turns the cavity "
          "surface field is "
          f"{r['templates']['core']['sc_surface_field_T'].get('cavity', float('nan')):.3g} T, far above niobium's "
          "critical field (the template's own `cost.sc_wall_cap` is 0.086 T), so the magnetic pressure "
          "B^2/2 mu0 is about 0.1 MPa. The cavity's flat end walls are 1 cm thick and span 1.4 m, and linear "
          "thin-plate bending under 0.1 MPa gives exactly what the table shows: ~0.18 m of deflection and ~300 MPa "
          "of bending stress. A deflection of 18 times the wall thickness is far outside small-deflection "
          "elasticity, and buckling is not modelled, so the number only says that this cavity needs a structural "
          "design (ribs, supports, thicker walls) at this excitation. It does illustrate that the solver picks up "
          "plate bending, not only membrane stress.", ""]

    r = rec["thermal_uniform"]
    g = r["geometry"]
    L += ["## 6. Conduction: cylinder with uniform heat generation", "",
          f"A generating cylinder r < {g['R_generating_m']} m inside a shell held at {g['T0_K']} K "
          f"(r = {g['R_generating_m']}..{g['R_sink_m']} m), q = {g['q_W_m3']:g} W/m^3, "
          f"k = {g['k_W_mK']:g} W/(m K), height {g['height_m']} m with adiabatic ends. Exact: "
          "T = T0 + q (R^2 - r^2)/(4 k). The solution is quadratic in r, so order >= 2 is exact to round-off; "
          "order 1 shows the expected O(h^2) convergence.", "",
          "| order | h [m] | DOFs | max \\|dT\\| [K] | T_max [K] | exact [K] | generated [W] | outflow [W] | "
          "balance |", "|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['order']} | {x['mesh_size']:g} | {x['dofs']:,} | {x['max_abs_err_K']:.2e} | "
                 f"{x['T_max_K']:.8f} | {x['T_max_exact_K']:.8f} | {x['generated_W']:.6f} | {x['outflow_W']:.6f} | "
                 f"{x['balance_relative']:.1e} |")
    L.append("")

    r = rec["thermal_kT"]
    m = r["material"]
    L += ["## 7. Conduction with k(T): the analytic Kirchhoff solution", "",
          f"The same geometry with a two-point table k(T) = {m['k0_W_mK']:g} (1 + {m['dk_dT_over_k0']:g} (T - "
          f"{T0:g})) W/(m K) over [{m['table'][0][0]:g}, {m['table'][1][0]:g}] K, i.e. k rises by a factor "
          f"{m['table'][1][1] / m['table'][0][1]:g} over the table. The Kirchhoff potential "
          "theta = int k dT satisfies the linear Poisson equation, so theta = q (R^2 - r^2)/4 and "
          "T = T0 + (sqrt(1 + 2 c theta/k0) - 1)/c exactly. The solver uses a Picard iteration on k(T_prev) "
          f"instead, so the table is a genuine test of the nonlinear path. Temperature rise {r['temperature_rise_K']:.4f} K.",
          "", "| h [m] | DOFs | Picard its | final change | max \\|dT\\| [K] | / rise | T_max [K] | exact [K] | "
          "balance |", "|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['mesh_size']:g} | {x['dofs']:,} | {x['picard_iterations']} | {x['picard_change']:.1e} | "
                 f"{x['max_abs_err_K']:.2e} | {x['max_rel_err_of_rise']:.1e} | {x['T_max_K']:.6f} | "
                 f"{x['T_max_exact_K']:.6f} | {x['balance_relative']:.1e} |")
    r = rec["thermal_two_sinks"]
    g = r["geometry"]
    L += ["## 8. Conduction between two prescribed temperatures (the cold-stage heat load)", "",
          f"A wall r = {g['wall_r_in_m']}..{g['wall_r_out_m']} m, height {g['height_m']} m, "
          f"k = {g['k_W_mK']:g} W/(m K), carried between a sink at {g['T_cold_K']} K and one at "
          f"{g['T_warm_K']} K. Exact: Q = 2 pi k H dT / ln(b/a) = {r['exact_W']:.6f} W and "
          "T = T_cold + dT ln(r/a)/ln(b/a).", "",
          "`outflow_W` is the **net** reaction over all prescribed-temperature degrees of freedom, so here "
          "it cancels to round-off: with heat entering at one boundary and leaving at another it is not the "
          "heat load. `heat_removed_W` (the sinks that absorb heat) and `heat_added_W` (the boundaries that "
          "supply it) are reported separately, and `balance_relative` is scaled by the largest of "
          "generated / removed / added, so it is meaningful even with no internal generation.", "",
          "| h [m] | DOFs | heat removed [W] | heat added [W] | exact [W] | rel. | net outflow [W] | "
          "balance | max \\|dT\\| [K] |", "|---|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['mesh_size']:g} | {x['dofs']:,} | {x['heat_removed_W']:.6f} | "
                 f"{x['heat_added_W']:.6f} | {x['exact_W']:.6f} | {x['heat_removed_rel']:+.1e} | "
                 f"{x['net_outflow_W']:.2e} | {x['balance_relative']:.1e} | {x['max_abs_err_K']:.2e} |")
    L += [""]

    L += ["", "## Summary", "",
          "| check | reference | agreement |", "|---|---|---|",
          "| Lamé u_r, ends fixed and free | exact | "
          f"{min(x['err_u_r'] for x in rec['lame']['rows']):.0e} (best) |",
          "| Lamé sigma_tt | exact | "
          f"{min(x['err_hoop'] for x in rec['lame']['rows']):.0e} |",
          "| solenoid winding hoop stress | exact distributed load p(Rm-t/6)/t | "
          f"{min(abs(x['hoop_exact_rel']) for x in rec['solenoid']['rows']):.0e} |",
          "| solenoid axial self-compression | thin shell, O(t/R) | "
          f"{min(abs(x['sigma_zz_rel']) for x in rec['solenoid']['rows']):.0e} |",
          "| superconductor surface pressure, sigma_tt(r) | exact thick-wall Lamé | "
          f"{min(x['err_hoop_exact_thick_wall'] for x in rec['sc_pressure']['rows']):.0e} |",
          "| superconductor surface pressure, thickness mean | exact -p b/t | "
          f"{min(abs(x['hoop_exact_mean_rel']) for x in rec['sc_pressure']['rows']):.0e} |",
          "| superconducting SHEET load on a structure | exact thick-wall Lamé | "
          f"{min(x['err_hoop_exact_thick_wall'] for x in rec['sc_sheet']['rows']):.0e} |",
          "| superconductor axial load vs forces[] | independent code path | "
          f"{max(abs(x['Fz_rel']) for x in rec['sc_pressure']['rows']):.0e} |",
          f"| free cooldown displacement | exact | {rec['cooldown']['max_rel_err_u_r']:.0e} |",
          f"| free cooldown stress | zero | {rec['cooldown']['von_mises_over_E_alpha']:.0e} of E alpha |",
          "| axial equilibrium closure | exact identity | "
          f"{max(max(x['equilibrium_Fz_round_off'] for x in rec['lame']['rows']), max(x['equilibrium_Fz_round_off'] for x in rec['solenoid']['rows'])):.0e} |",
          "| conduction, uniform q | exact | "
          f"{min(x['max_abs_err_K'] for x in rec['thermal_uniform']['rows']):.0e} K |",
          "| conduction heat balance | exact identity | "
          f"{max(x['balance_relative'] for x in rec['thermal_uniform']['rows']):.0e} |",
          "| conduction with k(T) | analytic Kirchhoff | "
          f"{min(x['max_rel_err_of_rise'] for x in rec['thermal_kT']['rows']):.0e} of the rise |",
          "| conduction between two sinks (heat load) | exact 2 pi k H dT/ln(b/a) | "
          f"{min(abs(x['heat_removed_rel']) for x in rec['thermal_two_sinks']['rows']):.0e} |", ""]
    return "\n".join(L) + "\n"


def main():
    wd = Path(sys.argv[1] if len(sys.argv) > 1 else "mechanics_thermal_runs")
    wd.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    rec = {"lame": lame(wd), "solenoid": solenoid(wd), "sc_pressure": sc_pressure(wd),
           "sc_sheet": sc_sheet(wd), "cooldown": cooldown(wd), "templates": templates(wd),
           "thermal_uniform": thermal_uniform(wd), "thermal_kT": thermal_kT(wd),
           "thermal_two_sinks": thermal_two_sinks(wd)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report(rec))
    print(OUT)


if __name__ == "__main__":
    main()
