"""Validation of the axisymmetric critical-state (HTS screening-current) solver.

    python benchmarks/hts_validation.py [workdir]

Writes benchmarks/records/hts_validation.json and docs/validation/hts.md.

References used here
  * Bean critical state of a long cylinder of radius a in an axial field: penetration field
    Hp = Jc a, virgin magnetization M(Ha) = -(Jc a/3)[1 - (1 - Ha/Hp)^3] for Ha <= Hp and
    M = -Jc a/3 above it, magnetic moment m = M V.  The field at the centre is
    B = mu0 (Ha - Hp) once the sample is fully penetrated.
  * Bean hysteresis loop (same geometry): the descending branch from H0 is
    M_dec(H) = M_v(H0) - 2 M_v((H0 - H)/2) and the ascending branch is its mirror image; the loss
    per cycle per unit volume is mu0 * the loop area.
  * Critical surfaces: Bottura (NbTi), Durham/ITER (Nb3Sn), Grilli et al. (REBCO); see the
    conductor table in the report.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records" / "hts_validation.json"
OUT = ROOT / "docs" / "validation" / "hts.md"
MU0 = 1.25663706212e-6


# ---------------------------------------------------------------- Bean references
def bean_virgin(h: float) -> float:
    """M(h)/(Jc a/3) of a long cylinder, h = Ha/Hp (negative = diamagnetic)."""
    return -(1.0 - (1.0 - min(h, 1.0)) ** 3)


def bean_loop_area(h0: float, npts: int = 40001) -> float:
    """Loss per cycle per unit volume of a symmetric cycle of amplitude h0 = H0/Hp, in units of
    mu0 Hp^2 (the area of the M-H loop with M in units of Hp)."""
    def Mv(h):
        return bean_virgin(h) / 3.0

    def Mdec(h):
        return Mv(h0) - 2.0 * Mv((h0 - h) / 2.0)

    hs = np.linspace(-h0, h0, npts)
    dec = np.array([Mdec(h) for h in hs])
    asc = np.array([-Mdec(-h) for h in hs])
    return float(np.trapezoid(dec - asc, hs)) if hasattr(np, "trapezoid") else float(np.trapz(dec - asc, hs))


# ---------------------------------------------------------------- the test design
def cylinder(wd: Path, tag: str, *, a=0.05, Hcyl=0.6, Jc=1.0e6, n=25, h=0.004, order=2,
             scales=(0.0, 1.0), times=None, amp=2.0, steps=20, e_floor=1e-6, field_cooled=False,
             t_ramp=None, max_newton=120, newton_tol=1e-4, domain=1.5, verbose=0):
    """Solid HTS cylinder (radius a, height Hcyl) in a uniform axial background field ramped by the
    drive waveform.  `amp` is the peak applied field in units of the Bean penetration field
    Hp = Jc a.  The default ramp rate puts the characteristic electric field (dB/dt) a/2 at E0, so
    |J| ~ Jc in the penetrated region and the finite-n correction is small."""
    Bp = MU0 * Jc * a
    if t_ramp is None:
        t_ramp = amp * Bp * a / (2.0 * 1e-4)  # (dB/dt) a / 2 = E0
    if times is None:
        times = [t_ramp * i for i in range(len(scales))]
    doc = {
        "schema": "axionfem.design/1", "name": f"hts_cyl_{tag}",
        "parameters": {"a": a, "Hc": Hcyl},
        "conductors": {"bean": {"law": "table", "B_T": [0.0, 100.0], "Jc": [Jc, Jc],
                                "n_value": n, "E0": 1e-4, "T_op": 20.0}},
        "regions": [{"name": "bulk", "shape": {"rect": {"r": [0, "a"], "z": ["-Hc/2", "Hc/2"]}},
                     "material": "superconductor", "mesh_size": h}],
        "background": {"type": "uniform", "Bz": amp * Bp},
        "hts": {"regions": [{"region": "bulk", "conductor": "bean",
                             "initial": "field_cooled" if field_cooled else "zero_field_cooled"}],
                "waveform": {"times": list(times), "scales": list(scales)},
                "steps_per_segment": steps, "e_floor": e_floor,
                "max_newton": max_newton, "newton_tol": newton_tol,
                "field_of_view": [[0.0, 0.0]], "profile_points": 41},
        "mesh": {"order": order, "size": 0.15, "domain_radius": domain},
    }
    t0 = time.time()
    res = af.Design(doc).solve(wd / tag, verbose=verbose)
    hts = res.data["hts"]
    V = math.pi * a * a * Hcyl
    m_full = -Jc * a / 3.0 * V  # Bean full-penetration moment
    out = {"tag": tag, "a_m": a, "H_m": Hcyl, "Jc_A_m2": Jc, "n": n, "mesh_size_m": h, "order": order,
           "Bp_T": Bp, "amp_over_Hp": amp, "steps_per_segment": steps, "e_floor": e_floor,
           "t_ramp_s": times[1] - times[0], "V_m3": V, "m_full_bean_Am2": m_full,
           "dofs": res.data["mesh"]["dofs"], "elements": res.data["mesh"]["elements"],
           "seconds": time.time() - t0, "hts_seconds": hts["seconds"],
           "newton": hts["newton"], "balance_be": hts["energy_balance_backward_euler_worst_rel"],
           "balance": hts["energy_balance_worst_rel"], "total_loss_J": hts["total_loss_J"],
           "total_work_J": hts["total_work_J"],
           "steps": [{"t": s["t_s"], "scale": s["scale"], "m": s["regions"][0]["moment_Am2"],
                      "I": s["regions"][0]["screening_current_A"],
                      "maxJ_over_Jc": s["regions"][0]["max_J_over_Jc"],
                      "Bz0": s["field_of_view"][0]["Bz_T"],
                      "loss": s["cumulative_loss_J"],
                      "bal": s.get("energy_balance_rel"), "be": s.get("energy_balance_backward_euler_rel"),
                      "nw": s.get("newton_iterations"), "upd": s.get("newton_update")}
                     for s in hts["steps"]],
           "final": {"m": hts["final"]["regions"][0]["moment_Am2"],
                     "maxJ_over_Jc": hts["final"]["regions"][0]["max_J_over_Jc"]},
           "profile": [[p["r_m"] / a, p["J_A_m2"] / Jc] for p in
                       hts["final"]["current_profiles"][0]["points"]],
           "warnings": [w for w in res.data["warnings"] if "critical" in w]}
    if "trapped" in hts["final"]:
        out["trapped"] = {"m": hts["final"]["trapped"]["moment_Am2"],
                          "Bz0_T": hts["final"]["trapped"]["points"][0]["Bz_T"]}
    return out


# ---------------------------------------------------------------- cases
def case_bean_limit(wd):
    """Full-penetration moment and the J profile as n grows: the Bean limit."""
    rows = []
    for n in (10, 15, 25, 50, 100):
        r = cylinder(wd, f"n{n}", n=n, amp=2.0, steps=24)
        rows.append({"n": n, "m": r["final"]["m"], "m_bean": r["m_full_bean_Am2"],
                     "ratio": r["final"]["m"] / r["m_full_bean_Am2"], "maxJ_over_Jc": r["final"]["maxJ_over_Jc"],
                     "balance_be": r["balance_be"], "seconds": r["hts_seconds"],
                     "profile": r["profile"], "steps_not_converged": r["newton"]["steps_not_converged"]})
    return rows


def case_magnetization(wd, Hcyl, tag):
    """Virgin magnetization curve against the Bean cylinder formula."""
    r = cylinder(wd, tag, Hcyl=Hcyl, n=50, amp=1.6, steps=40)
    Bp, m_full = r["Bp_T"], r["m_full_bean_Am2"]
    rows = []
    for s in r["steps"]:
        h = s["scale"] * r["amp_over_Hp"]
        if h <= 0:
            continue
        rows.append({"h": h, "m_over_full": s["m"] / m_full, "bean": -bean_virgin(h),
                     "Bz0_T": s["Bz0"], "Bz0_bean_T": max(0.0, (h - 1.0)) * Bp})
    return {"aspect": Hcyl / (2 * r["a_m"]), "Bp_T": Bp, "m_full": m_full, "rows": rows,
            "dofs": r["dofs"], "seconds": r["hts_seconds"], "balance_be": r["balance_be"]}


def case_trapped(wd):
    """Trapped flux after a ramp-down: zero-field cooled versus field cooled."""
    out = []
    # ZFC: 0 -> Hp -> 0.  Bean: m(0) = m_v(Hp) - 2 m_v(Hp/2) = (+0.75) m_full.
    r = cylinder(wd, "zfc_trap", n=50, amp=1.0, scales=(0.0, 1.0, 0.0), steps=30)
    out.append({"history": "zero_field_cooled, 0 -> Hp -> 0", "m": r["trapped"]["m"],
                "m_over_full": r["trapped"]["m"] / r["m_full_bean_Am2"], "bean_over_full": -0.75,
                "Bz0_T": r["trapped"]["Bz0_T"], "balance_be": r["balance_be"]})
    # FC at Hp, then down to 0.  Bean: the mirrored virgin curve, m(0) = -m_full (full reverse).
    r = cylinder(wd, "fc_trap", n=50, amp=1.0, scales=(1.0, 0.0), steps=30, field_cooled=True)
    out.append({"history": "field_cooled at Hp, Hp -> 0", "m": r["trapped"]["m"],
                "m_over_full": r["trapped"]["m"] / r["m_full_bean_Am2"], "bean_over_full": -1.0,
                "Bz0_T": r["trapped"]["Bz0_T"], "balance_be": r["balance_be"]})
    # ZFC to 2 Hp and back: fully reversed, m(0) = -m_full.
    r = cylinder(wd, "zfc_trap2", n=50, amp=2.0, scales=(0.0, 1.0, 0.0), steps=30)
    out.append({"history": "zero_field_cooled, 0 -> 2Hp -> 0", "m": r["trapped"]["m"],
                "m_over_full": r["trapped"]["m"] / r["m_full_bean_Am2"], "bean_over_full": -1.0,
                "Bz0_T": r["trapped"]["Bz0_T"], "balance_be": r["balance_be"]})
    return out


def case_ac_loss(wd):
    """AC loss per cycle of a symmetric cycle against the Bean loop area."""
    rows = []
    for h0 in (0.5, 1.0, 2.0):
        # 0 -> h0 -> -h0 -> h0 -> -h0: the last full cycle is the steady loop
        r = cylinder(wd, f"ac{h0:g}", n=50, amp=h0, steps=30,
                     scales=(0.0, 1.0, -1.0, 1.0, -1.0))
        V, Bp = r["V_m3"], r["Bp_T"]
        tb = [s["t"] for s in r["steps"]]
        cum = [s["loss"] for s in r["steps"]]
        # segment ends: indices 30, 60, 90, 120 (steps per segment = 30)
        k = r["steps_per_segment"]
        q = (cum[3 * k] - cum[1 * k])  # -h0 -> h0 -> -h0 ... one full cycle after the virgin ramp
        q2 = (cum[4 * k] - cum[2 * k])  # the next full cycle (check for a steady loop)
        area = bean_loop_area(h0)
        q_bean = MU0 * (Bp / MU0) ** 2 * area * V  # mu0 Hp^2 * area * V
        rows.append({"h0": h0, "Q_cycle_J": q2, "Q_previous_cycle_J": q, "Q_bean_J": q_bean,
                     "ratio": q2 / q_bean, "loop_area": area, "balance_be": r["balance_be"],
                     "dofs": r["dofs"], "seconds": r["hts_seconds"], "t_s": tb[-1]})
    return rows


def case_convergence(wd):
    """Convergence with the time step, the mesh and the E-J floor."""
    out = {"time_step": [], "mesh": [], "e_floor": []}
    for steps in (12, 24, 48, 96):
        r = cylinder(wd, f"dt{steps}", n=50, amp=2.0, steps=steps)
        out["time_step"].append({"steps": steps, "m": r["final"]["m"], "loss": r["total_loss_J"],
                                 "balance": r["balance"], "balance_be": r["balance_be"],
                                 "seconds": r["hts_seconds"]})
    for h in (0.008, 0.004, 0.002):
        r = cylinder(wd, f"h{h:g}", n=50, amp=2.0, steps=24, h=h)
        out["mesh"].append({"h": h, "elements": r["elements"], "dofs": r["dofs"], "m": r["final"]["m"],
                            "loss": r["total_loss_J"], "balance_be": r["balance_be"],
                            "seconds": r["hts_seconds"]})
    for ef in (1e-4, 1e-6, 1e-8):
        r = cylinder(wd, f"ef{ef:g}", n=50, amp=2.0, steps=24, e_floor=ef)
        out["e_floor"].append({"e_floor": ef, "m": r["final"]["m"], "loss": r["total_loss_J"],
                               "maxJ_over_Jc": r["final"]["maxJ_over_Jc"], "balance_be": r["balance_be"],
                               "seconds": r["hts_seconds"]})
    return out


def case_order(wd):
    """Finite-element order."""
    rows = []
    for order in (1, 2, 3):
        r = cylinder(wd, f"p{order}", n=50, amp=2.0, steps=24, order=order)
        rows.append({"order": order, "dofs": r["dofs"], "m": r["final"]["m"], "loss": r["total_loss_J"],
                     "balance_be": r["balance_be"], "seconds": r["hts_seconds"]})
    return rows


def case_conductors():
    """Critical surfaces against typical values (and, for REBCO, the published fit parameter)."""
    rows = [
        {"conductor": "nbti", "point": "5 T, 4.2 K", "Jc_A_mm2": af.jc("nbti", 5.0) / 1e6,
         "reference": "~3000 (LHC NbTi wire, Bottura 2000 fit)"},
        {"conductor": "nbti", "point": "8 T, 4.2 K", "Jc_A_mm2": af.jc("nbti", 8.0) / 1e6,
         "reference": "~1200-1500 (LHC strand data)"},
        {"conductor": "nbti", "point": "10 T, 1.9 K", "Jc_A_mm2": af.jc("nbti", 10.0, 0.0, 1.9) / 1e6,
         "reference": ">=1500 (LHC inner-layer strand specification)"},
        {"conductor": "nb3sn", "point": "12 T, 4.2 K, eps = 0", "Jc_A_mm2": af.jc("nb3sn", 12.0) / 1e6,
         "reference": "~1000-1100 (ITER-grade strand, unstrained)"},
        {"conductor": "nb3sn", "point": "12 T, 4.2 K, eps = -0.25 %",
         "Jc_A_mm2": af.jc({"law": "nb3sn_durham", "strain": -0.0025}, 12.0) / 1e6,
         "reference": ">=800 (ITER TF strand specification, non-Cu)"},
        {"conductor": "rebco", "point": "self field, 77 K", "Jc_A_mm2": af.jc("rebco", 0.0) / 1e6,
         "reference": "4.75e4 A/mm^2 = Jc0 of Grilli et al. 2014 set #1"},
        {"conductor": "rebco", "point": "0.1 T perpendicular, 77 K", "Jc_A_mm2": af.jc("rebco", 0.1) / 1e6,
         "reference": "Jc0/(1+B/B0)^b with B0 = 35 mT, b = 0.6"},
        {"conductor": "rebco", "point": "0.1 T parallel, 77 K", "Jc_A_mm2": af.jc("rebco", 0.0, 0.1) / 1e6,
         "reference": "same with k B = 25 mT (anisotropy k = 0.25)"},
    ]
    # Ic of a 4 mm x 1 um tape in self field (the paper quotes 158 A including the self field)
    rows.append({"conductor": "rebco", "point": "Ic of a 4 mm x 1 um tape, self field (no self-field reduction)",
                 "Jc_A_mm2": af.jc("rebco", 0.0) * 4e-3 * 1e-6, "reference": "158 A measured (Grilli et al. 2014)"})
    return rows


def case_margins(wd):
    """Load-line and temperature margins of a solenoid winding (NbTi)."""
    d = af.template("solenoid", R=1.0, H=2.0)
    d.doc["windings"][0]["conductor"] = "coil"
    d.doc["conductors"] = {"coil": {"law": "nbti_bottura", "T_op": 4.2, "fill_factor": 0.3}}
    d.doc["cost"] = {"load_line_cap": 0.8}
    r = d.solve(wd / "margins", verbose=0)
    b = r.benchmark("optimistic")
    pick = ("J_engineering_A_m2", "J_superconductor_A_m2", "B_peak_T", "Jc_A_m2", "load_line_fraction",
            "current_sharing_T_K", "temperature_margin_K", "short_sample", "status")
    def take(m):
        return {k: {kk: v[kk] for kk in pick if kk in v} for k, v in m.items()}
    return {"as_designed": take(r.conductors), "amplitude_scale": b["amplitude_scale"],
            "at_requirement": take(b["conductor_margins"]),
            "length_scale": b["cost_optimal_size"]["length_scale"],
            "at_cost_optimum": take(b["cost_optimal_size"]["conductor_margins"]),
            "caps_at_requirement": b.get("caps_at_req"),
            "caps_at_cost_optimum": b["cost_optimal_size"].get("caps"),
            "warnings": r.data["warnings"]}


def case_core_fc(wd):
    """Consistency check of the critical-state solve and of the recomputed Psi: a field-cooled CORE
    cavity held at constant drive must stay static (no screening currents, no dissipation) and its Psi
    must agree with the ideal (Meissner) result up to the difference between the two wall models - the
    ideal model excludes the field from the wall volume, the critical state lets it through a wall that
    carries no net current, an O(wall thickness / radius) difference."""
    d = af.template("core", Rc=0.6, b=1.2, H=1.2, w=0.012, NI=5e5,
                    mesh={"order": 2, "size": 0.06, "surface_size": 0.006})
    d.doc["conductors"] = {"tape": {"law": "rebco_kim", "T_op": 20.0, "fill_factor": 0.5, "Jc0": 1.0e10,
                                    "n_value": 25, "tape_normal": [1, 0]}}
    d.doc["hts"] = {"regions": [{"region": "cavity", "conductor": "tape", "initial": "field_cooled"}],
                    "waveform": {"times": [0.0, 300.0], "scales": [1.0, 1.0]},
                    "steps_per_segment": 3, "field_of_view": [[0.0, 0.0], [0.3, 0.0]], "profile_points": 0}
    r = d.solve(wd / "core_fc", verbose=0)
    h = r.hts
    fin = h["final"]["regions"][0]
    # the same cavity, zero-field cooled and ramped up instead: the wall screens the bore
    d2 = af.Design(json.loads(json.dumps(d.doc)))
    d2.doc["hts"] = {"regions": [{"region": "cavity", "conductor": "tape", "initial": "zero_field_cooled"}],
                     "drive": {"t_ramp": 300.0, "ramp_down": True}, "steps_per_segment": 6,
                     "field_of_view": [[0.0, 0.0], [0.3, 0.0]], "profile_points": 0}
    r2 = d2.solve(wd / "core_zfc", verbose=0)
    h2 = r2.hts
    up = [s for s in h2["steps"] if s["scale"] == 1.0][0]
    return {"Psi_ideal": r.Psi, "L_p": r.L_p, "wall_m": 0.012, "radius_m": 1.2,
            "held": {"Psi": h["psi"]["Psi_critical_state_SI_H_T_m2"],
                     "rel_to_ideal": h["psi"]["relative_change"],
                     "screening_current_A": fin["screening_current_A"], "max_J_over_Jc": fin["max_J_over_Jc"],
                     "loss_J": h["total_loss_J"],
                     "scif_at_centre_T": h["steps"][-1]["field_of_view"][0]["scif_Bz_T"],
                     "Bz_centre_T": h["steps"][-1]["field_of_view"][0]["Bz_T"]},
            "zfc_ramped_up": {"Psi": up["Psi_SI_H_T_m2"], "rel_to_ideal": up["Psi_SI_H_T_m2"] / r.Psi - 1,
                              "screening_current_A": up["regions"][0]["screening_current_A"],
                              "max_J_over_Jc": up["regions"][0]["max_J_over_Jc"],
                              "Bz_centre_T": up["field_of_view"][0]["Bz_T"],
                              "Bz_applied_T": up["field_of_view"][0]["Bz_applied_T"]},
            "zfc_trapped": h2["final"].get("trapped", {}).get("max_Bz_T"),
            "zfc_loss_J": h2["total_loss_J"], "dofs": r.data["mesh"]["dofs"],
            "warnings": r.data["warnings"] + r2.data["warnings"]}


def case_poloidal_margin(wd):
    """Conductor margins of a poloidal (toroidal-magnet) winding: the current density is J = grad(G)/r,
    not ampere-turns/area, and it is singular where the winding path turns a sharp corner."""
    rows = []
    for ss in (0.01, 0.005, 0.0025):
        d = af.template("toroid", R=1.0, y=0.6, H=1.0, NI=1e6,
                        mesh={"order": 2, "size": 0.08, "surface_size": ss})
        d.doc["windings"][0]["conductor"] = "coil"
        d.doc["conductors"] = {"coil": {"law": "nbti_bottura", "T_op": 4.2, "fill_factor": 0.3}}
        r = d.solve(wd / f"tor_{ss:g}", verbose=0)
        m = r.conductors["toroid"]
        rows.append({"surface_size_m": ss, "dofs": r.data["mesh"]["dofs"],
                     "J_worst": m["J_engineering_A_m2"], "J_mean": m["J_engineering_mean_A_m2"],
                     "load_line_fraction": m["load_line_fraction"], "B_T": m["worst_point"]["B_T"],
                     "worst_r": m["worst_point"]["r_m"], "worst_z": m["worst_point"]["z_m"],
                     "warnings": [w for w in r.data["warnings"] if "current density" in w]})
    # thin-leg reference: a uniform current sheet of thickness tw at the inner radius
    import math as _m
    rows.append({"reference": "NI/(2 pi r t) at the inner leg (r = 0.6 m, t = 0.01 m)",
                 "J": 1e6 / (2 * _m.pi * 0.6 * 0.01)})
    return rows


def case_shield(wd):
    """Showcase: a solenoid with a REBCO tube shield around a shorted coaxial readout.  The ideal
    (Meissner) model excludes the field from the tube; the critical state lets flux creep through it,
    which changes the field in the bore (SCIF) and the coupling Psi."""
    R, H = 0.5, 1.2
    doc = {"schema": "axionfem.design/1", "name": "hts_shield",
           "parameters": {"R": R, "H": H, "x": 0.4, "tw": 0.01, "tc": 0.004, "tp": 0.01, "ts": 0.01,
                          "g": 0.02, "NI": 1.0e6},
           "conductors": {"tape": {"law": "rebco_kim", "T_op": 20.0, "fill_factor": 0.5,
                                   "Jc0": 2.0e10, "n_value": 25, "tape_normal": [1, 0]}},
           "regions": [
               {"name": "winding", "shape": {"rect": {"r": ["R", "R+tw"], "z": ["-H/2", "H/2"]}},
                "material": "winding"},
               {"name": "shield", "shape": {"rect": {"r": ["R-g-ts", "R-g"], "z": ["-H/2", "H/2"]}},
                "material": "superconductor", "mesh_size": 0.0025},
               {"name": "coax_inner", "shape": {"rect": {"r": ["x*R-tc", "x*R"], "z": ["-H/2", "H/2"]}},
                "material": "copper"},
               {"name": "coax_outer", "shape": {"rect": {"r": ["R-g-ts-0.02-tc", "R-g-ts-0.02"],
                                                         "z": ["-H/2", "H/2"]}}, "material": "copper"},
               {"name": "coax_short", "shape": {"rect": {"r": ["x*R-tc", "R-g-ts-0.02"],
                                                         "z": ["H/2-tc", "H/2"]}}, "material": "copper"},
               {"name": "port", "shape": {"rect": {"r": ["x*R", "R-g-ts-0.02-tc"],
                                                   "z": ["-H/2", "-H/2+tp"]}}, "material": "vacuum"}],
           "windings": [{"name": "solenoid", "region": "winding", "type": "azimuthal", "ampere_turns": "NI"}],
           "pickup": {"type": "poloidal", "port": "port"},
           "hts": {"regions": [{"region": "shield", "conductor": "tape"}],
                   "drive": {"t_ramp": 600.0, "t_hold": 200.0, "ramp_down": True},
                   "steps_per_segment": 12, "max_newton": 120,
                   "field_of_view": [[0.0, 0.0], [0.1, 0.0], [0.18, 0.0], [0.0, 0.3], [0.1, 0.3]],
                   "profile_points": 25},
           "mesh": {"order": 2, "size": 0.05, "surface_size": 0.005, "domain_radius": 1.6}}
    t0 = time.time()
    r = af.Design(doc).solve(wd / "shield", verbose=0)
    h = r.data["hts"]
    rows = []
    for s in h["steps"]:
        rows.append({"t": s["t_s"], "scale": s["scale"], "I_screen": s["regions"][0]["screening_current_A"],
                     "maxJ_over_Jc": s["regions"][0]["max_J_over_Jc"], "loss": s["cumulative_loss_J"],
                     "Psi": s.get("Psi_SI_H_T_m2"),
                     "scif": [p["scif_Bz_T"] for p in s["field_of_view"]],
                     "Bz": [p["Bz_T"] for p in s["field_of_view"]]})
    return {"Psi_ideal": r.coupling["Psi_SI_H_T_m2"], "L_p": r.coupling["L_p_H"],
            "psi": h["psi"], "final": {k: v for k, v in h["final"].items() if k != "current_profiles"},
            "profile": [[p["r_m"], p["J_A_m2"], p["Jc_A_m2"]] for p in
                        h["final"]["current_profiles"][0]["points"]],
            "ac_loss_per_cycle_J": h.get("ac_loss_per_cycle_J"), "total_loss_J": h["total_loss_J"],
            "rows": rows, "dofs": r.data["mesh"]["dofs"], "seconds": time.time() - t0,
            "hts_seconds": h["seconds"], "newton": h["newton"],
            "balance_be": h["energy_balance_backward_euler_worst_rel"],
            "warnings": r.data["warnings"]}


# ---------------------------------------------------------------- report
def report(rec):
    L = ["# Critical-state (HTS screening-current) validation", "",
         "Generated by `benchmarks/hts_validation.py` (axisymmetric, order 2 unless stated).",
         "Formulation and the regularized E-J law: `cpp/include/axionfem/hts.hpp`.  The reference solutions are the Bean",
         "critical state of a long cylinder in an axial field (penetration field Hp = Jc a, virgin magnetization",
         "M = -(Jc a/3)[1 - (1 - Ha/Hp)^3], centre field mu0 (Ha - Hp) when fully penetrated, and the",
         "hysteresis loop built from the same curve).", ""]

    L += ["## Conductor critical surfaces against published parametrizations", "",
          "Jc of the built-in conductors at reference points.  NbTi: Bottura's practical fit with the LHC",
          "strand parameters (C0 = 90.3 kA T/mm^2, Bc20 = 14.5 T, Tc0 = 9.2 K, alpha = 0.57, beta = 0.90,",
          "gamma = 2.32; L. Bottura, IEEE Trans. Appl. Supercond. 10 (2000) 1054).  Nb3Sn: the Durham/ITER",
          "unified scaling law at fixed axial strain (C = 83075 A T/mm^2, Bc20 = 33.24 T, Tc0 = 16.34 K,",
          "p = 0.593, q = 2.156, Ca1 = 50.06, Ca2 = 0, eps_0a = 3.12e-3; A. Godeke et al., Supercond. Sci.",
          "Technol. 19 (2006) R100; L. Bottura and B. Bordini, IEEE TAS 19 (2009) 1521).  REBCO: the",
          "anisotropic (elliptical) Kim form Jc0/[1 + sqrt((k B_par)^2 + B_perp^2)/B0]^b with the 77 K",
          "parameters of F. Grilli et al., IEEE TAS 24 (2014) 8000508, Table II set #1 (Jc0 = 4.75e10 A/m^2,",
          "B0 = 35 mT, b = 0.6, k = 0.25).", "",
          "The last column gives typical values for orientation: the NbTi and Nb3Sn figures are indicative values of",
          "LHC and ITER strands, not traced here to a specification document; only the REBCO entry is a published",
          "parameter.", "",
          "| conductor | point | Jc [A/mm^2] | typical value (indicative) |", "|---|---|---|---|"]
    for r in rec["conductors"]:
        L.append(f"| {r['conductor']} | {r['point']} | {r['Jc_A_mm2']:.4g} | {r['reference']} |")
    L += ["", "The REBCO temperature factor ((Tc - T)/(Tc - T_ref))^m_T (Tc = 92 K, m_T = 1.5) is a",
          "convenience extension of a field fit measured at 77 K and must be refitted for quantitative work",
          "away from 77 K.  Both low-temperature fits diverge as B -> 0 (NbTi as B^-0.43), so Jc is evaluated",
          "at max(|B|, B_min) with B_min = 50 mT by default.", ""]

    b = rec["bean_limit"]
    L += ["## Bean limit: full-penetration moment of a long cylinder", "",
          "Solid cylinder a = 5 cm, height 60 cm (aspect 6), Jc = 1 MA/m^2 (Hp = Jc a = 5e4 A/m,",
          "mu0 Hp = 62.8 mT), zero-field cooled and ramped to 2 Hp with (dB/dt) a/2 = E0, 24 steps,",
          "h = 4 mm.  At full penetration the Bean moment is m = -Jc V a/3 independently of the aspect",
          "ratio; the power law gives |J| = Jc (E/E0)^(1/n) < Jc where E < E0, so the deficit falls like 1/n.",
          "", "| n | m [A m^2] | m/m_Bean | max \\|J\\|/Jc | backward-Euler energy identity | steps not converged | s |",
          "|---|---|---|---|---|---|---|"]
    for r in b:
        L.append(f"| {r['n']} | {r['m']:.4f} | {r['ratio']:.4f} | {r['maxJ_over_Jc']:.4f} | "
                 f"{r['balance_be']:.1e} | {r['steps_not_converged']} | {r['seconds']:.0f} |")
    L += ["", "Screening-current profile J(r)/Jc at mid-height after full penetration (n = 100):", "",
          "| r/a | " + " | ".join(f"{p[0]:.2f}" for p in b[-1]["profile"][::5]) + " |",
          "|---|" + "---|" * len(b[-1]["profile"][::5]),
          "| J/Jc | " + " | ".join(f"{p[1]:+.3f}" for p in b[-1]["profile"][::5]) + " |", ""]

    for key, title in (("magnetization_6", "aspect 6"), ("magnetization_12", "aspect 12")):
        mg = rec[key]
        L += [f"## Virgin magnetization and centre field ({title}, n = 50)", "",
              f"Hp = {mg['Bp_T'] / MU0:.4g} A/m (mu0 Hp = {mg['Bp_T'] * 1e3:.3f} mT), "
              f"{mg['dofs']:,} DOFs, {mg['seconds']:.0f} s.  A finite cylinder concentrates the applied flux at",
              "its equator (demagnetization), so penetration starts below Hp and |m| runs above the Bean",
              "curve; the deviation falls with the aspect ratio.  Both moment columns are magnitudes in units",
              "of the Bean full-penetration moment |m_full| = Jc a V/3 (the moment itself is diamagnetic).", "",
              "| Ha/Hp | \\|m\\|/\\|m_full\\| | Bean | deviation | B(0) [mT] | Bean [mT] |", "|---|---|---|---|---|---|"]
        for r in mg["rows"]:
            L.append(f"| {r['h']:.3f} | {r['m_over_full']:.4f} | {r['bean']:.4f} | "
                     f"{r['m_over_full'] / r['bean'] - 1:+.3f} | {r['Bz0_T'] * 1e3:.3f} | {r['Bz0_bean_T'] * 1e3:.3f} |")
        L.append("")

    L += ["## Trapped flux and cooldown history (n = 50)", "",
          "Bean predicts m(0)/m_full = +0.75 for a zero-field-cooled cylinder cycled 0 -> Hp -> 0 and +1 for",
          "a field-cooled one brought from Hp to 0 (and for a zero-field-cooled cycle to 2 Hp, which fully",
          "reverses the critical state).  The remaining moment is paramagnetic (it holds the flux in), so the",
          "ratio is positive here while it is diamagnetic, and negative, during the ramp up.", "",
          "| history | m [A m^2] | m/m_full | Bean | B(0) trapped [mT] | energy identity |", "|---|---|---|---|---|---|"]
    for r in rec["trapped"]:
        L.append(f"| {r['history']} | {r['m']:.4f} | {-r['m_over_full']:.4f} | {-r['bean_over_full']:.2f} | "
                 f"{r['Bz0_T'] * 1e3:.3f} | {r['balance_be']:.1e} |")
    L.append("")

    L += ["## AC loss per cycle against the Bean hysteresis loop (n = 50)", "",
          "Symmetric cycles of amplitude h0 = H0/Hp after the virgin ramp; the reference is mu0 Hp^2 V times",
          "the area of the Bean M-H loop.  `Q_cycle` is the last full cycle and `Q_previous` the one before",
          "it (equal to within the step error once the loop is periodic).", "",
          "| H0/Hp | Q_cycle [J] | previous cycle [J] | Bean [J] | ratio | loop area | energy identity |",
          "|---|---|---|---|---|---|---|"]
    for r in rec["ac_loss"]:
        L.append(f"| {r['h0']:g} | {r['Q_cycle_J']:.5g} | {r['Q_previous_cycle_J']:.5g} | {r['Q_bean_J']:.5g} | "
                 f"{r['ratio']:.4f} | {r['loop_area']:.4f} | {r['balance_be']:.1e} |")
    excess = ", ".join("%+.1f%% at %g Hp" % (100 * (r["ratio"] - 1), r["h0"]) for r in rec["ac_loss"])
    L += ["", f"The loss runs above the Bean loop and the excess falls with the amplitude ({excess}), as",
          "expected: the finite-n flux creep dissipates on top of the critical state, and the small-amplitude",
          "loop of a finite cylinder is the most sensitive to the equatorial field concentration.", ""]

    cv = rec["convergence"]
    L += ["## Convergence (n = 50, cylinder ramped to 2 Hp)", "",
          "`balance` is the worst single-step relative error of the physical energy balance (step work -",
          "dissipation - change in stored energy).  Being a worst-of-steps measure it is dominated by the",
          "first steps of the ramp, where the step's own work is smallest, and it does not fall with the step",
          "size; the quantities that converge are the moment and the loss.  `balance_be` adds the exact",
          "backward-Euler term 1/2 dU^T K dU and is then an identity of the discrete equations, so it measures",
          "how well each step was solved, not how accurate the time discretization is.", "",
          "### Time step", "", "| steps per segment | m [A m^2] | loss [J] | balance | balance_be | s |",
          "|---|---|---|---|---|---|"]
    for r in cv["time_step"]:
        L.append(f"| {r['steps']} | {r['m']:.5f} | {r['loss']:.5f} | {r['balance']:.1e} | {r['balance_be']:.1e} | "
                 f"{r['seconds']:.0f} |")
    ts = cv["time_step"]
    dm = max(abs(r["m"] / ts[-1]["m"] - 1) for r in ts[1:])
    dq = abs(ts[0]["loss"] / ts[-1]["loss"] - 1)
    L += ["", f"From 24 steps per segment on the moment is within {dm:.1e} of the finest run, and the total",
          f"loss changes by {dq:.1e} between 12 and 96 steps per segment.", ""]
    L += ["### Mesh", "",
          "The element size inside the cylinder is set by the region's `mesh_size`.  (`Mesh.MeshSizeMin` is a",
          "hard floor in gmsh; it is set below the smallest region size as well as below `mesh.surface_size`, so a",
          "smaller region size is not silently ignored.)", "",
          "| h [m] | elements | DOFs | m [A m^2] | loss [J] | balance_be | s |",
          "|---|---|---|---|---|---|---|"]
    for r in cv["mesh"]:
        L.append(f"| {r['h']:g} | {r['elements']:,} | {r['dofs']:,} | {r['m']:.5f} | {r['loss']:.5f} | "
                 f"{r['balance_be']:.1e} | {r['seconds']:.0f} |")
    L += ["", "### Finite-element order", "", "| order | DOFs | m [A m^2] | loss [J] | balance_be | s |",
          "|---|---|---|---|---|---|"]
    for r in rec["order"]:
        L.append(f"| {r['order']} | {r['dofs']:,} | {r['m']:.5f} | {r['loss']:.5f} | {r['balance_be']:.1e} | "
                 f"{r['seconds']:.0f} |")
    L += ["", "### E-J floor (`hts.e_floor`)", "",
          "The inverse power law J = Jc (|E|/E0)^(1/n) has an infinite slope at E = 0, so in the",
          "A-formulation the round-off of E inside a flux-free region would produce a current of order Jc.",
          "Below |E| = e_floor E0 the law is continued linearly (a minimum resistivity).  The results must",
          "not depend on the floor:", "",
          "| e_floor | m [A m^2] | loss [J] | max \\|J\\|/Jc | balance_be | s |", "|---|---|---|---|---|---|"]
    for r in cv["e_floor"]:
        L.append(f"| {r['e_floor']:g} | {r['m']:.5f} | {r['loss']:.5f} | {r['maxJ_over_Jc']:.4f} | "
                 f"{r['balance_be']:.1e} | {r['seconds']:.0f} |")
    L.append("")

    mg = rec["margins"]
    L += ["## Conductor margins on the solenoid template", "",
          "`solenoid` template (R = 1 m, H = 2 m, 1 MA-turns) with a NbTi winding at 4.2 K and a fill factor",
          "of 0.3 (superconductor area / winding area).  The load line scales the current and the field",
          "together; `short_sample.scale` is where it meets the critical surface.",
          "", "| case | J_sc [A/mm^2] | B at the worst point [T] | Jc [A/mm^2] | I/Ic | short-sample scale | T_cs [K] | margin [K] |",
          "|---|---|---|---|---|---|---|---|"]
    # the record also holds the margins at the cost metric's optimal size (library output); the page shows the
    # design and the requirement only
    for lab, key in (("as designed", "as_designed"), (f"at the requirement (x{mg['amplitude_scale']:.1f})", "at_requirement")):
        for name, v in mg[key].items():
            ll = v["load_line_fraction"]
            L.append(f"| {lab} [{name}] | {v['J_superconductor_A_m2'] / 1e6:.4g} | {v['B_peak_T']:.4g} | "
                     f"{v['Jc_A_m2'] / 1e6:.4g} | {ll:.4g} | {v['short_sample']['scale']:.4g} | "
                     f"{v['current_sharing_T_K']:.3f} | {v['temperature_margin_K']:.3f} |")
    L += ["", f"`cost.load_line_cap` = 0.8: at the requirement {mg['caps_at_requirement']}.", ""]

    pm = rec.get("poloidal_margin")
    if pm:
        ref = pm[-1]["J"]
        L += ["## Poloidal (toroidal-magnet) winding: current density and current crowding", "",
              "`toroid` template (R = 1 m, y = 0.6, H = 1 m, 1 MA-turns, NbTi, fill 0.3). The current density",
              "of a poloidal winding is J = grad(G)/r, not ampere-turns divided by the meridional area; along a",
              f"straight inner leg it is NI/(2 pi r t) = {ref:.4g} A/m^2. Where the winding path turns a sharp",
              "corner the current crowds and J is singular, so the peak grows under mesh refinement while the",
              "area average does not:", "",
              "| element size in the winding [m] | DOFs | J at the worst point [A/m^2] | area mean [A/m^2] | I/Ic | worst point (r, z) [m] |",
              "|---|---|---|---|---|---|"]
        for r in pm[:-1]:
            L.append(f"| {r['surface_size_m']:g} | {r['dofs']:,} | {r['J_worst']:.4g} | {r['J_mean']:.4g} | "
                     f"{r['load_line_fraction']:.5f} | ({r['worst_r']:.3f}, {r['worst_z']:.3f}) |")
        L += ["", "The run warns about this; `J_engineering_mean_A_m2` is the mesh-independent number and a",
              "fillet on the winding shape removes the singularity.", ""]
    cf = rec.get("core_fc")
    if cf:
        L += ["## Field-cooled consistency, and the two wall models", "",
              "CORE template (R_c = 0.6 m, cavity outer radius 1.2 m, 12 mm wall, 0.5 MA-turns) with the cavity",
              "as a critical-state region.  Held at constant drive after field cooling, the critical state must",
              "not move, and Psi must agree with the ideal (Meissner) result up to the difference between the",
              "two wall models: the ideal superconductor excludes the field from the wall volume, while the",
              "critical state lets it through a wall that carries no net current (an O(t/R) difference).", "",
              f"{cf['dofs']:,} DOFs.  Psi (ideal) = {cf['Psi_ideal']:.6e} H T m^2, L_p = {cf['L_p']:.4e} H.", "",
              "| state | Psi [H T m^2] | vs ideal | screening current [A] | max \\|J\\|/Jc | loss [J] | B_z(0) [T] |",
              "|---|---|---|---|---|---|---|"]
        hd, zf = cf["held"], cf["zfc_ramped_up"]
        L.append(f"| field cooled, held | {hd['Psi']:.6e} | {hd['rel_to_ideal']:+.4f} | "
                 f"{hd['screening_current_A']:.2e} | {hd['max_J_over_Jc']:.1e} | {hd['loss_J']:.1e} | "
                 f"{hd['Bz_centre_T']:.5f} |")
        L.append(f"| zero-field cooled, ramped to full drive | {zf['Psi']:.6e} | {zf['rel_to_ideal']:+.4f} | "
                 f"{zf['screening_current_A']:.4g} | {zf['max_J_over_Jc']:.2e} | {cf['zfc_loss_J']:.3g} | "
                 f"{zf['Bz_centre_T']:.5f} (applied {zf['Bz_applied_T']:.5f}) |")
        L += ["", "The field-cooled row is the static check: nothing moves (the screening current and the",
              "dissipation are at round-off) and the remaining difference in Psi is the wall model alone.  The",
              "zero-field-cooled row is a different magnetic history: the wall screens the bore, which changes",
              "the coupling completely - the baseline Psi of this design assumes a field-cooled cavity",
              "(`superconductors: {azimuthal: {net_current: 0}}`).  After the ramp-down the trapped field at the",
              f"centre is {cf['zfc_trapped']:.3e} T.", ""]
    sh = rec.get("shield")
    if sh:
        L += ["## Showcase: a REBCO tube shield inside a solenoid with a coaxial readout", "",
              "A 1 cm REBCO tube (homogenized, fill factor 0.5, Jc0 = 2e10 A/m^2 at 20 K, tape normal along r)",
              "between a solenoid and a shorted coaxial pickup.  The static solve keeps the ideal",
              "(Meissner) model for the tube; the critical-state analysis ramps the magnet up, holds and ramps",
              "it down.", "",
              f"{sh['dofs']:,} DOFs, {sh['hts_seconds']:.0f} s for {len(sh['rows']) - 1} steps; worst",
              f"backward-Euler energy identity {sh['balance_be']:.1e}.", "",
              "| t [s] | drive | screening current [A] | max \\|J\\|/Jc | SCIF at (0,0) [T] | Psi [H T m^2] | cumulative loss [J] |",
              "|---|---|---|---|---|---|---|"]
        for r in sh["rows"]:
            L.append(f"| {r['t']:.0f} | {r['scale']:.3f} | {r['I_screen']:.4g} | {r['maxJ_over_Jc']:.3f} | "
                     f"{r['scif'][0]:+.4e} | {r['Psi']:.5e} | {r['loss']:.4g} |")
        # field quality at the peak of the drive, where the static field quality is the quantity of interest
        # (hts.final is the end of the ramp-down, where the applied field is zero)
        pk = max(sh["rows"], key=lambda r: r["scale"])
        applied = [b - s for b, s in zip(pk["Bz"], pk["scif"])]

        def ptp(v):
            mean = sum(v) / len(v)
            return (max(v) - min(v)) / abs(mean) if mean else float("nan")

        fq = sh["final"].get("field_quality", {})
        L += ["", f"Psi with the ideal model {sh['psi']['Psi_ideal_SI_H_T_m2']:.6e}, with the critical state "
              f"{sh['psi']['Psi_critical_state_SI_H_T_m2']:.6e} ({sh['psi'].get('relative_change', 0):+.3%}).",
              "", f"Field quality over the five field-of-view points at the peak of the drive (t = {pk['t']:.0f} s): "
              f"peak-to-peak/mean {ptp(pk['Bz']):.4f} with screening currents versus {ptp(applied):.4f} for the "
              f"applied field alone, the mean field {sum(pk['Bz']) / len(pk['Bz']):.4e} T against an applied "
              f"{sum(applied) / len(applied):.4e} T: the tube screens the bore almost completely and what is "
              "left is the leakage through its ends.",
              "", f"At the end of the ramp-down the largest remaining (trapped) SCIF is "
              f"{fq.get('max_scif_Bz_T', float('nan')):.3e} T.", ""]
        if sh.get("ac_loss_per_cycle_J") is not None:
            L.append(f"Loss over the ramp-up/hold/ramp-down cycle: {sh['ac_loss_per_cycle_J']:.4g} J.")
            L.append("")

    L += ["## Known limitations", "",
          "* The screening currents are azimuthal: the model covers poloidal static fields (azimuthal",
          "  windings, axial background fields).  A toroidal static field (poloidal winding) is rejected.",
          "* Thin-film geometries are not validated against the Brandt / Clem-Sanchez thin-strip and",
          "  thin-disk results in this release.",
          "* Nonlinear iron is linearized at its initial permeability in the time-dependent solve, and",
          "  persistent circuits are not supported there (both are errors or warnings, never silent).",
          "* Other ideal superconductors in the same design keep their static constraint at every step.",
          "* Convergence is measured on the Newton update; the residual ratio is reported next to it and is",
          "  pessimistic because the power-law Jacobian spans many orders of magnitude.  The damped Newton",
          "  iteration converges linearly (the dissipation potential is nearly degenerate for large n), so",
          "  tens of iterations per step are normal.", ""]
    return "\n".join(L)


def main(argv):
    wd = Path(argv[1] if len(argv) > 1 else "/tmp/axf_hts").resolve()
    wd.mkdir(parents=True, exist_ok=True)
    rec = {"generated": time.strftime("%Y-%m-%d %H:%M"), "version": af.version()}
    rec["conductors"] = case_conductors()
    rec["margins"] = case_margins(wd)
    rec["bean_limit"] = case_bean_limit(wd)
    rec["magnetization_6"] = case_magnetization(wd, 0.6, "mag6")
    rec["magnetization_12"] = case_magnetization(wd, 1.2, "mag12")
    rec["trapped"] = case_trapped(wd)
    rec["ac_loss"] = case_ac_loss(wd)
    rec["convergence"] = case_convergence(wd)
    rec["order"] = case_order(wd)
    rec["poloidal_margin"] = case_poloidal_margin(wd)
    rec["core_fc"] = case_core_fc(wd)
    rec["shield"] = case_shield(wd)
    REC.parent.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(rec, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report(rec))
    print(f"wrote {REC} and {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
