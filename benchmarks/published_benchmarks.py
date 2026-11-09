"""Published benchmarks: published geometry factors and sensitivities of real detectors.

Builds axionFEM designs of ABRACADABRA-10 cm, DMRadio-50L, DMRadio-m^3,
DMRadio-GUT and DMRadio-Core from public information (every dimension and its
source in ``benchmarks/published/sources.json``), solves them, and compares our
coupling quantities (Psi, L_p, F = Psi^4/L_p^2, W_mag) and reach curves with the
published numbers.

    python benchmarks/published_benchmarks.py all
    python benchmarks/published_benchmarks.py abra10cm dmradio_50l
    python benchmarks/published_benchmarks.py designs     # write the design JSONs only
    python benchmarks/published_benchmarks.py report      # md + figures from the records

Records go to ``benchmarks/records/published_<name>.json``, designs to
``benchmarks/published/<name>.json``, the report to
``docs/validation/published_benchmarks.md`` and figures to
``docs/validation/figures/``.

Unit conventions and the conversions between the published coupling definitions
and ours are derived in the report; the code implements them in
:func:`V_B_from_Psi`, :func:`c_PU_from_F` and :func:`F_from_c_PU`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import axionfem as af  # noqa: E402
from axionfem import scan  # noqa: E402

MU0 = 1.25663706212e-6
REC = ROOT / "benchmarks" / "records"
DES = ROOT / "benchmarks" / "published"
RUNS = ROOT / "runs" / "published"
FIG = ROOT / "docs" / "validation" / "figures"
SRC = json.loads((DES / "sources.json").read_text())


# ------------------------------------------------------------------ bookkeeping
# What the pass flag of these checks does and does not mean.  It is printed with
# every result and stored in every record, because it has been misread before:
# a "[PASS]" here never asserts that axionFEM agrees with a publication.
PASS_MEANS = ("SELF-CHECK = internal consistency only: identities, convergence and "
              "agreement with our own closed forms.  It says NOTHING about agreement "
              "with the publication; that is the 'verdict' column of the summary table "
              "in docs/validation/published_benchmarks.md, and several rows there "
              "disagree while the self-check passes.")


def record(name: str, data: dict) -> dict:
    REC.mkdir(parents=True, exist_ok=True)
    data = {"check": f"published_{name}", "axionfem_version": af.version(),
            "date": time.strftime("%Y-%m-%d %H:%M"), "pass_means": PASS_MEANS, **data}
    (REC / f"published_{name}.json").write_text(json.dumps(data, indent=2))
    status = "PASS" if data.get("pass") else "FAIL"
    print(f"[SELF-CHECK {status}] {name}: {data.get('headline', '')}", flush=True)
    if data.get("pass_scope"):
        print(f"    self-check covers: {data['pass_scope']}", flush=True)
    return data


def save_design(d: af.Design, name: str) -> af.Design:
    DES.mkdir(parents=True, exist_ok=True)
    d.doc["name"] = name
    d.save(DES / f"{name}.json")
    return d


def rel(a: float, b: float) -> float:
    """Signed relative deviation of a from the reference b."""
    return float("nan") if b == 0 else (a - b) / b


# --------------------------------------------------- published <-> our quantities
def V_B_from_Psi(Psi_natural: float, B_max: float) -> float:
    """KST Eq. (8) effective volume V_B [m^3] from our Psi.  Psi_natural = B_max V_B."""
    return Psi_natural / B_max


def c_PU_from_F(F_natural: float, B0: float, V_PU: float) -> float:
    """DMRadio c_PU (M3 Eq. A5) from F = Psi^4/L_p^2 in natural units.

    F = 4 c_PU^4 B0^4 V_PU^(10/3)  =>  c_PU = F^(1/4) / (sqrt(2) B0 V_PU^(5/6)).
    """
    return F_natural ** 0.25 / (math.sqrt(2.0) * B0 * V_PU ** (5.0 / 6.0))


def F_from_c_PU(c_PU: float, B0: float, V_PU: float) -> float:
    return 4.0 * c_PU ** 4 * B0 ** 4 * V_PU ** (10.0 / 3.0)


def eta_A_at(f_Hz: float) -> float:
    """Amplifier noise eta(omega) of the DMRadio dc-SQUID chain, log-interpolated
    from the published optimum envelope (arXiv:2504.20398 Fig. 3; see sources.json,
    key dmradio_core/sensitivity_inputs/eta_A_table_MHz).  Clamped at the ends."""
    tab = SRC["dmradio_core"]["sensitivity_inputs"]["eta_A_table_MHz"][0]
    fs = np.array(sorted(float(k) for k in tab)) * 1e6
    es = np.array([tab[str(int(f / 1e6))] for f in fs])
    return float(np.interp(f_Hz, fs, es))


def eta_A_band(f1: float, f2: float, n: int = 21) -> float:
    """Band representative of eta(omega): the value whose constant use gives the same
    scan time as the real eta(omega) under the log-uniform allocation, i.e. the
    log-average (the time per ln(nu) is proportional to eta)."""
    fs = np.exp(np.linspace(math.log(f1), math.log(f2), n))
    return float(np.exp(np.mean(np.log([eta_A_at(f) for f in fs]))))


def scan_time_yr(F_natural: float, benchmark: dict, **scan_kw) -> float:
    """Live time [yr] for which F_req = F, i.e. the time to reach the target
    coupling over the whole band (target allocation).  F_req ~ 1/t exactly."""
    req = af.requirement(benchmark, live_time=1.0, **scan_kw)
    return req["F_req_natural_T4_m10"] / F_natural


# ============================================================ semi-analytic refs
def a_phi_hat(s, z, r):
    """A_phi/(mu0 I) of a circular filament loop of radius r at the point (s, z)."""
    from scipy.special import ellipe, ellipk
    m = 4 * r * s / ((r + s) ** 2 + z ** 2)
    return (1.0 / (np.pi * np.sqrt(m))) * np.sqrt(r / s) * ((1 - m / 2) * ellipk(m) - ellipe(m))


def Psi_loop_in_toroid(r: float, R: float, a: float, h: float) -> float:
    """Exact free-space Psi/B_max [m^3] for a filament pickup loop of radius r on
    the axis of a rectangular-cross-section toroid (R < s < R + a, |z| < h/2)
    whose field is B_max R/s phi-hat:

        Psi/B_max = 2 pi R int ds int dz A_phi(s, z)/(mu0 I).

    This is the reciprocity integral of docs/physics/scan_and_signal.md (S1.1)
    evaluated with the exact elliptic-integral vector potential.  Independent of
    the FEM; also shown in the report to equal KST Eq. (8) identically.
    """
    from scipy import integrate
    val, _ = integrate.dblquad(lambda z, s: a_phi_hat(s, z, r), R, R + a, -h / 2, h / 2,
                               epsabs=1e-14, epsrel=1e-11)
    return 2 * np.pi * R * val


def V_B_KST_eq8(r: float, R: float, a: float, h: float) -> float:
    """KST (arXiv:1602.01086) Eq. (8), evaluated exactly as written."""
    from scipy import integrate

    def f(th, s, rp):
        rt2 = s * s + rp * rp - 2 * s * rp * np.cos(th)
        return R * h * rp * (s - rp * np.cos(th)) / (rt2 * np.sqrt(h * h + 4 * rt2))

    val, _ = integrate.tplquad(f, 0, r, lambda rp: R, lambda rp: R + a,
                               lambda rp, s: 0.0, lambda rp, s: 2 * np.pi,
                               epsabs=1e-13, epsrel=1e-10)
    return val


def L_thin_ring(r: float, a: float) -> float:
    """Self inductance [H] of a circular ring, loop radius r, wire radius a,
    current on the surface (superconducting / high frequency): mu0 r (ln 8r/a - 2).
    KST Eq. (10)."""
    return MU0 * r * (math.log(8 * r / a) - 2.0)


def M_coplanar_rings(r1: float, r2: float) -> float:
    """Mutual inductance [H] of two coaxial coplanar filament rings."""
    from scipy.special import ellipe, ellipk
    m = 4 * r1 * r2 / (r1 + r2) ** 2
    k = math.sqrt(m)
    return MU0 * math.sqrt(r1 * r2) * ((2 / k - k) * ellipk(m) - (2 / k) * ellipe(m))


# ==================================================================== designs
def design_abra10cm(tw: float = 2e-4, shield_radius: float | None = None,
                    calibration_loop: bool = False, mesh_size: float = 0.004,
                    surface_size: float = 2e-4, order: int = 2) -> af.Design:
    """ABRACADABRA-10 cm: counter-wound toroid (rectangular cross-section
    3 cm < r < 6 cm, |z| < 6 cm) with a single circular superconducting pickup
    loop of radius 2 cm and wire radius 0.5 mm in the bore.

    The counter-winding (ABRA10 'counter-wound to cancel azimuthal currents') is
    exactly our ``poloidal`` winding: a purely toroidal field, no net azimuthal
    current.  ``tw`` is the winding sheet thickness (not published); the field
    box is the published envelope and the winding sits just outside it, so the
    tw -> 0 limit is the idealised toroid of KST Eq. (8).
    """
    g = SRC["abra10cm"]["geometry"]
    regions = [
        {"name": "winding",
         "shape": {"frame": {"r": ["Rin-tw", "Rout+tw"], "z": ["-H/2-tw", "H/2+tw"],
                             "thickness": "tw"}},
         "material": "winding", "notes": "counter-wound toroid; poloidal winding = zero net azimuthal current"},
        {"name": "loop", "shape": {"disk": {"center": ["rp", 0.0], "radius": "aw"}},
         "material": "nb", "notes": "NbTi pickup loop, 1 mm wire diameter"}]
    scs = [{"body": "loop", "azimuthal_slit": True,
            "notes": "the loop carries the readout current around the axis"}]
    windings = [{"name": "toroid", "region": "winding", "type": "poloidal", "ampere_turns": "NI"}]
    if shield_radius:
        regions.append({"name": "shield",
                        "shape": {"frame": {"r": [0.0, "Rsh"], "z": ["-Rsh", "Rsh"], "thickness": "tsh"}},
                        "material": "nb",
                        "notes": "tin-coated copper shell, superconducting below 3.7 K; size not published"})
        scs.append({"body": "shield", "azimuthal": {"net_current": 0}, "poloidal": {"net_current": 0}})
    if calibration_loop:
        regions.append({"name": "cal", "shape": {"disk": {"center": ["rcal", 0.0], "radius": "aw"}},
                        "material": "winding",
                        "notes": "calibration wire at r = 4.5 cm, modelled as a one-turn loop"})
        windings.append({"name": "calibration", "region": "cal", "type": "azimuthal", "ampere_turns": 1.0})
    doc = {
        "schema": "axionfem.design/1", "name": "abra10cm",
        "description": "ABRACADABRA-10 cm (arXiv:1810.12257): toroidal magnet with a 2 cm "
                       "superconducting pickup loop in the bore.",
        "parameters": {"Rin": g["R_in_m"][0], "Rout": g["R_out_m"][0], "H": g["H_m"][0],
                       "tw": tw, "rp": g["pickup_loop_radius_m"][0], "aw": g["pickup_wire_radius_m"][0],
                       "rcal": g["calibration_loop_radius_m"][0],
                       "NI": g["turns"][0] * g["current_A"][0],
                       "Rsh": shield_radius or 0.09, "tsh": 1e-3},
        "regions": regions, "windings": windings, "superconductors": scs,
        "pickup": {"type": "azimuthal", "conductor": "loop"},
        "mesh": {"order": order, "size": mesh_size, "surface_size": surface_size},
        "notes": "Dimensions and sources: benchmarks/published/sources.json, key 'abra10cm'.",
    }
    return af.Design(doc)


def design_dmradio_50l(scale: float = 1.0, mesh_size: float | None = None,
                       clearance_scale: float = 1.0, order: int = 2) -> af.Design:
    """DMRadio-50L: toroid template (magnet + slit superconducting sheath,
    azimuthal pickup) with the Patras-2023 talk dimensions."""
    g = SRC["dmradio_50l"]["geometry"]
    a, b, H = g["R_in_m"][0] * scale, g["R_out_m"][0] * scale, g["H_m"][0] * scale
    NI = g["turns"][0] * g["current_A"][0] * scale  # keeps B_peak = mu0 NI/(2 pi a) fixed
    kw = {"R": b, "y": a / b, "H": H, "NI": NI,
          "tw": g["winding_thickness_m"][0] * scale,
          "g": g["sheath_clearance_m"][0] * scale * clearance_scale,
          "ts": g["sheath_thickness_m"][0] * scale,
          "f": 0.02 * b}
    d = af.template("toroid", **kw)
    if mesh_size:
        d.doc["mesh"]["size"] = mesh_size
    d.doc["mesh"]["order"] = order
    d.doc["description"] = ("DMRadio-50L: ~50 L toroidal magnet in a slit superconducting pickup "
                            "sheath.  Dimensions from the public Patras-2023 status talk.")
    d.doc["notes"] = "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_50l'."
    return d


def design_dmradio_gut(mesh_size: float | None = None, order: int = 2,
                       tw_factor: float = 1.0, h_over_R_in: float | None = None) -> af.Design:
    """DMRadio-GUT: the 50L toroid scaled uniformly to V = 10 m^3 at B_peak = 16 T.

    ``tw_factor`` scales the (unpublished) winding-sheet thickness; ``h_over_R_in``
    replaces the (unpublished) height-to-inner-radius ratio, keeping V = 10 m^3,
    the aspect ratio and B_peak = 16 T fixed.  Both default to the nominal design,
    so the saved ``dmradio_gut.json`` is unaffected.
    """
    g = SRC["dmradio_gut"]["geometry"]
    g50 = SRC["dmradio_50l"]["geometry"]
    a, b, H = g["R_in_m"][0], g["R_out_m"][0], g["H_m"][0]
    if h_over_R_in is not None:
        ar = b / a
        a = (g["V_m3"][0] / (math.pi * (ar ** 2 - 1) * h_over_R_in)) ** (1.0 / 3.0)
        b, H = ar * a, h_over_R_in * a
    s = a / g50["R_in_m"][0]
    NI = 2 * math.pi * a * g["B_peak_T"][0] / MU0
    d = af.template("toroid", R=b, y=a / b, H=H, NI=NI,
                    tw=g50["winding_thickness_m"][0] * s * tw_factor,
                    g=g50["sheath_clearance_m"][0] * s,
                    ts=g50["sheath_thickness_m"][0] * s, f=0.02 * b)
    if mesh_size:
        d.doc["mesh"]["size"] = mesh_size
    d.doc["mesh"]["order"] = order
    d.doc["description"] = ("DMRadio-GUT: toroid with the DMRadio-50L aspect ratio scaled to "
                            "V = 10 m^3 and B_peak = 16 T (573 MJ).")
    d.doc["notes"] = "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_gut'."
    return d


def design_dmradio_m3(NI: float, h_coax: float | None = None, mesh_size: float | None = None,
                      order: int = 2, copper_loss: bool = False, ri: float | None = None,
                      ro: float | None = None) -> af.Design:
    """DMRadio-m^3: the published solenoid (1.45 m bore, 1.72 m OD, 1.35 m tall)
    with the published coaxial pickup (r_i = 0.336 m, r_o = 0.628 m, one of six
    heights), shorted at the bottom and read out across the annular gap at the
    open top.  Geometry from arXiv:2302.14084; the wall thickness, the port gap
    and the coax's axial position are the only chosen quantities.
    """
    g = SRC["dmradio_m3"]["geometry"]
    ri = g["coax_inner_radius_m"][0] if ri is None else ri
    ro = g["coax_outer_radius_m"][0] if ro is None else ro
    h = g["coax_heights_m"][0][0] if h_coax is None else h_coax
    tc, tp = g["coax_wall_m"][0], g["port_gap_m"][0]
    Rw, H = g["bore_diameter_m"][0] / 2, g["H_m"][0]
    tw = g["outer_diameter_m"][0] / 2 - Rw
    mats = {}
    if copper_loss:
        # normal-skin-effect copper; the real pickup is OFHC copper at ~20 mK, where
        # M3_EM quotes a Q enhancement of 6-11 over this room-temperature value.
        mats["copper_rf"] = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.68e-8,
                                                         "T_K": 293.0}}
    cu = "copper_rf" if copper_loss else "copper"
    doc = {
        "schema": "axionfem.design/1",
        "name": f"dmradio_m3_h{h:g}".replace(".", "p"),
        "description": (f"DMRadio-m^3: published solenoid with the published coaxial pickup "
                        f"(r_i = {ri} m, r_o = {ro} m, h = {h} m), shorted at the bottom, read out "
                        "across the annular gap at the open top.  Bucking coils and the segmented "
                        "magnet are not modelled."),
        "parameters": {"Rw": Rw, "tw": tw, "H": H, "ri": ri, "ro": ro, "h": h, "tc": tc,
                       "tp": tp, "NI": NI},
        "regions": [
            {"name": "winding", "shape": {"rect": {"r": ["Rw", "Rw+tw"], "z": ["-H/2", "H/2"]}},
             "material": "winding"},
            {"name": "coax_inner", "shape": {"rect": {"r": ["ri-tc", "ri"], "z": ["-h/2", "h/2"]}},
             "material": cu},
            {"name": "coax_outer", "shape": {"rect": {"r": ["ro", "ro+tc"], "z": ["-h/2", "h/2"]}},
             "material": cu},
            {"name": "coax_short", "shape": {"rect": {"r": ["ri-tc", "ro+tc"], "z": ["-h/2", "-h/2+tc"]}},
             "material": cu, "notes": "the shorted end"},
            {"name": "port", "shape": {"rect": {"r": ["ri", "ro"], "z": ["h/2-tp", "h/2"]}},
             "material": "vacuum", "notes": "the open top: the readout gap"}],
        "windings": [{"name": "solenoid", "region": "winding", "type": "azimuthal",
                      "ampere_turns": "NI"}],
        "pickup": {"type": "poloidal", "port": "port"},
        "mesh": {"order": order, "size": mesh_size or 0.03, "surface_size": tc},
        "notes": "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_m3'.",
    }
    if mats:
        doc["materials"] = mats
    return af.Design(doc)


def design_core_magnet(NI: float, iron: bool = False, order: int = 2) -> af.Design:
    """DMRadio-Core magnet only: 37 cm bore, 60 cm OD, 2.95 m tall, published
    5 T peak and 4.3 MJ.  The 9-fold segmentation and the 3 soft-iron segments
    are not modelled (positions not published)."""
    g = SRC["dmradio_core"]["geometry"]
    doc = {"schema": "axionfem.design/1", "name": "dmradio_core_magnet",
           "description": "DMRadio-Core magnet: uniform-current solenoid with the published envelope.",
           "parameters": {"Rc": g["bore_diameter_m"][0] / 2, "Ro": g["outer_diameter_m"][0] / 2,
                          "H": g["H_m"][0], "NI": NI},
           "regions": [{"name": "coil", "shape": {"rect": {"r": ["Rc", "Ro"], "z": ["-H/2", "H/2"]}},
                        "material": "winding"}],
           "windings": [{"name": "solenoid", "region": "coil", "type": "azimuthal", "ampere_turns": "NI"}],
           "mesh": {"order": order, "size": 0.05},
           "notes": "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_core'."}
    return af.Design(doc)


def design_dmradio_core(NI: float, outer_diameter: float = 1.20, order: int = 2) -> af.Design:
    """DMRadio-Core, one pickup segment, with the CORE template: a split solenoid
    inside a superconducting toroidal cavity read out across a neck gap.  The
    published pickup radii and height are used; the real pickup's copper plates
    crossing the bore and its 30-degree azimuthal port are not axisymmetric and
    are not represented (see the report)."""
    g = SRC["dmradio_core"]["geometry"]
    Rc = g["bore_diameter_m"][0] / 2
    tw = g["outer_diameter_m"][0] / 2 - Rc
    a = g["pickup_inner_diameter_m"][0] / 2
    b = outer_diameter / 2
    w = 0.005
    d = af.template("core", Rc=Rc, tw=tw, H=g["H_m"][0], gc=g["plate_spacing_m"][0],
                    a=a, b=b, h=g["pickup_height_m"][0], w=w, f=0.02,
                    gn=g["plate_spacing_m"][0], fl=0.3 * w, NI=NI)
    d.doc["mesh"]["order"] = order
    d.doc["description"] = (f"DMRadio-Core pickup segment (outer diameter {outer_diameter * 100:.0f} cm) "
                            "around the published 5 T magnet, CORE topology.")
    d.doc["notes"] = "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_core'."
    return d


# ==================================================================== checks
def check_abra10cm() -> dict:
    """ABRACADABRA-10 cm: the geometric factor G and the pickup inductance."""
    g = SRC["abra10cm"]["geometry"]
    pub = SRC["abra10cm"]["published_quantities"]
    Rin, Rout, H = g["R_in_m"][0], g["R_out_m"][0], g["H_m"][0]
    rp, aw = g["pickup_loop_radius_m"][0], g["pickup_wire_radius_m"][0]
    NI = g["turns"][0] * g["current_A"][0]
    B_in = MU0 * NI / (2 * math.pi * Rin)          # field at the inner radius from N I
    V_rect = math.pi * (Rout ** 2 - Rin ** 2) * H  # the rectangular annulus
    V_pub = pub["V_field_m3"][0]                   # the published "volume of magnetic field"
    V_tor = V_pub                                  # the denominator of the published G_V
    out: dict = {"inputs": {"R_in_m": Rin, "R_out_m": Rout, "H_m": H, "NI_A": NI,
                            "B_in_from_NI_T": B_in, "B_in_quoted_T": g["B_max_T"][0],
                            "V_field_published_m3": V_pub, "V_toroid_rect_m3": V_rect,
                            "V_published_over_rect": V_pub / V_rect,
                            "pickup_radius_m": rp, "pickup_wire_radius_m": aw}}

    # ---- 1. the published V_B definition is our reciprocity integral
    kst = {"r": 0.85, "R": 0.85, "a": 0.85, "h": 2.55, "V_B_quoted_m3": 1.0}
    kst["V_B_eq8_m3"] = V_B_KST_eq8(kst["r"], kst["R"], kst["a"], kst["h"])
    kst["Psi_over_Bmax_exact_m3"] = Psi_loop_in_toroid(kst["r"], kst["R"], kst["a"], kst["h"])
    kst["eq8_over_exact"] = kst["V_B_eq8_m3"] / kst["Psi_over_Bmax_exact_m3"]
    kst["rel_to_quoted"] = rel(kst["V_B_eq8_m3"], 1.0)
    out["kst_benchmark"] = kst

    eq8 = V_B_KST_eq8(rp, Rin, Rout - Rin, H)
    exact = Psi_loop_in_toroid(rp, Rin, Rout - Rin, H)
    out["semi_analytic"] = {"V_B_eq8_m3": eq8, "Psi_over_Bmax_exact_m3": exact,
                            "eq8_over_exact": eq8 / exact,
                            "G_free_space": exact / V_tor,
                            "Psi_natural_T_m3": exact * B_in}

    # ---- 2. FEM, free space: convergence in the winding sheet thickness and the mesh
    rows = []
    for tw in (1e-3, 5e-4, 2e-4, 1e-4):
        d = design_abra10cm(tw=tw)
        r = d.solve(RUNS / f"abra10_tw{tw:g}", verbose=0)
        VB = V_B_from_Psi(r.coupling["Psi_natural_T_m3"], B_in)
        rows.append({"winding_thickness_m": tw, "mesh_size_m": 0.004, "order": 2,
                     "Psi_natural_T_m3": r.coupling["Psi_natural_T_m3"], "V_B_m3": VB,
                     "G": VB / V_tor, "L_p_H": r.L_p, "dofs": r.data["mesh"]["dofs"],
                     "rel_to_exact": rel(VB, exact), "warnings": r.data.get("warnings", [])})
    out["winding_thickness_convergence"] = rows

    mrows = []
    for size, ssize, order in ((0.008, 4e-4, 2), (0.004, 2e-4, 2), (0.002, 1e-4, 2), (0.004, 2e-4, 3)):
        d = design_abra10cm(tw=2e-4, mesh_size=size, surface_size=ssize, order=order)
        r = d.solve(RUNS / f"abra10_h{size:g}_p{order}", verbose=0)
        VB = V_B_from_Psi(r.coupling["Psi_natural_T_m3"], B_in)
        mrows.append({"mesh_size_m": size, "surface_size_m": ssize, "order": order,
                      "V_B_m3": VB, "G": VB / V_tor, "L_p_H": r.L_p,
                      "dofs": r.data["mesh"]["dofs"], "rel_to_exact": rel(VB, exact),
                      "L_p_rel_to_thin_ring": rel(r.L_p, L_thin_ring(rp, aw)),
                      "warnings": r.data.get("warnings", [])})
    out["mesh_convergence"] = mrows
    # The headline FEM row is the MOST REFINED one (most DOFs), not the one that
    # happens to sit closest to the semi-analytic reference: choosing by
    # min|rel_to_exact| would select the coarsest mesh here and would hide the
    # +0.49 % bias of the finite winding-sheet thickness behind mesh error.
    best = max(mrows, key=lambda x: x["dofs"])
    out["headline_row"] = {
        "selected_by": "most DOFs (the most refined mesh)",
        "mesh_size_m": best["mesh_size_m"], "order": best["order"], "dofs": best["dofs"],
        "note": "Selected by DOFs, not by min|rel_to_exact| (which would pick the coarsest row).  "
                "Every FEM row here carries the +0.49 % bias of the t_w = 0.2 mm winding "
                "sheet; the t_w -> 0 reference is the semi-analytic value, which is what "
                "the report quotes as the headline G.",
    }

    # ---- 3. the superconducting shield (radius not published)
    srows = []
    for Rsh in (0.075, 0.09, 0.12, 0.20):
        d = design_abra10cm(tw=2e-4, shield_radius=Rsh)
        r = d.solve(RUNS / f"abra10_sh{Rsh:g}", verbose=0)
        VB = V_B_from_Psi(r.coupling["Psi_natural_T_m3"], B_in)
        srows.append({"shield_radius_m": Rsh, "V_B_m3": VB, "G": VB / V_tor, "L_p_H": r.L_p,
                      "V_B_over_free_space": VB / exact, "dofs": r.data["mesh"]["dofs"],
                      "warnings": r.data.get("warnings", [])})
    out["shield_scan"] = srows

    # ---- 4. inductances
    L_ring = L_thin_ring(rp, aw)
    rcal, acal = g["calibration_loop_radius_m"][0], g["calibration_wire_radius_m"][0]
    out["inductance"] = {
        "L_p_FEM_H": best["L_p_H"], "L_p_thin_ring_H": L_ring,
        "L_p_FEM_over_thin_ring": rel(best["L_p_H"], L_ring),
        "L_p_published_H": pub["L_p_H"][0], "L_p_FEM_over_published": rel(best["L_p_H"], pub["L_p_H"][0]),
        "L_cal_thin_ring_H": L_thin_ring(rcal, acal), "L_cal_published_H": pub["L_cal_H"][0],
        "L_cal_rel_to_published": rel(L_thin_ring(rcal, acal), pub["L_cal_H"][0]),
        "L_min_KST_eq11_H": MU0 * math.pi * Rin ** 2 / H,
    }

    # ---- 5. calibration-loop mutual inductance (published ~50 nH)
    # M = (flux through a circle of radius r_cal at z = 0) per ampere of pickup current
    #   = 2 pi * ppsi(r_cal, 0), where ppsi is the pickup mode's flux function.
    # By reciprocity this is also the pickup flux per ampere of calibration current.
    rcal = g["calibration_loop_radius_m"][0]
    M_fil = M_coplanar_rings(rp, rcal)
    Ms = {}
    for tag, Rsh in (("free_space", None), ("shield_0.09m", 0.09)):
        d = design_abra10cm(tw=2e-4, shield_radius=Rsh)
        rr = d.solve(RUNS / f"abra10_M_{tag}", verbose=0)
        Ms[tag] = 2 * math.pi * float(rr.sample([rcal], [0.0])["pickup_psi"][0])
    out["calibration_coupling"] = {
        "calibration_radius_m": rcal,
        "M_FEM_free_space_H": Ms["free_space"], "M_FEM_shield_0.09m_H": Ms["shield_0.09m"],
        "M_coplanar_filaments_H": M_fil,
        "M_FEM_rel_to_filaments": rel(Ms["free_space"], M_fil),
        "M_published_H": pub["M_cal_H"][0],
        "rel_to_published": rel(Ms["free_space"], pub["M_cal_H"][0]),
        "M_PRL_quoted_H": 5.0e-08,
        "note": "The instrument paper fixes the path: 'a 9 cm diameter loop concentric with and in "
                "the same plane as the pickup loop', and quotes M_CP = 19.3 nH from both an analytic "
                "calculation and COMSOL.  The PRL's '~50 nH' for the same quantity disagrees with "
                "its own instrument paper; 19.3 nH is the value consistent with the stated geometry.",
    }

    # ---- 6. comparison with the published G_V = V_B/V, and what would reconcile them
    G_fem = best["G"]
    G_pub = pub["G_V"][0]
    V_B_pub = pub["V_B_m3"][0]
    lo, hi = rp, 0.03
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if Psi_loop_in_toroid(mid, Rin, Rout - Rin, H) < V_B_pub:
            lo = mid
        else:
            hi = mid
    # Field regions whose volume IS the published 890 cm^3, reached three ways from
    # the "minimum 3 cm / maximum 6 cm / maximum 12 cm" envelope.  Shrinking the box
    # to the published volume makes the V_B disagreement WORSE, not better, so the
    # 890 cm^3 cannot be explained as a smaller field region of this shape.
    fv = []
    H_s = V_pub / (math.pi * (Rout ** 2 - Rin ** 2))
    Rout_s = math.sqrt(V_pub / (math.pi * H) + Rin ** 2)
    Rin_s = math.sqrt(Rout ** 2 - V_pub / (math.pi * H))
    for lab, (r0, r1, h) in (("shorter: R 3-6 cm, H reduced", (Rin, Rout, H_s)),
                             ("narrower: R_out reduced, H = 12 cm", (Rin, Rout_s, H)),
                             ("bored out: R_in raised, H = 12 cm", (Rin_s, Rout, H))):
        v = Psi_loop_in_toroid(rp, r0, r1 - r0, h)
        fv.append({"variant": lab, "R_in_m": r0, "R_out_m": r1, "H_m": h,
                   "V_field_m3": math.pi * (r1 ** 2 - r0 ** 2) * h,
                   "V_B_m3": v, "G": v / V_pub, "rel_to_published_V_B": rel(v, V_B_pub)})
    out["field_volume_variants"] = {
        "rows": fv,
        "note": "Semi-analytic (t_w -> 0) V_B for field boxes of exactly the published "
                "890 cm^3.  All three are FURTHER below the published V_B than the full "
                "3-6 cm x 12 cm envelope, so a field region of 890 cm^3 does not reconcile "
                "the published G_V = 0.027 with ours -- it widens the gap.",
    }

    out["geometric_factor"] = {
        "V_B_FEM_m3": best["V_B_m3"], "V_B_semi_analytic_m3": exact,
        "V_B_published_m3": V_B_pub, "V_B_rel_to_published": rel(best["V_B_m3"], V_B_pub),
        "G_V_FEM": G_fem, "G_V_published": G_pub, "rel_to_published": rel(G_fem, G_pub),
        "G_V_FEM_if_V_were_rectangular": best["V_B_m3"] / V_rect,
        "rel_to_published_if_V_rectangular": rel(best["V_B_m3"] / V_rect, G_pub),
        # the headline: the winding-sheet thickness is not published, so the t_w -> 0
        # limit (= the semi-analytic free-space integral) is the number to quote.
        "G_V_tw_zero": exact / V_tor,
        "G_V_tw_zero_if_V_were_rectangular": exact / V_rect,
        "V_B_tw_zero_rel_to_published": rel(exact, V_B_pub),
        "tw_bias_of_FEM_headline_row": rel(best["V_B_m3"], exact),
        "G_V_FEM_with_shield_0.09m": srows[1]["G"],
        "what_would_reconcile": {
            "pickup_radius_m_for_V_B_pub": 0.5 * (lo + hi),
            "pickup_radius_published_m": rp,
            "field_volume_m3_for_G_pub": best["V_B_m3"] / G_pub},
        "Psi_natural_T_m3_ours": best["V_B_m3"] * B_in,
        "Psi_natural_T_m3_published": V_B_pub * g["B_max_T"][0],
        # Psi mixes two differences and must NOT be labelled with the V_B deviation:
        # ours uses B_max = mu0 N I / 2 pi R_in = 1.0325 T, the paper's uses its own
        # quoted 1 T, so the Psi ratio is the V_B ratio times 1.0325.
        "Psi_rel_to_published": rel(best["V_B_m3"] * B_in, V_B_pub * g["B_max_T"][0]),
        "Psi_B_max_convention_factor": B_in / g["B_max_T"][0],
        "note": "The published G_V is V_B/V with V = 890 cm^3, the paper's 'total volume of magnetic "
                "field', which is 12.6 % below the rectangular annulus (1017.9 cm^3).  V_B itself is "
                "the quantity a FEM computes, so V_B is the primary comparison and G_V follows.  "
                "The published V_B is 0.027 * 890 cm^3 = 24.03 cm^3; it is NOT 0.027 * 1017.9 cm^3 "
                "= 27.5 cm^3, and the -22 % that comparison gave is an artefact of the wrong V.  "
                "Psi_rel_to_published differs from V_B_rel_to_published by the B_max convention "
                "(+3.25 %): use V_B, or G_V, for the geometric comparison.",
    }

    # ---- 7. what our Psi implies for the published limit (g_95 ~ 1/Psi)
    si = SRC["abra10cm"]["sensitivity_inputs"]
    out["limit_rescaling"] = {
        "note": "ABRA-10cm used a broadband (untuned) readout, so its limit cannot be reproduced "
                "from the resonant optimal-scan machinery (its flux-noise floor, 0.6 uPhi_0^2/Hz, is "
                "published but the measured PSD is not).  At fixed data g_95 ~ 1/Psi, so our V_B "
                "rescales the published limit by V_B_published/V_B_ours.",
        "g_scale_free_space": G_pub / G_fem,
        "g_scale_shield_0.09m": G_pub / srows[1]["G"],
        "g_limit_best_published_GeV": si["g_limit_best_GeV"][0],
        "g_limit_best_rescaled_GeV": si["g_limit_best_GeV"][0] * G_pub / G_fem,
        "g_limit_worst_published_GeV": si["g_limit_worst_GeV"][0],
        "g_limit_worst_rescaled_GeV": si["g_limit_worst_GeV"][0] * G_pub / G_fem,
    }

    # ---- 8. the coupling in DMRadio language, for comparison with 50L
    F_nat = best["V_B_m3"] ** 4 * B_in ** 4 / (best["L_p_H"] / MU0) ** 2
    out["dmradio_language"] = {"F_natural_T4_m10": F_nat, "V_field_m3": V_tor,
                               "c_PU": c_PU_from_F(F_nat, B_in, V_tor)}

    ok = (abs(kst["eq8_over_exact"] - 1) < 1e-6 and abs(kst["rel_to_quoted"]) < 0.01
          and abs(best["rel_to_exact"]) < 0.01 and abs(out["inductance"]["L_p_FEM_over_thin_ring"]) < 0.01)
    save_design(design_abra10cm(tw=2e-4), "abra10cm")
    save_design(design_abra10cm(tw=2e-4, shield_radius=0.09), "abra10cm_shielded")
    return record("abra10cm", {
        **out, "pass": bool(ok),
        "pass_scope": ("KST Eq. (8) == our reciprocity integral; KST's own quoted V_B; the FEM "
                       "within 1 % of the semi-analytic free-space integral; L_p within 1 % of the "
                       "thin-ring formula.  It does NOT test agreement with the published G_V, "
                       "which disagrees by ~10 %."),
        "headline": (f"G_V = {exact / V_tor:.5f} (t_w -> 0) vs published 0.027 "
                     f"({rel(exact, V_B_pub) * 100:+.1f} %); V_B = {exact * 1e6:.2f} cm^3 "
                     f"(FEM at the finest mesh {best['V_B_m3'] * 1e6:.2f}, +0.49 % from t_w = 0.2 mm) "
                     f"vs published {V_B_pub * 1e6:.2f} = 0.027 x 890 cm^3; "
                     f"L_p = {best['L_p_H'] * 1e9:.1f} nH vs 95.5 "
                     f"({out['inductance']['L_p_FEM_over_published'] * 100:+.1f} %); "
                     f"M_cal = {out['calibration_coupling']['M_FEM_free_space_H'] * 1e9:.1f} nH vs 19.3 "
                     f"({out['calibration_coupling']['rel_to_published'] * 100:+.1f} %)"),
    })


def check_dmradio_50l() -> dict:
    """DMRadio-50L: coupling, inductance and the projected reach."""
    g = SRC["dmradio_50l"]["geometry"]
    si = SRC["dmradio_50l"]["sensitivity_inputs"]
    a, b, H = g["R_in_m"][0], g["R_out_m"][0], g["H_m"][0]
    NI = g["turns"][0] * g["current_A"][0]
    B_in = MU0 * NI / (2 * math.pi * a)
    V = math.pi * (b ** 2 - a ** 2) * H
    B_rms = B_in * math.sqrt(2 * a ** 2 * math.log(b / a) / (b ** 2 - a ** 2))
    out: dict = {"inputs": {"R_in_m": a, "R_out_m": b, "H_m": H, "turns": g["turns"][0],
                            "current_A": g["current_A"][0], "NI_A": NI,
                            "B_in_from_NI_T": B_in, "B_in_quoted_T": g["B_max_T"][0],
                            "B_rms_T": B_rms, "B_rms_over_peak": B_rms / B_in,
                            "V_field_m3": V, "V_field_L": V * 1e3}}

    rows = []
    for size in (0.016, 0.010, 0.006, 0.004):
        d = design_dmradio_50l(mesh_size=size)
        r = d.solve(RUNS / f"50l_h{size:g}", verbose=0)
        c = r.coupling
        rows.append({"mesh_size_m": size, "order": 2, "Psi_natural_T_m3": c["Psi_natural_T_m3"],
                     "L_p_H": r.L_p, "L_p_natural_m": r.L_p / MU0, "W_mag_J": c["W_mag_J"],
                     "F_natural_T4_m10": c["F_natural_T4_m10"], "g_factor": c["g_factor"],
                     "c_PU": c_PU_from_F(c["F_natural_T4_m10"], B_in, V),
                     "L_p_from_flux_H": r.data.get("pickup", {}).get("L_p_from_flux_H"),
                     "dofs": r.data["mesh"]["dofs"], "warnings": r.data.get("warnings", [])})
    out["mesh_convergence"] = rows
    fine = rows[-1]
    out["mesh_convergence_summary"] = {
        "Psi_rel_change_last_refinement": rel(rows[-1]["Psi_natural_T_m3"], rows[-2]["Psi_natural_T_m3"]),
        "L_p_rel_change_last_refinement": rel(rows[-1]["L_p_H"], rows[-2]["L_p_H"]),
        "F_rel_change_last_refinement": rel(rows[-1]["F_natural_T4_m10"], rows[-2]["F_natural_T4_m10"]),
    }

    # sensitivity to the unpublished sheath clearance
    crows = []
    for cs in (0.5, 1.0, 2.0, 5.0):
        d = design_dmradio_50l(clearance_scale=cs, mesh_size=0.006)
        r = d.solve(RUNS / f"50l_g{cs:g}", verbose=0)
        c = r.coupling
        crows.append({"clearance_scale": cs, "clearance_m": g["sheath_clearance_m"][0] * cs,
                      "Psi_natural_T_m3": c["Psi_natural_T_m3"], "L_p_H": r.L_p,
                      "F_natural_T4_m10": c["F_natural_T4_m10"],
                      "c_PU": c_PU_from_F(c["F_natural_T4_m10"], B_in, V)})
    out["sheath_clearance_scan"] = crows

    # scale invariance of c_PU (a property of the definition; also a code check)
    srows = []
    for s in (0.5, 1.0, 2.0):
        d = design_dmradio_50l(scale=s, mesh_size=0.006 * s)
        r = d.solve(RUNS / f"50l_s{s:g}", verbose=0)
        Vs = math.pi * ((b * s) ** 2 - (a * s) ** 2) * (H * s)
        srows.append({"scale": s, "V_m3": Vs, "F_natural_T4_m10": r.coupling["F_natural_T4_m10"],
                      "c_PU": c_PU_from_F(r.coupling["F_natural_T4_m10"], B_in, Vs),
                      "g_factor": r.coupling["g_factor"]})
    out["scale_invariance"] = srows

    # ---- reach: the published projection is a flat 5e-15 GeV^-1 block over the band,
    # i.e. the log-uniform (flat-in-g) allocation.  Find the live time that reaches it.
    bench = {"name": "dmradio_50l", "Q": si["Q"][0], "T": si["T_K"][0], "eta_A": si["eta_A"][0]}
    f1, f2 = si["f_range_Hz"][0]
    kw = dict(f_min=f1, f_max=f2, allocation="log_uniform", n=60,
              Q_axion=si["Q_axion"][0], Theta=si["SNR"][0] ** 2, rho=si["rho_DM_GeV_cm3"][0])
    Psi_SI, L_H = fine["Psi_natural_T_m3"] * MU0, fine["L_p_H"]
    r1 = scan.reach(Psi_SI, L_H, bench, live_time_yr=1.0, **kw)
    g_flat_1yr = float(np.median(r1["g95_GeV"]))
    # g_95 ~ t^(-1/4) at fixed geometry
    t_needed = (g_flat_1yr / si["g_reach_GeV"][0]) ** 4
    out["reach"] = {
        "benchmark": bench, "f_range_Hz": [f1, f2], "allocation": "log_uniform",
        "Psi_SI_H_T_m2": Psi_SI, "L_p_H": L_H, "F_natural_T4_m10": fine["F_natural_T4_m10"],
        "g95_flat_1yr_GeV": g_flat_1yr,
        "g95_flat_spread": float(np.max(r1["g95_GeV"]) / np.min(r1["g95_GeV"])),
        "g_reach_published_GeV": si["g_reach_GeV"][0],
        "live_time_for_published_reach_yr": t_needed,
        "g95_3yr_GeV": g_flat_1yr * 3.0 ** -0.25,
        "curve": {"f_Hz": list(r1["f_Hz"]), "m_eV": list(r1["m_eV"]),
                  "g95_1yr_GeV": list(r1["g95_GeV"]), "g_DFSZ_GeV": list(r1["g_target_GeV"])},
        "note": "The published 5e-15 GeV^-1 is drawn flat across 20 peV - 20 neV, which is the "
                "log-uniform (equal time per ln nu) allocation; the live time is not published.",
    }
    # also the DFSZ-target allocation, for the record
    r2 = scan.reach(Psi_SI, L_H, bench, live_time_yr=1.0, f_min=f1, f_max=f2, allocation="target",
                    n=60, Q_axion=si["Q_axion"][0], Theta=si["SNR"][0] ** 2, rho=si["rho_DM_GeV_cm3"][0])
    out["reach_target_allocation"] = {
        "g95_over_DFSZ_1yr": float(np.median(np.array(r2["g95_GeV"]) / np.array(r2["g_target_GeV"]))),
        "curve": {"f_Hz": list(r2["f_Hz"]), "g95_1yr_GeV": list(r2["g95_GeV"])},
    }

    out["comparison"] = {
        "c_PU_ours": fine["c_PU"],
        "c_PU_published": SRC["dmradio_50l"]["published_quantities"]["c_PU"][0],
        "c_PU_GUT_typical": 0.1,
        "L_p_ours_H": fine["L_p_H"], "L_p_published_H": None,
        "V_field_L_ours": V * 1e3, "V_published_L": 50.0,
    }
    save_design(design_dmradio_50l(), "dmradio_50l")
    conv = abs(out["mesh_convergence_summary"]["F_rel_change_last_refinement"]) < 0.01
    return record("dmradio_50l", {
        **out, "pass": bool(conv),
        "pass_scope": ("mesh convergence of F only.  Nothing about the 50L coupling is published, "
                       "so there is nothing here to agree or disagree with; the reach row is a "
                       "one-parameter consistency statement."),
        "headline": (f"Psi = {fine['Psi_natural_T_m3']:.4e} T m^3, L_p = {fine['L_p_H'] * 1e9:.1f} nH, "
                     f"F = {fine['F_natural_T4_m10']:.4e} T^4 m^10, c_PU = {fine['c_PU']:.4f}; "
                     f"flat reach 5e-15 GeV^-1 needs {t_needed:.2f} yr"),
    })


def check_dmradio_m3() -> dict:
    """DMRadio-m^3: the published coaxial pickups (six heights), their inductance,
    coupling and reach.  All six coax geometries are published in arXiv:2302.14084."""
    g = SRC["dmradio_m3"]["geometry"]
    em = SRC["dmradio_m3"]["em_model"]
    pub = SRC["dmradio_m3"]["published_quantities"]
    si = SRC["dmradio_m3"]["sensitivity_inputs"]
    ri, ro = g["coax_inner_radius_m"][0], g["coax_outer_radius_m"][0]
    heights = g["coax_heights_m"][0]
    bands = g["coax_bands_MHz"][0]
    Z0 = 376.730313412                       # exact free-space impedance
    Zc = Z0 / (2 * math.pi) * math.log(ro / ri)
    out: dict = {"inputs": {
        "coax_inner_radius_m": ri, "coax_outer_radius_m": ro, "x": ri / ro,
        "coax_heights_m": heights, "coax_bands_MHz": bands,
        "Z_c_ohm": Zc, "Z_c_published_ohm": em["Z_c_ohm"][0],
        "solenoid_inner_radius_m": g["bore_diameter_m"][0] / 2,
        "solenoid_outer_radius_m": g["outer_diameter_m"][0] / 2, "solenoid_H_m": g["H_m"][0]}}

    # ---- ampere-turns from the published on-axis characteristic field
    NI = 6.6e6
    for _ in range(5):
        r = design_dmradio_m3(NI, mesh_size=0.05).solve(RUNS / "m3_cal", verbose=0)
        Bc = float(r.sample([1e-9], [0.0])["Bz"][0])
        if abs(Bc / g["B_0_axis_T"][0] - 1) < 1e-4:
            break
        NI *= g["B_0_axis_T"][0] / Bc
    B0 = g["B_0_axis_T"][0]
    out["magnet"] = {"NI_A": NI, "B_z_axis_T": Bc, "B_max_T": r.B_max,
                     "B_max_published_T": g["B_peak_T"][0], "W_mag_J": r.coupling["W_mag_J"],
                     "W_published_order_J": pub["stored_energy_J"][0],
                     "note": "A uniform-current solenoid over the published envelope, with the "
                             "ampere-turns set by the published on-axis characteristic field.  The "
                             "real magnet is segmented and bucked, so its field is flatter over the "
                             "coax than ours and its peak is higher."}

    # ---- the six published coaxes
    rows = []
    for h, band in zip(heights, bands):
        d = design_dmradio_m3(NI, h_coax=h, mesh_size=0.025)
        rr = d.solve(RUNS / f"m3_h{h:g}", verbose=0)
        V_cyl = math.pi * ro ** 2 * h                    # the paper's "bore volume ~ 2 m^3"
        V_ann = math.pi * (ro ** 2 - ri ** 2) * h        # the coax annulus
        L_an = Zc * h / 299792458.0                      # M3_EM: L = Z_c h / c
        Psi = rr.coupling["Psi_natural_T_m3"]
        Lp_nat = rr.L_p / MU0
        # uniform-field long-coax upper bound: Psi = H R^2 B (1-x^2)/4, L_p = (H/2pi) ln(1/x)
        Psi_ideal = h * ro ** 2 * B0 * (1 - (ri / ro) ** 2) / 4
        Lp_ideal = h / (2 * math.pi) * math.log(ro / ri)
        F_ideal = Psi_ideal ** 4 / Lp_ideal ** 2
        rows.append({
            "h_coax_m": h, "band_MHz": band, "V_cylinder_m3": V_cyl, "V_annulus_m3": V_ann,
            "Psi_natural_T_m3": Psi, "L_p_H": rr.L_p, "L_p_analytic_H": L_an,
            "L_p_rel_to_analytic": rel(rr.L_p, L_an),
            "W_mag_J": rr.coupling["W_mag_J"], "F_natural_T4_m10": rr.coupling["F_natural_T4_m10"],
            "F_SI_T4_H2_m8": rr.F, "g_factor": rr.coupling["g_factor"],
            "c_PU_V_cylinder": c_PU_from_F(rr.coupling["F_natural_T4_m10"], B0, V_cyl),
            "c_PU_V_annulus": c_PU_from_F(rr.coupling["F_natural_T4_m10"], B0, V_ann),
            "c_PU_max_V_cylinder": c_PU_from_F(F_ideal, B0, V_cyl),
            "c_PU_max_V_annulus": c_PU_from_F(F_ideal, B0, V_ann),
            "Psi_over_uniform_field_ideal": Psi / Psi_ideal,
            "lambda_quarter_Hz": 299792458.0 / (4 * h),
            "dofs": rr.data["mesh"]["dofs"], "warnings": rr.data.get("warnings", [])})
    out["coaxes"] = rows
    tall = rows[0]

    # ---- mesh convergence on the tallest coax
    mrows = []
    for size in (0.05, 0.03, 0.02, 0.013):
        rr = design_dmradio_m3(NI, h_coax=heights[0], mesh_size=size).solve(
            RUNS / f"m3_h{heights[0]:g}_m{size:g}", verbose=0)
        mrows.append({"mesh_size_m": size, "dofs": rr.data["mesh"]["dofs"],
                      "Psi_natural_T_m3": rr.coupling["Psi_natural_T_m3"], "L_p_H": rr.L_p,
                      "F_natural_T4_m10": rr.coupling["F_natural_T4_m10"]})
    out["mesh_convergence"] = mrows
    out["mesh_convergence_summary"] = {
        "Psi_rel_change_last_refinement": rel(mrows[-1]["Psi_natural_T_m3"], mrows[-2]["Psi_natural_T_m3"]),
        "F_rel_change_last_refinement": rel(mrows[-1]["F_natural_T4_m10"], mrows[-2]["F_natural_T4_m10"])}

    out["comparison"] = {
        "Z_c_ours_ohm": Zc, "Z_c_published_ohm": em["Z_c_ohm"][0],
        "Z_c_rel": rel(Zc, em["Z_c_ohm"][0]),
        "L_p_rel_to_analytic_range": [min(x["L_p_rel_to_analytic"] for x in rows),
                                      max(x["L_p_rel_to_analytic"] for x in rows)],
        "L_p_tallest_H": tall["L_p_H"], "L_p_tallest_analytic_H": tall["L_p_analytic_H"],
        "L_PU_published_upper_bound_H": pub["L_PU_H"][0],
        "V_PU_tallest_cylinder_m3": tall["V_cylinder_m3"], "V_PU_published_m3": pub["V_PU_m3"][0],
        "c_PU_range_V_cylinder": [min(x["c_PU_V_cylinder"] for x in rows),
                                  max(x["c_PU_V_cylinder"] for x in rows)],
        "c_PU_range_V_annulus": [min(x["c_PU_V_annulus"] for x in rows),
                                 max(x["c_PU_V_annulus"] for x in rows)],
        "c_PU_published_range": pub["c_PU"][0],
        "c_PU_published_lowest_coax": pub["c_PU_lowest_coax"][0],
        "c_PU_lowest_coax_ours_V_cylinder": tall["c_PU_V_cylinder"],
        "c_PU_lowest_coax_ours_V_annulus": tall["c_PU_V_annulus"],
        "c_PU_lowest_coax_uniform_field_bound_V_cylinder": tall["c_PU_max_V_cylinder"],
        "c_PU_lowest_coax_uniform_field_bound_V_annulus": tall["c_PU_max_V_annulus"],
        "published_0.16_exceeds_cylinder_bound": bool(
            pub["c_PU_lowest_coax"][0] > tall["c_PU_max_V_cylinder"]),
        "note": "c_PU depends on which characteristic B_0 and which V_PU are used.  With the "
                "paper's own 'bore volume V ~ 2 m^3' (pi r_o^2 h) even a perfectly uniform B_0 "
                "cannot exceed c_PU_max_V_cylinder, so the published 0.16 for the lowest-frequency "
                "coax must use a smaller volume; the coax annulus brings the bound up.",
    }

    # ---- reach: each coax over its own published band (M3_EM Table I)
    bench = {"name": "dmradio_m3", "Q": si["Q"][0], "T": si["T_K"][0], "eta_A": si["eta_A"][0]}
    kw = dict(Q_axion=si["Q_axion"][0], Theta=si["SNR"][0] ** 2, rho=si["rho_DM_GeV_cm3"][0])
    reach = []
    for row in rows:
        for lo, hi in row["band_MHz"]:
            for model in ("DFSZ", "KSVZ"):
                rr = scan.reach(row["Psi_natural_T_m3"] * MU0, row["L_p_H"], bench,
                                live_time_yr=si["live_time_yr"][0], f_min=lo * 1e6, f_max=hi * 1e6,
                                allocation="log_uniform", n=12, model=model, **kw)
                gg, gt = np.array(rr["g95_GeV"]), np.array(rr["g_target_GeV"])
                reach.append({"h_coax_m": row["h_coax_m"], "band_MHz": [lo, hi], "model": model,
                              "m_eV": list(rr["m_eV"]), "g95_GeV": list(gg),
                              "g_target_GeV": list(gt),
                              "reaches_target_fraction": float((gg <= gt).mean()),
                              "g95_over_target_min": float((gg / gt).min()),
                              "g95_over_target_max": float((gg / gt).max())})
    out["reach_by_coax"] = {"benchmark": bench, "live_time_yr": si["live_time_yr"][0],
                            "allocation": "log_uniform", "rows": reach,
                            "note": "Each published coax is used only over its published band "
                                    "(M3_EM Table I), as the experiment intends."}

    # ---- scan time against the published Table III live times
    t3 = []
    for e in si["live_times_table3"][0]:
        lo, hi = e["band_MHz"]
        # M3_EM L859-860: eta = 10 for 10-30 MHz, eta = 20 for 30-200 MHz
        eta = 10.0 if hi <= 30 else si["eta_A"][0]
        b = {"name": "dmradio_m3", "Q": si["Q"][0], "T": si["T_K"][0], "eta_A": eta}
        use = [row for row in rows if any(x[0] < hi and x[1] > lo for x in row["band_MHz"])]
        base = dict(f_min=lo * 1e6, f_max=hi * 1e6, rho_DM=si["rho_DM_GeV_cm3"][0],
                    Q_axion=si["Q_axion"][0], Theta95=si["SNR"][0] ** 2)
        if e["target"].startswith("g ="):
            # fixed target coupling: F_req ~ g^-4, so scale from the DFSZ requirement
            base["model"] = "DFSZ"
            req = af.requirement(b, live_time=1.0, **base)
            scale = (req["g_min_GeV"] / 1.87e-17) ** 4
        else:
            base["model"] = e["target"]
            req = af.requirement(b, live_time=1.0, **base)
            scale = 1.0
        ts = [req["F_req_natural_T4_m10"] * scale / row["F_natural_T4_m10"] for row in use]
        best = min(ts) if ts else None
        t3.append({"target": e["target"], "band_MHz": [lo, hi], "eta_A_used": eta,
                   "Q_used": si["Q"][0],
                   "published_years": e["years"], "published_years_scaled": e["years_scaled"],
                   "coax_heights_used_m": [row["h_coax_m"] for row in use],
                   "scan_time_per_coax_yr": ts, "scan_time_best_coax_yr": best,
                   "ratio_best_to_published": (best / e["years"]) if best else None,
                   # t ~ 1/Q, so the published time implies this Q with our geometry
                   "Q_implied_by_published_time": (si["Q"][0] * best / e["years"]) if best else None})
    out["scan_time_vs_table3"] = {
        "rows": t3,
        "note": "M3_EM Table III live times, with its per-band eta_A (10 below 30 MHz, 20 above) "
                "and the science paper's Q = 1e5.  The modelled Q(nu) is published only as a figure "
                "(M3_EM Fig. 7, 1e4-1e6), so the last column inverts the comparison: the Q our "
                "geometry would need to reproduce each published live time, to be read against that "
                "1e4-1e6 range.",
    }

    for h in heights:
        save_design(design_dmradio_m3(NI, h_coax=h), f"dmradio_m3_h{h:g}".replace(".", "p"))
    save_design(design_dmradio_m3(NI), "dmradio_m3")
    conv = abs(out["mesh_convergence_summary"]["F_rel_change_last_refinement"]) < 0.02
    lp_ok = all(abs(x["L_p_rel_to_analytic"]) < 0.05 for x in rows)
    return record("dmradio_m3", {
        **out, "pass": bool(conv and lp_ok),
        "pass_scope": ("mesh convergence of F, and L_p of every coax within 5 % of DMRadio's own "
                       "Z_c h/c.  The latter IS an agreement with the publication; the published "
                       "c_PU = 0.16 for the tallest coax is NOT reproduced (see the report S3.2)."),
        "headline": (f"6 published coaxes: L_p within "
                     f"{max(abs(x['L_p_rel_to_analytic']) for x in rows) * 100:.1f} % of Z_c h/c; "
                     f"c_PU = {out['comparison']['c_PU_range_V_cylinder'][0]:.3f}-"
                     f"{out['comparison']['c_PU_range_V_cylinder'][1]:.3f} (pi r_o^2 h) or "
                     f"{out['comparison']['c_PU_range_V_annulus'][0]:.3f}-"
                     f"{out['comparison']['c_PU_range_V_annulus'][1]:.3f} (annulus), "
                     f"published 0.1-0.2"),
    })


def check_dmradio_gut() -> dict:
    """DMRadio-GUT: the 573 MJ stored energy, c_PU = 0.1 and the 6.2 yr scan time."""
    g = SRC["dmradio_gut"]["geometry"]
    si = SRC["dmradio_gut"]["sensitivity_inputs"]
    a, b, H, B0, V = g["R_in_m"][0], g["R_out_m"][0], g["H_m"][0], g["B_peak_T"][0], g["V_m3"][0]
    out: dict = {"inputs": {"R_in_m": a, "R_out_m": b, "H_m": H, "aspect_ratio": b / a,
                            "B_peak_T": B0, "V_m3": math.pi * (b ** 2 - a ** 2) * H,
                            "NI_A": 2 * math.pi * a * B0 / MU0}}
    # the aspect ratio is fixed by B_rms/B_peak and by the stored energy
    # Analytic ideal toroid at fixed V and B_peak: V = pi (ar^2 - 1) a^2 H and
    # W = pi H a^2 B_peak^2 ln(ar)/mu0, so a^2 H = V/(pi (ar^2 - 1)) fixes W.
    out["aspect_ratio_determination"] = {}
    for ar, label in ((2.0, "GUT L198 'r_out/r_in = 2'"), (b / a, "DMRadio-50L aspect ratio 24/14")):
        a2H = V / (math.pi * (ar ** 2 - 1))
        W = math.pi * a2H * B0 ** 2 * math.log(ar) / MU0
        Brms = B0 * math.sqrt(2 * math.log(ar) / (ar ** 2 - 1))
        out["aspect_ratio_determination"][f"{ar:.4f}"] = {
            "source": label, "W_analytic_J": W, "rel_to_published_W": rel(W, g["stored_energy_J"][0]),
            "B_rms_T": Brms, "B_rms_over_peak": Brms / B0,
            "B_rms_published_T": g["B_rms_T"][0], "rel_to_published_B_rms": rel(Brms, g["B_rms_T"][0])}

    rows = []
    for size in (0.09, 0.055, 0.035, 0.022):
        d = design_dmradio_gut(mesh_size=size)
        r = d.solve(RUNS / f"gut_h{size:g}", verbose=0)
        c = r.coupling
        rows.append({"mesh_size_m": size, "order": 2, "Psi_natural_T_m3": c["Psi_natural_T_m3"],
                     "L_p_H": r.L_p, "W_mag_J": c["W_mag_J"], "F_natural_T4_m10": c["F_natural_T4_m10"],
                     "g_factor": c["g_factor"], "B_max_T": r.B_max,
                     "c_PU": c_PU_from_F(c["F_natural_T4_m10"], B0, V),
                     "dofs": r.data["mesh"]["dofs"], "warnings": r.data.get("warnings", [])})
    out["mesh_convergence"] = rows
    fine = rows[-1]
    out["mesh_convergence_summary"] = {
        "Psi_rel_change_last_refinement": rel(rows[-1]["Psi_natural_T_m3"], rows[-2]["Psi_natural_T_m3"]),
        "W_rel_change_last_refinement": rel(rows[-1]["W_mag_J"], rows[-2]["W_mag_J"]),
        "F_rel_change_last_refinement": rel(rows[-1]["F_natural_T4_m10"], rows[-2]["F_natural_T4_m10"]),
    }

    # ---- the two unpublished parameters that set the "agreements"
    # (a) winding-sheet thickness.  GUT publishes none; ours is the 50L value scaled.
    #     W moves across the published 573 MJ over the plausible range, so a FEM W
    #     near 573 MJ is a statement about t_w, not about the design.
    twrows = []
    for f_tw in (0.25, 0.5, 1.0, 2.0, 4.0):
        d = design_dmradio_gut(mesh_size=0.055, tw_factor=f_tw)
        r = d.solve(RUNS / f"gut_tw{f_tw:g}", verbose=0)
        c = r.coupling
        twrows.append({"tw_factor": f_tw,
                       "winding_thickness_m": d.doc["parameters"].get("tw", float("nan")),
                       "mesh_size_m": 0.055, "W_mag_J": c["W_mag_J"],
                       "W_rel_to_published": rel(c["W_mag_J"], g["stored_energy_J"][0]),
                       "Psi_natural_T_m3": c["Psi_natural_T_m3"], "L_p_H": r.L_p,
                       "B_max_T": r.B_max,
                       "c_PU": c_PU_from_F(c["F_natural_T4_m10"], B0, V),
                       "dofs": r.data["mesh"]["dofs"]})
    out["winding_thickness_scan"] = {
        "rows": twrows,
        "W_published_J": g["stored_energy_J"][0],
        "W_range_J": [min(x["W_mag_J"] for x in twrows), max(x["W_mag_J"] for x in twrows)],
        "note": "The winding-sheet thickness is not published for GUT or for 50L.  Over a factor "
                "16 in t_w the FEM stored energy straddles the published 573 MJ, so the agreement "
                "of the nominal design with 573 MJ is set by this unpublished parameter.",
    }

    # (b) H/R_in at fixed V, aspect ratio and B_peak.  W is independent of it
    #     analytically; Psi, L_p and c_PU are not, so our c_PU deviation from the
    #     published "typical ~0.1" is a statement about an unpublished choice.
    hrows = []
    for hr in (1.0, 2.0, H / a, 5.0, 8.0):
        d = design_dmradio_gut(mesh_size=None, h_over_R_in=hr)
        p = d.doc["parameters"]
        size = max(p["R"], p["H"]) / 35.0
        d.doc["mesh"]["size"] = size
        r = d.solve(RUNS / f"gut_hr{hr:.3f}", verbose=0)
        c = r.coupling
        hrows.append({"H_over_R_in": hr, "R_in_m": p["R"] * p["y"], "R_out_m": p["R"],
                      "H_m": p["H"], "mesh_size_m": size,
                      "V_m3": math.pi * (p["R"] ** 2 - (p["R"] * p["y"]) ** 2) * p["H"],
                      "W_mag_J": c["W_mag_J"], "Psi_natural_T_m3": c["Psi_natural_T_m3"],
                      "L_p_H": r.L_p, "B_max_T": r.B_max, "g_factor": c["g_factor"],
                      "c_PU": c_PU_from_F(c["F_natural_T4_m10"], B0, V),
                      "dofs": r.data["mesh"]["dofs"]})
    out["height_ratio_scan"] = {
        "rows": hrows, "c_PU_published": si["scenarios"][0]["c_PU"],
        "c_PU_range": [min(x["c_PU"] for x in hrows), max(x["c_PU"] for x in hrows)],
        "W_range_J": [min(x["W_mag_J"] for x in hrows), max(x["W_mag_J"] for x in hrows)],
        "note": "GUT fixes V and B_peak but neither H nor R_in.  At fixed V and aspect ratio the "
                "ideal-toroid W is independent of H/R_in, and the FEM W indeed moves by < 1 %, but "
                "c_PU moves by a factor 1.4.  The published 'typical c_PU ~ 0.1' is therefore "
                "bracketed by our own range, and the quoted deviation from it is set by an "
                "unpublished parameter rather than measured.",
    }

    # ---- scan time to reach DFSZ over the published band, their conventions
    f1, f2 = si["f_range_Hz"][0]
    base = dict(f_min=f1, f_max=f2, model="DFSZ", rho_DM=si["rho_DM_GeV_cm3"][0],
                Q_axion=si["Q_axion"][0], Theta95=si["SNR"][0] ** 2)
    scen = []
    for s in si["scenarios"]:
        bench = {"name": s["name"], "Q": s["Q"], "T": s["T_K"], "eta_A": s["eta_A"]}
        F_pub = F_from_c_PU(s["c_PU"], s["B0_T"], s["V_m3"])
        t_pub = scan_time_yr(F_pub, bench, **base)
        # our geometry, scaled to this scenario's B0 and V (F ~ B0^4 V^(10/3) at fixed c_PU)
        F_ours = fine["F_natural_T4_m10"] * (s["B0_T"] / B0) ** 4 * (s["V_m3"] / V) ** (10 / 3)
        t_ours = scan_time_yr(F_ours, bench, **base)
        # the same with the chiral-perturbation-theory C_gamma instead of the lattice default
        t_pub_Cg = scan_time_yr(F_pub, bench, **{**base, "C_QCD": 1.92})
        t_ours_Cg = scan_time_yr(F_ours, bench, **{**base, "C_QCD": 1.92})
        scen.append({"scenario": s["name"], "B0_T": s["B0_T"], "V_m3": s["V_m3"], "Q": s["Q"],
                     "eta_A": s["eta_A"], "T_K": s["T_K"],
                     "scan_time_published_yr": s["scan_time_yr"],
                     "F_from_published_c_PU_T4_m10": F_pub,
                     "scan_time_from_published_c_PU_yr": t_pub,
                     "scan_time_from_published_c_PU_Cgamma1.92_yr": t_pub_Cg,
                     "F_ours_T4_m10": F_ours, "c_PU_ours": fine["c_PU"],
                     "scan_time_ours_yr": t_ours,
                     "scan_time_ours_Cgamma1.92_yr": t_ours_Cg,
                     "published_over_eq3_with_published_c_PU": s["scan_time_yr"] / t_pub,
                     # the paper's own DFSZ convention, C_agammagamma ~ 0.75 => C_gamma = 1.92
                     "published_over_eq3_with_published_c_PU_Cgamma1.92":
                         s["scan_time_yr"] / t_pub_Cg})
    C_agg = SRC["dmradio_gut"]["sensitivity_inputs"]["C_agammagamma"][0]
    out["scan_time"] = {
        "band_Hz": [f1, f2], "conventions": {**base, "C_QCD_default": 1.77},
        "C_agammagamma_published": C_agg,
        "C_QCD_implied_by_published_C_agammagamma": 8.0 / 3.0 - C_agg,
        "C_QCD_used_for_the_headline": 1.92,
        "Cgamma_convention_factor_in_time": ((8.0 / 3.0 - 1.77) / (8.0 / 3.0 - 1.92)) ** 4,
        "scenarios": scen,
        "note": "scan_time_from_published_c_PU_yr is what GUT Eq. (3) + Table II imply in our "
                "implementation (which reproduces the 41 kHz/yr anchor of Eq. (3)); "
                "scan_time_published_yr is the number in Table II.  THE COMPARISON TO QUOTE is "
                "the C_gamma = 1.92 one: GUT fixes C_agammagamma ~ 0.75 (DFSZ), i.e. "
                "C_gamma = 8/3 - 0.75 = 1.92, not axionFEM's lattice default 1.77.  With the "
                "paper's own convention the Table II / Eq. (3) discrepancy is x1.60, not x3.34; "
                "the x3.34 is an artefact of the wrong C_gamma.",
    }
    # the relative scaling of the four scenarios (t ~ eta_A/(Q B0^4 V^(10/3)))
    b0 = scen[0]
    out["scan_time"]["relative_scaling_check"] = [
        {"scenario": s["scenario"],
         "published_ratio_to_baseline": s["scan_time_published_yr"] / b0["scan_time_published_yr"],
         "our_ratio_to_baseline": s["scan_time_from_published_c_PU_yr"] / b0["scan_time_from_published_c_PU_yr"]}
        for s in scen]

    bench0 = {"name": "baseline", "Q": si["Q"][0], "T": si["T_K"][0], "eta_A": si["eta_A"][0]}
    rr = scan.reach(fine["Psi_natural_T_m3"] * MU0, fine["L_p_H"], bench0,
                    live_time_yr=si["live_time_yr"][0], f_min=f1, f_max=f2, allocation="target",
                    n=60, Q_axion=si["Q_axion"][0], Theta=si["SNR"][0] ** 2, rho=si["rho_DM_GeV_cm3"][0])
    out["reach"] = {"benchmark": bench0, "live_time_yr": si["live_time_yr"][0], "allocation": "target",
                    "g95_over_DFSZ": float(np.median(np.array(rr["g95_GeV"]) / np.array(rr["g_target_GeV"]))),
                    "curve": {"f_Hz": list(rr["f_Hz"]), "m_eV": list(rr["m_eV"]),
                              "g95_GeV": list(rr["g95_GeV"]), "g_DFSZ_GeV": list(rr["g_target_GeV"])}}

    W_analytic = out["aspect_ratio_determination"][f"{b / a:.4f}"]["W_analytic_J"]
    out["comparison"] = {
        "W_ours_J": fine["W_mag_J"], "W_published_J": g["stored_energy_J"][0],
        "W_rel_to_published": rel(fine["W_mag_J"], g["stored_energy_J"][0]),
        # the non-circular energy check: the FEM against the closed form for the SAME
        # (a, b, H, NI), which differs only by the winding sheet, the fillet and the sheath
        "W_analytic_ideal_J": W_analytic,
        "W_rel_to_analytic_ideal": rel(fine["W_mag_J"], W_analytic),
        "W_published_is_a_definition": 0.5 * g["B_rms_T"][0] ** 2 * g["V_m3"][0] / MU0,
        "W_circularity_note":
            "573 MJ is not an independent datum: (1/2 mu0) B_rms^2 V with GUT's own 12 T and "
            "10 m^3 gives 572.96 MJ, and the aspect ratio of this design was CHOSEN to reproduce "
            "the (573 MJ, 12 T RMS) pair (see sources.json, dmradio_gut/geometry/aspect_ratio).  "
            "W_rel_to_published is therefore circular and is NOT an agreement.  It is also inside "
            "the spread of the unpublished winding thickness (winding_thickness_scan: 569-612 MJ). "
            " W_rel_to_analytic_ideal is the check that means something.",
        "B_peak_ours_T": fine["B_max_T"], "B_peak_published_T": B0,
        "c_PU_ours": fine["c_PU"], "c_PU_published": si["scenarios"][0]["c_PU"],
        "c_PU_rel_to_published": rel(fine["c_PU"], si["scenarios"][0]["c_PU"]),
        "c_PU_note": "The deviation from the published 'typical ~0.1' is set by the unpublished "
                     "H/R_in: see height_ratio_scan, where c_PU runs over 0.068-0.098.",
        "scan_time_ours_yr": scen[0]["scan_time_ours_yr"],
        "scan_time_published_yr": scen[0]["scan_time_published_yr"],
        "table2_over_eq3": scen[0]["published_over_eq3_with_published_c_PU_Cgamma1.92"],
        "table2_over_eq3_note": "With GUT's own C_agammagamma ~ 0.75 (C_gamma = 1.92).  The "
                                "lattice-C_gamma value, 3.34, should not be quoted.",
    }
    save_design(design_dmradio_gut(), "dmradio_gut")
    # Self-check: convergence, and the FEM energy against the closed form at the same
    # dimensions.  NOT against the published 573 MJ, which this design was built to hit.
    ok = (abs(out["comparison"]["W_rel_to_analytic_ideal"]) < 0.05
          and abs(out["mesh_convergence_summary"]["F_rel_change_last_refinement"]) < 0.01)
    return record("dmradio_gut", {
        **out, "pass": bool(ok),
        "pass_scope": ("mesh convergence of F, and W_FEM within 5 % of the analytic ideal "
                       "rectangular toroid at the same dimensions.  It does NOT test the "
                       "published 573 MJ (circular) nor the published c_PU (set by an "
                       "unpublished H/R_in) nor the Table II scan time (x1.60 off)."),
        "headline": (f"W = {fine['W_mag_J'] / 1e6:.1f} MJ, {rel(fine['W_mag_J'], W_analytic) * 100:+.1f} % "
                     f"from the analytic ideal toroid; vs 'published' 573 MJ "
                     f"({out['comparison']['W_rel_to_published'] * 100:+.1f} %) but that is CIRCULAR "
                     f"(573 MJ is defined by B_rms and V, and the aspect ratio was chosen for it; "
                     f"W spans {min(x['W_mag_J'] for x in twrows) / 1e6:.0f}-"
                     f"{max(x['W_mag_J'] for x in twrows) / 1e6:.0f} MJ over the unpublished t_w); "
                     f"c_PU = {fine['c_PU']:.4f} vs 'typical 0.1', but "
                     f"{min(x['c_PU'] for x in hrows):.4f}-{max(x['c_PU'] for x in hrows):.4f} over the "
                     f"unpublished H/R_in = 1-8; Table II 6.2 yr / Eq. (3) "
                     f"{scen[0]['scan_time_from_published_c_PU_Cgamma1.92_yr']:.2f} yr = "
                     f"x{scen[0]['published_over_eq3_with_published_c_PU_Cgamma1.92']:.2f} at the "
                     f"paper's C_agammagamma = 0.75"),
    })


def check_dmradio_core() -> dict:
    """DMRadio-Core: the published magnet (5 T, 4.3 MJ) and a CORE-template pickup."""
    g = SRC["dmradio_core"]["geometry"]
    si = SRC["dmradio_core"]["sensitivity_inputs"]
    out: dict = {"inputs": {k: g[k][0] for k in ("bore_diameter_m", "outer_diameter_m", "H_m",
                                                 "pickup_inner_diameter_m", "pickup_height_m",
                                                 "plate_spacing_m")}}
    # ---- 1. the magnet: ampere-turns from the published conductor, and from B_peak = 5 T
    Rc, Ro, H = g["bore_diameter_m"][0] / 2, g["outer_diameter_m"][0] / 2, g["H_m"][0]
    wire_d = 3e-3
    NI_conductor = (Ro - Rc) / wire_d * (H / wire_d) * 330.0
    rows = []
    for NI, label in ((NI_conductor, "from 3 mm NbTi at 330 A filling the winding cross-section"),):
        d = design_core_magnet(NI)
        r = d.solve(RUNS / "core_magnet_cond", verbose=0)
        rows.append({"NI_A": NI, "source": label, "B_z_axis_T": float(r.sample([1e-9], [0.0])["Bz"][0]),
                     "B_max_T": r.B_max, "W_mag_J": r.coupling["W_mag_J"]})
    scale = g["B_peak_T"][0] / rows[0]["B_max_T"]
    d = design_core_magnet(NI_conductor * scale)
    r = d.solve(RUNS / "core_magnet_5T", verbose=0)
    rows.append({"NI_A": NI_conductor * scale, "source": "rescaled to the published 5 T peak",
                 "B_z_axis_T": float(r.sample([1e-9], [0.0])["Bz"][0]), "B_max_T": r.B_max,
                 "W_mag_J": r.coupling["W_mag_J"]})
    out["magnet"] = {
        "rows": rows, "turns_estimate": (Ro - Rc) / wire_d * (H / wire_d),
        "W_published_J": g["stored_energy_J"][0],
        "W_rel_to_published": rel(rows[-1]["W_mag_J"], g["stored_energy_J"][0]),
        "B_peak_published_T": g["B_peak_T"][0],
        "note": "Uniform current density over the published winding envelope; the 9-fold axial "
                "segmentation and the 3 soft-iron flux-return segments are not modelled "
                "(their positions are not published).",
    }
    NI_core = NI_conductor * scale
    save_design(design_core_magnet(NI_core), "dmradio_core_magnet")

    # ---- 2. the pickup, both published outer diameters
    prows = []
    for od in g["pickup_outer_diameter_m"][0]:
        try:
            d = design_dmradio_core(NI_core, outer_diameter=od)
            r = d.solve(RUNS / f"core_pu{od:g}", verbose=0)
            c = r.coupling
            V_pu = math.pi * ((od / 2) ** 2 - (g["pickup_inner_diameter_m"][0] / 2) ** 2) * g["pickup_height_m"][0]
            prows.append({"outer_diameter_m": od, "Psi_natural_T_m3": c["Psi_natural_T_m3"],
                          "L_p_H": r.L_p, "W_mag_J": c["W_mag_J"],
                          "F_natural_T4_m10": c["F_natural_T4_m10"], "g_factor": c["g_factor"],
                          "B_max_T": r.B_max,
                          "superconductor_surface_T": r.data.get("fields", {}).get("superconductor_surface_regular_T"),
                          "V_pickup_shell_m3": V_pu, "dofs": r.data["mesh"]["dofs"],
                          "warnings": r.data.get("warnings", [])})
            save_design(design_dmradio_core(NI_core, outer_diameter=od),
                        f"dmradio_core_pickup_{int(od * 100)}cm")
        except Exception as exc:                                     # noqa: BLE001
            prows.append({"outer_diameter_m": od, "error": f"{type(exc).__name__}: {exc}"})
    out["pickup"] = {
        "rows": prows,
        "note": "The CORE template is a superconducting toroidal cavity around the solenoid, read "
                "out across a neck gap in its inner wall.  DMRadio-Core's pickup instead closes "
                "across the bore with two copper plates (6 cm apart) and is read out through a "
                "30-degree azimuthal sector of the slit; neither the copper plates nor the "
                "fractional port are axisymmetric, so these numbers are a model of the topology, "
                "not of the device.  CORE publishes no L_eff or |V| to compare with.",
    }

    # ---- 3. scan time implied by our pickup model, with the published noise inputs
    f1, f2 = si["f_range_Hz"][0]
    base = dict(f_min=f1, f_max=f2, model="DFSZ", rho_DM=si["rho_DM_GeV_cm3"][0],
                Q_axion=si["Q_axion"][0], Theta95=si["SNR"][0] ** 2)
    st = []
    good = [p for p in prows if "F_natural_T4_m10" in p]
    eta_pub = eta_A_band(f1, f2)
    for eta in (eta_pub, 1.0):
        bench = {"name": f"core_eta{eta:.3g}", "Q": si["Q"][0], "T": si["T_K"][0], "eta_A": eta}
        req = af.requirement(bench, live_time=si["live_time_yr"][0], **base)
        times = [scan_time_yr(p["F_natural_T4_m10"], bench, **base) for p in good]
        st.append({"eta_A": eta, "F_req_natural_T4_m10": req["F_req_natural_T4_m10"],
                   "live_time_yr": si["live_time_yr"][0],
                   "outer_diameter_m": [p["outer_diameter_m"] for p in good],
                   "F_over_F_req": [p["F_natural_T4_m10"] / req["F_req_natural_T4_m10"] for p in good],
                   "scan_time_for_DFSZ_yr": times,
                   "scan_time_over_published": [t / si["live_time_yr"][0] for t in times],
                   # CORE Eq. (5): rates add as (sum of sqrt(rate_i))^2, i.e. t_tot =
                   # 1/(sum 1/sqrt(t_i))^2 for pickups read coherently at the same frequency.
                   "scan_time_both_pickups_coherent_yr":
                       1.0 / sum(t ** -0.5 for t in times) ** 2 if all(t > 0 for t in times) else None})
    out["scan_time"] = {"band_Hz": [f1, f2], "published_live_time_yr": si["live_time_yr"][0],
                        "eta_A_published_band_average": eta_pub,
                        "eta_A_at_band_edges": [eta_A_at(f1), eta_A_at(f2)],
                        "rows": st,
                        "note": "eta_A is the published dc-SQUID optimum (arXiv:2504.20398 Fig. 3; "
                                "8.1 at 70 MHz to 11.7 at 140 MHz), log-averaged over the band; "
                                "eta_A = 1 (the SQL) is kept as a floor for reference.  These scan "
                                "times use the MQS Psi and L_p, which at 70-140 MHz are only the "
                                "omega -> 0 limit: the beyond-MQS values are in the `fullwave` "
                                "record and are much larger (section 6)."}

    ok = abs(out["magnet"]["W_rel_to_published"]) < 0.10 and bool(good)
    return record("dmradio_core", {
        **out, "pass": bool(ok),
        "pass_scope": ("the magnet stored energy within 10 % of the published 4.3 MJ on the "
                       "published envelope (a genuine agreement), and that the pickup variants "
                       "solved.  It does NOT test the Core scan time, which disagrees by 4-11x."),
        "headline": (f"magnet at 5 T peak: W = {rows[-1]['W_mag_J'] / 1e6:.2f} MJ vs published 4.3 MJ "
                     f"({out['magnet']['W_rel_to_published'] * 100:+.1f} %); "
                     f"pickup model: {len(good)}/{len(prows)} solved"),
    })


# ================================================== beyond MQS (full wave)
def _dX_exact(dz) -> float:
    """-Im dZ_00/dw from a sweep row's `dZ_domega_H`; NaN if absent or null (a non-finite value is written as null)."""
    if not dz:
        return math.nan
    v = dz["im"][0][0]
    return -float(v) if v is not None else math.nan


def _fullwave_rows(res) -> list[dict]:
    """The sweep, plus F(omega)/F_MQS and DMRadio's Foster (series-RLC) L_eff.

    axionFEM reports L(omega) = -Im Z/omega, the instantaneous reactance.  DMRadio
    (arXiv:2302.14084 Eq. 6) uses the series-RLC equivalent of the port impedance
    at the tuning frequency,

        L_eff(w) = [w dX/dw + X] / (2 w),   X = Im Z (their e^{+iwt} sign) = w L(w),

    which equals L in the MQS limit and, by Foster's reactance theorem (dX/dw > 0),
    stays positive above a self-resonance, where a tuning inductance replaces the
    tuning capacitance.  F_eff = |Psi|^4 / L_eff^2 is then defined over the whole
    band, unlike F built from -Im Z/omega.

    dX/dw is the solver's exact per-frequency derivative, -Im dZ_00/dw from
    `dZ_domega_H` (reciprocity on the stored port fields, exact to the solve
    residual), when the sweep carries it.  Older records fall back to a finite
    difference on the sweep, which is only as good as the sampling: trust it away
    from the poles, not within a point or two of one.  `dXdw_source` says which.
    """
    F_mqs = res.F
    sweep = res.fullwave.get("sweep", [])
    f = np.array([r["f_Hz"] for r in sweep], dtype=float)
    X = np.array([2 * math.pi * r["f_Hz"] * r["L_H"] if r.get("L_H") is not None else np.nan
                  for r in sweep], dtype=float)          # X = w L = -Im Z (our sign)
    w = 2 * math.pi * f
    dX_exact = np.array([_dX_exact(r.get("dZ_domega_H")) for r in sweep], dtype=float)
    dXdw = np.full_like(X, np.nan)
    if len(w) >= 3:
        dXdw = np.gradient(X, w)
    exact = np.isfinite(dX_exact)
    dXdw = np.where(exact, dX_exact, dXdw)
    L_eff = (w * dXdw + X) / (2 * w)
    out = []
    for i, row in enumerate(sweep):
        F = row.get("F_SI_T4_H2_m8")
        Le = float(L_eff[i]) if np.isfinite(L_eff[i]) else None
        out.append({**{k: row.get(k) for k in
                       ("f_Hz", "k_R", "Psi_abs", "Psi_over_MQS", "L_H", "L_over_MQS",
                        "R_re_ohm", "Z_re_ohm", "R_rad_ohm", "Q_rad", "F_SI_T4_H2_m8",
                        "radiated_W", "power_balance_rel")},
                    "F_over_F_MQS": (F / F_mqs) if (F and F_mqs) else None,
                    "L_eff_foster_H": Le,
                    "dXdw_source": ("exact" if exact[i] else
                                    "finite_difference" if np.isfinite(dXdw[i]) else None),
                    "L_eff_over_MQS": (Le / res.L_p) if (Le and res.L_p) else None,
                    "F_eff_SI_T4_H2_m8": ((row["Psi_abs"] ** 4 / Le ** 2)
                                          if (Le and Le > 0 and row.get("Psi_abs")) else None),
                    "F_eff_over_F_MQS": ((row["Psi_abs"] ** 4 / Le ** 2 / F_mqs)
                                         if (Le and Le > 0 and row.get("Psi_abs") and F_mqs) else None),
                    "capacitive": bool(row.get("L_over_MQS") is not None
                                       and row["L_over_MQS"] <= 0)})
    return out


def _first_self_resonance(rows: list[dict]) -> float | None:
    """Lowest frequency where L(omega) changes sign (log interpolation of 1/L)."""
    for a, b in zip(rows, rows[1:]):
        la, lb = a.get("L_over_MQS"), b.get("L_over_MQS")
        if la is None or lb is None or not (la > 0 >= lb):
            continue
        # L ~ L0/(1 - (f/f0)^2) near the pole: interpolate 1/L linearly in f^2
        ia, ib = 1.0 / la, 1.0 / lb
        fa2, fb2 = a["f_Hz"] ** 2, b["f_Hz"] ** 2
        return float(math.sqrt(fa2 + (fb2 - fa2) * ia / (ia - ib)))
    return None


def _mqs_validity(rows: list[dict], tol: float) -> float | None:
    """Lowest frequency at which |Psi(omega)|/Psi_MQS leaves 1 by more than tol."""
    for a, b in zip(rows, rows[1:]):
        ra, rb = a.get("Psi_over_MQS"), b.get("Psi_over_MQS")
        if ra is None or rb is None or not (abs(ra - 1) <= tol < abs(rb - 1)):
            continue
        t = (tol - abs(ra - 1)) / (abs(rb - 1) - abs(ra - 1))
        return float(a["f_Hz"] * (b["f_Hz"] / a["f_Hz"]) ** t)
    return None


def local_reach(rows: list[dict], bench: dict, f1: float, f2: float, live_time_yr: float,
                model: str = "DFSZ", **kw) -> list[dict]:
    """g_95 at each sweep frequency using the LOCAL F(omega) there.

    With the log-uniform allocation the dwell density tau(omega_0) does not depend
    on the detector (S5.7), and g_95(m) follows from the local F (S5.8), so
    evaluating scan.reach at (Psi(omega), L(omega)) and reading it at that
    frequency is the correct beyond-MQS reach.
    """
    out = []
    for row in rows:
        f = row["f_Hz"]
        if not (f1 <= f <= f2):
            continue
        m = scan.freq_to_mass(f)
        row_out = {"f_Hz": f, "m_eV": m, "capacitive": row["capacitive"],
                   "g_target_GeV": float(scan.g_qcd([m], model=model)[0]),
                   "g95_GeV": None, "g95_L_eff_GeV": None}
        # g_95 is flat in m for the log-uniform allocation, so any point is the local value
        if row.get("F_SI_T4_H2_m8"):
            rr = scan.reach(row["Psi_abs"], row["L_H"], bench, live_time_yr=live_time_yr,
                            f_min=f1, f_max=f2, allocation="log_uniform", n=3, model=model, **kw)
            row_out["g95_GeV"] = float(rr["g95_GeV"][0])
        if row.get("L_eff_foster_H") and row["L_eff_foster_H"] > 0 and row.get("Psi_abs"):
            r2 = scan.reach(row["Psi_abs"], row["L_eff_foster_H"], bench, live_time_yr=live_time_yr,
                            f_min=f1, f_max=f2, allocation="log_uniform", n=3, model=model, **kw)
            row_out["g95_L_eff_GeV"] = float(r2["g95_GeV"][0])
        out.append(row_out)
    return out


def check_fullwave() -> dict:
    """Beyond MQS: Psi(omega), L(omega), F(omega) and the reach for DMRadio-m^3 and
    the DMRadio-Core pickups over their published bands."""
    out: dict = {"note": (
        "Axisymmetric m = 0 full wave (docs/validation/fullwave.md).  The readout is driven "
        "uniformly around the axis: exact while the readout loop's circumference is small "
        "compared with the wavelength, and the uniform-feed limit otherwise.  DMRadio-m^3's coax "
        "'funnels to a narrow neck coupled to the matching circuit above', i.e. an on-axis "
        "(axisymmetric) feed, so the model is appropriate; DMRadio-Core is read out through a "
        "30-degree sector of its slit, so for Core these are the uniform-feed limit only.")}
    cases = []

    # ---- DMRadio-m^3: all six published coaxes over their own published bands.
    # Prediction to test: the first self-resonance is the lambda/4 pole of a shorted
    # coax seen from its open end, c/(4 h_coax)  (M3_EM Eq. 2, L370-375).
    si3 = SRC["dmradio_m3"]["sensitivity_inputs"]
    g3 = SRC["dmradio_m3"]["geometry"]
    heights, bands = g3["coax_heights_m"][0], g3["coax_bands_MHz"][0]
    NI3 = json.loads((DES / "dmradio_m3.json").read_text())["parameters"]["NI"]
    bench3 = {"name": "dmradio_m3", "Q": si3["Q"][0], "T": si3["T_K"][0], "eta_A": si3["eta_A"][0]}
    kw3 = dict(Q_axion=si3["Q_axion"][0], Theta=si3["SNR"][0] ** 2, rho=si3["rho_DM_GeV_cm3"][0])
    res_rows = []
    for h, band in zip(heights, bands):
        d = design_dmradio_m3(NI3, h_coax=h, mesh_size=0.03)
        lo = min(b[0] for b in band) * 1e6
        hi = max(b[1] for b in band) * 1e6
        d.doc["fullwave"] = {"f_min": min(2.0e6, lo / 2), "f_max": max(2.2e8, hi * 1.3),
                             "points": 34, "boundary": "radiation"}
        r = d.solve(RUNS / f"fw_m3_h{h:g}", verbose=0)
        rows = _fullwave_rows(r)
        f_res = _first_self_resonance(rows)
        f_q = 299792458.0 / (4 * h)
        res_rows.append({"h_coax_m": h, "band_MHz": band,
                         "first_self_resonance_Hz": f_res, "lambda_quarter_Hz": f_q,
                         "rel_to_lambda_quarter": rel(f_res, f_q) if f_res else None,
                         "resonance_inside_own_band": bool(
                             f_res and any(b[0] * 1e6 <= f_res <= b[1] * 1e6 for b in band)),
                         "mqs_within_5pct_up_to_Hz": _mqs_validity(rows, 0.05),
                         "mqs_within_20pct_up_to_Hz": _mqs_validity(rows, 0.20)})
        cases.append({
            "name": f"dmradio_m3_h{h:g}", "label": f"DMRadio-m^3 coax, h = {h} m",
            "published_band_Hz": [lo, hi], "published_bands_MHz": band,
            "device_radius_m": r.coupling["device_radius_m"],
            "h_coax_m": h, "lambda_quarter_Hz": f_q,
            "Psi_MQS_natural_T_m3": r.coupling["Psi_natural_T_m3"], "L_MQS_H": r.L_p,
            "F_MQS_SI": r.F, "mesh_kh_max": r.fullwave.get("mesh_kh_max"),
            "dofs": r.data["mesh"]["dofs"], "sweep": rows,
            "first_self_resonance_Hz": f_res,
            "mqs_within_5pct_up_to_Hz": _mqs_validity(rows, 0.05),
            "mqs_within_20pct_up_to_Hz": _mqs_validity(rows, 0.20),
            "reach_fullwave": local_reach(rows, bench3, lo, hi, si3["live_time_yr"][0], **kw3),
            "warnings": r.data.get("warnings", []),
        })
    # The ideal formula ignores the open end.  A real open-circuited line behaves as if
    # it were longer by a fringing extension dl, so f_res = c/(4(h + dl)) with dl set by
    # the gap width, not by h.  Fit the single dl and report the residual scatter.
    C0 = 299792458.0
    dls = [C0 / (4 * x["first_self_resonance_Hz"]) - x["h_coax_m"]
           for x in res_rows if x["first_self_resonance_Hz"]]
    dl = float(np.mean(dls)) if dls else None
    gap = g3["coax_outer_radius_m"][0] - g3["coax_inner_radius_m"][0]
    for x, d_i in zip([x for x in res_rows if x["first_self_resonance_Hz"]], dls):
        x["end_extension_m"] = d_i
        x["lambda_quarter_with_end_Hz"] = C0 / (4 * (x["h_coax_m"] + dl)) if dl else None
        x["rel_to_lambda_quarter_with_end"] = (
            rel(x["first_self_resonance_Hz"], x["lambda_quarter_with_end_Hz"]) if dl else None)
    out["m3_self_resonances"] = {
        "rows": res_rows,
        "published_prediction": "c/(4 h_coax), the pole of Z_p = i Z_c tan(omega h/c) "
                                "(M3_EM Eq. 2, L350-375), for an ideal line with no end effect",
        "max_abs_rel_to_lambda_quarter": max(
            (abs(x["rel_to_lambda_quarter"]) for x in res_rows if x["rel_to_lambda_quarter"]),
            default=None),
        "end_extension_fitted_m": dl,
        "end_extension_over_gap_width": (dl / gap) if dl else None,
        "gap_width_m": gap,
        "end_extension_spread": ((max(dls) - min(dls)) / dl) if dls and dl else None,
        "max_abs_rel_with_end_correction": max(
            (abs(x["rel_to_lambda_quarter_with_end"]) for x in res_rows
             if x.get("rel_to_lambda_quarter_with_end") is not None), default=None),
        "TE111_published_Hz": SRC["dmradio_m3"]["em_model"]["TE111_MHz"][0] * 1e6,
        "note": "Our m = 0 poles sit 9-25 % below c/(4 h_coax), and the deficit grows as the coax "
                "gets shorter.  A single open-end length extension dl (independent of h, set by the "
                "gap width) absorbs all of it: that is the standard fringing correction the ideal "
                "lossless-line formula omits.  Separately, the TE111 mode at ~100 MHz is an m = 1 "
                "mode that an axisymmetric solver cannot see; M3_EM L509-516 puts it at 125-185 MHz "
                "for coaxes 1-5, and it is the mode their single-mode requirement is about.",
    }

    # ---- DMRadio-Core pickups, published band 70-140 MHz (Ops 1), 30-200 MHz (Ops 2)
    sic = SRC["dmradio_core"]["sensitivity_inputs"]
    gc = SRC["dmradio_core"]["geometry"]
    f1c, f2c = sic["f_range_Hz"][0]
    for od in gc["pickup_outer_diameter_m"][0]:
        name = f"dmradio_core_pickup_{int(od * 100)}cm"
        d = af.load(DES / f"{name}.json")
        d.doc["fullwave"] = {"f_min": 1.0e7, "f_max": 2.0e8, "points": 30, "boundary": "radiation"}
        r = d.solve(RUNS / f"fw_{name}", verbose=0)
        rows = _fullwave_rows(r)
        benchc = {"name": name, "Q": sic["Q"][0], "T": sic["T_K"][0],
                  "eta_A": eta_A_band(f1c, f2c)}
        kwc = dict(Q_axion=sic["Q_axion"][0], Theta=sic["SNR"][0] ** 2,
                   rho=sic["rho_DM_GeV_cm3"][0])
        cases.append({
            "name": name, "label": f"DMRadio-Core pickup, outer diameter {od * 100:.0f} cm",
            "published_band_Hz": [f1c, f2c], "device_radius_m": r.coupling["device_radius_m"],
            "Psi_MQS_natural_T_m3": r.coupling["Psi_natural_T_m3"], "L_MQS_H": r.L_p,
            "F_MQS_SI": r.F, "mesh_kh_max": r.fullwave.get("mesh_kh_max"),
            "dofs": r.data["mesh"]["dofs"], "sweep": rows,
            "first_self_resonance_Hz": _first_self_resonance(rows),
            "mqs_within_5pct_up_to_Hz": _mqs_validity(rows, 0.05),
            "mqs_within_20pct_up_to_Hz": _mqs_validity(rows, 0.20),
            "reach_fullwave": local_reach(rows, benchc, f1c, f2c, sic["live_time_yr"][0], **kwc),
            "warnings": r.data.get("warnings", []),
        })
    out["cases"] = cases

    # ---- band-averaged effect on the scan
    summ = []
    for c in cases:
        band = [x for x in c["sweep"] if c["published_band_Hz"][0] <= x["f_Hz"] <= c["published_band_Hz"][1]]
        usable = [x for x in band if x.get("F_eff_over_F_MQS")]
        summ.append({
            "name": c["name"], "label": c["label"],
            "published_band_Hz": c["published_band_Hz"],
            "published_bands_MHz": c.get("published_bands_MHz"),
            "first_self_resonance_Hz": c["first_self_resonance_Hz"],
            "resonance_inside_published_band": bool(
                c["first_self_resonance_Hz"] and
                c["published_band_Hz"][0] <= c["first_self_resonance_Hz"] <= c["published_band_Hz"][1]),
            "mqs_within_5pct_up_to_Hz": c["mqs_within_5pct_up_to_Hz"],
            "mqs_within_20pct_up_to_Hz": c["mqs_within_20pct_up_to_Hz"],
            "band_points": len(band), "band_points_usable": len(usable),
            "band_points_capacitive": sum(1 for x in band if x["capacitive"]),
            "F_eff_over_F_MQS_min": min((x["F_eff_over_F_MQS"] for x in usable), default=None),
            "F_eff_over_F_MQS_max": max((x["F_eff_over_F_MQS"] for x in usable), default=None),
            "Psi_over_MQS_min": min((x["Psi_over_MQS"] for x in band if x.get("Psi_over_MQS")),
                                    default=None),
            "Psi_over_MQS_max": max((x["Psi_over_MQS"] for x in band if x.get("Psi_over_MQS")),
                                    default=None),
            "Psi_over_MQS_at_band_top": band[-1].get("Psi_over_MQS") if band else None,
        })
    out["summary"] = summ

    # ---- the beyond-MQS scan time for DMRadio-Core, against its published 0.75 yr.
    # The scan time is t = int df / (df/dt) with df/dt ~ F(nu) Q Gbar(nu); doing it
    # frequency by frequency with F(omega) is the whole point of the full-wave sweep.
    core_cases = [c for c in cases if c["name"].startswith("dmradio_core")]
    core_rows = []
    for c in core_cases:
        band = [x for x in c["sweep"] if f1c <= x["f_Hz"] <= f2c]
        if len(band) < 3:
            continue
        ts = {}
        for tag, key in (("MQS", None), ("fullwave_L_minus_ImZ", "F_SI_T4_H2_m8"),
                         ("fullwave_L_eff_foster", "F_eff_SI_T4_H2_m8")):
            # time per unit ln(nu) at frequency f is proportional to 1/F(f); integrate
            lf, inv = [], []
            for x in band:
                F = c["F_MQS_SI"] if key is None else x.get(key)
                if not F or F <= 0:
                    continue
                b = {"name": "core", "Q": sic["Q"][0], "T": sic["T_K"][0],
                     "eta_A": eta_A_at(x["f_Hz"])}
                # F_req over a narrow window around f, per unit ln f, from the local rate
                req = af.requirement(b, live_time=1.0, f_min=x["f_Hz"] * 0.995,
                                     f_max=x["f_Hz"] * 1.005, model="DFSZ",
                                     rho_DM=sic["rho_DM_GeV_cm3"][0], Q_axion=sic["Q_axion"][0],
                                     Theta95=sic["SNR"][0] ** 2)
                lf.append(math.log(x["f_Hz"]))
                inv.append(req["F_req_natural_T4_m10"] / (F / MU0 ** 2) / math.log(1.005 / 0.995))
            ts[tag] = (float(np.trapezoid(inv, lf)) if len(lf) > 2 else None,
                       len(lf), len(band))
        core_rows.append({
            "name": c["name"], "band_Hz": [f1c, f2c],
            "scan_time_MQS_yr": ts["MQS"][0],
            "scan_time_fullwave_yr": ts["fullwave_L_minus_ImZ"][0],
            "scan_time_fullwave_L_eff_yr": ts["fullwave_L_eff_foster"][0],
            "band_points_used": {k: v[1] for k, v in ts.items()},
            "band_points": ts["MQS"][2],
            "published_live_time_yr": sic["live_time_yr"][0],
            "fullwave_over_MQS": (ts["fullwave_L_eff_foster"][0] / ts["MQS"][0]
                                  if ts["MQS"][0] and ts["fullwave_L_eff_foster"][0] else None)})
    out["core_scan_time_beyond_mqs"] = {
        "rows": core_rows, "eta_A": "published dc-SQUID optimum, frequency by frequency",
        "eta_A_at_band_edges": [eta_A_at(f1c), eta_A_at(f2c)],
        "Q": sic["Q"][0], "T_K": sic["T_K"][0], "SNR": sic["SNR"][0],
        "note": "Time to cover 70-140 MHz at DFSZ, integrating the local scan rate with the local "
                "F.  'MQS' uses the constant omega -> 0 F; the full-wave columns use F(omega) from "
                "the sweep, with L(omega) = -Im Z/omega and with DMRadio's Foster L_eff.  The gap "
                "between them is how much of DMRadio-Core's reach comes from operating near the "
                "pickup's self-resonance, which the MQS numbers cannot see.",
    }

    ok = all(c["sweep"] for c in cases)
    rr = out["m3_self_resonances"]
    return record("fullwave", {
        **out, "pass": bool(ok and (rr["max_abs_rel_with_end_correction"] or 1) < 0.08),
        "pass_scope": ("the six coax self-resonances against the published c/(4 h) with a single "
                       "fitted open-end extension (a genuine agreement with an independent "
                       "published model), plus sweep sanity.  It does NOT test any scan time."),
        "headline": (f"m3: the six published coaxes resonate up to "
                     f"{(rr['max_abs_rel_to_lambda_quarter'] or 0) * 100:.0f} % below c/(4 h), all "
                     f"absorbed by one open-end extension dl = "
                     f"{(rr['end_extension_fitted_m'] or 0) * 100:.0f} cm = "
                     f"{(rr['end_extension_over_gap_width'] or 0):.2f} x the gap width "
                     f"(residual {(rr['max_abs_rel_with_end_correction'] or 0) * 100:.1f} %); MQS good to "
                     f"{(rr['rows'][0]['mqs_within_5pct_up_to_Hz'] or 0) / 1e6:.1f} MHz (5 %) for the "
                     f"tallest; Core pickups resonate at "
                     + ", ".join(f"{s['first_self_resonance_Hz'] / 1e6:.0f}" for s in summ
                                 if s["name"].startswith("dmradio_core") and s["first_self_resonance_Hz"])
                     + " MHz"),
    })


# ============================== 3D: DMRadio-Core's 30-degree readout port
def design_core_reduced(outer_diameter: float = 1.20, H_coil: float = 0.60,
                        NI_full: float | None = None, mesh_size: float = 0.05,
                        order: int = 2) -> af.Design:
    """The published DMRadio-Core pickup around a SHORTENED coil.

    The published magnet is 2.95 m tall, which makes a 3D full-wave domain at
    12 nodes per wavelength over 70-140 MHz far too large.  The port-geometry
    effect we are after is a property of the pickup, so we shorten the coil (at
    the same current density, hence the same peak field) and compare the
    axisymmetric and 3D solves of the *same* reduced design.  L_p is set by the
    pickup alone and is unchanged by the shortening; Psi is smaller because
    there is less magnet, which cancels in the 3D/2D ratio.
    """
    g = SRC["dmradio_core"]["geometry"]
    Rc = g["bore_diameter_m"][0] / 2
    tw = g["outer_diameter_m"][0] / 2 - Rc
    a = g["pickup_inner_diameter_m"][0] / 2
    b = outer_diameter / 2
    w = 0.005
    NI_full = NI_full if NI_full is not None else 1.18943e7
    d = af.template("core", Rc=Rc, tw=tw, H=H_coil, gc=g["plate_spacing_m"][0], a=a, b=b,
                    h=g["pickup_height_m"][0], w=w, f=0.02, gn=g["plate_spacing_m"][0],
                    fl=0.3 * w, NI=NI_full * H_coil / g["H_m"][0])
    d.doc["mesh"] = {"order": order, "size": mesh_size, "surface_size": w}
    d.doc["name"] = f"core_reduced_{int(outer_diameter * 100)}cm"
    d.doc["description"] = (f"DMRadio-Core pickup (outer diameter {outer_diameter * 100:.0f} cm) around a "
                            f"{H_coil} m coil instead of the published 2.95 m, so that the 3D full-wave "
                            "domain is tractable; used only for the 3D-vs-axisymmetric port comparison.")
    d.doc["notes"] = "Dimensions and sources: benchmarks/published/sources.json, key 'dmradio_core'."
    return d


def design_core_3d_port(d2: af.Design, port_deg: float = 30.0, mesh_size: float = 0.12,
                        order: int = 2, surface_size: float = 0.025) -> af.Design:
    """`d2` revolved to 3D with the readout confined to a `port_deg` sector of the slit.

    `to_3d` revolves a poloidal-port pickup over the full 360 degrees, i.e. the
    uniform annular feed the axisymmetric solver assumes.  DMRadio-Core instead
    connects its tuning and readout across "a 30 degree portion of the slit"
    (CORE L512-521), so here the revolved neck is split into a `port_deg` sector
    that carries the readout current and a remaining sector that stays an open
    (capacitive) gap.
    """
    from axionfem.convert import to_3d

    d3 = to_3d(d2)
    doc = d3.doc
    prof = next(r["shape"] for r in d2.doc["regions"] if r["name"] == "neck")
    regs = [r for r in doc["regions"] if r["name"] != "neck"]
    if port_deg >= 360.0:
        # the control: the full annular gap is the port, i.e. the 3D counterpart of
        # the axisymmetric uniform feed (no leftover sector to add)
        regs.append({"name": "neck_port", "material": "vacuum",
                     # The validator requires an explicit increasing range; its
                     # implicit default is a centred zero-width range in this
                     # 3D conversion path.
                     "solid": {"revolve": {"profile": prof, "angles": [0.0, 360.0]}},
                     "notes": "full 360 degree port: the 3D uniform-feed control"})
    else:
        regs += [
            {"name": "neck_port", "material": "vacuum",
             "solid": {"revolve": {"profile": prof, "angles": [-port_deg / 2, port_deg / 2]}},
             "notes": f"the published {port_deg:g} degree readout sector of the slit"},
            {"name": "neck_gap", "material": "vacuum",
             "solid": {"revolve": {"profile": prof, "angles": [port_deg / 2, 360 - port_deg / 2]}},
             "notes": "the rest of the slit: an open, undriven gap"}]
    doc["regions"] = regs
    doc["pickup"] = {"type": "port", "port": "neck_port"}
    doc["mesh"] = {"order": order, "size": mesh_size, "surface_size": surface_size}
    doc["name"] = d2.doc["name"] + f"_3d_port{int(port_deg)}"
    doc["description"] = (d2.doc["description"].rstrip(".") +
                          f"; revolved to 3D with the readout confined to a {port_deg:g} degree "
                          "sector of the slit.")
    return af.Design(doc)


CORE3D_PASS_SCOPE = ("that the 3D full-wave solves of the 30-degree port and the 360-degree control completed "
                     "(every point accepted at its true-residual gate) on the reduced design. Nothing about agreement "
                     "with DMRadio-Core's published scan time, which the reduced 0.6 m coil cannot reproduce; the "
                     "comparison is 3D vs the axisymmetric uniform feed (finite-difference dX/dw in these records; "
                     "see docs/validation/published_benchmarks.md section 7.2 for the exact-dZ/dw rerun).")


def check_core3d(mpi_ranks: int = 16, mesh_3d: float = 0.12, points: int = 7,
                 port_degs: tuple = (30.0, 360.0), tag: str = "") -> dict:
    """DMRadio-Core's non-axisymmetric readout port in 3D full wave, against the
    axisymmetric uniform-feed limit, over the published 70-140 MHz band."""
    sic = SRC["dmradio_core"]["sensitivity_inputs"]
    gc = SRC["dmradio_core"]["geometry"]
    f1, f2 = sic["f_range_Hz"][0]
    fw = {"f_min": f1, "f_max": f2, "points": points, "boundary": "radiation"}
    out: dict = {"band_Hz": [f1, f2], "mpi_ranks": mpi_ranks, "port_degs": list(port_degs),
                 "mesh_size_3d_m": mesh_3d, "sweep_points": points,
                 "note": (
        "The published magnet is 2.95 m tall; a 3D full-wave domain at 12 nodes per wavelength over "
        "70-140 MHz around it is out of reach, so the coil is shortened to 0.6 m at the same current "
        "density and the axisymmetric and 3D solves use the SAME reduced design.  What is compared is "
        "therefore the effect of the readout port's azimuthal extent, which is a property of the "
        "pickup; the absolute Psi of the reduced model is not the device's.")}

    d2 = design_core_reduced()
    r2 = d2.copy()
    r2.doc["fullwave"] = fw
    res2 = r2.solve(RUNS / f"core3d_axi{tag}", verbose=0)
    rows2 = _fullwave_rows(res2)
    out["axisymmetric"] = {
        "name": d2.doc["name"], "dofs": res2.data["mesh"]["dofs"],
        "Psi_MQS_natural_T_m3": res2.coupling["Psi_natural_T_m3"], "L_MQS_H": res2.L_p,
        "F_MQS_SI": res2.F, "sweep": rows2,
        "first_self_resonance_Hz": _first_self_resonance(rows2),
        "mesh_kh_max": res2.fullwave.get("mesh_kh_max"),
        "warnings": res2.data.get("warnings", [])}
    save_design(d2, "dmradio_core_reduced_120cm")

    out["three_d"] = []
    for pd in port_degs:
        surface_size = 0.025 if mesh_3d <= 0.16 else 0.06
        d3 = design_core_3d_port(d2, port_deg=pd, mesh_size=mesh_3d,
                                 surface_size=surface_size)
        save_design(d3, d3.doc["name"])
        row = {"port_deg": pd, "name": d3.doc["name"]}
        try:
            d3f = d3.copy()
            d3f.doc["fullwave"] = fw
            t0 = time.time()
            res3 = d3f.solve(RUNS / f"core3d_port{int(pd)}{tag}", verbose=0, mpi_ranks=mpi_ranks)
            rows3 = _fullwave_rows(res3)
            row.update({
                "dofs": res3.data["mesh"]["dofs"], "seconds": time.time() - t0,
                "Psi_MQS_natural_T_m3": res3.coupling["Psi_natural_T_m3"], "L_MQS_H": res3.L_p,
                "F_MQS_SI": res3.F, "sweep": rows3,
                "first_self_resonance_Hz": _first_self_resonance(rows3),
                "mesh_kh_max": res3.fullwave.get("mesh_kh_max"),
                "self_resonances": res3.fullwave.get("self_resonances"),
                "checks": res3.data.get("checks"), "warnings": res3.data.get("warnings", [])})
        except Exception as exc:                                     # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        out["three_d"].append(row)

    # ---- 3D / axisymmetric, frequency by frequency
    comps = []
    for row in out["three_d"]:
        if "sweep" not in row:
            continue
        per = []
        for a, b in zip(rows2, row["sweep"]):
            per.append({"f_Hz": a["f_Hz"],
                        "Psi_ratio": (b["Psi_abs"] / a["Psi_abs"]) if a.get("Psi_abs") else None,
                        "L_ratio": (b["L_H"] / a["L_H"]) if a.get("L_H") else None,
                        "L_eff_ratio": ((b["L_eff_foster_H"] / a["L_eff_foster_H"])
                                        if a.get("L_eff_foster_H") and b.get("L_eff_foster_H") else None),
                        "R_rad_3d_ohm": b.get("R_rad_ohm"), "R_rad_axi_ohm": a.get("R_rad_ohm"),
                        "F_eff_ratio": ((b["F_eff_SI_T4_H2_m8"] / a["F_eff_SI_T4_H2_m8"])
                                        if a.get("F_eff_SI_T4_H2_m8") and b.get("F_eff_SI_T4_H2_m8")
                                        else None)})
        good = [x for x in per if x["F_eff_ratio"]]
        comps.append({
            "port_deg": row["port_deg"], "per_frequency": per,
            "Psi_MQS_ratio": row["Psi_MQS_natural_T_m3"] / out["axisymmetric"]["Psi_MQS_natural_T_m3"],
            "L_MQS_ratio": row["L_MQS_H"] / out["axisymmetric"]["L_MQS_H"],
            "F_MQS_ratio": row["F_MQS_SI"] / out["axisymmetric"]["F_MQS_SI"],
            "F_eff_ratio_band_geomean": (math.exp(sum(math.log(x["F_eff_ratio"]) for x in good) / len(good))
                                         if good else None),
            "F_eff_ratio_min": min((x["F_eff_ratio"] for x in good), default=None),
            "F_eff_ratio_max": max((x["F_eff_ratio"] for x in good), default=None),
            "first_self_resonance_3d_Hz": row["first_self_resonance_Hz"],
            "first_self_resonance_axi_Hz": out["axisymmetric"]["first_self_resonance_Hz"]})
    out["comparison"] = comps

    # ---- propagate to the Core scan time with the published eta(omega)
    base = dict(model="DFSZ", rho_DM=sic["rho_DM_GeV_cm3"][0], Q_axion=sic["Q_axion"][0],
                Theta95=sic["SNR"][0] ** 2)
    scan_rows = []
    for c in comps:
        src_rows = next(r["sweep"] for r in out["three_d"] if r["port_deg"] == c["port_deg"])
        for model_tag, rows in (("axisymmetric_uniform_feed", rows2), ("three_d", src_rows)):
            lf, inv = [], []
            for x in rows:
                F = x.get("F_eff_SI_T4_H2_m8")
                if not F or F <= 0 or not (f1 <= x["f_Hz"] <= f2):
                    continue
                b = {"name": "core", "Q": sic["Q"][0], "T": sic["T_K"][0],
                     "eta_A": eta_A_at(x["f_Hz"])}
                req = af.requirement(b, live_time=1.0, f_min=x["f_Hz"] * 0.995,
                                     f_max=x["f_Hz"] * 1.005, **base)
                lf.append(math.log(x["f_Hz"]))
                inv.append(req["F_req_natural_T4_m10"] / (F / MU0 ** 2) / math.log(1.005 / 0.995))
            scan_rows.append({"port_deg": c["port_deg"], "model": model_tag, "points": len(lf),
                              "scan_time_yr": float(np.trapezoid(inv, lf)) if len(lf) > 2 else None})
    out["scan_time"] = {
        "rows": scan_rows, "published_live_time_yr": sic["live_time_yr"][0],
        "eta_A_at_band_edges": [eta_A_at(f1), eta_A_at(f2)],
        "note": "Scan time of the REDUCED design (shortened coil), so the absolute years are not the "
                "device's; the 3D/axisymmetric ratio is the result.  Published eta(omega) throughout."}
    ratio = None
    for c in comps:
        if c["port_deg"] == 30.0:
            ax = next((x["scan_time_yr"] for x in scan_rows
                       if x["port_deg"] == 30.0 and x["model"] == "axisymmetric_uniform_feed"), None)
            td = next((x["scan_time_yr"] for x in scan_rows
                       if x["port_deg"] == 30.0 and x["model"] == "three_d"), None)
            ratio = (td / ax) if (ax and td) else None
            out["scan_time"]["three_d_over_axisymmetric"] = ratio

    ok = any("sweep" in r for r in out["three_d"])
    c30 = next((c for c in comps if c["port_deg"] == 30.0), None)
    return record("core3d" + tag, {
        **out, "pass": bool(ok),
        "pass_scope": CORE3D_PASS_SCOPE,
        "headline": (
            (f"Core 30 deg port in 3D: F_eff is x{c30['F_eff_ratio_band_geomean']:.3g} the "
             f"axisymmetric uniform feed over 70-140 MHz "
             f"(Psi_MQS x{c30['Psi_MQS_ratio']:.3g}, L_MQS x{c30['L_MQS_ratio']:.3g}); "
             f"scan time x{ratio:.3g}" if c30 and c30.get("F_eff_ratio_band_geomean") and ratio
             else "3D Core port: see the record")),
    })


# ==================================================================== figures
def figures() -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    def logticks(ax, which="both"):
        """Decade labels only; matplotlib labels log minor ticks over narrow ranges."""
        for axis in ((ax.xaxis,) if which == "x" else (ax.yaxis,) if which == "y"
                     else (ax.xaxis, ax.yaxis)):
            if axis.get_scale() == "log":
                axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=99))
                axis.set_minor_formatter(NullFormatter())

    def load(name):
        p = REC / f"published_{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ---- Fig. 1: ABRA-10cm geometric factor
    a = load("abra10cm")
    if a:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
        rows = a["winding_thickness_convergence"]
        tw = np.array([r["winding_thickness_m"] for r in rows]) * 1e3
        err = np.array([r["rel_to_exact"] for r in rows]) * 100
        ax[0].plot(tw, err, "o-", color="#1f77b4")
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xscale("log")
        ax[0].set_xlabel("winding sheet thickness [mm]")
        ax[0].set_ylabel(r"FEM $V_B$ / exact $-$ 1  [%]")
        ax[0].set_title("ABRA-10 cm: FEM vs the exact\nreciprocity integral (free space)")
        ax[0].grid(alpha=0.3)

        s = a["shield_scan"]
        Rsh = np.array([r["shield_radius_m"] for r in s]) * 100
        G = np.array([r["G"] for r in s])
        ax[1].plot(Rsh, G, "s-", color="#d62728", label="FEM, with SC shield")
        ax[1].axhline(a["geometric_factor"]["G_V_FEM"], color="#1f77b4", ls="--",
                      label=f"FEM, free space ({a['geometric_factor']['G_V_FEM']:.4f})")
        ax[1].axhline(a["geometric_factor"]["G_V_published"], color="k", ls="-",
                      label=f"published ({a['geometric_factor']['G_V_published']}, COMSOL, shield-free)")
        ax[1].set_xlabel("superconducting shield radius [cm]  (not published)")
        ax[1].set_ylabel(r"geometric factor $G_V = V_B / V$   ($V = 890$ cm$^3$)")
        ax[1].set_title("ABRA-10 cm: geometric factor")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        p = FIG / "published_abra10cm_G.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # ---- Fig. 2: DMRadio-50L reach
    d50 = load("dmradio_50l")
    if d50:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        c = d50["reach"]["curve"]
        m = np.array(c["m_eV"])
        for t, style in ((1.0, "-"), (3.0, "--")):
            ax.plot(m, np.array(c["g95_1yr_GeV"]) * t ** -0.25, style, color="#1f77b4",
                    label=f"axionFEM, log-uniform, {t:g} yr")
        ax.plot(m, c["g_DFSZ_GeV"], color="#2ca02c", lw=1.2, label="DFSZ ($C_\\gamma=1.77$)")
        gp = d50["reach"]["g_reach_published_GeV"]
        ax.axhline(gp, color="k", lw=1.5, label=f"published projection ({gp:.0e})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xlabel(r"$m_a$ [eV]")
        ax.set_ylabel(r"$|g_{a\gamma\gamma}|$ [GeV$^{-1}$]")
        ax.set_title("DMRadio-50L: reach from the axionFEM coupling\n"
                     f"($\\Psi$ = {d50['mesh_convergence'][-1]['Psi_natural_T_m3']:.3e} T m$^3$, "
                     f"$L_p$ = {d50['mesh_convergence'][-1]['L_p_H'] * 1e9:.0f} nH, "
                     f"$Q=10^6$, 20 mK, $\\eta_A=20$)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = FIG / "published_dmradio_50l_reach.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # ---- Fig. 3: DMRadio-m3, each published coax over its own band
    m3 = load("dmradio_m3")
    if m3:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        cmap = plt.get_cmap("viridis")
        hs = sorted({x["h_coax_m"] for x in m3["reach_by_coax"]["rows"]}, reverse=True)
        seen = set()
        for x in m3["reach_by_coax"]["rows"]:
            if x["model"] != "DFSZ":
                continue
            col = cmap(hs.index(x["h_coax_m"]) / max(1, len(hs) - 1))
            lab = f"$h$ = {x['h_coax_m']} m" if x["h_coax_m"] not in seen else None
            seen.add(x["h_coax_m"])
            ax.plot(x["m_eV"], x["g95_GeV"], "-", color=col, lw=2.2, label=lab)
        for model, col in (("DFSZ", "#2ca02c"), ("KSVZ", "#d62728")):
            mt = np.array([v for y in m3["reach_by_coax"]["rows"] if y["model"] == model
                           for v in y["m_eV"]])
            gt = np.array([v for y in m3["reach_by_coax"]["rows"] if y["model"] == model
                           for v in y["g_target_GeV"]])
            o = np.argsort(mt)
            ax.plot(mt[o], gt[o], "--", color=col, lw=1.2, label=f"{model} line")
        ax.set_xscale("log")
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xlabel(r"$m_a$ [eV]")
        ax.set_ylabel(r"$|g_{a\gamma\gamma}|$ [GeV$^{-1}$]")
        ax.set_title("DMRadio-m$^3$: MQS reach of the six published coaxes over their own bands\n"
                     "($Q=10^5$, 20 mK, $\\eta_A=20$, 5 yr; MQS valid only below ~10 MHz, see §6)",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = FIG / "published_dmradio_m3_reach.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # ---- Fig. 4: GUT scan time, published vs ours
    gu = load("dmradio_gut")
    if gu:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        sc = gu["scan_time"]["scenarios"]
        x = np.arange(len(sc))
        w = 0.2
        ax[0].bar(x - 1.5 * w, [s["scan_time_published_yr"] for s in sc], w, label="GUT Table II")
        ax[0].bar(x - 0.5 * w, [s["scan_time_from_published_c_PU_Cgamma1.92_yr"] for s in sc], w,
                  label="Eq. (3), $c_{PU}=0.1$, their $C_{a\\gamma\\gamma}=0.75$")
        ax[0].bar(x + 0.5 * w, [s["scan_time_from_published_c_PU_yr"] for s in sc], w,
                  label="Eq. (3), $c_{PU}=0.1$, lattice $C_\\gamma=1.77$")
        ax[0].bar(x + 1.5 * w, [s["scan_time_ours_Cgamma1.92_yr"] for s in sc], w,
                  label=f"Eq. (3), our $c_{{PU}}$ = {sc[0]['c_PU_ours']:.3f}, "
                        "their $C_{a\\gamma\\gamma}$")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([s["scenario"].replace(" + ", "\n+ ") for s in sc], fontsize=7)
        ax[0].set_ylabel("scan time to DFSZ [yr]")
        ax[0].set_title("DMRadio-GUT: scan time, 0.4-120 neV")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3, axis="y")

        rsc = gu["scan_time"]["relative_scaling_check"]
        ax[1].plot([r["published_ratio_to_baseline"] for r in rsc],
                   [r["our_ratio_to_baseline"] for r in rsc], "o", ms=9)
        lim = [0, max(r["published_ratio_to_baseline"] for r in rsc) * 1.15]
        ax[1].plot(lim, lim, "k--", lw=1)
        for r in rsc:
            ax[1].annotate(r["scenario"].split(" +")[0], (r["published_ratio_to_baseline"],
                                                          r["our_ratio_to_baseline"]),
                           fontsize=7, xytext=(4, -9), textcoords="offset points")
        ax[1].set_xlabel("Table II scan time / baseline")
        ax[1].set_ylabel(r"our $\mathcal{F}_{\rm req}$ scaling / baseline")
        ax[1].set_title("Relative scaling of the four GUT scenarios")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        p = FIG / "published_dmradio_gut_scan_time.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # ---- Fig. 5: beyond MQS
    fw = load("fullwave")
    if fw:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
        for c in fw["cases"]:
            sw = c["sweep"]
            f = np.array([x["f_Hz"] for x in sw]) / 1e6
            lab = c["label"].replace("DMRadio-m^3 coax, ", "m$^3$ ").replace(
                "DMRadio-Core pickup, outer diameter ", "Core ")
            ls = "-" if "m$^3$" in lab else "--"
            ax[0].plot(f, [x.get("Psi_over_MQS", np.nan) for x in sw], ls, lw=1.2, label=lab)
            le = np.array([x.get("L_eff_over_MQS") or np.nan for x in sw], dtype=float)
            ax[1].plot(f, le, ls, lw=1.2)
            fe = np.array([x.get("F_eff_over_F_MQS") or np.nan for x in sw], dtype=float)
            ax[2].plot(f, fe, ls, lw=1.2)
        for a_, t, yl in ((ax[0], r"$|\Psi(\omega)|/\Psi_{\rm MQS}$", "coupling"),
                          (ax[1], r"$L_{\rm eff}(\omega)/L_{\rm MQS}$", "inductance (Foster)"),
                          (ax[2], r"$\mathcal{F}_{\rm eff}/\mathcal{F}_{\rm MQS}$", "figure of merit")):
            a_.axhline(1, color="k", lw=0.8, ls=":")
            a_.set_xscale("log")
            a_.set_yscale("log")
            logticks(a_, "x")
            a_.set_xlabel("frequency [MHz]")
            a_.set_ylabel(t)
            a_.set_title(yl, fontsize=10)
            a_.grid(alpha=0.3, which="both")
        ax[0].axvspan(5, 200, color="0.85", zorder=0)
        ax[0].legend(fontsize=6.5, ncol=2)
        fig.suptitle("Beyond MQS: DMRadio-m$^3$ coaxes (solid) and DMRadio-Core pickups (dashed); "
                     "shading = the m$^3$ band", fontsize=10)
        fig.tight_layout()
        p = FIG / "published_fullwave.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

        rr = fw["m3_self_resonances"]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        h = np.array([x["h_coax_m"] for x in rr["rows"]])
        fo = np.array([x["first_self_resonance_Hz"] for x in rr["rows"]]) / 1e6
        hh = np.linspace(0.45, 1.85, 100)
        ax.plot(hh, 299.792458 / (4 * hh), "k--", lw=1.2, label=r"published $c/4h$")
        ax.plot(hh, 299.792458 / (4 * (hh + rr["end_extension_fitted_m"])), "-", color="#2ca02c",
                lw=1.2, label=rf"$c/4(h+\delta\ell)$, $\delta\ell$ = {rr['end_extension_fitted_m'] * 100:.0f} cm")
        ax.plot(h, fo, "o", ms=9, color="#1f77b4", label="axionFEM full wave", zorder=3)
        ax.set_xlabel(r"coax height $h_{\rm coax}$ [m]")
        ax.set_ylabel("first self-resonance [MHz]")
        ax.set_title("DMRadio-m$^3$: the six published coaxes", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = FIG / "published_m3_resonances.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # ---- Fig. 6: the coupling of all the detectors on one axis
    allrec = {n: load(n) for n in ("abra10cm", "dmradio_50l", "dmradio_m3", "dmradio_gut")}
    pts = []
    if allrec["abra10cm"]:
        a = allrec["abra10cm"]
        pts.append(("ABRA-10 cm", a["dmradio_language"]["c_PU"], None, a["dmradio_language"]["V_field_m3"]))
    if allrec["dmradio_50l"]:
        d = allrec["dmradio_50l"]
        pts.append(("DMRadio-50L", d["mesh_convergence"][-1]["c_PU"], None, d["inputs"]["V_field_m3"]))
    if allrec["dmradio_m3"]:
        d = allrec["dmradio_m3"]
        pts.append(("DMRadio-m$^3$", d["comparison"]["c_PU_lowest_coax_ours_V_cylinder"],
                    d["comparison"]["c_PU_published_range"],
                    d["comparison"]["V_PU_tallest_cylinder_m3"]))
    if allrec["dmradio_gut"]:
        d = allrec["dmradio_gut"]
        pts.append(("DMRadio-GUT", d["mesh_convergence"][-1]["c_PU"], [0.1, 0.1],
                    d["inputs"]["V_m3"]))
    if pts:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for lab, c, pub, V in pts:
            ax.plot(V, c, "o", ms=10, color="#1f77b4", zorder=3)
            ax.annotate(lab, (V, c), xytext=(6, 7), textcoords="offset points", fontsize=9)
            if pub and pub[0] != pub[1]:
                ax.plot([V, V], pub, "-", color="k", lw=4, alpha=0.5, solid_capstyle="butt")
            elif pub:
                ax.plot([V * 0.75, V * 1.33], [pub[0]] * 2, "-", color="k", lw=4, alpha=0.5)
        ax.plot([], [], "o", color="#1f77b4", label="axionFEM")
        ax.plot([], [], "-", color="k", lw=4, alpha=0.5, label="published value / range")
        ax.set_xscale("log")
        ax.set_yscale("log")
        logticks(ax)
        ax.set_ylim(0.012, 0.3)
        ax.set_xlabel(r"field / pickup volume [m$^3$]")
        ax.set_ylabel(r"$c_{PU}$")
        ax.set_title("Pickup coupling in DMRadio units")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = FIG / "published_coupling_summary.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    for p in made:
        print(f"  figure {p.relative_to(ROOT)}")
    return made


# ==================================================================== report
def report() -> Path:
    from published_report import write_report       # noqa: PLC0415
    return write_report(ROOT, REC, SRC)


CHECKS = {"abra10cm": check_abra10cm, "dmradio_50l": check_dmradio_50l,
          "dmradio_m3": check_dmradio_m3, "dmradio_gut": check_dmradio_gut,
          "dmradio_core": check_dmradio_core, "fullwave": check_fullwave,
          "core3d": lambda: check_core3d(mpi_ranks=int(os.environ.get("AXF_CORE3D_RANKS", "16"))),
          # a cheaper variant that still resolves the port effect, for when the full
          # sweep does not fit in a wall-clock allocation
          "core3d_fast": lambda: check_core3d(mesh_3d=0.16, points=4, tag="_fast"),
          # coarse diagnostic fallback for allocations where the ND2 mesh does not
          # reach the first 3D solve in time; never substitute it silently.
          "core3d_coarse": lambda: check_core3d(mesh_3d=0.28, points=4, tag="_coarse")}


def write_designs() -> None:
    """Write every design JSON without solving (fast; used by the tests)."""
    save_design(design_abra10cm(tw=2e-4), "abra10cm")
    save_design(design_abra10cm(tw=2e-4, shield_radius=0.09), "abra10cm_shielded")
    save_design(design_dmradio_50l(), "dmradio_50l")
    save_design(design_dmradio_gut(), "dmradio_gut")
    save_design(design_dmradio_m3(NI=6.6e6), "dmradio_m3")
    g = SRC["dmradio_core"]["geometry"]
    NI = (g["outer_diameter_m"][0] / 2 - g["bore_diameter_m"][0] / 2) / 3e-3 * (g["H_m"][0] / 3e-3) * 330.0
    save_design(design_core_magnet(NI), "dmradio_core_magnet")
    for od in g["pickup_outer_diameter_m"][0]:
        save_design(design_dmradio_core(NI, outer_diameter=od), f"dmradio_core_pickup_{int(od * 100)}cm")
    for p in sorted(DES.glob("*.json")):
        if p.name != "sources.json":
            af.load(p).validate()
            print(f"  design {p.relative_to(ROOT)} (valid)")


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    names = argv or ["all"]
    if names == ["designs"]:
        write_designs()
        return 0
    if names == ["report"]:
        figures()
        print(f"  report {report().relative_to(ROOT)}")
        return 0
    todo = list(CHECKS) if names == ["all"] else names
    bad = [n for n in todo if n not in CHECKS]
    if bad:
        print(f"unknown check(s) {bad}; available: {', '.join(CHECKS)} (or 'all', 'designs', 'report')")
        return 2
    results = {}
    for n in todo:
        t0 = time.time()
        results[n] = CHECKS[n]()
        print(f"    ({time.time() - t0:.0f} s)", flush=True)
    if names == ["all"]:
        figures()
        print(f"  report {report().relative_to(ROOT)}")
    print(f"\nNOTE on [SELF-CHECK PASS]: {PASS_MEANS}", flush=True)
    return 0 if all(r.get("pass") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
