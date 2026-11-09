"""Conductor critical surfaces, current margins and the critical-state (screening-current) solver.

Fast tests: the Jc laws against published reference points and their shape, the margin arithmetic on a
solenoid, and a coarse critical-state run (sign and magnitude of the screening currents, the exact
backward-Euler energy identity, trapped flux, sampling).  The Bean magnetization curve and the AC loss
against the analytic hysteresis loop are slow and gated by AXF_SLOW; the full study is
benchmarks/hts_validation.py (docs/validation/hts.md).
"""
import math
import os

import pytest

import axionfem as af

MU0 = 1.25663706212e-6


# ---------------------------------------------------------------- critical surfaces
def test_builtin_conductors_reference_points():
    """Published values of the fits that the built-ins implement (docs/validation/hts.md)."""
    # NbTi, Bottura fit with the LHC parameters: Jc(5 T, 4.2 K) ~ 3000 A/mm^2
    assert af.jc("nbti", 5.0) / 1e6 == pytest.approx(3263, rel=2e-3)
    assert af.jc("nbti", 8.0) / 1e6 == pytest.approx(1355, rel=2e-3)
    assert af.jc("nbti", 10.0, 0.0, 1.9) / 1e6 == pytest.approx(1918, rel=2e-3)
    # Nb3Sn, Durham/ITER scaling at zero and at -0.25 % strain
    assert af.jc("nb3sn", 12.0) / 1e6 == pytest.approx(1058.6, rel=2e-3)
    assert af.jc({"law": "nb3sn_durham", "strain": -0.0025}, 12.0) / 1e6 == pytest.approx(940.9, rel=2e-3)
    # REBCO, anisotropic Kim (Grilli et al. set #1 at 77 K): Jc0 and a 4 mm x 1 um tape
    assert af.jc("rebco", 0.0) == pytest.approx(4.75e10, rel=1e-6)
    assert af.jc("rebco", 0.0) * 4e-3 * 1e-6 == pytest.approx(190.0, rel=1e-6)
    # the Kim denominator: Jc(B_perp)/Jc0 = (1 + B/B0)^-b
    assert af.jc("rebco", 0.1) / 4.75e10 == pytest.approx((1 + 0.1 / 0.035) ** -0.6, rel=1e-9)
    # a field in the tape plane is weighted by the anisotropy k = 0.25
    assert af.jc("rebco", 0.0, 0.1) / 4.75e10 == pytest.approx((1 + 0.25 * 0.1 / 0.035) ** -0.6, rel=1e-9)
    assert af.jc("rebco", 0.1) < af.jc("rebco", 0.0, 0.1)


def test_critical_surface_shape():
    """Monotone in B and T, zero above Bc2(T) and above Tc, and Bc2(T) itself."""
    for name in ("nbti", "nb3sn"):
        js = [af.jc(name, b) for b in (0.5, 1.0, 3.0, 6.0, 9.0)]
        assert all(a > b > 0 for a, b in zip(js, js[1:])), name
        ts = [af.jc(name, 3.0, 0.0, t) for t in (2.0, 4.0, 6.0, 8.0)]
        assert all(a > b > 0 for a, b in zip(ts, ts[1:])), name
    assert af.jc("nbti", 20.0) == 0.0              # above Bc20 = 14.5 T
    assert af.jc("nbti", 1.0, 0.0, 9.5) == 0.0     # above Tc0 = 9.2 K
    assert af.jc("rebco", 1.0, 0.0, 95.0) == 0.0   # above Tc = 92 K
    # Bc2(T) of the Bottura fit: Jc -> 0 as B -> Bc20 (1 - (T/Tc0)^1.7)
    Bc2 = 14.5 * (1 - (4.2 / 9.2) ** 1.7)
    assert af.jc("nbti", Bc2 * 0.999) > 0
    assert af.jc("nbti", Bc2 * 1.001) == 0.0


def test_table_conductor_and_low_field_floor():
    tab = {"law": "table", "B_T": [0.0, 1.0, 2.0], "Jc": [2e9, 1e9, 0.0], "T_op": 20.0}
    assert af.jc(tab, 0.5) == pytest.approx(1.5e9)          # linear between nodes
    assert af.jc(tab, 3.0) == pytest.approx(0.0)            # clamped at the last node
    assert af.jc(tab, -0.0) == pytest.approx(2e9)
    tab2d = dict(tab, T_K=[10.0, 30.0], Jc=[4e9, 2e9, 2e9, 1e9, 0.0, 0.0])
    assert af.jc(tab2d, 0.0, 0.0, 20.0) == pytest.approx(3e9)
    # the NbTi fit diverges as B -> 0; Jc is evaluated at max(|B|, B_min)
    assert af.jc("nbti", 0.0) == af.jc("nbti", 0.05) == af.jc("nbti", 0.01)


def test_unknown_conductor_is_an_error():
    doc = {"schema": "axionfem.design/1", "name": "x",
           "regions": [{"name": "c", "shape": {"rect": {"r": [1, 1.1], "z": [-1, 1]}}, "material": "winding"}],
           "windings": [{"region": "c", "type": "azimuthal", "ampere_turns": 1e5, "conductor": "unobtainium"}]}
    with pytest.raises(Exception, match="unknown conductor"):
        af.Design(doc).validate()


# ---------------------------------------------------------------- margins
def _solenoid(tmp_path, **cond):
    d = af.template("solenoid", R=1.0, H=2.0, mesh={"order": 1, "size": 0.2, "surface_size": 0.02})
    d.doc["windings"][0]["conductor"] = "coil"
    d.doc["conductors"] = {"coil": dict({"law": "nbti_bottura", "T_op": 4.2, "fill_factor": 0.3}, **cond)}
    d.doc["cost"] = {"load_line_cap": 0.8}
    return d.solve(tmp_path / "sol", verbose=0)


@pytest.fixture(scope="module")
def sol(tmp_path_factory):
    return _solenoid(tmp_path_factory.mktemp("sol"))


def test_solenoid_margins(sol):
    r = sol
    m = r.conductors["solenoid"]
    # engineering current density = ampere-turns / winding area; the superconductor carries J_e/fill
    assert m["J_engineering_A_m2"] == pytest.approx(1e6 / (0.01 * 2.0), rel=1e-6)
    assert m["J_superconductor_A_m2"] == pytest.approx(m["J_engineering_A_m2"] / 0.3, rel=1e-12)
    # the operating point is well below Jc, so the load line meets the critical surface far above it
    assert 0 < m["load_line_fraction"] < 0.1
    assert m["short_sample"]["scale"] > 5
    assert m["ok"] is True
    # the reported Jc is the law evaluated at the worst point of the winding
    w = m["worst_point"]
    assert af.jc("coil" if False else {"law": "nbti_bottura", "T_op": 4.2}, w["B_perp_T"], w["B_par_T"], 4.2) \
        == pytest.approx(m["Jc_A_m2"], rel=1e-9)
    # at the short-sample scale the current equals Jc at the scaled field
    s = m["short_sample"]["scale"]
    jc_s = af.jc({"law": "nbti_bottura", "T_op": 4.2}, s * w["B_perp_T"], s * w["B_par_T"], 4.2)
    assert s * m["J_superconductor_A_m2"] == pytest.approx(jc_s, rel=1e-6)
    # the temperature margin is where Jc(T_cs) = J
    jc_T = af.jc({"law": "nbti_bottura", "T_op": 4.2}, w["B_perp_T"], w["B_par_T"], m["current_sharing_T_K"])
    assert jc_T == pytest.approx(m["J_superconductor_A_m2"], rel=1e-5)
    assert m["temperature_margin_K"] == pytest.approx(m["current_sharing_T_K"] - 4.2)


def test_margins_scale_with_the_benchmark_amplitude(sol):
    r = sol
    b = r.benchmark("optimistic")
    s = b["amplitude_scale"]
    a, c = r.conductors["solenoid"], b["conductor_margins"]["solenoid"]
    assert c["J_superconductor_A_m2"] == pytest.approx(s * a["J_superconductor_A_m2"], rel=1e-9)
    assert c["worst_point"]["B_T"] == pytest.approx(s * a["worst_point"]["B_T"], rel=1e-9)
    # the load line is the same line, so the short-sample scale moves by 1/s
    assert c["short_sample"]["scale"] * s == pytest.approx(a["short_sample"]["scale"], rel=1e-6)
    # at the cost-optimal size lengths scale by l, so J scales as (field scale)/l
    o = b["cost_optimal_size"]
    l, co = o["length_scale"], o["conductor_margins"]["solenoid"]
    fs = s * l ** -2.5
    assert co["worst_point"]["B_T"] == pytest.approx(fs * a["worst_point"]["B_T"], rel=1e-9)
    assert co["J_superconductor_A_m2"] == pytest.approx(fs / l * a["J_superconductor_A_m2"], rel=1e-9)
    assert o["caps"]["load_line_fraction"] == pytest.approx(co["load_line_fraction"], rel=1e-12)


def test_over_critical_winding_warns(tmp_path):
    # a 10x smaller fill factor puts the winding over its critical current
    r = _solenoid(tmp_path, fill_factor=0.003, Tc0=9.2)
    m = r.conductors["solenoid"]
    assert m["load_line_fraction"] > 1 and m["ok"] is False
    assert any("critical current" in w or "critical surface" in w for w in r.data["warnings"])


# ---------------------------------------------------------------- critical state
def _cylinder(tmp_path, *, a=0.05, Hc=0.2, Jc=1e6, n=25, h=0.01, amp=2.0, scales=(0.0, 1.0),
              steps=6, field_cooled=False, fov=((0.0, 0.0),), order=2):
    """Solid HTS cylinder in a uniform axial field ramped by the drive waveform."""
    Bp = MU0 * Jc * a
    t_ramp = amp * Bp * a / (2 * 1e-4)  # (dB/dt) a/2 = E0
    doc = {"schema": "axionfem.design/1", "name": "cyl",
           "conductors": {"bean": {"law": "table", "B_T": [0.0, 100.0], "Jc": [Jc, Jc], "n_value": n,
                                   "E0": 1e-4, "T_op": 20.0}},
           "regions": [{"name": "bulk", "shape": {"rect": {"r": [0, a], "z": [-Hc / 2, Hc / 2]}},
                        "material": "superconductor", "mesh_size": h}],
           "background": {"type": "uniform", "Bz": amp * Bp},
           "hts": {"regions": [{"region": "bulk", "conductor": "bean",
                                "initial": "field_cooled" if field_cooled else "zero_field_cooled"}],
                   "waveform": {"times": [t_ramp * i for i in range(len(scales))], "scales": list(scales)},
                   "steps_per_segment": steps, "max_newton": 60,
                   "field_of_view": [list(p) for p in fov], "profile_points": 15},
           "mesh": {"order": order, "size": 0.12, "domain_radius": 1.0}}
    res = af.Design(doc).solve(tmp_path / "cyl", verbose=0)
    V = math.pi * a * a * Hc
    return res, {"Bp": Bp, "V": V, "m_full": -Jc * a / 3 * V, "Jc": Jc}


@pytest.fixture(scope="module")
def cyl(tmp_path_factory):
    return _cylinder(tmp_path_factory.mktemp("cyl"))


def test_critical_state_full_penetration(cyl):
    """Ramping a zero-field-cooled cylinder to 2 Hp fills it with |J| ~ Jc; the Bean moment is
    m = -Jc V a/3 independently of the aspect ratio."""
    res, ref = cyl
    h = res.hts
    fin = h["final"]["regions"][0]
    assert fin["moment_Am2"] / ref["m_full"] == pytest.approx(1.0, abs=0.08)
    assert 0.8 < fin["max_J_over_Jc"] < 1.05            # power law at E ~ E0
    assert fin["screening_current_A"] < 0               # diamagnetic
    # the field at the centre is mu0 (Ha - Hp) once penetrated (finite length and finite n shift it)
    Bz0 = h["steps"][-1]["field_of_view"][0]["Bz_T"]
    assert Bz0 == pytest.approx(ref["Bp"], rel=0.25)
    # the exact backward-Euler energy identity holds for every solved step
    assert h["energy_balance_backward_euler_worst_rel"] < 1e-2
    assert h["total_loss_J"] > 0 and h["total_work_J"] > h["total_loss_J"]
    # every quadrature point stays at or below the critical current density
    for s in h["steps"]:
        assert s["regions"][0]["max_J_over_Jc"] < 1.1


def test_critical_state_trapped_flux_and_cooldown(tmp_path):
    """After a ramp-down the moment reverses sign (trapped flux); a field-cooled sample traps more
    than a zero-field-cooled one cycled over the same range (Bean: 1 vs 0.75 of m_full)."""
    zfc, ref = _cylinder(tmp_path / "zfc", amp=1.0, scales=(0.0, 1.0, 0.0), steps=8)
    fc, _ = _cylinder(tmp_path / "fc", amp=1.0, scales=(1.0, 0.0), steps=8, field_cooled=True)
    tz = zfc.hts["final"]["trapped"]
    tf = fc.hts["final"]["trapped"]
    assert tz["moment_Am2"] > 0 and tf["moment_Am2"] > 0          # trapped flux, opposite to the drive
    rz = tz["moment_Am2"] / -ref["m_full"]
    rf = tf["moment_Am2"] / -ref["m_full"]
    assert rz == pytest.approx(0.75, abs=0.15)
    assert rf == pytest.approx(1.0, abs=0.15)
    assert rf > rz
    assert tz["points"][0]["Bz_T"] > 0                            # trapped field at the centre


def test_critical_state_sampling_and_profile(cyl):
    res, ref = cyl
    s = res.hts_sample([0.0, 0.5 * 0.05, 0.99 * 0.05, 0.5], [0.0, 0.0, 0.0, 0.0])
    assert list(s["in_hts"]) == [1.0, 1.0, 1.0, 0.0]
    assert all(abs(j) <= 1.1 * ref["Jc"] for j in s["J"][:3])
    assert s["J"][2] < 0                                   # screening current near the surface
    assert s["Jc"][0] == pytest.approx(ref["Jc"], rel=1e-9)
    assert s["J"][3] == 0.0 and s["Jc"][3] == 0.0          # outside the HTS region
    prof = res.hts["final"]["current_profiles"][0]["points"]
    assert len(prof) == 15
    assert all(p["J_A_m2"] <= 0 for p in prof)


def test_field_cooled_hold_is_static(tmp_path):
    """Field cooled and held at constant drive: no flux changes, so no screening currents, no
    dissipation, and the total field equals the applied field everywhere."""
    res, ref = _cylinder(tmp_path, amp=1.0, scales=(1.0, 1.0), steps=3, field_cooled=True,
                         fov=((0.0, 0.0), (0.04, 0.0)))
    h = res.hts
    fin = h["final"]["regions"][0]
    assert abs(fin["max_J_over_Jc"]) < 1e-8
    assert h["total_loss_J"] < 1e-20 * max(1.0, abs(h["total_work_J"]))
    for s in h["steps"]:
        for p in s["field_of_view"]:
            assert abs(p["scif_Bz_T"]) < 1e-10 * ref["Bp"]
            assert p["Bz_T"] == pytest.approx(ref["Bp"], rel=1e-9)   # the applied field, unscreened


def test_critical_state_input_checks(tmp_path):
    base = {"schema": "axionfem.design/1", "name": "bad",
            "regions": [{"name": "w", "shape": {"rect": {"r": [1, 1.1], "z": [-1, 1]}}, "material": "winding"},
                        {"name": "s", "shape": {"rect": {"r": [0.2, 0.3], "z": [-0.5, 0.5]}}, "material": "nb"}],
            "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": 1e5}],
            "hts": {"regions": [{"region": "s", "conductor": "rebco"}], "drive": {"t_ramp": 1.0}}}
    af.Design(base).validate()                      # the baseline design is valid
    bad = {k: v for k, v in base.items()}
    bad["hts"] = dict(base["hts"], regions=[{"region": "w", "conductor": "rebco"}])
    with pytest.raises(Exception, match="not made of a superconductor"):
        af.Design(bad).validate()
    bad["hts"] = dict(base["hts"], regions=[{"region": "s", "conductor": "nosuch"}])
    with pytest.raises(Exception, match="unknown conductor"):
        af.Design(bad).validate()
    bad["hts"] = dict(base["hts"], regions=[{"region": "s", "initial": "warm"}])
    with pytest.raises(Exception, match="zero_field_cooled"):
        af.Design(bad).validate()
    # a zero-field-cooled run must start at drive scale 0
    bad = dict(base)
    bad["hts"] = {"regions": [{"region": "s", "conductor": "rebco"}],
                  "waveform": {"times": [0.0, 1.0], "scales": [1.0, 0.0]}}
    with pytest.raises(Exception, match="zero-field-cooled"):
        af.Design(bad).solve(tmp_path / "zfc_bad", verbose=0)
    # a toroidal static field (poloidal winding) is not supported
    bad = dict(base)
    bad["regions"] = [{"name": "w", "shape": {"frame": {"r": [0.5, 1.0], "z": [-0.5, 0.5], "thickness": 0.05}},
                       "material": "winding"},
                      {"name": "s", "shape": {"rect": {"r": [0.2, 0.3], "z": [-0.5, 0.5]}}, "material": "nb"}]
    bad["windings"] = [{"region": "w", "type": "poloidal", "ampere_turns": 1e5}]
    with pytest.raises(Exception, match="needs a poloidal static field"):
        af.Design(bad).solve(tmp_path / "pol", verbose=0)


# ---------------------------------------------------------------- slow: Bean references
@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow (a few minutes); set AXF_SLOW=1")
def test_bean_magnetization_curve(tmp_path):
    """Virgin magnetization of a long cylinder against -(Jc a/3)[1 - (1 - h)^3]."""
    res, ref = _cylinder(tmp_path, Hc=1.0, n=50, h=0.0035, amp=1.6, steps=32)
    for s in res.hts["steps"]:
        h = s["scale"] * 1.6
        if not 0.3 < h < 0.9:
            continue
        bean = 1 - (1 - h) ** 3
        assert s["regions"][0]["moment_Am2"] / ref["m_full"] == pytest.approx(bean, rel=0.08), h


@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow (a few minutes); set AXF_SLOW=1")
def test_bean_ac_loss_per_cycle(tmp_path):
    """AC loss of a symmetric cycle against the area of the Bean hysteresis loop."""
    import numpy as np

    h0 = 1.0
    res, ref = _cylinder(tmp_path, Hc=1.0, n=50, h=0.0035, amp=h0, steps=24,
                         scales=(0.0, 1.0, -1.0, 1.0, -1.0))
    def Mv(x):
        return -(1 - (1 - min(x, 1.0)) ** 3) / 3.0
    hs = np.linspace(-h0, h0, 20001)
    dec = np.array([Mv(h0) - 2 * Mv((h0 - x) / 2) for x in hs])
    asc = -np.array([Mv(h0) - 2 * Mv((h0 + x) / 2) for x in hs])
    area = float(np.trapezoid(dec - asc, hs) if hasattr(np, "trapezoid") else np.trapz(dec - asc, hs))
    Hp = ref["Bp"] / MU0
    q_bean = MU0 * Hp * Hp * area * ref["V"]
    cum = [s["cumulative_loss_J"] for s in res.hts["steps"]]
    k = 24
    q = cum[4 * k] - cum[2 * k]
    assert q / q_bean == pytest.approx(1.0, rel=0.15)
