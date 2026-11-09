"""SRF heterodyne sensitivity and cost (python/axionfem/heterodyne.py).  Fast (< 30 s), login node."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from axionfem import circuit
from axionfem import heterodyne as H

RNG = np.random.default_rng(20261023)

# A second, rounded cell (a test fixture) for checks that compare two geometries.
ROUNDED = H.SRFGeometry(xi=7.7, eta=0.45, G_pump_ohm=520.0, G_signal_ohm=600.0, Bpk_over_B0=1.4, area_coeff=6.2,
                        Epk_over_cB0=1.9, name="rounded cell (test fixture)")


# ----------------------------------------------------------------------------- receiver factor
@pytest.mark.parametrize("n_e", [0.0, 0.3, 7.0])
def test_gbar_srf_equals_lumped_gbar_under_substitution(n_e):
    """xi -> A/(tau beta), eta_A -> A/tau, 2n+1 -> 2 (n_i - n_e) + A/tau turns the lumped G-bar (circuit.G_bar,
    before the optimisation over M) into G-bar_SRF exactly."""
    for _ in range(50):
        beta = 10 ** RNG.uniform(-2, 4)
        tau = RNG.uniform(0.2, 1.0)
        n_i = 10 ** RNG.uniform(-1, 3) + n_e
        A = 0.5 + RNG.uniform(0, 20) + tau * n_e
        s = H.lumped_substitution(beta, A, n_i, tau, n_e)
        lumped = circuit.G_bar(s["xi"], s["n"], s["eta_A"])
        assert float(H.gbar_srf(beta, A, n_i, tau, n_e)) == pytest.approx(lumped, rel=1e-12)
        if n_e:
            # the n_e = 0 form of the replacement, 2n+1 -> 2 n_i + A/tau, is NOT exact once n_e != 0
            n_lit = n_i + 0.5 * A / tau - 0.5
            assert abs(circuit.G_bar(s["xi"], n_lit, s["eta_A"]) / lumped - 1) > 1e-9


def test_beta_opt_is_the_maximum():
    for _ in range(40):
        tau = RNG.uniform(0.2, 1.0)
        n_e = RNG.choice([0.0, RNG.uniform(0, 5)])
        n_i = n_e + 10 ** RNG.uniform(-2, 4)
        A = 0.5 + RNG.uniform(0, 20) + tau * n_e
        b = float(H.beta_opt(A, n_i, tau, n_e))
        r = minimize_scalar(lambda lb: -float(H.gbar_srf(math.exp(lb), A, n_i, tau, n_e)), bounds=(-10, 15),
                            method="bounded", options={"xatol": 1e-10})
        assert b == pytest.approx(math.exp(r.x), rel=1e-4)


def test_beta_opt_limits():
    # large occupation, cold line: beta ~ 2 tau n_i/A and G-bar ~ tau omega/(6 sqrt3 A T)
    f, T = 58e6, 2.0
    n = float(H.occupation(f, T))
    for A, tau in [(1.0, 1.0), (10.0, 1.0), (1.0, 0.5)]:
        b = float(H.beta_opt(A, n, tau))
        assert b == pytest.approx(2 * tau * n / A, rel=3 / (2 * tau * n / A) + 1e-3)
        G = float(H.gbar_srf(b, A, n, tau))
        # corrections O(A/(tau n)): 0.6% at A = 10
        assert G == pytest.approx(float(H.gbar_srf_asymptotic(f, T, A, tau)), rel=3 * A / (tau * n) + 1e-3)
    # no thermal noise: beta = 2
    assert float(H.beta_opt(1.0, 0.0)) == pytest.approx(2.0)


# ----------------------------------------------------------------------------- scan rate
@pytest.mark.parametrize("beta", [3.0, 120.0, 1400.0])
def test_scan_rate_closed_form_equals_psd_quadrature(beta):
    f_a, f1, V, B0, eta, Qi, A, T = 5e5, 58.3e6, 244.0, 0.107, 0.45, 4e11, 1.0, 2.0
    n_i = float(H.occupation(f1, T))
    for tau, n_e in [(1.0, 0.0), (0.7, 2.0)]:
        Af = float(H.noise_flat(A - 0.5, tau, n_e, 0.0))
        r1 = float(H.scan_rate(f_a, f1, V, B0, eta, Qi, Af, n_i, beta=beta, tau=tau, n_e=n_e))
        r2 = H.scan_rate_from_psd(f_a, f1, V, B0, eta, Qi, Af, n_i, beta=beta, tau=tau, n_e=n_e)
        assert r1 == pytest.approx(r2, rel=1e-6)


def test_mode_power_natural_units():
    """P_sig = g^2 eta^2 B0^2 rho V Q_1/(4 omega_1), evaluated independently in eV units."""
    g, f1, B0, V, eta, Q1, rho = 1e-15, 1e8, 0.1, 10.0, 0.45, 1e10, 0.3
    hbar_eVs, hbarc_eVm = 6.582119569e-16, 1.973269804e-7
    T_eV2 = 195.3536                                   # 1 T in natural HL units [eV^2]
    w1 = 2 * math.pi * f1 * hbar_eVs                   # eV
    rho_eV4 = rho * 1e9 * (hbarc_eVm * 1e2) ** 3       # eV^4
    V_eV = V / hbarc_eVm ** 3                          # eV^-3
    P_eV2 = (g * 1e-9) ** 2 * eta ** 2 * (B0 * T_eV2) ** 2 * rho_eV4 * V_eV * Q1 / (4 * w1)
    P_W = P_eV2 / hbar_eVs * 1.602176634e-19
    assert float(H.mode_power(g, f1, B0, V, eta, Q1, rho)) == pytest.approx(P_W, rel=2e-4)


def test_lumped_requirement_through_shared_conventions():
    """The integration and target conventions used here reproduce the package's lumped F_req (C++ core)."""
    from axionfem import results
    tgt = H.DEFAULT_TARGET
    for name, b in [("optimistic", dict(Q=2e7, T=0.010, eta=0.1)), ("conservative", dict(Q=1e6, T=0.030, eta=10.0))]:
        def inv(f):
            kap = np.array([circuit.kappa_a_SI(float(g), tgt.rho_GeV_cm3) for g in tgt.g_GeV(f)])
            n = H.occupation(f, b["T"])
            s = 2 * n + 1
            xi = 2 * b["eta"] ** 2 / (s + np.sqrt(s * s + 8 * b["eta"] ** 2))
            G = xi / (xi * xi + 2 * s * xi + b["eta"] ** 2) ** 1.5
            return kap ** 4 * G
        t_unit = H.scan_time(inv, tgt.f_min_Hz, tgt.f_max_Hz)
        F = 16 * tgt.Theta * circuit.HBAR ** 2 * t_unit / (math.pi * tgt.Q_axion * b["Q"] * tgt.live_time_s)
        assert F == pytest.approx(results.requirement(name)["F_req_SI"], rel=2e-3)


# ----------------------------------------------------------------------------- model A
def test_model_A_default_benchmarks():
    """The benchmarks on their default geometry (the computed cell PROJECTX_086): regression values, the same
    numbers as benchmarks/srf/results/model_a_geometry.json (computed_wall, weak statistic), and Q_i = G_signal/R_s."""
    import json
    from pathlib import Path
    mag = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/srf/results/model_a_geometry.json").read_text())
    w = H.PROJECTX_086
    o = H.design_A(H.SCENARIOS_A["optimistic"])
    c = H.design_A(H.SCENARIOS_A["conservative"])
    assert H.SCENARIOS_A["optimistic"].geometry is w and H.SCENARIOS_A["conservative"].geometry is w
    assert o["V_m3"] == pytest.approx(259.926, rel=1e-4)
    assert c["V_m3"] == pytest.approx(4772.20, rel=1e-4)
    for key, d in (("optimistic", o), ("conservative", c)):
        rec = mag["results"][key]["computed_wall"]["weak"]
        for k in ("V_m3", "f0_Hz", "P_RF_W", "A_cav_m2", "tuning_fraction"):
            assert d[k] == pytest.approx(rec[k], rel=1e-9), (key, k)
        assert d["cost_MCHF"]["total"] == pytest.approx(rec["cost_MCHF"]["total"], rel=1e-9)
    assert o["Q_i"] == pytest.approx(w.G_signal_ohm / 1.5e-9) and c["Q_i"] == pytest.approx(w.G_signal_ohm / 20e-9)
    assert o["f0_Hz"] / 1e6 == pytest.approx(57.142, rel=1e-4) and c["f0_Hz"] / 1e6 == pytest.approx(21.660, rel=1e-4)
    assert o["P_RF_W"] == pytest.approx(1114.82, rel=1e-4) and c["P_RF_W"] == pytest.approx(37241.6, rel=1e-4)
    assert 100 * o["tuning_fraction"] == pytest.approx(1.5385, abs=1e-4)
    assert 100 * c["tuning_fraction"] == pytest.approx(4.0586, abs=1e-4)
    assert o["cost_MCHF"]["total"] == pytest.approx(64.693, rel=1e-4)
    assert c["cost_MCHF"]["total"] == pytest.approx(335.562, rel=1e-4)


def test_model_A_volume_scaling_law():
    sc = H.SCENARIOS_A["optimistic"]
    V0 = H.required_volume(sc)
    from dataclasses import replace
    V2 = H.required_volume(replace(sc, A=2 * sc.A, T_K=2 * sc.T_K))       # (A T)^(3/8)
    assert V2 / V0 == pytest.approx(4 ** 0.375, rel=3e-3)
    V3 = H.required_volume(replace(sc, Bpk_T=2 * sc.Bpk_T))                # B0^(-3/2)
    assert V3 / V0 == pytest.approx(2 ** -1.5, rel=3e-3)


def test_geometry_from_modes():
    g = H.SRFGeometry.from_modes(dict(xi=7.7, eta=0.45, G_pump_ohm=520, G_signal_ohm=600, Bpk_over_B0=1.4,
                                      area_coeff=6.2, unrelated=1))
    assert float(g.V_m3(g.f0_Hz(250.0))) == pytest.approx(250.0)


FOM = dict(xi_pump=7.64, xi_signal=7.65, eta_10=0.449, G_pump_ohm=524.0, G_signal_ohm=604.0, Bpk_over_B0=1.46,
           B_equator_over_B0=1.40, A_over_V23=6.18, Epk_over_cB0=1.85, f_pump_Hz=1.4e9)


def test_geometry_from_srf_figures_of_merit():
    """from_modes maps the axionfem.srf.figures_of_merit keys; bpk picks the wall maximum (default, the peak wall
    field) or the equator value (diagnostic).  eta is eta_10 itself, not its square."""
    w, e = H.SRFGeometry.from_modes(FOM), H.SRFGeometry.from_modes(FOM, bpk="equator", name="x")
    assert (w.xi, w.eta, w.G_pump_ohm, w.G_signal_ohm, w.area_coeff, w.Epk_over_cB0) == (7.64, 0.449, 524.0, 604.0,
                                                                                          6.18, 1.85)
    assert w.Bpk_over_B0 == 1.46 and e.Bpk_over_B0 == 1.40 and e.name == "x"
    with pytest.raises(ValueError, match="bpk"):
        H.SRFGeometry.from_modes(FOM, bpk="peak")
    with pytest.raises(KeyError, match="A_over_V23"):
        H.SRFGeometry.from_modes({k: v for k, v in FOM.items() if k != "A_over_V23"})


GOOD = dict(xi=7.643, eta=0.449, G_pump_ohm=524.5, G_signal_ohm=604.5, Bpk_over_B0=1.464, area_coeff=6.182,
            Epk_over_cB0=1.846)


@pytest.mark.parametrize("field,value,match", [
    ("xi", 7.643 / (2 * math.pi), "ANGULAR"),                    # f0 V^(1/3)/c: missing 2 pi
    ("xi", 500.0, "outside"),
    ("eta", 1.2, r"\(0, 1\]"),
    ("eta", 0.0, r"\(0, 1\]"),
    ("eta", None, "finite"),                                     # figures_of_merit_3d without overlaps
    ("eta", float("nan"), "finite"),
    ("G_signal_ohm", 604.5 / 376.730313, "ohm"),                 # natural units
    ("G_pump_ohm", -524.5, "ohm"),
    ("Bpk_over_B0", 1 / 1.464, "B0/B_pk"),                      # inverted
    ("area_coeff", 3.0, "isoperimetric"),                        # below the sphere
    ("area_coeff", "6.2x", "finite"),
    ("Epk_over_cB0", -1.0, "Epk_over_cB0"),
])
def test_geometry_validation_rejects_wrong_inputs(field, value, match):
    """Each wrong-but-plausible input (xi without 2 pi, G in natural units, B0/B_pk inverted, an impossible area, a
    missing value) raises instead of returning a plausible-looking design."""
    with pytest.raises(ValueError, match=match):
        H.SRFGeometry(**{**GOOD, field: value})
    with pytest.raises(ValueError, match=match):                 # the same through from_modes (own keys)
        H.SRFGeometry.from_modes({**GOOD, field: value})


def test_geometry_validation_accepts_limits_and_eta_squared_is_a_mapping_question():
    H.SRFGeometry(**{**GOOD, "area_coeff": H.AREA_COEFF_MIN, "eta": 1.0, "Bpk_over_B0": 1.0, "Epk_over_cB0": None})
    assert H.AREA_COEFF_MIN == pytest.approx(4.8360, abs=1e-4)
    # eta_10^2 = 0.20 is itself a valid overlap, so no range check can catch it: from_modes must map eta_10 itself,
    # and the volume law makes the slip a factor 0.449^-1.5 = 3.3 in V
    g = H.SRFGeometry.from_modes(FOM)
    assert g.eta == FOM["eta_10"]
    sc = H.SCENARIOS_A["optimistic"]
    from dataclasses import replace
    V1 = H.required_volume(replace(sc, geometry=g))
    V2 = H.required_volume(replace(sc, geometry=replace(g, eta=g.eta ** 2)))
    assert V2 / V1 == pytest.approx(g.eta ** -1.5, rel=3e-3)


def test_geometry_bpk_for_own_key_dicts():
    """bpk is honored (or refused) for own-key dicts; E_pk = (E_pk/(c B0)) c B0."""
    own = dict(GOOD, B_equator_over_B0=1.404)
    assert H.SRFGeometry.from_modes(own).Bpk_over_B0 == 1.464
    assert H.SRFGeometry.from_modes(own, bpk="equator").Bpk_over_B0 == 1.404
    with pytest.raises(ValueError, match="B_equator_over_B0"):
        H.SRFGeometry.from_modes(GOOD, bpk="equator")
    with pytest.raises(ValueError, match="bpk"):
        H.SRFGeometry.from_modes(GOOD, bpk="peak")
    with pytest.raises(TypeError):
        H.SRFGeometry(**GOOD, Epk_over_B0=1.9)          # one name per field
    # E_pk = (E_pk/(c B0)) c B0: 56.6 / 34.0 MV/m for the computed cell (smooth wall) at the two model-A presets
    w = H.PROJECTX_086
    for key, Ek in (("optimistic", 56.6e6), ("conservative", 34.0e6)):
        from dataclasses import replace
        sc = replace(H.SCENARIOS_A[key], geometry=w)
        assert w.Epk_V_per_m(sc.B0_T) == pytest.approx(Ek, rel=2e-3)


def test_computed_geometry_presets():
    """The cached record (benchmarks/srf/srf_geometry_record.py; docs/srf/README.md validation table).  The default
    is PROJECTX_086 = PROJECTX_086_WALL (B_pk = largest wall |B|); the equator value is only the labelled diagnostic
    PROJECTX_086_EQUATOR_DIAGNOSTIC."""
    w, q = H.PROJECTX_086, H.PROJECTX_086_EQUATOR_DIAGNOSTIC
    assert H.PROJECTX_086_WALL is w and H.GEOMETRIES["computed_wall"] is w and H.GEOMETRIES["computed"] is w
    assert H.GEOMETRIES["equator_diagnostic"] is q and "DIAGNOSTIC" in q.name
    assert list(H.GEOMETRIES) == ["computed_wall", "equator_diagnostic"]
    assert w.xi == pytest.approx(7.643, abs=1e-3) and w.eta == pytest.approx(0.4492, abs=1e-4)
    assert (w.G_pump_ohm, w.G_signal_ohm) == pytest.approx((524.47, 604.48), abs=0.05)
    assert w.Bpk_over_B0 == pytest.approx(1.4640, abs=2e-4) and q.Bpk_over_B0 == pytest.approx(1.4039, abs=2e-4)
    assert w.area_coeff == pytest.approx(6.182, abs=1e-3) and w.Epk_over_cB0 == pytest.approx(1.843, abs=1e-3)
    assert H.SCENARIOS_A["optimistic"].geometry is w and H.ScenarioA("x", 1e-9, 0.1, 1.0, 2.0).geometry is w
    rec = H._load_record(str(H.PROJECTX_RECORD))
    assert max(abs(v) for v in rec["rel_diff_check_vs_ref"].values()) < 5e-3
    with pytest.raises(AttributeError):
        H.PROJECTX_087


def test_geometry_record_fingerprint_is_checked(tmp_path):
    """The record carries its inputs and a fingerprint, checked on load; a record for another cell or a coarser mesh
    is refused; a missing record raises FileNotFoundError at preset access (not a None)."""
    import json
    rec = json.loads(H.PROJECTX_RECORD.read_text())
    inp, fp = rec["inputs"], rec["fingerprint"]
    assert inp["axial_factor"] == 0.86 and inp["tube_length_m"] == 0.100 and inp["tube_radius_m"] == 0.050
    assert (inp["pump"], inp["signal"], inp["order"]) == ("TM020", "TE011", 3) and inp["n"] >= 80
    assert fp["srf_py_sha256"] == H.srf_source_sha256(), "srf.py changed: regenerate the record"
    assert not fp["dirty"], f"record written from a dirty tree ({fp['git_describe']})"
    H.check_record(rec)
    for mut, msg in ((lambda r: r["inputs"].update(axial_factor=0.85), "axial_factor"),
                     (lambda r: r["inputs"]["cell"].update(half_length=0.1), "half_length"),
                     (lambda r: r["inputs"].update(tube_length_m=0.0), "tube_length"),
                     (lambda r: r["figures_of_merit"]["mesh"].update(h_over_Req=1 / 20), "R_eq/h = 20"),
                     (lambda r: r["figures_of_merit"]["mesh"].update(order=2), "order 2"),
                     (lambda r: r.pop("inputs"), "fingerprint")):
        bad = json.loads(json.dumps(rec))
        mut(bad)
        with pytest.raises(ValueError, match=msg):
            H.check_record(bad)
        p = tmp_path / f"bad_{msg.replace(' ', '_').replace('/', '')}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            H.projectx_geometry("wall", record=p)
    stale = json.loads(json.dumps(rec))
    stale["fingerprint"]["srf_py_sha256"] = "0" * 64
    with pytest.warns(UserWarning, match="different version of axionfem.srf"):
        H.check_record(stale)
    with pytest.raises(FileNotFoundError, match="srf_geometry_record.py"):
        H.projectx_geometry("wall", record=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="bpk"):
        H.projectx_geometry("peak")


def test_geometry_record_script_refuses_package_overwrite():
    """--out has no default and the package record is written only with --update-package-data and the preset
    definition (the argument checks fail before any solve), whether the package is imported from the tree or from an
    installed copy."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks/srf/srf_geometry_record.py"
    for args, msg in (([], "--out"), (["--update-package-data", "--n", "20"], "preset definition"),
                      (["--update-package-data", "--axial-factor", "0.85"], "preset definition"),
                      (["--out", str(H.PROJECTX_RECORD)], "update-package-data"),      # the imported package's copy
                      (["--out", str(root / "python/axionfem/data/srf_projectx086_geometry.json")],
                       "update-package-data")):
        r = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=120)
        assert r.returncode == 2 and msg in r.stderr, (args, r.stderr)


def test_derived_results_match_the_record():
    """The second-level caches (benchmarks/srf/model_a_geometry.py, srf_vibration_mixing.py report) were computed from
    the current geometry record and presets."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "benchmarks/srf/results"
    rec = json.loads(H.PROJECTX_RECORD.read_text())
    mag = json.loads((root / "model_a_geometry.json").read_text())
    assert mag["record"]["fingerprint"] == rec["fingerprint"] and mag["record"]["generated"] == rec["generated"]
    assert set(mag["geometries"]) == {"computed_wall", "equator_diagnostic"}
    for g, fields in mag["geometries"].items():
        live = H.GEOMETRIES[g]
        assert fields == {f: getattr(live, f) for f in H.SRFGeometry.__dataclass_fields__}, g
    vib = json.loads((root / "srf_vibration.json").read_text())["report"]
    assert vib["V_cell_m3"] == rec["figures_of_merit"]["V_m3"] and vib["record_fingerprint"] == rec["fingerprint"]
    for key, per in mag["results"].items():
        for geo, stats in per.items():
            for stat, r in stats.items():
                assert vib["rows"][f"{key}/{geo}/{stat}"]["V_m3"] == r["V_m3"]


def test_model_A_with_computed_geometry():
    """benchmarks/srf/results/model_a_geometry.json: the volumes with the wall B_pk (1.464, the default) and with the
    equator diagnostic (1.404)."""
    from dataclasses import replace
    for key, (Vw, Ve) in {"optimistic": (259.9, 244.1), "conservative": (4772.3, 4480.6)}.items():
        sc = H.SCENARIOS_A[key]
        assert H.required_volume(replace(sc, geometry=H.PROJECTX_086)) == pytest.approx(Vw, rel=1e-3)
        assert H.required_volume(replace(sc, geometry=H.PROJECTX_086_EQUATOR_DIAGNOSTIC)) == pytest.approx(Ve, rel=1e-3)


@pytest.mark.parametrize("key", ["optimistic", "conservative"])
def test_model_A_wall_vs_equator_is_the_B0_law(key):
    """Same cell, two B_pk values: V_wall/V_eq = (1.4640/1.4039)^(3/2) = 1.0650 (only B0 differs; the model gives
    1.06501 against the law's 1.06495).  Against a second cell (ROUNDED) the ratio factorizes as the volume law
    V ~ B0^-3/2 xi^3/4 eta^-3/2 Q_i^-3/8: +6.9 % from B0 alone ((1.464/1.4)^(3/2)), -0.6 % from xi, eta_10 and G."""
    from dataclasses import replace
    w, q, p = H.PROJECTX_086, H.PROJECTX_086_EQUATOR_DIAGNOSTIC, ROUNDED
    sc = H.SCENARIOS_A[key]
    Vw, Vq, Vp = (H.required_volume(replace(sc, geometry=g)) for g in (w, q, p))
    assert Vw / Vq == pytest.approx((w.Bpk_over_B0 / q.Bpk_over_B0) ** 1.5, rel=1e-3)
    law = ((w.Bpk_over_B0 / p.Bpk_over_B0) ** 1.5 * (w.xi / p.xi) ** 0.75 * (w.eta / p.eta) ** -1.5
           * (w.G_signal_ohm / p.G_signal_ohm) ** -0.375)
    assert Vw / Vp == pytest.approx(law, rel=1e-3)
    assert (w.Bpk_over_B0 / p.Bpk_over_B0) ** 1.5 == pytest.approx(1.0694, abs=2e-4)   # B0 alone: +6.9 %
    assert law / (w.Bpk_over_B0 / p.Bpk_over_B0) ** 1.5 == pytest.approx(0.9943, abs=3e-4)  # xi, eta, G: -0.6 %


def test_model_A_adiabatic_schedule():
    """The weak-signal sizing of the optimistic preset (optimal coupling beta*) exceeds the adiabatic condition at the
    top of the default band; raising beta where it binds restores the condition at a small volume increase (regression
    values)."""
    sc = H.SCENARIOS_A["optimistic"]
    d = H.design_A(sc)
    assert d["adiabatic_ratio_ends"][1] > 10 and d["adiabatic_ratio_ends"][0] < 0.1
    V = H.required_volume(sc, adiabatic_eps=0.1)
    assert V / d["V_m3"] == pytest.approx(1.0035, abs=1e-3)
    f_a = np.geomspace(120.9e3, 1e6, 50)
    rate, beta = H._rate_A(sc, V, H.DEFAULT_TARGET, "shifted", 0.1, return_beta=True)(f_a)
    f1 = float(sc.geometry.f0_Hz(V)) + f_a
    Qi = sc.geometry.G_signal_ohm / sc.Rs_ohm
    assert np.all(rate <= H.tracking_rate(beta, f_a, f1, Qi, 0.1) * (1 + 1e-9))


def test_model_A_wall_temperature():
    """The wall temperature sets n_i and the refrigerator factor (x3.3 at 2 K, x1 at 4.5 K; other temperatures are
    refused): the presets at another wall temperature, against regression values of the volume and the cost metric."""
    from dataclasses import replace
    c = H.SCENARIOS_A["conservative"]
    assert c.T_K == 4.5 and H.SCENARIOS_A["optimistic"].T_K == 2.0
    d2 = H.design_A(replace(c, T_K=2.0))
    assert d2["V_m3"] == pytest.approx(3520.4, rel=1e-3) and d2["f0_Hz"] == pytest.approx(23.97e6, rel=1e-3)
    assert d2["cost_MCHF"]["total"] == pytest.approx(377.6, rel=2e-3) and d2["cost_MCHF"]["factor"] == 3.3
    assert H.design_A(c)["cost_MCHF"]["factor"] == 1.0
    assert H.required_volume(replace(H.SCENARIOS_A["optimistic"], T_K=1.8)) == pytest.approx(249.85, rel=1e-3)
    with pytest.raises(ValueError):
        H.refrigerator_factor(1.8)


def test_asimov_density_limits():
    r = np.array([1e-6, 1e-4, 9.99e-4, 1.001e-3, 0.1, 2.2, 50.0])
    F = H.asimov_density(r)
    exact = 2 * (np.log1p(r) - r / (1 + r))
    assert np.allclose(F[3:], exact[3:], rtol=1e-14)
    assert np.allclose(F[:3] / r[:3] ** 2, 1 - 4 * r[:3] / 3, rtol=2e-6)
    assert F[2] == pytest.approx(exact[2], rel=1e-8)          # series meets the closed form at the switch
    assert np.all(F <= r * r)


def test_fisher_integral_weak_limit_and_peak_snr():
    """exact=False on the SHM grids = closed-form weak-signal rate x Q_SHM/6.8e5; peak per-bin SNR of the weak-signal
    optimum at the band ends."""
    sc = H.SCENARIOS_A["optimistic"]
    V = 259.9255
    qa = H._h.Q_a_shm()
    for fa in (120.9e3, 1e6):
        rx = H._receiver_A(sc, V, fa, H.DEFAULT_TARGET)
        b = float(H.beta_opt(sc.A, rx["n_i"]))
        c, D, w = H._bin_quantities(sc, rx, b)
        J = H.fisher_integral(c, sc.A, D, w, fa, exact=False) / H.DEFAULT_TARGET.Theta
        r0 = float(H._rate_A(sc, V, H.DEFAULT_TARGET, "shifted")(np.array([fa]))[0])
        assert J / r0 == pytest.approx(qa / 6.8e5, rel=1e-6)
    snr = H.peak_bin_snr(sc, V, [120.9e3, 1e6])
    assert snr[0] == pytest.approx(0.2672, rel=5e-3) and snr[1] == pytest.approx(2.1764, rel=5e-3)
    # exact statistic: lower rate, larger optimal beta at high SNR
    re, be = H.rate_exact_A(sc, V, 1e6)
    r0 = float(H._rate_A(sc, V, H.DEFAULT_TARGET, "shifted")(np.array([1e6]))[0])
    assert re / r0 == pytest.approx(0.3899, rel=2e-3)
    assert be > 2 * float(H.beta_opt(sc.A, H._receiver_A(sc, V, 1e6, H.DEFAULT_TARGET)["n_i"]))


def test_adiabatic_threshold_root():
    d = H.design_A(H.SCENARIOS_A["optimistic"])
    assert d["adiabatic_violated_above_Hz"] == pytest.approx(3.3606e5, rel=1e-3)


# ----------------------------------------------------------------------------- statistics and quadrature checks
def test_occupation_matches_haloscope_n_thermal():
    from axionfem import haloscope
    f = np.geomspace(1e3, 1e12, 40)
    for T in (0.0, 0.01, 2.0, 4.5, 300.0):
        assert np.allclose(H.occupation(f, T), haloscope.n_thermal(f, T), rtol=1e-12, atol=0)


def test_hot_port_bath_quadratures():
    """n_e > n_i (D < 0): the response L/(A + D L) is narrower than the cavity; both quadratures must still match the
    closed form (a substitution scaled with the cavity width instead would be 2.3% off here, a factor 2.5 at
    n_e = 1e5)."""
    qa = H._h.Q_a_shm()
    for n_e in (6250.0, 1e5):
        A, n_i, tau, fa = 0.5 + 0.5 + n_e, 0.008, 1.0, 5e5
        b = float(H.beta_opt(A, n_i, tau, n_e))
        D = float(H.noise_resonant(b, n_i, tau, n_e))
        width, c = 0.3, 1e3
        J = H.fisher_integral(c, A, D, width, fa, exact=False)
        closed = c ** 2 * math.pi * width / (4 * math.sqrt(A) * (A + D) ** 1.5) * qa / fa
        assert J == pytest.approx(closed, rel=1e-6)
        args = (fa, 1e9, 1.0, 0.1, 0.45, 1e10, A, n_i)
        assert H.scan_rate_from_psd(*args, beta=b, tau=tau, n_e=n_e) == pytest.approx(
            float(H.scan_rate(*args, beta=b, tau=tau, n_e=n_e)), rel=1e-6)


def test_exact_path_rejects_other_line_widths():
    from dataclasses import replace
    with pytest.raises(ValueError, match="SHM"):
        H.rate_exact_A(H.SCENARIOS_A["optimistic"], 259.9, 5e5, target=replace(H.DEFAULT_TARGET, Q_axion=1e6))


def test_bracket_log_widens():
    for root in (-15.0, -2.0, 0.1, 3.0, 15.0):
        lo, hi = H._bracket_log(lambda x: root - x, 0.0)
        assert lo <= root <= hi
    with pytest.raises(RuntimeError):
        H._bracket_log(lambda x: 1.0, 0.0, max_iter=5)


def test_required_volume_exact_short_live_time_and_band_ends():
    """A 0.03-yr target needs V far outside a fixed [0.7, 1.6] x V_weak bracket (coarse grids; the point is the
    widening bracket and the band-end diagnostics, not convergence)."""
    from dataclasses import replace
    tgt = replace(H.DEFAULT_TARGET, live_time_yr=0.03)
    sc = H.SCENARIOS_A["optimistic"]
    r = H.required_volume_exact(sc, tgt, n_f=6, n_x=121, n_v=80)
    Vw = H.required_volume(sc, tgt)
    assert r["V_m3"] > 1.6 * Vw
    assert r["f_ends_Hz"] == [tgt.f_min_Hz, tgt.f_max_Hz]


def test_endpoint_guard_is_negligible():
    for key in H.SCENARIOS_A:
        sc = H.SCENARIOS_A[key]
        V0, Vg = H.required_volume(sc), H.required_volume(sc, guard_Hz=20.0)
        assert 0 < Vg / V0 - 1 < 1e-3


def test_bandwidth_estimate_includes_tau():
    from dataclasses import replace
    for tau in (1.0, 0.5):
        d = H.design_A(replace(H.SCENARIOS_A["optimistic"], tau=tau))
        lo, hi = d["loaded_bw_Hz_range"]
        assert lo * 0.99 < d["bw_approx_tauT_over_piAQi_Hz"] < hi * 1.01


def test_exact_optimum_still_needs_the_schedule_at_1MHz():
    """At the exact-statistic sizing of the optimistic preset (model_a_geometry.json; beta re-optimised) the statistical
    rate at the top of the default band exceeds the adiabatic tracking limit, so the schedule binds there (regression
    values)."""
    sc = H.SCENARIOS_A["optimistic"]
    V, fa = 298.041, 1e6
    r, b = H.rate_exact_A(sc, V, fa)
    rx = H._receiver_A(sc, V, fa, H.DEFAULT_TARGET)
    tr = float(H.tracking_rate(b, fa, rx["f1"], rx["Qi"], 0.1))
    assert r / tr == pytest.approx(1.981, rel=0.02)
    ra, ba = H.rate_exact_A(sc, V, fa, adiabatic_eps=0.1)
    assert ba > b and ra <= float(H.tracking_rate(ba, fa, rx["f1"], rx["Qi"], 0.1)) * (1 + 1e-9)
