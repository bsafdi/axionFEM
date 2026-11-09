"""Axisymmetric steady conduction against closed-form references.

- cylinder with uniform heat generation and a fixed outer temperature: T = T0 + q (R^2 - r^2)/(4 k),
  which order-2 elements reproduce exactly (it is quadratic in r);
- the same with a linear k(T) table, against the analytic Kirchhoff solution
  theta = int k dT = q (R^2 - r^2)/4, inverted for T;
- a fixed temperature on a zero-thickness curve, with the generation + annulus log profile;
- `W` (total) and `W_m3` (density) heat loads agree;
- the heat balance (generated vs. outflow through the fixed temperatures).
"""
import math

import numpy as np
import pytest

import axionfem as af

T0, Q, RP, RS, HT = 4.0, 5000.0, 0.4, 0.5, 1.0
VAC = {"kind": "vacuum"}   # a coil material that takes no part in the thermal analysis


def _two_region(thermal_mat, loads=None, fixed=None, size=0.03, order=2, rs=RS):
    loads = loads if loads is not None else [{"region": "core", "W_m3": Q}]
    fixed = fixed if fixed is not None else [{"region": "sink", "T_K": T0}]
    return af.Design({
        "schema": "axionfem.design/1", "name": "th",
        "materials": {"m": thermal_mat, "cm": VAC},
        "regions": [{"name": "core", "shape": {"rect": {"r": [0.0, RP], "z": [-HT / 2, HT / 2]}},
                     "material": "m", "mesh_size": size},
                    {"name": "sink", "shape": {"rect": {"r": [RP, rs], "z": [-HT / 2, HT / 2]}},
                     "material": "m", "mesh_size": size},
                    {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}}, "material": "cm"}],
        "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
        "thermal": {"fixed": fixed, "loads": loads},
        "mesh": {"order": order, "size": 0.3, "domain_radius": 12.0}})


def test_uniform_heat_generation(tmp_path):
    k = 12.0
    res = _two_region({"kind": "linear", "thermal": {"k_W_mK": k}}).solve(tmp_path / "uni", verbose=0)
    r = np.linspace(0.0, RP - 1e-9, 9)
    s = res.sample_mech(r, np.zeros_like(r))
    exact = T0 + Q * (RP ** 2 - r ** 2) / (4 * k)
    # the exact solution is quadratic in r, so order-2 elements are exact to round-off
    assert np.max(np.abs(s["T_K"] - exact)) < 1e-10
    ck = res.thermal["checks"]
    assert ck["generated_W"] == pytest.approx(Q * math.pi * RP ** 2 * HT, rel=1e-10)
    assert ck["outflow_W"] == pytest.approx(ck["generated_W"], rel=1e-10)
    assert ck["balance_relative"] < 1e-10
    reg = res.thermal["regions"]
    assert reg["core"]["T_max_K"] == pytest.approx(T0 + Q * RP ** 2 / (4 * k), rel=1e-10)
    assert reg["core"]["k_W_mK"] == k
    assert reg["sink"]["T_max_K"] == pytest.approx(T0, abs=1e-10)
    assert res.thermal["T_max_K"] == pytest.approx(T0 + Q * RP ** 2 / (4 * k), rel=1e-10)
    assert not res.data["warnings"]


def test_total_power_load_matches_density(tmp_path):
    k = 12.0
    V = 2 * math.pi * (RP ** 2 / 2) * HT          # 2 pi int_0^RP r dr * h
    res = _two_region({"kind": "linear", "thermal": {"k_W_mK": k}},
                      loads=[{"region": "core", "W": Q * V}]).solve(tmp_path / "tot", verbose=0)
    assert res.thermal["loads"][0]["W_m3"] == pytest.approx(Q, rel=1e-10)
    assert res.thermal["loads"][0]["volume_m3"] == pytest.approx(V, rel=1e-10)
    assert res.thermal["regions"]["core"]["T_max_K"] == pytest.approx(T0 + Q * RP ** 2 / (4 * k), rel=1e-9)


def test_temperature_dependent_k_against_kirchhoff(tmp_path):
    """k(T) = k0 (1 + c (T - T0)) as a two-point table: the Kirchhoff potential
    theta = k0[(T-T0) + c (T-T0)^2/2] solves -laplacian theta = q, so
    T = T0 + (sqrt(1 + 2 c theta / k0) - 1)/c with theta = q (RP^2 - r^2)/4."""
    k0, c = 10.0, 0.08
    table = [[T0, k0], [T0 + 30.0, k0 * (1 + c * 30.0)]]
    res = _two_region({"kind": "linear", "thermal": {"k_W_mK": table}},
                      size=0.02).solve(tmp_path / "kT", verbose=0)
    r = np.linspace(0.0, RP - 1e-9, 9)
    s = res.sample_mech(r, np.zeros_like(r))
    theta = Q * (RP ** 2 - r ** 2) / 4.0
    exact = T0 + (np.sqrt(1 + 2 * c * theta / k0) - 1) / c
    rise = exact[0] - T0
    assert np.max(np.abs(s["T_K"] - exact)) < 1e-4 * rise      # discretization error only
    sol = res.thermal["solver"]
    assert sol["nonlinear"] and sol["converged"]
    assert 1 < sol["picard_iterations"] < 60
    assert res.thermal["checks"]["balance_relative"] < 1e-10
    kreg = res.thermal["regions"]["core"]["k_W_mK"]
    assert kreg["at_T_min"] == pytest.approx(k0, rel=1e-6)
    assert kreg["at_T_max"] == pytest.approx(k0 * (1 + c * rise), rel=1e-4)
    assert not res.data["warnings"]


def test_fixed_temperature_on_a_curve(tmp_path):
    """A conducting sheet at r = RS is the heat sink: generation inside RP plus a log profile in
    the annulus RP < r < RS."""
    k = 12.0
    d = _two_region({"kind": "linear", "thermal": {"k_W_mK": k}},
                    fixed=[{"curve": "strap", "T_K": T0}])
    d.doc["curves"] = [{"name": "strap", "path": {"segment": [[RS, -HT / 2], [RS, HT / 2]]},
                        "material": "copper"}]
    res = d.solve(tmp_path / "curve", verbose=0)
    r = np.array([0.0, 0.2, 0.35, 0.45])
    s = res.sample_mech(r, np.zeros_like(r))
    P_per_len = Q * math.pi * RP ** 2
    T_rp = T0 + P_per_len / (2 * math.pi * k) * math.log(RS / RP)
    exact = np.where(r <= RP, T_rp + Q * (RP ** 2 - np.minimum(r, RP) ** 2) / (4 * k),
                     T0 + P_per_len / (2 * math.pi * k) * np.log(RS / np.maximum(r, 1e-30)))
    assert np.max(np.abs(s["T_K"] - exact)) < 2e-3 * (exact[0] - T0)
    assert res.thermal["checks"]["balance_relative"] < 1e-10
    assert res.thermal["fixed"] == [{"curve": "strap", "T_K": T0}]


def test_builtin_thermal_defaults(tmp_path):
    res = _two_region({"kind": "linear", "thermal": {"k_W_mK": 5.0}}).solve(tmp_path / "def", verbose=0)
    assert res.thermal["regions"]["core"]["k_W_mK"] == 5.0
    # built-ins
    doc = {"schema": "axionfem.design/1", "name": "bi",
           # the coil wraps the copper core, so the two conduct as one bonded assembly
           "regions": [{"name": "cu", "shape": {"rect": {"r": [0.0, 0.5], "z": [-0.2, 0.2]}},
                        "material": "copper", "mesh_size": 0.05},
                       {"name": "coil", "shape": {"rect": {"r": [0.5, 0.52], "z": [-0.2, 0.2]}},
                        "material": "winding", "mesh_size": 0.05}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
           "thermal": {"fixed": [{"region": "coil", "T_K": 4.0}], "loads": [{"region": "cu", "W": 1.0}]},
           "mesh": {"order": 2, "size": 0.2, "domain_radius": 4.0}}
    res = af.Design(doc).solve(tmp_path / "bi", verbose=0)
    assert res.thermal["regions"]["cu"]["k_W_mK"] == pytest.approx(600.0)     # OFHC near 4 K
    assert res.thermal["regions"]["coil"]["k_W_mK"] == pytest.approx(1.0)     # impregnated coil
    assert res.thermal["checks"]["generated_W"] == pytest.approx(1.0, rel=1e-9)


def test_isolated_conducting_part_is_an_error(tmp_path):
    """A conducting part with no path to a heat sink has an undetermined temperature level."""
    d = _two_region({"kind": "linear", "thermal": {"k_W_mK": 12.0}})
    d.doc["regions"].append({"name": "island", "shape": {"rect": {"r": [1.0, 1.1], "z": [-0.1, 0.1]}},
                             "material": "m", "mesh_size": 0.05})
    with pytest.raises(Exception, match="no prescribed temperature"):
        d.solve(tmp_path / "island", verbose=0)


def test_thermal_input_errors(tmp_path):
    d = _two_region({"kind": "linear", "thermal": {"k_W_mK": 12.0}})
    bad = d.copy()
    bad.doc["thermal"]["loads"] = [{"region": "core", "W": 1.0, "W_m3": 1.0}]
    with pytest.raises(Exception, match="exactly one"):
        bad.validate()
    bad = d.copy()
    bad.doc["thermal"]["fixed"] = [{"region": "core", "curve": "x", "T_K": 4.0}]
    with pytest.raises(Exception, match="exactly one"):
        bad.validate()
    bad = d.copy()
    bad.doc["thermal"]["loads"] = [{"region": "coil", "W_m3": 1.0}]
    with pytest.raises(Exception, match="not part of the thermal domain"):
        bad.solve(tmp_path / "bad", verbose=0)
    bad = d.copy()
    bad.doc["materials"]["m"]["thermal"] = {"k_W_mK": [[10.0, 1.0], [5.0, 2.0]]}
    with pytest.raises(Exception, match="strictly increasing"):
        bad.validate()


# ============================================================ heat balance with several sinks; conflicting temperatures

def _two_sinks(T_mid=None):
    k = 10.0
    fixed = [{"region": "cold", "T_K": 4.0}, {"region": "warm", "T_K": 300.0}]
    if T_mid is not None:
        fixed.insert(1, {"region": "mid", "T_K": T_mid})
    return {"schema": "axionfem.design/1", "name": "two_sinks",
            "materials": {"m": {"kind": "vacuum", "thermal": {"k_W_mK": k}}, "cv": {"kind": "vacuum"}},
            "regions": [{"name": "cold", "shape": {"rect": {"r": [0.3, 0.4], "z": [-0.5, 0.5]}},
                         "material": "m", "mesh_size": 0.02},
                        {"name": "mid", "shape": {"rect": {"r": [0.4, 0.5], "z": [-0.5, 0.5]}},
                         "material": "m", "mesh_size": 0.02},
                        {"name": "warm", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.5, 0.5]}},
                         "material": "m", "mesh_size": 0.02},
                        {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                         "material": "cv"}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
            "thermal": {"fixed": fixed},
            "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}


def test_two_sink_heat_balance(tmp_path):
    """`outflow_W` is the NET flow over all prescribed-temperature DOFs, so it cancels for a part held between two
    sinks (a 4 K / 300 K cylinder passing 83.3 kW has ~0 W of it). heat_removed_W and heat_added_W report the
    cold-stage heat load, and balance_relative stays meaningful (not ~1e290)."""
    r = af.Design(_two_sinks()).solve(tmp_path / "two_sinks", verbose=0)
    c = r.thermal["checks"]
    exact = 2 * math.pi * 10.0 * 1.0 * (300.0 - 4.0) / math.log(0.5 / 0.4)
    assert c["heat_removed_W"] == pytest.approx(exact, rel=1e-6)
    assert c["heat_added_W"] == pytest.approx(exact, rel=1e-6)
    assert abs(c["outflow_W"]) < 1e-6 * exact        # the net really is zero
    assert c["balance_relative"] < 1e-10             # and the balance is now meaningful, not 1e290
    # the temperature profile is the exact 1D log solution
    rr = np.linspace(0.405, 0.495, 7)
    s = r.sample_mech(rr, np.zeros_like(rr))
    exact_T = 4.0 + (300.0 - 4.0) * np.log(rr / 0.4) / math.log(0.5 / 0.4)
    assert np.max(np.abs(np.asarray(s["T_K"]) - exact_T)) < 1e-5 * (300.0 - 4.0)


def test_uniform_generation_balance_is_unchanged(tmp_path):
    """The one-sink case must keep reporting the generated power as the removed heat."""
    r = _two_region({"kind": "linear", "thermal": {"k_W_mK": 12.0}}).solve(tmp_path / "gen", verbose=0)
    c = r.thermal["checks"]
    assert c["heat_removed_W"] == pytest.approx(c["generated_W"], rel=1e-10)
    assert c["heat_added_W"] < 1e-6 * c["generated_W"]
    assert c["outflow_W"] == pytest.approx(c["generated_W"], rel=1e-10)


def test_conflicting_prescribed_temperatures_are_rejected(tmp_path):
    """Two touching fixed-temperature regions with different temperatures are refused (not last-writer-wins)."""
    with pytest.raises(Exception, match="conflicting prescribed temperatures"):
        af.Design(_two_sinks(T_mid=77.0)).solve(tmp_path / "conflict", verbose=0)
    # the same temperature on both is fine (no conflict)
    doc = _two_sinks()
    doc["thermal"]["fixed"] = [{"region": "cold", "T_K": 4.0}, {"region": "mid", "T_K": 4.0}]
    r = af.Design(doc).solve(tmp_path / "same", verbose=0)
    assert r.thermal["T_min_K"] == pytest.approx(4.0, abs=1e-9)
