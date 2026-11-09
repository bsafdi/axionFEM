"""Surface losses of the 3D full-wave readout (docs/physics/fullwave_surface_losses.md; FwSurfaceLosses).

The design is the sealed coax of benchmarks/fullwave_surface_losses.py: a radial port slab between two shorted stubs,
whose field is the exact TEM line at every frequency (closed-form Gamma, Z, L_eff and Q).

Default suite (order 1, coarse meshes, in this process):
- the coax: the keys and the formula chain (R_loss = R_s Gamma, Q_loss, the tuned Q from the row's exact dZ/domega,
  T/Q), the perturbation warning (a resistive top plate at 1 MHz), the MQS limit on the same mesh (Gamma at 1 MHz
  against pickup.losses, after the closed form's own (kL)^2 change), L_eff = 4 W_m, the frequency dependence against
  the closed form, H_n = 0 on the conductor faces, and the Python columns;
- the two-port frame: the multiport Gamma and R_loss matrices;
- the solenoid template with a magnetoresistance model (skin and a measured table): the weight against pickup.losses',
  and Q_total withheld where Re Z is unresolved;
- the TEM line with PEC plates: no loss faces, reported; the R_loss interpolation guard; the reach's exact L_eff.
Slow (AXF_SLOW): order 2 against the closed form up to 1.8 % below the TEM_1 pole and above it (the tuned Q), the
axisymmetric engine's twin, 1 against 3 ranks, and an adaptive sweep's rows.
"""
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import axionfem as af
from axionfem import circuit, scan
from axionfem.convert import to_3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import fullwave_surface_losses as FSL  # noqa: E402  (the coax design and its TEM closed form)

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1")
# every test but the four pure-Python ones solves a 3D design
needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build; the wheels have none)")
SURFACES = ("inner", "outer", "top", "bottom")
RANKS = 4 if os.environ.get("AXF_MPI") else 1


def _coax3d(frequencies, mesh):
    return to_3d(af.Design(FSL.coax_doc(frequencies, FSL.COAX_2D)), mesh=mesh)


# a resistive top plate (constant R_s = 2 ohm): R_loss/|X| = 0.14 at 1 MHz, above the perturbation flag
LOSSY_TOP = {"kind": "conductor", "rf": {"model": "power", "Rs_ohm": 2.0, "f_ref": 1e6, "exponent": 0, "T_K": 0.03}}


@pytest.fixture(scope="module")
def coax_o1(tmp_path_factory):
    wd = tmp_path_factory.mktemp("coax_o1")
    d = _coax3d([1e6, 1e8], {"order": 1, "size": 0.3, "surface_size": 0.05})   # 137k DOFs, about 40 s
    d.doc.setdefault("materials", {})["lossy_top"] = LOSSY_TOP
    for region in d.doc["regions"]:
        if region["name"] == "top":
            region["material"] = "lossy_top"
    return d.solve(wd, verbose=0).data


def _Rs(surface, f):
    if surface == "top":
        return LOSSY_TOP["rf"]["Rs_ohm"]
    return math.sqrt(math.pi * f * FSL.MU0 * FSL.RHO_CU)   # copper_classical: the classical skin effect


@needs_3d
def test_keys_and_formula_chain(coax_o1):
    sweep = coax_o1["fullwave"]["sweep"]
    summary = coax_o1["fullwave"]["surface_losses"]
    assert set(summary["surfaces"]) == set(SURFACES)
    assert summary["surfaces"]["inner"]["rf_model"] == "skin"
    assert summary["faces"] > 0 and summary["quadrature_points"] == summary["faces"] * 6   # degree 2p + 2 = 4
    assert summary["surfaces"]["top"]["rf_model"] == "power"
    for row in sweep:
        f, w = row["f_Hz"], 2 * math.pi * row["f_Hz"]
        surfaces = row["surface_losses"]["surfaces"]
        assert set(row["surface_Gamma"]) == set(SURFACES)
        R = 0.0
        for k in SURFACES:
            e = surfaces[k]
            assert e["Gamma"] == row["surface_Gamma"][k]
            assert sum(e["Gamma_by_field_region"].values()) == pytest.approx(e["Gamma"], rel=1e-12)
            assert e["R_s_ohm"] == pytest.approx(_Rs(k, f), rel=1e-12)
            assert e["R_eff_ohm"] == pytest.approx(_Rs(k, f) * e["Gamma"], rel=1e-12)
            assert e["Q"] == pytest.approx(w * row["L_H"] / e["R_eff_ohm"], rel=1e-12)
            R += e["R_eff_ohm"]
        assert row["R_loss_ohm"] == pytest.approx(R, rel=1e-12)
        assert row["Q_loss"] == pytest.approx(w * row["L_H"] / R, rel=1e-12)
        # the tuned pole: L_eff = (L - Im dZ/domega) / 2 from the row's exact derivative
        L_eff = 0.5 * (row["L_H"] - row["dZ_domega_H"]["im"][0][0])
        assert row["L_eff_H"] == pytest.approx(L_eff, rel=1e-12)
        assert row["Q_loss_tuned"] == pytest.approx(w * L_eff / R, rel=1e-12)
        assert row["T_over_Q_K"] == pytest.approx(0.03 / row["Q_loss"], rel=1e-12)
        assert row["R_loss_over_abs_X"] == pytest.approx(R / abs(row["Z_im_ohm"]), rel=1e-12)
        # perfect conductor faces: E_t = 0, so H_n = 0 to rounding
        assert row["surface_losses"]["H_normal_share"] < 1e-12
        assert row["port_solutions"][0]["R_loss_ohm"] == row["R_loss_ohm"]


@needs_3d
def test_perturbation_flag(coax_o1):
    """The resistive top plate makes R_loss/|X| = 0.14 at 1 MHz (flagged and warned about); at 100 MHz |X| is 140 times
    larger and the row is not flagged."""
    low, high = coax_o1["fullwave"]["sweep"]
    assert low["R_loss_over_abs_X"] > 0.1 > high["R_loss_over_abs_X"]
    perturbation = coax_o1["fullwave"]["surface_losses"]["perturbation"]
    assert perturbation["rows_above_flag"] == 1 and perturbation["at_Hz"] == 1e6
    assert any("R_loss/|X| exceeds 0.1 in 1 port rows" in w for w in coax_o1["warnings"])


@needs_3d
def test_mqs_limit_on_the_same_mesh(coax_o1):
    """At 1 MHz (k L = 0.02) Gamma / Gamma_MQS of the same mesh changes as the closed form's (about 1e-4)."""
    mqs = {k: v["Gamma"] for k, v in coax_o1["pickup"]["losses"]["surfaces"].items()}
    row = coax_o1["fullwave"]["sweep"][0]
    exact = FSL.tem_closed_form(row["f_Hz"])["Gamma"]
    static = FSL.tem_static()["Gamma"]
    for k in SURFACES:
        assert row["surface_Gamma"][k] / mqs[k] == pytest.approx(exact[k] / static[k], abs=2e-5), k
    # and the tuned Q equals Q_loss to the same order, as both tend to 2 pi f L_p / R_eff
    assert row["Q_loss_tuned"] / row["Q_loss"] == pytest.approx(1.0, abs=1e-3)


@needs_3d
def test_tuned_inductance_is_four_magnetic_energies(coax_o1):
    """Closed and lossless: Foster's theorem makes L_eff = (X / omega + dX/domega) / 2 = 4 W_m, so the tuned Q is
    2 omega W_m / P_loss. For the discrete system this is an identity, so it checks the code (the exact dZ/domega
    against the energy Gram of the same fields), not the accuracy of L_eff."""
    for row in coax_o1["fullwave"]["sweep"]:
        Wm = row["energy_magnetic_J_per_A2"]["re"][0][0]
        assert row["L_eff_H"] == pytest.approx(4 * Wm, rel=1e-9)


@needs_3d
def test_frequency_dependence_follows_the_closed_form(coax_o1):
    """Gamma(100 MHz) / Gamma(1 MHz) of each surface against the TEM closed form's 2.1-3.7. Order 1 on this mesh is
    up to 36 % off in Gamma itself (three elements across the 1/r field), and its ratios are 3-8 % off (its wave
    speed); the order-2 comparisons are slow tests below."""
    low, high = coax_o1["fullwave"]["sweep"]
    ref_low, ref_high = FSL.tem_closed_form(low["f_Hz"]), FSL.tem_closed_form(high["f_Hz"])
    for k in SURFACES:
        ratio = high["surface_Gamma"][k] / low["surface_Gamma"][k]
        assert ratio == pytest.approx(ref_high["Gamma"][k] / ref_low["Gamma"][k], rel=0.15), k


@needs_3d
def test_python_columns(coax_o1):
    t = circuit.fullwave_losses(coax_o1)
    assert list(t["f_Hz"]) == [1e6, 1e8]
    for key in ("R_loss_ohm", "Q_loss", "Q_loss_tuned", "L_eff_H", "R_loss_over_abs_X"):
        assert all(math.isfinite(x) for x in t[key]), key
    assert set(t["Gamma"]) == set(SURFACES)
    # the reach and circuit inputs add R_loss to Re Z; two rows, the second 1.6 times below the pole, are far too sparse
    # to interpolate R_loss between, and the check says so
    with pytest.warns(RuntimeWarning, match="between the rows at 1e\\+06 and 1e\\+08 Hz R_loss interpolated"):
        tables = circuit.fullwave_tables(coax_o1)
    assert tables["R_loss_ohm"][1] == pytest.approx(t["R_loss_ohm"][1], rel=1e-12)


@needs_3d
def test_multiport_matrices(tmp_path):
    """The two-port frame (copper loop, two gap ports, closed sphere, order 1): per surface the Hermitian
    Gamma matrix oint H_j^* . H_k dA, whose diagonal is each port's Gamma, and the R_loss matrix, whose diagonal is
    each port's R_loss."""
    doc = json.loads((ROOT / "benchmarks" / "designs" / "stage1_twoport.json").read_text())
    for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
        doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [1e7]
    row = af.Design(doc).solve(tmp_path / "s1", verbose=0).data["fullwave"]["sweep"][0]
    ports = row["port_solutions"]
    G = row["surface_Gamma_matrix"]["loop"]
    R = row["R_loss_matrix_ohm"]
    for j in range(2):
        assert G["re"][j][j] == pytest.approx(ports[j]["surface_Gamma"]["loop"], rel=1e-12)
        assert abs(G["im"][j][j]) <= 1e-12 * G["re"][j][j]
        assert R["re"][j][j] == pytest.approx(ports[j]["R_loss_ohm"], rel=1e-12)
        for k in range(2):
            assert G["re"][j][k] == pytest.approx(G["re"][k][j], rel=1e-12)
            assert G["im"][j][k] == pytest.approx(-G["im"][k][j], rel=1e-12, abs=1e-14 * G["re"][0][0])
    # positive semidefinite, and the ports share the loop's current (a coupling well above zero)
    cross = math.hypot(G["re"][0][1], G["im"][0][1])
    assert 0.01 * math.sqrt(G["re"][0][0] * G["re"][1][1]) < cross <= math.sqrt(G["re"][0][0] * G["re"][1][1])


# a measured R_s table that does not reach 1 Hz: the magnetoresistance weight must not evaluate R_s;
# "tiny": a table 1e4 times lower, whose R_loss is below 100 times the residual's reach on Re Z
MR_RF = {"skin": {"model": "skin", "rho_ohm_m": 1.7e-10, "T_K": 0.03},
         "table": {"model": "table", "f_Hz": [1e3, 1e9], "Rs_ohm": [1e-6, 1e-3], "T_K": 0.03},
         "tiny": {"model": "table", "f_Hz": [1e3, 1e9], "Rs_ohm": [1e-10, 1e-7], "T_K": 0.03}}


def _mr_Rs(model, f):
    if model == "skin":
        return math.sqrt(math.pi * f * FSL.MU0 * FSL.RHO_CU)
    scale = 1e-6 if model == "table" else 1e-10
    return scale * (f / 1e3) ** 0.5   # the table's log-log line


@needs_3d
@pytest.mark.parametrize("model", ["skin", "table", "tiny"])
def test_magnetoresistance_weight_matches_pickup_losses(tmp_path, model):
    """The solenoid template with a Kohler magnetoresistance on the coax copper (1 + (B / 0.3 T)^2 in the 0.3-0.45 T
    of the solenoid), order 1, full wave at kR = 1e-3: the full-wave weight Gamma_magnetoresistance / Gamma equals
    pickup.losses' (the same static field on the same faces), and R_eff takes the weighted Gamma. With a measured R_s
    table (1 kHz to 1 GHz) the run must not abort: the weight is Rs_factor(B), not R_s(1 Hz, B) / R_s(1 Hz). At
    kR = 1e-3 the radiation is far below what the residual resolves: the row says
    Re_Z_unresolved. Q_total is kept where R_loss dominates the residual's reach on Re Z (skin, table: Q_total = Q_loss
    to 1e-4) and withheld where it does not (tiny)."""
    d2 = af.template("solenoid", **FSL.SOLENOID, mesh=FSL.SOLENOID_2D)
    d2.doc.setdefault("materials", {})["cu_mr"] = {
        "kind": "conductor", "rf": {**MR_RF[model], "magnetoresistance": {"model": "kohler", "B0_T": 0.3}}}
    for region in d2.doc["regions"]:
        if region["material"] == "copper":
            region["material"] = "cu_mr"
    d3 = to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 1})
    d3.doc["fullwave"] = {"frequencies": [1e-3 * FSL.C / (2 * math.pi * 1.05)]}
    data = d3.solve(tmp_path / "mr", verbose=0).data
    mqs = data["pickup"]["losses"]["surfaces"]
    row = data["fullwave"]["sweep"][0]
    summary = data["fullwave"]["surface_losses"]
    assert summary["surfaces"]["coax_inner"]["magnetoresistance"].startswith("weighted")
    Rs = _mr_Rs(model, row["f_Hz"])
    for k, e in row["surface_losses"]["surfaces"].items():
        weight = e["Gamma_magnetoresistance"] / e["Gamma"]
        assert weight > 1.1, k
        assert weight == pytest.approx(mqs[k]["Gamma_magnetoresistance"] / mqs[k]["Gamma"], rel=1e-6), k
        assert e["R_eff_ohm"] == pytest.approx(Rs * e["Gamma_magnetoresistance"], rel=1e-9), k
    # a sealed coax radiates nothing resolvable at kR = 1e-3: Re Z / |X| is far below 100 times the residual
    rho = row["solver"]["relative_residual"]
    unresolved = row["Re_Z_unresolved"]
    assert unresolved["Re_Z_over_abs_X"] < 100 * rho and unresolved["true_relative_residual"] == rho
    assert any("R_rad_ohm and Q_rad are not resolved" in w for w in data["warnings"])
    assert "Q_loss_tuned" in row
    uncertainty = rho * abs(row["Z_im_ohm"])
    if model == "tiny":
        assert uncertainty > 0.01 * row["R_loss_ohm"]
        assert "Q_total" not in row and "Q_total_tuned" not in row
        assert row["surface_losses"]["Q_total_withheld"]["true_relative_residual"] == rho
        assert summary["Q_total_withheld_port_rows"] == 1
        assert any("Q_total is withheld in 1 port rows" in w for w in data["warnings"])
    else:
        assert uncertainty < 1e-3 * row["R_loss_ohm"]
        assert row["Q_total"] == pytest.approx(row["Q_loss"], rel=1e-4)
        assert row["Q_total_tuned"] == pytest.approx(row["Q_loss_tuned"], rel=1e-4)
        assert "Q_total_withheld" not in row["surface_losses"] and "Q_total_withheld_port_rows" not in summary


@needs_3d
def test_pec_boundary_conductors_are_reported(tmp_path):
    """The shorted parallel-plate TEM line of the indicator's tests: its plates are PEC boundary groups, so no surface
    carries loss, and the summary and a warning say so."""
    import fullwave_indicator as FWI
    doc, _ = FWI.tem_design(tmp_path, "pec", 1, 0, [1e8])
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    summary = data["fullwave"]["surface_losses"]
    assert summary["faces"] == 0 and summary["pec_boundary_groups_without_loss"]
    assert "surface_Gamma" not in data["fullwave"]["sweep"][0]
    assert any("no surface loss is computed" in w for w in data["warnings"])


def _rows(R_over_X, f=(1e6, 2e6, 3e6, 4e6)):
    return [{"f_Hz": fi, "Z_im_ohm": -1.0, "R_loss_ohm": r} for fi, r in zip(f, R_over_X)]


def _lumped_rows(freqs, exact):
    """A lumped pole at 163 MHz (the coax's TEM_1): L = L0/(1 - x^2), L_eff = L0/(1 - x^2)^2 and R_loss =
    R0 (f / 1 MHz)^(1/2) L_eff / L0 (a slowly varying tuned Q), x = f/f0."""
    L0, f0, R0 = 1e-7, 1.63e8, 1e-4
    rows = []
    for f in freqs:
        x = f / f0
        L = L0 / (1 - x * x)
        r = {"f_Hz": f, "Z_im_ohm": -2 * math.pi * f * L, "L_H": L,
             "R_loss_ohm": R0 * math.sqrt(f / 1e6) / (1 - x * x) ** 2}
        if exact:
            r["L_eff_H"] = L0 / (1 - x * x) ** 2
        rows.append(r)
    return rows


def test_loss_interpolation_check():
    """Far from a pole R_loss/|X| changes like f^(-1/2) to f; toward a pole like 1/Delta. A log-log slope above 3
    between neighbouring rows is flagged, and so is an interval whose log-log R_loss is estimated more than
    5 % off: from the curvature of the neighbouring rows, or, with the rows'
    exact L_eff, against omega L_eff / Q_tuned."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert circuit.loss_interpolation_check(_rows([1e-4, 1.4e-4, 1.7e-4, 2e-4])) == []
    with pytest.warns(RuntimeWarning, match="unreliable") as record:
        flagged = circuit.loss_interpolation_check(_rows([1e-4, 1.2e-4, 1e-3, 1.2e-3]))
    assert (2e6, 3e6) in flagged
    assert any("self-resonance is near" in str(w.message) for w in record)
    # the coax benchmark's rows on a lumped pole: 100-140 MHz has a slope of 2.1 only, but log-log R_loss is 38 % off
    freqs = [1e6, 5e7, 1e8, 1.4e8, 1.55e8, 1.6e8]
    for exact in (True, False):
        with pytest.warns(RuntimeWarning, match="unreliable") as record:
            flagged = circuit.loss_interpolation_check(_lumped_rows(freqs, exact))
        assert (1e8, 1.4e8) in flagged, exact
        assert any("1e+08 and 1.4e+08 Hz R_loss interpolated log-log is off by about" in str(w.message)
                   for w in record), exact
    # dense rows far below the pole pass
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for exact in (True, False):
            assert circuit.loss_interpolation_check(_lumped_rows(list(np.geomspace(1e5, 1e7, 5)), exact)) == []


def _reach_rows(L_eff_offset=1e-8, inconsistent=False):
    """Rows with L = L_p (1 + 0.02 ln(f / 1 MHz)), so L_eff = L + 1e-8 H exactly; inconsistent: L_eff = 1.5 L_p with a
    constant L, which no interpolant of L reproduces."""
    rows = []
    for fi in [1e6, 1.5e6, 2e6, 3e6, 4e6]:
        L = 1e-6 if inconsistent else 1e-6 * (1 + 0.02 * math.log(fi / 1e6))
        rows.append({"f_Hz": fi, "Psi_re": 1e-5, "Psi_im": 0.0, "L_H": L, "Z_re_ohm": 0.0,
                     "Z_im_ohm": -2 * math.pi * fi * L, "R_loss_ohm": 1e-6,
                     "L_eff_H": 1.5e-6 if inconsistent else L + L_eff_offset})
    return {"name": "synthetic", "coupling": {"Psi_SI_H_T_m2": 1e-5, "L_p_H": 1e-6}, "fullwave": {"sweep": rows}}


def test_reach_uses_the_rows_exact_tuned_inductance():
    """With L_eff_H in every row (3D), the full-wave scan takes L_eff from a cubic Hermite interpolant of L with the
    rows' exact slopes, guarded by the linear interpolation of the rows' L_eff, and the MQS comparison keeps L_p
    (not the full-wave L_eff)."""
    bench = {"Q": 1e6, "T": 0.02, "eta_A": 1.0}
    data = _reach_rows()
    r = scan.reach_fullwave(data, bench, n=5)
    L = 1e-6 * (1 + 0.02 * np.log(r["f_Hz"] / 1e6))
    assert r["L_eff_H"] == pytest.approx(L + 1e-8, rel=1e-9)
    assert r["band"]["pole_reduction"].startswith("L_eff from the rows' exact dZ/domega")
    assert r["mqs"]["L_eff_H"] == pytest.approx(1e-6, rel=1e-12)
    # without the rows' L_eff the spline of L gives the same L_eff here (L is linear in ln f), and the same ratios
    for row in data["fullwave"]["sweep"]:
        del row["L_eff_H"]
    r2 = scan.reach_fullwave(data, bench, n=5)
    assert r2["L_eff_H"] == pytest.approx(r["L_eff_H"], rel=1e-9)
    assert r2["mqs"]["L_eff_H"] == pytest.approx(1e-6, rel=1e-12)
    assert r2["ratio_g95_fullwave_over_mqs"] == pytest.approx(r["ratio_g95_fullwave_over_mqs"], rel=1e-9)
    # a caller's L_eff_H goes to the full-wave scan only
    r3 = scan.reach_fullwave(_reach_rows(), bench, n=5, L_eff_H=1.2e-6)
    assert r3["L_eff_H"] == pytest.approx(1.2e-6, rel=1e-12)
    assert r3["mqs"]["L_eff_H"] == pytest.approx(1e-6, rel=1e-12)
    assert r3["band"]["pole_reduction"].startswith("L_eff supplied by the caller")
    # rows whose L_eff no interpolant of L reproduces are refused, as the other path refuses sparse rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the interpolation check flags these rows too
        with pytest.raises(ValueError, match="between interpolants of the rows' exact L_eff"):
            scan.reach_fullwave(_reach_rows(inconsistent=True), bench, n=5)


def test_tables_warn_where_re_z_is_unresolved():
    """fullwave_tables warns where a row's unresolved Re Z (Re_Z_unresolved) can move R_ohm by more than 1 %."""
    data = _reach_rows()
    rows = data["fullwave"]["sweep"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit.fullwave_tables(data)
    rows[0]["Re_Z_unresolved"] = {"true_relative_residual": 1e-12}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit.fullwave_tables(data)   # 1e-12 |X| is far below 1 % of R_loss
    rows[0]["Re_Z_unresolved"] = {"true_relative_residual": 1e-6}
    with pytest.warns(RuntimeWarning, match="Re Z is not resolved by the solve at 1 rows"):
        circuit.fullwave_tables(data)


def test_rows_without_dz_domega_have_no_tuned_q():
    """2D rows have no exact dZ/domega, so fullwave_losses leaves their tuned columns NaN. A finite difference of L
    between the rows was tried and removed: on the coax benchmark's 2D rows it was -21 % to +106 % off the closed form,
    without a warning."""
    L0, f0, R, T = 1e-6, 1e8, 1e-3, 4.0
    rows = []
    for fi in np.geomspace(1e6, 5e7, 8):
        L = L0 / (1 - (fi / f0) ** 2)
        w = 2 * math.pi * fi
        rows.append({"f_Hz": fi, "L_H": L, "Z_re_ohm": 0.0, "Z_im_ohm": -w * L, "surface_Gamma": {"wall": 1.0},
                     "R_loss_ohm": R, "Q_loss": w * L / R, "Q_total": w * L / R, "T_over_Q_K": T * R / (w * L)})
    t = circuit.fullwave_losses({"fullwave": {"sweep": rows}})
    assert all(math.isfinite(q) for q in t["Q_loss"])
    for key in ("L_eff_H", "Q_loss_tuned", "Q_total_tuned", "T_over_Q_tuned_K"):
        assert all(math.isnan(v) for v in t[key]), key
    assert all(isinstance(v, (np.ndarray, dict)) for v in t.values()), "only arrays and per-surface dicts"


# ------------------------------------------------------------------ slow
@pytest.fixture(scope="module")
def coax_o2(tmp_path_factory):
    """Level 0 of the benchmark (order 2, surface size 0.04, 0.8M DOFs) at 100 MHz, 1.8 % below the TEM_1 pole and
    7 % above it."""
    d = FSL.coax_design3d(0)
    d.doc["fullwave"]["frequencies"] = [1e8, 1.6e8, 1.75e8]
    return d.solve(tmp_path_factory.mktemp("coax_o2"), verbose=0, mpi_ranks=RANKS).data


@needs_3d
@slow
def test_order2_against_the_closed_form_up_to_the_pole(coax_o2):
    """Order 2 against the TEM closed form. This mesh puts 12.6 faces around the inner conductor, whose 1/r field
    order 2 resolves to 5-6 % in Gamma there (the walls further out and the end plates to 2 %); the h-convergence
    is the benchmark's (benchmarks/fullwave_surface_losses.py). Near the pole Q_loss collapses (to 0.036 of the tuned
    Q at 160 MHz) while Q_loss_tuned stays the resonance's own Q, and it keeps the off-pole accuracy: it does not
    depend on where exactly the discrete pole sits."""
    tolerance = {"inner": 0.07, "outer": 0.025, "top": 0.025, "bottom": 0.025}
    for row in coax_o2["fullwave"]["sweep"]:
        ref = FSL.tem_closed_form(row["f_Hz"])
        for k in SURFACES:
            assert row["surface_Gamma"][k] == pytest.approx(ref["Gamma"][k], rel=tolerance[k]), (row["f_Hz"], k)
        if row["L_H"] > 0:
            assert row["Q_loss_tuned"] == pytest.approx(ref["Q_loss_tuned"], rel=0.05), row["f_Hz"]
            Wm = row["energy_magnetic_J_per_A2"]["re"][0][0]
            assert row["L_eff_H"] == pytest.approx(4 * Wm, rel=1e-9)
        else:   # above the pole: capacitive, no Q
            assert "Q_loss" not in row and "Q_loss_tuned" not in row and "R_loss_ohm" in row
    near = coax_o2["fullwave"]["sweep"][1]
    assert near["Q_loss"] < 0.05 * near["Q_loss_tuned"]


@needs_3d
@slow
def test_three_dimensional_against_the_axisymmetric_engine(coax_o2, tmp_path):
    """The axisymmetric engine's full-wave surface_Gamma on the same coax (order 3: within 1e-4 of the closed form),
    and the 3D Gamma's change with frequency against it: Gamma(f) / Gamma_MQS of each engine's own run, which
    removes the 3D mesh's error in the 1/r profile (the same at every f in a TEM line)."""
    d2 = af.Design(FSL.coax_doc([1e8], FSL.COAX_2D))
    r2 = d2.solve(tmp_path / "c2", verbose=0).data
    row2 = r2["fullwave"]["sweep"][0]
    ref = FSL.tem_closed_form(1e8)
    for k in SURFACES:
        assert row2["surface_Gamma"][k] == pytest.approx(ref["Gamma"][k], rel=1e-4), k
    mqs2 = {k: v["Gamma"] for k, v in r2["pickup"]["losses"]["surfaces"].items()}
    mqs3 = {k: v["Gamma"] for k, v in coax_o2["pickup"]["losses"]["surfaces"].items()}
    row3 = coax_o2["fullwave"]["sweep"][0]
    for k in SURFACES:
        change3 = row3["surface_Gamma"][k] / mqs3[k]
        change2 = row2["surface_Gamma"][k] / mqs2[k]
        assert change3 == pytest.approx(change2, rel=3e-3), k


@needs_3d
@slow
@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs MPI ranks (AXF_MPI=1)")
def test_ranks_agree(tmp_path):
    """One process against three ranks: the same faces, evaluated on different ranks and reduced."""
    d = _coax3d([1e8], {"order": 1, "size": 0.3, "surface_size": 0.05})
    one = d.solve(tmp_path / "r1", verbose=0, mpi_ranks=1).data
    three = d.solve(tmp_path / "r3", verbose=0, mpi_ranks=3).data
    assert three["fullwave"]["surface_losses"]["faces"] == one["fullwave"]["surface_losses"]["faces"]
    a, b = one["fullwave"]["sweep"][0], three["fullwave"]["sweep"][0]
    for k in SURFACES:
        assert b["surface_Gamma"][k] == pytest.approx(a["surface_Gamma"][k], rel=1e-8), k
    assert b["R_loss_ohm"] == pytest.approx(a["R_loss_ohm"], rel=1e-8)
    assert b["Q_loss_tuned"] == pytest.approx(a["Q_loss_tuned"], rel=1e-8)


@needs_3d
@slow
def test_adaptive_rows_carry_the_losses(tmp_path):
    """An adaptive sweep's solved rows carry the losses; the model's dense curve does not."""
    d = _coax3d([1e8], {"order": 1, "size": 0.3, "surface_size": 0.05})
    d.doc["fullwave"] = {"boundary": "pec",
                         "adaptive": {"f_min_Hz": 5e7, "f_max_Hz": 1.2e8, "tolerance": 1e-3, "max_solves": 8}}
    data = d.solve(tmp_path / "ad", verbose=0).data
    fw = data["fullwave"]
    assert all("R_loss_ohm" in row and "surface_Gamma" in row for row in fw["sweep"])
    assert fw["adaptive"]["dense"] is not None   # the model was fitted (dense is null only when nothing was)
    assert "R_loss_ohm" not in fw["adaptive"]["dense"]
    assert "adaptive_sweep" in fw["surface_losses"]
    assert fw["surface_losses"]["evaluations"] == len(fw["adaptive"]["solved_frequencies_Hz"])
    t = circuit.fullwave_losses(data)
    assert len(t["f_Hz"]) == len(fw["sweep"])
