"""Surface-resistance models of `rf` walls (beta): anomalous skin effect, BCS + residual, flux flow.

Formulas, regimes and sources: docs/physics/surface_resistance.md. Reference numbers: independent scipy evaluations
and an independent (blind) derivation of the same quantities, reconciled with them
(C++ unit test with more limits: cpp/tests/test_surface_resistance.cpp).
"""
import json
import math

import numpy as np
import pytest
from scipy.linalg import solve_banded

import axionfem as af
from axionfem import cavity as cv

MU0 = 1.25663706212e-6
C = 299792458.0
A, L = 0.1, 0.2
CHI01 = 2.404825557695773
F_TM010 = C * CHI01 / (2 * math.pi * A)
G_TM010 = 2 * math.pi * F_TM010 * MU0 * A * L / (2 * (A + L))  # 301.990 ohm
RHOL_CU = 6.6e-16
CU_RRR100 = {"model": "anomalous", "RRR": 100, "rho_300K_ohm_m": 1.68e-8, "rho_l_ohm_m2": RHOL_CU, "T_K": 4.0}
# vortex parameters (Nb3Sn-like film; docs/physics/surface_resistance.md Sec. 4)
FF = {"Bc2_T": 10.0, "rho_n_ohm_m": 1.5e-7, "lambda_m": 1.1e-7, "f_depin_Hz": 1e9}
NB = {"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 17.67 / 9.25, "Tc_K": 9.25,
      "R_res_ohm": 5e-9, "T_K": 2.0}


def test_anomalous_copper_values():
    # independent derivation (RRR 100, rho(300 K) = 1.68e-8, diffuse, full RS crossover), confirmed by scipy quad
    for f, ref in [(1e9, 1.353456459e-3), (5e9, 3.716853468e-3), (10e9, 5.780878115e-3)]:
        assert cv.surface_resistance(CU_RRR100, f) == pytest.approx(ref, rel=1e-8, abs=0)
    # same material by rho_ohm_m
    rho = dict(CU_RRR100)
    del rho["RRR"], rho["rho_300K_ohm_m"]
    rho["rho_ohm_m"] = 1.68e-10
    assert cv.surface_resistance(rho, 2.3e9) == pytest.approx(cv.surface_resistance(CU_RRR100, 2.3e9), rel=1e-14, abs=0)
    # R_s(4 K) / R_s(300 K) at 1.5 GHz: 0.1746 from the independent derivation
    r = cv.surface_resistance(CU_RRR100, 1.5e9) / cv.skin_Rs(1.5e9, 1.68e-8)
    assert r == pytest.approx(0.17462148808, rel=1e-8, abs=0)


def test_anomalous_limits():
    f = 3e9
    # classical (l -> 0 at fixed rho): skin
    cl = {"model": "anomalous", "rho_ohm_m": 1.7e-8, "rho_l_ohm_m2": 1.7e-8 * 1e-12, "T_K": 300}
    assert cv.surface_resistance(cl, f) == pytest.approx(cv.skin_Rs(f, 1.7e-8), rel=1e-6, abs=0)
    # extreme anomalous (rho -> 0 at fixed rho_l): R_inf, diffuse and 8/9 of it specular
    w = 2 * math.pi * f
    rinf = (math.sqrt(3) / (16 * math.pi) * RHOL_CU * (MU0 * w) ** 2) ** (1 / 3)
    rho = (0.75 * MU0 * w * RHOL_CU**2 / 1e15) ** (1 / 3)  # alpha = 1e15
    ex = {"model": "anomalous", "rho_ohm_m": rho, "rho_l_ohm_m2": RHOL_CU, "T_K": 0.01}
    assert cv.surface_resistance(ex, f) == pytest.approx(rinf, rel=2e-4, abs=0)
    assert cv.surface_resistance(dict(ex, specularity=1), f) == pytest.approx(8 / 9 * rinf, rel=3e-4, abs=0)
    # specularity is monotone between the two exact ends
    rs = [cv.surface_resistance(dict(CU_RRR100, specularity=p), 5e9) for p in (0, 0.25, 0.5, 0.75, 1)]
    assert all(a > b for a, b in zip(rs, rs[1:]))


def test_copper_cavity_data_point():
    """Measured S-band copper TM010 cavities (Cahill et al., IPAC2016 MOPMW038, Table 1): Q0 = 17184 at 293 K
    (2.845 GHz) -> 82931 at 4 K (2.855 GHz), ratio 4.83 (the twin cavity: 4.78). RRR not measured (brazed OFHC; the
    paper assumes 400 and notes the RS prediction is too high). With rho(293 K) = 1.678e-8/0.95 (95 % IACS, as the
    paper) the diffuse RS ratio is 4.35 (RRR 30), 4.85 (RRR 50), 5.37 (RRR 100), 5.93 (RRR 400): the measurement lies
    in the model band and corresponds to an effective RRR ~ 50. Theory overestimating cold Q by 10-25 % is the usual
    finding (surface damage, oxide, roughness)."""
    r293 = 1.678e-8 / 0.95
    q_ratio = 82931 / 17184

    def ratio(rrr):
        cold = {"model": "anomalous", "RRR": rrr, "rho_300K_ohm_m": r293, "rho_l_ohm_m2": RHOL_CU, "T_K": 4}
        return cv.skin_Rs(2.845e9, r293) / cv.surface_resistance(cold, 2.855e9)  # G cancels (same cavity)
    assert ratio(30) < q_ratio < ratio(400)
    assert ratio(50) == pytest.approx(q_ratio, rel=0.02, abs=0)
    assert ratio(400) == pytest.approx(5.9298155, rel=1e-6, abs=0)


def test_bcs_and_flux_flow():
    # independent derivation (Padamsee-style Nb parameters): 10.93 nOhm (2 K), 532.5 nOhm (4.2 K) at 1.3 GHz, + R_res
    assert cv.surface_resistance(NB, 1.3e9) == pytest.approx(1.0932322604e-8 + 5e-9, rel=1e-9, abs=0)
    assert cv.surface_resistance(dict(NB, T_K=4.2), 1.3e9) == pytest.approx(5.325323121e-7 + 5e-9, rel=1e-9, abs=0)
    # tabulated R_BCS(f, T): log-log in f, ln R linear in 1/T; at grid points it returns the table
    tab = {"model": "bcs", "f_Hz": [1e9, 2e9], "T_grid_K": [2.0, 4.0], "Rs_bcs_ohm": [[1e-8, 4e-8], [4e-7, 1.6e-6]],
           "R_res_ohm": 2e-9, "T_K": 4.0}
    assert cv.surface_resistance(tab, 1e9) == pytest.approx(4e-7 + 2e-9, rel=1e-12, abs=0)
    t3 = cv.surface_resistance(dict(tab, T_K=3.0), 2 ** 0.5 * 1e9)
    lnR = math.log(2e-8) + (1 / 3 - 1 / 2) / (1 / 4 - 1 / 2) * math.log(40)  # 1/T interpolation at f = sqrt2 GHz
    assert t3 == pytest.approx(math.exp(lnR) + 2e-9, rel=1e-12, abs=0)
    # flux flow: additive Re sqrt(i mu0 w (i mu0 w lambda^2 + rho_v)), B_T given, orientation factor 1 (upper bound)
    ff = dict(NB, flux_flow=dict(FF, B_T=2.0))
    Rs, part = cv.surface_resistance(ff, 3e9, flux_flow_part=True)
    assert part == pytest.approx(_vortex_R(3e9, 2.0, **_ff_args(FF)), rel=1e-12, abs=0)
    assert Rs == pytest.approx(cv.surface_resistance(NB, 3e9) + part, rel=1e-14, abs=0)
    # BCS log factor: ln(2.246 kT/hf) normalized at f_ref
    lg = cv.surface_resistance(dict(NB, R_res_ohm=0, log_factor=True), 10e9)
    assert lg == pytest.approx(cv.surface_resistance(dict(NB, R_res_ohm=0), 10e9) * 0.541042850697, rel=1e-10, abs=0)


def _ff_args(ff):
    return dict(rho_n=ff["rho_n_ohm_m"], lam=ff["lambda_m"], Bc2=ff["Bc2_T"], f0=ff["f_depin_Hz"],
                eps=ff.get("creep_epsilon", 0.0))


def _rho_v(f, B, rho_n, Bc2, f0, eps=0.0):
    rff = rho_n * abs(B) / Bc2
    return rff if f0 == 0 else rff * (eps + 1j * f / f0) / (1 + 1j * f / f0)


def _vortex_R(f, B, rho_n, lam, Bc2, f0, eps=0.0, s=1.0):
    """Closed form Re sqrt(i mu0 w (i mu0 w lambda^2 + s rho_v)) (written independently of the C++)."""
    w = 2 * math.pi * f
    return (np.sqrt(1j * w * MU0 * (1j * w * MU0 * lam**2 + s * _rho_v(f, B, rho_n, Bc2, f0, eps)))).real


def _Z_halfspace_fd(f, rho_t, N):
    """Independent reference: half-space z > 0 with local E = rho_t J; Maxwell dE/dz = -i w mu0 H, dH/dz = -E/rho_t,
    H(0) = 1, E(z_max) = 0 (40 decay lengths), second-order finite differences; Z = E(0)."""
    w = 2 * math.pi * f
    k2 = 1j * w * MU0 / rho_t
    h = 40 / np.sqrt(k2).real / N
    ab = np.zeros((3, N), complex)
    ab[0, 1:], ab[2, :-1], ab[1, :] = 1, 1, -2 - k2 * h * h
    ab[0, 1] = 2  # ghost point E_-1 = E_1 + 2 h i w mu0 (E'(0) = -i w mu0 H(0))
    b = np.zeros(N, complex)
    b[0] = -2 * h * 1j * w * MU0
    return solve_banded((1, 1), ab, b)[0]


@pytest.mark.parametrize("f, B, ff, regime", [
    (5e9, 8.0, {"Bc2_T": 25.0, "rho_n_ohm_m": 1.5e-7, "lambda_m": 110e-9, "f_depin_Hz": 0.0}, "large"),   # Nb3Sn
    (5e9, 8.0, {"Bc2_T": 11.0, "rho_n_ohm_m": 6e-7, "lambda_m": 300e-9, "f_depin_Hz": 0.0}, "large"),     # NbTi
    (3e9, 1.0, dict(FF, creep_epsilon=0.3, f_depin_Hz=5e9), "large"),
    (1e9, 0.1, {"Bc2_T": 10.0, "rho_n_ohm_m": 1e-9, "lambda_m": 40e-9, "f_depin_Hz": 1e10}, "small"),
    (2e9, 1.0, {"Bc2_T": 10.0, "rho_n_ohm_m": 1e-12, "lambda_m": 50e-9, "f_depin_Hz": 4e9}, "small"),
])
def test_flux_flow_full_impedance_vs_halfspace(f, B, ff, regime):
    """The vortex term is the full local impedance, valid for small and large vortex resistivity. Reference:
    finite-difference half-space solution (Richardson, ~1e-8), and the two asymptotic laws Re rho_v / (2 lambda)
    (|rho_v| << mu0 w lambda^2) and Re sqrt(i mu0 w rho_v) (|rho_v| >> mu0 w lambda^2)."""
    rf = {"model": "power", "Rs_ohm": 1e-30, "T_K": 4.2, "flux_flow": dict(ff, B_T=B)}
    _, R = cv.surface_resistance(rf, f, flux_flow_part=True)
    w = 2 * math.pi * f
    rv = _rho_v(f, B, ff["rho_n_ohm_m"], ff["Bc2_T"], ff["f_depin_Hz"], ff.get("creep_epsilon", 0.0))
    rho_t = 1j * w * MU0 * ff["lambda_m"] ** 2 + rv
    z1, z2 = _Z_halfspace_fd(f, rho_t, 200000), _Z_halfspace_fd(f, rho_t, 400000)
    assert R == pytest.approx(((4 * z2 - z1) / 3).real, rel=1e-6, abs=0)
    ratio = abs(rv) / (w * MU0 * ff["lambda_m"] ** 2)
    if regime == "small":
        assert ratio < 0.1
        assert R == pytest.approx(rv.real / (2 * ff["lambda_m"]), rel=ratio, abs=0)
    else:
        assert ratio > 30
        assert R == pytest.approx(np.sqrt(1j * w * MU0 * rv).real, rel=1 / ratio, abs=0)
        if ff["f_depin_Hz"] == 0:
            assert R < 0.2 * rv.real / (2 * ff["lambda_m"])  # an additive vortex term would be 7-8x high here


@pytest.mark.parametrize("rf, what", [
    ({"model": "anomalous", "rho_l_ohm_m2": 6.6e-16, "T_K": 4}, "either rho_ohm_m"),
    ({"model": "anomalous", "rho_ohm_m": 1e-10, "RRR": 100, "rho_300K_ohm_m": 1.7e-8, "rho_l_ohm_m2": 6.6e-16,
      "T_K": 4}, "either rho_ohm_m"),
    ({"model": "anomalous", "rho_ohm_m": 1e-10, "T_K": 4}, "rho_l_ohm_m2"),
    ({"model": "anomalous", "rho_ohm_m": 1e-10, "rho_l_ohm_m2": 6.6e-16, "specularity": 1.5, "T_K": 4}, r"\[0, 1\]"),
    ({"model": "anomalous", "RRR": 0.5, "rho_300K_ohm_m": 1.7e-8, "rho_l_ohm_m2": 6.6e-16, "T_K": 4}, "RRR must"),
    ({"model": "anomalous", "rho_ohm_m": 1e-10, "rho_l_ohm_m2": 6.6e-16, "T_K": 4,
      "magnetoresistance": {"B0_T": 5}}, "not combined"),
    ({"model": "skin", "rho_ohm_m": 1e-10, "specularity": 0, "T_K": 4}, "does not belong to the 'skin' model"),
    ({"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 1.9, "Tc_K": 9.25, "T_K": 2}, "R_res_ohm"),
    ({"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 1.9, "Tc_K": 9.25, "R_res_ohm": 0,
      "T_K": 6}, "T < Tc/2"),
    ({"model": "bcs", "f_Hz": [1e9, 2e9], "T_grid_K": [2, 4], "Rs_bcs_ohm": [[1e-8, 2e-8]], "R_res_ohm": 0,
      "T_K": 3}, "one per T_grid_K"),
    ({"model": "bcs", "f_Hz": [1e9, 2e9], "T_grid_K": [2, 4], "Rs_bcs_ohm": [[1e-8, 2e-8], [1e-7, 2e-7]],
      "R_res_ohm": 0, "T_K": 5}, "no extrapolation"),
    (dict(NB, flux_flow=dict(FF, B_T=12)), "mixed state"),
    ({"model": "skin", "rho_ohm_m": 1e-10, "T_K": 4, "flux_flow": dict(FF, B_T=1)}, "does not belong"),
    (dict(NB, flux_flow={"B_T": 1, "Bc2_T": 10, "Rn_ohm": 1e-3, "f_depin_Hz": 1e9}), "Rn_ohm' is not an input"),
    (dict(NB, flux_flow=dict(FF, B_T=1, orientation=1.5)), r"in \[0, 1\]"),
    (dict(NB, flux_flow=dict(FF, B_T=1, orientation="auto")), "\"field\" or a number"),
    (dict(NB, flux_flow=dict(FF, B_T=1, creep_epsilon=2)), "creep_epsilon"),
    (dict(NB, flux_flow={k: v for k, v in FF.items() if k != "lambda_m"} | {"B_T": 1}), "lambda_m"),
    (dict(NB, flux_flow=dict(FF)), "local static field"),          # B from the field: cavity wall losses only
    # magnetoresistance refusals with bcs and with flux_flow
    (dict(NB, magnetoresistance={"B0_T": 5}), "not combined with the 'bcs' model"),
    ({"model": "power", "Rs_ohm": 1e-6, "T_K": 4, "flux_flow": dict(FF, B_T=1), "magnetoresistance": {"B0_T": 5}},
     "cannot be combined"),
    # RRR is a residual-resistivity ratio, not rho(300 K)/rho(T)
    ({"model": "anomalous", "RRR": 100, "rho_300K_ohm_m": 1.725e-8, "rho_l_ohm_m2": 6.6e-16, "T_K": 77},
     "give rho_ohm_m at T_K|valid below ~20 K"),
    (dict(NB, log_factor=True, T_K=0.02), "h f_ref < k T_K"),
])
def test_rf_model_validation(rf, what):
    with pytest.raises(af.AxionFEMError, match=what):
        cv.surface_resistance(rf, 1e9)


def test_pillbox_q0_copper_300K_and_4K(tmp_path):
    """Independent derivation: TM010 of a copper pillbox a = 0.1 m, L = 0.2 m: G = 302.0 ohm, Q0 = 3.462e4
    (300 K, rho = 1.68e-8) and 2.049e5 (4 K, RRR 100, diffuse RS crossover)."""
    out = {}
    for tag, rf in [("300K", {"model": "skin", "rho_ohm_m": 1.68e-8, "T_K": 300}), ("4K", CU_RRR100)]:
        d = cv.pillbox_design(A, L, h=0.004, modes=2, families=("TM",), wall_material={"kind": "conductor", "rf": rf})
        m = d.solve(tmp_path / tag, verbose=0).data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
        assert m["f_Hz"] == pytest.approx(F_TM010, rel=1e-6, abs=0)
        out[tag] = m["losses"]
    w300, w4 = out["300K"]["walls"]["wall"], out["4K"]["walls"]["wall"]
    assert w300["Rs_model"] == "skin" and w4["Rs_model"] == "anomalous"
    assert w4["Rs_ohm"] == pytest.approx(1.474048838e-3, rel=1e-6, abs=0)  # R_s at the computed f (rel 1e-6 of F_TM010)
    assert out["300K"]["Q0"] == pytest.approx(G_TM010 / cv.skin_Rs(F_TM010, 1.68e-8), rel=2e-5, abs=0)
    assert out["300K"]["Q0"] == pytest.approx(3.462e4, rel=2e-4, abs=0)
    assert out["4K"]["Q0"] == pytest.approx(2.049e5, rel=2e-4, abs=0)
    assert out["4K"]["Q0"] == pytest.approx(G_TM010 / 1.474048838e-3, rel=2e-5, abs=0)


def test_rf_model_more_cases():
    # RRR accepted at 20 K, refused above
    rrr = {"model": "anomalous", "RRR": 100, "rho_300K_ohm_m": 1.725e-8, "rho_l_ohm_m2": RHOL_CU}
    assert cv.surface_resistance(dict(rrr, T_K=20), 1e9) == pytest.approx(
        cv.surface_resistance(dict(rrr, T_K=4), 1e9), rel=1e-15, abs=0)
    # anomalous + relaxation with 0 < p < 1 lies strictly between its exact ends
    vf = dict(CU_RRR100, v_F_m_s=1.57e6)
    r0, rp, r1 = (cv.surface_resistance(dict(vf, specularity=p), 10e9) for p in (0.0, 0.5, 1.0))
    assert r1 < rp < r0
    assert r0 == pytest.approx(cv.surface_resistance(CU_RRR100, 10e9) * (1 - 8.867e-3), rel=2e-5, abs=0)


def test_mqs_pickup_losses_with_anomalous(tmp_path):
    """The MQS pickup-loss path (R_eff = R_s(f) Gamma at scan.f_min / f_max) with an anomalous wall."""
    d = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                    mesh={"order": 2, "size": 0.08, "surface_size": 0.03})
    dd = af.Design(json.loads(json.dumps(d.doc)))
    dd.doc.setdefault("materials", {})["cu"] = {"kind": "conductor", "rf": CU_RRR100}
    for rg in dd.doc["regions"]:
        if rg["material"] == "copper":
            rg["material"] = "cu"
    fmin, fmax = 2e5, 2e7
    dd.doc["scan"] = dict(dd.doc.get("scan", {}), f_min=fmin, f_max=fmax)
    r = dd.solve(tmp_path / "mqs", verbose=0)
    surf = r.data["pickup"]["losses"]["surfaces"]
    assert surf
    for k, e in surf.items():
        for tag, f in (("f_min", fmin), ("f_max", fmax)):
            assert e["R_eff_ohm"][tag] == pytest.approx(cv.surface_resistance(CU_RRR100, f) * e["Gamma"], rel=1e-12, abs=0)


def _split_pillbox(rf, *, h=0.01, families=("TM",), modes=1):
    """Pillbox a = 0.1, L = 0.2 in a uniform axial field of 1 T with the barrel and the two end caps as separate wall
    bodies (same rf material)."""
    t = A / 10
    d = cv.pillbox_design(A, L, h=h, modes=modes, families=families, wall_material={"kind": "conductor", "rf": rf})
    d.doc["regions"] = [d.doc["regions"][0],
                        {"name": "barrel", "shape": {"rect": {"r": [A, A + t], "z": [-L / 2 - t, L / 2 + t]}},
                         "material": "wall_cu"},
                        {"name": "cap_top", "shape": {"rect": {"r": [0.0, A], "z": [L / 2, L / 2 + t]}},
                         "material": "wall_cu"},
                        {"name": "cap_bot", "shape": {"rect": {"r": [0.0, A], "z": [-L / 2 - t, -L / 2]}},
                         "material": "wall_cu"}]
    return d


def test_cavity_reports_bcs_and_flux_flow(tmp_path):
    """A superconducting-wall pillbox (bcs + flux_flow with an explicit B_T and a fixed orientation factor): the wall
    reports the model, the flux-flow part and the order-of-magnitude warning; Q = G / R_s with R_s = R_BCS + R_res +
    R_ff at the mode frequency."""
    rf = dict(NB, T_K=4.2, flux_flow=dict(FF, B_T=1.0, f_depin_Hz=5e9, orientation=1))
    d = cv.pillbox_design(A, L, h=0.01, modes=1, families=("TM",), wall_material={"kind": "conductor", "rf": rf})
    r = d.solve(tmp_path / "sc", verbose=0)
    m = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
    w = m["losses"]["walls"]["wall"]
    f = m["f_Hz"]
    rs, ff = cv.surface_resistance(rf, f, flux_flow_part=True)
    assert w["Rs_model"] == "bcs" and w["Rs_ohm"] == pytest.approx(rs, rel=1e-12, abs=0)
    assert w["Rs_flux_flow_ohm"] == pytest.approx(_vortex_R(f, 1.0, **_ff_args(dict(FF, f_depin_Hz=5e9))), rel=1e-12,
                                                  abs=0)
    assert w["Rs_flux_flow_ohm"] == pytest.approx(ff, rel=1e-12, abs=0)
    assert w["flux_flow_orientation_factor"] == pytest.approx(1.0, rel=1e-14, abs=0)
    assert m["losses"]["Q0"] == pytest.approx(w["G_ohm"] / rs, rel=1e-10, abs=0)
    assert any("flux_flow" in x and "order-of-magnitude" in x for x in r.data["warnings"])


def test_flux_flow_orientation_pillbox_axial_field(tmp_path):
    """In an axial B0, TM010 wall currents are axial on the barrel (parallel to the vortices: no Lorentz
    drive, orientation factor 0) and radial on the end caps (perpendicular: factor 1); TE011 currents are azimuthal
    (factor 1 everywhere). |B| is the local static field (B_T omitted) = the 1 T background."""
    rf = dict(NB, T_K=4.2, flux_flow=dict(FF))
    out = {}
    for fam in ("TM", "TE"):
        d = _split_pillbox(rf, families=(fam,))
        r = d.solve(tmp_path / fam, verbose=0)
        m = r.data["fullwave"]["eigenmodes"][fam]["modes"][0]
        out[fam] = (m, r.data["warnings"])
    m, warn = out["TM"]
    f = m["f_Hz"]
    assert f == pytest.approx(F_TM010, rel=1e-5, abs=0)
    full = _vortex_R(f, 1.0, **_ff_args(FF))
    walls = m["losses"]["walls"]
    assert walls["barrel"]["Rs_flux_flow_ohm"] == pytest.approx(0.0, abs=1e-12 * full)
    assert walls["barrel"]["flux_flow_orientation_factor"] == pytest.approx(0.0, abs=1e-12)
    assert walls["barrel"]["Rs_flux_flow_perpendicular_ohm"] == pytest.approx(full, rel=1e-9, abs=0)
    base = cv.surface_resistance(dict(NB, T_K=4.2), f)
    assert walls["barrel"]["Rs_ohm"] == pytest.approx(base, rel=1e-9, abs=0)
    for cap in ("cap_top", "cap_bot"):
        assert walls[cap]["Rs_flux_flow_ohm"] == pytest.approx(full, rel=1e-9, abs=0)
        assert walls[cap]["flux_flow_orientation_factor"] == pytest.approx(1.0, rel=1e-12, abs=0)
        assert walls[cap]["Rs_ohm"] == pytest.approx(base + full, rel=1e-9, abs=0)
    # Q0 from the per-wall R_s; the barrel carries most of the TM010 wall loss without flux flow (G_b smallest)
    inv_q = sum(1 / w["Q"] for w in walls.values())
    assert m["losses"]["Q0"] == pytest.approx(1 / inv_q, rel=1e-12, abs=0)
    assert any("sin^2(J_rf, B0)" in x for x in warn)
    m, _ = out["TE"]
    fte = m["f_Hz"]
    for w in m["losses"]["walls"].values():
        assert w["flux_flow_orientation_factor"] == pytest.approx(1.0, rel=1e-12, abs=0)
        assert w["Rs_flux_flow_ohm"] == pytest.approx(_vortex_R(fte, 1.0, **_ff_args(FF)), rel=1e-9, abs=0)


def test_cavity_warns_classical_skin_in_anomalous_regime(tmp_path):
    """A cavity mode evaluated with a classical-skin wall in the anomalous regime (here the builtin
    `copper_classical`, rho = 1.7e-10 ohm m, alpha ~ 600 at 1.15 GHz) warns; the anomalous model (including the
    builtin `copper`) and a 300 K skin wall do not."""
    def warns(wall):
        d = cv.pillbox_design(A, L, h=0.02, modes=1, families=("TM",))
        d.doc["regions"][1]["material"] = wall
        if isinstance(wall, dict):
            d.doc["materials"]["w"] = wall
            d.doc["regions"][1]["material"] = "w"
        r = d.solve(tmp_path / f"w{len(list(tmp_path.iterdir()))}", verbose=0)
        return [x for x in r.data["warnings"] if "anomalous-skin regime" in x]
    assert len(warns("copper_classical")) == 1
    assert warns("copper") == []
    assert warns({"kind": "conductor", "rf": CU_RRR100}) == []
    assert warns({"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.725e-8, "T_K": 300}}) == []


def test_cavity_warns_bcs_frequency_scaling(tmp_path):
    """Analytic BCS used far from f_ref without the log factor warns (TM010 at 1.15 GHz vs f_ref = 10 GHz);
    with log_factor it does not."""
    out = []
    for lf in (False, True):
        rf = dict(NB, f_ref=10e9, log_factor=lf)
        d = cv.pillbox_design(A, L, h=0.02, modes=1, families=("TM",), wall_material={"kind": "conductor", "rf": rf})
        out.append([x for x in d.solve(tmp_path / str(lf), verbose=0).data["warnings"] if "ln(2.246 kT/hf)" in x])
    assert len(out[0]) == 1 and out[1] == []


# the builtin `copper`: diffuse quasi-static RS with the RRR ~ 100 residual resistivity
CU_BUILTIN = {"model": "anomalous", "rho_ohm_m": 1.7e-10, "rho_l_ohm_m2": RHOL_CU, "specularity": 0, "T_K": 0.03}
CU_CLASSICAL = {"model": "skin", "rho_ohm_m": 1.7e-10, "T_K": 0.03}


def test_builtin_copper_is_anomalous_in_the_cavity(tmp_path):
    """A pillbox with builtin `copper` walls reports the anomalous model and R_s = RS(rho 1.7e-10, rho*l 6.6e-16,
    p = 0) at the TM010 frequency; `copper_classical` gives the classical sqrt(pi f mu0 rho) of the same rho. At
    1.15 GHz the classical value is 0.60x the anomalous one (docs/physics/surface_resistance.md Sec. 1.1), so the
    builtin (anomalous) copper gives a Q0 lower by that factor than copper_classical."""
    out = {}
    for mat in ("copper", "copper_classical"):
        d = cv.pillbox_design(A, L, h=0.02, modes=1, families=("TM",))
        d.doc["regions"][1]["material"] = mat
        m = d.solve(tmp_path / mat, verbose=0).data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
        out[mat] = (m["f_Hz"], m["losses"])
    f, lo = out["copper"]
    w = lo["walls"]["wall"]
    assert w["Rs_model"] == "anomalous"
    assert w["Rs_ohm"] == pytest.approx(cv.surface_resistance(CU_BUILTIN, f), rel=1e-12, abs=0)
    fc, lc = out["copper_classical"]
    assert lc["walls"]["wall"]["Rs_model"] == "skin"
    assert lc["walls"]["wall"]["Rs_ohm"] == pytest.approx(cv.skin_Rs(fc, 1.7e-10), rel=1e-12, abs=0)
    assert fc == pytest.approx(f, rel=1e-12, abs=0)
    ratio = lo["Q0"] / lc["Q0"]
    assert ratio == pytest.approx(cv.skin_Rs(f, 1.7e-10) / w["Rs_ohm"], rel=1e-10, abs=0)
    assert 0.59 < ratio < 0.62


def test_builtin_copper_reduces_to_classical_at_low_frequency():
    """alpha = (3/4) mu0 omega l^2 / rho -> 0: RS -> classical skin. Ratios from the exact diffuse RS integral
    (docs/physics/surface_resistance.md Sec. 1.1 table): R_skin/R_RS = 0.996 at 0.12 MHz, 0.976 at 1 MHz, 0.906 at
    10 MHz, 0.770 at 100 MHz, 0.604 at 1 GHz."""
    r = lambda f: cv.skin_Rs(f, 1.7e-10) / cv.surface_resistance(CU_BUILTIN, f)  # noqa: E731
    assert r(10.0) == pytest.approx(1.0, rel=2e-5, abs=0)
    assert r(1e3) == pytest.approx(1.0, rel=1e-3, abs=0)
    for f, want in ((1.209e5, 0.996), (1e6, 0.976), (1e7, 0.906), (1e8, 0.770), (1e9, 0.604)):
        assert r(f) == pytest.approx(want, abs=6e-4), f
    assert cv.surface_resistance(CU_CLASSICAL, 1e9) == pytest.approx(cv.skin_Rs(1e9, 1.7e-10), rel=1e-14, abs=0)


def test_builtin_copper_mqs_pickup_losses(tmp_path):
    """The MQS pickup-loss report of the solenoid template (builtin copper coax) uses the anomalous R_s at the band
    edges: R_eff = R_s,RS(f) Gamma. Default band (120.9 kHz - 1 MHz): +0.4 % / +2.4 % vs the old classical value."""
    d = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                    mesh={"order": 2, "size": 0.08, "surface_size": 0.03})
    fmin, fmax = 120.9e3, 1e6  # the default scan band (spec.hpp), set explicitly here
    d.doc["scan"] = dict(d.doc.get("scan", {}), f_min=fmin, f_max=fmax)
    r = d.solve(tmp_path / "mqs", verbose=0)
    surf = r.data["pickup"]["losses"]["surfaces"]
    assert surf
    for e in surf.values():
        for tag, f in (("f_min", fmin), ("f_max", fmax)):
            assert e["R_eff_ohm"][tag] == pytest.approx(cv.surface_resistance(CU_BUILTIN, f) * e["Gamma"], rel=1e-12,
                                                        abs=0)
