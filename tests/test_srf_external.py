"""External SRF benchmarks (docs/srf/README.md, "External benchmarks"; benchmarks/srf/external/). Beta.

Fast regression of the checks against published numbers: Lasenby 1912.11056 Table IV from the TE012/TM013 closed
forms, the axionFEM bcs model against the digitized Aune et al. (2000) Fig. 1, the first-order Slater coefficients
(against an independent analytic derivation), and the TESLA mid cell against Juntong & Jones
0910.1305 Table 1 (coarse 2D solve).
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/srf/external"))


def test_lasenby_table_iv_from_closed_forms():
    import lasenby
    from axionfem import srf
    cp = srf.cylinder_pair("TE012/TM013")
    assert cp["L_over_R"] == pytest.approx(2.35, abs=0.005)          # "degenerate at d/R ~ 2.35"
    assert cp["eta_10"] ** 2 == pytest.approx(0.19, abs=0.005)       # "C01 = 0.19"
    rows = lasenby.table_rows(cp)
    # Lasenby rounds to 2 significant figures: compare at the 5 % level
    for k, tol in (("U_J", 0.05), ("C01U_J", 0.05), ("Q0", 0.05), ("Q1", 0.05), ("Brms_T", 0.05)):
        assert rows["60L"]["ratio"][k] == pytest.approx(1, abs=tol), k
    assert rows["60L"]["ours"]["f_Hz"] == pytest.approx(1.1e9, rel=0.02)
    assert rows["900L"]["ours"]["f_Hz"] == pytest.approx(440e6, rel=0.02)
    assert lasenby.eq24_coefficient(cp["L_over_R"]) == pytest.approx(0.09, abs=0.005)


def test_bcs_vs_aune_fig1():
    import rs_check
    a = rs_check.part_a()
    fit = a["fits"]["R_res"]
    assert 2.0 < fit["R_res_nOhm"] < 4.0                              # the paper's line: 3 nOhm
    assert fit["median_data_over_model_Tc/T<5"] == pytest.approx(1.13, abs=0.05)   # model low by ~13 %
    assert a["paper_line"]["model_slope_same_x"] == pytest.approx(a["paper_line"]["slope_dlnR_dx"], rel=0.06)


def test_slater_coefficients_match_independent_derivation():
    import slater_ellipse as s
    p = s.predictions()
    # independent derivation: 0.8919, 0.6735, 0.3208, 4.641
    assert p["TM111"]["c_split"] == pytest.approx(0.8919, abs=1e-4)
    assert p["TM112"]["c_split"] == pytest.approx(0.6735, abs=1e-4)
    assert p["TE113"]["c_split"] == pytest.approx(0.3208, abs=1e-4)
    assert p["TE012_Er"]["Er_over_cBwallmax_per_e"] == pytest.approx(4.641, abs=1e-3)


def test_tesla_mid_cell_coarse(tmp_path):
    import tesla
    x = tesla.run_cell("TESLA", tesla.TESLA["mid"], 20, 2, str(tmp_path))
    ref = tesla.REF_CELL["TESLA"]
    assert x["f_pi_Hz"] / 1e9 == pytest.approx(ref["f_pi_GHz"], rel=5e-4)
    assert x["kc_pct"] == pytest.approx(ref["kc_pct"], abs=0.02)
    assert x["Bpk_over_Eacc_mT_per_MVm"] == pytest.approx(ref["Bpk_Eacc_mT_MVm"], rel=0.02)
    assert x["G_ohm"] == pytest.approx(270, rel=0.01)                   # Aune Table 2 (9-cell) 270 Ohm
    assert x["Epk_over_Eacc"] == pytest.approx(2.0, rel=0.02)           # Aune Table 2: 2.0
    assert math.isfinite(x["R_over_Q_linac_ohm"])


RES = ROOT / "benchmarks/srf/results"


def _json(name, *keys):
    """The record, or the nested entry keys[0][keys[1]]...; skip when the file or a key is missing."""
    import json
    f = RES / name
    if not f.exists():
        pytest.skip(f"{name} not generated")
    d = json.loads(f.read_text())
    for k in keys:
        if k not in d:
            pytest.skip(f"{name}: no {'/'.join(keys)}")
        d = d[k]
    return d


def _row(rows, key, val, tol=1e-12):
    hit = [r for r in rows if r.get(key) is not None and abs(r[key] - val) <= tol * max(1.0, abs(val))]
    if not hit:
        pytest.skip(f"no row with {key} = {val}")
    return hit[0]


def test_recorded_3d_slater_matches_first_order():
    """The committed 3D record (h = R/10, order 3) against first-order theory, and the 3D barrel E_r against
    Lasenby Eq. 23 (40 kV/m at f = 1e-3, max wall |B| = 0.2 T)."""
    import slater_ellipse as s
    from axionfem.cavity import C_LIGHT
    d = _json("external_slater.json", "n10_o3")
    p = s.predictions()
    for row in d["rows"]:
        e = row["e"]
        if not e or e > 0.01:
            continue
        for name, x in row["doublets"].items():
            assert x["c_measured"] == pytest.approx(p[name]["c_split"], rel=2e-4), (name, e)
        assert row["TE012"]["Er_over_cBbarrel_per_e"] == pytest.approx(p["TE012_Er"]["Er_over_cBbarrel_per_e"], rel=2e-3)
        assert all(t["E_phi_fraction"] < 1e-3 for t in row["TE012"]["TM013_like"])     # TM013 unmixed
    r = _row(d["rows"], "e", 0.0025)
    wallmax_over_barrel = p["TE012_Er"]["cB_cap_max_over_A"] / p["TE012_Er"]["cB_barrel_max_over_A"]
    Er_kVm = r["TE012"]["Er_over_cBbarrel"] / wallmax_over_barrel * C_LIGHT * 0.2 * (1e-3 / r["f_flattening"]) / 1e3
    assert Er_kVm == pytest.approx(139, rel=0.01)
    assert Er_kVm / 40 == pytest.approx(3.48, rel=0.01)


def test_recorded_tesla_cell():
    d = _json("external_tesla.json", "cell")
    t, ll = _row(d["TESLA"]["rows"], "h_over_Req", 1 / 80), _row(d["LL"]["rows"], "h_over_Req", 1 / 80)
    assert t["f_pi_Hz"] / 1e9 == pytest.approx(1.30116, rel=3e-4)
    assert t["kc_pct"] == pytest.approx(1.89, abs=0.01)
    assert t["Bpk_over_Eacc_mT_per_MVm"] == pytest.approx(4.18, rel=0.01)
    assert t["Epk_over_Eacc"] == pytest.approx(2.0, rel=0.015)
    assert ll["f_pi_Hz"] / 1e9 == pytest.approx(1.30008, rel=3e-4)
    assert ll["kc_pct"] == pytest.approx(1.53, abs=0.01)
    assert ll["Bpk_over_Eacc_mT_per_MVm"] == pytest.approx(3.64, rel=0.015)
    n9 = _row(_json("external_tesla.json", "ninecell", "tube0.2_o3")["rows"], "h_over_Req", 1 / 40)
    assert n9["R_over_Q_aune_ohm"] == pytest.approx(518, rel=0.04)
    assert n9["G_ohm"] == pytest.approx(270, rel=0.01)
