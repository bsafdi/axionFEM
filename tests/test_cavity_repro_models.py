"""Fast checks of the reference models and helpers used by the published-cavity reproduction scripts
(benchmarks/cavity/repro, docs/validation/cavity_reproductions.md); no FEM solves. The cavity module is beta."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parents[1] / "benchmarks" / "cavity" / "repro"
REFS = Path(__file__).resolve().parents[1] / "benchmarks" / "cavity" / "refs"


def _load(name):
    sys.path.insert(0, str(REPRO))
    try:
        spec = importlib.util.spec_from_file_location(name, REPRO / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(REPRO))


def test_capp_sector_closed_form_matches_refs_and_paper():
    pytest.importorskip("scipy")
    capp = _load("capp_multicell_3d")
    ref = json.loads((REFS / "capp_multicell_1710.json").read_text())
    rows = {r["N_cells"]: r for r in ref["reference_values"]["analytic_check"]["rows"]}
    tab2 = {r[0]: r[2] for r in ref["reference_values"]["tables"][1]["rows"]}
    for N in (2, 4, 6, 8):
        cf = capp.closed_form(N, 5.96e7)
        assert cf["Q_no_caps"] == pytest.approx(rows[N]["Q_no_endcaps"], rel=1e-5)
        assert cf["C"] == pytest.approx(rows[N]["C"], abs=1e-5)
        assert cf["P_rad_over_P_azi"] == pytest.approx(rows[N]["Prad_over_Pazi"], abs=1e-4)
        # the paper's analytical column used sigma = 5.998e7 (COMSOL copper): digit for digit
        assert round(capp.closed_form(N, 5.998e7)["Q_no_caps"]) == tab2[N]


def test_layered_model_matches_the_independent_derivation_with_flat_ends():
    pytest.importorskip("scipy")
    q = _load("quax_2201_offset")
    MM = 1e-3
    lay = [(10.7 * MM, 1.0), (12.6 * MM, 11.2), (19.8 * MM, 1.0), (21.4 * MM, 11.2), (29 * MM, 1.0)]
    best = None
    for f in q.roots(lay, 29 * MM, 10.5e9, 11.3e9, 400):
        pr = q.props(f, lay, 29 * MM)
        if best is None or pr["C_flat"] > best[1]["C_flat"]:
            best = (f, pr)
    # an independent (blind) derivation of the same layered flat-ended cavity
    assert best[0] == pytest.approx(10.8825727063e9, rel=1e-8)
    assert best[1]["C_flat"] == pytest.approx(0.0409835344281, rel=1e-6)
    assert best[1]["p_e"] == pytest.approx(0.167773168717, rel=1e-6)


def test_rades_layouts_and_rounded_box():
    r5 = _load("rades_5cav_1803")
    cav, iris, tot = r5.layout()
    assert tot == pytest.approx(136.36e-3)
    assert len(cav) == 5 and len(iris) == 4
    assert cav[0][0] == pytest.approx(-tot / 2) and cav[-1][1] == pytest.approx(tot / 2)
    v_cav, v_ap = r5.volumes()
    assert v_cav == pytest.approx(22.86e-3 * 10.16e-3 * (2 * 26.68e-3 + 3 * 25e-3))
    assert v_ap == pytest.approx(4 * 8e-3 * 2e-3 * 10.16e-3)
    cm = _load("_common")
    rb = cm.rounded_box_xz(0, 1, 0, 2, 0, 3, 0.25)
    assert len(rb["union"]) == 6
    assert cm.rounded_box_xz(0, 1, 0, 2, 0, 3, 0.0) == {"box": {"min": [0, 0, 0], "max": [1, 2, 3]}}
    r6 = _load("rades_6cav_alt_2002")
    lay, tot6 = r6.layout()
    assert tot6 == pytest.approx(181.3e-3)
    assert [k for k, _, _ in lay].count("cap") == 3 and [k for k, _, _ in lay].count("ind") == 2
    # z-mirror symmetric about the central capacitive iris
    mid = lay[5]
    assert mid[0] == "cap" and mid[1] == pytest.approx(-mid[2])
    op = r6.iris_openings("cap", 0.0, 1.0, variant="U_mid_up")
    assert len(op) == 3 and op[1]["box"]["max"][1] == pytest.approx(r6.B / 2)


def test_birme_port_impedance():
    b = _load("birme_probe_port")
    assert b.ZC == pytest.approx(49.92, abs=0.02)
    f, Q = b.closed(0)
    assert f == pytest.approx(299792458.0 * 2.404825557695773 / (2 * math.pi * 0.044835))


def test_rades_helpers_k_and_pattern():
    r6 = _load("rades_6cav_alt_2002")
    assert r6.k_from(1.01, 1.0) == pytest.approx((1.01 ** 2 - 1) / (1.01 ** 2 + 1))
    assert r6.k_from(1.0, 1.01) < 0
    r5 = _load("rades_5cav_1803")
    # 1/8 model, z-even mode: probes in pockets 3, 4, 5 (z >= 0) mirrored to 1, 2
    mode = {"probes": {"E": [[0, 1.0, 0], [0, 0.5, 0], [0, -0.7, 0]]}}
    assert r5.pattern(mode, False, "pmc") == "- + + + -"
    mode = {"probes": {"E": [[0, 0.0, 0], [0, 0.6, 0], [0, 0.8, 0]]}}
    assert r5.pattern(mode, False, "pec") == "+ + 0 - -"  # odd mode, normalised so that pocket 2 is +
