"""Fast regression checks of the published-benchmark designs.

Each design in ``benchmarks/published/`` is solved on a deliberately coarse mesh
and its Psi and L_p compared with the stored values in
``tests/reference/published_coarse.json``.  These are *regression* tolerances
(1e-4 relative), not physics tolerances: the physics comparison with the
published numbers lives in ``benchmarks/published_benchmarks.py`` and
``docs/validation/published_benchmarks.md``.

Regenerate the reference after an intentional change:

    PYTHONPATH=python AXF_MESHER=build/axf-mesher python tests/test_published.py --update

Add a single new design without disturbing the stored values of the others:

    ... python tests/test_published.py --add dmradio_core_pickup_146cm
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# the source tree only as a fallback: an installed (non-editable) axionfem must not be shadowed by python/axionfem,
# which holds no _core in that case
sys.path.append(str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import axionfem as af  # noqa: E402

MU0 = 1.25663706212e-6
DES = ROOT / "benchmarks" / "published"
REF = ROOT / "tests" / "reference" / "published_coarse.json"
RUNS = ROOT / "runs" / "test_published"
RTOL = 1e-4

# design name -> coarse mesh overrides {order, size, surface_size}
COARSE = {
    "abra10cm": {"order": 2, "size": 0.010, "surface_size": 5e-4},
    "abra10cm_shielded": {"order": 2, "size": 0.010, "surface_size": 5e-4},
    "dmradio_50l": {"order": 2, "size": 0.020},
    "dmradio_gut": {"order": 2, "size": 0.12},
    "dmradio_m3": {"order": 2, "size": 0.08},
    "dmradio_m3_h0p56": {"order": 2, "size": 0.08},
    "dmradio_core_magnet": {"order": 2, "size": 0.08},
    "dmradio_core_pickup_120cm": {"order": 2, "size": 0.08},
    # both published Core pickup diameters are tested: CORE L509 states four pickups at
    # 120 cm and four at 146 cm, and the report compares both, so neither may be untested
    "dmradio_core_pickup_146cm": {"order": 2, "size": 0.08},
}


def coarse_design(name: str) -> af.Design:
    d = af.load(DES / f"{name}.json")
    d.doc.setdefault("mesh", {}).update(COARSE[name])
    return d


def solve(name: str):
    return coarse_design(name).solve(RUNS / name, verbose=0)


def reference() -> dict:
    if not REF.exists():
        pytest.skip(f"{REF.relative_to(ROOT)} missing; regenerate with "
                    "'python tests/test_published.py --update'")
    return json.loads(REF.read_text())


# ------------------------------------------------------------------ designs
def test_sources_covers_every_design():
    import re
    src = json.loads((DES / "sources.json").read_text())
    names = {p.stem for p in DES.glob("*.json")} - {"sources"}
    assert names, "no published designs found"
    for name in names:
        key = re.sub(r"(_3d_port(?:30|360)|_reduced_120cm|_shielded|_magnet|_pickup_\d+cm|_h\d+p?\d*)$", "", name)
        key = re.sub(r"(_3d_port(?:30|360)|_reduced_120cm)$", "", key)
        assert key in src, f"{name}.json has no entry '{key}' in sources.json"


@pytest.mark.parametrize("name", sorted(COARSE))
def test_design_validates(name):
    af.load(DES / f"{name}.json").validate()


# ------------------------------------------------- unit-conversion identities
def test_c_PU_identity():
    """c_PU = F^(1/4)/(sqrt2 B0 V^(5/6)) must invert F = 4 c_PU^4 B0^4 V^(10/3)."""
    from published_benchmarks import F_from_c_PU, c_PU_from_F
    for c, B0, V in ((0.1, 16.0, 10.0), (0.16, 4.0, 2.0), (0.0173, 1.03, 1.018e-3)):
        F = F_from_c_PU(c, B0, V)
        assert c_PU_from_F(F, B0, V) == pytest.approx(c, rel=1e-12)


def test_V_B_identity_against_KST_eq8():
    """KST Eq. (8) is our reciprocity integral: check at one small geometry."""
    from published_benchmarks import Psi_loop_in_toroid, V_B_KST_eq8
    args = (0.02, 0.03, 0.03, 0.12)
    assert V_B_KST_eq8(*args) == pytest.approx(Psi_loop_in_toroid(*args), rel=1e-7)


def test_thin_ring_inductance_formula():
    from published_benchmarks import L_thin_ring
    assert L_thin_ring(0.02, 5e-4) == pytest.approx(MU0 * 0.02 * (math.log(320.0) - 2), rel=1e-12)


def test_core3d_port_designs_validate():
    """The 360-degree control must be an explicit increasing full revolution.

    In particular, an implicit angle range must not resolve to a zero-width
    [180, 180] range in the 3D validator; the 30-degree split and the 360-degree
    control are both generated here so the conversion cannot regress silently.
    """
    from published_benchmarks import design_core_3d_port, design_core_reduced

    d2 = design_core_reduced()
    for deg in (30.0, 360.0):
        d3 = design_core_3d_port(d2, port_deg=deg)
        d3.validate()


# ------------------------------------------------------------------ regression
@pytest.mark.parametrize("name", sorted(COARSE))
def test_coupling_regression(name):
    ref = reference()
    assert name in ref, f"no reference for {name}; regenerate with --update"
    r, e = solve(name), ref[name]
    got = {"Psi_natural_T_m3": r.coupling.get("Psi_natural_T_m3", 0.0),
           "L_p_H": r.coupling.get("L_p_H", 0.0),
           "W_mag_J": r.coupling.get("W_mag_J", 0.0),
           "B_max_T": r.B_max}
    for k, want in e.items():
        if k == "dofs":
            assert got.get(k, r.data["mesh"]["dofs"]) == want or r.data["mesh"]["dofs"] == want, \
                f"{name}: DOFs {r.data['mesh']['dofs']} != {want} (mesh changed)"
            continue
        if want == 0.0:
            assert abs(got[k]) < 1e-30, f"{name}.{k}: expected 0, got {got[k]}"
        else:
            assert got[k] == pytest.approx(want, rel=RTOL), \
                f"{name}.{k}: {got[k]:.10g} vs {want:.10g} ({(got[k] / want - 1) * 100:+.4f} %)"


def test_abra10cm_matches_the_exact_reciprocity_integral():
    """The physics check that is still fast: the coarse free-space ABRA-10cm V_B
    is within 5 % of the exact free-space integral (the finite winding sheet and
    the coarse mesh account for the difference; see the report for convergence)."""
    from published_benchmarks import Psi_loop_in_toroid
    src = json.loads((DES / "sources.json").read_text())["abra10cm"]["geometry"]
    Rin, Rout, H = src["R_in_m"][0], src["R_out_m"][0], src["H_m"][0]
    NI = src["turns"][0] * src["current_A"][0]
    B_in = MU0 * NI / (2 * math.pi * Rin)
    exact = Psi_loop_in_toroid(src["pickup_loop_radius_m"][0], Rin, Rout - Rin, H)
    r = solve("abra10cm")
    assert r.coupling["Psi_natural_T_m3"] / B_in == pytest.approx(exact, rel=0.05)


def test_dmradio_m3_L_p_matches_the_published_coax_formula():
    """L_p of the published m^3 coax against DMRadio's own L = Z_c h_coax / c
    (arXiv:2302.14084 L385-388), for the tallest and the shortest pickup.  The
    FEM sits a little below because of the readout gap and the open end."""
    src = json.loads((DES / "sources.json").read_text())["dmradio_m3"]
    g = src["geometry"]
    Zc = 376.730313412 / (2 * math.pi) * math.log(g["coax_outer_radius_m"][0]
                                                  / g["coax_inner_radius_m"][0])
    assert Zc == pytest.approx(src["em_model"]["Z_c_ohm"][0], rel=2e-3)
    for name, h in (("dmradio_m3", g["coax_heights_m"][0][0]),
                    ("dmradio_m3_h0p56", g["coax_heights_m"][0][-1])):
        want = Zc * h / 299792458.0
        got = solve(name).L_p
        assert got == pytest.approx(want, rel=0.04), f"{name}: {got * 1e9:.2f} vs {want * 1e9:.2f} nH"
        assert got < want, f"{name}: the FEM should sit below the lossless-line formula"


# ---------------------------------------------- the published quantities and the records' claims
REC = ROOT / "benchmarks" / "records"


def _rec(name: str) -> dict:
    p = REC / f"published_{name}.json"
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} missing; run benchmarks/published_benchmarks.py {name}")
    return json.loads(p.read_text())


def test_abra10cm_published_V_B_uses_the_published_field_volume():
    """The published V_B is 0.027 x 890 cm^3 = 24.03 cm^3, NOT 0.027 x the 1017.9 cm^3
    rectangular annulus (27.5 cm^3).  Comparing against the latter is what produced the
    spurious '-22 %' headline."""
    pub = json.loads((DES / "sources.json").read_text())["abra10cm"]["published_quantities"]
    G, V = pub["G_V"][0], pub["V_field_m3"][0]
    assert V == pytest.approx(890e-6, rel=1e-12), "the published field volume is 890 cm^3"
    assert pub["V_B_m3"][0] == pytest.approx(G * V, rel=2e-4)
    V_rect = math.pi * (0.06 ** 2 - 0.03 ** 2) * 0.12
    assert pub["V_B_m3"][0] != pytest.approx(G * V_rect, rel=0.1), "must not be the rectangle"
    assert pub["L_p_H"][0] == pytest.approx(95.5e-9, rel=1e-12), "95.5 nH, not the PRL's ~100 nH"


def test_abra10cm_headline_is_the_finest_mesh_and_the_tw_zero_limit():
    """The headline must not be chosen by min|rel_to_exact| (that picks the
    COARSEST mesh here), and the quoted G must be the t_w -> 0 value, not the FEM value
    at the modelled 0.2 mm winding sheet, which is biased by about +0.5 %."""
    r = _rec("abra10cm")
    rows = r["mesh_convergence"]
    finest = max(rows, key=lambda x: x["dofs"])
    closest = min(rows, key=lambda x: abs(x["rel_to_exact"]))
    assert r["headline_row"]["dofs"] == finest["dofs"], "the headline row is the most refined one"
    assert closest["dofs"] < finest["dofs"], (
        "the row closest to the semi-analytic value is not the finest; that is exactly why "
        "min|rel_to_exact| must not be used to pick it")
    gf, sa = r["geometric_factor"], r["semi_analytic"]
    V_pub = r["inputs"]["V_field_published_m3"]
    assert gf["G_V_tw_zero"] == pytest.approx(sa["V_B_eq8_m3"] / V_pub, rel=1e-12)
    # the FEM row sits above the t_w -> 0 limit by the winding-sheet bias, ~ +0.5 %
    assert gf["tw_bias_of_FEM_headline_row"] == pytest.approx(0.0049, abs=5e-4)
    assert gf["G_V_FEM"] > gf["G_V_tw_zero"]
    # and the headline disagreement is ~10 %, not the old 22 %
    assert gf["V_B_tw_zero_rel_to_published"] == pytest.approx(-0.101, abs=0.01)


def test_abra10cm_Psi_row_deviation_matches_its_own_numbers():
    """The report's Psi row must carry its own deviation, not the G_V one: the numbers printed beside it
    give a different value.  Psi = B_max V_B, and our B_max comes
    from N I while theirs is the quoted 1 T, so the two deviations differ by that factor."""
    r = _rec("abra10cm")
    gf, i = r["geometric_factor"], r["inputs"]
    implied = gf["Psi_natural_T_m3_ours"] / gf["Psi_natural_T_m3_published"] - 1
    assert gf["Psi_rel_to_published"] == pytest.approx(implied, rel=1e-9)
    assert gf["Psi_rel_to_published"] != pytest.approx(gf["V_B_rel_to_published"], abs=1e-3)
    assert (1 + gf["V_B_rel_to_published"]) * gf["Psi_B_max_convention_factor"] == \
        pytest.approx(1 + gf["Psi_rel_to_published"], rel=1e-9)
    assert gf["Psi_B_max_convention_factor"] == pytest.approx(i["B_in_from_NI_T"] / i["B_in_quoted_T"],
                                                             rel=1e-12)


def test_abra10cm_a_890cm3_field_box_does_not_close_the_gap():
    """Shrinking the field region to the published 890 cm^3 makes V_B worse
    (about -14 %), so the 890 cm^3 cannot explain the published G_V = 0.027."""
    r = _rec("abra10cm")
    rows = r["field_volume_variants"]["rows"]
    assert rows, "no 890 cm^3 variants in the record"
    for row in rows:
        assert row["V_field_m3"] == pytest.approx(890e-6, rel=1e-6)
        assert row["rel_to_published_V_B"] < r["geometric_factor"]["V_B_tw_zero_rel_to_published"]


def test_dmradio_gut_table2_over_eq3_uses_the_papers_own_DFSZ_convention():
    """DMRadio-GUT fixes C_agammagamma ~ 0.75, i.e. C_gamma = 1.92.  With that the
    Table II / Eq. (3) discrepancy is x1.60, not the x3.34 the lattice C_gamma = 1.77
    gives.  t ~ g^-4 ~ |E/N - C_gamma|^-4 with E/N = 8/3 fixes the factor exactly."""
    r = _rec("dmradio_gut")
    st = r["scan_time"]
    C_agg = st["C_agammagamma_published"]
    assert C_agg == pytest.approx(0.75, abs=0.01)
    assert st["C_QCD_implied_by_published_C_agammagamma"] == pytest.approx(1.92, abs=0.01)
    want = ((8 / 3 - 1.77) / (8 / 3 - 1.92)) ** 4
    assert st["Cgamma_convention_factor_in_time"] == pytest.approx(want, rel=1e-12)
    for sc in st["scenarios"]:
        assert sc["scan_time_from_published_c_PU_Cgamma1.92_yr"] / \
            sc["scan_time_from_published_c_PU_yr"] == pytest.approx(want, rel=2e-3)
        assert sc["published_over_eq3_with_published_c_PU_Cgamma1.92"] == pytest.approx(
            sc["scan_time_published_yr"] / sc["scan_time_from_published_c_PU_Cgamma1.92_yr"], rel=1e-9)
    sc0 = st["scenarios"][0]
    assert sc0["scan_time_from_published_c_PU_Cgamma1.92_yr"] == pytest.approx(3.86, abs=0.05)
    assert sc0["published_over_eq3_with_published_c_PU_Cgamma1.92"] == pytest.approx(1.60, abs=0.02)
    assert r["comparison"]["table2_over_eq3"] == pytest.approx(1.60, abs=0.02)


def test_dmradio_gut_stored_energy_check_is_not_the_circular_one():
    """573 MJ is (1/2 mu0) B_rms^2 V, i.e. a definition, and the aspect ratio
    was chosen to reproduce it; W also moves across it with the unpublished winding
    thickness.  The self-check must therefore test W against the analytic ideal toroid."""
    r = _rec("dmradio_gut")
    c = r["comparison"]
    src = json.loads((DES / "sources.json").read_text())["dmradio_gut"]["geometry"]
    assert c["W_published_is_a_definition"] == pytest.approx(
        0.5 * src["B_rms_T"][0] ** 2 * src["V_m3"][0] / MU0, rel=1e-12)
    assert c["W_published_is_a_definition"] == pytest.approx(c["W_published_J"], rel=1e-3)
    assert abs(c["W_rel_to_analytic_ideal"]) < 0.05
    tw = r["winding_thickness_scan"]
    lo, hi = tw["W_range_J"]
    assert lo < c["W_published_J"] < hi, (
        "the published 573 MJ must be shown to lie inside the spread of the unpublished t_w")
    assert r["pass"] is True and "circular" in r["pass_scope"].lower()


def test_dmradio_gut_c_PU_is_bracketed_by_the_unpublished_height_ratio():
    """c_PU runs over 0.068-0.098 with H/R_in = 1-8, so the '-16 %' against
    the published 'typical 0.1' is set by a parameter DMRadio-GUT does not publish."""
    r = _rec("dmradio_gut")
    hr = r["height_ratio_scan"]
    lo, hi = hr["c_PU_range"]
    assert lo == pytest.approx(0.068, abs=0.004) and hi == pytest.approx(0.098, abs=0.004)
    assert lo < r["comparison"]["c_PU_ours"] < hi
    # W is independent of H/R_in at fixed V and aspect ratio, c_PU is not
    wlo, whi = hr["W_range_J"]
    assert whi / wlo - 1 < 0.02, "W must be (nearly) independent of H/R_in"
    assert hi / lo > 1.2, "c_PU must not be"
    for row in hr["rows"]:
        assert row["V_m3"] == pytest.approx(10.0, rel=1e-6)
        assert row["B_max_T"] == pytest.approx(16.0, rel=1e-6)


def test_every_record_says_what_its_pass_flag_covers():
    """'[PASS]' tests internal identities, not agreement with the publication.
    Every record must carry that distinction explicitly."""
    names = [p.stem.replace("published_", "") for p in sorted(REC.glob("published_*.json"))]
    assert names, "no published records found"
    for name in names:
        r = _rec(name)
        assert "pass_means" in r, f"{name}: no pass_means"
        assert "SELF-CHECK" in r["pass_means"]
        assert "pass_scope" in r, f"{name}: no pass_scope naming what the flag covers"
        assert len(r["pass_scope"]) > 40


def test_the_report_text_matches_the_records_it_is_generated_from():
    """The report is generated, never hand-edited, so the statements checked above must
    come back out of the generator.  Guards the headline G, the Psi row label, the
    GUT x1.60 and the DMRadio-50L check count."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from published_report import _sec_abra, _sec_50l, _sec_gut   # noqa: PLC0415
    src = json.loads((DES / "sources.json").read_text())
    a = _rec("abra10cm")
    txt = _sec_abra(a, src)
    gf = a["geometric_factor"]
    assert f"{gf['G_V_tw_zero']:.5g}" in txt, "the t_w -> 0 G_V must be quoted"
    assert f"{gf['Psi_rel_to_published'] * 100:+.2f} %" in txt, "the Psi row must carry its own deviation"
    assert "-22 %" not in txt and "−22 %" not in txt, "the -22 % headline must be gone"
    t50 = _sec_50l(_rec("dmradio_50l"), src)
    assert "**two** independent consistency checks" in t50
    assert "Check 3" not in t50, "the volume is not a third independent check"
    gu = _rec("dmradio_gut")
    tg = _sec_gut(gu, src)
    f160 = gu["scan_time"]["scenarios"][0]["published_over_eq3_with_published_c_PU_Cgamma1.92"]
    assert f"×{f160:.2f}" in tg, "the Table II / Eq. (3) factor must be the C_gamma = 1.92 one"
    assert f"×{gu['scan_time']['scenarios'][0]['published_over_eq3_with_published_c_PU']:.2f}" in tg, \
        "the lattice-C_gamma factor must be shown, labelled as the artefact"
    assert "circular" in tg
    assert "is a real, independent check and it passes" not in tg, "the W 'agreement' claim must be gone"


def _row(name: str) -> dict:
    r = solve(name)
    row = {"Psi_natural_T_m3": r.coupling.get("Psi_natural_T_m3", 0.0),
           "L_p_H": r.coupling.get("L_p_H", 0.0),
           "W_mag_J": r.coupling.get("W_mag_J", 0.0),
           "B_max_T": r.B_max, "dofs": r.data["mesh"]["dofs"]}
    print(f"  {name}: Psi = {row['Psi_natural_T_m3']:.8g} T m^3, "
          f"L_p = {row['L_p_H'] * 1e9:.6g} nH, dofs = {row['dofs']}", flush=True)
    return row


def _write(out: dict) -> None:
    REF.parent.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps({"_about": "Stored coarse-mesh values for tests/test_published.py; "
                                         "regenerate with 'python tests/test_published.py --update', "
                                         "or add one design with '--add <name>'",
                               "_axionfem_version": af.version(), **out}, indent=2))
    print(f"wrote {REF.relative_to(ROOT)}")


def _update() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    _write({name: _row(name) for name in sorted(COARSE)})


def _add(names: list[str]) -> None:
    """Store one or more designs without touching the others' stored values."""
    RUNS.mkdir(parents=True, exist_ok=True)
    cur = json.loads(REF.read_text()) if REF.exists() else {}
    rows = {k: v for k, v in cur.items() if not k.startswith("_")}
    for name in names:
        if name not in COARSE:
            raise SystemExit(f"{name} is not in COARSE; add its mesh overrides first")
        rows[name] = _row(name)
    _write(dict(sorted(rows.items())))


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    elif "--add" in sys.argv:
        _add(sys.argv[sys.argv.index("--add") + 1:])
    else:
        raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
