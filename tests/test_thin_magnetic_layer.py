"""Thin magnetic layers on curves (docs/physics/thin_magnetic_layer.md, validation
docs/validation/thin_magnetic_layer.md, driver benchmarks/thin_magnetic_layer.py).

Checks: spec errors and located refusals; a spherical shell in a uniform background against the
model's closed form and the exact thick shell; the model's second-order error law; the background
drive of a linear region and the reaction energy;
layers meeting superconductors (a coated ball in both region orders; a superconducting sheet ending
on or crossing a can wall) and magnetic regions (a coated linear core; a thin wall ending on meshed
caps, refused); the pickup problem against a meshed wall; the definiteness guard and the
order-aware warning; adjoint shape sensitivities; junctions of layers with sheets, superconducting regions,
pickups and other layers.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import axionfem as af
from axionfem._paths import axf_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import thin_magnetic_layer as tml  # noqa: E402

R = 1.0
MU_R = 1000.0
D = 0.01


def solve(doc, tmp_path, name):
    return af.Design(doc).solve(workdir=str(tmp_path / name), verbose=0)


# ---------------------------------------------------------------- spec


IRON = {"kind": "iron", "bh": "knee", "mu_r": 1000, "B_sat": 2}


def base_doc():
    return tml.sphere_layer(R, D, MU_R, 0.2)


@pytest.mark.parametrize("change, match", [
    (lambda d: d["curves"][0].update(thickness=-1.0), "thickness must be a positive length"),
    (lambda d: d["curves"][0].pop("thickness"), "has no thickness"),
    (lambda d: d["materials"]["shield"].update(mu_r=1.0), "needs mu_r > 1"),
    (lambda d: d["materials"].update(shield=IRON), "must be a linear material"),
    (lambda d: d["curves"][0].update(material="superconductor"), "not a linear magnetic material"),
    (lambda d: d.update(mechanics={"supports": []}), "does not model"),
])
def test_spec_errors_name_the_curve(change, match):
    doc = copy.deepcopy(base_doc())
    change(doc)
    with pytest.raises(af.AxionFEMError, match=match):
        af.Design(doc).validate()


def test_adapt_with_layers_is_refused():
    doc = base_doc()
    doc["mesh"]["adapt"] = {"cycles": 2}
    with pytest.raises(af.AxionFEMError, match="mesh.adapt cannot be used with thin layers"):
        af.Design(doc).validate()


def test_hts_with_background_and_linear_region_is_refused():
    doc = tml.solid_sphere(R, 10.0, 0.2)
    doc["regions"].append({"name": "tape", "material": "nb",
                           "shape": {"rect": {"r": [1.5, 1.6], "z": [-0.1, 0.1]}}})
    doc["hts"] = {"regions": [{"region": "tape"}]}
    with pytest.raises(af.AxionFEMError, match="critical-state"):
        af.Design(doc).validate()


def test_toroidal_problem_is_refused(tmp_path):
    doc = base_doc()
    doc["regions"] = [{"name": "coil", "shape": {"rect": {"r": [0.3, 0.4], "z": [-0.1, 0.1]}},
                       "material": "winding"}]
    doc["windings"] = [{"name": "w", "region": "coil", "type": "poloidal", "ampere_turns": 1.0}]
    with pytest.raises(af.AxionFEMError, match="thin magnetic layer 'shell' cannot be used"):
        solve(doc, tmp_path, "tor")


# ---------------------------------------------------------------- physics


def test_sphere_shield_matches_closed_forms(tmp_path):
    """FE thin layer vs the model's closed form (discretization) and the exact shell (model)."""
    res = solve(tml.sphere_layer(R, D, MU_R, 0.1, 3), tmp_path, "sph")
    S = 1.0 / tml.center_Bz(res)
    S_model = tml.shield_model(MU_R, R, D)
    S_exact = tml.shield_exact(MU_R, R - D / 2, R + D / 2)
    assert abs(S / S_model - 1) < 1e-6  # the FE reaches 9e-8 here
    # the model's own error at d/R = 0.01: -0.94e-4
    assert S / S_exact - 1 == pytest.approx(-0.943e-4, abs=0.01e-4)
    li = res.data["static"]["poloidal"]["thin_layers"][0]
    assert li["split_dofs"] > 0 and li["min_h_perp_over_d"] > 4
    assert li["P_min_H"] == pytest.approx((MU_R - 1) * 4e-7 * math.pi * D, rel=1e-12)
    # B in the layer = mu_r mu0 H_t, with H_t = B_in/mu0 at the equator
    assert li["B_layer_max_T"] == pytest.approx(MU_R / S, rel=0.01)


def test_model_error_law_is_second_order():
    """(S_model - 1)/(S_exact - 1) - 1 = -(13/12 + 1/(2 mu_r)) (d/R)^2: the plain first-order
    formula, whose error is -1.5 d/R in S - 1, must fail this test."""
    for mu_r in (10.0, 1000.0, 1.0e5):
        for dR in (1e-3, 3e-4):
            ex = tml.shield_exact(mu_r, R - dR / 2, R + dR / 2)
            law = tml.model_error_law(mu_r, dR)
            mo = tml.shield_model(mu_r, R, dR)
            assert (mo - 1) / (ex - 1) - 1 == pytest.approx(law, rel=0.01)
            naive = tml.shield_first_order(mu_r, R, dR)
            assert abs((naive - 1) / (ex - 1) - 1 - law) > 100 * abs(law)


def test_linear_region_responds_to_background(tmp_path):
    """Solid mu_r = 10 sphere in 1 T: B_in = 3 mu/(mu + 2) = 2.5 T (not 1.0 T).
    The reported W_mag is the reaction field's energy (the background's linear work excluded)."""
    res = solve(tml.solid_sphere(R, 10.0, 0.2), tmp_path, "solid")
    assert tml.center_Bz(res) == pytest.approx(2.5, rel=2e-4)
    W = res.data["static"]["poloidal"]["energy_J"]
    assert W == pytest.approx(tml.solid_sphere_reaction_energy(R, 10.0, 1.0), rel=2e-3)


def test_layer_on_superconductor_independent_of_region_order(tmp_path):
    """A layer on an nb ball in 1 T against the exact coated superconducting sphere, with a vacuum
    jacket listed after and before the ball (the background drive must not depend on which side keeps the
    original DOF numbers; done wrong, [jacket, sc] gives B - 1 with the wrong sign)."""
    z = 1.5
    ex = tml.coated_sc_axis_Bz(MU_R, R - D / 2, R + D / 2, z)
    got = []
    for jacket_first in (False, True):
        res = solve(tml.coated_sc_sphere(R, D, MU_R, 0.1, 3, jacket_first), tmp_path, f"coated{jacket_first}")
        bz = float(res.sample([1e-6], [z])["Bz"][0])
        assert (bz - 1) / (ex - 1) - 1 == pytest.approx(0.0, abs=5e-4)
        got.append(bz)
    assert got[0] == pytest.approx(got[1], abs=1e-5)


def test_layer_on_linear_core_matches_three_region_sphere(tmp_path):
    """A layer on a mu_c = 5 ball: the half strip on the core side is counted in the core's medium
    (kills a model that takes vacuum on both sides)."""
    res = solve(tml.layer_on_core(R, D, MU_R, 5.0, 0.1), tmp_path, "core")
    bex = tml.core_shell_Bin(5.0, MU_R, R - D / 2, R + D / 2)
    assert tml.center_Bz(res) / bex - 1 == pytest.approx(0.0, abs=2e-4)  # model error 8.8e-5


@pytest.mark.parametrize("sheet", ["touch", "cross"])
def test_superconducting_sheet_meeting_a_layer(tmp_path, sheet):
    """An annular superconducting sheet that ends on (T) or crosses a can wall, thin vs meshed wall
    (a sheet ending on the layer must not tie both copies and cut the layer's flux: +98 % at the centre)."""
    r_thin = R if sheet == "touch" else 1.2
    r_mesh = R - 0.0025 if sheet == "touch" else 1.2
    thin = solve(tml.can_with_sheet(R, 2.0, 0.005, MU_R, 0.05, False, r_thin), tmp_path, "thin")
    mesh = solve(tml.can_with_sheet(R, 2.0, 0.005, MU_R, 0.0025, True, r_mesh), tmp_path, "mesh")
    pts = ([1e-6, 0.5, 0.7], [0.0, 0.0, 0.5])
    bt, bm = thin.sample(*pts), mesh.sample(*pts)
    # the crossing sheet's free edges (r = 0.4 and 1.2) are singular and converge slowly in both
    # models (5e-3 between meshes); the T contact has no such edge at the wall
    tol = 3e-3 if sheet == "touch" else 1.2e-2
    for k in range(3):
        assert bt["Bz"][k] == pytest.approx(bm["Bz"][k], abs=tol)
        assert bt["Br"][k] == pytest.approx(bm["Br"][k], abs=tol)
    if sheet == "touch":  # the flux the uncut wall carries: the centre field is ~0.33 T, not 0.65 T
        assert bt["Bz"][0] < 0.4


def test_layer_ending_on_a_magnetic_region_is_refused(tmp_path):
    """A thin wall ending on meshed magnetic caps cannot pass its flux into them."""
    with pytest.raises(af.AxionFEMError, match="ends on the magnetic region 'cap") as err:
        solve(tml.hybrid_can(R, 4.0, 0.005, MU_R, 0.05), tmp_path, "hybrid")
    assert "/curves/0/path" in str(err.value)


def test_pickup_in_can_thin_matches_meshed(tmp_path):
    """L_p of an nb loop in a closed can (touching the axis): thin wall vs meshed wall to 3e-5."""
    lp, warns = {}, {}
    for shield, h in (("none", 0.1), ("thin", 0.1), ("meshed", 0.0025)):
        res = solve(tml.pickup_in_can(R, 2.0, 0.005, MU_R, h, shield), tmp_path, shield)
        lp[shield] = res.data["pickup"]["L_p_H"]
        warns[shield] = [w for w in res.data["warnings"] if "non-conducting" in w]
        assert res.data["pickup"]["L_p_from_flux_H"] == pytest.approx(lp[shield], rel=1e-6)
    assert lp["thin"] > 1.01 * lp["none"]
    assert lp["thin"] == pytest.approx(lp["meshed"], rel=3e-5)
    assert warns["thin"] and not warns["meshed"]  # the non-conducting assumption is flagged


def test_order_aware_warning(tmp_path):
    """h_perp/d = 5.2: fine at order 2 (threshold 4), flagged at order 5 (threshold 7)."""
    for order, flagged in ((2, False), (5, True)):
        res = solve(tml.sphere_layer(R, D, MU_R, 0.1, order), tmp_path, f"p{order}")
        warns = [w for w in res.data["warnings"] if "layer thicknesses high" in w]
        assert bool(warns) is flagged


def test_layer_too_thick_for_the_mesh_is_refused(tmp_path):
    """The half-strip term is negative: a mesh much thinner than the layer is refused."""
    with pytest.raises(af.AxionFEMError, match="not positive definite") as err:
        solve(tml.sphere_layer(R, 0.1, MU_R, 0.05), tmp_path, "thick")
    assert "/curves/0/surface_size" in str(err.value)


def _cli(tmp, doc, env_extra):
    """Solve through the CLI: the solver backend threshold is read once per process."""
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "d.json"
    p.write_text(json.dumps(doc))
    env = dict(os.environ, **env_extra)
    return subprocess.run([axf_path(), "solve", str(p), "-o", str(tmp), "-v", "0"],
                          capture_output=True, text=True, env=env)


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_guard_does_not_depend_on_the_backend(tmp_path):
    """With STRUMPACK asked for at every size, layers keep the Eigen LDLT and its definiteness guard
    (STRUMPACK solves the indefinite cases without a word)."""
    env = {"AXF_STRUMPACK_MIN_DOFS": "0"}
    bad = _cli(tmp_path / "thick", tml.sphere_layer(R, 0.1, MU_R, 0.05), env)
    assert bad.returncode != 0
    assert "not positive definite" in bad.stdout + bad.stderr
    good = _cli(tmp_path / "ok", tml.sphere_layer(R, D, MU_R, 0.1, 3), env)
    assert good.returncode == 0, good.stdout + good.stderr
    res = json.loads((tmp_path / "ok" / "results.json").read_text())
    assert res["static"]["poloidal"]["solver"]["backend"] == "eigen-ldlt"


def test_adjoint_shape_sensitivity_with_a_layer(tmp_path):
    """The layer enters the adjoint through the same assembly: adjoint = fd on the morphed mesh."""
    doc = tml.pickup_in_can(R, 2.0, 0.005, MU_R, 0.1, "thin")
    doc["parameters"] = {"Rc": 1.0}
    doc["curves"][0]["path"]["polyline"]["points"] = [[0, -1], ["Rc", -1], ["Rc", 1], [0, 1]]
    res = solve(doc, tmp_path, "sens")
    sa = res.sensitivities(params=["Rc"], objectives=["L_p"], method="adjoint")
    sf = res.sensitivities(params=["Rc"], objectives=["L_p"], method="fd_morph", step=1e-3)
    ga = sa["gradient"]["L_p"]["Rc"]
    gf = sf["gradient"]["L_p"]["Rc"]
    assert ga < 0
    assert ga == pytest.approx(gf, rel=1e-4)


# ---------------------------------------------------------------- layers between superconductors; guards


def _bodies(res):
    return sorted(tuple(b["names"]) for b in res.data["static"]["poloidal"]["bodies"])


def test_two_sheets_on_opposite_faces_stay_two_bodies(tmp_path):
    """Two superconducting sheets meeting the can wall from inside and outside at one height are two
    bodies, and the wall's flux passes between them (merged, they would give 1.266 T instead of 0.336 T)."""
    thin = solve(tml.two_sheet_can(R, 2.0, 0.005, MU_R, 0.05, False), tmp_path, "thin")
    mesh = solve(tml.two_sheet_can(R, 2.0, 0.005, MU_R, 0.0025, True), tmp_path, "mesh")
    assert _bodies(thin) == [("inner",), ("outer",)] == _bodies(mesh)
    pts = ([1e-6, 0.5, 0.7], [0.0, 0.0, 0.5])
    bt, bm = thin.sample(*pts), mesh.sample(*pts)
    for k in range(3):
        assert bt["Bz"][k] == pytest.approx(bm["Bz"][k], abs=3e-3)


def test_flanged_coated_ball(tmp_path):
    """An nb ball with a layer on its surface and a superconducting flange on the layer's outer
    face: the flange is its own body (merged with the ball, it would give 0.780 T instead of the meshed 1.453 T
    at (0, 0.8))."""
    thin = solve(tml.flanged_ball(0.5, 0.005, MU_R, 0.025, False), tmp_path, "thin")
    mesh = solve(tml.flanged_ball(0.5, 0.005, MU_R, 0.0025, True), tmp_path, "mesh")
    assert _bodies(thin) == [("flange",), ("sc",)]
    pts = ([1e-6, 0.3, 0.7], [0.8, 0.6, 0.2])
    bt, bm = thin.sample(*pts), mesh.sample(*pts)
    for k in range(3):
        assert bt["Bz"][k] == pytest.approx(bm["Bz"][k], abs=5e-3)


def test_superconductors_on_both_sides_of_a_layer(tmp_path):
    """Two nb rings separated by a layer: the layer carries their flux difference with P = mu d
    (both sides superconducting) and matches a meshed wall. A layer that ends exactly where the
    rings meet is refused with a hint to extend it."""
    def rings(zl, meshed=False):
        g = 0.0025 if meshed else 0.0
        doc = {"schema": "axionfem.design/1", "name": "rings", "materials": tml.material(MU_R),
               "regions": [{"name": "inner", "material": "nb",
                            "shape": {"rect": {"r": [0.5, 0.6 - g], "z": [-0.1, 0.1]}}},
                           {"name": "outer", "material": "nb",
                            "shape": {"rect": {"r": [0.6 + g, 0.7], "z": [-0.1, 0.1]}}}],
               "background": {"type": "uniform", "Bz": 1.0},
               "mesh": {"order": 2, "size": 0.05, "surface_size": 0.0025 if meshed else 0.02,
                        "domain_radius": 2.0}}
        if meshed:
            doc["regions"].append({"name": "wall", "material": "shield", "mesh_size": 0.0025,
                                   "shape": {"rect": {"r": [0.5975, 0.6025], "z": [-zl, zl]}}})
        else:
            doc["curves"] = [{"name": "wall", "material": "shield", "thickness": 0.005,
                              "path": {"polyline": {"points": [[0.6, -zl], [0.6, zl]]}}}]
        return doc
    msg = "meet at an end of the thin magnetic layer 'wall'"
    with pytest.raises(af.AxionFEMError, match=msg) as err:
        solve(rings(0.1), tmp_path, "end")
    assert "/curves/0/path" in str(err.value)
    thin = solve(rings(0.15), tmp_path, "ext")
    mesh = solve(rings(0.15, True), tmp_path, "mesh")
    assert _bodies(thin) == [("inner",), ("outer",)] == _bodies(mesh)
    pts = ([0.6, 0.3, 0.8], [0.2, 0.0, 0.0])
    bt, bm = thin.sample(*pts), mesh.sample(*pts)
    for k in range(3):   # two discretizations; the first point lies just above the ends of the rings and the wall
        assert bt["Bz"][k] == pytest.approx(bm["Bz"][k], abs=1.5e-2)


@pytest.mark.parametrize("mu_c", [50.0, 5.0])
def test_partial_coating_on_a_linear_core_is_accepted(tmp_path, mu_c):
    """A coating on the upper hemisphere of a linear ball ends at the equator while running along
    the core: accepted, and it matches the meshed coating (to 1.4e-3 and 6e-3)."""
    thin = solve(tml.partial_coating(R, 0.01, MU_R, mu_c, 0.1, False), tmp_path, "thin")
    mesh = solve(tml.partial_coating(R, 0.01, MU_R, mu_c, 0.005, True), tmp_path, "mesh")
    pts = ([1e-6, 1e-6, 0.5, 0.5], [0.0, 0.6, 0.3, -0.3])
    bt, bm = thin.sample(*pts), mesh.sample(*pts)
    for k in range(4):
        assert bt["Bz"][k] == pytest.approx(bm["Bz"][k], rel=1e-2)


def test_nearly_nonmagnetic_caps_do_not_trigger_the_end_refusal(tmp_path):
    """Caps of mu_r = 1.005 (a stainless part) at the wall's end: accepted (1 % tolerance)."""
    doc = tml.hybrid_can(R, 4.0, 0.005, MU_R, 0.1)
    doc["materials"]["steel"] = {"kind": "linear", "mu_r": 1.005}
    for g in doc["regions"]:
        g["material"] = "steel"
        g["mesh_size"] = g["surface_size"] = 0.05
    solve(doc, tmp_path, "steel")


def test_sheet_on_the_layer_path_is_refused(tmp_path):
    """A superconducting sheet drawn on the wall's path would short it (0.812 T instead of 0.312 T)."""
    doc = tml.tube(R, 4.0, 0.005, MU_R, 0.1, False, True)
    doc["curves"].append(tml.sc_sheet("strip", [[R, -0.2], [R, 0.2]], 0.1))
    msg = "'strip' and the thin magnetic layer 'wall' share"
    with pytest.raises(af.AxionFEMError, match=msg) as err:
        solve(doc, tmp_path, "short")
    assert "/curves/0/path" in str(err.value)


def test_guard_message_precedes_the_exterior_map_at_order_6(tmp_path):
    """At order 6 the exterior map's capacitance fails on its own (pre-existing); the guard must run
    first so that a too-thin mesh gets the guard's message."""
    with pytest.raises(af.AxionFEMError, match="not positive definite"):
        solve(tml.sphere_layer(R, 0.1, MU_R, 0.05, 6), tmp_path, "p6")


def test_h_perp_counts_field_sides_only(tmp_path):
    """A wall lined inside by a 2 cm region: as nb, only the outer (field) side counts; as vacuum,
    the thin liner elements count too."""
    ratio = {}
    for mat in ("nb", "vacuum"):
        doc = tml.tube(R, 2.0, 0.005, MU_R, 0.05, False, False)
        doc["regions"] = [{"name": "liner", "material": mat,
                           "shape": {"rect": {"r": [0.98, 1.0], "z": [-1.0, 1.0]}}}]
        res = solve(doc, tmp_path, mat)
        ratio[mat] = res.data["static"]["poloidal"]["thin_layers"][0]["min_h_perp_over_d"]
    assert ratio["nb"] > 1.5 * ratio["vacuum"]


def test_minor_results(tmp_path):
    """Saturation warning, force closure, warm layers, thermal warning."""
    res = solve(tml.sphere_layer(R, D, MU_R, 0.1, 3), tmp_path, "sat")
    key = "default saturation threshold of 0.7 T"
    sat = [w for w in res.data["warnings"] if "saturat" in w and key in w]
    assert len(sat) == 1
    doc = tml.sphere_layer(R, D, MU_R, 0.1, 3)
    doc["materials"]["shield"]["B_sat"] = 1000.0
    assert not [w for w in solve(doc, tmp_path, "sat2").data["warnings"] if "saturat" in w]

    plate = {"schema": "axionfem.design/1", "name": "plate", "materials": tml.material(MU_R),
             "regions": [{"name": "coil", "material": "winding", "surface_size": 0.01,
                          "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.05, 0.05]}}}],
             "windings": [{"name": "coil", "region": "coil", "type": "azimuthal",
                           "ampere_turns": 1e5}],
             "curves": [{"name": "plate", "material": "shield", "thickness": 0.005,
                         "surface_size": 0.05,
                         "path": {"polyline": {"points": [[0.0, 0.4], [2.5, 0.4]]}}}],
             "mesh": {"order": 2, "size": 0.1, "surface_size": 0.02}}
    res = solve(plate, tmp_path, "plate")
    newton = [c for c in res.data["checks"] if "Newton" in c.get("name", "")]
    assert newton and newton[0]["valid"] is False
    a_cold = res.data["geometry"]["A_cold_m2"]
    warm = copy.deepcopy(plate)
    warm["materials"]["shield"]["warm"] = True
    assert solve(warm, tmp_path, "warm").data["geometry"]["A_cold_m2"] < 0.5 * a_cold
    thermal = copy.deepcopy(plate)
    thermal["thermal"] = {"fixed": [{"region": "coil", "T_K": 4.2}]}
    warns = solve(thermal, tmp_path, "thermal").data["warnings"]
    assert any("has no thermal role" in w for w in warns)


# ---------------------------------------------------------------- layer ends, crossing regions, pickups, open ends


def test_end_refusal_is_per_region(tmp_path):
    """A weakly magnetic collar along the wall's last 5 cm does not let the caps in front of the
    end through (that would be +23.5 % at the centre, silently); caps of mu_r = 1.05 alone are refused
    too, which pins the 1 % threshold from above."""
    with pytest.raises(af.AxionFEMError, match="ends on the magnetic region 'cap"):
        solve(tml.hybrid_can_with_collar(1.05), tmp_path, "collar")
    doc = tml.hybrid_can(R, 4.0, 0.005, MU_R, 0.1)
    doc["materials"]["steel"] = {"kind": "linear", "mu_r": 1.05}
    for g in doc["regions"]:
        g["material"] = "steel"
        g["mesh_size"] = g["surface_size"] = 0.05
    with pytest.raises(af.AxionFEMError, match="ends on the magnetic region 'cap"):
        solve(doc, tmp_path, "steel105")


def test_region_crossing_a_layer_cuts_it(tmp_path):
    """An nb plate region across the can wall is one body and cuts the wall like a crossing sheet
    (split into two bodies, it would give 0.328 T instead of about 0.99 T)."""
    reg = solve(tml.plate_can(0.05, False), tmp_path, "region")
    sheet = solve(tml.plate_can(0.05, True), tmp_path, "sheet")
    assert _bodies(reg) == [("plate",)]
    assert tml.center_Bz(reg) == pytest.approx(tml.center_Bz(sheet), abs=2e-2)
    assert tml.center_Bz(reg) > 0.9


def test_pickup_ring_crossed_by_a_layer(tmp_path):
    """A pickup ring crossed by a weak layer stays one conductor: L_p from the energy and from the
    flux agree, and the weak layer changes L_p little (split, L_p would be 4x and 2x too large)."""
    lp0 = solve(tml.crossed_pickup(None), tmp_path, "none").data["pickup"]["L_p_H"]
    pk = solve(tml.crossed_pickup(1.5), tmp_path, "weak").data["pickup"]
    assert pk["L_p_from_flux_H"] == pytest.approx(pk["L_p_H"], rel=1e-6)
    assert pk["L_p_H"] == pytest.approx(lp0, rel=3e-3)


def test_pickup_conductor_in_several_bodies_is_refused(tmp_path):
    """A pickup conductor made of two disconnected pieces is refused (also without layers)."""
    doc = tml.crossed_pickup(None)
    doc["regions"][0]["shape"] = {"union": [{"rect": {"r": [0.45, 0.5], "z": [-0.02, 0.02]}},
                                            {"rect": {"r": [0.6, 0.65], "z": [-0.02, 0.02]}}]}
    with pytest.raises(af.AxionFEMError, match="forms 2 separate superconducting bodies"):
        solve(doc, tmp_path, "two")


def test_sheets_meeting_at_a_layer_end_are_refused(tmp_path):
    """Two sheets meeting the open tube's end from inside and outside are refused, as documented
    (they must not be merged into one body); one sheet across the end is accepted."""
    doc = tml.tube(R, 2.0, 0.005, MU_R, 0.05, False, False)
    doc["curves"] += [tml.sc_sheet("inner", [[0.4, 1.0], [R, 1.0]], 0.05),
                      tml.sc_sheet("outer", [[R, 1.0], [1.6, 1.0]], 0.05)]
    msg = "meet at an end of the thin magnetic layer 'wall'"
    with pytest.raises(af.AxionFEMError, match=msg) as err:
        solve(doc, tmp_path, "two")
    assert "/curves/0/path" in str(err.value)
    doc = tml.tube(R, 2.0, 0.005, MU_R, 0.05, False, False)
    doc["curves"] += [tml.sc_sheet("lid", [[0.4, 1.0], [1.6, 1.0]], 0.05)]
    assert _bodies(solve(doc, tmp_path, "one")) == [("lid",)]
