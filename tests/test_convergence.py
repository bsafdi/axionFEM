"""Three-mesh Richardson extrapolation and the uniform h-study (axionfem.convergence)."""
import math

import pytest

import axionfem as af
from axionfem.convergence import adapt_convergence, h_study, refine_and_compare, richardson, scaled_design


def test_richardson_recovers_order_and_limit_for_unequal_ratios():
    f = lambda h: 1.0 - 0.3 * h ** 1.5  # noqa: E731
    h = [0.4, 0.13, 0.25]                 # any order, unequal ratios
    r = richardson(h, [f(x) for x in h])
    assert r["order"] == pytest.approx(1.5, rel=1e-9)
    assert r["extrapolated"] == pytest.approx(1.0, rel=1e-12)
    assert r["behaviour"] == "monotone"
    assert r["h"] == [0.13, 0.25, 0.4]


@pytest.mark.parametrize("h,p", [((0.1, 0.15, 0.45), 1.0),   # a plain fixed point returns 1.957 here
                                 ((1.0, 0.5, 0.48), 2.0),    # ln r32 / ln r21 >> 2: falsely "no usable order"
                                 ((1.0, 0.5, 0.45), 3.0),
                                 ((1.0, 0.97, 0.5), 1.0)])
def test_richardson_exact_for_strongly_unequal_ratios(h, p):
    r = richardson(list(h), [1.0 + 0.5 * x ** p for x in h])
    assert r["order"] == pytest.approx(p, rel=1e-9)
    assert r["extrapolated"] == pytest.approx(1.0, rel=1e-12)


def test_richardson_noise_floor_and_formal_order():
    # one difference at round-off (a 2D Gamma exact on two meshes): no order, not an order of 69
    r = richardson([1.0, 0.7, 0.5], [0.5772098747589962, 0.5772098747589965, 0.5772115891])
    assert r["extrapolated"] is None and r["behaviour"].startswith("no usable order")
    assert richardson([1.0, 0.7, 0.5], [1.01, 1.0, 1.0])["extrapolated"] is None   # the finest moved: no "unchanged"
    # formal order 2 with an observed order 3: Roache's conservative variant (p = 2, safety 3)
    h = [1.0, 0.75, 0.55]
    c = richardson(h, [1.0 + 0.5 * x ** 3 for x in h], formal_order=2)
    assert c["order"] == pytest.approx(3.0) and c["order_used"] == 2 and c["safety_factor"] == 3.0
    high = richardson(h, [1.0 + 0.5 * x ** 9 for x in h])
    assert high["behaviour"] == "unreliable order" and high["order"] is None and high["order_exceeds"] == 8.0
    # far above the formal order: flagged, but Roache's conservative GCI is still given
    far = richardson(h, [1.0 + 0.5 * x ** 5 for x in h], formal_order=2)
    assert far["behaviour"].endswith("observed order far above formal") and far["gci_fine"] > 0
    assert far["order_used"] == 2 and far["safety_factor"] == 3.0


def test_formal_order_per_quantity():
    hist = {"design": {"symmetry": "3d"},
            "adapt": {"history": [{"dofs": n, "L_p_H": 1.0 + n ** (-4 / 3), "Gamma": {"wall": 2.0 + n ** (-2 / 3)}}
                                  for n in (1000, 8000, 64000)]}}
    out = adapt_convergence(hist, formal_order={"L_p_H": 4, "Gamma": 2})   # d = 3 from the design; h^4 and h^2
    assert out["L_p_H"]["order_used"] == pytest.approx(4.0) and out["L_p_H"]["safety_factor"] == 1.25
    assert out["Gamma:wall"]["order_used"] == pytest.approx(2.0) and out["Gamma:wall"]["safety_factor"] == 1.25


def test_richardson_flags_oscillation_and_rejects_bad_input():
    assert richardson([1, 0.7, 0.5], [3.1, 2.95, 3.02])["behaviour"] == "oscillatory"
    assert richardson([1, 0.7, 0.5], [2.0, 2.0, 2.0])["behaviour"] == "unchanged"
    flat = richardson([1, 0.7, 0.5], [1.0, 1.0 + 1e-9, 1.0 + 2e-9])   # equal differences: no order
    assert flat["extrapolated"] is None and flat["behaviour"].endswith("no usable order")
    with pytest.raises(ValueError):
        richardson([1, 1, 0.5], [1.0, 2.0, 3.0])


def test_scaled_design_scales_every_mesh_size():
    d = af.Design({"schema": "axionfem.design/1", "name": "s", "parameters": {"w": 0.1},
                   "regions": [{"name": "a", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.1, 0.1]}},
                                "material": "winding", "mesh_size": "w/4", "surface_size": 0.01}],
                   "windings": [{"region": "a", "type": "azimuthal", "ampere_turns": 1.0}],
                   "mesh": {"order": 2, "size": 0.2, "surface_size": 0.05}})
    s = scaled_design(d, 0.5)
    assert s.doc["mesh"]["size"] == pytest.approx(0.1)
    assert s.doc["mesh"]["surface_size"] == pytest.approx(0.025)
    assert s.doc["regions"][0]["surface_size"] == pytest.approx(0.005)
    assert s.doc["regions"][0]["mesh_size"] == "(w/4)*0.5"
    assert d.doc["mesh"]["size"] == 0.2   # the original is untouched
    s.validate()


def test_h_study_2d(tmp_path):
    """On the 2D solenoid at order 1: three uniformly scaled meshes; the extrapolated L_p is closer to a fine
    reference than the finest of the three. The 2D Gamma here is exact to round-off on every mesh (a 1/r field on
    straight walls), so it has no usable order, which the extrapolation must report rather than divide by."""
    p = dict(R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05)
    d = af.template("solenoid", **p, mesh={"order": 1, "size": 0.2, "surface_size": 0.08})
    out = h_study(d, [1.0, 0.7, 0.5], tmp_path)
    assert [row["factor"] for row in out["rows"]] == [1.0, 0.7, 0.5]
    assert {"Gamma:coax_inner", "Gamma:coax_outer", "Gamma:coax_short"} <= set(out["extrapolation"])
    ref = af.template("solenoid", **p, mesh={"order": 2, "size": 0.02, "surface_size": 0.004}).solve(
        tmp_path / "ref", verbose=0).coupling["L_p_H"]
    ex = out["extrapolation"]["L_p_H"]
    finest = out["rows"][-1]["quantities"]["L_p_H"]
    print(ex, finest, ref)
    assert ex["order"] > 0.5
    assert abs(ex["extrapolated"] - ref) < abs(finest - ref)
    for s in ("coax_inner", "coax_outer", "coax_short"):   # at round-off: no extrapolation, no absurd order
        g = out["extrapolation"][f"Gamma:{s}"]
        assert g["behaviour"] == "unchanged" or g["behaviour"].startswith("no usable order"), (s, g)


def test_adapt_convergence_on_a_history():
    hist = {"adapt": {"history": [{"dofs": n, "L_p_H": 1.0 - n ** -0.5, "Gamma": {"wall": 2.0 + n ** -0.5}}
                                  for n in (1000, 4000, 16000)]}}
    out = adapt_convergence(hist, dimension=2)    # h = n^(-1/2): first order in h
    assert out["L_p_H"]["order"] == pytest.approx(1.0, rel=1e-9)
    assert out["Gamma:wall"]["extrapolated"] == pytest.approx(2.0, rel=1e-12)
    assert "heuristic" in out["L_p_H"]["note"]


def test_refine_and_compare_on_synthetic_solves(tmp_path, monkeypatch):
    """The bookkeeping of refine_and_compare without a solver: L_n = L (1 - c h_n^4) per frequency and an indicator
    of twice the true error. The reference is the Richardson limit, and the effectivity 2 on every level."""
    from axionfem.design import Design

    limits = {1e6: 2.0e-7, 2e7: 3.0e-7}
    seen = []

    drop = {}   # refinements -> a frequency left out of that mesh's sweep

    class FakeResults:
        def __init__(self, refinements):
            h = 2.0 ** -refinements
            error = 0.01 * h ** 4
            sweep = []
            for f, limit in limits.items():
                if drop.get(refinements) == f:
                    continue
                indicator = {"estimated_relative_error": math.sqrt(2 * error), "L_relative_error_estimate": 2 * error,
                             "L_signed_relative_error_estimate": -error}
                sweep.append({"f_Hz": f, "L_H": limit * (1 - error), "Z_re_ohm": 0.0, "Z_im_ohm": -1.0,
                              "Psi_abs": 0.0, "error_indicator": indicator})
            if refinements % 2:   # the rows are paired by f_Hz, not by position
                sweep.reverse()
            self.data = {"mesh": {"dofs": 1000 * 8 ** refinements, "elements": 100 * 8 ** refinements},
                         "fullwave": {"sweep": sweep}}

    def fake_solve(self, workdir, verbose=0, mpi_ranks=1):
        seen.append(self.doc["mesh"]["refinements"])
        return FakeResults(self.doc["mesh"]["refinements"])

    monkeypatch.setattr(Design, "solve", fake_solve)
    doc = {"schema": "axionfem.design/1", "symmetry": "3d", "mesh": {"order": 2, "refinements": 1},
           "fullwave": {"frequencies": list(limits)}}
    out = refine_and_compare(doc, tmp_path, levels=3)
    assert seen == [1, 2, 3]   # on top of the design's own refinement
    assert [level["dofs"] for level in out["levels"]] == [8000, 64000, 512000]
    for freq in out["frequencies"]:
        assert freq["L_reference_H"] == pytest.approx(limits[freq["f_Hz"]], rel=1e-12)
        assert freq["L_extrapolation"]["order"] == pytest.approx(4.0, rel=1e-6)
        for row in freq["rows"]:
            assert row["effectivity"] == pytest.approx(2.0, rel=1e-6)
        # an exact signed estimate predicts the change of L between meshes (to the product of two errors)
        for change in freq["changes"]:
            assert change["predicted_change"] == pytest.approx(change["L_change_relative"], rel=1e-3)
    two = refine_and_compare(doc, tmp_path, levels=2)   # no extrapolation: the finest mesh is the reference
    assert two["frequencies"][0]["rows"][-1]["observed_relative_error"] is None
    assert two["frequencies"][0]["rows"][0]["effectivity"] == pytest.approx(2.0 * 16 / 15, rel=1e-3)
    with pytest.raises(ValueError, match="3D design with fullwave.frequencies"):
        refine_and_compare({"mesh": {}}, tmp_path)
    drop[2] = 2e7   # a frequency rejected on one mesh cannot be compared
    with pytest.raises(ValueError, match="a frequency missing on one mesh cannot be compared"):
        refine_and_compare(doc, tmp_path, levels=2)
