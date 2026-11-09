"""Adaptive refinement (mesh.adapt): re-meshing with Zienkiewicz-Zhu size fields converges Psi, L_p, W."""
import math

import pytest

import axionfem as af

needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build; the wheels have none)")


def test_adaptive_refinement_converges_towards_fine_mesh(tmp_path):
    ref = af.template("solenoid", mesh={"order": 2, "size": 0.02, "surface_size": 0.004})
    r_ref = ref.solve(tmp_path / "ref", verbose=0).coupling
    coarse = {"order": 2, "size": 0.2, "surface_size": 0.08}
    r0 = af.template("solenoid", mesh=coarse).solve(tmp_path / "c0", verbose=0).coupling
    d = af.template("solenoid", mesh={**coarse, "adapt": {"cycles": 6, "tol": 2e-4}})
    r = d.solve(tmp_path / "ad", verbose=0)
    ad = r.data["adapt"]
    print([(h["elements"], h["Psi_SI_H_T_m2"], h["L_p_H"], h["estimated_relative_error"]) for h in ad["history"]])
    assert len(ad["history"]) >= 2
    errs = [h["estimated_relative_error"] for h in ad["history"]]
    assert errs[-1] < 0.5 * errs[0]
    for k in ("Psi_SI_H_T_m2", "L_p_H", "W_mag_J"):
        e0 = abs(r0[k] / r_ref[k] - 1)
        e1 = abs(r.coupling[k] / r_ref[k] - 1)
        print(k, "coarse", e0, "adapted", e1)
        assert e1 < max(0.3 * e0, 2e-4)
    assert r.data["mesh"]["elements"] < 0.5 * len_elements(tmp_path / "ref")


def len_elements(wd):
    import json
    return json.loads((wd / "results.json").read_text())["mesh"]["elements"]


SOLENOID = dict(R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05)
MESH3D = {"size": 0.5, "surface_size": 0.2, "order": 1}


@pytest.fixture(scope="module")
def solenoid_3d_adapt(tmp_path_factory):
    """The 3D solenoid adapted twice with the default watch, and its 2D reference."""
    from axionfem.convert import to_3d

    wd = tmp_path_factory.mktemp("adapt3d")
    d2 = af.template("solenoid", **SOLENOID, mesh={"order": 2, "size": 0.03, "surface_size": 0.008})
    ref = d2.solve(wd / "ref2d", verbose=0).coupling
    r = to_3d(d2, mesh={**MESH3D, "adapt": {"cycles": 2, "tol": 1e-4}}).solve(wd / "ad3", verbose=0)
    return d2, ref, r


@needs_3d
def test_adaptive_refinement_3d(solenoid_3d_adapt):
    d2, ref, r = solenoid_3d_adapt
    h = r.data["adapt"]["history"]
    print([(x["elements"], x["Psi_SI_H_T_m2"], x["L_p_H"], x["estimated_relative_error"]) for x in h])
    assert len(h) == 3
    assert h[-1]["elements"] > h[0]["elements"]
    assert h[-1]["estimated_relative_error"] < h[0]["estimated_relative_error"]
    e0 = abs(h[0]["L_p_H"] / ref["L_p_H"] - 1)
    e2 = abs(h[-1]["L_p_H"] / ref["L_p_H"] - 1)
    assert e2 < e0
    # the history carries every pickup surface's loss factor (pickup.losses), so Gamma can be followed per cycle
    assert set(h[0]["Gamma"]) == {"coax_inner", "coax_outer", "coax_short"}
    assert r.data["adapt"]["watch"] == ["Psi", "L_p", "W_mag"]


def test_adapt_watch_gamma_2d(tmp_path):
    """Plumbing in 2D: the watch is echoed, the history rows carry Gamma per surface, and each cycle's change is the
    largest over the watched quantities. The copper coax has rf data, so R_eff is compared, whose relative changes
    equal Gamma's here (one R_s). The 2D Gamma barely moves on this geometry, so the stopping semantics themselves
    (1 % filter, R_eff preference, never-converged cases) are unit-tested in cpp/tests/test_adapt_change.cpp."""
    coarse = {"order": 2, "size": 0.2, "surface_size": 0.08}
    d = af.template("solenoid", mesh={**coarse, "adapt": {"cycles": 3, "tol": 1e-3, "watch": ["L_p", "Gamma"]}})
    ad = d.solve(tmp_path / "w", verbose=0).data["adapt"]
    assert ad["watch"] == ["L_p", "Gamma"]
    assert len(ad["history"]) >= 2
    for row in ad["history"]:
        assert set(row["Gamma"]) == {"coax_inner", "coax_outer", "coax_short"}
    for prev, row in zip(ad["history"], ad["history"][1:]):
        total = sum(row["Gamma"].values())
        g = max(abs(row["Gamma"][s] - prev["Gamma"][s]) / row["Gamma"][s]
                for s in row["Gamma"] if row["Gamma"][s] >= 0.01 * total)
        lp = abs(row["L_p_H"] - prev["L_p_H"]) / row["L_p_H"]
        assert row["max_relative_change"] == pytest.approx(max(g, lp), rel=1e-12)


@needs_3d
def test_adapt_watch_gamma_3d_refines_the_surfaces(solenoid_3d_adapt, tmp_path):
    """3D with Gamma watched: the indicator gains the pickup field's recovery error on the conductor faces. From the
    same start mesh, the surface-loss factors end closer to the closed form (benchmarks/rf_losses_3d.py) per element
    than with the default watch: the default run's final worst error, carried to the Gamma run's element count with
    its own last-cycle rate, is at least 10 % larger (measured about 20 %)."""
    from axionfem.convert import to_3d

    d2, _, default = solenoid_3d_adapt
    R, H, x, g, tc, tp = (SOLENOID[k] for k in ("R", "H", "x", "g", "tc", "tp"))
    length = H - tp - tc + tp / 3
    ri, ro = x * R, R - g - tc
    exact = {"coax_inner": length / (2 * math.pi * ri), "coax_outer": length / (2 * math.pi * ro),
             "coax_short": math.log(ro / ri) / (2 * math.pi)}

    def worst(row):
        return max(abs(row["Gamma"][s] / exact[s] - 1) for s in exact)

    d3 = to_3d(d2, mesh={**MESH3D, "adapt": {"cycles": 2, "tol": 1e-4, "watch": ["Gamma"]}})
    r = d3.solve(tmp_path / "g3", verbose=0)
    assert r.data["adapt"]["watch"] == ["Gamma", "L_p"]    # watching the loss implies L_p
    hg = r.data["adapt"]["history"]
    hd = default.data["adapt"]["history"]
    print([(row["elements"], worst(row)) for row in hd], [(row["elements"], worst(row)) for row in hg])
    assert len(hg) == 3
    assert worst(hg[-1]) < worst(hg[0])
    rate = math.log(worst(hd[1]) / worst(hd[2])) / math.log(hd[2]["elements"] / hd[1]["elements"])
    default_at_same_size = worst(hd[2]) * (hg[2]["elements"] / hd[2]["elements"]) ** (-rate)
    assert worst(hg[2]) < 0.9 * default_at_same_size


def test_adapt_watch_and_refinements_are_validated():
    with pytest.raises(Exception, match="unknown adapt.watch quantity"):
        af.template("solenoid", mesh={"adapt": {"cycles": 2, "watch": ["Q"]}}).validate()
    with pytest.raises(Exception, match="cannot be combined with mesh.adapt"):
        af.template("solenoid", mesh={"refinements": 1, "adapt": {"cycles": 2}}).validate()
    # Gamma needs a pickup and a conductor or superconductor surface
    doc = {"schema": "axionfem.design/1", "name": "bare",
           "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.1, 0.1]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mesh": {"adapt": {"cycles": 2, "watch": ["Gamma"]}}}
    with pytest.raises(Exception, match="needs a pickup and a conductor or superconductor surface"):
        af.Design(doc).validate()
    af.template("solenoid", mesh={"adapt": {"cycles": 2, "watch": ["Gamma"]}}).validate()   # has both: accepted
