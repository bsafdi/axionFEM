"""Lorentz forces of the axisymmetric engine: the winding self-force floor, Newton's-third-law closure in `checks`,
the per-body force distributions (`forces.distributions`), their size cap and the plotting helper.
"""
from __future__ import annotations

import json
import math

import pytest

import axionfem as af

MU0 = 4e-7 * math.pi


def _loop_sphere(with_sphere: bool, a: float = 0.06, z0: float = 0.4):
    regions = [{"name": "coil", "material": "winding",
                "shape": {"rect": {"r": [0.99, 1.01], "z": [-0.01, 0.01]}}, "surface_size": 0.005}]
    doc = {"schema": "axionfem.design/1", "name": "loop", "regions": regions,
           "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
           "mesh": {"order": 3, "size": 0.05, "surface_size": 0.01}}
    if with_sphere:
        regions.append({"name": "sphere", "material": "nb", "mesh_size": a / 6,
                        "shape": {"disk": {"center": [0, z0], "radius": a}}})
        doc["superconductors"] = [{"body": "sphere", "azimuthal": {"flux": 0}}]
    return af.Design(doc)


@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    return _loop_sphere(False).solve(workdir=tmp_path_factory.mktemp("loop"), verbose=0)


@pytest.fixture(scope="module")
def loop_sphere(tmp_path_factory):
    return _loop_sphere(True).solve(workdir=tmp_path_factory.mktemp("loopsph"), verbose=0)


def test_isolated_loop_self_force_is_removed(loop):
    """The net axial force of a loop on itself is exactly zero.  The raw Lorentz
    integral returns a residual of order 1e-2 N against a 365 N hoop load; subtracting the force
    computed from the same winding alone in vacuum removes it."""
    f = loop.forces["coil"]
    hoop = abs(f["radial_force_per_radian_N"]) * 2 * math.pi
    assert abs(f["Fz_self_N"]) == pytest.approx(abs(f["Fz_N"]), rel=1e-6)   # all of Fz is self-force
    assert abs(f["Fz_self_N"]) > 1e-4                                      # and it is not negligible
    assert abs(f["Fz_excluding_self_N"]) < 1e-9 * hoop


def test_loop_and_superconductor_obey_newtons_third_law(loop_sphere):
    f = loop_sphere.forces
    coil, sphere = f["coil"], f["superconductor:sphere"]
    # the sphere is pulled towards the loop; the reaction on the loop is equal and opposite
    assert coil["Fz_excluding_self_N"] == pytest.approx(-sphere["Fz_N"], rel=5e-3)
    # the raw value is not: the self-force floor is comparable to the real force
    assert abs(coil["Fz_N"] + sphere["Fz_N"]) > 100 * abs(coil["Fz_excluding_self_N"] + sphere["Fz_N"])
    # ... and it matches the point-dipole gradient force of a perfectly diamagnetic sphere,
    # F_z = -(pi a^3 / mu0) d(B^2)/dz, with B the on-axis field of a thin loop
    R, I, a, z0 = 1.0, 1e4, 0.06, 0.4
    Bz = MU0 * I * R**2 / (2 * (R**2 + z0**2) ** 1.5)
    dBz = -3 * MU0 * I * R**2 * z0 / (2 * (R**2 + z0**2) ** 2.5)
    ref = -(math.pi * a**3 / MU0) * 2 * Bz * dBz
    assert sphere["Fz_N"] == pytest.approx(ref, rel=0.015), (sphere["Fz_N"], ref)


def test_closure_check_is_reported(loop_sphere):
    f = loop_sphere.forces
    c = f["closure"]
    assert {"sum_Fz_N", "force_scale_N", "relative", "bodies"} <= set(c)
    assert abs(c["relative"]) < 1e-5
    names = {b["body"] for b in c["bodies"]}
    assert names == {"winding:coil", "superconductor:sphere"}
    chk = [c for c in loop_sphere.data["checks"] if "Newton's third law" in c["name"]]
    assert len(chk) == 1 and chk[0]["valid"] is True
    assert chk[0]["exact_value"] == 0.0
    # B.n = 0 on the superconductor is checked too
    bn = [c for c in loop_sphere.data["checks"] if "B.n" in c["name"]]
    assert len(bn) == 1 and bn[0]["max_Bn_over_Bmax"] < 1e-8


def test_distributions_integrate_to_the_net_force(loop_sphere):
    d = loop_sphere.forces["distributions"]
    assert d["sphere"]["kind"] == "superconductor" and d["sphere"]["geometry"] == "surface"
    s = d["sphere"]
    total = sum(fz * dl for fz, dl in zip(s["f_z_N_per_m"], s["dl_m"]))
    assert total == pytest.approx(s["Fz_N"], rel=1e-9)
    assert s["Fz_N"] == pytest.approx(loop_sphere.forces["superconductor:sphere"]["Fz_N"], rel=1e-4)
    for key in ("r_m", "z_m", "arclength_m", "dl_m", "n_r", "n_z",
                "traction_r_Pa", "traction_z_Pa", "f_r_N_per_m", "f_z_N_per_m"):
        assert len(s[key]) == s["samples"], key
    # the traction on a superconductor is pure pressure: t = -|B|^2/(2 mu0) n, so t . n < 0
    assert all(tr * nr + tz * nz < 0 for tr, tz, nr, nz in
               zip(s["traction_r_Pa"], s["traction_z_Pa"], s["n_r"], s["n_z"]))
    # the winding gets a volume distribution (J x B) and a separate boundary one
    v = d["coil"]
    assert v["geometry"] == "volume"
    assert sum(v["f_z_N"]) == pytest.approx(v["Fz_N"], abs=1e-9 + 1e-6 * abs(v["Fz_N"]))
    assert v["Fz_N"] == pytest.approx(loop_sphere.forces["coil"]["Fz_N"], rel=1e-9)
    assert d["coil:boundary"]["geometry"] == "surface"
    assert "Fz_lorentz_N" in d["coil:boundary"]


def test_sheet_distribution_counts_its_length_once(tmp_path):
    """A zero-thickness sheet is loaded from both faces.  Its distribution must report the sheet's
    true length with the two tractions summed, not twice the length with them averaged."""
    doc = {"schema": "axionfem.design/1", "name": "sh",
           "regions": [{"name": "coil", "material": "winding",
                        "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.1, 0.1]}}}],
           "curves": [{"name": "disc", "material": "nb",
                       "path": {"segment": [[0.2, 0.5], [0.8, 0.5]]}, "surface_size": 0.01}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
           "superconductors": [{"body": "disc", "azimuthal": {"flux": 0}}],
           "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02}}
    r = af.Design(doc).solve(workdir=tmp_path / "sheet", verbose=0)
    d = r.forces["distributions"]["disc"]
    assert d["length_m"] == pytest.approx(0.6, rel=1e-9)        # not 1.2
    assert sum(d["dl_m"]) == pytest.approx(0.6, rel=1e-9)
    total = sum(fz * dl for fz, dl in zip(d["f_z_N_per_m"], d["dl_m"]))
    assert total == pytest.approx(d["Fz_N"], rel=1e-9)
    assert d["Fz_N"] == pytest.approx(r.forces["superconductor:disc"]["Fz_N"], rel=1e-6)
    # the net load points away from the coil (the sheet is pushed up, off the z = 0 plane)
    assert d["Fz_N"] > 0


@pytest.mark.parametrize("n", [2000, 200, 50, 0])
def test_force_samples_bounds_results_json_without_changing_the_forces(tmp_path, n):
    """The sample arrays are ~160 bytes per sample per body and would otherwise dominate
    results.json (884 kB instead of 10 kB on CORE).  The merge that produces them preserves the
    integral, so the net forces must not depend on the cap -- only the file size does."""
    r = af.template("core").solve(workdir=tmp_path / f"fs{n}", verbose=0,
                                  self_force=False, force_samples=n)
    d = r.forces["distributions"]
    size = len(json.dumps(r.data))
    per_body = {k: v["samples"] for k, v in d.items()}
    assert all(v <= max(n, 0) for v in per_body.values()), per_body
    if n == 0:
        assert all(v == 0 for v in per_body.values())
        assert not any("r_m" in v for v in d.values())
        assert size < 50_000                      # back to the pre-distribution size
    else:
        assert all("r_m" in v and len(v["r_m"]) == v["samples"] for v in d.values())
    # the net forces and the closure are independent of the cap
    assert r.forces["distributions"]["cavity"]["Fz_N"] == pytest.approx(-8.73, rel=0.02)
    assert r.forces["closure"]["relative"] == pytest.approx(-2.404e-5, rel=0.05)


def test_force_distribution_plot(loop_sphere, tmp_path):
    pytest.importorskip("matplotlib")
    from axionfem import plots

    fig = plots.force_distribution(loop_sphere, bodies=["sphere"])
    out = tmp_path / "forces.png"
    fig.savefig(out, dpi=50)
    assert out.stat().st_size > 1000
    fig2 = plots.force_distribution(loop_sphere, quantity="traction_z", bodies=["sphere"])
    assert fig2 is not None
    with pytest.raises(KeyError):
        plots.force_distribution(loop_sphere, bodies=["nope"])


def test_scene_summary_omits_the_distribution_arrays(loop_sphere):
    """The viewer scene carries the net forces; the per-sample arrays would add megabytes to every
    scene.json and single-file HTML export."""
    sc = loop_sphere.scene()
    f = sc["summary"]["forces"]
    assert "distributions" not in f
    assert "closure" in f and "coil" in f          # the net forces and the closure check stay
    assert len(json.dumps(f)) < 20000, len(json.dumps(f))
    # no per-sample numeric array survived (closure.bodies is a short list of dicts, which is fine)
    for body in f.values():
        if not isinstance(body, dict):
            continue
        for v in body.values():
            assert not (isinstance(v, list) and v and isinstance(v[0], (int, float)))
    # the keys viewer/src/viewer.js reads in its Forces panel (no browser needed to pin the contract)
    assert {"sum_Fz_N", "relative"} <= set(f["closure"])
    assert {"Fz_N", "Fz_self_N", "Fz_excluding_self_N", "radial_force_per_radian_N"} <= set(f["coil"])
    assert "Fz_N" in f["superconductor:sphere"]


def test_linear_permeability_body_gets_a_force_and_enters_the_closure(tmp_path):
    """A `linear` material with mu_r != 1 is a magnetizable body: it feels a real axial force, so it
    must be in `forces` and in the Newton's-third-law sum, or the check would call a design balanced
    while silently leaving a body out of it."""
    doc = {"schema": "axionfem.design/1", "name": "mag",
           "materials": {"ferrite": {"kind": "linear", "mu_r": 100.0}},
           "regions": [{"name": "coil", "material": "winding",
                        "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.05, 0.05]}}, "surface_size": 0.01},
                       {"name": "slug", "material": "ferrite",
                        "shape": {"rect": {"r": [0.0, 0.3], "z": [0.3, 0.5]}}, "surface_size": 0.01}],
           "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
           "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02}}
    r = af.Design(doc).solve(workdir=tmp_path / "mag", verbose=0)
    f = r.forces
    assert "magnetic:slug" in f
    # the slug is pulled into the coil, i.e. towards -z; the coil is pushed the other way
    assert f["magnetic:slug"]["Fz_N"] < 0
    assert f["coil"]["Fz_excluding_self_N"] > 0
    bodies = {b["body"] for b in f["closure"]["bodies"]}
    assert bodies == {"winding:coil", "magnetic:slug"}
    assert abs(f["closure"]["relative"]) < 1e-2
    assert f["distributions"]["slug"]["kind"] == "magnetic"


def test_a_conductor_winding_keeps_both_distributions(tmp_path):
    """A winding region may be made of a normal conductor.  It must still be classified as a winding,
    or its volume distribution would overwrite the surface one under the same key."""
    doc = {"schema": "axionfem.design/1", "name": "cuw",
           "regions": [{"name": "coil", "material": "copper",
                        "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.05, 0.05]}}, "surface_size": 0.01}],
           "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
           "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02}}
    d = r = af.Design(doc).solve(workdir=tmp_path / "cuw", verbose=0)
    dists = r.forces["distributions"]
    assert dists["coil"]["geometry"] == "volume"
    assert dists["coil:boundary"]["geometry"] == "surface"
    assert dists["coil"]["kind"] == dists["coil:boundary"]["kind"] == "winding"
    assert sum(dists["coil"]["f_z_N"]) == pytest.approx(r.forces["coil"]["Fz_N"], abs=1e-6)
    assert d is r


def _ten_pack_design(n: int = 10):
    """A niobium box in an external field with n passive superconducting packs, each in its own persistent
    circuit at zero flux linkage."""
    regs = [{"name": "box", "material": "nb", "surface_size": 0.006,
             "shape": {"frame": {"r": [0.0, 0.25], "z": [-0.3, 0.3], "thickness": 0.01, "fillet": 0.03}}}]
    winds, circs = [], []
    for k in range(n):
        z = -0.45 + 0.1 * k
        regs.append({"name": f"w{k + 1}", "material": "winding", "surface_size": 0.008,
                     "shape": {"rect": {"r": [0.40, 0.45], "z": [z, z + 0.05]}}})
        winds.append({"name": f"w{k + 1}", "region": f"w{k + 1}", "type": "azimuthal",
                      "turns": 100, "circuit": f"c{k + 1}"})
        circs.append({"name": f"c{k + 1}", "type": "persistent", "flux_linkage": 0.0})
    return {"schema": "axionfem.design/1", "name": "tenpack", "regions": regs,
            "windings": winds, "circuits": circs,
            "superconductors": [{"body": "box", "azimuthal": {"flux": 0}, "poloidal": {"net_current": 0}}],
            "background": {"type": "uniform", "Bz": 0.05},
            "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02}}


def test_ten_passive_packs_in_persistent_circuits(tmp_path):
    """Ten passive packs in persistent circuits around a superconducting box.  Every pack must get a self-force
    estimate (not only the first few), every pack must get both distributions, and a zero-flux circuit that
    converged must not warn."""
    r = af.Design(_ten_pack_design()).solve(workdir=tmp_path / "pack10", verbose=0)
    f = r.forces
    packs = [f"w{k + 1}" for k in range(10)]
    assert all("Fz_self_N" in f[p] for p in packs)
    assert all("Fz_excluding_self_N" in f[p] for p in packs)
    d = f["distributions"]
    assert all(p in d and f"{p}:boundary" in d for p in packs)
    assert d["box"]["kind"] == "superconductor"
    # the packs sit symmetrically about z = 0, so the end packs are pushed in opposite directions
    assert f["w1"]["Fz_N"] * f["w10"]["Fz_N"] < 0
    # the closure is not valid here: the background field's source is outside the model
    chk = [c for c in r.data["checks"] if "Newton's third law" in c["name"]][0]
    assert chk["valid"] is False
    assert any("background" in c for c in f["closure"]["caveats"])
    # a persistent circuit at zero flux linkage cancels large contributions to reach its target;
    # judging the residual against the target alone warned on every single one of them
    assert not [w for w in r.data["warnings"] if "flux linkage" in w], r.data["warnings"]
    lk = [c for c in r.data["checks"] if c["name"].startswith("circuit flux linkage")]
    assert len(lk) == 10
    for c in lk:
        assert c["linkage_scale_Wb"] > 1.0        # the contributions really are O(1) Wb
        assert abs(c["relative"]) < 1e-9
        # the relaxed tolerance must not hide a real violation: a sign error in the constraint leaves a
        # residual of order the linkage scale itself, and the threshold here is 1e-6 of that, i.e. a few
        # microweber in absolute terms
        assert 1e-6 * c["linkage_scale_Wb"] < 1e-5


def test_poloidal_winding_axial_force_is_identically_zero(tmp_path):
    """An exact reference for the toroidal force path, which had none before.

        F_z = int J_r B_phi dV = -2 pi int (mu G / r^2) (dG/dz) r dr dz = -pi int (mu / r) [G^2] dr,

    and G vanishes above and below a closed poloidal winding, so the bracket is zero at every radius
    whatever the cross-section is -- here a trapezoidal ring, whose top and bottom legs differ in both
    thickness and radial extent.  So a nonzero Fz would be pure discretization error.
    """
    ring = {"difference": [{"polygon": {"points": [[0.6, -0.6], [1.2, -0.6], [0.9, 0.6], [0.6, 0.6]]}},
                           {"polygon": {"points": [[0.66, -0.5], [1.08, -0.5], [0.84, 0.5], [0.66, 0.5]]}}]}
    doc = {"schema": "axionfem.design/1", "name": "trap",
           "regions": [{"name": "winding", "material": "winding", "shape": ring, "surface_size": 0.012}],
           "windings": [{"name": "tor", "region": "winding", "type": "poloidal", "ampere_turns": 2e6}],
           "mesh": {"order": 3, "size": 0.04, "surface_size": 0.02}}
    r = af.Design(doc).solve(workdir=tmp_path / "trap", verbose=0)
    f = r.forces["tor"]
    hoop = abs(f["radial_force_per_radian_N"]) * 2 * math.pi
    assert hoop > 1e5                                   # the hoop load is the real force scale
    assert abs(f["Fz_N"]) < 1e-12 * hoop, (f["Fz_N"], hoop)
    # a single poloidal winding with no iron: the cancellation is exact, so no caveat about it
    caveats = " ".join(r.forces["closure"].get("caveats", []))
    assert "poloidal" not in caveats, caveats
    # nor about iron: a toroid has no poloidal static problem, so no iron force
    assert "nonlinear iron" not in caveats, caveats
    assert abs(r.forces["closure"]["relative"]) < 1e-12


def test_closure_warns_when_the_self_force_was_not_estimated(tmp_path):
    r = _loop_sphere(True).solve(workdir=tmp_path / "nsf2", verbose=0, self_force=False)
    caveats = " ".join(r.forces["closure"].get("caveats", []))
    assert "self-force floor" in caveats and "coil" in caveats


def test_self_force_can_be_switched_off(tmp_path):
    r = _loop_sphere(False).solve(workdir=tmp_path / "nsf", verbose=0, self_force=False)
    assert "Fz_self_N" not in r.forces["coil"]
    assert "Fz_excluding_self_N" not in r.forces["coil"]
