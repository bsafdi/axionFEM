"""Coarse 3D solves against the axisymmetric engine (fast smoke-level cross-checks).

Tight 3D-vs-2D agreement needs ND2 meshes and runs through benchmarks/m2_cross.py on SLURM;
these tests guard the plumbing: open boundary (infinity shell), port orientation, Newton for iron.
"""
import json
import math
import os
from pathlib import Path

import pytest

import axionfem as af
from axionfem import _core
from axionfem.convert import to_3d

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"], reason="needs the 3D engine (MPI build)")


def _run3d(d3, wd, order=1):
    return json.loads(_core.run_3d(json.dumps(d3.doc), str(wd), order, 0))


def test_solenoid_3d_vs_2d(tmp_path):
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    r2 = d2.solve(tmp_path / "s2", verbose=0)
    r3 = _run3d(to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 1}), tmp_path / "s3")
    c = r3["coupling"]
    assert r3["mesh"]["info"]["shell_attr"] > 0  # exact open boundary in use
    assert c["Psi_SI_H_T_m2"] > 0                # orientation convention: Psi >= 0 like 2D
    assert c["Psi_SI_H_T_m2"] == pytest.approx(r2.coupling["Psi_SI_H_T_m2"], rel=1e-2)
    assert c["L_p_H"] == pytest.approx(r2.coupling["L_p_H"], rel=1e-2)
    assert c["W_mag_J"] == pytest.approx(r2.coupling["W_mag_J"], rel=3e-2)
    # RF surface losses (pickup.losses): plumbing only at order 1, whose surface field is piecewise constant
    # (Gamma within ~20 %); test_surface_losses_3d_vs_2d_order2 checks the values
    l2, l3 = r2.data["pickup"]["losses"], r3["pickup"]["losses"]
    assert set(l3["surfaces"]) == set(l2["surfaces"])
    for k, v in l3["surfaces"].items():
        assert v["Gamma"] == pytest.approx(l2["surfaces"][k]["Gamma"], rel=0.3), k
    # the flat short sees a smooth 1/r field that order 1 already gets to ~1.3 %: a tighter check of the surface
    # measure (the inner and outer walls are -20 % and +15 %, and their T/Q errors partly cancel)
    assert l3["surfaces"]["coax_short"]["Gamma"] == pytest.approx(l2["surfaces"]["coax_short"]["Gamma"], rel=0.05)
    assert l3["T_over_Q_total_K"]["f_min"] == pytest.approx(l2["T_over_Q_total_K"]["f_min"], rel=0.1)


def test_surface_losses_magnetoresistance_3d_vs_2d(tmp_path):
    """The magnetoresistance weighting of the 3D pickup losses (static |B| on the field side) matches 2D. Kohler
    factor 1 + (B / 0.3 T)^2 on the coax copper, which sits in 0.3-0.45 T of the solenoid: the weight of R_s (skin)
    is about 1.4-1.8. The ratio Gamma_magnetoresistance / Gamma cancels most of the order-1 surface-field error."""
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    d2.doc.setdefault("materials", {})["cu_mr"] = {
        "kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-10, "T_K": 0.03,
                                    "magnetoresistance": {"model": "kohler", "B0_T": 0.3}}}
    for rg in d2.doc["regions"]:
        if rg["material"] == "copper":
            rg["material"] = "cu_mr"
    # an rf block on a non-conductor (here a mu_r = 1 linear ring, field-neutral) has no loss: both engines warn
    d2.doc["materials"]["lin_rf"] = {"kind": "linear", "mu_r": 1.0, "rf": {"model": "power", "Rs_ohm": 1e-3, "T_K": 4}}
    d2.doc["regions"].append({"name": "lin_ring", "shape": {"rect": {"r": [1.3, 1.4], "z": [-0.1, 0.1]}},
                              "material": "lin_rf"})
    r2 = d2.solve(tmp_path / "s2", verbose=0)
    l2 = r2.data["pickup"]["losses"]["surfaces"]
    r3 = _run3d(to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 1}), tmp_path / "s3")
    l3 = r3["pickup"]["losses"]["surfaces"]
    for w in (r2.data["warnings"], r3["warnings"]):
        assert any("'lin_rf' has an rf block but is not a conductor" in s for s in w), w
    assert set(l3) == set(l2)
    for k, v in l3.items():
        w2 = l2[k]["Gamma_magnetoresistance"] / l2[k]["Gamma"]
        w3 = v["Gamma_magnetoresistance"] / v["Gamma"]
        assert w2 > 1.2, k  # the weighting is not trivially 1
        assert w3 == pytest.approx(w2, rel=0.05), k
        assert v["R_eff_ohm"]["f_min"] / v["Gamma"] == pytest.approx(
            l2[k]["R_eff_ohm"]["f_min"] / l2[k]["Gamma"], rel=0.05), k


@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="ND2 3D solve (slow); set AXF_SLOW=1")
def test_surface_losses_3d_vs_2d_order2(tmp_path):
    """The 3D pickup surface losses reproduce the 2D engine's at order 2 (solenoid template: copper coax with the
    builtin rf skin block). Coarse ND2 (436k DOFs): Gamma within 1.5 %, Q and T/Q likewise. The finer mesh and the
    closed form are in benchmarks/rf_losses_3d.py (record benchmarks/records/rf_losses_3d.json)."""
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    l2 = d2.solve(tmp_path / "s2", verbose=0).data["pickup"]["losses"]
    r3 = _run3d(to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 2}), tmp_path / "s3", order=2)
    l3 = r3["pickup"]["losses"]
    for k, v in l3["surfaces"].items():
        ref = l2["surfaces"][k]
        assert v["Gamma"] == pytest.approx(ref["Gamma"], rel=2e-2), k
        for edge in ("f_min", "f_max"):
            assert v["Q"][edge] == pytest.approx(ref["Q"][edge], rel=2e-2), (k, edge)
    assert l3["T_over_Q_total_K"]["f_min"] == pytest.approx(l2["T_over_Q_total_K"]["f_min"], rel=2e-2)


def test_saturated_iron_newton_3d(tmp_path):
    doc = {"schema": "axionfem.design/1", "name": "fe",
           "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.2, 0.2]}}, "material": "winding"},
                       {"name": "core", "shape": {"rect": {"r": [0.0, 0.4], "z": [-0.3, 0.3]}}, "material": "iron"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 3e6}],
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.01}}
    d2 = af.Design(doc)
    W2 = d2.solve(tmp_path / "f2", verbose=0).data["static"]["poloidal"]["energy_J"]
    r3 = _run3d(to_3d(d2, mesh={"size": 0.3, "surface_size": 0.15, "order": 1}), tmp_path / "f3")
    nl = r3["static"]["nonlinear"]
    assert nl["converged"] and nl["newton_iterations"] <= 8
    assert nl["relative_residual"] < 1e-9
    assert r3["static"]["energy_J"] == pytest.approx(W2, rel=0.12)  # ND1 at h = 0.3 is coarse
    assert not r3["warnings"]


@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow (about 5-15 min); set AXF_SLOW=1")
def test_persistent_circuit_3d(tmp_path):
    """A zero-field-cooled 100-turn loop inside a solenoid: the 3D current matches the axisymmetric one."""
    doc = {"schema": "axionfem.design/1", "name": "circ",
           "regions": [{"name": "sol", "shape": {"rect": {"r": [1.0, 1.05], "z": [-1, 1]}}, "material": "winding"},
                       {"name": "pc", "shape": {"rect": {"r": [0.5, 0.55], "z": [0.3, 0.4]}}, "material": "winding",
                        "mesh_size": 0.025}],
           "windings": [{"name": "sol", "region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
                        {"name": "pc", "region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}],
           "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": 0}],
           "mesh": {"order": 2, "size": 0.08, "surface_size": 0.02}}
    r2 = af.Design(doc).solve(tmp_path / "c2", verbose=0)
    I2 = r2.data["static"]["poloidal"]["solver"]["circuits"][0]["current_A"]
    # the loop's self-inductance needs its cross-section resolved: local mesh_size, second order
    r3 = to_3d(af.Design(doc), mesh={"size": 0.4, "surface_size": 0.12, "order": 2}).solve(tmp_path / "c3", verbose=0)
    c3 = r3.data["static"]["circuits"][0]
    assert c3["current_A"] == pytest.approx(I2, rel=0.03)
    assert abs(c3["flux_linkage_Wb"]) < 1e-9 * abs(c3["current_A"])
    assert not r3.data["warnings"]
    assert "pc" in r3.data["forces"]["windings"]


# ---------------------------------------------------------------- input errors, warnings and regressions
def _ring(r0, r1, z0, z1, **kw):
    return {"revolve": {"profile": {"rect": {"r": [r0, r1], "z": [z0, z1]}}, **kw}}


def _doc3(regions, windings, **extra):
    return {"schema": "axionfem.design/1", "name": "t3", "symmetry": "3d", "regions": regions, "windings": windings,
            "mesh": {"size": 0.15, "surface_size": 0.08, "order": 1}, **extra}


_COIL = {"name": "coil", "solid": _ring(0.5, 0.6, -0.05, 0.05), "material": "winding"}
_WIND = [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}]


def test_port_between_two_conductors_is_an_error(tmp_path):
    plates = [{"name": "p1", "solid": {"box": {"min": [0.1, -0.2, -0.3], "max": [0.15, 0.2, 0.3]}}, "material": "copper"},
              {"name": "p2", "solid": {"box": {"min": [0.25, -0.2, -0.3], "max": [0.3, 0.2, 0.3]}}, "material": "copper"},
              {"name": "gap", "solid": {"box": {"min": [0.15, -0.2, 0.25], "max": [0.25, 0.2, 0.3]}}, "material": "vacuum"}]
    doc = _doc3([_COIL] + plates, _WIND, pickup={"type": "port", "port": "gap"})
    with pytest.raises(Exception, match="two different conductors"):
        _run3d(af.Design(doc), tmp_path / "p")


def test_poloidal_winding_on_solid_ring_is_an_error(tmp_path):
    doc = _doc3([_COIL], [{"name": "w", "region": "coil", "type": "poloidal", "ampere_turns": 1e4}])
    with pytest.raises(Exception, match="no loop around its cross-section"):
        _run3d(af.Design(doc), tmp_path / "pw")


def test_open_azimuthal_winding_warns(tmp_path):
    half = {"name": "coil", "solid": _ring(0.5, 0.6, -0.05, 0.05, angles=[0, 180]), "material": "winding"}
    r = _run3d(af.Design(_doc3([half], _WIND)), tmp_path / "half")
    assert any("not divergence-free" in w for w in r["warnings"])


def test_curves_are_rejected_in_3d(tmp_path):
    doc = _doc3([_COIL], _WIND, curves=[{"name": "s", "path": {"segment": [[0.0, 0.3], [0.8, 0.3]]},
                                          "material": "superconductor"}])
    with pytest.raises(Exception, match="curves"):
        _run3d(af.Design(doc), tmp_path / "c")


def test_constraint_on_tilted_ring_is_an_error(tmp_path):
    ring = {"name": "ring", "solid": _ring(0.3, 0.4, 0.0, 0.1, axis=[1, 0, 0]), "material": "superconductor"}
    coil = {"name": "coil", "solid": _ring(0.5, 0.6, -0.15, -0.05, axis=[1, 0, 0]), "material": "winding"}
    wind = [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4, "axis": {"direction": [1, 0, 0]}}]
    doc = _doc3([ring, coil], wind, superconductors=[{"body": "ring", "azimuthal": {"flux": 0}}])
    with pytest.raises(Exception, match="not loops about the z axis"):
        _run3d(af.Design(doc), tmp_path / "tilt")


def test_native_azimuthal_pickup_and_reproducible_mesh(tmp_path):
    d2 = af.template("toroid", R=1.0, H=1.0, tw=0.06, g=0.04, ts=0.05, f=0.06,
                     mesh={"order": 2, "size": 0.04, "surface_size": 0.015})
    r2 = d2.solve(tmp_path / "t2", verbose=0)
    d3 = to_3d(d2, mesh={"size": 0.45, "surface_size": 0.18, "order": 1})
    assert d3.doc["pickup"] == {"type": "azimuthal", "conductor": "sheath"}
    ra = _run3d(d3, tmp_path / "ta")
    rb = _run3d(d3, tmp_path / "tb")
    assert ra["mesh"]["dofs"] == rb["mesh"]["dofs"]                                    # deterministic meshing
    assert ra["coupling"]["W_mag_J"] == pytest.approx(rb["coupling"]["W_mag_J"], rel=1e-12)
    p = ra["pickup"]
    assert p["kind"] == "azimuthal"
    assert p["L_p_from_flux_H"] == pytest.approx(p["L_p_H"], rel=1e-6)              # energy vs fluxoid
    assert abs(ra["checks"]["Psi_reciprocity_rel_diff"]) < 1e-10
    assert ra["coupling"]["Psi_SI_H_T_m2"] == pytest.approx(r2.coupling["Psi_SI_H_T_m2"], rel=0.08)  # ND1, coarse
    assert ra["coupling"]["L_p_H"] == pytest.approx(r2.coupling["L_p_H"], rel=0.06)
    assert not ra["warnings"]


@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow (ND2, about 7 min); set AXF_SLOW=1")
def test_sample_beyond_the_shell(tmp_path):
    # isolated thin coil: the on-axis far field is that of a current loop (radius a, current NI); ND1 is too
    # coarse inside the mapped shell (B piecewise constant). With ND2, the volume size also sets the shell's
    # elements: size 0.08 gives < 1 % out to 3 Rb (0.03 % there), size 0.12 is 2.6 % off at 3 Rb
    a, NI = 0.55, 1e4
    coil = {"name": "coil", "solid": _ring(0.54, 0.56, -0.01, 0.01), "material": "winding"}
    doc = _doc3([coil], [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": NI}])
    doc["mesh"] = {"size": 0.08, "surface_size": 0.02, "order": 2}
    d3 = af.Design(doc)
    res = d3.solve(tmp_path / "s", verbose=0)
    Rb = res.data["mesh"]["info"]["shell_inner_radius"]
    zs = [1.1 * Rb, 3.0 * Rb]
    B = res.sample([0.0, 0.0], [0.0, 0.0], zs)
    for z, b in zip(zs, B["Bz"]):
        exact = 4e-7 * math.pi * NI * a * a / (2 * (a * a + z * z) ** 1.5)
        assert b == pytest.approx(exact, rel=0.01)


def test_background_field_superconducting_sphere_and_ring_3d(tmp_path):
    """Uniform background B0 z: a superconducting sphere (Meissner) has the surface field 3/2 B0 sin(theta);
    a zero-flux ring matches the axisymmetric engine."""
    B0 = 0.1
    sph = {"name": "ball", "solid": {"sphere": {"center": [0, 0, 0], "radius": 0.2}}, "material": "superconductor"}
    doc = _doc3([sph], [], background={"type": "uniform", "Bz": B0})
    doc["mesh"] = {"size": 0.1, "surface_size": 0.025, "order": 2}
    r = af.Design(doc).solve(tmp_path / "ball", verbose=0)
    peak = r.data["fields"]["superconductor_surface_regular_T"]["ball"]
    assert peak == pytest.approx(1.5 * B0, rel=0.05)
    s = r.sample([0.0, 0.0], [0.0, 0.0], [0.6, 0.21])
    assert s["Bz"][0] == pytest.approx(B0 * (1 - (0.2 / 0.6) ** 3), rel=0.01)  # on the axis: B0 (1 - a^3/r^3)
    # ring with zero fluxoid in the background (zero-field cooled, field applied): 3D vs 2D net current
    ring2 = {"name": "ring", "shape": {"rect": {"r": [0.3, 0.34], "z": [-0.02, 0.02]}}, "material": "superconductor"}
    d2 = af.Design({"schema": "axionfem.design/1", "name": "ringbg", "regions": [ring2],
                    "superconductors": [{"body": "ring", "azimuthal": {"flux": 0}}],
                    "background": {"type": "uniform", "Bz": B0},
                    "mesh": {"order": 2, "size": 0.03, "surface_size": 0.006}})
    r2 = d2.solve(tmp_path / "r2", verbose=0)
    d3 = to_3d(d2, mesh={"size": 0.15, "surface_size": 0.03, "order": 2})
    r3 = d3.solve(tmp_path / "r3", verbose=0)
    I3 = r3.data["static"]["bodies"][0]["generators"][0]["net_current_A"]
    I2 = [b for b in r2.data["static"]["poloidal"]["bodies"] if "ring" in b["names"]][0]["net_azimuthal_current_A"]
    print("ring current 3D", I3, "2D", I2)
    assert I3 == pytest.approx(I2, rel=0.03)


def test_magnetic_material_force_balances_coil_force_3d(tmp_path):
    """An axially offset magnetic core is pulled into the coil; the coil feels the opposite force (closure)."""
    coil = {"name": "coil", "solid": _ring(0.3, 0.36, -0.1, 0.1), "material": "winding"}
    core = {"name": "core", "solid": {"cylinder": {"base": [0, 0, 0.05], "axis": [0, 0, 0.3], "radius": 0.15}},
            "material": "mag"}
    doc = _doc3([coil, core], [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
                materials={"mag": {"kind": "linear", "mu_r": 50}})
    doc["mesh"] = {"size": 0.1, "surface_size": 0.015, "order": 2}
    r = af.Design(doc).solve(tmp_path / "core", verbose=0)
    f = r.data["forces"]
    Fc = f["magnetic_materials"]["core"]["F_N"][2]
    Fw = f["windings"]["w"]["F_N"][2]
    print("core", Fc, "coil", Fw, "sum", f["sum_F_N"])
    assert Fc < 0                                  # pulled towards the coil centre (-z)
    assert Fw == pytest.approx(-Fc, rel=0.05)      # Newton's third law (Maxwell stress converges with h)


def test_loop_winding_square_frame_3d(tmp_path):
    """A square frame coil ('loop' winding: harmonic current around its single loop): B at the centre matches
    the thin square loop sqrt(2) mu0 I / (pi a) (a: half side) and points along axis.direction."""
    outer = {"box": {"min": [-0.5, -0.5, -0.02], "max": [0.5, 0.5, 0.02]}}
    inner = {"box": {"min": [-0.45, -0.45, -0.03], "max": [0.45, 0.45, 0.03]}}
    frame = {"name": "frame", "solid": {"difference": [outer, inner]}, "material": "winding"}
    NI = 1e4
    for sgn in (1, -1):
        doc = _doc3([frame], [{"name": "sq", "region": "frame", "type": "loop", "ampere_turns": NI,
                               "axis": {"direction": [0, 0, sgn]}}])
        doc["mesh"] = {"size": 0.12, "surface_size": 0.02, "order": 2}
        r = af.Design(doc).solve(tmp_path / f"sq{sgn}", verbose=0)
        a = 0.475
        exact = math.sqrt(2) * 4e-7 * math.pi * NI / (math.pi * a)
        Bz = r.sample([0.0], [0.0], [0.0])["Bz"][0]
        print("square loop Bz", Bz, "thin-loop", sgn * exact)
        assert Bz == pytest.approx(sgn * exact, rel=0.03)  # finite width: the current crowds at the inner corners
        assert not [w for w in r.data["warnings"] if "divergence" in w]



def test_superconductor_axis_for_tilted_ring(tmp_path):
    """A ring about the x axis takes constraints once its axis is given; its induced current
    equals that of the same ring about z (the configuration is rotated as a whole)."""
    cur = {}
    for name, axis in (("z", [0, 0, 1]), ("x", [1, 0, 0])):
        ring = {"name": "ring", "solid": _ring(0.3, 0.4, 0.0, 0.1, axis=axis), "material": "superconductor"}
        coil = {"name": "coil", "solid": _ring(0.5, 0.6, -0.15, -0.05, axis=axis), "material": "winding"}
        wind = [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4, "axis": {"direction": axis}}]
        sc = [{"body": "ring", "azimuthal": {"flux": 0}}]
        if name == "x":
            sc[0]["axis"] = {"direction": axis}
        doc = _doc3([ring, coil], wind, superconductors=sc)
        doc["mesh"] = {"size": 0.15, "surface_size": 0.04, "order": 2}
        r = af.Design(doc).solve(tmp_path / name, verbose=0)
        cur[name] = r.data["static"]["bodies"][0]["generators"][0]["net_current_A"]
    print(cur)
    assert cur["x"] == pytest.approx(cur["z"], rel=0.03)


# ---------------------------------------------------------------- winding and circuit axes
def test_loop_moment_perpendicular_to_axis_is_an_error(tmp_path):
    """A frame in the xz plane has its moment along y: with the default axis (z) the orientation is undefined."""
    outer = {"box": {"min": [-0.5, -0.02, -0.5], "max": [0.5, 0.02, 0.5]}}
    inner = {"box": {"min": [-0.45, -0.03, -0.45], "max": [0.45, 0.03, 0.45]}}
    frame = {"name": "frame", "solid": {"difference": [outer, inner]}, "material": "winding"}
    doc = _doc3([frame], [{"name": "sq", "region": "frame", "type": "loop", "ampere_turns": 1e4}])
    with pytest.raises(Exception, match="nearly perpendicular to axis.direction"):
        af.Design(doc).solve(tmp_path / "perp", verbose=0)


def test_circuit_winding_axis_is_used(tmp_path):
    """`axis` of a winding in a circuit is parsed: the persistent loop about x in a
    field along x carries the same current as the loop about z in a field along z."""
    def doc(axis):
        sol = {"name": "sol", "solid": _ring(1.0, 1.05, -1, 1, axis=axis), "material": "winding"}
        pc = {"name": "pc", "solid": _ring(0.5, 0.55, 0.3, 0.4, axis=axis), "material": "winding"}
        return {"schema": "axionfem.design/1", "name": "ax", "symmetry": "3d", "regions": [sol, pc],
                "windings": [{"name": "sol", "region": "sol", "type": "azimuthal", "ampere_turns": 1e5,
                              "axis": {"direction": axis}},
                             {"name": "pc", "region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop",
                              "axis": {"direction": axis}}],
                "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": 0}],
                "mesh": {"size": 0.4, "surface_size": 0.12, "order": 1}}
    Iz = af.Design(doc([0, 0, 1])).solve(tmp_path / "z", verbose=0).data["static"]["circuits"][0]["current_A"]
    Ix = af.Design(doc([1, 0, 0])).solve(tmp_path / "x", verbose=0).data["static"]["circuits"][0]["current_A"]
    print("I about z", Iz, "about x", Ix)
    assert abs(Iz) > 10
    assert Ix == pytest.approx(Iz, rel=0.05)
def test_gauge_regularization_does_not_change_the_results(tmp_path, monkeypatch):
    """The gauge mass eps_e = kappa nu_e/h_e^2 of the curl-curl operator is a solver
    device. Against an exactly gauge-free solve (eps = 0 with AMS's singular mode) it must not change Psi,
    L_p or W; both must converge. See docs/validation/ams_robustness.md."""
    ring = {"name": "ring", "solid": _ring(0.3, 0.4, -0.05, 0.05), "material": "superconductor",
            "mesh_size": 0.05}
    doc = _doc3([_COIL, ring], _WIND, superconductors=[{"body": "ring", "azimuthal_slit": True}],
                pickup={"type": "azimuthal", "conductor": "ring"})
    doc["mesh"] = {"size": 0.2, "surface_size": 0.06, "order": 1}
    out = {}
    residual = 0.0
    for name, env in (("gauge", {}), ("gaugefree", {"AXF_GAUGE": "0", "AXF_AMS_SINGULAR": "1"})):
        with monkeypatch.context() as m:
            for k, v in env.items():
                m.setenv(k, v)
            r = _run3d(af.Design(doc), tmp_path / name)
        assert r["solver"]["max_relative_residual"] < 1e-8, (name, r["warnings"])
        assert not [w for w in r["warnings"] if "NOT reliable" in w], (name, r["warnings"])
        residual = max(residual, r["solver"]["max_relative_residual"])
        out[name] = r["coupling"]
    # Psi is zero by symmetry here: the winding is azimuthal, so B_lab is poloidal, and A_pick is azimuthal, so
    # A_pick . B_lab vanishes pointwise (the axisymmetric twin gives Psi = 0.0). The 3D value is the mesh's symmetry
    # breaking (of order 1e-13, and it differs between machines whose meshes differ), so a tolerance relative to Psi
    # scales with mesh noise: rel=2e-8 has failed at 1.2e-7 on another machine. Psi is compared on its natural scale
    # instead, Psi_nat = mu0 R sqrt(L_p W) (= Psi / sqrt(g)), where the solver's algebraic error is O(residual)
    # whatever the cancellation. Measured with the default AXF_SOLVER_TOL (1e-8): |dPsi| / Psi_nat / residual
    # 0.014-0.40. Bound: 10 x the larger residual (floor 1e-12).
    # L_p and W are energies, stationary in the field: they stay at rel=2e-8 (the gauge mass moves L_p by about
    # 1.4e-9, at every solver tolerance). A gauge defect would show there. All three are compared before asserting.
    ref = out["gaugefree"]
    psi_nat = 4e-7 * math.pi * ref["device_radius_m"] * math.sqrt(ref["L_p_H"] * ref["W_mag_J"])
    d_psi = abs(out["gauge"]["Psi_SI_H_T_m2"] - ref["Psi_SI_H_T_m2"]) / psi_nat
    checks = {"Psi_SI_H_T_m2 / Psi_nat": (d_psi, 10 * max(residual, 1e-12))}
    for k in ("L_p_H", "W_mag_J"):
        checks[k] = (abs(out["gauge"][k] - ref[k]) / abs(ref[k]), 2e-8)
    for k, (diff, bound) in checks.items():
        key = k.split(" ")[0]
        print(k, "gauge", out["gauge"][key], "gauge-free", ref[key], "difference", diff, "bound", bound)
    bad = {k: v for k, v in checks.items() if not v[0] < v[1]}
    assert not bad, (bad, "max residual", residual)


@pytest.mark.skipif(os.environ.get("AXF_SLOW") != "1", reason="two-body 3D constraint regression: AXF_SLOW=1")
def test_multiple_free_body_currents_with_inexact_lifts(tmp_path, monkeypatch):
    """Independent inexact lift solves must not cause one triangle of R to be discarded.

    The field tolerance is deliberately loose: this tests exact enforcement of the
    computed small reaction system, not physical field accuracy at that tolerance.
    """
    monkeypatch.setenv("AXF_GAUGE", "1e-8")
    monkeypatch.setenv("AXF_SOLVER_TOL", "1e-5")
    rings = [{"name": name, "solid": _ring(.25, .35, z-.05, z+.05), "material": "superconductor"}
             for name,z in (("lower", -.15), ("upper", .18))]
    targets = {"lower":11., "upper":-7.}
    doc = _doc3(rings, [], background={"type":"uniform", "Bz":.02},
                superconductors=[{"body":name, "azimuthal":{"net_current":value}}
                                 for name,value in targets.items()])
    doc['mesh'] = {"size":.25, "surface_size":.08, "order":2}
    result = af.Design(doc).solve(tmp_path/'two_loops', verbose=0).data
    (tmp_path/'constraint_diagnostics.json').write_text(json.dumps(dict(
        solver=result['solver'], bodies=result['static']['bodies'], mesh=result['mesh']), indent=2))
    for body in result['static']['bodies']:
        current = body['generators'][0]['net_current_A']
        print(body['names'], 'current', current, 'target', targets[body['names'][0]])
        assert current == pytest.approx(targets[body['names'][0]], abs=1e-7)
    systems = result['solver']['body_constraint_solves']
    assert len(systems) == 1
    system = systems[0]
    assert system['method'] == 'full-pivot LU'
    assert system['backward_error'] < 1e-12
    assert system['relative_asymmetry'] > 1e-12
    for body in result['static']['bodies']:
        g = body['generators'][0]
        assert g['net_current_A'] == pytest.approx(targets[body['names'][0]], abs=1e-7)
        assert abs(g['constraint_reaction_residual_A']) < max(1e-7, 1e-12*g['reaction_abs_sum_A'])


# ---------------------------------------------------------------- wedge_frame / AXF_MESHER_SURFACE_DUMP
def _wedge_doc(region_name="case"):
    """A coil ring and a wedge_frame coil case whose dimensions are named-parameter expressions."""
    wedge = {"name": region_name, "material": "copper",
             "solid": {"wedge_frame": {"polygon": [["x0", "-h"], ["x0 + 0.1", "-h"], ["x0 + 0.1", "h"], ["x0", "h"]],
                                       "hole": [["x0 + 0.03", "-h/2"], ["x0 + 0.07", "-h/2"],
                                                ["x0 + 0.07", "h/2"], ["x0 + 0.03", "h/2"]],
                                       "tan_a": "t", "c_pos": "c", "c_neg": "c"}}}
    doc = _doc3([_COIL, wedge], _WIND)
    doc["parameters"] = {"x0": 0.2, "h": 0.05, "t": 0.2, "c": "x0/10"}
    return doc


def test_wedge_frame_parameter_expressions_are_evaluated(tmp_path):
    """Regression: expression strings in a wedge_frame must be evaluated before they reach axf-mesher."""
    res = json.loads(_core.resolve(json.dumps(_wedge_doc())))
    wf = next(r for r in res["regions"] if r["name"] == "case")["solid"]["wedge_frame"]
    assert wf["tan_a"] == pytest.approx(0.2) and wf["c_pos"] == pytest.approx(0.02) and wf["c_neg"] == pytest.approx(0.02)
    assert wf["polygon"] == [pytest.approx(p) for p in ([0.2, -0.05], [0.3, -0.05], [0.3, 0.05], [0.2, 0.05])]
    assert wf["hole"][1] == pytest.approx([0.27, -0.025])
    out = json.loads(_core.mesh(json.dumps(_wedge_doc()), str(tmp_path / "m"), 0))  # the mesher accepts it
    assert os.path.exists(out["mesh"])
    bad = _wedge_doc()
    bad["regions"][1]["solid"]["wedge_frame"]["tan_a"] = "nosuchparam"
    with pytest.raises(Exception, match="tan_a"):
        _core.resolve(json.dumps(bad))
    narrow = _wedge_doc()
    narrow["parameters"]["c"] = 0.05  # 2 t x - 2 c < 0 at x = 0.2
    with pytest.raises(Exception, match="not positive"):
        _core.resolve(json.dumps(narrow))


def test_surface_dump_escapes_region_names(tmp_path, monkeypatch):
    """Regression: AXF_MESHER_SURFACE_DUMP must escape region names (unescaped, the file is invalid JSON)."""
    dump = tmp_path / "faces.json"
    monkeypatch.setenv("AXF_MESHER_SURFACE_DUMP", str(dump))
    name = 'case "A"\\1'
    _core.mesh(json.dumps(_wedge_doc(name)), str(tmp_path / "m"), 0)
    faces = json.loads(dump.read_text())
    assert faces and all(len(f["bbox"]) == 6 for f in faces)
    assert any(name in f["volumes"].split(";") for f in faces)
    # an unwritable path: the mesh is still made, and axf-mesher warns on stderr (captured in its log)
    monkeypatch.setenv("AXF_MESHER_SURFACE_DUMP", str(tmp_path / "no_such_dir" / "faces.json"))
    out = json.loads(_core.mesh(json.dumps(_wedge_doc()), str(tmp_path / "m2"), 0))
    assert os.path.exists(out["mesh"])
    assert "AXF_MESHER_SURFACE_DUMP: cannot write" in (tmp_path / "m2" / "mesh3d_mesher.log").read_text()


def test_region_names_that_meet_in_the_mesh_file_are_refused(tmp_path):
    """The mesh file writes a double quote of a region name as a single quote (MFEM's reader cannot read one), so
    'q"x' and "q'x" would get one physical name, and gmsh would drop the second."""
    doc = _wedge_doc('q"x')
    doc["regions"][0]["name"] = "q'x"
    doc["windings"][0]["region"] = "q'x"
    with pytest.raises(Exception, match="rename one of them"):
        _core.mesh(json.dumps(doc), str(tmp_path / "m"), 0)
    doc["regions"][0]["name"] = "q`x"  # distinct in the mesh file too: meshed, the names as written
    doc["windings"][0]["region"] = "q`x"
    out = json.loads(_core.mesh(json.dumps(doc), str(tmp_path / "m2"), 0))
    text = Path(out["mesh"]).read_bytes()
    assert b'"q\'x"' in text and b'"q`x"' in text
