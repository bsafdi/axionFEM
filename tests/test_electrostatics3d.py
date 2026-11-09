"""3D electrostatics (design key `electrostatics` on a "3d" design) against closed forms.

Fast by default (coarse meshes, a few seconds each):
- isolated sphere: C = 4 pi eps0 a, with the monopole carried to infinity by the infinity shell
- sphere inside a grounded sphere
- the schema, the result keys, the Python API and the error messages

AXF_SLOW adds the quantitative cases (finer meshes, minutes): the concentric-sphere matrix, the
published unit-cube value, two spheres that lie on no coordinate axis against the bispherical series,
the readout-port gap capacitance, and 3D against the axisymmetric engine on a revolved design.
AXF_MPI adds the rank-independence check.  The full set with records is
benchmarks/electrostatics3d_validation.py.
"""
import json
import math
import os

import pytest

import axionfem as af
from axionfem.convert import to_3d

MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * 299792458.0**2)
FPE = 4 * math.pi * EPS0
CUBE_FACTOR = 0.66067813  # C/(4 pi eps0 L) of a cube, published value

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow 3D solve; set AXF_SLOW=1")
needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build; the wheels have none)")


def ball(r, center=(0, 0, 0)):
    return {"sphere": {"center": list(center), "radius": r}}


def hollow(r_in, r_out):
    return {"difference": [ball(r_out), ball(r_in)]}


def solve(tmp_path, name, regions, materials=None, mesh=None, boundary="open", ranks=1, extra=None):
    doc = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d", "regions": regions,
           "electrostatics": {"boundary": boundary},
           "mesh": mesh or {"order": 2, "size": 0.35, "surface_size": 0.12, "domain_radius": 1.2}}
    if materials:
        doc["materials"] = materials
    if extra:
        doc.update(extra)
    return af.Design(doc).solve(tmp_path / name, verbose=0, mpi_ranks=ranks)


def body(e, name):
    return [b["name"] for b in e["bodies"]].index(name)


def assert_consistent(e, sym=1e-8, energy=1e-3, gauss=0.2, mono=0.2, slack=-1e-6):
    """The algebraic checks (symmetry, energy, superposition, the independent mixed-potential solve) are
    tight; the surface-flux and exterior-flux checks are discretization checks and converge with h."""
    c = e["checks"]
    assert c["symmetry_relative"] < sym
    assert c["energy_max_relative_difference"] < energy
    assert c["energy_superposition"]["relative_difference"] < energy
    assert c["mixed_potential_solve"]["max_relative_difference"] < sym
    assert c["charge_gauss_max_relative_difference"] < gauss
    assert c["exterior_monopole_max_relative_difference"] < mono
    assert c["off_diagonal_signs_ok"]
    assert c["min_slack_relative"] > slack
    assert e["solver"]["max_relative_residual"] < 1e-8


# ------------------------------------------------------------------ fast
@needs_3d
def test_isolated_sphere_3d(tmp_path):
    a = 0.5
    r = solve(tmp_path, "sphere", [{"name": "ball", "solid": ball(a), "material": "superconductor"}])
    e = r.data["electrostatics"]
    assert e["boundary"] == "open"
    assert [b["name"] for b in e["bodies"]] == ["ball"]
    assert e["solver"]["mpi_ranks"] == 1 and e["solver"]["order"] == 2
    # a coarse ND2/H1-2 mesh on a curved sphere: the geometry dominates the error
    assert e["capacitance_F"][0][0] == pytest.approx(FPE * a, rel=5e-3)
    assert e["capacitance_to_ground_F"][0] == pytest.approx(FPE * a, rel=5e-3)
    assert_consistent(e)
    assert not [w for w in r.data["warnings"] if "electrostatics" in w or "NOT reliable" in w]
    # the Python API is the one of the axisymmetric solver
    assert r.C_matrix == e["capacitance_F"]
    assert r.electrostatics is e or r.electrostatics == e
    assert "C[ball]" in r.summary()
    assert r.port_capacitance == {}
    assert r.data["timing"]["electrostatics_s"] > 0
    # a design with no magnet and no pickup still renders (as in 2D)
    assert r.scene()["schema"].startswith("axionfem.scene")
    # a `conductor` material is a conductor here too
    r2 = solve(tmp_path, "sphere_cu", [{"name": "ball", "solid": ball(a), "material": "copper"}])
    assert r2.data["electrostatics"]["capacitance_F"][0][0] == pytest.approx(FPE * a, rel=5e-3)


@needs_3d
def test_grounded_sphere_3d(tmp_path):
    a, Rb = 0.3, 1.0
    r = solve(tmp_path, "grounded", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
              boundary="grounded", mesh={"order": 2, "size": 0.25, "surface_size": 0.08, "domain_radius": Rb})
    e = r.data["electrostatics"]
    assert e["boundary"] == "grounded"
    assert e["capacitance_F"][0][0] == pytest.approx(FPE / (1 / a - 1 / Rb), rel=1e-2)
    # the shell is out of the domain, so there is no exterior flux to compare
    assert e["checks"]["exterior_monopole_max_relative_difference"] == 0.0
    assert_consistent(e)
    # grounding is a smaller capacitance than the open boundary at the same radius
    r_open = solve(tmp_path, "open", [{"name": "ball", "solid": ball(a), "material": "superconductor"}],
                   mesh={"order": 2, "size": 0.25, "surface_size": 0.08, "domain_radius": Rb})
    assert e["capacitance_F"][0][0] > r_open.data["electrostatics"]["capacitance_F"][0][0]


@needs_3d
def test_dielectric_region_3d(tmp_path):
    """A dielectric shell raises the capacitance of the sphere it covers (layered capacitor)."""
    a, b, eps_r = 0.3, 0.5, 4.0
    mesh = {"order": 2, "size": 0.25, "surface_size": 0.08, "domain_radius": 1.2}
    r = solve(tmp_path, "diel", [{"name": "ball", "solid": ball(a), "material": "superconductor"},
                                 {"name": "diel", "solid": hollow(a, b), "material": "dielectric"}],
              materials={"dielectric": {"kind": "linear", "eps_r": eps_r}}, mesh=mesh)
    e = r.data["electrostatics"]
    exact = FPE / ((1 / a - 1 / b) / eps_r + 1 / b)
    assert e["capacitance_F"][0][0] == pytest.approx(exact, rel=2e-2)
    assert_consistent(e)


@needs_3d
def test_electrostatics_3d_needs_a_conductor(tmp_path):
    doc = {"schema": "axionfem.design/1", "name": "nc3", "symmetry": "3d",
           "regions": [{"name": "blob", "solid": ball(0.2), "material": "vacuum"}],
           "electrostatics": {},
           "mesh": {"order": 1, "size": 0.3, "domain_radius": 1.0}}
    with pytest.raises(Exception, match="conductor"):
        af.Design(doc).solve(tmp_path / "nc3", verbose=0)


def test_3d_electrostatics_is_accepted_by_the_schema():
    doc = {"schema": "axionfem.design/1", "name": "ok3", "symmetry": "3d",
           "regions": [{"name": "ball", "solid": ball(0.2), "material": "superconductor"}],
           "electrostatics": {"boundary": "open", "notes": "3D"}}
    af.Design(doc).validate()
    doc["electrostatics"] = {"boundary": "nonsense"}
    with pytest.raises(Exception, match="open"):
        af.Design(doc).validate()


# ------------------------------------------------------------------ slow / quantitative
@slow
def test_concentric_spheres_3d(tmp_path):
    a, b, c = 0.3, 0.5, 0.6
    r = solve(tmp_path, "concentric",
              [{"name": "inner", "solid": ball(a), "material": "superconductor"},
               {"name": "outer", "solid": hollow(b, c), "material": "superconductor"}],
              mesh={"order": 2, "size": 0.1, "surface_size": 0.03, "domain_radius": 1.6})
    e = r.data["electrostatics"]
    i, o = body(e, "inner"), body(e, "outer")
    C = e["capacitance_F"]
    Cin = FPE / (1 / a - 1 / b)
    assert C[i][i] == pytest.approx(Cin, rel=5e-3)
    assert C[i][o] == pytest.approx(-Cin, rel=5e-3)
    assert C[o][o] == pytest.approx(Cin + FPE * c, rel=5e-3)
    # the inner conductor is fully screened, so its diagonal-dominance slack vanishes
    assert_consistent(e, gauss=0.06, mono=0.03, slack=-1e-3)
    pair = e["pairs"][0]
    assert pair["C_mutual_others_grounded_F"] == pytest.approx(Cin, rel=5e-3)
    assert e["checks"]["floating_solve"]["max_relative_floating_charge"] < 0.05


@slow
def test_open_boundary_does_not_depend_on_the_domain_radius(tmp_path):
    """The exterior map is exact, so the capacitance of an isolated sphere must not depend on where the
    domain ends -- including the monopole, which is what a capacitance to infinity is. A grounded boundary
    at the same radii does depend on it."""
    a = 0.5
    reg = [{"name": "ball", "solid": ball(a), "material": "superconductor"}]
    out = {}
    for Rb in (1.0, 2.0):
        mesh = {"order": 2, "surface_size": 0.06, "size": 0.2, "domain_radius": Rb}
        out[Rb] = (solve(tmp_path, f"o{Rb}", reg, mesh=mesh).data["electrostatics"]["capacitance_F"][0][0],
                   solve(tmp_path, f"g{Rb}", reg, mesh=mesh,
                         boundary="grounded").data["electrostatics"]["capacitance_F"][0][0])
    assert out[1.0][0] == pytest.approx(out[2.0][0], rel=3e-3)       # open: independent of R_b
    assert out[1.0][0] == pytest.approx(FPE * a, rel=5e-3)
    assert out[1.0][1] == pytest.approx(FPE / (1 / a - 1 / 1.0), rel=1e-2)   # grounded: depends on R_b
    assert out[2.0][1] == pytest.approx(FPE / (1 / a - 1 / 2.0), rel=1e-2)
    assert out[1.0][1] / out[2.0][1] > 1.1                           # and visibly so


@slow
def test_unit_cube_published_value(tmp_path):
    """A genuinely non-axisymmetric body against the published cube capacitance."""
    L = 0.4
    r = solve(tmp_path, "cube",
              [{"name": "cube", "solid": {"box": {"min": [-L / 2] * 3, "max": [L / 2] * 3}},
                "material": "superconductor"}],
              mesh={"order": 2, "size": 0.06, "surface_size": 0.015, "domain_radius": 1.2})
    e = r.data["electrostatics"]
    C = e["capacitance_F"][0][0]
    assert C / (FPE * L) == pytest.approx(CUBE_FACTOR, rel=1e-2)
    assert_consistent(e, gauss=0.06, mono=0.05)


@slow
def test_offset_spheres_non_axisymmetric(tmp_path):
    """Two equal spheres along (1,1,1)/sqrt(3), off the origin: the bispherical series depends only on
    the radius and the centre distance, so it is an exact reference for a configuration with no symmetry
    axis through the domain centre."""
    a, D = 0.1, 0.5
    u = 1.0 / math.sqrt(3.0)
    o = (0.13, -0.09, 0.07)
    c1 = tuple(o[k] - 0.5 * D * u for k in range(3))
    c2 = tuple(o[k] + 0.5 * D * u for k in range(3))
    eta = math.acosh(D / (2 * a))
    s = math.sinh(eta)
    # the terms fall off like exp(-k eta); stop before math.sinh overflows at k eta ~ 710
    def series(first):
        total, k = 0.0, first
        while k * eta < 700:
            total += 1.0 / math.sinh(k * eta)
            k += 2
        return total
    c11 = FPE * a * s * series(1)
    c12 = -FPE * a * s * series(2)
    r = solve(tmp_path, "offset", [{"name": "s1", "solid": ball(a, c1), "material": "superconductor"},
                                   {"name": "s2", "solid": ball(a, c2), "material": "superconductor"}],
              mesh={"order": 2, "size": 0.09, "surface_size": 0.016, "domain_radius": 1.2})
    e = r.data["electrostatics"]
    i, j = body(e, "s1"), body(e, "s2")
    C = e["capacitance_F"]
    assert C[i][i] == pytest.approx(c11, rel=2e-2)
    assert C[i][j] == pytest.approx(c12, rel=3e-2)
    assert C[i][i] == pytest.approx(C[j][j], rel=5e-3)  # the two spheres are equivalent
    assert_consistent(e, gauss=0.1, mono=0.05)


@slow
def test_3d_vs_axisymmetric(tmp_path):
    """A revolved axisymmetric design: the same capacitance matrix from both engines."""
    doc = {"schema": "axionfem.design/1", "name": "ring_can",
           "materials": {"support": {"kind": "linear", "eps_r": 3.0}},
           "regions": [
               {"name": "can", "shape": {"difference": [{"rect": {"r": [0.0, 0.5], "z": [-0.5, 0.5]}},
                                                        {"rect": {"r": [0.0, 0.45], "z": [-0.45, 0.45]}}]},
                "material": "superconductor"},
               {"name": "ring", "shape": {"rect": {"r": [0.25, 0.3], "z": [-0.03, 0.03]}},
                "material": "superconductor"},
               {"name": "support", "shape": {"rect": {"r": [0.3, 0.45], "z": [-0.03, 0.03]}},
                "material": "support"}],
           "electrostatics": {"boundary": "open"},
           "mesh": {"order": 3, "size": 0.05, "surface_size": 0.012, "domain_radius": 1.5}}
    d2 = af.Design(doc)
    e2 = d2.solve(tmp_path / "axi", verbose=0).data["electrostatics"]
    d3 = to_3d(d2, mesh={"order": 2, "size": 0.09, "surface_size": 0.03, "domain_radius": 1.5})
    e3 = d3.solve(tmp_path / "3d", verbose=0).data["electrostatics"]
    scale = abs(e2["capacitance_F"][body(e2, "can")][body(e2, "can")])
    for nm in ("can", "ring"):
        for nm2 in ("can", "ring"):
            a2 = e2["capacitance_F"][body(e2, nm)][body(e2, nm2)]
            a3 = e3["capacitance_F"][body(e3, nm)][body(e3, nm2)]
            assert abs(a3 - a2) < 3e-2 * scale, (nm, nm2, a2, a3)


@slow
def test_port_gap_capacitance_3d(tmp_path):
    """A split superconducting ring with a vacuum port in the gap: the two gap estimates are ordered and
    bracket the parallel-plate value of the gap, and the self-resonance follows from L_p."""
    R, w, t, g = 0.3, 0.05, 0.05, 0.02
    ang = math.degrees(g / R)
    prof = {"rect": {"r": [R - w / 2, R + w / 2], "z": [-t / 2, t / 2]}}
    doc = {"schema": "axionfem.design/1", "name": "port3", "symmetry": "3d",
           "regions": [{"name": "coil", "solid": {"revolve": {"profile": {"rect": {"r": [0.1, 0.12],
                                                                                  "z": [-0.1, 0.1]}}}},
                        "material": "winding", "mesh_size": 0.05},
                       {"name": "ring", "solid": {"revolve": {"profile": prof,
                                                              "angles": [ang / 2, 360 - ang / 2]}},
                        "material": "superconductor", "mesh_size": 0.02},
                       {"name": "gap", "solid": {"revolve": {"profile": prof, "angles": [-ang / 2, ang / 2]}},
                        "material": "vacuum", "mesh_size": 0.008}],
           "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
           "pickup": {"type": "port", "port": "gap"},
           "electrostatics": {},  # in 3D the readout-gap analysis runs with an electrostatics section
           "mesh": {"order": 2, "size": 0.15, "surface_size": 0.035, "domain_radius": 0.5}}
    r = af.Design(doc).solve(tmp_path / "port3", verbose=0)
    pc = r.data["pickup"]["port_capacitance"]
    assert pc["terminals_on_one_body"]
    plate = EPS0 * w * t / g
    assert pc["C_gap_port_region_F"] == pytest.approx(plate, rel=0.5)
    assert pc["C_differential_F"] >= pc["C_gap_port_region_F"]
    assert pc["checks"]["dirichlet_bound"]["holds"]
    C = pc["C_reduced_F"]
    assert pc["C_mutual_F"] == pytest.approx(-C[0][1])
    assert pc["C_differential_F"] == pytest.approx((C[0][0]+C[1][1]-2*C[0][1])/4)
    doc["electrostatics"]["boundary"] = "grounded"
    ground = af.Design(doc).solve(tmp_path / "ground", verbose=0).port_capacitance
    assert ground["gap_solver"]["boundary"] == "grounded"
    assert pc["gap_solver"]["boundary"] == "open"
    assert ground["C_differential_F"] > pc["C_differential_F"] * (1+1e-5)
    assert pc["self_resonance_Hz"] > 0
    assert pc["self_resonance_Hz"] == pytest.approx(
        1 / (2 * math.pi * math.sqrt(r.L_p * pc["C_gap_F"])), rel=1e-9)
    assert pc["checks"]["gap_symmetry_relative"] < 1e-6
    # the Python property of the axisymmetric solver works for 3D results too
    assert r.port_capacitance["C_gap_F"] == pc["C_gap_F"]


# ------------------------------------------------------------------ MPI
@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs >= 2 MPI slots; set AXF_MPI=1")
def test_capacitance_matrix_is_rank_independent(tmp_path):
    a, b, c = 0.3, 0.5, 0.6
    regions = [{"name": "inner", "solid": ball(a), "material": "superconductor"},
               {"name": "outer", "solid": hollow(b, c), "material": "superconductor"}]
    mesh = {"order": 2, "size": 0.15, "surface_size": 0.05, "domain_radius": 1.6}
    e1 = solve(tmp_path, "r1", regions, mesh=mesh).data["electrostatics"]
    e2 = solve(tmp_path, "r2", regions, mesh=mesh, ranks=2).data["electrostatics"]
    assert e2["solver"]["mpi_ranks"] == 2
    scale = max(abs(x) for row in e1["capacitance_F"] for x in row)
    for i in range(len(e1["capacitance_F"])):
        for j in range(len(e1["capacitance_F"])):
            assert abs(e2["capacitance_F"][i][j] - e1["capacitance_F"][i][j]) < 1e-8 * scale, (i, j)
