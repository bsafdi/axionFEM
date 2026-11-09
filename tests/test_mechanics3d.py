"""3D elasticity and steady conduction (design keys `mechanics`, `thermal` on a "3d" design) against closed forms.

Fast by default (coarse meshes, one process):
- free cooldown of an unsupported L-shaped body: u = -alpha (x - x_c), zero stress, six rigid motions pinned;
- Lame cylinder as a 40-degree sector between two inclined symmetry planes (rotated-constraint path);
- conduction in a slab: linear and quadratic T exact for P2, k(T) against the Kirchhoff solution, heat balance;
- two attracting coils: the Lorentz load of the elasticity assembly equals results["forces"] and minus the
  support reactions;
- input errors (Cartesian fixes, 3D-only keys, planes, temperatures).

AXF_SLOW adds the plane-strain Lame convergence; AXF_MPI the rank independence. The axisymmetric-vs-3D comparison
is in the validation script.  The full set with records is benchmarks/mech3d_validation.py (docs/validation/mech3d.md).
"""
import math
import os

import numpy as np
import pytest

import axionfem as af

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow 3D solve; set AXF_SLOW=1")
mpi = pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs MPI ranks; set AXF_MPI=1")
needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build; the wheels have none)")
MESH = {"order": 2, "size": 0.1, "surface_size": 0.05, "far_size": 0.4}


def cyl(r, z0, z1):
    return {"cylinder": {"base": [0, 0, z0], "axis": [0, 0, z1 - z0], "radius": r}}


def tube(r0, r1, z0, z1):
    return {"difference": [cyl(r1, z0, z1), cyl(r0, z0, z1)]}


def doc3(name, regions, materials, extra, mesh=None):
    d = {"schema": "axionfem.design/1", "name": name, "symmetry": "3d", "regions": regions, "materials": materials,
         "mesh": mesh or dict(MESH, domain_radius=1.0)}
    d.update(extra)
    return d


def solve(tmp_path, d, ranks=1):
    return af.Design(d).solve(tmp_path / d["name"], verbose=0, mpi_ranks=ranks)


STEEL = {"kind": "linear", "mechanical": {"E_Pa": 2e11, "nu": 0.3, "alpha": 0.0}}


# ------------------------------------------------------------------ elasticity
@needs_3d
def test_free_cooldown_is_a_stress_free_contraction(tmp_path):
    alpha = 2.5e-3
    boxes = [{"box": {"min": [0, 0, 0], "max": [0.4, 0.1, 0.1]}}, {"box": {"min": [0, 0, 0], "max": [0.1, 0.3, 0.1]}}]
    pts = [[0.35, 0.05, 0.05], [0.05, 0.25, 0.05], [0.2, 0.08, 0.02]]
    d = doc3("cool3d", [{"name": "body", "material": "m", "solid": {"union": boxes}, "mesh_size": 0.05}],
             {"m": {"kind": "linear", "mechanical": {"E_Pa": 1.2e11, "nu": 0.3, "alpha": alpha}}},
             {"mechanics": {"cooldown": True, "probes": pts}})
    m = solve(tmp_path, d).data["mechanics"]
    xc = np.array([0.4 * 0.2 - 0.1 * 0.05 + 0.3 * 0.05, 0.4 * 0.05 - 0.1 * 0.05 + 0.3 * 0.15, 0.6 * 0.05]) / 0.6
    u = np.array([p["u_m"] for p in m["probes"]])
    assert np.max(np.abs(u + alpha * (np.array(pts) - xc))) < 1e-9 * alpha
    assert m["regions"]["body"]["von_mises_max_Pa"] < 1e-8 * 1.2e11 * alpha
    assert np.max(np.abs([p["sigma_Pa"] for p in m["probes"]])) < 1e-8 * 1.2e11 * alpha  # no hydrostatic part either
    assert m["solver"]["rigid_mode_pins"] == 6
    assert m["rigid_modes"][0]["free_rigid_modes"] == 6
    assert m["checks"]["equilibrium_F_round_off"] < 1e-12


@needs_3d
def test_lame_sector_between_inclined_symmetry_planes(tmp_path):
    a, b, H, p = 0.1, 0.3, 0.4, 1e6
    sec = lambda r0, r1: {"revolve": {"profile": {"rect": {"r": [r0, r1], "z": [-H / 2, H / 2]}}, "angles": [-20, 20]}}
    planes = [{"point": [0, 0, -H / 2], "normal": [0, 0, 1]}, {"point": [0, 0, H / 2], "normal": [0, 0, 1]}]
    planes += [{"point": [0, 0, 0], "normal": [-math.sin(math.radians(s)), math.cos(math.radians(s)), 0]}
               for s in (-20, 20)]
    phi = math.radians(7)
    r = np.linspace(0.12, 0.28, 7)
    d = doc3("lame_sector", [{"name": "bore", "material": "vacuum", "solid": sec(0, a)},
                             {"name": "cyl", "material": "steel", "solid": sec(a, b), "mesh_size": 0.04}],
             {"steel": STEEL},
             {"mechanics": {"pressures": [{"on": "cyl", "from": "bore", "p_Pa": p}], "symmetry_planes": planes,
                            "probes": [[x * math.cos(phi), x * math.sin(phi), 0.0] for x in r]}})
    m = solve(tmp_path, d).data["mechanics"]
    assert m["solver"]["rotated_nodes"] > 0 and m["solver"]["rigid_mode_pins"] == 0
    k = p * a * a / (b * b - a * a)
    ur_e = 1.3 / 2e11 * k * (0.4 * r + b * b / r)                # plane strain, nu = 0.3
    stt_e = k * (1 + b * b / r ** 2)
    u = np.array([q["u_m"] for q in m["probes"]])
    ur = u[:, 0] * math.cos(phi) + u[:, 1] * math.sin(phi)
    s = np.array([q["sigma_Pa"] for q in m["probes"]])
    c, sn = math.cos(phi), math.sin(phi)
    stt = sn * sn * s[:, 0] + c * c * s[:, 1] - 2 * c * sn * s[:, 5]
    assert np.max(np.abs(ur / ur_e - 1)) < 5e-3
    assert np.max(np.abs(stt / stt_e - 1)) < 8e-2      # coarse P2 mesh (validation: 2.4e-3 at h = 0.01)
    assert np.max(np.abs(u[:, 2])) < 2e-3 * np.max(ur)           # plane strain: u_z = 0 up to discretization
    # the planes carry the pressure: the two inclined planes push back on the sector (compression)
    pl = {q["name"]: q for q in m["symmetry_planes"]}
    assert all(q["boundary_faces"] > 0 for q in pl.values())
    assert m["checks"]["equilibrium_F_round_off"] < 1e-10


@needs_3d
def test_hydrostatic_patch_in_inclined_sector_is_exact(tmp_path):
    # A sector between two inclined planes through the axis (10 and 55 degrees: neither is a coordinate plane) and the
    # plane z = 0, pressure p on every other face. sigma = -p I and u = -p/(3K) x satisfy every condition on any
    # (curved, unstructured) mesh and lie in the P2 space, so the discrete solution is exact up to round-off: a direct
    # test of the rotated-constraint elimination (Q^T K Q) and of the pressure load, without discretization error.
    p, E, nu = 1e6, 2e11, 0.3
    rng = np.random.default_rng(1)
    r, ph, z = rng.uniform(0.11, 0.29, 20), np.radians(rng.uniform(11, 54, 20)), rng.uniform(0.01, 0.19, 20)
    pts = np.c_[r * np.cos(ph), r * np.sin(ph), z]
    planes = [{"point": [0, 0, 0], "normal": [0, 0, 1]}]
    planes += [{"point": [0, 0, 0], "normal": [-math.sin(math.radians(s)), math.cos(math.radians(s)), 0]}
               for s in (10, 55)]
    sec = {"revolve": {"profile": {"rect": {"r": [0.1, 0.3], "z": [0.0, 0.2]}}, "angles": [10, 55]}}
    d = doc3("hydro_sector", [{"name": "cyl", "material": "steel", "solid": sec, "mesh_size": 0.04}], {"steel": STEEL},
             {"mechanics": {"pressures": [{"on": "cyl", "from": "vacuum", "p_Pa": p}], "symmetry_planes": planes,
                            "probes": pts.tolist()}})
    m = solve(tmp_path, d).data["mechanics"]
    assert m["solver"]["rotated_nodes"] > 0 and m["solver"]["rigid_mode_pins"] == 0
    u = np.array([q["u_m"] for q in m["probes"]])
    s = np.array([q["sigma_Pa"] for q in m["probes"]])
    ue = -p * 3 * (1 - 2 * nu) / (3 * E) * pts
    assert np.max(np.abs(u - ue)) < 1e-9 * np.max(np.abs(ue))
    assert np.max(np.abs(s - np.array([-p, -p, -p, 0, 0, 0]))) < 1e-9 * p


@needs_3d
def test_two_coils_lorentz_load_matches_forces_and_reactions(tmp_path):
    coils = [("upper", 0.15, 0.25), ("lower", -0.25, -0.15)]
    d = doc3("twocoils", [{"name": n, "material": "w", "solid": tube(0.3, 0.36, z0, z1), "mesh_size": 0.04}
                          for n, z0, z1 in coils],
             {"w": {"kind": "vacuum", "mechanical": {"E_Pa": 8e10, "nu": 0.3, "alpha": 0.0}}},
             {"windings": [{"region": n, "type": "azimuthal", "ampere_turns": 2e5} for n, _, _ in coils],
              "mechanics": {"supports": [{"region": n, "fix": "all"} for n, _, _ in coils]}},
             mesh=dict(MESH, domain_radius=1.5))
    res = solve(tmp_path, d)
    m = res.data["mechanics"]
    for n, _, _ in coils:
        Fm = np.array(m["lorentz_forces"][n]["F_N"])
        Ff = np.array(res.data["forces"]["windings"][n]["F_N"])
        R = np.array(next(s["reaction_F_N"] for s in m["supports"] if s["region"] == n))
        assert abs(Fm[2]) > 100.0
        assert abs(Fm[2] - Ff[2]) < 1e-9 * abs(Ff[2])
        assert np.linalg.norm(R + Fm) < 1e-8 * abs(Fm[2])
    assert m["lorentz_forces"]["upper"]["F_N"][2] < 0 < m["lorentz_forces"]["lower"]["F_N"][2]  # attraction


@needs_3d
def test_external_uniform_field_hoop_stress(tmp_path):
    """A thin ring with azimuthal current in a uniform external B_z: f_r = J B uniform, mean hoop = J B Rm."""
    R0, R1, h, NI, Bz = 0.5, 0.52, 0.05, 1e4, 2.0
    d = doc3("ring_ext", [{"name": "ring", "material": "w", "solid": tube(R0, R1, -h / 2, h / 2), "mesh_size": 0.02}],
             {"w": {"kind": "vacuum", "mechanical": {"E_Pa": 1e11, "nu": 0.3, "alpha": 0.0}}},
             {"windings": [{"region": "ring", "type": "azimuthal", "ampere_turns": NI}],
              "mechanics": {"external_field": {"kind": "uniform", "B_T": [0, 0, Bz]},
                            "probes": [[0.51, 0.0, 0.0], [0.0, 0.51, 0.01]]}},
             mesh=dict(MESH, domain_radius=1.5, surface_size=0.02))
    m = solve(tmp_path, d).data["mechanics"]
    J = NI / ((R1 - R0) * h)
    fx = m["lorentz_forces"]["ring"]["F_external_field_N"]
    assert abs(fx[0]) < 1e-6 * J * Bz and abs(fx[2]) < 1e-6 * J * Bz   # net force of a ring in a uniform field
    hoop = m["regions"]["ring"]["sigma_hoop_Pa"]
    exact = J * Bz * 0.51
    assert hoop["min"] > 0.8 * exact and hoop["max"] < 1.2 * exact


# ------------------------------------------------------------------ conduction
def slab(tmp_path, k, q=None, size=0.05):
    L, w = 1.0, 0.2
    mats = {"cond": {"kind": "linear", "thermal": {"k_W_mK": k}}, "sink": {"kind": "linear", "thermal": {"k_W_mK": 1e3}}}
    x = np.linspace(0.05, 0.95, 10)
    d = doc3("slab", [{"name": "slab", "material": "cond", "mesh_size": size,
                       "solid": {"box": {"min": [0, 0, 0], "max": [L, w, w]}}},
                      {"name": "cold", "material": "sink", "solid": {"box": {"min": [-0.02, 0, 0], "max": [0, w, w]}}},
                      {"name": "warm", "material": "sink", "solid": {"box": {"min": [L, 0, 0], "max": [L + 0.02, w, w]}}}],
             mats, {"thermal": {"fixed": [{"region": "cold", "T_K": 4.0}, {"region": "warm", "T_K": 30.0}],
                                "loads": [{"region": "slab", "W_m3": q}] if q else [],
                                "probes": [[float(v), 0.07, 0.13] for v in x]}},
             mesh=dict(MESH, domain_radius=2.0))
    t = solve(tmp_path, d).data["thermal"]
    return x, np.array([p["T_K"] for p in t["probes"]]), t


@needs_3d
def test_slab_conduction_is_exact_for_quadratic_temperatures(tmp_path):
    x, T, t = slab(tmp_path, 10.0)
    assert np.max(np.abs(T - (4 + 26 * x))) < 1e-8
    assert t["checks"]["heat_added_W"] == pytest.approx(10.0 * 26 * 0.04, rel=1e-8)
    assert t["checks"]["balance_relative"] < 1e-8
    x, T, t = slab(tmp_path, 10.0, q=400.0)
    assert np.max(np.abs(T - (4 + 26 * x + 400 * x * (1 - x) / 20))) < 1e-8
    assert t["checks"]["generated_W"] == pytest.approx(400 * 0.04, rel=1e-10)


@needs_3d
def test_slab_temperature_dependent_k_against_kirchhoff(tmp_path):
    c, k0 = 0.08, 10.0
    x, T, t = slab(tmp_path, [[4.0, k0], [30.0, k0 * (1 + c * 26)]])
    th = k0 * (26 + c * 26 ** 2 / 2) * x
    Te = 4 + (np.sqrt(1 + 2 * c * th / k0) - 1) / c
    assert np.max(np.abs(T - Te)) < 5e-3                # h = 0.05 (validation: 1.2e-4 at h = 0.025)
    assert t["solver"]["converged"] and t["solver"]["picard_iterations"] > 3


# ------------------------------------------------------------------ errors
def test_3d_input_errors(tmp_path):
    base = doc3("err", [{"name": "b", "material": "steel", "solid": {"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}}],
                {"steel": STEEL}, {"mechanics": {"supports": [{"region": "b", "fix": "r"}]}})
    with pytest.raises(Exception, match="x.*y.*z"):
        af.Design(base).validate()
    axi = {"schema": "axionfem.design/1", "name": "axi",
           "regions": [{"name": "b", "material": "nb", "shape": {"rect": {"r": [0.1, 0.2], "z": [0, 0.1]}}}],
           "mechanics": {"symmetry_planes": []}}
    with pytest.raises(Exception, match="3D designs"):
        af.Design(axi).validate()
    axi2 = {"schema": "axionfem.design/1", "name": "axi2",
            "regions": [{"name": "b", "material": "nb", "shape": {"rect": {"r": [0.1, 0.2], "z": [0, 0.1]}}}],
            "thermal": {"fixed": [{"region": "b", "T_K": 4}], "probes": []}}
    with pytest.raises(Exception, match="3D designs"):
        af.Design(axi2).validate()


@needs_3d
def test_3d_solve_time_errors(tmp_path):
    reg = [{"name": "b", "material": "steel", "solid": {"box": {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}}},
           {"name": "c", "material": "steel", "solid": {"box": {"min": [0.2, 0, 0], "max": [0.4, 0.2, 0.2]}}}]
    mats = {"steel": {**STEEL, "thermal": {"k_W_mK": 10.0}}}
    d = doc3("dup", reg, mats, {"mechanics": {"supports": [{"region": "b", "fix": "x"}, {"region": "b", "fix": "y"}]}})
    with pytest.raises(Exception, match="already has a support"):
        solve(tmp_path, d)
    d = doc3("zeron", reg, mats, {"mechanics": {"symmetry_planes": [{"point": [0, 0, 0], "normal": [0, 0, 0]}]}})
    with pytest.raises(Exception, match="must not vanish"):
        solve(tmp_path, d)
    d = doc3("tconf", reg, mats, {"thermal": {"fixed": [{"region": "b", "T_K": 4.0}, {"region": "c", "T_K": 5.0}]}})
    with pytest.raises(Exception, match="conflicting prescribed temperatures"):
        solve(tmp_path, d)
    reg2 = reg[:1] + [{"name": "c", "material": "steel", "solid": {"box": {"min": [0.3, 0, 0], "max": [0.4, 0.2, 0.2]}}}]
    d = doc3("tiso", reg2, mats, {"thermal": {"fixed": [{"region": "b", "T_K": 4.0}]}})
    with pytest.raises(Exception, match="no prescribed temperature"):
        solve(tmp_path, d)


@needs_3d
def test_partial_support_pins_the_remaining_motions(tmp_path):
    """fix "z" on the only region leaves x, y and the rotation about z free: they are pinned on constrained nodes."""
    d = doc3("fixz", [{"name": "b", "material": "m", "solid": {"box": {"min": [0, 0, 0], "max": [0.3, 0.2, 0.1]}},
                       "mesh_size": 0.05}],
             {"m": {"kind": "linear", "mechanical": {"E_Pa": 1e11, "nu": 0.3, "alpha": 2e-3}}},
             {"mechanics": {"cooldown": True, "supports": [{"region": "b", "fix": "z"}],
                            "probes": [[0.25, 0.15, 0.05]]}})
    m = solve(tmp_path, d).data["mechanics"]
    assert m["rigid_modes"][0]["free_rigid_modes"] == 3 and m["solver"]["rigid_mode_pins"] == 3
    # u_z = 0 everywhere; in-plane contraction about the centroid (0.15, 0.1)
    u = np.array(m["probes"][0]["u_m"])
    assert u == pytest.approx([-2.6e-3 * 0.10, -2.6e-3 * 0.05, 0.0], abs=1e-9)  # in-plane strain -alpha (1 + nu)
    assert m["probes"][0]["sigma_Pa"][2] == pytest.approx(+1e11 * 2e-3, rel=1e-8)  # held in z: tension E alpha


@needs_3d
def test_edge_contact_mechanism_is_rejected(tmp_path):
    reg = [{"name": "a", "material": "steel", "solid": {"box": {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}}},
           {"name": "b", "material": "steel", "solid": {"box": {"min": [0.2, 0.2, 0], "max": [0.4, 0.4, 0.2]}}}]
    d = doc3("hinge", reg, {"steel": STEEL}, {"mechanics": {"supports": [{"region": "a", "fix": "all"}]}})
    with pytest.raises(Exception, match="only along an edge or at a point"):
        solve(tmp_path, d)


@needs_3d
def test_heat_split_is_per_sink(tmp_path):
    """A loaded region that is itself a sink: all of its heat is removed there (P2 nodal reactions change sign)."""
    reg = [{"name": "s", "material": "c", "solid": {"box": {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}}},
           {"name": "w", "material": "c", "solid": {"box": {"min": [0.2, 0, 0], "max": [0.4, 0.2, 0.2]}}}]
    d = doc3("sinkload", reg, {"c": {"kind": "linear", "thermal": {"k_W_mK": 5.0}}},
             {"thermal": {"fixed": [{"region": "s", "T_K": 4.0}], "loads": [{"region": "s", "W": 2.0}]}})
    t = solve(tmp_path, d).data["thermal"]
    assert t["checks"]["heat_removed_W"] == pytest.approx(2.0, rel=1e-9)
    assert t["checks"]["heat_added_W"] == pytest.approx(0.0, abs=1e-9)
    assert t["fixed"][0]["heat_absorbed_W"] == pytest.approx(2.0, rel=1e-9)


@needs_3d
def test_duplicate_temperature_error(tmp_path):
    d = doc3("dupT", [{"name": "b", "material": "c", "solid": {"box": {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}}}],
             {"c": {"kind": "linear", "thermal": {"k_W_mK": 5.0}}},
             {"thermal": {"fixed": [{"region": "b", "T_K": 4.0}, {"region": "b", "T_K": 30.0}]}})
    with pytest.raises(Exception, match="already has a prescribed temperature"):
        solve(tmp_path, d)


# ------------------------------------------------------------------ slow / MPI
@slow
def test_lame_plane_strain_convergence(tmp_path):
    a, b, H, p = 0.1, 0.3, 0.4, 1e6
    errs = []
    for size in (0.04, 0.02):
        r = np.linspace(0.12, 0.28, 7)
        d = doc3(f"lame_{size}", [{"name": "bore", "material": "vacuum", "solid": cyl(a, -H / 2, H / 2)},
                                  {"name": "cyl", "material": "steel", "solid": tube(a, b, -H / 2, H / 2),
                                   "mesh_size": size}], {"steel": STEEL},
                 {"mechanics": {"pressures": [{"on": "cyl", "from": "bore", "p_Pa": p}],
                                "symmetry_planes": [{"point": [0, 0, -H / 2], "normal": [0, 0, 1]},
                                                    {"point": [0, 0, H / 2], "normal": [0, 0, 1]}],
                                "probes": [[x, 0.0, 0.0] for x in r]}}, mesh=dict(MESH, surface_size=size))
        m = solve(tmp_path, d).data["mechanics"]
        k = p * a * a / (b * b - a * a)
        ur_e = 1.3 / 2e11 * k * (0.4 * r + b * b / r)
        errs.append(np.max(np.abs(np.array([q["u_m"][0] for q in m["probes"]]) / ur_e - 1)))
    assert errs[1] < 1e-3 and errs[1] < errs[0] / 3


@mpi
def test_rank_independence(tmp_path):
    coils = [("upper", 0.15, 0.25), ("lower", -0.25, -0.15)]
    d = doc3("ranks", [{"name": n, "material": "w", "solid": tube(0.3, 0.36, z0, z1), "mesh_size": 0.04}
                       for n, z0, z1 in coils],
             {"w": {"kind": "vacuum", "mechanical": {"E_Pa": 8e10, "nu": 0.3, "alpha": 3e-3},
                    "thermal": {"k_W_mK": [[4.0, 1.0], [40.0, 3.0]]}}},
             {"windings": [{"region": n, "type": "azimuthal", "ampere_turns": 2e5} for n, _, _ in coils],
              "mechanics": {"cooldown": True, "supports": [{"region": n, "fix": "z"} for n, _, _ in coils],
                            "probes": [[0.33, 0.01, 0.2], [0.0, 0.35, -0.2]]},
              "thermal": {"fixed": [{"region": "lower", "T_K": 4.0}, {"region": "upper", "T_K": 4.0}],
                          "loads": [{"region": "upper", "W": 1.0}],
                          "probes": [[0.33, 0.01, 0.2]]}},
             mesh=dict(MESH, domain_radius=1.5))
    r1 = solve(tmp_path / "n1", d, ranks=1).data
    r4 = solve(tmp_path / "n4", d, ranks=4).data
    u1 = np.array([p["u_m"] for p in r1["mechanics"]["probes"]])
    u4 = np.array([p["u_m"] for p in r4["mechanics"]["probes"]])
    assert np.max(np.abs(u1 - u4)) < 1e-8 * np.max(np.abs(u1))
    for k in ("upper", "lower"):
        assert r4["mechanics"]["regions"][k]["von_mises_max_Pa"] == pytest.approx(
            r1["mechanics"]["regions"][k]["von_mises_max_Pa"], rel=1e-7)
    assert r4["thermal"]["probes"][0]["T_K"] == pytest.approx(r1["thermal"]["probes"][0]["T_K"], abs=1e-9)
    assert r4["thermal"]["checks"]["heat_removed_W"] == pytest.approx(r1["thermal"]["checks"]["heat_removed_W"],
                                                                      rel=1e-8, abs=1e-12)
