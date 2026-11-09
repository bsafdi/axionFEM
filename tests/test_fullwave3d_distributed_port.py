"""Distributed ports: one port region of several disjoint gap pieces, fed in parallel.

The conduction feed splits the current by conductance, w = G / sum(G), so a distributed port measures Z = w^T Z_p w and
Psi = w^T Psi_p, with Z_p, Psi_p of the pieces taken as separate ports; that is an ideal common bus
1 / (1^T Z_p^-1 1) only when a symmetry permutes the pieces (docs/physics/multiport_fullwave.md). A small closed
superconducting box with a full-circumference outer-wall slit (a CORE-type pickup, scaled down; `profiles` below) is
read out through pieces of the slit, the rest of it open:
  test_symmetric_pieces_equal_the_bus        two identical opposite 30-degree pieces as the pickup: split = bus.
  test_unequal_pieces_split_by_conductance   a 30- and a 90-degree piece as an EXTRA fullwave port (the strict path):
                                             Z = w^T Z_p w, which here differs from the bus by several per cent.
  test_distributed_port_orientation_errors   MQS only, coarse: no `toward`, and a `toward` across the crossing.
The full-wave cases use the direct (STRUMPACK) path when the build has it (closed PEC sphere, one frequency); the
distributed and the separate layouts give every gap region the same surface size, so they mesh alike. The full-wave
cases are slow (AXF_SLOW=1); the error test is not.
"""
import os

import numpy as np
import pytest

import axionfem as af

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1")
pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")

BOX = {"a": 0.10, "b": 0.22, "h": 0.16, "corner": 0.02}
WALL = 0.02
TOWARD = [0.0, 0.0, 1.0]


def _rect(r0, r1, z0, z1, fillet=None):
    s = {"rect": {"r": [r0, r1], "z": [z0, z1]}}
    if fillet is not None:
        s["rect"]["fillet"] = fillet
    return s


def profiles(box: dict, wall: float) -> tuple[dict, dict, float]:
    """The slotted wall and the port profile (r-z) for wall thickness and slit opening d = wall."""
    a, b, h, d = box["a"], box["b"], box["h"], wall
    rho = max(box["corner"], 2 * d)
    gs = 2 * d                                  # slot height between the lip centres; opening gs - d = d
    outer = _rect(a - d / 2, b + d / 2, -(h + d) / 2, (h + d) / 2, rho + d / 2)
    inner = _rect(a + d / 2, b - d / 2, -(h - d) / 2, (h - d) / 2, rho - d / 2)
    shape = {"union": [{"difference": [outer, inner, _rect(b - d, b + d, -gs / 2, gs / 2)]},
                       {"disk": {"center": [b, gs / 2], "radius": d / 2}},
                       {"disk": {"center": [b, -gs / 2], "radius": d / 2}}]}
    port = _rect(b - d / 4, b + d / 4, -gs / 2, gs / 2)
    return shape, port, rho


SHAPE, GAP, _ = profiles(BOX, WALL)


def region(name, *sectors):
    rev = [{"revolve": {"profile": GAP, "angles": list(s)}} for s in sectors]
    return {"name": name, "material": "vacuum", "solid": rev[0] if len(rev) == 1 else {"union": rev},
            "surface_size": WALL / 2}


def design(regions, pickup_region, ports=(), fullwave=True, order=2, toward=TOWARD):
    pickup = {"type": "port", "port": pickup_region}
    if toward is not None:
        pickup["toward"] = toward
    doc = {"schema": "axionfem.design/1", "name": "distributed_port_check", "symmetry": "3d",
           "regions": list(regions) + [{"name": "wall", "material": "nb", "solid": {"revolve": {"profile": SHAPE}}}],
           "superconductors": [{"body": "wall", "azimuthal": {"net_current": 0}}],
           "pickup": pickup,
           "mesh": {"order": order, "size": 0.08, "surface_size": WALL / 2}}
    if fullwave:
        doc["fullwave"] = {"frequencies": [60e6], "boundary": "pec",
                           "source": {"kind": "uniform", "B_T": [0.0, 0.0, 1.0]}}
        if ports:
            doc["fullwave"]["ports"] = [{"name": n, "region": r, "toward": TOWARD} for n, r in ports]
    return doc


def solve(doc, wd, monkeypatch):
    if af.build_info().get("strumpack"):
        monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    return af.Design(doc).solve(wd, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1).data


def matrix(row):
    Z = np.array(row["Z_matrix_re_ohm"]) + 1j * np.array(row["Z_matrix_im_ohm"])
    Psi = np.array(row["Psi_vector"]["re"]) + 1j * np.array(row["Psi_vector"]["im"])
    return Z, Psi


def bus(Z, Psi):
    Y = np.linalg.inv(Z)
    one = np.ones(len(Psi))
    Zb = 1 / (one @ Y @ one)
    return Zb, Zb * (one @ Y @ Psi)


@slow
def test_symmetric_pieces_equal_the_bus(tmp_path, monkeypatch):
    A0, B0, gaps = (-15, 15), (165, 195), [(15, 165), (195, 345)]
    rest = [region(f"gap{k}", g) for k, g in enumerate(gaps)]
    sep = solve(design([region("pa", A0), region("pb", B0)] + rest, "pa", [("pb", "pb")]), tmp_path / "s", monkeypatch)
    Z, Psi = matrix(sep["fullwave"]["sweep"][0])
    G = np.array([p["raw_conductance_m"] for p in sep["fullwave"]["ports"]])
    w = G / G.sum()
    Zs, Psi_s = w @ Z @ w, w @ Psi
    Zb, Psi_b = bus(Z, Psi)
    assert abs(Zs - Zb) / abs(Zb) < 1e-4, "identical opposite pieces: the split should be the bus"

    dist = solve(design([region("p", A0, B0)] + rest, "p"), tmp_path / "d", monkeypatch)
    row = dist["fullwave"]["sweep"][0]
    Zd, Psi_d = complex(row["Z_re_ohm"], row["Z_im_ohm"]), complex(row["Psi_re"], row["Psi_im"])
    assert abs(Zd - Zs) / abs(Zs) < 5e-3, (Zd, Zs)
    assert abs(abs(Psi_d) / abs(Psi_s) - 1) < 5e-3, (Psi_d, Psi_s)
    feed = dist["fullwave"]["ports"][0]
    assert feed["port_pieces"] == 2 and feed["terminal_patches"] == 4, feed
    assert all(abs(p["current_fraction"] - 0.5) < 1e-2 for p in feed["pieces"]), feed["pieces"]
    assert dist["pickup"]["port_feed"]["port_pieces"] == 2   # the MQS pickup results carry the feed too


@slow
def test_unequal_pieces_split_by_conductance(tmp_path, monkeypatch):
    A0, B0, C0 = (-15, 15), (135, 225), (75, 95)          # 30 and 90 degree pieces; a 20-degree pickup elsewhere
    gaps = [(15, 75), (95, 135), (225, 345)]
    rest = [region(f"gap{k}", g) for k, g in enumerate(gaps)]
    # the two pieces as separate ports, the pickup piece C left open
    sep = solve(design([region("pa", A0), region("pb", B0), region("gc", C0)] + rest, "pa", [("pb", "pb")]),
                tmp_path / "s", monkeypatch)
    Z, Psi = matrix(sep["fullwave"]["sweep"][0])
    G = np.array([p["raw_conductance_m"] for p in sep["fullwave"]["ports"]])
    w = G / G.sum()
    Zs, Psi_s = w @ Z @ w, w @ Psi
    Zb, _ = bus(Z, Psi)
    assert abs(Zb - Zs) / abs(Zs) > 2e-2, "these unequal pieces should separate the split from the bus"
    # the two pieces as ONE extra port (strict path); the pickup is C, open while the distributed port is driven
    dist = solve(design([region("pc", C0), region("pab", A0, B0)] + rest, "pc", [("modules", "pab")]),
                 tmp_path / "d", monkeypatch)
    Zd_m, Psi_d = matrix(dist["fullwave"]["sweep"][0])
    assert abs(Zd_m[1, 1] - Zs) / abs(Zs) < 5e-3, (Zd_m[1, 1], Zs, Zb)
    assert abs(abs(Psi_d[1]) / abs(Psi_s) - 1) < 5e-3, (Psi_d[1], Psi_s)
    fractions = sorted(p["current_fraction"] for p in dist["fullwave"]["ports"][1]["pieces"])
    assert np.allclose(fractions, sorted(w), atol=1e-2), (fractions, w)


def test_distributed_port_orientation_errors(tmp_path):
    A0, B0, gaps = (-15, 15), (165, 195), [(15, 165), (195, 345)]
    regs = [region("p", A0, B0)] + [region(f"gap{k}", g) for k, g in enumerate(gaps)]
    coarse = lambda doc: {**doc, "mesh": {"order": 1, "size": 0.1, "surface_size": WALL}}  # noqa: E731
    with pytest.raises(Exception, match="needs `toward`"):
        af.Design(coarse(design(regs, "p", fullwave=False, order=1, toward=None))).solve(tmp_path / "a", verbose=0)
    with pytest.raises(Exception, match="crossed roughly along toward"):
        af.Design(coarse(design(regs, "p", fullwave=False, order=1, toward=[1.0, 0.0, 0.0]))).solve(
            tmp_path / "b", verbose=0)
