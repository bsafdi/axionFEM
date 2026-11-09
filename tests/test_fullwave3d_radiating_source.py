"""Prescribed (Biot-Savart) full-wave source with the radiating exterior (fullwave.boundary = "radiation").

Two ports in one small model: port 0 is a slit copper loop (magnetic dipole, TE exterior field) and the extra
port 1 is the gap of a short electric dipole (two copper arms; TM exterior field, E_r != 0 on R1, so the exterior
overlap oint Phi_0 E_r dA is a large part of Psi). The source is a closed horizontal square current loop
around the dipole gap, running along the centre line of a closed copper frame: PEC elements are excluded
from the overlap (E = 0 there), so the 1/d Biot-Savart singularity is never sampled and the source
quadrature converges. (The frame is part of the device in every compared model; it is not a numerical trick
that leaves the port fields unchanged.) (With the filament in vacuum the TM-port Psi moved by 100% between source quadrature
orders 4, 8 and 12, because the electrostatic part of int E.B0 cancels exactly and Psi_1 is a ~4x
cancellation; see docs/validation/fullwave_radiating_source.md.) Checks:
  (a) R1-independence. TE port 0: Psi at R1 = .55 and .9 and in a closed PEC sphere (R = 1.2) agree to 0.5%
      (measured 0.12%). TM port 1: the exterior part is 4x Psi at R1 = .55 and changes by 2.5e-14 between
      the radii, while Psi changes by under 5% of that change (measured 2.7% on this mesh, 1.4% / 0.8% at
      mesh sizes .056 / .04: an O(h^1.75) discretization error, see docs/validation/fullwave_radiating_source.md).
      A missing exterior term gives 100%, a sign flip 200%. The PEC sphere is only a loose (10%) sanity bound
      for the TM port: the TM exterior part decays only as R^-3 (~40% of Psi beyond R = 1.2), and the PEC
      Psi moved 2% between R = 1.2 and 1.6 at mesh size .056, so its image error is not small;
  (b) the power leaving through R1, integrated from the FE field (Poynting), matches the port balance
      Re Z |I|^2/2 for both ports and both frequencies, and port 0 matches the magnetic-dipole formula at ka = .02.
Coarse ND2 meshes (35 min with 4 MPI ranks x 16 threads on one node). Set AXF_SLOW=1.
"""
import math
import os

import pytest

import axionfem as af

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

F_LO, F_HI = 2e7, 1.5e8


def _box(lo, hi):
    return {"box": {"min": list(lo), "max": list(hi)}}


def _design(boundary, radius, freqs):
    h, g, L = 0.02, 0.01, 0.25
    s = 0.12
    pts = [(s, -s), (s, s), (-s, s), (-s, -s)]
    segs = [[*pts[i], 0.0, *pts[(i + 1) % 4], 0.0, 1.0] for i in range(4)]
    loop = {"difference": [_box((0.30, -0.01, -0.05), (0.40, 0.01, 0.05)),
                           _box((0.32, -0.02, -0.03), (0.38, 0.02, 0.03)),
                           _box((0.345, -0.02, 0.025), (0.355, 0.02, 0.055))]}
    return {"schema": "axionfem.design/1", "name": "radsrc", "symmetry": "3d",
            "regions": [{"name": "loop", "material": "copper", "solid": loop},
                        {"name": "gap0", "material": "vacuum", "solid": _box((0.345, -0.01, 0.03), (0.355, 0.01, 0.05))},
                        {"name": "armp", "material": "copper", "solid": _box((-h, -h, g), (h, h, L))},
                        {"name": "armm", "material": "copper", "solid": _box((-h, -h, -L), (h, h, -g))},
                        {"name": "dip", "material": "vacuum", "solid": _box((-h, -h, -g), (h, h, g))},
                        {"name": "frame", "material": "copper",
                         "solid": {"difference": [_box((-s - 0.01, -s - 0.01, -0.01), (s + 0.01, s + 0.01, 0.01)),
                                                  _box((-s + 0.01, -s + 0.01, -0.02), (s - 0.01, s - 0.01, 0.02))]}}],
            "pickup": {"type": "port", "port": "gap0", "toward": [1, 0, 0]},
            "fullwave": {"frequencies": freqs, "boundary": boundary,
                         "ports": [{"name": "dipole", "region": "dip", "toward": [0, 0, 1]}],
                         "source": {"kind": "segments", "segments_m_A": segs}, "source_quadrature_order": 8},
            "mesh": {"order": 2, "size": 0.08, "surface_size": 0.02, "domain_radius": radius}}


def _solve(tmp_path, name, boundary, radius, freqs):
    r = af.Design(_design(boundary, radius, freqs)).solve(
        tmp_path / name, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    fw = r.fullwave
    # no background: none of the background's outputs (its code paths are behind the background's presence)
    assert "background" not in fw and "exterior_term_background" not in fw
    for row in fw["sweep"]:
        for p in row["port_solutions"]:
            assert p["solver"]["relative_residual"] < 1e-6
            assert "Psi_background" not in p
    return fw


def _psi(row, j):
    return complex(row["Psi_vector"]["re"][j], row["Psi_vector"]["im"][j])


def _ext(row, j):
    p = row["port_solutions"][j]
    return complex(p["Psi_exterior"]["re"], p["Psi_exterior"]["im"])


def test_radiating_prescribed_source(tmp_path):
    a = _solve(tmp_path, "rad055", "radiation", 0.55, [F_LO, F_HI])
    b = _solve(tmp_path, "rad09", "radiation", 0.9, [F_LO])
    pec = _solve(tmp_path, "pec12", "pec", 1.2, [F_LO])
    lo, hi, lo_b, lo_pec = a["sweep"][0], a["sweep"][1], b["sweep"][0], pec["sweep"][0]
    for tag, row in (("rad.55 lo", lo), ("rad.55 hi", hi), ("rad.9 lo", lo_b), ("pec1.2 lo", lo_pec)):
        print(tag, "Psi", row["Psi_vector"], "Z", row["Z_matrix_re_ohm"], row["Z_matrix_im_ohm"])
        for p in row["port_solutions"]:
            print("   port", p["port_index"], {k: p.get(k) for k in ("Psi_exterior", "R_rad_ohm", "R_rad_poynting_ohm",
                                                                   "R_static_dipole_ohm", "Z_re_ohm", "Z_im_ohm")})
    print("source:", a["source"], a["exterior_source_multipoles"], a["solver_settings"])
    # the default GMRES restart (600) and its memory bound are exported
    assert a["solver_settings"]["gmres_restart"] == int(os.environ.get("AXF_FW_KDIM", 600))
    assert a["solver_settings"]["gmres_krylov_basis_bytes_upper_bound_all_ranks"] > 0
    ext = a["source"]["exterior"]
    assert ext["max_current_distance_from_domain_center_m"] == pytest.approx(0.12 * math.sqrt(2))
    assert ext["max_relative_node_current_imbalance"] == 0
    for fw in (a, b, pec):  # the filament runs inside the PEC frame: no source-domain element meets it
        prox = fw["source"]["filament_proximity"]
        assert prox["source_domain_elements"] > 0 and prox["source_domain_elements_met_by_filament"] == 0
    assert a["exterior_source_multipoles"]["field"] == "prescribed source B0"
    assert abs(a["exterior_source_multipoles"]["unrepresented_fraction"]) < 1e-3
    for j in (0, 1):
        pa, pb, pp = _psi(lo, j), _psi(lo_b, j), _psi(lo_pec, j)
        ea, eb = _ext(lo, j), _ext(lo_b, j)
        print(f"port {j}: rad.55 {pa:.6e} (ext {ea:.3e}), rad.9 {pb:.6e} (ext {eb:.3e}), pec1.2 {pp:.6e}")
        # (a) R1-independence and the far-PEC limit
        if j == 0:  # TE: exterior part 8.8% (R1 = .55) / 2.1% (.9) of Psi, falling as R^-3; PEC image error small
            for x, y in ((pa, pb), (pa, pp), (pb, pp)):
                assert abs(x - y) < 0.005 * abs(pp)
        else:  # TM: the exterior part is large and R1-dependent, so the check is sign- and scale-sensitive
            assert abs(ea) > 2 * abs(pa)
            assert abs(pa - pb) < 0.05 * abs(ea - eb)
            assert abs(pa - pp) < 0.1 * abs(pp) and abs(pb - pp) < 0.1 * abs(pp)
    # (b) power balance: Poynting flux through R1 vs Re Z |I|^2/2 (lossless), every port and frequency
    for row in (lo, hi):
        for p in row["port_solutions"]:
            assert p["R_rad_ohm"] > 0
            assert p["R_rad_poynting_ohm"] == pytest.approx(p["R_rad_ohm"], rel=0.01)
    assert lo["port_solutions"][0]["R_rad_ohm"] == pytest.approx(lo["port_solutions"][0]["R_static_dipole_ohm"], rel=0.015)
    assert all(abs(z) < 1e-9 for z in (lo_pec["Z_re_ohm"],))
