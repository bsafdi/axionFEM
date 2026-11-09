"""3D and porous current distribution (axionfem.chem): 3D mesh families against the case meshes' hashes, the porous
electrode against its closed form, and a small 3D solve on 1 and 2 MPI ranks (the full cases run with
benchmarks/chem/run.py)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/chem/cases"
REG = ROOT / "benchmarks/chem/registry/cases"

chem = pytest.importorskip("axionfem.chem")
try:
    from axionfem._core import chem as _c
except Exception:  # pragma: no cover
    pytest.skip("axionfem._core has no chem submodule (rebuild)", allow_module_level=True)
if not hasattr(_c, "PorousElectrode1D"):
    pytest.skip("axionfem._core has no porous electrode (rebuild)", allow_module_level=True)

from axionfem.chem import PorousElectrode, Reaction, meshes3d as M3  # noqa: E402

import _mpi_launch as mpi  # noqa: E402  (tests/_mpi_launch.py: srun, mpirun or mpiexec)


def _checks():
    sys.path.insert(0, str(CASES))
    try:
        import c3_checks
        return c3_checks
    finally:
        sys.path.pop(0)


def _case(cid):
    return json.loads((REG / f"{cid}.json").read_text())


def _child_pythonpath() -> str:
    """This process's sys.path for a child interpreter, led by the directory of the axionfem imported here: a test
    may have appended the source tree (python/, without a compiled core) after site-packages, and in PYTHONPATH it
    would come first and shadow an installed package."""
    import axionfem
    first = str(Path(axionfem.__file__).resolve().parents[1])
    return os.pathsep.join([first] + [p for p in sys.path if p and p != first])


def test_runner_refuses_ranks_with_another_core():
    """The 3D runner refuses MPI ranks that report another compiled core than the launching process."""
    sys.path.insert(0, str(CASES))
    try:
        import c3b_current_distribution as R
    finally:
        del sys.path[:1]
    with pytest.raises(RuntimeError, match="ranks must run the core"):
        R._check_ranks([dict(rank=1, build=dict(R.core_identity(), core_sha256="other"))])


@pytest.mark.parametrize("N", [4, 8])
def test_kuhn_box_matches_the_case_hashes(N):
    C = _checks()
    lv = next(x for x in _case("chem_mms_c3_3d")["run"]["parameters"]["meshes"]["p1_3d"]["levels"]
              if x["n_elements"] == 6 * N ** 3)
    m = M3.box(N)
    assert C.vertex_hash(m.vertices, 0.01) == lv["vertex_hash"]
    assert C.connectivity_hash(m.vertices, m.T, 0.01) == lv["connectivity_hash"]
    assert np.bincount(m.battr).tolist() == [0] + [2 * N * N] * 6


@pytest.mark.parametrize("fam", ["cyl3d", "sph3d"])
def test_sectors_match_the_case_geometry_hash(fam):
    C = _checks()
    lv = _case("chem_analytic_c3_concentric_3d")["run"]["parameters"]["meshes"][f"{fam}_p2"]["levels"][0]
    m = M3.sector(4, fam, 2)
    assert C.vertex_hash(m.vertices, 0.01) == lv["vertex_hash"]
    assert C.vertex_hash(np.vstack([m.vertices, m.edge_xyz]), 0.01) == lv["geometry_node_hash"]


def _porous_exact(sigma, kappa, kv, l, I):
    k = math.sqrt(kv * (1 / sigma + 1 / kappa))
    nu = k * l
    B = -I / (sigma * k)
    A = I * (1 / kappa + math.cosh(nu) / sigma) / (k * math.sinh(nu))
    return A + kv / (kappa * k * k) * (A * (math.cosh(nu) - 1) + B * (math.sinh(nu) - nu))


@pytest.mark.parametrize("order,N,tol", [(1, 256, 1e-5), (2, 64, 1e-8)])
def test_porous_electrode_closed_form(order, N, tol):
    l, sg, kp, a_v, i0 = 1e-3, 10.0, 2.5, 1e5, 2.5
    rx = Reaction("linear", i0=i0, alpha_a=0.5, alpha_c=0.5)
    kv = a_v * i0 * 1.0 * chem.FARADAY / (chem.GAS_CONSTANT * 298.15)
    pe = PorousElectrode(l, N, order=order)
    pe.setup(sg, kp, a_v, rx, I=10.0)
    rep = pe.solve()
    assert rep["converged"] and rep["stop"] == "stagnation"
    z = np.asarray(pe.state())
    V = z[0] - z[-1]
    assert abs(V / _porous_exact(sg, kp, kv, l, 10.0) - 1) < tol
    # exchange identity: the electrolyte rows sum to -int j + I = 0, the collector row carries I
    res = np.asarray(pe.residual_full())
    assert abs(res[pe.collector_row()] - 10.0) < 1e-9


def test_porous_potentiostatic_matches_galvanostatic():
    rx = Reaction("linear", i0=2.5)
    pg = PorousElectrode(1e-3, 32, order=2)
    pg.setup(10.0, 2.5, 1e5, rx, I=10.0)
    pg.solve()
    V = float(np.asarray(pg.state())[0] - np.asarray(pg.state())[-1])
    pp = PorousElectrode(1e-3, 32, order=2)
    pp.setup(10.0, 2.5, 1e5, rx, V=V)
    pp.solve()
    I = -float(np.asarray(pp.residual_full())[pp.separator_row()])
    assert abs(I / 10.0 - 1) < 1e-12


@pytest.mark.parametrize("kind,extra", [("bv", {}), ("tafel_a", {"A": 0.06}), ("tafel_c", {"A": -0.06})])
def test_porous_refuses_nonlinear_exchange_kinetics(kind, extra):
    """Only linearized Butler-Volmer exchange is supported: bv and Tafel are refused with a clear error."""
    with pytest.raises(Exception, match="linearized Butler-Volmer exchange only"):
        PorousElectrode(1e-3, 4).setup(10.0, 2.5, 1e5, Reaction(kind, i0=2.5, **extra), I=10.0)


def test_porous_refuses_bad_input():
    with pytest.raises(Exception, match="conductivities"):
        PorousElectrode(1e-3, 4).setup(-1.0, 1.0, 1.0, Reaction("linear", i0=1.0), I=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        PorousElectrode(1e-3, 4).setup(1.0, 1.0, 1.0, Reaction("linear", i0=1.0))


_WORKER = r"""
import json, sys
import numpy as np
from axionfem.chem import CurrentDistribution3D, Reaction, meshes3d
cd = CurrentDistribution3D(meshes3d.box(4), order=int(sys.argv[2]))
cd.set_conductivity(1.0)
cd.add_electrode("a", [1], reactions=[Reaction("bv", i0=10, E_eq=0.05)], phi_s=0.6)
cd.add_electrode("c", [2], reactions=[Reaction("bv", i0=10, E_eq=-0.02)], phi_s=0.0)
rep = cd.solve()
q = cd.boundary_quadrature([1], cd.boundary_degree)
eta = 0.6 - q["phi"] - 0.05
f = 96485.33212 / (8.314462618 * 298.15)
I = float(np.sum(10 * (np.exp(0.5 * f * eta) - np.exp(-0.5 * f * eta)) * q["w"]))
json.dump(dict(rank=cd.rank(), ranks=cd.ranks(), I=I, it=rep["iterations"], conv=rep["converged"]),
          open(sys.argv[1] + f".{cd.rank()}", "w"))
"""


@pytest.mark.skipif(not hasattr(_c, "CurrentDistribution3D") or not mpi.available() or not os.environ.get("AXF_MPI"),
                    reason="needs the MPI build, an MPI launcher (srun, mpirun or mpiexec) and AXF_MPI=1")
@pytest.mark.parametrize("order", [1, 2])
def test_3d_rank_invariance(tmp_path, order):
    """The 1D-symmetric box problem: the anode current on 1 and on 2 ranks agrees to round-off."""
    script = tmp_path / "w.py"
    script.write_text(_WORKER)
    I = {}
    for n in (1, 2):
        out = tmp_path / f"o{n}"
        argv = mpi.command(n) + [sys.executable, str(script), str(out), str(order)]
        subprocess.run(argv, check=True, env=mpi.environment(argv, dict(os.environ, PYTHONPATH=_child_pythonpath())),
                       timeout=600)
        parts = [json.loads(Path(f"{out}.{r}").read_text()) for r in range(n)]
        assert all(p["conv"] and p["ranks"] == n for p in parts)
        I[n] = sum(p["I"] for p in parts)
    assert I[1] > 0
    assert abs(I[2] / I[1] - 1) < 1e-11


_MISUSE = r"""
import sys, numpy as np
from axionfem.chem import CurrentDistribution3D, Reaction, meshes3d
m = meshes3d.box(2)
out = []
cd = CurrentDistribution3D(m, order=1)
cd.set_conductivity(1.0)
cd.add_electrode("a", [1], reactions=[Reaction("bv", i0=10)], galvanostatic=True, current=1e-3)
cd.add_electrode("c", [2], reactions=[Reaction("bv", i0=10)], phi_s=0.0)
for what, fn in (("phi_s", lambda: cd.phi_s()),
                 ("index", lambda: CurrentDistribution3D(meshes3d.TetMesh(m.vertices, m.T + 1000, m.eattr, m.B, m.battr))),
                 ("degenerate", lambda: CurrentDistribution3D(meshes3d.TetMesh(m.vertices, np.vstack([m.T, [[0, 1, 2, 2]]]),
                                                                             np.ones(len(m.T) + 1, np.int32), m.B, m.battr)))):
    try:
        fn()
        out.append(what + ":accepted")
    except RuntimeError as e:
        out.append(what + ":" + str(e)[:80].replace(" ", "_"))
print(" ".join(out))
"""


@pytest.mark.skipif(not hasattr(_c, "CurrentDistribution3D") or not mpi.available() or not os.environ.get("AXF_MPI"),
                    reason="needs the MPI build, an MPI launcher (srun, mpirun or mpiexec) and AXF_MPI=1")
def test_3d_refuses_bad_input(tmp_path):
    """phi_s before a solve, out-of-range vertex indices and degenerate tetrahedra are clear errors."""
    script = tmp_path / "m.py"
    script.write_text(_MISUSE)
    argv = mpi.command(1) + [sys.executable, str(script)]
    out = subprocess.run(argv, env=mpi.environment(argv, dict(os.environ, PYTHONPATH=_child_pythonpath())),
                         capture_output=True, text=True, timeout=300, check=True).stdout
    assert "phi_s:" in out and "no_state" in out
    assert "outside" in out and "degenerate" in out
    assert "accepted" not in out
