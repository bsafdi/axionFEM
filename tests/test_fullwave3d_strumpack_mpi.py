"""The distributed STRUMPACK factor of the closed, lossless full-wave direct path against the rank-zero factor.

AXF_FW_STRUMPACK_DIST=1 factorizes K - w^2 M across the ranks (SparseSolverMPIDist); =0 gathers it to rank zero
(the original path). Both precondition the same GMRES against the same distributed operator, so Z and Psi agree to
the GMRES tolerance whatever the factor; the evidence that the distributed factor is the same preconditioner is the
GMRES iteration count of every solve (ports, direct source, and the resistively loaded validation solves, the only
ones with both columns of the 2-column solve nonzero), and the fill. The inertia records must agree.
Needs a build against an MPI STRUMPACK (CMake AXF_STRUMPACK_PREFIX, scripts/build_strumpack_mpi.sh) and AXF_MPI
with at least 3 slots. With the usual serial build/ it skips; configure build/ with AXF_STRUMPACK_PREFIX and run
`sbatch --ntasks=4 scripts/slurm/tests.slurm tests/test_fullwave3d_strumpack_mpi.py`.
The two-port frame at order 1 (closed PEC sphere): a few seconds per solve.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import axionfem as af

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs >= 3 MPI slots; set AXF_MPI=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

DESIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"


def _solve(tmp_path, monkeypatch, mode, ranks):
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    monkeypatch.setenv("AXF_FW_STRUMPACK_DIST", mode)
    doc = json.loads(DESIGN.read_text())
    return af.Design(doc).solve(tmp_path / f"dist{mode}_np{ranks}", verbose=0, mpi_ranks=ranks).data["fullwave"]


def _z(row):
    return np.array(row["Z_matrix_re_ohm"]) + 1j * np.array(row["Z_matrix_im_ohm"])


def _psi(row):
    return np.array(row["Psi_vector"]["re"]) + 1j * np.array(row["Psi_vector"]["im"])


def _iterations(row):
    its = [p["solver"]["gmres_iterations"] for p in row["port_solutions"]]
    return its + [row["direct_source_solver"]["gmres_iterations"], row["direct_loaded_solver"]["gmres_iterations"]]


def _same_preconditioner(r0, r1):
    """r1 (distributed) against r0 (rank zero) at one frequency."""
    f0, f1 = r0["direct_factorization"], r1["direct_factorization"]
    # the same matrix, ordered by METIS on a different graph layout: fill within 1 %
    assert abs(f1["factor_nonzeros"] / f0["factor_nonzeros"] - 1) < 1e-2, (f0["factor_nonzeros"], f1["factor_nonzeros"])
    i0, i1 = _iterations(r0), _iterations(r1)
    assert all(b <= a + 1 for a, b in zip(i0, i1)), (i0, i1)
    # the same inertia record (STRUMPACK 8.0.0 fails the backend self-check in both builds and withholds the counts)
    assert f1["inertia"] == f0["inertia"], (f0["inertia"], f1["inertia"])
    Z0, Z1 = _z(r0), _z(r1)
    assert np.max(np.abs(Z1 - Z0)) < 1e-8 * np.max(np.abs(Z0)), (Z0, Z1)
    P0, P1 = _psi(r0), _psi(r1)
    assert np.max(np.abs(P1 - P0)) < 1e-8 * np.max(np.abs(P0)), (P0, P1)


@pytest.fixture
def mpi_strumpack():
    if not af.build_info().get("strumpack_mpi"):
        pytest.skip("needs a build against an MPI STRUMPACK (AXF_STRUMPACK_PREFIX)")


def test_distributed_factor_matches_rank_zero(tmp_path, monkeypatch, mpi_strumpack):
    ref = _solve(tmp_path, monkeypatch, "0", 2)
    dist = _solve(tmp_path, monkeypatch, "1", 2)
    assert len(ref["sweep"]) == len(dist["sweep"]) == 2
    for r0, r1 in zip(ref["sweep"], dist["sweep"]):
        assert r0["direct_factorization"]["factor_owner_rank"] == 0
        assert r0["direct_factorization"]["factor_ranks"] == 1
        assert r1["direct_factorization"]["backend"].startswith("STRUMPACK MPI")
        assert r1["direct_factorization"]["factor_ranks"] == 2
        _same_preconditioner(r0, r1)


def test_distributed_factor_on_one_and_three_ranks(tmp_path, monkeypatch, mpi_strumpack):
    ref = _solve(tmp_path, monkeypatch, "0", 1)
    one = _solve(tmp_path, monkeypatch, "1", 1)
    three = _solve(tmp_path, monkeypatch, "auto", 3)   # auto: distributed on several ranks; an odd process grid
    assert len(ref["sweep"]) == len(one["sweep"]) == len(three["sweep"]) == 2
    for r0, r1, r3 in zip(ref["sweep"], one["sweep"], three["sweep"]):
        assert r1["direct_factorization"]["factor_ranks"] == 1
        assert r3["direct_factorization"]["factor_ranks"] == 3
        _same_preconditioner(r0, r1)
        _same_preconditioner(r0, r3)


def test_bad_mode_is_refused(tmp_path, monkeypatch):
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    with pytest.raises(Exception, match="AXF_FW_STRUMPACK_DIST must be auto, 0 or 1"):
        _solve(tmp_path, monkeypatch, "yes", 1)
