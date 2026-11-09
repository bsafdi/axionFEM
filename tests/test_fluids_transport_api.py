"""Scalar transport and Boussinesq through the fluids Python API (MPI builds): the refusals and conventions that the
fixed cases do not exercise.

  - ScalarTransport / solve_heat REFUSE more than one MPI rank (mpirun -np 2) instead of solving one problem per partition
  - the inward flux convention (T = x for T(0) = 0, g = +1 at x = 1)
  - `outflow` under conservative_flux is refused (it would be a zero TOTAL-flux wall); convective and
    conservative_divergence let the advective flux leave (the outlet probe: T = 1 everywhere)
  - total-flux Robin under conservative_flux is the documented semantics; flux is refused there
  - Boussinesq: lu_backend is recorded, gravity spellings per geometry, the BoussinesqSolution velocity is refused by solve_heat
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
if not hasattr(fluids, "solve_heat") or not hasattr(fluids, "solve_boussinesq"):
    pytest.skip("this build has no scalar transport", allow_module_level=True)

B, R, T, L = 1, 2, 3, 4  # bottom, right, top, left

MESH_SRC = textwrap.dedent('''
    import numpy as np
    from axionfem import fluids
    B, R, T, L = 1, 2, 3, 4
    def rect(nx, ny):
        xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
        V = np.array([[x, y] for y in ys for x in xs])
        vid = lambda i, j: j * (nx + 1) + i
        tri = []
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
                tri += [[a, b, c], [a, c, d]]
        seg, att = [], []
        for i in range(nx):
            seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, ny), vid(i, ny)]]
            att += [B, T]
        for j in range(ny):
            seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
            att += [R, L]
        return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                       np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))
''')
exec(MESH_SRC)  # defines rect (module level, like the 2-rank script below)


def _heat(mesh, bcs, form="convective", kappa=0.5, **kw):
    if form != "convective":
        kw.setdefault("div_velocity", lambda X, Y: 0 * X)  # b = (1, 0) is divergence free
    return fluids.solve_heat(mesh, kappa, velocity=lambda X, Y: np.stack([1 + 0 * X, 0 * X], -1), source=0.0, bcs=bcs,
                             form=form, lu="eigen", **kw)


def _outlet_bcs(outlet_kind):
    return [dict(attribute=L, kind="dirichlet", value=1.0), dict(attribute=R, kind=outlet_kind),
            dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")]


def test_inward_flux_convention():
    """T(0) = 0 and the INWARD flux g = n.kappa grad T = +1 at x = 1 give T = x (kappa = 1, no advection)."""
    bcs = [dict(attribute=L, kind="dirichlet", value=0.0), dict(attribute=R, kind="flux", value=1.0),
           dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")]
    s = fluids.solve_heat(rect(6, 6), 1.0, source=0.0, bcs=bcs, lu="eigen")
    assert s.diagnostics["converged"]
    X, Tn = s.boundary_nodes([R])
    assert np.max(np.abs(np.asarray(Tn) - np.asarray(X)[:, 0])) < 1e-10


def test_outlet_probe_convective_passes_advective_flux():
    s = _heat(rect(8, 8), _outlet_bcs("outflow"))
    assert s.diagnostics["converged"]
    assert np.max(np.abs(np.asarray(s.field()) - 1.0)) < 1e-9


@pytest.mark.parametrize("kind", ["outflow"])
def test_outflow_refused_under_conservative_flux(kind):
    with pytest.raises(Exception, match="outflow"):
        _heat(rect(4, 4), _outlet_bcs(kind), form="conservative_flux")


def test_conservative_divergence_outflow_is_fine():
    s = _heat(rect(8, 8), _outlet_bcs("outflow"), form="conservative_divergence")
    assert s.diagnostics["converged"]
    assert np.max(np.abs(np.asarray(s.field()) - 1.0)) < 1e-9


def test_conservative_flux_robin_is_total_flux_and_flux_is_refused():
    with pytest.raises(Exception, match="conservative_flux"):
        _heat(rect(4, 4), [dict(attribute=L, kind="dirichlet", value=1.0), dict(attribute=R, kind="flux", value=0.0),
                           dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")], form="conservative_flux")
    # total-flux Robin j_out = h (T - T_ext) with h = beta.n = 1, T_ext = 0 (j_out = T) passes the uniform state T = 1 (pure advection
    # through the outlet; the wall is a zero total flux)
    bcs = [dict(attribute=L, kind="dirichlet", value=1.0), dict(attribute=R, kind="robin", h=1.0, T_ext=0.0),
           dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")]
    s = _heat(rect(8, 8), bcs, form="conservative_flux")
    assert s.diagnostics["converged"]
    assert np.max(np.abs(np.asarray(s.field()) - 1.0)) < 1e-8


def _cavity(geometry="planar", gravity="-y", n=4):
    p = fluids.natconv_cavity_parameters(1e3, 0.71)
    return fluids.solve_boussinesq(
        rect(n, n), p["mu"], p["rho0"], p["cp"], p["k"], p["beta_g"], T_ref=0.0, gravity=gravity, geometry=geometry,
        velocity_bcs=[dict(attribute=a, kind="dirichlet") for a in (B, R, T, L)],
        temperature_bcs=[dict(attribute=L, kind="dirichlet", value=0.5), dict(attribute=R, kind="dirichlet", value=-0.5),
                         dict(attribute=B, kind="zero"), dict(attribute=T, kind="zero")], lu="eigen")


def test_boussinesq_records_lu_backend():
    s = _cavity()
    d = s.diagnostics
    assert d["converged"] and d["lu_backend"] == "eigen", d["lu_backend"]


def test_boussinesq_gravity_spellings_per_geometry():
    with pytest.raises(ValueError, match="planar"):
        _cavity(gravity="-z")
    with pytest.raises(ValueError, match="axisymmetric"):
        _cavity(geometry="axisymmetric", gravity="-y")
    with pytest.raises(ValueError, match="gravity"):
        _cavity(gravity="-x")


def test_solve_heat_refuses_a_boussinesq_velocity():
    s = _cavity()
    with pytest.raises(Exception):
        fluids.solve_heat(rect(4, 4), 1.0, velocity=s, bcs=[dict(attribute=a, kind="zero") for a in (B, R, T, L)],
                          lu="eigen")


RANK_SCRIPT = MESH_SRC + textwrap.dedent('''
    import sys
    mesh = rect(8, 8)
    bcs = [dict(attribute=4, kind="dirichlet", value=0.0), dict(attribute=2, kind="dirichlet", value=1.0),
           dict(attribute=1, kind="zero"), dict(attribute=3, kind="zero")]
    try:
        s = fluids.solve_heat(mesh, 1.0, velocity=lambda X, Y: np.stack([1 + 0 * X, 0 * X], -1), source=0.0, bcs=bcs,
                              supg=True, lu="eigen")
        print("RESULT solved converged=%s" % s.diagnostics["converged"], flush=True)
    except Exception as e:
        print("RESULT refused: %s" % str(e).replace("\\n", " ")[:300], flush=True)
''')


def test_solve_heat_refuses_two_mpi_ranks(tmp_path):
    """On 2 ranks every rank REFUSES (no silent per-partition solve returning converged = True)."""
    import _mpi_launch as mpi   # tests/_mpi_launch.py: srun, mpirun or mpiexec
    if not mpi.available():
        pytest.skip(mpi.skip_reason())
    script = tmp_path / "rank2.py"
    script.write_text(RANK_SCRIPT)
    argv = mpi.command(2) + [sys.executable, str(script)]
    env = mpi.environment(argv)
    env.setdefault("OMPI_MCA_pml", "ob1")
    env.setdefault("OMPI_MCA_btl", "self,vader")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("RESULT")]
    assert len(lines) == 2, (r.returncode, r.stdout[-800:], r.stderr[-800:])
    assert all(ln.startswith("RESULT refused") and "one MPI rank" in ln for ln in lines), lines
