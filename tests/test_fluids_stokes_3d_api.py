"""3D Stokes through the fluids Python API (axionfem.fluids, MPI builds; one rank): 3D meshes from arrays (tetrahedra,
hexahedra, curved P2/Q2 geometry), the API's 3D Taylor-Hood counts, evaluation at INPUT reference points by fixed
element id, P2/Q2 patch tests in both viscous forms, 3D slip/symmetry frames (one and two constraints), the
traction load on curved faces, the Babuska-Miller reaction and the MPI helpers.  Seconds on a login node."""
import sys
from pathlib import Path

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "fluids" / "cases"))
import f1c_stokes as R  # noqa: E402
import f1c_stokes_check as C  # noqa: E402


def api_mesh(fm, spec, curved=True):
    faces = R.Faces(spec)
    bf, ba = R.boundary_faces(fm, faces)
    nodes = None if (fm.X is None or not curved) else np.ascontiguousarray(fm.X, float)
    return fluids.Mesh3D.from_arrays(np.ascontiguousarray(fm.V, float), np.ascontiguousarray(fm.E, np.int32), bf, ba,
                                     nodes=nodes), faces


def solve(mesh, faces, u, form, mu=1.0, force=None, kinds=None, **kw):
    kinds = kinds or {}
    bcs = []
    for nm in faces.names:
        k = kinds.get(nm, "dirichlet")
        b = {"attribute": faces.attr(nm), "kind": k}
        if k == "dirichlet":
            b["value"] = R._xyz(u)
        bcs.append(b)
    return fluids.solve_stokes_3d(mesh, mu, force, bcs, form=form, **kw)


@pytest.mark.parametrize("spec", [{"family": "kuhn", "N": 2}, {"family": "hex", "N": 2},
                                  {"family": "standard", "N": 2}, {"family": "five", "N": 2}])
def test_taylor_hood_counts_match_check_module(spec):
    fm = R.build_mesh(spec)
    mesh, _ = api_mesh(fm, spec)
    api = dict(mesh.taylor_hood_check())
    for k, v in C.taylor_hood_condition(fm).items():
        assert api[k] == v, (spec, k, api, v)


@pytest.mark.parametrize("spec", [{"family": "hex", "N": 2}, {"family": "sphere_hex", "n": 2},
                                  {"family": "sphere_tet", "n": 2}, {"family": "dist", "k": 0, "N0": 4, "rel": 0.2,
                                                                     "seed": 20261101}])
def test_mesh_round_trip_and_geometry_nodes(spec):
    """The solver's mesh hashes like the fixed one (curved nodes included) and its map is the fixed map."""
    fm = R.build_mesh(spec)
    mesh, _ = api_mesh(fm, spec)
    X = np.asarray(mesh.geometry_nodes()) if fm.X is not None else None
    sm = C.FrozenMesh(fm.kind, mesh.vertices(), np.asarray(mesh.elements(), np.int64), X=X)
    assert C.mesh_hash(sm) == C.mesh_hash(fm)
    if fm.kind == "hex":
        assert np.array_equal(np.asarray(mesh.elements()), fm.E)


def test_quadratic_patch_both_forms_tets_and_hexes():
    """u in P2/Q2 and p in P1 with the matching forcing are reproduced to round-off; evaluation by fixed id at
    input reference points reproduces the fixed map (geometry_defect round-off)."""
    def u(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        return np.stack([y * y + z, -x * z + 0 * y, x * y + 0 * z], -1)       # div u = 0, quadratic

    for spec in ({"family": "kuhn", "N": 2}, {"family": "hex", "N": 2}, {"family": "dist", "k": 0, "N0": 4,
                                                                          "rel": 0.2, "seed": 20261101}):
        fm = R.build_mesh(spec)
        mesh, faces = api_mesh(fm, spec)
        for form in ("laplacian", "stress"):
            # -div(sigma) + grad p = f with mu = 1: lap u = (2, 0, 0); p = x + 2y - z (mean irrelevant)
            f = R._xyz(lambda P: np.broadcast_to(np.array([-2.0 + 1.0, 2.0, -1.0]), P.shape).copy())
            sol = solve(mesh, faces, u, form, force=f)
            d = sol.diagnostics
            assert d["converged"] and d["comm_size"] == 1 and d["shared_dofs"] == 0
            ev = R.Ev(sol)
            assert np.array_equal(np.sort(ev.ids), np.arange(fm.n_elements))
            g = C.combine([C.geometry_defect(fm, ev)])
            assert g["geometry_x"] < 1e-13 and g["geometry_detJ_rel"] < 1e-12
            q, _ = C.rule(fm.kind)
            x, _ = ev.geometry(q)
            uh, _ = ev.velocity(q)
            assert np.max(np.abs(uh - u(x))) < 1e-11, (spec, form)
            ph = ev.pressure(q)
            pe = x[..., 0] + 2 * x[..., 1] - x[..., 2]
            assert np.ptp(ph - pe) < 1e-10, (spec, form)


def test_rotated_slip_symmetry_edge_uniform_flow():
    """Uniform flow along the slip-symmetry edge of the rotated cube: in P2, reproduced exactly; the edge nodes get
    two constraints (u.n1 = u.n2 = 0), interior slip nodes one."""
    case = __import__("json").loads((Path(__file__).resolve().parents[1] / "benchmarks/fluids/registry/cases"
                                     / "mms_stokes_3d_slip.json").read_text())
    P = case["run"]["parameters"]
    F = C.load_field(P["slip_patch"]["field"])
    Rm = np.array(F.data["frame"]["R_float"])
    spec = {"family": "rot", "N": 2, "R": Rm.tolist()}
    fm = R.build_mesh(spec)
    mesh, faces = api_mesh(fm, spec)
    for form in ("laplacian", "stress"):
        sol = solve(mesh, faces, F.u, form, kinds={"z0": "slip", "x1": "symmetry"})
        d = sol.diagnostics
        assert d["converged"] and d["nullspace_detected"] and d["slip_nodes"] > 0 and d["slip_corner_nodes"] == 0
        ev = R.Ev(sol)
        b = C.combine([C.boundary_node_sums(fm, ev, [faces.pred(n) for n in ("x0", "y0", "y1", "z1")], F.u,
                                            [(faces.pred("z0"), faces.normal("z0")),
                                             (faces.pred("x1"), faces.normal("x1"))])])
        assert b["slip_un"] < 1e-13 and b["dirichlet_nodal_defect"] < 1e-14
        n = C.combine([C.norm_sums(fm, ev, F)])
        assert n["uL2"] < 1e-12


def test_curved_shell_volume_and_reaction_sign():
    """Curved Q2 shell: the element map integrates the shell volume to O(h^4); a uniform pressure p0 (u = 0) gives
    no net force; the reaction is collective and returns three components."""
    spec = {"family": "sphere_hex", "n": 2}
    fm = R.build_mesh(spec)
    mesh, faces = api_mesh(fm, spec)
    zero = R._const_vec((0, 0, 0))
    sol = solve(mesh, faces, zero, "stress", mu=0.37)
    ev = R.Ev(sol)
    class Zero:                                                 # any field: only the volume is used
        def u(self, P):
            return np.zeros(np.shape(P))

        def grad_u(self, P):
            return np.zeros(np.shape(P)[:-1] + (3, 3))

        def p(self, P):
            return np.zeros(np.shape(P)[:-1])
    n = C.combine([C.norm_sums(fm, ev, Zero())])
    assert abs(n["volume"] - 84 * np.pi) / (84 * np.pi) < 2e-3
    r = sol.reaction([faces.attr("body")], "stress")
    assert len(r["force"]) == 3 and max(abs(v) for v in r["force"]) < 1e-10


def test_mpi_helpers_one_rank():
    f = fluids._core._fluids
    assert f.world_size() == 1 and f.world_rank() == 0
    assert np.allclose(f.allreduce_sum(np.array([1.0, 2.0])), [1.0, 2.0])


def test_refusals():
    spec = {"family": "kuhn", "N": 2}
    fm = R.build_mesh(spec)
    faces = R.Faces(spec)
    bf, ba = R.boundary_faces(fm, faces)
    with pytest.raises(fluids.FluidsError):          # hexahedra need quadrilateral boundary faces
        fluids.Mesh3D.from_arrays(np.ascontiguousarray(fm.V, float), np.zeros((1, 8), np.int32), bf, ba)
    with pytest.raises(fluids.FluidsError):          # curved nodes of the wrong size
        fluids.Mesh3D.from_arrays(np.ascontiguousarray(fm.V, float), np.ascontiguousarray(fm.E, np.int32), bf, ba,
                                  nodes=np.zeros((fm.n_elements, 9, 3)))
    mesh, faces = api_mesh(fm, spec)
    with pytest.raises(fluids.FluidsError):          # an attribute without a condition
        fluids.solve_stokes_3d(mesh, 1.0, None, [{"attribute": 1, "kind": "dirichlet"}])


def test_3d_reaction_solved_is_the_physical_force_on_a_partial_boundary():
    """sol.reaction(attrs, "solved") on a 3D solution is the PHYSICAL force, as on the
    2D/axisymmetric solution. The Laplacian-form solution of a divergence-free quadratic field has the same u, p as
    the stress-form one; on ONE face the raw Laplacian residual (the pseudo-traction) differs from the Cauchy force
    by the null-Lagrangian term int (grad u^T) n = (0, 1, 1) there, and 'solved' must return the stress-form force."""
    def u(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        return np.stack([y * y + z, -x * z + 0 * y, x * y + 0 * z], -1)

    spec = {"family": "kuhn", "N": 2}
    fm = R.build_mesh(spec)
    mesh, faces = api_mesh(fm, spec)
    f = R._xyz(lambda P: np.broadcast_to(np.array([-1.0, 2.0, -1.0]), P.shape).copy())
    lap = solve(mesh, faces, u, "laplacian", force=f)
    sym = solve(mesh, faces, u, "stress", force=f)
    a = [faces.attr("x1")]
    f_phys = np.array(sym.reaction(a, "solved")["force"])
    f_lap_solved = np.array(lap.reaction(a, "solved")["force"])
    f_lap_raw = np.array(lap.reaction(a, "laplacian")["force"])
    assert np.max(np.abs(f_lap_solved - f_phys)) < 1e-9, (f_lap_solved, f_phys)
    assert np.max(np.abs(f_lap_raw - f_phys)) > 0.5, (f_lap_raw, f_phys)        # the pseudo-traction differs


def test_krylov_fallback_is_3d_only_and_switchable():
    """A 2D solve never reports LU GMRES iterations or the gmres label; the 3D
    solve takes lu_krylov_fallback (default True) and reports the boolean and the count; a normal 3D solve reports
    no fallback."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_fluids_stokes_planar_api import rect
    d2 = fluids.solve_stokes(rect(6, 6), 1.0, None, [{"attribute": a, "kind": "dirichlet",
                                                       "value": (lambda x, y: (1.0 if y > 1 - 1e-12 else 0.0, 0.0))}
                                                      for a in (1, 2, 3, 4)]).diagnostics
    assert d2.get("lu_krylov_iterations", 0) == 0 and not d2.get("lu_krylov_fallback", False)
    assert "gmres" not in d2["solver"]
    spec = {"family": "kuhn", "N": 2}
    fm = R.build_mesh(spec)
    mesh, faces = api_mesh(fm, spec)
    for flag in (True, False):
        d3 = solve(mesh, faces, R._const_vec((0, 0, 0)), "stress", lu_krylov_fallback=flag).diagnostics
        assert d3["converged"] and d3["lu_krylov_iterations"] == 0 and d3["lu_krylov_fallback"] is False
        assert "gmres" not in d3["solver"]
