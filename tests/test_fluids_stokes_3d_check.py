"""Self-tests of the 3D Stokes check module benchmarks/fluids/cases/f1c_stokes_check.py.

No solver: the discrete "solutions" are P2 (tet) / Q2 (hex) nodal INTERPOLANTS of the exact fields, built here from
the module's own fixed geometry and Lagrange bases, whose error orders are known (3 / 2 in L2 / H1), plus
exact-field evaluators for the drag functional and the identities.  Seconds on a login node: `pytest -m fast`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.dont_write_bytecode = True
import f1c_stokes_check as C  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


class InterpEval:
    """Nodal interpolant (P2/Q2 velocity at the geometry nodes, P1/Q1 pressure at the vertices) of an exact field."""

    def __init__(self, mesh, fld, ids=None, exact=False):
        self.m, self.fld, self.exact = mesh, fld, exact
        self.ids = np.arange(mesh.n_elements) if ids is None else np.asarray(ids)
        self.N = mesh.geometry_nodes()[self.ids]

    def geometry(self, ref):
        x, J = self.m.map(ref, self.ids)
        return x, np.linalg.det(J)

    def velocity(self, ref):
        if self.exact:
            x, _ = self.m.map(ref, self.ids)
            return self.fld.u(x), self.fld.grad_u(x)
        kind = "p2" if self.m.kind == "tet" else "q2"
        phi, dphi = C.lagrange(kind, ref)
        un = self.fld.u(self.N)
        _, J = self.m.map(ref, self.ids)
        u = np.einsum("qb,ebi->eqi", phi, un)
        gref = np.einsum("qbk,ebi->eqik", dphi, un)
        return u, np.einsum("eqik,eqkj->eqij", gref, np.linalg.inv(J))

    def pressure(self, ref):
        x, _ = self.m.map(ref, self.ids)
        if self.exact:
            return self.fld.p(x)
        kind = "p1" if self.m.kind == "tet" else "q1"
        phi, _ = C.lagrange(kind, ref)
        verts = self.m.V[self.m.E[self.ids]]
        return np.einsum("qb,eb->eq", phi, self.fld.p(verts))


@fast
def test_quadrature_exactness():
    q, w = C.tet_rule()
    assert abs(w.sum() - 1 / 6) < 1e-15
    for (i, j, k) in [(8, 0, 0), (0, 8, 0), (0, 0, 8), (3, 3, 2), (2, 2, 4), (1, 5, 2)]:
        ex = math.factorial(i) * math.factorial(j) * math.factorial(k) / math.factorial(i + j + k + 3)
        assert abs((w * q[:, 0] ** i * q[:, 1] ** j * q[:, 2] ** k).sum() / ex - 1) < 1e-12
    q, w = C.hex_rule()
    assert abs((w * q[:, 0] ** 9 * q[:, 2] ** 9).sum() - 0.01) < 1e-14


@fast
@pytest.mark.parametrize("kind", ["p2", "q2", "p1", "q1"])
def test_lagrange_bases(kind):
    nodes = {"p2": C.REF_P2_TET, "q2": C.REF_Q2_HEX, "p1": C.REF_TET_VERTS, "q1": C.REF_Q2_HEX[:8]}[kind]
    phi, dphi = C.lagrange(kind, nodes)
    assert np.allclose(phi, np.eye(len(nodes)), atol=1e-12)
    rng = np.random.default_rng(1)
    pts = rng.random((7, 3)) * (0.3 if kind in ("p1", "p2") else 1.0)
    phi, dphi = C.lagrange(kind, pts)
    assert np.allclose(phi.sum(1), 1) and np.allclose(dphi.sum(1), 0, atol=1e-12)
    # reproduces x*y (in the space for p2/q2 only) and x exactly with its gradient
    f = nodes[:, 0] * (nodes[:, 1] if kind in ("p2", "q2") else 1.0)
    want = pts[:, 0] * (pts[:, 1] if kind in ("p2", "q2") else 1.0)
    assert np.allclose(phi @ f, want, atol=1e-12)


@fast
def test_mesh_families_and_taylor_hood_condition():
    for N in (2, 4, 8):
        m = C.cube_tet(N)
        assert m.n_elements == 6 * N ** 3
        assert C.taylor_hood_condition(m) == {"tets_without_interior_vertex": 0, "tets_violating_weak_boffi": 0,
                                              "tets_all_p2_nodes_on_boundary": 0}
        assert len(m.boundary_faces()) == 12 * N * N and {len(v) for v in m.faces().values()} == {1, 2}
        s = C.taylor_hood_condition(C.cube_tet(N, "standard"))
        assert s["tets_without_interior_vertex"] > 0 and s["tets_all_p2_nodes_on_boundary"] == 0
        assert s["tets_violating_weak_boffi"] == s["tets_without_interior_vertex"]
        f = C.taylor_hood_condition(C.five_tet_cube(N))
        assert f["tets_all_p2_nodes_on_boundary"] > 0
    # nested: the Bey refinement of N = 4 IS the fixed N = 8 mesh; the distorted family refines its base
    m4 = C.cube_tet(4)
    V, T = C.bey_refine(m4.V, m4.E)
    assert C.mesh_hash(C.FrozenMesh("tet", V, T)) == C.mesh_hash(C.cube_tet(8))
    assert C.nested_in(C.cube_tet(8), m4)
    assert C.nested_in(C.cube_tet_distorted(1), C.cube_tet_distorted(0))
    d0 = C.cube_tet_distorted(0)
    assert np.max(np.abs(d0.V - C.cube_tet(4).V)) > 0.01          # really distorted, boundary fixed
    assert C.taylor_hood_condition(d0)["tets_without_interior_vertex"] == 0
    # numbering-independent hash, and a moved vertex changes it
    perm = np.random.default_rng(3).permutation(m4.n_elements)
    assert C.mesh_hash(C.FrozenMesh("tet", m4.V, m4.E[perm])) == C.mesh_hash(m4)
    V2 = m4.V.copy()
    V2[40] += 1e-6
    assert C.mesh_hash(C.FrozenMesh("tet", V2, m4.E)) != C.mesh_hash(m4)
    vm = C.verify_mesh(C.cube_tet(8), C.cube_tet(8), coarser=m4)
    assert vm["hash_match"] and vm["counts_match"] and vm["nested"] and vm["tets_without_interior_vertex"] == 0


@fast
@pytest.mark.parametrize("kind", ["hex", "tet"])
def test_sphere_shell_geometry(kind):
    errs = []
    for n in (2, 4):
        m = C.sphere_shell(n, kind)
        assert {len(v) for v in m.faces().values()} == {1, 2}           # conforming across the six patches
        q, w = C.rule(m.kind)
        _, J = m.map(q)
        dj = np.linalg.det(J)
        assert np.all(dj > 0)
        errs.append(abs((w[None] * dj).sum() - 4 / 3 * math.pi * 63))
        r = np.linalg.norm(m.geometry_nodes(), axis=-1)
        assert np.all(r > 1 - 1e-12) and np.all(r < 4 + 1e-12)
        if kind == "tet":
            assert C.taylor_hood_condition(m)["tets_without_interior_vertex"] == 0
    assert math.log2(errs[0] / errs[1]) > 3.7                            # isoparametric volume: order 4


@fast
def test_interpolation_orders_and_partial_sums():
    fld = C.load_field(entry("f1c_cube_u_mu037"))       # u = curl A and p = P: both norms non-trivial
    for builder in (C.cube_tet, C.cube_hex):
        e = []
        for N in (4, 8):
            m = builder(N)
            parts = [C.norm_sums(m, InterpEval(m, fld, ids), fld) for ids in np.array_split(np.arange(m.n_elements), 3)]
            whole = C.combine([C.norm_sums(m, InterpEval(m, fld), fld)])
            split = C.combine(parts)
            for k in ("uL2", "uH1", "pL2m", "pL2", "divL2"):
                assert abs(split[k] - whole[k]) <= 1e-12 * whole[k]
            e.append(whole)
        assert 2.8 < math.log2(e[0]["uL2"] / e[1]["uL2"]) < 3.3
        assert 1.8 < math.log2(e[0]["uH1"] / e[1]["uH1"]) < 2.3
        assert 1.8 < math.log2(e[0]["pL2m"] / e[1]["pL2m"]) < 2.3
    with pytest.raises(RuntimeError):
        C.load_field({"path": "tools/fluids_mms/out/f1c_cube_u.json", "sha256": "0" * 64})


@fast
def test_rotated_field_frame():
    d = json.loads((ROOT / "tools/fluids_mms/out/f1c_rot_slip_u.json").read_text())
    fld = C.Field(d)
    R = fld.R
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-15)
    m = C.rotated_cube_tet(4, R)
    P = m.V[:50] * 0.9 + 0.05 * (R @ np.ones(3))
    assert np.max(np.abs(np.trace(fld.grad_u(P), axis1=-2, axis2=-1))) < 1e-12        # div-free in physical frame
    n_slip = R @ np.array([0, 0, -1.0])
    face = (np.random.default_rng(2).random((20, 3)) * [1, 1, 0]) @ R.T                 # local z = 0
    assert np.max(np.abs(fld.u(face) @ n_slip)) < 1e-12
    for form in ("lap", "sym"):
        t = fld.traction(form, face, n_slip)
        tt = t - (t @ n_slip)[:, None] * n_slip
        assert np.max(np.abs(tt)) < 1e-12


@fast
def test_drag_functional_on_exact_field():
    for outer, F in (("unbounded", 111 * math.pi / 50), ("concentric", 12617 * math.pi / 2700)):
        fld = C.load_field(entry(f"f1c_sphere_{outer}"))
        assert abs(fld.constants["drag_z_exact"] - F) < 1e-12
        errs = []
        for n in (2, 4):
            m = C.sphere_shell(n, "hex")
            d = C.combine([C.drag_sums(m, InterpEval(m, fld, exact=True), mu=0.37)])["drag"]
            errs.append(abs(d - F) / F)
        assert errs[1] < 2e-3 and errs[1] < errs[0] / 8


@fast
def test_duct_series():
    ref = json.loads((ROOT / "tools/fluids_mms/duct/duct_series_reference.json").read_text())["quantities"]
    w, _, _ = C.duct_u(np.array([0.5]), np.array([0.25]))
    assert abs(w[0] - ref["w_centre"]["value"]) < 1e-15
    g, ww = np.polynomial.legendre.leggauss(60)
    s = 0.5 * (g + 1)
    X, Y = np.meshgrid(s, 0.5 * s, indexing="ij")
    Q = (np.outer(ww, ww) * 0.125 * C.duct_u(X, Y)[0]).sum()
    assert abs(Q - ref["Q"]["value"]) < 1e-11
    xs = np.array([0.0, 1.0, 0.3, 0.7, 0.0])
    ys = np.array([0.2, 0.1, 0.0, 0.5, 0.0])
    assert np.max(np.abs(C.duct_u(xs, ys)[0])) < 1e-15                     # walls
    h = 1e-3
    P = (np.array([0.3, 0.71, 0.2]), np.array([0.11, 0.4, 0.45]))
    lap = (C.duct_u(P[0] + h, P[1])[0] + C.duct_u(P[0] - h, P[1])[0] + C.duct_u(P[0], P[1] + h)[0]
           + C.duct_u(P[0], P[1] - h)[0] - 4 * C.duct_u(*P)[0]) / h ** 2
    assert np.max(np.abs(-lap - 2.0)) < 1e-4
    w0, gx, gy = C.duct_u(*P)
    hh = 1e-6
    assert np.max(np.abs(gx - (C.duct_u(P[0] + hh, P[1])[0] - C.duct_u(P[0] - hh, P[1])[0]) / (2 * hh))) < 1e-8


@fast
def test_boundary_helpers():
    fld = C.load_field(entry("f1c_cube_u"))
    m = C.cube_tet(4)
    tot, ref = C.p2_face_flux_defect(m, fld.u)
    assert ref > 0 and abs(tot) / ref < 1e-3                               # O(h^4) flux defect of P2 data
    tot8, ref8 = C.p2_face_flux_defect(C.cube_tet(8), fld.u)
    assert abs(tot8) < abs(tot) / 8
    P = m.V[:30]
    assert C.dirichlet_nodal_defect(P, fld.u(P), fld.u) == 0.0
    assert C.dirichlet_nodal_defect(P, fld.u(P) + 1e-9, fld.u) > 1e-10
    assert C.dirichlet_shared_node_mismatch(P[:3], [[fld.u, fld.u]] * 3) == 0.0
    n1, n2 = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    assert C.slip_nodal_un([np.array([0, 1, 0.0])], [[n1, n2]], 1.0) == 0.0
    assert C.slip_nodal_un([np.array([1e-3, 1, 0.0])], [[n1, n2]], 1.0) == pytest.approx(1e-3)
    assert C.slip_nodal_un([np.array([1, 1, 1.0])], [[n1]], 1.0, on_dirichlet=[True]) == 0.0


@fast
def test_inf_sup_counts_zero_modes():
    A = np.eye(3)
    B = np.array([[1.0, 0, 0], [0, 0, 0]])           # second pressure mode invisible to the velocity
    M = np.eye(2)
    r = C.inf_sup(A, B, M)
    assert r["zero_modes_beyond_constant"] == 0 and r["beta"] == pytest.approx(1.0)
    B3 = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert C.inf_sup(A, B3, np.eye(3))["zero_modes_beyond_constant"] == 1


@fast
def test_empty_rank_partial_sums():
    """A rank that owns no element contributes zero sums (coarse meshes on many ranks)."""
    fld = C.load_field(entry("f1c_cube_u_mu037"))
    m = C.cube_tet(2)
    e = InterpEval(m, fld, ids=np.array([], dtype=np.int64))
    full = C.combine([C.norm_sums(m, InterpEval(m, fld), fld)])
    both = C.combine([C.norm_sums(m, InterpEval(m, fld), fld), C.norm_sums(m, e, fld)])
    assert both["uL2"] == full["uL2"] and both["u_magnitude"] == full["u_magnitude"]
    assert C.geometry_defect(m, e)["max:geometry_x"] == 0.0 and C.drag_sums(m, e)["sum:drag"] == 0.0


class ShiftedEval(InterpEval):
    """The same discrete field with its pressure shifted by a constant (a different pressure normalization)."""

    def __init__(self, mesh, fld, c, ids=None):
        super().__init__(mesh, fld, ids)
        self.c = c

    def pressure(self, ref):
        return super().pressure(ref) + self.c


@fast
def test_two_pass_pressure_centering():
    """A constant pressure offset between two runs must not cost precision."""
    fld = C.load_field(entry("f1c_cube_u_mu037"))
    m = C.cube_tet(4)
    a = InterpEval(m, fld)
    for c in (1e-2, 0.5, 30.0):
        b = ShiftedEval(m, fld, c)
        shift = C.mean_of([C.pressure_mean_sums(m, a)]) - C.mean_of([C.pressure_mean_sums(m, b)])
        r = C.finish_difference([C.difference_sums(m, a, b, shift=shift)])
        assert r["p_rel"] < 1e-13 and abs(r["p_mean_diff"] - c) < 1e-12
    # pL2m with the two-pass shift equals the one-pass value for a well-centred field
    one = C.combine([C.norm_sums(m, a, fld)])
    pe_mean = one["pmean"]
    two = C.combine([C.norm_sums(m, a, fld, p_shift=one["pmean_h"] - pe_mean)])
    assert abs(two["pL2m"] - one["pL2m"]) < 1e-12 * one["pL2m"] + 1e-15 and two["pL2"] is None


@fast
def test_partition_report():
    ids = np.arange(100)
    good = C.partition_report(np.array_split(ids, 4), 100, 4)
    assert good == {"parts": 4, "parts_expected": 4, "empty_parts": 0, "duplicate_ids": 0, "missing_ids": 0,
                    "out_of_range_ids": 0}
    every_rank_all = C.partition_report([ids] * 4, 100, 4)          # whole problem on every rank
    assert every_rank_all["duplicate_ids"] == 100
    one_rank = C.partition_report([ids, [], [], []], 100, 4)         # solved on one rank
    assert one_rank["empty_parts"] == 3
    # a negative or too-large id is counted, not a crash; the id it displaced is then missing
    neg = C.partition_report([[-1, 0], [1]], 2, 2)
    assert neg["out_of_range_ids"] == 1 and neg["missing_ids"] == 0 and neg["duplicate_ids"] == 0, neg
    big = C.partition_report([[-3, 0], [5, 7]], 2, 2)
    assert big["out_of_range_ids"] == 3 and big["missing_ids"] == 1, big


@fast
def test_boundary_node_sums_and_energy():
    fld = C.load_field(entry("f1c_cube_u"))
    m = C.cube_tet(4)
    ev = InterpEval(m, fld)
    on_x0 = lambda X: np.abs(X[:, 0]) < 1e-12  # noqa: E731
    r = C.boundary_node_sums(m, ev, [on_x0], fld.u)
    assert r["max:dirichlet_nodal_defect"] < 1e-14 and r["sum:dirichlet_nodes"] > 0
    bad = C.boundary_node_sums(m, ShiftedVel(m, fld, 1e-9), [on_x0], fld.u)
    assert bad["max:dirichlet_nodal_defect"] > 1e-10
    # slip: a field tangent to z = 0 there has u.n = 0 at the nodes of that face
    d = json.loads((ROOT / "tools/fluids_mms/out/f1c_rot_slip_u.json").read_text())
    fr = C.Field(d)
    mr = C.rotated_cube_tet(4, fr.R)
    n_slip = fr.R @ np.array([0, 0, -1.0])
    on_slip = lambda X: np.abs(fr.local(X)[:, 2]) < 1e-12  # noqa: E731
    on_dir = lambda X: np.abs(fr.local(X)[:, 1]) < 1e-12  # noqa: E731
    rs = C.boundary_node_sums(mr, InterpEval(mr, fr), [on_dir], fr.u, slip=[(on_slip, n_slip)])
    assert rs["max:slip_un"] < 1e-13 and rs["sum:slip_nodes"] > 0
    e = C.combine([C.energy_sums(ev, m, "lap")])
    assert e["ekin"] > 0 and e["diss"] > 0


class ShiftedVel(InterpEval):
    def __init__(self, mesh, fld, c):
        super().__init__(mesh, fld)
        self.c = c

    def velocity(self, ref):
        u, g = super().velocity(ref)
        return u + self.c, g


# ------------------------------------------------------------------ rank proof in every 16-rank case
# the 3D Stokes case definitions (benchmarks/fluids/registry/cases, group F1c), in the layout {"cases": {id: case}}
CASES3D = {"cases": {c["id"]: c for c in (json.loads(f.read_text()) for f in
                                           sorted((ROOT / "benchmarks/fluids/registry/cases").glob("*.json")))
                      if c["capability"] == "F1c"}}
RANK_PROOF = ("comm_size_mismatch_runs", "partition_failures", "shared_dofs_min_runs")
# the coarsest fixed level(s) of each 16-rank case (the element counts the partition must cover)
RANK_CASE_MESHES = {
    "mms_stokes_3d": lambda: [C.cube_tet(4), C.cube_tet_distorted(0), C.cube_hex(4)],
    "mms_stokes_3d_traction": lambda: [C.cube_tet(4), C.cube_hex(4)],
    "mms_stokes_3d_do_nothing": lambda: [C.cube_tet(4)],
    "mms_stokes_3d_slip": lambda: [C.cube_tet(4)],
    "analytic_duct_poiseuille_3d": lambda: [C.duct_tet(0.25), C.duct_tet(0.125)],
    "analytic_stokes_sphere_3d": lambda: [C.sphere_shell(2, "hex"), C.sphere_shell(2, "tet")],
    "xcode_stokes_cavity_reg_3d": lambda: [C.cube_tet(8)],
}


def rank_proof_quantities(runs: list, n_ranks: int) -> dict:
    """THE REFERENCE RULE of the rank proof: the case runner
    computes the three quantities exactly so.  Input: one entry per run, {"level", "comm_size" (communicator size of
    the SOLVE), "shared_dofs" (shared/ghost true DOFs of the solve), "ids_per_rank" (the fixed element ids of each
    rank's evaluator), "n_elements"}.
    comm_size_mismatch_runs: runs whose size != ranks; partition_failures: levels where parts != ranks, a part is
    empty (when the level has >= ranks elements), or ids are duplicated, missing or out of range [0, n_elements)
    (C.partition_report); shared_dofs_min_runs: the minimum over runs of shared_dofs (proves the ASSEMBLY is distributed, not only the evaluation)."""
    mismatch = sum(int(r["comm_size"]) != n_ranks for r in runs)
    failed_levels = set()
    for r in runs:
        rep = C.partition_report(r["ids_per_rank"], r["n_elements"], n_ranks)
        if (rep["parts"] != n_ranks or rep["duplicate_ids"] or rep["missing_ids"] or rep["out_of_range_ids"]
                or (rep["empty_parts"] and r["n_elements"] >= n_ranks)):
            failed_levels.add(r["level"])
    return {"comm_size_mismatch_runs": mismatch, "partition_failures": len(failed_levels),
            "shared_dofs_min_runs": min(int(r["shared_dofs"]) for r in runs)}


def shared_vertices(mesh, parts) -> int:
    """A lower bound of the shared true DOFs of a solve partitioned as `parts`: the mesh vertices whose elements lie
    in more than one part (each carries velocity and pressure DOFs on two or more ranks)."""
    E = np.asarray(mesh.E, np.int64)
    owner = np.full(len(mesh.V), -1, np.int64)
    shared = np.zeros(len(mesh.V), bool)
    for r, ids in enumerate(parts):
        v = np.unique(E[np.asarray(ids, np.int64)].ravel())
        shared[v[owner[v] >= 0]] = True
        owner[v] = r
    return int(shared.sum())


def synthetic_runs(meshes: list, n_ranks: int, mode: str) -> list:
    """What a runner would report on each level.  'distributed': a correct 16-rank solve (a seeded permutation split
    into 16 parts, as a graph partitioner would); 'comm_self': 16 processes each solving the whole problem on
    MPI.COMM_SELF (communicator size 1, every part covers everything); 'one_rank': launched on 16 ranks, all
    elements owned by rank 0; 'serial': launched on 1 rank.  Two subtler fakes, which
    pass the first two checks: 'replicated': every rank solves the whole problem on an unpartitioned mesh, the solver
    object on COMM_WORLD (size 16), each rank evaluating the ids = r (mod 16); 'rank0_scatter': rank 0 solves alone,
    the solution is broadcast, each rank evaluates a slice and the runner reports COMM_WORLD.size.  Only the
    partitioned solve has shared DOFs."""
    rng = np.random.default_rng(20261101)
    out = []
    for i, m in enumerate(meshes):
        n = m.n_elements
        ids = np.arange(n)
        shared = 0
        if mode == "distributed":
            comm, parts = n_ranks, np.array_split(rng.permutation(n), n_ranks)
            shared = shared_vertices(m, parts)
        elif mode == "replicated":
            comm, parts = n_ranks, [ids[ids % n_ranks == r] for r in range(n_ranks)]
        elif mode == "rank0_scatter":
            comm, parts = n_ranks, np.array_split(ids, n_ranks)
        elif mode == "comm_self":
            comm, parts = 1, [ids] * n_ranks
        elif mode == "one_rank":
            comm, parts = n_ranks, [ids] + [ids[:0]] * (n_ranks - 1)
        elif mode == "serial":
            comm, parts = 1, [ids]
        else:
            raise ValueError(mode)
        out.append({"level": f"L{i}", "comm_size": comm, "shared_dofs": shared, "ids_per_rank": parts,
                    "n_elements": n})
    return out


def judged_rank_checks(case: dict, quantities: dict) -> dict:
    """benchmarks/fluids/acceptance.evaluate() on the case's acceptance block; the verdicts of the rank checks."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "fluids"))
    import acceptance  # noqa: E402
    out = acceptance.evaluate(case, {"runs": [], "series": {}, "quantities": quantities})
    return {c["quantity"]: c["verdict"] for c in out["checks"] if c["quantity"] in RANK_PROOF}


@fast
def test_every_16_rank_case_has_the_rank_proof():
    """The rank proof is in EVERY case run on 16 ranks, at one of the case's own levels; P1 and S1 also
    compare the API's Taylor-Hood counts."""
    sixteen = [cid for cid, c in CASES3D["cases"].items() if c["run"]["parameters"].get("ranks") == 16]
    assert set(sixteen) == set(RANK_CASE_MESHES)
    for cid in sixteen:
        c = CASES3D["cases"][cid]
        checks = {x["quantity"]: x for x in c["acceptance"]["checks"]}
        for q in RANK_PROOF:
            assert q in checks, (cid, q)
            bound = ("min", 1) if q == "shared_dofs_min_runs" else ("max", 0)
            assert checks[q].get(bound[0]) == bound[1] and checks[q]["level"] in c["levels"], (cid, q)
    for cid in ("analytic_duct_poiseuille_3d", "analytic_stokes_sphere_3d"):
        assert "api_th_check_disagreements" in {x["quantity"] for x in CASES3D["cases"][cid]["acceptance"]["checks"]}


@fast
@pytest.mark.parametrize("cid", sorted(RANK_CASE_MESHES))
def test_synthetic_non_distributed_run_is_caught(cid):
    """Non-distributed runs, synthesized as what the runner reports and judged by the acceptance check of the case:
    the correct 16-rank partition passes all three checks; COMM_SELF and a serial launch fail all three; a run
    whose work all sits on rank 0 fails partition_failures and shared_dofs_min_runs; the replicated solve
    with split evaluation and rank-0 solve with scattered evaluation fail shared_dofs_min_runs."""
    case = CASES3D["cases"][cid]
    n_ranks = case["run"]["parameters"]["ranks"]
    meshes = RANK_CASE_MESHES[cid]()
    assert all(m.n_elements >= n_ranks for m in meshes)
    want = {"distributed": "PPP", "comm_self": "FFF", "serial": "FFF", "one_rank": "PFF",
            "replicated": "PPF", "rank0_scatter": "PPF"}       # (comm size, partition, shared DOFs)
    for mode, v in want.items():
        q = rank_proof_quantities(synthetic_runs(meshes, n_ranks, mode), n_ranks)
        verdicts = {k: {"P": "PASS", "F": "FAIL"}[c] for k, c in zip(RANK_PROOF, v)}
        assert judged_rank_checks(case, q) == verdicts, (cid, mode, q)
        if mode == "distributed":
            assert q["shared_dofs_min_runs"] >= 1
    # a runner that omits the rank proof altogether fails too (the : quantity missing)
    assert judged_rank_checks(case, {}) == {k: "FAIL" for k in RANK_PROOF}
