"""Tuning sweeps with mode tracking (axionfem.cavity_track; part of the beta cavity module).

- synthetic analytic two-level avoided crossing (gap 2 kappa, character exchange, diabatic resolution), a true
  crossing, adaptive refinement of an avoided crossing narrower than the step, degenerate pairs, sign flips;
- 2D axisymmetric pillbox length sweep: the TM020/TM012 true crossing at the closed-form L* = 2 pi a /
  sqrt(chi02^2 - chi01^2), and an avoided crossing once a dielectric disk breaks separability;
- cross-section (xsec) rectangle a x b swept in b: TM21/TM12 and TM31/TM13 cross exactly at b = a; a small
  dielectric post opens gaps that match first-order degenerate perturbation theory;
- 3D (AXF_SLOW): a rod moved across a closed cylinder, probes through fullwave.field_probes, continuous branch.
"""
import json
import math
import os

import numpy as np
import pytest

import axionfem as af
from axionfem import cavity as cv
from axionfem import cavity_track as ct

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D cavity eigenmodes (slow); set AXF_SLOW=1")
needs_strumpack = pytest.mark.skipif(not af.build_info().get("strumpack"),
                                     reason="the 3D shift-invert eigensolver needs a STRUMPACK build")
needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build; the wheels have none)")
CHI01, CHI02 = 2.404825557695773, 5.520078110286311
C0 = 299792458.0


# ------------------------------------------------------------------ synthetic (no solver)
def test_two_level_avoided_crossing_exact():
    kappa, slope = 2e6, 1e8
    xs = np.linspace(-1, 1, 81) + 0.003
    T = ct.track(ct.two_level_steps(xs, kappa=kappa, slope=slope, seed=1))
    assert len(T.branches) == 2 and all(len(b) == len(xs) for b in T.branches)
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1 and not [e for e in T.crossings if e["kind"] == "crossing"]
    e = av[0]
    assert e["gap_min_Hz"] == pytest.approx(2 * kappa, rel=1e-3)
    assert e["kappa_Hz"] == pytest.approx(kappa, rel=1e-3)
    assert abs(e["x0"]) < 1e-3 and e["complete"] and e["exchange"] > 0.9 > 0.1 > e["stay"]
    # adiabatic branches: lower stays lower (no frequency jump), C exchanges between them
    lo = T.branch("lowest")
    f = lo.array("f_Hz")
    assert np.all(np.diff(np.abs(f - 5e9)) * np.sign(xs[1:] + xs[:-1]) >= -1)  # continuous
    C = lo.array("C")
    assert C[0] > 0.59 and C[-1] < 0.01
    # diabatic (resolved) branches keep the character: C stays high through the crossing (mixed only within a
    # few kappa/slope of x0, where every eigenmode is a mixture)
    D = T.diabatic()
    Cs = sorted(D, key=lambda b: -b.array("C")[0])[0].array("C")
    assert np.all(Cs[np.abs(xs) > 0.1] > 0.59) and np.all(Cs >= 0.3 - 1e-9)
    # exact two-level eigenvalues along the adiabatic branch
    np.testing.assert_allclose(f, 5e9 - np.sqrt((slope * xs) ** 2 + kappa ** 2), rtol=0, atol=1e-3)
    rows = T.branch_table([lo])
    assert len(rows) == len(xs) and {"branch", "value", "f_Hz", "C", "Q0", "overlap"} <= set(rows[0])


def test_true_crossing_and_extra_mode():
    xs = np.linspace(-1, 1, 41) + 0.0071
    T = ct.track(ct.two_level_steps(xs, kappa=2e6, extra=[(5.0e9 + 3e7, -0.5e8)], seed=4))
    kinds = sorted(e["kind"] for e in T.crossings)
    assert kinds == ["avoided", "crossing", "crossing"]
    for e in T.crossings:
        if e["kind"] == "crossing":
            assert min(e["overlaps"]) > 0.999 and not e["unresolved"]
    # the extra (uncoupled) mode keeps C = 0 along one branch through both crossings
    ex = T.branch(lambda r: -abs(r["Q0"] - 2e4))
    assert len(ex) == len(xs) and np.all(ex.array("C") == 0)


@pytest.mark.parametrize("kappa,kind", [(1e5, "avoided"), (0.0, "crossing")])
def test_adaptive_refinement(kappa, kind):
    """An avoided crossing 10^3 x narrower than the step looks like a crossing; bisection resolves it."""
    step = lambda x: ct.two_level_steps([x], kappa=kappa, slope=1e8, seed=3)[0]  # noqa: E731
    xs = np.linspace(-1, 1, 9) + 0.0123
    T0 = ct.track([step(x) for x in xs])
    assert [e["kind"] for e in T0.crossings] == ["crossing"]
    T = ct.adaptive_track(step, xs, refine=20, min_step=1e-5, track_kw={"degenerate_rtol": 1e-12})
    ev = [e for e in T.crossings]
    assert [e["kind"] for e in ev] == [kind]
    if kind == "avoided":
        assert ev[0]["gap_min_Hz"] == pytest.approx(2 * kappa, rel=0.02)
        assert abs(ev[0]["x0"]) < 2e-4
    else:
        assert ev[0]["resolution"] < 2e-5 and abs(ev[0]["x0"]) < 2e-5
        assert min(ev[0]["overlaps"]) > 0.999


def test_degenerate_pair_rotating_basis():
    """A degenerate pair whose basis is rotated at random at every step (as a remeshed solver returns it) is
    followed member by member through the Procrustes alignment, without ambiguous links."""
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((50, 3)))
    steps = []
    for k, x in enumerate(np.linspace(0, 1, 12)):
        t = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pair = R @ np.stack([Q[:, 0], Q[:, 1]])
        modes = [{"f_Hz": 1e9, "C": 0.0}, {"f_Hz": 1e9, "C": 0.0}, {"f_Hz": 1.2e9 + 1e7 * x, "C": 0.5}]
        steps.append(ct.ModeStep(float(x), modes, np.vstack([pair, Q[:, 2]])))
    T = ct.track(steps)
    assert len(T.branches) == 3 and all(len(b) == 12 for b in T.branches)
    assert not any(iv["ambiguous"] for iv in T.intervals)
    assert all(p["degenerate"] for p in T.branches[0].points)


def test_branch_ends_when_mode_leaves_window_and_save_load(tmp_path):
    steps = ct.two_level_steps(np.linspace(-1, 1, 11), kappa=5e6, seed=2)
    # drop the upper mode at the last three steps (a shift-invert window that moves)
    for s in steps[-3:]:
        s.modes, s.vectors = s.modes[:1], s.vectors[:1]
    s0 = steps[4]
    s0.save(tmp_path / "s")
    s1 = ct.ModeStep.load(tmp_path / "s")
    assert s1.value == s0.value and np.array_equal(s1.vectors, s0.vectors) and s1.modes == s0.modes
    T = ct.track(steps)
    assert sorted(len(b) for b in T.branches) == [8, 11]
    d = T.to_dict()
    assert {"branches", "crossings", "intervals", "warnings", "steps"} <= set(d)


def test_probe_points_3d_cylinder():
    d = cv.cylinder_design_3d(0.1, 0.2)
    P = np.array(ct.probe_points_3d(d, (6, 6, 5)))
    assert P.shape[1] == 3 and 0 < len(P) < 180
    assert np.all(np.hypot(P[:, 0], P[:, 1]) < 0.1) and np.all(np.abs(P[:, 2]) < 0.1)
    assert not np.any(np.isclose(P[:, 0], 0.0))  # jittered off the mirror planes


def test_steps_validation():
    with pytest.raises(ValueError):
        ct.ModeStep(0.0, [{"f_Hz": 1.0}], np.zeros((2, 3)))
    a = ct.ModeStep(0.0, [{"f_Hz": 1.0}], np.ones((1, 3)), valid=[True, False, False])
    b = ct.ModeStep(1.0, [{"f_Hz": 1.0}], np.ones((1, 3)), valid=[False, True, True])
    with pytest.raises(ValueError, match="no probe"):
        ct.track([a, b])


# ------------------------------------------------------------------ adversarial synthetic cases
def _levels(xs, H_of_x, C_vec, seed=0, n=60):
    """ModeSteps of a k-level model: H(x) symmetric (Hz), fields = orthonormal random vectors mixed by the
    eigenvectors; C = |<c, eigvec>|^2 * 0.6 with c = C_vec (the 'TM010-like' character)."""
    rng = np.random.default_rng(seed)
    k = H_of_x(0.0).shape[0]
    Qb, _ = np.linalg.qr(rng.standard_normal((n, k)))
    out = []
    for x in xs:
        w, U = np.linalg.eigh(H_of_x(x))
        modes = [{"f_Hz": float(w[j]), "C": float(0.6 * (np.asarray(C_vec) @ U[:, j]) ** 2), "Q0": 1e4} for j in range(k)]
        V = np.array([rng.choice([-1.0, 1.0]) * (Qb @ U[:, j]) for j in range(k)])
        out.append(ct.ModeStep(x, modes, V))
    return out


def _high_c_diabatic(T):
    D = T.diabatic()
    return max(D, key=lambda b: b.array("C")[0])


@pytest.mark.parametrize("n", [6, 10, 20, 54])
def test_symmetric_grid_single_event(n):
    """x0 exactly midway between two samples: one event (not two that cancel in diabatic())."""
    T = ct.track(ct.two_level_steps(np.linspace(-1, 1, n), kappa=3e7, seed=n))
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1
    Cs = _high_c_diabatic(T).array("C")
    assert Cs[0] > 0.5 and Cs[-1] > 0.5


@pytest.mark.parametrize("n", [41, 201, 801])
def test_dense_sampling_keeps_avoided_crossing(n):
    """The window is bounded by the gap ratio, not by a step count: the result does not depend on the density."""
    T = ct.track(ct.two_level_steps(np.linspace(-1, 1, n), kappa=2e7, slope=1e8, seed=5))
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1 and av[0]["gap_min_Hz"] == pytest.approx(4e7, rel=1e-2) and av[0]["complete"]


def test_range_narrower_than_crossing_warns():
    """Sweep range +-0.1 inside a crossing of width kappa/s = 0.2: exchange incomplete -> no event, a warning."""
    T = ct.track(ct.two_level_steps(np.linspace(-0.1, 0.1, 41), kappa=2e7, slope=1e8, seed=5))
    assert not [e for e in T.crossings if e["kind"] == "avoided"]
    assert any("wider than the range" in w for w in T.warnings)


@pytest.mark.parametrize("spec", [(5e9 + 1e5, 0.0), (5e9 + 1e5, 3e7)])
def test_spectator_mode_in_gap(spec):
    T = ct.track(ct.two_level_steps(np.linspace(-1, 1, 41) + 0.004, kappa=1e7, extra=[spec], seed=9))
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1 and av[0]["gap_min_Hz"] == pytest.approx(2e7, rel=2e-2)
    Cs = _high_c_diabatic(T).array("C")
    assert Cs[0] > 0.59 and Cs[-1] > 0.59


def test_three_level_crossing():
    """Three levels with slopes +1, 0, -1 (x 1e8 Hz) and pairwise couplings: the character of the lowest branch
    ends on the highest; the tracker reports a multi-branch avoided crossing and diabatic() recovers it."""
    g12, g13, g23 = 1.0e6, 1.3e6, 0.7e6
    H = lambda x: np.array([[5e9 + 1e8 * x, g12, g13], [g12, 5e9, g23], [g13, g23, 5e9 - 1e8 * x]])  # noqa: E731
    xs = np.linspace(-0.1, 0.1, 61) + 1e-4
    T = ct.track(_levels(xs, H, [1.0, 0.0, 0.0]))
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert av and len(av[0]["branches"]) >= 2
    Cs = _high_c_diabatic(T).array("C")
    assert Cs[0] > 0.55 and Cs[-1] > 0.55


def test_numpy_values_sorted_and_refined():
    step = lambda x: ct.two_level_steps([float(x)], kappa=0.0, slope=1e8, seed=3)[0]  # noqa: E731
    T = ct.adaptive_track(lambda x: ct.ModeStep(x, step(x).modes, step(x).vectors), np.array([6, -10, 2, -6, -2, 10]),
                          refine=6)
    assert [float(s.value) for s in T.steps] == sorted(float(s.value) for s in T.steps)
    cr = [e for e in T.crossings if e["kind"] == "crossing"]
    assert len(cr) == 1 and abs(cr[0]["x0"]) < 0.5 and T.options["refine_rounds"] > 0
    assert T.branch(np.int64(0)).id == 0


def test_rejected_link_is_ambiguous():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((30, 3)))
    a, c = Q[:, 0], Q[:, 2]
    s0 = ct.ModeStep(0.0, [{"f_Hz": 1e9}], a[None])
    s1 = ct.ModeStep(1.0, [{"f_Hz": 1e9}], (0.45 * a + 0.89 * c)[None])
    T = ct.track([s0, s1])
    iv = T.intervals[0]
    assert iv["ambiguous"] and iv["min_overlap"] == pytest.approx(0.45, abs=0.01) and iv["rejected"] == 1
    assert T.needs_refinement() == [(0.0, 1.0)]


def test_true_crossing_survives_refinement_default_rtol():
    step = lambda x: ct.two_level_steps([x], kappa=0.0, slope=1e8, seed=3)[0]  # noqa: E731
    T = ct.adaptive_track(step, np.linspace(-1, 1, 9) + 0.0123, refine=20, min_step=1e-5)
    cr = [e for e in T.crossings if e["kind"] == "crossing"]
    assert len(cr) == 1 and abs(cr[0]["x0"]) < 1e-4 and min(cr[0]["overlaps"]) > 0.999


def test_degenerate_start_no_spurious_crossing():
    """A pair degenerate at x = 0 (round-off order) that splits linearly afterwards: no crossing."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((40, 2)))
    for trial in range(20):
        steps = []
        for x in np.linspace(0, 1, 11):
            f1, f2 = 5e9 + 1e5 * x + rng.normal(0, 100), 5e9 - 1e5 * x + rng.normal(0, 100)
            if x == 0:
                f1 = f2 = 5e9 + rng.normal(0, 100)
            steps.append(ct.ModeStep(float(x), [{"f_Hz": f1}, {"f_Hz": f2}], Q.T.copy()))
        T = ct.track(steps)
        assert not [e for e in T.crossings if e["kind"] == "crossing"], trial


def test_degenerate_pair_unequal_probe_norms():
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((50, 3)))
    steps = []
    for x in np.linspace(0, 1, 12):
        t = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pair = np.diag([1.0, 10.0]) @ (R @ np.stack([Q[:, 0], Q[:, 1]]))
        modes = [{"f_Hz": 1e9}, {"f_Hz": 1e9}, {"f_Hz": 1.2e9 + 1e7 * x}]
        steps.append(ct.ModeStep(float(x), modes, np.vstack([pair, 3 * Q[:, 2]])))
    T = ct.track(steps)
    assert not any(iv["ambiguous"] for iv in T.intervals)
    assert sorted(len(b) for b in T.branches) == [12, 12, 12]


def test_edge_minimum_warns():
    T = ct.track(ct.two_level_steps(np.linspace(0, 1, 21), kappa=1e7, seed=2))
    assert any("sweep edge" in w for w in T.warnings)


# ------------------------------------------------------------------ ground-truth k-level models
def _kmodel(xs, H_of_x, *, n=80, noise=0.0, seed=0, window=None, extra_rot=None):
    """ModeSteps of a k-level model with exact eigenvectors (random orthonormal 'fields', random signs, optional
    Gaussian noise relative to the unit field norm); ``window(x) -> (fmin, fmax)`` keeps only modes inside;
    returns (steps, U list) where U[k] are the eigenvectors (columns) in the diabatic basis at x_k."""
    rng = np.random.default_rng(seed)
    k = H_of_x(xs[0]).shape[0]
    Qb, _ = np.linalg.qr(rng.standard_normal((n, k)))
    steps, Us = [], []
    for x in xs:
        w, U = np.linalg.eigh(H_of_x(x))
        keep = list(range(k)) if window is None else [j for j in range(k) if window(x)[0] <= w[j] <= window(x)[1]]
        V = []
        for j in keep:
            v = Qb @ U[:, j]
            if noise:
                v = v + noise * rng.standard_normal(n) / math.sqrt(n)
            V.append(rng.choice([-1.0, 1.0]) * v)
        modes = [{"f_Hz": float(w[j]), "C": float(0.6 * U[0, j] ** 2), "Q0": 1e4} for j in keep]
        steps.append(ct.ModeStep(float(x), modes, np.array(V) if V else np.zeros((0, n))))
        Us.append(U)
    return steps, Us


def _true_perm(Us):
    """Adiabatic index i at the first step -> index at the last step carrying its character (exact overlaps)."""
    M = (Us[0].T @ Us[-1]) ** 2
    return {i: int(np.argmax(M[i])) for i in range(M.shape[0])}


def _diabatic_perm(T):
    """Start mode index -> end mode index along the diabatic branches (all modes tracked over the whole range)."""
    out = {}
    last = len(T.steps) - 1
    for b in T.diabatic():
        p0 = [p for p in b.points if p["step"] == 0]
        p1 = [p for p in b.points if p["step"] == last]
        if p0 and p1:
            out[p0[0]["mode"]] = p1[0]["mode"]
    return out


def _three(g12, g13, g23, s=1e8):
    return lambda x: np.array([[5e9 + s * x, g12, g13], [g12, 5e9, g23], [g13, g23, 5e9 - s * x]])


@pytest.mark.parametrize("g", [(1.0e6, 1.3e6, 0.7e6), (1e6, 0.0, 1e6), (1e6, 1e5, 1e6)])
@pytest.mark.parametrize("n", [61, 80, 121])
def test_three_level_full_permutation(g, n):
    xs = np.linspace(-0.1, 0.1, n) + 1e-4
    steps, Us = _kmodel(xs, _three(*g), seed=n)
    T = ct.track(steps)
    assert _diabatic_perm(T) == _true_perm(Us), T.summary()


@pytest.mark.parametrize("split", [2e7, 4e7])
@pytest.mark.parametrize("n", [61, 81, 121])
def test_mode_crossing_a_doublet(split, n):
    """A steep mode crossing a split doublet, coupled to both members (kappa = 2e7 Hz): a 3-cycle."""
    kap = 2e7
    H = lambda x: np.array([[5e9 + 1e9 * x, kap, kap], [kap, 5e9 + split / 2, 0.0], [kap, 0.0, 5e9 - split / 2]])  # noqa: E731
    xs = np.linspace(-0.5, 0.5, n) + 1e-4
    steps, Us = _kmodel(xs, H, seed=n)
    T = ct.track(steps)
    assert _diabatic_perm(T) == _true_perm(Us), T.summary()


@pytest.mark.parametrize("n,noise", [(41, 0.05), (201, 0.05), (801, 0.05), (41, 0.2)])
def test_noise_does_not_erase_avoided_crossing(n, noise):
    found = 0
    for seed in range(5):
        H = lambda x: np.array([[5e9 + 1e8 * x, 2e7], [2e7, 5e9 - 1e8 * x]])  # noqa: E731
        steps, Us = _kmodel(np.linspace(-1, 1, n) + 1e-3, H, noise=noise, seed=seed)
        T = ct.track(steps)
        found += sum(e["kind"] == "avoided" for e in T.crossings) == 1 and _diabatic_perm(T) == _true_perm(Us)
    assert found == 5


def test_mesh_split_doublet_no_events():
    """A doublet split by 1e-5..1e-4 (random per step) whose basis rotates at random (a remeshed solver): no
    crossings, no avoided crossings, no ambiguous intervals (flagged as doublet mixing instead)."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((60, 3)))
    steps = []
    for x in np.linspace(0, 1, 41):
        t = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pair = R @ np.stack([Q[:, 0], Q[:, 1]])
        sp = 5e9 * 10 ** rng.uniform(-5, -4)
        modes = [{"f_Hz": 5e9 - sp / 2}, {"f_Hz": 5e9 + sp / 2}, {"f_Hz": 5.3e9 + 1e7 * x}]
        steps.append(ct.ModeStep(float(x), modes, np.vstack([pair, Q[:, 2]])))
    T = ct.track(steps, degenerate_rtol=1e-5)
    assert not T.crossings, T.summary()
    assert not any(iv["ambiguous"] for iv in T.intervals) and any(iv["doublet_mixing"] for iv in T.intervals)
    assert T.needs_refinement() == []


def test_window_end_and_start_in_one_interval_not_ambiguous():
    """Uncoupled modes with a moving window: a mode leaving and another entering in the same interval are an end
    and a start, not an ambiguous link."""
    fs = [5e9 + 3e7 * j for j in range(12)]
    H = lambda x: np.diag([f + 2e6 * j * x for j, f in enumerate(fs)])  # noqa: E731
    win = lambda x: (5.0e9 + 1.5e8 * x, 5.2e9 + 1.5e8 * x)  # noqa: E731  (window moves one spacing per 4 steps)
    steps, _ = _kmodel(np.linspace(0, 1, 21), H, window=win, seed=1)
    # force an end and a start in one interval: drop the lowest mode of step 10, add one above at step 10
    T = ct.track(steps)
    assert not any(iv["ambiguous"] for iv in T.intervals), [iv for iv in T.intervals if iv["ambiguous"]]
    assert sum(iv["ended"] for iv in T.intervals) >= 3 and sum(iv["started"] for iv in T.intervals) >= 3
    s9, s10 = steps[9], steps[10]
    both = [ct.ModeStep(s9.value, s9.modes, s9.vectors),
            ct.ModeStep(s10.value, s10.modes[1:] + [{"f_Hz": 5.4e9}], np.vstack([s10.vectors[1:], np.eye(80)[0]]))]
    T2 = ct.track(both)
    assert T2.intervals[0]["ended"] >= 1 and T2.intervals[0]["started"] >= 1 and not T2.intervals[0]["ambiguous"]


def test_crossing_on_a_sample_is_refined():
    step = lambda x: ct.two_level_steps([x], kappa=0.0, slope=1e8, seed=3)[0]  # noqa: E731
    T = ct.adaptive_track(step, np.linspace(-1, 1, 11), refine=10, min_step=1e-4)
    cr = [e for e in T.crossings if e["kind"] == "crossing"]
    assert len(cr) == 1 and cr[0]["resolution"] < 0.1 and T.options["refine_rounds"] > 0


@pytest.mark.parametrize("kappa", [3e4, 1e5, 2e5])
def test_narrow_resolved_avoided_crossing_kept(kappa):
    """kappa 30-200 kHz at 5 GHz (2 kappa / f < 1e-4), resolved by ~40 samples, with the 3D tolerance 1e-5."""
    w = kappa / 1e8
    H = lambda x: np.array([[5e9 + 1e8 * x, kappa], [kappa, 5e9 - 1e8 * x]])  # noqa: E731
    steps, Us = _kmodel(np.linspace(-20 * w, 20 * w, 41) + 1e-3 * w, H, seed=2)
    T = ct.track(steps, degenerate_rtol=1e-5)
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1 and av[0]["gap_min_Hz"] == pytest.approx(2 * kappa, rel=0.05)
    assert _diabatic_perm(T) == _true_perm(Us)


def test_to_dict_is_strict_json():
    T = ct.track(ct.two_level_steps(np.linspace(-1, 1, 21) + 0.01, kappa=2e7, seed=1))
    json.dumps(T.to_dict())


# ------------------------------------------------------------------ diabatic character, doublets, probe vectors
@pytest.mark.parametrize("ratio", [4.0, 6.0, 9.0, 11.0, 14.0])
def test_diabatic_character_between_pair_minima(ratio):
    """A steep level coupled (kappa) to two flat levels split by Delta = ratio kappa: the diabatic high-C line must
    carry the steep level's character at every point, including between the two pair minima (exact eigenvectors)."""
    kap, sl = 1e6, 1e8
    d = ratio * kap
    H = lambda x: np.array([[5e9 + sl * x, kap, kap], [kap, 5e9 + d / 2, 0.0], [kap, 0.0, 5e9 - d / 2]])  # noqa: E731
    xs = np.linspace(-0.6, 0.6, 241) + 1.3e-4
    steps, Us = _kmodel(xs, H, seed=3)
    T = ct.track(steps)
    hb = T.branch("max_C", diabatic=True)
    C = hb.array("C")
    exact = np.array([0.6 * np.max(U[0, :] ** 2) for U in Us])
    assert np.all(np.abs(C - exact) <= 0.2), np.max(np.abs(C - exact))


def test_unrelated_doublet_does_not_hide_narrow_crossing():
    """A resolved narrow avoided crossing (2 kappa = 0.2 MHz) next to an unrelated mesh-split doublet at 5.3 GHz."""
    kap = 1e5
    w = kap / 1e8
    rng = np.random.default_rng(4)
    Qb, _ = np.linalg.qr(rng.standard_normal((80, 4)))
    steps = []
    for x in np.linspace(-20 * w, 20 * w, 41) + 1e-3 * w:
        wv, U = np.linalg.eigh(np.array([[5e9 + 1e8 * x, kap], [kap, 5e9 - 1e8 * x]]))
        modes = [{"f_Hz": float(wv[j]), "C": float(0.6 * U[0, j] ** 2)} for j in range(2)]
        V = [rng.choice([-1, 1]) * (Qb[:, :2] @ U[:, j]) for j in range(2)]
        t = rng.uniform(0, 2 * np.pi)
        sp = 5.3e9 * 10 ** rng.uniform(-5.5, -4.5)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pair = R @ Qb[:, 2:4].T
        modes += [{"f_Hz": 5.3e9 - sp / 2}, {"f_Hz": 5.3e9 + sp / 2}]
        steps.append(ct.ModeStep(float(x), modes, np.array(V + [pair[0], pair[1]])))
    T = ct.track(steps, degenerate_rtol=1e-5)
    assert any(iv["doublet_mixing"] for iv in T.intervals)
    av = [e for e in T.crossings if e["kind"] == "avoided"]
    assert len(av) == 1 and set(av[0]["branches"]) == {0, 1}


def test_non_orthogonal_probe_vectors_not_ambiguous():
    """Distinct modes overlap by ~0.6 on the probes (coarse lattice): well-sampled steps must not be flagged."""
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((40, 4)))
    S = np.eye(4) + 0.6 * (np.ones((4, 4)) - np.eye(4)) / 3 * 2  # a fixed non-orthogonal 'sampling' map
    B = (S @ Q.T)
    steps = []
    for x in np.linspace(0, 1, 15):
        modes = [{"f_Hz": 5e9 + 1e7 * j + 1e6 * j * x} for j in range(4)]
        steps.append(ct.ModeStep(float(x), modes, B * rng.choice([-1.0, 1.0], size=(4, 1))))
    G = np.abs((B / np.linalg.norm(B, axis=1)[:, None]) @ (B / np.linalg.norm(B, axis=1)[:, None]).T)
    assert G[0, 1] > 0.5
    T = ct.track(steps)
    assert not any(iv["ambiguous"] for iv in T.intervals) and len(T.branches) == 4


def test_to_dict_numpy_values():
    T = ct.track([ct.ModeStep(np.int64(k), [{"f_Hz": np.float64(1e9 + k)}], np.eye(3)[:1]) for k in range(3)])
    json.dumps(T.to_dict())


# ------------------------------------------------------------------ 2D axisymmetric
def _pillbox(L, pert):
    a = 0.1
    d = cv.pillbox_design(a, L, h=a / 8, modes=6, families=("TM",), target_f_Hz=2.6e9)
    if pert:
        d.doc["materials"]["diel"] = {"kind": "linear", "eps_r": 1.3}
        d.doc["regions"].insert(1, {"name": "slab", "shape": {"rect": {"r": [0.0, 0.03], "z": [-L / 2, -L / 2 + 0.008]}},
                                    "material": "diel", "mesh_size": a / 16})
    return d


def test_axisymmetric_pillbox_true_and_avoided_crossing(tmp_path):
    a = 0.1
    Ls = 2 * math.pi * a / math.sqrt(CHI02 ** 2 - CHI01 ** 2)  # TM020 = TM012
    T = cv.sweep(lambda L: _pillbox(L, False), np.linspace(0.118, 0.134, 5), workdir=tmp_path / "p0", refine=6,
                 min_step=2e-5)
    cr = [e for e in T.crossings if e["kind"] == "crossing" and abs(e["x0"] - Ls) < 2e-3]
    assert len(cr) == 1 and min(cr[0]["overlaps"]) > 0.99
    assert cr[0]["x0"] == pytest.approx(Ls, rel=2e-4)
    assert cr[0]["f_mid_Hz"] == pytest.approx(C0 * CHI02 / (2 * math.pi * a), rel=1e-3)
    assert not [e for e in T.crossings if e["kind"] == "avoided"]
    Tp = cv.sweep(lambda L: _pillbox(L, True), np.linspace(0.118, 0.134, 9), workdir=tmp_path / "p1", refine=2)
    av = [e for e in Tp.crossings if e["kind"] == "avoided" and abs(e["x0"] - Ls) < 5e-3]
    assert len(av) == 1 and av[0]["gap_min_Hz"] > 1e6 and av[0]["exchange"] > av[0]["stay"]
    assert not [e for e in Tp.crossings if e["kind"] == "crossing" and abs(e["x0"] - Ls) < 5e-3]
    assert Tp.steps[0].meta["backend"] == "axisymmetric"


# ------------------------------------------------------------------ cross-section
def _rect(b, pert, a=0.1):
    from axionfem import cavity_xsec as cx

    regs = [{"name": "cav", "shape": {"rect": {"x": [0, a], "y": [0, b]}}, "material": "vacuum"}]
    mats = {}
    if pert:
        mats["diel"] = {"kind": "linear", "eps_r": 2.0}
        regs.append({"name": "post", "shape": {"disk": {"center": [0.31 * a, 0.27 * a], "radius": 0.04 * a}},
                     "material": "diel"})
    return cx.xsec_design(regs, materials=mats, modes=6, target_f_Hz=C0 / 2 * math.hypot(2 / a, 1 / a), h=a / 14,
                          allow_lossless_walls=True, interior_point=[0.6 * a, 0.6 * a])


def _post_gap(p, q, a=0.1, x0=0.031, y0=0.027, r=0.004, eps=2.0):
    """First-order degenerate perturbation theory: gap_min = (eps - 1) |<E_p|E_q>_post| / <E|E> f0."""
    from scipy import integrate

    E = lambda m, n, x, y: np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / a)  # noqa: E731
    f = lambda rr, t: E(*p, x0 + rr * np.cos(t), y0 + rr * np.sin(t)) * E(*q, x0 + rr * np.cos(t), y0 + rr * np.sin(t)) * rr  # noqa: E731
    W = integrate.dblquad(f, 0, 2 * np.pi, 0, r)[0] / (a * a / 4)
    return (eps - 1) * abs(W) * C0 / 2 * math.hypot(*p) / a


def test_xsec_rectangle_true_and_avoided_crossings(tmp_path):
    vals = np.linspace(0.08, 0.12, 5) + 1.3e-4
    T = cv.sweep(lambda b: _rect(b, False), vals, workdir=tmp_path / "r0", refine=8, min_step=5e-5)
    cr = sorted((e for e in T.crossings if e["kind"] == "crossing"), key=lambda e: e["f_mid_Hz"])
    assert len(cr) == 2 and not [e for e in T.crossings if e["kind"] == "avoided"]
    for e, (m, n) in zip(cr, [(2, 1), (3, 1)]):
        assert e["x0"] == pytest.approx(0.1, rel=1e-4)
        assert e["f_mid_Hz"] == pytest.approx(C0 / 2 * math.hypot(m, n) / 0.1, rel=1e-3)
        assert min(e["overlaps"]) > 0.99
    assert T.steps[0].meta["backend"] == "xsec"
    Tp = cv.sweep(lambda b: _rect(b, True), vals, workdir=tmp_path / "r1", refine=6, min_step=5e-5)
    av = sorted((e for e in Tp.crossings if e["kind"] == "avoided"), key=lambda e: e["f_mid_Hz"])
    assert len(av) == 2 and not [e for e in Tp.crossings if e["kind"] == "crossing"]
    # first-order degenerate perturbation theory (our quadrature, and an independent blind derivation:
    # 37.6983 / 6.60803 MHz at b = 99.6951 / 99.7738 mm; its estimate of the neglected higher orders:
    # 5-10 % / ~10 %)
    for e, (p, q), b0 in zip(av, [((2, 1), (1, 2)), ((3, 1), (1, 3))], (0.0996951, 0.0997738)):
        assert e["gap_min_Hz"] == pytest.approx(_post_gap(p, q), rel=0.10)
        assert e["x0"] == pytest.approx(b0, abs=1.5e-4)


# ------------------------------------------------------------------ 3D
@slow
@needs_3d
@needs_strumpack
def test_3d_rod_sweep_continuous_branch(tmp_path):
    """A dielectric rod (eps_r 4) spanning a closed cylinder moves off the axis (x = 0.01 ... 0.06): one TM010-like
    branch through all steps with overlaps near 1, f rising as the rod leaves the high-E region, C continuous;
    the probe vectors come from fullwave.field_probes (3D backend)."""
    a, L = 0.1, 0.1

    def build(x):
        return cv.cylinder_design_3d(a, L, h=a / 4, modes=4, target_f_Hz=1.05e9, symmetry={"y": "pmc"},
                                     rod={"radius": 0.015, "x": x, "material": {"kind": "linear", "eps_r": 4.0}},
                                     eigensolver={"method": "shift_invert"})

    xs = [0.01, 0.025, 0.04, 0.06]
    T = cv.sweep(build, xs, workdir=tmp_path / "s3", probe_n=(7, 7, 4),
                 solve_kw={"mpi_ranks": int(os.environ.get("AXF_TEST_RANKS", "2"))})
    assert T.steps[0].meta["backend"] == "3d"
    b = T.branch("max_C")
    assert len(b) == len(xs)
    assert min(p["overlap"] for p in b.points[1:]) > 0.9
    assert np.all(np.diff(b.array("f_Hz")) > 0)
    C = b.array("C")
    assert np.all(C > 0.3) and np.all(np.abs(np.diff(C)) < 0.15)
