"""Cavity tuning sweeps with mode tracking (beta).

A tuning sweep solves a cavity at a list of parameter values (a rod angle, a gap, a length) and returns the
eigenmodes of every step. Each step is remeshed, so finite-element vectors of different steps cannot be compared
directly. The modes are compared through *probes* instead: the electric field of every mode sampled on one fixed
set of points.

- axisymmetric 2D: ``(E_r, E_phi, E_z)`` on an (r, z) grid (``Results.cavity_mode_sample``), weight eps_r r;
- cross-section (``"symmetry": "xsec"``): ``(E_x, E_y, E_z)`` on an (x, y) grid, weight eps_r;
- 3D: ``fullwave.field_probes`` (points given before the solve), read from each mode's ``probes.E``, weight 1 (the
  3D probes carry no eps_r; on a uniform lattice this is the vacuum-weighted overlap, adequate for identification).

Probes that fall outside the cavity at a step (in a conductor, e.g. a moved metal rod removed from the mesh, or in
the half removed under ``mesh.symmetry``) are invalid there; an overlap only uses the probes valid at both steps.
Probes inside a dielectric are valid (in 2D/xsec they carry its eps_r weight).

:func:`track` links the modes of consecutive steps into *branches*:

1. Overlap ``O_ij = |<e_i, e'_j>_w|`` of the probe vectors (normalized on the common valid probes).
2. Degenerate clusters (relative splitting below ``degenerate_rtol``) are orthonormalized on the probes and rotated
   inside their span to align with the previous step (orthogonal Procrustes), so a TE111-like pair keeps its
   members.
3. Assignment by the Hungarian method on ``cost = (1 - O) + freq_weight * min(1, |f_j - f_pred_i| / s)^2``, where
   ``f_pred`` extrapolates the branch linearly in the swept value ("Rayleigh-quotient continuity": an eigenvalue
   moves continuously; ``s`` is the median mode spacing). A link needs ``O >= accept``; a best match with
   ``O <= reject_floor`` is a genuine end and start (e.g. the shift-invert window moved). The interval is *ambiguous*
   (the step is too coarse to identify the mode) when a link has ``O < good``, when a row has a second overlap above
   half the best (and above 0.3), or when a best match is rejected with ``reject_floor < O < accept`` — unless the
   competitor is a near-degenerate partner (``doublet_mixing``: refinement cannot help).

The resulting branches are *adiabatic*: they follow the eigenvector continuously. At a *true crossing* (modes of
different symmetry, which do not interact) the overlaps stay near 1, the branches pass through each other and the
frequency order of the two branches changes (``Tracking.crossings``, kind ``"crossing"``). The order must be well
defined on both sides (the nearest samples outside the near-degenerate band ``doublet`` x ``degenerate_rtol`` and
outside degenerate clusters have opposite signs); order changes of a doublet (split by mesh noise, or degenerate at
one end of the range) are not crossings. At an *avoided crossing* the branches approach to a minimum gap
``2 kappa`` and exchange character. For every local minimum of the gap of a branch pair, a window extends while the
gap grows, up to ``gap_factor`` x the minimum; the pair is a candidate if a branch loses its character across the
window (squared self-overlap < 0.5) to the other one (squared cross overlap >= 0.2). Overlaps across a window are
*transported*: the product of the consecutive-step overlap matrices of the branches involved, re-orthonormalized
(polar factor) at every step so that probe noise does not make it decay with the number of steps, and meaningful
when the cavity shape (hence the probe support) changes along the sweep. Candidates sharing a branch with
overlapping windows and comparable gaps (within 10x) form a cluster, extended by the branches lying between its
frequencies at the narrowest minimum (e.g. the members of a doublet that a steep mode crosses); the cluster's branches are matched between the window
ends by the assignment with the largest transported squared overlaps. A non-identity assignment with mean >= 0.5
(``exchange``; identity: ``stay``) is an avoided crossing (kind ``"avoided"``): two branches swap or several are
permuted (``permutation``). Spectators that only pass through the gap keep their character and are mapped to
themselves. The two-level form ``gap(x)^2 = (s (x - x0))^2 + (2 kappa)^2`` is fitted to the three samples nearest the
minimum of the narrowest pair (``x0``, ``gap_min_Hz`` = 2 kappa, ``kappa_Hz``; ``n_fit`` samples within the window).
Near-degenerate doublets whose basis rotates at random between solves are flagged (``doublet_mixing`` intervals, one
warning) and produce no events. An avoided crossing narrower than the step looks like a true crossing (the tracker
jumps the gap with overlaps near 1): :func:`sweep` refines every crossing interval (``refine``) and reports an
unresolved one as such. A gap minimum at the sweep edge with a character change, or a crossing wider than the range,
is warned about (unless an identified event explains it).

:meth:`Tracking.diabatic` *resolves* the avoided crossings: it permutes the branch tails at each ``x0`` and returns
character-following branches (for example the TM010-like line through the crossing, with its C dip on the
adiabatic branches). :meth:`Tracking.branch_table` reports f, C, Q0 per branch.

:func:`sweep` drives a solver: ``build(value) -> Design`` for each value, caches each step (modes and probe vectors
in ``<workdir>/step_<hash>/``) and bisects the intervals that need it (ambiguous links, crossings, avoided crossings
with too few points) up to ``refine`` rounds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = ["ModeStep", "Branch", "Tracking", "mode_step", "track", "sweep", "adaptive_track", "probe_grid_2d", "probe_points_3d",
           "two_level_steps"]

_ROW_KEYS = ("family", "index", "f_Hz", "C", "C_full", "C_rms", "V_m3", "G_ohm", "Q_wall", "Q_dielectric", "Q0",
             "residual_true", "energy_fraction", "E_azimuthal_fraction")


# ------------------------------------------------------------------------------------------------- data
class ModeStep:
    """The modes of one sweep step: ``value`` (the swept parameter), ``modes`` (rows with at least ``f_Hz``; C, Q0,
    ... when known), ``vectors`` (n_modes x n_samples probe samples of E, any fixed layout), ``valid``
    (n_samples bool; default all) and ``weights`` (n_samples quadrature weights; default 1)."""

    def __init__(self, value: Any, modes: Sequence[dict], vectors, valid=None, weights=None, meta: dict | None = None):
        import numpy as np

        self.value = value
        self.modes = [dict(m) for m in modes]
        V = np.asarray(vectors, dtype=float)
        if V.ndim != 2 or V.shape[0] != len(self.modes):
            raise ValueError(f"vectors must be (n_modes, n_samples) = ({len(self.modes)}, n); got {V.shape}")
        self.vectors = np.nan_to_num(V)
        n = V.shape[1]
        self.valid = np.ones(n, bool) if valid is None else np.asarray(valid, dtype=bool).reshape(n)
        good_rows = np.any(np.isfinite(V), axis=1) if V.shape[0] else np.zeros(0, bool)
        if good_rows.any():  # a mode without any finite sample does not invalidate the probes of the others
            self.valid &= np.all(np.isfinite(V[good_rows]), axis=0)
        self.weights = np.ones(n) if weights is None else np.nan_to_num(np.asarray(weights, dtype=float).reshape(n))
        self.meta = dict(meta or {})

    @property
    def f(self):
        import numpy as np

        return np.array([m["f_Hz"] for m in self.modes], dtype=float)

    def save(self, directory: str | Path) -> Path:
        import numpy as np

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "track_step.npz", vectors=self.vectors, valid=self.valid, weights=self.weights)
        (d / "track_step.json").write_text(json.dumps({"value": self.value, "modes": self.modes, "meta": self.meta},
                                                      default=_jsonable, indent=1))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "ModeStep":
        import numpy as np

        d = Path(directory)
        a = np.load(d / "track_step.npz")
        j = json.loads((d / "track_step.json").read_text())
        return cls(j["value"], j["modes"], a["vectors"], a["valid"], a["weights"], j.get("meta"))

    def __repr__(self) -> str:
        return f"ModeStep(value={self.value!r}, {len(self.modes)} modes, {self.vectors.shape[1]} samples)"


def _jsonable(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


class Branch:
    """One tracked branch: ``id``, ``points`` (list of dicts: ``step`` (index into Tracking.steps), ``value``,
    ``mode`` (index into that step's modes), ``overlap`` (with the previous point; None at the start), ``f_Hz``,
    ``C``, ``Q0``, ``degenerate`` (member of a degenerate cluster: per-mode quantities depend on the basis))."""

    def __init__(self, bid: int):
        self.id = bid
        self.points: list[dict] = []

    def array(self, key: str):
        import numpy as np

        return np.array([np.nan if p.get(key) is None else p[key] for p in self.points], dtype=float)

    @property
    def values(self):
        return [p["value"] for p in self.points]

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        if not self.points:
            return f"Branch({self.id}, empty)"
        return (f"Branch({self.id}, {len(self.points)} points, value {self.points[0]['value']!r}..{self.points[-1]['value']!r}, "
                f"f {self.points[0]['f_Hz'] / 1e9:.6g}..{self.points[-1]['f_Hz'] / 1e9:.6g} GHz)")


# ------------------------------------------------------------------------------------------------- linear algebra
def _is_num(v) -> bool:
    """A real number (Python or numpy scalar; not bool, str or array)."""
    import numbers

    return isinstance(v, numbers.Real) and not isinstance(v, bool) or \
        (hasattr(v, "dtype") and getattr(v, "shape", None) == () and getattr(v.dtype, "kind", "") in "iuf")


def _all_num(steps) -> bool:
    return bool(steps) and all(_is_num(s.value) for s in steps)


def _num(v, k):
    """Numeric coordinate of a swept value (the step index for non-numeric values)."""
    return float(v) if _is_num(v) else float(k)


def _normalized(step: ModeStep, mask, X=None):
    import numpy as np

    X = step.vectors if X is None else X
    Y = X[:, mask] * np.sqrt(np.abs(step.weights[mask]))[None, :]
    n = np.linalg.norm(Y, axis=1)
    n[n == 0] = 1.0
    return Y / n[:, None]


def _lowdin(Y):
    """Symmetric (Loewdin) orthonormalization of the rows of Y: distinct eigenmodes sampled on a probe lattice are not
    exactly orthogonal (their cross overlaps reached 0.6-0.8 on a 900-point 3D lattice), so raw overlaps between
    steps would show spurious competition. Falls back to Y when the Gram matrix is singular."""
    import numpy as np

    if Y.shape[0] < 2:
        return Y
    G = Y @ Y.T
    ev, EV = np.linalg.eigh(G)
    if ev.min() <= 1e-10 * max(ev.max(), 1e-300):
        return Y
    return (EV @ np.diag(ev ** -0.5) @ EV.T) @ Y


def _clusters(f, rtol: float) -> list[list[int]]:
    import numpy as np

    if len(f) == 0:
        return []
    order = list(np.argsort(f))
    out, cur = [], [order[0]]
    for i in order[1:]:
        if abs(f[i] - f[cur[-1]]) <= rtol * max(abs(f[i]), 1e-300):
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def _assign(cost):
    """Minimum-cost assignment (rows -> columns) of a rectangular cost matrix."""
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:  # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for idx in np.argsort(cost, axis=None):
            i, j = np.unravel_index(idx, cost.shape)
            if i not in used_r and j not in used_c:
                pairs.append((int(i), int(j)))
                used_r.add(i)
                used_c.add(j)
        return pairs


# ------------------------------------------------------------------------------------------------- tracking
class Tracking:
    """Result of :func:`track`: ``steps`` (sorted by value), ``branches`` (adiabatic), ``crossings`` (events:
    kind "crossing" / "avoided"), ``intervals`` (per consecutive pair of steps: values, minimum link overlap,
    ``ambiguous``), ``warnings``, ``options``."""

    def __init__(self, steps, branches, crossings, intervals, warnings, options, aligned):
        self.steps = steps
        self.branches = branches
        self.crossings = crossings
        self.intervals = intervals
        self.warnings = warnings
        self.options = options
        self._aligned = aligned  # per step: probe vectors after the degenerate-cluster alignment

    # -------- selection
    def branch(self, which: Any = "max_C", *, at: int = 0, diabatic: bool = False) -> Branch:
        """A branch: ``which`` = a branch id, ``"max_C"`` (the branch through the largest-C mode of step ``at``),
        ``"lowest"`` (through its lowest mode), or a callable ``row -> score`` (the branch through the highest-scoring
        mode of step ``at``)."""
        import numbers

        pool = self.diabatic() if diabatic else self.branches
        if isinstance(which, numbers.Integral) and not isinstance(which, bool):
            hit = [b for b in pool if b.id == int(which)]
            if not hit:
                raise ValueError(f"no branch with id {which}")
            return hit[0]
        if not -len(self.steps) <= at < len(self.steps):
            raise ValueError(f"step {at} out of range ({len(self.steps)} steps)")
        at = at % len(self.steps)
        rows = self.steps[at].modes
        if not rows:
            raise ValueError(f"step {at} has no modes")
        if which == "max_C":
            k = max(range(len(rows)), key=lambda i: rows[i].get("C") or 0.0)
        elif which == "lowest":
            k = min(range(len(rows)), key=lambda i: rows[i]["f_Hz"])
        elif callable(which):
            k = max(range(len(rows)), key=lambda i: which(rows[i]))
        else:
            raise ValueError(f"unknown branch selector {which!r}")
        return next(b for b in pool if any(p["step"] == at and p["mode"] == k for p in b.points))

    def branch_table(self, branches: Sequence[Branch] | None = None) -> list[dict]:
        """Flat rows (one per branch point): ``branch``, ``value``, ``f_Hz``, ``C``, ``Q0``, ``overlap``,
        ``degenerate``, ``ambiguous`` (the link to this point came through an ambiguous interval), ``step``,
        ``mode``."""
        amb = {iv["step"] + 1 for iv in self.intervals if iv["ambiguous"]}
        out = []
        for b in branches if branches is not None else self.branches:
            for p in b.points:
                out.append({"branch": b.id, "value": p["value"], "f_Hz": p["f_Hz"], "C": p.get("C"), "Q0": p.get("Q0"),
                            "overlap": p.get("overlap"), "degenerate": p.get("degenerate", False),
                            "ambiguous": p.get("overlap") is not None and p["step"] in amb, "step": p["step"],
                            "mode": p["mode"]})
        return out

    # -------- resolution of avoided crossings
    def diabatic(self, events: Sequence[dict] | None = None) -> list[Branch]:
        """Character-following (diabatic) branches: copies of the adiabatic branches whose tails are permuted at each
        avoided crossing (``events``: default all of kind "avoided"). A two-branch event swaps the two tails beyond
        ``x0``; a multi-branch event applies its pair-minimum ``swaps`` in x order when they compose to its
        ``permutation`` (so the character is right between the minima too), else the permutation at ``x0``. Within a
        few kappa/slope of a crossing every eigenmode is a mixture, so the diabatic quantities there are those of
        mixed modes."""
        evs = [e for e in (self.crossings if events is None else events) if e["kind"] == "avoided"]
        ops, seen = [], set()
        for e in evs:
            key = (tuple(sorted(e["branches"])), round(float(e["x0"]), 12))
            if key in seen:
                continue
            seen.add(key)
            if e.get("swaps"):
                ops += [(float(w["x0"]), {int(w["branches"][0]): int(w["branches"][1]),
                                          int(w["branches"][1]): int(w["branches"][0])}) for w in e["swaps"]]
            else:
                ops.append((float(e["x0"]), {int(i): int(j) for i, j in e["permutation"].items()}))
        ops.sort(key=lambda o: o[0])
        cur = {b.id: [dict(p) for p in b.points] for b in self.branches}
        owner = {b.id: b.id for b in self.branches}  # adiabatic branch -> diabatic branch holding its future tail
        for x0, perm in ops:
            if not all(i in owner and j in owner for i, j in perm.items()):
                continue
            tails = {i: [p for p in cur[owner[i]] if p["x"] > x0] for i in perm}
            new_owner = dict(owner)
            for i, j in perm.items():  # the character on adiabatic branch i continues on branch j's tail
                d = owner[i]
                cur[d] = [p for p in cur[d] if p["x"] <= x0] + tails[j]
                new_owner[j] = d
            owner = new_owner
        out = []
        for bid, pts in cur.items():
            b = Branch(bid)
            b.points = sorted(pts, key=lambda p: p["x"])
            out.append(b)
        return out

    def needs_refinement(self, *, crossings: bool = True, min_points: int = 3) -> list[tuple]:
        """Intervals (lo, hi) of swept values to bisect: ambiguous links, crossings (``crossings``: a true crossing
        and an avoided crossing narrower than the step look alike) and avoided crossings with fewer than
        ``min_points`` steps inside the fit window. Non-numeric sweeps return []."""
        if not _all_num(self.steps):
            return []
        out = set()
        for iv in self.intervals:
            if iv["ambiguous"]:
                out.add((float(iv["lo"]), float(iv["hi"])))
        for e in self.crossings:
            if e["kind"] == "crossing" and crossings:
                for lo, hi in e.get("refine") or [(e["lo"], e["hi"])]:
                    out.add((float(lo), float(hi)))
            if e["kind"] == "avoided" and e.get("n_fit", min_points) < min_points:
                for lo, hi in e.get("refine", []):
                    out.add((float(lo), float(hi)))
        return sorted(out)

    # -------- output
    def summary(self) -> str:
        lines = [f"Tracking: {len(self.steps)} steps, {len(self.branches)} branches, "
                 f"{sum(e['kind'] == 'crossing' for e in self.crossings)} crossings, "
                 f"{sum(e['kind'] == 'avoided' for e in self.crossings)} avoided crossings, "
                 f"{sum(iv['ambiguous'] for iv in self.intervals)} ambiguous intervals"]
        for e in self.crossings:
            if e["kind"] == "avoided":
                lines.append(f"  avoided  branches {e['branches']}: x0 = {e['x0']:.6g}, gap_min = {e['gap_min_Hz'] / 1e6:.4g} MHz, "
                             f"exchange {e['exchange']:.2f} (stay {e['stay']:.2f}), {e['n_fit']} fit points"
                             f"{', window truncated' if e.get('truncated') else ''}")
            else:
                lines.append(f"  crossing branches {e['branches']}: x = {e['x0']:.6g} (overlaps {e['overlaps'][0]:.3f}, "
                             f"{e['overlaps'][1]:.3f}){' UNRESOLVED' if e.get('unresolved') else ''}")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-ready record: branches (rows), crossings, intervals, warnings, options."""
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return _jsonable(o) if not isinstance(o, (str, int, float, bool, type(None))) else o

        return clean({"branches": self.branch_table(), "crossings": self.crossings, "intervals": self.intervals,
                "warnings": self.warnings, "options": self.options,
                "steps": [{"value": s.value, "modes": [{k: m.get(k) for k in ("family", "f_Hz", "C", "Q0") if k in m}
                                                       for m in s.modes]} for s in self.steps]})

    def plot(self, path: str | Path | None = None, *, ax=None, quantity: str = "f_Hz", scale: float = 1e-9,
             color_by: str = "C", branches: Sequence[Branch] | None = None, label: str = "f [GHz]",
             xlabel: str = "swept value", highlight: Sequence[int] = (), vmax: float | None = None):
        """Mode map: every mode as a dot coloured by ``color_by`` (C), the branches as lines (``highlight``: branch
        ids drawn thick), avoided crossings as open circles and crossings as crosses."""
        import matplotlib.pyplot as plt

        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
        xs, ys, cs = [], [], []
        numeric = _all_num(self.steps)
        for k, s in enumerate(self.steps):
            for m in s.modes:
                if m.get(quantity) is None:
                    continue
                xs.append(float(s.value) if numeric else float(k)); ys.append(m[quantity] * scale)
                cs.append(m.get(color_by) or 0.0)
        sc = ax.scatter(xs, ys, c=cs, s=10, cmap="viridis", vmin=0, vmax=vmax or max(max(cs, default=1.0), 1e-12),
                        zorder=3)
        for b in branches if branches is not None else self.branches:
            x = [p["x"] for p in b.points]
            y = [(p.get(quantity) or math.nan) * scale for p in b.points]
            ax.plot(x, y, "-", lw=2.2 if b.id in highlight else 0.8, color="C3" if b.id in highlight else "0.55",
                    zorder=2)
        for e in self.crossings:
            if e["kind"] == "avoided" and quantity == "f_Hz":
                ax.plot([e["x0"]], [e["f_mid_Hz"] * scale], "o", mfc="none", mec="C1", ms=9, zorder=4)
            elif quantity == "f_Hz":
                ax.plot([e["x0"]], [e["f_mid_Hz"] * scale], "x", color="C0", ms=7, zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        if own:
            plt.colorbar(sc, ax=ax, label=color_by)
            fig.tight_layout()
            if path is not None:
                fig.savefig(path, dpi=150)
                plt.close(fig)
                return Path(path)
        return ax

    def __repr__(self) -> str:
        return self.summary().splitlines()[0]


def track(steps: Sequence[ModeStep], *, accept: float = 0.5, good: float = 0.9, degenerate_rtol: float = 1e-6,
          freq_weight: float = 0.1, gap_factor: float = 3.0, reject_floor: float = 0.2,
          doublet: float = 100.0) -> Tracking:
    """Link the modes of consecutive :class:`ModeStep` s into adiabatic branches and classify the crossings
    (module docstring). Steps are sorted by value (numeric, incl. numpy scalars) or kept in order.

    ``accept``: minimum overlap of a link (below it the branch ends and a new one starts); ``good``: links below it,
    a competing second overlap (above half the best and above 0.3), or a rejected best match with overlap above
    ``reject_floor`` mark the interval ambiguous; ``degenerate_rtol``: relative splitting below which modes form a
    degenerate cluster (their frequency order is noise: no crossing is read from it); ``freq_weight``: weight of the
    frequency-continuity term in the assignment cost; ``gap_factor``: an avoided-crossing window extends from the
    gap minimum until the gap exceeds ``gap_factor`` x the minimum (or stops growing, or the common range ends:
    ``truncated``); ``doublet``: pairs split by less than ``doublet`` x ``degenerate_rtol`` on both sides of a
    frequency-order change or a gap minimum are treated as a near-degenerate doublet (no crossing, no avoided crossing,
    links flagged ``doublet_mixing`` instead of ambiguous)."""
    import numpy as np

    steps = list(steps)
    numeric = _all_num(steps)
    if numeric:
        steps.sort(key=lambda s: float(s.value))
    xs = [float(s.value) if numeric else float(k) for k, s in enumerate(steps)]
    warnings: list[str] = []
    aligned = [None] * len(steps)
    if not steps:
        return Tracking([], [], [], [], ["no steps"], {}, [])
    aligned[0] = steps[0].vectors.copy()
    branches: list[Branch] = []
    owner: dict[tuple, Branch] = {}  # (step, mode) -> branch
    degid = [dict() for _ in steps]  # step -> {mode: cluster id} for members of degenerate clusters
    for k, s in enumerate(steps):
        for c, cl in enumerate(_clusters(s.f, degenerate_rtol)):
            if len(cl) > 1:
                for i in cl:
                    degid[k][i] = c

    def point(k, i, ov):
        m = steps[k].modes[i]
        return {"step": k, "value": steps[k].value, "x": xs[k], "mode": i, "overlap": ov, "f_Hz": m["f_Hz"],
                "C": m.get("C"), "Q0": m.get("Q0"), "degenerate": i in degid[k]}

    for i in range(len(steps[0].modes)):
        b = Branch(len(branches))
        b.points.append(point(0, i, None))
        branches.append(b)
        owner[(0, i)] = b
    intervals = []
    dbl_pairs: dict = {}
    for k in range(len(steps) - 1):
        A, B = steps[k], steps[k + 1]
        mask = A.valid & B.valid
        if not mask.any():
            raise ValueError(f"no probe is valid at both steps {A.value!r} and {B.value!r}")
        Va = aligned[k]
        Vb = B.vectors.copy()
        Ah = _lowdin(_normalized(A, mask, Va))
        Bh = _lowdin(_normalized(B, mask, Vb))
        # align each degenerate cluster of step k+1 with step k: orthonormalize the cluster on the common probes
        # (Loewdin), then the orthogonal Procrustes rotation inside its span; the same linear map acts on the raw
        # vectors kept for the next step
        w = np.sqrt(np.abs(B.weights[mask]))
        for cl in _clusters(B.f, degenerate_rtol):
            if len(cl) < 2 or Ah.shape[0] == 0:
                continue
            Y = Vb[cl][:, mask] * w[None, :]
            G = Y @ Y.T
            ev, EV = np.linalg.eigh(G)
            if ev.min() <= 1e-14 * max(ev.max(), 1e-300):
                continue  # linearly dependent on the probes: leave the cluster as it is
            L = EV @ np.diag(ev ** -0.5) @ EV.T  # G^{-1/2}
            Yo = L @ Y
            P = Ah @ Yo.T  # (n_k, d)
            d = len(cl)
            rows = list(np.argsort(-np.linalg.norm(P, axis=1))[:d])
            if len(rows) < d:
                continue
            U, _, Wt = np.linalg.svd(P[rows])
            Q = U @ Wt  # rows of Q Yo align with A[rows]
            T = Q @ L
            Vb[cl] = T @ Vb[cl]
            Bh[cl] = Q @ Yo
        aligned[k + 1] = Vb
        O = np.abs(Ah @ Bh.T) if Ah.shape[0] and Bh.shape[0] else np.zeros((Ah.shape[0], Bh.shape[0]))
        fb = B.f
        # predicted frequencies (linear extrapolation along each branch)
        fpred = np.empty(len(A.modes))
        for i in range(len(A.modes)):
            pts = owner[(k, i)].points
            if len(pts) >= 2 and pts[-1]["x"] != pts[-2]["x"]:
                p1, p2 = pts[-2], pts[-1]
                fpred[i] = p2["f_Hz"] + (p2["f_Hz"] - p1["f_Hz"]) / (p2["x"] - p1["x"]) * (xs[k + 1] - p2["x"])
            else:
                fpred[i] = A.modes[i]["f_Hz"]
        sp = np.diff(np.sort(fb))
        sp = sp[sp > 0]
        sscale = float(np.median(sp)) if sp.size else max(float(np.max(np.abs(fb))) * 1e-2, 1.0) if fb.size else 1.0
        D = np.minimum(np.abs(fb[None, :] - fpred[:, None]) / sscale, 1.0) if fb.size else np.zeros_like(O)
        cost = (1.0 - O) + freq_weight * D ** 2
        pairs = _assign(cost) if O.size else []
        linked_b = set()
        min_ov = None
        rejected = []
        for i, j in pairs:
            ov = float(O[i, j])
            if ov < accept:
                if ov > reject_floor:  # a plausible match turned down: the step is too coarse
                    rejected.append((i, j, ov))
                    min_ov = ov if min_ov is None else min(min_ov, ov)
                continue  # (O <= reject_floor: a genuine end and start, e.g. the window moved)
            min_ov = ov if min_ov is None else min(min_ov, ov)
            bb = owner[(k, i)]
            bb.points.append(point(k + 1, j, ov))
            owner[(k + 1, j)] = bb
            linked_b.add(j)
        ended = sum(1 for i, j in pairs if float(O[i, j]) < accept) + len(A.modes) - len(pairs)
        for j in range(len(B.modes)):
            if j not in linked_b:
                nb = Branch(len(branches))
                nb.points.append(point(k + 1, j, None))
                branches.append(nb)
                owner[(k + 1, j)] = nb
        # weak or mixed links (a second overlap above half the best). If the competitor is a near-degenerate
        # partner (a doublet split by less than `doublet` x rtol, whose basis the remeshed solver rotates at random)
        # the link is flagged as doublet mixing, which refinement cannot cure; otherwise the interval is ambiguous
        mixed, weak, dbl = [], [], []  # (dbl_pairs: interval -> {frozenset(modes at k+1)})
        for i, j in pairs:
            ov = float(O[i, j])
            if ov < accept or O.shape[1] < 1:
                continue
            o2, j2 = -1.0, None
            if O.shape[1] > 1:
                j2 = int(np.argsort(O[i])[-2]) if int(np.argmax(O[i])) == j else int(np.argmax(O[i]))
                o2 = float(O[i, j2])
            bad_mix = o2 > 0.5 * ov and o2 > 0.3
            if not (ov < good or bad_mix):
                continue
            if j2 is not None and abs(fb[j] - fb[j2]) <= doublet * degenerate_rtol * max(abs(fb[j]), 1e-300):
                dbl.append((i, j, j2))
                dbl_pairs.setdefault(k, set()).add(frozenset((int(j), int(j2))))
            elif bad_mix:
                mixed.append((i, j, o2))
            else:
                weak.append((i, j, ov))
        amb = bool(weak) or bool(mixed) or bool(rejected)
        intervals.append({"step": k, "lo": A.value, "hi": B.value, "min_overlap": min_ov, "ambiguous": amb,
                          "doublet_mixing": bool(dbl), "ended": ended, "started": len(B.modes) - len(linked_b),
                          "rejected": len(rejected), "valid_probes": int(mask.sum())})
        if amb:
            why = ", ".join(t for t, c in (("weak link", weak), ("mixed", mixed), ("rejected link", rejected)) if c)
            warnings.append(f"interval {A.value!r} -> {B.value!r}: ambiguous links (min overlap "
                            f"{min_ov if min_ov is not None else float('nan'):.3f}{', ' + why if why else ''}); refine the step")

    nd = sum(iv["doublet_mixing"] for iv in intervals)
    if nd:
        warnings.append(f"{nd} intervals link near-degenerate doublets (split < {doublet:g} x degenerate_rtol) whose "
                        "basis rotates between steps: their members are followed as a pair, not individually")
    crossings, cw = _classify(steps, xs, branches, aligned, degid, gap_factor=gap_factor, good=good,
                              doublet_tol=doublet * degenerate_rtol,
                              dbl_pairs=dbl_pairs)
    warnings += cw
    opts = {"accept": accept, "good": good, "degenerate_rtol": degenerate_rtol, "freq_weight": freq_weight,
            "gap_factor": gap_factor, "reject_floor": reject_floor, "doublet": doublet}
    return Tracking(steps, branches, crossings, intervals, warnings, opts, aligned)


def _transport(steps, aligned, at, bl, km, kp):
    """Squared overlaps M[i, j] = |<i(km) | j(kp)>|^2 between branches ``bl`` at steps km and kp, transported along the
    sweep: the product of the signed consecutive-step overlap matrices of these branches. Unlike a direct overlap of
    distant steps it stays meaningful when the cavity changes shape (the probe support and field shapes of distant
    steps need not match), and it measures how the character moves inside the branches' span."""
    import numpy as np

    d = len(bl)
    P = np.eye(d)
    for k in range(km, kp):
        A, B = steps[k], steps[k + 1]
        mask = A.valid & B.valid
        if not mask.any():
            return None
        Ah = _lowdin(_normalized(A, mask, aligned[k]))[[at[k][b]["mode"] for b in bl]]
        Bh = _lowdin(_normalized(B, mask, aligned[k + 1]))[[at[k + 1][b]["mode"] for b in bl]]
        P = P @ (Ah @ Bh.T)
        # re-orthonormalize (polar factor): probe noise makes every consecutive overlap slightly < 1, and a product of
        # many such matrices would decay geometrically with the number of steps in the window
        U, _, Vt = np.linalg.svd(P)
        P = U @ Vt
    return np.minimum(P ** 2, 1.0)


def _classify(steps, xs, branches, aligned, degid, *, gap_factor, good, doublet_tol, dbl_pairs=None,
              n_local: int = 3):
    """Crossings and avoided crossings between the adiabatic branches; returns (events, warnings).

    Crossing: the frequency order of two branches changes between two steps at which the pair is not inside one
    degenerate cluster (degenerate steps in between are skipped; the interval then spans them, and its two halves
    are offered for refinement). A pair that stays a near-degenerate doublet (gap < ``doublet_tol`` f) around the
    change is not reported.
    Avoided crossing: a local minimum of the gap of a branch pair (strict on the left, so a minimum straddled
    symmetrically by two samples counts once), with a window that extends while the gap grows, up to ``gap_factor``
    x the minimum. A pair is a candidate when a branch loses its character across the window (squared self-overlap
    < 0.5) to the other one (squared cross overlap >= 0.2), both measured by the transported overlap. Candidates that
    share a branch and have overlapping windows form a cluster; the cluster's branches are matched between the ends
    of the cluster's window by the assignment that maximizes the transported squared overlaps. A non-identity
    assignment with mean >= 0.5 (and above the identity's) is an avoided crossing: two branches swap
    (``exchange``/``stay``), or several are permuted (``permutation``; a multi-level crossing)."""
    import numpy as np

    events, pending_warn = [], []
    at: dict = {}
    for b in branches:
        for p in b.points:
            at.setdefault(p["step"], {})[b.id] = p
    ids = [b.id for b in branches]

    def degpair(k, a, b):
        ma, mb = at[k][a]["mode"], at[k][b]["mode"]
        return ma in degid[k] and degid[k].get(ma) == degid[k].get(mb)

    def near(gv, f):
        return gv < doublet_tol * abs(f)

    cands = []
    for ai in range(len(ids)):
        for bi in range(ai + 1, len(ids)):
            a, b = ids[ai], ids[bi]
            ks = sorted(k for k in at if a in at[k] and b in at[k])
            if len(ks) < 2:
                continue
            runs, cur = [], [ks[0]]
            for k in ks[1:]:
                if k == cur[-1] + 1:
                    cur.append(k)
                else:
                    runs.append(cur)
                    cur = [k]
            runs.append(cur)
            for run in runs:
                d = np.array([at[k][a]["f_Hz"] - at[k][b]["f_Hz"] for k in run])
                g = np.abs(d)
                fa = np.array([at[k][a]["f_Hz"] for k in run])
                dg = [degpair(k, a, b) for k in run]
                # the pair behaves as a doublet: near-degenerate links flagged as doublet mixing somewhere in the run
                dp = dbl_pairs or {}
                pair_doublet = any(frozenset((at[run[u + 1]][a]["mode"], at[run[u + 1]][b]["mode"])) in dp.get(run[u], ())
                                   and near(g[u], fa[u]) for u in range(len(run) - 1))
                # ---- crossings (sign change between consecutive non-degenerate samples)
                nd = [t for t in range(len(run)) if not dg[t]]
                for t0, t1 in zip(nd, nd[1:]):
                    if not d[t0] * d[t1] < 0:
                        continue
                    # the order must be well defined on both sides: the nearest samples outside the near-degenerate
                    # band (gap >= doublet_tol f) on the left and on the right have opposite signs; otherwise the
                    # change is the noise of a doublet (mesh-split, or degenerate at one end of the range)
                    lft = next((u for u in range(t0, -1, -1) if not near(g[u], fa[u]) and not dg[u]), None)
                    rgt = next((u for u in range(t1, len(run)) if not near(g[u], fa[u]) and not dg[u]), None)
                    if lft is None or rgt is None or not d[lft] * d[rgt] < 0:
                        continue
                    k0, k1 = run[t0], run[t1]
                    x0 = xs[k0] + (xs[k1] - xs[k0]) * g[t0] / (g[t0] + g[t1])
                    f0 = 0.25 * (at[k0][a]["f_Hz"] + at[k0][b]["f_Hz"] + at[k1][a]["f_Hz"] + at[k1][b]["f_Hz"])
                    ova = min((at[run[t]][a]["overlap"] or 0.0) for t in range(t0 + 1, t1 + 1))
                    ovb = min((at[run[t]][b]["overlap"] or 0.0) for t in range(t0 + 1, t1 + 1))
                    ev = {"kind": "crossing", "branches": [a, b], "lo": steps[k0].value, "hi": steps[k1].value,
                          "x0": float(x0), "f_mid_Hz": float(f0), "overlaps": [float(ova), float(ovb)], "step": k0,
                          "unresolved": bool(min(ova, ovb) < good), "through_degenerate": t1 - t0 > 1,
                          "note": "frequency order changes with the characters preserved: a true crossing, or an "
                                  "avoided crossing narrower than this interval"}
                    if t1 - t0 > 1:  # the crossing sits on degenerate sample(s): refine on both sides of them
                        ev["refine"] = [(steps[k0].value, steps[run[t0 + 1]].value),
                                        (steps[run[t1 - 1]].value, steps[k1].value)]
                    events.append(ev)
                # ---- avoided-crossing candidates (interior gap minima)
                for t in range(1, len(run) - 1):
                    if not (g[t] < g[t - 1] and g[t] <= g[t + 1]) or d[t - 1] * d[t + 1] < 0 or dg[t]:
                        continue
                    lo = t  # (ties allowed: a minimum straddled symmetrically has two equal samples)
                    while lo > 0 and g[lo - 1] >= g[lo] * (1 - 1e-9) and g[lo] < gap_factor * g[t]:
                        lo -= 1
                    hi = t
                    while hi < len(run) - 1 and g[hi + 1] >= g[hi] * (1 - 1e-9) and g[hi] < gap_factor * g[t]:
                        hi += 1
                    if pair_doublet and all(near(g[u], fa[u]) for u in range(lo, hi + 1)):
                        continue  # a near-degenerate doublet whose basis rotates at random between solves
                    km, kp = run[lo], run[hi]
                    Mw = _transport(steps, aligned, at, [a, b], km, kp)
                    if Mw is None:
                        continue
                    sa, sb, xab = float(Mw[0, 0]), float(Mw[1, 1]), float(0.5 * (Mw[0, 1] + Mw[1, 0]))
                    trunc = bool(g[lo] < gap_factor * g[t] or g[hi] < gap_factor * g[t])
                    if not (min(sa, sb) < 0.5 and xab >= 0.2):
                        if trunc and xab >= 0.15 and min(sa, sb) <= 0.85:
                            pending_warn.append(((a, b), (xs[km], xs[kp]),
                                                 f"branches {a}, {b}: gap minimum {g[t] / 1e6:.4g} MHz at x = "
                                                 f"{xs[run[t]]:.6g} with a partial character exchange ({xab:.2f}) inside "
                                                 "a window cut by the sweep range: an avoided crossing wider than the "
                                                 "range; extend the sweep"))
                        continue
                    sel = [u for u in range(lo, hi + 1) if g[u] <= gap_factor * g[t]] or [t]
                    sel = sorted(sorted(sel, key=lambda u: (abs(xs[run[u]] - xs[run[t]]), u))[:n_local])
                    xx = np.array([xs[run[u]] for u in sel])
                    yy = g[sel] ** 2
                    x0, gmin = xs[run[t]], float(g[t])
                    if len(sel) >= 3:
                        c2, c1, c0 = np.polyfit(xx, yy, 2)
                        if c2 > 0:
                            xv = -c1 / (2 * c2)
                            if xx.min() - (xx.max() - xx.min()) <= xv <= xx.max() + (xx.max() - xx.min()):
                                x0 = float(xv)
                                gmin = float(math.sqrt(max(c0 - c1 * c1 / (4 * c2), 0.0)))
                    kk = run[t]
                    cands.append({"pair": (a, b), "t_step": kk, "km": km, "kp": kp, "x0": float(x0), "gmin": gmin,
                                  "g_sampled": float(g[t]),
                                  "n_fit": sum(1 for u in range(lo, hi + 1) if g[u] <= gap_factor * g[t]),
                                  "truncated": trunc, "f_mid": float(0.5 * (at[kk][a]["f_Hz"] + at[kk][b]["f_Hz"])),
                                  "refine": [(steps[run[t - 1]].value, steps[kk].value),
                                             (steps[kk].value, steps[run[t + 1]].value)],
                                  "xr": (xs[km], xs[kp]), "stay2": 0.5 * (sa + sb), "exch2": xab})
                # ---- sweep-edge minima: a branch losing its character there
                if len(run) >= 3:
                    for t, nb_ in ((0, 1), (len(run) - 1, len(run) - 2)):
                        if run[t] not in (0, len(steps) - 1):
                            continue  # the pair's common range ends inside the sweep (a branch ends): not an edge
                        if not g[t] < g[nb_] or dg[t] or near(g[t], fa[t]):
                            continue
                        u = t
                        step_ = 1 if t == 0 else -1
                        while 0 <= u + step_ < len(run) and g[u + step_] > g[u] and g[u] < gap_factor * g[t]:
                            u += step_
                        k1, k2 = sorted((run[t], run[u]))
                        Mw = _transport(steps, aligned, at, [a, b], k1, k2) if k2 > k1 else None
                        if Mw is None:
                            continue
                        sa, sb, xe = Mw[0, 0], Mw[1, 1], 0.5 * (Mw[0, 1] + Mw[1, 0])
                        if min(sa, sb) < 0.8 and xe > 0.1:
                            pending_warn.append(((a, b), (xs[k1], xs[k2]),
                                                 f"branches {a}, {b}: gap minimum at the sweep edge "
                                                 f"({steps[run[t]].value!r}) with character change (self-overlap^2 "
                                                 f"{min(sa, sb):.2f}): an avoided crossing may lie at or beyond the "
                                                 "edge; extend the sweep"))
    # ---- clusters of candidates (shared branch, overlapping windows)
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            ci, cj = cands[i], cands[j]
            comparable = max(ci["gmin"], cj["gmin"]) <= 10.0 * max(min(ci["gmin"], cj["gmin"]), 1e-300)
            if set(ci["pair"]) & set(cj["pair"]) and ci["xr"][0] <= cj["xr"][1] and cj["xr"][0] <= ci["xr"][1] \
                    and comparable:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(cands[i])
    bpts = {b.id: b.points for b in branches}
    warns = []
    for grp in groups.values():
        # duplicates of one pair (overlapping windows) keep the narrowest gap
        byp: dict = {}
        for c in grp:
            key = tuple(sorted(c["pair"]))
            if key not in byp or c["gmin"] < byp[key]["gmin"]:
                byp[key] = c
        grp = list(byp.values())
        best = min(grp, key=lambda c: c["gmin"])
        kk = best["t_step"]
        bset = sorted({x for c in grp for x in c["pair"]})
        kmin, kmax = min(c["km"] for c in grp), max(c["kp"] for c in grp)
        # branches between (or within one cluster gap of) the cluster's frequencies at the narrowest minimum take
        # part too: e.g. the members of a doublet that a steep mode crosses (a 3-cycle with only one gap minimum)
        fk = [at[kk][x]["f_Hz"] for x in bset if x in at.get(kk, {})]
        extra = [bid for bid, p in at.get(kk, {}).items() if bid not in bset and fk and
                 min(fk) < p["f_Hz"] < max(fk)]
        if len(grp) == 1 and not extra:
            c = grp[0]
            a, b = c["pair"]
            bl, km, kp = [a, b], c["km"], c["kp"]
            M = np.array([[c["stay2"], c["exch2"]], [c["exch2"], c["stay2"]]])
        else:
            bset = sorted(set(bset) | set(extra))
            bl = [bid for bid in bset if bid in at.get(kk, {})]
            km = kk
            while km - 1 >= kmin and all(bid in at.get(km - 1, {}) for bid in bl):
                km -= 1
            kp = kk
            while kp + 1 <= kmax and all(bid in at.get(kp + 1, {}) for bid in bl):
                kp += 1
            if len(bl) < 2 or kp <= km:
                continue
            M = _transport(steps, aligned, at, bl, km, kp)
            if M is None:
                continue
            M = np.nan_to_num(M)
        pr = _assign(-M)
        perm = {bl[i]: bl[j] for i, j in pr}
        score = float(np.mean([M[i, j] for i, j in pr]))
        ident = float(np.mean(np.diag(M)))
        moved = {int(i): int(j) for i, j in perm.items() if i != j}
        if (not moved or not (score >= 0.5 and score > ident)) and len(grp) == 1 and len(bl) > 2:
            # a single pair whose cluster was extended by in-between branches: fall back to the pair's own test
            c = grp[0]
            a, b = c["pair"]
            bl, km, kp = [a, b], c["km"], c["kp"]
            M = np.array([[c["stay2"], c["exch2"]], [c["exch2"], c["stay2"]]])
            score, ident = float(c["exch2"]), float(c["stay2"])
            moved = {int(a): int(b), int(b): int(a)} if score >= 0.5 and score > ident else {}
        if not moved or not (score >= 0.5 and score > ident):
            lost = [bl[i] for i in range(len(bl)) if M[i, i] < 0.5]
            if lost:
                warns.append(f"branches {lost} lose their character near x = {xs[kk]:.6g} without an identified "
                             f"partner (best match {score:.2f}): a partner outside the window or the solved set, or "
                             "too coarse steps")
            continue
        modes = {}
        for bid in moved:
            p = min(bpts[bid], key=lambda q: abs(q["step"] - kk))
            if p["step"] == kk:
                modes[int(bid)] = int(p["mode"])
        mv = sorted(moved)
        ev = {"kind": "avoided", "branches": mv, "permutation": moved, "x0": best["x0"], "gap_min_Hz": best["gmin"],
              "kappa_Hz": 0.5 * best["gmin"], "gap_sampled_Hz": best["g_sampled"], "f_mid_Hz": best["f_mid"],
              "exchange": score, "stay": ident, "complete": bool(score > 0.8),
              "n_fit": int(min(c["n_fit"] for c in grp)), "truncated": bool(any(c["truncated"] for c in grp)),
              "window": [xs[km], xs[kp]], "step": kk, "modes": modes,
              "lo": best["refine"][0][0], "hi": best["refine"][1][1],
              "refine": [iv for c in grp if c["n_fit"] < 3 for iv in c["refine"]]}
        if len(grp) > 1:
            ev["pair_minima"] = [{"branches": list(c["pair"]), "x0": c["x0"], "gap_min_Hz": c["gmin"]} for c in grp]
            # the pair minima applied as swaps in x order: if they compose to the cluster's permutation they place
            # the character between the minima too (diabatic() uses them); otherwise one permutation at x0
            sw = sorted((c["x0"], int(c["pair"][0]), int(c["pair"][1])) for c in grp)
            pos = {int(b): int(b) for b in bl}  # character starting on branch b sits on branch pos[b]
            for _, u, v in sw:
                for key in pos:
                    if pos[key] == u:
                        pos[key] = v
                    elif pos[key] == v:
                        pos[key] = u
            if {i: j for i, j in pos.items() if i != j} == moved:
                ev["swaps"] = [{"x0": x, "branches": [u, v]} for x, u, v in sw]
        events.append(ev)
    # warnings next to an identified event (same branch, overlapping window) are artefacts of that event
    for (a, b), (x1, x2), msg in pending_warn:
        if any(e["kind"] == "avoided" and ({a, b} & set(e["branches"])) and e["window"][0] <= x2 and x1 <= e["window"][1]
               for e in events):
            continue
        warns.append(msg)
    events.sort(key=lambda e: e["x0"])
    return events, warns


# ------------------------------------------------------------------------------------------------- probes
def probe_grid_2d(lo: Sequence[float], hi: Sequence[float], n: int | Sequence[int] = 40, *, inset: float = 1e-3):
    """Points of a regular grid in the box [lo, hi] (2D: (r, z) or (x, y)), shrunk by ``inset`` x the box size so
    the edge rows sit just inside the walls; returns two flat arrays."""
    import numpy as np

    nx, ny = (n, n) if isinstance(n, int) else (int(n[0]), int(n[1]))
    dx, dy = (hi[0] - lo[0]) * inset, (hi[1] - lo[1]) * inset
    X, Y = np.meshgrid(np.linspace(lo[0] + dx, hi[0] - dx, nx), np.linspace(lo[1] + dy, hi[1] - dy, ny))
    return X.ravel(), Y.ravel()


def _solid_bbox(s: dict):
    import numpy as np

    if "cylinder" in s:
        c = s["cylinder"]
        b, ax, r = np.array(c["base"], float), np.array(c["axis"], float), float(c["radius"])
        if np.count_nonzero(ax) != 1:
            raise ValueError("probe_points_3d: only axis-aligned cylinders have a default probe box")
        e = np.array([r, r, r])
        e[np.nonzero(ax)[0][0]] = 0
        return np.minimum(b, b + ax) - e, np.maximum(b, b + ax) + e, ("cylinder", b, ax, r)
    if "box" in s:
        return np.array(s["box"]["min"], float), np.array(s["box"]["max"], float), None
    for key in ("union", "difference", "intersection"):
        if key in s:
            parts = s[key] if key == "union" else s[key][:1]
            bbs = [_solid_bbox(p) for p in parts]
            return np.min([b[0] for b in bbs], axis=0), np.max([b[1] for b in bbs], axis=0), \
                (bbs[0][2] if len(bbs) == 1 else None)
    raise ValueError(f"probe_points_3d: no default probe box for solid {list(s)}; pass points or bounds")


def probe_points_3d(design=None, n: Sequence[int] = (9, 9, 13), *, bounds=None, inset: float = 0.02,
                    jitter: float = 0.137) -> list[list[float]]:
    """Probe points for a 3D sweep (``fullwave.field_probes``): a lattice of n = (nx, ny, nz) points in ``bounds``
    ((lo, hi) corners) or the box of the design's ``mesh.domain_solid`` (cylinder or box), shrunk by ``inset`` x
    the size; for a cylinder only the points inside it are kept. The lattice is shifted by ``jitter`` x a cell (off
    the mirror planes, where odd modes vanish)."""
    import numpy as np

    shape = None
    if bounds is None:
        if design is None:
            raise ValueError("probe_points_3d needs a design (with mesh.domain_solid) or bounds")
        lo, hi, shape = _solid_bbox(design.doc["mesh"]["domain_solid"])
    else:
        lo, hi = np.array(bounds[0], float), np.array(bounds[1], float)
    size = hi - lo
    lo2, hi2 = lo + inset * size, hi - inset * size
    axes = []
    for d in range(3):
        m = int(n[d])
        h = (hi2[d] - lo2[d]) / m
        axes.append(lo2[d] + h * (np.arange(m) + 0.5 + jitter * (d + 1) / 3 - jitter / 2))
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if shape and shape[0] == "cylinder":
        _, b, ax, r = shape
        u = ax / np.linalg.norm(ax)
        rel = P - b
        rad = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        P = P[rad < r * (1 - inset)]
    return P.tolist()


# ------------------------------------------------------------------------------------------------- results -> step
def _row(m: dict) -> dict:
    return {k: m.get(k) for k in _ROW_KEYS if k in m}


def mode_step(result, value: Any, *, probes=None, families: Sequence[str] | None = None) -> ModeStep:
    """The :class:`ModeStep` of a solved result (2D axisymmetric ``Results``, ``XsecResults`` or ``Results3D``).

    ``probes``: 2D/xsec: a pair of flat arrays (r, z) or (x, y) where E is sampled (:func:`probe_grid_2d`); 3D:
    ignored (the design's ``fullwave.field_probes`` are read from each mode's ``probes.E``). ``families``: keep
    only these mode families ("TM", "TE"; 3D rows are "3D")."""
    import numpy as np

    rows = result.cavity_modes()
    if families is not None:
        rows = [r for r in rows if r["family"] in families]
    if rows and rows[0]["family"] == "3D":
        eig = result.fullwave.get("eigenmodes", {})
        fp = eig.get("field_probes")
        if not fp:
            raise ValueError("3D tracking needs fullwave.field_probes in the design (probe_points_3d)")
        valid = np.repeat(np.asarray(fp["valid"], dtype=bool), 3)
        vecs, out = [], []
        for r in rows:
            m = eig["modes"][r["index"]]
            vecs.append(np.asarray(m["probes"]["E"], dtype=float).ravel())
            rr = dict(r)
            for k in ("E_azimuthal_fraction",):
                if k in m:
                    rr[k] = m[k]
            out.append(_row(rr))
        V = np.array(vecs) if vecs else np.zeros((0, valid.size))
        return ModeStep(value, out, V, valid, None, {"backend": "3d", "n_probes": len(fp["valid"])})
    if probes is None:
        raise ValueError("2D/xsec tracking needs probe points (probe_grid_2d)")
    p0, p1 = (np.asarray(p, dtype=float) for p in probes)
    is_xsec = type(result).__name__ == "XsecResults"
    comps = ("E_x", "E_y", "E_z") if is_xsec else ("E_r", "E_phi", "E_z")
    vecs, eps = [], None
    for r in rows:
        S = result.cavity_mode_sample(r, p0, p1)
        vecs.append(np.concatenate([np.asarray(S[c], dtype=float).ravel() for c in comps]))
        if eps is None:
            eps = np.asarray(S.get("eps_r", np.ones(p0.size)), dtype=float).ravel()
    n = p0.size
    V = np.array(vecs) if vecs else np.zeros((0, 3 * n))
    valid = np.all(np.isfinite(V), axis=0) if V.shape[0] else np.ones(3 * n, bool)
    w1 = np.nan_to_num(eps if eps is not None else np.ones(n), nan=0.0)
    if not is_xsec:
        w1 = w1 * np.abs(p0)  # axisymmetric volume element r dr dz
    W = np.tile(w1, 3)
    return ModeStep(value, [_row(r) for r in rows], np.nan_to_num(V), valid, W,
                    {"backend": "xsec" if is_xsec else "axisymmetric", "n_probes": n})


def _bbox_2d(result):
    import numpy as np

    rows = result.cavity_modes()
    F = result.cavity_mode_field(rows[0], subdiv=1)
    k0, k1 = ("x", "y") if "x" in F else ("r", "z")
    a, b = np.asarray(F[k0]), np.asarray(F[k1])
    return (float(a.min()), float(b.min())), (float(a.max()), float(b.max()))


# ------------------------------------------------------------------------------------------------- driver
def _doc_hash(design) -> str:
    return hashlib.sha256(json.dumps(design.doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sweep(build: Callable[[Any], Any], values: Sequence[Any], *, workdir: str | Path = "runs/cavity_sweep",
          probes=None, probe_n: int | Sequence[int] | None = None, families: Sequence[str] | None = None,
          solve_kw: dict | None = None, refine: int = 0, min_step: float | None = None,
          refine_crossings: bool = True, track_kw: dict | None = None, verbose: int = 0,
          map_fn: Callable | None = None) -> Tracking:
    """Solve ``build(v)`` (a Design) for each value, collect the modes and their probe vectors, and :func:`track`
    them; returns the :class:`Tracking`.

    Probes: 2D/xsec: ``probes`` = (p0, p1) flat arrays, default a ``probe_n`` grid (40 x 40) over the union of the
    cavity's bounding boxes at the first and last values (two extra solves; stored in ``<workdir>/probes.json`` and reused); 3D: ``probes`` = a list of
    [x, y, z], default :func:`probe_points_3d` of the first design (``probe_n`` = (nx, ny, nz)), written into every
    design's ``fullwave.field_probes``.
    ``solve_kw``: passed to ``Design.solve`` (e.g. ``mpi_ranks``). Each step is cached in
    ``<workdir>/step_<hash of the design document>/`` (``track_step.npz``/``.json``): a re-run reuses it.
    ``refine``: rounds of bisection of :meth:`Tracking.needs_refinement` intervals (numeric values; not below
    ``min_step``; ``refine_crossings``: also crossings, to tell true crossings from narrow avoided ones).
    ``map_fn``: optional parallel map over the values of one round, e.g.
    ``concurrent.futures.ThreadPoolExecutor(n).map`` (the step function is a closure, so process pools cannot pickle
    it; 3D solves run as MPI subprocesses, so threads suffice)."""
    import numpy as np

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    solve_kw = dict(solve_kw or {})
    track_kw = dict(track_kw or {})
    pfile = wd / "probes.json"
    vals = list(values)
    if not vals:
        raise ValueError("sweep needs at least one value")
    d0 = build(vals[0])
    is3d0 = d0.doc.get("symmetry") == "3d"
    backend = "3d" if is3d0 else "2d"
    meta = {"backend": backend, "probe_n": list(probe_n) if isinstance(probe_n, (list, tuple)) else probe_n,
            "range": [str(vals[0]), str(vals[-1])],
            "designs": [_doc_hash(d0), _doc_hash(build(vals[-1])) if len(vals) > 1 else None]}
    state: dict = {"probes": None}
    if probes is not None:  # explicit: (p0, p1) flat arrays in 2D, [[x, y, z], ...] in 3D
        state["probes"] = [list(map(float, q)) for q in probes] if is3d0 else \
            [np.asarray(q, float).ravel().tolist() for q in probes]
    elif pfile.exists():  # a default probe set from an earlier run with the same backend, probe_n and range
        old = json.loads(pfile.read_text())
        if isinstance(old, dict) and old.get("meta") == json.loads(json.dumps(meta)):
            state["probes"] = old["probes"]
    fam_key = "" if families is None else json.dumps(sorted(families))

    def key(d, is3d):
        h = _doc_hash(d)
        if not is3d or fam_key:  # 2D: the probe set is not in the document; families filter the rows
            h = hashlib.sha256((h + ("" if is3d else json.dumps(state["probes"])) + fam_key).encode()).hexdigest()[:16]
        return h

    def fix_probes_2d(vs):
        """Default 2D/xsec probe grid: the union of the cavity bounding boxes at the given values."""
        los, his, axisym, ress = [], [], True, []
        for v in vs:
            d = build(v)
            axisym = d.doc.get("symmetry", "axisymmetric") != "xsec"
            res = d.solve(workdir=wd / "probe_box" / f"v{len(los)}", verbose=verbose, **solve_kw)
            lo, hi = _bbox_2d(res)
            los.append(lo)
            his.append(hi)
            ress.append((v, d, res))
        lo = (min(x[0] for x in los), min(x[1] for x in los))
        hi = (max(x[0] for x in his), max(x[1] for x in his))
        if axisym:
            lo = (max(lo[0], 0.0), lo[1])
        p0, p1 = probe_grid_2d(lo, hi, probe_n or 40)
        state["probes"] = [p0.tolist(), p1.tolist()]
        pfile.write_text(json.dumps({"meta": meta, "probes": state["probes"]}))
        for v, d, res in ress:  # these solves become cached steps
            h = key(d, False)
            sd = wd / f"step_{h}"
            if not (sd / "track_step.npz").exists():
                st = mode_step(res, v, probes=state["probes"], families=families)
                st.meta["design_hash"] = h
                st.save(sd)

    def solve_one(v):
        d = build(v)
        is3d = d.doc.get("symmetry") == "3d"
        if is3d:
            if state["probes"] is None:
                state["probes"] = probe_points_3d(d, probe_n or (9, 9, 13))
                pfile.write_text(json.dumps({"meta": meta, "probes": state["probes"]}))
            d.doc.setdefault("fullwave", {})["field_probes"] = state["probes"]
        elif state["probes"] is None:
            raise RuntimeError("internal: 2D probes not fixed")
        h = key(d, is3d)
        sd = wd / f"step_{h}"
        if (sd / "track_step.npz").exists() and (sd / "track_step.json").exists():
            st = ModeStep.load(sd)
            st.value = v
            return st
        res = d.solve(workdir=sd / "run", verbose=verbose, **solve_kw)
        st = mode_step(res, v, probes=None if is3d else state["probes"], families=families)
        st.meta["design_hash"] = h
        st.save(sd)
        return st

    if state["probes"] is None and not is3d0:
        fix_probes_2d([vals[0], vals[-1]] if len(vals) > 1 else vals[:1])
    return adaptive_track(solve_one, values, refine=refine, min_step=min_step, refine_crossings=refine_crossings,
                          track_kw=track_kw, map_fn=map_fn)


def adaptive_track(step_fn: Callable[[Any], ModeStep], values: Sequence[Any], *, refine: int = 0,
                   min_step: float | None = None, refine_crossings: bool = True, track_kw: dict | None = None,
                   map_fn: Callable | None = None) -> Tracking:
    """:func:`track` the steps ``step_fn(v)`` for ``values``, then bisect the intervals of
    :meth:`Tracking.needs_refinement` for up to ``refine`` rounds (not below ``min_step``). The first value is
    solved alone (it may fix the probe set), the rest through ``map_fn`` (default: serial). Crossings get
    ``resolution`` = the width of their final interval: a crossing that survives refinement with overlaps near 1
    is a true crossing to that resolution."""
    track_kw = dict(track_kw or {})
    vals = list(values)
    if not vals:
        raise ValueError("adaptive_track needs at least one value")
    steps: list[ModeStep] = [step_fn(vals[0])]
    rest = vals[1:]
    steps += list(map_fn(step_fn, rest)) if map_fn else [step_fn(v) for v in rest]
    T = track(steps, **track_kw)
    rounds = 0
    for _ in range(int(refine)):
        need = T.needs_refinement(crossings=refine_crossings)
        mids = []
        for lo, hi in need:
            if min_step is not None and abs(hi - lo) / 2 < min_step:
                continue
            m = 0.5 * (lo + hi)
            if not any(abs(m - s.value) <= 1e-12 * max(1.0, abs(m)) for s in steps) and m not in mids:
                mids.append(m)
        if not mids:
            break
        rounds += 1
        steps += list(map_fn(step_fn, mids)) if map_fn else [step_fn(v) for v in mids]
        T = track(steps, **track_kw)
    T.options["refine_rounds"] = rounds
    for e in T.crossings:
        if e["kind"] == "crossing":
            e["resolution"] = abs(float(e["hi"]) - float(e["lo"])) if _is_num(e["hi"]) else None
    return T


# ------------------------------------------------------------------------------------------------- synthetic model
def two_level_steps(xs: Sequence[float], *, f0: float = 5e9, slope: float = 1e8, kappa: float = 1e6,
                    n_samples: int = 64, extra: Sequence[tuple] = (), seed: int = 0, noise: float = 0.0,
                    rotate_sign: bool = True) -> list[ModeStep]:
    """Synthetic sweep with an analytic two-level avoided crossing (for tests and docs): diabatic modes
    f_a = f0 + slope x, f_b = f0 - slope x coupled by kappa [Hz]; adiabatic f_+- = f0 +- sqrt((slope x)^2 +
    kappa^2), eigenvectors (cos t, sin t) in the (a, b) basis with tan 2t = kappa / (slope x). ``extra``: further
    uncoupled modes (f_at_0, slope) with their own orthogonal fields (kappa = 0 between them: true crossings).
    The "fields" are orthonormal random vectors of length ``n_samples``; signs are flipped at random per step (as a
    real eigensolver does) when ``rotate_sign``. ``C`` is carried by the a-mode: C = 0.6 cos^2 t (the "TM010-like"
    character), so the adiabatic branches show the C exchange."""
    import numpy as np

    rng = np.random.default_rng(seed)
    nb = 2 + len(extra)
    Qb, _ = np.linalg.qr(rng.standard_normal((n_samples, nb)))
    ea, eb = Qb[:, 0], Qb[:, 1]
    steps = []
    for x in xs:
        d = slope * x
        H = np.array([[f0 + d, kappa], [kappa, f0 - d]])
        w, U = np.linalg.eigh(H)
        modes, vecs = [], []
        for j in range(2):
            c = U[:, j]
            v = c[0] * ea + c[1] * eb
            if noise:
                v = v + noise * rng.standard_normal(n_samples)
            s = rng.choice([-1.0, 1.0]) if rotate_sign else 1.0
            modes.append({"f_Hz": float(w[j]), "C": float(0.6 * c[0] ** 2), "Q0": 1e4 * (1 + 0.1 * c[1] ** 2)})
            vecs.append(s * v)
        for q, (fe, se) in enumerate(extra):
            modes.append({"f_Hz": float(fe + se * x), "C": 0.0, "Q0": 2e4})
            vecs.append(Qb[:, 2 + q] * (rng.choice([-1.0, 1.0]) if rotate_sign else 1.0))
        order = np.argsort([m["f_Hz"] for m in modes])
        steps.append(ModeStep(float(x), [modes[i] for i in order], np.array(vecs)[order]))
    return steps
