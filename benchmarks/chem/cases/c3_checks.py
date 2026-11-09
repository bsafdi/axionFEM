"""Current-distribution check module (planar, axisymmetric and 3D cases): the error computation, numpy and SciPy only.

It never imports SymPy or mpmath: manufactured solutions are read from benchmarks/chem/casedata/c3/mms_*.json (sha256
in run.parameters) through their NumPy spellings; closed-form and cross-code references are reference values of the
case definitions, never read here.

The case runner calls these functions with RAW numbers from the solver (quadrature points, weights without
r, field values and gradients, the full residual vector, mesh vertices and elements); every current, balance, error,
order series and check value comes from here.  reaction_current() assumes a partition-of-unity (Lagrange or
Bernstein) basis: with a hierarchical basis only the vertex dofs of the electrode may be summed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

FORMAT = "axionfem.chem.c3.checks/1"


# ------------------------------------------------------------------------------------------------ data
class DataError(ValueError):
    pass


def load_json(path, sha256: str):
    """Load a fixed data file and verify its sha256 (refuses a modified file)."""
    raw = Path(path).read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != sha256:
        raise DataError(f"{path}: sha256 {h} differs from the fixed {sha256}")
    return json.loads(raw)


def np_fn(src: str, coords):
    """Callable X (n, d) -> (n,) from a NumPy spelling written by c3_mms.py (namespace: numpy, coordinates)."""
    code = compile(src, "<mms>", "eval")

    def f(X):
        ns = {"numpy": np}
        for i, c in enumerate(coords):
            ns[c] = X[:, i]
        v = eval(code, {"__builtins__": {}}, ns)
        return np.broadcast_to(np.asarray(v, dtype=float), (X.shape[0],)).copy()
    return f


def mms_functions(data: dict, variant: str | None = None) -> dict:
    """phi, grad, kappa, f (and per-electrode g, flux) of a manufactured field as numpy callables."""
    c = data["coords"]
    n = data["numpy"] if "numpy" in data else {}
    out = {}
    if "phi" in n:
        out["phi"] = np_fn(n["phi"], c)
        gs = [np_fn(s, c) for s in n["grad_phi"]]
        out["grad"] = lambda X: np.stack([g(X) for g in gs], 1)
        out["kappa"] = np_fn(n["kappa"], c)
        out["f"] = np_fn(n["f"], c)
    else:   # piecewise (kappa jump at x = L/2)
        Lh = 0.5 * float(eval(data["L"], {"__builtins__": {}}, {}))
        pl, pr = np_fn(n["phi_left"], c), np_fn(n["phi_right"], c)
        gl = [np_fn(s, c) for s in n["grad_left"]]
        gr = [np_fn(s, c) for s in n["grad_right"]]
        left = lambda X: X[:, 0] < Lh
        out["phi"] = lambda X: np.where(left(X), pl(X), pr(X))
        out["grad"] = lambda X: np.where(left(X)[:, None], np.stack([g(X) for g in gl], 1), np.stack([g(X) for g in gr], 1))
    if variant is not None:
        for role, e in data["variants"][variant].items():
            out[f"g_{role}"] = np_fn(e["numpy"]["g"], c)
            out[f"flux_{role}"] = np_fn(e["numpy"]["flux"], c)
    return out


# ------------------------------------------------------------------------------------------------ norms
def field_errors(x, w, phi_h, grad_h, fns: dict, axisymmetric: bool) -> dict:
    """L2 and H1-seminorm errors from the solver's quadrature data.

    x (n, d): physical quadrature points; w (n,): weights INCLUDING |det J| but NOT the axisymmetric r (this module
    applies r itself, so a solver that drops or doubles r in its weights cannot hide it); phi_h (n,), grad_h (n, d).
    The exact solution is evaluated at the points (never interpolated).  Axisymmetric: r-weighted norms (the order
    items) and unweighted ones (reported)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ue, ge = fns["phi"](x), fns["grad"](x)
    e2 = (np.asarray(phi_h) - ue) ** 2
    g2 = np.sum((np.asarray(grad_h) - ge) ** 2, 1)
    out = {"L2": math.sqrt(float(np.sum(e2 * w))), "H1": math.sqrt(float(np.sum(g2 * w))),
           "magnitude": float(np.max(np.abs(ue))), "grad_magnitude": float(np.max(np.linalg.norm(ge, axis=1)))}
    if axisymmetric:
        out["L2_w"] = math.sqrt(float(np.sum(e2 * w * x[:, 0])))
        out["H1_w"] = math.sqrt(float(np.sum(g2 * w * x[:, 0])))
        out["volume_w"] = float(np.sum(w * x[:, 0]))
    out["volume"] = float(np.sum(w))
    return out


def quadrature_volume_problems(volume, expected, rel=1e-12, what="domain"):
    """The weights must integrate 1 over the (straight or exactly mapped) domain to round-off."""
    if not abs(volume - expected) <= rel * abs(expected):
        return [f"quadrature weights integrate the {what} to {volume!r}, expected {expected!r} (rel {rel:g})"]
    return []


def rel_err(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


# ------------------------------------------------------------------------------------------------ meshes
def vertex_hash(vertices, scale: float) -> str:
    """Order-independent hash of a vertex set: coordinates / scale rounded to 1e-9, sorted, sha256 (16 hex)."""
    V = np.rint(np.asarray(vertices, float) / scale * 1e9).astype(np.int64)
    V = V[np.lexsort(V.T[::-1])]
    return hashlib.sha256(V.tobytes()).hexdigest()[:16]


def mesh_problems(level_spec: dict, vertices, n_elements: int) -> list:
    """The solver's mesh is the fixed one: vertex-set hash and element count (run.parameters.meshes.<family>)."""
    probs = []
    h = vertex_hash(vertices, level_spec["hash_scale"])
    if h != level_spec["vertex_hash"]:
        probs.append(f"mesh {level_spec['name']}: vertex hash {h} != fixed {level_spec['vertex_hash']}")
    if int(n_elements) != int(level_spec["n_elements"]):
        probs.append(f"mesh {level_spec['name']}: {n_elements} elements, fixed {level_spec['n_elements']}")
    return probs


# ------------------------------------------------------------------------------------------------ series / checks
def order_series(levels, errors, magnitude) -> dict:
    return {"levels": list(levels), "error": [float(e) for e in errors], "solution_magnitude": float(magnitude)}


def functional_series(levels, values) -> dict:
    return {"levels": list(levels), "values": [float(v) for v in values]}


def taylor_test(eps, remainders) -> dict:
    """Jacobian Taylor test: remainders r(eps) = |R(z + eps v) - R(z) - eps J v|; returns pairwise slopes and the
    least-squares slope over the window where r is above 1e3 * machine-epsilon * |R| (the caller passes only those)."""
    e, r = np.log(np.asarray(eps, float)), np.log(np.asarray(remainders, float))
    slopes = [float((r[i + 1] - r[i]) / (e[i + 1] - e[i])) for i in range(len(e) - 1)]
    A = np.vstack([e, np.ones_like(e)]).T
    fit = float(np.linalg.lstsq(A, r, rcond=None)[0][0])
    return {"pairwise": slopes, "fit": fit, "min": min(slopes)}


def newton_quadratic_constant(history, last=3) -> dict:
    """Quadratic-convergence diagnostics from the relative-residual history: C_k = r_{k+1} / r_k^2 over the last
    iterations above round-off (1e-13).  A bounded C (and r_{k+1}/r_k -> 0) is quadratic convergence; a Picard or
    wrong-Jacobian iteration shows r_{k+1}/r_k -> const > 0."""
    h = [float(v) for v in history if v > 1e-13]
    if len(h) < 3:
        return {"ratios": [], "C": [], "final_ratio": 0.0}
    ratios = [h[i + 1] / h[i] for i in range(len(h) - 1)]
    C = [h[i + 1] / h[i] ** 2 for i in range(len(h) - 1)]
    return {"ratios": ratios[-last:], "C": C[-last:], "final_ratio": ratios[-1]}


def sampled_profile_problems(r_values, values, frozen_r, at_tol):
    if len(r_values) != len(frozen_r) or any(abs(a - b) > at_tol for a, b in zip(r_values, frozen_r)):
        return [f"profile sampled at {list(r_values)}, fixed {list(frozen_r)}"]
    return []


# ----------------------------------------------------------------------------------------------------- currents
F_CONST, R_CONST = 96485.33212, 8.314462618


def _num(v):
    """A rational/decimal string or a number from the data file (no SymPy)."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def kinetic_density(reactions, eta_by_reaction, T=298.15):
    """Sum_j i_j(eta_j) at points, an implementation independent of the solver's kernel.
    eta_by_reaction: list of arrays, one per reaction (eta_j = phi_s - phi_h - E_eq,j at the points)."""
    f = F_CONST / (R_CONST * T)
    tot = 0.0
    for rx, eta in zip(reactions, eta_by_reaction):
        k, i0 = rx["kind"], _num(rx["i0"])
        if k == "bv":
            tot = tot + i0 * (np.exp(_num(rx["alpha_a"]) * f * eta) - np.exp(-_num(rx["alpha_c"]) * f * eta))
        elif k == "linear":
            tot = tot + i0 * (_num(rx["alpha_a"]) + _num(rx["alpha_c"])) * f * eta
        elif k == "tafel_a":
            tot = tot + i0 * 10.0 ** (eta / _num(rx["A"]))
        elif k == "tafel_c":
            tot = tot - i0 * 10.0 ** (eta / _num(rx["A"]))
        else:
            raise DataError(f"unknown kinetics {k}")
    return tot


def electrode_current(x, w, phi_h, phi_s, reactions, axisymmetric, g=None, weight=None, T=298.15):
    """Kinetic current of an electrode from the solver's RAW boundary quadrature: points x (n, d), weights w (n,)
    with the surface Jacobian but WITHOUT r, the discrete potential phi_h at the points, the metal potential, the
    fixed reactions (and the manufactured g for MMS).  3D totals (2 pi int r dA in axisymmetry).  `weight`: an
    optional callable w(x) for the junction-weighted current."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    etas = [phi_s - np.asarray(phi_h, float) - _num(rx["E_eq"]) for rx in reactions]
    dens = kinetic_density(reactions, etas, T)
    if g is not None:
        dens = dens + g(x)
    if weight is not None:
        dens = dens * weight(x)
    ww = w * x[:, 0] * 2 * math.pi if axisymmetric else w
    return float(np.sum(dens * ww))


def boundary_load(x, w, q, axisymmetric):
    """int q dA from raw boundary quadrature (insulator flux data), 3D totals."""
    x = np.asarray(x, float)
    ww = np.asarray(w, float) * (x[:, 0] * 2 * math.pi if axisymmetric else 1.0)
    return float(np.sum(q(x) * ww))


def volume_load(x, w, f, axisymmetric):
    x = np.asarray(x, float)
    ww = np.asarray(w, float) * (x[:, 0] * 2 * math.pi if axisymmetric else 1.0)
    return float(np.sum(f(x) * ww))


def electrode_measure(x, w, axisymmetric):
    """int 1 dA over an electrode from the solver's raw boundary quadrature (2 pi int r dl in axisymmetry):
    the electrode-measure identity (facet tagging)."""
    x = np.asarray(x, float)
    return float(np.sum(np.asarray(w, float) * (x[:, 0] * 2 * math.pi if axisymmetric else 1.0)))


def reaction_current(residual, electrode_dofs, axisymmetric):
    """Reaction (variational) current: the sum of the full residual vector (weak form WITHOUT the electrode's own
    boundary term) over the electrode's dofs; per radian in axisymmetry, hence x 2 pi."""
    v = float(np.sum(np.asarray(residual, float)[np.asarray(electrode_dofs, int)]))
    return v * (2 * math.pi if axisymmetric else 1.0)


def balance(currents, other_loads, scale):
    """Current-balance identity value |sum I_e + sum loads| / scale (bound newton_tol)."""
    return abs(sum(currents) + sum(other_loads)) / abs(scale)


def connectivity_hash(vertices, elements, scale: float) -> str:
    """Order-independent hash of the triangulation itself: each element as the sorted tuple of
    its rounded vertex coordinates; the sorted list of elements is hashed.  A flipped diagonal or any reconnection
    of the same vertex set changes it."""
    V = np.rint(np.asarray(vertices, float) / scale * 1e9).astype(np.int64)
    E = np.asarray(elements, int)
    rows = []
    for el in E:
        pts = sorted(tuple(V[i]) for i in el)
        rows.append(tuple(c for p in pts for c in p))
    rows.sort()
    return hashlib.sha256(np.array(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


def newton_window_ratio(history, lo=1e-12, hi=1e-3):
    """The last residual reduction factor among iterates with relative residual in [lo, hi]: a
    definition independent of where the iteration happens to stop.  0 when fewer than two iterates fall in it."""
    win = [float(v) for v in history if lo <= v <= hi]
    return win[-1] / win[-2] if len(win) >= 2 else 0.0
