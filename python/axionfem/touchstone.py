"""S-, Y- and Z-parameters of full-wave port results, and Touchstone v1.0/v1.1 files.

axionFEM's full-wave engines (axisymmetric and 3D) use the time dependence ``e^{-i omega t}``: a passive inductor has
``Z = -i omega L`` (``docs/physics/multiport_fullwave.md``, :mod:`axionfem.circuit`).  Touchstone files, network
analysers and circuit simulators use ``e^{+j omega t}``, where the same inductor has ``Z = +j omega L``.  A physical
response ``Re[Z e^{-i omega t}]`` equals ``Re[conj(Z) e^{+j omega t}]``, so the impedance matrix in the engineering
convention is the element-wise complex conjugate of the engine's (no transpose: the port order and the reciprocity
``Z = Z^T`` are unchanged).  Every function here returns engineering-convention (``e^{+j omega t}``) quantities;
the functions that take an impedance matrix say which convention it is in through a required ``convention``
argument (``"axionfem"``: ``e^{-i omega t}``, conjugated on input; ``"engineering"``: ``e^{+j omega t}``, used as
given).

With real positive reference impedances ``z0_k`` (one per port, or one for all) and ``D = diag(sqrt(z0))``, the
normalised impedance is ``z = D^-1 Z D^-1`` and

    S = (z - 1)(z + 1)^-1,        Z = D (1 + S)(1 - S)^-1 D,        Y = Z^-1.

These are the power-wave (Kurokawa) scattering parameters, which coincide with the pseudo-wave ones for real ``z0``.
``S`` is symmetric exactly when ``Z`` is (reciprocity), and the network is passive exactly when the singular values
of ``S`` are at most 1 (equivalently ``(Z + Z^H)/2 >= 0``).  A passivity violation is reported, never repaired.

Touchstone v1.0 (``.sNp``): option line ``# Hz S RI R 50``, frequencies strictly increasing, one data block per
frequency; a 2-port block is ``N11 N21 N12 N22`` (the v1 exception), 3 ports and more are written row by row with at
most four complex values per line; Z- and Y-parameters are stored normalised (``Z/R`` and ``Y R``).  A per-port
reference resistance is written as the Version 1.1 option line ``# Hz S RI R r1 ... rn``, for S-parameters only:
Z and Y with per-port resistances are refused (readers do not agree on their normalisation).

See ``docs/physics/touchstone.md``.
"""
from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_Z0 = 50.0
# the largest sigma_max(S) - 1 that is not reported as a passivity violation (roundoff of a lossless solve)
PASSIVITY_TOLERANCE = 1e-9
# the condition number of the equilibrated Z (D^-1 Z D^-1, D = diag sqrt|Z_kk|) above which z_to_y warns
Y_CONDITION_LIMIT = 1e12
# the estimated relative error of Y (the error estimate of Z over sigma_min(Z)) above which a Y export warns
Y_RELATIVE_ERROR_LIMIT = 0.1
# R_loss / |X| above which the perturbative surface loss is more than a ~1 % approximation (the engine's own warning)
SURFACE_LOSS_PERTURBATION_LIMIT = 0.1
CONVENTIONS = ("axionfem", "engineering")
KINDS = ("S", "Z", "Y")
_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
# a Touchstone number: decimal or scientific notation (no nan, inf, hex or '_' digit separators)
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# command-line exit codes
EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NOT_PASSIVE = 3


# ------------------------------------------------------------------ small helpers
def _clean(text: Any) -> str:
    """One comment-safe US-ASCII line (Touchstone allows only ASCII graphic characters): control characters
    (newlines included) become spaces, every other character outside 0x20-0x7E (non-ASCII letters, U+0085, U+2028,
    U+2029) becomes '?'."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(text))
    return re.sub(r"[^\x20-\x7e]", "?", text)


def safe_stem(name: Any, fallback: str = "fullwave") -> str:
    """A file-name stem from a design name: letters, digits, '_', '-', '.'; no path separators, no leading dot or
    dash (the MCP name rule; a leading '-' reads as an option to shell tools)."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or ""))[:80].lstrip(".-")
    return stem or fallback


def _fmt(x: float) -> str:
    return f"{x:.17g}"


def _hz_list(f: Sequence[float], limit: int = 8) -> str:
    vals = [f"{x:.9g}" for x in list(f)[:limit]]
    more = f" and {len(f) - limit} more" if len(f) > limit else ""
    return ", ".join(vals) + more


# ------------------------------------------------------------------ conventions and conversions
def _square(M: Any, who: str) -> np.ndarray:
    A = np.asarray(M, dtype=complex)
    if A.ndim == 0:
        A = A.reshape(1, 1)
    elif A.ndim == 1:
        A = A.reshape(-1, 1, 1)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{who}: expected a square matrix or a stack of them (... x n x n), got shape "
                         f"{np.shape(M)}; give one-port data as a 1-D array of values or an (n_f, 1, 1) array")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{who}: the matrix has non-finite entries; remove those frequencies first")
    return A


def to_engineering(Z: Any, convention: str) -> np.ndarray:
    """``Z`` (or any network matrix) in the engineering convention ``e^{+j omega t}``.

    ``convention="axionfem"``: ``Z`` is in the engine's ``e^{-i omega t}`` convention (``Z = -i omega L`` for an
    inductor) and is complex-conjugated element-wise.  ``convention="engineering"``: returned unchanged.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be 'axionfem' (the engine's exp(-i omega t), Z = -i omega L for an "
                         f"inductor) or 'engineering' (exp(+j omega t), Z = +j omega L), not {convention!r}")
    A = np.asarray(Z, dtype=complex)
    return np.conj(A) if convention == "axionfem" else A.copy()


def _z0_vector(z0: Any, n: int, who: str) -> np.ndarray:
    v = np.asarray(z0, dtype=complex).ravel()
    if v.size == 1:
        v = np.full(n, v[0])
    if v.size != n:
        raise ValueError(f"{who}: z0 has {v.size} entries for {n} ports; give one value or one per port")
    if np.any(v.imag != 0) or np.any(~np.isfinite(v.real)) or np.any(v.real <= 0):
        raise ValueError(f"{who}: z0 = {np.asarray(z0).tolist()} must be real, finite and positive (ohm); complex "
                         "reference impedances (power-wave versus pseudo-wave definitions) are not supported")
    return v.real


def _right_divide(num: np.ndarray, den: np.ndarray, who: str, what: str) -> np.ndarray:
    """``num @ inv(den)`` per matrix; a singular ``den`` raises an error naming the matrices (indices)."""
    try:
        return np.swapaxes(np.linalg.solve(np.swapaxes(den, -1, -2), np.swapaxes(num, -1, -2)), -1, -2)
    except np.linalg.LinAlgError:
        flat = den.reshape(-1, den.shape[-2], den.shape[-1])
        bad = [i for i, m in enumerate(flat) if np.linalg.matrix_rank(m) < m.shape[0]]
        raise ValueError(f"{who}: {what} is singular at matrix index {bad[:8]} of {flat.shape[0]}; the conversion "
                         "does not exist there (drop those frequencies or use another parameter type)") from None


def z_to_s(Z: Any, z0: Any = DEFAULT_Z0, *, convention: str) -> np.ndarray:
    """Scattering matrix ``S = (z - 1)(z + 1)^-1``, ``z = D^-1 Z D^-1``, ``D = diag(sqrt(z0))``.

    ``Z``: an ``n x n`` impedance matrix or a stack (``... x n x n``; a 1-D array is a one-port sweep) in ohm.
    ``z0``: the real positive reference impedance, one value or one per port (ohm).  ``convention``: the time
    convention of ``Z`` (``"axionfem"`` or ``"engineering"``, see :func:`to_engineering`).  The result is in the
    engineering convention ``e^{+j omega t}``, with the shape of ``Z`` (``n_f x 1 x 1`` for a 1-D input).  Raises a
    ValueError naming the matrices where ``Z + z0`` is singular (e.g. ``Z = -z0`` for a one-port).
    """
    Ze = _square(to_engineering(Z, convention), "z_to_s")
    n = Ze.shape[-1]
    d = np.sqrt(_z0_vector(z0, n, "z_to_s"))
    z = Ze / d[:, None] / d[None, :]
    eye = np.eye(n)
    return _right_divide(z - eye, z + eye, "z_to_s", "Z + z0")


def s_to_z(S: Any, z0: Any = DEFAULT_Z0) -> np.ndarray:
    """Impedance matrix ``Z = D (1 + S)(1 - S)^-1 D`` (ohm) of engineering-convention S-parameters (the inverse of
    :func:`z_to_s`; the result is in the engineering convention too).  Raises a ValueError where ``1 - S`` is
    singular (Z does not exist: an open, an ideal thru)."""
    A = _square(S, "s_to_z")
    n = A.shape[-1]
    d = np.sqrt(_z0_vector(z0, n, "s_to_z"))
    eye = np.eye(n)
    z = _right_divide(eye + A, eye - A, "s_to_z", "1 - S")
    return z * d[:, None] * d[None, :]


def equilibrated_condition(Z: Any) -> np.ndarray:
    """Condition number of ``D^-1 Z D^-1`` with ``D = diag(sqrt|Z_kk|)`` (the raw ``Z`` where a diagonal entry is 0):
    insensitive to the scale of the ports, so a graded but well-determined matrix is not flagged.  1 for a one-port."""
    A = _square(Z, "equilibrated_condition")
    diag = np.abs(np.diagonal(A, axis1=-2, axis2=-1))
    ok = np.all(diag > 0, axis=-1, keepdims=True)
    d = np.sqrt(np.where(ok, diag, 1.0))
    return np.linalg.cond(A / d[..., :, None] / d[..., None, :])


def z_to_y(Z: Any, *, convention: str) -> np.ndarray:
    """Admittance matrix ``Y = Z^-1`` (siemens) in the engineering convention.

    Warns where the equilibrated condition number of ``Z`` (:func:`equilibrated_condition`) exceeds
    :data:`Y_CONDITION_LIMIT`, i.e. where the ports are so nearly dependent that ``Y`` is not determined by ``Z``.  A
    one-port is never flagged here: its ``Y = 1/Z`` has the relative error of ``Z``, which :func:`write_touchstone`
    reports for every kind (``Z_poorly_determined_Hz``; e.g. near a series resonance, where ``|Z|`` approaches the
    absolute error of ``Z``).
    """
    Ze = _square(to_engineering(Z, convention), "z_to_y")
    cond = np.atleast_1d(equilibrated_condition(Ze))
    bad = cond > Y_CONDITION_LIMIT
    if np.any(bad):
        warnings.warn(f"z_to_y: Z is nearly singular at {int(bad.sum())} of {bad.size} matrices (equilibrated "
                      f"condition number up to {float(np.max(cond)):.3g} > {Y_CONDITION_LIMIT:.0e}): Y there is not "
                      "determined by Z", RuntimeWarning, stacklevel=2)
    try:
        return np.linalg.inv(Ze)
    except np.linalg.LinAlgError:
        raise ValueError("z_to_y: Z is exactly singular (Y does not exist), e.g. a lossless series resonance at a "
                         "solved frequency; export S or Z instead") from None


def y_to_z(Y: Any) -> np.ndarray:
    """Impedance matrix ``Z = Y^-1`` (ohm) of an engineering-convention admittance matrix (ValueError where ``Y`` is
    singular)."""
    A = _square(Y, "y_to_z")
    return _right_divide(np.broadcast_to(np.eye(A.shape[-1]), A.shape).astype(complex), A, "y_to_z", "Y")


def reciprocity_error(M: Any) -> np.ndarray:
    """``||M - M^T||_F / ||M||_F`` per matrix (0 for a one-port): the asymmetry of ``Z``, ``Y`` or ``S`` (scalar
    ``z0``) of a reciprocal network, which the engine does not symmetrise."""
    A = _square(M, "reciprocity_error")
    num = np.linalg.norm(A - np.swapaxes(A, -1, -2), axis=(-2, -1))
    den = np.linalg.norm(A, axis=(-2, -1))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def passivity(S: Any, tol: float = PASSIVITY_TOLERANCE) -> dict[str, Any]:
    """Largest singular value of each scattering matrix and the indices where it exceeds ``1 + tol``.

    Returns ``sigma_max`` (array), ``excess`` (``sigma_max - 1``) and ``violations`` (indices).  Nothing is clipped:
    a passive network has ``sigma_max <= 1``, a lossless one ``= 1``; an excess above roundoff in an FEM result is
    a solver or discretisation error (``docs/physics/multiport_fullwave.md``: the engine does not symmetrise or
    project its Z) and is reported, not repaired.
    """
    A = _square(S, "passivity")
    sigma = np.linalg.svd(A, compute_uv=False)[..., 0]
    excess = sigma - 1.0
    return {"sigma_max": sigma, "excess": excess, "violations": np.flatnonzero(np.atleast_1d(excess) > tol)}


# ------------------------------------------------------------------ the network of a solved design
def _matrix_of(re: Any, im: Any) -> np.ndarray:
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


def _port_info(results: Mapping[str, Any], n: int) -> tuple[list[str], list[str]]:
    """Port names (comment-safe) and, per port, its polarity and terminals for the comment block."""
    ports = (results.get("fullwave") or {}).get("ports") or []
    if len(ports) != n:
        ports = [{}] * n
    names, details = [], []
    for k, p in enumerate(ports):
        p = p if isinstance(p, Mapping) else {}
        names.append(_clean(p.get("name") or f"port_{k + 1}").replace("|", "/").strip() or f"port_{k + 1}")
        bits = []
        for key in ("toward", "direction"):
            if p.get(key) is not None:
                bits.append(f"{key} {_vector_text(p[key])}")
        for key in ("region", "boundary", "terminal_patches", "port_pieces"):
            if p.get(key) is not None:
                bits.append(f"{key} {_clean(p[key])}")
        details.append("; ".join(bits))
    return names, details


def _vector_text(v: Any) -> str:
    """``[x, y, z]`` for a numeric vector; anything else as cleaned text (a decoration is never fatal)."""
    try:
        return "[" + ", ".join(f"{float(x):.6g}" for x in v) + "]"
    except (TypeError, ValueError):
        return _clean(v)


def _row_matrix(r: Mapping[str, Any], n: int | None) -> np.ndarray:
    if "Z_matrix_re_ohm" in r:
        return _matrix_of(r["Z_matrix_re_ohm"], r["Z_matrix_im_ohm"])
    if n not in (None, 1):
        raise ValueError(f"port_parameters: the row at {r['f_Hz']:.9g} Hz has no Z_matrix_re_ohm/Z_matrix_im_ohm "
                         f"while other rows have {n} ports; the sweep mixes port sets, re-solve the design")
    return np.array([[complex(r["Z_re_ohm"], r["Z_im_ohm"])]])


def _row_residual(r: Mapping[str, Any]) -> float:
    """The row's true relative linear residual (NaN when the solve does not report one, e.g. a direct 2D solve)."""
    unresolved = r.get("Re_Z_unresolved")
    if isinstance(unresolved, Mapping) and unresolved.get("true_relative_residual") is not None:
        return float(unresolved["true_relative_residual"])
    solver = r.get("solver")
    if isinstance(solver, Mapping) and isinstance(solver.get("relative_residual"), (int, float)):
        return float(solver["relative_residual"])
    return float("nan")


def _row_error(r: Mapping[str, Any], M: np.ndarray) -> tuple[float, str]:
    """The estimated normwise relative error of the row's Z, ``||dZ|| / ||Z||_2``, and what it rests on.

    Per port the error indicator's ``Z_signed_error_estimate_ohm`` (``|re + i im|``) is preferred; the unsigned
    ``Z_error_estimate_ohm`` counts the electric and magnetic parts with the same sign and is an upper estimate near a
    self-resonance (docs/physics/fullwave_error_indicator.md).  The solve's true relative residual is added as a
    relative error.  NaN (and "none") when the row reports neither (e.g. a direct 2D solve)."""
    est = []
    kinds = set()
    for src in [r] + list(r.get("port_solutions") or []):
        ei = src.get("error_indicator") if isinstance(src, Mapping) else None
        if not isinstance(ei, Mapping):
            continue
        signed = ei.get("Z_signed_error_estimate_ohm")
        if isinstance(signed, Mapping) and isinstance(signed.get("re"), (int, float)):
            est.append(abs(complex(float(signed["re"]), float(signed.get("im") or 0.0))))
            kinds.add("signed error indicator")
        elif isinstance(ei.get("Z_error_estimate_ohm"), (int, float)):
            est.append(float(ei["Z_error_estimate_ohm"]))
            kinds.add("unsigned error indicator (an upper estimate)")
    norm = float(np.linalg.norm(M, 2))
    rel = max(est) / norm if est and norm > 0 else float("nan")
    rho = _row_residual(r)
    if math.isfinite(rho):
        rel = rho if not math.isfinite(rel) else rel + rho
        kinds.add("solver residual")
    return rel, (", ".join(sorted(kinds)) or "none")


def port_parameters(results: Mapping[str, Any], *, dense: bool = False, surface_losses: bool = False) -> dict:
    """The port impedance matrix of a full-wave sweep, sorted by frequency, in the engineering convention.

    ``results``: a results document (``Results.data`` or a loaded ``results.json``) with ``fullwave.sweep``.
    ``dense=False`` takes the solved rows (``Z_matrix_re_ohm``/``Z_matrix_im_ohm``, or ``Z_re_ohm``/``Z_im_ohm`` for a
    one-port 2D sweep); ``dense=True`` the adaptive sweep's dense model curve (``fullwave.adaptive.dense``,
    :func:`axionfem.circuit.fullwave_dense`).  ``surface_losses=True`` adds the resistive part of the perturbative
    conductor surface loss of the solved rows, which ``Z`` does not contain (``docs/physics/
    fullwave_surface_losses.md``): ``R_loss_ohm`` to a one-port's ``Re Z``; for a multiport ``R_loss_matrix_ohm``,
    which is the impedance perturbation only for in-phase port fields, so it is refused when that matrix has an
    imaginary part above 1e-6 of its norm or ``Z`` has a resistive part (radiation or bulk loss).  The surface
    reactance (internal inductance, kinetic inductance) is not added.  Rows where ``R_loss/|X|`` exceeds
    :data:`SURFACE_LOSS_PERTURBATION_LIMIT` are warned about and listed (the perturbation is not accurate there).

    Returns ``f_Hz``, ``Z`` (``n_f x n x n``, ohm, ``e^{+j omega t}``), ``ports`` (names in matrix order; port 1
    is the pickup), ``port_details`` (polarity and terminals), ``source``, ``notes`` (lines for a file header),
    ``Re_Z_unresolved_Hz`` (solved rows whose ``Re Z`` the solve's residual does not resolve; for ``dense=True`` the
    rows the model was fitted to), ``Re_Z_unresolved`` (per output frequency: flagged itself, or for the dense curve
    between flagged neighbouring solves), ``Z_relative_error`` (per output frequency, the estimated normwise relative
    error of Z: the rows' error indicator, signed where given, plus the solve's residual; the dense curve's model
    error; NaN when not known), ``Z_relative_error_source``, ``Z_relative_error_kind`` (``"normwise"`` for the rows;
    ``"proportional"`` for the dense curve, whose barycentric model shares one denominator over all entries, so that
    its error near a pole moves the entries together) and ``surface_loss_beyond_perturbation_Hz``.
    """
    fw = results.get("fullwave") if isinstance(results, Mapping) else None
    if not isinstance(fw, Mapping):
        raise ValueError("port_parameters: the results have no 'fullwave' section.  Add a \"fullwave\" section "
                         "(frequencies, or adaptive) to the design and solve it (docs/design_format.md)")
    notes: list[str] = []
    sweep = fw.get("sweep") or []
    rows = sorted(sweep, key=lambda r: r["f_Hz"])
    row_f = np.array([float(r["f_Hz"]) for r in rows])
    row_flag = np.array(["Re_Z_unresolved" in r for r in rows], dtype=bool)
    beyond: list[float] = []
    if dense:
        if surface_losses:
            raise ValueError("port_parameters: surface_losses=True needs the solved rows: the adaptive model does "
                             "not carry the surface losses (docs/physics/fullwave_surface_losses.md); pass "
                             "dense=False, or dense=True without surface_losses")
        from .circuit import fullwave_dense

        d = fullwave_dense(results)
        f = np.asarray(d["f_Hz"], dtype=float)
        Z = np.asarray(d["Z_matrix"], dtype=complex)
        err = np.asarray(d["estimated_relative_error"], dtype=float)
        ad = fw.get("adaptive") or {}
        tol = (ad.get("settings") or {}).get("tolerance", ad.get("tolerance", "n/a"))
        worst = float(np.nanmax(err)) if err.size and np.any(np.isfinite(err)) else float("nan")
        source = (f"adaptive model, dense curve (fullwave.adaptive.dense, {f.size} points; tolerance {tol}, "
                  f"converged {ad.get('converged')}, estimated relative error up to {worst:.3g})")
        if ad.get("converged") is False:
            warnings.warn(f"port_parameters: the adaptive sweep did not converge (stop_reason "
                          f"{ad.get('stop_reason')!r}): the dense curve is not certified to its tolerance",
                          RuntimeWarning, stacklevel=2)
        notes.append("Z is the adaptive rational model fitted to the solved rows (radiation, bulk and boundary losses "
                     "as solved); the perturbative conductor surface loss is NOT included (the model does not carry "
                     "it)")
        # a dense point is tied to the solves around it: flagged when either neighbouring solve is flagged
        flags = np.zeros(f.size, dtype=bool)
        if row_f.size:
            hi = np.clip(np.searchsorted(row_f, f), 0, row_f.size - 1)
            lo = np.clip(hi - 1, 0, row_f.size - 1)
            flags = row_flag[lo] | row_flag[hi]
        # the model's own estimated relative error (dZ ~ eps Z)
        zrel = np.where(np.isfinite(err), err, np.nan)
        zrel_source = "adaptive model's estimated relative error, taken as proportional (dZ = eps Z)"
        zrel_kind = "proportional"
        unresolved = [float(x) for x in row_f[row_flag]]
        if unresolved:
            notes.append(f"the model was fitted to {len(unresolved)} solved rows whose Re Z the solve's residual does "
                         f"not resolve (Re_Z_unresolved, {_hz_list(unresolved, 4)} Hz)")
    else:
        if not rows:
            raise ValueError("port_parameters: the results have no 'fullwave'/'sweep'.  Add a \"fullwave\" section "
                             "(frequencies, or adaptive) to the design and solve it (docs/design_format.md)")
        n = None
        mats = []
        has_loss = any("R_loss_ohm" in r or "R_loss_matrix_ohm" in r for r in rows)
        for r in rows:
            M = _row_matrix(r, n)
            n = M.shape[0]
            if surface_losses:
                add, ratio = _surface_loss_matrix(r, M)
                M = M + add
                if ratio > SURFACE_LOSS_PERTURBATION_LIMIT:
                    beyond.append(float(r["f_Hz"]))
            mats.append(M)
        shapes = {m.shape for m in mats}
        if len(shapes) != 1:
            raise ValueError(f"port_parameters: the rows have different port counts {sorted(shapes)}; re-solve the "
                             "design so that every row carries the full impedance matrix")
        f = row_f
        Z = np.array(mats)
        flags = row_flag
        errs = [_row_error(r, M) for r, M in zip(rows, mats)]
        zrel = np.array([e[0] for e in errs])
        zrel_source = "; ".join(sorted({e[1] for e in errs}))
        zrel_kind = "normwise"
        unresolved = [float(x) for x in row_f[row_flag]]
        source = f"solved rows (fullwave.sweep, {f.size} frequencies)"
        if surface_losses:
            notes.append("Z = the solved port impedance (radiation, bulk and boundary losses) PLUS the resistive part "
                         "of the perturbative conductor surface loss (R_loss); the surface reactance (internal or "
                         "kinetic inductance) is NOT included")
            if beyond:
                notes.append(f"R_loss/|X| > {SURFACE_LOSS_PERTURBATION_LIMIT} (perturbation not accurate) at "
                             f"{_hz_list(beyond, 4)} Hz")
                warnings.warn(f"port_parameters: R_loss/|X| exceeds {SURFACE_LOSS_PERTURBATION_LIMIT} at "
                              f"{len(beyond)} rows ({_hz_list(beyond)} Hz): the perturbative surface loss added there "
                              "is not accurate (near a self-resonance it grows like 1/distance^2 while the true "
                              "resistance stays bounded); add solves or export without surface_losses",
                              RuntimeWarning, stacklevel=2)
        else:
            notes.append("Z is the solved port impedance (radiation, bulk and boundary losses); the perturbative "
                         "conductor surface loss is NOT included"
                         + (" (R_loss_ohm in the rows; surface_losses=True adds it)" if has_loss else ""))
        if unresolved:
            notes.append(f"Re Z not resolved by the solve's residual at {len(unresolved)} rows (Re_Z_unresolved, "
                         f"{_hz_list(unresolved, 4)} Hz)")
    if f.size == 0:
        raise ValueError("port_parameters: no frequencies to export")
    bad = ~np.all(np.isfinite(Z.reshape(Z.shape[0], -1)), axis=1) | ~np.isfinite(f)
    if np.any(bad):
        raise ValueError(f"port_parameters: non-finite Z at {_hz_list(f[bad])} Hz; these frequencies have no usable "
                         "solve (see fullwave.adaptive.refused and the warnings)")
    if np.any(f <= 0):
        raise ValueError("port_parameters: frequencies must be positive")
    if np.any(np.diff(f) <= 0):
        dup = f[1:][np.diff(f) <= 0]
        raise ValueError(f"port_parameters: frequencies repeat ({_hz_list(dup)} Hz); Touchstone needs strictly "
                         "increasing frequencies: drop the duplicate rows")
    names, details = _port_info(results, Z.shape[-1])
    return {"f_Hz": f, "Z": np.conj(Z), "ports": names, "port_details": details, "source": source, "notes": notes,
            "Re_Z_unresolved_Hz": unresolved, "Re_Z_unresolved": np.asarray(flags, dtype=bool),
            "Z_relative_error": zrel, "Z_relative_error_source": zrel_source, "Z_relative_error_kind": zrel_kind,
            "surface_loss_beyond_perturbation_Hz": beyond}


def _surface_loss_matrix(r: Mapping[str, Any], M: np.ndarray) -> tuple[np.ndarray, float]:
    """The surface-loss term added to the engine-convention Z of one row (a real resistance is the same in both
    conventions) and the row's largest R_loss/|X|."""
    f = float(r["f_Hz"])
    n = M.shape[0]
    if n == 1:
        if r.get("R_loss_ohm") is None:
            raise ValueError(f"port_parameters: surface_losses=True but the row at {f:.9g} Hz has no R_loss_ohm "
                             "(no rf surface data there: see surface_losses.R_loss_omitted in the row); pass "
                             "surface_losses=False or give the conductors an rf model")
        R = float(r["R_loss_ohm"])
        X = abs(M[0, 0].imag)
        ratio = float(r.get("R_loss_over_abs_X", R / X if X > 0 else math.inf))
        return np.array([[R]], dtype=complex), ratio
    Rj = r.get("R_loss_matrix_ohm")
    if Rj is None:
        raise ValueError(f"port_parameters: surface_losses=True but the row at {f:.9g} Hz has no R_loss_matrix_ohm "
                         "(R_s missing for a surface); pass surface_losses=False")
    Rm = _matrix_of(Rj["re"], Rj["im"])
    scale = max(float(np.linalg.norm(Rm)), np.finfo(float).tiny)
    resistive = float(np.max(np.abs(M.real))) > 1e-9 * float(np.max(np.abs(M)))
    if float(np.linalg.norm(Rm.imag)) > 1e-6 * scale or resistive:
        raise ValueError(f"port_parameters: at {f:.9g} Hz the port fields are not in phase (R_loss_matrix_ohm is "
                         "complex, or Z has a resistive part from radiation or bulk loss), so R_loss_matrix_ohm is "
                         "not the impedance perturbation (docs/physics/fullwave_surface_losses.md); pass "
                         "surface_losses=False and account for the conductor loss separately")
    X = np.abs(np.diagonal(M).imag)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(X > 0, np.abs(np.diagonal(Rm).real) / X, np.inf)
    return Rm.real.astype(complex), float(np.max(ratios))


# ------------------------------------------------------------------ Touchstone writing
def _block(M: np.ndarray) -> list[str]:
    """The data values of one frequency's matrix, in Touchstone v1 order, as lines (without the frequency)."""
    n = M.shape[0]
    if n == 2:
        vals = [M[0, 0], M[1, 0], M[0, 1], M[1, 1]]
        return ["  ".join(f"{_fmt(v.real)} {_fmt(v.imag)}" for v in vals)]
    lines = []
    for j in range(n):
        row = list(M[j])
        for k in range(0, n, 4):
            lines.append("  ".join(f"{_fmt(v.real)} {_fmt(v.imag)}" for v in row[k:k + 4]))
    return lines


def _version() -> str:
    try:
        from ._core import version
        return str(version())
    except Exception:  # noqa: BLE001  (pure-Python use without the compiled module)
        return "unknown"


def write_touchstone_data(path: str | Path, f_Hz: Any, Z: Any, *, convention: str, kind: str = "S",
                          z0: Any = DEFAULT_Z0, ports: Sequence[str] | None = None,
                          port_details: Sequence[str] | None = None, comments: Sequence[str] = (),
                          passivity_tol: float = PASSIVITY_TOLERANCE, Re_Z_unresolved: Any = None,
                          Z_relative_error: Any = None, Z_relative_error_source: str = "",
                          Z_relative_error_kind: str = "normwise") -> dict[str, Any]:
    """Write an impedance-matrix sweep as a Touchstone v1.0 file (v1.1 option line for a per-port ``z0``).

    ``f_Hz`` (strictly increasing, Hz) and ``Z`` (``n_f x n x n``, ohm).  ``convention`` (required): the time
    convention of ``Z``, ``"axionfem"`` (the engine's ``e^{-i omega t}``, conjugated here) or ``"engineering"``
    (``e^{+j omega t}``, :func:`port_parameters` returns that).  ``kind``: ``"S"``, ``"Z"`` or ``"Y"``.  ``z0``: the
    reference resistance (ohm); one value (v1.0; Z and Y are written normalised by it, ``Z/R`` and ``Y R``) or, for
    S only, one per port (Version 1.1 option line ``R r1 ... rn``, per-port power waves).  ``path`` must end in
    ``.sNp`` with ``N`` the port count; an existing file is overwritten (``overwrote`` in the report).
    ``Re_Z_unresolved`` (bool per frequency) lets the report separate passivity violations at rows whose ``Re Z`` is
    unresolved.  ``Z_relative_error`` (per frequency, the estimated normwise relative error of Z, NaN where unknown;
    :func:`port_parameters`) is reported for every kind, and for ``kind="Y"`` it decides whether the inversion
    amplifies an otherwise small error: relative error x equilibrated condition number > 0.1 while the relative error
    itself is below 0.1 (``Z_relative_error_kind="normwise"``; for ``"proportional"``, dZ = eps Z, the inversion does
    not amplify it: dY = -eps Y).  Where Z's own estimated error is 0.1 or more, Z itself is poorly determined and Y
    is not flagged separately: the report lists those frequencies in ``Z_poorly_determined_Hz`` for every kind (for a
    one-port, Y's relative error equals Z's).

    Returns a report: ``path``, ``overwrote``, ``kind``, ``z0_ohm``, ``ports``, ``n_frequencies``, ``f_min_Hz``,
    ``f_max_Hz``, ``sigma_max`` (the largest singular value of S over the band), ``passivity_violations_Hz``
    (``sigma_max > 1 + passivity_tol``), ``passivity_violations_at_unresolved_Re_Z_Hz`` (those of them at rows whose
    ``Re Z`` the solve does not resolve), ``reciprocity_error`` (the largest ``||Z - Z^T|| / ||Z||``),
    ``Z_relative_error_max`` (NaN: no estimate), ``Z_poorly_determined_Hz``, ``Y_check`` (what the Y check rested on,
    or "no error estimate: Y not checked"), ``Y_ill_determined_Hz`` and ``warnings`` (every warning raised, also
    emitted as a Python warning).
    """
    if kind not in KINDS:
        raise ValueError(f"write_touchstone: kind must be one of {KINDS}, not {kind!r}")
    f = np.asarray(f_Hz, dtype=float).ravel()
    Ze = _square(to_engineering(Z, convention), "write_touchstone")
    if Ze.ndim != 3 or Ze.shape[0] != f.size:
        raise ValueError(f"write_touchstone: Z has shape {Ze.shape} for {f.size} frequencies; expected "
                         f"({f.size}, n, n)")
    if np.any(~np.isfinite(f)) or np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError("write_touchstone: frequencies must be finite, non-negative and strictly increasing "
                         "(sort them and drop duplicates; port_parameters does)")
    n = Ze.shape[-1]
    z0v = _z0_vector(z0, n, "write_touchstone")
    per_port = bool(np.any(z0v != z0v[0]))   # equal values collapse to the v1.0 single R
    if per_port and kind != "S":
        raise ValueError(f"write_touchstone: per-port reference resistances {z0v.tolist()} are written only for "
                         "S-parameters (Touchstone v1.1 'R r1 ... rn'): readers do not agree on how Z and Y are "
                         "normalised with them.  Z and Y do not depend on z0: export them with one z0, or export S "
                         "with per-port z0")
    p = Path(path)
    if p.suffix.lower() != f".s{n}p":
        raise ValueError(f"write_touchstone: {p.name!r} must end in '.s{n}p' for {n} port(s): Touchstone readers "
                         f"take the port count from the extension.  Use e.g. {p.with_suffix(f'.s{n}p').name!r}")
    names = ([_clean(x).replace("|", "/").strip() or f"port_{k + 1}" for k, x in enumerate(ports)]
             if ports is not None else [f"port_{k + 1}" for k in range(n)])
    if len(names) != n:
        raise ValueError(f"write_touchstone: {len(names)} port names for {n} ports")
    details = [_clean(x) for x in port_details] if port_details is not None else [""] * n
    if len(details) != n:
        raise ValueError(f"write_touchstone: {len(details)} port details for {n} ports")
    flags = np.zeros(f.size, dtype=bool) if Re_Z_unresolved is None else np.asarray(Re_Z_unresolved, dtype=bool)
    warn: list[str] = []

    def note(msg: str) -> None:
        warn.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    S = z_to_s(Ze, z0v, convention="engineering")
    pas = passivity(S, passivity_tol)
    recip = float(np.max(reciprocity_error(Ze)))
    viol = pas["violations"]
    at_unresolved = [float(x) for x in f[viol][flags[viol]]]
    if viol.size:
        bad = f[viol]
        msg = (f"write_touchstone: the data are not passive at {bad.size} of {f.size} frequencies "
               f"({_hz_list(bad)} Hz): sigma_max(S) - 1 up to {float(np.max(pas['excess'])):.3g} > "
               f"{passivity_tol:.1e}. Written as computed (not clipped). ")
        if at_unresolved:
            msg += (f"{len(at_unresolved)} of them are at rows whose Re Z the solve's residual does not resolve "
                    f"(Re_Z_unresolved: {_hz_list(at_unresolved)} Hz), where the excess can be solver-level (a "
                    "tighter AXF_FW_RTOL resolves it); ")
        rest = bad.size - len(at_unresolved)
        if rest:
            msg += (f"{rest} are at resolved rows: check the mesh (h/p convergence) and the solver residuals "
                    "(resistance_min_eigenvalue_ohm)")
        note(msg.rstrip("; "))
    rel = np.full(f.size, np.nan) if Z_relative_error is None else np.asarray(Z_relative_error, dtype=float)
    known = np.isfinite(rel)
    rel_max = float(np.max(rel[known])) if np.any(known) else float("nan")
    poor = [float(x) for x in f[known & (np.nan_to_num(rel, nan=0.0) >= Y_RELATIVE_ERROR_LIMIT)]]
    y_bad: list[float] = []
    y_check = "not a Y export"
    if kind == "Y":
        cond = np.atleast_1d(equilibrated_condition(Ze))
        sig_min = np.linalg.svd(Ze, compute_uv=False)[..., -1]
        if np.any(sig_min == 0):
            raise ValueError(f"write_touchstone: Z is exactly singular at {_hz_list(f[sig_min == 0])} Hz, so Y does "
                             "not exist there; export S or Z, or drop those frequencies")
        r0 = np.nan_to_num(rel, nan=0.0)
        if Z_relative_error_kind not in ("normwise", "proportional"):
            raise ValueError(f"write_touchstone: Z_relative_error_kind must be 'normwise' or 'proportional', not "
                             f"{Z_relative_error_kind!r}")
        gain = cond if Z_relative_error_kind == "normwise" else np.ones_like(cond)
        amplified = known & (r0 < Y_RELATIVE_ERROR_LIMIT) & (r0 * gain > Y_RELATIVE_ERROR_LIMIT)
        mask = (cond > Y_CONDITION_LIMIT) | amplified
        y_bad = [float(x) for x in f[mask]]
        if not np.any(known):
            y_check = (f"no error estimate: Y not checked against Z's error (only the equilibrated condition number "
                       f"against {Y_CONDITION_LIMIT:.0e})")
        else:
            how = "x equilibrated condition number" if Z_relative_error_kind == "normwise" else "(not amplified)"
            missing = "" if np.all(known) else f"; {int((~known).sum())} frequencies without an estimate"
            y_check = (f"relative error of Z ({Z_relative_error_source or 'given'}) {how} "
                       f"against {Y_RELATIVE_ERROR_LIMIT}{missing}")
        if y_bad:
            note(f"write_touchstone: Y = Z^-1 is less well determined than Z at {len(y_bad)} frequencies "
                 f"({_hz_list(y_bad)} Hz): the inversion amplifies Z's estimated relative error past "
                 f"{Y_RELATIVE_ERROR_LIMIT} (relative error x equilibrated condition number), or the ports are "
                 f"nearly dependent (equilibrated condition number above {Y_CONDITION_LIMIT:.0e}); export S or Z")
        data = np.linalg.inv(Ze) * z0v[0]
    elif kind == "Z":
        data = Ze / z0v[0]
    else:
        data = S
    version = "v1.1 (per-port reference resistances)" if per_port else "v1.0"
    r_text = " ".join(_fmt(x) for x in z0v) if per_port else _fmt(z0v[0])
    lines = [f"! axionFEM {_version()}: Touchstone {version} export of full-wave port results",
             "! time convention exp(+j omega t): the engine's exp(-i omega t) impedances are complex-conjugated "
             "(an inductor is Z = +j omega L here)",
             f"! parameters {kind}, reference resistance {r_text} ohm"
             + (" (port by port)" if per_port else " on every port")
             + ("; stored normalised (Z/R), as Touchstone v1 requires" if kind == "Z" else
                "; stored normalised (Y*R), as Touchstone v1 requires" if kind == "Y" else "")]
    lines += [f"! {_clean(c)}" for c in comments]
    lines.append(f"! sigma_max(S) over the band {float(np.max(pas['sigma_max'])):.12g}; reciprocity "
                 f"||Z - Z^T||/||Z|| up to {recip:.3g}")
    if np.any(known):
        lines.append(f"! estimated relative error of Z up to {rel_max:.3g} ({Z_relative_error_source or 'given'})"
                     + (f"; 0.1 or more at {len(poor)} frequencies ({_hz_list(poor, 4)} Hz)" if poor else ""))
    else:
        lines.append("! no error estimate of Z is available" + ("; Y not checked against it" if kind == "Y" else ""))
    lines.append("! port order (polarity: the engine's port orientation; a reversed port flips the sign of its "
                 "off-diagonal entries):")
    for k, (nm, de) in enumerate(zip(names, details)):
        lines.append(f"!   port {k + 1}: {nm}" + (f" | {de}" if de else ""))
    lines.append(f"# Hz {kind} RI R {r_text}")
    for fi, M in zip(f, data):
        block = _block(M)
        lines.append(f"{_fmt(fi)}  {block[0]}")
        lines += [f"  {b}" for b in block[1:]]
    overwrote = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {"path": str(p), "overwrote": overwrote, "kind": kind,
            "z0_ohm": z0v.tolist() if per_port else float(z0v[0]), "ports": names,
            "n_frequencies": int(f.size), "f_min_Hz": float(f[0]), "f_max_Hz": float(f[-1]),
            "sigma_max": float(np.max(pas["sigma_max"])),
            "passivity_violations_Hz": [float(x) for x in f[viol]],
            "passivity_violations_at_unresolved_Re_Z_Hz": at_unresolved,
            "reciprocity_error": recip, "Z_relative_error_max": rel_max, "Z_poorly_determined_Hz": poor,
            "Y_check": y_check, "Y_ill_determined_Hz": y_bad, "warnings": warn}


def write_touchstone(results: Mapping[str, Any], path: str | Path, kind: str = "S", z0: Any = DEFAULT_Z0,
                     dense: bool = False, *, surface_losses: bool = False,
                     passivity_tol: float = PASSIVITY_TOLERANCE) -> dict[str, Any]:
    """Write the port parameters of a full-wave sweep as a Touchstone file (``.sNp``).

    ``results``: a results document (``Results.data`` or ``results.json``).  ``kind``: ``"S"`` (default), ``"Z"``
    or ``"Y"``; ``z0``: reference resistance (ohm; one for every port, or for S one per port, written as a v1.1
    option line); ``dense``: the adaptive model's dense curve instead of the solved rows; ``surface_losses``: add the
    resistive part of the perturbative conductor surface loss (:func:`port_parameters`).  The values are in the
    engineering convention ``e^{+j omega t}`` (the engine's Z is conjugated), sorted by frequency, with a comment
    block giving the source, the port order and polarity, and the checks.  Passivity violations
    (``sigma_max(S) > 1 + passivity_tol``) are warned about and written as computed.  Returns the report of
    :func:`write_touchstone_data` plus ``source``, ``Re_Z_unresolved_Hz`` and
    ``surface_loss_beyond_perturbation_Hz``; its ``warnings`` holds every warning raised on the way.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net = port_parameters(results, dense=dense, surface_losses=surface_losses)
    early = [str(w.message) for w in caught]
    for msg in early:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    design = results.get("design")
    comments = []
    if isinstance(design, Mapping) and design.get("name"):
        comments.append(f"design {design['name']}")
    comments.append(f"source: {net['source']}")
    comments += net["notes"]
    out = write_touchstone_data(path, net["f_Hz"], net["Z"], convention="engineering", kind=kind, z0=z0,
                                ports=net["ports"], port_details=net["port_details"], comments=comments,
                                passivity_tol=passivity_tol, Re_Z_unresolved=net["Re_Z_unresolved"],
                                Z_relative_error=net["Z_relative_error"],
                                Z_relative_error_source=net["Z_relative_error_source"],
                                Z_relative_error_kind=net["Z_relative_error_kind"])
    out["warnings"] = early + out["warnings"]
    out["source"] = net["source"]
    out["Re_Z_unresolved_Hz"] = net["Re_Z_unresolved_Hz"]
    out["surface_loss_beyond_perturbation_Hz"] = net["surface_loss_beyond_perturbation_Hz"]
    return out


# ------------------------------------------------------------------ Touchstone reading
def _per_frequency(fn, stack: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """``fn`` applied to the whole stack; where that fails as singular, matrix by matrix, with NaN (and the index
    listed) where a matrix fails."""
    try:
        return np.asarray(fn(stack), dtype=complex), []
    except (ValueError, np.linalg.LinAlgError):
        pass
    out = np.full(stack.shape, np.nan + 1j * np.nan, dtype=complex)
    bad = []
    for i, M in enumerate(stack):
        try:
            out[i] = fn(M)
        except (ValueError, np.linalg.LinAlgError):
            bad.append(i)
    return out, bad


def read_touchstone(path: str | Path, nports: int | None = None) -> dict[str, Any]:
    """Read a Touchstone v1.0/v1.1 file (``.sNp``) with S-, Z- or Y-parameters.

    ``nports`` defaults to the ``N`` of the extension.  Handles the frequency units (Hz, kHz, MHz, GHz), the formats
    RI, MA and DB, the v1 defaults (``# GHz S MA R 50``), values spread over any number of lines, the 2-port order
    ``N11 N21 N12 N22``, a v1.1 per-port ``R r1 ... rn`` (S-parameters only) and a 2-port file's trailing noise
    parameters (checked and ignored; ``noise_rows_ignored``).  Refuses data before the option line, Touchstone v2
    keywords and H/G parameters.

    Returns ``f_Hz``, ``kind``, ``z0_ohm`` (a float, or a list per port), ``data`` (``n_f x n x n`` as stored but
    de-normalised: Z in ohm, Y in siemens), ``S`` and ``Z`` (both engineering convention, ``e^{+j omega t}``),
    ``ports`` (names from an axionFEM port-order comment, else ``port_k``), ``comments``, ``option_line`` and
    ``noise_rows_ignored``.  ``S`` is always present (stored S, or converted where it exists); ``Z`` (and ``S`` of a
    Y file) is NaN at the frequencies where it does not exist (an open, an ideal thru, a singular Y), with a warning
    naming the file and the frequencies (``Z_undefined_Hz``, ``S_undefined_Hz``).
    """
    p = Path(path)
    if nports is None:
        m = re.fullmatch(r"\.s(\d+)p", p.suffix.lower())
        if not m:
            raise ValueError(f"read_touchstone: cannot take the port count from {p.name!r} (expected a '.sNp' "
                             "extension); pass nports=")
        nports = int(m.group(1))
    n = int(nports)
    if n < 1:
        raise ValueError(f"read_touchstone: nports must be at least 1, not {n}")
    unit, kind, fmt, z0 = "GHZ", "S", "MA", [50.0]
    option = None
    comments: list[str] = []
    names: dict[int, str] = {}
    tokens: list[float] = []
    # Touchstone is US-ASCII with CR/LF line ends: decode bytes as Latin-1 (never fails) and split on CR/LF only,
    # not on Unicode line separators
    for lineno, raw in enumerate(re.split(r"\r\n|\r|\n", p.read_bytes().decode("latin-1")), 1):
        text, sep, comment = raw.partition("!")
        if sep:
            comments.append(comment.strip())
            m = re.match(r"\s*port (\d+): ([^|]+)", comment)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
        text = text.strip()
        if not text:
            continue
        if text.startswith("["):
            raise ValueError(f"read_touchstone: {p.name}:{lineno} has the keyword {text.split()[0]!r}: this is a "
                             "Touchstone v2 file, which this reader does not handle")
        if text.startswith("#"):
            if option is not None:
                continue   # the specification: only the first option line counts
            if tokens:
                raise ValueError(f"read_touchstone: {p.name}:{lineno}: the option line comes after data lines; "
                                 "Touchstone v1 requires it before any data (move it to the top)")
            option = text
            words = text[1:].upper().split()
            i = 0
            while i < len(words):
                w = words[i]
                if w in _UNITS:
                    unit = w
                elif w in ("S", "Y", "Z", "H", "G"):
                    kind = w
                elif w in ("RI", "MA", "DB"):
                    fmt = w
                elif w == "R":
                    vals = []
                    while i + 1 < len(words):
                        try:
                            vals.append(float(words[i + 1]))
                        except ValueError:
                            break
                        i += 1
                    if not vals:
                        raise ValueError(f"read_touchstone: {p.name}:{lineno}: 'R' without a value in {text!r}")
                    z0 = vals
                else:
                    raise ValueError(f"read_touchstone: {p.name}:{lineno}: unknown option {w!r} in {text!r}")
                i += 1
            continue
        words = text.split()
        bad_words = [w for w in words if not _NUMBER.fullmatch(w)]
        if bad_words:
            raise ValueError(f"read_touchstone: {p.name}:{lineno}: {bad_words[0]!r} is not a decimal or scientific "
                             f"number in {text!r}")
        values = [float(w) for w in words]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"read_touchstone: {p.name}:{lineno}: a value overflows to infinity in {text!r}")
        tokens += values
    if kind not in KINDS:
        raise ValueError(f"read_touchstone: {kind}-parameters are not supported (S, Z and Y are)")
    if option is None:
        warnings.warn(f"read_touchstone: {p.name} has no option line ('# ...'), which Touchstone requires: read with "
                      "the v1 defaults # GHz S MA R 50; check the units", RuntimeWarning, stacklevel=2)
    if len(z0) not in (1, n):
        raise ValueError(f"read_touchstone: {p.name}: the option line gives {len(z0)} reference resistances for {n} "
                         "ports; give one, or one per port (Version 1.1)")
    per_port = len(z0) == n and n > 1 and len(set(z0)) > 1
    if per_port and kind != "S":
        raise ValueError(f"read_touchstone: {p.name}: Touchstone v1.1 per-port reference resistances with "
                         f"{kind}-parameters are not supported (readers do not agree on their normalisation); "
                         "S-parameters with per-port R are")
    z0v = _z0_vector(z0 if per_port else z0[0], n, "read_touchstone")
    scale = _UNITS[unit]
    per = 1 + 2 * n * n
    freqs, mats = [], []
    i = 0
    noise_rows = 0
    while i < len(tokens):
        if n == 2 and freqs and tokens[i] * scale <= freqs[-1]:
            noise = tokens[i:]
            if len(noise) % 5:
                raise ValueError(f"read_touchstone: {p.name}: a frequency {tokens[i]:.9g} at or below the previous "
                                 f"one starts the noise block, but the {len(noise)} numbers after it are not rows "
                                 "of 5 (f, NFmin, |Gamma_opt|, angle, Rn/R); is a frequency repeated in the network "
                                 "data?")
            nf = np.array(noise[0::5]) * scale
            if np.any(np.diff(nf) <= 0):
                raise ValueError(f"read_touchstone: {p.name}: the noise block's frequencies are not increasing; is a "
                                 "frequency repeated or out of order in the network data?")
            noise_rows = len(nf)
            warnings.warn(f"read_touchstone: {p.name}: ignored {noise_rows} noise-parameter rows "
                          f"({nf[0]:.9g} to {nf[-1]:.9g} Hz)", RuntimeWarning, stacklevel=2)
            break
        if i + per > len(tokens):
            raise ValueError(f"read_touchstone: {p.name}: {len(tokens) - i} numbers left after "
                             f"{len(freqs)} frequencies, fewer than the {per} of a {n}-port block; is nports right?")
        block = np.array(tokens[i + 1:i + per])
        a, b = block[0::2], block[1::2]
        if fmt == "RI":
            v = a + 1j * b
        elif fmt == "MA":
            v = a * np.exp(1j * np.deg2rad(b))
        else:
            v = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
        M = v.reshape(n, n).T if n == 2 else v.reshape(n, n)
        freqs.append(tokens[i] * scale)
        mats.append(M)
        i += per
    if not freqs:
        raise ValueError(f"read_touchstone: {p.name} has no data")
    f = np.array(freqs)
    if np.any(np.diff(f) <= 0):
        raise ValueError(f"read_touchstone: {p.name}: frequencies are not strictly increasing")
    data = np.array(mats)
    z0_out: Any = z0v.tolist() if per_port else float(z0v[0])
    s_bad: list[int] = []
    if kind == "S":
        S = data
        Z, z_bad = _per_frequency(lambda M: s_to_z(M, z0v), S)
    elif kind == "Z":
        data = data * z0v[0]
        Z = data
        z_bad = []
        S, s_bad = _per_frequency(lambda M: z_to_s(M, z0v, convention="engineering"), Z)
    else:
        data = data / z0v[0]
        Z, z_bad = _per_frequency(y_to_z, data)
        eye = np.eye(n)
        # S = (1 - y)(1 + y)^-1 with y = Y z0: exists wherever 1 + y is regular, Z or not
        S, s_bad = _per_frequency(lambda Y: _right_divide(eye - Y * z0v[0], eye + Y * z0v[0], "read_touchstone",
                                                          "1 + Y z0"), data)
    if z_bad:
        warnings.warn(f"read_touchstone: {p.name}: Z does not exist at {len(z_bad)} frequencies "
                      f"({_hz_list(f[z_bad])} Hz; an open, an ideal thru or a singular Y): Z is NaN there, S and the "
                      "stored data are complete", RuntimeWarning, stacklevel=2)
    if s_bad:
        warnings.warn(f"read_touchstone: {p.name}: S does not exist at {len(s_bad)} frequencies "
                      f"({_hz_list(f[s_bad])} Hz; Z = -z0): S is NaN there", RuntimeWarning, stacklevel=2)
    return {"f_Hz": f, "kind": kind, "z0_ohm": z0_out, "data": data, "S": S, "Z": Z,
            "ports": [names.get(k + 1, f"port_{k + 1}") for k in range(n)], "comments": comments,
            "option_line": option or "# GHz S MA R 50 (v1 default; the file has no option line)",
            "noise_rows_ignored": noise_rows, "Z_undefined_Hz": [float(x) for x in f[z_bad]],
            "S_undefined_Hz": [float(x) for x in f[s_bad]]}


# ------------------------------------------------------------------ command line: axf export-touchstone
def main(argv: Sequence[str] | None = None) -> int:
    """``axf export-touchstone results.json [-o OUT.sNp] [--kind S|Z|Y] [--z0 50] [--dense] [--surface-losses]``.

    Exit status: 0 written and passive; 3 written but not passive (sigma_max(S) > 1 + 1e-9 somewhere); 2 nothing
    written (an error, reported on stderr; argparse's usage errors are 2 as well)."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="axf export-touchstone",
                                 description="Write a solved design's full-wave port parameters as Touchstone v1 "
                                             "(exp(+j omega t); docs/physics/touchstone.md).  Exit status: 0 "
                                             "written, 3 written but not passive, 2 error")
    ap.add_argument("results", help="results.json of a solved design with a fullwave section")
    ap.add_argument("-o", "--output", help="output file (default: <results dir>/<design>[_dense]_<kind>.s<N>p)")
    ap.add_argument("--kind", default="S", choices=KINDS)
    ap.add_argument("--z0", type=float, default=DEFAULT_Z0, help="reference resistance, ohm (default 50)")
    ap.add_argument("--dense", action="store_true", help="the adaptive model's dense curve, not the solved rows")
    ap.add_argument("--surface-losses", action="store_true",
                    help="add the resistive part of the perturbative conductor surface loss")
    a = ap.parse_args(list(argv) if argv is not None else None)
    rp = Path(a.results)
    try:
        results = json.loads(rp.read_text())
        if not isinstance(results, Mapping):
            raise ValueError(f"{rp} is not a results document (a JSON object with a 'fullwave' section)")
        out = a.output
        if out is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net = port_parameters(results, dense=a.dense, surface_losses=a.surface_losses)
            design = results.get("design")
            stem = safe_stem(design.get("name") if isinstance(design, Mapping) else None)
            out = rp.parent / f"{stem}{'_dense' if a.dense else ''}_{a.kind}.s{net['Z'].shape[-1]}p"
        rep = write_touchstone(results, out, a.kind, a.z0, a.dense, surface_losses=a.surface_losses)
    except Exception as e:  # noqa: BLE001  (every failure is exit 2, never Python's 1)
        print(f"axf export-touchstone: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return EXIT_ERROR
    print(f"wrote {rep['path']}{' (overwrote)' if rep['overwrote'] else ''}: {rep['n_frequencies']} frequencies "
          f"{rep['f_min_Hz']:.6g}-{rep['f_max_Hz']:.6g} Hz, {len(rep['ports'])} port(s) {rep['ports']}, "
          f"sigma_max(S) {rep['sigma_max']:.12g}, reciprocity {rep['reciprocity_error']:.3g}")
    return EXIT_NOT_PASSIVE if rep["passivity_violations_Hz"] else EXIT_OK
