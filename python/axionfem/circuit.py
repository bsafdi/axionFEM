"""Linear readout-circuit networks for axion pickups (tuning capacitor, SQUID input
coil, leads, transformers) and their reduction to the optimal-scan inputs.

Everything here is a *linear* frequency-domain calculation, in the package's full-wave
time convention ``e^{-i omega t}`` (the same one the C++ full-wave engine uses, where
the readout impedance is reported as ``Z = R - i omega L``):

====================  ================================  =============================
element               impedance                         note
====================  ================================  =============================
resistor              ``Z = R``                         Johnson noise at ``T``
inductor              ``Z = -i omega L``                ``V = L dI/dt``
capacitor             ``Z = i/(omega C)``               ``tan_delta`` adds loss
mutual inductance     ``Z_{jk} = -i omega M_{jk}``      signed; or give ``k``
generic impedance     ``Z = Z(omega)``                  callable or interpolated table
ideal transformer     ``V_2 = n V_1``, ``I_1 = -n I_2``  lossless, no leakage
voltage/current src.  ``V(p)-V(m) = Z I - emf``         ``Z = 0``: ideal
====================  ================================  =============================

The pickup is one branch: a series EMF ``V = -i omega kappa_a Psi`` with the pickup
impedance ``Z_p(omega)``, where ``kappa_a = g_agg sqrt(2 rho_DM)`` in the package's
conventions, so that ``Phi_a = kappa_a Psi`` is the flux *amplitude*
(``docs/physics/scan_and_signal.md`` Sec. 1.3).  ``Z_p`` is either the MQS form
``R_loss - i omega L_p`` or the full-wave ``Z(omega)`` interpolated from a solved
design's frequency sweep (:func:`pickup_impedance`).

PSDs use the Foster, Rodd & Safdi (1711.10489) normalisation that the rest of the
package uses, ``<X^2> = (1/pi) int_0^inf domega S_X(omega)``, i.e. *half* the usual
one-sided-in-nu PSD.  Johnson noise of a dissipative element at temperature ``T`` is
therefore ``S_V = Re Z(omega) hbar omega (2 n(omega, T) + 1)`` (``-> 2 k_B T R``
classically), spec Eq. (S3.3).

See ``docs/physics/readout_circuit.md`` for the formulas, the derivation of the
equivalent single-pole detector and the mapping onto the optimal-scan inputs
``(Q, T, eta_A)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# ----------------------------------------------------------------- constants (CODATA 2018)
HBAR = 1.054571817e-34        # J s
KB = 1.380649e-23             # J/K
MU0 = 1.25663706212e-6        # H/m  (same value as the C++ core)
PHI0 = 2.067833848e-15        # Wb, h/2e (magnetic flux quantum)
HBAR_C_eV_m = 1.973269804e-7  # eV m
EV_J = 1.602176634e-19        # J/eV
GEV_J = 1.602176634e-10       # J/GeV
RHO_GeV_cm3_TO_J_m3 = GEV_J * 1e6   # 1 GeV/cm^3 in J/m^3

ZFunc = Callable[[float], complex]


# ----------------------------------------------------------------- axion coupling, occupation
def kappa_a_SI(g_GeV: float, rho_GeV_cm3: float = 0.3) -> float:
    """``kappa_a`` in H^-1, so that the flux amplitude is ``Phi_a [Wb] = kappa_a Psi_SI``
    with ``Psi_SI = mu0 Psi`` [T H m^2]  (spec S2.2-S2.3).

    ``kappa_a^SI = g_SI sqrt(2 rho_SI hbar c)/mu0``, ``g_SI [J^-1] = g [GeV^-1]/(1.602e-10 J/GeV)``.
    Check: ``g = 1e-19 GeV^-1``, ``rho = 0.3 GeV/cm^3`` -> ``8.65882567e-19`` H^-1.
    """
    g_SI = g_GeV / GEV_J
    rho_SI = rho_GeV_cm3 * RHO_GeV_cm3_TO_J_m3
    return g_SI * math.sqrt(2.0 * rho_SI * HBAR_C_eV_m * EV_J) / MU0


def kappa_a_inv_m(g_GeV: float, rho_GeV_cm3: float = 0.3) -> float:
    """``kappa_a`` in m^-1, so that ``Phi_a [Wb] = kappa_a Psi [T m^3]`` (natural-HL ``Psi``)."""
    return MU0 * kappa_a_SI(g_GeV, rho_GeV_cm3)


def n_bose(omega: float, T_K: float | None) -> float:
    """Bose-Einstein occupation ``1/expm1(hbar omega/k_B T)``; 0 for ``T <= 0``  (spec S3.1)."""
    if T_K is None or T_K <= 0.0 or omega <= 0.0:
        return 0.0
    x = HBAR * omega / (KB * T_K)
    if x > 700.0:
        return 0.0
    return 1.0 / math.expm1(x)


def T_from_occupation(omega: float, n: float) -> float:
    """Inverse of :func:`n_bose`: the temperature whose occupation at ``omega`` is ``n``."""
    if n <= 0.0 or omega <= 0.0:
        return 0.0
    return HBAR * omega / (KB * math.log1p(1.0 / n))


def xi_opt(n: float, eta_A: float) -> float:
    """``xi_opt = 2 eta^2/(2n+1+sqrt((2n+1)^2+8 eta^2))``  (spec S3.9; [C] F54)."""
    s = 2.0 * n + 1.0
    return 2.0 * eta_A ** 2 / (s + math.sqrt(s * s + 8.0 * eta_A ** 2))


def A_noise(xi: float, n: float, eta_A: float) -> float:
    """``A = xi^2 + 2(2n+1) xi + eta^2``  (spec S3.7)."""
    return xi * xi + 2.0 * (2.0 * n + 1.0) * xi + eta_A * eta_A


def G_bar(xi: float, n: float, eta_A: float) -> float:
    """``G-bar = xi/A^{3/2}``  (spec S3.8; [P] eq:Gbar; [C] F50, F53)."""
    return xi / A_noise(xi, n, eta_A) ** 1.5


# ----------------------------------------------------------------- impedance helpers
def _as_zfunc(value: Any, kind: str) -> ZFunc:
    """Normalise an element value into a callable ``Z(omega)``."""
    if callable(value):
        if kind == "R":
            return lambda w: complex(value(w), 0.0)
        if kind == "L":
            return lambda w: complex(0.0, -w * value(w))
        if kind == "C":
            return lambda w: complex(0.0, 1.0 / (w * value(w)))
    v = float(value)
    if kind == "R":
        return lambda w, v=v: complex(v, 0.0)
    if kind == "L":
        return lambda w, v=v: complex(0.0, -w * v)
    if kind == "C":
        return lambda w, v=v: complex(0.0, 1.0 / (w * v))
    raise ValueError(f"unknown element kind {kind!r}")


def impedance_table(f_Hz: Sequence[float], Z: Sequence[complex] | None = None, *,
                    R_ohm: Sequence[float] | None = None, L_H: Sequence[float] | None = None,
                    extrapolate: str = "linear") -> ZFunc:
    """``Z(omega)`` interpolated from a table.

    Give either complex ``Z`` or the pair (``R_ohm``, ``L_H``).  The interpolation is on
    ``R(f)`` and ``L(f) = -Im Z/omega``, both smooth and nearly constant in the MQS limit,
    rather than on ``Im Z`` itself: the MQS behaviour ``Z -> -i omega L`` is then exact
    between the samples and down to ``omega -> 0``.  (The C++ full-wave sweep reports
    exactly these two columns, ``Z_re_ohm`` and ``L_H``.)

    ``R`` is interpolated and extrapolated in **log-log** whenever every sample is positive,
    because a surface or radiation resistance is a power law in ``f`` (``R_s ~ f^2`` for a
    superconductor, ``f^{1/2}`` for a normal metal, ``f^4`` for dipole radiation); a table with a
    zero or negative entry falls back to linear-in-``log f`` with a clip at 0.  ``L`` is always
    linear in ``log f``: it changes sign at a self-resonance.

    ``extrapolate``: ``"linear"`` (default; continue the end slope), ``"const"`` (hold the end
    values) or ``"error"``.
    """
    f = np.asarray(f_Hz, dtype=float)
    if f.ndim != 1 or f.size < 1:
        raise ValueError("impedance_table: f_Hz must be a non-empty 1-D sequence")
    if np.any(f <= 0):
        raise ValueError("impedance_table: frequencies must be positive")
    order = np.argsort(f)
    f = f[order]
    if Z is not None:
        if R_ohm is not None or L_H is not None:
            raise ValueError("impedance_table: give either Z or (R_ohm, L_H), not both")
        Zc = np.asarray(Z, dtype=complex)[order]
        R, L = Zc.real.copy(), -Zc.imag / (2.0 * math.pi * f)
    else:
        if R_ohm is None or L_H is None:
            raise ValueError("impedance_table: give either Z or both R_ohm and L_H")
        R = np.asarray(R_ohm, dtype=float)[order]
        L = np.asarray(L_H, dtype=float)[order]
    if R.shape != f.shape or L.shape != f.shape:
        raise ValueError("impedance_table: value arrays must match f_Hz in length")
    if extrapolate not in ("linear", "const", "error"):
        raise ValueError("impedance_table: extrapolate must be 'linear', 'const' or 'error'")
    lf = np.log(f)

    def interp(lx: float, y: np.ndarray) -> float:
        if y.size == 1:
            return float(y[0])
        if lx < lf[0] or lx > lf[-1]:
            if extrapolate == "error":
                raise ValueError(
                    f"impedance_table: f = {math.exp(lx):.6g} Hz is outside the tabulated range "
                    f"[{f[0]:.6g}, {f[-1]:.6g}] Hz; widen the sweep or pass "
                    "extrapolate='linear' or 'const'")
            if extrapolate == "const":
                return float(y[0] if lx < lf[0] else y[-1])
            i = 0 if lx < lf[0] else y.size - 2
        else:
            i = int(np.clip(np.searchsorted(lf, lx) - 1, 0, y.size - 2))
        t = (lx - lf[i]) / (lf[i + 1] - lf[i])
        return float(y[i] + t * (y[i + 1] - y[i]))

    log_R = extrapolate == "linear" and R.size >= 2 and bool(np.all(R > 0))
    logR = np.log(R) if log_R else R

    def Zof(w: float) -> complex:
        lx = math.log(w / (2.0 * math.pi))
        # R is a power law in f (R_s ~ f^2 for a superconductor, f^{1/2} for a normal metal,
        # f^4 for dipole radiation), so interpolate and extrapolate it in log-log whenever the
        # table allows: linear-in-log-f extrapolation of R undershoots and has to be clipped at
        # zero.  A table with a zero or negative entry falls back to the linear form with the clip.
        if log_R:
            Rv = math.exp(interp(lx, logR))
        elif extrapolate == "linear":
            Rv = max(interp(lx, R), 0.0)
        else:
            Rv = interp(lx, R)
        return complex(Rv, -w * interp(lx, L))

    Zof.f_Hz, Zof.R_ohm, Zof.L_H = f, R, L   # type: ignore[attr-defined]
    return Zof


def coupling_table(f_Hz: Sequence[float], Psi_SI: Sequence[float], *,
                   extrapolate: str = "const") -> Callable[[float], float]:
    """``Psi_SI(omega)`` [T H m^2] interpolated from a table, log-log in ``f``.

    The signal coupling of a full-wave sweep grows as ``1 + O((kR)^2)`` and then diverges
    towards a self-resonance, so ``log |Psi|`` versus ``log f`` is the smooth variable (the
    same choice :func:`impedance_table` makes for ``R`` and ``L``).  ``extrapolate``:
    ``"const"`` (default; hold the end values), ``"linear"`` (continue the end slope in
    log-log) or ``"error"``.

    Only ``|Psi|`` is kept: the pickup is the only source in the network, so the phase of
    ``Psi(omega)`` is an overall phase of the signal and cancels in ``|Phi_SQ|`` and in every
    PSD.  (It would matter for two coherently summed pickups -- see limitation 8.)
    """
    f = np.asarray(f_Hz, dtype=float)
    P = np.abs(np.asarray(Psi_SI, dtype=float))
    if f.ndim != 1 or f.size < 1 or f.shape != P.shape:
        raise ValueError("coupling_table: f_Hz and Psi_SI must be 1-D sequences of equal length")
    if np.any(f <= 0):
        raise ValueError("coupling_table: frequencies must be positive")
    if np.any(P <= 0):
        raise ValueError("coupling_table: Psi_SI must be positive (it is used as |Psi|)")
    if extrapolate not in ("const", "linear", "error"):
        raise ValueError("coupling_table: extrapolate must be 'const', 'linear' or 'error'")
    o = np.argsort(f)
    f, P = f[o], P[o]
    lf, lp = np.log(f), np.log(P)

    def Psi_of(w: float) -> float:
        lx = math.log(w / (2.0 * math.pi))
        if lp.size == 1:
            return float(P[0])
        if lx < lf[0] or lx > lf[-1]:
            if extrapolate == "error":
                raise ValueError(
                    f"coupling_table: f = {math.exp(lx):.6g} Hz is outside the tabulated range "
                    f"[{f[0]:.6g}, {f[-1]:.6g}] Hz; widen the sweep or pass extrapolate="
                    "'const' or 'linear'")
            if extrapolate == "const":
                return float(P[0] if lx < lf[0] else P[-1])
            i = 0 if lx < lf[0] else lp.size - 2
        else:
            i = int(np.clip(np.searchsorted(lf, lx) - 1, 0, lp.size - 2))
        t = (lx - lf[i]) / (lf[i + 1] - lf[i])
        return float(math.exp(lp[i] + t * (lp[i + 1] - lp[i])))

    Psi_of.f_Hz, Psi_of.Psi_SI = f, P   # type: ignore[attr-defined]
    return Psi_of


def fullwave_tables(results: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The columns of ``results["fullwave"]["sweep"]`` as sorted arrays.

    Returns ``f_Hz``, ``k_R``, ``Psi_SI`` (``= |Psi_re + i Psi_im|`` [T H m^2], the same SI
    normalisation as ``coupling/Psi_SI_H_T_m2``), ``L_H`` (``= -Im Z/omega``; **negative above
    the first self-resonance**), ``R_ohm`` (``Z_re_ohm + R_loss_ohm``: radiation plus surface
    and bulk absorption), ``R_rad_ohm`` and ``R_loss_ohm`` (bulk plus surface material loss), plus
    ``Psi_over_MQS`` and ``L_over_MQS`` when the solver reported them, and ``L_eff_H`` (the tuned
    pole's ``L + (omega/2) dL/domega`` from the exact ``dZ/domega``) when every row has it (3D).
    These are the inputs of a beyond-MQS reach (:func:`axionfem.scan.reach_dispersive`).

    The surface loss ``R_loss`` grows like 1/Delta^2 toward a self-resonance (Delta: the distance
    to it), faster than any interpolation between sparse rows follows; a warning names
    neighbouring rows between which ``R_loss/|X|`` changes steeply (:func:`loss_interpolation_check`).
    """
    sweep = (results.get("fullwave") or {}).get("sweep") or []
    if not sweep:
        raise ValueError(
            "fullwave_tables: the results have no 'fullwave'/'sweep'.  Add a "
            '"fullwave": {"f_min": ..., "f_max": ..., "points": ...} block to the design and re-solve.')
    rows = sorted(sweep, key=lambda r: r["f_Hz"])
    out = {
        "f_Hz": np.array([r["f_Hz"] for r in rows]),
        "k_R": np.array([r.get("k_R", float("nan")) for r in rows]),
        "Psi_SI": np.array([abs(complex(r["Psi_re"], r["Psi_im"])) for r in rows]),
        "L_H": np.array([r["L_H"] for r in rows]),
        "R_rad_ohm": np.array([r.get("R_rad_ohm", r.get("Z_re_ohm", 0.0) - r.get("R_bulk_ohm", 0.0)) for r in rows]),
        "R_loss_ohm": np.array([r.get("R_loss_ohm", 0.0) + r.get("R_bulk_ohm", 0.0) for r in rows]),
        "R_bulk_ohm": np.array([r.get("R_bulk_ohm", 0.0) for r in rows]),
    }
    out["R_ohm"] = out["R_rad_ohm"] + out["R_loss_ohm"]
    expected = np.array([r.get("Z_re_ohm", r.get("R_rad_ohm", 0.) + r.get("R_bulk_ohm", 0.))
                         + r.get("R_loss_ohm", 0.) for r in rows])
    if not np.allclose(out["R_ohm"], expected, rtol=1e-6, atol=1e-14):
        raise ValueError("fullwave_tables: inconsistent radiation/bulk/surface resistance columns")
    for key in ("Psi_over_MQS", "L_over_MQS", "L_eff_H"):
        if all(key in r for r in rows):
            out[key] = np.array([r[key] for r in rows])
    # rows whose Re Z the solve did not resolve (Re_Z_unresolved, 3D): the residual can move R_rad and so R_ohm by up to
    # about rho |X|; say so where that is more than 1 % of R_ohm (the engine's Q_total rule)
    uncertain = [float(r["f_Hz"]) for r, R in zip(rows, out["R_ohm"]) if "Re_Z_unresolved" in r
                 and r["Re_Z_unresolved"].get("true_relative_residual", 0.0) * abs(r.get("Z_im_ohm", 0.0)) > 0.01 * R]
    if uncertain:
        import warnings
        warnings.warn(f"fullwave_tables: Re Z is not resolved by the solve at {len(uncertain)} rows (from "
                      f"{min(uncertain):.6g} Hz): R_rad_ohm and R_ohm there can be off by more than 1 % "
                      "(Re_Z_unresolved in the rows).", RuntimeWarning, stacklevel=2)
    loss_interpolation_check(rows, "fullwave_tables")
    return out


# the largest |d ln(R_loss/|X|) / d ln f| between neighbouring rows that interpolation of R_loss is trusted with: far
# from a self-resonance it is about 1/2 (skin effect) to 1 (superconductors); toward a pole it grows like f / Delta
LOSS_SLOPE_LIMIT = 3.0
# the largest estimated relative error of R_loss interpolated log-log between neighbouring rows
LOSS_INTERPOLATION_TOLERANCE = 0.05


def _tuned_pole_inductance(x0: float, x1: float, L0: float, L1: float, Le0: float, Le1: float,
                           xq: np.ndarray) -> np.ndarray:
    """L + (1/2) dL/dx between two rows (x = ln f) from the cubic Hermite interpolant of L with the rows' exact slopes
    dL/dx = 2 (L_eff - L); it equals the rows' L_eff at the rows."""
    h = x1 - x0
    s0 = 2.0 * (Le0 - L0)
    s1 = 2.0 * (Le1 - L1)
    t = (xq - x0) / h
    value = ((2 * t ** 3 - 3 * t ** 2 + 1) * L0 + (t ** 3 - 2 * t ** 2 + t) * h * s0
             + (-2 * t ** 3 + 3 * t ** 2) * L1 + (t ** 3 - t ** 2) * h * s1)
    slope = ((6 * t ** 2 - 6 * t) * (L0 - L1) / h + (3 * t ** 2 - 4 * t + 1) * s0 + (3 * t ** 2 - 2 * t) * s1)
    return value + 0.5 * slope


def loss_interpolation_check(rows: Sequence[Mapping[str, Any]], who: str = "fullwave") -> list[tuple[float, float]]:
    """Warn where interpolating the surface loss between neighbouring sweep rows is unreliable.

    ``fullwave_tables``, ``pickup_impedance`` and the reach interpolate ``R_loss`` log-log between
    the rows.  Toward a self-resonance ``R_loss`` grows like ``1/Delta^2`` and ``X`` like ``1/Delta``,
    faster than such an interpolation between sparse rows follows.
    An interval between two rows is flagged when

    - the log-log slope of ``R_loss/|X|`` exceeds :data:`LOSS_SLOPE_LIMIT`, or
    - the estimated error of the log-log ``R_loss`` exceeds :data:`LOSS_INTERPOLATION_TOLERANCE`
      (a slope test alone misses wide intervals).
      - **Rows with the exact ``L_eff_H``** (3D): near a pole ``R_loss`` follows ``omega L_eff``,
        so ``1/Q_tuned = R_loss/(omega L_eff)`` varies slowly.  The estimate compares the
        log-log ``R_loss`` with ``omega L_eff (1/Q_tuned)``, ``L_eff`` from a cubic Hermite
        interpolant of ``L`` with the exact slopes and ``1/Q_tuned`` interpolated log-log.
      - **Other rows** (2D): the curvature of ``ln R_loss`` in ``ln f`` at the neighbouring rows
        times the interval squared over 8; at the first and last interval of a run, also the
        neighbouring interval's line extrapolated to the midpoint.

    Add solves where an interval is flagged (or adaptive initial points).  Rows with ``L <= 0``
    (above the first self-resonance) are not checked.  Returns the flagged ``(f_low, f_high)`` pairs.
    """
    import warnings

    pts = []
    for r in sorted(rows, key=lambda r: r["f_Hz"]):
        R, X = r.get("R_loss_ohm"), -r.get("Z_im_ohm", 0.0)
        f = float(r["f_Hz"])
        if R is not None and R > 0 and X > 0 and f > 0:
            L_eff = r.get("L_eff_H")
            exact = L_eff is not None and math.isfinite(L_eff) and L_eff > 0
            pts.append({"x": math.log(f), "f": f, "R": float(R), "X": X, "L": X / (2 * math.pi * f),
                        "L_eff": float(L_eff) if exact else None})
        else:
            pts.append(None)
    # runs of neighbouring valid rows; the intervals inside a run
    runs = []
    current = []
    for p in pts:
        if p is None or (current and p["x"] <= current[-1]["x"]):
            if len(current) > 1:
                runs.append(current)
            current = [] if p is None else [p]
        else:
            current.append(p)
    if len(current) > 1:
        runs.append(current)
    flagged = []
    for run in runs:
        x = np.array([p["x"] for p in run])
        y = np.log([p["R"] for p in run])
        h = np.diff(x)
        s = np.diff(y) / h
        node = np.zeros(len(run))
        for j in range(1, len(run) - 1):
            node[j] = (s[j] - s[j - 1]) / ((h[j - 1] + h[j]) / 2)
        for i, (a, b) in enumerate(zip(run[:-1], run[1:])):
            slope = math.log((b["R"] / b["X"]) / (a["R"] / a["X"])) / (b["x"] - a["x"])
            if a["L_eff"] is not None and b["L_eff"] is not None:
                xq = np.linspace(a["x"], b["x"], 9)[1:-1]
                wq = 2 * math.pi * np.exp(xq)
                L_eff = _tuned_pole_inductance(a["x"], b["x"], a["L"], b["L"], a["L_eff"], b["L_eff"], xq)
                inv_q = [math.log(p["R"] / (2 * math.pi * p["f"] * p["L_eff"])) for p in (a, b)]
                reference = wq * L_eff * np.exp(np.interp(xq, [a["x"], b["x"]], inv_q))
                log_log = np.exp(np.interp(xq, [a["x"], b["x"]], [y[i], y[i + 1]]))
                with np.errstate(divide="ignore", invalid="ignore"):
                    deviation = np.abs(log_log / reference - 1)
                error = float(np.max(np.where(reference > 0, deviation, np.inf)))
                how = "against omega L_eff / Q_tuned"
            else:
                bend = max(abs(node[i]), abs(node[i + 1])) * h[i] ** 2 / 8
                if i == 0 and len(run) > 2:
                    bend = max(bend, abs(s[1] - s[0]) * h[0] / 2)
                if i == len(run) - 2 and len(run) > 2:
                    bend = max(bend, abs(s[i] - s[i - 1]) * h[i] / 2)
                error = math.expm1(bend)
                how = "from the curvature of the neighbouring rows"
            steep = abs(slope) > LOSS_SLOPE_LIMIT
            if steep or error > LOSS_INTERPOLATION_TOLERANCE:
                flagged.append((a["f"], b["f"]))
                reason = (f"R_loss/|X| changes as f^{slope:.3g} (a self-resonance is near)" if steep else
                          f"R_loss interpolated log-log is off by about {100 * error:.2g} % ({how})")
                warnings.warn(f"{who}: between the rows at {a['f']:.6g} and {b['f']:.6g} Hz {reason}: R_loss "
                              "interpolated between them is unreliable. Add solves there "
                              "(docs/physics/fullwave_surface_losses.md).", RuntimeWarning, stacklevel=3)
    return flagged


def fullwave_losses(results: Mapping[str, Any], port: int = 0) -> dict[str, Any]:
    """The surface losses of a full-wave sweep per frequency (both engines), as sorted arrays.  The R_s models
    behind ``R_loss`` (the material ``rf`` blocks) are beta.

    Returns ``f_Hz``; ``Gamma`` (``{surface: array}``, ``surface_Gamma``: ``oint |H|^2 dA / I^2``, dimensionless);
    ``R_loss_ohm`` (sum over the surfaces with rf data of ``R_s(f) Gamma``), ``Q_loss`` (``omega L/R_loss``),
    ``Q_total`` (``omega L/(R_loss + Re Z)``) and ``T_over_Q_K``; from the 3D engine also ``L_eff_H``
    (``L + (omega/2) dL/domega`` from the exact ``dZ/domega``), ``Q_loss_tuned`` (``omega L_eff/R_loss``, the Q of the
    readout tuned to f by an ideal capacitor), ``Q_total_tuned``, ``T_over_Q_tuned_K``, ``R_loss_over_abs_X`` (the
    perturbation parameter) and ``R_eff_ohm`` (``{surface: array}``).  ``Q_loss_tuned`` is the headline Q; ``Q_loss``
    is the reactance's Q, the key shared with 2D and MQS.  Missing values are NaN (e.g. Q above the
    first self-resonance, where L <= 0).  ``port`` selects the port of a multiport 3D sweep
    (``port_solutions[port]``; the 2D engine has port 0 only).  The rows are the solved frequencies: an adaptive
    sweep's model does not carry the losses (docs/physics/fullwave_surface_losses.md).  The 2D engine has no exact
    dZ/domega, so its tuned columns are NaN: a finite difference of L between the rows is not accurate enough at the
    rows' spacing (it is -21 % to +106 % off on the coax benchmark's rows).
    """
    sweep = (results.get("fullwave") or {}).get("sweep") or []
    if not sweep:
        raise ValueError("fullwave_losses: the results have no 'fullwave'/'sweep'")
    rows = []
    for r in sorted(sweep, key=lambda r: r["f_Hz"]):
        if port:
            ports = r.get("port_solutions") or []
            if port >= len(ports):
                raise ValueError(f"fullwave_losses: port {port} is not in the sweep (it has {len(ports)} ports)")
            rows.append({**ports[port], "f_Hz": r["f_Hz"]})
        else:
            rows.append(r)
    if not any("surface_Gamma" in r for r in rows):
        raise ValueError("fullwave_losses: the sweep has no surface losses (no conductor or superconductor "
                         "surface)")

    def column(key: str) -> np.ndarray:
        return np.array([float(r[key]) if r.get(key) is not None else np.nan for r in rows])

    surfaces = sorted({s for r in rows for s in r.get("surface_Gamma", {})})
    out: dict[str, Any] = {"f_Hz": column("f_Hz")}
    out["Gamma"] = {s: np.array([r.get("surface_Gamma", {}).get(s, np.nan) for r in rows]) for s in surfaces}
    for key in ("R_loss_ohm", "Q_loss", "Q_total", "T_over_Q_K", "L_eff_H", "Q_loss_tuned", "Q_total_tuned",
                "T_over_Q_tuned_K", "R_loss_over_abs_X"):
        out[key] = column(key)
    if any("surface_losses" in r for r in rows):
        out["R_eff_ohm"] = {s: np.array([r.get("surface_losses", {}).get("surfaces", {}).get(s, {})
                                         .get("R_eff_ohm", np.nan) for r in rows]) for s in surfaces}
    return out


def _adaptive_block(results: Mapping[str, Any], who: str) -> Mapping[str, Any]:
    block = (results.get("fullwave") or {}).get("adaptive")
    if not block:
        raise ValueError(
            f"{who}: the results have no 'fullwave'/'adaptive' block.  Run a 3D design with "
            '"fullwave": {"adaptive": {"f_min_Hz": ..., "f_max_Hz": ...}} (docs/physics/adaptive_sweep.md).')
    if block.get("model") is None:
        raise ValueError(
            f"{who}: the adaptive sweep has no model: no frequency gave a usable solve (stop_reason "
            f"{block.get('stop_reason')!r}; see fullwave.adaptive.refused and the warnings).")
    return block


def fullwave_dense(results: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Beta (the adaptive sweep). The dense curve of an adaptive 3D full-wave sweep
    (``results["fullwave"]["adaptive"]["dense"]``) as arrays.

    Returns ``f_Hz``, ``Z`` (port 0's complex impedance, ohm), ``L_H`` (``= -Im Z/omega``), ``Psi`` (port 0's complex
    Psi), ``Z_matrix`` (points x ports x ports), ``Psi_vector`` (points x ports), ``dZ_domega_H`` (points x ports x
    ports, the model's derivative) and ``estimated_relative_error`` (the sweep's error estimate at each point).
    The solved rows themselves stay in ``results["fullwave"]["sweep"]`` (:func:`fullwave_tables`).
    """
    d = _adaptive_block(results, "fullwave_dense")["dense"]

    def array(x) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in np.ravel(x)], dtype=float).reshape(np.shape(x))

    return {
        "f_Hz": array(d["f_Hz"]),
        "Z": array(d["Z_re_ohm"]) + 1j * array(d["Z_im_ohm"]),
        "L_H": array(d["L_H"]),
        "Psi": array(d["Psi_re"]) + 1j * array(d["Psi_im"]),
        "Z_matrix": array(d["Z_matrix_re_ohm"]) + 1j * array(d["Z_matrix_im_ohm"]),
        "Psi_vector": array(d["Psi_vector"]["re"]) + 1j * array(d["Psi_vector"]["im"]),
        "dZ_domega_H": array(d["dZ_domega_H"]["re"]) + 1j * array(d["dZ_domega_H"]["im"]),
        "estimated_relative_error": array(d["estimated_relative_error"]),
    }


def fullwave_model(results: Mapping[str, Any]) -> Callable[[Any], dict[str, np.ndarray]]:
    """Beta (the adaptive sweep). The rational model of an adaptive 3D full-wave sweep as a function of frequency.

    ``model(f_Hz)`` returns ``Z_matrix`` (... x ports x ports, ohm) and ``Psi_vector`` (... x ports) at any frequencies,
    from the barycentric form in ``results["fullwave"]["adaptive"]["model"]``:
    ``r(f) = sum_j w_j F_j / (f - f_j) / sum_j w_j / (f - f_j)`` (exact at the support points).  Outside
    ``[f_min_Hz, f_max_Hz]`` it extrapolates; nothing there was checked.
    """
    block = _adaptive_block(results, "fullwave_model")
    m = block["model"]
    zs = np.asarray(m["support_Hz"], dtype=float)
    w = np.asarray(m["weights"]["re"], dtype=float) + 1j * np.asarray(m["weights"]["im"], dtype=float)
    F = np.asarray(m["values"]["re"], dtype=float) + 1j * np.asarray(m["values"]["im"], dtype=float)
    n = len(block["dense"]["Psi_vector"]["re"][0])

    def model(f_Hz) -> dict[str, np.ndarray]:
        f = np.asarray(f_Hz, dtype=float)
        flat = np.atleast_1d(f).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            C = w[None, :] / (flat[:, None] - zs[None, :])
            r = (C @ F) / C.sum(axis=1)[:, None]
        hit_f, hit_s = np.nonzero(flat[:, None] == zs[None, :])
        r[hit_f] = F[hit_s]
        r = r.reshape(f.shape + (F.shape[1],))
        return {"Z_matrix": r[..., :n * n].reshape(f.shape + (n, n)), "Psi_vector": r[..., n * n:]}

    return model


def pickup_coupling(results: Mapping[str, Any], *, extrapolate: str = "const") -> Callable[[float], float]:
    """``Psi_SI(omega)`` of a solved design, interpolated from its full-wave sweep.

    This is circuit limitation 7 of ``docs/physics/readout_circuit.md`` removed: the readout
    takes ``Psi(omega)`` as well as ``Z_p(omega)``, so a scan that crosses a significant
    variation of ``|Psi|`` (``kR`` of order 0.1-1) is treated correctly.  Pass the result as
    ``Psi_SI=`` to :func:`series_readout` or :class:`Readout`.
    """
    t = fullwave_tables(results)
    return coupling_table(t["f_Hz"], t["Psi_SI"], extrapolate=extrapolate)


def _loss_band(losses: Mapping[str, Any], L_p: float) -> tuple[float, float, float, float] | None:
    """Recover ``(f_lo, R_lo, f_hi, R_hi)`` from a results ``pickup/losses`` block.

    The solver stores ``R_eff_ohm`` and ``Q`` at the two edges of the design's scan band but
    not the band itself, so the frequencies are read back from ``Q = 2 pi f L_p/R_eff``:
    ``f = Q R_eff/(2 pi L_p)``, using the first surface that has both.  ``R_lo``/``R_hi`` are
    the sums over all surfaces with loss data.  Returns ``None`` if nothing usable is there.
    """
    surfaces = losses.get("surfaces") or {}
    lo = hi = 0.0
    f_lo = f_hi = None
    for s in surfaces.values():
        Re, Qs = s.get("R_eff_ohm"), s.get("Q")
        if not Re:
            continue
        lo += Re.get("f_min", 0.0)
        hi += Re.get("f_max", 0.0)
        if f_lo is None and Qs and L_p > 0:
            a = Qs.get("f_min", 0.0) * Re.get("f_min", 0.0) / (2.0 * math.pi * L_p)
            b = Qs.get("f_max", 0.0) * Re.get("f_max", 0.0) / (2.0 * math.pi * L_p)
            if a > 0 and b > 0:
                f_lo, f_hi = a, b
    if lo <= 0 or f_lo is None:
        return None
    return f_lo, lo, float(f_hi), hi


def pickup_impedance(results: Mapping[str, Any] | None = None, *, L_p_H: float | None = None,
                     R_loss_ohm: float | None = None, Q: float | None = None,
                     f_ref_Hz: float | None = None, model: str = "mqs",
                     extrapolate: str = "linear") -> ZFunc:
    """Pickup impedance ``Z_p(omega)`` of a solved design.

    ``model="mqs"`` (default): ``Z_p = R_loss(omega) - i omega L_p``, with ``L_p`` from
    ``results["coupling"]["L_p_H"]`` and ``R_loss`` from, in order of precedence,
    an explicit ``R_loss_ohm``; a quality factor ``Q`` (``R = omega L_p/Q``, or the
    constant ``R = 2 pi f_ref L_p/Q`` when ``f_ref_Hz`` is given); or the RF surface losses
    in ``results["pickup"]["losses"]["surfaces"][*]["R_eff_ohm"]``, which the solver stores
    (summed over surfaces) at the two edges of the design's scan band.  Those two values are
    interpolated as a power law in ``f`` whose exponent is read off the data, not assumed
    (``R_s ~ f^2`` for a superconductor, ``~ f^{1/2}`` for a normal metal).  The band edges
    themselves are recovered from the stored ``Q = 2 pi f L_p/R_eff`` (see :func:`_loss_band`),
    because the results document does not echo the scan band.

    ``model="fullwave"``: ``Z(omega)`` from ``results["fullwave"]["sweep"]``
    (``Z_re_ohm + R_loss_ohm`` and ``L_H``), which already carries the radiation
    resistance and the frequency dependence of the inductance.
    """
    if model not in ("mqs", "fullwave"):
        raise ValueError("pickup_impedance: model must be 'mqs' or 'fullwave'")
    if model == "fullwave":
        if results is None:
            raise ValueError("pickup_impedance(model='fullwave') needs the results document")
        sweep = results.get("fullwave", {}).get("sweep", [])
        if not sweep:
            raise ValueError(
                "pickup_impedance(model='fullwave'): the results have no 'fullwave'/'sweep'.  Add a "
                '"fullwave": {"frequencies": [...]} block to the design and re-solve.')
        loss_interpolation_check(sweep, "pickup_impedance")
        return impedance_table([r["f_Hz"] for r in sweep],
                               R_ohm=[r["Z_re_ohm"] + r.get("R_loss_ohm", 0.0) for r in sweep],
                               L_H=[r["L_H"] for r in sweep], extrapolate=extrapolate)

    if L_p_H is None:
        if results is None:
            raise ValueError("pickup_impedance: give L_p_H or the results document")
        L_p_H = results.get("coupling", {}).get("L_p_H")
        if not L_p_H:
            raise ValueError("pickup_impedance: the results have no coupling/L_p_H")
    L_p = float(L_p_H)
    if L_p <= 0:
        raise ValueError(f"pickup_impedance: L_p must be > 0 (got {L_p})")

    if R_loss_ohm is not None:
        R_of: Callable[[float], float] = lambda w, R=float(R_loss_ohm): R   # noqa: E731
    elif Q is not None:
        if Q <= 0:
            raise ValueError(f"pickup_impedance: Q must be > 0 (got {Q})")
        if f_ref_Hz is None:
            R_of = lambda w, L=L_p, Q=Q: w * L / Q           # noqa: E731
        else:
            R_of = lambda w, R=2 * math.pi * f_ref_Hz * L_p / Q: R   # noqa: E731
    elif results is not None and results.get("pickup", {}).get("losses", {}).get("surfaces"):
        band = _loss_band(results["pickup"]["losses"], L_p)
        if band is None:
            R_of = lambda w: 0.0       # noqa: E731
        else:
            f_lo, R_lo, f_hi, R_hi = band
            if R_hi > 0 and f_hi != f_lo:
                p = math.log(R_hi / R_lo) / math.log(f_hi / f_lo)
                R_of = lambda w, f0=f_lo, R0=R_lo, p=p: R0 * (w / (2 * math.pi * f0)) ** p  # noqa: E731
            else:
                R_of = lambda w, R=R_lo: R   # noqa: E731
    else:
        R_of = lambda w: 0.0           # noqa: E731

    def Zp(w: float) -> complex:
        return complex(R_of(w), -w * L_p)

    Zp.L_p_H = L_p   # type: ignore[attr-defined]
    return Zp


# ----------------------------------------------------------------- the network
@dataclass
class Branch:
    """A two-terminal branch: ``V(p) - V(m) = Z(omega) I - emf``, current ``I`` from p to m.

    The sign convention is the usual one for an EMF drawn *inside* a loop: a positive ``emf``
    drives positive current in the p -> m reference direction, so an isolated branch closed by
    an external impedance ``Z_ext`` carries ``I = emf/(Z + Z_ext)``.  Passive elements obey the
    plain passive sign convention ``V(p) - V(m) = Z I``.
    """
    name: str
    p: str
    m: str
    kind: str                      # "R", "L", "C", "Z", "V", "I", "short"
    Z: ZFunc | None = None
    T_K: float | None = None       # physical temperature for Johnson noise
    emf: complex | Callable[[float], complex] = 0.0
    value: float | None = None     # the R/L/C value, when constant (for reporting and guesses)
    noise: bool = True             # include this branch's Johnson noise

    def Z_at(self, w: float) -> complex:
        return 0j if self.Z is None else complex(self.Z(w))

    def emf_at(self, w: float) -> complex:
        return complex(self.emf(w)) if callable(self.emf) else complex(self.emf)


@dataclass
class _Transformer:
    name: str
    p1: str
    m1: str
    p2: str
    m2: str
    ratio: float                   # n = N2/N1: V_2 = n V_1, I_1 = -n I_2


@dataclass
class Solution:
    """Result of one network solve at ``omega``."""
    omega: float
    node_v: dict[str, complex]
    branch_i: dict[str, complex]

    def V(self, p: str, m: str = "gnd") -> complex:
        return self.node_v[p] - self.node_v[m]

    def I(self, branch: str) -> complex:
        return self.branch_i[branch]


class Circuit:
    """A linear circuit solved by complex modified nodal analysis at one frequency.

    Every two-terminal element is a *branch* carrying an unknown current, so mutual
    inductance, ideal voltage sources, ideal transformers and zero-impedance shorts all use
    the same stamp.  The system is

        [  0    A  ] [ V ]   [   J  ]
        [ A^T  -Z  ] [ I ] = [ -emf ]

    with ``A[node][branch]`` the incidence matrix (+1 at ``p``, -1 at ``m``), ``Z`` the
    branch impedance matrix including the mutual terms, ``emf`` the branch EMFs and ``J`` the
    nodal current injections.  ``A^T V = Z I - emf`` is the branch constitutive law and the
    first block row is Kirchhoff's current law.  The matrix is *symmetric* for every
    reciprocal network, which :meth:`check_reciprocity` verifies.

    Node ``"gnd"`` (or whatever ``ground`` names) is the reference, created automatically.
    """

    def __init__(self, ground: str = "gnd"):
        self.ground = ground
        self._branches: list[Branch] = []
        self._by_name: dict[str, int] = {}
        self._mutual: list[tuple[str, str, float]] = []
        self._transformers: list[_Transformer] = []
        self._nodes: list[str] = [ground]

    # ------------------------------------------------------------ construction
    def _node(self, n: str) -> str:
        if n not in self._nodes:
            self._nodes.append(n)
        return n

    def _used(self, name: str) -> bool:
        return name in self._by_name or any(t.name == name for t in self._transformers)

    def _add(self, b: Branch) -> Branch:
        if self._used(b.name):
            raise ValueError(f"circuit: duplicate element name {b.name!r}")
        if b.p == b.m:
            raise ValueError(f"circuit: element {b.name!r} has both terminals on node {b.p!r}")
        self._node(b.p)
        self._node(b.m)
        self._by_name[b.name] = len(self._branches)
        self._branches.append(b)
        return b

    def _get(self, name: str) -> Branch:
        if name not in self._by_name:
            raise ValueError(f"circuit: no element named {name!r} "
                             f"(have {sorted(self._by_name)})")
        return self._branches[self._by_name[name]]

    def resistor(self, name: str, p: str, m: str, R: float, T_K: float | None = None) -> Branch:
        """Resistor ``R`` [ohm] (or a callable ``R(omega)``); ``T_K`` adds its Johnson noise."""
        if not callable(R) and float(R) < 0:
            raise ValueError(f"circuit.resistor {name!r}: R must be >= 0 (got {R})")
        return self._add(Branch(name, p, m, "R", _as_zfunc(R, "R"), T_K,
                                value=None if callable(R) else float(R)))

    def inductor(self, name: str, p: str, m: str, L: float) -> Branch:
        """Inductor ``L`` [H]; ``Z = -i omega L``.  Lossless -- add a series resistor for loss."""
        if not callable(L) and float(L) <= 0:
            raise ValueError(f"circuit.inductor {name!r}: L must be > 0 (got {L})")
        return self._add(Branch(name, p, m, "L", _as_zfunc(L, "L"),
                                value=None if callable(L) else float(L)))

    def capacitor(self, name: str, p: str, m: str, C: float, T_K: float | None = None,
                  tan_delta: float = 0.0) -> Branch:
        """Capacitor ``C`` [F]; ``Z = i/(omega C)``.  ``tan_delta`` (``= 1/Q_C``) adds a series
        loss resistance ``R = tan_delta/(omega C)`` whose Johnson noise is taken at ``T_K``."""
        if not callable(C) and float(C) <= 0:
            raise ValueError(f"circuit.capacitor {name!r}: C must be > 0 (got {C})")
        if tan_delta < 0:
            raise ValueError(f"circuit.capacitor {name!r}: tan_delta must be >= 0 (got {tan_delta})")
        val = None if callable(C) else float(C)
        if tan_delta == 0.0:
            b = self._add(Branch(name, p, m, "C", _as_zfunc(C, "C"), value=val))
        else:
            def Zc(w: float, C=C, td=float(tan_delta)) -> complex:
                Cw = float(C(w)) if callable(C) else float(C)
                return complex(td / (w * Cw), 1.0 / (w * Cw))
            b = self._add(Branch(name, p, m, "C", Zc, T_K, value=val))
        b.tan_delta = float(tan_delta)   # type: ignore[attr-defined]
        return b

    def impedance(self, name: str, p: str, m: str, Z: ZFunc | complex, T_K: float | None = None,
                  noise: bool = True) -> Branch:
        """Generic impedance branch; ``Z`` is a callable ``Z(omega)`` or a constant.
        Its Johnson noise is ``Re Z(omega) hbar omega (2n+1)`` at ``T_K``."""
        Zf = Z if callable(Z) else (lambda w, z=complex(Z): z)
        return self._add(Branch(name, p, m, "Z", Zf, T_K, noise=noise))

    def source(self, name: str, p: str, m: str, emf: complex | Callable[[float], complex] = 0.0,
               Z: ZFunc | complex = 0.0, T_K: float | None = None) -> Branch:
        """Voltage source: ``V(p) - V(m) = Z(omega) I - emf``, i.e. ``emf`` drives positive
        current from ``p`` to ``m``.  ``Z = 0`` is an ideal source.  Its Johnson noise uses
        ``Re Z`` at ``T_K`` -- this is how the pickup is represented."""
        Zf = Z if callable(Z) else (lambda w, z=complex(Z): z)
        return self._add(Branch(name, p, m, "V", Zf, T_K, emf=emf))

    def current_source(self, name: str, p: str, m: str,
                       value: complex | Callable[[float], complex]) -> Branch:
        """Ideal current source driving ``value`` *into* node ``p`` and out of ``m``."""
        return self._add(Branch(name, p, m, "I", None, emf=value))

    def short(self, name: str, p: str, m: str) -> Branch:
        """Ideal short (zero impedance) whose current is an unknown -- an ammeter."""
        return self._add(Branch(name, p, m, "short", lambda w: 0j))

    def mutual(self, name_a: str, name_b: str, M: float | None = None, *,
               k: float | None = None) -> None:
        """Mutual inductance between two inductive branches: ``Z_{ab} = Z_{ba} = -i omega M``.

        Positive ``M`` means the mutual flux *adds* when both currents flow from their ``p``
        terminal to their ``m`` terminal.  Give ``M`` [H] or the coupling coefficient
        ``k = M/sqrt(L_a L_b)`` (``|k| <= 1``; ``|k| = 1`` makes the pair singular if they
        sit in the same loop).
        """
        ba, bb = self._get(name_a), self._get(name_b)
        if name_a == name_b:
            raise ValueError("circuit.mutual: an element cannot be coupled to itself")
        if (M is None) == (k is None):
            raise ValueError("circuit.mutual: give exactly one of M or k")
        if k is not None:
            if abs(k) > 1.0:
                raise ValueError(f"circuit.mutual: |k| must be <= 1 (got {k})")
            if ba.value is None or bb.value is None:
                raise ValueError("circuit.mutual: k needs both elements to have a constant L; "
                                 "pass M instead")
            M = k * math.sqrt(ba.value * bb.value)
        for b in (ba, bb):
            if b.kind not in ("L", "Z", "V"):
                raise ValueError(f"circuit.mutual: {b.name!r} (kind {b.kind!r}) is not inductive; "
                                 "couple inductors, generic-impedance or source branches")
        self._mutual.append((name_a, name_b, float(M)))

    def transformer(self, name: str, p1: str, m1: str, p2: str, m2: str, ratio: float) -> None:
        """Ideal transformer with turns ratio ``n = ratio = N2/N1``:
        ``V(p2)-V(m2) = n [V(p1)-V(m1)]`` and ``I_1 = -n I_2``, so ``V_1 I_1 + V_2 I_2 = 0``
        exactly (lossless, no leakage, infinite magnetising inductance).  For a real
        transformer use two :meth:`inductor` branches and :meth:`mutual`."""
        if self._used(name):
            raise ValueError(f"circuit: duplicate element name {name!r}")
        if ratio == 0:
            raise ValueError(f"circuit.transformer {name!r}: ratio must be nonzero")
        for n in (p1, m1, p2, m2):
            self._node(n)
        self._transformers.append(_Transformer(name, p1, m1, p2, m2, float(ratio)))

    # ------------------------------------------------------------ assembly and solve
    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    @property
    def branches(self) -> list[Branch]:
        return list(self._branches)

    def _assemble(self, w: float) -> tuple[np.ndarray, list[str], list[str]]:
        if not (w > 0):
            raise ValueError(f"circuit: omega must be > 0 (got {w})")
        free = [n for n in self._nodes if n != self.ground]
        nn, nb = len(free), len(self._branches) + len(self._transformers)
        if nb == 0:
            raise ValueError("circuit: the network has no elements")
        idx = {n: i for i, n in enumerate(free)}
        A = np.zeros((nn, nb), dtype=complex)
        Z = np.zeros((nb, nb), dtype=complex)
        names: list[str] = []
        for j, b in enumerate(self._branches):
            names.append(b.name)
            if b.kind == "I":
                continue                       # not an unknown; enters the RHS
            if b.p in idx:
                A[idx[b.p], j] += 1.0
            if b.m in idx:
                A[idx[b.m], j] -= 1.0
            Z[j, j] = b.Z_at(w)
        for j0, t in enumerate(self._transformers):
            j = len(self._branches) + j0
            names.append(t.name)
            # unknown is I_2 (into p2); I_1 = -n I_2, and A^T V = 0 gives V_2 = n V_1
            for node, s in ((t.p1, t.ratio), (t.m1, -t.ratio), (t.p2, -1.0), (t.m2, 1.0)):
                if node in idx:
                    A[idx[node], j] += s
        for na, nb_, M in self._mutual:
            ia, ib = self._by_name[na], self._by_name[nb_]
            Z[ia, ib] += complex(0.0, -w * M)
            Z[ib, ia] += complex(0.0, -w * M)
        K = np.zeros((nn + nb, nn + nb), dtype=complex)
        K[:nn, nn:] = A
        K[nn:, :nn] = A.T
        K[nn:, nn:] = -Z
        for j, b in enumerate(self._branches):
            if b.kind == "I":                  # pin the unused unknown
                K[nn + j, :] = 0.0
                K[nn + j, nn + j] = 1.0
        return K, free, names

    def solve(self, omega: float, emf: Mapping[str, complex] | None = None,
              inject: Mapping[str, complex] | None = None, replace: bool = False) -> Solution:
        """Solve at ``omega``.  ``emf`` overrides branch EMFs (merged onto the stored ones
        unless ``replace``); ``inject`` adds a current [A] injected into a node."""
        return self.solve_many(omega, [emf or {}], inject, replace)[0]

    def solve_many(self, omega: float, emfs: Sequence[Mapping[str, complex]],
                   inject: Mapping[str, complex] | None = None,
                   replace: bool = False) -> list[Solution]:
        """Solve at ``omega`` for several EMF patterns sharing one factorisation.
        ``replace=True``: only the listed branches have an EMF (all others zero)."""
        K, free, names = self._assemble(omega)
        nn, nb = len(free), len(names)
        idx = {n: i for i, n in enumerate(free)}
        rhs = np.zeros((nn + nb, max(len(emfs), 1)), dtype=complex)
        for c, pattern in enumerate(emfs):
            for nm in pattern:
                self._get(nm)
            for j, b in enumerate(self._branches):
                val = pattern.get(b.name, None)
                v = complex(val) if val is not None else (0j if replace else b.emf_at(omega))
                if b.kind == "I":
                    if b.p in idx:
                        rhs[idx[b.p], c] += v
                    if b.m in idx:
                        rhs[idx[b.m], c] -= v
                else:
                    # constitutive law A^T V - Z I = -emf, so a positive emf drives
                    # positive branch current in the p -> m direction
                    rhs[nn + j, c] -= v
            if inject:
                for n, v2 in inject.items():
                    if n not in self._nodes:
                        raise ValueError(f"circuit.solve: no node named {n!r}")
                    if n in idx:
                        rhs[idx[n], c] += complex(v2)
        try:
            x = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"circuit.solve: the network is singular at omega = {omega:.6g} rad/s ({exc}). "
                "Usual causes: a node reachable only through zero-impedance branches or current "
                "sources; a loop of ideal voltage sources or shorts; a sub-network with no path "
                f"to the ground node {self.ground!r}; or |k| = 1 between two inductors that sit "
                "in the same loop.") from exc
        return [Solution(omega, {self.ground: 0j, **{n: x[idx[n], c] for n in free}},
                         {names[j]: x[nn + j, c] for j in range(nb)})
                for c in range(x.shape[1])]

    def check_reciprocity(self, omega: float) -> float:
        """Max relative asymmetry ``|K - K^T|/max|K|`` of the assembled MNA matrix.

        **This is a tautology for the present element set** and is kept only as a guard for
        future non-reciprocal ones.  The stamp is ``[[0, A], [A^T, -Z]]`` with ``Z`` symmetric
        (mutual inductance is stamped symmetrically and a complex ``M`` is rejected), so the
        matrix is symmetric by construction and this returns exactly 0 -- it is *not* evidence
        that the network is reciprocal.  The statement with
        content is :meth:`transfer_reciprocity`, which goes through the solve."""
        K, _, _ = self._assemble(omega)
        scale = float(np.max(np.abs(K)))
        return float(np.max(np.abs(K - K.T)) / scale) if scale > 0 else 0.0

    def transfer_reciprocity(self, omega: float, branches: Sequence[str] | None = None) -> float:
        """Max relative asymmetry of the branch transfer matrix ``T_bc`` (the current in branch
        ``c`` for a unit EMF in branch ``b``), computed from actual solves.

        ``T_bc = T_cb`` for a reciprocal network; unlike :meth:`check_reciprocity` this exercises
        the incidence matrix, the transformer stamp, the solve and the extraction, so it does
        fail if any of them is wrong.  ``branches`` defaults to every two-terminal branch
        (``O(n_b)`` solves in one factorisation)."""
        names = [b.name for b in self._branches] if branches is None else list(branches)
        if len(names) < 2:
            return 0.0
        sols = self.solve_many(omega, [{nm: 1.0 + 0j} for nm in names], replace=True)
        T = np.array([[s.I(nm) for nm in names] for s in sols])
        scale = float(np.max(np.abs(T)))
        return float(np.max(np.abs(T - T.T)) / scale) if scale > 0 else 0.0

    # ------------------------------------------------------------ derived quantities
    def driving_point_impedance(self, omega: float, branch: str) -> complex:
        """``Z_eff = emf/I`` for a unit EMF in ``branch``: the series impedance of the loop
        closed through that branch, *including* the branch's own impedance."""
        I = self.solve_many(omega, [{branch: 1.0 + 0j}], replace=True)[0].I(branch)
        if I == 0:
            raise ValueError(f"circuit.driving_point_impedance: no current flows in {branch!r}")
        return 1.0 / I

    def port_impedance(self, omega: float, p: str, m: str) -> complex:
        """``Z = V/I`` between two nodes for 1 A injected at ``p`` and drawn at ``m``,
        with every internal source off."""
        s = self.solve_many(omega, [{}], inject={p: 1.0 + 0j, m: -1.0 + 0j}, replace=True)[0]
        return s.V(p, m)

    def transfer(self, omega: float, source: str, readout: Callable[[Solution], complex]) -> complex:
        """``readout(solution)`` for a unit EMF in branch ``source`` and nothing else."""
        return readout(self.solve_many(omega, [{source: 1.0 + 0j}], replace=True)[0])

    def draw(self, annotate: bool = True, *, omega: float | None = None, title: str | None = None):
        """Schematic of this network, generated from its element list (``schemdraw``; install
        ``axionfem[draw]``).  Series chains between branch points are drawn as loops/ladders, the
        ground node as ground symbols, mutual inductances and ideal transformers as dashed
        couplings.  ``annotate`` adds the element values (``omega`` [rad/s] evaluates the
        frequency-dependent ones).  Returns a ``schemdraw.Drawing``: ``.save("x.svg")``.
        See :mod:`axionfem.draw`."""
        from .draw import draw_network
        return draw_network(self, annotate=annotate, omega=omega, title=title)

    def noisy_branches(self, omega: float) -> list[tuple[str, float, float]]:
        """``(name, Re Z(omega), T_K)`` for each dissipative branch.

        A branch with ``T_K = None`` is taken at **T = 0**, i.e. it still carries its vacuum
        noise ``hbar omega Re Z`` (``2n + 1 = 1``): fluctuation-dissipation does not allow a
        dissipative element to be silent, and dropping it would let ``N_V`` fall below the quantum
        limit.  Set ``noise=False`` on the branch to silence it
        deliberately -- e.g. a resistor standing in for a radiation resistance whose noise is
        accounted elsewhere."""
        out = []
        for b in self._branches:
            if not b.noise or b.kind == "I":
                continue
            R = b.Z_at(omega).real
            if R > 0:
                out.append((b.name, R, 0.0 if b.T_K is None else float(b.T_K)))
        return out

    def thermal_noise(self, omega: float, readout: Callable[[Solution], complex]) -> dict:
        """Johnson noise of ``readout`` summed over every dissipative branch.

        Each branch contributes an independent series EMF with
        ``S_V = Re Z(omega) hbar omega (2 n(omega, T) + 1)``  (spec S3.3).  They are
        independent because the mutual terms ``-i omega M`` are purely reactive, so the
        branch noise matrix ``~ Re Z`` is diagonal (see the limitations in the docs).

        Returns ``total`` (``[readout units]^2 s``), ``by_branch``, and ``transfer2``
        (``|readout per unit branch EMF|^2``, needed to refer the noise elsewhere).
        """
        noisy = self.noisy_branches(omega)
        if not noisy:
            return {"total": 0.0, "by_branch": {}, "transfer2": {}}
        sols = self.solve_many(omega, [{nm: 1.0 + 0j} for nm, _, _ in noisy], replace=True)
        total, parts, t2 = 0.0, {}, {}
        for (nm, R, T), s in zip(noisy, sols):
            S = R * HBAR * omega * (2.0 * n_bose(omega, T) + 1.0)
            h2 = abs(readout(s)) ** 2
            t2[nm] = h2
            parts[nm] = h2 * S
            total += parts[nm]
        return {"total": total, "by_branch": parts, "transfer2": t2}

    def nyquist_check(self, omega: float, branch: str) -> float:
        """Nyquist's theorem as a self-test: at a *uniform* temperature the noise EMF referred
        in series with ``branch`` must equal ``Re Z_eff(omega) hbar omega (2n+1)``.  Returns the
        relative error ``|sum_k |H_k/H_b|^2 R_k - Re Z_eff| / Re Z_eff`` (temperature-free,
        so it holds whatever the branch temperatures are set to)."""
        Zeff = self.driving_point_impedance(omega, branch)
        noisy = self.noisy_branches(omega)
        if not noisy or Zeff.real == 0:
            raise ValueError("circuit.nyquist_check: needs a dissipative network")
        readout: Callable[[Solution], complex] = lambda s: s.I(branch)
        Hb = self.transfer(omega, branch, readout)
        R_sum = sum(R * abs(self.transfer(omega, nm, readout) / Hb) ** 2 for nm, R, _ in noisy)
        return abs(R_sum - Zeff.real) / abs(Zeff.real)


# ----------------------------------------------------------------- SQUID amplifier
@dataclass
class Squid:
    """A flux-to-voltage amplifier (dc SQUID) coupled to an input coil by ``M``.

    Noise is parametrised exactly as the rest of the package does it (spec S3.5):
    imprecision flux PSD ``S_Phi`` [Wb^2 s] and backaction circulating-current PSD ``S_J``
    [A^2 s], both Foster-normalised (half the usual one-sided values in Wb^2/Hz, A^2/Hz), with

        eta_A = 2 sqrt(S_Phi S_J)/hbar          (1 at the standard quantum limit)
        xi    = 2 R_0 S_Phi/(hbar omega_0 M^2)  (the readout coupling; xi_opt maximises G-bar)

    The SQUID's input impedance is neglected, as in [C] Eq. (E33): ``M`` does not load the
    circuit, so the loaded Q is set by the circuit's own losses.

    ``S_PhiJ`` [Wb A s, Foster] is the real cross-spectrum ``<Phi_n J*>`` of the imprecision flux
    and the circulating current, sign included (0 by default: uncorrelated).  The circulating
    current puts the series EMF ``+i omega M J`` into the input
    coil, so the output noise gains ``2 Re[conj(i omega M H_in) S_PhiJ]``
    (``docs/physics/squid_chain.md`` Sec. 4).  With a correlation, ``eta_A`` should be the
    correlated ``2 sqrt(S_Phi S_J - S_PhiJ^2)/hbar`` (what :meth:`axionfem.squid.Chain.squid` sets).
    """
    M_H: float
    eta_A: float = 1.0
    S_Phi: float = 0.0        # Wb^2 s
    S_J: float = 0.0          # A^2 s
    input_coil: str = "L_in"  # branch whose current threads the SQUID
    S_PhiJ: float = 0.0       # Wb A s, real <Phi_n J*> (sign included)

    def correlation_term(self, omega: float, H_in: complex) -> float:
        """Output flux PSD [Wb^2 s] of the imprecision-backaction cross term,
        ``2 Re[conj(i omega M H_in) S_PhiJ]``, ``H_in`` the flux per volt of EMF in the input coil.

        Raises when ``S_PhiJ^2 > S_Phi S_J`` (not a covariance)."""
        if not self.S_PhiJ:
            return 0.0
        if not np.isfinite(self.S_PhiJ) or self.S_PhiJ ** 2 > self.S_Phi * self.S_J * (1.0 + 1e-12):
            raise ValueError("Squid: need S_PhiJ^2 <= S_Phi S_J (a positive semidefinite noise covariance)")
        return float(2.0 * (np.conj(1j * omega * self.M_H * H_in) * self.S_PhiJ).real)

    @staticmethod
    def from_xi(xi: float, omega0: float, R0: float, M_H: float, eta_A: float = 1.0) -> "Squid":
        """SQUID whose noise realises ``xi`` at ``(omega0, R0)`` with mutual ``M``  (spec S3.5):
        ``S_Phi = hbar xi omega0 M^2/(2 R0)``, ``S_J = (hbar eta_A/2)^2/S_Phi``."""
        if min(xi, omega0, R0, abs(M_H), eta_A) <= 0:
            raise ValueError("Squid.from_xi: xi, omega0, R0, |M| and eta_A must all be > 0")
        S_Phi = HBAR * xi * omega0 * M_H * M_H / (2.0 * R0)
        return Squid(M_H=M_H, eta_A=eta_A, S_Phi=S_Phi, S_J=(HBAR * eta_A / 2.0) ** 2 / S_Phi)

    @staticmethod
    def M_for_xi(xi: float, omega0: float, R0: float, S_Phi: float) -> float:
        """``M = sqrt(2 R0 S_Phi/(hbar omega0 xi))``  (spec S3.5)."""
        if min(xi, omega0, R0, S_Phi) <= 0:
            raise ValueError("Squid.M_for_xi: xi, omega0, R0 and S_Phi must all be > 0")
        return math.sqrt(2.0 * R0 * S_Phi / (HBAR * omega0 * xi))

    @staticmethod
    def from_flux_noise(sqrt_S_Phi_one_sided: float, eta_A: float = 1.0, M_H: float = 1e-9,
                        unit: str = "Phi0") -> "Squid":
        """SQUID from a datasheet flux noise: ``sqrt_S_Phi_one_sided`` in ``Phi_0/sqrt(Hz)``
        (``unit="Phi0"``) or ``Wb/sqrt(Hz)`` (``unit="Wb"``).  The Foster-normalised
        ``S_Phi`` is *half* the one-sided value: ``S_Phi = (sqrt S^(1))^2/2``."""
        if unit not in ("Phi0", "Wb"):
            raise ValueError("Squid.from_flux_noise: unit must be 'Phi0' or 'Wb'")
        a = sqrt_S_Phi_one_sided * (PHI0 if unit == "Phi0" else 1.0)
        S_Phi = a * a / 2.0
        return Squid(M_H=M_H, eta_A=eta_A, S_Phi=S_Phi, S_J=(HBAR * eta_A / 2.0) ** 2 / S_Phi)

    def xi_at(self, omega0: float, R0: float, M_eff: float | None = None) -> float:
        M = abs(M_eff) if M_eff is not None else abs(self.M_H)
        return 2.0 * R0 * self.S_Phi / (HBAR * omega0 * M * M) if M > 0 else 0.0


# ----------------------------------------------------------------- the readout
@dataclass
class Readout:
    """A pickup + readout network, its noise budget and its reduction to the scan inputs.

    Build one with :func:`series_readout` (DMRadio style), or assemble a :class:`Circuit`
    yourself and name the pickup branch and the SQUID.
    """
    circuit: Circuit
    squid: Squid
    pickup: str = "pickup"            # the branch carrying the axion EMF
    # [T H m^2]; Phi_a = kappa_a Psi_SI.  A callable Psi_SI(omega) -- e.g. from
    # pickup_coupling(results) -- makes the coupling frequency dependent (limitation 7).
    Psi_SI: float | Callable[[float], float] = 0.0
    L_p_H: float = 0.0                # pickup inductance, for the participation ratio
    label: str = "readout"
    tuner: str | None = None          # the tuning-capacitor branch, if any

    # ------------------------------------------------------------ transfer functions
    def Psi_at(self, omega: float) -> float:
        """The pickup coupling ``Psi_SI`` [T H m^2] at ``omega``.

        A constant unless ``Psi_SI`` was given as a callable (a full-wave ``Psi(omega)``); the
        axion line is narrower than ``omega/Q_axion ~ 10^-6``, so one value per tuning is all
        the signal PSD needs."""
        p = self.Psi_SI
        return float(p(omega)) if callable(p) else float(p)

    def flux(self, sol: Solution) -> complex:
        """Flux coupled into the SQUID loop, ``Phi = M I_in`` [Wb]."""
        return self.squid.M_H * sol.I(self.squid.input_coil)

    def H_signal(self, omega: float) -> complex:
        """``Phi_SQ`` per volt of pickup EMF [Wb/V]."""
        return self.circuit.transfer(omega, self.pickup, self.flux)

    def Z_eff(self, omega: float) -> complex:
        """Impedance in series with the pickup branch, including ``Z_p`` [ohm]."""
        return self.circuit.driving_point_impedance(omega, self.pickup)

    def M_eff(self, omega: float) -> complex:
        """``Phi_SQ/I_pickup`` [H]: the mutual inductance an equivalent single loop needs.
        Equals ``M`` whenever the pickup and the input coil carry the same current."""
        return self.H_signal(omega) * self.Z_eff(omega)

    # ------------------------------------------------------------ resonance and tuning
    def resonance(self, f_guess_Hz: float | None = None, span: float = 10.0,
                  n_scan: int = 400) -> float:
        """Resonant angular frequency: the ``omega`` nearest the guess where ``Im Z_eff = 0``.

        Scans ``[f_guess/span, f_guess*span]`` log-uniformly for a sign change and bisects
        in ``log omega``.  Default guess: the series resonance of the stored L and C.
        """
        if f_guess_Hz is None:
            f_guess_Hz = self._resonance_guess()
        w0 = 2.0 * math.pi * float(f_guess_Hz)
        ws = np.geomspace(w0 / span, w0 * span, n_scan)
        x = np.array([self.Z_eff(float(w)).imag for w in ws])
        cross = np.nonzero(np.sign(x[:-1]) * np.sign(x[1:]) < 0)[0]
        if cross.size == 0:
            raise ValueError(
                f"Readout.resonance: Im Z_eff has no zero between {w0/span/2/math.pi:.4g} and "
                f"{w0*span/2/math.pi:.4g} Hz.  Pass a better f_guess_Hz or a larger span; a "
                "circuit with no tuning capacitor has no series resonance at all.")
        k = int(cross[np.argmin(np.abs(np.log(ws[cross] / w0)))])
        a, b, fa = float(ws[k]), float(ws[k + 1]), float(x[k])
        for _ in range(200):
            mid = math.sqrt(a * b)
            fm = self.Z_eff(mid).imag
            if fm == 0.0 or (b - a) < 1e-15 * mid:
                return mid
            if fa * fm < 0:
                b = mid
            else:
                a, fa = mid, fm
        return math.sqrt(a * b)

    def _resonance_guess(self) -> float:
        """A starting frequency for :meth:`resonance`: 1/(2 pi sqrt(L C)) with L the sum of the
        constant inductances and C the *tuning* capacitor.

        Only the tuner is used.  Other capacitors in the network (a parasitic C across the
        pickup, a tap) are not in series with it, so combining them as ``1/sum(1/C)`` would give
        a guess that can be far off.  It is only
        a guess: :meth:`resonance` scans a decade either side and bisects.

        L includes the pickup's own inductance ``L_p_H`` whenever the pickup branch carries it
        (a source branch with an inductive ``Z_p``, as :func:`series_readout` builds it): summing
        only the L branches would leave it out as soon as an input coil is present, and with
        ``L_in << L_p`` the guess would sit more than a decade above the resonance, outside the
        range :meth:`resonance` scans."""
        L = sum(b.value for b in self.circuit.branches if b.kind == "L" and b.value)
        if self.L_p_H and self._pickup_is_inductive():
            L += self.L_p_H
        L = L or self.L_p_H
        C = None
        if self.tuner is not None:
            C = self.circuit._get(self.tuner).value
        if C is None:
            Cs = [b.value for b in self.circuit.branches if b.kind == "C" and b.value]
            C = max(Cs) if Cs else None      # the largest: the likeliest series tuner
        if not C or not L:
            raise ValueError("Readout.resonance: pass f_guess_Hz "
                             "(no constant L and C to guess a resonance from)")
        return 1.0 / (2.0 * math.pi * math.sqrt(L * C))

    def _pickup_is_inductive(self) -> bool:
        """Whether the pickup branch itself carries an inductive reactance (``Im Z_p < 0``)."""
        try:
            b = self.circuit._get(self.pickup)
        except ValueError:
            return False
        if b.kind == "L":
            return False                     # already counted among the L branches
        try:
            return b.Z_at(2.0 * math.pi * 1e6).imag < 0
        except Exception:
            return False

    def tune(self, f_Hz: float, branch: str | None = None, span: float = 1e6) -> float:
        """Set the tuning capacitor so that the loop resonates at ``f_Hz``; returns ``C`` [F].

        The start value is the closed form ``C = -1/(omega X_rest)`` with ``X_rest`` the loop
        reactance with the capacitor shorted; bisection on ``Im Z_eff(2 pi f) = 0`` then makes
        it exact for any network (a transformer-coupled secondary, a frequency-dependent
        ``Z_p``, ...).
        """
        branch = branch or self.tuner
        if branch is None:
            raise ValueError("Readout.tune: this readout has no tuning capacitor; "
                             "pass branch=<capacitor name>")
        b = self.circuit._get(branch)
        if b.kind != "C":
            raise ValueError(f"Readout.tune: branch {branch!r} is a {b.kind!r}, not a capacitor")
        if not (f_Hz > 0):
            raise ValueError(f"Readout.tune: f_Hz must be > 0 (got {f_Hz})")
        w = 2.0 * math.pi * float(f_Hz)
        C_old, td = b.value, getattr(b, "tan_delta", 0.0)

        def set_C(C: float) -> None:
            if td:
                b.Z = lambda ww, C=C, td=td: complex(td / (ww * C), 1.0 / (ww * C))
            else:
                b.Z = _as_zfunc(C, "C")
            b.value = C

        def resid(C: float) -> float:
            set_C(C)
            return self.Z_eff(w).imag

        big = 1e12 / w        # a capacitance that is effectively a short at w
        set_C(big)
        X_rest = self.Z_eff(w).imag - 1.0 / (w * big)
        C0 = -1.0 / (w * X_rest) if X_rest < 0 else (C_old or 1.0 / (w * w * max(self.L_p_H, 1e-18)))
        if not (C0 > 0 and math.isfinite(C0)):
            C0 = C_old or 1e-9
        f0 = resid(C0)
        # The natural size of Im Z_eff is the tuning reactance itself, 1/(omega C0).  For a plain
        # series loop the closed form is exact, so f0 sits at round-off; accepting it there is not
        # just a short cut -- continuing would bisect on a residual whose *sign* is round-off, and
        # one ulp of C0 already moves Im Z_eff by more than f0 (dX/dC = 1/(omega C^2)).
        scale = 1.0 / (w * C0)
        if abs(f0) <= 1e-12 * scale:
            return C0
        # Bracket Im Z_eff = 0 by expanding geometrically about C0.  Both endpoints of the
        # bracket are points that were actually evaluated (never recomputed by dividing back
        # out), so the bisection below starts from a genuine sign change.
        lo = hi = C0
        f_lo = f_hi = f0
        bracket = None
        for _ in range(400):
            lo_new, hi_new = lo / 1.15, hi * 1.15
            f_new = resid(lo_new)
            if f_new * f_lo <= 0:
                bracket = (lo_new, f_new, lo, f_lo)
                break
            f_hi_new = resid(hi_new)
            if f_hi_new * f_hi <= 0:
                bracket = (hi, f_hi, hi_new, f_hi_new)
                break
            lo, f_lo, hi, f_hi = lo_new, f_new, hi_new, f_hi_new
            if hi / C0 > span:
                break
        if bracket is None:
            if C_old:
                set_C(C_old)    # leave the circuit as we found it
            raise ValueError(
                f"Readout.tune: no capacitance within a factor {span:.3g} of {C0:.6g} F resonates "
                f"at {f_Hz:.6g} Hz.  Is the loop reactance inductive there?  Above the pickup's "
                "self-resonance a series capacitor cannot tune the loop.")
        a, fa, b2, fb = bracket
        for _ in range(200):
            mid = math.sqrt(a * b2)
            fm = resid(mid)
            if fm == 0.0 or (b2 - a) <= 1e-15 * mid:
                a = b2 = mid
                break
            if fa * fm < 0:
                b2, fb = mid, fm
            else:
                a, fa = mid, fm
        C = math.sqrt(a * b2)
        set_C(C)
        return C

    # ------------------------------------------------------------ equivalent single pole
    def equivalent(self, omega0: float | None = None, f_Hz: float | None = None,
                   dlog: float = 1e-5, corr_tol: float = 1e-2) -> dict:
        """Reduce the network to the single-pole detector the optimal scan assumes.

        Definitions (all exact for the series topology):

        * ``omega0``   : ``Im Z_eff(omega0) = 0``
        * ``R_eff``    : ``Re Z_eff(omega0)``
        * ``L_eff``    : ``|dIm Z_eff/domega|/2`` at ``omega0`` (the reactance-slope
          inductance; ``= L_p + L_in + L_lead`` for a series loop)
        * ``Q_loaded`` : ``omega0 L_eff/R_eff``
        * ``n_eff``    : from ``sum_k |H_k/H_s|^2 R_k (2 n_k + 1) = R_eff (2 n_eff + 1)``, so the
          referred thermal noise is matched exactly; ``T_eff/Q = sum_k T_k/Q_k`` classically
        * ``xi``       : ``2 R_eff S_Phi/(hbar omega0 |M_eff|^2)``
        * ``participation``: ``L_p/L_eff``, the amplitude factor; the scan
          rate carries its square

        With a correlated SQUID (``S_PhiJ != 0``) the noise is reduced by completing the square;
        ``correlated_reduction_resid`` compares that pole
        with the exact ``N_V`` at the noise-doubling points and the call raises above ``corr_tol``.

        ``react_resid``, ``R_noise_over_R_eff``, ``backaction_transfer_ratio`` and
        ``reciprocity_resid`` are the diagnostics that say how good the reduction is: the
        first should be ~0, the next two ~1, the last ~0.  ``R_noise_over_R_eff`` is the Nyquist
        identity referred with the *pickup current* (the flux referral is not an
        identity); ``reciprocity_resid`` is the asymmetry of the solved branch transfer matrix,
        and ``matrix_symmetry_resid`` the tautological assembled-matrix one.  ``port_coupling``
        is ``Q1/sqrt(1 + Q1^2)`` with ``Q1 = coupling_Q1``, not 1.
        """
        if omega0 is None:
            omega0 = 2.0 * math.pi * f_Hz if f_Hz else self.resonance()
        w0 = float(omega0)
        Z0 = self.Z_eff(w0)
        R_eff = Z0.real
        if not (R_eff > 0):
            raise ValueError(
                f"Readout.equivalent: Re Z_eff = {R_eff:.6g} ohm at {w0/2/math.pi:.6g} Hz is not "
                "positive -- the circuit is lossless, so Q and the thermal noise are undefined.  "
                "Give the pickup an R_loss_ohm or a Q, or add a resistor.")
        h = dlog * w0
        dXdw = (self.Z_eff(w0 + h).imag - self.Z_eff(w0 - h).imag) / (2.0 * h)
        L_eff = abs(dXdw) / 2.0
        Q = w0 * L_eff / R_eff

        Hs = self.H_signal(w0)
        if Hs == 0:
            raise ValueError(
                "Readout.equivalent: the pickup EMF produces no flux in the SQUID.  Check the "
                f"input-coil branch name ({self.squid.input_coil!r}) and the mutual inductance.")
        th = self.circuit.thermal_noise(w0, self.flux)
        Hs2 = abs(Hs) ** 2
        N_th = th["total"] / Hs2
        # The Nyquist identity (C4.3) is a statement about the noise referred as a series EMF in
        # the PICKUP branch, i.e. with the *pickup-current* transfer T_{b,p}/T_{pp}.  Referring it
        # with the flux transfer H_b/H_s is not an identity (it fails by orders of magnitude far
        # off resonance in a multi-branch network).
        Ip: Callable[[Solution], complex] = lambda s: s.I(self.pickup)
        th_I = self.circuit.thermal_noise(w0, Ip)
        Tpp2 = abs(self.circuit.transfer(w0, self.pickup, Ip)) ** 2
        R_noise = (sum(R * th_I["transfer2"][nm] / Tpp2
                       for nm, R, _ in self.circuit.noisy_branches(w0)) if Tpp2 > 0 else 0.0)
        n_eff = max(N_th / (R_eff * HBAR * w0) - 1.0, 0.0) / 2.0
        T_eff = T_from_occupation(w0, n_eff)

        Me = self.M_eff(w0)
        xi = self.squid.xi_at(w0, R_eff, abs(Me)) if self.squid.S_Phi > 0 else 0.0
        eta = self.squid.eta_A
        H_in = self.circuit.transfer(w0, self.squid.input_coil, self.flux)
        ba_ratio = abs(H_in) ** 2 / Hs2
        N_ba = ba_ratio * w0 ** 2 * self.squid.M_H ** 2 * self.squid.S_J
        N_imp = self.squid.S_Phi / Hs2
        N_V = N_th + N_ba + N_imp
        # eta_A of the *equivalent* single pole, from N_ba = (hbar w0 R_eff/2 xi) eta_eff^2.
        # With T_bc the current in branch c for a unit EMF in branch b (T_bc = T_cb, the
        # network being reciprocal), this works out to
        #     eta_eff = eta_A |T_pp| |T_in,in| / |T_p,in|^2 = eta_A / port_coupling,
        # so it is the amplifier's eta_A exactly when the pickup and the input coil carry the
        # same current -- the same loop, or an *ideal* transformer, where the 1/n^2 in
        # |H_in/H_s|^2 cancels against the n^2 in |M_eff|^2.  Otherwise, with all the loss in
        # the pickup, port_coupling = Q1/sqrt(1 + Q1^2) with Q1 the quality factor of the loop
        # seen through the coupling with the pickup port open ((C5.1c)): it is 1 only for
        # Q1 >> 1, NOT for every lossless coupling network.
        eta_eff = (math.sqrt(2.0 * xi * N_ba / (HBAR * w0 * R_eff)) if xi > 0 else eta)
        corr = None
        if self.squid.S_PhiJ:
            # Imprecision-backaction correlation (docs/physics/squid_chain.md Sec. 5).  Near the
            # pole the referred cross term is N_c(w0) - hbar w0 R_eff c x, x = 2Q(w - w0)/w0;
            # completing the square shifts the band by x0 = c/xi and replaces
            # eta^2 -> eta_0^2 + 2 xi N_c(w0)/(hbar w0 R_eff) - c^2 (N_c(w0) = 0 for a series loop).
            # port_coupling stays the uncorrelated ratio, so it compares like with like.
            def N_c(w):
                return (self.squid.correlation_term(w, self.circuit.transfer(w, self.squid.input_coil, self.flux))
                        / abs(self.H_signal(w)) ** 2)
            N_c0 = N_c(w0)
            c = -(N_c(w0 + h) - N_c(w0 - h)) / (2.0 * h) / (2.0 * HBAR * w0 * L_eff)
            e2 = eta_eff ** 2 + 2.0 * xi * N_c0 / (HBAR * w0 * R_eff) - c * c
            # The completed square assumes the series-loop curvature xi/2 in x.  A network with a
            # reactive background between pickup and coil (a real flux transformer) adds quadratic
            # correlation and backaction terms, and the reduction then fails: check it against the
            # exact N_V at the noise-doubling points.
            hint = ("the single-pole correlated reduction does not describe this network (reactive "
                    "background between pickup and input coil); integrate the exact output noise "
                    "instead (Readout.noise_psd, or MultiportReadout.output_snr_rate)")
            if not e2 > 0:
                raise ValueError(f"Readout.equivalent: completed-square eta^2 = {e2:.6g} <= 0: {hint}")
            A_c = xi * xi + 2.0 * (2.0 * n_eff + 1.0) * xi + e2
            x0, xs = c / xi, math.sqrt(A_c) / xi
            resid = 0.0
            for x in (x0 - xs, x0 + xs):
                exact = self.noise_psd(w0 * (1.0 + x / (2.0 * Q)))["N_V_V2_s"]
                model = HBAR * w0 * R_eff / (2.0 * xi) * (xi * xi * xs * xs + A_c)
                resid = max(resid, abs(exact / model - 1.0))
            if resid > corr_tol:
                raise ValueError(f"Readout.equivalent: correlated single-pole N_V is off by {resid:.3g} "
                                 f"(> corr_tol = {corr_tol:g}) at the noise-doubling points: {hint}")
            corr = {"eta_A_eff_uncorrelated": eta_eff, "correlation_c": c,
                    "eta_A_uncorrelated": 2.0 * math.sqrt(self.squid.S_Phi * self.squid.S_J) / HBAR,
                    "band_shift_x0": x0, "correlated_reduction_resid": resid,
                    "N_V_correlation_at_omega0_V2_s": N_c0}
            eta_eff = math.sqrt(e2)
            N_V += N_c0
        xo = xi_opt(n_eff, eta_eff)

        out: dict[str, Any] = {
            "label": self.label,
            "omega0_rad_s": w0, "f0_Hz": w0 / (2.0 * math.pi),
            "Z_eff_ohm": complex(Z0), "R_eff_ohm": R_eff, "L_eff_H": L_eff, "Q_loaded": Q,
            "L_p_H": self.L_p_H,
            "participation": (self.L_p_H / L_eff) if (self.L_p_H and L_eff) else float("nan"),
            "n_eff": n_eff, "T_eff_K": T_eff, "T_over_Q_K": T_eff / Q if Q > 0 else float("nan"),
            "xi": xi, "eta_A": eta, "eta_A_eff": eta_eff, "xi_opt": xo,
            "A": A_noise(xi, n_eff, eta_eff) if xi > 0 else float("nan"),
            "G_bar": G_bar(xi, n_eff, eta_eff) if xi > 0 else float("nan"),
            "G_bar_opt": G_bar(xo, n_eff, eta_eff),
            "M_H": self.squid.M_H, "M_eff_H": complex(Me),
            "M_for_xi_opt_H": (Squid.M_for_xi(xo, w0, R_eff, self.squid.S_Phi)
                               * abs(self.squid.M_H / Me) if self.squid.S_Phi > 0 and Me != 0
                               else float("nan")),
            "H_signal_Wb_per_V": complex(Hs),
            "N_V_V2_s": N_V, "N_V_thermal_V2_s": N_th, "N_V_backaction_V2_s": N_ba,
            "N_V_imprecision_V2_s": N_imp,
            "thermal_by_branch_Wb2_s": th["by_branch"],
            "react_resid": abs(Z0.imag) / abs(Z0) if abs(Z0) > 0 else 0.0,
            "R_noise_over_R_eff": R_noise / R_eff,
            "backaction_transfer_ratio": ba_ratio,
            "port_coupling": (corr["eta_A_uncorrelated"] / corr["eta_A_eff_uncorrelated"] if corr else
                              (eta / eta_eff if eta_eff > 0 else float("nan"))),
            "reciprocity_resid": self.circuit.transfer_reciprocity(w0),
            "matrix_symmetry_resid": self.circuit.check_reciprocity(w0),
        }
        # With all the dissipation in the pickup, port_coupling = cos(arg T_in,in) =
        # Q1/sqrt(1 + Q1^2) exactly, Q1 being the quality factor of the loop seen through the
        # coupling with the pickup port open (for a flux transformer Q1 = omega0 M^2/((L_2 +
        # L_in) R)).  It is 1 only for Q1 >> 1 -- NOT for every lossless network.  Reporting Q1 back
        # makes the regime explicit.
        if corr:
            out.update(corr)
        pc = out["port_coupling"]
        out["coupling_Q1"] = (pc / math.sqrt(1.0 - pc * pc) if 0.0 < pc < 1.0
                              else (float("inf") if pc >= 1.0 else float("nan")))
        out["scan_inputs"] = {
            "Psi_SI_H_T_m2": self.Psi_at(w0), "L_H": L_eff, "Q": Q, "T_K": T_eff, "eta_A": eta_eff,
            "note": "scan.reach(Psi_SI, L_H, {'Q': Q, 'T': T_K, 'eta_A': eta_A}); passing L_eff "
                    "rather than L_p carries the (L_p/L_eff)^2 signal-power penalty of [C] F37, "
                    "and eta_A_eff absorbs any transformer stage between pickup and SQUID",
        }
        if xi > 0:
            A = out["A"]
            s = 2.0 * n_eff + 1.0
            out["bandwidth_Hz"] = {
                "linewidth": w0 / Q / (2 * math.pi),
                "noise_doubling": (w0 / Q) * math.sqrt(A) / xi / (2 * math.pi),
                "equivalent": (math.pi / 4) * (w0 / Q) * math.sqrt(A) / xi / (2 * math.pi),
                "thermal_gt_squid": ((w0 / Q)
                                     * math.sqrt(max(2 * xi * s - eta_eff ** 2 - xi ** 2, 0.0))
                                     / xi / (2 * math.pi)),
            }
            out["ringup_time_s"] = 2.0 * Q / w0
            # the single-pole reduction reproduces N_V(omega0) exactly by construction
            # with a correlation the band centre sits at x0 = c/xi, so N_V(omega0) = (A + c^2)/(2 xi)
            out["N_V_single_pole_V2_s"] = HBAR * w0 * R_eff * (A + (corr["correlation_c"] ** 2 if corr else 0.0)) / (2.0 * xi)
            out["N_V_reduction_resid"] = abs(out["N_V_single_pole_V2_s"] / N_V - 1.0)
        return out

    # ------------------------------------------------------------ signal, noise, sensitivity
    def signal(self, omega: float, g_GeV: float = 1e-19, rho_GeV_cm3: float = 0.3) -> dict:
        """Signal amplitudes at ``omega`` for an axion of coupling ``g``.

        The pickup EMF amplitude is ``|V| = omega kappa_a Psi_SI`` (with
        ``Phi_a = kappa_a Psi_SI``).  Returns the EMF, the pickup current, the flux into the
        SQUID in Wb and in ``Phi_0``, and two ratios: ``Phi_SQUID_per_kappa_Wb_H`` and
        ``flux_gain = |Phi_SQ|/Phi_a``.  ``flux_gain`` is a *resonant* gain, not an
        efficiency: on resonance it is ~ Q M/L and is routinely much larger than 1.  The
        dc flux-transfer efficiency of a non-resonant superconducting loop is instead
        ``M/(L_p + L_in)``, which :func:`flux_transformer_circuit` exercises.
        """
        kappa = kappa_a_SI(g_GeV, rho_GeV_cm3)
        Phi_a = kappa * self.Psi_at(omega)
        V = complex(0.0, -omega) * Phi_a
        s = self.circuit.solve_many(omega, [{self.pickup: V}], replace=True)[0]
        Phi_SQ = self.flux(s)
        return {
            "omega_rad_s": omega, "f_Hz": omega / (2.0 * math.pi),
            "g_GeV": g_GeV, "kappa_a_inv_H": kappa, "Phi_a_Wb": Phi_a,
            "V_emf_V": V, "I_pickup_A": s.I(self.pickup),
            "Phi_SQUID_Wb": Phi_SQ, "Phi_SQUID_Phi0": abs(Phi_SQ) / PHI0,
            "Phi_SQUID_per_kappa_Wb_H": abs(Phi_SQ) / kappa if kappa else float("nan"),
            "flux_gain": abs(Phi_SQ) / Phi_a if Phi_a else float("nan"),
            "H_signal_Wb_per_V": complex(self.H_signal(omega)),
        }

    def noise_psd(self, omega: float) -> dict:
        """Noise PSDs at ``omega``, referred both to the SQUID flux [Wb^2 s] and to a series
        EMF in the pickup branch [V^2 s], so that ``s = S_Va/N_V`` exactly."""
        Hs2 = abs(self.H_signal(omega)) ** 2
        th = self.circuit.thermal_noise(omega, self.flux)
        H_in = self.circuit.transfer(omega, self.squid.input_coil, self.flux)
        ba = abs(H_in) ** 2 * omega ** 2 * self.squid.M_H ** 2 * self.squid.S_J
        imp = self.squid.S_Phi
        cross = self.squid.correlation_term(omega, H_in)
        tot = th["total"] + ba + imp + cross
        return {
            "omega_rad_s": omega, "f_Hz": omega / (2.0 * math.pi),
            "S_Phi_thermal_Wb2_s": th["total"], "S_Phi_backaction_Wb2_s": ba,
            "S_Phi_imprecision_Wb2_s": imp, "S_Phi_correlation_Wb2_s": cross, "S_Phi_total_Wb2_s": tot,
            "thermal_by_branch_Wb2_s": th["by_branch"],
            "N_V_thermal_V2_s": th["total"] / Hs2, "N_V_backaction_V2_s": ba / Hs2,
            "N_V_imprecision_V2_s": imp / Hs2, "N_V_correlation_V2_s": cross / Hs2, "N_V_V2_s": tot / Hs2,
            "Z_eff_ohm": self.Z_eff(omega),
        }

    def sweep(self, f_Hz: Sequence[float], g_GeV: float = 1e-19,
              rho_GeV_cm3: float = 0.3) -> dict[str, np.ndarray]:
        """Signal and noise versus frequency (one tuning): ``|Z_eff|``, ``|H_signal|``,
        ``Phi_SQUID``, and the EMF-referred ``N_V`` and its parts (the correlation part is 0
        for an uncorrelated SQUID)."""
        f = np.asarray(f_Hz, dtype=float)
        keys = ("Z_abs_ohm", "Z_re_ohm", "Z_im_ohm", "H_signal_Wb_per_V", "Phi_SQUID_Wb",
                "N_V_V2_s", "N_V_thermal_V2_s", "N_V_backaction_V2_s", "N_V_imprecision_V2_s",
                "N_V_correlation_V2_s")
        out = {k: np.zeros(f.size) for k in keys}
        out["f_Hz"] = f
        for i, fi in enumerate(f):
            w = 2.0 * math.pi * float(fi)
            Z = self.Z_eff(w)
            n = self.noise_psd(w)
            s = self.signal(w, g_GeV, rho_GeV_cm3)
            out["Z_abs_ohm"][i] = abs(Z)
            out["Z_re_ohm"][i] = Z.real
            out["Z_im_ohm"][i] = Z.imag
            out["H_signal_Wb_per_V"][i] = abs(s["H_signal_Wb_per_V"])
            out["Phi_SQUID_Wb"][i] = abs(s["Phi_SQUID_Wb"])
            for k in ("N_V_V2_s", "N_V_thermal_V2_s", "N_V_backaction_V2_s",
                      "N_V_imprecision_V2_s", "N_V_correlation_V2_s"):
                out[k][i] = n[k]
        return out

    def noise_curve(self, omega0: float | None = None, *, half_widths: float = 60.0,
                    decades: float = 3.0, points_per_decade: float = 48.0,
                    extra_omega: Sequence[float] | None = None,
                    omega: Sequence[float] | None = None) -> dict[str, np.ndarray | float]:
        """The exact ``N_V(omega)`` of this setting tabulated for the finite-band kernel.

        The finite-band kernel needs ``N_V(omega)`` of each *setting* (one tuning capacitance, one
        coupling), not the single-pole reduction of it; this returns it on a grid that the C++
        ``kernel_row_noise`` interpolates log-log.  The grid is symmetric in the detuning,

            ``omega = omega0`` and ``omega0 +- hw x``,  ``x`` geometric over
            ``[10^-decades, half_widths]`` at ``points_per_decade`` points per decade,

        with ``hw`` half the noise-doubling width of the equivalent pole, so it resolves the
        resonance (where ``N_V`` is flat, so the ``x < 10^-decades`` gap costs nothing) and the
        ``|Z|^2 ~ (2 L domega)^2`` wings (where ``N_V`` is a power law, hence exact in log-log).
        ``half_widths = 60`` covers the window the discrete sums use (+-40
        half-widths truncate below 1e-5).  The interpolation error of ``N_V`` is second order in
        the step, ``~(ln 10/ppd)^2/8``, so 48 points per decade keeps ``K`` (which goes as
        ``1/N_V^2``) accurate to a few times ``10^-4``; halve ``points_per_decade`` to check.

        ``extra_omega`` merges extra frequencies into the grid.  Use it for structure that does
        *not* scale with the detuning -- above all a self-resonance of ``Z_p`` (a pole at a fixed
        frequency), which a grid centred on ``omega_0`` resolves only by accident.

        Pass ``omega=`` to use your own grid instead.  Returns ``omega_rad_s``, ``f_Hz``,
        ``N_V_V2_s`` and its components (thermal, backaction, imprecision, correlation), plus ``omega0_rad_s``, ``f0_Hz`` and
        ``N_V_at_omega0_V2_s``.
        """
        w0 = float(omega0) if omega0 is not None else self.resonance()
        if omega is not None:
            w = np.unique(np.asarray(omega, dtype=float))
            if w.size < 2 or np.any(w <= 0):
                raise ValueError("Readout.noise_curve: omega must hold >= 2 positive values")
        else:
            if not (half_widths > 0 and decades > 0 and points_per_decade >= 4):
                raise ValueError("Readout.noise_curve: need half_widths > 0, decades > 0 and "
                                 "points_per_decade >= 4")
            eq = self.equivalent(w0)
            hw = 0.5 * 2.0 * math.pi * eq["bandwidth_Hz"]["noise_doubling"] if eq.get("bandwidth_Hz") \
                else 0.5 * w0 / eq["Q_loaded"]
            if not (hw > 0):
                raise ValueError("Readout.noise_curve: the equivalent pole has zero bandwidth "
                                 "(no loss and no amplifier noise?)")
            n_side = max(int(round((decades + math.log10(half_widths)) * points_per_decade)), 4)
            x = np.geomspace(10.0 ** -decades, half_widths, n_side)
            w = np.unique(np.concatenate([w0 - hw * x[::-1], [w0], w0 + hw * x]))
            w = w[w > 0]
        if extra_omega is not None:
            e = np.asarray(extra_omega, dtype=float).ravel()
            if np.any(e <= 0):
                raise ValueError("Readout.noise_curve: extra_omega must be positive")
            w = np.unique(np.concatenate([w, e]))
        keys = ("N_V_V2_s", "N_V_thermal_V2_s", "N_V_backaction_V2_s", "N_V_imprecision_V2_s",
                "N_V_correlation_V2_s")
        vals = {k: np.zeros(w.size) for k in keys}
        for i, wi in enumerate(w):
            d = self.noise_psd(float(wi))
            for k in keys:
                vals[k][i] = d[k]
        if not np.all(vals["N_V_V2_s"] > 0):
            raise ValueError("Readout.noise_curve: N_V is not positive everywhere on the grid; "
                             "give the circuit some loss (R_loss_ohm / Q) or the SQUID an S_Phi")
        out: dict[str, Any] = {"omega_rad_s": w, "f_Hz": w / (2.0 * math.pi), **vals}
        out["omega0_rad_s"] = w0
        out["f0_Hz"] = w0 / (2.0 * math.pi)
        out["N_V_at_omega0_V2_s"] = float(self.noise_psd(w0)["N_V_V2_s"])
        return out

    def kernel(self, omega0: float | None = None, g_GeV: float = 1e-19, Q_axion: float = 6.8e5,
               rho_GeV_cm3: float = 0.3, Theta: float = 2.71) -> dict:
        """Narrow-line matched-filter kernel at resonance, in SI:

            K = (pi/8) Q_a omega_a^3 Phi_a^4/N_V(omega_a)^2    [1/s]

        (``Phi_a`` in Wb, ``N_V`` in V^2 s; no ``hbar`` appears because
        ``Wb^4 s^-3/(V^2 s)^2 = s^-1``).  ``K`` is the Asimov TS gained per second of dwell at
        coupling ``g`` for an axion on resonance, and ``dwell_for_Theta_s = Theta/K`` is the
        time one tuning needs for a 95% limit there.  Unlike the scan *rate*, ``K`` does not
        involve ``L``: a bigger total inductance narrows the band without changing the
        on-resonance SNR."""
        w0 = float(omega0) if omega0 is not None else self.resonance()
        Phi_a = kappa_a_SI(g_GeV, rho_GeV_cm3) * self.Psi_at(w0)
        N_V = self.noise_psd(w0)["N_V_V2_s"]
        K = (math.pi / 8.0) * Q_axion * w0 ** 3 * Phi_a ** 4 / (N_V ** 2) if N_V > 0 else 0.0
        return {"omega0_rad_s": w0, "f0_Hz": w0 / (2 * math.pi), "g_GeV": g_GeV,
                "Phi_a_Wb": Phi_a, "N_V_V2_s": N_V, "K_per_s": K,
                "dwell_for_Theta_s": Theta / K if K > 0 else float("inf")}

    # ------------------------------------------------------------ the whole story
    def analyze(self, f_Hz: float | None = None, tune: bool = True, g_GeV: float | None = None,
                m_a_eV: float | None = None, rho_GeV_cm3: float = 0.3, match_xi: bool = False,
                reach: bool = False, **reach_kw) -> dict:
        """Tune to ``f_Hz`` (if there is a tuning capacitor), then report everything:
        the resonance, the loaded Q, the equivalent single-pole detector, the signal flux,
        the noise budget, the bandwidths and the scan inputs.

        ``g_GeV``/``m_a_eV`` default to the QCD (DFSZ) line at the resonant frequency.
        ``match_xi=True`` re-scales ``M`` to ``xi_opt`` at the resonance (what the optimal
        scan assumes).  ``reach=True`` also calls :func:`axionfem.scan.reach`.
        """
        C = self.tune(f_Hz) if (f_Hz is not None and tune and self.tuner is not None) else None
        w0 = self.resonance(f_Hz) if f_Hz is None else 2.0 * math.pi * f_Hz
        if match_xi:
            self.match_xi(w0)
        eq = self.equivalent(w0)
        if C is not None:
            eq["C_tuned_F"] = C
        if g_GeV is None:
            from . import scan as _scan
            m = m_a_eV if m_a_eV is not None else _scan.freq_to_mass(eq["f0_Hz"])
            g_GeV = float(_scan.g_qcd([m])[0])
            eq["m_a_eV"] = m
            eq["g_target_GeV"] = g_GeV
        eq["signal"] = self.signal(w0, g_GeV, rho_GeV_cm3)
        eq["noise"] = self.noise_psd(w0)
        eq["kernel"] = self.kernel(w0, g_GeV, rho_GeV_cm3=rho_GeV_cm3)
        if reach:
            from . import scan as _scan
            # eta_A_eff, not eta_A: the scan inputs are (Psi, L_eff, Q, T_eff, eta_A_eff) --
            # (C5.1a)/(C5.2) and the scan_inputs block below.  A coupling network with
            # port_coupling < 1 (weak k, low pickup Q) makes eta_A_eff >> eta_A, and using
            # eta_A there overstates the reach (by 60 % in g for a flux transformer with Q_p = 100,
            # k = 0.02).
            eq["reach"] = _scan.reach(self.Psi_at(w0), eq["L_eff_H"],
                                      {"Q": eq["Q_loaded"], "T": eq["T_eff_K"],
                                       "eta_A": eq["eta_A_eff"]}, **reach_kw)
        return eq

    def match_xi(self, omega0: float | None = None, xi: float | None = None) -> float:
        """Scale ``M`` so that the readout coupling is ``xi`` (default ``xi_opt``) at
        ``omega0``; returns the new ``M`` [H].  ``S_Phi`` and ``S_J`` (hence ``eta_A``) are
        amplifier properties and are left alone."""
        w0 = float(omega0) if omega0 is not None else self.resonance()
        eq = self.equivalent(w0)
        if self.squid.S_Phi <= 0:
            raise ValueError("Readout.match_xi: the SQUID has no flux noise (S_Phi = 0), "
                             "so xi is undefined")
        target = xi if xi is not None else eq["xi_opt"]
        if not (target > 0):
            raise ValueError(f"Readout.match_xi: xi must be > 0 (got {target})")
        # xi ~ 1/|M_eff|^2 and M_eff ~ M, so M_new = M sqrt(xi_now/xi_target)
        self.squid.M_H *= math.sqrt(eq["xi"] / target)
        return self.squid.M_H

    def summary(self, f_Hz: float | None = None, **kw) -> str:
        a = self.analyze(f_Hz, **kw)
        s, n = a["signal"], a["noise"]
        L = [f"readout {a['label']!r} at f0 = {a['f0_Hz']:.6g} Hz"]
        if "C_tuned_F" in a:
            L.append(f"  C_tune   = {a['C_tuned_F']:.4g} F")
        L += [f"  L_eff    = {a['L_eff_H']:.4g} H  (L_p = {a['L_p_H']:.4g} H, "
              f"participation L_p/L_eff = {a['participation']:.4g})",
              f"  R_eff    = {a['R_eff_ohm']:.4g} ohm   Q_loaded = {a['Q_loaded']:.6g}",
              f"  T_eff    = {a['T_eff_K']:.4g} K  (n_eff = {a['n_eff']:.4g}), "
              f"T/Q = {a['T_over_Q_K']:.4g} K",
              f"  xi       = {a['xi']:.4g}  (xi_opt = {a['xi_opt']:.4g}), eta_A = {a['eta_A']:.4g}"
              + ("" if abs(a['eta_A_eff'] / a['eta_A'] - 1) < 1e-9
                 else f" (eta_A_eff = {a['eta_A_eff']:.4g})"),
              f"  G_bar    = {a['G_bar']:.4g}  (G_bar_opt = {a['G_bar_opt']:.4g})",
              f"  M        = {a['M_H']:.4g} H  (M_eff = {abs(a['M_eff_H']):.4g} H, "
              f"M for xi_opt = {a['M_for_xi_opt_H']:.4g} H)",
              f"  signal   = {abs(s['Phi_SQUID_Wb']):.4g} Wb = {s['Phi_SQUID_Phi0']:.4g} Phi_0 "
              f"at g = {s['g_GeV']:.4g} GeV^-1  (resonant flux gain {s['flux_gain']:.4g})",
              f"  noise    N_V = {n['N_V_V2_s']:.4g} V^2 s = thermal {n['N_V_thermal_V2_s']:.4g} "
              f"+ backaction {n['N_V_backaction_V2_s']:.4g} "
              f"+ imprecision {n['N_V_imprecision_V2_s']:.4g}"]
        for k, v in a["thermal_by_branch_Wb2_s"].items():
            L.append(f"      thermal from {k}: {v / a['noise']['S_Phi_thermal_Wb2_s'] * 100:.3g} %")
        if a.get("bandwidth_Hz"):
            bw = a["bandwidth_Hz"]
            L.append(f"  bandwidth  linewidth {bw['linewidth']:.4g} Hz, noise-doubling "
                     f"{bw['noise_doubling']:.4g} Hz, ring-up {a['ringup_time_s']:.4g} s")
        L.append(f"  kernel   K = {a['kernel']['K_per_s']:.4g} /s -> dwell for Theta95 = "
                 f"{a['kernel']['dwell_for_Theta_s']:.4g} s")
        si = a["scan_inputs"]
        L.append(f"  scan inputs: Psi = {si['Psi_SI_H_T_m2']:.4g} H T m^2, L = {si['L_H']:.4g} H, "
                 f"Q = {si['Q']:.6g}, T = {si['T_K']:.4g} K, eta_A = {si['eta_A']:.4g}")
        L.append(f"  diagnostics: react_resid {a['react_resid']:.3g}, R_noise/R_eff "
                 f"{a['R_noise_over_R_eff']:.6g}, backaction transfer "
                 f"{a['backaction_transfer_ratio']:.6g}, reciprocity {a['reciprocity_resid']:.3g}, "
                 f"N_V reduction {a.get('N_V_reduction_resid', float('nan')):.3g}")
        return "\n".join(L)

    def draw(self, annotate: bool = True, *, f_Hz: float | None = None, title: str | None = None):
        """Effective-circuit schematic of this readout (``schemdraw``; install ``axionfem[draw]``):
        the network exactly as built -- the pickup branch drawn as its EMF and series impedance
        (``R_loss``, ``L_p`` when ``Z_p`` has the MQS form, a ``Z_p(omega)`` box otherwise) -- with
        the SQUID coupled to its input coil by ``M``.  ``annotate=True`` adds the element values
        (resistances with their noise temperatures, ``C`` and ``tan delta``, ``L_in``, ``M``), the
        SQUID's noise and ``xi`` against ``xi_opt`` at ``f_Hz`` (default: the resonance);
        ``annotate=False`` gives the symbolic figure.
        Returns a ``schemdraw.Drawing``."""
        from .draw import draw_readout
        return draw_readout(self, annotate=annotate, f_Hz=f_Hz, title=title)

    def __repr__(self) -> str:   # pragma: no cover
        kind = "Psi(omega)" if callable(self.Psi_SI) else f"{float(self.Psi_SI):.4g}"
        return f"<Readout {self.label!r}: L_p = {self.L_p_H:.4g} H, Psi = {kind}>"


# ----------------------------------------------------------------- standard topologies
def series_readout(L_p_H: float | None = None, *, Z_p: ZFunc | None = None,
                   Psi_SI: float | Callable[[float], float] = 0.0,
                   f_Hz: float = 1.209e5, Q: float | None = None, R_loss_ohm: float = 0.0,
                   T_K: float = 0.02, C_F: float | None = None, tan_delta: float = 0.0,
                   T_C_K: float | None = None, C_par_F: float = 0.0,
                   tan_delta_par: float = 0.0, T_par_K: float | None = None,
                   L_in_H: float = 0.0, R_in_ohm: float = 0.0, T_in_K: float | None = None,
                   L_lead_H: float = 0.0, R_lead_ohm: float = 0.0, T_lead_K: float | None = None,
                   lead_length_m: float = 0.0, L_per_m: float = 0.0, R_per_m: float = 0.0,
                   M_H: float | None = None, k_in: float | None = None, L_squid_H: float = 0.0,
                   eta_A: float = 1.0, xi: float | None = None, S_Phi: float | None = None,
                   transformer_ratio: float | None = None, L_1_H: float = 0.0, L_2_H: float = 0.0,
                   k_T: float = 1.0, R_2_ohm: float = 0.0, T_2_K: float | None = None,
                   label: str | None = None) -> Readout:
    """DMRadio-style readout: pickup in series with a tuning capacitor and the SQUID input coil.

    Plain series topology (the default)::

        gnd -[ pickup: EMF, Z_p ]- a -[ L_lead ]- b -[ R_lead ]- c
            -[ C (tan_delta) ]- d -[ R_in ]- e -[ L_in ]- gnd          (M couples L_in -> SQUID)

    With ``transformer_ratio`` an ideal transformer closes the resonant loop and the input
    coil (plus ``R_2_ohm``) sits on its secondary; with ``L_1_H`` and ``L_2_H`` a *real* flux
    transformer (two inductors coupled by ``k_T``) does the same job with finite inductance
    and leakage.

    Parameters
    ----------
    L_p_H, Z_p
        the pickup, either as an inductance (``Z_p = R_loss - i omega L_p``, ``R_loss`` from
        ``R_loss_ohm`` or from ``Q`` at ``f_Hz``) or as a callable ``Z_p(omega)`` -- e.g.
        :func:`pickup_impedance` built from a solved design's full-wave sweep.  With ``Z_p``
        the pickup inductance used for the participation ratio is ``-Im Z_p(2 pi f_Hz)/omega``
        unless ``L_p_H`` is also given.
    Psi_SI
        the design's signal coupling [T H m^2]; the EMF amplitude is ``omega kappa_a Psi_SI``.
        A callable ``Psi_SI(omega)`` -- e.g. :func:`pickup_coupling` built from the design's
        full-wave sweep -- makes the coupling frequency dependent.
    C_F
        the tuning capacitance; ``None`` (default) tunes it to resonate at ``f_Hz``.
    C_par_F
        a parasitic capacitance across the pickup terminals -- e.g. the solver's
        ``results["pickup"]["port_capacitance"]["C_gap_F"]``, which sets the self-resonance
        ``f_SR = 1/(2 pi sqrt(L_p C_gap))``.  The loop is then not a plain series chain, and
        the reduction diagnostics (``backaction_transfer_ratio``, ``react_resid``) show it.
    L_lead_H, R_lead_ohm, lead_length_m, L_per_m, R_per_m
        the leads: either ``L_lead_H``/``R_lead_ohm``, or ``lead_length_m`` with ``L_per_m`` [H/m] and
        ``R_per_m`` [ohm/m].
    T_K, T_C_K, T_in_K, T_lead_K, T_2_K
        temperatures: ``T_K`` is the default for every dissipative element; ``T_C_K``, ``T_in_K``,
        ``T_lead_K`` and ``T_2_K`` override it per element (the referred ``T_eff`` then obeys
        ``T_eff/Q = sum_k T_k/Q_k``).
    M_H, k_in, L_squid_H, S_Phi
        SQUID coupling: ``M_H`` directly; or ``k_in`` with ``L_squid_H`` (``M = k sqrt(L_in L_squid)``); or
        neither, in which case ``M`` is derived from ``xi`` (default: ``xi_opt`` at the
        resonance, which is what the optimal scan assumes).  ``S_Phi`` [Wb^2 s] fixes the
        amplifier's flux noise; left out, it is derived from ``xi`` with the reference
        ``M = L_in`` (``k = 1`` to a SQUID of the input coil's inductance), or ``M = L_eff``
        when there is no input coil.  **Only the absolute flux in Wb depends on that
        reference**; ``Q``, ``T_eff``, ``xi``, ``G-bar``, ``N_V`` and the reach do not.
    """
    if L_p_H is None and Z_p is None:
        raise ValueError("series_readout: give L_p_H (an MQS pickup inductance) or Z_p (a "
                        "callable pickup impedance Z_p(omega), e.g. from pickup_impedance())")
    if not (f_Hz > 0):
        raise ValueError(f"series_readout: f_Hz must be > 0 (got {f_Hz})")
    if Q is not None and R_loss_ohm:
        raise ValueError("series_readout: give either Q or R_loss_ohm for the pickup, not both")
    if transformer_ratio is not None and (L_1_H or L_2_H):
        raise ValueError("series_readout: give transformer_ratio or (L_1_H, L_2_H), not both")
    if bool(L_1_H) != bool(L_2_H):
        raise ValueError("series_readout: a real flux transformer needs both L_1_H and L_2_H")
    if k_in is not None and M_H is not None:
        raise ValueError("series_readout: give M_H or k_in, not both")

    w0g = 2.0 * math.pi * f_Hz
    if Z_p is None:
        L_p = float(L_p_H)
        if L_p <= 0:
            raise ValueError(f"series_readout: L_p_H must be > 0 (got {L_p_H})")
        if Q is not None:
            if Q <= 0:
                raise ValueError(f"series_readout: Q must be > 0 (got {Q})")
            Rp = w0g * L_p / Q
        else:
            Rp = float(R_loss_ohm)
        Zp: ZFunc = lambda w, R=Rp, L=L_p: complex(R, -w * L)   # noqa: E731
    else:
        Zp = Z_p
        L_p = float(L_p_H) if L_p_H else -Zp(w0g).imag / w0g
        if L_p <= 0:
            raise ValueError(
                f"series_readout: the pickup is capacitive at {f_Hz:.6g} Hz "
                f"(-Im Z_p/omega = {L_p:.4g} H <= 0).  Above the pickup's self-resonance a "
                "series tuning capacitor cannot resonate the loop; a tunable inductor is "
                "needed instead (2211.00008 Sec. I).")

    if lead_length_m:
        L_lead_H = L_lead_H or lead_length_m * L_per_m
        R_lead_ohm = R_lead_ohm or lead_length_m * R_per_m
    T_in_K = T_K if T_in_K is None else T_in_K
    T_lead_K = T_K if T_lead_K is None else T_lead_K
    T_C_K = T_K if T_C_K is None else T_C_K
    T_2_K = T_K if T_2_K is None else T_2_K
    real_T = bool(L_1_H and L_2_H)

    if not (Zp(w0g).real > 0 or R_lead_ohm > 0 or R_in_ohm > 0 or R_2_ohm > 0 or tan_delta > 0
            or (C_par_F and tan_delta_par > 0)):
        raise ValueError(
            "series_readout: the circuit has no dissipation, so the series resonance is a pole "
            "(infinite current) and Q, T_eff and xi are all undefined.  Give the pickup a Q= or "
            "an R_loss_ohm=, or add R_lead_ohm / R_in_ohm / R_2_ohm / tan_delta.  With a "
            "Z_p callable, make sure Re Z_p > 0 at the tuning frequency.")

    c = Circuit()
    c.source("pickup", "gnd", "a", emf=0.0, Z=Zp, T_K=T_K)
    if C_par_F:
        # parasitic capacitance across the pickup terminals (e.g. the solver's
        # pickup/port_capacitance/C_gap_F): the loop is no longer a plain series chain
        c.capacitor("C_par", "a", "gnd", C_par_F,
                    T_K=T_K if T_par_K is None else T_par_K, tan_delta=tan_delta_par)
    node = "a"
    if L_lead_H:
        c.inductor("L_lead", node, "b", L_lead_H)
        node = "b"
    if R_lead_ohm:
        c.resistor("R_lead", node, "c", R_lead_ohm, T_K=T_lead_K)
        node = "c"
    L_guess = L_p + L_lead_H + (L_in_H if (transformer_ratio is None and not real_T) else 0.0) \
        + (L_1_H if real_T else 0.0)
    c.capacitor("C", node, "d", C_F if C_F else 1.0 / (w0g ** 2 * L_guess),
                T_K=T_C_K, tan_delta=tan_delta)
    node = "d"

    if transformer_ratio is not None:
        c.transformer("T", node, "gnd", "s1", "gnd", transformer_ratio)
        sec = "s1"
        if R_2_ohm:
            c.resistor("R_2", sec, "s2", R_2_ohm, T_K=T_2_K)
            sec = "s2"
        if R_in_ohm:
            c.resistor("R_in", sec, "s3", R_in_ohm, T_K=T_in_K)
            sec = "s3"
        if L_in_H:
            c.inductor("L_in", sec, "gnd", L_in_H)
        else:
            c.short("L_in", sec, "gnd")
    elif real_T:
        c.inductor("L_1", node, "gnd", L_1_H)
        sec = "s1"
        c.inductor("L_2", "gnd", sec, L_2_H)
        c.mutual("L_1", "L_2", k=k_T)
        if R_2_ohm:
            c.resistor("R_2", sec, "s2", R_2_ohm, T_K=T_2_K)
            sec = "s2"
        if R_in_ohm:
            c.resistor("R_in", sec, "s3", R_in_ohm, T_K=T_in_K)
            sec = "s3"
        if L_in_H:
            c.inductor("L_in", sec, "gnd", L_in_H)
        else:
            c.short("L_in", sec, "gnd")
    else:
        if R_in_ohm:
            c.resistor("R_in", node, "e", R_in_ohm, T_K=T_in_K)
            node = "e"
        if L_in_H:
            c.inductor("L_in", node, "gnd", L_in_H)
        else:
            c.short("L_in", node, "gnd")   # no input coil: the loop current threads the SQUID

    sq = Squid(M_H=M_H if M_H is not None else 1.0, eta_A=eta_A, S_Phi=1.0, S_J=1.0,
               input_coil="L_in")
    r = Readout(circuit=c, squid=sq, pickup="pickup", Psi_SI=Psi_SI, L_p_H=L_p,
                label=label or "DMRadio-style series readout", tuner="C")
    if C_F is None:
        r.tune(f_Hz)
    w0 = 2.0 * math.pi * f_Hz

    # --- amplifier: fix (M, S_Phi, S_J) at the true resonance, from eta_A and xi
    Z0 = r.Z_eff(w0)
    R0 = Z0.real
    if R0 <= 0:
        raise ValueError(
            "series_readout: the circuit has no loss, so xi, Q and the thermal noise are "
            "undefined.  Give the pickup a Q or an R_loss_ohm, or add R_in_ohm/R_lead_ohm/"
            "tan_delta.")
    ratio = abs(r.circuit.transfer(w0, "pickup", lambda s: s.I("L_in"))
                / r.circuit.transfer(w0, "pickup", lambda s: s.I("pickup")))  # |I_in/I_p|
    n0 = n_bose(w0, T_K)
    xi_use = xi if xi is not None else xi_opt(n0, eta_A)
    if not (xi_use > 0):
        raise ValueError(f"series_readout: xi must be > 0 (got {xi_use})")
    if k_in is not None:
        if not (L_in_H and L_squid_H):
            raise ValueError("series_readout: k_in needs L_in_H and L_squid_H")
        if abs(k_in) > 1:
            raise ValueError(f"series_readout: |k_in| must be <= 1 (got {k_in})")
        M_H = k_in * math.sqrt(L_in_H * L_squid_H)
    if S_Phi is None:
        # reference mutual when the user fixed neither M nor S_Phi: k = 1 to a SQUID whose
        # inductance equals the input coil's, or the loop's own inductance if there is none.
        # Only the absolute flux in Wb depends on this; xi, Q, T_eff, G-bar and N_V do not.
        M_ref = abs(M_H) if M_H is not None else (L_in_H or L_guess)
        if M_ref <= 0:
            M_ref = 1e-9
        # M_eff = M * ratio, and xi = 2 R0 S_Phi/(hbar w0 M_eff^2)
        S_Phi = HBAR * xi_use * w0 * (M_ref * max(ratio, 1e-300)) ** 2 / (2.0 * R0)
    if S_Phi <= 0:
        raise ValueError(f"series_readout: S_Phi must be > 0 (got {S_Phi})")
    if M_H is None:
        if ratio <= 0:
            raise ValueError("series_readout: no current reaches the input coil at the resonance")
        M_H = Squid.M_for_xi(xi_use, w0, R0, S_Phi) / ratio
    sq.M_H = float(M_H)
    sq.S_Phi = float(S_Phi)
    sq.S_J = (HBAR * eta_A / 2.0) ** 2 / float(S_Phi)
    return r


def flux_transformer_circuit(L_1_H: float, L_2_H: float, k: float = 1.0, L_load_H: float = 0.0,
                             R_1_ohm: float = 0.0, R_2_ohm: float = 0.0) -> tuple[Circuit, float]:
    """A flux transformer: a primary loop ``L_1`` driven by branch ``"drive"`` and coupled by
    ``k`` to a secondary ``L_2`` in series with ``L_load``.  Returns the circuit and
    ``M = k sqrt(L_1 L_2)``.

    Driving ``"drive"`` with ``emf = -i omega Phi`` applies a flux ``Phi`` to the primary.  With
    no resistance, the solve reproduces fluxoid conservation in two superconducting loops
    (``L_1 I_1 + M I_2 = Phi``, ``M I_1 + (L_2 + L_load) I_2 = 0``) at every frequency, so the
    currents are frequency-independent:

        I_1 = Phi (L_2 + L_load)/D,   I_2 = -Phi M/D,   D = L_1 (L_2 (1 - k^2) + L_load)

    and a SQUID coupled to ``L_load`` by ``M_sq`` sees ``|Phi_SQ| = M_sq |Phi| M/D``.  (The
    overall sign follows the EMF-vs-flux convention; magnitudes do not.)  ``k = 1`` with
    ``L_load = 0`` is genuinely singular (``D = 0``): a perfectly coupled, shorted secondary is a
    short across the primary.

    Used by the validation suite; real readouts should use :func:`series_readout`.
    """
    if not (0.0 <= abs(k) <= 1.0):
        raise ValueError(f"flux_transformer_circuit: |k| must be <= 1 (got {k})")
    if abs(k) == 1.0 and L_load_H == 0.0 and R_2_ohm == 0.0:
        raise ValueError(
            "flux_transformer_circuit: k = 1 with L_load_H = 0 and no secondary resistance is "
            "singular (a perfectly coupled, shorted secondary shorts the primary).  Give "
            "L_load_H > 0 (the SQUID input coil), or |k| < 1.")
    c = Circuit()
    M = k * math.sqrt(L_1_H * L_2_H)
    c.source("drive", "gnd", "a", emf=0.0, Z=0.0)
    node = "a"
    if R_1_ohm:
        c.resistor("R_1", node, "a2", R_1_ohm, T_K=0.0)
        node = "a2"
    c.inductor("L_1", node, "gnd", L_1_H)
    sec = "b"
    c.inductor("L_2", "gnd", sec, L_2_H)
    c.mutual("L_1", "L_2", M=M)
    if R_2_ohm:
        c.resistor("R_2", sec, "b2", R_2_ohm, T_K=0.0)
        sec = "b2"
    if L_load_H:
        c.inductor("L_load", sec, "gnd", L_load_H)
    else:
        c.short("L_load", sec, "gnd")
    return c, M


# ----------------------------------------------------------------- S/Y/Z parameters and Touchstone files
def write_touchstone(results: Mapping[str, Any], path, kind: str = "S", z0: float = 50.0, dense: bool = False,
                     **kw) -> dict[str, Any]:
    """Write a full-wave sweep's port parameters as a Touchstone v1 ``.sNp`` file in the engineering convention
    ``e^{+j omega t}`` (the engine's ``e^{-i omega t}`` Z is conjugated).  See
    :func:`axionfem.touchstone.write_touchstone` (``surface_losses``, ``passivity_tol``) and
    docs/physics/touchstone.md."""
    from . import touchstone as _ts
    return _ts.write_touchstone(results, path, kind, z0, dense, **kw)


def read_touchstone(path, nports: int | None = None) -> dict[str, Any]:
    """Read a Touchstone v1 file (S, Z or Y): ``f_Hz``, ``S``, ``Z``, ``data``, ``z0_ohm``, ``ports``
    (:func:`axionfem.touchstone.read_touchstone`)."""
    from . import touchstone as _ts
    return _ts.read_touchstone(path, nports)


def port_parameters(results: Mapping[str, Any], *, dense: bool = False, surface_losses: bool = False) -> dict:
    """A full-wave sweep's port impedance matrix in the engineering convention, sorted by frequency
    (:func:`axionfem.touchstone.port_parameters`; S and Y from :func:`axionfem.touchstone.z_to_s` /
    :func:`axionfem.touchstone.z_to_y` with ``convention="engineering"``)."""
    from . import touchstone as _ts
    return _ts.port_parameters(results, dense=dense, surface_losses=surface_losses)
