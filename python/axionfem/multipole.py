"""Multi-band and multipole scans of a full-wave detector response Z(omega), V(omega).  Beta.

Two strategies, both specified in ``docs/physics/multipole_scan.md`` (equation tags ``M*``; the sources [R], [P] and
[C] are defined there):

1. **Multi-band scan** of a reference implementation of the multipole scan strategy of
   arXiv:2211.00008 ([R], a Python notebook, not distributed; the options that follow its conventions
   are named ``notebook*``; see the spec for what the comparison with [P]'s figures shows):
   the tunable bands are the frequency intervals in which a capacitor (source inductive) or
   an inductor (source capacitive) of a realistic value range resonates the source
   (:func:`valid_intervals`, M1-M2); in each band the local scan rate of arXiv:2302.14084 with
   the reactance-slope inductance ``L_eff`` is used (:func:`scan_rate`, M3-M5); a time budget is
   spent on the cheapest frequency chunks of all bands together (:func:`greedy_allocation`,
   M6) or at uniform dwell (:func:`uniform_dwell_reach`, M7).  :func:`multiband_scan` runs the
   chain.

2. **The 2211.00008 single-pole and multipole sensitivity** reconstructed from the text of
   arXiv:2211.00008 ([P]) and validated against its figures (:func:`single_pole_ratio`, :func:`mqs_schedule`,
   :func:`multipole_theta`, :func:`reach_2211`; M8-M13): the MQS scan schedule from
   ``f_MQS`` downwards (one capacitive tuning per ``delta f/f = 1e-6``, dwell until the MQS
   on-resonance sensitivity reaches the target), evaluated with the full-wave source on
   resonance (single pole) or with the full response of every tuning at every frequency
   (multipole: all poles of each tuning plus the off-resonant response).

Units: SI (Hz, ohm, H, V, K, s) unless a name says otherwise.  Reactance convention: ``X`` is
the source reactance with **inductive positive** (COMSOL ``exp(+j w t)``: ``X = Im Zport``;
axionFEM ``exp(-i w t)``: ``X = -Im Z = omega L_H``).  :func:`source_table` converts.

The module needs only numpy/scipy; nothing here calls the C++ core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- constants (CODATA 2018, exact SI)
H_PLANCK = 6.62607015e-34          # J s
HBAR = H_PLANCK / (2.0 * math.pi)
KB = 1.380649e-23                  # J/K
E_CHARGE = 1.602176634e-19         # C
H_EV_S = H_PLANCK / E_CHARGE       # eV s
HBAR_EV_S = HBAR / E_CHARGE
HBARC_EV_M = 1.973269804e-7        # eV m
ALPHA = 7.2973525693e-3
YEAR_JULIAN_S = 365.25 * 86400.0
T_TESLA_EV2 = 195.35277            # 1 T in natural Heaviside-Lorentz units [eV^2] (e = sqrt(4 pi alpha))
E_NAT = math.sqrt(4.0 * math.pi * ALPHA)
VOLT_TO_EV = 1.0 / E_NAT                          # 1 V -> eV  ([R]: 3.30227)
HENRY_TO_INV_EV = 1.0 / (HBAR_EV_S * 376.730313668)  # 1 H = 1 ohm s -> eV^-1 ([R]: 4.03278e12)
AMP_PER_M2_TO_EV3 = (E_NAT / E_CHARGE) * HBAR_EV_S * HBARC_EV_M ** 2  # 1 A/m^2 -> eV^3
GEV_CM3_TO_EV4 = 1e9 * (HBARC_EV_M * 100.0) ** 3

# Statistics / halo conventions of [R] (and DMRadio)
THETA_NOTEBOOK = 2.705543454095404   # chi2.ppf(0.9, df=1): one-sided 95% (snr_target**2)
QA_NOTEBOOK = 6.4e5
RHO_NOTEBOOK = 0.45                  # GeV/cm^3, implied by [R]'s 10.6049 A/m^2 GeV/T
KSVZ_NOTEBOOK = 1.92
DFSZ_NOTEBOOK = 0.75


def g_notebook(C_ag: float, m_eV):
    """The QCD line of [R] and 2211.00008, ``g = 2e-10 C_ag m_a/eV`` [GeV^-1] (M0)."""
    return 2e-10 * C_ag * np.asarray(m_eV, dtype=float)


def g_dfsz_notebook(f_Hz):
    """``g_notebook(0.75, h f)``: the DFSZ target of [R] and of 2211.00008's figures."""
    return g_notebook(DFSZ_NOTEBOOK, H_EV_S * np.asarray(f_Hz, dtype=float))


def J_eff_A_per_m2(g_GeV, B_T: float, rho_GeV_cm3: float = RHO_NOTEBOOK):
    """Peak axion effective current density ``|J_eff| = g B sqrt(2 rho)`` [A/m^2] (M0).

    With ``rho = 0.45`` GeV/cm^3 this is [R]'s ``10.6049 * g[GeV^-1] * B[T]``.
    """
    g = np.asarray(g_GeV, dtype=float) * 1e-9            # eV^-1
    J = g * math.sqrt(2.0 * rho_GeV_cm3 * GEV_CM3_TO_EV4) * B_T * T_TESLA_EV2
    return J / AMP_PER_M2_TO_EV3


# ---------------------------------------------------------------- source table
@dataclass
class SourceTable:
    """Full-wave equivalent circuit of a detector port: ``f_Hz``, reactance ``X_ohm``
    (inductive > 0), source resistance ``R_ohm`` and open-circuit source amplitude ``V``
    per unit source (``source_unit``: ``"A/m^2"`` for a COMSOL Je run, ``"kappa_SI"`` for
    axionFEM, where ``V = omega |Psi_SI|`` per unit ``kappa_a`` [H^-1])."""
    f_Hz: np.ndarray
    X_ohm: np.ndarray
    R_ohm: np.ndarray
    V: np.ndarray
    source_unit: str = "A/m^2"
    meta: dict = field(default_factory=dict)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * math.pi * self.f_Hz

    def at(self, f_Hz) -> dict[str, np.ndarray]:
        """Linear interpolation of X, R, V at ``f_Hz`` (no extrapolation: raises outside)."""
        f = np.asarray(f_Hz, dtype=float)
        if np.any(f < self.f_Hz[0] * (1 - 1e-12)) or np.any(f > self.f_Hz[-1] * (1 + 1e-12)):
            raise ValueError(f"SourceTable.at: frequency outside the table [{self.f_Hz[0]:.6g}, "
                             f"{self.f_Hz[-1]:.6g}] Hz; the response is never extrapolated")
        return {"X_ohm": np.interp(f, self.f_Hz, self.X_ohm), "R_ohm": np.interp(f, self.f_Hz, self.R_ohm),
                "V": np.interp(f, self.f_Hz, self.V)}

    def L_eff(self, f_Hz, rule: str = "tuned") -> np.ndarray:
        """:func:`effective_inductance` from a cached cubic spline of ``X(f)``."""
        if getattr(self, "_spline", None) is None:
            from scipy.interpolate import CubicSpline
            object.__setattr__(self, "_spline", CubicSpline(self.f_Hz, self.X_ohm))
        fe = np.asarray(f_Hz, float)
        X, dXdf = self._spline(fe), self._spline(fe, 1)
        w = 2.0 * math.pi * fe
        dXdw = dXdf / (2.0 * math.pi)
        if rule == "notebook":
            return 0.5 * (X / w + dXdw)
        if rule == "tuned":
            return 0.5 * (dXdw + np.abs(X) / w)
        raise ValueError("L_eff: rule must be 'tuned' or 'notebook'")

    def restrict(self, f_min: float | None = None, f_max: float | None = None) -> "SourceTable":
        s = np.ones(self.f_Hz.size, bool)
        if f_min is not None:
            s &= self.f_Hz >= f_min
        if f_max is not None:
            s &= self.f_Hz <= f_max
        return SourceTable(self.f_Hz[s], self.X_ohm[s], self.R_ohm[s], self.V[s], self.source_unit, dict(self.meta))


def source_table(f_Hz, Z_ohm, V, *, convention: str, V_f_Hz=None, port_load_ohm: float | None = None,
                 source_unit: str = "A/m^2", check_inductive: bool = True, meta: dict | None = None) -> SourceTable:
    """Build a :class:`SourceTable` from a port impedance ``Z_ohm`` (complex) and source-run
    port voltage ``V`` (complex or magnitude).

    ``convention``: ``"exp(+jwt)"`` (COMSOL, engineering: inductor ``+j w L``) or ``"exp(-iwt)"``
    (axionFEM, physics: inductor ``-i w L``; ``Z_axf = conj(Z_comsol)``).  ``V_f_Hz`` is the
    voltage's own grid (default: ``f_Hz``); the overlap of the two grids is kept.
    ``port_load_ohm``: the voltage run loaded the port with this resistance (COMSOL lumped port
    with excitation off loads it with ``Zref``), so the open-circuit (Thevenin) voltage is
    ``V_oc = V (Z + Z_load)/Z_load`` (M0b).  ``check_inductive`` refuses a table whose
    reactance is not inductive at its lowest frequency (an MQS pickup always is; a negative
    value means the sign convention is wrong -- [R]'s own table needs ``X = -Im Z``).
    """
    f = np.asarray(f_Hz, dtype=float).ravel()
    Z = np.asarray(Z_ohm, dtype=complex).ravel()
    if convention == "exp(-iwt)":
        Z = np.conj(Z)
        Vc = np.conj(np.asarray(V, dtype=complex).ravel())
    elif convention == "exp(+jwt)":
        Vc = np.asarray(V, dtype=complex).ravel()
    else:
        raise ValueError("source_table: convention must be 'exp(+jwt)' (COMSOL) or 'exp(-iwt)' (axionFEM)")
    if f.size != Z.size:
        raise ValueError(f"source_table: f_Hz ({f.size}) and Z_ohm ({Z.size}) differ in length")
    fv = f if V_f_Hz is None else np.asarray(V_f_Hz, dtype=float).ravel()
    if fv.size != Vc.size:
        raise ValueError(f"source_table: V ({Vc.size}) and its grid ({fv.size}) differ in length")
    for name, grid in (("f_Hz", f), ("V_f_Hz", fv)):
        if np.any(~np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
            raise ValueError(f"source_table: {name} must be positive, finite and strictly increasing")
    lo, hi = max(f[0], fv[0]), min(f[-1], fv[-1])
    if V_f_Hz is None:
        keep = np.ones(f.size, bool)
    else:
        keep = (f >= lo * (1 - 1e-12)) & (f <= hi * (1 + 1e-12))
    f, Z = f[keep], Z[keep]
    if V_f_Hz is not None:
        Vc = np.interp(f, fv, Vc.real) + 1j * np.interp(f, fv, Vc.imag)   # exact where the grids coincide
    if port_load_ohm is not None:
        if not (port_load_ohm > 0):
            raise ValueError("source_table: port_load_ohm must be positive")
        Vc = Vc * (Z + port_load_ohm) / port_load_ohm
    if np.any(~np.isfinite(Z)) or np.any(~np.isfinite(Vc)):
        raise ValueError("source_table: Z or V contains NaN/inf")
    X = Z.imag
    neg = Z.real < -1e-9 * np.abs(Z)
    if neg.mean() > 0.5:
        raise ValueError(f"source_table: Re Z < 0 at {int(neg.sum())} of {neg.size} points (min {Z.real.min():.3g} ohm); "
                         "a passive port has Re Z >= 0, so the table is probably sign-flipped (Z_file = -Z), not "
                         "conjugated: pass -Z with convention='exp(+jwt)'")
    meta = dict(meta or {})
    if neg.any():   # isolated numerical negatives (e.g. at poles of a lossy COMSOL sweep): kept, recorded
        import warnings
        warnings.warn(f"source_table: Re Z < 0 at {int(neg.sum())} isolated points (min {Z.real.min():.3g} ohm)")
        meta["negative_R_points"] = int(neg.sum())
    if check_inductive and not X[0] > 0:
        raise ValueError(f"source_table: reactance at the lowest frequency {f[0]:.6g} Hz is {X[0]:.6g} ohm "
                         "(not inductive); the sign convention is probably wrong -- pass the other "
                         "convention (or check_inductive=False for a table that starts above a self-resonance)")
    return SourceTable(f, X, Z.real.copy(), np.abs(Vc), source_unit, meta)


def table_from_results(results: Mapping[str, Any]) -> SourceTable:
    """A :class:`SourceTable` from an axionFEM full-wave result (``circuit.fullwave_tables``):
    ``X = omega L_H`` (axionFEM ``L_H = -Im Z/omega``), ``R = R_rad + R_loss``,
    ``V = omega |Psi_SI|`` per unit ``kappa_a`` [H^-1].  MQS references go to ``meta``."""
    from . import circuit as ck
    data = getattr(results, "data", results)
    t = ck.fullwave_tables(data)
    w = 2.0 * math.pi * t["f_Hz"]
    cp = data.get("coupling", {}) or {}
    meta = {"L_MQS_H": cp.get("L_p_H"), "Psi_MQS_SI": cp.get("Psi_SI_H_T_m2"), "origin": "axionFEM fullwave"}
    return SourceTable(t["f_Hz"], w * t["L_H"], t["R_ohm"], w * t["Psi_SI"], "kappa_SI", meta)


# ---------------------------------------------------------------- multi-band scan: tuning bands (M1-M2)
def tuning_elements(f_Hz, X_ohm) -> dict[str, np.ndarray]:
    """Series tuning element that cancels ``X``: capacitor ``C = 1/(omega X)`` where ``X > 0``,
    inductor ``L = -X/omega`` where ``X < 0`` (NaN elsewhere)  (M1)."""
    w = 2.0 * math.pi * np.asarray(f_Hz, float)
    X = np.asarray(X_ohm, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(X > 0, 1.0 / (w * X), np.nan)
        L = np.where(X < 0, -X / w, np.nan)
    return {"C_F": C, "L_H": L}


def valid_intervals(f_Hz, X_ohm, *, L_range=(10e-9, 500e-9), C_range=(30e-12, 300e-12),
                    edge_margin_Hz: float = 2.5e6, min_width_Hz: float = 5e6,
                    allow_inductive: bool = True) -> list[dict]:
    """Tunable bands (M2).

    A frequency is tunable when the element of :func:`tuning_elements` lies strictly inside
    its range (open intervals, as in [R]).  Contiguous tunable runs whose span
    ``f[last] - f[first]`` is at least ``min_width_Hz`` are kept and shrunk by
    ``edge_margin_Hz`` at both ends.  Each band reports its ``kind`` (``"capacitive"`` tuning:
    source inductive; ``"inductive"``: source capacitive) from the sign of ``X`` at its lower
    edge ([R]'s colouring rule) and whether it mixes both.
    ``allow_inductive=False`` drops inductor tunings (the 2211.00008 rule).
    """
    f = np.asarray(f_Hz, float)
    X = np.asarray(X_ohm, float)
    el = tuning_elements(f, X)
    okC = (el["C_F"] > C_range[0]) & (el["C_F"] < C_range[1])
    okL = (el["L_H"] > L_range[0]) & (el["L_H"] < L_range[1]) if allow_inductive else np.zeros(f.size, bool)
    valid = okC | okL
    d = np.diff(valid.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [f.size]
    out = []
    for s, e in zip(starts, ends):
        if f[e - 1] - f[s] >= min_width_Hz:
            lo, hi = f[s] + edge_margin_Hz, f[e - 1] - edge_margin_Hz
            if hi <= lo:
                continue
            sel = (f >= lo) & (f <= hi)
            x_lo = float(np.interp(lo, f, X))
            out.append({"f_lo_Hz": float(lo), "f_hi_Hz": float(hi),
                        "kind": "capacitive" if x_lo >= 0 else "inductive",
                        "mixed": bool(np.any(X[sel] > 0) and np.any(X[sel] < 0)),
                        "n_points": int(sel.sum())})
    return out


def band_mask(f_Hz, bands: Sequence[Mapping]) -> np.ndarray:
    f = np.asarray(f_Hz, float)
    m = np.zeros(f.size, bool)
    for b in bands:
        m |= (f >= b["f_lo_Hz"]) & (f <= b["f_hi_Hz"])
    return m


# ---------------------------------------------------------------- effective inductance (M3)
def effective_inductance(f_Hz, X_ohm, *, rule: str = "tuned", f_eval=None) -> np.ndarray:
    """Slope (scan) inductance of the series-tuned port  (M3).

    ``rule="notebook"``: [R]'s ``L_eff = (X + f dX/df)/(4 pi f) = (X/omega + dX/domega)/2``
    (arXiv:2302.14084 Eq. 6; correct for a **capacitor** tuning, ``X > 0``).
    ``rule="tuned"``: ``L_eff = (dX/domega + |X|/omega)/2``, the half slope of the total reactance
    ``X - 1/(omega C)`` (capacitor) or ``X + omega L_t`` (inductor) at the zero it is tuned to.
    The two agree for ``X > 0``; for an inductor tuning (``X < 0``) the rule of [R]
    underestimates ``L_eff`` by ``|X|/omega`` (and so overestimates the rate).
    Derivative: cubic spline of ``X(f)`` (as in [R]).
    """
    from scipy.interpolate import CubicSpline
    f = np.asarray(f_Hz, float)
    X = np.asarray(X_ohm, float)
    fe = f if f_eval is None else np.asarray(f_eval, float)
    sp = CubicSpline(f, X)
    Xe, dXdf = sp(fe), sp(fe, 1)
    w = 2.0 * math.pi * fe
    dXdw = dXdf / (2.0 * math.pi)
    if rule == "notebook":
        return 0.5 * (Xe / w + dXdw)
    if rule == "tuned":
        return 0.5 * (dXdw + np.abs(Xe) / w)
    raise ValueError("effective_inductance: rule must be 'tuned' or 'notebook'")


# ---------------------------------------------------------------- scan rate (M4-M5)
def n_bose(f_Hz, T_K):
    x = H_PLANCK * np.asarray(f_Hz, float) / (KB * np.asarray(T_K, float))
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(x)


def gbar_opt(f_Hz, T_K, eta_A):
    """``G_bar(xi_opt)`` of spec (S3.8)-(S3.10) = [R]'s ``get_GBar`` (M4)."""
    a = 2.0 * n_bose(f_Hz, T_K) + 1.0
    eta = np.asarray(eta_A, float)
    xi = 2.0 * eta ** 2 / (a + np.sqrt(a * a + 8.0 * eta ** 2))
    return xi / (xi ** 2 + 2.0 * a * xi + eta ** 2) ** 1.5


def source_amplitude(table_V, source_unit: str, g_GeV, *, B_T: float | None = None,
                     rho_GeV_cm3: float = RHO_NOTEBOOK):
    """Peak EMF [V] at coupling ``g``: ``V * |J_eff|`` (``A/m^2`` tables, needs ``B_T``) or
    ``V * kappa_SI`` (axionFEM tables; ``kappa_SI = 8.65882567e-19 (g/1e-19) sqrt(rho/0.3)`` H^-1,
    spec Sec. 8)."""
    if source_unit == "A/m^2":
        if B_T is None:
            raise ValueError("source_amplitude: an A/m^2 (COMSOL Je) table needs the field B_T")
        return np.asarray(table_V, float) * J_eff_A_per_m2(g_GeV, B_T, rho_GeV_cm3)
    if source_unit == "kappa_SI":
        k = 8.65882567e-19 * (np.asarray(g_GeV, float) / 1e-19) * math.sqrt(rho_GeV_cm3 / 0.3)
        return np.asarray(table_V, float) * k
    raise ValueError(f"source_amplitude: unknown source unit {source_unit!r}")


def scan_rate(f_Hz, V_peak, L_eff_H, Q, T_K, eta_A, *, Q_a: float = QA_NOTEBOOK,
              Theta: float = THETA_NOTEBOOK) -> np.ndarray:
    """Local resonant scan rate ``d nu/dt`` [Hz/s]  (M5; spec S5.1 with ``Phi_a = V/m_a``):

        ``d nu/dt = (pi/16) (Q_a/Theta) V^4 Q G_bar / (m_a^4 L_eff^2)``  (natural units),

    [R]'s ``dnu_dt`` expression (prefactor ``6.4e5 pi/(snr^2 16 m^4)``), converted to SI
    with ``hbar`` ([R]'s ``h`` conversion makes its MHz/yr ``(2 pi)^2`` too small).
    """
    f = np.asarray(f_Hz, float)
    m = H_EV_S * f                                    # eV
    Vn = np.asarray(V_peak, float) * VOLT_TO_EV       # eV
    Ln = np.asarray(L_eff_H, float) * HENRY_TO_INV_EV  # eV^-1
    G = gbar_opt(f, T_K, eta_A)
    r = Q_a * math.pi / Theta / 16.0 / m ** 4 * Vn ** 4 * np.asarray(Q, float) * G / Ln ** 2  # eV^2
    # natural -> SI: nu_SI = nu_nat/hbar and t_SI = hbar t_nat, so d nu/dt = r/hbar^2.  ([R]
    # divides by h^2 instead, i.e. its MHz/yr are (2 pi)^2 = 39.5 times too small; see notebook_dnu_dt.)
    return r / HBAR_EV_S ** 2


def notebook_dnu_dt(f_MHz, table: SourceTable, *, Q: float = 2e5, T_mK: float = 20.0,
                    C_ag: float = KSVZ_NOTEBOOK, B_T: float = 5.0, eta_A: float = 20.0,
                    rounded_constants: bool = True, notebook_units: bool = True) -> np.ndarray:
    """[R]'s ``dnu_dt`` [MHz/yr] on a :class:`SourceTable` built as [R] builds it
    (spline of ``X`` and ``|V|`` in MHz).  [R] hard-wires ``eta_A = 20``
    inside ``dnu_dt`` (``get_GBar(nu_r, T)``); here it is an argument with that default.
    ``notebook_units=True`` keeps [R]'s eV^2 -> MHz/yr conversion with ``h`` where ``hbar``
    is needed (rate 1/(2 pi)^2 too small); ``False`` gives the physical rate (= :func:`scan_rate`)."""
    from scipy.interpolate import CubicSpline
    fM = np.asarray(f_MHz, float)
    fT = table.f_Hz / 1e6
    Le = effective_inductance(table.f_Hz, table.X_ohm, rule="notebook", f_eval=fM * 1e6)
    V = CubicSpline(fT, table.V)(fM)
    Vp = source_amplitude(V, table.source_unit, g_notebook(C_ag, H_EV_S * fM * 1e6), B_T=B_T)
    r = scan_rate(fM * 1e6, Vp, Le, Q, T_mK * 1e-3, eta_A)
    if notebook_units:      # [R] converts eV^2 -> MHz/yr with h(eV/MHz) h(eV yr): 1/(2 pi)^2 low
        r = r / (2.0 * math.pi) ** 2
    if rounded_constants:   # [R]'s 6-digit 3.30227, 4.03278e12, 10.6049 (1.6e-5 in the rate)
        r = r * ((3.30227 / VOLT_TO_EV) * (10.6049 / float(J_eff_A_per_m2(1.0, 1.0, RHO_NOTEBOOK)))) ** 4 \
            / (4.03278e12 / HENRY_TO_INV_EV) ** 2
    return r / 1e6 * YEAR_JULIAN_S


# ---------------------------------------------------------------- allocation (M6-M7)
def greedy_allocation(f_Hz, rate_Hz_per_s, bands: Sequence[Mapping], live_time_s: float, *,
                      chunk_Hz: float = 0.1e6, merge: str = "adjacent") -> dict:
    """Spend ``live_time_s`` on the cheapest chunks of all bands (M6).

    Each band is cut into ``round(width/chunk)`` equal chunks; a chunk costs
    ``int df / max(rate, 1e-10 MHz/yr)`` with a cubic spline of the rate (as in [R]);
    chunks are taken cheapest first while the cumulative time stays below the budget.
    ``merge="adjacent"`` joins kept chunks that share an edge; ``merge="notebook"`` follows
    [R]'s rounding-based merge (edges rounded to 10 kHz, split where the gap differs
    from the chunk width by more than 1e-8 MHz).
    Returns the kept chunks, merged intervals, total time used and the per-chunk times.
    """
    from scipy import integrate
    from scipy.interpolate import CubicSpline
    fM = np.asarray(f_Hz, float) / 1e6
    rate = np.asarray(rate_Hz_per_s, float) / 1e6 * YEAR_JULIAN_S        # MHz/yr, as in [R]
    sp = CubicSpline(fM, rate)
    budget = live_time_s / YEAR_JULIAN_S
    dM = chunk_Hz / 1e6
    bounds, times = [], []
    for b in bands:
        lo, hi = b["f_lo_Hz"] / 1e6, b["f_hi_Hz"] / 1e6
        e = np.linspace(lo, hi, 1 + int(np.around((hi - lo) / dM)))
        for a, c in zip(e[:-1], e[1:]):
            bounds.append([a, c])
            times.append(integrate.quad(lambda x: 1.0 / np.maximum(1e-10, sp(x)), a, c,
                                        limlst=100000, limit=10000)[0])
    order = np.argsort(times, kind="quicksort")
    t = np.asarray(times)[order]
    bd = np.asarray(bounds).reshape(-1, 2)[order]
    keep = np.flatnonzero(np.cumsum(t) < budget)
    kept = bd[keep]
    if merge == "notebook":
        pts = np.unique(np.around(kept.flatten(), 2))
        brk = np.flatnonzero(np.abs(np.diff(pts) - dM) > 1e-8)
        st = np.r_[0, brk + 1]
        en = np.r_[brk + 1, len(pts)]
        merged = [[float(pts[s]), float(pts[e - 1])] for s, e in zip(st, en)]
    elif merge == "adjacent":
        merged = []
        for a, c in sorted(kept.tolist()):
            if merged and abs(a - merged[-1][1]) <= 1e-9 * max(1.0, abs(a)):
                merged[-1][1] = c
            else:
                merged.append([a, c])
    else:
        raise ValueError("greedy_allocation: merge must be 'adjacent' or 'notebook'")
    return {"chunks_MHz": kept, "chunk_time_yr": t[keep], "intervals_MHz": merged,
            "time_used_s": float(np.sum(t[keep]) * YEAR_JULIAN_S), "n_chunks": int(len(times)),
            "n_kept": int(keep.size), "all_chunk_time_yr": t, "all_chunks_MHz": bd}


def uniform_dwell_reach(f_Hz, rate_at_target_Hz_per_s, g_target_GeV, mask, live_time_s: float,
                        bandwidth_Hz: float) -> np.ndarray:
    """Uniform time density ``dt/dnu = T/Delta nu`` over the tunable band (the uniform-dwell
    variant; M7): ``g = g_t [Delta nu/(T rate_t)]^(1/4)`` inside ``mask``, NaN outside."""
    r = np.asarray(rate_at_target_Hz_per_s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (bandwidth_Hz / (live_time_s * r)) ** 0.25
    return np.where(np.asarray(mask, bool), np.asarray(g_target_GeV, float) * c, np.nan)


def target_allocation_reach(f_Hz, rate_at_target_Hz_per_s, g_target_GeV, mask, live_time_s: float):
    """Optimal (target-curve) allocation restricted to ``mask`` (spec S5.5-S5.6):
    ``t_req = int_band df/rate``, ``g = g_t (t_req/T)^(1/4)``; returns ``(g, t_req_s)``."""
    f = np.asarray(f_Hz, float)
    r = np.asarray(rate_at_target_Hz_per_s, float)
    m = np.asarray(mask, bool) & (r > 0) & np.isfinite(r)
    if not m.any():
        raise ValueError("target_allocation_reach: no frequency with a positive scan rate inside the mask")
    inv = np.where(m, 1.0 / np.where(m, r, 1.0), 0.0)
    # integrate only over masked runs (no bridging between bands)
    t_req = 0.0
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= 2:
            t_req += float(np.trapezoid(inv[s:e], f[s:e]) if hasattr(np, "trapezoid") else np.trapz(inv[s:e], f[s:e]))
    g = np.where(m, np.asarray(g_target_GeV, float) * (t_req / live_time_s) ** 0.25, np.nan)
    return g, t_req


def multiband_scan(table: SourceTable, *, Q: float | None = None, Q_mqs: float | None = None,
                   T_K: float = 0.020, eta_A: float = 20.0, B_T: float | None = 5.0,
                   live_time_yr: float = 1.0, g_target: Callable = g_dfsz_notebook,
                   L_range=(10e-9, 500e-9), C_range=(30e-12, 300e-12), edge_margin_Hz: float = 2.5e6,
                   min_width_Hz: float = 5e6, allow_inductive: bool = True, Leff_rule: str = "tuned",
                   chunk_Hz: float = 0.1e6, Q_a: float = QA_NOTEBOOK, Theta: float = THETA_NOTEBOOK,
                   rho_GeV_cm3: float = RHO_NOTEBOOK, f_max_Hz: float | None = None,
                   notebook_units: bool = False) -> dict:
    """Beta. The multi-band scan of [R] on any full-wave table (M1-M7).

    Loss: give ``Q`` (fixed resonator Q, as in [R]: ``Q = 1e5`` with a lossless table) **or**
    ``Q_mqs`` (intrinsic MQS Q: ``r = omega L_MQS/Q_mqs`` added to the table's ``R``, so
    ``Q = omega L_eff/(R + r)``, the loaded Q of 2211.00008).  Returns the bands, the rate at the
    target coupling, the greedy coverage at the target, the uniform-dwell and the
    target-allocation reach over the bands.
    """
    L_mqs_full = table.meta.get("L_MQS_H") or table.X_ohm[0] / (2.0 * math.pi * table.f_Hz[0])
    t = table if f_max_Hz is None else table.restrict(f_max=f_max_Hz)
    f, X = t.f_Hz, t.X_ohm
    bands = valid_intervals(f, X, L_range=L_range, C_range=C_range, edge_margin_Hz=edge_margin_Hz,
                            min_width_Hz=min_width_Hz, allow_inductive=allow_inductive)
    Le = effective_inductance(f, X, rule=Leff_rule)
    w = 2.0 * math.pi * f
    if (Q is None) == (Q_mqs is None):
        raise ValueError("multiband_scan: give exactly one of Q= (fixed) and Q_mqs= (intrinsic MQS loss)")
    if Q is not None:
        Qv = np.full(f.size, float(Q))
    else:
        L_mqs = L_mqs_full
        Qv = w * Le / (t.R_ohm + w * L_mqs / Q_mqs)
    m_eV = H_EV_S * f
    gt = g_target(f)
    Vp = source_amplitude(t.V, t.source_unit, gt, B_T=B_T, rho_GeV_cm3=rho_GeV_cm3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(Le > 0, scan_rate(f, Vp, Le, Qv, T_K, eta_A, Q_a=Q_a, Theta=Theta), 0.0)
    if notebook_units:   # reproduce [R]'s (2 pi)^2-low rate
        rate = rate / (2.0 * math.pi) ** 2
    mask = band_mask(f, bands)
    T = live_time_yr * YEAR_JULIAN_S
    width = float(sum(b["f_hi_Hz"] - b["f_lo_Hz"] for b in bands))
    greedy = greedy_allocation(f, rate, bands, T, chunk_Hz=chunk_Hz) if bands else None
    g_uni = uniform_dwell_reach(f, rate, gt, mask, T, width) if bands else np.full(f.size, np.nan)
    try:
        g_tgt, t_req = target_allocation_reach(f, rate, gt, mask, T)
    except ValueError:      # no band, or no positive rate in the bands
        g_tgt, t_req = np.full(f.size, np.nan), 0.0
    return {"f_Hz": f, "m_eV": m_eV, "bands": bands, "L_eff_H": Le, "Q": Qv, "rate_Hz_per_s": rate,
            "rate_MHz_per_yr": rate / 1e6 * YEAR_JULIAN_S, "g_target_GeV": gt, "greedy": greedy,
            "g_uniform_GeV": g_uni, "g_target_allocation_GeV": g_tgt, "t_req_s": t_req,
            "band_width_Hz": width,
            "params": {"T_K": T_K, "eta_A": eta_A, "B_T": B_T, "Q": Q, "Q_mqs": Q_mqs, "Q_a": Q_a,
                       "Theta": Theta, "rho_GeV_cm3": rho_GeV_cm3, "live_time_yr": live_time_yr,
                       "L_range": list(L_range), "C_range": list(C_range), "Leff_rule": Leff_rule,
                       "allow_inductive": allow_inductive}}


# ================================================================ 2211.00008 reconstruction (M8-M13)
def mqs_reference(table: SourceTable, f_ref_Hz: float | None = None) -> dict:
    """MQS references: ``L_MQS = X/omega`` and ``Psi_MQS = V/omega`` at ``f_ref`` (default: the
    lowest table frequency; 2211.00008 used 5 MHz), unless ``table.meta`` carries
    ``L_MQS_H``/``Psi_MQS_SI`` (axionFEM static values)  (M8)."""
    f0 = table.f_Hz[0] if f_ref_Hz is None else f_ref_Hz
    a = table.at(f0)
    w0 = 2.0 * math.pi * f0
    L = table.meta.get("L_MQS_H") if f_ref_Hz is None and table.meta.get("L_MQS_H") else a["X_ohm"] / w0
    psi = table.meta.get("Psi_MQS_SI") if f_ref_Hz is None and table.meta.get("Psi_MQS_SI") else a["V"] / w0
    return {"f_ref_Hz": float(f0), "L_MQS_H": float(L), "Psi_MQS": float(psi)}


def _noise_E(f_Hz, T_K, eta_A, classical: bool):
    """Thermal energy per mode ``2 k T`` (classical, as in [P]) or ``hbar w (2n+1)``, and the
    amplifier energy ``eta hbar w``."""
    w = 2.0 * math.pi * np.asarray(f_Hz, float)
    th = 2.0 * KB * T_K * np.ones_like(w) if classical else HBAR * w * (2.0 * n_bose(f_Hz, T_K) + 1.0)
    return th, eta_A * HBAR * w


def single_pole_ratio(table: SourceTable, *, T_K: float = 0.020, eta_A: float = 20.0, Q_mqs: float = 1e6,
                      noise: str = "paper2211", f_ref_Hz: float | None = None, f_eval=None) -> dict:
    """On-resonance sensitivity relative to MQS, ``g/g_MQS`` at each tuning, and ``Q_B``  (M9-M11).

    ``Q_B = omega L_eff/R_t`` with ``L_eff = (dX/domega + |X|/omega)/2``, ``R_t = R + r``,
    ``r = omega L_MQS/Q_mqs``.  With ``E_th = 2kT`` and ``E_A = eta hbar omega``:

    * ``noise="paper2211"`` (reproduces 2211.00008 Figs. 3, S1-S3, S5):
      ``g/g_MQS = (V_MQS/V) (Q_B/Q_mqs)^(1/4) [(E_th r + E_A R_t)/(r (E_th + E_A))]^(1/2)`` --
      Johnson noise from the intrinsic ``r`` only, amplifier noise-matched to ``R_t``;
    * ``noise="fdt"``: the table's ``R`` radiates Johnson noise at ``T`` as well (fluctuation-
      dissipation): ``... [R_t/r]^(1/2)`` (independent of ``eta_A``).

    Lossless tables (``R = 0``) give ``(V_MQS/V)(Q_B/Q_mqs)^(1/4)`` in both.  ``tuning`` is
    ``"capacitive"`` where ``X > 0``; elsewhere the ratio is still returned (inductor tuning)
    and flagged.
    """
    ref = mqs_reference(table, f_ref_Hz)
    f = table.f_Hz if f_eval is None else np.asarray(f_eval, float)
    a = table.at(f)
    w = 2.0 * math.pi * f
    Le = table.L_eff(f)
    r = w * ref["L_MQS_H"] / Q_mqs
    Rt = a["R_ohm"] + r
    QB = w * Le / Rt
    V_mqs = w * ref["Psi_MQS"]
    Eth, EA = _noise_E(f, T_K, eta_A, classical=True)
    if noise == "paper2211":
        loss = (Eth * r + EA * Rt) / (r * (Eth + EA))
    elif noise == "fdt":
        loss = Rt / r
    else:
        raise ValueError("single_pole_ratio: noise must be 'paper2211' or 'fdt'")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (V_mqs / a["V"]) * (QB / Q_mqs) ** 0.25 * np.sqrt(loss)
    return {"f_Hz": f, "g_over_gMQS": ratio, "Q_B": QB, "L_eff_H": Le, "R_t_ohm": Rt, "r_ohm": r,
            "V_over_VMQS": a["V"] / V_mqs, "tuning": np.where(a["X_ohm"] > 0, "capacitive", "inductive"),
            "reference": ref, "noise": noise}


def _norm_const(noise_norm: str) -> float:
    # per-bin SNR s = c_s S_VV / N, N = R E: [P]'s text (S_II = Q(2T+eta w)/(pi m L)) -> c_s = pi;
    # the normalisation of arXiv:1711.10489 used by axionFEM (S_V,th = 2 k T R) -> c_s = 1
    if noise_norm == "paper2211":
        return math.pi
    if noise_norm == "foster":
        return 1.0
    raise ValueError("noise_norm must be 'paper2211' or 'foster'")


def mqs_schedule(table: SourceTable, *, T_K: float = 0.020, eta_A: float = 20.0, Q_mqs: float = 1e6,
                 B_T: float = 10.0, f_start_Hz: float = 150e6, live_time_yr: float = 1.0,
                 f_end_Hz: float | None = None, delta: float = 1e-6, g_target: Callable = g_dfsz_notebook,
                 Q_a: float = 6.8e5, Theta: float = THETA_NOTEBOOK, rho_GeV_cm3: float = 0.3,
                 noise_norm: str = "paper2211", f_ref_Hz: float | None = None, n_grid: int = 200001) -> dict:
    """The 2211.00008 MQS scan schedule  (M12).

    Tunings at ``delta f/f = delta`` from ``f_start`` downwards; each dwells until the MQS
    on-resonance Asimov statistic at the target reaches ``Theta``:
    ``t_i = Theta/K_MQS(f_i)``, ``K = (1/2pi) int s^2 domega``, ``s = c_s (pi/2) V^2 L(omega)/(r E)``
    with ``int L^2 domega = Q_a/omega`` and ``E = 2kT + eta hbar omega`` (classical, as in [P]),
    i.e. ``K_MQS = c_s^2 pi Q_a V_MQS^4/(8 omega r^2 E^2)``.  The scan stops when the live time is
    spent (``f_end``), or at a given ``f_end_Hz``.  Returns the dwell density per ``ln f``
    (``dens * t``) on a log grid and ``f_end``.  Frequencies with ``X <= 0`` get no dwell.

    ``noise_norm="paper2211"`` (default, reproduction) keeps [P]'s ``S_II^noise = Q(2T+eta w)/(pi m L)``,
    whose 1/pi makes Theta pi^2 too large (g sqrt(pi) = 1.77x optimistic) relative to the normalisation of
    Foster, Rodd and Safdi (arXiv:1711.10489) used in docs/physics/scan_and_signal.md (``noise_norm="foster"``,
    physical).
    """
    ref = mqs_reference(table, f_ref_Hz)
    c = _norm_const(noise_norm)
    lo = table.f_Hz[0] if f_end_Hz is None else f_end_Hz
    f = np.geomspace(lo, f_start_Hz, n_grid)
    w = 2.0 * math.pi * f
    r = w * ref["L_MQS_H"] / Q_mqs
    Eth, EA = _noise_E(f, T_K, eta_A, classical=True)
    Vm = source_amplitude(w * ref["Psi_MQS"], table.source_unit, g_target(f), B_T=B_T, rho_GeV_cm3=rho_GeV_cm3)
    K = c ** 2 * math.pi * Q_a * Vm ** 4 / (8.0 * w * (r * (Eth + EA)) ** 2)
    t = Theta / K
    # only capacitive tunings (X > 0) are made ([P]: inductive-tuning frequencies are excluded)
    cap = np.interp(f, table.f_Hz, table.X_ohm) > 0
    t = np.where(cap, t, 0.0)
    per_ln = t / delta                           # seconds per unit ln f
    u = np.log(f)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (per_ln[1:] + per_ln[:-1]) * np.diff(u))])
    total_from_top = cum[-1] - cum                # time to scan from f_start down to f
    T = live_time_yr * YEAR_JULIAN_S
    if f_end_Hz is None:
        if total_from_top[0] < T:
            f_end = float(f[0])
            note = "live time not exhausted at the table's lowest frequency"
        else:
            f_end = float(np.exp(np.interp(T, total_from_top[::-1], u[::-1])))
            note = "f_end from the live time"
    else:
        f_end = float(f_end_Hz)
        note = "f_end imposed"
    return {"f_end_Hz": f_end, "f_start_Hz": float(f_start_Hz), "delta": delta,
            "t_per_tuning_s": lambda fq: np.interp(np.log(fq), u, t),
            "f_grid_Hz": f, "t_grid_s": t, "time_from_top_s": total_from_top,
            "time_used_s": float(np.interp(math.log(f_end), u, total_from_top)),
            "reference": ref, "note": note, "g_target": g_target,
            "params": {"T_K": T_K, "eta_A": eta_A, "Q_mqs": Q_mqs, "B_T": B_T, "Q_a": Q_a, "Theta": Theta,
                       "rho_GeV_cm3": rho_GeV_cm3, "noise_norm": noise_norm}}


def _tuning_kernel(table: SourceTable, ref: dict, f: float, f0: np.ndarray, *, T_K, eta_A, Q_mqs, noise,
                   amp_model: str) -> np.ndarray:
    """``(V(f)^4/omega) / N_i(f)^2`` for tunings at ``f0`` (constant prefactors dropped)  (M13)."""
    w = 2.0 * math.pi * f
    a = table.at(f)
    b = table.at(f0)
    w0 = 2.0 * math.pi * f0
    r = w * ref["L_MQS_H"] / Q_mqs
    r0 = w0 * ref["L_MQS_H"] / Q_mqs
    Rt0 = b["R_ohm"] + r0
    Xi = a["X_ohm"] - w0 * b["X_ohm"] / w          # X(f) - 1/(omega C_i),  1/C_i = omega0 X(f0)
    Z2 = (a["R_ohm"] + r) ** 2 + Xi ** 2
    Eth, _ = _noise_E(np.array([f]), T_K, eta_A, classical=True)
    if noise not in ("paper2211", "fdt"):
        raise ValueError("noise must be 'paper2211' or 'fdt'")
    th = Eth[0] * (r if noise == "paper2211" else a["R_ohm"] + r)
    if amp_model == "matched":       # S3.5 with xi = eta (noise matched at omega0): imprecision + backaction
        amp = 0.5 * eta_A * HBAR * (w0 * Z2 / Rt0 + Rt0 * w * w / w0)
    elif amp_model == "flat":        # amplifier current noise fixed at its on-resonance value
        amp = eta_A * HBAR * w0 * Z2 / Rt0
    else:
        raise ValueError("amp_model must be 'matched' or 'flat'")
    return a["V"] ** 4 / w / (th + amp) ** 2


def _roots_u(u_grid, f_grid, target):
    """Frequencies where the piecewise-linear ``u(f) = omega X`` crosses ``target``."""
    d = u_grid - target
    with np.errstate(invalid="ignore"):
        s = np.flatnonzero(((d[:-1] <= 0) & (d[1:] > 0)) | ((d[:-1] >= 0) & (d[1:] < 0)))
    return f_grid[s] + (f_grid[s + 1] - f_grid[s]) * d[s] / (d[s] - d[s + 1])


def multipole_theta(table: SourceTable, schedule: dict, f_targets, *, noise: str = "paper2211",
                    amp_model: str = "matched", n_s: int = 1201, exclude_primary: bool = False,
                    check_convergence: bool = False, resonant_only: bool = False) -> dict:
    """Asimov statistic at each target frequency from the whole MQS schedule  (M13):

        ``Theta(f)/Theta_95 = int_{tunings} dln f0 (1/delta) K_i(f)/K_MQS(f0)``,

    the continuum limit of ``sum_i t_i K_i(f)``.  Tunings are capacitive (``X(f0) > 0``) between
    ``f_end`` and ``f_start``; ``T_K``, ``eta_A``, ``Q_mqs`` and ``g_target`` are the schedule's.
    Each tuning ``C_i`` resonates the target ``f`` where ``u(f0) = omega0 X(f0)`` equals
    ``omega X(f)``.  Nodes: an ``asinh`` grid centred on every such root (width = the mapped
    resonance half width) merged with every table point of the segment (the kernel also varies on
    the table scale, e.g. through ``R(f0)`` near poles); trapezoid in ``ln f0``.
    ``exclude_primary`` drops the ``f0 = f`` root (single-pole part).  ``check_convergence`` repeats
    with ``4 n_s`` and reports the relative change of Theta.  ``step_primary``/``step_secondary``:
    tuning step over the mapped resonance half width (> 1: [P]'s discrete comb is not resolved, so
    the continuum limit is not what a discrete scan gives).  ``resonant_only`` keeps only the
    resonance peaks (+-200 half widths around every root, no table nodes): a diagnostic that drops
    the off-resonant tails.  Returns ``g/g_target = Theta^(-1/4)``.
    """
    p = schedule["params"]
    T_K, eta_A, Q_mqs = p["T_K"], p["eta_A"], p["Q_mqs"]
    gt = schedule.get("g_target", g_dfsz_notebook)
    ref = schedule["reference"]
    fg = table.f_Hz
    sel = (fg >= schedule["f_end_Hz"] * (1 - 1e-12)) & (fg <= schedule["f_start_Hz"] * (1 + 1e-12))
    f_dom = fg[sel]
    X_dom = table.X_ohm[sel]
    ok = X_dom > 0
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    seg_idx = [(a_, b_) for a_, b_ in zip(edges[::2], edges[1::2]) if b_ - a_ >= 2]
    segs = [(f_dom[a_], f_dom[b_ - 1]) for a_, b_ in seg_idx]
    u_dom = 2.0 * math.pi * f_dom * X_dom
    delta = schedule["delta"]
    ft = np.atleast_1d(f_targets).astype(float)

    def theta_at(f, ns):
        w = 2.0 * math.pi * f
        a = table.at(f)
        uf = w * a["X_ohm"]
        roots = list(_roots_u(np.where(ok, u_dom, np.nan), f_dom, uf)) if a["X_ohm"] > 0 else []
        if exclude_primary:
            roots = [x for x in roots if abs(math.log(x / f)) > 1e-3]
        Rt_f = a["R_ohm"] + w * ref["L_MQS_H"] / Q_mqs
        total, st_p, st_s = 0.0, 0.0, 0.0
        for (lo, hi), (i0, i1) in zip(segs, seg_idx):
            xs = [] if resonant_only else [np.log(f_dom[i0:i1])]
            for c in (x for x in roots if lo <= x <= hi):
                Le_c = float(table.L_eff(np.array([c]))[0])
                wc = 2.0 * math.pi * c
                h = min(max(Rt_f * w / max(2.0 * wc * wc * abs(Le_c), 1e-300), 1e-14), 0.1)
                lc = math.log(c)
                a_lo, a_hi = math.log(lo), math.log(hi)
                if resonant_only:
                    a_lo, a_hi = max(a_lo, lc - 200 * h), min(a_hi, lc + 200 * h)
                sg = np.linspace(math.asinh((a_lo - lc) / h), math.asinh((a_hi - lc) / h), ns)
                xs.append(lc + h * np.sinh(sg))
                step = delta / (2.0 * h)
                if abs(math.log(c / f)) < 1e-3:
                    st_p = max(st_p, step)
                else:
                    st_s = max(st_s, step)
            if not xs:
                continue
            x = np.unique(np.clip(np.concatenate(xs), math.log(lo), math.log(hi)))
            if x.size < 2:
                continue
            f0 = np.exp(x)
            Ki = _tuning_kernel(table, ref, f, f0, T_K=T_K, eta_A=eta_A, Q_mqs=Q_mqs, noise=noise,
                                amp_model=amp_model)
            w0 = 2.0 * math.pi * f0
            r0 = w0 * ref["L_MQS_H"] / Q_mqs
            Eth0, EA0 = _noise_E(f0, T_K, eta_A, classical=True)
            Kmqs = (w0 * ref["Psi_MQS"]) ** 4 / w0 / (r0 * (Eth0 + EA0)) ** 2
            integrand = Ki * (gt(f) / gt(f0)) ** 4 / Kmqs / delta
            total += float(np.trapezoid(integrand, x) if hasattr(np, "trapezoid") else np.trapz(integrand, x))
        return total, len(roots), st_p, st_s

    th = np.zeros(ft.size)
    nr = np.zeros(ft.size, int)
    sp_ = np.zeros(ft.size)
    ss_ = np.zeros(ft.size)
    conv = np.full(ft.size, np.nan)
    for k, f in enumerate(ft):
        th[k], nr[k], sp_[k], ss_[k] = theta_at(f, n_s)
        if check_convergence and th[k] > 0:
            conv[k] = theta_at(f, 4 * n_s)[0] / th[k] - 1
    with np.errstate(divide="ignore"):
        g_ratio = th ** -0.25
    return {"f_Hz": ft, "Theta_over_Theta95": th, "g_over_gtarget": g_ratio, "n_roots": nr,
            "step_primary": sp_, "step_secondary": ss_, "convergence_rel_change": conv,
            "segments_Hz": segs, "noise": noise, "amp_model": amp_model}


def reach_2211(table: SourceTable, *, T_K: float = 0.020, eta_A: float = 20.0, Q_mqs: float = 1e6,
               B_T: float = 10.0, f_start_Hz: float = 150e6, live_time_yr: float = 1.0,
               f_end_Hz: float | None = None, f_targets=None, noise: str = "paper2211",
               amp_model: str = "matched", g_target: Callable = g_dfsz_notebook, **schedule_kw) -> dict:
    """Beta. Single-pole and multipole reach of 2211.00008 for one table  (M8-M13).

    ``g_single = g_t (g/g_MQS)`` at capacitive tunings inside the schedule ([P]'s
    single-pole accounting, which assigns each tuning ``Q_mqs/Q_B`` times the MQS dwell);
    ``g_multi = g_t Theta(f)^(-1/4)`` from :func:`multipole_theta`.
    """
    sch = mqs_schedule(table, T_K=T_K, eta_A=eta_A, Q_mqs=Q_mqs, B_T=B_T, f_start_Hz=f_start_Hz,
                       live_time_yr=live_time_yr, f_end_Hz=f_end_Hz, g_target=g_target, **schedule_kw)
    sp = single_pole_ratio(table, T_K=T_K, eta_A=eta_A, Q_mqs=Q_mqs, noise=noise)
    f = sp["f_Hz"]
    insched = (f >= sch["f_end_Hz"]) & (f <= sch["f_start_Hz"]) & (table.X_ohm > 0)
    g_single = np.where(insched, g_target(f) * sp["g_over_gMQS"], np.nan)
    out = {"schedule": sch, "single_pole": sp, "f_Hz": f, "g_single_GeV": g_single, "g_target_GeV": g_target(f)}
    if f_targets is not None:
        mp = multipole_theta(table, sch, f_targets, noise=noise, amp_model=amp_model)
        mp["g_multi_GeV"] = g_target(mp["f_Hz"]) * mp["g_over_gtarget"]
        out["multipole"] = mp
    return out


# ================================================================ 2211.00008 as computed in 2022 (M14)
# The 2022 computation behind [P]'s projections ([C], not distributed) is transcribed in
# benchmarks/multipole_scan/reference_jf2022/jf2022.py; this is the SI port.  See docs/physics/multipole_scan.md M14.
JF_Q_A = 1e6                 # axion quality factor (= 1/sig_width)
JF_F_MQS_HZ = 149.896e6
JF_LIVE_S = 365 * 86400.0
JF_F_SENS_MAX_HZ = 466.40568517314e6
JF_B_T = 10.0
JF_RHO = 0.45                # inferred: one scale B sqrt(rho) = 10 T sqrt(0.45) fits both Fig. S3 endpoints at 20 mK
JF_T_K = 0.020


def _jf_qb(table: SourceTable, Q_mqs: float, L_mqs: float):
    """[C]'s Q_Bandwidth (reconstructed, assumption A6 of jf2022.py): ``w L_eff/(R + r)``,
    ``L_eff = (X/w + dX/dw)/2`` with
    ``np.gradient``; points with Q <= 0 dropped; linear interpolation, linear extrapolation."""
    f = table.f_Hz
    w = 2.0 * math.pi * f
    Q = w * 0.5 * (table.X_ohm / w + np.gradient(table.X_ohm, w)) / (table.R_ohm + w * L_mqs / Q_mqs)
    ok = np.isfinite(Q) & (Q > 0)
    fc, qc = f[ok], Q[ok]

    def interp_q(x):
        x = np.asarray(x, float)
        y = np.interp(x, fc, qc)
        lo, hi = x < fc[0], x > fc[-1]
        y = np.where(lo, qc[0] + (x - fc[0]) * (qc[1] - qc[0]) / (fc[1] - fc[0]), y)
        return np.where(hi, qc[-1] + (x - fc[-1]) * (qc[-1] - qc[-2]) / (fc[-1] - fc[-2]), y)
    return interp_q


def _jf_step_grid(f_lo, f_hi, interp_q, sig_width):
    out = [f_lo]
    f = f_lo
    while f < f_hi:
        f = f * (1.0 + max(sig_width, 1.0 / float(interp_q(f))))
        out.append(f)
    return np.array(out)


def _jf_energies(f_Hz, T_K, eta_A, thermal):
    w = 2.0 * math.pi * np.asarray(f_Hz, float)
    th = HBAR * w * (2.0 * n_bose(f_Hz, T_K) + 1.0) if thermal == "bose" else 2.0 * KB * T_K * np.ones_like(w)
    return th, eta_A * HBAR * w


def jf2022_schedule(table: SourceTable, *, eta_A: float, T_K: float = JF_T_K, B_T: float = JF_B_T,
                    rho_GeV_cm3: float = JF_RHO, Q_a: float = JF_Q_A, Q_mqs: float = 1e6,
                    f_MQS_Hz: float = JF_F_MQS_HZ, live_time_s: float = JF_LIVE_S, thermal: str = "bose",
                    Theta: float = THETA_NOTEBOOK, g_target: Callable = g_dfsz_notebook) -> dict:
    """The 2022 scan strategy of 2211.00008  (M14).

    1. MQS schedule: tunings at ``delta = 1/Q_a`` from the table's lowest frequency to ``f_MQS``; one-second MQS
       statistic ``K = pi Q_a V_MQS^4/(2 w r^2 E^2)``, ``E = E_th + eta hbar w``; the cheapest tunings are kept while
       the cumulative time stays within ``live_time_s`` (plus one), giving the start frequency ``f_start``.
    2. Strategy: tunings ``f_{n+1} = f_n (1 + max(1/Q_a, 1/Q_B(f_n)))`` from ``f_start`` to ``f_MQS``, each dwelling
       the (interpolated) MQS time times ``max(1, Q_mqs/Q_B)``; tunings where the source is capacitive (``X < 0``)
       are dropped together with their time.
    3. Sensitivity grid: same recursion from 5 MHz (the table start) to ``JF_F_SENS_MAX_HZ``.
    """
    ref = mqs_reference(table)
    L, f0 = ref["L_MQS_H"], table.f_Hz[0]
    sig = 1.0 / Q_a
    n_a = int(np.ceil(np.log(f_MQS_Hz / f0) / np.log(1 + sig)))
    fm = np.geomspace(f0, f_MQS_Hz, n_a)
    wm = 2.0 * math.pi * fm
    rm = wm * L / Q_mqs
    Eth, EA = _jf_energies(fm, T_K, eta_A, thermal)
    Vm = source_amplitude(wm * ref["Psi_MQS"], table.source_unit, g_target(fm), B_T=B_T, rho_GeV_cm3=rho_GeV_cm3)
    K = math.pi * Q_a * Vm ** 4 / (2.0 * wm * rm ** 2 * (Eth + EA) ** 2)
    t = Theta / K
    order = np.argsort(t)
    ts = t[order]
    ts[np.searchsorted(np.cumsum(ts), live_time_s) + 1:] = np.nan
    coll = np.empty_like(ts)
    coll[order] = ts
    f_start = float(fm[np.flatnonzero(np.isfinite(coll))[0]])
    iq = _jf_qb(table, Q_mqs, L)
    hf = _jf_step_grid(f_start, f_MQS_Hz, iq, sig)
    if hf[-1] > table.f_Hz[-1]:
        raise ValueError(f"jf2022_schedule: the last tuning {hf[-1]:.6g} Hz lies above the table end "
                         f"{table.f_Hz[-1]:.6g} Hz; extend the table past f_MQS")
    # [C] multiplies by max(1, 1e6/Q) with a literal 1e6 (= its MQS_Q); the step uses 1/Q_a
    th = np.interp(hf, fm, coll) * np.maximum(1.0, Q_mqs / iq(hf))
    keep = table.at(hf)["X_ohm"] >= 0
    return {"f_start_Hz": f_start, "tunings_Hz": hf[keep], "dwell_s": th[keep], "all_tunings_Hz": hf,
            "interp_q": iq, "reference": ref, "g_target": g_target, "mqs_f_Hz": fm, "mqs_time_s": coll,
            "params": {"eta_A": eta_A, "T_K": T_K, "B_T": B_T, "rho_GeV_cm3": rho_GeV_cm3, "Q_a": Q_a,
                       "Q_mqs": Q_mqs, "f_MQS_Hz": f_MQS_Hz, "live_time_s": live_time_s, "thermal": thermal,
                       "Theta": Theta}}


def jf2022_sensitivity_grid(table: SourceTable, schedule: dict, f_max_Hz: float = JF_F_SENS_MAX_HZ) -> np.ndarray:
    """[C]'s sensitivity grid (the default f_max is [C]'s 466.406 MHz for the toroid)."""
    return _jf_step_grid(table.f_Hz[0], f_max_Hz, schedule["interp_q"], 1.0 / schedule["params"]["Q_a"])


def jf2022_theta(table: SourceTable, schedule: dict, f_targets, *, chunk: int = 32,
                 amp_model: str = "jf2022") -> dict:
    """``Theta(f) = sum_k t_k K_k(f)`` over the discrete strategy of :func:`jf2022_schedule`  (M14).

    Per second ``K_k(f) = Q_a/(2 pi w) |I|^4 / S^2`` with ``I = V(f)/|Z_k(f)|``,
    ``Z_k(f) = R(f) + r(f) + j(X(f) - X(f_k) f_k/f)`` (capacitor tuned at ``f_k``) and the current PSD
    ``S = E_th r(f)/(pi |Z_k|^2) + eta hbar w_k/(2 pi R_k) [(w/w_k)^2 + R_k^2/|Z_k|^2]``, ``R_k = R(f_k) + r(f_k)``.
    ``amp_model="m13"`` swaps the frequency weights of the two amplifier terms (axionFEM's earlier M13 form,
    ``[1 + (w/w_k)^2 R_k^2/|Z_k|^2]``), kept for the comparison.  Returns ``g = g_t (Theta_95/Theta)^(1/4)``.
    """
    p = schedule["params"]
    ref = schedule["reference"]
    eta, Q_a, Q_mqs = p["eta_A"], p["Q_a"], p["Q_mqs"]
    L = ref["L_MQS_H"]
    gt = schedule["g_target"]
    fk = schedule["tunings_Hz"]
    tk = schedule["dwell_s"]
    b = table.at(fk)
    if np.any(b["X_ohm"] < 0):
        raise ValueError("jf2022_theta: a tuning has X(f_k) < 0 (source capacitive); [C] masks those, and only the "
                         "capacitor branch is implemented")
    wk = 2.0 * math.pi * fk
    Rk = b["R_ohm"] + wk * L / Q_mqs
    uk = b["X_ohm"] * fk                         # X(f_k) f_k: capacitor reactance X(f_k) f_k / f
    Bk = eta * HBAR * wk / (2.0 * math.pi * Rk)
    ft = np.atleast_1d(np.asarray(f_targets, float))
    a = table.at(ft)
    w = 2.0 * math.pi * ft
    r = w * L / Q_mqs
    Eth, _ = _jf_energies(ft, p["T_K"], eta, p["thermal"])
    A = Eth * r / math.pi
    V = source_amplitude(a["V"], table.source_unit, gt(ft), B_T=p["B_T"], rho_GeV_cm3=p["rho_GeV_cm3"])
    cs = Q_a / (2.0 * math.pi * w) * V ** 4
    Rt = a["R_ohm"] + r
    th = np.zeros(ft.size)
    for i0 in range(0, ft.size, chunk):
        s = slice(i0, i0 + chunk)
        Xt = a["X_ohm"][s, None] - uk / ft[s, None]
        Z2 = Xt * Xt + Rt[s, None] ** 2
        q = (w[s, None] / wk) ** 2
        if amp_model == "jf2022":
            D = A[s, None] + Bk * Rk ** 2 + Bk * q * Z2
        elif amp_model == "m13":
            D = A[s, None] + Bk * Z2 + Bk * Rk ** 2 * q
        else:
            raise ValueError("jf2022_theta: amp_model must be 'jf2022' or 'm13'")
        th[s] = cs[s] * (tk / (D * D)).sum(axis=1)
    with np.errstate(divide="ignore"):
        g = gt(ft) * (p["Theta"] / th) ** 0.25
    return {"f_Hz": ft, "Theta": th, "g_GeV": g, "g_over_gtarget": g / gt(ft), "amp_model": amp_model}


def jf2022_plot_curve(f_Hz, g, sigma: float = 101, every: int = 100):
    """What 2211.00008's multipole figures draw ([C]): ``gaussian_filter(g, 101)[::100]`` on the
    sensitivity grid (index-space smoothing of g, then decimation)."""
    from scipy import ndimage
    return np.asarray(f_Hz)[::every], ndimage.gaussian_filter(np.asarray(g, float), sigma)[::every]
