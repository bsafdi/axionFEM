"""SRF heterodyne axion search: scan rate, optimal coupling, required volume, RF load and cost.

Beta.  Pure Python; no FEM.  The cavity enters only through the geometry figures of merit of
:class:`SRFGeometry` (xi, eta_10, G, B_pk/B_0, A_cav/V^(2/3)), which the eigenmode side (:mod:`axionfem.srf`)
computes.  Presets: :data:`PROJECTX_086` (= ``PROJECTX_086_WALL``; the Project X 650 MHz beta = 0.9 cell at the
axial factor that makes TM020/TE011 degenerate, computed and cached in a record; B_pk = the largest wall |B|, the
default geometry of :class:`ScenarioA`) and the diagnostic ``PROJECTX_086_EQUATOR_DIAGNOSTIC`` (B_pk := equator |B|;
not a field limit).  Every formula is in ``docs/srf/sensitivity.md``.

The scan model (model A in the names: :class:`ScenarioA`, :func:`design_A`) is one scaled elliptical cell,
omega_0 = xi c V^(-1/3), Q = G/R_s, the weak-signal scan rate (:func:`scan_rate`) with the receiver factor G-bar_SRF
(:func:`gbar_srf`) and the coupling optimised at every mass, an optional adiabatic scan schedule
(:func:`tracking_rate`), the exact Asimov statistic (:func:`required_volume_exact`) and a parametric cost metric
(:class:`CostLawA`).  :class:`ScenarioA`, :func:`required_volume`, :func:`design_A`, :func:`cost_live`, presets
:data:`SCENARIOS_A`.  It rests on the physics core: the signal power (:func:`mode_power`), the receiver factor
:func:`gbar_srf` with its exact optimum :func:`beta_opt`, and the scan rate :func:`scan_rate`.

Conventions
-----------
* SI internally; inputs carry their unit in the name.  ``eta`` is the overlap eta_10 itself
  (C_01 = eta^2 in Lasenby, arXiv:1912.11056).
* ``B0_T`` is the spatial RMS of the peak-in-time pump field, B_0^2 = V^-1 int |B0~|^2, so the stored
  pump energy is U_0 = B_0^2 V/(2 mu0).
* Rates are d(nu_1 - nu_0)/dt in Hz/s (= d nu_a/dt); the angular-frequency rate is 2 pi times this.
* The DFSZ target, rho_DM, Q_axion, Theta_95, the band and live time are the defaults of :class:`Target`
  (:data:`DEFAULT_TARGET`, an illustrative default: pass a :class:`Target` for your own search) and use the package's QCD-line coupling (:func:`axionfem.scan.g_qcd`,
  lattice C_gamma = 1.77, m_a f_a = 5.70e-3 GeV^2) through :func:`axionfem.haloscope.g_axion`.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from . import haloscope as _h
from .circuit import EV_J, HBAR, KB, MU0
from .haloscope import C_LIGHT, H_PLANCK
from .scan import YEAR


# =============================================================================
# 0. Scan target
# =============================================================================
@dataclass(frozen=True)
class Target:
    """Scan target.  The defaults are illustrative, not a recommended search: DFSZ (E/N = 8/3, lattice
    C_gamma = 1.77) over m_a = 0.5-4.1 neV (120.9 kHz - 1 MHz) in 3 live years, rho_DM = 0.3 GeV/cm^3,
    Q_axion = 6.8e5, Theta_95 = 2.71.  Set the band, live time and model of your own study."""

    f_min_Hz: float = 120.9e3
    f_max_Hz: float = 1.0e6
    live_time_yr: float = 3.0
    rho_GeV_cm3: float = 0.3
    Q_axion: float = 6.8e5
    Theta: float = 2.71
    model: str = "DFSZ"
    C_gamma: float = 1.77

    @property
    def live_time_s(self) -> float:
        return self.live_time_yr * YEAR

    def g_GeV(self, f_a_Hz):
        """|g_agg| [GeV^-1] of the target at axion frequency f_a (QCD line, vectorised)."""
        m = H_PLANCK * np.asarray(f_a_Hz, dtype=float) / EV_J
        return _h.g_axion(m, self.model, convention="qcd", C_gamma=self.C_gamma)


#: The default scan target (all :class:`Target` defaults; illustrative).
DEFAULT_TARGET = Target()


def occupation(f_Hz, T_K):
    """Bose occupation 1/expm1(h f/k T) (vectorised; 0 for T <= 0).  Same values as
    :func:`axionfem.haloscope.n_thermal` (pinned by a test), but pure numpy instead of np.vectorize over the
    scalar C++ helper, which matters inside the root-finding loops here."""
    f = np.asarray(f_Hz, dtype=float)
    T = np.asarray(T_K, dtype=float)
    with np.errstate(over="ignore", divide="ignore"):
        x = H_PLANCK * f / (KB * np.where(T > 0, T, 1.0))
        return np.where(T > 0, 1.0 / np.expm1(x), 0.0)


# =============================================================================
# 1. Physics core
# =============================================================================
def mode_power(g_GeV, f1_Hz, B0_T, V_m3, eta, Q_1, rho_GeV_cm3: float = 0.3):
    """Signal power deposited in the signal mode on resonance, narrow axion line [W]:

        P_sig = (1/4) g^2 eta^2 B_0^2 rho_DM V Q_1/omega_1         (natural units)

    = Q_1/4 x :func:`axionfem.haloscope.power_coefficient` with omega -> omega_1 and C -> eta^2
    (one sideband carries half the effective-current amplitude of a static field of amplitude B_0).
    The power leaving through the port is beta/(1+beta) of this; at the amplifier, tau times that."""
    return 0.25 * np.asarray(Q_1) * _h.power_coefficient(g_GeV, f1_Hz, B0_T, V_m3, np.asarray(eta) ** 2,
                                                         rho_GeV_cm3)


def noise_flat(n_add, tau: float = 1.0, n_e=0.0, n_l=0.0):
    """A = 1/2 + n_add + tau n_e + (1 - tau) n_l: flat noise in quanta/Hz."""
    return 0.5 + np.asarray(n_add) + tau * np.asarray(n_e) + (1.0 - tau) * np.asarray(n_l)


def noise_resonant(beta, n_i, tau: float = 1.0, n_e=0.0):
    """D = 4 tau beta (n_i - n_e)/(1 + beta)^2: cavity thermal noise, filtered like the signal."""
    beta = np.asarray(beta, dtype=float)
    return 4.0 * tau * beta * (np.asarray(n_i) - np.asarray(n_e)) / (1.0 + beta) ** 2


def gbar_srf(beta, A, n_i, tau: float = 1.0, n_e=0.0):
    """G-bar_SRF = tau^2 beta^2 / [(1+beta)^3 sqrt(A) (A + D)^(3/2)]: the receiver factor of :func:`scan_rate`."""
    beta = np.asarray(beta, dtype=float)
    A = np.asarray(A, dtype=float)
    D = noise_resonant(beta, n_i, tau, n_e)
    return tau ** 2 * beta ** 2 / ((1.0 + beta) ** 3 * np.sqrt(A) * (A + D) ** 1.5)


def beta_opt(A, n_i, tau: float = 1.0, n_e=0.0):
    """Exact maximiser of :func:`gbar_srf` over beta.

    d ln G/d beta = 0 reduces to beta^2 - (1 + k/2) beta - 2 = 0 with k = 4 tau (n_i - n_e)/A, so
    beta* = [(1 + k/2) + sqrt((1 + k/2)^2 + 8)]/2  (-> 2 for k -> 0, -> 2 tau (n_i - n_e)/A for k >> 1)."""
    k = 4.0 * tau * (np.asarray(n_i, dtype=float) - np.asarray(n_e)) / np.asarray(A, dtype=float)
    b = 1.0 + 0.5 * k
    return 0.5 * (b + np.sqrt(b * b + 8.0))


def gbar_srf_asymptotic(f1_Hz, T_K, A, tau: float = 1.0):
    """Large-occupation optimum G-bar_SRF ~ tau omega_1/(6 sqrt3 A T)  (hbar = k_B = 1)."""
    return tau * H_PLANCK * np.asarray(f1_Hz) / (6.0 * math.sqrt(3.0) * np.asarray(A) * KB * np.asarray(T_K))


def lumped_substitution(beta, A, n_i, tau: float = 1.0, n_e=0.0) -> dict:
    """Correspondence to the lumped-element G-bar(xi; n, eta_A) of :func:`axionfem.circuit.G_bar`:
    xi -> A/(tau beta), eta_A -> A/tau, 2n + 1 -> 2(n_i - n_e) + A/tau (2 n_i + A/tau for n_e = 0)."""
    return dict(xi=A / (tau * beta), eta_A=A / tau, n=(n_i - n_e) + 0.5 * A / tau - 0.5)


def scan_rate(f_a_Hz, f1_Hz, V_m3, B0_T, eta, Q_i, A, n_i, *, beta=None, tau: float = 1.0, n_e=0.0,
              target: Target = DEFAULT_TARGET, g_GeV=None):
    """Weak-signal dense-scan rate d(nu_1 - nu_0)/dt [Hz/s]  (the angular-frequency rate divided by 2 pi):

        d(omega_1 - omega_0)/dt = (pi^2/32) (Q_axion/Theta_95) g^4 rho^2 eta^4 B_0^4 V^2 Q_i
                                  / (m_a omega_1^3) x G-bar_SRF(beta).

    In SI with P0 = g^2 rho eta^2 B_0^2 V/omega_1 [W] (= 4 P_sig/Q_1) this is
    (pi^2/32)(Q_a/Theta) P0^2 Q_i G-bar/(hbar^2 omega_a omega_1).  ``beta`` None -> :func:`beta_opt` at
    every mass.  Vectorised over f_a (and any broadcastable argument)."""
    f_a = np.asarray(f_a_Hz, dtype=float)
    g = target.g_GeV(f_a) if g_GeV is None else np.asarray(g_GeV, dtype=float)
    if beta is None:
        beta = beta_opt(A, n_i, tau, n_e)
    P0 = _h.power_coefficient(g, f1_Hz, B0_T, V_m3, np.asarray(eta) ** 2, target.rho_GeV_cm3)
    G = gbar_srf(beta, A, n_i, tau, n_e)
    w_a = 2.0 * math.pi * f_a
    w_1 = 2.0 * math.pi * np.asarray(f1_Hz, dtype=float)
    dw_dt = (math.pi ** 2 / 32.0) * (target.Q_axion / target.Theta) * P0 ** 2 * np.asarray(Q_i) * G / (
        HBAR ** 2 * w_a * w_1)
    return dw_dt / (2.0 * math.pi)


def scan_rate_from_psd(f_a_Hz, f1_Hz, V_m3, B0_T, eta, Q_i, A, n_i, *, beta, tau: float = 1.0, n_e=0.0,
                       target: Target = DEFAULT_TARGET, n: int = 4001):
    """Independent route for tests: the same rate by direct quadrature of the per-tuning Fisher
    density int dnu [S_a/S_N]^2 with one-sided PSDs S_a = tau P_ext L(nu - nu_c) p_a,
    S_N = h nu_1 [A + D L], L = 1/(1 + 4 delta^2/Delta_c^2), Delta_c = nu_1 (1+beta)/Q_i, and
    d nu_c/dt = (Q_a/(nu_a Theta)) int d delta [tau P_ext L/(h nu_1 (A + D L))]^2."""
    g = float(target.g_GeV(f_a_Hz))
    Q_L = Q_i / (1.0 + beta)
    P_ext = float(mode_power(g, f1_Hz, B0_T, V_m3, eta, Q_L, target.rho_GeV_cm3)) * beta / (1.0 + beta)
    D = float(noise_resonant(beta, n_i, tau, n_e))
    width = f1_Hz * (1.0 + beta) / Q_i
    th, w = np.polynomial.legendre.leggauss(n)
    th = 0.5 * math.pi * th * (1 - 1e-12)
    w = 0.5 * math.pi * w
    s = 0.5 * _sensitivity_width(width, A, D)
    delta = s * np.tan(th)
    jac = s / np.cos(th) ** 2
    L = 1.0 / (1.0 + 4.0 * delta ** 2 / width ** 2)
    ratio = tau * P_ext * L / (H_PLANCK * f1_Hz * (A + D * L))
    I = float(np.sum(w * jac * ratio ** 2))
    return target.Q_axion / (f_a_Hz * target.Theta) * I


def _sensitivity_width(width_Hz, A, D):
    """FWHM of L/(A + D L) in the detuning: Delta_c sqrt((A + D)/A).  It is narrower than the cavity width when the
    port bath is hotter than the walls (n_e > n_i, D < 0); the tan substitutions below scale with it so that they
    resolve the integrand in both cases."""
    return width_Hz * math.sqrt(max((A + D) / A, 1e-300))


def _gl_log(f_min: float, f_max: float, n: int):
    """Gauss-Legendre nodes in ln f on [f_min, f_max]; weights for int df (= int f d ln f)."""
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = math.log(f_min), math.log(f_max)
    lf = 0.5 * (b - a) * (x + 1.0) + a
    f = np.exp(lf)
    return f, 0.5 * (b - a) * w * f


def scan_time(rate_fn: Callable[[np.ndarray], np.ndarray], f_min_Hz: float, f_max_Hz: float, n: int = 256) -> float:
    """t = int df_a / (d nu_a/dt) over the band [s] (Gauss-Legendre in ln f; the rate is smooth)."""
    f, w = _gl_log(f_min_Hz, f_max_Hz, n)
    return float(np.sum(w / rate_fn(f)))


# =============================================================================
# 2. Geometry figures of merit (the join to the eigenmode side, axionfem.srf)
# =============================================================================
#: Isoperimetric minimum of A/V^(2/3) (a sphere): (36 pi)^(1/3) = 4.836.
AREA_COEFF_MIN = (36.0 * math.pi) ** (1.0 / 3.0)


@dataclass(frozen=True)
class SRFGeometry:
    """Scale-free figures of merit of a cavity and its (pump, signal) mode pair.

    xi:            omega_0 = xi c V^(-1/3)  (angular frequency; f_0 V^(1/3)/c is xi/2pi)
    eta:           overlap eta_10 (C_01 = eta^2), NOT its square
    G_pump_ohm, G_signal_ohm: geometry factors in ohm, G = omega mu0 int|H|^2 dV / int|H_t|^2 dA, Q = G/R_s
                   (the dimensionless natural-unit G times Z0 = 376.73 ohm)
    Bpk_over_B0:   peak wall magnetic field of the pump / B_0 (B_0 = spatial RMS, peak in time),
                   i.e. the largest |B| anywhere on the wall
    area_coeff:    A_cav / V^(2/3)  (>= 4.836, the sphere)
    Epk_over_cB0:  peak wall electric field of the pump / (c B_0), dimensionless, informational: the pump's peak
                   surface E is Epk_over_cB0 * c * B_0 (NOT times c B_pk).

    The constructor validates every value (finite, in a physically possible range, and in the units above) and
    raises ValueError naming the field, so that a slipped convention (eta^2 for eta is the one it cannot catch;
    xi without 2 pi, G in natural units, B0/B_pk inverted, an impossible area) fails loudly instead of returning
    a plausible-looking design."""

    xi: float
    eta: float
    G_pump_ohm: float
    G_signal_ohm: float
    Bpk_over_B0: float
    area_coeff: float
    Epk_over_cB0: float | None = None
    name: str = ""
    source: str = ""

    def __post_init__(self):
        where = f"SRFGeometry({self.name or 'unnamed'})"
        for f in ("xi", "eta", "G_pump_ohm", "G_signal_ohm", "Bpk_over_B0", "area_coeff", "Epk_over_cB0"):
            v = getattr(self, f)
            if v is None and f == "Epk_over_cB0":
                continue
            try:
                ok = v is not None and not isinstance(v, bool) and math.isfinite(float(v))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"{where}: {f} = {v!r} is not a finite number (a figures-of-merit dict without that "
                                 "quantity, e.g. figures_of_merit_3d without overlaps, cannot build a geometry)")
            object.__setattr__(self, f, float(v))
        if not 0.0 < self.eta <= 1.0:
            raise ValueError(f"{where}: eta = {self.eta} must lie in (0, 1] (it is a normalized overlap |cos|)")
        if not 2.0 <= self.xi <= 100.0:
            raise ValueError(f"{where}: xi = {self.xi} is outside [2, 100]; xi = omega_0 V^(1/3)/c with the ANGULAR "
                             f"frequency (the computed Project X cell: 7.64; xi/2pi = {self.xi / (2 * math.pi):.3g} would be "
                             "f0 V^(1/3)/c)")
        for f in ("G_pump_ohm", "G_signal_ohm"):
            g = getattr(self, f)
            if not 10.0 <= g <= 1e5:
                raise ValueError(f"{where}: {f} = {g} is outside [10, 1e5] ohm; G must be in ohm (natural-unit G "
                                 "times Z0 = 376.73 ohm; the computed Project X cell: 524 / 604 ohm)")
        if not 1.0 <= self.Bpk_over_B0 <= 10.0:
            raise ValueError(f"{where}: Bpk_over_B0 = {self.Bpk_over_B0} is outside [1, 10]; it is the peak WALL |B| "
                             "over the volume rms B0 (the computed Project X cell: 1.464), not B0/B_pk")
        if self.area_coeff < AREA_COEFF_MIN * (1 - 1e-9) or self.area_coeff > 100.0:
            raise ValueError(f"{where}: area_coeff = A/V^(2/3) = {self.area_coeff} is outside "
                             f"[{AREA_COEFF_MIN:.4f} (a sphere, the isoperimetric minimum), 100]")
        if self.Epk_over_cB0 is not None and not 0.0 <= self.Epk_over_cB0 <= 100.0:
            raise ValueError(f"{where}: Epk_over_cB0 = {self.Epk_over_cB0} is outside [0, 100] (E_pk/(c B0), "
                             "dimensionless; the computed Project X cell: 1.843)")

    def f0_Hz(self, V_m3):
        return self.xi * C_LIGHT * np.asarray(V_m3, dtype=float) ** (-1.0 / 3.0) / (2.0 * math.pi)

    def V_m3(self, f0_Hz):
        return (self.xi * C_LIGHT / (2.0 * math.pi * np.asarray(f0_Hz, dtype=float))) ** 3

    def area_m2(self, V_m3):
        return self.area_coeff * np.asarray(V_m3, dtype=float) ** (2.0 / 3.0)

    def Epk_V_per_m(self, B0_T):
        """Peak surface electric field of the pump [V/m] at volume-rms field B0_T: Epk_over_cB0 c B0."""
        if self.Epk_over_cB0 is None:
            raise ValueError(f"SRFGeometry({self.name or 'unnamed'}): Epk_over_cB0 is not set")
        return self.Epk_over_cB0 * C_LIGHT * B0_T

    @classmethod
    def from_modes(cls, modes: dict, *, bpk: str = "wall", **kw) -> "SRFGeometry":
        """Build from the eigenmode side's figures of merit.

        ``modes`` is either the output of :func:`axionfem.srf.figures_of_merit` (keys ``xi_pump``, ``eta_10``,
        ``G_pump_ohm``, ``G_signal_ohm``, ``Bpk_over_B0``, ``B_equator_over_B0``, ``A_over_V23``, ``Epk_over_cB0``) or a dict with this
        class's own field names.  ``bpk`` picks the peak-field value:

        * ``"wall"`` (default): the largest pump |B| anywhere on the wall, 1.464 for the Project X cell shortened by 0.86 (iris-side
          lobe).  This is the peak wall magnetic field, to which the field limit applies.
        * ``"equator"``: DIAGNOSTIC ONLY, the pump |B| at the equator (1.404).  It is not the peak wall field: a
          design built on it runs the iris-side wall 4.3 % above the stated field limit.  For an own-key dict it
          needs a ``B_equator_over_B0`` entry.

        ``kw`` overrides any field.  The result is checked by :class:`SRFGeometry`."""
        if bpk not in ("wall", "equator"):
            raise ValueError(f"SRFGeometry.from_modes: bpk must be 'wall' or 'equator', got {bpk!r}")
        if "eta_10" in modes or "xi_pump" in modes:
            try:
                keys = dict(xi=modes["xi_pump"], eta=modes["eta_10"], G_pump_ohm=modes["G_pump_ohm"],
                            G_signal_ohm=modes["G_signal_ohm"],
                            Bpk_over_B0=modes["Bpk_over_B0" if bpk == "wall" else "B_equator_over_B0"],
                            area_coeff=modes["A_over_V23"], Epk_over_cB0=modes.get("Epk_over_cB0"))
            except KeyError as e:
                raise KeyError(f"SRFGeometry.from_modes: the figures-of-merit dict lacks {e}; pass the output of "
                               "axionfem.srf.figures_of_merit (2D) or a dict with SRFGeometry's own keys") from None
        else:
            keys = {k: modes[k] for k in cls.__dataclass_fields__ if k in modes}
            if bpk == "equator":
                if "B_equator_over_B0" not in modes:
                    raise ValueError("SRFGeometry.from_modes: bpk='equator' with an own-key dict needs a "
                                     "'B_equator_over_B0' entry (its 'Bpk_over_B0' is the wall maximum)")
                keys["Bpk_over_B0"] = modes["B_equator_over_B0"]
        keys.update(kw)
        return cls(**keys)


#: Cached record of the computed figures of merit (written by benchmarks/srf/srf_geometry_record.py).
PROJECTX_RECORD = Path(__file__).resolve().parent / "data" / "srf_projectx086_geometry.json"

#: What the record must have been computed for (the preset definition; checked on every load).
PROJECTX_RECORD_INPUTS = {"axial_factor": 0.86, "tube_radius_m": 0.050, "tube_length_m": 0.100, "pump": "TM020",
                          "signal": "TE011", "wall": "smooth"}
#: Minimum resolution of the record's reference mesh (R_eq/h and polynomial order).
PROJECTX_RECORD_MIN_MESH = {"n": 80, "order": 3}


def srf_source_sha256() -> str:
    """sha256 of python/axionfem/srf.py (the eigenmode code behind the record), as stored in the record."""
    import hashlib
    return hashlib.sha256((Path(__file__).resolve().parent / "srf.py").read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _load_record(path: str) -> dict:
    import json
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SRF geometry record {p} not found: regenerate it with "
                                "`python benchmarks/srf/srf_geometry_record.py --update-package-data` "
                                "(the computed presets PROJECTX_086* and GEOMETRIES['computed_wall' | "
                                "'equator_diagnostic'] need it)")
    return json.loads(p.read_text())


def check_record(rec: dict, *, path: str = "record") -> None:
    """Check a geometry record against the preset definition: cell, tubes, mode pair and mesh (ValueError on a
    mismatch, e.g. a record regenerated with --axial-factor 0.85 or --n 20); warn if srf.py changed since, or the
    tree was dirty when it was written."""
    from . import srf as _srf
    inp = rec.get("inputs")
    if not isinstance(inp, dict):
        raise ValueError(f"SRF geometry {path} carries no 'inputs' fingerprint (schema {rec.get('schema')!r}); "
                         "regenerate it with benchmarks/srf/srf_geometry_record.py")
    bad = []
    cell = inp.get("cell") or {}
    for k, v in _srf.PROJECTX_650_B09.items():
        got = cell.get(k)
        if got is None or not np.allclose(np.atleast_1d(got), np.atleast_1d(v), rtol=0, atol=1e-12):
            bad.append(f"cell.{k} = {got!r} (expected {v!r})")
    for k, v in PROJECTX_RECORD_INPUTS.items():
        got = inp.get(k)
        if (got != v) if isinstance(v, str) else (got is None or abs(float(got) - v) > 1e-12):
            bad.append(f"{k} = {got!r} (expected {v!r})")
    m = rec.get("figures_of_merit", {}).get("mesh", {})
    n = 1.0 / m["h_over_Req"] if m.get("h_over_Req") else 0.0
    if n < PROJECTX_RECORD_MIN_MESH["n"] - 1e-6 or (m.get("order") or 0) < PROJECTX_RECORD_MIN_MESH["order"]:
        bad.append(f"reference mesh R_eq/h = {n:.0f}, order {m.get('order')} (need >= "
                   f"{PROJECTX_RECORD_MIN_MESH['n']}, >= {PROJECTX_RECORD_MIN_MESH['order']})")
    if bad:
        raise ValueError(f"SRF geometry {path} does not describe the preset (Project X 650 MHz beta = 0.9 cell x0.86, "
                         "50 mm x 100 mm tubes, TM020/TE011, order >= 3, h <= R_eq/80): " + "; ".join(bad)
                         + ". Regenerate it with `python benchmarks/srf/srf_geometry_record.py --update-package-data`")
    fp = rec.get("fingerprint", {})
    if fp.get("srf_py_sha256") != srf_source_sha256():
        warnings.warn(f"SRF geometry {path}: the record was computed with a different version of axionfem.srf "
                      f"(git {fp.get('git_describe', '?')}); its figures of merit may not match this version",
                      stacklevel=3)
    if fp.get("dirty"):
        warnings.warn(f"SRF geometry {path} was written from a dirty tree ({fp.get('git_describe')})", stacklevel=3)


@lru_cache(maxsize=None)
def _projectx_geometry(bpk: str, record: str) -> SRFGeometry:
    rec = _load_record(record)
    check_record(rec, path=record)
    fm = rec["figures_of_merit"]
    m = fm.get("mesh", {})
    n = f"{1 / m['h_over_Req']:.0f}" if m.get("h_over_Req") else "?"
    gen = rec.get("generated", {})
    name = ("Project X cell x0.86, computed (B_pk = largest wall |B|)" if bpk == "wall" else
            "Project X cell x0.86, computed, B_pk := equator |B| (DIAGNOSTIC; not the peak wall field)")
    return SRFGeometry.from_modes(
        fm, bpk=bpk, name=name,
        source=f"axionfem.srf 2D, order {m.get('order')}, h = R_eq/{n} ({gen.get('by', '?')}, "
               f"git {rec.get('fingerprint', {}).get('git_describe', gen.get('git', '?'))})")


def projectx_geometry(bpk: str = "wall", record: str | Path | None = None) -> SRFGeometry:
    """The Project X 650 MHz beta = 0.9 cell at axial factor 0.86 (TM020/TE011 degenerate, see :mod:`axionfem.srf`)
    with 50 mm x 100 mm tubes, with the figures of merit computed by :mod:`axionfem.srf` (2D, order 3,
    h = R_eq/80; cached in :data:`PROJECTX_RECORD`).  The record is checked against the preset definition on load
    (:func:`check_record`).

    ``bpk="wall"`` (default; B_pk/B0 = 1.464, the largest wall |B|, to which a field limit applies) or
    ``"equator"`` (1.404, diagnostic only; see :meth:`SRFGeometry.from_modes`)."""
    if bpk not in ("wall", "equator"):
        raise ValueError(f"projectx_geometry: bpk must be 'wall' or 'equator', got {bpk!r}")
    return _projectx_geometry(bpk, str(record or PROJECTX_RECORD))


class _Geometries(Mapping):
    """Geometry presets by short name; the computed ones are loaded (and checked) on first access, so a missing
    record raises FileNotFoundError there and not as a None deep inside a scenario."""

    _keys = ("computed_wall", "equator_diagnostic")
    _alias = {"computed": "computed_wall"}

    def __getitem__(self, key):
        key = self._alias.get(key, key)
        if key == "computed_wall":
            return projectx_geometry("wall")
        if key == "equator_diagnostic":
            return projectx_geometry("equator")
        raise KeyError(f"GEOMETRIES: no preset {key!r}; have {list(self._keys)}")

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"GEOMETRIES({list(self._keys)})"


#: All geometry presets by short name: "computed_wall" (PROJECTX_086, the default) and "equator_diagnostic"
#: (PROJECTX_086_EQUATOR_DIAGNOSTIC).  "computed" is an alias of "computed_wall".
GEOMETRIES = _Geometries()


def __getattr__(name):
    # Computed presets, loaded lazily from the cached record (a missing record raises FileNotFoundError here).
    #   PROJECTX_086 = PROJECTX_086_WALL: B_pk/B0 = 1.464, the largest pump |B| on the wall (the peak wall
    #     magnetic field, to which a field limit applies); the default.
    #   PROJECTX_086_EQUATOR_DIAGNOSTIC: B_pk/B0 := 1.404, the pump |B| at the equator.  NOT the peak wall
    #     field: with it the iris-side wall exceeds the field limit by 4.3 %.  Kept as a diagnostic.
    if name in ("PROJECTX_086", "PROJECTX_086_WALL"):
        return projectx_geometry("wall")
    if name == "PROJECTX_086_EQUATOR_DIAGNOSTIC":
        return projectx_geometry("equator")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# 3. Surface resistance
# =============================================================================
def rs_bcs_repo(f_Hz, T_K: float, R_res_ohm: float = 0.0, *, log_factor: bool = False):
    """R_s(f) [ohm] of niobium from the package's ``bcs`` rf model (A = 2e-4 ohm K at 1.5 GHz,
    Delta/kT_c = 1.91, T_c = 9.25 K; docs/physics/surface_resistance.md S3), via the C++ evaluator."""
    from .cavity import surface_resistance
    rf = {"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 17.67 / 9.25, "Tc_K": 9.25,
          "R_res_ohm": R_res_ohm, "T_K": T_K, "log_factor": log_factor}
    return surface_resistance(rf, f_Hz)


# =============================================================================
# 4. Model A: single scaled cell
# =============================================================================
@dataclass(frozen=True)
class ScenarioA:
    """One model-A benchmark.

    ``A`` is the total flat noise in quanta, 1/2 + n_add + tau n_e + (1 - tau) n_l, given directly (the presets:
    A = 1 / 10 with tau = 1, n_e = n_l = 0).  ``tau`` and ``n_e`` enter only the signal transmission and the
    resonant term D; changing ``tau`` does NOT update ``A`` (build A with :func:`noise_flat` if the line terms
    matter).  ``Rs_ohm`` is a constant or a callable R_s(f, T); the signal mode uses G_signal, the pump G_pump.

    ``T_K`` is the wall temperature.  It sets both the wall occupation n_i and the refrigerator factor
    (:func:`refrigerator_factor` unless ``refrigerator_factor`` is given).  The presets use 2 K (optimistic) and
    4.5 K (conservative).

    ``geometry`` defaults to :data:`PROJECTX_086`, the computed cell (loaded from its record on first use)."""

    name: str
    Rs_ohm: float | Callable
    Bpk_T: float
    A: float
    T_K: float
    geometry: SRFGeometry = field(default_factory=lambda: projectx_geometry("wall"))
    tau: float = 1.0
    n_e: float = 0.0
    refrigerator_factor: float | None = None   # P_eq/P at T; None -> refrigerator_factor(T_K): 1 at 4.5 K, 3.3 at 2 K

    @property
    def B0_T(self) -> float:
        return self.Bpk_T / self.geometry.Bpk_over_B0

    def Rs(self, f_Hz):
        return self.Rs_ohm(f_Hz, self.T_K) if callable(self.Rs_ohm) else self.Rs_ohm


class _ScenariosA(Mapping):
    """The model-A benchmarks by name, built on first access (their default geometry, :data:`PROJECTX_086`, is
    read from the geometry record; a missing record raises FileNotFoundError there, not at import). They are
    illustrative presets: replace them with the parameters of the cavity and readout under study."""

    _keys = ("optimistic", "conservative")

    def __init__(self):
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(f"SCENARIOS_A: no benchmark {key!r}; have {list(self._keys)}")
        if key not in self._cache:
            self._cache[key] = (
                ScenarioA("SRF optimistic", Rs_ohm=1.5e-9, Bpk_T=0.150, A=1.0, T_K=2.0) if key == "optimistic" else
                ScenarioA("SRF conservative", Rs_ohm=20e-9, Bpk_T=0.090, A=10.0, T_K=4.5))
        return self._cache[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"SCENARIOS_A({list(self._keys)})"


#: Model-A benchmarks: "optimistic" (R_s = 1.5 nOhm, B_pk = 150 mT, A = 1, 2 K) and "conservative" (20 nOhm,
#: 90 mT, A = 10, 4.5 K), both on the default geometry :data:`PROJECTX_086`.
SCENARIOS_A = _ScenariosA()


def tracking_rate(beta, f_a, f1, Q_i, eps: float = 0.1, Q_axion: float = 6.8e5):
    """Sufficient adiabatic-scan condition (optional: ``adiabatic_eps`` of the sizing functions imposes it):
    nu_dot <= eps pi Delta_c min(Delta_c, nu_a/Q_axion) [Hz/s], Delta_c = nu_1 (1+beta)/Q_i the loaded width."""
    width = np.asarray(f1) * (1.0 + np.asarray(beta)) / Q_i
    return eps * math.pi * width * np.minimum(width, np.asarray(f_a) / Q_axion)


def adiabatic_schedule(stat_rate: Callable, track: Callable, beta0):
    """(rate, beta): beta0 (the statistical optimum) where stat_rate(beta0) <= track(beta0); elsewhere beta is
    raised to the crossing stat_rate(beta) = track(beta) (above beta0 the statistical rate falls and the
    tracking rate rises, so the crossing is unique), and the rate is min(stat, track) there."""
    b0 = np.asarray(beta0, dtype=float)
    limited = stat_rate(b0) > track(b0)
    lo, hi = np.log(b0), np.full_like(b0, math.log(1e8))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fast = stat_rate(np.exp(mid)) > track(np.exp(mid))
        lo = np.where(limited & fast, mid, lo)
        hi = np.where(limited & ~fast, mid, hi)
    beta = np.where(limited, np.exp(0.5 * (lo + hi)), b0)
    return np.minimum(stat_rate(beta), track(beta)), beta


# -----------------------------------------------------------------------------
# 4b. Beyond the weak-signal statistic: per-bin SNR and the exact Asimov density
# -----------------------------------------------------------------------------
def asimov_density(r, exact: bool = True):
    """Asimov test-statistic density per unit (bin x time) for an exponential bin with mean b(1 + r) tested against b
    (median exclusion, data = background): q = 2[ln(1 + r) - r/(1 + r)]  (-> r^2 for r -> 0, the weak-signal
    density behind :func:`scan_rate`).  Series 2(r^2/2 - 2r^3/3 + 3r^4/4) below r = 1e-3 (cancellation)."""
    r = np.asarray(r, dtype=float)
    if not exact:
        return r * r
    small = r < 1e-3
    rs = np.where(small, r, 0.0)
    rl = np.where(small, 1.0, r)
    return np.where(small, rs * rs - 4.0 * rs ** 3 / 3.0 + 1.5 * rs ** 4, 2.0 * (np.log1p(rl) - rl / (1.0 + rl)))


@lru_cache(maxsize=4)
def _shm_nodes(n_v: int, v0_kms: float, vobs_kms: float):
    """Gauss-Legendre nodes in v (units of c) over the SHM line and q(v) = f(v)/v, so that the normalised line is
    p_a(nu) = q(v)/nu_a at nu = nu_a(1 + v^2/2) and int dnu F(p_a) = int dv nu_a v F(q(v)/nu_a)."""
    v, w = _h._v_nodes(v0_kms, vobs_kms, n_v)
    q = _h._f_shm(v, v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT) / v
    return v, w, q


def _shm_qmax(v0_kms: float = _h.V0_KMS, vobs_kms: float = _h.VOBS_KMS) -> float:
    """max_v f(v)/v: the peak of nu_a p_a(nu) for the SHM line."""
    v = np.linspace(1e-7, (vobs_kms + 4 * v0_kms) * 1e3 / C_LIGHT, 20001)
    return float(np.max(_h._f_shm(v, v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT) / v))


def fisher_integral(c, A, D, width_Hz, f_a_Hz, *, exact: bool = True, n_x: int = 401, n_v: int = 300,
                    v0_kms: float = _h.V0_KMS, vobs_kms: float = _h.VOBS_KMS):
    """J = int d(nu - nu_c) int d(nu - nu_a) F(r) [Hz^2 = Hz/s] for a dense scan, with the per-bin ratio
    r = c L(x) p_a(y)/(A + D L(x)), L(x) = 1/(1 + 4x^2/Delta_c^2), c = tau P_ext/(h nu_1) [Hz] (P_ext on resonance,
    narrow line) and p_a the normalised SHM line.  The accumulated statistic at mass m_a is Theta = J/nu_dot, so the
    rate that reaches Theta_95 is J/Theta_95.  F = :func:`asimov_density`; with exact=False, J = c^2 (pi Delta_c/
    (4 sqrt(A)(A + D)^{3/2})) int p_a^2, the weak-signal rate (test).  Scalar inputs."""
    th, wt = np.polynomial.legendre.leggauss(n_x)
    th = 0.5 * math.pi * th * (1 - 1e-12)
    wt = 0.5 * math.pi * wt
    s = 0.5 * _sensitivity_width(width_Hz, A, D)
    x = s * np.tan(th)
    jac = wt * s / np.cos(th) ** 2
    L = 1.0 / (1.0 + 4.0 * x * x / width_Hz ** 2)
    gx = L / (A + D * L)                                   # (n_x,)
    v, wv, q = _shm_nodes(n_v, v0_kms, vobs_kms)
    r = c * gx[:, None] * (q[None, :] / f_a_Hz)            # (n_x, n_v)
    Fy = asimov_density(r, exact) @ (wv * f_a_Hz * v)      # int dy F, per x
    return float(np.sum(jac * Fy))


def _check_line_Q(target: Target, rtol: float = 0.01) -> None:
    q = _h._Q_a_shm_cached(_h.V0_KMS, _h.VOBS_KMS)
    if abs(target.Q_axion / q - 1.0) > rtol:
        raise ValueError(f"the exact-statistic path integrates over the package SHM line (Q_SHM = {q:.4g}); "
                         f"target.Q_axion = {target.Q_axion:.4g} differs by more than {rtol:.0%}.  Use the weak-signal "
                         f"closed form (required_volume) for other line widths, or keep Q_axion = Q_SHM.")


def _receiver_A(sc: "ScenarioA", V: float, f_a: float, target: Target, omega1: str = "shifted") -> dict:
    geo = sc.geometry
    f0 = float(geo.f0_Hz(V))
    f1 = f0 + f_a if omega1 == "shifted" else f0
    Qi = geo.G_signal_ohm / float(sc.Rs(f1))
    n_i = float(occupation(f1, sc.T_K))
    g = float(target.g_GeV(f_a))
    P0 = float(mode_power(g, f1, sc.B0_T, V, geo.eta, 1.0, target.rho_GeV_cm3))   # per unit Q_1
    return dict(f1=f1, Qi=Qi, n_i=n_i, P0=P0)


def _bin_quantities(sc: "ScenarioA", rx: dict, beta: float):
    Q_L = rx["Qi"] / (1.0 + beta)
    c = sc.tau * rx["P0"] * Q_L * beta / (1.0 + beta) / (H_PLANCK * rx["f1"])
    D = float(noise_resonant(beta, rx["n_i"], sc.tau, sc.n_e))
    width = rx["f1"] * (1.0 + beta) / rx["Qi"]
    return c, D, width


def peak_bin_snr(sc: "ScenarioA", V: float, f_a, *, beta=None, target: Target = DEFAULT_TARGET,
                 omega1: str = "shifted"):
    """Largest per-bin lambda_a/lambda_B over the scan at mass f_a: on resonance (L = 1) at the peak of the SHM line,
    r_max = c max(p_a)/(A + D).  beta None -> :func:`beta_opt` (the weak-signal optimum).  The weak-signal statistic
    of :func:`scan_rate` needs r << 1."""
    out = []
    qmax = _shm_qmax()
    for fa in np.atleast_1d(f_a):
        rx = _receiver_A(sc, V, float(fa), target, omega1)
        b = float(beta_opt(sc.A, rx["n_i"], sc.tau, sc.n_e)) if beta is None else beta
        c, D, _ = _bin_quantities(sc, rx, b)
        out.append(c * qmax / float(fa) / (sc.A + D))
    return np.array(out) if np.ndim(f_a) else out[0]


def rate_exact_A(sc: "ScenarioA", V: float, f_a: float, *, target: Target = DEFAULT_TARGET, exact: bool = True,
                 adiabatic_eps: float | None = None, omega1: str = "shifted", n_x: int = 401, n_v: int = 300) -> tuple:
    """(nu_dot [Hz/s], beta) at one mass with the Asimov statistic of :func:`asimov_density`, beta re-optimised:
    maximise J(beta)/Theta_95 (or min(J/Theta_95, tracking rate) with an adiabatic schedule).  For r >> 1 the exact
    density grows only like 2 ln r, so overcoupling (lower r, wider response) is favoured over the weak-signal beta*.

    The line is the package SHM (Q_SHM = 6.78e5), not ``target.Q_axion``; a target whose Q_axion differs from Q_SHM by
    more than 1% is rejected rather than silently ignored (the default 6.8e5 is 0.3% away)."""
    from scipy.optimize import minimize_scalar
    _check_line_Q(target)
    rx = _receiver_A(sc, V, f_a, target, omega1)

    def rate(lb):
        b = math.exp(lb)
        c, D, width = _bin_quantities(sc, rx, b)
        r = fisher_integral(c, sc.A, D, width, f_a, exact=exact, n_x=n_x, n_v=n_v) / target.Theta
        if adiabatic_eps is not None:
            r = min(r, float(tracking_rate(b, f_a, rx["f1"], rx["Qi"], adiabatic_eps, target.Q_axion)))
        return r
    b0 = float(beta_opt(sc.A, rx["n_i"], sc.tau, sc.n_e))
    lb0 = math.log(b0)
    res = minimize_scalar(lambda lb: -rate(lb), bounds=(lb0 - 3.0, lb0 + 5.0), method="bounded",
                          options={"xatol": 1e-4})
    return -res.fun, math.exp(res.x)


def required_volume_exact(sc: "ScenarioA", target: Target = DEFAULT_TARGET, *, exact: bool = True,
                          adiabatic_eps: float | None = None, n_f: int = 32, n_x: int = 401, n_v: int = 300,
                          V_guess: float | None = None, guard_Hz: float = 0.0) -> dict:
    """Required V with the Asimov statistic (exact or weak, the latter as a same-grid control) and beta re-optimised
    at every mass; mass integral by n_f-point Gauss-Legendre in ln f.  Heavier than :func:`required_volume`
    (~1 min per call; benchmarks/srf/model_a_geometry.py evaluates the benchmarks in parallel).  ``V_guess``
    (default: the weak-signal closed form) only seeds the root bracket, which is widened geometrically until it
    encloses the root.
    ``guard_Hz``: as :func:`required_volume`.  The band-end diagnostics are evaluated at f_min and f_max themselves."""
    from scipy.optimize import brentq
    _check_line_Q(target)
    f, w = _gl_log(target.f_min_Hz - guard_Hz, target.f_max_Hz + guard_Hz, n_f)

    def resid(lv):
        V = math.exp(lv)
        t = float(np.sum(w / np.array([rate_exact_A(sc, V, float(x), target=target, exact=exact,
                                                     adiabatic_eps=adiabatic_eps, n_x=n_x, n_v=n_v)[0] for x in f])))
        return math.log(t / target.live_time_s)
    lo, hi = _bracket_log(resid, math.log(V_guess or required_volume(sc, target, guard_Hz=guard_Hz)))
    V = math.exp(brentq(resid, lo, hi, xtol=1e-6))
    ends = (target.f_min_Hz, target.f_max_Hz)
    betas = [rate_exact_A(sc, V, x, target=target, exact=exact, adiabatic_eps=adiabatic_eps,
                          n_x=n_x, n_v=n_v)[1] for x in ends]
    return dict(V_m3=V, f0_Hz=float(sc.geometry.f0_Hz(V)), beta_ends=betas, exact=exact, adiabatic_eps=adiabatic_eps,
                n_f=n_f, n_x=n_x, n_v=n_v, guard_Hz=guard_Hz, f_ends_Hz=list(ends),
                peak_bin_snr_ends=[float(peak_bin_snr(sc, V, x, beta=b, target=target)) for x, b in zip(ends, betas)])


def _bracket_log(resid: Callable[[float], float], x0: float, step: float = math.log(1.6), max_iter: int = 40):
    """(lo, hi) around x0 with resid(lo) <= 0 <= resid(hi) for an increasing-V-decreasing-time residual
    (resid > 0: scan too slow, V too small), widening by ``step`` in ln V; raises if none within max_iter steps."""
    lo, hi = x0 - 0.5 * step, x0 + step
    r_lo, r_hi = resid(lo), resid(hi)
    for _ in range(max_iter):
        if r_lo >= 0.0 >= r_hi:
            return lo, hi
        if r_lo < 0.0:            # lower end already fast enough: move down
            hi, r_hi, lo = lo, r_lo, lo - step
            r_lo = resid(lo)
        else:                     # upper end still too slow: move up
            lo, r_lo, hi = hi, r_hi, hi + step
            r_hi = resid(hi)
    raise RuntimeError(f"required_volume_exact: no root within exp(+-{max_iter * step:.0f}) of V = {math.exp(x0):.4g} m^3")


def _rate_A(sc: ScenarioA, V: float, target: Target, omega1: str, adiabatic_eps: float | None = None,
            return_beta: bool = False):
    geo = sc.geometry
    f0 = float(geo.f0_Hz(V))

    def rate(f_a):
        f_a = np.asarray(f_a, dtype=float)
        f1 = f0 + f_a if omega1 == "shifted" else np.full_like(f_a, f0)
        Qi = geo.G_signal_ohm / sc.Rs(f1)
        n_i = occupation(f1, sc.T_K)
        b0 = beta_opt(sc.A, n_i, sc.tau, sc.n_e) + 0.0 * f_a
        stat = lambda b: scan_rate(f_a, f1, V, sc.B0_T, geo.eta, Qi, sc.A, n_i, beta=b, tau=sc.tau,  # noqa: E731
                                   n_e=sc.n_e, target=target)
        if adiabatic_eps is None:
            out = (stat(b0), b0)
        else:
            out = adiabatic_schedule(stat, lambda b: tracking_rate(b, f_a, f1, Qi, adiabatic_eps, target.Q_axion), b0)
        return out if return_beta else out[0]
    return rate


def required_volume(sc: ScenarioA, target: Target = DEFAULT_TARGET, *, omega1: str = "shifted",
                    n: int = 256, adiabatic_eps: float | None = None, guard_Hz: float = 0.0) -> float:
    """Smallest V [m^3] whose optimal scan covers the target band in the live time, with
    omega_0 = xi c V^(-1/3) and the coupling optimised at every mass.  omega1: "shifted"
    (omega_1 = omega_0 + m_a, the definition) or "pump" (omega_1 = omega_0).  adiabatic_eps: None
    (weak-signal optimum, no schedule constraint) or eps of :func:`tracking_rate` (e.g. 0.1).

    guard_Hz: the resonance sweep is extended by this much beyond both band ends.  The dense-scan rate assumes every
    target mass sees the whole detuning response and its whole (one-sided, upward) SHM line; without a guard the
    masses within ~ (Delta_c + nu_a v^2/2) of the ends are under-exposed (at 1 MHz an unguarded end collects only a few
    % of the statistic).  The default, 0, integrates over the band only; 20 Hz guards
    raise V by 1.3e-4 here because the rate at the ends is fast."""
    from scipy.optimize import brentq

    def resid(lnV):
        return math.log(scan_time(_rate_A(sc, math.exp(lnV), target, omega1, adiabatic_eps),
                                  target.f_min_Hz - guard_Hz, target.f_max_Hz + guard_Hz, n) / target.live_time_s)
    return math.exp(brentq(resid, math.log(1e-3), math.log(1e9), xtol=1e-12, rtol=1e-12))


def refrigerator_factor(T_K: float) -> float:
    """4.5 K-equivalent multiplier of a heat load at T: 1 at 4.5 K, ~3.3 at 2 K (European XFEL TDR)."""
    if abs(T_K - 4.5) < 1e-9:
        return 1.0
    if abs(T_K - 2.0) < 1e-9:
        return 3.3
    raise ValueError(f"refrigerator_factor: the 4.5 K-equivalent factor is tabulated only at 2 K (3.3) and "
                     f"4.5 K (1); got T = {T_K} K -- pass ScenarioA.refrigerator_factor explicitly")


@dataclass(frozen=True)
class CostLawA:
    """Parametric cost law of model A: C = c_Nb A_cav + a kappa A_cav^0.8 + C_ref (P_eq/P_ref)^0.63 [MCHF]
    (niobium area, cryostat, refrigeration plant).  A scaling metric for comparing designs, not a construction
    estimate; the default coefficients are illustrative (no source is claimed for them): set your own."""

    c_Nb_MCHF_m2: float = 0.07
    a_kappa_MCHF: float = 0.30
    plant_ref_MCHF: float = 60.0
    plant_ref_W: float = 18e3
    plant_exponent: float = 0.63
    P_static_W: float = 0.0       # the RF pump dissipation dominates the heat load


def cost_live(A_cav_m2: float, P_load_W: float, T_K: float, law: CostLawA = CostLawA(),
              factor: float | None = None) -> dict:
    """Cost of :class:`CostLawA` with P_eq = f(T)(P_RF + P_static)."""
    f = refrigerator_factor(T_K) if factor is None else factor
    P_eq = f * (P_load_W + law.P_static_W)
    parts = dict(niobium=law.c_Nb_MCHF_m2 * A_cav_m2, cryostat=law.a_kappa_MCHF * A_cav_m2 ** 0.8,
                 refrigerator=law.plant_ref_MCHF * (P_eq / law.plant_ref_W) ** law.plant_exponent)
    return dict(parts, total=sum(parts.values()), P_eq_W=P_eq, factor=f)


def design_A(sc: ScenarioA, target: Target = DEFAULT_TARGET, *, omega1: str = "shifted", law: CostLawA = CostLawA(),
             n: int = 256) -> dict:
    """Required volume and the design derived from it: frequency, Q, stored energy, RF load, wall area, cost and
    the scan diagnostics (coupling, bandwidth, per-bin SNR, adiabatic margin)."""
    geo = sc.geometry
    V = required_volume(sc, target, omega1=omega1, n=n)
    f0 = float(geo.f0_Hz(V))
    f_a = np.geomspace(target.f_min_Hz, target.f_max_Hz, 201)
    f1 = f0 + f_a if omega1 == "shifted" else np.full_like(f_a, f0)
    Rs1 = np.broadcast_to(np.asarray(sc.Rs(f1), dtype=float), f1.shape)
    Qi = geo.G_signal_ohm / Rs1
    Q0 = geo.G_pump_ohm / sc.Rs(f0)
    n_i = occupation(f1, sc.T_K)
    beta = beta_opt(sc.A, n_i, sc.tau, sc.n_e)
    bw = f1 * (1.0 + beta) / Qi
    U0 = sc.B0_T ** 2 * V / (2.0 * MU0)
    P_RF = 2.0 * math.pi * f0 * U0 / Q0
    A_cav = float(geo.area_m2(V))
    fac = sc.refrigerator_factor
    cost = cost_live(A_cav, P_RF, sc.T_K, law, fac)
    rates = _rate_A(sc, V, target, omega1)(f_a)
    # peak per-bin lambda_a/lambda_B (SHM line, on resonance, weak-signal beta*): the weak-signal statistic needs << 1
    snr_ends = peak_bin_snr(sc, V, np.array([target.f_min_Hz, target.f_max_Hz]), target=target, omega1=omega1)
    # rate / (0.1 pi Delta_c min(Delta_c, nu_a/Q_a)): > 1 means the weak-signal optimum violates the sufficient
    # adiabatic condition of tracking_rate (model A does not impose it)
    adiab = rates / tracking_rate(beta, f_a, f1, Qi, 0.1, target.Q_axion)
    return dict(scenario=sc.name, V_m3=V, f0_Hz=f0, Q_i=float(np.median(Qi)), Q_i_range=[float(Qi.min()), float(Qi.max())],
                Q0=float(Q0), Rs_signal_ohm=float(np.median(Rs1)), B0_T=sc.B0_T, Bpk_T=sc.Bpk_T, U0_J=U0, P_RF_W=P_RF,
                A_cav_m2=A_cav, T_K=sc.T_K, A=sc.A, n_i_range=[float(n_i.min()), float(n_i.max())],
                beta_range=[float(beta.min()), float(beta.max())], loaded_bw_Hz_range=[float(bw.min()), float(bw.max())],
                bw_approx_tauT_over_piAQi_Hz=float(sc.tau * KB * sc.T_K / (HBAR * math.pi * sc.A * np.median(Qi))),
                scan_rate_Hz_s_ends=[float(rates[0]), float(rates[-1])],
                adiabatic_ratio_ends=[float(adiab[0]), float(adiab[-1])],
                adiabatic_violated_above_Hz=_adiabatic_threshold(sc, V, target, omega1),
                peak_bin_snr_ends=[float(x) for x in snr_ends],
                tuning_fraction=(target.f_max_Hz - target.f_min_Hz) / f0,
                linear_size_m=V ** (1 / 3), cost_MCHF=cost, omega1=omega1)


def _adiabatic_threshold(sc: ScenarioA, V: float, target: Target, omega1: str):
    """Axion frequency above which the weak-signal optimum exceeds the adiabatic condition (eps = 0.1),
    by root-finding on ln(rate/tracking); None if never violated in the band (the ratio grows with f_a here)."""
    from scipy.optimize import brentq
    geo = sc.geometry
    f0 = float(geo.f0_Hz(V))

    def lratio(lf):
        fa = np.array([math.exp(lf)])
        f1 = f0 + fa if omega1 == "shifted" else np.array([f0])
        Qi = geo.G_signal_ohm / np.asarray(sc.Rs(f1), dtype=float)
        b = beta_opt(sc.A, occupation(f1, sc.T_K), sc.tau, sc.n_e)
        return float(np.log(_rate_A(sc, V, target, omega1)(fa) / tracking_rate(b, fa, f1, Qi, 0.1, target.Q_axion))[0])
    a, b = math.log(target.f_min_Hz), math.log(target.f_max_Hz)
    if lratio(b) <= 0:
        return None
    if lratio(a) > 0:
        return target.f_min_Hz
    return math.exp(brentq(lratio, a, b, xtol=1e-10))


def volume_scaling_exponents() -> dict:
    """V proportional to xi^(3/4) (A T/(tau Q_i))^(3/8) (eta B_0)^(-3/2) in the n_i >> 1 limit
    (rate ~ V^(8/3) at fixed xi)."""
    return dict(xi=0.75, A=0.375, T=0.375, tau=-0.375, Q_i=-0.375, eta=-1.5, B0=-1.5, rate_V=8.0 / 3.0)
